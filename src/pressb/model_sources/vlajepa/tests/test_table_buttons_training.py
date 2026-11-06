"""Joint-semantic initialization, unit metrics and checkpoint retention tests."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from starVLA.training.train_table_buttons import (
    INTERFACE_KEYS, initialize_interfaces, joint_metric_sums,
    select_joint7_pretrained, summarize_joint_metrics, write_checkpoint_files, reconcile_best_checkpoint,
)


def fake_states():
    target = {"qwen_vl_interface.test": torch.ones(2, 2), "vj_encoder.test": torch.ones(2),
              "vj_predictor.test": torch.ones(2),
              "action_model.action_encoder.layer1.weight": torch.ones(4, 7),
              "action_model.state_encoder.layer1.weight": torch.ones(4, 7),
              "action_model.action_decoder.layer2.weight": torch.ones(7, 4),
              "action_model.action_decoder.layer2.bias": torch.ones(7)}
    source = dict(target)
    source["action_model.state_encoder.layer1.weight"] = torch.ones(4, 8)
    return source, target


class TrainingTests(unittest.TestCase):
    def test_same_shape_action_interface_is_never_loaded(self):
        source, target = fake_states()
        selected, report = select_joint7_pretrained(source, target)
        self.assertEqual(set(selected), set(source) - INTERFACE_KEYS)
        self.assertEqual({x["key"] for x in report["adapted_keys"]}, INTERFACE_KEYS)
        self.assertEqual(report["loaded_keys"], 3)

    def test_other_missing_or_mismatched_keys_fail(self):
        source, target = fake_states()
        source["vj_encoder.test"] = torch.ones(3)
        with self.assertRaises(ValueError):
            select_joint7_pretrained(source, target)
        source, target = fake_states()
        del source["qwen_vl_interface.test"]
        with self.assertRaises(ValueError):
            select_joint7_pretrained(source, target)

    def test_only_known_official_state8_accepted(self):
        source, target = fake_states()
        source["action_model.state_encoder.layer1.weight"] = torch.ones(4, 7)
        with self.assertRaises(ValueError):
            select_joint7_pretrained(source, target)

    def test_reset_only_four_parameters_and_reproducible(self):
        model = torch.nn.Module()
        model.action_model = torch.nn.Module()
        for branch, layer, input_size, output_size in (("action_encoder", "layer1", 7, 4),
                                                    ("state_encoder", "layer1", 7, 4),
                                                    ("action_decoder", "layer2", 4, 7)):
            module = torch.nn.Module()
            setattr(module, layer, torch.nn.Linear(input_size, output_size))
            setattr(model.action_model, branch, module)
        old = {name: value.detach().clone() for name, value in model.named_parameters()}
        torch.manual_seed(42)
        initialize_interfaces(model)
        first = {name: value.detach().clone() for name, value in model.named_parameters()}
        for name in first:
            self.assertEqual(torch.equal(old[name], first[name]), name not in INTERFACE_KEYS)
        torch.manual_seed(42)
        initialize_interfaces(model)
        for name, value in model.named_parameters():
            self.assertTrue(torch.equal(first[name], value))

    def test_denormalized_metrics_padding_and_state_copy(self):
        truth = np.zeros((3, 7))
        prediction = np.array([[.1] * 6 + [.25], [.2] * 6 + [.5], [999.] * 7])
        state = np.array([[.05] * 6 + [.125]])
        totals = joint_metric_sums(prediction, truth, state, np.array([False, False, True]))
        result = summarize_joint_metrics(totals)
        self.assertEqual(result["valid_actions"], 2)
        self.assertAlmostEqual(result["first_joint_mae_deg"], 18.)
        self.assertAlmostEqual(result["joint_mae_deg"], 27.)
        self.assertAlmostEqual(result["gripper_mae_cm"], 3.)
        self.assertAlmostEqual(result["first_gripper_mae_cm"], 2.)
        self.assertAlmostEqual(result["state_copy_baseline_first_joint_mae_deg"], 9.)
        self.assertAlmostEqual(result["state_copy_baseline_first_gripper_mae_cm"], 1.)
        self.assertAlmostEqual(result["selection_score"], 63.)

    def test_nonfinite_or_first_padding_rejected(self):
        for prediction, mask in ((np.full((2, 7), np.nan), np.array([False, False])),
                                 (np.zeros((2, 7)), np.array([True, False]))):
            with self.assertRaises(ValueError):
                joint_metric_sums(prediction, np.zeros((2, 7)), np.zeros(7), mask)

    def test_checkpoint_last_and_best_retention_and_hardlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, training = {"weight": torch.ones(2)}, {"step": 1, "optimizer": {}}
            write_checkpoint_files(root, model, training, {"step": 1}, True)
            self.assertFalse((root / "best/training.pt").exists())
            self.assertEqual((root / "best/model.pt").stat().st_ino, (root / "last/model.pt").stat().st_ino)
            write_checkpoint_files(root, {"weight": torch.zeros(2)}, {"step": 2}, {"step": 2}, False)
            self.assertEqual((root / "best").resolve().name, "best_step_000001")
            self.assertEqual((root / "last").resolve().name, "last_step_000002")
            self.assertFalse((root / "last_step_000001").exists())
            self.assertTrue(torch.equal(torch.load(root / "best/model.pt", weights_only=True)["weight"], model["weight"]))
            write_checkpoint_files(root, model, {"step": 3}, {"step": 3}, True)
            self.assertEqual(sorted(p.name for p in root.iterdir()), ["best", "best_step_000003", "last", "last_step_000003"])
            manifest = json.loads((root / "last/manifest.json").read_text())
            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["training_bytes"], (root / "last/training.pt").stat().st_size)

    def test_failed_staging_never_replaces_valid_alias(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_checkpoint_files(root, {}, {}, {"step": 1}, True)
            # Publication of an existing generation fails before replacing aliases.
            with self.assertRaises(FileExistsError):
                write_checkpoint_files(root, {"w": torch.ones(2)}, {}, {"step": 1}, False)
            self.assertEqual(torch.load(root / "last/model.pt", weights_only=True), {})

    def test_resume_repairs_interrupted_best_alias_publication(self):
        from starVLA.training import train_table_buttons as trainer
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_checkpoint_files(root, {"w": torch.zeros(2)}, {}, {"step": 1, "best_score": 10.}, True)
            original = trainer._replace_alias
            def fail_best(directory, alias, destination):
                if alias == "best":
                    raise RuntimeError("Simulated interruption between last and best publication")
                original(directory, alias, destination)
            with patch.object(trainer, "_replace_alias", side_effect=fail_best):
                with self.assertRaises(RuntimeError):
                    write_checkpoint_files(root, {"w": torch.ones(2)}, {}, {"step": 2, "best_score": 5.}, True)
            self.assertEqual((root / "best").resolve().name, "best_step_000001")
            self.assertEqual((root / "last").resolve().name, "last_step_000002")
            manifest = json.loads((root / "last/manifest.json").read_text())
            reconcile_best_checkpoint(root, (root / "last").resolve(), manifest)
            self.assertEqual((root / "best").resolve().name, "best_step_000002")
            self.assertEqual((root / "best/model.pt").stat().st_ino, (root / "last/model.pt").stat().st_ino)


if __name__ == "__main__":
    unittest.main()
