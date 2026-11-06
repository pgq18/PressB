"""Joint6 action isolation, exact data projection, metrics and checkpoint schema."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from starVLA.dataloader.table_buttons_lerobot import TableButtonsLeRobotDataset
from starVLA.dataloader.table_buttons_joint6_lerobot import (
    TableButtonsJoint6LeRobotDataset, NORMALIZATION, normalize_joint6, denormalize_joint6,
)
from starVLA.training.train_table_buttons import (
    INTERFACE_KEYS, SCHEMA, JOINT6_SCHEMA, select_joint_pretrained, joint_metric_sums,
    summarize_joint_metrics, action_schema, write_checkpoint_files, reconcile_best_checkpoint,
)


def fake_states():
    source = {"qwen_vl_interface.test": torch.ones(2),
        "action_model.action_encoder.layer1.weight": torch.ones(4, 7),
        "action_model.state_encoder.layer1.weight": torch.ones(4, 8),
        "action_model.action_decoder.layer2.weight": torch.ones(7, 4),
        "action_model.action_decoder.layer2.bias": torch.ones(7)}
    target = {key: value.clone() for key, value in source.items()}
    target.update({"action_model.action_encoder.layer1.weight": torch.zeros(4, 6),
                   "action_model.state_encoder.layer1.weight": torch.zeros(4, 7),
                   "action_model.action_decoder.layer2.weight": torch.zeros(6, 4),
                   "action_model.action_decoder.layer2.bias": torch.zeros(6)})
    return source, target


class Joint6Tests(unittest.TestCase):
    def test_official_interfaces_project_six_output_and_seven_state(self):
        source, target = fake_states()
        loaded, report = select_joint_pretrained(source, target, action_dim=6)
        self.assertEqual(set(loaded), set(source) - INTERFACE_KEYS)
        self.assertEqual(len(report["adapted_keys"]), 4)
        self.assertEqual(report["initialization"], "official_pretrained_with_new_joint6_interfaces")
        with self.assertRaises(ValueError):
            select_joint_pretrained(source, target, action_dim=7)
        target["qwen_vl_interface.test"] = torch.zeros(3)
        with self.assertRaises(ValueError):
            select_joint_pretrained(source, target, action_dim=6)

    def test_joint6_metrics_ignore_measured_gripper_and_mask_padding(self):
        target = np.zeros((3, 6))
        prediction = np.array([[.1] * 6, [.2] * 6, [999.] * 6])
        padding = np.array([False, False, True])
        state = np.array([[.05] * 6 + [10000.]])
        result = summarize_joint_metrics(joint_metric_sums(prediction, target, state, padding, action_dim=6), action_dim=6)
        self.assertEqual(result["valid_actions"], 2)
        self.assertAlmostEqual(result["joint_mae_deg"], 27.)
        self.assertAlmostEqual(result["first_joint_mae_deg"], 18.)
        self.assertAlmostEqual(result["state_copy_baseline_first_joint_mae_deg"], 9.)
        self.assertEqual(result["selection_score"], result["first_joint_mae_deg"])
        self.assertFalse(any("gripper" in key for key in result))
        with self.assertRaises(ValueError):
            joint_metric_sums(np.zeros((3, 7)), np.zeros((3, 7)), state, padding, action_dim=6)

    def test_joint6_normalization_has_no_seventh_coordinate(self):
        degrees = np.array([[0., 90., 180., -90., -180., 45.]], dtype=np.float32)
        normalized = normalize_joint6(degrees)
        self.assertTrue(np.array_equal(normalized, [[0., .5, 1., -.5, -1., .25]]))
        self.assertTrue(np.array_equal(denormalize_joint6(normalized), degrees))
        with self.assertRaises(ValueError):
            normalize_joint6(np.zeros((7, 7)))

    def test_source_gripper_cannot_be_silently_discarded_if_variable(self):
        def fake_parent(obj, *args, **kwargs):
            obj._actions = np.zeros((2, 7), dtype=np.float32)
        with patch.object(TableButtonsLeRobotDataset, "__init__", fake_parent):
            with self.assertRaises(ValueError):
                TableButtonsJoint6LeRobotDataset("unused")

    def test_new_schema_survives_best_last_and_reconcile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = dict(schema=JOINT6_SCHEMA, state_dim=7, action_dim=6, step=1, best_score=5.)
            write_checkpoint_files(root, {"arm": torch.ones(6)}, {"schema": JOINT6_SCHEMA}, metadata, True)
            for alias in ("best", "last"):
                self.assertEqual(json.loads((root / alias / "manifest.json").read_text())["schema"], JOINT6_SCHEMA)
            manifest = json.loads((root / "last/manifest.json").read_text())
            reconcile_best_checkpoint(root, (root / "last").resolve(), manifest)
            write_checkpoint_files(root, {"arm": torch.zeros(6)}, {}, dict(metadata, step=2), False)
            self.assertEqual(json.loads((root / "best/manifest.json").read_text())["schema"], JOINT6_SCHEMA)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_checkpoint_files(root, {}, {}, {"step": 1}, True)
            self.assertEqual(json.loads((root / "last/manifest.json").read_text())["schema"], SCHEMA)

    def test_schema_separates_source_state_action_and_fixed_gripper(self):
        dataset = SimpleNamespace(action_dim=6, normalization=NORMALIZATION)
        result = action_schema(dataset, dict(schema=JOINT6_SCHEMA, action_dim=6, state_dim=7, dataset_group="medium_red"))
        self.assertEqual(result["state_dim"], 7)
        self.assertEqual(result["action_dim"], 6)
        self.assertEqual(result["normalization"]["state"]["scale"], [180.] * 6 + [8.])
        self.assertEqual(result["normalization"]["action"]["scale"], [180.] * 6)
        self.assertEqual(result["fixed_gripper_full_width_m"], .00019999999552965166)

    def test_real_medium_red_projection_keeps_original_split_and_observed_state(self):
        root = Path("/home/pengguanqi/Datasets/piper_table_buttons_20261023/medium_red")
        datasets = []
        try:
            for split in ("train", "val"):
                seven = TableButtonsLeRobotDataset(root, split=split)
                six = TableButtonsJoint6LeRobotDataset(root, split=split)
                datasets.extend((seven, six))
                self.assertEqual(seven.split_manifest_sha256, six.split_manifest_sha256)
                self.assertEqual(seven.source_episode_ids, six.source_episode_ids)
                self.assertEqual(len(seven), len(six))
                self.assertNotEqual(seven.fingerprint_sha256, six.fingerprint_sha256)
                self.assertTrue(np.array_equal(seven._states, six._states))
                self.assertTrue(np.array_equal(seven._actions[:, :6], six._actions))
                self.assertGreater(np.ptp(six._states[:, 6]), 0)
                episode = six.source_episode_ids[0]
                for frame in (0, six.episode_lengths[episode] - 1):
                    a, b = seven.get_frame(episode, frame), six.get_frame(episode, frame)
                    self.assertEqual(b["action"].shape, (7, 6))
                    self.assertEqual(b["state"].shape, (1, 7))
                    self.assertTrue(np.array_equal(a["action"][:, :6], b["action"]))
                    self.assertTrue(np.array_equal(a["state"], b["state"]))
                    self.assertTrue(np.array_equal(a["video"], b["video"]))
                    self.assertTrue(np.array_equal(a["action_is_pad"], b["action_is_pad"]))
                    self.assertEqual(a["lang"], b["lang"])
                if split == "val":
                    self.assertEqual(six.source_episode_ids, (18, 21, 27, 33, 42, 50, 56, 59, 71, 92))
        finally:
            for dataset in datasets:
                dataset.close()


if __name__ == "__main__":
    unittest.main()
