"""CPU checks of model-only retention, peak payloads and legacy opt-in behavior."""
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

from starVLA.training import train_table_buttons as trainer


def metadata(step):
    return dict(step=step, epoch=step // 10, batch_in_epoch=0, best_score=1 / step,
                schema=trainer.JOINT6_SCHEMA, training_fingerprint="contract", dataset_sha256="dataset")


def payloads(root):
    """Count bytes by inode: same-step final/best are hardlinks, not copies."""
    unique = {}
    for file in root.glob("*_step_*/model.pt"):
        stat = file.stat()
        unique[(stat.st_dev, stat.st_ino)] = stat.st_size
    return unique


@pytest.mark.parametrize("final_improves", [False, True])
def test_retention_and_peak_never_exceed_two_model_payloads(tmp_path, final_improves):
    state = {"weight": torch.arange(128, dtype=torch.float32)}
    peaks = []
    original_save, original_link, original_replace = torch.save, os.link, os.replace

    def observe(operation):
        def wrapped(*args, **kwargs):
            result = operation(*args, **kwargs)
            peaks.append(payloads(tmp_path))
            return result
        return wrapped

    with patch.object(torch, "save", observe(original_save)), patch.object(os, "link", observe(original_link)), \
         patch.object(os, "replace", observe(original_replace)):
        for step in (1, 2, 3):
            trainer.write_model_only_checkpoint_files(tmp_path, state, metadata(step), is_best=True, final=False)
            assert len(payloads(tmp_path)) == 1
            assert not (tmp_path / "last").exists()
            assert sorted(p.name for p in tmp_path.iterdir()) == ["best", f"best_step_{step:06d}"]
        before = {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*")}
        trainer.write_model_only_checkpoint_files(tmp_path, state, metadata(4), is_best=False, final=False)
        assert before == {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*")}
        trainer.write_model_only_checkpoint_files(tmp_path, state, metadata(5), is_best=final_improves, final=True)

    assert max(len(p) for p in peaks) == 2
    model_bytes = (tmp_path / "best/model.pt").stat().st_size
    assert max(sum(p.values()) for p in peaks) <= 2 * model_bytes
    assert len(payloads(tmp_path)) == (1 if final_improves else 2)
    assert not list(tmp_path.rglob("training.pt"))
    assert not list(tmp_path.glob("*.tmp*"))
    for alias in ("best", "last"):
        saved = json.loads((tmp_path / alias / "manifest.json").read_text())
        assert saved["kind"] == "model_only_" + alias
        assert saved["checkpoint_policy"] == trainer.MODEL_ONLY_CHECKPOINT_POLICY
        assert saved["optimizer_retained"] is saved["exact_resume_supported"] is False
        assert saved["complete"] and saved["training_bytes"] == 0
        assert saved["schema"] == trainer.JOINT6_SCHEMA and saved["dataset_sha256"] == "dataset"
        assert torch.equal(torch.load(tmp_path / alias / "model.pt", weights_only=True)["weight"], state["weight"])
    assert json.loads((tmp_path / "last/manifest.json").read_text())["step"] == 5
    if final_improves:
        assert os.path.samefile(tmp_path / "best/model.pt", tmp_path / "last/model.pt")
    with pytest.raises(ValueError, match="already finalized"):
        trainer.write_model_only_checkpoint_files(tmp_path, state, metadata(6), is_best=True, final=False)


@pytest.mark.parametrize("failure", ["serialization", "alias"])
def test_failed_replacement_keeps_old_best_and_removes_partial_payload(tmp_path, failure):
    state = {"weight": torch.ones(32)}
    trainer.write_model_only_checkpoint_files(tmp_path, state, metadata(1), is_best=True, final=False)
    original = (tmp_path / "best").resolve()
    if failure == "serialization":
        def fail_save(state, path):
            Path(path).write_bytes(b"partial model")
            raise OSError("interrupted serialization")
        context = patch.object(torch, "save", fail_save)
    else:
        replace = os.replace
        def fail_alias(src, dst):
            if Path(dst).name == "best":
                raise OSError("interrupted alias replacement")
            return replace(src, dst)
        context = patch.object(os, "replace", fail_alias)
    with context, pytest.raises(OSError, match="interrupted"):
        trainer.write_model_only_checkpoint_files(tmp_path, state, metadata(2), is_best=True, final=False)
    assert (tmp_path / "best").resolve() == original
    assert sorted(p.name for p in tmp_path.iterdir()) == ["best", "best_step_000001"]
    assert len(payloads(tmp_path)) == 1


def test_model_only_does_not_read_optimizer_scheduler_or_rng(tmp_path):
    model = torch.nn.Linear(2, 2)
    forbidden = Mock()
    forbidden.state_dict.side_effect = AssertionError("optimizer/scheduler must not be serialized")
    with patch.object(trainer.shared, "get_rng", side_effect=AssertionError("RNG must not be collected")), \
         patch.object(trainer.dist, "all_gather_object", side_effect=AssertionError("No RNG gathering")), \
         patch.object(trainer.dist, "barrier"):
        trainer.save_checkpoint(model, forbidden, forbidden, tmp_path, metadata(1), True, 0, 1, {},
                                checkpoint_policy=trainer.MODEL_ONLY_CHECKPOINT_POLICY, final=True)
    assert not list(tmp_path.rglob("training.pt"))


def test_nonbest_nonfinal_does_not_even_read_model_state(tmp_path):
    model = Mock()
    model.state_dict.side_effect = AssertionError("No model checkpoint on ordinary validation")
    trainer.save_checkpoint(model, None, None, tmp_path, metadata(1), False, 0, 1, {},
                            checkpoint_policy=trainer.MODEL_ONLY_CHECKPOINT_POLICY, final=False)
    assert not (tmp_path / "checkpoints").exists()


def test_policy_is_opt_in_and_resume_is_rejected_before_cuda(tmp_path):
    assert trainer.configured_checkpoint_policy({"trainer": {}}) == trainer.RESUMABLE_CHECKPOINT_POLICY
    assert trainer.configured_checkpoint_policy({"trainer": {}}, "old-last") == trainer.RESUMABLE_CHECKPOINT_POLICY
    with pytest.raises(ValueError, match="Unknown checkpoint"):
        trainer.configured_checkpoint_policy({"trainer": {"checkpoint_policy": "typo"}})
    config = tmp_path / "config.yaml"
    config.write_text("trainer:\n  checkpoint_policy: best_and_final_model_only\n")
    with patch("sys.argv", ["train", "--config", str(config), "--resume", "old-last"]), \
         patch.object(trainer, "setup_distributed", side_effect=AssertionError("Must reject before CUDA")), \
         pytest.raises(ValueError, match="cannot resume"):
        trainer.main()


def test_legacy_default_still_serializes_training_state(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer, scheduler = Mock(), Mock()
    optimizer.state_dict.return_value = {"optimizer": "kept"}
    scheduler.state_dict.return_value = {"scheduler": "kept"}
    progress = {key: metadata(1)[key] for key in ("step", "epoch", "batch_in_epoch", "best_score")}
    with patch.object(trainer.shared, "get_rng", return_value={"rng": "kept"}), \
         patch.object(trainer.dist, "all_gather_object") as gather, patch.object(trainer.dist, "barrier"):
        trainer.save_checkpoint(model, optimizer, scheduler, tmp_path, progress, True, 0, 1, {})
    optimizer.state_dict.assert_called_once()
    scheduler.state_dict.assert_called_once()
    gather.assert_called_once()
    saved = json.loads((tmp_path / "checkpoints/last/manifest.json").read_text())
    assert saved["kind"] == "resumable_last" and saved["training_bytes"] > 0
    assert "checkpoint_policy" not in saved
    training = torch.load(tmp_path / "checkpoints/last/training.pt", weights_only=False)
    assert training["optimizer"] == {"optimizer": "kept"}


def test_nondefault_policy_is_part_of_contract_but_default_is_unchanged():
    base = {"trainer": {}, "framework": {"action_model": {"action_dim": 6}}}
    with patch.object(trainer.shared, "training_contract", return_value=({"old_contract": 1}, "ignored")):
        old_contract, old_sha = trainer.build_contract(base, {"scale": 180})
    explicit_default = dict(base, trainer={"checkpoint_policy": trainer.RESUMABLE_CHECKPOINT_POLICY})
    with patch.object(trainer.shared, "training_contract", return_value=({"old_contract": 1}, "ignored")):
        default_contract, default_sha = trainer.build_contract(explicit_default, {"scale": 180})
    model_only = dict(base, trainer={"checkpoint_policy": trainer.MODEL_ONLY_CHECKPOINT_POLICY})
    with patch.object(trainer.shared, "training_contract", return_value=({"old_contract": 1}, "ignored")):
        new_contract, new_sha = trainer.build_contract(model_only, {"scale": 180})
    assert old_contract == default_contract and old_sha == default_sha
    assert "checkpoint_policy" not in old_contract
    assert new_contract["checkpoint_policy"] == trainer.MODEL_ONLY_CHECKPOINT_POLICY
    assert new_sha != old_sha
