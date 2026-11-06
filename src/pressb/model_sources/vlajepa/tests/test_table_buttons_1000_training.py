"""CPU regression for complete 1000-episode sets and unchanged legacy splits."""
import copy
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from test_table_buttons_lerobot import buttons, synthetic_root, fake_decode

joint6 = importlib.import_module(f"{buttons.__package__}.table_buttons_joint6_lerobot")


def test_original_100_split_serialization_is_unchanged():
    entries = [dict(episode_index=index) for index in range(100)]
    shuffled = np.random.default_rng(42).permutation(list(range(100))).tolist()
    legacy = dict(schema_version=1, kind="table_buttons_episode_split", seed=42,
        source_export_manifest_sha256="legacy_manifest", camera_keys=list(buttons.CAMERA_KEYS),
        state_action_names=list(buttons.JOINT7_NAMES), normalization=copy.deepcopy(buttons.NORMALIZATION),
        train_source_episode_ids=sorted(shuffled[10:]), val_source_episode_ids=sorted(shuffled[:10]),
        counts={"train": 90, "val": 10},
        selection="one shared seeded permutation of episode IDs; no frame-level split")
    actual = buttons._build_split(entries, "legacy_manifest", 42)
    assert json.dumps(actual, indent=2).encode() == json.dumps(legacy, indent=2).encode()


def test_expanded_split_has_900_training_and_100_validation_episode_ids():
    entries = [dict(episode_index=index) for index in range(1000)]
    splits = [buttons._build_split(entries[::-1], group, 42) for group in ("near_red", "medium_red", "far_red")]
    expected = np.random.default_rng(42).permutation(1000)
    for split in splits:
        assert split["counts"] == {"train": 900, "val": 100}
        assert split["train_source_episode_ids"] == sorted(expected[100:].tolist())
        assert split["val_source_episode_ids"] == sorted(expected[:100].tolist())
        assert not set(split["train_source_episode_ids"]) & set(split["val_source_episode_ids"])
        assert set(split["train_source_episode_ids"]) | set(split["val_source_episode_ids"]) == set(range(1000))


@pytest.mark.parametrize("ids", [list(range(999)), list(range(1001)), list(range(999)) + [998], list(range(1,1001))])
def test_incomplete_or_nonconsecutive_expansion_rejected(ids):
    with pytest.raises(ValueError, match="100 or 1000"):
        buttons._build_split([dict(episode_index=i) for i in ids], "manifest", 42)


@pytest.mark.parametrize("synthetic_root", [1000], indirect=True)
def test_expanded_joint6_reader_retains_measured_state7_and_episode_boundaries(synthetic_root, tmp_path):
    path = tmp_path / "run/split.json"
    train = joint6.TableButtonsJoint6LeRobotDataset(synthetic_root, split_manifest=path)
    val = joint6.TableButtonsJoint6LeRobotDataset(synthetic_root, split="val", split_manifest=path)
    try:
        assert train.source_episode_count == val.source_episode_count == 1000
        assert len(train.source_episode_ids) == 900 and len(val.source_episode_ids) == 100
        for source_id in (0, 100, 500, 999):
            dataset = train if source_id in train.source_episode_ids else val
            dataset._video_frames = fake_decode
            final = dataset.episode_lengths[source_id] - 1
            sample = dataset.get_frame(source_id, final)
            assert sample["state"].shape == (1, 7) and sample["action"].shape == (7, 6)
            assert sample["action_is_pad"].tolist() == [False] + [True] * 6
            assert sample["video_is_pad"].tolist() == [False] + [True] * 7
            np.testing.assert_array_equal(sample["action"], np.repeat(sample["action"][:1], 7, axis=0))
            assert sample["state"][0, 1] != sample["action"][0, 0]  # no state/action substitution
        assert train.split_manifest == val.split_manifest
    finally:
        train.close(); val.close()


@pytest.mark.parametrize("synthetic_root", [1000], indirect=True)
def test_reserved_physical_evaluation_seed_rejected(synthetic_root):
    path = synthetic_root / "meta/export_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["episodes"][-1]["seed"] = 10000
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Held-out evaluation seeds"):
        buttons.TableButtonsLeRobotDataset(synthetic_root)


def test_training_contract_and_validation_geometry_keep_old_behavior():
    from starVLA.training import train_table_buttons as trainer
    from omegaconf import OmegaConf
    repository = Path(__file__).resolve().parents[1]
    config = OmegaConf.to_container(OmegaConf.load(repository / "tests/fixtures/medium_red_joint6.yaml"), resolve=True)
    legacy = trainer.build_contract(config, joint6.NORMALIZATION)
    old = trainer.build_contract(config, joint6.NORMALIZATION, {"counts": {"train": 90, "val": 10}})
    assert legacy == old
    assert "source_episode_count" not in old[0]
    new, _ = trainer.build_contract(config, joint6.NORMALIZATION, {"counts": {"train": 900, "val": 100}})
    assert new["source_episode_count"] == 1000 and new["validation_anchor_count"] == 500
    assert new["validation_selection_weight"] == 0
    for train_anchors in (104404, 104406):
        geometry = trainer.shared.epoch_geometry(train_anchors, 1, 64, 1, 3)
        assert geometry["target_steps"] == 4896
        assert round(.05 * geometry["target_steps"]) == 245
    dataset = SimpleNamespace(split_manifest={"counts": {"train": 900, "val": 100}})
    assert trainer.validate_anchor_count(dataset, [None] * 500) == 500
    with pytest.raises(ValueError, match="500"):
        trainer.validate_anchor_count(dataset, [None] * 50)


@pytest.mark.parametrize("group", ["near_red", "medium_red", "far_red"])
def test_real_expanded_red_data(group):
    root = Path(os.environ.get("TABLE_BUTTONS_1000_TEST_ROOT", "/home/pengguanqi/Datasets/piper_table_buttons_1000_20261025")) / group
    if not (root / "meta/export_manifest.json").exists():
        pytest.skip("Expanded datasets must be transferred for real CPU reader checks")
    train = joint6.TableButtonsJoint6LeRobotDataset(root, frame_cache_size=16, decoder_cache_size=2)
    val = joint6.TableButtonsJoint6LeRobotDataset(root, split="val", split_manifest=train.split_manifest,
                                               frame_cache_size=16, decoder_cache_size=2)
    try:
        assert train.source_episode_count == 1000
        assert (len(train.source_episode_ids), len(val.source_episode_ids)) == (900, 100)
        assert len(train) == (104406 if group == "far_red" else 104404)
        assert len(val) == 11600
        assert len(train._all_episodes) == 1000
        assert {entry["entry"]["seed"] for entry in train._all_episodes} == set(range(1000))
        for source_id in (0, 100, 500, 999):
            dataset = train if source_id in train.source_episode_ids else val
            for frame in (0, dataset.episode_lengths[source_id] - 1):
                sample = dataset.get_frame(source_id, frame)
                assert sample["state"].shape == (1, 7) and sample["action"].shape == (7, 6)
                assert sample["video"].shape == (2, 8, 224, 224, 3)
                assert sample["lang"] == "Press the red button."
    finally:
        train.close(); val.close()
