"""Layout grouping and six-joint, color-balanced supervision contracts."""
import copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import yaml

from starVLA.dataloader.piper_lerobot import _canonical_sha
from starVLA.dataloader.table_buttons_multitask_joint6_lerobot import _build_split, COLORS, NORMALIZATION
from starVLA.training.train_table_buttons_multitask_joint6 import (
    validate_anchor_balance, summarize_multitask_metrics, build_contract,
)


def entries():
    result = []
    for task_index, color in enumerate(COLORS):
        for sid in range(1000):
            # Different source episode indices can describe the same layout.
            layout = (sid + 17) % 1000 if color == "yellow" else sid
            positions = {c: [layout / 10000., index / 10.] for index, c in enumerate(COLORS)}
            result.append(dict(episode_index=task_index * 1000 + sid, source_episode_index=sid,
                target_color=color, task_index=task_index, spacing_profile="near", positions_xy_m=positions,
                seed=layout + (2000 if color == "yellow" else 0),
                layout_group_id=_canonical_sha(dict(spacing_profile="near", positions_xy_m=positions))))
    return result


def test_actual_layout_grouping_avoids_color_split_leakage_despite_source_id_changes():
    source = entries()
    split = _build_split(source, "source-digest", 42)
    assert split["counts"] == {"train": 2700, "val": 300}
    membership = {episode: part for part in ("train", "val") for episode in split[f"{part}_source_episode_ids"]}
    groups = {}
    for entry in source:
        groups.setdefault(entry["layout_group_id"], set()).add(membership[entry["episode_index"]])
    assert len(groups) == 1000 and all(len(parts) == 1 for parts in groups.values())
    for part, n in (("train", 900), ("val", 100)):
        ids = set(split[f"{part}_source_episode_ids"])
        assert {color: sum(e["episode_index"] in ids and e["target_color"] == color for e in source)
                for color in COLORS} == {color: n for color in COLORS}
    assert split == _build_split(source[::-1], "source-digest", 42)


def test_layout_digest_and_missing_color_rejected():
    source = entries()
    source[0]["positions_xy_m"]["red"][0] += .001
    with pytest.raises(ValueError, match="layout_group_id"):
        _build_split(source, "digest", 42)
    source = entries()
    source[0]["layout_group_id"] = source[1]["layout_group_id"]
    source[0]["positions_xy_m"] = source[1]["positions_xy_m"]
    with pytest.raises(ValueError, match="exactly one"):
        _build_split(source, "digest", 42)


def test_physical_evaluation_seed_is_rejected():
    source = entries()
    source[0]["seed"] = 10020
    with pytest.raises(ValueError, match="evaluation seeds"):
        _build_split(source, "digest", 42)


def test_metric_color_balance_excludes_gripper():
    totals = np.zeros((3, 12))
    for index, count in enumerate((1, 2, 4)):
        totals[index, :8] = [count, count * 3, 6 * count * 3 * (index + 1), 999,
                             6 * count * (index + 1), 999, 6 * count * 10, 999]
        totals[index, 8:] = [count * .5, count, count * .6, count]
    overall, per_color = summarize_multitask_metrics(totals)
    assert overall["selection_score"] == 2
    assert overall["first_joint_mae_deg"] == pytest.approx(17 / 7)
    assert not any("gripper" in name for row in (overall, *per_color.values()) for name in row)


def test_validation_has_all_1500_distinct_balanced_anchors():
    dataset = SimpleNamespace(_episodes={i: {"target_color": COLORS[i // 100]} for i in range(300)})
    anchors = [(i, frame) for i in range(300) for frame in range(5)]
    validate_anchor_balance(dataset, anchors)
    for bad in (anchors[:-1], anchors[:-1] + [anchors[0]]):
        with pytest.raises(ValueError):
            validate_anchor_balance(dataset, bad)


def test_real_configuration_contract_has_3000_episodes_and_model_only_retention():
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "tests/fixtures/near_multitask_joint6_3000.yaml").read_text())
    contract, digest = build_contract(cfg, NORMALIZATION, _build_split(entries(), "digest", 42))
    assert contract["source_episode_count"] == 3000
    assert contract["split_counts"] == {"train": 2700, "val": 300}
    assert contract["checkpoint_policy"] == "best_and_final_model_only"
    assert contract["validation_anchor_count"] == 1500
    assert contract["layout_group_key"] == "layout_group_id"
    assert len(digest) == 64


@pytest.mark.parametrize("profile", ("near", "medium", "far"))
def test_real_multitask_data(profile, tmp_path):
    """Full numeric projection plus decoded held-out samples for each instruction."""
    import json
    import os
    import pyarrow.parquet as pq
    from starVLA.dataloader.table_buttons_multitask_joint6_lerobot import TableButtonsMultitaskJoint6LeRobotDataset
    from starVLA.dataloader.table_buttons_lerobot import normalize_joint7
    from starVLA.training.train_piper import validation_anchors
    root = Path(os.environ.get("TABLE_BUTTONS_MULTITASK_3000_TEST_ROOT",
        "/home/pengguanqi/Datasets/piper_table_buttons_multitask_3000_20261027")) / f"{profile}_multitask"
    assert root.is_dir(), f"Complete dataset not yet published: {root}"
    common = dict(root=root, split_manifest=tmp_path / "split.json", seed=42,
                  resolution=224, frame_cache_size=32)
    train = TableButtonsMultitaskJoint6LeRobotDataset(split="train", **common)
    val = TableButtonsMultitaskJoint6LeRobotDataset(split="val", **common)
    try:
        assert len(train.source_episode_ids) == 2700 and len(val.source_episode_ids) == 300
        assert set(train.source_episode_ids).isdisjoint(val.source_episode_ids)
        assert set(train.split_manifest["layout_group_ids_by_split"]["train"]).isdisjoint(
                   train.split_manifest["layout_group_ids_by_split"]["val"])
        raw_actions, raw_states, indices = [], [], []
        for path in sorted((root / "data").rglob("*.parquet")):
            table = pq.read_table(path, columns=["index", "action", "observation.state"])
            indices.extend(table["index"].to_pylist())
            raw_actions.extend(table["action"].to_pylist())
            raw_states.extend(table["observation.state"].to_pylist())
        order = np.argsort(np.asarray(indices).reshape(-1))
        raw_actions = np.asarray(raw_actions, dtype=np.float32)[order]
        raw_states = np.asarray(raw_states, dtype=np.float32)[order]
        assert np.array_equal(train._actions, normalize_joint7(raw_actions)[:, :6])
        assert np.array_equal(val._actions, train._actions)
        assert np.array_equal(train._states, normalize_joint7(raw_states))
        assert np.array_equal(val._states, train._states)
        assert np.ptp(train._states[:, 6]) > 0
        assert train.normalization["action"]["scale"] == [180.] * 6
        assert train.normalization["state"]["scale"] == [180.] * 6 + [8.]
        anchors = validation_anchors(val, 5, 0)
        validate_anchor_balance(val, anchors)
        for color in COLORS:
            sid = next(i for i in val.source_episode_ids if val._episodes[i]["target_color"] == color)
            for frame in (0, val.episode_lengths[sid] - 1):
                sample = val.get_frame(sid, frame)
                assert sample["lang"] == f"Press the {color} button."
                assert sample["action"].shape == (7, 6) and sample["state"].shape == (1, 7)
                assert sample["video"].shape == (2, 8, 224, 224, 3)
                assert sample["video"].dtype == np.uint8
                if frame:
                    assert sample["action_is_pad"].tolist() == [False] + [True] * 6
                    assert np.array_equal(sample["action"], np.repeat(sample["action"][:1], 7, axis=0))
    finally:
        train.close()
        val.close()
