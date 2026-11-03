"""Checks for merge metadata/normalization errors that can silently alter training."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "merge_table_buttons", Path(__file__).resolve().parents[1] / "scripts/merge_table_buttons_datasets.py")
merge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(merge)


def test_recomputed_global_indices_and_constant_gripper():
    values = np.arange(300, dtype=np.int64)
    stats = merge.numeric_stats(values)
    assert stats["min"] == [0.0]
    assert stats["max"] == [299.0]
    assert stats["mean"] == [149.5]
    assert stats["q50"] == [149.5]
    assert stats["count"] == [300]
    constant = merge.numeric_stats(np.full((35000, 7), 0.02, dtype=np.float32))
    assert constant["std"] == [0.0] * 7


@pytest.mark.parametrize("values", [[], [1, float("nan")], [float("inf")]])
def test_invalid_numeric_statistics_rejected(values):
    with pytest.raises(ValueError):
        merge.numeric_stats(values)


@pytest.mark.parametrize("episodes_per_color", [100, 1000])
def test_manifest_task_mapping_keeps_shared_layout_provenance(episodes_per_color):
    sources = []
    for color in merge.COLORS:
        sources.append({"group": f"near_{color}", "instruction": f"Press the {color} button.",
                        "semantics": {"units": ["degree"] * 6 + ["cm"]},
                        "episodes": [{"episode_index": i, "seed": i, "target_color": color,
                                      "instruction": f"Press the {color} button.",
                                      "positions_xy_m": {"red": [.3 + i * .00001, .1]}}
                                     for i in range(episodes_per_color)]})
    manifest = merge.merged_manifest([Path(s["group"]) for s in sources], "near", sources, [{}, {}, {}])
    assert manifest["tasks"] == list(merge.TASKS)
    assert [e["episode_index"] for e in manifest["episodes"]] == list(range(3 * episodes_per_color))
    assert [s["episode_offset"] for s in manifest["sources"]] == [0, episodes_per_color, 2 * episodes_per_color]
    assert manifest["split_guidance"]["unique_layouts"] == episodes_per_color
    assert manifest["split_guidance"]["group_key"] == "layout_group_id"
    for task_index in range(3):
        e = manifest["episodes"][task_index * episodes_per_color + 17]
        assert e["source_episode_index"] == e["seed"] == 17
        assert e["task_index"] == task_index
        assert e["source_group"] == f"near_{merge.COLORS[task_index]}"
        assert e["group"] == "near_multitask"
        assert e["layout_group_id"] == manifest["episodes"][17]["layout_group_id"]
    assert sources[2]["episodes"][17]["episode_index"] == 17


def test_sources_and_manifest_preserve_unpaired_retry_layouts(tmp_path):
    roots = []
    for color in merge.COLORS:
        root = tmp_path / f"near_{color}"
        roots.append(root)
        instruction = f"Press the {color} button."
        episodes = [{"episode_index": i, "seed": i, "success": True,
                     "target_color": color, "instruction": instruction,
                     "positions_xy_m": {"red": [.3 + i * .01, .1], "yellow": [.36, .1],
                                        "blue": [.33, .16]}} for i in range(2)]
        if color == "yellow":
            episodes[0]["seed"] = 10000
            episodes[0]["positions_xy_m"]["red"] = [.4, .1]
        manifest = {"group": root.name, "spacing_profile": "near", "target_color": color,
                    "instruction": instruction, "fps": 30, "format_version": "v3.0",
                    "lerobot_version": "0.6.1", "semantics": {"units": ["degree"] * 6 + ["cm"]},
                    "episodes": episodes}
        merge.write_json(root / "meta/export_manifest.json", manifest)
        merge.write_json(root / "meta/info.json", {"fps": 30, "features": {}, "robot_type": "piper",
                                                  "codebase_version": "v3.0", "total_episodes": 2,
                                                  "total_tasks": 1})
    sources = merge.source_manifests(roots, "near", expected_episodes_per_color=2)
    manifest = merge.merged_manifest(roots, "near", sources, [{}, {}, {}])
    assert manifest["episodes"][2]["source_episode_index"] == 0
    assert manifest["episodes"][2]["seed"] == 10000
    assert manifest["episodes"][2]["positions_xy_m"]["red"] == [.4, .1]
    assert manifest["episodes"][2]["layout_group_id"] != manifest["episodes"][0]["layout_group_id"]
    assert manifest["episodes"][4]["layout_group_id"] == manifest["episodes"][0]["layout_group_id"]
    assert manifest["split_guidance"]["unique_layouts"] == 3
    with pytest.raises(ValueError, match="must contain 1000 episodes"):
        merge.source_manifests(roots, "near", expected_episodes_per_color=1000)


def test_source_snapshot_detects_modification_and_new_files(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "sample").write_bytes(b"immutable")
    before = merge.snapshot(source)
    assert before == merge.snapshot(source)
    (source / "sample").write_bytes(b"different")
    assert before != merge.snapshot(source)
    (source / "sample").write_bytes(b"immutable")
    (source / "extra").write_bytes(b"new")
    assert before != merge.snapshot(source)
