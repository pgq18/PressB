"""Capture rates must survive conversion without mixing cadence or horizons."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from export_lerobot import (
    LEROBOT_VERSION, MANIFEST, SEMANTICS, TASKS, collection_timing, existing_parts,
    load_arrays, manifest_for, semantics_for_collection, validate_episode_collection,
    write_dataset_readme,
)
from test_dataset_provenance import make_collection


def timing(fps=30):
    return {"raw_schema_version": 10, "fps": fps, "physics_hz": 120,
            "capture_stride": 120 // fps, "action_horizon_s": 1 / fps}


@pytest.mark.parametrize("fps", [10, 20, 30])
def test_collection_manifest_and_card_use_capture_rate(tmp_path, fps):
    raw, dataset, collection, _ = make_collection(tmp_path, schema=10, fps=fps)
    semantics = semantics_for_collection(collection)
    assert semantics["fps"] == fps
    assert semantics["action_horizon_s"] == 1 / fps
    assert semantics["action_joint_target"] == "next_sample_endpoint_clamped_at_episode_end"
    manifest = manifest_for([{"episode_id": 0, "floor": 24, "frames": fps + 1}], "part", raw)
    assert manifest["semantics"] == semantics
    write_dataset_readme(dataset, manifest, {"success": True, "full_video_decode": True,
                                             "total_episodes": 1, "total_frames": fps + 1})
    card = (dataset / "README.md").read_text()
    assert f"at {fps} Hz" in card
    assert f"one sample ({1 / fps:.9g} s) ahead" in card


def test_dynamic_semantics_do_not_change_legacy_resume_contract():
    original = copy.deepcopy(SEMANTICS)
    for schema in (7, 9):
        assert semantics_for_collection({"raw_schema_version": schema, "fps": 10}) == original
    semantics_for_collection(timing(30))
    assert SEMANTICS == original
    assert semantics_for_collection({"raw_schema_version": 9, "fps": 10}) == original
    assert original["action_joint_target"] == "next_100ms_endpoint_clamped_at_episode_end"


@pytest.mark.parametrize("fps", [0, -10, 25, 29.97, True, "30"])
def test_invalid_capture_rates_fail(fps):
    with pytest.raises(ValueError, match="positive integer divisor"):
        collection_timing({**timing(), "fps": fps})


@pytest.mark.parametrize("patch", [
    {"physics_hz": 60}, {"capture_stride": 12}, {"action_horizon_s": .1},
    {"action_horizon_s": float("nan")}, {"capture_stride": 4.0},
])
def test_inconsistent_timing_fails(patch):
    with pytest.raises(ValueError, match="incompatible"):
        collection_timing({**timing(), **patch})


@pytest.mark.parametrize("key", ["physics_hz", "capture_stride", "action_horizon_s"])
def test_schema10_cannot_omit_timing_identity(key):
    collection = timing()
    del collection[key]
    with pytest.raises(ValueError, match=f"missing {key}"):
        collection_timing(collection)


def test_legacy_capture_cannot_be_relabelled_30hz():
    with pytest.raises(ValueError, match="Legacy collection schemas require 10 fps"):
        collection_timing({"raw_schema_version": 9, "fps": 30})


def test_episodes_cannot_mix_fps_or_action_horizons(tmp_path):
    _, _, collection, _ = make_collection(tmp_path, schema=10, fps=30)
    episode = {**timing(), "collection_fingerprint": collection["collection_fingerprint"],
               "env_offset_m": [0., 0., 0.], "robot_base_world_m": [-.24, 0., .76]}
    validate_episode_collection(episode, collection, tmp_path)
    with pytest.raises(ValueError, match="Episode fps does not match"):
        validate_episode_collection({**episode, **timing(20)}, collection, tmp_path)
    with pytest.raises(ValueError, match="action_horizon_s is incompatible"):
        validate_episode_collection({**episode, "action_horizon_s": .1}, collection, tmp_path)


@pytest.mark.parametrize("fps", [10, 20, 30])
def test_numeric_cadence_requires_the_same_collection_rate(tmp_path, fps):
    count = 8
    pose = np.zeros((count, 8))
    pose[:, 3], pose[:, 7] = 1., .008
    np.savez(tmp_path / "frames.npz", state=pose, action=pose,
             q_actual=np.zeros((count, 6)), q_target=np.zeros((count, 6)),
             sim_time=np.arange(count) / fps)
    assert len(load_arrays(tmp_path, fps)["state"]) == count
    with pytest.raises(ValueError, match="timestamps are not consecutive"):
        load_arrays(tmp_path, 30 if fps == 10 else 10)


def test_resume_rejects_shard_from_another_rate(tmp_path):
    raw, _, _, _ = make_collection(tmp_path, schema=10, fps=30)
    parts = tmp_path / "parts"
    shard = parts / "part_00000"
    (shard / "meta").mkdir(parents=True)
    manifest = {"lerobot_version": LEROBOT_VERSION, "tasks": TASKS, "raw_root": str(raw),
                "semantics": semantics_for_collection(timing(20)), "episodes": []}
    (shard / MANIFEST).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Incompatible committed shard"):
        existing_parts(parts, raw)
