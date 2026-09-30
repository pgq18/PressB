"""Negative controls for the independent first-success dataset verifier."""
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("av", reason="PyAV dataset audit tests run in the separate LeRobot environment",
                    exc_type=ModuleNotFoundError)
pytest.importorskip("cv2", reason="OpenCV is required for the dataset image feedback audit",
                    exc_type=ModuleNotFoundError)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_press_dataset import CAMERAS, audit_stats, compare_rgb, source_prefix, verify_light_prefix, verify_numeric_prefix


def fixture():
    collection = {"raw_schema_version": 10, "fps": 30, "physics_hz": 120,
                  "capture_stride": 4, "action_horizon_s": 1 / 30,
                  "config": {"press_threshold": .0015, "release_threshold": .0008,
                             "home_q": [0] * 6, "home_tolerance_rad": .025}}
    q = np.repeat(np.arange(13, dtype=float)[:, None] / 1000, 6, axis=1)
    physics = {"q_actual": q.copy(), "q_command": q.copy(),
               "button_travel": np.zeros((13, 12)), "contact_force": np.zeros((13, 12)),
               "lights": np.zeros((13, 12), dtype=np.uint8)}
    physics["button_travel"][7:10, 0] = .002
    physics["contact_force"][7:10, 0] = .1
    physics["lights"][7:10, 0] = 1
    frames = {"state": np.arange(32, dtype=float).reshape(4, 8),
              "action": np.arange(32, dtype=float).reshape(4, 8) + 100,
              "q_actual": q[[0, 4, 8, 12]], "q_target": q[[4, 8, 12, 12]],
              "physics_index": np.array([0, 4, 8, 12]), "sim_time": np.arange(4) / 30,
              "phase": np.array(["settle", "approach", "press", "retract"]),
              "lights": physics["lights"][[0, 4, 8, 12]]}
    metadata = {"floor": 24, "episode_id": 0, "seed": 12, "physics_steps": 13,
                "num_frames": 4, "unexpected_collisions": [],
                "events": [{"type": "pressed", "floor": 24, "physics_index": 7},
                           {"type": "released", "floor": 24, "physics_index": 10}]}
    return frames, physics, metadata, collection


def values(expected, n=None):
    n = n or len(expected["state"])
    result = {key: expected[source].astype(np.float32).copy()
              for key, source in (("observation.state", "state"), ("action", "action"),
                                  ("observation.joint_position", "q_actual"), ("action.joint_target", "q_target"))}
    result.update({"observation.sim_time": expected["sim_time"].copy(), "timestamp": np.arange(n) / 30,
                   "frame_index": np.arange(n), "index": np.arange(n), "episode_index": np.zeros(n),
                   "task_index": np.zeros(n), "source_episode_id": np.zeros(n),
                   "source_seed": np.full(n, 12), "floor": np.full(n, 24)})
    return result


def test_first_success_keeps_exactly_one_lit_endpoint_and_clamps_plan():
    frames, physics, metadata, collection = fixture()
    prefix, result = source_prefix(frames, physics, metadata, collection)
    assert result["cut_frame_index"] == 2 and result["first_press_physics_index"] == 7
    assert len(prefix["state"]) == 3 and prefix["lights"].sum() == 1
    assert np.array_equal(prefix["state"], frames["state"][:3])
    assert np.array_equal(prefix["action"][-1], frames["action"][1])
    assert np.array_equal(prefix["q_target"][-1], physics["q_command"][8])
    verify_numeric_prefix(values(prefix), prefix, 0, 0, metadata)


@pytest.mark.parametrize("delta", [-1, 1])
def test_dropping_success_or_retaining_extra_frame_is_rejected(delta):
    frames, physics, meta, collection = fixture()
    prefix, _ = source_prefix(frames, physics, meta, collection)
    bad = values(prefix)
    bad["observation.state"] = frames["state"][:3 + delta].astype(np.float32)
    with pytest.raises(ValueError, match="prefix"):
        verify_numeric_prefix(bad, prefix, 0, 0, meta)


@pytest.mark.parametrize("key,source", [("action", "action"), ("action.joint_target", "q_target")])
def test_unclamped_terminal_target_is_rejected(key, source):
    frames, physics, meta, collection = fixture()
    prefix, _ = source_prefix(frames, physics, meta, collection)
    bad = values(prefix)
    bad[key][-1] = frames[source][2]
    with pytest.raises(ValueError, match="terminal clamp"):
        verify_numeric_prefix(bad, prefix, 0, 0, meta)


def test_lamp_without_real_force_is_rejected():
    frames, physics, meta, collection = fixture()
    physics["contact_force"][:] = .02
    with pytest.raises(ValueError, match="physical contact"):
        source_prefix(frames, physics, meta, collection)


def test_return_phase_cannot_be_retained_even_with_lit_label():
    frames, physics, meta, collection = fixture()
    frames["phase"][2] = "retract"
    with pytest.raises(ValueError, match="not pressing"):
        source_prefix(frames, physics, meta, collection)


def test_wrong_camera_image_rejected():
    rgb = np.full((480, 640, 3), 90, dtype=np.uint8)
    compare_rgb(rgb, rgb.copy())
    with pytest.raises(ValueError, match="RGB differs"):
        compare_rgb(rgb, np.zeros_like(rgb))


@pytest.mark.parametrize("counts", [[0, 0, 0], [0, 900, 900], [0, 900, 0]])
def test_missing_early_or_extra_lit_frame_rejected(counts):
    with pytest.raises(ValueError):
        verify_light_prefix(counts, "wrist")


def test_occluded_global_allowed_but_wrist_required():
    assert not verify_light_prefix([0, 0, 1], "global")["sufficient_visibility"]
    assert verify_light_prefix([0, 0, 900], "wrist")["visible_first_lit_frame"] == 2


def test_aggregate_statistics_must_follow_remapped_indices():
    x = np.arange(12)[:, None]
    scalar = {"min": [0.0], "max": [11.0], "mean": [5.5], "std": [float(x.std())], "count": [12],
              **{q: [5.5] for q in ["q01", "q10", "q50", "q90", "q99"]}}
    stats = {"episode_index": scalar}
    moments = {}
    for camera in CAMERAS:
        stats[camera] = {key: np.zeros((3, 1, 1)).tolist() for key in ["min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99"]}
        stats[camera]["count"] = [12 * 120 * 160]
        moments[camera] = {"sum": np.zeros(3), "sum2": np.zeros(3), "count": 12 * 120 * 160}
    audit_stats(stats, {"episode_index": [x]}, 12, moments)
    stats["episode_index"]["max"] = [3.0]
    with pytest.raises(ValueError, match="episode_index/max"):
        audit_stats(stats, {"episode_index": [x]}, 12, moments)
