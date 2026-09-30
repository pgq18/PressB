"""Reject retimed labels without resampling physical observations and commands."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "collector_timing_under_test", Path(__file__).resolve().parents[1] / "scripts/collect_dataset.py")
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


@pytest.mark.parametrize("fps,stride", [(10, 12), (20, 6), (30, 4), (60, 2)])
def test_capture_rate_has_exact_physics_boundaries(fps, stride):
    assert collector.capture_stride_for(fps) == stride


@pytest.mark.parametrize("fps", [0, -1, 29, 121, 29.97, True])
def test_invalid_rates_are_rejected(fps):
    with pytest.raises(ValueError):
        collector.capture_stride_for(fps)


def test_30hz_arrays_require_four_step_actions_and_real_30hz_times(tmp_path):
    steps, stride = 17, 4
    indices = np.arange(0, steps, stride)
    q = np.arange(steps * 6, dtype=float).reshape(steps, 6) * .0001
    physics = dict(q_actual=q, q_command=q + .001,
                   gripper_actual=np.tile([.004, -.004], (steps, 1)),
                   button_travel=np.zeros((steps, 12)), contact_force=np.zeros((steps, 12)),
                   lights=np.zeros((steps, 12), dtype=np.uint8))
    pose = np.tile([0., 0., 0., 1., 0., 0., 0., .008], (len(indices), 1))
    frames = dict(state=pose, action=pose, sim_time=indices / 120,
                  q_actual=q[indices], q_target=physics["q_command"][np.minimum(indices + stride, steps - 1)],
                  physics_index=indices, phase=np.full(len(indices), "approach"),
                  lights=physics["lights"][indices])
    metadata = dict(raw_schema_version=10, fps=30, physics_hz=120, capture_stride=4,
                    action_horizon_s=1/30, num_frames=len(indices), physics_steps=steps,
                    plan_physics_steps=steps)
    np.savez(tmp_path / "physics.npz", **physics)
    np.savez(tmp_path / "frames.npz", **frames)
    collector.validate_source_arrays(tmp_path, metadata)

    old_targets = frames["q_target"].copy()
    frames["q_target"] = physics["q_command"][np.minimum(indices + 12, steps - 1)]
    np.savez(tmp_path / "frames.npz", **frames)
    with pytest.raises(ValueError, match="alignment mismatch"):
        collector.validate_source_arrays(tmp_path, metadata)

    frames["q_target"] = old_targets
    frames["sim_time"] = np.arange(len(indices)) / 10
    np.savez(tmp_path / "frames.npz", **frames)
    with pytest.raises(ValueError, match="alignment mismatch"):
        collector.validate_source_arrays(tmp_path, metadata)
