"""Legacy elevator RL feature and action composition contract."""
from __future__ import annotations

import numpy as np

from pressb.control.pose_protocol import finite_array, pose8_to_pose9
from pressb.tasks.elevator.semantics import task_floor

PROTOCOL_VERSION = 1
ACTION_HORIZON = 7
POSE_DIM = 9
CAMERA_ORDER = ("global", "wrist")


def learner_observation(encoded, observation):
    """No ground-truth goal/layout fields are admitted to the learning policy."""
    feature = finite_array(encoded["feature"], (2048,), "encoder feature", np.float32)
    state = finite_array(encoded["state_pose9"], (9,), "state_pose9", np.float32)
    expected = pose8_to_pose9(observation["state"])
    if not np.allclose(state, expected, atol=2e-5, rtol=0):
        raise ValueError("Inference state changed the simulator's measured TCP convention")
    control = finite_array(observation["control_state"], name="control_state", dtype=np.float32)
    if control.ndim != 1:
        raise ValueError("control_state must be a vector")
    onehot = np.zeros(12, dtype=np.float32)
    onehot[task_floor(observation["task"]) - 24] = 1.
    return np.concatenate((feature, state, control, onehot))


def residual_actions(base_actions, residual, scale, *, residual_mode="pose9"):
    """Compose a full pose9 chunk while retaining frozen rotations in XYZ mode."""
    if residual_mode not in ("pose9", "xyz"):
        raise ValueError("residual_mode must be pose9 or xyz")
    width = 3 if residual_mode == "xyz" else POSE_DIM
    base = finite_array(base_actions, (7, 9), "base action")
    action = finite_array(residual, name="residual").reshape(7, width)
    scales = finite_array(scale, (width,), "residual scale")
    if np.any(np.abs(action) > 1. + 1e-6) or np.any(scales < 0):
        raise ValueError("Residual must be bounded and scales nonnegative")
    if residual_mode == "xyz":
        result = base.copy()
        result[:, :3] += action * scales
        return result
    return base + action * scales


def initial_noise(action, noise_steps=1, scale=1.5):
    if type(noise_steps) is not int or noise_steps not in (1, 7):
        raise ValueError("noise_steps must divide the seven-step horizon (1 or 7)")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("noise scale must be positive")
    action = finite_array(action, name="noise action").reshape(noise_steps, 9)
    if np.any(np.abs(action) > 1. + 1e-6):
        raise ValueError("Normalized noise action must be in [-1,1]")
    return np.tile(action, (7 // noise_steps, 1)) * scale
