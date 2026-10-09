"""Versioned, dependency-light wire primitives shared by the three RL nodes."""
from __future__ import annotations

import base64
from io import BytesIO
import re

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

PROTOCOL_VERSION = 1
ACTION_HORIZON = 7
POSE_DIM = 9
CAMERA_ORDER = ("global", "wrist")
TASK_PATTERN = re.compile(r"Press (2[4-9]|3[0-5]) floor\.")


def finite_array(value, shape=None, name="array", dtype=np.float64):
    raw = np.asarray(value)
    if np.iscomplexobj(raw) or raw.dtype.kind in "bUSO":
        raise ValueError(f"{name} must contain real numeric values")
    result = np.asarray(raw, dtype=dtype)
    if (shape is not None and result.shape != tuple(shape)) or not np.isfinite(result).all():
        raise ValueError(f"Invalid {name}: expected finite values with shape {shape}, got {result.shape}")
    return result


def encode_image(rgb):
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.shape != (480, 640, 3):
        raise ValueError("RGB must be uint8 [480,640,3]")
    buffer = BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_image(value):
    if not isinstance(value, str) or len(value) > 12_000_000:
        raise ValueError("Expected a bounded base64 PNG image")
    data = base64.b64decode(value, validate=True)
    with Image.open(BytesIO(data)) as picture:
        if picture.format != "PNG" or picture.size != (640, 480) or picture.mode != "RGB":
            raise ValueError("Expected a lossless RGB PNG of size 640x480")
        return np.array(picture, dtype=np.uint8, copy=True)


def task_floor(task):
    match = TASK_PATTERN.fullmatch(task) if isinstance(task, str) else None
    if match is None:
        raise ValueError("Task must be exactly 'Press 24 floor.' through 'Press 35 floor.'")
    return int(match.group(1))


def pose8_to_pose9(pose):
    pose = finite_array(pose, name="pose8")
    if pose.ndim < 1 or pose.shape[-1] != 8:
        raise ValueError("Expected pose8 with final dimension 8")
    flat = pose.reshape(-1, 8)
    norms = np.linalg.norm(flat[:, 3:7], axis=-1)
    if np.any(np.abs(norms - 1.) > 1e-5):
        raise ValueError("pose8 quaternion must be unit wxyz")
    rotation = Rotation.from_quat(flat[:, [4, 5, 6, 3]]).as_matrix()
    return np.concatenate((flat[:, :3], rotation[:, :2, :].reshape(-1, 6)), axis=-1).reshape(*pose.shape[:-1], 9)


def pose9_to_pose8(pose):
    """Project rotation rows only; keep model/residual XYZ in absolute metres."""
    pose = finite_array(pose, name="pose9")
    if pose.ndim < 1 or pose.shape[-1] != 9:
        raise ValueError("Expected pose9 with final dimension 9")
    flat = pose.reshape(-1, 9)
    rows = flat[:, 3:].reshape(-1, 2, 3)
    first = rows[:, 0]
    n1 = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(n1 < 1e-8):
        raise ValueError("Degenerate first rotation6D row")
    first = first / n1
    second = rows[:, 1] - (rows[:, 1] * first).sum(-1, keepdims=True) * first
    n2 = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(n2 < 1e-8):
        raise ValueError("Degenerate second rotation6D row")
    second = second / n2
    rotation = np.stack((first, second, np.cross(first, second)), axis=1)
    quaternion = Rotation.from_matrix(rotation).as_quat()[:, [3, 0, 1, 2]]
    quaternion *= np.where(quaternion[:, :1] < 0, -1., 1.)
    return np.concatenate((flat[:, :3], quaternion, np.full((len(flat), 1), .008)), axis=-1).reshape(*pose.shape[:-1], 8)


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
