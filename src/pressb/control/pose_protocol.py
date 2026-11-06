"""Task-independent pose8/pose9 conversion with explicit gripper width.

Pose8 uses absolute XYZ metres, a unit wxyz quaternion and gripper width.
Pose9 uses absolute XYZ metres and the first two rotation-matrix rows.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def finite_array(value, shape=None, name="array", dtype=np.float64):
    raw = np.asarray(value)
    if np.iscomplexobj(raw) or raw.dtype.kind in "bUSO":
        raise ValueError(f"{name} must contain real numeric values")
    result = np.asarray(raw, dtype=dtype)
    if (shape is not None and result.shape != tuple(shape)) or not np.isfinite(result).all():
        raise ValueError(f"Invalid {name}: expected finite values with shape {shape}, got {result.shape}")
    return result


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


def pose9_to_pose8(pose, *, gripper_width_m):
    """Project rotation rows and append the caller's explicit gripper width."""
    width = float(finite_array(gripper_width_m, (), name="gripper_width_m"))
    if width < 0:
        raise ValueError("gripper_width_m must be nonnegative")
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
    return np.concatenate((flat[:, :3], quaternion, np.full((len(flat), 1), width)), axis=-1).reshape(*pose.shape[:-1], 8)

