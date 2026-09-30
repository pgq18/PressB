"""Transport and bounded pose control for learned-policy simulator evaluation.

This module has no Isaac imports and never reads demonstrations or button goals.
The raw policy targets stay unchanged; URDF-limited controller projection is
reported separately from those targets.
"""
from __future__ import annotations

import base64
from io import BytesIO
import json
import time
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from .replay_control import ContinuousPoseIK, action_pose


class InvalidPolicyAction(ValueError):
    """The model returned an invalid, nonfinite, or implausible arm command."""


def validate_policy_reply(reply, horizon=7):
    try:
        pose8 = np.asarray(reply["actions_pose8"], dtype=np.float64)
        pose9 = np.asarray(reply["actions_pose9"], dtype=np.float64)
        if pose8.shape != (horizon, 8) or pose9.shape != (horizon, 9):
            raise ValueError("Expected seven pose8 and pose9 actions")
        if not np.isfinite(pose8).all() or not np.isfinite(pose9).all():
            raise ValueError("Nonfinite model output")
        if not np.allclose(pose8[:, 7], .008, atol=1e-8, rtol=0):
            raise ValueError("Controller gripper width must stay fixed at 0.008 m")
        if not np.allclose(pose8[:, :3], pose9[:, :3], atol=1e-7, rtol=0):
            raise ValueError("pose8 conversion changed model XYZ")
        if np.any(np.linalg.norm(pose8[:, :3], axis=1) > 1.5):
            raise ValueError("Policy target farther than 1.5 m from PiPER base")
        for action in pose8:
            action_pose(action)
        for key, expected in (("pose_frame", "base_link"), ("pose_link", "gripper_tcp"),
                              ("gripper_is_learned", False)):
            if key in reply and reply[key] != expected:
                raise ValueError(f"Unexpected {key}: {reply[key]}")
        # The model uses the first two rotation-matrix rows. Check the received
        # conversion independently so a convention mismatch cannot move the arm.
        for action, six in zip(pose8, pose9[:, 3:]):
            first = six[:3] / np.linalg.norm(six[:3])
            second = six[3:] - np.dot(first, six[3:]) * first
            second /= np.linalg.norm(second)
            matrix = np.stack((first, second, np.cross(first, second)))
            if not np.isfinite(matrix).all() or not np.allclose(
                    matrix, action_pose(action)[1], atol=2e-5, rtol=0):
                raise ValueError("pose8 quaternion differs from model rotation6D rows")
    except (ValueError, TypeError, KeyError) as exc:
        raise InvalidPolicyAction(str(exc)) from exc
    return pose8.copy(), pose9.copy()


class PolicyClient:
    def __init__(self, endpoint, timeout=120.):
        self.endpoint, self.timeout = endpoint, float(timeout)

    def predict(self, task, state, images, seed):
        encoded = {}
        for name in ("global", "wrist"):
            rgb = np.asarray(images[name])
            if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
                raise ValueError(f"Invalid live {name} RGB: {rgb.shape}/{rgb.dtype}")
            stream = BytesIO()
            Image.fromarray(rgb).save(stream, format="PNG")
            encoded[name] = base64.b64encode(stream.getvalue()).decode("ascii")
        payload = dict(task=task, state=np.asarray(state).tolist(), images=encoded, seed=int(seed))
        request = Request(self.endpoint, data=json.dumps(payload, allow_nan=False).encode(),
                          headers={"Content-Type": "application/json"}, method="POST")
        started = time.monotonic()
        with urlopen(request, timeout=self.timeout) as response:
            reply = json.load(response)
        return reply, time.monotonic() - started


class PolicyPoseController:
    """Single-seed least-squares IK with 30 Hz URDF velocity projection.

    The implementation reuses replay IK, but permits imperfect pose fits from a
    learned policy. One diagnostic solve uses only joint-position bounds; the
    commanded solve additionally bounds every joint displacement by v_max/fps.
    Both start on the preceding commanded branch and use no alternate seeds.
    Neither solver receives task labels or button positions.
    """
    def __init__(self, urdf, initial_q, fps=30):
        self.bounded = ContinuousPoseIK(urdf, initial_q, fps=fps,
                                       position_tolerance_m=10., rotation_tolerance_rad=4.)
        self.unrestricted = ContinuousPoseIK(urdf, initial_q, fps=fps,
                                            position_tolerance_m=10., rotation_tolerance_rad=4.)
        self.unrestricted.maximum_joint_step = (
            self.unrestricted.kinematics.upper - self.unrestricted.kinematics.lower)

    @property
    def q(self):
        return self.bounded.q.copy()

    def solve(self, action):
        previous = self.bounded.q.copy()
        self.unrestricted.q = previous.copy()
        self.unrestricted._previous_pose = None
        unrestricted_q = self.unrestricted.solve(action)
        # A repeated raw target can still be far from the preceding projected
        # command. Replay's exact-target cache must not freeze that motion.
        self.bounded._previous_pose = None
        result = self.bounded.solve(action)
        return result, {
            "unrestricted_joint_target": unrestricted_q.tolist(),
            "unrestricted_position_residual_m": self.unrestricted.last_position_error_m,
            "unrestricted_rotation_residual_rad": self.unrestricted.last_rotation_error_rad,
            "command_position_residual_m": self.bounded.last_position_error_m,
            "command_rotation_residual_rad": self.bounded.last_rotation_error_rad,
            "velocity_saturated_joints": np.flatnonzero(np.abs(result - previous) >=
                self.bounded.maximum_joint_step - 1e-7).tolist(),
            "position_limited_joints": np.flatnonzero(
                (result <= self.bounded.kinematics.lower + 1e-7) |
                (result >= self.bounded.kinematics.upper - 1e-7)).tolist(),
            "solver_evaluations": self.bounded.last_nfev,
        }
