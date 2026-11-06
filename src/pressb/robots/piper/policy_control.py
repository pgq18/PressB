"""Bounded PiPER pose control for learned-policy evaluation."""
from __future__ import annotations

import numpy as np

from .pose_control import ContinuousPoseIK


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
