"""Continuous, bounded 6-DoF IK for recorded base-frame gripper TCP actions.

Only the 8D actions and one initial joint seed enter the controller. Recorded
joint targets, joint histories, and collection motion plans are not inputs.
The gripper TCP is link6 local [0, 0, .1358] m, not the .24 m pressing tip.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from pressb.kinematics import PiperKinematics


GRIPPER_TCP_OFFSET_M = .1358


class ReplayIKError(ValueError):
    """An action cannot be reached accurately on the continuous bounded branch."""


def action_pose(action):
    """Validate [x,y,z,qw,qx,qy,qz,total_width] and return position, rotation, width.

    Float32 quaternion rounding is normalized; materially non-unit quaternions
    are rejected rather than silently interpreting malformed action data.
    """
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (8,) or not np.isfinite(action).all():
        raise ValueError("action must contain eight finite base-frame TCP values")
    norm = float(np.linalg.norm(action[3:7]))
    if abs(norm - 1.) > 1e-5:
        raise ValueError("action quaternion must be unit wxyz (within 1e-5 rounding tolerance)")
    quaternion = action[3:7] / norm
    rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
    return action[:3].copy(), rotation, float(action[7])


def _pose_errors(transform, position, rotation):
    return (float(np.linalg.norm(transform[:3, 3] - position)),
            float(Rotation.from_matrix(rotation @ transform[:3, :3].T).magnitude()))


class ContinuousPoseIK:
    """Full-pose IK seeded exclusively by the preceding successful solution.

    Per-sample bounds are the intersection of URDF joint limits and the
    previous solution +/- URDF velocity/fps. There are no alternate/global
    restarts that could silently switch elbow or wrist branches.
    """

    def __init__(self, urdf_path: str | Path, initial_q, *, fps=30,
                 position_tolerance_m=1e-5, rotation_tolerance_rad=1e-4):
        if type(fps) is not int or fps <= 0 or 120 % fps:
            raise ValueError("fps must be a positive integer divisor of 120")
        if (not np.isfinite(position_tolerance_m) or position_tolerance_m <= 0
                or not np.isfinite(rotation_tolerance_rad) or rotation_tolerance_rad <= 0):
            raise ValueError("IK error tolerances must be finite and positive")
        self.kinematics = PiperKinematics(urdf_path, tip_offset=GRIPPER_TCP_OFFSET_M)
        seed = np.asarray(initial_q, dtype=np.float64)
        if seed.shape != (6,) or not np.isfinite(seed).all():
            raise ValueError("initial_q must contain six finite joint angles")
        kin = self.kinematics
        # PhysX can report tiny joint-limit penetration at the initial sample.
        # Permit only this documented sub-milliradian seed correction.
        if np.any(seed < kin.lower - 1e-3) or np.any(seed > kin.upper + 1e-3):
            raise ValueError("initial_q is outside URDF joint limits by more than 1e-3 rad")
        self.q = np.clip(seed, kin.lower, kin.upper)
        self.initial_seed_clipped = bool(np.any(self.q != seed))
        self.fps = fps
        self.maximum_joint_step = kin.velocity_limits / fps
        self.position_tolerance_m = float(position_tolerance_m)
        self.rotation_tolerance_rad = float(rotation_tolerance_rad)
        self.last_position_error_m = 0.
        self.last_rotation_error_rad = 0.
        self.last_nfev = 0
        self._previous_pose = None

        root = ET.parse(urdf_path).getroot()
        fingers = [root.find(f"joint[@name='joint{number}']/limit") for number in (7, 8)]
        if any(limit is None for limit in fingers):
            raise ValueError("PiPER URDF must contain gripper joint7/joint8 limits")
        lower = np.array([float(limit.get("lower")) for limit in fingers])
        upper = np.array([float(limit.get("upper")) for limit in fingers])
        self.minimum_width = float(max(0., 2 * lower[0], -2 * upper[1]))
        self.maximum_width = float(min(2 * upper[0], -2 * lower[1]))
        self.maximum_width_speed = 2 * min(float(limit.get("velocity")) for limit in fingers)

    def validate_width(self, width):
        if (not np.isfinite(width) or width < self.minimum_width - 1e-10
                or width > self.maximum_width + 1e-10):
            raise ValueError("total gripper width is outside symmetric URDF finger limits")
        return float(np.clip(width, self.minimum_width, self.maximum_width))

    def solve(self, action):
        position, rotation, width = action_pose(action)
        self.validate_width(width)
        kin, seed = self.kinematics, self.q.copy()
        previous = self._previous_pose
        repeated = (previous is not None and np.array_equal(position, previous[0])
                    and np.allclose(rotation, previous[1], atol=1e-14, rtol=0))
        if repeated:
            result_q, nfev = seed, 0
        else:
            lower = np.maximum(kin.lower, seed - self.maximum_joint_step)
            upper = np.minimum(kin.upper, seed + self.maximum_joint_step)

            def residual(q):
                actual = kin.fk(q)
                angle = Rotation.from_matrix(rotation @ actual[:3, :3].T).as_rotvec()
                return np.r_[actual[:3, 3] - position, .15 * angle, 1e-6 * (q - seed)]

            result = least_squares(residual, seed, bounds=(lower, upper),
                                   ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=150)
            result_q, nfev = result.x, result.nfev
        distance, angle = _pose_errors(kin.fk(result_q), position, rotation)
        if (distance > self.position_tolerance_m or angle > self.rotation_tolerance_rad
                or np.any(result_q < kin.lower) or np.any(result_q > kin.upper)
                or np.any(np.abs(result_q - seed) > self.maximum_joint_step + 1e-10)):
            raise ReplayIKError(f"Continuous IK failed: position={distance:.8g} m, "
                                f"rotation={angle:.8g} rad; no alternate seed was used")
        self.q = result_q.copy()
        self._previous_pose = (position, rotation)
        self.last_position_error_m, self.last_rotation_error_rad = distance, angle
        self.last_nfev = int(nfev)
        return self.q.copy()


@dataclass(frozen=True)
class ReplayTrajectory:
    """120 Hz commands including initial row0, plus action-space IK diagnostics."""
    joint_positions: np.ndarray
    gripper_widths: np.ndarray
    times_s: np.ndarray
    action_index_by_step: np.ndarray
    ik_solutions: np.ndarray
    position_errors_m: np.ndarray
    rotation_errors_rad: np.ndarray
    solver_evaluations: np.ndarray
    fps: int
    physics_hz: int
    capture_stride: int
    initial_seed_clipped: bool


def solve_action_trajectory(urdf_path, actions, initial_q, *, fps=30, physics_hz=120,
                            initial_gripper_width=.008):
    """Solve 8D actions, then interpolate each command interval at physics rate.

    For N recorded observations there are N-1 intervals. Row0 is the initial
    seed; rows1..stride interpolate toward IK(action0), rowstride reaches it.
    Next, action1 reaches row2*stride. The terminal-clamped final action is
    solved/checked but does not introduce an extra interval. Thus commands
    have (N-1)*stride+1 rows, at relative times 0 .. (N-1)/fps. If reproducing
    the collector's absolute physics clock, add 1/physics_hz to these times.

    Linear interpolation is in joint space; exact TCP target accuracy applies
    at sample endpoints, not to a Cartesian straight line between endpoints.
    """
    if type(physics_hz) is not int or physics_hz != 120:
        raise ValueError("This scene's replay physics rate must be 120 Hz")
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 8 or len(actions) < 2:
        raise ValueError("actions must be an N x 8 array with at least two samples")
    solver = ContinuousPoseIK(urdf_path, initial_q, fps=fps)
    initial_q = solver.q.copy()
    initial_width = solver.validate_width(initial_gripper_width)
    poses = [action_pose(action) for action in actions]
    if (np.linalg.norm(poses[-1][0] - poses[-2][0]) > 1e-6
            or Rotation.from_matrix(poses[-1][1] @ poses[-2][1].T).magnitude() > 1e-5
            or abs(poses[-1][2] - poses[-2][2]) > 1e-8):
        raise ValueError("Final action must repeat the terminal-clamped previous target")
    solutions, distances, angles, evaluations, widths = [], [], [], [], []
    previous_width = initial_width
    for index, action in enumerate(actions):
        width = solver.validate_width(action[7])
        if abs(width - previous_width) * fps > solver.maximum_width_speed + 1e-9:
            raise ReplayIKError(f"Action {index}: gripper width exceeds URDF velocity limit")
        try:
            solutions.append(solver.solve(action))
        except ReplayIKError as exc:
            raise ReplayIKError(f"Action {index}: {exc}") from exc
        distances.append(solver.last_position_error_m)
        angles.append(solver.last_rotation_error_rad)
        evaluations.append(solver.last_nfev)
        widths.append(width)
        previous_width = width
    solutions, widths = np.asarray(solutions), np.asarray(widths)
    stride = physics_hz // fps
    knots = np.vstack([initial_q, solutions[:-1]])
    width_knots = np.r_[initial_width, widths[:-1]]
    alpha = np.arange(1, stride + 1, dtype=float) / stride
    segments = knots[:-1, None, :] + alpha[None, :, None] * np.diff(knots, axis=0)[:, None, :]
    width_segments = width_knots[:-1, None] + alpha[None, :] * np.diff(width_knots)[:, None]
    commands = np.vstack([initial_q, segments.reshape(-1, 6)])
    gripper = np.r_[initial_width, width_segments.reshape(-1)]
    if np.any(np.abs(np.diff(commands, axis=0)) * physics_hz > solver.kinematics.velocity_limits + 1e-9):
        raise ReplayIKError("Interpolated arm commands exceed URDF velocity limits")
    return ReplayTrajectory(commands, gripper, np.arange(len(commands)) / physics_hz,
                            np.r_[-1, np.repeat(np.arange(len(actions) - 1), stride)],
                            solutions, np.asarray(distances), np.asarray(angles), np.asarray(evaluations),
                            fps, physics_hz, stride, solver.initial_seed_clipped)
