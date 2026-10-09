"""Vectorized URDF FK/Jacobian and bounded damped-least-squares pose control.

This is an explicit fast control contract, not a bitwise replacement for the
historical SciPy trust-region solver. Targets remain absolute base-frame TCP
poses; every command obeys the same URDF position and per-action speed limits.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from pressb.kinematics import PiperKinematics


class BatchedPoseController:
    contract = "urdf_analytic_jacobian_bounded_dls_v1"

    def __init__(self, urdf, initial_q, *, fps=30, tip_offset=.1358,
                 iterations=12, damping=1e-3, rotation_weight=.15):
        self.kinematics = PiperKinematics(urdf, tip_offset=tip_offset)
        self.fps = int(fps)
        if self.fps <= 0 or 120 % self.fps or int(iterations) != iterations or iterations < 1:
            raise ValueError("Require positive iterations and an action rate dividing 120 Hz")
        if not np.isfinite([damping, rotation_weight]).all() or damping <= 0 or rotation_weight <= 0:
            raise ValueError("Require positive finite damping/rotation weight")
        initial_q = np.asarray(initial_q, dtype=np.float64)
        if initial_q.ndim != 2 or initial_q.shape[1] != 6 or not len(initial_q):
            raise ValueError("initial_q must be [N,6]")
        self.q = np.empty_like(initial_q)
        self.maximum_joint_step = self.kinematics.velocity_limits / self.fps
        self.iterations, self.damping, self.rotation_weight = int(iterations), float(damping), float(rotation_weight)
        self.reset(np.arange(len(initial_q)), initial_q)

    def _ids(self, env_ids):
        ids = np.asarray(env_ids)
        if ids.ndim != 1 or ids.dtype.kind not in "iu" or len(np.unique(ids)) != len(ids):
            raise ValueError("env_ids must be unique integer indices")
        if np.any(ids < 0) or np.any(ids >= len(self.q)):
            raise ValueError("env_id out of bounds")
        return ids.astype(np.int64, copy=False)

    def reset(self, env_ids, q):
        ids = self._ids(env_ids)
        q = np.asarray(q, dtype=np.float64)
        kin = self.kinematics
        if q.shape != (len(ids), 6) or not np.isfinite(q).all():
            raise ValueError("Reset q must be finite [len(env_ids),6]")
        if np.any(q < kin.lower - 1e-3) or np.any(q > kin.upper + 1e-3):
            raise ValueError("Reset q exceeds URDF joint limits")
        self.q[ids] = np.clip(q, kin.lower, kin.upper)

    def fk(self, q, *, tip_offset=None, jacobian=False):
        q = np.asarray(q, dtype=np.float64)
        if q.ndim != 2 or q.shape[1] != 6 or not np.isfinite(q).all():
            raise ValueError("q must be finite [N,6]")
        count = len(q)
        transform = np.broadcast_to(np.eye(4), (count, 4, 4)).copy()
        origins, axes = [], []
        joint = 0
        for origin, axis in self.kinematics._chain:
            transform = transform @ origin
            if axis is not None:
                if jacobian:
                    origins.append(transform[:, :3, 3].copy())
                    axes.append(np.einsum("bij,j->bi", transform[:, :3, :3], axis))
                # Rodrigues with a constant URDF axis; avoid constructing one
                # scipy Rotation/Python object for each environment/joint.
                x, y, z = axis
                cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
                theta = q[:, joint, None, None]
                rotation = np.eye(3) + np.sin(theta) * cross + (1 - np.cos(theta)) * (cross @ cross)
                transform[:, :3, :3] = transform[:, :3, :3] @ rotation
                joint += 1
        offset = self.kinematics.tip_offset if tip_offset is None else float(tip_offset)
        transform[:, :3, 3] += transform[:, :3, 2] * offset
        if not jacobian:
            return transform
        axes = np.stack(axes, axis=2)
        origins = np.stack(origins, axis=2)
        displacement = transform[:, :3, 3, None] - origins
        linear = np.cross(axes, displacement, axisa=1, axisb=1, axisc=1)
        return transform, np.concatenate((linear, axes), axis=1)

    def _error(self, transforms, positions, rotations):
        angular = Rotation.from_matrix(rotations @ transforms[:, :3, :3].transpose(0, 2, 1)).as_rotvec()
        return np.concatenate((positions - transforms[:, :3, 3], self.rotation_weight * angular), axis=1)

    def solve(self, env_ids, actions_pose8):
        ids = self._ids(env_ids)
        actions = np.asarray(actions_pose8, dtype=np.float64)
        if not len(ids) or actions.shape != (len(ids), 8) or not np.isfinite(actions).all():
            raise ValueError("Actions must be finite [len(env_ids),8]")
        norms = np.linalg.norm(actions[:, 3:7], axis=1)
        if np.any(np.abs(norms - 1) > 1e-5) or np.any(np.abs(actions[:, 7] - .008) > 1e-8):
            raise ValueError("Actions require unit wxyz quaternion and fixed 0.008 m gripper")
        rotations = Rotation.from_quat(actions[:, [4, 5, 6, 3]]).as_matrix()
        positions = actions[:, :3]
        seed = self.q[ids].copy()
        kin = self.kinematics
        lower = np.maximum(kin.lower, seed - self.maximum_joint_step)
        upper = np.minimum(kin.upper, seed + self.maximum_joint_step)
        q = seed.copy()
        best_error = self._error(self.fk(q), positions, rotations)
        best_cost = np.einsum("bi,bi->b", best_error, best_error)
        used = np.zeros(len(ids), dtype=int)
        eye = np.eye(6)
        for iteration in range(self.iterations):
            transform, jac = self.fk(q, jacobian=True)
            error = self._error(transform, positions, rotations)
            jac[:, 3:] *= self.rotation_weight
            gradient = np.einsum("bij,bi->bj", jac, error)
            fixed = ((q <= lower + 1e-10) & (gradient < 0)) | ((q >= upper - 1e-10) & (gradient > 0))
            jac *= (~fixed)[:, None, :]
            transpose = jac.transpose(0, 2, 1)
            delta = np.linalg.solve(transpose @ jac + self.damping**2 * eye,
                                    (transpose @ error[..., None]))[..., 0]
            # A near-singular Newton direction can be many radians long.
            # Scale the whole direction before box projection so independent
            # clipping does not turn a descent direction into an ascent step.
            relative = np.max(np.abs(delta) / self.maximum_joint_step, axis=1, keepdims=True)
            delta /= np.maximum(relative, 1.)
            # Per-environment backtracking avoids accepting a worse projection
            # near joint limits, singularities, or far-out policy targets.
            improved = np.zeros(len(ids), dtype=bool)
            for scale in (1., .5, .25, .125):
                candidate = np.clip(q + scale * delta, lower, upper)
                candidate_error = self._error(self.fk(candidate), positions, rotations)
                cost = np.einsum("bi,bi->b", candidate_error, candidate_error)
                better = cost < best_cost - 1e-16
                q[better] = candidate[better]
                best_cost[better] = cost[better]
                improved |= better
                if np.all(better | (best_cost < 1e-16)):
                    break
            used[improved] = iteration + 1
            if not np.any(improved):
                break
        error = self._error(self.fk(q), positions, rotations)
        if not np.isfinite(q).all() or np.any(np.abs(q - seed) > self.maximum_joint_step + 1e-10):
            raise RuntimeError("Batched controller violated a command bound")
        self.q[ids] = q
        diagnostics = [dict(
            command_position_residual_m=float(np.linalg.norm(error[i, :3])),
            command_rotation_residual_rad=float(np.linalg.norm(error[i, 3:]) / self.rotation_weight),
            velocity_saturated_joints=np.flatnonzero(np.abs(q[i] - seed[i]) >= self.maximum_joint_step - 1e-7).tolist(),
            position_limited_joints=np.flatnonzero((q[i] <= kin.lower + 1e-7) | (q[i] >= kin.upper - 1e-7)).tolist(),
            solver_evaluations=int(used[i]), controller=self.contract,
        ) for i in range(len(ids))]
        return q.copy(), diagnostics
