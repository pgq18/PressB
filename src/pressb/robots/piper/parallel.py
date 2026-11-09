"""PiPER tensor actuators for the task-independent parallel simulation loop.

These drivers map high-level actions into actuator setpoints. They never step
physics, render cameras, randomize scene objects or decide task outcomes.
"""
from __future__ import annotations

from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation

from pressb.control.joint_actions import JointActionPostprocessor
from pressb.control.pose_protocol import finite_array, pose9_to_pose8
from pressb.runtime.parallel_contract import ParallelRobot
from pressb.runtime.physics import TensorArms
from .kinematics import PiperKinematics
from .batched_control import BatchedPoseController


def _rows_for_roots(paths, roots):
    """Map nested articulation roots without assuming lexical PhysX ordering."""
    paths = [str(p) for p in paths]
    rows = []
    for root in roots:
        root = str(root).rstrip("/")
        matches = [i for i, path in enumerate(paths)
                   if path == root or path.startswith(root + "/")]
        if len(matches) != 1:
            raise RuntimeError(f"Expected exactly one articulation under {root}")
        rows.append(matches[0])
    if len(set(rows)) != len(rows) or len(rows) != len(paths):
        raise RuntimeError("PhysX articulation view contains unrelated or repeated roots")
    return np.asarray(rows, dtype=np.int64)


class PiperJointDriver(ParallelRobot):
    """Six arm joints plus a full-width gripper, optionally fixed at action_dim=6."""

    def __init__(self, engine, scene_config, initial_state, *, action_dim=7,
                 fixed_gripper_width_m=None, smoothing=None, arms=None):
        if type(action_dim) is not int or action_dim not in (6, 7):
            raise ValueError("PiPER joint actions require six or seven dimensions")
        self.engine, self.scene_config = engine, dict(scene_config)
        self.num_envs, self.action_dim = engine.num_envs, action_dim
        self.kin = PiperKinematics(scene_config["robot_urdf"], tip_offset=.1358)
        self.lower = np.r_[self.kin.lower, 0.]
        self.upper = np.r_[self.kin.upper, scene_config.get("gripper_max_width_m", .08)]
        initial = finite_array(initial_state, name="initial joint state")
        if initial.shape == (6,):
            if fixed_gripper_width_m is None:
                raise ValueError("Six-joint initial state needs an explicit gripper width")
            initial = np.r_[initial, fixed_gripper_width_m]
        if initial.shape == (7,):
            initial = np.repeat(initial[None], self.num_envs, axis=0)
        if initial.shape != (self.num_envs, 7):
            raise ValueError("Initial joints must be [7] or [num_envs,7]")
        self._validate_states(initial)
        self.initial_states = initial.copy()
        self.fixed_gripper_width_m = (float(initial[0, 6]) if fixed_gripper_width_m is None
                                     else float(finite_array(fixed_gripper_width_m, (), "fixed gripper width")))
        if not 0 <= self.fixed_gripper_width_m <= self.upper[6]:
            raise ValueError("Fixed gripper width exceeds robot travel")
        if action_dim == 6 and not np.allclose(initial[:, 6], self.fixed_gripper_width_m,
                                             rtol=0, atol=1e-8):
            raise ValueError("Six-joint initial state must use the configured fixed gripper")
        self.arms = arms if arms is not None else TensorArms(engine.simulation_view, engine.robot_paths)
        self.arm_rows = _rows_for_roots(self.arms.prim_paths, engine.robot_paths)
        self.dofs = np.array([self.arms.get_dof_index(f"joint{i}") for i in range(1, 9)])
        self.arm_dofs, self.finger_dofs = self.dofs[:6], self.dofs[6:]
        self.commands = self.arms.get_joint_positions()[self.arm_rows].copy()
        self.commands[:, self.dofs] = self._full(initial)
        self._set_gains()
        self.arms.set_joint_positions(self.commands, self.arm_rows)
        self.arms.set_joint_velocities(np.zeros_like(self.commands), self.arm_rows)
        self.arms.set_joint_position_targets(self.commands, self.arm_rows)
        self.last_sent = initial.copy()
        self.postprocessors = [JointActionPostprocessor(**(smoothing or {"mode": "none"}))
                               for _ in range(self.num_envs)]
        for i in range(self.num_envs):
            self.postprocessors[i].reset(initial[i])
        self.settled_joints = self.settled_velocities = None

    @staticmethod
    def _full(states):
        return np.c_[states[:, :6], states[:, 6] / 2, -states[:, 6] / 2]

    def _ids(self, env_ids):
        values = np.asarray(env_ids)
        if (values.ndim != 1 or values.dtype.kind not in "iu" or not len(values)
                or len(np.unique(values)) != len(values)
                or np.any(values < 0) or np.any(values >= self.num_envs)):
            raise ValueError("Expected distinct existing robot environment IDs")
        return values.astype(np.int64, copy=False)

    def _validate_states(self, states):
        if (not np.isfinite(states).all() or np.any(states < self.lower - 1e-6)
                or np.any(states > self.upper + 1e-6)):
            raise ValueError("Joint/gripper command exceeds URDF bounds")

    def _set_gains(self):
        keys = ("arm_stiffness", "arm_damping", "gripper_stiffness", "gripper_damping")
        if not any(key in self.scene_config for key in keys):
            return
        if not all(key in self.scene_config for key in keys):
            raise ValueError("Specify both arm and gripper stiffness/damping")
        stiffness = np.zeros_like(self.arms.view.get_dof_stiffnesses())
        damping = np.zeros_like(self.arms.view.get_dof_dampings())
        stiffness[:, self.arm_dofs] = self.scene_config["arm_stiffness"]
        damping[:, self.arm_dofs] = self.scene_config["arm_damping"]
        stiffness[:, self.finger_dofs] = self.scene_config["gripper_stiffness"]
        damping[:, self.finger_dofs] = self.scene_config["gripper_damping"]
        if not np.isfinite(stiffness).all() or not np.isfinite(damping).all():
            raise ValueError("Robot gains must be finite")
        rows = np.arange(self.num_envs, dtype=np.uint32)
        self.arms.view.set_dof_stiffnesses(stiffness, rows)
        self.arms.view.set_dof_dampings(damping, rows)
        if (not np.allclose(self.arms.view.get_dof_stiffnesses(), stiffness)
                or not np.allclose(self.arms.view.get_dof_dampings(), damping)):
            raise RuntimeError("PhysX robot gains do not match requested per-radian gains")

    def joint_arrays(self):
        q = self.arms.get_joint_positions()[self.arm_rows].copy()
        velocity = self.arms.get_joint_velocities()[self.arm_rows].copy()
        if not np.isfinite(q).all() or not np.isfinite(velocity).all():
            raise RuntimeError("Nonfinite measured PiPER articulation state")
        return q, velocity

    def states(self):
        q = self.joint_arrays()[0][:, self.dofs]
        return np.c_[q[:, :6], q[:, 6] - q[:, 7]]

    def cache_initial_state(self):
        self.settled_joints, self.settled_velocities = self.joint_arrays()
        self.last_sent = self.states()
        for i in range(self.num_envs):
            self.postprocessors[i].reset(self.last_sent[i])

    def _reset_values(self, ids, specs):
        if self.settled_joints is None:
            raise RuntimeError("Cache the settled robot state before episode reset")
        values = self.settled_joints[ids].copy()
        for row, i in enumerate(ids):
            if "initial_state_joint_rad_m" not in specs[int(i)]:
                continue
            initial = finite_array(specs[int(i)]["initial_state_joint_rad_m"], (7,), "reset joints")
            self._validate_states(initial)
            if self.action_dim == 6 and abs(initial[6] - self.fixed_gripper_width_m) > 1e-8:
                raise ValueError("Reset must preserve the configured fixed gripper width")
            values[row, self.dofs] = self._full(initial[None])[0]
        return values

    def reset(self, env_ids, specs):
        ids = self._ids(env_ids)
        values = self._reset_values(ids, specs)
        self.arms.set_joint_positions(values, self.arm_rows[ids])
        self.arms.set_joint_velocities(np.zeros_like(values), self.arm_rows[ids])
        self.arms.set_joint_position_targets(values, self.arm_rows[ids])
        self.commands[ids] = values
        self.last_sent[ids] = self.states()[ids]
        for i in ids:
            self.postprocessors[int(i)].reset(self.last_sent[i])

    def begin_action(self, env_ids, actions):
        ids = self._ids(env_ids)
        values = finite_array(actions, (len(ids), self.action_dim), "joint actions")
        if self.action_dim == 6:
            values = np.c_[values, np.full(len(ids), self.fixed_gripper_width_m)]
        self._validate_states(values)
        # Validate all rows before mutating any filter or command history.
        for row, i in enumerate(ids):
            self.last_sent[i] = self.postprocessors[int(i)].apply(values[row])
        self.commands[np.ix_(ids, self.dofs)] = self._full(self.last_sent[ids])
        return {int(i): self.last_sent[i].tolist() for i in ids}

    def validate_actions(self, actions):
        values = finite_array(actions, name="joint action sequence")
        if values.ndim != 2 or values.shape[1] != self.action_dim or not len(values):
            raise ValueError("Joint action sequence has the wrong shape")
        full = (np.c_[values, np.full(len(values), self.fixed_gripper_width_m)]
                if self.action_dim == 6 else values)
        self._validate_states(full)
        return values.copy()

    def before_tick(self, env_ids, substep, ticks_per_action):
        ids = self._ids(env_ids)
        # Position targets persist across the substeps of a control interval.
        if substep == 0:
            self.arms.set_joint_position_targets(self.commands[ids], self.arm_rows[ids])

    def observations(self, env_ids):
        ids = self._ids(env_ids)
        states = self.states()
        return {int(i): {"state": states[i].tolist(), "control_state": self.last_sent[i].tolist()}
                for i in ids}

    def diagnostic_state(self, env_ids):
        ids = self._ids(env_ids)
        q, velocity = self.joint_arrays()
        states = self.states()
        return {int(i): {"state": states[i].tolist(), "joints": q[i].tolist(),
                         "joint_velocity": velocity[i].tolist(), "command_joints": self.commands[i].tolist(),
                         "control_state": self.last_sent[i].tolist()} for i in ids}


class PiperPoseDriver(PiperJointDriver):
    """Absolute pose9 commands with bounded batched IK and causal joint smoothing."""

    def __init__(self, engine, scene_config, initial_state, *, smoothing_window=3,
                 ik_iterations=16, fixed_gripper_width_m=.008, arms=None):
        if type(smoothing_window) is not int or smoothing_window not in (1, 3, 5, 7, 9, 11):
            raise ValueError("Pose smoothing window must be one of 1,3,5,7,9,11")
        if abs(float(fixed_gripper_width_m) - .008) > 1e-8:
            raise ValueError("Current PiPER pose controller requires an 8 mm fixed gripper")
        scene = dict(scene_config)
        # Match the established pose controller's actuator gains in radians.
        scene.setdefault("arm_stiffness", 12000.)
        scene.setdefault("arm_damping", 600.)
        scene.setdefault("gripper_stiffness", 12000.)
        scene.setdefault("gripper_damping", 600.)
        super().__init__(engine, scene, initial_state, fixed_gripper_width_m=fixed_gripper_width_m,
                         smoothing={"mode": "none"}, arms=arms)
        self.action_dim = 9
        self.window = smoothing_window
        self.home = self.initial_states[:, :6].copy()
        self.batched_controller = BatchedPoseController(scene["robot_urdf"], self.home,
            fps=int(engine.config["control_hz"]), iterations=ik_iterations)
        self.history = [deque((q.copy() for _ in range(self.window)), maxlen=self.window)
                        for q in self.home]
        self.previous = self.home.copy()
        self.targets = self.home.copy()
        self.last_diagnostics = {i: {} for i in range(self.num_envs)}

    def reset(self, env_ids, specs):
        ids = self._ids(env_ids)
        values = self._reset_values(ids, specs)
        velocities = self.settled_velocities[ids].copy()
        targets = self.home[ids].copy()
        for row, i in enumerate(ids):
            if "initial_state_joint_rad_m" in specs[int(i)]:
                requested = np.asarray(specs[int(i)]["initial_state_joint_rad_m"], dtype=float)
                if abs(requested[6] - self.fixed_gripper_width_m) > 1e-8:
                    raise ValueError("Pose reset requires the fixed gripper width")
                targets[row] = requested[:6]
                velocities[row] = 0.
        self.arms.set_joint_positions(values, self.arm_rows[ids])
        self.arms.set_joint_velocities(velocities, self.arm_rows[ids])
        self.commands[np.ix_(ids, self.arm_dofs)] = targets
        self.commands[np.ix_(ids, self.finger_dofs)] = [self.fixed_gripper_width_m / 2,
                                                      -self.fixed_gripper_width_m / 2]
        self.arms.set_joint_position_targets(self.commands[ids], self.arm_rows[ids])
        self.batched_controller.reset(ids, targets)
        self.previous[ids] = self.targets[ids] = targets
        self.last_sent[ids] = self.states()[ids]
        for row, i in enumerate(ids):
            self.history[int(i)] = deque((targets[row].copy() for _ in range(self.window)), maxlen=self.window)
            self.last_diagnostics[int(i)] = {}

    def begin_action(self, env_ids, actions):
        ids = self._ids(env_ids)
        poses = finite_array(actions, (len(ids), 9), "pose9 actions")
        if np.any(np.linalg.norm(poses[:, :3], axis=1) > 1.5):
            raise ValueError("Pose target is more than 1.5 m from the robot base")
        converted = pose9_to_pose8(poses, gripper_width_m=self.fixed_gripper_width_m)
        previous = self.batched_controller.q[ids].copy()
        targets, diagnostics = self.batched_controller.solve(ids, converted)
        self.previous[ids], self.targets[ids] = previous, targets
        for row, i in enumerate(ids):
            self.last_diagnostics[int(i)] = dict(action_pose8=converted[row].tolist(),
                q_before=previous[row].tolist(), q_target=targets[row].tolist(), **diagnostics[row])
        return {int(i): poses[row].tolist() for row, i in enumerate(ids)}

    def validate_actions(self, actions):
        values = finite_array(actions, name="pose9 action sequence")
        if values.ndim != 2 or values.shape[1] != 9 or not len(values):
            raise ValueError("Pose9 action sequence has the wrong shape")
        if np.any(np.linalg.norm(values[:, :3], axis=1) > 1.5):
            raise ValueError("Pose target is more than 1.5 m from the robot base")
        pose9_to_pose8(values, gripper_width_m=self.fixed_gripper_width_m)
        return values.copy()

    def before_tick(self, env_ids, substep, ticks_per_action):
        ids = self._ids(env_ids)
        if (type(ticks_per_action) is not int or ticks_per_action < 1
                or not 0 <= substep < ticks_per_action):
            raise ValueError("Invalid control interpolation tick")
        linear = self.previous[ids] + (self.targets[ids] - self.previous[ids]) * (substep + 1) / ticks_per_action
        for row, i in enumerate(ids):
            self.history[int(i)].append(linear[row].copy())
            self.commands[i, self.arm_dofs] = np.mean(self.history[int(i)], axis=0)
        self.arms.set_joint_position_targets(self.commands[ids], self.arm_rows[ids])

    def observations(self, env_ids):
        ids = self._ids(env_ids)
        q, velocity = self.joint_arrays()
        transforms = self.batched_controller.fk(q[:, self.arm_dofs])
        quaternions = Rotation.from_matrix(transforms[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]
        measured = np.c_[transforms[:, :3, 3], quaternions,
                          q[:, self.finger_dofs[0]] - q[:, self.finger_dofs[1]]]
        ticks = self.engine.backend.ticks
        elapsed = ticks * self.engine.config["physics_dt"]
        maximum = self.engine.config["max_episode_seconds"]
        return {int(i): {"state": measured[i].tolist(), "control_state": np.r_[
            q[i, self.arm_dofs], velocity[i, self.arm_dofs], self.batched_controller.q[i],
            np.asarray(self.history[int(i)]).ravel(), elapsed[i] / maximum].tolist()} for i in ids}

    def diagnostic_state(self, env_ids):
        result = super().diagnostic_state(env_ids)
        observations = self.observations(env_ids)
        for i, values in result.items():
            values.update(observations[i])
            values.update(ik_endpoint=self.batched_controller.q[i].tolist(),
                          interpolation_start=self.previous[i].tolist(),
                          interpolation_target=self.targets[i].tolist(),
                          linear_history=np.asarray(self.history[i]).tolist())
        return result
