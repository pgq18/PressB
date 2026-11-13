"""Compound-object cleanup hooks for the task-independent parallel runtime."""
from __future__ import annotations
from copy import deepcopy
import math
import numpy as np

from pressb.runtime.parallel_contract import ParallelTask, SceneTemplate
from pressb.runtime.physics import ContactForceReducer, ContactPathClassifier
from pressb.robots.piper.kinematics import PiperKinematics
from pressb.tasks.playground.parallel import PlaygroundTask, EXECUTION_KEYS
from pressb.tasks.playground.backend import ROOT, STATE_NAMES, STATE_UNITS, sample_reset_spec
from .configuration import load_config, sample_objects, task_identity
from .geometry import KINDS
from .task import INSTRUCTION, TabletopCleanupOutcome


class TabletopCleanupTask(PlaygroundTask):
    def __init__(self, config):
        self.config = cfg = load_config(config)
        self.execution = {key: deepcopy(cfg[key]) for key in EXECUTION_KEYS if key in cfg}
        self.execution.update(terminal_boundary="physics_tick", timeout_reason="timeout",
                             warmup_frames=cfg["scene"].get("rendering", {}).get("warmup_frames", 8))
        kin = PiperKinematics(cfg["scene"]["robot_urdf"], tip_offset=.1358)
        self.lower, self.upper = np.r_[kin.lower, 0.], np.r_[kin.upper, cfg["scene"]["gripper_max_width_m"]]
        self.identity = task_identity(cfg)

    def metadata(self):
        cfg = self.config
        return dict(schema_version=2, node_type="simulation", task_id="tabletop_cleanup", num_envs=cfg["num_envs"],
            **deepcopy(self.identity),
            observation={"state": {"names": STATE_NAMES.copy(), "units": STATE_UNITS.copy()}, "cameras": ["up", "wrist"]},
            action={"representation": "joint_position", "frame": "robot_base", "names": STATE_NAMES.copy(), "units": STATE_UNITS.copy(),
                "horizon": cfg["horizon"], "control_hz": cfg["control_hz"], "lower": self.lower.tolist(), "upper": self.upper.tolist()},
            control_state_dim=7, task_features_dim=3, reset_mode="indexed", smoothing=deepcopy(cfg["smoothing"]),
            discount={"gamma": cfg["gamma"], "timebase_hz": cfg["control_hz"]}, task_instruction=INSTRUCTION,
            object_slots=list(KINDS), active_object_count=2,
            task_features={"names": [kind+"_active" for kind in KINDS], "units": ["binary"]*3},
            reward=deepcopy(cfg["reward"]), reward_schema={"kind": "sparse_physical_event", "values": deepcopy(cfg["reward"]),
                "success_reward": "Once, when both active objects satisfy the continuous released support dwell", "timeout_reward": 0.},
            success_rule="Both active objects have unsupported bilateral grasp/lift history; full surface XY geometry inside the bin, above its floor, supported through a physical contact graph to the floor, released and low-speed for the full dwell.",
            outcome_thresholds=deepcopy(cfg["outcome"]), fixed_bin=deepcopy(cfg["bin"]),
            object_reset={"exact_active_count": 2, "randomization": deepcopy(cfg["randomization"]),
                "options": "objects: exactly two distinct pool kind keys, each with center_xy/yaw_deg",
                "inactive": "Native PhysX simulation disabled, invisible, zero velocity, parked below workspace; no per-tick object teleport"},
            contact_units={"impulse": "N*s", "vertical_support_force": "sum(abs(world_Z_impulse))/physics_dt in N",
                "finger_force": "sum(norm(impulse))/physics_dt in N", "separation": "m"})

    def build_template(self):
        from pressb.robots.piper.isaac import build_piper_runtime
        from pressb.runtime.workbench import build_tripod
        from .scene import build_scene
        cfg = deepcopy(self.config["scene"])
        runtime = build_piper_runtime(cfg, build_scene, asset_root=ROOT,
            camera_hardware_builder=build_tripod if cfg.get("camera_tripod", False) else None, initialize=False)
        return SceneTemplate(runtime.world, runtime.stage, runtime.robot_path, runtime.camera_paths,
                             data={"scene": runtime.scene, "camera_specs": runtime.camera_specs})

    def sample_reset(self, env_id, seed, options):
        if not isinstance(options, dict):
            raise ValueError("Reset options must be a mapping")
        options = deepcopy(options)
        if options.get("task", INSTRUCTION) != INSTRUCTION:
            raise ValueError("Cleanup uses a fixed language instruction")
        layout = sample_objects(self.config, env_id, seed, options.pop("objects", None))
        options["task"] = INSTRUCTION
        spec = sample_reset_spec(self, seed, options)
        spec.update(env_id=env_id, **layout)
        return spec

    def bind(self, engine):
        ParallelTask.bind(self, engine)
        self.scene = engine.template.data["scene"]
        self.object_paths = {kind: self.scene["objects"][kind]["path"] for kind in KINDS}
        self.bodies, self.rows = engine.rigid_bodies(list(self.object_paths.values()))
        self.table_path = self.scene["root_path"]+"/Table/Top"
        self.bin_floor_path = self.scene["bin"]["floor_path"]
        self.contact_paths = ContactPathClassifier([
            *(self.object_paths[kind] for kind in KINDS),
            *(engine.template.robot_path+f"/link{7+i}" for i in range(2)),
            self.table_path, self.bin_floor_path])
        self.contact_forces = ContactForceReducer(self.execution["physics_dt"],
            self.config["outcome"]["max_support_normal_tilt_deg"],
            self.config["outcome"]["contact_separation_tolerance_m"],
            "Malformed cleanup contact data")
        self.active_masks = np.zeros((engine.num_envs, 3), dtype=bool)
        self.contacts = [self._empty_contacts() for _ in range(engine.num_envs)]
        self.checkers = [None]*engine.num_envs
        self.last_outcomes = [{} for _ in range(engine.num_envs)]
        # PhysX stores transforms in float32. Cache that exact target so large
        # clone offsets do not look like drift due to metre-scale ULP rounding.
        self.parked_positions = np.array([[engine.offsets[eid]+[j*.3, -2., -5.] for j in range(3)]
            for eid in range(engine.num_envs)], dtype=np.float32).astype(np.float64)
        # All template slots are absent until the first reset. They never settle
        # together or collide during startup, and the tensor view keeps stable rows.
        ids = list(range(engine.num_envs))
        self._write_slots(ids, {eid: {"active_mask": [False]*3, "objects": {}} for eid in ids})

    def after_settle(self):
        self._check_inactive(list(range(self.engine.num_envs)))

    @staticmethod
    def _empty_contacts():
        return dict(finger_forces_n=np.zeros((3, 2)), table_forces_n=np.zeros(3),
                    bin_floor_forces_n=np.zeros(3), object_support_forces_n=np.zeros((3, 3)))

    def _visibility(self, env_id, index, visible=None):
        from pxr import UsdGeom
        path = self.engine.relocate(env_id, self.object_paths[KINDS[index]])
        imageable = UsdGeom.Imageable(self.engine.stage.GetPrimAtPath(path))
        if visible is not None:
            imageable.MakeVisible() if visible else imageable.MakeInvisible()
        return imageable.GetVisibilityAttr().Get() != UsdGeom.Tokens.invisible

    def _disabled(self):
        values = np.asarray(self.bodies.view.get_disable_simulations()).copy()
        if values.shape != (len(self.bodies.prim_paths), 1):
            raise RuntimeError("Unexpected native rigid-body simulation mask shape")
        return values

    def _write_slots(self, env_ids, specs):
        indices = np.concatenate([self.rows[eid] for eid in env_ids]).astype(np.uint32)
        flags = self._disabled()
        flags[indices] = 0
        self.bodies.view.set_disable_simulations(flags, indices)
        base = np.asarray(self.scene["robot_base_position_world"])
        positions, rotations = [], []
        for eid in env_ids:
            mask = np.asarray(specs[eid]["active_mask"], dtype=bool)
            self.active_masks[eid] = mask
            for j, kind in enumerate(KINDS):
                if mask[j]:
                    pose = specs[eid]["objects"][kind]
                    half = math.radians(pose["yaw_deg"])/2
                    bottom = float(np.asarray(self.config["asset_specs"][j]["support_vertices"])[:, 2].min())
                    p = np.r_[base[:2]+pose["center_xy"], float(self.scene["table_top_z"])-bottom+.0005]+self.engine.offsets[eid]
                    q = [math.cos(half), 0., 0., math.sin(half)]
                else:
                    p, q = self.parked_positions[eid, j], [1., 0., 0., 0.]
                positions.append(p); rotations.append(q)
                flags[self.rows[eid, j], 0] = 0 if mask[j] else 1
                self._visibility(eid, j, visible=bool(mask[j]))
        self.bodies.set_world_poses(np.asarray(positions), np.asarray(rotations), indices)
        self.bodies.set_velocities(np.zeros((len(indices), 6)), indices)
        self.bodies.view.set_disable_simulations(flags, indices)
        measured = self._disabled()
        if not np.array_equal(measured[indices], flags[indices]):
            raise RuntimeError("Native object simulation mask did not apply")
        self._check_inactive(env_ids)

    def _check_inactive(self, env_ids):
        flags = self._disabled()
        positions, rotations = self.bodies.get_world_poses()
        velocities = self.bodies.get_velocities()
        for eid in env_ids:
            for j in range(3):
                row, active = self.rows[eid, j], bool(self.active_masks[eid, j])
                if bool(flags[row, 0]) == active:
                    raise RuntimeError("Native simulation participation differs from task active mask")
                if self._visibility(eid, j) != active:
                    raise RuntimeError("Object visibility differs from task active mask")
                if not active and (not np.allclose(positions[row], self.parked_positions[eid, j], rtol=0, atol=1e-6)
                        or not np.allclose(velocities[row], 0., rtol=0, atol=1e-8)):
                    raise RuntimeError("Inactive object drifted despite disabled native simulation")

    def reset_task(self, env_ids, specs):
        self._write_slots(env_ids, specs)
        info = {}
        for eid in env_ids:
            box = deepcopy(self.scene["bin"])
            box["center_world"] = (np.asarray(box["center_world"])+self.engine.offsets[eid]).tolist()
            self.checkers[eid] = TabletopCleanupOutcome(self.config["asset_specs"],
                float(self.scene["table_top_z"]+self.engine.offsets[eid, 2]), box, self.active_masks[eid], **self.config["outcome"])
            self.contacts[eid] = self._empty_contacts()
            self.last_outcomes[eid] = {}
            info[eid] = dict(active_names=list(specs[eid]["active_names"]), active_mask=self.active_masks[eid].tolist(),
                object_teleport_after_reset=False, zero_tick_reset=True, task_outcome_available=True)
        return info

    def observation_fields(self, env_ids):
        values = ParallelTask.observation_fields(self, env_ids)
        for eid in env_ids:
            values[eid]["task_features"] = self.active_masks[eid].astype(float).tolist()
        return values

    def before_tick(self, env_ids):
        for eid in env_ids:
            self.contacts[eid] = self._empty_contacts()

    def on_contact(self, env_id, actor_paths, contacts):
        if not self.engine.backend.ready[env_id] or self.checkers[env_id] is None or not len(contacts):
            return
        if len(actor_paths) != 4:
            raise ValueError("Compound contact routing requires actors and colliders")
        route = self.contact_paths(actor_paths)
        touched = [j for j in range(len(KINDS)) if route[j]]
        if not touched:
            return
        # Active/ready state is deliberately checked outside the topology cache.
        if any(not self.active_masks[env_id, j] for j in touched):
            raise RuntimeError("Inactive object leaked into a live contact report")
        force, vertical = self.contact_forces(contacts)
        finger0, finger1, table, bin_floor = route[len(KINDS):]
        fingers = (finger0, finger1)
        row = self.contacts[env_id]
        for j in touched:
            for finger in range(2):
                if fingers[finger]:
                    row["finger_forces_n"][j, finger] += force
            if table:
                row["table_forces_n"][j] += vertical
            if bin_floor:
                row["bin_floor_forces_n"][j] += vertical
        if len(touched) == 2:
            a, b = touched
            row["object_support_forces_n"][a, b] += vertical
            row["object_support_forces_n"][b, a] += vertical

    def after_tick(self, env_ids):
        self._check_inactive(env_ids)
        positions, quats = self.bodies.get_world_poses()
        velocities, states, result = self.bodies.get_velocities(), self.engine.robot.observations(env_ids), {}
        for eid in env_ids:
            rows = self.rows[eid]
            outcome = self.checkers[eid].update(seconds=float(self.engine.backend.ticks[eid])*self.config["physics_dt"],
                positions=positions[rows], quaternions_wxyz=quats[rows], velocities=velocities[rows],
                gripper_width_m=float(states[eid]["state"][6]), contacts=self.contacts[eid])
            self.last_outcomes[eid] = deepcopy(outcome)
            success, dropped = outcome["success"], outcome["dropped"]
            terminal = bool(dropped or (success and self.config["terminate_on_success"]))
            result[eid] = dict(success=success, terminated=terminal, truncated=False,
                reason="object_dropped" if dropped else "success" if terminal else None,
                reward=self.config["reward"]["failure"] if dropped else self.config["reward"]["success"] if outcome["success_event"] else 0.,
                info=dict(physical_outcome=outcome, active_mask=self.active_masks[eid].tolist(),
                    active_names=[k for j, k in enumerate(KINDS) if self.active_masks[eid, j]],
                    contacts={key: value.tolist() for key, value in self.contacts[eid].items()},
                    object_teleport_after_reset=False, task_outcome_available=True))
        return result

    def diagnostic_state(self, env_ids):
        positions, quats = self.bodies.get_world_poses()
        velocities, flags = self.bodies.get_velocities(), self._disabled()
        return {eid: {"active_mask": self.active_masks[eid].tolist(), "objects": {
            kind: dict(position_world=positions[self.rows[eid, j]].tolist(), orientation_wxyz=quats[self.rows[eid, j]].tolist(),
                velocity=velocities[self.rows[eid, j]].tolist(), active=bool(self.active_masks[eid, j]),
                simulation_disabled=bool(flags[self.rows[eid, j], 0]), visible=self._visibility(eid, j))
            for j, kind in enumerate(KINDS)}} for eid in env_ids}


def create_backend(config, *, engine_factory=None):
    from pressb.runtime.parallel import SharedWorldBackend
    return SharedWorldBackend(TabletopCleanupTask(config), engine_factory=engine_factory)
