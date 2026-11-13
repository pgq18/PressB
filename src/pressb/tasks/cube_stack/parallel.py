"""Cube-stack task hooks; all ticks, rendering and scheduling are shared."""
from __future__ import annotations

from copy import deepcopy
import numpy as np

from pressb.robots.piper.kinematics import PiperKinematics
from pressb.runtime.physics import ContactForceReducer, ContactPathClassifier
from pressb.tasks.playground.parallel import PlaygroundTask, EXECUTION_KEYS
from pressb.tasks.playground.backend import STATE_NAMES, STATE_UNITS, sample_reset_spec
from .configuration import load_config, sample_objects, task_identity
from .task import COLORS, INSTRUCTION, CubeStackOutcome


class CubeStackTask(PlaygroundTask):
    """Reuse the calibrated workspace authoring and joint driver, with task hooks."""
    def __init__(self, config):
        # Do not call PlaygroundTask.__init__: its workspace identity and
        # no-success semantics are intentionally not this task's identity.
        self.config = cfg = load_config(config)
        self.execution = {key: deepcopy(cfg[key]) for key in EXECUTION_KEYS if key in cfg}
        self.execution.update(terminal_boundary="physics_tick", timeout_reason="timeout",
            warmup_frames=cfg["scene"].get("rendering", {}).get("warmup_frames", 8))
        kin = PiperKinematics(cfg["scene"]["robot_urdf"], tip_offset=.1358)
        self.lower = np.r_[kin.lower, 0.]
        self.upper = np.r_[kin.upper, cfg["scene"]["gripper_max_width_m"]]
        self.object_specs = {obj["name"]: deepcopy(obj) for obj in cfg["scene"]["objects"]}
        self.identity = task_identity(cfg)

    def metadata(self):
        cfg = self.config
        return dict(schema_version=2, node_type="simulation", task_id="cube_stack", num_envs=cfg["num_envs"],
            **deepcopy(self.identity),
            observation={"state": {"names": STATE_NAMES.copy(), "units": STATE_UNITS.copy()}, "cameras": ["up", "wrist"]},
            action={"representation": "joint_position", "frame": "robot_base", "names": STATE_NAMES.copy(),
                "units": STATE_UNITS.copy(), "horizon": cfg["horizon"], "control_hz": cfg["control_hz"],
                "lower": self.lower.tolist(), "upper": self.upper.tolist()},
            control_state_dim=7, task_features_dim=0, reset_mode="indexed",
            discount={"gamma": cfg["gamma"], "timebase_hz": cfg["control_hz"]}, smoothing=deepcopy(cfg["smoothing"]),
            task_instruction=INSTRUCTION, reward=deepcopy(cfg["reward"]),
            reward_schema={"kind": "sparse_physical_event", "values": deepcopy(cfg["reward"]),
                "success_reward": "Once, after continuous supported released stack dwell",
                "within_chunk_discount": "gamma ** ((event_tick - 1) / physics_ticks_per_action)", "timeout_reward": 0.},
            success_rule="After a simultaneous bilateral red grasp and lift, the upright red cube rests on the upright blue cube with vertical contact support, the blue cube rests on the table, and both cubes remain released and low-speed for the required dwell.",
            outcome_thresholds=deepcopy(cfg["outcome"]),
            object_reset={"objects": list(COLORS), "options": ["center_xy", "yaw_deg"],
                "frame": "robot_base XY metres; Z supported on tabletop",
                "randomization": deepcopy(cfg["randomization"]),
                "semantics": "Independent seeded proposals conditioned on full-footprint bounds and pair separation; no teleports after reset"},
            contact_units={"impulse": "N*s", "support_force": "sum(abs(world_Z_impulse))/physics_dt in N",
                           "finger_force": "sum(norm(impulse))/physics_dt in N", "separation": "m"})

    def build_template(self):
        template = super().build_template()
        from pxr import PhysxSchema
        scene = template.data["scene"]
        scene["task_id"] = "cube_stack"
        scene["outcome_semantics"] = "Physical contact-supported red-on-blue stacking with grasp/lift history"
        for color in COLORS:
            prim = template.stage.GetPrimAtPath(scene["objects"][color]["path"])
            # Robot reports alone cannot report cube/cube or cube/table support.
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
            # Keep support reports live throughout dwell, including a resting
            # stack. Only these two dynamic cubes are kept awake.
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateSleepThresholdAttr(0.)
        return template

    def sample_reset(self, env_id, seed, options):
        if not isinstance(options, dict):
            raise ValueError("reset options must be a mapping")
        options = deepcopy(options)
        if options.get("task", INSTRUCTION) != INSTRUCTION:
            raise ValueError("cube_stack uses its fixed red-on-blue language instruction")
        objects = sample_objects(self.config, env_id, seed, options.pop("objects", {}))
        options["task"] = INSTRUCTION
        spec = sample_reset_spec(self, seed, options)
        spec.update(env_id=env_id, objects=objects)
        return spec

    def bind(self, engine):
        super().bind(engine)
        if self.object_names != list(COLORS):
            raise RuntimeError("Cube-stack tensor row ordering must be red, blue")
        self.object_paths = {color: self.scene["objects"][color]["path"] for color in COLORS}
        self.table_path = self.scene["root_path"]+"/Table/Top"
        self.contact_paths = ContactPathClassifier([
            *(self.object_paths[color] for color in COLORS),
            *(engine.template.robot_path+f"/link{7+i}" for i in range(2)), self.table_path])
        self.contact_forces = ContactForceReducer(self.execution["physics_dt"],
            self.config["outcome"]["max_support_normal_tilt_deg"],
            self.config["outcome"]["contact_separation_tolerance_m"],
            "Nonfinite or malformed measured cube contact")
        self.contacts = [self._empty_contacts() for _ in range(engine.num_envs)]
        self.checkers = [None]*engine.num_envs
        self.contact_counts = np.zeros((engine.num_envs, 2, 2), dtype=np.int64)
        self.last_outcomes = [{} for _ in range(engine.num_envs)]

    @staticmethod
    def _empty_contacts():
        return dict(red_finger_forces_n=[0., 0.], blue_finger_forces_n=[0., 0.],
                    red_blue_support_force_n=0., blue_table_support_force_n=0., red_table_force_n=0.)

    def reset_task(self, env_ids, specs):
        # Parent hook only restores selected object poses and velocities; there
        # is no parent scheduler or parent reward/termination implementation.
        info = super().reset_task(env_ids, specs)
        cfg = self.config
        for eid in env_ids:
            self.checkers[eid] = CubeStackOutcome(cfg["cubes"]["red"]["edge_m"], cfg["cubes"]["blue"]["edge_m"],
                float(self.scene["table_top_z"]+self.engine.offsets[eid, 2]), **cfg["outcome"])
            self.contacts[eid] = self._empty_contacts()
            self.contact_counts[eid] = 0
            self.last_outcomes[eid] = {}
            info[eid].update(task_outcome_available=True, object_teleport_after_reset=False)
        return info

    def before_tick(self, env_ids):
        for eid in env_ids:
            self.contacts[eid] = self._empty_contacts()

    def on_contact(self, env_id, actor_paths, contacts):
        if not self.engine.backend.ready[env_id] or self.checkers[env_id] is None or not len(contacts):
            return
        if len(actor_paths) != 4:
            raise ValueError("Cube contacts require actor0/actor1/collider0/collider1 paths")
        red, blue, finger0, finger1, table = self.contact_paths(actor_paths)
        touched, fingers = (red, blue), (finger0, finger1)
        if not (red or blue):
            return
        cfg = self.config["outcome"]
        force, vertical = self.contact_forces(contacts)
        current = self.contacts[env_id]
        for object_index, color in enumerate(COLORS):
            if touched[object_index]:
                for finger in range(2):
                    if fingers[finger]:
                        current[color+"_finger_forces_n"][finger] += force
                        if force >= cfg["min_grasp_force_n"]:
                            self.contact_counts[env_id, object_index, finger] += 1
        if red and blue:
            current["red_blue_support_force_n"] += vertical
        if table:
            if red:
                current["red_table_force_n"] += vertical
            if blue:
                current["blue_table_support_force_n"] += vertical

    def after_tick(self, env_ids):
        positions, quaternions = self.bodies.get_world_poses()
        velocities = self.bodies.get_velocities()
        if not all(np.isfinite(value).all() for value in (positions, quaternions, velocities)):
            raise RuntimeError("Nonfinite cube-stack rigid-body state")
        states, result = self.engine.robot.observations(env_ids), {}
        cfg = self.config
        for eid in env_ids:
            rows = self.rows[eid]
            outcome = self.checkers[eid].update(seconds=float(self.engine.backend.ticks[eid])*cfg["physics_dt"],
                positions=positions[rows], quaternions_wxyz=quaternions[rows], velocities=velocities[rows],
                gripper_width_m=float(states[eid]["state"][6]), contacts=self.contacts[eid])
            self.last_outcomes[eid] = deepcopy(outcome)
            success, dropped = outcome["success"], outcome["dropped"]
            terminal = bool(dropped or (success and cfg["terminate_on_success"]))
            result[eid] = dict(reward=cfg["reward"]["failure"] if dropped else cfg["reward"]["success"] if outcome["success_event"] else 0.,
                success=success, terminated=terminal, truncated=False,
                reason="object_dropped" if dropped else "success" if terminal else None,
                info=dict(physical_outcome=outcome, contacts=deepcopy(self.contacts[eid]),
                    finger_contact_counts=self.contact_counts[eid].tolist(), object_teleport_after_reset=False,
                    task_outcome_available=True))
        return result


def create_backend(config, *, engine_factory=None):
    from pressb.runtime.parallel import SharedWorldBackend
    return SharedWorldBackend(CubeStackTask(config), engine_factory=engine_factory)
