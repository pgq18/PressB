"""Calibrated workspace task hooks using the shared parallel simulation core."""
from __future__ import annotations

from copy import deepcopy
import math
import numpy as np

from pressb.robots.piper.kinematics import PiperKinematics
from pressb.runtime.contract import finite_array
from pressb.runtime.parallel_contract import ParallelTask, SceneTemplate
from .backend import ROOT, STATE_NAMES, STATE_UNITS, load_config, sample_reset_spec, task_identity

EXECUTION_KEYS = ("num_envs", "gpu", "horizon", "physics_dt", "control_hz", "max_episode_seconds",
    "settling_ticks", "gamma", "headless", "spacing_m", "render_resolution", "renderer_profile",
    "render_subframes", "observation_render_frames", "reset_render_frames", "cpu_threads",
    "png_workers", "log_dir", "image_preprocessing", "image_output_resolution")


class PlaygroundTask(ParallelTask):
    def __init__(self, config):
        self.config = load_config(config)
        self.execution = {key: deepcopy(self.config[key]) for key in EXECUTION_KEYS if key in self.config}
        self.execution.update(terminal_boundary="control_interval", timeout_reason="time_limit")
        self.execution.setdefault("log_dir", str(ROOT / "outputs/playground_parallel"))
        kin = PiperKinematics(self.config["scene"]["robot_urdf"], tip_offset=.1358)
        self.lower = np.r_[kin.lower, 0.]
        self.upper = np.r_[kin.upper, self.config["scene"]["gripper_max_width_m"]]
        self.identity = task_identity(self.config)
        objects = self.config["scene"]["objects"]
        if not isinstance(objects, list) or any(not isinstance(obj, dict) or not isinstance(obj.get("name"), str) for obj in objects):
            raise ValueError("Playground objects must be named object mappings")
        self.object_specs = {obj["name"]: deepcopy(obj) for obj in objects}
        if len(self.object_specs) != len(objects):
            raise ValueError("Playground object names must be distinct")
        self._object_options({})

    def metadata(self):
        cfg = self.config
        return dict(schema_version=2, node_type="simulation", task_id="playground", num_envs=cfg["num_envs"],
            **self.identity,
            observation={"state": {"names": STATE_NAMES.copy(), "units": STATE_UNITS.copy()}, "cameras": ["up", "wrist"]},
            action={"representation": "joint_position", "frame": "robot_base", "names": STATE_NAMES.copy(),
                "units": STATE_UNITS.copy(), "horizon": cfg["horizon"], "control_hz": cfg["control_hz"],
                "lower": self.lower.tolist(), "upper": self.upper.tolist()},
            control_state_dim=7, task_features_dim=0, reset_mode="indexed",
            discount={"gamma": cfg["gamma"], "timebase_hz": cfg["control_hz"]}, smoothing=deepcopy(cfg["smoothing"]),
            reward_schema={"kind": "none", "value": 0.},
            success_rule="Workspace twin only: no automatic success or task reward; timeout truncates episodes.",
            object_reset={"objects": list(self.object_specs), "options": ["center_xy", "yaw_deg"],
                "semantics": "Existing dynamic objects only; selected poses and velocities restored without physics ticks"})

    def build_template(self):
        from pressb.robots.piper.isaac import build_piper_runtime
        from pressb.runtime.workbench import build_tripod
        from .scene import build_scene
        cfg = deepcopy(self.config["scene"])
        runtime = build_piper_runtime(cfg, build_scene, asset_root=ROOT,
            camera_hardware_builder=build_tripod if cfg.get("camera_tripod", False) else None,
            initialize=False)
        return SceneTemplate(runtime.world, runtime.stage, runtime.robot_path, runtime.camera_paths,
            data={"scene": runtime.scene, "camera_specs": runtime.camera_specs})

    def create_robot(self, engine):
        from pressb.robots.piper.parallel import PiperJointDriver
        return PiperJointDriver(engine, self.config["scene"], self.config["scene"]["initial_state_joint_rad_m"],
            action_dim=7, smoothing=self.config["smoothing"])

    def _object_options(self, overrides):
        if not isinstance(overrides, dict) or set(overrides) - set(self.object_specs):
            raise ValueError("objects must map existing dynamic object names to reset positions")
        scene = self.config["scene"]
        table = scene.get("table", {})
        xmin = float(scene.get("table_x_min", table.get("x_min", -.12)))
        depth = float(scene.get("table_depth", table.get("depth", 1.0)))
        width = float(scene.get("table_width", table.get("width", 1.6)))
        result = {}
        for name, obj in self.object_specs.items():
            changes = overrides.get(name, {})
            if not isinstance(changes, dict) or set(changes) - {"center_xy", "yaw_deg"}:
                raise ValueError("Object resets support center_xy and yaw_deg only")
            xy = finite_array(changes.get("center_xy", obj.get("center_xy")), (2,), f"objects.{name}.center_xy", dtype=np.float64)
            size = finite_array(obj.get("size"), (3,), f"objects.{name}.size", dtype=np.float64)
            if np.any(size <= 0):
                raise ValueError("Object size must be positive")
            yaw = changes.get("yaw_deg", obj.get("yaw_deg", 0.))
            if isinstance(yaw, bool) or not isinstance(yaw, (int, float)) or not math.isfinite(yaw):
                raise ValueError("Object yaw_deg must be finite")
            angle = math.radians(yaw)
            ex, ey = (abs(math.cos(angle))*size[0] + abs(math.sin(angle))*size[1])/2, (abs(math.sin(angle))*size[0] + abs(math.cos(angle))*size[1])/2
            if not (xmin <= xy[0]-ex and xy[0]+ex <= xmin+depth and -width/2 <= xy[1]-ey and xy[1]+ey <= width/2):
                raise ValueError(f"Object {name} must be fully supported by the tabletop")
            result[name] = {"center_xy": xy.tolist(), "yaw_deg": float(yaw), "yaw_override": "yaw_deg" in changes}
        return result

    def sample_reset(self, env_id, seed, options):
        if not isinstance(options, dict):
            raise ValueError("reset options must be a mapping")
        options = deepcopy(options)
        objects = self._object_options(options.pop("objects", {}))
        spec = sample_reset_spec(self, seed, options)
        spec.update(env_id=env_id, objects=objects)
        return spec

    def bind(self, engine):
        super().bind(engine)
        self.scene = engine.template.data["scene"]
        self.object_names = list(self.object_specs)
        paths = [self.scene["objects"][name]["path"] for name in self.object_names]
        self.bodies, self.rows = engine.rigid_bodies(paths) if paths else (None, np.empty((engine.num_envs, 0), dtype=np.int64))

    def after_settle(self):
        if self.bodies is not None:
            positions, orientations = self.bodies.get_world_poses()
            self.initial_positions = positions[self.rows].copy()
            self.initial_orientations = orientations[self.rows].copy()

    def reset_task(self, env_ids, specs):
        info = {eid: {**deepcopy(specs[eid]), "task_outcome_available": False, "zero_tick_reset": True} for eid in env_ids}
        if self.bodies is None:
            return info
        positions, orientations, indices = [], [], []
        for eid in env_ids:
            local = self.initial_positions[eid].copy()
            quats = self.initial_orientations[eid].copy()
            for j, name in enumerate(self.object_names):
                requested = specs[eid]["objects"][name]
                local[j, :2] += np.subtract(requested["center_xy"], self.object_specs[name]["center_xy"])
                if requested["yaw_override"]:
                    half = math.radians(requested["yaw_deg"])/2
                    quats[j] = [math.cos(half), 0., 0., math.sin(half)]
            positions.extend(local);orientations.extend(quats);indices.extend(self.rows[eid])
        indices = np.asarray(indices, dtype=np.int64)
        self.bodies.set_world_poses(np.asarray(positions), np.asarray(orientations), indices)
        self.bodies.set_velocities(np.zeros((len(indices), 6)), indices)
        return info

    def after_tick(self, env_ids):
        if self.bodies is not None:
            positions, orientations = self.bodies.get_world_poses()
            velocities = self.bodies.get_velocities()
            if not all(np.isfinite(value).all() for value in (positions, orientations, velocities)):
                raise RuntimeError("Nonfinite playground object physics state")
        return {eid: {"reward": 0., "success": False, "terminated": False,
            "info": {"task_outcome_available": False}} for eid in env_ids}

    def diagnostic_state(self, env_ids):
        if self.bodies is None:
            return {eid: {"objects": {}} for eid in env_ids}
        positions, orientations = self.bodies.get_world_poses()
        velocities = self.bodies.get_velocities()
        return {eid: {"objects": {name: {"position_world": positions[self.rows[eid, j]].tolist(),
            "orientation_wxyz": orientations[self.rows[eid, j]].tolist(),
            "velocity": velocities[self.rows[eid, j]].tolist()} for j, name in enumerate(self.object_names)}} for eid in env_ids}


def create_backend(config, *, engine_factory=None):
    from pressb.runtime.parallel import SharedWorldBackend
    return SharedWorldBackend(PlaygroundTask(config), engine_factory=engine_factory)
