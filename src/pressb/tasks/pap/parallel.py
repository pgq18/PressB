"""Pick-and-place hooks for the common parallel simulator.

The runtime owns physics stepping, slot scheduling, episode clocks, terminal
observations and rendering. This plugin only authors and interprets the task.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

import numpy as np

from pressb.runtime.parallel_contract import ParallelTask, SceneTemplate
from pressb.runtime.image_transport import validate_image_preprocessing
from pressb.robots.piper.kinematics import PiperKinematics
from .backend import (NAMES, UNITS, _integer, _path, _positive, _source_identity,
                      _source_revision, load_config as load_pap_config,
                      sample_reset, task_identity)
from .initialization import load_policy_initial
from .task import PickPlaceOutcome


RESET_CONTRACT = "indexed_zero_tick_cached_settled_arm_and_supported_object_v1"
_SOURCE_FILES = ("tasks/pap/parallel.py", "tasks/pap/vector.py")


def renderer_profile(name):
    from pressb.runtime.isaac_parallel import renderer_profile as common_profile
    return common_profile(name)


def load_config(config):
    result = load_pap_config(config)
    if len(result["objects"]) != 1:
        raise ValueError("Shared-World PAP currently supports one configured object geometry; use process PAP for variants")
    if len({w["gpu"] for w in result["workers"]}) != 1:
        raise ValueError("A shared-World PAP process uses one GPU; use a separate node per GPU")
    result["gpu"] = result["workers"][0]["gpu"]
    result["spacing_m"] = _positive(result.get("spacing_m", 6.), "spacing_m")
    if result["spacing_m"] < 6.:
        raise ValueError("spacing_m must be >= 6 to separate the calibrated room floors")
    render = result["scene"].get("rendering", {})
    result["renderer_profile"] = result.get("renderer_profile", "pap_calibrated")
    renderer_profile(result["renderer_profile"])
    fast = result["renderer_profile"] == "rl_fast"
    for name, default in (("cpu_threads", 8), ("png_workers", 8),
                          ("render_subframes", 1 if fast else render.get("subframes_per_frame", 1)),
                          ("observation_render_frames", 2 if fast else 1),
                          ("reset_render_frames", 2 if fast else render.get("warmup_frames", 8))):
        result[name] = _integer(result.get(name, default), name)
    resolution = result.get("render_resolution", result["scene"]["up_camera"]["resolution"])
    if (not isinstance(resolution, (list, tuple)) or len(resolution) != 2
            or any(type(v) is not int or v <= 0 for v in resolution)):
        raise ValueError("render_resolution requires two positive integer dimensions")
    result["render_resolution"] = list(resolution)
    result["image_preprocessing"] = validate_image_preprocessing(result.get("image_preprocessing"))
    for camera in ("up_camera", "wrist_camera"):
        width, height = result["scene"][camera]["resolution"]
        if resolution[0] * height != resolution[1] * width:
            raise ValueError("render_resolution must preserve each calibrated camera's native aspect ratio")
    return result


class PAPTask(ParallelTask):
    """PAP geometry, object reset sampling and measured grasp/place outcomes."""

    def __init__(self, config):
        self.config = cfg = load_config(config)
        self.identity = task_identity(cfg)
        semantics = {"task": self.identity["task_config_sha256"], "reset": RESET_CONTRACT,
                     "render_resolution": cfg["render_resolution"], "spacing_m": cfg["spacing_m"],
                     "renderer_profile": cfg["renderer_profile"],
                     "image_preprocessing": cfg["image_preprocessing"],
                     "renderer_settings": renderer_profile(cfg["renderer_profile"]),
                     "observation_render_frames": cfg["observation_render_frames"],
                     "render_subframes": cfg["render_subframes"], "reset_render_frames": cfg["reset_render_frames"]}
        self.identity["task_config_sha256"] = hashlib.sha256(json.dumps(semantics, sort_keys=True).encode()).hexdigest()
        self.identity["source_identity"].update(_source_identity(_SOURCE_FILES))
        self.identity["source_revision"] = _source_revision(self.identity["source_identity"])
        kin = PiperKinematics(cfg["scene"]["robot_urdf"], tip_offset=.1358)
        self.lower = np.r_[kin.lower, 0.]
        self.upper = np.r_[kin.upper, cfg["scene"]["gripper_max_width_m"]]
        self.initial = load_policy_initial(cfg["trajectory"], cfg["scene"]["gripper_max_width_m"])["initial"]
        if np.any(self.initial < self.lower-1e-6) or np.any(self.initial > self.upper+1e-6):
            raise ValueError("Initial recorded observation exceeds URDF arm/gripper limits")
        keys = ("num_envs", "gpu", "physics_dt", "control_hz", "horizon", "max_episode_seconds",
                "settling_ticks", "spacing_m", "render_resolution", "render_subframes",
                "observation_render_frames", "reset_render_frames", "cpu_threads", "png_workers", "gamma",
                "image_preprocessing")
        self.execution = {key: copy.deepcopy(cfg[key]) for key in keys}
        self.execution.update(renderer_profile="rl_fast" if cfg["renderer_profile"] == "rl_fast" else "calibrated",
                              log_dir=str(_path(cfg.get("log_dir", "outputs/pap_vector_workers"))),
                              warmup_frames=cfg["scene"].get("rendering", {}).get("warmup_frames", 8),
                              terminal_boundary="physics_tick", timeout_reason="timeout",
                              reset_contract=RESET_CONTRACT)

    def metadata(self):
        cfg = self.config
        return dict(schema_version=2, node_type="simulation", task_id="pap", num_envs=cfg["num_envs"],
            **copy.deepcopy(self.identity), smoothing=copy.deepcopy(cfg["smoothing"]),
            observation={"state": {"names": NAMES.copy(), "units": UNITS.copy()}, "cameras": ["up", "wrist"]},
            action={"representation": "joint_position", "frame": "robot_base", "names": NAMES.copy(),
                    "units": UNITS.copy(), "horizon": cfg["horizon"], "control_hz": cfg["control_hz"],
                    "lower": self.lower.tolist(), "upper": self.upper.tolist()},
            control_state_dim=7, task_features_dim=0,
            discount={"gamma": cfg["gamma"], "timebase_hz": cfg["control_hz"]}, reset_mode="indexed",
            execution={"smoothing": copy.deepcopy(cfg["smoothing"]), "reset_contract": RESET_CONTRACT,
                "camera_contract": "original calibrated 4:3 aperture/offset/pose; no crop or square-frustum substitution"},
            reward=copy.deepcopy(cfg["reward"]),
            reward_schema={"kind": "sparse_physical_event", "values": copy.deepcopy(cfg["reward"]),
                "within_chunk_discount": "gamma ** ((event_tick - 1) / physics_ticks_per_action)",
                "timeout_reward": 0., "success_requires_bilateral_contact_history": True},
            success_rule="lift + whole object in box + release + stable floor rest + bilateral contact history")

    def build_template(self):
        from .worker import _build
        source = _build(self.config, initialize=False)
        return SceneTemplate(world=source.world, stage=source.stage, robot_path="/World/Piper",
            camera_paths=source.camera_paths,
            data={"scene": copy.deepcopy(source.scene), "base": np.asarray(source.base).copy(),
                  "object_paths": list(source.object_paths), "material_paths": list(source.material_paths)})

    def create_robot(self, engine):
        from pressb.robots.piper.parallel import PiperJointDriver
        return PiperJointDriver(engine, scene_config=self.config["scene"], initial_state=self.initial,
                                action_dim=7, smoothing=self.config["smoothing"])

    def bind(self, engine):
        super().bind(engine)
        data = engine.template.data
        self.source_scene, self.source_base = data["scene"], np.asarray(data["base"])
        self.object_path = data["object_paths"][0]
        self.bodies, rows = engine.rigid_bodies([self.object_path])
        self.object_rows = np.asarray(rows)[:, 0]
        self.material_paths = [engine.relocate(i, data["material_paths"][0]) for i in range(engine.num_envs)]
        self.contact_counts = np.zeros((engine.num_envs, 2), dtype=np.int64)
        self.finger_contact = np.zeros(engine.num_envs, dtype=bool)
        self.outcomes = [None] * engine.num_envs

    def after_settle(self):
        self.settled_object_z = self.bodies.get_world_poses()[0][self.object_rows, 2].copy()

    def sample_reset(self, env_id, seed, options):
        return sample_reset(self.config, seed, env_id, options)

    def reset_task(self, env_ids, specs):
        from pxr import Gf, UsdShade
        geometry = self.config["objects"][0]["geometry"]
        cavity = self.source_scene["box_inside"]
        positions, quaternions, result = [], [], {}
        for i in env_ids:
            spec, offset = specs[i], self.engine.offsets[i]
            position = np.r_[np.asarray(spec["center_xy"])+self.source_base[:2]+offset[:2], self.settled_object_z[i]]
            angle = math.radians(spec["yaw_deg"])/2
            positions.append(position)
            quaternions.append([math.cos(angle), 0., 0., math.sin(angle)])
            for suffix, name in (("Mdl", "diffuse_color_constant"), ("Preview", "diffuseColor")):
                UsdShade.Shader(self.engine.stage.GetPrimAtPath(self.material_paths[i]+"/"+suffix)).GetInput(name).Set(
                    Gf.Vec3f(*spec["color_rgb"]))
            box = dict(center_world=(np.asarray(cavity["origin_world"])+offset).tolist(), yaw_deg=cavity["yaw_deg"],
                inner_size_xy=(np.asarray(cavity["max_local"])-cavity["min_local"])[:2].tolist(),
                floor_z_world=float(cavity["origin_world"][2]+cavity["min_local"][2]+offset[2]),
                rim_z_world=float(cavity["origin_world"][2]+cavity["max_local"][2]+offset[2]))
            self.outcomes[i] = PickPlaceOutcome(geometry, float(self.config["scene"]["table_height"]+offset[2]), box,
                                               **self.config["outcome"])
            self.contact_counts[i] = 0
            self.finger_contact[i] = False
            result[i] = dict(object_geometry=copy.deepcopy(geometry), object_position_world=position.tolist(),
                settling_physics_steps=0, startup_settling_physics_steps=self.config["settling_ticks"],
                reset_contract=RESET_CONTRACT, smoothing=self.engine.robot.postprocessors[i].summary())
        ids = np.asarray(env_ids)
        self.bodies.set_world_poses(np.asarray(positions), np.asarray(quaternions), self.object_rows[ids])
        self.bodies.set_velocities(np.zeros((len(ids), 6)), self.object_rows[ids])
        return result

    def before_tick(self, env_ids):
        self.finger_contact[:] = False

    def on_contact(self, env_id, actor_paths, contacts):
        if not len(contacts) or not self.engine.backend.ready[env_id] or self.object_path not in actor_paths:
            return
        for finger in (0, 1):
            if self.engine.template.robot_path + f"/link{7+finger}" in actor_paths:
                self.contact_counts[env_id, finger] += 1
                self.finger_contact[env_id] = True

    def after_tick(self, env_ids):
        positions, quaternions = self.bodies.get_world_poses()
        positions, quaternions = positions[self.object_rows], quaternions[self.object_rows]
        velocities = self.bodies.get_velocities()[self.object_rows]
        if not np.isfinite(np.r_[positions.ravel(), quaternions.ravel(), velocities.ravel()]).all():
            raise RuntimeError("Nonfinite PAP object physics state")
        cfg, result = self.config, {}
        states = self.engine.robot.observations(env_ids)
        for i in env_ids:
            outcome = self.outcomes[i].update(self.engine.backend.ticks[i]*cfg["physics_dt"], positions[i], quaternions[i],
                velocities[i], float(states[i]["state"][6]), finger_contact=bool(self.finger_contact[i]))
            success = bool(outcome["success"] and all(self.contact_counts[i]))
            dropped = cfg["terminate_on_drop"] and positions[i, 2] < cfg["scene"]["table_height"]+self.engine.offsets[i, 2]-cfg["drop_below_table_m"]
            reason = "success" if success else "object_dropped" if dropped else None
            result[i] = dict(success=success, terminated=reason is not None, truncated=False, reason=reason,
                reward=cfg["reward"]["success"] if success else cfg["reward"]["failure"] if dropped else 0.,
                info=dict(physical_outcome=copy.deepcopy(outcome),
                    contact_counts={"link7": int(self.contact_counts[i, 0]), "link8": int(self.contact_counts[i, 1])},
                    object_position_world=positions[i].tolist(), object_teleport_after_reset=False))
        return result

    def diagnostic_state(self, env_ids):
        positions, quaternions = self.bodies.get_world_poses()
        velocities = self.bodies.get_velocities()
        return {i: dict(object_position_world=positions[self.object_rows[i]].tolist(),
                        object_quaternion_wxyz=quaternions[self.object_rows[i]].tolist(),
                        object_velocity=velocities[self.object_rows[i]].tolist()) for i in env_ids}


def create_backend(config):
    from pressb.runtime.parallel import SharedWorldBackend
    return SharedWorldBackend(PAPTask(config))
