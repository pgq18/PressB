"""Workspace configuration and legacy single-slot telemetry implementation.

This plugin exposes measured robot state, camera images and configurable action
execution. It intentionally has no task-success detector: reward is always zero,
and an episode ends only at the configured time limit or external retirement.
The default factory delegates to the shared parallel runtime.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from pressb.control.joint_actions import JointActionPostprocessor
from pressb.robots.piper.kinematics import PiperKinematics
from pressb.runtime.contract import encode_image, finite_array, fingerprint, validate_actions, validate_metadata

ROOT = Path(__file__).resolve().parents[4]
STATE_NAMES = [f"joint{i}" for i in range(1, 7)] + ["gripper_width"]
STATE_UNITS = ["rad"] * 6 + ["m"]
SOURCE_FILES = ("tasks/playground/backend.py", "tasks/playground/parallel.py", "tasks/playground/scene.py", "runtime/workbench.py",
                "runtime/scene.py", "robots/piper/isaac.py", "robots/piper/wrist_camera.py",
                "robots/piper/kinematics.py", "sensors/camera.py", "sensors/color_response.py", "control/joint_actions.py")


def _path(value):
    path = Path(value).expanduser()
    return (ROOT/path).resolve() if not path.is_absolute() else path.resolve()


def _merge(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = deepcopy(value)


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def load_config(config):
    """Resolve configuration without importing USD, CUDA or Isaac Sim."""
    allowed = {"scene_config", "scene_overrides", "num_envs", "horizon", "control_hz", "physics_dt",
               "max_episode_seconds", "gamma", "gpu", "headless", "settling_ticks", "smoothing", "model_inputs", "spacing_m", "render_resolution", "renderer_profile", "render_subframes",
               "observation_render_frames", "reset_render_frames", "cpu_threads", "png_workers", "color_response_workers", "log_dir",
               "image_preprocessing", "image_output_resolution"}
    if not isinstance(config, dict) or set(config)-allowed:
        raise ValueError(f"Playground configuration supports only {sorted(allowed)}")
    result = deepcopy(config)
    path = _path(result.get("scene_config", "configs/piper_playground.json"))
    scene = json.loads(path.read_text())
    if not isinstance(scene, dict):
        raise ValueError("scene_config must contain an object")
    overrides = result.get("scene_overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("scene_overrides must be a mapping")
    _merge(scene, overrides)
    if "camera_color_response" in scene:
        from pressb.sensors.color_response import normalize_camera_responses
        scene["camera_color_response"] = normalize_camera_responses(scene["camera_color_response"], ("up", "wrist"))
    for key in ("robot_usd", "robot_urdf"):
        scene[key] = str(_path(scene[key]))
        if not Path(scene[key]).is_file():
            raise FileNotFoundError(scene[key])
    if "initial_state_joint_rad_m" in scene:
        initial = finite_array(scene["initial_state_joint_rad_m"], (7,), "initial_state_joint_rad_m", dtype=np.float64)
        scene["initial_joint_positions_rad"] = initial[:6].tolist()
        scene["initial_gripper_width_m"] = float(initial[6])
    else:
        initial = finite_array([*scene["initial_joint_positions_rad"], scene["initial_gripper_width_m"]],
                               (7,), "initial_state_joint_rad_m", dtype=np.float64)
    kin = PiperKinematics(scene["robot_urdf"], tip_offset=.1358)
    maximum_width = _number(scene["gripper_max_width_m"], "gripper_max_width_m")
    if np.any(initial < np.r_[kin.lower, 0.]) or np.any(initial > np.r_[kin.upper, maximum_width]):
        raise ValueError("Initial state exceeds robot joint/gripper limits")
    scene["initial_state_joint_rad_m"] = initial.tolist()
    scene.setdefault("root_path", "/World/Playground")
    scene.setdefault("objects", [])
    scene.setdefault("static_props", [])
    result.update(scene_config=str(path), scene=scene)
    result["num_envs"] = _integer(result.get("num_envs", 1), "num_envs")
    result["horizon"] = _integer(result.get("horizon", 32), "horizon")
    result["physics_dt"] = _number(result.get("physics_dt", scene["physics_dt"]), "physics_dt")
    result["control_hz"] = _number(result.get("control_hz", scene["fps"]), "control_hz")
    stride = 1 / result["physics_dt"] / result["control_hz"]
    if stride < 1 or not math.isclose(stride, round(stride), abs_tol=1e-9):
        raise ValueError("control_hz must divide the physics frequency")
    result["physics_ticks_per_action"] = round(stride)
    scene["physics_dt"], scene["fps"] = result["physics_dt"], result["control_hz"]
    result["max_episode_seconds"] = _number(result.get("max_episode_seconds", 30.), "max_episode_seconds")
    steps = result["max_episode_seconds"] * result["control_hz"]
    if steps < 1 or not math.isclose(steps, round(steps), abs_tol=1e-8):
        raise ValueError("max_episode_seconds must contain whole control intervals")
    result["max_control_steps"] = round(steps)
    result["gamma"] = _number(result.get("gamma", .99), "gamma")
    if result["gamma"] > 1:
        raise ValueError("gamma must be in (0, 1]")
    result["gpu"] = _integer(result.get("gpu", 0), "gpu", 0)
    result["settling_ticks"] = _integer(result.get("settling_ticks", 24), "settling_ticks", 0)
    result["headless"] = result.get("headless", True)
    if type(result["headless"]) is not bool:
        raise ValueError("headless must be boolean")
    smoothing = result.get("smoothing", {"mode": "none", "alpha": .8})
    if not isinstance(smoothing, dict) or set(smoothing)-{"mode", "alpha"}:
        raise ValueError("smoothing supports mode and alpha")
    post = JointActionPostprocessor(**smoothing)
    result["smoothing"] = {"mode": post.mode, "alpha": post.alpha}
    result.setdefault("model_inputs", {})
    if not isinstance(result["model_inputs"], dict):
        raise ValueError("model_inputs must be a mapping")
    from pressb.runtime.parallel import execution_config
    execution_keys = ("num_envs", "gpu", "horizon", "physics_dt", "control_hz", "max_episode_seconds",
        "settling_ticks", "gamma", "headless", "spacing_m", "render_resolution", "renderer_profile",
        "render_subframes", "observation_render_frames", "reset_render_frames", "cpu_threads",
        "png_workers", "color_response_workers", "log_dir", "image_preprocessing", "image_output_resolution")
    execution = execution_config({**{key: result[key] for key in execution_keys if key in result},
        "render_resolution": result.get("render_resolution", scene["up_camera"]["resolution"]),
        "reset_render_frames": result.get("reset_render_frames", scene.get("rendering", {}).get("warmup_frames", 8)),
        "terminal_boundary": "control_interval"})
    # The renderer's aspect controls the sensor view; output resizing does not.
    render_width, render_height = execution["render_resolution"]
    camera_resolutions = {"up": scene["up_camera"]["resolution"],
        "wrist": scene.get("wrist_camera", {}).get("resolution", scene["wrist_camera_resolution"])}
    for name, size in camera_resolutions.items():
        if not isinstance(size, (list, tuple)) or len(size) != 2 or any(type(v) is not int or v <= 0 for v in size):
            raise ValueError(f"{name} camera resolution must contain two positive integers")
        if render_width * size[1] != render_height * size[0]:
            raise ValueError(f"render_resolution must preserve the {name} camera aspect ratio {size[0]}:{size[1]}; use image_output_resolution for transport resizing")
    if execution["renderer_profile"] not in ("calibrated", "rl_fast"):
        raise ValueError("renderer_profile must be calibrated or rl_fast")
    if execution["image_preprocessing"] is not None:
        from pressb.runtime.image_transport import validate_image_preprocessing
        validate_image_preprocessing(execution["image_preprocessing"])
    result.update({key: execution[key] for key in execution_keys if key in execution})
    return result


def task_identity(config):
    scene = deepcopy(config["scene"])
    assets = {key: hashlib.sha256(Path(scene.pop(key)).read_bytes()).hexdigest()
              for key in ("robot_usd", "robot_urdf")}
    sources = {name: hashlib.sha256((ROOT/"src"/"pressb"/name).read_bytes()).hexdigest() for name in SOURCE_FILES}
    semantics = {key: deepcopy(config[key]) for key in ("horizon", "control_hz", "physics_dt",
        "max_episode_seconds", "gamma", "settling_ticks", "smoothing", "model_inputs")}
    semantics.update(scene=scene, asset_sha256=assets, outcome_version="workspace_no_task_reward_v1")
    return {"task_config_sha256": fingerprint(semantics), "asset_sha256": assets,
            "source_revision": "sha256:"+fingerprint(sources), "source_identity": sources,
            "asset_scope": "URDF and root robot USD; external USD references are not recursively hashed"}


def sample_reset_spec(self, seed, options):
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(options, dict) or set(options)-{"task", "initial_state_joint_rad_m", "model_inputs"}:
        raise ValueError("Playground reset options support task, initial_state_joint_rad_m, model_inputs")
    initial = finite_array(options.get("initial_state_joint_rad_m", self.config["scene"]["initial_state_joint_rad_m"]),
                           (7,), "initial_state_joint_rad_m", dtype=np.float64)
    if np.any(initial < self.lower) or np.any(initial > self.upper):
        raise ValueError("Initial state exceeds robot joint/gripper limits")
    task = options.get("task", self.config["scene"].get("task_instruction", "Explore the tabletop workspace."))
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be a nonempty string")
    model_inputs = options.get("model_inputs", self.config["model_inputs"])
    if not isinstance(model_inputs, dict):
        raise ValueError("model_inputs must be a mapping")
    return {"env_id": 0, "seed": seed, "task": task, "initial_state_joint_rad_m": initial.tolist(),
            "model_inputs": deepcopy(model_inputs)}


class PlaygroundBackend:
    def __init__(self, config, *, environment_factory=None):
        self.config = load_config(config)
        if self.config["num_envs"] != 1:
            raise ValueError("Legacy backend only supports num_envs=1; use create_backend for shared parallel execution")
        kin = PiperKinematics(self.config["scene"]["robot_urdf"], tip_offset=.1358)
        self.lower = np.r_[kin.lower, 0.]
        self.upper = np.r_[kin.upper, self.config["scene"]["gripper_max_width_m"]]
        self.identity = task_identity(self.config)
        self._environment_factory = environment_factory or IsaacPlaygroundEnvironment
        self.environment = None
        self._closed, self._ready, self._fault = False, False, None
        self.metadata = self.describe()

    def describe(self):
        cfg = self.config
        return validate_metadata(dict(schema_version=2, node_type="simulation", task_id="playground",
            num_envs=1, **self.identity,
            observation={"state": {"names": STATE_NAMES.copy(), "units": STATE_UNITS.copy()}, "cameras": ["up", "wrist"]},
            action={"representation": "joint_position", "frame": "robot_base", "names": STATE_NAMES.copy(),
                    "units": STATE_UNITS.copy(), "horizon": cfg["horizon"], "control_hz": cfg["control_hz"],
                    "lower": self.lower.tolist(), "upper": self.upper.tolist()},
            control_state_dim=7, task_features_dim=0, reset_mode="indexed",
            discount={"gamma": cfg["gamma"], "timebase_hz": cfg["control_hz"]}, smoothing=deepcopy(cfg["smoothing"]),
            execution={"backend": "isaac_single_world", "workers": 1, "vectorized": False, "gpu_ids": [cfg["gpu"]],
                       "physics_hz": 1/cfg["physics_dt"], "max_episode_seconds": cfg["max_episode_seconds"],
                       "terminal_boundary": "full_control_interval"},
            reward_schema={"kind": "none", "value": 0.},
            success_rule="No automatic task outcome: workspace twin only; time limit truncates episodes."), "simulation")

    def _check(self):
        if self._closed or self._fault:
            raise RuntimeError(f"Playground backend is closed or faulted: {self._fault}")

    def _reset_spec(self, seed, options):
        return sample_reset_spec(self, seed, options)

    def _observe(self):
        state = finite_array(self.environment.state(), (7,), "measured joints", dtype=np.float64)
        images = self.environment.images()
        return {"state": state.tolist(), "images": {name: encode_image(images[name]) for name in ("up", "wrist")},
                "task": self.spec["task"], "model_inputs": deepcopy(self.spec["model_inputs"]),
                "control_state": self.last_sent.tolist(), "task_features": []}

    def reset(self, env_ids, seed, options=None):
        self._check()
        if not isinstance(env_ids, list) or env_ids != [0] or type(env_ids[0]) is not int:
            raise ValueError("Playground reset requires env_ids=[0]")
        spec = self._reset_spec(seed, {} if options is None else options)
        try:
            if self.environment is None:
                self.environment = self._environment_factory(self.config)
            self.environment.reset(spec)
            self.spec = spec
            self.last_sent = finite_array(self.environment.state(), (7,), "initial measured joints", dtype=np.float64)
            self.postprocessor = JointActionPostprocessor(**self.config["smoothing"])
            self.postprocessor.reset(self.last_sent)
            self.ticks = self.control_steps = 0
            self._ready = True
            observation = self._observe()
        except Exception as error:
            self._fault = f"reset: {type(error).__name__}: {error}"
            raise
        return [{"env_id": 0, "observation": observation,
                 "info": {**deepcopy(spec), "success": False, "task_outcome_available": False,
                          "settling_physics_steps": self.config["settling_ticks"], "elapsed_control_steps": 0}}]

    def end_episode(self, env_ids):
        self._check()
        if not isinstance(env_ids, list) or env_ids != [0] or type(env_ids[0]) is not int:
            raise ValueError("Playground end_episode requires env_ids=[0]")
        if not self._ready:
            raise ValueError("Only an active playground episode may end")
        self._ready = False

    def step(self, items):
        self._check()
        if not self._ready:
            raise ValueError("Reset must precede step; terminal episodes do not automatically reset")
        if (not isinstance(items, list) or len(items) != 1 or not isinstance(items[0], dict)
                or type(items[0].get("env_id")) is not int or items[0]["env_id"] != 0):
            raise ValueError("Playground step requires exactly environment 0")
        actions = validate_actions(items[0].get("actions"), self.metadata["action"], allow_prefix=True)
        try:
            cfg, before, sent = self.config, self.ticks, []
            truncated = False
            for action in actions:
                self.last_sent = self.postprocessor.apply(action)
                sent.append(self.last_sent.tolist())
                self.environment.apply_action(self.last_sent)
                for _ in range(cfg["physics_ticks_per_action"]):
                    self.environment.tick()
                    self.ticks += 1
                self.control_steps += 1
                truncated = self.control_steps >= cfg["max_control_steps"]
                if truncated:
                    break
            elapsed = (self.ticks-before)/cfg["physics_ticks_per_action"]
            self._ready = not truncated
            return [{"env_id": 0, "observation": self._observe(), "reward": 0.,
                     "discount": cfg["gamma"]**elapsed, "terminated": False, "truncated": truncated,
                     "info": {"success": False, "task_outcome_available": False,
                              "termination_reason": "time_limit" if truncated else None,
                              "physics_steps": self.ticks-before, "elapsed_control_steps": elapsed,
                              "episode_physics_steps": self.ticks, "episode_seconds": self.ticks*cfg["physics_dt"],
                              "applied_actions_rad_m": sent, "requested_action_count": len(actions)}}]
        except Exception as error:
            self._fault = f"step: {type(error).__name__}: {error}"
            raise

    def close(self):
        if not self._closed:
            if self.environment is not None:
                self.environment.close()
            self._closed = True


class IsaacPlaygroundEnvironment:
    """Lazy Isaac boundary; all articulation and camera construction is shared."""
    def __init__(self, config):
        self.config = config
        self.runtime = self.app = None

    def reset(self, spec):
        if self.app is None:
            from isaacsim import SimulationApp
            self.app = SimulationApp(dict(headless=self.config["headless"], active_gpu=self.config["gpu"],
                physics_gpu=self.config["gpu"], multi_gpu=False, renderer="RayTracedLighting",
                anti_aliasing=1, fast_shutdown=True, limit_cpu_threads=8, create_new_stage=False,
                extra_args=["--/app/asyncRendering=false", "--/rtx/ecoMode/enabled=false",
                    "--/validate/p2p/enabled=false", "--/validate/iommu/enabled=false",
                    "--/plugins/carb.tasking.plugin/threadCount=8", "--/plugins/omni.tbb.globalcontrol/maxThreadCount=8"]))
        from isaacsim.core.api import World
        from isaacsim.core.utils.stage import create_new_stage
        from pressb.robots.piper.isaac import build_piper_runtime, close_piper_runtime
        from pressb.runtime.workbench import build_tripod
        from .scene import build_scene
        if self.runtime is not None:
            close_piper_runtime(self.runtime)
            self.runtime = None
            World.clear_instance()
        create_new_stage()
        cfg = deepcopy(self.config["scene"])
        cfg["initial_joint_positions_rad"] = spec["initial_state_joint_rad_m"][:6]
        cfg["initial_gripper_width_m"] = spec["initial_state_joint_rad_m"][6]
        self.runtime = build_piper_runtime(cfg, build_scene, asset_root=ROOT,
            camera_hardware_builder=build_tripod if cfg.get("camera_tripod", False) else None)
        for _ in range(self.config["settling_ticks"]):
            self.runtime.world.step(render=False)
        for _ in range(self.runtime.warmup_frames):
            self.runtime.world.render()

    def state(self):
        values = self.runtime.robot.get_joint_positions()[self.runtime.indices]
        return np.r_[values[:6], values[6]-values[7]]

    def images(self):
        # Tensor articulation setters and non-rendering physics ticks do not
        # reliably publish link transforms to USD. Keep rendered wrist pose
        # synchronized with the measured state without advancing simulation.
        import omni.physx
        self.runtime.world.physics_sim_view.update_articulations_kinematic()
        omni.physx.get_physx_interface().update_transformations(False, True, True)
        for _ in range(self.runtime.subframes):
            self.runtime.world.render()
        return {name: np.ascontiguousarray(camera.get_rgba()[..., :3])
                for name, camera in self.runtime.cameras.items()}

    def apply_action(self, command):
        from isaacsim.core.utils.types import ArticulationAction
        self.runtime.controller.apply_action(ArticulationAction(joint_positions=self.runtime.full_command(command)))

    def tick(self):
        self.runtime.world.step(render=False)

    def close(self):
        try:
            if self.runtime is not None:
                from pressb.robots.piper.isaac import close_piper_runtime
                close_piper_runtime(self.runtime)
                self.runtime = None
        finally:
            if self.app is not None:
                self.app.close()
                self.app = None


def create_backend(config):
    from .parallel import create_backend as create_shared_backend
    return create_shared_backend(config)
