"""Single-world physical button task with the task-independent runtime contract.

The backend owns task randomization and physical outcome rules. Isaac only
supplies measured joints, cap travel and contact impulses. No target-distance
reward, scripted press action or model-specific unit conversion is used here.
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
from pressb.runtime.contract import (encode_image, finite_array, fingerprint,
                                     validate_actions, validate_metadata)
from .collision_policy import (
    DUAL_COLLISION_POLICY, STRICT_COLLISION_POLICY, collision_settings,
    is_allowed_finger_closure, strict_outcome,
)
from pressb.tasks.table_buttons.spacing import SPACING_PROFILES, resolve_spacing
from pressb.tasks.table_buttons.task import COLORS, TableButtonOutcome, button_in_camera, task_instruction


ROOT = Path(__file__).resolve().parents[4]
STATE_NAMES = [f"joint{i}" for i in range(1, 7)] + ["gripper_width"]
STATE_UNITS = ["rad"] * 6 + ["m"]
SOURCE_FILES = (
    "tasks/table_buttons/backend.py", "robots/piper/isaac.py", "sensors/camera.py", "runtime/scene.py", "runtime/workbench.py",
    "tasks/table_buttons/task.py", "tasks/table_buttons/scene.py", "tasks/table_buttons/placement.py",
    "tasks/table_buttons/spacing.py", "tasks/table_buttons/camera_domain.py", "tasks/table_buttons/collision.py",
    "tasks/table_buttons/workspace.py", "tasks/table_buttons/motion.py", "tasks/table_buttons/motion_templates.py",
    "tasks/table_buttons/target_relative_motion.py", "tasks/table_buttons/collision_policy.py",
    "tasks/table_buttons/collection.py", "robots/piper/wrist_camera.py", "robots/piper/kinematics.py", "control/joint_actions.py",
)


def _path(value):
    path = Path(value).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _merge(base, changes):
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def load_config(config):
    """Validate configuration without importing or starting Isaac."""
    allowed = {"scene_config", "scene_overrides", "num_envs", "action_dim", "fixed_gripper_width_m",
               "horizon", "control_hz", "physics_dt", "max_episode_seconds", "gamma", "gpu",
               "headless", "settling_ticks", "collision_policy", "randomization", "smoothing",
               "model_inputs"}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError(f"Table buttons configuration supports only {sorted(allowed)}")
    result = deepcopy(config)
    scene_path = _path(result.get("scene_config", "configs/piper_table_buttons.json"))
    scene = json.loads(scene_path.read_text())
    overrides = result.get("scene_overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("scene_overrides must be a mapping")
    _merge(scene, overrides)
    for key in ("robot_usd", "robot_urdf"):
        scene[key] = str(_path(scene[key]))
        if not Path(scene[key]).is_file():
            raise FileNotFoundError(scene[key])
    result["scene_config"], result["scene"] = str(scene_path), scene
    result["num_envs"] = _integer(result.get("num_envs", 1), "num_envs")
    if result["num_envs"] != 1:
        raise ValueError("Table buttons currently supports num_envs=1; no vectorized execution is advertised")
    result["action_dim"] = result.get("action_dim", 6)
    if type(result["action_dim"]) is not int or result["action_dim"] not in (6, 7):
        raise ValueError("action_dim must be 6 (fixed gripper) or 7 (controlled full width)")
    width = result.setdefault("fixed_gripper_width_m", scene["initial_gripper_width_m"])
    if (isinstance(width, bool) or not isinstance(width, (float, int)) or not math.isfinite(width)
            or not 0 <= width <= scene["gripper_max_width_m"]):
        raise ValueError("fixed_gripper_width_m must be within the physical gripper range")
    if result["action_dim"] == 6:
        scene["initial_gripper_width_m"] = width
    finite_array(scene["initial_joint_positions_rad"], (6,), "initial_joint_positions_rad")
    result["horizon"] = _integer(result.get("horizon", 7), "horizon")
    result["physics_dt"] = _positive(result.get("physics_dt", scene["physics_dt"]), "physics_dt")
    result["control_hz"] = _positive(result.get("control_hz", scene["fps"]), "control_hz")
    stride = 1 / result["physics_dt"] / result["control_hz"]
    if stride < 1 or not math.isclose(stride, round(stride), abs_tol=1e-9):
        raise ValueError("control_hz must divide the physics frequency")
    result["physics_ticks_per_action"] = round(stride)
    scene["physics_dt"], scene["fps"] = result["physics_dt"], result["control_hz"]
    seconds = result["max_episode_seconds"] = _positive(result.get("max_episode_seconds", 10.), "max_episode_seconds")
    intervals = seconds * result["control_hz"]
    if intervals < 1 or not math.isclose(intervals, round(intervals), abs_tol=1e-8):
        raise ValueError("max_episode_seconds must contain a whole number of control intervals")
    result["max_control_steps"] = round(intervals)
    result["gamma"] = _positive(result.get("gamma", .99), "gamma")
    if result["gamma"] > 1:
        raise ValueError("gamma must be in (0, 1]")
    result["gpu"] = _integer(result.get("gpu", 0), "gpu", 0)
    result["settling_ticks"] = _integer(result.get("settling_ticks", 24), "settling_ticks", 0)
    result["headless"] = result.get("headless", True)
    if type(result["headless"]) is not bool:
        raise ValueError("headless must be boolean")
    policy = result.get("collision_policy", "finger_closure_dual")
    if policy == "finger_closure_dual":
        policy = DUAL_COLLISION_POLICY
    elif policy == "collector_strict":
        policy = STRICT_COLLISION_POLICY
    result["collision_policy"] = collision_settings(policy)
    smoothing = result.get("smoothing", {"mode": "none", "alpha": .8})
    if not isinstance(smoothing, dict) or set(smoothing) - {"mode", "alpha"}:
        raise ValueError("smoothing supports mode and alpha")
    postprocessor = JointActionPostprocessor(**smoothing)
    result["smoothing"] = {"mode": postprocessor.mode, "alpha": postprocessor.alpha}
    randomization = result.setdefault("randomization", {})
    if not isinstance(randomization, dict) or set(randomization) - {"profiles", "colors", "positions"}:
        raise ValueError("randomization supports profiles, colors, positions")
    for key, choices in (("profiles", tuple(SPACING_PROFILES)), ("colors", COLORS)):
        values = randomization.setdefault(key, list(choices))
        if (not isinstance(values, list) or not values or any(v not in choices for v in values)
                or len(set(values)) != len(values)):
            raise ValueError(f"randomization.{key} must contain distinct values from {choices}")
    if type(randomization.setdefault("positions", True)) is not bool:
        raise ValueError("randomization.positions must be boolean")
    result.setdefault("model_inputs", {})
    if not isinstance(result["model_inputs"], dict):
        raise ValueError("model_inputs must be a mapping")
    # Keep outcome thresholds and all three profile bands validated even before reset.
    TableButtonOutcome("red", **scene["success"])
    for profile in randomization["profiles"]:
        resolve_spacing({**scene.get("position_sampling", {}), "spacing_profile": profile},
                        scene["button_geometry"]["base_radius_m"])
    return result


def task_identity(config):
    semantics = {k: deepcopy(config[k]) for k in (
        "action_dim", "fixed_gripper_width_m", "horizon", "control_hz", "physics_dt", "gamma",
        "max_episode_seconds", "settling_ticks", "collision_policy", "randomization", "smoothing", "model_inputs")}
    scene = deepcopy(config["scene"])
    assets = {}
    for key in ("robot_urdf", "robot_usd"):
        digest = hashlib.sha256()
        with Path(scene.pop(key)).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        assets[key] = digest.hexdigest()
    semantics.update(scene=scene, asset_sha256=assets, outcome_version="physical_press_strict_v1")
    sources = {}
    for name in SOURCE_FILES:
        content = (ROOT / "src" / "pressb" / name).read_bytes()
        sources[name] = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
    return {"task_config_sha256": fingerprint(semantics), "asset_sha256": assets,
            "source_revision": "sha256:" + fingerprint(sources), "source_identity": sources,
            "asset_scope": "URDF and root robot USD; external USD references are not recursively hashed"}


class TableButtonsBackend:
    """Real single-world task, usable directly or through SimulationService.

    ``environment_factory`` and ``layout_sampler`` permit CPU tests with measured
    telemetry fixtures. Production defaults always build the Isaac environment.
    """

    def __init__(self, config, *, environment_factory=None, layout_sampler=None):
        self.config = load_config(config)
        self.kin = PiperKinematics(self.config["scene"]["robot_urdf"], tip_offset=.1358)
        self.lower = np.r_[self.kin.lower, 0.]
        self.upper = np.r_[self.kin.upper, self.config["scene"]["gripper_max_width_m"]]
        self.identity = task_identity(self.config)
        self._environment_factory = environment_factory or IsaacTableButtonsEnvironment
        self._layout_sampler = layout_sampler
        self._placements = {}
        self.environment = None
        self._closed = False
        self._ready = False
        self._fault = None
        self.metadata = self.describe()

    def describe(self):
        cfg, width = self.config, self.config["action_dim"]
        result = dict(schema_version=2, node_type="simulation", task_id="table_buttons", num_envs=1,
            **self.identity,
            observation={"state": {"names": STATE_NAMES.copy(), "units": STATE_UNITS.copy()},
                         "cameras": ["up", "wrist"]},
            action={"representation": "joint_position", "frame": "robot_base",
                    "names": STATE_NAMES[:width], "units": STATE_UNITS[:width],
                    "horizon": cfg["horizon"], "control_hz": cfg["control_hz"],
                    "lower": self.lower[:width].tolist(), "upper": self.upper[:width].tolist()},
            control_state_dim=7, task_features_dim=0,
            discount={"gamma": cfg["gamma"], "timebase_hz": cfg["control_hz"]}, reset_mode="indexed",
            smoothing=deepcopy(cfg["smoothing"]),
            execution={"backend": "isaac_single_world", "workers": 1, "vectorized": False,
                       "gpu_ids": [cfg["gpu"]], "physics_hz": 1 / cfg["physics_dt"],
                       "max_episode_seconds": cfg["max_episode_seconds"],
                       "terminal_boundary": "full_control_interval"},
            reward_schema={"kind": "sparse_physical_press", "success": 1., "failure": 0., "timeout": 0.,
                           "within_chunk_discount": "gamma ** event_control_interval_index"},
            success_rule="TableButtonOutcome cap displacement + finger force + dwell; no wrong or blocking contact",
            collision_policy=deepcopy(cfg["collision_policy"]),
            randomization=deepcopy(cfg["randomization"]))
        if width == 6:
            result["fixed_gripper_width_m"] = cfg["fixed_gripper_width_m"]
        return validate_metadata(result, "simulation")

    def _check(self):
        if self._closed or self._fault:
            raise RuntimeError(f"Table buttons backend is closed or faulted: {self._fault}")

    def _sample(self, profile, seed):
        if self._layout_sampler is not None:
            return self._layout_sampler(profile, seed)
        from pressb.tasks.table_buttons.placement import ReachablePlacement
        if profile not in self._placements:
            cfg = deepcopy(self.config["scene"])
            cfg["position_sampling"]["spacing_profile"] = profile
            self._placements[profile] = ReachablePlacement(self.kin, cfg)
        return self._placements[profile].sample(seed)

    def _reset_spec(self, seed, options):
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not isinstance(options, dict):
            raise ValueError("reset options must be a mapping")
        options = deepcopy(options)
        per_env = options.pop("by_env", {})
        if not isinstance(per_env, dict) or set(per_env) - {0, "0"}:
            raise ValueError("options.by_env supports only environment 0")
        selected = per_env.get("0", per_env.get(0, {}))
        if not isinstance(selected, dict):
            raise ValueError("options.by_env.0 must be a mapping")
        options.update(selected)
        if set(options) - {"profile", "target_color", "positions", "model_inputs"}:
            raise ValueError("reset options supports profile, target_color, positions, model_inputs")
        rng = np.random.default_rng(seed)
        randomization = self.config["randomization"]
        profile = options.get("profile", str(rng.choice(randomization["profiles"])))
        color = options.get("target_color", str(rng.choice(randomization["colors"])))
        if profile not in SPACING_PROFILES or color not in COLORS:
            raise ValueError("Unknown spacing profile or target color")
        cfg = self.config["scene"]
        sample_seed = int(rng.integers(0, 2**32))
        sampled = None
        if "positions" in options:
            positions = options["positions"]
        elif randomization["positions"]:
            sampled = self._sample(profile, sample_seed)
            positions = sampled["positions"]
        else:
            positions = {name: cfg["buttons"][name]["center_xy"] for name in COLORS}
        if not isinstance(positions, dict) or set(positions) != set(COLORS):
            raise ValueError("positions must specify red, yellow and blue XY centres")
        positions = {name: finite_array(positions[name], (2,), f"positions.{name}", dtype=np.float64).tolist()
                     for name in COLORS}
        geometry = cfg["button_geometry"]
        band = resolve_spacing({**cfg["position_sampling"], "spacing_profile": profile}, geometry["base_radius_m"])
        # Fixed legacy scene coordinates may span several named bands. Do not
        # label them a sampled near/medium/far layout unless a band was requested.
        enforce_band = randomization["positions"] or "positions" in options or "profile" in options
        for i, name in enumerate(COLORS):
            x, y = positions[name]
            radius = geometry["base_radius_m"]
            if not (cfg["table_x_min"] + radius <= x <= cfg["table_x_min"] + cfg["table_depth"] - radius
                    and abs(y) <= cfg["table_width"] / 2 - radius):
                raise ValueError(f"{name} button must be fully supported by the tabletop")
            if not button_in_camera(positions[name], cfg["up_camera"],
                    button_radius=geometry["base_radius_m"],
                    button_top_z=geometry["base_height_m"] + geometry["cap_height_m"],
                    margin_px=cfg["position_sampling"].get("pixel_margin", 16.)):
                raise ValueError(f"{name} button must fit inside the configured camera")
            for other in COLORS[i+1:]:
                distance = float(np.linalg.norm(np.subtract(positions[name], positions[other])))
                if distance < 2 * geometry["base_radius_m"] or (enforce_band and not band.accepts(distance)):
                    raise ValueError("Button positions violate the selected spacing or housing separation")
        inputs = options.get("model_inputs", self.config["model_inputs"])
        if not isinstance(inputs, dict):
            raise ValueError("model_inputs must be a mapping")
        return {"env_id": 0, "seed": seed, "layout_seed": sample_seed,
                "profile": profile if enforce_band else None, "target_color": color,
                "positions": positions, "task": task_instruction(color), "model_inputs": deepcopy(inputs),
                "layout_sampling": "reachable_camera_domain" if sampled is not None else "specified_positions"}

    def _observe(self):
        state = finite_array(self.environment.state(), (7,), "measured joints", dtype=np.float64)
        images = self.environment.images()
        return {"state": state.tolist(), "images": {name: encode_image(images[name]) for name in ("up", "wrist")},
                "task": self.spec["task"], "model_inputs": deepcopy(self.spec["model_inputs"]),
                "control_state": self.last_sent.tolist(), "task_features": []}

    def reset(self, env_ids, seed, options=None):
        self._check()
        if env_ids != [0] or any(type(value) is not int for value in env_ids):
            raise ValueError("Table buttons reset requires env_ids=[0]")
        spec = self._reset_spec(seed, {} if options is None else options)
        try:
            if self.environment is None:
                self.environment = self._environment_factory(self.config)
            self.environment.reset(spec)
            self.spec = spec
            self.checker = TableButtonOutcome(spec["target_color"], **self.config["scene"]["success"])
            self.postprocessor = JointActionPostprocessor(**self.config["smoothing"])
            self.last_sent = finite_array(self.environment.state(), (7,), "initial measured joints", dtype=np.float64)
            self.postprocessor.reset(self.last_sent)
            self.ticks = self.control_steps = 0
            self.wrong_contact = False
            self.first_wrong_press = None
            self.blocking_contacts = []
            self._ready = True
            observation = self._observe()
        except Exception as error:
            self._fault = f"reset: {type(error).__name__}: {error}"
            raise
        return [{"env_id": 0, "observation": observation, "info": {
            **deepcopy(spec), "success": False, "settling_physics_steps": self.config["settling_ticks"],
            "elapsed_control_steps": 0}}]

    def step(self, items):
        self._check()
        if not self._ready:
            raise ValueError("Reset must precede step; terminal episodes do not automatically reset")
        if (not isinstance(items, list) or len(items) != 1 or items[0].get("env_id") != 0
                or type(items[0].get("env_id")) is not int):
            raise ValueError("Table buttons step requires exactly environment 0")
        actions = validate_actions(items[0].get("actions"), self.metadata["action"], allow_prefix=True)
        try:
            return [self._execute(actions)]
        except Exception as error:
            self._fault = f"step: {type(error).__name__}: {error}"
            raise

    def _execute(self, actions):
        cfg = self.config
        before = self.ticks
        result, sent = {}, []
        success = terminated = truncated = False
        reason = None
        for action in actions:
            command = (np.r_[action.astype(np.float64), cfg["fixed_gripper_width_m"]]
                       if cfg["action_dim"] == 6 else action)
            self.last_sent = self.postprocessor.apply(command)
            sent.append(self.last_sent.tolist())
            self.environment.apply_action(self.last_sent)
            # Inspect all ticks in the interval so a later wrong contact cannot
            # be hidden by the outcome checker's earlier success latch.
            for _ in range(cfg["physics_ticks_per_action"]):
                sample = self.environment.tick()
                self.ticks += 1
                self.blocking_contacts.extend(sample.get("blocking_contacts", []))
                self.wrong_contact |= any(sample["contact_forces"][name] > .02
                                          for name in COLORS if name != self.spec["target_color"])
                result = self.checker.update(self.ticks * cfg["physics_dt"], sample["travels"], sample["contact_forces"])
                wrong = [name for name in result["press_events"] if name != self.spec["target_color"]]
                if wrong and self.first_wrong_press is None:
                    self.first_wrong_press = {"color": wrong[0], "time_s": self.ticks * cfg["physics_dt"]}
            self.control_steps += 1
            success, reason = strict_outcome(result, wrong_press=self.first_wrong_press,
                wrong_contact=self.wrong_contact, unexpected_contacts=self.blocking_contacts, model_error=None)
            terminated = bool(result["terminated"] or self.wrong_contact or self.blocking_contacts)
            truncated = bool(not terminated and self.control_steps >= cfg["max_control_steps"])
            if terminated or truncated:
                break
        elapsed = (self.ticks - before) / cfg["physics_ticks_per_action"]
        self._ready = not (terminated or truncated)
        # TableButtonOutcome supplies the physical reward. Strict contact failure
        # can veto it; a distance or an arm command can never create a success.
        reward = float(result["reward"] if success else 0.) * cfg["gamma"] ** (elapsed - 1)
        return {"env_id": 0, "observation": self._observe(), "reward": reward,
                "discount": cfg["gamma"] ** elapsed, "terminated": terminated, "truncated": truncated,
                "info": {"success": bool(success), "termination_reason": reason if terminated or truncated else None,
                         "physics_steps": self.ticks - before, "elapsed_control_steps": elapsed,
                         "episode_physics_steps": self.ticks, "episode_seconds": self.ticks * cfg["physics_dt"],
                         "physical_outcome": result, "wrong_contact": bool(self.wrong_contact),
                         "first_wrong_press": self.first_wrong_press,
                         "blocking_contact_count": len(self.blocking_contacts), "applied_actions_rad_m": sent,
                         "requested_action_count": len(actions), "target_color": self.spec["target_color"],
                         "spacing_profile": self.spec["profile"], "button_teleport_after_reset": False}}

    def close(self):
        if not self._closed:
            self._closed = True
            if self.environment is not None:
                self.environment.close()


class IsaacTableButtonsEnvironment:
    """Isaac sensor/actuator boundary, with the legacy collector's exact setup."""

    def __init__(self, config):
        self.config = config
        self.runtime = self.subscription = self.app = None

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
        from isaacsim.core.prims import SingleRigidPrim
        from isaacsim.core.utils.stage import create_new_stage
        from pressb.robots.piper.isaac import build_piper_runtime, close_piper_runtime
        from pressb.runtime.workbench import build_tripod
        from pressb.tasks.table_buttons.scene import build_scene
        import omni.physx
        if self.runtime is not None:
            self.subscription = None
            close_piper_runtime(self.runtime)
            self.runtime = None
            World.clear_instance()
        create_new_stage()
        scene_cfg = deepcopy(self.config["scene"])
        scene_cfg.update(task_color=spec["target_color"], task_instruction=spec["task"])
        for name in COLORS:
            scene_cfg["buttons"][name]["center_xy"] = spec["positions"][name]
        self.runtime = build_piper_runtime(scene_cfg, build_scene, asset_root=ROOT,
                                          camera_hardware_builder=build_tripod)
        self.caps = {name: SingleRigidPrim(prim_path=value["body_path"], name=f"cap_{name}")
                     for name, value in self.runtime.scene["buttons"].items()}
        for cap in self.caps.values():
            cap.initialize()
        self.contact_forces = {name: 0. for name in COLORS}
        self.blocking_contacts = []
        self.subscription = omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(self._on_contact)
        for _ in range(self.config["settling_ticks"]):
            self.runtime.world.step(render=False)
        for _ in range(self.runtime.warmup_frames):
            self.runtime.world.render()
        self.contact_forces = {name: 0. for name in COLORS}
        self.blocking_contacts = []

    def _on_contact(self, headers, data):
        from pxr import PhysicsSchemaTools
        buttons = self.runtime.scene["buttons"]
        for header in headers:
            if not header.num_contact_data:
                continue
            actors = [str(PhysicsSchemaTools.intToSdfPath(value)) for value in (header.actor0, header.actor1)]
            force = sum(float(np.linalg.norm(data[i].impulse)) for i in
                        range(header.contact_data_offset, header.contact_data_offset + header.num_contact_data)) / self.config["physics_dt"]
            fingers = [name for name in ("link7", "link8") if f"/World/Piper/{name}" in actors]
            for name, button in buttons.items():
                if button["body_path"] in actors and fingers:
                    self.contact_forces[name] += force
            moving_robot = [actor for actor in actors if actor.startswith("/World/Piper/") and actor != "/World/Piper/base_link"]
            self_contact = all(actor.startswith("/World/Piper/") for actor in actors) and actors[0] != actors[1]
            environment_contact = bool(moving_robot) and any(not actor.startswith("/World/Piper/") and
                not any(button["body_path"] == actor for button in buttons.values()) for actor in actors)
            non_finger_cap = not fingers and any(button["body_path"] in actors for button in buttons.values())
            if force > .02 and moving_robot and (self_contact or environment_contact or non_finger_cap):
                if not is_allowed_finger_closure(actors, self.config["collision_policy"]):
                    self.blocking_contacts.append({"actors": actors, "force_n": force})

    def state(self):
        from pressb.tasks.table_buttons.collection import measured_state
        return measured_state(self.runtime.robot.get_joint_positions()[self.runtime.indices])

    def images(self):
        for _ in range(self.runtime.subframes):
            self.runtime.world.render()
        return {name: np.ascontiguousarray(camera.get_rgba()[..., :3])
                for name, camera in self.runtime.cameras.items()}

    def apply_action(self, command):
        from isaacsim.core.utils.types import ArticulationAction
        self.runtime.controller.apply_action(ArticulationAction(joint_positions=self.runtime.full_command(command)))

    def tick(self):
        self.contact_forces = {name: 0. for name in COLORS}
        self.blocking_contacts = []
        self.runtime.world.step(render=False)
        return {"travels": {name: float(self.runtime.scene["buttons"][name]["cap_rest_world"][2] -
                                       self.caps[name].get_world_pose()[0][2]) for name in COLORS},
                "contact_forces": dict(self.contact_forces), "blocking_contacts": list(self.blocking_contacts)}

    def close(self):
        try:
            self.subscription = None
            if self.runtime is not None:
                from pressb.robots.piper.isaac import close_piper_runtime
                close_piper_runtime(self.runtime)
                self.runtime = None
        finally:
            if self.app is not None:
                self.app.close()
                self.app = None


def create_backend(config):
    return TableButtonsBackend(config)
