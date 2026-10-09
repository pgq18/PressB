"""Long-lived vector simulation with explicit, non-autoresetting RL transitions.

The transport/state machine is CPU-only. Isaac imports happen only when an
IsaacVectorBackend is constructed after SimulationApp on the serving thread.
"""
from __future__ import annotations

import copy
from collections import OrderedDict, deque
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import threading
import uuid

import numpy as np

from pressb.motion_smoothing import JointCommandSmoother, smoothing_settings
from pressb.policy_eval import PolicyPoseController
from pressb.replay_control import action_pose


CONTROL_STATE_SCHEMA = [
    {"name": "actual_joint_angles_rad", "size": 6},
    {"name": "actual_joint_velocities_rad_s", "size": 6},
    {"name": "preceding_ik_endpoint_rad", "size": 6},
    {"name": "linear_command_history_oldest_first_rad", "size": "6 * smoothing_window"},
    {"name": "elapsed_fraction", "size": 1},
]


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def control_state(actual_q, actual_velocity, ik_endpoint, linear_history, elapsed, maximum):
    """Expose the persistent actuator state without scene goals or coordinates."""
    vectors = [np.asarray(value, dtype=np.float64) for value in (actual_q, actual_velocity, ik_endpoint)]
    history = np.asarray(linear_history, dtype=np.float64)
    if any(value.shape != (6,) for value in vectors) or history.ndim != 2 or history.shape[1] != 6:
        raise ValueError("Control state requires three six-joint vectors and six-joint history")
    if not math.isfinite(maximum) or maximum <= 0 or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("Invalid control state episode time")
    result = np.r_[*vectors, history.ravel(), elapsed / maximum]
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite control state")
    return result.tolist()


def validate_actions_pose8(value):
    """Validate the entire seven-action chunk before any environment advances."""
    try:
        actions = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("actions_pose8 must contain seven finite eight-dimensional poses") from error
    if actions.shape != (7, 8) or not np.isfinite(actions).all():
        raise ValueError("actions_pose8 must have shape [7,8] and contain finite values")
    if not np.allclose(actions[:, 7], .008, atol=1e-8, rtol=0):
        raise ValueError("The commanded gripper width must remain 0.008 m")
    if np.any(np.linalg.norm(actions[:, :3], axis=1) > 1.5):
        raise ValueError("Policy target is more than 1.5 m from base_link")
    for action in actions:
        action_pose(action)
    return actions.copy()


class SimulationService:
    """Own HTTP sequencing, idempotency, vector barriers and RL reward semantics.

    The injected backend has health(), reset(episodes, seed), and step(actions),
    all called synchronously. step returns one item for each supplied env_id.
    Only a small number of image-bearing responses are cached; request digests
    remain for the session so an evicted ID can never execute a second mutation.
    """

    def __init__(self, backend, *, single_gamma=.99, response_cache_size=2):
        self.backend = backend
        self.single_gamma = _finite(single_gamma, "single_gamma")
        if not 0 < self.single_gamma <= 1:
            raise ValueError("single_gamma must be in (0,1]")
        self.response_cache_size = _integer(response_cache_size, "response_cache_size", 1)
        self.settings = copy.deepcopy(backend.health())
        self.num_envs = _integer(self.settings["num_envs"], "num_envs", 1)
        self.window = _integer(self.settings["smoothing_window"], "smoothing_window", 1)
        self.run_id = None
        self.cohort_id = None
        self.step_id = 0
        self.items = None
        self.fault = None
        self._signatures = {}
        self._responses = OrderedDict()
        self._closed_runs = set()

    def health(self, payload=None):
        result = copy.deepcopy(self.settings)
        cfg = self.settings["config"]
        randomization = cfg.get("panel_randomization", {})
        panel_bounds = {}
        for axis in "xy":
            for which in ("min", "max"):
                key = f"{which}_offset_{axis}_m"
                panel_bounds[key] = float(randomization[key] if randomization.get("enabled") is True
                    else cfg.get(f"panel_offset_{axis}_m", 0.))
        result.update(service="simulation", ready=self.fault is None, fault=self.fault,
            protocol_version=1, num_envs=self.num_envs, physics_hz=120, action_fps=30,
            action_horizon=7, smoothing_window=self.window, control_state_dim=19 + 6 * self.window,
            control_state_schema=copy.deepcopy(CONTROL_STATE_SCHEMA), single_gamma=self.single_gamma,
            run_id=self.run_id, cohort_id=self.cohort_id, step_id=self.step_id,
            active_env_ids=self._active(), all_done=self.items is not None and not self._active(),
            reset_mode="whole_vector_barrier_no_autoreset", response_cache_size=self.response_cache_size,
            panel_bounds=panel_bounds,
            reward_contract={"kind": "sparse_physical_success", "success": 1., "other": 0.,
                "per_action_gamma": self.single_gamma,
                "reward": "success * single_gamma ** ((executed_physics_steps - 1) / 4)",
                "discount": "single_gamma ** (executed_physics_steps / 4)",
                "timeout_bootstrap": "learner_decision_default_false"})
        return result

    def _active(self):
        return [] if self.items is None else [i for i, item in enumerate(self.items)
                                               if not item["terminated"] and not item["truncated"]]

    def _request(self, operation, payload):
        if not isinstance(payload, dict):
            raise ValueError("Request must be an object")
        for name in ("run_id", "request_id"):
            value = payload.get(name)
            if not isinstance(value, str) or not value.strip() or len(value) > 256:
                raise ValueError(f"{name} must be a nonempty string of at most 256 characters")
        if self.run_id is not None and payload["run_id"] != self.run_id:
            raise ValueError("A different run_id requires restarting the simulation service")
        if payload["run_id"] in self._closed_runs and operation != "close_run":
            raise ValueError("This run_id has been closed and cannot be reused")
        if self.fault is not None:
            raise RuntimeError(f"Simulation service is faulted and requires restart: {self.fault}")
        try:
            canonical = json.dumps([operation, payload], sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("Request must contain finite JSON values") from error
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        key = payload["request_id"]
        if key in self._signatures:
            if digest != self._signatures[key]:
                raise ValueError("request_id was already used with a different payload or operation")
            if key not in self._responses:
                raise ValueError("Cached response expired; this request cannot be executed again")
            return key, digest, copy.deepcopy(self._responses[key])
        return key, digest, None

    def _save(self, key, digest, response):
        self._signatures[key] = digest
        self._responses[key] = copy.deepcopy(response)
        while len(self._responses) > self.response_cache_size:
            self._responses.popitem(last=False)
        return copy.deepcopy(response)

    def _backend_failure(self, error):
        self.fault = f"{type(error).__name__}: {error}"
        if hasattr(self.backend, "mark_fault"):
            try:
                self.backend.mark_fault(self.fault)
            except Exception:
                pass  # Preserve the original infrastructure error if logging fails.
        raise RuntimeError(f"Simulation infrastructure failed; restart required: {self.fault}") from error

    def _episodes(self, episodes):
        if not isinstance(episodes, list) or len(episodes) != self.num_envs:
            raise ValueError("reset requires exactly num_envs episode descriptions")
        cfg = self.settings["config"]
        randomization = cfg.get("panel_randomization", {})
        normalized = []
        for episode in episodes:
            if not isinstance(episode, dict):
                raise ValueError("Every episode description must be an object")
            floor = _integer(episode.get("floor"), "floor", 24)
            if floor > 35:
                raise ValueError("floor must be in 24..35")
            row = {"floor": floor}
            for axis in "xy":
                value = _finite(episode.get(f"offset_{axis}_m"), f"offset_{axis}_m")
                if randomization.get("enabled") is True:
                    low = float(randomization[f"min_offset_{axis}_m"])
                    high = float(randomization[f"max_offset_{axis}_m"])
                else:
                    low = high = float(cfg.get(f"panel_offset_{axis}_m", 0.))
                if not low <= value <= high:
                    raise ValueError(f"Episode {axis} offset is outside the frozen training bounds")
                row[f"offset_{axis}_m"] = value
            normalized.append(row)
        return normalized

    def reset(self, payload):
        key, digest, cached = self._request("reset", payload)
        if cached is not None:
            return cached
        if self.items is not None and self._active():
            raise ValueError("reset is only allowed after every environment has ended")
        episodes = self._episodes(payload.get("episodes"))
        seed = _integer(payload.get("seed"), "seed")
        self.run_id = payload["run_id"]
        try:
            if hasattr(self.backend, "record_run"):
                self.backend.record_run(self.run_id, "active")
            output = self.backend.reset(episodes, seed)
            rows = self._rows(output, set(range(self.num_envs)))
            cohort = uuid.uuid4().hex
            items = []
            for env_id in range(self.num_envs):
                row = rows[env_id]
                items.append(dict(env_id=env_id, episode_id=f"{cohort}:{env_id}",
                    observation=copy.deepcopy(row["observation"]), terminated=False, truncated=False,
                    info=copy.deepcopy(row.get("info", {}))))
        except Exception as error:
            self._backend_failure(error)
        self.cohort_id, self.step_id, self.items = cohort, 0, items
        response = dict(protocol_version=1, cohort_id=cohort, step_id=0, items=items, all_done=False)
        return self._save(key, digest, response)

    def close_run(self, payload):
        """Release an ended vector so another distinct experiment can own it."""
        key, digest, cached = self._request("close_run", payload)
        if cached is not None:
            return cached
        if self.run_id is None or self.items is None or self._active():
            raise ValueError("close_run requires an owned run with every environment done")
        run_id, cohort_id = self.run_id, self.cohort_id
        try:
            if hasattr(self.backend, "record_run"):
                self.backend.record_run(run_id, "closed")
        except Exception as error:
            self._backend_failure(error)
        self._closed_runs.add(run_id)
        self.run_id = self.cohort_id = self.items = None
        self.step_id = 0
        return self._save(key, digest, dict(protocol_version=1, closed=True, run_id=run_id,
            cohort_id=cohort_id, all_done=True))

    @staticmethod
    def _rows(output, expected):
        if not isinstance(output, list) or len(output) != len(expected):
            raise RuntimeError("Backend returned an incorrect environment count")
        rows = {}
        for item in output:
            env_id = _integer(item.get("env_id"), "backend env_id")
            if env_id in rows or env_id not in expected or not isinstance(item.get("observation"), dict):
                raise RuntimeError("Backend returned invalid or duplicate environment data")
            rows[env_id] = item
        if set(rows) != expected:
            raise RuntimeError("Backend omitted an environment")
        return rows

    def step(self, payload):
        key, digest, cached = self._request("step", payload)
        if cached is not None:
            return cached
        if self.items is None or payload.get("cohort_id") != self.cohort_id:
            raise ValueError("Missing reset or stale cohort_id")
        if _integer(payload.get("step_id"), "step_id") != self.step_id:
            raise ValueError("Stale step_id")
        active = set(self._active())
        if not active:
            raise ValueError("All environments have ended; reset the vector before stepping")
        values = payload.get("actions")
        if not isinstance(values, list) or len(values) != len(active):
            raise ValueError("step must contain exactly the currently active environments")
        actions = {}
        for row in values:
            if not isinstance(row, dict):
                raise ValueError("Every action item must be an object")
            env_id = _integer(row.get("env_id"), "env_id")
            if env_id not in active or env_id in actions:
                raise ValueError("Duplicate, inactive, or unknown env_id")
            actions[env_id] = validate_actions_pose8(row.get("actions_pose8"))
        try:
            rows = self._rows(self.backend.step(actions), active)
            items = copy.deepcopy(self.items)
            for env_id, item in enumerate(items):
                item.update(reward=0., discount=1., executed_physics_steps=0)
                if env_id not in active:
                    continue
                row = rows[env_id]
                ticks = _integer(row.get("executed_physics_steps"), "executed_physics_steps", 1)
                if ticks > 28:
                    raise RuntimeError("Backend executed more than one action chunk")
                terminated, truncated = row.get("terminated"), row.get("truncated")
                if type(terminated) is not bool or type(truncated) is not bool or (terminated and truncated):
                    raise RuntimeError("Invalid backend termination flags")
                info = copy.deepcopy(row.get("info", {}))
                success = info.get("success") is True
                if success and (not terminated or truncated):
                    raise RuntimeError("A physical success must be a true termination")
                item.update(observation=copy.deepcopy(row["observation"]), info=info,
                    terminated=terminated, truncated=truncated, executed_physics_steps=ticks,
                    reward=self.single_gamma ** ((ticks - 1) / 4) if success else 0.,
                    discount=self.single_gamma ** (ticks / 4))
        except Exception as error:
            self._backend_failure(error)
        self.items, self.step_id = items, self.step_id + 1
        response = dict(protocol_version=1, cohort_id=self.cohort_id, step_id=self.step_id,
            items=items, all_done=not self._active())
        return self._save(key, digest, response)


def _file_identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def gpu_dynamics_memory_config(num_envs):
    """Explicit GPU capacities with headroom for PressB's articulated scenes.

    PhysX's standalone defaults can drop broadphase interactions at 64 copies.
    Scale the working buffers with environment count and never rely on those
    defaults. A PhysX error still invalidates the run instead of accepting loss.
    """
    count = _integer(num_envs, "num_envs", 1)
    def capacity(minimum, per_env):
        required = max(minimum, count * per_env)
        return 1 << (required - 1).bit_length()
    return dict(max_rigid_contact_count=capacity(2**20, 2**14),
        max_rigid_patch_count=capacity(2**16, 2**10),
        found_lost_pairs_capacity=capacity(2**18, 2**12),
        found_lost_aggregate_pairs_capacity=capacity(2**16, 2**8),
        total_aggregate_pairs_capacity=capacity(2**18, 2**12),
        collision_stack_size=capacity(2**26, 2**20),
        heap_capacity=capacity(2**26, 2**20),
        temp_buffer_capacity=capacity(2**24, 2**18))


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class IsaacVectorBackend:
    """Physical backend: one Isaac World, whole-vector resets, independent tasks.

    Construct and call exclusively on the main HTTP thread. Waiting for the next
    request performs no app.update/world.step, so network and inference latency
    do not consume simulation time.
    """

    def __init__(self, app, *, project_root, config, snapshot, dataset, output,
                 num_envs=3, max_seconds=15., smoothing_window=3, asset_bundle=None, gpu=0,
                 single_gamma=.99, camera_resolution=(640, 480), gpu_dynamics=False):
        self._thread = threading.get_ident()
        self.app = app
        self.root = Path(project_root).resolve()
        self.output = Path(output).resolve()
        if self.output.exists():
            raise FileExistsError(f"Simulation output directory must be new: {self.output}")
        self.cfg = json.loads(Path(config).read_text())
        collection_path = Path(dataset) / "meta/collection_metadata.json"
        collection = json.loads(collection_path.read_text())
        source_identity = _file_identity(snapshot)
        if self.cfg != collection["config"] or source_identity["sha256"] != collection["scene_sha256"]:
            raise ValueError("Simulation config/snapshot differs from the frozen training collection")
        self.num_envs = _integer(num_envs, "num_envs", 1)
        self.max_seconds = _finite(max_seconds, "max_seconds")
        if not 0 < self.max_seconds <= 120 or abs(self.max_seconds * 120 - round(self.max_seconds * 120)) > 1e-7:
            raise ValueError("max_seconds must be in (0,120] and on a 120 Hz tick")
        if abs(float(self.cfg["physics_dt"]) - 1 / 120) > 1e-12:
            raise ValueError("Simulation physics_dt must be 1/120")
        if not np.allclose(self.cfg["gripper_joint_positions_m"], [.004, -.004], rtol=0., atol=1e-10):
            raise ValueError("The fixed gripper must use joint positions [+0.004,-0.004] m")
        self.smoothing = smoothing_settings(smoothing_window)
        self.window = self.smoothing["window"]
        self.max_steps = round(self.max_seconds * 120)
        self.dt = 1 / 120
        self.home = np.asarray(self.cfg["home_q"], dtype=float)
        self.output.mkdir(parents=True, exist_ok=False)
        self._cohort_index = -1
        self._items = None
        self._dirty_lights = False
        self._chunk_index = 0
        self._world_ticks = 0
        self._run_id = None
        self._idle_time = None
        self._idle_joints = None

        # Reuse the current platform's rendering contract without modifying or
        # importing any simulation from the historical evaluator.
        script_root = str(self.root / "scripts")
        if script_root not in sys.path:
            sys.path.insert(0, script_root)
        from collect_dataset import (DATASET_ANTI_ALIASING, LIGHT_SETTLE_CAPTURES,
            LIGHT_SETTLE_SUBFRAMES, RENDER_EXPOSURE_CONTROLS, RENDER_HISTORY_CONTROLS)
        from pressb.dataset_scene import create_envs, build_tiled_rgb
        from pressb.kinematics import PiperKinematics
        from isaacsim.core.api import World
        from pxr import UsdLux
        import carb
        import omni.physx
        import omni.replicator.core as rep

        self._light_captures, self._light_subframes = LIGHT_SETTLE_CAPTURES, LIGHT_SETTLE_SUBFRAMES
        runtime = Path(snapshot).resolve()
        relocation = None
        if asset_bundle is not None:
            from pressb.scene_portability import prepare_runtime_snapshot
            relocation = prepare_runtime_snapshot(runtime, Path(asset_bundle), self.output)
            runtime = Path(relocation["runtime_snapshot"])
        self.urdf = self.root / self.cfg["robot_urdf"]
        self.tcp_kin = PiperKinematics(self.urdf, tip_offset=.1358)
        self.tip_kin = PiperKinematics(self.urdf, tip_offset=.24)
        self.world = World(stage_units_in_meters=1., physics_dt=self.dt, rendering_dt=1 / 30,
            backend="numpy", device="cpu")
        self.gpu_memory_config = {}
        if gpu_dynamics:
            # Experimental opt-in: NumPy tensor reads remain on the host, so
            # keep PhysX readback enabled. The default legacy path is unchanged.
            carb.settings.get_settings().set_bool("/physics/suppressReadback", False)
            context = self.world.get_physics_context()
            context.set_broadphase_type("GPU")
            context.enable_gpu_dynamics(True)
            context.enable_ccd(False)
            for name, value in gpu_dynamics_memory_config(self.num_envs).items():
                getattr(context, "set_gpu_" + name)(value)
                actual = int(getattr(context, "get_gpu_" + name)())
                if actual != value:
                    raise RuntimeError(f"PhysX GPU capacity {name} was not applied: {actual} != {value}")
                self.gpu_memory_config[name] = actual
        self.world.get_physics_context().set_physx_update_transformations_settings(
            update_to_usd=True, update_velocities_to_usd=True)
        self.envs = create_envs(self.world, runtime, self.num_envs, cfg=self.cfg)
        paths = [path for env in self.envs for path in (env.wrist_camera_path, env.global_camera_path)]
        self.product, self.annotator, self.split_rgb = build_tiled_rgb(paths, resolution=camera_resolution)
        rep.orchestrator.set_capture_on_play(False)
        self.world.reset()
        self.controls = []
        for env in self.envs:
            arm = np.array([env.robot.get_dof_index(name) for name in self.tcp_kin.joint_names])
            fingers = np.array([env.robot.get_dof_index(name) for name in ("joint7", "joint8")])
            command = env.robot.get_joint_positions().copy()
            controller = env.robot.get_articulation_controller()
            controller.set_gains(kps=np.full(len(command), 12000.), kds=np.full(len(command), 600.))
            self.controls.append((arm, fingers, command, controller))
        domes = [{"path": str(prim.GetPath()), "active": prim.IsActive()}
            for prim in self.world.stage.TraverseAll() if prim.IsA(UsdLux.DomeLight)]
        if sum(row["active"] for row in domes) != 1:
            raise RuntimeError("Vector simulation requires exactly one active DomeLight")
        settings = carb.settings.get_settings()
        for key, value in {**RENDER_HISTORY_CONTROLS, **RENDER_EXPOSURE_CONTROLS}.items():
            settings.set(key, value)
        self.forces = np.zeros((self.num_envs, 12))
        self.collisions = [[] for _ in self.envs]
        self.recording = set()
        self.body_map = {info["body_path"]: (env.env_id, floor)
            for env in self.envs for floor, info in env.button_info.items()}
        self.tool_paths = {path for env in self.envs for path in env.tool_colliders}
        self.subscription = omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(self._on_contact)
        import importlib.metadata
        versions = {name: importlib.metadata.version(name) for name in ("isaacsim", "torch", "numpy", "scipy", "Pillow")}
        sources = ["src/pressb/online_rl/simulation.py", "scripts/serve_rl_simulation.py",
            "src/pressb/policy_eval.py", "src/pressb/replay_control.py", "src/pressb/motion_smoothing.py",
            "src/pressb/dataset_scene.py", "scripts/collect_dataset.py", "src/pressb/scene_portability.py"]
        self.manifest = dict(protocol_version=1, service="simulation", created_at=datetime.now(timezone.utc).isoformat(),
            config=copy.deepcopy(self.cfg), config_identity=_file_identity(config), scene_identity=source_identity,
            config_sha256=_file_identity(config)["sha256"], robot_urdf_sha256=_file_identity(self.urdf)["sha256"],
            scene_sha256=source_identity["sha256"], collection_identity=_file_identity(collection_path),
            collection_fingerprint=collection["collection_fingerprint"], robot_urdf=_file_identity(self.urdf),
            scene_relocation=relocation, runtime_snapshot=str(runtime), runtime_versions=versions,
            sources={name: _file_identity(self.root / name) for name in sources}, gpu=gpu,
            num_envs=self.num_envs, max_seconds=self.max_seconds, physics_hz=120, action_fps=30, action_horizon=7,
            smoothing_window=self.window, motion_smoothing=self.smoothing,
            single_gamma=float(single_gamma), control_state_dim=19 + 6 * self.window,
            control_state_schema=CONTROL_STATE_SCHEMA, pose_frame="base_link", pose_link="gripper_tcp",
            quaternion_order="wxyz", tcp_offset_m=.1358, physical_tip_offset_m=.24,
            gripper_width_m=.008, camera_order=["global", "wrist"], image_resolution=[640, 480],
            renderer_settings=dict(anti_aliasing=DATASET_ANTI_ALIASING, light_settle_captures=LIGHT_SETTLE_CAPTURES,
                light_settle_subframes=LIGHT_SETTLE_SUBFRAMES, history_controls=RENDER_HISTORY_CONTROLS,
                exposure_controls=RENDER_EXPOSURE_CONTROLS), domes=domes,
            reset_mode="whole_vector_barrier_no_autoreset", recorded_actions_used=False,
            target_planner_used=False, clock="Physics advances only during reset settling and explicit step requests",
            logging="Reset/episode manifests and action/contact records; no per-frame training video")
        _write_json(self.output / "simulation_manifest.json", self.manifest)
        self._status("ready")

    def _check_thread(self):
        if threading.get_ident() != self._thread:
            raise RuntimeError("Isaac may only be called from its owner HTTP/main thread")

    def health(self):
        self._check_thread()
        return copy.deepcopy(self.manifest)

    def _status(self, state, **extra):
        _write_json(self.output / "status.json", dict(status=state, cohort_index=self._cohort_index,
            chunk_index=self._chunk_index, world_ticks=self._world_ticks,
            updated_at=datetime.now(timezone.utc).isoformat(), **extra))

    def record_run(self, run_id, status):
        self._check_thread()
        if status == "active" and self._run_id == run_id:
            return
        with (self.output / "runs.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(run_id=run_id, status=status,
                cohort_index=self._cohort_index, created_at=datetime.now(timezone.utc).isoformat())) + "\n")
        self._run_id = run_id if status == "active" else None

    def _remember_frozen_state(self):
        self._idle_time = self.world.current_time
        self._idle_joints = [env.robot.get_joint_positions().copy() for env in self.envs]

    def _check_frozen_state(self):
        if self._idle_time is not None and (abs(self.world.current_time - self._idle_time) > 1e-10
                or any(not np.array_equal(env.robot.get_joint_positions(), q)
                    for env, q in zip(self.envs, self._idle_joints))):
            raise RuntimeError("Simulation time or joints changed while waiting for the next HTTP request")

    def mark_fault(self, error):
        self._check_thread()
        self._status("faulted", error=str(error))

    def _on_contact(self, headers, data):
        from pxr import PhysicsSchemaTools
        for header in headers:
            if not header.num_contact_data:
                continue
            actors = [str(PhysicsSchemaTools.intToSdfPath(value)) for value in (header.actor0, header.actor1)]
            colliders = [str(PhysicsSchemaTools.intToSdfPath(value)) for value in (header.collider0, header.collider1)]
            pair = next((self.body_map[actor] for actor in actors if actor in self.body_map), None)
            magnitude = sum(float(np.linalg.norm(data[k].impulse)) / self.dt for k in
                range(header.contact_data_offset, header.contact_data_offset + header.num_contact_data))
            if pair is not None and any(collider in self.tool_paths for collider in colliders):
                self.forces[pair[0], pair[1] - 24] += magnitude
                continue
            for env in self.envs:
                eid = env.env_id
                if eid not in self.recording:
                    continue
                prefix = env.robot_path + "/"
                robot_actors = [actor for actor in actors if actor.startswith(prefix)]
                if not robot_actors or magnitude <= .1:
                    continue
                shaft = prefix + "link6/PressStylus"
                pads = {prefix + "link7/collisions", prefix + "link8/collisions"}
                if shaft in colliders and any(collider in pads for collider in colliders):
                    continue
                reason = None
                if pair is not None:
                    reason = "non_stylus_button_contact"
                elif len(robot_actors) == 2 and actors[0] != actors[1]:
                    reason = "robot_self_contact"
                elif len(robot_actors) == 1 and robot_actors[0].rsplit("/", 1)[-1] not in ("dummy_link", "base_link", "link1"):
                    reason = "robot_environment_contact"
                if reason and len(self.collisions[eid]) < 100:
                    self.collisions[eid].append(dict(physics_index=self._items[eid]["ticks"],
                        actors=actors, colliders=colliders, force_n=magnitude, reason=reason))

    def _measured(self, env_id):
        from scipy.spatial.transform import Rotation
        env = self.envs[env_id]
        arm, fingers, _, _ = self.controls[env_id]
        joints = env.robot.get_joint_positions().astype(np.float64)
        q, finger_q = joints[arm], joints[fingers]
        velocity = env.robot.get_joint_velocities()[arm].astype(np.float64)
        tcp = self.tcp_kin.fk(q)
        quaternion = Rotation.from_matrix(tcp[:3, :3]).as_quat()[[3, 0, 1, 2]]
        state = np.r_[tcp[:3, 3], quaternion, finger_q[0] - finger_q[1]]
        if not np.isfinite(np.r_[state, velocity]).all():
            raise RuntimeError("Nonfinite measured robot state")
        return state, q, velocity

    def _capture(self, changed=False):
        import omni.replicator.core as rep
        import omni.usd
        before = self.world.current_time
        prior = [env.robot.get_joint_positions().copy() for env in self.envs]
        changed = changed or self._dirty_lights
        if changed:
            omni.usd.get_context().reset_renderer_accumulation()
            for _ in range(self._light_captures):
                rep.orchestrator.step(delta_time=0., pause_timeline=False, rt_subframes=2)
        rep.orchestrator.step(delta_time=0., pause_timeline=False,
            rt_subframes=self._light_subframes if changed else 2)
        if abs(self.world.current_time - before) > 1e-10:
            raise RuntimeError("RGB capture advanced simulation time")
        if any(not np.array_equal(env.robot.get_joint_positions(), q) for env, q in zip(self.envs, prior)):
            raise RuntimeError("RGB capture changed actual robot joints")
        tiles = self.split_rgb(self.annotator.get_data())
        if any(rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8 or rgb.std() < 2 for rgb in tiles):
            raise RuntimeError("Invalid live camera RGB")
        self._dirty_lights = False
        return tiles

    def _observation(self, env_id, tiles):
        from .protocol import encode_image
        item = self._items[env_id]
        state, q, velocity = self._measured(env_id)
        return dict(task=f"Press {item['floor']} floor.", state=state.tolist(),
            images={"global": encode_image(tiles[2 * env_id + 1]), "wrist": encode_image(tiles[2 * env_id])},
            control_state=control_state(q, velocity, item["solver"].q, list(item["history"]),
                item["ticks"] * self.dt, self.max_seconds))

    def reset(self, episodes, seed):
        self._check_thread()
        self._check_frozen_state()
        from isaacsim.core.utils.types import ArticulationAction
        from pressb.dataset_scene import set_light, set_panel_offset, validate_panel_layout
        if self._items is not None and any(not item["done"] for item in self._items):
            raise RuntimeError("Backend requires a whole-vector episode barrier")
        self.recording.clear()
        self._cohort_index += 1
        self._chunk_index = 0
        for env, episode, control in zip(self.envs, episodes, self.controls):
            arm, fingers, command, controller = control
            command[arm], command[fingers] = self.home, self.cfg["gripper_joint_positions_m"]
            env.robot.set_joint_positions(command)
            env.robot.set_joint_velocities(np.zeros_like(command))
            controller.apply_action(ArticulationAction(joint_positions=command))
            for floor in range(24, 36):
                set_light(env, floor, False)
            set_panel_offset(self.world, env, offset_y_m=episode["offset_y_m"], offset_x_m=episode["offset_x_m"])
        for step in range(90):
            self.forces.fill(0.)
            self.world.step(render=False)
            self._world_ticks += 1
            if step % 4 == 0:
                self.world.render()
        for _ in range(8):
            self.world.render()
        evidence = [validate_panel_layout(self.world, env) for env in self.envs]
        self.collisions = [[] for _ in self.envs]
        self.forces.fill(0.)
        cohort_dir = self.output / f"cohort_{self._cohort_index:06d}"
        cohort_dir.mkdir()
        self._items = []
        for env, episode, control in zip(self.envs, episodes, self.controls):
            arm, _, command, _ = control
            state, q, velocity = self._measured(env.env_id)
            if np.max(np.abs(q - self.home)) > self.cfg["home_tolerance_rad"]:
                raise RuntimeError("Episode did not start at the recorded home pose")
            travel = self._travel(env)
            if np.max(np.abs(travel)) > self.cfg["release_threshold"]:
                raise RuntimeError("Buttons did not return before the episode")
            initial = command[arm].astype(np.float64)
            solver = PolicyPoseController(self.urdf, initial, fps=30)
            self._items.append(dict(floor=episode["floor"], ticks=0, done=False, termination=None,
                lit=set(), events=[], solver=solver, smoother=JointCommandSmoother(initial, window=self.window),
                history=deque((initial.copy() for _ in range(self.window)), maxlen=self.window),
                min_target_distance=float("inf"), max_tracking_error=0., max_target_force=0., max_target_travel=0.,
                reset_episode=copy.deepcopy(episode), directory=cohort_dir, seed=seed,
                layout_evidence=evidence[env.env_id]))
            self.recording.add(env.env_id)
        tiles = None
        for attempt in range(12):
            try:
                tiles = self._capture(changed=True)
                break
            except ValueError:
                if attempt == 11:
                    raise
        rows = []
        for eid, item in enumerate(self._items):
            self._measure_metrics(eid)
            observation = self._observation(eid, tiles)
            info = self._info(eid)
            rows.append(dict(env_id=eid, observation=observation, info=info))
            _write_json(cohort_dir / f"env_{eid:03d}_reset.json", dict(env_id=eid, seed=seed,
                run_id=self._run_id, episode=episodes[eid], initial_state=observation["state"], initial_control_state=observation["control_state"],
                layout=evidence[eid], info=info))
        self._status("ready", active_env_ids=list(range(self.num_envs)))
        self._remember_frozen_state()
        return rows

    @staticmethod
    def _travel(env):
        return np.array([env.buttons[floor].get_world_pose()[0][0] - env.button_info[floor]["rest_x"]
            for floor in range(24, 36)])

    def _measure_metrics(self, eid):
        from pxr import Gf, Usd, UsdGeom
        env, item = self.envs[eid], self._items[eid]
        _, q, _ = self._measured(eid)
        transform = UsdGeom.Xformable(self.world.stage.GetPrimAtPath(env.link6_path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        tip = np.asarray(transform.Transform(Gf.Vec3d(0, 0, .24)))
        fk_error = float(np.linalg.norm(tip - self.tip_kin.fk(q)[:3, 3] - env.base_position))
        if not math.isfinite(fk_error) or fk_error > .005:
            raise RuntimeError(f"Measured FK differs from physical tool pose by {fk_error} m")
        distance = float(np.linalg.norm(tip - np.asarray(env.button_info[item["floor"]]["center"])))
        arm, _, command, _ = self.controls[eid]
        tracking = float(np.max(np.abs(q - command[arm])))
        travel = self._travel(env)
        if not np.isfinite(np.r_[travel, self.forces[eid], distance, tracking]).all():
            raise RuntimeError("Nonfinite physical metrics")
        item["min_target_distance"] = min(item["min_target_distance"], distance)
        item["max_tracking_error"] = max(item["max_tracking_error"], tracking)
        item["max_target_force"] = max(item["max_target_force"], float(self.forces[eid, item["floor"] - 24]))
        item["max_target_travel"] = max(item["max_target_travel"], float(travel[item["floor"] - 24]))
        return travel

    def _info(self, eid):
        item = self._items[eid]
        pressed = [event["floor"] for event in item["events"] if event["type"] == "pressed"]
        return dict(termination=item["termination"], success=item["termination"] == "target_pressed",
            task_success=pressed == [item["floor"]], pressed_floors=pressed,
            physics_index=item["ticks"], sim_seconds=item["ticks"] * self.dt,
            events=copy.deepcopy(item["events"]), unexpected_collisions=copy.deepcopy(self.collisions[eid]),
            collision_count=len(self.collisions[eid]), min_target_tip_distance_m=item["min_target_distance"],
            max_joint_tracking_error_rad=item["max_tracking_error"],
            max_target_contact_force_n=item["max_target_force"], max_target_travel_m=item["max_target_travel"],
            final_partial_interval=bool(item["ticks"] % 4),
            motion_smoothing=self.smoothing)

    def step(self, actions):
        self._check_thread()
        self._check_frozen_state()
        from isaacsim.core.utils.types import ArticulationAction
        from pressb.dataset_scene import set_light
        from scipy.spatial.transform import Rotation
        if not self.app.is_running():
            raise RuntimeError("Simulator closed before the request")
        expected = {eid for eid, item in enumerate(self._items or []) if not item["done"]}
        if set(actions) != expected or not expected:
            raise RuntimeError("Backend step requires every active environment exactly once")
        before = {eid: self._items[eid]["ticks"] for eid in expected}
        terminal_observations = {}
        action_records = {eid: [] for eid in expected}
        # Backend validates again so direct callers cannot bypass safety checks.
        actions = {eid: validate_actions_pose8(value) for eid, value in actions.items()}
        for index in range(7):
            active = sorted(eid for eid in expected if not self._items[eid]["done"])
            if not active:
                break
            pending = {}
            for eid in active:
                item = self._items[eid]
                previous = item["solver"].q
                target, diagnostics = item["solver"].solve(actions[eid][index])
                pending[eid] = dict(action_index=index, action_pose8=actions[eid][index].tolist(),
                    q_before=previous.tolist(), q_target=target.tolist(), physics_start_index=item["ticks"],
                    **diagnostics)
            for substep in range(1, 5):
                active = sorted(eid for eid in expected if not self._items[eid]["done"])
                if not active:
                    break
                if not self.app.is_running():
                    raise RuntimeError("Simulator closed while executing actions")
                for eid in active:
                    item, action = self._items[eid], pending[eid]
                    arm, fingers, command, controller = self.controls[eid]
                    q0, q1 = np.asarray(action["q_before"]), np.asarray(action["q_target"])
                    linear = q0 + (q1 - q0) * substep / 4
                    item["history"].append(linear.copy())
                    command[arm] = item["smoother"].step(linear)
                    command[fingers] = self.cfg["gripper_joint_positions_m"]
                    controller.apply_action(ArticulationAction(joint_positions=command))
                    item["ticks"] += 1
                self.forces.fill(0.)
                self.world.step(render=False)
                self._world_ticks += 1
                terminal = []
                for eid in active:
                    item, env = self._items[eid], self.envs[eid]
                    travel = self._measure_metrics(eid)
                    for floor in range(24, 36):
                        on = floor not in item["lit"] and travel[floor - 24] >= self.cfg["press_threshold"] and self.forces[eid, floor - 24] > .02
                        off = floor in item["lit"] and travel[floor - 24] <= self.cfg["release_threshold"]
                        if on or off:
                            self._dirty_lights = True
                            item["lit"].add(floor) if on else item["lit"].remove(floor)
                            set_light(env, floor, on)
                            item["events"].append(dict(type="pressed" if on else "released", floor=floor,
                                physics_index=item["ticks"], sim_time=item["ticks"] * self.dt,
                                travel_m=float(travel[floor - 24]), force_n=float(self.forces[eid, floor - 24])))
                    presses = [event["floor"] for event in item["events"] if event["type"] == "pressed"]
                    reason = ("unexpected_collision" if self.collisions[eid] else
                        "target_pressed" if presses == [item["floor"]] else
                        "wrong_button_pressed" if presses else
                        "time_limit" if item["ticks"] >= self.max_steps else None)
                    if reason:
                        item.update(done=True, termination=reason)
                        self.recording.discard(eid)
                        terminal.append(eid)
                    if substep == 4 or reason:
                        record = pending[eid]
                        arm, _, command, _ = self.controls[eid]
                        executed = command[arm].astype(np.float64)
                        tcp = self.tcp_kin.fk(executed)
                        position, rotation, _ = action_pose(actions[eid][index])
                        record.update(physics_end_index=item["ticks"], q_executed_endpoint=executed.tolist(),
                            executed_position_residual_m=float(np.linalg.norm(tcp[:3, 3] - position)),
                            executed_rotation_residual_rad=float(Rotation.from_matrix(rotation @ tcp[:3, :3].T).magnitude()))
                        action_records[eid].append(record)
                # Match the evaluator's 30 Hz renderer cadence even though only
                # the next chunk observation is transported. Terminal RGB/state
                # is captured at its exact substep and cached before more ticks.
                if substep == 4 or terminal:
                    tiles = self._capture()
                    for eid in terminal:
                        terminal_observations[eid] = self._observation(eid, tiles)
        rows = []
        for eid in sorted(expected):
            item = self._items[eid]
            info = self._info(eid)
            info.update(executed_action_steps=len(action_records[eid]),
                executed_sim_seconds=(item["ticks"] - before[eid]) * self.dt,
                max_chunk_ik_position_residual_m=max(row["command_position_residual_m"] for row in action_records[eid]),
                velocity_limited_actions=sum(bool(row["velocity_saturated_joints"]) for row in action_records[eid]))
            observation = terminal_observations[eid] if item["done"] else self._observation(eid, tiles)
            rows.append(dict(env_id=eid, observation=observation,
                terminated=item["done"] and item["termination"] != "time_limit",
                truncated=item["termination"] == "time_limit",
                executed_physics_steps=item["ticks"] - before[eid], info=info))
            log = dict(run_id=self._run_id, cohort_index=self._cohort_index, chunk_index=self._chunk_index, env_id=eid,
                physics_start_index=before[eid], physics_end_index=item["ticks"],
                state=observation["state"], control_state=observation["control_state"],
                actions=action_records[eid], info=info)
            with (item["directory"] / f"env_{eid:03d}_actions.jsonl").open("a") as stream:
                stream.write(json.dumps(log, allow_nan=False) + "\n")
            if item["done"]:
                _write_json(item["directory"] / f"env_{eid:03d}_terminal.json", dict(
                    run_id=self._run_id, episode=item["reset_episode"], seed=item["seed"], info=info,
                    terminal_state=observation["state"], terminal_control_state=observation["control_state"]))
        self._chunk_index += 1
        self._status("ready", active_env_ids=sorted(self.recording))
        self._remember_frozen_state()
        return rows

    def close(self):
        self._check_thread()
        self.recording.clear()
        self.subscription = None
        self._status("closed")
