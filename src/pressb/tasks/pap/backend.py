"""Process-parallel PAP backend for the task-neutral RL simulation service.

Each environment owns an Isaac application in a separate Python process. Commands
are dispatched to every selected worker before any results are awaited. Resetting
one environment therefore does not step another environment's physics. This is a
compatibility backend, not the tensor-vectorized elevator implementation: every
worker has its own renderer and associated GPU memory cost.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
import time
from multiprocessing.connection import Listener

import numpy as np

from pressb.control.joint_actions import JointActionPostprocessor
from pressb.robots.piper.kinematics import PiperKinematics
from pressb.tasks.pap.evaluation import COLORS, placement_region, sample_polygon
from pressb.tasks.pap.task import object_geometry


ROOT = Path(__file__).resolve().parents[4]
NAMES = [f"joint{i}" for i in range(1, 7)] + ["gripper_width"]
UNITS = ["rad"] * 6 + ["m"]
_MAX_MESSAGE = 64 * 1024 * 1024
_TASK_SOURCE_FILES = (
    "tasks/pap/backend.py", "tasks/pap/worker.py",
    "tasks/pap/task.py", "tasks/pap/scene.py", "tasks/pap/evaluation.py",
    "tasks/pap/initialization.py", "runtime/scene.py", "runtime/workbench.py",
    "robots/piper/wrist_camera.py", "robots/piper/kinematics.py",
    "control/joint_actions.py", "sensors/camera.py",
)


def _source_identity(names):
    """Record task implementations, shared dependencies and worker entry points."""
    result = {}
    for name in names:
        content = (ROOT / "src" / "pressb" / name).read_bytes()
        result[name] = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
    return result


def _source_revision(sources):
    content = json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _path(value):
    path = Path(value).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def load_config(config):
    """Resolve all paths and validate settings before launching expensive workers."""
    result = copy.deepcopy(config)
    scene_path = _path(result.get("scene_config", "configs/piper_pick_place.json"))
    scene = json.loads(scene_path.read_text())
    scene.update(copy.deepcopy(result.get("scene_overrides", {})))
    if not scene.get("task_props", True):
        raise ValueError("PAP requires the task object and open box")
    for key in ("robot_usd", "robot_urdf"):
        scene[key] = str(_path(scene[key]))
    result["scene_config"] = str(scene_path)
    result["scene"] = scene
    result["num_envs"] = _integer(result.get("num_envs", 1), "num_envs")
    result["horizon"] = _integer(result.get("horizon", 32), "horizon")
    result["physics_dt"] = _positive(scene["physics_dt"], "physics_dt")
    result["control_hz"] = _positive(scene["fps"], "control_hz")
    stride = 1 / result["physics_dt"] / result["control_hz"]
    if not math.isclose(stride, round(stride), abs_tol=1e-9):
        raise ValueError("Control frequency must divide the physics frequency")
    result["physics_ticks_per_action"] = round(stride)
    result["max_episode_seconds"] = _positive(result.get("max_episode_seconds", 20.), "max_episode_seconds")
    if result["max_episode_seconds"] < result["physics_dt"]:
        raise ValueError("max_episode_seconds must allow at least one physics tick")
    result["gamma"] = _positive(result.get("gamma", .99), "gamma")
    if result["gamma"] > 1:
        raise ValueError("gamma must be in (0, 1]")
    for name, default in (("startup_timeout", 300.), ("command_timeout", 180.)):
        result[name] = _positive(result.get(name, default), name)
    result["python"] = str(_path(result.get("python", ".conda/envs/pressb/bin/python")))
    if not Path(result["python"]).is_file():
        raise FileNotFoundError(result["python"])
    trajectory = result.get("trajectory", scene.get("episode_source", {}).get("trajectory",
        "outputs/piper_pap_reconstruction_20261022/dataset_audit/episode_000000.npz"))
    result["trajectory"] = str(_path(trajectory))
    if not Path(result["trajectory"]).is_file():
        raise FileNotFoundError(result["trajectory"])
    expected = scene.get("episode_source", {}).get("episode_index")
    if expected is not None and Path(result["trajectory"]).stem != f"episode_{int(expected):06d}":
        raise ValueError("Scene episode_source and initialization trajectory differ")
    smoothing = result.get("smoothing", scene.get("policy_execution", {}).get("smoothing", {}))
    result["smoothing"] = {"mode": smoothing.get("mode", "none"), "alpha": smoothing.get("alpha", .8)}
    JointActionPostprocessor(**result["smoothing"])
    if result["smoothing"]["mode"] != "none" and result["control_hz"] != 30:
        raise ValueError("JointActionPostprocessor is calibrated for 30 Hz control")
    result["reward"] = {"success": 1., "failure": 0., **result.get("reward", {})}
    if set(result["reward"]) != {"success", "failure"} or not all(
            isinstance(x, (float, int)) and not isinstance(x, bool) and math.isfinite(x)
            for x in result["reward"].values()):
        raise ValueError("reward requires finite success and failure values")
    result["outcome"] = {"stable_seconds": .5, "lift_height": .04, **result.get("outcome", {})}
    if set(result["outcome"]) != {"stable_seconds", "lift_height"}:
        raise ValueError("outcome supports stable_seconds and lift_height")
    for key, value in result["outcome"].items():
        _positive(value, f"outcome.{key}")
    result["terminate_on_drop"] = result.get("terminate_on_drop", True)
    if type(result["terminate_on_drop"]) is not bool:
        raise ValueError("terminate_on_drop must be boolean")
    result["drop_below_table_m"] = _positive(result.get("drop_below_table_m", .05), "drop_below_table_m")
    result["settling_ticks"] = _integer(result.get("settling_ticks", 24), "settling_ticks", 0)
    result["threads_per_worker"] = _integer(result.get("threads_per_worker", 2), "threads_per_worker")
    result["task"] = result.get("task", scene["task_instruction"])
    if not isinstance(result["task"], str) or not result["task"].strip():
        raise ValueError("task must be a nonempty language instruction")
    result["model_inputs"] = copy.deepcopy(result.get("model_inputs", {}))
    if not isinstance(result["model_inputs"], dict):
        raise ValueError("model_inputs must be a mapping")
    source = scene.get("object", {"shape": "cube", **scene.get("cube", {})})
    result["objects"] = copy.deepcopy(result.get("objects", [source]))
    if not isinstance(result["objects"], list) or not result["objects"]:
        raise ValueError("objects must be a nonempty list of primitive object configurations")
    for obj in result["objects"]:
        if not isinstance(obj, dict):
            raise ValueError("Each object configuration must be a mapping")
        obj.setdefault("shape", "cube")
        obj.setdefault("center_xy", source["center_xy"])
        obj.setdefault("yaw_deg", source.get("yaw_deg", 0.))
        obj.setdefault("mass", source.get("mass", .025))
        material_color = obj.get("material", {}).get("color") if isinstance(obj.get("material", {}), dict) else None
        obj.setdefault("color", material_color if material_color is not None else
            scene.get("materials", {}).get("RedCube", {}).get("color",
                scene.get("material_colors", {}).get("RedCube", COLORS["red"])))
        obj["geometry"] = object_geometry(obj)
        if obj["shape"] == "cube":
            obj["edge"] = obj["geometry"]["edge"]
        elif obj["shape"] == "cuboid":
            obj["size"] = obj["geometry"]["size_xyz"]
        _positive(obj["mass"], "object.mass")
        clearance = obj.setdefault("clearance", .0005)
        if isinstance(clearance, bool) or not isinstance(clearance, (float, int)) or not math.isfinite(clearance) or clearance < 0:
            raise ValueError("Object clearance must be finite and nonnegative")
        material = obj.get("material", {})
        if not isinstance(material, dict) or set(material) - {"color", "roughness", "metallic", "specular"}:
            raise ValueError("Object material supports color, roughness, metallic, specular")
        for key, value in material.items():
            values = np.asarray(value, dtype=float)
            if ((key == "color" and values.shape != (3,)) or (key != "color" and values.shape != ())
                    or not np.isfinite(values).all() or np.any((values < 0) | (values > 1))):
                raise ValueError(f"Invalid object material {key}")
    randomization = result.setdefault("randomization", {})
    if not isinstance(randomization, dict):
        raise ValueError("randomization must be a mapping")
    if set(randomization) - {"position", "colors", "object_indices", "yaw_range_deg", "center_polygon"}:
        raise ValueError("Unsupported PAP randomization key")
    if randomization.get("position", "fixed") not in ("fixed", "markers", "polygon"):
        raise ValueError("randomization.position must be fixed, markers, or polygon")
    gpu_ids = result.get("gpu_ids", [result.get("gpu", 0)])
    if not isinstance(gpu_ids, list) or not gpu_ids:
        raise ValueError("gpu_ids must be a nonempty list")
    result["workers"] = result.get("workers", [{"gpu": gpu_ids[i % len(gpu_ids)]} for i in range(result["num_envs"])])
    if len(result["workers"]) != result["num_envs"]:
        raise ValueError("workers must have exactly num_envs entries")
    for worker in result["workers"]:
        if not isinstance(worker, dict) or set(worker) != {"gpu"}:
            raise ValueError("Each workers entry supports only gpu; task semantics must be shared")
        _integer(worker["gpu"], "worker.gpu", 0)
    # Validate stochastic configuration even when no environment has reset yet.
    for index in range(len(result["objects"])):
        sample_reset(result, 0, index, {"object_index": index})
    return result


def task_identity(config):
    """Fingerprint transition semantics without process placement/log paths."""
    from pressb.tasks.pap.initialization import load_policy_initial
    scene = copy.deepcopy(config["scene"])
    assets = {}
    for key in ("robot_urdf", "robot_usd"):
        path = Path(scene.pop(key))
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for data in iter(lambda: stream.read(1024*1024), b""):
                digest.update(data)
        assets[key] = digest.hexdigest()
    # Dataset location and historical fit reports do not change the runtime;
    # actual numeric camera/geometry values are retained in the scene content.
    for key in ("dataset", "episode_source", "camera_fit_reference", "box_fit_reference"):
        scene.pop(key, None)
    if isinstance(scene.get("wrist_camera"), dict):
        scene["wrist_camera"].pop("refinement_reference", None)
    if isinstance(scene.get("appearance_revision"), dict):
        scene["appearance_revision"].pop("reference", None)
    initial = load_policy_initial(config["trajectory"], scene["gripper_max_width_m"])["initial"].tolist()
    semantics = {key: copy.deepcopy(config[key]) for key in (
        "horizon", "physics_dt", "control_hz", "max_episode_seconds", "gamma", "smoothing",
        "reward", "outcome", "terminate_on_drop", "drop_below_table_m", "settling_ticks", "task",
        "model_inputs", "objects", "randomization")}
    semantics.update(scene=scene, asset_sha256=assets, initial_q_si=initial,
        success_rule_version="pap_bilateral_lift_full_containment_release_floor_rest_v1")
    serialized = json.dumps(semantics, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    sources = _source_identity(_TASK_SOURCE_FILES)
    return {"task_config_sha256": hashlib.sha256(serialized).hexdigest(), "asset_sha256": assets,
            "source_revision": _source_revision(sources), "source_identity": sources, "initial_q_si": initial,
            "asset_scope": "URDF and root robot USD file contents; external USD references are not recursively hashed"}


def sample_reset(config, seed, env_id, options):
    """Independent deterministic reset sampling, with explicit per-reset overrides."""
    allowed = {"center_xy", "color", "color_rgb", "object_index", "yaw_deg", "task", "model_inputs"}
    if not isinstance(options, dict) or set(options) - allowed:
        raise ValueError(f"PAP reset options support {sorted(allowed)}")
    _integer(seed, "seed", 0)
    rng = np.random.default_rng(np.random.SeedSequence([seed, env_id]))
    randomization = config.get("randomization", {})
    indices = randomization.get("object_indices", list(range(len(config["objects"]))))
    if not isinstance(indices, list) or not indices or any(type(i) is not int or not 0 <= i < len(config["objects"]) for i in indices):
        raise ValueError("randomization.object_indices contains an invalid object index")
    index = options.get("object_index", int(rng.choice(indices)))
    if type(index) is not int or not 0 <= index < len(config["objects"]):
        raise ValueError("object_index out of range")
    obj = config["objects"][index]
    yaw_range = randomization.get("yaw_range_deg", [obj["yaw_deg"], obj["yaw_deg"]])
    limits = np.asarray(yaw_range, dtype=float)
    if limits.shape != (2,) or not np.isfinite(limits).all() or limits[0] > limits[1]:
        raise ValueError("yaw_range_deg must be two ordered finite values")
    yaw = float(options.get("yaw_deg", rng.uniform(*limits)))
    if not math.isfinite(yaw):
        raise ValueError("yaw_deg must be finite")
    colors = randomization.get("colors")
    if colors is not None and (not isinstance(colors, list) or not colors or any(c not in COLORS for c in colors)):
        raise ValueError("randomization.colors must be a nonempty list of known colors")
    color = options.get("color", str(rng.choice(colors)) if colors else None)
    if color is not None and color not in COLORS:
        raise ValueError(f"Unknown color {color!r}")
    rgb = np.asarray(options.get("color_rgb", COLORS[color] if color else obj["color"]), dtype=float)
    if rgb.shape != (3,) or not np.isfinite(rgb).all() or np.any((rgb < 0) | (rgb > 1)):
        raise ValueError("color_rgb must have three finite values in [0, 1]")
    region = None
    mode = randomization.get("position", "fixed")
    if mode == "markers":
        # A bounding square is conservative for cuboids and cylinders; cubes use
        # their exact footprint. The four marker long axes define the four edges.
        edge = float(max(obj["geometry"]["size_xyz"][:2]))
        region = placement_region(config["scene"], edge=edge, yaw_deg=yaw)
        position = sample_polygon(region["valid_center_polygon_xy"], 1, rng)[0]
    elif mode == "polygon":
        position = sample_polygon(randomization["center_polygon"], 1, rng)[0]
    else:
        position = obj["center_xy"]
    position = np.asarray(options.get("center_xy", position), dtype=float)
    if position.shape != (2,) or not np.isfinite(position).all():
        raise ValueError("center_xy must be two finite base-frame metres")
    task = options.get("task", config["task"])
    inputs = options.get("model_inputs", config["model_inputs"])
    if not isinstance(task, str) or not task.strip() or not isinstance(inputs, dict):
        raise ValueError("reset task must be nonempty text and model_inputs must be a mapping")
    return {"object_index": index, "center_xy": position.tolist(), "yaw_deg": yaw,
            "color_rgb": rgb.tolist(), "color": color, "task": task, "model_inputs": copy.deepcopy(inputs),
            "seed": seed, "env_id": env_id, "sampling_region": region}


class _Worker:
    """Authenticated loopback IPC; JSON payloads, no untrusted pickle decoding."""
    def __init__(self, config, env_id, directory):
        auth = secrets.token_bytes(32)
        self.connection = None
        self.process = None
        self.listener = Listener(("127.0.0.1", 0), authkey=auth)
        self.listener._listener._socket.settimeout(config["startup_timeout"])
        self.log = (directory / f"worker_{env_id}.log").open("wb")
        worker_config = {**config, "env_id": env_id, **config["workers"][env_id]}
        config_path = directory / f"worker_{env_id}.json"
        config_path.write_text(json.dumps(worker_config, allow_nan=False))
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src")
        env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env[name] = str(config.get("threads_per_worker", 2))
        host, port = self.listener.address
        try:
            self.process = subprocess.Popen([config["python"], "-m", "pressb.tasks.pap.worker",
                "--config", str(config_path), "--host", host, "--port", str(port), "--auth", auth.hex()],
                cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=self.log, stderr=subprocess.STDOUT)
            self.connection = self.listener.accept()
            self.listener.close()
            self.receive(config["startup_timeout"])
        except BaseException:
            self.close()
            raise

    def send(self, method, payload):
        if self.process.poll() is not None:
            raise RuntimeError(f"PAP worker exited with status {self.process.returncode}; see {self.log.name}")
        self.connection.send_bytes(json.dumps({"method": method, "payload": payload}, allow_nan=False).encode())

    def receive(self, timeout):
        if not self.connection.poll(timeout):
            raise TimeoutError(f"PAP worker timed out; physical state is unknown; see {self.log.name}")
        value = json.loads(self.connection.recv_bytes(_MAX_MESSAGE))
        if not value.get("ok"):
            raise RuntimeError(f"PAP worker error: {value.get('error')}; see {self.log.name}")
        return value.get("result")

    def close(self):
        if self.connection is not None:
            try:
                self.connection.send_bytes(b'{"method":"close","payload":{}}')
            except (EOFError, OSError):
                pass
            self.connection.close()
        if self.process is not None:
            try:
                self.process.wait(timeout=10.)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5.)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5.)
        self.listener.close()
        self.log.close()


class PAPParallelBackend:
    def __init__(self, config):
        self.config = load_config(config)
        self.identity = task_identity(self.config)
        self.kin = PiperKinematics(self.config["scene"]["robot_urdf"], tip_offset=.1358)
        self.lower = np.r_[self.kin.lower, 0.]
        self.upper = np.r_[self.kin.upper, self.config["scene"]["gripper_max_width_m"]]
        self.workers, self._broken, self._closed = [], False, False
        self._ready = [False] * self.config["num_envs"]
        if "log_dir" in config:
            directory = _path(config["log_dir"])
            directory.mkdir(parents=True, exist_ok=True)
        else:
            directory = ROOT / "outputs" / "pap_rl_workers"
            directory.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="run_", dir=directory))
        self.directory = directory
        try:
            # Serial startup limits peak shader/asset initialization pressure.
            # Sampling commands are concurrent after every worker is ready.
            for i in range(self.config["num_envs"]):
                self.workers.append(_Worker(self.config, i, directory))
        except BaseException:
            self.close()
            raise

    def describe(self):
        cfg = self.config
        identity = getattr(self, "identity", None) or task_identity(cfg)
        return dict(schema_version=2, node_type="simulation", task_id="pap", num_envs=cfg["num_envs"],
            **identity, smoothing=copy.deepcopy(cfg["smoothing"]),
            observation={"state": {"names": NAMES.copy(), "units": UNITS.copy()}, "cameras": ["up", "wrist"]},
            action={"representation": "joint_position", "frame": "robot_base", "names": NAMES.copy(),
                    "units": UNITS.copy(), "horizon": cfg["horizon"], "control_hz": cfg["control_hz"],
                    "lower": self.lower.tolist(), "upper": self.upper.tolist()},
            control_state_dim=7, task_features_dim=0,
            discount={"gamma": cfg["gamma"], "timebase_hz": cfg["control_hz"]}, reset_mode="indexed",
            execution={"backend": "isaac_process_parallel", "workers": cfg["num_envs"],
                "gpu_ids": [w["gpu"] for w in cfg["workers"]], "shared_renderer": False,
                "memory_note": "Each environment owns a separate Isaac application and renderer",
                "smoothing": copy.deepcopy(cfg["smoothing"]), "max_episode_seconds": cfg["max_episode_seconds"],
                "physics_hz": 1/cfg["physics_dt"], "worker_logs": str(self.directory)},
            reward=copy.deepcopy(cfg["reward"]),
            reward_schema={"kind": "sparse_physical_event", "values": copy.deepcopy(cfg["reward"]),
                "within_chunk_discount": "gamma ** ((event_tick - 1) / physics_ticks_per_action)",
                "timeout_reward": 0., "success_requires_bilateral_contact_history": True},
            success_rule="lift + whole object in box + release + stable floor rest + bilateral contact history")

    def _ids(self, env_ids):
        if self._broken or self._closed:
            raise RuntimeError("PAP backend is closed or failed; state is unknown and commands will not be retried")
        if not isinstance(env_ids, list) or not env_ids or len(set(env_ids)) != len(env_ids):
            raise ValueError("env_ids must be a nonempty list of distinct environment ids")
        if any(type(i) is not int or not 0 <= i < len(self.workers) for i in env_ids):
            raise ValueError("Environment id out of range")

    def _parallel(self, requests):
        try:
            for env_id, method, payload in requests:
                self.workers[env_id].send(method, payload)
            # All workers are executing now; deadline bounds the whole cohort.
            deadline = time.monotonic() + self.config["command_timeout"]
            return [self.workers[i].receive(max(0., deadline-time.monotonic())) for i, _, _ in requests]
        except BaseException:
            self._broken = True
            raise

    def reset(self, env_ids, seed, options=None):
        self._ids(env_ids)
        options = {} if options is None else copy.deepcopy(options)
        per_env = options.pop("by_env", {})
        if not isinstance(per_env, dict):
            raise ValueError("reset options.by_env must be a mapping")
        requests = []
        for i in env_ids:
            selected = {**options, **per_env.get(str(i), per_env.get(i, {}))}
            # Sampling and all reset validation happen before any worker mutates.
            requests.append((i, "reset", sample_reset(self.config, seed, i, selected)))
        results = self._parallel(requests)
        for i in env_ids:
            self._ready[i] = True
        return results

    def step(self, items):
        if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
            raise ValueError("step items must be a list of action mappings")
        ids = [item["env_id"] for item in items]
        self._ids(ids)
        requests = []
        for item in items:
            i = item["env_id"]
            if not self._ready[i]:
                raise ValueError(f"Environment {i} requires reset before step")
            value = np.asarray(item["actions"])
            if (value.ndim != 2 or value.shape[1] != 7 or not 1 <= len(value) <= self.config["horizon"]
                    or value.dtype.kind not in "fiu" or not np.isfinite(value).all()):
                raise ValueError(f"Expected finite [1..{self.config['horizon']},7] actions in radians/metres")
            if np.any(value < self.lower-1e-6) or np.any(value > self.upper+1e-6):
                raise ValueError("PAP action exceeds URDF arm or gripper travel limits; no environment advanced")
            requests.append((i, "step", {"actions": value.tolist()}))
        results = self._parallel(requests)
        for i, value in zip(ids, results):
            if value["terminated"] or value["truncated"]:
                self._ready[i] = False
        return results

    def close(self):
        if self._closed:
            return
        self._closed = True
        for worker in self.workers:
            worker.close()


def create_backend(config):
    return PAPParallelBackend(config)
