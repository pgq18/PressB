"""Native elevator task using the qualified fast physics engine on one thread.

The legacy engine and indexed episode state machine remain unchanged. A local
dispatcher supplies their protocol to the schema-v2 task adapter without an
HTTP server or a separately launched simulation process.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import threading

from pressb.runtime.contract import fingerprint
from .backend import ElevatorBackend


ROOT = Path(__file__).resolve().parents[4]
_INPUT = ROOT / "outputs/rtx5090_eval_comparison/input"
_DEFAULTS = dict(gpu=0, num_envs=2, max_seconds=15., smoothing_window=3, single_gamma=.99,
    response_cache_size=2, camera_resolution=224, render_subframes=1,
    light_settle_captures=1, light_settle_subframes=2, ik_iterations=16,
    log_actions=False, png_workers=4, cpu_threads=8, gpu_dynamics=False)
_PATHS = dict(config=_INPUT / "config.json", snapshot=_INPUT / "scene.usda",
              dataset=_INPUT / "dataset_metadata", asset_bundle=_INPUT / "assets")
_ADAPTER_KEYS = {"floors", "panel_randomization", "run_id", "expected_metadata"}


def _path(value):
    path = Path(value).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def load_config(config):
    """Validate the launch and frozen scene before importing Isaac or Kit."""
    allowed = set(_DEFAULTS) | set(_PATHS) | _ADAPTER_KEYS | {"output"}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError(f"Native elevator configuration supports {sorted(allowed)}")
    values = {**_DEFAULTS, **copy.deepcopy(config)}
    for key in ("gpu", "num_envs", "response_cache_size", "render_subframes",
                "light_settle_captures", "light_settle_subframes", "ik_iterations",
                "png_workers", "cpu_threads"):
        minimum = 0 if key in ("gpu", "light_settle_captures") else 1
        if type(values[key]) is not int or values[key] < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    if type(values["smoothing_window"]) is not int or values["smoothing_window"] not in (1, 3, 5, 7, 9, 11):
        raise ValueError("smoothing_window must be one of 1,3,5,7,9,11")
    if type(values["camera_resolution"]) is not int or values["camera_resolution"] not in (224, 640):
        raise ValueError("camera_resolution must be 224 or 640")
    for key in ("max_seconds", "single_gamma"):
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
    if not 0 < values["max_seconds"] <= 120 or abs(values["max_seconds"] * 120 - round(values["max_seconds"] * 120)) > 1e-7:
        raise ValueError("max_seconds must be in (0,120] and on a 120 Hz tick")
    if not 0 < values["single_gamma"] <= 1:
        raise ValueError("single_gamma must be in (0,1]")
    for key in ("log_actions", "gpu_dynamics"):
        if type(values[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    for key, default in _PATHS.items():
        value = values.get(key, default)
        values[key] = None if key == "asset_bundle" and value is None else _path(value)
    if not isinstance(values.get("output"), (str, Path)) or not str(values["output"]):
        raise ValueError("Native elevator requires a new output directory")
    values["output"] = _path(values["output"])
    values["native_log_path"] = values["output"].with_name(values["output"].name + ".kit.log")
    for path in (values["output"], values["native_log_path"]):
        if path.exists():
            raise FileExistsError(f"Native elevator output must be new: {path}")
    scene = json.loads(values["config"].read_text())
    collection = json.loads((values["dataset"] / "meta/collection_metadata.json").read_text())
    scene_hash = hashlib.sha256(values["snapshot"].read_bytes()).hexdigest()
    if scene != collection["config"] or scene_hash != collection["scene_sha256"]:
        raise ValueError("Config and scene must match the frozen training collection")
    bundle = values["asset_bundle"]
    if bundle is not None and not (bundle / "asset_bundle.json").is_file():
        raise FileNotFoundError("asset_bundle.json is missing from the specified bundle")
    # Apply the bridge's cheap configuration validation before opening Kit.
    floors = values.get("floors", list(range(24, 36)))
    if (not isinstance(floors, (list, tuple)) or not floors
            or any(type(f) is not int or not 24 <= f <= 35 for f in floors)
            or len(set(floors)) != len(floors)):
        raise ValueError("floors must be distinct integers in 24..35")
    if type(values.get("panel_randomization", True)) is not bool:
        raise ValueError("panel_randomization must be boolean")
    if "run_id" in values and (not isinstance(values["run_id"], str)
            or not values["run_id"].strip() or len(values["run_id"]) > 256):
        raise ValueError("run_id must be a nonempty string of at most 256 characters")
    if not isinstance(values.get("expected_metadata", {}), dict):
        raise ValueError("expected_metadata must be a mapping")
    return values


class _LocalLegacyClient:
    """Direct main-thread dispatch with the same endpoint payloads as RPC."""

    def __init__(self, service):
        self.service = service
        self.owner = threading.get_ident()
        self.closed = False
        self.handlers = {"/health": service.health, "/reset": service.reset,
                         "/reset_envs": service.reset_envs, "/step": service.step,
                         "/close_run": service.close_run}

    def call(self, path, payload=None):
        if self.closed or threading.get_ident() != self.owner:
            raise RuntimeError("Native elevator calls require the owning simulator thread")
        if path not in self.handlers:
            raise ValueError(f"Unknown local elevator route: {path}")
        return self.handlers[path](copy.deepcopy(payload))

    def close(self):
        self.closed = True


class NativeElevatorBackend(ElevatorBackend):
    """Schema-v2 elevator backend owning its application and physical engine."""

    def __init__(self, config):
        self._app = self._engine = self._legacy_service = None
        self._native_closed = False
        self._owner = threading.get_ident()
        self.settings = load_config(config)
        try:
            self._start_runtime()
            bridge = {key: self.settings[key] for key in _ADAPTER_KEYS if key in self.settings}
            # The base adapter validates this string; the supplied local client
            # never opens it or constructs an RPC client.
            bridge["endpoint"] = "http://native.elevator.invalid"
            super().__init__(bridge, client=_LocalLegacyClient(self._legacy_service))
            self._metadata["execution"].update(backend="native_fast_isaac", transport="in_process",
                gpu=self.settings["gpu"], simulator_thread="caller_main_thread")
            sources = copy.deepcopy(self._engine.manifest.get("sources", {}))
            for name in ("src/pressb/tasks/elevator/native.py", "src/pressb/tasks/elevator/backend.py",
                         "src/pressb/runtime/contract.py", "src/pressb/control/pose_protocol.py",
                         "src/pressb/tasks/elevator/semantics.py", "src/pressb/tasks/elevator/collection.py"):
                contents = (ROOT / name).read_bytes()
                sources[name] = {"sha256": hashlib.sha256(contents).hexdigest(), "bytes": len(contents)}
            self._metadata["source_identity"] = sources
            self._metadata["source_revision"] = "sha256:" + fingerprint(
                {name: value["sha256"] for name, value in sources.items()})
        except BaseException:
            self._close_runtime(exit_code=1)
            raise

    def _start_runtime(self):
        cfg = self.settings
        cfg["native_log_path"].parent.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "8")
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(name, "1")
        cache_tmp = ROOT / ".cache/tmp"
        cache_tmp.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TMPDIR", str(cache_tmp))
        from isaacsim import SimulationApp
        from .collection import DATASET_ANTI_ALIASING
        self._app = SimulationApp(dict(headless=True, create_new_stage=False, width=640, height=480,
            active_gpu=cfg["gpu"], physics_gpu=cfg["gpu"], multi_gpu=False,
            disable_viewport_updates=True, limit_cpu_threads=cfg["cpu_threads"],
            renderer="RayTracedLighting", anti_aliasing=DATASET_ANTI_ALIASING, fast_shutdown=True,
            extra_args=["--/app/asyncRendering=false", "--/rtx/ecoMode/enabled=false",
                f"--/log/file={cfg['native_log_path']}", "--/log/fileFlushLevel=verbose", "--/log/fileAppend=false",
                "--/log/async=false", "--/plugins/carb.tasking.plugin/threadCount=8",
                "--/plugins/omni.tbb.globalcontrol/maxThreadCount=8", "--/persistent/physics/numThreads=4",
                "--/validate/p2p/enabled=false", "--/validate/iommu/enabled=false"]))
        from .fast_simulation import FastIsaacVectorBackend, IndexedSimulationService
        engine_keys = set(_DEFAULTS) - {"response_cache_size"}
        engine = {key: cfg[key] for key in engine_keys}
        engine.update({key: cfg[key] for key in (*_PATHS, "output", "native_log_path")})
        self._engine = FastIsaacVectorBackend(self._app, project_root=ROOT, **engine)
        self._legacy_service = IndexedSimulationService(self._engine,
            single_gamma=cfg["single_gamma"], response_cache_size=cfg["response_cache_size"])

    def _close_runtime(self, *, exit_code=0):
        if self._native_closed:
            return
        if threading.get_ident() != self._owner:
            raise RuntimeError("Native elevator must close on the owning simulator thread")
        self._native_closed = self._closed = True
        client = getattr(self, "client", None)
        if client is not None:
            client.close()
        try:
            if self._engine is not None:
                self._engine.close()
        except BaseException:
            exit_code = 1
            raise
        finally:
            if self._app is not None:
                try:
                    self._app.app.post_quit(exit_code)
                finally:
                    self._app.close()

    def close(self):
        self._close_runtime()


def create_backend(config):
    return NativeElevatorBackend(config)
