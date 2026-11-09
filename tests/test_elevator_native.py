"""CPU compatibility checks for the explicitly selected historical adapter."""
from copy import deepcopy
import hashlib
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from pressb.tasks.elevator.fast_simulation import IndexedSimulationService
from pressb.control.pose_protocol import pose8_to_pose9
from pressb.runtime.simulation import SimulationService
from pressb.tasks.elevator import backend, legacy_native as native
from test_rl_nodes_elevator import PhysicsFixture, STATE


@pytest.fixture
def config(tmp_path):
    scene = tmp_path / "scene.usda"
    scene.write_text("#usda 1.0\n")
    cfg = tmp_path / "config.json"
    cfg.write_text("{}")
    dataset = tmp_path / "dataset"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta/collection_metadata.json").write_text(json.dumps({
        "config": {}, "scene_sha256": hashlib.sha256(scene.read_bytes()).hexdigest()}))
    return {"config": str(cfg), "snapshot": str(scene), "dataset": str(dataset),
            "asset_bundle": None, "output": str(tmp_path / "sim")}


@pytest.mark.parametrize("change", [
    {"gpu": True}, {"num_envs": 0}, {"camera_resolution": 320}, {"max_seconds": .011},
    {"single_gamma": 1.01}, {"gpu_dynamics": 1}, {"smoothing_window": 2},
    {"floors": [23]}, {"panel_randomization": "yes"}, {"endpoint": "http://unused"},
])
def test_invalid_native_launch_is_rejected_before_runtime_start(config, monkeypatch, change):
    def must_not_start(self):
        raise AssertionError("Started the simulator for invalid configuration")
    monkeypatch.setattr(native.NativeElevatorBackend, "_start_runtime", must_not_start)
    with pytest.raises(ValueError):
        native.NativeElevatorBackend({**config, **change})


def test_existing_outputs_and_changed_frozen_scene_are_rejected(config, tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        native.load_config({**config, "output": str(existing)})
    (tmp_path / "config.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="frozen training"):
        native.load_config(config)


@pytest.fixture
def owned_backend(config, monkeypatch):
    events = []

    def start(self):
        self._app = SimpleNamespace(app=SimpleNamespace(post_quit=lambda code: events.append(("quit", code))),
                                    close=lambda: events.append(("app_close",)))
        self._engine = PhysicsFixture()
        self._engine.manifest = {"sources": {"engine.py": {"sha256": "engine-digest"}}}
        self._engine.close = lambda: events.append(("engine_close",))
        self._legacy_service = IndexedSimulationService(self._engine, single_gamma=.99)

    def no_rpc(*args, **kwargs):
        raise AssertionError("Native task must not open an RPC client")
    monkeypatch.setattr(native.NativeElevatorBackend, "_start_runtime", start)
    monkeypatch.setattr(backend, "RPCClient", no_rpc)
    value = native.NativeElevatorBackend(config)
    yield value, events
    value.close()


def test_native_factory_uses_schema2_service_and_direct_legacy_dispatch(owned_backend):
    value, events = owned_backend
    metadata = value.describe()
    assert metadata["execution"]["backend"] == "native_fast_isaac"
    assert metadata["execution"]["transport"] == "in_process"
    assert metadata["scheduling"] == "all_active" and metadata["execution_horizons"] == [7]
    assert "src/pressb/tasks/elevator/native.py" in metadata["source_identity"]
    assert metadata["source_identity"]["engine.py"]["sha256"] == "engine-digest"
    service = SimulationService(value)
    initial = service.reset({"run_id": "native-test", "request_id": "r0",
        "env_ids": [0, 1], "seed": 7, "options": {}})
    value._engine.events = {0: (True, 5), 1: (True, 9)}
    chunk = pose8_to_pose9(np.tile(STATE, (7, 1))).tolist()
    request = {"run_id": "native-test", "request_id": "s0", "items": [
        {key: row[key] for key in ("env_id", "episode_id", "step_id")} | {"actions": chunk}
        for row in initial["items"]]}
    ended = service.step(request)
    assert service.step(deepcopy(request)) == ended
    assert len([call for call in value._engine.calls if call[0] == "step"]) == 1
    assert [row["discount"] for row in ended["items"]] == pytest.approx([.99 ** 1.25, .99 ** 2.25])
    peer = deepcopy(value._items[1])
    service.reset({"run_id": "native-test", "request_id": "r1",
                   "env_ids": [0], "seed": 8, "options": {"floor": 33}})
    assert value._items[1] == peer
    service.close()
    value.close()
    assert events == [("engine_close",), ("quit", 0), ("app_close",)]


def test_native_dispatch_rejects_other_threads_before_physics(owned_backend):
    value, _ = owned_backend
    errors = []
    def other_thread():
        try:
            value.client.call("/health")
        except RuntimeError as error:
            errors.append(str(error))
    worker = threading.Thread(target=other_thread)
    worker.start()
    worker.join()
    assert errors and "owning simulator thread" in errors[0]
    assert not value._engine.calls


def test_app_closes_even_if_engine_close_fails(owned_backend):
    value, events = owned_backend
    def failed_close():
        raise RuntimeError("engine close failed")
    value._engine.close = failed_close
    with pytest.raises(RuntimeError, match="engine close failed"):
        value.close()
    assert events == [("quit", 1), ("app_close",)]


def test_initialization_failure_releases_app(config, monkeypatch):
    events = []
    def start(self):
        self._app = SimpleNamespace(app=SimpleNamespace(post_quit=lambda code: events.append(code)),
                                    close=lambda: events.append("closed"))
        raise ValueError("failed startup")
    monkeypatch.setattr(native.NativeElevatorBackend, "_start_runtime", start)
    with pytest.raises(ValueError, match="failed startup"):
        native.NativeElevatorBackend(config)
    assert events == [1, "closed"]
