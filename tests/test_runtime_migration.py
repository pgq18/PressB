"""CPU-only canonical ownership and dependency boundaries for the runtime."""
import ast
import importlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import textwrap
from types import ModuleType

import pytest


RUNTIME_SYMBOLS = [
    ("contract", "negotiate"),
    ("client", "NodeClient"),
    ("inference", "InferenceService"),
    ("simulation", "SimulationService"),
    ("trainer", "GenericTrainer"),
    ("server", "load_backend"),
    ("image_transport", "ValidatedObservation"),
    ("learner", "SACLearner"),
    ("rpc", "RPCClient"),
]


@pytest.mark.parametrize("runtime_name,symbol", RUNTIME_SYMBOLS)
def test_runtime_symbols_belong_to_the_canonical_implementation(runtime_name, symbol):
    runtime = importlib.import_module(f"pressb.runtime.{runtime_name}")
    assert getattr(runtime, symbol).__module__ == f"pressb.runtime.{runtime_name}"
    assert Path(runtime.__file__) == (
        Path(__file__).resolve().parents[1] / "src/pressb/runtime" / (runtime_name + ".py"))


def test_runtime_wire_schema_version_is_unchanged():
    from pressb import runtime
    from pressb.runtime import contract

    assert runtime.SCHEMA_VERSION == contract.SCHEMA_VERSION == 2


def test_transport_primitives_and_legacy_model_adapters_have_separate_owners():
    from pressb.compat.elevator_rl import inference_transport as adapter
    from pressb.runtime import inference_transport as runtime

    for symbol in ("_pack", "_unpack", "_pack_array", "_unpack_array",
                   "_wire_payload", "InferenceClient", "OPERATIONS", "NAMESPACE"):
        assert hasattr(runtime, symbol)
        assert not hasattr(adapter, symbol)
    assert runtime.InferenceClient.__module__ == "pressb.runtime.inference_transport"
    for symbol in ("IdentityDataProcessor", "InferencePolicyAdapter", "load_wallx_server"):
        assert getattr(adapter, symbol).__module__ == "pressb.compat.elevator_rl.inference_transport"
        assert not hasattr(runtime, symbol)


def test_runtime_client_uses_the_shared_rpc_dependency(monkeypatch):
    from pressb.runtime import client as runtime

    created = []

    class FakeRPCClient:
        def __init__(self, endpoint, timeout, token):
            created.append((endpoint, timeout, token))

        def call(self, path, payload):
            return {"path": path, "payload": payload}

    monkeypatch.setattr(runtime, "RPCClient", FakeRPCClient)
    client = runtime.NodeClient("http://runtime.test", timeout=3, token="fixture")
    try:
        assert isinstance(client._http, FakeRPCClient)
        assert created == [("http://runtime.test", 3, "fixture")]
        assert client.call("/health", {"probe": True}) == {
            "path": "/health", "payload": {"probe": True},
        }
    finally:
        client.close()


@pytest.mark.parametrize("runtime_name,symbol,args", [
    ("trainer", "EndpointConfig", ()),
    ("learner", "LearnerConfig", ("initial_noise", 4)),
    ("rpc", "RPCError", ("saved failure",)),
    ("image_transport", "ValidatedObservation",
     ({"state": [1., 2.], "images": {}}, {})),
])
def test_runtime_instances_roundtrip_with_canonical_pickle_paths(runtime_name, symbol, args):
    runtime = importlib.import_module(f"pressb.runtime.{runtime_name}")
    cls = getattr(runtime, symbol)
    original = cls(*args)
    saved = pickle.dumps(original, protocol=4)
    assert runtime.__name__.encode() in saved
    restored = pickle.loads(saved)
    assert type(restored) is getattr(runtime, symbol)
    assert vars(restored) == vars(original)
    if isinstance(original, dict):
        assert restored == original
    if isinstance(original, BaseException):
        assert restored.args == original.args


def test_runtime_trainer_imports_without_task_model_or_simulator_dependencies():
    # A fresh interpreter catches accidental imports even if another test has
    # already loaded an adapter. It also keeps this check independent of order.
    source = textwrap.dedent("""
        import importlib.abc
        import sys
        import torch

        torch.set_num_threads(1)
        def no_cuda(*args, **kwargs):
            raise AssertionError("runtime import attempted CUDA initialization")
        torch.cuda._lazy_init = no_cuda

        forbidden = (
            "pressb.tasks", "pressb.models", "pressb.model_sources", "pressb.compat",
            "pressb.pick_place", "pressb.rl_nodes", "pressb.online_rl",
            "isaacsim", "omni", "pxr", "transformers",
        )
        class RejectAdapterImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if any(fullname == name or fullname.startswith(name + ".")
                       for name in forbidden):
                    raise AssertionError("runtime imported an adapter: " + fullname)

        sys.meta_path.insert(0, RejectAdapterImports())
        from pressb.runtime import trainer
        assert trainer.GenericTrainer.__module__ == "pressb.runtime.trainer"
        assert trainer.SACLearner.__module__ == "pressb.runtime.learner"
        assert not any(name == prefix or name.startswith(prefix + ".")
                       for name in sys.modules for prefix in forbidden)
    """)
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               CUDA_VISIBLE_DEVICES="")
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(root / "src"), env.get("PYTHONPATH"))))
    result = subprocess.run([sys.executable, "-c", source], cwd=root, env=env,
                            capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr


def test_runtime_trainer_has_no_deferred_task_or_model_imports():
    from pressb.runtime import trainer

    tree = ast.parse(Path(trainer.__file__).read_text())
    forbidden = ("pressb.tasks", "pressb.models", "pressb.model_sources", "pressb.compat",
                 "pressb.pick_place", "pressb.rl_nodes")
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            name = "." * node.level + (node.module or "")
            imported.append(importlib.util.resolve_name(name, trainer.__package__))
    assert not [name for name in imported if any(
        name == prefix or name.startswith(prefix + ".") for prefix in forbidden)]


@pytest.fixture
def backend_factory(monkeypatch):
    module = ModuleType("_pressb_runtime_test_backend")
    calls = []
    result = object()

    def build(config):
        calls.append(config)
        return result

    module.build = build
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return f"{module.__name__}:build", calls, result


@pytest.fixture
def fake_registry(monkeypatch):
    import pressb

    registry = ModuleType("pressb.registry")

    def unexpected_resolve(*args, **kwargs):
        pytest.fail("Invalid backend selection must not resolve a plugin")

    registry.resolve = unexpected_resolve
    monkeypatch.setitem(sys.modules, registry.__name__, registry)
    monkeypatch.setattr(pressb, "registry", registry, raising=False)
    return registry


def test_explicit_factory_loads_backend_with_legacy_call_signature(backend_factory):
    from pressb.runtime.server import load_backend

    factory, calls, result = backend_factory
    assert load_backend({"factory": factory, "config": {"marker": 7}}) is result
    assert load_backend({"factory": factory}) is result
    assert calls == [{"marker": 7}, {}]


@pytest.mark.parametrize("kind", ["task", "model"])
def test_plugin_loads_only_the_resolved_factory(monkeypatch, backend_factory, fake_registry, kind):
    from pressb.runtime.server import load_backend

    factory, calls, result = backend_factory
    resolved = []

    def resolve(requested_kind, name):
        resolved.append((requested_kind, name))
        return factory

    monkeypatch.setattr(fake_registry, "resolve", resolve)
    assert load_backend({"plugin": "fixture", "config": {"marker": 8}}, kind=kind) is result
    assert resolved == [(kind, "fixture")]
    assert calls == [{"marker": 8}]


@pytest.mark.parametrize("spec,kind", [
    (None, None),
    ({}, None),
    ({"factory": "fixture:build", "plugin": "fixture"}, "task"),
    ({"factory": "missing_colon"}, None),
    ({"factory": ":build"}, None),
    ({"factory": "fixture:"}, None),
    ({"factory": "fixture:build", "config": []}, None),
    ({"plugin": "fixture"}, None),
    ({"plugin": "fixture"}, "inference"),
    ({"plugin": "fixture"}, "simulation"),
])
def test_invalid_backend_selection_fails_before_loading(fake_registry, spec, kind):
    from pressb.runtime.server import load_backend

    with pytest.raises(ValueError):
        load_backend(spec, kind=kind)


@pytest.mark.parametrize("node_kind,plugin_kind,service_class", [
    ("inference", "model", "InferenceService"),
    ("simulation", "task", "SimulationService"),
    ("real_robot", "task", "SimulationService"),
])
def test_run_service_maps_node_kind_to_plugin_kind(
        tmp_path, monkeypatch, node_kind, plugin_kind, service_class):
    from pressb.runtime import server

    backend = object()
    backend_spec = {"plugin": "fixture", "config": {"marker": 9}}
    config = tmp_path / "node.json"
    config.write_text(json.dumps({
        "backend": backend_spec,
        "server": {"transport": "ws", "host": "runtime.test", "port": 9999,
                   "max_message_bytes": 4096},
        "service": {"fixture_option": True},
    }))
    loaded, created, served, closed = [], [], [], []

    def fake_load_backend(spec, *, kind):
        loaded.append((spec, kind))
        return backend

    class FakeService:
        def __init__(self, actual_backend, **options):
            created.append((actual_backend, options))
            self.handlers = {"/health": lambda _: {
                "node_type": "simulation" if node_kind == "real_robot" else node_kind,
                "environment_type": node_kind,
            }}

        def close(self):
            closed.append(self)

    async def fake_serve_websocket(service, host, port, token, *, max_message_bytes):
        served.append((service, host, port, token, max_message_bytes))

    def unexpected_http_server(*args, **kwargs):
        pytest.fail("Fixture service must not open an HTTP server")

    module_name = "simulation" if node_kind == "real_robot" else node_kind
    service_module = importlib.import_module(f"pressb.runtime.{module_name}")
    monkeypatch.setattr(server, "load_backend", fake_load_backend)
    monkeypatch.setattr(service_module, service_class, FakeService)
    monkeypatch.setattr(server, "serve_websocket", fake_serve_websocket)
    monkeypatch.setattr(server, "RPCServer", unexpected_http_server)
    server.run_service(node_kind, config)
    assert loaded == [(backend_spec, plugin_kind)]
    assert created == [(backend, {"fixture_option": True})]
    assert len(served) == 1
    service, *connection = served[0]
    assert isinstance(service, FakeService)
    assert connection == ["runtime.test", 9999, None, 4096]
    assert closed == [service]


def test_run_service_rejects_unknown_node_kind_before_loading_backend(tmp_path, monkeypatch):
    from pressb.runtime import server

    config = tmp_path / "node.json"
    config.write_text(json.dumps({"backend": {"plugin": "fixture"}}))

    def unexpected_backend(*args, **kwargs):
        pytest.fail("Unknown node kind must fail before backend loading")

    monkeypatch.setattr(server, "load_backend", unexpected_backend)
    with pytest.raises(ValueError, match="Unknown node kind"):
        server.run_service("unknown", config)
