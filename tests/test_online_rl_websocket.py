"""Real socket contracts for the Wall-X wire format and PressB extension.

These exercise the server imported by Wall-X's launch script and its original
client, without importing a Wall-X checkpoint or requiring a GPU.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
import socket
import threading
import time
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
import pytest

pytest.importorskip("websockets.asyncio.server")
pytest.importorskip("websockets.sync.client")
pytest.importorskip("msgpack")
pytest.importorskip("torch")

from pressb.online_rl.inference import InferenceApplication
from pressb.online_rl.inference_transport import (
    InferenceClient,
    InferencePolicyAdapter,
    load_wallx_server,
)
from pressb.online_rl.rpc import RPCError, RPCServer

# Share the real, CPU-only encoder/flow implementation used by adapter tests.
from test_online_rl_inference import backend, decode_request, encode_request, observation
from test_fast_online_rl_runner import (
    BatchInference,
    IndexedSimulation,
    config as runner_config,
    small_cpu_workload,
)
from test_online_rl_runner import TOKEN, serve
from pressb.online_rl.fast_runner import FastOnlineRunner


class RecordingProcessor:
    def __init__(self):
        self.calls = 0

    def in_process(self, obs):
        self.calls += 1
        # Applying this twice changes the value, so the assertions below detect
        # accidental duplicate normalization introduced by the new dispatch.
        obs["state"] = obs["state"] * 2 - 1
        return obs


class RecordingPolicy:
    def __init__(self):
        self.calls = []

    def infer(self, obs, prev_action=None, is_rtc=False):
        self.calls.append((obs, prev_action, is_rtc))
        return {"action": np.broadcast_to(obs["state"], (1, 4, 3)).copy()}


class IdentityProcessor:
    def in_process(self, obs):
        return obs


@contextmanager
def running_server(policy, processor, *, metadata=None, rpc_handler=None, api_key=None):
    """Run the actual server lifecycle and close every listener after the test."""
    server_class = load_wallx_server()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = server_class(policy, processor, host="127.0.0.1", port=port,
                          metadata=metadata, rpc_handler=rpc_handler, api_key=api_key)
    loop = asyncio.new_event_loop()
    pending = []
    errors = []

    def worker():
        asyncio.set_event_loop(loop)
        task = loop.create_task(server.run())
        pending.append(task)
        try:
            loop.run_until_complete(task)
        except asyncio.CancelledError:
            pass
        except BaseException as exc:
            errors.append(exc)
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    endpoint = f"ws://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 10
        while True:
            if errors:
                raise errors[0]
            try:
                with urlopen(f"http://127.0.0.1:{port}/healthz", timeout=.25) as response:
                    assert response.status == 200 and response.read() == b"OK\n"
                break
            except (URLError, OSError):
                if time.monotonic() >= deadline:
                    raise
                time.sleep(.01)
        yield endpoint
    finally:
        if pending and thread.is_alive():
            loop.call_soon_threadsafe(pending[0].cancel)
        thread.join(timeout=10)
        assert not thread.is_alive(), "WebSocket server did not shut down"
        if errors:
            raise errors[0]


def legacy_modules():
    # load_wallx_server sets up the same package path as inference/launch_server.py.
    load_wallx_server()
    from openpi_client import msgpack_numpy
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    from websockets.sync.client import connect
    return msgpack_numpy, WebsocketClientPolicy, connect


def legacy_observation():
    return {"state": np.array([.25, .5, .75], dtype=np.float32),
            "face_view": np.arange(36, dtype=np.uint8).reshape(3, 4, 3),
            "prompt": "pick up the block"}


@pytest.mark.parametrize("extended", [False, True])
def test_original_raw_and_rtc_clients_keep_normalization_shapes_and_timing(extended):
    codec, legacy_client, connect = legacy_modules()
    policy, processor = RecordingPolicy(), RecordingProcessor()
    extension_calls = []

    def extension(request):
        extension_calls.append(request)
        return {"protocol_version": 1, "extension": True}

    metadata = {"model": "legacy-wall-x", "rtc_config": {"enabled": True}}
    with running_server(policy, processor, metadata=metadata,
                        rpc_handler=extension if extended else None) as endpoint:
        with connect(endpoint, compression=None, max_size=None) as wire:
            assert codec.unpackb(wire.recv()) == metadata
            for i in range(2):
                wire.send(codec.packb(legacy_observation()))
                result = codec.unpackb(wire.recv())
                assert result["action"].shape == (1, 4, 3)
                assert result["action"].dtype == np.float32
                np.testing.assert_array_equal(result["action"][0, 0], [-.5, 0, .5])
                assert result["server_timing"]["infer_ms"] >= 0
                assert ("prev_total_ms" in result["server_timing"]) == (i > 0)
            if extended:
                request = {"version": 1, "method": "health", "payload": {}}
                wire.send(codec.packb({"__pressb_rpc__": request}))
                assert codec.unpackb(wire.recv()) == {"protocol_version": 1, "extension": True}
                assert extension_calls == [request]
                assert processor.calls == len(policy.calls) == 2

        client = legacy_client(host=endpoint)
        try:
            assert client.get_server_metadata() == metadata
            previous = np.arange(12, dtype=np.float32).reshape(1, 4, 3)
            result = client.infer(legacy_observation(), prev_action=previous, is_rtc=True)
            assert result["action"].shape == (1, 4, 3)
            np.testing.assert_array_equal(policy.calls[-1][1], previous)
            assert policy.calls[-1][2] is True
            result = client.infer(legacy_observation())
            assert "prev_total_ms" in result["server_timing"]
            assert policy.calls[-1][1] is None and policy.calls[-1][2] is False
        finally:
            client._ws.close()
    assert processor.calls == len(policy.calls) == 4
    for obs, _, _ in policy.calls:
        np.testing.assert_array_equal(obs["state"], [-.5, 0, .5])
        np.testing.assert_array_equal(obs["face_view"], legacy_observation()["face_view"])


def test_reserved_key_is_an_ordinary_observation_without_opt_in():
    codec, _, connect = legacy_modules()
    policy, processor = RecordingPolicy(), RecordingProcessor()
    with running_server(policy, processor) as endpoint:
        with connect(endpoint) as wire:
            wire.recv()
            obs = legacy_observation()
            obs["__pressb_rpc__"] = {"version": 999}
            wire.send(codec.packb(obs))
            assert "action" in codec.unpackb(wire.recv())
    assert policy.calls[0][0]["__pressb_rpc__"] == {"version": 999}
    assert processor.calls == 1


@pytest.mark.parametrize("envelope", [
    {"version": 2, "method": "health", "payload": {}},
    {"version": True, "method": "health", "payload": {}},
    {"version": 1, "method": "shutdown", "payload": {}},
    {"version": 1, "method": "health", "payload": []},
])
def test_invalid_extension_requests_do_not_reach_policy_or_break_legacy(envelope):
    codec, _, connect = legacy_modules()
    policy, processor = RecordingPolicy(), RecordingProcessor()
    called = []
    with running_server(policy, processor, rpc_handler=lambda value: called.append(value)) as endpoint:
        with connect(endpoint) as wire:
            wire.recv()
            wire.send(codec.packb({"__pressb_rpc__": envelope}))
            result = codec.unpackb(wire.recv())
            assert result["protocol_version"] == 1 and "error" in result
            assert called == [] and processor.calls == 0 and policy.calls == []
            wire.send(codec.packb(legacy_observation()))
            assert "action" in codec.unpackb(wire.recv())


@pytest.mark.parametrize("native_images", [False, True])
def test_split_inference_roundtrip_preserves_arrays_noise_and_cached_context(backend, native_images):
    app = InferenceApplication(backend, batch_size=2)
    adapter = InferencePolicyAdapter(app)
    expected = app.encode(encode_request(13, 29, include=True))["items"]
    with running_server(adapter, IdentityProcessor(), metadata=adapter.metadata,
                        rpc_handler=adapter.rpc_handler) as endpoint:
        client = InferenceClient(endpoint, timeout=5)
        try:
            assert client.call("/health")["ready"] is True
            payload = encode_request(13, 29, include=True)
            payload["seeds"] = np.asarray(payload["seeds"], dtype=np.int64)
            for obs in payload["observations"]:
                obs["state"] = np.asarray(obs["state"], dtype=np.float32)
                if native_images:
                    obs["images"] = {
                        "global": np.full((480, 640, 3), [230, 0, 0], dtype=np.uint8),
                        "wrist": np.full((480, 640, 3), [0, 220, 0], dtype=np.uint8),
                    }
            actual = client.call("/encode", payload)["items"]
            for got, want in zip(actual, expected):
                for key in ("feature", "state_pose9", "base_actions_pose9", "base_actions_pose8"):
                    np.testing.assert_array_equal(got[key], want[key])
            encoder_calls = backend.model.qwen_vl_interface.calls
            noises = [np.zeros((7, 9), dtype=np.float32),
                      np.arange(63, dtype=np.float32).reshape(7, 9) / 63]
            expected_decoded = app.decode({"items": [
                {"context_id": item["context_id"], "initial_noise": noise.tolist()}
                for item, noise in zip(expected, noises)]})["items"]
            decoded = client.call("/decode", {"items": [
                {"context_id": item["context_id"], "initial_noise": noise}
                for item, noise in zip(actual, noises)]})["items"]
            for got, want in zip(decoded, expected_decoded):
                np.testing.assert_array_equal(got["actions_pose9"], want["actions_pose9"])
                np.testing.assert_array_equal(got["actions_pose8"], want["actions_pose8"])
            assert backend.model.qwen_vl_interface.calls == encoder_calls
            assert client.call("/release", {"context_ids": [item["context_id"] for item in actual]})["released"] == 2
            with pytest.raises(RPCError, match="missing or expired"):
                client.call("/decode", decode_request(actual[0]["context_id"]))
            # Validation errors leave the persistent connection usable.
            assert client.call("/health")["ready"] is True
        finally:
            client.close()


def test_original_wallx_client_can_call_frozen_vla_without_extra_normalization(backend):
    _, legacy_client, _ = legacy_modules()
    app = InferenceApplication(backend)
    adapter = InferencePolicyAdapter(app)
    expected = app.encode(encode_request(37, include=True))["items"][0]
    obs = {"prompt": "Press 24 floor.", "state": np.asarray(observation()["state"], dtype=np.float32),
           "face_view": np.full((480, 640, 3), [230, 0, 0], dtype=np.uint8),
           "left_wrist_view": np.full((480, 640, 3), [0, 220, 0], dtype=np.uint8), "seed": 37}
    cache_count = len(app.contexts)
    with running_server(adapter, IdentityProcessor(), metadata=adapter.metadata,
                        rpc_handler=adapter.rpc_handler) as endpoint:
        client = legacy_client(host=endpoint)
        try:
            metadata = client.get_server_metadata()
            assert metadata["pressb_rl"]["version"] == 1
            assert set(metadata["pressb_rl"]["operations"]) == {"health", "encode", "decode", "release"}
            result = client.infer(obs)
            assert result["action"].shape == (1, 7, 9)
            np.testing.assert_array_equal(result["action"][0], expected["base_actions_pose9"])
            assert result["server_timing"]["infer_ms"] >= 0
            assert len(app.contexts) == cache_count
        finally:
            client._ws.close()
    with pytest.raises(ValueError, match="RTC|rtc"):
        adapter.infer(obs, is_rtc=True)
    with pytest.raises(ValueError, match="RTC|rtc|prev_action"):
        adapter.infer(obs, prev_action=np.zeros((1, 7, 9)))


def test_inference_client_retains_authenticated_http_transport():
    received = []

    def encode(payload):
        received.append(payload)
        return {"items": [{"context_id": "http-context"}]}

    server = RPCServer(("127.0.0.1", 0), {"/health": lambda _: {"ready": True}, "/encode": encode}, token="test-token")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = InferenceClient(f"http://127.0.0.1:{server.server_port}", timeout=5, token="test-token")
    try:
        assert client.call("/health") == {"ready": True, "protocol_version": 1}
        assert client.call("/encode", {"seeds": [17]})["items"] == [{"context_id": "http-context"}]
        assert received == [{"seeds": [17], "protocol_version": 1}]
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def test_old_server_without_split_capability_is_not_silently_used_for_rl():
    policy, processor = RecordingPolicy(), RecordingProcessor()
    with running_server(policy, processor, metadata={"model": "old-wall-x"}) as endpoint:
        with pytest.raises((RPCError, ValueError), match="[Cc]apabil|pressb_rl|[Ss]plit|[Pp]rotocol"):
            client = InferenceClient(endpoint, timeout=2)
            try:
                client.call("/health")
            finally:
                client.close()
    assert policy.calls == [] and processor.calls == 0


def test_websocket_api_key_works_for_both_clients_and_health_stays_public(backend):
    _, legacy_client, _ = legacy_modules()
    adapter = InferencePolicyAdapter(InferenceApplication(backend))
    with running_server(adapter, IdentityProcessor(), metadata=adapter.metadata,
                        rpc_handler=adapter.rpc_handler, api_key="test-token") as endpoint:
        with pytest.raises(RPCError):
            client = InferenceClient(endpoint, timeout=2, token="wrong")
            try:
                client.call("/health")
            finally:
                client.close()
        client = InferenceClient(endpoint, timeout=2, token="test-token")
        try:
            assert client.call("/health")["ready"] is True
        finally:
            client.close()
        original = legacy_client(host=endpoint, api_key="test-token")
        try:
            assert original.get_server_metadata()["pressb_rl"]["version"] == 1
        finally:
            original._ws.close()


def test_transport_failure_does_not_retry_or_fall_back_to_legacy(backend):
    adapter = InferencePolicyAdapter(InferenceApplication(backend))
    requests = []
    legacy_policy = RecordingPolicy()

    def fail_after_receiving(request):
        requests.append(request)
        raise RuntimeError("simulated backend failure after request receipt")

    with running_server(legacy_policy, IdentityProcessor(), metadata=adapter.metadata,
                        rpc_handler=fail_after_receiving) as endpoint:
        client = InferenceClient(endpoint, timeout=2)
        try:
            with pytest.raises(RPCError, match="simulated backend failure"):
                client.call("/encode", encode_request(3))
        finally:
            client.close()
    assert len(requests) == 1
    assert legacy_policy.calls == []


def test_dropped_connection_is_never_automatically_reconnected(backend, monkeypatch):
    import websockets.sync.client

    adapter = InferencePolicyAdapter(InferenceApplication(backend))
    original_connect = websockets.sync.client.connect
    attempts = []

    def record_connect(*args, **kwargs):
        attempts.append(args[0])
        return original_connect(*args, **kwargs)

    monkeypatch.setattr(websockets.sync.client, "connect", record_connect)
    with running_server(adapter, IdentityProcessor(), metadata=adapter.metadata,
                        rpc_handler=adapter.rpc_handler) as endpoint:
        client = InferenceClient(endpoint, timeout=2)
        assert client.call("/health")["ready"] is True
    try:
        for _ in range(2):
            with pytest.raises(RPCError, match="not retried"):
                client.call("/health")
        assert attempts == [endpoint]
    finally:
        client.close()


@pytest.mark.parametrize("batch_size", [1, 2])
def test_legacy_infer_does_not_evict_pending_split_context(backend, batch_size):
    _, legacy_client, _ = legacy_modules()
    app = InferenceApplication(backend, cache_size=1, batch_size=batch_size)
    adapter = InferencePolicyAdapter(app)
    with running_server(adapter, IdentityProcessor(), metadata=adapter.metadata,
                        rpc_handler=adapter.rpc_handler) as endpoint:
        split = InferenceClient(endpoint, timeout=5)
        legacy = legacy_client(host=endpoint)
        try:
            key = split.call("/encode", encode_request(51))["items"][0]["context_id"]
            expected = split.call("/decode", decode_request(key))["items"][0]
            # Two clients share a capacity-one cache. A one-shot prediction
            # must not insert a temporary context and evict the split request.
            result = legacy.infer({**observation(), "seed": 97})
            assert result["action"].shape == (1, 7, 9)
            assert list(app.contexts) == [key]
            actual = split.call("/decode", decode_request(key))["items"][0]
            np.testing.assert_array_equal(actual["actions_pose9"], expected["actions_pose9"])
            assert backend.model.qwen_vl_interface.calls == 2
            assert split.call("/release", {"context_ids": [key]})["released"] == 1
        finally:
            legacy._ws.close()
            split.close()


def test_all_advertised_feature_encodings_are_accepted(backend):
    adapter = InferencePolicyAdapter(InferenceApplication(backend))
    with running_server(adapter, IdentityProcessor(), metadata=adapter.metadata,
                        rpc_handler=adapter.rpc_handler) as endpoint:
        client = InferenceClient(endpoint, timeout=5)
        try:
            health = client.call("/health")
            assert health["feature_encodings"]
            # NumPy describes the response wire format, not a valid /encode
            # feature_encoding selector understood by InferenceApplication.
            assert "numpy" not in health["feature_encodings"]
            for encoding in health["feature_encodings"]:
                payload = {**encode_request(51), "feature_encoding": encoding}
                item = client.call("/encode", payload)["items"][0]
                assert "feature" in item or "feature_f32_b64" in item
                assert client.call("/release", {"context_ids": [item["context_id"]]})["released"] == 1
        finally:
            client.close()


@pytest.mark.parametrize("method,mode", [
    ("base", "eval"),
    ("action_residual", "train"),
    ("initial_noise", "train"),
])
def test_fast_runner_ws_inference_to_http_simulation_and_real_sac(tmp_path, method, mode):
    simulation, inference = IndexedSimulation(), BatchInference()
    adapter = InferencePolicyAdapter(inference)
    cfg = runner_config(method=method, mode=mode)
    with serve(simulation.handlers) as sim_url:
        with running_server(adapter, IdentityProcessor(), metadata=adapter.metadata,
                            rpc_handler=adapter.rpc_handler, api_key=TOKEN) as inference_url:
            runner = FastOnlineRunner(cfg, sim_url, inference_url, tmp_path / method,
                                      token=TOKEN, timeout=5, device="cpu")
            result = runner.run()

    assert result["state"] == "complete"
    assert runner.feature_encoding == "json"
    assert len(simulation.resets) == 1 and len(simulation.closed) == 1
    assert not inference.contexts and not runner.contexts and not runner.pending_releases
    assert len(inference.released) == len(inference.encoded)
    # Real HTTP serialization validates the NumPy→JSON boundary, particularly
    # base mode which forwards the WebSocket's native actions without residuals.
    for request in simulation.steps:
        for action in request["actions"]:
            assert isinstance(action["actions_pose8"], list)
            pose = np.asarray(action["actions_pose8"])
            assert pose.shape == (7, 8) and np.isfinite(pose).all()
            np.testing.assert_allclose(pose[:, 7], .008)
            np.testing.assert_allclose(np.linalg.norm(pose[:, 3:7], axis=1), 1)

    if mode == "eval":
        assert result["episodes"] == runner.scheduled_episodes == 5
        assert result["transitions"] == 7 and result["updates"] == 0
        assert runner.learner is None and runner.replay is None
    else:
        assert result["transitions"] == result["updates"] == len(runner.replay) == 6
        assert result["update_credit"] == 0
        assert result["episodes"] == 4 and result["successes"] == 3
        replay = runner.replay.state_dict()["arrays"]
        np.testing.assert_array_equal(replay["obs"][:, 0], [0, 100, 1000, 101, 2000, 102])
        np.testing.assert_array_equal(replay["next_obs"][:, 0], [1, 101, 1001, 102, 2001, 103])
        np.testing.assert_array_equal(replay["terminated"][:, 0], [True, False, True, False, True, False])
        np.testing.assert_array_equal(replay["truncated"][:, 0], [False, False, False, False, False, True])
        np.testing.assert_allclose(replay["discount"][:, 0], .99 ** (np.array([3, 28, 3, 28, 3, 5]) / 4))
        assert runner.learner.q_optimizer.state and runner.learner.actor_optimizer.state
        assert (runner.output / "last.pt").is_file()
    assert bool(inference.decoded) == (method == "initial_noise")
