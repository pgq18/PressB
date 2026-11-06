"""Real socket tests for independent HTTP and MessagePack node processes."""
from contextlib import contextmanager
import asyncio
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time

import numpy as np
import pytest

from pressb.runtime.inference_transport import _pack, _unpack
from pressb.runtime.rpc import MAX_BODY_BYTES, RPCError, RPCServer
from pressb.runtime.client import NodeClient
from pressb.runtime.contract import encode_image
from pressb.runtime.server import run_service, serve_websocket
from pressb.runtime.simulation import SimulationService


class SimBackend:
    def __init__(self):
        self.calls = []

    def describe(self):
        return dict(schema_version=2, node_type="simulation", task_id="socket-test", num_envs=2,
                    observation=dict(state=dict(names=["joint1"], units=["rad"]), cameras=[]),
                    action=dict(representation="joint_position", frame="robot_base", names=["joint1"],
                                units=["rad"], horizon=3, control_hz=30),
                    control_state_dim=0, task_features_dim=0, reset_mode="indexed",
                    discount=dict(gamma=.99, timebase_hz=30))

    @staticmethod
    def observation(value=0):
        return dict(state=[value], task="Move the joint.", images={}, control_state=[], task_features=[])

    def reset(self, env_ids, seed, options):
        self.calls.append(("reset", threading.get_ident(), list(env_ids)))
        return [dict(env_id=eid, observation=self.observation()) for eid in env_ids]

    def step(self, commands):
        self.calls.append(("step", threading.get_ident(), [item["env_id"] for item in commands]))
        return [dict(env_id=item["env_id"], observation=self.observation(item["actions"][-1][0]),
                     reward=1., discount=.99 ** len(item["actions"]), terminated=True, truncated=False,
                     info=dict(elapsed_control_steps=len(item["actions"]),
                               physics_steps=4*len(item["actions"]), success=True)) for item in commands]


@contextmanager
def http_node(service, token=None):
    with RPCServer(("127.0.0.1", 0), service.handlers, token=token) as server:
        worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
        worker.start()
        client = NodeClient(f"http://127.0.0.1:{server.server_port}", timeout=2, token=token)
        try:
            yield client, server
        finally:
            client.close()
            server.shutdown()
            worker.join(timeout=2)


def test_real_http_reset_step_retry_and_indexed_reset_are_serial():
    backend = SimBackend()
    service = SimulationService(backend)
    with http_node(service, token="socket-secret") as (client, _):
        assert client.call("/health")["num_envs"] == 2
        reset = client.call("/reset", dict(run_id="run", request_id="r1", env_ids=[0, 1], seed=4))
        first = reset["items"][0]
        request = dict(run_id="run", request_id="s1", items=[dict(
            env_id=0, episode_id=first["episode_id"], step_id=0, actions=[[.2], [.3]])])
        result = client.call("/step", request)
        assert result["items"][0]["terminated"] is True
        assert result["items"][0]["info"]["elapsed_control_steps"] == 2
        assert client.call("/step", request) == result
        reset_again = client.call("/reset", dict(run_id="run", request_id="r2", env_ids=[0], seed=6))
        assert reset_again["items"][0]["episode_id"] != first["episode_id"]
        status = client.call("/status")
        assert status["environments"]["1"]["step_id"] == 0
        assert status["environments"]["1"]["episode_id"] == reset["items"][1]["episode_id"]
        assert [row[0] for row in backend.calls] == ["reset", "step", "reset"]
        assert len({row[1] for row in backend.calls}) == 1


def test_http_auth_and_failure_close_the_client_without_retry():
    backend = SimBackend()
    service = SimulationService(backend)
    with http_node(service, token="socket-secret") as (_, server):
        client = NodeClient(f"http://127.0.0.1:{server.server_port}", timeout=2, token="wrong")
        with pytest.raises(RPCError, match="401"):
            client.call("/reset", dict(run_id="run", request_id="r1", env_ids=[0], seed=4))
        with pytest.raises(RPCError, match="closed or failed"):
            client.call("/health")
        assert not backend.calls


_WS_PROCESS = r'''
import asyncio, json, sys
import numpy as np
from pressb.runtime.inference import InferenceService
from pressb.runtime.server import serve_websocket

class Backend:
    def describe(self):
        return dict(schema_version=2,node_type="inference",model_id="transport-tiny",feature_dim=2,
                    capabilities=["encode","decode","infer","features","context_cache","external_initial_noise"],
                    observation=dict(state=dict(names=["joint1"],units=["rad"]),cameras=["up"]),
                    action=dict(representation="joint_position",frame="robot_base",names=["joint1"],
                                units=["rad"],horizon=3,control_hz=30),noise=dict(shape=[3,2]))
    def encode_batch(self, observations, seeds):
        return [((float(obs["state"][0]),int(seed)),np.array([obs["state"][0],seed],dtype=np.float32))
                for obs,seed in zip(observations,seeds)]
    def decode_batch(self, contexts, noises):
        return [np.full((3,1),context[0]+context[1],dtype=np.float32) if noise is None else
                np.asarray(noise).sum(-1,keepdims=True)+context[0] for context,noise in zip(contexts,noises)]

port,token,unsupported,max_message_bytes,metadata_padding,image_optimizations=json.loads(sys.argv[1])
service=InferenceService(Backend(),cache_size=4,batch_size=2)
if image_optimizations:
    from pressb.runtime.image_transport import walloss_resize_spec
    service.metadata.update(image_encodings=["png_base64", "png_bytes"], image_mode="RGB",
                            image_preprocessing=walloss_resize_spec(256))
if unsupported:
    service.metadata["schema_version"]=1
if metadata_padding:
    service.metadata["padding"]="x"*metadata_padding
asyncio.run(serve_websocket(service,"127.0.0.1",port,token,max_message_bytes=max_message_bytes))
'''


@contextmanager
def ws_node(tmp_path, *, token="socket-secret", unsupported=False,
            max_message_bytes=MAX_BODY_BYTES, metadata_padding=0, image_optimizations=False):
    root = Path(__file__).resolve().parents[1]
    executable = root / ".conda/envs/wallx-inference/bin/python"
    if not executable.exists():
        pytest.skip("isolated Wall-X websockets>=14 environment is not available")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src"), "PYTHONNOUSERSITE": "1"}
    log_path = tmp_path / "websocket.log"
    with log_path.open("w") as log:
        process = subprocess.Popen([str(executable), "-u", "-c", _WS_PROCESS,
                                    json.dumps([port, token, unsupported, max_message_bytes,
                                                metadata_padding, image_optimizations])], cwd=root, env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    pytest.fail("WebSocket child exited: " + log_path.read_text())
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=.1):
                        break
                except OSError:
                    time.sleep(.02)
            else:
                pytest.fail("WebSocket child did not bind: " + log_path.read_text())
            yield f"ws://127.0.0.1:{port}"
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def _observation(value=.1):
    return dict(task="Move the joint.", state=[value],
                images={"up": encode_image(np.zeros((4, 4, 3), dtype=np.uint8))})


def test_websocket_15_server_and_websocket_12_client_wire_compatibility(tmp_path):
    with ws_node(tmp_path) as endpoint:
        client = NodeClient(endpoint, timeout=3, token="socket-secret")
        try:
            assert client.call("/health")["schema_version"] == 2
            assert client.metadata["model_id"] == "transport-tiny"
            encoded = client.call("/encode", dict(observations=[_observation()], seeds=[42], include_base_actions=True))
            item = encoded["items"][0]
            np.testing.assert_allclose(item["base_actions"], 42.1)
            decoded = client.call("/decode", dict(items=[dict(context_id=item["context_id"],
                                                               initial_noise=np.ones((3, 2), dtype=np.float32))]))
            np.testing.assert_allclose(decoded["items"][0]["actions"], 2.1)
            inferred = client.call("/infer", dict(observations=[_observation()], seeds=[42]))
            np.testing.assert_array_equal(inferred["items"][0]["actions"], item["base_actions"])
            assert client.call("/release", dict(context_ids=[item["context_id"]]))["released"] == 1
            assert client.call("/health")["cache"]["size"] == 0
        finally:
            client.close()


def test_native_messagepack_envelope_auth_and_structured_errors(tmp_path):
    from websockets.sync.client import connect
    with ws_node(tmp_path) as endpoint:
        unauthorized = NodeClient(endpoint, timeout=3, token="wrong")
        with pytest.raises(Exception, match="401"):
            unauthorized.call("/health")
        with pytest.raises(RPCError, match="closed or failed"):
            unauthorized.call("/health")
        with connect(endpoint, additional_headers={"Authorization": "Api-Key socket-secret"}, compression=None) as ws:
            assert _unpack(ws.recv())["schema_version"] == 2
            ws.send(_pack({"__pressb_rpc__": {"version": 1, "method": "health", "payload": {}}}))
            assert _unpack(ws.recv())["protocol_version"] == 1
            ws.send(_pack({"__pressb_rpc__": {"version": 1, "method": "not_a_method", "payload": {}}}))
            error = _unpack(ws.recv())
            assert "Unknown node operation" in error["error"]
            assert error["protocol_version"] == 1


def test_websocket_unsupported_schema_is_rejected_before_request(tmp_path):
    with ws_node(tmp_path, unsupported=True) as endpoint:
        client = NodeClient(endpoint, timeout=3, token="socket-secret")
        with pytest.raises(RPCError, match="schema_version=2"):
            client.call("/health")
        with pytest.raises(RPCError, match="closed or failed"):
            client.call("/health")


@pytest.mark.parametrize("limit", [0, -1, True, False, 1.5, "1024", None])
def test_message_limit_requires_positive_integer_before_backend_loading(tmp_path, monkeypatch, limit):
    from pressb.runtime import server
    with pytest.raises(ValueError, match="max_message_bytes must be a positive integer"):
        NodeClient("ws://127.0.0.1:1", max_message_bytes=limit)
    with pytest.raises(ValueError, match="max_message_bytes must be a positive integer"):
        asyncio.run(serve_websocket(None, "127.0.0.1", 1, max_message_bytes=limit))
    config = tmp_path / "node.json"
    config.write_text(json.dumps({"server": {"max_message_bytes": limit}, "backend": {}}))
    def unexpected_backend(_):
        pytest.fail("Invalid transport config must fail before loading the backend")
    monkeypatch.setattr(server, "load_backend", unexpected_backend)
    with pytest.raises(ValueError, match="max_message_bytes must be a positive integer"):
        run_service("inference", config)


@pytest.mark.parametrize("configured", [None, 192 * 1024 * 1024])
def test_node_config_passes_websocket_message_limit(tmp_path, monkeypatch, configured):
    from pressb.runtime import server
    server_config = {"transport": "ws", "host": "127.0.0.1", "port": 8101}
    if configured is not None:
        server_config["max_message_bytes"] = configured
    config = tmp_path / "node.json"
    config.write_text(json.dumps({"server": server_config, "backend": {}}))
    limits = []
    async def fake_serve(service, host, port, token=None, *, max_message_bytes):
        assert service.health()["node_type"] == "simulation"
        limits.append(max_message_bytes)
    monkeypatch.setattr(server, "load_backend", lambda _: SimBackend())
    monkeypatch.setattr(server, "serve_websocket", fake_serve)
    run_service("simulation", config)
    assert limits == [MAX_BODY_BYTES if configured is None else configured]
    client = NodeClient("ws://127.0.0.1:1", **({} if configured is None else {
        "max_message_bytes": configured}))
    assert client.max_message_bytes == limits[0]
    client.close()


def test_custom_websocket_limit_allows_complete_batch(tmp_path):
    # Use small budgets to exercise a real binary message without allocating
    # hundreds of MiB or changing the original observation batch grouping.
    with ws_node(tmp_path, max_message_bytes=8192) as endpoint:
        client = NodeClient(endpoint, timeout=3, token="socket-secret", max_message_bytes=8192)
        observations = [_observation(.1), _observation(.2)]
        for observation in observations:
            observation["task"] = "x" * 2048
        try:
            result = client.call("/encode", dict(observations=observations, seeds=[1, 2]))
            assert len(result["items"]) == 2
            np.testing.assert_allclose(result["items"][0]["features"], [.1, 1])
            np.testing.assert_allclose(result["items"][1]["features"], [.2, 2])
            assert client.call("/health")["encoded_count"] == 2
        finally:
            client.close()


def test_client_message_limit_rejects_before_send_and_does_not_retry(tmp_path):
    with ws_node(tmp_path, max_message_bytes=8192) as endpoint:
        client = NodeClient(endpoint, timeout=3, token="socket-secret", max_message_bytes=1024)
        observation = _observation()
        observation["task"] = "x" * 2048
        with pytest.raises(ValueError, match="max_message_bytes=1024"):
            client.call("/encode", dict(observations=[observation], seeds=[1]))
        with pytest.raises(RPCError, match="closed or failed"):
            client.call("/health")
        inspector = NodeClient(endpoint, timeout=3, token="socket-secret")
        try:
            assert inspector.call("/health")["encoded_count"] == 0
        finally:
            inspector.close()


def test_server_message_limit_rejects_before_model_execution(tmp_path):
    from websockets.exceptions import ConnectionClosed
    with ws_node(tmp_path, max_message_bytes=1024) as endpoint:
        client = NodeClient(endpoint, timeout=3, token="socket-secret", max_message_bytes=8192)
        observation = _observation()
        observation["task"] = "x" * 2048
        with pytest.raises(ConnectionClosed, match="1009"):
            client.call("/encode", dict(observations=[observation], seeds=[1]))
        with pytest.raises(RPCError, match="closed or failed"):
            client.call("/health")
        inspector = NodeClient(endpoint, timeout=3, token="socket-secret")
        try:
            assert inspector.call("/health")["encoded_count"] == 0
        finally:
            inspector.close()


def test_client_message_limit_applies_to_received_metadata(tmp_path):
    from websockets.exceptions import ConnectionClosed
    with ws_node(tmp_path, max_message_bytes=8192, metadata_padding=2048) as endpoint:
        client = NodeClient(endpoint, timeout=3, token="socket-secret", max_message_bytes=1024)
        with pytest.raises(ConnectionClosed, match="1009"):
            client.call("/health")
        with pytest.raises(RPCError, match="closed or failed"):
            client.call("/health")


def test_websocket_message_limit_does_not_cap_http_payload(tmp_path):
    backend = SimBackend()
    with http_node(SimulationService(backend)) as (_, server):
        client = NodeClient(f"http://127.0.0.1:{server.server_port}", timeout=2, max_message_bytes=1)
        try:
            assert client.call("/health")["num_envs"] == 2
            assert len(client.call("/reset", dict(run_id="run", request_id="r1", env_ids=[0, 1],
                                                   seed=4))["items"]) == 2
        finally:
            client.close()


def test_negotiated_binary_and_pre_resize_real_websocket_round_trip(tmp_path):
    with ws_node(tmp_path, image_optimizations=True) as endpoint:
        legacy = NodeClient(endpoint, timeout=3, token="socket-secret")
        optimized = NodeClient(endpoint, timeout=3, token="socket-secret",
                               image_transport="auto", preprocess_images=True, image_workers=4)
        observation = _observation()
        observation["images"]["up"] = encode_image(
            np.random.default_rng(8).integers(0, 256, (480, 640, 3), dtype=np.uint8))
        payload = dict(observations=[observation], seeds=[4])
        try:
            original = legacy.call("/infer", payload)
            compact = optimized.call("/infer", payload)
            assert compact == original
            assert optimized.last_call_timing["request_bytes"] < legacy.last_call_timing["request_bytes"] / 4
            assert observation["images"]["up"]["encoding"] == "png_base64"
        finally:
            legacy.close()
            optimized.close()


@pytest.mark.parametrize("kwargs", [{"image_transport": "binary"}, {"preprocess_images": 1},
                                   {"preprocess_images": "true"}, {"image_workers": 0},
                                   {"image_workers": True}, {"image_workers": 1.5}])
def test_image_transport_configuration_is_strict(kwargs):
    with pytest.raises(ValueError):
        NodeClient("ws://127.0.0.1:1", **kwargs)
