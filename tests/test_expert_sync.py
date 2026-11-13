"""Action-expert protocol/atomicity tests, not robot task-success evidence.

The tiny modules make every changed weight and frozen encoder byte observable.
HTTP and WebSocket cases exercise the actual node transports on local sockets.
"""
from contextlib import contextmanager
from copy import deepcopy
import asyncio
import base64
import hashlib
import io
import json
import socket
import threading
import time

import numpy as np
import pytest
import torch
from torch import nn

from pressb.runtime.client import NodeClient
from pressb.runtime.expert_sync import (
    ExpertWeights, publish_expert, state_sha256, tree_from_wire, tree_to_wire,
)
from pressb.runtime.inference import InferenceService
from pressb.runtime.inference_transport import _pack, _unpack
from pressb.runtime.rpc import RPCError, RPCServer
from pressb.runtime.server import serve_websocket


class ToyExpert(nn.Module):
    horizon, latent_dim, denoising_steps = 3, 2, 2

    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(2))
        self.invalidations = 0

    def describe(self):
        return dict(kind="protocol-test", horizon=3, latent_dim=2, denoising_steps=2)

    def pack_contexts(self, contexts):
        # Tuples (including nested tuples) and bf16 occur in model caches.
        return (torch.cat(contexts).to(torch.bfloat16), {"cache": (torch.ones(len(contexts), 1),)})

    def transition(self, x, steps, conditioning):
        assert isinstance(conditioning, tuple)
        assert isinstance(conditioning[1]["cache"], tuple)
        assert conditioning[0].dtype == torch.bfloat16
        return x * .25 + conditioning[0].float()[:, None, :] + self.bias, .1

    def actions_from_latent(self, x):
        return x

    def invalidate_cache(self):
        self.invalidations += 1


class ToyBackend:
    def __init__(self):
        self.encoder = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.encoder.weight.copy_(torch.eye(2))
        self.encoder.requires_grad_(False).eval()
        self.expert = ToyExpert()
        self.batches = []
        self.closed = False

    def describe(self):
        return dict(
            schema_version=2, node_type="inference", model_id="sync-protocol-test", feature_dim=2,
            frozen=True, capabilities=["encode", "decode", "infer", "external_initial_noise",
                                       "features", "context_cache"],
            observation=dict(state=dict(names=["j1", "j2"], units=["rad", "rad"]), cameras=[]),
            action=dict(representation="joint_position", frame="robot_base", names=["j1", "j2"],
                        units=["rad", "rad"], horizon=3, control_hz=30), noise=dict(shape=[3, 2]))

    def create_action_expert(self, config):
        return self.expert

    def encode_batch(self, observations, seeds):
        self.batches.append(len(observations))
        with torch.no_grad():
            return [(value, value[0].numpy()) for value in (
                self.encoder(torch.tensor(obs["state"], dtype=torch.float32)[None]) for obs in observations)]

    def decode_batch(self, contexts, initial_noises):
        with torch.no_grad():
            return [(context + self.expert.bias).expand(3, -1).numpy().copy() for context in contexts]

    def close(self):
        self.closed = True


def observation(value=.2):
    return dict(task="protocol test", state=[value, .4], images={})


def chain_request(version=0, count=3, deterministic=False):
    return dict(observations=[observation(.1 * i) for i in range(count)],
                seeds=list(range(31, 31 + count)), expected_policy_version=version,
                deterministic=deterministic)


def snapshot(expert):
    return {name: value.detach().clone() for name, value in expert.state_dict().items()}


def assert_state(expert, state):
    assert set(expert.state_dict()) == set(state)
    for name, value in expert.state_dict().items():
        assert torch.equal(value, state[name])


def upload_payload(state, **overrides):
    stream = io.BytesIO()
    torch.save(state, stream)
    blob = stream.getvalue()
    data = dict(expected_version=0, new_version=1, size=len(blob),
                sha256=hashlib.sha256(blob).hexdigest(), weights_sha256=state_sha256(state))
    data.update(overrides)
    return data, blob


def stage(owner, state, **overrides):
    payload, blob = upload_payload(state, **overrides)
    receipt = owner.begin(payload)
    owner.append(dict(upload_id=receipt["upload_id"], offset=0,
                      data=base64.b64encode(blob).decode()))
    return dict(upload_id=receipt["upload_id"])


class LocalClient:
    def __init__(self, service):
        self.service = service
        self.calls = []

    def call(self, route, payload=None):
        self.calls.append((route, deepcopy(payload)))
        return self.service.handlers[route](payload or {})


@contextmanager
def running_node(service, transport):
    if transport == "http":
        with RPCServer(("127.0.0.1", 0), service.handlers, token="test-token") as server:
            thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
            thread.start()
            try:
                yield f"http://127.0.0.1:{server.server_port}"
            finally:
                server.shutdown()
                thread.join(3)
                assert not thread.is_alive()
        return

    pytest.importorskip("websockets.asyncio.server", reason="WebSocket serving requires websockets >=14")
    # Execute the production asynchronous server, with its own owner thread.
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        port = reserve.getsockname()[1]
    loop = asyncio.new_event_loop()
    tasks, errors = [], []

    def worker():
        asyncio.set_event_loop(loop)
        task = loop.create_task(serve_websocket(service, "127.0.0.1", port, "test-token"))
        tasks.append(task)
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
            probe = NodeClient(endpoint, timeout=.5, token="test-token")
            try:
                probe.call("/health")
                break
            except (OSError, TimeoutError):
                if time.monotonic() >= deadline:
                    raise
                time.sleep(.02)
            finally:
                probe.close()
        yield endpoint
    finally:
        if tasks and thread.is_alive():
            loop.call_soon_threadsafe(tasks[0].cancel)
        thread.join(5)
        assert not thread.is_alive(), "WebSocket server did not terminate"
        if errors:
            raise errors[0]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float64,
                                   torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool])
@pytest.mark.parametrize("shape", [(), (0, 3), (2, 3)])
def test_tensor_wire_preserves_dtype_shape_and_exact_bytes(dtype, shape):
    source = torch.arange(6).reshape(2, 3).to(dtype)
    source = (source[0, 0] if shape == () else source[:0] if shape == (0, 3) else source.T.contiguous().T)
    for transport_copy in (lambda x: json.loads(json.dumps(x)), lambda x: _unpack(_pack(x))):
        result = tree_from_wire(transport_copy(tree_to_wire(source)))
        assert result.dtype == dtype and result.shape == source.shape
        assert torch.equal(result, source)
        if source.numel():
            assert result.data_ptr() != source.data_ptr()


def test_tensor_tree_preserves_nested_tuples_lists_and_bf16():
    tree = (torch.ones(1, 2, dtype=torch.bfloat16), {"kv": [(torch.zeros(1),), (), []]}, None, 3, True)
    copy = tree_from_wire(json.loads(json.dumps(tree_to_wire(tree))))
    assert isinstance(copy, tuple) and isinstance(copy[1]["kv"], list)
    assert isinstance(copy[1]["kv"][0], tuple) and copy[1]["kv"][1] == ()
    assert copy[1]["kv"][2] == [] and copy[0].dtype == torch.bfloat16
    assert copy[2:] == tree[2:]


@pytest.mark.parametrize("alteration", [
    {"dtype": "complex64"}, {"shape": [True]}, {"shape": [-1]}, {"shape": [4]},
    {"shape": [1] * 9}, {"data": "!not base64!"}, {"__tensor__": 2}, {"extra": 1},
])
def test_invalid_tensor_envelopes_are_rejected(alteration):
    envelope = tree_to_wire(torch.ones(1))
    envelope.update(alteration)
    with pytest.raises((ValueError, TypeError)):
        tree_from_wire(envelope)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_conditioning_is_rejected(value):
    with pytest.raises(ValueError, match="nonfinite"):
        tree_from_wire(tree_to_wire(torch.tensor([value])))


def test_successful_publication_changes_only_expert_and_invalidates_cached_contexts():
    backend = ToyBackend()
    service = InferenceService(backend, action_expert={})
    encoder = snapshot(backend.encoder)
    key = service.encode(dict(observations=[observation()], seeds=[1]))["items"][0]["context_id"]
    before = service.action_expert.sample_chain(chain_request(count=1))
    replacement = {"bias": torch.tensor([.75, -.25])}
    client = LocalClient(service)
    receipt = publish_expert(client, replacement, 0, chunk_bytes=127)
    assert receipt == dict(committed=True, policy_version=1, weights_sha256=state_sha256(replacement))
    assert_state(backend.encoder, encoder)
    assert_state(backend.expert, replacement)
    assert backend.expert.invalidations == 1 and not service.contexts
    with pytest.raises(ValueError, match="expired"):
        service.decode(dict(items=[dict(context_id=key)]))
    after = service.action_expert.sample_chain(chain_request(1, count=1))
    assert before["items"][0]["actions"] != after["items"][0]["actions"]
    assert after["policy_version"] == after["items"][0]["policy_version"] == 1
    offsets = [payload["offset"] for route, payload in client.calls if route == "/weights_append"]
    assert len(offsets) > 1 and offsets == list(range(0, offsets[-1] + 1, 127))
    assert sum(route == "/weights_commit" for route, _ in client.calls) == 1
    service.close()


@pytest.mark.parametrize("fault", ["missing", "extra", "shape", "dtype", "nan", "blob_hash", "state_hash"])
def test_invalid_upload_cannot_change_live_policy(fault):
    expert = ToyExpert()
    initial = snapshot(expert)
    owner = ExpertWeights(expert)
    state, overrides = {"bias": torch.ones(2)}, {}
    if fault == "missing":
        state = {}
    elif fault == "extra":
        state["encoder.weight"] = torch.ones(2, 2)
    elif fault == "shape":
        state["bias"] = torch.ones(3)
    elif fault == "dtype":
        state["bias"] = torch.ones(2, dtype=torch.float64)
    elif fault == "nan":
        state["bias"][0] = float("nan")
    elif fault == "blob_hash":
        overrides["sha256"] = "0" * 64
    elif fault == "state_hash":
        overrides["weights_sha256"] = "0" * 64
    upload = stage(owner, state, **overrides)
    try:
        with pytest.raises(ValueError):
            owner.commit(upload)
        assert owner.metadata() == dict(policy_version=0, weights_sha256=state_sha256(initial))
        assert_state(expert, initial)
    finally:
        owner.close()


def test_incomplete_upload_keeps_old_policy_then_commits_once():
    expert = ToyExpert()
    owner = ExpertWeights(expert)
    state = {"bias": torch.ones(2)}
    payload, blob = upload_payload(state)
    upload = {"upload_id": owner.begin(payload)["upload_id"]}
    owner.append(dict(upload, offset=0, data=base64.b64encode(blob[:23]).decode()))
    with pytest.raises(ValueError, match="incomplete"):
        owner.commit(upload)
    assert owner.version == 0 and torch.equal(expert.bias, torch.zeros(2))
    owner.append(dict(upload, offset=23, data=base64.b64encode(blob[23:]).decode()))
    assert owner.commit(upload)["policy_version"] == 1
    with pytest.raises(ValueError, match="Unknown"):
        owner.commit(upload)
    assert owner.version == 1 and torch.equal(expert.bias, torch.ones(2))


@pytest.mark.parametrize("offset", [-1, True, 1.0, 1, 99999])
def test_out_of_order_or_invalid_offsets_do_not_advance_upload(offset):
    owner = ExpertWeights(ToyExpert())
    payload, _ = upload_payload({"bias": torch.ones(2)})
    upload = {"upload_id": owner.begin(payload)["upload_id"]}
    try:
        with pytest.raises(ValueError, match="sequential"):
            owner.append(dict(upload, offset=offset, data="YQ=="))
        assert owner.pending["offset"] == 0 and owner.version == 0
    finally:
        owner.close()


@pytest.mark.parametrize("data", ["", "!", base64.b64encode(b"12345").decode()])
def test_empty_invalid_or_overrun_chunks_are_rejected(data):
    owner = ExpertWeights(ToyExpert())
    payload, _ = upload_payload({"bias": torch.ones(2)}, size=4)
    upload = {"upload_id": owner.begin(payload)["upload_id"]}
    try:
        with pytest.raises(ValueError):
            owner.append(dict(upload, offset=0, data=data))
        assert owner.pending["offset"] == 0
    finally:
        owner.close()


@pytest.mark.parametrize("overrides", [
    {"expected_version": 1}, {"expected_version": True}, {"new_version": 2}, {"new_version": True},
    {"size": 0}, {"size": -1}, {"size": True}, {"size": 10001}, {"sha256": "bad"},
    {"weights_sha256": "A" * 64},
])
def test_bad_begin_is_rejected_before_staging(overrides):
    owner = ExpertWeights(ToyExpert(), max_bytes=10000)
    payload, _ = upload_payload({"bias": torch.ones(2)}, **overrides)
    with pytest.raises(ValueError):
        owner.begin(payload)
    assert owner.pending is None and owner.version == 0


def test_pending_upload_abort_expiry_and_restart(monkeypatch):
    now = [1.]
    monkeypatch.setattr("pressb.runtime.expert_sync.time.monotonic", lambda: now[0])
    owner = ExpertWeights(ToyExpert(), ttl=5)
    payload, _ = upload_payload({"bias": torch.ones(2)})
    first = {"upload_id": owner.begin(payload)["upload_id"]}
    with pytest.raises(ValueError, match="in progress"):
        owner.begin(payload)
    with pytest.raises(ValueError, match="Unknown"):
        owner.abort(dict(upload_id="another upload"))
    assert owner.abort(first)["aborted"] and owner.pending is None
    second = {"upload_id": owner.begin(payload)["upload_id"]}
    now[0] += 5
    with pytest.raises(ValueError, match="expired"):
        owner.append(dict(second, offset=0, data="YQ=="))
    assert owner.pending is None and owner.version == 0
    third = owner.begin(payload)
    assert third["upload_id"] not in (first["upload_id"], second["upload_id"])
    owner.close()
    assert owner.pending is None


def test_backend_invalidation_failure_rolls_back_weights_and_version():
    expert = ToyExpert()
    initial = snapshot(expert)
    def reject():
        raise RuntimeError("backend invalidation failed")
    owner = ExpertWeights(expert, on_commit=reject)
    upload = stage(owner, {"bias": torch.ones(2)})
    try:
        with pytest.raises(RuntimeError, match="invalidation failed"):
            owner.commit(upload)
        assert_state(expert, initial)
        assert owner.metadata() == dict(policy_version=0, weights_sha256=state_sha256(initial))
    finally:
        owner.close()


@pytest.mark.parametrize("changed", ["schedule", "normalizer.scale", "schedule_alias"])
def test_sampler_and_normalization_buffers_are_immutable_including_aliases(changed):
    expert = ToyExpert()
    expert.register_buffer("schedule", torch.tensor([.8, .2]))
    expert.register_buffer("schedule_alias", expert.schedule)
    expert.normalizer = nn.Module()
    expert.normalizer.register_buffer("scale", torch.tensor([1., 2.]))
    expert.register_buffer("scratch", torch.zeros(3), persistent=False)
    initial = snapshot(expert)
    owner = ExpertWeights(expert)
    bad = snapshot(expert)
    bad["bias"].add_(1)
    bad[changed].add_(.5)
    upload = stage(owner, bad)
    try:
        with pytest.raises(ValueError, match="immutable"):
            owner.commit(upload)
        assert_state(expert, initial)
        assert owner.version == 0
        owner.abort(upload)
        good = snapshot(expert)
        good["bias"].add_(1)
        assert owner.commit(stage(owner, good))["policy_version"] == 1
        assert_state(expert, good)
    finally:
        owner.close()


@pytest.mark.parametrize("fault", ["append_offset", "commit_false", "commit_version", "commit_hash"])
def test_publication_requires_matching_receipts_and_never_retries(fault):
    service = InferenceService(ToyBackend(), action_expert={})
    inner = LocalClient(service)
    class CorruptReceipt:
        def call(self, route, payload):
            result = inner.call(route, payload)
            if route == "/weights_append" and fault == "append_offset":
                result["offset"] -= 1
            if route == "/weights_commit":
                if fault == "commit_false":
                    result["committed"] = False
                if fault == "commit_version":
                    result["policy_version"] += 1
                if fault == "commit_hash":
                    result["weights_sha256"] = "0" * 64
            return result
    try:
        with pytest.raises(RuntimeError, match="acknowledge"):
            publish_expert(CorruptReceipt(), {"bias": torch.ones(2)}, 0, chunk_bytes=127)
        routes = [route for route, _ in inner.calls]
        assert routes.count("/weights_begin") == 1
        if fault == "append_offset":
            assert routes.count("/weights_append") == 1 and "/weights_commit" not in routes
            assert service.action_expert.weights.version == 0
        else:
            # A lost/corrupt receipt cannot undo the server's committed version;
            # clients fail rather than replaying an ambiguously executed update.
            assert routes.count("/weights_commit") == 1
            assert service.action_expert.weights.version == 1
    finally:
        service.close()


@pytest.mark.parametrize("transport", ["http", "ws"])
def test_real_transport_disconnect_mid_upload_leaves_old_policy_active(transport):
    service = InferenceService(ToyBackend(), action_expert={})
    initial = snapshot(service.backend.expert)
    with running_node(service, transport) as endpoint:
        writer = NodeClient(endpoint, timeout=5, token="test-token")
        baseline = writer.call("/sample_chain", chain_request(count=1, deterministic=True))
        payload, blob = upload_payload({"bias": torch.ones(2)})
        upload = {"upload_id": writer.call("/weights_begin", payload)["upload_id"]}
        writer.call("/weights_append", dict(upload, offset=0, data=base64.b64encode(blob[:23]).decode()))
        writer.close()
        reader = NodeClient(endpoint, timeout=5, token="test-token")
        try:
            after = reader.call("/sample_chain", chain_request(count=1, deterministic=True))
            assert after["policy_version"] == 0 and after["items"] == baseline["items"]
            assert_state(service.backend.expert, initial)
            assert reader.call("/weights_abort", upload)["aborted"]
            assert publish_expert(reader, {"bias": torch.ones(2)}, 0, chunk_bytes=127)["policy_version"] == 1
        finally:
            reader.close()
    service.close()


@pytest.mark.parametrize("transport", ["http", "ws"])
def test_real_transport_chain_publish_new_version_and_frozen_encoder(transport):
    backend = ToyBackend()
    initial_encoder = snapshot(backend.encoder)
    service = InferenceService(backend, batch_size=2, action_expert={})
    with running_node(service, transport) as endpoint:
        client = NodeClient(endpoint, timeout=5, token="test-token")
        try:
            health = client.call("/health")
            assert health["encoder_frozen"] and not health["frozen"]
            assert health["action_expert"]["policy_version"] == 0
            assert {"sample_chain", "action_expert_update"} <= set(health["capabilities"])
            sampled = client.call("/sample_chain", chain_request())
            assert backend.batches == [2, 1]
            assert len(sampled["items"]) == 3 and sampled["policy_version"] == 0
            for item in sampled["items"]:
                chain = tree_from_wire(item["chain"])
                logprobs = tree_from_wire(item["logprobs"])
                conditioning = tree_from_wire(item["conditioning"])
                assert chain.shape == (3, 3, 2) and logprobs.shape == (2, 3, 2)
                assert chain.dtype == logprobs.dtype == torch.float32
                assert isinstance(conditioning, tuple)
                assert isinstance(conditioning[1]["cache"], tuple)
                assert conditioning[0].dtype == torch.bfloat16
                for step in range(2):
                    score = service.action_expert.policy.log_prob(
                        conditioning, chain[None, step], chain[None, step + 1], step)
                    torch.testing.assert_close(score[0], logprobs[step], rtol=0, atol=0)
                np.testing.assert_array_equal(item["actions"], chain[-1].numpy())
            assert client.call("/health")["cache"]["size"] == 0
            baseline = client.call("/sample_chain", chain_request(deterministic=True))
            assert all(item["logprobs"] is None for item in baseline["items"])
            replacement = {"bias": torch.tensor([.7, -.2])}
            receipt = publish_expert(client, replacement, 0, chunk_bytes=127)
            assert receipt["policy_version"] == 1 and receipt["weights_sha256"] == state_sha256(replacement)
            updated = client.call("/sample_chain", chain_request(1, deterministic=True))
            assert updated["items"][0]["actions"] != baseline["items"][0]["actions"]
            assert all(item["policy_version"] == 1 for item in updated["items"])
            assert_state(backend.encoder, initial_encoder)
            assert all(not parameter.requires_grad for parameter in backend.encoder.parameters())
        finally:
            client.close()
        for route, payload in (("/sample_chain", chain_request()),
                               ("/weights_begin", upload_payload({"bias": torch.zeros(2)})[0])):
            stale = NodeClient(endpoint, timeout=5, token="test-token")
            try:
                with pytest.raises(RPCError, match="Stale policy version"):
                    stale.call(route, payload)
            finally:
                stale.close()
        assert service.action_expert.weights.version == 1
    service.close()
    assert backend.closed


@pytest.mark.parametrize("enabled", [False, True])
def test_optional_expert_does_not_change_existing_routes(enabled):
    service = InferenceService(ToyBackend(), action_expert={} if enabled else None)
    assert ("/sample_chain" in service.handlers) == enabled
    assert ("action_expert_update" in service.health()["capabilities"]) == enabled
    request = dict(observations=[observation()], seeds=[4])
    encoded = service.encode(dict(request, include_base_actions=True))["items"][0]
    decoded = service.decode(dict(items=[dict(context_id=encoded["context_id"])]))["items"][0]
    inferred = service.infer(request)["items"][0]
    assert encoded["base_actions"] == decoded["actions"] == inferred["actions"]
    assert service.release(dict(context_ids=[encoded["context_id"]])) == {"released": 1}
    assert service.health()["cache"]["size"] == 0
    service.close()


def test_bad_chain_request_rejected_before_encoding_and_closed_service_rejects_upload():
    backend = ToyBackend()
    service = InferenceService(backend, action_expert={})
    for overrides in ({"expected_policy_version": True}, {"expected_policy_version": 1},
                      {"deterministic": 1}, {"seeds": [True]}, {"extra": 1}):
        payload = chain_request(count=1)
        payload.update(overrides)
        with pytest.raises(ValueError):
            service.action_expert.sample_chain(payload)
    assert backend.batches == []
    service.close()
    with pytest.raises(RuntimeError, match="closed"):
        service.handlers["/weights_begin"](upload_payload({"bias": torch.ones(2)})[0])
