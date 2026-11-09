"""Streaming checkpoint durability, bounded paths and disk-free learner resume."""
from contextlib import contextmanager
import hashlib
import http.client
import json
from pathlib import Path
import threading
import time

import pytest
import torch

from pressb.runtime.checkpoint_sink import (CheckpointSinkConfig, CheckpointSinkServer,
    save_checkpoint_stream, load_checkpoint_stream)


@contextmanager
def sink(directory, **kwargs):
    server = CheckpointSinkServer(("127.0.0.1", 0), directory, **kwargs)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield server, endpoint
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def config(endpoint):
    return CheckpointSinkConfig(endpoint, "run_test", timeout=5.)


def test_roundtrip_tensor_optimizer_and_rng_without_sender_tempfile(tmp_path, monkeypatch):
    parameter = torch.nn.Parameter(torch.tensor([.3, -.8]))
    optimizer = torch.optim.AdamW([parameter], lr=.01)
    parameter.square().sum().backward()
    optimizer.step()
    payload = {"expert": {"weight": parameter.detach()}, "optimizer": optimizer.state_dict(),
               "rng": torch.get_rng_state(), "steps": 123, "label": "full training state"}
    with sink(tmp_path / "storage") as (server, endpoint):
        receipt = save_checkpoint_stream(payload, "step_000000123.pt", config(endpoint))
        stored = server.directory / "run_test/step_000000123.pt"
        assert stored.is_file()
        assert receipt["bytes"] == stored.stat().st_size
        assert receipt["sha256"] == hashlib.sha256(stored.read_bytes()).hexdigest()
        restored = load_checkpoint_stream(receipt["url"], timeout=5.)
        assert torch.equal(restored["expert"]["weight"], parameter)
        assert torch.equal(restored["rng"], payload["rng"])
        assert torch.equal(restored["optimizer"]["state"][0]["exp_avg"], optimizer.state_dict()["state"][0]["exp_avg"])
        assert restored["steps"] == 123
        assert not list(tmp_path.rglob("*.partial"))
        assert len(list(tmp_path.rglob("*.pt"))) == 1


@pytest.mark.parametrize("namespace", ["..", "../escape", "a/b", "%2e%2e", "a..b", "", "/tmp"])
def test_client_rejects_unsafe_namespace(namespace):
    with pytest.raises(ValueError):
        CheckpointSinkConfig("http://127.0.0.1:1", namespace)


@pytest.mark.parametrize("route", ["/checkpoints/../escape.pt", "/checkpoints/good/../evil.pt",
                                   "/checkpoints/%2e%2e/out.pt", "/checkpoints/good/secret.json"])
def test_server_rejects_traversal_and_non_checkpoint_files(tmp_path, route):
    with sink(tmp_path / "storage") as (server, endpoint):
        connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5.)
        connection.request("PUT", route, body=b"0\r\n\r\n", headers={"Transfer-Encoding": "chunked"})
        assert connection.getresponse().status == 400
        connection.close()
        assert not list(tmp_path.rglob("*.pt"))


def test_existing_checkpoint_never_overwritten(tmp_path):
    with sink(tmp_path) as (_, endpoint):
        first = save_checkpoint_stream({"x": torch.ones(8)}, "step_1.pt", config(endpoint))
        with pytest.raises((RuntimeError, OSError)):
            save_checkpoint_stream({"x": torch.zeros(8)}, "step_1.pt", config(endpoint))
        restored = load_checkpoint_stream(first["url"], timeout=5.)
        assert torch.equal(restored["x"], torch.ones(8))


def test_corrupt_stream_hash_does_not_publish(tmp_path, monkeypatch):
    from pressb.runtime.checkpoint_sink import _Handler
    reply = _Handler._reply
    checked = []
    def verify_before_reply(handler, status, body):
        if status == 400:
            # Verify ordering on the server thread, independent of client timing.
            assert not list(tmp_path.rglob('*.partial'))
            assert not handler.server.upload_lock.locked()
            checked.append(True)
        return reply(handler, status, body)
    monkeypatch.setattr(_Handler, '_reply', verify_before_reply)
    with sink(tmp_path) as (server, endpoint):
        connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5.)
        body = b"3\r\nabc\r\n0\r\nX-Checkpoint-SHA256: " + b"0"*64 + b"\r\nX-Checkpoint-Bytes: 3\r\n\r\n"
        connection.request("PUT", "/checkpoints/run/step_1.pt", body=body, headers={"Transfer-Encoding": "chunked"})
        assert connection.getresponse().status == 400
        connection.close()
        assert not list(tmp_path.rglob("*.pt"))
        assert not list(tmp_path.rglob("*.partial"))
        assert checked == [True]


def test_interrupted_stream_cleans_partial_file(tmp_path):
    with sink(tmp_path) as (server, endpoint):
        connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5.)
        connection.putrequest("PUT", "/checkpoints/run/step_1.pt")
        connection.putheader("Transfer-Encoding", "chunked")
        connection.endheaders()
        connection.send(b"100\r\nshort")
        deadline = time.monotonic() + 3
        while not list(tmp_path.rglob("*.partial")) and time.monotonic() < deadline:
            time.sleep(.01)
        assert list(tmp_path.rglob("*.partial"))
        connection.close()
        while list(tmp_path.rglob("*.partial")) and time.monotonic() < deadline:
            time.sleep(.01)
        assert not list(tmp_path.rglob("*.pt"))
        assert not list(tmp_path.rglob("*.partial"))


def test_receiver_size_limit_and_symlink_guard(tmp_path):
    with sink(tmp_path / "storage", max_bytes=128) as (server, endpoint):
        with pytest.raises((RuntimeError, OSError)):
            save_checkpoint_stream({"x": torch.ones(1000)}, "step_1.pt", config(endpoint))
        assert not list(tmp_path.rglob("*.pt"))
        outside = tmp_path / "outside"
        outside.mkdir()
        (server.directory / "evil").symlink_to(outside, target_is_directory=True)
        with pytest.raises(ValueError, match="traverse"):
            server.destination("/checkpoints/evil/step_1.pt")


def test_optional_bearer_authentication(tmp_path, monkeypatch):
    with sink(tmp_path, token="test-token") as (_, endpoint):
        monkeypatch.delenv("PRESSB_CHECKPOINT_TOKEN", raising=False)
        with pytest.raises((RuntimeError, OSError)):
            save_checkpoint_stream({"x": torch.ones(1)}, "step_1.pt", config(endpoint))
        monkeypatch.setenv("PRESSB_CHECKPOINT_TOKEN", "test-token")
        receipt = save_checkpoint_stream({"x": torch.ones(1)}, "step_1.pt", config(endpoint))
        assert load_checkpoint_stream(receipt["url"], timeout=5.)["x"].item() == 1


def test_dppo_stream_save_and_http_resume_publish_before_rollout(tmp_path):
    from test_dppo_training import make_runner
    with sink(tmp_path / "storage") as (_, endpoint):
        runner, _, _ = make_runner(tmp_path / "run", steps=4)
        runner.config.checkpoint_sink = config(endpoint)
        result = runner.run()
        assert result["last_checkpoint"].startswith(endpoint)
        assert not list((tmp_path / "run").rglob("*.pt"))
        assert len(list((tmp_path / "storage").rglob("*.pt"))) == 1  # no duplicate final save
        last = json.loads((tmp_path / "run/checkpoints/latest.json").read_text())
        resumed, client, _ = make_runner(tmp_path / "resumed", steps=8)
        resumed.config.checkpoint.resume = last["url"]
        resumed.config.checkpoint_sink = CheckpointSinkConfig(endpoint, "resumed", timeout=5.)
        next_result = resumed.run()
        assert next_result["transitions"] == 8
        routes = [route for route, _ in client.calls]
        assert routes.index("/weights_commit") < routes.index("/sample_chain")
        assert not list((tmp_path / "resumed").rglob("*.pt"))
