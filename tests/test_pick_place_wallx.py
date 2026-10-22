"""Native Wall-X wire contract, physical units and closed-loop chunk boundaries."""
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from pressb.pick_place.wallx import (WallXClient, WallXChunkPolicy, decode_actions,
    load_policy_initial, observation_from_sim, pack, unpack)


def test_native_server_roundtrip_and_fresh_observation_at_prefix_boundary(tmp_path):
    from websockets.sync.server import serve
    received = []
    def handler(connection):
        connection.send(pack(dict(action_dim=7, pred_horizon=32)))
        for request_index in range(2):
            received.append(unpack(connection.recv()))
            action = np.zeros((1, 32, 7), np.float32)
            action[0, :, 0] = np.arange(32) + request_index*40
            action[0, :, 6] = 3.5
            action[0, 0, 6] = -.17
            connection.send(pack(dict(action=action, server_timing=dict(infer_ms=1.))))
    with serve(handler, "127.0.0.1", 0, compression=None) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        client = WallXClient(f"ws://127.0.0.1:{server.socket.getsockname()[1]}", timeout=5., token="test")
        kin = SimpleNamespace(lower=np.full(6, -np.pi), upper=np.full(6, np.pi))
        cfg = dict(gripper_max_width_m=.1, task_instruction="Grab the stuff into the box.")
        policy = WallXChunkPolicy(client, kin, cfg, tmp_path, execute_steps=2)
        q = np.r_[np.deg2rad([10, 20, 30, 40, 50, 60]), .0175, -.0175]
        images = dict(up_rgb=np.full((48, 64, 3), 123, np.uint8),
                      wrist_rgb=np.full((48, 64, 3), 77, np.uint8))
        try:
            a, indices = policy.command(0., joints=q, **images)
            assert indices == dict(policy_request=0, policy_action_index=0)
            assert a[0] == 0. and a[6] == 0.  # Explicit recorded gripper closure saturation.
            assert not policy.needs_observation
            b, _ = policy.command(1/30)  # No second inference or stale observation needed.
            np.testing.assert_allclose(b[[0, 6]], [np.deg2rad(1), .035])
            assert policy.needs_observation
            q[0] += .01
            c, indices = policy.command(2/30, joints=q, **images)
            np.testing.assert_allclose(c[0], np.deg2rad(40))
            assert indices == dict(policy_request=1, policy_action_index=0)
        finally:
            client.close()
            server.shutdown()
            thread.join(timeout=5)
    assert len(received) == 2
    assert set(received[0]) == {"obs", "is_rtc"} and received[0]["is_rtc"] is False
    obs = received[0]["obs"]
    np.testing.assert_allclose(obs["state"], [10, 20, 30, 40, 50, 60, 3.5])
    assert obs["state"].dtype == np.float32
    assert obs["dataset_names"] == ["lerobot/piper"]
    assert obs["face_view"].dtype == np.uint8 and obs["left_wrist_view"][0, 0, 0] == 77
    assert received[1]["obs"]["state"][0] > received[0]["obs"]["state"][0]
    logs = [json.loads(line) for line in (tmp_path/"policy_requests.jsonl").read_text().splitlines()]
    assert [row["simulation_time_s"] for row in logs] == [0., 2/30]
    assert logs[0]["checks"]["gripper_saturated_samples"] == 1
    assert unpack((tmp_path/"policy_requests/request_000000.request.msgpack").read_bytes())["obs"]["prompt"] == cfg["task_instruction"]
    assert len(list((tmp_path/"policy_requests").glob("*.response.msgpack"))) == 2
    assert not policy.summary()["expert_actions_used"]


@pytest.mark.parametrize("bad", [np.zeros((1, 31, 7)), np.full((1, 32, 7), np.nan)])
def test_bad_action_shape_and_nonfinite_fail(bad):
    with pytest.raises(ValueError, match="finite Wall-X"):
        decode_actions(bad, np.full(6, -1.), np.full(6, 1.), .1)


def test_arm_out_of_limits_fails_unless_explicitly_clipped():
    raw = np.zeros((32, 7))
    raw[3, 2] = 90
    with pytest.raises(ValueError, match="row 3, joint 3"):
        decode_actions(raw, np.full(6, -1.), np.full(6, 1.), .1)
    commands, checks = decode_actions(raw, np.full(6, -1.), np.full(6, 1.), .1, arm_limit_mode="clip")
    assert commands[3, 2] == 1 and checks["arm_saturated_values"] == 1


def test_policy_initial_requires_no_expert_action_or_timestamps(tmp_path):
    path = tmp_path/"initial.npz"
    np.savez(path, state_raw=np.array([[10, 20, 30, 40, 50, 60, 7.]]))
    trajectory = load_policy_initial(path, .1)
    assert trajectory["commands"].shape == (0, 7)
    np.testing.assert_allclose(trajectory["initial"][:6], np.deg2rad([10, 20, 30, 40, 50, 60]))
    assert trajectory["initial"][6] == .07
    assert not trajectory["metadata"]["expert_action_targets_loaded"]


def test_invalid_response_is_logged_without_masking_validation_error(tmp_path):
    class Client:
        expected_horizon = 32
        last_latency_s = .01
        def infer(self, observation, record_prefix):
            return dict(action=np.full((1, 32, 7), np.nan))
    policy = WallXChunkPolicy(Client(), SimpleNamespace(lower=np.full(6, -1), upper=np.full(6, 1)),
        dict(gripper_max_width_m=.1, task_instruction="test"), tmp_path)
    image = np.zeros((2, 2, 3), np.uint8)
    with pytest.raises(ValueError, match="finite Wall-X"):
        policy.command(0., joints=np.zeros(8), up_rgb=image, wrist_rgb=image)
    record = json.loads((tmp_path/"policy_requests.jsonl").read_text())
    assert record["status"] == "failed" and "finite Wall-X" in record["error"]


def test_observation_rejects_normalized_float_images():
    with pytest.raises(ValueError, match="uint8 RGB"):
        observation_from_sim(np.zeros(8), np.zeros((2, 2, 3)), np.zeros((2, 2, 3)), "test")
