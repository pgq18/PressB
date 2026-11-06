"""Adapter exercises the actual legacy service state machine with CPU physics fixtures."""
from copy import deepcopy

import numpy as np
import pytest

from pressb.online_rl.fast_simulation import IndexedSimulationService
from pressb.online_rl.protocol import pose8_to_pose9
from pressb.rl_nodes.contract import encode_image, negotiate, validate_metadata
from pressb.rl_nodes.elevator import ElevatorBackend
from pressb.rl_nodes.simulation import SimulationService


PNG = encode_image(np.zeros((8, 8, 3), np.uint8))["data"]
STATE = [0.3, 0., 0.4, 1., 0., 0., 0., .008]


class PhysicsFixture:
    def __init__(self):
        self.episodes = {}
        self.calls = []
        self.events = {}

    def health(self):
        return dict(num_envs=2, smoothing_window=1, max_seconds=15,
                    pose_frame="base_link", pose_link="gripper_tcp", quaternion_order="wxyz",
                    gripper_width_m=.008, camera_order=["global", "wrist"],
                    config_sha256="scene-config", scene_sha256="scene",
                    motion_smoothing={"window": 1},
                    config=dict(panel_offset_x_m=0., panel_offset_y_m=0., panel_randomization=dict(
                        enabled=True, min_offset_x_m=-.01, max_offset_x_m=.01,
                        min_offset_y_m=-.025, max_offset_y_m=.025)))

    def observe(self, i):
        return dict(env_id=i, observation=dict(state=STATE, task=f"Press {self.episodes[i]['floor']} floor.",
                    images={"global": PNG, "wrist": PNG}, control_state=[0.] * 25), info={})

    def reset(self, episodes, seed):
        self.calls.append(("reset", deepcopy(episodes), seed))
        self.episodes.update(enumerate(episodes))
        return [self.observe(i) for i in range(2)]

    def reset_envs(self, episodes, seed):
        self.calls.append(("reset_envs", deepcopy(episodes), seed))
        self.episodes.update(episodes)
        return [self.observe(i) for i in episodes]

    def step(self, actions):
        self.calls.append(("step", deepcopy(actions)))
        result = []
        for i in actions:
            row = self.observe(i)
            end, ticks = self.events.pop(i, (False, 28))
            row.update(terminated=end, truncated=False, executed_physics_steps=ticks,
                       info={"success": end})
            result.append(row)
        return result


class LocalClient:
    def __init__(self):
        self.physics = PhysicsFixture()
        self.service = IndexedSimulationService(self.physics, single_gamma=.995)
        self.requests = []
        self.tamper = None
        self.closed = False

    def call(self, path, payload=None):
        self.requests.append((path, deepcopy(payload)))
        result = getattr(self.service, path[1:])(payload)
        if self.tamper:
            self.tamper(result)
        return result

    def close(self):
        self.closed = True


def adapter(**overrides):
    client = LocalClient()
    return ElevatorBackend({"endpoint": "http://127.0.0.1:9000", **overrides}, client=client), client


def actions(ids=(0, 1)):
    chunk = pose8_to_pose9(np.tile(STATE, (7, 1))).tolist()
    return [{"env_id": i, "actions": chunk} for i in ids]


def test_schema_declares_true_scheduling_and_observation_contract():
    backend, client = adapter()
    meta = validate_metadata(backend.describe(), "simulation")
    assert meta["scheduling"] == "all_active" and meta["execution_horizons"] == [7]
    assert meta["initial_reset"] == "all_envs" and meta["reset_mode"] == "indexed"
    assert meta["discount"] == {"gamma": .995, "timebase_hz": 30}
    rows = backend.reset([1, 0], seed=42, options={"by_env": {"0": {"floor": 31}, "1": {"task": "Press 25 floor."}}})
    assert [r["env_id"] for r in rows] == [1, 0]
    observation = rows[1]["observation"]
    assert observation["images"]["global"] == {"encoding": "png_base64", "data": PNG}
    assert observation["state"] == pytest.approx(STATE)
    assert observation["task_features"][7] == 1 and sum(observation["task_features"]) == 1
    assert len(observation["control_state"]) == 25
    backend.close()
    assert client.closed


def test_indexed_reset_retains_other_episode_and_maps_duration_discount():
    backend, client = adapter(floors=[24, 31])
    backend.reset([0, 1], 11)
    client.physics.events[0] = (True, 5)
    rows = backend.step(actions())
    assert rows[0]["reward"] == pytest.approx(.995)
    assert rows[0]["discount"] == pytest.approx(.995 ** 1.25)
    assert rows[0]["info"]["elapsed_control_steps"] == 1.25
    commands = client.physics.calls[-1][1]
    np.testing.assert_allclose(commands[0], np.tile(STATE, (7, 1)))
    alive_episode = client.service.items[1]["episode_id"]
    calls = len(client.physics.calls)
    # The ended slot need not hold up another chunk for the remaining slot.
    backend.step(actions([1]))
    assert len(client.physics.calls) == calls + 1
    backend.reset([0], 12, {"floor": 31})
    assert client.physics.calls[-1][0] == "reset_envs"
    assert client.service.items[1]["episode_id"] == alive_episode
    assert client.service.step_id == 2
    backend.step(actions())


def test_invalid_requests_are_rejected_before_remote_mutation():
    backend, client = adapter()
    with pytest.raises(ValueError, match="all environments"):
        backend.reset([0], 4)
    with pytest.raises(ValueError, match="outside"):
        backend.reset([0, 1], 4, {"offset_x_m": 1.})
    with pytest.raises(ValueError, match="existing environments"):
        backend.reset([0, 1], 4, {"by_env": {"slot0": {"floor": 31}}})
    assert client.physics.calls == []
    backend.reset([0, 1], 4)
    before = len(client.requests)
    for bad in (actions([0]), actions([0, 0]), [{"env_id": 0, "actions": [[0.] * 9]}, actions()[1]]):
        with pytest.raises(ValueError):
            backend.step(bad)
    degenerate = actions()
    degenerate[1]["actions"] = [[0.] * 9] * 7
    with pytest.raises(ValueError, match="Degenerate"):
        backend.step(degenerate)
    with pytest.raises(ValueError, match="ended"):
        backend.reset([0], 5)
    assert len(client.requests) == before


def test_reset_randomization_is_seeded_per_slot_and_within_remote_bounds():
    a, ca = adapter()
    b, cb = adapter()
    a.reset([0, 1], 123)
    b.reset([1, 0], 123)
    assert ca.physics.episodes == cb.physics.episodes
    for episode in ca.physics.episodes.values():
        assert -.01 <= episode["offset_x_m"] <= .01
        assert -.025 <= episode["offset_y_m"] <= .025


def test_remote_metadata_and_existing_ownership_are_not_overridden():
    with pytest.raises(ValueError, match="scene_sha256"):
        adapter(expected_metadata={"scene_sha256": "wrong"})
    client = LocalClient()
    original = client.physics.health
    client.physics.health = lambda: {**original(), "pose_frame": "world"}
    # Service captures its backend metadata at initialization.
    client.service = IndexedSimulationService(client.physics)
    with pytest.raises(ValueError, match="pose_frame"):
        ElevatorBackend({"endpoint": "http://x", "expected_metadata": {"pose_frame": "world"}}, client=client)
    client = LocalClient()
    client.service.run_id = "other-owner"
    with pytest.raises(ValueError, match="already owned"):
        ElevatorBackend({"endpoint": "http://x"}, client=client)


def test_corrupt_post_step_identity_faults_adapter_without_retry():
    backend, client = adapter()
    backend.reset([0, 1], 10)
    client.tamper = lambda response: response.update(step_id=99)
    with pytest.raises(ValueError, match="stale"):
        backend.step(actions())
    count = len(client.requests)
    with pytest.raises(RuntimeError, match="faulted"):
        backend.step(actions())
    assert len(client.requests) == count


def test_generic_service_ids_and_retry_never_repeat_legacy_physics():
    backend, client = adapter()
    service = SimulationService(backend)
    metadata = backend.describe()
    inference = {**metadata, "node_type": "inference", "model_id": "vlajepa-test", "feature_dim": 8,
                 "noise": {"shape": [7, 9]},
                 "capabilities": ["encode", "decode", "features", "context_cache", "external_initial_noise"]}
    negotiate(inference, metadata, "dsrl")
    with pytest.raises(ValueError, match="horizon"):
        negotiate(inference, metadata, "dsrl", execution_horizon=1)
    reset = service.reset({"run_id": "generic-owner", "request_id": "reset", "env_ids": [0, 1], "seed": 8})
    request = {"run_id": "generic-owner", "request_id": "step", "items": [
        {**command, "episode_id": row["episode_id"], "step_id": row["step_id"]}
        for command, row in zip(actions(), reset["items"])]}
    result = service.step(request)
    assert result["items"][0]["step_id"] == 1
    assert result["items"][0]["episode_id"] != client.service.items[0]["episode_id"]
    calls = len(client.physics.calls)
    assert service.step(request) == result
    assert len(client.physics.calls) == calls
