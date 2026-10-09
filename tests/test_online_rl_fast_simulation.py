"""CPU tests for indexed episode recycling and the fast collector wire contract."""

import base64
from copy import deepcopy
from io import BytesIO

import numpy as np
from PIL import Image
import pytest

from pressb.online_rl.fast_simulation import (
    RESET_MODE,
    IndexedSimulationService,
    encode_fast_image,
    view_env_order,
)
from test_online_rl_simulation import FakeBackend, reset_request, step_request


class IndexedFakeBackend(FakeBackend):
    """Recycling a slot changes only that slot's task and initial observation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.indexed_reset_calls = []
        self.indexed_result_override = None

    def reset_envs(self, episodes, seed):
        self.indexed_reset_calls.append((deepcopy(episodes), seed))
        for env_id, episode in episodes.items():
            self.episodes[env_id] = deepcopy(episode)
        if self.indexed_result_override is not None:
            return self.indexed_result_override
        self.last_result = [
            {"env_id": env_id, "observation": self.observation(env_id), "info": {"seed": seed}}
            for env_id in reversed(episodes)
        ]
        return self.last_result


@pytest.fixture
def indexed_pair():
    backend = IndexedFakeBackend()
    return IndexedSimulationService(backend, single_gamma=0.9), backend


def indexed_request(previous, *, request_id="indexed-0", env_ids=(0,)):
    return {
        "run_id": "run-a",
        "request_id": request_id,
        "cohort_id": previous["cohort_id"],
        "step_id": previous["step_id"],
        "seed": 19,
        "episodes": [
            {"env_id": env_id, "floor": 30 + env_id, "offset_x_m": 0.01, "offset_y_m": -0.025}
            for env_id in env_ids
        ],
    }


def end_first(service, backend):
    initial = service.reset(reset_request())
    backend.transitions.append({0: ("target_pressed", 3)})
    return service.step(step_request(initial))


def test_indexed_reset_preserves_active_peer_terminal_transition_and_global_clock(indexed_pair):
    service, backend = indexed_pair
    terminal_response = end_first(service, backend)
    before = deepcopy(terminal_response)
    peer = deepcopy(service.items[1])
    old_episode = service.items[0]["episode_id"]

    fresh = service.reset_envs(indexed_request(terminal_response))

    assert fresh["cohort_id"] == before["cohort_id"]
    assert fresh["step_id"] == before["step_id"] == 1
    assert fresh["reset_env_ids"] == [0]
    assert fresh["all_done"] is False
    assert [item["env_id"] for item in fresh["items"]] == [0]
    item = fresh["items"][0]
    assert item["episode_id"] != old_episode
    assert item["observation"]["task"] == "Press 30 floor."
    assert item["terminated"] is False and item["truncated"] is False
    assert "reward" not in item and "executed_physics_steps" not in item
    assert service.items[1] == peer
    assert terminal_response == before
    assert terminal_response["items"][0]["reward"] == pytest.approx(0.9 ** 0.5)
    assert terminal_response["items"][0]["executed_physics_steps"] == 3
    assert len(backend.step_calls) == len(backend.reset_calls) == 1
    assert backend.indexed_reset_calls == [({0: {
        "floor": 30, "offset_x_m": 0.01, "offset_y_m": -0.025,
    }}, 19)]

    # Reset responses and backend-owned buffers cannot mutate service state.
    fresh["items"][0]["observation"]["state"][0] = 99
    backend.last_result[0]["observation"]["state"][0] = -99
    assert service.items[0]["observation"]["state"][0] == 0.3
    health = service.health()
    assert health["supports_indexed_reset"] is True
    assert health["reset_mode"] == RESET_MODE
    assert health["active_env_ids"] == [0, 1]


def test_step_after_recycling_mixes_new_and_continuing_episodes(indexed_pair):
    service, backend = indexed_pair
    terminal = end_first(service, backend)
    old_peer_episode = terminal["items"][1]["episode_id"]
    fresh = service.reset_envs(indexed_request(terminal))
    backend.transitions.append({1: ("time_limit", 9)})
    advanced = service.step(step_request(fresh, "step-1"))
    assert set(backend.step_calls[-1]) == {0, 1}
    assert advanced["step_id"] == 2
    new, old = advanced["items"]
    assert new["episode_id"] == fresh["items"][0]["episode_id"]
    assert new["observation"]["task"] == "Press 30 floor."
    assert new["executed_physics_steps"] == 28 and not new["terminated"]
    assert old["episode_id"] == old_peer_episode
    assert old["truncated"] is True and old["terminated"] is False
    assert old["executed_physics_steps"] == 9
    assert old["discount"] == pytest.approx(0.9 ** (9 / 4))
    second_fresh = service.reset_envs(indexed_request(advanced, request_id="indexed-1", env_ids=(1,)))
    assert second_fresh["items"][0]["episode_id"] != old_peer_episode
    assert second_fresh["step_id"] == advanced["step_id"]
    assert service.items[0] == new


def test_indexed_reset_can_restart_all_ended_slots_without_new_cohort(indexed_pair):
    service, backend = indexed_pair
    initial = service.reset(reset_request())
    backend.transitions.append({0: ("time_limit", 28), 1: ("wrong_button_pressed", 2)})
    ended = service.step(step_request(initial))
    assert ended["all_done"] is True
    result = service.reset_envs(indexed_request(ended, env_ids=(1, 0)))
    assert result["reset_env_ids"] == [0, 1]
    assert [item["env_id"] for item in result["items"]] == [0, 1]
    assert result["cohort_id"] == initial["cohort_id"]
    assert result["step_id"] == 1
    assert result["all_done"] is False
    assert {item["episode_id"] for item in result["items"]}.isdisjoint(
        item["episode_id"] for item in ended["items"])


def test_indexed_reset_duplicate_request_is_idempotent_even_after_slot_restarts(indexed_pair):
    service, backend = indexed_pair
    ended = end_first(service, backend)
    payload = indexed_request(ended)
    first = service.reset_envs(payload)
    duplicate = service.reset_envs(deepcopy(payload))
    assert duplicate == first
    assert len(backend.indexed_reset_calls) == 1
    changed = deepcopy(payload)
    changed["seed"] += 1
    with pytest.raises(ValueError, match="request_id"):
        service.reset_envs(changed)
    # Changing only the ID cannot reset a slot which is active again.
    changed = deepcopy(payload)
    changed["request_id"] = "cannot-reset-active"
    with pytest.raises(ValueError, match="Only ended"):
        service.reset_envs(changed)
    assert len(backend.indexed_reset_calls) == 1


@pytest.mark.parametrize("mutation", [
    lambda request: request.update(episodes=[]),
    lambda request: request.update(episodes=None),
    lambda request: request.update(episodes=[None]),
    lambda request: request["episodes"].append(deepcopy(request["episodes"][0])),
    lambda request: request["episodes"][0].update(env_id=2),
    lambda request: request["episodes"][0].update(env_id=-1),
    lambda request: request["episodes"][0].update(env_id=True),
    lambda request: request["episodes"][0].update(env_id=1),
    lambda request: request["episodes"][0].update(floor=36),
    lambda request: request["episodes"][0].update(offset_x_m=0.0101),
    lambda request: request["episodes"][0].update(offset_y_m=float("nan")),
    lambda request: request.update(seed=-1),
    lambda request: request.update(cohort_id="stale-cohort"),
    lambda request: request.update(step_id=0),
    lambda request: request.update(run_id="other-owner"),
])
def test_invalid_indexed_reset_is_rejected_before_mutation(indexed_pair, mutation):
    service, backend = indexed_pair
    ended = end_first(service, backend)
    before = deepcopy(service.items)
    request = indexed_request(ended)
    mutation(request)
    with pytest.raises(ValueError):
        service.reset_envs(request)
    assert backend.indexed_reset_calls == []
    assert service.items == before
    assert service.step_id == 1
    assert service.fault is None


def test_indexed_reset_validates_entire_subset_before_resetting_any_slot(indexed_pair):
    service, backend = indexed_pair
    ended = end_first(service, backend)
    before = deepcopy(service.items)
    # First is ended, second is active: the first slot must remain terminal.
    with pytest.raises(ValueError, match="Only ended"):
        service.reset_envs(indexed_request(ended, env_ids=(0, 1)))
    assert service.items == before
    assert backend.indexed_reset_calls == []


def test_indexed_reset_requires_initial_full_reset(indexed_pair):
    service, backend = indexed_pair
    with pytest.raises(ValueError, match="Missing reset"):
        service.reset_envs(indexed_request({"cohort_id": "unknown", "step_id": 0}))
    assert backend.indexed_reset_calls == []


def test_backend_reset_failure_faults_service_and_prevents_further_mutations(indexed_pair):
    service, backend = indexed_pair
    ended = end_first(service, backend)
    backend.indexed_result_override = []
    with pytest.raises(RuntimeError, match="infrastructure failed"):
        service.reset_envs(indexed_request(ended))
    assert service.health()["ready"] is False
    with pytest.raises(RuntimeError, match="faulted"):
        service.step(step_request(ended, "cannot-step", active=(1,)))
    assert len(backend.step_calls) == 1


def test_stale_reset_after_later_vector_step_cannot_recycle_wrong_episode(indexed_pair):
    service, backend = indexed_pair
    ended = end_first(service, backend)
    stale = indexed_request(ended)
    continued = service.step(step_request(ended, "step-1", active=(1,)))
    with pytest.raises(ValueError, match="Stale step_id"):
        service.reset_envs(stale)
    assert backend.indexed_reset_calls == []
    fresh = service.reset_envs(indexed_request(continued))
    assert fresh["step_id"] == 2


def test_evicted_indexed_request_cannot_execute_twice():
    backend = IndexedFakeBackend()
    service = IndexedSimulationService(backend, response_cache_size=1)
    ended = end_first(service, backend)
    request = indexed_request(ended)
    fresh = service.reset_envs(request)
    service.step(step_request(fresh, "step-1"))
    with pytest.raises(ValueError, match="expired"):
        service.reset_envs(request)
    assert len(backend.indexed_reset_calls) == 1


def test_view_order_maps_lexical_physx_order_to_numeric_environment_order():
    paths = [f"/World/env_{i}/Robot" for i in sorted(range(12), key=str)]
    order = view_env_order(paths, 12)
    assert [paths[index] for index in order] == [f"/World/env_{i}/Robot" for i in range(12)]


@pytest.mark.parametrize("paths,count", [
    (["/World/env_0/Robot", "/World/env_0/Robot2"], 2),
    (["/World/env_0/Robot", "/World/env_2/Robot"], 2),
    (["/World/env_0/Robot"], 2),
    (["/World/not_an_env/Robot"], 1),
    (["/World/env_0_extra/Robot"], 1),
])
def test_view_order_rejects_duplicate_missing_and_unrecognized_slots(paths, count):
    with pytest.raises(ValueError):
        view_env_order(paths, count)


@pytest.mark.parametrize("shape", [(224, 224, 3), (480, 640, 3)])
def test_fast_image_encoding_is_lossless_rgb_png(shape):
    rgb = np.random.default_rng(7).integers(0, 256, shape, dtype=np.uint8)
    encoded = encode_fast_image(rgb)
    raw = base64.b64decode(encoded, validate=True)
    assert raw.startswith(b"\x89PNG\r\n\x1a\n")
    with Image.open(BytesIO(raw)) as decoded:
        assert decoded.mode == "RGB"
        np.testing.assert_array_equal(np.asarray(decoded), rgb)


def test_four_by_three_render_uses_original_policy_square_resize_geometry():
    # Two landmarks at 1/3 and 2/3 of the trained vertical field of view
    # must stay there. Native square RTX rendering displaced them toward
    # the centre by a factor of .75 despite unchanged USD apertures.
    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    rgb[79:82, :, 0] = 255
    rgb[159:162, :, 1] = 255
    with Image.open(BytesIO(base64.b64decode(encode_fast_image(rgb)))) as frame:
        assert frame.size == (224, 224)
        pixels = np.asarray(frame)
    assert abs(int(np.argmax(pixels[:, 112, 0])) - 75) <= 1
    assert abs(int(np.argmax(pixels[:, 112, 1])) - 149) <= 1


@pytest.mark.parametrize("rgb", [
    np.zeros((224, 224, 3), dtype=np.float32),
    np.zeros((224, 224, 4), dtype=np.uint8),
    np.zeros((640, 480, 3), dtype=np.uint8),
    np.zeros((32, 32, 3), dtype=np.uint8),
])
def test_fast_image_rejects_wrong_dtype_resolution_and_channel_count(rgb):
    with pytest.raises(ValueError):
        encode_fast_image(rgb)


def test_parallel_image_encoding_preserves_environment_and_camera_order():
    from concurrent.futures import ThreadPoolExecutor
    from pressb.online_rl.fast_simulation import FastIsaacVectorBackend
    backend = object.__new__(FastIsaacVectorBackend)
    backend.timing = {"encode_seconds": 0.}
    tiles = [np.full((224, 224, 3), index, dtype=np.uint8) for index in range(6)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        backend._image_pool = pool
        result = backend._encode_images([2, 0], tiles)
    for eid in (2, 0):
        for view, index in (("global", 2 * eid + 1), ("wrist", 2 * eid)):
            with Image.open(BytesIO(base64.b64decode(result[eid][view]))) as decoded:
                np.testing.assert_array_equal(np.asarray(decoded), tiles[index])


def test_raw_tensor_arm_setter_preserves_unselected_rows():
    from pressb.online_rl.fast_simulation import _TensorArms
    class View:
        def __init__(self):
            self.positions = np.arange(18, dtype=np.float32).reshape(3, 6)
        def get_dof_positions(self):
            return self.positions
        def set_dof_positions(self, values, indices):
            assert indices.dtype == np.uint32
            assert values.shape == (3, 6)
            np.testing.assert_array_equal(self.positions[1], np.arange(6, 12))
            self.positions[indices] = values[indices]
    arm = object.__new__(_TensorArms)
    arm.view = View()
    before = arm.view.positions.copy()
    arm.set_joint_positions(np.full((1, 6), 99.), indices=np.array([1]))
    np.testing.assert_array_equal(arm.view.positions[[0, 2]], before[[0, 2]])
    np.testing.assert_array_equal(arm.view.positions[1], np.full(6, 99.))


def test_raw_button_pose_setter_uses_xyzw_and_preserves_peer():
    from pressb.online_rl.fast_simulation import _TensorButtons
    class View:
        def __init__(self):
            self.poses = np.array([[1., 2., 3., 0., 0., 0., 1.], [4., 5., 6., 0., 0., 0., 1.]])
        def get_transforms(self):
            return self.poses
        def set_transforms(self, values, indices):
            assert values.shape == (2, 7)
            self.poses[indices] = values[indices]
    buttons = object.__new__(_TensorButtons)
    buttons.view = View()
    old_positions, old_quaternions = buttons.get_world_poses()
    before = buttons.view.poses.copy()
    buttons.set_world_poses(np.array([[7., 8., 9.]]), np.array([[.5, np.sqrt(.75), 0., 0.]]), indices=np.array([0]))
    np.testing.assert_array_equal(buttons.view.poses[1], before[1])
    np.testing.assert_array_equal(buttons.view.poses[0], [7., 8., 9., np.sqrt(.75), 0., 0., .5])
    np.testing.assert_array_equal(old_positions, before[:, :3])
    np.testing.assert_array_equal(old_quaternions, [[1., 0., 0., 0.], [1., 0., 0., 0.]])


@pytest.mark.parametrize("actors,colliders,button_pair,reasons", [
    (("link6", "button"), ("link6/PressStylus", "button"), (0, 24), ()),
    (("link3", "button"), ("link3/collisions", "button"), None, ((0, "non_stylus_button_contact"),)),
    (("link2", "link3"), ("link2/collisions", "link3/collisions"), None, ((0, "robot_self_contact"),)),
    (("link6", "link7"), ("link6/PressStylus", "link7/collisions"), None, ()),
    (("link1", "wall"), ("link1/collisions", "wall"), None, ()),
    (("link6", "wall"), ("link6/collisions", "wall"), None, ((0, "robot_environment_contact"),)),
    (("button", "wall"), ("button", "wall"), None, ()),
    (("link6", "link6"), ("link6/a", "link6/b"), None, ()),
])
def test_contact_classification_preserves_physical_allowlists(actors, colliders, button_pair, reasons):
    from types import SimpleNamespace
    from pressb.online_rl.fast_simulation import FastIsaacVectorBackend
    backend = object.__new__(FastIsaacVectorBackend)
    prefix = "/World/envs/env_0/Piper"
    button = "/World/envs/env_0/Panel/Floor24/Cap"
    def path(value):
        return button if value == "button" else "/World/Wall" if value == "wall" else prefix + "/" + value
    backend.envs = [SimpleNamespace(robot_path=prefix)]
    backend.body_map = {button: (0, 24)}
    backend.tool_paths = {prefix + "/link6/PressStylus", prefix + "/link6/PressTip"}
    assert backend._classify_contact(list(map(path, actors)), list(map(path, colliders))) == (button_pair, reasons)


@pytest.mark.parametrize("count", [1, 2, 64, 128, 1024])
def test_gpu_capacity_plan_has_explicit_headroom_and_scales(count):
    from pressb.online_rl.simulation import gpu_dynamics_memory_config
    config = gpu_dynamics_memory_config(count)
    assert config["found_lost_aggregate_pairs_capacity"] >= max(65536, count * 256)
    assert config["max_rigid_contact_count"] >= count * 16384
    assert all(value > 0 and value & (value - 1) == 0 for value in config.values())
    for name, value in config.items():
        assert value >= gpu_dynamics_memory_config(1)[name]


def _physics_guard(tmp_path):
    from pressb.online_rl.fast_simulation import PhysXErrorGuard
    path = tmp_path / "native.kit.log"
    path.write_text("")
    def append(message):
        with path.open("a") as stream:
            stream.write(message)
    return PhysXErrorGuard(path), append


@pytest.mark.parametrize("level,message", [
    ("Error", "increase foundLostAggregatePairsCapacity to 3072"),
    ("Warning", "insufficient GPU buffer: simulation will miss interactions"),
    ("Warning", "Some contacts will be dropped because GPU buffers are full"),
    ("Warning", "Increase PxGpuDynamicsMemoryConfig allocation"),
    ("Error", "PhysX unexpected internal failure"),
])
def test_native_physics_error_latches_and_cannot_return_healthy(tmp_path, level, message):
    guard, append = _physics_guard(tmp_path)
    append(f"2026-10-02 [123ms] [{level}] [omni.physx.plugin] {message}\n")
    with pytest.raises(RuntimeError, match="PhysX invalidated"):
        guard.check()
    guard.close()
    guard.close()


def test_gpu_guard_ignores_documented_ccd_warning_and_nonphysics_driver_noise(tmp_path):
    guard, append = _physics_guard(tmp_path)
    append("[Warning] [omni.physx.plugin] Disabling CCD for GPU dynamics as its not supported\n")
    append("[Error] [warp] optional driver function unavailable\n")
    guard.check()
    assert not guard.errors
    guard.close()


def test_async_physics_error_changes_service_health_and_blocks_next_mutation(tmp_path):
    backend = FakeBackend()
    service = IndexedSimulationService(backend)
    guard, append = _physics_guard(tmp_path)
    backend._physics_error_guard = guard
    append("[Error] [omni.physx.plugin] GPU capacity exceeded\n")
    health = service.health()
    assert not health["ready"] and "capacity exceeded" in health["fault"]
    with pytest.raises(RuntimeError, match="faulted"):
        service.reset(reset_request())
    assert not backend.reset_calls
    guard.close()


def test_guard_reads_existing_and_split_native_log_records(tmp_path):
    guard, append = _physics_guard(tmp_path)
    append("[Info] [omni.physx.plugin] GPU initialized\n[Er")
    guard.check()
    append("ror] [omni.physx.plugin] GPU capacity exceeded")
    with pytest.raises(RuntimeError, match="capacity exceeded"):
        guard.check()
    assert len(guard.errors) == 1
    with pytest.raises(RuntimeError):
        guard.check()
    assert len(guard.errors) == 1
    guard.close()


@pytest.mark.parametrize("operation", ["remove", "truncate", "replace"])
def test_guard_fails_closed_when_native_log_evidence_is_lost(tmp_path, operation):
    guard, append = _physics_guard(tmp_path)
    append("[Info] started\n")
    guard.check()
    if operation == "remove":
        guard.path.unlink()
    elif operation == "truncate":
        guard.path.write_text("")
    else:
        guard.path.rename(tmp_path / "old.log")
        guard.path.write_text("[Info] replacement log\n")
    with pytest.raises(RuntimeError, match="PhysX invalidated"):
        guard.check()
    guard.close()
