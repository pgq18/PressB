"""CPU contract tests for vector simulation, independent of Isaac and policy inference."""

import base64
from collections import deque
from copy import deepcopy
from io import BytesIO
import json

import numpy as np
from PIL import Image
import pytest

from pressb.online_rl.simulation import SimulationService, control_state


def _rgb_png():
    buffer = BytesIO()
    Image.new("RGB", (640, 480)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


PNG = _rgb_png()


class FakeBackend:
    """Record real service calls and emit explicitly scheduled physical events."""

    def __init__(self, num_envs=2, smoothing_window=3):
        self.num_envs = num_envs
        self.smoothing_window = smoothing_window
        self.reset_calls = []
        self.step_calls = []
        self.transitions = deque()
        self.last_result = None
        self.reset_error = None
        self.episodes = None
        self.config = {
            "panel_offset_x_m": 0.0,
            "panel_offset_y_m": 0.0,
            "panel_randomization": {
                "enabled": True,
                "min_offset_x_m": -0.01,
                "max_offset_x_m": 0.01,
                "min_offset_y_m": -0.025,
                "max_offset_y_m": 0.025,
            },
        }

    def health(self):
        return {
            "num_envs": self.num_envs,
            "smoothing_window": self.smoothing_window,
            "max_seconds": 15.0,
            "config": self.config,
            "identity": {"scene_sha256": "fake-scene", "collection_id": "fake-collection"},
        }

    def observation(self, env_id, marker=0):
        floor = self.episodes[env_id]["floor"] if self.episodes is not None else 24 + env_id
        return {
            "task": f"Press {floor} floor.",
            "state": [0.3 + marker / 1000, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0, 0.008],
            "images": {"global": PNG, "wrist": PNG},
            "control_state": [marker / 1000] * (19 + 6 * self.smoothing_window),
        }

    def reset(self, episodes, seed):
        self.reset_calls.append((deepcopy(episodes), seed))
        if self.reset_error is not None:
            raise self.reset_error
        self.episodes = deepcopy(episodes)
        self.last_result = [
            {"env_id": env_id, "observation": self.observation(env_id), "info": {"seed": seed}}
            for env_id in range(self.num_envs)
        ]
        return self.last_result

    def step(self, actions):
        self.step_calls.append({env_id: np.array(value, copy=True) for env_id, value in actions.items()})
        outcomes = self.transitions.popleft() if self.transitions else {}
        if isinstance(outcomes, Exception):
            raise outcomes
        result = []
        for env_id in actions:
            reason, ticks = outcomes.get(env_id, (None, 28))
            result.append({
                "env_id": env_id,
                "observation": self.observation(env_id, marker=len(self.step_calls) * 100 + env_id),
                "terminated": reason in {"target_pressed", "wrong_button_pressed", "unexpected_collision"},
                "truncated": reason == "time_limit",
                "executed_physics_steps": ticks,
                "info": {"termination": reason, "success": reason == "target_pressed"},
            })
        self.last_result = result
        return result


def reset_request(request_id="reset-0", run_id="run-a", num_envs=2):
    return {
        "run_id": run_id,
        "request_id": request_id,
        "seed": 17,
        "episodes": [{"floor": 24 + env_id, "offset_x_m": 0.0, "offset_y_m": 0.0}
                     for env_id in range(num_envs)],
    }


def action_chunk():
    return [[0.3, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0, 0.008] for _ in range(7)]


def step_request(previous, request_id="step-0", run_id="run-a", active=(0, 1)):
    return {
        "run_id": run_id,
        "request_id": request_id,
        "cohort_id": previous["cohort_id"],
        "step_id": previous["step_id"],
        "actions": [{"env_id": env_id, "actions_pose8": action_chunk()} for env_id in active],
    }


@pytest.fixture
def service_pair():
    backend = FakeBackend()
    return SimulationService(backend, single_gamma=0.9), backend


@pytest.mark.parametrize("window", [1, 3, 5, 11])
def test_health_exposes_physical_and_control_state_contract_without_reset(window):
    backend = FakeBackend(smoothing_window=window)
    service = SimulationService(backend, single_gamma=0.97)
    health = service.health({})
    assert health["protocol_version"] == 1
    assert health["service"] == "simulation"
    assert health["ready"] is True
    assert health["num_envs"] == 2
    assert health["physics_hz"] == 120
    assert health["action_fps"] == 30
    assert health["action_horizon"] == 7
    assert health["smoothing_window"] == window
    assert health["control_state_dim"] == 19 + 6 * window
    assert health["max_seconds"] == 15
    assert health["single_gamma"] == 0.97
    json.dumps(health, allow_nan=False)
    assert backend.reset_calls == backend.step_calls == []


@pytest.mark.parametrize("window", [1, 3, 5, 11])
def test_control_state_preserves_measured_state_command_history_and_normalized_clock(window):
    actual_q = np.arange(6) / 10
    actual_velocity = np.arange(6) / 20 - 0.3
    endpoint = actual_q + 0.05
    # Each row is a distinct time, making history reversal visible to this test.
    history = np.stack([actual_q + time / 100 for time in range(window)])
    result = control_state(actual_q, actual_velocity, endpoint, history, elapsed=3, maximum=15)
    assert len(result) == 19 + 6 * window
    np.testing.assert_array_equal(result[:6], actual_q)
    np.testing.assert_array_equal(result[6:12], actual_velocity)
    np.testing.assert_array_equal(result[12:18], endpoint)
    np.testing.assert_array_equal(np.asarray(result[18:-1]).reshape(window, 6), history)
    assert result[-1] == 0.2
    assert control_state(actual_q, actual_velocity, endpoint, history, 0, 15)[-1] == 0
    assert control_state(actual_q, actual_velocity, endpoint, history, 15, 15)[-1] == 1
    snapshot = deepcopy(result)
    actual_q[:] = 100
    history[:] = -100
    assert result == snapshot
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("argument,value", [
    ("actual_q", [0.0] * 5),
    ("actual_velocity", [[0.0] * 6]),
    ("ik_endpoint", [float("nan")] * 6),
    ("linear_history", [0.0] * 6),
    ("linear_history", [[float("inf")] * 6] * 3),
    ("elapsed", -0.1),
    ("elapsed", float("nan")),
    ("maximum", 0),
    ("maximum", float("inf")),
])
def test_control_state_rejects_invalid_measured_inputs_or_episode_time(argument, value):
    inputs = dict(actual_q=np.zeros(6), actual_velocity=np.zeros(6), ik_endpoint=np.zeros(6),
                  linear_history=np.zeros((3, 6)), elapsed=1, maximum=15)
    inputs[argument] = value
    with pytest.raises(ValueError):
        control_state(**inputs)


def test_reset_creates_full_vector_with_unique_episode_ids(service_pair):
    service, backend = service_pair
    payload = reset_request()
    response = service.reset(payload)
    assert response["protocol_version"] == 1
    assert response["cohort_id"]
    assert response["step_id"] == 0
    assert response["all_done"] is False
    assert [item["env_id"] for item in response["items"]] == [0, 1]
    assert len({item["episode_id"] for item in response["items"]}) == 2
    assert all(not item["terminated"] and not item["truncated"] for item in response["items"])
    assert all(len(item["observation"]["control_state"]) == 37 for item in response["items"])
    assert backend.reset_calls == [(payload["episodes"], 17)]
    assert backend.step_calls == []
    json.dumps(response, allow_nan=False)


@pytest.mark.parametrize("ticks", [1, 2, 4, 5, 27, 28])
def test_success_reward_and_discount_use_actual_partial_physics_duration(service_pair, ticks):
    service, backend = service_pair
    initial = service.reset(reset_request())
    backend.transitions.append({0: ("target_pressed", ticks)})
    response = service.step(step_request(initial))
    success, continuing = response["items"]
    assert response["protocol_version"] == 1
    assert response["step_id"] == 1
    assert response["all_done"] is False
    assert success["episode_id"] == initial["items"][0]["episode_id"]
    assert success["terminated"] is True and success["truncated"] is False
    assert success["info"]["success"] is True
    assert success["reward"] == pytest.approx(0.9 ** ((ticks - 1) / 4))
    assert success["discount"] == pytest.approx(0.9 ** (ticks / 4))
    assert success["executed_physics_steps"] == ticks
    assert continuing["reward"] == 0
    assert continuing["discount"] == pytest.approx(0.9 ** 7)
    assert continuing["executed_physics_steps"] == 28
    assert continuing["terminated"] is False and continuing["truncated"] is False
    json.dumps(response, allow_nan=False)


@pytest.mark.parametrize("reason,terminated,truncated", [
    ("wrong_button_pressed", True, False),
    ("unexpected_collision", True, False),
    ("time_limit", False, True),
])
def test_non_success_events_remain_sparse_and_timeout_is_truncation(service_pair, reason, terminated, truncated):
    service, backend = service_pair
    initial = service.reset(reset_request())
    backend.transitions.append({0: (reason, 3), 1: (reason, 19)})
    result = service.step(step_request(initial))
    assert result["all_done"] is True
    for item, ticks in zip(result["items"], [3, 19]):
        assert item["reward"] == 0
        assert item["discount"] == pytest.approx(0.9 ** (ticks / 4))
        assert item["info"]["success"] is False
        assert item["info"]["termination"] == reason
        assert item["terminated"] is terminated
        assert item["truncated"] is truncated


def test_terminal_observation_is_frozen_across_later_vector_steps_and_caller_mutation(service_pair):
    service, backend = service_pair
    initial = service.reset(reset_request())
    backend.transitions.append({0: ("target_pressed", 3)})
    first = service.step(step_request(initial))
    terminal = deepcopy(first["items"][0])
    # Neither a backend recycling its response buffers nor an HTTP caller may
    # corrupt the stored event observation used by subsequent vector replies.
    backend.last_result[0]["observation"]["state"][0] = -10
    backend.last_result[0]["observation"]["control_state"][0] = -10
    first["items"][0]["observation"]["state"][0] = 10
    first["items"][0]["info"]["success"] = False
    second = service.step(step_request(first, "step-1", active=(1,)))
    cached = second["items"][0]
    assert set(backend.step_calls[-1]) == {1}
    assert cached["observation"] == terminal["observation"]
    assert cached["info"] == terminal["info"]
    assert cached["episode_id"] == terminal["episode_id"]
    assert cached["terminated"] is True and cached["truncated"] is False
    assert cached["reward"] == 0
    assert cached["executed_physics_steps"] == 0
    assert cached["discount"] == 1
    assert second["items"][1]["observation"] != first["items"][1]["observation"]
    assert second["step_id"] == 2


def test_full_barrier_blocks_reset_until_all_environments_end(service_pair):
    service, backend = service_pair
    initial = service.reset(reset_request())
    with pytest.raises((ValueError, RuntimeError)):
        service.reset(reset_request("too-early"))
    backend.transitions.append({0: ("target_pressed", 1)})
    one_done = service.step(step_request(initial))
    with pytest.raises((ValueError, RuntimeError)):
        service.reset(reset_request("only-one-done"))
    assert len(backend.reset_calls) == 1
    backend.transitions.append({1: ("time_limit", 2)})
    all_done = service.step(step_request(one_done, "step-1", active=(1,)))
    assert all_done["all_done"] is True
    next_initial = service.reset(reset_request("reset-1"))
    assert len(backend.reset_calls) == 2
    assert next_initial["step_id"] == 0
    assert next_initial["cohort_id"] != initial["cohort_id"]
    previous_ids = {item["episode_id"] for item in initial["items"]}
    assert previous_ids.isdisjoint(item["episode_id"] for item in next_initial["items"])
    assert all(not item["terminated"] and not item["truncated"] for item in next_initial["items"])


def test_close_run_requires_every_environment_to_end(service_pair):
    service, backend = service_pair
    initial = service.reset(reset_request())
    with pytest.raises((ValueError, RuntimeError)):
        service.close_run({"run_id": "run-a", "request_id": "close-before-step"})
    backend.transitions.append({0: ("target_pressed", 1)})
    first = service.step(step_request(initial))
    with pytest.raises((ValueError, RuntimeError)):
        service.close_run({"run_id": "run-a", "request_id": "close-while-active"})
    assert service.health({})["run_id"] == "run-a"
    assert len(backend.reset_calls) == 1 and len(backend.step_calls) == 1
    backend.transitions.append({1: ("time_limit", 1)})
    assert service.step(step_request(first, "finish-run", active=(1,)))["all_done"] is True


def test_close_run_releases_finished_session_for_a_fresh_run(service_pair):
    service, backend = service_pair
    initial = service.reset(reset_request())
    backend.transitions.append({0: ("target_pressed", 3), 1: ("time_limit", 9)})
    assert service.step(step_request(initial))["all_done"] is True
    response = service.close_run({"run_id": "run-a", "request_id": "close-a"})
    assert response["protocol_version"] == 1
    health = service.health({})
    assert health["ready"] is True
    assert health["run_id"] is None
    assert health["cohort_id"] is None
    assert len(backend.reset_calls) == 1 and len(backend.step_calls) == 1
    next_initial = service.reset(reset_request("reset-b", run_id="run-b"))
    assert next_initial["step_id"] == 0
    assert next_initial["cohort_id"] != initial["cohort_id"]
    assert service.health({})["run_id"] == "run-b"
    assert len(backend.reset_calls) == 2


def test_closed_run_cannot_be_reused_and_old_cohort_never_addresses_new_run(service_pair):
    service, backend = service_pair
    initial = service.reset(reset_request())
    backend.transitions.append({0: ("time_limit", 1), 1: ("time_limit", 1)})
    service.step(step_request(initial))
    service.close_run({"run_id": "run-a", "request_id": "close-a"})
    # Check retirement while there is no owner, so failure cannot merely be an
    # ownership conflict with the next run.
    with pytest.raises(ValueError):
        service.reset(reset_request("retired-run-reset"))
    with pytest.raises(ValueError):
        service.step(step_request(initial, "retired-run-step"))
    next_initial = service.reset(reset_request("reset-b", run_id="run-b"))
    with pytest.raises(ValueError):
        service.step(step_request(initial, "old-cohort-new-owner", run_id="run-b"))
    with pytest.raises(ValueError):
        service.step(step_request(next_initial, "new-cohort-old-owner", run_id="run-a"))
    assert len(backend.reset_calls) == 2 and len(backend.step_calls) == 1
    response = service.step(step_request(next_initial, "valid-new-run-step", run_id="run-b"))
    assert response["step_id"] == 1


def test_close_run_cannot_release_another_runs_session(service_pair):
    service, backend = service_pair
    initial = service.reset(reset_request())
    backend.transitions.append({0: ("time_limit", 1), 1: ("time_limit", 1)})
    service.step(step_request(initial))
    with pytest.raises(ValueError):
        service.close_run({"run_id": "run-b", "request_id": "close-other-run"})
    assert service.health({})["run_id"] == "run-a"
    assert service.health({})["cohort_id"] == initial["cohort_id"]


def test_reset_idempotency_replays_detached_response_without_reentering_backend(service_pair):
    service, backend = service_pair
    payload = reset_request()
    first = service.reset(payload)
    expected = deepcopy(first)
    first["items"][0]["observation"]["state"][0] = -999
    backend.last_result[0]["observation"]["control_state"][0] = -999
    replayed = service.reset(deepcopy(payload))
    assert replayed == expected
    assert len(backend.reset_calls) == 1
    replayed["items"][1]["info"]["seed"] = -999
    assert service.reset(payload) == expected
    assert len(backend.reset_calls) == 1


def test_step_idempotency_replays_detached_response_after_sequence_has_advanced(service_pair):
    service, backend = service_pair
    initial = service.reset(reset_request())
    payload = step_request(initial)
    first = service.step(payload)
    expected = deepcopy(first)
    service.step(step_request(first, "step-1"))
    first["items"][0]["observation"]["state"][0] = -999
    replayed = service.step(deepcopy(payload))
    assert replayed == expected
    assert len(backend.step_calls) == 2
    replayed["items"][1]["observation"]["control_state"][0] = -999
    assert service.step(payload) == expected
    assert len(backend.step_calls) == 2


def test_expired_reset_request_is_never_executed_again_when_barrier_allows_a_new_reset():
    backend = FakeBackend()
    service = SimulationService(backend, response_cache_size=1)
    payload = reset_request()
    initial = service.reset(payload)
    backend.transitions.append({0: ("time_limit", 1), 1: ("time_limit", 1)})
    assert service.step(step_request(initial))["all_done"] is True
    # The one-slot response cache now contains step's reply. Retaining request
    # identity must prevent a delayed reset retry from creating another cohort.
    with pytest.raises(ValueError):
        service.reset(payload)
    assert len(backend.reset_calls) == 1
    assert service.health({})["cohort_id"] == initial["cohort_id"]


def test_request_id_cannot_be_reused_with_a_different_payload_or_operation(service_pair):
    service, backend = service_pair
    initial = service.reset(reset_request())
    changed_reset = reset_request()
    changed_reset["seed"] += 1
    with pytest.raises(ValueError):
        service.reset(changed_reset)
    with pytest.raises(ValueError):
        service.step(step_request(initial, "reset-0"))
    payload = step_request(initial)
    first = service.step(payload)
    changed_action = deepcopy(payload)
    changed_action["actions"][0]["actions_pose8"][0][0] += 0.001
    with pytest.raises(ValueError):
        service.step(changed_action)
    with pytest.raises(ValueError):
        service.step(step_request(first, "step-0"))
    assert len(backend.reset_calls) == 1
    assert len(backend.step_calls) == 1


@pytest.mark.parametrize("operation", ["reset", "step"])
def test_server_session_rejects_a_different_run(service_pair, operation):
    service, backend = service_pair
    initial = service.reset(reset_request())
    if operation == "reset":
        # Test run ownership after the reset barrier is otherwise satisfied.
        backend.transitions.append({0: ("time_limit", 1), 1: ("time_limit", 1)})
        service.step(step_request(initial))
    payload = (reset_request("other-reset", run_id="run-b") if operation == "reset"
               else step_request(initial, "other-step", run_id="run-b"))
    with pytest.raises((ValueError, RuntimeError)):
        getattr(service, operation)(payload)
    assert len(backend.reset_calls) == 1
    assert len(backend.step_calls) == (1 if operation == "reset" else 0)


def test_step_before_first_reset_cannot_enter_backend(service_pair):
    service, backend = service_pair
    placeholder = {"cohort_id": "not-yet-created", "step_id": 0}
    with pytest.raises((ValueError, RuntimeError)):
        service.step(step_request(placeholder))
    assert backend.reset_calls == backend.step_calls == []


def test_env_ids_align_results_even_when_actions_arrive_in_reverse_order(service_pair):
    service, backend = service_pair
    initial = service.reset(reset_request())
    payload = step_request(initial, active=(1, 0))
    payload["actions"][0]["actions_pose8"][0][0] = 0.7
    response = service.step(payload)
    assert [item["env_id"] for item in response["items"]] == [0, 1]
    assert backend.step_calls[0][1][0, 0] == 0.7
    assert backend.step_calls[0][0][0, 0] == 0.3
    assert response["items"][0]["observation"]["task"] == "Press 24 floor."
    assert response["items"][1]["observation"]["task"] == "Press 25 floor."


@pytest.mark.parametrize("field,value", [("cohort_id", "unknown-cohort"), ("step_id", -1), ("step_id", 1)])
def test_stale_or_future_sequence_is_rejected_before_physics(service_pair, field, value):
    service, backend = service_pair
    initial = service.reset(reset_request())
    payload = step_request(initial)
    payload[field] = value
    with pytest.raises(ValueError):
        service.step(payload)
    assert backend.step_calls == []
    assert service.step(step_request(initial, "valid-after-error"))["step_id"] == 1


def test_old_cohort_and_old_step_cannot_be_reexecuted_under_new_request_ids(service_pair):
    service, backend = service_pair
    initial = service.reset(reset_request())
    first = service.step(step_request(initial))
    with pytest.raises(ValueError):
        service.step(step_request(initial, "repeat-stale-step"))
    backend.transitions.append({0: ("time_limit", 1), 1: ("time_limit", 1)})
    service.step(step_request(first, "finish-cohort"))
    next_initial = service.reset(reset_request("reset-1"))
    with pytest.raises(ValueError):
        service.step(step_request(initial, "repeat-old-cohort"))
    assert len(backend.step_calls) == 2
    assert service.step(step_request(next_initial, "new-cohort-step"))["step_id"] == 1


@pytest.mark.parametrize("bad_chunk", [
    [],
    [[0.0] * 8] * 6,
    [[0.0] * 7] * 7,
    [[0.0] * 8] * 8,
    [[0.3, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0, float("nan")]] * 7,
    [[float("inf"), 0.0, 0.4, 1.0, 0.0, 0.0, 0.0, 0.008]] * 7,
    [[0.3, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.008]] * 7,
    "not-an-action-array",
])
def test_entire_action_batch_is_validated_before_any_physics(service_pair, bad_chunk):
    service, backend = service_pair
    initial = service.reset(reset_request())
    payload = step_request(initial)
    # The first environment is valid: validation must finish before its actions
    # can execute, even when a later item is malformed.
    payload["actions"][-1]["actions_pose8"] = bad_chunk
    with pytest.raises(ValueError):
        service.step(payload)
    assert backend.step_calls == []
    assert service.health({})["ready"] is True
    result = service.step(step_request(initial, "valid-after-error"))
    assert result["step_id"] == 1
    assert len(backend.step_calls) == 1


@pytest.mark.parametrize("env_ids", [(0,), (0, 0), (0, 1, 2), (0, 2), (-1, 1), (0, True), (0, 1.0)])
def test_actions_must_cover_exactly_the_active_vector(service_pair, env_ids):
    service, backend = service_pair
    initial = service.reset(reset_request())
    with pytest.raises(ValueError):
        service.step(step_request(initial, active=env_ids))
    assert backend.step_calls == []


def test_actions_for_an_already_done_environment_are_rejected(service_pair):
    service, backend = service_pair
    initial = service.reset(reset_request())
    backend.transitions.append({0: ("target_pressed", 1)})
    first = service.step(step_request(initial))
    with pytest.raises(ValueError):
        service.step(step_request(first, "includes-dead-env"))
    assert len(backend.step_calls) == 1
    assert service.step(step_request(first, "active-only", active=(1,)))["step_id"] == 2


@pytest.mark.parametrize("num_episodes", [0, 1, 3])
def test_reset_requires_exactly_num_envs_episodes(service_pair, num_episodes):
    service, backend = service_pair
    with pytest.raises(ValueError):
        service.reset(reset_request(num_envs=num_episodes))
    assert backend.reset_calls == []
    # A rejected initial request must not seize run ownership.
    result = service.reset(reset_request("valid-reset", run_id="run-b"))
    assert result["step_id"] == 0
    assert len(backend.reset_calls) == 1


@pytest.mark.parametrize("field,value", [
    ("floor", 23), ("floor", 36), ("floor", True), ("floor", 24.5),
    ("offset_x_m", -0.010001), ("offset_x_m", 0.010001),
    ("offset_y_m", -0.025001), ("offset_y_m", 0.025001),
    ("offset_x_m", float("nan")), ("offset_y_m", float("inf")),
])
def test_reset_rejects_out_of_training_distribution_before_backend(service_pair, field, value):
    service, backend = service_pair
    payload = reset_request()
    payload["episodes"][-1][field] = value
    with pytest.raises(ValueError):
        service.reset(payload)
    assert backend.reset_calls == []
    assert service.health({})["ready"] is True
    service.reset(reset_request("valid-reset", run_id="run-b"))
    assert len(backend.reset_calls) == 1


def test_reset_accepts_inclusive_training_boundaries(service_pair):
    service, backend = service_pair
    payload = reset_request()
    payload["episodes"] = [
        {"floor": 24, "offset_x_m": -0.01, "offset_y_m": -0.025},
        {"floor": 35, "offset_x_m": 0.01, "offset_y_m": 0.025},
    ]
    service.reset(payload)
    assert backend.reset_calls == [(payload["episodes"], payload["seed"])]


def test_fixed_layout_configuration_rejects_offset_changes():
    backend = FakeBackend()
    backend.config["panel_randomization"]["enabled"] = False
    backend.config["panel_offset_x_m"] = 0.004
    backend.config["panel_offset_y_m"] = -0.003
    service = SimulationService(backend)
    with pytest.raises(ValueError):
        service.reset(reset_request())
    assert backend.reset_calls == []
    payload = reset_request("fixed-layout")
    for episode in payload["episodes"]:
        episode.update(offset_x_m=0.004, offset_y_m=-0.003)
    service.reset(payload)
    assert backend.reset_calls == [(payload["episodes"], payload["seed"])]


@pytest.mark.parametrize("operation", ["reset", "step"])
def test_backend_failure_faults_the_service_without_fabricating_episode_timeout(service_pair, operation):
    service, backend = service_pair
    if operation == "reset":
        backend.reset_error = RuntimeError("physics initialization failed")
        payload = reset_request()
    else:
        initial = service.reset(reset_request())
        backend.transitions.append(RuntimeError("renderer lost its device"))
        payload = step_request(initial)
    with pytest.raises(RuntimeError):
        getattr(service, operation)(payload)
    assert service.health({})["ready"] is False
    call_counts = (len(backend.reset_calls), len(backend.step_calls))
    # A fault is sticky, including retries, and requires a fresh server process.
    with pytest.raises(RuntimeError):
        getattr(service, operation)(payload)
    with pytest.raises(RuntimeError):
        service.reset(reset_request("after-fault"))
    with pytest.raises(RuntimeError):
        service.step({"run_id": "run-a", "request_id": "step-after-fault", "cohort_id": "unused",
                      "step_id": 0, "actions": []})
    assert (len(backend.reset_calls), len(backend.step_calls)) == call_counts
