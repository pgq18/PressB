"""Validated rollout observer behavior with CPU-only fake node clients."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pressb.runtime.trainer import GenericTrainer
from test_rl_nodes_trainer import make_runner, state_only_runner


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class RecordingObserver:
    def __init__(self):
        self.events = []
        self.resets = []
        self.transitions = []
        self.statuses = []

    def on_reset(self, rows):
        self.events.append("reset")
        self.resets.append(rows)

    def on_transition(self, before, actions_executed, rows):
        self.events.append("transition")
        self.transitions.append((before, actions_executed, rows))

    def close(self, status):
        self.events.append("close")
        self.statuses.append(status)


def observed_runner(output, observer, *, method="base", steps=8, fused=False):
    factory = state_only_runner if fused else make_runner
    template, inference, simulation = factory(output, method=method, steps=steps)
    template.config.rollout.fused_step_reset = fused
    runner = GenericTrainer(template.config, inference_client=inference,
                            simulation_client=simulation, observer=observer)
    return runner, inference, simulation


@pytest.mark.parametrize("missing", ["on_reset", "on_transition", "close"])
def test_observer_requires_all_callbacks_before_starting(tmp_path, missing):
    observer = SimpleNamespace(on_reset=lambda rows: None,
                               on_transition=lambda before, actions, rows: None,
                               close=lambda status: None)
    setattr(observer, missing, None)
    with pytest.raises(TypeError, match=rf"observer\.{missing}.*callable"):
        observed_runner(tmp_path, observer)
    assert not list(tmp_path.iterdir())


def test_base_evaluation_observer_gets_ordered_validated_clipped_prefixes(tmp_path):
    observer = RecordingObserver()
    runner, inference, simulation = observed_runner(tmp_path, observer, steps=4)
    simulation.meta["num_envs"] = 1
    simulation.meta["action"].update(lower=[-.1] * 7, upper=[.1] * 7)
    runner.config.rollout.action_bounds = "clip"
    runner.config.rollout.eval_episodes = 2
    cleanup_at_close = []
    original_close = observer.close

    def close_after_cleanup(status):
        cleanup_at_close.append((not inference.contexts, not runner.live_contexts,
                                 getattr(simulation, "closed_run_id", None)))
        original_close(status)

    observer.close = close_after_cleanup
    result = runner.run()
    assert result["state"] == "completed" and result["episodes"] == 2
    assert observer.events == ["reset", "transition", "reset", "transition", "close"]
    assert observer.statuses == [result]
    assert cleanup_at_close == [(True, True, runner.run_id)]
    for (before, actions, rows), request, reset_rows in zip(
            observer.transitions, simulation.steps, observer.resets, strict=True):
        assert set(before[0]) >= {"observation", "encoded", "vector", "base"}
        assert before[0]["episode_id"] == reset_rows[0]["episode_id"]
        assert before[0]["observation"] == reset_rows[0]["observation"]
        assert before[0]["step_id"] == 0 and rows[0]["step_id"] == 1
        assert before[0]["episode_id"] == rows[0]["episode_id"]
        assert isinstance(actions, np.ndarray) and actions.shape == (1, 2, 7)
        np.testing.assert_allclose(actions, .1)
        np.testing.assert_allclose(actions, [item["actions"] for item in request["items"]])
        np.testing.assert_allclose(before[0]["base"], .2)
        assert rows[0]["info"]["elapsed_control_steps"] == .5
        assert rows[0]["terminated"]


def test_observer_cannot_mutate_live_slots_actions_rows_or_status(tmp_path):
    class MutatingObserver(RecordingObserver):
        def on_reset(self, rows):
            super().on_reset(deepcopy(rows))
            rows[-1]["observation"]["state"][0] = float("nan")
            rows[-1]["episode_id"] = "observer-overwrite"

        def on_transition(self, before, actions_executed, rows):
            super().on_transition(deepcopy(before), actions_executed.copy(), deepcopy(rows))
            before[-1]["observation"]["state"][0] = float("nan")
            before[-1]["encoded"]["context_id"] = "observer-overwrite"
            before[-1]["vector"][:] = float("nan")
            before[-1]["base"][:] = float("nan")
            actions_executed[:] = float("nan")
            rows[-1]["observation"]["state"][0] = float("nan")
            rows[-1]["discount"] = -100
            rows[-1]["env_id"] = 999

        def close(self, status):
            super().close(deepcopy(status))
            status["state"] = "observer-overwrite"

    observer = MutatingObserver()
    runner, inference, simulation = observed_runner(
        tmp_path, observer, method="action_residual", steps=4)
    result = runner.run()
    assert result["state"] == "completed" and result["transitions"] == 4
    assert runner.health()["state"] == "completed"
    assert json.loads((tmp_path / "status.json").read_text())["state"] == "completed"
    assert not inference.contexts and not runner.live_contexts
    assert len(observer.statuses) == 1
    for array in runner.replay.arrays.values():
        assert np.isfinite(array[:len(runner.replay)]).all()
    for request in simulation.steps:
        assert all(np.isfinite(item["actions"]).all() for item in request["items"])
        assert all(item["episode_id"] != "observer-overwrite" for item in request["items"])
    for request in inference.encode_calls:
        assert all(np.isfinite(row["state"]).all() for row in request["observations"])


@pytest.mark.parametrize("invalid_stage", ["reset", "step"])
def test_invalid_response_batch_is_never_sent_to_observer(tmp_path, invalid_stage):
    observer = RecordingObserver()
    runner, inference, simulation = observed_runner(tmp_path, observer)
    original = simulation.call

    def invalid_response(path, payload=None):
        response = original(path, payload)
        if path == f"/{invalid_stage}":
            # Corrupt the last row: no earlier row from this same batch may
            # have been recorded before validation discovers this failure.
            response["items"][-1]["observation"]["state"] = []
        return response

    simulation.call = invalid_response
    with pytest.raises(ValueError):
        runner.run()
    assert observer.events == (["reset"] if invalid_stage == "step" else []) + ["close"]
    assert not observer.transitions
    assert observer.statuses[0]["state"] == runner.health()["state"] == "failed"
    assert not inference.contexts and simulation.closed_run_id == runner.run_id


def test_fused_response_records_transition_before_its_new_episode_reset(tmp_path):
    observer = RecordingObserver()
    runner, inference, simulation = observed_runner(
        tmp_path, observer, method="dsrl", steps=3, fused=True)
    simulation.meta["num_envs"] = 1
    result = runner.run()
    assert result["state"] == "completed" and result["transitions"] == 3
    assert observer.events == ["reset", "transition", "reset", "transition",
                               "reset", "transition", "close"]
    assert len(simulation.fused_calls) == 3 and not inference.contexts
    for index, (before, actions, rows) in enumerate(observer.transitions):
        assert before[0]["episode_id"] == observer.resets[index][0]["episode_id"]
        assert rows[0]["episode_id"] == before[0]["episode_id"]
        assert rows[0]["terminal_observation_omitted"]
        assert rows[0]["observation"]["state"][0] == pytest.approx(index + 1.001)
        assert actions.shape == (1, 2, 7)
        if index + 1 < len(observer.resets):
            new = observer.resets[index + 1][0]
            assert new["episode_id"] != rows[0]["episode_id"]
            assert new["step_id"] == 0
            assert new["observation"]["state"][0] == index + 2


@pytest.mark.parametrize("invalid_part", ["items", "reset_items"])
def test_fused_response_validates_both_parts_before_recording_either(tmp_path, invalid_part):
    observer = RecordingObserver()
    runner, inference, simulation = observed_runner(
        tmp_path, observer, method="dsrl", steps=8, fused=True)
    original = simulation.call

    def invalid_response(path, payload=None):
        response = original(path, payload)
        if path == "/step_and_reset":
            assert response["reset_items"]
            response[invalid_part][-1]["observation"]["state"] = []
        return response

    simulation.call = invalid_response
    with pytest.raises(ValueError):
        runner.run()
    assert observer.events == ["reset", "close"]
    assert len(observer.resets) == 1 and not observer.transitions
    assert len(simulation.fused_calls) == 1 and len(simulation.steps) == 1
    assert observer.statuses[0]["state"] == "failed"
    assert not inference.contexts and simulation.closed_run_id == runner.run_id


@pytest.mark.parametrize("callback", ["on_reset", "on_transition"])
def test_observer_callback_failure_fails_run_and_closes_once(tmp_path, callback):
    observer = RecordingObserver()
    runner, inference, simulation = observed_runner(tmp_path, observer)
    failure = RuntimeError(f"fixture {callback} failure")

    def fail(*args):
        raise failure

    setattr(observer, callback, fail)
    with pytest.raises(RuntimeError) as caught:
        runner.run()
    assert caught.value is failure
    assert observer.events.count("close") == 1
    assert observer.statuses[0]["state"] == runner.health()["state"] == "failed"
    assert not inference.contexts and simulation.closed_run_id == runner.run_id


def test_observer_close_failure_turns_success_into_failed_run(tmp_path):
    observer = RecordingObserver()
    runner, inference, simulation = observed_runner(tmp_path, observer)
    failure = RuntimeError("fixture observer flush failure")
    original = observer.close

    def fail(status):
        assert not inference.contexts
        assert simulation.closed_run_id == runner.run_id
        original(status)
        raise failure

    observer.close = fail
    with pytest.raises(RuntimeError) as caught:
        runner.run()
    assert caught.value is failure
    assert observer.events.count("close") == 1
    assert observer.statuses[0]["state"] == "completed"
    assert runner.health()["state"] == "failed"
    saved = json.loads((tmp_path / "status.json").read_text())
    assert saved["state"] == "failed" and saved["error_type"] == "RuntimeError"


def test_observer_close_failure_preserves_the_original_rollout_exception(tmp_path):
    observer = RecordingObserver()
    runner, inference, simulation = observed_runner(tmp_path, observer)
    simulation.lose_step_response = True
    original = observer.close

    def fail(status):
        original(status)
        raise RuntimeError("secondary observer close failure")

    observer.close = fail
    with pytest.raises(TimeoutError, match="lost response after physics advanced"):
        runner.run()
    assert observer.events.count("close") == 1
    assert observer.statuses[0]["state"] == "failed"
    assert runner.health()["error_type"] == "TimeoutError"
    assert not inference.contexts and simulation.closed_run_id == runner.run_id
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert any(event["event"] == "observer_close_failed" for event in events)
