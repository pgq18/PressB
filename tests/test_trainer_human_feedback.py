"""Human outcome integration at the shared CPU trainer/replay boundary."""

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pressb.runtime.trainer import GenericTrainer, TrainerConfig
from test_rl_nodes_trainer import Inference, Simulation, config, metadata


class FeedbackCancelled(RuntimeError):
    pass


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class HumanCapableSimulation(Simulation):
    def __init__(self, meta, events, *, ongoing=False):
        super().__init__(meta)
        self.events = events
        self.ongoing = ongoing
        self.ends = []
        self.fail_end = False
        self.bad_end_ack = False

    def call(self, path, payload=None):
        self.events.append(path)
        if path == "/end_episode":
            self.ends.append(deepcopy(payload))
            if self.fail_end:
                raise TimeoutError("lost end_episode response")
            for row in payload["items"]:
                current = self.slots[row["env_id"]]
                assert not current["ended"]
                assert row["episode_id"] == current["episode_id"]
                assert row["step_id"] == current["step_id"]
                current["ended"] = True
            return {"ended": [] if self.bad_end_ack else deepcopy(payload["items"])}
        response = super().call(path, payload)
        if path == "/step" and self.ongoing:
            for row in response["items"]:
                self.slots[row["env_id"]]["ended"] = False
                row.update(terminated=False, truncated=False, reward=.5)
                row["info"]["success"] = False
        return response


class FakeFeedback:
    def __init__(self, settings, events, *, early=False, fail=False):
        assert "mode" not in settings
        self.settings = settings
        self.events = events
        self.early = early
        self.fail = fail
        self.pending_end = []
        self.begins = []
        self.resolved = []
        self.closed = 0

    def begin(self, rows):
        self.events.append("feedback_begin")
        self.begins.append(deepcopy(rows))

    def resolve(self, rows, *, cancel_event=None):
        self.events.append("feedback_resolve")
        if self.fail:
            raise RuntimeError("operator timeout")
        self.pending_end = []
        result = deepcopy(rows)
        for row in result:
            environmental_terminal = row["terminated"] or row["truncated"]
            row["info"]["environment_outcome"] = {
                "reward": row["reward"], "success": row["info"]["success"]}
            if environmental_terminal or self.early:
                success = self.early  # Early manual success, otherwise override automatic success with failure.
                row.update(reward=self.settings["success_reward" if success else "failure_reward"],
                           terminated=True, truncated=False)
                row["info"].update(success=success, reward_source="human")
                if not environmental_terminal:
                    self.pending_end.append({key: row[key] for key in ("env_id", "episode_id", "step_id")})
            else:
                row["reward"] = 0.
                row["info"]["success"] = False
        self.resolved.append(deepcopy(result))
        return result

    def close(self):
        self.events.append("feedback_close")
        self.closed += 1


class Observer:
    def __init__(self, events):
        self.events = events
        self.rows = []

    def on_reset(self, rows):
        self.events.append("observer_reset")

    def on_transition(self, before, actions, rows):
        self.events.append("observer_transition")
        self.rows.extend(deepcopy(rows))

    def close(self, status):
        self.events.append("observer_close")


def runner_with_feedback(tmp_path, monkeypatch, *, method="dsrl", early=False, fail=False,
                         feedback=None, steps=4, resume=None, capabilities=True):
    events, created = [], []

    def factory(settings):
        value = FakeFeedback(settings, events, early=early, fail=fail)
        created.append(value)
        return value

    monkeypatch.setitem(sys.modules, "pressb.runtime.human_feedback", SimpleNamespace(
        HumanFeedback=factory, HumanFeedbackCancelled=FeedbackCancelled))
    inf, sim = metadata(horizon=4, action_dim=3, noise_dim=5, state_dim=3)
    if capabilities:
        sim["capabilities"] = ["external_episode_end"]
    inference = Inference(inf)
    simulation = HumanCapableSimulation(sim, events, ongoing=early)
    cfg = config(tmp_path, method=method, steps=steps)
    cfg["update"].update(utd=0., learning_starts=100)
    cfg["human_feedback"] = feedback if feedback is not None else {"mode": "human", "success_reward": .75}
    if resume:
        cfg["checkpoint"]["resume"] = str(resume)
    if method == "base":
        cfg["mode"] = "eval"
    if method == "action_residual":
        cfg["learner"].update(residual_indices=[0, 2], residual_scale=[.02, .03])
    observer = Observer(events)
    trainer = GenericTrainer(cfg, inference_client=inference, simulation_client=simulation, observer=observer)
    return trainer, simulation, observer, created, events


@pytest.mark.parametrize("method", ["base", "dsrl", "action_residual"])
def test_early_manual_terminal_is_closed_before_observer_replay_and_reset(tmp_path, monkeypatch, method):
    trainer, simulation, observer, created, events = runner_with_feedback(
        tmp_path, monkeypatch, method=method, early=True)
    result = trainer.run()
    feedback = created[0]
    assert result["state"] == "completed"
    assert result["episodes"] == result["successes"] == 4
    assert len(simulation.ends) == len(simulation.steps) == 2
    assert len(simulation.resets) == len(feedback.begins) == 2
    assert feedback.closed == 1
    assert all(row["reward"] == .75 and row["terminated"] and row["info"]["success"] for row in observer.rows)
    for payload in simulation.ends:
        assert payload["run_id"] == trainer.run_id and payload["request_id"]
        assert all(set(row) == {"env_id", "episode_id", "step_id"} for row in payload["items"])
    for index, event in enumerate(events):
        if event == "feedback_resolve":
            assert events[index + 1:index + 3] == ["/end_episode", "observer_transition"]
        if event == "/reset":
            assert events[index + 1:index + 3] == ["feedback_begin", "observer_reset"]
    assert events[-2:] == ["feedback_close", "observer_close"]
    if method != "base":
        replay = trainer.replay.state_dict()["arrays"]
        np.testing.assert_allclose(replay["reward"], .75)
        assert replay["terminated"].all() and not replay["truncated"].any()
        assert trainer.learner.updates == 0
        extra = torch.load(trainer.last_checkpoint, weights_only=False)["extra"]
        assert extra["human_feedback"]["mode"] == "human"
        assert extra["human_feedback"]["success_reward"] == .75


def test_manual_failure_overrides_environment_success_and_unlabeled_reward(tmp_path, monkeypatch):
    trainer, simulation, observer, created, _ = runner_with_feedback(tmp_path, monkeypatch, steps=2)
    result = trainer.run()
    assert result["episodes"] == 1 and result["successes"] == 0
    assert not simulation.ends
    assert len(created[0].begins) == 1 and created[0].closed == 1
    assert observer.rows[0]["info"]["environment_outcome"]["success"] is True
    assert observer.rows[0]["reward"] == 0. and not observer.rows[0]["info"]["success"]
    assert observer.rows[1]["reward"] == 0. and not observer.rows[1]["terminated"]
    np.testing.assert_array_equal(trainer.replay.state_dict()["arrays"]["reward"], 0.)


def test_environment_mode_never_constructs_feedback(tmp_path, monkeypatch):
    trainer, simulation, observer, created, _ = runner_with_feedback(
        tmp_path, monkeypatch, feedback={"mode": "environment"}, capabilities=False, steps=2)
    result = trainer.run()
    assert created == [] and trainer.human_feedback is None
    assert result["successes"] == 1 and not simulation.ends
    assert observer.rows[0]["reward"] == 1.


@pytest.mark.parametrize("failure", ["resolve", "end_episode"])
def test_feedback_closes_after_label_or_end_episode_failure(tmp_path, monkeypatch, failure):
    trainer, simulation, observer, created, events = runner_with_feedback(
        tmp_path, monkeypatch, fail=failure == "resolve", early=failure == "end_episode")
    simulation.fail_end = failure == "end_episode"
    with pytest.raises((RuntimeError, TimeoutError), match="operator timeout|lost end_episode"):
        trainer.run()
    assert created[0].closed == 1
    assert trainer.health()["state"] == "failed"
    assert not observer.rows and len(trainer.replay) == 0
    assert len(simulation.resets) == 1
    assert simulation.closed_run_id == trainer.run_id
    assert events[-2:] == ["feedback_close", "observer_close"]
    assert ("unknown" in trainer.health()["execution_state"]) == (failure == "end_episode")


def test_required_capability_is_checked_before_reset_or_feedback_construction(tmp_path, monkeypatch):
    trainer, simulation, _, created, _ = runner_with_feedback(tmp_path, monkeypatch, capabilities=False)
    with pytest.raises(ValueError, match="external_episode_end"):
        trainer.run()
    assert not simulation.resets and not created


def test_end_episode_identity_confirmation_is_required_before_replay(tmp_path, monkeypatch):
    trainer, simulation, observer, created, _ = runner_with_feedback(tmp_path, monkeypatch, early=True)
    simulation.bad_end_ack = True
    with pytest.raises(ValueError, match="end_episode confirmation"):
        trainer.run()
    assert not observer.rows and len(trainer.replay) == 0
    assert len(simulation.resets) == 1 and created[0].closed == 1


@pytest.mark.parametrize("rollout", [
    {"terminal_observation_mode": "state_only_if_no_bootstrap"},
    {"terminal_observation_mode": "state_only_if_no_bootstrap", "fused_step_reset": True},
    {"fused_step_reset": True},
])
def test_human_mode_rejects_prelabel_reset_or_missing_terminal_images(rollout):
    with pytest.raises(ValueError, match="full terminal|fused_step_reset"):
        TrainerConfig(human_feedback={"mode": "human"}, rollout=rollout)


@pytest.mark.parametrize("feedback", [None, {"mode": "invalid"}, {"unknown": 1},
                                        {"timeout_s": 0}, {"poll_interval_s": float("nan")},
                                        {"success_reward": True}, {"directory": ""}])
def test_invalid_feedback_settings_fail_early(feedback):
    with pytest.raises(ValueError, match="human_feedback"):
        TrainerConfig(human_feedback=feedback)


@pytest.mark.parametrize("feedback", [
    {"mode": "environment"},
    {"mode": "human", "success_reward": 1.},
    {"mode": "human", "success_reward": .75, "failure_reward": -.5},
])
def test_resume_rejects_reward_semantic_changes_before_reset(tmp_path, monkeypatch, feedback):
    original, _, _, _, _ = runner_with_feedback(tmp_path / "first", monkeypatch, early=True, steps=2)
    original.run()
    resumed, simulation, _, created, _ = runner_with_feedback(
        tmp_path / "resume", monkeypatch, early=True, resume=original.last_checkpoint, feedback=feedback)
    with pytest.raises(ValueError, match="human_feedback mode/reward scheme"):
        resumed.run()
    assert not simulation.resets and not created


def test_resume_allows_new_operator_directory_and_wait_settings(tmp_path, monkeypatch):
    original, _, _, _, _ = runner_with_feedback(tmp_path / "first", monkeypatch, early=True, steps=2)
    original.run()
    settings = {"mode": "human", "success_reward": .75, "directory": str(tmp_path / "new_operator"),
                "timeout_s": 1., "poll_interval_s": .01}
    resumed, simulation, _, created, _ = runner_with_feedback(
        tmp_path / "resume", monkeypatch, early=True, resume=original.last_checkpoint, feedback=settings)
    result = resumed.run()
    assert result["transitions"] == 4 and result["successes"] == 4
    assert len(simulation.resets) == 1
    assert created[0].settings["directory"] == settings["directory"]


def test_legacy_checkpoint_without_feedback_key_remains_environment_mode(tmp_path, monkeypatch):
    original, _, _, _, _ = runner_with_feedback(
        tmp_path / "first", monkeypatch, feedback={"mode": "environment"}, steps=2)
    original.run()
    payload = torch.load(original.last_checkpoint, weights_only=False)
    payload["extra"].pop("human_feedback")
    path = tmp_path / "legacy.pt"
    torch.save(payload, path)
    resumed, _, _, created, _ = runner_with_feedback(
        tmp_path / "resume", monkeypatch, feedback={"mode": "environment"}, resume=path)
    assert resumed.run()["transitions"] == 4
    assert not created


def test_stop_interrupts_actual_feedback_wait_without_replaying_unlabeled_batch(tmp_path, monkeypatch):
    # Keep the actual mailbox implementation; use fake CPU node clients only.
    from pressb.runtime import human_feedback as real_feedback

    trainer, simulation, observer, _, _ = runner_with_feedback(
        tmp_path, monkeypatch, steps=4,
        feedback={"mode": "human", "timeout_s": 300., "poll_interval_s": 30.})
    monkeypatch.setitem(sys.modules, "pressb.runtime.human_feedback", real_feedback)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(trainer.run)
        deadline = time.monotonic() + 5.
        try:
            while time.monotonic() < deadline:
                current = trainer.human_feedback
                if current is not None and current.status()["phase"] == "waiting_for_labels":
                    break
                if future.done():
                    future.result()
                    pytest.fail("Trainer finished before reaching the human label wait")
                time.sleep(.005)
            else:
                pytest.fail("Trainer did not enter human label wait")
            started = time.monotonic()
            trainer.stop()
            result = future.result(timeout=2.)
            assert time.monotonic() - started < 1.
        finally:
            # A regression must not leave a 300-second worker behind.
            trainer.stop()
            if trainer.human_feedback is not None:
                trainer.human_feedback.close()
            future.result(timeout=2.)
    assert result["state"] == "stopped" and "error" not in result
    assert result["transitions"] == result["episodes"] == result["successes"] == 0
    assert len(trainer.replay) == 0 and not observer.rows
    assert len(simulation.steps) == len(simulation.resets) == 1 and not simulation.ends
    assert simulation.closed_run_id == trainer.run_id
    assert trainer.human_feedback.status()["phase"] == "closed"
    extra = torch.load(trainer.last_checkpoint, weights_only=False)["extra"]
    assert extra["transitions"] == extra["replay"]["size"] == 0


def test_stop_does_not_hide_feedback_errors_or_create_failure_labels(tmp_path, monkeypatch):
    trainer, simulation, observer, created, _ = runner_with_feedback(tmp_path, monkeypatch)
    original_call = simulation.call

    def stop_then_fail(path, payload=None):
        result = original_call(path, payload)
        if path == "/step":
            trainer.stop()
            raise TimeoutError("lost hardware response after stop")
        return result

    simulation.call = stop_then_fail
    with pytest.raises(TimeoutError, match="lost hardware response"):
        trainer.run()
    assert trainer.health()["state"] == "failed"
    assert len(trainer.replay) == 0 and not observer.rows
    assert created[0].closed == 1


def test_unsolicited_feedback_cancellation_is_still_a_failure(tmp_path, monkeypatch):
    trainer, _, observer, created, _ = runner_with_feedback(tmp_path, monkeypatch)
    def cancel(self, rows, *, cancel_event=None):
        raise FeedbackCancelled("not requested by trainer")

    monkeypatch.setattr(FakeFeedback, "resolve", cancel)
    with pytest.raises(FeedbackCancelled, match="not requested"):
        trainer.run()
    assert trainer.health()["state"] == "failed" and not trainer.stop_event.is_set()
    assert len(trainer.replay) == 0 and not observer.rows
    assert created[0].closed == 1
