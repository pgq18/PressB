"""Trainer-side human labels override task outcomes without inventing labels."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from pressb.runtime.human_feedback import (
    HumanFeedback, HumanFeedbackCancelled, read_status, send_command,
)


def reset_row(env_id=0, episode_id="episode-a"):
    return dict(env_id=env_id, episode_id=episode_id, step_id=0, observation={"state": [0.]}, info={})


def step_row(env_id=0, episode_id="episode-a", step_id=1, *, terminated=False, truncated=False,
             reward=8., success=True):
    return dict(env_id=env_id, episode_id=episode_id, step_id=step_id, terminated=terminated,
                truncated=truncated, reward=reward, discount=.96, observation={"state": [1.], "images": {"up": "png"}},
                info=dict(success=success, reason="environment_reason", physics_steps=4,
                          elapsed_control_steps=1., details={"measured": 12}))


@pytest.fixture
def feedback(tmp_path):
    instance = HumanFeedback(dict(directory=tmp_path / "feedback", timeout_s=2., poll_interval_s=.005))
    yield instance
    instance.close()


def wait_phase(feedback, phase):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        status = feedback.status()
        if status["phase"] == phase:
            return status
        time.sleep(.002)
    raise AssertionError(f"Phase never became {phase}: {feedback.status()}")


def cli(feedback, *arguments):
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    return subprocess.run([sys.executable, "-m", "pressb.runtime.human_feedback", "--directory",
                           str(feedback.directory), *arguments], env=env, capture_output=True,
                          text=True, timeout=5)


def test_nonterminal_environment_reward_and_success_are_discarded(feedback):
    feedback.begin([reset_row()])
    row = step_row()
    before = deepcopy(row)
    actual = feedback.resolve([row])[0]
    assert row == before
    assert actual["reward"] == 0.
    assert actual["info"]["success"] is False
    assert actual["info"]["outcome_source"] == "human"
    assert actual["info"]["human_label"] is None
    assert not actual["terminated"] and not actual["truncated"]
    assert actual["discount"] == .96
    assert actual["observation"] == row["observation"]
    assert actual["info"]["environment_outcome"] == dict(
        reward=8., terminated=False, truncated=False, success=True, info=before["info"])
    assert feedback.pending_end == []


@pytest.mark.parametrize("label,reward,success", [("success", 1., True), ("failure", 0., False)])
def test_early_label_ends_exact_episode_at_step_boundary(feedback, label, reward, success):
    feedback.begin([reset_row()])
    send_command(feedback.directory, label, "episode-a")
    actual = feedback.resolve([step_row()])[0]
    assert actual["reward"] == reward
    assert actual["terminated"] is True and actual["truncated"] is False
    assert actual["info"]["success"] is success
    assert actual["info"]["human_label"] == label
    assert feedback.pending_end == [{"env_id": 0, "episode_id": "episode-a", "step_id": 1}]
    with pytest.raises(ValueError, match="inactive episode"):
        send_command(feedback.directory, label, "episode-a")
    assert feedback.status()["episodes"][0]["phase"] == "retired"
    feedback.begin([reset_row(0, "episode-next")])
    feedback.resolve([step_row(0, "episode-next")])
    assert feedback.pending_end == []


@pytest.mark.parametrize("terminated,truncated", [(True, False), (False, True)])
def test_environment_done_waits_for_cli_label_and_can_be_overridden(feedback, terminated, truncated):
    feedback.begin([reset_row()])
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(feedback.resolve, [step_row(terminated=terminated, truncated=truncated)])
        status = wait_phase(feedback, "waiting_for_labels")
        assert not future.done()
        assert status["episodes"][0]["phase"] == "awaiting_label"
        shown = cli(feedback, "status")
        assert shown.returncode == 0, shown.stderr
        assert json.loads(shown.stdout)["episodes"][0]["episode_id"] == "episode-a"
        queued = cli(feedback, "failure", "--episode", "episode-a")
        assert queued.returncode == 0, queued.stderr
        actual = future.result(timeout=2)[0]
    assert actual["terminated"] is True and actual["truncated"] is False
    assert actual["info"]["success"] is False
    assert actual["reward"] == 0.
    original = actual["info"]["environment_outcome"]
    assert original["terminated"] is terminated and original["truncated"] is truncated
    assert original["success"] is True
    assert feedback.pending_end == []


def test_multiple_environments_wait_for_each_terminal_label(feedback):
    feedback.begin([reset_row(0, "a"), reset_row(1, "b"), reset_row(2, "c")])
    send_command(feedback.directory, "success", "c")
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(feedback.resolve, [step_row(1, "b", truncated=True), step_row(0, "a"), step_row(2, "c")])
        wait_phase(feedback, "waiting_for_labels")
        assert not future.done()
        send_command(feedback.directory, "failure", "b")
        actual = future.result(timeout=2)
    assert [row["env_id"] for row in actual] == [1, 0, 2]
    assert [row["terminated"] for row in actual] == [True, False, True]
    assert [row["info"]["success"] for row in actual] == [False, False, True]
    assert feedback.pending_end == [{"env_id": 2, "episode_id": "c", "step_id": 1}]
    feedback.begin([reset_row(1, "b2"), reset_row(2, "c2")])
    assert {row["episode_id"] for row in feedback.status()["episodes"]} == {"a", "b2", "c2"}
    feedback.resolve([step_row(0, "a", 2), step_row(1, "b2"), step_row(2, "c2")])
    assert feedback.pending_end == []


def test_label_for_unstepped_episode_stays_pending(feedback):
    feedback.begin([reset_row(0, "a"), reset_row(1, "b")])
    send_command(feedback.directory, "success", "b")
    first = feedback.resolve([step_row(0, "a")])
    assert not first[0]["terminated"]
    assert feedback.pending_end == []
    second = feedback.resolve([step_row(1, "b")])
    assert second[0]["terminated"] and second[0]["info"]["success"]
    assert feedback.pending_end == [{"env_id": 1, "episode_id": "b", "step_id": 1}]


def test_abort_overrides_a_queued_label_and_raises_infrastructure_error(feedback):
    feedback.begin([reset_row()])
    send_command(feedback.directory, "success", "episode-a")
    send_command(feedback.directory, "abort", "episode-a")
    with pytest.raises(RuntimeError, match="Human requested abort"):
        feedback.resolve([step_row()])
    assert feedback.status()["phase"] == "aborted"
    assert feedback.pending_end == []
    with pytest.raises(ValueError, match="session is aborted"):
        send_command(feedback.directory, "failure", "episode-a")
    with pytest.raises(RuntimeError, match="aborted"):
        feedback.resolve([step_row()])


def test_abort_on_another_active_episode_interrupts_wait(feedback):
    feedback.begin([reset_row(0, "a"), reset_row(1, "b")])
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(feedback.resolve, [step_row(0, "a", terminated=True)])
        wait_phase(feedback, "waiting_for_labels")
        send_command(feedback.directory, "abort", "b")
        with pytest.raises(RuntimeError, match="Human requested abort.*b"):
            future.result(timeout=2)


def test_timeout_aborts_without_fake_terminal_result(tmp_path):
    feedback = HumanFeedback(dict(directory=tmp_path, timeout_s=.025, poll_interval_s=.005))
    try:
        feedback.begin([reset_row()])
        with pytest.raises(RuntimeError, match="Timed out"):
            feedback.resolve([step_row(terminated=True)])
        assert feedback.status()["phase"] == "aborted"
        assert feedback.status()["episodes"][0]["human_label"] is None
        assert feedback.pending_end == []
    finally:
        feedback.close()


def test_close_interrupts_label_wait_and_releases_directory(feedback):
    feedback.begin([reset_row()])
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(feedback.resolve, [step_row(truncated=True)])
        wait_phase(feedback, "waiting_for_labels")
        feedback.close()
        with pytest.raises(RuntimeError, match="closed"):
            future.result(timeout=2)
    assert read_status(feedback.directory)["phase"] == "closed"
    feedback.close()
    replacement = HumanFeedback(dict(directory=feedback.directory))
    try:
        assert replacement.epoch_token != feedback.epoch_token
    finally:
        replacement.close()


def test_cancelled_before_resolve_leaves_episode_and_pending_label_unchanged(feedback):
    feedback.begin([reset_row()])
    send_command(feedback.directory, "success", "episode-a")
    command = next(feedback.directory.glob("*.command.json"))
    before = feedback.status()["episodes"]
    cancel_event = threading.Event()
    cancel_event.set()
    with pytest.raises(HumanFeedbackCancelled, match="cancelled"):
        feedback.resolve([step_row()], cancel_event=cancel_event)
    assert feedback.status()["episodes"] == before
    assert command.exists()
    assert feedback.pending_end == []


def test_stop_interrupts_long_poll_wait_without_fabricating_outcome(tmp_path):
    feedback = HumanFeedback(dict(directory=tmp_path, timeout_s=10., poll_interval_s=5.))
    cancel_event = threading.Event()
    try:
        feedback.begin([reset_row()])
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(feedback.resolve, [step_row(terminated=True)],
                                 cancel_event=cancel_event)
            wait_phase(feedback, "waiting_for_labels")
            cancel_event.set()
            with pytest.raises(HumanFeedbackCancelled):
                future.result(timeout=.5)
        episode = feedback.status()["episodes"][0]
        assert episode["phase"] == "awaiting_label"
        assert episode["step_id"] == 0 and episode["human_label"] is None
        assert feedback.status()["phase"] != "aborted"
        assert feedback.pending_end == []
        # Cancelling a resolve does not close the mailbox or retire its episodes.
        send_command(feedback.directory, "failure", "episode-a")
        actual = feedback.resolve([step_row(terminated=True)])[0]
        assert actual["terminated"] and actual["info"]["human_label"] == "failure"
    finally:
        feedback.close()


def test_cancellation_after_reading_label_preserves_command_until_retry(feedback, monkeypatch):
    feedback.begin([reset_row()])
    send_command(feedback.directory, "success", "episode-a")
    command = next(feedback.directory.glob("*.command.json"))
    cancel_event = threading.Event()
    original_requests = feedback._requests

    def cancel_after_read():
        requests = original_requests()
        assert requests == {"episode-a": "success"}
        cancel_event.set()
        return requests

    monkeypatch.setattr(feedback, "_requests", cancel_after_read)
    with pytest.raises(HumanFeedbackCancelled):
        feedback.resolve([step_row()], cancel_event=cancel_event)
    episode = feedback.status()["episodes"][0]
    assert episode["phase"] == "active" and episode["step_id"] == 0
    assert episode["human_label"] is None
    assert command.exists() and feedback.pending_end == []
    monkeypatch.setattr(feedback, "_requests", original_requests)
    cancel_event.clear()
    actual = feedback.resolve([step_row()], cancel_event=cancel_event)[0]
    assert actual["terminated"] and actual["info"]["success"]
    assert not command.exists()


def test_stale_ids_steps_and_duplicate_labels_are_rejected(feedback):
    feedback.begin([reset_row()])
    for stale in ("previous-episode", "", "../status.json"):
        with pytest.raises(ValueError, match="inactive episode"):
            send_command(feedback.directory, "success", stale)
    for wrong in (step_row(step_id=0), step_row(step_id=2), step_row(episode_id="wrong")):
        with pytest.raises(ValueError, match="out-of-order"):
            feedback.resolve([wrong])
    send_command(feedback.directory, "failure", "episode-a")
    with pytest.raises(ValueError, match="already pending"):
        send_command(feedback.directory, "success", "episode-a")
    feedback.resolve([step_row()])
    with pytest.raises(ValueError, match="reuse"):
        feedback.begin([reset_row()])
    with pytest.raises(ValueError, match="inactive episode"):
        send_command(feedback.directory, "failure", "episode-a")


def test_active_episode_cannot_be_replaced(feedback):
    feedback.begin([reset_row()])
    with pytest.raises(ValueError, match="replace an active"):
        feedback.begin([reset_row(0, "new-id")])
    assert feedback.status()["episodes"][0]["episode_id"] == "episode-a"


def test_restart_never_consumes_old_epoch_command(feedback):
    feedback.begin([reset_row()])
    send_command(feedback.directory, "success", "episode-a")
    old_paths = set(feedback.directory.glob("*.command.json"))
    assert len(old_paths) == 1
    feedback.close()
    replacement = HumanFeedback(dict(directory=feedback.directory))
    try:
        replacement.begin([reset_row()])
        assert replacement.resolve([step_row()])[0]["terminated"] is False
        assert set(feedback.directory.glob("*.command.json")) == old_paths
    finally:
        replacement.close()


def test_forged_current_command_fails_closed(feedback):
    feedback.begin([reset_row()])
    send_command(feedback.directory, "success", "episode-a")
    path = next(feedback.directory.glob("*.command.json"))
    request = json.loads(path.read_text())
    request["epoch_token"] = "stale"
    path.write_text(json.dumps(request))
    with pytest.raises(RuntimeError, match="Invalid human-feedback mailbox"):
        feedback.resolve([step_row()])
    assert feedback.status()["phase"] == "aborted"


def test_directory_lock_and_cli_required_episode(feedback):
    with pytest.raises(RuntimeError, match="already owns"):
        HumanFeedback(dict(directory=feedback.directory))
    response = cli(feedback, "success")
    assert response.returncode == 2 and "--episode" in response.stderr
    feedback.close()
    with pytest.raises(RuntimeError, match="No trainer owns"):
        send_command(feedback.directory, "abort", "episode-a")


def test_custom_rewards_only_depend_on_human_label(tmp_path):
    feedback = HumanFeedback(dict(directory=tmp_path, success_reward=3., failure_reward=-2.))
    try:
        feedback.begin([reset_row(0, "a"), reset_row(1, "b")])
        send_command(tmp_path, "success", "a")
        send_command(tmp_path, "failure", "b")
        rows = feedback.resolve([step_row(0, "a", success=False, reward=-99.),
                                 step_row(1, "b", success=True, reward=99.)])
        assert [row["reward"] for row in rows] == [3., -2.]
        assert [row["info"]["success"] for row in rows] == [True, False]
        assert feedback.describe()["environment_rewards"] == "discarded"
    finally:
        feedback.close()


@pytest.mark.parametrize("config", [
    {}, {"directory": ""}, {"directory": ".", "mode": "human"},
    {"directory": ".", "timeout_s": 0}, {"directory": ".", "poll_interval_s": -1},
    {"directory": ".", "success_reward": True}, {"directory": ".", "failure_reward": float("nan")},
])
def test_configuration_rejected_before_touching_directory(config):
    with pytest.raises(ValueError):
        HumanFeedback(config)


def test_corrupted_epoch_token_cannot_write_outside_mailbox(feedback):
    feedback.begin([reset_row()])
    path = feedback.directory / "status.json"
    status = json.loads(path.read_text())
    status["epoch_token"] = "../escape"
    path.write_text(json.dumps(status))
    with pytest.raises(ValueError, match="epoch token"):
        send_command(feedback.directory, "success", "episode-a")
    assert not list(feedback.directory.glob("*.command.json"))
