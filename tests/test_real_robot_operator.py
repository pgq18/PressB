"""Local operator commands remain usable while the real node waits for reset."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from pressb.runtime.real_robot_operator import ManualTask, create_task, read_status, send_command


@pytest.fixture
def task(tmp_path):
    instance = create_task(dict(directory=tmp_path / "operator", reset_timeout_s=2., poll_interval_s=.005))
    yield instance
    instance.close()


def wait_phase(task, phase):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        status = task.status()
        if status["phase"] == phase:
            return status
        time.sleep(.002)
    raise AssertionError(f"Phase never became {phase}: {task.status()}")


def ready(task):
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(task.reset, 7, {"instruction": "Place a new object"})
        status = wait_phase(task, "waiting_reset")
        send_command(task.directory, "ready", status["episode_token"])
        assert future.result(timeout=2) is None
    return status["episode_token"]


def cli(task, *arguments):
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    return subprocess.run([sys.executable, "-m", "pressb.runtime.real_robot_operator",
                           "--directory", str(task.directory), *arguments],
                          env=env, capture_output=True, text=True, timeout=5)


def test_cli_confirms_ready_while_reset_is_blocked(task):
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(task.reset, 19, {"object": "fresh"})
        status = wait_phase(task, "waiting_reset")
        assert not future.done()
        assert task.poll() == dict(reward=0., terminated=False, truncated=False, success=False, reason="")
        external = cli(task, "status")
        assert external.returncode == 0, external.stderr
        displayed = json.loads(external.stdout)
        assert displayed["episode_token"] == status["episode_token"]
        assert displayed["options"] == {"object": "fresh"}
        assert displayed["seed"] == 19
        response = cli(task, "ready", "--episode", displayed["episode_token"])
        assert response.returncode == 0, response.stderr
        assert json.loads(response.stdout)["accepted"] is True
        assert future.result(timeout=2) is None
    assert task.status()["phase"] == "running"
    assert not list(task.directory.glob("*.command.json"))


@pytest.mark.parametrize("command,reward,success", [("success", 1., True), ("failure", 0., False)])
def test_terminal_labels_are_explicit_and_stable(task, command, reward, success):
    token = ready(task)
    assert task.poll()["terminated"] is False
    send_command(task.directory, command, token)
    assert (task.directory / f"{token}.command.json").is_file()
    expected = dict(reward=reward, terminated=True, truncated=False, success=success,
                    reason="operator_" + command)
    assert task.poll() == expected
    assert task.poll() == expected
    assert read_status(task.directory)["outcome"] == expected
    assert task.status()["phase"] == "terminal"
    with pytest.raises(ValueError, match="phase terminal"):
        send_command(task.directory, "success", token)


def test_custom_rewards_and_description(tmp_path):
    task = ManualTask(dict(directory=tmp_path, success_reward=2.5, failure_reward=-.25,
                           poll_interval_s=.005))
    try:
        assert task.describe()["success_reward"] == 2.5
        assert task.describe()["failure_reward"] == -.25
        token = ready(task)
        send_command(tmp_path, "failure", token)
        assert task.poll()["reward"] == -.25
        token = ready(task)
        send_command(tmp_path, "success", token)
        assert task.poll()["reward"] == 2.5
    finally:
        task.close()


def test_stale_token_wrong_phase_and_duplicate_commands_are_rejected(task):
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(task.reset, 1, {})
        token = wait_phase(task, "waiting_reset")["episode_token"]
        with pytest.raises(ValueError, match="Stale"):
            send_command(task.directory, "ready", "0" * 32)
        with pytest.raises(ValueError, match="phase waiting_reset"):
            send_command(task.directory, "success", token)
        send_command(task.directory, "ready", token)
        future.result(timeout=2)
    with pytest.raises(ValueError, match="phase running"):
        send_command(task.directory, "ready", token)
    send_command(task.directory, "success", token)
    with pytest.raises(ValueError, match="already pending"):
        send_command(task.directory, "failure", token)
    task.poll()
    next_token = ready(task)
    assert next_token != token
    with pytest.raises(ValueError, match="Stale"):
        send_command(task.directory, "success", token)
    with pytest.raises(RuntimeError, match="End the active"):
        task.reset(1, {})


def test_abort_preempts_pending_outcome_and_is_not_an_rl_failure(task):
    token = ready(task)
    send_command(task.directory, "success", token)
    send_command(task.directory, "abort", token)
    with pytest.raises(RuntimeError, match="abort"):
        task.poll()
    status = task.status()
    assert status["phase"] == "aborted"
    assert status["outcome"] == dict(reward=0., terminated=False, truncated=False, success=False, reason="")
    with pytest.raises(RuntimeError, match="aborted"):
        task.poll()


def test_abort_interrupts_blocking_reset(task):
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(task.reset, 3, {})
        token = wait_phase(task, "waiting_reset")["episode_token"]
        send_command(task.directory, "abort", token)
        with pytest.raises(RuntimeError, match="abort"):
            future.result(timeout=2)
    assert task.status()["phase"] == "aborted"


def test_end_and_close_interrupt_blocking_reset(task):
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(task.reset, 3, {})
        token = wait_phase(task, "waiting_reset")["episode_token"]
        task.end("runtime_timeout")
        with pytest.raises(RuntimeError, match="interrupted"):
            future.result(timeout=2)
        assert task.status()["reason"] == "runtime_timeout"
        with pytest.raises(ValueError, match="phase ended"):
            send_command(task.directory, "ready", token)
        future = pool.submit(task.reset, 4, {})
        wait_phase(task, "waiting_reset")
        task.close()
        with pytest.raises(RuntimeError, match="closed"):
            future.result(timeout=2)
    assert read_status(task.directory)["phase"] == "closed"
    task.close()


def test_reset_timeout_never_becomes_automatic_readiness(tmp_path):
    task = ManualTask(dict(directory=tmp_path, reset_timeout_s=.025, poll_interval_s=.005))
    try:
        with pytest.raises(TimeoutError, match="operator ready"):
            task.reset(0, {})
        assert task.status()["phase"] == "ended"
        assert task.status()["reason"] == "reset_timeout"
        assert not task.poll()["terminated"]
    finally:
        task.close()


def test_directory_owner_lock_and_restart_identity(task):
    token = ready(task)
    with pytest.raises(RuntimeError, match="already owns"):
        ManualTask(dict(directory=task.directory))
    owner_id = task.status()["owner_id"]
    task.close()
    with pytest.raises(RuntimeError, match="No real-robot task"):
        send_command(task.directory, "success", token)
    replacement = ManualTask(dict(directory=task.directory, poll_interval_s=.005))
    try:
        assert replacement.status()["owner_id"] != owner_id
        next_token = ready(replacement)
        assert next_token != token
        with pytest.raises(ValueError, match="Stale"):
            send_command(replacement.directory, "success", token)
    finally:
        replacement.close()


def test_malformed_current_command_fails_closed(task):
    token = ready(task)
    path = task.directory / f"{token}.command.json"
    path.write_text('{"command":"success","episode_token":"wrong"}')
    with pytest.raises(RuntimeError, match="Invalid operator mailbox"):
        task.poll()
    assert task.status()["phase"] == "aborted"
    assert task.status()["outcome"]["terminated"] is False


def test_cli_mutations_require_episode_and_reject_wrong_phase(task):
    response = cli(task, "ready")
    assert response.returncode == 2
    assert "--episode" in response.stderr
    response = cli(task, "ready", "--episode", "../status.json")
    assert response.returncode == 2
    assert "Stale or invalid" in response.stderr


@pytest.mark.parametrize("config", [
    {}, {"directory": ""}, {"directory": ".", "unknown": 1},
    {"directory": ".", "reset_timeout_s": 0}, {"directory": ".", "poll_interval_s": -1},
    {"directory": ".", "poll_interval_s": True}, {"directory": ".", "success_reward": float("nan")},
    {"directory": ".", "failure_reward": "0"},
])
def test_invalid_configuration_is_rejected_before_creating_mailbox(config):
    with pytest.raises(ValueError):
        ManualTask(config)


def test_reset_only_mode_disables_backend_labels_and_keeps_abort(tmp_path):
    task = ManualTask(dict(directory=tmp_path, labels_enabled=False, poll_interval_s=.005))
    try:
        token = ready(task)
        assert task.describe()["outcomes"] == "external_trainer_labels"
        assert task.status()["labels_enabled"] is False
        for command in ("success", "failure"):
            with pytest.raises(ValueError, match="labels are disabled"):
                send_command(task.directory, command, token)
        assert task.poll() == dict(reward=0., terminated=False, truncated=False, success=False, reason="")
        send_command(task.directory, "abort", token)
        with pytest.raises(RuntimeError, match="abort"):
            task.poll()
    finally:
        task.close()


def test_reset_only_mode_rejects_forged_backend_label(tmp_path):
    task = ManualTask(dict(directory=tmp_path, labels_enabled=False, poll_interval_s=.005))
    try:
        token = ready(task)
        (tmp_path / f"{token}.command.json").write_text(json.dumps(dict(
            schema_version=1, owner_id=task.status()["owner_id"], episode_token=token, command="success")))
        with pytest.raises(RuntimeError, match="Invalid operator mailbox"):
            task.poll()
    finally:
        task.close()


@pytest.mark.parametrize("command", ["success", "failure"])
def test_watchdog_abort_check_preserves_outcome_until_step_poll(task, command):
    token = ready(task)
    send_command(task.directory, command, token)
    task.check_abort()
    assert task.status()["phase"] == "running"
    assert (task.directory / f"{token}.command.json").is_file()
    assert task.poll()["reason"] == "operator_" + command


def test_watchdog_abort_check_raises_without_waiting_for_step(task):
    token = ready(task)
    send_command(task.directory, "abort", token)
    with pytest.raises(RuntimeError, match="abort"):
        task.check_abort()
    assert not (task.directory / f"{token}.command.json").exists()
    assert task.status()["phase"] == "aborted"
    with pytest.raises(RuntimeError, match="aborted"):
        task.check_abort()


def test_watchdog_abort_check_does_not_confirm_readiness(task, monkeypatch):
    import threading

    sleeping, proceed = threading.Event(), threading.Event()

    def pause_reset(timeout):
        sleeping.set()
        assert proceed.wait(2), "test did not release reset wait"

    monkeypatch.setattr(task._wake, "wait", pause_reset)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(task.reset, 7, {})
        try:
            assert sleeping.wait(2)
            token = task.status()["episode_token"]
            send_command(task.directory, "ready", token)
            task.check_abort()
            assert task.status()["phase"] == "waiting_reset"
            assert (task.directory / f"{token}.command.json").is_file()
            assert not future.done()
        finally:
            proceed.set()
        assert future.result(timeout=2) is None
