"""Local operator reset and outcome hooks for a real-robot episode.

The node owns a directory lock for its lifetime. A separate short-lived mailbox
lock lets the CLI publish commands while the node blocks inside ``reset``.
No command confirms readiness automatically, and aborts raise for the runtime's
emergency-stop path rather than becoming an ordinary unsuccessful episode.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import json
import math
from numbers import Real
import os
from pathlib import Path
import tempfile
import threading
import time
import uuid


_COMMANDS = {"ready": {"waiting_reset"}, "success": {"running"},
             "failure": {"running"}, "abort": {"waiting_reset", "running"}}


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _atomic_json(path, value):
    data = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix="." + path.name + ".",
                                         suffix=".tmp", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def _mailbox(directory):
    # Never unlink lock files: replacing their inode would permit two owners.
    with (directory / ".mailbox.lock").open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _read_status(directory):
    value = json.loads((directory / "status.json").read_text())
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("Invalid operator status")
    return value


def read_status(directory):
    """Read the last atomic status without contacting the blocked HTTP server."""
    directory = Path(directory).expanduser().resolve(strict=True)
    with _mailbox(directory):
        return _read_status(directory)


def _require_live_owner(directory):
    with (directory / ".owner.lock").open("r+b") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    raise RuntimeError("No real-robot task owns this operator directory")


def send_command(directory, command, episode_token):
    """Publish one phase-checked command; an abort supersedes a queued outcome."""
    if command not in _COMMANDS:
        raise ValueError(f"Unknown operator command: {command}")
    directory = Path(directory).expanduser().resolve(strict=True)
    with _mailbox(directory):
        _require_live_owner(directory)
        status = _read_status(directory)
        if (not isinstance(episode_token, str) or len(episode_token) != 32
                or any(char not in "0123456789abcdef" for char in episode_token)
                or episode_token != status.get("episode_token")):
            raise ValueError("Stale or invalid episode token; read status for the active episode")
        if status.get("phase") not in _COMMANDS[command]:
            raise ValueError(f"Cannot send {command} during phase {status.get('phase')}")
        if command in ("success", "failure") and not status.get("labels_enabled", True):
            raise ValueError("Task outcome labels are disabled; use the trainer feedback channel")
        path = directory / f"{episode_token}.command.json"
        if path.exists() and command != "abort":
            raise ValueError("An operator command is already pending for this episode")
        request = dict(schema_version=1, owner_id=status["owner_id"], episode_token=episode_token,
                       command=command, created_at_unix_s=time.time())
        _atomic_json(path, request)
        return dict(accepted=True, command=command, episode_token=episode_token)


def _empty_result():
    return dict(reward=0., terminated=False, truncated=False, success=False, reason="")


class ManualTask:
    """Wait for a local human to reset the scene and label episode outcomes."""

    def __init__(self, config):
        allowed = {"directory", "reset_timeout_s", "poll_interval_s", "success_reward", "failure_reward",
                   "labels_enabled"}
        if not isinstance(config, dict) or set(config) - allowed or "directory" not in config:
            raise ValueError(f"Manual task requires directory and supports {sorted(allowed)}")
        directory = config["directory"]
        if not isinstance(directory, (str, Path)) or not str(directory).strip():
            raise ValueError("Manual task directory must be a nonempty path")
        self.directory = Path(directory).expanduser().resolve()
        self.reset_timeout_s = _number(config.get("reset_timeout_s", 300.), "reset_timeout_s", positive=True)
        self.poll_interval_s = _number(config.get("poll_interval_s", .05), "poll_interval_s", positive=True)
        self.success_reward = _number(config.get("success_reward", 1.), "success_reward")
        self.failure_reward = _number(config.get("failure_reward", 0.), "failure_reward")
        self.labels_enabled = config.get("labels_enabled", True)
        if type(self.labels_enabled) is not bool:
            raise ValueError("labels_enabled must be boolean")
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._closed = False
        self._result = _empty_result()
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._owner = (self.directory / ".owner.lock").open("a+b")
        try:
            fcntl.flock(self._owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self._owner.close()
            raise RuntimeError("Another real-robot task already owns this operator directory") from error
        try:
            with _mailbox(self.directory):
                self._status = dict(schema_version=1, owner_id=uuid.uuid4().hex, pid=os.getpid(),
                                    episode_token=None, phase="idle", seed=None, options={}, reason="",
                                    labels_enabled=self.labels_enabled)
                self._publish()
        except BaseException:
            self._owner.close()
            raise

    def _publish(self):
        self._status["updated_at_unix_s"] = time.time()
        self._status["outcome"] = deepcopy(self._result)
        _atomic_json(self.directory / "status.json", self._status)

    def _command_path(self):
        token = self._status["episode_token"]
        return None if token is None else self.directory / f"{token}.command.json"

    def _consume(self):
        path = self._command_path()
        if path is None or not path.exists():
            return
        try:
            request = json.loads(path.read_text())
            command = request.get("command") if isinstance(request, dict) else None
            if (not isinstance(request, dict) or request.get("schema_version") != 1
                    or request.get("owner_id") != self._status["owner_id"]
                    or request.get("episode_token") != self._status["episode_token"]
                    or command not in _COMMANDS or self._status["phase"] not in _COMMANDS[command]
                    or (command in ("success", "failure") and not self.labels_enabled)):
                raise ValueError("Invalid, stale or wrong-phase operator command")
        except (ValueError, TypeError, OSError) as error:
            self._status.update(phase="aborted", reason="invalid_operator_command")
            self._publish()
            raise RuntimeError("Invalid operator mailbox command; stop the robot") from error
        path.unlink()
        if command == "abort":
            self._status.update(phase="aborted", reason="operator_abort")
            self._publish()
            raise RuntimeError("Operator requested abort; stop the robot")
        if command == "ready":
            self._status.update(phase="running", reason="operator_ready")
        else:
            success = command == "success"
            self._result = dict(reward=self.success_reward if success else self.failure_reward,
                                terminated=True, truncated=False, success=success,
                                reason="operator_" + command)
            self._status.update(phase="terminal", reason=self._result["reason"])
        self._publish()

    def reset(self, seed, options):
        """Create a fresh token and wait until this episode receives ``ready``."""
        if type(seed) is not int or seed < 0:
            raise ValueError("Reset seed must be a nonnegative integer")
        if not isinstance(options, dict):
            raise ValueError("Reset options must be a mapping")
        options = json.loads(json.dumps(options, allow_nan=False))
        with self._lock, _mailbox(self.directory):
            if self._closed:
                raise RuntimeError("Manual task is closed")
            if self._status["phase"] in ("waiting_reset", "running"):
                raise RuntimeError("End the active episode before resetting it")
            self._wake.clear()
            self._result = _empty_result()
            token = uuid.uuid4().hex
            self._status.update(episode_token=token, phase="waiting_reset", seed=seed,
                                options=options, reason="awaiting_operator_ready")
            self._publish()
        deadline = time.monotonic() + self.reset_timeout_s
        while True:
            with self._lock, _mailbox(self.directory):
                if self._closed or self._status["episode_token"] != token:
                    raise RuntimeError("Manual reset was closed or replaced")
                self._consume()
                phase = self._status["phase"]
                if phase == "running":
                    return None
                if phase != "waiting_reset":
                    raise RuntimeError(f"Manual reset interrupted: {self._status['reason']}")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._status.update(phase="ended", reason="reset_timeout")
                    self._publish()
                    raise TimeoutError("Timed out waiting for operator ready")
            self._wake.wait(min(self.poll_interval_s, remaining))

    def poll(self):
        with self._lock, _mailbox(self.directory):
            if self._closed:
                raise RuntimeError("Manual task is closed")
            if self._status["phase"] == "aborted":
                raise RuntimeError(f"Operator task aborted: {self._status['reason']}")
            self._consume()
            return deepcopy(self._result)

    def check_abort(self):
        """Let a watchdog observe abort without consuming readiness or outcomes."""
        with self._lock, _mailbox(self.directory):
            if self._closed:
                return
            if self._status["phase"] == "aborted":
                raise RuntimeError(f"Operator task aborted: {self._status['reason']}")
            path = self._command_path()
            if path is None or not path.exists():
                return
            try:
                request = json.loads(path.read_text())
            except (ValueError, OSError) as error:
                self._status.update(phase="aborted", reason="invalid_operator_command")
                self._publish()
                raise RuntimeError("Invalid operator mailbox command; stop the robot") from error
            if isinstance(request, dict) and request.get("command") == "abort":
                self._consume()

    def describe(self):
        return dict(kind="manual_operator", directory=str(self.directory),
                    reset="explicit_episode_ready", labels_enabled=self.labels_enabled,
                    outcomes="operator_success_or_failure" if self.labels_enabled else "external_trainer_labels",
                    abort="raises_for_emergency_stop", reset_timeout_s=self.reset_timeout_s,
                    poll_interval_s=self.poll_interval_s, success_reward=self.success_reward,
                    failure_reward=self.failure_reward)

    def end(self, reason):
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Episode end reason must be a nonempty string")
        with self._lock, _mailbox(self.directory):
            if self._closed:
                return
            path = self._command_path()
            if path is not None:
                path.unlink(missing_ok=True)
            self._status.update(phase="ended", reason=reason)
            self._publish()
            self._wake.set()

    def status(self):
        with self._lock:
            return deepcopy(self._status)

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._wake.set()
            try:
                with _mailbox(self.directory):
                    path = self._command_path()
                    if path is not None:
                        path.unlink(missing_ok=True)
                    self._status.update(phase="closed", reason="closed")
                    self._publish()
            finally:
                self._owner.close()


def create_task(config):
    return ManualTask(config)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status", help="Show the active episode token and phase")
    for command in _COMMANDS:
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--episode", required=True, help="Exact episode_token from status")
    args = parser.parse_args(argv)
    try:
        result = (read_status(args.directory) if args.command == "status" else
                  send_command(args.directory, args.command, args.episode))
    except (ValueError, RuntimeError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
