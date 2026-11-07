"""Trainer-side human outcome labels, independent of the environment backend.

Commands are queued through a local filesystem mailbox. Labels take effect at
an action-chunk boundary. Environment termination without a label waits for a
human; timeout and abort raise instead of inventing an unsuccessful episode.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import tempfile
import threading
import time
import uuid


_COMMANDS = ("success", "failure", "abort")
_ACTIVE_PHASES = ("active", "awaiting_label")


class HumanFeedbackCancelled(RuntimeError):
    """The trainer stopped before a human outcome was committed."""


def _check_cancelled(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise HumanFeedbackCancelled("Human feedback resolution cancelled")


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
    with (directory / ".mailbox.lock").open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _read_status(directory):
    value = json.loads((directory / "status.json").read_text())
    if (not isinstance(value, dict) or value.get("schema_version") != 1
            or value.get("kind") != "human_feedback" or not isinstance(value.get("episodes"), list)):
        raise ValueError("Invalid trainer human-feedback status")
    epoch = value.get("epoch_token")
    if (not isinstance(epoch, str) or len(epoch) != 32
            or any(char not in "0123456789abcdef" for char in epoch)):
        raise ValueError("Invalid human-feedback epoch token")
    return value


def read_status(directory):
    directory = Path(directory).expanduser().resolve(strict=True)
    with _mailbox(directory):
        return _read_status(directory)


def _command_path(directory, epoch_token, episode_id):
    # Episode IDs come from the environment and need not be safe path components.
    digest = hashlib.sha256(episode_id.encode("utf-8")).hexdigest()
    return directory / f"{epoch_token}.{digest}.command.json"


def _require_live_owner(directory):
    with (directory / ".owner.lock").open("r+b") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    raise RuntimeError("No trainer owns this human-feedback directory")


def send_command(directory, command, episode_id):
    """Queue a label for one active actual episode ID; abort overrides a label."""
    if command not in _COMMANDS:
        raise ValueError(f"Unknown human-feedback command: {command}")
    directory = Path(directory).expanduser().resolve(strict=True)
    with _mailbox(directory):
        _require_live_owner(directory)
        status = _read_status(directory)
        if status.get("phase") in ("closed", "aborted"):
            raise ValueError(f"Human-feedback session is {status['phase']}")
        matches = [row for row in status["episodes"] if row["episode_id"] == episode_id
                   and row["phase"] in _ACTIVE_PHASES]
        if not isinstance(episode_id, str) or not episode_id or len(matches) != 1:
            raise ValueError("Stale or inactive episode ID; read status for active episodes")
        row = matches[0]
        path = _command_path(directory, status["epoch_token"], episode_id)
        if path.exists() and command != "abort":
            raise ValueError("A human label is already pending for this episode")
        _atomic_json(path, dict(schema_version=1, epoch_token=status["epoch_token"],
                                env_id=row["env_id"], episode_id=episode_id, command=command,
                                created_at_unix_s=time.time()))
        return dict(accepted=True, command=command, env_id=row["env_id"], episode_id=episode_id,
                    effective_at="next_step_boundary")


class HumanFeedback:
    """Replace environment rewards and outcomes with explicit human labels."""

    def __init__(self, config):
        allowed = {"directory", "timeout_s", "poll_interval_s", "success_reward", "failure_reward"}
        if not isinstance(config, dict) or set(config) - allowed or "directory" not in config:
            raise ValueError(f"Human feedback requires directory and supports {sorted(allowed)}")
        directory = config["directory"]
        if not isinstance(directory, (str, Path)) or not str(directory).strip():
            raise ValueError("Human-feedback directory must be a nonempty path")
        self.directory = Path(directory).expanduser().resolve()
        self.timeout_s = _number(config.get("timeout_s", 300.), "timeout_s", positive=True)
        self.poll_interval_s = _number(config.get("poll_interval_s", .05), "poll_interval_s", positive=True)
        self.success_reward = _number(config.get("success_reward", 1.), "success_reward")
        self.failure_reward = _number(config.get("failure_reward", 0.), "failure_reward")
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._closed = self._resolving = False
        self._slots, self._seen = {}, set()
        self._pending_end = []
        self.epoch_token = uuid.uuid4().hex
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._owner = (self.directory / ".owner.lock").open("a+b")
        try:
            fcntl.flock(self._owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self._owner.close()
            raise RuntimeError("Another node already owns this human-feedback directory") from error
        try:
            with _mailbox(self.directory):
                self._phase, self._reason = "idle", ""
                self._publish()
        except BaseException:
            self._owner.close()
            raise

    def _snapshot(self):
        return dict(schema_version=1, kind="human_feedback", epoch_token=self.epoch_token,
                    pid=os.getpid(), phase=self._phase, reason=self._reason,
                    episodes=[deepcopy(self._slots[key]) for key in sorted(self._slots)],
                    updated_at_unix_s=time.time())

    def _publish(self):
        _atomic_json(self.directory / "status.json", self._snapshot())

    def _require_open(self):
        if self._closed:
            raise RuntimeError("Human feedback is closed")
        if self._phase == "aborted":
            raise RuntimeError(f"Human feedback aborted: {self._reason}")

    @staticmethod
    def _identity(row):
        if not isinstance(row, dict):
            raise ValueError("Feedback rows must be mappings")
        env_id, episode_id, step_id = (row.get(key) for key in ("env_id", "episode_id", "step_id"))
        if (type(env_id) is not int or env_id < 0 or not isinstance(episode_id, str) or not episode_id
                or type(step_id) is not int or step_id < 0):
            raise ValueError("Feedback rows require actual top-level env_id, episode_id and step_id")
        return env_id, episode_id, step_id

    def begin(self, reset_rows):
        """Register completed resets, retaining any other active environments."""
        if not isinstance(reset_rows, list) or not reset_rows:
            raise ValueError("begin requires nonempty reset rows")
        identities = [self._identity(row) for row in reset_rows]
        if (len({item[0] for item in identities}) != len(identities)
                or len({item[1] for item in identities}) != len(identities)
                or any(item[2] != 0 for item in identities)):
            raise ValueError("Reset rows require distinct environments/episodes at step zero")
        with self._lock, _mailbox(self.directory):
            self._require_open()
            if self._resolving:
                raise RuntimeError("Cannot register resets while feedback resolution is pending")
            for env_id, episode_id, _ in identities:
                prior = self._slots.get(env_id)
                if episode_id in self._seen or (prior is not None and prior["phase"] in _ACTIVE_PHASES):
                    raise ValueError("Cannot reuse an episode ID or replace an active episode")
            for env_id, episode_id, step_id in identities:
                self._slots[env_id] = dict(env_id=env_id, episode_id=episode_id, step_id=step_id,
                                          phase="active", human_label=None)
                self._seen.add(episode_id)
            self._phase, self._reason = "running", ""
            self._publish()

    def _abort(self, reason):
        self._phase, self._reason = "aborted", reason
        self._publish()
        raise RuntimeError(reason)

    def _requests(self):
        requests = {}
        for slot in self._slots.values():
            if slot["phase"] not in _ACTIVE_PHASES:
                continue
            path = _command_path(self.directory, self.epoch_token, slot["episode_id"])
            if not path.exists():
                continue
            try:
                request = json.loads(path.read_text())
                if (not isinstance(request, dict) or request.get("schema_version") != 1
                        or request.get("epoch_token") != self.epoch_token
                        or request.get("env_id") != slot["env_id"]
                        or request.get("episode_id") != slot["episode_id"]
                        or request.get("command") not in _COMMANDS):
                    raise ValueError("Invalid or stale human-feedback command")
            except (ValueError, TypeError, OSError) as error:
                self._abort(f"Invalid human-feedback mailbox command: {error}")
            if request["command"] == "abort":
                self._abort(f"Human requested abort for episode {slot['episode_id']}")
            requests[slot["episode_id"]] = request["command"]
        return requests

    def _validate_steps(self, rows):
        if not isinstance(rows, list) or not rows:
            raise ValueError("resolve requires nonempty step rows")
        identities = [self._identity(row) for row in rows]
        if len({item[0] for item in identities}) != len(identities):
            raise ValueError("Duplicate environment in step rows")
        for row, (env_id, episode_id, step_id) in zip(rows, identities):
            slot = self._slots.get(env_id)
            if (slot is None or slot["phase"] not in _ACTIVE_PHASES or slot["episode_id"] != episode_id
                    or step_id != slot["step_id"] + 1):
                raise ValueError("Step rows refer to a stale, inactive or out-of-order episode")
            if (type(row.get("terminated")) is not bool or type(row.get("truncated")) is not bool
                    or (row["terminated"] and row["truncated"]) or not isinstance(row.get("info"), dict)):
                raise ValueError("Step rows require exclusive termination flags and info")
            if row.get("terminal_observation_omitted", False):
                raise ValueError("Human feedback requires full terminal observations")
            _number(row.get("reward"), "environment reward")

    def resolve(self, step_rows, *, cancel_event=None):
        """Override a completed chunk, preserving pending labels on cancellation."""
        _check_cancelled(cancel_event)
        with self._lock, _mailbox(self.directory):
            self._require_open()
            _check_cancelled(cancel_event)
            if self._resolving:
                raise RuntimeError("A human-feedback resolution is already pending")
            self._pending_end = []
            self._validate_steps(step_rows)
            rows = deepcopy(step_rows)
            self._resolving = True
        deadline = time.monotonic() + self.timeout_s
        try:
            while True:
                _check_cancelled(cancel_event)
                with self._lock, _mailbox(self.directory):
                    self._require_open()
                    _check_cancelled(cancel_event)
                    requests = self._requests()
                    _check_cancelled(cancel_event)
                    waiting = [row for row in rows if (row["terminated"] or row["truncated"])
                               and row["episode_id"] not in requests]
                    if not waiting:
                        _check_cancelled(cancel_event)
                        return self._finish(rows, requests)
                    for row in waiting:
                        self._slots[row["env_id"]]["phase"] = "awaiting_label"
                    self._phase, self._reason = "waiting_for_labels", "environment_finished_without_human_label"
                    self._publish()
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self._abort("Timed out waiting for human outcome labels; abort the run")
                wait_s = min(self.poll_interval_s, remaining)
                if cancel_event is not None:
                    wait_s = min(wait_s, .05)
                self._wake.wait(wait_s)
        finally:
            with self._lock:
                self._resolving = False

    def _finish(self, rows, requests):
        pending_end = []
        for row in rows:
            slot = self._slots[row["env_id"]]
            original = deepcopy(row["info"])
            environment_done = row["terminated"] or row["truncated"]
            row["info"]["environment_outcome"] = dict(
                reward=row["reward"], terminated=row["terminated"], truncated=row["truncated"],
                success=original.get("success", False), info=original)
            label = requests.get(row["episode_id"])
            row["info"].update(success=label == "success", outcome_source="human", human_label=label)
            row["reward"] = 0. if label is None else (self.success_reward if label == "success" else self.failure_reward)
            slot["step_id"] = row["step_id"]
            if label is not None:
                if not environment_done:
                    pending_end.append({key: row[key] for key in ("env_id", "episode_id", "step_id")})
                row.update(terminated=True, truncated=False)
                slot.update(phase="retired", human_label=label)
                _command_path(self.directory, self.epoch_token, row["episode_id"]).unlink()
            else:
                row.update(terminated=False, truncated=False)
                slot.update(phase="active", human_label=None)
        self._pending_end = pending_end
        self._phase = "running" if any(slot["phase"] in _ACTIVE_PHASES for slot in self._slots.values()) else "idle"
        self._reason = ""
        self._publish()
        return rows

    @property
    def pending_end(self):
        with self._lock:
            return deepcopy(self._pending_end)

    def describe(self):
        return dict(mode="human", directory=str(self.directory), timeout_s=self.timeout_s,
                    poll_interval_s=self.poll_interval_s, success_reward=self.success_reward,
                    failure_reward=self.failure_reward, environment_rewards="discarded",
                    labels_effective_at="action_chunk_boundary")

    def status(self):
        with self._lock:
            return self._snapshot()

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._wake.set()
            try:
                with _mailbox(self.directory):
                    self._phase, self._reason = "closed", "closed"
                    self._publish()
            finally:
                self._owner.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status", help="Show actual per-environment episode IDs and phases")
    for command in _COMMANDS:
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--episode", required=True, help="Actual episode_id from trainer status")
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
