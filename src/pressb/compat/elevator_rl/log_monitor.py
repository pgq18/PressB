"""Read existing online-RL logs without importing or touching the trainer.

``TrainingLogReader(train_directory).poll()`` returns ``(new_history, latest)``.
History is emitted once in metrics.jsonl order; latest is a status.json snapshot.
Derived episode metrics use each row's completed-episode prefix, never the newer
tail of episodes.jsonl. Missing denominators/rate spans are represented by None.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter, deque
import json
import math
import os
from pathlib import Path


class LogConsistencyError(ValueError):
    """A log was replaced, malformed, or contradicts its episode prefix."""


class _JsonlTail:
    def __init__(self, path):
        self.path = path
        self.offset = 0
        self.line_number = 0
        self.identity = None
        self.anchor = b""

    def read(self):
        try:
            stream = self.path.open("rb")
        except FileNotFoundError:
            if self.identity is not None:
                raise LogConsistencyError(f"Previously observed log disappeared: {self.path}")
            return []
        with stream:
            stat = os.fstat(stream.fileno())
            identity = (stat.st_dev, stat.st_ino)
            if self.identity is not None and identity != self.identity:
                raise LogConsistencyError(f"Log was rotated or replaced: {self.path}")
            if stat.st_size < self.offset:
                raise LogConsistencyError(f"Log was truncated: {self.path}")
            stream.seek(self.offset - len(self.anchor))
            if stream.read(len(self.anchor)) != self.anchor:
                raise LogConsistencyError(f"Previously read log bytes changed: {self.path}")
            rows, offset, line_number = [], self.offset, self.line_number
            while line := stream.readline():
                if not line.endswith(b"\n"):
                    break  # The writer has not committed this line yet.
                line_number += 1
                try:
                    row = json.loads(line)
                except (ValueError, UnicodeError) as error:
                    raise LogConsistencyError(f"Malformed JSON in {self.path}:{line_number}") from error
                if not isinstance(row, dict):
                    raise LogConsistencyError(f"Expected object in {self.path}:{line_number}")
                rows.append(row)
                offset = stream.tell()
            # Commit only after validating every complete line in this read.
            stream.seek(max(0, offset - 64))
            self.anchor = stream.read(min(offset, 64))
            self.offset, self.line_number, self.identity = offset, line_number, identity
            return rows


def _number(value, name, *, integer=False):
    if (type(value) not in (int, float) or not math.isfinite(value) or value < 0
            or (integer and value != int(value))):
        raise LogConsistencyError(f"Invalid {name}: {value!r}")
    return int(value) if integer else float(value)


class TrainingLogReader:
    """Incremental, read-only monitoring of a train or eval output directory."""

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.metrics = _JsonlTail(self.run_dir / "metrics.jsonl")
        self.episodes = _JsonlTail(self.run_dir / "episodes.jsonl")
        self.pending = deque()
        # Keep no contacts, observations, event payloads, or model state.
        self.completed = []  # (success, sim_seconds, termination, floor)
        self.success_prefix = [0]
        self.seconds_prefix = [0.]
        self.reasons = {"target_pressed", "wrong_button_pressed", "time_limit"}
        self.times = []
        self.points = []  # Only history's (wall_seconds, transitions, updates).

    def _append_episode(self, row):
        success = row.get("success")
        if type(success) is not bool:
            raise LogConsistencyError("Episode success must be boolean")
        seconds = _number(row.get("sim_seconds"), "episode sim_seconds")
        reason = row.get("termination")
        if not isinstance(reason, str) or not reason:
            raise LogConsistencyError("Episode termination must be a nonempty string")
        layout = row.get("layout", {})
        floor = layout.get("floor", row.get("floor")) if isinstance(layout, dict) else None
        if type(floor) is not int or floor not in range(24, 36):
            raise LogConsistencyError(f"Invalid episode floor: {floor!r}")
        self.completed.append((success, seconds, reason, floor))
        self.success_prefix.append(self.success_prefix[-1] + success)
        self.seconds_prefix.append(self.seconds_prefix[-1] + seconds)
        self.reasons.add(reason)

    def _enrich(self, row):
        count = _number(row.get("episodes"), "episodes", integer=True)
        successes = _number(row.get("successes"), "successes", integer=True)
        if successes > count:
            raise LogConsistencyError("successes exceeds completed episodes")
        if count > len(self.completed):
            return None  # A concurrent episode append can catch up next poll.
        if self.success_prefix[count] != successes:
            raise LogConsistencyError(
                f"Episode prefix mismatch at {count}: log successes={successes}, "
                f"episodes.jsonl successes={self.success_prefix[count]}")
        result = dict(row)
        for window in (100, 1000):
            start = max(0, count - window)
            denominator = count - start
            result[f"recent_count_{window}"] = denominator
            result[f"recent_sr_{window}"] = (
                (successes - self.success_prefix[start]) / denominator if denominator else None)
        start = max(0, count - 1000)
        denominator = count - start
        result["episode_seconds_recent_1000"] = (
            (self.seconds_prefix[count] - self.seconds_prefix[start]) / denominator
            if denominator else None)
        reasons, floors, floor_successes = Counter(), Counter(), Counter()
        for success, _, reason, floor in self.completed[start:count]:
            reasons[reason] += 1
            floors[floor] += 1
            floor_successes[floor] += success
        for reason in sorted(self.reasons):
            result[f"termination_recent_{reason}"] = reasons[reason] / denominator if denominator else None
        for floor in range(24, 36):
            result[f"floor_recent_count_{floor}"] = floors[floor]
            result[f"floor_recent_sr_{floor}"] = (
                floor_successes[floor] / floors[floor] if floors[floor] else None)
        wall = _number(row.get("wall_seconds"), "wall_seconds")
        transitions = _number(row.get("transitions"), "transitions", integer=True)
        updates = _number(row.get("updates"), "updates", integer=True)
        before = bisect_right(self.times, wall - 60.) - 1
        if before < 0 and bisect_left(self.times, wall) > 0:
            before = 0  # Less than 60 seconds available; expose the real span.
        result["speed_window_seconds"] = 0.
        result["transitions_per_second_recent_60s"] = None
        result["updates_per_second_recent_60s"] = None
        if before >= 0:
            old_wall, old_transitions, old_updates = self.points[before]
            span = wall - old_wall
            if transitions < old_transitions or updates < old_updates:
                raise LogConsistencyError("Training counters decreased across rate samples")
            result["speed_window_seconds"] = span
            result["transitions_per_second_recent_60s"] = (transitions - old_transitions) / span
            result["updates_per_second_recent_60s"] = (updates - old_updates) / span
        return result

    def poll(self):
        """Backfill then tail complete history lines and enrich the current status.

        Rows waiting for their episode prefix remain queued in original order.
        A missing status or a status whose prefix is incomplete yields None.
        A repeated poll never re-emits history, but may return the same status.
        """
        self.pending.extend(self.metrics.read())
        try:
            status_text = (self.run_dir / "status.json").read_text()
        except FileNotFoundError:
            status = None
        else:
            try:
                status = json.loads(status_text)
            except ValueError as error:
                raise LogConsistencyError("Malformed status.json") from error
            if not isinstance(status, dict):
                raise LogConsistencyError("status.json must contain an object")
        # Read episodes AFTER snapshots so their associated prefixes are present.
        for episode in self.episodes.read():
            self._append_episode(episode)
        history = []
        while self.pending:
            row = self._enrich(self.pending[0])
            if row is None:
                break
            point = (float(row["wall_seconds"]), int(row["transitions"]), int(row["updates"]))
            if self.points and any(new < old for new, old in zip(point, self.points[-1])):
                raise LogConsistencyError("Historical time or training counters decreased")
            history.append(row)
            self.pending.popleft()
            self.times.append(point[0])
            self.points.append(point)
        return history, self._enrich(status) if status is not None else None
