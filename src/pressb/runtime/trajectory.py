"""Task-independent episode recorder for validated chunk transitions.

This is a chunk-level interchange format, NOT a camera-frame LeRobot dataset.
Requested action prefixes and the measured elapsed duration are both preserved;
we never fabricate per-frame states or label an unexecuted suffix as executed.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported trajectory value: {type(value).__name__}")


class ChunkTrajectoryWriter:
    def __init__(self, directory, *, include_images=True):
        if type(include_images) is not bool:
            raise ValueError("include_images must be boolean")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.include_images = include_images
        self.episodes = {}
        self.transitions = 0
        self.closed = False

    def _write(self, path, row, *, append=False):
        with path.open("a" if append else "x") as stream:
            stream.write(json.dumps(row, default=_json, allow_nan=False) + "\n")

    def _observation(self, observation):
        result = deepcopy(observation)
        if not self.include_images:
            result.pop("images", None)
        return result

    def on_reset(self, rows):
        for row in rows:
            identity = row["episode_id"]
            if identity in self.episodes:
                raise ValueError("Recorder cannot reuse an episode identity")
            folder = self.directory / hashlib.sha256(identity.encode()).hexdigest()[:24]
            folder.mkdir()
            self.episodes[identity] = {"directory": folder.name, "env_id": row["env_id"],
                "transitions": 0, "complete": False, "success": False}
            initial = {key: deepcopy(value) for key, value in row.items() if key != "observation"}
            initial["observation"] = self._observation(row["observation"])
            self._write(folder / "initial.json", initial)

    def on_transition(self, before, actions_executed, rows):
        if not (len(before) == len(actions_executed) == len(rows)):
            raise ValueError("Recorder transition cardinality mismatch")
        for previous, prefix, row in zip(before, actions_executed, rows):
            identity = previous["episode_id"]
            entry = self.episodes[identity]
            if row["episode_id"] != identity or entry["complete"]:
                raise ValueError("Recorder received an invalid episode transition")
            record = {"env_id": previous["env_id"], "episode_id": identity,
                "step_id": previous["step_id"], "observation": self._observation(previous["observation"]),
                "requested_action_prefix": prefix, "next_step_id": row["step_id"],
                "next_observation": self._observation(row["observation"]),
                "reward": row["reward"], "discount": row["discount"],
                "terminated": row["terminated"], "truncated": row["truncated"],
                "info": row["info"], "terminal_observation_omitted": row.get("terminal_observation_omitted", False)}
            self._write(self.directory / entry["directory"] / "transitions.jsonl", record, append=True)
            entry["transitions"] += 1
            self.transitions += 1
            if row["terminated"] or row["truncated"]:
                entry["complete"] = True
                entry["success"] = row["info"]["success"]
                self._write(self.directory / entry["directory"] / "outcome.json", entry)

    def close(self, status):
        if self.closed:
            return
        self.closed = True
        self._write(self.directory / "manifest.json", {"schema": "pressb.chunk_trajectory.v1",
            "include_images": self.include_images, "transitions": self.transitions,
            "episodes": self.episodes, "run_status": status,
            "action_semantics": "requested prefix after runner projection, before optional simulator smoothing; actual duration in info.elapsed_control_steps",
            "timebase": "chunk transitions; no synthetic per-frame samples"})
