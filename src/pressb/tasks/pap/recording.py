"""Optional read-only PAP state recording for faithful offline visualization.

No renderer call, physics step, command update, or object attachment is performed
here. Samples contain measured state at control boundaries and exact endings.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def _identity(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size}


class PAPStateRecorder:
    def __init__(self, backend, config):
        if not isinstance(config, dict) or set(config)-{"directory", "env_ids", "max_episodes_per_env"}:
            raise ValueError("trajectory_recording supports directory, env_ids and max_episodes_per_env")
        if not isinstance(config.get("directory"), str) or not config["directory"]:
            raise ValueError("trajectory_recording.directory must be a nonempty path")
        self.ids = config.get("env_ids", list(range(min(3, backend.num_envs))))
        backend._ids(self.ids)
        self.maximum = config.get("max_episodes_per_env", 1)
        if type(self.maximum) is not int or self.maximum <= 0:
            raise ValueError("trajectory_recording.max_episodes_per_env must be a positive integer")
        self.directory = Path(config["directory"]).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        if any(self.directory.iterdir()):
            raise ValueError("trajectory_recording.directory must be empty")
        self.backend = backend
        self.counts = {i: 0 for i in self.ids}
        self.slots = {}
        self.config = copy.deepcopy(backend.config)
        self.config.pop("trajectory_recording", None)
        self.manifest = dict(schema_version=1, format="pap_measured_state", env_ids=self.ids,
            max_episodes_per_env=self.maximum, task_identity=copy.deepcopy(backend.task.identity),
            runtime_identity=copy.deepcopy(backend.describe()),
            joint_names=list(backend.engine.robot.arms.view.shared_metatype.dof_names),
            physics_dt=backend.config["physics_dt"], control_hz=backend.config["control_hz"],
            sample_contract="initial, each control boundary, exact terminal tick; read-only tensor reads",
            config=self.config, recorder_source=_identity(Path(__file__)))
        _json(self.directory/"manifest.json", self.manifest)

    def begin(self, env_ids):
        for i in env_ids:
            if i in self.slots:
                self.finish(i, complete=False, info={"termination_reason": "reset_before_terminal"})
        selected = [i for i in env_ids if i in self.counts and self.counts[i] < self.maximum]
        for i in selected:
            self.slots[i] = dict(samples=[], episode_index=self.counts[i],
                reset=copy.deepcopy(self.backend.reset_specs[i]),
                source_env_offset=self.backend.engine.offsets[i].tolist())
            self.counts[i] += 1
        self.append(selected)

    def append(self, env_ids, results=None, endings=None):
        selected = [i for i in env_ids if i in self.slots]
        if not selected:
            return
        snapshots = self.backend.snapshot(selected)
        for i in selected:
            row = snapshots[str(i)]
            row["command"] = self.backend.engine.robot.commands[i].tolist()
            row["contact_counts"] = self.backend.task.contact_counts[i].tolist()
            row["success"] = bool((endings or {}).get(i) == "success")
            outcome = (results or {}).get(i, {})
            row["physical_success"] = bool(outcome.get("info", {}).get("physical_outcome", {}).get(
                "success", outcome.get("success", False)))
            samples = self.slots[i]["samples"]
            tick = row["episode_physics_steps"]
            if samples and tick <= samples[-1]["episode_physics_steps"]:
                raise RuntimeError("Recorded PAP physics samples must have increasing episode ticks")
            samples.append(row)

    def finish(self, env_id, *, complete, info):
        slot = self.slots.pop(env_id, None)
        if slot is None:
            return
        samples = slot.pop("samples")
        if not samples:
            raise RuntimeError("Cannot save empty PAP trajectory")
        final_tick = samples[-1]["episode_physics_steps"]
        if complete and final_tick != info["episode_physics_steps"]:
            raise RuntimeError("PAP recording does not include the exact terminal physics state")
        if complete:
            samples[-1]["success"] = bool(info["success"])
            samples[-1]["physical_success"] = bool(info.get("physical_outcome", {}).get("success", info["success"]))
        arrays = {key: np.asarray([row[key] for row in samples]) for key in samples[0]}
        if any(not np.isfinite(value).all() for value in arrays.values()):
            raise RuntimeError("PAP recorded state is nonfinite")
        stem = f"env_{env_id:03d}_episode_{slot['episode_index']:03d}"
        path = self.directory/f"{stem}.npz"
        np.savez_compressed(path, **arrays)
        meta = dict(schema_version=1, format="pap_measured_state", complete=bool(complete),
            env_id=env_id, **slot, sample_count=len(samples), final_physics_tick=final_tick,
            duration_seconds=final_tick*self.backend.config["physics_dt"], info=copy.deepcopy(info),
            trajectory=_identity(path), manifest=_identity(self.directory/"manifest.json"))
        _json(self.directory/f"{stem}.json", meta)
        with (self.directory/"index.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(metadata=str((self.directory/f"{stem}.json").resolve()),
                env_id=env_id, episode_index=slot["episode_index"], complete=bool(complete),
                success=bool(info.get("success", False)), duration_seconds=meta["duration_seconds"]))+"\n")

    def on_reset(self, backend, env_ids, rows):
        self.begin(env_ids)

    def after_tick(self, backend, live_ids, outcomes):
        # The shared runtime calls this after outcome/ready updates, before a
        # peer can advance the world or fused reset can replace the terminal.
        stride = backend.execution["physics_ticks_per_action"]
        selected = []
        for i in live_ids:
            if i not in self.slots:
                continue
            tick = int(backend.ticks[i])
            samples = self.slots[i]["samples"]
            last = samples[-1]["episode_physics_steps"] if samples else -1
            if tick > last and (tick % stride == 0 or not backend.ready[i]):
                selected.append(i)
        self.append(selected, outcomes, {i: outcomes[i].get("reason") for i in selected})

    def after_step(self, backend, rows):
        for row in rows:
            if row["terminated"] or row["truncated"]:
                self.finish(row["env_id"], complete=True, info=row["info"])

    def close(self):
        for i in list(self.slots):
            self.finish(i, complete=False, info={"termination_reason": "recorder_closed_before_terminal"})


# Recording subscribes to the common runtime; it never wraps world.step or
# duplicates the episode scheduler. Source identity is recorded separately.
from .vector import PAPVectorBackend


class PAPRecordingVectorBackend(PAPVectorBackend):
    """Shared simulation with read-only measured-state observer hooks."""

    def __init__(self, config, *, engine_factory=None):
        self.recorder = None
        super().__init__(config, engine_factory=engine_factory)
        try:
            self.recorder = PAPStateRecorder(self, self.config.get("trajectory_recording"))
            self.add_observer(self.recorder)
        except BaseException:
            self.close()
            raise

    def describe(self):
        result = super().describe()
        if self.recorder is not None:
            result["trajectory_recording"] = {"directory": str(self.recorder.directory),
                "env_ids": self.recorder.ids, "manifest": _identity(self.recorder.directory/"manifest.json"),
                "source": _identity(Path(__file__)), "additional_render_or_physics_calls": 0}
        return result


def create_backend(config):
    return PAPRecordingVectorBackend(config)
