"""Record measured fast-rollout states without changing its render cadence.

The resulting trajectories are intended for offline video rendering.  This
backend adds only tensor reads and file writes: policy observations still use
the inherited chunk-boundary/exact-terminal captures, and physics advances only
where :class:`FastIsaacVectorBackend` already advances it.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .fast_simulation import FastIsaacVectorBackend
from .simulation import _file_identity, _write_json


class RecordingFastIsaacVectorBackend(FastIsaacVectorBackend):
    """Save full 120 Hz measured trajectories alongside ordinary rollouts.

    ``trajectory_directory`` defaults to ``output / "trajectories"``.  Each
    reset gets a fresh episode directory containing ``physics.npz`` and
    ``metadata.json``; completed and interrupted episodes are listed in
    ``index.jsonl``.  No Isaac imports or initialization occur at module import.
    """

    def __init__(self, *args, trajectory_directory=None, **kwargs):
        self._trajectory_slots = {}
        self._trajectory_serial = 0
        self._trajectory_step_active = False
        self._trajectory_reset_depth = 0
        self._trajectory_closed = False
        super().__init__(*args, **kwargs)
        self.trajectory_directory = (
            Path(trajectory_directory).resolve() if trajectory_directory is not None
            else self.output / "trajectories"
        )
        self.trajectory_directory.mkdir(parents=True, exist_ok=True)
        self._trajectory_source = _file_identity(Path(__file__))
        extensions = {"trajectory_recording": self._trajectory_source}
        server_extension = self.root / "scripts/serve_rl_recording_simulation.py"
        if server_extension.is_file():
            extensions["recording_server"] = _file_identity(server_extension)
        _write_json(self.trajectory_directory / "trajectory_manifest.json", dict(
            schema_version=1, extensions=extensions, capture_hz=round(1 / self.dt),
            simulation_manifest=str(self.output / "simulation_manifest.json"),
            additional_physics_steps=0, additional_rgb_captures=0,
            operations="Read measured PhysX tensor state and write it to disk",
            policy_observation_render_cadence="unchanged_chunk_boundary_and_exact_terminal",
            physical_termination="inherited_fast_backend_without_override",
            offline_rendering="Apply saved measured states without integrating physics",
        ))

    def reset(self, episodes, seed):
        # The inherited full reset dispatches to self.reset_envs after its
        # first cohort.  Only the outermost call starts/records new episodes.
        self._trajectory_reset_depth += 1
        try:
            rows = super().reset(episodes, seed)
        finally:
            self._trajectory_reset_depth -= 1
        if self._trajectory_reset_depth == 0:
            self._start_trajectories(range(self.num_envs))
        return rows

    def reset_envs(self, episodes, seed):
        self._trajectory_reset_depth += 1
        try:
            rows = super().reset_envs(episodes, seed)
        finally:
            self._trajectory_reset_depth -= 1
        if self._trajectory_reset_depth == 0:
            self._start_trajectories(sorted(episodes))
        return rows

    def _start_trajectories(self, ids):
        ids = [int(eid) for eid in ids]
        run_id = self._run_id
        run_label = str(run_id) if run_id is not None else "unowned"
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_label)[:80] or "run"
        run_hash = hashlib.sha256(run_label.encode("utf-8")).hexdigest()[:12]
        run_directory = self.trajectory_directory / f"run_{safe_label}_{run_hash}"
        run_directory.mkdir(parents=True, exist_ok=True)
        joint_names = list(self.arms.view.shared_metatype.dof_names)
        for eid in ids:
            previous = self._trajectory_slots.get(eid)
            if previous is not None and not previous["finalized"]:
                raise RuntimeError(f"Reset would discard unfinalized trajectory for env {eid}")
            item, env = self._items[eid], self.envs[eid]
            if item["ticks"] != 0 or item["done"]:
                raise RuntimeError("A recorded trajectory must begin at live physics tick zero")
            serial = self._trajectory_serial
            self._trajectory_serial += 1
            directory = run_directory / f"episode_{serial:06d}_env_{eid:03d}"
            directory.mkdir(exist_ok=False)
            self._trajectory_slots[eid] = dict(
                item_identity=id(item), directory=directory, rows=[], finalized=False,
                metadata=dict(
                    schema_version=1, trajectory_id=serial, run_id=run_id, env_id=eid,
                    joint_names=joint_names, env_offset_m=np.asarray(env.offset).tolist(),
                    reset_episode=copy.deepcopy(item["reset_episode"]), seed=item["seed"],
                    physics_dt=self.dt, capture_hz=round(1 / self.dt),
                    button_floors=list(range(24, 36)),
                    button_position_frame="world", button_quaternion_order="wxyz",
                    button_velocity_order=["linear_x", "linear_y", "linear_z",
                                           "angular_x", "angular_y", "angular_z"],
                    state_frame="base_link", state_link="gripper_tcp",
                    source=self._trajectory_source,
                    recording_contract="read_only_actual_state_after_each_original_physics_tick",
                    policy_render_cadence="unchanged_chunk_boundary_and_exact_terminal",
                    lighting_reconstruction="Apply info.events through each physics_index",
                ),
            )
        self._append_trajectory_samples(ids)

    def _refresh_metrics(self):
        super()._refresh_metrics()
        if self._trajectory_step_active:
            # Called after world.step and before this tick's termination/light
            # decisions.  Include newly terminal states; reconstruct lights
            # from the final timestamped events, which are saved after step.
            ids = [int(eid) for eid in sorted(self.recording)
                   if not self._items[eid]["done"]]
            self._append_trajectory_samples(ids)

    def _append_trajectory_samples(self, ids):
        if not ids:
            return
        q, velocity = self._joint_arrays()
        positions, orientations = self.button_view.get_world_poses()
        button_velocities = self.button_view.get_velocities()
        positions = np.asarray(positions)[self.button_rows]
        orientations = np.asarray(orientations)[self.button_rows]
        button_velocities = np.asarray(button_velocities)[self.button_rows]
        world_time = float(self.world.current_time)
        for eid in ids:
            slot = self._trajectory_slots.get(eid)
            item = self._items[eid]
            if (slot is None or slot["finalized"]
                    or slot["item_identity"] != id(item)):
                raise RuntimeError(f"Missing live trajectory for env {eid}")
            tick = int(item["ticks"])
            expected_tick = len(slot["rows"])
            if tick != expected_tick:
                raise RuntimeError(
                    f"Non-contiguous trajectory for env {eid}: {tick} != {expected_tick}"
                )
            state = self._metrics_cache["state"][eid]
            row = dict(
                physics_index=np.int64(tick), sim_time=tick * self.dt,
                world_time=world_time, q_actual=q[eid].copy(),
                qd_actual=velocity[eid].copy(), state=np.asarray(state).copy(),
                button_position_world=positions[eid].copy(),
                button_orientation_wxyz=orientations[eid].copy(),
                button_velocity_world=button_velocities[eid].copy(),
                contact_force=self.forces[eid].copy(),
            )
            if not all(np.isfinite(value).all() for value in row.values()):
                raise RuntimeError(f"Nonfinite trajectory state for env {eid} at tick {tick}")
            slot["rows"].append(row)

    def step(self, actions):
        if self._trajectory_step_active:
            raise RuntimeError("Nested recorded simulation step")
        self._trajectory_step_active = True
        try:
            rows = super().step(actions)
        finally:
            self._trajectory_step_active = False
        for row in rows:
            if row["terminated"] or row["truncated"]:
                self._save_trajectory(int(row["env_id"]), complete=True, info=row["info"])
        return rows

    def _save_trajectory(self, eid, *, complete, info=None):
        slot = self._trajectory_slots[eid]
        if slot["finalized"]:
            return
        samples = slot["rows"]
        if not samples:
            raise RuntimeError(f"Cannot save empty trajectory for env {eid}")
        arrays = {key: np.asarray([row[key] for row in samples]) for key in samples[0]}
        if not np.array_equal(arrays["physics_index"], np.arange(len(samples))):
            raise RuntimeError(f"Trajectory contains missing physics ticks for env {eid}")
        final_info = copy.deepcopy(info if info is not None else self._info(eid))
        if complete and int(final_info["physics_index"]) != int(arrays["physics_index"][-1]):
            raise RuntimeError(f"Terminal trajectory/observation tick mismatch for env {eid}")
        directory = slot["directory"]
        path = directory / "physics.npz"
        temporary = directory / "physics.npz.tmp"
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)
        metadata = dict(
            slot["metadata"], status="complete" if complete else "incomplete",
            samples=len(samples), first_physics_index=int(arrays["physics_index"][0]),
            last_physics_index=int(arrays["physics_index"][-1]), info=final_info,
            arrays={key: dict(shape=list(value.shape), dtype=str(value.dtype))
                    for key, value in arrays.items()},
            physics_file=_file_identity(path),
        )
        _write_json(directory / "metadata.json", metadata)
        entry = dict(
            trajectory_id=metadata["trajectory_id"], run_id=metadata["run_id"],
            env_id=eid, directory=str(directory), status=metadata["status"],
            floor=metadata["reset_episode"]["floor"], samples=metadata["samples"],
            success=final_info.get("success"), termination=final_info.get("termination"),
        )
        with (self.trajectory_directory / "index.jsonl").open("a") as stream:
            stream.write(json.dumps(entry, allow_nan=False, sort_keys=True) + "\n")
        slot["finalized"] = True
        slot["rows"] = []

    def close(self):
        if self._trajectory_closed:
            return
        self._trajectory_closed = True
        try:
            for eid, slot in list(self._trajectory_slots.items()):
                if not slot["finalized"]:
                    item = self._items[eid]
                    complete = bool(item["done"]) and (
                        slot["rows"][-1]["physics_index"] == item["ticks"]
                    )
                    self._save_trajectory(eid, complete=complete)
        finally:
            super().close()
