"""One parallel episode state machine for every native simulation task.

Task plugins never step physics or render. The same loop handles joint and pose
actions, tick-accurate or control-interval outcomes, indexed resets and fused
unbootstrapped terminals. Isaac is loaded only by the selected engine factory.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
import math
from pathlib import Path
import threading
import time

import numpy as np

from .contract import fingerprint, normalize_env_seeds, validate_actions, validate_metadata


def execution_config(values):
    cfg = deepcopy(values)
    defaults = dict(num_envs=1, gpu=0, horizon=32, physics_dt=1/120,
        control_hz=30., max_episode_seconds=20., settling_ticks=90, spacing_m=6.,
        render_resolution=[640, 480], render_subframes=1, observation_render_frames=2,
        reset_render_frames=2, cpu_threads=8, png_workers=8, gamma=.995,
        renderer_profile="calibrated", terminal_boundary="physics_tick", timeout_reason="timeout",
        headless=True, image_preprocessing=None, image_output_resolution=None)
    cfg = {**defaults, **cfg}
    for key in ("num_envs", "gpu", "horizon", "settling_ticks", "render_subframes",
                "observation_render_frames", "reset_render_frames", "cpu_threads", "png_workers"):
        minimum = 0 if key in ("gpu", "settling_ticks") else 1
        if type(cfg[key]) is not int or cfg[key] < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    for key in ("physics_dt", "control_hz", "max_episode_seconds", "spacing_m", "gamma"):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], (int, float)) or not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if cfg["gamma"] > 1:
        raise ValueError("gamma must be <= 1")
    stride = 1 / (cfg["physics_dt"] * cfg["control_hz"])
    if not math.isclose(stride, round(stride), abs_tol=1e-8) or stride < 1:
        raise ValueError("control_hz must divide the physics frequency")
    cfg["physics_ticks_per_action"] = int(round(stride))
    cfg["max_ticks"] = int(math.floor(cfg["max_episode_seconds"] / cfg["physics_dt"] + 1e-9))
    if cfg["max_ticks"] < 1:
        raise ValueError("Episode duration must contain at least one physics tick")
    if cfg["terminal_boundary"] not in ("physics_tick", "control_interval"):
        raise ValueError("terminal_boundary must be physics_tick or control_interval")
    if cfg["terminal_boundary"] == "control_interval" and cfg["max_ticks"] % cfg["physics_ticks_per_action"]:
        raise ValueError("Control-interval time limit must end on a control boundary")
    for name in ("render_resolution", "image_output_resolution"):
        shape = cfg[name]
        if shape is not None and (not isinstance(shape, (tuple, list)) or len(shape) != 2 or any(type(x) is not int or x < 1 for x in shape)):
            raise ValueError(f"{name} must contain width and height")
    if type(cfg["headless"]) is not bool:
        raise ValueError("headless must be boolean")
    return cfg


class SharedWorldBackend:
    """Task-independent, main-thread parallel environment backend."""
    def __init__(self, task, *, engine_factory=None):
        self.task = task
        self.config = task.config
        self.execution = execution_config(task.execution)
        task.execution = self.execution
        self.num_envs = self.execution["num_envs"]
        self.ready = np.zeros(self.num_envs, dtype=bool)
        self.ticks = np.zeros(self.num_envs, dtype=np.int64)
        self.reset_specs = [None] * self.num_envs
        self._owner_thread = threading.get_ident()
        self._closed = False
        self._fault = None
        self.engine = None
        self._observers = []
        self.timings = dict(physics_seconds=0., render_seconds=0., encode_seconds=0.,
            step_seconds=0., reset_seconds=0., world_ticks=0, captures=0,
            fused_step_reset_seconds=0., fused_steps=0)
        try:
            if engine_factory is None:
                from .isaac_parallel import IsaacParallelWorld
                engine_factory = IsaacParallelWorld
            self.engine = engine_factory(task, self)
            self.engine.initialize()
            self._metadata = self._build_metadata()
        except BaseException:
            self.close()
            raise

    def _build_metadata(self):
        meta = deepcopy(self.task.metadata())
        cfg = self.execution
        meta.update(num_envs=self.num_envs, initial_reset="all_envs", reset_mode="indexed",
            scheduling="all_active", equal_execution_horizon=True,
            execution_horizons=list(range(1, cfg["horizon"] + 1)),
            capabilities=["per_env_reset_seeds", "terminal_state_only", "step_and_reset"],
            terminal_observation_modes=["full", "state_only_if_no_bootstrap"])
        if cfg["image_preprocessing"] is not None:
            meta["image_preprocessing"] = deepcopy(cfg["image_preprocessing"])
        details = getattr(self.engine, "execution_metadata", lambda: {})()
        meta["execution"] = {**meta.get("execution", {}), **details,
            "backend": "isaac_shared_world", "parallel_core": "pressb.runtime.parallel.SharedWorldBackend",
            "workers": 1, "vectorized": True, "gpu_ids": [cfg["gpu"]],
            "scheduling": "all_active_equal_prefix", "indexed_reset_physics_ticks": 0,
            "initial_settling_ticks": cfg["settling_ticks"], "terminal_boundary": cfg["terminal_boundary"],
            "physics_hz": 1/cfg["physics_dt"], "max_episode_seconds": cfg["max_episode_seconds"],
            "reset_contract": "indexed_zero_tick_cached_state_v2"}
        sources = deepcopy(meta.get("source_identity", {}))
        root = Path(__file__).resolve().parents[1]
        paths = [Path(__file__), Path(__file__).with_name("parallel_contract.py"),
                 Path(__file__).with_name("isaac_parallel.py"),
                 Path(__file__).with_name("physics.py"), Path(__file__).with_name("image_transport.py"),
                 Path(__file__).with_name("contract.py"), root/"sensors/tiled.py"]
        robot = getattr(self.engine, "robot", None)
        if robot is not None:
            source = inspect.getsourcefile(type(robot))
            if source and Path(source).resolve().is_relative_to(root):
                paths.append(Path(source))
        for path in paths:
            if path.is_file():
                content = path.read_bytes()
                sources[str(path.resolve().relative_to(root))] = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
        meta["source_identity"] = sources
        meta["source_revision"] = "sha256:" + fingerprint(sources)
        # Keep deployment addresses/log paths out of the semantic task identity.
        semantics = {key: cfg[key] for key in ("physics_dt", "control_hz", "horizon", "max_episode_seconds",
            "settling_ticks", "spacing_m", "render_resolution", "image_output_resolution", "renderer_profile",
            "render_subframes", "observation_render_frames", "reset_render_frames", "image_preprocessing",
            "terminal_boundary", "gamma")}
        semantics.update({key: deepcopy(cfg[key]) for key in
            ("anti_aliasing", "renderer_settings", "reset_render_subframes", "warmup_frames", "gpu_dynamics") if key in cfg})
        meta["task_config_sha256"] = fingerprint({"task": meta.get("task_config_sha256"), "parallel": semantics})
        return validate_metadata(meta, "simulation")

    def describe(self):
        return deepcopy(self._metadata)

    def add_observer(self, observer):
        """Attach read-only recording without adding another execution loop."""
        self._check()
        if observer in self._observers:
            raise ValueError("Observer is already attached")
        self._observers.append(observer)

    def _notify(self, method, *args):
        for observer in self._observers:
            callback = getattr(observer, method, None)
            if callback is not None:
                before = self.engine.world.current_time
                callback(self, *args)
                if self.engine.world.current_time != before:
                    raise RuntimeError("A recording observer advanced physical time")

    def _check(self):
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("Parallel physics and rendering require their owning main thread")
        if self._closed or self._fault:
            raise RuntimeError(f"Parallel backend closed or faulted: {self._fault}")
        self.engine.check()

    def _ids(self, ids):
        if (not isinstance(ids, list) or not ids or any(type(i) is not int or not 0 <= i < self.num_envs for i in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError("Expected distinct valid environment ids")

    def snapshot(self, env_ids=None):
        self._check()
        ids = list(range(self.num_envs)) if env_ids is None else env_ids
        self._ids(ids)
        robots = self.engine.robot.diagnostic_state(ids)
        objects = self.task.diagnostic_state(ids)
        return {str(i): {**deepcopy(robots[i]), **deepcopy(objects[i]),
            "episode_physics_steps": int(self.ticks[i]), "ready": bool(self.ready[i]),
            "world_time": float(self.engine.world.current_time)} for i in ids}

    diagnostic_snapshot = snapshot

    def _state_observations(self, ids):
        robots = self.engine.robot.observations(ids)
        fields = self.task.observation_fields(ids)
        return {i: {**deepcopy(fields[i]), **deepcopy(robots[i]), "images": {}} for i in ids}

    def _observations(self, ids, *, render_frames=None, reset_history=True):
        result = self._state_observations(ids)
        images = self.engine.capture(ids, self.execution["observation_render_frames"] if render_frames is None else render_frames,
                                     reset_history=reset_history)
        for i in ids:
            result[i]["images"] = images[i]
        return result

    def _prepare_reset(self, env_ids, seed, options=None, *, env_seeds=None):
        self._ids(env_ids)
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("reset seed must be an integer in [0,2**63)")
        seeds = normalize_env_seeds(env_ids, env_seeds)
        if options is not None and not isinstance(options, dict):
            raise ValueError("reset options must be a mapping")
        options = deepcopy(options or {})
        per_env = options.pop("by_env", {})
        if not isinstance(per_env, dict):
            raise ValueError("reset options.by_env must be a mapping")
        indexed = {}
        for key, value in per_env.items():
            if type(key) is int:
                eid = key
            elif isinstance(key, str) and key.isdecimal() and str(int(key)) == key:
                eid = int(key)
            else:
                raise ValueError("by_env keys must identify environment ids")
            if not 0 <= eid < self.num_envs or eid in indexed or not isinstance(value, dict):
                raise ValueError("by_env contains duplicate/unknown ids or non-mapping options")
            indexed[eid] = value
        return {i: deepcopy(self.task.sample_reset(i, seed if seeds is None else seeds[i],
                {**options, **indexed.get(i, {})})) for i in env_ids}

    def reset(self, env_ids, seed, options=None, *, env_seeds=None):
        self._check()
        specs = self._prepare_reset(env_ids, seed, options, env_seeds=env_seeds)
        if all(spec is None for spec in self.reset_specs) and set(env_ids) != set(range(self.num_envs)):
            raise ValueError("Initial reset must include all environments")
        try:
            return self._reset(env_ids, specs)
        except BaseException as exc:
            self._fault = f"reset: {type(exc).__name__}: {exc}"
            raise

    def _reset(self, env_ids, specs, *, capture_images=True):
        started = time.perf_counter()
        peers = [i for i in range(self.num_envs) if i not in env_ids]
        before = self.snapshot(peers) if peers else None
        clock = self.engine.world.current_time
        self.engine.robot.reset(env_ids, specs)
        for i in env_ids:
            self.reset_specs[i] = deepcopy(specs[i])
            self.ticks[i] = 0
        info = self.task.reset_task(env_ids, specs)
        self.engine.publish_transforms()
        self.engine.reset_render_history()
        self.engine._reset_capture = True
        try:
            observations = (self._observations(env_ids, render_frames=self.execution["reset_render_frames"])
                            if capture_images else self._state_observations(env_ids))
        finally:
            self.engine._reset_capture = False
        if self.engine.world.current_time != clock or (peers and self.snapshot(peers) != before):
            raise RuntimeError("Indexed reset changed a peer or advanced physical time")
        self.ready[env_ids] = True
        self.timings["reset_seconds"] += time.perf_counter()-started
        rows = [{"env_id": i, "observation": observations[i], "info": {
            **deepcopy(info.get(i, {})), "reset": deepcopy(specs[i]), "success": False,
            "settling_physics_steps": 0, "startup_settling_physics_steps": self.execution["settling_ticks"],
            "episode_physics_steps": 0, "elapsed_control_steps": 0.,
            "reset_contract": "indexed_zero_tick_cached_state_v2"}} for i in env_ids]
        self._notify("on_reset", env_ids, rows)
        return rows

    def end_episode(self, env_ids):
        self._check()
        self._ids(env_ids)
        if any(not self.ready[i] for i in env_ids):
            raise ValueError("Only active environments may end")
        self.ready[env_ids] = False

    def _commands(self, items):
        if not isinstance(items, list) or any(not isinstance(item, dict) or "env_id" not in item for item in items):
            raise ValueError("step requires action mappings")
        ids = [item["env_id"] for item in items]
        self._ids(ids)
        if set(ids) != set(np.flatnonzero(self.ready)):
            raise ValueError("Shared World requires every active environment in each step")
        commands = {item["env_id"]: validate_actions(item.get("actions"), self._metadata["action"], allow_prefix=True)
                    for item in items}
        if len({len(value) for value in commands.values()}) != 1:
            raise ValueError("Shared World requires equal action prefix lengths")
        validate_robot = getattr(self.engine.robot, "validate_actions", None)
        if validate_robot is not None:
            for action in commands.values():
                validate_robot(action)
        return ids, commands

    def step(self, items, *, terminal_observation_mode="full"):
        return self._step(items, terminal_observation_mode=terminal_observation_mode)

    def _step(self, items, *, terminal_observation_mode="full", defer_live_images=False):
        self._check()
        if terminal_observation_mode not in ("full", "state_only_if_no_bootstrap"):
            raise ValueError("Unsupported terminal observation mode")
        ids, commands = self._commands(items)
        cfg, started = self.execution, time.perf_counter()
        before = self.ticks.copy()
        applied = {i: [] for i in ids}
        rewards = {i: 0. for i in ids}
        results, observations = {}, {}
        try:
            for action_index in range(len(commands[ids[0]])):
                live = [i for i in ids if self.ready[i]]
                if not live:
                    break
                sent = self.engine.robot.begin_action(live, np.asarray([commands[i][action_index] for i in live]))
                for i in live:
                    applied[i].append(np.asarray(sent[i]).tolist())
                for substep in range(cfg["physics_ticks_per_action"]):
                    live = [i for i in ids if self.ready[i]]
                    if not live:
                        break
                    self.engine.robot.before_tick(live, substep, cfg["physics_ticks_per_action"])
                    self.task.before_tick(live)
                    self.ticks[live] += 1
                    tick_started = time.perf_counter()
                    self.engine.tick(live)
                    self.timings["physics_seconds"] += time.perf_counter()-tick_started
                    self.timings["world_ticks"] += 1
                    outcomes = self.task.after_tick(live)
                    boundary = cfg["terminal_boundary"] == "physics_tick" or substep == cfg["physics_ticks_per_action"]-1
                    if not boundary:
                        self._notify("after_tick", live, outcomes)
                        continue
                    ended = []
                    for i in live:
                        value = {"reward": 0., "success": False, "terminated": False, "truncated": False,
                                 "reason": None, "info": {}, **deepcopy(outcomes[i])}
                        for flag in ("success", "terminated", "truncated"):
                            if type(value[flag]) is not bool:
                                raise ValueError(f"Task outcome {flag} must be boolean")
                        if value["terminated"] and value["truncated"]:
                            raise ValueError("Task outcome cannot terminate and truncate simultaneously")
                        reward = value["reward"]
                        if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
                            raise ValueError("Task reward must be finite")
                        elapsed = int(self.ticks[i]-before[i])
                        exponent = (elapsed-1)/cfg["physics_ticks_per_action"]
                        if cfg["terminal_boundary"] == "control_interval":
                            exponent = math.floor(exponent)
                        rewards[i] += float(reward)*cfg["gamma"]**exponent
                        if not value["terminated"] and not value["truncated"] and self.ticks[i] >= cfg["max_ticks"]:
                            value.update(truncated=True, reason=cfg["timeout_reason"])
                        results[i] = value
                        if value["terminated"] or value["truncated"]:
                            self.ready[i] = False
                            ended.append(i)
                    self._notify("after_tick", live, results)
                    if ended:
                        observations.update(self._state_observations(ended) if terminal_observation_mode == "state_only_if_no_bootstrap"
                                            else self._observations(ended))
            live = [i for i in ids if self.ready[i]]
            if live:
                observations.update(self._state_observations(live) if defer_live_images else self._observations(live))
            rows = []
            for i in ids:
                value = results[i]
                elapsed = int(self.ticks[i]-before[i])
                controls = elapsed/cfg["physics_ticks_per_action"]
                info = {**value["info"], "success": value["success"], "termination_reason": value["reason"],
                    "physics_steps": elapsed, "elapsed_control_steps": controls,
                    "episode_physics_steps": int(self.ticks[i]), "episode_seconds": self.ticks[i]*cfg["physics_dt"],
                    "applied_actions": applied[i], "requested_action_count": len(commands[i])}
                row = dict(env_id=i, observation=observations[i], reward=rewards[i], discount=cfg["gamma"]**controls,
                    terminated=value["terminated"], truncated=value["truncated"], info=info)
                if (value["terminated"] or value["truncated"]) and terminal_observation_mode == "state_only_if_no_bootstrap":
                    row["terminal_observation_omitted"] = True
                rows.append(row)
            self.timings["step_seconds"] += time.perf_counter()-started
            self._notify("after_step", rows)
            return rows
        except BaseException as exc:
            self._fault = f"step: {type(exc).__name__}: {exc}"
            raise

    def step_and_reset(self, items, *, reset_seed, reset_options, max_active_envs):
        self._check()
        ids, _ = self._commands(items)
        if type(max_active_envs) is not int or not 0 <= max_active_envs <= self.num_envs:
            raise ValueError("max_active_envs must be an integer in [0,num_envs]")
        specs = self._prepare_reset(ids, reset_seed, reset_options)
        started = time.perf_counter()
        rows = self._step(items, terminal_observation_mode="state_only_if_no_bootstrap", defer_live_images=True)
        done = [r["env_id"] for r in rows if r["terminated"] or r["truncated"]]
        live = [r["env_id"] for r in rows if not (r["terminated"] or r["truncated"])]
        reset_ids = done[:max(0, max_active_envs-len(live))]
        try:
            resets = self._reset(reset_ids, specs, capture_images=False) if reset_ids else []
            active = sorted(live+reset_ids)
            if active:
                frames = max(self.execution["observation_render_frames"], self.execution["reset_render_frames"] if reset_ids else 0)
                self.engine._reset_capture = bool(reset_ids)
                try:
                    observations = self._observations(active, render_frames=frames)
                finally:
                    self.engine._reset_capture = False
                for row in rows:
                    if row["env_id"] in live:
                        row["observation"] = observations[row["env_id"]]
                for row in resets:
                    row["observation"] = observations[row["env_id"]]
            self.timings["fused_steps"] += 1
            self.timings["fused_step_reset_seconds"] += time.perf_counter()-started
            return {"items": rows, "reset_items": resets}
        except BaseException as exc:
            self._fault = f"fused_reset: {type(exc).__name__}: {exc}"
            raise

    def close(self):
        if not self._closed:
            self._closed = True
            try:
                for observer in self._observers:
                    close = getattr(observer, "close", None)
                    if close is not None:
                        close()
            finally:
                if self.engine is not None:
                    self.engine.close()
