"""Indexed-reset rollout collection with main-thread SAC / RPC overlap.

The simulator still advances every active environment together for a chunk.
Episode boundaries are independent.  Only blocking HTTP runs in a worker;
all policy reads, replay writes and gradient updates stay on the calling thread.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
import base64
import binascii
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from .protocol import learner_observation
from .runner import OnlineRunner, RunConfig, write_json


RESET_MODE = "indexed_zero_step_cached_settled_state"
RUNNER_CONTRACT = "independent_episodes_pipeline_v1"


@dataclass
class FastRunConfig(RunConfig):
    max_transitions: int = 1_000_000
    training_frequency: int = 1
    checkpoint_every: int = 100_000
    rolling_checkpoint_every: int = 10_000
    pipeline_updates: bool = True
    status_interval_seconds: float = 5.
    metrics_every_updates: int = 256
    fused_context_release: bool = True

    def validate(self):
        super().validate()
        for name in ("pipeline_updates", "fused_context_release"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if not math.isfinite(self.status_interval_seconds) or self.status_interval_seconds <= 0:
            raise ValueError("status_interval_seconds must be positive and finite")
        if type(self.metrics_every_updates) is not int or self.metrics_every_updates < 1:
            raise ValueError("metrics_every_updates must be a positive integer")
        if type(self.rolling_checkpoint_every) is not int or self.rolling_checkpoint_every < 1:
            raise ValueError("rolling_checkpoint_every must be a positive integer")


class _ProfiledClient:
    def __init__(self, owner, client, service):
        self.owner, self.client, self.service = owner, client, service

    def call(self, path, payload=None):
        return self.owner._call(self.client, self.service, path, payload)


class FastOnlineRunner(OnlineRunner):
    def __init__(self, *args, warmstart_actor=None, **kwargs):
        if warmstart_actor is not None and (kwargs.get("checkpoint") is not None or kwargs.get("resume", False)):
            raise ValueError("--warmstart-actor cannot be combined with --checkpoint or --resume")
        super().__init__(*args, **kwargs)
        if warmstart_actor is not None and (self.config.mode != "train" or self.config.method == "base"):
            raise ValueError("Actor warmstart requires a learned-policy training run")
        self.warmstart_actor = Path(warmstart_actor).resolve() if warmstart_actor is not None else None
        self.warmstart_provenance = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rollout-http")
        self.sim = _ProfiledClient(self, self.sim, "simulation")
        self.inference = _ProfiledClient(self, self.inference, "inference")
        self.timings = {}
        self.rpc_counts = {}
        self.pending_releases = set()
        self.update_budget = 0
        self.earned_update_credit = 0.
        self.last_metrics_update = 0
        self.last_metrics = {}
        self.last_status_time = 0.
        self.last_rolling_bucket = 0
        self.last_progress_bucket = 0
        self.episode_specs = {}
        self.started = None
        self.session_start = {}
        self.draining = False
        self._cleanup = False

    def stop(self, _=None):
        self.stop_event.set()
        return dict(accepted=True, boundary="stop refilling, finish active episodes, checkpoint, then exit")

    def _time(self, name, seconds):
        self.timings[name] = self.timings.get(name, 0.) + seconds

    def _call(self, client, service, path, payload):
        name = service + path.replace("/", "_")

        def request():
            before = time.monotonic()
            return client.call(path, payload), time.monotonic() - before

        if not self.config.pipeline_updates or self._cleanup:
            result, duration = request()
        else:
            future = self.executor.submit(request)
            try:
                while not future.done():
                    if self._can_update():
                        self._one_update(overlapped=True)
                    else:
                        before = time.monotonic()
                        try:
                            future.result(timeout=.01)
                        except FutureTimeout:
                            pass
                        finally:
                            self._time("io_wait", time.monotonic() - before)
                result, duration = future.result()
            except BaseException:
                # A local gradient failure must not abandon contexts created by
                # an already-dispatched encode request. Transport errors still
                # have unknown execution state and are never auto-retried.
                try:
                    result, _ = future.result()
                    if service == "inference" and path == "/encode":
                        self._track_contexts(result.get("items", []))
                        released = (payload or {}).get("release_context_ids", [])
                        self.contexts.difference_update(released)
                        self.pending_releases.difference_update(released)
                except BaseException:
                    pass
                raise
        self._time(name, duration)
        self.rpc_counts[name] = self.rpc_counts.get(name, 0) + 1
        return result

    def _verify_nodes(self):
        self.sim_health = self.sim.call("/health")
        self.inference_health = self.inference.call("/health")
        for health, service in ((self.sim_health, "simulation"), (self.inference_health, "inference")):
            if health.get("service") != service or health.get("ready") is not True:
                raise ValueError(f"{service} node is not ready")
            if health.get("action_horizon") != 7:
                raise ValueError(f"{service} action horizon must be 7")
        health = self.inference_health
        if (health.get("checkpoint_step") != self.config.expected_checkpoint_step
                or health.get("checkpoint_sha256") != self.config.expected_checkpoint_sha256
                or health.get("checkpoint_verified") is not True):
            raise ValueError("Frozen model checkpoint step/SHA verification failed")
        if health.get("feature_dim") != 2048:
            raise ValueError("Expected mean-pooled 2048D VLA feature")
        if (self.sim_health.get("reset_mode") != RESET_MODE
                or not self.sim_health.get("simulation_contract")):
            raise ValueError("Fast runner requires an explicit indexed zero-step simulator contract")
        if not np.isclose(self.sim_health.get("single_gamma", -1), self.config.single_gamma, atol=1e-12, rtol=0):
            raise ValueError("Simulator discounted reward and learner gamma differ")
        self.num_envs = int(self.sim_health["num_envs"])
        if self.num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.obs_dim = 2048 + 9 + int(self.sim_health["control_state_dim"]) + 12
        if self.config.mode == "eval" and self.config.eval_episodes < self.num_envs:
            raise ValueError("eval_episodes must be at least num_envs for the initial reset")
        self.fused_release = self.config.fused_context_release and health.get("encode_release_context_ids") is True
        self.feature_encoding = ("float32_base64" if "float32_base64" in health.get("feature_encodings", []) else "json")
        self.identities = {
            "runner_contract": RUNNER_CONTRACT,
            "inference": {key: health.get(key) for key in
                          ("checkpoint_step", "checkpoint_sha256", "feature_schema", "source_identity",
                           "image_preprocess_device", "encoder_batch_mode")},
            "simulation": {key: self.sim_health.get(key) for key in (
                "scene_sha256", "config_sha256", "collection_fingerprint", "robot_urdf_sha256",
                "smoothing_window", "control_state_dim", "max_seconds", "single_gamma",
                "simulation_contract", "reset_mode", "controller_mode", "render_mode", "image_size",
                "image_width", "image_height", "render_interval", "physics_dt", "control_backend",
                "physics_backend", "ik_iterations", "image_resolution", "camera_contract",
                "rollout_render_interval_physics_steps", "renderer_settings")},
        }
        sources = self.sim_health.get("sources", {})
        if not isinstance(sources, dict):
            raise ValueError("Simulator source identities must be an object")
        # Source paths/byte counts are deployment metadata. Content hashes
        # prevent mixing replay after a controller implementation change even
        # when its human-readable contract name was not changed.
        self.identities["simulation"]["source_sha256"] = {
            name: identity.get("sha256") if isinstance(identity, dict) else identity
            for name, identity in sources.items()}

    def _init_learning(self):
        if self.warmstart_actor is None:
            super()._init_learning()
            if self.resume:
                extra = self.learner.checkpoint_extra
                self.warmstart_provenance = extra.get("warmstart_actor")
                profile = extra.get("fast_runner", {})
                if profile.get("contract") != RUNNER_CONTRACT:
                    raise ValueError("Checkpoint has no matching fast-runner scheduling contract")
                prior = extra.get("run_config", {})
                if any(prior.get(name) != getattr(self.config, name)
                       for name in ("checkpoint_every", "rolling_checkpoint_every")):
                    raise ValueError("Resume fast checkpoint schedule differs from checkpoint")
                self.update_budget = min(int(self.update_credit), int(profile.get("update_budget", 0)))
                self.last_rolling_bucket = self.status["transitions"] // self.config.rolling_checkpoint_every
                self.last_progress_bucket = self.status["transitions"] // 1000
                self._append("events.jsonl", dict(event="indexed_resume", exact_physical_resume=False,
                    simulator="fresh indexed episodes; matching fast simulation contract"))
            return
        import torch
        from .learner import CHECKPOINT_VERSION, SOURCE_COMMIT, LearnerConfig, ReplayBuffer, SACLearner
        payload = torch.load(self.warmstart_actor, map_location="cpu", weights_only=False)
        if payload.get("version") != CHECKPOINT_VERSION or payload.get("source_commit") != SOURCE_COMMIT:
            raise ValueError("Unsupported actor warmstart checkpoint identity")
        requested = LearnerConfig(method=self.config.method, obs_dim=self.obs_dim, seed=self.config.seed,
                                  **self.config.learner)
        previous = LearnerConfig(**payload["config"])
        actor_fields = ("method", "obs_dim", "hidden_dim", "noise_steps", "noise_scale",
                        "residual_mode", "residual_scale", "log_std_min", "log_std_max")
        if any(getattr(previous, name) != getattr(requested, name) for name in actor_fields):
            raise ValueError("Actor warmstart architecture or action semantics differ")
        old_inference = payload.get("extra", {}).get("identities", {}).get("inference", {})
        if any(old_inference.get(name) != self.identities["inference"].get(name)
               for name in ("checkpoint_step", "checkpoint_sha256", "feature_schema")):
            raise ValueError("Actor warmstart belongs to a different frozen model or feature schema")
        old_source, new_source = old_inference.get("source_identity"), self.identities["inference"].get("source_identity")
        same_source = (isinstance(old_source, dict) and isinstance(new_source, dict)
                       and isinstance(old_source.get("source_sha256"), dict)
                       and bool(old_source["source_sha256"])
                       and old_source["source_sha256"] == new_source.get("source_sha256"))
        if not same_source:
            raise ValueError("Actor warmstart core VLA source files differ")
        self.learner = SACLearner(requested, device=self.device)
        self.learner.actor.load_state_dict(payload["actor"], strict=True)
        self.replay = ReplayBuffer(self.config.replay_capacity, self.obs_dim, self.learner.action_dim,
                                   seed=self.config.seed)
        self.warmstart_provenance = dict(checkpoint=str(self.warmstart_actor),
            parent_counters=payload.get("extra", {}).get("counters", {}), parent_updates=payload["updates"],
            parent_identities=payload.get("extra", {}).get("identities"),
            restored="actor only", reset="critics, target critics, temperature, optimizers, replay, RNG, counters",
            reason="explicit actor transfer across simulation contracts; replay is not interchangeable")
        self._append("events.jsonl", dict(event="actor_warmstart", **self.warmstart_provenance))

    def _schedule_ids(self, ids):
        bounds = self.sim_health["panel_bounds"]
        xmin, xmax = bounds["min_offset_x_m"], bounds["max_offset_x_m"]
        ymin, ymax = bounds["min_offset_y_m"], bounds["max_offset_y_m"]
        positions = [((xmin + xmax) / 2, (ymin + ymax) / 2), (xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)]
        specs, seed_indices = [], {}
        for eid in ids:
            index = self.scheduled_episodes
            x, y = (positions[(index // 12) % 5] if self.config.mode == "eval"
                    else (self.rng.uniform(xmin, xmax), self.rng.uniform(ymin, ymax)))
            specs.append(dict(env_id=eid, floor=24 + index % 12, offset_x_m=float(x), offset_y_m=float(y)))
            seed_indices[eid] = index
            self.scheduled_episodes += 1
        return specs, seed_indices

    def _register_reset(self, items, specs, seed_indices, previous_ids=()):
        expected = {row["env_id"] for row in specs}
        if len(items) != len(expected) or {row["env_id"] for row in items} != expected:
            raise ValueError("Simulator reset returned the wrong environment IDs")
        old = set(previous_ids) | set(self.episode_seed_indices)
        seen = set()
        by_id = {row["env_id"]: row for row in specs}
        for item in items:
            eid, episode_id = item["env_id"], item["episode_id"]
            if not isinstance(episode_id, str) or not episode_id or episode_id in old or episode_id in seen:
                raise ValueError("Reset must allocate distinct new episode IDs")
            if item["terminated"] is not False or item["truncated"] is not False:
                raise ValueError("Reset observation must begin a live episode")
            seen.add(episode_id)
            self.episode_seed_indices[episode_id] = seed_indices[eid]
            self.episode_specs[episode_id] = {key: value for key, value in by_id[eid].items() if key != "env_id"}

    def _track_contexts(self, rows):
        self.contexts.update(row["context_id"] for row in rows
                             if isinstance(row, dict) and isinstance(row.get("context_id"), str))

    def _encode(self, items):
        if not items:
            return {}
        if self.pending_releases and not self.fused_release:
            self._release(list(self.pending_releases))
        seeds = []
        for item in items:
            episode_id = item["episode_id"]
            count = self.episode_encode_counts.get(episode_id, 0)
            seeds.append(int((self.config.seed + self.episode_seed_indices[episode_id] * 10000 + count) % (2**31 - 1)))
            self.episode_encode_counts[episode_id] = count + 1
            self.observation_index += 1
        release = sorted(self.pending_releases) if self.fused_release else []
        payload = dict(observations=[item["observation"] for item in items], seeds=seeds,
                       include_base_actions=self.config.method in ("base", "action_residual"))
        if release:
            payload["release_context_ids"] = release
        if self.feature_encoding != "json":
            payload["feature_encoding"] = self.feature_encoding
        result = self.inference.call("/encode", payload)["items"]
        self.contexts.difference_update(release)
        self.pending_releases.difference_update(release)
        self._track_contexts(result)
        if len(result) != len(items):
            raise ValueError("Inference returned the wrong number of contexts")
        prepared = {}
        for item, encoded, seed in zip(items, result, seeds, strict=True):
            if "feature_f32_b64" in encoded:
                if ("feature" in encoded or type(encoded.get("feature_dim")) is not int
                        or encoded["feature_dim"] != 2048 or not isinstance(encoded["feature_f32_b64"], str)):
                    raise ValueError("Packed encoder feature has an invalid dimension or ambiguous representation")
                try:
                    raw = base64.b64decode(encoded.pop("feature_f32_b64"), validate=True)
                except (ValueError, binascii.Error) as error:
                    raise ValueError("Packed encoder feature is not valid base64") from error
                if len(raw) != 2048 * 4:
                    raise ValueError("Packed encoder feature must contain exactly 2048 float32 values")
                feature = np.frombuffer(raw, dtype="<f4")
                if not np.isfinite(feature).all():
                    raise ValueError("Packed encoder feature must be finite")
                encoded["feature"] = feature
            encoded["rollout_seed"] = seed
            vector = learner_observation(encoded, item["observation"])
            if vector.shape != (self.obs_dim,):
                raise ValueError("Simulator control-state dimension changed during rollout")
            prepared[item["env_id"]] = (item, encoded, vector)
        return prepared

    def _release(self, ids):
        ids = list(ids)
        super()._release(ids)
        self.pending_releases.difference_update(ids)

    def _accrue_updates(self, before, after):
        if self.config.mode != "train":
            return
        # A vector batch can straddle warmup. Credit only genuinely postwarmup
        # transitions, rather than granting the entire crossing batch credit.
        eligible = max(0, after - self.config.learning_start) - max(0, before - self.config.learning_start)
        credit = eligible * self.config.utd
        self.update_credit += credit
        self.earned_update_credit += credit
        bucket = after // self.config.training_frequency
        if bucket > self.last_training_bucket:
            self.last_training_bucket = bucket
            self.update_budget = int(self.update_credit + 1e-9)
        if not self.config.pipeline_updates:
            while self._can_update():
                self._one_update()

    def _can_update(self):
        return (not self._cleanup and self.config.mode == "train" and self.learner is not None
                and self.update_budget > 0 and self.update_credit >= 1 - 1e-9
                and len(self.replay) >= self.learner.config.batch_size)

    def _one_update(self, *, overlapped=False):
        started = time.monotonic()
        self.last_metrics = self.learner.update(self.replay.sample(self.learner.config.batch_size))
        duration = time.monotonic() - started
        self._time("updates", duration)
        if overlapped:
            self._time("updates_during_rpc", duration)
        self.update_credit = max(0., self.update_credit - 1)
        self.update_budget -= 1
        self._set_status(updates=self.learner.updates, update_credit=self.update_credit)
        if self.learner.updates - self.last_metrics_update >= self.config.metrics_every_updates:
            self._write_metrics()

    def _flush_updates(self):
        self.update_budget = int(self.update_credit + 1e-9)
        while self._can_update():
            self._one_update()

    def _refresh_status(self):
        if self.started is None:
            return
        wall = time.monotonic() - self.started
        sampled = self.status["transitions"] - self.session_start.get("transitions", 0)
        updates = self.status["updates"] - self.session_start.get("updates", 0)
        self._set_status(wall_seconds=wall, session_transitions=sampled, session_updates=updates,
            transitions_per_second=sampled / max(wall, 1e-9), updates_per_second=updates / max(wall, 1e-9),
            update_credit=self.update_credit, earned_update_credit=self.earned_update_credit,
            runnable_update_budget=self.update_budget, timing_seconds=dict(self.timings), rpc_counts=dict(self.rpc_counts),
            draining=self.draining, success_rate=self.status["successes"] / max(self.status["episodes"], 1))

    def _write_metrics(self):
        self._refresh_status()
        if self.last_metrics:
            self._append("metrics.jsonl", {**self.health(), **self.last_metrics, "replay_size": len(self.replay)})
            self.last_metrics_update = self.learner.updates

    def _publish(self, *, force=False):
        now = time.monotonic()
        if force or now - self.last_status_time >= self.config.status_interval_seconds:
            self._refresh_status()
            write_json(self.output / "status.json", self.health())
            self.last_status_time = now

    def _save(self, name):
        if self.config.mode != "train" or self.learner is None:
            return
        self._flush_updates()
        self._refresh_status()
        extra = dict(identities=self.identities, run_config=asdict(self.config), counters=self.health(),
            rollout_rng=self.rng.bit_generator.state, observation_index=self.observation_index,
            scheduled_episodes=self.scheduled_episodes, update_credit=self.update_credit,
            replay=self.replay.state_dict(), physical_resume="fresh indexed episodes; no physical snapshot",
            warmstart_actor=self.warmstart_provenance,
            fast_runner=dict(contract=RUNNER_CONTRACT, update_budget=self.update_budget,
                timings=dict(self.timings), rpc_counts=dict(self.rpc_counts), pipeline_updates=self.config.pipeline_updates))
        started = time.monotonic()
        self.learner.save(self.output / name, extra=extra)
        self._time("checkpoints", time.monotonic() - started)

    def _validate_step(self, result, cohort, active):
        if result["step_id"] != cohort["step_id"] + 1 or result["cohort_id"] != cohort["cohort_id"]:
            raise ValueError("Simulator step/cohort sequence mismatch")
        items = {item["env_id"]: item for item in result["items"]}
        if len(items) != len(result["items"]) or not set(active) <= set(items):
            raise ValueError("Simulator returned duplicate or missing environment IDs")
        for eid, previous in active.items():
            item = items[eid]
            if item["episode_id"] != previous[0]["episode_id"]:
                raise ValueError("Simulator changed an episode ID without reset_envs")
            k = item["executed_physics_steps"]
            if type(k) is not int or not 1 <= k <= 28:
                raise ValueError("Active environment must execute between 1 and 28 physical ticks")
            if not np.isclose(item["discount"], self.config.single_gamma ** (k / 4), rtol=0, atol=1e-10):
                raise ValueError("Transition discount does not match actual physical duration")
            terminated, truncated = item["terminated"], item["truncated"]
            if type(terminated) is not bool or type(truncated) is not bool or terminated and truncated:
                raise ValueError("Invalid terminal/truncation flags")
        return items

    def run(self):
        self.started = time.monotonic()
        cohort = None
        try:
            self._verify_nodes()
            self._init_learning()
            self.session_start = {key: self.status[key] for key in ("transitions", "updates")}
            sources = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in Path(__file__).parent.glob("*.py")}
            write_json(self.output / "manifest.json", dict(run_id=self.run_id, config=asdict(self.config),
                learner_config=asdict(self.learner.config) if self.learner else None,
                simulation=self.sim_health, inference=self.inference_health, sources=sources,
                identities=self.identities, device=self.device, checkpoint=str(self.checkpoint) if self.checkpoint else None,
                warmstart_actor=self.warmstart_provenance, base_parameters_updated=False,
                feature_encoding=self.feature_encoding,
                transition_unit="one environment action chunk, at most 7 controls / 28 physical ticks",
                update_credit_rule="utd per transition strictly beyond learning_start; fractional credit retained",
                timing_note="RPC duration overlaps updates_during_rpc; timings must not be summed"))
            self._set_status(ready=True, state="running")
            active = {}
            if not self.stop_event.is_set() and not (self.config.mode == "train" and self.status["transitions"] >= self.config.max_transitions):
                specs, indices = self._schedule_ids(range(self.num_envs))
                cohort = self.sim.call("/reset", dict(run_id=self.run_id, request_id=self._request_id(),
                    seed=self.config.seed + self.scheduled_episodes,
                    episodes=[{k: v for k, v in row.items() if k != "env_id"} for row in specs]))
                self._register_reset(cohort["items"], specs, indices)
                active = self._encode(cohort["items"])
            while active:
                actions, rl_actions = self._actions(active)
                result = self.sim.call("/step", dict(run_id=self.run_id, request_id=self._request_id(),
                    cohort_id=cohort["cohort_id"], step_id=cohort["step_id"], actions=actions))
                items = self._validate_step(result, cohort, active)
                # Those remote contexts are no longer needed for flow. Their
                # CPU features/base actions stay available for replay insertion.
                self.pending_releases.update(value[1]["context_id"] for value in active.values())
                next_prepared = self._encode([items[eid] for eid in sorted(active)])
                ticks = finished = succeeded = 0
                ended = []
                for eid in sorted(active):
                    previous, encoded, obs = active[eid]
                    item, next_encoded, next_obs = next_prepared[eid]
                    if self.config.mode == "train":
                        kwargs = (dict(base_action=encoded["base_actions_pose9"], next_base_action=next_encoded["base_actions_pose9"])
                                  if self.config.method == "action_residual" else {})
                        self.replay.add(obs, rl_actions[eid], item["reward"], item["discount"], next_obs,
                                        item["terminated"], item["truncated"], **kwargs)
                    ticks += item["executed_physics_steps"]
                    if item["terminated"] or item["truncated"]:
                        ended.append(eid)
                        finished += 1
                        succeeded += int(item["info"].get("success", False))
                        self._append("episodes.jsonl", {**item["info"], "env_id": eid, "episode_id": item["episode_id"],
                            "task": item["observation"]["task"], "layout": self.episode_specs[item["episode_id"]],
                            "terminated": item["terminated"], "truncated": item["truncated"]})
                        self.pending_releases.add(next_encoded["context_id"])
                added = len(active)
                before = self.status["transitions"]
                active = {eid: value for eid, value in next_prepared.items() if eid not in ended}
                if result["all_done"] != (not active):
                    raise ValueError("Simulator all_done does not match terminal observations")
                cohort = result
                self._set_status(transitions=before + added, physical_ticks=self.status["physical_ticks"] + ticks,
                    control_steps=(self.status["physical_ticks"] + ticks) / 4,
                    episodes=self.status["episodes"] + finished, successes=self.status["successes"] + succeeded)
                self._accrue_updates(before, before + added)
                self.draining = (self.draining or self.stop_event.is_set()
                    or self.config.mode == "train" and self.status["transitions"] >= self.config.max_transitions)
                refill = [] if self.draining else ended
                if self.config.mode == "eval":
                    refill = refill[:max(0, self.config.eval_episodes - self.scheduled_episodes)]
                if refill:
                    specs, indices = self._schedule_ids(refill)
                    reset = self.sim.call("/reset_envs", dict(run_id=self.run_id, request_id=self._request_id(),
                        cohort_id=cohort["cohort_id"], step_id=cohort["step_id"],
                        seed=self.config.seed + self.scheduled_episodes, episodes=specs))
                    cohort = {**cohort, "all_done": False}
                    if reset["cohort_id"] != cohort["cohort_id"] or reset["step_id"] != cohort["step_id"]:
                        raise ValueError("Indexed reset changed the shared simulation sequence")
                    self._register_reset(reset["items"], specs, indices,
                                         previous_ids=[items[eid]["episode_id"] for eid in refill])
                    active.update(self._encode(reset["items"]))
                    cohort = {**cohort, "all_done": False}
                for eid in ended:
                    old_id = items[eid]["episode_id"]
                    self.episode_seed_indices.pop(old_id, None)
                    self.episode_encode_counts.pop(old_id, None)
                    self.episode_specs.pop(old_id, None)
                self._set_status(active_envs=len(active), scheduled_episodes=self.scheduled_episodes)
                bucket = self.status["transitions"] // self.config.checkpoint_every
                if bucket > self.last_checkpoint_bucket:
                    self._save(f"step_{self.status['transitions']:08d}.pt")
                    self.last_checkpoint_bucket = bucket
                rolling = self.status["transitions"] // self.config.rolling_checkpoint_every
                if rolling > self.last_rolling_bucket:
                    self._save("last.pt")
                    self.last_rolling_bucket = rolling
                self._publish(force=True)
                progress_bucket = self.status["transitions"] // 1000
                if progress_bucket > self.last_progress_bucket:
                    print(json.dumps({"event": "progress", **self.health()}, allow_nan=False), flush=True)
                    self.last_progress_bucket = progress_bucket
            self._flush_updates()
            self._set_status(state="stopped" if self.stop_event.is_set() else "complete", ready=False, active_envs=0)
            self._save("last.pt")
            self._write_metrics()
        except BaseException as error:
            self._set_status(state="failed", ready=False, error=f"{type(error).__name__}: {error}")
            raise
        finally:
            self._cleanup = True
            if self.contexts:
                try:
                    self._release(list(self.contexts))
                except Exception as error:
                    self._append("events.jsonl", dict(event="release_failed", error=str(error)))
            if cohort is not None and cohort.get("all_done"):
                try:
                    self.sim.call("/close_run", dict(run_id=self.run_id, request_id=self._request_id()))
                except Exception as error:
                    self._append("events.jsonl", dict(event="close_run_failed", error=str(error)))
            self.executor.shutdown(wait=True, cancel_futures=True)
            self._publish(force=True)
        write_json(self.output / "summary.json", {**self.health(), "experiment_kind": self.config.mode,
            "claim": "Measured run; training success is not a fixed-condition evaluation result"})
        return self.health()
