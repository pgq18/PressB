"""Online rollout coordinator; only this node owns replay and trainable models."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import threading
import time
import uuid

import numpy as np

from .protocol import (finite_array, initial_noise, learner_observation,
                       pose9_to_pose8, residual_actions)
from .rpc import RPCClient

BEST_STEP = 10600
BEST_SHA = "2456b1fff5ef2d94a173b502d55b244a6b0810b92c6e39fc6b1527e1d6019312"


@dataclass
class RunConfig:
    method: str = "action_residual"
    mode: str = "train"
    seed: int = 42
    max_transitions: int = 100000
    eval_episodes: int = 60
    learning_start: int = 2000
    training_frequency: int = 1000
    utd: float = 1.
    progressive_exploration: int = 50000
    replay_capacity: int = 100000
    checkpoint_every: int = 10000
    single_gamma: float = .99
    expected_checkpoint_step: int = BEST_STEP
    expected_checkpoint_sha256: str = BEST_SHA
    deterministic_eval: bool = False
    learner: dict = field(default_factory=dict)

    def validate(self):
        if self.method not in ("action_residual", "initial_noise", "base") or self.mode not in ("train", "eval"):
            raise ValueError("Invalid method/mode")
        if self.method == "base" and self.mode != "eval":
            raise ValueError("The frozen baseline is evaluation-only")
        for name in ("max_transitions", "eval_episodes", "training_frequency", "replay_capacity", "checkpoint_every"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.learning_start < 0 or self.progressive_exploration < 0 or not np.isfinite(self.utd) or self.utd < 0:
            raise ValueError("Invalid learning/exploration schedule")
        if not 0 < self.single_gamma <= 1:
            raise ValueError("single_gamma must be in (0,1]")


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class OnlineRunner:
    def __init__(self, config, simulation, inference, output, *, device="cpu", checkpoint=None,
                 resume=False, token=None, timeout=180.):
        config.validate()
        self.config = config
        self.sim = RPCClient(simulation, timeout=timeout, token=token)
        self.inference = RPCClient(inference, timeout=timeout, token=token)
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.device, self.checkpoint, self.resume = device, checkpoint, resume
        if resume and (checkpoint is None or config.mode != "train"):
            raise ValueError("--resume requires a training checkpoint")
        if config.mode == "eval" and config.method != "base" and checkpoint is None:
            raise ValueError("Learned-policy evaluation requires --checkpoint")
        self.run_id = str(uuid.uuid4())
        self.stop_event = threading.Event()
        self.status_lock = threading.Lock()
        self.status = dict(service="learner", ready=False, state="initializing", run_id=self.run_id,
                           transitions=0, physical_ticks=0, episodes=0, successes=0, updates=0)
        self.rng = np.random.default_rng(config.seed)
        self.request_index = 0
        self.observation_index = 0
        self.scheduled_episodes = 0
        self.learner = None
        self.replay = None
        self.update_credit = 0.
        self.last_training_bucket = 0
        self.last_checkpoint_bucket = 0
        self.contexts = set()
        self.episode_seed_indices = {}
        self.episode_encode_counts = {}

    def health(self, _=None):
        with self.status_lock:
            return dict(self.status)

    def stop(self, _=None):
        self.stop_event.set()
        return {"accepted": True, "boundary": "finish current cohort, checkpoint, then exit"}

    def _set_status(self, **values):
        with self.status_lock:
            self.status.update(values)

    def _request_id(self):
        self.request_index += 1
        return f"{self.run_id}:{self.request_index}"

    def _append(self, filename, record):
        with (self.output / filename).open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")

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
        if not np.isclose(self.sim_health.get("single_gamma", -1), self.config.single_gamma, atol=1e-12, rtol=0):
            raise ValueError("Simulator discounted reward and learner gamma differ")
        self.num_envs = int(self.sim_health["num_envs"])
        self.obs_dim = 2048 + 9 + int(self.sim_health["control_state_dim"]) + 12
        if self.config.mode == "eval" and self.config.eval_episodes % self.num_envs:
            raise ValueError("eval_episodes must be divisible by num_envs for whole-cohort evaluation")
        self.identities = {
            "inference": {key: health.get(key) for key in ("checkpoint_step", "checkpoint_sha256", "feature_schema", "source_identity")},
            "simulation": {key: self.sim_health.get(key) for key in ("scene_sha256", "config_sha256", "collection_fingerprint",
                "robot_urdf_sha256", "smoothing_window", "control_state_dim", "max_seconds", "single_gamma")},
        }

    def _init_learning(self):
        if self.config.method == "base":
            return
        from .learner import LearnerConfig, ReplayBuffer, SACLearner
        requested = LearnerConfig(method=self.config.method, obs_dim=self.obs_dim, seed=self.config.seed,
                                  **self.config.learner)
        if self.checkpoint:
            self.learner = SACLearner.load(self.checkpoint, device=self.device)
            if self.learner.config.method != self.config.method or self.learner.config.obs_dim != self.obs_dim:
                raise ValueError("RL checkpoint method/observation contract differs")
            if self.learner.config.residual_mode != requested.residual_mode:
                raise ValueError("RL checkpoint residual_mode differs from requested action semantics")
            extra = self.learner.checkpoint_extra
            if extra.get("identities") != self.identities:
                raise ValueError("RL checkpoint belongs to a different frozen model or simulator contract")
            if self.resume and asdict(self.learner.config) != asdict(requested):
                raise ValueError("Resume learner configuration differs from checkpoint")
            if self.resume:
                prior = extra.get("run_config", {})
                schedule_fields = ("method", "seed", "learning_start", "training_frequency", "utd",
                                   "progressive_exploration", "single_gamma", "replay_capacity")
                if any(prior.get(key) != getattr(self.config, key) for key in schedule_fields):
                    raise ValueError("Resume sampling/update schedule differs from checkpoint")
        else:
            self.learner = SACLearner(requested, device=self.device)
        self.replay = ReplayBuffer(self.config.replay_capacity, self.obs_dim, self.learner.action_dim, seed=self.config.seed)
        if self.resume:
            extra = self.learner.checkpoint_extra
            if "replay" not in extra:
                raise ValueError("Checkpoint has no replay state for resume")
            self.replay.load_state_dict(extra["replay"])
            for key in ("transitions", "physical_ticks", "episodes", "successes"):
                self.status[key] = int(extra["counters"][key])
            self.status["control_steps"] = self.status["physical_ticks"] / 4
            self.status["updates"] = self.learner.updates
            self.rng.bit_generator.state = extra["rollout_rng"]
            self.observation_index = int(extra["observation_index"])
            self.scheduled_episodes = int(extra["scheduled_episodes"])
            self.update_credit = float(extra.get("update_credit", 0.))
            self.last_training_bucket = self.status["transitions"] // self.config.training_frequency
            self.last_checkpoint_bucket = self.status["transitions"] // self.config.checkpoint_every
            self._append("events.jsonl", {"event": "resume", "simulator": "fresh whole-cohort reset", "exact_physical_resume": False})

    def _schedule(self):
        b = self.sim_health["panel_bounds"]
        xmin, xmax = b["min_offset_x_m"], b["max_offset_x_m"]
        ymin, ymax = b["min_offset_y_m"], b["max_offset_y_m"]
        positions = [((xmin + xmax) / 2, (ymin + ymax) / 2), (xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)]
        result = []
        for _ in range(self.num_envs):
            index = self.scheduled_episodes
            floor = 24 + index % 12
            if self.config.mode == "eval":
                x, y = positions[(index // 12) % 5]
            else:
                x, y = self.rng.uniform(xmin, xmax), self.rng.uniform(ymin, ymax)
            result.append(dict(floor=floor, offset_x_m=float(x), offset_y_m=float(y)))
            self.scheduled_episodes += 1
        return result

    def _encode(self, items):
        # Each episode/chunk owns its seed. Other environments finishing early
        # cannot shift this episode's Gaussian draws in a paired evaluation.
        seeds = []
        for item in items:
            episode_id = item["episode_id"]
            index = self.episode_seed_indices[episode_id]
            count = self.episode_encode_counts.get(episode_id, 0)
            seeds.append(int((self.config.seed + index * 10000 + count) % (2**31 - 1)))
            self.episode_encode_counts[episode_id] = count + 1
            self.observation_index += 1
        result = self.inference.call("/encode", {"observations": [item["observation"] for item in items],
            "seeds": seeds, "include_base_actions": self.config.method in ("base", "action_residual")})["items"]
        # All contexts already exist remotely. Register the full batch before
        # validating any row so malformed responses cannot strand later rows.
        self.contexts.update(row["context_id"] for row in result
                             if isinstance(row, dict) and isinstance(row.get("context_id"), str))
        if len(result) != len(items):
            raise ValueError("Inference returned the wrong number of contexts")
        prepared = {}
        for item, encoded, seed in zip(items, result, seeds):
            encoded["rollout_seed"] = seed
            vector = learner_observation(encoded, item["observation"])
            if vector.shape != (self.obs_dim,):
                raise ValueError("Simulator control-state dimension changed during rollout")
            prepared[item["env_id"]] = (item, encoded, vector)
        return prepared

    def _release(self, ids):
        ids = list(ids)
        if ids:
            self.inference.call("/release", {"context_ids": ids})
            self.contexts.difference_update(ids)

    def _actions(self, active):
        ids = sorted(active)
        cfg = self.config
        if cfg.method == "base":
            return [{"env_id": eid, "actions_pose8": active[eid][1]["base_actions_pose8"]} for eid in ids], None
        vectors = np.stack([active[eid][2] for eid in ids])
        base = np.asarray([active[eid][1]["base_actions_pose9"] for eid in ids]) if cfg.method == "action_residual" else None
        progress = min(self.status["transitions"] / cfg.progressive_exploration, 1.) if cfg.progressive_exploration else 1.
        deterministic = cfg.mode == "eval" and (cfg.method == "action_residual" or cfg.deterministic_eval)
        if cfg.mode == "eval" and cfg.method == "initial_noise" and not deterministic:
            import torch
            device = self.learner.device
            devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
            sampled = []
            for eid, vector in zip(ids, vectors):
                # Stochastic evaluation still has independent reproducible
                # streams for each condition, irrespective of batch membership.
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(active[eid][1]["rollout_seed"] + 17000000)
                    sampled.append(self.learner.act(vector[None], deterministic=False)[0])
            actions = np.stack(sampled)
        else:
            actions = self.learner.act(vectors, base, deterministic=deterministic,
                warmup=cfg.mode == "train" and self.status["transitions"] < cfg.learning_start,
                exploration_probability=progress if cfg.mode == "train" else 1.)
        if cfg.method == "action_residual":
            # Warmup and progressive-exploration masks execute the original
            # wire action exactly, including float32 quaternion rounding.
            residual_mode = getattr(self.learner.config, "residual_mode", "pose9")
            pose8 = []
            for eid, b, a in zip(ids, base, actions):
                wire_base = active[eid][1]["base_actions_pose8"]
                if not np.any(a):
                    pose8.append(wire_base)
                    continue
                combined = residual_actions(b, a, self.learner.config.residual_scale,
                                            residual_mode=residual_mode)
                if residual_mode == "xyz":
                    # XYZ residuals must not reproject or recanonicalize the
                    # decoder's quaternion, or alter its gripper command.
                    value = finite_array(wire_base, (7, 8), "base action pose8").copy()
                    value[:, :3] = combined[:, :3]
                else:
                    value = pose9_to_pose8(combined)
                pose8.append(value)
        else:
            decoded = self.inference.call("/decode", {"items": [dict(context_id=active[eid][1]["context_id"],
                initial_noise=initial_noise(action, self.learner.config.noise_steps, self.learner.config.noise_scale).tolist())
                for eid, action in zip(ids, actions)]})["items"]
            if len(decoded) != len(ids) or any(d["context_id"] != active[eid][1]["context_id"] for eid, d in zip(ids, decoded)):
                raise ValueError("Noise decoder returned mismatched contexts")
            pose8 = [finite_array(item["actions_pose8"], (7, 8), "decoded action") for item in decoded]
        return [dict(env_id=eid, actions_pose8=np.asarray(value).tolist()) for eid, value in zip(ids, pose8)], dict(zip(ids, actions))

    def _train(self, added):
        cfg = self.config
        if cfg.mode != "train" or self.status["transitions"] < cfg.learning_start:
            return
        self.update_credit += added * cfg.utd
        bucket = self.status["transitions"] // cfg.training_frequency
        if bucket <= self.last_training_bucket or len(self.replay) < self.learner.config.batch_size:
            return
        self.last_training_bucket = bucket
        metrics = {}
        while self.update_credit >= 1:
            metrics = self.learner.update(self.replay.sample(self.learner.config.batch_size))
            self.update_credit -= 1
        self._set_status(updates=self.learner.updates)
        self._append("metrics.jsonl", {**self.health(), **metrics, "replay_size": len(self.replay)})

    def _save(self, name):
        if self.config.mode != "train" or self.learner is None:
            return
        extra = dict(identities=self.identities, run_config=asdict(self.config), counters=self.health(),
                     rollout_rng=self.rng.bit_generator.state, observation_index=self.observation_index,
                     scheduled_episodes=self.scheduled_episodes, update_credit=self.update_credit,
                     replay=self.replay.state_dict(), physical_resume="fresh cohort, no simulator snapshot")
        self.learner.save(self.output / name, extra=extra)

    def run(self):
        started = time.monotonic()
        cohort = None
        try:
            self._verify_nodes()
            self._init_learning()
            sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
            write_json(self.output / "manifest.json", dict(run_id=self.run_id, config=asdict(self.config),
                learner_config=asdict(self.learner.config) if self.learner else None,
                simulation=self.sim_health, inference=self.inference_health, sources=sources,
                identities=self.identities, device=self.device, checkpoint=str(self.checkpoint) if self.checkpoint else None,
                reward="sparse measured success; per-physical-tick discount", base_parameters_updated=False))
            self._set_status(ready=True, state="running")
            while True:
                if self.stop_event.is_set() or (self.config.mode == "train" and self.status["transitions"] >= self.config.max_transitions):
                    break
                if self.config.mode == "eval" and self.status["episodes"] >= self.config.eval_episodes:
                    break
                episode_specs = self._schedule()
                cohort = self.sim.call("/reset", dict(run_id=self.run_id, request_id=self._request_id(),
                    seed=self.config.seed + self.scheduled_episodes, episodes=episode_specs))
                self.episode_seed_indices = {item["episode_id"]: self.scheduled_episodes - self.num_envs + item["env_id"]
                                             for item in cohort["items"]}
                self.episode_encode_counts = {}
                active = self._encode(cohort["items"])
                while not cohort["all_done"]:
                    actions, rl_actions = self._actions(active)
                    result = self.sim.call("/step", dict(run_id=self.run_id, request_id=self._request_id(),
                        cohort_id=cohort["cohort_id"], step_id=cohort["step_id"], actions=actions))
                    if result["step_id"] != cohort["step_id"] + 1 or result["cohort_id"] != cohort["cohort_id"]:
                        raise ValueError("Simulator step/cohort sequence mismatch")
                    items = {item["env_id"]: item for item in result["items"]}
                    next_prepared = self._encode([items[eid] for eid in sorted(active)])
                    added, ticks, finished, succeeded = 0, 0, 0, 0
                    for eid in sorted(active):
                        previous, encoded, obs = active[eid]
                        item, next_encoded, next_obs = next_prepared[eid]
                        k = item["executed_physics_steps"]
                        if type(k) is not int or not 1 <= k <= 28:
                            raise ValueError("Active environment must execute between 1 and 28 physical ticks")
                        if not np.isclose(item["discount"], self.config.single_gamma ** (k / 4), rtol=0, atol=1e-10):
                            raise ValueError("Transition discount does not match actual physical duration")
                        terminated, truncated = item["terminated"], item["truncated"]
                        if type(terminated) is not bool or type(truncated) is not bool or terminated and truncated:
                            raise ValueError("Invalid terminal/truncation flags")
                        if self.config.mode == "train":
                            kwargs = {}
                            if self.config.method == "action_residual":
                                kwargs = dict(base_action=encoded["base_actions_pose9"], next_base_action=next_encoded["base_actions_pose9"])
                            self.replay.add(obs, rl_actions[eid], item["reward"], item["discount"], next_obs,
                                            terminated, truncated, **kwargs)
                        added += 1
                        ticks += k
                        if terminated or truncated:
                            finished += 1
                            succeeded += int(item["info"].get("success", False))
                            self._append("episodes.jsonl", {**item["info"], "env_id": eid, "episode_id": item["episode_id"],
                                "task": item["observation"]["task"], "layout": episode_specs[eid], "terminated": terminated,
                                "truncated": truncated})
                            self._release([next_encoded["context_id"]])
                    self._release([encoded["context_id"] for _, encoded, _ in active.values()])
                    active = {eid: value for eid, value in next_prepared.items() if not (value[0]["terminated"] or value[0]["truncated"])}
                    if result["all_done"] != (not active):
                        raise ValueError("Simulator all_done does not match its terminal observations")
                    cohort = result
                    self._set_status(transitions=self.status["transitions"] + added, physical_ticks=self.status["physical_ticks"] + ticks,
                        episodes=self.status["episodes"] + finished, successes=self.status["successes"] + succeeded,
                        control_steps=self.status.get("control_steps", 0.) + ticks / 4,
                        wall_seconds=time.monotonic() - started)
                    self._train(added)
                    checkpoint_bucket = self.status["transitions"] // self.config.checkpoint_every
                    if checkpoint_bucket > self.last_checkpoint_bucket:
                        self._save(f"step_{self.status['transitions']:08d}.pt")
                        self.last_checkpoint_bucket = checkpoint_bucket
                    write_json(self.output / "status.json", self.health())
                print(json.dumps({"event": "cohort_complete", **self.health()}, allow_nan=False), flush=True)
            self._set_status(state="stopped" if self.stop_event.is_set() else "complete", ready=False,
                success_rate=self.status["successes"] / max(self.status["episodes"], 1), wall_seconds=time.monotonic() - started)
            self._save("last.pt")
            write_json(self.output / "status.json", self.health())
            write_json(self.output / "summary.json", {**self.health(), "experiment_kind": self.config.mode,
                "claim": "Measured run only; short integration runs do not establish learning improvement"})
            return self.health()
        except BaseException as error:
            self._set_status(state="failed", ready=False, error=f"{type(error).__name__}: {error}")
            write_json(self.output / "status.json", self.health())
            raise
        finally:
            if self.contexts:
                try:
                    self._release(list(self.contexts))
                except Exception:
                    pass  # bounded inference cache eventually expires
            if cohort is not None and cohort.get("all_done"):
                try:
                    self.sim.call("/close_run", dict(run_id=self.run_id, request_id=self._request_id()))
                except Exception as error:
                    self._append("events.jsonl", {"event": "close_run_failed", "error": str(error)})
