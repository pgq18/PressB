"""Fresh residual SAC on actions decoded from an immutable learned noise policy."""
from __future__ import annotations

from dataclasses import asdict
import copy
import hashlib
import math
from pathlib import Path
import time

import numpy as np
import torch

from .fast_runner import FastOnlineRunner
from .learner import CHECKPOINT_VERSION, SOURCE_COMMIT, LearnerConfig, ReplayBuffer, _Actor
from .protocol import finite_array, initial_noise
from .runner import write_json


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def actor_sha256(actor):
    digest = hashlib.sha256()
    for name, value in sorted(actor.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _noise_objective_transfer(source, runtime):
    """An immutable actor can transfer across gamma, never physical contracts.

    Gamma changes only the transition reward/discount; it is not an actor or
    flow input. No source critic, optimizer or replay is restored here. The
    residual's own checkpoint validation remains strict, including gamma.
    """
    if not isinstance(source, dict) or not isinstance(runtime, dict):
        raise ValueError("Frozen noise checkpoint has invalid identities")
    source_sim, runtime_sim = source.get("simulation", {}), runtime.get("simulation", {})
    if not isinstance(source_sim, dict) or not isinstance(runtime_sim, dict):
        raise ValueError("Frozen noise checkpoint has invalid simulation identities")
    source_gamma, runtime_gamma = source_sim.get("single_gamma"), runtime_sim.get("single_gamma")
    for gamma in (source_gamma, runtime_gamma):
        if gamma is not None and (type(gamma) not in (int, float)
                                  or not math.isfinite(gamma) or not 0 < gamma <= 1):
            raise ValueError("Frozen noise transfer requires valid single_gamma in (0,1]")
    if source != runtime:
        if source_gamma is None or runtime_gamma is None:
            raise ValueError("Frozen noise checkpoint belongs to a different model or simulator contract")
        before, after = copy.deepcopy(source), copy.deepcopy(runtime)
        before["simulation"].pop("single_gamma")
        after["simulation"].pop("single_gamma")
        if before != after:
            raise ValueError("Frozen noise checkpoint belongs to a different model or simulator contract")
    return dict(source_single_gamma=source_gamma, runtime_single_gamma=runtime_gamma,
                changed=source_gamma != runtime_gamma)


class FrozenNoisePolicy:
    """Load only actor weights; never restore replay, optimizers or training RNG."""

    def __init__(self, checkpoint, *, device, identities, obs_dim):
        self.checkpoint = Path(checkpoint).resolve()
        self.checkpoint_sha256 = file_sha256(self.checkpoint)
        payload = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        if payload.get("version") != CHECKPOINT_VERSION or payload.get("source_commit") != SOURCE_COMMIT:
            raise ValueError("Unsupported frozen noise checkpoint identity")
        self.config = LearnerConfig(**payload["config"])
        if self.config.method != "initial_noise" or self.config.obs_dim != obs_dim:
            raise ValueError("Frozen noise checkpoint method/observation contract differs")
        self.objective_transfer = _noise_objective_transfer(
            payload.get("extra", {}).get("identities"), identities)
        self.device = torch.device(device)
        self.source_updates = payload["updates"]
        self.deterministic = payload.get("extra", {}).get("run_config", {}).get("deterministic_eval", False)
        if type(self.deterministic) is not bool:
            raise ValueError("Invalid frozen noise sampling configuration")
        # Construction consumes CPU randomness even though all weights are
        # immediately overwritten. Preserve the caller's initialization stream.
        with torch.random.fork_rng(devices=[]):
            self.actor = _Actor(obs_dim, self.config.action_dim, self.config)
        self.actor.load_state_dict(payload["actor"], strict=True)
        self.actor.to(self.device).eval().requires_grad_(False)
        self.initial_actor_sha256 = actor_sha256(self.actor)
        self.identity = dict(
            contract="frozen_noise_then_fresh_residual_v1",
            checkpoint_sha256=self.checkpoint_sha256,
            actor_sha256=self.initial_actor_sha256,
            noise_steps=self.config.noise_steps, noise_scale=self.config.noise_scale,
            deterministic=self.deterministic,
            seed_contract="(seed + episode_index*10000 + chunk_index) % (2**31-1) + 17000000",
            base_action_contract="raw learned-noise decoder pose9; cached once per observation",
        )

    @torch.no_grad()
    def act(self, vectors, seeds):
        vectors = finite_array(vectors, (len(seeds), self.config.obs_dim), "frozen noise observations")
        devices = ([self.device.index if self.device.index is not None else torch.cuda.current_device()]
                   if self.device.type == "cuda" else [])
        sampled = []
        for vector, seed in zip(vectors, seeds, strict=True):
            # Fork ONLY this forward pass. RPC waits outside this scope may
            # perform residual SAC updates whose RNG must never be rewound.
            with torch.random.fork_rng(devices=devices):
                torch.random.default_generator.manual_seed(int(seed) + 17000000)
                if self.device.type == "cuda":
                    with torch.cuda.device(self.device):
                        torch.cuda.manual_seed(int(seed) + 17000000)
                obs = torch.as_tensor(vector[None], device=self.device, dtype=torch.float32)
                action = (self.actor.deterministic(obs) if self.deterministic else self.actor.sample(obs)[0])
                sampled.append(action[0].cpu().numpy().copy())
        return finite_array(sampled, (len(seeds), self.config.action_dim), "frozen noise actions")

    def verify_frozen(self):
        digest = actor_sha256(self.actor)
        if digest != self.initial_actor_sha256 or any(p.requires_grad or p.grad is not None for p in self.actor.parameters()):
            raise RuntimeError("Frozen initial-noise actor was modified or received gradients")
        return digest


class BudgetReplayBuffer(ReplayBuffer):
    """Exclude episode-drain transitions beyond the explicit training budget."""

    def __init__(self, *args, max_transitions, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_transitions = max_transitions
        self.accepted_transitions = 0

    def add(self, *args, **kwargs):
        if self.accepted_transitions < self.max_transitions:
            super().add(*args, **kwargs)
            self.accepted_transitions += 1


class ResidualOnNoiseRunner(FastOnlineRunner):
    def __init__(self, config, *args, noise_checkpoint, **kwargs):
        if config.method != "action_residual":
            raise ValueError("Residual-on-noise requires method=action_residual")
        if kwargs.get("resume") or kwargs.get("warmstart_actor") is not None:
            raise ValueError("Residual-on-noise training must start from scratch; resume/warmstart forbidden")
        if config.mode == "train" and kwargs.get("checkpoint") is not None:
            raise ValueError("Fresh residual training cannot load a residual checkpoint")
        super().__init__(config, *args, **kwargs)
        self.noise_checkpoint = Path(noise_checkpoint).resolve()
        self.noise_policy = None
        self.training_budget_reached = False

    def _init_learning(self):
        # Verify the frozen policy against the original simulator/model
        # identity BEFORE adding composition provenance for the new residual.
        self.noise_policy = FrozenNoisePolicy(self.noise_checkpoint, device=self.device,
            identities=self.identities, obs_dim=self.obs_dim)
        self.identities = {**self.identities, "frozen_initial_noise": self.noise_policy.identity}
        super()._init_learning()
        if self.config.mode == "eval":
            requested = LearnerConfig(method=self.config.method, obs_dim=self.obs_dim,
                                      seed=self.config.seed, **self.config.learner)
            if (self.learner.config.residual_mode != requested.residual_mode
                    or self.learner.config.residual_scale != requested.residual_scale):
                raise ValueError("Residual evaluation checkpoint mode/scale differs from requested configuration")
        if self.config.mode == "train":
            self.replay = BudgetReplayBuffer(self.config.replay_capacity, self.obs_dim,
                self.learner.action_dim, seed=self.config.seed, max_transitions=self.config.max_transitions)
            if self.learner.updates != 0 or len(self.replay) or any(
                    optimizer.state for optimizer in (self.learner.actor_optimizer,
                                                       self.learner.q_optimizer, self.learner.alpha_optimizer)):
                raise RuntimeError("Residual learner did not start from a fresh initialization")
            self.learner.save(self.output / "initial.pt", extra=dict(
                identities=self.identities, initialization="from scratch", replay_size=0,
                replay_action_dim=self.replay.action_dim,
                target_entropy=self.learner.target_entropy,
                counters=self.health(), run_config=asdict(self.config)))
        else:
            for module in (self.learner.actor, self.learner.qs, self.learner.q_targets):
                module.eval().requires_grad_(False)
            self.learner.log_alpha.requires_grad_(False)
        write_json(self.output / "composition.json", dict(
            method="residual_trained_on_frozen_initial_noise", mode=self.config.mode,
            formula="a_noise=Flow(encode(obs),frozen_noise_actor(obs)); "
                    + ("a_final.xyz=a_noise.xyz+scale*residual_actor(obs,a_noise); "
                       "a_final.rotation=a_noise.rotation"
                       if self.learner.config.residual_mode == "xyz" else
                       "a_final=a_noise+scale*residual_actor(obs,a_noise)"),
            residual_mode=self.learner.config.residual_mode,
            residual_action_dim=self.learner.action_dim,
            replay_action_dim=self.replay.action_dim,
            target_entropy=self.learner.target_entropy,
            residual_scale=list(self.learner.config.residual_scale),
            frozen_noise={**self.noise_policy.identity, "checkpoint": str(self.noise_checkpoint),
                          "source_updates": self.noise_policy.source_updates,
                          "config": asdict(self.noise_policy.config)},
            frozen_noise_objective_transfer=self.noise_policy.objective_transfer,
            residual_initialization="from scratch" if self.config.mode == "train" else str(self.checkpoint),
            residual_actor_sha256_at_start=actor_sha256(self.learner.actor),
            residual_updates_at_start=self.learner.updates, replay_size_at_start=len(self.replay),
            training_transition_budget=self.config.max_transitions,
            budget_contract="only first max_transitions enter replay or earn updates; "
                            "active episodes then finish with fixed weights",
            base_parameters_updated=False, noise_parameters_updated=False,
        ))

    def _call(self, client, service, path, payload):
        if service == "inference" and path == "/encode":
            payload = {**payload, "include_base_actions": False}
        return super()._call(client, service, path, payload)

    def _encode(self, items):
        prepared = super()._encode(items)
        if not prepared:
            return prepared
        ids = sorted(prepared)
        before = time.monotonic()
        noise = self.noise_policy.act(np.stack([prepared[eid][2] for eid in ids]),
                                      [prepared[eid][1]["rollout_seed"] for eid in ids])
        self._time("frozen_noise_actor", time.monotonic() - before)
        cfg = self.noise_policy.config
        decoded = self.inference.call("/decode", {"items": [dict(
            context_id=prepared[eid][1]["context_id"],
            initial_noise=initial_noise(action, cfg.noise_steps, cfg.noise_scale).tolist())
            for eid, action in zip(ids, noise, strict=True)]})["items"]
        if len(decoded) != len(ids) or any(row["context_id"] != prepared[eid][1]["context_id"]
                                           for eid, row in zip(ids, decoded)):
            raise ValueError("Frozen noise decoder returned mismatched contexts")
        for eid, row in zip(ids, decoded, strict=True):
            encoded = prepared[eid][1]
            # Preserve raw rotation6D, and exact wire pose8 for zero residual.
            # The same cached action is used by replay's next_base_action and
            # the next real residual action; never resample in _actions.
            encoded["base_actions_pose9"] = finite_array(row["actions_pose9"], (7, 9), "noise base pose9")
            encoded["base_actions_pose8"] = finite_array(row["actions_pose8"], (7, 8), "noise base pose8")
        return prepared

    def _accrue_updates(self, before, after):
        if self.config.mode != "train":
            return
        limit = self.config.max_transitions
        super()._accrue_updates(min(before, limit), min(after, limit))
        if after >= limit and not self.training_budget_reached:
            self._flush_updates()
            self.training_budget_reached = True
            self._append("events.jsonl", dict(event="training_budget_reached",
                training_transitions=limit, updates=self.learner.updates,
                residual_actor_sha256=actor_sha256(self.learner.actor),
                frozen_noise_actor_sha256=self.noise_policy.verify_frozen()))

    def _refresh_status(self):
        super()._refresh_status()
        if self.config.mode == "train":
            count = self.status["transitions"]
            self._set_status(training_transitions=min(count, self.config.max_transitions),
                             drain_transitions=max(0, count - self.config.max_transitions),
                             training_transition_budget=self.config.max_transitions)

    def _save(self, name):
        self.noise_policy.verify_frozen()
        super()._save(name)

    def run(self):
        result = super().run()
        unchanged = file_sha256(self.noise_checkpoint) == self.noise_policy.checkpoint_sha256
        if not unchanged:
            raise RuntimeError("Frozen noise checkpoint file changed during the run")
        write_json(self.output / "freeze_verification.json", dict(
            noise_actor_sha256=self.noise_policy.verify_frozen(),
            residual_actor_sha256=actor_sha256(self.learner.actor),
            residual_updates=self.learner.updates,
            noise_checkpoint_unchanged=unchanged,
            noise_parameters_updated=False, base_parameters_updated=False))
        return result
