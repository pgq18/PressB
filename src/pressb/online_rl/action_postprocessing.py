"""Evaluation-only, causal target-pose smoothing outside the learned policy.

The checkpoint/simulator identities stay intact. This module records a separate
inference-time override and never changes the residual actor or its inputs.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .protocol import finite_array
from .residual_on_noise import ResidualOnNoiseRunner, actor_sha256, file_sha256
from .runner import write_json


MODES = ("none", "xyz_ema", "pose_ema")
CONTROL_HZ = 30


def quaternion_slerp(previous, target, alpha):
    """Normalized wxyz SLERP along the shortest rotation arc."""
    a = finite_array(previous, (4,), "previous quaternion").copy()
    b = finite_array(target, (4,), "target quaternion").copy()
    norms = np.linalg.norm(a), np.linalg.norm(b)
    if min(norms) < 1e-12:
        raise ValueError("Cannot interpolate a zero quaternion")
    a /= norms[0]
    b /= norms[1]
    dot = float(np.dot(a, b))
    if dot < 0:
        b = -b
        dot = -dot
    dot = float(np.clip(dot, 0., 1.))
    if dot > .9995:
        result = (1. - alpha) * a + alpha * b
    else:
        angle = math.acos(dot)
        result = (math.sin((1. - alpha) * angle) * a
                  + math.sin(alpha * angle) * b) / math.sin(angle)
    return result / np.linalg.norm(result)


class CausalActionPostprocessor:
    """Keep one last sent target per simulator slot, reset by episode identity."""

    def __init__(self, mode="none", alpha=1.):
        if mode not in MODES:
            raise ValueError(f"smoothing mode must be one of {MODES}")
        if type(alpha) not in (int, float) or not math.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError("smoothing alpha must be finite and in (0,1]")
        self.mode, self.alpha = mode, float(alpha)
        self._slots = {}

    def apply(self, env_id, episode_id, raw_actions, observation_state):
        if type(env_id) is not int or env_id < 0 or not isinstance(episode_id, str) or not episode_id:
            raise ValueError("Expected a nonnegative environment ID and nonempty episode ID")
        raw = finite_array(raw_actions, (7, 8), "raw action pose8")
        observed = finite_array(observation_state, (8,), "observed state pose8")
        slot = self._slots.get(env_id)
        reset = slot is None or slot["episode_id"] != episode_id
        previous = observed.copy() if reset else slot["last_sent"].copy()
        previous_at_start = previous.copy()
        chunk_index = 0 if reset else slot["chunk_index"] + 1
        sent = raw.copy()
        if self.mode != "none":
            for row in sent:
                # Each target depends only on earlier sent targets and itself,
                # never on future elements of this seven-control chunk.
                row[:3] = self.alpha * row[:3] + (1. - self.alpha) * previous[:3]
                if self.mode == "pose_ema":
                    row[3:7] = quaternion_slerp(previous[3:7], row[3:7], self.alpha)
                previous = row.copy()
        self._slots[env_id] = dict(episode_id=episode_id, last_sent=sent[-1].copy(),
                                  chunk_index=chunk_index)
        return sent, dict(env_id=env_id, episode_id=episode_id, chunk_index=chunk_index,
            reset=reset, mode=self.mode, alpha=self.alpha, control_hz=CONTROL_HZ,
            observation_state_pose8=observed.tolist(), previous_sent_pose8=previous_at_start.tolist(),
            raw_actions_pose8=raw.tolist(), sent_actions_pose8=sent.tolist())


class SmoothedResidualOnNoiseRunner(ResidualOnNoiseRunner):
    """Frozen XYZ residual evaluation with an explicitly logged pose override."""

    def __init__(self, config, *args, smoothing_mode="none", smoothing_alpha=1., **kwargs):
        if config.mode != "eval":
            raise ValueError("Action postprocessing is evaluation-only; training is forbidden")
        if config.learner.get("residual_mode", "pose9") != "xyz":
            raise ValueError("Action postprocessing requires an XYZ residual checkpoint/configuration")
        self.postprocessor = CausalActionPostprocessor(smoothing_mode, smoothing_alpha)
        super().__init__(config, *args, **kwargs)
        self._postprocessing_manifest = None
        self._residual_hash_at_start = None
        self._updates_at_start = None

    def _init_learning(self):
        # Original strict checkpoint, model, noise, simulator and scale checks.
        super()._init_learning()
        if self.learner.config.residual_mode != "xyz":
            raise ValueError("Action postprocessing requires an XYZ residual checkpoint")
        self._residual_hash_at_start = actor_sha256(self.learner.actor)
        self._updates_at_start = self.learner.updates
        self._postprocessing_manifest = dict(
            contract="causal_pose8_ema_eval_override_v1", mode=self.postprocessor.mode,
            alpha=self.postprocessor.alpha, control_hz=CONTROL_HZ,
            position_rule="sent[t].xyz = alpha*raw[t].xyz + (1-alpha)*sent[t-1].xyz",
            orientation_rule=("shortest-arc normalized wxyz SLERP(sent[t-1],raw[t],alpha)"
                              if self.postprocessor.mode == "pose_ema" else "exactly unchanged"),
            gripper_rule="exactly unchanged", none_rule="original wire action values exactly unchanged",
            initialization="first measured observation.state per episode; last sent target across chunks",
            state_scope="environment slot plus episode identity; reset on new episode",
            applied_after="frozen learned-noise flow decode plus frozen XYZ action residual",
            inference_time_override=True, trained_simulation_identity_unchanged=True,
            training_or_checkpoint_modified=False, learned_parameters_updated=False,
            source_sha256={Path(__file__).name: file_sha256(__file__)},
            residual_actor_sha256_at_start=self._residual_hash_at_start,
            residual_updates_at_start=self._updates_at_start,
            residual_checkpoint=str(Path(self.checkpoint).resolve()),
            residual_scale=list(self.learner.config.residual_scale),
            frozen_noise_actor_sha256=self.noise_policy.initial_actor_sha256,
            note="Logged targets include all seven commands; terminal simulation may execute only a prefix.",
        )
        write_json(self.output / "postprocessing.json", self._postprocessing_manifest)

    def _actions(self, active):
        actions, rl_actions = super()._actions(active)
        processed = []
        for row in actions:
            eid = row["env_id"]
            item = active[eid][0]
            sent, record = self.postprocessor.apply(eid, item["episode_id"], row["actions_pose8"],
                                                     item["observation"]["state"])
            record["episode_index"] = self.episode_seed_indices[item["episode_id"]]
            self._append("action_postprocessing.jsonl", record)
            # Preserve original wire values and types in the baseline path.
            processed.append(row if self.postprocessor.mode == "none"
                             else {**row, "actions_pose8": sent.tolist()})
        return processed, rl_actions

    def run(self):
        result = super().run()
        digest = actor_sha256(self.learner.actor)
        if (digest != self._residual_hash_at_start or self.learner.updates != self._updates_at_start
                or result["updates"] != 0 or len(self.replay)
                or any(p.requires_grad or p.grad is not None for p in self.learner.actor.parameters())):
            raise RuntimeError("Evaluation modified residual parameters or performed learning")
        write_json(self.output / "postprocessing.json", {**self._postprocessing_manifest,
            "residual_actor_sha256_at_end": digest, "residual_updates_at_end": self.learner.updates,
            "evaluation_updates": result["updates"], "freeze_verification_passed": True})
        return result
