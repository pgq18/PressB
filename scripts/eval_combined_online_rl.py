#!/usr/bin/env python3
"""Evaluate frozen learned flow noise followed by a frozen action residual."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pressb.online_rl.fast_runner import FastOnlineRunner, FastRunConfig
from pressb.online_rl.learner import SACLearner
from pressb.online_rl.protocol import finite_array, pose9_to_pose8, residual_actions
from pressb.online_rl.runner import write_json


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def actor_sha256(learner):
    digest = hashlib.sha256()
    for name, value in sorted(learner.actor.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class CombinedEvaluationRunner(FastOnlineRunner):
    def __init__(self, *args, residual_checkpoint, **kwargs):
        super().__init__(*args, **kwargs)
        if self.config.mode != "eval" or self.config.method != "initial_noise":
            raise ValueError("Composition is evaluation-only and requires the initial-noise configuration")
        self.residual_checkpoint = Path(residual_checkpoint).resolve()
        self.residual_learner = None

    def _init_learning(self):
        super()._init_learning()
        self.residual_learner = SACLearner.load(self.residual_checkpoint, device=self.device)
        residual = self.residual_learner
        if residual.config.method != "action_residual" or residual.config.obs_dim != self.obs_dim:
            raise ValueError("Residual checkpoint method/observation contract differs")
        if residual.checkpoint_extra.get("identities") != self.identities:
            raise ValueError("Residual checkpoint belongs to a different frozen model or simulator contract")
        self.frozen_before = {}
        for name, learner, checkpoint in (
            ("initial_noise", self.learner, self.checkpoint),
            ("action_residual", residual, self.residual_checkpoint),
        ):
            for module in (learner.actor, learner.qs, learner.q_targets):
                module.eval().requires_grad_(False)
            learner.log_alpha.requires_grad_(False)
            self.frozen_before[name] = dict(actor_sha256=actor_sha256(learner), updates=learner.updates)
            self.frozen_before[name].update(checkpoint=str(checkpoint), checkpoint_sha256=file_sha256(checkpoint))
        write_json(self.output / "composition.json", dict(
            method="initial_noise_then_action_residual",
            formula="a_noise=Flow(encode(observation),noise_actor(observation)); "
                    "a_final=a_noise+residual_scale*residual_actor(observation,a_noise)",
            residual_conditioning="Raw actions_pose9 returned by learned-noise decode, before pose conversion",
            noise_sampling="deterministic" if self.config.deterministic_eval else "stochastic",
            noise_seed="(seed + condition_index*10000 + chunk_index) % (2**31-1) + 17000000",
            residual_sampling="deterministic tanh(mean)",
            training=False, checkpoints=self.frozen_before,
            residual_config=asdict(residual.config),
            caveat="Residual actor was trained with unmodulated base actions; its conditioning distribution changes.",
            evaluation=dict(episodes=self.config.eval_episodes, seed=self.config.seed,
                            layout_order="center, xmin/ymin, xmin/ymax, xmax/ymin, xmax/ymax",
                            task_order="24..35", condition_index="repeat*60 + layout_index*12 + floor-24"),
        ))

    def _actions(self, active):
        # Reuse the original noise actor's per-condition CUDA RNG and flow RPC.
        # Its normalized actions are returned for diagnostics; no replay is written.
        noise_rows, sampled_noise = super()._actions(active)
        ids = sorted(active)
        # OnlineRunner validates /decode context IDs. Capture its raw pose9
        # values through the narrow RPC wrapper below, avoiding pose8 roundtrips.
        decoded = self._last_noise_decode
        if [row["context_id"] for row in decoded] != [active[eid][1]["context_id"] for eid in ids]:
            raise ValueError("Captured noise decoder contexts do not match the active observations")
        base = np.stack([finite_array(row["actions_pose9"], (7, 9), "noise-decoded pose9") for row in decoded])
        vectors = np.stack([active[eid][2] for eid in ids])
        residuals = self.residual_learner.act(vectors, base, deterministic=True)
        rows = []
        for eid, action, residual, noise_row in zip(ids, base, residuals, noise_rows, strict=True):
            if noise_row["env_id"] != eid:
                raise ValueError("Noise action environment ordering changed")
            combined = residual_actions(action, residual, self.residual_learner.config.residual_scale)
            pose8 = (np.asarray(noise_row["actions_pose8"]) if not np.any(residual)
                     else pose9_to_pose8(combined))
            rows.append(dict(env_id=eid, actions_pose8=pose8.tolist()))
            item, encoded, _ = active[eid]
            self._append("policy_actions.jsonl", dict(
                episode_id=item["episode_id"], env_id=eid,
                condition_index=self.episode_seed_indices[item["episode_id"]],
                physics_index=item["info"]["physics_index"], rollout_seed=encoded["rollout_seed"],
                normalized_noise=np.asarray(sampled_noise[eid]).tolist(),
                noise_actions_pose9=action.tolist(), normalized_residual=residual.tolist(),
                combined_actions_pose9=combined.tolist(), actions_pose8=pose8.tolist(),
            ))
        return rows, sampled_noise

    def _call(self, client, service, path, payload):
        result = super()._call(client, service, path, payload)
        if service == "inference" and path == "/decode":
            self._last_noise_decode = result["items"]
        return result

    def _one_update(self, *args, **kwargs):
        raise RuntimeError("Parameter updates are forbidden in combined evaluation")

    def run(self):
        result = super().run()
        checks = {}
        for name, learner in (("initial_noise", self.learner), ("action_residual", self.residual_learner)):
            before = self.frozen_before[name]
            checks[name] = dict(actor_sha256_before=before["actor_sha256"],
                               actor_sha256_after=actor_sha256(learner),
                               updates_before=before["updates"], updates_after=learner.updates,
                               checkpoint_sha256_after=file_sha256(before["checkpoint"]))
            check = checks[name]
            check["unchanged"] = (check["actor_sha256_before"] == check["actor_sha256_after"]
                and check["updates_before"] == check["updates_after"]
                and before["checkpoint_sha256"] == check["checkpoint_sha256_after"])
        write_json(self.output / "frozen_verification.json", checks)
        if not all(check["unchanged"] for check in checks.values()) or result["updates"] != 0:
            raise RuntimeError("Frozen evaluation changed parameters/checkpoints or performed updates")
        summary = json.loads((self.output / "summary.json").read_text())
        summary.update(method="initial_noise_then_action_residual", frozen_verification=checks,
                       claim="Fixed-condition evaluation of two independently trained frozen actors in sequence")
        write_json(self.output / "summary.json", summary)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    old = ROOT / "outputs/online_rl_fast_20261003"
    parser.add_argument("--config", type=Path, default=old / "online_rl_fast_initial_noise.json")
    parser.add_argument("--noise-checkpoint", type=Path, default=old / "initial_noise_train/last.pt")
    parser.add_argument("--residual-checkpoint", type=Path, default=old / "action_residual_train/last.pt")
    parser.add_argument("--simulation", default="http://127.0.0.1:19880")
    parser.add_argument("--inference", default="http://127.0.0.1:19891")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    values = json.loads(args.config.read_text())
    values.update(method="initial_noise", mode="eval", eval_episodes=args.episodes, seed=args.seed)
    runner = CombinedEvaluationRunner(FastRunConfig(**values), args.simulation, args.inference,
        args.output, device=args.device, checkpoint=args.noise_checkpoint.resolve(),
        residual_checkpoint=args.residual_checkpoint.resolve())
    print(json.dumps(runner.run(), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
