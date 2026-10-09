#!/usr/bin/env python3
"""Independently verify a live XYZ-only residual run without using a GPU.

The audit reads the immutable initial checkpoint, reconstructs fresh networks
from the experiment's frozen source, and checks the live process provenance.
It never loads a training replay or changes any learner state.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "outputs/online_rl_residual_on_noise_gamma0995_xyz002_400k_20261008"
REFERENCES = {
    "fullscale": ROOT / "outputs/online_rl_residual_on_noise_gamma0995_400k_20261007",
    "halfscale": ROOT / "outputs/online_rl_residual_on_noise_gamma0995_scale050_400k_20261008",
}
NOISE_CHECKPOINT_SHA = "e102342eb9791e90281759ea681ce37cce680f412cc22b5d65aabc38fae058c1"
NOISE_ACTOR_SHA = "96ee3f337913dc09d6a2c9d577da56d3a58a810681253cc6f9ec3c1a0094ce0a"
VLA_SHA = "2456b1fff5ef2d94a173b502d55b244a6b0810b92c6e39fc6b1527e1d6019312"


def read(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha256(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def strip_selected_fields(config):
    value = copy.deepcopy(config)
    value["learner"].pop("residual_scale", None)
    value["learner"].pop("residual_mode", None)
    return value


def canonical_noise(value):
    value = copy.deepcopy(value)
    # The new dataclass serializes the old policy's implicit pose9 default.
    value["config"].setdefault("residual_mode", "pose9")
    return value


def live_process(record):
    directory = Path("/proc") / str(record["pid"])
    fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
    argv = (directory / "cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
    # Read only the device-selection entry; never expose other environment data.
    device = next((part.split(b"=", 1)[1].decode() for part in
                   (directory / "environ").read_bytes().split(b"\0")
                   if part.startswith(b"CUDA_VISIBLE_DEVICES=")), None)
    return dict(pid=record["pid"], argv=argv, starttime_ticks=int(fields[19])), device


def audit(run):
    run = Path(run).resolve()
    report = dict(status="running", created_at=datetime.now(timezone.utc).isoformat(),
                  scope="CPU startup checkpoint and live execution provenance; not final training/evaluation results",
                  run=str(run), references={k: str(v.resolve()) for k, v in REFERENCES.items()}, checks={})

    def check(name, condition):
        report["checks"][name] = bool(condition)
        if not condition:
            raise AssertionError(name)

    try:
        # Import exclusively the source used by the recorded learner command.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        sys.path.insert(0, str(run / "source/src"))
        import torch
        from pressb.online_rl import learner as learner_module
        from pressb.online_rl.learner import LearnerConfig, SACLearner
        from pressb.online_rl.residual_on_noise import BudgetReplayBuffer
        check("audit_imports_frozen_experiment_source", Path(learner_module.__file__).resolve()
              == run / "source/src/pressb/online_rl/learner.py")
        check("audit_cuda_not_initialized_at_start", not torch.cuda.is_initialized())
        torch.set_num_threads(1)
        config = read(run / "train_config.json")
        manifest = read(run / "train/manifest.json")
        composition = read(run / "train/composition.json")
        plan = read(run / "run_plan.json")
        initial = torch.load(run / "train/initial.pt", map_location="cpu", weights_only=False)
        cfg = LearnerConfig(**initial["config"])
        fresh = SACLearner(cfg, device="cpu")
        configs = [config["learner"], manifest["config"]["learner"], manifest["learner_config"],
                   initial["config"], initial["extra"]["run_config"]["learner"]]
        check("xyz_mode_everywhere", all(c.get("residual_mode") == "xyz" for c in configs)
              and composition.get("residual_mode") == plan.get("residual_mode") == "xyz")
        check("xyz_scale_everywhere", all(list(c["residual_scale"]) == [.02] * 3 for c in configs)
              and composition.get("residual_scale") == plan.get("residual_scale") == [.02] * 3)
        check("gamma_0995_everywhere", all(c["single_gamma"] == .995 for c in
              [config, manifest["config"], initial["extra"]["run_config"], plan,
               manifest["identities"]["simulation"], manifest["simulation"]]))
        check("budget_400k_fresh_seed42", config["max_transitions"] == plan["training_transitions"]
              == composition["training_transition_budget"] == 400000
              and config["learning_start"] == 2000 and config["seed"] == cfg.seed == 42
              and plan["expected_updates"] == 398000)
        check("initial_updates_and_replay_zero", initial["updates"] == initial["extra"]["replay_size"]
              == composition["residual_updates_at_start"] == composition["replay_size_at_start"] == 0)
        check("initial_counters_zero", all(initial["extra"]["counters"][key] == 0 for key in
              ("transitions", "physical_ticks", "episodes", "successes", "updates")))
        check("from_scratch_metadata", initial["extra"]["initialization"]
              == composition["residual_initialization"] == "from scratch")
        sizes = {key: len(initial[key]["state"]) for key in
                 ("actor_optimizer", "q_optimizer", "alpha_optimizer")}
        check("all_optimizer_states_empty", all(size == 0 for size in sizes.values()))
        check("initial_temperature_matches_fresh", torch.equal(initial["log_alpha"], fresh.log_alpha.detach()))
        for name, module in (("actor", fresh.actor), ("qs", fresh.qs), ("q_targets", fresh.q_targets)):
            expected = module.state_dict()
            check(f"initial_{name}_exactly_matches_fresh_seed42", initial[name].keys() == expected.keys()
                  and all(torch.equal(initial[name][key], value) for key, value in expected.items()))
        initial_actor_sha = state_sha256(initial["actor"])
        check("initial_actor_record_matches_weights", initial_actor_sha == composition["residual_actor_sha256_at_start"])
        check("actor_true_output_21", initial["actor"]["mean.weight"].shape[0]
              == initial["actor"]["log_std.weight"].shape[0] == fresh.action_dim
              == composition["residual_action_dim"] == plan["residual_action_dim"] == 21)
        check("actor_conditions_on_obs_plus_all63_base_actions", cfg.obs_dim == 2106
              and initial["actor"]["net.0.weight"].shape[1] == cfg.obs_dim + 63)
        check("critic_conditions_on_all63_combined_actions", initial["qs"]["net.0.weight"].shape[1]
              == initial["q_targets"]["net.0.weight"].shape[1] == cfg.obs_dim + 63)
        replay = BudgetReplayBuffer(1, cfg.obs_dim, fresh.action_dim, seed=cfg.seed, max_transitions=400000)
        check("fresh_replay_action_width_21", replay.action_dim == 21 and replay.arrays["action"].shape == (1, 21))
        check("actual_replay_action_width_21", initial["extra"].get("replay_action_dim")
              == composition.get("replay_action_dim") == 21)
        check("target_entropy_adjusts_to_minus10_5", cfg.target_entropy is None and fresh.target_entropy == -10.5)
        check("actual_learner_target_entropy_minus10_5", initial["extra"].get("target_entropy")
              == composition.get("target_entropy") == -10.5)
        # Independently exercise the frozen learner's XYZ scatter and unchanged
        # raw rotation columns, using distinct values for all seven time steps.
        base = torch.arange(126, dtype=torch.float32).reshape(2, 63) / 64
        action = torch.linspace(-1., 1., 42).reshape(2, 21)
        combined = fresh._critic_action(action, base).reshape(2, 7, 9)
        expected_xyz = base.reshape(2, 7, 9)[..., :3] + action.reshape(2, 7, 3) * .02
        check("all21_xyz_components_scaled_correctly", torch.equal(combined[..., :3], expected_xyz))
        check("all42_rotation_components_exactly_unchanged", torch.equal(combined[..., 3:], base.reshape(2, 7, 9)[..., 3:]))
        check("no_residual_checkpoint_or_warmstart", manifest.get("checkpoint") is None
              and manifest.get("warmstart_actor") is None and plan.get("resume") is False
              and plan.get("warmstart_actor") is None)
        identities = manifest["identities"]
        check("initial_checkpoint_runtime_identities_match", initial["extra"]["identities"] == identities)
        check("base_and_noise_marked_frozen", manifest["base_parameters_updated"] is False
              and composition["base_parameters_updated"] is False and composition["noise_parameters_updated"] is False
              and manifest["inference"].get("frozen") is True)
        check("vla_checkpoint_sha_correct", identities["inference"]["checkpoint_sha256"]
              == manifest["inference"]["checkpoint_sha256"] == VLA_SHA)
        noise = composition["frozen_noise"]
        check("frozen_noise_actor_identity_correct", noise["actor_sha256"]
              == identities["frozen_initial_noise"]["actor_sha256"] == NOISE_ACTOR_SHA)
        check("frozen_noise_settings_unchanged", noise["noise_scale"] == 1.5 and noise["noise_steps"] == 1)
        check("frozen_noise_checkpoint_bytes_unchanged", noise["checkpoint_sha256"]
              == plan["noise_checkpoint"]["sha256"] == sha256(noise["checkpoint"]) == NOISE_CHECKPOINT_SHA)
        source_manifest = read(run / "source_manifest.json")
        check("frozen_learner_source_matches_manifest", sha256(learner_module.__file__)
              == source_manifest["src/pressb/online_rl/learner.py"]["sha256"]
              == manifest["sources"]["learner.py"])
        references = {}
        for label, path in REFERENCES.items():
            reference = read(path / "train/manifest.json")
            refcomposition = read(path / "train/composition.json")
            refconfig = read(path / "train_config.json")
            refactor = read(path / "train/freeze_verification.json")["residual_actor_sha256"]
            check(f"only_mode_and_scale_changed_vs_{label}", strip_selected_fields(config) == strip_selected_fields(refconfig))
            check(f"all_frozen_identities_match_{label}", identities == reference["identities"])
            check(f"frozen_noise_composition_matches_{label}", canonical_noise(noise)
                  == canonical_noise(refcomposition["frozen_noise"]))
            check(f"noise_gamma_transfer_matches_{label}", composition["frozen_noise_objective_transfer"]
                  == refcomposition["frozen_noise_objective_transfer"])
            check(f"actor_differs_from_trained_{label}", initial_actor_sha != refactor)
            references[label] = dict(trained_residual_actor_sha256=refactor)
        process_records = {}
        for name in ("train", "simulation", "inference"):
            record = read(run / f"{name}_process.json")
            current, device = live_process(record)
            check(f"live_{name}_pid_argv_starttime_match", current == record)
            argv = record["argv"]
            if name == "train":
                check("trainer_no_checkpoint_resume_warmstart_arguments", not
                      any(flag in ("--checkpoint", "--resume", "--warmstart-actor") for flag in argv))
                check("trainer_uses_own_frozen_source", str(run / "source/scripts/run_residual_on_noise.py") in argv)
                check("trainer_uses_selected_config", argv[argv.index("--config") + 1] == str(run / "train_config.json"))
                check("trainer_uses_frozen_noise_checkpoint", Path(argv[argv.index("--noise-checkpoint") + 1]).resolve()
                      == Path(noise["checkpoint"]).resolve())
                check("learner_gpu0", device == "0" and argv[argv.index("--device") + 1] == "cuda:0")
            elif name == "inference":
                check("inference_gpu1_batch64", device == "1" and argv[argv.index("--batch-size") + 1] == "64")
            else:
                check("simulation_gpu1_64env_gamma0995", argv[argv.index("--gpu") + 1] == "1"
                      and argv[argv.index("--num-envs") + 1] == "64"
                      and float(argv[argv.index("--single-gamma") + 1]) == .995)
            process_records[name] = dict(pid=record["pid"], starttime_ticks=record["starttime_ticks"], cuda_visible_devices=device)
        check("simulation_runtime_64env_gpu1", manifest["simulation"]["num_envs"] == 64
              and manifest["simulation"]["gpu"] == 1)
        check("audit_never_initialized_cuda", not torch.cuda.is_initialized())
        report.update(status="pass", single_gamma=.995, residual_mode="xyz", residual_scale=[.02] * 3,
                      actor_output_dim=21, actor_input_dim=cfg.obs_dim + 63, critic_action_dim=63,
                      replay_action_dim=replay.action_dim, target_entropy=fresh.target_entropy,
                      previous_pose9_target_entropy=-31.5,
                      entropy_note="Default entropy target follows the actual learned action dimension: -21/2, previously -63/2.",
                      optimizer_state_sizes=sizes, initial_residual_actor_sha256=initial_actor_sha,
                      initial_checkpoint_sha256=sha256(run / "train/initial.pt"),
                      frozen_noise_checkpoint_sha256=NOISE_CHECKPOINT_SHA, frozen_noise_actor_sha256=NOISE_ACTOR_SHA,
                      frozen_vla_checkpoint_sha256=VLA_SHA, references=references, live_processes=process_records,
                      cuda_initialized=False)
    except Exception as error:
        report.update(status="fail", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["check_count"] = len(report["checks"])
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        output = run / "startup_verification.json"
        if run.is_dir():
            temporary = output.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            temporary.replace(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    result = audit(args.run)
    print(json.dumps({key: result[key] for key in ("status", "check_count", "run", "initial_residual_actor_sha256")}, indent=2))


if __name__ == "__main__":
    main()
