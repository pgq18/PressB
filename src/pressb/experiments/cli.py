"""Launch configured inference, simulation, real robot, training and evaluation nodes."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import signal

from .config import load_experiment, materialize, PROJECT_ROOT
from pressb.registry import available


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("list", "validate", "prepare", "inference", "simulation", "real_robot", "train", "eval", "collect"))
    parser.add_argument("--config", type=Path, help="Declarative experiment JSON")
    parser.add_argument("--resolved-dir", type=Path, help="Output directory for immutable resolved node configs")
    parser.add_argument("--output", type=Path, help="Runner output directory")
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int, help="Runner seed; use matching held-out seeds for before/after evaluation")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--max-transitions", type=int)
    parser.add_argument("--allow-transition-limit", action="store_true",
                        help="For eval diagnostics, accept a normally completed transition-limited run")
    parser.add_argument("--resume", help="Full RL checkpoint path, or DPPO checkpoint-sink HTTP(S) URL")
    parser.add_argument("--inference", help="Override runner inference endpoint")
    parser.add_argument("--simulation", "--environment", dest="simulation",
                        help="Override runner environment endpoint (simulation or real robot)")
    parser.add_argument("--token-env", default="PRESSB_RL_TOKEN")
    args = parser.parse_args(argv)
    if args.allow_transition_limit and args.command != "eval":
        parser.error("--allow-transition-limit is supported only by eval")
    if args.command == "list":
        print(json.dumps(available(), indent=2))
        return 0
    if not args.config:
        parser.error("--config is required")
    experiment = load_experiment(args.config)
    run = experiment["nodes"]["trainer"]
    if args.output is not None:
        run["output_dir"] = str(args.output.expanduser().resolve())
    if args.device:
        run["device"] = args.device
    if args.seed is not None:
        run["seed"] = args.seed
    for key in ("inference", "simulation"):
        value = getattr(args, key)
        if value:
            run.setdefault("endpoints", {})[key] = value
    if args.episodes is not None:
        run.setdefault("rollout", {})["eval_episodes"] = args.episodes
    if args.max_transitions is not None:
        run.setdefault("rollout", {})["max_transitions"] = args.max_transitions
    if args.resume:
        remote = args.resume.startswith(("http://", "https://"))
        if remote and run.get("method") != "dppo":
            parser.error("HTTP(S) checkpoint resume is currently supported by DPPO only")
        run.setdefault("checkpoint", {})["resume"] = args.resume if remote else str(Path(args.resume).expanduser().resolve())
    if args.command in ("eval", "collect"):
        run["mode"] = "eval"
        rollout = run.setdefault("rollout", {})
        if args.command == "collect":
            rollout.update(terminal_observation_mode="full", fused_step_reset=False)
        else:
            rollout.setdefault("terminal_observation_mode", "full")
            rollout.setdefault("fused_step_reset", False)
    elif args.command == "train":
        run["mode"] = "train"
    # Validate before writing resolved configs or launching GPU processes. The
    # service-only environments don't need torch/training dependencies to start.
    if args.command in ("validate", "prepare", "train", "eval", "collect"):
        if run.get("method") == "dppo":
            from pressb.runtime.dppo_trainer import DPPOTrainerConfig as TrainerConfig
        else:
            from pressb.runtime.trainer import TrainerConfig
        TrainerConfig(**deepcopy(run))
    if args.command == "validate":
        print(json.dumps({"valid": True, "name": experiment["name"], "factories": experiment["factories"],
                          "nodes": experiment["nodes"], "sources": experiment["sources"]}, indent=2))
        return 0
    resolved = args.resolved_dir or PROJECT_ROOT / "outputs" / experiment["name"] / "resolved" / args.command
    paths = materialize(experiment, resolved)
    if args.command == "prepare":
        print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2))
        return 0
    if args.command in ("inference", "simulation", "real_robot"):
        from pressb.runtime.server import run_service
        # Physical environments keep the same config and endpoint keys as
        # simulators, so existing trainers and trajectory collectors work.
        node_key = "simulation" if args.command == "real_robot" else args.command
        run_service(args.command, paths[node_key])
        return 0
    if run.get("method") == "dppo":
        from pressb.runtime.dppo_trainer import DPPOTrainer as GenericTrainer
    else:
        from pressb.runtime.trainer import GenericTrainer
    if os.environ.get(args.token_env):
        run.setdefault("endpoints", {})["token"] = os.environ[args.token_env]
    observer = None
    if args.command == "collect":
        from pressb.runtime.trajectory import ChunkTrajectoryWriter
        collection = experiment["collection"]
        destination = collection.get("directory", str(Path(run["output_dir"]).with_name(Path(run["output_dir"]).name + "_trajectories")))
        # The trainer requires a fresh run dir. Never silently relocate an explicit dataset path.
        if Path(destination).resolve().is_relative_to(Path(run["output_dir"]).resolve()):
            raise ValueError("collection.directory must be outside run.output_dir")
        observer = ChunkTrajectoryWriter(destination, include_images=collection.get("include_images", True))
    try:
        runner = GenericTrainer(run, observer=observer)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: runner.stop())
        result = runner.run()
    except BaseException:
        if observer is not None and not observer.closed:
            observer.close({"state": "failed", "phase": "runner_setup_or_execution"})
        raise
    print(json.dumps(result, indent=2))
    accepted = result["state"] in ("completed", "stopped")
    if args.allow_transition_limit and result["state"] == "transition_limit":
        accepted = True
    return 0 if accepted else 1
