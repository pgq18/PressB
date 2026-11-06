"""One entry point for configured model, simulation, training and evaluation nodes."""
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
    parser.add_argument("command", choices=("list", "validate", "prepare", "inference", "simulation", "train", "eval", "collect"))
    parser.add_argument("--config", type=Path, help="Declarative experiment JSON")
    parser.add_argument("--resolved-dir", type=Path, help="Output directory for immutable resolved node configs")
    parser.add_argument("--output", type=Path, help="Runner output directory")
    parser.add_argument("--device")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--max-transitions", type=int)
    parser.add_argument("--resume", type=Path, help="Trusted full RL checkpoint, with optimizer/replay")
    parser.add_argument("--inference", help="Override runner inference endpoint")
    parser.add_argument("--simulation", help="Override runner simulation endpoint")
    parser.add_argument("--token-env", default="PRESSB_RL_TOKEN")
    args = parser.parse_args(argv)
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
    for key in ("inference", "simulation"):
        value = getattr(args, key)
        if value:
            run.setdefault("endpoints", {})[key] = value
    if args.episodes is not None:
        run.setdefault("rollout", {})["eval_episodes"] = args.episodes
    if args.max_transitions is not None:
        run.setdefault("rollout", {})["max_transitions"] = args.max_transitions
    if args.resume:
        run.setdefault("checkpoint", {})["resume"] = str(args.resume.expanduser().resolve())
    if args.command in ("eval", "collect"):
        run["mode"] = "eval"
        run.setdefault("rollout", {}).update(terminal_observation_mode="full", fused_step_reset=False)
    elif args.command == "train":
        run["mode"] = "train"
    # Validate before writing resolved configs or launching GPU processes. The
    # service-only environments don't need torch/training dependencies to start.
    if args.command in ("validate", "prepare", "train", "eval", "collect"):
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
    if args.command in ("inference", "simulation"):
        from pressb.runtime.server import run_service
        run_service(args.command, paths[args.command])
        return 0
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
    return 0 if result["state"] in ("completed", "stopped") else 1
