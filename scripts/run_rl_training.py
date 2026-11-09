#!/usr/bin/env python3
"""Run DSRL, action residuals or DPPO against inference and environment nodes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--inference")
    parser.add_argument("--simulation")
    parser.add_argument("--resume", help="Full RL checkpoint path (or HTTP(S) URL for DPPO); simulation starts fresh episodes")
    parser.add_argument("--mode", choices=("train", "eval"))
    parser.add_argument("--max-transitions", type=int)
    parser.add_argument("--eval-episodes", type=int)
    parser.add_argument("--token-env", default="PRESSB_RL_TOKEN")
    args = parser.parse_args()
    data = json.loads(args.config.read_text())
    for name in ("device", "mode", "seed"):
        if getattr(args, name) is not None:
            data[name] = getattr(args, name)
    if args.output:
        data["output_dir"] = str(args.output)
    for name in ("inference", "simulation"):
        if getattr(args, name):
            data.setdefault("endpoints", {})[name] = getattr(args, name)
    if os.environ.get(args.token_env):
        data.setdefault("endpoints", {})["token"] = os.environ[args.token_env]
    if args.resume:
        data.setdefault("checkpoint", {})["resume"] = str(args.resume)
    for name in ("max_transitions", "eval_episodes"):
        if getattr(args, name) is not None:
            data.setdefault("rollout", {})[name] = getattr(args, name)
    if data.get("method") == "dppo":
        from pressb.runtime.dppo_trainer import DPPOTrainer as GenericTrainer, DPPOTrainerConfig as TrainerConfig
    else:
        from pressb.runtime.trainer import GenericTrainer, TrainerConfig
    runner = GenericTrainer(TrainerConfig(**data))
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: runner.stop())
    result = runner.run()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
