#!/usr/bin/env python3
"""Prepare paired PAP placement plans or summarize a running/completed evaluation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from pressb.pick_place.evaluation import effective_config, make_plan, plot_plan, write_json, write_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="Write a deterministic paired-position evaluation plan")
    prepare.add_argument("--config", type=Path, default=Path("configs/piper_pick_place.json"))
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--seed", type=int, default=20261023)
    prepare.add_argument("--episodes-per-group", type=int, default=100)
    summarize = subparsers.add_parser("summarize", help="Summarize all currently available episode reports")
    summarize.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        config = json.loads(args.config.read_text())
        plan = make_plan(config, args.seed, args.episodes_per_group)
        if (args.output/"plan.json").exists():
            existing = json.loads((args.output/"plan.json").read_text())
            if existing != plan:
                parser.error("An incompatible plan already exists in this output directory; use a new directory")
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output/"plan.json", plan)
        write_json(args.output/"effective_config.json", effective_config(config))
        plot_plan(plan, args.output/"placement_plan.png")
        print(json.dumps(dict(plan=str(args.output/"plan.json"), episodes=len(plan["episodes"]), region=plan["region"]), indent=2))
    else:
        summary = write_summary(args.output)
        plan = json.loads((args.output/"plan.json").read_text())
        plot_plan(plan, args.output/"success_by_position.png", summary)
        print(json.dumps(dict(complete=summary["complete"], completed=summary["completed"],
                              planned=summary["planned"], groups=summary["groups"]), indent=2))


if __name__ == "__main__":
    main()
