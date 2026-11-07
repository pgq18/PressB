#!/usr/bin/env python3
"""Inspect or explicitly execute a finite absolute-SI PiPER replay plan."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pressb.control.real_replay import ReplayPlan, execute_replay


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Enable physical robot and execute the validated plan.")
    parser.add_argument("--output", type=Path, help="Fresh directory for replay.jsonl and summary.json; required for execution.")
    parser.add_argument("--stop-file", type=Path, help="Creating this file cancels replay and requests a latched stop.")
    parser.add_argument("--execution-mode", choices=("reference", "tracked"),
                        help="Override dispatch mode. Default: plan's explicit mode, otherwise reference.")
    args = parser.parse_args(argv)
    data = json.loads(args.plan.read_text())
    replay = dict(data.get("replay", {}))
    replay["execution_mode"] = args.execution_mode or replay.get("execution_mode", "reference")
    data["replay"] = replay
    plan = ReplayPlan.from_mapping(data)
    if not args.execute:
        print(json.dumps({**plan.summary(), "executed": False}, indent=2))
        return 0
    if args.output is None:
        parser.error("--execute requires --output")
    # Importing the driver is deliberately deferred until explicit execution.
    from pressb.robots.piper.real import PiperHardware
    cancel = threading.Event()
    previous = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous[sig] = signal.signal(sig, lambda *_: cancel.set())
    try:
        result = execute_replay(plan, PiperHardware, args.output,
                                stop_file=args.stop_file, cancel=cancel)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
