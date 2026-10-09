#!/usr/bin/env python3
"""Evaluate an immutable XYZ residual with causal target-pose smoothing."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pressb.online_rl.action_postprocessing import MODES, SmoothedResidualOnNoiseRunner
from pressb.online_rl.fast_runner import FastRunConfig
from pressb.online_rl.rpc import RPCServer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--noise-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--simulation", default="http://127.0.0.1:19880")
    parser.add_argument("--inference", default="http://127.0.0.1:19891")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, help="Optional status RPC server; omitted by default")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("eval",), default="eval")
    parser.add_argument("--eval-episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--timeout", type=float, default=180.)
    parser.add_argument("--smoothing-mode", choices=MODES, required=True)
    parser.add_argument("--smoothing-alpha", type=float, default=1.)
    args = parser.parse_args()
    values = json.loads(args.config.read_text())
    for name in ("mode", "eval_episodes", "seed"):
        if getattr(args, name) is not None:
            values[name] = getattr(args, name)
    token = os.environ.get("PRESSB_RL_TOKEN")
    runner = SmoothedResidualOnNoiseRunner(FastRunConfig(**values), args.simulation, args.inference,
        args.output, device=args.device, checkpoint=args.checkpoint, noise_checkpoint=args.noise_checkpoint,
        smoothing_mode=args.smoothing_mode, smoothing_alpha=args.smoothing_alpha,
        token=token, timeout=args.timeout)
    server = thread = None
    if args.port is not None:
        server = RPCServer((args.host, args.port),
            {"/health": runner.health, "/status": runner.health, "/stop": runner.stop}, token=token)
        thread = threading.Thread(target=server.serve_forever, name="smoothed-residual-status", daemon=True)
        thread.start()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: runner.stop())
    try:
        runner.run()
    finally:
        if server is not None:
            server.shutdown()
            thread.join()
            server.server_close()


if __name__ == "__main__":
    main()
