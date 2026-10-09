#!/usr/bin/env python3
"""Train a fresh action residual on a frozen learned-noise VLA policy, or evaluate it."""
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

from pressb.online_rl.fast_runner import FastRunConfig
from pressb.online_rl.residual_on_noise import ResidualOnNoiseRunner
from pressb.online_rl.rpc import RPCServer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--noise-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Residual checkpoint; evaluation only")
    parser.add_argument("--simulation", default="http://127.0.0.1:19880")
    parser.add_argument("--inference", default="http://127.0.0.1:19891")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19882)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("train", "eval"))
    parser.add_argument("--eval-episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--timeout", type=float, default=180.)
    args = parser.parse_args()
    values = json.loads(args.config.read_text())
    for name in ("mode", "eval_episodes", "seed"):
        if getattr(args, name) is not None:
            values[name] = getattr(args, name)
    token = os.environ.get("PRESSB_RL_TOKEN")
    runner = ResidualOnNoiseRunner(FastRunConfig(**values), args.simulation, args.inference, args.output,
        device=args.device, checkpoint=args.checkpoint, noise_checkpoint=args.noise_checkpoint,
        token=token, timeout=args.timeout)
    server = RPCServer((args.host, args.port),
        {"/health": runner.health, "/status": runner.health, "/stop": runner.stop}, token=token)
    thread = threading.Thread(target=server.serve_forever, name="residual-on-noise-status", daemon=True)
    thread.start()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: runner.stop())
    try:
        runner.run()
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


if __name__ == "__main__":
    main()
