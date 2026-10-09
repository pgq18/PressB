#!/usr/bin/env python3
"""Train with independent episode resets and overlapping SAC / network work."""
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

from pressb.online_rl.fast_runner import FastOnlineRunner, FastRunConfig
from pressb.online_rl.rpc import RPCServer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--simulation", default="http://127.0.0.1:19880")
    parser.add_argument("--inference", default="http://127.0.0.1:19881")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19882)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--warmstart-actor", type=Path,
                        help="Explicitly transfer only actor weights; initialize fresh critics, replay and counters")
    parser.add_argument("--mode", choices=("train", "eval"))
    parser.add_argument("--method", choices=("action_residual", "initial_noise", "base"))
    parser.add_argument("--max-transitions", type=int, help="Default 1,000,000, unless set in --config")
    parser.add_argument("--eval-episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--pipeline-updates", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--deterministic-eval", action="store_true", default=None)
    parser.add_argument("--token-env", default="PRESSB_RL_TOKEN")
    parser.add_argument("--timeout", type=float, default=180.)
    args = parser.parse_args()
    values = json.loads(args.config.read_text()) if args.config else {}
    for name in ("mode", "method", "max_transitions", "eval_episodes", "seed", "pipeline_updates", "deterministic_eval"):
        if getattr(args, name) is not None:
            values[name] = getattr(args, name)
    token = os.environ.get(args.token_env)
    runner = FastOnlineRunner(FastRunConfig(**values), args.simulation, args.inference, args.output,
        device=args.device, checkpoint=args.checkpoint, resume=args.resume,
        warmstart_actor=args.warmstart_actor, token=token, timeout=args.timeout)
    server = RPCServer((args.host, args.port),
        {"/health": runner.health, "/status": runner.health, "/stop": runner.stop}, token=token)
    thread = threading.Thread(target=server.serve_forever, name="fast-learner-status", daemon=True)
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
