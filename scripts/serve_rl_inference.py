#!/usr/bin/env python3
"""Serve frozen PiPER features and explicit-noise flow decoding on H200."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vla-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-vlm", type=Path,
                        help="Relocated Qwen3-VL config/tokenizer directory; preserves saved run metadata")
    parser.add_argument("--base-encoder", type=Path,
                        help="Relocated V-JEPA config/video processor directory; no base weights needed")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19871)
    parser.add_argument("--cache-size", type=int, default=256)
    parser.add_argument("--cache-ttl", type=float, default=600.0)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Maximum encoder/flow microbatch; 1 preserves the legacy serial path")
    parser.add_argument("--torch-threads", type=int, default=4,
                        help="CPU intra-op threads used by image preprocessing")
    parser.add_argument("--image-preprocess-device", choices=("cpu", "cuda"), default="cpu",
                        help="Saved image processor execution device; CUDA can change resize rounding")
    parser.add_argument("--token-env", default="PRESSB_RL_TOKEN")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be in 1..65535")
    if args.cache_size < 1 or not math.isfinite(args.cache_ttl) or args.cache_ttl <= 0:
        parser.error("cache-size and cache-ttl must be positive and finite")
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    if args.torch_threads < 1:
        parser.error("torch-threads must be positive")
    import torch
    torch.set_num_threads(args.torch_threads)
    from pressb.online_rl.inference import FrozenPiperBackend, InferenceApplication
    from pressb.online_rl.rpc import RPCServer
    backend = FrozenPiperBackend.load(args.vla_repo, args.checkpoint, device=args.device,
                                      image_preprocess_device=args.image_preprocess_device,
                                      base_vlm=args.base_vlm, base_encoder=args.base_encoder)
    application = InferenceApplication(backend, cache_size=args.cache_size, cache_ttl=args.cache_ttl,
                                       batch_size=args.batch_size)
    token = os.environ.get(args.token_env) if args.token_env else None
    with RPCServer((args.host, args.port), application.handlers, token=token) as server:
        print(json.dumps(dict(event="ready", address=f"http://{args.host}:{args.port}",
                              **application.health()), allow_nan=False), flush=True)
        try:
            server.serve_forever(poll_interval=.25)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
