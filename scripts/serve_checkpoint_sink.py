#!/usr/bin/env python3
"""Receive checkpoints on a storage host; safe to expose over an SSH tunnel."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pressb.runtime.checkpoint_sink import CheckpointSinkServer, MAX_CHECKPOINT_BYTES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18391)
    parser.add_argument("--max-bytes", type=int, default=MAX_CHECKPOINT_BYTES)
    parser.add_argument("--timeout", type=float, default=1800.)
    parser.add_argument("--token-env", default="PRESSB_CHECKPOINT_TOKEN")
    args = parser.parse_args()
    with CheckpointSinkServer((args.host, args.port), args.directory, token=os.environ.get(args.token_env),
                              max_bytes=args.max_bytes, timeout=args.timeout) as server:
        print(json.dumps({"ready": True, "address": server.server_address, "directory": str(server.directory)}), flush=True)
        try:
            server.serve_forever(poll_interval=.2)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
