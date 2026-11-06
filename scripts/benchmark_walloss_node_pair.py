#!/usr/bin/env python3
"""Measure one native B64 node with fixed real inputs; save pairable arrays."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pressb.rl_nodes.client import NodeClient
from benchmark_walloss_batch import write_json


def run(args):
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    raw = args.observation.read_bytes()
    observations = json.loads(raw)
    if not isinstance(observations, list) or len(observations) != 64:
        raise ValueError("Expected exactly 64 independently saved observations")
    report = {"state": "connecting", "endpoint": args.endpoint,
              "observation_path": str(args.observation.resolve()),
              "observation_sha256": hashlib.sha256(raw).hexdigest(), "seed": args.seed,
              "batch_size": 64, "warmups": 1, "repeats": args.repeats, "rounds": []}
    write_json(args.output, report)
    client = NodeClient(args.endpoint, timeout=600, max_message_bytes=192 * 1024**2,
                        image_transport="auto", preprocess_images=True, image_workers=8)
    contexts = []
    try:
        report["health_before"] = client.call("/health", {})
        health = report["health_before"]
        assert health["microbatch_size"] == 64
        assert health["noise"]["shape"] == [32, 20]
        assert health["num_inference_timesteps"] == 10
        assert health["diffusion_prefix_cache_mode"] == "full_attention"
        noise = np.random.default_rng(args.seed).standard_normal((64, 32, 20), dtype=np.float32)
        seeds = list(range(args.seed, args.seed + 64))
        reference = None
        for index in range(args.repeats + 1):
            row = {"index": index, "warmup": index == 0}
            start = time.perf_counter()
            encoded = client.call("/encode", {"observations": observations, "seeds": seeds})
            row["encode_seconds"] = time.perf_counter() - start
            row["encode_client_timing"] = dict(client.last_call_timing)
            row["encode_server_timing"] = encoded.get("timing", {})
            contexts = [item["context_id"] for item in encoded["items"]]
            features = np.asarray([item["features"] for item in encoded["items"]], dtype=np.float32)
            start = time.perf_counter()
            decoded = client.call("/decode", {"items": [
                {"context_id": context, "initial_noise": value.tolist()}
                for context, value in zip(contexts, noise)]})
            row["decode_seconds"] = time.perf_counter() - start
            row["decode_client_timing"] = dict(client.last_call_timing)
            row["decode_server_timing"] = decoded.get("timing", {})
            actions = np.asarray([item["actions"] for item in decoded["items"]], dtype=np.float32)
            if features.shape != (64, 2048) or actions.shape != (64, 32, 7):
                raise ValueError("Unexpected feature/action shape")
            if not np.isfinite(features).all() or not np.isfinite(actions).all():
                raise ValueError("Nonfinite features/actions")
            if reference is None:
                reference = (features.copy(), actions.copy())
            row["repeat_feature_max_abs"] = float(np.max(np.abs(features - reference[0])))
            row["repeat_action_max_abs"] = float(np.max(np.abs(actions - reference[1])))
            released = client.call("/release", {"context_ids": contexts})
            if released["released"] != 64:
                raise ValueError("Incomplete context release")
            contexts = []
            report["rounds"].append(row)
            write_json(args.output, report)
            print(json.dumps(row), flush=True)
        arrays = args.output.with_suffix(".npz")
        np.savez(arrays, features=features, actions=actions, initial_noise=noise,
                 seeds=np.asarray(seeds, dtype=np.int64),
                 states=np.asarray([row["state"] for row in observations], dtype=np.float32))
        steady = report["rounds"][1:]
        encode = float(np.mean([row["encode_seconds"] for row in steady]))
        decode = float(np.mean([row["decode_seconds"] for row in steady]))
        report.update(state="passed", array_path=str(arrays.resolve()),
                      array_sha256=hashlib.sha256(arrays.read_bytes()).hexdigest(),
                      mean_encode_seconds=encode, mean_decode_seconds=decode,
                      mean_total_seconds=encode + decode, chunks_per_second=64 / (encode + decode),
                      repeat_exact=all(row["repeat_feature_max_abs"] == row["repeat_action_max_abs"] == 0
                                       for row in report["rounds"]),
                      health_after=client.call("/health", {}))
        write_json(args.output, report)
        return 0
    except Exception as error:
        report.update(state="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        write_json(args.output, report)
        traceback.print_exc()
        return 1
    finally:
        if contexts:
            try:
                client.call("/release", {"context_ids": contexts})
            except Exception:
                pass
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--repeats", type=int, default=3)
    raise SystemExit(run(parser.parse_args()))
