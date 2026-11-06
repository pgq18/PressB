#!/usr/bin/env python3
"""Check full-attention prefix reuse on independent saved PAP observations."""
import argparse
import base64
import copy
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

from benchmark_walloss_batch import (
    ROOT, action_errors, context_fingerprint, load_observations, timed, write_json,
)


def run(args):
    report = {"state": "loading", "started_unix": time.time(), "quality_checks_passed": False,
              "scope": "Fixed-batch frozen model numerical check; not closed-loop success evaluation"}
    write_json(args.output, report)
    backend = None
    try:
        import torch
        from pressb.runtime.server import load_backend
        device = torch.device(args.device)
        torch.cuda.set_device(device)
        observations, files = load_observations([args.observation])
        if len(observations) != 64:
            raise ValueError("This acceptance check requires exactly 64 supplied observations, without repetition")
        hashes = [{name: hashlib.sha256(base64.b64decode(image["data"])).hexdigest()
                   for name, image in row["images"].items()} for row in observations]
        if len({row["up"] for row in hashes}) != 64:
            raise ValueError("All 64 third-person image payloads must be distinct")
        config = json.loads(args.config.read_text())
        spec = copy.deepcopy(config.get("backend", config))
        spec["config"].update(device=args.device, dtype="bfloat16", flow_prefix_cache=False)
        backend = load_backend(spec)
        report.update(backend=backend.describe(), observation_files=files,
                      source_observation_count=64, repeated_source_rows=False,
                      unique_third_person_images=64, image_sha256=hashes,
                      prompts=sorted({row["task"] for row in observations}),
                      precision="bfloat16", batch_size=64,
                      flow_steps=backend.num_inference_timesteps, action_horizon=backend.horizon)
        if backend.num_inference_timesteps != 10 or backend.horizon != 32:
            raise ValueError("Expected unchanged 10-step flow and 32-action horizon")
        baseline_encoded, baseline_encode_s = timed(torch, device, lambda: backend.encode_batch(observations, list(range(64))))
        baseline_contexts = [row[0] for row in baseline_encoded]
        baseline_features = np.stack([row[1] for row in baseline_encoded])
        backend.policy.model.flow_prefix_cache = True
        backend.policy.model.flow_prefix_cache_mode = "full_attention"
        optimized_encoded, optimized_encode_s = timed(torch, device, lambda: backend.encode_batch(observations, list(range(64))))
        optimized_contexts = [row[0] for row in optimized_encoded]
        optimized_features = np.stack([row[1] for row in optimized_encoded])
        report["features"] = {"exact_equal": bool(np.array_equal(baseline_features, optimized_features)),
                              "max_abs": float(np.max(np.abs(baseline_features - optimized_features))),
                              "shape": list(baseline_features.shape)}
        report["encode_seconds"] = {"baseline": baseline_encode_s, "optimized": optimized_encode_s}
        fingerprint = context_fingerprint(optimized_contexts)
        rng = np.random.default_rng(20261104)
        shape = (64, 32, 20)
        noises = {"standard_normal": rng.standard_normal(shape, dtype=np.float32),
                  "rl_clipped_normal": np.clip(rng.standard_normal(shape, dtype=np.float32), -1.5, 1.5),
                  "rl_uniform_range": rng.uniform(-1.5, 1.5, shape).astype(np.float32)}
        report.update(state="checking", comparisons=[])
        write_json(args.output, report)
        for name, noise in noises.items():
            backend.policy.model.flow_prefix_cache = False
            baseline, baseline_s = timed(torch, device, lambda: backend.decode_batch(baseline_contexts, noise))
            backend.policy.model.flow_prefix_cache = True
            optimized, optimized_s = timed(torch, device, lambda: backend.decode_batch(optimized_contexts, noise))
            comparison = {"noise": name, "noise_min": float(noise.min()), "noise_max": float(noise.max()),
                          "noise_sha256": hashlib.sha256(noise.tobytes()).hexdigest(),
                          "exact_equal": bool(np.array_equal(baseline, optimized)),
                          "finite": bool(np.isfinite(baseline).all() and np.isfinite(optimized).all()),
                          "errors": action_errors(optimized, baseline),
                          "decode_seconds": {"baseline": baseline_s, "optimized": optimized_s}}
            report["comparisons"].append(comparison)
            write_json(args.output, report)
            print(json.dumps(comparison), flush=True)
        repeated = backend.decode_batch(optimized_contexts, noise)
        report["same_context_repeat_exact"] = bool(np.array_equal(repeated, optimized))
        indices = list(reversed(range(64)))
        reordered = backend.decode_batch([optimized_contexts[i] for i in indices], noise[indices])
        report["same_batch_reorder_exact"] = bool(np.array_equal(np.asarray(reordered)[indices], optimized))
        report["encoded_contexts_unchanged"] = fingerprint == context_fingerprint(optimized_contexts)
        report["quality_checks_passed"] = (report["features"]["exact_equal"]
            and all(row["exact_equal"] and row["finite"] for row in report["comparisons"])
            and report["same_context_repeat_exact"] and report["same_batch_reorder_exact"]
            and report["encoded_contexts_unchanged"])
        report.update(state="completed" if report["quality_checks_passed"] else "failed_validation", completed_unix=time.time())
        write_json(args.output, report)
        return 0 if report["quality_checks_passed"] else 1
    except Exception as error:
        report.update(state="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        write_json(args.output, report)
        traceback.print_exc()
        return 1
    finally:
        if backend is not None:
            backend.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    raise SystemExit(run(parser.parse_args()))
