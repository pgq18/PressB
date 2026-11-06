#!/usr/bin/env python3
"""Validate optional flow prefix reuse against the complete frozen transformer."""
import argparse
import copy
import json
from pathlib import Path
import time
import traceback

import numpy as np

from benchmark_walloss_batch import ROOT, action_errors, load_observations, make_observations, timed, write_json


def run(args):
    report = {"state": "loading", "started_unix": time.time(), "results": []}
    write_json(args.output, report)
    try:
        import torch
        from pressb.runtime.server import load_backend
        torch.cuda.set_device(torch.device(args.device))
        config = json.loads(args.config.read_text())
        spec = copy.deepcopy(config.get("backend", config))
        spec["config"]["device"] = args.device
        backend = load_backend(spec)
        device = torch.device(args.device)
        sources, files = load_observations([args.observation])
        observations = make_observations(sources, 64, True)
        noises = np.random.default_rng(20261104).standard_normal((64, 32, 20), dtype=np.float32)
        report.update(backend=backend.describe(), observation_files=files)
        for dtype, size in (("float32", 2), ("bfloat16", 64)):
            backend.policy.model.to(dtype=getattr(torch, dtype))
            backend.policy.model.flow_prefix_cache = False
            encoded = backend.encode_batch(observations[:size], list(range(size)))
            contexts = [row[0] for row in encoded]
            baseline = None
            for mode in ("disabled", "suffix", "full_attention"):
                enabled = mode != "disabled"
                name = f"{dtype}_b{size}_prefix_{mode}"
                report.update(state="benchmarking", current=name)
                write_json(args.output, report)
                backend.policy.model.flow_prefix_cache = enabled
                backend.policy.model.flow_prefix_cache_mode = mode if enabled else "suffix"
                samples = []
                torch.cuda.reset_peak_memory_stats(device)
                for repeat in range(args.repeats + 1):
                    actions, seconds = timed(torch, device, lambda: backend.decode_batch(contexts, noises[:size]))
                    if repeat:
                        samples.append(seconds)
                row = {"name": name, "dtype": dtype, "batch_size": size, "prefix_enabled": enabled,
                       "prefix_mode": mode,
                       "decode_samples_seconds": samples, "decode_mean_seconds": float(np.mean(samples)),
                       "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3}
                if baseline is None:
                    baseline = np.asarray(actions)
                else:
                    row["comparison"] = action_errors(actions, baseline)
                    repeated = backend.decode_batch(contexts, noises[:size])
                    row["repeat_comparison"] = action_errors(repeated, actions)
                    indices = list(reversed(range(size)))
                    reordered = backend.decode_batch([contexts[i] for i in indices], noises[indices])
                    row["reordered_comparison"] = action_errors(np.asarray(reordered)[indices], actions)
                report["results"].append(row)
                write_json(args.output, report)
                print(json.dumps(row), flush=True)
            del encoded, contexts
        report.update(state="completed", completed_unix=time.time())
        write_json(args.output, report)
        return 0
    except Exception as error:
        report.update(state="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        write_json(args.output, report)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=3)
    raise SystemExit(run(parser.parse_args()))
