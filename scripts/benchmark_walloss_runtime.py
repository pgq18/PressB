#!/usr/bin/env python3
"""Compare frozen Wall-OSS runtime options, leaving serving processes untouched.

This direct-model benchmark preserves BF16 weights, all ten flow steps and the
action horizon. Optional runtime changes are scoped to this process only.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import nullcontext
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from PIL import Image, __version__ as pillow_version

from benchmark_walloss_batch import (
    ROOT, action_errors, load_observations, make_observations, timed,
    timing_summary, write_json,
)


def pre_resize(observations):
    result = copy.deepcopy(observations)
    hashes = []
    for observation in result:
        for name, payload in observation["images"].items():
            image = Image.open(io.BytesIO(base64.b64decode(payload["data"]))).convert("RGB")
            width, height = image.size
            size = (256, int(256 * height / width)) if width > height else (int(256 * width / height), 256)
            resized = image.resize(size, resample=Image.Resampling.BICUBIC)
            hashes.append({"camera": name, "shape": list(np.asarray(resized).shape),
                           "sha256": hashlib.sha256(np.asarray(resized).tobytes()).hexdigest()})
            buffer = io.BytesIO()
            resized.save(buffer, format="PNG")
            payload["data"] = base64.b64encode(buffer.getvalue()).decode("ascii")
    return result, hashes


def run(args):
    report = {"state": "loading", "started_unix": time.time(), "results": [],
              "scope": "Direct frozen model; excludes simulation/network/SAC", "pillow": pillow_version}
    write_json(args.output, report)
    backend = None
    try:
        import torch
        from pressb.runtime.server import load_backend
        torch.cuda.set_device(torch.device(args.device))
        sources, files = load_observations([args.observation])
        observations = make_observations(sources, 128, True)
        config = json.loads(args.config.read_text())
        spec = copy.deepcopy(config.get("backend", config))
        spec["config"]["device"] = args.device
        backend = load_backend(spec)
        device = torch.device(args.device)
        report.update(backend=backend.describe(), files=files, torch=torch.__version__, device=str(device))
        resized, hashes = pre_resize(sources)
        native_inputs = backend.policy.prepare_observations([backend._prepare_observation(row) for row in sources])
        resized_inputs = backend.policy.prepare_observations([backend._prepare_observation(row) for row in resized])
        comparisons = {}
        for key in native_inputs:
            a, b = native_inputs[key], resized_inputs[key]
            comparisons[key] = {"equal": bool(torch.equal(a, b)), "shape": list(a.shape)} if torch.is_tensor(a) else {"equal": a == b}
        report["resize_equivalence"] = {"inputs": comparisons, "resized_rgb_hashes": hashes,
                                        "all_equal": all(row["equal"] for row in comparisons.values())}
        del native_inputs, resized_inputs
        write_json(args.output, report)
        noises = np.random.default_rng(20261104).standard_normal((128, 32, 20), dtype=np.float32)
        original_forward = backend.policy.model.model.forward
        references = {}
        def no_cache_forward(*forward_args, **kwargs):
            kwargs["use_cache"] = False
            return original_forward(*forward_args, **kwargs)
        for name, size, inference_mode, no_cache in (
            ("baseline_b64", 64, False, False),
            ("baseline_b128", 128, False, False),
            ("inference_mode_b64", 64, True, False),
            ("no_unused_kv_b64", 64, False, True),
            ("inference_mode_no_unused_kv_b64", 64, True, True),
        ):
            report.update(state="benchmarking", current=name)
            write_json(args.output, report)
            backend.policy.model.model.forward = no_cache_forward if no_cache else original_forward
            samples = []
            torch.cuda.reset_peak_memory_stats(device)
            with torch.inference_mode() if inference_mode else nullcontext():
                for repeat in range(args.repeats + 1):
                    encoded, encode_time = timed(torch, device, lambda: backend.encode_batch(observations[:size], list(range(size))))
                    actions, decode_time = timed(torch, device, lambda: backend.decode_batch([row[0] for row in encoded], noises[:size]))
                    features = np.stack([row[1] for row in encoded])
                    del encoded
                    if repeat:
                        samples.append({"encode_seconds": encode_time, "decode_seconds": decode_time})
            row = {"name": name, "batch_size": size, "timing": timing_summary(samples, size),
                   "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3}
            if size not in references:
                references[size] = (features, np.asarray(actions))
            else:
                ref_features, ref_actions = references[size]
                row["comparison"] = {"actions": action_errors(actions, ref_actions),
                                      "features_max_abs": float(np.max(np.abs(features - ref_features)))}
            report["results"].append(row)
            write_json(args.output, report)
            print(json.dumps(row), flush=True)
        backend.policy.model.model.forward = original_forward
        report.update(state="completed", completed_unix=time.time())
        write_json(args.output, report)
        return 0
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
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--repeats", type=int, default=3)
    raise SystemExit(run(parser.parse_args()))
