#!/usr/bin/env python3
"""Real-weight Wall-OSS batch benchmark; no simulator, service or subprocess launch.

Uses identical external flow noise for native-single and batched inference.
Times include CPU preprocessing / output copies and synchronized CUDA execution.
This measures inference, not simulation or end-to-end RL throughput. BF16 batch
kernels may change rounding; errors are reported in physical output units.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
GIB = 1024 ** 3


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def parse_batch_sizes(value):
    try:
        values = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("batch sizes must be comma-separated integers") from error
    if not values or any(item < 1 or item > 64 for item in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("batch sizes must be unique integers from 1 to 64")
    return values


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--observation", type=Path, action="append", required=True,
                        help="Repeat for diverse real observations; each file is an observation or list.")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--batch-sizes", type=parse_batch_sizes, default=parse_batch_sizes("1,2,4,8,16,32,64"))
    result.add_argument("--warmups", type=int, default=1)
    result.add_argument("--repeats", type=int, default=3)
    result.add_argument("--seed", type=int, default=20261104)
    result.add_argument("--mixed-prompts", action=argparse.BooleanOptionalAction, default=True,
                        help="Vary instruction lengths, retaining the PAP pick-and-place task.")
    result.add_argument("--mode", choices=("all", "throughput", "correctness"), default="all")
    result.add_argument("--max-arm-error-rad", type=float, default=None,
                        help="Optional acceptance threshold; absent means report numerical errors only.")
    result.add_argument("--max-gripper-error-m", type=float, default=None)
    result.add_argument("--max-feature-error", type=float, default=None)
    return result


def load_observations(paths):
    observations, identities = [], []
    for path in paths:
        raw = path.read_bytes()
        value = json.loads(raw)
        values = value if isinstance(value, list) else [value]
        for observation in values:
            if not isinstance(observation, dict) or not {"state", "images", "task"} <= observation.keys():
                raise ValueError(f"{path}: expected observation objects with state, images and task")
        observations.extend(values)
        identities.append({"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
                           "observations": len(values)})
    if not observations:
        raise ValueError("No observations supplied")
    return observations, identities


def make_observations(sources, count, mixed_prompts):
    alternatives = (None, "Place the object in the box.",
                    "Pick up the object and place it in the box.",
                    "Pick up the object from the tabletop, move it over the box, and release it inside the box.")
    observations = []
    for index in range(count):
        observation = copy.deepcopy(sources[index % len(sources)])
        if mixed_prompts and alternatives[index % len(alternatives)]:
            observation["task"] = alternatives[index % len(alternatives)]
        observations.append(observation)
    return observations


def error_stats(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError(f"Nonfinite or mismatched outputs: {actual.shape}, {expected.shape}")
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    return {"max_abs": float(np.max(np.abs(difference))),
            "rms": float(np.sqrt(np.mean(difference * difference)))}


def action_errors(actual, expected):
    if np.shape(actual)[-1] != 7:
        raise ValueError("This physical-unit benchmark requires six arm joints plus gripper width")
    return {"arm_rad": error_stats(np.asarray(actual)[..., :6], np.asarray(expected)[..., :6]),
            "gripper_m": error_stats(np.asarray(actual)[..., 6], np.asarray(expected)[..., 6])}


def compare_outputs(features, actions, reference_features, reference_actions, args):
    result = {"features": error_stats(features, reference_features),
              "actions": action_errors(actions, reference_actions)}
    checks = {"features": (result["features"]["max_abs"], args.max_feature_error),
              "arm_rad": (result["actions"]["arm_rad"]["max_abs"], args.max_arm_error_rad),
              "gripper_m": (result["actions"]["gripper_m"]["max_abs"], args.max_gripper_error_m)}
    result["thresholds"] = {key: {"limit": limit, "passed": error <= limit}
                            for key, (error, limit) in checks.items() if limit is not None}
    result["status"] = ("passed" if all(row["passed"] for row in result["thresholds"].values()) else "failed") if result["thresholds"] else "measured_without_acceptance_threshold"
    return result


def context_fingerprint(contexts):
    """Detect persistent cache mutation; excludes no tensor fields."""
    digest = hashlib.sha256()

    def visit(value):
        if hasattr(value, "detach"):
            import torch
            tensor = value.detach().cpu().contiguous()
            digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, np.ndarray):
            digest.update(str((value.shape, value.dtype)).encode())
            digest.update(value.tobytes())
        elif isinstance(value, dict):
            for key in sorted(value):
                digest.update(str(key).encode())
                visit(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif hasattr(value, "conditioning"):
            visit(value.conditioning)
            visit(value.baseline_noise)
        else:
            digest.update(repr(value).encode())

    visit(contexts)
    return digest.hexdigest()


class ForwardEvidence:
    """Inspect actual torch module calls, independently of advertised metadata."""
    def __init__(self, model):
        self.vision, self.transformer = [], []
        self.handles = [model.visual.register_forward_pre_hook(self.vision_hook, with_kwargs=True),
                        model.model.register_forward_pre_hook(self.transformer_hook, with_kwargs=True)]

    def vision_hook(self, module, args, kwargs):
        grid = kwargs.get("grid_thw")
        pixels = args[0] if args else kwargs.get("hidden_states")
        self.vision.append({"images": int(grid.shape[0]) if grid is not None else None,
                            "pixel_shape": list(pixels.shape) if pixels is not None else None})

    def transformer_hook(self, module, args, kwargs):
        value = kwargs.get("inputs_embeds")
        if value is None:
            value = kwargs.get("input_ids")
        if value is None and args:
            value = args[0]
        self.transformer.append({"batch": int(value.shape[0]) if value is not None else None,
                                 "shape": list(value.shape) if value is not None else None})

    def clear(self):
        self.vision.clear()
        self.transformer.clear()

    def summary(self):
        def counts(rows, key):
            return {str(value): sum(row[key] == value for row in rows)
                    for value in sorted({row[key] for row in rows}, key=str)}
        return {"vision_calls": len(self.vision), "vision_image_count_histogram": counts(self.vision, "images"),
                "transformer_calls": len(self.transformer),
                "transformer_batch_histogram": counts(self.transformer, "batch"),
                "vision_examples": self.vision[:2], "transformer_examples": self.transformer[:2]}

    def close(self):
        for handle in self.handles:
            handle.remove()


def prove_batch(encode_evidence, decode_evidence, batch_size, camera_count):
    transformer = [int(key) for key in decode_evidence["transformer_batch_histogram"] if key != "None"]
    images = [int(key) for key in encode_evidence["vision_image_count_histogram"] if key != "None"]
    return {"actual_transformer_batch_max": max(transformer, default=0),
            "actual_vision_images_max": max(images, default=0),
            "decode_reuses_vision_cache": decode_evidence["vision_calls"] == 0,
            "single_native_batch": (encode_evidence["vision_calls"] == 1 and images == [batch_size * camera_count]
                                    and transformer == [batch_size]),
            "not_serial_single_environment_calls": batch_size == 1 or
                (max(transformer, default=0) > 1 and max(images, default=0) > camera_count)}


def timed(torch, device, fn):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    begin = time.perf_counter()
    value = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return value, time.perf_counter() - begin


def timing_summary(samples, batch_size):
    encode = np.array([row["encode_seconds"] for row in samples])
    decode = np.array([row["decode_seconds"] for row in samples])
    total = encode + decode
    return {"samples": samples, "preprocess_encode_mean_seconds": float(encode.mean()),
            "decode_mean_seconds": float(decode.mean()), "total_mean_seconds": float(total.mean()),
            "total_median_seconds": float(np.median(total)),
            "observations_per_second": float(batch_size / total.mean())}


def native_references(backend, observations, noises, torch, device, progress):
    features, actions, timings = [], [], []
    for index, (observation, noise) in enumerate(zip(observations, noises)):
        context, encode_time = timed(torch, device, lambda: backend.policy.encode(
            backend._prepare_observation(observation), num_inference_timesteps=backend.num_inference_timesteps))
        reference = context["inputs"]["inputs_embeds"]
        tensor_noise = torch.as_tensor(noise, device=reference.device).unsqueeze(0)
        output, decode_time = timed(torch, device, lambda: backend.policy.decode(context, initial_noise=tensor_noise))
        features.append(context["feature"][0].detach().float().cpu().numpy().copy())
        actions.append(np.asarray(output["action"][0], dtype=np.float32) * backend.action_scale)
        timings.append({"encode_seconds": encode_time, "decode_seconds": decode_time})
        del context, reference, output, tensor_noise
        progress(index + 1)
    return np.stack(features), np.stack(actions), timings


def check_cache(backend, contexts, noises, expected, args):
    fingerprint = context_fingerprint(contexts)
    count = len(contexts)
    cases = {"reversed": list(reversed(range(count))),
             "subset": list(dict.fromkeys([count - 1, 0, count // 2])), "duplicate": [0, 0]}
    results = {}
    for name, indices in cases.items():
        output = backend.decode_batch([contexts[index] for index in indices], [noises[index] for index in indices])
        results[name] = {"indices": indices, "actions_vs_native_single": action_errors(output, expected[indices])}
        results[name]["thresholds_passed"] = all(limit is None or results[name]["actions_vs_native_single"][key]["max_abs"] <= limit
            for key, limit in (("arm_rad", args.max_arm_error_rad), ("gripper_m", args.max_gripper_error_m)))
    changed_noise = noises[:1].copy()
    changed_noise *= -1
    backend.decode_batch(contexts[:1], changed_noise)
    results["cache_unchanged_after_reorder_subset_duplicate_and_alternate_noise"] = fingerprint == context_fingerprint(contexts)
    if not results["cache_unchanged_after_reorder_subset_duplicate_and_alternate_noise"]:
        raise AssertionError("Decode mutated cached conditioning or baseline noise")
    return results


def run(args):
    if args.warmups < 0 or args.repeats < 1:
        raise ValueError("warmups must be nonnegative and repeats must be positive")
    for name in ("max_arm_error_rad", "max_gripper_error_m", "max_feature_error"):
        value = getattr(args, name)
        if value is not None and (not np.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be finite and nonnegative")
    report = {"state": "loading", "started_utc": datetime.now(timezone.utc).isoformat(),
              "argv": sys.argv, "python": sys.executable, "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "mode": args.mode, "batch_sizes": args.batch_sizes, "mixed_prompts": args.mixed_prompts,
              "warmups": args.warmups, "repeats": args.repeats, "seed": args.seed,
              "scope": "Direct model calls; excludes network, simulation and learner.", "results": []}
    write_json(args.output, report)
    backend = evidence = None
    try:
        import torch
        from pressb.runtime.server import load_backend
        sources, identities = load_observations(args.observation)
        observations = make_observations(sources, max(args.batch_sizes), args.mixed_prompts)
        report.update(observation_files=identities, supplied_observation_count=len(sources),
                      unique_prompts=sorted({row["task"] for row in observations}))
        config = json.loads(args.config.read_text())
        spec = config.get("backend", config)
        report["backend_config"] = spec
        backend = load_backend(spec)
        metadata = backend.describe()
        if metadata["action"]["units"] != ["rad"] * 6 + ["m"]:
            raise ValueError("Benchmark requires six radian arm joints and metre gripper width")
        device = torch.device(backend.config.get("device", "cuda:0"))
        report.update(metadata=metadata, torch_version=torch.__version__, cuda_version=torch.version.cuda,
                      device=str(device), gpu_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None)
        report["attention"] = {
            "action_transformer": backend.policy.model.config._attn_implementation,
            "vision": backend.policy.model.visual.config._attn_implementation,
            "native_multirow_vision": "batched_per_observation_with_original_internal_masks",
        }
        report["torch_cpu_threads"] = torch.get_num_threads()
        evidence = ForwardEvidence(backend.policy.model)
        noises = np.random.default_rng(args.seed).standard_normal(
            (len(observations), backend.horizon, backend.noise_dim), dtype=np.float32)
        report["noise_sha256"] = hashlib.sha256(noises.tobytes()).hexdigest()
        features_ref = actions_ref = None
        if args.mode != "throughput":
            def reference_progress(completed):
                report.update(state="native_single_reference", reference_completed=completed)
                write_json(args.output, report)
            features_ref, actions_ref, samples = native_references(backend, observations, noises, torch, device, reference_progress)
            report["native_single_reference_timing"] = timing_summary(samples, 1)
        for size in args.batch_sizes:
            report.update(state="benchmarking", current_batch_size=size)
            write_json(args.output, report)
            seeds = list(range(args.seed, args.seed + size))
            current = observations[:size]
            for _ in range(args.warmups):
                warm = backend.encode_batch(current, seeds)
                backend.decode_batch([row[0] for row in warm], noises[:size])
                del warm
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            samples = []
            for repeat in range(args.repeats):
                evidence.clear()
                encoded, encode_time = timed(torch, device, lambda: backend.encode_batch(current, seeds))
                encode_evidence = evidence.summary()
                contexts = [row[0] for row in encoded]
                features = np.stack([row[1] for row in encoded])
                evidence.clear()
                actions, decode_time = timed(torch, device, lambda: backend.decode_batch(contexts, noises[:size]))
                decode_evidence = evidence.summary()
                samples.append({"encode_seconds": encode_time, "decode_seconds": decode_time})
                if repeat + 1 < args.repeats:
                    del contexts, encoded, features, actions
            row = {"batch_size": size, "timing": timing_summary(samples, size),
                   "encode_forward_evidence": encode_evidence, "decode_forward_evidence": decode_evidence,
                   "batch_proof": prove_batch(encode_evidence, decode_evidence, size, len(backend.camera_mapping))}
            if device.type == "cuda":
                row["cuda_memory"] = {"peak_allocated_gib": torch.cuda.max_memory_allocated(device) / GIB,
                                      "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / GIB}
            if features_ref is not None:
                row["comparison_vs_native_single"] = compare_outputs(features, actions, features_ref[:size], actions_ref[:size], args)
                row["cache_checks"] = check_cache(backend, contexts, noises[:size], actions_ref[:size], args)
                row["valid_token_lengths"] = [int(context.conditioning["inputs"]["attention_mask"].sum().item())
                    for context in contexts if context.conditioning["inputs"].get("attention_mask") is not None]
            report["results"].append(row)
            write_json(args.output, report)
            del contexts, encoded, features, actions
        report["native_batch_proof_passed"] = all(row["batch_proof"]["single_native_batch"] and
            row["batch_proof"]["decode_reuses_vision_cache"] for row in report["results"])
        report["all_supplied_thresholds_passed"] = all(row.get("comparison_vs_native_single", {}).get("status") != "failed"
            and all(check.get("thresholds_passed", True) for check in row.get("cache_checks", {}).values() if isinstance(check, dict))
            for row in report["results"])
        report["state"] = "completed" if report["native_batch_proof_passed"] and report["all_supplied_thresholds_passed"] else "failed_validation"
        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(args.output, report)
        return 0 if report["state"] == "completed" else 1
    except Exception as error:
        report.update(state="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        write_json(args.output, report)
        traceback.print_exc()
        return 1
    finally:
        if evidence is not None:
            evidence.close()
        if backend is not None:
            backend.close()


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
