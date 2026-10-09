#!/usr/bin/env python3
"""Benchmark frozen inference on saved real observations, without simulator calls.

Inputs: a directory with manifest.json containing task/state/seed and images
mapping camera names to {file, sha256}. Reports backend and application timing;
these are inference microbenchmarks, never end-to-end training throughput.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from PIL import Image
import torch

from pressb.online_rl.inference import FrozenPiperBackend, InferenceApplication


def compare(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    delta = np.abs(actual - expected)
    return dict(max_abs_error=float(delta.max()), mean_abs_error=float(delta.mean()),
                array_equal=bool(np.array_equal(actual, expected)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vla-repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 64])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--image-preprocess-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--feature-encoding", choices=("json", "float32_base64"), default="json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(args.torch_threads)
    began = time.monotonic()
    backend = FrozenPiperBackend.load(args.vla_repo, args.checkpoint, device=args.device,
                                      image_preprocess_device=args.image_preprocess_device)
    rows = json.loads((args.inputs / "manifest.json").read_text())
    observations, seeds, wire = [], [], []
    for row in rows:
        images = {}
        encoded = {}
        for key, description in row["images"].items():
            data = (args.inputs / description["file"]).read_bytes()
            assert hashlib.sha256(data).hexdigest() == description["sha256"]
            images[key] = Image.open(BytesIO(data)).convert("RGB")
            buffer = BytesIO()
            images[key].resize((224, 224), Image.Resampling.BILINEAR).save(buffer, format="PNG")
            encoded[key] = base64.b64encode(buffer.getvalue()).decode()
        observations.append(dict(task=row["task"], state=row["state"], global_image=images["global"],
                                 wrist_image=images["wrist"]))
        seeds.append(row["seed"])
        wire.append(dict(task=row["task"], state=row["state"], images=encoded))
    report = dict(kind="frozen_inference_saved_real_observations_microbenchmark",
                  started_at=datetime.now(timezone.utc).isoformat(),
                  distinct_observations=len(rows), inputs_manifest_sha256=hashlib.sha256(
                      (args.inputs / "manifest.json").read_bytes()).hexdigest(),
                  input_sources=rows, device=str(backend.device), gpu=torch.cuda.get_device_name(backend.device),
                  health=backend.health(), torch_threads=args.torch_threads,
                  feature_encoding=args.feature_encoding,
                  load_seconds=time.monotonic() - began, results=[],
                  limitations=["Saved real images are repeated when batch size exceeds dataset size.",
                               "Application timing includes PNG decoding and JSON serialization, not network transport.",
                               "No simulator, learner updates, or task success evaluation is included."])

    def persist():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def timed(call, count):
        values = []
        for _ in range(args.repeats):
            torch.cuda.synchronize(backend.device)
            start = time.perf_counter()
            result = call()
            torch.cuda.synchronize(backend.device)
            values.append(time.perf_counter() - start)
            del result
        mean = statistics.mean(values)
        return dict(seconds=values, mean_seconds=mean, median_seconds=statistics.median(values),
                    observations_per_second=count / mean)

    backend.encode(observations[0], seed=seeds[0], include_base_actions=True)
    serial = [backend.encode(observation, seed=seed, include_base_actions=True)
              for observation, seed in zip(observations, seeds)]
    for size in args.batch_sizes:
        batch = [observations[index % len(rows)] for index in range(size)]
        sample_seeds = [seeds[index % len(rows)] for index in range(size)]
        sample_wire = [wire[index % len(rows)] for index in range(size)]
        expected = [serial[index % len(rows)] for index in range(size)]
        encode = ((lambda include: [backend.encode(batch[0], seed=sample_seeds[0], include_base_actions=include)])
                  if size == 1 else
                  (lambda include: backend.encode_batch(batch, seeds=sample_seeds, include_base_actions=include)))
        try:
            torch.cuda.reset_peak_memory_stats(backend.device)
            actual = encode(True)
            contexts = [item[0] for item in actual]
            noise = np.random.default_rng(42).normal(size=(size, 7, 9)).astype(np.float32)
            expected_decode = np.stack([backend.decode(context, item) for context, item in zip(contexts, noise)])
            decode = ((lambda: np.stack([backend.decode(contexts[0], noise[0])])) if size == 1 else
                      (lambda: backend.decode_batch(contexts, noise)))
            entry = dict(batch_size=size,
                         base_actions=compare(np.stack([item[1] for item in actual]),
                                              np.stack([item[1] for item in expected])),
                         features=compare(np.stack([item[0].feature for item in actual]),
                                          np.stack([item[0].feature for item in expected])),
                         cached_decode=compare(decode(), expected_decode),
                         encode_and_base=timed(lambda: encode(True), size),
                         encode_only=timed(lambda: encode(False), size),
                         decode_only=timed(decode, size),
                         peak_allocated_bytes=torch.cuda.max_memory_allocated(backend.device))
            action_batch = np.stack([item[1] for item in actual])
            action_serial = np.stack([item[1] for item in expected])
            entry["base_actions_xyz_metres"] = compare(action_batch[..., :3], action_serial[..., :3])
            entry["base_actions_rotation6d"] = compare(action_batch[..., 3:], action_serial[..., 3:])
            batch_pose = backend.pose_converter(action_batch)
            serial_pose = backend.pose_converter(action_serial)
            qa, qb = batch_pose[..., 3:7].astype(np.float64), serial_pose[..., 3:7].astype(np.float64)
            qa /= np.linalg.norm(qa, axis=-1, keepdims=True)
            qb /= np.linalg.norm(qb, axis=-1, keepdims=True)
            angles = np.rad2deg(2 * np.arccos(np.clip(np.abs((qa * qb).sum(-1)), 0, 1)))
            entry["base_actions_rotation_angle_deg"] = dict(max=float(angles.max()), mean=float(angles.mean()))
            application = InferenceApplication(backend, batch_size=size, cache_size=max(256, size * 2))
            previous_keys = []

            def application_call():
                response = application.encode(dict(observations=sample_wire, seeds=sample_seeds,
                    include_base_actions=True, release_context_ids=previous_keys.copy(),
                    feature_encoding=args.feature_encoding))
                previous_keys[:] = [item["context_id"] for item in response["items"]]
                return json.dumps(response, allow_nan=False)

            application_call()
            entry["application_encode224_and_base"] = timed(application_call, size)
            report["results"].append(entry)
            print(json.dumps(entry, allow_nan=False), flush=True)
            del application, actual, contexts, expected_decode
        except torch.cuda.OutOfMemoryError as error:
            report["results"].append(dict(batch_size=size, error=str(error), out_of_memory=True))
            torch.cuda.empty_cache()
        persist()
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["total_seconds"] = time.monotonic() - began
    report["frozen"] = all(not parameter.requires_grad for parameter in backend.model.parameters())
    persist()


if __name__ == "__main__":
    main()
