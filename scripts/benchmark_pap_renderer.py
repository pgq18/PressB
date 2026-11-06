#!/usr/bin/env python3
"""Audit PAP reset RGB convergence and physics clocks for an explicit profile.

The report compares the first post-teleport observation against additional
zero-time captures of the same scene. RGB differences include stochastic RTX
noise; retained red pixels away from the converged cube expose stale poses.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def rgb(payload):
    return np.array(Image.open(io.BytesIO(base64.b64decode(payload["data"]))).convert("RGB"))


def red_mask(image):
    values = image.astype(np.int16)
    return (values[..., 0] > 80) & (values[..., 0] > values[..., 1] * 1.5) & (values[..., 0] > values[..., 2] * 1.5)


def stale_pose_overlap(old_reference, new_reference, image):
    """Measure red remaining at the old cube, excluding common static box RGB."""
    from scipy.ndimage import label
    new_mask = np.asarray(Image.fromarray(red_mask(new_reference).astype(np.uint8)*255)
                          .filter(ImageFilter.MaxFilter(7))) > 0
    regions, count = label(red_mask(old_reference) & ~new_mask)
    if not count:
        return {"old_pose_pixels": 0, "retained_red_fraction": None}
    sizes = np.bincount(regions.ravel())
    sizes[0] = 0
    old_region = regions == sizes.argmax()
    return {"old_pose_pixels": int(old_region.sum()),
            "retained_red_fraction": float(red_mask(image)[old_region].mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/rl_nodes/pap_vector_simulation.json")
    parser.add_argument("--profile", choices=("pap_calibrated", "rl_fast"), required=True)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--reset-frames", type=int, default=2)
    parser.add_argument("--subframes", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--setting", action="append", default=[], metavar="PATH=JSON",
                        help="Explicit experimental RTX override, recorded in report")
    parser.add_argument("--reset-setting", action="append", default=[], metavar="PATH=JSON",
                        help="Experimental setting applied immediately before each reset")
    parser.add_argument("--profile-python", action="store_true", help="Record cProfile for one hold chunk")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    args.output.mkdir(parents=True)
    for key, value in {"PYTHONNOUSERSITE": "1", "PXR_WORK_THREAD_LIMIT": "8",
                       "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}.items():
        os.environ.setdefault(key, value)
    from pressb.rl_nodes.pap_vector import PAPVectorBackend
    cfg = json.loads(args.config.read_text())["backend"]["config"]
    cfg.update(num_envs=args.num_envs, gpu_ids=[args.gpu], renderer_profile=args.profile,
               render_resolution=[args.width, args.width * 3 // 4], reset_render_frames=args.reset_frames,
               render_subframes=args.subframes, log_dir=str(args.output / "worker"))
    report = {"requested_config": cfg, "stages": []}
    def record():
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    backend = None
    try:
        start = time.perf_counter()
        backend = PAPVectorBackend(cfg)
        import carb.settings
        settings = carb.settings.get_settings()
        report["experimental_settings"] = {}
        for entry in args.setting:
            key, value = entry.split("=", 1)
            settings.set(key, json.loads(value))
            report["experimental_settings"][key] = settings.get(key)
        report["renderer_probe"] = {key: settings.get(key) for key in (
            "/rtx/post/aa/op", "/rtx-transient/dldenoiser/enabled", "/rtx/reflections/denoiser/enabled",
            "/rtx/ambientOcclusion/temporalSqrtLength", "/rtx/directLighting/sampledLighting/ris/historyLength",
            "/rtx/indirectDiffuse/denoiser/temporal/maxHistory", "/rtx/lightspeed/NRD/method")}
        report["startup_seconds"] = time.perf_counter() - start
        report["metadata"] = backend.describe()
        print(json.dumps({"event": "ready", "startup_seconds": report["startup_seconds"]}), flush=True)
        ids = list(range(args.num_envs))
        report["reset_settings"] = {entry.split("=", 1)[0]: json.loads(entry.split("=", 1)[1])
                                    for entry in args.reset_setting}
        old_reference = None
        # Positions are inside the recorded four-edge sampling rectangle.
        for scenario, xy in (("old", [.28, -.09]), ("new", [.44, .02])):
            for key, value in report["reset_settings"].items():
                settings.set(key, value)
            before_time = backend.world.current_time
            start = time.perf_counter()
            rows = backend.reset(ids, 20261104, {"center_xy": xy, "color": "red", "yaw_deg": 0.})
            elapsed = time.perf_counter() - start
            frames = {0: [rgb(rows[0]["observation"]["images"][view]) for view in ("up", "wrist")]}
            snapshot = backend.snapshot()
            captures = [1, 2, 4, 8, 16, 48]
            for frame in range(1, max(captures)+1):
                tiles = backend._capture(return_tiles=frame in captures)
                if frame in captures:
                    frames[frame] = [np.asarray(tiles[j]).copy() for j in (0, 1)]
            assert backend.world.current_time == before_time
            assert backend.snapshot() == snapshot
            comparisons = []
            for frame, images in frames.items():
                per_view = {}
                for j, view in enumerate(("up", "wrist")):
                    Image.fromarray(images[j]).save(args.output / f"{scenario}_{view}_extra_{frame:02d}.png")
                    reference = frames[48][j]
                    reference_red = red_mask(reference)
                    dilated = np.asarray(Image.fromarray(reference_red.astype(np.uint8)*255).filter(ImageFilter.MaxFilter(7))) > 0
                    mask = red_mask(images[j])
                    coords = np.argwhere(mask)
                    per_view[view] = {"mae_to_48": float(np.abs(images[j].astype(float)-reference).mean()),
                        "p99_error_to_48": float(np.quantile(np.abs(images[j].astype(float)-reference), .99)),
                        "red_pixels": int(mask.sum()), "red_pixels_outside_converged_plus3px": int((mask & ~dilated).sum()),
                        "red_centroid_yx": coords.mean(0).tolist() if len(coords) else None}
                    if old_reference is not None:
                        per_view[view]["stale_pose"] = stale_pose_overlap(old_reference[j], reference, images[j])
                comparisons.append({"extra_captures": frame, "views": per_view})
            report["stages"].append({"scenario": scenario, "center_xy": xy, "reset_seconds": elapsed,
                "physics_unchanged": True, "comparisons": comparisons})
            old_reference = frames[48]
            record()
            print(json.dumps({"event": "reset_audit", "scenario": scenario, "seconds": elapsed,
                              "initial_comparison": comparisons[0]}), flush=True)
        # Peer slots must not advance when only one slot resets.
        peer_ids = ids[1:]
        before = backend.snapshot(peer_ids) if peer_ids else None
        start = time.perf_counter()
        backend.reset([0], 44, {"center_xy": [.32, -.06], "color": "red"})
        report["indexed_reset_seconds"] = time.perf_counter()-start
        report["peer_snapshot_unchanged"] = not peer_ids or backend.snapshot(peer_ids) == before
        assert report["peer_snapshot_unchanged"]
        start = time.perf_counter()
        commands = [{"env_id": i, "actions": np.repeat(backend.initial[None], 32, axis=0).tolist()} for i in ids]
        if args.profile_python:
            import cProfile
            import pstats
            profiler = cProfile.Profile()
            rows = profiler.runcall(backend.step, commands, terminal_observation_mode="state_only_if_no_bootstrap")
            profiler.dump_stats(str(args.output / "step.prof"))
            with (args.output / "step_profile.txt").open("w") as stream:
                pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(60)
        else:
            rows = backend.step(commands, terminal_observation_mode="state_only_if_no_bootstrap")
        report["hold_step_seconds"] = time.perf_counter()-start
        report["hold_step_transitions"] = len(rows)
        report["timings"] = backend.timings
        record()
        print(json.dumps({"event": "complete", "indexed_reset_seconds": report["indexed_reset_seconds"],
                          "hold_step_seconds": report["hold_step_seconds"]}), flush=True)
    finally:
        if backend is not None:
            backend.close()


if __name__ == "__main__":
    main()
