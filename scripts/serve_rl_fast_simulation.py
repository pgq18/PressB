#!/usr/bin/env python3
"""Serve indexed-reset Isaac rollouts with batched control and chunk-boundary RGB.

Example (no inference or learner runs in this process):
  .conda/envs/pressb/bin/python scripts/serve_rl_fast_simulation.py \
    --gpu 1 --port 19870 --num-envs 64 --output outputs/online_rl/sim_new
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import sys
import traceback


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    frozen = ROOT / "outputs/rtx5090_eval_comparison/input"
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19970)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--config", type=Path, default=frozen / "config.json")
    parser.add_argument("--snapshot", type=Path, default=frozen / "scene.usda")
    parser.add_argument("--dataset", type=Path, default=frozen / "dataset_metadata")
    parser.add_argument("--asset-bundle", type=Path, default=frozen / "assets")
    parser.add_argument("--no-asset-bundle", action="store_true", help="Use the original scene paths directly")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-seconds", type=float, default=15.)
    parser.add_argument("--smoothing-window", type=int, choices=(1, 3, 5, 7, 9, 11), default=3)
    parser.add_argument("--single-gamma", type=float, default=.99)
    parser.add_argument("--response-cache-size", type=int, default=2)
    parser.add_argument("--camera-resolution", type=int, choices=(224, 640), default=224,
        help="224: render 320x240 with the trained 4:3 FOV, then resize to 224 square; 640: original 640x480 RGB")
    parser.add_argument("--render-subframes", type=int, default=1)
    parser.add_argument("--light-settle-captures", type=int, default=1)
    parser.add_argument("--light-settle-subframes", type=int, default=2)
    parser.add_argument("--ik-iterations", type=int, default=16)
    parser.add_argument("--log-actions", action="store_true")
    parser.add_argument("--png-workers", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--gpu-dynamics", action="store_true",
        help="Experimental PhysX GPU dynamics with NumPy readback; disables CCD, explicit new contract")
    args = parser.parse_args(argv)
    if (not 1 <= args.port <= 65535 or args.gpu < 0 or args.num_envs < 1
            or args.response_cache_size < 1 or args.render_subframes < 1
            or args.light_settle_captures < 0 or args.light_settle_subframes < 1 or args.ik_iterations < 1
            or args.png_workers < 1 or args.cpu_threads < 1
            or not math.isfinite(args.max_seconds)
            or not 0 < args.max_seconds <= 120
            or abs(args.max_seconds * 120 - round(args.max_seconds * 120)) > 1e-7
            or not math.isfinite(args.single_gamma) or not 0 < args.single_gamma <= 1):
        parser.error("Require valid port/GPU/counts, duration on a 120 Hz tick, and gamma in (0,1]")
    for name in ("config", "snapshot", "dataset", "output", "asset_bundle"):
        setattr(args, name, getattr(args, name).resolve())
    if args.no_asset_bundle:
        args.asset_bundle = None
    if args.output.exists():
        parser.error(f"Output directory must be new: {args.output}")
    native_log_path = args.output.with_name(args.output.name + ".kit.log")
    if native_log_path.exists():
        parser.error(f"Native log must be new: {native_log_path}")
    native_log_path.parent.mkdir(parents=True, exist_ok=True)

    # Fail on identity/configuration errors before opening Kit or a GPU context.
    from pressb.online_rl.simulation import _file_identity
    config = json.loads(args.config.read_text())
    collection = json.loads((args.dataset / "meta/collection_metadata.json").read_text())
    if config != collection["config"] or _file_identity(args.snapshot)["sha256"] != collection["scene_sha256"]:
        parser.error("Config and scene must match the frozen training collection")
    if args.asset_bundle is not None and not (args.asset_bundle / "asset_bundle.json").is_file():
        parser.error("asset_bundle.json is missing from the specified bundle")
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "8")
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    cache_tmp = ROOT / ".cache/tmp"
    cache_tmp.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TMPDIR", str(cache_tmp))
    app = backend = server = None
    exit_code = 1

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        from isaacsim import SimulationApp
        from collect_dataset import DATASET_ANTI_ALIASING
        app = SimulationApp(dict(headless=True, create_new_stage=False, width=640, height=480,
            active_gpu=args.gpu, physics_gpu=args.gpu, multi_gpu=False,
            disable_viewport_updates=True, limit_cpu_threads=args.cpu_threads,
            renderer="RayTracedLighting", anti_aliasing=DATASET_ANTI_ALIASING, fast_shutdown=True,
            extra_args=["--/app/asyncRendering=false", "--/rtx/ecoMode/enabled=false",
                f"--/log/file={native_log_path}", "--/log/fileFlushLevel=verbose", "--/log/fileAppend=false",
                "--/log/async=false",
                "--/plugins/carb.tasking.plugin/threadCount=8", "--/plugins/omni.tbb.globalcontrol/maxThreadCount=8",
                "--/persistent/physics/numThreads=4", "--/validate/p2p/enabled=false", "--/validate/iommu/enabled=false"]))
        from pressb.online_rl.rpc import RPCServer
        from pressb.online_rl.fast_simulation import FastIsaacVectorBackend, IndexedSimulationService
        backend = FastIsaacVectorBackend(app, project_root=ROOT, config=args.config, snapshot=args.snapshot,
            dataset=args.dataset, output=args.output, num_envs=args.num_envs, max_seconds=args.max_seconds,
            smoothing_window=args.smoothing_window, asset_bundle=args.asset_bundle, gpu=args.gpu,
            single_gamma=args.single_gamma, camera_resolution=args.camera_resolution,
            render_subframes=args.render_subframes, light_settle_captures=args.light_settle_captures,
            light_settle_subframes=args.light_settle_subframes, ik_iterations=args.ik_iterations,
            log_actions=args.log_actions, png_workers=args.png_workers, cpu_threads=args.cpu_threads,
            gpu_dynamics=args.gpu_dynamics, native_log_path=native_log_path)
        service = IndexedSimulationService(backend, single_gamma=args.single_gamma,
            response_cache_size=args.response_cache_size)
        server = RPCServer((args.host, args.port), {"/health": service.health,
            "/reset": service.reset, "/reset_envs": service.reset_envs,
            "/step": service.step, "/close_run": service.close_run},
            token=os.environ.get("PRESSB_RL_TOKEN"))
        print(json.dumps(dict(event="simulation_ready", host=args.host, port=args.port,
            gpu=args.gpu, num_envs=args.num_envs, output=str(args.output)), allow_nan=False), flush=True)
        # HTTPServer dispatches every backend call on this main/Kit thread.
        # Do not replace it with ThreadingHTTPServer or call app.update in idle.
        server.serve_forever(poll_interval=.25)
        exit_code = 0
    except KeyboardInterrupt:
        exit_code = 0
    except Exception:
        traceback.print_exc()
    finally:
        try:
            if server is not None:
                server.server_close()
        finally:
            try:
                if backend is not None:
                    backend.close()
            except Exception:
                # A logging/disk failure must never skip releasing Isaac/GPU.
                exit_code = 1
                traceback.print_exc()
            finally:
                if app is not None:
                    try:
                        app.app.post_quit(exit_code)
                    finally:
                        app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
