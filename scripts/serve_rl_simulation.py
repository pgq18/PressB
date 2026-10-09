#!/usr/bin/env python3
"""Serve persistent Isaac vector reset/step on one GPU and one HTTP port.

Example (no inference or learner runs in this process):
  .conda/envs/pressb/bin/python scripts/serve_rl_simulation.py \
    --gpu 1 --port 19870 --num-envs 3 --output outputs/online_rl/sim_new
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
    parser.add_argument("--port", type=int, default=19870)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=3)
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
    args = parser.parse_args(argv)
    if (not 1 <= args.port <= 65535 or args.gpu < 0 or args.num_envs < 1
            or args.response_cache_size < 1 or not math.isfinite(args.max_seconds)
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
            renderer="RayTracedLighting", anti_aliasing=DATASET_ANTI_ALIASING, fast_shutdown=True,
            extra_args=["--/app/asyncRendering=false", "--/rtx/ecoMode/enabled=false",
                "--/plugins/carb.tasking.plugin/threadCount=8", "--/plugins/omni.tbb.globalcontrol/maxThreadCount=8",
                "--/persistent/physics/numThreads=4", "--/validate/p2p/enabled=false", "--/validate/iommu/enabled=false"]))
        from pressb.online_rl.rpc import RPCServer
        from pressb.online_rl.simulation import IsaacVectorBackend, SimulationService
        backend = IsaacVectorBackend(app, project_root=ROOT, config=args.config, snapshot=args.snapshot,
            dataset=args.dataset, output=args.output, num_envs=args.num_envs, max_seconds=args.max_seconds,
            smoothing_window=args.smoothing_window, asset_bundle=args.asset_bundle, gpu=args.gpu,
            single_gamma=args.single_gamma)
        service = SimulationService(backend, single_gamma=args.single_gamma,
            response_cache_size=args.response_cache_size)
        server = RPCServer((args.host, args.port), {"/health": service.health,
            "/reset": service.reset, "/step": service.step, "/close_run": service.close_run},
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
