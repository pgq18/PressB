#!/usr/bin/env python3
"""Render auxiliary views of a saved manipulation collection's initial pose.

Uses the shared simulator and independent spectator cameras. No robot hardware,
trajectory playback, or changes to the policy's two calibrated cameras.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    import numpy as np
    from PIL import Image
    from pressb.registry import resolve
    from pressb.runtime.recording import write

    manifest = json.loads((args.raw / "manifest.json").read_text())
    metadata = json.loads((args.raw / "episodes" / f"episode_{args.episode:06d}" / "metadata.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    config = deepcopy(manifest["config"])
    config.update(num_envs=1, gpu=args.gpu, headless=True, horizon=32,
                  terminate_on_success=False, log_dir=str(args.output / "simulation"))
    module, attribute = resolve("task", manifest["task_id"]).split(":")
    backend = getattr(importlib.import_module(module), attribute)(config)
    product = annotator = None
    try:
        scene = backend.task.config["scene"]
        original = {key:deepcopy(value) for key,value in scene.items() if "camera" in key}
        objects = {name:{key:pose[key] for key in ("center_xy", "yaw_deg")}
                   for name,pose in metadata["reset_spec"]["objects"].items()}
        backend.reset([0], metadata["seed"], {"by_env":{"0":{"objects":objects}}})
        home = scene["initial_state_joint_rad_m"]
        row = backend.step([dict(env_id=0, actions=[home]*30)])[0]
        engine = backend.engine
        import omni.replicator.core as rep
        from pressb.sensors.camera import make_camera
        from pressb.sensors.tiled import build_tiled_rgb
        base = np.asarray(backend.task.scene["robot_base_position_world"]) + engine.offsets[0]
        views = {
            "three_quarter":dict(eye_base_m=[.78,-.72,.52], target_base_m=[.20,.015,.20],
                                 fx=1000.,fy=1000.,cx=640.,cy=400.,resolution=[1280,800]),
            "side":dict(eye_base_m=[.14,-.85,.38], target_base_m=[.20,.025,.20],
                        fx=1050.,fy=1050.,cx=640.,cy=400.,resolution=[1280,800]),
        }
        paths = [make_camera(engine.stage, f"/World/Inspection_{name}", spec, base)
                 for name,spec in views.items()]
        product, annotator, split = build_tiled_rgb(paths, (1280,800))
        before = engine.world.current_time
        engine.publish_transforms()
        engine.reset_render_history()
        for _ in range(8):
            rep.orchestrator.step(delta_time=0., pause_timeline=False, rt_subframes=8)
        images = split(annotator.get_data())
        assert engine.world.current_time == before, "Inspecting a pose advanced simulation time"
        assert original == {k:v for k,v in scene.items() if "camera" in k}, "Policy camera configuration changed"
        for name,rgb in zip(views,images):
            if rgb.shape != (800,1280,3) or rgb.std() < 5:
                raise ValueError(f"Invalid spectator image: {name}")
            Image.fromarray(np.ascontiguousarray(rgb)).save(args.output / f"{name}.png")
        write(args.output / "report.json", dict(status="passed", task_id=manifest["task_id"],
            episode=args.episode, simulation_only=True, initial_pose=backend.task.config["initial_pose"],
            commanded_state=home, measured_state=row["observation"]["state"], views=views,
            calibrated_cameras_unchanged=True, physics_advanced_during_capture=False,
            snapshot=backend.snapshot([0])))
    except BaseException as error:
        # Isaac fast shutdown can exit from close(); preserve the error first.
        write(args.output / "report.json", dict(status="failed", error=f"{type(error).__name__}: {error}"))
        raise
    finally:
        try:
            if annotator is not None:
                annotator.detach([product.path])
            if product is not None:
                product.destroy()
        finally:
            backend.close()


if __name__ == "__main__":
    main()
