#!/usr/bin/env python3
"""Render identical poses in every clone and compare with a single-env reference.

Uses the production create_envs function unchanged. Static RGB diagnostics only;
this does not recollect, edit, or certify physical demonstration episodes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def identity(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def compare(output, reference):
    import numpy as np
    from PIL import Image

    report_path = output / "report.json"
    current = json.loads(report_path.read_text())
    baseline = json.loads(reference.read_text())
    errors, rows = [], []
    if baseline["num_envs"] != 1:
        errors.append("Reference must contain exactly one environment")
    for key in ("config", "snapshot_sha256", "source_pose_sha256", "camera_calibration",
                "exposure_used", "renderer", "anti_aliasing", "renderer_history_controls", "rgb_format"):
        if current[key] != baseline[key]:
            errors.append(f"Reference condition differs: {key}")
    refs = {(row["pose"], row["camera"]): row for row in baseline["images"]}
    if len(current["active_domes"]) != 1:
        errors.append("Expected exactly one active global DomeLight")
    if len(current["domes"]) != current["num_envs"]:
        errors.append("Missing authored DomeLight records")
    for row in current["images"]:
        ref = refs[row["pose"], row["camera"]]
        a = np.array(Image.open(reference.parent / ref["file"]), dtype=np.float64)
        b = np.array(Image.open(output / row["file"]), dtype=np.float64)
        relative = 100. * (row["mean_luma"] / ref["mean_luma"] - 1.)
        mae = float(np.abs(a - b).mean())
        same_camera = (row["camera_attributes"] == ref["camera_attributes"])
        transform_difference = float(np.max(np.abs(np.array(row["camera_to_env_matrix"]) -
                                                    np.array(ref["camera_to_env_matrix"]))))
        q_difference = float(np.max(np.abs(np.array(row["joint_positions"]) -
                                          np.array(ref["joint_positions"])) ))
        passed = (abs(relative) <= 2. and mae <= 3. and same_camera and
                  transform_difference <= .001 and q_difference <= .001)
        result = {"env_id": row["env_id"], "pose": row["pose"], "camera": row["camera"],
                  "mean_luma_change_percent": relative, "rgb_mae_255": mae,
                  "p95_absolute_channel_difference": float(np.percentile(np.abs(a-b), 95)),
                  "same_camera_attributes": same_camera,
                  "camera_transform_max_difference": transform_difference,
                  "joint_max_difference_rad": q_difference,
                  "clipped_any_channel_fraction": row["clipped_any_channel_fraction"],
                  "passed": passed}
        rows.append(result)
        if not passed:
            errors.append(f"Lighting comparison failed: env {row['env_id']} {row['pose']} {row['camera']}")
    if len(rows) != current["num_envs"] * 4:
        errors.append("Expected two poses and two RGB views per environment")
    result = {"success": not errors, "errors": errors, "reference": str(reference),
              "thresholds": {"absolute_mean_luma_change_percent": 2., "rgb_mae_255": 3.,
                             "camera_transform_max_difference": .001, "joint_max_difference_rad": .001},
              "comparisons": rows}
    (output / "comparison.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return not errors


def render(args):
    import numpy as np
    from PIL import Image

    if args.num_envs < 1:
        raise ValueError("--num-envs must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "report.json").exists():
        raise ValueError("Output report already exists; choose a new diagnostic directory")
    cfg = json.loads(args.config.read_text())
    inputs = [args.config, args.snapshot, args.source_episode / "frames.npz",
              args.source_episode / "physics.npz", ROOT / "src/pressb/dataset_scene.py"]
    identities = [identity(path) for path in inputs]
    calibration = {}
    for view in ("wrist", "global"):
        path = args.snapshot.parent / f"{view}_camera/intrinsics.json"
        calibration[view] = json.loads(path.read_text())
        identities.append(identity(path))
    with np.load(args.source_episode / "frames.npz", allow_pickle=False) as frames:
        lit = np.flatnonzero(frames["lights"][:, 11])
        if not len(lit):
            raise ValueError("Source episode must show floor 35 illuminated")
        source_frame = int(lit[len(lit) // 2])
        press_q = frames["q_actual"][source_frame].copy()
        physics_index = int(frames["physics_index"][source_frame])
    with np.load(args.source_episode / "physics.npz", allow_pickle=False) as physics:
        press_travel = physics["button_travel"][physics_index].copy()
    spec = importlib.util.spec_from_file_location("lighting_collection_settings", ROOT / "scripts/collect_dataset.py")
    collection = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collection)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "create_new_stage": False,
        "width": 640, "height": 480, "active_gpu": args.gpu, "physics_gpu": args.gpu,
        "multi_gpu": False, "renderer": "RayTracedLighting", "anti_aliasing": 2,
        "fast_shutdown": True, "extra_args": ["--/app/asyncRendering=false",
        "--/rtx/ecoMode/enabled=false", "--/plugins/carb.tasking.plugin/threadCount=8",
        "--/plugins/omni.tbb.globalcontrol/maxThreadCount=8", "--/persistent/physics/numThreads=4",
        "--/validate/p2p/enabled=false", "--/validate/iommu/enabled=false"]})
    try:
        import carb
        import omni.usd
        import omni.replicator.core as rep
        from pxr import Usd, UsdGeom, UsdLux
        from isaacsim.core.api import World
        from isaacsim.core.utils.types import ArticulationAction
        from pressb.dataset_scene import create_envs, build_tiled_rgb, set_light

        world = World(stage_units_in_meters=1., physics_dt=1/120, rendering_dt=.1,
                      backend="numpy", device="cpu")
        world.get_physics_context().set_physx_update_transformations_settings(
            update_to_usd=True, update_velocities_to_usd=True)
        envs = create_envs(world, args.snapshot, args.num_envs, cfg=cfg)
        product, annotator, split = build_tiled_rgb([path for env in envs
            for path in (env.wrist_camera_path, env.global_camera_path)])
        rep.orchestrator.set_capture_on_play(False)
        world.reset()
        controllers, arms, joints = [], [], []
        for env in envs:
            robot = env.robot
            arm = np.array([robot.get_dof_index(f"joint{i}") for i in range(1, 7)])
            fingers = np.array([robot.get_dof_index(f"joint{i}") for i in (7, 8)])
            q = robot.get_joint_positions().copy()
            q[arm], q[fingers] = cfg["home_q"], cfg["gripper_joint_positions_m"]
            robot.set_joint_positions(q)
            robot.set_joint_velocities(np.zeros_like(q))
            controller = robot.get_articulation_controller()
            controller.set_gains(kps=np.full(len(q), 12000.), kds=np.full(len(q), 600.))
            controller.apply_action(ArticulationAction(joint_positions=q))
            controllers.append(controller)
            arms.append(arm)
            joints.append(q)
        for i in range(90):
            world.step(render=False)
            if i % 4 == 0:
                world.render()
        settings = carb.settings.get_settings()
        for key, value in collection.RENDER_HISTORY_CONTROLS.items():
            settings.set(key, value)
        exposure = {"/rtx/post/histogram/enabled": False,
                    "/rtx/post/tonemap/filmIso": 100., "/rtx/post/tonemap/exposureTime": .02,
                    "/rtx/post/tonemap/fNumber": 5., "/rtx/post/tonemap/op": 6}
        before = {key: settings.get(key) for key in exposure}
        for key, value in exposure.items():
            settings.set(key, value)
        domes = []
        for prim in world.stage.TraverseAll():
            if prim.IsA(UsdLux.DomeLight):
                spec = world.stage.GetRootLayer().GetPrimAtPath(prim.GetPath())
                domes.append({"path": str(prim.GetPath()), "active": prim.IsActive(),
                    "has_authored_active": prim.HasAuthoredActive(),
                    "root_layer_active_opinion": spec.active if spec and spec.HasActive() else None,
                    "composed_intensity": prim.GetAttribute("inputs:intensity").Get()})
        active = [d["path"] for d in domes if d["active"]]
        if len(active) != 1 or len(domes) != args.num_envs:
            raise ValueError(f"Expected one active Dome and {args.num_envs-1} inactive duplicates: {domes}")
        report = {"num_envs": args.num_envs, "gpu": args.gpu, "config": cfg,
            "source_files": identities, "snapshot_sha256": identities[1]["sha256"],
            "source_pose_sha256": identities[2]["sha256"], "source_episode_frame": source_frame,
            "camera_calibration": calibration, "renderer": "RayTracedLighting", "anti_aliasing": 2,
            "renderer_history_controls": collection.RENDER_HISTORY_CONTROLS,
            "rgb_format": {"width": 640, "height": 480, "channels": 3, "dtype": "uint8", "encoding": "PNG"},
            "exposure_before": before, "exposure_used": {key: settings.get(key) for key in exposure},
            "domes": domes, "active_domes": active, "images": [],
            "method": "Production create_envs, static pose replay, reset accumulation, zero-dt rendering"}
        for pose in ("home", "press35"):
            for index, env in enumerate(envs):
                q = joints[index]
                q[arms[index]] = cfg["home_q"] if pose == "home" else press_q
                env.robot.set_joint_positions(q)
                env.robot.set_joint_velocities(np.zeros_like(q))
                controllers[index].apply_action(ArticulationAction(joint_positions=q))
                for floor, body in env.buttons.items():
                    position, orientation = body.get_world_pose()
                    position[:] = env.button_info[floor]["body_center"]
                    position[0] = env.button_info[floor]["rest_x"] + (press_travel[floor-24] if pose == "press35" else 0.)
                    body.set_world_pose(position=position, orientation=orientation)
                    set_light(env, floor, pose == "press35" and floor == 35)
            world.step(render=False)
            frozen_time = world.current_time
            frozen_q = [env.robot.get_joint_positions().copy() for env in envs]
            frozen_buttons = [np.array([b.get_world_pose()[0] for b in env.buttons.values()]) for env in envs]
            omni.usd.get_context().reset_renderer_accumulation()
            tiles = None
            for _ in range(10):
                rep.orchestrator.step(delta_time=0., pause_timeline=False, rt_subframes=4)
                try:
                    tiles = split(annotator.get_data())
                except ValueError:
                    continue
            if tiles is None:
                raise RuntimeError("No valid RGB after static settling")
            assert abs(world.current_time-frozen_time) < 1e-10
            for index, env in enumerate(envs):
                assert np.array_equal(env.robot.get_joint_positions(), frozen_q[index])
                assert np.array_equal(np.array([b.get_world_pose()[0] for b in env.buttons.values()]), frozen_buttons[index])
                for view_index, (view, camera_path) in enumerate((("wrist", env.wrist_camera_path), ("global", env.global_camera_path))):
                    rgb = tiles[2*index+view_index]
                    filename = f"env_{index}_{pose}_{view}.png"
                    Image.fromarray(rgb).save(args.output / filename)
                    camera = UsdGeom.Camera(world.stage.GetPrimAtPath(camera_path))
                    matrix = np.array(UsdGeom.Xformable(camera).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
                    matrix[3, :3] -= env.offset
                    attributes = {key: camera.GetPrim().GetAttribute(key).Get() for key in
                        ("focalLength", "horizontalAperture", "verticalAperture", "horizontalApertureOffset", "verticalApertureOffset", "projection", "focusDistance", "fStop")}
                    y = rgb @ np.array([.2126, .7152, .0722])
                    report["images"].append({"env_id": index, "pose": pose, "camera": view, "file": filename,
                        "environment_world_offset": env.offset.tolist(), "camera_attributes": attributes,
                        "camera_to_env_matrix": matrix.tolist(), "joint_positions": frozen_q[index].tolist(),
                        "button_world_positions": frozen_buttons[index].tolist(), "simulation_time": frozen_time,
                        "mean_luma": float(y.mean()), "p10_luma": float(np.percentile(y,10)),
                        "p90_luma": float(np.percentile(y,90)), "dark_fraction": float((y<10).mean()),
                        "clipped_luma_fraction": float((y>=250).mean()),
                        "clipped_any_channel_fraction": float((rgb.max(axis=2)>=250).mean()),
                        "local_light_intensities": {name: world.stage.GetPrimAtPath(env.prefix+"/Lights/"+name)
                            .GetAttribute("inputs:intensity").Get() for name in ("Ambient", "Ceiling", "Softbox")}})
                print(f"CAPTURE env={index} pose={pose} both RGB views", flush=True)
            (args.output / "report.json").write_text(json.dumps(report, indent=2))
        assert all(identity(Path(row["path"])) == row for row in identities)
        report["source_files_unchanged"] = True
        report["success"] = True
        (args.output / "report.json").write_text(json.dumps(report, indent=2))
        print(f"LIGHTING_RENDER_SUCCESS {args.output / 'report.json'}", flush=True)
    finally:
        app.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=5)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/scene.json")
    parser.add_argument("--snapshot", type=Path, default=ROOT / "outputs/edge_feedback/scene.usda")
    parser.add_argument("--source-episode", type=Path, default=ROOT / "datasets/piper_elevator_raw/episode_000011")
    parser.add_argument("--reference", type=Path, help="Single-env report.json for CPU pixel comparison")
    parser.add_argument("--compare-only", action="store_true", help="Read existing output report; never start Isaac Sim")
    args = parser.parse_args()
    for key in ("output", "config", "snapshot", "source_episode", "reference"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.resolve())
    if args.compare_only and args.reference is None:
        parser.error("--compare-only requires --reference")
    if not args.compare_only:
        render(args)
    if args.reference and not compare(args.output, args.reference):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
