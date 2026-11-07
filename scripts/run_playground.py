#!/usr/bin/env python3
"""Render the independent PiPER playground; never connects to real hardware.

The default scene is an empty workbench. Calibration objects require an explicit
flag. Optional replay renders measured joint feedback without advancing physics;
it is a visualization, not a new policy rollout or task-success evaluation.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pressb.runtime.recording import Video, write


def load_feedback(path):
    """Read PressB real-replay JSONL in radians/metres with its elapsed clock."""
    import numpy as np

    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if "observed" not in row:
            continue
        state = np.asarray(row["observed"], dtype=float)
        elapsed = float(row["elapsed_s"])
        if state.shape != (7,) or not np.isfinite(state).all() or not math.isfinite(elapsed):
            raise ValueError(f"Invalid measured state/time in {path}:{number}")
        if elapsed < 0 or (rows and elapsed <= rows[-1][0]):
            raise ValueError(f"Feedback times must be nonnegative and strictly increasing: {path}:{number}")
        if state[6] < 0:
            raise ValueError(f"Negative gripper opening in {path}:{number}")
        rows.append((elapsed, state))
    if not rows:
        raise ValueError(f"No observed states in {path}")
    return np.asarray([row[0] for row in rows]), np.stack([row[1] for row in rows])


def run(app, args, cfg, feedback):
    import numpy as np
    from PIL import Image
    from pxr import UsdGeom, UsdPhysics
    import omni.physx
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.sensors.camera import Camera
    from pressb.robots.piper.isaac import build_piper_runtime
    from pressb.robots.piper.kinematics import PiperKinematics
    from pressb.runtime.workbench import build_tripod
    from pressb.sensors.camera import camera_intrinsics_from_usd, make_camera, set_intrinsics
    from pressb.tasks.playground.scene import build_scene

    runtime = build_piper_runtime(
        cfg, build_scene, asset_root=ROOT,
        camera_hardware_builder=build_tripod if cfg.get("camera_tripod", True) else None,
    )
    world, robot, stage = runtime.world, runtime.robot, runtime.stage
    kinematics = PiperKinematics(ROOT / cfg["robot_urdf"], tip_offset=0.)
    transform_errors = {"link_translation_m": 0., "link_rotation_matrix": 0.,
                        "wrist_translation_m": 0., "wrist_rotation_matrix": 0.}
    cameras, specs = runtime.cameras, runtime.camera_specs
    overview_spec = cfg["overview_camera"]
    overview_path = make_camera(stage, runtime.scene["root_path"] + "/OverviewCamera", overview_spec, runtime.base)
    overview = Camera(prim_path=overview_path, name="overview", resolution=tuple(overview_spec["resolution"]), frequency=-1)
    overview.initialize()
    cameras["overview"] = overview
    specs["overview"] = (overview_path, overview_spec["resolution"])
    calibrated_specs = {"up": cfg["up_camera"], "wrist": cfg.get("wrist_camera"), "overview": overview_spec}
    # Isaac's Camera constructor enforces square pixels (fy=fx). Restore the
    # measured, slightly unequal focal lengths after every sensor initializes.
    for name, spec in calibrated_specs.items():
        if spec is not None:
            set_intrinsics(stage.GetPrimAtPath(specs[name][0]), spec)
    if not args.headless:
        from omni.kit.viewport.utility import get_active_viewport
        viewport = get_active_viewport()
        if viewport:
            viewport.set_active_camera(overview_path)

    def place_state(state):
        q = runtime.full_command(state)
        robot.set_joint_positions(q)
        robot.set_joint_velocities(np.zeros(robot.num_dof))
        runtime.controller.apply_action(ArticulationAction(joint_positions=q))
        return q

    def sync_and_check_transforms():
        # Tensor joint setters do not update USD link transforms by themselves.
        # Advance forward kinematics and publish it to rendering without a step.
        world.physics_sim_view.update_articulations_kinematic()
        omni.physx.get_physx_interface().update_transformations(False, True, True)
        cache = UsdGeom.XformCache()
        actual_q = robot.get_joint_positions()[runtime.indices[:6]]
        link_frames = kinematics.link_transforms(actual_q)
        expected_link6 = None
        current = {key: 0. for key in transform_errors}
        for name, local in link_frames.items():
            expected = local.copy()
            expected[:3, 3] += runtime.base
            actual = np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(f"/World/Piper/{name}"))).T
            current["link_translation_m"] = max(current["link_translation_m"], float(np.max(np.abs(actual[:3, 3]-expected[:3, 3]))))
            current["link_rotation_matrix"] = max(current["link_rotation_matrix"], float(np.max(np.abs(actual[:3, :3]-expected[:3, :3]))))
            if name == "link6":
                expected_link6 = expected
        registration = cfg.get("wrist_camera", {}).get("link6_from_camera_usd")
        if registration is not None:
            expected = expected_link6 @ np.asarray(registration)
            actual = np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(runtime.wrist_info["camera_path"]))).T
            current["wrist_translation_m"] = float(np.max(np.abs(actual[:3, 3]-expected[:3, 3])))
            current["wrist_rotation_matrix"] = float(np.max(np.abs(actual[:3, :3]-expected[:3, :3])))
        for key, value in current.items():
            transform_errors[key] = max(transform_errors[key], value)
        if any(value > 2e-4 for value in current.values()):
            raise RuntimeError(f"USD robot/camera poses disagree with URDF forward kinematics: {current}")

    # Rendering a measured pose must not let gravity or controller lag move it.
    # This runner also holds static snapshots exactly at the configured pose.
    target = place_state(runtime.initial_state if feedback is None else feedback[1][0])
    initial_world_time = float(world.current_time)
    sync_and_check_transforms()
    for _ in range(runtime.warmup_frames):
        world.render()

    def capture():
        sync_and_check_transforms()
        for _ in range(runtime.subframes):
            world.render()
        images = {}
        for name, camera in cameras.items():
            image = np.asarray(camera.get_rgba())
            width, height = specs[name][1]
            if image.shape != (height, width, 4):
                raise RuntimeError(f"{name} camera has no valid RGBA image: {image.shape}")
            images[name] = np.ascontiguousarray(image[..., :3], dtype=np.uint8)
        if float(world.current_time) != initial_world_time:
            raise RuntimeError("Physics time advanced during playground pose visualization")
        return images

    for name, image in capture().items():
        Image.fromarray(image).save(args.output / f"{name}.png")
    # Persist current drive targets as well as the scene's authored transforms.
    for name, value in zip(robot.dof_names, target):
        prim = stage.GetPrimAtPath(f"/World/Piper/joints/{name}")
        if prim.IsA(UsdPhysics.RevoluteJoint):
            UsdPhysics.DriveAPI(prim, "angular").GetTargetPositionAttr().Set(float(np.rad2deg(value)))
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            UsdPhysics.DriveAPI(prim, "linear").GetTargetPositionAttr().Set(float(value))
    stage.GetRootLayer().Export(str(args.output / "scene.usda"))
    cache = UsdGeom.XformCache()
    camera_report = {}
    for name, (path, size) in specs.items():
        width, height = size
        # Camera.get_intrinsics_matrix() hard-codes a centred principal point.
        # Read the actual USD projection, including both authored offsets.
        matrix = camera_intrinsics_from_usd(stage.GetPrimAtPath(path), size)
        spec = calibrated_specs[name]
        error = None
        if spec is not None:
            expected = np.array([[spec["fx"], 0., spec.get("cx", width/2)],
                                 [0., spec["fy"], spec.get("cy", height/2)], [0., 0., 1.]])
            error = float(np.max(np.abs(matrix-expected)))
            if error > 1e-3:
                raise RuntimeError(f"{name} USD intrinsics disagree with configuration by {error} pixels")
        camera_report[name] = {
            "path": path, "resolution": size, "intrinsics": matrix.tolist(),
            "intrinsics_source": "USD focal/aperture/aperture-offset attributes",
            "max_intrinsics_error_px": error,
            "world_from_camera_usd": np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path))).T.tolist(),
        }
    report = {
        "task": "playground", "config": cfg, "scene": runtime.scene,
        "reference_objects_enabled": args.reference_objects,
        "cameras": camera_report, "wrist_camera": runtime.wrist_info,
        "initial_q_target": target[runtime.indices].tolist(),
        "initial_q_actual": robot.get_joint_positions()[runtime.indices].tolist(),
        "physics_steps": 0, "hardware_access": False, "policy_or_RL_evaluation": False,
        "usd_forward_kinematics_max_errors": transform_errors,
        "presentation": "Configured pose" if feedback is None else "Measured real joint feedback; zero-order hold at original elapsed times",
    }
    if feedback is not None:
        report["feedback_source"] = {
            "path": str(args.replay_jsonl.resolve()),
            "sha256": hashlib.sha256(args.replay_jsonl.read_bytes()).hexdigest(),
            "samples": len(feedback[0]), "first_elapsed_s": float(feedback[0][0]),
            "last_elapsed_s": float(feedback[0][-1]),
        }

    if args.video:
        seconds = args.seconds if args.seconds is not None else (float(feedback[0][-1]) if feedback is not None else 2.)
        frames = max(1, math.ceil(seconds * runtime.fps))
        if feedback is not None and args.seconds is None:
            # Include the terminal observed state in the next display slot;
            # its rounding delay is less than one frame, without speeding up.
            frames += 1
        pipes = {}
        frame_records = []
        maximum_pose_error = 0.
        try:
            for name, (_, size) in specs.items():
                pipes[name] = Video(args.output / f"{name}.mp4", size, runtime.fps)
            for frame in range(frames):
                timestamp = frame / runtime.fps
                if feedback is not None:
                    sample = max(0, min(len(feedback[0]) - 1, int(np.searchsorted(feedback[0], timestamp, side="right") - 1)))
                    target = place_state(feedback[1][sample])
                    frame_records.append({"frame": frame, "video_seconds": timestamp, "source_sample": sample,
                                          "source_elapsed_s": float(feedback[0][sample])})
                images = capture()
                maximum_pose_error = max(maximum_pose_error, float(np.max(np.abs(robot.get_joint_positions() - target))))
                if maximum_pose_error > 2e-6:
                    raise RuntimeError(f"Measured pose changed during rendering: {maximum_pose_error}")
                for name, image in images.items():
                    pipes[name].add(image)
                if frame % max(1, round(runtime.fps * 5)) == 0:
                    print(json.dumps({"event": "render_progress", "frame": frame, "frames": frames}), flush=True)
        finally:
            for pipe in pipes.values():
                pipe.close()
        report["video"] = {"fps": runtime.fps, "frames": frames, "duration_s": frames / runtime.fps,
                           "playback_speed": 1., "max_pose_error": maximum_pose_error,
                           "paths": {name: str(pipe.path) for name, pipe in pipes.items()}}
        if feedback is not None:
            write(args.output / "video_frames.json", frame_records)
        for name, image in capture().items():
            Image.fromarray(image).save(args.output / f"final_{name}.png")
    write(args.output / "summary.json", report)
    print(json.dumps({"event": "playground_complete", "output": str(args.output)}), flush=True)
    if not args.headless:
        while app.is_running():
            world.render()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/piper_playground.json")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seconds", type=float, help="Video duration (default: 2 seconds, or full feedback span)")
    parser.add_argument("--video", action="store_true", help="Write one MP4 per camera at the configured FPS")
    parser.add_argument("--reference-objects", action="store_true", help="Show calibration-only objects instead of the default empty table")
    parser.add_argument("--replay-jsonl", type=Path, help="PressB replay log with observed states in rad/m and elapsed_s; implies --video")
    args = parser.parse_args()
    if args.gpu < 0 or (args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0)):
        parser.error("gpu must be nonnegative; seconds must be finite and positive")
    cfg = json.loads(args.config.read_text())
    if "initial_state_joint_rad_m" in cfg:
        initial = cfg["initial_state_joint_rad_m"]
        if len(initial) != 7 or not all(math.isfinite(float(value)) for value in initial):
            parser.error("initial_state_joint_rad_m must contain seven finite rad/m values")
        cfg["initial_joint_positions_rad"] = initial[:6]
        cfg["initial_gripper_width_m"] = initial[6]
    if args.reference_objects:
        if not cfg.get("calibration_objects"):
            parser.error("This config has no calibration_objects")
        cfg["objects"] = copy.deepcopy(cfg["calibration_objects"])
    feedback = load_feedback(args.replay_jsonl) if args.replay_jsonl else None
    if feedback is not None:
        if feedback[1][:, 6].max() > float(cfg["gripper_max_width_m"]):
            parser.error("Recorded gripper opening exceeds this scene's configured maximum")
        cfg["initial_joint_positions_rad"] = feedback[1][0, :6].tolist()
        cfg["initial_gripper_width_m"] = float(feedback[1][0, 6])
        args.video = True
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "config.json", cfg)
    status = {"state": "starting", "started_at": datetime.now(timezone.utc).isoformat()}
    write(args.output / "status.json", status)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    app = None
    try:
        # All Isaac/pxr imports intentionally happen after SimulationApp starts.
        from isaacsim import SimulationApp
        app = SimulationApp({
            "headless": args.headless, "active_gpu": args.gpu, "physics_gpu": args.gpu,
            "multi_gpu": False, "renderer": "RayTracedLighting", "anti_aliasing": 1, "fast_shutdown": True,
            "extra_args": ["--/app/asyncRendering=false", "--/rtx/ecoMode/enabled=false",
                           "--/validate/p2p/enabled=false", "--/validate/iommu/enabled=false",
                           "--/plugins/carb.tasking.plugin/threadCount=8", "--/plugins/omni.tbb.globalcontrol/maxThreadCount=8"],
        })
        run(app, args, cfg, feedback)
        status["state"] = "complete"
    except BaseException as error:
        status.update(state="failed", error=repr(error))
        traceback.print_exc()
        raise
    finally:
        status["finished_at"] = datetime.now(timezone.utc).isoformat()
        write(args.output / "status.json", status)
        if app is not None:
            app.app.post_quit(0 if status["state"] == "complete" else 1)
            app.close()


if __name__ == "__main__":
    main()
