#!/usr/bin/env python3
"""Render measured PiPER poses for camera calibration without advancing physics.

This is a pose visualization, not a grasp replay or a physical success test.
Only the articulation is placed at supplied measured joint states; no hardware
driver is imported, no CAN is accessed, and no object trajectory is synthesized.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from pressb.runtime.recording import write


def load_poses(path):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list) or not data:
        raise ValueError("Pose file must contain a nonempty list")
    names, poses = set(), []
    for value in data:
        if not isinstance(value, dict):
            raise ValueError("Pose entries must be objects")
        name = value.get("name")
        state = np.asarray(value.get("observed"), dtype=float)
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in names:
            raise ValueError("Pose names must be distinct filesystem-safe identifiers")
        if state.shape != (7,) or not np.isfinite(state).all():
            raise ValueError("Every observed pose must contain seven finite SI values")
        names.add(name)
        poses.append({**value, "observed": state.tolist()})
    return poses


def render(args, cfg, poses):
    from PIL import Image
    from pxr import UsdGeom
    import omni.physx
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.sensors.camera import Camera
    from pressb.robots.piper.isaac import build_piper_runtime
    from pressb.robots.piper.kinematics import PiperKinematics
    from pressb.runtime.workbench import build_tripod
    from pressb.tasks.playground.scene import build_scene
    from pressb.sensors.camera import make_camera, set_intrinsics, camera_intrinsics_from_usd

    runtime = build_piper_runtime(cfg, build_scene, asset_root=ROOT,
        camera_hardware_builder=build_tripod if cfg.get("camera_tripod", True) else None)
    world, robot, stage = runtime.world, runtime.robot, runtime.stage
    initial_time = float(world.current_time)
    kin = PiperKinematics(ROOT/cfg["robot_urdf"], tip_offset=0.)
    overview = cfg["overview_camera"]
    path = make_camera(stage, runtime.scene["root_path"]+"/OverviewCamera", overview, runtime.base)
    camera = Camera(prim_path=path, name="overview", resolution=tuple(overview["resolution"]), frequency=-1)
    camera.initialize()
    runtime.cameras["overview"] = camera
    runtime.camera_specs["overview"] = (path, overview["resolution"])
    for name in runtime.cameras:
        set_intrinsics(stage.GetPrimAtPath(runtime.camera_specs[name][0]), cfg[name+"_camera"])
    reports = []
    for pose in poses:
        state = np.asarray(pose["observed"])
        full = runtime.full_command(state)
        robot.set_joint_positions(full)
        robot.set_joint_velocities(np.zeros(robot.num_dof))
        runtime.controller.apply_action(ArticulationAction(joint_positions=full))
        world.physics_sim_view.update_articulations_kinematic()
        omni.physx.get_physx_interface().update_transformations(False, True, True)
        # Rendering-only flush: remove buffered images from the preceding pose.
        for _ in range(max(16, runtime.warmup_frames)):
            world.render()
        if float(world.current_time) != initial_time:
            raise RuntimeError("Physics advanced during pose-only visualization")
        actual = robot.get_joint_positions()
        if np.max(np.abs(actual-full)) > 2e-6:
            raise RuntimeError("Measured pose changed during rendering")
        cache = UsdGeom.XformCache()
        expected_link = kin.fk(state[:6])
        expected_link[:3, 3] += runtime.base
        actual_link = np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath("/World/Piper/link6"))).T
        if np.max(np.abs(actual_link-expected_link)) > 2e-4:
            raise RuntimeError("Rendered link6 transform disagrees with measured-state FK")
        result = {**pose, "physics_steps": 0, "pose_visualization_only": True,
                  "max_link6_transform_error": float(np.max(np.abs(actual_link-expected_link))), "cameras": {}}
        for name, cam in runtime.cameras.items():
            image = np.asarray(cam.get_rgba())
            path, size = runtime.camera_specs[name]
            if image.shape != (size[1], size[0], 4):
                raise RuntimeError(f"Invalid {name} image shape: {image.shape}")
            filename = args.output/(pose["name"]+"_"+name+".png")
            Image.fromarray(image[..., :3]).save(filename)
            camera_prim = stage.GetPrimAtPath(path)
            result["cameras"][name] = dict(image=str(filename), resolution=size,
                intrinsics=camera_intrinsics_from_usd(camera_prim, size).tolist(),
                world_from_camera_usd=np.asarray(cache.GetLocalToWorldTransform(camera_prim)).T.tolist())
        reports.append(result)
        print(json.dumps(dict(event="pose_rendered", name=pose["name"])), flush=True)
    write(args.output/"summary.json", dict(physics_steps=0, hardware_access=False,
        pose_visualization_only=True, grasp_validation=False, config=cfg,
        poses_file_sha256=hashlib.sha256(args.poses.read_bytes()).hexdigest(), poses=reports))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/piper_playground.json")
    parser.add_argument("--poses", type=Path, required=True, help="List of {name, observed:[6 radians, width meters], ...provenance}")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    poses = load_poses(args.poses)
    cfg = json.loads(args.config.read_text())
    cfg["objects"] = copy.deepcopy(cfg["calibration_objects"])
    cfg["initial_joint_positions_rad"] = poses[0]["observed"][:6]
    cfg["initial_gripper_width_m"] = poses[0]["observed"][6]
    cfg["initial_state_joint_rad_m"] = poses[0]["observed"]
    from pressb.robots.piper.kinematics import PiperKinematics
    kin = PiperKinematics(ROOT/cfg["robot_urdf"])
    for pose in poses:
        state = np.asarray(pose["observed"])
        if np.any(state[:6] < kin.lower) or np.any(state[:6] > kin.upper) or not 0 <= state[6] <= cfg["gripper_max_width_m"]:
            parser.error(f"Pose {pose['name']} exceeds URDF/gripper bounds")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output/"config.json", cfg)
    write(args.output/"poses.json", poses)
    if args.validate_only:
        print(json.dumps(dict(validated=True, poses=len(poses), hardware_access=False, physics_steps=0)))
        return
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaacsim import SimulationApp
    app = SimulationApp(dict(headless=True, active_gpu=args.gpu, physics_gpu=args.gpu,
        multi_gpu=False, renderer="RayTracedLighting", anti_aliasing=1, fast_shutdown=True,
        extra_args=["--/app/asyncRendering=false", "--/rtx/ecoMode/enabled=false",
                    "--/validate/p2p/enabled=false", "--/validate/iommu/enabled=false",
                    "--/plugins/carb.tasking.plugin/threadCount=8", "--/plugins/omni.tbb.globalcontrol/maxThreadCount=8"]))
    try:
        render(args, cfg, poses)
    finally:
        app.close()


if __name__ == "__main__":
    main()
