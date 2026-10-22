#!/usr/bin/env python3
"""Build the separate PAP scene, replay recorded motion or run a native Wall-X policy.

The object is a free PhysX rigid body throughout replay. No object trajectory,
attachment constraint, or placement teleport is used to manufacture a grasp.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


class Video:
    def __init__(self, path, size, fps):
        self.path, self.count = Path(path), 0
        self.process = subprocess.Popen([
            "ffmpeg", "-v", "error", "-nostdin", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s:v", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "pipe:0", "-an",
            "-c:v", "libx264", "-preset", "fast", "-crf", "19", "-pix_fmt", "yuv420p",
            "-threads", "1", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)

    def add(self, rgb):
        self.process.stdin.write(rgb.tobytes())
        self.count += 1

    def close(self):
        if not self.process.stdin.closed:
            self.process.stdin.close()
        if self.process.wait(timeout=60):
            raise RuntimeError(f"Encoding failed: {self.path}")


def set_intrinsics(prim, spec):
    from pxr import UsdGeom
    camera = UsdGeom.Camera(prim)
    width, height = spec["resolution"]
    focal = 2.
    horizontal = focal * width / spec["fx"]
    vertical = focal * height / spec["fy"]
    camera.CreateFocalLengthAttr(focal)
    camera.CreateHorizontalApertureAttr(horizontal)
    camera.CreateVerticalApertureAttr(vertical)
    camera.CreateHorizontalApertureOffsetAttr((width / 2 - spec.get("cx", width/2)) / width * horizontal)
    camera.CreateVerticalApertureOffsetAttr((spec.get("cy", height/2) - height/2) / height * vertical)
    camera.CreateFStopAttr(0.)


def make_camera(stage, path, spec, base):
    import numpy as np
    from pxr import UsdGeom, Gf
    prim = UsdGeom.Camera.Define(stage, path)
    if "base_from_camera_usd" in spec:
        transform = np.asarray(spec["base_from_camera_usd"], dtype=float).copy()
        transform[:3, 3] += base
        matrix = Gf.Matrix4d(*transform.T.reshape(-1).tolist())
    else:
        eye, target = np.asarray(spec["eye_base_m"]) + base, np.asarray(spec["target_base_m"]) + base
        matrix = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse()
    prim.AddTransformOp().Set(matrix)
    prim.CreateClippingRangeAttr(Gf.Vec2f(.005, 20.))
    set_intrinsics(prim.GetPrim(), spec)
    return str(prim.GetPath())


def run(app, args, cfg, trajectory, policy_client=None):
    import numpy as np
    from PIL import Image
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema, PhysicsSchemaTools
    import omni.physx
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.sensors.camera import Camera
    from pressb.pick_place.scene import build_scene, build_tripod
    from pressb.pick_place.task import PickPlaceOutcome
    from pressb.wrist_camera import apply_black_materials, attach_wrist_camera
    import carb

    # Extra render samples never advance the physical trajectory or video time.
    rendering = cfg.get("rendering", {})
    allowed_rendering = {"warmup_frames", "subframes_per_frame", "direct_samples_per_pixel", "reflection_samples_per_pixel"}
    if not isinstance(rendering, dict) or set(rendering) - allowed_rendering:
        raise ValueError(f"rendering supports only {sorted(allowed_rendering)}")
    warmup_frames = rendering.get("warmup_frames", 8)
    subframes = rendering.get("subframes_per_frame", 1)
    if type(warmup_frames) is not int or not 8 <= warmup_frames <= 256:
        raise ValueError("rendering.warmup_frames must be an integer in [8, 256]")
    if type(subframes) is not int or not 1 <= subframes <= 8:
        raise ValueError("rendering.subframes_per_frame must be an integer in [1, 8]")
    renderer_settings = carb.settings.get_settings()
    sample_paths = {"direct_samples_per_pixel": "/rtx/directLighting/sampledLighting/samplesPerPixel",
                    "reflection_samples_per_pixel": "/rtx/reflections/sampledLighting/samplesPerPixel"}
    for key, setting in sample_paths.items():
        if key in rendering:
            value = rendering[key]
            if type(value) is not int or value not in (1, 2, 4, 8):
                raise ValueError(f"rendering.{key} must be 1, 2, 4 or 8")
            renderer_settings.set_int(setting, value)
    from pressb.kinematics import PiperKinematics

    kin = PiperKinematics(ROOT / cfg["robot_urdf"], tip_offset=.1358)
    for q in (trajectory["initial"][:6], *trajectory["commands"][:, :6]):
        if np.any(q < kin.lower-1e-3) or np.any(q > kin.upper+1e-3):
            raise ValueError("Dataset arm command exceeds URDF limit; check units before replay")
    dt, fps = cfg["physics_dt"], cfg["fps"]
    stride = round(1 / dt / fps)
    if abs(stride*dt-1/fps) > 1e-9:
        raise ValueError("Video fps must divide physics rate")
    world = World(stage_units_in_meters=1., physics_dt=dt, rendering_dt=1/fps,
                  backend="numpy", device="cpu")
    world.get_physics_context().set_physx_update_transformations_settings(
        update_to_usd=True, update_velocities_to_usd=True)
    stage = world.stage
    scene = build_scene(stage, cfg)
    base = np.asarray(scene["robot_base_position_world"])
    add_reference_to_stage(str((ROOT/cfg["robot_usd"]).resolve()), "/World/Piper")
    robot_prim = stage.GetPrimAtPath("/World/Piper")
    transform = UsdGeom.Xformable(robot_prim)
    transform.ClearXformOpOrder()
    transform.AddTranslateOp().Set(Gf.Vec3d(*base))
    apply_black_materials(stage)
    width_max = cfg["gripper_max_width_m"]
    for prim in Usd.PrimRange(robot_prim):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation = PhysxSchema.PhysxArticulationAPI.Apply(prim)
            articulation.CreateSolverPositionIterationCountAttr(32)
            articulation.CreateSolverVelocityIterationCountAttr(8)
            articulation.CreateEnabledSelfCollisionsAttr(True)
        if prim.IsA(UsdPhysics.RevoluteJoint):
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
            drive.CreateStiffnessAttr(cfg["arm_stiffness"])
            drive.CreateDampingAttr(cfg["arm_damping"])
            drive.CreateMaxForceAttr(cfg["arm_max_effort"])
        if prim.IsA(UsdPhysics.PrismaticJoint):
            drive = UsdPhysics.DriveAPI.Apply(prim, "linear")
            drive.CreateStiffnessAttr(cfg["gripper_stiffness"])
            drive.CreateDampingAttr(cfg["gripper_damping"])
            drive.CreateMaxForceAttr(cfg["gripper_max_effort"])
            joint = UsdPhysics.PrismaticJoint(prim)
            if prim.GetName() == "joint7":
                joint.CreateUpperLimitAttr(width_max/2)
            elif prim.GetName() == "joint8":
                joint.CreateLowerLimitAttr(-width_max/2)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            api.CreateContactOffsetAttr(.0005)
            api.CreateRestOffsetAttr(0.)
    wrist_info = attach_wrist_camera(stage, "/World/Piper/link6", cfg)
    wrist_prim = stage.GetPrimAtPath(wrist_info["camera_path"])
    if "wrist_camera" in cfg:
        spec = cfg["wrist_camera"]
        set_intrinsics(wrist_prim, spec)
        if "link6_from_camera_usd" in spec:
            # A world-independent sensor transform relative to link6. Keep the
            # official housing and bracket geometry separate from registration.
            link6_matrix = np.asarray(spec["link6_from_camera_usd"])
            parent = wrist_prim.GetParent()
            cache = UsdGeom.XformCache()
            parent_world = np.array(cache.GetLocalToWorldTransform(parent)).T
            link_world = np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath("/World/Piper/link6"))).T
            local = np.linalg.inv(parent_world) @ link_world @ link6_matrix
            xf = UsdGeom.Xformable(wrist_prim)
            xf.ClearXformOpOrder()
            xf.AddTransformOp().Set(Gf.Matrix4d(*local.T.reshape(-1).tolist()))
        wrist_info["pap_sensor_registration"] = spec
    finger_material = UsdShade.Material.Define(stage, "/World/PickPlace/Looks/FingerContact")
    material_api = UsdPhysics.MaterialAPI.Apply(finger_material.GetPrim())
    material_api.CreateStaticFrictionAttr(cfg["finger_static_friction"])
    material_api.CreateDynamicFrictionAttr(cfg["finger_dynamic_friction"])
    material_api.CreateRestitutionAttr(0.)
    for prim in Usd.PrimRange(robot_prim):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
        if prim.HasAPI(UsdPhysics.CollisionAPI) and any(f"/link{i}/" in str(prim.GetPath()) for i in (7, 8)):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(finger_material, materialPurpose="physics")
    up_path = make_camera(stage, "/World/PickPlace/UpCamera", cfg["up_camera"], base)
    overview_path = make_camera(stage, "/World/PickPlace/OverviewCamera", cfg["overview_camera"], base)
    camera_cfg = cfg["up_camera"]
    if "eye_base_m" in camera_cfg:
        scene["table_camera_hardware"] = build_tripod(stage, np.asarray(camera_cfg["eye_base_m"])+base,
                     np.asarray(camera_cfg["target_base_m"])+base, cfg["table_height"])
    robot = world.scene.add(SingleArticulation(prim_path="/World/Piper", name="piper"))
    cube = world.scene.add(SingleRigidPrim(prim_path=scene["object_path"], name="object"))
    camera_specs = {
        "up": (up_path, cfg["up_camera"]["resolution"]),
        "wrist": (wrist_info["camera_path"], cfg.get("wrist_camera", {}).get("resolution", cfg["wrist_camera_resolution"])),
        "overview": (overview_path, cfg["overview_camera"]["resolution"]),
    }
    if args.evaluation_plan is not None:
        # The evaluated policy receives the same two sensors as single episodes.
        # The third, human-facing overview render product is unnecessary here.
        camera_specs.pop("overview")
    cameras = {name: Camera(prim_path=path, name=name, resolution=tuple(size), frequency=-1)
               for name, (path, size) in camera_specs.items()}
    if not args.headless:
        from omni.kit.viewport.utility import get_active_viewport
        viewport = get_active_viewport()
        if viewport:
            viewport.set_active_camera(overview_path)
    world.reset()
    for camera in cameras.values():
        camera.initialize()
    indices = np.array([robot.get_dof_index(f"joint{i}") for i in range(1, 9)])

    def full_command(value):
        result = np.zeros(robot.num_dof)
        result[indices] = np.r_[value[:6], value[6]/2, -value[6]/2]
        return result

    initial = full_command(trajectory["initial"])
    robot.set_joint_positions(initial)
    robot.set_joint_velocities(np.zeros(robot.num_dof))
    controller = robot.get_articulation_controller()
    kps, kds = np.zeros(robot.num_dof), np.zeros(robot.num_dof)
    kps[indices[:6]], kds[indices[:6]] = cfg["arm_stiffness"], cfg["arm_damping"]
    kps[indices[6:]], kds[indices[6:]] = cfg["gripper_stiffness"], cfg["gripper_damping"]
    controller.set_gains(kps=kps, kds=kds)
    controller.apply_action(ArticulationAction(joint_positions=initial))
    tick, contact_counts, contact_events = [0], {"link7": 0, "link8": 0}, []
    finger_contact = [False]

    def on_contact(headers, data):
        for header in headers:
            actors = [str(PhysicsSchemaTools.intToSdfPath(value)) for value in (header.actor0, header.actor1)]
            if scene["object_path"] not in actors or not header.num_contact_data:
                continue
            for name in contact_counts:
                if f"/World/Piper/{name}" in actors:
                    finger_contact[0] = True
                    contact_counts[name] += 1
                    impulse = sum(float(np.linalg.norm(data[i].impulse)) for i in
                                  range(header.contact_data_offset, header.contact_data_offset+header.num_contact_data))
                    contact_events.append(dict(tick=tick[0], finger=name, impulse_ns=impulse))
    subscription = omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(on_contact)
    base_cfg, base_scene = copy.deepcopy(cfg), copy.deepcopy(scene)
    cube_initial_quaternion = np.array(cube.get_world_pose()[1], copy=True)

    def execute_episode(output, episode=None):
        nonlocal cfg, scene
        episode_started = time.perf_counter()
        if episode is not None:
            cfg, scene = copy.deepcopy(base_cfg), copy.deepcopy(base_scene)
            object_key = "object" if "object" in cfg else "cube"
            cfg[object_key]["center_xy"] = episode["center_xy"]
            cfg.setdefault("materials", {}).setdefault("RedCube", {})["color"] = episode["color_rgb"]
            if object_key == "object":
                cfg["object"]["color"] = episode["color_rgb"]
            position = np.asarray(base_scene["object_initial_position_world"], dtype=float).copy()
            position[:2] = np.asarray(episode["center_xy"]) + base[:2]
            scene["object_initial_position_world"] = position.tolist()
            scene["cube_initial_position_world"] = position.tolist()
            material = scene["appearance"]["object_material"]
            material["color"] = episode["color_rgb"]
            scene["appearance"]["materials"]["RedCube"]["color"] = episode["color_rgb"]
            for suffix, input_name in (("Mdl", "diffuse_color_constant"), ("Preview", "diffuseColor")):
                shader = UsdShade.Shader(stage.GetPrimAtPath(material["path"] + "/" + suffix))
                shader.GetInput(input_name).Set(Gf.Vec3f(*episode["color_rgb"]))
            robot.set_joint_positions(initial)
            robot.set_joint_velocities(np.zeros(robot.num_dof))
            controller.apply_action(ArticulationAction(joint_positions=initial))
            cube.set_world_pose(position, cube_initial_quaternion)
            cube.set_linear_velocity(np.zeros(3))
            cube.set_angular_velocity(np.zeros(3))
            tick[0] = 0
            contact_counts.update(link7=0, link8=0)
            contact_events.clear()
            finger_contact[0] = False
            write(output/"config.json", cfg)
            write(output/"status.json", dict(state="running", started_at=datetime.now(timezone.utc).isoformat(),
                episode=episode))
        for _ in range(24):
            world.step(render=False)
        for _ in range(warmup_frames):
            world.render()
        for name, camera in cameras.items():
            rgb = np.asarray(camera.get_rgba())[..., :3]
            if rgb.ndim != 3 or rgb.size == 0:
                raise RuntimeError(f"No RGB data: {name}")
            Image.fromarray(rgb).save(output / f"initial_{name}.jpg", quality=95)
        if episode is not None:
            # Reset warm-up contacts must not contribute to this task's outcome.
            contact_counts.update(link7=0, link8=0)
            contact_events.clear()
            finger_contact[0] = False
        # Persist initial drive values so opening the scene and pressing Play keeps
        # the dataset initial pose. Source asset layers are never modified.
        for name, value in zip(robot.dof_names, initial):
            prim = stage.GetPrimAtPath(f"/World/Piper/joints/{name}")
            if prim.IsA(UsdPhysics.RevoluteJoint):
                UsdPhysics.DriveAPI(prim, "angular").GetTargetPositionAttr().Set(float(np.rad2deg(value)))
            elif prim.IsA(UsdPhysics.PrismaticJoint):
                UsdPhysics.DriveAPI(prim, "linear").GetTargetPositionAttr().Set(float(value))
        if episode is None:
            stage.GetRootLayer().Export(str(output / "scene.usda"))
        elif not (args.output / "scene.usda").exists():
            stage.GetRootLayer().Export(str(args.output / "scene.usda"))
        report = dict(task="piper_pick_place", mode=args.mode, scene=scene, config=cfg,
                      trajectory=str(args.trajectory.resolve()), trajectory_semantics=trajectory["metadata"],
                      source_frames=len(trajectory["times"]), initial_q_target=initial[indices].tolist(),
                      initial_q_actual=robot.get_joint_positions()[indices].tolist(), wrist_camera=wrist_info,
                      simulation="Isaac Sim 5.0 / PhysX; 120 Hz", object_attachment_used=False,
                      object_teleport_after_initialization=False, policy_or_RL_evaluation=args.mode == "policy")
        if episode is not None:
            report["evaluation_episode"] = episode
            report["evaluation_status"] = "running"
            report["initial_object_position_world"] = cube.get_world_pose()[0].tolist()
            report["initial_object_velocity"] = np.r_[cube.get_linear_velocity(), cube.get_angular_velocity()].tolist()
            report["initial_joint_velocities"] = robot.get_joint_velocities()[indices].tolist()
            report["reset"] = dict(arm_pose_from_dataset_initial=True, joint_velocities_zeroed=True,
                object_pose_and_velocities_reset=True, settling_physics_ticks=24,
                smoothing_state="measured state after settling", policy_buffer="new per episode",
                outcome_and_contacts="new per episode")
        report["rendering"] = dict(warmup_frames=warmup_frames, subframes_per_frame=subframes,
            **{key: renderer_settings.get(setting) for key, setting in sample_paths.items()})
        cache = UsdGeom.XformCache()
        report["cameras"] = {name: dict(path=path, resolution=size,
            intrinsics=np.asarray(cameras[name].get_intrinsics_matrix()).tolist(),
            world_from_camera_usd_at_initial=np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path))).T.tolist())
            for name, (path, size) in camera_specs.items()}
        if args.mode == "snapshot":
            write(output / "report.json", report)
            if args.hold:
                while app.is_running():
                    world.render()
            return
        cavity = scene["box_inside"]
        box = dict(center_world=cavity["origin_world"], yaw_deg=cavity["yaw_deg"],
                   inner_size_xy=(np.asarray(cavity["max_local"])-np.asarray(cavity["min_local"]))[:2].tolist(),
                   floor_z_world=cavity["origin_world"][2]+cavity["min_local"][2],
                   rim_z_world=cavity["origin_world"][2]+cavity["max_local"][2])
        outcome = PickPlaceOutcome(scene["object_geometry"], cfg["table_height"], box)
        times, commands = trajectory["times"], trajectory["commands"]
        policy, postprocessor = None, None
        if args.mode == "policy":
            from pressb.pick_place.wallx import WallXChunkPolicy
            from pressb.action_postprocessing import JointActionPostprocessor
            if policy_client is None or fps != 30 or abs(dt-1/120) > 1e-9:
                raise ValueError("Piper Wall-X evaluation requires a connected policy and 30 Hz control / 120 Hz physics")
            policy = WallXChunkPolicy(policy_client, kin, cfg, output,
                execute_steps=args.policy_execute_steps, prompt=args.policy_prompt,
                dataset_name=args.policy_dataset_name, arm_limit_mode=args.policy_arm_limit_mode,
                gripper_limit_mode=args.policy_gripper_limit_mode)
            postprocessor = JointActionPostprocessor(args.policy_smoothing, args.policy_smoothing_alpha)
            measured = robot.get_joint_positions()[indices]
            postprocessor.reset(np.r_[measured[:6], measured[6]-measured[7]])
            total_ticks = math.floor(args.duration*fps + 1e-9)*stride
        else:
            duration = min(args.duration, float(times[-1]))
            total_ticks = round(duration/dt)
        # Source timestamps may store 30 Hz times as float32. Report the time of
        # the actual final physics tick, rather than a rounded source timestamp.
        duration = total_ticks*dt
        videos = {name: Video(output/f"{name}.mp4", spec[1], fps) for name, spec in camera_specs.items()
                  if episode is None or episode.get("record_video", False)}
        records, result = [], {}
        target = initial.copy()
        raw_target = initial.copy()
        policy_indices = dict(policy_request=-1, policy_action_index=-1)
        try:
            for step in range(total_ticks+1):
                seconds = step*dt
                if policy is not None:
                    # Request only on a control boundary. The last rendered camera
                    # image and measured state refer to that exact physical time.
                    # Waiting on the network does not call world.step or app.update.
                    if step and (step-1) % stride == 0:
                        observation = {}
                        if policy.needs_observation:
                            observation = dict(joints=robot.get_joint_positions()[indices],
                                up_rgb=np.ascontiguousarray(cameras["up"].get_rgba()[..., :3]),
                                wrist_rgb=np.ascontiguousarray(cameras["wrist"].get_rgba()[..., :3]))
                        try:
                            command, policy_indices = policy.command((step-1)*dt, **observation)
                        except ValueError as error:
                            # Decode failures are a model outcome, while transport,
                            # sensor and simulator failures invalidate the run.
                            text = str(error)
                            action_error = text.startswith(("Policy arm limit exceeded", "Policy gripper command exceeds",
                                                           "Expected finite Wall-X actions with shape"))
                            if episode is not None and action_error:
                                report.update(evaluation_error=f"{type(error).__name__}: {error}",
                                              failure_reason="invalid_policy_action", evaluation_status="complete")
                                break
                            raise
                        raw_target = full_command(command)
                        target = full_command(postprocessor.apply(command))
                else:
                    value = np.array([np.interp(seconds, times, commands[:, j]) for j in range(7)])
                    target = full_command(value) if step else initial.copy()
                if step:
                    tick[0] = step
                    finger_contact[0] = False
                    controller.apply_action(ArticulationAction(joint_positions=target))
                    world.step(render=False)
                q = robot.get_joint_positions()[indices]
                position, quaternion = cube.get_world_pose()
                velocity = np.r_[cube.get_linear_velocity(), cube.get_angular_velocity()]
                result = outcome.update(seconds, position, quaternion, velocity, float(q[6]-q[7]),
                                        finger_contact=finger_contact[0])
                records.append(dict(time_s=seconds, q_actual=q.copy(), q_target=target[indices],
                                    cube_position=position.copy(), cube_quaternion=quaternion.copy(),
                                    cube_velocity=velocity.copy(),
                                    **(dict(q_target_raw=raw_target[indices], **policy_indices) if policy else {})))
                stop_success = (episode is not None and args.evaluation_data.get("stop_on_success", True)
                                and step % stride == 0 and result.get("success", False) and all(contact_counts.values()))
                if step % stride == 0 or step == total_ticks:
                    for _ in range(subframes):
                        world.render()
                    for name, camera in cameras.items():
                        rgb = np.ascontiguousarray(camera.get_rgba()[..., :3])
                        if name in videos:
                            videos[name].add(rgb)
                        if step == total_ticks or stop_success:
                            Image.fromarray(rgb).save(output/f"final_{name}.jpg", quality=95)
                if stop_success:
                    report["termination"] = "success"
                    break
                if step % 120 == 0:
                    print(json.dumps(dict(event=args.mode, seconds=seconds, cube=position.tolist(), **result,
                        **(dict(policy_requests=policy.requests) if policy else {}))), flush=True)
        except BaseException as error:
            report["evaluation_error"] = f"{type(error).__name__}: {error}"
            if episode is not None:
                report["evaluation_status"] = "interrupted"
            raise
        finally:
            for video in videos.values():
                video.close()
            if records:
                arrays = {key: np.asarray([row[key] for row in records]) for key in records[0]}
                np.savez_compressed(output/"physics.npz", **arrays)
                report.update(duration_s=float(records[-1]["time_s"]), physics_samples=len(records), physical_outcome=result,
                    requested_duration_s=args.duration, source_duration_s=None if policy else float(times[-1]),
                    contact_counts=contact_counts, gripper_contact_both_fingers=all(contact_counts.values()),
                    video_frames={name:video.count for name,video in videos.items()},
                    maximum_arm_tracking_error_rad=float(np.max(np.abs(arrays["q_actual"][:,:6]-arrays["q_target"][:,:6]))))
            report["physical_success"] = bool(result.get("success", False) and all(contact_counts.values()))
            if episode is not None:
                if report["evaluation_status"] == "running":
                    report["evaluation_status"] = "complete"
                report.setdefault("termination", "horizon" if not report.get("evaluation_error") else "error")
                report.setdefault("failure_reason", None if report["physical_success"] else "timeout")
                report["wall_time_s"] = time.perf_counter() - episode_started
                for name, camera in cameras.items():
                    Image.fromarray(np.ascontiguousarray(camera.get_rgba()[..., :3])).save(
                        output/f"final_{name}.jpg", quality=95)
            if policy:
                report.update(policy=policy.summary(), expert_actions_used=False, control_period_s=1/fps,
                    physics_ticks_per_action=stride, planned_duration_s=duration,
                    policy_note="Absolute policy targets optionally pass through joint EMA at 30 Hz, then are held for four physics ticks. Raw and sent targets are recorded separately. Fresh simulated RGB and measured joints are requested after each chunk prefix; no replay commands or object attachment are used.")
                report["policy"]["smoothing"] = postprocessor.summary()
            else:
                report["replay_note"] = "Recorded commands are linearly resampled to physics time; observed motion and object dynamics are simulated. The explicitly selected duration may omit remaining dataset frames."
            write(output/"contacts.json", contact_events)
            write(output/"report.json", report)
            if episode is not None:
                write(output/"status.json", dict(state=report["evaluation_status"],
                    physical_success=report["physical_success"], finished_at=datetime.now(timezone.utc).isoformat()))
        if args.hold:
            while app.is_running():
                world.render()

        return report

    if args.evaluation_plan is None:
        execute_episode(args.output)
        return
    episodes = args.evaluation_data["episodes"]
    results = []
    for index, episode in enumerate(episodes):
        output = args.output / "episodes" / episode["episode_id"]
        previous = output / "report.json"
        if args.resume and previous.exists():
            saved = json.loads(previous.read_text())
            if saved.get("evaluation_episode") != episode:
                raise ValueError(f"Existing episode differs from evaluation plan: {output}")
            if saved.get("evaluation_status") == "complete":
                results.append(saved)
                continue
        if output.exists():
            # Preserve interrupted evidence, then restart this episode from its
            # prescribed initialization. It has not entered the SR denominator.
            if not args.resume:
                raise FileExistsError(output)
            output.rename(output.with_name(output.name + ".interrupted_" + str(time.time_ns())))
        output.mkdir(parents=True)
        write(args.output/"progress.json", dict(state="running", completed=len(results), total=len(episodes),
            current_episode=episode, updated_at=datetime.now(timezone.utc).isoformat()))
        report = execute_episode(output, episode)
        results.append(report)
        row = dict(episode=episode, physical_success=report["physical_success"],
                   evaluation_status=report["evaluation_status"], failure_reason=report["failure_reason"],
                   duration_s=report["duration_s"], wall_time_s=report["wall_time_s"],
                   physical_outcome=report["physical_outcome"], report=str(output/"report.json"))
        with (args.output/"results.jsonl").open("a") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
        print(json.dumps(dict(event="evaluation_episode_complete", index=index+1, total=len(episodes), **row)), flush=True)
        write(args.output/"progress.json", dict(state="running", completed=len(results), total=len(episodes),
            successes=sum(int(row["physical_success"]) for row in results),
            last_episode=episode, updated_at=datetime.now(timezone.utc).isoformat()))
    write(args.output/"progress.json", dict(state="complete", completed=len(results), total=len(episodes),
        successes=sum(int(row["physical_success"]) for row in results),
        updated_at=datetime.now(timezone.utc).isoformat()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/piper_pick_place.json")
    parser.add_argument("--trajectory", type=Path,
                        help="Recorded episode NPZ; defaults to the episode_source in the scene configuration")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation-plan", type=Path,
                        help="JSON episode list for persistent-world policy evaluation")
    parser.add_argument("--resume", action="store_true",
                        help="Resume an identical evaluation plan; completed reports are retained")
    parser.add_argument("--mode", choices=("snapshot", "replay", "policy"), default="replay")
    parser.add_argument("--source", choices=("action", "observation", "observed_arm_action_gripper"),
                        default="observed_arm_action_gripper",
                        help="Default follows measured arm poses and recorded gripper effort-producing position commands")
    parser.add_argument("--duration", type=float, default=8.5)
    parser.add_argument("--policy-endpoint", default="ws://127.0.0.1:8094")
    parser.add_argument("--policy-execute-steps", type=int, default=32,
                        help="Execute this many of the 32 predicted 30 Hz actions before requesting fresh observations")
    parser.add_argument("--policy-timeout", type=float, default=180.)
    parser.add_argument("--policy-prompt", help="Defaults to the scene task instruction")
    parser.add_argument("--policy-dataset-name", default="lerobot/piper")
    parser.add_argument("--policy-arm-limit-mode", choices=("error", "clip"), default="error")
    parser.add_argument("--policy-gripper-limit-mode", choices=("error", "clip"), default="clip")
    parser.add_argument("--policy-smoothing", choices=("none", "joint_ema"), default=None,
                        help="Override policy_execution.smoothing.mode; defaults to none; gripper unchanged")
    parser.add_argument("--policy-smoothing-alpha", type=float, default=None,
                        help="EMA weight of each new target, in (0,1]; smaller means stronger smoothing; 1 is unchanged")
    parser.add_argument("--policy-token-env", default="PRESSB_RL_TOKEN", help="Environment variable containing an optional server API key")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--hold", action="store_true")
    args = parser.parse_args()
    args.evaluation_data = None
    if args.resume and args.evaluation_plan is None:
        parser.error("--resume requires --evaluation-plan")
    if args.evaluation_plan is not None:
        if args.mode != "policy" or args.hold:
            parser.error("An evaluation plan requires --mode policy without --hold")
        try:
            args.evaluation_data = json.loads(args.evaluation_plan.read_text())
            episodes = args.evaluation_data["episodes"]
            if not isinstance(episodes, list) or not episodes:
                raise ValueError("episodes must be a nonempty list")
            ids = set()
            for episode in episodes:
                identifier = episode["episode_id"]
                if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", identifier) or identifier in ids:
                    raise ValueError("episode_id must be unique and contain only letters, digits, _ or -")
                ids.add(identifier)
                if not isinstance(episode["group"], str) or not isinstance(episode["color"], str):
                    raise ValueError("group and color must be strings")
                xy, rgb = episode["center_xy"], episode["color_rgb"]
                if len(xy) != 2 or not all(math.isfinite(float(v)) for v in xy):
                    raise ValueError("center_xy must contain two finite coordinates")
                if len(rgb) != 3 or not all(math.isfinite(float(v)) and 0 <= float(v) <= 1 for v in rgb):
                    raise ValueError("color_rgb must contain three linear RGB values in [0, 1]")
                if type(episode.get("record_video", False)) is not bool:
                    raise ValueError("record_video must be a boolean")
            if type(args.evaluation_data.get("stop_on_success", True)) is not bool:
                raise ValueError("stop_on_success must be a boolean")
            args.duration = float(args.evaluation_data.get("duration_s", args.duration))
        except (KeyError, TypeError, ValueError, OSError) as error:
            parser.error(f"Invalid evaluation plan: {error}")
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("duration must be positive")
    if args.mode == "policy" and (not 1 <= args.policy_execute_steps <= 32 or args.duration < 1/30):
        parser.error("Policy requires 1..32 executed actions per request and at least 1/30 s duration")
    cfg = json.loads(args.config.read_text())
    if args.mode == "policy":
        from pressb.action_postprocessing import JointActionPostprocessor
        try:
            smoothing = cfg.get("policy_execution", {}).get("smoothing", {})
            if args.policy_smoothing is None:
                args.policy_smoothing = smoothing.get("mode", "none")
            if args.policy_smoothing_alpha is None:
                args.policy_smoothing_alpha = smoothing.get("alpha", .8)
            # Validate before connecting or starting the simulator.
            JointActionPostprocessor(args.policy_smoothing, args.policy_smoothing_alpha)
            # Save the effective CLI-over-config values with the rollout.
            cfg.setdefault("policy_execution", {})["smoothing"] = {
                "mode": args.policy_smoothing, "alpha": args.policy_smoothing_alpha}
        except (AttributeError, TypeError, ValueError) as error:
            parser.error(f"Invalid policy smoothing configuration: {error}")
    elif args.policy_smoothing is not None or args.policy_smoothing_alpha is not None:
        parser.error("policy-smoothing options apply only to --mode policy")
    if args.trajectory is None:
        args.trajectory = ROOT / cfg.get("episode_source", {}).get(
            "trajectory", "outputs/piper_pap_reconstruction_20261022/dataset_audit/episode_000000.npz")
    expected_episode = cfg.get("episode_source", {}).get("episode_index")
    if expected_episode is not None and args.trajectory.stem != f"episode_{int(expected_episode):06d}":
        parser.error("Episode-specific scene and trajectory differ; use the matching episode configuration")
    if args.mode == "policy":
        from pressb.pick_place.wallx import load_policy_initial
        trajectory = load_policy_initial(args.trajectory, cfg["gripper_max_width_m"])
    else:
        from pressb.pick_place.task import load_trajectory
        trajectory = load_trajectory(args.trajectory, args.source, cfg["gripper_max_width_m"])
    args.output = args.output.resolve()
    identity = dict(config=cfg, plan=args.evaluation_data, trajectory=str(args.trajectory.resolve()),
        policy_endpoint=args.policy_endpoint, policy_execute_steps=args.policy_execute_steps,
        policy_prompt=args.policy_prompt, policy_dataset_name=args.policy_dataset_name,
        policy_arm_limit_mode=args.policy_arm_limit_mode, policy_gripper_limit_mode=args.policy_gripper_limit_mode,
        duration_s=args.duration)
    if args.resume:
        identity_path = args.output / "evaluation_identity.json"
        if not identity_path.exists() or json.loads(identity_path.read_text()) != identity:
            parser.error("Resume requires exactly the same plan, effective scene configuration and policy settings")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    if args.evaluation_plan is not None:
        write(args.output/"evaluation_plan.json", args.evaluation_data)
        write(args.output/"evaluation_identity.json", identity)
    write(args.output/"config.json", cfg)
    status = dict(state="starting", started_at=datetime.now(timezone.utc).isoformat())
    write(args.output/"status.json", status)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    app, policy_client = None, None
    try:
        if args.mode == "policy":
            from pressb.pick_place.wallx import WallXClient
            policy_client = WallXClient(args.policy_endpoint, timeout=args.policy_timeout,
                token=os.environ.get(args.policy_token_env))
            write(args.output/"policy_server_metadata.json", policy_client.connect())
        from isaacsim import SimulationApp
        app = SimulationApp({"headless":args.headless, "active_gpu":args.gpu, "physics_gpu":args.gpu,
            "multi_gpu":False, "renderer":"RayTracedLighting", "anti_aliasing":1, "fast_shutdown":True,
            "extra_args":["--/app/asyncRendering=false", "--/rtx/ecoMode/enabled=false",
                          "--/validate/p2p/enabled=false", "--/validate/iommu/enabled=false"]})
        run(app, args, cfg, trajectory, policy_client=policy_client)
        status["state"] = "complete"
    except BaseException as error:
        status.update(state="failed", error=repr(error))
        traceback.print_exc()
        raise
    finally:
        status["finished_at"] = datetime.now(timezone.utc).isoformat()
        write(args.output/"status.json", status)
        if policy_client:
            policy_client.close()
        if app:
            app.app.post_quit(0 if status["state"] == "complete" else 1)
            app.close()


if __name__ == "__main__":
    main()
