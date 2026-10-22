#!/usr/bin/env python3
"""Inspect the independent tabletop color-button task or run physical IK checks.

Robot, fixed cameras, workbench, and lighting retain the PAP configuration.
The default wrist optics match the elevator task. Demo mode uses a scripted
controller for environment validation, not a learned policy.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
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


def run(app, args, cfg):
    import numpy as np
    from PIL import Image
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema, PhysicsSchemaTools
    import omni.physx
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.sensors.camera import Camera
    from pressb.table_buttons.scene import build_scene
    from pressb.pick_place.scene import build_tripod
    from pressb.table_buttons.task import COLORS, TableButtonOutcome, task_instruction, project_base_points
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
    initial_state = np.r_[cfg["initial_joint_positions_rad"], cfg["initial_gripper_width_m"]]
    if np.any(initial_state[:6] < kin.lower) or np.any(initial_state[:6] > kin.upper):
        raise ValueError("Initial joints exceed robot limits")
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
        # Preserve explicit historical camera registrations on replay. The
        # default task config uses the elevator optics from attach_wrist_camera.
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
    finger_material = UsdShade.Material.Define(stage, scene["root_path"] + "/Looks/FingerContact")
    material_api = UsdPhysics.MaterialAPI.Apply(finger_material.GetPrim())
    material_api.CreateStaticFrictionAttr(cfg["finger_static_friction"])
    material_api.CreateDynamicFrictionAttr(cfg["finger_dynamic_friction"])
    material_api.CreateRestitutionAttr(0.)
    for prim in Usd.PrimRange(robot_prim):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
        if prim.HasAPI(UsdPhysics.CollisionAPI) and any(f"/link{i}/" in str(prim.GetPath()) for i in (7, 8)):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(finger_material, materialPurpose="physics")
    up_path = make_camera(stage, scene["root_path"] + "/UpCamera", cfg["up_camera"], base)
    overview_path = make_camera(stage, scene["root_path"] + "/OverviewCamera", cfg["overview_camera"], base)
    camera_cfg = cfg["up_camera"]
    if "eye_base_m" in camera_cfg:
        scene["table_camera_hardware"] = build_tripod(stage, np.asarray(camera_cfg["eye_base_m"])+base,
                     np.asarray(camera_cfg["target_base_m"])+base, cfg["table_height"],
                     root_path=scene["root_path"] + "/Tripod")
    robot = world.scene.add(SingleArticulation(prim_path="/World/Piper", name="piper"))
    camera_specs = {
        "up": (up_path, cfg["up_camera"]["resolution"]),
        "wrist": (wrist_info["camera_path"], cfg.get("wrist_camera", {}).get("resolution", cfg["wrist_camera_resolution"])),
        "overview": (overview_path, cfg["overview_camera"]["resolution"]),
    }
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

    initial = full_command(initial_state)
    robot.set_joint_positions(initial)
    robot.set_joint_velocities(np.zeros(robot.num_dof))
    controller = robot.get_articulation_controller()
    kps, kds = np.zeros(robot.num_dof), np.zeros(robot.num_dof)
    kps[indices[:6]], kds[indices[:6]] = cfg["arm_stiffness"], cfg["arm_damping"]
    kps[indices[6:]], kds[indices[6:]] = cfg["gripper_stiffness"], cfg["gripper_damping"]
    controller.set_gains(kps=kps, kds=kds)
    controller.apply_action(ArticulationAction(joint_positions=initial))
    caps = {color: SingleRigidPrim(prim_path=spec['body_path'], name=f'cap_{color}')
            for color, spec in scene['buttons'].items()}
    for cap in caps.values():
        cap.initialize()
    tick = [0]
    contact_forces = {color: 0. for color in COLORS}
    contact_events, unexpected_contacts = [], []

    def on_contact(headers, data):
        for header in headers:
            if not header.num_contact_data:
                continue
            actors = [str(PhysicsSchemaTools.intToSdfPath(value)) for value in (header.actor0, header.actor1)]
            force = sum(float(np.linalg.norm(data[i].impulse)) for i in
                        range(header.contact_data_offset, header.contact_data_offset+header.num_contact_data)) / dt
            fingers = [name for name in ('link7', 'link8') if f'/World/Piper/{name}' in actors]
            for color, spec in scene['buttons'].items():
                if spec['body_path'] in actors and fingers:
                    contact_forces[color] += force
                    contact_events.append(dict(tick=tick[0], color=color, fingers=fingers, force_n=force))
            moving_robot = [actor for actor in actors if actor.startswith('/World/Piper/')
                            and actor != '/World/Piper/base_link']
            self_contact = all(actor.startswith('/World/Piper/') for actor in actors) and actors[0] != actors[1]
            environment_contact = any('/Table/Top' in actor or '/Housing' in actor or '/Bezel' in actor for actor in actors)
            non_finger_cap_contact = not fingers and any(spec['body_path'] in actors for spec in scene['buttons'].values())
            if force > .02 and moving_robot and (self_contact or environment_contact or non_finger_cap_contact):
                unexpected_contacts.append(dict(tick=tick[0], actors=actors, force_n=force))

    subscription = omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(on_contact)
    for _ in range(24):
        world.step(render=False)
    for _ in range(warmup_frames):
        world.render()
    for name, camera in cameras.items():
        Image.fromarray(np.asarray(camera.get_rgba())[..., :3]).save(args.output/f'initial_{name}.jpg', quality=95)
    for name, value in zip(robot.dof_names, initial):
        prim = stage.GetPrimAtPath(f'/World/Piper/joints/{name}')
        if prim.IsA(UsdPhysics.RevoluteJoint):
            UsdPhysics.DriveAPI(prim, 'angular').GetTargetPositionAttr().Set(float(np.rad2deg(value)))
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            UsdPhysics.DriveAPI(prim, 'linear').GetTargetPositionAttr().Set(float(value))
    stage.GetRootLayer().Export(str(args.output/'scene.usda'))
    report = dict(task='piper_table_buttons', mode=args.mode, config=cfg, scene=scene,
                  initial_q_target=initial[indices].tolist(), initial_q_actual=robot.get_joint_positions()[indices].tolist(),
                  wrist_camera=wrist_info, physics_dt=dt, video_fps=fps, seed=args.seed,
                  policy_or_RL_evaluation=False, controller='scripted bounded IK' if args.mode=='demo' else None,
                  button_motion='PhysX spring joint and contact only; no cap teleport or scripted press state')
    cache = UsdGeom.XformCache()
    report['cameras'] = {name: dict(path=path, resolution=size,
        intrinsics=np.asarray(cameras[name].get_intrinsics_matrix()).tolist(),
        world_from_camera_usd_at_initial=np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path))).T.tolist())
        for name, (path, size) in camera_specs.items()}
    centers = [np.asarray(scene['buttons'][color]['neutral_top_world'])-base for color in COLORS]
    projected, depth = project_base_points(centers, cfg['up_camera'])
    report['button_pixels_up'] = {color: dict(xy=projected[i].tolist(), depth_m=float(depth[i])) for i,color in enumerate(COLORS)}
    report['neutral_travel_m'] = {color: float(scene['buttons'][color]['cap_rest_world'][2]-cap.get_world_pose()[0][2]) for color,cap in caps.items()}
    if args.mode == 'snapshot':
        write(args.output/'report.json', report)
        if args.hold:
            while app.is_running():
                world.render()
        return

    targets = list(COLORS) if args.target == 'all' else [args.target]
    videos = {name: Video(args.output/f'{name}.mp4', size, fps) for name, (_,size) in camera_specs.items()}
    records, episodes = [], []
    contacts_start = len(contact_events)
    try:
        for episode_index, color in enumerate(targets):
            instruction = task_instruction(color)
            path_times, path_commands, plan_report = plan_press(kin, initial_state,
                np.asarray(scene['buttons'][color]['neutral_top_world'])-base, fps=fps,
                approach=cfg.get('press_approaches', {}).get(color))
            checker = TableButtonOutcome(color, **cfg['success'])
            total_steps = round(path_times[-1]/dt)
            maximum_depth = {name:0. for name in COLORS}
            maximum_force = {name:0. for name in COLORS}
            result = None
            held_command = None
            for step in range(total_steps):
                seconds = step*dt
                command = np.array([np.interp(seconds, path_times, path_commands[:,j]) for j in range(7)])
                # Limit further commanded descent from measured travel.
                # Only the robot command is held; the cap
                # remains a free spring-driven PhysX body.
                if held_command is not None and 3.1 <= seconds < 4.45:
                    command = held_command.copy()
                elif held_command is not None and 4.45 <= seconds < 5.55:
                    u = (seconds-4.45)/1.1
                    blend = u*u*u*(10+u*(-15+6*u))
                    hover_command = np.array([np.interp(5.55,path_times,path_commands[:,j]) for j in range(7)])
                    command = held_command+(hover_command-held_command)*blend
                q_target = full_command(command)
                tick[0] += 1
                for name in COLORS:
                    contact_forces[name] = 0.
                controller.apply_action(ArticulationAction(joint_positions=q_target))
                world.step(render=False)
                travels = {name:float(scene['buttons'][name]['cap_rest_world'][2]-caps[name].get_world_pose()[0][2]) for name in COLORS}
                if held_command is None and 3.1 <= seconds < 4.45 and travels[color] >= cfg['success']['press_depth_m']+.0002:
                    held_command = command.copy()
                measured_seconds = (step+1)*dt
                result = checker.update(measured_seconds, travels, contact_forces)
                for name in COLORS:
                    maximum_depth[name] = max(maximum_depth[name], travels[name])
                    maximum_force[name] = max(maximum_force[name], contact_forces[name])
                q_actual = robot.get_joint_positions()[indices]
                records.append(dict(time_s=tick[0]*dt, episode=episode_index, q_actual=q_actual.copy(),
                                    q_target=q_target[indices], travel_m=np.array([travels[name] for name in COLORS]),
                                    finger_force_n=np.array([contact_forces[name] for name in COLORS])))
                if step % stride == 0:
                    for _ in range(subframes):
                        world.render()
                    for name, camera in cameras.items():
                        videos[name].add(np.ascontiguousarray(camera.get_rgba()[..., :3]))
                if step % 120 == 0:
                    print(json.dumps(dict(event='press_validation', target=color, seconds=measured_seconds,
                        travel_m=travels, success=result['success'], wrong_button=result['wrong_button'])), flush=True)
            episodes.append(dict(target=color, instruction=instruction, duration_s=total_steps*dt,
                outcome=result, maximum_travel_m=maximum_depth, maximum_finger_force_n=maximum_force,
                final_travel_m=travels, plan=plan_report))
            write(args.output/'episodes.json', episodes)
    finally:
        for video in videos.values():
            video.close()
    arrays = {key:np.asarray([row[key] for row in records]) for key in records[0]}
    np.savez_compressed(args.output/'physics.npz', **arrays)
    report.update(episodes=episodes, duration_s=len(records)*dt,
        first_video_sample_time_s=dt,
        video_frames={name:video.count for name,video in videos.items()},
        all_targets_pressed=all(episode['outcome']['success'] for episode in episodes),
        all_caps_returned=all(abs(v)<cfg['success']['release_depth_m'] for v in travels.values()),
        maximum_arm_tracking_error_rad=float(np.max(np.abs(arrays['q_actual'][:,:6]-arrays['q_target'][:,:6]))),
        finger_contact_events=len(contact_events)-contacts_start, unexpected_table_or_housing_contacts=len(unexpected_contacts),
        all_paths_clear=not unexpected_contacts)
    for name, camera in cameras.items():
        Image.fromarray(np.asarray(camera.get_rgba())[..., :3]).save(args.output/f'final_{name}.jpg', quality=95)
    write(args.output/'contacts.json', contact_events)
    write(args.output/'unexpected_contacts.json', unexpected_contacts)
    write(args.output/'report.json', report)
    if not report['all_targets_pressed'] or not report['all_caps_returned'] or not report['all_paths_clear']:
        raise RuntimeError('Physical press/release validation did not pass; inspect report.json')
    if args.hold:
        while app.is_running():
            world.render()


def plan_press(kin, initial, top, *, fps=30, approach=None):
    from pressb.table_buttons.motion import plan_press as build_plan
    return build_plan(kin, initial, top, fps=fps, approach=approach)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/piper_table_buttons.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=('snapshot','demo'),default='snapshot')
    parser.add_argument('--target',choices=('red','yellow','blue','all'),default=None)
    parser.add_argument('--instruction',help='Single-color English or Chinese press instruction')
    parser.add_argument('--seed',type=int,help='Sample a reproducible visible layout instead of configured positions')
    parser.add_argument('--gpu',type=int,default=1)
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--hold',action='store_true')
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text())
    from pressb.table_buttons.task import parse_instruction,button_in_camera
    if args.instruction:
        color=parse_instruction(args.instruction)
        if args.target is not None and args.target!=color:
            parser.error('--target and --instruction disagree')
        args.target=color
    args.target=args.target or cfg.get('task_color','red')
    cfg['task_color']=args.target
    cfg['task_instruction']=args.instruction or ('Press each requested color in a separate validation episode.' if args.target=='all' else f'Press the {args.target} button.')
    geometry=cfg['button_geometry']
    top=geometry['base_height_m']+geometry['cap_height_m']
    if cfg.get('position_sampling', {}).get('domain') == 'camera_reachable':
        import numpy as np
        from pressb.kinematics import PiperKinematics
        from pressb.table_buttons.placement import ReachablePlacement
        placement = ReachablePlacement(PiperKinematics(ROOT/cfg['robot_urdf'], tip_offset=.1358), cfg)
        if args.seed is not None:
            layout = placement.sample(args.seed)
            for color, xy in layout['positions'].items():
                cfg['buttons'][color]['center_xy'] = np.asarray(xy).tolist()
            approaches = layout['approaches']
        else:
            approaches = {color: placement.candidate(spec['center_xy']) for color, spec in cfg['buttons'].items()}
            if any(value is None for value in approaches.values()):
                parser.error('Configured buttons must admit visible, collision-screened press paths')
            clear, diagnostic = placement.validate_layout(
                {color: spec['center_xy'] for color, spec in cfg['buttons'].items()}, approaches)
            if not clear:
                parser.error(f'Configured buttons obstruct a press path: {diagnostic}')
        def serializable(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, dict):
                return {key: serializable(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [serializable(item) for item in value]
            if isinstance(value, np.generic):
                return value.item()
            return value
        if args.mode == 'demo':
            # Check the actual interpolated command path, in addition to the
            # lighter candidate screening used during rejection sampling.
            from pressb.table_buttons.motion import plan_press as build_plan
            initial_state = np.r_[cfg['initial_joint_positions_rad'], cfg['initial_gripper_width_m']]
            positions = {color: spec['center_xy'] for color, spec in cfg['buttons'].items()}
            for color in (list(positions) if args.target == 'all' else [args.target]):
                times, commands, _ = build_plan(placement.workspace.kin, initial_state,
                    np.r_[positions[color], top], fps=cfg['fps'], approach=approaches[color])
                physics_times = np.arange(0., times[-1]+cfg['physics_dt']*.5, cfg['physics_dt'])
                path = np.column_stack([np.interp(physics_times, times, commands[:,j]) for j in range(6)])
                clear, diagnostic = placement.clearance.check_path(path, cfg['initial_gripper_width_m'])
                if not clear:
                    parser.error(f'{color} dense motion preflight failed: {diagnostic}')
                clear, diagnostic = placement.clearance.check_button_path(
                    path, positions, color, cfg['initial_gripper_width_m'])
                if not clear:
                    parser.error(f'{color} dense button preflight failed: {diagnostic}')
        cfg['press_approaches'] = serializable(approaches)
        cfg['sampling_domain'] = serializable(layout['domain'] if args.seed is not None else placement.domain)
    elif args.seed is not None:
        # Retain explicit old rectangle configs for historical replays.
        from pressb.table_buttons.task import sample_button_positions
        sampling=cfg['position_sampling']
        sampled=sample_button_positions(cfg['up_camera'],args.seed,bounds_xy=sampling['bounds_xy'],
            min_separation=sampling['min_separation'],margin_px=sampling['pixel_margin'],
            button_radius=geometry['base_radius_m'],button_top_z=top)
        for color,xy in sampled.items():
            cfg['buttons'][color]['center_xy']=xy
    for color,spec in cfg['buttons'].items():
        if not button_in_camera(spec['center_xy'],cfg['up_camera'],button_radius=geometry['base_radius_m'],button_top_z=top):
            parser.error(f'{color} housing is outside the third-person camera view')
    args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    write(args.output/'config.json',cfg)
    status=dict(state='starting',started_at=datetime.now(timezone.utc).isoformat())
    write(args.output/'status.json',status)
    os.environ.setdefault('OMNI_KIT_ACCEPT_EULA','YES')
    app=None
    try:
        from isaacsim import SimulationApp
        app=SimulationApp({'headless':args.headless,'active_gpu':args.gpu,'physics_gpu':args.gpu,
            'multi_gpu':False,'renderer':'RayTracedLighting','anti_aliasing':1,'fast_shutdown':True,
            'extra_args':['--/app/asyncRendering=false','--/rtx/ecoMode/enabled=false',
                          '--/validate/p2p/enabled=false','--/validate/iommu/enabled=false',
                          '--/plugins/carb.tasking.plugin/threadCount=8','--/plugins/omni.tbb.globalcontrol/maxThreadCount=8']})
        run(app,args,cfg)
        status['state']='complete'
    except BaseException as error:
        status.update(state='failed',error=repr(error))
        traceback.print_exc()
        raise
    finally:
        status['finished_at']=datetime.now(timezone.utc).isoformat()
        write(args.output/'status.json',status)
        if app:
            app.app.post_quit(0 if status['state']=='complete' else 1)
            app.close()


if __name__=='__main__':
    main()
