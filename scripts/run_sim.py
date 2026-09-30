#!/usr/bin/env python3
"""Run a measured 12-button physical pressing episode in Isaac Sim."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/scene.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/edge_feedback")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--max-steps", type=int, default=0, help="Diagnostic partial run; cannot pass full validation")
    parser.add_argument("--hold", action="store_true", help="Keep GUI open after the episode")
    args = parser.parse_args()
    if args.max_steps < 0:
        parser.error("--max-steps must be nonnegative")
    cfg = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    status_path = args.output / "run_status.json"
    status = {"started_at": datetime.now(timezone.utc).isoformat(), "status": "running"}
    status_path.write_text(json.dumps(status, indent=2))
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    app = None
    try:
        from isaacsim import SimulationApp
        app = SimulationApp({"headless": args.headless, "width": args.width, "height": args.height,
                             "active_gpu": args.gpu, "physics_gpu": args.gpu, "multi_gpu": False,
                             "renderer": "RayTracedLighting", "anti_aliasing": 1, "fast_shutdown": True,
                             "extra_args": ["--/app/asyncRendering=false", "--/rtx/ecoMode/enabled=false",
                                            "--/validate/p2p/enabled=false",
                                            "--/validate/iommu/enabled=false"]})
        run(app, args, cfg)
        status.update(status="complete", finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        # Kit's supported fast shutdown exits before Python can print a pending
        # exception. Log it now and explicitly pass the failure code to Kit.
        traceback.print_exc()
        raise
    finally:
        status_path.write_text(json.dumps(status, indent=2))
        if app:
            # Unloading every native extension can crash during Python GC in
            # Isaac 4.5; use its default fast shutdown without masking failures.
            app.app.post_quit(0 if status["status"] == "complete" else 1)
            app.close()


def run(app, args, cfg):
    import numpy as np
    from PIL import Image
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, PhysxSchema, Gf, PhysicsSchemaTools
    import omni.physx
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.sensors.camera import Camera
    from pressb.kinematics import PiperKinematics
    from pressb.planning import make_plan
    from pressb.scene import build_scene, set_button_light
    from pressb.recording import Recorder
    from pressb.wrist_camera import apply_black_materials, attach_wrist_camera
    from pressb.fixed_camera import build_global_camera
    from pressb.camera_recording import WristCameraRecorder

    kin = PiperKinematics(ROOT / cfg["robot_urdf"], tip_offset=cfg["tip_offset"])
    plan = make_plan(kin, cfg)
    render_stride = int(cfg.get("render_stride", 12))
    video_fps = 1. / (cfg["physics_dt"] * render_stride)
    world = World(stage_units_in_meters=1., physics_dt=cfg["physics_dt"], rendering_dt=1. / video_fps,
                  backend="numpy", device="cpu")
    world.get_physics_context().set_physx_update_transformations_settings(
        update_to_usd=True, update_velocities_to_usd=True)
    stage = world.stage
    scene = build_scene(stage, cfg)
    add_reference_to_stage(str(ROOT / cfg["robot_usd"]), "/World/Piper")
    robot_prim = stage.GetPrimAtPath("/World/Piper")
    scene["assets"]["piper_material"] = apply_black_materials(stage, "/World/Piper")
    xf = UsdGeom.Xformable(robot_prim)
    xf.ClearXformOpOrder()
    base_position = np.array([cfg.get("robot_base_x", 0.), cfg.get("robot_base_y", 0.), cfg["table_height"]])
    xf.AddTranslateOp().Set(Gf.Vec3d(*base_position))
    # Configure the existing official articulation, retaining its masses/limits/collisions.
    for prim in Usd.PrimRange(robot_prim):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            phys = PhysxSchema.PhysxArticulationAPI.Apply(prim)
            phys.CreateSolverPositionIterationCountAttr(32)
            phys.CreateSolverVelocityIterationCountAttr(4)
            phys.CreateEnabledSelfCollisionsAttr(True)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
        if prim.IsA(UsdPhysics.RevoluteJoint):
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
            drive.CreateStiffnessAttr(12000.)
            drive.CreateDampingAttr(600.)
            drive.CreateMaxForceAttr(100.)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            collision.CreateContactOffsetAttr(0.0005)
            collision.CreateRestOffsetAttr(0.)
    link6 = next(p for p in Usd.PrimRange(robot_prim) if p.GetName() == "link6")
    # A rigid stylus is attached to link6. Its spherical front is the IK tool point.
    tool_radius = .005
    tip_center = cfg["tip_offset"] - tool_radius
    shaft = UsdGeom.Cylinder.Define(stage, str(link6.GetPath()) + "/PressStylus")
    shaft.CreateRadiusAttr(.004)
    # Extend into the flat finger pads while retaining the same pressing tip.
    shaft_root = .125
    shaft.CreateHeightAttr(tip_center - shaft_root)
    shaft.AddTranslateOp().Set(Gf.Vec3d(0, 0, (shaft_root + tip_center) / 2))
    shaft.CreateDisplayColorAttr([(0.15, 0.18, 0.20)])
    sphere = UsdGeom.Sphere.Define(stage, str(link6.GetPath()) + "/PressTip")
    sphere.CreateRadiusAttr(tool_radius)
    sphere.AddTranslateOp().Set(Gf.Vec3d(0, 0, tip_center))
    sphere.CreateDisplayColorAttr([(0.05, 0.07, 0.08)])
    for prim in (shaft.GetPrim(), sphere.GetPrim()):
        UsdPhysics.CollisionAPI.Apply(prim)
        PhysxSchema.PhysxCollisionAPI.Apply(prim).CreateContactOffsetAttr(.0002)
    PhysxSchema.PhysxContactReportAPI.Apply(link6).CreateThresholdAttr(0.)
    wrist_info = attach_wrist_camera(stage, str(link6.GetPath()), cfg)
    scene["assets"]["wrist_camera"] = wrist_info
    global_info = build_global_camera(stage, cfg)
    scene["assets"]["global_camera"] = global_info
    # The imported mount may add rigid bodies after the initial robot setup.
    # Subscribe to every composed robot body before reset and warmup.
    for prim in Usd.PrimRange(robot_prim):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
    robot = world.scene.add(SingleArticulation(prim_path="/World/Piper", name="piper"))
    buttons = scene["buttons"]
    bodies = {floor: world.scene.add(SingleRigidPrim(prim_path=info["body_path"], name=f"button_{floor}"))
              for floor, info in buttons.items()}
    camera = Camera(prim_path=scene["camera_path"], resolution=(args.width, args.height), frequency=-1)
    wrist_camera = Camera(prim_path=wrist_info["camera_path"], name="wrist_realsense",
                          resolution=tuple(wrist_info["resolution"]), frequency=-1)
    global_camera = Camera(prim_path=global_info["camera_path"], name="global_realsense",
                           resolution=tuple(global_info["resolution"]), frequency=-1)
    if not args.headless:
        from omni.kit.viewport.utility import get_active_viewport
        viewport = get_active_viewport()
        if viewport:
            viewport.set_active_camera(scene["camera_path"])
    world.reset()
    camera.initialize()
    wrist_camera.initialize()
    wrist_camera.add_distance_to_image_plane_to_frame()
    global_camera.initialize()
    global_camera.add_distance_to_image_plane_to_frame()
    indices = np.array([robot.get_dof_index(name) for name in kin.joint_names])
    initial = robot.get_joint_positions().copy()
    initial[indices] = plan.q[0]
    gripper_names = ("joint7", "joint8")
    gripper_indices = np.array([robot.get_dof_index(name) for name in gripper_names])
    # The original flat finger faces meet at q=0. Close around the 8 mm
    # shaft, not through it, and maintain this opening during every arm phase.
    gripper_target = np.asarray(cfg.get("gripper_joint_positions_m", [.004, -.004]), dtype=float)
    if (gripper_target.shape != (2,) or not np.isfinite(gripper_target).all()
            or not (0 <= gripper_target[0] <= .035 and -.035 <= gripper_target[1] <= 0)):
        raise ValueError("Expected joint7/joint8 positions in metres within the official finger limits")
    initial[gripper_indices] = gripper_target
    robot.set_joint_positions(initial)
    robot.set_joint_velocities(np.zeros_like(initial))
    controller = robot.get_articulation_controller()
    controller.set_gains(kps=np.full(len(initial), 12000.), kds=np.full(len(initial), 600.))
    controller.apply_action(ArticulationAction(joint_positions=initial))
    forces = np.zeros(12)
    body_to_floor = {info["body_path"]: f for f, info in buttons.items()}
    tool_colliders = {str(shaft.GetPath()), str(sphere.GetPath())}
    unexpected = []
    contact_step = [-1]  # All initial settling contacts remain part of validation.

    def on_contact(headers, data):
        for header in headers:
            actors = [str(PhysicsSchemaTools.intToSdfPath(x)) for x in (header.actor0, header.actor1)]
            colliders = [str(PhysicsSchemaTools.intToSdfPath(x)) for x in (header.collider0, header.collider1)]
            floor = next((body_to_floor[a] for a in actors if a in body_to_floor), None)
            if floor is not None and any(c in tool_colliders for c in colliders):
                for k in range(header.contact_data_offset, header.contact_data_offset + header.num_contact_data):
                    forces[floor - 24] += float(np.linalg.norm(data[k].impulse)) / cfg["physics_dt"]
            elif floor is not None and header.num_contact_data and any(a.startswith("/World/Piper/") for a in actors):
                # The camera housing and fingers must not actuate neighbouring caps.
                force = sum(float(np.linalg.norm(data[k].impulse)) / cfg["physics_dt"]
                            for k in range(header.contact_data_offset, header.contact_data_offset + header.num_contact_data))
                if force > .1 and len(unexpected) < 100:
                    unexpected.append({"step": contact_step[0], "actors": actors, "colliders": colliders,
                                       "force_n": force, "reason": "non_stylus_button_contact"})
            if header.num_contact_data and floor is None:
                robot_actors = [a for a in actors if a.startswith("/World/Piper/")]
                external_actors = [a for a in actors if not a.startswith("/World/Piper/")]
                # Distinct robot links must not collide, including the wrist
                # bracket/camera compound. Keep PhysX's adjacent-joint filtering;
                # do not suppress additional pairs merely because both are robot parts.
                if len(robot_actors) == 2 and actors[0] != actors[1]:
                    force = sum(float(np.linalg.norm(data[k].impulse)) / cfg["physics_dt"]
                                for k in range(header.contact_data_offset, header.contact_data_offset + header.num_contact_data))
                    if force > .1 and len(unexpected) < 100:
                        unexpected.append({"step": contact_step[0], "actors": actors, "colliders": colliders,
                                           "force_n": force, "reason": "robot_self_contact"})
                # Base/table contact is the intended mount; moving links must stay clear.
                elif robot_actors and external_actors and any(a.rsplit("/", 1)[-1] not in ("dummy_link", "base_link", "link1") for a in robot_actors):
                    force = sum(float(np.linalg.norm(data[k].impulse)) / cfg["physics_dt"]
                                for k in range(header.contact_data_offset, header.contact_data_offset + header.num_contact_data))
                    if force > .1 and len(unexpected) < 100:
                        unexpected.append({"step": contact_step[0], "actors": actors, "colliders": colliders,
                                           "force_n": force, "reason": "robot_environment_contact"})

    contact_subscription = omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(on_contact)
    for step in range(90):
        forces.fill(0.)
        world.step(render=False)
        if step % 4 == 0:
            world.render()
    def export_scene(filename, command):
        # Tensor drive targets do not author USD. Persist equivalent targets for
        # opening the snapshot in Isaac Sim and pressing Play without snapping.
        for name, value in zip(robot.dof_names, command):
            joint = stage.GetPrimAtPath(f"/World/Piper/joints/{name}")
            if joint and joint.IsA(UsdPhysics.RevoluteJoint):
                UsdPhysics.DriveAPI(joint, "angular").GetTargetPositionAttr().Set(float(np.degrees(value)))
            elif joint and joint.IsA(UsdPhysics.PrismaticJoint):
                UsdPhysics.DriveAPI(joint, "linear").GetTargetPositionAttr().Set(float(value))
        stage.GetRootLayer().Export(str(args.output / filename))
    export_scene("scene.usda", initial)
    camera_stride = int(cfg.get("wrist_camera_capture_stride", 12))
    camera_recorder = WristCameraRecorder(args.output, wrist_camera, wrist_info, camera_stride, len(plan.q))
    global_stride = int(cfg.get("global_camera_capture_stride", camera_stride))
    global_recorder = WristCameraRecorder(args.output, global_camera, global_info,
                                          global_stride, len(plan.q), output_subdir="global_camera")

    recorder = Recorder(args.output, cfg, kin.joint_names)
    illuminated = set()
    frame_dir = args.output / "frames"
    if args.video:
        frame_dir.mkdir(exist_ok=True)
    max_error = 0.
    max_fk_error = 0.
    max_gripper_error = 0.
    maximum_travel = np.zeros(12)
    maximum_force = np.zeros(12)
    nsteps = min(args.max_steps or len(plan.q), len(plan.q))
    frame_count = 0
    press_previews = set()
    episode_start = world.current_time
    for i in range(nsteps):
        if not app.is_running():
            break
        desired = initial.copy()
        desired[indices] = plan.q[i]
        controller.apply_action(ArticulationAction(joint_positions=desired))
        forces.fill(0.)
        contact_step[0] = i
        render = i % render_stride == 0
        sensor_render = i % camera_stride == 0 or i % global_stride == 0
        # step(render=True) can advance multiple physics steps per render tick.
        # Advance exactly one physics dt, then refresh rendering independently.
        world.step(render=False)
        elapsed = float(world.current_time - episode_start)
        measured_joints = robot.get_joint_positions()
        actual = measured_joints[indices]
        gripper_actual = measured_joints[gripper_indices]
        max_gripper_error = max(max_gripper_error, float(np.max(np.abs(gripper_actual - gripper_target))))
        velocity = robot.get_joint_velocities()[indices]
        tip_mat = UsdGeom.Xformable(link6).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        tip = np.array(tip_mat.Transform(Gf.Vec3d(0, 0, cfg["tip_offset"])))
        fk_tip = kin.fk(actual)[:3, 3] + base_position
        max_fk_error = max(max_fk_error, float(np.linalg.norm(tip - fk_tip)))
        max_error = max(max_error, float(np.max(np.abs(actual - plan.q[i]))))
        travels = np.array([bodies[f].get_world_pose()[0][0] - buttons[f]["rest_x"] for f in range(24, 36)])
        maximum_travel = np.maximum(maximum_travel, travels)
        maximum_force = np.maximum(maximum_force, forces)
        for floor in range(24, 36):
            j = floor - 24
            if floor not in illuminated and travels[j] >= cfg["press_threshold"] and forces[j] > .02:
                illuminated.add(floor)
                set_button_light(stage, buttons[floor], True)
                recorder.event(type="button_pressed", floor=floor, time=elapsed,
                               travel_m=float(travels[j]), contact_force_n=float(forces[j]),
                               commanded_floor=int(plan.floor[i]), tip_world=tip.tolist())
                print(f"PRESS floor={floor} travel={travels[j]*1000:.2f}mm contact={forces[j]:.3f}N", flush=True)
            elif floor in illuminated and travels[j] <= cfg.get("release_threshold", .0008):
                illuminated.remove(floor)
                set_button_light(stage, buttons[floor], False)
                recorder.event(type="button_released", floor=floor, time=elapsed,
                               travel_m=float(travels[j]), contact_force_n=float(forces[j]),
                               commanded_floor=int(plan.floor[i]), tip_world=tip.tolist())
                print(f"RELEASE floor={floor} travel={travels[j]*1000:.2f}mm", flush=True)
        if plan.phase[i] == "home_hold" and (i + 1 == len(plan.q) or plan.phase[i + 1] != "home_hold"):
            home_error = float(np.max(np.abs(actual - np.asarray(cfg["home_q"]))))
            recorder.event(type="cycle_home", floor=int(plan.floor[i]), time=elapsed, q_error_rad=home_error)
            print(f"HOME floor={plan.floor[i]} error={home_error:.5f}rad", flush=True)
        recorder.record(time=elapsed, floor=int(plan.floor[i]), phase=str(plan.phase[i]),
                        q_target=plan.q[i].copy(), q_actual=actual.copy(), qd_actual=velocity.copy(),
                        gripper_q_target=gripper_target.copy(), gripper_q_actual=gripper_actual.copy(),
                        tip_target=plan.target_tip[i].copy(), tip_actual=tip.copy(),
                        button_travel=travels.copy(), contact_force=forces.copy(),
                        lights=np.array([int(f in illuminated) for f in range(24, 36)]))
        # Render after applying this step's measured light state.
        if render or sensor_render:
            world.render()
        if i % camera_stride == 0:
            camera_recorder.capture(elapsed, plan.floor[i], plan.phase[i])
        if i % global_stride == 0:
            global_recorder.capture(elapsed, plan.floor[i], plan.phase[i])
        if render and (args.video or i == 0):
            rgba = camera.get_rgba()
            if rgba is not None and np.asarray(rgba).size:
                rgb = np.asarray(rgba)[..., :3].astype(np.uint8)
                if i == 0:
                    Image.fromarray(rgb).save(args.output / "scene.png")
                if args.video:
                    Image.fromarray(rgb).save(frame_dir / f"{frame_count:06d}.png", compress_level=1)
                    frame_count += 1
        if render and plan.phase[i] == "hold" and plan.floor[i] not in press_previews:
            rgba = np.asarray(camera.get_rgba())
            if rgba.size:
                Image.fromarray(rgba[..., :3].astype(np.uint8)).save(args.output / f"press_{plan.floor[i]}.png")
                press_previews.add(plan.floor[i])
        if i % 480 == 0:
            print(f"STEP {i}/{nsteps} floor={plan.floor[i]} phase={plan.phase[i]} lit={len(illuminated)}/12", flush=True)
    # Capture the returned home pose, with all spring buttons released and dark.
    for _ in range(30):
        world.step(render=False)
        world.render()
    rgba = camera.get_rgba()
    if rgba is not None and np.asarray(rgba).size:
        Image.fromarray(np.asarray(rgba)[..., :3].astype(np.uint8)).save(args.output / "completed.png")
    global_rgba = np.asarray(global_camera.get_rgba())
    if global_rgba.size:
        Image.fromarray(global_rgba[..., :3].astype(np.uint8)).save(args.output / "global_camera/completed.png")
    # Additional front view makes floor numbering and amber feedback legible.
    panel_prim = UsdGeom.Camera.Get(stage, scene["panel_camera_path"])
    panel_prim.GetHorizontalApertureAttr().Set(24.)
    panel_prim.GetVerticalApertureAttr().Set(36.)
    panel_camera = Camera(prim_path=scene["panel_camera_path"], name="panel_camera", resolution=(480, 720), frequency=-1)
    panel_camera.initialize()
    for _ in range(12):
        world.render()
    panel_rgba = panel_camera.get_rgba()
    if panel_rgba is not None and np.asarray(panel_rgba).size:
        Image.fromarray(np.asarray(panel_rgba)[..., :3].astype(np.uint8)).save(args.output / "panel_closeup.png")
    # Orthographic side view makes the folded upper/lower arm arrangement clear
    # without perspective foreshortening. The final pose is the same home as at start.
    home_camera = Camera(prim_path=scene["home_camera_path"], name="home_side_camera",
                         resolution=(1280, 720), frequency=-1)
    home_camera.initialize()
    for _ in range(8):
        world.render()
    home_rgba = np.asarray(home_camera.get_rgba())
    if home_rgba.size:
        Image.fromarray(home_rgba[..., :3].astype(np.uint8)).save(args.output / "home_side.png")
    # An actual RTX close-up exposes the reused bracket geometry and mounting
    # interface; the wrist camera cannot photograph its own assembly.
    from pressb.scene import _camera
    wrist_matrix = UsdGeom.Xformable(link6).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    mount_target = np.array(wrist_matrix.Transform(Gf.Vec3d(-.04, 0, .06)))
    mount_view_path = _camera(stage, "/World/WristAssemblyCamera",
                              mount_target + [.30, -.42, .22], mount_target, 60)
    UsdGeom.Camera.Get(stage, mount_view_path).CreateVerticalApertureAttr(27.)
    mount_camera = Camera(prim_path=mount_view_path, name="wrist_assembly_camera",
                          resolution=(960, 720), frequency=-1)
    mount_camera.initialize()
    # Allow this newly created close-up render product to converge.
    for _ in range(80):
        world.render()
    mount_rgba = np.asarray(mount_camera.get_rgba())
    if mount_rgba.size:
        closeup = Image.fromarray(mount_rgba[..., :3].astype(np.uint8))
        closeup.save(args.output / "wrist_mount_closeup.png")
        closeup.save(args.output / "gripper_closed.png")
    # A separate layout view shows the physical fixed camera and its support.
    setup_path = _camera(stage, "/World/GlobalCameraSetupView", [-1.1, .95, 1.75],
                         [-.08, .03, 1.00], 40)
    UsdGeom.Camera.Get(stage, setup_path).CreateVerticalApertureAttr(27.)
    setup_camera = Camera(prim_path=setup_path, name="global_setup_camera",
                          resolution=(960, 720), frequency=-1)
    setup_camera.initialize()
    for _ in range(60):
        world.render()
    setup_rgba = np.asarray(setup_camera.get_rgba())
    if setup_rgba.size:
        Image.fromarray(setup_rgba[..., :3].astype(np.uint8)).save(args.output / "global_camera_setup.png")
    export_scene("completed_scene.usda", desired)
    ordered = [e["floor"] for e in recorder.events if e["type"] == "button_pressed"]
    released = [e["floor"] for e in recorder.events if e["type"] == "button_released"]
    home_events = [e for e in recorder.events if e["type"] == "cycle_home"]
    camera_report = camera_recorder.finish()
    global_report = global_recorder.finish()
    complete_success = (ordered == cfg["sequence"] and len(recorder.rows) == len(plan.q)
                        and released == cfg["sequence"] and not illuminated
                        and [e["floor"] for e in home_events] == cfg["sequence"]
                        and all(e["q_error_rad"] <= cfg["home_tolerance_rad"] for e in home_events)
                        and camera_report["success"] and global_report["success"]
                        and max_fk_error < .005 and max_error < .15
                        and max_gripper_error <= .00025 and not unexpected)
    report = recorder.save({"success": complete_success,
                            "steps_expected": len(plan.q), "steps_completed": len(recorder.rows),
                            "max_joint_tracking_error_rad": max_error,
                            "max_usd_vs_urdf_tip_error_m": max_fk_error,
                            "max_button_travel_m": maximum_travel.tolist(),
                            "max_contact_force_n": maximum_force.tolist(), "rendered_frames": frame_count,
                            "unexpected_collisions": unexpected,
                            "home_returns": home_events, "wrist_camera": camera_report,
                            "global_camera": global_report,
                            "gripper": {"joint_names": list(gripper_names), "target_positions_m": gripper_target.tolist(),
                                        "maximum_tracking_error_m": max_gripper_error,
                                        "shaft_diameter_m": .008, "shaft_root_link6_z_m": shaft_root},
                            "asset_sources": scene.get("assets", {})})
    print(json.dumps(report, indent=2), flush=True)
    if args.video and frame_count:
        import subprocess
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(video_fps),
                        "-i", str(frame_dir / "%06d.png"), "-c:v", "libx264", "-crf", "20",
                        "-pix_fmt", "yuv420p", "-frames:v", str(frame_count),
                        str(args.output / "episode.mp4")], check=True)
    for sensor_recorder in (camera_recorder, global_recorder):
        if sensor_recorder.rows:
            import subprocess
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate",
                            str(1. / (cfg["physics_dt"] * sensor_recorder.stride)), "-i",
                            str(sensor_recorder.output / "rgb/%06d.png"), "-c:v", "libx264", "-crf", "20",
                            "-pix_fmt", "yuv420p", "-frames:v", str(len(sensor_recorder.rows)),
                            str(sensor_recorder.output / "rgb.mp4")], check=True)
    if args.hold and not args.headless:
        while app.is_running():
            world.step(render=True)
    if not report["success"] or len(recorder.rows) != len(plan.q) or max_fk_error > .005:
        raise RuntimeError("Episode did not pass complete physical validation; inspect report.json")


if __name__ == "__main__":
    main()
