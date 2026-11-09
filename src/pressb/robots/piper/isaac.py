"""PiPER articulation and camera runtime with lazy Isaac imports."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pressb.sensors.camera import make_camera, set_intrinsics


def build_piper_runtime(cfg, scene_builder, *, asset_root, camera_hardware_builder=None,
                        initialize=True):
    """Build a scene with the common PiPER drives and camera registration.

    ``scene_builder(stage, cfg)`` owns task geometry and returns root_path and
    robot_base_position_world. Wrist defaults retain elevator calibration; an
    explicit historical wrist registration remains supported for replay. The
    caller must own an active SimulationApp and an empty stage. Optional camera
    hardware is visual geometry supplied by the scene's builder callback.
    ``initialize=False`` authors a cloneable template without resetting physics,
    creating camera render products, or accessing articulation state. The shared
    parallel engine subsequently owns initialization and rendering.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema
    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import add_reference_to_stage
    from .wrist_camera import apply_black_materials, attach_wrist_camera
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
    initial_state = np.r_[cfg["initial_joint_positions_rad"], cfg["initial_gripper_width_m"]]
    dt, fps = cfg["physics_dt"], cfg["fps"]
    stride = round(1 / dt / fps)
    if abs(stride*dt-1/fps) > 1e-9:
        raise ValueError("Video fps must divide physics rate")
    world = World(stage_units_in_meters=1., physics_dt=dt, rendering_dt=1/fps,
                  backend="numpy", device="cpu")
    world.get_physics_context().set_physx_update_transformations_settings(
        update_to_usd=True, update_velocities_to_usd=True)
    stage = world.stage
    scene = scene_builder(stage, cfg)
    base = np.asarray(scene["robot_base_position_world"])
    add_reference_to_stage(str((Path(asset_root)/cfg["robot_usd"]).resolve()), "/World/Piper")
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
    camera_cfg = cfg["up_camera"]
    if "eye_base_m" in camera_cfg and camera_hardware_builder is not None:
        scene["table_camera_hardware"] = camera_hardware_builder(stage, np.asarray(camera_cfg["eye_base_m"])+base,
                     np.asarray(camera_cfg["target_base_m"])+base, cfg["table_height"],
                     root_path=scene["root_path"] + "/Tripod")
    camera_specs = {
        "up": (up_path, cfg["up_camera"]["resolution"]),
        "wrist": (wrist_info["camera_path"], cfg.get("wrist_camera", {}).get("resolution", cfg["wrist_camera_resolution"])),
    }
    camera_paths = {name: path for name, (path, _) in camera_specs.items()}
    if not initialize:
        return SimpleNamespace(world=world, stage=stage, scene=scene, base=base,
            robot_path="/World/Piper", robot=None, controller=None, indices=None,
            cameras={}, camera_paths=camera_paths, camera_specs=camera_specs,
            wrist_info=wrist_info, initial_state=initial_state, dt=dt, fps=fps,
            stride=stride, warmup_frames=warmup_frames, subframes=subframes)
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.sensors.camera import Camera
    robot = world.scene.add(SingleArticulation(prim_path="/World/Piper", name="piper"))
    cameras = {name: Camera(prim_path=path, name=name, resolution=tuple(size), frequency=-1)
               for name, (path, size) in camera_specs.items()}
    world.reset()
    for camera in cameras.values():
        camera.initialize()
    # Camera construction normalizes vertical aperture to square pixels. Real
    # calibration can have fx != fy; restore it after all sensor initialization.
    set_intrinsics(stage.GetPrimAtPath(up_path), cfg["up_camera"])
    if "wrist_camera" in cfg:
        set_intrinsics(wrist_prim, cfg["wrist_camera"])
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
    return SimpleNamespace(world=world, stage=stage, scene=scene, base=base,
        robot=robot, controller=controller, indices=indices, cameras=cameras,
        robot_path="/World/Piper", camera_paths=camera_paths,
        camera_specs=camera_specs, wrist_info=wrist_info, initial=initial,
        initial_state=initial_state, full_command=full_command, dt=dt, fps=fps,
        stride=stride, warmup_frames=warmup_frames, subframes=subframes)


def close_piper_runtime(runtime):
    """Release camera graphs before a caller replaces its owned Isaac stage."""
    runtime.world.stop()
    for camera in runtime.cameras.values():
        camera.pause()
        camera._stage_open_callback_fn(None)
    from omni.syntheticdata import SyntheticData
    SyntheticData.Get().reset()
    for camera in runtime.cameras.values():
        camera._custom_annotators.clear()
        camera.destroy()
