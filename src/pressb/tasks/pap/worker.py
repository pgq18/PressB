"""One stateful Isaac PAP environment; launched only by :mod:`.backend`.

All USD, PhysX, rendering, reset, and step calls stay on this process's main
thread. Waiting for a command never calls ``app.update`` or advances physics.
"""
from __future__ import annotations

import argparse
import base64
import copy
import io
import json
import math
from multiprocessing.connection import Client
from pathlib import Path
import traceback
from types import SimpleNamespace

import numpy as np

from pressb.runtime.isaac import set_intrinsics as _intrinsics, make_camera as _camera


def _build(config, *, initialize=True):
    """Same physical/contact/camera setup as run_pap_scene, with two sensors."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.sensors.camera import Camera
    from pressb.tasks.pap.scene import build_scene
    from pressb.runtime.workbench import build_tripod
    from pressb.runtime.scene import _bind, _cube, _material, _material_settings
    from pressb.wrist_camera import apply_black_materials, attach_wrist_camera
    import carb

    cfg = copy.deepcopy(config["scene"])
    cfg["object"] = copy.deepcopy(config["objects"][0])
    world = World(stage_units_in_meters=1., physics_dt=cfg["physics_dt"], rendering_dt=1/cfg["fps"],
                  backend="numpy", device="cpu")
    world.get_physics_context().set_physx_update_transformations_settings(
        update_to_usd=True, update_velocities_to_usd=True)
    stage = world.stage
    scene = build_scene(stage, cfg)
    base = np.asarray(scene["robot_base_position_world"])
    add_reference_to_stage(cfg["robot_usd"], "/World/Piper")
    robot_prim = stage.GetPrimAtPath("/World/Piper")
    transform = UsdGeom.Xformable(robot_prim)
    transform.ClearXformOpOrder()
    transform.AddTranslateOp().Set(Gf.Vec3d(*base))
    apply_black_materials(stage)
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
                joint.CreateUpperLimitAttr(cfg["gripper_max_width_m"]/2)
            elif prim.GetName() == "joint8":
                joint.CreateLowerLimitAttr(-cfg["gripper_max_width_m"]/2)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            api.CreateContactOffsetAttr(.0005)
            api.CreateRestOffsetAttr(0.)
    wrist_info = attach_wrist_camera(stage, "/World/Piper/link6", cfg)
    wrist_prim = stage.GetPrimAtPath(wrist_info["camera_path"])
    if "wrist_camera" in cfg:
        spec = cfg["wrist_camera"]
        _intrinsics(wrist_prim, spec)
        if "link6_from_camera_usd" in spec:
            cache = UsdGeom.XformCache()
            parent_world = np.array(cache.GetLocalToWorldTransform(wrist_prim.GetParent())).T
            link_world = np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath("/World/Piper/link6"))).T
            local = np.linalg.inv(parent_world) @ link_world @ np.asarray(spec["link6_from_camera_usd"])
            xf = UsdGeom.Xformable(wrist_prim)
            xf.ClearXformOpOrder()
            xf.AddTransformOp().Set(Gf.Matrix4d(*local.T.reshape(-1).tolist()))
    material = UsdShade.Material.Define(stage, "/World/PickPlace/Looks/FingerContact")
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(cfg["finger_static_friction"])
    physics.CreateDynamicFrictionAttr(cfg["finger_dynamic_friction"])
    physics.CreateRestitutionAttr(0.)
    for prim in Usd.PrimRange(robot_prim):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
        if prim.HasAPI(UsdPhysics.CollisionAPI) and any(f"/link{i}/" in str(prim.GetPath()) for i in (7, 8)):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")
    up_path = _camera(stage, "/World/PickPlace/UpCamera", cfg["up_camera"], base)
    spec = cfg["up_camera"]
    if "eye_base_m" in spec:
        build_tripod(stage, np.asarray(spec["eye_base_m"])+base,
                     np.asarray(spec["target_base_m"])+base, cfg["table_height"])
    robot = world.scene.add(SingleArticulation(prim_path="/World/Piper", name="piper"))
    object_paths = [scene["object_path"]]
    material_paths = [scene["appearance"]["object_material"]["path"]]
    for index, obj in enumerate(config["objects"][1:], start=1):
        # Pre-create rigid variants. Reset enables exactly one body and parks the
        # others below the floor; no geometry rebuild occurs during a transition.
        path = f"/World/PickPlace/ObjectVariant{index}"
        geo = obj["geometry"]
        appearance = _material_settings(scene["appearance"]["materials"]["RedCube"],
            obj.get("material", {}), f"objects[{index}].material")
        object_material = _material(stage, f"/World/PickPlace/Looks/ObjectVariant{index}", obj["color"],
            roughness=appearance["roughness"], metallic=appearance["metallic"],
            specular=appearance["specular"] if appearance["specular_authored"] else None,
            static_friction=.8, dynamic_friction=.65)
        material_paths.append(str(object_material.GetPath()))
        if geo["shape"] == "cylinder":
            shape = UsdGeom.Cylinder.Define(stage, path)
            shape.CreateAxisAttr("Z")
            shape.CreateRadiusAttr(geo["radius"])
            shape.CreateHeightAttr(geo["height"])
            shape.AddTranslateOp().Set(Gf.Vec3d(0, 0, -5-index))
            _bind(shape.GetPrim(), object_material, collision=True)
            radius, height = geo["radius"], geo["height"]
            transverse = obj["mass"]*(3*radius*radius+height*height)/12
            inertia = [transverse, transverse, obj["mass"]*radius*radius/2]
        else:
            shape = _cube(stage, path, [0, 0, -5-index], geo["size_xyz"], object_material, collision=True)
            x, y, z = geo["size_xyz"]
            inertia = obj["mass"]/12*np.array([y*y+z*z, x*x+z*z, x*x+y*y])
        body = UsdPhysics.RigidBodyAPI.Apply(shape.GetPrim())
        body.CreateRigidBodyEnabledAttr(True)
        body.CreateKinematicEnabledAttr(False)
        mass = UsdPhysics.MassAPI.Apply(shape.GetPrim())
        mass.CreateMassAttr(obj["mass"])
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in inertia]))
        mass.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))
        api = PhysxSchema.PhysxRigidBodyAPI.Apply(shape.GetPrim())
        api.CreateSolverPositionIterationCountAttr(16)
        api.CreateSolverVelocityIterationCountAttr(4)
        api.CreateEnableCCDAttr(True)
        object_paths.append(path)
    objects = [world.scene.add(SingleRigidPrim(prim_path=path, name=f"object{i}"))
               for i, path in enumerate(object_paths)]
    cameras = {
        "up": Camera(prim_path=up_path, name="up", resolution=tuple(cfg["up_camera"]["resolution"]), frequency=-1),
        "wrist": Camera(prim_path=wrist_info["camera_path"], name="wrist",
            resolution=tuple(cfg.get("wrist_camera", {}).get("resolution", cfg["wrist_camera_resolution"])), frequency=-1),
    } if initialize else {}
    rendering = cfg.get("rendering", {})
    settings = carb.settings.get_settings()
    for key, setting in {"direct_samples_per_pixel": "/rtx/directLighting/sampledLighting/samplesPerPixel",
                         "reflection_samples_per_pixel": "/rtx/reflections/sampledLighting/samplesPerPixel"}.items():
        if key in rendering:
            if type(rendering[key]) is not int or rendering[key] not in (1, 2, 4, 8):
                raise ValueError(f"rendering.{key} must be 1, 2, 4, or 8")
            settings.set_int(setting, rendering[key])
    if not initialize:
        # The vector backend reuses this exact USD construction as a template,
        # before any physics handles or per-camera render products are created.
        return SimpleNamespace(world=world, stage=stage, scene=scene, base=base,
            robot=robot, objects=objects, object_paths=object_paths,
            material_paths=material_paths,
            camera_paths={"up": up_path, "wrist": wrist_info["camera_path"]})
    world.reset()
    for camera in cameras.values():
        camera.initialize()
    indices = np.array([robot.get_dof_index(f"joint{i}") for i in range(1, 9)])
    controller = robot.get_articulation_controller()
    kps, kds = np.zeros(robot.num_dof), np.zeros(robot.num_dof)
    kps[indices[:6]], kds[indices[:6]] = cfg["arm_stiffness"], cfg["arm_damping"]
    kps[indices[6:]], kds[indices[6:]] = cfg["gripper_stiffness"], cfg["gripper_damping"]
    controller.set_gains(kps=kps, kds=kds)
    return SimpleNamespace(world=world, stage=stage, scene=scene, base=base, robot=robot,
        controller=controller, indices=indices, objects=objects, object_paths=object_paths,
        material_paths=material_paths, cameras=cameras)


class PAPEnvironment:
    """A single episode state machine; no simulator import until construction."""
    def __init__(self, config):
        from pressb.action_postprocessing import JointActionPostprocessor
        from pressb.tasks.pap.initialization import load_policy_initial
        from pressb.kinematics import PiperKinematics
        from pxr import PhysicsSchemaTools
        import omni.physx

        self.config = config
        self.runtime = _build(config)
        self.initial = load_policy_initial(config["trajectory"], config["scene"]["gripper_max_width_m"])["initial"]
        kin = PiperKinematics(config["scene"]["robot_urdf"], tip_offset=.1358)
        self.lower = np.r_[kin.lower, 0.]
        self.upper = np.r_[kin.upper, config["scene"]["gripper_max_width_m"]]
        if np.any(self.initial < self.lower-1e-6) or np.any(self.initial > self.upper+1e-6):
            raise ValueError("Initial recorded observation exceeds URDF limits")
        self.postprocessor = JointActionPostprocessor(**config["smoothing"])
        self.contact_counts = {"link7": 0, "link8": 0}
        self.finger_contact = False
        self.active_object = 0
        self.ready = False
        self.ticks = 0
        self.last_sent = self.initial.copy()
        self.max_ticks = max(1, math.floor(config["max_episode_seconds"] / config["physics_dt"] + 1e-9))
        self.warmup_frames = config["scene"].get("rendering", {}).get("warmup_frames", 8)
        self.subframes = config["scene"].get("rendering", {}).get("subframes_per_frame", 1)
        if type(self.warmup_frames) is not int or not 8 <= self.warmup_frames <= 256:
            raise ValueError("rendering.warmup_frames must be in [8, 256]")
        if type(self.subframes) is not int or not 1 <= self.subframes <= 8:
            raise ValueError("rendering.subframes_per_frame must be in [1, 8]")

        def on_contact(headers, data):
            for header in headers:
                actors = [str(PhysicsSchemaTools.intToSdfPath(value)) for value in (header.actor0, header.actor1)]
                if self.runtime.object_paths[self.active_object] not in actors or not header.num_contact_data:
                    continue
                for name in self.contact_counts:
                    if f"/World/Piper/{name}" in actors:
                        self.finger_contact = True
                        self.contact_counts[name] += 1
        self.subscription = omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(on_contact)

    def _full_command(self, command):
        value = np.zeros(self.runtime.robot.num_dof)
        value[self.runtime.indices] = np.r_[command[:6], command[6]/2, -command[6]/2]
        return value

    def _state(self):
        value = self.runtime.robot.get_joint_positions()[self.runtime.indices]
        state = np.r_[value[:6], value[6]-value[7]]
        if not np.isfinite(state).all():
            raise RuntimeError("Non-finite simulated robot state")
        return state

    def _observe(self):
        from PIL import Image
        images = {}
        for name, camera in self.runtime.cameras.items():
            rgba = np.asarray(camera.get_rgba())
            if rgba.ndim != 3 or rgba.shape[2] != 4 or rgba.dtype != np.uint8 or rgba.size == 0:
                raise RuntimeError(f"Missing or invalid rendered RGB sensor {name}")
            image = Image.fromarray(np.ascontiguousarray(rgba[..., :3]))
            buffer = io.BytesIO()
            image.save(buffer, format="PNG", compress_level=3)
            images[name] = {"encoding": "png_base64", "data": base64.b64encode(buffer.getvalue()).decode("ascii")}
        return {"state": self._state().tolist(), "images": images, "task": self.reset_spec["task"],
            "model_inputs": copy.deepcopy(self.reset_spec["model_inputs"]),
            "control_state": self.last_sent.tolist(), "task_features": []}

    def reset(self, spec):
        from pxr import Gf, UsdGeom, UsdShade
        from isaacsim.core.utils.types import ArticulationAction
        from pressb.tasks.pap.task import PickPlaceOutcome
        rt = self.runtime
        self.ready = False
        self.active_object = spec["object_index"]
        self.reset_spec = copy.deepcopy(spec)
        full = self._full_command(self.initial)
        rt.robot.set_joint_positions(full)
        rt.robot.set_joint_velocities(np.zeros(rt.robot.num_dof))
        rt.controller.apply_action(ArticulationAction(joint_positions=full))
        obj = self.config["objects"][self.active_object]
        position = np.r_[np.asarray(spec["center_xy"])+rt.base[:2],
            self.config["scene"]["table_height"]+obj["geometry"]["size_xyz"][2]/2+obj.get("clearance", .0005)]
        yaw = math.radians(spec["yaw_deg"])/2
        quaternion = np.array([math.cos(yaw), 0., 0., math.sin(yaw)])
        for index, body in enumerate(rt.objects):
            body.set_linear_velocity(np.zeros(3))
            body.set_angular_velocity(np.zeros(3))
            imageable = UsdGeom.Imageable(rt.stage.GetPrimAtPath(rt.object_paths[index]))
            if index == self.active_object:
                body.set_world_pose(position, quaternion)
                body.enable_rigid_body_physics()
                imageable.MakeVisible()
            else:
                body.set_world_pose(np.array([0., 0., -5.-index]), np.array([1., 0., 0., 0.]))
                body.disable_rigid_body_physics()
                imageable.MakeInvisible()
        material_path = rt.material_paths[self.active_object]
        for suffix, input_name in (("Mdl", "diffuse_color_constant"), ("Preview", "diffuseColor")):
            UsdShade.Shader(rt.stage.GetPrimAtPath(f"{material_path}/{suffix}")).GetInput(input_name).Set(Gf.Vec3f(*spec["color_rgb"]))
        for _ in range(self.config["settling_ticks"]):
            rt.world.step(render=False)
        for _ in range(self.warmup_frames):
            rt.world.render()
        self.ticks = 0
        self.contact_counts = {"link7": 0, "link8": 0}
        self.finger_contact = False
        self.last_sent = self._state()
        self.postprocessor.reset(self.last_sent)
        cavity = rt.scene["box_inside"]
        box = dict(center_world=cavity["origin_world"], yaw_deg=cavity["yaw_deg"],
            inner_size_xy=(np.asarray(cavity["max_local"])-np.asarray(cavity["min_local"]))[:2].tolist(),
            floor_z_world=cavity["origin_world"][2]+cavity["min_local"][2],
            rim_z_world=cavity["origin_world"][2]+cavity["max_local"][2])
        self.outcome = PickPlaceOutcome(obj["geometry"], self.config["scene"]["table_height"], box,
            **self.config["outcome"])
        observation = self._observe()
        self.ready = True
        return {"env_id": self.config["env_id"], "observation": observation, "info": {
            "reset": copy.deepcopy(spec), "object_geometry": obj["geometry"],
            "object_position_world": rt.objects[self.active_object].get_world_pose()[0].tolist(),
            "settling_physics_steps": self.config["settling_ticks"], "elapsed_control_steps": 0.,
            "smoothing": self.postprocessor.summary(), "success": False}}

    def step(self, payload):
        from isaacsim.core.utils.types import ArticulationAction
        if not self.ready:
            raise ValueError("Environment requires reset before step; no automatic reset")
        actions = np.asarray(payload["actions"])
        if (actions.ndim != 2 or actions.shape[1] != 7 or not 1 <= len(actions) <= self.config["horizon"]
                or actions.dtype.kind not in "fiu" or not np.isfinite(actions).all()
                or np.any(actions < self.lower-1e-6) or np.any(actions > self.upper+1e-6)):
            raise ValueError("Invalid PAP executable actions")
        rt, cfg = self.runtime, self.config
        ticks_before = self.ticks
        terminated = truncated = success = False
        termination_reason = None
        sent_actions = []
        result = {}
        cube = rt.objects[self.active_object]
        for command in actions:
            self.last_sent = self.postprocessor.apply(command)
            sent_actions.append(self.last_sent.tolist())
            target = self._full_command(self.last_sent)
            for _ in range(cfg["physics_ticks_per_action"]):
                self.finger_contact = False
                rt.controller.apply_action(ArticulationAction(joint_positions=target))
                rt.world.step(render=False)
                self.ticks += 1
                position, quaternion = cube.get_world_pose()
                velocity = np.r_[cube.get_linear_velocity(), cube.get_angular_velocity()]
                if not np.isfinite(np.r_[position, quaternion, velocity]).all():
                    raise RuntimeError("Non-finite rigid-body physics state")
                result = self.outcome.update(self.ticks*cfg["physics_dt"], position, quaternion, velocity,
                    float(self._state()[6]), finger_contact=self.finger_contact)
                success = bool(result["success"] and all(self.contact_counts.values()))
                dropped = cfg["terminate_on_drop"] and position[2] < cfg["scene"]["table_height"]-cfg["drop_below_table_m"]
                if success or dropped:
                    terminated = True
                    termination_reason = "success" if success else "object_dropped"
                elif self.ticks >= self.max_ticks:
                    truncated = True
                    termination_reason = "timeout"
                if terminated or truncated:
                    break
            if terminated or truncated:
                break
        for _ in range(self.subframes):
            rt.world.render()
        elapsed_ticks = self.ticks-ticks_before
        elapsed_controls = elapsed_ticks/cfg["physics_ticks_per_action"]
        reward = cfg["reward"]["success"] if success else cfg["reward"]["failure"] if terminated else 0.
        # Event reward at the last physical tick; gamma is per control interval,
        # and returned discount always reports actual elapsed simulated time.
        reward *= cfg["gamma"] ** ((elapsed_ticks-1)/cfg["physics_ticks_per_action"])
        self.ready = not (terminated or truncated)
        return {"env_id": cfg["env_id"], "observation": self._observe(), "reward": float(reward),
            "discount": cfg["gamma"] ** elapsed_controls, "terminated": terminated, "truncated": truncated,
            "info": {"success": success, "termination_reason": termination_reason,
                "physics_steps": elapsed_ticks, "elapsed_control_steps": elapsed_controls,
                "episode_physics_steps": self.ticks, "episode_seconds": self.ticks*cfg["physics_dt"],
                "physical_outcome": result, "contact_counts": dict(self.contact_counts),
                "applied_actions": sent_actions, "requested_action_count": len(actions),
                "object_position_world": cube.get_world_pose()[0].tolist(),
                "object_teleport_after_reset": False}}

    def close(self):
        self.subscription = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--auth", required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    app = environment = None
    connection = Client((args.host, args.port), authkey=bytes.fromhex(args.auth))
    try:
        from isaacsim import SimulationApp
        app = SimulationApp({"headless": True, "active_gpu": config["gpu"], "physics_gpu": config["gpu"],
            "multi_gpu": False, "renderer": "RayTracedLighting", "anti_aliasing": 1, "fast_shutdown": True,
            "extra_args": ["--/app/asyncRendering=false", "--/rtx/ecoMode/enabled=false",
                           "--/validate/p2p/enabled=false", "--/validate/iommu/enabled=false"]})
        environment = PAPEnvironment(config)
        connection.send_bytes(json.dumps({"ok": True, "result": {"ready": True}}).encode())
        while True:
            request = json.loads(connection.recv_bytes(64*1024*1024))
            method, payload = request["method"], request["payload"]
            if method == "close":
                break
            if method not in ("reset", "step"):
                raise ValueError(f"Unknown PAP worker method {method}")
            result = getattr(environment, method)(payload)
            connection.send_bytes(json.dumps({"ok": True, "result": result}, allow_nan=False).encode())
    except (EOFError, BrokenPipeError, ConnectionResetError):
        pass
    except BaseException as error:
        traceback.print_exc()
        try:
            connection.send_bytes(json.dumps({"ok": False, "error": f"{type(error).__name__}: {error}"}).encode())
        except (EOFError, OSError):
            pass
        raise
    finally:
        if environment:
            environment.close()
        connection.close()
        if app:
            app.app.post_quit(0)
            app.close()


if __name__ == "__main__":
    main()
