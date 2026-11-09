"""Elevator scene, randomization and physical outcomes for the shared runtime.

The common engine owns cloning, simulation clocks, physics stepping and cameras.
The PiPER pose driver owns IK, interpolation and smoothing. This task only
authors its scene and interprets the twelve spring-loaded buttons and contacts.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from pressb.runtime.contract import finite_array, fingerprint
from pressb.runtime.parallel_contract import ParallelTask, SceneTemplate
from .configuration import ROOT, load_config
from .collection import DATASET_ANTI_ALIASING, RENDER_EXPOSURE_CONTROLS, RENDER_HISTORY_CONTROLS
from .semantics import task_floor

FLOORS = tuple(range(24, 36))
BUTTON_PATHS = tuple(f"/World/Panel/Floor{floor}/Cap" for floor in FLOORS)
RESET_CONTRACT = "shared_parallel_indexed_zero_tick_elevator_v1"


def panel_positions(scene, origin, offset_x_m, offset_y_m):
    origin = finite_array(origin, (3,), "environment origin", dtype=np.float64)
    return np.asarray([[scene["button_face_x"] + offset_x_m + .003,
        scene["button_column_y"] * (1 if floor < 30 else -1) + offset_y_m,
        scene["button_bottom_z"] + (floor - 24) % 6 * scene["button_pitch_z"]]
        for floor in FLOORS], dtype=np.float64) + origin


def classify_contact(paths):
    """Classify a contact after the engine localizes paths to the template."""
    if len(paths) != 4:
        raise ValueError("Elevator contacts require actor0/actor1/collider0/collider1")
    actors, colliders = paths[:2], paths[2:]
    matches = [re.fullmatch(r"/World/Panel/Floor(\d+)/Cap", path) for path in actors]
    match = next((value for value in matches if value is not None), None)
    button = int(match[1]) - 24 if match is not None else None
    if button is not None and not 0 <= button < 12:
        raise ValueError("Contact references an unknown elevator button")
    prefix = "/World/Piper/"
    tools = {prefix + "link6/PressStylus", prefix + "link6/PressTip"}
    if button is not None and tools.intersection(colliders):
        return button, None
    robot_actors = [path for path in actors if path.startswith(prefix)]
    pads = {prefix + "link7/collisions", prefix + "link8/collisions"}
    if prefix + "link6/PressStylus" in colliders and pads.intersection(colliders):
        return None, None
    reason = ("non_stylus_button_contact" if button is not None and robot_actors else
        "robot_self_contact" if len(robot_actors) == 2 and actors[0] != actors[1] else
        "robot_environment_contact" if len(robot_actors) == 1 and
        robot_actors[0].rsplit("/", 1)[-1] not in ("dummy_link", "base_link", "link1") else None)
    return None, reason


class ElevatorTask(ParallelTask):
    def __init__(self, config):
        self.config = load_config(config)
        cfg = self.config
        if cfg["gpu_dynamics"]:
            raise ValueError("Shared parallel elevator currently uses CPU PhysX; gpu_dynamics is unsupported")
        self.scene = json.loads(cfg["config"].read_text())
        for name in ("table_height", "button_face_x", "button_bottom_z", "button_pitch_z",
                     "button_column_y", "press_threshold", "release_threshold"):
            value = self.scene.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Frozen elevator scene requires finite {name}")
        if abs(float(self.scene.get("physics_dt", 0)) - 1 / 120) > 1e-12:
            raise ValueError("Elevator physics_dt must be 1/120")
        self.home = finite_array(self.scene.get("home_q"), (6,), "home_q", dtype=np.float64)
        fingers = finite_array(self.scene.get("gripper_joint_positions_m"), (2,), "gripper joints")
        if not np.allclose(fingers, [.004, -.004], rtol=0., atol=1e-9):
            raise ValueError("Elevator requires fixed gripper joints [+0.004,-0.004] m")
        urdf = Path(self.scene["robot_urdf"])
        self.urdf = urdf if urdf.is_absolute() else ROOT / urdf
        if not self.urdf.is_file():
            raise FileNotFoundError(self.urdf)
        self.floors, self.randomize_panel = cfg.get("floors", list(FLOORS)), cfg.get("panel_randomization", True)
        randomization = self.scene.get("panel_randomization", {})
        self.bounds = {f"{which}_offset_{axis}_m": float(randomization[f"{which}_offset_{axis}_m"]
            if randomization.get("enabled") is True else self.scene.get(f"panel_offset_{axis}_m", 0.))
            for axis in "xy" for which in ("min", "max")}
        for axis in "xy":
            low, high = (self.bounds[f"{name}_offset_{axis}_m"] for name in ("min", "max"))
            if not np.isfinite([low, high]).all() or low > high:
                raise ValueError("Invalid frozen panel randomization bounds")
        square = cfg["camera_resolution"] == 224
        self.execution = dict(num_envs=cfg["num_envs"], gpu=cfg["gpu"], physics_dt=1/120,
            control_hz=30, horizon=7, max_episode_seconds=cfg["max_seconds"], settling_ticks=90,
            spacing_m=6., render_resolution=[320, 240] if square else [640, 480],
            image_output_resolution=[224, 224] if square else [640, 480],
            render_subframes=cfg["render_subframes"], observation_render_frames=1,
            reset_render_frames=1 + cfg["light_settle_captures"],
            reset_render_subframes=cfg["light_settle_subframes"],
            cpu_threads=cfg["cpu_threads"], png_workers=cfg["png_workers"], log_dir=str(cfg["output"]),
            gamma=cfg["single_gamma"], terminal_boundary="physics_tick", timeout_reason="time_limit",
            headless=True, renderer_profile="calibrated", anti_aliasing=DATASET_ANTI_ALIASING,
            renderer_settings={**RENDER_HISTORY_CONTROLS, **RENDER_EXPOSURE_CONTROLS})
        self.engine = None
        for key, expected in cfg.get("expected_metadata", {}).items():
            actual = self.metadata().get(key)
            if actual != expected:
                raise ValueError(f"Shared elevator metadata mismatch for {key}: {actual!r} != {expected!r}")

    def metadata(self):
        cfg, window = self.config, self.config["smoothing_window"]
        paths = ("tasks/elevator/parallel.py", "tasks/elevator/configuration.py", "tasks/elevator/dataset_scene.py",
                 "tasks/elevator/scene.py", "tasks/elevator/collection.py", "robots/piper/batched_control.py")
        sources = {name: {"sha256": hashlib.sha256((ROOT/"src/pressb"/name).read_bytes()).hexdigest()}
                   for name in paths}
        scene_hash = hashlib.sha256(cfg["snapshot"].read_bytes()).hexdigest()
        return dict(schema_version=2, node_type="simulation", task_id="elevator", num_envs=cfg["num_envs"],
            observation=dict(state=dict(names=["x", "y", "z", "qw", "qx", "qy", "qz", "gripper_width"],
                units=["m"]*3 + ["unitless"]*4 + ["m"]), cameras=["global", "wrist"]),
            action=dict(representation="pose9", frame="robot_base", names=["x", "y", "z"] +
                [f"rot6d_{i}" for i in range(6)], units=["m"]*3 + ["unitless"]*6, horizon=7, control_hz=30),
            control_state_dim=19+6*window, task_features_dim=12,
            discount=dict(gamma=cfg["single_gamma"], timebase_hz=30), reset_mode="indexed",
            capabilities=["per_env_reset_seeds"], scheduling="all_active", initial_reset="all_envs",
            execution_horizons=[7], terminal_observation_modes=["full"], fixed_gripper_width_m=.008,
            quaternion_order="wxyz", rotation6d="first_two_matrix_rows_row_major", smoothing=dict(window=window),
            task_config_sha256=fingerprint(dict(scene=self.scene, snapshot=scene_hash, floors=self.floors,
                bounds=self.bounds, randomize_panel=self.randomize_panel,
                execution={k:v for k,v in self.execution.items() if k != "log_dir"}, reset=RESET_CONTRACT)),
            source_identity=sources, source_revision="sha256:"+fingerprint(sources),
            asset_sha256=dict(snapshot=scene_hash, robot_urdf=hashlib.sha256(self.urdf.read_bytes()).hexdigest()),
            reward=dict(kind="sparse_physical_success", success=1., other=0.),
            execution=dict(backend="shared_parallel_isaac", reset=RESET_CONTRACT,
                control="urdf_analytic_jacobian_bounded_dls_v1", physics_hz=120,
                terminal_observation="exact_terminal_physics_tick"))

    def build_template(self):
        from isaacsim.core.api import World
        from pxr import UsdGeom
        import carb
        snapshot = self.config["snapshot"]
        if self.config["asset_bundle"] is not None:
            from pressb.runtime.assets import prepare_runtime_snapshot
            relocated = prepare_runtime_snapshot(snapshot, self.config["asset_bundle"], self.config["output"]/"assets")
            snapshot = Path(relocated["runtime_snapshot"])
        world = World(stage_units_in_meters=1., physics_dt=1/120, rendering_dt=1/30, backend="numpy", device="cpu")
        stage = world.stage
        root = UsdGeom.Xform.Define(stage, "/World")
        if not root.GetPrim().GetReferences().AddReference(str(snapshot), "/World"):
            raise RuntimeError("Could not reference the frozen elevator scene")
        UsdGeom.Xformable(stage.GetPrimAtPath("/World/GlobalCamera")).SetResetXformStack(False)
        for key, value in self.execution["renderer_settings"].items():
            carb.settings.get_settings().set(key, value)
        return SceneTemplate(world, stage, "/World/Piper",
            {"global":"/World/GlobalCamera/CameraLink/ColorCamera",
             "wrist":"/World/Piper/link6/WristRealSense/CameraLink/ColorCamera"},
            data=dict(scene=deepcopy(self.scene), robot_urdf=str(self.urdf), base=[self.scene.get("robot_base_x",0.),
                self.scene.get("robot_base_y",0.), self.scene["table_height"]]))

    def create_robot(self, engine):
        from pressb.robots.piper.parallel import PiperPoseDriver
        return PiperPoseDriver(engine, {**self.scene, "robot_urdf":str(self.urdf)}, np.r_[self.home,.008],
            smoothing_window=self.config["smoothing_window"], ik_iterations=self.config["ik_iterations"],
            fixed_gripper_width_m=.008)

    def bind(self, engine):
        super().bind(engine)
        from pxr import UsdGeom, UsdPhysics, UsdShade
        self.buttons, self.button_rows = engine.rigid_bodies(BUTTON_PATHS)
        self.forces, self.items = np.zeros((engine.num_envs,12)), [None]*engine.num_envs
        self.panels, self.source_offsets = [], []
        for prefix in engine.prefixes:
            panel = engine.stage.GetPrimAtPath(prefix+"/Panel")
            self.source_offsets.append((float(panel.GetAttribute("pressb:panelOffsetXM").Get() or 0.),
                                        float(panel.GetAttribute("pressb:panelOffsetYM").Get() or 0.)))
            lights, transforms, anchors = {}, {}, []
            for floor in FLOORS:
                cap = engine.stage.GetPrimAtPath(prefix+f"/Panel/Floor{floor}/Cap")
                shader = UsdShade.Shader.Get(engine.stage, prefix+f"/Looks/Button{floor}/Shader")
                lights[floor] = dict(diffuse=shader.GetInput("diffuseColor"), emissive=shader.GetInput("emissiveColor"),
                                     illuminated=cap.GetAttribute("pressb:illuminated"))
                if not all(lights[floor].values()):
                    raise ValueError("Frozen button feedback material is incomplete")
                transforms[floor] = [(str(op.GetName()),op.Get()) for op in UsdGeom.Xformable(cap).GetOrderedXformOps()]
                joint = UsdPhysics.Joint.Get(engine.stage, prefix+f"/Panel/Floor{floor}/SpringJoint")
                for relation, attribute in ((joint.GetBody0Rel(),joint.GetLocalPos0Attr()),(joint.GetBody1Rel(),joint.GetLocalPos1Attr())):
                    if not relation.GetTargets():
                        anchors.append((attribute,np.asarray(attribute.Get(),dtype=float).copy()))
            if len(anchors) != 12:
                raise ValueError("Each elevator requires twelve world-anchored button springs")
            self.panels.append(dict(prefix=prefix,lights=lights,transforms=transforms,anchors=anchors))
        self.rests = np.stack([panel_positions(self.scene,origin,*offset)
                              for origin,offset in zip(engine.offsets,self.source_offsets)])

    def after_settle(self):
        positions, orientations = self.buttons.get_world_poses()
        self.settled_relative = np.asarray(positions)[self.button_rows].copy()-self.rests
        self.settled_orientations = np.asarray(orientations)[self.button_rows].copy()
        self.settled_velocities = np.asarray(self.buttons.get_velocities())[self.button_rows].copy()
        if np.max(np.linalg.norm(self.settled_relative,axis=-1)) > .0002:
            raise RuntimeError("Elevator buttons did not settle near their spring anchors")

    def sample_reset(self, env_id, seed, options):
        if not isinstance(options,dict) or set(options)-{"floor","task","offset_x_m","offset_y_m"}:
            raise ValueError("Elevator reset supports floor/task and panel x/y offsets")
        rng = np.random.default_rng(np.random.SeedSequence([seed,env_id]))
        floor = options.get("floor",int(rng.choice(self.floors)))
        if "task" in options:
            requested = task_floor(options["task"])
            if "floor" in options and floor != requested:
                raise ValueError("floor and task name different elevator targets")
            floor = requested
        if type(floor) is not int or floor not in self.floors:
            raise ValueError("Reset floor is outside the configured floor set")
        spec = dict(floor=floor,task=f"Press {floor} floor.")
        for axis in "xy":
            low,high = (self.bounds[f"{which}_offset_{axis}_m"] for which in ("min","max"))
            default = rng.uniform(low,high) if self.randomize_panel else self.scene.get(f"panel_offset_{axis}_m",0.)
            value = finite_array([options.get(f"offset_{axis}_m",default)],(1,),f"offset_{axis}_m",dtype=np.float64)[0]
            if not low <= value <= high:
                raise ValueError("Panel reset offset exceeds the frozen bounds")
            spec[f"offset_{axis}_m"] = float(value)
        return spec

    def _set_light(self, eid, floor, on):
        from pxr import Gf
        from .scene import BUTTON_OFF_COLOR, BUTTON_ON_COLOR, BUTTON_ON_EMISSION
        inputs = self.panels[eid]["lights"][floor]
        inputs["diffuse"].Set(Gf.Vec3f(*(BUTTON_ON_COLOR if on else BUTTON_OFF_COLOR)))
        inputs["emissive"].Set(Gf.Vec3f(*(BUTTON_ON_EMISSION if on else (0.,0.,0.))))
        inputs["illuminated"].Set(bool(on))

    def reset_task(self, env_ids, specs):
        from pxr import Gf
        from .dataset_scene import _panel_shift
        positions = []
        for eid in env_ids:
            spec,panel = specs[eid],self.panels[eid]
            x,y = spec["offset_x_m"],spec["offset_y_m"]
            source_x,source_y = self.source_offsets[eid]
            _panel_shift(self.engine.stage,panel["prefix"],source_y,y,source_x,x)
            delta = np.asarray([x-source_x,y-source_y,0.])
            for attribute,initial in panel["anchors"]:
                attribute.Set(Gf.Vec3f(*(initial+delta)))
            for floor in FLOORS:
                cap = self.engine.stage.GetPrimAtPath(panel["prefix"]+f"/Panel/Floor{floor}/Cap")
                for name,value in panel["transforms"][floor]:
                    cap.GetAttribute(name).Set(value)
                self._set_light(eid,floor,False)
            self.rests[eid] = panel_positions(self.scene,self.engine.offsets[eid],x,y)
            positions.append(self.rests[eid]+self.settled_relative[eid])
            self.items[eid] = dict(floor=spec["floor"],lit=set(),events=[],collisions=[],
                min_target_distance=1e6,max_tracking_error=0.,max_target_force=0.,max_target_travel=0.)
            self.forces[eid] = 0.
        ids = np.asarray(env_ids,dtype=np.int64)
        rows = self.button_rows[ids].reshape(-1)
        self.buttons.set_world_poses(np.asarray(positions).reshape(-1,3),
            self.settled_orientations[ids].reshape(-1,4),indices=rows)
        self.buttons.set_velocities(self.settled_velocities[ids].reshape(-1,6),indices=rows)
        return {eid:dict(reset=deepcopy(specs[eid]),success=False,reset_physics_steps=0,
                        reset_contract=RESET_CONTRACT) for eid in env_ids}

    def before_tick(self, env_ids):
        self.forces[np.asarray(env_ids,dtype=np.int64)] = 0.

    def on_contact(self, env_id, actor_paths, contacts):
        if self.items[env_id] is None or not self.engine.backend.ready[env_id]:
            return
        button,reason = classify_contact(tuple(map(str,actor_paths)))
        if button is None and reason is None:
            return
        force = sum(float(np.linalg.norm(contact.impulse)) for contact in contacts)/self.execution["physics_dt"]
        if not math.isfinite(force):
            raise RuntimeError("Nonfinite elevator contact force")
        if button is not None:
            self.forces[env_id,button] += force
        elif force > .1 and len(self.items[env_id]["collisions"]) < 100:
            self.items[env_id]["collisions"].append(dict(physics_index=int(self.engine.backend.ticks[env_id]),
                actors=list(actor_paths[:2]),colliders=list(actor_paths[2:]),force_n=force,reason=reason))

    def after_tick(self, env_ids):
        robot = self.engine.robot
        q,_ = robot.joint_arrays()
        arm = q[:,robot.arm_dofs]
        transform = robot.batched_controller.fk(arm)
        base = np.asarray(self.engine.template.data["base"])+self.engine.offsets
        tips = transform[:,:3,3]+transform[:,:3,2]*(.24-.1358)+base
        positions = np.asarray(self.buttons.get_world_poses()[0])[self.button_rows]
        travel = positions[:,:,0]-self.rests[:,:,0]
        if not np.isfinite(np.r_[travel.ravel(),tips.ravel(),self.forces.ravel()]).all():
            raise RuntimeError("Nonfinite elevator physical metrics")
        output = {}
        for eid in env_ids:
            item = self.items[eid]
            index,ticks = item["floor"]-24,int(self.engine.backend.ticks[eid])
            target = self.rests[eid,index]-[.003,0.,0.]
            item["min_target_distance"] = min(item["min_target_distance"],float(np.linalg.norm(tips[eid]-target)))
            item["max_tracking_error"] = max(item["max_tracking_error"],float(np.max(np.abs(arm[eid]-robot.commands[eid,robot.arm_dofs]))))
            item["max_target_force"] = max(item["max_target_force"],float(self.forces[eid,index]))
            item["max_target_travel"] = max(item["max_target_travel"],float(travel[eid,index]))
            pressed = (travel[eid]>=self.scene["press_threshold"]) & (self.forces[eid]>.02)
            released = travel[eid]<=self.scene["release_threshold"]
            for floor in FLOORS:
                on = bool(pressed[floor-24])
                if (floor not in item["lit"] and on) or (floor in item["lit"] and released[floor-24]):
                    item["lit"].add(floor) if on else item["lit"].remove(floor)
                    self._set_light(eid,floor,on)
                    item["events"].append(dict(type="pressed" if on else "released",floor=floor,
                        physics_index=ticks,sim_time=ticks/120,travel_m=float(travel[eid,floor-24]),force_n=float(self.forces[eid,floor-24])))
            presses = [event["floor"] for event in item["events"] if event["type"]=="pressed"]
            reason = ("unexpected_collision" if item["collisions"] else "target_pressed" if presses==[item["floor"]]
                      else "wrong_button_pressed" if presses else
                      "time_limit" if ticks >= round(self.config["max_seconds"] * 120) else None)
            success = reason=="target_pressed"
            info = dict(termination=reason,success=success,task_success=presses==[item["floor"]],pressed_floors=presses,
                physics_index=ticks,sim_seconds=ticks/120,events=deepcopy(item["events"]),
                unexpected_collisions=deepcopy(item["collisions"]),collision_count=len(item["collisions"]),
                min_target_tip_distance_m=item["min_target_distance"],max_joint_tracking_error_rad=item["max_tracking_error"],
                max_target_contact_force_n=item["max_target_force"],max_target_travel_m=item["max_target_travel"],
                final_partial_interval=bool(ticks%4),motion_smoothing=dict(window=self.config["smoothing_window"]))
            output[eid] = dict(reward=float(success),success=success,terminated=reason not in (None,"time_limit"),
                               truncated=reason=="time_limit",reason=reason or "running",info=info)
        return output

    def observation_fields(self, env_ids):
        result = {}
        for eid in env_ids:
            floor = self.engine.backend.reset_specs[eid]["floor"]
            features = np.zeros(12,np.float32)
            features[floor-24] = 1.
            result[eid] = dict(task=f"Press {floor} floor.",model_inputs={},task_features=features.tolist())
        return result

    def diagnostic_state(self, env_ids):
        positions,orientations = self.buttons.get_world_poses()
        velocity = self.buttons.get_velocities()
        return {eid:dict(button_positions=np.asarray(positions)[self.button_rows[eid]].tolist(),
            button_orientations=np.asarray(orientations)[self.button_rows[eid]].tolist(),
            button_velocities=np.asarray(velocity)[self.button_rows[eid]].tolist()) for eid in env_ids}


def create_backend(config):
    from pressb.runtime.parallel import SharedWorldBackend
    return SharedWorldBackend(ElevatorTask(config))
