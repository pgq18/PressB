"""Reuse the verified scene in independent environments for dataset collection.

Isaac/Omniverse imports are deliberately inside functions. Call ``create_envs``
after constructing World and before its first reset. Physics handles in the
returned objects become usable after the collector calls ``world.reset()``.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class DatasetEnv:
    env_id: int
    prefix: str
    offset: np.ndarray
    config: dict
    robot: Any
    robot_path: str
    link6_path: str
    base_position: np.ndarray
    buttons: dict[int, Any]
    button_info: dict[int, dict]
    light_inputs: dict[int, dict[str, Any]]
    wrist_camera_path: str
    global_camera_path: str
    tool_colliders: set[str]
    looks_path: str
    world_joint_anchors: list[dict]
    panel_source_offset_y_m: float = 0.0
    panel_offset_y_m: float = 0.0
    panel_source_offset_x_m: float = 0.0
    panel_offset_x_m: float = 0.0
    button_source_xforms: dict = field(default_factory=dict)
    fixed_rig_transforms: dict = field(default_factory=dict)


def _panel_shift(stage, prefix, source_offset, target_offset, source_x=0., target_x=0.):
    """Author an absolute delta over a referenced panel without editing its asset."""
    from pxr import Gf, Sdf, UsdGeom
    from .scene import set_back_wall_offset
    panel = UsdGeom.Xformable(stage.GetPrimAtPath(prefix + "/Panel"))
    name = "xformOp:translate:collectionPanelOffset"
    attribute = panel.GetPrim().GetAttribute(name)
    operation = (UsdGeom.XformOp(attribute) if attribute else
                 panel.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, "collectionPanelOffset"))
    operation.Set(Gf.Vec3d(target_x - source_x, target_offset - source_offset, 0.))
    panel.GetPrim().CreateAttribute("pressb:panelOffsetYM", Sdf.ValueTypeNames.Double).Set(target_offset)
    panel.GetPrim().CreateAttribute("pressb:panelOffsetXM", Sdf.ValueTypeNames.Double).Set(target_x)
    set_back_wall_offset(stage, prefix + "/Environment", target_x - source_x, "collectionPanelOffset")


def _world_matrix(stage, path):
    from pxr import Usd, UsdGeom
    return np.asarray(UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()), dtype=float)


def create_envs(world, snapshot: Path, num_envs: int, spacing: float = 6.0, cfg: dict | None = None):
    """Reference the snapshot's /World into env_i, translated by (0,i*spacing,0).

    Each reference has independent opinions for button materials and state.
    Exactly one global DomeLight stays active; each room retains its local lights.
    The source /physicsScene is outside /World and is therefore not referenced.
    Configuration defaults to the report.json alongside the snapshot. The
    collector owns contact subscriptions, reset/settling, rendering and saving.
    """
    from isaacsim.core.cloner import Cloner
    from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade
    from .scene import BUTTON_BORDER_SIDES, BUTTON_FEEDBACK_STYLE, BUTTON_OFF_COLOR

    snapshot = Path(snapshot).resolve()
    if not snapshot.is_file():
        raise FileNotFoundError(snapshot)
    if not isinstance(num_envs, int) or num_envs <= 0:
        raise ValueError("num_envs must be a positive integer")
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("Environment spacing must be positive and finite")
    if cfg is None:
        cfg = json.loads((snapshot.parent / "report.json").read_text())["config"]
    cfg = deepcopy(cfg)
    panel_offset = float(cfg.get("panel_offset_y_m", 0.0))
    panel_offset_x = float(cfg.get("panel_offset_x_m", 0.0))
    if not np.isfinite([panel_offset, panel_offset_x]).all():
        raise ValueError("panel offsets must be finite")
    for name in ("table_height", "button_face_x", "button_bottom_z", "button_pitch_z", "button_column_y"):
        if name not in cfg or not np.isfinite(cfg[name]):
            raise ValueError(f"Missing or nonfinite scene configuration: {name}")

    stage = world.stage
    parent = "/World/envs"
    if stage.GetPrimAtPath(parent):
        raise ValueError("/World/envs already exists; create_envs requires a fresh World")
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Scope.Define(stage, parent)
    envs = []
    shared_dome_path = None
    for index in range(num_envs):
        prefix = f"{parent}/env_{index}"
        offset = np.array([0., index * float(spacing), 0.])
        root = UsdGeom.Xform.Define(stage, prefix)
        if not root.GetPrim().GetReferences().AddReference(str(snapshot), "/World"):
            raise RuntimeError(f"Unable to reference scene snapshot: {snapshot}")
        root.ClearXformOpOrder()
        root.SetResetXformStack(False)
        root.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*offset))

        robot_path = prefix + "/Piper"
        link6_path = robot_path + "/link6"
        wrist_path = link6_path + "/WristRealSense/CameraLink/ColorCamera"
        global_path = prefix + "/GlobalCamera/CameraLink/ColorCamera"
        for path in (robot_path, link6_path, prefix + "/Panel", prefix + "/GlobalCamera"):
            if not stage.GetPrimAtPath(path):
                raise ValueError(f"Snapshot is missing required prim: {path}")
        for path in (wrist_path, global_path):
            if not stage.GetPrimAtPath(path).IsA(UsdGeom.Camera):
                raise ValueError(f"Snapshot has no camera at {path}")
        source_attribute = stage.GetPrimAtPath(prefix + "/Panel").GetAttribute("pressb:panelOffsetYM")
        source_panel_offset = float(source_attribute.Get()) if source_attribute else 0.0
        source_x_attribute = stage.GetPrimAtPath(prefix + "/Panel").GetAttribute("pressb:panelOffsetXM")
        source_panel_offset_x = float(source_x_attribute.Get()) if source_x_attribute else 0.0
        if not np.isfinite([source_panel_offset, source_panel_offset_x]).all():
            raise ValueError("Snapshot has a nonfinite panel offset")
        _panel_shift(stage, prefix, source_panel_offset, panel_offset, source_panel_offset_x, panel_offset_x)
        # The single-scene fixed rig explicitly cancels ancestors. Its authored
        # transforms already describe the environment-local pose for cloning.
        UsdGeom.Xformable(stage.GetPrimAtPath(prefix + "/GlobalCamera")).SetResetXformStack(False)

        # A DomeLight is global even when it is nested under an environment.
        # RTX can still select an active zero-intensity Dome as the environment
        # light, suppressing the intended shared illumination. Deactivate the
        # duplicate prims instead, and gather them before changing composition
        # so USD traversal is not invalidated while visiting their descendants.
        domes = [prim for prim in Usd.PrimRange(root.GetPrim()) if prim.IsA(UsdLux.DomeLight)]
        if len(domes) != 1:
            raise ValueError(f"Expected one source DomeLight in {prefix}, found {len(domes)}")
        if index == 0:
            shared_dome_path = domes[0].GetPath()
        else:
            domes[0].SetActive(False)

        anchors = []
        for prim in Usd.PrimRange(root.GetPrim()):
            if not prim.IsA(UsdPhysics.Joint):
                continue
            joint = UsdPhysics.Joint(prim)
            sides = ((joint.GetBody0Rel(), joint.GetLocalPos0Attr(), 0),
                     (joint.GetBody1Rel(), joint.GetLocalPos1Attr(), 1))
            if not any(relation.GetTargets() for relation, _, _ in sides):
                raise ValueError(f"Joint has no body on either side: {prim.GetPath()}")
            for relation, attribute, side in sides:
                if relation.GetTargets():
                    for target in relation.GetTargets():
                        if not str(target).startswith(prefix + "/"):
                            raise ValueError(f"Joint escaped its environment: {prim.GetPath()} -> {target}")
                    continue
                local_position = np.asarray(attribute.Get(), dtype=float)
                if local_position.shape != (3,) or not np.isfinite(local_position).all():
                    raise ValueError(f"Invalid world joint anchor: {prim.GetPath()}")
                shifted = local_position + offset
                is_panel_anchor = str(prim.GetPath()).startswith(prefix + "/Panel/")
                if is_panel_anchor:
                    shifted[0] += panel_offset_x - source_panel_offset_x
                    shifted[1] += panel_offset - source_panel_offset
                attribute.Set(Gf.Vec3f(*shifted))
                anchors.append({"joint_path": str(prim.GetPath()), "world_side": side,
                                "source_position": local_position.tolist(), "world_position": shifted.tolist(),
                                "is_panel_anchor": is_panel_anchor})
        if len(anchors) != 13:
            raise ValueError(f"Expected 12 button springs and one robot root world anchor, found {len(anchors)}")

        light_inputs, info, buttons, source_xforms = {}, {}, {}, {}
        for floor in range(24, 36):
            row = (floor - 24) % 6
            column = cfg["button_column_y"] * (1 if floor < 30 else -1) + panel_offset
            z = cfg["button_bottom_z"] + row * cfg["button_pitch_z"]
            rest = np.array([cfg["button_face_x"] + panel_offset_x + .003, column, z]) + offset
            body_path = prefix + f"/Panel/Floor{floor}/Cap"
            body = stage.GetPrimAtPath(body_path)
            if not body or not body.HasAPI(UsdPhysics.RigidBodyAPI):
                raise ValueError(f"Missing rigid button: {body_path}")
            composed = np.asarray(UsdGeom.Xformable(body).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()).ExtractTranslation())
            if not np.allclose(composed, rest, rtol=0, atol=1e-4):
                raise ValueError(f"Button {floor} is not at its configured rest pose: {composed} vs {rest}")
            shader_path = prefix + f"/Looks/Button{floor}/Shader"
            shader = UsdShade.Shader.Get(stage, shader_path)
            if not shader:
                raise ValueError(f"Missing independent button shader: {shader_path}")
            inputs = {"diffuse": shader.GetInput("diffuseColor"),
                      "emissive": shader.GetInput("emissiveColor"),
                      "illuminated": body.GetAttribute("pressb:illuminated")}
            if not all(inputs.values()):
                raise ValueError(f"Incomplete button feedback attributes: {shader_path}")
            # Author local opinions even when the referenced snapshot was lit.
            inputs["diffuse"].Set(Gf.Vec3f(*BUTTON_OFF_COLOR))
            inputs["emissive"].Set(Gf.Vec3f(0., 0., 0.))
            inputs["illuminated"].Set(False)
            light_inputs[floor] = inputs
            material_path = prefix + f"/Looks/Button{floor}"
            feedback_path = body_path + "/FeedbackBorder"
            has_border = bool(stage.GetPrimAtPath(feedback_path))
            feedback_style = body.GetAttribute("pressb:feedbackStyle").Get()
            if has_border or feedback_style == BUTTON_FEEDBACK_STYLE:
                if not has_border or feedback_style != BUTTON_FEEDBACK_STYLE:
                    raise ValueError(f"Incomplete edge feedback geometry or style: {body_path}")
                feedback_geometry_paths = [f"{feedback_path}/{side}" for side in BUTTON_BORDER_SIDES]
                for path in feedback_geometry_paths:
                    border = stage.GetPrimAtPath(path)
                    if not border.IsA(UsdGeom.Cube) or border.HasAPI(UsdPhysics.CollisionAPI):
                        raise ValueError(f"Expected noncolliding button feedback border: {path}")
                    bound, _ = UsdShade.MaterialBindingAPI(border).ComputeBoundMaterial()
                    if not bound or str(bound.GetPath()) != material_path:
                        raise ValueError(f"Button border material was not remapped into {prefix}: {path}")
                face_material_path = prefix + "/Looks/ButtonFace"
            else:
                # Explicit legacy snapshots keep their original full-face
                # appearance. Never silently retrofit an archived dataset scene.
                feedback_style = "full_face"
                feedback_path = body_path
                feedback_geometry_paths = [body_path + "/Collision", body_path + "/Indicator"]
                face_material_path = material_path
            bound, _ = UsdShade.MaterialBindingAPI(body.GetChild("Collision")).ComputeBoundMaterial()
            if not bound or str(bound.GetPath()) != face_material_path:
                raise ValueError(f"Button face material was not remapped into {prefix}: {floor}")
            info[floor] = {"body_path": body_path, "collision_path": body_path + "/Collision",
                           "joint_path": prefix + f"/Panel/Floor{floor}/SpringJoint",
                           "material_path": material_path, "shader_path": shader_path,
                           "feedback_style": feedback_style, "feedback_path": feedback_path,
                           "feedback_geometry_paths": feedback_geometry_paths,
                           "face_material_path": face_material_path,
                           "rest_x": float(rest[0]), "body_center": rest.tolist(),
                           "center": (rest - [.003, 0., 0.]).tolist(), "floor": floor, "row": row,
                           "column": "left" if floor < 30 else "right"}
            if has_border:
                info[floor]["border_width_m"] = float(body.GetAttribute("pressb:borderWidth").Get())
            buttons[floor] = world.scene.add(SingleRigidPrim(
                prim_path=body_path, name=f"env_{index}_button_{floor}"))
            source_xforms[floor] = [(str(op.GetName()), op.Get())
                                    for op in UsdGeom.Xformable(body).GetOrderedXformOps()]
        robot = world.scene.add(SingleArticulation(prim_path=robot_path, name=f"env_{index}_piper"))
        base = np.array([cfg.get("robot_base_x", 0.), cfg.get("robot_base_y", 0.), cfg["table_height"]]) + offset
        envs.append(DatasetEnv(
            env_id=index, prefix=prefix, offset=offset, config=deepcopy(cfg), robot=robot,
            robot_path=robot_path, link6_path=link6_path, base_position=base,
            buttons=buttons, button_info=info, light_inputs=light_inputs,
            wrist_camera_path=wrist_path, global_camera_path=global_path,
            tool_colliders={link6_path + "/PressStylus", link6_path + "/PressTip"},
            looks_path=prefix + "/Looks", world_joint_anchors=anchors,
            panel_source_offset_y_m=source_panel_offset, panel_offset_y_m=panel_offset,
            panel_source_offset_x_m=source_panel_offset_x, panel_offset_x_m=panel_offset_x,
            button_source_xforms=source_xforms,
            fixed_rig_transforms={path: _world_matrix(stage, path) for path in
                                  (robot_path, prefix + "/GlobalCamera", global_path)}))

    active_domes = [prim.GetPath() for prim in stage.Traverse() if prim.IsA(UsdLux.DomeLight)]
    if active_domes != [shared_dome_path]:
        raise ValueError(f"Expected only the shared DomeLight {shared_dome_path}, found {active_domes}")

    physics_path = world.get_physics_context().prim_path
    Cloner(stage=stage).filter_collisions(physics_path, "/World/DatasetCollisionGroups",
                                          [env.prefix for env in envs], [])
    return envs


def set_panel_offset(world, env: DatasetEnv, offset_y_m: float, offset_x_m: float = 0.):
    """Move one complete panel to an absolute episode offset between physics ticks.

    Call after ``world.reset()`` initialized the rigid handles and after the
    previous episode returned the arm home. The timeline may remain playing;
    this function never steps, pauses or resets the shared World. The collector
    must settle physics and then call ``validate_panel_layout`` before capture.
    World-side joint anchors need explicit updates: they do not inherit the
    panel transform. Defaults and USD poses are updated alongside live physics
    handles so a later World reset cannot restore the old button positions.
    """
    from pxr import Gf, UsdPhysics
    offset_y_m = float(offset_y_m)
    offset_x_m = float(offset_x_m)
    if not np.isfinite([offset_x_m, offset_y_m]).all():
        raise ValueError("panel offsets must be finite")
    stage = world.stage
    _panel_shift(stage, env.prefix, env.panel_source_offset_y_m, offset_y_m,
                 env.panel_source_offset_x_m, offset_x_m)
    for anchor in env.world_joint_anchors:
        if not anchor["is_panel_anchor"]:
            continue
        target = np.asarray(anchor["source_position"], dtype=float) + env.offset
        target[0] += offset_x_m - env.panel_source_offset_x_m
        target[1] += offset_y_m - env.panel_source_offset_y_m
        joint = UsdPhysics.Joint.Get(stage, anchor["joint_path"])
        attribute = joint.GetLocalPos0Attr() if anchor["world_side"] == 0 else joint.GetLocalPos1Attr()
        attribute.Set(Gf.Vec3f(*target))
        anchor["world_position"] = target.tolist()
    for floor, body in env.buttons.items():
        info = env.button_info[floor]
        center = np.array([env.config["button_face_x"] + offset_x_m,
                           (1 if floor < 30 else -1) * env.config["button_column_y"] + offset_y_m,
                           env.config["button_bottom_z"] + info["row"] * env.config["button_pitch_z"]]) + env.offset
        rest = center + [.003, 0., 0.]
        # PhysX's live tensor setter does not immediately author USD transforms.
        # Restore the source local pose too, preventing stale render geometry or
        # displacement accumulated through successive parent moves.
        prim = stage.GetPrimAtPath(info["body_path"])
        for name, value in env.button_source_xforms[floor]:
            prim.GetAttribute(name).Set(value)
        orientation = np.array([1., 0., 0., 0.])
        body.set_world_pose(position=rest, orientation=orientation)
        body.set_linear_velocity(np.zeros(3))
        body.set_angular_velocity(np.zeros(3))
        body.set_default_state(position=rest, orientation=orientation,
                               linear_velocity=np.zeros(3), angular_velocity=np.zeros(3))
        info.update(center=center.tolist(), body_center=rest.tolist(), rest_x=float(rest[0]))
        set_light(env, floor, False)
    env.config["panel_offset_y_m"] = offset_y_m
    env.config["panel_offset_x_m"] = offset_x_m
    env.panel_offset_y_m = offset_y_m
    env.panel_offset_x_m = offset_x_m
    return {"panel_offset_y_m": offset_y_m, "panel_offset_x_m": offset_x_m}


def validate_panel_layout(world, env: DatasetEnv, rest_tolerance_m: float = .0002):
    """Verify the settled physical panel and its fixed-camera frustum coverage.

    This checks the complete rendered panel bounding box, including the frame,
    housing and labels. Frustum containment alone does not prove no occlusion.
    Return JSON-compatible per-episode evidence for the raw dataset validator.
    """
    from itertools import product
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    if not np.isfinite(rest_tolerance_m) or rest_tolerance_m <= 0:
        raise ValueError("rest_tolerance_m must be finite and positive")
    stage = world.stage
    fixed = all(np.allclose(_world_matrix(stage, path), expected, rtol=0., atol=1e-7)
                for path, expected in env.fixed_rig_transforms.items())
    if not fixed:
        raise RuntimeError("Panel repositioning changed the robot base or fixed camera rig")
    measured, rest_error, anchor_error = {}, 0., 0.
    for floor, body in env.buttons.items():
        position, orientation = body.get_world_pose()
        position, orientation = np.asarray(position), np.asarray(orientation)
        if not np.isfinite(np.r_[position, orientation]).all():
            raise RuntimeError(f"Nonfinite settled button pose: {floor}")
        measured[str(floor)] = position.tolist()
        rest_error = max(rest_error, float(np.linalg.norm(position - env.button_info[floor]["body_center"])))
    for anchor in env.world_joint_anchors:
        joint = UsdPhysics.Joint.Get(stage, anchor["joint_path"])
        value = (joint.GetLocalPos0Attr() if anchor["world_side"] == 0 else joint.GetLocalPos1Attr()).Get()
        error = float(np.linalg.norm(np.asarray(value) - np.asarray(anchor["world_position"])))
        if not np.isfinite(error):
            raise RuntimeError("Nonfinite spring world anchor")
        anchor_error = max(anchor_error, error)
    if rest_error > rest_tolerance_m or anchor_error > rest_tolerance_m:
        raise RuntimeError(f"Panel did not settle at its new layout: rest={rest_error}, anchor={anchor_error}")
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(
        stage.GetPrimAtPath(env.prefix + "/Panel")).ComputeAlignedRange()
    minimum, maximum = np.asarray(bounds.GetMin()), np.asarray(bounds.GetMax())
    corners = np.asarray(list(product(*zip(minimum, maximum))))
    camera = UsdGeom.Camera.Get(stage, env.global_camera_path)
    camera_matrix = UsdGeom.Xformable(camera.GetPrim()).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    eye = np.asarray(env.config["global_camera_eye"], dtype=float) + env.offset
    target = np.asarray(env.config["global_camera_target"], dtype=float) + env.offset
    expected_camera = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0., 0., 1.)).GetInverse()
    if not np.allclose(np.asarray(camera_matrix), np.asarray(expected_camera), rtol=0., atol=1e-6):
        raise RuntimeError("Snapshot fixed-camera optical pose disagrees with scene configuration")
    inverse = camera_matrix.GetInverse()
    optical = np.asarray([inverse.Transform(Gf.Vec3d(*point)) for point in corners])
    depth = -optical[:, 2]
    width, height = map(int, env.config.get("global_camera_resolution", [640, 480]))
    focal, horizontal, vertical = (float(attr.Get()) for attr in
                                   (camera.GetFocalLengthAttr(), camera.GetHorizontalApertureAttr(),
                                    camera.GetVerticalApertureAttr()))
    if not np.isfinite(np.r_[corners.ravel(), optical.ravel(), focal, horizontal, vertical]).all() or min(focal, horizontal, vertical) <= 0:
        raise RuntimeError("Invalid panel bounding box or fixed-camera intrinsics")
    clip_near, clip_far = camera.GetClippingRangeAttr().Get()
    margin = float(env.config.get("panel_randomization", {}).get("camera_margin_px", 24.))
    if not np.isfinite(margin) or margin < 0 or 2 * margin >= min(width, height):
        raise ValueError("Invalid camera_margin_px")
    if np.any(depth <= clip_near) or np.any(depth >= clip_far):
        raise RuntimeError("Panel bounding box is outside the fixed-camera depth range")
    pixels = np.column_stack((width / 2 + width * focal / horizontal * optical[:, 0] / depth,
                              height / 2 - height * focal / vertical * optical[:, 1] / depth))
    inside = bool(np.all(pixels[:, 0] >= margin) and np.all(pixels[:, 0] <= width - 1 - margin)
                  and np.all(pixels[:, 1] >= margin) and np.all(pixels[:, 1] <= height - 1 - margin))
    if not inside:
        raise RuntimeError(f"Full panel does not fit fixed camera with {margin}px margin: {pixels.tolist()}")
    wall_bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(
        stage.GetPrimAtPath(env.prefix + "/Environment/Wall")).ComputeAlignedRange()
    wall_min, wall_max = np.asarray(wall_bounds.GetMin()), np.asarray(wall_bounds.GetMax())
    expected_wall_x = env.config.get("wall_x", env.config["button_face_x"] + .06) + env.panel_offset_x_m + env.offset[0]
    wall_error = abs(float(wall_min[0] - expected_wall_x))
    if not np.isfinite(wall_error) or wall_error > 1e-6:
        raise RuntimeError("Back wall is not aligned with the moved panel")
    return {"panel_offset_y_m": env.panel_offset_y_m, "panel_offset_x_m": env.panel_offset_x_m,
            "fixed_rig_unchanged": fixed,
            "button_rest_positions_world_m": measured, "max_button_rest_error_m": rest_error,
            "max_spring_anchor_error_m": anchor_error, "rest_tolerance_m": rest_tolerance_m,
            "panel_world_bounds_m": [minimum.tolist(), maximum.tolist()],
            "back_wall_world_bounds_m": [wall_min.tolist(), wall_max.tolist()],
            "back_wall_alignment_error_m": wall_error,
            "global_camera": {"path": env.global_camera_path, "resolution": [width, height],
                              "margin_px": margin, "all_inside": inside, "corners_px": pixels.tolist(),
                              "corner_depths_m": depth.tolist(), "coverage_kind": "frustum_only"}}


def set_light(env: DatasetEnv, floor: int, on: bool):
    """Update one environment's measured button feedback without cross-talk."""
    from pxr import Gf
    from .scene import BUTTON_OFF_COLOR, BUTTON_ON_COLOR, BUTTON_ON_EMISSION
    inputs = env.light_inputs[floor]
    inputs["diffuse"].Set(Gf.Vec3f(*(BUTTON_ON_COLOR if on else BUTTON_OFF_COLOR)))
    inputs["emissive"].Set(Gf.Vec3f(*(BUTTON_ON_EMISSION if on else (0., 0., 0.))))
    inputs["illuminated"].Set(bool(on))


def split_tiled_rgb(buffer, num_cameras: int, resolution=(640, 480)) -> list[np.ndarray]:
    """Copy RGB tiles in the original camera-list order (left-to-right, then down).

    Matches IsaacLab 2.0.2 ``reshape_tiled_image`` and installed Kit 106.5:
    columns=ceil(sqrt(N)), rows=ceil(N/columns). Camera i is at
    (row=i//columns, column=i%columns); no vertical flip is applied.
    Copies prevent the following render from mutating frames queued for writing.
    """
    if not isinstance(num_cameras, int) or num_cameras <= 0:
        raise ValueError("num_cameras must be a positive integer")
    width, height = (int(value) for value in resolution)
    if width <= 0 or height <= 0:
        raise ValueError("Tile resolution must be positive")
    if isinstance(buffer, dict):
        buffer = buffer["data"]
    data = np.asarray(buffer)
    columns = math.ceil(math.sqrt(num_cameras))
    rows = math.ceil(num_cameras / columns)
    if (data.ndim != 3 or data.shape[:2] != (rows * height, columns * width)
            or data.shape[2] not in (3, 4) or data.dtype != np.uint8):
        raise ValueError(f"Invalid tiled RGB buffer {data.shape}/{data.dtype}; expected "
                         f"({rows * height}, {columns * width}, 3 or 4) uint8")
    return [np.array(data[(i // columns) * height:(i // columns + 1) * height,
                          (i % columns) * width:(i % columns + 1) * width, :3], copy=True, order="C")
            for i in range(num_cameras)]


def build_tiled_rgb(camera_paths, resolution=(640, 480)):
    """Return (render_product, CPU_RGB_annotator, split_fn) for existing cameras.

    No Core Camera wrappers or additional render products are created. Caller
    renders once, reads ``annotator.get_data()``, then invokes ``split_fn``.
    ``camera_paths`` order is preserved by the render product relationship.
    """
    import omni.replicator.core as rep
    import omni.usd
    from pxr import UsdGeom

    paths = [str(path) for path in camera_paths]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Expected a nonempty list of distinct camera paths")
    dimensions = np.asarray(resolution)
    if (dimensions.shape != (2,) or not np.isfinite(dimensions).all()
            or np.any(dimensions <= 0) or np.any(dimensions != np.floor(dimensions))):
        raise ValueError("Camera resolution must contain two positive integers")
    resolution = tuple(int(value) for value in dimensions)
    stage = omni.usd.get_context().get_stage()
    for path in paths:
        if not stage.GetPrimAtPath(path).IsA(UsdGeom.Camera):
            raise ValueError(f"Missing Camera prim: {path}")
    render_product = rep.create.render_product_tiled(cameras=paths, tile_resolution=resolution, force_new=True)
    authored = [str(path) for path in stage.GetPrimAtPath(render_product.path).GetRelationship("camera").GetTargets()]
    if authored != paths:
        raise RuntimeError("Tiled render product changed the requested camera order")
    annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu", do_array_copy=True)
    annotator.attach([render_product.path])
    return render_product, annotator, partial(split_tiled_rgb, num_cameras=len(paths), resolution=resolution)
