"""Task-independent laboratory workbench, mounting hardware and camera tripod.

PAP and table-button scenes share this builder without importing each other.
The legacy material palette (including unused prop swatches) is retained so
existing USD paths, material overrides and serialized scene metadata stay exact.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

from .scene import (
    _positive, _vector, _color, _material_settings, _material, _bind, _cube,
    _cylinder_between,
)


def _materials(stage, root, overrides=None, material_overrides=None, effective=None):
    colors = {
        "Mat": ((0.008, 0.135, 0.100), 0., 0.82),
        "Aluminium": ((0.50, 0.54, 0.55), 0.65, 0.34),
        "WhiteTrim": ((0.77, 0.79, 0.78), 0.15, 0.48),
        "DarkRubber": ((0.012, 0.015, 0.017), 0., 0.76),
        "MountBlack": ((0.009, 0.011, 0.013), 0.25, 0.42),
        "ClampOrange": ((0.95, 0.16, 0.008), 0.05, 0.35),
        "Cardboard": ((0.43, 0.265, 0.117), 0., 0.95),
        "CardboardCut": ((0.30, 0.18, 0.083), 0., 0.96),
        "CardboardDark": ((0.13, 0.069, 0.026), 0., 0.96),
        "RedCube": ((0.57, 0.018, 0.025), 0., 0.63),
        "Tape": ((0.44, 0.58, 0.11), 0., 0.78),
        "Wall": ((0.77, 0.78, 0.76), 0., 0.89),
        "Beam": ((0.23, 0.27, 0.28), 0.32, 0.50),
        "Floor": ((0.17, 0.18, 0.19), 0., 0.99),
    }
    material_overrides = {} if material_overrides is None else material_overrides
    if not isinstance(material_overrides, Mapping):
        raise ValueError("materials must be a mapping of material names to properties")
    if "CardboardOuter" in material_overrides:
        # Optional visual skin: do not author even an unused material in legacy
        # scenes. Its unspecified values inherit the effective inner cardboard.
        colors["CardboardOuter"] = colors["Cardboard"]
    if set(material_overrides) - set(colors):
        raise ValueError(f"Unknown material names: {sorted(set(material_overrides) - set(colors))}")
    result, resolved = {}, {}
    for name, (color, metallic, roughness) in colors.items():
        color = (overrides or {}).get(name, color)
        defaults = resolved["Cardboard"] if name == "CardboardOuter" else dict(color=color, metallic=metallic, roughness=roughness)
        settings = _material_settings(defaults,
                                      material_overrides.get(name, {}), f"materials.{name}")
        resolved[name] = settings
        contact = name in {"Mat", "Cardboard", "RedCube", "Aluminium", "Floor"}
        result[name] = _material(stage, f"{root}/Looks/{name}", settings["color"],
                                 metallic=settings["metallic"], roughness=settings["roughness"],
                                 specular=settings["specular"] if settings["specular_authored"] else None,
                                 static_friction=0.8 if contact else None,
                                 dynamic_friction=0.65 if contact else None)
        if effective is not None:
            effective[name] = {**settings, "path": str(result[name].GetPath()),
                               "color_space": "linear RGB"}
    return result


def _lighting(stage, root, cfg, base_xy, height, table_x, table_y):
    """Author legacy lights or base-relative area lights with explicit aim."""
    lighting = cfg.get("lighting", {})
    if not isinstance(lighting, Mapping):
        raise ValueError("lighting must be a mapping")
    if set(lighting) - {"ambient", "area_lights"}:
        raise ValueError("lighting supports ambient and area_lights")

    def intensity(value, name):
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
        return value

    ambient = lighting.get("ambient", {})
    if not isinstance(ambient, Mapping) or set(ambient) - {"intensity", "color"}:
        raise ValueError("lighting.ambient supports intensity and color")
    ambient = dict(path=root + "/Lighting/Ambient",
                   intensity=intensity(ambient.get("intensity", cfg.get("ambient_intensity", 350.)),
                                       "lighting.ambient.intensity"),
                   color=_color(ambient.get("color", [1., .985, .96]), "lighting.ambient.color"))
    lights = []
    if "area_lights" in lighting:
        if not isinstance(lighting["area_lights"], (list, tuple)):
            raise ValueError("lighting.area_lights must be a list")
        names = {"Ambient"}
        allowed = {"name", "position_base_m", "target_base_m", "width", "height", "intensity", "color"}
        for i, config in enumerate(lighting["area_lights"]):
            label = f"lighting.area_lights[{i}]"
            if not isinstance(config, Mapping) or set(config) - allowed:
                raise ValueError(f"{label} supports {sorted(allowed)}")
            name = config.get("name", f"Area{i}")
            if not isinstance(name, str) or not Sdf.Path.IsValidIdentifier(name) or name in names:
                raise ValueError(f"{label}.name must be a unique USD identifier other than Ambient")
            names.add(name)
            position = _vector(config.get("position_base_m", []), 3, f"{label}.position_base_m")
            target = _vector(config.get("target_base_m", []), 3, f"{label}.target_base_m")
            origin = [*base_xy, height]
            position_world = [v + o for v, o in zip(position, origin)]
            target_world = [v + o for v, o in zip(target, origin)]
            direction = Gf.Vec3d(*target_world) - Gf.Vec3d(*position_world)
            if direction.GetLength() < 1e-8:
                raise ValueError(f"{label} position and target must differ")
            direction.Normalize()
            # Vertical downlights need a horizontal up-vector; otherwise retain
            # world Z as the roll convention for repeatable rectangular shadows.
            up = Gf.Vec3d(0, 1, 0) if abs(direction[2]) > .999 else Gf.Vec3d(0, 0, 1)
            transform = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*position_world), Gf.Vec3d(*target_world), up).GetInverse()
            lights.append(dict(name=name, path=root + "/Lighting/" + name,
                               position_base_m=position, target_base_m=target,
                               position_world_m=position_world, target_world_m=target_world,
                               optical_forward_world=list(direction), transform=transform,
                               width=_positive(config.get("width", 0), f"{label}.width"),
                               height=_positive(config.get("height", 0), f"{label}.height"),
                               intensity=intensity(config.get("intensity", 1600.), f"{label}.intensity"),
                               color=_color(config.get("color", [1., 1., 1.]), f"{label}.color")))
    else:
        # Preserve exactly the previous authored translation and default white
        # color, including not adding a transform or redundant color attribute.
        position = [table_x, table_y - .15, height + 1.4]
        lights = [dict(name="Ceiling", path=root + "/Lighting/Ceiling",
                       position_world_m=position, target_world_m=[table_x, table_y - .15, height],
                       position_base_m=[table_x - base_xy[0], -.15, 1.4],
                       target_base_m=[table_x - base_xy[0], -.15, 0.],
                       optical_forward_world=[0., 0., -1.], width=1.2, height=.7,
                       intensity=intensity(cfg.get("key_intensity", 1600.), "key_intensity"),
                       color=[1., 1., 1.])]
    enabled = bool(cfg.get("lights", True))
    if enabled:
        dome = UsdLux.DomeLight.Define(stage, ambient["path"])
        dome.CreateIntensityAttr(ambient["intensity"])
        dome.CreateColorAttr(Gf.Vec3f(*ambient["color"]))
        for info in lights:
            light = UsdLux.RectLight.Define(stage, info["path"])
            if "transform" in info:
                light.AddTransformOp().Set(info["transform"])
                light.CreateColorAttr(Gf.Vec3f(*info["color"]))
            else:
                light.AddTranslateOp().Set(Gf.Vec3d(*info["position_world_m"]))
            light.CreateWidthAttr(info["width"])
            light.CreateHeightAttr(info["height"])
            light.CreateIntensityAttr(info["intensity"])
    return dict(enabled=enabled, ambient=ambient if enabled else None,
                area_lights=[{k: v for k, v in info.items() if k != "transform"} for info in lights] if enabled else [],
                coordinate_frame="position_base_m and target_base_m relative to robot base at tabletop",
                color_space="linear RGB")


def _mount_hardware(stage, root, base_xy, height, table_x_min, mats):
    """Photographic mounting plate and clamps, with no physics APIs.

    The plate intentionally lies almost flush with the tabletop: the robot's
    measured base frame is unchanged. The slightly raised screw heads, clamps,
    and threaded rods are visible context, not new robot collision geometry.
    """
    path = root + "/MountHardware"
    UsdGeom.Xform.Define(stage, path)
    _cube(stage, path + "/BasePlate", (base_xy[0] - .04, base_xy[1], height - .002),
          (.22, .24, .006), mats["MountBlack"])
    for i, (x, y) in enumerate([(-.13, -.104), (-.13, .104), (.052, -.104), (.052, .104)]):
        x, y = base_xy[0] + x, base_xy[1] + y
        _cylinder_between(stage, f"{path}/Bolt{i}/Head", (x, y, height + .001),
                          (x, y, height + .004), .005, mats["Aluminium"])
        _cylinder_between(stage, f"{path}/Bolt{i}/Socket", (x, y, height + .004),
                          (x, y, height + .0042), .0022, mats["DarkRubber"])
    front_x = base_xy[0] + table_x_min
    # Cast-orange C frames face into the table (+X). Segment boxes leave the
    # centre visibly open; a steel screw rises from the lower jaw to the desk.
    for index, sign in enumerate([-1, 1]):
        clamp_path = f"{path}/Clamp{index}"
        y = base_xy[1] + sign * .10
        for name, dx, dz, sx, sz in [
            ("TopJaw", -.010, .018, .105, .024),
            ("OuterSpine", -.071, -.082, .025, .176),
            ("LowerJaw", -.015, -.179, .122, .026),
        ]:
            _cube(stage, f"{clamp_path}/{name}", (front_x + dx, y, height + dz),
                  (sx, .025, sz), mats["ClampOrange"])
        for name, a, b in [
            ("UpperCurve", (-.062, .018), (-.078, -.013)),
            ("LowerCurve", (-.078, -.154), (-.065, -.178)),
        ]:
            _cylinder_between(stage, f"{clamp_path}/{name}",
                              (front_x + a[0], y, height + a[1]),
                              (front_x + b[0], y, height + b[1]), .013, mats["ClampOrange"])
        screw_x = front_x + .025
        _cylinder_between(stage, f"{clamp_path}/ThreadedRod", (screw_x, y, height - .245),
                          (screw_x, y, height - .049), .004, mats["Aluminium"])
        # A narrow alternating sleeve pattern suggests the screw threads while
        # avoiding any contact or solver cost from decorative fine geometry.
        for ring in range(20):
            z = height - .16 + ring * .005
            _cylinder_between(stage, f"{clamp_path}/Thread{ring}", (screw_x, y, z),
                              (screw_x, y, z + .001), .0046, mats["MountBlack"])
        _cylinder_between(stage, f"{clamp_path}/PressurePad", (screw_x, y, height - .052),
                          (screw_x, y, height - .042), .012, mats["MountBlack"])
        _cylinder_between(stage, f"{clamp_path}/Handle", (screw_x, y - .035, height - .226),
                          (screw_x, y + .035, height - .226), .004, mats["Aluminium"])
    return {"root_path": path, "visual_only": True,
            "plate_center_base": [-.04, 0., -.002], "plate_size": [.22, .24, .006]}


def _validate_workbench(stage, cfg, *, default_root="/World/PickPlace"):
    """Resolve the shared geometry before any USD prim is authored."""
    root = str(cfg.get("root_path", default_root)).rstrip("/")
    if not root.startswith("/") or root == "/" or not Sdf.Path.IsValidPathString(root):
        raise ValueError("root_path must be a valid absolute USD prim path")
    if stage.GetPrimAtPath(root):
        raise ValueError(f"Scene root already exists: {root}")
    table = cfg.get("table", {})
    if not isinstance(table, Mapping):
        raise ValueError("table must be a mapping")
    height = _positive(cfg.get("table_height", table.get("height", .76)), "table height")
    depth = _positive(cfg.get("table_depth", table.get("depth", 1.0)), "table depth")
    width = _positive(cfg.get("table_width", table.get("width", 1.6)), "table width")
    x_min = float(cfg.get("table_x_min", table.get("x_min", -.12)))
    base_xy = _vector(cfg.get("robot_base_xy", [0., 0.]), 2, "robot_base_xy")
    if not math.isfinite(x_min):
        raise ValueError("table_x_min must be finite")
    return dict(root=root, height=height, depth=depth, width=width,
                x_min=x_min, base_xy=base_xy)


def _prepare_workbench(stage, cfg, geometry):
    """Author the table and mount before task props, preserving prim order."""
    root, height, depth, width, x_min, base_xy = (
        geometry[name] for name in ("root", "height", "depth", "width", "x_min", "base_xy"))
    if height <= .06 or min(depth, width) <= .15:
        raise ValueError("Table is too small for its support geometry")

    UsdGeom.Xform.Define(stage, root)
    effective_materials = {}
    mats = _materials(stage, root, cfg.get("material_colors"), cfg.get("materials"), effective_materials)
    table_x = base_xy[0] + x_min + depth / 2
    table_y = base_xy[1]
    collision_paths = []

    def collider(name, center, size, material, **kwargs):
        shape = _cube(stage, f"{root}/{name}", center, size, material,
                      collision=True, **kwargs)
        collision_paths.append(str(shape.GetPath()))
        return shape

    # The green slab's top is the precise work plane. A separate white edge
    # sits below it so decorative surfaces cannot alter grasp heights.
    collider("Table/Top", (table_x, table_y, height - .018),
             (depth, width, .036), mats["Mat"])
    _cube(stage, root + "/Table/FrontTrim", (base_xy[0] + x_min - .003, table_y, height - .035),
          (.012, width + .012, .038), mats["WhiteTrim"])
    for sign, name in [(-1, "Right"), (1, "Left")]:
        _cube(stage, f"{root}/Table/{name}Trim", (table_x, table_y + sign * (width / 2 + .003), height - .035),
              (depth, .012, .038), mats["WhiteTrim"])
    leg_z = (height - .052) / 2
    for index, (dx, dy) in enumerate([(0.055, .055), (depth - .055, .055),
                                      (0.055, width - .055), (depth - .055, width - .055)]):
        pos = (base_xy[0] + x_min + dx, table_y - width / 2 + dy, leg_z)
        collider(f"Table/Leg{index}", pos, (.035, .035, 2 * leg_z), mats["Aluminium"])
        _cube(stage, f"{root}/Table/Foot{index}", (pos[0], pos[1], .012),
              (.044, .044, .024), mats["DarkRubber"])
    for sign in [-1, 1]:
        _cube(stage, f"{root}/Table/UnderRail{'Left' if sign > 0 else 'Right'}",
              (table_x, table_y + sign * (width / 2 - .055), height - .12),
              (depth - .11, .026, .028), mats["Aluminium"])

    mounting = (_mount_hardware(stage, root, base_xy, height, x_min, mats)
                if cfg.get("mount_hardware", True) else None)

    return dict(geometry, effective_materials=effective_materials, mats=mats,
                table_x=table_x, table_y=table_y, collision_paths=collision_paths,
                collider=collider, mounting=mounting)


def _finish_workbench(stage, cfg, workbench):
    """Author the background/lights after task props and return shared metadata."""
    root, height, depth, width, x_min, base_xy, table_x, table_y = (
        workbench[name] for name in
        ("root", "height", "depth", "width", "x_min", "base_xy", "table_x", "table_y"))
    mats, collision_paths, collider, mounting, effective_materials = (
        workbench[name] for name in
        ("mats", "collision_paths", "collider", "mounting", "effective_materials"))
    if cfg.get("background", True):
        # White laboratory partition beyond the rear table edge, grey framing,
        # outlet strip and carpet visible in the supplied room photographs.
        back_x = base_xy[0] + x_min + depth + .07
        _cube(stage, root + "/Room/BackWall", (back_x, table_y, 1.40),
              (.035, width + .6, 2.8), mats["Wall"])
        for i, z in enumerate([height + .08, height + .83, 2.4]):
            _cube(stage, f"{root}/Room/HorizontalBeam{i}", (back_x - .025, table_y, z),
                  (.035, width + .6, .055), mats["Beam"])
        for i, y in enumerate([table_y - width / 2 - .12, table_y + width / 2 + .12]):
            _cube(stage, f"{root}/Room/VerticalBeam{i}", (back_x - .03, y, 1.4),
                  (.05, .065, 2.8), mats["Aluminium"])
        for i in range(7):
            y = table_y + (i - 3) * .14
            _cube(stage, f"{root}/Room/Outlet{i}/Plate", (back_x - .05, y, height + .10),
                  (.009, .09, .09), mats["WhiteTrim"])
            for j, dy in enumerate([-.019, .019]):
                _cube(stage, f"{root}/Room/Outlet{i}/Slot{j}", (back_x - .055, y + dy, height + .105),
                      (.001, .004, .018), mats["DarkRubber"])
            _cube(stage, f"{root}/Room/Outlet{i}/GroundSlot", (back_x - .055, y, height + .079),
                  (.001, .006, .016), mats["DarkRubber"])
        collider("Room/Floor", (table_x, table_y, -.025), (6., 6., .05), mats["Floor"])
    effective_lighting = _lighting(stage, root, cfg, base_xy, height, table_x, table_y)

    return {
        "root_path": root,
        "table_top_z": height,
        "robot_base_position_world": [*base_xy, height],
        "table_bounds_xy_world": [[base_xy[0] + x_min, table_y - width / 2],
                                   [base_xy[0] + x_min + depth, table_y + width / 2]],
        "collision_paths": collision_paths,
        "mount_hardware": mounting,
        "appearance": {"materials": effective_materials, "lighting": effective_lighting},
        "task_props_enabled": False,
    }


def build_workbench(stage, cfg=None):
    """Build only the common table, room, mounting hardware and lighting.

    Task-specific object, box and marker settings are not interpreted here.
    ``root_path`` defaults to the historical workbench root for compatibility.
    """
    cfg = dict(cfg or {})
    geometry = _validate_workbench(stage, cfg)
    workbench = _prepare_workbench(stage, cfg, geometry)
    return _finish_workbench(stage, cfg, workbench)


def build_tripod(stage, eye_world, target_world, tabletop_z=.76,
                 root_path="/World/PickPlace/Tripod"):
    """Official wrist-matching D435 housing on a visual-only tabletop tripod.

    ``eye_world`` is the RGB optical centre, not the housing or mount origin.
    The official camera-link, visual and bottom-screw transforms are solved
    backwards from this optical pose. The runner authors the actual sensor.
    No restriction on the sign of Y is imposed here.
    """
    import hashlib
    import json
    import xml.etree.ElementTree as ET

    import numpy as np

    from pressb.robots.piper.wrist_camera import (
        ROOT, ASSEMBLY_URDF, ASSEMBLY_SHA256, ASSEMBLY_COMMIT,
        _origin, _set_transform, _require_mesh,
    )

    eye = Gf.Vec3d(*_vector(eye_world, 3, "eye_world"))
    target = Gf.Vec3d(*_vector(target_world, 3, "target_world"))
    z0 = float(tabletop_z)
    if not math.isfinite(z0) or eye[2] <= z0 + .12:
        raise ValueError("Tripod camera must be at least 12 cm above its support")
    direction = target - eye
    if direction.GetLength() <= 1e-6:
        raise ValueError("Camera eye and target must differ")
    direction.Normalize()
    if abs(direction[2]) > 1. - 1e-8:
        raise ValueError("Tripod look-at requires a nonvertical optical direction")
    if stage.GetPrimAtPath(root_path):
        raise ValueError(f"Tripod root already exists: {root_path}")
    runtime_asset = ROOT / "assets/generated/official_wrist/assembly.usda"
    urdf_path = ROOT / "vendor/piper_isaac_sim" / ASSEMBLY_URDF
    if not runtime_asset.is_file() or not urdf_path.is_file():
        raise FileNotFoundError("Missing official D435 asset; prepare the wrist assembly first")
    provenance = json.loads((runtime_asset.parent / "provenance.json").read_text())
    runtime_hash = hashlib.sha256(runtime_asset.read_bytes()).hexdigest()
    if provenance.get("source_sha256") != ASSEMBLY_SHA256 or provenance.get("output_sha256") != runtime_hash:
        raise ValueError("Compatible D435 mesh provenance/checksum mismatch")
    source = ET.parse(urdf_path).getroot()
    bottom_to_link = _origin(source.find("joint[@name='camera_link_joint']"))
    link_to_optical = np.eye(4)
    link_to_optical[:3, :3] = [[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]]
    link_to_optical[:3, 3] = [0., .015, 0.]
    optical_gf = Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 0, 1)).GetInverse()
    world_optical = np.asarray(optical_gf).T
    world_link = world_optical @ np.linalg.inv(link_to_optical)
    world_housing = world_link @ _origin(source.find("link[@name='camera_link']/visual"))
    world_bottom = world_link @ np.linalg.inv(bottom_to_link)
    ball_local = np.array([0., 0., -.050])
    ball_world = world_bottom[:3, :3] @ ball_local + world_bottom[:3, 3]
    if ball_world[2] <= z0 + .07:
        raise ValueError("Official D435 mount is too low for the tabletop tripod")
    UsdGeom.Xform.Define(stage, root_path)
    black = _material(stage, root_path + "/Looks/Black", (.025, .031, .038), metallic=.60, roughness=.34)
    silver = _material(stage, root_path + "/Looks/Silver", (.38, .42, .47), metallic=.80, roughness=.30)
    housing_path = root_path + "/D435"
    housing = UsdGeom.Xform.Define(stage, housing_path)
    housing.GetPrim().GetReferences().AddReference(str(runtime_asset), "/meshes/d435")
    _require_mesh(stage, housing_path + "/ID1/mesh")
    _set_transform(housing, world_housing)
    UsdShade.MaterialBindingAPI.Apply(housing.GetPrim()).Bind(
        silver, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    bottom_path = root_path + "/BottomScrewFrame"
    bottom = UsdGeom.Xform.Define(stage, bottom_path)
    _set_transform(bottom, world_bottom)
    # The plate's upper surface is exactly the official mounting plane; the
    # tilted neck joins its underside to a ball centred on the vertical mast.
    _cube(stage, bottom_path + "/MountingPlate", (0., 0., -.003), (.034, .040, .006), black)
    _cylinder_between(stage, bottom_path + "/TiltStem", (0., 0., -.006),
                      tuple(ball_local), .009, black)
    ball = UsdGeom.Sphere.Define(stage, bottom_path + "/BallHead")
    ball.CreateRadiusAttr(.018)
    ball.AddTranslateOp().Set(Gf.Vec3d(*ball_local))
    _bind(ball.GetPrim(), black)
    head = ball_world
    pole_bottom = (head[0], head[1], z0 + .08)
    pole_top = tuple(head)
    _cylinder_between(stage, root_path + "/Column", pole_bottom, pole_top, .009, black)
    leg_attach = (head[0], head[1], z0 + (head[2] - z0) * .52)
    spread = min(.17, (eye[2] - z0) * .35)
    for i in range(3):
        angle = math.radians(30 + i * 120)
        foot = (head[0] + spread * math.cos(angle), head[1] + spread * math.sin(angle), z0 + .013)
        _cylinder_between(stage, f"{root_path}/Leg{i}", leg_attach, foot, .009, black)
        _cylinder_between(stage, f"{root_path}/Brace{i}", (head[0], head[1], z0 + .09),
                          ((leg_attach[0] + foot[0]) / 2, (leg_attach[1] + foot[1]) / 2,
                           (leg_attach[2] + foot[2]) / 2), .003, black)
    return {"root_path": root_path, "eye_world": list(eye), "target_world": list(target),
            "support_z": z0, "visual_only": True, "model": "Intel RealSense D435",
            "housing_path": housing_path, "housing_source_prim": "/meshes/d435",
            "runtime_asset_usd": str(runtime_asset), "runtime_asset_sha256": runtime_hash,
            "source_revision": ASSEMBLY_COMMIT,
            "world_optical_transform": world_optical.tolist(),
            "world_camera_link_transform": world_link.tolist(),
            "world_housing_transform": world_housing.tolist(),
            "world_bottom_screw_transform": world_bottom.tolist(),
            "bottom_screw_position_world": world_bottom[:3, 3].tolist(),
            "ball_center_world": ball_world.tolist()}
