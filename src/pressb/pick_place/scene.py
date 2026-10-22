"""Independent, parameterized PiPER tabletop pick-and-place geometry.

The robot base faces +X; +Y is its left. Task XY coordinates are relative to
``robot_base_xy`` and task Z coordinates are measured from the tabletop. The
module authors USD geometry and contact properties only. It never constrains,
attaches, teleports, or otherwise controls the object after scene construction.

Run after SimulationApp starts, or with an isolated USD Python interpreter for
structural inspection. The dimensions are configurable reconstruction estimates;
camera registration and measured robot poses are handled by the task runner.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, UsdLux

try:
    from pxr import PhysxSchema
except ImportError:  # Plain USD inspection does not require Isaac Sim.
    PhysxSchema = None


def _positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


def _vector(value, length, name):
    values = [float(v) for v in value]
    if len(values) != length or not all(math.isfinite(v) for v in values):
        raise ValueError(f"{name} must contain {length} finite numbers")
    return values


def _unit(value, name):
    value = float(value)
    if not math.isfinite(value) or not 0. <= value <= 1.:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return value


def _color(value, name):
    return [_unit(v, name) for v in _vector(value, 3, name)]


def _material_settings(defaults, overrides, name):
    """Resolve numeric appearance only; contact parameters are never accepted."""
    if not isinstance(overrides, Mapping):
        raise ValueError(f"{name} must be a mapping of numeric material properties")
    allowed = {"color", "roughness", "metallic", "specular"}
    unknown = set(overrides) - allowed
    if unknown:
        raise ValueError(f"Unsupported {name} properties: {sorted(unknown)}; supported: {sorted(allowed)}")
    settings = {**defaults, **overrides}
    settings["color"] = _color(settings["color"], f"{name}.color (linear RGB)")
    settings["metallic"] = _unit(settings["metallic"], f"{name}.metallic")
    settings["roughness"] = _unit(settings["roughness"], f"{name}.roughness")
    if "specular" in overrides:
        settings["specular"] = _unit(overrides["specular"], f"{name}.specular")
        settings["specular_authored"] = True
    else:
        settings.setdefault("specular", .5)
        settings.setdefault("specular_authored", False)
    return settings


def _material(stage, path, color, *, metallic=0.0, roughness=0.6,
              static_friction=None, dynamic_friction=None, specular=None):
    """MDL for RTX plus PreviewSurface for ordinary USD viewers."""
    material = UsdShade.Material.Define(stage, path)
    mdl = UsdShade.Shader.Define(stage, path + "/Mdl")
    mdl.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    mdl.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    mdl.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    mdl.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    mdl.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(float(metallic))
    mdl.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(float(roughness))
    if specular is not None:
        mdl.CreateInput("specular_level", Sdf.ValueTypeNames.Float).Set(float(specular))
    mdl.CreateOutput("out", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput("mdl").ConnectToSource(mdl.ConnectableAPI(), "out")
    preview = UsdShade.Shader.Define(stage, path + "/Preview")
    preview.CreateIdAttr("UsdPreviewSurface")
    preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    preview.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    preview.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    if specular is not None:
        # OmniPBR uses normal reflectivity 0.08 weighted by specular_level.
        # PreviewSurface's metallic workflow exposes IOR instead: match its
        # dielectric F0 while retaining metallic reflectance and diffuse color.
        sqrt_f0 = math.sqrt(.08 * float(specular))
        preview.CreateInput("ior", Sdf.ValueTypeNames.Float).Set((1 + sqrt_f0) / (1 - sqrt_f0))
    material.CreateSurfaceOutput().ConnectToSource(preview.ConnectableAPI(), "surface")
    if static_friction is not None:
        physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics.CreateStaticFrictionAttr(float(static_friction))
        physics.CreateDynamicFrictionAttr(float(dynamic_friction))
        physics.CreateRestitutionAttr(0.0)
    return material


def _bind(prim, material, collision=False):
    binding = UsdShade.MaterialBindingAPI.Apply(prim)
    binding.Bind(material)
    if collision:
        binding.Bind(material, materialPurpose="physics")
        UsdPhysics.CollisionAPI.Apply(prim)
        if PhysxSchema is not None:
            api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            api.CreateContactOffsetAttr(0.0005)
            api.CreateRestOffsetAttr(0.0)


def _cube(stage, path, center, size, material, *, collision=False, yaw_deg=0.):
    shape = UsdGeom.Cube.Define(stage, path)
    shape.CreateSizeAttr(1.)
    shape.AddTranslateOp().Set(Gf.Vec3d(*center))
    if yaw_deg:
        shape.AddRotateZOp().Set(float(yaw_deg))
    shape.AddScaleOp().Set(Gf.Vec3f(*size))
    _bind(shape.GetPrim(), material, collision)
    return shape


def _cylinder_between(stage, path, start, end, radius, material):
    start, end = Gf.Vec3d(*start), Gf.Vec3d(*end)
    delta = end - start
    length = delta.GetLength()
    if length <= 0:
        raise ValueError("Cylinder endpoints must differ")
    shape = UsdGeom.Cylinder.Define(stage, path)
    shape.CreateAxisAttr("Z")
    shape.CreateHeightAttr(length)
    shape.CreateRadiusAttr(float(radius))
    shape.AddTranslateOp().Set((start + end) / 2.)
    rotation = Gf.Rotation(Gf.Vec3d(0, 0, 1), delta.GetNormalized()).GetQuat()
    shape.AddOrientOp().Set(Gf.Quatf(rotation))
    _bind(shape.GetPrim(), material)
    return shape


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


def _box_outer_surfaces(stage, box_path, dimensions, material):
    """Four outward-facing visual quads, independent of the five box colliders."""
    lx, ly, lz = dimensions
    offset = .00005
    x, y = lx / 2, ly / 2
    faces = [
        ("Front", (-1., 0., 0.), [(-x-offset, -y, 0), (-x-offset, -y, lz),
                                    (-x-offset, y, lz), (-x-offset, y, 0)]),
        ("Back", (1., 0., 0.), [(x+offset, -y, 0), (x+offset, y, 0),
                                  (x+offset, y, lz), (x+offset, -y, lz)]),
        ("Right", (0., -1., 0.), [(-x, -y-offset, 0), (x, -y-offset, 0),
                                    (x, -y-offset, lz), (-x, -y-offset, lz)]),
        ("Left", (0., 1., 0.), [(-x, y+offset, 0), (-x, y+offset, lz),
                                  (x, y+offset, lz), (x, y+offset, 0)]),
    ]
    parent = box_path + "/OuterAppearance"
    UsdGeom.Xform.Define(stage, parent)
    paths = []
    for name, normal, points in faces:
        surface = UsdGeom.Mesh.Define(stage, parent + "/" + name)
        surface.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
        surface.CreateFaceVertexCountsAttr([4])
        surface.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        surface.CreateNormalsAttr([Gf.Vec3f(*normal)])
        surface.SetNormalsInterpolation("constant")
        surface.CreateSubdivisionSchemeAttr("none")
        surface.CreateDoubleSidedAttr(False)
        _bind(surface.GetPrim(), material)
        paths.append(str(surface.GetPath()))
    return dict(enabled=True, visual_only=True, paths=paths,
                material_path=str(material.GetPath()), offset_m=offset)


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


def build_scene(stage, cfg=None):
    """Build the task and return paths and metric geometry for its runner.

    Flat table keys: ``table_height``, ``table_depth``, ``table_width``,
    ``table_x_min``. Alternatively pass ``table={height, depth, width, x_min}``.
    ``box={center_xy, size, wall_thickness, yaw_deg}`` uses outer XYZ size.
    ``cube={center_xy, edge, mass, yaw_deg, clearance}`` uses a measured or
    estimated edge length and mass in kilograms. ``clearance`` defaults to
    0.5 mm above the tabletop to let gravity establish a real contact.
    Alternatively ``object={shape: "cube", edge: e, ...}``,
    ``object={shape: "cuboid", size: [x, y, z], ...}`` or
    ``object={shape: "cylinder", radius: r, height: h, ...}`` provides another
    primitive rigid object. All accept ``center_xy``, ``mass``, ``yaw_deg``,
    ``clearance``, optional linear-RGB ``color``, and ``grasp_width_m``. Cylinders
    start upright along Z. The legacy ``cube`` path is unchanged when no
    ``object`` is configured; explicit object geometry takes precedence.
    Explicit cubes require one ``edge``; an optional ``size`` must agree with
    that edge on all three axes, so rectangular fits cannot silently pass.
    ``mount_hardware=False`` omits the visual mounting plate and orange clamps.
    ``task_props=False`` builds only the shared table, mounting, room and lights;
    no box, movable object or tape markers are authored. The default is unchanged.

    ``materials={name: {color, roughness, metallic, specular}}`` overrides the
    older ``material_colors`` palette. Colors are linear RGB in [0, 1]; the
    scalar appearance coefficients are also in [0, 1]. ``object.material``
    overrides the object's palette entry, with ``object.color`` retaining
    highest color priority. Appearance never changes contact properties.
    Optional ``materials.CardboardOuter`` adds four outward-facing visual
    surfaces to the box; its inner walls and five collision faces are unchanged.
    ``lighting={ambient: {intensity, color}, area_lights: [...]}`` controls
    lighting. Each area light has name, position_base_m, target_base_m, width,
    height, intensity and color. Positions are relative to the robot base,
    whose world Z equals the tabletop; local -Z points towards the target.

    ``box_inside`` describes the open cavity in the box's local frame; its
    transform is returned separately so the runner can test the measured object
    against the rotated cavity. Bounds describe containment geometry, not a
    declaration that a grasp or placement has succeeded.
    """
    cfg = dict(cfg or {})
    root = str(cfg.get("root_path", "/World/PickPlace")).rstrip("/")
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
    box_cfg, cube_cfg = cfg.get("box", {}), cfg.get("cube", {})
    box_xy = _vector(box_cfg.get("center_xy", [.40, .23]), 2, "box.center_xy")
    outer = _vector(box_cfg.get("size", [.18, .16, .075]), 3, "box.size")
    thickness = _positive(box_cfg.get("wall_thickness", .005), "box.wall_thickness")
    if min(outer[:2]) <= 2 * thickness or outer[2] <= thickness:
        raise ValueError("Box dimensions must leave a positive open cavity")
    box_yaw = float(box_cfg.get("yaw_deg", 0.))
    object_cfg = cfg.get("object") if "object" in cfg else cube_cfg
    if not isinstance(object_cfg, Mapping):
        raise ValueError("object/cube must be a mapping")
    prefix = "object" if "object" in cfg else "cube"
    object_shape = str(object_cfg.get("shape", "cuboid")) if prefix == "object" else "cube"
    if prefix == "object" and object_shape not in {"cube", "cuboid", "cylinder"}:
        raise ValueError("object.shape must be cube, cuboid or cylinder")
    radius = None
    if object_shape == "cube":
        if prefix == "object" and "edge" not in object_cfg:
            raise ValueError("object.shape='cube' requires a single object.edge")
        edge = _positive(object_cfg.get("edge", .04), f"{prefix}.edge")
        size_xyz = [edge, edge, edge]
        if prefix == "object":
            for name in ("size", "size_xyz"):
                if name in object_cfg:
                    supplied_size = _vector(object_cfg[name], 3, f"object.{name}")
                    if any(not math.isclose(v, edge, rel_tol=1e-9, abs_tol=1e-12)
                           for v in supplied_size):
                        raise ValueError(f"object.{name} must equal [object.edge] * 3 for a cube")
    elif object_shape == "cuboid":
        edge = None
        size_xyz = _vector(object_cfg.get("size", []), 3, "object.size")
        for component in size_xyz:
            _positive(component, "object.size component")
    elif object_shape == "cylinder":
        edge = None
        radius = _positive(object_cfg.get("radius", 0), "object.radius")
        object_height = _positive(object_cfg.get("height", 0), "object.height")
        size_xyz = [2 * radius, 2 * radius, object_height]
    else:
        raise ValueError("object.shape must be cube, cuboid or cylinder")
    cube_xy = _vector(object_cfg.get("center_xy", [.36, -.10]), 2, f"{prefix}.center_xy")
    mass = _positive(object_cfg.get("mass", .035), f"{prefix}.mass")
    cube_yaw = float(object_cfg.get("yaw_deg", 0.))
    clearance = float(object_cfg.get("clearance", .0005))
    grasp_width = _positive(object_cfg.get("grasp_width_m", min(size_xyz[:2])), f"{prefix}.grasp_width_m")
    object_color = None
    if prefix == "object" and "color" in object_cfg:
        object_color = _vector(object_cfg["color"], 3, "object.color")
        if any(c < 0 or c > 1 for c in object_color):
            raise ValueError("object.color must contain linear RGB components in [0, 1]")
    if not all(math.isfinite(v) for v in [box_yaw, cube_yaw, clearance]) or clearance < 0:
        raise ValueError("Box/object yaw must be finite and object clearance nonnegative")
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

    if cfg.get("task_props", True):
        # Five static box faces form an open container. No top collider or hidden
        # attraction force is added. All dimensions below are local to its floor.
        box_path = root + "/Box"
        box = UsdGeom.Xform.Define(stage, box_path)
        box_center_world = [base_xy[0] + box_xy[0], base_xy[1] + box_xy[1], height]
        box.AddTranslateOp().Set(Gf.Vec3d(*box_center_world))
        box.AddRotateZOp().Set(box_yaw)
        lx, ly, lz = outer
        faces = [
            ("Bottom", (0, 0, thickness / 2), (lx, ly, thickness)),
            ("Front", (-(lx - thickness) / 2, 0, (lz + thickness) / 2), (thickness, ly, lz - thickness)),
            ("Back", ((lx - thickness) / 2, 0, (lz + thickness) / 2), (thickness, ly, lz - thickness)),
            ("Right", (0, -(ly - thickness) / 2, (lz + thickness) / 2), (lx - 2 * thickness, thickness, lz - thickness)),
            ("Left", (0, (ly - thickness) / 2, (lz + thickness) / 2), (lx - 2 * thickness, thickness, lz - thickness)),
        ]
        for name, center, size in faces:
            shape = _cube(stage, f"{box_path}/{name}", center, size, mats["Cardboard"], collision=True)
            collision_paths.append(str(shape.GetPath()))
        outer_surfaces = (_box_outer_surfaces(stage, box_path, outer, mats["CardboardOuter"])
                          if "CardboardOuter" in mats else
                          dict(enabled=False, visual_only=True, paths=[], material_path=None, offset_m=0.))
        # Thin dark marks on the cut edge suggest exposed corrugation without
        # modifying the simple collision boundary used by grasping physics.
        for axis, long_size, short_size in [(0, lx, ly), (1, ly, lx)]:
            for sign in [-1, 1]:
                count = max(2, round((long_size - 2 * thickness) / .005))
                for i in range(count):
                    along = -long_size / 2 + thickness + (i + .5) * (long_size - 2 * thickness) / count
                    center = [along, sign * (short_size - thickness) / 2, lz + .00006]
                    size = [.0008, thickness * .72, .00012]
                    if axis == 1:
                        center[0], center[1] = center[1], center[0]
                        size[0], size[1] = size[1], size[0]
                    _cube(stage, f"{box_path}/CutEdge/A{axis}S{'P' if sign > 0 else 'M'}_{i}",
                          center, size, mats["CardboardDark"])
        _cube(stage, box_path + "/FloorFold", (0, 0, thickness + .00008),
              (.0007, ly - 2 * thickness, .00012), mats["CardboardCut"])

        object_material = mats["RedCube"]
        # Reuse the existing physical material coefficients regardless of which
        # appearance overrides create a dedicated object visual material.
        effective_object = _material_settings(effective_materials["RedCube"],
                                              object_cfg.get("material", {}), f"{prefix}.material")
        if object_color is not None:
            effective_object["color"] = object_color
        if object_color is not None or object_cfg.get("material"):
            object_material = _material(stage, root + "/Looks/TaskObject", effective_object["color"],
                                        roughness=effective_object["roughness"], metallic=effective_object["metallic"],
                                        specular=effective_object["specular"] if effective_object["specular_authored"] else None,
                                        static_friction=.8, dynamic_friction=.65)
        effective_object["path"] = str(object_material.GetPath())
        cube_pos = [base_xy[0] + cube_xy[0], base_xy[1] + cube_xy[1], height + size_xyz[2] / 2 + clearance]
        if object_shape == "cylinder":
            cube = UsdGeom.Cylinder.Define(stage, root + "/Object")
            cube.CreateAxisAttr("Z")
            cube.CreateRadiusAttr(radius)
            cube.CreateHeightAttr(size_xyz[2])
            cube.AddTranslateOp().Set(Gf.Vec3d(*cube_pos))
            if cube_yaw:
                cube.AddRotateZOp().Set(cube_yaw)
            _bind(cube.GetPrim(), object_material, collision=True)
            collision_paths.append(str(cube.GetPath()))
        else:
            cube = collider("Object", cube_pos, tuple(size_xyz), object_material, yaw_deg=cube_yaw)
        body = UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
        body.CreateRigidBodyEnabledAttr(True)
        body.CreateKinematicEnabledAttr(False)
        mass_api = UsdPhysics.MassAPI.Apply(cube.GetPrim())
        mass_api.CreateMassAttr(mass)
        # Analytic inertia of the physical primitive in its local principal frame.
        # Keep the original cube expression exactly for existing reconstruction runs.
        if object_shape == "cube":
            inertia = mass * edge * edge / 6
            diagonal_inertia = [inertia, inertia, inertia]
        elif object_shape == "cylinder":
            transverse = mass * (3 * radius * radius + size_xyz[2] ** 2) / 12
            diagonal_inertia = [transverse, transverse, mass * radius * radius / 2]
        else:
            sx, sy, sz = size_xyz
            diagonal_inertia = [mass * (sy * sy + sz * sz) / 12,
                                mass * (sx * sx + sz * sz) / 12,
                                mass * (sx * sx + sy * sy) / 12]
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*diagonal_inertia))
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))
        if PhysxSchema is not None:
            rb = PhysxSchema.PhysxRigidBodyAPI.Apply(cube.GetPrim())
            rb.CreateSolverPositionIterationCountAttr(16)
            rb.CreateSolverVelocityIterationCountAttr(4)
            rb.CreateEnableCCDAttr(True)

        tape = cfg.get("tape_markers", [
            {"xy": [.23, -.18], "yaw_deg": 0}, {"xy": [.23, .04], "yaw_deg": 90},
            {"xy": [.47, -.18], "yaw_deg": 90}, {"xy": [.47, .04], "yaw_deg": 0},
        ])
        for i, marker in enumerate(tape):
            xy = _vector(marker["xy"], 2, f"tape_markers[{i}].xy")
            _cube(stage, f"{root}/Tape/Marker{i}", (base_xy[0] + xy[0], base_xy[1] + xy[1], height + .00012),
                  (.035, .012, .0002), mats["Tape"], yaw_deg=float(marker.get("yaw_deg", 0.)))

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

    if not cfg.get("task_props", True):
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

    inside_min = [-lx / 2 + thickness, -ly / 2 + thickness, thickness]
    inside_max = [lx / 2 - thickness, ly / 2 - thickness, lz]
    theta = math.radians(box_yaw)
    corners = [[box_center_world[0] + math.cos(theta) * x - math.sin(theta) * y,
                box_center_world[1] + math.sin(theta) * x + math.cos(theta) * y,
                height + z]
               for x in [inside_min[0], inside_max[0]]
               for y in [inside_min[1], inside_max[1]]
               for z in [inside_min[2], inside_max[2]]]
    return {
        "root_path": root,
        "table_top_z": height,
        "robot_base_position_world": [*base_xy, height],
        "table_bounds_xy_world": [[base_xy[0] + x_min, table_y - width / 2],
                                   [base_xy[0] + x_min + depth, table_y + width / 2]],
        "cube_path": str(cube.GetPath()),
        "cube_initial_position_world": cube_pos,
        "cube_edge": edge,
        "cube_mass": mass,
        "object_path": str(cube.GetPath()),
        "object_initial_position_world": cube_pos,
        "object_mass": mass,
        "object_geometry": {"shape": object_shape, "size_xyz": size_xyz, "edge": edge,
                            "radius": radius, "height": size_xyz[2],
                            "grasp_width_m": grasp_width},
        "box_path": box_path,
        "box_inside": {"min_local": inside_min, "max_local": inside_max,
                       "origin_world": box_center_world, "yaw_deg": box_yaw},
        "box_inner_bounds_world": {"min": [min(p[i] for p in corners) for i in range(3)],
                                   "max": [max(p[i] for p in corners) for i in range(3)],
                                   "is_axis_aligned_envelope": True},
        "collision_paths": collision_paths,
        "mount_hardware": mounting,
        "appearance": {"materials": effective_materials, "object_material": effective_object,
                       "lighting": effective_lighting,
                       "box_outer_surfaces": outer_surfaces,
                       "specular_preview_mapping": "Dielectric F0 = 0.08 * specular; equivalent PreviewSurface IOR",
                       "texture_support": "Numeric materials only; texture properties are not accepted"},
        "geometry_estimates": {"table_dimensions": [depth, width, height],
                               "box_outer_dimensions": outer, "box_wall_thickness": thickness,
                               "cube_edge": edge, "cube_mass": mass,
                               "object_shape": object_shape, "object_size_xyz": size_xyz,
                               "object_initial_center_xy_base": cube_xy,
                               "object_dimensions_and_position_status": "Reconstruction estimates unless independently measured",
                               "note": "Configurable reconstruction estimates; not direct measurements."},
        "object_physics": "Dynamic free rigid body; no attachment, teleportation or grasp constraint.",
    }


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

    from ..wrist_camera import (
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
