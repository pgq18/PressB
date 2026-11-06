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

from ...runtime.scene import (
    PhysxSchema, _positive, _vector, _unit, _color, _material_settings,
    _material, _bind, _cube, _cylinder_between,
)
from ...runtime.workbench import (
    _materials, _lighting, _mount_hardware, _validate_workbench,
    _prepare_workbench, _finish_workbench, build_tripod,
)


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
    geometry = _validate_workbench(stage, cfg)
    root, height, depth, width, x_min, base_xy = (
        geometry[name] for name in ("root", "height", "depth", "width", "x_min", "base_xy"))
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
    workbench = _prepare_workbench(stage, cfg, geometry)
    effective_materials, mats, table_x, table_y, collision_paths, collider, mounting = (
        workbench[name] for name in
        ("effective_materials", "mats", "table_x", "table_y", "collision_paths", "collider", "mounting"))

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

    common = _finish_workbench(stage, cfg, workbench)
    effective_lighting = common["appearance"]["lighting"]

    if not cfg.get("task_props", True):
        return common

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
