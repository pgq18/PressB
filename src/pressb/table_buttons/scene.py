"""PAP workbench with three physical, vertically spring-loaded color buttons.

Only the task props differ from the pick-and-place scene. The existing PAP
builder retains its defaults; here its table, mounting hardware, room, and
lighting are reused without a box, loose object, or tabletop tape marks.
All button positions are metres relative to the robot base at the tabletop.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

from pxr import Gf, Sdf, UsdGeom, UsdPhysics

from ..pick_place.scene import (
    PhysxSchema, _bind, _color, _cube, _material, _positive, _vector,
    build_scene as build_workbench,
)

COLORS = ("red", "yellow", "blue")
DEFAULT_BUTTONS = {
    "red": {"center_xy": [.33, .12], "color": [.72, .008, .012]},
    "yellow": {"center_xy": [.38, 0.], "color": [.82, .59, .012]},
    "blue": {"center_xy": [.33, -.12], "color": [.008, .09, .78]},
}
DEFAULT_GEOMETRY = dict(
    cap_radius_m=.016, cap_height_m=.008, base_radius_m=.022,
    base_height_m=.017, travel_m=.004, mass_kg=.015,
    stiffness_n_m=300., damping_n_s_m=3., max_spring_force_n=5.,
)


def _cylinder(stage, path, center, radius, height, material, *, collision=False):
    shape = UsdGeom.Cylinder.Define(stage, path)
    shape.CreateAxisAttr("Z")
    shape.CreateRadiusAttr(radius)
    shape.CreateHeightAttr(height)
    shape.AddTranslateOp().Set(Gf.Vec3d(*center))
    _bind(shape.GetPrim(), material, collision=collision)
    return shape


def _bezel(stage, path, center, inner_radius, outer_radius, height, material):
    """Render an open ring; collision strips keep the cap's travel unobstructed."""
    segments = 48
    vertices = []
    for z in (-height / 2, height / 2):
        for radius in (inner_radius, outer_radius):
            for index in range(segments):
                angle = index * math.tau / segments
                vertices.append(Gf.Vec3f(radius * math.cos(angle), radius * math.sin(angle), z))
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        faces.extend([
            [i, j, segments+j, segments+i],
            [2*segments+i, 3*segments+i, 3*segments+j, 2*segments+j],
            [i, 2*segments+i, 2*segments+j, j],
            [segments+i, segments+j, 3*segments+j, 3*segments+i],
        ])
    ring = UsdGeom.Mesh.Define(stage, path)
    ring.CreatePointsAttr(vertices)
    ring.CreateFaceVertexCountsAttr([4] * len(faces))
    ring.CreateFaceVertexIndicesAttr([index for face in faces for index in face])
    ring.CreateSubdivisionSchemeAttr("none")
    ring.AddTranslateOp().Set(Gf.Vec3d(*center))
    _bind(ring.GetPrim(), material)
    paths = []
    collision_segments = 24
    radius = (inner_radius + outer_radius) / 2
    for i in range(collision_segments):
        angle = i * math.tau / collision_segments
        part = _cube(stage, f"{path}/Collision{i}",
                     (radius * math.cos(angle), radius * math.sin(angle), 0.),
                     (outer_radius - inner_radius, 2 * radius * math.tan(math.pi / collision_segments), height),
                     material, collision=True, yaw_deg=math.degrees(angle))
        # Render the smooth continuous ring above; the solver uses these boxes.
        UsdGeom.Imageable(part).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        paths.append(str(part.GetPath()))
    return paths


def build_scene(stage, cfg=None):
    """Return common workbench metadata and measured-physics button references.

    Each cap is a dynamic body with a world-anchored Z prismatic joint, limits
    ``[-travel_m, 0]`` and a force spring at zero. Gravity is disabled on the
    cap as for the elevator switches, so unpressed travel remains exactly zero.
    Positive press depth is ``cap_rest_world[2] - measured_cap_world_z``.
    No kinematic motion, attachment, or scripted press event is introduced.
    """
    cfg = dict(cfg or {})
    cfg["root_path"] = cfg.get("root_path", "/World/TableButtons")
    cfg["task_props"] = False
    cfg["tape_markers"] = []
    buttons = cfg.get("buttons", DEFAULT_BUTTONS)
    if not isinstance(buttons, Mapping) or set(buttons) != set(COLORS):
        raise ValueError("buttons must contain exactly red, yellow and blue")
    overrides = cfg.get("button_geometry", {})
    if not isinstance(overrides, Mapping) or set(overrides) - set(DEFAULT_GEOMETRY):
        raise ValueError("Unsupported button_geometry properties")
    geometry = {name: _positive(value, f"button_geometry.{name}")
                for name, value in {**DEFAULT_GEOMETRY, **overrides}.items()}
    cap_radius, cap_height = geometry["cap_radius_m"], geometry["cap_height_m"]
    base_radius, base_height = geometry["base_radius_m"], geometry["base_height_m"]
    travel = geometry["travel_m"]
    gap = .0015
    if base_radius <= cap_radius + 2 * gap:
        raise ValueError("Button housing must leave clearance around the cap")
    if base_height <= 2 * travel or cap_height <= travel:
        raise ValueError("Button base/cap heights must leave room for the full spring travel")
    specs = {}
    for color in COLORS:
        spec = buttons[color]
        if not isinstance(spec, Mapping):
            raise ValueError(f"buttons.{color} must be a mapping")
        specs[color] = dict(center_xy=_vector(spec.get("center_xy"), 2, f"buttons.{color}.center_xy"),
                            color=_color(spec.get("color", DEFAULT_BUTTONS[color]["color"]),
                                         f"buttons.{color}.color"))
    for i, color in enumerate(COLORS):
        for other in COLORS[i + 1:]:
            if math.dist(specs[color]["center_xy"], specs[other]["center_xy"]) <= 2 * base_radius:
                raise ValueError(f"Button housings overlap: {color}, {other}")

    scene = build_workbench(stage, cfg)
    root, base = scene["root_path"], scene["robot_base_position_world"]
    bounds = scene["table_bounds_xy_world"]
    black = _material(stage, root + "/Looks/ButtonHousing", [.012, .016, .02],
                      roughness=.38, metallic=.15, static_friction=.8, dynamic_friction=.65)
    bezel_material = _material(stage, root + "/Looks/ButtonBezel", [.12, .14, .16],
                               roughness=.3, metallic=.7, static_friction=.8, dynamic_friction=.65)
    scene["buttons"] = {}
    for color, spec in specs.items():
        x, y = [base[i] + spec["center_xy"][i] for i in range(2)]
        if any(p - base_radius < bounds[0][i] or p + base_radius > bounds[1][i]
               for i, p in enumerate((x, y))):
            raise ValueError(f"Button {color} must lie fully on the tabletop")
        path = f"{root}/Buttons/{color.title()}"
        UsdGeom.Xform.Define(stage, path)
        material = _material(stage, f"{root}/Looks/Button{color.title()}", spec["color"],
                             roughness=.22, metallic=0., specular=.5,
                             static_friction=.8, dynamic_friction=.65)
        # Lower disk and a hollow upper bezel: no collider intersects the cap,
        # including at the deepest joint limit. The housing is fixed to table.
        lower_height = base_height - 2 * travel
        lower = _cylinder(stage, path + "/Housing", (x, y, base[2] + lower_height / 2),
                          base_radius, lower_height, black, collision=True)
        scene["collision_paths"].append(str(lower.GetPath()))
        scene["collision_paths"].extend(_bezel(
            stage, path + "/Bezel", (x, y, base[2] + lower_height + travel),
            cap_radius + gap, base_radius, 2 * travel, bezel_material))
        rest = [x, y, base[2] + base_height + cap_height / 2]
        body = UsdGeom.Xform.Define(stage, path + "/Cap")
        body.AddTranslateOp().Set(Gf.Vec3d(*rest))
        prim = body.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(False)
        mass = UsdPhysics.MassAPI.Apply(prim)
        m = geometry["mass_kg"]
        mass.CreateMassAttr(m)
        transverse = m * (3 * cap_radius**2 + cap_height**2) / 12
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(transverse, transverse, m * cap_radius**2 / 2))
        mass.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))
        if PhysxSchema is not None:
            api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            api.CreateDisableGravityAttr(True)
            api.CreateEnableCCDAttr(True)
            api.CreateSolverPositionIterationCountAttr(16)
            api.CreateSolverVelocityIterationCountAttr(4)
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
        cap = _cylinder(stage, path + "/Cap/Collision", (0., 0., 0.), cap_radius,
                        cap_height, material, collision=True)
        scene["collision_paths"].append(str(cap.GetPath()))
        prim.CreateAttribute("pressb:color", Sdf.ValueTypeNames.Token).Set(color)
        prim.CreateAttribute("pressb:pressed", Sdf.ValueTypeNames.Bool).Set(False)
        joint = UsdPhysics.PrismaticJoint.Define(stage, path + "/SpringJoint")
        joint.CreateBody1Rel().SetTargets([prim.GetPath()])
        joint.CreateAxisAttr("Z")
        joint.CreateLocalPos0Attr(Gf.Vec3f(*rest))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        joint.CreateLowerLimitAttr(-travel)
        joint.CreateUpperLimitAttr(0.)
        joint.CreateCollisionEnabledAttr(False)
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
        drive.CreateTypeAttr("force")
        drive.CreateTargetPositionAttr(0.)
        drive.CreateTargetVelocityAttr(0.)
        drive.CreateStiffnessAttr(geometry["stiffness_n_m"])
        drive.CreateDampingAttr(geometry["damping_n_s_m"])
        drive.CreateMaxForceAttr(geometry["max_spring_force_n"])
        scene["buttons"][color] = dict(
            body_path=str(prim.GetPath()), collision_path=str(cap.GetPath()),
            joint_path=str(joint.GetPath()), material_path=str(material.GetPath()),
            shader_path=str(material.GetPath()) + "/Mdl", center_xy_base=spec["center_xy"],
            cap_rest_world=rest, neutral_top_world=[x, y, rest[2] + cap_height / 2],
            cap_radius_m=cap_radius, base_radius_m=base_radius,
            travel=travel, travel_m=travel, color=spec["color"],
        )
    scene["button_geometry"] = geometry
    scene["button_physics"] = "Dynamic caps; world-anchored Z springs; downward displacement from real contact."
    return scene
