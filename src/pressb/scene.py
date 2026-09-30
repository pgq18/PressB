"""Isaac Sim scene assets and a physically actuated, twelve-button lift panel.

Call after SimulationApp starts. Coordinates are metres, Z up; the robot faces
+X and presses the panel along +X. Looking towards the panel, +Y is left.
"""

from __future__ import annotations

from pathlib import Path
import math
import warnings

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

try:
    from pxr import PhysxSchema
except ImportError:  # Allows USD-only structural inspection outside Isaac Sim.
    PhysxSchema = None


ROOT = Path(__file__).resolve().parents[2]
TABLE_RELATIVE = "Environments/Simple_Room/Props/table_low.usd"
ASSET_ROOT_URL = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/"
BUTTON_FEEDBACK_STYLE = "edge_border"
BUTTON_BORDER_WIDTH_M = 0.002
BUTTON_BORDER_SIDES = ("Top", "Bottom", "Left", "Right")
BUTTON_OFF_COLOR = (0.14, 0.17, 0.20)
BUTTON_ON_COLOR = (0.95, 0.27, 0.015)
BUTTON_ON_EMISSION = (1.7, 0.33, 0.012)
BACK_WALL_COMPONENTS = ("Wall", "Baseboard", "CornerTrim", "Handrail", "LightStrip",
                        *(f"CladdingSeam{i}" for i in range(4)), *(f"RailMount{i}" for i in range(2)))


def set_back_wall_offset(stage, environment_path, offset_x_m, op_suffix="panelOffset"):
    """Translate the back wall and its attached trim in world X, before scaling."""
    for name in BACK_WALL_COMPONENTS:
        prim = stage.GetPrimAtPath(f"{environment_path}/{name}")
        if not prim:
            raise ValueError(f"Missing back-wall component: {environment_path}/{name}")
        xform = UsdGeom.Xformable(prim)
        attribute = prim.GetAttribute(f"xformOp:translate:{op_suffix}")
        operation = UsdGeom.XformOp(attribute) if attribute else xform.AddTranslateOp(
            UsdGeom.XformOp.PrecisionDouble, op_suffix)
        operation.Set(Gf.Vec3d(float(offset_x_m), 0., 0.))
        # Cube dimensions are authored as scale; append-order translation would
        # scale the offset and detach trim from the wall.
        order = [op for op in xform.GetOrderedXformOps() if op.GetName() != operation.GetName()]
        xform.SetXformOpOrder([operation, *order], resetXformStack=xform.GetResetXformStack())


def _material(stage, name, color, metallic=0.0, roughness=0.45, emission=(0, 0, 0)):
    material = UsdShade.Material.Define(stage, f"/World/Looks/{name}")
    shader = UsdShade.Shader.Define(stage, f"{material.GetPath()}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emission))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _bind(prim, material):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def _cube(stage, path, center, size, material, collision=False):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(*center))
    cube.AddScaleOp().Set(Gf.Vec3f(*size))
    _bind(cube.GetPrim(), material)
    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        if PhysxSchema:
            api = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
            api.CreateContactOffsetAttr(0.0005)
            api.CreateRestOffsetAttr(0.0)
    return cube.GetPrim()


def _texture_material(stage, name, texture, roughness=0.6):
    material = _material(stage, name, (0.7, 0.7, 0.7), roughness=roughness)
    shader = UsdShade.Shader.Get(stage, f"{material.GetPath()}/Shader")
    reader = UsdShade.Shader.Define(stage, f"{material.GetPath()}/UV")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tex = UsdShade.Shader.Define(stage, f"{material.GetPath()}/Texture")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(Path(texture).resolve()))
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    shader.GetInput("diffuseColor").ConnectToSource(tex.ConnectableAPI(), "rgb")
    return material


def _table(stage, cfg, material):
    height = float(cfg.get("table_height", 0.76))
    edge = float(cfg.get("table_edge_x", 0.10))
    depth, width = float(cfg.get("table_depth", 0.75)), float(cfg.get("table_width", 1.0))
    center = Gf.Vec3d(edge - depth / 2, 0, height / 2)
    # The stock mesh is an irregular scanned desk; its bounding-box maximum is
    # not a level work surface. Retain it as the stand and add a precise slab.
    top_thickness = 0.04
    support_height = height - top_thickness
    asset = Path(cfg.get("table_asset", ROOT / "assets/isaac" / TABLE_RELATIVE))
    if not asset.is_absolute():
        asset = ROOT / asset
    result = {"source_url": ASSET_ROOT_URL + TABLE_RELATIVE, "local_path": str(asset), "available": asset.is_file()}
    if asset.is_file():
        container = UsdGeom.Xform.Define(stage, "/World/Environment/Table")
        source = UsdGeom.Xform.Define(stage, "/World/Environment/Table/OfficialAsset")
        source.GetPrim().GetReferences().AddReference(str(asset.resolve()))
        bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(source.GetPrim()).ComputeAlignedBox()
        minimum, maximum = bbox.GetMin(), bbox.GetMax()
        extent = maximum - minimum
        if min(extent) <= 0:
            raise RuntimeError(f"Invalid official table asset dimensions: {extent}")
        scale = Gf.Vec3d(depth / extent[0], width / extent[1], support_height / extent[2])
        support_center = Gf.Vec3d(center[0], 0, support_height / 2)
        offset = support_center - Gf.CompMult((minimum + maximum) / 2, scale)
        container.AddTranslateOp().Set(offset)
        container.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(scale)
        for prim in Usd.PrimRange(source.GetPrim()):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
            if prim.IsA(UsdGeom.Mesh):
                _bind(prim, material)
        tabletop = _material(stage, "PrecisionTabletop", (0.20, 0.23, 0.25), roughness=0.64)
        edge_trim = _material(stage, "TabletopEdgeTrim", (0.07, 0.085, 0.10), metallic=0.30, roughness=0.42)
        _cube(stage, "/World/Environment/PrecisionTabletop", (center[0], 0, height - top_thickness / 2), (depth, width, top_thickness), tabletop)
        _cube(stage, "/World/Environment/TabletopEdgeTrim", (center[0], 0, height - top_thickness + 0.005), (depth + 0.001, width + 0.001, 0.010), edge_trim)
        result.update(mode="official_mesh_with_precision_tabletop", original_extent=list(extent), scale=list(scale), tabletop_z=height, tabletop_thickness=top_thickness)
    else:
        warnings.warn(f"Official table asset missing: {asset}; using a clearly reported geometric fallback.")
        _cube(stage, "/World/Environment/Table/Top", (center[0], 0, height - 0.02), (depth, width, 0.04), material)
        for i, (x, y) in enumerate([(edge - 0.06, -width / 2 + 0.06), (edge - 0.06, width / 2 - 0.06), (edge - depth + 0.06, -width / 2 + 0.06), (edge - depth + 0.06, width / 2 - 0.06)]):
            _cube(stage, f"/World/Environment/Table/Leg{i}", (x, y, (height - 0.04) / 2), (0.045, 0.045, height - 0.04), material, True)
        result["mode"] = "procedural_fallback"
    # A simple, exact collision top avoids contacts on decorative mesh triangles.
    proxy = _cube(stage, "/World/Environment/TableCollision", (center[0], 0, height - 0.02), (depth, width, 0.04), material, True)
    UsdGeom.Imageable(proxy).MakeInvisible()
    result["collision_proxy"] = str(proxy.GetPath())
    return result


_SEGMENTS = {
    "0": "abcedf", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc",
    "5": "afgcd", "6": "afgecd", "7": "abc", "8": "abcdefg", "9": "abfgcd",
}


def _number(stage, parent, number, x, y, z, material, height=0.014):
    """Mesh digits, with +Y the text's left, visible without font dependencies."""
    width = height * 0.48
    stroke = height * 0.10
    spacing = width * 1.42
    text = str(number)
    centers = [(len(text) - 1) * spacing / 2 - i * spacing for i in range(len(text))]
    for i, (digit, cy) in enumerate(zip(text, centers)):
        for segment in _SEGMENTS[digit]:
            # Screen-right corresponds to world -Y, top corresponds to +Z.
            positions = {
                "a": (0, height / 2, width, stroke), "g": (0, 0, width, stroke),
                "d": (0, -height / 2, width, stroke),
                "f": (width / 2, height / 4, stroke, height / 2),
                "b": (-width / 2, height / 4, stroke, height / 2),
                "e": (width / 2, -height / 4, stroke, height / 2),
                "c": (-width / 2, -height / 4, stroke, height / 2),
            }
            dy, dz, sy, sz = positions[segment]
            _cube(stage, f"{parent}/Digit{i}_{segment}", (x, y + cy + dy, z + dz), (0.00015, sy, sz), material)


def _label(stage, path, center, width, height, text, name):
    """Locally generated typography on a small plane, independent of simulation."""
    from PIL import Image, ImageDraw, ImageFont

    directory = ROOT / "assets/generated/labels"
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"{name}.png"
    im = Image.new("RGB", (1024, 160), (18, 23, 29))
    draw = ImageDraw.Draw(im)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 64) if Path(font_path).exists() else ImageFont.load_default()
    draw.text((512, 80), text, font=font, anchor="mm", fill=(225, 234, 240))
    im.save(output)
    material = _texture_material(stage, name, output, 0.65)
    mesh = UsdGeom.Mesh.Define(stage, path)
    x, y, z = center
    mesh.CreatePointsAttr([(x, y + width / 2, z - height / 2), (x, y - width / 2, z - height / 2), (x, y - width / 2, z + height / 2), (x, y + width / 2, z + height / 2)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex").Set([(0, 0), (1, 0), (1, 1), (0, 1)])
    _bind(mesh.GetPrim(), material)


def _camera(stage, path, eye, target, focal_length=35):
    camera = UsdGeom.Camera.Define(stage, path)
    matrix = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse()
    camera.AddTransformOp().Set(matrix)
    camera.CreateFocalLengthAttr(focal_length)
    camera.CreateHorizontalApertureAttr(36)
    camera.CreateVerticalApertureAttr(20.25)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100))
    return path


def _button_feedback_border(stage, body_path, material):
    """Add a 2 mm light border to the moving cap without changing its collider.

    Four nonoverlapping strips leave a 22 mm dark square around the legends.
    Their front faces sit just ahead of the cap surface to avoid z-fighting.
    """
    path = f"{body_path}/FeedbackBorder"
    UsdGeom.Xform.Define(stage, path)
    outer, width = 0.026, BUTTON_BORDER_WIDTH_M
    inner = outer - 2 * width
    offset = (outer - width) / 2
    strips = (
        ("Top", (0, offset), (outer, width)),
        ("Bottom", (0, -offset), (outer, width)),
        ("Left", (offset, 0), (width, inner)),
        ("Right", (-offset, 0), (width, inner)),
    )
    for name, (y, z), (sy, sz) in strips:
        _cube(stage, f"{path}/{name}", (-0.00315, y, z), (0.00015, sy, sz), material)
    return path


def set_button_light(stage, button_info, on=True):
    """Update the feedback shader from measured contact/travel.

    New scenes bind this shader only to the border. Explicit legacy snapshots
    retain their original full-face binding and remain readable by this API.
    """
    shader = UsdShade.Shader.Get(stage, button_info["shader_path"])
    shader.GetInput("diffuseColor").Set(Gf.Vec3f(*(BUTTON_ON_COLOR if on else BUTTON_OFF_COLOR)))
    shader.GetInput("emissiveColor").Set(Gf.Vec3f(*(BUTTON_ON_EMISSION if on else (0, 0, 0))))
    stage.GetPrimAtPath(button_info["body_path"]).GetAttribute("pressb:illuminated").Set(bool(on))


def build_scene(stage, cfg=None):
    """Build the environment, return physical button paths and asset provenance.

    ``center`` is the unpressed *front surface*, while ``body_center`` and
    ``rest_x`` locate the rigid body. No button animation or trigger is faked:
    each cap has a free, spring-loaded prismatic joint constrained to 0–4 mm.
    """
    cfg = cfg or {}
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Xform.Define(stage, "/World/Environment")
    UsdGeom.Scope.Define(stage, "/World/Looks")
    height = float(cfg.get("table_height", 0.76))
    face_x = float(cfg.get("button_face_x", 0.46))
    bottom = float(cfg.get("button_bottom_z", height + 0.16))
    pitch = float(cfg.get("button_pitch_z", cfg.get("button_pitch", 0.04)))
    column = float(cfg.get("button_column_y", cfg.get("column_y", 0.045)))
    panel_offset_y = float(cfg.get("panel_offset_y_m", 0.0))
    panel_offset_x = float(cfg.get("panel_offset_x_m", 0.0))
    if not all(math.isfinite(value) for value in (panel_offset_x, panel_offset_y)):
        raise ValueError("panel offsets must be finite")
    travel = float(cfg.get("button_travel", 0.004))
    wall_x = float(cfg.get("wall_x", face_x + 0.06))
    mid_z = bottom + 2.5 * pitch
    steel = _material(stage, "BrushedSteel", (0.42, 0.47, 0.50), metallic=0.82, roughness=0.32)
    frame = _material(stage, "SatinNickel", (0.70, 0.73, 0.75), metallic=0.80, roughness=0.23)
    charcoal = _material(stage, "Charcoal", (0.035, 0.045, 0.055), metallic=0.15, roughness=0.5)
    floor = _material(stage, "StoneFloor", (0.24, 0.27, 0.29), roughness=0.78)
    grout = _material(stage, "Grout", (0.09, 0.11, 0.13), roughness=0.9)
    digits = _material(stage, "ButtonLegends", (0.88, 0.91, 0.94), roughness=0.44)
    button_face = _material(stage, "ButtonFace", BUTTON_OFF_COLOR, metallic=0.15, roughness=0.45)
    light_mat = _material(stage, "WarmLight", (1, 0.86, 0.64), emission=(3, 2.5, 1.7))
    texture = ROOT / "assets/isaac/Environments/Simple_Room/Materials/Textures/DefaultMaterial_Base_Color.png"
    table_mat = _texture_material(stage, "OfficialTableMaterial", texture) if texture.is_file() else _material(stage, "TableSurface", (0.52, 0.52, 0.52))
    assets = {"table": _table(stage, cfg, table_mat)}

    _cube(stage, "/World/Environment/Floor", (-0.50, 0, -0.04), (3.1, 2.4, 0.08), floor, True)
    for i in range(6):
        _cube(stage, f"/World/Environment/FloorJointX{i}", (-1.9 + i * 0.5, 0, 0.0001), (0.002, 2.4, 0.0002), grout)
    for i in range(5):
        _cube(stage, f"/World/Environment/FloorJointY{i}", (-0.50, -1 + i * 0.5, 0.0001), (3.1, 0.002, 0.0002), grout)
    _cube(stage, "/World/Environment/Wall", (wall_x + 0.025, 0, 1.2), (0.05, 2.4, 2.4), steel, True)
    for i, y in enumerate([-0.8, -0.4, 0.4, 0.8]):
        _cube(stage, f"/World/Environment/CladdingSeam{i}", (wall_x - 0.0005, y, 1.2), (0.001, 0.002, 2.4), charcoal)
    _cube(stage, "/World/Environment/Baseboard", (wall_x - 0.008, 0, 0.055), (0.016, 2.4, 0.11), frame)
    _cube(stage, "/World/Environment/SideWall", (-0.46, 1.20, 1.2), (2.01, 0.05, 2.4), steel, True)
    _cube(stage, "/World/Environment/CornerTrim", (wall_x - 0.008, 1.173, 1.2), (0.025, 0.025, 2.4), frame)
    # Rail stays outside the robot's workspace, below the panel.
    _cube(stage, "/World/Environment/Handrail", (wall_x - 0.065, 0, 0.64), (0.028, 1.8, 0.035), frame, True)
    for i, y in enumerate([-0.72, 0.72]):
        _cube(stage, f"/World/Environment/RailMount{i}", (wall_x - 0.028, y, 0.64), (0.075, 0.025, 0.045), charcoal)
    _cube(stage, "/World/Environment/LightStrip", (wall_x - 0.018, 0, 2.30), (0.018, 2.16, 0.023), light_mat)
    set_back_wall_offset(stage, "/World/Environment", panel_offset_x)

    panel_path = "/World/Panel"
    panel = UsdGeom.Xform.Define(stage, panel_path)
    panel.AddTranslateOp(opSuffix="panelOffset").Set(Gf.Vec3d(panel_offset_x, panel_offset_y, 0.))
    panel.GetPrim().CreateAttribute("pressb:panelOffsetYM", Sdf.ValueTypeNames.Double).Set(panel_offset_y)
    panel.GetPrim().CreateAttribute("pressb:panelOffsetXM", Sdf.ValueTypeNames.Double).Set(panel_offset_x)
    panel_height = 5 * pitch + 0.14
    _cube(stage, f"{panel_path}/Frame", (face_x + 0.033, 0, mid_z), (0.032, 0.20, panel_height), frame, True)
    _cube(stage, f"{panel_path}/Faceplate", (face_x + 0.015, 0, mid_z), (0.004, 0.19, panel_height - 0.010), charcoal)
    _label(stage, f"{panel_path}/Header", (face_x + 0.0128, 0, bottom + 5 * pitch + 0.043), 0.154, 0.023, "FLOOR SELECT", "floor_select")
    _label(stage, f"{panel_path}/Footer", (face_x + 0.0128, 0, bottom - 0.044), 0.146, 0.015, "24 — 35   /   PIPER", "floor_range")
    for i, (y, z) in enumerate([(y, z) for y in [-0.083, 0.083] for z in [mid_z - panel_height / 2 + 0.013, mid_z + panel_height / 2 - 0.013]]):
        screw = UsdGeom.Cylinder.Define(stage, f"{panel_path}/Screw{i}")
        screw.CreateAxisAttr("X")
        screw.CreateRadiusAttr(0.0025)
        screw.CreateHeightAttr(0.001)
        screw.AddTranslateOp().Set(Gf.Vec3d(face_x + 0.0125, y, z))
        _bind(screw.GetPrim(), frame)
        _cube(stage, f"{panel_path}/ScrewSlot{i}", (face_x + 0.0119, y, z), (0.0002, 0.0033, 0.00045), charcoal)

    buttons = {}
    for col, start in [(column, 24), (-column, 30)]:
        for row in range(6):
            number = start + row
            z = bottom + row * pitch
            name = f"Floor{number}"
            root = f"{panel_path}/{name}"
            UsdGeom.Xform.Define(stage, root)
            # Each floor owns its border shader. The dark face and white
            # legends use constant materials so illumination never hides text.
            material = _material(stage, f"Button{number}", BUTTON_OFF_COLOR, metallic=0.15, roughness=0.30)
            _cube(stage, f"{root}/Bezel", (face_x + 0.0065, col, z), (0.004, 0.032, 0.032), frame)
            # Collider lives below an unscaled body Xform, so the joint anchor is
            # in metres and decorative labels inherit only rigid motion.
            rest_x = face_x + 0.003
            body = UsdGeom.Xform.Define(stage, f"{root}/Cap")
            body.AddTranslateOp().Set(Gf.Vec3d(rest_x, col, z))
            prim = body.GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(prim)
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(0.015)
            if PhysxSchema:
                api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                api.CreateDisableGravityAttr(True)
                api.CreateEnableCCDAttr(True)
                api.CreateSolverPositionIterationCountAttr(16)
                api.CreateSolverVelocityIterationCountAttr(4)
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
            cap = _cube(stage, f"{root}/Cap/Collision", (0, 0, 0), (0.006, 0.026, 0.026), button_face, True)
            prim.CreateAttribute("pressb:floor", Sdf.ValueTypeNames.Int).Set(number)
            prim.CreateAttribute("pressb:illuminated", Sdf.ValueTypeNames.Bool).Set(False)
            prim.CreateAttribute("pressb:feedbackStyle", Sdf.ValueTypeNames.Token).Set(BUTTON_FEEDBACK_STYLE)
            prim.CreateAttribute("pressb:borderWidth", Sdf.ValueTypeNames.Float).Set(BUTTON_BORDER_WIDTH_M)
            _number(stage, f"{root}/Cap", number, -0.00313, 0, 0.001, digits)
            feedback_path = _button_feedback_border(stage, str(prim.GetPath()), material)
            joint = UsdPhysics.PrismaticJoint.Define(stage, f"{root}/SpringJoint")
            joint.CreateBody1Rel().SetTargets([prim.GetPath()])
            joint.CreateAxisAttr("X")
            # Body 0 is the world, so this anchor does not inherit Panel's
            # transform even though the joint prim is nested below Panel.
            joint.CreateLocalPos0Attr(Gf.Vec3f(rest_x + panel_offset_x, col + panel_offset_y, z))
            joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
            joint.CreateLowerLimitAttr(0.0)
            joint.CreateUpperLimitAttr(travel)
            joint.CreateCollisionEnabledAttr(False)
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
            drive.CreateTypeAttr("force")
            drive.CreateTargetPositionAttr(0.0)
            drive.CreateTargetVelocityAttr(0.0)
            drive.CreateStiffnessAttr(float(cfg.get("button_stiffness", 300.0)))
            drive.CreateDampingAttr(float(cfg.get("button_damping", 3.0)))
            drive.CreateMaxForceAttr(5.0)
            buttons[number] = {
                "body_path": str(prim.GetPath()), "collision_path": str(cap.GetPath()),
                "joint_path": str(joint.GetPath()), "material_path": str(material.GetPath()),
                "shader_path": f"{material.GetPath()}/Shader", "rest_x": rest_x + panel_offset_x,
                "feedback_style": BUTTON_FEEDBACK_STYLE, "feedback_path": feedback_path,
                "feedback_geometry_paths": [f"{feedback_path}/{side}" for side in BUTTON_BORDER_SIDES],
                "border_width_m": BUTTON_BORDER_WIDTH_M,
                "face_material_path": str(button_face.GetPath()),
                "center": [face_x + panel_offset_x, col + panel_offset_y, z],
                "body_center": [rest_x + panel_offset_x, col + panel_offset_y, z],
                "travel": travel, "floor": number, "row": row,
                "column": "left" if col > 0 else "right",
            }

    lighting_scale = float(cfg.get("lighting_intensity_scale", 1.0))
    assets["lighting"] = {
        "intensity_scale": lighting_scale,
        "base_intensities": {"Ambient": 650, "Ceiling": 1900, "Softbox": 1100},
        "authored_intensities": {
            "Ambient": 650 * lighting_scale,
            "Ceiling": 1900 * lighting_scale,
            "Softbox": 1100 * lighting_scale,
        },
    }
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Ambient")
    dome.CreateIntensityAttr(650 * lighting_scale)
    dome.CreateColorAttr(Gf.Vec3f(0.88, 0.92, 1.0))
    for name, eye, target, intensity, color, size in [
        ("Ceiling", (-0.10, 0, 2.05), (-0.05, 0, 0.85), 1900, (1.0, 0.88, 0.72), (0.7, 1.2)),
        ("Softbox", (-1.1, -0.7, 1.7), (0.30, 0, 1.0), 1100, (0.82, 0.89, 1.0), (0.8, 0.8)),
    ]:
        light = UsdLux.RectLight.Define(stage, f"/World/Lights/{name}")
        light.CreateIntensityAttr(intensity * lighting_scale)
        light.CreateColorAttr(Gf.Vec3f(*color))
        light.CreateWidthAttr(size[0])
        light.CreateHeightAttr(size[1])
        up = Gf.Vec3d(0, 1, 0) if name == "Ceiling" else Gf.Vec3d(0, 0, 1)
        light.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), up).GetInverse())
    camera = _camera(stage, "/World/Camera", cfg.get("camera_eye", [-0.95, -1.0, 1.36]), cfg.get("camera_target", [0.14, 0, 1.02]), 35)
    # Offset beside the arm so the close-up is not occluded by its shoulder.
    panel_camera = _camera(stage, "/World/PanelCamera", [face_x - 0.40, -0.28, mid_z + 0.03], [face_x, 0, mid_z], 46)
    # A true side elevation makes the canonical folded zero pose and tabletop
    # height easy to judge, without perspective shortening of the arm sections.
    profile_x = float(cfg.get("robot_base_x", 0.0)) - 0.10
    profile_z = height + 0.235
    home_camera = _camera(stage, "/World/HomeSideCamera", [profile_x, -1.8, profile_z], [profile_x, 0, profile_z], 35)
    profile = UsdGeom.Camera.Get(stage, home_camera)
    profile.CreateProjectionAttr("orthographic")
    # USD camera aperture uses tenths of a world unit: 12 -> 1.2 metres here.
    profile.CreateHorizontalApertureAttr(12.0)
    profile.CreateVerticalApertureAttr(6.75)
    return {"buttons": buttons, "camera_path": camera, "panel_camera_path": panel_camera, "home_camera_path": home_camera, "assets": assets, "table_height": height, "button_face_x": face_x}
