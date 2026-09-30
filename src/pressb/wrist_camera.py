"""Black PiPER finish and the original AgileX RealSense D435 wrist assembly.

The RGB image and aligned depth are RTX outputs. They are not an emulation of
the D435 stereo matching, minimum disparity, exposure or device noise pipeline.
Call after SimulationApp is initialized and the official PiPER USD is loaded.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

try:
    from pxr import PhysxSchema
except ImportError:
    PhysxSchema = None


ROOT = Path(__file__).resolve().parents[2]
ASSEMBLY_COMMIT = "8e1f88fdb7afca49c40e9a0c1c01cc588e86f0d2"
ASSEMBLY_URDF = "piper_description/urdf/piper_description_v100_realsense_camera_v2.urdf"
ASSEMBLY_USD = "piper_description/urdf/piper_description_v100_realsense_camera_v2/configuration/piper_description_v100_realsense_camera_v2_base.usd"
ASSEMBLY_SOURCE = f"https://raw.githubusercontent.com/agilexrobotics/piper_isaac_sim/{ASSEMBLY_COMMIT}/{ASSEMBLY_USD}"
ASSEMBLY_SHA256 = "7561a29c320017aab66ca8a067af48722e6644e47598a2315bde8eb8e9738ec4"


def _material(stage, name, color, metallic=0.0, roughness=0.4):
    """Author both MDL and PreviewSurface so RTX and plain USD agree."""
    mat = UsdShade.Material.Define(stage, "/World/Looks/" + name)
    mdl = UsdShade.Shader.Define(stage, str(mat.GetPath()) + "/Mdl")
    mdl.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    mdl.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    mdl.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    mdl.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    mdl.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(metallic)
    mdl.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(roughness)
    mdl.CreateOutput("out", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput("mdl").ConnectToSource(mdl.ConnectableAPI(), "out")
    preview = UsdShade.Shader.Define(stage, str(mat.GetPath()) + "/Preview")
    preview.CreateIdAttr("UsdPreviewSurface")
    preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    preview.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    preview.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    mat.CreateSurfaceOutput().ConnectToSource(preview.ConnectableAPI(), "surface")
    return mat


def _bind(prim, material, color=None):
    binding = UsdShade.MaterialBindingAPI.Apply(prim)
    binding.UnbindAllBindings()
    binding.Bind(material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    if color is not None and prim.IsA(UsdGeom.Gprim):
        mesh = UsdGeom.Gprim(prim)
        mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        mesh.GetDisplayColorPrimvar().SetInterpolation("constant")


def apply_black_materials(stage, robot_path="/World/Piper"):
    """De-instance visual meshes only and apply a verified dark MDL finish.

    Collision meshes, masses and joint properties are untouched. Explicit
    per-mesh bindings avoid RTX retaining the stock white prototype material.
    """
    black = (0.018, 0.022, 0.028)
    matte = _material(stage, "PiperGraphiteBlack", black, 0.22, 0.37)
    deinstanced, meshes = [], []
    robot = stage.GetPrimAtPath(robot_path)
    if not robot:
        raise ValueError(f"Robot not found: {robot_path}")
    visuals = [p for p in Usd.PrimRange(robot) if p.GetName() == "visuals"]
    for visual in visuals:
        # Native instance children are not traversed until their parent is
        # expanded. Repeat to handle nested instances without editing prototypes.
        while True:
            instances = [p for p in Usd.PrimRange(visual) if p.IsInstance()]
            if not instances:
                break
            for p in instances:
                p.SetInstanceable(False)
                deinstanced.append(str(p.GetPath()))
        _bind(visual, matte)
        for prim in Usd.PrimRange(visual):
            if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Subset):
                _bind(prim, matte, black)
                meshes.append(str(prim.GetPath()))
    return {"material_path": str(matte.GetPath()), "deinstanced_visuals": deinstanced, "bound_visual_meshes": meshes, "color_linear_rgb": list(black)}


def _box(stage, path, center, size, material, collision=False, invisible=False):
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
    if invisible:
        cube.MakeInvisible()
    return cube


def _transform(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    from scipy.spatial.transform import Rotation
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    result[:3, 3] = xyz
    return result


def _origin(element):
    origin = element.find("origin")
    if origin is None:
        return np.eye(4)
    return _transform(np.fromstring(origin.get("xyz", "0 0 0"), sep=" "),
                      np.fromstring(origin.get("rpy", "0 0 0"), sep=" "))


def _set_transform(xform, transform):
    # NumPy uses column vectors; USD/Gf uses row vectors.
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(Gf.Matrix4d(*transform.T.ravel().tolist()))


def _pose_fields(prefix, transform):
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(transform[:3, :3]).as_quat()
    return {prefix + "_position_link6": transform[:3, 3].tolist(),
            prefix + "_quaternion_wxyz_link6": [w, x, y, z]}


def _require_mesh(stage, path):
    mesh = UsdGeom.Mesh.Get(stage, path)
    if not mesh or not mesh.GetPointsAttr().Get() or not mesh.GetFaceVertexCountsAttr().Get() or not mesh.GetFaceVertexIndicesAttr().Get():
        raise RuntimeError(f"Official hardware mesh is missing or empty: {path}")


def _add_payload(link, mount_transform, camera_transform, camera_mass, camera_dimensions, camera_center):
    """Combine estimated payload inertias into the existing wrist rigid body.

    The upstream stand has zero mass; D435 inertia is explicitly marked
    unreliable. Use the watertight stock stand mesh at assumed ABS density
    1050 kg/m^3 and a uniform camera box, preserving the upstream 72 g mass.
    See assets/official_wrist_registration.json for the mesh mass calculation.
    """
    mass_api = UsdPhysics.MassAPI.Apply(link)
    old_mass = float(mass_api.GetMassAttr().Get())
    old_center = np.asarray(mass_api.GetCenterOfMassAttr().Get(), dtype=float)
    old_axes = mass_api.GetPrincipalAxesAttr().Get()
    old_rotation = np.asarray(Gf.Matrix3d(Gf.Rotation(Gf.Quatd(old_axes))), dtype=float).T
    old_inertia = old_rotation @ np.diag(mass_api.GetDiagonalInertiaAttr().Get()) @ old_rotation.T
    mount_mass = 0.03229625733065188
    mount_com = np.array([-0.0030700752381914186, 0.0020752806930360314, 0.009525246142449458])
    mount_inertia = np.array([[2.2108712331272495e-5, 8.631643610023923e-7, 5.531759592246163e-6],
                              [8.631643610023923e-7, 3.2136906058648883e-5, -2.725884824187642e-6],
                              [5.531759592246163e-6, -2.725884824187642e-6, 2.1341785871301453e-5]])
    parts = [(old_mass, old_center, old_inertia)]
    for mass, com, inertia, transform in [
        (mount_mass, mount_com, mount_inertia, mount_transform),
        (camera_mass, camera_center,
         np.diag(camera_mass / 12 * (sum(camera_dimensions**2) - camera_dimensions**2)), camera_transform),
    ]:
        rotation = transform[:3, :3]
        parts.append((mass, transform[:3, 3] + rotation @ com, rotation @ inertia @ rotation.T))
    total_mass = sum(p[0] for p in parts)
    combined_center = sum(mass * center for mass, center, _ in parts) / total_mass
    inertia = np.zeros((3, 3))
    for mass, position, local_inertia in parts:
        displacement = position - combined_center
        inertia += local_inertia + mass * (np.dot(displacement, displacement) * np.eye(3) - np.outer(displacement, displacement))
    diagonal, axes = np.linalg.eigh(inertia)
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1
    quaternion = Gf.Matrix3d(*axes.T.ravel().tolist()).ExtractRotation().GetQuat()
    mass_api.CreateMassAttr(total_mass)
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*combined_center))
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*diagonal))
    mass_api.CreatePrincipalAxesAttr(Gf.Quatf(quaternion))
    link.CreateAttribute("pressb:wristPayloadMass", Sdf.ValueTypeNames.Float).Set(camera_mass + mount_mass)
    return camera_mass + mount_mass


def attach_wrist_camera(stage, link6_path, cfg=None):
    """Reference AgileX's original stand and matching D435, without redrawing.

    Meshes come unchanged from the pinned piper_isaac_sim base USD, through an
    ASCII-only compatibility layer for Isaac 4.5's older USD path parser.
    Installation is the stock URDF fixed chain, with only the measured rigid
    conversion between that model's wrist frame and robot_lab's wrist frame.
    """
    import hashlib
    import json
    import xml.etree.ElementTree as ET
    cfg = cfg or {}
    link = stage.GetPrimAtPath(link6_path)
    if not link or not link.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError(f"Expected a rigid wrist link: {link6_path}")
    asset = ROOT / "vendor/piper_isaac_sim" / ASSEMBLY_USD
    runtime_asset = ROOT / "assets/generated/official_wrist/assembly.usda"
    urdf_path = ROOT / "vendor/piper_isaac_sim" / ASSEMBLY_URDF
    if not asset.is_file() or not urdf_path.is_file():
        raise FileNotFoundError("Missing official wrist assembly; run scripts/fetch_assets.py")
    if hashlib.sha256(asset.read_bytes()).hexdigest() != ASSEMBLY_SHA256:
        raise ValueError("Official wrist assembly checksum mismatch")
    if not runtime_asset.is_file():
        raise FileNotFoundError("Missing compatible official wrist mesh; run scripts/prepare_wrist_asset.py with the isolated USD reader")
    provenance = json.loads((runtime_asset.parent / "provenance.json").read_text())
    runtime_sha256 = hashlib.sha256(runtime_asset.read_bytes()).hexdigest()
    if provenance.get("source_sha256") != ASSEMBLY_SHA256 or provenance.get("output_sha256") != runtime_sha256:
        raise ValueError("Compatible wrist mesh provenance/checksum mismatch")
    source = ET.parse(urdf_path).getroot()
    joints = {element.get("name"): element for element in source.findall("joint")}
    links = {element.get("name"): element for element in source.findall("link")}
    # Align the matching original gripper surfaces, not the different arm zero
    # poses: old +Y -> current -X and old flange sits 4 mm farther along +Z.
    adapter = _transform((0, 0, -0.004), (0, 0, math.pi / 2))
    stand_transform = adapter @ _origin(joints["camera_stand_joint"])
    camera_transform = (adapter @ _origin(joints["d435_camera_joint"])
                        @ _origin(joints["camera_joint"]) @ _origin(joints["camera_link_joint"]))
    housing_transform = camera_transform @ _origin(links["camera_link"].find("visual"))
    black = _material(stage, "OfficialCameraStandBlack", (0.010, 0.014, 0.019), 0.08, 0.36)
    silver = _material(stage, "RealSenseD435Aluminum", (0.38, 0.42, 0.47), 0.80, 0.30)

    mount_path = link6_path + "/CameraMount"
    UsdGeom.Xform.Define(stage, mount_path)
    mount = UsdGeom.Xform.Define(stage, mount_path + "/OfficialStand")
    mount.GetPrim().GetReferences().AddReference(str(runtime_asset), "/meshes/realsense_mid_stand")
    _require_mesh(stage, str(mount.GetPath()) + "/Scene/mesh")
    _set_transform(mount, stand_transform)
    _bind(mount.GetPrim(), black)
    for prim in Usd.PrimRange(mount.GetPrim()):
        if prim.IsA(UsdGeom.Mesh):
            _bind(prim, black)
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexDecomposition")
            if PhysxSchema:
                collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                collision.CreateContactOffsetAttr(0.0005)
                collision.CreateRestOffsetAttr(0.0)
                decomposition = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
                decomposition.CreateHullVertexLimitAttr(64)
                decomposition.CreateMaxConvexHullsAttr(32)
    mount.GetPrim().CreateAttribute("pressb:sourceMesh", Sdf.ValueTypeNames.String).Set("/meshes/realsense_mid_stand")
    mount.GetPrim().CreateAttribute("pressb:sourceRevision", Sdf.ValueTypeNames.String).Set(ASSEMBLY_COMMIT)

    rig_path = link6_path + "/WristRealSense"
    UsdGeom.Xform.Define(stage, rig_path)
    housing = UsdGeom.Xform.Define(stage, rig_path + "/D435")
    housing.GetPrim().GetReferences().AddReference(str(runtime_asset), "/meshes/d435")
    _require_mesh(stage, str(housing.GetPath()) + "/ID1/mesh")
    _set_transform(housing, housing_transform)
    _bind(housing.GetPrim(), silver)
    camera_link = UsdGeom.Xform.Define(stage, rig_path + "/CameraLink")
    _set_transform(camera_link, camera_transform)
    collision = links["camera_link"].find("collision")
    dimensions = np.fromstring(collision.find("geometry/box").get("size"), sep=" ")
    collision_transform = _origin(collision)
    _box(stage, str(camera_link.GetPath()) + "/HousingCollision", collision_transform[:3, 3], dimensions,
         silver, collision=True, invisible=True)

    camera_path = str(camera_link.GetPath()) + "/ColorCamera"
    camera = UsdGeom.Camera.Define(stage, camera_path)
    # Nominal +15 mm RGB offset from the bundled _d435.urdf.xacro. Map ROS
    # camera_link (+X forward, +Y left, +Z up) to USD camera axes.
    optical_local = np.eye(4)
    optical_local[:3, :3] = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
    optical_local[:3, 3] = [0, 0.015, 0]
    _set_transform(camera, optical_local)
    optical = camera_transform @ optical_local
    width, height = [int(x) for x in cfg.get("wrist_camera_resolution", [640, 480])]
    # D435's nominal full RGB FOV is 69 x 42 degrees. Use a square-pixel
    # centred crop at 4:3; these are simulation intrinsics, not device calibration.
    focal = 1.88
    native_h = 2 * focal * math.tan(math.radians(69 / 2))
    native_v = 2 * focal * math.tan(math.radians(42 / 2))
    horizontal = min(native_h, native_v * width / height)
    vertical = horizontal * height / width
    camera.CreateFocalLengthAttr(focal)
    camera.CreateHorizontalApertureAttr(horizontal)
    camera.CreateVerticalApertureAttr(vertical)
    camera.CreateProjectionAttr("perspective")
    camera.GetPrim().CreateAttribute("cameraProjectionType", Sdf.ValueTypeNames.Token).Set("pinhole")
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.005, 10.0))
    camera.CreateFStopAttr(0.0)
    camera.CreateFocusDistanceAttr(0.30)
    camera_mass = float(links["camera_link"].find("inertial/mass").get("value"))
    payload = _add_payload(link, stand_transform, camera_transform, camera_mass,
                           dimensions, collision_transform[:3, 3])
    return {
        "model": "Intel RealSense D435", "camera_path": camera_path,
        "rig_path": rig_path, "mount_path": mount_path, "link6_path": link6_path,
        "resolution": [width, height],
        "intrinsics": [[focal / horizontal * width, 0, width / 2], [0, focal / vertical * height, height / 2], [0, 0, 1]],
        "focal_length_mm": focal, "horizontal_aperture_mm": horizontal, "vertical_aperture_mm": vertical,
        "field_of_view_deg": [math.degrees(2 * math.atan(horizontal / (2 * focal))), math.degrees(2 * math.atan(vertical / (2 * focal)))],
        **_pose_fields("optical", optical), **_pose_fields("mount", stand_transform),
        **_pose_fields("housing", housing_transform),
        "optical_forward_link6": (-optical[:3, 2]).tolist(),
        "optical_frame_convention": "USD camera: +X right, +Y up, -Z forward",
        "payload_mass_kg": payload,
        "payload_model": "D435 upstream 72 g, uniform-box inertia; original watertight stand estimated at ABS density 1050 kg/m^3 (32.296 g). Upstream stand mass is zero and camera inertia is explicitly unreliable.",
        "fixture_source": "Original AgileX piper_isaac_sim realsense_mid_stand mesh and official URDF fixed joint; no custom bracket geometry",
        "source_url": ASSEMBLY_SOURCE, "source_sha256": ASSEMBLY_SHA256,
        "source_revision": ASSEMBLY_COMMIT, "source_usd": str(asset), "source_urdf": str(urdf_path),
        "runtime_asset_usd": str(runtime_asset),
        "runtime_asset_sha256": runtime_sha256,
        "compatibility_note": "Original mesh arrays copied unchanged to ASCII USD; unsupported Chinese material prim names omitted and replaced by local materials",
        "mount_source_prim": "/meshes/realsense_mid_stand", "housing_source_prim": "/meshes/d435",
        "official_link6_to_current_link6": adapter.tolist(),
        "projection_model": "Ideal rectified pinhole, square-pixel centred crop of nominal D435 RGB FOV 69 x 42 deg; zero distortion",
        "optics_source_url": "https://www.realsenseai.com/products/stereo-depth-camera-d435/",
        "depth_model": "RTX geometric distance_to_image_plane aligned to RGB; not the RealSense stereo or noise pipeline",
    }
