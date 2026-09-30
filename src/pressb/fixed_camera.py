"""A fixed, left-side D435 on a resized NVIDIA stand beside the arm.

Call after SimulationApp starts. The eye/target describes the RGB optical frame,
not the housing centre. The official bottom screw, housing and optical transforms
are preserved when solving backwards from that requested world optical pose.
"""

from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation

from .wrist_camera import (
    ASSEMBLY_COMMIT, ASSEMBLY_SHA256, ASSEMBLY_SOURCE, ASSEMBLY_URDF, ASSEMBLY_USD,
    ROOT, _bind, _box, _material, _origin, _require_mesh, _set_transform,
)


STAND_RELATIVE = "Props/Mounts/Stand/stand.usd"
STAND_URL = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/" + STAND_RELATIVE
STAND_SHA256 = "9788fa192ae3a28117daa6f99a38375d3d1c9ae844f58236aa0bfaae33b6b8ab"


def _cylinder(stage, path, center, radius, height, material):
    shape = UsdGeom.Cylinder.Define(stage, path)
    shape.CreateAxisAttr("Z")
    shape.CreateRadiusAttr(float(radius))
    shape.CreateHeightAttr(float(height))
    shape.AddTranslateOp().Set(Gf.Vec3d(*center))
    _bind(shape.GetPrim(), material)
    UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
    return shape


def _world_pose(transform):
    x, y, z, w = Rotation.from_matrix(transform[:3, :3]).as_quat()
    return transform[:3, 3].tolist(), [float(w), float(x), float(y), float(z)]


def build_global_camera(stage, cfg=None):
    """Build the static rig and return calibrated metadata for a Camera recorder.

    Config: ``global_camera_eye``, ``global_camera_target``,
    ``global_camera_resolution``, ``global_camera_support_surface`` and
    ``global_camera_stand_footprint_m`` and ``global_camera_head_riser_m``.
    Position vectors use world metres. The
    camera must remain on the +Y (left) side. Images use the same nominal D435
    RGB optics and aligned ideal RTX depth as the moving wrist sensor.
    """
    cfg = cfg or {}
    eye = np.asarray(cfg.get("global_camera_eye", [-.28, .24, 1.05]), dtype=float)
    target = np.asarray(cfg.get("global_camera_target", [.43, 0., 1.0675]), dtype=float)
    support_surface = cfg.get("global_camera_support_surface", "tabletop")
    if support_surface not in ("tabletop", "floor"):
        raise ValueError("global_camera_support_surface must be tabletop or floor")
    support_z = float(cfg.get("table_height", .76)) if support_surface == "tabletop" else 0.
    footprint = float(cfg.get("global_camera_stand_footprint_m", .18))
    head_riser = float(cfg.get("global_camera_head_riser_m", .05))
    if not np.isfinite([support_z, footprint, head_riser]).all() or footprint <= 0 or head_riser < .02:
        raise ValueError("Camera stand dimensions must be finite, footprint positive and head riser at least 20 mm")
    if any(p.shape != (3,) or not np.isfinite(p).all() for p in (eye, target)):
        raise ValueError("global_camera_eye and global_camera_target must be finite XYZ vectors")
    if eye[1] <= 0:
        raise ValueError("The global D435 must be on the left side of the scene (+Y)")
    forward = target - eye
    distance = float(np.linalg.norm(forward))
    if distance < 1e-6:
        raise ValueError("Global camera eye and target must be different")
    forward /= distance
    right = np.cross(forward, [0., 0., 1.])
    if np.linalg.norm(right) < 1e-6:
        raise ValueError("Global camera cannot look vertically with the fixed world-up convention")
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    world_optical = np.eye(4)
    world_optical[:3, :3] = np.column_stack((right, up, -forward))
    world_optical[:3, 3] = eye

    source_usd = ROOT / "vendor/piper_isaac_sim" / ASSEMBLY_USD
    urdf_path = ROOT / "vendor/piper_isaac_sim" / ASSEMBLY_URDF
    runtime_asset = ROOT / "assets/generated/official_wrist/assembly.usda"
    stand_asset = ROOT / "assets/isaac" / STAND_RELATIVE
    for path in (source_usd, urdf_path, stand_asset):
        if not path.is_file():
            raise FileNotFoundError(f"Missing official camera/stand asset: {path}; run scripts/fetch_assets.py")
    if hashlib.sha256(source_usd.read_bytes()).hexdigest() != ASSEMBLY_SHA256:
        raise ValueError("Official D435 source checksum mismatch")
    if hashlib.sha256(stand_asset.read_bytes()).hexdigest() != STAND_SHA256:
        raise ValueError("Official fixed-camera stand checksum mismatch")
    if not runtime_asset.is_file():
        raise FileNotFoundError("Run scripts/prepare_wrist_asset.py with the isolated USD reader first")
    provenance = json.loads((runtime_asset.parent / "provenance.json").read_text())
    runtime_hash = hashlib.sha256(runtime_asset.read_bytes()).hexdigest()
    if provenance.get("source_sha256") != ASSEMBLY_SHA256 or provenance.get("output_sha256") != runtime_hash:
        raise ValueError("Compatible D435 mesh provenance/checksum mismatch")

    source = ET.parse(urdf_path).getroot()
    joints = {j.get("name"): j for j in source.findall("joint")}
    camera_element = source.find("link[@name='camera_link']")
    bottom_to_link = _origin(joints["camera_link_joint"])
    link_to_optical = np.eye(4)
    link_to_optical[:3, :3] = [[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]]
    link_to_optical[:3, 3] = [0., .015, 0.]
    # W<-C = (W<-optical) (C<-optical)^-1; W<-bottom follows the
    # inverse of the official bottom-screw-to-camera-link joint.
    world_link = world_optical @ np.linalg.inv(link_to_optical)
    world_bottom = world_link @ np.linalg.inv(bottom_to_link)
    world_housing = world_link @ _origin(camera_element.find("visual"))

    rig_path = "/World/GlobalCamera"
    root = UsdGeom.Xform.Define(stage, rig_path)
    root.SetResetXformStack(True)
    silver = _material(stage, "GlobalD435Aluminum", (.38, .42, .47), .80, .30)
    dark = _material(stage, "GlobalCameraHead", (.025, .031, .038), .60, .34)
    housing_path = rig_path + "/D435"
    housing = UsdGeom.Xform.Define(stage, housing_path)
    housing.GetPrim().GetReferences().AddReference(str(runtime_asset), "/meshes/d435")
    _require_mesh(stage, housing_path + "/ID1/mesh")
    _set_transform(housing, world_housing)
    _bind(housing.GetPrim(), silver)
    link = UsdGeom.Xform.Define(stage, rig_path + "/CameraLink")
    _set_transform(link, world_link)
    collider = camera_element.find("collision")
    dimensions = np.fromstring(collider.find("geometry/box").get("size"), sep=" ")
    _box(stage, str(link.GetPath()) + "/HousingCollision", _origin(collider)[:3, 3],
         dimensions, silver, collision=True, invisible=True)

    bottom = UsdGeom.Xform.Define(stage, rig_path + "/BottomScrewFrame")
    _set_transform(bottom, world_bottom)
    # A small task-specific gimbal interface is sufficient; the support is
    # the original NVIDIA mesh. This plate's top face coincides with the D435's
    # actual bottom screw plane z=0, not with its optical centre or mesh origin.
    _box(stage, str(bottom.GetPath()) + "/MountingPlate", [0., 0., -.003],
         [.034, .040, .006], dark, collision=True)
    _cylinder(stage, str(bottom.GetPath()) + "/TiltStem", [0., 0., -.026], .009, .040, dark)
    ball_local = np.array([0., 0., -.050])
    ball_world = world_bottom[:3, :3] @ ball_local + world_bottom[:3, 3]
    ball = UsdGeom.Sphere.Define(stage, str(bottom.GetPath()) + "/BallHead")
    ball.CreateRadiusAttr(.018)
    ball.AddTranslateOp().Set(Gf.Vec3d(*ball_local))
    _bind(ball.GetPrim(), dark)
    UsdPhysics.CollisionAPI.Apply(ball.GetPrim())

    # Keep the stock top plate below the lens using a configurable short riser.
    platform_z = float(ball_world[2] - head_riser)
    if platform_z <= support_z + .06:
        raise ValueError("Global camera is too low for its support surface and ball head")
    stand_path = rig_path + "/Stand"
    stand = UsdGeom.Xform.Define(stage, stand_path)
    stand_source = UsdGeom.Xform.Define(stage, stand_path + "/OfficialAsset")
    stand_source.GetPrim().GetReferences().AddReference(str(stand_asset))
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(stand_source.GetPrim()).ComputeAlignedBox()
    lower, upper = np.asarray(bbox.GetMin()), np.asarray(bbox.GetMax())
    if not np.isfinite([lower, upper]).all() or np.any(upper <= lower):
        raise RuntimeError("Official stand has invalid bounds")
    horizontal_scale = footprint / max(upper[:2] - lower[:2])
    scale = np.array([horizontal_scale, horizontal_scale,
                      (platform_z - support_z) / (upper[2] - lower[2])])
    offset = np.array([ball_world[0], ball_world[1], platform_z]) - scale * np.array([0., 0., upper[2]])
    stand_matrix = np.eye(4)
    stand_matrix[:3, :3] = np.diag(scale)
    stand_matrix[:3, 3] = offset
    _set_transform(stand, stand_matrix)
    _cylinder(stage, rig_path + "/HeadRiser",
              [ball_world[0], ball_world[1], platform_z + head_riser / 2], .012, head_riser, dark)
    # The original collider covers the column. Add exact plate proxies because
    # the stock visual includes a wider foot and top platform.
    stand_lower, stand_upper = lower * scale + offset, upper * scale + offset
    table_edge = float(cfg.get("table_edge_x", -.14))
    table_depth = float(cfg.get("table_depth", .75))
    table_width = float(cfg.get("table_width", 1.))
    table_bounds = [[table_edge - table_depth, -table_width / 2, support_z],
                    [table_edge, table_width / 2, support_z]]
    if support_surface == "tabletop" and (
            np.any(stand_lower[:2] < np.asarray(table_bounds[0])[:2])
            or np.any(stand_upper[:2] > np.asarray(table_bounds[1])[:2])):
        raise ValueError("The complete camera-stand foot must remain on the tabletop")
    # Source vertices outside the 120 mm column extend 69.083 mm above the
    # foot. A 70 mm source-height proxy encloses that original wider base.
    foot_height = .070 * scale[2]
    _box(stage, rig_path + "/FootCollision", [ball_world[0], ball_world[1], support_z + foot_height / 2],
         [float((upper[0] - lower[0]) * scale[0]), float((upper[1] - lower[1]) * scale[1]), foot_height],
         dark, collision=True, invisible=True)
    _box(stage, rig_path + "/TopPlateCollision", [ball_world[0], ball_world[1], platform_z - .009 * scale[2]],
         [.18 * scale[0], .18 * scale[1], .018 * scale[2]], dark, collision=True, invisible=True)

    camera_path = str(link.GetPath()) + "/ColorCamera"
    camera = UsdGeom.Camera.Define(stage, camera_path)
    _set_transform(camera, link_to_optical)
    resolution = np.asarray(cfg.get("global_camera_resolution", [640, 480]))
    if resolution.shape != (2,) or not np.isfinite(resolution).all() or np.any(resolution < 1) or np.any(resolution != np.floor(resolution)):
        raise ValueError("global_camera_resolution must contain two positive integer pixel counts")
    width, height = map(int, resolution)
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
    camera.CreateClippingRangeAttr(Gf.Vec2f(.005, 10.))
    camera.CreateFStopAttr(0.)
    camera.CreateFocusDistanceAttr(distance)
    position, quaternion = _world_pose(world_optical)
    return {
        "model": "Intel RealSense D435", "camera_path": camera_path,
        "rig_path": rig_path, "mount_path": stand_path, "housing_path": housing_path,
        "resolution": [width, height],
        "intrinsics": [[focal / horizontal * width, 0., width / 2],
                       [0., focal / vertical * height, height / 2], [0., 0., 1.]],
        "focal_length_mm": focal, "horizontal_aperture_mm": horizontal, "vertical_aperture_mm": vertical,
        "field_of_view_deg": [math.degrees(2 * math.atan(horizontal / (2 * focal))),
                              math.degrees(2 * math.atan(vertical / (2 * focal)))],
        "fixed_position_world": position, "fixed_quaternion_wxyz_world": quaternion,
        "optical_forward_world": forward.tolist(), "look_at_world": target.tolist(),
        "world_optical_transform": world_optical.tolist(),
        "world_camera_link_transform": world_link.tolist(),
        "world_housing_transform": world_housing.tolist(),
        "world_bottom_screw_transform": world_bottom.tolist(),
        "bottom_screw_position_world": world_bottom[:3, 3].tolist(),
        "optical_frame_convention": "USD camera: +X right, +Y up, -Z forward",
        "source_url": ASSEMBLY_SOURCE, "source_sha256": ASSEMBLY_SHA256,
        "source_revision": ASSEMBLY_COMMIT, "source_usd": str(source_usd), "source_urdf": str(urdf_path),
        "runtime_asset_usd": str(runtime_asset), "runtime_asset_sha256": runtime_hash,
        "housing_source_prim": "/meshes/d435",
        "fixture_source": "Original NVIDIA Stand asset resized to the configured footprint and support height; ball-head interface at official D435 bottom screw",
        "stand": {"source_url": STAND_URL, "source_sha256": STAND_SHA256, "local_path": str(stand_asset),
                  "support_surface": support_surface, "support_z": support_z,
                  "footprint_m": footprint, "head_riser_m": head_riser,
                  "optical_height_above_support_m": float(eye[2] - support_z),
                  "support_bounds_world_m": table_bounds if support_surface == "tabletop" else None,
                  "source_bounds_m": [lower.tolist(), upper.tolist()], "scale_xyz": scale.tolist(),
                  "world_bounds_m": [stand_lower.tolist(), stand_upper.tolist()],
                  "platform_position_world": [float(ball_world[0]), float(ball_world[1]), platform_z],
                  "ball_center_world": ball_world.tolist(), "bottom_screw_position_world": world_bottom[:3, 3].tolist(),
                  "static_collision": True},
        "projection_model": "Ideal rectified pinhole, square-pixel centred crop of nominal D435 RGB FOV 69 x 42 deg; zero distortion",
        "optics_source_url": "https://www.realsenseai.com/products/stereo-depth-camera-d435/",
        "depth_model": "RTX geometric distance_to_image_plane aligned to RGB; not the RealSense stereo or noise pipeline",
    }
