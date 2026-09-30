#!/usr/bin/env python3
"""Audit official wrist hardware in exported USD without starting Isaac Sim.

Example: PYTHONPATH=.cache/usd-inspect .conda/envs/pressb/bin/python \
scripts/audit_mount_asset.py outputs/latest --urdf PATH_TO_OFFICIAL_URDF

Source checks use assets/sources.lock.json. Installation transforms are derived
from the supplied URDF fixed joints and visual origins; no pose is assumed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "vendor/piper_isaac_sim/piper_description/urdf/piper_description_v100_realsense_camera_v2/configuration/piper_description_v100_realsense_camera_v2_base.usd"
URDF = ROOT / "vendor/piper_isaac_sim/piper_description/urdf/piper_description_v100_realsense_camera_v2.urdf"
COMPATIBILITY_ASSET = ROOT / "assets/generated/official_wrist/assembly.usda"
LEGACY_NAMES = {"Foot", "Riser", "Cradle", "M3Bolt0", "M3Bolt1"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_transform(prim, parent) -> np.ndarray:
    """Return a column-vector homogeneous transform, including authored scale."""
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    return np.asarray(cache.GetLocalToWorldTransform(prim) *
                      cache.GetLocalToWorldTransform(parent).GetInverse(), dtype=float).T


def pose_matrix(position, quaternion) -> np.ndarray:
    xyz, q = np.asarray(position, dtype=float), np.asarray(quaternion, dtype=float)
    if xyz.shape != (3,) or q.shape != (4,) or not np.isfinite(np.r_[xyz, q]).all():
        raise ValueError("Pose requires finite xyz and wxyz quaternion")
    if abs(np.linalg.norm(q) - 1.) > 1e-5:
        raise ValueError("Pose quaternion is not normalized")
    w, x, y, z = q / np.linalg.norm(q)
    result = np.eye(4)
    result[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                      [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                      [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    result[:3, 3] = xyz
    return result


def origin_matrix(element) -> np.ndarray:
    result = np.eye(4)
    if element is None:
        return result
    xyz = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
    rpy = np.fromstring(element.get("rpy", "0 0 0"), sep=" ")
    if xyz.shape != (3,) or rpy.shape != (3,) or not np.isfinite(np.r_[xyz, rpy]).all():
        raise ValueError("Invalid URDF origin")
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    result[:3, :3] = [[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                      [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]]
    result[:3, 3] = xyz
    return result


def urdf_visual_transform(path: Path, root_link: str, target_link: str) -> np.ndarray:
    robot = ET.parse(path).getroot()
    parents = {joint.find("child").get("link"): joint for joint in robot.findall("joint")}
    chain, visited, name = [], set(), target_link
    while name != root_link:
        if name in visited or name not in parents:
            raise ValueError(f"No fixed URDF chain from {root_link} to {target_link}")
        visited.add(name)
        joint = parents[name]
        if joint.get("type") != "fixed":
            raise ValueError(f"URDF mount chain contains movable joint {joint.get('name')}")
        chain.append(origin_matrix(joint.find("origin")))
        name = joint.find("parent").get("link")
    result = np.eye(4)
    for transform in reversed(chain):
        result = result @ transform
    link = next((link for link in robot.findall("link") if link.get("name") == target_link), None)
    if link is None or len(link.findall("visual")) != 1:
        raise ValueError(f"Expected one visual on official URDF link {target_link}")
    return result @ origin_matrix(link.find("visual/origin"))


def source_record(path: Path, lock: dict) -> dict:
    matches = [entry for entry in lock.get("files", [])
               if (ROOT / entry.get("local_path", "")).resolve() == path.resolve()]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one source-lock entry for {path}")
    return matches[0]


def verify_source(path: Path, lock: dict) -> dict:
    record = source_record(path, lock)
    actual = sha256(path)
    url = record.get("source_url", "")
    pinned = bool(re.match(r"https://raw\.githubusercontent\.com/agilexrobotics/piper_isaac_sim/[0-9a-f]{40}/", url))
    return {"path": str(path), "sha256": actual, "locked_sha256": record.get("sha256"),
            "source_url": url, "hash_matches": actual == record.get("sha256"), "official_pinned_source": pinned}


def descendants(root):
    return list(Usd.PrimRange(root, Usd.TraverseInstanceProxies()))


def mesh_map(root) -> dict:
    return {str(prim.GetPath().MakeRelativePath(root.GetPath())): prim
            for prim in descendants(root) if prim.IsA(UsdGeom.Mesh)}


def compare_meshes(source_root, actual_root) -> dict:
    source, actual = mesh_map(source_root), mesh_map(actual_root)
    checks = {"mesh_paths_identical": bool(source and set(source) == set(actual))}
    meshes = []
    for name, expected_prim in source.items():
        if name not in actual:
            continue
        expected, observed = UsdGeom.Mesh(expected_prim), UsdGeom.Mesh(actual[name])
        points = np.asarray(observed.GetPointsAttr().Get())
        counts = np.asarray(observed.GetFaceVertexCountsAttr().Get())
        indices = np.asarray(observed.GetFaceVertexIndicesAttr().Get())
        checks[f"{name}:nonempty_valid_mesh"] = bool(
            points.ndim == 2 and points.shape[0] > 0 and points.shape[1] == 3
            and np.isfinite(points).all() and counts.ndim == 1 and counts.size > 0
            and indices.ndim == 1 and indices.size > 0
            and np.issubdtype(counts.dtype, np.integer) and np.issubdtype(indices.dtype, np.integer)
            and np.all(counts >= 3) and np.sum(counts) == indices.size
            and np.min(indices) >= 0 and np.max(indices) < len(points)
        )
        arrays = {}
        for field, getter in (("points", "GetPointsAttr"), ("face_vertex_counts", "GetFaceVertexCountsAttr"),
                              ("face_vertex_indices", "GetFaceVertexIndicesAttr"), ("normals", "GetNormalsAttr")):
            expected_attr, actual_attr = getattr(expected, getter)(), getattr(observed, getter)()
            a, b = expected_attr.Get(), actual_attr.Get()
            equal = a is None and b is None if a is None or b is None else np.array_equal(np.asarray(a), np.asarray(b))
            checks[f"{name}:{field}_identical"] = bool(equal and not actual_attr.GetTimeSamples())
            arrays[field] = {"count": len(a) if a is not None else 0,
                             "source_sha256": hashlib.sha256(np.asarray(a).tobytes()).hexdigest() if a is not None else None}
        checks[f"{name}:normal_interpolation_identical"] = expected.GetNormalsInterpolation() == observed.GetNormalsInterpolation()
        checks[f"{name}:internal_transform_identical"] = bool(np.allclose(
            relative_transform(expected_prim, source_root), relative_transform(actual[name], actual_root), rtol=0, atol=1e-9))
        checks[f"{name}:visible"] = UsdGeom.Imageable(actual[name]).ComputeVisibility() != UsdGeom.Tokens.invisible
        meshes.append({"relative_path": name, "arrays": arrays})
    return {"checks": checks, "meshes": meshes}


def verify_compatibility_asset(path: Path, source_stage, source_sha256: str, source_roots: list[str]) -> dict:
    """Accept only the declared compatibility layer, with independently checked arrays."""
    provenance_path = path.parent / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    source_path = Path(source_stage.GetRootLayer().realPath).resolve()
    checks = {
        "approved_compatibility_path": path.resolve() == COMPATIBILITY_ASSET.resolve(),
        "provenance_source_path": (ROOT / provenance["source_usd"]).resolve() == source_path,
        "provenance_source_sha256": provenance["source_sha256"] == source_sha256,
        "provenance_output_path": (ROOT / provenance["output_usd"]).resolve() == path.resolve(),
        "provenance_output_sha256": provenance["output_sha256"] == sha256(path),
    }
    runtime_stage = Usd.Stage.Open(str(path))
    if not runtime_stage:
        raise ValueError("Could not open compatibility USD")
    mappings = provenance["mesh_mappings"]
    expected_meshes = {str(prim.GetPath()) for root in source_roots
                       for prim in mesh_map(source_stage.GetPrimAtPath(root)).values()}
    checks["provenance_maps_every_original_mesh"] = bool(
        len(mappings) == len(expected_meshes) and expected_meshes
        and {record["source_prim"] for record in mappings} == expected_meshes
        and all(record["runtime_prim"] == record["source_prim"] for record in mappings)
    )
    checks["compatibility_contains_only_original_meshes"] = {
        str(prim.GetPath()) for prim in runtime_stage.TraverseAll() if prim.IsA(UsdGeom.Mesh)
    } == expected_meshes
    for root in source_roots:
        source_root, actual_root = source_stage.GetPrimAtPath(root), runtime_stage.GetPrimAtPath(root)
        if not source_root or not actual_root:
            raise ValueError(f"Compatibility layer is missing official root {root}")
        compared = compare_meshes(source_root, actual_root)
        checks.update({f"{root}:{name}": value for name, value in compared["checks"].items()})
    for mapping in mappings:
        original = source_stage.GetPrimAtPath(mapping["source_prim"])
        copied = runtime_stage.GetPrimAtPath(mapping["runtime_prim"])
        if not original or not copied:
            raise ValueError("Compatibility provenance references missing mesh")
        array_hashes = mapping["array_hashes"]
        required = {name for name in ("points", "faceVertexCounts", "faceVertexIndices", "normals")
                    if original.GetAttribute(name).Get() is not None}
        checks[f"{mapping['source_prim']}:provenance_records_geometry_arrays"] = required <= set(array_hashes)
        for name, record in array_hashes.items():
            source_array = np.ascontiguousarray(original.GetAttribute(name).Get())
            actual_array = np.ascontiguousarray(copied.GetAttribute(name).Get())
            checks[f"{mapping['source_prim']}:{name}:provenance_array_hash"] = bool(
                source_array.dtype != object and actual_array.dtype == source_array.dtype
                and list(source_array.shape) == record["shape"] and np.dtype(record["dtype"]) == source_array.dtype
                and hashlib.sha256(source_array.tobytes()).hexdigest() == record["sha256"]
                and np.array_equal(source_array, actual_array)
            )
    return {"path": str(path), "sha256": sha256(path), "provenance_path": str(provenance_path),
            "provenance_sha256": sha256(provenance_path), "checks": checks, "success": all(checks.values()),
            "failed_checks": [name for name, passed in checks.items() if not passed]}


def owning_body(prim) -> str | None:
    while prim and not prim.IsPseudoRoot():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get() is not False:
            return str(prim.GetPath())
        prim = prim.GetParent()
    return None


def audit_stage(path: Path, source_stage, args, metadata: dict, expected_transforms: dict) -> dict:
    stage = Usd.Stage.Open(str(path))
    if not stage:
        raise ValueError(f"Could not open stage {path}")
    checks, details = {}, {}
    link = stage.GetPrimAtPath(args.link6_prim)
    if not link:
        raise ValueError(f"Missing wrist body {args.link6_prim}")
    checks["link6_is_enabled_rigid_body"] = owning_body(link) == args.link6_prim
    for label, source_path, target_path, prefix in (
        ("stand", args.stand_source_prim, args.stand_prim, args.mount_metadata_prefix),
        ("camera", args.camera_source_prim, args.camera_prim, args.camera_metadata_prefix),
    ):
        source_root, target_root = source_stage.GetPrimAtPath(source_path), stage.GetPrimAtPath(target_path)
        if not source_root or not target_root:
            raise ValueError(f"Missing {label} source or installed reference root")
        references = target_root.GetMetadata("references")
        referenced = list(references.ApplyOperations([])) if references else []
        reference_filename = Path(getattr(args, "runtime_reference_asset", source_stage.GetRootLayer().realPath)).resolve()
        checks[f"{label}:verified_source_reference"] = any(
            str(reference.primPath) == source_path
            and (path.parent / reference.assetPath).resolve() == reference_filename for reference in referenced)
        compared = compare_meshes(source_root, target_root)
        checks.update({f"{label}:{key}": value for key, value in compared["checks"].items()})
        # The camera optical frame and housing collider are siblings of the
        # referenced visual mesh; all must remain part of the same wrist body.
        prims = descendants(target_root if label == "stand" else target_root.GetParent())
        checks[f"{label}:belongs_to_link6"] = all(owning_body(prim) == args.link6_prim for prim in prims)
        checks[f"{label}:no_independent_rigid_body"] = all(not prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in prims)
        actual_transform = relative_transform(target_root, link)
        checks[f"{label}:urdf_installation_transform"] = label in expected_transforms and bool(np.allclose(
            actual_transform, expected_transforms.get(label), rtol=0, atol=args.transform_tolerance))
        position_key, quaternion_key = prefix + "_position_link6", prefix + "_quaternion_wxyz_link6"
        meta_transform = pose_matrix(metadata[position_key], metadata[quaternion_key]) if position_key in metadata and quaternion_key in metadata else None
        checks[f"{label}:metadata_matches_installation"] = meta_transform is not None and bool(np.allclose(
            actual_transform, meta_transform, rtol=0, atol=args.transform_tolerance))
        colliders = [prim for prim in prims if prim.HasAPI(UsdPhysics.CollisionAPI)]
        checks[f"{label}:collision_enabled"] = bool(colliders) and all(
                UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False for prim in colliders)
        if label == "stand":
            checks["stand:convex_decomposition"] = bool(colliders) and all(
                prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.MeshCollisionAPI)
                and UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == "convexDecomposition" for prim in colliders)
        details[label] = {"source_prim": source_path, "installed_prim": target_path,
                          "transform_link6_column_matrix": actual_transform.tolist(),
                          "expected_urdf_column_matrix": expected_transforms[label].tolist() if label in expected_transforms else None,
                          "colliders": [str(prim.GetPath()) for prim in colliders], "meshes": compared["meshes"]}
    mount_parent = str(stage.GetPrimAtPath(args.stand_prim).GetParent().GetPath())
    legacy = [str(prim.GetPath()) for prim in stage.TraverseAll()
              if str(prim.GetPath()).startswith(args.link6_prim + "/") and prim.GetName() in LEGACY_NAMES]
    authored_legacy = []
    def inspect_spec(spec):
        if str(spec.path).startswith(args.link6_prim + "/") and spec.name in LEGACY_NAMES:
            authored_legacy.append(str(spec.path))
        for child in spec.nameChildren.values():
            inspect_spec(child)
    for prim_spec in stage.GetRootLayer().rootPrims:
        inspect_spec(prim_spec)
    checks["legacy_custom_mount_absent"] = not legacy and not authored_legacy
    official_mesh_paths = {str(prim.GetPath()) for prim in mesh_map(stage.GetPrimAtPath(args.stand_prim)).values()}
    extras = [str(prim.GetPath()) for prim in descendants(stage.GetPrimAtPath(mount_parent))
              if prim.IsA(UsdGeom.Gprim) and str(prim.GetPath()) not in official_mesh_paths]
    checks["mount_contains_only_official_geometry"] = not extras
    details.update(legacy_composed_prims=legacy, legacy_authored_specs=authored_legacy, unexpected_mount_geometry=extras)
    return {"path": str(path), "sha256": sha256(path), "checks": checks, "details": details,
            "success": all(checks.values()), "failed_checks": [key for key, value in checks.items() if not value]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--stage", action="append", type=Path, help="Override the default initial and completed stages")
    parser.add_argument("--source-usd", type=Path, default=SOURCE)
    parser.add_argument("--source-lock", type=Path, default=ROOT / "assets/sources.lock.json")
    parser.add_argument("--urdf", type=Path, default=URDF, help="Official, source-locked URDF defining the mount fixed joints")
    parser.add_argument("--registration", type=Path, default=ROOT / "assets/official_wrist_registration.json",
                        help="Independent shared-gripper registration evidence in column-matrix convention")
    parser.add_argument("--stand-link", default="camera_stand_link")
    parser.add_argument("--camera-link", default="camera_link")
    parser.add_argument("--root-link", default="link6")
    parser.add_argument("--link6-prim", default="/World/Piper/link6")
    parser.add_argument("--stand-source-prim", default="/meshes/realsense_mid_stand")
    parser.add_argument("--camera-source-prim", default="/meshes/d435")
    parser.add_argument("--stand-prim", default="/World/Piper/link6/CameraMount/OfficialStand")
    parser.add_argument("--camera-prim", default="/World/Piper/link6/WristRealSense/D435")
    parser.add_argument("--mount-metadata-prefix", default="mount")
    parser.add_argument("--camera-metadata-prefix", default="housing")
    parser.add_argument("--metadata", type=Path, help="Camera report or sensor metadata JSON; defaults to episode wrist_camera/camera_report.json")
    parser.add_argument("--transform-tolerance", type=float, default=1e-6)
    parser.add_argument("--output", type=Path, help="Defaults to episode/mount_asset_audit.json")
    args = parser.parse_args()
    result = {"audited_at": datetime.now(timezone.utc).isoformat(), "success": False,
              "checks": {}, "stages": [], "sources": [], "errors": []}
    try:
        lock = json.loads(args.source_lock.read_text())
        source = verify_source(args.source_usd, lock)
        result["sources"].append(source)
        result["checks"]["source_usd_hash_matches_lock"] = source["hash_matches"]
        result["checks"]["source_usd_is_pinned_official"] = source["official_pinned_source"]
        registration_report = json.loads(args.registration.read_text())
        registration = np.asarray(registration_report["registration_current_link6_from_official_link6"], dtype=float)
        if (registration.shape != (4, 4) or not np.isfinite(registration).all()
                or not np.allclose(registration[3], [0., 0., 0., 1.], atol=1e-9, rtol=0)
                or not np.allclose(registration[:3, :3].T @ registration[:3, :3], np.eye(3), atol=1e-9, rtol=0)
                or abs(np.linalg.det(registration[:3, :3]) - 1.) > 1e-9):
            raise ValueError("Registration must be a finite rigid column-vector transform")
        result["registration"] = {"path": str(args.registration), "sha256": sha256(args.registration),
                                  "column_matrix": registration.tolist()}
        result["checks"]["registration_uses_locked_source"] = (
            registration_report["stand"]["usd_sha256"] == source["sha256"]
            and registration_report["official_commit"] == source["source_url"].split("/")[5])
        expected = {}
        if args.urdf:
            urdf_source = verify_source(args.urdf, lock)
            result["sources"].append(urdf_source)
            result["checks"]["urdf_hash_matches_lock"] = urdf_source["hash_matches"]
            result["checks"]["urdf_is_pinned_official"] = urdf_source["official_pinned_source"]
            expected = {"stand": registration @ urdf_visual_transform(args.urdf, args.root_link, args.stand_link),
                        "camera": registration @ urdf_visual_transform(args.urdf, args.root_link, args.camera_link)}
        result["checks"]["official_urdf_installation_available"] = bool(expected)
        metadata_path = args.metadata or args.episode / "wrist_camera/camera_report.json"
        metadata = json.loads(metadata_path.read_text())
        metadata = metadata.get("sensor", metadata)
        result["checks"]["metadata_source_matches_locked_asset"] = metadata.get("source_sha256") == source["sha256"]
        metadata_registration = np.asarray(metadata.get("official_link6_to_current_link6", []), dtype=float)
        result["checks"]["metadata_registration_matches_evidence"] = bool(
            metadata_registration.shape == (4, 4)
            and np.allclose(metadata_registration, registration, rtol=0, atol=args.transform_tolerance))
        source_stage = Usd.Stage.Open(str(args.source_usd))
        if not source_stage:
            raise ValueError("Could not open official source USD")
        runtime_asset = (ROOT / metadata.get("runtime_asset_usd", str(args.source_usd))).resolve()
        args.runtime_reference_asset = args.source_usd.resolve()
        if runtime_asset != args.source_usd.resolve():
            compatibility = verify_compatibility_asset(runtime_asset, source_stage, source["sha256"],
                                                       [args.stand_source_prim, args.camera_source_prim])
            result["compatibility_asset"] = compatibility
            result["checks"]["compatibility_asset_verified"] = compatibility["success"]
            if compatibility["success"]:
                args.runtime_reference_asset = runtime_asset
        for path in args.stage or [args.episode / "scene.usda", args.episode / "completed_scene.usda"]:
            result["stages"].append(audit_stage(path, source_stage, args, metadata, expected))
        result["checks"]["all_stages_pass"] = bool(result["stages"]) and all(stage["success"] for stage in result["stages"])
    except Exception as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["failed_checks"] = [key for key, value in result["checks"].items() if not value]
    result["success"] = bool(result["checks"] and all(result["checks"].values()) and not result["errors"])
    output = args.output or args.episode / "mount_asset_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{'PASS' if result['success'] else 'FAIL'}: {output}")
    for error in result["errors"]:
        print(error)
    for stage in result["stages"]:
        if stage["failed_checks"]:
            print(f"{stage['path']}: " + ", ".join(stage["failed_checks"]))
    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
