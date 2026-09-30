#!/usr/bin/env python3
"""Extract unchanged official wrist meshes into an Isaac Sim 4.5 compatible USD.

Run with a standalone modern ``usd-core`` Python environment, not Isaac's bundled
USD, because the original asset uses Unicode prim identifiers unsupported by
Isaac Sim 4.5. For example:

    python -m pip install --target /tmp/pressb-usd-tools usd-core==24.11 numpy
    PYTHONPATH=/tmp/pressb-usd-tools python scripts/prepare_wrist_asset.py

Only material bindings are removed. No geometry is regenerated, simplified,
transformed, or triangulated. Original arrays and identity ancestors are checked
after saving the ASCII layer. The original downloaded file remains untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / (
    "vendor/piper_isaac_sim/piper_description/urdf/"
    "piper_description_v100_realsense_camera_v2/configuration/"
    "piper_description_v100_realsense_camera_v2_base.usd"
)
OUTPUT = ROOT / "assets/generated/official_wrist/assembly.usda"
SOURCE_SHA256 = "7561a29c320017aab66ca8a067af48722e6644e47598a2315bde8eb8e9738ec4"
MESH_PATHS = (
    "/meshes/realsense_mid_stand/Scene/mesh",
    "/meshes/d435/ID1/mesh",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def array_hashes(prim: Usd.Prim) -> dict:
    """Hash every authored numeric array, including normals and texture UVs."""
    result = {}
    for attr in prim.GetAuthoredAttributes():
        if not attr.GetTypeName().isArray:
            continue
        value = attr.Get()
        if value is None:
            continue
        array = np.ascontiguousarray(np.asarray(value))
        if array.dtype.kind not in "biufc":
            continue
        result[attr.GetName()] = {
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            "dtype": str(array.dtype),
            "shape": list(array.shape),
        }
    return result


def prepare(source: Path, output: Path) -> dict:
    source_hash = sha256(source)
    if source_hash != SOURCE_SHA256:
        raise RuntimeError(f"Official source checksum mismatch: {source_hash}")
    source_stage = Usd.Stage.Open(str(source))
    if source_stage is None:
        raise RuntimeError("Cannot open source; use a standalone modern usd-core.")

    # Anonymous output avoids editing a previously composed or cached layer.
    stage = Usd.Stage.CreateInMemory("official_wrist_compat.usda")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/meshes").GetPrim())
    mappings = []
    identity = Gf.Matrix4d(1.0)
    for mesh_path in MESH_PATHS:
        prim = source_stage.GetPrimAtPath(mesh_path)
        if not prim or not prim.IsA(UsdGeom.Mesh):
            raise RuntimeError(f"Official mesh missing: {mesh_path}")
        ancestors = []
        parent = prim.GetParent()
        while parent and not parent.IsPseudoRoot():
            if parent.IsA(UsdGeom.Xformable):
                local = UsdGeom.Xformable(parent).GetLocalTransformation()
                if not Gf.IsClose(local, identity, 1e-12):
                    raise RuntimeError(f"Non-identity source ancestor: {parent.GetPath()}")
            ancestors.append(str(parent.GetPath()))
            parent = parent.GetParent()
        for ancestor in reversed(ancestors):
            UsdGeom.Xform.Define(stage, ancestor)
        if not Sdf.CopySpec(
            source_stage.GetRootLayer(), Sdf.Path(mesh_path),
            stage.GetRootLayer(), Sdf.Path(mesh_path),
        ):
            raise RuntimeError(f"Could not copy official mesh: {mesh_path}")
        copied = stage.GetPrimAtPath(mesh_path)
        removed_bindings = {}
        for relation in list(copied.GetRelationships()):
            if relation.GetName().startswith("material:binding"):
                removed_bindings[relation.GetName()] = [str(x) for x in relation.GetTargets()]
                copied.RemoveProperty(relation.GetName())
        copied.ClearMetadata("apiSchemas")
        mappings.append({
            "source_prim": mesh_path,
            "runtime_prim": mesh_path,
            "array_hashes": array_hashes(prim),
            "removed_material_bindings": removed_bindings,
            "identity_ancestors": list(reversed(ancestors)),
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    stage.GetRootLayer().Export(str(output))
    if not output.read_bytes().startswith(b"#usda 1.0"):
        raise RuntimeError("Expected portable ASCII USDA output.")
    runtime_stage = Usd.Stage.Open(str(output))
    for item in mappings:
        mesh = runtime_stage.GetPrimAtPath(item["runtime_prim"])
        if array_hashes(mesh) != item["array_hashes"]:
            raise RuntimeError(f"Saved geometry differs: {item['runtime_prim']}")
        if any(rel.GetTargets() for rel in mesh.GetRelationships()):
            raise RuntimeError(f"Unexpected external relationship: {item['runtime_prim']}")
    # ASCII identifiers only; no copied material subtree or stale external path.
    for prim in runtime_stage.Traverse():
        str(prim.GetPath()).encode("ascii")
    report = {
        "schema_version": 1,
        "purpose": "Isaac Sim 4.5 USD format compatibility; unchanged official geometry",
        "source_repository": "https://github.com/agilexrobotics/piper_isaac_sim",
        "source_commit": "8e1f88fdb7afca49c40e9a0c1c01cc588e86f0d2",
        "source_usd": relative_path(source),
        "source_sha256": source_hash,
        "output_usd": relative_path(output),
        "output_sha256": sha256(output),
        "export_usd_version": list(Usd.GetVersion()),
        "mesh_mappings": mappings,
        "changes": [
            "Copy only the two official Mesh specs and identity Xform ancestors.",
            "Remove bindings and MaterialBindingAPI for omitted original materials.",
            "Serialize as ASCII USDA with ASCII-only prim paths.",
        ],
        "geometry_arrays_bitwise_equal_after_reload": True,
        "geometry_redrawn": False,
    }
    (output.parent / "provenance.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = prepare(args.source, args.output)
    print(json.dumps({
        "output_usd": report["output_usd"],
        "output_sha256": report["output_sha256"],
        "mesh_count": len(report["mesh_mappings"]),
        "geometry_arrays_bitwise_equal_after_reload": True,
    }, indent=2))


if __name__ == "__main__":
    main()
