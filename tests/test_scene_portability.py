"""A portable scene must retain original opinions and verified dependencies."""
import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("pxr.UsdUtils")
from pxr import Sdf

from pressb.scene_portability import prepare_runtime_snapshot, verify_runtime_snapshot


def identity(path):
    data = path.read_bytes()
    return dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def save_manifest(bundle, manifest):
    (bundle / "asset_bundle.json").write_text(json.dumps(manifest))


@pytest.fixture
def scene(tmp_path):
    original_root = "/old/project"
    bundle = tmp_path / "bundle"
    (bundle / "assets").mkdir(parents=True)
    (bundle / "assets/texture.png").write_bytes(b"immutable image bytes")
    (bundle / "assets/mesh.usda").write_text('''#usda 1.0
def Xform "Mesh" {
    asset texture = @texture.png@
    float3[] points = [(1, 2, 3)]
}
''')
    source = tmp_path / "source.usda"
    source.write_text('''#usda 1.0
def Xform "World" (
    references = @/old/project/assets/mesh.usda@</Mesh>
) {
    string note = "/old/project/assets/mesh.usda"
    asset mdl = @OmniPBR.mdl@
    double3 xformOp:translate = (1, 2, 3)
    uniform token[] xformOpOrder = ["xformOp:translate"]
}
''')
    manifest = dict(schema_version=1, source_project_root=original_root, files=[
        dict(source_path=f"{original_root}/assets/{name}", relative_path=f"assets/{name}",
             **identity(bundle / "assets" / name)) for name in ("mesh.usda", "texture.png")
    ])
    save_manifest(bundle, manifest)
    return source, bundle, tmp_path / "output", manifest


def test_relocation_preserves_strings_geometry_source_and_complete_closure(scene):
    source, bundle, output, manifest = scene
    original = source.read_bytes()
    report = prepare_runtime_snapshot(source, bundle, output)
    runtime = Path(report["runtime_snapshot"])
    text = runtime.read_text()
    assert f'@{bundle}/assets/mesh.usda@' in text
    assert 'string note = "/old/project/assets/mesh.usda"' in text
    assert "@OmniPBR.mdl@" in text
    assert "double3 xformOp:translate = (1, 2, 3)" in text
    assert source.read_bytes() == original
    assert report["source_sha256"] == identity(source)["sha256"]
    assert report["runtime_sha256"] != report["source_sha256"]
    assert len(report["assets"]) == 2  # Includes the nested texture dependency.
    assert len(report["asset_path_map"]) == 1
    assert report["authored_content_preserved"] is True
    assert verify_runtime_snapshot(source, output / "scene_relocation.json") == report
    with pytest.raises(FileExistsError):
        prepare_runtime_snapshot(source, bundle, output)


@pytest.mark.parametrize("relative", ["../outside.usda", "/outside.usda", "assets/../mesh.usda", "assets//mesh.usda"])
def test_manifest_rejects_paths_outside_or_noncanonical(scene, relative):
    source, bundle, output, manifest = scene
    manifest["files"][0]["relative_path"] = relative
    save_manifest(bundle, manifest)
    with pytest.raises(ValueError, match="relative_path"):
        prepare_runtime_snapshot(source, bundle, output)


def test_manifest_rejects_escaped_symlink(scene):
    source, bundle, output, manifest = scene
    texture = bundle / "assets/texture.png"
    outside = bundle.parent / "outside.png"
    texture.rename(outside)
    texture.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        prepare_runtime_snapshot(source, bundle, output)


def test_manifest_file_cannot_escape_bundle(scene):
    source, bundle, output, manifest = scene
    path = bundle / "asset_bundle.json"
    outside = bundle.parent / "outside_manifest.json"
    path.rename(outside)
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="manifest escapes"):
        prepare_runtime_snapshot(source, bundle, output)


def test_output_cannot_overwrite_a_dangling_symlink(scene):
    source, bundle, output, manifest = scene
    output.mkdir()
    destination = output.parent / "must_not_be_created.usda"
    (output / "runtime_scene.usda").symlink_to(destination)
    with pytest.raises(FileExistsError):
        prepare_runtime_snapshot(source, bundle, output)
    assert not destination.exists()


def test_manifest_rejects_duplicate_entries(scene):
    source, bundle, output, manifest = scene
    manifest["files"].append(manifest["files"][0].copy())
    save_manifest(bundle, manifest)
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_runtime_snapshot(source, bundle, output)


@pytest.mark.parametrize("when", ["prepare", "audit"])
def test_changed_dependency_bytes_rejected_even_at_same_size(scene, when):
    source, bundle, output, manifest = scene
    if when == "audit":
        prepare_runtime_snapshot(source, bundle, output)
    texture = bundle / "assets/texture.png"
    data = texture.read_bytes()
    texture.write_bytes(b"X" + data[1:])
    with pytest.raises(ValueError, match="identity mismatch"):
        if when == "prepare":
            prepare_runtime_snapshot(source, bundle, output)
        else:
            verify_runtime_snapshot(source, output / "scene_relocation.json")


def test_unlisted_nested_dependency_rejected(scene):
    source, bundle, output, manifest = scene
    manifest["files"] = manifest["files"][:1]
    save_manifest(bundle, manifest)
    with pytest.raises(ValueError, match="dependency missing"):
        prepare_runtime_snapshot(source, bundle, output)


def test_unlisted_snapshot_absolute_asset_rejected(scene):
    source, bundle, output, manifest = scene
    source.write_text(source.read_text().replace("/old/project/assets/mesh.usda", "/unlisted/mesh.usda"))
    with pytest.raises(ValueError, match="not an explicit bundle mapping"):
        prepare_runtime_snapshot(source, bundle, output)


def test_unportable_nested_absolute_asset_rejected(scene):
    source, bundle, output, manifest = scene
    mesh = bundle / "assets/mesh.usda"
    mesh.write_text(mesh.read_text().replace("@texture.png@", "@/old/project/assets/texture.png@"))
    manifest["files"][0].update(identity(mesh))
    save_manifest(bundle, manifest)
    with pytest.raises(ValueError, match="unsupported absolute asset"):
        prepare_runtime_snapshot(source, bundle, output)


@pytest.mark.parametrize("field", ["geometry", "material", "string"])
def test_audit_independently_rejects_changed_opinions_even_with_rehashed_report(scene, field):
    source, bundle, output, manifest = scene
    report = prepare_runtime_snapshot(source, bundle, output)
    runtime = Path(report["runtime_snapshot"])
    layer = Sdf.Layer.OpenAsAnonymous(str(runtime))
    if field == "geometry":
        from pxr import Gf
        layer.GetAttributeAtPath("/World.xformOp:translate").default = Gf.Vec3d(4, 5, 6)
    elif field == "material":
        layer.GetAttributeAtPath("/World.mdl").default = Sdf.AssetPath("Different.mdl")
    else:
        layer.GetAttributeAtPath("/World.note").default = "modified metadata"
    layer.Export(str(runtime))
    report_path = output / "scene_relocation.json"
    disk_report = json.loads(report_path.read_text())
    disk_report.update(runtime_sha256=identity(runtime)["sha256"], runtime_bytes=identity(runtime)["bytes"])
    report_path.write_text(json.dumps(disk_report))
    with pytest.raises(ValueError, match="authored content|unexpected asset"):
        verify_runtime_snapshot(source, report_path)


def test_audit_rejects_report_manifest_and_source_changes(scene):
    source, bundle, output, manifest = scene
    prepare_runtime_snapshot(source, bundle, output)
    report_path = output / "scene_relocation.json"
    original_report = report_path.read_bytes()
    report = json.loads(original_report)
    report["asset_path_map"] = {}
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="report identity"):
        verify_runtime_snapshot(source, report_path)
    report_path.write_bytes(original_report)
    manifest_path = bundle / "asset_bundle.json"
    original_manifest = manifest_path.read_bytes()
    manifest_path.write_bytes(original_manifest + b"\n")
    with pytest.raises(ValueError, match="report identity"):
        verify_runtime_snapshot(source, report_path)
    manifest_path.write_bytes(original_manifest)
    source.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="report identity"):
        verify_runtime_snapshot(source, report_path)


def test_report_reformatting_changes_external_report_identity(scene):
    source, bundle, output, manifest = scene
    original = prepare_runtime_snapshot(source, bundle, output)
    report_path = output / "scene_relocation.json"
    report_path.write_text(json.dumps(json.loads(report_path.read_text()), sort_keys=True))
    audited = verify_runtime_snapshot(source, report_path)
    assert audited["report_sha256"] != original["report_sha256"]
    assert audited != original  # Caller compares this record to eval_manifest.
