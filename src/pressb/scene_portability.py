"""Relocate a frozen scene's verified assets without changing its identity.

The original snapshot remains the dataset's identity. A separate runtime layer
only changes asset paths; inverse relocation must reproduce every authored USD
opinion. Import USD lazily so metadata checks do not initialize Isaac Sim.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import posixpath
import re


ENGINE_ASSETS = frozenset({"OmniPBR.mdl"})


def _identity(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def _json_hash(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_bundle(asset_bundle: Path):
    bundle = Path(asset_bundle).resolve(strict=True)
    manifest_path = bundle / "asset_bundle.json"
    if not manifest_path.resolve(strict=True).is_relative_to(bundle):
        raise ValueError("Asset manifest escapes its bundle")
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Unsupported asset bundle schema")
    root = manifest.get("source_project_root")
    if (not isinstance(root, str) or not root.startswith("/") or root == "/"
            or str(PurePosixPath(root)) != root or posixpath.normpath(root) != root):
        raise ValueError("source_project_root must be a canonical absolute POSIX path")
    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Asset bundle must list its complete dependency closure")
    assets, sources, destinations = [], set(), set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Invalid bundle file entry")
        relative = row.get("relative_path")
        if (not isinstance(relative, str) or not relative or "\\" in relative
                or PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts
                or str(PurePosixPath(relative)) != relative or relative == "."):
            raise ValueError("Asset relative_path must stay inside its bundle")
        source = row.get("source_path")
        if source != str(PurePosixPath(root) / relative):
            raise ValueError("Asset source_path disagrees with its project-relative path")
        destination = (bundle / relative).resolve(strict=True)
        if not destination.is_relative_to(bundle) or not destination.is_file():
            raise ValueError("Asset file escapes its bundle or is not a regular file")
        if source in sources or destination in destinations:
            raise ValueError("Duplicate asset source or destination")
        sources.add(source)
        destinations.add(destination)
        expected_hash, expected_size = row.get("sha256"), row.get("bytes")
        if (not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash)
                or type(expected_size) is not int or expected_size < 0):
            raise ValueError("Invalid asset SHA256 or byte count")
        actual = _identity(destination)
        if actual != {"sha256": expected_hash, "bytes": expected_size}:
            raise ValueError(f"Asset identity mismatch: {relative}")
        assets.append(dict(source_path=source, relative_path=relative,
                           path=str(destination), **actual))
    return bundle, manifest_path, sorted(assets, key=lambda item: item["source_path"])


def _open_layer(path: Path):
    from pxr import Sdf

    # OpenAsAnonymous reads fresh bytes instead of reusing an Sdf layer cache.
    layer = Sdf.Layer.OpenAsAnonymous(str(path))
    if layer is None:
        raise ValueError(f"Unable to read USD layer: {path}")
    return layer


def _validate_closure(assets):
    from pxr import UsdUtils

    by_source = {item["source_path"]: item for item in assets}
    for item in assets:
        if PurePosixPath(item["relative_path"]).suffix.lower() not in {".usd", ".usda", ".usdc"}:
            continue
        layer = _open_layer(Path(item["path"]))

        def check(asset):
            if not asset or asset in ENGINE_ASSETS:
                return asset
            # Bundle layers are immutable, so they must already be portable.
            if asset.startswith("/") or ":" in asset or "\\" in asset:
                raise ValueError(f"Bundle USD has an unsupported absolute asset: {asset}")
            original = posixpath.normpath(posixpath.join(posixpath.dirname(item["source_path"]), asset))
            if original not in by_source:
                raise ValueError(f"Asset dependency missing from bundle: {original}")
            resolved = (Path(item["path"]).parent / asset).resolve(strict=True)
            if resolved != Path(by_source[original]["path"]):
                raise ValueError(f"Relative USD dependency resolves to a different asset: {asset}")
            return asset

        UsdUtils.ModifyAssetPaths(layer, check)


def _relocated_layer(source_snapshot, assets):
    from pxr import Sdf, UsdUtils

    source = _open_layer(source_snapshot)
    runtime = Sdf.Layer.CreateAnonymous("runtime_scene.usda")
    runtime.TransferContent(source)
    mapping = {item["source_path"]: item["path"] for item in assets}
    used = {}

    def relocate(asset):
        if not asset or asset in ENGINE_ASSETS:
            return asset
        if asset not in mapping:
            raise ValueError(f"Snapshot asset is not an explicit bundle mapping: {asset}")
        used[asset] = mapping[asset]
        return mapping[asset]

    UsdUtils.ModifyAssetPaths(runtime, relocate)
    return source, runtime, dict(sorted(used.items()))


def _assert_authored_content(source, runtime, mapping):
    from pxr import Sdf, UsdUtils

    restored = Sdf.Layer.CreateAnonymous("restored_scene.usda")
    restored.TransferContent(runtime)
    inverse = {destination: original for original, destination in mapping.items()}

    def restore(asset):
        if not asset or asset in ENGINE_ASSETS:
            return asset
        if asset not in inverse:
            raise ValueError(f"Runtime scene contains an unexpected asset path: {asset}")
        return inverse[asset]

    UsdUtils.ModifyAssetPaths(restored, restore)
    if restored.ExportToString() != source.ExportToString():
        raise ValueError("Runtime scene changed authored content beyond asset paths")


def _report(source_snapshot, runtime_snapshot, bundle, manifest_path, assets, mapping):
    from pxr import Usd

    source_id, runtime_id = _identity(source_snapshot), _identity(runtime_snapshot)
    return dict(schema_version=1, source_snapshot=str(source_snapshot),
                source_sha256=source_id["sha256"], source_bytes=source_id["bytes"],
                runtime_snapshot=str(runtime_snapshot), runtime_sha256=runtime_id["sha256"],
                runtime_bytes=runtime_id["bytes"], bundle_root=str(bundle),
                bundle_manifest_path=str(manifest_path),
                bundle_manifest_sha256=_identity(manifest_path)["sha256"],
                assets=assets, asset_path_map=mapping, asset_path_map_sha256=_json_hash(mapping),
                assets_sha256=_json_hash(assets), authored_content_preserved=True,
                original_snapshot_unchanged=True, engine_assets=sorted(ENGINE_ASSETS),
                usd_version=list(Usd.GetVersion()))


def _with_report_identity(report, report_path):
    return dict(report, report_path=str(report_path), report_sha256=_identity(report_path)["sha256"])


def prepare_runtime_snapshot(source_snapshot: Path, asset_bundle: Path, output_dir: Path) -> dict:
    """Verify a complete asset bundle and write a separately identified runtime USD.

    The returned report also includes its on-disk SHA256; save that return value
    in the evaluation manifest so an auditor can detect report modifications.
    """
    source_snapshot = Path(source_snapshot).resolve(strict=True)
    source_identity = _identity(source_snapshot)
    bundle, manifest_path, assets = _load_bundle(asset_bundle)
    _validate_closure(assets)
    source, runtime, mapping = _relocated_layer(source_snapshot, assets)
    _assert_authored_content(source, runtime, mapping)
    output_dir = Path(output_dir).resolve()
    runtime_snapshot = output_dir / "runtime_scene.usda"
    report_path = output_dir / "scene_relocation.json"
    if any(path.exists() or path.is_symlink() for path in (runtime_snapshot, report_path)):
        raise FileExistsError("Runtime snapshot/report already exists; use a fresh output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not runtime.Export(str(runtime_snapshot)):
        raise RuntimeError("Unable to export relocated runtime snapshot")
    _assert_authored_content(source, _open_layer(runtime_snapshot), mapping)
    if _identity(source_snapshot) != source_identity:
        raise ValueError("Original snapshot changed while preparing relocation")
    report = _report(source_snapshot, runtime_snapshot, bundle, manifest_path, assets, mapping)
    with report_path.open("x") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return _with_report_identity(report, report_path)


def verify_runtime_snapshot(source_snapshot: Path, report_path: Path) -> dict:
    """Independently verify disk assets and USD opinions; return report plus hash.

    Callers must additionally compare the returned dict with the relocation
    record saved in the original evaluation manifest (including report_sha256).
    """
    source_snapshot = Path(source_snapshot).resolve(strict=True)
    report_path = Path(report_path).resolve(strict=True)
    report = json.loads(report_path.read_text())
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise ValueError("Unsupported scene relocation report schema")
    if report.get("source_snapshot") != str(source_snapshot):
        raise ValueError("Relocation report names a different source snapshot")
    runtime_snapshot = report_path.parent / "runtime_scene.usda"
    if report.get("runtime_snapshot") != str(runtime_snapshot):
        raise ValueError("Runtime snapshot is outside the reported evaluation output")
    bundle, manifest_path, assets = _load_bundle(Path(report["bundle_root"]))
    _validate_closure(assets)
    source, expected_runtime, mapping = _relocated_layer(source_snapshot, assets)
    runtime = _open_layer(runtime_snapshot)
    _assert_authored_content(source, runtime, mapping)
    if runtime.ExportToString() != expected_runtime.ExportToString():
        raise ValueError("Runtime scene differs from verified path relocation")
    expected = _report(source_snapshot, runtime_snapshot, bundle, manifest_path, assets, mapping)
    # USD version is recorded provenance, not a requirement that an independent
    # auditor run the same engine. Both comparisons above use this auditor's USD.
    version = report.get("usd_version")
    if not isinstance(version, list) or len(version) != 3 or any(type(v) is not int for v in version):
        raise ValueError("Invalid USD version provenance")
    expected["usd_version"] = version
    if report != expected:
        raise ValueError("Relocation report identity, mapping, or asset hashes changed")
    return _with_report_identity(report, report_path)
