"""Source/build integrity checks that do not import the model dependencies."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1] / "src/pressb/model_sources/walloss"
SFT_SOURCE_ADAPTATIONS = {
    "train_qact.py",
    "wall_x/data/load_lerobot_dataset.py",
    "wall_x/trainer/qwen_vl_act_trainer.py",
}


def build_module():
    spec = importlib.util.spec_from_file_location("pressb_walloss_build_test", ROOT / "build_extension.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_imported_source_hashes_match_deployed_snapshot():
    manifest = json.loads((ROOT / "source_manifest.json").read_text())
    assert len(manifest["source_commit"]) == 40
    assert manifest["files"]
    changed_sources = set()
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        assert path.is_relative_to(ROOT)
        assert path.stat().st_size == entry["bytes"], entry["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["bundled_sha256"], entry["path"]
        if entry["bundled_sha256"] != entry["source_sha256"]:
            changed_sources.add(entry["path"])
        if entry["path"] in SFT_SOURCE_ADAPTATIONS:
            assert entry["adaptation"], entry["path"]
    # Model, serving and CUDA sources retain exact upstream equality; only
    # explicitly named SFT entrypoint/data/trainer files may contain adaptations.
    assert changed_sources == SFT_SOURCE_ADAPTATIONS
    for entry in manifest["pressb_support_files"]:
        path = ROOT / entry["path"]
        assert path.is_relative_to(ROOT)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], entry["path"]


def test_inference_training_and_wire_dependencies_are_bundled():
    for relative in (
        "train_qact.py", "wall_x/trainer/qwen_vl_act_trainer.py",
        "wall_x/data/load_lerobot_dataset.py", "wall_x/model/action_head.py",
        "wall_x/model/flow_context.py", "wall_x/model/flow_prefix.py",
        "wall_x/model/vision_batch.py", "inference/launch_server.py",
        "packages/openpi_client/websocket_policy_server.py",
        "packages/openpi_client/msgpack_numpy.py", "csrc/window_index.cu",
        "scripts/compute_norm_stats.py", "scripts/normalize.py",
        "scripts/merge_sharded_weights.py", "licenses/CUTLASS-LICENSE.txt",
    ):
        assert (ROOT / relative).is_file(), relative


@pytest.mark.parametrize("value,expected", [("9.0", ("9.0", "90")), ("sm_120", ("12.0", "120")), ("120", ("12.0", "120")), ("compute_90", ("9.0", "90"))])
def test_extension_build_uses_explicit_native_architecture(value, expected):
    assert build_module().architecture(value) == expected


@pytest.mark.parametrize("value", ["12.0;9.0", "12.0+PTX", "native", "", "0.0", "1"])
def test_extension_build_rejects_ambiguous_architecture(value):
    with pytest.raises(ValueError, match="one CUDA architecture"):
        build_module().architecture(value)


def test_cutlass_verification_rejects_wrong_revision_or_modified_headers(tmp_path, monkeypatch):
    module = build_module()
    include = tmp_path / "include/cutlass/cutlass.h"
    include.parent.mkdir(parents=True)
    include.write_text("fixture")
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "a" * 40)
    with pytest.raises(ValueError, match="revision mismatch"):
        module.verify_cutlass(tmp_path)
    outputs = iter([module.CUTLASS_COMMIT, " M include/cutlass/cutlass.h"])
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: next(outputs))
    with pytest.raises(ValueError, match="working-tree changes"):
        module.verify_cutlass(tmp_path)


def test_build_dependency_pins_agree_with_manifest():
    module = build_module()
    manifest = json.loads((ROOT / "source_manifest.json").read_text())
    assert module.CUTLASS_COMMIT == manifest["build_dependencies"]["cutlass"]["commit"]
    lerobot = manifest["training_dependencies"]["lerobot"]
    assert lerobot["repository"] + "@" + lerobot["commit"] in (ROOT / "requirements-training.txt").read_text()
