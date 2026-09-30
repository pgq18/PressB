"""Checkpoint provenance and fixed-scene guards work across trained weights."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_policy_eval import policy_identity, scene_identity, sha
from summarize_policy_eval import summary_identity


def service_fixture(step=12800, expected_step=True):
    digest = "c2502396b093ec45d2b3fa2ac1d602b29ecafaa147a7bfe14768971028834c72"
    manifest = {"expected_checkpoint_sha256": digest, "arguments": {"expected_checkpoint_sha256": digest}}
    if expected_step:
        manifest["expected_checkpoint_step"] = step
        manifest["arguments"]["expected_checkpoint_step"] = step
    health = dict(status="ready", checkpoint_verified=True, checkpoint_step=step, model_sha256=digest,
                  provenance=dict(checkpoint=f"/runs/checkpoints/step_{step:06d}", model_sha256=digest),
                  fps=30, action_horizon=7, camera_order=["global", "wrist"])
    return manifest, health


@pytest.mark.parametrize("step,expected_step", [(2000, False), (2000, True), (12800, True)])
def test_identity_reads_served_step_and_accepts_legacy_manifest(step, expected_step):
    manifest, health = service_fixture(step, expected_step)
    proof = policy_identity(manifest, health)
    assert proof["checkpoint_step"] == step
    assert proof["checkpoint"] == health["provenance"]["checkpoint"]
    assert proof["checkpoint_sha256"] == manifest["expected_checkpoint_sha256"]


@pytest.mark.parametrize("change", ["hash", "provenance_hash", "step", "argument_step", "unverified", "camera_order"])
def test_wrong_serving_checkpoint_or_contract_is_rejected(change):
    manifest, health = service_fixture()
    if change == "hash":
        health["model_sha256"] = "f" * 64
    elif change == "provenance_hash":
        health["provenance"]["model_sha256"] = "e" * 64
    elif change == "step":
        health["checkpoint_step"] = 2000
    elif change == "argument_step":
        manifest["arguments"]["expected_checkpoint_step"] = 2000
    elif change == "unverified":
        health["checkpoint_verified"] = False
    else:
        health["camera_order"] = ["wrist", "global"]
    with pytest.raises(ValueError):
        policy_identity(manifest, health)


def scene_fixture(tmp_path, cfg=None):
    snapshot = tmp_path / "scene.usda"
    snapshot.write_text("#usda 1.0\n")
    cfg = {} if cfg is None else cfg
    identity = dict(config=cfg, scene_sha256=sha(snapshot))
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    collection = dict(identity=identity, collection_fingerprint=fingerprint, **identity)
    manifest = dict(**identity, arguments={"snapshot": str(snapshot)}, collection_fingerprint=fingerprint,
                    recorded_actions_used=False, target_planner_used=False)
    return manifest, collection


def test_fixed_scene_checks_actual_snapshot_and_collection_fingerprint(tmp_path):
    manifest, collection = scene_fixture(tmp_path)
    proof = scene_identity(manifest, collection)
    assert proof["panel_offset_x_m"] == proof["panel_offset_y_m"] == 0
    assert proof["panel_randomization_enabled"] is False
    bad = deepcopy(manifest)
    bad["collection_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="fingerprint"):
        scene_identity(bad, collection)
    Path(manifest["arguments"]["snapshot"]).write_text("changed snapshot")
    with pytest.raises(ValueError, match="training snapshot"):
        scene_identity(manifest, collection)


@pytest.mark.parametrize("cfg", [{"panel_offset_x_m": .001}, {"panel_offset_y_m": -.001},
                                {"panel_randomization": {"enabled": True}}])
def test_randomized_or_shifted_scene_cannot_be_reported_as_original(tmp_path, cfg):
    manifest, collection = scene_fixture(tmp_path, cfg)
    with pytest.raises(ValueError, match="unshifted"):
        scene_identity(manifest, collection)


@pytest.mark.parametrize("legacy_audit", [False, True])
def test_summary_identity_binds_actual_served_step_to_audit(tmp_path, legacy_audit):
    manifest, health = service_fixture(2000 if legacy_audit else 12800, not legacy_audit)
    for name, value in (("eval_manifest.json", manifest), ("policy_service.json", health)):
        (tmp_path / name).write_text(json.dumps(value))
    proof = dict(checkpoint_sha256=manifest["expected_checkpoint_sha256"],
                 eval_manifest_sha256=sha(tmp_path / "eval_manifest.json"))
    if not legacy_audit:
        proof.update(checkpoint_step=12800, policy_service_sha256=sha(tmp_path / "policy_service.json"))
    result, _, _ = summary_identity(tmp_path, {"provenance": proof})
    assert result["checkpoint_step"] == health["checkpoint_step"]
    health["model_sha256"] = "f" * 64
    (tmp_path / "policy_service.json").write_text(json.dumps(health))
    with pytest.raises(ValueError, match="hash"):
        summary_identity(tmp_path, {"provenance": proof})
