"""An altered capture identity must not pass the standalone LeRobot audit."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_lerobot import audit_collection_metadata, audit_episode_collection
from export_lerobot import COLLECTION_METADATA, SEMANTICS, TASKS, collection_snapshot


def make_collection(tmp_path, schema=9, fps=10):
    raw, dataset = tmp_path / "raw", tmp_path / "dataset"
    raw.mkdir()
    (dataset / "meta").mkdir(parents=True)
    identity = {"raw_schema_version": schema, "fps": fps, "pose_frame": "base_link",
                "pose_link": "gripper_tcp"}
    if schema >= 10:
        identity.update(physics_hz=120, capture_stride=120 // fps, action_horizon_s=1 / fps)
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    collection = {**identity, "identity": identity, "collection_fingerprint": fingerprint,
                  "task_texts": TASKS,
                  **{key: SEMANTICS[key] for key in ("tcp_offset_link6_m", "press_tip_offset_link6_m")}}
    for view, position_key, orientation_key in (
            ("wrist", "optical_position_link6", "optical_quaternion_wxyz_link6"),
            ("global", "fixed_position_world", "fixed_quaternion_wxyz_world")):
        collection[view] = {"width": 640, "height": 480,
                            "K": [[625., 0., 320.], [0., 625., 240.], [0., 0., 1.]],
                            "sensor": {position_key: [0., 0., 0.], orientation_key: [1., 0., 0., 0.]}}
    (raw / "collection_metadata.json").write_text(json.dumps(collection, indent=2))
    shutil.copyfile(raw / "collection_metadata.json", dataset / COLLECTION_METADATA)
    manifest = {"collection_metadata": collection_snapshot(raw)[2]}
    return raw, dataset, collection, manifest


@pytest.mark.parametrize("schema,fps", [(7, 10), (9, 10), (10, 10), (10, 20), (10, 30)])
def test_valid_existing_and_new_capture_identities(tmp_path, schema, fps):
    raw, dataset, expected, manifest = make_collection(tmp_path, schema, fps)
    collection, evidence = audit_collection_metadata(dataset, raw, manifest)
    assert collection == expected and evidence["byte_identical_to_raw"]
    relocated = tmp_path / "relocated_raw"
    raw.rename(relocated)
    _, evidence = audit_collection_metadata(dataset, relocated, manifest)
    assert evidence["source_path"] != evidence["source_path_at_export"]


def test_changed_exported_calibration_fails_even_with_intact_numeric_data(tmp_path):
    raw, dataset, collection, manifest = make_collection(tmp_path)
    collection["wrist"]["K"][0][0] += 1.
    (dataset / COLLECTION_METADATA).write_text(json.dumps(collection))
    with pytest.raises(ValueError, match="differs from its source"):
        audit_collection_metadata(dataset, raw, manifest)


def test_missing_exported_calibration_or_provenance_fails(tmp_path):
    raw, dataset, _, manifest = make_collection(tmp_path)
    with pytest.raises(ValueError, match="missing collection_metadata"):
        audit_collection_metadata(dataset, raw, {})
    (dataset / COLLECTION_METADATA).unlink()
    with pytest.raises(ValueError, match="metadata is missing"):
        audit_collection_metadata(dataset, raw, manifest)


def test_changed_raw_identity_cannot_be_hidden_by_matching_exported_copy(tmp_path):
    raw, dataset, collection, manifest = make_collection(tmp_path)
    collection["raw_schema_version"] = 7
    payload = json.dumps(collection)
    (raw / "collection_metadata.json").write_text(payload)
    (dataset / COLLECTION_METADATA).write_text(payload)
    with pytest.raises(ValueError, match="fingerprint does not match"):
        audit_collection_metadata(dataset, raw, manifest)


def test_manifest_hash_must_identify_actual_calibration(tmp_path):
    raw, dataset, _, manifest = make_collection(tmp_path)
    manifest["collection_metadata"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="sha256 differs"):
        audit_collection_metadata(dataset, raw, manifest)


def test_episodes_cannot_mix_capture_settings_or_coordinate_metadata(tmp_path):
    _, _, collection, _ = make_collection(tmp_path)
    metadata = {"raw_schema_version": 9, "fps": 10,
                "collection_fingerprint": collection["collection_fingerprint"],
                "env_offset_m": [0., 6., 0.], "robot_base_world_m": [-.24, 6., .76]}
    entry = copy.deepcopy(metadata)
    audit_episode_collection(entry, metadata, collection, tmp_path)
    bad = {**metadata, "collection_fingerprint": "0" * 64}
    with pytest.raises(ValueError, match="collection_fingerprint does not match"):
        audit_episode_collection(entry, bad, collection, tmp_path)
    entry["env_offset_m"] = [0., 0., 0.]
    with pytest.raises(ValueError, match="env_offset_m differs"):
        audit_episode_collection(entry, metadata, collection, tmp_path)
