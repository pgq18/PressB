"""Resume must reject changed training prefixes, settings, and encoded artifacts."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from export_press_dataset import dataset_inventory, plan_records, press_semantics, validate_committed
from export_lerobot import SEMANTICS, write_json


def test_terminal_semantics_are_explicit_without_changing_capture_semantics():
    before = deepcopy(SEMANTICS)
    derived = press_semantics({"raw_schema_version": 10, "fps": 30, "physics_hz": 120,
                               "capture_stride": 4, "action_horizon_s": 1 / 30})
    assert derived["episode_end"] == "first_sampled_target_light_on"
    assert derived["terminal_action"] == "repeat_previous_planned_target"
    assert derived["action_horizon_s"] == 1 / 30
    assert SEMANTICS == before


@pytest.mark.parametrize("records", [[], [dict(source_episode_id=0, floor=24, cut_frame_index=0, kept_frames=1)],
    [dict(source_episode_id=0, floor=24, cut_frame_index=5, kept_frames=5)],
    [dict(source_episode_id=0, floor=24, cut_frame_index=5, kept_frames=6)] * 2])
def test_empty_duplicate_or_invalid_cut_records_are_rejected(records):
    with pytest.raises(ValueError):
        plan_records({"episodes": records})


def committed_fixture(tmp_path):
    record = dict(source_episode_id=0, floor=24, cut_frame_index=5, kept_frames=6)
    identity = {"cut_plan_sha256": "plan-a", "source_episode_ids": [0], "part_size": 25}
    write_json(tmp_path / "meta/export_manifest.json", {"export_identity": identity,
                                                        "episodes": [{"press_prefix": record}]})
    write_json(tmp_path / "meta/audit.json", {"success": True, "full_video_decode": True})
    (tmp_path / "videos").mkdir()
    (tmp_path / "videos/test.mp4").write_bytes(b"verified encoded prefix")
    write_json(tmp_path / "meta/commit_inventory.json", dataset_inventory(tmp_path))
    return record, identity


def test_resume_rejects_modified_video_even_with_old_success_report(tmp_path):
    record, identity = committed_fixture(tmp_path)
    validate_committed(tmp_path, [record], identity)
    (tmp_path / "videos/test.mp4").write_bytes(b"later frames appended")
    with pytest.raises(ValueError, match="changed since"):
        validate_committed(tmp_path, [record], identity)


def test_resume_rejects_mixed_selection_or_plan(tmp_path):
    record, identity = committed_fixture(tmp_path)
    with pytest.raises(ValueError, match="mix conversion"):
        validate_committed(tmp_path, [record], {**identity, "cut_plan_sha256": "plan-b"})
    with pytest.raises(ValueError, match="different source selection"):
        validate_committed(tmp_path, [{**record, "cut_frame_index": 6, "kept_frames": 7}], identity)


def test_extra_readback_report_does_not_change_committed_training_artifacts(tmp_path):
    record, identity = committed_fixture(tmp_path)
    write_json(tmp_path / "meta/published_readback.json", {"success": True})
    validate_committed(tmp_path, [record], identity)
