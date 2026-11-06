"""Held-out selection and evaluation reporting, using CPU mocks only."""
import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


evaluation = load_module("piper_evaluation_test", ROOT / "scripts/eval_piper_checkpoint.py")
poses = load_module("piper_eval_pose_test", ROOT / "starVLA/dataloader/piper_lerobot.py")


def examples():
    entries = [dict(episode_id=i, floor=24+i % 12, frames=248+i % 7, task=f"Press {24+i % 12} floor.") for i in range(1200)]
    split = poses._build_split(entries, "a"*64, 42)
    return entries, split


def test_selection_covers_12_floors_36_distinct_heldout_anchors_deterministically():
    entries, split = examples()
    anchors = evaluation.select_heldout_anchors(entries, split)
    assert anchors == evaluation.select_heldout_anchors(list(reversed(entries)), split)
    assert len(anchors) == 36 and len({a["source_episode_id"] for a in anchors}) == 12
    for floor in range(24, 36):
        group = [a for a in anchors if a["floor"] == floor]
        source = min(i for i in split["val_source_episode_ids"] if entries[i]["floor"] == floor)
        n = entries[source]["frames"]
        assert [a["frame_index"] for a in group] == [0, round((n-1)/2), n-1]
        assert all(a["source_episode_id"] == source for a in group)
        assert source not in split["train_source_episode_ids"]


def test_selection_rejects_leakage_missing_episode_and_wrong_floor_counts():
    entries, split = examples()
    leaked = copy.deepcopy(split)
    leaked["val_source_episode_ids"][0] = leaked["train_source_episode_ids"][0]
    with pytest.raises(ValueError, match="disjoint"):
        evaluation.select_heldout_anchors(entries, leaked)
    with pytest.raises(ValueError, match="cover"):
        evaluation.select_heldout_anchors(entries[:-1], split)
    changed = copy.deepcopy(entries)
    changed[split["val_source_episode_ids"][0]]["floor"] = 35
    with pytest.raises(ValueError, match="90 train/10"):
        evaluation.select_heldout_anchors(changed, split)


def fixture_anchor():
    entries, split = examples()
    return evaluation.select_heldout_anchors(entries, split)[-1]


def sample_for(anchor):
    truth = np.tile([.1, -.2, .3, 1, 0, 0, 0, 1, 0], (7, 1)).astype(np.float32)
    return dict(source_episode_id=anchor["source_episode_id"], floor=anchor["floor"], frame_index=anchor["frame_index"],
                lang=f"Press {anchor['floor']} floor.", image=["global", "wrist"], action=truth,
                state=truth[:1].copy(), action_is_pad=anchor["frame_index"] + np.arange(7) >= anchor["episode_frames"])


def prediction_for(sample):
    raw = sample["action"].copy()
    raw[:, 0] += .03
    control = np.empty((7, 8), np.float32)
    control[:, :3] = raw[:, :3]
    control[:, 3:7] = poses.rotation_6d_to_quaternion_wxyz(raw[:, 3:])
    control[:, 7] = .008
    return dict(raw_pose9=raw, pose8=control, controller_gripper_width_m=.008, gripper_is_learned=False,
                pose_frame="base_link", pose_link="gripper_tcp", xyz_units="metres")


def test_terminal_uses_one_real_action_and_excludes_future_padding_from_errors():
    anchor = fixture_anchor()
    sample = sample_for(anchor)
    prediction = prediction_for(sample)
    prediction["raw_pose9"][1:, 0] += 100
    prediction["pose8"][1:, 0] += 100
    result = evaluation.measure_prediction(prediction, sample, anchor, .008, poses)
    assert result["valid_actions"] == 1
    assert result["position_errors_m"] == pytest.approx([.03], abs=1e-7)
    assert result["rotation_errors_deg"] == [0.]
    assert result["first_action_position_m"] == pytest.approx(.03, abs=1e-7)
    assert result["state_copy_baseline_first_position_m"] == 0.


@pytest.mark.parametrize("bad,match", [("padding", "padding"), ("gripper", "gripper"),
                                        ("xyz", "changed raw xyz"), ("quaternion", "quaternion"),
                                        ("finite", "finite"), ("frame", "identity")])
def test_output_corruption_is_rejected(bad, match):
    anchor = fixture_anchor()
    sample = sample_for(anchor)
    prediction = prediction_for(sample)
    if bad == "padding":
        sample["action_is_pad"][0] = True
    elif bad == "gripper":
        prediction["pose8"][0, 7] = .009
    elif bad == "xyz":
        prediction["pose8"][0, 0] += .001
    elif bad == "quaternion":
        prediction["pose8"][0, 3] = 0
    elif bad == "finite":
        prediction["raw_pose9"][0, 0] = np.nan
    elif bad == "frame":
        sample["frame_index"] -= 1
    with pytest.raises(ValueError, match=match):
        evaluation.measure_prediction(prediction, sample, anchor, .008, poses)


def test_evaluate_36_with_one_policy_and_report_failures_without_false_success():
    entries, split = examples()
    anchors = evaluation.select_heldout_anchors(entries, split)
    lookup = {(a["source_episode_id"], a["frame_index"]): a for a in anchors}
    dataset = SimpleNamespace(split="val", export_manifest_sha256="a"*64, source_episode_ids=split["val_source_episode_ids"],
                              get_frame=lambda source, frame: sample_for(lookup[source, frame]))

    class MockPolicy:
        schema = {"dataset_sha256": "a"*64}
        controller_gripper_width_m = .008

        def __init__(self):
            self.calls = []
            self.failed_seed = None

        def predict(self, **kwargs):
            self.calls.append(kwargs)
            assert kwargs["global_image"] == "global" and kwargs["wrist_image"] == "wrist"
            if kwargs["seed"] == self.failed_seed:
                raise ValueError("Degenerate rotation 6D parallel rows")
            return prediction_for(sample_for(anchors[0]))

    policy = MockPolicy()
    result = evaluation.evaluate_anchors(policy, dataset, anchors, poses, 500042)
    assert result["success"] and result["overall"]["passed"] == 36
    assert len(policy.calls) == 36 and result["overall"]["valid_actions"] == 12 * 15
    assert all(row["passed"] == 3 for row in result["floors"].values())
    assert all(row["passed"] == 12 for row in result["anchor_positions"].values())
    assert policy.calls[0]["seed"] == 500042 + anchors[0]["source_episode_id"] * 100003
    policy.failed_seed = policy.calls[5]["seed"]
    failed = evaluation.evaluate_anchors(policy, dataset, anchors, poses, 500042)
    assert not failed["success"] and failed["overall"]["failed"] == 1
    assert failed["anchors"][5]["error_type"] == "ValueError"


def test_evaluation_refuses_train_split_before_model_calls():
    entries, split = examples()
    anchors = evaluation.select_heldout_anchors(entries, split)
    with pytest.raises(ValueError, match="held-out"):
        evaluation.evaluate_anchors(SimpleNamespace(schema={"dataset_sha256": "a"*64}),
                                    SimpleNamespace(split="train"), anchors, poses, 42)
