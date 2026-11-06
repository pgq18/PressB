"""Evaluate 36 held-out Piper anchors with one strictly restored GPU model.

Run from the repository root: python -m scripts.eval_piper_checkpoint --help.
Success means complete, finite, schema-correct inference, not a robot success
rate or an accuracy acceptance threshold. No IK/simulation is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np


def select_heldout_anchors(entries, split):
    """One lowest-source-ID held-out episode per floor; first/middle/last."""
    train = split["train_source_episode_ids"]
    val = split["val_source_episode_ids"]
    if (len(train) != len(set(train)) or len(val) != len(set(val))
            or set(train) & set(val) or len(train) != 1080 or len(val) != 120):
        raise ValueError("Expected disjoint 1080/120 whole-episode train/validation split")
    by_id = {row["episode_id"]: row for row in entries}
    if len(by_id) != len(entries) or set(by_id) != set(train) | set(val):
        raise ValueError("Split does not cover the source episodes exactly")
    result = []
    for floor in range(24, 36):
        selected = sorted(sid for sid in val if by_id[sid]["floor"] == floor)
        if len(selected) != 10 or sum(by_id[sid]["floor"] == floor for sid in train) != 90:
            raise ValueError(f"Expected floor {floor} to contain 90 train/10 held-out episodes")
        source_id = selected[0]
        length = by_id[source_id]["frames"]
        if type(length) is not int or length < 3 or by_id[source_id]["task"] != f"Press {floor} floor.":
            raise ValueError("Invalid held-out episode length or task")
        # Matches the trainer's evenly-spaced endpoint/midpoint selection.
        for label, frame in zip(("first", "middle", "last"), (0, int(round((length-1)/2)), length-1)):
            result.append(dict(source_episode_id=source_id, floor=floor, frame_index=frame,
                               anchor=label, episode_frames=length))
    if len(result) != 36 or len({(r["source_episode_id"], r["frame_index"]) for r in result}) != 36:
        raise ValueError("Expected 36 distinct held-out anchors")
    return result


def measure_prediction(prediction, sample, anchor, gripper_width_m, pose_helpers):
    """Independent output/padding checks and physical errors, without thresholds."""
    for key in ("source_episode_id", "floor", "frame_index"):
        if sample[key] != anchor[key]:
            raise ValueError(f"Selected held-out sample identity differs: {key}")
    if sample["lang"] != f"Press {anchor['floor']} floor.":
        raise ValueError("Sample task differs from its floor")
    raw = np.asarray(prediction["raw_pose9"])
    control = np.asarray(prediction["pose8"])
    truth = np.asarray(sample["action"])
    state = np.asarray(sample["state"])
    pad = np.asarray(sample["action_is_pad"])
    if (raw.shape != (7, 9) or control.shape != (7, 8) or truth.shape != (7, 9)
            or state.shape != (1, 9) or not all(np.isfinite(a).all() for a in (raw, control, truth, state))):
        raise ValueError("Pose arrays must have the declared shape and finite values")
    expected_pad = anchor["frame_index"] + np.arange(7) >= anchor["episode_frames"]
    if pad.dtype != np.bool_ or pad.shape != (7,) or not np.array_equal(pad, expected_pad):
        raise ValueError("Action padding differs from the selected episode boundary")
    if (prediction.get("gripper_is_learned") is not False
            or prediction.get("controller_gripper_width_m") != gripper_width_m
            or not np.array_equal(control[:, 7], np.full(7, gripper_width_m, dtype=control.dtype))):
        raise ValueError("Fixed controller gripper opening changed or was marked learned")
    if (prediction.get("pose_frame"), prediction.get("pose_link"), prediction.get("xyz_units")) != ("base_link", "gripper_tcp", "metres"):
        raise ValueError("Prediction coordinate frame, link or units differ")
    if not np.array_equal(raw[:, :3], control[:, :3]):
        raise ValueError("Output conversion changed raw xyz")
    if not np.allclose(np.linalg.norm(control[:, 3:7], axis=1), 1., rtol=0, atol=1e-6):
        raise ValueError("Output quaternion is not unit length")
    predicted_rot = pose_helpers.rotation_6d_to_matrix(raw[:, 3:])
    restored_rot = pose_helpers.quaternion_wxyz_to_matrix(control[:, 3:7])
    if not np.allclose(predicted_rot, restored_rot, atol=2e-6, rtol=0):
        raise ValueError("8D quaternion does not represent raw predicted row-major rot6")
    target_rot = pose_helpers.rotation_6d_to_matrix(truth[:, 3:])
    cosine = ((predicted_rot * target_rot).sum(axis=(-1, -2)) - 1) / 2
    rotation = np.degrees(np.arccos(np.clip(cosine, -1., 1.)))
    position = np.linalg.norm(raw[:, :3].astype(np.float64) - truth[:, :3], axis=-1)
    baseline_rotation = pose_helpers.rotation_6d_to_matrix(state[0, 3:])
    baseline_cosine = ((baseline_rotation * target_rot[0]).sum() - 1) / 2
    valid = ~pad
    return dict(valid_actions=int(valid.sum()), finite=True, fixed_gripper=True, quaternion_conversion=True,
                position_errors_m=position[valid].tolist(), rotation_errors_deg=rotation[valid].tolist(),
                first_action_position_m=float(position[0]), first_action_rotation_deg=float(rotation[0]),
                state_copy_baseline_first_position_m=float(np.linalg.norm(state[0, :3] - truth[0, :3])),
                state_copy_baseline_first_rotation_deg=float(np.degrees(np.arccos(np.clip(baseline_cosine, -1., 1.)))),
                raw_pose9=raw.tolist(), pose8=control.tolist(), target_pose9=truth.tolist(),
                state9=state[0].tolist(), action_is_pad=pad.tolist())


def summarize(rows):
    passed = [row for row in rows if row["success"]]
    if not passed:
        return dict(anchors=len(rows), passed=0, failed=len(rows), valid_actions=0)
    position = np.concatenate([row["position_errors_m"] for row in passed])
    rotation = np.concatenate([row["rotation_errors_deg"] for row in passed])
    return dict(anchors=len(rows), passed=len(passed), failed=len(rows)-len(passed), valid_actions=len(position),
                position_mean_m=float(position.mean()), position_max_m=float(position.max()),
                rotation_mean_deg=float(rotation.mean()), rotation_max_deg=float(rotation.max()),
                first_action_position_mean_m=float(np.mean([r["first_action_position_m"] for r in passed])),
                first_action_rotation_mean_deg=float(np.mean([r["first_action_rotation_deg"] for r in passed])),
                state_copy_baseline_first_position_mean_m=float(np.mean([r["state_copy_baseline_first_position_m"] for r in passed])),
                state_copy_baseline_first_rotation_mean_deg=float(np.mean([r["state_copy_baseline_first_rotation_deg"] for r in passed])))


def evaluate_anchors(policy, dataset, anchors, pose_helpers, evaluation_seed, *, progress=None):
    """Reuse the supplied model for every anchor; failures remain in the report."""
    if dataset.split != "val" or dataset.export_manifest_sha256 != policy.schema["dataset_sha256"]:
        raise ValueError("Evaluation must read this checkpoint's held-out dataset")
    if any(anchor["source_episode_id"] not in dataset.source_episode_ids for anchor in anchors):
        raise ValueError("Evaluation anchor is not in the saved held-out split")
    rows = []
    for ordinal, anchor in enumerate(anchors):
        seed = int(evaluation_seed) + anchor["source_episode_id"] * 100003 + anchor["frame_index"]
        row = dict(anchor, seed=seed)
        started = time.monotonic()
        try:
            sample = dataset.get_frame(anchor["source_episode_id"], anchor["frame_index"])
            prediction = policy.predict(global_image=sample["image"][0], wrist_image=sample["image"][1],
                                        state=sample["state"][0], task=sample["lang"], seed=seed)
            row.update(measure_prediction(prediction, sample, anchor, policy.controller_gripper_width_m, pose_helpers))
            row["success"] = True
        except Exception as error:
            row.update(success=False, error_type=type(error).__name__, error=str(error))
        row["seconds"] = time.monotonic() - started
        rows.append(row)
        if progress:
            progress(ordinal + 1, len(anchors), row)
    overall = summarize(rows)
    return dict(success=len(rows) == 36 and overall["failed"] == 0,
                success_definition="all 36 held-out anchors produced finite schema-correct poses and fixed gripper; no accuracy or robot-success threshold applied",
                overall=overall,
                floors={str(floor): summarize([row for row in rows if row["floor"] == floor]) for floor in range(24, 36)},
                anchor_positions={label: summarize([row for row in rows if row["anchor"] == label]) for label in ("first", "middle", "last")},
                episodes=sorted({row["source_episode_id"] for row in rows}), anchors=rows)


def _runtime_contract(policy):
    cfg, model = policy.config, policy.model
    if (cfg.framework.qwenvl.init_from_config is not True
            or cfg.framework.vj2_model.init_from_config is not True
            or cfg.framework.qwenvl.features_only is not True
            or model.training or any(parameter.requires_grad for parameter in model.parameters())):
        raise ValueError("Expected from-config backbones, features-only Qwen and frozen eval model")
    counts = {}
    for name, parameter in model.named_parameters():
        component = name.split(".", 1)[0]
        counts.setdefault(component, Counter())[str(parameter.dtype)] += parameter.numel()
    return dict(backbones_init_from_config=True, qwen_features_only=True, model_eval=True,
                trainable_parameters=0, parameter_elements_by_dtype={key: dict(value) for key, value in counts.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--base-vlm", type=Path)
    parser.add_argument("--base-encoder", type=Path)
    parser.add_argument("--controller-gripper-width-m", type=float, default=.008)
    parser.add_argument("--evaluation-seed", type=int, help="Defaults to saved training seed + 500000, as in trainer validation")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation report: {args.output}")
    from starVLA.inference.piper_policy import PiperPolicy, sha256
    from starVLA.dataloader import piper_lerobot as poses
    started = time.monotonic()
    policy = PiperPolicy(args.checkpoint, device=args.device, base_vlm=args.base_vlm,
                         base_encoder=args.base_encoder, controller_gripper_width_m=args.controller_gripper_width_m)
    loaded_seconds = time.monotonic() - started
    runtime = _runtime_contract(policy)
    seed = args.evaluation_seed if args.evaluation_seed is not None else int(policy.config.seed) + 500000
    dataset = poses.PiperLeRobotDataset(args.dataset_root, split="val", split_manifest=policy.split,
                                       seed=int(policy.config.seed), resolution=policy.resolution)
    try:
        anchors = select_heldout_anchors(dataset.manifest["episodes"], policy.split)
        def progress(done, total, row):
            print(json.dumps(dict(event="heldout_prediction", done=done, total=total,
                                  source_episode_id=row["source_episode_id"], frame_index=row["frame_index"],
                                  success=row["success"], error=row.get("error"))), flush=True)
        result = evaluate_anchors(policy, dataset, anchors, poses, seed, progress=progress)
        import torch
        runtime.update(device=str(policy.device), gpu_name=torch.cuda.get_device_name(policy.device),
                       cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(policy.device),
                       cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(policy.device))
        result.update(kind="piper_independent_heldout_checkpoint_inference", schema_version=1,
                      created_utc=datetime.now(timezone.utc).isoformat(), evaluation_seed=seed,
                      selection="lowest held-out source_episode_id per floor; frames 0, round((N-1)/2), N-1",
                      dataset_root=str(dataset.root), camera_keys=list(poses.CAMERA_KEYS),
                      controller_gripper_width_m=policy.controller_gripper_width_m, gripper_is_learned=False,
                      pose_frame="base_link", pose_link="gripper_tcp", xyz_units="metres", normalization="none",
                      runtime=runtime, model_loads=1, model_load_seconds=loaded_seconds,
                      total_seconds=time.monotonic()-started, provenance=policy.provenance,
                      code_sha256=dict(evaluation=sha256(__file__), adapter=sha256(poses.__file__),
                                       policy=sha256(Path(__file__).resolve().parents[1] / "starVLA/inference/piper_policy.py")))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps(dict(output=str(args.output.resolve()), success=result["success"], overall=result["overall"])), flush=True)
        return 0 if result["success"] else 1
    finally:
        dataset.close()


if __name__ == "__main__":
    raise SystemExit(main())
