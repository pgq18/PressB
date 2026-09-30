#!/usr/bin/env python3
"""Extract recorded Cartesian actions through the official LeRobot reader.

Run with .conda/envs/lerobot/bin/python. By default, source IDs 0..11 select
one episode for each floor, using the saved source-to-dataset mapping. The
dataset's episode_index is not assumed to equal source_episode_id.

action[k] is an absolute base_link gripper-TCP target nominally one sample
after state[k] (1/30 s for this collection). Press-only terminal targets repeat
the previous planned endpoint; legacy full-cycle targets clamp to the final
plan pose. The dataset clock starts at the first measured
post-physics state; that state was captured at physics time 1/120 s.

Only action may drive the replay after initialization. initial_joint_position
may initialize the articulation and seed its first IK solve once. Arrays with
the diagnostic_ prefix are recorded references for comparison only; they must
not drive the robot or supply per-frame IK seeds.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
POSE_NAMES = ["x_m", "y_m", "z_m", "qw", "qx", "qy", "qz", "gripper_width_m"]
ARRAY_SOURCES = {
    "action": ("action", np.float32, (8,)),
    "state": ("observation.state", np.float32, (8,)),
    "timestamp": ("timestamp", np.float32, ()),
    "sim_time": ("observation.sim_time", np.float64, ()),
    "diagnostic_joint_position": ("observation.joint_position", np.float32, (6,)),
    "diagnostic_joint_target": ("action.joint_target", np.float32, (6,)),
    "dataset_index": ("index", np.int64, ()),
    "frame_index": ("frame_index", np.int64, ()),
}
FULL_CYCLE_END = "full_cycle_return_home"
PRESS_ONLY_END = "first_sampled_target_light_on"


def episode_end_policy(semantics):
    """Legacy exports omit termination; never infer it from replay outcomes."""
    policy = semantics.get("episode_end", FULL_CYCLE_END)
    if policy not in (FULL_CYCLE_END, PRESS_ONLY_END):
        raise ValueError(f"Unsupported recorded episode_end: {policy}")
    return policy


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def scalar(value):
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Expected one scalar, found shape {array.shape}")
    return array.reshape(-1)[0].item()


def extract_episode(dataset, entry, episode_index, fps, physics_offset):
    """Read every saved numeric row with get_raw_item; no video or raw-file reads."""
    meta = dataset.meta.episodes[episode_index]
    start, stop = int(meta["dataset_from_index"]), int(meta["dataset_to_index"])
    count = stop - start
    if count != entry["frames"] or count != int(meta["length"]):
        raise ValueError(f"Episode length differs from provenance: source {entry['episode_id']}")
    expected_task = f"Press {entry['floor']} floor."
    if entry["task"] != expected_task or meta["tasks"] != [expected_task]:
        raise ValueError(f"Episode task mapping differs: source {entry['episode_id']}")
    task_index = dataset.meta.get_task_index(expected_task)
    arrays = {name: np.empty((count,) + tail, dtype=dtype)
              for name, (_, dtype, tail) in ARRAY_SOURCES.items()}
    for local_index, index in enumerate(range(start, stop)):
        row = dataset.get_raw_item(index)
        expected = {"episode_index": episode_index, "source_episode_id": entry["episode_id"],
                    "source_seed": entry["seed"], "floor": entry["floor"], "task_index": task_index,
                    "index": index, "frame_index": local_index}
        for key, value in expected.items():
            if scalar(row[key]) != value:
                raise ValueError(f"Row {index} has an unexpected {key}")
        for name, (feature, dtype, tail) in ARRAY_SOURCES.items():
            value = np.asarray(row[feature], dtype=dtype)
            if not tail:
                value = np.asarray(scalar(value), dtype=dtype)
            if value.shape != tail or not np.isfinite(value).all():
                raise ValueError(f"Invalid {feature} at dataset row {index}: {value.shape}")
            arrays[name][local_index] = value
    exact_time = np.arange(count, dtype=np.float64) / fps
    if (not np.allclose(arrays["timestamp"], exact_time, rtol=0, atol=2e-6)
            or not np.allclose(arrays["sim_time"], exact_time, rtol=0, atol=1e-10)):
        raise ValueError(f"Recorded timestamps differ from {fps} Hz: source {entry['episode_id']}")
    for name in ("action", "state"):
        pose = arrays[name]
        if (not np.allclose(np.linalg.norm(pose[:, 3:7], axis=1), 1., rtol=0, atol=2e-5)
                or np.any((pose[:, 7] < 0) | (pose[:, 7] > .1))):
            raise ValueError(f"Invalid recorded TCP quaternion or gripper width in {name}")
    arrays["initial_joint_position"] = arrays["diagnostic_joint_position"][0].copy()
    arrays["sample_physics_time"] = arrays["sim_time"] + physics_offset
    return arrays, start, stop


def prepare(dataset_root: Path, output: Path, source_ids: list[int]):
    dataset_root, output = dataset_root.resolve(), output.resolve()
    if output == dataset_root or dataset_root in output.parents or output in dataset_root.parents:
        raise ValueError("Replay output must be separate from the recorded dataset")
    if output.exists():
        raise FileExistsError(f"Replay input already exists; use a new --output to preserve it: {output}")
    if not source_ids or len(source_ids) != len(set(source_ids)) or any(x < 0 for x in source_ids):
        raise ValueError("Source episode IDs must be distinct, nonnegative integers")
    manifest_path = dataset_root / "meta/export_manifest.json"
    collection_path = dataset_root / "meta/collection_metadata.json"
    audit_path = dataset_root / "meta/audit.json"
    exported, collection, audit = map(read_json, (manifest_path, collection_path, audit_path))
    manifest_hash, collection_hash = sha256(manifest_path), sha256(collection_path)
    if not audit.get("success") or audit.get("manifest_sha256") != manifest_hash:
        raise ValueError("The current export manifest has no matching successful dataset audit")
    if collection_hash != exported["collection_metadata"]["sha256"]:
        raise ValueError("Embedded collection settings differ from the exported provenance")
    actual_version = version("lerobot")
    if actual_version != exported["lerobot_version"]:
        raise ValueError(f"Use lerobot=={exported['lerobot_version']}, found {actual_version}")
    semantics = exported["semantics"]
    episode_end = episode_end_policy(semantics)
    expected = {"pose_frame": "base_link", "pose_link": "gripper_tcp", "quaternion_order": "wxyz",
                "pose_names": POSE_NAMES, "action_type": "absolute_gripper_tcp_pose_and_total_opening"}
    if any(semantics.get(key) != value for key, value in expected.items()):
        raise ValueError("Replay requires base_link absolute gripper-TCP actions in wxyz order")
    fps, physics_hz, stride = (collection[key] for key in ("fps", "physics_hz", "capture_stride"))
    if (type(fps) is not int or fps <= 0 or physics_hz != 120 or stride * fps != physics_hz
            or semantics["fps"] != fps or abs(semantics["action_horizon_s"] - 1 / fps) > 1e-12):
        raise ValueError("Inconsistent sample frequency, physics clock, or action horizon")
    offset = collection["first_sample_physics_time_s"]
    if abs(offset - 1 / physics_hz) > 1e-12 or abs(collection["config"]["physics_dt"] - offset) > 1e-12:
        raise ValueError("Unexpected first-sample physics time")
    snapshot = Path(collection["source_snapshot"]).resolve()
    snapshot_hash = sha256(snapshot)
    if snapshot_hash != collection["scene_sha256"]:
        raise ValueError(f"Recorded scene snapshot has changed: {snapshot}")
    by_source = {entry["episode_id"]: (index, entry) for index, entry in enumerate(exported["episodes"])}
    if len(by_source) != len(exported["episodes"]):
        raise ValueError("Duplicate source IDs in export provenance")
    missing = sorted(set(source_ids) - set(by_source))
    if missing:
        raise ValueError(f"Unknown source episode IDs: {missing}")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/hf"))
    os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset("local/piper_elevator", root=dataset_root, video_backend="pyav")
    if dataset.fps != fps or dataset.meta.info.codebase_version != "v3.0":
        raise ValueError("Official reader rate or dataset format differs from provenance")
    for name in ("action", "observation.state"):
        feature = dataset.features[name]
        if feature["dtype"] != "float32" or tuple(feature["shape"]) != (8,) or feature["names"] != POSE_NAMES:
            raise ValueError(f"Unexpected official reader feature schema: {name}")
    numeric_files = {}
    for source_id in source_ids:
        index, _ = by_source[source_id]
        relative = str(dataset.meta.get_data_file_path(index))
        if relative not in numeric_files:
            path = dataset_root / relative
            numeric_files[relative] = {"relative_path": relative, "sha256": sha256(path),
                                       "bytes": path.stat().st_size}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.preparing.{uuid.uuid4().hex}")
    staging.mkdir()
    records = []
    try:
        # The default HF torch transform narrows Python floats to float32.
        # This in-memory context preserves the stored float64 sim_time column;
        # get_raw_item still supplies every numeric value through LeRobot.
        with dataset.hf_dataset.formatted_as(type=None):
            for source_id in source_ids:
                index, entry = by_source[source_id]
                arrays, start, stop = extract_episode(dataset, entry, index, fps, offset)
                stem = f"episode_{source_id:06d}"
                npz_path, json_path = staging / f"{stem}.npz", staging / f"{stem}.json"
                np.savez_compressed(npz_path, **arrays)
                with np.load(npz_path, allow_pickle=False) as saved:
                    if set(saved.files) != set(arrays) or any(
                            not np.array_equal(saved[key], value) for key, value in arrays.items()):
                        raise ValueError(f"Prepared NPZ numeric round trip changed values: {stem}")
                record = {"source_episode_id": source_id, "dataset_episode_index": index,
                          "episode_end": episode_end,
                          "episode_end_source_export_manifest_sha256": manifest_hash,
                          "source_seed": entry["seed"], "floor": entry["floor"], "task": entry["task"],
                          "frames": stop - start, "dataset_from_index": start, "dataset_to_index": stop,
                          "npz_file": npz_path.name, "npz_sha256": sha256(npz_path),
                          "numeric_source_file": numeric_files[str(dataset.meta.get_data_file_path(index))],
                          "recorded_env_offset_m": entry["env_offset_m"],
                          "recorded_robot_base_world_m": entry["robot_base_world_m"],
                          "initialization": {"array": "initial_joint_position",
                              "source_feature": "observation.joint_position", "source_frame_index": 0,
                              "permitted_use": "initialize articulation and first IK seed once only"},
                          "control_source": "action", "numeric_reader": "LeRobotDataset.get_raw_item",
                          "diagnostic_only_arrays": ["diagnostic_joint_position", "diagnostic_joint_target"],
                          "diagnostic_restriction": "Comparison only; never actuator commands or per-frame IK seeds.",
                          "npz_roundtrip_exact": True}
                for key in ("panel_offset_x_m", "panel_offset_y_m", "panel_layout"):
                    if key in entry:
                        record[key] = entry[key]
                    elif collection["raw_schema_version"] >= 11:
                        raise ValueError(f"Randomized replay source is missing {key}")
                write_json(json_path, record)
                records.append({**record, "manifest_file": json_path.name, "manifest_sha256": sha256(json_path)})
                print(f"Prepared source {source_id:06d} -> dataset episode {index}, "
                      f"floor {entry['floor']}, {stop-start} frames", flush=True)
        for relative, evidence in numeric_files.items():
            if sha256(dataset_root / relative) != evidence["sha256"]:
                raise ValueError(f"Dataset numeric source changed during extraction: {relative}")
        if (sha256(manifest_path) != manifest_hash or sha256(collection_path) != collection_hash
                or sha256(snapshot) != snapshot_hash):
            raise ValueError("Recorded provenance or scene changed during extraction")
        report = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                  "dataset": {"path": str(dataset_root), "format_version": "v3.0",
                              "lerobot_version": actual_version, "python_version": sys.version.split()[0],
                              "export_manifest_path": str(manifest_path), "export_manifest_sha256": manifest_hash},
                  "collection_metadata": {"path": str(collection_path), "sha256": collection_hash,
                                          "collection_fingerprint": collection["collection_fingerprint"]},
                  "source_snapshot": {"path": str(snapshot), "sha256": snapshot_hash, "verified": True},
                  "collection_config": collection["config"], "semantics": semantics,
                  "episode_end": episode_end,
                  "timing": {"fps": fps, "physics_hz": physics_hz, "capture_stride": stride,
                             "action_horizon_s": semantics["action_horizon_s"],
                             "first_sample_physics_time_s": offset,
                             "sample_time_origin": collection["sample_time_origin"],
                             "action_rule": collection["action_semantics"],
                             "replay_time_origin": "Initialize at recorded state[0]; replay t=0 corresponds to the first post-physics sample.",
                             "action_deadline": ("action[k] is the absolute TCP endpoint one sample after state[k]; "
                                 "the terminal row repeats the previous planned target and creates no extra replay interval."
                                 if episode_end == PRESS_ONLY_END else
                                 "action[k] is the absolute TCP endpoint one sample after state[k]; terminal target values are clamped to the recorded plan endpoint."),
                             "sample_physics_time": "sim_time + first_sample_physics_time_s; derived from the exact stored float64 clock."},
                  "control_source": "action", "numeric_reader": "LeRobotDataset.get_raw_item",
                  "numeric_read_mode": "HF formatting disabled in memory to preserve float64; no video decoding, resampling, or IK reconstruction.",
                  "numeric_source_files": list(numeric_files.values()),
                  "array_schema": {**{name: {"shape": ["N", *tail], "dtype": np.dtype(dtype).name,
                                            "source_feature": feature}
                                       for name, (feature, dtype, tail) in ARRAY_SOURCES.items()},
                                   "initial_joint_position": {"shape": [6], "dtype": "float32", "role": "initialization/first IK seed once"},
                                   "sample_physics_time": {"shape": ["N"], "dtype": "float64", "unit": "seconds", "role": "derived capture clock"}},
                  "selected_source_episode_ids": source_ids, "total_episodes": len(records),
                  "total_frames": sum(record["frames"] for record in records),
                  "floor_counts": dict(sorted(Counter(str(record["floor"]) for record in records).items())),
                  "episodes": records}
        write_json(staging / "manifest.json", report)
        staging.rename(output)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    print(f"Prepared {len(records)} replay episodes at {output}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/piper_elevator_lerobot_edge_30hz")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/replay_edge_30hz/input")
    parser.add_argument("--source-episode-ids", type=int, nargs="+", default=list(range(12)),
                        help="Source IDs, separated by spaces; default 0 through 11, one per floor")
    args = parser.parse_args()
    prepare(args.dataset, args.output, args.source_episode_ids)


if __name__ == "__main__":
    main()
