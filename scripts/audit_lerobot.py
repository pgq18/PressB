#!/usr/bin/env python3
"""Independently audit a local PiPER LeRobot v3 dataset and its raw provenance."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from export_lerobot import (CAMERAS, COLLECTION_METADATA, EPISODE_CONTEXT, PANEL_EPISODE_CONTEXT, FLOORS, IMAGE_SHAPE,
                           LEROBOT_VERSION, MANIFEST, SOURCE_FILES, SEMANTICS, TASKS, URDF,
                           collection_snapshot, collection_timing, configure_runtime, feature_spec,
                           load_arrays, read_json, require_version, semantics_for_collection, sha256,
                           validate_episode_collection, video_frames, write_json)


def audit_collection_metadata(dataset_root: Path, raw: Path, manifest: dict) -> tuple[dict, dict]:
    """Require the exported calibration bytes, provenance and source to agree.

    The saved absolute source path remains historical provenance. An explicitly
    relocated raw root is accepted only when its contents have the same hash.
    """
    payload, collection, actual = collection_snapshot(raw)
    declared = manifest.get("collection_metadata")
    if not isinstance(declared, dict):
        raise ValueError("Export manifest is missing collection_metadata provenance")
    if not collection.get("collection_fingerprint") or "raw_schema_version" not in collection:
        raise ValueError("Collection metadata is missing its schema or fingerprint")
    for key in ("path", "sha256", "collection_fingerprint", "raw_schema_version"):
        if declared.get(key) != actual[key]:
            raise ValueError(f"Export manifest collection_metadata {key} differs from the raw source")
    target = dataset_root / COLLECTION_METADATA
    if not target.is_file():
        raise ValueError(f"Exported collection metadata is missing: {target}")
    if target.read_bytes() != payload or sha256(target) != declared["sha256"]:
        raise ValueError(f"Exported collection metadata differs from its source or recorded hash: {target}")
    evidence = {**actual, "source_path_at_export": declared.get("source_path"),
                "exported_copy_sha256": sha256(target), "byte_identical_to_raw": True}
    return collection, evidence


def audit_episode_collection(entry: dict, metadata: dict, collection: dict, source: Path):
    """Reject different capture identities and changed per-episode coordinates."""
    validate_episode_collection(metadata, collection, source)
    for key in EPISODE_CONTEXT:
        if key not in entry or entry[key] != metadata.get(key):
            raise ValueError(f"Exported episode {key} differs from the raw metadata: {source}")
    for key in PANEL_EPISODE_CONTEXT:
        if (key in entry) != (key in metadata) or (key in metadata and entry[key] != metadata[key]):
            raise ValueError(f"Exported episode {key} differs from the raw metadata: {source}")


def _rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) * np.cos(angle) + (1 - np.cos(angle)) * np.outer(axis, axis) + np.sin(angle) * cross


class IndependentFK:
    """URDF FK using only XML and NumPy, independent of the collector's helper."""

    def __init__(self, path: Path):
        root = ET.parse(path).getroot()
        by_child = {joint.find("child").get("link"): joint for joint in root.findall("joint")}
        chain, child = [], "link6"
        while child != "base_link":
            joint = by_child[child]
            chain.append(joint)
            child = joint.find("parent").get("link")
            if len(chain) > len(by_child):
                raise ValueError("Cycle in URDF")
        self.chain, self.lower, self.upper = [], [], []
        for joint in reversed(chain):
            origin = joint.find("origin")
            xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
            rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
            transform = np.eye(4)
            transform[:3, :3] = (_rotation([0, 0, 1], rpy[2]) @ _rotation([0, 1, 0], rpy[1])
                                 @ _rotation([1, 0, 0], rpy[0]))
            transform[:3, 3] = xyz
            axis = None
            if joint.get("type") != "fixed":
                if joint.get("type") != "revolute":
                    raise ValueError("Unexpected arm joint type")
                axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                limit = joint.find("limit")
                self.lower.append(float(limit.get("lower")))
                self.upper.append(float(limit.get("upper")))
            self.chain.append((transform, axis))
        self.lower, self.upper = np.asarray(self.lower), np.asarray(self.upper)
        if len(self.lower) != 6:
            raise ValueError("Expected six Piper arm joints")

    def batch(self, q):
        q = np.asarray(q, dtype=float)
        transform = np.broadcast_to(np.eye(4), (len(q), 4, 4)).copy()
        joint_index = 0
        for origin, axis in self.chain:
            transform = transform @ origin
            if axis is not None:
                turns = np.broadcast_to(np.eye(4), transform.shape).copy()
                turns[:, :3, :3] = np.stack([_rotation(axis, angle) for angle in q[:, joint_index]])
                transform = transform @ turns
                joint_index += 1
        transform[:, :3, 3] += transform[:, :3, :3] @ np.array(SEMANTICS["tcp_offset_link6_m"])
        return transform


def quaternion_matrices(wxyz):
    w, x, y, z = np.asarray(wxyz, dtype=float).T
    return np.stack([
        1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w),
        2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w),
        2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y),
    ], axis=1).reshape(-1, 3, 3)


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def audit_dataset(dataset_root: Path, raw: Path | None = None, episodes_per_task: int | None = 100,
                  allow_partial: bool = False, decode_all: bool = True) -> dict:
    """Return a JSON-serializable report; any missing/invalid content fails closed."""
    require_version()
    configure_runtime()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset_root = Path(dataset_root).resolve()
    report = {"success": False, "dataset": str(dataset_root), "lerobot_version": LEROBOT_VERSION,
              "audited_utc": datetime.now(timezone.utc).isoformat(), "errors": [],
              "episodes": [], "video_files": [], "floor_counts": {str(f): 0 for f in FLOORS},
              "limits": ["Physics success is inherited from committed raw metadata; this audit validates dataset content and alignment."]}
    errors = report["errors"]

    def check(condition, message):
        if not bool(condition):
            errors.append(message)

    try:
        manifest = read_json(dataset_root / MANIFEST)
        check(manifest["tasks"] == TASKS, "Manifest task strings differ from required Press {f} floor.")
        check(manifest["lerobot_version"] == LEROBOT_VERSION, "Manifest LeRobot version mismatch")
        raw = Path(raw or manifest["raw_root"]).resolve()
        collection, collection_evidence = audit_collection_metadata(dataset_root, raw, manifest)
        timing = collection_timing(collection)
        fps = timing["fps"]
        semantics = semantics_for_collection(collection)
        check(manifest["semantics"] == semantics, "Manifest coordinate/action/timing semantics mismatch")
        report["collection_metadata"] = collection_evidence
        report["timing"] = timing
        if manifest.get("kind") == "aggregate":
            from audit_raw_dataset import audit as audit_raw_physics
            physics = audit_raw_physics(raw, episodes_per_task or 100, allow_partial=allow_partial)
            write_json(raw / "physics_audit.json", physics)
            write_json(dataset_root / "meta/physics_audit.json", physics)
            report["raw_physics_audit"] = {key: value for key, value in physics.items() if key != "episodes"}
            check(physics["success"], f"Independent raw physics audit failed: {physics['errors'][:5]}")
            report["limits"] = ["Physics evidence and dataset content are checked separately; ideal Isaac RGB data does not model hardware sensor noise."]
        fk = IndependentFK(URDF)
        check(sha256(URDF) == manifest["urdf"]["sha256"], "Source URDF hash mismatch")
        dataset = LeRobotDataset(repo_id="local/piper_elevator_1200", root=dataset_root, video_backend="pyav")
        check(dataset.meta.info.codebase_version == "v3.0", "Dataset format must be v3.0")
        check(dataset.fps == fps, f"Dataset fps differs from collection rate {fps}")
        check(dataset.meta.robot_type == "piper", "Unexpected robot_type")
        check(len(dataset.meta.tasks) == 12, "Task table must contain exactly 12 tasks")
        for index, task in enumerate(TASKS):
            check(dataset.meta.get_task_index(task) == index, f"Task mapping mismatch for {task}")
        for key, specification in feature_spec().items():
            actual = dataset.features.get(key, {})
            check(actual.get("dtype") == specification["dtype"] and
                  tuple(actual.get("shape", ())) == specification["shape"] and
                  actual.get("names") == specification["names"], f"Feature schema mismatch: {key}")
        entries = manifest["episodes"]
        check(len(entries) == dataset.num_episodes, "Manifest and official reader episode counts differ")
        ids = [entry["episode_id"] for entry in entries]
        check(len(set(ids)) == len(ids), "Duplicate source_episode_id in manifest")
        check(dataset.num_episodes > 0, "Dataset is empty")
        video_counts, video_ranges = {}, {}
        expected_start, total_sampled = 0, 0
        # The default HF torch transform casts Python floats to float32. Inspect
        # Arrow's original values so the separately stored float64 clock is audited.
        numeric_table = dataset.hf_dataset.with_format(None)
        check(len(numeric_table) == len(dataset), "Parquet row count differs from dataset metadata")
        for episode_index, entry in enumerate(entries):
            try:
                floor = int(entry["floor"])
                check(floor in FLOORS, f"Invalid floor in episode {episode_index}")
                report["floor_counts"][str(floor)] += 1
                directory = raw / f"episode_{entry['episode_id']:06d}"
                metadata = read_json(directory / "metadata.json")
                audit_episode_collection(entry, metadata, collection, directory)
                check(metadata.get("success") is True, f"Raw episode {entry['episode_id']} is not successful")
                check(int(metadata["episode_id"]) == entry["episode_id"] and int(metadata["floor"]) == floor
                      and int(metadata["seed"]) == entry["seed"], f"Source metadata mismatch: {directory}")
                check(entry["task"] == TASKS[floor - 24], f"Manifest episode task mismatch: {episode_index}")
                check(set(entry["source_sha256"]) == set(SOURCE_FILES), f"Incomplete source hashes: {directory}")
                for key in ("pose_frame", "pose_link", "tcp_offset_link6_m", "press_tip_offset_link6_m"):
                    check(key not in metadata or metadata[key] == semantics[key],
                          f"Source coordinate declaration differs for {key}: {directory}")
                for name, expected_hash in entry["source_sha256"].items():
                    check(sha256(directory / name) == expected_hash, f"Raw hash mismatch: {directory / name}")
                arrays = load_arrays(directory, fps)
                count = len(arrays["state"])
                meta = dataset.meta.episodes[episode_index]
                start, stop = int(meta["dataset_from_index"]), int(meta["dataset_to_index"])
                check(start == expected_start and stop - start == count == int(meta["length"]) == entry["frames"],
                      f"Episode range/length mismatch: {episode_index}")
                check(int(meta["episode_index"]) == episode_index, f"Episode metadata index mismatch: {episode_index}")
                check(meta["tasks"] == [TASKS[floor - 24]], f"Episode task metadata mismatch: {episode_index}")
                expected_start = stop
                rows = numeric_table[start:stop]
                values = {key: _numpy(value) for key, value in rows.items()}
                for key, source in (("observation.state", "state"), ("action", "action"),
                                    ("observation.joint_position", "q_actual"), ("action.joint_target", "q_target")):
                    actual = values[key]
                    check(np.isfinite(actual).all(), f"Nonfinite {key}: episode {episode_index}")
                    check(actual.shape == arrays[source].shape and
                          np.array_equal(actual, arrays[source].astype(np.float32)),
                          f"Raw numeric round trip mismatch {key}: episode {episode_index}")
                check(np.allclose(values["observation.sim_time"].reshape(-1), arrays["sim_time"], atol=1e-9, rtol=0),
                      f"Raw timestamp round trip mismatch: episode {episode_index}")
                check(np.allclose(values["timestamp"].reshape(-1), np.arange(count) / fps, atol=2e-6, rtol=0),
                      f"LeRobot timestamps mismatch: episode {episode_index}")
                for key, expected in (("frame_index", np.arange(count)), ("index", np.arange(start, stop)),
                                      ("episode_index", episode_index), ("task_index", floor - 24),
                                      ("source_episode_id", entry["episode_id"]), ("source_seed", entry["seed"]),
                                      ("floor", floor)):
                    check(np.all(values[key].reshape(-1) == expected), f"Incorrect {key}: episode {episode_index}")
                pose_errors = {}
                for pose_key, joint_key in (("state", "q_actual"), ("action", "q_target")):
                    q = arrays[joint_key]
                    check(np.all(q >= fk.lower - 0.005) and np.all(q <= fk.upper + 0.005),
                          f"Joint limit violation {joint_key}: episode {episode_index}")
                    transform = fk.batch(q)
                    pose = arrays[pose_key]
                    position_error = float(np.max(np.linalg.norm(transform[:, :3, 3] - pose[:, :3], axis=1)))
                    matrix_error = float(np.max(np.abs(transform[:, :3, :3] - quaternion_matrices(pose[:, 3:7]))))
                    check(position_error <= 2e-5 and matrix_error <= 2e-5,
                          f"Base-frame TCP FK mismatch {pose_key}: episode {episode_index} ({position_error}m, R {matrix_error})")
                    pose_errors[pose_key] = {"max_position_error_m": position_error, "max_rotation_matrix_error": matrix_error}
                check(np.allclose(arrays["action"][:, 7], 0.008, atol=1e-6, rtol=0),
                      f"Commanded total gripper width must be .008m: episode {episode_index}")
                samples = sorted({start, (start + stop - 1) // 2, stop - 1})
                raw_sample_indices = {index - start for index in samples}
                raw_images = {
                    key: {index: image for index, image in enumerate(video_frames(directory / filename, fps))
                          if index in raw_sample_indices}
                    for key, filename in CAMERAS.items()
                }
                image_errors = []
                for index in samples:
                    row = dataset[index]  # Official decoder resolves episode offsets inside shared videos.
                    check(row["task"] == TASKS[floor - 24], f"Official reader task mismatch at frame {index}")
                    for key in CAMERAS:
                        image = _numpy(row[key])
                        check(image.shape == (3, IMAGE_SHAPE[0], IMAGE_SHAPE[1]) and np.isfinite(image).all()
                              and image.min() >= 0 and image.max() <= 1,
                              f"Official reader RGB invalid: frame {index}, {key}")
                        reference = raw_images[key][index - start].astype(np.float32)
                        difference = image.transpose(1, 2, 0) * 255 - reference
                        mae = float(np.mean(np.abs(difference)))
                        mse = float(np.mean(difference * difference))
                        psnr = float(10 * np.log10(255**2 / max(mse, 1e-12)))
                        # Re-encoding is lossy. These thresholds allow H264 CRF18
                        # artifacts while detecting channel/camera/episode mixups.
                        check(mae <= 8 and psnr >= 25,
                              f"Decoded RGB does not match source: frame {index}, {key}, MAE={mae:.3f}, PSNR={psnr:.2f}")
                        image_errors.append({"frame_index": index, "camera": key, "source_mae_255": mae,
                                             "source_psnr_db": psnr})
                    total_sampled += 1
                for key in CAMERAS:
                    path = dataset_root / dataset.meta.get_video_file_path(episode_index, key)
                    video_counts[path] = video_counts.get(path, 0) + count
                    from_time, to_time = float(meta[f"videos/{key}/from_timestamp"]), float(meta[f"videos/{key}/to_timestamp"])
                    check(abs(to_time - from_time - count / fps) < 2e-4,
                          f"Video episode duration mismatch: episode {episode_index}, {key}")
                    video_ranges.setdefault(path, []).append((from_time, to_time))
                report["episodes"].append({"episode_index": episode_index, "source_episode_id": entry["episode_id"],
                                           "floor": floor, "frames": count, "official_reader_sample_indices": samples,
                                           "base_frame_fk": pose_errors, "rgb_source_comparison": image_errors})
            except Exception as exc:
                errors.append(f"Episode {episode_index}: {type(exc).__name__}: {exc}")
        check(expected_start == len(dataset), "Last episode endpoint differs from dataset frame count")
        report["total_episodes"] = dataset.num_episodes
        report["total_frames"] = len(dataset)
        report["official_reader_sampled_frames"] = total_sampled
        report["official_reader_sampled_images"] = 2 * total_sampled
        for floor, count in report["floor_counts"].items():
            if episodes_per_task is not None:
                check(count <= episodes_per_task if allow_partial else count == episodes_per_task,
                      f"Floor {floor}: expected {'at most' if allow_partial else 'exactly'} {episodes_per_task}, got {count}")
        check(set((dataset_root / "videos").rglob("*.mp4")) == set(video_counts),
              "Unreferenced or missing video files in dataset")
        for path, count in sorted(video_counts.items()):
            try:
                ranges = sorted(video_ranges[path])
                check(abs(ranges[0][0]) < 2e-4 and all(abs(a[1] - b[0]) < 2e-4 for a, b in zip(ranges, ranges[1:])),
                      f"Gaps or overlapping episode intervals in video {path}")
                decoded = sum(1 for _ in video_frames(path, fps)) if decode_all else None
                if decode_all:
                    check(decoded == count, f"Full video decode frame mismatch: {path}: {decoded} != {count}")
                report["video_files"].append({"path": str(path.relative_to(dataset_root)), "sha256": sha256(path),
                                              "expected_frames": count, "decoded_frames": decoded})
            except Exception as exc:
                errors.append(f"Video {path}: {type(exc).__name__}: {exc}")
        report["full_video_decode"] = decode_all
        report["manifest_sha256"] = sha256(dataset_root / MANIFEST)
        report["success"] = not errors
    except Exception as exc:
        errors.append(f"Dataset: {type(exc).__name__}: {exc}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", type=Path)
    parser.add_argument("--output", type=Path, help="Dataset root (alternative to positional argument)")
    parser.add_argument("--raw", type=Path, help="Relocated raw episode root; defaults to saved provenance")
    parser.add_argument("--episodes-per-task", type=int, default=100)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    dataset = args.dataset or args.output
    if dataset is None or (args.dataset and args.output):
        parser.error("Provide exactly one dataset root, as a positional argument or --output")
    if args.episodes_per_task < 1:
        parser.error("--episodes-per-task must be positive")
    result = audit_dataset(dataset, args.raw, args.episodes_per_task, args.allow_partial, decode_all=True)
    report_path = args.report or dataset / "meta/audit.json"
    write_json(report_path, result)
    print(f"{'PASS' if result['success'] else 'FAIL'}: {result.get('total_episodes', 0)} episodes; "
          f"floor counts={result['floor_counts']}; report={report_path}")
    for error in result["errors"][:30]:
        print(f"  {error}")
    raise SystemExit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
