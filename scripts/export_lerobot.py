#!/usr/bin/env python3
"""Convert committed PiPER raw episodes to local, resumable LeRobot v3 shards.

Run with the separate LeRobot 0.6.1 environment, never the Isaac environment.
No Hub upload or authentication is performed. A root directory is published only
after the official writer has finalized and the independent audit has passed.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pressb.panel_metadata import (PANEL_EPISODE_CONTEXT, episode_panel_context,
                                  validate_episode_panel_metadata)
LEROBOT_VERSION = "0.6.1"
FPS = 10
PHYSICS_HZ = 120
IMAGE_SHAPE = (480, 640, 3)
FLOORS = tuple(range(24, 36))
TASKS = [f"Press {floor} floor." for floor in FLOORS]
POSE_NAMES = ["x_m", "y_m", "z_m", "qw", "qx", "qy", "qz", "gripper_width_m"]
CAMERAS = {"observation.images.wrist": "wrist.mp4", "observation.images.global": "global.mp4"}
SOURCE_FILES = ("metadata.json", "frames.npz", "physics.npz", "wrist.mp4", "global.mp4")
MANIFEST = "meta/export_manifest.json"
COLLECTION_METADATA = "meta/collection_metadata.json"
EPISODE_CONTEXT = ("raw_schema_version", "collection_fingerprint", "env_offset_m", "robot_base_world_m")
URDF = ROOT / "vendor/robot_lab/source/robot_lab/data/Robots/Agilex/PIPER/piper_description.urdf"
SEMANTICS = {
    "pose_frame": "base_link", "pose_link": "gripper_tcp",
    "tcp_offset_link6_m": [0.0, 0.0, 0.1358],
    "press_tip_offset_link6_m": [0.0, 0.0, 0.24],
    "pose_names": POSE_NAMES, "quaternion_order": "wxyz",
    "action_type": "absolute_gripper_tcp_pose_and_total_opening",
    "action_horizon_s": 0.1,
    "action_joint_target": "next_100ms_endpoint_clamped_at_episode_end",
    "fps": FPS, "image_shape_hwc": list(IMAGE_SHAPE),
}


def collection_timing(collection: dict) -> dict:
    """Resolve immutable timing, retaining the exact schema 7/9 defaults.

    Schema 10 records every timing field explicitly. Older captures were fixed
    at 10 Hz and may omit the stride/horizon fields; a different fps cannot be
    retroactively assigned to those captures.
    """
    fps = collection.get("fps")
    if type(fps) is not int or fps <= 0 or PHYSICS_HZ % fps:
        raise ValueError("Collection fps must be a positive integer divisor of 120")
    schema = collection.get("raw_schema_version", 9)
    if type(schema) is not int or schema < 1:
        raise ValueError("Collection raw_schema_version must be a positive integer")
    if schema < 10 and fps != FPS:
        raise ValueError("Legacy collection schemas require 10 fps")
    expected = {"fps": fps, "physics_hz": PHYSICS_HZ,
                "capture_stride": PHYSICS_HZ // fps, "action_horizon_s": 1 / fps}
    for key in ("physics_hz", "capture_stride", "action_horizon_s"):
        if key not in collection:
            if schema >= 10:
                raise ValueError(f"Collection schema {schema} is missing {key}")
            continue
        value = collection[key]
        valid = (type(value) in (int, float) and np.isfinite(value)
                 and abs(value - expected[key]) <= 1e-12) if key == "action_horizon_s" else (
                     type(value) is int and value == expected[key])
        if not valid:
            raise ValueError(f"Collection {key} is incompatible with {fps} fps")
    return expected


def semantics_for_collection(collection: dict) -> dict:
    timing = collection_timing(collection)
    # Never mutate SEMANTICS: legacy manifests must remain byte-for-byte
    # comparable when resuming old 10 Hz exports in the same Python process.
    result = {**SEMANTICS, "fps": timing["fps"], "action_horizon_s": timing["action_horizon_s"]}
    if collection.get("raw_schema_version", 9) >= 10:
        result["action_joint_target"] = "next_sample_endpoint_clamped_at_episode_end"
    return result


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def collection_snapshot(raw: Path) -> tuple[bytes, dict, dict]:
    """Read the immutable calibration/configuration as one byte-identical snapshot."""
    source = raw / "collection_metadata.json"
    payload = source.read_bytes()
    metadata = json.loads(payload)
    semantics = semantics_for_collection(metadata)
    if metadata.get("task_texts") != TASKS:
        raise ValueError(f"Collection task vocabulary is incompatible: {source}")
    for key in ("pose_frame", "pose_link", "tcp_offset_link6_m", "press_tip_offset_link6_m"):
        if metadata.get(key) != semantics[key]:
            raise ValueError(f"Collection {key} is incompatible: {source}")
    for view in ("wrist", "global"):
        calibration = metadata.get(view, {})
        intrinsic = np.asarray(calibration.get("K", []), dtype=float)
        if (calibration.get("width") != IMAGE_SHAPE[1] or calibration.get("height") != IMAGE_SHAPE[0]
                or intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all()
                or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0):
            raise ValueError(f"Missing or invalid embedded {view} calibration: {source}")
        sensor = calibration.get("sensor", {})
        keys = (("optical_position_link6", "optical_quaternion_wxyz_link6") if view == "wrist"
                else ("fixed_position_world", "fixed_quaternion_wxyz_world"))
        position, quaternion = (np.asarray(sensor.get(key, []), dtype=float) for key in keys)
        if (position.shape != (3,) or quaternion.shape != (4,) or not np.isfinite(position).all()
                or not np.isfinite(quaternion).all() or abs(np.linalg.norm(quaternion) - 1) > 2e-5):
            raise ValueError(f"Missing or invalid embedded {view} extrinsics: {source}")
    fingerprint = metadata.get("collection_fingerprint")
    if fingerprint is not None:
        identity = metadata.get("identity")
        if not isinstance(identity, dict):
            raise ValueError(f"Collection fingerprint has no identity: {source}")
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"),
                                           allow_nan=False).encode()).hexdigest()
        if fingerprint != digest or any(metadata.get(key) != value for key, value in identity.items()):
            raise ValueError(f"Collection fingerprint does not match its identity: {source}")
    record = {"path": COLLECTION_METADATA, "source_path": str(source.resolve()),
              "sha256": hashlib.sha256(payload).hexdigest(),
              "collection_fingerprint": fingerprint, "raw_schema_version": metadata.get("raw_schema_version")}
    return payload, metadata, record


def validate_episode_collection(metadata: dict, collection: dict, source: Path):
    """Reject mixed capture settings even when both episodes have valid shapes."""
    for key in ("raw_schema_version", "collection_fingerprint"):
        if key in collection and metadata.get(key) != collection[key]:
            raise ValueError(f"Episode {key} does not match collection metadata: {source}")
    timing = collection_timing(collection)
    episode_timing = collection_timing(metadata)
    for key, expected in timing.items():
        if episode_timing[key] != expected:
            raise ValueError(f"Episode {key} does not match collection metadata: {source}")
    for key in ("env_offset_m", "robot_base_world_m"):
        vector = np.asarray(metadata.get(key, []), dtype=float)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError(f"Missing or invalid {key}: {source}")
    validate_episode_panel_metadata(collection, metadata)


def copy_collection_metadata(raw: Path, dataset: Path, expected: dict):
    payload, _, record = collection_snapshot(raw)
    if record != expected:
        raise ValueError("Collection metadata changed during export; refusing to mix calibrations")
    target = dataset / COLLECTION_METADATA
    if target.exists():
        if sha256(target) != record["sha256"]:
            raise ValueError(f"Exported collection metadata was modified: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(target)


def require_version():
    actual = version("lerobot")
    if actual != LEROBOT_VERSION:
        raise RuntimeError(f"Use the isolated lerobot=={LEROBOT_VERSION} environment, found {actual}")


def configure_runtime():
    """Keep CPU conversion, caches and all Hub access separate from Isaac."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/hf"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(Path(os.environ["HF_HOME"]) / "datasets"))
    os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    import av
    av.logging.set_level(av.logging.ERROR)


def feature_spec():
    features = {
        key: {"dtype": "video", "shape": IMAGE_SHAPE,
              "names": ["height", "width", "channels"]}
        for key in CAMERAS
    }
    for key in ("observation.state", "action"):
        features[key] = {"dtype": "float32", "shape": (8,), "names": POSE_NAMES}
    for key in ("observation.joint_position", "action.joint_target"):
        features[key] = {"dtype": "float32", "shape": (6,),
                         "names": [f"joint{i}" for i in range(1, 7)]}
    features["observation.sim_time"] = {"dtype": "float64", "shape": (1,), "names": ["seconds"]}
    for key in ("source_episode_id", "source_seed", "floor"):
        features[key] = {"dtype": "int64", "shape": (1,), "names": None}
    return features


def discover_raw(raw: Path, limit: int) -> list[dict]:
    """Only final, atomically published episode_NNNNNN directories are accepted."""
    episodes = []
    _, collection, _ = collection_snapshot(raw)
    counts = {f: 0 for f in FLOORS}
    for directory in sorted(raw.glob("episode_*")):
        if not directory.is_dir() or not re.fullmatch(r"episode_[0-9]{6,}", directory.name):
            continue
        metadata = read_json(directory / "metadata.json")
        if metadata.get("success") is not True:
            continue
        validate_episode_collection(metadata, collection, directory)
        episode_id, floor = int(metadata["episode_id"]), int(metadata["floor"])
        if episode_id != int(directory.name.removeprefix("episode_")) or floor not in FLOORS:
            raise ValueError(f"Invalid episode ID or floor in {directory}")
        for name in SOURCE_FILES:
            if not (directory / name).is_file():
                raise ValueError(f"Committed episode is incomplete: {directory / name}")
        for key in ("pose_frame", "pose_link", "tcp_offset_link6_m", "press_tip_offset_link6_m"):
            if key in metadata and metadata[key] != SEMANTICS[key]:
                raise ValueError(f"Unexpected {key} in {directory}: {metadata[key]!r}")
        counts[floor] += 1
        if counts[floor] > limit:
            raise ValueError(f"More than {limit} successful raw episodes for floor {floor}; select one raw root")
        episodes.append({"episode_id": episode_id, "floor": floor, "seed": int(metadata["seed"]),
                         "directory": str(directory.resolve()), "metadata": metadata})
    if len({x["episode_id"] for x in episodes}) != len(episodes):
        raise ValueError("Duplicate source_episode_id")
    return sorted(episodes, key=lambda x: x["episode_id"])


def load_arrays(directory: Path, fps: int) -> dict[str, np.ndarray]:
    with np.load(directory / "frames.npz", allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in ("state", "action", "sim_time", "q_actual", "q_target")}
    count = len(arrays["state"])
    if count < 2:
        raise ValueError(f"Episode has fewer than two frames: {directory}")
    for key, tail in (("state", (8,)), ("action", (8,)), ("q_actual", (6,)), ("q_target", (6,)), ("sim_time", ())):
        value = arrays[key]
        if value.shape != (count,) + tail or not np.isfinite(value).all():
            raise ValueError(f"Invalid shape/non-finite {key} in {directory}: {value.shape}")
    if not np.allclose(np.diff(arrays["sim_time"]), 1 / fps, rtol=0, atol=2e-6):
        raise ValueError(f"Raw simulator timestamps are not consecutive {fps} Hz samples: {directory}")
    for key in ("state", "action"):
        pose = arrays[key]
        if np.max(np.abs(np.linalg.norm(pose[:, 3:7], axis=1) - 1)) > 2e-5:
            raise ValueError(f"Non-unit wxyz quaternion in {directory}/{key}")
        if np.any((pose[:, 7] < 0) | (pose[:, 7] > 0.1)):
            raise ValueError(f"Invalid total gripper opening in {directory}/{key}")
    return arrays


def video_frames(path: Path, fps: int):
    """Decode every frame and check its actual presentation time, shape and fps."""
    import av
    with av.open(str(path)) as container:
        if len(container.streams.video) != 1:
            raise ValueError(f"Expected exactly one video stream: {path}")
        stream = container.streams.video[0]
        if stream.average_rate is None or abs(float(stream.average_rate) - fps) > 1e-6:
            raise ValueError(f"Expected {fps} fps: {path}, found {stream.average_rate}")
        first_time = None
        for index, frame in enumerate(container.decode(stream)):
            if frame.pts is None or frame.time_base is None:
                raise ValueError(f"Video has no presentation timestamps: {path}")
            timestamp = float(frame.pts * frame.time_base)
            if first_time is None:
                first_time = timestamp
            if abs(timestamp - first_time - index / fps) > 1e-4:
                raise ValueError(f"Nonconsecutive video timestamp in {path} at frame {index}")
            image = frame.to_ndarray(format="rgb24")
            if image.shape != IMAGE_SHAPE or image.dtype != np.uint8:
                raise ValueError(f"Unexpected RGB frame in {path}: {image.shape}/{image.dtype}")
            yield image


def entry_provenance(episode: dict, count: int) -> dict:
    path = Path(episode["directory"])
    return {"episode_id": episode["episode_id"], "floor": episode["floor"], "seed": episode["seed"],
            "task": TASKS[episode["floor"] - 24], "frames": count,
            "raw_directory": str(path),
            **{key: episode["metadata"][key] for key in EPISODE_CONTEXT if key in episode["metadata"]},
            **episode_panel_context(episode["metadata"]),
            "source_sha256": {name: sha256(path / name) for name in SOURCE_FILES}}


def manifest_for(entries: list[dict], kind: str, raw: Path) -> dict:
    _, collection, record = collection_snapshot(raw)
    return {"schema_version": 1, "kind": kind, "lerobot_version": LEROBOT_VERSION,
            "format_version": "v3.0", "created_utc": datetime.now(timezone.utc).isoformat(),
            "raw_root": str(raw.resolve()), "tasks": TASKS, "semantics": semantics_for_collection(collection),
            "collection_metadata": record,
            "urdf": {"path": str(URDF), "sha256": sha256(URDF)},
            "episodes": entries}


def write_dataset_readme(dataset: Path, manifest: dict, audit: dict):
    """Create only a missing dataset card, and only after a successful full audit."""
    if not audit.get("success") or not audit.get("full_video_decode"):
        raise ValueError("Cannot describe an unverified dataset as published")
    target = dataset / "README.md"
    if target.exists():
        return
    entries = manifest["episodes"]
    fps = manifest["semantics"]["fps"]
    horizon = manifest["semantics"]["action_horizon_s"]
    counts = {floor: sum(entry["floor"] == floor for entry in entries) for floor in FLOORS}
    count, frames = len(entries), sum(entry["frames"] for entry in entries)
    if count != audit.get("total_episodes") or frames != audit.get("total_frames"):
        raise ValueError("Dataset card counts disagree with the independent audit")
    lines = ["# PiPER elevator button demonstrations", "",
             f"This local LeRobot v3.0 dataset contains **{count} episodes and {frames:,} frames** "
             f"at {fps} Hz. Both `observation.images.wrist` and `observation.images.global` "
             "contain 640 × 480 RGB video from simulated D435 pinhole cameras. "
             "Depth images are not included.", "",
             "| Task | Episodes |", "| --- | ---: |"]
    lines.extend(f"| {TASKS[floor - 24]} | {counts[floor]} |" for floor in FLOORS)
    lines.extend(["", "`observation.state` and `action` each contain "
                  "`[x_m, y_m, z_m, qw, qx, qy, qz, gripper_width_m]` in `base_link`. "
                  "The pose is the gripper TCP at link6 local `[0, 0, 0.1358]` m; "
                  "the pressing tip at 0.24 m is a different point. State is measured; action "
                  f"is the absolute planned target one sample ({horizon:.9g} s) ahead, clamped to the episode end. "
                  "Gripper width is the total opening in metres.", "",
                  "Episodes follow committed shard order. Parallel collectors may finish source IDs "
                  "out of order, so `episode_index` is not `source_episode_id`. Every frame retains "
                  "the source ID, seed and floor; `meta/export_manifest.json` records the explicit "
                  "episode-index mapping, source file hashes and each environment's translation.", "",
                  "`meta/collection_metadata.json` is a byte-identical copy of the collection settings "
                  "and complete embedded calibration. `wrist.sensor` stores the optical pose relative "
                  "to link6; `global.sensor` stores the optical pose in the reference world's frame. "
                  "Add an episode's `env_offset_m` to that global camera position for its cloned "
                  "environment. `robot_base_world_m` locates its base. Camera axes are +X right, "
                  "+Y up, −Z forward, with wxyz quaternions. These are ideal Isaac images, not "
                  "the hardware RealSense noise or stereo pipeline.", "",
                  "`meta/audit.json` records the successful independent check of numeric data, "
                  "kinematics, official-reader first/middle/last RGB samples and all decoded video "
                  "frame counts. `meta/physics_audit.json` records the raw 120 Hz physics checks. "
                  "Source-image correspondence is checked against the captured raw videos; "
                  "this is not an independent pixel-level validation of rendered button lighting.", "",
                  "Read locally with the separate `lerobot==0.6.1` environment:", "", "```python",
                  "import os", "from pathlib import Path", "os.environ['HF_HUB_OFFLINE'] = '1'",
                  "from lerobot.datasets.lerobot_dataset import LeRobotDataset", "",
                  "# Run from this dataset directory.",
                  "dataset = LeRobotDataset('local/piper_elevator', root=Path('.').resolve(),",
                  "                         video_backend='pyav')", "sample = dataset[0]", "```", ""])
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text("\n".join(lines))
    temporary.replace(target)


def discard_unpublished(path: Path):
    """Remove only recognizable unfinished converter output, never raw data."""
    if not path.exists():
        return
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"Refusing to remove unexpected staging path: {path}")
    if any(path.iterdir()):
        info_path = path / "meta/info.json"
        info = read_json(info_path) if info_path.is_file() else {}
        if (info.get("robot_type") != "piper" or info.get("codebase_version") != "v3.0"
                or not set(feature_spec()) <= set(info.get("features", {}))):
            raise ValueError(f"Refusing to remove unrecognized staging directory: {path}")
    shutil.rmtree(path)


def write_part(episodes: list[dict], destination: Path, raw: Path):
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from audit_lerobot import audit_dataset

    _, collection, collection_record = collection_snapshot(raw)
    fps = collection_timing(collection)["fps"]
    for episode in episodes:
        validate_episode_collection(episode["metadata"], collection, Path(episode["directory"]))
    staging = destination.with_name(destination.name + ".inprogress")
    discard_unpublished(staging)
    # LeRobot 0.6.1's live recorder drops frames after a 0.1-second timeout
    # on a full streaming queue. Offline conversion must never do that: reserve
    # enough slots for an entire episode plus the end sentinel, even if an
    # encoder thread is descheduled. Queues are reused/reset between episodes.
    queue_capacity = max(len(load_arrays(Path(episode["directory"]), fps)["state"])
                         for episode in episodes) + 1
    dataset = LeRobotDataset.create(
        repo_id=f"local/piper_{destination.name}", root=staging, fps=fps, robot_type="piper",
        features=feature_spec(), video_backend="pyav", use_videos=True,
        rgb_encoder=RGBEncoderConfig(vcodec="h264", pix_fmt="yuv420p", crf=18, g=2, preset="ultrafast"),
        streaming_encoding=True, encoder_threads=4, encoder_queue_maxsize=queue_capacity,
        metadata_buffer_size=1,
    )
    dataset.meta.save_episode_tasks(TASKS)
    entries = []
    try:
        for episode in episodes:
            directory = Path(episode["directory"])
            arrays = load_arrays(directory, fps)
            count = len(arrays["state"])
            generators = {key: video_frames(directory / name, fps) for key, name in CAMERAS.items()}
            try:
                for index in range(count):
                    frame = {}
                    for key, generator in generators.items():
                        try:
                            frame[key] = next(generator)
                        except StopIteration as exc:
                            raise ValueError(f"Short video {directory}/{CAMERAS[key]} at frame {index}") from exc
                    frame.update({
                        "observation.state": arrays["state"][index].astype(np.float32),
                        "action": arrays["action"][index].astype(np.float32),
                        "observation.joint_position": arrays["q_actual"][index].astype(np.float32),
                        "action.joint_target": arrays["q_target"][index].astype(np.float32),
                        "observation.sim_time": np.array([arrays["sim_time"][index]], dtype=np.float64),
                        "source_episode_id": np.array([episode["episode_id"]], dtype=np.int64),
                        "source_seed": np.array([episode["seed"]], dtype=np.int64),
                        "floor": np.array([episode["floor"]], dtype=np.int64),
                        "task": TASKS[episode["floor"] - 24],
                    })
                    dataset.add_frame(frame)
                for key, generator in generators.items():
                    if next(generator, None) is not None:
                        raise ValueError(f"Video longer than numeric arrays: {directory}/{CAMERAS[key]}")
                dataset.save_episode()
            finally:
                for generator in generators.values():
                    generator.close()
            entries.append(entry_provenance(episode, count))
            print(f"Saved raw episode {episode['episode_id']:06d}, floor {episode['floor']}, {count} frames", flush=True)
    finally:
        dataset.finalize()
    manifest = manifest_for(entries, "part", raw)
    manifest["collection_metadata"] = collection_record
    manifest["encoding"] = {"preset": "ultrafast", "encoder_threads_per_camera": 4,
                            "encoder_queue_maxsize": queue_capacity,
                            "queue_policy": "capacity_exceeds_largest_complete_episode_no_timeout_drops"}
    copy_collection_metadata(raw, staging, collection_record)
    write_json(staging / MANIFEST, manifest)
    result = audit_dataset(staging, raw=raw, episodes_per_task=None, decode_all=True)
    result["dataset"] = str(destination)
    write_json(staging / "meta/audit.json", result)
    if not result["success"]:
        raise RuntimeError(f"Part failed independent audit: {staging}: {result['errors'][:5]}")
    staging.rename(destination)


def existing_parts(parts_root: Path, raw: Path):
    result, seen = [], set()
    _, collection, collection_record = collection_snapshot(raw)
    semantics = semantics_for_collection(collection)
    for directory in sorted(parts_root.glob("part_*")):
        if not directory.is_dir() or not re.fullmatch(r"part_[0-9]{5}", directory.name):
            continue
        manifest = read_json(directory / MANIFEST)
        if (manifest["lerobot_version"] != LEROBOT_VERSION or manifest["semantics"] != semantics
                or manifest["tasks"] != TASKS or Path(manifest["raw_root"]) != raw):
            raise ValueError(f"Incompatible committed shard: {directory}")
        if "collection_metadata" in manifest:
            if manifest["collection_metadata"] != collection_record:
                raise ValueError(f"Committed shard belongs to a different collection: {directory}")
            if sha256(directory / COLLECTION_METADATA) != collection_record["sha256"]:
                raise ValueError(f"Committed shard calibration was modified: {directory}")
        for entry in manifest["episodes"]:
            if entry["episode_id"] in seen:
                raise ValueError(f"Duplicate committed episode: {entry['episode_id']}")
            seen.add(entry["episode_id"])
            source = raw / f"episode_{entry['episode_id']:06d}"
            validate_episode_collection(read_json(source / "metadata.json"), collection, source)
            for name, expected in entry["source_sha256"].items():
                if sha256(source / name) != expected:
                    raise ValueError(f"Committed source was modified: {source / name}")
        result.append((directory, manifest))
    return result, seen


def publish_dataset(parts: list[tuple[Path, dict]], output: Path, raw: Path,
                    episodes_per_task: int, allow_partial: bool):
    from lerobot.datasets.aggregate import aggregate_datasets
    from audit_lerobot import audit_dataset

    _, collection, collection_record = collection_snapshot(raw)
    semantics = semantics_for_collection(collection)
    entries = []
    for directory, part_manifest in parts:
        if part_manifest.get("semantics") != semantics:
            raise ValueError(f"Shard timing or pose semantics differ from the current collection: {directory}")
        if ("collection_metadata" in part_manifest
                and part_manifest["collection_metadata"] != collection_record):
            raise ValueError(f"Shard collection differs from the current calibration: {directory}")
        for saved in part_manifest["episodes"]:
            entry = dict(saved)
            source = raw / f"episode_{entry['episode_id']:06d}"
            metadata = read_json(source / "metadata.json")
            validate_episode_collection(metadata, collection, source)
            # Also make aggregates from older shards self-contained. The audit
            # checks the source metadata hash before this aggregate is published.
            for key in (*EPISODE_CONTEXT, *PANEL_EPISODE_CONTEXT):
                if key in metadata:
                    if key in entry and entry[key] != metadata[key]:
                        raise ValueError(f"Shard episode context differs from the source: {source}/{key}")
                    entry[key] = metadata[key]
            entries.append(entry)
    if not entries:
        raise ValueError("No successful committed episodes to export")
    source_ids = [entry["episode_id"] for entry in entries]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("Cannot publish duplicate source episode IDs")
    order = {"policy": "part_commit_order_then_part_local_order",
             "source_id_column": "source_episode_id",
             "source_ids_globally_sorted": source_ids == sorted(source_ids),
             "episode_index_to_source_episode_id": source_ids}
    old_manifest = read_json(output / MANIFEST) if output.exists() else None
    if old_manifest is not None:
        if (old_manifest.get("raw_root") != str(raw) or old_manifest.get("semantics") != semantics):
            raise ValueError(f"Refusing to replace unrelated output: {output}")
        if ("collection_metadata" in old_manifest
                and old_manifest["collection_metadata"] != collection_record):
            raise ValueError(f"Existing output belongs to a different collection: {output}")
        if old_manifest["episodes"] == entries:
            copy_collection_metadata(raw, output, collection_record)
            old_manifest["collection_metadata"] = collection_record
            old_manifest["episode_order"] = order
            write_json(output / MANIFEST, old_manifest)
            result = audit_dataset(output, raw, episodes_per_task, allow_partial=allow_partial, decode_all=True)
            write_json(output / "meta/audit.json", result)
            if not result["success"]:
                raise RuntimeError(f"Existing output audit failed: {result['errors'][:5]}")
            write_dataset_readme(output, old_manifest, result)
            print(f"Already exported and verified: {len(entries)} episodes at {output}", flush=True)
            return

    staging = output.with_name(output.name + ".inprogress")
    backup = output.with_name(output.name + ".previous")
    if backup.exists():
        saved = read_json(backup / MANIFEST)
        if saved.get("raw_root") != str(raw) or saved.get("semantics") != semantics:
            raise ValueError(f"Unrecognized previous-output directory: {backup}")
        if not output.exists():
            backup.rename(output)
        else:
            shutil.rmtree(backup)
    discard_unpublished(staging)
    aggregate_datasets(
        repo_ids=[f"local/piper_{directory.name}" for directory, _ in parts],
        roots=[directory for directory, _ in parts],
        aggr_repo_id="local/piper_elevator_1200", aggr_root=staging,
        concatenate_videos=False, concatenate_data=False,
    )
    manifest = manifest_for(entries, "aggregate", raw)
    manifest["collection_metadata"] = collection_record
    manifest["episode_order"] = order
    manifest["parts"] = [{"path": str(directory), "manifest_sha256": sha256(directory / MANIFEST)}
                         for directory, _ in parts]
    manifest["episodes_per_task"] = episodes_per_task
    copy_collection_metadata(raw, staging, collection_record)
    write_json(staging / MANIFEST, manifest)
    result = audit_dataset(staging, raw, episodes_per_task, allow_partial=allow_partial, decode_all=True)
    result["dataset"] = str(output)
    write_json(staging / "meta/audit.json", result)
    if not result["success"]:
        raise RuntimeError(f"Aggregated dataset audit failed: {result['errors'][:5]}")
    if (output / "README.md").is_file():
        shutil.copy2(output / "README.md", staging / "README.md")
    write_dataset_readme(staging, manifest, result)
    if output.exists():
        output.rename(backup)
    try:
        staging.rename(output)
    except BaseException:
        if backup.exists() and not output.exists():
            backup.rename(output)
        raise
    if backup.exists():
        shutil.rmtree(backup)
    print(f"Published {len(entries)} verified LeRobot episodes: {output}", flush=True)


@contextmanager
def export_lock(parts: Path, output: Path | None = None):
    parts.mkdir(parents=True, exist_ok=True)
    paths = [parts / ".export.lock"]
    if output is not None:
        paths.append(output.parent / f".{output.name}.export.lock")
    with ExitStack() as stack:
        for path in paths:
            stream = stack.enter_context(path.open("a"))
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"Another converter owns {path}; raw collectors may run concurrently") from exc
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--parts", type=Path)
    parser.add_argument("--episodes-per-task", type=int, default=100)
    parser.add_argument("--part-size", type=int, default=25)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--watch", action="store_true", help="Convert arriving complete parts, publish when all tasks are complete")
    parser.add_argument("--poll-seconds", type=float, default=10)
    args = parser.parse_args()
    if args.episodes_per_task < 1 or args.part_size < 1 or args.poll_seconds <= 0:
        parser.error("Episode counts, part size and poll interval must be positive")
    require_version()
    configure_runtime()
    raw, output = args.raw.resolve(), args.output.resolve()
    parts_root = (args.parts or output.with_name(output.name + "_parts")).resolve()
    if not raw.is_dir():
        parser.error(f"Raw directory does not exist: {raw}")
    if (raw == output or raw == parts_root or output == parts_root
            or output in parts_root.parents or parts_root in output.parents):
        parser.error("Raw, output and parts must be different; output and parts must not contain each other")
    output.parent.mkdir(parents=True, exist_ok=True)
    with export_lock(parts_root, output):
        parts, seen = existing_parts(parts_root, raw)
        last_progress = None
        while True:
            episodes = discover_raw(raw, args.episodes_per_task)
            current_ids = {entry["episode_id"] for entry in episodes}
            if not seen <= current_ids:
                raise ValueError("Previously exported episodes are no longer committed successful raw episodes")
            counts = {floor: sum(entry["floor"] == floor for entry in episodes) for floor in FLOORS}
            complete = all(count == args.episodes_per_task for count in counts.values())
            pending = [entry for entry in episodes if entry["episode_id"] not in seen]
            while len(pending) >= args.part_size or (pending and (complete or not args.watch)):
                batch, pending = pending[:args.part_size], pending[args.part_size:]
                number = max([int(directory.name[-5:]) for directory, _ in parts], default=-1) + 1
                destination = parts_root / f"part_{number:05d}"
                write_part(batch, destination, raw)
                manifest = read_json(destination / MANIFEST)
                parts.append((destination, manifest))
                seen.update(entry["episode_id"] for entry in batch)
            if complete or not args.watch:
                if not complete and not args.allow_partial:
                    raise RuntimeError(f"Saved resumable parts, but dataset is incomplete: {counts}; use --watch or --allow-partial")
                publish_dataset(parts, output, raw, args.episodes_per_task, args.allow_partial)
                return
            progress = tuple(counts.items())
            if progress != last_progress:
                print(f"Waiting for committed raw episodes: {counts}; {len(seen)} converted", flush=True)
                last_progress = progress
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
