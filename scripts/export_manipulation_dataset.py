#!/usr/bin/env python3
"""Export committed joint7 manipulation demonstrations to official LeRobot v3.

Run in .conda/envs/lerobot. Raw capture uses radians/metres; published datasets
use the same degrees/full-gripper-centimetres interface as real PiPER PAP data.
Only a fully finalized and audited staging directory is published. Optional
v2.1 output is a separate per-episode compatibility copy, never a relabelled v3.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import re
import shutil
import time
import uuid

import numpy as np

FPS = 30
LEROBOT_VERSION = "0.6.1"
CAMERAS = {"observation.images.up": "up.mp4", "observation.images.wrist": "wrist.mp4"}
RAW_UNITS = ["rad"] * 6 + ["m"]
NAMES = [f"joint_{i}.pos" for i in range(1, 7)] + ["gripper.pos"]
TERMINAL_ACTION = "repeat_previous_interval_target"


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def configure_runtime():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    if version("lerobot") != LEROBOT_VERSION:
        raise RuntimeError(f"Use isolated lerobot=={LEROBOT_VERSION} environment")
    import av
    import torch
    av.logging.set_level(av.logging.ERROR)
    torch.set_num_threads(2)


def feature_spec():
    return {**{key: dict(dtype="float32", shape=(7,), names=NAMES.copy())
               for key in ("action", "observation.state")},
            **{key: dict(dtype="video", shape=(480, 640, 3), names=["height", "width", "channels"])
               for key in CAMERAS}}


def to_pap_units(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7 or not np.isfinite(values).all():
        raise ValueError("Expected finite N x 7 radians/full-width-metres values")
    result = values.copy()
    result[:, :6] = np.rad2deg(result[:, :6])
    result[:, 6] *= 100.
    return result.astype(np.float32)


def load_episode(directory, manifest):
    directory = Path(directory)
    match = re.fullmatch(r"episode_([0-9]{6,})", directory.name)
    if not match:
        raise ValueError(f"Invalid committed episode name: {directory}")
    metadata = read_json(directory / "metadata.json")
    required = dict(episode_index=int(match[1]), task_id=manifest["task_id"],
                    instruction=manifest["instruction"], fps=FPS, camera_resolution=[640, 480],
                    state_units=RAW_UNITS, action_units=RAW_UNITS,
                    gripper_representation="full_width", terminal_action=TERMINAL_ACTION)
    for key, value in required.items():
        if metadata.get(key) != value:
            raise ValueError(f"{directory}: incompatible {key}: {metadata.get(key)!r}")
    if (metadata.get("success") is not True or type(metadata.get("seed")) is not int
            or type(metadata.get("frame_count")) is not int or metadata["frame_count"] < 2):
        raise ValueError(f"{directory}: a successful episode, integer seed and frame count are required")
    for name in ("data.npz", *CAMERAS.values()):
        if not (directory / name).is_file():
            raise ValueError(f"Missing committed file: {directory / name}")
    with np.load(directory / "data.npz", allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in ("sim_time", "state", "action")}
    count = metadata["frame_count"]
    if (arrays["sim_time"].shape != (count,) or not np.allclose(
            arrays["sim_time"], np.arange(count) / FPS, rtol=0, atol=5e-6)):
        raise ValueError(f"{directory}: timestamps must start at zero and advance at 30 Hz")
    for key in ("state", "action"):
        if arrays[key].shape != (count, 7) or not np.isfinite(arrays[key]).all():
            raise ValueError(f"{directory}: malformed {key}")
    if not np.array_equal(arrays["action"][-1], arrays["action"][-2]):
        raise ValueError(f"{directory}: terminal action must repeat the previous applied target")
    if "duration_s" in metadata and not math.isclose(float(metadata["duration_s"]), (count-1)/FPS, abs_tol=5e-6):
        raise ValueError(f"{directory}: duration disagrees with frame timeline")
    return dict(directory=directory, metadata=metadata, frames=count,
                state=to_pap_units(arrays["state"]), action=to_pap_units(arrays["action"]))


def discover_episodes(raw, expected_count):
    raw = Path(raw)
    manifest = read_json(raw / "manifest.json")
    if (not isinstance(manifest.get("task_id"), str) or not manifest["task_id"]
            or not isinstance(manifest.get("instruction"), str) or not manifest["instruction"].strip()):
        raise ValueError("Raw manifest must provide a task_id and instruction")
    if manifest.get("target_episodes", expected_count) != expected_count:
        raise ValueError("Requested episode count differs from immutable raw manifest")
    directories = sorted(p for p in (raw / "episodes").glob("episode_*")
                         if p.is_dir() and re.fullmatch(r"episode_[0-9]{6,}", p.name))
    if len(directories) > expected_count:
        raise ValueError("Too many committed episodes; refusing silent truncation")
    entries = [load_episode(p, manifest) for p in directories]
    if [e["metadata"]["episode_index"] for e in entries] != list(range(len(entries))):
        raise ValueError("Committed episode indices must be consecutive from zero")
    seeds = [e["metadata"]["seed"] for e in entries]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Repeated seed in the same task dataset")
    return manifest, entries


def video_frames(path, *, decode_threads=2):
    import av
    with av.open(str(path)) as container:
        if len(container.streams.video) != 1 or container.streams.audio:
            raise ValueError(f"Expected one RGB video stream without audio: {path}")
        stream = container.streams.video[0]
        stream.thread_count = decode_threads
        stream.thread_type = "FRAME"
        if (stream.width, stream.height) != (640, 480) or abs(float(stream.average_rate)-FPS) > 1e-8:
            raise ValueError(f"Expected 640 x 480 RGB at 30 Hz: {path}")
        for index, frame in enumerate(container.decode(stream)):
            if frame.pts is None or abs(float(frame.pts*frame.time_base)-index/FPS) > 5e-6:
                raise ValueError(f"Missing/shifted RGB timestamp: {path}, frame {index}")
            yield frame.to_ndarray(format="rgb24")


def export_manifest(raw, source, entries):
    return dict(schema_version=1, task_id=source["task_id"], instruction=source["instruction"],
                format_version="v3.0", lerobot_version=LEROBOT_VERSION, raw_root=str(Path(raw).resolve()),
                raw_manifest=source, fps=FPS, episodes=[e["metadata"] for e in entries],
                semantics=dict(state="measured_joint_position_and_full_gripper_width",
                    action="commanded_target_for_following_1_over_30_second_interval",
                    units=["degree"]*6+["cm"], terminal_action=TERMINAL_ACTION,
                    image_layout="RGB HWC 480 x 640; up and wrist", perturbations="Stored actions include applied noise"),
                state_action_semantics=dict(state_units=["deg"]*6+["cm"], action_units=["deg"]*6+["cm"],
                    gripper_representation="full_width", action="next_interval_absolute_joint_target",
                    terminal_action="repeat_last_applied"),
                encoding=dict(codec="h264", crf=18, preset="ultrafast", g=2,
                    writer="official LeRobot create/add_frame/save_episode", source_mp4_reencoded=True,
                    frame_dropping=False, retimed=False))


def numeric_stats(values):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    minimum, maximum = array.min(0), array.max(0)
    std = array.std(0); std[minimum == maximum] = 0.
    result = dict(min=minimum.tolist(), max=maximum.tolist(), mean=array.mean(0).tolist(),
                  std=std.tolist(), count=[len(array)])
    result.update({f"q{int(q*100):02d}": np.quantile(array, q, axis=0).tolist()
                   for q in (.01, .1, .5, .9, .99)})
    return result


def refresh_numeric_stats(root):
    import pyarrow.parquet as pq
    tables = [pq.read_table(p) for p in sorted((root / "data").rglob("*.parquet"))]
    import pyarrow as pa
    table = pa.concat_tables(tables)
    stats = read_json(root / "meta/stats.json")
    for key in ("action", "observation.state", "timestamp", "frame_index", "episode_index", "index", "task_index"):
        stats[key] = numeric_stats(table[key].to_pylist())
    write_json(root / "meta/stats.json", stats)
    for path in sorted((root / "meta/episodes").rglob("*.parquet")):
        ep_table = pq.read_table(path)
        rows = ep_table.to_pylist()
        for row in rows:
            start, end = row["dataset_from_index"], row["dataset_to_index"]
            data = table.slice(start, end-start)
            for key in ("action", "observation.state", "timestamp", "frame_index", "episode_index", "index", "task_index"):
                for stat, value in numeric_stats(data[key].to_pylist()).items():
                    row[f"stats/{key}/{stat}"] = value
        pq.write_table(pa.Table.from_pylist(rows, schema=ep_table.schema), path)


def audit_dataset(root, entries, *, workers=4):
    import pyarrow.parquet as pq
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    root = Path(root)
    manifest, info = read_json(root / "meta/export_manifest.json"), read_json(root / "meta/info.json")
    total = sum(e["frames"] for e in entries)
    if (info["codebase_version"] != "v3.0" or info["fps"] != FPS or info["total_frames"] != total
            or info["total_episodes"] != len(entries) or info["total_tasks"] != 1):
        raise ValueError("Published dataset totals disagree with source")
    table_rows = [row for p in sorted((root / "data").rglob("*.parquet")) for row in pq.read_table(p).to_pylist()]
    if [r["index"] for r in table_rows] != list(range(total)):
        raise ValueError("Frame indices have gaps or duplicates")
    dataset = LeRobotDataset(f"local/{manifest['task_id']}", root=root, video_backend="pyav")
    offset, max_mae, sampled_frames = 0, 0., 0
    for episode_index, entry in enumerate(entries):
        count = entry["frames"]
        rows = table_rows[offset:offset+count]
        if ([r["episode_index"] for r in rows] != [episode_index]*count
                or [r["frame_index"] for r in rows] != list(range(count))
                or [r["task_index"] for r in rows] != [0]*count
                or not np.allclose([r["timestamp"] for r in rows], np.arange(count)/FPS, rtol=0, atol=5e-6)):
            raise ValueError(f"Invalid episode/time/task indexing: {episode_index}")
        for key, source in (("action", "action"), ("observation.state", "state")):
            if not np.array_equal(np.asarray([r[key] for r in rows], np.float32), entry[source]):
                raise ValueError(f"Numeric conversion or alignment mismatch: {episode_index}, {key}")
        selection = {0, count//2, count-1}
        source_frames = {key: {i: rgb for i, rgb in enumerate(video_frames(entry["directory"]/name)) if i in selection}
                         for key, name in CAMERAS.items()}
        for index in selection:
            sample = dataset[offset+index]
            sampled_frames += 1
            if sample["task"] != manifest["instruction"]:
                raise ValueError("Reader returned wrong language instruction")
            for key in CAMERAS:
                rgb = sample[key].cpu().numpy()
                if rgb.shape != (3, 480, 640) or not np.isfinite(rgb).all():
                    raise ValueError("Reader returned invalid RGB")
                mae = float(np.abs(np.moveaxis(rgb, 0, -1)*255.-source_frames[key][index]).mean())
                if mae > 8.:
                    raise ValueError(f"Source/output RGB alignment failed: {episode_index}, {key}, {index}")
                max_mae = max(max_mae, mae)
        offset += count
    def count_video(path):
        return sum(1 for _ in video_frames(path))
    counts = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for key in CAMERAS:
            counts[key] = sum(pool.map(count_video, sorted((root / "videos" / key).rglob("*.mp4"))))
            if counts[key] != total:
                raise ValueError(f"Output video has dropped or extra frames: {key}")
    stats = read_json(root / "meta/stats.json")
    for key in feature_spec():
        if key not in stats or not all(k in stats[key] for k in ("min", "max", "mean", "std", "count")):
            raise ValueError(f"Missing feature statistics: {key}")
    return dict(success=True, episodes=len(entries), frames=total, numeric_rows_compared=total,
                full_video_decode=True, decoded_frames=counts, official_reader_samples=sampled_frames,
                max_source_to_output_rgb_mae=max_mae, checked_utc=datetime.now(timezone.utc).isoformat())


@contextmanager
def output_lock(output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / f".{output.name}.export.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another exporter owns {output}") from exc
        yield


def export_dataset(raw, output, source, entries, *, workers=4):
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    output = Path(output)
    manifest = export_manifest(raw, source, entries)
    if output.exists():
        if read_json(output / "meta/export_manifest.json") != manifest:
            raise ValueError("Existing output has different source/settings")
        return audit_dataset(output, entries, workers=workers)
    staging = output.with_name(f".{output.name}.inprogress-{uuid.uuid4().hex[:8]}")
    # v0.6.1 feed_frame uses a timed nonblocking queue and drops when full.
    # Capacity for a whole episode guarantees no losses during offline bursts.
    dataset = LeRobotDataset.create(repo_id=f"local/{source['task_id']}", root=staging,
        fps=FPS, robot_type="piper_follower", features=feature_spec(), use_videos=True,
        video_backend="pyav", streaming_encoding=True, encoder_threads=2,
        encoder_queue_maxsize=max(e["frames"] for e in entries)+1, metadata_buffer_size=1,
        rgb_encoder=RGBEncoderConfig(vcodec="h264", pix_fmt="yuv420p", crf=18, g=2, preset="ultrafast"))
    try:
        for entry in entries:
            readers = {key: video_frames(entry["directory"]/name) for key, name in CAMERAS.items()}
            try:
                for index in range(entry["frames"]):
                    try:
                        images = {key: next(reader) for key, reader in readers.items()}
                    except StopIteration as exc:
                        raise ValueError(f"Short source video: {entry['directory']}, frame {index}") from exc
                    dataset.add_frame({**images, "observation.state": entry["state"][index],
                                       "action": entry["action"][index], "task": source["instruction"]})
                if any(next(reader, None) is not None for reader in readers.values()):
                    raise ValueError(f"Source video has excess frames: {entry['directory']}")
                dataset.save_episode()
            finally:
                for reader in readers.values():
                    reader.close()
            print(json.dumps(dict(exported=entry["metadata"]["episode_index"]+1,
                                  total=len(entries), task=source["task_id"])), flush=True)
    finally:
        dataset.finalize()
    write_json(staging / "meta/export_manifest.json", manifest)
    refresh_numeric_stats(staging)
    audit = audit_dataset(staging, entries, workers=workers)
    write_json(staging / "meta/audit.json", audit)
    (staging / "README.md").write_text(
        f"# {source['task_id']}\n\n{len(entries)} successful physical Isaac demonstrations at 30 Hz.\n\n"
        f"Instruction: **{source['instruction']}**\n\n"
        "`action` and `observation.state` are six absolute joint angles in degrees and full gripper opening in centimetres. "
        "State/RGB at t are paired with the actual commanded target for the following 1/30 s interval, including applied noise. "
        "The terminal action repeats the last executed target. Two 640×480 RGB views: up and wrist.\n\n"
        "Written by official LeRobot 0.6.1 as v3.0. Complete numeric rows and decoded video counts are audited; "
        "first/middle/last frames per episode are also checked through the official reader. "
        "Numeric moments and quantiles are recomputed from all rows in float64. Image statistics use the official streaming writer. "
        "Raw rejected attempts are excluded. Layouts, noise settings and capture provenance are in meta/export_manifest.json.\n")
    staging.rename(output)
    return audit


def export_v21_compatibility(raw, v3_root, output, entries):
    """Write genuine per-episode v2.1 paths, with copied original RGB videos."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    output, v3_root = Path(output), Path(v3_root)
    source_manifest = read_json(v3_root / "meta/export_manifest.json")
    identity = dict(format_version="v2.1", source_v3=str(v3_root.resolve()),
                    source_export_manifest=source_manifest, source_mp4_reencoded=False)
    if output.exists():
        if read_json(output / "meta/compatibility_manifest.json") != identity:
            raise ValueError("Existing v2.1 copy has another source")
        return
    stage = output.with_name(f".{output.name}.inprogress-{uuid.uuid4().hex[:8]}")
    (stage / "meta").mkdir(parents=True)
    info = read_json(v3_root / "meta/info.json")
    info.update(codebase_version="v2.1", chunks_size=1000,
                total_chunks=math.ceil(len(entries)/1000), total_videos=len(entries)*len(CAMERAS),
                data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                video_path="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")
    for key in ("data_files_size_in_mb", "video_files_size_in_mb"):
        info.pop(key, None)
    table = pa.concat_tables([pq.read_table(p) for p in sorted((v3_root/"data").rglob("*.parquet"))])
    ep_rows = [row for p in sorted((v3_root/"meta/episodes").rglob("*.parquet")) for row in pq.read_table(p).to_pylist()]
    with (stage/"meta/episodes.jsonl").open("w") as eps, (stage/"meta/episodes_stats.jsonl").open("w") as sts:
        offset = 0
        for index, (entry, ep_meta) in enumerate(zip(entries, ep_rows, strict=True)):
            data = table.slice(offset, entry["frames"])
            path = stage / info["data_path"].format(episode_chunk=index//1000, episode_index=index)
            path.parent.mkdir(parents=True, exist_ok=True); pq.write_table(data, path)
            for key, name in CAMERAS.items():
                dst = stage / info["video_path"].format(episode_chunk=index//1000, episode_index=index, video_key=key)
                dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(entry["directory"]/name, dst)
            eps.write(json.dumps(dict(episode_index=index, tasks=[source_manifest["instruction"]], length=entry["frames"]))+"\n")
            stats = {}
            for key, value in ep_meta.items():
                if key.startswith("stats/"):
                    _, feature, statistic = key.split("/", 2)
                    stats.setdefault(feature, {})[statistic] = value
            sts.write(json.dumps(dict(episode_index=index, stats=stats))+"\n")
            offset += entry["frames"]
    for key in CAMERAS:
        info["features"][key]["info"]["video.codec"] = "h264"
    (stage/"meta/tasks.jsonl").write_text(json.dumps(dict(task_index=0, task=source_manifest["instruction"]))+"\n")
    write_json(stage/"meta/info.json", info)
    write_json(stage/"meta/compatibility_manifest.json", identity)
    write_json(stage/"meta/export_manifest.json", {**source_manifest, "format_version":"v2.1"})
    shutil.copy2(v3_root/"meta/stats.json", stage/"meta/stats.json")
    shutil.copy2(v3_root/"README.md", stage/"README.md")
    with (stage/"README.md").open("a") as stream:
        stream.write("\nThis directory is the v2.1 per-episode compatibility copy. Original raw MP4 files are copied without re-encoding.\n")
    stage.rename(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=15.)
    parser.add_argument("--workers", type=int, default=4, help="Parallel full-video audit workers")
    parser.add_argument("--v21-output", type=Path)
    args = parser.parse_args(argv)
    if args.episodes < 1 or args.workers < 1 or not math.isfinite(args.poll_seconds) or args.poll_seconds <= 0:
        parser.error("Episode count, workers and poll interval must be positive")
    raw, output = args.raw.resolve(), args.output.resolve()
    paths = [raw, output]+([args.v21_output.resolve()] if args.v21_output else [])
    if any(a == b or a in b.parents or b in a.parents for i, a in enumerate(paths) for b in paths[i+1:]):
        parser.error("Raw and output directories must be separate")
    configure_runtime()
    with output_lock(output):
        previous_count = None
        while True:
            source, entries = discover_episodes(raw, args.episodes)
            if len(entries) == args.episodes:
                audit = export_dataset(raw, output, source, entries, workers=args.workers)
                if args.v21_output:
                    with output_lock(args.v21_output):
                        export_v21_compatibility(raw, output, args.v21_output.resolve(), entries)
                print(json.dumps(audit, indent=2), flush=True)
                return 0
            if not args.watch:
                raise ValueError(f"Only {len(entries)}/{args.episodes} committed episodes; use --watch")
            if len(entries) != previous_count:
                print(f"Waiting for {source['task_id']}: {len(entries)}/{args.episodes}", flush=True)
                previous_count = len(entries)
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
