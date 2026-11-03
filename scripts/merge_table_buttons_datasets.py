#!/usr/bin/env python3
"""Merge table-button color datasets using the official LeRobot v3 aggregator.

Run in .conda/envs/lerobot. Source datasets remain immutable. Videos are copied
without decoding/re-encoding; indices are remapped and every numeric row audited.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import uuid

import numpy as np

COLORS = ("red", "yellow", "blue")
PROFILES = ("near", "medium", "far")
CAMERAS = ("observation.images.up", "observation.images.wrist")
TASKS = tuple(f"Press the {color} button." for color in COLORS)
NUMERIC_KEYS = ("action", "observation.state", "timestamp", "frame_index",
                "episode_index", "index", "task_index")


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot(root: Path) -> dict:
    return {str(path.relative_to(root)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(root.rglob("*")) if path.is_file()}


def numeric_arrays(root: Path) -> dict:
    import pyarrow.parquet as pq
    parts = {key: [] for key in NUMERIC_KEYS}
    for path in sorted((root / "data").glob("chunk-*/*.parquet")):
        table = pq.read_table(path)
        for key in NUMERIC_KEYS:
            parts[key].append(np.asarray(table[key].to_pylist()))
    return {key: np.concatenate(values) for key, values in parts.items()}


def numeric_stats(values) -> dict:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    if not len(data) or not np.isfinite(data).all():
        raise ValueError("Statistics require nonempty finite numeric data")
    minimum, maximum = data.min(axis=0), data.max(axis=0)
    std = data.std(axis=0)
    std[minimum == maximum] = 0.0
    result = {"min": minimum.tolist(), "max": maximum.tolist(),
              "mean": data.mean(axis=0).tolist(), "std": std.tolist(), "count": [len(data)]}
    result.update({f"q{int(q * 100):02d}": np.quantile(data, q, axis=0).tolist()
                   for q in (.01, .10, .50, .90, .99)})
    return result


def source_manifests(roots: list[Path], profile: str,
                     expected_episodes_per_color: int | None = None) -> list[dict]:
    manifests = [read_json(root / "meta/export_manifest.json") for root in roots]
    infos = [read_json(root / "meta/info.json") for root in roots]
    for color, manifest, info in zip(COLORS, manifests, infos, strict=True):
        expected = {"group": f"{profile}_{color}", "spacing_profile": profile,
                    "target_color": color, "instruction": f"Press the {color} button.",
                    "fps": 30, "format_version": "v3.0", "lerobot_version": "0.6.1"}
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(f"Source {color} has incompatible {key}")
        episodes = manifest["episodes"]
        if not episodes or [e["episode_index"] for e in episodes] != list(range(len(episodes))):
            raise ValueError(f"Source {color} must contain nonempty, consecutively ordered episodes")
        if expected_episodes_per_color is not None and len(episodes) != expected_episodes_per_color:
            raise ValueError(f"Source {color} must contain {expected_episodes_per_color} episodes")
        if any(not e.get("success") or e["target_color"] != color
               or e["instruction"] != expected["instruction"] for e in episodes):
            raise ValueError(f"Source {color} has invalid task or unsuccessful episodes")
        if manifest["semantics"] != manifests[0]["semantics"]:
            raise ValueError("Cannot merge different action/state semantics")
        for key in ("fps", "features", "robot_type", "codebase_version"):
            if info[key] != infos[0][key]:
                raise ValueError(f"Cannot merge incompatible {key}")
        if info["total_episodes"] != len(episodes) or info["total_tasks"] != 1:
            raise ValueError("Unexpected source dataset totals")
    return manifests


def layout_group_id(profile: str, episode: dict) -> str:
    """Group identical colored-button positions, independent of retries or task IDs."""
    layout = {"spacing_profile": profile, "positions_xy_m": episode["positions_xy_m"]}
    encoded = json.dumps(layout, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def merged_manifest(roots: list[Path], profile: str, manifests: list[dict], snapshots: list[dict]) -> dict:
    episodes, sources = [], []
    episode_offset = 0
    episode_order = []
    for task_index, (root, source, files) in enumerate(zip(roots, manifests, snapshots, strict=True)):
        count = len(source["episodes"])
        sources.append({"group": source["group"], "root": str(root), "files": files,
                        "episode_offset": episode_offset, "task_index": task_index,
                        "episodes": count, "instruction": source["instruction"]})
        for entry in source["episodes"]:
            episodes.append({**entry, "episode_index": episode_offset + entry["episode_index"],
                             "group": f"{profile}_multitask", "source_group": source["group"],
                             "source_episode_index": entry["episode_index"], "task_index": task_index,
                             "layout_group_id": layout_group_id(profile, entry)})
        episode_order.append(f"{COLORS[task_index]} {episode_offset}:{episode_offset + count}")
        episode_offset += count
    return {"schema_version": 3, "kind": "table_buttons_multitask", "format_version": "v3.0",
            "lerobot_version": "0.6.1", "group": f"{profile}_multitask", "spacing_profile": profile,
            "target_colors": list(COLORS), "tasks": list(TASKS), "fps": 30,
            "semantics": manifests[0]["semantics"], "sources": sources, "episodes": episodes,
            "merge": {"tool": "lerobot.datasets.aggregate.aggregate_datasets",
                      "concatenate_videos": False, "concatenate_data": False,
                      "source_mp4_reencoded": False, "source_files_modified": False,
                      "episode_order": ", ".join(episode_order),
                      "statistics": "Numeric statistics and quantiles recomputed exactly in float64. "
                      "Image moments and sample counts pooled from source statistics by official LeRobot; "
                      "image quantiles are the official approximate weighted source quantiles."},
            "split_guidance": {"group_key": "layout_group_id",
                               "unique_layouts": len({e["layout_group_id"] for e in episodes}),
                               "reason": "Group by actual colored-button positions, independent of "
                               "source episode index or retry seed. Keep every task with an identical "
                               "layout in the same train/validation partition."}}


def refresh_numeric_stats(root: Path, arrays: dict):
    """The official finalizer pools unshifted source index stats; recompute them.

    Recompute other numeric features too so quantiles are global, not averaged
    source quantiles. Per-episode stats must describe the new bookkeeping IDs.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    stats = read_json(root / "meta/stats.json")
    stats.update({key: numeric_stats(value) for key, value in arrays.items()})
    write_json(root / "meta/stats.json", stats)
    for path in sorted((root / "meta/episodes").glob("chunk-*/*.parquet")):
        table = pq.read_table(path)
        rows = table.to_pylist()
        for row in rows:
            start, end = row["dataset_from_index"], row["dataset_to_index"]
            for key, value in arrays.items():
                for stat, number in numeric_stats(value[start:end]).items():
                    row[f"stats/{key}/{stat}"] = number
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)


def audit_merged(root: Path, roots: list[Path], manifest: dict, source_snapshots: list[dict]) -> dict:
    import pyarrow.parquet as pq
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import torch
    info = read_json(root / "meta/info.json")
    arrays = numeric_arrays(root)
    stats = read_json(root / "meta/stats.json")
    rows = [r for path in sorted((root / "meta/episodes").glob("chunk-*/*.parquet"))
            for r in pq.read_table(path).to_pylist()]
    total_episodes = len(manifest["episodes"])
    if (info["total_episodes"], info["total_tasks"], len(rows)) != (total_episodes, 3, total_episodes):
        raise ValueError(f"Merged dataset must contain {total_episodes} episodes and three tasks")
    tasks = pq.read_table(root / "meta/tasks.parquet").to_pylist()
    if [(r["task_index"], r["task"]) for r in tasks] != list(enumerate(TASKS)):
        raise ValueError("Merged task vocabulary is incorrect")
    frame_offset = 0
    video_checks = []
    reader_checks = []
    # Reader caches live outside source datasets. Both source and merged readers
    # are read-only and force no Hub synchronization.
    merged = LeRobotDataset(f"local/{manifest['group']}", root=root, video_backend="pyav",
                           return_uint8=True, download_videos=False)
    for task_index, source_root in enumerate(roots):
        source_meta = manifest["sources"][task_index]
        episode_offset = source_meta["episode_offset"]
        source_arrays = numeric_arrays(source_root)
        count = len(source_arrays["index"])
        expected = {key: value.copy() for key, value in source_arrays.items()}
        expected["index"] += frame_offset
        expected["episode_index"] += episode_offset
        expected["task_index"][:] = task_index
        for key, value in expected.items():
            if not np.array_equal(arrays[key][frame_offset:frame_offset + count], value):
                raise ValueError(f"Merged numeric column {key} differs from {source_root.name}")
        source_rows = [r for path in sorted((source_root / "meta/episodes").glob("chunk-*/*.parquet"))
                       for r in pq.read_table(path).to_pylist()]
        if len(source_rows) != source_meta["episodes"]:
            raise ValueError("Source episode metadata count differs from export manifest")
        checked_files = set()
        for local_episode, source_row in enumerate(source_rows):
            row = rows[episode_offset + local_episode]
            if source_row["episode_index"] != local_episode or row["episode_index"] != episode_offset + local_episode:
                raise ValueError("Episode index remapping is incorrect")
            if row["tasks"] != [TASKS[task_index]] or row["length"] != source_row["length"]:
                raise ValueError("Episode task/length mismatch")
            for key in ("dataset_from_index", "dataset_to_index"):
                if row[key] != source_row[key] + frame_offset:
                    raise ValueError(f"Episode {key} offset is incorrect")
            for camera in CAMERAS:
                for key in ("from_timestamp", "to_timestamp"):
                    if row[f"videos/{camera}/{key}"] != source_row[f"videos/{camera}/{key}"]:
                        raise ValueError("Non-concatenated video timestamps must remain unchanged")
                def video_path(base, record):
                    return base / info["video_path"].format(video_key=camera,
                        chunk_index=record[f"videos/{camera}/chunk_index"],
                        file_index=record[f"videos/{camera}/file_index"])
                src_video, dst_video = video_path(source_root, source_row), video_path(root, row)
                if str(dst_video) not in checked_files:
                    digest = source_snapshots[task_index][str(src_video.relative_to(source_root))]["sha256"]
                    if digest != sha256(dst_video):
                        raise ValueError("Video bytes changed during aggregation")
                    src_stat, dst_stat = src_video.stat(), dst_video.stat()
                    if (src_stat.st_dev, src_stat.st_ino) == (dst_stat.st_dev, dst_stat.st_ino):
                        raise ValueError("Merged video must be an independent copy, not a hard link")
                    video_checks.append({"source_group": source_root.name,
                                         "source": str(src_video.relative_to(source_root)),
                                         "destination": str(dst_video.relative_to(root)), "sha256": digest})
                    checked_files.add(str(dst_video))
        source = LeRobotDataset(f"local/{source_root.name}", root=source_root, video_backend="pyav",
                               return_uint8=True, download_videos=False)
        # Covers initial/final source frames, all color transitions and a middle
        # episode with nonzero concatenated-video offsets in each source.
        for local_episode in sorted({0, (len(source_rows) - 1) // 2, len(source_rows) - 1}):
            source_row = source_rows[local_episode]
            for local_index in (source_row["dataset_from_index"], source_row["dataset_to_index"] - 1):
                original, combined = source[local_index], merged[frame_offset + local_index]
                for camera in CAMERAS:
                    if not torch.equal(original[camera], combined[camera]):
                        raise ValueError(f"Official reader image differs: {source_root.name}/{local_index}/{camera}")
                if combined["task"] != TASKS[task_index]:
                    raise ValueError("Official reader task text differs")
                for key in ("action", "observation.state", "timestamp", "frame_index"):
                    if not torch.equal(original[key], combined[key]):
                        raise ValueError(f"Official reader numeric value differs: {key}")
                reader_checks.append({"source_group": source_root.name, "source_frame": local_index,
                                      "merged_frame": frame_offset + local_index})
        frame_offset += count
    if frame_offset != info["total_frames"] or not np.array_equal(arrays["index"], np.arange(frame_offset)):
        raise ValueError("Incorrect total frames or global frame sequence")
    for key in NUMERIC_KEYS:
        if stats[key] != numeric_stats(arrays[key]):
            raise ValueError(f"Incorrect merged numeric statistics: {key}")
    image_counts = {}
    for camera in CAMERAS:
        expected_count = sum(read_json(p / "meta/stats.json")[camera]["count"][0] for p in roots)
        if stats[camera]["count"] != [expected_count]:
            raise ValueError(f"Incorrect image sample count: {camera}")
        image_counts[camera] = expected_count
    for path, before in zip(roots, source_snapshots, strict=True):
        if snapshot(path) != before:
            raise ValueError(f"Source files changed during merge: {path}")
    return {"success": True, "dataset": manifest["group"], "episodes": total_episodes,
            "frames": frame_offset, "tasks": list(TASKS),
            "episodes_per_task": [source["episodes"] for source in manifest["sources"]],
            "all_numeric_rows_equal_to_reindexed_sources": True, "source_files_unchanged": True,
            "video_reencoded": False, "video_files": video_checks,
            "official_reader": {"lerobot_version": "0.6.1", "backend": "pyav",
                                "both_cameras_pixel_identical": True, "samples": reader_checks},
            "image_statistics_sample_counts": image_counts,
            "audited_utc": datetime.now(timezone.utc).isoformat()}


def merge_profile(source_root: Path, destination_root: Path, profile: str,
                  expected_episodes_per_color: int | None = None) -> dict:
    from lerobot.datasets.aggregate import aggregate_datasets
    roots = [source_root / f"{profile}_{color}" for color in COLORS]
    target = destination_root / f"{profile}_multitask"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite an existing dataset: {target}")
    manifests = source_manifests(roots, profile, expected_episodes_per_color)
    originals = [snapshot(root) for root in roots]
    manifest = merged_manifest(roots, profile, manifests, originals)
    temporary = destination_root / f".{target.name}.{uuid.uuid4().hex}.partial"
    try:
        aggregate_datasets([f"local/{root.name}" for root in roots], f"local/{target.name}",
                           roots=roots, aggr_root=temporary,
                           concatenate_videos=False, concatenate_data=False)
        refresh_numeric_stats(temporary, numeric_arrays(temporary))
        write_json(temporary / "meta/export_manifest.json", manifest)
        report = audit_merged(temporary, roots, manifest, originals)
        write_json(temporary / "meta/audit.json", report)
        task_counts = ", ".join(f"{source['episodes']:,} {color}" for color, source
                                in zip(COLORS, manifest["sources"], strict=True))
        episode_ranges = ", ".join(
            f"{color} {source['episode_offset']}–{source['episode_offset'] + source['episodes'] - 1}"
            for color, source in zip(COLORS, manifest["sources"], strict=True))
        (temporary / "README.md").write_text(
            f"# Table buttons: {profile} spacing, three language tasks\n\n"
            f"{report['episodes']:,} successful scripted-controller demonstrations: {task_counts}.\n\n"
            + "\n".join(f"- `{task}`" for task in TASKS)
            + f"\n\nLeRobot v3, 30 Hz, {report['frames']:,} frames, PiPER joint7 actions/state, "
            "two RGB videos (`observation.images.up`, `observation.images.wrist`). "
            "The six joint angles use degrees; the gripper value is full opening in centimeters.\n\n"
            f"Episode IDs: {episode_ranges}. Every numeric row matches "
            "its source exactly after remapping episode/global-frame/task IDs. Videos are independent "
            "byte-identical copies, with no re-encoding. The source datasets were not modified.\n\n"
            "`meta/export_manifest.json` records per-episode source IDs, task, layout seed, positions "
            "and original file hashes. `meta/audit.json` records numeric, video-byte and official-reader checks.\n\n"
            "**Split by `layout_group_id`, not independently by merged episode:** this key groups "
            "identical colored-button positions even if source episode indices or retry seeds differ. "
            f"There are {manifest['split_guidance']['unique_layouts']:,} distinct layouts. Put every task "
            "for a layout in the same train/validation partition. "
            f"The format's `train: 0:{report['episodes']}` denotes available data; downstream training must "
            "apply its separately recorded split.\n\n"
            "Numeric statistics and quantiles were recomputed exactly. Image moments and sample counts "
            "are pooled from source statistics; image quantiles retain LeRobot's approximate weighted "
            "source-quantile aggregation. No new physics demonstrations were generated.\n")
        temporary.rename(target)
        return {"profile": profile, "path": str(target), "episodes": report["episodes"],
                "frames": report["frames"], "audit_success": True}
    except BaseException:
        # Only this invocation's unpublished output is disposable; originals and
        # any previously published destination are never removed or modified.
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("/home/pengguanqi/Datasets/piper_table_buttons_20261023"))
    parser.add_argument("--destination", type=Path,
                        default=Path("/home/pengguanqi/Datasets/piper_table_buttons_multitask_20261023"))
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=list(PROFILES))
    parser.add_argument("--expected-episodes-per-color", type=int,
                        help="Reject sources with a different episode count (for example, 1000)")
    args = parser.parse_args()
    if args.expected_episodes_per_color is not None and args.expected_episodes_per_color < 1:
        parser.error("--expected-episodes-per-color must be positive")
    if version("lerobot") != "0.6.1":
        raise RuntimeError("Use the isolated LeRobot 0.6.1 environment")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    args.destination.mkdir(parents=True, exist_ok=True)
    reports = []
    for profile in args.profiles:
        report = merge_profile(args.source.resolve(), args.destination.resolve(), profile,
                               args.expected_episodes_per_color)
        reports.append(report)
        print(json.dumps(report), flush=True)
    write_json(args.destination / "merge_summary.json", {"success": True, "datasets": reports,
               "total_episodes": sum(r["episodes"] for r in reports),
               "total_frames": sum(r["frames"] for r in reports)})
    (args.destination / "README.md").write_text(
        "# Multi-task PiPER table-button datasets\n\n"
        "Each selected spacing configuration combines its red, yellow and blue button demonstrations "
        "into one independent LeRobot v3 dataset. Original datasets remain unchanged. "
        "Language instructions are identical across spacing configurations.\n\n"
        "| Dataset | Episodes | Frames | Tasks |\n| --- | ---: | ---: | --- |\n"
        + "\n".join(f"| {r['profile']}_multitask | {r['episodes']} | {r['frames']} | red, yellow, blue |" for r in reports)
        + "\n\nAll datasets retain the original 30 Hz joint7 numeric values and byte-identical video copies. "
        "See each dataset's README and `meta/export_manifest.json` for units, provenance and layout-grouped "
        "train/validation split guidance.\n")


if __name__ == "__main__":
    main()
