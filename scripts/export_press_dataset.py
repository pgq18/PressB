#!/usr/bin/env python3
"""Export first-press-only LeRobot episodes without altering recorded sources.

Use the isolated LeRobot environment. A deterministic, immutable cut plan is
required. Each part and the final aggregate are published only after the
independent press-prefix audit passes. Interrupted parts may be safely resumed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import multiprocessing
from pathlib import Path
import re
import sys

import numpy as np

from export_lerobot import (
    CAMERAS, COLLECTION_METADATA, EPISODE_CONTEXT, FLOORS, LEROBOT_VERSION, episode_panel_context,
    MANIFEST, ROOT, SOURCE_FILES, TASKS, URDF, collection_snapshot,
    collection_timing, configure_runtime, copy_collection_metadata,
    discard_unpublished, export_lock, feature_spec, read_json, require_version,
    semantics_for_collection, sha256, validate_episode_collection, video_frames,
    write_json,
)

sys.path.insert(0, str(ROOT / "src"))

CUT_PLAN = "meta/cut_plan.json"
TRANSFORM_NAME = "first_press_prefix"
TRANSFORM_VERSION = 1
TERMINAL_POLICY = {
    "action": "source.action[cut_frame_index - 1]",
    "action.joint_target": "source.q_target[cut_frame_index - 1]",
    "meaning": "hold the previously issued planned endpoint that led to the terminal observation",
    "measured_state_used_as_target": False,
    "nonterminal_actions": "source prefix unchanged",
}


def press_semantics(collection: dict) -> dict:
    return {
        **semantics_for_collection(collection),
        "action_joint_target": "next_sample_endpoint_except_terminal_repeat_previous_planned_target",
        "episode_end": "first_sampled_target_light_on",
        "terminal_action": "repeat_previous_planned_target",
    }


def plan_records(plan: dict) -> dict[int, dict]:
    records = plan.get("episodes")
    if not isinstance(records, list) or not records:
        raise ValueError("Cut plan requires a nonempty episodes list")
    result = {}
    for entry in records:
        source_id = entry.get("source_episode_id")
        k = entry.get("cut_frame_index")
        if (type(source_id) is not int or source_id < 0 or source_id in result
                or type(k) is not int or k < 1 or entry.get("kept_frames") != k + 1
                or entry.get("floor") not in FLOORS):
            raise ValueError(f"Invalid or duplicate cut record: {source_id}")
        result[source_id] = entry
    return result


def validate_plan_source(plan: dict, raw: Path, collection: dict, collection_record: dict):
    """Bind even a selected pilot subset to the complete original collection."""
    from pressb.press_prefix import CUT_POLICY, TERMINAL_ACTION_POLICY
    expected = {"schema_version": 1, "kind": "press_prefix_cut_plan", "success": True,
                "raw_root": str(raw), "source_collection_metadata_sha256": collection_record["sha256"],
                "source_collection_fingerprint": collection["collection_fingerprint"],
                "source_raw_schema_version": collection["raw_schema_version"],
                "cut_policy": CUT_POLICY, "terminal_action_policy": TERMINAL_ACTION_POLICY,
                **collection_timing(collection)}
    for key, value in expected.items():
        if plan.get(key) != value:
            raise ValueError(f"Cut plan source or policy mismatch: {key}")
    records = plan_records(plan)
    if plan.get("total_episodes") != len(records):
        raise ValueError("Cut plan total episode count mismatch")


def transformation_record(plan_path: Path) -> dict:
    return {"name": TRANSFORM_NAME, "version": TRANSFORM_VERSION,
            "cut_plan": {"path": CUT_PLAN, "source_path": str(plan_path), "sha256": sha256(plan_path)},
            "observation_selection": "source frames 0 through cut_frame_index inclusive",
            "video_selection": "decode and re-encode only the same retained prefix for both cameras",
            "terminal_action_policy": TERMINAL_POLICY}


def verify_plan_record(directory: Path, collection: dict, expected: dict):
    from pressb.press_prefix import inspect_press_prefix
    actual = inspect_press_prefix(directory, collection, project_root=ROOT)
    if actual != expected:
        raise ValueError(f"Raw episode no longer matches immutable cut plan: {directory}")


def copy_plan(plan_path: Path, target: Path, expected: dict):
    payload = plan_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected["cut_plan"]["sha256"]:
        raise ValueError("Cut plan changed during export")
    destination = target / CUT_PLAN
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)


def source_provenance(raw: Path, record: dict, collection: dict) -> dict:
    source_id = record["source_episode_id"]
    directory = raw / f"episode_{source_id:06d}"
    metadata = read_json(directory / "metadata.json")
    validate_episode_collection(metadata, collection, directory)
    if (metadata.get("success") is not True or metadata.get("episode_id") != source_id
            or metadata.get("floor") != record["floor"]):
        raise ValueError(f"Source episode identity or success mismatch: {directory}")
    verify_plan_record(directory, collection, record)
    return {"episode_id": source_id, "floor": record["floor"], "seed": int(metadata["seed"]),
            "task": TASKS[record["floor"] - 24], "frames": record["kept_frames"],
            "raw_directory": str(directory),
            **{key: metadata[key] for key in EPISODE_CONTEXT},
            **episode_panel_context(metadata),
            "source_sha256": {name: sha256(directory / name) for name in SOURCE_FILES},
            "press_prefix": record}


def manifest_for(entries: list[dict], kind: str, raw: Path, plan_path: Path, identity: dict) -> dict:
    _, collection, collection_record = collection_snapshot(raw)
    return {"schema_version": 2, "kind": kind, "lerobot_version": LEROBOT_VERSION,
            "format_version": "v3.0", "created_utc": datetime.now(timezone.utc).isoformat(),
            "raw_root": str(raw), "tasks": TASKS, "semantics": press_semantics(collection),
            "collection_metadata": collection_record,
            "transformation": transformation_record(plan_path), "export_identity": identity,
            "urdf": {"path": str(URDF), "sha256": sha256(URDF)}, "episodes": entries}


def dataset_inventory(dataset: Path) -> dict[str, str]:
    """Hash official data, videos, metadata and audit for safe committed resume."""
    named = {MANIFEST, COLLECTION_METADATA, CUT_PLAN, "meta/info.json", "meta/stats.json",
             "meta/tasks.parquet", "meta/audit.json", "README.md"}
    return {str(path.relative_to(dataset)): sha256(path)
            for path in sorted(dataset.rglob("*"))
            if path.is_file() and (str(path.relative_to(dataset)) in named
                                   or str(path.relative_to(dataset)).startswith(("data/", "videos/", "meta/episodes/")))}


def validate_committed(directory: Path, records: list[dict], identity: dict):
    manifest = read_json(directory / MANIFEST)
    if manifest.get("export_identity") != identity:
        raise ValueError(f"Cannot mix conversion settings or cut plans in {directory}")
    if [entry.get("press_prefix") for entry in manifest["episodes"]] != records:
        raise ValueError(f"Committed part has a different source selection: {directory}")
    audit = read_json(directory / "meta/audit.json")
    if not audit.get("success") or not audit.get("full_video_decode"):
        raise ValueError(f"Committed output lacks a successful complete audit: {directory}")
    inventory = read_json(directory / "meta/commit_inventory.json")
    if inventory != dataset_inventory(directory):
        raise ValueError(f"Committed output changed since its complete audit: {directory}")
    return manifest


def audit_output(staging: Path, destination: Path, raw: Path, count: int,
                 episodes_per_task: int | None):
    from audit_press_dataset import audit_press_dataset
    print(f"Auditing {count} press-prefix episodes in {staging.name}", flush=True)
    report = audit_press_dataset(staging, raw=raw, expected_episodes=count,
                                episodes_per_task=episodes_per_task,
                                allow_partial=episodes_per_task is None, decode_all=True)
    report["dataset"] = str(destination)
    write_json(staging / "meta/audit.json", report)
    if not report.get("success") or not report.get("full_video_decode"):
        raise RuntimeError(f"Independent press-prefix audit failed: {staging}: {report.get('errors', [])[:5]}")
    return report


def write_part(records: list[dict], destination: Path, raw: Path, plan_path: Path, identity: dict):
    require_version()
    configure_runtime()
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from pressb.press_prefix import crop_press_prefix_arrays

    _, collection, collection_record = collection_snapshot(raw)
    fps = collection_timing(collection)["fps"]
    entries = [source_provenance(raw, record, collection) for record in records]
    staging = destination.with_name(destination.name + ".inprogress")
    discard_unpublished(staging)
    queue_capacity = max(record["kept_frames"] for record in records) + 1
    dataset = LeRobotDataset.create(
        repo_id=f"local/piper_press_{destination.name}", root=staging, fps=fps, robot_type="piper",
        features=feature_spec(), video_backend="pyav", use_videos=True,
        rgb_encoder=RGBEncoderConfig(vcodec="h264", pix_fmt="yuv420p", crf=18, g=2, preset="ultrafast"),
        streaming_encoding=True, encoder_threads=2, encoder_queue_maxsize=queue_capacity,
        metadata_buffer_size=1,
    )
    dataset.meta.save_episode_tasks(TASKS)
    try:
        for entry in entries:
            directory = Path(entry["raw_directory"])
            with np.load(directory / "frames.npz", allow_pickle=False) as archive:
                frames = {key: np.asarray(archive[key]) for key in archive.files}
            with np.load(directory / "physics.npz", allow_pickle=False) as archive:
                physics = {key: np.asarray(archive[key]) for key in archive.files}
            arrays, _ = crop_press_prefix_arrays(frames, physics, entry["press_prefix"])
            count = entry["frames"]
            generators = {key: video_frames(directory / name, fps) for key, name in CAMERAS.items()}
            try:
                for index in range(count):
                    row = {}
                    for key, generator in generators.items():
                        try:
                            row[key] = next(generator)
                        except StopIteration as exc:
                            raise ValueError(f"Source video too short: {directory}/{CAMERAS[key]}") from exc
                    row.update({
                        "observation.state": arrays["state"][index].astype(np.float32),
                        "action": arrays["action"][index].astype(np.float32),
                        "observation.joint_position": arrays["q_actual"][index].astype(np.float32),
                        "action.joint_target": arrays["q_target"][index].astype(np.float32),
                        "observation.sim_time": np.array([arrays["sim_time"][index]], dtype=np.float64),
                        "source_episode_id": np.array([entry["episode_id"]], dtype=np.int64),
                        "source_seed": np.array([entry["seed"]], dtype=np.int64),
                        "floor": np.array([entry["floor"]], dtype=np.int64),
                        "task": entry["task"],
                    })
                    dataset.add_frame(row)
                # Intentionally stop decoding at the cutoff: no later RGB frame
                # is submitted to the writer or retained in the output videos.
                dataset.save_episode()
            finally:
                for generator in generators.values():
                    generator.close()
            print(f"Saved press prefix {entry['episode_id']:06d}, floor {entry['floor']}, {count} frames", flush=True)
    finally:
        dataset.finalize()
    # Source validation is repeated after reading, before publication.
    for entry in entries:
        for name, expected in entry["source_sha256"].items():
            if sha256(Path(entry["raw_directory"]) / name) != expected:
                raise ValueError(f"Source changed during conversion: {entry['episode_id']}/{name}")
    manifest = manifest_for(entries, "press_prefix_part", raw, plan_path, identity)
    manifest["encoding"] = {"vcodec": "h264", "pix_fmt": "yuv420p", "crf": 18, "g": 2,
                            "preset": "ultrafast", "encoder_threads_per_camera": 2,
                            "encoder_queue_maxsize": queue_capacity,
                            "queue_policy": "capacity_exceeds_largest_complete_episode_no_timeout_drops"}
    copy_collection_metadata(raw, staging, collection_record)
    copy_plan(plan_path, staging, manifest["transformation"])
    write_json(staging / MANIFEST, manifest)
    audit_output(staging, destination, raw, len(entries), None)
    write_json(staging / "meta/commit_inventory.json", dataset_inventory(staging))
    staging.rename(destination)
    print(f"Committed audited part {destination.name}: {len(entries)} episodes", flush=True)
    return str(destination)


def write_readme(staging: Path, manifest: dict, audit: dict):
    entries = manifest["episodes"]
    timing = manifest["semantics"]
    lines = ["# PiPER first-button-press demonstrations", "",
             f"{len(entries)} episodes, {sum(e['frames'] for e in entries):,} frames, "
             f"{timing['fps']} Hz; two 640 × 480 RGB D435 camera views. LeRobot v3.0, "
             f"verified with LeRobot {LEROBOT_VERSION} and its PyAV reader.", "",
             "Every episode starts at the recorded folded home pose and ends on the first sampled "
             "target-button orange light. The terminal frame is included. No following dwell, "
             "continued press, withdrawal, or return-home frames are retained. Both videos were "
             "decoded and re-encoded from exactly this prefix; official metadata and statistics "
             "were regenerated. The original full-length raw and LeRobot datasets are preserved.", "",
             "`observation.state` and `action` are 8D "
             "`[x_m, y_m, z_m, qw, qx, qy, qz, gripper_width_m]` in `base_link`, with a "
             "gripper TCP at link6 local `[0, 0, 0.1358]` m. State is measured. Nonterminal "
             f"actions are the original absolute planned TCP targets one sample ({timing['action_horizon_s']:.9g} s) "
             "ahead. The terminal action repeats the preceding planned target, including its "
             "gripper width; it is never replaced with measured state. `action.joint_target` uses "
             "the same terminal repeat for diagnostics. Training action chunks should pad beyond "
             "the terminal frame by repeating that final action.", "",
             "Timestamps start at the first captured post-physics observation (physical time "
             "1/120 s). Local `frame_index` and times retain their source values. Global `index` "
             "and `episode_index` are rebuilt; use `source_episode_id` or the explicit manifest "
             "mapping to locate original recordings.", "",
             "`meta/collection_metadata.json` preserves the original capture configuration and "
             "camera calibration byte-for-byte. Its full-cycle success conditions describe the "
             "recording source. The derived training boundary and terminal-action policy are "
             "defined in `meta/export_manifest.json`, with source hashes and the immutable "
             "`meta/cut_plan.json`. `meta/audit.json` independently verifies the retained numeric "
             "prefix, physical first press, regenerated metadata/statistics, videos, and reader access.", "",
             "| Task | Episodes |", "| --- | ---: |"]
    counts = Counter(entry["floor"] for entry in entries)
    lines.extend(f"| {TASKS[floor - 24]} | {counts[floor]} |" for floor in FLOORS)
    lines.extend(["", "```python", "from pathlib import Path",
                  "from lerobot.datasets.lerobot_dataset import LeRobotDataset",
                  "dataset = LeRobotDataset('local/piper_elevator_press', root=Path('.').resolve(),",
                  "                         video_backend='pyav')", "sample = dataset[0]", "```", ""])
    (staging / "README.md").write_text("\n".join(lines))


def recompute_aggregate_index_statistics(staging: Path) -> dict:
    """Repair the official aggregate's global stats after its index remapping.

    LeRobot 0.6.1 correctly shifts per-episode statistics but aggregates global
    statistics from the original part metadata. Read the actual final Parquet
    columns, then use the official statistics implementation for all moments
    and quantiles of the two affected bookkeeping features.
    """
    import pyarrow.parquet as pq
    from lerobot.datasets.compute_stats import get_feature_stats
    from lerobot.datasets.io_utils import write_stats

    names = ("index", "episode_index")
    chunks = {name: [] for name in names}
    for path in sorted((staging / "data").rglob("*.parquet")):
        table = pq.read_table(path, columns=list(names))
        for name in names:
            chunks[name].append(np.asarray(table[name].to_numpy(), dtype=np.int64))
    values = {name: np.concatenate(parts) for name, parts in chunks.items()}
    if not np.array_equal(values["index"], np.arange(len(values["index"]))):
        raise ValueError("Aggregate global indices are not consecutive")
    checked = 0
    for path in sorted((staging / "meta/episodes").rglob("*.parquet")):
        for row in pq.read_table(path).to_pylist():
            start, stop = int(row["dataset_from_index"]), int(row["dataset_to_index"])
            for name in names:
                sample = values[name][start:stop].astype(np.float64)
                if (len(sample) != row["length"]
                        or (name == "episode_index" and not np.all(sample == row["episode_index"]))):
                    raise ValueError("Aggregate per-episode indices do not match actual rows")
                for stat, expected in (("min", sample.min()), ("max", sample.max()),
                                       ("mean", sample.mean()), ("count", len(sample))):
                    actual = np.asarray(row[f"stats/{name}/{stat}"]).reshape(-1)
                    if actual.size != 1 or not np.allclose(actual, expected, rtol=2e-5, atol=5e-6):
                        raise ValueError(f"Aggregate per-episode {name}/{stat} was not reindexed")
                quantiles = np.array([row[f"stats/{name}/{stat}"]
                                      for stat in ("q01", "q10", "q50", "q90", "q99")])
                if (not np.isfinite(quantiles).all() or np.any(np.diff(quantiles, axis=0) < -1e-8)
                        or quantiles.min() < sample.min() - 1e-5
                        or quantiles.max() > sample.max() + 1e-5):
                    raise ValueError(f"Aggregate per-episode {name} quantiles were not reindexed")
            checked += 1
    stats = read_json(staging / "meta/stats.json")
    for name in names:
        stats[name] = get_feature_stats(values[name].astype(np.float64), axis=0, keepdims=False)
    write_stats(stats, staging)
    return {"features": list(names), "method": "official get_feature_stats on final Parquet values",
            "all_moments_and_quantiles_recomputed": True,
            "per_episode_reindexed_statistics_checked": checked}


def publish(parts: list[Path], output: Path, raw: Path, plan_path: Path,
            identity: dict, episodes_per_task: int | None):
    from lerobot.datasets.aggregate import aggregate_datasets
    entries = [entry for part in parts for entry in read_json(part / MANIFEST)["episodes"]]
    staging = output.with_name(output.name + ".inprogress")
    discard_unpublished(staging)
    aggregate_datasets(repo_ids=[f"local/piper_press_{part.name}" for part in parts], roots=parts,
                       aggr_repo_id="local/piper_elevator_press", aggr_root=staging,
                       concatenate_videos=False, concatenate_data=False)
    manifest = manifest_for(entries, "press_prefix_aggregate", raw, plan_path, identity)
    manifest["aggregate_index_statistics"] = recompute_aggregate_index_statistics(staging)
    ids = [entry["episode_id"] for entry in entries]
    manifest["episode_order"] = {"policy": "source_episode_id_ascending",
                                 "source_id_column": "source_episode_id",
                                 "source_ids_globally_sorted": ids == sorted(ids),
                                 "episode_index_to_source_episode_id": ids}
    manifest["parts"] = [{"path": str(part), "manifest_sha256": sha256(part / MANIFEST)} for part in parts]
    manifest["episodes_per_task"] = episodes_per_task
    copy_collection_metadata(raw, staging, manifest["collection_metadata"])
    copy_plan(plan_path, staging, manifest["transformation"])
    write_json(staging / MANIFEST, manifest)
    audit = audit_output(staging, output, raw, len(entries), episodes_per_task)
    write_readme(staging, manifest, audit)
    write_json(staging / "meta/commit_inventory.json", dataset_inventory(staging))
    staging.rename(output)
    print(f"Published {len(entries)} audited first-press episodes at {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "datasets/piper_elevator_raw_edge_30hz")
    parser.add_argument("--cut-plan", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/piper_elevator_lerobot_press_30hz")
    parser.add_argument("--parts", type=Path)
    parser.add_argument("--source-episode-ids", type=int, nargs="+")
    parser.add_argument("--episodes-per-task", type=int, default=100)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--part-size", type=int, default=25)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if min(args.episodes_per_task, args.part_size, args.workers) < 1:
        parser.error("Episode count, part size and workers must be positive")
    require_version()
    configure_runtime()
    raw, output, plan_path = args.raw.resolve(), args.output.resolve(), args.cut_plan.resolve()
    parts_root = (args.parts or output.with_name(output.name + "_parts")).resolve()
    roots = [raw, output, parts_root]
    if any(a == b or a in b.parents or b in a.parents
           for i, a in enumerate(roots) for b in roots[i + 1:]):
        parser.error("Raw, output and parts directories must be separate and nonoverlapping")
    plan = read_json(plan_path)
    mapping = plan_records(plan)
    selected = args.source_episode_ids if args.source_episode_ids is not None else sorted(mapping)
    if len(set(selected)) != len(selected) or any(source_id not in mapping for source_id in selected):
        parser.error("Source IDs must be unique and present in the immutable cut plan")
    selected = sorted(selected)
    records = [mapping[source_id] for source_id in selected]
    counts = Counter(record["floor"] for record in records)
    if not args.allow_partial and any(counts[floor] != args.episodes_per_task for floor in FLOORS):
        parser.error(f"Selected episodes do not satisfy {args.episodes_per_task} per task: {dict(counts)}")
    _, collection, collection_record = collection_snapshot(raw)
    validate_plan_source(plan, raw, collection, collection_record)
    identity = {"schema_version": 1, "name": TRANSFORM_NAME, "version": TRANSFORM_VERSION,
                "raw_root": str(raw), "collection_sha256": collection_record["sha256"],
                "cut_plan_sha256": sha256(plan_path), "source_episode_ids": selected,
                "part_size": args.part_size, "episodes_per_task": args.episodes_per_task,
                "allow_partial": args.allow_partial, "semantics": press_semantics(collection),
                "lerobot_version": LEROBOT_VERSION,
                "source_code_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in (
                    Path(__file__), ROOT / "scripts/export_lerobot.py",
                    ROOT / "scripts/audit_press_dataset.py", ROOT / "src/pressb/press_prefix.py")}}
    batches = [records[i:i + args.part_size] for i in range(0, len(records), args.part_size)]
    output.parent.mkdir(parents=True, exist_ok=True)
    with export_lock(parts_root, output):
        identity_path = parts_root / "conversion_identity.json"
        if identity_path.exists():
            if read_json(identity_path) != identity:
                raise ValueError("Parts directory belongs to a different plan, source selection, or implementation")
        else:
            if any(parts_root.glob("part_*")):
                raise ValueError("Existing parts lack a conversion identity; refusing to mix them")
            write_json(identity_path, identity)
        parts = [parts_root / f"part_{index:05d}" for index in range(len(batches))]
        unexpected = [path for path in parts_root.glob("part_*")
                      if re.fullmatch(r"part_[0-9]{5}", path.name) and path not in parts]
        if unexpected:
            raise ValueError(f"Unexpected committed parts: {unexpected}")
        pending = []
        for part, batch in zip(parts, batches, strict=True):
            if part.exists():
                validate_committed(part, batch, identity)
                # A saved audit alone cannot authorize mixing changed raw inputs.
                for record in batch:
                    verify_plan_record(raw / f"episode_{record['source_episode_id']:06d}", collection, record)
                print(f"Resumed unchanged audited part {part.name}", flush=True)
            else:
                pending.append((batch, part, raw, plan_path, identity))
        if output.exists():
            validate_committed(output, records, identity)
            print(f"Existing first-press dataset remains verified: {output}", flush=True)
            return
        if args.workers == 1:
            for job in pending:
                write_part(*job)
        elif pending:
            with ProcessPoolExecutor(max_workers=min(args.workers, len(pending)),
                                     mp_context=multiprocessing.get_context("spawn")) as executor:
                futures = {executor.submit(write_part, *job): job[1] for job in pending}
                for future in as_completed(futures):
                    future.result()
                    print(f"Part ready: {futures[future].name}", flush=True)
        publish(parts, output, raw, plan_path, identity,
                None if args.allow_partial else args.episodes_per_task)


if __name__ == "__main__":
    main()
