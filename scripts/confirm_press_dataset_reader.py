#!/usr/bin/env python3
"""Confirm a published first-press LeRobot dataset using its official reader.

Run in the isolated lerobot==0.6.1 environment. Every episode's first, middle,
and last observation is decoded through PyAV in DataLoader batches. Eight-step
action chunks must stay inside the episode and expose exact padding masks.
Complete-video and visible-first-light coverage are inherited only from the
successful, hash-bound publication audit; this script does not repeat that scan.
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
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CAMERAS = ("observation.images.wrist", "observation.images.global")
MANIFEST = "meta/export_manifest.json"
CHUNK_SIZE = 8


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def scalar(value):
    value = numpy(value)
    require(value.size == 1, "Expected a scalar feature")
    return value.reshape(-1)[0].item()


def validate_published_evidence(root, manifest, expected_episodes):
    """Bind the full audit and all committed bytes to this published location."""
    require(manifest.get("kind") == "press_prefix_aggregate" and manifest.get("schema_version") == 2,
            "Expected a published press-prefix aggregate")
    require(manifest.get("lerobot_version") == "0.6.1", "Wrong exported LeRobot version")
    audit_path = root / "meta/audit.json"
    audit = read_json(audit_path)
    require(audit.get("success") is True and not audit.get("errors") and audit.get("full_video_decode") is True,
            "A successful complete publication audit is required")
    require(audit.get("dataset") == str(root), "Audit does not identify the published dataset path")
    require(audit.get("manifest_sha256") == sha256(root / MANIFEST), "Publication audit/manifest hash mismatch")
    require(audit.get("audit_source_sha256") == sha256(ROOT / "scripts/audit_press_dataset.py"), "Publication audit code changed")
    entries = manifest["episodes"]
    total = sum(entry["frames"] for entry in entries)
    require(len(entries) == audit.get("total_episodes") == expected_episodes, "Published episode count differs")
    require(total == audit.get("total_frames") and audit.get("decoded_rgb_frames") == total * 2,
            "Publication audit did not decode every retained RGB frame")
    plan_path = root / "meta/cut_plan.json"
    cut_sha = sha256(plan_path)
    require(cut_sha == audit.get("cut_plan_sha256") == manifest["transformation"]["cut_plan"]["sha256"],
            "Cut plan is not bound to manifest and audit")
    require(manifest["transformation"]["terminal_action_policy"]["measured_state_used_as_target"] is False,
            "Measured terminal state was declared as an action")
    semantics = manifest["semantics"]
    require(semantics.get("episode_end") == "first_sampled_target_light_on"
            and semantics.get("terminal_action") == "repeat_previous_planned_target"
            and semantics.get("action_joint_target") == "next_sample_endpoint_except_terminal_repeat_previous_planned_target",
            "Wrong terminal action semantics")
    inventory_path = root / "meta/commit_inventory.json"
    inventory = read_json(inventory_path)
    for relative in (MANIFEST, "meta/audit.json", "meta/cut_plan.json", "meta/info.json", "meta/tasks.parquet"):
        require(relative in inventory, f"Commit inventory lacks {relative}")
    for relative, digest in inventory.items():
        path = root / relative
        require(path.resolve().is_relative_to(root) and path.is_file(), f"Invalid inventory path: {relative}")
        require(sha256(path) == digest, f"Published file changed since complete audit: {relative}")
    current_payload = {str(p.relative_to(root)) for parent in ("data", "videos", "meta/episodes")
                       for p in (root / parent).rglob("*") if p.is_file()}
    inventoried_payload = {p for p in inventory if p.startswith(("data/", "videos/", "meta/episodes/"))}
    require(current_payload == inventoried_payload, "Untracked published numeric/video/episode files")
    for video in audit["video_files"]:
        require(inventory.get(video["path"]) == video["sha256"], "Audited video identity differs from committed video")
    audited = {entry["source_episode_id"]: entry for entry in audit["episodes"]}
    require(len(audited) == len(entries), "Duplicate or incomplete audited episodes")
    return audit, audited, {
        "manifest_sha256": sha256(root / MANIFEST), "publication_audit_sha256": sha256(audit_path),
        "publication_audit_code_sha256": audit["audit_source_sha256"],
        "commit_inventory_sha256": sha256(inventory_path), "committed_files_verified": len(inventory),
        "cut_plan_sha256": cut_sha, "prior_full_decode_rgb_frames": audit["decoded_rgb_frames"],
    }


def prepare_samples(dataset, manifest, audited):
    """Expected values come directly from immutable raw arrays, not crop helpers."""
    samples, episodes = [], []
    source_ids, expected_start = set(), 0
    for episode_index, entry in enumerate(manifest["episodes"]):
        source_id, floor = entry["episode_id"], entry["floor"]
        require(source_id not in source_ids, "Duplicate source episode ID")
        source_ids.add(source_id)
        task = f"Press {floor} floor."
        meta = dataset.meta.episodes[episode_index]
        start, stop = int(meta["dataset_from_index"]), int(meta["dataset_to_index"])
        cut = entry["press_prefix"]
        k, n = cut["cut_frame_index"], entry["frames"]
        require(k >= 1 and n == k + 1 == cut["kept_frames"] == stop - start == int(meta["length"]),
                "Cut, manifest and official reader episode ranges differ")
        require(start == expected_start and int(meta["episode_index"]) == episode_index, "Episode range gap or misordered identity")
        require(entry["task"] == task and meta["tasks"] == [task], "Task metadata mismatch")
        evidence = audited[source_id]
        require(evidence["episode_index"] == episode_index and evidence["frames"] == n and evidence["floor"] == floor,
                "Audited episode differs from official reader episode")
        contact = evidence["contact"]
        require(contact["cut_frame_index"] == k and contact["sampled_lit_frames_retained"] == 1
                and contact["terminal_phase"] == "press"
                and (k - 1) * 4 < contact["first_press_physics_index"] <= k * 4,
                "Audit lacks the exact first-light physical boundary")
        wrist = evidence["views"]["wrist"]
        require(wrist.get("sufficient_visibility") is True and wrist["visible_first_lit_frame"] == k
                and wrist["terminal_amber_pixels"] >= 20
                and wrist["maximum_preterminal_amber_pixels"] < wrist["dominant_threshold_pixels"],
                "Audit lacks visible first-light evidence in the final wrist image")
        directory = Path(manifest["raw_root"]) / f"episode_{source_id:06d}"
        frame_path = directory / "frames.npz"
        require(sha256(frame_path) == entry["source_sha256"]["frames.npz"] == cut["source_files"]["frames.npz"]["sha256"],
                "Numeric source differs from published provenance")
        with np.load(frame_path, allow_pickle=False) as source:
            frames = dict(source)
        target = floor - 24
        lit = np.flatnonzero(frames["lights"][:, target])
        require(len(lit) and int(lit[0]) == k and not frames["lights"][:k].any()
                and frames["lights"][k].sum() == 1 and frames["phase"][k] == "press", "Source first-light boundary mismatch")
        require(np.array_equal(cut["terminal_action"], frames["action"][k - 1])
                and np.array_equal(cut["terminal_q_target"], frames["q_target"][k - 1])
                and cut["terminal_measured_state_used"] is False, "Cut plan terminal target mismatch")
        selection = [0, k // 2, k]
        require(len(set(selection)) == 3, "Episode too short for three distinct samples")
        for local in selection:
            requested = local + np.arange(CHUNK_SIZE)
            # Source row k-1 already targets the final sampled physics step.
            target_rows = np.minimum(requested, k - 1)
            expected = dict(dataset_index=start + local, frame_index=local, episode_index=episode_index,
                            source_episode_id=source_id, floor=floor, task=task, frames=n,
                            state=frames["state"][local].astype(np.float32),
                            action=frames["action"][target_rows].astype(np.float32),
                            joint_target=frames["q_target"][target_rows].astype(np.float32),
                            padding=requested > k)
            row = dataset.get_raw_item(start + local)
            require(np.array_equal(numpy(row["observation.state"]), expected["state"]), "Raw official-reader state differs")
            require(np.array_equal(numpy(row["action"]), expected["action"][0]), "Raw official-reader action differs")
            exact = dataset.hf_dataset.with_format(None)[start + local]
            require(abs(scalar(exact["observation.sim_time"]) - local / 30) < 1e-9, "Stored simulation clock differs")
            samples.append(expected)
        episodes.append(dict(episode_index=episode_index, source_episode_id=source_id, floor=floor, frames=n,
                             dataset_from_index=start, dataset_to_index=stop, sample_frame_indices=selection,
                             terminal_action_clamped=True, visible_first_lit_frame=k,
                             terminal_action_padding_mask=[False] + [True] * 7))
        expected_start = stop
    require(expected_start == len(dataset), "Dataset contains rows outside the episode ranges")
    return samples, episodes


def confirm_reader(dataset_root, expected_episodes, *, batch_size=4):
    root = Path(dataset_root).resolve()
    report = dict(success=False, dataset=str(root), audited_utc=datetime.now(timezone.utc).isoformat(),
                  errors=[], reader_code_sha256=sha256(__file__), expected_episodes=expected_episodes,
                  full_video_decode_repeated=False, sample_policy="first_middle_last_every_episode",
                  chunk_size=CHUNK_SIZE,
                  delta_timestamps={key: [i/30 for i in range(CHUNK_SIZE)] for key in ("action", "action.joint_target")},
                  dataloader={"batch_size": batch_size, "num_workers": 0, "shuffle": False},
                  limits=["Complete RGB coverage and the visible first-light boundary rely on the hash-bound publication audit."])
    try:
        require(version("lerobot") == "0.6.1", "Use the isolated lerobot==0.6.1 environment")
        require(expected_episodes > 0 and batch_size > 0, "Counts must be positive")
        for key, value in {"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                           "HF_HOME": str(ROOT / ".cache/hf"), "HF_DATASETS_DISABLE_PROGRESS_BARS": "1",
                           "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}.items():
            os.environ.setdefault(key, value)
        import av
        av.logging.set_level(av.logging.ERROR)
        import torch
        torch.set_num_threads(2)
        from torch.utils.data import DataLoader, Subset
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        import lerobot.datasets.dataset_reader as reader_module
        manifest = read_json(root / MANIFEST)
        audit, audited, evidence = validate_published_evidence(root, manifest, expected_episodes)
        report.update(evidence)
        report["versions"] = {name: version(name) for name in ("lerobot", "av", "torch")}
        report["official_reader_source_sha256"] = sha256(reader_module.__file__)
        dataset = LeRobotDataset("local/piper_elevator_press", root=root, video_backend="pyav")
        require(dataset.num_episodes == expected_episodes and dataset.fps == 30
                and dataset.meta.info.codebase_version == "v3.0", "Official reader counts, rate or version differ")
        require(len(dataset.meta.tasks) == 12, "Expected twelve tasks")
        for floor in range(24, 36):
            require(dataset.meta.get_task_index(f"Press {floor} floor.") == floor - 24, "Official task table is reordered")
        samples, episodes = prepare_samples(dataset, manifest, audited)
        deltas = {key: [i/30 for i in range(CHUNK_SIZE)] for key in ("action", "action.joint_target")}
        chunk_dataset = LeRobotDataset("local/piper_elevator_press", root=root, video_backend="pyav", delta_timestamps=deltas)
        loader = DataLoader(Subset(chunk_dataset, [s["dataset_index"] for s in samples]), batch_size=batch_size,
                            num_workers=0, shuffle=False, drop_last=False)
        observed = batches = images = 0
        for batch in loader:
            size = len(batch["task"])
            for key in CAMERAS:
                values = batch[key]
                require(values.dtype == torch.float32 and tuple(values.shape) == (size, 3, 480, 640)
                        and torch.isfinite(values).all().item() and values.min().item() >= 0 and values.max().item() <= 1,
                        "DataLoader RGB dtype, shape or range differs")
                require((values.flatten(1).std(dim=1) > .01).all().item(), "DataLoader returns blank/constant RGB")
                images += size
            require(tuple(batch["observation.state"].shape) == (size, 8)
                    and tuple(batch["action"].shape) == (size, CHUNK_SIZE, 8)
                    and tuple(batch["action.joint_target"].shape) == (size, CHUNK_SIZE, 6), "Batch numeric feature shapes differ")
            require(all(batch[key].dtype == torch.float32 for key in ("observation.state", "action", "action.joint_target"))
                    and all(batch[key].dtype == torch.bool for key in ("action_is_pad", "action.joint_target_is_pad")),
                    "Batch numeric or padding feature dtype differs")
            for i in range(size):
                expected = samples[observed]
                for key in ("index", "frame_index", "episode_index", "source_episode_id", "floor", "task_index"):
                    wanted = expected["dataset_index"] if key == "index" else expected["floor"] - 24 if key == "task_index" else expected[key]
                    require(scalar(batch[key][i]) == wanted, f"Batch identity mismatch: {key}")
                require(batch["task"][i] == expected["task"], "Batch task text differs")
                require(abs(scalar(batch["timestamp"][i]) - expected["frame_index"]/30) < 2e-6, "Batch timestamp mismatch")
                for key, wanted in (("observation.state", expected["state"]), ("action", expected["action"]),
                                    ("action.joint_target", expected["joint_target"]),
                                    ("action_is_pad", expected["padding"]), ("action.joint_target_is_pad", expected["padding"])):
                    require(np.array_equal(numpy(batch[key][i]), wanted), f"Chunk values/padding differ: {key}, source={expected['source_episode_id']}")
                observed += 1
            batches += 1
            if batches % 100 == 0:
                print(f"READER samples={observed}/{len(samples)} batches={batches}", flush=True)
        require(observed == expected_episodes * 3 and images == observed * 2, "Incomplete official-reader sample coverage")
        require(sha256(root / MANIFEST) == report["manifest_sha256"]
                and sha256(root / "meta/audit.json") == report["publication_audit_sha256"], "Published provenance changed during reading")
        counts = Counter(e["floor"] for e in episodes)
        report.update(total_episodes=len(episodes), total_frames=len(dataset), sampled_frames=observed,
                      sampled_images=images, raw_numeric_samples_checked=observed,
                      action_chunks_checked=observed, terminal_padding_checks=len(episodes),
                      dataloader_batches=batches, floor_counts={str(f): counts[f] for f in range(24, 36)}, episodes=episodes,
                      success=True)
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Report already exists; preserve it and choose a new report path")
    report = confirm_reader(args.dataset, args.expected_episodes, batch_size=args.batch_size)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=args.report.parent, suffix=".reader.tmp", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, args.report)
    finally:
        temporary.unlink()
    print(json.dumps({k: v for k, v in report.items() if k != "episodes"}, indent=2), flush=True)
    raise SystemExit(0 if report["success"] else 1)
