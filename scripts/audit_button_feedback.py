#!/usr/bin/env python3
"""Audit both videos around every recorded button press/release with exact PTS seeks.

Reuses audit_camera_sync's FK, projection and light summary. ROI geometry and
amber/orange classifier and thresholds are identical to that audit. The selected window is the
whole recorded lit interval plus 0.3 seconds before and after (rounded up to
whole frames). This does not
repeat full-motion geometry or full-video integrity checks.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import time

import av
import numpy as np

import audit_camera_sync as original
from audit_raw_dataset import sampling_parameters
from pressb.panel_metadata import validate_episode_panel_metadata


ROOT = Path(__file__).resolve().parents[1]
MARGIN_SECONDS = 0.3


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decoded_window(path, start, stop, fps, state_times):
    """Yield inclusive [start,stop] by presentation timestamp, never by seek-relative index."""
    with av.open(str(path)) as container:
        require(len(container.streams.video) == 1, "Expected one video stream")
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        require(stream.average_rate is not None and abs(float(stream.average_rate) - fps) <= 1e-8,
                "Video frame rate differs from episode metadata")
        require(stream.time_base is not None, "Video stream lacks a time base")
        target_pts = Fraction(start, fps) / stream.time_base
        container.seek(target_pts.numerator // target_pts.denominator,
                       stream=stream, backward=True, any_frame=False)
        expected = start
        decoded_count = 0
        for frame in container.decode(stream):
            decoded_count += 1
            require(frame.pts is not None and frame.time_base is not None, "Decoded frame lacks PTS")
            timestamp = frame.pts * frame.time_base
            rational_index = timestamp * fps
            require(rational_index.denominator == 1, "Video PTS does not map exactly to an integer sample index")
            index = int(rational_index)
            if index < start:
                continue
            if index > stop:
                break
            require(index == expected, f"Missing/duplicate/out-of-order frame: expected {expected}, got {index}")
            require(abs(float(timestamp) - float(state_times[index])) <= 1e-8,
                    f"PTS/state timestamp mismatch at frame {index}")
            rgb = frame.to_ndarray(format="rgb24")
            require(rgb.shape == (480, 640, 3) and rgb.dtype == np.uint8,
                    f"Wrong RGB format: {rgb.shape}/{rgb.dtype}")
            yield index, float(timestamp), rgb, decoded_count
            expected += 1
            if index == stop:
                break
        require(expected == stop + 1, f"Incomplete decoded window: stopped at {expected-1}, expected {stop}")


def light_sample(rgb, index, timestamp, label, travel, floor, config, camera, intrinsics,
                 panel_offset_x_m=0., panel_offset_y_m=0.):
    # Exact ROI bounds and colour predicate from audit_camera_sync.audit_episode.
    x = config["button_face_x"] + panel_offset_x_m + travel
    y = config["button_column_y"] * (1 if floor < 30 else -1) + panel_offset_y_m
    z = config["button_bottom_z"] + (floor-24) % 6 * config["button_pitch_z"]
    corners = np.array([[x, y+dy, z+dz] for dy, dz in itertools.product([-.013, .013], repeat=2)])
    uv, depth = original.project(corners, camera, intrinsics)
    left, top = np.maximum(np.floor(uv.min(0)).astype(int)-2, [0, 0])
    right, bottom = np.minimum(np.ceil(uv.max(0)).astype(int)+3, [640, 480])
    valid = bool((depth > 0).all() and left < right and top < bottom)
    roi = rgb[top:bottom, left:right] if valid else np.zeros((1, 1, 3), dtype=np.uint8)
    orange = original.feedback_pixel_mask(roi)
    return {"frame": index, "time_s": timestamp, "label_lit": int(label),
            "orange_pixels": int(orange.sum()), "bbox_valid": valid,
            "bbox_xyxy": [int(left), int(top), int(right), int(bottom)]}


def episode(directory, manifest, fk):
    metadata = json.loads((directory / "metadata.json").read_text())
    require(metadata["success"] is True, "Episode did not pass collection")
    require(metadata["collection_fingerprint"] == manifest["collection_fingerprint"],
            "Episode has a different collection fingerprint")
    eid, floor, fps = metadata["episode_id"], metadata["floor"], metadata["fps"]
    rates = sampling_parameters(manifest, metadata)
    offset_x, offset_y = validate_episode_panel_metadata(manifest, metadata)
    require(directory.name == f"episode_{eid:06d}" and floor == 24 + eid % 12, "Invalid episode ID/floor")
    with np.load(directory / "frames.npz", allow_pickle=False) as f:
        frames = {key: f[key] for key in ("q_actual", "lights", "physics_index", "sim_time")}
    with np.load(directory / "physics.npz", allow_pickle=False) as f:
        travel = f["button_travel"]
    count = metadata["num_frames"]
    require(all(len(values) == count for values in frames.values()), "NPZ/metadata frame counts differ")
    require(np.array_equal(frames["physics_index"], np.arange(count) * rates["capture_stride"]),
            "Captured physics indices differ from configured sampling stride")
    require(np.allclose(frames["sim_time"], np.arange(count) / fps, rtol=0, atol=1e-10),
            "Recorded sample times differ from configured frame rate")
    lit = np.flatnonzero(frames["lights"][:, floor-24])
    require(len(lit) > 0 and lit[0] > 0 and lit[-1] < count-1, "Missing recorded off/press/release samples")
    first, last = int(lit[0]), int(lit[-1])
    require(np.array_equal(lit, np.arange(first, last+1)), "Recorded lit interval is not contiguous")
    margin_frames = math.ceil(MARGIN_SECONDS * fps)
    start, stop = max(0, first-margin_frames), min(count-1, last+margin_frames)
    cfg = manifest["config"]
    base = np.array([cfg["robot_base_x"], cfg["robot_base_y"], cfg["table_height"]])
    wrist = manifest["wrist"]["sensor"]
    local = np.eye(4)
    local[:3, :3] = original.quaternion_matrix(wrist["optical_quaternion_wxyz_link6"])
    local[:3, 3] = wrist["optical_position_link6"]
    fixed = np.asarray(manifest["global"]["sensor"]["world_optical_transform"])
    cameras = {}
    for index in range(start, stop+1):
        pose = fk.link6(frames["q_actual"][index])
        pose[:3, 3] += base
        cameras[index] = pose @ local
    result = {"episode_id": eid, "floor": floor, "seed": metadata["seed"],
              "panel_offset_x_m": offset_x, "panel_offset_y_m": offset_y,
              "env_offset_m": metadata["env_offset_m"], "window_first_frame": start,
              "window_last_frame": stop, "window_margin_frames": margin_frames,
              "fps": fps, "lit_frame_indices": lit.tolist(), "views": {}}
    frame_keys = ("label_first_lit_frame", "label_first_unlit_frame", "visible_first_lit_frame",
                  "visible_first_unlit_frame", "strict_zero_after_release_frame")
    for view in ("wrist", "global"):
        samples, decoded = [], 0
        intrinsics = np.asarray(manifest[view]["K"])
        for index, timestamp, rgb, decoded in decoded_window(directory / f"{view}.mp4", start, stop,
                                                            fps, frames["sim_time"]):
            samples.append(light_sample(rgb, index, timestamp, frames["lights"][index, floor-24],
                travel[frames["physics_index"][index], floor-24], floor, cfg,
                cameras[index] if view == "wrist" else fixed, intrinsics,
                panel_offset_x_m=offset_x, panel_offset_y_m=offset_y))
        summary = original.summarize_light(samples, first-start, last-start)
        for key in frame_keys:
            if summary.get(key) is not None:
                summary[key] += start
        result["views"][view] = {"samples": samples, "light_timing": summary,
            "checked_rgb_frames": len(samples), "decoded_frames_including_keyframe_preroll": decoded,
            "exact_pts_match_state_indices": True,
            "release_search_is_window_bounded": True,
            "success": summary.get("aligned") is not False}
    result["success"] = (all(v["success"] for v in result["views"].values())
                          and any(v["light_timing"].get("aligned") is True for v in result["views"].values()))
    return result


def compare_prior(results, prior_paths):
    """Require identical selected samples and timing conclusions in existing full-video reports."""
    comparisons = []
    by_id = {row["episode_id"]: row for row in results}
    for path in prior_paths:
        previous = json.loads(path.read_text())
        rows, checked = previous["episodes"], 0
        for old in rows:
            if old["episode_id"] not in by_id:
                continue
            new = by_id[old["episode_id"]]
            for view in ("wrist", "global"):
                old_samples = {sample["frame"]: sample for sample in old["views"][view]["samples"]}
                for sample in new["views"][view]["samples"]:
                    require(sample == old_samples[sample["frame"]],
                            f"Window/full sample mismatch for episode {new['episode_id']} {view} frame {sample['frame']}")
                require(new["views"][view]["light_timing"] == old["views"][view]["light_timing"],
                        f"Window/full light conclusion mismatch for episode {new['episode_id']} {view}")
                checked += 1
        require(checked > 0, f"Prior report has no selected episodes in common: {path}")
        comparisons.append({"report": str(path.resolve()), "sha256": sha256(path),
                            "matched_episode_views": checked, "exact_samples_and_conclusions_match": True})
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes-per-task", type=int, default=100)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--episode-ids", type=int, nargs="+")
    parser.add_argument("--compare-report", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new report to preserve previous evidence")
    if args.episodes_per_task < 1 or (args.episode_ids is not None and
            (len(args.episode_ids) != len(set(args.episode_ids)) or min(args.episode_ids) < 0)):
        parser.error("Invalid episode count or IDs")
    raw = args.raw.resolve()
    manifest = json.loads((raw / "collection_metadata.json").read_text())
    rates = sampling_parameters(manifest)
    fk = original.UrdfForward(ROOT / manifest["config"]["robot_urdf"])
    directories = sorted(path for path in raw.glob("episode_*")
                         if path.is_dir() and re.fullmatch(r"episode_[0-9]{6,}", path.name))
    if args.episode_ids is not None:
        directories = [raw / f"episode_{eid:06d}" for eid in args.episode_ids]
    report = {"success": False, "source_root": str(raw),
              "collection_fingerprint": manifest["collection_fingerprint"],
              "schema": manifest["raw_schema_version"], "episodes_per_task": args.episodes_per_task,
              "allow_partial": args.allow_partial, "requested_episode_ids": args.episode_ids,
              **rates, "window_margin_seconds": MARGIN_SECONDS,
              "scope": "Every selected episode's complete lit interval and 0.3 seconds on either side, "
                       "rounded up to whole frames; "
                       "exact presentation timestamp selection after keyframe seek. No full-motion geometry check.",
              "light_pixel_classifier": original.LIGHT_PIXEL_CLASSIFIER,
              "thresholds": {"orange": original.LIGHT_PIXEL_CLASSIFIER["predicate"],
                             "minimum_median_lit_pixels": 20, "dominant_threshold": "max(5,0.25*median_lit_pixels)"},
              "limitations": "Occlusion/too-small ROI is reported as insufficient visibility. At least one view "
                             "must establish correct timing. Release/strict-zero searches stop at the recorded "
                             "window end; a residual lasting past that end is never accepted as aligned. "
                             "Full-video completeness and state/geometry checks are audited separately.",
              "source_code_sha256": {"audit_camera_sync.py": sha256(Path(original.__file__)),
                                     "audit_button_feedback.py": sha256(Path(__file__))},
              "episodes": [], "errors": []}
    started = time.monotonic()
    for index, directory in enumerate(directories):
        try:
            row = episode(directory, manifest, fk)
            report["episodes"].append(row)
            if not row["success"]:
                report["errors"].append(f"Button feedback failed: {directory.name}")
        except Exception as exc:
            report["errors"].append(f"{directory.name}: {type(exc).__name__}: {exc}")
        if (index+1) % 50 == 0 or index+1 == len(directories):
            print(json.dumps({"processed": index+1, "selected": len(directories),
                              "errors": len(report["errors"]), "elapsed_s": time.monotonic()-started}), flush=True)
    counts = Counter(row["floor"] for row in report["episodes"])
    report["floor_counts"] = {str(floor): counts[floor] for floor in range(24, 36)}
    if not report["episodes"] or (not args.allow_partial and
            any(counts[floor] != args.episodes_per_task for floor in range(24, 36))):
        report["errors"].append("Incomplete required episode counts")
    try:
        report["prior_report_comparisons"] = compare_prior(report["episodes"], args.compare_report)
    except Exception as exc:
        report["errors"].append(f"Prior comparison: {type(exc).__name__}: {exc}")
    views = [value for row in report["episodes"] for value in row["views"].values()]
    report.update(total_episodes=len(report["episodes"]), passed_episodes=sum(r["success"] for r in report["episodes"]),
        checked_rgb_frames=sum(v["checked_rgb_frames"] for v in views),
        decoded_frames_including_keyframe_preroll=sum(v["decoded_frames_including_keyframe_preroll"] for v in views),
        visible_aligned_views=sum(v["light_timing"].get("aligned") is True for v in views),
        visible_failed_views=sum(v["light_timing"].get("aligned") is False for v in views),
        insufficient_visibility_views=sum(not v["light_timing"]["sufficient_visibility"] for v in views),
        elapsed_s=time.monotonic()-started)
    report["success"] = not report["errors"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key != "episodes"}, indent=2))
    raise SystemExit(0 if report["success"] else 1)


if __name__ == "__main__":
    main()
