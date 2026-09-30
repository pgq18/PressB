#!/usr/bin/env python3
"""Independent audit of first-success PiPER LeRobot prefixes (no return phase).

This verifier never imports the prefix exporter or its cut implementation. It
reconstructs contact hysteresis from the source physics, compares every numeric
row, uses the official LeRobot reader, and fully decodes published RGB streams.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import itertools
from pathlib import Path

import av
import numpy as np

from audit_camera_sync import LIGHT_PIXEL_CLASSIFIER, feedback_pixel_mask, project, quaternion_matrix
from audit_lerobot import IndependentFK, audit_collection_metadata, audit_episode_collection, _numpy
from export_lerobot import (CAMERAS, FLOORS, IMAGE_SHAPE, LEROBOT_VERSION, MANIFEST, SOURCE_FILES,
                            PANEL_EPISODE_CONTEXT, validate_episode_panel_metadata,
                            TASKS, URDF, collection_timing, configure_runtime, feature_spec,
                            read_json, require_version, semantics_for_collection, sha256, write_json)


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def source_prefix(frames, physics, metadata, collection):
    """Find the first visible-sampling opportunity using actual contact physics."""
    cfg = collection["config"]
    validate_episode_panel_metadata(collection, metadata)
    rate = collection_timing(collection)
    require(rate["fps"] == 30 and rate["physics_hz"] == 120, "Expected 30/120 Hz source")
    stride = rate["capture_stride"]
    n = len(frames["state"])
    p = len(physics["q_actual"])
    target = int(metadata["floor"]) - 24
    require(0 <= target < 12, "Invalid target floor")
    for key in ("button_travel", "contact_force", "lights"):
        require(physics[key].shape == (p, 12) and np.isfinite(physics[key]).all(), f"Invalid physics {key}")
    require(p == metadata["physics_steps"] and n == metadata["num_frames"], "Source lengths differ from metadata")
    indices = np.arange(n, dtype=np.int64) * stride
    require(indices[-1] < p and np.array_equal(frames["physics_index"], indices), "Source sample/physics mapping")
    require(np.allclose(frames["sim_time"], np.arange(n) / rate["fps"], atol=1e-10, rtol=0), "Source sample times")
    lamps = np.zeros(12, dtype=bool)
    rebuilt = np.zeros((p, 12), dtype=np.uint8)
    events = []
    for i, (travel, force) in enumerate(zip(physics["button_travel"], physics["contact_force"])):
        pressed = ~lamps & (travel >= cfg["press_threshold"]) & (force > .02)
        released = lamps & (travel <= cfg["release_threshold"])
        for j in np.flatnonzero(pressed):
            events.append(("pressed", int(j) + 24, i))
        for j in np.flatnonzero(released):
            events.append(("released", int(j) + 24, i))
        lamps = (lamps | pressed) & ~released
        rebuilt[i] = lamps
    require(np.array_equal(rebuilt, physics["lights"]), "Source lights are inconsistent with physical contact")
    require(np.array_equal(frames["lights"], rebuilt[indices]), "Source image light labels differ from physics")
    require(events == [(e["type"], e["floor"], e["physics_index"]) for e in metadata["events"]], "Source event metadata differs from physics")
    lit = np.flatnonzero(rebuilt[indices, target])
    require(len(lit) > 0, "No captured physical success")
    k = int(lit[0])
    require(k > 0, "Initial frame must precede contact")
    endpoint = int(indices[k])
    prefix_events = [e for e in events if e[2] <= endpoint]
    require(len(prefix_events) == 1 and prefix_events[0][:2] == ("pressed", target + 24), "Prefix has incorrect/extra contact events")
    trigger = prefix_events[0][2]
    require(indices[k - 1] < trigger <= endpoint, "First physical contact lies outside terminal sample interval")
    require(not rebuilt[indices[:k]].any() and rebuilt[endpoint].sum() == 1, "Prefix must contain exactly one lit terminal frame")
    require(np.all(rebuilt[trigger:endpoint + 1, target]), "Light extinguishes before terminal frame")
    require(frames["phase"][k] == "press", "Terminal sample is not pressing")
    require(not np.isin(frames["phase"][:k + 1], ["hold", "retract", "return_home"]).any(), "Post-success phase retained")
    require(not metadata["unexpected_collisions"], "Source collider monitor reports unexpected contact")
    require(np.max(np.abs(physics["q_actual"][0] - cfg["home_q"])) < cfg["home_tolerance_rad"], "Source does not start at home")
    require(np.array_equal(frames["q_actual"], physics["q_actual"][indices]), "Source measured joint alignment")
    require(np.array_equal(frames["q_target"][k - 1], physics["q_command"][endpoint]), "Terminal copied target was not executed at endpoint")
    clipped = {key: value[:k + 1].copy() for key, value in frames.items()}
    for key in ("action", "q_target"):
        clipped[key][-1] = frames[key][k - 1]
    evidence = {"cut_frame_index": k, "kept_frames": k + 1,
                "kept_physics_steps": endpoint + 1, "first_press_physics_index": trigger,
                "end_time_s": k / rate["fps"], "removed_frames": n - k - 1,
                "terminal_phase": str(frames["phase"][k]),
                "trigger_travel_m": float(physics["button_travel"][trigger, target]),
                "trigger_force_n": float(physics["contact_force"][trigger, target]),
                "sampled_lit_frames_retained": 1}
    return clipped, evidence


def verify_numeric_prefix(values, expected, start, episode_index, metadata):
    n = len(expected["state"])
    for key, source in (("observation.state", "state"), ("action", "action"),
                        ("observation.joint_position", "q_actual"), ("action.joint_target", "q_target")):
        actual = np.asarray(values[key])
        require(actual.shape == expected[source].shape and np.isfinite(actual).all(), f"Invalid numeric prefix {key}")
        require(np.array_equal(actual, expected[source].astype(np.float32)), f"Numeric prefix/terminal clamp differs: {key}")
    require(np.allclose(np.asarray(values["observation.sim_time"]).reshape(-1), expected["sim_time"], atol=1e-9, rtol=0), "Simulation timestamps differ")
    require(np.allclose(np.asarray(values["timestamp"]).reshape(-1), np.arange(n) / 30, atol=2e-6, rtol=0), "Frame timestamps differ")
    for key, value in (("frame_index", np.arange(n)), ("index", np.arange(start, start + n)),
                       ("episode_index", episode_index), ("task_index", metadata["floor"] - 24),
                       ("source_episode_id", metadata["episode_id"]), ("source_seed", metadata["seed"]),
                       ("floor", metadata["floor"])):
        require(np.all(np.asarray(values[key]).reshape(-1) == value), f"Wrong {key}")


def compare_rgb(actual, reference):
    require(actual.shape == reference.shape == IMAGE_SHAPE, "Wrong RGB dimensions")
    require(np.isfinite(actual).all(), "Nonfinite RGB")
    diff = actual.astype(np.float32) - reference.astype(np.float32)
    mae = float(np.abs(diff).mean())
    psnr = float(10 * np.log10(255**2 / max(float(np.square(diff).mean()), 1e-12)))
    require(mae <= 8 and psnr >= 25, f"RGB differs from source: MAE={mae:.3f}, PSNR={psnr:.3f}")
    return {"source_mae_255": mae, "source_psnr_db": psnr}


def video_images(path, fps=30):
    """Yield decoded RGB with absolute PTS checks, including first PTS zero."""
    with av.open(str(path)) as container:
        require(len(container.streams.video) == 1, f"Expected one video stream: {path}")
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        require(stream.average_rate == Fraction(fps), f"Wrong actual video rate: {path}")
        for i, frame in enumerate(container.decode(stream)):
            require(frame.pts is not None and frame.time_base is not None, f"Missing PTS: {path}")
            require(abs(float(frame.pts * frame.time_base) - i / fps) < 1e-4, f"Wrong PTS at {path}:{i}")
            rgb = frame.to_ndarray(format="rgb24")
            require(rgb.shape == IMAGE_SHAPE and rgb.dtype == np.uint8, f"Invalid RGB: {path}")
            yield rgb


def target_boxes(collection, expected, physics, floor, fk, panel_offset_y_m=0., panel_offset_x_m=0.):
    cfg = collection["config"]
    transforms = fk.batch(expected["q_actual"])
    transforms[:, :3, 3] -= transforms[:, :3, :3] @ np.array([0, 0, .1358])
    transforms[:, :3, 3] += [cfg["robot_base_x"], cfg["robot_base_y"], cfg["table_height"]]
    sensor = collection["wrist"]["sensor"]
    local = np.eye(4)
    local[:3, :3] = quaternion_matrix(sensor["optical_quaternion_wxyz_link6"])
    local[:3, 3] = sensor["optical_position_link6"]
    wrist = transforms @ local
    fixed = np.asarray(collection["global"]["sensor"]["world_optical_transform"])
    result = {}
    for view in ("wrist", "global"):
        boxes = []
        for i, index in enumerate(expected["physics_index"]):
            x = cfg["button_face_x"] + panel_offset_x_m + physics["button_travel"][index, floor - 24]
            y = cfg["button_column_y"] * (1 if floor < 30 else -1) + panel_offset_y_m
            z = cfg["button_bottom_z"] + (floor - 24) % 6 * cfg["button_pitch_z"]
            corners = [[x, y + dy, z + dz] for dy, dz in itertools.product([-.013, .013], repeat=2)]
            uv, depth = project(corners, wrist[i] if view == "wrist" else fixed, np.asarray(collection[view]["K"]))
            left, top = np.maximum(np.floor(uv.min(0)).astype(int) - 2, [0, 0])
            right, bottom = np.minimum(np.ceil(uv.max(0)).astype(int) + 3, [640, 480])
            boxes.append([int(left), int(top), int(right), int(bottom)] if (depth > 0).all() and left < right and top < bottom else None)
        result[view] = boxes
    return result


def amber_count(rgb, box):
    if box is None:
        return 0
    left, top, right, bottom = box
    return int(feedback_pixel_mask(rgb[top:bottom, left:right]).sum())


def verify_light_prefix(counts, view):
    require(len(counts) > 1, "Insufficient light evidence")
    terminal = int(counts[-1])
    if view == "global" and terminal < 20:
        return {"sufficient_visibility": False, "terminal_amber_pixels": terminal,
                "reason": "Global target may be occluded; wrist provides required terminal evidence"}
    require(terminal >= 20, f"{view} successful terminal frame is not visibly illuminated")
    threshold = max(5., .25 * terminal)
    require(max(counts[:-1]) < threshold, f"{view} light appears before the terminal frame")
    return {"sufficient_visibility": True, "terminal_amber_pixels": terminal,
            "maximum_preterminal_amber_pixels": int(max(counts[:-1])),
            "dominant_threshold_pixels": threshold, "visible_first_lit_frame": len(counts) - 1}


def audit_stats(stats, numeric, total_frames, image_moments):
    """Recompute moments; tolerate documented float32 writer accumulation only."""
    results = {}
    for key, chunks in numeric.items():
        x = np.concatenate(chunks).astype(np.float64)
        if x.ndim == 1:
            x = x[:, None]
        s = stats[key]
        require(s["count"] == [total_frames], f"Wrong stats count: {key}")
        errors = {}
        for name, expected in (("min", x.min(0)), ("max", x.max(0)), ("mean", x.mean(0)), ("std", x.std(0))):
            actual = np.asarray(s[name]).reshape(-1)
            require(actual.shape == expected.shape and np.isfinite(actual).all(), f"Invalid stats {key}/{name}")
            # StreamingStats in LeRobot 0.6.1 keeps float32 mean/squares;
            # tiny-variance channels can lose precision in their subtraction.
            if name == "std":
                # Judge the variance error at the scale where float32
                # mean-of-squares subtraction occurs. In the already verified
                # 12-episode source export, constant .008m actions have a
                # spurious 3.10e-5m std (relative second-moment error 1.50e-5).
                variance_bound = 2e-5 * np.square(x).mean(0) + 1e-12
                require((actual >= 0).all() and (np.abs(actual**2 - expected**2) <= variance_bound).all(), f"Stats differ from cropped rows: {key}/{name}")
            else:
                require(np.allclose(actual, expected, atol=5e-6, rtol=2e-5), f"Stats differ from cropped rows: {key}/{name}")
            errors[name] = float(np.abs(actual - expected).max())
        quantiles = np.array([s[q] for q in ("q01", "q10", "q50", "q90", "q99")])
        require(np.isfinite(quantiles).all() and (np.diff(quantiles, axis=0) >= -1e-8).all(), f"Invalid approximate quantiles: {key}")
        require((quantiles >= x.min(0) - 1e-5).all() and (quantiles <= x.max(0) + 1e-5).all(), f"Quantiles outside bounds: {key}")
        results[key] = errors
    for key in CAMERAS:
        s = stats[key]
        # Official streaming writer samples every fourth pixel of every RGB.
        require(s["count"] == [total_frames * 120 * 160], f"Wrong cropped image-stat count: {key}")
        for name in ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99"):
            value = np.asarray(s[name])
            require(value.shape == (3, 1, 1) and np.isfinite(value).all() and value.min() >= -1e-8 and value.max() <= 1 + 1e-8, f"Invalid image statistic {key}/{name}")
        moment = image_moments[key]
        mean = moment["sum"] / moment["count"] / 255
        std = np.sqrt(np.maximum(0, moment["sum2"] / moment["count"] - (mean * 255)**2)) / 255
        # Images were re-encoded after writer statistics: allow measured H264
        # differences, rather than claiming bit-exact post-encode moments.
        require(np.max(np.abs(np.asarray(s["mean"]).reshape(3) - mean)) <= .025, f"RGB mean differs from decoded cropped data: {key}")
        require(np.max(np.abs(np.asarray(s["std"]).reshape(3) - std)) <= .025, f"RGB std differs from decoded cropped data: {key}")
        results[key] = {"decoded_downsampled_mean": mean.tolist(), "decoded_downsampled_std": std.tolist(), "comparison_tolerance": .025}
    return results


def audit_press_dataset(dataset_root: Path, raw: Path | None = None, *, expected_episodes: int | None = None,
                        episodes_per_task: int | None = None, allow_partial: bool = False,
                        decode_all: bool = True) -> dict:
    require_version()
    configure_runtime()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    root = Path(dataset_root).resolve()
    report = {"success": False, "dataset": str(root), "audited_utc": datetime.now(timezone.utc).isoformat(),
              "audit_source_sha256": sha256(Path(__file__)), "errors": [], "episodes": [], "video_files": [],
              "allow_partial": allow_partial, "expected_episodes": expected_episodes,
              "full_video_decode": decode_all, "light_pixel_classifier": LIGHT_PIXEL_CLASSIFIER,
              "limits": ["Global target feedback may be occluded; wrist feedback is mandatory.",
                         "Published RGB is lossy H264; first/middle/last official-reader images use MAE<=8 and PSNR>=25 against source.",
                         "Numeric moments are recomputed; histogram quantiles are checked for ordering/bounds, not exact equality."]}
    try:
        require(decode_all, "This audit requires complete output video decoding")
        require(expected_episodes is not None and expected_episodes > 0, "Explicit expected_episodes is required")
        manifest = read_json(root / MANIFEST)
        require(manifest["schema_version"] == 2 and manifest["kind"] in ("press_prefix_part", "press_prefix_aggregate"), "Not a press-prefix manifest")
        require(manifest["tasks"] == TASKS and manifest["lerobot_version"] == LEROBOT_VERSION, "Manifest vocabulary/version")
        raw = Path(raw or manifest["raw_root"]).resolve()
        collection, evidence = audit_collection_metadata(root, raw, manifest)
        rate = collection_timing(collection)
        require(rate["fps"] == 30, "Expected 30 Hz capture")
        semantics = {**semantics_for_collection(collection), "episode_end": "first_sampled_target_light_on",
                     "terminal_action": "repeat_previous_planned_target",
                     "action_joint_target": "next_sample_endpoint_except_terminal_repeat_previous_planned_target"}
        require(manifest["semantics"] == semantics, "Wrong prefix/action semantics")
        require(sha256(URDF) == manifest["urdf"]["sha256"], "URDF changed")
        cut_path = root / "meta/cut_plan.json"
        plan = read_json(cut_path)
        plan_entries = {e["source_episode_id"]: e for e in plan["episodes"]}
        require(len(plan_entries) == len(plan["episodes"]), "Duplicate cut-plan source IDs")
        transformation = manifest["transformation"]
        require(transformation.get("name") == "first_press_prefix" and transformation.get("version") == 1, "Unknown transformation")
        policy = transformation.get("terminal_action_policy", {})
        require(policy.get("action") == "source.action[cut_frame_index - 1]" and
                policy.get("action.joint_target") == "source.q_target[cut_frame_index - 1]" and
                policy.get("measured_state_used_as_target") is False and
                policy.get("nonterminal_actions") == "source prefix unchanged", "Wrong terminal-action policy declaration")
        report["transformation"] = transformation
        # Exact transformation SHA field is checked below without trusting its
        # prose/version as the actual cut algorithm.
        declared_cut_sha = transformation.get("cut_plan_sha256")
        if declared_cut_sha is None and isinstance(transformation.get("cut_plan"), dict):
            declared_cut_sha = transformation["cut_plan"].get("sha256")
        require(declared_cut_sha == sha256(cut_path), "Cut-plan provenance hash mismatch")
        require(plan.get("kind") == "press_prefix_cut_plan" and plan.get("schema_version") == 1 and plan.get("success") is True, "Invalid cut plan")
        require(plan["source_collection_metadata_sha256"] == evidence["sha256"] and
                plan["source_collection_fingerprint"] == collection["collection_fingerprint"], "Cut plan uses another collection")
        report.update(collection_metadata=evidence, timing=rate, cut_plan_sha256=sha256(cut_path))
        ds = LeRobotDataset("local/piper_elevator_press", root=root, video_backend="pyav")
        require(ds.fps == 30 and ds.meta.info.codebase_version == "v3.0" and ds.meta.robot_type == "piper", "Reader format/rate/type")
        require(len(ds.meta.tasks) == 12, "Task table size")
        for i, task in enumerate(TASKS):
            require(ds.meta.get_task_index(task) == i, f"Wrong task mapping {task}")
        for key, spec in feature_spec().items():
            actual = ds.features[key]
            require(actual.get("dtype") == spec["dtype"] and tuple(actual.get("shape", ())) == spec["shape"] and actual.get("names") == spec["names"], f"Feature differs {key}")
        entries = manifest["episodes"]
        require(len(entries) == ds.num_episodes == expected_episodes, "Wrong explicit episode count")
        require(len({e["episode_id"] for e in entries}) == len(entries), "Duplicate source episode IDs")
        table = ds.hf_dataset.with_format(None)
        require(len(table) == len(ds), "Arrow/reader length mismatch")
        counts, video_segments, numeric, contexts = Counter(), {}, {}, {}
        fk = IndependentFK(URDF)
        start_expected = sampled = 0
        for ep_index, entry in enumerate(entries):
            try:
                directory = raw / f"episode_{entry['episode_id']:06d}"
                metadata = read_json(directory / "metadata.json")
                audit_episode_collection(entry, metadata, collection, directory)
                require(metadata["success"] is True, "Uncommitted source")
                floor = int(metadata["floor"])
                require(floor in FLOORS and entry["floor"] == floor and entry["seed"] == metadata["seed"] and entry["episode_id"] == metadata["episode_id"], "Source identity mismatch")
                require(entry["task"] == metadata["task"] == TASKS[floor - 24], "Source task mismatch")
                require(set(entry["source_sha256"]) == set(SOURCE_FILES), "Incomplete source-file provenance")
                for name, digest in entry["source_sha256"].items():
                    require(sha256(directory / name) == digest, f"Source changed: {name}")
                with np.load(directory / "frames.npz", allow_pickle=False) as loaded:
                    frames = dict(loaded)
                with np.load(directory / "physics.npz", allow_pickle=False) as loaded:
                    physics = dict(loaded)
                expected, contact = source_prefix(frames, physics, metadata, collection)
                require(entry["press_prefix"] == plan_entries[entry["episode_id"]], "Manifest/cut-plan entry mismatch")
                cut = entry["press_prefix"]
                for key in PANEL_EPISODE_CONTEXT:
                    require((key in cut) == (key in metadata) and (key not in metadata or cut[key] == metadata[key]),
                            f"Cut plan dropped or changed episode panel context: {key}")
                require(cut["source_episode_id"] == metadata["episode_id"] and cut["floor"] == floor and cut["task"] == metadata["task"], "Cut-plan source identity")
                require(cut["terminal_source_action_index"] == cut["terminal_source_q_target_index"] == contact["cut_frame_index"] - 1 and cut["terminal_measured_state_used"] is False, "Cut-plan terminal target source")
                require(np.array_equal(cut["terminal_action"], expected["action"][-1]) and
                        np.array_equal(cut["terminal_q_target"], expected["q_target"][-1]), "Cut-plan terminal target values")
                for name in SOURCE_FILES:
                    require(cut["source_files"][name] == {"sha256": entry["source_sha256"][name], "bytes": (directory / name).stat().st_size}, f"Cut-plan source provenance {name}")
                for key in ("cut_frame_index", "kept_frames", "kept_physics_steps", "first_press_physics_index", "end_time_s", "removed_frames"):
                    require(entry["press_prefix"][key] == contact[key], f"Cut plan differs from independently recomputed {key}")
                n = len(expected["state"])
                em = ds.meta.episodes[ep_index]
                start, stop = int(em["dataset_from_index"]), int(em["dataset_to_index"])
                require(start == start_expected and stop - start == n == entry["frames"] == int(em["length"]), "Too many/few retained frames or noncontiguous range")
                require(int(em["episode_index"]) == ep_index and em["tasks"] == [TASKS[floor - 24]], "Episode metadata identity")
                values = {k: _numpy(v) for k, v in table[start:stop].items()}
                verify_numeric_prefix(values, expected, start, ep_index, metadata)
                for key, value in values.items():
                    if key not in CAMERAS:
                        numeric.setdefault(key, []).append(value)
                offset_x, offset_y = validate_episode_panel_metadata(collection, metadata)
                boxes = target_boxes(collection, expected, physics, floor, fk,
                                     panel_offset_y_m=offset_y, panel_offset_x_m=offset_x)
                selection = sorted({0, (n - 1) // 2, n - 1})
                comparisons = []
                source_images = {}
                for key, filename in CAMERAS.items():
                    generator = video_images(directory / filename)
                    try:
                        source_images[key] = {i: rgb for i, rgb in itertools.islice(enumerate(generator), n) if i in selection}
                    finally:
                        generator.close()
                    require(set(source_images[key]) == set(selection), "Source video shorter than prefix")
                for i in selection:
                    row = ds[start + i]
                    require(row["task"] == TASKS[floor - 24], "Official-reader task mismatch")
                    for key in CAMERAS:
                        rgb = _numpy(row[key])
                        require(rgb.shape == (3, 480, 640) and rgb.min() >= 0 and rgb.max() <= 1, "Official reader RGB range/shape")
                        comparisons.append({"frame_index": i, "camera": key, **compare_rgb(rgb.transpose(1, 2, 0) * 255, source_images[key][i])})
                    sampled += 1
                report["episodes"].append({"episode_index": ep_index, "source_episode_id": entry["episode_id"], "floor": floor,
                                           "frames": n, "contact": contact, "official_reader_sample_indices": selection,
                                           "rgb_source_comparison": comparisons, "views": {}})
                contexts[ep_index] = {"boxes": boxes, "counts": {v: [] for v in ("wrist", "global")}, "report": report["episodes"][-1]}
                for key in CAMERAS:
                    path = root / ds.meta.get_video_file_path(ep_index, key)
                    a, b = float(em[f"videos/{key}/from_timestamp"]), float(em[f"videos/{key}/to_timestamp"])
                    require(abs(a * 30 - round(a * 30)) < .005 and abs(b - a - n / 30) < 2e-4, "Video interval does not equal prefix length")
                    video_segments.setdefault(path, []).append((round(a * 30), round(a * 30) + n, ep_index, key))
                counts[floor] += 1
                start_expected = stop
            except Exception as exc:
                report["errors"].append(f"Episode {ep_index}: {type(exc).__name__}: {exc}")
        require(start_expected == len(ds), "Last cropped endpoint differs from dataset length")
        report.update(total_episodes=ds.num_episodes, total_frames=len(ds), official_reader_sampled_frames=sampled,
                      official_reader_sampled_images=sampled * 2, floor_counts={str(f): counts[f] for f in FLOORS})
        if not allow_partial:
            require(episodes_per_task is not None and all(counts[f] == episodes_per_task for f in FLOORS), "Wrong exact task counts")
        elif episodes_per_task is not None:
            require(all(counts[f] <= episodes_per_task for f in FLOORS), "Partial task count exceeds limit")
        require(set((root / "videos").rglob("*.mp4")) == set(video_segments), "Missing/unreferenced output video files")
        moments = {key: {"sum": np.zeros(3), "sum2": np.zeros(3), "count": 0} for key in CAMERAS}
        for path, segments in sorted(video_segments.items()):
            try:
                segments.sort()
                require(segments[0][0] == 0 and all(a[1] == b[0] for a, b in zip(segments, segments[1:])), "Video interval gaps/overlap")
                segment_index, decoded = 0, 0
                for i, rgb in enumerate(video_images(path)):
                    require(i < segments[-1][1], "Output video retains extra frames")
                    while i >= segments[segment_index][1]:
                        segment_index += 1
                    begin, end, ep_index, key = segments[segment_index]
                    view = key.rsplit(".", 1)[1]
                    ctx = contexts[ep_index]
                    ctx["counts"][view].append(amber_count(rgb, ctx["boxes"][view][i - begin]))
                    pixels = rgb[::4, ::4].reshape(-1, 3).astype(np.float64)
                    moments[key]["sum"] += pixels.sum(0)
                    moments[key]["sum2"] += np.square(pixels).sum(0)
                    moments[key]["count"] += len(pixels)
                    decoded += 1
                require(decoded == segments[-1][1], "Output video loses retained success/frame(s)")
                report["video_files"].append({"path": str(path.relative_to(root)), "sha256": sha256(path), "decoded_frames": decoded, "fps": 30, "all_pts_valid": True})
            except Exception as exc:
                report["errors"].append(f"Video {path}: {type(exc).__name__}: {exc}")
        for ep_index, ctx in contexts.items():
            for view, counts_ in ctx["counts"].items():
                try:
                    require(len(counts_) == ctx["report"]["frames"], "Missing decoded episode frames")
                    ctx["report"]["views"][view] = verify_light_prefix(counts_, view)
                    ctx["report"]["views"][view]["frames_with_target_outside_view"] = sum(box is None for box in ctx["boxes"][view])
                except Exception as exc:
                    report["errors"].append(f"Episode {ep_index} {view}: {type(exc).__name__}: {exc}")
        report["statistics"] = audit_stats(read_json(root / "meta/stats.json"), numeric, len(ds), moments)
        report["decoded_rgb_frames"] = sum(e["decoded_frames"] for e in report["video_files"])
        require(report["decoded_rgb_frames"] == len(ds) * 2, "Incomplete RGB decode count")
        report["manifest_sha256"] = sha256(root / MANIFEST)
    except Exception as exc:
        report["errors"].append(f"Dataset: {type(exc).__name__}: {exc}")
    report["success"] = not report["errors"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--episodes-per-task", type=int)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.expected_episodes < 1 or (not args.allow_partial and not args.episodes_per_task):
        parser.error("Positive --expected-episodes and exact --episodes-per-task (or --allow-partial) are required")
    report = audit_press_dataset(args.dataset, args.raw, expected_episodes=args.expected_episodes,
                                 episodes_per_task=args.episodes_per_task, allow_partial=args.allow_partial)
    path = args.report or args.dataset / "meta/press_audit.json"
    write_json(path, report)
    print(f"{'PASS' if report['success'] else 'FAIL'}: {report.get('total_episodes', 0)} episodes; {report.get('total_frames', 0)} frames; report={path}", flush=True)
    for error in report["errors"][:30]:
        print(error, flush=True)
    raise SystemExit(0 if report["success"] else 1)


if __name__ == "__main__":
    main()
