#!/usr/bin/env python3
"""Audit nine tabletop-button raw groups and optional published LeRobot groups.

Physical press evidence is recomputed from every 120 Hz sample. This does not
use the collector's array validator or rely on its reported success flag.
Use the PressB environment for raw-only audits, or the separate LeRobot
environment when supplying --lerobot-root (requires pyarrow). No Isaac
application is needed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pressb.tasks.table_buttons.spacing import resolve_spacing
from pressb.tasks.table_buttons.task import COLORS, TableButtonOutcome, button_in_camera, task_instruction

PROFILES = ("near", "medium", "far")
GROUPS = tuple(f"{profile}_{color}" for profile in PROFILES for color in COLORS)
UNITS = ["rad"] * 6 + ["m"]


@lru_cache(maxsize=4)
def arm_limits(urdf):
    root = ET.parse(urdf).getroot()
    limits = {joint.attrib["name"]: joint.find("limit") for joint in root.findall("joint")}
    return np.array([[float(limits[f"joint{index}"].attrib[key]) for key in ("lower", "upper")]
                     for index in range(1, 7)])


def read_json(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def close(actual, expected, message, tolerance=1e-10):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.shape == expected.shape and np.allclose(actual, expected, rtol=0, atol=tolerance), message)


def audit_arrays(frames, physics):
    """Verify independent state/action indexing, including the final observation."""
    count = len(frames["sim_time"])
    require(count >= 2, "Episode needs an initial and a terminal observation")
    close(frames["sim_time"], np.arange(count) / 30, "Frame time must be t=0+k/30")
    physical_count = (count - 1) * 4 + 1
    close(physics["time_s"], np.arange(physical_count) / 120, "Physics endpoints/rate differ from RGB")
    for key in ("state", "action"):
        value = frames[key]
        require(value.shape == (count, 7) and np.isfinite(value).all(), f"Malformed frame {key}")
    for key, width in (("q_actual", 8), ("q_target", 8), ("travel_m", 3), ("finger_force_n", 3)):
        value = physics[key]
        require(value.shape == (physical_count, width) and np.isfinite(value).all(), f"Malformed physics {key}")
    require(np.min(physics["finger_force_n"]) >= 0, "Negative contact force")
    # Isaac stores individual measured joints as float32; the observation
    # contract promotes them before subtracting the two finger coordinates.
    # Subtracting in float32 first introduces an additional rounding error.
    joints = np.asarray(physics["q_actual"][::4], dtype=np.float64)
    expected_state = np.column_stack((joints[:, :6], joints[:, 6] - joints[:, 7]))
    close(frames["state"], expected_state, "State/RGB boundary is shifted or gripper is not full opening")
    targets = np.column_stack((frames["action"][:-1, :6], frames["action"][:-1, 6] / 2,
                               -frames["action"][:-1, 6] / 2))
    close(physics["q_target"][1:], np.repeat(targets, 4, axis=0),
          "Action is not the actual target applied during the following four physics ticks")
    require(np.array_equal(frames["action"][-1], frames["action"][-2]),
            "Terminal action must repeat the previous interval target")
    return count, physical_count


def audit_video(path, expected_frames):
    """Decode every frame and verify consecutive original-speed timestamps."""
    import av

    count = 0
    with av.open(str(path)) as container:
        require(len(container.streams.video) == 1 and not container.streams.audio,
                f"Expected one silent video stream: {path.name}")
        stream = container.streams.video[0]
        stream.thread_count = 1
        require((stream.width, stream.height) == (640, 480), f"Wrong image size: {path.name}")
        require(stream.average_rate is not None and abs(float(stream.average_rate) - 30) < 1e-8,
                f"Wrong frame rate: {path.name}")
        for index, frame in enumerate(container.decode(stream)):
            require(frame.pts is not None and abs(float(frame.pts * frame.time_base) - index / 30) < 5e-6,
                    f"Video PTS is shifted or has gaps: {path.name}, frame {index}")
            require((frame.width, frame.height) == (640, 480), f"Changing image size: {path.name}")
            if index == 0:
                require(frame.to_ndarray(format="rgb24").std() > 1, f"Blank initial image: {path.name}")
            count += 1
    require(count == expected_frames, f"Video count mismatch: {path.name}: {count} vs {expected_frames}")
    return {"frames": count, "fps": 30, "full_decode": True, "first_time_s": 0,
            "last_time_s": (count - 1) / 30}


def audit_physics(physics, config, target, contacts, unexpected):
    """Recompute the press, including wrong events after a latched success."""
    times, travel, force = (physics[key] for key in ("time_s", "travel_m", "finger_force_n"))
    require(not unexpected, "Unexpected robot/environment contact was recorded")
    settings = config["success"]
    checker = TableButtonOutcome(target, **settings)
    events = []
    for index, seconds in enumerate(times):
        result = checker.update(float(seconds), dict(zip(COLORS, travel[index])), dict(zip(COLORS, force[index])))
        events.extend({"time_s": float(seconds), "color": color} for color in result["press_events"])
    require(result["success"] and result["wrong_button"] is None, "Recorded physics does not yield target success")
    require(events and all(event["color"] == target for event in events),
            "Non-target press event occurred, including after target success")
    other_indices = [index for index, color in enumerate(COLORS) if color != target]
    require(np.max(force[:, other_indices]) <= .02, "Non-target finger contact exceeds 0.02 N")
    require(np.max(travel[:, other_indices]) < settings["press_depth_m"], "Non-target cap reached press depth")
    require(np.max(np.abs(travel[0])) < settings["release_depth_m"], "Initial caps were not released")
    require(np.max(force[0]) <= .02, "Initial frame already has button contact")
    reconstructed = np.zeros_like(force)
    for event in contacts:
        tick, color = event.get("tick"), event.get("color")
        require(type(tick) is int and 0 <= tick < len(times) and color in COLORS, "Invalid contact event index/color")
        require(event.get("fingers") and set(event["fingers"]) <= {"link7", "link8"}, "Invalid contact finger")
        magnitude = float(event["force_n"])
        require(np.isfinite(magnitude) and magnitude >= 0, "Invalid event force")
        reconstructed[tick, COLORS.index(color)] += magnitude
    close(reconstructed, force, "Contact events and per-tick contact forces disagree", tolerance=1e-8)
    return {"first_success_time_s": result["first_success_time_s"], "press_events": events,
            "maximum_travel_m": dict(zip(COLORS, np.max(travel, axis=0).tolist())),
            "maximum_finger_force_n": dict(zip(COLORS, np.max(force, axis=0).tolist())),
            "max_arm_tracking_error_rad": float(np.max(np.abs(physics["q_actual"][:, :6] - physics["q_target"][:, :6])))}


def audit_episode(directory, group, *, decode_videos=True):
    directory = Path(directory)
    profile, color = group.split("_")
    metadata = read_json(directory / "metadata.json")
    config = read_json(directory / "config.json")
    expected = dict(schema_version="pressb.table_buttons.raw.v1", group=group, spacing_profile=profile,
                    target_color=color, instruction=task_instruction(color), fps=30,
                    camera_resolution=[640, 480], state_units=UNITS, action_units=UNITS,
                    gripper_representation="full_width", success=True,
                    terminal_action="repeat_previous_interval_target")
    for key, value in expected.items():
        require(metadata.get(key) == value, f"Incorrect metadata {key}")
    require(type(metadata["success"]) is bool, "Success flag must be boolean")
    require(type(metadata.get("seed")) is int, "Layout seed must be an integer")
    require(type(metadata.get("frame_count")) is int, "Frame count must be an integer")
    require(metadata["episode_index"] == int(directory.name.split("_")[1]), "Episode ID differs from directory")
    require(abs(metadata["physics_dt"] - 1 / 120) < 1e-12 and config["fps"] == 30
            and abs(config["physics_dt"] - 1 / 120) < 1e-12, "Unexpected physical/control frequency")
    require(config["task_color"] == color and config["task_instruction"] == task_instruction(color),
            "Configuration has a different color or instruction")
    require(config["position_sampling"]["spacing_profile"] == profile, "Configuration spacing differs")
    require(set(metadata["positions_xy_m"]) == set(COLORS), "Incomplete button positions")
    positions = {key: np.asarray(metadata["positions_xy_m"][key], dtype=float) for key in COLORS}
    spacing = resolve_spacing(config["position_sampling"], config["button_geometry"]["base_radius_m"])
    distances = {}
    geometry = config["button_geometry"]
    for index, key in enumerate(COLORS):
        require(positions[key].shape == (2,) and np.isfinite(positions[key]).all(), "Malformed button XY")
        close(positions[key], config["buttons"][key]["center_xy"], "Metadata and scene button positions differ")
        require(button_in_camera(positions[key], config["up_camera"], button_radius=geometry["base_radius_m"],
                                 button_top_z=geometry["base_height_m"] + geometry["cap_height_m"],
                                 margin_px=config["position_sampling"].get("pixel_margin", 16)),
                f"Button {key} falls outside the fixed camera")
        for other in COLORS[index + 1:]:
            distance = float(np.linalg.norm(positions[key] - positions[other]))
            require(spacing.accepts(distance), f"Distance outside {profile} band: {key}/{other}: {distance}")
            distances[f"{key}_{other}"] = distance
    with np.load(directory / "data.npz", allow_pickle=False) as archive:
        frames = {key: np.asarray(archive[key]) for key in ("sim_time", "state", "action")}
    with np.load(directory / "physics.npz", allow_pickle=False) as archive:
        physics = {key: np.asarray(archive[key]) for key in ("time_s", "q_actual", "q_target", "travel_m", "finger_force_n")}
    count, physical_count = audit_arrays(frames, physics)
    initial = np.r_[config["initial_joint_positions_rad"], config["initial_gripper_width_m"] / 2,
                    -config["initial_gripper_width_m"] / 2]
    close(physics["q_target"][0], initial, "Initial drive target differs from the configured reset pose")
    limits = arm_limits(str((ROOT / config["robot_urdf"]).resolve()))
    require(np.all(physics["q_target"][:, :6] >= limits[:, 0] - 1e-8)
            and np.all(physics["q_target"][:, :6] <= limits[:, 1] + 1e-8),
            "Applied arm targets exceed radian joint limits")
    require(np.all(frames["action"][:, 6] >= 0)
            and np.all(frames["action"][:, 6] <= config["gripper_max_width_m"]),
            "Applied gripper target is not a valid full width in metres")
    require(count == metadata["frame_count"], "Frame count differs from metadata")
    require(abs(frames["sim_time"][-1] - metadata["duration_s"]) < 1e-10, "Duration differs from final observation")
    physical = audit_physics(physics, config, color, read_json(directory / "contacts.json"),
                             read_json(directory / "unexpected_contacts.json"))
    success_time = physical["first_success_time_s"]
    require(abs(success_time - metadata["first_success_time_s"]) < 1e-10, "Reported success time differs from physics")
    boundary_tick = int(np.ceil((success_time * 120 - 1e-8) / 4)) * 4
    require(physical_count - 1 == boundary_tick, "Episode does not end at the first sampled success boundary")
    for key, observed in (("initial_state_measured_rad_m", frames["state"][0]),
                          ("initial_state_target_rad_m", np.r_[initial[:6], initial[6] - initial[7]])):
        if key in metadata:
            close(metadata[key], observed, f"Incorrect {key}")
    if "button_color_order" in metadata:
        require(metadata["button_color_order"] == list(COLORS), "Incorrect physics color order")
    videos = {view: audit_video(directory / f"{view}.mp4", count) for view in ("up", "wrist")} if decode_videos else {}
    return dict(episode_index=metadata["episode_index"], seed=metadata["seed"], group=group,
                instruction=metadata["instruction"], frames=count, physical_samples=physical_count,
                duration_s=metadata["duration_s"], positions_xy_m=metadata["positions_xy_m"],
                pairwise_distances_m=distances, physics=physical, videos=videos, success=True)


def audit_exported_group(root, group, episodes):
    """Check published group identity against raw; numeric/video audits belong to the official exporter."""
    import pyarrow.parquet as pq

    root = Path(root) / group
    manifest, info, audit = (read_json(root / "meta" / name) for name in ("export_manifest.json", "info.json", "audit.json"))
    instruction = task_instruction(group.split("_")[1])
    count, frames = len(episodes), sum(item["frames"] for item in episodes)
    require(manifest["group"] == group and manifest["instruction"] == instruction, "Export group/language mismatch")
    require(info["codebase_version"] == "v3.0" and info["fps"] == 30 and info["total_tasks"] == 1
            and info["total_episodes"] == count and info["total_frames"] == frames, "Export counts/schema mismatch")
    names = [f"joint_{i}.pos" for i in range(1, 7)] + ["gripper.pos"]
    for key in ("observation.state", "action"):
        feature = info["features"][key]
        require(feature["dtype"] == "float32" and feature["shape"] == [7] and feature["names"] == names,
                f"Export numerical feature mismatch: {key}")
    for view in ("up", "wrist"):
        feature = info["features"][f"observation.images.{view}"]
        require(feature["dtype"] == "video" and feature["shape"] == [480, 640, 3], "Export camera mismatch")
    require(manifest["semantics"]["units"] == ["degree"] * 6 + ["cm"], "Export action/state units differ from PAP")
    require(len(manifest["episodes"]) == count, "Export manifest episode count mismatch")
    for exported, raw in zip(manifest["episodes"], episodes):
        for key in ("episode_index", "seed", "group", "instruction", "positions_xy_m"):
            require(exported[key] == raw[key], f"Export/raw identity mismatch: {key}")
        require(exported["frame_count"] == raw["frames"], "Export/raw frame count mismatch")
    table = pq.read_table(root / "meta/tasks.parquet").to_pandas()
    require(table.index.tolist() == [instruction] and table["task_index"].tolist() == [0], "Export task vocabulary mismatch")
    require(audit.get("success") is True and audit.get("episodes") == count and audit.get("frames") == frames
            and audit.get("full_video_decode") is True and audit.get("numeric_rows_compared") == frames,
            "Official-reader export audit missing or incomplete")
    require(audit.get("decoded_frames") == {f"observation.images.{view}": frames for view in ("up", "wrist")},
            "Export decode counts mismatch")
    return dict(success=True, episodes=count, frames=frames, instruction=instruction,
                official_reader_audit=str(root / "meta/audit.json"))


def audit_collection(raw_root, expected_per_group=100, *, groups=GROUPS, decode_videos=True, lerobot_root=None):
    raw_root = Path(raw_root).resolve()
    report = dict(success=False, raw_root=str(raw_root), expected_per_group=expected_per_group,
                  checked_utc=datetime.now(timezone.utc).isoformat(), full_video_decode=decode_videos,
                  groups={}, errors=[], total_episodes=0, total_frames=0,
                  action_semantics="observed t=k/30; action is the following four physics ticks; terminal repeats previous",
                  physical_validation="target displacement and finger force for required dwell; no wrong press/contact")
    for group in groups:
        rows, errors = [], []
        directories = sorted(path for path in (raw_root / group).glob("episode_*") if path.is_dir())
        expected_names = [f"episode_{index:06d}" for index in range(expected_per_group)]
        if [path.name for path in directories] != expected_names:
            errors.append(f"Expected {expected_per_group} consecutive accepted episodes, found {len(directories)}")
        for directory in directories:
            try:
                require(re.fullmatch(r"episode_[0-9]{6,}", directory.name), "Malformed accepted episode directory")
                rows.append(audit_episode(directory, group, decode_videos=decode_videos))
            except Exception as error:
                errors.append(f"{directory.name}: {type(error).__name__}: {error}")
        seeds = [row["seed"] for row in rows]
        if len(set(seeds)) != len(seeds):
            errors.append("Repeated layout seed within group")
        layouts = [json.dumps(row["positions_xy_m"], sort_keys=True) for row in rows]
        if len(set(layouts)) != len(layouts):
            errors.append("Repeated exact button layout within group")
        exported = None
        if lerobot_root is not None and not errors:
            try:
                exported = audit_exported_group(lerobot_root, group, rows)
            except Exception as error:
                errors.append(f"LeRobot: {type(error).__name__}: {error}")
        report["groups"][group] = dict(success=not errors, episodes=rows, errors=errors, exported=exported)
        report["errors"].extend(f"{group}: {error}" for error in errors)
        report["total_episodes"] += len(rows)
        report["total_frames"] += sum(row["frames"] for row in rows)
    report["success"] = not report["errors"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--expected-per-group", type=int, default=100)
    parser.add_argument("--groups", nargs="+", choices=GROUPS, default=GROUPS)
    parser.add_argument("--lerobot-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.expected_per_group < 1:
        parser.error("--expected-per-group must be positive")
    report = audit_collection(args.raw_root, args.expected_per_group, groups=args.groups, lerobot_root=args.lerobot_root)
    output = args.output or args.raw_root / "audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("success", "total_episodes", "total_frames", "errors")}, ensure_ascii=False))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
