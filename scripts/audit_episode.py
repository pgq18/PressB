#!/usr/bin/env python3
"""Recompute physical episode checks from saved measurements and the official URDF.

This does not start Isaac Sim or modify any trajectory data. It writes a separate
``audit_summary.json``, including one row of measured statistics per floor.
Reports with ``config.button_light_mode="momentary"`` additionally require
verified releases, measured returns home, and complete wrist RGB-D captures.
Older reports without this field retain the original latched-light audit.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pressb.kinematics import PiperKinematics


def audit_gripper(data: dict, target_positions, urdf_path: Path, count: int) -> dict:
    """Check recorded joint7/joint8 positions in metres against the fixed grip."""
    checks = {name: False for name in (
        "gripper_data_shape_and_finite", "gripper_targets_match_config",
        "gripper_tracking_within_0_25_mm", "gripper_joints_within_official_limits",
    )}
    result = {"joint_names": ["joint7", "joint8"], "checks": checks,
              "tracking_tolerance_m": .00025, "joint_limit_tolerance_m": .0001,
              "target_positions_m": None, "max_position_error_m": None,
              "actual_min_m": None, "actual_max_m": None}
    try:
        target = np.asarray(target_positions, dtype=float)
        actual = np.asarray(data.get("gripper_q_actual", []), dtype=float)
        commanded = np.asarray(data.get("gripper_q_target", []), dtype=float)
        checks["gripper_data_shape_and_finite"] = bool(
            target.shape == (2,) and actual.shape == commanded.shape == (count, 2)
            and count > 0 and np.isfinite(target).all()
            and np.isfinite(actual).all() and np.isfinite(commanded).all()
        )
        if not checks["gripper_data_shape_and_finite"]:
            return result
        joints = {joint.get("name"): joint for joint in ET.parse(urdf_path).getroot().findall("joint")}
        limits = []
        for name in result["joint_names"]:
            joint = joints.get(name)
            if joint is None or joint.get("type") != "prismatic" or joint.find("limit") is None:
                raise ValueError(f"Missing official prismatic limits for {name}")
            limit = joint.find("limit")
            limits.append([float(limit.get("lower")), float(limit.get("upper"))])
        lower, upper = np.asarray(limits).T
        if not np.isfinite(limits).all() or np.any(lower > upper):
            raise ValueError("Invalid official gripper limits")
        maximum_error = float(np.max(np.abs(actual - target)))
        checks["gripper_targets_match_config"] = bool(np.allclose(commanded, target, rtol=0, atol=1e-9))
        checks["gripper_tracking_within_0_25_mm"] = maximum_error <= .00025
        checks["gripper_joints_within_official_limits"] = bool(
            np.all(target >= lower) and np.all(target <= upper)
            and np.all(actual >= lower - .0001) and np.all(actual <= upper + .0001)
        )
        result.update(target_positions_m=target.tolist(), max_position_error_m=maximum_error,
                      actual_min_m=np.min(actual, axis=0).tolist(), actual_max_m=np.max(actual, axis=0).tolist(),
                      urdf_lower_m=lower.tolist(), urdf_upper_m=upper.tolist())
    except (OSError, ET.ParseError, TypeError, ValueError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def home_geometry(kin: PiperKinematics, q: np.ndarray, cfg: dict) -> dict:
    """Describe folding and casing tilt using official link origins and mesh axes."""
    frames = kin.link_transforms(q)
    housing_tilts = kin.housing_tilts_deg(q)
    base = np.array([cfg.get("robot_base_x", 0.), cfg.get("robot_base_y", 0.), cfg["table_height"]])
    shoulder, elbow, wrist = [frames[name][:3, 3] + base for name in ("link2", "link3", "link5")]
    upper_interval = sorted((float(shoulder[0]), float(elbow[0])))
    lower_interval = sorted((float(elbow[0]), float(wrist[0])))
    overlap = max(0., min(upper_interval[1], lower_interval[1]) - max(upper_interval[0], lower_interval[0]))
    behind = bool(elbow[0] < shoulder[0])
    forward = bool(wrist[0] > elbow[0])
    above = bool(min(shoulder[2], elbow[2], wrist[2]) > cfg["table_height"])
    return {
        "geometry_basis": "Official URDF link2/link3/link5 joint origins",
        "world_positions_m": {name: point.tolist() for name, point in
                              (("shoulder", shoulder), ("elbow", elbow), ("wrist", wrist))},
        "upper_lower_x_projection_overlap_m": overlap,
        "elbow_behind_shoulder": behind,
        "wrist_forward_of_elbow": forward,
        "joint_origins_above_table": above,
        "folded_joint_geometry": bool(overlap > .20 and behind and forward and above),
        "housing_axis_basis": "Official STL long planar casing directions transformed by URDF link rotations",
        "housing_tilts_deg": {"upper": float(housing_tilts[0]), "forearm": float(housing_tilts[1])},
        "max_abs_housing_tilt_deg": float(np.max(np.abs(housing_tilts))),
    }


def audit_camera(output: Path, cfg: dict, data: dict, expected_steps: int) -> dict:
    """Inspect saved pixels/depth independently of the runtime camera report."""
    from PIL import Image
    from scipy.spatial.transform import Rotation

    root = (output / "wrist_camera").resolve()
    checks = {}
    errors = []
    result = {"checks": checks, "errors": errors, "frames": 0}
    try:
        report = json.loads((root / "camera_report.json").read_text())
        intrinsics = json.loads((root / "intrinsics.json").read_text())
        records = [json.loads(line) for line in (root / "timestamps.jsonl").read_text().splitlines() if line.strip()]
        width, height = int(intrinsics["width"]), int(intrinsics["height"])
        calibration = np.asarray([intrinsics[k] for k in ("fx", "fy", "cx", "cy")], dtype=float)
        checks["calibration"] = bool(
            width > 0 and height > 0 and np.isfinite(calibration).all()
            and calibration[0] > 0 and calibration[1] > 0
            and 0 <= calibration[2] < width and 0 <= calibration[3] < height
            and intrinsics["depth_unit"] == "m"
        )
        stride = int(report.get("capture_stride", cfg.get("wrist_camera_capture_stride", 0)))
        expected = (expected_steps - 1) // stride + 1 if stride > 0 else -1
        checks["complete_frame_count"] = bool(
            len(records) > 1 and len(records) == report["valid_frames"] == report["expected_frames"] == expected
        )
        result.update(frames=len(records), expected_frames=expected, capture_stride=stride)
        checks["sequential_frame_ids"] = [r["frame"] for r in records] == list(range(len(records)))
        times = np.asarray([r["time"] for r in records], dtype=float)
        capture_times = data["time"][::stride] if stride > 0 else np.array([])
        checks["timestamps_match_physics"] = bool(
            np.isfinite(times).all() and np.all(np.diff(times) > 0)
            and times.shape == capture_times.shape and np.allclose(times, capture_times, rtol=0, atol=1e-7)
        )
        poses = np.asarray([r["position_world"] for r in records], dtype=float)
        rotations = np.asarray([r["orientation_world"] for r in records], dtype=float)
        checks["valid_camera_poses"] = bool(
            poses.shape == (len(records), 3) and rotations.shape == (len(records), 4)
            and np.isfinite(poses).all() and np.isfinite(rotations).all()
            and np.allclose(np.linalg.norm(rotations, axis=1), 1., rtol=0, atol=1e-3)
        )
        checks["camera_moves_with_wrist"] = bool(
            len(poses) > 1 and poses.shape == (len(records), 3)
            and np.max(np.linalg.norm(poses - poses[0], axis=1)) > .02
        )
        sensor = report["sensor"]
        local_position = np.asarray(sensor["optical_position_link6"], dtype=float)
        local_quaternion = np.asarray(sensor["optical_quaternion_wxyz_link6"], dtype=float)
        checks["valid_optical_mount_transform"] = bool(
            local_position.shape == (3,) and local_quaternion.shape == (4,)
            and np.isfinite(local_position).all() and np.isfinite(local_quaternion).all()
            and abs(np.linalg.norm(local_quaternion) - 1.) < 1e-3
        )
        checks["optical_position_matches_measured_wrist"] = False
        checks["optical_orientation_matches_measured_wrist"] = False
        if (len(records) and checks["timestamps_match_physics"] and checks["valid_camera_poses"]
                and checks["valid_optical_mount_transform"]):
            kin = PiperKinematics(ROOT / cfg["robot_urdf"], tip_offset=cfg["tip_offset"])
            tool_poses = np.asarray([kin.fk(q) for q in data["q_actual"][::stride]])
            wrist_rotations = tool_poses[:, :3, :3]
            # FK includes the stylus offset; undo it to recover link6's origin.
            wrist_origins = tool_poses[:, :3, 3] - wrist_rotations @ np.array([0., 0., cfg["tip_offset"]])
            base = np.array([cfg.get("robot_base_x", 0.), cfg.get("robot_base_y", 0.), cfg["table_height"]])
            expected_positions = wrist_origins + wrist_rotations @ local_position + base
            local_rotation = Rotation.from_quat(local_quaternion[[1, 2, 3, 0]]).as_matrix()
            expected_rotations = wrist_rotations @ local_rotation
            measured_rotations = Rotation.from_quat(rotations[:, [1, 2, 3, 0]]).as_matrix()
            position_errors = np.linalg.norm(poses - expected_positions, axis=1)
            orientation_errors = Rotation.from_matrix(
                np.swapaxes(expected_rotations, 1, 2) @ measured_rotations
            ).magnitude()
            checks["optical_position_matches_measured_wrist"] = bool(np.max(position_errors) < .005)
            checks["optical_orientation_matches_measured_wrist"] = bool(np.max(orientation_errors) < .01)
            result.update(max_wrist_mount_position_error_m=float(np.max(position_errors)),
                          max_wrist_mount_orientation_error_rad=float(np.max(orientation_errors)))
        rgb_hashes, depth_hashes, paths = set(), set(), set()
        fractions = []
        minimum_depth, maximum_depth = float("inf"), 0.
        dimensions_ok, nonempty_rgb = True, True
        for row in records:
            rgb_path = (root / row["rgb"]).resolve()
            depth_path = (root / row["depth"]).resolve()
            if not rgb_path.is_relative_to(root / "rgb") or not depth_path.is_relative_to(root / "depth"):
                raise ValueError("Camera frame paths must remain inside wrist_camera/rgb and wrist_camera/depth")
            if rgb_path in paths or depth_path in paths:
                raise ValueError("Multiple timestamps reference the same camera file")
            paths.update((rgb_path, depth_path))
            with Image.open(rgb_path) as image:
                rgb = np.asarray(image.convert("RGB"))
            depth = np.load(depth_path, allow_pickle=False)
            dimensions_ok &= rgb.shape == (height, width, 3) and depth.shape == (height, width)
            if depth.shape != (height, width) or not np.issubdtype(depth.dtype, np.floating):
                raise ValueError(f"Invalid metric depth array: {depth_path.name}")
            nonempty_rgb &= bool(np.ptp(rgb) > 0)
            rgb_hashes.add(hashlib.sha256(rgb.tobytes()).hexdigest())
            depth_hashes.add(hashlib.sha256(depth.tobytes()).hexdigest())
            valid = np.isfinite(depth) & (depth > 0)
            fractions.append(float(np.mean(valid)))
            if valid.any():
                minimum_depth = min(minimum_depth, float(np.min(depth[valid])))
                maximum_depth = max(maximum_depth, float(np.max(depth[valid])))
        fractions = np.asarray(fractions)
        mean_fraction = float(np.mean(fractions)) if len(fractions) else 0.
        aggregation = report.get("finite_depth_fraction_aggregation", "mean")
        reported_fraction = float(report["finite_depth_fraction"])
        measured_report_fraction = float(np.min(fractions)) if aggregation == "min" and len(fractions) else mean_fraction
        checks.update({
            "image_and_depth_dimensions": bool(dimensions_ok and len(records) > 0),
            "nonempty_rgb": bool(nonempty_rgb and len(records) > 0),
            "rgb_changes_over_time": len(rgb_hashes) > 1,
            "depth_changes_over_time": len(depth_hashes) > 1,
            "positive_metric_depth": bool(len(fractions) > 0 and np.min(fractions) >= .05 and mean_fraction >= .1),
            "reported_depth_fraction_matches": bool(
                aggregation in ("mean", "min") and np.isfinite(reported_fraction)
                and abs(reported_fraction - measured_report_fraction) < 1e-5
            ),
            "no_unreferenced_rgb_frames": set((root / "rgb").glob("*.png")) == {p for p in paths if p.suffix == ".png"},
            "no_unreferenced_depth_frames": set((root / "depth").glob("*.npy")) == {p for p in paths if p.suffix == ".npy"},
        })
        result.update(unique_rgb_frames=len(rgb_hashes), unique_depth_frames=len(depth_hashes),
                      finite_positive_depth_fraction=mean_fraction,
                      minimum_frame_depth_fraction=float(np.min(fractions)) if len(fractions) else 0.,
                      minimum_positive_depth_m=minimum_depth if np.isfinite(minimum_depth) else None,
                      maximum_positive_depth_m=maximum_depth)
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        checks["camera_files_readable_and_well_formed"] = False
    result["success"] = bool(checks and all(checks.values()) and not errors)
    return result


def audit(output: Path) -> dict:
    report = json.loads((output / "report.json").read_text())
    status = json.loads((output / "run_status.json").read_text())
    events = json.loads((output / "events.json").read_text())
    cfg = report["config"]
    momentary = cfg.get("button_light_mode", "latched") == "momentary"
    release_threshold = float(cfg.get("release_threshold", .0008))
    archive_path = output / "trajectory.npz"
    with np.load(archive_path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    count = len(data["time"])
    if count == 0:
        raise ValueError("The trajectory contains no measurements")
    shapes = {
        "time": (count,), "floor": (count,), "phase": (count,),
        "q_actual": (count, 6), "q_target": (count, 6), "qd_actual": (count, 6),
        "tip_actual": (count, 3), "tip_target": (count, 3),
        "button_travel": (count, 12), "contact_force": (count, 12), "lights": (count, 12),
    }
    for name, shape in shapes.items():
        if name not in data or data[name].shape != shape:
            raise ValueError(f"Unexpected data shape for {name}; expected {shape}")
        if name != "phase" and not np.isfinite(data[name]).all():
            raise ValueError(f"Nonfinite measurements in {name}")

    urdf_path = ROOT / cfg["robot_urdf"]
    kin = PiperKinematics(urdf_path, tip_offset=cfg["tip_offset"])
    poses = np.asarray([kin.fk(q) for q in data["q_actual"]])
    computed_tip = poses[:, :3, 3].copy()
    computed_tip += [cfg.get("robot_base_x", 0.), cfg.get("robot_base_y", 0.), cfg["table_height"]]
    fk_error = np.linalg.norm(computed_tip - data["tip_actual"], axis=1)
    tip_tracking_error = np.linalg.norm(data["tip_actual"] - data["tip_target"], axis=1)
    joint_tracking_error = np.abs(data["q_actual"] - data["q_target"])
    period_error = np.abs(np.diff(data["time"]) - cfg["physics_dt"])
    presses = [event for event in events if event["type"] == "button_pressed"]
    releases = [event for event in events if event["type"] == "button_released"]
    homes = [event for event in events if event["type"] == "cycle_home"]
    home_q = np.asarray(cfg.get("home_q", []), dtype=float)
    tolerance = float(cfg.get("home_tolerance_rad", .025))
    checks = {
        "run_finished": status.get("status") == "complete",
        "recognized_light_mode": cfg.get("button_light_mode", "latched") in ("momentary", "latched"),
        "runtime_physics_validation": report.get("success") is True and report.get("physics_validated") is True,
        "complete_sample_count": count == report["steps_expected"] == report["steps_completed"],
        "sample_period": bool(count > 1 and np.all(period_error < 1e-7)),
        "joint_names_match_urdf": report["joint_names"] == kin.joint_names,
        "measured_joints_within_limits": bool(
            np.all(data["q_actual"] >= kin.lower - .002)
            and np.all(data["q_actual"] <= kin.upper + .002)
        ),
        "joint_tracking": bool(np.max(joint_tracking_error) < .15),
        "recomputed_urdf_matches_measured_tip": bool(np.max(fk_error) < .005),
        "press_order": [event["floor"] for event in presses] == cfg["sequence"],
        "binary_lights": bool(np.isin(data["lights"], [0, 1]).all()),
        "nonnegative_contact_force": bool(np.all(data["contact_force"] >= 0)),
        "button_travel_within_physical_stops": bool(
            np.min(data["button_travel"]) >= -.0008
            and np.max(data["button_travel"]) <= cfg["button_travel"] + .0008
        ),
        "all_buttons_spring_back": bool(np.max(np.abs(data["button_travel"][-1])) <= release_threshold),
        "no_reported_unexpected_contacts": report.get("unexpected_collisions") == [],
    }
    gripper = None
    if "gripper_joint_positions_m" in cfg:
        gripper = audit_gripper(data, cfg["gripper_joint_positions_m"], urdf_path, count)
        checks.update(gripper["checks"])
    if momentary:
        cycle_starts = np.r_[True, np.diff(data["floor"]) != 0]
        checks.update({
            "valid_hysteresis_thresholds": 0 < release_threshold < cfg["press_threshold"],
            "release_order": [event["floor"] for event in releases] == cfg["sequence"],
            "home_order": [event["floor"] for event in homes] == cfg["sequence"],
            "all_final_lights_off": bool(np.all(data["lights"][-1] == 0)),
            "at_most_one_light_on": bool(np.all(np.sum(data["lights"], axis=1) <= 1)),
            "commanded_cycle_order": data["floor"][cycle_starts].tolist() == cfg["sequence"],
            "initial_and_final_measured_home": bool(
                home_q.shape == (6,) and np.isfinite(home_q).all() and 0 < tolerance <= .1
                and np.max(np.abs(data["q_actual"][[0, -1]] - home_q)) <= tolerance
            ),
        })
        # Recompute the complete contact/travel hysteresis state, including rows
        # without events; fabricated event lists cannot hide an early release.
        state = np.zeros(12, dtype=bool)
        expected_lights = np.empty_like(data["lights"])
        for row in range(count):
            previous = state.copy()
            state[previous & (data["button_travel"][row] <= release_threshold)] = False
            state[~previous & (data["button_travel"][row] >= cfg["press_threshold"])
                  & (data["contact_force"][row] > .02)] = True
            expected_lights[row] = state
        checks["lights_follow_measured_press_and_release"] = bool(np.array_equal(expected_lights, data["lights"]))
        commanded = data["floor"][:, None] == np.arange(24, 36)[None, :]
        checks["no_wrong_floor_lights_or_presses"] = bool(np.all((data["lights"] == 0) | commanded))
    else:
        checks["lights_never_unlatch"] = bool(np.all(np.diff(data["lights"], axis=0) >= 0))
        checks["all_final_lights_on"] = bool(np.all(data["lights"][-1] == 1))

    # Match each event to the same measured row, including the light transition.
    transitions = np.diff(np.vstack([np.zeros((1, 12)), data["lights"]]), axis=0)
    rows = []
    measured_home_geometries = []
    for floor in range(24, 36):
        index = floor - 24
        floor_events = [event for event in presses if event["floor"] == floor]
        matched = len(floor_events) == 1
        event_time = None
        press_row = None
        if matched:
            event = floor_events[0]
            event_time = float(event["time"])
            row = int(np.argmin(np.abs(data["time"] - event_time)))
            press_row = row
            matched = bool(
                abs(data["time"][row] - event_time) < 1e-7
                and data["floor"][row] == floor == event["commanded_floor"]
                and data["button_travel"][row, index] >= cfg["press_threshold"]
                and data["contact_force"][row, index] > .02
                and transitions[row, index] == 1
                and np.sum(transitions[:, index] == 1) == 1
            )
        checks[f"floor_{floor}_event_matches_measurements"] = matched
        floor_result = {
            "floor": floor,
            "event_verified": matched,
            "press_time_s": event_time,
            "peak_travel_mm": float(np.max(data["button_travel"][:, index]) * 1000),
            "peak_contact_force_n": float(np.max(data["contact_force"][:, index])),
            "contact_duration_above_threshold_s": float(
                np.count_nonzero(data["contact_force"][:, index] > .02) * cfg["physics_dt"]
            ),
            "final_travel_mm": float(data["button_travel"][-1, index] * 1000),
        }
        if momentary:
            floor_releases = [event for event in releases if event["floor"] == floor]
            release_ok, release_time, release_row = False, None, None
            if len(floor_releases) == 1 and press_row is not None:
                event = floor_releases[0]
                release_time = float(event["time"])
                release_row = int(np.argmin(np.abs(data["time"] - release_time)))
                release_ok = bool(
                    abs(data["time"][release_row] - release_time) < 1e-7 and release_row > press_row
                    and data["floor"][release_row] == floor == event["commanded_floor"]
                    and data["button_travel"][release_row, index] <= release_threshold
                    and transitions[release_row, index] == -1
                    and np.count_nonzero(transitions[:, index] == -1) == 1
                )
            checks[f"floor_{floor}_release_matches_measurements"] = release_ok
            floor_homes = [event for event in homes if event["floor"] == floor]
            home_ok, home_time, home_error = False, None, None
            if len(floor_homes) == 1 and release_row is not None and home_q.shape == (6,):
                event = floor_homes[0]
                home_time = float(event["time"])
                home_row = int(np.argmin(np.abs(data["time"] - home_time)))
                home_error = float(np.max(np.abs(data["q_actual"][home_row] - home_q)))
                measured_home_geometries.append({
                    "floor": floor, "time_s": home_time,
                    **home_geometry(kin, data["q_actual"][home_row], cfg),
                })
                cycle_rows = np.flatnonzero(data["floor"] == floor)
                hold_rows = cycle_rows[data["phase"][cycle_rows] == "home_hold"]
                home_ok = bool(
                    np.isfinite(home_q).all() and 0 < tolerance <= .1
                    and abs(data["time"][home_row] - home_time) < 1e-7 and home_row > release_row
                    and len(hold_rows) > 0 and home_row == hold_rows[-1] == cycle_rows[-1]
                    and np.all(np.diff(cycle_rows) == 1)
                    and home_error <= tolerance
                    and np.max(np.abs(data["q_target"][home_row] - home_q)) < 1e-6
                    and np.all(data["lights"][home_row] == 0)
                    and np.max(np.abs(data["button_travel"][home_row])) <= release_threshold
                )
            checks[f"floor_{floor}_returned_home_after_release"] = home_ok
            floor_result.update(release_verified=release_ok, release_time_s=release_time,
                                home_verified=home_ok, home_time_s=home_time, home_error_rad=home_error)
        rows.append(floor_result)
    camera = audit_camera(output, cfg, data, report["steps_expected"]) if momentary else None
    if camera is not None:
        checks.update({f"wrist_camera_{name}": passed for name, passed in camera["checks"].items()})
        checks["wrist_camera_complete"] = camera["success"]
    home_geometry_report = None
    if momentary and home_q.shape == (6,) and np.isfinite(home_q).all():
        home_geometry_report = {
            "configured": home_geometry(kin, home_q, cfg),
            "initial_measured": home_geometry(kin, data["q_actual"][0], cfg),
            "measured_returns": measured_home_geometries,
            "all_measured_returns_folded": bool(
                len(measured_home_geometries) == len(cfg["sequence"])
                and all(row["folded_joint_geometry"] for row in measured_home_geometries)
            ),
        }
    if cfg.get("home_posture") == "folded":
        checks["configured_home_geometry_folded"] = bool(
            home_geometry_report and home_geometry_report["configured"]["folded_joint_geometry"]
        )
        checks["all_measured_home_returns_folded"] = bool(
            home_geometry_report and home_geometry_report["all_measured_returns_folded"]
        )
    if "home_max_housing_tilt_deg" in cfg:
        tilt_limit = float(cfg["home_max_housing_tilt_deg"])
        checks["valid_home_housing_tilt_limit"] = bool(np.isfinite(tilt_limit) and 0 < tilt_limit <= 90)
        checks["configured_home_housing_tilts_within_limit"] = bool(
            home_geometry_report and home_geometry_report["configured"]["max_abs_housing_tilt_deg"] <= tilt_limit
        )
        checks["initial_home_housing_tilts_within_limit"] = bool(
            home_geometry_report and home_geometry_report["initial_measured"]["max_abs_housing_tilt_deg"] <= tilt_limit
        )
        checks["all_measured_home_housing_tilts_within_limit"] = bool(
            len(measured_home_geometries) == len(cfg["sequence"])
            and all(row["max_abs_housing_tilt_deg"] <= tilt_limit for row in measured_home_geometries)
        )
        if home_geometry_report is not None:
            home_geometry_report["housing_tilt_limit_deg"] = tilt_limit
    result = {
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "success": all(checks.values()),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "trajectory_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "urdf_sha256": hashlib.sha256(urdf_path.read_bytes()).hexdigest(),
        "samples": count,
        "duration_s": float(data["time"][-1]),
        "max_recomputed_fk_error_m": float(np.max(fk_error)),
        "max_joint_tracking_error_rad": float(np.max(joint_tracking_error)),
        "max_tip_tracking_error_m": float(np.max(tip_tracking_error)),
        "max_measured_joint_speed_rad_s": float(np.max(np.abs(data["qd_actual"]))),
        "floor_measurements": rows,
        "button_light_mode": "momentary" if momentary else "latched",
        "scope": "Recorded state/contact audit; unexpected collision check uses the runtime contact report.",
    }
    if camera is not None:
        result["wrist_camera"] = camera
    if home_geometry_report is not None:
        result["home_geometry"] = home_geometry_report
    if gripper is not None:
        result["gripper"] = gripper
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = audit(args.output)
    path = args.output / "audit_summary.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{'PASS' if result['success'] else 'FAIL'}: {path}")
    for row in result["floor_measurements"]:
        extra = (f", release_verified={row['release_verified']}, home_verified={row['home_verified']}"
                 if "release_verified" in row else "")
        print(f"  {row['floor']}: {row['peak_travel_mm']:.3f} mm, {row['peak_contact_force_n']:.3f} N, "
              f"event_verified={row['event_verified']}{extra}")
    if "wrist_camera" in result:
        camera = result["wrist_camera"]
        print(f"  wrist RGB-D: {camera['frames']} frames, success={camera['success']}")
        for error in camera["errors"]:
            print(f"  camera error: {error}")
    if not result["success"]:
        print("Failed checks: " + ", ".join(result["failed_checks"]))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
