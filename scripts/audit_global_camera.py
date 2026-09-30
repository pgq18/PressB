#!/usr/bin/env python3
"""Independently audit fixed D435 RGB-D recordings without starting Isaac Sim.

Coverage checks the complete panel bounds and reports every measured robot
pose. ``global_camera_coverage=panel`` allows robot cropping; the legacy default
requires all checked points in view. This does not prove lack of occlusion.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pressb.kinematics import PiperKinematics


def read_json(path):
    return json.loads(path.read_text())


def read_manifest(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def project_coverage(points, position, world_from_camera, matrix, width, height, margin=5.):
    """Project world points using USD camera axes, and retain worst-case margins."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    local = (points - position) @ world_from_camera
    depth = -local[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.column_stack((matrix[0, 0] * local[:, 0] / depth + matrix[0, 2],
                              matrix[1, 2] - matrix[1, 1] * local[:, 1] / depth))
    finite = np.isfinite(uv).all(axis=1) & np.isfinite(depth)
    margins = np.column_stack((uv[:, 0], width - 1 - uv[:, 0],
                               uv[:, 1], height - 1 - uv[:, 1]))
    inside = finite & (depth > .005) & np.all(margins >= margin, axis=1)
    return {"points": len(points), "all_inside": bool(inside.all()),
            "inside_fraction": float(inside.mean()), "required_margin_px": margin,
            "minimum_optical_depth_m": float(depth.min()),
            "minimum_margin_px": float(margins.min()),
            "pixel_bounds": [uv.min(axis=0).tolist(), uv.max(axis=0).tolist()]}


def audit_tabletop_mount(sensor, cfg, position):
    """Check the recorded stand bounds against scene.py's precise tabletop."""
    checks = {name: False for name in (
        "tabletop_support_metadata", "stand_base_on_tabletop", "stand_inside_table_edges",
        "camera_left_beside_robot", "camera_above_tabletop",
    )}
    result = {"checks": checks, "bounds_basis": "Recorded stand world_bounds_m and configured tabletop geometry",
              "support_height_tolerance_m": .001, "table_edge_tolerance_m": .0001,
              "maximum_horizontal_distance_from_robot_m": .6}
    try:
        stand = sensor["stand"]
        height = float(cfg["table_height"])
        support_z = float(stand["support_z"])
        bounds = np.asarray(stand["world_bounds_m"], dtype=float)
        edge = float(cfg.get("table_edge_x", .10))
        depth, width = float(cfg.get("table_depth", .75)), float(cfg.get("table_width", 1.))
        table_xy = np.array([[edge - depth, -width / 2], [edge, width / 2]])
        base = np.array([cfg.get("robot_base_x", 0.), cfg.get("robot_base_y", 0.), height])
        position = np.asarray(position, dtype=float)
        checks["tabletop_support_metadata"] = bool(
            stand["support_surface"] == "tabletop" and bounds.shape == (2, 3)
            and np.isfinite(bounds).all() and np.all(bounds[1] > bounds[0])
            and np.isfinite([support_z, height, edge, depth, width]).all()
            and depth > 0 and width > 0 and abs(support_z - height) <= .001
            and position.shape == (3,) and np.isfinite(position).all())
        if not checks["tabletop_support_metadata"]:
            return result
        checks["stand_base_on_tabletop"] = bool(abs(bounds[0, 2] - height) <= .001)
        edge_clearance = np.concatenate((bounds[0, :2] - table_xy[0], table_xy[1] - bounds[1, :2]))
        checks["stand_inside_table_edges"] = bool(np.all(edge_clearance >= -.0001))
        horizontal_distance = float(np.linalg.norm(position[:2] - base[:2]))
        checks["camera_left_beside_robot"] = bool(position[1] > base[1] and horizontal_distance < .6)
        checks["camera_above_tabletop"] = bool(position[2] > height)
        result.update(support_surface=stand["support_surface"], support_z=support_z,
                      stand_world_bounds_m=bounds.tolist(), table_xy_bounds_m=table_xy.tolist(),
                      stand_base_height_error_m=float(abs(bounds[0, 2] - height)),
                      minimum_table_edge_clearance_m=float(edge_clearance.min()),
                      camera_robot_horizontal_distance_m=horizontal_distance,
                      camera_robot_vertical_offset_m=float(position[2] - height))
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def audit(output):
    output = Path(output).resolve()
    root = output / "global_camera"
    checks, errors = {}, []
    result = {"success": False, "checked_at": datetime.now(timezone.utc).isoformat(),
              "episode": str(output), "checks": checks, "errors": errors}
    try:
        episode = read_json(output / "report.json")
        status = read_json(output / "run_status.json")
        cfg = episode["config"]
        panel_offset = episode.get("panel_offset_y_m", cfg.get("panel_offset_y_m", 0.))
        panel_offset_x = episode.get("panel_offset_x_m", cfg.get("panel_offset_x_m", 0.))
        if any(type(value) not in (int, float) or not np.isfinite(value) for value in (panel_offset_x, panel_offset)):
            raise ValueError("Invalid episode panel offsets")
        result["panel_offset_x_m"] = float(panel_offset_x)
        result["panel_offset_y_m"] = float(panel_offset)
        coverage_mode = cfg.get("global_camera_coverage", "all")
        checks["known_coverage_mode"] = coverage_mode in ("all", "panel")
        result["coverage_mode"] = coverage_mode
        report = read_json(root / "camera_report.json")
        calibration = read_json(root / "intrinsics.json")
        records = read_manifest(root / "timestamps.jsonl")
        sensor = report["sensor"]
        with np.load(output / "trajectory.npz", allow_pickle=False) as archive:
            data = {key: archive[key] for key in ("time", "floor", "phase", "q_actual")}
        count = len(data["time"])
        expected_steps = int(episode["steps_expected"])
        checks["completed_physics_episode"] = bool(
            status["status"] == "complete" and episode["success"] and episode["physics_executed"]
            and count == episode["samples"] == episode["steps_completed"] == expected_steps and count > 1)
        checks["finite_measured_trajectory"] = bool(
            data["time"].shape == data["floor"].shape == data["phase"].shape == (count,)
            and data["q_actual"].shape == (count, 6) and np.isfinite(data["q_actual"]).all()
            and np.isfinite(data["time"]).all())
        stride = int(report["capture_stride"])
        if stride <= 0:
            raise ValueError("Capture stride must be positive")
        expected_frames = (expected_steps - 1) // stride + 1
        checks["complete_frame_count"] = bool(
            len(records) > 1 and len(records) == expected_frames == report["expected_frames"]
            == report["valid_frames"] and report["success"]
            and stride == cfg["global_camera_capture_stride"])
        checks["sequential_frame_ids"] = [row["frame"] for row in records] == list(range(len(records)))
        times = np.asarray([row["time"] for row in records], dtype=float)
        sample_times = data["time"][::stride]
        checks["timestamps_match_physics"] = bool(
            times.shape == sample_times.shape and np.isfinite(times).all()
            and np.allclose(times, sample_times, atol=1e-7, rtol=0))
        checks["uniform_capture_interval"] = bool(
            len(times) > 1 and np.allclose(np.diff(times), stride * cfg["physics_dt"], atol=1e-6, rtol=0))
        checks["frame_phase_matches_physics"] = bool(
            [row["phase"] for row in records] == data["phase"][::stride].tolist()
            and [row["floor"] for row in records] == data["floor"][::stride].tolist())
        wrist_records = read_manifest(output / "wrist_camera" / "timestamps.jsonl")
        wrist_times = np.asarray([row["time"] for row in wrist_records])
        checks["synchronized_with_wrist_camera"] = bool(
            wrist_times.shape == times.shape and np.allclose(wrist_times, times, rtol=0, atol=1e-7)
            and [row["frame"] for row in wrist_records] == [row["frame"] for row in records])

        width, height = int(calibration["width"]), int(calibration["height"])
        matrix = np.asarray(calibration["K"], dtype=float)
        focal_px = 480. / (2. * np.tan(np.radians(42. / 2.)))
        nominal = np.array([[focal_px, 0., 320.], [0., focal_px, 240.], [0., 0., 1.]])
        checks["resolution_640_by_480"] = bool(
            (width, height) == (640, 480) and sensor["resolution"] == [640, 480]
            and cfg["global_camera_resolution"] == [640, 480])
        checks["nominal_d435_calibration"] = bool(
            matrix.shape == (3, 3) and np.isfinite(matrix).all()
            and np.allclose(matrix, nominal, atol=1e-3, rtol=0)
            and np.allclose([calibration[key] for key in ("fx", "fy", "cx", "cy")],
                            [nominal[0, 0], nominal[1, 1], 320., 240.], atol=1e-3, rtol=0)
            and np.allclose(sensor["intrinsics"], nominal, atol=1e-3, rtol=0)
            and "D435" in sensor["model"])
        nominal_fov = np.degrees(2. * np.arctan(np.array([640., 480.]) / (2. * focal_px)))
        optics_fov = np.degrees(2. * np.arctan(np.array([
            sensor["horizontal_aperture_mm"], sensor["vertical_aperture_mm"]])
            / (2. * sensor["focal_length_mm"])))
        checks["sensor_optics_match_calibration"] = bool(
            sensor["focal_length_mm"] > 0. and np.allclose(optics_fov, nominal_fov, rtol=0, atol=1e-3)
            and np.allclose(sensor["field_of_view_deg"], nominal_fov, rtol=0, atol=1e-3)
            and calibration["sensor"] == sensor and episode["global_camera"]["sensor"] == sensor)
        checks["metric_depth_and_pose_conventions"] = bool(
            calibration["depth_unit"] == "m" and calibration["depth_type"] == "distance_to_image_plane"
            and calibration["quaternion_order"] == "wxyz"
            and calibration["pose_axes"] == "USD camera: +X right, +Y up, -Z forward")

        position = np.asarray(sensor["fixed_position_world"], dtype=float)
        quaternion = np.asarray(sensor["fixed_quaternion_wxyz_world"], dtype=float)
        positions = np.asarray([row["position_world"] for row in records], dtype=float)
        quaternions = np.asarray([row["orientation_world"] for row in records], dtype=float)
        checks["valid_fixed_camera_poses"] = bool(
            position.shape == (3,) and quaternion.shape == (4,)
            and positions.shape == (len(records), 3) and quaternions.shape == (len(records), 4)
            and np.isfinite(position).all() and np.isfinite(quaternion).all()
            and np.isfinite(positions).all() and np.isfinite(quaternions).all()
            and np.isclose(np.linalg.norm(quaternion), 1., rtol=0, atol=1e-5)
            and np.allclose(np.linalg.norm(quaternions, axis=1), 1., rtol=0, atol=1e-5))
        if not checks["valid_fixed_camera_poses"] or not checks["finite_measured_trajectory"]:
            raise ValueError("Invalid camera poses or measured joint trajectory")
        rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]])
        world_from_camera = rotation.as_matrix()
        position_error = np.linalg.norm(positions - position, axis=1)
        rotation_error = (rotation.inv() * Rotation.from_quat(quaternions[:, [1, 2, 3, 0]])).magnitude()
        checks["camera_world_pose_is_fixed"] = bool(position_error.max() <= 1e-5 and rotation_error.max() <= 1e-5)
        target_ray = np.asarray(cfg["global_camera_target"], dtype=float) - position
        target_ray /= np.linalg.norm(target_ray)
        checks["fixed_pose_matches_scene_configuration"] = bool(
            np.allclose(position, cfg["global_camera_eye"], rtol=0, atol=1e-5)
            and np.allclose(-world_from_camera[:, 2], target_ray, rtol=0, atol=1e-5))
        result.update(frames=len(records), expected_frames=expected_frames, physics_samples=count,
                      capture_stride=stride, calibration_K=matrix.tolist(),
                      maximum_position_error_m=float(position_error.max()),
                      maximum_orientation_error_rad=float(rotation_error.max()))
        if (coverage_mode == "panel" or cfg.get("global_camera_support_surface") == "tabletop"
                or sensor.get("stand", {}).get("support_surface") == "tabletop"):
            mount = audit_tabletop_mount(sensor, cfg, position)
            result["tabletop_mount"] = mount
            checks.update(mount["checks"])
            if "error" in mount:
                errors.append("tabletop mount: " + mount["error"])

        fractions, rgb_hashes, depth_hashes, referenced = [], set(), set(), set()
        frame_errors = []
        for row in records:
            try:
                rgb_path = (root / row["rgb"]).resolve()
                depth_path = (root / row["depth"]).resolve()
                if (rgb_path.parent != (root / "rgb").resolve() or rgb_path.suffix != ".png"
                        or depth_path.parent != (root / "depth").resolve() or depth_path.suffix != ".npy"
                        or rgb_path in referenced or depth_path in referenced):
                    raise ValueError("Duplicate or out-of-directory frame path")
                referenced.update((rgb_path, depth_path))
                with Image.open(rgb_path) as image:
                    rgb = np.asarray(image)
                depth = np.load(depth_path, allow_pickle=False)
                if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8 or float(rgb.std()) < 1.:
                    raise ValueError("RGB shape/type is invalid or image is blank")
                if depth.shape != (480, 640) or depth.dtype != np.float32:
                    raise ValueError("Depth must be a 480x640 float32 array")
                valid = np.isfinite(depth) & (depth > 0.)
                fraction = float(valid.mean())
                if fraction < .05 or np.any(np.isfinite(depth) & (depth < 0.)):
                    raise ValueError("Insufficient positive finite depth or negative metric depth")
                fractions.append(fraction)
                rgb_hashes.add(hashlib.sha256(rgb.tobytes()).digest())
                depth_hashes.add(hashlib.sha256(depth.tobytes()).digest())
            except (OSError, ValueError, TypeError, KeyError) as exc:
                frame_errors.append(f"frame {row.get('frame')}: {exc}")
        present = {p.resolve() for p in (root / "rgb").glob("*.png")}
        present.update(p.resolve() for p in (root / "depth").glob("*.npy"))
        checks["all_rgb_depth_frames_valid"] = not frame_errors and len(fractions) == len(records) and bool(records)
        checks["no_unreferenced_or_missing_frame_files"] = present == referenced and len(referenced) == 2 * len(records)
        checks["rgb_and_depth_change_during_motion"] = len(rgb_hashes) > 1 and len(depth_hashes) > 1
        checks["depth_fraction_matches_report"] = bool(
            fractions and np.mean(fractions) >= .1
            and abs(float(np.mean(fractions)) - report["finite_depth_fraction"]) <= 1e-6
            and abs(min(fractions) - report["minimum_finite_depth_fraction"]) <= 1e-6
            and report["finite_depth_fraction_aggregation"] == "mean")
        errors.extend(frame_errors[:20])
        result.update(finite_depth_fraction=float(np.mean(fractions)) if fractions else 0.,
                      minimum_finite_depth_fraction=min(fractions, default=0.),
                      distinct_rgb_frames=len(rgb_hashes), distinct_depth_frames=len(depth_hashes),
                      invalid_frames=len(frame_errors))

        base = np.array([cfg.get("robot_base_x", 0.), cfg.get("robot_base_y", 0.), cfg["table_height"]])
        kin = PiperKinematics(ROOT / cfg["robot_urdf"], tip_offset=cfg["tip_offset"])
        link_names = ["base_link"] + [f"link{i}" for i in range(1, 7)]
        arm_points, tip_points = [], []
        for q in data["q_actual"]:
            frames = kin.link_transforms(q)
            arm_points.append([frames[name][:3, 3] + base for name in link_names])
            tip_points.append(frames["link6"][:3, 3] + base
                              + frames["link6"][:3, :3] @ np.array([0., 0., cfg["tip_offset"]]))
        panel_center = np.array([cfg["button_face_x"] + panel_offset_x + .033, panel_offset,
                                 cfg["button_bottom_z"] + 2.5 * cfg["button_pitch_z"]])
        panel_half = np.array([.016, .10, (5. * cfg["button_pitch_z"] + .14) / 2.])
        corners = np.asarray(list(product((-1., 1.), repeat=3)))
        panel_points = panel_center + corners * panel_half
        cap_points = np.concatenate([
            np.array([cfg["button_face_x"] + panel_offset_x + .002, y + panel_offset, cfg["button_bottom_z"] + row * cfg["button_pitch_z"]])
            + corners * [.002, .013, .013]
            for y in (-cfg["button_column_y"], cfg["button_column_y"]) for row in range(6)])
        coverage = {name: project_coverage(points, position, world_from_camera, matrix, width, height)
                    for name, points in (("panel_frame", panel_points), ("button_caps", cap_points),
                                         ("robot_link_origins", arm_points), ("probe_tip", tip_points))}
        for name, group in coverage.items():
            group["required"] = coverage_mode != "panel" or name in ("panel_frame", "button_caps")
            if group["required"]:
                checks[f"{name}_inside_camera_frustum"] = group["all_inside"]
        result["coverage"] = coverage
        result["coverage_scope"] = ("All measured physics samples, official URDF base_link/link1..6 origins and "
                                    "probe tip; full panel frame and 12 cap bounding corners. "
                                    "Projection with >=5 pixel margins; occlusion is not evaluated. "
                                    + ("Only panel/caps coverage is required; robot cropping is allowed."
                                       if coverage_mode == "panel" else "All checked points must be in view."))
        checks["audit_completed"] = True
    except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        checks["audit_completed"] = False
        errors.append(f"{type(exc).__name__}: {exc}")
    result["failed_checks"] = [name for name, passed in checks.items() if not passed]
    result["success"] = bool(checks) and all(checks.values())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--output", type=Path, help="Defaults to EPISODE/global_camera_audit.json")
    args = parser.parse_args()
    result = audit(args.episode)
    destination = args.output or args.episode / "global_camera_audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
