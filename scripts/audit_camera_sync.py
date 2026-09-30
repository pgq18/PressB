#!/usr/bin/env python3
"""Read-only RGB/state synchrony audit using decoded video and physical geometry.

Run with the LeRobot environment (av, numpy, OpenCV, Pillow); no Isaac or GPU:
  .conda/envs/lerobot/bin/python scripts/audit_camera_sync.py DATASET_ROOT \
      --floors 24 29 35 --output outputs/dataset_camera_sync_audit
Use --episode-ids ID [ID ...] to audit exactly those committed episodes instead
of selecting the earliest episodes for each requested floor.

Positive pose lag means RGB matches an older measured joint configuration.
Silver button bezel edges test wrist-camera geometry; the known 5 mm pressing
sphere tests robot geometry in the fixed camera. Orange-pixel transitions are
reported separately from small antialiasing/denoising residuals. These are image
measurements, not a substitute for the 120 Hz physical-success/source audit.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import av
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pressb.panel_metadata import validate_episode_panel_metadata
LAGS = (-2, -1, 0, 1, 2)
LIGHT_PIXEL_CLASSIFIER = {
    "version": "amber_orange_hsv_v2",
    "color_space": "HSV computed from decoded RGB, hue in degrees",
    "predicate": "10 <= H <= 60 and S >= 0.35 and V > 100/255 and R-B > 60/255",
    "reason": "The lit edge is tone-mapped to orange/amber; its measured hue near 45 degrees "
              "must not be mistaken for off when R/G crosses the legacy 1.25 cutoff.",
    "legacy_count_field": "orange_pixels counts this amber/orange mask; the field name is retained for report compatibility",
}


def feedback_pixel_mask(rgb):
    """Classify saturated orange/amber light without accepting neutral digits.

    Hue selects the physical feedback color; saturation, brightness, and
    red-minus-blue contrast reject white/silver edges and dim neutral pixels.
    This changes color measurement only. Visibility, timing, and pose tests
    remain unchanged, including rejection of a one-frame release delay.
    """
    values = np.asarray(rgb, dtype=np.float32) / 255.
    hsv = cv2.cvtColor(values, cv2.COLOR_RGB2HSV)
    hue, saturation, value = np.moveaxis(hsv, -1, 0)
    return ((hue >= 10.) & (hue <= 60.) & (saturation >= .35)
            & (value > 100./255.) & ((values[..., 0] - values[..., 2]) > 60./255.))


def validate_video_rate(stream, metadata, manifest):
    """Reject rate mismatches even when a video and its frame times agree."""
    expected = Fraction(str(manifest["fps"]))
    episode_rate = Fraction(str(metadata["fps"]))
    if expected <= 0 or episode_rate != expected:
        raise ValueError("Episode fps differs from a valid collection fps")
    if stream.average_rate is None or Fraction(stream.average_rate) != expected:
        raise ValueError(f"Video fps {stream.average_rate} differs from collection fps {expected}")
    return float(expected)


def axis_rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + np.sin(angle) * cross + (1. - np.cos(angle)) * (cross @ cross)


class UrdfForward:
    """Independent NumPy evaluation of the source URDF's base_link→link6 chain."""
    def __init__(self, path):
        joints = ET.parse(path).getroot().findall("joint")
        by_child = {joint.find("child").get("link"): joint for joint in joints}
        chain, child = [], "link6"
        while child != "base_link":
            joint = by_child[child]
            chain.append(joint)
            child = joint.find("parent").get("link")
            if len(chain) > len(joints):
                raise ValueError("Cyclic URDF joint chain")
        self.chain, names = [], []
        for joint in reversed(chain):
            origin = joint.find("origin")
            xyz = np.fromstring(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" ")
            rpy = np.fromstring(origin.get("rpy", "0 0 0") if origin is not None else "0 0 0", sep=" ")
            transform = np.eye(4)
            transform[:3, :3] = (axis_rotation([0, 0, 1], rpy[2])
                                  @ axis_rotation([0, 1, 0], rpy[1])
                                  @ axis_rotation([1, 0, 0], rpy[0]))
            transform[:3, 3] = xyz
            axis = None
            if joint.get("type") != "fixed":
                if joint.get("type") != "revolute":
                    raise ValueError("Expected only fixed/revolute arm joints")
                axis_node = joint.find("axis")
                axis = np.fromstring(axis_node.get("xyz", "1 0 0") if axis_node is not None else "1 0 0", sep=" ")
                names.append(joint.get("name"))
            self.chain.append((transform, axis))
        if names != [f"joint{i}" for i in range(1, 7)]:
            raise ValueError(f"Unexpected measured-joint order: {names}")

    def link6(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError("Expected six finite measured joint angles")
        transform, index = np.eye(4), 0
        for origin, axis in self.chain:
            transform = transform @ origin
            if axis is not None:
                rotation = np.eye(4)
                rotation[:3, :3] = axis_rotation(axis, q[index])
                transform = transform @ rotation
                index += 1
        return transform


def quaternion_matrix(wxyz):
    w, x, y, z = np.asarray(wxyz, dtype=float) / np.linalg.norm(wxyz)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def project(points, camera, intrinsics):
    local = (np.asarray(points) - camera[:3, 3]) @ camera[:3, :3]
    depth = -local[:, 2]
    uv = np.column_stack((intrinsics[0, 0] * local[:, 0] / depth + intrinsics[0, 2],
                          intrinsics[1, 2] - intrinsics[1, 1] * local[:, 1] / depth))
    return uv, depth


def bezel_points(cfg, panel_offset_y_m=0., panel_offset_x_m=0.):
    points = []
    for floor in range(24, 36):
        x = cfg["button_face_x"] + panel_offset_x_m + .0045
        y = cfg["button_column_y"] * (1 if floor < 30 else -1) + panel_offset_y_m
        z = cfg["button_bottom_z"] + (floor - 24) % 6 * cfg["button_pitch_z"]
        for t in np.linspace(-.015, .015, 13):
            points.extend([[x, y-.016, z+t], [x, y+.016, z+t],
                           [x, y+t, z-.016], [x, y+t, z+.016]])
    return np.asarray(points)


def tip_front_arc(link6, fixed_camera, intrinsics):
    center = link6[:3, 3] + link6[:3, :3] @ [0., 0., .235]
    uv, depth = project(np.array([center, center + link6[:3, 2] * .005]), fixed_camera, intrinsics)
    direction = uv[1] - uv[0]
    angle = np.arctan2(direction[1], direction[0])
    theta = np.linspace(angle - np.pi/3, angle + np.pi/3, 24)
    radius = intrinsics[0, 0] * .005 / depth[0]
    arc = uv[0] + radius * np.column_stack((np.cos(theta), np.sin(theta)))
    return arc, np.full(len(arc), depth[0])


def edge_scores(rgb, projections, index, view):
    if index < 3 or index >= len(projections) - 3:
        return None
    candidates = [projections[index - lag] for lag in LAGS]
    visible = np.ones(len(candidates[0][0]), dtype=bool)
    height, width = rgb.shape[:2]
    for uv, depth in candidates:
        visible &= ((depth > .03) & (uv[:, 0] > 4) & (uv[:, 0] < width-5)
                    & (uv[:, 1] > 4) & (uv[:, 1] < height-5))
    if visible.sum() < (65 if view == "wrist" else 20):
        return None
    motion = np.linalg.norm(candidates[2][0][visible] - candidates[3][0][visible], axis=1).mean()
    if motion < 1.5:
        return None
    gray = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32), (3, 3), .7)
    gradient = cv2.magnitude(cv2.Scharr(gray, cv2.CV_32F, 1, 0), cv2.Scharr(gray, cv2.CV_32F, 0, 1))
    scores = [float(cv2.remap(gradient, uv[visible, 0].astype(np.float32).reshape(-1, 1),
                             uv[visible, 1].astype(np.float32).reshape(-1, 1), cv2.INTER_LINEAR).mean())
              for uv, _ in candidates]
    return {"frame": index, "projected_motion_pixels": float(motion),
            "visible_edge_samples": int(visible.sum()), "scores_by_lag": scores,
            "best_pose_lag_frames": LAGS[int(np.argmax(scores))]}


def summarize_pose(samples, view):
    if not samples:
        return {"sufficient_evidence": False, "aligned": False, "samples": []}
    mean_scores = np.mean([sample["scores_by_lag"] for sample in samples], axis=0)
    counts = Counter(sample["best_pose_lag_frames"] for sample in samples)
    best = LAGS[int(np.argmax(mean_scores))]
    ratio = float(mean_scores[2] / max(mean_scores[3], 1e-9))
    # Fixed-view tool edges have frequent occlusion; use their aggregate score.
    enough = len(samples) >= 20
    aligned = enough and best == 0 and ratio > 1.15
    if view == "wrist":
        aligned = aligned and counts[0] / len(samples) >= .70
    return {"sufficient_evidence": enough, "aligned": bool(aligned),
            "tested_lags_frames": list(LAGS), "best_aggregate_pose_lag_frames": best,
            "current_over_previous_edge_score_ratio": ratio,
            "best_lag_counts": {str(lag): counts[lag] for lag in LAGS},
            "mean_edge_scores": mean_scores.tolist(), "samples": samples}


def summarize_light(samples, first, last):
    counts = np.array([sample["orange_pixels"] for sample in samples])
    plateau = float(np.median(counts[first:last+1]))
    if plateau < 20:
        return {"sufficient_visibility": False, "aligned": None,
                "reason": "Target button is occluded or too small for an orange-pixel timing decision"}
    dominant_threshold = max(5., plateau * .25)
    press = next((i for i in range(max(0, first-3), min(len(counts), first+4))
                  if counts[i] >= dominant_threshold), None)
    release = next((i for i in range(last, min(len(counts), last+20))
                    if counts[i] < dominant_threshold), None)
    return {"sufficient_visibility": True, "aligned": press == first and release == last+1,
            "label_first_lit_frame": first, "label_first_unlit_frame": last+1,
            "visible_first_lit_frame": press, "visible_first_unlit_frame": release,
            "median_lit_orange_pixels": plateau, "dominant_light_threshold_pixels": dominant_threshold,
            "first_unlit_orange_pixels": int(counts[last+1]),
            "first_unlit_residual_fraction": float(counts[last+1] / plateau),
            "strict_zero_after_release_frame": next((i for i in range(last+1, len(counts)) if counts[i] == 0), None),
            "note": "Dominant light timing allows <25% residual orange pixels from edge filtering/denoising; exact residual is reported, not hidden."}


def font():
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), 14) if path.exists() else ImageFont.load_default()


def save_sheets(output, name, floor, view, images, samples, first, last):
    for transition, center in (("press", first), ("release", last+1)):
        indices = range(max(0, center-3), min(len(samples), center+4))
        sheet = Image.new("RGB", (320 * len(indices), 464), (24, 26, 30))
        draw = ImageDraw.Draw(sheet)
        for col, index in enumerate(indices):
            sample, image = samples[index], images[index]
            x = col * 320
            sheet.paste(image.resize((320, 240)), (x, 34))
            draw.text((x+4, 4), f"{view} F{floor} frame {index} light={sample['label_lit']}", font=font(),
                      fill=(255, 210, 40) if sample["label_lit"] else "white")
            if sample["bbox_valid"]:
                crop = image.crop(sample["bbox_xyxy"])
                crop.thumbnail((150, 150))
                sheet.paste(crop, (x + (320-crop.width)//2, 274 + (150-crop.height)//2))
            draw.text((x+4, 432), f"t={sample['time_s']:.3f}s amber={sample['orange_pixels']}", font=font(), fill="white")
        sheet.save(output / f"{name}_floor{floor}_{view}_{transition}.png")


def audit_episode(directory, manifest, output, fk):
    meta = json.loads((directory / "metadata.json").read_text())
    cfg, floor = manifest["config"], meta["floor"]
    panel_offset_x, panel_offset_y = validate_episode_panel_metadata(manifest, meta)
    with np.load(directory / "frames.npz", allow_pickle=False) as source:
        frames = dict(source)
    with np.load(directory / "physics.npz", allow_pickle=False) as source:
        physics = dict(source)
    lit = np.flatnonzero(frames["lights"][:, floor-24])
    if not len(lit) or lit[-1] >= len(frames["state"])-1:
        raise ValueError("Episode lacks a captured press and release")
    first, last = int(lit[0]), int(lit[-1])
    keep = (set(range(max(0, first-3), min(len(frames["state"]), first+4)))
            | set(range(max(0, last-2), min(len(frames["state"]), last+5)))
            | {0, (first+last)//2, len(frames["state"])-1})
    base = np.array([cfg["robot_base_x"], cfg["robot_base_y"], cfg["table_height"]])
    link6_poses = [fk.link6(q) for q in frames["q_actual"]]
    for pose in link6_poses:
        pose[:3, 3] += base
    wrist = manifest["wrist"]["sensor"]
    local_camera = np.eye(4)
    local_camera[:3, :3] = quaternion_matrix(wrist["optical_quaternion_wxyz_link6"])
    local_camera[:3, 3] = wrist["optical_position_link6"]
    wrist_poses = [pose @ local_camera for pose in link6_poses]
    fixed = np.asarray(manifest["global"]["sensor"]["world_optical_transform"])
    result = {"episode_id": meta["episode_id"], "floor": floor, "seed": meta["seed"],
              "panel_offset_x_m": panel_offset_x, "panel_offset_y_m": panel_offset_y,
              "env_offset_m": meta["env_offset_m"], "source_directory": str(directory),
              "lit_frame_indices": lit.tolist(), "events": meta["events"], "views": {}}
    overview_images = {}
    for view in ("wrist", "global"):
        intrinsics = np.asarray(manifest[view]["K"])
        camera_poses = wrist_poses if view == "wrist" else [fixed] * len(link6_poses)
        geometry = ([project(bezel_points(cfg, panel_offset_y, panel_offset_x), pose, intrinsics) for pose in camera_poses]
                    if view == "wrist" else [tip_front_arc(pose, fixed, intrinsics) for pose in link6_poses])
        samples, edge_samples, images, hashes, diffs = [], [], {}, set(), []
        previous = None
        with av.open(str(directory / f"{view}.mp4")) as container:
            stream = container.streams.video[0]
            fps = validate_video_rate(stream, meta, manifest)
            stream.codec_context.thread_count = 1
            for index, video_frame in enumerate(container.decode(video=0)):
                if index >= len(frames["state"]):
                    raise ValueError(f"Too many {view} video frames")
                rgb = video_frame.to_ndarray(format="rgb24")
                if rgb.shape != (480, 640, 3):
                    raise ValueError(f"Wrong RGB dimensions: {rgb.shape}")
                hashes.add(hashlib.sha256(rgb.tobytes()).hexdigest())
                if previous is not None:
                    diffs.append(float(np.mean(abs(rgb.astype(np.int16)-previous.astype(np.int16)))))
                previous = rgb
                x = cfg["button_face_x"] + panel_offset_x + physics["button_travel"][frames["physics_index"][index], floor-24]
                y = cfg["button_column_y"] * (1 if floor < 30 else -1) + panel_offset_y
                z = cfg["button_bottom_z"] + (floor-24) % 6 * cfg["button_pitch_z"]
                corners = np.array([[x, y+dy, z+dz] for dy, dz in itertools.product([-.013, .013], repeat=2)])
                uv, depth = project(corners, camera_poses[index], intrinsics)
                left, top = np.maximum(np.floor(uv.min(0)).astype(int)-2, [0, 0])
                right, bottom = np.minimum(np.ceil(uv.max(0)).astype(int)+3, [640, 480])
                valid = bool((depth > 0).all() and left < right and top < bottom)
                roi = rgb[top:bottom, left:right] if valid else np.zeros((1, 1, 3), dtype=np.uint8)
                orange = feedback_pixel_mask(roi)
                samples.append({"frame": index, "time_s": float(video_frame.pts * video_frame.time_base),
                                "label_lit": int(frames["lights"][index, floor-24]),
                                "orange_pixels": int(orange.sum()), "bbox_valid": valid,
                                "bbox_xyxy": [int(left), int(top), int(right), int(bottom)]})
                scored = edge_scores(rgb, geometry, index, view)
                if scored:
                    edge_samples.append(scored)
                if index in keep:
                    images[index] = Image.fromarray(rgb)
        count_matches = len(samples) == len(frames["state"]) == meta["num_frames"]
        if not count_matches:
            raise ValueError(f"{view} video/NPZ frame count mismatch")
        pts_match = bool(np.allclose([sample["time_s"] for sample in samples], frames["sim_time"], rtol=0, atol=1e-8))
        dynamic = len(hashes) > len(samples) * .5 and float(np.mean(diffs)) > .05
        pose = summarize_pose(edge_samples, view)
        light = summarize_light(samples, first, last)
        result["views"][view] = {"decoded_frames": len(samples), "frame_count_matches": count_matches,
            "fps": fps, "video_rate_matches_metadata": True,
            "video_pts_match_state_times": pts_match, "unique_frame_hashes": len(hashes),
            "mean_adjacent_rgb_change": float(np.mean(diffs)), "dynamic_video": dynamic,
            "pose_geometry": pose, "light_timing": light, "samples": samples,
            "success": bool(count_matches and pts_match and dynamic and pose["aligned"] and light["aligned"] is not False)}
        save_sheets(output, directory.name, floor, view, images, samples, first, last)
        overview_images[view] = images
    overview = Image.new("RGB", (1440, 780), (24, 26, 30))
    draw = ImageDraw.Draw(overview)
    for row, view in enumerate(("wrist", "global")):
        for col, index in enumerate((0, (first+last)//2, len(frames["state"])-1)):
            overview.paste(overview_images[view][index].resize((480, 360)), (480*col, 390*row+30))
            draw.text((col*480+5, row*390+5), f"{directory.name} F{floor} {view} frame {index}", font=font(), fill="white")
    overview.save(output / f"{directory.name}_floor{floor}_overview.png")
    result["success"] = all(view["success"] for view in result["views"].values())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Committed raw episode dataset root")
    parser.add_argument("--floors", type=int, nargs="+", default=[24, 29, 35])
    parser.add_argument("--episodes-per-floor", type=int, default=1)
    parser.add_argument("--episode-ids", type=int, nargs="+",
                        help="Exact committed episode IDs; overrides floor-based selection")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raw = args.root.resolve()
    output = (args.output or ROOT / "outputs" / f"{raw.name}_camera_sync_audit").resolve()
    if output == raw or not all(24 <= floor <= 35 for floor in args.floors) or args.episodes_per_floor < 1:
        parser.error("Use an independent output directory and valid floor/sample counts")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((raw / "collection_metadata.json").read_text())
    fk = UrdfForward(ROOT / manifest["config"]["robot_urdf"])
    selection_errors, directories = [], []
    if args.episode_ids is not None:
        if any(value < 0 for value in args.episode_ids) or len(set(args.episode_ids)) != len(args.episode_ids):
            parser.error("--episode-ids must contain distinct nonnegative IDs")
        requested_floors = []
        for episode_id in args.episode_ids:
            directory = raw / f"episode_{episode_id:06d}"
            metadata_path = directory / "metadata.json"
            if not metadata_path.is_file():
                parser.error(f"Requested committed episode does not exist: {metadata_path}")
            meta = json.loads(metadata_path.read_text())
            if meta.get("episode_id") != episode_id or meta.get("success") is not True:
                parser.error(f"Invalid committed episode identity/success: {metadata_path}")
            directories.append(directory)
            if meta["floor"] not in requested_floors:
                requested_floors.append(meta["floor"])
    else:
        requested_floors = args.floors
        selected = {floor: [] for floor in args.floors}
        for directory in sorted(raw.glob("episode_*")):
            if not (directory / "metadata.json").is_file():
                continue
            meta = json.loads((directory / "metadata.json").read_text())
            floor = meta["floor"]
            if floor in selected and len(selected[floor]) < args.episodes_per_floor:
                selected[floor].append(directory)
        for floor, matching in selected.items():
            if len(matching) < args.episodes_per_floor:
                selection_errors.append(f"Insufficient committed episodes for floor {floor}")
            directories.extend(matching)
    report = {"success": False, "source_root": str(raw), "collection_fingerprint": manifest.get("collection_fingerprint"),
              "light_pixel_classifier": LIGHT_PIXEL_CLASSIFIER,
              "source_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "requested_floors": requested_floors, "requested_episode_ids": args.episode_ids,
              "selection_mode": "explicit_episode_ids" if args.episode_ids is not None else "earliest_per_floor",
              "episodes": [], "errors": selection_errors,
              "pose_lag_sign": "Positive lag means RGB fits an older measured joint configuration",
              "world_offset_handling": "Both camera and scene are translated equally by env_offset; projection uses environment-zero coordinates",
              "limitations": "Silver-frame and sphere silhouette scores are image evidence; occlusion/rounding can make individual frames ambiguous. Full physics/source integrity is audited separately."}
    for directory in directories:
        try:
            result = audit_episode(directory, manifest, output, fk)
            report["episodes"].append(result)
            if not result["success"]:
                report["errors"].append(f"Visual synchrony checks failed: {directory.name}")
            print(json.dumps({"episode": directory.name, "success": result["success"],
                "views": {view: {"pose_lag": values["pose_geometry"].get("best_aggregate_pose_lag_frames"),
                                 "pose_ratio_current_over_previous": values["pose_geometry"].get("current_over_previous_edge_score_ratio"),
                                 "light": values["light_timing"]} for view, values in result["views"].items()}}), flush=True)
        except Exception as exc:
            report["errors"].append(f"{directory.name}: {type(exc).__name__}: {exc}")
    report["success"] = bool(report["episodes"]) and not report["errors"]
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"success": report["success"], "errors": report["errors"], "report": str(output / "report.json")}))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
