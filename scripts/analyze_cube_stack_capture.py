#!/usr/bin/env python3
"""Offline cube measurements from a stationary dual-camera RGB-D capture.

Requires explicit top-face pixel annotations. It does not recalibrate the scene,
connect to hardware, or treat approximate depth as ground-truth cube dimensions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pressb.robots.piper.kinematics import PiperKinematics


def identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def rays(uv, intr):
    uv = np.asarray(uv, dtype=float)
    return np.c_[(uv[:, 0] - intr["cx"]) / intr["fx"],
                 (uv[:, 1] - intr["cy"]) / intr["fy"], np.ones(len(uv))]


def intersect_height(uv, intr, transform, height):
    directions = rays(uv, intr) @ transform[:3, :3].T
    length = (height - transform[2, 3]) / directions[:, 2]
    if not np.all(np.isfinite(length)) or np.any(length <= 0):
        raise ValueError("Annotated top rays do not meet the selected height in front of the camera")
    return transform[:3, 3] + directions * length[:, None]


def measure(capture, config_path, annotations):
    meta = json.loads((capture / "capture.json").read_text())
    cfg = json.loads(config_path.read_text())
    if not meta["success"] or max(meta["feedback"]["joint_range_deg"]) > .1:
        raise ValueError("Need a successful stationary capture")
    urdf = Path(cfg["robot_urdf"])
    urdf = urdf if urdf.is_absolute() else ROOT / urdf
    flip = np.diag([1., -1., -1., 1.])
    q = np.deg2rad(meta["feedback"]["median_joints_deg"])
    transforms = {
        "global": np.asarray(cfg["up_camera"]["base_from_camera_usd"]) @ flip,
        "wrist": PiperKinematics(urdf, tip_offset=0).fk(q)
                 @ np.asarray(cfg["wrist_camera"]["link6_from_camera_usd"]) @ flip,
    }
    table_z = -float(cfg["robot_base_height_above_table_m"])
    report = dict(capture=identity(capture / "capture.json"), scene=identity(config_path),
                  measured_joints_deg=meta["feedback"]["median_joints_deg"],
                  cameras={}, suggested_objects={}, cross_view={})
    for name, view in annotations["views"].items():
        intr = meta["cameras"][name]["aligned_depth_intrinsics"]
        if any(abs(v) > 1e-12 for v in intr["coeffs"]):
            raise ValueError("This pinhole analysis requires zero aligned-depth distortion")
        arrays = np.load(capture / f"{name}_depth.npz")
        depth, count = arrays["depth_m"], arrays["valid_count"]
        vv, uu = np.indices(depth.shape)
        xyz = rays(np.c_[uu.ravel(), vv.ravel()], intr).reshape(*depth.shape, 3) * depth[..., None]
        valid = np.isfinite(depth) & (depth > .1) & (depth < 2.) & (count >= 10)
        x0, y0, x1, y1 = view["table_roi_xyxy"]
        table = np.zeros(depth.shape, np.uint8)
        table[y0:y1, x0:x1] = 1
        for corners in view["top_corners_uv"].values():
            region = np.zeros_like(table)
            cv2.fillConvexPoly(region, np.asarray(corners, np.int32), 1)
            table[cv2.dilate(region, np.ones((15, 15), np.uint8)) > 0] = 0
        points = xyz[(table > 0) & valid][::5]
        for _ in range(7):
            if len(points) < 100:
                raise ValueError("Insufficient bare-table depth support")
            centre = points.mean(axis=0)
            _, _, vectors = np.linalg.svd(points - centre, full_matrices=False)
            normal = vectors[-1]
            distance = normal @ centre
            if distance < 0:
                normal, distance = -normal, -distance
            points = points[np.abs(points @ normal - distance) < .004]
        transform = transforms[name]
        normal_base = transform[:3, :3] @ normal
        constant_base = distance + normal_base @ transform[:3, 3]
        row = dict(intrinsics=intr, base_from_camera_cv=transform.tolist(),
            table_plane=dict(normal_cv=normal.tolist(), distance_m=float(distance),
                inliers=len(points), rms_m=float(np.sqrt(np.mean((points @ normal - distance)**2))),
                upward_normal_base=(-normal_base).tolist(),
                normal_tilt_deg=float(np.rad2deg(np.arccos(np.clip(-normal_base[2], -1, 1))))), objects={})
        for color, corners in view["top_corners_uv"].items():
            corners = np.asarray(corners, float)
            edge = float(annotations["objects"][color]["edge_prior_m"])
            if corners.shape != (4, 2) or edge <= 0:
                raise ValueError("Require four top corners and a positive cube size prior")
            region = np.zeros(depth.shape, np.uint8)
            cv2.fillConvexPoly(region, corners.astype(np.int32), 1)
            region = cv2.erode(region, np.ones((5, 5), np.uint8)).astype(bool) & valid
            if region.sum() < 20:
                raise ValueError("Too few valid cube-top depth samples")
            top_distances = xyz[region] @ normal
            top_distance = float(np.median(top_distances))
            top_rays = rays(corners, intr)
            top_depth_points = top_rays * (top_distance / (top_rays @ normal))[:, None]
            depth_edges = np.linalg.norm(np.roll(top_depth_points, -1, axis=0) - top_depth_points, axis=1)
            height = float(distance - top_distance)
            measured_center_cv = top_depth_points.mean(axis=0) + normal * height / 2
            measured_center_base = transform[:3, :3] @ measured_center_cv + transform[:3, 3]
            fixed_top = intersect_height(corners, intr, transform, table_z + edge)
            edges = np.roll(fixed_top, -1, axis=0) - fixed_top
            angles = np.arctan2(edges[:, 1], edges[:, 0])
            yaw = float(np.rad2deg(np.angle(np.mean(np.exp(4j * angles)))) / 4)
            xy = fixed_top.mean(axis=0)[:2]
            object_report = dict(top_corners_uv=corners.tolist(), size_prior_m=edge,
                independent_rgbd=dict(top_samples=int(region.sum()),
                    cube_height_m=height, top_distance_std_m=float(np.std(top_distances)),
                    cube_height_percentiles_m=np.percentile(distance - top_distances, [10, 50, 90]).tolist(),
                    top_edge_lengths_m=depth_edges.tolist(), center_base_m=measured_center_base.tolist()),
                fixed_template_projection=dict(center_xy_base_m=xy.tolist(),
                    center_z_base_m=table_z + edge / 2, top_z_base_m=table_z + edge,
                    top_edge_lengths_m=np.linalg.norm(edges, axis=1).tolist(), yaw_deg_mod_90=yaw,
                    depth_table_z_at_cube_xy_m=float((constant_base - normal_base[:2] @ xy) / normal_base[2]),
                    template_table_z_base_m=table_z))
            row["objects"][color] = object_report
            if name == "global":
                report["suggested_objects"][color] = dict(center_xy=xy.tolist(), size=[edge] * 3,
                    yaw_deg=yaw, source="global RGB corners + fixed playground top plane; size remains approximate")
        report["cameras"][name] = row
    common = set(report["cameras"].get("global", {}).get("objects", {})) & set(report["cameras"].get("wrist", {}).get("objects", {}))
    for color in common:
        g, w = [report["cameras"][v]["objects"][color] for v in ("global", "wrist")]
        delta = np.subtract(g["fixed_template_projection"]["center_xy_base_m"],
                            w["fixed_template_projection"]["center_xy_base_m"])
        report["cross_view"][color] = dict(global_minus_wrist_xy_m=delta.tolist(),
                                         discrepancy_xy_m=float(np.linalg.norm(delta)))
    report["limitations"] = [
        "Corner annotations have approximately 1–2 pixel uncertainty; sizes are estimates, not caliper measurements.",
        "Global stereo depth has object-edge/background contamination; its top median can underestimate cube height.",
        "Wrist table-plane estimates depend on the working-area ROI; fitted plane residual is not absolute depth accuracy.",
        "The existing calibrated global view defines suggested placement; wrist geometry is a cross-check, not silently fused.",
        "The 6 mm effective simulated base mount and existing wrist registration are retained, not re-estimated.",
        "Blue is clipped in the wrist view; no wrist blue size or position is inferred.",
        "Measured joints are for offline optical reprojection only; they are not an authorized reset or motion target.",
        "No geometry, lighting, camera calibration or real hardware state is changed by this analysis.",
    ]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    annotations = json.loads(args.annotations.read_text())
    report = measure(args.capture, args.config, annotations)
    report.update(annotations=identity(args.annotations), analysis_source=identity(__file__))
    (args.output / "measurements.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6), constrained_layout=True)
    for axis, name in zip(axes, ("global", "wrist")):
        axis.imshow(Image.open(args.capture / f"{name}.png"))
        for color, obj in report["cameras"][name]["objects"].items():
            uv = np.array(obj["top_corners_uv"])
            axis.plot(*np.vstack([uv, uv[0]]).T, color="yellow", marker=".", linewidth=1)
            projection = obj["fixed_template_projection"]
            xy = np.asarray(projection["center_xy_base_m"]) * 1000
            axis.text(uv[:, 0].mean(), uv[:, 1].min() - (30 if color == "blue" else 10),
                      f"{color}: XY ({xy[0]:.1f}, {xy[1]:.1f}) mm",
                      ha="center", fontsize=9, color="white", bbox=dict(facecolor="black", alpha=.6))
        axis.set_title(f"{name.capitalize()} / existing playground calibration")
        axis.set_xlabel("u (pixels)")
        axis.set_ylabel("v (pixels)")
    discrepancy = report["cross_view"]["red"]["discrepancy_xy_m"] * 1000
    fig.suptitle(f"Stationary RGB-D evidence; red cross-view XY discrepancy {discrepancy:.1f} mm\n"
                 "Suggested sizes: red 29 mm / blue 30 mm; approximate, no recalibration")
    fig.savefig(args.output / "annotated_measurements.png", dpi=120)
    plt.close(fig)
    print(json.dumps({"output": str(args.output), "suggested_objects": report["suggested_objects"],
                      "cross_view": report["cross_view"]}))


if __name__ == "__main__":
    main()
