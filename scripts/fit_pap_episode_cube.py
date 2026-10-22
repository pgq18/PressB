#!/usr/bin/env python3
"""Estimate a cube's initial pose and ONE edge length with frozen PAP cameras.

Image annotations specify six visible corners per camera. The cube constraint
is exact; the user's approximate 30 mm edge is only a weak size prior. This
utility does not inspect physical replay outcomes or adjust camera parameters.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pressb.kinematics import PiperKinematics

# Episode 461, frame zero: four top corners (--),(-+),(++),(+-), then
# two bottom corners on the visible front face of each respective camera.
OBSERVATIONS = {
    "up": [[238, 269.5], [271, 269.5], [269.5, 298.5], [234, 298], [236, 312], [270.5, 312.5]],
    "wrist": [[460.5, 96], [407, 101.5], [400.5, 61.5], [454, 56.5], [459.5, 128], [405.5, 133]],
}


def scores(error):
    error = np.asarray(error)
    return dict(coordinate_rms_px=float(np.sqrt(np.mean(error**2))),
                point_rms_px=float(np.sqrt(np.mean(np.sum(error**2, axis=1)))),
                max_point_error_px=float(np.linalg.norm(error, axis=1).max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=Path, help="Optional audited initial RGB directory")
    parser.add_argument("--annotations", type=Path,
                        help="JSON with up/wrist corner arrays; defaults to episode 461 frame zero")
    args = parser.parse_args()
    cfg = json.loads(args.source_config.read_text())
    episode = int(cfg["episode_source"]["episode_index"])
    if not args.annotations and episode != 461:
        parser.error("Supply --annotations for episodes other than 461")
    if args.trajectory.stem != f"episode_{episode:06d}":
        parser.error("Trajectory must match the configured episode")
    args.output.mkdir(parents=True, exist_ok=False)
    obs = json.loads(args.annotations.read_text()) if args.annotations else OBSERVATIONS
    obs = {view: np.asarray(obs[view], dtype=float) for view in ("up", "wrist")}
    if any(v.shape != (6, 2) or not np.isfinite(v).all() for v in obs.values()):
        parser.error("Each camera requires six finite (u, v) corner observations")
    with np.load(args.trajectory) as data:
        q = data["q_rad"][0]
    kinematics = PiperKinematics(ROOT / cfg["robot_urdf"], 0)
    cv_from_usd = np.diag([1., -1., -1., 1.])
    cameras = {
        "up": np.array(cfg["up_camera"]["base_from_camera_usd"]) @ cv_from_usd,
        "wrist": kinematics.fk(q) @ np.array(cfg["wrist_camera"]["link6_from_camera_usd"]) @ cv_from_usd,
    }

    def project(parameters, view):
        x, y, edge, yaw = parameters
        corners = np.array([[-.5, -.5, 1], [-.5, .5, 1], [.5, .5, 1], [.5, -.5, 1],
                            [.5, -.5, 0], [.5, .5, 0]]) * edge
        if view == "wrist":
            corners[4:] = np.array([[-.5, -.5, 0], [-.5, .5, 0]]) * edge
        points = corners @ Rotation.from_euler("z", yaw).as_matrix().T + [x, y, 0]
        transform = cameras[view]
        local = (points - transform[:3, 3]) @ transform[:3, :3]
        optical = cfg[f"{view}_camera"]
        return local[:, :2] / local[:, 2, None] * [optical["fx"], optical["fy"]] + [optical["cx"], optical["cy"]]

    original = cfg["object"]
    start = [*original["center_xy"], .03, np.deg2rad(original.get("yaw_deg", 0))]
    results = {}
    for name, wrist_weight in [("up_only", 0.), ("up_primary_wrist_soft", .25), ("both_views_equal", 1.)]:
        def residual(parameters):
            return np.r_[(project(parameters, "up") - obs["up"]).ravel(),
                         wrist_weight * (project(parameters, "wrist") - obs["wrist"]).ravel(),
                         (parameters[2] - .03) / .009 * .5]
        fit = least_squares(residual, start, bounds=([.2, -.2, .015, -.5], [.6, .15, .05, .5]),
                            loss="soft_l1", f_scale=2., max_nfev=1000)
        results[name] = dict(parameters_xy_edge_yaw=fit.x.tolist(),
                             metrics={v: scores(project(fit.x, v)-obs[v]) for v in obs},
                             predictions={v: project(fit.x, v).tolist() for v in obs})
        print(name, json.dumps(results[name]))
    selected = "up_primary_wrist_soft"
    p = results[selected]["parameters_xy_edge_yaw"]
    candidate = copy.deepcopy(cfg)
    obj = candidate["object"]
    obj.pop("size", None)
    obj.pop("size_xyz", None)
    obj.update(shape="cube", edge=p[2], center_xy=p[:2], yaw_deg=float(np.rad2deg(p[3])),
               grasp_width_m=p[2],
               geometry_status="Exact cube constraint from user; single edge and pose estimated from frozen cameras with up:wrist residual weights 1:0.25; not measured dimensions")
    report = dict(episode_index=episode, source_config=str(args.source_config),
                  source_trajectory=str(args.trajectory), annotations={v: a.tolist() for v, a in obs.items()},
                  annotation_uncertainty_px="up/top corners approximately 1–2; shadow-blended wrist bottom approximately 3",
                  annotation_reference=str(args.annotations) if args.annotations else "OBSERVATIONS in this script; episode 461 frame zero",
                  shape_constraint="Three equal edges; only one edge length is optimized",
                  size_prior_m=.03, size_prior_status="User approximate estimate, not a measurement",
                  selected=selected, selection_basis="Retain prior up-primary weighting; enforce cube independently of replay outcome",
                  results=results, cameras_reestimated=False, fitted_from_simulation_success=False,
                  validation_scope="Selected camera residuals are in-sample; equal-view and up-only fits are sensitivity diagnostics",
                  original_object=original, revised_object=obj)
    candidate["geometry_fit_reference"] = str(args.output / "fit_report.json")
    (args.output / "candidate_cube.json").write_text(json.dumps(candidate, indent=2) + "\n")
    (args.output / "fit_report.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.frames:
        for view in obs:
            image = Image.open(args.frames / f"episode_{episode:06d}_0000_initial_{view}.jpg").convert("RGB")
            draw = ImageDraw.Draw(image)
            for (u, v), (a, b) in zip(obs[view], project(p, view)):
                draw.ellipse((u-3, v-3, u+3, v+3), outline="lime", width=1)
                draw.line((a-4, b, a+4, b), fill="magenta", width=1)
                draw.line((a, b-4, a, b+4), fill="magenta", width=1)
            draw.text((8, 8), "Exact cube: green=annotation, magenta=fit", fill="white", stroke_width=1, stroke_fill="black")
            image.save(args.output / f"overlay_{view}.jpg", quality=95)


if __name__ == "__main__":
    main()
