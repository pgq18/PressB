#!/usr/bin/env python3
"""Estimate PAP camera/layout geometry from explicit image correspondences.

This is an approximate reconstruction, not device calibration. No image or
geometry is synthesized. The annotations refer to episode 0, packed video
frames 1, 31, ..., obtained by seeking the dataset's +0.03 s metadata offset.
Run after analyze_pap_dataset.py (or pass an equivalent episode NPZ).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pressb.kinematics import PiperKinematics

# Manually inspected 640 x 480 pixels. Names below are metadata-relative frame
# numbers; physical packed-video frame index is one greater (see docstring).
# Gripper points: midpoint of two tip endpoints, negative-Y and positive-Y
# endpoints of the 145 mm stock gripper crossbar. Occluded endpoints omitted.
GRIPPER_UV = {
    90: [[323, 168], [223, 78], [415, 71]],
    120: [[323, 208], [225, 121], [409, 119]],
    150: [[314, 168], [206, 66], [416, 61]],
    180: [[506, 106], [375, 32], None],
    210: [[560, 112], [440, 57], None],
    240: [[450, 77], None, None],
}
LOCAL_GRIPPER = np.array([[0, 0, .1358], [0, -.0725, .0604], [0, .0725, .0604]])
MARKER_UP = np.array([[324, 151], [143, 271], [389, 253], [269, 398]], float)
# Four green tape centers in order: near robot, image left, image right,
# near the fixed camera. Wrist rows are [metadata frame, marker ID, u, v].
MARKER_WRIST = np.array([
    [0, 2, 266, 143], [0, 0, 400.5, 396.6], [30, 2, 275.9, 185.4],
    [60, 2, 299.7, 239.9], [60, 3, 491.4, 16], [90, 2, 285, 237],
    [120, 2, 247, 199], [150, 2, 274, 355], [240, 2, 550, 413],
    [270, 2, 255.3, 240], [270, 3, 428.5, 30.8],
    [300, 2, 219.9, 202], [300, 0, 369.6, 442.7],
], float)
BOX_UV = np.array([[470, 113], [593, 122], [617, 235], [481, 235], [466, 266], [594, 272]], float)
CUBE_UV = np.array([[307, 182], [336, 182], [338, 205], [306, 207], [308, 223], [334, 224]], float)


def project(camera, points):
    rotation = Rotation.from_rotvec(camera[:3]).as_matrix()
    xyz = (points - camera[3:6]) @ rotation.T
    return xyz[:, :2] / xyz[:, 2, None] * np.exp(camera[6]) + [320, 240]


def intersect_plane(camera, pixels, z=0.):
    rotation = Rotation.from_rotvec(camera[:3]).as_matrix()
    rays = np.column_stack(((pixels - [320, 240]) / np.exp(camera[6]), np.ones(len(pixels)))) @ rotation
    depth = (z - camera[5]) / rays[:, 2]
    return camera[3:6] + depth[:, None] * rays


def metrics(error):
    error = np.asarray(error).reshape(-1, 2)
    norm = np.linalg.norm(error, axis=1)
    return {"point_count": len(error), "coordinate_rms_px": float(np.sqrt(np.mean(error ** 2))),
            "point_rms_px": float(np.sqrt(np.mean(norm ** 2))),
            "median_point_error_px": float(np.median(norm)), "max_point_error_px": float(norm.max())}


def box_points(parameters):
    cx, cy, sx, sy, height, yaw = parameters
    xyz = np.array([[-sx/2, -sy/2, height], [-sx/2, sy/2, height],
                    [sx/2, sy/2, height], [sx/2, -sy/2, height],
                    [sx/2, -sy/2, 0], [sx/2, sy/2, 0]])
    return xyz @ Rotation.from_euler("z", yaw).as_matrix().T + [cx, cy, 0]


def cube_points(parameters):
    x, y, edge, yaw = parameters
    return box_points([x, y, edge, edge, edge, yaw])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-npz", type=Path, default=ROOT / "outputs/piper_pap_reconstruction_20261022/dataset_audit/episode_000000.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/piper_pap_reconstruction_20261022/camera_fit")
    parser.add_argument("--focal-prior", type=float, default=620.)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    q = np.load(args.episode_npz)["q_rad"]
    urdf = ROOT / "vendor/robot_lab/source/robot_lab/data/Robots/Agilex/PIPER/piper_description.urdf"
    kinematics = PiperKinematics(urdf, tip_offset=0.)
    descriptors, observed = [], []
    for frame, points in GRIPPER_UV.items():
        for local_index, pixel in enumerate(points):
            if pixel is not None:
                descriptors.append((frame, local_index))
                observed.append(pixel)
    observed = np.asarray(observed, float)

    def link_pose(frame, shift):
        angles = [np.interp(frame + shift, np.arange(len(q)), q[:, j]) for j in range(6)]
        return kinematics.fk(angles)

    def geometry(shift):
        poses = {frame: link_pose(frame, shift) for frame in set(GRIPPER_UV) | set(MARKER_WRIST[:, 0])}
        points = [poses[frame][:3, :3] @ LOCAL_GRIPPER[j] + poses[frame][:3, 3] for frame, j in descriptors]
        wrist_poses = np.array([poses[frame] for frame in MARKER_WRIST[:, 0]])
        return np.asarray(points), wrist_poses

    def wrist_project(camera, markers, transforms):
        rotation = np.einsum("nij,jk->nik", transforms[:, :3, :3], Rotation.from_rotvec(camera[:3]).as_matrix())
        eye = transforms[:, :3, 3] + np.einsum("nij,j->ni", transforms[:, :3, :3], camera[3:6])
        xyz = np.einsum("nji,nj->ni", rotation, markers[MARKER_WRIST[:, 1].astype(int)] - eye)
        return xyz[:, :2] / xyz[:, 2, None] * np.exp(camera[6]) + [320, 240]

    eye = np.array([.75, 0, .45])
    forward = np.array([.3, 0, .02]) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1]); right /= np.linalg.norm(right)
    initial_up = np.r_[Rotation.from_matrix(np.array([right, np.cross(forward, right), forward])).as_rotvec(), eye, np.log(args.focal_prior)]
    points, transforms = geometry(0)
    first = least_squares(lambda v: (project(np.r_[v, initial_up[6]], points) - observed).ravel(), initial_up[:6], loss="soft_l1", f_scale=3.)
    initial_up[:6] = first.x
    registration = json.loads((ROOT / "assets/official_wrist_registration.json").read_text())
    transform = np.array(registration["camera_link_to_current_link6"])
    cv = np.eye(4); cv[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]; cv[:3, 3] = [0, .015, 0]
    transform = transform @ cv
    initial_wrist = np.r_[Rotation.from_matrix(transform[:3, :3]).as_rotvec(), transform[:3, 3], np.log(args.focal_prior)]
    initial_markers = intersect_plane(initial_up, MARKER_UP)[:, :2]
    initial = np.r_[initial_up, initial_wrist, initial_markers.ravel(), 0.,
                    [.316, .199, .15, .10, .06, -.05], [.316, .013, .03, 0.]]

    def predictions(parameters):
        points, transforms = geometry(parameters[22])
        markers = np.column_stack((parameters[14:22].reshape(-1, 2), np.zeros(4)))
        return (project(parameters[:7], points), project(parameters[:7], markers),
                wrist_project(parameters[7:14], markers, transforms))

    def errors(parameters):
        a, b, c = predictions(parameters)
        return np.r_[(a - observed).ravel(), (b - MARKER_UP).ravel(), (c - MARKER_WRIST[:, 2:]).ravel(),
                     (project(parameters[:7], box_points(parameters[23:29])) - BOX_UV).ravel(),
                     (project(parameters[:7], cube_points(parameters[29:33])) - CUBE_UV).ravel()]

    def objective(parameters, box_prior):
        # This is an explicit modelling prior, not recovered sensor intrinsics.
        prior = (parameters[[6, 13]] - np.log(args.focal_prior)) * 25.
        # User dimensions are approximate. A 15% deviation is a 3 px-equivalent
        # soft penalty; image evidence can move the dimensions.
        size_prior = (parameters[25:28] - box_prior) / (.15 * box_prior) * 3.
        cube_prior = (parameters[31] - .03) / .0045 * 3.
        return np.r_[errors(parameters), prior, size_prior, cube_prior]

    lower, upper = np.full(len(initial), -np.inf), np.full(len(initial), np.inf)
    lower[22], upper[22] = -4., 4.
    lower[[25, 26, 27, 31]] = [.07, .07, .03, .02]
    upper[[25, 26, 27, 31]] = [.22, .22, .1, .04]
    lower[[28, 32]], upper[[28, 32]] = -.5, .5
    candidates = []
    for box_prior in [np.array([.15, .10, .06]), np.array([.10, .15, .06])]:
        start = initial.copy(); start[25:28] = box_prior
        candidate = least_squares(lambda v: objective(v, box_prior), start, bounds=(lower, upper), loss="soft_l1", f_scale=3., max_nfev=1000)
        candidates.append((candidate, box_prior))
    fit, chosen_prior = min(candidates, key=lambda value: value[0].cost)
    parameters = fit.x
    up, wrist = parameters[:7], parameters[7:14]
    base_cv = np.eye(4); base_cv[:3, :3] = Rotation.from_rotvec(up[:3]).as_matrix().T; base_cv[:3, 3] = up[3:6]
    link_cv = np.eye(4); link_cv[:3, :3] = Rotation.from_rotvec(wrist[:3]).as_matrix(); link_cv[:3, 3] = wrist[3:6]
    cv_from_usd = np.diag([1., -1., -1., 1.])
    box, cube = parameters[23:29], parameters[29:33]
    predicted = predictions(parameters)
    errors_split = [predicted[0] - observed, predicted[1] - MARKER_UP, predicted[2] - MARKER_WRIST[:, 2:]]
    report = {
        "status": "approximate_image_correspondence_fit_not_device_calibration",
        "reference": "episode 0 of piper_pap_20260606_offset",
        "frame_convention": "annotation name n uses packed video frame n+1; joint index n+estimated shift",
        "estimated_joint_index_shift_relative_to_annotation_name": float(parameters[22]),
        "estimated_state_time_minus_video_time_s": float((parameters[22] - 1) / 30),
        "assumptions": ["Joint values are degrees before q_rad conversion; robot_lab URDF FK is applicable.",
                        "Stock 145 mm gripper crossbar provides metric scale; real gripper differences are a systematic uncertainty.",
                        "Table plane is base z=0. Cameras are pinhole, centered at (320,240), square pixels, no distortion.",
                        f"Both focal lengths use a soft prior {args.focal_prior:g}px, not measured device intrinsics.",
                        "User box 10x15x6 cm and cube 3 cm estimates are soft priors with 15% tolerance, not exact measurements.",
                        "Manual landmarks, potential video/state latency, real pads and mesh differences limit accuracy.",
                        "Metrics are in-sample reprojection errors, not held-out calibration accuracy."],
        "up_camera": {"resolution": [640, 480], "fx": float(np.exp(up[6])), "fy": float(np.exp(up[6])), "cx": 320., "cy": 240.,
                      "eye_base_m": up[3:6].tolist(), "target_base_m": (up[3:6] + base_cv[:3, 2] * .5).tolist(),
                      "base_from_camera_usd": (base_cv @ cv_from_usd).tolist(),
                      "horizontal_fov_deg": float(np.degrees(2 * np.arctan(320 / np.exp(up[6]))))},
        "wrist_camera": {"resolution": [640, 480], "fx": float(np.exp(wrist[6])), "fy": float(np.exp(wrist[6])), "cx": 320., "cy": 240.,
                         "link6_from_camera_usd": (link_cv @ cv_from_usd).tolist()},
        "marker_positions_base_m": np.column_stack((parameters[14:22].reshape(-1, 2), np.zeros(4))).tolist(),
        "layout_estimates": {"box_center_xy_m": box[:2].tolist(), "box_outer_size_m": box[2:5].tolist(), "box_yaw_rad": float(box[5]),
                             "box_reprojection": metrics(project(up, box_points(box)) - BOX_UV),
                             "cube_center_xy_m": cube[:2].tolist(), "cube_edge_m": float(cube[2]), "cube_yaw_rad": float(cube[3]),
                             "cube_reprojection": metrics(project(up, cube_points(cube)) - CUBE_UV),
                             "cube_top_ray_xy_given_edge": {str(edge): intersect_plane(up, np.array([[322., 194.]]), edge)[0, :2].tolist() for edge in [.025, .03, .035]}},
        "user_size_prior": {"status": "approximate_user_estimate", "box_size_xyz_m": chosen_prior.tolist(), "cube_edge_m": .03, "relative_sigma": .15,
                            "orientation_candidates": [{"box_prior_xyz_m": prior.tolist(), "cost": float(candidate.cost), "fitted_box_size_xyz_m": candidate.x[25:28].tolist()} for candidate, prior in candidates]},
        "reprojection": {"all": metrics(errors(parameters)), "up_gripper": metrics(errors_split[0]), "up_tape": metrics(errors_split[1]), "wrist_tape": metrics(errors_split[2])},
        "annotations": {"gripper_frame_local_index": descriptors, "gripper_uv": observed.tolist(), "up_tape_uv": MARKER_UP.tolist(), "wrist_frame_marker_uv": MARKER_WRIST.tolist(), "box_uv": BOX_UV.tolist(), "cube_uv": CUBE_UV.tolist()},
        "predictions": {"up_gripper_uv": predicted[0].tolist(), "up_tape_uv": predicted[1].tolist(), "wrist_tape_uv": predicted[2].tolist(), "box_uv": project(up, box_points(box)).tolist(), "cube_uv": project(up, cube_points(cube)).tolist()},
        "optimizer": {"success": bool(fit.success), "cost": float(fit.cost), "parameters": parameters.tolist()},
    }
    (args.output / "camera_fit.json").write_text(json.dumps(report, indent=2) + "\n")
    # Overlay existing reference frames only; no GPU or simulation required.
    from PIL import Image, ImageDraw
    for camera in ["up", "wrist"]:
        for frame in ([0, 90, 120, 150, 180, 210, 240] if camera == "up" else [0, 30, 60, 90, 120, 150, 240, 270, 300]):
            path = args.output / f"{camera}_{frame:03d}.jpg"
            if not path.is_file():
                continue
            im = Image.open(path).convert("RGB"); draw = ImageDraw.Draw(im)
            pairs = []
            if camera == "up" and frame == 0:
                pairs = list(zip(MARKER_UP, predicted[1])) + list(zip(BOX_UV, project(up, box_points(box)))) + list(zip(CUBE_UV, project(up, cube_points(cube))))
            elif camera == "up":
                pairs = [(observed[i], predicted[0][i]) for i, d in enumerate(descriptors) if d[0] == frame]
            else:
                pairs = [(row[2:], predicted[2][i]) for i, row in enumerate(MARKER_WRIST) if row[0] == frame]
            for measured, estimated in pairs:
                u, v = measured; a, b = estimated
                draw.ellipse((u-4, v-4, u+4, v+4), outline="lime", width=2)
                draw.line((a-5, b, a+5, b), fill="magenta", width=2); draw.line((a, b-5, a, b+5), fill="magenta", width=2)
            draw.text((8, 8), "green: manual observation | magenta: projection", fill="white", stroke_width=1, stroke_fill="black")
            im.save(args.output / f"overlay_{camera}_{frame:03d}.jpg")
    print(json.dumps({"output": str(args.output / "camera_fit.json"), "up_camera": report["up_camera"], "reprojection": report["reprojection"], "layout_estimates": report["layout_estimates"]}, indent=2))


if __name__ == "__main__":
    main()
