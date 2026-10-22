#!/usr/bin/env python3
"""Refine PAP wrist optics using tabletop, cube and rigid fingertip landmarks.

Only the returned wrist camera differs from the frozen scene configuration.
The manually inspected image points are listed explicitly below. This is an
approximate image registration, not calibrated device intrinsics or a physical
task evaluation. Run with the ordinary pressb Python environment; no GPU.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pressb.kinematics import PiperKinematics

CV_FROM_USD = np.diag([1., -1., -1., 1.])
# These names are the original metadata-relative frame labels. Their physical
# packed video frame is label+1. The state/image index shift is estimated.
# Tape rows: frame, marker index, pixel u,v. Marker ordering follows the config.
TAPE = np.array([
    [0, 2, 266, 143], [0, 0, 400.5, 396.6], [30, 2, 275.9, 185.4],
    [60, 2, 299.7, 239.9], [60, 3, 491.4, 16], [90, 2, 285, 237],
    [120, 2, 247, 199], [150, 2, 274, 355], [240, 2, 550, 413],
    [270, 2, 255.3, 240], [270, 3, 428.5, 30.8],
    [300, 2, 219.9, 202], [300, 0, 369.6, 442.7],
], dtype=float)
# Four stationary red-cube top corners, in world local order (--),(-+),(++),(+-).
CUBE = {
    0: [[433, 229], [366, 229], [369, 173], [428, 172]],
    30: [[439, 271], [376, 272], [375, 215], [434, 214]],
    60: [[483, 355], [412, 349], [412, 287], [477, 287]],
    90: [[578, 444], [466, 450], [458, 335], [569, 330]],
}
# Stock link8 rigid-tip corners in link6 coordinates. The first two use the
# observed 55.6 mm opening, the last two use the observed 0.2 mm opening.
FINGER_XYZ = np.array([
    [-.01, .0556/2, .1358], [-.01, .0556/2+.005884, .1358],
    [-.01, .0001, .1358], [-.01, .005984, .1358],
])
FINGER_UV = np.array([[427, 351], [399, 350], [555, 351], [526, 351]], float)
# Independent initial-pose frames, not used by the optimizer. Approximate
# manually read tape centers (a few-pixel localization uncertainty).
OTHER_EPISODES = {
    10: {2: [287, 102], 0: [409, 347]},
    100: {3: [361, 73], 1: [578, 183], 2: [229, 244], 0: [357, 436]},
    400: {3: [370, 66], 1: [608, 182], 2: [230, 255], 0: [370, 470]},
    499: {3: [437, 22], 2: [284, 187], 0: [395, 394]},
}


def scores(error):
    error = np.asarray(error).reshape(-1, 2)
    norm = np.linalg.norm(error, axis=1)
    return dict(points=len(error), coordinate_rms_px=float(np.sqrt(np.mean(error**2))),
                point_rms_px=float(np.sqrt(np.mean(norm**2))),
                median_point_error_px=float(np.median(norm)), maximum_point_error_px=float(norm.max()))


def matrix(parameters):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(parameters[:3]).as_matrix()
    result[:3, 3] = parameters[3:6]
    return result


def main():
    default_run = ROOT / "outputs/piper_pap_reconstruction_20261022"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-config", type=Path, default=default_run / "final_scene/config.json")
    parser.add_argument("--dataset-audit", type=Path, default=default_run / "dataset_audit")
    parser.add_argument("--output", type=Path, default=default_run / "wrist_revision/selected_refinement")
    args = parser.parse_args()
    cfg = json.loads(args.baseline_config.read_text())
    q = np.load(args.dataset_audit / "episode_000000.npz")["q_rad"]
    kinematics = PiperKinematics(ROOT / cfg["robot_urdf"], tip_offset=0.)
    transform = np.array(cfg["wrist_camera"]["link6_from_camera_usd"]) @ CV_FROM_USD
    initial = np.r_[Rotation.from_matrix(transform[:3, :3]).as_rotvec(), transform[:3, 3],
                    np.log(cfg["wrist_camera"]["fx"]), -.1801895775302747]
    markers = np.column_stack(([marker["xy"] for marker in cfg["tape_markers"]], np.zeros(4)))
    frames, points, observed, kinds = [], [], [], []
    for frame, marker, u, v in TAPE:
        frames.append(frame); points.append(markers[int(marker)]); observed.append([u, v]); kinds.append("tape")
    edge = cfg["cube"]["edge"]
    corners = np.array([[-edge/2, -edge/2, edge], [-edge/2, edge/2, edge],
                        [edge/2, edge/2, edge], [edge/2, -edge/2, edge]])
    corners = corners @ Rotation.from_euler("z", np.deg2rad(cfg["cube"].get("yaw_deg", 0))).as_matrix().T
    corners += [*cfg["cube"]["center_xy"], 0.]
    for frame, pixels in CUBE.items():
        for point, pixel in zip(corners, pixels):
            frames.append(frame); points.append(point); observed.append(pixel); kinds.append("cube")
    frames, points, observed, kinds = np.array(frames), np.array(points), np.array(observed), np.array(kinds)
    # Exclude these observations from this refinement. This is not independent
    # calibration validation: the fixed marker geometry came from the original
    # fit, which already used these same ep0 tape observations.
    train = (kinds == "tape") | np.isin(frames, [0, 60, 90])
    train[(kinds == "tape") & np.isin(frames, [30, 270, 300])] = False

    def project(parameters):
        transforms = []
        for frame in frames:
            angles = [np.interp(frame + parameters[7], np.arange(len(q)), q[:, j]) for j in range(6)]
            transforms.append(kinematics.fk(angles))
        transforms = np.array(transforms)
        rotations = transforms[:, :3, :3] @ Rotation.from_rotvec(parameters[:3]).as_matrix()
        eyes = transforms[:, :3, 3] + np.einsum("nij,j->ni", transforms[:, :3, :3], parameters[3:6])
        xyz = np.einsum("nji,nj->ni", rotations, points-eyes)
        return xyz[:, :2] / xyz[:, 2, None] * np.exp(parameters[6]) + [320., 240.]

    def project_fingers(parameters):
        xyz = (FINGER_XYZ-parameters[3:6]) @ Rotation.from_rotvec(parameters[:3]).as_matrix()
        return xyz[:, :2] / xyz[:, 2, None] * np.exp(parameters[6]) + [320., 240.]

    def objective(parameters):
        # Open/closed rigid tips fix near-depth scale independently of the cube.
        # Modest pose/focal priors discourage large, weakly identified changes.
        return np.r_[(project(parameters)[train]-observed[train]).ravel(),
                     2.*(project_fingers(parameters)-FINGER_UV).ravel(),
                     2.*(parameters[:3]-initial[:3])/.05,
                     2.*(parameters[3:6]-initial[3:6])/.03,
                     2.*(parameters[6]-initial[6])/.15]

    fitted = least_squares(objective, initial, loss="soft_l1", f_scale=3., max_nfev=1000)
    value = fitted.x
    candidate = copy.deepcopy(cfg)
    candidate["wrist_camera"]["fx"] = candidate["wrist_camera"]["fy"] = float(np.exp(value[6]))
    candidate["wrist_camera"]["link6_from_camera_usd"] = (matrix(value) @ CV_FROM_USD).tolist()
    candidate["wrist_camera"]["status"] = "Approximate multi-depth image registration using tape, cube and open/closed fingertips; scene geometry unchanged"

    def heldout_episodes(parameters):
        errors = []
        for episode, annotation in OTHER_EPISODES.items():
            angles = np.load(args.dataset_audit / f"episode_{episode:06d}.npz")["q_rad"][0]
            camera = kinematics.fk(angles) @ matrix(parameters)
            xyz = (markers-camera[:3, 3]) @ camera[:3, :3]
            pixels = xyz[:, :2]/xyz[:, 2, None]*np.exp(parameters[6])+[320, 240]
            errors.extend(pixels[index]-pixel for index, pixel in annotation.items())
        return np.asarray(errors)

    def parameter_metrics(parameters):
        error = project(parameters)-observed
        return dict(
            tape_all=scores(error[kinds == "tape"]), cube_all=scores(error[kinds == "cube"]),
            training_scene_points=scores(error[train]), heldout_same_episode=scores(error[~train]),
            fingertip_open_closed=scores(project_fingers(parameters)-FINGER_UV),
            heldout_other_episodes=scores(heldout_episodes(parameters)),
        )
    metrics = {name: parameter_metrics(parameters) for name, parameters in [("baseline", initial), ("selected", value)]}
    common_shift = 1.
    common_metrics = {}
    for name, parameters in [("baseline", initial), ("selected", value)]:
        common_parameters = parameters.copy()
        common_parameters[7] = common_shift
        common_metrics[name] = parameter_metrics(common_parameters)
    report = {
        "status": "Selected approximate camera registration; no physical task result implied",
        "baseline_config": str(args.baseline_config.resolve()),
        "baseline_camera": cfg["wrist_camera"], "selected_camera": candidate["wrist_camera"],
        "geometry_changed": False, "physics_changed": False,
        "optimizer": dict(success=bool(fitted.success), cost=float(fitted.cost), parameters=value.tolist()),
        "state_index_shift_relative_to_metadata_frame_label": float(value[7]),
        "state_time_minus_packed_video_time_s": float((value[7]-1.)/30.),
        "metrics": metrics,
        "metrics_interpretation": "Original optimization diagnostics use each camera's fitted state-index shift; they mix camera and timing changes.",
        "common_state_index_shift_comparison": {
            "common_state_index_shift": common_shift,
            "meaning": "Both cameras use state index annotation_frame+1, matching the physical packed-video frame. This removes fitted timing differences from the before/after comparison.",
            "metrics": common_metrics,
        },
        "heldout_same_episode_scope": "Excluded only from this refinement. The fixed tabletop-marker geometry comes from the original calibration fit, which used these ep0 marker observations. This is not fully independent calibration validation.",
        "fingertip_prediction_baseline": project_fingers(initial).tolist(),
        "fingertip_prediction_selected": project_fingers(value).tolist(),
        "annotations": dict(frames=frames.tolist(), world_points=points.tolist(), pixels=observed.tolist(),
                            kinds=kinds.tolist(), training_mask=train.tolist(),
                            finger_local_xyz=FINGER_XYZ.tolist(), finger_pixels=FINGER_UV.tolist(),
                            independent_initial_episode_pixels=OTHER_EPISODES),
        "predictions": dict(baseline=project(initial).tolist(), selected=project(value).tolist()),
        "interpretation": [
            "Table-plane-only fitting did not constrain near-depth parallax adequately.",
            "Uniform zoom worsens fixed tape alignment and cannot correct depth-dependent image displacement.",
            "Selected registration balances the cube and near gripper while preserving independent-pose tape alignment.",
            "Cube dimensions and rigid stock finger geometry remain approximate; no long-finger variant is established.",
            "Manual point correspondence errors and incomplete camera calibration remain; errors are pixel reconstruction metrics.",
            "The fitted timing variable absorbs model/annotation/recording differences and is not proof of sensor synchronization.",
            "Use the common-state-index comparison to assess camera changes; the original metrics remain optimization diagnostics.",
            "Same-episode holdout excludes observations only from this refinement, not from the earlier marker-geometry estimation.",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "baseline_config.json").write_text(json.dumps(cfg, indent=2)+"\n")
    (args.output / "candidate_config.json").write_text(json.dumps(candidate, indent=2)+"\n")
    (args.output / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"output": str(args.output), "selected_camera": candidate["wrist_camera"], "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
