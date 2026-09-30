"""Seeded, single-button demonstrations and base-frame gripper pose labels.

The derived gripper TCP is link6 translated 0.1358 m along local +Z, at the
centre of the two official finger-tip joint origins. It is distinct from the
pressing-tool tip at local Z=0.24 m. Poses are xyz, quaternion wxyz, opening.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from .planning import Trajectory, base_position, make_plan, panel_face_x
from .panel_randomization import sample_panel_layout


GRIPPER_TCP_OFFSET_M = 0.1358
GRIPPER_WIDTH_M = 0.008
POSE_NAMES = ("x", "y", "z", "qw", "qx", "qy", "qz", "gripper_width")
ROOT = Path(__file__).resolve().parents[2]


@dataclass
class DatasetEpisodePlan(Trajectory):
    metadata: dict


def ee_pose_base(kin, q, width=GRIPPER_WIDTH_M):
    """Measured or commanded gripper TCP, relative to the official base_link.

    Use measured q/width for observations and desired q/width for actions.
    Quaternion w>=0 resolves the double cover for this bounded workspace.
    """
    if not np.isfinite(width) or width < 0:
        raise ValueError("Gripper width must be finite and nonnegative")
    transform = kin.link_transforms(np.asarray(q, dtype=float))["link6"]
    point = transform[:3, 3] + transform[:3, :3] @ [0., 0., GRIPPER_TCP_OFFSET_M]
    xyzw = Rotation.from_matrix(transform[:3, :3]).as_quat()
    quat = np.roll(xyzw, 1)
    if quat[0] < 0:
        quat *= -1
    return np.r_[point, quat, float(width)]


def sample_state_action(kin, plan, q_actual, gripper_actual, physics_index, stride=12):
    """Pair observation at i with desired absolute TCP at min(i+stride,last).

    Both RGB images must describe the same post-physics state as q_actual.
    At 120 Hz physics and stride=12, action is the next 100 ms endpoint, not
    the already-applied target that preceded the current observation.
    """
    if not isinstance(physics_index, (int, np.integer)) or not 0 <= physics_index < len(plan.q):
        raise ValueError("physics_index is outside this episode")
    if not isinstance(stride, (int, np.integer)) or stride <= 0:
        raise ValueError("stride must be a positive physics-step count")
    fingers = np.asarray(gripper_actual, dtype=float)
    if fingers.shape != (2,) or not np.isfinite(fingers).all():
        raise ValueError("gripper_actual must contain measured joint7 and joint8")
    state = ee_pose_base(kin, q_actual, width=float(fingers[0] - fingers[1]))
    future = min(int(physics_index) + int(stride), len(plan.q) - 1)
    action = ee_pose_base(kin, plan.q[future])
    # A target and its observation should not differ solely by quaternion sign.
    if np.dot(state[3:7], action[3:7]) < 0:
        action[3:7] *= -1
    return state, action


def _corners(bounds):
    from itertools import product
    return np.array(list(product(*np.asarray(bounds).T)), dtype=float)


@lru_cache(maxsize=32)
def _stl_bounds(path):
    """Read the original binary STL bounds without an extra mesh dependency."""
    raw = Path(path).read_bytes()
    if len(raw) < 84:
        raise ValueError(f"Invalid official STL: {path}")
    count = int.from_bytes(raw[80:84], "little")
    if len(raw) != 84 + count * 50:
        raise ValueError(f"Expected binary official STL: {path}")
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    points = np.frombuffer(raw, dtype=dtype, offset=84)["vertices"].reshape(-1, 3)
    return _corners([points.min(0), points.max(0)])


@lru_cache(maxsize=8)
def _local_geometry(urdf_path):
    urdf = ET.parse(urdf_path).getroot()
    directory = Path(urdf_path).parent / "meshes"
    result = {}
    for name in ("link2", "link3", "link4", "link5", "link6", "gripper_base", "link7", "link8"):
        points = _stl_bounds(str(directory / f"{name}.STL")).copy()
        frame = name
        if name in ("gripper_base", "link7", "link8"):
            frame = "link6"
        if name in ("link7", "link8"):
            joint = urdf.find(f"joint[@name='joint{name[-1]}']")
            origin = joint.find("origin")
            xyz = np.fromstring(origin.get("xyz"), sep=" ")
            rpy = np.fromstring(origin.get("rpy"), sep=" ")
            axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
            opening = .004 if name == "link7" else -.004
            points = Rotation.from_euler("xyz", rpy).apply(points + opening * axis) + xyz
        result[name] = (frame, points)
    proof = json.loads((ROOT / "assets/official_wrist_registration.json").read_text())
    for name, key in (("wrist_stand", "stand_bounds_current_link6_m"),
                      ("wrist_camera", "camera_bounds_current_link6_m")):
        result[name] = ("link6", _corners(proof[key]))
    return result


def _batch_frames(kin, q, base):
    count = len(q)
    transform = np.broadcast_to(np.eye(4), (count, 4, 4)).copy()
    transform[:, :3, 3] = base
    result = {}
    index = 0
    for child, (origin, axis) in zip(kin._chain_children, kin._chain):
        transform = transform @ origin
        if axis is not None:
            turn = np.broadcast_to(np.eye(4), (count, 4, 4)).copy()
            turn[:, :3, :3] = Rotation.from_rotvec(q[:, index, None] * axis).as_matrix()
            transform = transform @ turn
            index += 1
        result[child] = transform.copy()
    return result


def _fixed_camera_box(cfg):
    """Conservative box of the existing tabletop D435 including its entire foot.

    Optical-to-camera and bottom-screw offsets match the pinned Intel/AgileX
    URDF used by fixed_camera.py. This box intentionally fills stand gaps.
    """
    if cfg.get("global_camera_support_surface") != "tabletop":
        raise ValueError("Dataset safety check requires the configured tabletop camera")
    eye = np.asarray(cfg["global_camera_eye"], dtype=float)
    target = np.asarray(cfg["global_camera_target"], dtype=float)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    world_optical = np.eye(4)
    world_optical[:3, :3] = np.column_stack((right, up, -forward))
    world_optical[:3, 3] = eye
    link_optical = np.eye(4)
    link_optical[:3, :3] = [[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]]
    link_optical[:3, 3] = [0., .015, 0.]
    bottom_link = np.eye(4)
    bottom_link[:3, 3] = [.0106, .0175, .0125]
    world_bottom = world_optical @ np.linalg.inv(link_optical) @ np.linalg.inv(bottom_link)
    ball = world_bottom[:3, :3] @ [0., 0., -.050] + world_bottom[:3, 3]
    half = float(cfg["global_camera_stand_footprint_m"]) / 2
    low = np.array([ball[0] - half, ball[1] - half, cfg["table_height"]])
    high = np.array([ball[0] + half, ball[1] + half, eye[2] + .05])
    # The top head and camera fit within the stock foot's XY at current poses.
    # Enclose their separate conservative 60 mm radius if the footprint changes.
    low[:2] = np.minimum(low[:2], eye[:2] - .06)
    high[:2] = np.maximum(high[:2], eye[:2] + .06)
    return low, high


def validate_episode_plan(kin, plan, cfg):
    """Reject unsafe kinematics/static geometry before starting an episode.

    PhysX contact, successful press, release and final home remain mandatory
    runtime checks; this conservative test is not a dynamic success guarantee.
    """
    q = np.asarray(plan.q)
    face_x = panel_face_x(cfg)
    if q.ndim != 2 or q.shape[1] != 6 or len(q) < 2 or not np.isfinite(q).all():
        raise ValueError("Invalid episode joint trajectory")
    if np.any(q < kin.lower - 1e-7) or np.any(q > kin.upper + 1e-7):
        raise ValueError("Episode exceeds official joint limits")
    speed = float(np.max(np.abs(np.diff(q, axis=0))) / cfg["physics_dt"])
    if speed > cfg["joint_speed"] * 1.01:
        raise ValueError("Episode exceeds joint speed bound")
    home = np.asarray(cfg["home_q"])
    if not np.allclose(q[[0, -1]], home, atol=1e-10, rtol=0):
        raise ValueError("Episode must start and end at unchanged folded home")
    frames = _batch_frames(kin, q, base_position(cfg))
    local = dict(_local_geometry(str(kin.urdf_path.resolve())))
    local["probe"] = ("link6", _corners([[-.005, -.005, .125], [.005, .005, kin.tip_offset]]))
    camera_low, camera_high = _fixed_camera_box(cfg)
    table_edge = float(cfg["table_edge_x"])
    table_low = np.array([table_edge - cfg.get("table_depth", .75), -cfg.get("table_width", 1.) / 2])
    table_high = np.array([table_edge, cfg.get("table_width", 1.) / 2])
    clearances = {"table_m": float("inf"), "wall_or_panel_m": float("inf"), "fixed_camera_m": float("inf")}
    for name, (frame_name, points) in local.items():
        transform = frames[frame_name]
        world = np.einsum("nij,pj->npi", transform[:, :3, :3], points) + transform[:, None, :3, 3]
        low, high = world.min(1), world.max(1)
        overlap_xy = np.all((high[:, :2] >= table_low) & (low[:, :2] <= table_high), axis=1)
        if overlap_xy.any():
            distance = float(np.min(low[overlap_xy, 2] - cfg["table_height"]))
            clearances["table_m"] = min(clearances["table_m"], distance)
            if distance < .005:
                raise ValueError(f"{name} enters table clearance: {distance:.6f} m")
        if name != "probe":
            # Using the frontmost panel plane everywhere is conservative.
            distance = float(face_x + .017 - high[:, 0].max())
            clearances["wall_or_panel_m"] = min(clearances["wall_or_panel_m"], distance)
            if distance < .005:
                raise ValueError(f"{name} enters wall/panel clearance: {distance:.6f} m")
        distance = float(np.linalg.norm(np.maximum(np.maximum(camera_low - high, low - camera_high), 0), axis=1).min())
        clearances["fixed_camera_m"] = min(clearances["fixed_camera_m"], distance)
        if distance < .015:
            raise ValueError(f"{name} enters fixed-camera clearance: {distance:.6f} m")
    flange = frames["link6"]
    tip = flange[:, :3, 3] + flange[:, :3, 2] * kin.tip_offset
    error = float(np.linalg.norm(tip - plan.target_tip, axis=1).max())
    if error > .0002:
        raise ValueError(f"Planned FK disagrees with target tip: {error:.6f} m")
    transfer = np.isin(plan.phase, ["approach", "depart_home", "return_home", "settle", "home_hold"])
    front = tip[:, 0] + .005 * (1. - flange[:, 0, 2])
    if np.any(front[transfer] >= face_x - .005):
        raise ValueError("Free-space transfer enters the button clearance")
    return {"maximum_joint_speed_rad_s": speed, "maximum_fk_target_error_m": error,
            "conservative_clearances": clearances, "physics_samples_checked": len(q)}


def make_episode_plan(kin, cfg, floor, seed, *, episode_index=None):
    """Generate a reproducible new trajectory for one floor and episode seed.

    Approach standoff and timing vary; optional world-XY panel layouts
    are sampled independently and checked for all-button reachability and full
    fixed-camera coverage. Home, cameras and 8 mm gripper opening stay fixed.
    Stratified layouts require episode_index, the ordinal within this floor.
    """
    if not isinstance(floor, (int, np.integer)) or not 24 <= floor <= 35:
        raise ValueError("floor must be an integer from 24 through 35")
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not np.isclose(kin.tip_offset, cfg["tip_offset"], rtol=0, atol=1e-12):
        raise ValueError("Planner kinematics must use the physical pressing-tool offset")
    if not np.allclose(cfg.get("gripper_joint_positions_m", [.004, -.004]), [.004, -.004], atol=1e-10):
        raise ValueError("Dataset requires the unchanged 8 mm closed gripper")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(floor)]))
    episode_cfg, panel_randomization = sample_panel_layout(kin, cfg, floor, seed, episode_index=episode_index)
    variation = {
        "approach_distance": float(cfg["approach_distance"] + rng.uniform(-.003, .003)),
        "joint_speed": float(cfg["joint_speed"] * rng.uniform(.88, 1.0)),
        "press_duration": float(rng.uniform(1.30, 1.65)),
        "retract_duration": float(rng.uniform(1.30, 1.65)),
        "dwell_duration": float(rng.uniform(.32, .46)),
        "home_hold_duration": float(rng.uniform(.60, .75)),
        "panel_offset_y_m": float(episode_cfg.get("panel_offset_y_m", 0.)),
        "panel_offset_x_m": float(episode_cfg.get("panel_offset_x_m", 0.)),
    }
    episode_cfg.update(variation, sequence=[int(floor)])
    plan = make_plan(kin, episode_cfg)
    checks = validate_episode_plan(kin, plan, episode_cfg)
    metadata = {
        "floor": int(floor), "seed": int(seed), "task": f"Press {int(floor)} floor.", "variation": variation,
        "scene_randomized": panel_randomization["enabled"], "home_q": list(cfg["home_q"]),
        "panel_offset_y_m": variation["panel_offset_y_m"], "panel_randomization": panel_randomization,
        "panel_offset_x_m": variation["panel_offset_x_m"],
        "ee_frame": "derived_gripper_tip_center", "reference_frame": "base_link",
        "ee_translation_from_link6_m": [0., 0., GRIPPER_TCP_OFFSET_M],
        "pose_names": list(POSE_NAMES), "quaternion_order": "wxyz",
        "gripper_width_m": GRIPPER_WIDTH_M, "world_base_translation_m": base_position(cfg).tolist(),
        "physics_dt_s": cfg["physics_dt"], "validation": checks,
        "joint_trajectory_sha256": hashlib.sha256(np.ascontiguousarray(plan.q).tobytes()).hexdigest(),
        "success_criteria": "Runtime PhysX press + release + final folded-home, no unexpected contacts",
    }
    return DatasetEpisodePlan(plan.time, plan.q, plan.floor, plan.phase, plan.target_tip, metadata)
