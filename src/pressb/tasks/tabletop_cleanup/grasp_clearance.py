"""Conservative finger sweep clearance for geometry-aligned cleanup grasps.

The calculation uses the actual URDF finger meshes and the task's separate
convex collision parts. It is an offline geometric check, not a whole-arm
collision checker or a physical contact/force guarantee.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from .geometry import asset_geometry


def _finite(value, shape, name):
    value = np.asarray(value, dtype=float)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{name} must have shape {shape} and finite values")
    return value


def _vector(text):
    return _finite(np.fromstring(text, sep=" "), (3,), "URDF vector")


@lru_cache(maxsize=4)
def _finger_geometry(urdf_path):
    """Return STL vertices in the gripper-base frame at zero opening."""
    path = Path(urdf_path)
    tree = ET.parse(path).getroot()
    result = []
    for name in ("joint7", "joint8"):
        joint = tree.find(f"joint[@name='{name}']")
        link = tree.find(f"link[@name='{joint.find('child').get('link')}']")
        origin = joint.find("origin")
        rotation = Rotation.from_euler("xyz", _vector(origin.get("rpy", "0 0 0"))).as_matrix()
        translation = _vector(origin.get("xyz", "0 0 0"))
        axis = rotation @ _vector(joint.find("axis").get("xyz"))
        collision = link.find("collision")
        local = collision.find("origin")
        local_r = Rotation.from_euler("xyz", _vector(local.get("rpy", "0 0 0"))).as_matrix()
        local_t = _vector(local.get("xyz", "0 0 0"))
        mesh = collision.find("geometry/mesh")
        filename = path.parent / mesh.get("filename")
        raw = filename.read_bytes()
        count = int.from_bytes(raw[80:84], "little")
        if len(raw) != 84+50*count or count == 0:
            raise ValueError("Expected a complete binary STL finger mesh")
        dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
        vertices = np.frombuffer(raw, dtype=dtype, offset=84)["vertices"].reshape(-1, 3).astype(float)
        vertices *= _vector(mesh.get("scale", "1 1 1"))
        vertices = (vertices @ local_r.T+local_t) @ rotation.T+translation
        limit = joint.find("limit")
        result.append((name, vertices, axis, float(limit.get("lower")), float(limit.get("upper"))))
    return tuple(result)


def _halfspaces(points):
    # Millimetres keep LP tolerances well below the clearance being measured.
    equations = ConvexHull(np.unique(np.round(points*1000., 9), axis=0)).equations
    equations = np.unique(np.round(equations, 10), axis=0)
    return equations[:, :3], -equations[:, 3]


def sweep_clearance(kind, *, urdf_path, opening_m, tcp_local_m, rotation_local,
                    approach_direction_local=(0., 0., 1.), approach_distance_m=.12,
                    scale=1., tip_offset_m=.1358, max_opening_m=None):
    """First inward finger translation to touch each convex object piece.

    Poses are expressed in the object's local frame. Each finger is enclosed
    by its STL convex hull swept over the *continuous* straight descent, not
    isolated waypoints. Positive results are free lateral travel; zero means
    overlap. A convex finger envelope can underestimate the real clearance.
    max_opening_m applies the same travel override as the Isaac scene.
    Omission uses the original URDF limits.
    """
    tcp = _finite(tcp_local_m, (3,), "tcp_local_m")
    rotation = _finite(rotation_local, (3, 3), "rotation_local")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7) or not np.isclose(np.linalg.det(rotation), 1., atol=1e-7):
        raise ValueError("rotation_local must be a proper rotation")
    direction = _finite(approach_direction_local, (3,), "approach_direction_local")
    if not np.isclose(np.linalg.norm(direction), 1.):
        raise ValueError("approach direction must be a unit vector")
    if not np.isfinite([opening_m, approach_distance_m, tip_offset_m]).all() or opening_m <= 0 or approach_distance_m <= 0:
        raise ValueError("Opening and sweep length must be finite and positive")
    if max_opening_m is not None and (not np.isfinite(max_opening_m) or max_opening_m <= 0):
        raise ValueError("max_opening_m must be finite and positive")
    parts, _ = asset_geometry(kind, scale)
    hulls = [(part.name, *_halfspaces(np.asarray(part.vertices))) for part in parts if part.collision]
    results = {}
    for name, vertices, axis, lower, upper in _finger_geometry(str(Path(urdf_path).resolve())):
        if max_opening_m is not None:
            if name == "joint7":
                upper = max_opening_m/2
            else:
                lower = -max_opening_m/2
        q = opening_m/2 if name == "joint7" else -opening_m/2
        if q < lower-1e-10 or q > upper+1e-10:
            raise ValueError("Opening exceeds configured finger travel")
        points = (vertices+axis*q-np.array([0., 0., tip_offset_m])) @ rotation.T+tcp
        swept = np.concatenate([points, points+direction*approach_distance_m])
        a_finger, b_finger = _halfspaces(swept)
        # joint7 occupies -tool-Y; joint8 +tool-Y. Move towards the cavity.
        inward = rotation[:, 1] * (1. if name == "joint7" else -1.)
        side = "negative" if name == "joint7" else "positive"
        closest = None
        for part_name, a_object, b_object in hulls:
            a = np.vstack([np.c_[a_object, np.zeros(len(a_object))],
                           np.c_[a_finger, -a_finger @ inward]])
            b = np.r_[b_object, b_finger]
            solution = linprog([0., 0., 0., 1.], A_ub=a, b_ub=b,
                               bounds=[(None, None)]*3+[(0., 200.)], method="highs")
            if solution.status == 2:  # Swept X/Z silhouettes never intersect.
                continue
            if not solution.success:
                raise ValueError(f"Finger clearance optimization failed: {solution.message}")
            distance = max(0., float(solution.x[3])/1000.)
            if closest is None or distance < closest["clearance_m"]:
                closest = dict(clearance_m=distance, part=part_name, finger=name)
        if closest is None:
            raise ValueError(f"No graspable geometry inside the {side} finger sweep")
        results[side] = closest
    return results


def optimize_grasp_center(kind, *, urdf_path, opening_m, scale=1.,
                          contact_to_tip_m=.004, approach_distance_m=.12,
                          max_center_shift_m=.006, max_opening_m=None):
    """Balance the two complete finger sweeps along the local closing axis."""
    if not np.isfinite([contact_to_tip_m, max_center_shift_m]).all() or contact_to_tip_m <= 0 or max_center_shift_m < 0:
        raise ValueError("Contact offset and centering bound must be finite and valid")
    _, spec = asset_geometry(kind, scale)
    candidate = spec["grasp_candidates"][0]
    axis = np.asarray(candidate["closing_axis_local"], dtype=float)
    angle = np.arctan2(axis[1], axis[0])+np.pi/2
    rotation = Rotation.from_euler("z", angle).as_matrix() @ np.diag([1., -1., -1.])
    center = np.asarray(candidate["center_local_m"], dtype=float)
    tcp = center-np.array([0., 0., contact_to_tip_m])
    kwargs = dict(urdf_path=urdf_path, opening_m=opening_m, rotation_local=rotation,
                  max_opening_m=max_opening_m,
                  approach_distance_m=approach_distance_m, scale=scale)
    before = sweep_clearance(kind, tcp_local_m=tcp, **kwargs)
    negative, positive = [before[side]["clearance_m"] for side in ("negative", "positive")]
    if min(negative, positive) <= 0:
        raise ValueError("Nominal object-aligned finger sweep overlaps the object; cannot balance a collision")
    shift = float(np.clip((negative-positive)/2, -max_center_shift_m, max_center_shift_m))
    tcp = tcp+rotation[:, 1]*shift
    center = center+rotation[:, 1]*shift
    after = sweep_clearance(kind, tcp_local_m=tcp, **kwargs)
    if min(item["clearance_m"] for item in after.values())+1e-8 < min(negative, positive):
        raise ValueError("Centering reduced the minimum finger sweep clearance")
    return dict(center_local_m=center.tolist(), tcp_local_m=tcp.tolist(),
                rotation_local=rotation.tolist(), center_shift_m=shift,
                before_clearance_m={s: v["clearance_m"] for s, v in before.items()},
                after_clearance_m={s: v["clearance_m"] for s, v in after.items()},
                limiting_parts={s: v["part"] for s, v in after.items()},
                opening_m=float(opening_m), approach_distance_m=float(approach_distance_m),
                method="Continuous STL convex finger sweeps against individual convex object parts; lateral first-contact LP",
                scope="Geometric finger sweep only; excludes whole-arm, table, other-object and bin collisions and contact offsets")
