"""URDF-derived PiPER kinematics, in metres and radians.

The root is ``base_link``. The returned tool frame is ``link6`` translated by
``tip_offset`` along its local positive Z axis. The same offset must be used
when attaching the physical probe in the simulation.
"""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


class PiperKinematics:
    """Read the actual asset's joint origins and limits instead of assuming DH data."""

    def __init__(self, urdf_path: str | Path, tip_offset: float = 0.24):
        self.urdf_path = Path(urdf_path)
        self.tip_offset = float(tip_offset)
        if not np.isfinite(self.tip_offset) or self.tip_offset < 0:
            raise ValueError("tip_offset must be a finite nonnegative distance")
        root = ET.parse(self.urdf_path).getroot()
        by_child = {joint.find("child").get("link"): joint for joint in root.findall("joint")}
        chain = []
        child = "link6"
        while child != "base_link":
            if child not in by_child:
                raise ValueError(f"No URDF chain from base_link to link6: missing {child}")
            joint = by_child[child]
            chain.append(joint)
            child = joint.find("parent").get("link")
            if len(chain) > len(by_child):
                raise ValueError("URDF joint graph contains a cycle")
        chain.reverse()

        self.joint_names: list[str] = []
        self._chain: list[tuple[np.ndarray, np.ndarray | None]] = []
        self._chain_children: list[str] = []
        lower, upper, velocity = [], [], []
        for joint in chain:
            self._chain_children.append(joint.find("child").get("link"))
            origin = joint.find("origin")
            xyz = self._vector(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0")
            rpy = self._vector(origin.get("rpy", "0 0 0") if origin is not None else "0 0 0")
            transform = np.eye(4)
            transform[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
            transform[:3, 3] = xyz
            kind = joint.get("type")
            if kind == "fixed":
                self._chain.append((transform, None))
                continue
            if kind != "revolute":
                raise ValueError(f"Expected revolute PiPER arm joint, got {kind}")
            axis_element = joint.find("axis")
            axis = self._vector(axis_element.get("xyz", "1 0 0") if axis_element is not None else "1 0 0")
            axis /= np.linalg.norm(axis)
            limit = joint.find("limit")
            if limit is None:
                raise ValueError(f"Missing limits on joint {joint.get('name')}")
            self.joint_names.append(joint.get("name"))
            lower.append(float(limit.get("lower")))
            upper.append(float(limit.get("upper")))
            velocity.append(float(limit.get("velocity", "1")))
            self._chain.append((transform, axis))
        self.lower = np.asarray(lower)
        self.upper = np.asarray(upper)
        self.velocity_limits = np.asarray(velocity)
        if len(self.joint_names) != 6 or np.any(self.lower >= self.upper):
            raise ValueError("PiPER requires six revolute arm joints with valid limits")

    @staticmethod
    def _vector(text: str) -> np.ndarray:
        vector = np.fromstring(text, sep=" ")
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"Invalid URDF vector: {text}")
        return vector

    def fk(self, q: np.ndarray) -> np.ndarray:
        """Return the tool's 4×4 transform relative to the robot base."""
        q = np.asarray(q, dtype=float)
        if q.shape != (6,) or not np.all(np.isfinite(q)):
            raise ValueError("q must contain six finite joint angles")
        transform = np.eye(4)
        index = 0
        for origin, axis in self._chain:
            transform = transform @ origin
            if axis is not None:
                turn = np.eye(4)
                turn[:3, :3] = Rotation.from_rotvec(axis * q[index]).as_matrix()
                transform = transform @ turn
                index += 1
        transform[:3, 3] += transform[:3, :3] @ np.array([0.0, 0.0, self.tip_offset])
        return transform

    def link_transforms(self, q: np.ndarray) -> dict[str, np.ndarray]:
        """Return each arm link frame relative to ``base_link`` for geometry checks."""
        q = np.asarray(q, dtype=float)
        if q.shape != (6,) or not np.all(np.isfinite(q)):
            raise ValueError("q must contain six finite joint angles")
        result = {"base_link": np.eye(4)}
        transform = np.eye(4)
        index = 0
        for child, (origin, axis) in zip(self._chain_children, self._chain):
            transform = transform @ origin
            if axis is not None:
                turn = np.eye(4)
                turn[:3, :3] = Rotation.from_rotvec(axis * q[index]).as_matrix()
                transform = transform @ turn
                index += 1
            result[child] = transform.copy()
        return result

    def housing_tilts_deg(self, q: np.ndarray) -> np.ndarray:
        """Signed elevations of the bundled PiPER's long upper/forearm casings.

        The local axes follow the largest planar casing faces in the official
        link2/link3 STL meshes, pointing toward the elbow/wrist respectively.
        These dogleg castings differ from their joint-center lines. The mesh
        hashes, selected faces and derivation are in logs/home-housing-geometry.json.
        A positive result means that the casing rises toward its distal end.
        """
        axes = {
            "link2": np.array([.9823388128536438, -.1871108140035837, -6.6490610639940875e-6]),
            "link3": np.array([-2.463289051054295e-7, -.9999999999788073, 6.505770111053744e-6]),
        }
        frames = self.link_transforms(q)
        return np.array([
            np.degrees(np.arcsin(np.clip((frames[name][:3, :3] @ axis)[2], -1., 1.)))
            for name, axis in axes.items()
        ])

    def solve(
        self,
        position: np.ndarray,
        seed: np.ndarray,
        rotation: np.ndarray | None = None,
    ) -> np.ndarray:
        """Solve bounded IK, preferring the branch near ``seed``.

        ``position`` is a base-frame XYZ target. ``rotation``, when supplied,
        is the desired 3×3 tool orientation in that frame. Unreachable targets
        raise ``ValueError``; failed solver results are never passed to a robot.
        """
        position = np.asarray(position, dtype=float)
        seed = np.asarray(seed, dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("position must contain three finite coordinates")
        if seed.shape != (6,) or not np.all(np.isfinite(seed)):
            raise ValueError("seed must contain six finite joint angles")
        if rotation is not None:
            rotation = np.asarray(rotation, dtype=float)
            if (
                rotation.shape != (3, 3)
                or not np.all(np.isfinite(rotation))
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)
            ):
                raise ValueError("rotation must be a proper 3x3 rotation matrix")
        seed = np.clip(seed, self.lower + 1e-9, self.upper - 1e-9)

        def residual(q):
            pose = self.fk(q)
            terms = [pose[:3, 3] - position]
            if rotation is not None:
                error = Rotation.from_matrix(rotation @ pose[:3, :3].T).as_rotvec()
                terms.append(0.15 * error)
            # A tiny regularizer resolves redundant position-only IK smoothly.
            terms.append(1e-6 * (q - seed))
            return np.concatenate(terms)

        alternatives = [seed, (self.lower + self.upper) / 2]
        for shoulder, elbow in [(0.6, -0.9), (1.2, -1.8), (2.2, -2.0), (2.8, -0.6)]:
            candidate = seed.copy()
            candidate[1:3] = [shoulder, elbow]
            alternatives.append(np.clip(candidate, self.lower + 1e-9, self.upper - 1e-9))
        best_error = float("inf")
        best_orientation_error = float("inf")
        for start in alternatives:
            result = least_squares(
                residual,
                start,
                bounds=(self.lower, self.upper),
                ftol=1e-11,
                xtol=1e-11,
                gtol=1e-11,
                max_nfev=250,
            )
            pose = self.fk(result.x)
            distance = float(np.linalg.norm(pose[:3, 3] - position))
            angle = 0.0 if rotation is None else float(
                Rotation.from_matrix(rotation @ pose[:3, :3].T).magnitude()
            )
            if distance < 5e-4 and angle < 1e-2:
                return result.x
            if distance < best_error:
                best_error, best_orientation_error = distance, angle
        raise ValueError(
            f"Unreachable PiPER tool target {position.tolist()}: "
            f"best position error {best_error:.6f} m, "
            f"orientation error {best_orientation_error:.6f} rad"
        )
