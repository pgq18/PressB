"""CPU collision screening for sampled PiPER joint paths.

The robot's URDF collision meshes become hpp-fcl convex hulls, matching the
kind of collision approximation used by the robot USD. The official wrist
housing is a box; its stand is conservatively covered by one convex hull
(PhysX uses convex decomposition there). The finite tabletop is checked in
base coordinates, as are nonadjacent robot bodies. Same-body and directly
adjacent joint groups are excluded; the two finger siblings are not excluded.

This is discrete geometry screening, not a continuous collision planner or a
PhysX-equivalence claim. Contact offsets, compliance, servo tracking, room,
clamps and tripod geometry are not simulated. Validate accepted paths in Isaac.
No SimulationApp, GPU or USD import is needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[3]


def _vector(text, length=3):
    value = np.fromstring(text, sep=" ")
    if value.shape != (length,) or not np.isfinite(value).all():
        raise ValueError(f"Expected {length} finite URDF values: {text}")
    return value


def _origin(element):
    origin = element.find("origin")
    result = np.eye(4)
    if origin is not None:
        result[:3, :3] = Rotation.from_euler("xyz", _vector(origin.get("rpy", "0 0 0"))).as_matrix()
        result[:3, 3] = _vector(origin.get("xyz", "0 0 0"))
    return result


@dataclass(frozen=True)
class _Geometry:
    name: str
    link: str
    joint: int
    local_pose: np.ndarray
    shape: object


@lru_cache(maxsize=4)
def _load_robot(urdf_path):
    import pinocchio as pin

    model = pin.buildModelFromUrdf(urdf_path)
    geometry = pin.buildGeomFromUrdf(model, urdf_path, pin.GeometryType.COLLISION,
                                    package_dirs=[str(Path(urdf_path).parent)])
    shapes = []
    for item in geometry.geometryObjects:
        mesh = item.geometry
        if not hasattr(mesh, "buildConvexHull"):
            raise ValueError(f"Expected the Piper URDF collision mesh for {item.name}")
        mesh.buildConvexHull(True, "Qt")
        frame = model.frames[item.parentFrame]
        # Both Pinocchio placements are relative to parentJoint. Convert the
        # collider back to its own link frame for PiperKinematics transforms.
        local = (frame.placement.inverse() * item.placement).homogeneous.copy()
        shapes.append(_Geometry(item.name, frame.name, item.parentJoint, local, mesh.convex))
    return model, tuple(shapes)


@lru_cache(maxsize=1)
def _load_wrist():
    """Use the same source URDF/registration as attach_wrist_camera, not optics."""
    import hppfcl

    asset_root = ROOT / "vendor/piper_isaac_sim"
    source = ET.parse(asset_root / "piper_description/urdf/piper_description_v100_realsense_camera_v2.urdf").getroot()
    joints = {joint.get("name"): joint for joint in source.findall("joint")}
    links = {link.get("name"): link for link in source.findall("link")}
    adapter = np.eye(4)
    adapter[:3, :3] = Rotation.from_euler("z", np.pi / 2).as_matrix()
    adapter[2, 3] = -.004
    stand_pose = adapter @ _origin(joints["camera_stand_joint"])
    camera_pose = (adapter @ _origin(joints["d435_camera_joint"])
                   @ _origin(joints["camera_joint"]) @ _origin(joints["camera_link_joint"]))
    stand = links["camera_stand_link"].find("collision")
    mesh_spec = stand.find("geometry/mesh")
    filename = mesh_spec.get("filename")
    if not filename.startswith("package://"):
        raise ValueError("Expected official stand package mesh")
    mesh = hppfcl.MeshLoader().load(str(asset_root / filename.removeprefix("package://")),
                                  _vector(mesh_spec.get("scale", "1 1 1")))
    mesh.buildConvexHull(True, "Qt")
    camera = links["camera_link"].find("collision")
    size = _vector(camera.find("geometry/box").get("size"))
    return (("wrist_stand", stand_pose @ _origin(stand), mesh.convex),
            ("wrist_camera", camera_pose @ _origin(camera), hppfcl.Box(*size)))


class RobotClearance:
    """Screen finite-table and nonadjacent-self collisions using real assets.

    ``check(q6, gripper_width)`` returns ``(clear, diagnostic)``. Width is the
    full opening in metres, with joint7=+width/2 and joint8=-width/2. The task's
    explicit 100 mm gripper overlay is honored; values are never clipped.
    ``check_path`` checks the supplied samples only, without interpolation.
    """

    def __init__(self, kin, cfg, *, include_wrist=True):
        import hppfcl

        self.kin = kin
        self.cfg = cfg
        self._fcl = hppfcl
        self.model, robot_shapes = _load_robot(str(kin.urdf_path.resolve()))
        self.geometries = list(robot_shapes)
        self.parents = tuple(int(value) for value in self.model.parents)
        source = ET.parse(kin.urdf_path).getroot()
        self._extra_joints = []
        arm_links = {"base_link", *kin.link_transforms(np.zeros(6))}
        for joint in source.findall("joint"):
            child = joint.find("child").get("link")
            if child in arm_links:
                continue
            parent = joint.find("parent").get("link")
            name, kind = joint.get("name"), joint.get("type")
            if kind == "fixed":
                self._extra_joints.append((parent, child, _origin(joint), None, 0.))
            elif kind == "prismatic" and name in ("joint7", "joint8"):
                axis = _vector(joint.find("axis").get("xyz"))
                axis /= np.linalg.norm(axis)
                self._extra_joints.append((parent, child, _origin(joint), axis, .5 if name == "joint7" else -.5))
            else:
                raise ValueError(f"Unsupported additional robot joint {name}: {kind}")
        finger_limits = {joint.get("name"): joint.find("limit") for joint in source.findall("joint")
                         if joint.get("name") in ("joint7", "joint8")}
        urdf_width = 2 * min(float(finger_limits["joint7"].get("upper")),
                             -float(finger_limits["joint8"].get("lower")))
        self.max_width = float(cfg.get("gripper_max_width_m", urdf_width))
        if not np.isfinite(self.max_width) or self.max_width <= 0:
            raise ValueError("gripper_max_width_m must be positive and finite")
        if include_wrist:
            wrist_joint = int(self.model.getJointId("joint6"))
            self.geometries.extend(_Geometry(name, "link6", wrist_joint, pose, shape)
                                   for name, pose, shape in _load_wrist())
        table = cfg.get("table", {})
        depth = float(cfg.get("table_depth", table.get("depth", 1.0)))
        width = float(cfg.get("table_width", table.get("width", 1.6)))
        x_min = float(cfg.get("table_x_min", table.get("x_min", -.12)))
        if not np.isfinite([depth, width, x_min]).all() or min(depth, width) <= 0:
            raise ValueError("Invalid tabletop dimensions")
        # Same 36 mm slab as pick_place.scene. robot_base_xy and absolute
        # table_height cancel because all collision work is in base_link.
        self.table_shape = hppfcl.Box(depth, width, .036)
        self.table_pose = hppfcl.Transform3f(np.eye(3), np.array([x_min + depth / 2, 0., -.018]))
        self.table_pairs = tuple(i for i, item in enumerate(self.geometries) if item.joint != 0)
        self.self_pairs = tuple((i, j) for i, a in enumerate(self.geometries)
                                for j, b in enumerate(self.geometries[:i])
                                if a.joint != b.joint and self.parents[a.joint] != b.joint
                                and self.parents[b.joint] != a.joint)
        self._request = hppfcl.CollisionRequest()
        self._request.enable_contact = True
        self._request.num_max_contacts = 1
        self._distance_request = hppfcl.DistanceRequest()
        self._local_bounds = [self._bounds(item.shape) for item in self.geometries]
        self.model_info = dict(
            model="URDF convex hulls, official wrist box/stand hull, finite tabletop",
            robot_meshes=len(robot_shapes), wrist_included=bool(include_wrist),
            table_pairs=len(self.table_pairs), self_pairs=len(self.self_pairs),
            table_size_m=[depth, width, .036], table_center_base_m=[x_min+depth/2, 0., -.018],
            exclusions=["fixed base/table", "same rigid body", "directly adjacent joint groups"],
            limitations=["stand hull conservatively covers PhysX convex decomposition",
                         "sampled configurations only; no continuous swept-volume check",
                         "no contact offsets, dynamics, tracking error, room, clamps or tripod"],
        )
        self.link_transforms(np.zeros(6), 0.)  # Detect incomplete URDF graphs now.

    def link_transforms(self, q, gripper_width=0.):
        width = float(gripper_width)
        if not np.isfinite(width) or not 0 <= width <= self.max_width:
            raise ValueError(f"Full gripper width must be in [0, {self.max_width}] metres")
        frames = self.kin.link_transforms(q)
        pending = list(self._extra_joints)
        while pending:
            remaining = []
            for parent, child, origin, axis, multiplier in pending:
                if parent not in frames:
                    remaining.append((parent, child, origin, axis, multiplier))
                    continue
                frame = frames[parent] @ origin
                if axis is not None:
                    shift = np.eye(4)
                    shift[:3, 3] = axis * width * multiplier
                    frame = frame @ shift
                frames[child] = frame
            if len(remaining) == len(pending):
                raise ValueError("Unresolved additional URDF collision-link transforms")
            pending = remaining
        missing = {item.link for item in self.geometries} - set(frames)
        if missing:
            raise ValueError(f"Missing collision-link transforms: {sorted(missing)}")
        return frames

    def _poses(self, q, width):
        frames = self.link_transforms(q, width)
        poses = []
        for item in self.geometries:
            transform = frames[item.link] @ item.local_pose
            poses.append(self._fcl.Transform3f(transform[:3, :3], transform[:3, 3]))
        return poses

    def _collision(self, first, first_pose, second, second_pose):
        result = self._fcl.CollisionResult()
        self._fcl.collide(first, first_pose, second, second_pose, self._request, result)
        if not result.isCollision():
            return None
        contact = result.getContact(0)
        return dict(penetration_depth_m=float(contact.penetration_depth),
                    contact_position_base_m=np.asarray(contact.pos).tolist())

    @staticmethod
    def _bounds(shape):
        shape.computeLocalAABB()
        return np.array([shape.aabb_local.min_, shape.aabb_local.max_])

    @staticmethod
    def _world_bounds(bounds, pose):
        center = pose.transform((bounds[0] + bounds[1]) / 2)
        extent = np.abs(pose.getRotation()) @ ((bounds[1] - bounds[0]) / 2)
        return np.array([center - extent, center + extent])

    def check(self, q, gripper_width=0.):
        """Return the first detected collision, or a clear-state diagnostic."""
        q = np.asarray(q, dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError("q must contain six finite arm joint angles")
        if np.any(q < self.kin.lower - 1e-9) or np.any(q > self.kin.upper + 1e-9):
            return False, dict(reason="joint_limits")
        poses = self._poses(q, gripper_width)
        minimum = float("inf")
        near = []
        for index in self.table_pairs:
            item = self.geometries[index]
            hit = self._collision(item.shape, poses[index], self.table_shape, self.table_pose)
            if hit is not None:
                return False, dict(reason="table_collision", pair=[item.name, "table"], **hit)
            result = self._fcl.DistanceResult()
            distance = float(self._fcl.distance(item.shape, poses[index], self.table_shape,
                                               self.table_pose, self._distance_request, result))
            minimum = min(minimum, distance)
            if distance <= .001:
                near.append(item.name)
        for i, j in self.self_pairs:
            a, b = self.geometries[i], self.geometries[j]
            hit = self._collision(a.shape, poses[i], b.shape, poses[j])
            if hit is not None:
                return False, dict(reason="self_collision", pair=[a.name, b.name], **hit)
        return True, dict(reason="clear", min_table_distance_m=minimum, near_table_links=near)

    @staticmethod
    def _path_inputs(q_sequence, gripper_width):
        path = np.asarray(q_sequence, dtype=float)
        if path.ndim != 2 or path.shape[1] != 6 or not len(path) or not np.isfinite(path).all():
            raise ValueError("Joint path must be a nonempty finite Nx6 array")
        widths = np.asarray(gripper_width, dtype=float)
        if widths.ndim == 0:
            widths = np.full(len(path), float(widths))
        if widths.shape != (len(path),) or not np.isfinite(widths).all():
            raise ValueError("gripper_width must be one finite width or one per path sample")
        return path, widths

    def check_path(self, q_sequence, gripper_width=0.):
        """Check every supplied sample; this does not infer between-sample motion."""
        path, widths = self._path_inputs(q_sequence, gripper_width)
        minimum = float("inf")
        near_samples = []
        for index, (q, width) in enumerate(zip(path, widths)):
            clear, diagnostic = self.check(q, width)
            if not clear:
                return False, dict(diagnostic, sample_index=index, checked_samples=index+1)
            minimum = min(minimum, diagnostic["min_table_distance_m"])
            if diagnostic["near_table_links"]:
                near_samples.append(index)
        return True, dict(reason="clear", checked_samples=len(path),
                          min_table_distance_m=minimum, near_table_sample_indices=near_samples)

    def _button_shapes(self, positions):
        """Reproduce scene.py's lower disks, 24-strip bezels and neutral caps."""
        colors = ("red", "yellow", "blue")
        if not positions or set(positions) - set(colors):
            raise ValueError("Provide a nonempty subset of red/yellow/blue button positions")
        cfg = self.cfg["button_geometry"]
        radius, height = float(cfg["cap_radius_m"]), float(cfg["cap_height_m"])
        outer, base_height = float(cfg["base_radius_m"]), float(cfg["base_height_m"])
        travel = float(cfg["travel_m"])
        if not np.isfinite([radius, height, outer, base_height, travel]).all() or min(radius, height, travel) <= 0:
            raise ValueError("Invalid button collision dimensions")
        inner = radius + .0015
        if outer <= inner + .0015 or base_height <= 2 * travel or height <= travel:
            raise ValueError("Button housing must leave room for the cap travel")
        lower_height = base_height - 2 * travel
        ring_radius = (inner + outer) / 2
        pieces = []

        def add(color, kind, shape, position, yaw=0.):
            rotation = Rotation.from_euler("z", yaw).as_matrix()
            pose = self._fcl.Transform3f(rotation, np.asarray(position))
            pieces.append(dict(color=color, kind=kind, shape=shape, pose=pose,
                               bounds=self._world_bounds(self._bounds(shape), pose)))

        for color in colors:
            if color not in positions:
                continue
            xy = positions[color]
            if isinstance(xy, dict):
                xy = xy["center_xy"]
            xy = np.asarray(xy, dtype=float)
            if xy.shape != (2,) or not np.isfinite(xy).all():
                raise ValueError(f"Invalid {color} button XY")
            x, y = xy
            add(color, "housing", self._fcl.Cylinder(outer, lower_height), [x, y, lower_height / 2])
            for index in range(24):
                angle = index * 2 * np.pi / 24
                box = self._fcl.Box(outer - inner, 2 * ring_radius * np.tan(np.pi / 24), 2 * travel)
                add(color, f"bezel_{index}", box,
                    [x + ring_radius * np.cos(angle), y + ring_radius * np.sin(angle), lower_height + travel], angle)
            add(color, "cap", self._fcl.Cylinder(radius, height), [x, y, base_height + height / 2])
        return pieces

    def check_button_path(self, q_sequence, positions, target_color, gripper_width=0.):
        """Check robot/table/self plus all buttons along a sampled joint path.

        Only the target cap may touch link7/link8; its housing/bezel and every
        other cap remain obstacles for every robot part. Caps stay at neutral
        height in this geometric screen; the target's actual compression is
        left to PhysX. Stand-convex-hull conservatism and discrete-path limits
        are the same as for ``check_path``.
        """
        if target_color not in ("red", "yellow", "blue"):
            raise ValueError("target_color must be red, yellow or blue")
        if target_color not in positions:
            raise ValueError("Button positions must include the target color")
        path, widths = self._path_inputs(q_sequence, gripper_width)
        pieces = self._button_shapes(positions)
        obstacle_bounds = np.array([item["bounds"] for item in pieces])
        for index, (q, width) in enumerate(zip(path, widths)):
            clear, diagnostic = self.check(q, width)
            if not clear:
                return False, dict(diagnostic, sample_index=index, checked_samples=index+1)
            poses = self._poses(q, width)
            robot_bounds = np.array([self._world_bounds(bounds, pose)
                                     for bounds, pose in zip(self._local_bounds, poses)])
            overlap = (np.all(robot_bounds[:, None, 0] <= obstacle_bounds[None, :, 1], axis=2)
                       & np.all(obstacle_bounds[None, :, 0] <= robot_bounds[:, None, 1], axis=2))
            for robot_index, obstacle_index in zip(*np.nonzero(overlap)):
                robot = self.geometries[robot_index]
                button = pieces[obstacle_index]
                if (button["color"] == target_color and button["kind"] == "cap"
                        and robot.link in ("link7", "link8")):
                    continue
                hit = self._collision(robot.shape, poses[robot_index], button["shape"], button["pose"])
                if hit is not None:
                    return False, dict(reason="button_collision", sample_index=index,
                        checked_samples=index+1, target_color=target_color,
                        pair=[robot.name, f"{button['color']}/{button['kind']}"], **hit)
        return True, dict(reason="clear", checked_samples=len(path), target_color=target_color)
