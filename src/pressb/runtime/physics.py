"""Task-independent PhysX tensor adapters and native error monitoring.

Isaac imports stay inside adapter construction so configuration, recording, and
CPU tests can import this module without starting a simulation application.
"""
from __future__ import annotations

from pathlib import Path
import math
import re

import numpy as np


class PhysXErrorGuard:
    """Read the flushed native log on the Kit thread, never via a Python callback.

    Native PhysX workers can log while the Kit thread holds the GIL and waits
    for those workers. Registering a Python logger therefore risks deadlock.
    The synchronous native file sink supplies the same error evidence without
    crossing that thread boundary. Missing/replaced/truncated logs fail closed.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.errors = []
        self._stream = self.path.open("rb")
        stat = self.path.stat()
        self._identity = (stat.st_dev, stat.st_ino)
        self._pending = b""

    def _inspect(self, message):
        lower = message.lower()
        lost_physics = any(marker in lower for marker in
            ("miss interactions", "buffer overflow", "capacity exceeded", "pxgpudynamicsmemoryconfig"))
        lost_physics |= any(word in lower for word in ("dropped", "discarded")) and any(
            kind in lower for kind in ("contact", "constraint", "pair", "interaction"))
        if "physx" in lower and ("[error]" in lower or "physx error:" in lower or
                (("[warning]" in lower or "[warn]" in lower) and lost_physics)):
            if len(self.errors) < 16:
                self.errors.append(dict(source="native_physx_log", message=message[:4096],
                    file=str(self.path)))

    def check(self):
        if not self.errors:
            try:
                stat = self.path.stat()
                if (stat.st_dev, stat.st_ino) != self._identity or stat.st_size < self._stream.tell():
                    raise OSError("Native physics log was replaced or truncated")
                data = self._stream.read()
                if data:
                    lines = (self._pending + data).split(b"\n")
                    self._pending = lines.pop()[-8192:]
                    for line in lines:
                        self._inspect(line.decode("utf-8", errors="replace"))
                    # A flushed error need not wait for its trailing newline.
                    self._inspect(self._pending.decode("utf-8", errors="replace"))
            except (OSError, ValueError) as error:
                self.errors.append(dict(source="native_physx_log", message=str(error), file=str(self.path)))
        if self.errors:
            error = self.errors[0]
            raise RuntimeError(f"PhysX invalidated simulation: {error['source']}: {error['message']}")

    def close(self):
        self._stream.close()



def view_env_order(paths, count):
    """Explicitly map PhysX view order, which is not guaranteed numeric order."""
    ids = []
    for path in paths:
        match = re.search(r"/env_(\d+)(?:/|$)", str(path))
        if not match:
            raise ValueError(f"Unexpected environment prim path: {path}")
        ids.append(int(match.group(1)))
    if sorted(ids) != list(range(count)):
        raise ValueError("Articulation view omitted or duplicated an environment")
    return np.argsort(ids)



class TensorArms:
    """Small NumPy adapter over PhysX's public batched tensor API.

    Avoids Isaac 5 Articulation wrapper's unrelated-prim-deletion callback,
    which deletes the physics view when Replicator changes its render graph.
    """
    def __init__(self, simulation_view, paths):
        from isaacsim.core.utils.prims import get_articulation_root_api_prim_path
        self.view = simulation_view.create_articulation_view(
            [get_articulation_root_api_prim_path(path) for path in paths])
        self.prim_paths = self.view.prim_paths

    def get_dof_index(self, name):
        return self.view.shared_metatype.dof_names.index(name)

    def get_joint_positions(self):
        return self.view.get_dof_positions()

    def get_joint_velocities(self):
        return self.view.get_dof_velocities()

    def _set(self, getter, setter, value, indices):
        data = getter().copy()
        data[indices] = value
        setter(data, np.asarray(indices, dtype=np.uint32))

    def set_joint_positions(self, value, indices):
        self._set(self.view.get_dof_positions, self.view.set_dof_positions, value, indices)

    def set_joint_velocities(self, value, indices):
        self._set(self.view.get_dof_velocities, self.view.set_dof_velocities, value, indices)

    def set_joint_position_targets(self, value, indices):
        self._set(self.view.get_dof_position_targets, self.view.set_dof_position_targets, value, indices)



class TensorRigidBodies:
    def __init__(self, simulation_view, paths):
        self.view = simulation_view.create_rigid_body_view(paths)
        self.prim_paths = self.view.prim_paths

    def get_world_poses(self):
        poses = self.view.get_transforms()
        return poses[:, :3].copy(), poses[:, [6, 3, 4, 5]].copy()

    def get_velocities(self):
        return self.view.get_velocities()

    def set_world_poses(self, positions, orientations, indices):
        poses = self.view.get_transforms().copy()
        poses[indices, :3] = positions
        poses[indices, 3:] = orientations[:, [1, 2, 3, 0]]
        self.view.set_transforms(poses, np.asarray(indices, dtype=np.uint32))

    def set_velocities(self, velocities, indices):
        data = self.view.get_velocities().copy()
        data[indices] = velocities
        self.view.set_velocities(data, np.asarray(indices, dtype=np.uint32))



def path_rows(paths, requested):
    mapping = {str(path): i for i, path in enumerate(paths)}
    if len(mapping) != len(requested) or set(mapping) != set(requested):
        raise RuntimeError("PhysX view contains missing, duplicated, or unexpected bodies")
    return np.array([mapping[path] for path in requested], dtype=np.int64)



class ContactPathClassifier:
    """Cache immutable scene-topology matches, never slot activity or outcomes.

    Native routing supplies template-local actor/collider paths. All cloned
    environments therefore share this small mapping; recreate it on bind when
    the scene topology changes.
    """
    def __init__(self, roots):
        self.roots = tuple(roots)
        self._cache = {}

    def __call__(self, actor_paths):
        paths = tuple(actor_paths)
        result = self._cache.get(paths)
        if result is None:
            result = tuple(any(path == root or path.startswith(root+"/") for path in paths)
                           for root in self.roots)
            self._cache[paths] = result
        return result


def _contact_xyz(value, error):
    """Read native carb.Float3 without allocating a NumPy normal vector.

    Non-native inputs retain the original (3,) shape check. In particular, a
    tuple of one-element arrays must not become an accidentally valid vector.
    """
    try:
        if len(value) != 3:
            raise RuntimeError(error)
        x, y, z = value[0], value[1], value[2]
        if type(x) is float and type(y) is float and type(z) is float:
            return x, y, z
        array = np.asarray(value, dtype=float)
        if array.shape != (3,):
            raise RuntimeError(error)
        return float(array[0]), float(array[1]), float(array[2])
    except (TypeError, ValueError, IndexError) as exception:
        raise RuntimeError(error) from exception


class ContactForceReducer:
    """Reduce native points without changing impulse norms or summation order.

    Keep NumPy's impulse norm bit-for-bit, because accumulated finger forces
    control grasp/release predicates. Only the normal's three-scalar norm has
    a fast path; near either support decision boundary it uses the original
    NumPy expression. Shape, finite, normal, and separation checks remain.
    """
    def __init__(self, physics_dt, max_support_normal_tilt_deg, separation_tolerance_m,
                 error="Malformed measured contact data"):
        self.dt = physics_dt
        self.cosine = math.cos(math.radians(max_support_normal_tilt_deg))
        self.tolerance = separation_tolerance_m
        self.error = error

    def __call__(self, contacts):
        force, vertical = 0., 0.
        dt, cosine, tolerance, error = self.dt, self.cosine, self.tolerance, self.error
        for contact in contacts:
            impulse = np.asarray(contact.impulse, dtype=float)
            if impulse.shape != (3,):
                raise RuntimeError(error)
            ix, iy, iz = float(impulse[0]), float(impulse[1]), float(impulse[2])
            raw_normal = contact.normal
            nx, ny, nz = _contact_xyz(raw_normal, error)
            separation = float(contact.separation)
            if not (math.isfinite(ix) and math.isfinite(iy) and math.isfinite(iz)
                    and math.isfinite(nx) and math.isfinite(ny) and math.isfinite(nz)
                    and math.isfinite(separation)):
                raise RuntimeError(error)
            # Exact NumPy 1.26 norm fast path for a validated real float64
            # vector: keep the K-order flatten, ndarray.dot, and np.sqrt.
            # This removes only generic norm argument/dtype/shape dispatch.
            flat = impulse.ravel(order="K")
            force += float(np.sqrt(flat.dot(flat)))/dt
            normal_norm = math.sqrt(nx*nx+ny*ny+nz*nz)
            # Three positive products have no cancellation. These guards are
            # deliberately much wider than scalar-vs-dot rounding error so a
            # normal at, or nextafter either threshold takes the exact old path.
            if (abs(normal_norm-1e-8) <= 1e-20 or
                    (normal_norm > 1e-8 and abs(abs(nz)/normal_norm-cosine) <= 1e-12)):
                normal_norm = float(np.linalg.norm(np.asarray(raw_normal, dtype=float)))
            if (normal_norm > 1e-8 and abs(nz)/normal_norm >= cosine
                    and abs(separation) <= tolerance):
                vertical += abs(iz)/dt
        return force, vertical
