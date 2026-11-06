"""Task-independent PhysX tensor adapters and native error monitoring.

Isaac imports stay inside adapter construction so configuration, recording, and
CPU tests can import this module without starting a simulation application.
"""
from __future__ import annotations

from pathlib import Path
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

