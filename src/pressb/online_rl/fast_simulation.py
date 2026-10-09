"""Indexed-reset Isaac collector with batched control and observation-rate rendering.

This is an explicit new simulation contract.  The original evaluator/backend is
kept intact: faster IK, camera resolution, render cadence and reset state are
recorded in the manifest rather than presented as pixel/trajectory equivalent.
"""
from __future__ import annotations

import base64
import copy
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
import re
import time
import uuid

import numpy as np
from PIL import Image

from .simulation import (IsaacVectorBackend, SimulationService, _file_identity,
                         _integer, _write_json, control_state, validate_actions_pose8)


RESET_MODE = "indexed_zero_step_cached_settled_state"


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


def encode_fast_image(rgb):
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.shape not in ((224, 224, 3), (240, 320, 3), (480, 640, 3)):
        raise ValueError("Fast RGB must be uint8 224x224, 320x240 or 640x480")
    frame = Image.fromarray(rgb)
    if rgb.shape == (240, 320, 3):
        # RTX pinhole rendering keeps square pixels, deriving vertical FOV
        # from horizontal aperture and render aspect. Render the trained 4:3
        # frustum first, then apply the policy's original square resize.
        frame = frame.resize((224, 224), Image.Resampling.BILINEAR)
    stream = BytesIO()
    frame.save(stream, format="PNG", compress_level=1)
    return base64.b64encode(stream.getvalue()).decode("ascii")


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


class IndexedSimulationService(SimulationService):
    """Keep true terminal transitions; only an explicit reset replaces a slot."""

    def health(self, payload=None):
        guard = getattr(self.backend, "_physics_error_guard", None)
        if guard is not None and self.fault is None:
            try:
                guard.check()
            except RuntimeError as error:
                self.fault = str(error)
                if hasattr(self.backend, "mark_fault"):
                    self.backend.mark_fault(self.fault)
        result = super().health(payload)
        result["reset_mode"] = RESET_MODE
        result["supports_indexed_reset"] = True
        if hasattr(self.backend, "timing"):
            result["timing"] = copy.deepcopy(self.backend.timing)
        return result

    def reset_envs(self, payload):
        key, digest, cached = self._request("reset_envs", payload)
        if cached is not None:
            return cached
        if self.items is None or payload.get("cohort_id") != self.cohort_id:
            raise ValueError("Missing reset or stale cohort_id")
        if _integer(payload.get("step_id"), "step_id") != self.step_id:
            raise ValueError("Stale step_id")
        seed = _integer(payload.get("seed"), "seed")
        values = payload.get("episodes")
        if not isinstance(values, list) or not values:
            raise ValueError("reset_envs requires at least one episode")
        episodes = {}
        for value in values:
            if not isinstance(value, dict):
                raise ValueError("Every episode must be an object")
            eid = _integer(value.get("env_id"), "env_id")
            if eid >= self.num_envs or eid in episodes:
                raise ValueError("Duplicate or unknown reset env_id")
            if not (self.items[eid]["terminated"] or self.items[eid]["truncated"]):
                raise ValueError("Only ended environments may reset")
            # Reuse the frozen bounds/floor validator before any mutation.
            episodes[eid] = self._episodes([value] * self.num_envs)[0]
        try:
            rows = self._rows(self.backend.reset_envs(episodes, seed), set(episodes))
            fresh = []
            for eid in sorted(episodes):
                row = rows[eid]
                item = dict(env_id=eid, episode_id=f"{self.cohort_id}:{eid}:{uuid.uuid4().hex}",
                    observation=copy.deepcopy(row["observation"]), terminated=False, truncated=False,
                    info=copy.deepcopy(row.get("info", {})))
                self.items[eid] = item
                fresh.append(item)
        except Exception as error:
            self._backend_failure(error)
        response = dict(protocol_version=1, cohort_id=self.cohort_id, step_id=self.step_id,
            items=fresh, reset_env_ids=sorted(episodes), all_done=not self._active())
        return self._save(key, digest, response)


class _SolverSlot:
    def __init__(self, controller, eid):
        self.controller, self.eid = controller, eid

    @property
    def q(self):
        return self.controller.q[self.eid].copy()


class _TensorArms:
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


class _TensorButtons:
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


class FastIsaacVectorBackend(IsaacVectorBackend):
    """Batch PhysX reads/writes and IK while keeping 120 Hz physical rewards."""

    def __init__(self, *args, camera_resolution=224, render_subframes=1,
                 light_settle_captures=1, light_settle_subframes=2,
                 ik_iterations=16, log_actions=False, png_workers=4, cpu_threads=8, gpu_dynamics=False,
                 native_log_path=None, **kwargs):
        if camera_resolution not in (224, 640):
            raise ValueError("camera_resolution must be 224 (square) or 640 (640x480)")
        self.resolution = (224, 224) if camera_resolution == 224 else (640, 480)
        self.render_resolution = (320, 240) if camera_resolution == 224 else (640, 480)
        self.render_subframes = _integer(render_subframes, "render_subframes", 1)
        self.fast_light_captures = _integer(light_settle_captures, "light_settle_captures")
        self.fast_light_subframes = _integer(light_settle_subframes, "light_settle_subframes", 1)
        self.ik_iterations = _integer(ik_iterations, "ik_iterations", 1)
        self.log_actions = bool(log_actions)
        self.png_workers = _integer(png_workers, "png_workers", 1)
        self.cpu_threads = _integer(cpu_threads, "cpu_threads", 1)
        if type(gpu_dynamics) is not bool:
            raise ValueError("gpu_dynamics must be a boolean")
        self.gpu_dynamics = gpu_dynamics
        self._image_pool = ThreadPoolExecutor(max_workers=self.png_workers, thread_name_prefix="rl_png")
        self._views_ready = False
        self._metrics_cache = None
        self._settled = None
        self._reset_serial = 0
        self.timing = dict(chunks=0, render_calls=0, reset_calls=0, physics_seconds=0.,
            control_seconds=0., metric_seconds=0., capture_seconds=0., encode_seconds=0.,
            contact_seconds=0., contact_headers=0, relevant_contact_headers=0,
            total_step_seconds=0., total_reset_seconds=0.)
        if native_log_path is None:
            import carb.settings
            native_log_path = carb.settings.get_settings().get("/log/file")
        if not native_log_path:
            raise ValueError("Fast simulation requires a flushed native Kit log file")
        self._physics_error_guard = PhysXErrorGuard(native_log_path)
        try:
            super().__init__(*args, camera_resolution=self.render_resolution, gpu_dynamics=gpu_dynamics, **kwargs)
            self._physics_error_guard.check()
        except BaseException as error:
            self._physics_error_guard.close()
            self._image_pool.shutdown(wait=False, cancel_futures=True)
            if hasattr(self, "_world_ticks") and self.output.exists():
                try:
                    self._status("faulted", error=str(error))
                except OSError:
                    pass
            raise
        from threadpoolctl import threadpool_info, threadpool_limits
        # Kit loads NumPy/BLAS before user modules and can override thread env
        # vars. Tiny batched 6x6 IK solves must not fan out to dozens of cores.
        self._blas_limit = threadpool_limits(limits=1, user_api="blas")
        from isaacsim.core.simulation_manager import SimulationManager
        from .batched_control import BatchedPoseController

        simulation_view = SimulationManager.get_physics_sim_view()
        self.arms = _TensorArms(simulation_view, [env.robot_path for env in self.envs])
        self.arm_rows = view_env_order(self.arms.prim_paths, self.num_envs)
        self.arm_dofs = np.array([self.arms.get_dof_index(name) for name in self.tcp_kin.joint_names])
        self.finger_dofs = np.array([self.arms.get_dof_index(name) for name in ("joint7", "joint8")])
        paths = [env.button_info[floor]["body_path"] for env in self.envs for floor in range(24, 36)]
        self.button_view = _TensorButtons(simulation_view, paths)
        mapping = {str(path): i for i, path in enumerate(self.button_view.prim_paths)}
        if len(mapping) != len(paths) or any(path not in mapping for path in paths):
            raise RuntimeError("Button PhysX view is incomplete")
        self.button_rows = np.array([mapping[path] for path in paths]).reshape(self.num_envs, 12)
        self.batched_controller = BatchedPoseController(self.urdf,
            np.repeat(self.home[None], self.num_envs, axis=0), fps=30, iterations=self.ik_iterations)
        self._commands = np.stack([control[2].copy() for control in self.controls])
        self._views_ready = True
        # Reward/control reads use PhysX tensors. Publish USD transforms only
        # at requested observation boundaries, rather than every 120 Hz tick.
        self.world.get_physics_context().set_physx_update_transformations_settings(
            update_to_usd=False, update_velocities_to_usd=False)
        self._light_captures = self.fast_light_captures
        self._light_subframes = self.fast_light_subframes
        physics_context = self.world.get_physics_context()
        if bool(physics_context.is_gpu_dynamics_enabled()) != gpu_dynamics:
            raise RuntimeError("PhysX dynamics backend differs from the explicitly requested mode")
        self.manifest.update(reset_mode=RESET_MODE, supports_indexed_reset=True,
            control_backend="batched_numpy_analytic_jacobian_damped_least_squares",
            physics_backend="PhysX_numpy_tensor_views", ik_iterations=self.ik_iterations,
            gpu_dynamics=gpu_dynamics, ccd_enabled=bool(physics_context.is_ccd_enabled()),
            gpu_memory_capacities=copy.deepcopy(self.gpu_memory_config),
            physics_error_policy="fail_closed_on_native_physx_error",
            physics_error_monitor="main_thread_incremental_native_log",
            native_log_path=str(self._physics_error_guard.path),
            gpu_readback="enabled_for_numpy" if gpu_dynamics else "not_applicable",
            cpu_threads=self.cpu_threads, blas_threadpools=threadpool_info(), disable_default_viewport_updates=True,
            image_resolution=list(self.resolution), render_resolution=list(self.render_resolution),
            camera_contract="native_4_3_tiled_projection_then_policy_bilinear_square_resize",
            transform_publication="observation_boundaries_explicit_usd_sync",
            logging="Episode reset/terminal records and chunk timings; optional action diagnostics",
            clock="Physics advances only in explicit step and first full-reset settling",
            rollout_render_interval_physics_steps=28,
            terminal_render="At the exact terminating physics tick, before reset or further physics",
            reset_initialization="One 90-tick whole-world home settling; indexed restoration of cached settled state",
            checkpoint_compatibility="New collector contract; legacy weights may initialize a new experiment, replay is not equivalent")
        self.manifest["renderer_settings"].update(render_subframes=self.render_subframes,
            light_settle_captures=self.fast_light_captures, light_settle_subframes=self.fast_light_subframes,
            cadence="chunk_boundary_and_exact_terminal", png_compress_level=1, png_workers=self.png_workers)
        self.manifest["simulation_contract"] = dict(version="fast_indexed_v2_camera4_3",
            reset=RESET_MODE, control=self.batched_controller.contract,
            camera_resolution=list(self.resolution), render_interval_physics_steps=28,
            render_resolution=list(self.render_resolution), camera_projection="trained_4_3_frustum",
            camera_resize="PIL_bilinear_224_square" if camera_resolution == 224 else "policy_node_PIL_bilinear_224_square",
            terminal_observation="exact_terminal_tick_before_reset",
            gpu_dynamics=gpu_dynamics, ccd_enabled=bool(physics_context.is_ccd_enabled()),
            gpu_memory_capacities=copy.deepcopy(self.gpu_memory_config),
            physics_error_policy="fail_closed_on_native_physx_error",
            physics_error_monitor="main_thread_incremental_native_log",
            transform_publication="observation_boundaries_explicit_usd_sync")
        for name in ("src/pressb/online_rl/fast_simulation.py", "src/pressb/online_rl/batched_control.py",
                     "scripts/serve_rl_fast_simulation.py"):
            self.manifest["sources"][name] = _file_identity(self.root / name)
        _write_json(self.output / "simulation_manifest.json", self.manifest)

    def health(self):
        self._physics_error_guard.check()
        result = super().health()
        result["timing"] = copy.deepcopy(self.timing)
        return result

    def mark_fault(self, error):
        import traceback
        traceback.print_exc()
        super().mark_fault(error)

    def _classify_contact(self, actors, colliders):
        """Classify static paths before reading impulses or active slot state."""
        pair = next((self.body_map[actor] for actor in actors if actor in self.body_map), None)
        if pair is not None and any(collider in self.tool_paths for collider in colliders):
            return pair, ()
        candidates = set()
        for actor in actors:
            match = re.search(r"/env_(\d+)/Piper/", actor)
            if match:
                candidates.add(int(match.group(1)))
        reasons = []
        for eid in candidates:
            prefix = self.envs[eid].robot_path + "/"
            robot_actors = [actor for actor in actors if actor.startswith(prefix)]
            shaft = prefix + "link6/PressStylus"
            pads = {prefix + "link7/collisions", prefix + "link8/collisions"}
            if shaft in colliders and any(collider in pads for collider in colliders):
                continue
            reason = ("non_stylus_button_contact" if pair is not None else
                "robot_self_contact" if len(robot_actors) == 2 and actors[0] != actors[1] else
                "robot_environment_contact" if len(robot_actors) == 1 and
                robot_actors[0].rsplit("/", 1)[-1] not in ("dummy_link", "base_link", "link1") else None)
            if reason:
                reasons.append((eid, reason))
        return None, tuple(reasons)

    def _on_contact(self, headers, data):
        from pxr import PhysicsSchemaTools
        started = time.perf_counter()
        if not hasattr(self, "_contact_path_cache"):
            self._contact_path_cache = {}
            self._contact_classes = {}
        def path(value):
            key = int(value)
            if key not in self._contact_path_cache:
                self._contact_path_cache[key] = str(PhysicsSchemaTools.intToSdfPath(value))
            return self._contact_path_cache[key]
        for header in headers:
            self.timing["contact_headers"] += 1
            if not header.num_contact_data:
                continue
            key = tuple(int(value) for value in (header.actor0, header.actor1, header.collider0, header.collider1))
            if key not in self._contact_classes:
                actors = [path(value) for value in (header.actor0, header.actor1)]
                colliders = [path(value) for value in (header.collider0, header.collider1)]
                pair, reasons = self._classify_contact(actors, colliders)
                self._contact_classes[key] = pair, reasons, actors, colliders
            pair, reasons, actors, colliders = self._contact_classes[key]
            if pair is not None:
                self.timing["relevant_contact_headers"] += 1
                magnitude = sum(float(np.linalg.norm(data[k].impulse)) / self.dt for k in
                    range(header.contact_data_offset, header.contact_data_offset + header.num_contact_data))
                self.forces[pair[0], pair[1] - 24] += magnitude
                continue
            live_reasons = [(eid, reason) for eid, reason in reasons
                if eid in self.recording and len(self.collisions[eid]) < 100]
            if not live_reasons:
                continue
            self.timing["relevant_contact_headers"] += 1
            magnitude = sum(float(np.linalg.norm(data[k].impulse)) / self.dt for k in
                range(header.contact_data_offset, header.contact_data_offset + header.num_contact_data))
            if magnitude <= .1:
                continue
            for eid, reason in live_reasons:
                self.collisions[eid].append(dict(physics_index=self._items[eid]["ticks"],
                    actors=actors, colliders=colliders, force_n=magnitude, reason=reason))
        self.timing["contact_seconds"] += time.perf_counter() - started

    def _joint_arrays(self):
        q = np.asarray(self.arms.get_joint_positions())[self.arm_rows].astype(np.float64)
        v = np.asarray(self.arms.get_joint_velocities())[self.arm_rows].astype(np.float64)
        return q, v

    def _remember_frozen_state(self):
        if not self._views_ready:
            return super()._remember_frozen_state()
        self._idle_time = self.world.current_time
        self._idle_joints = self._joint_arrays()[0]

    def _check_frozen_state(self):
        self._physics_error_guard.check()
        if not self._views_ready:
            return super()._check_frozen_state()
        if self._idle_time is not None and (self.world.current_time != self._idle_time or
                not np.array_equal(self._joint_arrays()[0], self._idle_joints)):
            raise RuntimeError("Simulation changed while waiting for the next request")

    def _capture(self, changed=False):
        self._physics_error_guard.check()
        if not self._views_ready:
            return super()._capture(changed)
        import omni.replicator.core as rep
        import omni.physx
        import omni.usd
        started = time.perf_counter()
        before, prior = self.world.current_time, self._joint_arrays()[0]
        self.world.physics_sim_view.update_articulations_kinematic()
        omni.physx.get_physx_interface().update_transformations(False, True, True)
        changed = changed or self._dirty_lights
        if changed:
            omni.usd.get_context().reset_renderer_accumulation()
            for _ in range(self.fast_light_captures):
                rep.orchestrator.step(delta_time=0., pause_timeline=False, rt_subframes=self.fast_light_subframes)
                self.timing["render_calls"] += 1
        rep.orchestrator.step(delta_time=0., pause_timeline=False,
            rt_subframes=self.fast_light_subframes if changed else self.render_subframes)
        self._physics_error_guard.check()
        self.timing["render_calls"] += 1
        if self.world.current_time != before or not np.array_equal(self._joint_arrays()[0], prior):
            raise RuntimeError("RGB capture advanced physical state")
        tiles = self.split_rgb(self.annotator.get_data())
        shape = (self.render_resolution[1], self.render_resolution[0], 3)
        if any(rgb.shape != shape or rgb.dtype != np.uint8 or rgb.std() < 2 for rgb in tiles):
            raise RuntimeError("Invalid live camera RGB")
        self._dirty_lights = False
        self.timing["capture_seconds"] += time.perf_counter() - started
        return tiles

    def _refresh_metrics(self):
        from scipy.spatial.transform import Rotation
        q, velocity = self._joint_arrays()
        arm = q[:, self.arm_dofs]
        transforms = self.batched_controller.fk(arm)
        quaternions = Rotation.from_matrix(transforms[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]
        state = np.c_[transforms[:, :3, 3], quaternions,
            q[:, self.finger_dofs[0]] - q[:, self.finger_dofs[1]]]
        positions = np.asarray(self.button_view.get_world_poses()[0])[self.button_rows]
        rest = np.array([[env.button_info[floor]["rest_x"] for floor in range(24, 36)] for env in self.envs])
        travel = positions[:, :, 0] - rest
        # Physical state comes from PhysX. FK is batched for proprioception and
        # distance diagnostics, never used as the success/travel measurement.
        tip = transforms[:, :3, 3] + transforms[:, :3, 2] * (.24 - .1358)
        tip += np.stack([env.base_position for env in self.envs])
        if not np.isfinite(np.r_[state.ravel(), velocity.ravel(), travel.ravel(), self.forces.ravel()]).all():
            raise RuntimeError("Nonfinite physical metrics")
        self._metrics_cache = dict(q=q, arm=arm, velocity=velocity[:, self.arm_dofs],
            state=state, travel=travel, tip=tip)

    def _measured(self, eid):
        if not self._views_ready or self._metrics_cache is None:
            return super()._measured(eid)
        data = self._metrics_cache
        return data["state"][eid].copy(), data["arm"][eid].copy(), data["velocity"][eid].copy()

    def _measure_metrics(self, eid):
        if not self._views_ready or self._metrics_cache is None:
            return super()._measure_metrics(eid)
        item, data, env = self._items[eid], self._metrics_cache, self.envs[eid]
        distance = np.linalg.norm(data["tip"][eid] - env.button_info[item["floor"]]["center"])
        tracking = np.max(np.abs(data["arm"][eid] - self._commands[eid, self.arm_dofs]))
        index = item["floor"] - 24
        item["min_target_distance"] = min(item["min_target_distance"], float(distance))
        item["max_tracking_error"] = max(item["max_tracking_error"], float(tracking))
        item["max_target_force"] = max(item["max_target_force"], float(self.forces[eid, index]))
        item["max_target_travel"] = max(item["max_target_travel"], float(data["travel"][eid, index]))
        return data["travel"][eid]

    def _encode_images(self, ids, tiles):
        started = time.perf_counter()
        ids = list(ids)
        ordered = [tiles[index] for eid in ids for index in (2 * eid + 1, 2 * eid)]
        encoded = list(self._image_pool.map(encode_fast_image, ordered))
        self.timing["encode_seconds"] += time.perf_counter() - started
        return {eid: {"global": encoded[2 * row], "wrist": encoded[2 * row + 1]}
                for row, eid in enumerate(ids)}

    def _observation(self, eid, tiles, images=None):
        started = time.perf_counter()
        item = self._items[eid]
        state, q, v = self._measured(eid)
        result = dict(task=f"Press {item['floor']} floor.", state=state.tolist(),
            images=images if images is not None else {
                "global": encode_fast_image(tiles[2 * eid + 1]), "wrist": encode_fast_image(tiles[2 * eid])},
            control_state=control_state(q, v, item["solver"].q, list(item["history"]),
                item["ticks"] * self.dt, self.max_seconds))
        self.timing["encode_seconds"] += time.perf_counter() - started
        return result

    def reset(self, episodes, seed):
        if self._settled is not None:
            if any(not item["done"] for item in self._items):
                raise RuntimeError("Full reset requires all episodes ended")
            return self.reset_envs(dict(enumerate(episodes)), seed)
        started = time.perf_counter()
        self._metrics_cache = None
        rows = super().reset(episodes, seed)
        q, v = self._joint_arrays()
        positions, orientations = self.button_view.get_world_poses()
        velocities = self.button_view.get_velocities()
        rests = np.array([[env.button_info[floor]["body_center"] for floor in range(24, 36)] for env in self.envs])
        self._settled = dict(q=q.copy(), velocity=v.copy(),
            button_relative=np.asarray(positions)[self.button_rows] - rests,
            button_orientation=np.asarray(orientations)[self.button_rows].copy(),
            button_velocity=np.asarray(velocities)[self.button_rows].copy())
        self.batched_controller.reset(np.arange(self.num_envs), np.repeat(self.home[None], self.num_envs, axis=0))
        for eid, item in enumerate(self._items):
            item["solver"] = _SolverSlot(self.batched_controller, eid)
            self._commands[eid] = self.controls[eid][2]
        self._refresh_metrics()
        self.timing["reset_calls"] += 1
        self.timing["total_reset_seconds"] += time.perf_counter() - started
        return rows

    def reset_envs(self, episodes, seed):
        self._check_thread()
        self._check_frozen_state()
        if self._settled is None:
            raise RuntimeError("Initial full reset required")
        ids = np.array(sorted(episodes), dtype=np.int64)
        if not len(ids) or any(eid >= self.num_envs or not self._items[eid]["done"] for eid in ids):
            raise ValueError("Indexed reset requires ended environments")
        started = time.perf_counter()
        from pressb.dataset_scene import set_panel_offset, validate_panel_layout
        before_time = self.world.current_time
        before_q, before_v = self._joint_arrays()
        peers = np.array([eid for eid in range(self.num_envs) if eid not in episodes], dtype=np.int64)
        self.recording.difference_update(map(int, ids))
        rows = self.arm_rows[ids]
        self.arms.set_joint_positions(self._settled["q"][ids], indices=rows)
        self.arms.set_joint_velocities(self._settled["velocity"][ids], indices=rows)
        self._commands[ids[:, None], self.arm_dofs] = self.home
        self._commands[ids[:, None], self.finger_dofs] = self.cfg["gripper_joint_positions_m"]
        self.arms.set_joint_position_targets(self._commands[ids], indices=rows)
        self.batched_controller.reset(ids, np.repeat(self.home[None], len(ids), axis=0))
        self._reset_serial += 1
        directory = self.output / f"indexed_reset_{self._reset_serial:06d}"
        directory.mkdir()
        for eid in map(int, ids):
            episode, env = episodes[eid], self.envs[eid]
            set_panel_offset(self.world, env, episode["offset_y_m"], episode["offset_x_m"])
            rests = np.array([env.button_info[floor]["body_center"] for floor in range(24, 36)])
            self.button_view.set_world_poses(rests + self._settled["button_relative"][eid],
                self._settled["button_orientation"][eid], indices=self.button_rows[eid])
            self.button_view.set_velocities(self._settled["button_velocity"][eid], indices=self.button_rows[eid])
            initial = self.home.copy()
            self._items[eid] = dict(floor=episode["floor"], ticks=0, done=False, termination=None,
                lit=set(), events=[], solver=_SolverSlot(self.batched_controller, eid),
                history=deque((initial.copy() for _ in range(self.window)), maxlen=self.window),
                min_target_distance=float("inf"), max_tracking_error=0., max_target_force=0., max_target_travel=0.,
                reset_episode=copy.deepcopy(episode), directory=directory, seed=seed,
                layout_evidence=validate_panel_layout(self.world, env))
            self.collisions[eid] = []
            self.forces[eid] = 0.
            self.recording.add(eid)
        # Tensor state writes must also reach link transforms and the rendered
        # USD scene before a zero-physics-tick observation is taken. This is
        # the CPU/USD counterpart of IsaacLab SimulationContext.forward().
        import omni.physx
        self.world.physics_sim_view.update_articulations_kinematic()
        omni.physx.get_physx_interface().update_transformations(False, True, True)
        self._metrics_cache = None
        self._refresh_metrics()
        tiles = self._capture(changed=True)
        after_q, after_v = self._joint_arrays()
        if self.world.current_time != before_time or (len(peers) and (
                not np.array_equal(before_q[peers], after_q[peers]) or
                not np.array_equal(before_v[peers], after_v[peers]))):
            raise RuntimeError("Indexed reset changed peer physics")
        output = []
        encoded = self._encode_images(ids, tiles)
        for eid in map(int, ids):
            self._measure_metrics(eid)
            observation, info = self._observation(eid, tiles, encoded[eid]), self._info(eid)
            info["reset_physics_steps"] = 0
            output.append(dict(env_id=eid, observation=observation, info=info))
            _write_json(directory / f"env_{eid:03d}_reset.json", dict(env_id=eid, seed=seed,
                run_id=self._run_id, episode=episodes[eid], initial_state=observation["state"],
                layout=self._items[eid]["layout_evidence"], info=info))
        self.timing["reset_calls"] += 1
        self.timing["total_reset_seconds"] += time.perf_counter() - started
        self._remember_frozen_state()
        return output

    def step(self, actions):
        self._check_thread()
        self._check_frozen_state()
        from pressb.dataset_scene import set_light
        if not self.app.is_running():
            raise RuntimeError("Simulator closed")
        expected = {eid for eid, item in enumerate(self._items or []) if not item["done"]}
        if set(actions) != expected or not expected:
            raise RuntimeError("step requires every active environment exactly once")
        actions = {eid: validate_actions_pose8(value) for eid, value in actions.items()}
        started = time.perf_counter()
        before = {eid: self._items[eid]["ticks"] for eid in expected}
        terminals, records = {}, {eid: [] for eid in expected}
        for index in range(7):
            active = np.array(sorted(eid for eid in expected if not self._items[eid]["done"]), dtype=np.int64)
            if not len(active):
                break
            control_started = time.perf_counter()
            previous = self.batched_controller.q[active].copy()
            targets, diagnostics = self.batched_controller.solve(active,
                np.stack([actions[eid][index] for eid in active]))
            for row, eid in enumerate(active):
                records[int(eid)].append(dict(action_index=index, action_pose8=actions[eid][index].tolist(),
                    q_before=previous[row].tolist(), q_target=targets[row].tolist(), **diagnostics[row]))
            self.timing["control_seconds"] += time.perf_counter() - control_started
            for substep in range(1, 5):
                live = np.array([row for row, eid in enumerate(active) if not self._items[eid]["done"]], dtype=np.int64)
                if not len(live):
                    break
                ids = active[live]
                linear = previous[live] + (targets[live] - previous[live]) * substep / 4
                for row, eid in enumerate(ids):
                    item = self._items[eid]
                    item["history"].append(linear[row].copy())
                    self._commands[eid, self.arm_dofs] = np.mean(item["history"], axis=0)
                    item["ticks"] += 1
                self.arms.set_joint_position_targets(self._commands[ids], indices=self.arm_rows[ids])
                self.forces.fill(0.)
                tick_started = time.perf_counter()
                self.world.step(render=False)
                self._physics_error_guard.check()
                self._world_ticks += 1
                self.timing["physics_seconds"] += time.perf_counter() - tick_started
                metric_started = time.perf_counter()
                self._refresh_metrics()
                ended = []
                for eid in map(int, ids):
                    item, env = self._items[eid], self.envs[eid]
                    travel = self._measure_metrics(eid)
                    pressed = (travel >= self.cfg["press_threshold"]) & (self.forces[eid] > .02)
                    released = travel <= self.cfg["release_threshold"]
                    changes = [(floor, bool(pressed[floor - 24])) for floor in range(24, 36)
                        if (floor not in item["lit"] and pressed[floor - 24]) or
                           (floor in item["lit"] and released[floor - 24])]
                    for floor, on in changes:
                        self._dirty_lights = True
                        item["lit"].add(floor) if on else item["lit"].remove(floor)
                        set_light(env, floor, on)
                        item["events"].append(dict(type="pressed" if on else "released", floor=floor,
                            physics_index=item["ticks"], sim_time=item["ticks"] * self.dt,
                            travel_m=float(travel[floor - 24]), force_n=float(self.forces[eid, floor - 24])))
                    presses = [event["floor"] for event in item["events"] if event["type"] == "pressed"]
                    reason = ("unexpected_collision" if self.collisions[eid] else
                        "target_pressed" if presses == [item["floor"]] else
                        "wrong_button_pressed" if presses else
                        "time_limit" if item["ticks"] >= self.max_steps else None)
                    if reason:
                        item.update(done=True, termination=reason)
                        self.recording.discard(eid)
                        ended.append(eid)
                self.timing["metric_seconds"] += time.perf_counter() - metric_started
                if ended:
                    tiles = self._capture()
                    encoded = self._encode_images(ended, tiles)
                    for eid in ended:
                        terminals[eid] = self._observation(eid, tiles, encoded[eid])
        # One render per ordinary chunk. No intermediate 30 Hz captures.
        live = sorted(eid for eid in expected if not self._items[eid]["done"])
        if live:
            tiles = self._capture()
            encoded = self._encode_images(live, tiles)
        rows = []
        for eid in sorted(expected):
            item = self._items[eid]
            observation = terminals[eid] if item["done"] else self._observation(eid, tiles, encoded[eid])
            info = self._info(eid)
            info.update(executed_action_steps=len(records[eid]),
                executed_sim_seconds=(item["ticks"] - before[eid]) * self.dt,
                max_chunk_ik_position_residual_m=max(row["command_position_residual_m"] for row in records[eid]),
                velocity_limited_actions=sum(bool(row["velocity_saturated_joints"]) for row in records[eid]))
            rows.append(dict(env_id=eid, observation=observation,
                terminated=item["done"] and item["termination"] != "time_limit",
                truncated=item["termination"] == "time_limit",
                executed_physics_steps=item["ticks"] - before[eid], info=info))
            if self.log_actions:
                import json
                with (item["directory"] / f"env_{eid:03d}_actions.jsonl").open("a") as stream:
                    stream.write(json.dumps(dict(chunk_index=self._chunk_index,
                        physics_start_index=before[eid], physics_end_index=item["ticks"],
                        actions=records[eid], info=info), allow_nan=False) + "\n")
            if item["done"]:
                _write_json(item["directory"] / f"env_{eid:03d}_terminal.json", dict(
                    run_id=self._run_id, episode=item["reset_episode"], seed=item["seed"], info=info,
                    terminal_state=observation["state"], terminal_control_state=observation["control_state"]))
        self._chunk_index += 1
        self.timing["chunks"] += 1
        self.timing["total_step_seconds"] += time.perf_counter() - started
        self._status("ready", active_env_ids=sorted(self.recording), timing=self.timing)
        self._remember_frozen_state()
        return rows

    def close(self):
        self._image_pool.shutdown(wait=True)
        try:
            super().close()
        finally:
            self._physics_error_guard.close()
