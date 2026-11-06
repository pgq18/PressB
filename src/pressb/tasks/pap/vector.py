"""Shared-World PAP backend with PhysX tensor control and tiled RGB rendering.

The process-parallel PAP backend remains available. This backend is an explicit
different execution contract: all live slots step together, while indexed reset
restores the cached settled arm and places only the selected object, with no
physics tick. Original calibrated camera apertures and poses are retained; the
optional smaller render resolution changes pixel count, not the camera frustum.
"""
from __future__ import annotations

import base64
import copy
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import time

import numpy as np

from pressb.runtime.physics import path_rows as _path_rows

from .backend import (ROOT, PAPParallelBackend, _integer, _path, _positive, _source_identity, _source_revision,
                  load_config as load_pap_config, sample_reset, task_identity)
from pressb.runtime.image_transport import encode_preprocessed_image, validate_image_preprocessing, resized_size


RESET_CONTRACT = "indexed_zero_tick_cached_settled_arm_and_supported_object_v1"
_VECTOR_SOURCE_FILES = (
    "tasks/pap/vector.py", "sensors/tiled.py",
    "runtime/physics.py", "runtime/image_transport.py", "runtime/contract.py",
)


def renderer_profile(name):
    """Explicit rendering contract; no change to geometry, cameras or exposure.

    The fast profile keeps DLSS Ray Reconstruction's spatial quality and
    explicitly resets its history at observation boundaries. Kit 107's Ray
    Reconstruction overrides requested FXAA/TAA with DLSS (AA mode 3), so merely
    asking for FXAA does not remove stale poses. The additional non-temporal
    NRD controls match the existing elevator collector.
    """
    if name == "pap_calibrated":
        return {"anti_aliasing": 1, "history_controls": {}}
    if name != "rl_fast":
        raise ValueError("renderer_profile must be pap_calibrated or rl_fast")
    return {"anti_aliasing": 3, "reset_dlss_history": True, "history_controls": {
        "/rtx-transient/dldenoiser/enabled": True,
        "/rtx/indirectDiffuse/denoiser/temporal/enabled": False,
        "/rtx/directLighting/sampledLighting/ris/enableSpatioTemporalRis": False,
        **{f"/rtx/lightspeed/NRD_Reblur{channel}/{key}": 0
           for channel in ("Diffuse", "Specular")
           for key in ("maxAccumulatedFrameNum", "maxFastAccumulatedFrameNum")},
        **{f"/rtx/lightspeed/NRD_ReLAX/{key}": 0 for key in
           ("diffuseHistoryFrames", "diffuseFastHistoryFrames", "specularHistoryFrames", "specularFastHistoryFrames")},
    }}


def load_config(config):
    result = load_pap_config(config)
    if len(result["objects"]) != 1:
        raise ValueError("Shared-World PAP currently supports one configured object geometry; use process PAP for variants")
    if len({w["gpu"] for w in result["workers"]}) != 1:
        raise ValueError("A shared-World PAP process uses one GPU; use a separate node per GPU")
    result["gpu"] = result["workers"][0]["gpu"]
    result["spacing_m"] = _positive(result.get("spacing_m", 6.), "spacing_m")
    if result["spacing_m"] < 6.:
        raise ValueError("spacing_m must be >= 6 to separate the calibrated room floors")
    render = result["scene"].get("rendering", {})
    result["renderer_profile"] = result.get("renderer_profile", "pap_calibrated")
    renderer_profile(result["renderer_profile"])
    fast = result["renderer_profile"] == "rl_fast"
    for name, default in (("cpu_threads", 8), ("png_workers", 8),
                          ("render_subframes", 1 if fast else render.get("subframes_per_frame", 1)),
                          ("observation_render_frames", 2 if fast else 1),
                          ("reset_render_frames", 2 if fast else render.get("warmup_frames", 8))):
        result[name] = _integer(result.get(name, default), name)
    resolution = result.get("render_resolution", result["scene"]["up_camera"]["resolution"])
    if (not isinstance(resolution, (list, tuple)) or len(resolution) != 2
            or any(type(v) is not int or v <= 0 for v in resolution)):
        raise ValueError("render_resolution requires two positive integer dimensions")
    result["render_resolution"] = list(resolution)
    result["image_preprocessing"] = validate_image_preprocessing(result.get("image_preprocessing"))
    # Tiled RTX derives vertical FOV from horizontal aperture and tile aspect.
    # Both sensors must therefore retain their independently calibrated aspect.
    specs = [result["scene"]["up_camera"], result["scene"]["wrist_camera"]]
    for spec in specs:
        w, h = spec["resolution"]
        if resolution[0] * h != resolution[1] * w:
            raise ValueError("render_resolution must preserve each calibrated camera's native aspect ratio")
    return result


def _encode_png(rgb, image_preprocessing=None):
    if image_preprocessing is not None:
        return encode_preprocessed_image(rgb, image_preprocessing)
    from PIL import Image
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise RuntimeError("Expected uint8 RGB tile")
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb)).save(buffer, format="PNG", compress_level=1)
    return {"encoding": "png_base64", "data": base64.b64encode(buffer.getvalue()).decode("ascii")}


class PAPVectorBackend:
    def __init__(self, config):
        self.config = load_config(config)
        cfg = self.config
        self._owner_thread = threading.get_ident()
        self._closed = False
        self.app = self.world = self.subscription = self._guard = None
        self._pool = None
        self.num_envs = cfg["num_envs"]
        directory = _path(cfg.get("log_dir", "outputs/pap_vector_workers"))
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix="run_", dir=directory))
        (self.directory / "config.json").write_text(json.dumps(cfg, indent=2))
        self.identity = task_identity(cfg)
        semantics = {"task": self.identity["task_config_sha256"], "reset": RESET_CONTRACT,
                     "render_resolution": cfg["render_resolution"], "spacing_m": cfg["spacing_m"],
                     "renderer_profile": cfg["renderer_profile"],
                     "image_preprocessing": cfg["image_preprocessing"],
                     "renderer_settings": renderer_profile(cfg["renderer_profile"]),
                     "observation_render_frames": cfg["observation_render_frames"],
                     "render_subframes": cfg["render_subframes"], "reset_render_frames": cfg["reset_render_frames"]}
        self.identity["task_config_sha256"] = hashlib.sha256(json.dumps(semantics, sort_keys=True).encode()).hexdigest()
        self.identity["source_identity"].update(_source_identity(_VECTOR_SOURCE_FILES))
        self.identity["source_revision"] = _source_revision(self.identity["source_identity"])
        self.timings = dict(physics_seconds=0., render_seconds=0., encode_seconds=0.,
                            step_seconds=0., reset_seconds=0., world_ticks=0, captures=0,
                            fused_step_reset_seconds=0., fused_steps=0)
        try:
            self._initialize()
        except BaseException:
            self.close()
            raise

    def _initialize(self):
        cfg = self.config
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        os.environ.setdefault("PXR_WORK_THREAD_LIMIT", str(cfg["cpu_threads"]))
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(name, "1")
        from isaacsim import SimulationApp
        log = self.directory / "kit.log"
        self.app = SimulationApp(dict(headless=True, active_gpu=cfg["gpu"], physics_gpu=cfg["gpu"],
            multi_gpu=False, renderer="RayTracedLighting",
            anti_aliasing=renderer_profile(cfg["renderer_profile"])["anti_aliasing"], fast_shutdown=True,
            disable_viewport_updates=True, limit_cpu_threads=cfg["cpu_threads"],
            extra_args=["--/app/asyncRendering=false", "--/rtx/ecoMode/enabled=false",
                f"--/log/file={log}", "--/log/fileFlushLevel=verbose", "--/log/fileAppend=false",
                "--/log/async=false", f"--/plugins/carb.tasking.plugin/threadCount={cfg['cpu_threads']}",
                f"--/plugins/omni.tbb.globalcontrol/maxThreadCount={cfg['cpu_threads']}",
                "--/persistent/physics/numThreads=4", "--/validate/p2p/enabled=false", "--/validate/iommu/enabled=false"]))
        from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics
        from isaacsim.core.cloner import Cloner
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.core.utils.prims import get_articulation_root_api_prim_path
        import omni.physx
        import omni.replicator.core as rep
        from pressb.control.joint_actions import JointActionPostprocessor
        from pressb.sensors.tiled import build_tiled_rgb
        from pressb.robots.piper.kinematics import PiperKinematics
        from pressb.runtime.physics import TensorArms, TensorRigidBodies, PhysXErrorGuard
        from pressb.tasks.pap.initialization import load_policy_initial
        from .worker import _build

        template = _build(cfg, initialize=False)
        self.world, self.stage = template.world, template.stage
        self.source_scene = copy.deepcopy(template.scene)
        self.source_base = np.asarray(template.base)
        snapshot = self.directory / "template.usdc"
        self.stage.Flatten().Export(str(snapshot))
        self.world.scene.clear(registry_only=True)
        physics_path = self.world.get_physics_context().prim_path
        if str(physics_path).startswith("/World/"):
            raise RuntimeError("Shared-World builder requires physicsScene outside the cloned /World")
        self.stage.RemovePrim("/World")
        UsdGeom.Xform.Define(self.stage, "/World")
        UsdGeom.Scope.Define(self.stage, "/World/envs")
        self.prefixes = [f"/World/envs/env_{i}" for i in range(self.num_envs)]
        columns = math.ceil(math.sqrt(self.num_envs))
        self.offsets = np.array([[i // columns * cfg["spacing_m"], i % columns * cfg["spacing_m"], 0.]
                                 for i in range(self.num_envs)])
        self.object_paths, self.robot_paths, self.material_paths, camera_paths = [], [], [], []
        for i, (prefix, offset) in enumerate(zip(self.prefixes, self.offsets)):
            root = UsdGeom.Xform.Define(self.stage, prefix)
            if not root.GetPrim().GetReferences().AddReference(str(snapshot), "/World"):
                raise RuntimeError("Cannot reference PAP template")
            root.ClearXformOpOrder()
            root.SetResetXformStack(False)
            root.AddTranslateOp().Set(Gf.Vec3d(*offset))
            domes = [p for p in Usd.PrimRange(root.GetPrim()) if p.IsA(UsdLux.DomeLight)]
            if len(domes) != 1:
                raise RuntimeError("PAP template requires exactly one DomeLight")
            if i:
                domes[0].SetActive(False)
            anchors = 0
            for prim in Usd.PrimRange(root.GetPrim()):
                if not prim.IsA(UsdPhysics.Joint):
                    continue
                joint = UsdPhysics.Joint(prim)
                for relation, attribute in ((joint.GetBody0Rel(), joint.GetLocalPos0Attr()),
                                            (joint.GetBody1Rel(), joint.GetLocalPos1Attr())):
                    targets = relation.GetTargets()
                    if targets:
                        if any(not str(path).startswith(prefix+"/") for path in targets):
                            raise RuntimeError("Joint target escaped its PAP clone")
                    else:
                        value = np.asarray(attribute.Get(), dtype=float)
                        if value.shape != (3,):
                            raise RuntimeError("Missing robot world anchor")
                        attribute.Set(Gf.Vec3f(*[float(v) for v in value+offset]))
                        anchors += 1
            if anchors != 1:
                raise RuntimeError(f"Expected one fixed robot world anchor, got {anchors}")
            relocate = lambda path: prefix + str(path).removeprefix("/World")
            self.robot_paths.append(prefix + "/Piper")
            self.object_paths.append(relocate(template.object_paths[0]))
            self.material_paths.append(relocate(template.material_paths[0]))
            camera_paths.extend([relocate(template.camera_paths[name]) for name in ("up", "wrist")])
        Cloner().filter_collisions(str(physics_path), "/World/collisionGroups", self.prefixes, [])
        self.product, self.annotator, self.split_rgb = build_tiled_rgb(camera_paths, tuple(cfg["render_resolution"]))
        rep.orchestrator.set_capture_on_play(False)
        self.world.reset()
        simulation_view = SimulationManager.get_physics_sim_view()
        self.arms = TensorArms(simulation_view, self.robot_paths)
        requested = [get_articulation_root_api_prim_path(path) for path in self.robot_paths]
        self.arm_rows = _path_rows(self.arms.prim_paths, requested)
        self.dofs = np.array([self.arms.get_dof_index(f"joint{i}") for i in range(1, 9)])
        # Match the original runtime controller.set_gains, not merely the USD
        # angular drive attributes (USD degrees vs PhysX radians differ).
        stiffness = np.zeros_like(self.arms.view.get_dof_stiffnesses())
        damping = np.zeros_like(self.arms.view.get_dof_dampings())
        stiffness[:, self.dofs[:6]] = cfg["scene"]["arm_stiffness"]
        damping[:, self.dofs[:6]] = cfg["scene"]["arm_damping"]
        stiffness[:, self.dofs[6:]] = cfg["scene"]["gripper_stiffness"]
        damping[:, self.dofs[6:]] = cfg["scene"]["gripper_damping"]
        rows = np.arange(self.num_envs, dtype=np.uint32)
        self.arms.view.set_dof_stiffnesses(stiffness, rows)
        self.arms.view.set_dof_dampings(damping, rows)
        if (not np.allclose(self.arms.view.get_dof_stiffnesses(), stiffness)
                or not np.allclose(self.arms.view.get_dof_dampings(), damping)):
            raise RuntimeError("PAP vector drive gains differ from the original per-radian runtime gains")
        self.bodies = TensorRigidBodies(simulation_view, self.object_paths)
        self.object_rows = _path_rows(self.bodies.prim_paths, self.object_paths)
        self.world.get_physics_context().set_physx_update_transformations_settings(
            update_to_usd=False, update_velocities_to_usd=False)
        self._guard = PhysXErrorGuard(log)
        self.kin = PiperKinematics(cfg["scene"]["robot_urdf"], tip_offset=.1358)
        self.lower = np.r_[self.kin.lower, 0.]
        self.upper = np.r_[self.kin.upper, cfg["scene"]["gripper_max_width_m"]]
        self.initial = load_policy_initial(cfg["trajectory"], cfg["scene"]["gripper_max_width_m"])["initial"]
        if np.any(self.initial < self.lower-1e-6) or np.any(self.initial > self.upper+1e-6):
            raise ValueError("Initial recorded observation exceeds URDF arm/gripper limits")
        self.commands = self.arms.get_joint_positions()[self.arm_rows].copy()
        self.commands[:, self.dofs] = np.r_[self.initial[:6], self.initial[6]/2, -self.initial[6]/2]
        all_ids = np.arange(self.num_envs)
        self.arms.set_joint_positions(self.commands, self.arm_rows)
        self.arms.set_joint_velocities(np.zeros_like(self.commands), self.arm_rows)
        self.arms.set_joint_position_targets(self.commands, self.arm_rows)
        self._pool = ThreadPoolExecutor(max_workers=cfg["png_workers"], thread_name_prefix="pap_png")
        # One startup-only settling pass precedes every episode. Later indexed
        # reset restores this state; it never advances already-running slots.
        for _ in range(cfg["settling_ticks"]):
            self.world.step(render=False)
            self._guard.check()
        self.settled_joints = self.arms.get_joint_positions()[self.arm_rows].copy()
        self.settled_object_z = self.bodies.get_world_poses()[0][self.object_rows, 2].copy()
        self.ready = np.zeros(self.num_envs, dtype=bool)
        self.ticks = np.zeros(self.num_envs, dtype=np.int64)
        self.last_sent = self._states()
        self.reset_specs = [None] * self.num_envs
        self.outcomes = [None] * self.num_envs
        self.postprocessors = [JointActionPostprocessor(**cfg["smoothing"]) for _ in all_ids]
        self.contact_counts = np.zeros((self.num_envs, 2), dtype=np.int64)
        self.finger_contact = np.zeros(self.num_envs, dtype=bool)
        self.max_ticks = math.floor(cfg["max_episode_seconds"] / cfg["physics_dt"] + 1e-9)
        self._contact_paths, self._contact_pairs = {}, {}
        self._object_env = {path: i for i, path in enumerate(self.object_paths)}
        self.subscription = omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(self._on_contact)
        import carb.settings
        settings = carb.settings.get_settings()
        requested_settings = {"/rtx/post/aa/op": renderer_profile(cfg["renderer_profile"])["anti_aliasing"],
                              **renderer_profile(cfg["renderer_profile"])["history_controls"]}
        for key, value in requested_settings.items():
            settings.set(key, value)
        self._requested_renderer_settings = requested_settings
        self.renderer_settings = {key: settings.get(key) for key in requested_settings}
        if self.renderer_settings != requested_settings:
            raise RuntimeError("PAP renderer did not apply the requested rendering profile")
        warmup = cfg["scene"].get("rendering", {}).get("warmup_frames", 8)
        for _ in range(warmup):
            self._capture(return_tiles=False)
        self._guard.check()

    def _check(self):
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("PAP physics and rendering must run on the owning main thread")
        if self._closed:
            raise RuntimeError("PAP backend is closed")
        self._guard.check()

    def _on_contact(self, headers, data):
        from pxr import PhysicsSchemaTools
        for header in headers:
            if not header.num_contact_data:
                continue
            key = (int(header.actor0), int(header.actor1))
            if key not in self._contact_pairs:
                actors = []
                for value in key:
                    if value not in self._contact_paths:
                        self._contact_paths[value] = str(PhysicsSchemaTools.intToSdfPath(value))
                    actors.append(self._contact_paths[value])
                found = None
                for actor in actors:
                    eid = self._object_env.get(actor)
                    if eid is not None:
                        for finger in (0, 1):
                            if self.robot_paths[eid] + f"/link{7+finger}" in actors:
                                found = (eid, finger)
                self._contact_pairs[key] = found
            pair = self._contact_pairs[key]
            if pair is not None and self.ready[pair[0]]:
                self.contact_counts[pair] += 1
                self.finger_contact[pair[0]] = True

    def _states(self):
        values = self.arms.get_joint_positions()[self.arm_rows][:, self.dofs]
        states = np.c_[values[:, :6], values[:, 6]-values[:, 7]]
        if not np.isfinite(states).all():
            raise RuntimeError("Nonfinite PAP joint state")
        return states

    def _capture(self, *, return_tiles=True):
        import omni.physx
        import omni.replicator.core as rep
        before = self.world.current_time
        started = time.perf_counter()
        self.world.physics_sim_view.update_articulations_kinematic()
        omni.physx.get_physx_interface().update_transformations(False, True, True)
        rep.orchestrator.step(delta_time=0., pause_timeline=False, rt_subframes=self.config["render_subframes"])
        # Kit can override requested AA when Ray Reconstruction becomes active.
        # Record effective values after rendering, and fail if a fast experiment
        # silently switches away from the qualified rendering profile.
        import carb.settings
        settings = carb.settings.get_settings()
        self.renderer_settings = {key: settings.get(key) for key in self._requested_renderer_settings}
        if (self.config["renderer_profile"] == "rl_fast"
                and self.renderer_settings != self._requested_renderer_settings):
            raise RuntimeError("PAP renderer changed the qualified fast profile during capture")
        self._guard.check()
        if self.world.current_time != before:
            raise RuntimeError("PAP image capture advanced physical time")
        images = None
        if return_tiles:
            images = self.split_rgb(self.annotator.get_data())
            shape = (self.config["render_resolution"][1], self.config["render_resolution"][0], 3)
            if len(images) != self.num_envs*2 or any(v.shape != shape or v.dtype != np.uint8 for v in images):
                raise RuntimeError("Missing or malformed live PAP camera tiles")
        self.timings["captures"] += 1
        self.timings["render_seconds"] += time.perf_counter()-started
        return images

    def _state_observations(self, ids):
        """Exact current state without render; only valid for unbootstrapped ends."""
        states = self._states()
        return {i: dict(state=states[i].tolist(), images={},
            task=self.reset_specs[i]["task"], model_inputs=copy.deepcopy(self.reset_specs[i]["model_inputs"]),
            control_state=self.last_sent[i].tolist(), task_features=[]) for i in ids}

    def _begin_render_boundary(self):
        if self.config["renderer_profile"] == "rl_fast":
            import carb.settings
            # reset_renderer_accumulation() alone leaves DLSS recurrent history
            # intact. This plugin setting resets all tiled views, preventing a
            # former cube/arm pose from contaminating the next observation.
            carb.settings.get_settings().set_bool("/rtx-transient/post/dlss/forceParamReset", True)

    def _observations(self, ids, *, reset_history=True, render_frames=None):
        if reset_history:
            self._begin_render_boundary()
            frames = self.config["observation_render_frames"] if render_frames is None else render_frames
            for _ in range(frames-1):
                self._capture(return_tiles=False)
        tiles = self._capture()
        result = self._state_observations(ids)
        started = time.perf_counter()
        encode = partial(_encode_png, image_preprocessing=self.config["image_preprocessing"])
        encoded = list(self._pool.map(encode, [tiles[2*i+j] for i in ids for j in (0, 1)]))
        for row, i in enumerate(ids):
            result[i]["images"] = {"up": encoded[2*row], "wrist": encoded[2*row+1]}
        self.timings["encode_seconds"] += time.perf_counter()-started
        return result

    def describe(self):
        metadata = PAPParallelBackend.describe(self)
        metadata.update(initial_reset="all_envs", scheduling="all_active",
                        execution_horizons=list(range(1, self.config["horizon"]+1)),
                        equal_execution_horizon=True,
                        capabilities=["terminal_state_only", "step_and_reset"],
                        image_preprocessing=copy.deepcopy(self.config["image_preprocessing"]),
                        terminal_observation_modes=["full", "state_only_if_no_bootstrap"])
        metadata["execution"] = dict(backend="isaac_shared_world_tiled", workers=1,
            gpu_ids=[self.config["gpu"]], shared_renderer=True, physics_backend="CPU PhysX tensor views",
            render_resolution=self.config["render_resolution"], camera_count=2*self.num_envs,
            transmitted_image_resolution=list(resized_size(tuple(self.config["render_resolution"]),
                self.config["image_preprocessing"])) if self.config["image_preprocessing"] else self.config["render_resolution"],
            renderer_profile=self.config["renderer_profile"], renderer_settings=self.renderer_settings,
            observation_render_frames=self.config["observation_render_frames"],
            render_subframes=self.config["render_subframes"],
            dlss_history_reset_at_observation_boundaries=self.config["renderer_profile"] == "rl_fast",
            camera_contract="original calibrated 4:3 aperture/offset/pose; no crop or square-frustum substitution",
            reset_contract=RESET_CONTRACT, scheduling="all active environments, equal action prefix length",
            reset_render_frames=self.config["reset_render_frames"],
            terminal_observation="cached at exact terminating tick before peers continue",
            initial_settling_ticks=self.config["settling_ticks"], indexed_reset_physics_ticks=0,
            max_episode_seconds=self.config["max_episode_seconds"], physics_hz=1/self.config["physics_dt"],
            smoothing=self.config["smoothing"], worker_logs=str(self.directory))
        return metadata

    def snapshot(self, env_ids=None):
        """Read-only diagnostics for indexed-reset/clock verification; no render."""
        self._check()
        ids = list(range(self.num_envs)) if env_ids is None else env_ids
        self._ids(ids)
        positions, rotations = self.bodies.get_world_poses()
        q = self.arms.get_joint_positions()[self.arm_rows]
        qv = self.arms.get_joint_velocities()[self.arm_rows]
        velocity = self.bodies.get_velocities()[self.object_rows]
        states = self._states()
        return {str(i): dict(state=states[i].tolist(), joints=q[i].tolist(), joint_velocity=qv[i].tolist(),
            object_position_world=positions[self.object_rows[i]].tolist(),
            object_quaternion_wxyz=rotations[self.object_rows[i]].tolist(), object_velocity=velocity[i].tolist(),
            episode_physics_steps=int(self.ticks[i]), ready=bool(self.ready[i]),
            world_time=float(self.world.current_time)) for i in ids}

    diagnostic_snapshot = snapshot

    def _ids(self, env_ids):
        if (not isinstance(env_ids, list) or not env_ids or len(set(env_ids)) != len(env_ids)
                or any(type(i) is not int or not 0 <= i < self.num_envs for i in env_ids)):
            raise ValueError("Expected distinct valid PAP environment ids")

    def reset(self, env_ids, seed, options=None):
        self._check()
        self._ids(env_ids)
        return self._reset(env_ids, self._prepare_reset(env_ids, seed, options))

    def _prepare_reset(self, env_ids, seed, options):
        """Validate/sample before any world mutation; the result owns its data."""
        _integer(seed, "reset seed", 0)
        if options is not None and not isinstance(options, dict):
            raise ValueError("reset options must be a mapping")
        options = copy.deepcopy(options if options is not None else {})
        by_env = options.pop("by_env", {})
        if not isinstance(by_env, dict):
            raise ValueError("reset options.by_env must be a mapping")
        return {i: sample_reset(self.config, seed, i, {**options, **by_env.get(str(i), by_env.get(i, {}))})
                for i in env_ids}

    def _reset(self, env_ids, specs, *, capture_images=True):
        from pxr import Gf, UsdShade
        from pressb.tasks.pap.task import PickPlaceOutcome
        start = time.perf_counter()
        peers = [i for i in range(self.num_envs) if i not in env_ids]
        before = self.snapshot(peers) if peers else None
        ids = np.asarray(env_ids)
        q = self.settled_joints[ids].copy()
        self.arms.set_joint_positions(q, self.arm_rows[ids])
        self.arms.set_joint_velocities(np.zeros_like(q), self.arm_rows[ids])
        self.arms.set_joint_position_targets(q, self.arm_rows[ids])
        self.commands[ids] = q
        positions, quaternions = [], []
        geometry = self.config["objects"][0]["geometry"]
        cavity = self.source_scene["box_inside"]
        for i in env_ids:
            spec = specs[i]
            offset = self.offsets[i]
            position = np.r_[np.asarray(spec["center_xy"])+self.source_base[:2]+offset[:2], self.settled_object_z[i]]
            angle = math.radians(spec["yaw_deg"])/2
            positions.append(position)
            quaternions.append([math.cos(angle), 0., 0., math.sin(angle)])
            for suffix, name in (("Mdl", "diffuse_color_constant"), ("Preview", "diffuseColor")):
                UsdShade.Shader(self.stage.GetPrimAtPath(self.material_paths[i]+"/"+suffix)).GetInput(name).Set(
                    Gf.Vec3f(*spec["color_rgb"]))
            box = dict(center_world=(np.asarray(cavity["origin_world"])+offset).tolist(), yaw_deg=cavity["yaw_deg"],
                inner_size_xy=(np.asarray(cavity["max_local"])-cavity["min_local"])[:2].tolist(),
                floor_z_world=cavity["origin_world"][2]+cavity["min_local"][2],
                rim_z_world=cavity["origin_world"][2]+cavity["max_local"][2])
            self.outcomes[i] = PickPlaceOutcome(geometry, self.config["scene"]["table_height"], box,
                                               **self.config["outcome"])
            self.reset_specs[i] = spec
            self.ticks[i] = 0
            self.contact_counts[i] = 0
            self.finger_contact[i] = False
        self.bodies.set_world_poses(np.asarray(positions), np.asarray(quaternions), self.object_rows[ids])
        self.bodies.set_velocities(np.zeros((len(ids), 6)), self.object_rows[ids])
        self.last_sent[ids] = self._states()[ids]
        for i in env_ids:
            self.postprocessors[i].reset(self.last_sent[i])
        # Reset is a discontinuity for temporal renderer history, but not a
        # physical step. Preserve the calibrated PAP warmup by default; callers
        # may explicitly lower reset_render_frames for a throughput experiment.
        import omni.usd
        omni.usd.get_context().reset_renderer_accumulation()
        if capture_images:
            self._begin_render_boundary()
            for _ in range(self.config["reset_render_frames"]-1):
                self._capture(return_tiles=False)
            observations = self._observations(env_ids, reset_history=False)
        else:
            observations = self._state_observations(env_ids)
        if peers and self.snapshot(peers) != before:
            raise RuntimeError("Indexed PAP reset changed a peer's physics state or clock")
        self.ready[ids] = True
        self.timings["reset_seconds"] += time.perf_counter()-start
        return [dict(env_id=i, observation=observations[i], info=dict(reset=specs[i], success=False,
            object_geometry=geometry, object_position_world=positions[row].tolist(),
            settling_physics_steps=0, startup_settling_physics_steps=self.config["settling_ticks"],
            episode_physics_steps=0, elapsed_control_steps=0., reset_contract=RESET_CONTRACT,
            smoothing=self.postprocessors[i].summary())) for row, i in enumerate(env_ids)]

    def step(self, items, *, terminal_observation_mode="full"):
        return self._step(items, terminal_observation_mode=terminal_observation_mode)

    def _step(self, items, *, terminal_observation_mode="full", defer_live_images=False):
        self._check()
        if terminal_observation_mode not in ("full", "state_only_if_no_bootstrap"):
            raise ValueError("Unsupported terminal_observation_mode")
        if not isinstance(items, list) or any(not isinstance(v, dict) for v in items):
            raise ValueError("step requires action mappings")
        ids = [v["env_id"] for v in items]
        self._ids(ids)
        if set(ids) != set(np.flatnonzero(self.ready)):
            raise ValueError("Shared-World PAP requires every active environment in each step")
        commands = {}
        for item in items:
            action = np.asarray(item["actions"])
            if (action.ndim != 2 or action.shape[1] != 7 or not 1 <= len(action) <= self.config["horizon"]
                    or action.dtype.kind not in "fiu" or not np.isfinite(action).all()
                    or np.any(action < self.lower-1e-6) or np.any(action > self.upper+1e-6)):
                raise ValueError("PAP executable actions must respect SI joint/gripper bounds")
            commands[item["env_id"]] = action
        if len({len(v) for v in commands.values()}) != 1:
            raise ValueError("Shared-World PAP requires equal action prefix lengths")
        cfg, start = self.config, time.perf_counter()
        ticks_before = self.ticks.copy()
        applied = {i: [] for i in ids}
        results, observations, endings, terminal_positions, terminal_contacts = {}, {}, {}, {}, {}
        for action_index in range(len(next(iter(commands.values())))):
            live = [i for i in ids if self.ready[i]]
            if not live:
                break
            for i in live:
                self.last_sent[i] = self.postprocessors[i].apply(commands[i][action_index])
                applied[i].append(self.last_sent[i].tolist())
                self.commands[i, self.dofs] = np.r_[self.last_sent[i, :6], self.last_sent[i, 6]/2, -self.last_sent[i, 6]/2]
            self.arms.set_joint_position_targets(self.commands[live], self.arm_rows[live])
            for _ in range(cfg["physics_ticks_per_action"]):
                live = [i for i in ids if self.ready[i]]
                if not live:
                    break
                self.finger_contact[:] = False
                tick_start = time.perf_counter()
                self.world.step(render=False)
                self._guard.check()
                self.timings["physics_seconds"] += time.perf_counter()-tick_start
                self.timings["world_ticks"] += 1
                self.ticks[live] += 1
                position, quaternion = self.bodies.get_world_poses()
                position, quaternion = position[self.object_rows], quaternion[self.object_rows]
                velocity = self.bodies.get_velocities()[self.object_rows]
                states = self._states()
                if not np.isfinite(np.r_[position.ravel(), quaternion.ravel(), velocity.ravel()]).all():
                    raise RuntimeError("Nonfinite PAP object physics state")
                ended = []
                for i in live:
                    result = self.outcomes[i].update(self.ticks[i]*cfg["physics_dt"], position[i], quaternion[i],
                        velocity[i], float(states[i, 6]), finger_contact=bool(self.finger_contact[i]))
                    results[i] = result
                    success = bool(result["success"] and all(self.contact_counts[i]))
                    dropped = cfg["terminate_on_drop"] and position[i, 2] < cfg["scene"]["table_height"]-cfg["drop_below_table_m"]
                    reason = "success" if success else "object_dropped" if dropped else "timeout" if self.ticks[i] >= self.max_ticks else None
                    if reason:
                        self.ready[i] = False
                        endings[i] = reason
                        terminal_positions[i] = position[i].tolist()
                        terminal_contacts[i] = self.contact_counts[i].copy()
                        ended.append(i)
                if ended:
                    # Capture before another physics tick; dead objects may
                    # subsequently settle while peers run, but are never used
                    # as another transition until explicit indexed reset.
                    if terminal_observation_mode == "state_only_if_no_bootstrap":
                        observations.update(self._state_observations(ended))
                    else:
                        observations.update(self._observations(ended))
        live = [i for i in ids if self.ready[i]]
        if live:
            observations.update(self._state_observations(live) if defer_live_images else self._observations(live))
        positions = self.bodies.get_world_poses()[0][self.object_rows]
        rows = []
        for i in ids:
            reason = endings.get(i)
            success = reason == "success"
            terminated = reason is not None and reason != "timeout"
            elapsed_ticks = int(self.ticks[i]-ticks_before[i])
            controls = elapsed_ticks/cfg["physics_ticks_per_action"]
            reward = cfg["reward"]["success"] if success else cfg["reward"]["failure"] if terminated else 0.
            reward *= cfg["gamma"] ** ((elapsed_ticks-1)/cfg["physics_ticks_per_action"])
            contacts = terminal_contacts.get(i, self.contact_counts[i])
            rows.append(dict(env_id=i, observation=observations[i], reward=float(reward),
                discount=cfg["gamma"] ** controls, terminated=terminated, truncated=reason == "timeout",
                info=dict(success=success, termination_reason=reason, physics_steps=elapsed_ticks,
                    elapsed_control_steps=controls, episode_physics_steps=int(self.ticks[i]),
                    episode_seconds=self.ticks[i]*cfg["physics_dt"], physical_outcome=results[i],
                    contact_counts={"link7": int(contacts[0]), "link8": int(contacts[1])},
                    applied_actions=applied[i], requested_action_count=len(commands[i]),
                    object_position_world=terminal_positions.get(i, positions[i].tolist()),
                    object_teleport_after_reset=False)))
            if reason is not None and terminal_observation_mode == "state_only_if_no_bootstrap":
                rows[-1]["terminal_observation_omitted"] = True
        self.timings["step_seconds"] += time.perf_counter()-start
        return rows

    def step_and_reset(self, items, *, reset_seed, reset_options, max_active_envs):
        """Atomic rollout/reset with a single shared render boundary.

        Terminal rows keep their exact pre-reset state and omit RGB. Surviving
        transitions and newly reset episodes receive separate observations from
        the same zero-time render. Only callers with zero bootstrap on every
        terminal/truncated transition may use this explicit fast capability.
        """
        self._check()
        if (not isinstance(items, list) or any(not isinstance(row, dict) or "env_id" not in row for row in items)):
            raise ValueError("step requires action mappings")
        ids = [row["env_id"] for row in items]
        self._ids(ids)
        if type(max_active_envs) is not int or not 0 <= max_active_envs <= self.num_envs:
            raise ValueError("max_active_envs must be an integer in [0, num_envs]")
        # Every active slot might terminate. Validate all possible reset data
        # before executing any action, even if this call eventually resets none.
        specs = self._prepare_reset(ids, reset_seed, reset_options)
        started = time.perf_counter()
        rows = self._step(items, terminal_observation_mode="state_only_if_no_bootstrap", defer_live_images=True)
        done = [row["env_id"] for row in rows if row["terminated"] or row["truncated"]]
        live = [row["env_id"] for row in rows if not (row["terminated"] or row["truncated"])]
        reset_ids = done[:max(0, max_active_envs-len(live))]
        reset_rows = self._reset(reset_ids, specs, capture_images=False) if reset_ids else []
        active = sorted(live+reset_ids)
        if active:
            frames = max(self.config["observation_render_frames"],
                         self.config["reset_render_frames"] if reset_ids else 0)
            observations = self._observations(active, render_frames=frames)
            for row in rows:
                if row["env_id"] in live:
                    row["observation"] = observations[row["env_id"]]
            for row in reset_rows:
                row["observation"] = observations[row["env_id"]]
        self.timings["fused_step_reset_seconds"] = self.timings.get("fused_step_reset_seconds", 0.) + time.perf_counter()-started
        self.timings["fused_steps"] = self.timings.get("fused_steps", 0) + 1
        return {"items": rows, "reset_items": reset_rows}

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.subscription = None
        if self._pool is not None:
            self._pool.shutdown(wait=True)
        if self._guard is not None:
            self._guard.close()
        if getattr(self, "annotator", None) is not None:
            try:
                self.annotator.detach([self.product.path])
                self.product.destroy()
            except Exception:
                pass
        if self.app is not None:
            self.app.app.post_quit(0)
            self.app.close()


def create_backend(config):
    return PAPVectorBackend(config)
