"""Shared Isaac lifecycle, scene cloning, contacts and tiled image sampling.

No task or model packages are imported here. The injected task authors one scene
and declares its robot/cameras; the runtime owns every clone and physics tick.
"""
from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import partial
import io
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from .image_transport import encode_preprocessed_image, resized_size, validate_image_preprocessing
from .physics import PhysXErrorGuard, TensorRigidBodies, path_rows


def renderer_profile(name):
    if name in ("calibrated", "pap_calibrated"):
        return {"anti_aliasing": 1, "history_controls": {}}
    if name != "rl_fast":
        raise ValueError("renderer_profile must be calibrated, pap_calibrated or rl_fast")
    return {"anti_aliasing": 3, "reset_dlss_history": True, "history_controls": {
        "/rtx-transient/dldenoiser/enabled": True,
        "/rtx/indirectDiffuse/denoiser/temporal/enabled": False,
        "/rtx/directLighting/sampledLighting/ris/enableSpatioTemporalRis": False,
        **{f"/rtx/lightspeed/NRD_Reblur{channel}/{key}": 0
           for channel in ("Diffuse", "Specular") for key in ("maxAccumulatedFrameNum", "maxFastAccumulatedFrameNum")},
        **{f"/rtx/lightspeed/NRD_ReLAX/{key}": 0 for key in
           ("diffuseHistoryFrames", "diffuseFastHistoryFrames", "specularHistoryFrames", "specularFastHistoryFrames")}}}


def encode_png(rgb, image_preprocessing=None):
    if image_preprocessing is not None:
        return encode_preprocessed_image(rgb, image_preprocessing)
    from PIL import Image
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise RuntimeError("Expected uint8 RGB image")
    stream = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb)).save(stream, format="PNG", compress_level=1)
    return {"encoding": "png_base64", "data": base64.b64encode(stream.getvalue()).decode("ascii")}


def _localize_reset_stacks(stage, root_path):
    """Preserve authored world poses while making descendant transforms cloneable."""
    from pxr import Usd, UsdGeom
    cache = UsdGeom.XformCache()
    edits = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        if str(prim.GetPath()) == root_path or not prim.IsA(UsdGeom.Xformable):
            continue
        xf = UsdGeom.Xformable(prim)
        if xf.GetResetXformStack():
            world = cache.GetLocalToWorldTransform(prim)
            parent = cache.GetLocalToWorldTransform(prim.GetParent())
            edits.append((xf, world * parent.GetInverse()))
    for xf, local in edits:
        xf.ClearXformOpOrder()
        xf.SetResetXformStack(False)
        xf.AddTransformOp(opSuffix="parallelLocal").Set(local)


class IsaacParallelWorld:
    def __init__(self, task, backend):
        self.task, self.backend, self.config = task, backend, backend.execution
        self.num_envs = backend.num_envs
        self.app = self.world = self.stage = self.robot = self.template = None
        self.subscription = self._guard = self._pool = None
        self._blas_limit = None
        self.blas_threadpools = []
        self.product = self.annotator = None
        self._closed = False
        self._contact_paths = {}
        self._contact_error = None
        self._rigid_views = []

    def initialize(self):
        cfg = self.config
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        os.environ.setdefault("PXR_WORK_THREAD_LIMIT", str(cfg["cpu_threads"]))
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(name, "1")
        directory = Path(cfg.get("log_dir", "outputs/parallel_simulation")).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix="run_", dir=directory))
        (self.directory/"execution.json").write_text(json.dumps(cfg, indent=2, default=str)+"\n")
        profile = renderer_profile(cfg["renderer_profile"])
        aa = cfg.get("anti_aliasing", profile["anti_aliasing"])
        if type(aa) is not int or aa not in (0, 1, 2, 3, 4):
            raise ValueError("anti_aliasing must be a supported integer mode")
        if cfg.get("gpu_dynamics", False):
            raise ValueError("Shared parallel engine currently qualifies CPU PhysX tensor views only")
        self.image_preprocessing = validate_image_preprocessing(cfg["image_preprocessing"])
        self.log_path = self.directory/"kit.log"
        from isaacsim import SimulationApp
        self.app = SimulationApp(dict(headless=cfg["headless"], active_gpu=cfg["gpu"], physics_gpu=cfg["gpu"],
            multi_gpu=False, renderer="RayTracedLighting", anti_aliasing=aa, fast_shutdown=True,
            disable_viewport_updates=True, limit_cpu_threads=cfg["cpu_threads"],
            extra_args=["--/app/asyncRendering=false", "--/rtx/ecoMode/enabled=false",
                f"--/log/file={self.log_path}", "--/log/fileFlushLevel=verbose", "--/log/fileAppend=false",
                "--/log/async=false", f"--/plugins/carb.tasking.plugin/threadCount={cfg['cpu_threads']}",
                f"--/plugins/omni.tbb.globalcontrol/maxThreadCount={cfg['cpu_threads']}",
                "--/persistent/physics/numThreads=4", "--/validate/p2p/enabled=false", "--/validate/iommu/enabled=false"]))
        self.template = self.task.build_template()
        self.world, self.stage = self.template.world, self.template.stage
        self.camera_names = list(self.template.camera_paths)
        if not self.camera_names or len(set(self.camera_names)) != len(self.camera_names):
            raise ValueError("Template must declare uniquely named cameras")
        expected_cameras = self.task.metadata()["observation"]["cameras"]
        if self.camera_names != expected_cameras:
            raise ValueError("Template camera ordering differs from task observation contract")
        from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics
        from isaacsim.core.cloner import Cloner
        from isaacsim.core.simulation_manager import SimulationManager
        import omni.physx
        import omni.replicator.core as rep
        from pressb.sensors.tiled import build_tiled_rgb
        source_root = self.template.root_path
        if source_root != "/World":
            raise ValueError("Native parallel templates must use /World as their scene root")
        physics_path = str(self.world.get_physics_context().prim_path)
        if physics_path == source_root or physics_path.startswith(source_root+"/"):
            raise ValueError("Physics scene must be outside the cloned scene root")
        _localize_reset_stacks(self.stage, source_root)
        snapshot = self.directory/"template.usdc"
        self.stage.Flatten().Export(str(snapshot))
        self.world.scene.clear(registry_only=True)
        self.stage.RemovePrim(source_root)
        UsdGeom.Xform.Define(self.stage, "/World")
        UsdGeom.Scope.Define(self.stage, "/World/envs")
        self.prefixes = [f"/World/envs/env_{i}" for i in range(self.num_envs)]
        columns = math.ceil(math.sqrt(self.num_envs))
        self.offsets = np.array([[i//columns*cfg["spacing_m"], i%columns*cfg["spacing_m"], 0.] for i in range(self.num_envs)])
        self.robot_paths, camera_paths = [], []
        for eid, (prefix, offset) in enumerate(zip(self.prefixes, self.offsets)):
            root = UsdGeom.Xform.Define(self.stage, prefix)
            if not root.GetPrim().GetReferences().AddReference(str(snapshot), source_root):
                raise RuntimeError("Cannot reference authored scene template")
            root.ClearXformOpOrder()
            root.SetResetXformStack(False)
            root.AddTranslateOp().Set(Gf.Vec3d(*offset))
            for prim in Usd.PrimRange(root.GetPrim()):
                if eid and prim.IsA(UsdLux.DomeLight):
                    prim.SetActive(False)
                if not prim.IsA(UsdPhysics.Joint):
                    continue
                joint = UsdPhysics.Joint(prim)
                for relation, attribute in ((joint.GetBody0Rel(), joint.GetLocalPos0Attr()),
                                            (joint.GetBody1Rel(), joint.GetLocalPos1Attr())):
                    targets = relation.GetTargets()
                    if targets:
                        if any(not str(path).startswith(prefix+"/") for path in targets):
                            raise RuntimeError("Joint relationship escaped its environment clone")
                    else:
                        value = attribute.Get()
                        value = np.zeros(3) if value is None else np.asarray(value, dtype=float)
                        if value.shape != (3,):
                            raise RuntimeError("Invalid world joint anchor")
                        attribute.Set(Gf.Vec3f(*[float(x) for x in value+offset]))
            self.robot_paths.append(self.relocate(eid, self.template.robot_path))
            camera_paths.extend(self.relocate(eid, self.template.camera_paths[name]) for name in self.camera_names)
        Cloner().filter_collisions(physics_path, "/World/collisionGroups", self.prefixes, [])
        self.product, self.annotator, self.split_rgb = build_tiled_rgb(camera_paths, tuple(cfg["render_resolution"]))
        rep.orchestrator.set_capture_on_play(False)
        self.world.reset()
        self.simulation_view = SimulationManager.get_physics_sim_view()
        self._guard = PhysXErrorGuard(self.log_path)
        self.robot = self.task.create_robot(self)
        self.task.bind(self)
        self._configure_blas()
        self.subscription = omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(self._on_contact)
        self.world.get_physics_context().set_physx_update_transformations_settings(
            update_to_usd=False, update_velocities_to_usd=False)
        for _ in range(cfg["settling_ticks"]):
            self.world.step(render=False)
            self.check()
        self.robot.cache_initial_state()
        self.task.after_settle()
        self._pool = ThreadPoolExecutor(max_workers=cfg["png_workers"], thread_name_prefix="parallel_png")
        import carb.settings
        settings = carb.settings.get_settings()
        self.requested_renderer_settings = {"/rtx/post/aa/op": aa, **profile["history_controls"],
                                           **cfg.get("renderer_settings", {})}
        for key, value in self.requested_renderer_settings.items():
            settings.set(key, value)
        self.renderer_settings = {key: settings.get(key) for key in self.requested_renderer_settings}
        if self.renderer_settings != self.requested_renderer_settings:
            raise RuntimeError("Renderer did not apply the declared settings")
        for _ in range(cfg.get("warmup_frames", 8)):
            self._capture(return_tiles=False)
        self.check()

    def _configure_blas(self):
        """Limit already-loaded BLAS pools used by small batched control solves.

        NumPy and Kit load native libraries before environment defaults can take
        effect. Apply the limit after Kit and robot/task imports, retain the
        controller for shutdown restoration, and record the effective settings.
        """
        from threadpoolctl import threadpool_info, threadpool_limits
        if self._blas_limit is not None:
            raise RuntimeError("BLAS limits are already configured")
        self._blas_limit = threadpool_limits(limits=1, user_api="blas")
        self.blas_threadpools = deepcopy(threadpool_info())
        if any(pool.get("num_threads") != 1 for pool in self.blas_threadpools
               if pool.get("user_api") == "blas"):
            raise RuntimeError("BLAS thread pools did not apply the one-thread control limit")

    def relocate(self, env_id, path):
        root = self.template.root_path
        path = str(path)
        if path != root and not path.startswith(root+"/"):
            raise ValueError("Scene declaration must refer to a prim inside the template root")
        return self.prefixes[env_id]+path[len(root):]

    def rigid_bodies(self, template_paths):
        if not template_paths:
            raise ValueError("Rigid body view requires at least one declared body")
        paths = [self.relocate(i, path) for i in range(self.num_envs) for path in template_paths]
        view = TensorRigidBodies(self.simulation_view, paths)
        rows = path_rows(view.prim_paths, paths).reshape(self.num_envs, len(template_paths))
        self._rigid_views.append(view)
        return view, rows

    def _contact_path(self, value):
        from pxr import PhysicsSchemaTools
        value = int(value)
        if value not in self._contact_paths:
            self._contact_paths[value] = str(PhysicsSchemaTools.intToSdfPath(value)) if value else ""
        return self._contact_paths[value]

    def _on_contact(self, headers, data):
        try:
            for header in headers:
                if not header.num_contact_data:
                    continue
                paths = [self._contact_path(getattr(header, name, 0)) for name in ("actor0", "actor1", "collider0", "collider1")]
                env_ids = set()
                for path in paths:
                    if path.startswith("/World/envs/env_"):
                        token = path.split("/", 4)[3]
                        env_ids.add(int(token.removeprefix("env_")))
                if len(env_ids) > 1:
                    raise RuntimeError("Contact escaped environment collision isolation")
                if not env_ids:
                    continue
                eid = env_ids.pop()
                if not self.backend.ready[eid]:
                    continue
                prefix = self.prefixes[eid]
                local = [self.template.root_path+p[len(prefix):] if p == prefix or p.startswith(prefix+"/") else p for p in paths]
                contacts = data[header.contact_data_offset:header.contact_data_offset+header.num_contact_data]
                self.task.on_contact(eid, local, contacts)
        except BaseException as exc:
            # Native callback exceptions may otherwise be swallowed by PhysX.
            self._contact_error = exc

    def check(self):
        if self._contact_error is not None:
            raise RuntimeError(f"Contact processing failed: {self._contact_error}") from self._contact_error
        if self._guard is not None:
            self._guard.check()

    def tick(self, live_ids):
        self.world.step(render=False)
        self.check()

    def publish_transforms(self):
        import omni.physx
        before = self.world.current_time
        self.world.physics_sim_view.update_articulations_kinematic()
        omni.physx.get_physx_interface().update_transformations(False, True, True)
        if self.world.current_time != before:
            raise RuntimeError("Publishing measured transforms advanced physics")

    def reset_render_history(self):
        import omni.usd
        omni.usd.get_context().reset_renderer_accumulation()

    def _capture(self, *, return_tiles=True, subframes=None):
        import omni.replicator.core as rep
        import carb.settings
        before = self.world.current_time
        started = time.perf_counter()
        self.publish_transforms()
        rep.orchestrator.step(delta_time=0., pause_timeline=False,
                             rt_subframes=self.config["render_subframes"] if subframes is None else subframes)
        settings = carb.settings.get_settings()
        self.renderer_settings = {key: settings.get(key) for key in self.requested_renderer_settings}
        if self.config["renderer_profile"] == "rl_fast" and self.renderer_settings != self.requested_renderer_settings:
            raise RuntimeError("Renderer changed the qualified fast profile")
        self.check()
        if self.world.current_time != before:
            raise RuntimeError("Camera capture advanced physical time")
        images = None
        if return_tiles:
            images = self.split_rgb(self.annotator.get_data())
            shape = (self.config["render_resolution"][1], self.config["render_resolution"][0], 3)
            if len(images) != self.num_envs*len(self.camera_names) or any(v.shape != shape or v.dtype != np.uint8 for v in images):
                raise RuntimeError("Missing or malformed live camera tiles")
        self.backend.timings["captures"] += 1
        self.backend.timings["render_seconds"] += time.perf_counter()-started
        return images

    def capture(self, ids, render_frames, *, reset_history=True):
        import carb.settings
        if reset_history and self.config["renderer_profile"] == "rl_fast":
            carb.settings.get_settings().set_bool("/rtx-transient/post/dlss/forceParamReset", True)
        # Optional task-authored reset subframes change rendering only, not time.
        subframes = self.config.get("reset_render_subframes") if getattr(self, "_reset_capture", False) else None
        for _ in range(render_frames-1):
            self._capture(return_tiles=False, subframes=subframes)
        tiles = self._capture(subframes=subframes)
        count = len(self.camera_names)
        selected = [tiles[count*i+j] for i in ids for j in range(count)]
        output = self.config.get("image_output_resolution")
        started = time.perf_counter()
        if output is not None:
            from PIL import Image
            selected = [np.asarray(Image.fromarray(rgb).resize(tuple(output), Image.Resampling.BILINEAR)) for rgb in selected]
        encoded = list(self._pool.map(partial(encode_png, image_preprocessing=self.image_preprocessing), selected))
        self.backend.timings["encode_seconds"] += time.perf_counter()-started
        return {eid: {name: encoded[row*count+j] for j, name in enumerate(self.camera_names)} for row, eid in enumerate(ids)}

    def execution_metadata(self):
        cfg = self.config
        output = cfg.get("image_output_resolution") or cfg["render_resolution"]
        return dict(shared_renderer=True, physics_backend="CPU PhysX tensor views",
            blas_thread_limit=1, blas_threadpools=deepcopy(self.blas_threadpools),
            camera_count=self.num_envs*len(self.camera_names), render_resolution=cfg["render_resolution"],
            transmitted_image_resolution=list(resized_size(tuple(output), self.image_preprocessing)) if self.image_preprocessing else list(output),
            renderer_profile=cfg["renderer_profile"], renderer_settings=deepcopy(self.renderer_settings),
            observation_render_frames=cfg["observation_render_frames"], reset_render_frames=cfg["reset_render_frames"],
            render_subframes=cfg["render_subframes"], camera_contract="task-authored intrinsics/extrinsics; aspect-preserving render",
            worker_logs=str(self.directory))

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.subscription = None
            if self._pool is not None:
                self._pool.shutdown(wait=True)
            self.task.close()
            if self.robot is not None:
                self.robot.close()
            if self.annotator is not None and self.product is not None:
                self.annotator.detach([self.product])
            if self.product is not None:
                self.product.destroy()
        finally:
            try:
                if self.world is not None:
                    self.world.stop()
                    self.world.scene.clear(registry_only=True)
                    from isaacsim.core.api import World
                    World.clear_instance()
                if self._guard is not None:
                    self._guard.close()
            finally:
                try:
                    if self._blas_limit is not None:
                        controller, self._blas_limit = self._blas_limit, None
                        controller.restore_original_limits()
                finally:
                    if self.app is not None:
                        self.app.close()
