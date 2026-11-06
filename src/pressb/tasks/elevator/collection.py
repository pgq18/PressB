"""Frozen elevator collection rendering, sampling and raw-data contract."""
from __future__ import annotations

import math

RAW_SCHEMA_VERSION = 11
LIGHTING_POLICY = {
    "global_dome": "deactivate_duplicate_dome_prims",
    "active_dome_count": 1,
    "shared_dome_environment": 0,
    "local_rect_lights_per_environment": 2,
}
DATASET_ANTI_ALIASING = 2  # FXAA: no temporal anti-aliasing history across samples.
LIGHT_SETTLE_SUBFRAMES = 16
# Kit 107 needs additional captures to drain visible edge-light history.
# All settling captures keep physics frozen and are recorded in the manifest.
LIGHT_SETTLE_CAPTURES = 16
RENDER_EXPOSURE_CONTROLS = {
    "/rtx/post/histogram/enabled": False,
    "/rtx/post/tonemap/filmIso": 100.,
    "/rtx/post/tonemap/exposureTime": .02,
    "/rtx/post/tonemap/fNumber": 5.,
    "/rtx/post/tonemap/op": 6,
}
# NRD 3.2.3 permits zero history. Keep spatial denoising, but do not mix a
# previous illuminated frame into the current unlit observation.
RENDER_HISTORY_CONTROLS = {
    "/rtx/indirectDiffuse/denoiser/temporal/enabled": False,
    "/rtx/directLighting/sampledLighting/ris/enableSpatioTemporalRis": False,
    **{f"/rtx/lightspeed/NRD_Reblur{channel}/{key}": 0
       for channel in ("Diffuse", "Specular")
       for key in ("maxAccumulatedFrameNum", "maxFastAccumulatedFrameNum")},
    **{f"/rtx/lightspeed/NRD_ReLAX/{key}": 0 for key in
       ("diffuseHistoryFrames", "diffuseFastHistoryFrames", "specularHistoryFrames", "specularFastHistoryFrames")},
}
SOURCE_NAMES = ("frames.npz", "physics.npz", "wrist.mp4", "global.mp4")
START_SPEED_TOLERANCE_RAD_S = .02


def capture_stride_for(fps, physics_dt=1 / 120):
    """Keep every RGB/state sample on an exact 120 Hz physics boundary."""
    if (type(fps) is not int or fps <= 0 or 120 % fps
            or not math.isfinite(physics_dt) or abs(physics_dt - 1 / 120) > 1e-12):
        raise ValueError("Require integer FPS dividing 120 and physics_dt=1/120 seconds")
    return 120 // fps


