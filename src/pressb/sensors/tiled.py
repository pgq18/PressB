"""Task-independent tiled RGB capture for existing simulation cameras."""

from __future__ import annotations

from functools import partial
import math

import numpy as np


def split_tiled_rgb(buffer, num_cameras: int, resolution=(640, 480)) -> list[np.ndarray]:
    """Copy RGB tiles in the original camera-list order (left-to-right, then down).

    Matches IsaacLab 2.0.2 ``reshape_tiled_image`` and installed Kit 106.5:
    columns=ceil(sqrt(N)), rows=ceil(N/columns). Camera i is at
    (row=i//columns, column=i%columns); no vertical flip is applied.
    Copies prevent the following render from mutating frames queued for writing.
    """
    if not isinstance(num_cameras, int) or num_cameras <= 0:
        raise ValueError("num_cameras must be a positive integer")
    width, height = (int(value) for value in resolution)
    if width <= 0 or height <= 0:
        raise ValueError("Tile resolution must be positive")
    if isinstance(buffer, dict):
        buffer = buffer["data"]
    data = np.asarray(buffer)
    columns = math.ceil(math.sqrt(num_cameras))
    rows = math.ceil(num_cameras / columns)
    if (data.ndim != 3 or data.shape[:2] != (rows * height, columns * width)
            or data.shape[2] not in (3, 4) or data.dtype != np.uint8):
        raise ValueError(f"Invalid tiled RGB buffer {data.shape}/{data.dtype}; expected "
                         f"({rows * height}, {columns * width}, 3 or 4) uint8")
    return [np.array(data[(i // columns) * height:(i // columns + 1) * height,
                          (i % columns) * width:(i % columns + 1) * width, :3], copy=True, order="C")
            for i in range(num_cameras)]


def build_tiled_rgb(camera_paths, resolution=(640, 480)):
    """Return (render_product, CPU_RGB_annotator, split_fn) for existing cameras.

    No Core Camera wrappers or additional render products are created. Caller
    renders once, reads ``annotator.get_data()``, then invokes ``split_fn``.
    ``camera_paths`` order is preserved by the render product relationship.
    """
    import omni.replicator.core as rep
    import omni.usd
    from pxr import UsdGeom

    paths = [str(path) for path in camera_paths]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Expected a nonempty list of distinct camera paths")
    dimensions = np.asarray(resolution)
    if (dimensions.shape != (2,) or not np.isfinite(dimensions).all()
            or np.any(dimensions <= 0) or np.any(dimensions != np.floor(dimensions))):
        raise ValueError("Camera resolution must contain two positive integers")
    resolution = tuple(int(value) for value in dimensions)
    stage = omni.usd.get_context().get_stage()
    for path in paths:
        if not stage.GetPrimAtPath(path).IsA(UsdGeom.Camera):
            raise ValueError(f"Missing Camera prim: {path}")
    render_product = rep.create.render_product_tiled(cameras=paths, tile_resolution=resolution, force_new=True)
    authored = [str(path) for path in stage.GetPrimAtPath(render_product.path).GetRelationship("camera").GetTargets()]
    if authored != paths:
        raise RuntimeError("Tiled render product changed the requested camera order")
    annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu", do_array_copy=True)
    annotator.attach([render_product.path])
    return render_product, annotator, partial(split_tiled_rgb, num_cameras=len(paths), resolution=resolution)
