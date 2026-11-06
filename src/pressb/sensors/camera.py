"""Calibrated USD cameras with lazy simulator imports."""


def set_intrinsics(prim, spec):
    from pxr import UsdGeom
    camera = UsdGeom.Camera(prim)
    width, height = spec["resolution"]
    focal = 2.
    horizontal = focal * width / spec["fx"]
    vertical = focal * height / spec["fy"]
    camera.CreateFocalLengthAttr(focal)
    camera.CreateHorizontalApertureAttr(horizontal)
    camera.CreateVerticalApertureAttr(vertical)
    camera.CreateHorizontalApertureOffsetAttr((width / 2 - spec.get("cx", width/2)) / width * horizontal)
    camera.CreateVerticalApertureOffsetAttr((spec.get("cy", height/2) - height/2) / height * vertical)
    camera.CreateFStopAttr(0.)


def make_camera(stage, path, spec, base):
    import numpy as np
    from pxr import UsdGeom, Gf
    prim = UsdGeom.Camera.Define(stage, path)
    if "base_from_camera_usd" in spec:
        transform = np.asarray(spec["base_from_camera_usd"], dtype=float).copy()
        transform[:3, 3] += base
        matrix = Gf.Matrix4d(*transform.T.reshape(-1).tolist())
    else:
        eye, target = np.asarray(spec["eye_base_m"]) + base, np.asarray(spec["target_base_m"]) + base
        matrix = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse()
    prim.AddTransformOp().Set(matrix)
    prim.CreateClippingRangeAttr(Gf.Vec2f(.005, 20.))
    set_intrinsics(prim.GetPrim(), spec)
    return str(prim.GetPath())


