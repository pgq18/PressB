"""Calibrated USD cameras with lazy simulator imports."""


def camera_intrinsics_from_usd(prim, resolution):
    """Return the authored pinhole K, including an off-centre principal point.

    Isaac Camera.get_intrinsics_matrix() assumes a centred principal point;
    read USD directly so measured RealSense calibration remains observable.
    """
    import numpy as np
    from pxr import UsdGeom

    camera = UsdGeom.Camera(prim)
    width, height = resolution
    focal = camera.GetFocalLengthAttr().Get()
    horizontal = camera.GetHorizontalApertureAttr().Get()
    vertical = camera.GetVerticalApertureAttr().Get()
    ox = camera.GetHorizontalApertureOffsetAttr().Get()
    oy = camera.GetVerticalApertureOffsetAttr().Get()
    return np.array([[focal*width/horizontal, 0., width/2-ox*width/horizontal],
                     [0., focal*height/vertical, height/2+oy*height/vertical], [0., 0., 1.]])


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

