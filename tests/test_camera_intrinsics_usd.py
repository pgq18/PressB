"""Measured focal lengths and principal points survive USD camera projection."""
import numpy as np
import pytest

pytest.importorskip("pxr")
from pxr import Usd, UsdGeom

from pressb.sensors.camera import camera_intrinsics_from_usd, set_intrinsics


@pytest.mark.parametrize("fx,fy,cx,cy", [
    (667.7304832524369, 667.7304832524369, 320., 240.),
    (606.1517333984375, 605.0548095703125, 327.25390625, 249.86172485351562),
    (605.6944580078125, 605.6384887695312, 320.6307678222656, 250.77304077148438),
])
def test_usd_projection_matches_calibrated_pixels(fx, fy, cx, cy):
    stage = Usd.Stage.CreateInMemory()
    camera = UsdGeom.Camera.Define(stage, "/Camera")
    set_intrinsics(camera.GetPrim(), dict(resolution=[640, 480], fx=fx, fy=fy, cx=cx, cy=cy))
    # USD cameras look along -Z, with +Y up. Use USD's own projection rather
    # than repeating the aperture-to-K equations as the only correctness test.
    xyz = np.array([[0., 0., -1.], [.1, .2, -1.], [-.3, .1, -2.]])
    clip = np.c_[xyz, np.ones(len(xyz))] @ np.asarray(camera.GetCamera().frustum.ComputeProjectionMatrix())
    ndc = clip[:, :3] / clip[:, 3:]
    pixels = np.c_[(ndc[:, 0]+1)*320, (1-ndc[:, 1])*240]
    expected = np.c_[fx*xyz[:, 0]/-xyz[:, 2]+cx, fy*-xyz[:, 1]/-xyz[:, 2]+cy]
    np.testing.assert_allclose(pixels, expected, atol=1e-4, rtol=0)
    np.testing.assert_allclose(camera_intrinsics_from_usd(camera.GetPrim(), [640, 480]),
                               [[fx, 0., cx], [0., fy, cy], [0., 0., 1.]], atol=1e-4, rtol=0)
