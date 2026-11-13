"""Photometric transfer, native camera parity and unmodified sensor storage."""
import json

import numpy as np
import pytest

from pressb.sensors.color_response import (
    ColorResponse, USD_RESPONSE_KEY, author_camera_response, bind_sim_camera_response,
    normalize_camera_responses, normalize_response, read_camera_response,
)


def test_identity_retains_every_byte_and_does_not_wrap_camera():
    rgba = np.arange(256, dtype=np.uint8).reshape(8, 8, 4)
    rgb = rgba[..., :3]
    assert ColorResponse().apply(rgb) is rgb
    assert ColorResponse({"exposure_ev": -1, "linear_rgb_gain": [2, 2, 2]}).apply_rgba(rgba) is rgba
    camera = FakeCamera(rgba)
    original = camera.get_rgba
    bind_sim_camera_response(camera)
    assert camera.get_rgba == original
    assert camera.get_rgba() is rgba


def test_exposure_applies_in_linear_light_and_clips_white():
    response = ColorResponse({"exposure_ev": 1})
    rgb = np.array([[[0, 64, 128], [180, 200, 255]]], dtype=np.uint8)
    result = response.apply(rgb)
    # Standard sRGB: doubling linear 128/255 encodes to 176, not 255.
    assert result.tolist() == [[[0, 90, 176], [245, 255, 255]]]
    assert rgb.tolist() == [[[0, 64, 128], [180, 200, 255]]]


def test_zero_saturation_uses_linear_rec709_luminance():
    rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [128, 128, 128]]], dtype=np.uint8)
    result = ColorResponse({"saturation": 0}).apply(rgb)
    assert result.tolist() == [[[127, 127, 127], [220, 220, 220], [76, 76, 76], [128, 128, 128]]]


def test_gain_is_after_saturation_and_preserves_source():
    rgb = np.array([[[0, 160, 220], [40, 40, 40]]], dtype=np.uint8)
    original = rgb.copy()
    value = {"saturation": .55, "linear_rgb_gain": [1.1, 1., .65], "exposure_ev": -.2}
    result = ColorResponse(value).apply(rgb)
    assert result[0, 0, 0] > 0  # desaturating cyan adds red without offsetting black
    assert np.max(result[0, 1]) < 45
    assert np.array_equal(rgb, original)
    assert np.array_equal(ColorResponse(value).apply(np.zeros_like(rgb)), np.zeros_like(rgb))


@pytest.mark.parametrize("saturation,gain,ev", [(.55, [1.1, 1., .65], -.2),
    (0., [.8, 1.3, 1.6], 2.), (2., [2., .7, 1.4], -2.), (.85, [1., 1., 1.], 0.)])
def test_fast_matrix_matches_float64_srgb_reference_within_one_lsb(saturation, gain, ev):
    rgb = np.random.default_rng(214).integers(0, 256, (240, 320, 3), dtype=np.uint8)
    srgb = rgb.astype(np.float64) / 255.
    linear = np.where(srgb <= .04045, srgb / 12.92, ((srgb + .055) / 1.055) ** 2.4)
    luminance = np.sum(linear * [.2126, .7152, .0722], axis=-1, keepdims=True)
    adjusted = np.clip((luminance + saturation * (linear - luminance)) * np.array(gain) * 2.**ev, 0., 1.)
    expected = np.rint(np.where(adjusted <= .0031308, adjusted * 12.92,
                                1.055 * adjusted ** (1 / 2.4) - .055) * 255.).astype(np.uint8)
    actual = ColorResponse(dict(saturation=saturation, linear_rgb_gain=gain, exposure_ev=ev)).apply(rgb)
    assert np.max(np.abs(actual.astype(np.int16) - expected.astype(np.int16))) <= 1


class FakeCamera:
    def __init__(self, rgba):
        self.rgba = rgba

    def get_rgba(self, device=None):
        return self.rgba

    def get_rgb(self, device=None):
        return self.rgba[..., :3]


def test_camera_wrapper_has_same_response_as_tiles_without_double_application():
    rgba = np.random.default_rng(3).integers(0, 256, (13, 17, 4), dtype=np.uint8)
    original = rgba.copy()
    value = {"saturation": .65, "linear_rgb_gain": [1.1, 1.0, .7], "exposure_ev": -.1}
    camera = FakeCamera(rgba)
    bind_sim_camera_response(camera, value)
    bound = camera.get_rgba
    bind_sim_camera_response(camera, value)
    assert camera.get_rgba is bound
    expected = ColorResponse(value).apply(rgba[..., :3])
    assert np.array_equal(camera.get_rgba()[..., :3], expected)
    assert np.array_equal(camera.get_rgb(), expected)
    assert np.array_equal(camera.get_rgba()[..., 3], rgba[..., 3])
    assert np.array_equal(rgba, original)
    with pytest.raises(ValueError, match="different"):
        bind_sim_camera_response(camera, {"exposure_ev": 1})
    with pytest.raises(ValueError, match="CPU"):
        camera.get_rgba(device="cuda:0")


@pytest.mark.parametrize("unready", [None, np.array([], dtype=np.uint8)])
def test_camera_wrapper_preserves_unready_capture(unready):
    camera = FakeCamera(unready)
    bind_sim_camera_response(camera, {"exposure_ev": 1})
    assert camera.get_rgba() is unready
    assert camera.get_rgb() is None


def test_workspace_config_validates_response_before_loading_simulator():
    from pressb.tasks.playground.backend import load_config
    with pytest.raises(ValueError, match="camera_color_response"):
        load_config({"scene_overrides": {"camera_color_response": {"unconfigured_camera": {}}}})
    with pytest.raises(ValueError, match="saturation"):
        load_config({"scene_overrides": {"camera_color_response": {"wrist": {"saturation": -1}}}})
    default = load_config({})["scene"]["camera_color_response"]["wrist"]
    configured = load_config({"scene_overrides": {"camera_color_response": {"wrist": {"saturation": .8}}}})
    assert configured["scene"]["camera_color_response"]["wrist"] == {**default, "saturation": .8}


@pytest.mark.parametrize("bad", [
    {"exposure_ev": True}, {"exposure_ev": float("nan")}, {"exposure_ev": 17},
    {"linear_rgb_gain": [1, -1, 1]}, {"linear_rgb_gain": [1, 2]},
    {"linear_rgb_gain": [1, float("inf"), 1]}, {"linear_rgb_gain": [True, 1, 1]},
    {"saturation": -1}, {"saturation": 2.1}, {"saturation": False}, {"offset": .1},
])
def test_invalid_config_rejected(bad):
    with pytest.raises(ValueError):
        normalize_response(bad)


def test_camera_names_checked_and_omitted_cameras_remain_unconfigured():
    assert normalize_camera_responses(None, ["up", "wrist"]) == {}
    assert list(normalize_camera_responses({"wrist": {}}, ["up", "wrist"])) == ["wrist"]
    with pytest.raises(ValueError):
        normalize_camera_responses({"wrong": {}}, ["up", "wrist"])


def test_usd_metadata_roundtrip_preserves_calibration():
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    stage = Usd.Stage.CreateInMemory()
    camera = UsdGeom.Camera.Define(stage, "/Camera")
    camera.CreateFocalLengthAttr(2.)
    camera.CreateHorizontalApertureAttr(2.1)
    optical = {str(attr.GetName()): attr.Get() for attr in camera.GetPrim().GetAttributes()}
    assert read_camera_response(camera.GetPrim()).identity
    value = {"exposure_ev": -.25, "linear_rgb_gain": [1.2, 1, .8], "saturation": .7}
    author_camera_response(camera.GetPrim(), value)
    assert read_camera_response(camera.GetPrim()).config == normalize_response(value)
    assert json.loads(camera.GetPrim().GetCustomDataByKey(USD_RESPONSE_KEY)) == normalize_response(value)
    assert optical == {str(attr.GetName()): attr.Get() for attr in camera.GetPrim().GetAttributes()}
    reloaded = Usd.Stage.CreateInMemory()
    reloaded.GetRootLayer().ImportFromString(stage.GetRootLayer().ExportToString())
    assert read_camera_response(reloaded.GetPrimAtPath("/Camera")).config == normalize_response(value)
