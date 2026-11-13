"""Optional deterministic color response for simulated RGB cameras.

This is a fixed, empirical match to recorded imagery, not a measurement of a
real camera's white-balance or exposure controls. It operates on the renderer's
display RGB after tone mapping, using the standard sRGB transfer functions.
It never changes optics, depth, geometry, or real-camera acquisition.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from functools import lru_cache

import numpy as np

USD_RESPONSE_KEY = "pressb:cameraColorResponse"


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def normalize_response(value=None):
    value = {} if value is None else value
    if not isinstance(value, Mapping) or set(value) - {"exposure_ev", "linear_rgb_gain", "saturation"}:
        raise ValueError("Color response supports exposure_ev, linear_rgb_gain and saturation")
    ev = _number(value.get("exposure_ev", 0.), "exposure_ev")
    if not -16 <= ev <= 16:
        raise ValueError("exposure_ev must be in [-16, 16]")
    gains = value.get("linear_rgb_gain", [1., 1., 1.])
    if not isinstance(gains, (list, tuple)) or len(gains) != 3:
        raise ValueError("linear_rgb_gain must contain three positive finite numbers")
    gains = [_number(v, "linear_rgb_gain") for v in gains]
    if any(v <= 0 or not math.isfinite(v * 2.**ev) for v in gains):
        raise ValueError("linear_rgb_gain must be positive with finite combined exposure gain")
    saturation = _number(value.get("saturation", 1.), "saturation")
    if not 0 <= saturation <= 2:
        raise ValueError("saturation must be in [0, 2]")
    return dict(exposure_ev=ev, linear_rgb_gain=gains, saturation=saturation)


def normalize_camera_responses(value, camera_names):
    value = {} if value is None else value
    if not isinstance(value, Mapping) or set(value) - set(camera_names):
        raise ValueError(f"camera_color_response supports only cameras {list(camera_names)}")
    return {name: normalize_response(response) for name, response in value.items()}


def _encode(linear):
    linear = np.clip(linear, 0., 1.)
    srgb = np.where(linear <= .0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - .055)
    return np.rint(srgb * 255.).astype(np.uint8)


_LINEAR_ENCODE_LUT = _encode(np.arange(65536, dtype=np.float64) / 65535.)


class ColorResponse:
    """Compiled response; gain/exposure alone use a 256-entry per-channel LUT."""

    def __init__(self, value=None):
        self.config = normalize_response(value)
        self.gain = np.asarray(self.config["linear_rgb_gain"], dtype=np.float64) * 2.**self.config["exposure_ev"]
        self.saturation = self.config["saturation"]
        self.identity = bool(self.saturation == 1. and np.array_equal(self.gain, np.ones(3)))
        srgb = np.arange(256, dtype=np.float64) / 255.
        self.linear = np.where(srgb <= .04045, srgb / 12.92, ((srgb + .055) / 1.055) ** 2.4)
        self.lut = _encode(self.linear[:, None] * self.gain)
        # The normal camera range uses a float32 color matrix and a 16-bit
        # linear-to-sRGB table. Its quantization is below one output LSB, even
        # in the steep near-black sRGB segment, and avoids per-pixel powers.
        self.fast_matrix = bool(np.all(self.gain >= 1e-4) and np.all(self.gain <= 1e4))
        self.linear32 = self.linear.astype(np.float32)
        self.matrix32 = ((np.eye(3) * self.saturation +
                          (1. - self.saturation) * np.array([.2126, .7152, .0722])[None, :]) *
                         self.gain[:, None]).astype(np.float32) if self.fast_matrix else None

    def apply(self, rgb):
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError("Color response requires HWC uint8 RGB")
        if self.identity:
            return rgb
        if self.saturation == 1.:
            result = np.empty_like(rgb)
            for channel in range(3):
                result[..., channel] = self.lut[:, channel][rgb[..., channel]]
            return result
        if self.fast_matrix:
            linear = np.einsum("...c,dc->...d", self.linear32[rgb], self.matrix32)
            indices = np.rint(np.clip(linear, 0., 1.) * 65535.).astype(np.uint16)
            return _LINEAR_ENCODE_LUT[indices]
        linear = self.linear[rgb]
        luminance = np.sum(linear * [0.2126, 0.7152, 0.0722], axis=-1, keepdims=True)
        return _encode((luminance + self.saturation * (linear - luminance)) * self.gain)

    def apply_rgba(self, rgba):
        rgba = np.asarray(rgba)
        if rgba.dtype != np.uint8 or rgba.ndim != 3 or rgba.shape[-1] != 4:
            raise ValueError("Color response requires HWC uint8 RGBA")
        if self.identity:
            return rgba
        result = rgba.copy()
        result[..., :3] = self.apply(rgba[..., :3])
        return result


@lru_cache(maxsize=64)
def _compiled_response(encoded):
    return ColorResponse(json.loads(encoded))


def apply_color_response(rgb, value=None):
    """Convenience API for saved/rendered RGB with cached validated responses."""
    encoded = json.dumps(normalize_response(value), sort_keys=True)
    return _compiled_response(encoded).apply(rgb)


def bind_sim_camera_response(camera, value=None):
    """Wrap one CPU simulation Camera's RGBA accessor without mutating its cache.

    Both get_rgb and get_rgba use the same wrapped raw accessor.
    No wrapper is installed for identity; repeat installation is idempotent.
    Explicit CUDA reads are rejected for an enabled CPU response rather than
    silently changing the returned tensor's device or type.
    """
    response = ColorResponse(value)
    existing = getattr(camera, "_pressb_color_response", None)
    if existing is not None:
        if existing.config != response.config:
            raise ValueError("Camera already has a different color response")
        return camera
    if response.identity:
        return camera
    raw_rgba = camera.get_rgba

    def get_rgba(device=None):
        if device not in (None, "cpu"):
            raise ValueError("Configured simulation camera color response requires CPU RGB readback")
        rgba = raw_rgba(device=device)
        # Preserve native "not captured yet" behavior during camera startup.
        if rgba is None or (isinstance(rgba, np.ndarray) and rgba.size == 0):
            return rgba
        return response.apply_rgba(rgba)

    def get_rgb(device=None):
        rgba = get_rgba(device=device)
        if rgba is None or rgba.size == 0:
            return None
        return rgba[..., :3]

    camera.get_rgba = get_rgba
    camera.get_rgb = get_rgb
    camera._pressb_color_response = response
    return camera


def author_camera_response(prim, value):
    """Serialize non-optical settings so native tiled and Camera paths agree."""
    prim.SetCustomDataByKey(USD_RESPONSE_KEY, json.dumps(normalize_response(value), sort_keys=True))


def read_camera_response(prim):
    value = prim.GetCustomDataByKey(USD_RESPONSE_KEY)
    return ColorResponse(None if value is None else json.loads(value))
