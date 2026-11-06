"""Lossless 640x480 RGB PNG wire format for the legacy elevator engine."""
from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
from PIL import Image


def encode_image(rgb):
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.shape != (480, 640, 3):
        raise ValueError("RGB must be uint8 [480,640,3]")
    buffer = BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_image(value):
    if not isinstance(value, str) or len(value) > 12_000_000:
        raise ValueError("Expected a bounded base64 PNG image")
    data = base64.b64decode(value, validate=True)
    with Image.open(BytesIO(data)) as picture:
        if picture.format != "PNG" or picture.size != (640, 480) or picture.mode != "RGB":
            raise ValueError("Expected a lossless RGB PNG of size 640x480")
        return np.array(picture, dtype=np.uint8, copy=True)


