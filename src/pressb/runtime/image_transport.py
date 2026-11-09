"""Negotiated lossless image transport, separate from model input semantics.

Only the first, existing Wall-OSS PIL resize may move to the sender. The
subsequent smart resize and Qwen processor stay in the model implementation.
Raw render resolution, camera optics, batch membership and pixels seen by the
model are unchanged. Legacy clients continue sending full-size base64 PNGs.
"""
from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

from PIL import Image
import numpy as np

from .contract import decode_image


class ValidatedObservation(dict):
    """Request-local decoded images; never accepted from a wire dictionary."""

    def __init__(self, observation, decoded_images):
        super().__init__(observation)
        self.decoded_images = decoded_images
        self.prepared_walloss = None


def walloss_resize_spec(longest_edge):
    if longest_edge is None:
        return None
    if type(longest_edge) is not int or longest_edge < 1:
        raise ValueError("input_image_resolution must be a positive integer or null")
    return {"version": 1, "kind": "walloss_pil_longest_edge", "longest_edge": longest_edge,
            "mode": "RGB", "resample": "bicubic", "rounding": "truncate"}


def _check_spec(spec):
    if not isinstance(spec, dict) or spec != walloss_resize_spec(spec.get("longest_edge")):
        raise ValueError("Unsupported image_preprocessing specification")


def validate_image_preprocessing(spec):
    """Validate an optional static producer setting; return an owned copy."""
    if spec is None:
        return None
    _check_spec(spec)
    return dict(spec)


def resized_size(size, spec):
    _check_spec(spec)
    width, height = size
    if (type(width) is not int or type(height) is not int
            or width < 1 or height < 1 or width * height > 4096 * 4096):
        raise ValueError("Invalid original image size")
    edge = spec["longest_edge"]
    result = (edge, int(edge * height / width)) if width > height else (int(edge * width / height), edge)
    if min(result) < 1:
        raise ValueError("Image aspect ratio is too extreme for pre-resize")
    return result


def resize_for_transport(image, spec):
    """Exactly Wall process_images' first resize (PIL RGB defaults to bicubic)."""
    if image.mode != "RGB":
        raise ValueError("Pre-resize requires an RGB PNG")
    return image.resize(resized_size(image.size, spec), resample=Image.Resampling.BICUBIC)


def encode_preprocessed_image(rgb, spec, *, encoding="png_base64"):
    """Move the same model resize before the producer's first PNG encoding.

    The source remains the native rendered RGB tile. Level 6 matches the
    original client-side PNG encoding and avoids extra cross-node bandwidth.
    This function never modifies the caller's array or PIL image.
    """
    _check_spec(spec)
    if encoding not in ("png_base64", "png_bytes"):
        raise ValueError("encoding must be png_base64 or png_bytes")
    if isinstance(rgb, Image.Image):
        image = rgb
    else:
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError("Expected uint8 RGB source image")
        image = Image.fromarray(np.ascontiguousarray(rgb))
    buffer = BytesIO()
    resize_for_transport(image, spec).save(buffer, format="PNG", compress_level=6)
    raw = buffer.getvalue()
    return {"encoding": encoding, "data": raw if encoding == "png_bytes" else base64.b64encode(raw).decode("ascii"),
            "preprocessing": {"spec": dict(spec), "original_size": list(image.size)}}


def validate_preprocessing_marker(value, decoded_image, expected_spec):
    marker = value.get("preprocessing")
    if marker is None:
        return
    if (not isinstance(marker, dict) or set(marker) != {"spec", "original_size"}
            or expected_spec is None or marker["spec"] != expected_spec):
        raise ValueError("Image preprocessing does not match the model's advertised specification")
    original = marker["original_size"]
    if not isinstance(original, (list, tuple)) or len(original) != 2:
        raise ValueError("Image preprocessing requires original_size=[width,height]")
    if decoded_image.size != resized_size(original, expected_spec):
        raise ValueError("Preprocessed image dimensions do not match the advertised resize")


def prepare_wire_payload(path, payload, metadata, *, image_transport="legacy", preprocess_images=False,
                         image_workers=1):
    """Copy only changed dictionaries; do not mutate reusable observations.

    Call only for binary MessagePack transport. No negotiation means no change.
    No batching or retry behavior is altered by this image optimization.
    """
    if (path.removeprefix("/") not in ("encode", "infer", "sample_chain") or not isinstance(payload, dict)
            or not isinstance(payload.get("observations"), list) or not isinstance(metadata, dict)):
        return payload
    binary = image_transport == "auto" and "png_bytes" in metadata.get("image_encodings", [])
    spec = metadata.get("image_preprocessing") if preprocess_images else None
    if not binary and spec is None:
        return payload
    if spec is not None:
        _check_spec(spec)
    if type(image_workers) is not int or image_workers < 1:
        raise ValueError("image_workers must be a positive integer")
    if spec is not None and image_workers > 1 and len(payload["observations"]) > 1:
        # PIL PNG decoding, resizing and encoding release the GIL. Each worker
        # owns its PIL images; map preserves observation/camera order. Joining
        # before returning also ensures every image validates before RPC send.
        def prepare_row(observation):
            single = prepare_wire_payload(path, {"observations": [observation]}, metadata,
                                          image_transport=image_transport, preprocess_images=preprocess_images)
            return single["observations"][0]
        with ThreadPoolExecutor(max_workers=min(image_workers, len(payload["observations"])),
                                thread_name_prefix="pressb-image") as executor:
            return {**payload, "observations": list(executor.map(prepare_row, payload["observations"]))}
    result = dict(payload)
    result["observations"] = []
    for observation in payload["observations"]:
        if not isinstance(observation, dict) or not isinstance(observation.get("images"), dict):
            result["observations"].append(observation)
            continue
        row = dict(observation)
        row["images"] = dict(observation["images"])
        for camera in metadata.get("observation", {}).get("cameras", []):
            value = row["images"].get(camera)
            if not isinstance(value, dict):
                continue  # authoritative validation remains on the server
            updated = dict(value)
            if spec is not None:
                if value.get("preprocessing") is not None:
                    raw = base64.b64decode(value["data"], validate=True) if value.get("encoding") == "png_base64" else value["data"]
                    # A producer already applied the agreed transform. Inspect
                    # only the PNG header, avoiding a redundant full decode.
                    # The inference service still fully decodes EVERY image
                    # before any GPU batch/cache mutation, including this path.
                    if not isinstance(raw, bytes) or not raw or len(raw) > 16 * 1024 * 1024:
                        raise ValueError("Invalid preprocessed PNG payload")
                    with Image.open(BytesIO(raw)) as header:
                        if header.format != "PNG" or header.mode != "RGB":
                            raise ValueError("Preprocessed image must be an RGB PNG")
                        validate_preprocessing_marker(value, header, spec)
                else:
                    image = decode_image(value, require_rgb=True)
                    buffer = BytesIO()
                    resize_for_transport(image, spec).save(buffer, format="PNG")
                    raw = buffer.getvalue()
                    updated["preprocessing"] = {"spec": dict(spec), "original_size": list(image.size)}
            elif value.get("encoding") == "png_base64":
                raw = base64.b64decode(value["data"], validate=True)
            elif value.get("encoding") == "png_bytes":
                raw = value["data"]
            else:
                continue
            updated.update(encoding="png_bytes" if binary else "png_base64",
                           data=raw if binary else base64.b64encode(raw).decode("ascii"))
            row["images"][camera] = updated
        result["observations"].append(row)
    return result
