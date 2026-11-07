"""Standalone, explicitly mapped RealSense RGB cameras (no simulator/LeRobot imports)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import importlib
import math
import threading
import time
from typing import Any

import numpy as np


def _load_sdk():
    try:
        return importlib.import_module("pyrealsense2")
    except ImportError as exc:
        raise ImportError("RealSense capture requires pyrealsense2; install pressb[real-robot].") from exc


@dataclass(frozen=True)
class RealSenseCameraConfig:
    serial: str
    width: int = 640
    height: int = 480
    fps: int = 30

    def __post_init__(self):
        if not isinstance(self.serial, str) or not self.serial.strip() or self.serial != self.serial.strip():
            raise ValueError("Every camera needs an explicit nonempty serial string.")
        for field in ("width", "height", "fps"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Camera {field} must be a positive integer.")


def _device_info(sdk, device):
    result = {}
    for key in ("serial_number", "name", "firmware_version", "usb_type_descriptor", "physical_port"):
        token = getattr(sdk.camera_info, key)
        result[key] = device.get_info(token) if device.supports(token) else None
    return result


def discover_realsense_devices() -> list[dict[str, Any]]:
    """Enumerate devices without opening streams or assigning physical camera roles."""
    sdk = _load_sdk()
    context = sdk.context()
    return [_device_info(sdk, device) for device in context.query_devices()]


class RealSenseCameras:
    """Read RGB arrays from a logical-name -> physical-serial mapping.

    Each read discards queued frames and waits for new frames within one batch
    timeout. Frame numbers and device timestamps must advance. System/global
    timestamps additionally enforce capture age; hardware-clock timestamps cannot
    be compared across devices or to host time, so their capture age is unknown.
    ``metadata`` distinguishes capture age from age since host receipt. Cameras
    are acquired independently; this class does not claim hardware synchronization.

    The class is single-caller: do not call read/connect/close concurrently. The
    SDK is imported only by connect/discovery. No persistent sensor options,
    firmware settings, depth emitters, or hardware synchronization are changed.
    """

    def __init__(self, config: Mapping[str, Any], *, timeout_ms: int = 2000,
                 max_frame_age_s: float = .5, close_timeout_s: float = 5.):
        if not isinstance(config, Mapping):
            raise ValueError("Camera config must map logical names to explicit serials.")
        if "cameras" in config:
            extra = set(config) - {"cameras", "timeout_ms", "max_frame_age_s", "close_timeout_s"}
            if extra:
                raise ValueError(f"Unknown RealSense source settings: {sorted(extra)}")
            timeout_ms = config.get("timeout_ms", timeout_ms)
            max_frame_age_s = config.get("max_frame_age_s", max_frame_age_s)
            close_timeout_s = config.get("close_timeout_s", close_timeout_s)
            config = config["cameras"]
        if not isinstance(config, Mapping) or not config:
            raise ValueError("At least one explicit camera name/serial mapping is required.")
        self.config = {}
        for name, value in config.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Camera logical names must be nonempty strings.")
            if isinstance(value, str):
                value = RealSenseCameraConfig(serial=value)
            elif isinstance(value, Mapping):
                value = dict(value)
                # Read the existing LeRobot-style serial spelling without importing it.
                if "serial_number_or_name" in value and "serial" not in value:
                    value["serial"] = value.pop("serial_number_or_name")
                try:
                    value = RealSenseCameraConfig(**value)
                except TypeError as exc:
                    raise ValueError(f"Invalid camera configuration for {name!r}: {exc}") from exc
            if not isinstance(value, RealSenseCameraConfig):
                raise ValueError(f"Invalid camera configuration for {name!r}.")
            self.config[name] = value
        serials = [value.serial for value in self.config.values()]
        if len(set(serials)) != len(serials):
            raise ValueError("Duplicate camera serials: each logical camera must use a distinct device.")
        if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int) or timeout_ms <= 0:
            raise ValueError("timeout_ms must be a positive integer.")
        for name, value in (("max_frame_age_s", max_frame_age_s), ("close_timeout_s", close_timeout_s)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        self.timeout_ms = timeout_ms
        self.max_frame_age_s = float(max_frame_age_s)
        self.close_timeout_s = float(close_timeout_s)
        self.metadata: dict[str, dict[str, Any]] = {}
        self.devices: dict[str, dict[str, Any]] = {}
        self._pipelines: dict[str, Any] = {}
        self._previous: dict[str, tuple[int, float]] = {}
        self._stopping: dict[str, tuple[threading.Thread, list[Exception]]] = {}
        self._connected = False
        self._sdk = None
        self._context = None

    @property
    def is_connected(self):
        return self._connected

    def connect(self):
        if self._connected or self._pipelines or self._stopping:
            raise RuntimeError("Cameras are already connected or still stopping.")
        sdk = self._sdk = _load_sdk()
        self._context = sdk.context()
        available = {_device_info(sdk, device)["serial_number"]: _device_info(sdk, device)
                     for device in self._context.query_devices()}
        missing = [spec.serial for spec in self.config.values() if spec.serial not in available]
        if missing:
            raise ConnectionError(f"Missing RealSense serials {missing}; available: {sorted(available)}")
        try:
            for name, spec in self.config.items():
                pipeline = sdk.pipeline(self._context)
                stream_config = sdk.config()
                stream_config.enable_device(spec.serial)
                stream_config.enable_stream(sdk.stream.color, spec.width, spec.height, sdk.format.rgb8, spec.fps)
                profile = pipeline.start(stream_config)
                self._pipelines[name] = pipeline
                stream = profile.get_stream(sdk.stream.color).as_video_stream_profile()
                actual = (stream.width(), stream.height(), stream.fps(), stream.format())
                expected = (spec.width, spec.height, spec.fps, sdk.format.rgb8)
                if actual != expected:
                    raise RuntimeError(f"Camera {name!r} stream differs from requested RGB mode: {actual}")
                self.devices[name] = available[spec.serial]
            self._previous.clear()
            self.metadata.clear()
            self._connected = True
        except Exception as exc:
            try:
                self.close()
            except Exception as cleanup_exc:
                raise RuntimeError(f"Camera connect failed ({exc}); cleanup also failed: {cleanup_exc}") from exc
            raise
        return self

    def read(self) -> dict[str, np.ndarray]:
        if not self._connected:
            raise RuntimeError("Connect cameras before reading.")
        deadline = time.monotonic() + self.timeout_ms / 1000.
        # Clear stale output even on failure: consumers must never reuse old metadata.
        self.metadata = {}
        for name, pipeline in self._pipelines.items():
            while pipeline.poll_for_frames():
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out draining queued frames for camera {name!r}.")
        images, metadata = {}, {}
        for name, pipeline in self._pipelines.items():
            while True:
                remaining_ms = math.ceil((deadline - time.monotonic()) * 1000.)
                if remaining_ms <= 0:
                    raise TimeoutError(f"Timed out waiting for fresh RGB frame from camera {name!r}.")
                try:
                    frames = pipeline.wait_for_frames(remaining_ms)
                except RuntimeError as exc:
                    raise TimeoutError(f"Failed waiting for RGB frame from camera {name!r}: {exc}") from exc
                frame = frames.get_color_frame()
                if not frame:
                    continue
                received = time.monotonic()
                received_wall = time.time()
                number, timestamp = int(frame.get_frame_number()), float(frame.get_timestamp())
                if not math.isfinite(timestamp):
                    raise RuntimeError(f"Camera {name!r} returned a nonfinite timestamp.")
                previous = self._previous.get(name)
                if previous is not None and (number <= previous[0] or timestamp <= previous[1]):
                    continue
                domain = frame.get_frame_timestamp_domain()
                comparable = domain in (self._sdk.timestamp_domain.system_time,
                                         self._sdk.timestamp_domain.global_time)
                capture_age = received_wall - timestamp / 1000. if comparable else None
                if capture_age is not None and (capture_age > self.max_frame_age_s or capture_age < -.1):
                    raise RuntimeError(f"Camera {name!r} returned a stale or future frame (age={capture_age:.3f}s).")
                array = np.asanyarray(frame.get_data())
                spec = self.config[name]
                if array.shape != (spec.height, spec.width, 3) or array.dtype != np.uint8:
                    raise RuntimeError(f"Camera {name!r} returned invalid RGB shape/dtype: {array.shape}, {array.dtype}")
                images[name] = array.copy()  # SDK buffers may be reused after the frame is released.
                metadata[name] = {
                    "serial": spec.serial, "frame_number": number, "device_timestamp_ms": timestamp,
                    "timestamp_domain": str(domain), "host_received_monotonic_s": received,
                    "host_received_unix_s": received_wall, "capture_age_s": capture_age,
                    "color_order": "RGB", "shape": list(array.shape),
                }
                self._previous[name] = (number, timestamp)
                break
        now = time.monotonic()
        for name, row in metadata.items():
            row["frame_age_s"] = now - row["host_received_monotonic_s"]
            age = row["frame_age_s"] + (row["capture_age_s"] or 0.)
            if age > self.max_frame_age_s:
                raise RuntimeError(f"Camera {name!r} frame became stale while reading the camera batch.")
        self.metadata = metadata
        return images

    def close(self):
        """Stop every opened pipeline, with a bounded wait and retryable timeout."""
        self._connected = False
        self.metadata.clear()
        for name, pipeline in self._pipelines.items():
            if name in self._stopping:
                continue
            errors: list[Exception] = []

            def stop(target=pipeline, failures=errors):
                try:
                    target.stop()
                except Exception as exc:
                    failures.append(exc)

            thread = threading.Thread(target=stop, name=f"pressb-realsense-stop-{name}", daemon=True)
            self._stopping[name] = (thread, errors)
            thread.start()
        deadline = time.monotonic() + self.close_timeout_s
        failures, pending = [], []
        for name, (thread, errors) in list(self._stopping.items()):
            thread.join(max(0., deadline - time.monotonic()))
            if thread.is_alive():
                pending.append(name)
            else:
                self._stopping.pop(name)
                if errors:
                    failures.append(f"{name}: {errors[0]}")
                else:
                    self._pipelines.pop(name)
        if pending:
            raise TimeoutError(f"Camera stop timed out after {self.close_timeout_s}s: {pending}")
        if failures:
            raise RuntimeError("Camera stop failed: " + "; ".join(failures))
        self._context = None

    def __enter__(self):
        return self.connect()

    def __exit__(self, exc_type, exc, tb):
        self.close()
