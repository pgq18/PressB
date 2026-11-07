"""RealSense source contracts, exercised without USB hardware or its SDK."""

from collections import deque
import importlib
from threading import Event
import time
from types import SimpleNamespace

import numpy as np
import pytest


class FakeFrame:
    def __init__(self, number=1, timestamp_ms=100.0, domain="hardware_clock", data=None):
        self.number = number
        self.timestamp_ms = timestamp_ms
        self.domain = domain
        self.data = np.full((2, 3, 3), number, dtype=np.uint8) if data is None else data

    def get_data(self):
        return self.data

    def get_frame_number(self):
        return self.number

    def get_timestamp(self):
        return self.timestamp_ms

    def get_frame_timestamp_domain(self):
        return self.domain


class FakeFrames:
    def __init__(self, frame):
        self.frame = frame

    def get_color_frame(self):
        return self.frame


class FakeConfig:
    def __init__(self):
        self.serial = None
        self.streams = []

    def enable_device(self, serial):
        self.serial = serial

    def enable_stream(self, *args):
        self.streams.append(args)


class FakePipeline:
    def __init__(self, sdk):
        self.sdk = sdk
        self.serial = None
        self.config = None
        self.waited = []
        self.poll_count = 0
        self.stop_count = 0

    def start(self, config):
        self.config = config
        self.serial = config.serial
        if self.serial in self.sdk.start_errors:
            raise RuntimeError("camera start failed")
        _, width, height, pixel_format, fps = config.streams[0]
        video = SimpleNamespace(
            width=lambda: width,
            height=lambda: height,
            fps=lambda: fps,
            format=lambda: pixel_format,
        )
        return SimpleNamespace(
            get_stream=lambda _stream: SimpleNamespace(as_video_stream_profile=lambda: video)
        )

    def poll_for_frames(self):
        self.poll_count += 1
        queue = self.sdk.queued.get(self.serial)
        return FakeFrames(queue.popleft()) if queue else None

    def wait_for_frames(self, timeout_ms):
        self.waited.append(timeout_ms)
        queue = self.sdk.frames.get(self.serial)
        if queue:
            return FakeFrames(queue.popleft())
        raise RuntimeError("Frame didn't arrive within timeout")

    def stop(self):
        self.stop_count += 1
        if self.serial in self.sdk.stop_events:
            self.sdk.stop_events[self.serial].wait(2)


class FakeSDK:
    stream = SimpleNamespace(color="color")
    format = SimpleNamespace(rgb8="rgb8")
    camera_info = SimpleNamespace(
        serial_number="serial_number", name="name", firmware_version="firmware_version",
        usb_type_descriptor="usb_type_descriptor", physical_port="physical_port",
    )
    timestamp_domain = SimpleNamespace(
        hardware_clock="hardware_clock", system_time="system_time", global_time="global_time"
    )

    def __init__(self, serials=("camera-a", "camera-b")):
        self.serials = serials
        self.pipelines = []
        self.frames = {}
        self.queued = {}
        self.start_errors = set()
        self.stop_events = {}

    def context(self):
        devices = [
            SimpleNamespace(
                supports=lambda key: key == "serial_number", get_info=lambda _key, s=s: s
            )
            for s in self.serials
        ]
        return SimpleNamespace(query_devices=lambda: devices)

    def config(self):
        return FakeConfig()

    def pipeline(self, _context):
        pipeline = FakePipeline(self)
        self.pipelines.append(pipeline)
        return pipeline


@pytest.fixture
def source():
    return importlib.import_module("pressb.sensors.realsense")


@pytest.fixture
def sdk(source, monkeypatch):
    fake = FakeSDK()
    monkeypatch.setattr(source, "_load_sdk", lambda: fake)
    return fake


def camera_config(serial="camera-a"):
    return {"serial": serial, "width": 3, "height": 2, "fps": 30}


@pytest.mark.parametrize("serial", ["", " "])
def test_serial_is_required(source, serial):
    with pytest.raises(ValueError):
        source.RealSenseCameras({"front": camera_config(serial)})


def test_duplicate_serials_are_rejected(source):
    with pytest.raises(ValueError):
        source.RealSenseCameras({"front": "camera-a", "wrist": "camera-a"})


def test_serial_cannot_be_omitted(source):
    with pytest.raises(ValueError):
        source.RealSenseCameras({"front": {"width": 3, "height": 2}})


def test_constructing_camera_config_does_not_load_sdk(source, monkeypatch):
    def fail_if_loaded():
        pytest.fail("Constructing the source should not load the optional SDK")

    monkeypatch.setattr(source, "_load_sdk", fail_if_loaded)
    cameras = source.RealSenseCameras({"front": "camera-a"})
    assert not cameras.is_connected
    cameras.close()


def test_missing_device_fails_without_starting_other_cameras(source, sdk):
    cameras = source.RealSenseCameras(
        {"front": camera_config(), "wrist": camera_config("missing")}
    )
    with pytest.raises(ConnectionError):
        cameras.connect()
    assert not cameras.is_connected
    assert not any(p.config is not None for p in sdk.pipelines)
    cameras.close()


def test_partial_connect_stops_every_started_pipeline(source, sdk):
    sdk.start_errors.add("camera-b")
    cameras = source.RealSenseCameras(
        {"front": camera_config(), "wrist": camera_config("camera-b")}
    )
    with pytest.raises(RuntimeError, match="start failed"):
        cameras.connect()
    assert not cameras.is_connected
    first = next(p for p in sdk.pipelines if p.serial == "camera-a")
    assert first.stop_count == 1
    cameras.close()
    assert first.stop_count == 1


def test_rgb_stream_configuration_frames_and_metadata(source, sdk):
    sdk.frames = {
        "camera-a": deque([FakeFrame(5, 155.0)]),
        "camera-b": deque([FakeFrame(8, 255.0)]),
    }
    cameras = source.RealSenseCameras({
        "front": source.RealSenseCameraConfig("camera-a", width=3, height=2, fps=30),
        "wrist": camera_config("camera-b"),
    })
    try:
        cameras.connect()
        assert cameras.is_connected
        received_before = time.monotonic()
        images = cameras.read()
        received_after = time.monotonic()
        assert set(images) == {"front", "wrist"}
        for name, serial, number, stamp in [
            ("front", "camera-a", 5, 155.0),
            ("wrist", "camera-b", 8, 255.0),
        ]:
            assert images[name].shape == (2, 3, 3)
            assert images[name].dtype == np.uint8
            np.testing.assert_array_equal(images[name], number)
            metadata = cameras.metadata[name]
            assert metadata["serial"] == serial
            assert metadata["frame_number"] == number
            assert metadata["device_timestamp_ms"] == stamp
            assert received_before <= metadata["host_received_monotonic_s"] <= received_after
            assert 0 <= metadata["frame_age_s"] <= 0.5
            assert metadata["capture_age_s"] is None
        assert all(p.config.streams == [("color", 3, 2, "rgb8", 30)] for p in sdk.pipelines)
    finally:
        cameras.close()


def test_read_returns_owned_rgb_data_and_drains_queued_frames(source, sdk):
    queued = FakeFrame(1, 100.0)
    fresh = FakeFrame(2, 200.0)
    sdk.queued["camera-a"] = deque([queued])
    sdk.frames["camera-a"] = deque([fresh])
    cameras = source.RealSenseCameras({"front": camera_config()})
    try:
        cameras.connect()
        image = cameras.read()["front"]
        assert sdk.pipelines[0].poll_count >= 2
        assert not sdk.queued["camera-a"]
        assert not np.shares_memory(image, fresh.data)
        fresh.data.fill(99)
        np.testing.assert_array_equal(image, 2)
        assert cameras.metadata["front"]["frame_number"] == 2
    finally:
        cameras.close()


@pytest.mark.parametrize("number,timestamp", [(1, 100.0), (1, 101.0), (2, 100.0)])
def test_both_frame_number_and_timestamp_must_advance(source, sdk, number, timestamp):
    sdk.frames["camera-a"] = deque([FakeFrame(1, 100.0), FakeFrame(number, timestamp)])
    cameras = source.RealSenseCameras({"front": camera_config()}, timeout_ms=20)
    try:
        cameras.connect()
        cameras.read()
        with pytest.raises(TimeoutError):
            cameras.read()
        assert cameras.metadata == {}
    finally:
        cameras.close()


@pytest.mark.parametrize("domain", ["system_time", "global_time"])
def test_wall_clock_timestamp_rejects_stale_frames(source, sdk, domain):
    sdk.frames["camera-a"] = deque([
        FakeFrame(1, (time.time() - 10) * 1000, domain=domain)
    ])
    cameras = source.RealSenseCameras(
        {"front": camera_config()}, timeout_ms=20, max_frame_age_s=0.5
    )
    try:
        cameras.connect()
        with pytest.raises(RuntimeError, match="stale"):
            cameras.read()
    finally:
        cameras.close()


def test_close_is_idempotent_and_disallows_further_reads(source, sdk):
    cameras = source.RealSenseCameras({"front": camera_config()})
    cameras.close()
    cameras.connect()
    cameras.close()
    cameras.close()
    assert not cameras.is_connected
    assert sdk.pipelines[0].stop_count == 1
    with pytest.raises(RuntimeError):
        cameras.read()


def test_batch_rejects_frame_that_ages_while_waiting_for_second_camera(source, sdk, monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(source, "time", SimpleNamespace(monotonic=lambda: clock[0], time=time.time))
    sdk.frames = {
        "camera-a": deque([FakeFrame(1, 100.0)]),
        "camera-b": deque([FakeFrame(1, 100.0)]),
    }
    cameras = source.RealSenseCameras(
        {"front": camera_config(), "wrist": camera_config("camera-b")},
        timeout_ms=2000, max_frame_age_s=0.5,
    )
    cameras.connect()
    second = sdk.pipelines[1]
    original_wait = second.wait_for_frames

    def wait_for_delayed_frame(timeout_ms):
        clock[0] += 0.6
        return original_wait(timeout_ms)

    monkeypatch.setattr(second, "wait_for_frames", wait_for_delayed_frame)
    try:
        with pytest.raises(RuntimeError, match="stale"):
            cameras.read()
        assert cameras.metadata == {}
    finally:
        cameras.close()


def test_close_timeout_does_not_block_on_sdk_stop(source, sdk):
    release = Event()
    sdk.stop_events["camera-a"] = release
    cameras = source.RealSenseCameras({"front": camera_config()}, close_timeout_s=0.02)
    cameras.connect()
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            cameras.close()
        assert time.monotonic() - started < 0.5
        assert not cameras.is_connected
        with pytest.raises(RuntimeError, match="stopping"):
            cameras.connect()
        release.set()
        cameras.close()
        cameras.close()
        assert sdk.pipelines[0].stop_count == 1
    finally:
        release.set()
