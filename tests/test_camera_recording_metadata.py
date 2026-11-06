"""Exercise camera timestamp serialization without starting Isaac Sim."""
import json

import numpy as np
import pytest

from pressb.tasks.elevator.camera_recording import WristCameraRecorder
from pressb.sensors.recording import RGBDCameraRecorder


class FakeCamera:
    def __init__(self, rendering_frame):
        self.raw_frame = {
            "rendering_frame": rendering_frame,
            "rendering_time": np.float64(3.125),
        }

    def get_resolution(self):
        return 4, 3

    def get_intrinsics_matrix(self):
        return np.array([[100., 0., 2.], [0., 100., 1.5], [0., 0., 1.]])

    def get_rgba(self):
        return np.arange(48, dtype=np.uint8).reshape(3, 4, 4)

    def get_depth(self):
        return np.ones((3, 4), dtype=np.float32)

    def get_world_pose(self, camera_axes):
        assert camera_axes == "usd"
        return np.array([1., 2., 3.]), np.array([1., 0., 0., 0.])

    def get_current_frame(self):
        return self.raw_frame


def recorder(tmp_path, rendering_frame):
    return WristCameraRecorder(
        tmp_path, FakeCamera(rendering_frame), {"model": "D435"},
        capture_stride=1, expected_steps=1,
    )


@pytest.mark.parametrize(
    "rendering_frame, expected, kind",
    [
        (19, 19, "frame_index"),
        (np.int64(19), 19, "frame_index"),
        (
            {"referenceTimeNumerator": 9007199254740993, "referenceTimeDenominator": 120},
            {"referenceTimeNumerator": 9007199254740993, "referenceTimeDenominator": 120},
            "reference_time_rational",
        ),
        (
            {"referenceTimeNumerator": np.int64(9007199254740993),
             "referenceTimeDenominator": np.int64(120)},
            {"referenceTimeNumerator": 9007199254740993, "referenceTimeDenominator": 120},
            "reference_time_rational",
        ),
    ],
    ids=["legacy-python-int", "legacy-numpy-int", "rational-python-int", "rational-numpy-int"],
)
def test_capture_preserves_frame_identity_and_independent_times(tmp_path, rendering_frame, expected, kind):
    recording = recorder(tmp_path, rendering_frame)
    recording.capture(np.float64(1.625), np.int64(24), "approach")

    lines = (recording.output / "timestamps.jsonl").read_text().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    # Values above 2**53 catch an accidental float conversion of the Fabric numerator.
    assert row["rendering_frame"] == expected
    assert row["rendering_frame_kind"] == kind
    values = row["rendering_frame"].values() if isinstance(expected, dict) else [row["rendering_frame"]]
    assert all(type(value) is int for value in values)
    assert row["rendering_time"] == 3.125
    assert row["time"] == 1.625
    assert row["floor"] == 24
    assert row["phase"] == "approach"
    assert row["position_world"] == [1., 2., 3.]
    assert row["orientation_world"] == [1., 0., 0., 0.]
    assert row == recording.rows[0]
    assert (recording.output / row["rgb"]).is_file()
    np.testing.assert_array_equal(np.load(recording.output / row["depth"]), np.ones((3, 4)))
    assert recording.finish()["success"] is True


@pytest.mark.parametrize("denominator", [0, -1])
def test_capture_rejects_nonpositive_reference_denominator(tmp_path, denominator):
    recording = recorder(tmp_path, {
        "referenceTimeNumerator": np.int64(37),
        "referenceTimeDenominator": np.int64(denominator),
    })
    with pytest.raises(ValueError, match="denominator must be positive"):
        recording.capture(1., 24, "approach")
    assert (recording.output / "timestamps.jsonl").read_text() == ""
    assert recording.rows == []


def test_shared_recorder_accepts_task_metadata_without_inventing_elevator_fields(tmp_path):
    recording = RGBDCameraRecorder(
        tmp_path, FakeCamera(np.int64(19)), {"name": "inspection"},
        capture_stride=1, expected_steps=2, output_subdir="inspection_camera",
    )
    recording.capture(0.)
    recording.capture(.1, metadata={"object_id": "block-a", "phase": "grasp"},
                      preview_stems=("grasp",))
    first, second = [json.loads(row) for row in (recording.output / "timestamps.jsonl").read_text().splitlines()]
    assert "floor" not in first and "phase" not in first
    assert "floor" not in second
    assert second["object_id"] == "block-a" and second["phase"] == "grasp"
    assert first["rendering_frame"] == second["rendering_frame"] == 19
    assert [first["frame"], second["frame"]] == [0, 1]
    assert (recording.output / "grasp.png").is_file()
    assert (recording.output / "initial_depth.png").is_file()
    assert recording.finish()["success"] is True


def test_elevator_recorder_preserves_floor_and_phase_preview_names(tmp_path):
    recording = WristCameraRecorder(
        tmp_path, FakeCamera(19), {"model": "D435"}, capture_stride=1, expected_steps=2,
    )
    assert isinstance(recording, RGBDCameraRecorder)
    recording.capture(0., 24, "approach panel")
    recording.capture(.1, 24, "hold")
    for name in ("initial", "phase_approach_panel", "floor_24", "phase_hold"):
        assert (recording.output / (name + ".png")).is_file()
        assert (recording.output / (name + "_depth.png")).is_file()
    assert recording.finish()["success"] is True
