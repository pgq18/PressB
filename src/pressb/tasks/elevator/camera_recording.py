"""Elevator RGB-D recording with floor and phase preview labels."""
import re

from pressb.sensors.recording import RGBDCameraRecorder


class WristCameraRecorder(RGBDCameraRecorder):
    """Preserve the elevator capture manifest and its historical preview names."""

    def __init__(self, output, camera, metadata, capture_stride, expected_steps, output_subdir="wrist_camera"):
        super().__init__(output, camera, metadata, capture_stride, expected_steps,
                         output_subdir=output_subdir)
        self.preview_floors = set()
        self.preview_phases = set()

    def capture(self, elapsed, floor, phase):
        super().capture(elapsed, metadata={"floor": int(floor), "phase": str(phase)})

    def _preview_stems(self, row, preview_stems):
        stems = super()._preview_stems(row, preview_stems)
        floor, phase = row["floor"], row["phase"]
        if phase == "hold" and floor not in self.preview_floors:
            stems.append(f"floor_{floor}")
            self.preview_floors.add(floor)
        if phase not in self.preview_phases:
            stems.append("phase_" + re.sub(r"[^A-Za-z0-9_-]", "_", str(phase)))
            self.preview_phases.add(phase)
        return stems
