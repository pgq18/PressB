"""Save independent RGB-D streams and the calibration needed to use them."""
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image


class WristCameraRecorder:
    """The default preserves the wrist API; use output_subdir for other cameras."""

    def __init__(self, output, camera, metadata, capture_stride, expected_steps, output_subdir="wrist_camera"):
        subdir = Path(output_subdir)
        if subdir.is_absolute() or ".." in subdir.parts or not subdir.parts:
            raise ValueError("Camera output_subdir must be a nonempty relative directory")
        if capture_stride <= 0 or expected_steps <= 0:
            raise ValueError("Camera capture stride and expected steps must be positive")
        self.output = Path(output) / subdir
        self.camera_label = f"{metadata.get('name') or metadata.get('model') or 'RGB-D camera'} ({subdir})"
        for name in ("rgb", "depth"):
            (self.output / name).mkdir(parents=True, exist_ok=True)
            suffix = ".png" if name == "rgb" else ".npy"
            for previous in (self.output / name).glob("*" + suffix):
                if len(previous.stem) == 6 and previous.stem.isdecimal():
                    previous.unlink()
        self.camera = camera
        self.stride = capture_stride
        self.expected_frames = (expected_steps - 1) // capture_stride + 1
        self.rows = []
        self.fractions = []
        self.metadata = metadata
        self.preview_floors = set()
        self.preview_phases = set()
        width, height = camera.get_resolution()
        matrix = np.asarray(camera.get_intrinsics_matrix())
        calibration = {
            "width": int(width), "height": int(height), "fx": float(matrix[0, 0]),
            "fy": float(matrix[1, 1]), "cx": float(matrix[0, 2]), "cy": float(matrix[1, 2]),
            "K": matrix.tolist(), "depth_unit": "m", "depth_type": "distance_to_image_plane",
            "pose_axes": "USD camera: +X right, +Y up, -Z forward",
            "quaternion_order": "wxyz", "model": "pinhole",
            "sensor": metadata,
        }
        (self.output / "intrinsics.json").write_text(json.dumps(calibration, indent=2))
        # Truncate a previous run's manifest; only the current manifest names valid frames.
        (self.output / "timestamps.jsonl").write_text("")

    def capture(self, elapsed, floor, phase):
        rgb = np.asarray(self.camera.get_rgba())
        depth = self.camera.get_depth()
        if rgb.ndim != 3 or rgb.shape[-1] < 3 or depth is None:
            raise RuntimeError(f"{self.camera_label} did not produce RGB and depth")
        rgb = rgb[..., :3].astype(np.uint8)
        depth = np.asarray(depth, dtype=np.float32).squeeze()
        if depth.shape != rgb.shape[:2]:
            raise RuntimeError(f"{self.camera_label} RGB/depth dimensions disagree: {rgb.shape}, {depth.shape}")
        valid = np.isfinite(depth) & (depth > 0.)
        fraction = float(np.mean(valid))
        if fraction < .05 or float(np.std(rgb)) < 1.:
            raise RuntimeError(f"{self.camera_label} frame is blank or has insufficient valid depth")
        frame = len(self.rows)
        rgb_path = f"rgb/{frame:06d}.png"
        depth_path = f"depth/{frame:06d}.npy"
        Image.fromarray(rgb).save(self.output / rgb_path, compress_level=1)
        np.save(self.output / depth_path, depth, allow_pickle=False)
        position, orientation = self.camera.get_world_pose(camera_axes="usd")
        raw = self.camera.get_current_frame()
        rendering_frame = raw.get("rendering_frame", 0)
        if isinstance(rendering_frame, dict):
            # Isaac Sim 5.0 returns a rational Fabric reference time here,
            # not a frame index. Preserve its numerator and denominator.
            rendering_frame = {
                key: int(rendering_frame[key])
                for key in ("referenceTimeNumerator", "referenceTimeDenominator")
            }
            if rendering_frame["referenceTimeDenominator"] <= 0:
                raise ValueError("Camera reference time denominator must be positive")
            rendering_frame_kind = "reference_time_rational"
        else:
            rendering_frame = int(rendering_frame)
            rendering_frame_kind = "frame_index"
        row = {"frame": frame, "time": float(elapsed), "floor": int(floor), "phase": str(phase),
               "rgb": rgb_path, "depth": depth_path,
               "position_world": np.asarray(position).tolist(),
               "orientation_world": np.asarray(orientation).tolist(),
               "rendering_time": float(raw.get("rendering_time", 0.)),
               "rendering_frame": rendering_frame, "rendering_frame_kind": rendering_frame_kind}
        self.rows.append(row)
        self.fractions.append(fraction)
        with (self.output / "timestamps.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        preview_stems = ["initial"] if frame == 0 else []
        if phase == "hold" and floor not in self.preview_floors:
            preview_stems.append(f"floor_{floor}")
            self.preview_floors.add(floor)
        if phase not in self.preview_phases:
            preview_stems.append("phase_" + re.sub(r"[^A-Za-z0-9_-]", "_", str(phase)))
            self.preview_phases.add(phase)
        if preview_stems:
            # Display only; raw metric depth remains in depth/*.npy.
            from matplotlib import colormaps
            scaled = np.clip(np.where(valid, depth, 2.) / 2., 0., 1.)
            colors = (colormaps["turbo_r"](scaled)[..., :3] * 255).astype(np.uint8)
            colors[~valid] = 0
            for stem in preview_stems:
                Image.fromarray(rgb).save(self.output / f"{stem}.png")
                Image.fromarray(colors).save(self.output / f"{stem}_depth.png")

    def finish(self):
        report = {"valid_frames": len(self.rows), "expected_frames": self.expected_frames,
                  "capture_stride": self.stride,
                  "finite_depth_fraction": float(np.mean(self.fractions)) if self.fractions else 0.,
                  "minimum_finite_depth_fraction": min(self.fractions, default=0.),
                  "finite_depth_fraction_aggregation": "mean", "sensor": self.metadata,
                  "success": len(self.rows) == self.expected_frames and bool(self.rows)}
        (self.output / "camera_report.json").write_text(json.dumps(report, indent=2))
        return report
