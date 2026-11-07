#!/usr/bin/env python3
"""Discover/capture real RGB cameras and inspect CAN availability without actuation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata as package_metadata
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pressb.sensors.realsense import RealSenseCameraConfig, RealSenseCameras, discover_realsense_devices


def can_interfaces(sysfs_root=Path("/sys/class/net")):
    """Read Linux ARPHRD_CAN=280; never bring interfaces up or open a CAN socket."""
    result = []
    for path in sorted(sysfs_root.iterdir()):
        if (path / "type").read_text().strip() == "280":
            result.append({"name": path.name, "state": (path / "operstate").read_text().strip()})
    return result


def package_versions():
    result = {}
    for name in ("numpy", "pyrealsense2", "Pillow", "piper_sdk", "python-can"):
        try:
            result[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            result[name] = None
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discover", action="store_true", help="Enumerate camera serials; no streams opened.")
    parser.add_argument("--probe-can", action="store_true", help="Report Linux CAN interfaces (read only).")
    parser.add_argument("--camera", action="append", default=[], metavar="NAME=SERIAL",
                        help="Explicit logical name/serial mapping; repeat for each camera.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=15)
    parser.add_argument("--timeout-ms", type=int, default=3000)
    parser.add_argument("--max-frame-age-s", type=float, default=.5)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if not args.discover and not args.camera and not args.probe_can:
        parser.error("Use --discover, --probe-can, or explicit --camera NAME=SERIAL mappings.")
    if args.camera and args.output_dir is None:
        parser.error("Capturing requires --output-dir.")
    if args.frames <= 0 or args.warmup_frames < 0:
        parser.error("--frames must be positive and --warmup-frames nonnegative.")
    configs = {}
    for value in args.camera:
        name, separator, serial = value.partition("=")
        if not separator or not name or not serial or Path(name).name != name or name in (".", ".."):
            parser.error("Each --camera must be NAME=SERIAL with a filename-safe name.")
        if name in configs:
            parser.error(f"Duplicate logical camera name: {name}")
        configs[name] = RealSenseCameraConfig(serial, args.width, args.height, args.fps)
    output = args.output_dir.resolve() if args.output_dir else None
    if output:
        output.mkdir(parents=True, exist_ok=True)
        if (output / "probe.json").exists():
            parser.error(f"Refusing to overwrite existing probe evidence: {output / 'probe.json'}")
    report = {"started_at": datetime.now(timezone.utc).isoformat(), "hostname": platform.node(),
              "python": sys.executable, "packages": package_versions(), "success": False,
              "can_interfaces": can_interfaces(), "robot_actuated": False,
              "camera_mapping": {name: spec.serial for name, spec in configs.items()},
              "physical_mount_mapping": "UNKNOWN; CLI names reflect the supplied mapping, not mount verification.",
              "hardware_synchronized": False, "captures": [], "close_succeeded": None}
    source = None
    try:
        if args.discover or configs:
            report["devices"] = discover_realsense_devices()
        if configs:
            from PIL import Image

            source = RealSenseCameras(configs, timeout_ms=args.timeout_ms, max_frame_age_s=args.max_frame_age_s)
            source.connect()
            for _ in range(args.warmup_frames):
                source.read()
            start = time.monotonic()
            for frame_index in range(args.frames):
                images = source.read()
                capture = {"index": frame_index,
                           "cameras": {name: dict(row) for name, row in source.metadata.items()}}
                report["captures"].append(capture)
                if frame_index in {0, args.frames - 1}:
                    for name, pixels in images.items():
                        filename = f"{name}_{frame_index:04d}.png"
                        path = output / filename
                        Image.fromarray(pixels).save(path)
                        capture["cameras"][name]["image_file"] = filename
                        capture["cameras"][name]["image_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            report["capture_duration_s"] = time.monotonic() - start
            report["batch_rate_hz"] = args.frames / report["capture_duration_s"]
            # Stop before opening a fresh source, proving device handles are released.
            source.close()
            report["close_succeeded"] = True
            source = RealSenseCameras(configs, timeout_ms=args.timeout_ms, max_frame_age_s=args.max_frame_age_s)
            source.connect()
            source.read()
            report["reopen_after_close_succeeded"] = True
        report["success"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if source is not None:
            try:
                source.close()
                report["close_succeeded"] = True
            except Exception as exc:
                report["success"] = False
                report["close_succeeded"] = False
                report["close_error"] = f"{type(exc).__name__}: {exc}"
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        if output:
            (output / "probe.json").write_text(json.dumps(report, indent=2) + "\n")
        summary = {key: value for key, value in report.items() if key != "captures"}
        summary["captured_batches"] = len(report["captures"])
        print(json.dumps(summary, indent=2))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
