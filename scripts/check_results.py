#!/usr/bin/env python3
"""Check measured episode and every RGB-D stream required by its saved report."""
import argparse
import json
from pathlib import Path

from audit_episode import audit
from audit_global_camera import audit as audit_global_camera


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = json.loads((args.output / "report.json").read_text())
    requires_global = ("global_camera_eye" in report.get("config", {})
                       or "global_camera" in report
                       or "global_camera" in report.get("asset_sources", {}))
    result = audit(args.output)
    global_camera = audit_global_camera(args.output) if requires_global else None
    failures = list(result["failed_checks"]) if not result["success"] else []
    if global_camera is not None and not global_camera["success"]:
        failures.extend("global_camera." + name for name in global_camera["failed_checks"])
    if failures:
        print("FAIL: " + ", ".join(failures))
        if global_camera is not None:
            for error in global_camera.get("errors", []):
                print("  global camera: " + error)
        raise SystemExit(1)
    camera = result.get("wrist_camera")
    suffix = f"; {camera['frames']} validated wrist RGB-D frames" if camera else ""
    if global_camera is not None:
        suffix += f"; {global_camera['frames']} validated global RGB-D frames"
    print(f"PASS: {len(result['floor_measurements'])}/12 physically pressed in order; "
          f"{result['samples']} measured samples; {result['button_light_mode']} lights{suffix}.")


if __name__ == "__main__":
    main()
