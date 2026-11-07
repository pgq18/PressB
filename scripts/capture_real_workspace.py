#!/usr/bin/env python3
"""Capture a stationary RGB-D workspace and passive PiPER feedback.

Standalone on a camera host: requires numpy, Pillow, pyrealsense2, python-can.
This script never transmits CAN frames, enables motors, or changes CAN settings.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import threading
import time

import numpy as np


def intrinsics(profile):
    i = profile.as_video_stream_profile().get_intrinsics()
    return dict(width=i.width, height=i.height, fx=i.fx, fy=i.fy,
                cx=i.ppx, cy=i.ppy, model=str(i.model), coeffs=list(i.coeffs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--global-serial", default="261622078356")
    parser.add_argument("--wrist-serial", default="243322073170")
    parser.add_argument("--can", default="can0")
    args = parser.parse_args()
    import can
    import pyrealsense2 as rs
    from PIL import Image

    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(started_at_unix_s=time.time(), hardware_synchronized=False,
                  can_access="receive only; no send calls", success=False, cameras={})
    done = threading.Event()
    feedback = []
    receiver_errors = []
    bus = can.Bus(interface="socketcan", channel=args.can,
                  receive_own_messages=False, local_loopback=False)

    def receive():
        try:
            with (args.output / "can_frames.jsonl").open("x") as f:
                latest = {}
                while not done.is_set():
                    msg = bus.recv(timeout=.1)
                    if msg is None or msg.arbitration_id not in (0x2A1, 0x2A5, 0x2A6, 0x2A7, 0x2A8):
                        continue
                    row = dict(id=hex(msg.arbitration_id), data=bytes(msg.data).hex(),
                               kernel_time_s=msg.timestamp, receive_time_s=time.time())
                    f.write(json.dumps(row) + "\n")
                    if msg.arbitration_id in (0x2A5, 0x2A6, 0x2A7):
                        pair = [(int.from_bytes(msg.data[i:i+4], "big", signed=True) / 1000.)
                                for i in (0, 4)]
                        latest[msg.arbitration_id] = (msg.timestamp, pair)
                    if msg.arbitration_id == 0x2A8:
                        latest[0x2A8] = (msg.timestamp, int.from_bytes(msg.data[:4], "big", signed=True) / 1e6)
                    if all(k in latest for k in (0x2A5, 0x2A6, 0x2A7, 0x2A8)):
                        if max(v[0] for v in latest.values()) - min(v[0] for v in latest.values()) < .1:
                            q = sum((latest[k][1] for k in (0x2A5, 0x2A6, 0x2A7)), [])
                            feedback.append(dict(wall_time_s=time.time(), joints_deg=q,
                                                 gripper_width_m=latest[0x2A8][1]))
        except BaseException as exc:
            receiver_errors.append(repr(exc))

    thread = threading.Thread(target=receive, daemon=True)
    thread.start()
    pipes = []
    try:
        for name, serial in (("global", args.global_serial), ("wrist", args.wrist_serial)):
            pipe, cfg = rs.pipeline(), rs.config()
            cfg.enable_device(serial)
            cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
            cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            profile = pipe.start(cfg)
            pipes.append(pipe)
            dev = profile.get_device()
            color_profile = profile.get_stream(rs.stream.color)
            depth_profile = profile.get_stream(rs.stream.depth)
            extr = depth_profile.get_extrinsics_to(color_profile)
            info = dict(serial=serial, color=intrinsics(color_profile), depth=intrinsics(depth_profile),
                        depth_scale_m=dev.first_depth_sensor().get_depth_scale(),
                        depth_to_color=dict(rotation_column_major=list(extr.rotation), translation_m=list(extr.translation)),
                        sensor_settings={})
            for sensor in dev.query_sensors():
                settings = {}
                for opt in (rs.option.exposure, rs.option.gain, rs.option.white_balance,
                            rs.option.enable_auto_exposure, rs.option.enable_auto_white_balance):
                    if sensor.supports(opt):
                        settings[str(opt)] = sensor.get_option(opt)
                info["sensor_settings"][sensor.get_info(rs.camera_info.name)] = settings
            align = rs.align(rs.stream.color)
            for _ in range(90):
                pipe.wait_for_frames(timeout_ms=5000)
            depth_stack, times = [], []
            for i in range(30):
                frames = align.process(pipe.wait_for_frames(timeout_ms=5000))
                rgb, depth = frames.get_color_frame(), frames.get_depth_frame()
                rgb_array = np.asanyarray(rgb.get_data()).copy()
                depth_stack.append(np.asanyarray(depth.get_data()).copy())
                times.append(dict(wall_time_s=time.time(), color_timestamp_ms=rgb.get_timestamp(),
                                  depth_timestamp_ms=depth.get_timestamp(), frame_number=rgb.get_frame_number()))
                if i in (0, 14, 29):
                    Image.fromarray(rgb_array).save(args.output / f"{name}_{i:02d}.png")
            stack = np.stack(depth_stack)
            values = np.where(stack > 0, stack.astype(float), np.nan)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                depth_m = np.nanmedian(values, axis=0) * info["depth_scale_m"]
            np.savez_compressed(args.output / f"{name}_depth.npz", depth_m=depth_m,
                                depth_raw=stack[-1], valid_count=(stack > 0).sum(axis=0))
            Image.fromarray(rgb_array).save(args.output / f"{name}.png")
            info.update(frames=times, aligned_depth_intrinsics=intrinsics(depth.profile))
            report["cameras"][name] = info
            pipe.stop()
            pipes.remove(pipe)
        if receiver_errors:
            raise RuntimeError(receiver_errors)
        if not feedback:
            raise RuntimeError("No coherent joint/gripper feedback received")
        q = np.array([r["joints_deg"] for r in feedback])
        report["feedback"] = dict(samples=len(feedback), first=feedback[0], last=feedback[-1],
                                  median_joints_deg=np.median(q, axis=0).tolist(),
                                  joint_range_deg=np.ptp(q, axis=0).tolist(),
                                  median_gripper_width_m=float(np.median([r["gripper_width_m"] for r in feedback])))
        (args.output / "feedback.json").write_text(json.dumps(feedback) + "\n")
        report["success"] = True
    finally:
        for pipe in pipes:
            pipe.stop()
        done.set()
        thread.join(timeout=2)
        bus.shutdown()
        report["finished_at_unix_s"] = time.time()
        (args.output / "capture.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(output=str(args.output), success=report["success"], feedback=report.get("feedback"))))


if __name__ == "__main__":
    main()
