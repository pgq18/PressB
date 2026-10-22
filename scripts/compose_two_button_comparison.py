#!/usr/bin/env python3
"""Compose two button tasks into one simultaneous 1080p comparison, on CPU.

Example:
  env -u PYTHONPATH .conda/envs/pressb/bin/python \
    scripts/compose_two_button_comparison.py \
    --manifest /absolute/run/renders/render_manifest.json \
    --output /absolute/run/videos --floors 25 31 \
    --methods base residual_xyz_smoothed

Rows are independent target-floor tasks, synchronized from their initial states.
Each method cell contains uncropped global and wrist views, side by side. Exact
measured timestamps determine the displayed frame. Terminal frames are held and
marked; failures remain visible. The 30 Hz output ends one second after the last
termination, rounded up only to the next output-frame boundary.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont

import compose_rl_comparison_videos as common


WIDTH, HEIGHT, FPS = 1920, 1080, 30
VIEW_WIDTH, VIEW_HEIGHT = 480, 360
CELL_WIDTH = 2 * VIEW_WIDTH
TOP_HEIGHT, METHOD_HEIGHT = 84, 60
ROW_HEIGHT, ROW_BAND, STATUS_HEIGHT = 468, 44, 64
END_HOLD_SECONDS = 1.0
METHODS = ("base", "residual_xyz_smoothed")
METHOD_LABELS = {
    "base": "原始 VLA",
    "residual_xyz_smoothed": "冻结噪声 + XYZ 残差 + XYZ 平滑  α = 0.8",
}
BG, FG, MUTED = "#101922", "#f3f6fa", "#b4c1cf"
ROW_BG = "#1b2937"
COLORS = {"base": "#536b84", "residual_xyz_smoothed": "#326f63"}


def identity(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": common.sha256(path)}


class CameraReader(common.FrameReader):
    """Reuse the sequential timestamp reader, caching spatially resized frames."""

    def __init__(self, path, metadata, times):
        super().__init__(path, metadata, times)
        self.display_index, self.display_image = -1, None

    def at(self, seconds):
        original = super().at(seconds)
        if self.display_index != self.index:
            self.display_image = original.resize((VIEW_WIDTH, VIEW_HEIGHT), Image.Resampling.LANCZOS)
            self.display_index = self.index
        return self.display_image


class VideoWriter:
    def __init__(self, path, threads, crf):
        self.path = Path(path)
        self.temporary = self.path.with_name(self.path.stem + ".partial.mp4")
        self.count = 0
        self.process = subprocess.Popen([
            "ffmpeg", "-v", "error", "-nostdin", "-n", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-s:v", f"{WIDTH}x{HEIGHT}", "-r", str(FPS),
            "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "fast",
            "-crf", str(crf), "-pix_fmt", "yuv420p", "-threads", str(threads),
            "-movflags", "+faststart", str(self.temporary),
        ], stdin=subprocess.PIPE)

    def write(self, image):
        try:
            self.process.stdin.write(image.tobytes())
        except BrokenPipeError as error:
            raise RuntimeError(f"Video encoder failed: {self.path}") from error
        self.count += 1

    def finish(self):
        self.process.stdin.close()
        code = self.process.wait(timeout=120)
        if code:
            raise RuntimeError(f"Video encoder failed ({code}): {self.path}")
        metadata = common.probe(self.temporary)
        if (metadata["frame_count"] != self.count
                or (metadata["width"], metadata["height"]) != (WIDTH, HEIGHT)
                or abs(metadata["fps"] - FPS) > 1e-9):
            raise RuntimeError(f"Encoded video geometry or timing differs: {metadata}")
        self.temporary.replace(self.path)
        return {**metadata, **identity(self.path)}

    def abort(self):
        if self.process.stdin and not self.process.stdin.closed:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
        if self.process.poll() is None:
            self.process.terminate()
        self.process.wait(timeout=10)


def load_fonts(custom):
    _, path = common.load_fonts("zh", custom)
    return {key: ImageFont.truetype(path, size) for key, size in {
        "title": 32, "method": 30, "normal": 25, "small": 20, "clock": 34,
    }.items()}, path


def put(draw, xy, text, font, fill=FG):
    draw.text(xy, text, font=font, fill=fill, anchor="lt")


def put_right(draw, right, y, text, font, fill=FG):
    put(draw, (right - draw.textlength(text, font=font), y), text, font, fill)


def layout_for(cases, methods):
    rows = []
    for row_index, case in enumerate(cases):
        row_top = TOP_HEIGHT + METHOD_HEIGHT + row_index * ROW_HEIGHT
        camera_top = row_top + ROW_BAND
        rows.append({
            "row": row_index, "floor": case["floor"], "case_id": case["case_id"],
            "row_band_xywh": [0, row_top, WIDTH, ROW_BAND],
            "cells": [{
                "column": col, "method": method, "label": METHOD_LABELS[method],
                "global_xywh": [col * CELL_WIDTH, camera_top, VIEW_WIDTH, VIEW_HEIGHT],
                "wrist_xywh": [col * CELL_WIDTH + VIEW_WIDTH, camera_top, VIEW_WIDTH, VIEW_HEIGHT],
                "status_xywh": [col * CELL_WIDTH, camera_top + VIEW_HEIGHT, CELL_WIDTH, STATUS_HEIGHT],
            } for col, method in enumerate(methods)],
        })
    return rows


def make_background(cases, methods, fonts, geometry):
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    floor_text = " / ".join(str(case["floor"]) for case in cases)
    put(draw, (20, 12), f"{floor_text} 楼按钮任务 · 同步对比", fonts["title"])
    put(draw, (22, 53), "实测轨迹回放 · 原速 1× · 统一仿真时间轴", fonts["small"], MUTED)
    for col, method in enumerate(methods):
        left = col * CELL_WIDTH
        draw.rectangle((left, TOP_HEIGHT, left + CELL_WIDTH - 1, TOP_HEIGHT + METHOD_HEIGHT - 1),
                       fill=COLORS[method])
        label = METHOD_LABELS[method]
        if draw.textlength(label, font=fonts["method"]) > CELL_WIDTH - 40:
            raise ValueError(f"Method label does not fit: {label}")
        put(draw, (left + 20, TOP_HEIGHT + 15), label, fonts["method"])
    for case, geometry_row in zip(cases, geometry):
        _, top, _, band_height = geometry_row["row_band_xywh"]
        draw.rectangle((0, top, WIDTH - 1, top + band_height - 1), fill=ROW_BG)
        panel = case.get("layout", {})
        x, y = float(panel.get("offset_x_m", 0)), float(panel.get("offset_y_m", 0))
        location = "面板居中" if abs(x) < 1e-8 and abs(y) < 1e-8 else f"面板偏移 ({x:g}, {y:g}) m"
        put(draw, (20, top + 10), f"目标按钮：{case['floor']} 楼", fonts["normal"])
        put_right(draw, WIDTH - 20, top + 12, location, fonts["small"], MUTED)
    return image


def compose_frame(background, cases, methods, readers, geometry, fonts, seconds):
    image = background.copy()
    draw = ImageDraw.Draw(image)
    put_right(draw, WIDTH - 22, 15, f"统一时间  t = {seconds:06.2f} s", fonts["clock"])
    for case, geometry_row in zip(cases, geometry):
        for method, cell in zip(methods, geometry_row["cells"]):
            row = case["methods"][method]
            for view, name in (("global", "全局视角"), ("wrist", "腕部视角")):
                left, top, _, _ = cell[f"{view}_xywh"]
                image.paste(readers[(case["case_id"], method, view)].at(seconds), (left, top))
                label_width = math.ceil(draw.textlength(name, font=fonts["small"])) + 20
                draw.rectangle((left + 9, top + 9, left + 9 + label_width, top + 40), fill=BG)
                put(draw, (left + 19, top + 14), name, fonts["small"])
            left, top, width, _ = cell["status_xywh"]
            end = row["sim_seconds"]
            ended = seconds + 1e-9 >= end
            label, color = common.status_label(row, ended, "zh")
            put(draw, (left + 18, top + 8), label, fonts["normal"], color)
            put_right(draw, left + width - 18, top + 10, f"实测终止：{end:.3f} s", fonts["small"], MUTED)
            note = "已终止 · 保持末帧" if ended else "任务执行中"
            put(draw, (left + 18, top + 40), note, fonts["small"], MUTED)
            put_right(draw, left + width - 18, top + 40,
                      f"轨迹时间：{min(seconds, end):.3f} s", fonts["small"], MUTED)
    return image


def select_cases(manifest, root, floors, methods):
    cases = common.normalize_manifest(manifest, root, methods)
    selected = []
    for floor in floors:
        matches = [case for case in cases if case["floor"] == floor]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one matched task for floor {floor}, found {len(matches)}")
        selected.append(matches[0])
    return selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--floors", nargs=2, type=int, default=[25, 31], help="Top and bottom target floors")
    parser.add_argument("--methods", nargs=2, choices=METHODS, default=list(METHODS))
    parser.add_argument("--ffmpeg-threads", type=int, choices=(1, 2), default=2)
    parser.add_argument("--crf", type=int, default=19)
    parser.add_argument("--font", type=Path)
    args = parser.parse_args(argv)
    if len(set(args.floors)) != 2 or tuple(args.methods) != METHODS:
        parser.error("Select two distinct floors; methods must be base residual_xyz_smoothed, in that order")
    if not 0 <= args.crf <= 51:
        parser.error("CRF must be between 0 and 51")
    manifest_path = args.manifest.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status", "complete") != "complete":
        raise ValueError("The source render manifest must be complete before composition")
    cases = select_cases(manifest, manifest_path.parent, args.floors, args.methods)
    fonts, font_path = load_fonts(args.font)
    geometry = layout_for(cases, args.methods)
    background = make_background(cases, args.methods, fonts, geometry)
    common_end = max(row["sim_seconds"] for case in cases for row in case["methods"].values())
    requested_duration = common_end + END_HOLD_SECONDS
    frame_count = math.ceil(requested_duration * FPS - 1e-8)
    output_times = [index / FPS for index in range(frame_count)]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "schema_version": 1, "status": "composing", "created_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": identity(manifest_path), "source_script": identity(__file__),
        "shared_reader_script": identity(common.__file__), "font": font_path,
        "dimensions": [WIDTH, HEIGHT], "fps": FPS, "playback_speed": 1.0,
        "floors_top_to_bottom": args.floors, "methods_left_to_right": args.methods,
        "method_labels": METHOD_LABELS, "geometry": geometry,
        "frame_count": frame_count, "duration_seconds": frame_count / FPS,
        "common_terminal_seconds": common_end, "end_hold_seconds": END_HOLD_SECONDS,
        "requested_duration_seconds": requested_duration,
        "duration_rounding_seconds": frame_count / FPS - requested_duration,
        "output_frame_seconds": output_times,
        "timing": "Each output frame at index/30 shows the latest measured frame at or before that time. "
                  "No motion interpolation or time compression. Each terminal frame is held after termination. "
                  "Duration is the longest measured episode plus one second, rounded up to a 30 Hz boundary.",
        "selection": "Only target floors select episodes; successes and failures are both retained. "
                     "Each row is an independent button task, with all four trajectories starting at t=0.",
        "ffmpeg_threads_per_encoder": args.ffmpeg_threads, "crf": args.crf,
        "episodes": [],
    }
    common.write_json(output / "normalized_input.json", {"source": str(manifest_path), "cases": cases})
    with ExitStack() as stack:
        readers = {}
        for row_index, case in enumerate(cases):
            for col, method in enumerate(args.methods):
                row = case["methods"][method]
                sources = {}
                measured_times = None
                for view in ("global", "wrist"):
                    path = Path(row[f"{view}_video"])
                    metadata = common.probe(path)
                    if abs(metadata["fps"] - row["fps"]) > 1e-6:
                        raise ValueError(f"Manifest/video fps differ: {path}")
                    times = common.frame_times(row, metadata["frame_count"])
                    if measured_times is not None and times != measured_times:
                        raise ValueError(f"Camera timestamps differ: {case['case_id']} {method}")
                    measured_times = times
                    readers[(case["case_id"], method, view)] = stack.enter_context(CameraReader(path, metadata, times))
                    sources[view] = {**metadata, **identity(path)}
                mapping = [max(0, bisect_right(measured_times, seconds + 1e-9) - 1) for seconds in output_times]
                terminal_output_frame = mapping.index(len(measured_times) - 1)
                record["episodes"].append({
                    "row": row_index, "column": col, "floor": case["floor"], "case_id": case["case_id"],
                    "method": method, "label": METHOD_LABELS[method],
                    "outcome": {key: row.get(key) for key in
                                ("success", "termination", "pressed_floors", "sim_seconds", "physics_index")},
                    "source_identity": {key: row.get(key) for key in
                                        ("run_id", "source_env_id", "source_trajectory", "source_trajectory_sha256",
                                         "source_metadata_sha256", "source_metadata", "source_physics", "seed", "reset_episode")},
                    "source_videos": sources, "source_frame_seconds": measured_times,
                    "source_physics_indices": row.get("physics_indices"),
                    "source_frame_for_each_output_frame": mapping,
                    "first_terminal_output_frame": terminal_output_frame,
                    "first_terminal_output_seconds": terminal_output_frame / FPS,
                    "terminal_display_delay_seconds": terminal_output_frame / FPS - row["sim_seconds"],
                    "held_terminal_frame_count": frame_count - terminal_output_frame,
                    "held_terminal_duration_seconds": (frame_count - terminal_output_frame) / FPS,
                })
        common.write_json(output / "composition_manifest.json", record)
        writer = VideoWriter(output / "two_button_comparison.mp4", args.ffmpeg_threads, args.crf)
        try:
            for index, seconds in enumerate(output_times):
                image = compose_frame(background, cases, args.methods, readers, geometry, fonts, seconds)
                writer.write(image)
                if index == 0:
                    image.save(output / "start.jpg", quality=95)
                if index == frame_count - 1:
                    image.save(output / "end.jpg", quality=95)
                if index % 150 == 0:
                    print(json.dumps({"event": "compose_progress", "frame": index, "total_frames": frame_count}), flush=True)
            record["video"] = writer.finish()
        except BaseException:
            writer.abort()
            raise
    record["status"] = "complete"
    record["completed_at"] = datetime.now(timezone.utc).isoformat()
    record["preview_images"] = {name: identity(output / f"{name}.jpg") for name in ("start", "end")}
    common.write_json(output / "composition_manifest.json", record)
    print(json.dumps({"event": "complete", "output": str(output), "video": record["video"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
