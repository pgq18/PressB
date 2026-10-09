#!/usr/bin/env python3
"""Compose measured, time-aligned VLA-JEPA/RL rollout videos without a GPU.

Input is a JSON object with ``fps`` and ``cases``. Each case contains ``floor``,
``case_id`` (optional), ``layout``, ``seed`` and ``methods``. The latter maps
base/action_residual/initial_noise/combined/residual_on_noise/residual_fullscale/residual_xyz/residual_xyz_smoothed to objects
containing global_video, wrist_video, sim_seconds, success, termination,
pressed_floors and optionally frame_sim_seconds. Paths are relative to the JSON
file. A flat ``episodes`` or ``videos`` list with a ``method`` field is also accepted.

Explicit frame_sim_seconds is preferred: it preserves an exact terminal frame
between regular 30 Hz samples. If absent, frames are assumed to be t=0, regular
1/fps samples, and the exact terminal frame; their count must match that rule.
No motion is interpolated. The latest measured frame at or before each output
time is shown. Ended episodes freeze, are marked as ended, and never loop.

Example:
  .conda/envs/pressb/bin/python scripts/compose_rl_comparison_videos.py \
    --manifest outputs/.../comparison_input.json --output outputs/.../videos
  # Compare the frozen noise-then-residual composition with the original model:
  ... --manifest outputs/.../render_manifest.json --output outputs/.../videos \
    --methods base combined
  # Compare the new residual trained for 400k transitions on frozen learned noise:
  ... --manifest outputs/.../render_manifest.json --output outputs/.../videos \
    --methods base residual_on_noise
  # Compare original and half residual scales with the same frozen noise policy:
  ... --manifest outputs/.../render_manifest.json --output outputs/.../videos \
    --methods residual_fullscale residual_on_noise
  # Compare 9D pose residuals with position-only XYZ residuals:
  ... --manifest outputs/.../render_manifest.json --output outputs/.../videos \
    --methods residual_fullscale residual_xyz
  # Compare XYZ residual execution before and after action postprocessing:
  ... --manifest outputs/.../render_manifest.json --output outputs/.../videos \
    --methods residual_xyz residual_xyz_smoothed
  # Override displayed method names using a JSON object of method -> label:
  ... --method-labels outputs/.../method_labels.json
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from contextlib import ExitStack
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

from PIL import Image, ImageDraw, ImageFont


METHODS = ("base", "action_residual", "initial_noise")
METHOD_NAMES = METHODS + ("combined", "residual_on_noise", "residual_fullscale", "residual_xyz", "residual_xyz_smoothed")
LABELS = {
    "en": {"base": "Before RL: VLA-JEPA", "action_residual": "After RL: action residual",
           "initial_noise": "After RL: initial noise", "combined": "After RL: noise + action residual",
           "residual_on_noise": "Frozen noise + new residual (400k)",
           "residual_fullscale": "Frozen noise + 9D residual (original scale)",
           "residual_xyz": "XYZ residual: no postprocessing",
           "residual_xyz_smoothed": "XYZ residual: action smoothing"},
    "zh": {"base": "强化学习前：VLA-JEPA", "action_residual": "强化学习后：动作残差",
           "initial_noise": "强化学习后：初始噪声", "combined": "强化学习后：初始噪声＋动作残差",
           "residual_on_noise": "冻结噪声＋新训动作残差（400k）",
           "residual_fullscale": "冻结噪声＋9D 动作残差（原 scale）",
           "residual_xyz": "XYZ 残差：未加后处理",
           "residual_xyz_smoothed": "XYZ 残差：动作平滑后处理"},
}
COLORS = {"base": "#62748a", "action_residual": "#3188d0", "initial_noise": "#d87c32",
          "combined": "#308f73", "residual_on_noise": "#7654ad", "residual_fullscale": "#3188d0",
          "residual_xyz": "#308f73", "residual_xyz_smoothed": "#a17c35"}
PAIR_METHODS = {"before_vs_action_residual": ("base", "action_residual"),
                "before_vs_initial_noise": ("base", "initial_noise"),
                "before_vs_combined": ("base", "combined"),
                "before_vs_residual_on_noise": ("base", "residual_on_noise"),
                "fullscale_vs_halfscale": ("residual_fullscale", "residual_on_noise"),
                "pose9_vs_xyz": ("residual_fullscale", "residual_xyz"),
                "before_vs_residual_xyz": ("base", "residual_xyz"),
                "xyz_vs_smoothed": ("residual_xyz", "residual_xyz_smoothed")}
BG, FG, MUTED = "#101922", "#f3f6fa", "#b4c1cf"
WIDTH, VIEW_HEIGHT = 640, 480
HEADER, METHOD_HEADER, FOOTER = 64, 48, 80
HEIGHT = HEADER + METHOD_HEADER + 2 * VIEW_HEIGHT + FOOTER


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def probe(path):
    result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,nb_frames,avg_frame_rate,codec_name,pix_fmt,duration",
        "-of", "json", str(path)], check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"Expected one video stream: {path}")
    stream = streams[0]
    if not str(stream.get("nb_frames", "")).isdigit():
        result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
            "-count_frames", "-show_entries", "stream=nb_read_frames", "-of", "json", str(path)],
            check=True, capture_output=True, text=True)
        stream["nb_frames"] = json.loads(result.stdout)["streams"][0]["nb_read_frames"]
    stream["frame_count"] = int(stream["nb_frames"])
    stream["fps"] = float(Fraction(stream["avg_frame_rate"]))
    return stream


def safe_case_id(value):
    result = str(value)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", result):
        raise ValueError(f"Unsafe case ID: {value!r}")
    return result


def validate_methods(methods):
    methods = tuple(methods)
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("Select at least one policy method, without duplicates")
    if any(method not in METHOD_NAMES for method in methods):
        raise ValueError(f"Unknown policy method; choose from {METHOD_NAMES}")
    return methods


def selected_pairs(methods):
    return {name: pair for name, pair in PAIR_METHODS.items()
            if all(method in methods for method in pair)}


def resolve_method_labels(language, overrides=None):
    labels = dict(LABELS[language])
    if overrides is not None:
        if not isinstance(overrides, dict):
            raise ValueError("Method labels must be a JSON object mapping method names to labels")
        for method, label in overrides.items():
            if method not in METHOD_NAMES:
                raise ValueError(f"Unknown method label key: {method!r}")
            if not isinstance(label, str) or not label.strip() or any(char in label for char in "\r\n"):
                raise ValueError(f"Method label for {method!r} must be a nonempty single-line string")
            labels[method] = label
    return labels


def normalize_manifest(manifest, root, methods=METHODS):
    """Accept either the comparison schema or the replay renderer's flat list."""
    methods = validate_methods(methods)
    cases = manifest.get("cases")
    if cases is None:
        rows = manifest.get("episodes", manifest.get("videos", manifest.get("results")))
        if not isinstance(rows, list):
            raise ValueError("Manifest needs a cases list or a flat episodes/videos list")
        grouped = {}
        for raw in rows:
            row = dict(raw)
            method = row["method"]
            if method not in methods:
                continue
            case_id = row.get("case_id", f"floor_{int(row['floor']):02d}_center")
            case = grouped.setdefault(case_id, {"case_id": case_id, "floor": int(row["floor"]),
                "layout": row.get("layout", {}), "seed": row.get("seed", manifest.get("seed")),
                "methods": {}})
            if method in case["methods"]:
                raise ValueError(f"Duplicate {method} for {case_id}; supply distinct case_id values")
            case["methods"][method] = row
        cases = list(grouped.values())
    if not isinstance(cases, list) or not cases:
        raise ValueError("At least one case is required")
    result, seen = [], set()
    for raw_case in cases:
        case = dict(raw_case)
        case["floor"] = int(case["floor"])
        case["case_id"] = safe_case_id(case.get("case_id", f"floor_{case['floor']:02d}_center"))
        if case["case_id"] in seen:
            raise ValueError(f"Duplicate case ID: {case['case_id']}")
        seen.add(case["case_id"])
        missing = set(methods) - set(case["methods"])
        if missing:
            raise ValueError(f"{case['case_id']} is missing selected policy methods: {sorted(missing)}")
        case["methods"] = {method: dict(case["methods"][method]) for method in methods}
        for method, row in case["methods"].items():
            if "floor" in row and int(row["floor"]) != case["floor"]:
                raise ValueError(f"Different target floors within {case['case_id']}")
            if "layout" in row:
                for key in ("offset_x_m", "offset_y_m"):
                    expected = float(case.get("layout", {}).get(key, 0))
                    if abs(float(row["layout"].get(key, 0)) - expected) > 1e-9:
                        raise ValueError(f"Different panel layouts within {case['case_id']}")
            info = row.get("info", {})
            for field in ("sim_seconds", "success", "termination", "pressed_floors", "physics_index"):
                if field not in row and field in info:
                    row[field] = info[field]
            row.setdefault("termination", row.get("termination_reason"))
            row.setdefault("pressed_floors", [])
            if type(row.get("success")) is not bool:
                raise ValueError(f"{case['case_id']} {method}: success must be boolean")
            row["sim_seconds"] = float(row["sim_seconds"])
            if not math.isfinite(row["sim_seconds"]) or row["sim_seconds"] <= 0:
                raise ValueError("sim_seconds must be finite and positive")
            for view in ("global", "wrist"):
                value = row.get(f"{view}_video", row.get(view))
                if value is None and isinstance(row.get("videos"), dict):
                    value = row["videos"].get(view)
                if not isinstance(value, str):
                    raise ValueError(f"Missing {view}_video for {case['case_id']} {method}")
                path = Path(value)
                path = path if path.is_absolute() else root / path
                path = path.resolve(strict=True)
                row[f"{view}_video"] = str(path)
            row["fps"] = float(row.get("fps", manifest.get("fps", 30)))
            if not math.isfinite(row["fps"]) or row["fps"] <= 0:
                raise ValueError("Source fps must be finite and positive")
        result.append(case)
    return sorted(result, key=lambda row: (row["floor"], row["case_id"]))


def frame_times(row, count):
    end = row["sim_seconds"]
    times = row.get("frame_sim_seconds", row.get("frame_times_seconds"))
    if times is None:
        fps = row["fps"]
        last_regular = int(math.floor(end * fps + 1e-8))
        times = [index / fps for index in range(last_regular + 1)]
        if end - times[-1] > 1e-8:
            times.append(end)
    times = [float(t) for t in times]
    if len(times) != count:
        raise ValueError(f"Frame count {count} differs from {len(times)} recorded timestamps")
    if any(not math.isfinite(t) for t in times) or abs(times[0]) > 1e-8:
        raise ValueError("Frame times must be finite and begin at zero")
    if any(second <= first for first, second in zip(times, times[1:])):
        raise ValueError("Frame times must be strictly increasing")
    if abs(times[-1] - end) > 1e-7:
        raise ValueError(f"Last frame {times[-1]} does not match exact termination {end}")
    return times


class FrameReader:
    """Sequential raw decoder; memory use stays independent of video duration."""
    def __init__(self, path, metadata, times):
        self.path, self.times = path, times
        self.width, self.height = metadata["width"], metadata["height"]
        if abs(self.width / self.height - 4 / 3) > 1e-8:
            raise ValueError(f"Expected an uncropped 4:3 camera frame: {path}")
        self.process = subprocess.Popen(["ffmpeg", "-v", "error", "-nostdin", "-threads", "1",
            "-i", str(path), "-map", "0:v:0", "-vsync", "0", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-threads", "1", "pipe:1"], stdout=subprocess.PIPE)
        self.index, self.image = -1, None

    def at(self, seconds):
        target = max(0, bisect_right(self.times, seconds + 1e-9) - 1)
        while self.index < target:
            remaining = self.width * self.height * 3
            blocks = []
            while remaining:
                block = self.process.stdout.read(remaining)
                if not block:
                    raise RuntimeError(f"Unexpected end of decoded video: {self.path}")
                blocks.append(block)
                remaining -= len(block)
            self.image = Image.frombytes("RGB", (self.width, self.height), b"".join(blocks))
            if self.image.size != (WIDTH, VIEW_HEIGHT):
                self.image = self.image.resize((WIDTH, VIEW_HEIGHT), Image.Resampling.LANCZOS)
            self.index += 1
        return self.image

    def close(self):
        if self.process.stdout:
            self.process.stdout.close()
        if self.process.poll() is None:
            self.process.terminate()
        self.process.wait(timeout=10)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class VideoWriter:
    def __init__(self, path, width, fps, threads, crf):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = self.path.with_name(self.path.stem + ".partial.mp4")
        self.count = 0
        self.process = subprocess.Popen(["ffmpeg", "-v", "error", "-nostdin", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s:v", f"{width}x{HEIGHT}",
            "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "fast",
            "-crf", str(crf), "-pix_fmt", "yuv420p", "-threads", str(threads),
            "-movflags", "+faststart", str(self.temporary)], stdin=subprocess.PIPE)

    def write(self, image):
        try:
            self.process.stdin.write(image.tobytes())
        except BrokenPipeError as error:
            raise RuntimeError(f"Video encoder failed: {self.path}") from error
        self.count += 1

    def finish(self):
        self.process.stdin.close()
        returncode = self.process.wait(timeout=120)
        if returncode:
            raise RuntimeError(f"Video encoder failed ({returncode}): {self.path}")
        metadata = probe(self.temporary)
        if metadata["frame_count"] != self.count:
            raise RuntimeError(f"Encoded frame count differs for {self.path}")
        self.temporary.replace(self.path)
        return metadata

    def abort(self):
        if self.process.stdin and not self.process.stdin.closed:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
        if self.process.poll() is None:
            self.process.terminate()
        self.process.wait(timeout=10)


@dataclass
class Fonts:
    title: object
    method: object
    normal: object
    small: object


def load_fonts(language, custom=None):
    candidates = ([Path(custom)] if custom else []) + ([
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc")] if language == "zh" else [])
    candidates += [Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]
    path = next((path for path in candidates if path.is_file()), None)
    if path is None:
        raise FileNotFoundError("No suitable font; supply --font")
    if language == "zh" and not custom and "CJK" not in path.name:
        raise FileNotFoundError("Chinese labels need Noto Sans CJK; use --language en or --font")
    return Fonts(*(ImageFont.truetype(str(path), size) for size in (25, 25, 22, 18))), str(path)


def fit_text(draw, text, font, width):
    if draw.textlength(text, font=font) <= width:
        return text
    while text and draw.textlength(text + "…", font=font) > width:
        text = text[:-1]
    return text + "…"


def draw_text(draw, xy, text, font, fill=FG, width=None):
    if width is not None:
        text = fit_text(draw, text, font, width)
    draw.text(xy, text, font=font, fill=fill, anchor="lt")


def header(draw, case, seconds, fonts, language, width):
    floor = case["floor"]
    layout = case.get("layout", {})
    x, y = layout.get("offset_x_m", 0), layout.get("offset_y_m", 0)
    centered = abs(float(x)) < 1e-8 and abs(float(y)) < 1e-8
    if language == "zh":
        title = f"目标：{floor} 楼   |   " + ("面板居中" if centered else f"面板偏移 ({x:g}, {y:g}) m")
        subtitle = "实测轨迹回放 · 原速 1× · 按仿真时间同步"
    else:
        title = f"Target: floor {floor}   |   " + ("Centered panel" if centered else f"Panel ({x:g}, {y:g}) m")
        subtitle = "Measured trajectory replay | 1x | aligned by simulation time"
    draw_text(draw, (16, 8), title, fonts.title, width=width - 170)
    draw_text(draw, (16, 40), subtitle, fonts.small, MUTED, width=width - 20)
    text = f"t = {seconds:05.2f} s"
    draw_text(draw, (width - 160, 11), text, fonts.normal)


def status_label(row, ended, language):
    if not ended:
        return ("执行中", "#77b8f2") if language == "zh" else ("RUNNING", "#77b8f2")
    floors = ", ".join(str(value) for value in row.get("pressed_floors", []))
    if row["success"]:
        return ((f"成功 · 按中 {floors} 楼" if floors else "成功"), "#62d1a3") if language == "zh" else (
            "SUCCESS" + (f" | pressed {floors}" if floors else ""), "#62d1a3")
    reason = row.get("termination", "unknown")
    zh = {"wrong_button_pressed": "失败 · 按错按钮", "time_limit": "失败 · 超时",
          "unexpected_collision": "失败 · 意外碰撞"}
    en = {"wrong_button_pressed": "FAILED | wrong button", "time_limit": "FAILED | timeout",
          "unexpected_collision": "FAILED | collision"}
    text = (zh if language == "zh" else en).get(reason, f"{'失败' if language == 'zh' else 'FAILED'} | {reason}")
    if floors:
        text += f" ({floors})"
    return text, "#f4a087"


def make_column(case, method, seconds, readers, fonts, language, common_end, method_labels=None):
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    header(draw, case, min(seconds, common_end), fonts, language, WIDTH)
    draw.rectangle((0, HEADER, WIDTH, HEADER + METHOD_HEADER), fill=COLORS[method])
    label = (method_labels if method_labels is not None else LABELS[language])[method]
    draw_text(draw, (16, HEADER + 11), label, fonts.method, width=WIDTH - 30)
    for index, view in enumerate(("global", "wrist")):
        top = HEADER + METHOD_HEADER + index * VIEW_HEIGHT
        image.paste(readers[view].at(seconds), (0, top))
        name = ({"global": "全局视角", "wrist": "腕部视角"} if language == "zh"
                else {"global": "GLOBAL CAMERA", "wrist": "WRIST CAMERA"})[view]
        label_width = int(draw.textlength(name, font=fonts.small)) + 18
        draw.rectangle((8, top + 8, 8 + label_width, top + 37), fill=BG)
        draw_text(draw, (17, top + 13), name, fonts.small)
    row = case["methods"][method]
    end = row["sim_seconds"]
    ended = seconds + 1e-9 >= end
    label, color = status_label(row, ended, language)
    top = HEIGHT - FOOTER
    draw_text(draw, (16, top + 12), label, fonts.normal, color, width=WIDTH - 30)
    note = ("已终止 · 保留末帧" if language == "zh" else "TERMINATED | final frame held") if ended else (
        "正在执行任务" if language == "zh" else "Episode in progress")
    draw_text(draw, (16, top + 45), note, fonts.small, MUTED, width=WIDTH - 210)
    draw_text(draw, (WIDTH - 180, top + 46), f"t = {min(seconds, end):.3f} s", fonts.small)
    return image


def compose_case(case, output, fps, hold, fonts, language, threads, crf, methods=METHODS, method_labels=None):
    methods = validate_methods(methods)
    method_labels = resolve_method_labels(language, method_labels)
    pairs = selected_pairs(methods)
    comparison_width = len(methods) * WIDTH
    directory = output / "cases" / case["case_id"]
    directory.mkdir(parents=True, exist_ok=False)
    common_end = max(case["methods"][method]["sim_seconds"] for method in methods)
    count = int(math.ceil(common_end * fps - 1e-8)) + round(hold * fps) + 1
    sources, results = {}, {}
    with ExitStack() as stack:
        readers = {}
        for method in methods:
            row = case["methods"][method]
            readers[method], sources[method] = {}, {}
            for view in ("global", "wrist"):
                path = Path(row[f"{view}_video"])
                metadata = probe(path)
                if abs(metadata["fps"] - row["fps"]) > 1e-6:
                    raise ValueError(f"Manifest/video fps differ: {path}")
                times = frame_times(row, metadata["frame_count"])
                readers[method][view] = stack.enter_context(FrameReader(path, metadata, times))
                sources[method][view] = {"path": str(path), "sha256": sha256(path), **metadata}
        writers = {"comparison": VideoWriter(directory / "comparison.mp4", comparison_width, fps, threads, crf)}
        try:
            for method in methods:
                writers[method] = VideoWriter(directory / f"{method}.mp4", WIDTH, fps, threads, crf)
            for name in pairs:
                writers[name] = VideoWriter(directory / f"{name}.mp4", 2 * WIDTH, fps, threads, crf)
            for index in range(count):
                seconds = index / fps
                combined = Image.new("RGB", (comparison_width, HEIGHT), BG)
                columns = {}
                for col, method in enumerate(methods):
                    single = make_column(case, method, seconds, readers[method], fonts, language, common_end, method_labels)
                    columns[method] = single
                    writers[method].write(single)
                    combined.paste(single, (col * WIDTH, 0))
                draw = ImageDraw.Draw(combined)
                draw.rectangle((0, 0, comparison_width, HEADER - 1), fill=BG)
                header(draw, case, min(seconds, common_end), fonts, language, comparison_width)
                for col in range(1, len(methods)):
                    draw.line((col * WIDTH, HEADER, col * WIDTH, HEIGHT), fill=BG, width=2)
                writers["comparison"].write(combined)
                for name, pair in pairs.items():
                    paired = Image.new("RGB", (2 * WIDTH, HEIGHT), BG)
                    for col, method in enumerate(pair):
                        paired.paste(columns[method], (col * WIDTH, 0))
                    pair_draw = ImageDraw.Draw(paired)
                    pair_draw.rectangle((0, 0, 2 * WIDTH, HEADER - 1), fill=BG)
                    header(pair_draw, case, min(seconds, common_end), fonts, language, 2 * WIDTH)
                    pair_draw.line((WIDTH, HEADER, WIDTH, HEIGHT), fill=BG, width=2)
                    writers[name].write(paired)
                    if index == count - 1:
                        paired.save(directory / f"{name}_end.jpg", quality=93)
                if index == 0:
                    combined.save(directory / "start.jpg", quality=93)
                if index == count - 1:
                    combined.save(directory / "end.jpg", quality=93)
            for name, writer in writers.items():
                results[name] = {"path": str(writer.path), **writer.finish()}
        except BaseException:
            for writer in writers.values():
                writer.abort()
            raise
    value = {"case_id": case["case_id"], "floor": case["floor"], "frame_count": count,
        "fps": fps, "methods": list(methods), "method_labels": {method: method_labels[method] for method in methods},
        "duration_seconds": count / fps, "common_terminal_seconds": common_end,
        "end_hold_seconds": hold, "sources": sources, "results": results,
        "outcomes": {method: {key: case["methods"][method].get(key) for key in
            ("sim_seconds", "success", "termination", "pressed_floors", "physics_index")}
            for method in methods}}
    write_json(directory / "composition.json", value)
    return value


def concatenate(paths, destination, expected_count):
    # ffconcat single-quote escaping; subprocess receives an argument array.
    list_path = destination.with_suffix(".ffconcat")
    lines = ["ffconcat version 1.0"]
    for path in paths:
        escaped = str(Path(path).resolve()).replace("'", "'\\''")
        if "\n" in escaped or "\r" in escaped:
            raise ValueError("Newlines are not allowed in video paths")
        lines.append(f"file '{escaped}'")
    list_path.write_text("\n".join(lines) + "\n")
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-n", "-f", "concat", "-safe", "0",
        "-i", str(list_path), "-c", "copy", "-movflags", "+faststart", str(destination)], check=True)
    metadata = probe(destination)
    if metadata["frame_count"] != expected_count:
        raise RuntimeError(f"Concatenation lost frames: {destination}")
    return {"path": str(destination), "sha256": sha256(destination), **metadata}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing outputs are never overwritten")
    parser.add_argument("--methods", nargs="+", choices=METHOD_NAMES, default=METHODS,
                        help="Policies in left-to-right order; defaults to the original three policies")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--end-hold", type=float, default=1.)
    parser.add_argument("--ffmpeg-threads", type=int, default=2, help="Threads per encoder; selected views encode concurrently")
    parser.add_argument("--crf", type=int, default=19)
    parser.add_argument("--language", choices=("zh", "en"), default="zh")
    parser.add_argument("--method-labels", type=Path,
                        help="JSON object mapping method names to display labels; overrides the selected language's defaults")
    parser.add_argument("--font", type=Path)
    parser.add_argument("--scope", default="New fixed-condition final-policy recordings; these videos do not replace the original 120-episode evaluation",
                        help="Recorded evaluation scope/provenance written into the composition manifest")
    parser.add_argument("--require-all-center-floors", action="store_true", help="Require exactly floors 24 through 35, all centered")
    args = parser.parse_args()
    if args.fps < 1 or args.ffmpeg_threads < 1 or not 0 <= args.crf <= 51:
        parser.error("Invalid fps, ffmpeg thread count, or CRF")
    if not math.isfinite(args.end_hold) or args.end_hold < 0:
        parser.error("end-hold must be finite and nonnegative")
    try:
        methods = validate_methods(args.methods)
        labels_path = args.method_labels.resolve(strict=True) if args.method_labels else None
        method_labels = resolve_method_labels(args.language, json.loads(labels_path.read_text()) if labels_path else None)
    except ValueError as error:
        parser.error(str(error))
    pairs = selected_pairs(methods)
    manifest_path = args.manifest.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text())
    cases = normalize_manifest(manifest, manifest_path.parent, methods)
    if args.require_all_center_floors:
        if [row["floor"] for row in cases] != list(range(24, 36)):
            raise ValueError("Expected each floor 24–35 exactly once")
        for case in cases:
            layout = case.get("layout", {})
            if any(abs(float(layout.get(key, 0))) > 1e-8 for key in ("offset_x_m", "offset_y_m")):
                raise ValueError("Every requested layout must be centered")
    fonts, font_path = load_fonts(args.language, args.font)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    selected_labels = {method: method_labels[method] for method in methods}
    write_json(output / "normalized_input.json", {"source": str(manifest_path), "methods": list(methods),
                                                "method_labels": selected_labels, "cases": cases})
    record = {"source_manifest": str(manifest_path), "source_manifest_sha256": sha256(manifest_path),
        "script_sha256": sha256(Path(__file__)), "fps": args.fps, "end_hold_seconds": args.end_hold,
        "language": args.language, "font": font_path, "methods": list(methods), "method_labels": selected_labels,
        "method_labels_source": str(labels_path) if labels_path else None,
        "dimensions": [len(methods) * WIDTH, HEIGHT],
        "method_dimensions": [WIDTH, HEIGHT], "ffmpeg_threads_per_encoder": args.ffmpeg_threads,
        "timing": "Real-time simulation alignment; latest measured frame at or before output time; no motion interpolation; terminal frames held explicitly",
        "scope": args.scope,
        "cases": [], "collections": {}}
    for case in cases:
        print(json.dumps({"event": "compose_case", "case_id": case["case_id"]}), flush=True)
        result = compose_case(case, output, args.fps, args.end_hold, fonts, args.language,
                              args.ffmpeg_threads, args.crf, methods, method_labels)
        record["cases"].append(result)
        write_json(output / "composition_manifest.json", record)
        print(json.dumps({"event": "case_complete", "case_id": case["case_id"],
                          "frames": result["frame_count"]}), flush=True)
    frame_count = sum(row["frame_count"] for row in record["cases"])
    for name in ("comparison",) + methods + tuple(pairs):
        paths = [row["results"][name]["path"] for row in record["cases"]]
        filename = f"{name}.mp4" if name in pairs else f"{name}_all_floors.mp4"
        record["collections"][name] = concatenate(paths, output / filename, frame_count)
    write_json(output / "composition_manifest.json", record)
    print(json.dumps({"event": "complete", "output": str(output), "cases": len(cases),
                      "collections": record["collections"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
