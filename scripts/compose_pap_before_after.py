#!/usr/bin/env python3
"""Show PAP source RGB beside two Isaac material/lighting revisions, at 1x.

The three columns are real / simulation before / simulation after. Each column
contains the original 640x480 up and wrist frames; no frame resizing, cropping,
motion interpolation, terminal holds, or fitted temporal shifts are applied.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont

from compose_pap_comparison import Reader, source_episode, common

FPS, WIDTH, HEIGHT = 30, 1920, 1080
CELL_WIDTH, CELL_HEIGHT, HEADER = 640, 480, 80
BG, FG, MUTED = "#101922", "#f3f6fa", "#c0ccd8"


def simulation_sources(directory, episode):
    directory = directory.resolve(strict=True)
    report_path = directory / "report.json"
    report = json.loads(report_path.read_text())
    semantics = report["trajectory_semantics"]
    if (report.get("mode") != "replay" or semantics.get("arm_source") != "observation.state"
            or semantics.get("gripper_source") != "action"):
        raise ValueError(f"Expected measured-arm plus commanded-gripper replay: {directory}")
    trajectory = Path(report["trajectory"])
    if trajectory.stem != f"episode_{episode:06d}":
        raise ValueError(f"Simulation trajectory does not match episode {episode}: {trajectory}")
    expected = round(report["duration_s"]*FPS)+1
    if expected < 1 or abs((expected-1)/FPS-report["duration_s"]) > 1e-7:
        raise ValueError(f"Simulation duration is not on the regular 30 Hz timeline: {directory}")
    cameras = {}
    for camera in ("up", "wrist"):
        path = directory / f"{camera}.mp4"
        metadata = common.probe(path)
        if (metadata["frame_count"] != report["video_frames"][camera]
                or metadata["frame_count"] != expected
                or metadata["fps"] != FPS
                or (metadata["width"], metadata["height"]) != (CELL_WIDTH, CELL_HEIGHT)):
            raise ValueError(f"Simulation video geometry/timing differs from report: {path}")
        cameras[camera] = dict(path=str(path), metadata=metadata)
    evidence = dict(report=str(report_path), trajectory=str(trajectory.resolve(strict=True)),
                    trajectory_sha256=common.sha256(trajectory), semantics=semantics,
                    duration_s=report["duration_s"], source_frames=report["source_frames"],
                    physical_success=report.get("physical_success"))
    return cameras, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font", type=Path)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--ffmpeg-threads", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    dataset = args.dataset.resolve(strict=True)
    real = source_episode(dataset, args.episode)
    before, before_evidence = simulation_sources(args.before, args.episode)
    after, after_evidence = simulation_sources(args.after, args.episode)
    if before_evidence["trajectory_sha256"] != after_evidence["trajectory_sha256"]:
        raise ValueError("The before/after simulations must replay identical recorded trajectories")
    for evidence in (before_evidence, after_evidence):
        if evidence["source_frames"] != real["up"]["frame_count"]:
            raise ValueError("Simulation source trajectory length differs from the selected real episode")
    count = min(*(camera["frame_count"] for camera in real.values()),
                *(camera["metadata"]["frame_count"] for source in (before, after) for camera in source.values()))
    if count < 1:
        raise ValueError("No common source frames")
    _, font_path = common.load_fonts("zh", args.font)
    fonts = {key: ImageFont.truetype(font_path, size) for key, size in
             {"title": 27, "column": 23, "clock": 27, "note": 16}.items()}
    args.output.mkdir(parents=True, exist_ok=False)
    background = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(background)
    draw.text((16, 8), f"Piper 抓取入盒 · 材质与光照对比 · Episode {args.episode}", font=fonts["title"], fill=FG)
    for column, title in enumerate(("原始真机", "修改前仿真", "材质光照调整后")):
        draw.text((column*CELL_WIDTH+16, 48), title+"｜上：第三视角　下：腕部", font=fonts["column"], fill=FG)
    draw.text((16, 1042), "仿真：实测六轴轨迹驱动＋记录夹爪命令；非策略评测。三个画面均为原始相机图像，无画面后处理。",
              font=fonts["note"], fill=FG, anchor="lt")
    draw.text((16, 1061), "名义时间对齐，原速 1×；相机与状态仍有约一帧级时延不确定性；截取各来源最短共同长度。",
              font=fonts["note"], fill=MUTED, anchor="lt")
    partial = args.output / "real_before_after.partial.mp4"
    final = args.output / "real_before_after.mp4"
    encoder = subprocess.Popen([
        "ffmpeg", "-v", "error", "-nostdin", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s:v", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "pipe:0", "-an",
        "-c:v", "libx264", "-preset", "fast", "-crf", str(args.crf), "-pix_fmt", "yuv420p",
        "-threads", str(args.ffmpeg_threads), "-movflags", "+faststart", str(partial),
    ], stdin=subprocess.PIPE)
    snapshots = {}
    try:
        with ExitStack() as stack:
            readers = {(column, row): stack.enter_context(Reader(item["path"], item.get("frame_start", 0)))
                       for column, sources in enumerate((real, before, after))
                       for row, camera in enumerate(("up", "wrist")) for item in [sources[camera]]}
            for index in range(count):
                canvas = background.copy()
                for (column, row), reader in readers.items():
                    canvas.paste(reader.next(), (column*CELL_WIDTH, HEADER+row*CELL_HEIGHT))
                draw = ImageDraw.Draw(canvas)
                clock = f"t = {index/FPS:05.2f} s"
                draw.text((WIDTH-16-draw.textlength(clock, font=fonts["clock"]), 8), clock, font=fonts["clock"], fill=FG)
                names = []
                if index == 0:
                    names.append("initial_comparison.jpg")
                if index == count-1:
                    names.append("final_comparison.jpg")
                if index in (185, 216):
                    names.append(f"frame_{index:04d}_comparison.jpg")
                for name in names:
                    canvas.save(args.output / name, quality=95)
                    snapshots[name] = dict(frame_index=index, nominal_time_s=index/FPS)
                encoder.stdin.write(canvas.tobytes())
                if index % 120 == 0:
                    print(json.dumps(dict(frame=index, total=count, seconds=index/FPS)), flush=True)
        encoder.stdin.close()
        if encoder.wait(timeout=60):
            raise RuntimeError("ffmpeg encoder failed")
    except BaseException:
        if encoder.poll() is None:
            encoder.terminate()
            encoder.wait(timeout=10)
        raise
    encoded = common.probe(partial)
    if (encoded["frame_count"] != count or encoded["fps"] != FPS
            or (encoded["width"], encoded["height"]) != (WIDTH, HEIGHT)):
        raise RuntimeError("Encoded video geometry/timing mismatch")
    # Decode every encoded output frame; failure leaves the partial file for diagnosis.
    subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-threads", str(args.ffmpeg_threads),
                    "-i", str(partial), "-f", "null", "-"], check=True, timeout=60)
    partial.replace(final)
    manifest = dict(episode=args.episode, source_dataset=str(dataset), real_sources=real,
        before_sources=before, after_sources=after, before=before_evidence, after=after_evidence,
        fps=FPS, frames=count, first_nominal_time_s=0, last_nominal_time_s=(count-1)/FPS,
        encoded_duration_s=count/FPS, output=str(final.resolve()), output_metadata=encoded,
        snapshots=snapshots, full_output_decode_verified=True,
        layout=dict(width=WIDTH, height=HEIGHT, cell_size=[CELL_WIDTH, CELL_HEIGHT], header_height=HEADER,
                    footer_height=40, columns=["real", "before", "after"], rows=["up", "wrist"]),
        playback="One source frame per output frame; 1x nominal time; no interpolation, speed matching, or terminal holds",
        duration_rule="Minimum of real episode length and all four simulated camera frame counts",
        image_processing="Uncropped and unresized 640x480 RGB source frames; labels outside image cells; no image grading; H.264 encoding only",
        source_video_indexing="Verified packed episode lengths and constant metadata offset; cumulative source frame offsets",
        synchronization="Nominal timestamps only; sensor/state latency uncertain at approximately one frame; no fitted shift applied",
        policy_evaluation=False, font=font_path)
    common.write_json(args.output / "composition_manifest.json", manifest)
    print(json.dumps(dict(output=str(final.resolve()), frames=count, full_decode_passed=True)), flush=True)


if __name__ == "__main__":
    main()
