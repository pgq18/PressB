#!/usr/bin/env python3
"""Compose source PAP RGB and an Isaac trajectory replay on a nominal 30 Hz clock.

Uses packed episode frame indices, preserving the source pixels and recorded
speed. It neither estimates sensor latency nor interpolates motion. The video
ends when either source ends; no next-episode frames or terminal holds are added.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import sys

import av
from PIL import Image, ImageDraw, ImageFont

import compose_rl_comparison_videos as common

ROOT = Path(__file__).resolve().parents[1]
try:
    import pyarrow.parquet as pq
except ImportError:
    for directory in (ROOT / ".conda/envs/vlajepa-inference/lib").glob("python*/site-packages"):
        if (directory / "pyarrow").exists():
            sys.path.append(str(directory))
            break
    import pyarrow.parquet as pq

FPS, WIDTH, HEIGHT = 30, 1280, 1080
CELL_WIDTH, CELL_HEIGHT, HEADER = 640, 480, 80
BG, FG, MUTED = "#101922", "#f3f6fa", "#c0ccd8"


def source_episode(dataset, episode):
    info = json.loads((dataset / "meta/info.json").read_text())
    if info["fps"] != FPS:
        raise ValueError("This compositor requires the original 30 Hz PAP data")
    episodes = []
    for path in sorted((dataset / "meta/episodes").rglob("*.parquet")):
        episodes.extend(pq.read_table(path).to_pylist())
    matches = [row for row in episodes if row["episode_index"] == episode]
    if len(matches) != 1:
        raise ValueError(f"Expected one episode {episode}, found {len(matches)}")
    selected = matches[0]
    cameras = {}
    for camera in ("up", "wrist"):
        key = f"observation.images.{camera}"
        chunk, index = (selected[f"videos/{key}/{name}"] for name in ("chunk_index", "file_index"))
        packed = sorted((row for row in episodes if row[f"videos/{key}/chunk_index"] == chunk
                         and row[f"videos/{key}/file_index"] == index), key=lambda row: row["episode_index"])
        path = dataset / info["video_path"].format(video_key=key, chunk_index=chunk, file_index=index)
        probe = common.probe(path)
        if probe["frame_count"] != sum(row["length"] for row in packed):
            raise ValueError(f"Packed episode lengths do not match video frame count: {path}")
        start, offsets, selected_start = 0, [], None
        for row in packed:
            offsets.append(row[f"videos/{key}/from_timestamp"] - start/FPS)
            if row["episode_index"] == episode:
                selected_start = start
            start += row["length"]
        if max(offsets) - min(offsets) > 1e-7:
            raise ValueError(f"Video metadata offsets are not constant within {path}")
        cameras[camera] = dict(path=str(path.resolve()), frame_start=selected_start,
            frame_count=selected["length"], packed_video=probe,
            metadata_from_timestamp_s=selected[f"videos/{key}/from_timestamp"],
            packed_from_timestamp_s=selected_start/FPS, metadata_offset_s=offsets[0])
    return cameras


class Reader:
    def __init__(self, path, start=0):
        self.path, self.start = Path(path), start
        self.container = av.open(str(path))
        self.stream = self.container.streams.video[0]
        self.stream.thread_count = 1
        if ((self.stream.width, self.stream.height) != (CELL_WIDTH, CELL_HEIGHT)
                or abs(float(self.stream.average_rate)-FPS) > 1e-9
                or abs(float((self.stream.start_time or 0)*self.stream.time_base)) > 1e-9):
            self.container.close()
            raise ValueError(f"Expected 640×480, 30 fps, PTS zero: {path}")
        if start:
            self.container.seek(max(0, int((start/FPS-.1)/float(self.stream.time_base))), stream=self.stream)
        self.frames = self.container.decode(self.stream)
        self.next_index = start

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.container.close()

    def next(self):
        for frame in self.frames:
            seconds = float(frame.pts*self.stream.time_base)
            frame_index = round(seconds*FPS)
            if abs(seconds-frame_index/FPS) > 1e-7:
                raise ValueError(f"Non-uniform PTS in {self.path}")
            if frame_index < self.next_index:
                continue
            if frame_index != self.next_index:
                raise ValueError(f"Skipped frame in {self.path}: {frame_index} != {self.next_index}")
            self.next_index += 1
            return frame.to_image()
        raise EOFError(f"Unexpected end of {self.path}, needed {self.next_index}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--simulation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font", type=Path)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--ffmpeg-threads", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    dataset, simulation = args.dataset.resolve(strict=True), args.simulation.resolve(strict=True)
    report_path = simulation / "report.json"
    report = json.loads(report_path.read_text())
    semantics = report["trajectory_semantics"]
    if (report["mode"] != "replay" or semantics.get("arm_source") != "observation.state"
            or semantics.get("gripper_source") != "action"):
        raise ValueError("Video labels require measured six-joint replay with recorded gripper commands")
    if Path(report["trajectory"]).stem != f"episode_{args.episode:06d}":
        raise ValueError("Simulation trajectory episode does not match requested dataset episode")
    source = source_episode(dataset, args.episode)
    sim = {camera: dict(path=str(simulation / f"{camera}.mp4"),
                       metadata=common.probe(simulation / f"{camera}.mp4")) for camera in ("up", "wrist")}
    for camera, item in sim.items():
        metadata = item["metadata"]
        if metadata["frame_count"] != report["video_frames"][camera]:
            raise ValueError(f"Simulation frame count differs from report: {camera}")
        # The runner includes t=0, then every 1/30 s sample. A final extra
        # off-grid image must not be relabeled as another regular 30 Hz step.
        expected = round(report["duration_s"]*FPS)+1
        if (metadata["frame_count"] != expected
                or abs((expected-1)/FPS-report["duration_s"]) > 1e-7):
            raise ValueError("Simulation final frame is not on the regular 30 Hz timeline")
    count = min(*(item["frame_count"] for item in source.values()),
                *(item["metadata"]["frame_count"] for item in sim.values()))
    if count < 1:
        raise ValueError("No shared frames")
    _, font_path = common.load_fonts("zh", args.font)
    fonts = {name: ImageFont.truetype(font_path, size) for name, size in
             {"title": 27, "column": 23, "clock": 27, "note": 16}.items()}
    args.output.mkdir(parents=True, exist_ok=False)
    background = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(background)
    draw.text((16, 8), f"Piper 抓取入盒 · 真机与仿真 · Episode {args.episode}", font=fonts["title"], fill=FG)
    draw.text((16, 48), "真机数据｜上：第三视角　下：腕部", font=fonts["column"], fill=FG)
    draw.text((656, 48), "仿真重建｜上：第三视角　下：腕部", font=fonts["column"], fill=FG)
    draw.text((16, 1042), "仿真：实测六轴轨迹驱动＋记录夹爪命令；非策略评测", font=fonts["note"], fill=FG, anchor="lt")
    draw.text((16, 1061), "名义时间对齐，原速 1×；相机与状态仍有约一帧级时延不确定性。", font=fonts["note"], fill=MUTED, anchor="lt")
    partial = args.output / "real_vs_sim.partial.mp4"
    final = args.output / "real_vs_sim.mp4"
    encoder = subprocess.Popen([
        "ffmpeg", "-v", "error", "-nostdin", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s:v", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "pipe:0", "-an",
        "-c:v", "libx264", "-preset", "fast", "-crf", str(args.crf), "-pix_fmt", "yuv420p",
        "-threads", str(args.ffmpeg_threads), "-movflags", "+faststart", str(partial),
    ], stdin=subprocess.PIPE)
    try:
        with ExitStack() as stack:
            readers = {}
            for column, inputs in enumerate((source, sim)):
                for row, camera in enumerate(("up", "wrist")):
                    item = inputs[camera]
                    readers[(column, row)] = stack.enter_context(Reader(item["path"], item.get("frame_start", 0)))
            for index in range(count):
                canvas = background.copy()
                for (column, row), reader in readers.items():
                    canvas.paste(reader.next(), (column*CELL_WIDTH, HEADER+row*CELL_HEIGHT))
                draw = ImageDraw.Draw(canvas)
                clock = f"t = {index/FPS:05.2f} s"
                draw.text((WIDTH-16-draw.textlength(clock, font=fonts["clock"]), 8), clock, font=fonts["clock"], fill=FG)
                if index == 0:
                    canvas.save(args.output / "initial_comparison.jpg", quality=95)
                if index == count-1:
                    canvas.save(args.output / "final_comparison.jpg", quality=95)
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
    partial.replace(final)
    manifest = dict(episode=args.episode, source_dataset=str(dataset), simulation_report=str(report_path),
        real_sources=source, simulated_sources=sim, fps=FPS, frames=count,
        first_nominal_time_s=0, last_nominal_time_s=(count-1)/FPS, encoded_duration_s=count/FPS,
        output=str(final.resolve()), output_metadata=encoded,
        layout=dict(width=WIDTH, height=HEIGHT, cell_size=[CELL_WIDTH, CELL_HEIGHT],
                    header_height=HEADER, columns=["real", "simulation"], rows=["up", "wrist"]),
        playback="One source frame per output frame; 1x nominal time; no interpolation, speed matching, or terminal holds",
        duration_rule="Minimum of requested real episode length and both simulated camera frame counts",
        image_processing="Uncropped, unresized source RGB in 640x480 cells; text outside image cells; H.264 encoding only",
        source_video_indexing="Verified packed frame count and constant metadata offset; use cumulative episode frame offsets",
        synchronization="Nominal timestamps only; sensor/state latency uncertain at approximately one frame; no fitted shift applied",
        simulation_semantics=semantics, policy_evaluation=False, font=font_path)
    common.write_json(args.output / "composition_manifest.json", manifest)
    print(json.dumps(dict(output=str(final.resolve()), frames=count, last_nominal_time_s=(count-1)/FPS)), flush=True)


if __name__ == "__main__":
    main()
