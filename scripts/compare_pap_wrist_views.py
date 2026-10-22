#!/usr/bin/env python3
"""Compare real PAP wrist RGB with two physical replays, without image zooming."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont

from compose_pap_comparison import Reader, source_episode, common


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = source_episode(args.dataset, args.episode)["wrist"]
    simulations = []
    for directory in (args.before, args.after):
        report = json.loads((directory / "report.json").read_text())
        metadata = common.probe(directory / "wrist.mp4")
        if (report["mode"] != "replay"
                or report["trajectory_semantics"]["source"] != "observed_arm_action_gripper"
                or Path(report["trajectory"]).stem != f"episode_{args.episode:06d}"
                or metadata["fps"] != 30
                or metadata["frame_count"] != round(report["duration_s"] * 30) + 1
                or abs(round(report["duration_s"] * 30) / 30 - report["duration_s"]) > 1e-7):
            raise ValueError(f"Replay semantics/timing mismatch: {directory}")
        simulations.append(dict(path=str((directory / "wrist.mp4").resolve()),
                                frame_start=0, frame_count=metadata["frame_count"]))
    inputs = [source, *simulations]
    count = min(item["frame_count"] for item in inputs)
    if count < 1:
        raise ValueError("No shared frames")
    args.output.mkdir(parents=True, exist_ok=False)
    _, font_path = common.load_fonts("zh", None)
    title = ImageFont.truetype(font_path, 26)
    note = ImageFont.truetype(font_path, 18)
    background = Image.new("RGB", (1920, 620), "#101922")
    draw = ImageDraw.Draw(background)
    for column, label in enumerate(("真机腕部相机", "仿真 · 调整前", "仿真 · 调整后")):
        draw.text((column * 640 + 16, 12), label, font=title, fill="white")
    draw.text((16, 558), "相同名义时间 · 原速 1× · 原始 640×480 画面，无裁切或数字放大", font=note, fill="white")
    draw.text((16, 588), "数据集轨迹驱动的物理回放，非策略评测；相机与关节仍有约一帧级时延不确定性。", font=note, fill="#c0ccd8")
    partial = args.output / "wrist_before_after.partial.mp4"
    final = args.output / "wrist_before_after.mp4"
    encoder = subprocess.Popen([
        "ffmpeg", "-v", "error", "-nostdin", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s:v", "1920x620", "-r", "30", "-i", "pipe:0", "-an", "-c:v", "libx264",
        "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", "-threads", "2",
        "-movflags", "+faststart", str(partial)], stdin=subprocess.PIPE)
    try:
        with ExitStack() as stack:
            readers = [stack.enter_context(Reader(item["path"], item["frame_start"])) for item in inputs]
            for index in range(count):
                canvas = background.copy()
                for column, reader in enumerate(readers):
                    canvas.paste(reader.next(), (column * 640, 70))
                draw = ImageDraw.Draw(canvas)
                draw.text((16, 44), f"Episode {args.episode} | t = {index / 30:.2f} s", font=note, fill="#c0ccd8")
                if index in (0, 60, 90, 120, 150, 180, 210, count - 1):
                    canvas.save(args.output / f"compare_{index:03d}.jpg", quality=95)
                encoder.stdin.write(canvas.tobytes())
        encoder.stdin.close()
        if encoder.wait(timeout=60):
            raise RuntimeError("Video encoding failed")
    except BaseException:
        if encoder.poll() is None:
            encoder.terminate()
            encoder.wait(timeout=10)
        raise
    encoded = common.probe(partial)
    if (encoded["frame_count"] != count or encoded["fps"] != 30
            or (encoded["width"], encoded["height"]) != (1920, 620)):
        raise RuntimeError("Encoded timing or dimensions mismatch")
    partial.replace(final)
    manifest = dict(episode=args.episode, inputs=inputs, fps=30, frame_count=count,
                    last_nominal_time_s=(count - 1) / 30,
                    processing="Original source RGB: no crop, scaling, retiming, or interpolation; encoding only",
                    columns=["real", "before", "after"], output=str(final.resolve()), metadata=encoded)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
