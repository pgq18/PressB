"""Small shared recording helpers for local simulation entry points."""
from pathlib import Path
import json
import subprocess


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temporary.replace(path)


class Video:
    def __init__(self, path, size, fps):
        self.path, self.count = Path(path), 0
        self.process = subprocess.Popen([
            "ffmpeg", "-v", "error", "-nostdin", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s:v", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "pipe:0", "-an",
            "-c:v", "libx264", "-preset", "fast", "-crf", "19", "-pix_fmt", "yuv420p",
            "-threads", "1", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)

    def add(self, rgb):
        self.process.stdin.write(rgb.tobytes())
        self.count += 1

    def close(self):
        if not self.process.stdin.closed:
            self.process.stdin.close()
        if self.process.wait(timeout=60):
            raise RuntimeError(f"Encoding failed: {self.path}")

