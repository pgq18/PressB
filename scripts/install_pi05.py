"""Install the pinned LeRobot π0.5 package into the active PressB environment."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory
from urllib.request import urlopen


REVISION = "d324ffe810d17264a0b1e628698aa1fa09aa639c"
ARCHIVE_URL = f"https://codeload.github.com/huggingface/lerobot/tar.gz/{REVISION}"
ARCHIVE_SHA256 = "6e22ce65a5b4c65e03c491a7a86d0cf9d7fe14e08032e45b99063bdabb0179bd"
OPTIONAL_TOOLS = {
    "rerun-viewer": "rerun-sdk>=0.24.0,<0.27.0",
    "wandb-logging": "wandb>=0.24.0,<0.25.0",
}


def adapt_dependencies(path):
    # Policy, processors and PyAV data reading do not use these tools.
    # Their NumPy and Sentry dependencies conflict with Isaac Sim.
    text = path.read_text(encoding="utf-8")
    header = "[project.optional-dependencies]\n"
    extras = header
    for name, requirement in OPTIONAL_TOOLS.items():
        line = f'    "{requirement}",\n'
        if text.count(line) != 1:
            raise ValueError(f"Expected one base dependency on {requirement}")
        text = text.replace(line, "")
        extras += f'{name} = ["{requirement}"]\n'
    packaging = '"packaging>=24.2,<26.0"'
    if text.count(header) != 1 or text.count(packaging) != 1:
        raise ValueError("Unexpected dependency metadata in the pinned LeRobot source")
    # LeRobot uses the Version/parse API already present in Isaac's 23.0 pin.
    text = text.replace(header, extras).replace(packaging, '"packaging>=23.0,<26.0"')
    path.write_text(text, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use a local copy of the pinned upstream archive")
    parser.add_argument("--constraint", type=Path, action="append", default=[], help="Pass a constraint file to pip")
    parser.add_argument("--dry-run", action="store_true", help="Resolve dependencies without installing them")
    args = parser.parse_args(argv)
    project = Path(__file__).resolve().parents[1]
    with TemporaryDirectory(prefix="pressb-pi05-") as temp:
        root = Path(temp)
        archive = args.archive
        if archive is None:
            archive = root / "lerobot.tar.gz"
            with urlopen(ARCHIVE_URL, timeout=60) as response, archive.open("wb") as output:
                shutil.copyfileobj(response, output)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
            raise ValueError("LeRobot archive SHA256 differs from the pinned source")
        with tarfile.open(archive, "r:gz") as source_archive:
            source_archive.extractall(root, filter="data")
        source = root / f"lerobot-{REVISION}"
        adapt_dependencies(source / "pyproject.toml")
        command = [sys.executable, "-m", "pip", "install"]
        for constraint in args.constraint:
            command.extend(("--constraint", str(constraint)))
        if args.dry_run:
            command.append("--dry-run")
        command.extend(("-e", str(project), f"{source}[pi]", "transformers==5.3.0", "tokenizers==0.22.2",
                        "pyarrow==21.0.0", "PyYAML", "sentencepiece==0.2.1", "protobuf>=3.20,<8"))
        return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
