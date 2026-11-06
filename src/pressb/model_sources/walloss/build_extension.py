#!/usr/bin/env python3
"""Build Wall-X's CUDA extension without the original checkout or in-tree output.

The small Wall-X kernels are bundled. CUTLASS is an ordinary, pinned external
build dependency, fetched into the user's cache only when requested by a build.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import sysconfig
import tempfile


SOURCE_ROOT = Path(__file__).resolve().parent
CUTLASS_REPOSITORY = "https://github.com/NVIDIA/cutlass.git"
CUTLASS_COMMIT = "e51efbfe18fe4f4cbb66ab814c55bf4aa0185491"


def architecture(value: str) -> tuple[str, str]:
    value = value.removeprefix("sm_").removeprefix("compute_")
    if re.fullmatch(r"[0-9]{2,3}", value):
        value = value[:-1] + "." + value[-1]
    if not re.fullmatch(r"[1-9][0-9]?\.[0-9]", value):
        raise ValueError("Select one CUDA architecture, e.g. 9.0 (H200) or 12.0 (RTX 5090)")
    return value, value.replace(".", "")


def verify_cutlass(directory: Path) -> Path:
    directory = directory.expanduser().resolve()
    if not (directory / "include/cutlass/cutlass.h").is_file():
        raise ValueError(f"CUTLASS include tree is missing: {directory}")
    revision = subprocess.check_output(
        ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != CUTLASS_COMMIT:
        raise ValueError(f"CUTLASS revision mismatch: expected {CUTLASS_COMMIT}, got {revision}")
    changed = subprocess.check_output(
        ["git", "-C", str(directory), "status", "--porcelain", "--", "include"], text=True
    )
    if changed.strip():
        raise ValueError("CUTLASS headers contain working-tree changes")
    return directory


def fetch_cutlass(directory: Path) -> Path:
    if not directory.exists():
        directory.mkdir(parents=True)
        subprocess.run(["git", "init", "--quiet", str(directory)], check=True)
        subprocess.run(["git", "-C", str(directory), "remote", "add", "origin", CUTLASS_REPOSITORY], check=True)
    if not (directory / ".git").exists():
        raise ValueError(f"Refusing to replace a non-Git CUTLASS directory: {directory}")
    try:
        return verify_cutlass(directory)
    except (ValueError, subprocess.CalledProcessError):
        # Only initialize our empty cache. Never reset an existing modified tree.
        current = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"], capture_output=True)
        if current.returncode == 0:
            raise
    subprocess.run(["git", "-C", str(directory), "fetch", "--depth", "1", "origin", CUTLASS_COMMIT], check=True)
    subprocess.run(["git", "-C", str(directory), "checkout", "--detach", "FETCH_HEAD"], check=True)
    return verify_cutlass(directory)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", help="One native CUDA architecture (default: current GPU)")
    parser.add_argument("--cutlass-dir", type=Path, help="Existing clean CUTLASS checkout at the pinned commit")
    parser.add_argument("--cache-dir", type=Path, default=Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "pressb/walloss")
    parser.add_argument("--output-dir", type=Path, help="Directory for extension and build receipt, outside the source tree")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="Print the build plan without fetching or compiling")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import torch

    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    target = args.arch
    if target is None:
        target = ".".join(map(str, torch.cuda.get_device_capability()))
    target, capability = architecture(target)
    cache = args.cache_dir.expanduser().resolve()
    build_tag = f"{sysconfig.get_config_var('SOABI')}-torch{torch.__version__}-cuda{torch.version.cuda}-sm{capability}"
    output = (args.output_dir or cache / "extensions" / build_tag).expanduser().resolve()
    if output == SOURCE_ROOT or SOURCE_ROOT in output.parents:
        raise ValueError("Build output must be outside the bundled source tree")
    cutlass = (args.cutlass_dir or cache / "cutlass" / CUTLASS_COMMIT).expanduser().resolve()
    sources = sorted((SOURCE_ROOT / "csrc").glob("*.cu"))
    inputs = sorted([*sources, *(SOURCE_ROOT / "csrc").glob("*.h")])
    plan = {
        "source_root": str(SOURCE_ROOT), "output_dir": str(output),
        "cutlass_dir": str(cutlass), "cutlass_commit": CUTLASS_COMMIT,
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "python_version": sys.version, "architecture": target,
        "source_sha256": {str(p.relative_to(SOURCE_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        "extension_name": "wallx_csrc", "max_jobs": args.jobs,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    if args.cutlass_dir:
        cutlass = verify_cutlass(cutlass)
    else:
        cutlass = fetch_cutlass(cutlass)
    output.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_CUDA_ARCH_LIST"] = target
    os.environ["MAX_JOBS"] = str(args.jobs)
    from torch.utils.cpp_extension import load
    extension = load(
        name="wallx_csrc", sources=[str(p) for p in sources],
        extra_include_paths=[str(cutlass / "include"), str(SOURCE_ROOT / "csrc")],
        extra_cflags=["-fopenmp", "-fPIC", "-Wno-strict-aliasing"],
        extra_cuda_cflags=["-std=c++17", "-DENABLE_BF16", f"-DGROUPED_GEMM_DEVICE_CAPABILITY={capability}"],
        build_directory=str(output), with_cuda=True, verbose=True,
    )
    binary = Path(extension.__file__).resolve()
    plan.update(extension_path=str(binary), extension_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    receipt = output / "build_receipt.json"
    receipt.write_text(json.dumps(plan, indent=2) + "\n")
    current = {
        "extension_dir": str(output), "extension_path": str(binary),
        "extension_sha256": plan["extension_sha256"],
        "python_cache_tag": sys.implementation.cache_tag,
        "python_soabi": sysconfig.get_config_var("SOABI"),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "architecture": target, "build_receipt": str(receipt),
    }
    pointer = cache / "extensions/current.json"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=pointer.parent, delete=False) as handle:
        json.dump(current, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    try:
        os.replace(temporary, pointer)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(plan, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
