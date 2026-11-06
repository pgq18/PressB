#!/usr/bin/env python3
"""Deploy PressB inference source, including its bundled model implementations.

Model weights, backbone caches, datasets and Python environments are provisioned
separately. Explicit external-source overrides remain a caller responsibility.
"""
import argparse
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FILES = ["src/pressb/", "scripts/serve_rl_inference.py", "scripts/serve_inference_node.py",
         "scripts/run_experiment.py", "configs/experiments/"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="h200")
    parser.add_argument("--remote-root", default="/home/pengguanqi/Worksapce/Research/PressB-online-rl-20261002")
    parser.add_argument("--ssh-socket", default="/tmp/pressb-5090-h200-eval.sock")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.host.startswith("-") or not args.remote_root.startswith("/"):
        parser.error("Use an SSH host and an absolute remote directory")
    ssh = ["ssh", "-o", "ConnectTimeout=120"]
    if args.ssh_socket:
        ssh += ["-S", args.ssh_socket]
    if not args.dry_run:
        subprocess.run([*ssh, args.host, "mkdir -p -- " + shlex.quote(args.remote_root)], check=True)
    command = ["rsync", "-a", "--relative", "--protect-args",
               "--exclude=__pycache__/", "--exclude=*.pyc", "--exclude=.git/",
               "-e", shlex.join(ssh)]
    if args.dry_run:
        command.append("--dry-run")
    subprocess.run([*command, *FILES, args.host + ":" + args.remote_root.rstrip("/") + "/"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
