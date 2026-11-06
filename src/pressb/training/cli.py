"""A shared dispatcher for model-specific supervised training configurations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

from pressb.model_sources import activate_source, builtin_source_root


VLA_TRAINERS = {
    "pose9": "starVLA.training.train_piper",
    "joint6": "starVLA.training.train_table_buttons",
    "joint7": "starVLA.training.train_table_buttons",
    "joint6_multitask": "starVLA.training.train_table_buttons_multitask_joint6",
    "joint7_multitask": "starVLA.training.train_table_buttons_multitask",
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("walloss", "vlajepa"), required=True)
    parser.add_argument("--profile", choices=tuple(VLA_TRAINERS), help="VLA-JEPA training contract; defaults to pose9")
    parser.add_argument("--config", type=Path, required=True, help="Native training YAML configuration")
    parser.add_argument("--source-root", type=Path, help="Explicit legacy checkout override; default is bundled code")
    parser.add_argument("--dry-run", action="store_true", help="Inspect paths/arguments without importing training dependencies")
    parser.add_argument("native_args", nargs=argparse.REMAINDER, help="Arguments after -- are forwarded unchanged to the native trainer")
    args = parser.parse_args(argv)
    if args.model == "walloss" and args.profile is not None:
        parser.error("--profile applies to VLA-JEPA only")
    config = args.config.expanduser().resolve(strict=True)
    if not config.is_file():
        parser.error("--config must name a YAML file")
    root = args.source_root.expanduser().resolve(strict=True) if args.source_root else builtin_source_root(args.model)
    profile = args.profile or "pose9"
    module = VLA_TRAINERS[profile] if args.model == "vlajepa" else None
    entry = root / (module.replace(".", "/") + ".py" if module else "train_qact.py")
    if not entry.is_file():
        parser.error(f"Training entry point is missing: {entry}")
    native_args = args.native_args[1:] if args.native_args[:1] == ["--"] else args.native_args
    command_args = [str(entry), "--config", str(config), *native_args]
    if args.dry_run:
        print(json.dumps(dict(model=args.model, profile=profile if module else None,
            source_root=str(root), source="bundled" if args.source_root is None else "explicit",
            entry=str(entry), module=module, argv=command_args,
            config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
            launches_training=False), indent=2))
        return 0
    activate_source(args.model, root)
    previous = sys.argv
    try:
        sys.argv = command_args
        if module:
            runpy.run_module(module, run_name="__main__", alter_sys=True)
        else:
            runpy.run_path(str(entry), run_name="__main__")
    finally:
        sys.argv = previous
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
