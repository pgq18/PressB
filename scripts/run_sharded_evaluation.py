#!/usr/bin/env python3
"""Evaluate independent simulator workers through one shared inference service."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pressb.runtime.sharded_eval import main


if __name__ == "__main__":
    raise SystemExit(main())
