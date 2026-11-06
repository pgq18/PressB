#!/usr/bin/env python3
"""Run a configured frozen inference adapter (separate model environment)."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pressb.runtime.server import run_service

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    run_service("inference", parser.parse_args().config)
