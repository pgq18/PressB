#!/usr/bin/env python3
"""Run the existing fast simulation server with read-only trajectory recording.

All CLI flags and the simulation implementation are those of
serve_rl_fast_simulation.py. The only extension reads measured PhysX state;
it neither advances physics nor requests additional policy camera frames.
Trajectories are written beneath OUTPUT/trajectories. The extension source
identity is recorded separately, leaving strict policy/controller checks intact.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from pressb.online_rl import fast_simulation
    from pressb.online_rl.trajectory_recording import RecordingFastIsaacVectorBackend
    from serve_rl_fast_simulation import main as serve

    # The original launcher imports this factory only after SimulationApp
    # initialization. No Isaac API is imported by the recording module itself.
    original = fast_simulation.FastIsaacVectorBackend
    fast_simulation.FastIsaacVectorBackend = RecordingFastIsaacVectorBackend
    try:
        return serve()
    finally:
        fast_simulation.FastIsaacVectorBackend = original


if __name__ == "__main__":
    raise SystemExit(main())
