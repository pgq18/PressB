#!/usr/bin/env python3
"""Create a kinematic plan; this is not a physics success certificate."""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from pressb.kinematics import PiperKinematics
from pressb.planning import make_plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/scene.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/plan")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    kin = PiperKinematics(ROOT / cfg["robot_urdf"], tip_offset=cfg["tip_offset"])
    plan = make_plan(kin, cfg)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "planned_trajectory.npz", **vars(plan))
    summary = {"kind": "kinematic_plan_only", "physics_validated": False,
               "floors": cfg["sequence"], "samples": len(plan.time),
               "duration_s": float(plan.time[-1]), "joint_names": kin.joint_names}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
