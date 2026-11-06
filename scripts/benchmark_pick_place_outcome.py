#!/usr/bin/env python3
"""CPU-only benchmark and exact regression oracle for PAP outcome caching.

The reference class preserves the pre-optimization 2026-11-04 update, from
task.py SHA256 ba5ce32a7365b736e345f2ec341ce89ec9114355071c3f97dbbdb4f1a7962a5b.
Its public geometry/support helpers are intentionally unchanged by this patch.
These timings exclude PhysX, rendering, model inference and transport.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from scipy.spatial.transform import Rotation

from pressb.tasks.pap.task import PickPlaceOutcome, _half_extent, object_geometry, object_inside_box


class ReferencePickPlaceOutcome:
    """Original scalar algorithm, retained only as benchmark/regression evidence."""
    def __init__(self, edge, tabletop_z, box, stable_seconds=.5, lift_height=.04):
        self.geometry = object_geometry(edge)
        self.edge = self.geometry["grasp_width_m"]
        self.tabletop_z, self.box = tabletop_z, box
        self.stable_seconds, self.lift_height = stable_seconds, lift_height
        self.max_lift_m = 0.
        self.max_bottom_clearance_m = 0.
        self.stable_since = None
        self.success = False
        self.first_success_time_s = None

    def update(self, seconds, position, quaternion, velocity, gripper_width, finger_contact=False):
        self.max_lift_m = max(self.max_lift_m, float(position[2])-self.tabletop_z-self.geometry["size_xyz"][2]/2)
        inside = object_inside_box(position, quaternion, self.geometry, self.box)
        velocity = np.asarray(velocity)
        released = gripper_width >= self.edge + .008 and not finger_contact
        rotation = Rotation.from_quat(np.asarray(quaternion)[[1, 2, 3, 0]]).as_matrix()
        bottom = float(position[2]) - float(_half_extent(rotation, self.geometry, [0., 0., 1.]))
        self.max_bottom_clearance_m = max(self.max_bottom_clearance_m, bottom-self.tabletop_z)
        lifted = self.max_bottom_clearance_m >= self.lift_height
        resting_on_box_floor = abs(bottom-float(self.box["floor_z_world"])) <= .003
        stable = np.linalg.norm(velocity[:3]) < .025 and np.linalg.norm(velocity[3:]) < .25
        valid = inside and released and stable and resting_on_box_floor and lifted
        if not valid:
            self.stable_since = None
        elif self.stable_since is None:
            self.stable_since = float(seconds)
        if valid and seconds-self.stable_since >= self.stable_seconds and not self.success:
            self.success = True
            self.first_success_time_s = float(seconds)
        return dict(inside_box=inside, gripper_released=released, object_stable=bool(stable),
                    resting_on_box_floor=resting_on_box_floor, finger_contact=bool(finger_contact),
                    lifted=lifted, max_lift_m=self.max_lift_m,
                    max_bottom_clearance_m=self.max_bottom_clearance_m,
                    success=self.success, first_success_time_s=self.first_success_time_s)


def benchmark(envs=64, ticks=128, repeats=5, seed=42):
    rng = np.random.default_rng(seed)
    quaternions = rng.normal(size=(ticks, envs, 4))
    quaternions /= np.linalg.norm(quaternions, axis=-1, keepdims=True)
    positions = rng.normal(0, .03, size=(ticks, envs, 3))+[.4, -.2, .78]
    velocities = np.zeros((ticks, envs, 6))
    box = dict(center_world=[.4, -.2, .76], inner_size_xy=[.1, .15],
               floor_z_world=.765, rim_z_world=.82, yaw_deg=5.)
    shapes = [dict(shape="cube", edge=.03), dict(shape="cuboid", size_xyz=[.03, .04, .025]),
              dict(shape="cylinder", radius=.015, height=.03)]
    rows = []
    for geometry in shapes:
        def run(cls, keep=False):
            checkers = [cls(geometry, .76, box) for _ in range(envs)]
            outcomes = []
            for tick in range(ticks):
                for env in range(envs):
                    outcome = checkers[env].update(tick/120, positions[tick, env],
                        quaternions[tick, env], velocities[tick, env], .06)
                    if keep:
                        outcomes.append(outcome)
            return outcomes

        expected, actual = run(ReferencePickPlaceOutcome, True), run(PickPlaceOutcome, True)
        if expected != actual:
            raise AssertionError(f"Outcome changed for {geometry['shape']}")
        samples = {"reference": [], "optimized": []}
        for repeat in range(repeats):
            order = [("reference", ReferencePickPlaceOutcome), ("optimized", PickPlaceOutcome)]
            if repeat % 2:
                order.reverse()
            for name, cls in order:
                started = time.perf_counter()
                run(cls)
                samples[name].append(time.perf_counter()-started)
        medians = {name: float(np.median(values)) for name, values in samples.items()}
        rows.append(dict(geometry=geometry, updates_per_trial=envs*ticks,
            exact_matching_outputs=len(actual), seconds=samples, median_seconds=medians,
            speedup=medians["reference"]/medians["optimized"],
            saved_seconds_per_chunk=medians["reference"]-medians["optimized"]))
    source = Path(__file__).resolve().parents[1] / "src/pressb/tasks/pap/task.py"
    return dict(scope="CPU outcome calculation only; not end-to-end training throughput",
                envs=envs, ticks=ticks, repeats=repeats, seed=seed, measurements=rows,
                runtime=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                             machine=platform.machine()),
                optimized_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                reference_source_sha256="ba5ce32a7365b736e345f2ec341ce89ec9114355071c3f97dbbdb4f1a7962a5b")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--ticks", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.envs, args.ticks, args.repeats) <= 0:
        parser.error("envs, ticks and repeats must be positive")
    result = benchmark(args.envs, args.ticks, args.repeats)
    payload = json.dumps(result, indent=2, allow_nan=False)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
