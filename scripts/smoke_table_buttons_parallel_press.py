#!/usr/bin/env python3
"""Contact regression for randomized table buttons on the shared Isaac backend.

Generate collision-screened plans before Kit starts, then execute through the
normal reset/step API. A read-only observer checks every physics tick: fixtures
and spring anchors stay fixed while a finger physically displaces the cap.
This is scripted regression evidence, not learned-policy evaluation.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
COLORS = ("red", "yellow", "blue")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--scene-config", default="configs/piper_table_buttons.json")
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--num-envs", type=int, default=2)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--profile", choices=("near", "medium", "far"), default="medium")
    p.add_argument("--colors", nargs="+", choices=COLORS, default=["red", "blue"])
    p.add_argument("--seed", type=int, default=20261109)
    p.add_argument("--plan-only", action="store_true", help="CPU planning only; never starts Isaac")
    return p


def prepare_plans(config, args):
    """Load collision libraries and prepare all layouts before importing Kit."""
    from pressb.tasks.table_buttons.backend import load_config
    from pressb.tasks.table_buttons.collection import LayoutPlanner, serializable
    from pressb.robots.piper.kinematics import PiperKinematics
    cfg = load_config(config)
    kin = PiperKinematics(cfg["scene"]["robot_urdf"], tip_offset=.1358)
    planner = LayoutPlanner(kin, cfg["scene"])
    plans = []
    for round_id in range(args.rounds):
        batch = []
        for eid in range(args.num_envs):
            seed = args.seed + round_id * 1009 + eid * 37
            color = args.colors[(round_id * args.num_envs + eid) % len(args.colors)]
            layout = planner.sample(args.profile, seed)
            _, times, commands, report = planner.prepare(args.profile, seed, color, layout)
            batch.append(serializable(dict(env_id=eid, seed=seed, profile=args.profile,
                target_color=color, layout=layout, times=times, commands=commands, plan_report=report)))
        plans.append(batch)
    return plans


def fixture_errors(before, after):
    """Return geometric errors in metres using public diagnostic snapshots."""
    import numpy as np
    initial = np.asarray(before["button_body_positions"]).reshape(3, 3, 3)
    current = np.asarray(after["button_body_positions"]).reshape(3, 3, 3)
    anchors = np.asarray(after["button_spring_anchors"])
    rest = np.asarray(after["button_rest_world"])
    values = [initial, current, anchors, rest, np.asarray(before["button_spring_anchors"])]
    if not all(np.isfinite(value).all() for value in values):
        raise RuntimeError("Nonfinite fixture geometry")
    return dict(fixture_drift_m=float(np.max(np.abs(current[:, 1:] - initial[:, 1:]))),
        anchor_drift_m=float(np.max(np.abs(anchors - before["button_spring_anchors"]))),
        anchor_rest_error_m=float(np.max(np.abs(anchors - rest))),
        cap_lateral_error_m=float(np.max(np.abs(current[:, 0, :2] - rest[:, :2]))),
        fixture_lateral_error_m=float(np.max(np.abs(current[:, 1:, :2] - rest[:, None, :2]))))


class ContactAudit:
    def __init__(self, backend, plans, stream):
        self.backend, self.plans, self.stream = backend, plans, stream
        self.current_commands, self.held = {}, {}
        self.baseline = backend.snapshot()
        self.samples = 0
        self.max_errors = {}
        self.max_depth = {i: 0. for i in range(backend.num_envs)}
        self.max_force = {i: 0. for i in range(backend.num_envs)}

    def after_tick(self, backend, ids, outcomes):
        if self.stream is None:
            return
        from pressb.tasks.table_buttons.execution import in_press_window, press_phase_times
        snapshots = backend.snapshot(ids)
        for eid in ids:
            errors = fixture_errors(self.baseline[str(eid)], snapshots[str(eid)])
            for key, value in errors.items():
                self.max_errors[key] = max(self.max_errors.get(key, 0.), value)
            if max(errors.values()) > 2e-5:
                raise RuntimeError(f"Environment {eid} fixture drift: {errors}")
            physical = outcomes[eid]["info"]["physical_outcome"]
            plan = self.plans[eid]
            color = plan["target_color"]
            depth, force = physical["travel_m"][color], physical["finger_contact_force_n"][color]
            self.max_depth[eid] = max(self.max_depth[eid], depth)
            self.max_force[eid] = max(self.max_force[eid], force)
            tick = int(backend.ticks[eid])
            seconds = ((tick-1)//backend.execution["physics_ticks_per_action"])/backend.execution["control_hz"]
            if (eid not in self.held and in_press_window(seconds, press_phase_times(plan["plan_report"]))
                    and depth >= backend.config["scene"]["success"]["press_depth_m"]+.0002):
                self.held[eid] = self.current_commands[eid].copy()
            self.stream.write(json.dumps(dict(env_id=eid, physics_tick=tick, target_color=color,
                travel_m=physical["travel_m"], finger_contact_force_n=physical["finger_contact_force_n"],
                success=physical["success"], fixture_errors=errors), allow_nan=False)+"\n")
            self.samples += 1


def save_images(rows, directory, label):
    from pressb.runtime.contract import decode_image
    for row in rows:
        for camera, encoded in row["observation"]["images"].items():
            decode_image(encoded).save(directory/f"{label}_env{row['env_id']}_{camera}.png")


def run(args, *, plan_factory=prepare_plans, backend_factory=None):
    import numpy as np
    from pressb.tasks.table_buttons.execution import contact_limited_command, press_phase_times
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = dict(scene_config=args.scene_config, num_envs=args.num_envs, gpu=args.gpu,
        action_dim=7, horizon=1, max_episode_seconds=10., collision_policy="collector_strict",
        randomization=dict(profiles=[args.profile], colors=list(COLORS), positions=False),
        smoothing=dict(mode="none"), log_dir=str(output/"simulation"),
        render_resolution=[320,240], image_output_resolution=[320,240],
        observation_render_frames=1, reset_render_frames=2)
    report = dict(kind="table_buttons_shared_parallel_contact_regression", status="running",
        learned_policy_evaluation=False, physical_robot=False, num_envs=args.num_envs,
        rounds=[], started_unix=time.time(), closed=False)
    backend = None
    try:
        plans = plan_factory(config, args)
        config["max_episode_seconds"] = float(math.ceil(max(plan["times"][-1] for batch in plans for plan in batch)))
        (output/"plans.json").write_text(json.dumps(plans, indent=2, allow_nan=False)+"\n")
        (output/"config.json").write_text(json.dumps(config, indent=2)+"\n")
        if args.plan_only:
            report.update(status="planned", physical_execution=False, planned_episodes=sum(map(len,plans)))
            return 0
        if backend_factory is None:
            from pressb.tasks.table_buttons.parallel import create_backend
            backend_factory = create_backend
        backend = backend_factory(config)
        from pressb.runtime.parallel import SharedWorldBackend
        if not isinstance(backend, SharedWorldBackend):
            raise RuntimeError("Regression requires the common SharedWorldBackend")
        report["metadata"] = backend.describe()
        ids = list(range(args.num_envs))
        audit = None
        for round_id, batch in enumerate(plans):
            folder = output/f"round_{round_id:02d}"
            folder.mkdir()
            options = {str(i):dict(profile=plan["profile"],target_color=plan["target_color"],
                positions=plan["layout"]["positions"]) for i,plan in enumerate(batch)}
            initial = []
            # The first reset includes all slots. Later resets independently move
            # each fixture while the core proves peers and the clock unchanged.
            for subset in ([ids] if round_id == 0 else [[i] for i in ids]):
                initial.extend(backend.reset(subset,args.seed,dict(by_env=options)))
            save_images(initial,folder,"initial")
            rows, terminal = initial, []
            with (folder/"physics.jsonl").open("w") as stream:
                if audit is None:
                    audit = ContactAudit(backend,batch,stream)
                    backend.add_observer(audit)
                else:
                    audit.__init__(backend,batch,stream)
                try:
                    for _ in range(round(config["max_episode_seconds"]*backend.execution["control_hz"])):
                        commands = []
                        for row in rows:
                            eid = row["env_id"]
                            plan = batch[eid]
                            times, path = np.asarray(plan["times"]), np.asarray(plan["commands"])
                            seconds = float(backend.ticks[eid])*backend.execution["physics_dt"]
                            nominal = np.array([np.interp(seconds,times,path[:,j]) for j in range(7)])
                            command = contact_limited_command(seconds,nominal,audit.held.get(eid),times,path,
                                                              press_phase_times(plan["plan_report"]))
                            audit.current_commands[eid] = command
                            commands.append(dict(env_id=eid,actions=[command.tolist()]))
                        advanced = backend.step(commands)
                        terminal.extend(row for row in advanced if row["terminated"] or row["truncated"])
                        rows = [row for row in advanced if not (row["terminated"] or row["truncated"])]
                        if not rows:
                            break
                finally:
                    # The same read-only observer is rebound to the next round.
                    audit.stream = None
                save_images(terminal,folder,"terminal")
                summary = dict(round_id=round_id,physics_samples=audit.samples,
                    maximum_fixture_errors=audit.max_errors,maximum_target_depth_m=audit.max_depth,
                    maximum_target_force_n=audit.max_force,episodes=[{**row,"observation":{
                        **row["observation"],"images":list(row["observation"]["images"])}} for row in terminal])
                report["rounds"].append(summary)
                if rows or len(terminal)!=args.num_envs or not all(row["info"]["success"] for row in terminal):
                    raise RuntimeError(f"Physical press regression failed in round {round_id}")
        report.update(status="passed",physical_execution=True,timings=deepcopy(backend.timings))
    except Exception as exc:
        import traceback
        report.update(status="failed",error=f"{type(exc).__name__}: {exc}",traceback=traceback.format_exc())
    finally:
        report["finished_unix"] = time.time()
        # Isaac fast_shutdown can exit the process before close returns.
        (output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
        if backend is not None:
            backend.close()
            report["closed"] = True
            (output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    return 0 if report["status"] in ("passed","planned") else 1


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.num_envs < 2 or args.rounds < 1 or args.seed < 0 or args.gpu < 0:
        p.error("Use at least two environments, a positive round count, nonnegative seed and GPU")
    if args.output.expanduser().exists():
        p.error("--output must be new")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
