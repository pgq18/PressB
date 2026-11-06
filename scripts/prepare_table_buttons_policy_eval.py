#!/usr/bin/env python3
"""Prepare unseen paired layouts for real-policy tabletop-button evaluation.

The original collection configuration is frozen into the plan. Scripted motion
planning is used only to reject unreachable layouts; no expert action sequence
is saved or supplied to the policy runtime.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROFILES = ("near", "medium", "far")
COLORS = ("red", "yellow", "blue")
DUAL_CONTACT_POLICY = dict(version="table_buttons_dual_contact_v1",
    allowed_task_contact_pairs=[["/World/Piper/link7", "/World/Piper/link8"]],
    task_success="physical_target_press_without_wrong_or_blocking_contact",
    strict_collector_success="task_success_and_no_original_unexpected_contact")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def policy_seed(profile, color, episode_index):
    """Disjoint request RNG windows, shared by the paired single/multitask run."""
    if profile not in PROFILES or color not in COLORS or not 0 <= episode_index < 100:
        raise ValueError("Policy RNG allocation requires known task and episode index0..99")
    return 2_000_000 + PROFILES.index(profile) * 1_000_000 + COLORS.index(color) * 100_000 + episode_index * 1_000


def add_collision_policy(plan, policy):
    plan = deepcopy(plan)
    if policy == "finger_closure_dual":
        plan["defaults"]["collision_policy"] = deepcopy(DUAL_CONTACT_POLICY)
        plan["metrics"] = dict(primary="task_success", secondary="strict_collector_success",
            closure="finger_closure_episode_rate", success_alias="task_success",
            interpretation="Task success allows only the declared exact finger/finger closure pair. The original collector metric also rejects that contact; all contacts remain recorded.")
    elif policy != "strict_original":
        raise ValueError("Unknown collision policy")
    return plan


def reuse_plan(parent_path, output, collision_policy):
    """Publish a new protocol while preserving original layouts and evidence."""
    parent_path, output = Path(parent_path).resolve(), Path(output).resolve()
    destination = output / "plan.json"
    if destination == parent_path:
        raise ValueError("A derived evaluation plan must not overwrite its parent")
    parent = json.loads(parent_path.read_text())
    if parent.get("schema") != "pressb.table_buttons.policy_eval_plan.v1":
        raise ValueError("Unexpected parent evaluation plan schema")
    plan = add_collision_policy(parent, collision_policy)
    plan["parent_plan"] = dict(path=str(parent_path), sha256=sha256(parent_path),
        reuse="All scene configurations,300 paired layouts,episode IDs,and policy seeds are reused without regeneration",
        change="Only declared contact-outcome accounting and metric definitions change; physics and policy actions do not")
    if destination.exists() and json.loads(destination.read_text()) != plan:
        raise ValueError("Existing derived plan belongs to another protocol")
    atomic_json(destination, plan)
    atomic_json(output / "preparation_complete.json", dict(success=True, plan=str(destination),
        plan_sha256=sha256(destination), parent_plan=plan["parent_plan"],
        layouts_reused=sum(len(cell["episodes"]) for cell in parent["cells"] if cell["model_type"] == "single" and cell["target_color"] == "red"),
        cells=len(plan["cells"]), episodes=plan["planned_episodes"], expert_action_sequences_saved=False))
    return plan


def build_cells(episodes, endpoint_host="127.0.0.1", first_port=18110):
    cells = []
    for index, profile in enumerate(PROFILES):
        for model_type in ("single", "multitask"):
            for color in COLORS:
                model_id = f"{profile}_{color}" if model_type == "single" else f"{profile}_multitask"
                cell_id = model_id if model_type == "single" else f"{model_id}_{color}"
                entries = [dict(entry, policy_seed=policy_seed(profile, color, entry["episode_index"]))
                           for entry in episodes[profile][color]]
                cells.append(dict(cell_id=cell_id, model_id=model_id, model_type=model_type,
                    spacing_profile=profile, target_color=color, instruction=f"Press the {color} button.",
                    inference_endpoint=f"http://{endpoint_host}:{first_port + index}", episodes=entries))
    return cells


def prepare_profile(arguments):
    profile, config, output_string, count, seed_start = arguments
    from pressb.kinematics import PiperKinematics
    from pressb.tasks.table_buttons.collection import LayoutPlanner
    output = Path(output_string)
    state_path = output / "preparation" / f"{profile}.json"
    planner = LayoutPlanner(PiperKinematics(ROOT / config["robot_urdf"], tip_offset=.1358), config)
    planner.placement(profile)
    started = time.monotonic()
    results = {color: [] for color in COLORS}
    try:
        for episode_index in range(count):
            seed = seed_start + episode_index
            layout_path = output / "layouts" / profile / f"seed_{seed}.json"
            if layout_path.exists():
                layout = json.loads(layout_path.read_text())
            else:
                layout = planner.sample(profile, seed)
                atomic_json(layout_path, layout)
            for color in COLORS:
                directory = output / "scenes" / profile / color
                cfg_path = directory / f"episode_{episode_index:06d}.json"
                preflight_path = directory / f"episode_{episode_index:06d}_preflight.json"
                if not cfg_path.is_file() or not preflight_path.is_file():
                    effective, _times, _commands, report = planner.prepare(profile, seed, color, layout)
                    # These returned arrays are reachability diagnostics only.
                    # Discard immediately; no executable expert targets enter
                    # the evaluation plan or generated files.
                    del _times, _commands
                    atomic_json(cfg_path, effective)
                    atomic_json(preflight_path, dict(report, usage="reachability validation only; policy supplies every action"))
                results[color].append(dict(episode_index=episode_index, seed=seed,
                    layout_file=str(layout_path), layout_sha256=sha256(layout_path),
                    effective_config_file=str(cfg_path), effective_config_sha256=sha256(cfg_path),
                    preflight_file=str(preflight_path), preflight_sha256=sha256(preflight_path)))
            status = dict(state="preparing", profile=profile, completed=episode_index + 1, total=count,
                elapsed_seconds=time.monotonic() - started, seed=seed)
            atomic_json(state_path, status)
            if episode_index % 10 == 0 or episode_index + 1 == count:
                print(json.dumps(status), flush=True)
        atomic_json(state_path, dict(status, state="complete"))
        return profile, results
    except Exception as error:
        atomic_json(state_path, dict(state="failed", profile=profile, episode_index=episode_index,
            seed=seed, error=repr(error), elapsed_seconds=time.monotonic() - started))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-plan", type=Path, help="Reuse the exact existing scene/RNG plan without recomputing reachability")
    parser.add_argument("--collision-policy", choices=("strict_original", "finger_closure_dual"), default="strict_original")
    parser.add_argument("--source-collection", type=Path, default=ROOT / "outputs/table_buttons_dataset_20261023")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout-s", type=float, default=12.)
    parser.add_argument("--execute-steps", type=int, default=7)
    parser.add_argument("--endpoint-host", default="127.0.0.1")
    parser.add_argument("--first-port", type=int, default=18110)
    args = parser.parse_args()
    if args.reuse_plan:
        plan = reuse_plan(args.reuse_plan, args.output, args.collision_policy)
        print(json.dumps(dict(event="derived_plan_ready", plan=str((args.output / "plan.json").resolve()),
            episodes=plan["planned_episodes"], collision_policy=args.collision_policy)), flush=True)
        return
    if not 1 <= args.episodes <= 100 or args.seed_start < 10000 or not 1 <= args.workers <= 3:
        parser.error("Require1..100 episodes, unseen seed_start>=10000, and1..3 CPU preparation workers")
    if args.timeout_s <= 0 or args.execute_steps != 7:
        parser.error("Timeout must be positive; this evaluation executes the complete seven-action chunk")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_paths = [args.source_collection / "raw_shards" / profile / "plan.json" for profile in PROFILES]
    source_plans = [json.loads(path.read_text()) for path in source_paths]
    config = source_plans[0]["config"]
    if any(plan["config"] != config for plan in source_plans):
        raise ValueError("Original per-spacing collection configurations differ")
    if any(plan["seed_start"] != 0 or plan["episodes_per_group"] != 100 for plan in source_plans):
        raise ValueError("Unexpected original training layout seed allocation")
    identity = dict(schema="pressb.table_buttons.policy_eval_preparation.v1", episodes=args.episodes,
        seed_start=args.seed_start, original_config_sha256=config_digest(config),
        source_collection_plans={profile: dict(path=str(path.resolve()), sha256=sha256(path))
            for profile, path in zip(PROFILES, source_paths)}, timeout_s=args.timeout_s,
        execute_steps=args.execute_steps, endpoint_host=args.endpoint_host, first_port=args.first_port)
    identity_path = output / "preparation_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("This output belongs to a different evaluation plan")
    atomic_json(identity_path, identity)
    # Reachability imports run in separate CPU workers before any Isaac import.
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = dict(executor.map(prepare_profile, [(profile, config, str(output), args.episodes, args.seed_start)
                                                     for profile in PROFILES]))
    cells = build_cells(results, args.endpoint_host, args.first_port)
    plan = dict(schema="pressb.table_buttons.policy_eval_plan.v1", config=config,
        source=identity, cells=cells, planned_episodes=sum(len(cell["episodes"]) for cell in cells),
        models=12, cells_count=len(cells), episode_pairing="same profile/color/layout index uses identical scene and request RNG for single and multitask models",
        defaults=dict(timeout_s=args.timeout_s, fps=30, physics_dt=1/120, execute_steps=args.execute_steps,
            smoothing=dict(mode="none", alpha=.8),
            policy_request_seed_rule="episode.policy_seed + zero_based_request_index; policy_seed=2000000+profile_index*1000000+color_index*100000+episode_index*1000",
            policy_request_seed_window=1000, unexpected_contact_force_n=.02,
            non_target_finger_contact_force_n=.02, simulation_time_excludes_inference_latency=True),
        layout_acceptance="all three target paths passed original collector dense reachability checks; expert motions discarded",
        evaluation="policy joint7 actions only; measured physical press/contact outcome; all valid outcomes count, never success-only resampling")
    plan = add_collision_policy(plan, args.collision_policy)
    plan_path = output / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Existing published evaluation plan differs")
    atomic_json(plan_path, plan)
    atomic_json(output / "preparation_complete.json", dict(success=True, plan=str(plan_path),
        plan_sha256=sha256(plan_path), layouts=3 * args.episodes, cells=18,
        episodes=18 * args.episodes, expert_action_sequences_saved=False))
    print(json.dumps(dict(event="plan_ready", plan=str(plan_path), episodes=18 * args.episodes)), flush=True)


if __name__ == "__main__":
    main()
