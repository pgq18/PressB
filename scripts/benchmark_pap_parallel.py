#!/usr/bin/env python3
"""Bounded, isolated real-PAP capacity probe (simulation only; no learned policy).

Example (use the Isaac Python environment):
  python scripts/benchmark_pap_parallel.py --num-envs 64 \
    --factory pressb.rl_nodes.pap_vector:create_backend --gpu-id 0 \
    --output-dir outputs/pap_capacity_64

The parent never imports Isaac. A fresh child process owns the entire simulator,
and a resource watchdog stops only that process session if memory becomes scarce.
Reported transitions/s count one environment/action-chunk result, not episodes or
individual control ticks. Startup, reset, rendering and steady step timings are
reported separately; this is not end-to-end DSRL training throughput.
"""
from __future__ import annotations

import argparse
import base64
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
GIB = 1024 ** 3


def write_json(path, value):
    """Atomic small status/report writes, including on watchdog interruption."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def json_payload_evidence(request, response):
    """Measure HTTP-compatible JSON envelopes against the shared 64 MiB budget."""
    from pressb.online_rl.rpc import MAX_BODY_BYTES
    begin = time.monotonic()
    # Match RPCClient's request formatting and RPCServer's compact responses.
    request_bytes = len(json.dumps({**request, "protocol_version": 1}, allow_nan=False).encode())
    response_bytes = len(json.dumps({**response, "protocol_version": 1},
                                   allow_nan=False, separators=(",", ":")).encode())
    return dict(request_json_bytes=request_bytes, response_json_bytes=response_bytes,
                within_64mib_budget=max(request_bytes, response_bytes) <= MAX_BODY_BYTES,
                measurement_seconds=time.monotonic()-begin)


def gpu_snapshot():
    """Return global GPU use and all compute PID allocations; unavailable is explicit."""
    result = {"devices": [], "processes": [], "errors": []}
    queries = (("devices", ["--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu"]),
               ("processes", ["--query-compute-apps=gpu_uuid,pid,used_memory"]))
    for kind, query in queries:
        try:
            completed = subprocess.run(["nvidia-smi", *query, "--format=csv,noheader,nounits"],
                                       capture_output=True, text=True, timeout=5, check=True)
            for line in completed.stdout.splitlines():
                parts = [part.strip() for part in line.split(",")]
                if kind == "devices":
                    index, uuid, name, total, used, free, utilization = parts
                    result[kind].append(dict(index=int(index), uuid=uuid, name=name,
                        total_mib=float(total), used_mib=float(used), free_mib=float(free),
                        utilization_percent=float(utilization)))
                else:
                    uuid, pid, memory = parts
                    result[kind].append(dict(uuid=uuid, pid=int(pid),
                        used_mib=None if memory in ("[N/A]", "N/A", "[Not Supported]") else float(memory)))
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            result["errors"].append(f"{kind}: {type(error).__name__}: {error}")
    return result


def process_sample(parent_pid, tracked):
    """Track PID plus creation time; never infer ownership from executable names."""
    import psutil
    try:
        owner = psutil.Process(parent_pid)
        candidates = [owner, *owner.children(recursive=True)]
    except psutil.NoSuchProcess:
        candidates = []
    for process in candidates:
        try:
            tracked[process.pid] = process.create_time()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    rows = []
    for pid, created in tracked.items():
        try:
            process = psutil.Process(pid)
            if process.create_time() != created:
                continue
            rows.append({"pid": pid, "rss_bytes": process.memory_info().rss,
                         "status": process.status()})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return rows


def owned_signal(pid, tracked, sig):
    """Signal our isolated session and verified descendants, including orphaned workers."""
    import psutil
    signaled_group = False
    try:
        process = psutil.Process(pid)
        if tracked.get(pid) == process.create_time() and os.getpgid(pid) == pid:
            os.killpg(pid, sig)
            signaled_group = True
    except (ProcessLookupError, psutil.NoSuchProcess):
        pass
    for child_pid, created in tracked.items():
        try:
            process = psutil.Process(child_pid)
            if process.create_time() == created and process.status() != psutil.STATUS_ZOMBIE:
                if signaled_group and os.getpgid(child_pid) == pid:
                    continue
                process.send_signal(sig)
        except (ProcessLookupError, psutil.NoSuchProcess):
            pass


def stop_owned(process, tracked):
    """Graceful simulator close first, bounded escalation only for this launch."""
    for sig, seconds in ((signal.SIGINT, 30.), (signal.SIGTERM, 10.), (signal.SIGKILL, 5.)):
        rows = process_sample(process.pid, tracked)
        if not any(row["status"] != "zombie" for row in rows):
            break
        owned_signal(process.pid, tracked, sig)
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            process.poll()
            rows = process_sample(process.pid, tracked)
            if not any(row["status"] != "zombie" for row in rows):
                break
            time.sleep(.25)
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass
    return process_sample(process.pid, tracked)


def resource_reason(sample, gpu_ids, min_host_gib, min_gpu_gib):
    if sample["host_available_bytes"] < min_host_gib * GIB:
        return "host_available_below_guard"
    devices = {device["index"]: device for device in sample["gpu"]["devices"]}
    if any(index not in devices for index in gpu_ids):
        return "gpu_monitor_unavailable"
    if any(devices[index]["free_mib"] < min_gpu_gib * 1024 for index in gpu_ids):
        return "gpu_free_below_guard"
    return None


def resolved_config(args):
    config = json.loads(Path(args.config).read_text())
    config = copy.deepcopy(config)
    spec = config["backend"]
    if args.factory:
        spec["factory"] = args.factory
    backend = spec.setdefault("config", {})
    backend["num_envs"] = args.num_envs
    backend["gpu_ids"] = [args.gpu_id]
    backend["gpu"] = args.gpu_id
    backend.pop("workers", None)
    backend["horizon"] = args.action_horizon
    backend["max_episode_seconds"] = (args.max_episode_seconds if args.max_episode_seconds is not None
                                      else 20. if args.replay_episode else 2.)
    backend["log_dir"] = str(args.output_dir / "workers")
    if args.render_resolution:
        backend["render_resolution"] = args.render_resolution
    if args.reset_render_frames is not None:
        backend["reset_render_frames"] = args.reset_render_frames
    if args.render_subframes is not None:
        backend["render_subframes"] = args.render_subframes
    if args.replay_episode:
        # q_target records already-filtered commands; filtering twice would not
        # replay the same controller input. Use the source scene and object pose.
        backend["scene_config"] = str(args.replay_episode.resolve() / "config.json")
        backend["smoothing"] = {"mode": "none", "alpha": .8}
        backend["randomization"] = {"position": "fixed", "colors": ["red"]}
    return config


def run_parent(args):
    import psutil
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config_path = args.output_dir / "simulation_config.json"
    write_json(config_path, resolved_config(args))
    start = time.monotonic()
    tracked = {}

    def sample(process=None):
        processes = [] if process is None else process_sample(process.pid, tracked)
        gpu = gpu_snapshot()
        owned = {row["pid"] for row in processes}
        return dict(elapsed_seconds=time.monotonic()-start,
            host_available_bytes=psutil.virtual_memory().available,
            process_tree_rss_bytes=sum(row["rss_bytes"] for row in processes),
            processes=processes, gpu=gpu,
            owned_gpu_mib=sum(row["used_mib"] or 0 for row in gpu["processes"] if row["pid"] in owned))

    baseline = sample()
    report = dict(schema_version=1, scope="simulation_only_no_inference_or_learner",
        started_utc=datetime.now(timezone.utc).isoformat(), config=str(config_path),
        baseline=baseline, guard={"min_host_available_gib": args.min_host_gib,
        "min_gpu_free_gib": args.min_gpu_gib, "timeout_seconds": args.timeout},
        samples_file=str(args.output_dir / "resources.jsonl"))
    reason = resource_reason(baseline, [args.gpu_id], args.min_host_gib, args.min_gpu_gib)
    if reason:
        report.update(state="not_started", reason=reason)
        write_json(args.output_dir / "report.json", report)
        print(json.dumps(report, indent=2), flush=True)
        return 2
    command = [sys.executable, str(Path(__file__).resolve()), "--child", "--config", str(config_path),
               "--output-dir", str(args.output_dir), "--chunks", str(args.chunks),
               "--action-horizon", str(args.action_horizon), "--seed", str(args.seed)]
    if args.replay_episode:
        command += ["--replay-episode", str(args.replay_episode.resolve())]
    report["launch"] = {"parent_argv": sys.argv, "child_argv": command,
        "python": sys.executable, "working_directory": str(ROOT),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "benchmark_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    process = None
    samples = [baseline]
    try:
        with (args.output_dir / "child.log").open("wb") as log, \
             (args.output_dir / "resources.jsonl").open("w") as resource_log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            tracked[process.pid] = psutil.Process(process.pid).create_time()
            print(json.dumps({"event": "launched", "pid": process.pid,
                "output": str(args.output_dir), "num_envs": args.num_envs}), flush=True)
            while True:
                value = sample(process)
                samples.append(value)
                resource_log.write(json.dumps(value, allow_nan=False) + "\n")
                resource_log.flush()
                code = process.poll()
                if code is not None:
                    break
                reason = resource_reason(value, [args.gpu_id], args.min_host_gib, args.min_gpu_gib)
                if time.monotonic()-start >= args.timeout:
                    reason = "wall_time_limit"
                if reason:
                    report.update(state="stopped_by_watchdog", reason=reason)
                    print(json.dumps({"event": "watchdog_stop", "reason": reason}), flush=True)
                    break
                time.sleep(args.sample_seconds)
    except KeyboardInterrupt:
        report.update(state="interrupted", reason="parent_interrupt")
    finally:
        survivors = stop_owned(process, tracked) if process else []
        report.update(exit_code=process.returncode if process else None,
            cleanup_surviving_processes=survivors, owned_pids=sorted(tracked),
            elapsed_seconds=time.monotonic()-start,
            peak_process_tree_rss_gib=max(value["process_tree_rss_bytes"] for value in samples)/GIB,
            peak_owned_gpu_gib=max(value["owned_gpu_mib"] for value in samples)/1024,
            minimum_host_available_gib=min(value["host_available_bytes"] for value in samples)/GIB,
            final_resources=sample())
        report["gpu_peaks"] = {str(index): {
            "peak_global_used_gib": max(device["used_mib"] for value in samples
                for device in value["gpu"]["devices"] if device["index"] == index)/1024,
            "minimum_global_free_gib": min(device["free_mib"] for value in samples
                for device in value["gpu"]["devices"] if device["index"] == index)/1024}
            for index in [args.gpu_id]}
        child_path = args.output_dir / "simulation_results.json"
        if child_path.exists():
            report["simulation"] = json.loads(child_path.read_text())
        report.setdefault("state", "completed" if process and process.returncode == 0 else "failed")
        if any(row["status"] != "zombie" for row in survivors):
            report["state"] = "cleanup_failed"
        write_json(args.output_dir / "report.json", report)
        print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return 0 if report["state"] == "completed" else 1


def image_evidence(rows, output_dir, prefix="reset"):
    """Decode every observation camera; save labeled actual RGB contact sheets."""
    import numpy as np
    from PIL import Image, ImageDraw
    cameras = sorted(rows[0]["observation"]["images"])
    result = {}
    for camera in cameras:
        columns = min(8, len(rows))
        tile_width, tile_height = 160, 140
        canvas = Image.new("RGB", (columns*tile_width, math.ceil(len(rows)/columns)*tile_height), "white")
        draw = ImageDraw.Draw(canvas)
        stats = []
        for index, row in enumerate(rows):
            wire = row["observation"]["images"][camera]
            if wire["encoding"] != "png_base64":
                raise ValueError("PAP benchmark expects PNG RGB observations")
            image = Image.open(io.BytesIO(base64.b64decode(wire["data"], validate=True))).convert("RGB")
            array = np.asarray(image)
            std = float(array.std())
            if std < 1. or min(image.size) < 16:
                raise RuntimeError(f"Empty/constant image for env {row['env_id']} camera {camera}")
            stats.append(dict(env_id=row["env_id"], width=image.width, height=image.height,
                              std=std, minimum=int(array.min()), maximum=int(array.max())))
            x, y = (index % columns)*tile_width, (index // columns)*tile_height
            image.thumbnail((160, 120))
            canvas.paste(image, (x, y+20))
            draw.text((x+4, y+3), f"env {row['env_id']} | {camera}", fill="black")
        path = output_dir / f"{prefix}_{camera}_all_envs.jpg"
        canvas.save(path, quality=90)
        result[camera] = {"contact_sheet": str(path), "images": stats}
    return result


def optional_snapshot(backend, ids):
    """A backend may expose physics-free numeric diagnostics for reset isolation."""
    callback = getattr(backend, "diagnostic_snapshot", None)
    return callback(ids) if callable(callback) else None


def load_replay(episode, control_hz):
    """Convert recorded eight-DOF applied targets to 6 joints + gripper width."""
    import numpy as np
    episode = Path(episode)
    scene = json.loads((episode / "config.json").read_text())
    if not math.isclose(float(scene["fps"]), control_hz):
        raise ValueError("Replay and simulation control rates differ")
    stride = round(1/float(scene["physics_dt"])/control_hz)
    if stride < 1 or not math.isclose(stride*scene["physics_dt"]*control_hz, 1.):
        raise ValueError("Replay physics/control rates require an integral stride")
    with np.load(episode / "physics.npz", allow_pickle=False) as trajectory:
        raw = np.asarray(trajectory["q_target"], dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 8 or len(raw) < stride+1 or not np.isfinite(raw).all():
        raise ValueError("Expected finite applied q_target [physics_ticks+initial,8]")
    selected = raw[1::stride]
    for index, command in enumerate(selected):
        block = raw[1+index*stride:1+(index+1)*stride]
        if not np.allclose(block, command, atol=1e-8, rtol=0):
            raise ValueError("Recorded commands change within a control interval")
    commands = np.c_[selected[:, :6], selected[:, 6]-selected[:, 7]]
    obj = scene.get("object") or scene["cube"]
    options = {"center_xy": obj["center_xy"], "yaw_deg": obj.get("yaw_deg", 0.), "color": "red"}
    metadata = {"source_episode": str(episode.resolve()), "control_commands": len(commands),
        "source_physics_samples": len(raw), "physics_stride": stride,
        "command_sha256": hashlib.sha256(commands.tobytes()).hexdigest(),
        "already_smoothed": True, "continuation": "hold final command until terminal/timeout"}
    return commands, options, metadata


def replay_probe(service, request, initial, commands, args, config, results):
    """Qualify actual grasp/transport physics using identical prerecorded commands."""
    import numpy as np
    rows_by_id = {row["env_id"]: row for row in initial}
    active = sorted(rows_by_id)
    horizon = min(args.action_horizon, service.metadata["action"]["horizon"])
    control_hz = service.metadata["action"]["control_hz"]
    max_seconds = config["backend"]["config"]["max_episode_seconds"]
    maximum_chunks = math.ceil(max_seconds*control_hz/horizon)+2
    timings, terminal = [], {}
    controls = physics = transitions = cursor = 0
    start = time.monotonic()
    for chunk in range(maximum_chunks):
        if not active:
            break
        indices = np.minimum(np.arange(cursor, cursor+horizon), len(commands)-1)
        actions = commands[indices].tolist()
        items = [dict(env_id=i, episode_id=rows_by_id[i]["episode_id"],
                      step_id=rows_by_id[i]["step_id"], actions=actions) for i in active]
        begin = time.monotonic()
        rows = request("step", items=items)["items"]
        seconds = time.monotonic()-begin-results["json_payloads"]["calls"][-1]["measurement_seconds"]
        cursor += horizon
        transitions += len(rows)
        controls += sum(row["info"]["elapsed_control_steps"] for row in rows)
        physics += sum(row["info"]["physics_steps"] for row in rows)
        timings.append(dict(chunk=chunk, seconds=seconds, transitions=len(rows)))
        for row in rows:
            i = row["env_id"]
            rows_by_id[i] = row
            if row["terminated"] or row["truncated"]:
                info = row["info"]
                terminal[i] = dict(env_id=i, terminated=row["terminated"], truncated=row["truncated"],
                    success=info["success"], reward=row["reward"],
                    termination_reason=info.get("termination_reason"),
                    episode_physics_steps=info.get("episode_physics_steps"),
                    episode_seconds=info.get("episode_seconds"),
                    contact_counts=info.get("contact_counts"),
                    physical_outcome=info.get("physical_outcome"),
                    object_position_world=info.get("object_position_world"))
        active = [i for i in active if i not in terminal]
        print(json.dumps({"event": "replay_chunk", "chunk": chunk, "step_seconds": seconds,
            "active_envs": len(active), "successful_replays": sum(row["success"] for row in terminal.values())}),
            flush=True)
    if active:
        raise RuntimeError("Replay exhausted its bounded chunk budget without environment termination")
    elapsed = time.monotonic()-start-sum(row["measurement_seconds"] for row in
        results["json_payloads"]["calls"] if row["method"] == "step")
    step_seconds = sum(row["seconds"] for row in timings)
    results.update(state="completed", mode="recorded_action_physics_qualification",
        chunks=timings, transitions=transitions, physics_ticks=physics, control_steps=controls,
        step_seconds=step_seconds, rollout_including_resets_seconds=elapsed,
        step_transitions_per_second=transitions/step_seconds,
        rollout_transitions_per_second=transitions/elapsed,
        control_steps_per_second=controls/step_seconds, physics_ticks_per_second=physics/step_seconds,
        replay_outcomes=[terminal[i] for i in sorted(terminal)],
        replay_success_count=sum(row["success"] for row in terminal.values()),
        policy_success_rate_not_evaluated=True,
        final_images=image_evidence([rows_by_id[i] for i in sorted(rows_by_id)], args.output_dir, prefix="final"))


def run_child(args):
    import numpy as np
    from pressb.rl_nodes.server import load_backend
    from pressb.rl_nodes.simulation import SimulationService
    config = json.loads(Path(args.config).read_text())
    output = args.output_dir / "simulation_results.json"
    results = {"state": "starting", "scope": "simulation_only_no_inference_or_learner",
               "transitions_definition": "one environment action-chunk response",
               "json_payloads": {"budget_bytes": 64*1024*1024, "calls": [],
                   "all_within_64mib_budget": True,
                   "measurement_cost_excluded_from_step_and_rollout_times": True}}
    write_json(output, results)
    service = backend = None
    started = time.monotonic()
    try:
        backend = load_backend(config["backend"])
        service = SimulationService(backend, request_cache_size=2)
        results.update(startup_seconds=time.monotonic()-started, metadata=service.metadata)
        ids = list(range(service.metadata["num_envs"]))
        replay = load_replay(args.replay_episode, service.metadata["action"]["control_hz"]) if args.replay_episode else None
        request_number = 0
        run_id = "pap_capacity_probe"

        def request(method, **payload):
            nonlocal request_number
            request_number += 1
            body = {"run_id": run_id, "request_id": f"request_{request_number}", **payload}
            response = getattr(service, method)(body)
            evidence = json_payload_evidence(body, response)
            results["json_payloads"]["calls"].append({"method": method, **evidence})
            results["json_payloads"]["all_within_64mib_budget"] &= evidence["within_64mib_budget"]
            return response

        begin = time.monotonic()
        initial = request("reset", env_ids=ids, seed=args.seed, options={} if replay is None else replay[1])["items"]
        results["initial_reset_seconds"] = time.monotonic()-begin-results["json_payloads"]["calls"][-1]["measurement_seconds"]
        results["images"] = image_evidence(initial, args.output_dir)
        positions = [row["info"].get("reset", {}).get("center_xy") for row in initial]
        results["initial_object_centers_xy"] = positions
        results["unique_initial_object_centers"] = len({tuple(x) for x in positions if x is not None})
        if replay is not None:
            results["replay"] = replay[2]
            replay_probe(service, request, initial, replay[0], args, config, results)
            service.close_run({"run_id": run_id})
            write_json(output, results)
            return 0
        states = {row["env_id"]: row for row in initial}
        initial_states = {i: np.asarray(row["observation"]["state"]) for i, row in states.items()}
        action = service.metadata["action"]
        lower, upper = np.asarray(action["lower"]), np.asarray(action["upper"])
        horizon = min(args.action_horizon, action["horizon"])
        target = {i: np.clip(q+np.r_[.004*(2*(i+.5)/len(ids)-1), np.zeros(len(q)-1)], lower, upper)
                  for i, q in initial_states.items()}
        timings, resets, endings, isolation = [], [], [], []
        transitions = ticks = controls = 0
        results["state"] = "running"
        write_json(output, results)
        rollout_start = time.monotonic()
        payload_call_start = len(results["json_payloads"]["calls"])
        for chunk in range(args.chunks):
            items = [dict(env_id=i, episode_id=states[i]["episode_id"], step_id=states[i]["step_id"],
                          actions=np.repeat(target[i][None], horizon, axis=0).tolist()) for i in ids]
            begin = time.monotonic()
            rows = request("step", items=items)["items"]
            seconds = time.monotonic()-begin-results["json_payloads"]["calls"][-1]["measurement_seconds"]
            transitions += len(rows)
            ticks += sum(row["info"]["physics_steps"] for row in rows)
            controls += sum(row["info"]["elapsed_control_steps"] for row in rows)
            timings.append({"chunk": chunk, "seconds": seconds, "transitions": len(rows)})
            states.update({row["env_id"]: row for row in rows})
            if chunk == 0:
                measured = [float(states[i]["observation"]["state"][0]) for i in ids]
                results["first_chunk_joint1"] = measured
                results["first_chunk_joint1_change"] = [
                    measured[i]-float(initial_states[i][0]) for i in ids]
            done = [row["env_id"] for row in rows if row["terminated"] or row["truncated"]]
            for row in rows:
                if row["terminated"] or row["truncated"]:
                    endings.append({key: row[key] for key in ("env_id", "terminated", "truncated", "reward")}
                                   | {"reason": row["info"].get("termination_reason"),
                                      "episode_physics_steps": row["info"].get("episode_physics_steps")})
            if done:
                # Reset exactly one first, so the other slots must retain episode
                # IDs/clocks and, when exposed, physical state without advancing.
                groups = [done[:1], done[1:]] if len(done) > 1 else [done]
                for selected in filter(None, groups):
                    untouched = [i for i in ids if i not in selected]
                    before_slots = copy.deepcopy(service.slots)
                    before = optional_snapshot(backend, untouched) if untouched else None
                    begin = time.monotonic()
                    new = request("reset", env_ids=selected, seed=args.seed+chunk+1)["items"]
                    resets.append(dict(env_ids=selected, seconds=time.monotonic()-begin
                        -results["json_payloads"]["calls"][-1]["measurement_seconds"]))
                    after = optional_snapshot(backend, untouched) if untouched else None
                    unchanged_slots = all(service.slots[i] == before_slots[i] for i in untouched)
                    if not unchanged_slots:
                        raise RuntimeError("Indexed reset changed another environment's protocol state")
                    physical_equal = None if before is None else before == after
                    if physical_equal is False:
                        raise RuntimeError("Indexed reset changed another environment's physical diagnostic state")
                    isolation.append(dict(reset_ids=selected, untouched_count=len(untouched),
                        protocol_state_unchanged=unchanged_slots, physical_snapshot_unchanged=physical_equal))
                    states.update({row["env_id"]: row for row in new})
            print(json.dumps({"event": "chunk", "chunk": chunk, "step_seconds": seconds,
                              "transitions": transitions, "episodes": len(endings)}), flush=True)
        rollout_seconds = time.monotonic()-rollout_start-sum(row["measurement_seconds"] for row in
            results["json_payloads"]["calls"][payload_call_start:])
        step_seconds = sum(row["seconds"] for row in timings)
        results.update(state="completed", chunks=timings, resets=resets, episode_endings=endings,
            indexed_reset_checks=isolation, transitions=transitions,
            physics_ticks=ticks, control_steps=controls, step_seconds=step_seconds,
            rollout_including_resets_seconds=rollout_seconds,
            step_transitions_per_second=transitions/step_seconds,
            rollout_transitions_per_second=transitions/rollout_seconds,
            control_steps_per_second=controls/step_seconds,
            physics_ticks_per_second=ticks/step_seconds,
            final_joint1=[float(states[i]["observation"]["state"][0]) for i in ids],
            commanded_joint1=[float(target[i][0]) for i in ids],
            success_rate_not_evaluated=True)
        service.close_run({"run_id": run_id})
        write_json(output, results)
        return 0
    except BaseException as error:
        results.update(state="failed", error=f"{type(error).__name__}: {error}",
                       traceback=traceback.format_exc(), elapsed_seconds=time.monotonic()-started)
        write_json(output, results)
        traceback.print_exc()
        return 1
    finally:
        if service is not None:
            service.close()
        elif backend is not None:
            backend.close()


def parser():
    value = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    value.add_argument("--config", default=str(ROOT / "configs/rl_nodes/pap_simulation.json"))
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--factory", help="Override backend module:function")
    value.add_argument("--num-envs", type=int, default=2)
    value.add_argument("--gpu-id", type=int, default=0)
    value.add_argument("--render-resolution", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"))
    value.add_argument("--reset-render-frames", type=int,
                       help="Vector backend RGB history warmup override; omitted retains scene/default")
    value.add_argument("--render-subframes", type=int,
                       help="Vector backend render frames per observation; omitted retains scene/default")
    value.add_argument("--chunks", type=int, default=10)
    value.add_argument("--action-horizon", type=int, default=32)
    value.add_argument("--max-episode-seconds", type=float,
                       help="Default: 2 seconds for capacity, 20 seconds for recorded-action replay")
    value.add_argument("--replay-episode", type=Path,
                       help="Replay config.json + physics.npz applied targets to qualify physical behavior")
    value.add_argument("--seed", type=int, default=20261104)
    value.add_argument("--timeout", type=float, default=1800.)
    value.add_argument("--sample-seconds", type=float, default=1.)
    value.add_argument("--min-host-gib", type=float, default=8.)
    value.add_argument("--min-gpu-gib", type=float, default=2.)
    value.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    return value


def main():
    args = parser().parse_args()
    args.output_dir = args.output_dir.resolve()
    for name in ("num_envs", "chunks", "action_horizon", "max_episode_seconds", "timeout", "sample_seconds",
                 "reset_render_frames", "render_subframes"):
        if getattr(args, name) is not None and (not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0):
            raise SystemExit(f"--{name.replace('_', '-')} must be positive and finite")
    if args.gpu_id < 0 or args.seed < 0 or min(args.min_host_gib, args.min_gpu_gib) < 0:
        raise SystemExit("GPU ID, seed and memory thresholds must be nonnegative")
    return run_child(args) if args.child else run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
