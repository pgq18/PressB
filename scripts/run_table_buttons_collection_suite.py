#!/usr/bin/env python3
"""Collect a fresh dataset or expand a raw prefix with three collectors and two CPU exporters.

Preparation is the default: only --run starts subprocesses. The collector,
demonstrations, rendering, physical success checks and LeRobot writer are reused
unchanged. A failed attempt stays rejected; an interrupted write stays pending.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ("near", "medium", "far")
COLORS = ("red", "yellow", "blue")
GROUPS = tuple(f"{profile}_{color}" for profile in PROFILES for color in COLORS)
EPISODE_FILES = ("config.json", "contacts.json", "data.npz", "final_up.jpg", "final_wrist.jpg",
                 "initial_up.jpg", "initial_wrist.jpg", "metadata.json", "physics.npz", "preflight.json",
                 "report.json", "unexpected_contacts.json", "up.mp4", "wrist.mp4")


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            result.update(block)
    return result.hexdigest()


def immutable_json(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError(f"Existing preparation differs: {path}")
    else:
        write(path, value)


@contextmanager
def exclusive_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".supervisor.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another supervisor owns {output}") from error
        yield


def frozen_paths():
    scripts = ["collect_table_buttons.py", "export_table_buttons.py", "audit_table_buttons_dataset.py",
               "run_table_buttons_collection_suite.py"]
    paths = [ROOT / "scripts" / name for name in scripts]
    paths += [ROOT / "src/pressb" / name for name in ("__init__.py", "kinematics.py", "wrist_camera.py")]
    paths += sorted((ROOT / "src/pressb/table_buttons").glob("*.py"))
    paths += [ROOT / "src/pressb/pick_place" / name for name in ("__init__.py", "scene.py")]
    return sorted(set(path for path in paths if path.is_file()))


def freeze_code(output):
    manifest_path = output / "source_archive/manifest.json"
    if manifest_path.exists():
        verify_code(output)
        return
    entries = {}
    for source in frozen_paths():
        relative = source.relative_to(ROOT)
        destination = output / "source_archive" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        entries[str(relative)] = {"sha256": digest(source), "bytes": source.stat().st_size}
        if digest(destination) != entries[str(relative)]["sha256"]:
            raise RuntimeError(f"Source changed while freezing: {source}")
    write(manifest_path, {"created_at": now(), "repository": str(ROOT), "files": entries,
                         "policy": "Verify every listed live dependency before each subprocess launch; stop on change."})


def verify_code(output):
    for relative, entry in read(output / "source_archive/manifest.json")["files"].items():
        source = ROOT / relative
        if not source.is_file() or digest(source) != entry["sha256"]:
            raise RuntimeError(f"Frozen generation dependency changed: {source}")
    imports = output / "import_manifest.json"
    if imports.exists():
        for profile, entry in read(imports)["source_plans"].items():
            plan_path = output / "raw_shards" / profile / "plan.json"
            if digest(plan_path) != entry["new_plan_sha256"] or read(plan_path)["config"] != read(plan_path.parent / "frozen_config.json"):
                raise RuntimeError(f"Frozen collection plan/config changed: {plan_path}")


def validate_metadata(directory, group, index):
    if directory.is_symlink():
        raise ValueError(f"Accepted episode must be a real copy/directory: {directory}")
    for name in EPISODE_FILES:
        path = directory / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing regular episode file: {path}")
    metadata = read(directory / "metadata.json")
    if metadata.get("success") is not True or metadata.get("group") != group or metadata.get("episode_index") != index:
        raise ValueError(f"Invalid accepted episode identity: {directory}")
    return metadata


def fresh_plan(config, profile, quota):
    # Keep the immutable contract identical to collect_table_buttons.py so the
    # first launch and all later rotations use its normal --resume checks.
    return dict(schema_version="pressb.table_buttons.collection_plan.v1",
                config=config, profiles=[profile], targets=list(COLORS),
                episodes_per_group=quota, seed_start=0, max_retries=50, fps=30, physics_hz=120,
                instructions={color: f"Press the {color} button." for color in COLORS},
                seed_rule="first attempt seed_start+episode_index; retry 1000000+1000*(seed_start+episode_index)+retry",
                success_only=True, demonstration_end="first control boundary after physical success")


def prepare(args):
    source = args.source.resolve() if args.source is not None else None
    output, destination = args.output.resolve(), args.dataset_root.resolve()
    config_path = getattr(args, "config", None)
    fresh = config_path is not None
    if fresh:
        if source is not None or args.reuse_per_group != 0 or args.episodes_per_group < 1:
            raise ValueError("Fresh collection requires --config, no --source, zero reused episodes, and positive quota")
        config_path = config_path.resolve()
        config = read(config_path)
        if config.get("fps") != 30 or abs(config.get("physics_dt", 0) - 1/120) > 1e-12:
            raise ValueError("Fresh collection requires 30 Hz cameras/actions and 120 Hz physics")
    else:
        if source is None:
            raise ValueError("Specify --config for fresh collection or --source for expansion")
        if source == output or source in output.parents or output in source.parents:
            raise ValueError("Source and new output must be separate, non-nested roots")
        if args.reuse_per_group < 1 or args.episodes_per_group <= args.reuse_per_group:
            raise ValueError("Expansion requires quota > reused prefix > 0")
    contract = dict(schema="pressb.table_buttons.collection_fresh.v1" if fresh else "pressb.table_buttons.collection_expansion.v1",
                    source=str(source) if source is not None else None, output=str(output),
                    dataset_root=str(destination), episodes_per_group=args.episodes_per_group,
                    reuse_per_group=args.reuse_per_group, seed_start=0, max_retries=50,
                    groups=list(GROUPS), requested_episodes=9 * args.episodes_per_group,
                    reused_episodes=9 * args.reuse_per_group,
                    new_episodes=9 * (args.episodes_per_group - args.reuse_per_group))
    if fresh:
        contract.update(config_source=str(config_path), config_sha256=digest(config_path))
    immutable_json(output / "dataset_plan.json", contract)
    freeze_code(output)
    records, source_plans = {}, {}
    for profile in PROFILES:
        shard = output / "raw_shards" / profile
        if fresh:
            plan = fresh_plan(config, profile, args.episodes_per_group)
            origin = {"path": str(config_path), "sha256": contract["config_sha256"], "kind": "fresh_config"}
            old_layouts = []
        else:
            old_shard = source / "raw_shards" / profile
            old_plan = read(old_shard / "plan.json")
            if (old_plan["profiles"] != [profile] or old_plan["targets"] != list(COLORS)
                    or old_plan["seed_start"] != 0 or old_plan["max_retries"] != 50
                    or old_plan["episodes_per_group"] != args.reuse_per_group):
                raise ValueError(f"Unexpected source collection contract: {old_shard}")
            plan = dict(old_plan, episodes_per_group=args.episodes_per_group)
            origin = {"path": str(old_shard / "plan.json"), "sha256": digest(old_shard / "plan.json")}
            old_layouts = sorted((old_shard / "layouts").glob("*.json"))
        immutable_json(shard / "plan.json", plan)
        immutable_json(shard / "frozen_config.json", plan["config"])
        source_plans[profile] = dict(origin, new_plan_sha256=digest(shard / "plan.json"))
        layout_dir = shard / "layouts"
        layout_dir.mkdir(parents=True, exist_ok=True)
        for path in old_layouts:
            dest = layout_dir / path.name
            if not dest.exists():
                shutil.copy2(path, dest)
            if dest.is_symlink() or digest(path) != digest(dest) or path.stat().st_ino == dest.stat().st_ino:
                raise ValueError(f"Layout is not an independent exact copy: {dest}")
        for color in COLORS:
            group = f"{profile}_{color}"
            group_dir = shard / group
            group_dir.mkdir(exist_ok=True)
            for index in range(args.reuse_per_group):
                name = f"episode_{index:06d}"
                old, dest = old_shard / group / name, group_dir / name
                validate_metadata(old, group, index)
                if not dest.exists():
                    temporary = group_dir / f".importing-{name}-{uuid.uuid4().hex[:8]}"
                    shutil.copytree(old, temporary, copy_function=shutil.copy2)
                    temporary.rename(dest)
                validate_metadata(dest, group, index)
                for path in sorted(old.rglob("*")):
                    if not path.is_file() or path.is_symlink():
                        raise ValueError(f"Source episode contains non-regular entry: {path}")
                    target = dest / path.relative_to(old)
                    file_hash = digest(path)
                    if target.is_symlink() or digest(target) != file_hash or path.stat().st_ino == target.stat().st_ino:
                        raise ValueError(f"Prefix is not an independent exact copy: {target}")
                    records[str(target.relative_to(output))] = {"source": str(path), "sha256": file_hash,
                                                              "bytes": path.stat().st_size}
            aggregate = output / "raw" / group
            aggregate.parent.mkdir(exist_ok=True)
            if not aggregate.exists() and not aggregate.is_symlink():
                aggregate.symlink_to(group_dir, target_is_directory=True)
            if not aggregate.is_symlink() or aggregate.resolve() != group_dir:
                raise ValueError(f"Raw aggregation must point to this run's new group: {aggregate}")
    immutable_json(output / "import_manifest.json", dict(source_plans=source_plans, files=records,
                   copied_episodes=contract["reused_episodes"], copied_bytes=sum(row["bytes"] for row in records.values()),
                   copy_method="none; all episodes newly collected" if fresh else
                       "shutil.copy2; regular files; distinct inode; source/destination SHA-256 verified"))
    immutable_json(output / "prepared.json", dict(**contract, preparation_complete=True,
                   source_archive=str(output / "source_archive/manifest.json")))
    return contract


class EpisodeIndex:
    def __init__(self):
        self.cache = {}

    def scan(self, output, quota):
        groups = {}
        for group in GROUPS:
            directory = output / "raw_shards" / group.split("_")[0] / group
            paths = sorted(directory.glob("episode_*"))
            if [path.name for path in paths] != [f"episode_{i:06d}" for i in range(len(paths))] or len(paths) > quota:
                raise ValueError(f"Nonconsecutive or excessive accepted episodes: {directory}")
            rows = []
            for index, path in enumerate(paths):
                if path not in self.cache:
                    self.cache[path] = validate_metadata(path, group, index)
                rows.append(self.cache[path])
            if len({row["seed"] for row in rows}) != len(rows):
                raise ValueError(f"Duplicate accepted seed: {group}")
            groups[group] = rows
        return groups


def pairing_report(groups):
    checked, mismatches = 0, []
    for profile in PROFILES:
        rows = [groups[f"{profile}_{color}"] for color in COLORS]
        for index in range(min(map(len, rows))):
            entries = [items[index] for items in rows]
            checked += 1
            same = all((entry["seed"], entry["positions_xy_m"]) ==
                       (entries[0]["seed"], entries[0]["positions_xy_m"]) for entry in entries[1:])
            if not same:
                mismatches.append(dict(profile=profile, episode_index=index,
                    seeds={color: entry["seed"] for color, entry in zip(COLORS, entries)},
                    positions_xy_m={color: entry["positions_xy_m"] for color, entry in zip(COLORS, entries)}))
    return dict(checked_episode_indices=checked, mismatch_count=len(mismatches), mismatches=mismatches,
                interpretation="Independent per-color rejection retries may break pairing; valid independent datasets remain valid.")


def classify_collector_exit(returncode, status, count, expected):
    if returncode == 75 and status.get("state") == "restart_required":
        if status.get("restart_reason") == "requested":
            return "stop_requested"
        if status.get("restart_reason") == "release_cpu_stage_caches":
            return "rotation"
    if returncode == 0 and status.get("state") == "complete" and count == expected:
        return "complete"
    if "within retry budget" in status.get("error", ""):
        return "retry_budget_exhausted"
    return "infrastructure_failure"


def exported_complete(directory, group, quota, raw):
    if not directory.exists():
        return False
    manifest, audit = read(directory / "meta/export_manifest.json"), read(directory / "meta/audit.json")
    info = read(directory / "meta/info.json")
    return (manifest.get("group") == group and Path(manifest["raw_root"]).resolve() == raw.resolve()
            and len(manifest.get("episodes", [])) == quota and info.get("total_episodes") == quota
            and audit.get("success") is True and audit.get("episodes") == quota)


def environment():
    env = os.environ.copy()
    env.setdefault("DISPLAY", ":1")
    for key in ("CUDA_VISIBLE_DEVICES", "PYTHONPATH"):
        env.pop(key, None)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    return env


def process_identity(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if stat[0] == "Z" else {"pid": pid, "start_ticks": stat[19],
                    "cmdline": Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()}
    except (FileNotFoundError, ProcessLookupError):
        return None


def ensure_no_owned_children(output):
    path = output / "suite_status.json"
    if path.exists():
        for child in read(path).get("active_processes", {}).values():
            current = process_identity(child["pid"])
            if current and current["start_ticks"] == child["start_ticks"]:
                raise RuntimeError(f"Previous supervisor's child remains alive; do not duplicate: {current}")


def collection_command(args, profile):
    shard = args.output / "raw_shards" / profile
    profile_gpus = getattr(args, "profile_gpus", None)
    gpu = profile_gpus[PROFILES.index(profile)] if profile_gpus is not None else args.gpu
    return [str(ROOT / ".conda/envs/pressb/bin/python"), str(ROOT / "scripts/collect_table_buttons.py"),
            "--config", str(shard / "frozen_config.json"), "--output", str(shard), "--profiles", profile,
            "--targets", *COLORS, "--episodes-per-group", str(args.episodes_per_group), "--seed-start", "0",
            "--max-retries", "50", "--max-new-episodes", str(args.max_new_episodes), "--gpu", str(gpu), "--resume"]


class Controller:
    def __init__(self, args):
        self.args, self.output = args, args.output
        self.active, self.index, self.stopping = {}, EpisodeIndex(), False
        self.state = read(self.output / "suite_status.json") if (self.output / "suite_status.json").exists() else {}
        self.state.update(state="starting", controller=process_identity(os.getpid()), active_processes={}, updated_at=now())
        self.state.setdefault("started_at", now())
        self.state.setdefault("process_events", [])
        self.state.setdefault("infrastructure_failures", {})
        self.state.setdefault("rotations", {profile: 0 for profile in PROFILES})
        self.state.setdefault("exports", {})
        self.attempt_offsets, self.attempt_counts = {}, {}

    def start(self, key, command, kind):
        verify_code(self.output)
        number = sum(event.get("event") == "start" and event.get("key") == key for event in self.state["process_events"])
        log = self.output / "logs" / f"{key}_{number:04d}.log"
        log.parent.mkdir(exist_ok=True)
        with log.open("a") as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=environment(), stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        identity = process_identity(process.pid)
        record = dict(**identity, kind=kind, command=command, log=str(log), started_at=now())
        self.active[key] = (process, record)
        self.state["process_events"].append(dict(event="start", key=key, **record))
        self.publish()

    def attempts(self):
        for profile in PROFILES:
            counts = self.attempt_counts.setdefault(profile, dict(accepted=0, physical_rejections=0,
                    planning_rejections=0, retry_attempts=0))
            path = self.output / "raw_shards" / profile / "attempts.jsonl"
            if not path.exists():
                continue
            with path.open() as handle:
                handle.seek(self.attempt_offsets.get(profile, 0))
                while True:
                    offset, line = handle.tell(), handle.readline()
                    if not line or not line.endswith("\n"):
                        handle.seek(offset)
                        break
                    row = json.loads(line)
                    counts["retry_attempts"] += int(row["retry"] > 0)
                    key = "accepted" if row["accepted"] else ("planning_rejections" if row["stop_reason"] == "exception" else "physical_rejections")
                    counts[key] += 1
                self.attempt_offsets[profile] = handle.tell()
        return self.attempt_counts

    def publish(self, groups=None):
        if groups is not None:
            self.state["groups"] = {group: dict(accepted=len(rows), reused=min(len(rows), self.args.reuse_per_group),
                new=max(0, len(rows)-self.args.reuse_per_group), requested=self.args.episodes_per_group,
                remaining=self.args.episodes_per_group-len(rows)) for group, rows in groups.items()}
            self.state["totals"] = {key: sum(row[key] for row in self.state["groups"].values())
                                     for key in ("accepted", "reused", "new", "requested", "remaining")}
            pairing = pairing_report(groups)
            write(self.output / "pairing_report.json", pairing)
            self.state["pairing"] = {key: pairing[key] for key in ("checked_episode_indices", "mismatch_count")}
        self.state.update(updated_at=now(), active_processes={key: record for key, (_, record) in self.active.items()},
                          attempts=self.attempts())
        write(self.output / "suite_status.json", self.state)

    def fail_process(self, key, reason):
        failures = self.state["infrastructure_failures"]
        failures[key] = failures.get(key, 0) + 1
        if failures[key] > self.args.max_infrastructure_restarts:
            raise RuntimeError(f"{key}: infrastructure restart budget exhausted: {reason}")

    def reap(self, groups):
        for key, (process, record) in list(self.active.items()):
            code = process.poll()
            if code is None:
                continue
            del self.active[key]
            event = dict(event="exit", key=key, returncode=code, finished_at=now(), log=record["log"])
            kind = record["kind"]
            if kind == "collector":
                profile = key.removeprefix("collect_")
                status_path = self.output / "raw_shards" / profile / "status.json"
                status = read(status_path) if status_path.exists() else {}
                # The last atomic commit can occur between the loop's scan and
                # poll(). Re-read before deciding a completed worker is short.
                groups.update(self.index.scan(self.output, self.args.episodes_per_group))
                count = sum(len(groups[f"{profile}_{color}"]) for color in COLORS)
                result = classify_collector_exit(code, status, count, 3*self.args.episodes_per_group)
                event.update(classification=result, collector_status=status)
                self.state["process_events"].append(event)
                if result == "rotation":
                    self.state["rotations"][profile] += 1
                elif result == "stop_requested":
                    self.stopping = True
                elif result == "retry_budget_exhausted":
                    raise RuntimeError(status["error"])
                elif result == "infrastructure_failure" and not self.stopping:
                    self.fail_process(key, status.get("error", code))
            elif kind == "export":
                group = key.removeprefix("export_")
                valid = code == 0 and exported_complete(self.args.dataset_root / group, group,
                          self.args.episodes_per_group, self.output / "raw" / group)
                event["classification"] = "complete" if valid else "infrastructure_failure"
                self.state["process_events"].append(event)
                self.state["exports"][group] = "complete" if valid else "retry_pending"
                if not valid and not self.stopping:
                    self.fail_process(key, code)
            else:
                path = self.output / "audit.inprogress.json"
                audit = read(path) if path.exists() else {}
                valid = code == 0 and audit.get("success") is True and audit.get("total_episodes") == 9*self.args.episodes_per_group
                event["classification"] = "complete" if valid else "audit_failed"
                self.state["process_events"].append(event)
                if not valid:
                    raise RuntimeError(f"Final audit did not pass: {path}; exit={code}")
                path.replace(self.output / "audit.json")
                self.state["state"] = "complete"

    def stop_children(self):
        for process, record in self.active.values():
            if process.poll() is None:
                if record["kind"] == "collector":
                    # Let the collector finish its episode; its ffmpeg children
                    # must keep encoding until that final write is committed.
                    process.send_signal(signal.SIGTERM)
                else:
                    os.killpg(process.pid, signal.SIGTERM)
        deadline = time.monotonic()+90
        while time.monotonic() < deadline and any(p.poll() is None for p, _ in self.active.values()):
            time.sleep(1)
        for process, _ in self.active.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        self.active.clear()

    def run(self):
        def stop(*_):
            self.stopping = True
        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        try:
            while not self.stopping:
                groups = self.index.scan(self.output, self.args.episodes_per_group)
                self.reap(groups)
                if self.stopping:
                    break
                if self.state["state"] == "complete":
                    self.publish(groups)
                    write(self.output / "completion_summary.json", dict(success=True, completed_at=now(),
                          totals=self.state["totals"], exports=self.state["exports"], attempts=self.state["attempts"],
                          pairing=self.state["pairing"], audit=str(self.output / "audit.json"),
                          dataset_root=str(self.args.dataset_root), task="Physical scripted demonstrations; not policy evaluation."))
                    return 0
                if self.state["state"] != "auditing":
                    self.state["state"] = "collecting_and_exporting"
                    for profile in PROFILES:
                        key = f"collect_{profile}"
                        if key not in self.active and any(len(groups[f"{profile}_{c}"]) < self.args.episodes_per_group for c in COLORS):
                            self.start(key, collection_command(self.args, profile), "collector")
                    for group in GROUPS:
                        key = f"export_{group}"
                        if len(groups[group]) != self.args.episodes_per_group or key in self.active or self.state["exports"].get(group) == "complete":
                            continue
                        if sum(record["kind"] == "export" for _, record in self.active.values()) >= 2:
                            break
                        command = [str(ROOT / ".conda/envs/lerobot/bin/python"), str(ROOT / "scripts/export_table_buttons.py"),
                                   "--raw", str(self.output / "raw" / group), "--output", str(self.args.dataset_root / group),
                                   "--episodes", str(self.args.episodes_per_group)]
                        self.state["exports"][group] = "running"
                        self.start(key, command, "export")
                    if len(self.state["exports"]) == 9 and all(value == "complete" for value in self.state["exports"].values()) and not self.active:
                        self.state["state"] = "auditing"
                        self.start("final_audit", [str(ROOT / ".conda/envs/lerobot/bin/python"),
                            str(ROOT / "scripts/audit_table_buttons_dataset.py"), "--raw-root", str(self.output / "raw"),
                            "--expected-per-group", str(self.args.episodes_per_group), "--lerobot-root", str(self.args.dataset_root),
                            "--output", str(self.output / "audit.inprogress.json")], "audit")
                self.publish(groups)
                print(json.dumps({"time": now(), "state": self.state["state"], "totals": self.state["totals"],
                    "groups": {g: len(r) for g, r in groups.items()}, "active": list(self.active),
                    "exports": self.state["exports"], "infrastructure_failures": self.state["infrastructure_failures"]}), flush=True)
                deadline = time.monotonic()+self.args.poll_seconds
                while not self.stopping and time.monotonic() < deadline:
                    time.sleep(min(1, max(0, deadline-time.monotonic())))
            self.state["state"] = "stopped"
            return 130
        except BaseException as error:
            self.state.update(state="failed", error=repr(error))
            raise
        finally:
            self.stop_children()
            self.publish(self.index.scan(self.output, self.args.episodes_per_group))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    input_mode = parser.add_mutually_exclusive_group()
    input_mode.add_argument("--source", type=Path,
        help="Expand an existing raw suite; defaults to the historical 100-episode suite when --config is absent")
    input_mode.add_argument("--config", type=Path,
        help="Collect all nine groups from scratch using this config; no episodes or layouts are imported")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/table_buttons_dataset_1000_20261025")
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/pengguanqi/Datasets/piper_table_buttons_1000_20261025"))
    parser.add_argument("--episodes-per-group", type=int, default=1000)
    parser.add_argument("--reuse-per-group", type=int,
        help="Defaults to 0 with --config and 100 for expansion; fresh collection requires 0")
    parser.add_argument("--max-new-episodes", type=int, default=50)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--profile-gpus", type=int, nargs=3, metavar=("NEAR", "MEDIUM", "FAR"),
        help="Physical GPU indices for the three collectors; overrides --gpu")
    parser.add_argument("--max-infrastructure-restarts", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=15)
    parser.add_argument("--run", action="store_true", help="Start collection/export only after preparation; omitted means prepare only")
    args = parser.parse_args()
    if args.source is None and args.config is None:
        args.source = ROOT / "outputs/table_buttons_dataset_20261023"
    if args.reuse_per_group is None:
        args.reuse_per_group = 0 if args.config is not None else 100
    for name in ("output", "source", "config", "dataset_root"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if args.max_new_episodes < 1 or args.max_infrastructure_restarts < 0 or args.poll_seconds <= 0:
        parser.error("Positive rotation/poll interval and nonnegative retry budget required")
    if args.gpu < 0 or (args.profile_gpus is not None and min(args.profile_gpus) < 0):
        parser.error("GPU indices must be nonnegative")
    with exclusive_lock(args.output):
        ensure_no_owned_children(args.output)
        contract = prepare(args)
        write(args.output / "launch_commands.json", dict(collectors={p: collection_command(args, p) for p in PROFILES},
              gpu=args.gpu, profile_gpus=dict(zip(PROFILES, args.profile_gpus or [args.gpu]*3)),
              rotation_new_episodes=args.max_new_episodes, simultaneous_cpu_exports=2,
              environment={key: value for key, value in environment().items() if key in (
                  "DISPLAY", "XAUTHORITY", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
              unset_environment=["CUDA_VISIBLE_DEVICES", "PYTHONPATH"]))
        if not args.run:
            print(json.dumps(dict(prepared=True, **contract)), flush=True)
            return 0
        return Controller(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
