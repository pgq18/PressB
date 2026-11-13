#!/usr/bin/env python3
"""Sequential fixed-attempt collection benchmark, including both camera videos.

Use the production collector and immutable sampling sequence at every env count.
Resource sampling is read-only; no robot or model service is involved.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import time

import monitor_runtime_resources as monitor

ROOT = Path(__file__).resolve().parents[1]


def write(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def source_files():
    paths = [ROOT/'scripts/collect_manipulation.py', Path(__file__).resolve(),
             ROOT/'scripts/monitor_runtime_resources.py']
    for folder in ('src/pressb/runtime', 'src/pressb/robots/piper', 'src/pressb/sensors',
                   'src/pressb/tasks/cube_stack', 'src/pressb/tasks/tabletop_cleanup',
                   'src/pressb/tasks/playground', 'configs'):
        paths.extend(p for p in (ROOT/folder).rglob('*') if p.suffix in ('.py', '.json', '.yaml', '.yml'))
    return sorted(set(paths))


def source_hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files()}


def run_case(args, task, envs, baseline_hashes):
    if source_hashes() != baseline_hashes:
        raise RuntimeError('Source/config changed during benchmark; use a new output for the changed variant')
    case = args.output/f'{task}_{envs}'
    case.mkdir()
    raw = case/'raw'
    cmd = [sys.executable, '-u', str(ROOT/'scripts/collect_manipulation.py'),
           '--task', task, '--output', str(raw), '--episodes', str(args.schedule_attempts),
           '--num-envs', str(envs), '--gpu', str(args.gpu), '--seed', str(args.seed),
           '--max-batches', str(args.attempts//envs), '--max-attempts', str(args.attempts),
           '--layout-env-id', '0', '--pair-schedule-by', 'attempt',
           '--trajectory-profile', 'compact', '--noise-deg', '0.12']
    if args.config:
        cmd += ['--config', str(args.config)]
    environment = dict(os.environ)
    environment.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       OMNI_KIT_ACCEPT_EULA='YES', PYTHONUNBUFFERED='1')
    start, mono = time.time(), time.monotonic()
    receipt = dict(task=task, num_envs=envs, command=cmd, start_unix=start,
                   start_utc=datetime.fromtimestamp(start, timezone.utc).isoformat(),
                   variant=args.variant, gpu=args.gpu, status='running')
    records = []
    with (case/'collector.log').open('x') as log, (case/'resources.jsonl').open('x', buffering=1) as stream:
        child = subprocess.Popen(cmd, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        receipt['pid'] = child.pid
        write(case/'receipt.json', receipt)
        processes = monitor.ProcessSampler(pids=[('collector', child.pid)])
        try:
            while child.poll() is None:
                tick, now = time.monotonic(), time.time()
                gpu = monitor.gpu_snapshot()
                status = monitor.read_json_file(raw/'progress.json', now)
                record = dict(schema_version=1, hostname=socket.gethostname(), wall_time=now,
                    monotonic_seconds=tick, sample_index=len(records),
                    monitor_elapsed_seconds=tick-mono, requested_interval_seconds=args.interval,
                    status=status, marker=None,
                    phase=dict(name='collection', source='driver', explicit=True),
                    gpu=gpu, host=monitor.host_snapshot(first=not records),
                    processes=processes.sample(gpu['processes'], time.monotonic()))
                records.append(record)
                stream.write(json.dumps(record, allow_nan=False)+'\n')
                write(args.output/'status.json', dict(status='running', case=case.name,
                    elapsed_seconds=tick-mono, collector=(status or {}).get('data'), pid=child.pid))
                time.sleep(max(0, args.interval-(time.monotonic()-tick)))
        except BaseException:
            child.send_signal(__import__('signal').SIGINT)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.terminate()
                child.wait(timeout=30)
            raise
    receipt.update(exit_code=child.returncode, end_unix=time.time(),
                   process_wall_seconds=time.monotonic()-mono, status='finished')
    write(case/'receipt.json', receipt)
    if child.returncode:
        raise RuntimeError(f'{case.name}: collector exited {child.returncode}; see {case}/collector.log')
    profile = json.loads((raw/'collection_profile.json').read_text())
    final_progress = json.loads((raw/'progress.json').read_text())
    if profile['counts'].get('sampled_layouts') != args.attempts:
        raise RuntimeError(f'Unexpected attempt count: {profile["counts"]}')
    resources = monitor.summarize(records, gpu_selectors=[str(args.gpu)], roles=['collector'])
    write(case/'resources_summary.json', resources)
    # Startup includes import/Isaac setup. The remaining interval includes reset,
    # planning, execution and file close; it is not called pure execution time.
    post_startup = start + profile['exclusive_stage_seconds'].get('startup', 0)
    post_records = [r for r in records if r['wall_time'] >= post_startup]
    write(case/'resources_post_startup.json', monitor.summarize(post_records,
        gpu_selectors=[str(args.gpu)], roles=['collector']))
    result = dict(receipt=receipt, progress=final_progress, profile=profile,
                  resources=resources, path=str(case))
    write(case/'result.json', result)
    print(json.dumps(dict(case=case.name, wall_seconds=profile['wall_seconds'],
                         counts=profile['counts'], stages=profile['exclusive_stage_seconds'])), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tasks', nargs='+', choices=['cube_stack', 'tabletop_cleanup'],
                        default=['cube_stack', 'tabletop_cleanup'])
    parser.add_argument('--envs', nargs='+', type=int, default=[8, 16])
    parser.add_argument('--attempts', type=int, default=32)
    parser.add_argument('--schedule-attempts', type=int,
                        help='Keep an original longer object-pair schedule when testing its attempt prefix')
    parser.add_argument('--seed', type=int, default=20261113)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--interval', type=float, default=2.)
    parser.add_argument('--variant', default='baseline')
    parser.add_argument('--config', type=Path)
    args = parser.parse_args()
    args.schedule_attempts = args.attempts if args.schedule_attempts is None else args.schedule_attempts
    args.output = args.output.resolve()
    if args.attempts < 1 or any(n < 1 or args.attempts % n for n in args.envs) or args.interval <= 0:
        parser.error('Positive counts required; attempts must be divisible by each env count')
    if args.schedule_attempts < args.attempts:
        parser.error('Schedule size must cover all attempted layouts')
    if args.config:
        args.config = args.config.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    hashes = source_hashes()
    for relative in hashes:
        destination = args.output/'source_snapshot'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/relative, destination)
    write(args.output/'manifest.json', dict(schema_version=1, variant=args.variant,
        tasks=args.tasks, envs=args.envs, attempts=args.attempts, schedule_attempts=args.schedule_attempts,
        seed=args.seed, gpu=args.gpu, config=None if args.config is None else str(args.config),
        config_sha256=None if args.config is None else hashlib.sha256(args.config.read_bytes()).hexdigest(),
        python=sys.executable, platform=platform.platform(), hostname=socket.gethostname(),
        cpu_count=os.cpu_count(), source_sha256=hashes,
        sampling='Fixed attempt index selects seed/object pair, logical layout env ID=0; physical slot varies.',
        video='Both cameras 640x480 at 30 Hz, production x264 fast CRF19, thread=1 per camera.',
        limitations=['These attempts are a throughput sample, not a full success-rate evaluation.',
                     'Wall includes simulator startup; excludes teardown and LeRobot export.',
                     'Resource maxima sampled every 2 seconds may miss brief peaks.']))
    results = []
    try:
        for task in args.tasks:
            for envs in args.envs:
                results.append(run_case(args, task, envs, hashes))
                write(args.output/'results.json', results)
        write(args.output/'status.json', dict(status='complete', cases=len(results)))
    except BaseException as error:
        write(args.output/'status.json', dict(status='failed', completed_cases=len(results), error=str(error)))
        raise


if __name__ == '__main__':
    main()
