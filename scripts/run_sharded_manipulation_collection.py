#!/usr/bin/env python3
"""Finish a canonical raw collection with isolated, resumable GPU workers.

Existing episodes and manifest are immutable. Each worker owns a disjoint set of
global episode IDs and attempt-number residue class. Successful episodes are
hardlinked (copied across filesystems) into a temporary canonical directory and
published by rename; worker data remain intact for restart accounting.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import fcntl
import hashlib
import importlib.util
from itertools import combinations
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('_manipulation_pipeline_util', ROOT/'scripts/run_manipulation_dataset_pipeline.py')
_util = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_util)
read_json, write_json, process_identity = _util.read_json, _util.write_json, _util.process_identity
PAIRS = tuple(combinations(('gamepad', 'speaker', 'lidar'), 2))
REQUIRED = ('metadata.json', 'data.npz', 'up.mp4', 'wrist.mp4', 'plan.json', 'settled.json', 'physics.jsonl')


def trajectory_profile(record):
    """Old manifests/episodes predate profiles and describe legacy motion."""
    profile = record.get('trajectory_profile', 'legacy')
    if profile not in ('legacy', 'compact'):
        raise ValueError(f'Invalid trajectory profile: {profile!r}')
    return profile


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024*1024), b''):
            h.update(data)
    return h.hexdigest()


def validate_episode(path, manifest, expected_id=None):
    match = re.fullmatch(r'episode_(\d{6,})', path.name)
    if not match:
        raise ValueError(f'Invalid episode directory: {path}')
    index = int(match[1]); metadata = read_json(path/'metadata.json', {})
    if (expected_id is not None and index != expected_id) or not 0 <= index < manifest['target_episodes']:
        raise ValueError(f'Episode ID outside assigned range: {path}')
    if trajectory_profile(metadata) != trajectory_profile(manifest):
        raise ValueError(f'Episode trajectory profile differs from canonical collection: {path}')
    if (metadata.get('episode_index') != index or metadata.get('task_id') != manifest['task_id']
            or metadata.get('instruction') != manifest['instruction'] or metadata.get('success') is not True
            or metadata.get('audit', {}).get('passed') is not True
            or type(metadata.get('frame_count')) is not int or metadata['frame_count'] < 2
            or type(metadata.get('seed')) is not int or type(metadata.get('attempt_index')) is not int
            or metadata['seed'] != manifest['seed'] + 1009*metadata['attempt_index']
            or metadata.get('noise', {}).get('amplitude_deg') != manifest['noise_deg']
            or metadata.get('fps') != 30 or metadata.get('camera_resolution') != [640, 480]
            or metadata.get('state_units') != ['rad']*6+['m']
            or metadata.get('action_units') != ['rad']*6+['m']
            or metadata.get('gripper_representation') != 'full_width'
            or metadata.get('terminal_action') != 'repeat_previous_interval_target'):
        raise ValueError(f'Invalid physical acceptance / capture metadata: {path}')
    if any(not (path/name).is_file() or (path/name).is_symlink() for name in REQUIRED):
        raise ValueError(f'Missing or nonregular episode files: {path}')
    if manifest['task_id'] == 'tabletop_cleanup':
        expected = set(PAIRS[manifest['pair_schedule'][index]])
        if set(metadata.get('reset_spec', {}).get('objects', {})) != expected:
            raise ValueError(f'Global color/object-pair schedule mismatch: {path}')
    return metadata


def scan_episodes(raw, manifest):
    result = {}; seeds = set()
    for path in sorted((raw/'episodes').glob('episode_*')):
        metadata = validate_episode(path, manifest)
        if metadata['seed'] in seeds:
            raise ValueError(f'Duplicate seed in canonical dataset: {metadata["seed"]}')
        seeds.add(metadata['seed']); result[metadata['episode_index']] = metadata
    return result


def max_previous_attempt(raw, episodes):
    maximum = int(read_json(raw/'progress.json', {}).get('attempts', 0))
    maximum = max([maximum]+[m['attempt_index'] for m in episodes.values()])
    for folder in ('attempts', 'rejected'):
        for path in (raw/folder).glob('attempt_*'):
            match = re.match(r'attempt_(\d+)', path.name)
            if match:
                maximum = max(maximum, int(match[1]))
    for path in raw.glob('batch_ending_attempt_*.json'):
        match = re.match(r'batch_ending_attempt_(\d+)', path.name)
        if match:
            maximum = max(maximum, int(match[1]))
    return maximum


def source_files(task):
    relative = ['scripts/collect_manipulation.py', 'scripts/run_sharded_manipulation_collection.py',
                'src/pressb/runtime/demonstrations.py', 'src/pressb/runtime/parallel.py',
                'src/pressb/runtime/isaac_parallel.py', f'src/pressb/tasks/{task}/collection.py',
                f'src/pressb/tasks/{task}/planning.py']
    return {name:digest(ROOT/name) for name in relative}


def make_plan(raw, manifest, episodes, gpus, num_envs, max_attempts):
    profile = trajectory_profile(manifest)
    missing = sorted(set(range(manifest['target_episodes']))-set(episodes))
    maximum = max_previous_attempt(raw, episodes)
    config = raw/'shards/base_config.json'
    write_json(config, manifest['config'])
    workers = []
    for index, gpu in enumerate(gpus):
        ids = missing[index::len(gpus)]
        folder = raw/'shards'/f'shard_{index:02d}'
        ids_path = raw/'shards'/f'episode_indices_{index:02d}.json'
        write_json(ids_path, ids)
        workers.append(dict(index=index, gpu=gpu, num_envs=num_envs, episode_indices=ids,
                            episode_indices_path=str(ids_path), episode_indices_sha256=digest(ids_path),
                            output=str(folder), attempt_offset=maximum+index, attempt_stride=len(gpus),
                            runtime_metadata_path=str(folder/'runtime_metadata.json'),
                            receipt=str(raw/'shards'/f'worker_{index:02d}.json'),
                            log=str(raw/'shards'/f'worker_{index:02d}.log')))
    return dict(schema_version=2, task_id=manifest['task_id'], target_episodes=manifest['target_episodes'],
                trajectory_profile=profile,
                canonical=str(raw), canonical_manifest_sha256=digest(raw/'manifest.json'),
                seed=manifest['seed'], noise_deg=manifest['noise_deg'], max_previous_attempt=maximum,
                initial_episode_metadata_sha256={str(i):digest(raw/'episodes'/f'episode_{i:06d}'/'metadata.json') for i in episodes},
                initial_accepted=len(episodes), max_attempts=max_attempts, config_path=str(config),
                config_sha256=digest(config), source_files=source_files(manifest['task_id']),
                workers=workers, created_unix=time.time(),
                publication='Hardlink/copy complete episode to hidden directory; atomic rename; worker source retained')


def validate_plan(plan, raw, manifest, episodes, gpus, num_envs, max_attempts):
    if plan.get('schema_version') not in (1, 2):
        raise ValueError('Unsupported shard plan schema version')
    if plan['schema_version'] == 2 and 'trajectory_profile' not in plan:
        raise ValueError('Shard plan schema 2 requires an explicit trajectory profile')
    if (plan['canonical_manifest_sha256'] != digest(raw/'manifest.json')
            or plan['canonical'] != str(raw) or plan['task_id'] != manifest['task_id']
            or trajectory_profile(plan) != trajectory_profile(manifest)
            or [w['gpu'] for w in plan['workers']] != gpus
            or any(w['num_envs'] != num_envs for w in plan['workers'])
            or plan['max_attempts'] != max_attempts
            or digest(plan['config_path']) != plan['config_sha256']):
        raise ValueError('Existing shard plan differs from requested configuration')
    initial = {int(k) for k in plan['initial_episode_metadata_sha256']}
    assigned = [i for w in plan['workers'] for i in w['episode_indices']]
    if len(set(assigned)) != len(assigned) or set(assigned) & initial or initial | set(assigned) != set(range(manifest['target_episodes'])):
        raise ValueError('Shard ownership overlaps or does not cover every global episode')
    for index in initial:
        if index not in episodes or digest(raw/'episodes'/f'episode_{index:06d}'/'metadata.json') != plan['initial_episode_metadata_sha256'][str(index)]:
            raise ValueError(f'Preexisting episode changed or disappeared: {index}')
    for w in plan['workers']:
        if digest(w['episode_indices_path']) != w['episode_indices_sha256'] or read_json(w['episode_indices_path']) != w['episode_indices']:
            raise ValueError('Assigned episode-index file changed')


def link_or_copy(source, destination):
    try:
        return os.link(source, destination)
    except OSError:
        return shutil.copy2(source, destination)


def publish_episode(source, destination, metadata, existing):
    if destination.exists():
        if metadata != existing:
            raise ValueError(f'Refusing to overwrite different committed episode: {destination}')
        for name in REQUIRED:
            if not os.path.samestat((source/name).stat(), (destination/name).stat()) and digest(source/name) != digest(destination/name):
                raise ValueError(f'Existing canonical contents differ: {destination/name}')
        return False
    staging = destination.parent/f'.incoming_{destination.name}_{uuid.uuid4().hex}'
    try:
        shutil.copytree(source, staging, copy_function=link_or_copy, symlinks=False)
        if read_json(staging/'metadata.json') != metadata:
            raise ValueError(f'Metadata changed while publishing: {source}')
        staging.rename(destination)
        return True
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def merge_workers(raw, manifest, plan, episodes):
    seeds = {m['seed']:i for i, m in episodes.items()}
    published = 0
    for w in plan['workers']:
        worker_manifest = read_json(Path(w['output'])/'manifest.json')
        if worker_manifest is not None:
            expected_config = deepcopy(manifest['config'])
            expected_config.update(gpu=w['gpu'], num_envs=w['num_envs'],
                                   log_dir=str(Path(w['output'])/'simulation'))
            if (worker_manifest.get('episode_indices') != w['episode_indices']
                    or worker_manifest.get('global_target_episodes') != manifest['target_episodes']
                    or worker_manifest.get('pair_schedule') != manifest.get('pair_schedule')
                    or worker_manifest.get('seed') != manifest['seed']
                    or worker_manifest.get('noise_deg') != manifest['noise_deg']
                    or trajectory_profile(worker_manifest) != trajectory_profile(manifest)
                    or worker_manifest.get('config') != expected_config):
                raise ValueError(f'Worker changed collection semantics/config: {w["output"]}')
        allowed = set(w['episode_indices'])
        for source in sorted((Path(w['output'])/'episodes').glob('episode_*')):
            metadata = validate_episode(source, manifest)
            index, seed, attempt = (metadata[k] for k in ('episode_index', 'seed', 'attempt_index'))
            if index not in allowed or attempt <= w['attempt_offset'] or (attempt-w['attempt_offset']) % w['attempt_stride']:
                raise ValueError(f'Worker violated global ID/attempt ownership: {source}')
            if seed in seeds and seeds[seed] != index:
                raise ValueError(f'Worker reused a canonical seed: {seed}')
            destination = raw/'episodes'/source.name
            if publish_episode(source, destination, metadata, episodes.get(index)):
                episodes[index] = metadata; seeds[seed] = index; published += 1
    return published


def worker_command(w, plan, manifest, sim_python):
    if trajectory_profile(plan) != trajectory_profile(manifest):
        raise ValueError('Shard worker trajectory profile differs from canonical collection')
    return [str(sim_python), '-u', str(ROOT/'scripts/collect_manipulation.py'),
            '--task', manifest['task_id'], '--output', w['output'], '--config', plan['config_path'],
            '--episodes', str(manifest['target_episodes']), '--episode-indices', w['episode_indices_path'],
            '--num-envs', str(w['num_envs']), '--gpu', str(w['gpu']), '--seed', str(manifest['seed']),
            '--noise-deg', str(manifest['noise_deg']), '--settle-seconds', str(manifest['settle_seconds']),
            '--trajectory-profile', trajectory_profile(manifest),
            '--attempt-offset', str(w['attempt_offset']), '--attempt-stride', str(w['attempt_stride']),
            '--max-attempts', str(plan['max_attempts'])]


def start_worker(w, plan, manifest, sim_python, utility_python, retry_failed):
    if not w['episode_indices']:
        return
    path = Path(w['receipt']); row = read_json(path)
    command = worker_command(w, plan, manifest, sim_python)
    if row and row['command'] != command:
        # Earlier receipts omitted the switch because legacy was the only
        # planner. Preserve their provenance while monitoring or finishing
        # them; any retry below writes a fresh, explicitly pinned command.
        old_command = list(row['command'])
        without_profile = command.copy()
        index = without_profile.index('--trajectory-profile')
        del without_profile[index:index+2]
        compatible_legacy = (trajectory_profile(manifest) == 'legacy'
            and '--trajectory-profile' not in old_command and old_command == without_profile
            and row.get('status') in ('running', 'failed', 'exited'))
        if not compatible_legacy:
            raise ValueError(f'Existing worker command differs: {path}')
    if row and row.get('status') in ('failed', 'exited'):
        complete = read_json(Path(w['output'])/'progress.json', {}).get('status') == 'complete'
        if row.get('exit_code') == 0 and complete:
            return
        if not retry_failed:
            raise RuntimeError(f'Prior worker failed; inspect before --retry-failed-workers: {path}')
        path.rename(path.with_name(f'{path.stem}.prior_{uuid.uuid4().hex}.json')); row = None
    if row is None:
        write_json(path, dict(status='launching', command=command, cwd=str(ROOT),
                             log=w['log'], created_unix=time.time(),
                             source_files=source_files(manifest['task_id'])))
        subprocess.Popen([str(utility_python), str(Path(__file__).resolve()), 'worker', str(path)],
                         cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)


def worker_state(w):
    if not w['episode_indices']:
        return dict(status='complete', accepted=0)
    receipt = read_json(w['receipt'], {})
    progress = read_json(Path(w['output'])/'progress.json', {})
    result = dict(receipt=receipt, progress=progress, accepted=progress.get('accepted', 0))
    if receipt.get('status') in ('exited', 'failed'):
        count = len(w['episode_indices'])
        complete = (receipt.get('exit_code') == 0 and progress.get('status') == 'complete'
                    and progress.get('accepted') == count and progress.get('target') == count)
        return dict(result, status='complete' if complete else 'failed')
    if receipt.get('status') == 'launching':
        return dict(result, status='running' if time.time()-receipt['created_unix'] < 60 else 'failed')
    alive = (receipt.get('status') == 'running' and receipt.get('supervisor_start_ticks') is not None
             and process_identity(receipt.get('supervisor_pid')) == receipt['supervisor_start_ticks'])
    return dict(result, status='running' if alive else 'failed')


def pair_counts(episodes):
    return dict(Counter('+'.join(sorted(m['reset_spec']['objects'])) for m in episodes.values()))


def run(args):
    raw = args.canonical.resolve(); started = time.time()
    if not (raw/'manifest.json').is_file():
        raise ValueError('An existing canonical collection manifest is required')
    with (raw/'.collection.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = read_json(raw/'manifest.json'); episodes = scan_episodes(raw, manifest)
        plan_path = raw/'collection_shards.json'
        plan = read_json(plan_path)
        try:
            if plan is None:
                plan = make_plan(raw, manifest, episodes, args.gpus, args.num_envs, args.max_attempts)
                write_json(plan_path, plan)
            validate_plan(plan, raw, manifest, episodes, args.gpus, args.num_envs, args.max_attempts)
            # Retain the original plan source identity and append each supervisor
            # restart's current source hashes; never rewrite previous provenance.
            write_json(raw/'shards'/f'runner_{int(time.time())}_{os.getpid()}.json',
                       dict(pid=os.getpid(), start_ticks=process_identity(os.getpid()), source_files=source_files(manifest['task_id']),
                            created_unix=time.time(), plan_sha256=digest(plan_path)))
            for w in plan['workers']:
                start_worker(w, plan, manifest, args.sim_python, args.utility_python, args.retry_failed_workers)
            while True:
                merge_workers(raw, manifest, plan, episodes)
                states = {str(w['index']):worker_state(w) for w in plan['workers']}
                running = any(s['status']=='running' for s in states.values())
                failed = any(s['status']=='failed' for s in states.values())
                complete = len(episodes) == manifest['target_episodes'] and all(s['status']=='complete' for s in states.values())
                if complete:
                    if set(episodes) != set(range(manifest['target_episodes'])):
                        raise ValueError('Canonical dataset has an incomplete global episode sequence')
                    if manifest['task_id']=='tabletop_cleanup':
                        expected = Counter('+'.join(sorted(PAIRS[i])) for i in manifest['pair_schedule'])
                        if Counter(pair_counts(episodes)) != expected:
                            raise ValueError('Final canonical object-pair balance differs from global schedule')
                attempts = max([plan['max_previous_attempt']]+[s.get('progress', {}).get('attempts', 0) for s in states.values()])
                progress = dict(task_id=manifest['task_id'], status='complete' if complete else ('running' if running else 'failed'),
                                accepted=len(episodes), target=manifest['target_episodes'], attempts=attempts,
                                session_elapsed_s=time.time()-started, updated_unix=time.time(),
                                stage='sharded_collection', runner_pid=os.getpid(), workers=states,
                                pair_counts=pair_counts(episodes), shard_plan=str(plan_path))
                write_json(raw/'progress.json', progress)
                print(json.dumps({k:v for k,v in progress.items() if k != 'workers'}), flush=True)
                if complete:
                    write_json(raw/'shards/completion.json', dict(success=True, episodes=len(episodes),
                               pair_counts=pair_counts(episodes), completed_unix=time.time(),
                               canonical_manifest_sha256=digest(raw/'manifest.json'), plan_sha256=digest(plan_path)))
                    return 0
                if not running:
                    raise RuntimeError(f'Shard collection stopped incomplete: {len(episodes)}/{manifest["target_episodes"]}; failed={failed}')
                time.sleep(args.poll_seconds)
        except BaseException:
            prior = read_json(raw/'progress.json', {})
            prior.update(status='failed', accepted=len(episodes), target=manifest['target_episodes'],
                         error=traceback.format_exc(), updated_unix=time.time())
            write_json(raw/'progress.json', prior)
            raise


def expand_worker_gpus(gpus, workers_per_gpu=1):
    """Validate physical devices first, then allocate isolated workers per GPU."""
    if (not isinstance(gpus, (list, tuple)) or not gpus
            or any(type(gpu) is not int or gpu < 0 for gpu in gpus)
            or len(set(gpus)) != len(gpus)):
        raise ValueError('Select distinct, nonnegative GPU indices')
    if type(workers_per_gpu) is not int or workers_per_gpu < 1:
        raise ValueError('Workers per GPU must be a positive integer')
    return [gpu for gpu in gpus for _ in range(workers_per_gpu)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--canonical', type=Path, required=True)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--workers-per-gpu', type=int, default=1,
                        help='Independent collector processes per GPU; each retains disjoint episode IDs and attempt seeds')
    parser.add_argument('--num-envs', type=int, default=16)
    parser.add_argument('--max-attempts', type=int, default=5000)
    parser.add_argument('--poll-seconds', type=float, default=10.)
    parser.add_argument('--sim-python', type=Path, default=ROOT/'.conda/envs/pressb/bin/python')
    parser.add_argument('--utility-python', type=Path, default=ROOT/'.conda/envs/vlajepa-inference/bin/python')
    parser.add_argument('--retry-failed-workers', action='store_true')
    args = parser.parse_args(argv)
    try:
        args.gpus = expand_worker_gpus(args.gpus, args.workers_per_gpu)
    except ValueError as exc:
        parser.error(str(exc))
    if min(args.num_envs, args.max_attempts, args.poll_seconds) <= 0:
        parser.error('Counts and polling interval must be positive')
    return run(args)


if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1]=='worker':
        raise SystemExit(_util.worker(sys.argv[2]))
    raise SystemExit(main())
