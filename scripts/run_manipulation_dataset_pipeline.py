#!/usr/bin/env python3
"""Collect two datasets, audit/export, make videos, and optionally launch SFT.

Run this process under nohup or start_new_session=True. Child supervisors keep
their own exit receipts if the orchestrator is interrupted. Existing collectors
are observed through their output lock; this script never signals them.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
TASKS = ('cube_stack', 'tabletop_cleanup')


def read_json(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else default


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def process_identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except (FileNotFoundError, PermissionError, IndexError):
        return None


def worker(receipt_path):
    """A detached supervisor writes the exit result even if its parent exits."""
    row = read_json(receipt_path)
    row.update(status='running', supervisor_pid=os.getpid(),
               supervisor_start_ticks=process_identity(os.getpid()), started_unix=time.time())
    write_json(receipt_path, row)
    try:
        with open(row['log'], 'a', buffering=1) as log:
            process = subprocess.Popen(row['command'], cwd=row['cwd'], stdout=log,
                                       stderr=subprocess.STDOUT)
            row.update(child_pid=process.pid, child_start_ticks=process_identity(process.pid))
            write_json(receipt_path, row)
            code = process.wait()
        row.update(status='exited', exit_code=code, finished_unix=time.time())
    except Exception:
        row.update(status='failed', exit_code=2, error=traceback.format_exc(), finished_unix=time.time())
    write_json(receipt_path, row)
    return 0


def lock_held(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(stream, fcntl.LOCK_UN)
    return False


def external_collector(output, raw, task):
    """Recognize an already launched collector without trusting a reused PID."""
    row = read_json(output/f'{task}_collector_receipt.json', {})
    pid = row.get('pid')
    identity = process_identity(pid)
    if identity is None:
        return None
    if row.get('start_ticks') is not None and row['start_ticks'] != identity:
        return None
    try:
        command = [p.decode() for p in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if p]
        destination = Path(command[command.index('--output')+1])
        if not destination.is_absolute():
            destination = Path(f'/proc/{pid}/cwd').resolve()/destination
        if (destination.resolve() != raw.resolve() or '--task' not in command
                or command[command.index('--task')+1] != task
                or not any(p.endswith('collect_manipulation.py') for p in command)):
            return None
    except (FileNotFoundError, PermissionError, ValueError, IndexError):
        return None
    return dict(pid=pid, start_ticks=identity, command=command, log=row.get('log'))


def raw_complete(raw, task, count=500):
    progress = read_json(raw/'progress.json', {})
    if progress.get('status') != 'complete':
        return False
    manifest = read_json(raw/'manifest.json', {})
    if (manifest.get('task_id') != task or manifest.get('target_episodes') != count
            or progress.get('accepted') != count or progress.get('target') != count):
        raise ValueError(f'Collection completion metadata disagree: {raw}')
    directories = sorted((raw/'episodes').glob('episode_*'))
    if [p.name for p in directories] != [f'episode_{i:06d}' for i in range(count)]:
        raise ValueError(f'Committed episode indices are incomplete: {raw}')
    for index, path in enumerate(directories):
        metadata = read_json(path/'metadata.json', {})
        if (metadata.get('success') is not True or metadata.get('audit', {}).get('passed') is not True
                or metadata.get('episode_index') != index or metadata.get('task_id') != task
                or any(not (path/name).is_file() for name in ('data.npz', 'up.mp4', 'wrist.mp4'))):
            raise ValueError(f'Committed episode lacks physical acceptance or files: {path}')
    return True


def export_complete(dataset, task, count=500):
    audit = read_json(dataset/'meta/audit.json', {})
    if not audit:
        return False
    info = read_json(dataset/'meta/info.json', {})
    manifest = read_json(dataset/'meta/export_manifest.json', {})
    frames = info.get('total_frames')
    if (audit.get('success') is not True or audit.get('full_video_decode') is not True
            or type(frames) is not int or frames < count*2
            or audit.get('episodes') != count or info.get('total_episodes') != count
            or audit.get('numeric_rows_compared') != frames or audit.get('frames') != frames
            or manifest.get('task_id') != task or info.get('fps') != 30
            or len(manifest.get('episodes', [])) != count
            or any(e.get('success') is not True for e in manifest.get('episodes', []))
            or info.get('codebase_version') != 'v3.0'
            or any(audit.get('decoded_frames', {}).get(f'observation.images.{v}') != frames
                   for v in ('up', 'wrist'))):
        raise ValueError(f'Export audit failed: {dataset}')
    return True


class Pipeline:
    def __init__(self, args):
        self.args = args
        self.output = args.output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.status_path = self.output/'pipeline_status.json'
        self.receipts = self.output/'pipeline_receipts'
        self.receipts.mkdir(exist_ok=True)
        self.logs = self.output/'pipeline_logs'
        self.logs.mkdir(exist_ok=True)
        self.mutex = threading.RLock()
        self.stop = threading.Event()
        self.state = read_json(self.status_path, {})
        self.state.update(status='running', pid=os.getpid(), start_ticks=process_identity(os.getpid()),
                          tasks=self.state.get('tasks', {}), updated_unix=time.time())
        self.gpus = {'cube_stack': args.cube_gpu, 'tabletop_cleanup': args.cleanup_gpu}
        self.envs = {'cube_stack': args.cube_num_envs, 'tabletop_cleanup': args.cleanup_num_envs}
        self.gpu_locks = {gpu: threading.Lock() for gpu in set(self.gpus.values())}
        self.export_lock = threading.Lock()
        self.flush()

    def flush(self):
        with self.mutex:
            self.state['updated_unix'] = time.time()
            write_json(self.status_path, self.state)

    def update(self, task=None, **values):
        with self.mutex:
            target = self.state if task is None else self.state['tasks'].setdefault(task, {})
            target.update(values)
            self.flush()

    def pause(self, seconds=None):
        if self.stop.wait(self.args.poll_seconds if seconds is None else seconds):
            raise RuntimeError('Another pipeline task failed; no further stages launched')

    def run_command(self, label, command):
        """Start or adopt only this pipeline's recorded child supervisor."""
        path = self.receipts/f'{label}.json'
        row = read_json(path)
        if row is not None and row['command'] != command:
            # A retry changes only the runner's permission to restart its
            # failed workers, not the persisted shard plan or collection.
            runner = str(ROOT/'scripts/run_sharded_manipulation_collection.py')
            retry_only = (runner in row['command'] and runner in command
                          and [v for v in row['command'] if v != '--retry-failed-workers']
                          == [v for v in command if v != '--retry-failed-workers'])
            if not retry_only:
                raise ValueError(f'Existing child command differs: {path}')
        if row and row.get('status') in ('exited', 'failed'):
            if row.get('exit_code') == 0 or not self.args.retry_failed:
                return row.get('exit_code', 2)
            backup = path.with_name(f'{label}.{uuid.uuid4().hex}.json')
            path.rename(backup)
            row = None
        if row is None:
            row = dict(status='launching', command=command, cwd=str(ROOT),
                       log=str(self.logs/f'{label}.log'), created_unix=time.time())
            write_json(path, row)
            subprocess.Popen([str(self.args.utility_python), str(Path(__file__).resolve()),
                              'worker', str(path)], cwd=ROOT, start_new_session=True,
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        while True:
            row = read_json(path)
            collecting = next((task for task in TASKS if label == f'{task}_collect'), None)
            if collecting is not None:
                self.update(collecting, collection_progress=read_json(
                    self.output/f'{collecting}_raw/progress.json', {}))
            if row.get('status') in ('exited', 'failed'):
                return row.get('exit_code', 2)
            if row.get('status') == 'launching':
                if time.time()-row['created_unix'] > 60:
                    raise RuntimeError(f'Child supervisor did not start: {path}')
            elif process_identity(row.get('supervisor_pid')) != row.get('supervisor_start_ticks'):
                raise RuntimeError(f'Child supervisor disappeared without exit receipt: {path}')
            self.pause()

    def validate_existing(self, raw, task):
        manifest = read_json(raw/'manifest.json')
        if manifest is None:
            return
        expected = dict(task_id=task, target_episodes=self.args.episodes, seed=self.args.seed,
                        noise_deg=self.args.noise_deg)
        if any(manifest.get(k) != v for k, v in expected.items()):
            raise ValueError(f'Existing collection settings differ: {raw}')
        config = manifest.get('config', {})
        if config.get('gpu') != self.gpus[task] or config.get('num_envs') != self.envs[task]:
            raise ValueError(f'Existing GPU/environment settings differ: {raw}')
        if manifest.get('trajectory_profile', 'legacy') != self.args.trajectory_profile:
            raise ValueError(f'Existing trajectory profile differs: {raw}')
        if self.args.collection_config:
            overrides = read_json(self.args.collection_config)
            overrides = overrides.get('backend', {}).get('config', overrides)
            # These values are fixed by the collector or explicit CLI flags.
            forced = {'num_envs', 'gpu', 'horizon', 'observation_render_frames',
                      'reset_render_frames', 'max_episode_seconds', 'terminate_on_success', 'log_dir'}
            def matches(actual, requested, path=()):
                if not isinstance(requested, dict):
                    return actual == requested
                if not isinstance(actual, dict):
                    return False
                return all(matches(actual.get(key), value, path+(key,))
                           for key, value in requested.items()
                           if not (not path and key in forced)
                           and path+(key,) != ('scene_overrides', 'rendering', 'warmup_frames'))
            if not matches(config, overrides):
                raise ValueError(f'Existing collection config differs: {raw}')
        plan = read_json(raw/'collection_shards.json')
        if plan is not None:
            workers = plan.get('workers', [])
            if (len(workers) != self.args.workers_per_gpu
                    or any(w.get('gpu') != self.gpus[task] or w.get('num_envs') != self.envs[task]
                           for w in workers)):
                raise ValueError(f'Existing shard worker settings differ: {raw}')

    def collector_command(self, task, raw):
        command = [str(self.args.sim_python), '-u', str(ROOT/'scripts/collect_manipulation.py'),
                   '--task', task, '--output', str(raw), '--episodes', str(self.args.episodes),
                   '--num-envs', str(self.envs[task]), '--gpu', str(self.gpus[task]),
                   '--seed', str(self.args.seed), '--noise-deg', str(self.args.noise_deg),
                   '--max-attempts', str(self.args.max_attempts)]
        # Retain historical default command receipts; compact is the collector default.
        if self.args.trajectory_profile != 'compact':
            command += ['--trajectory-profile', self.args.trajectory_profile]
        if self.args.collection_config:
            command += ['--config', str(self.args.collection_config)]
        return command

    def collect(self, task, raw):
        with self.gpu_locks[self.gpus[task]]:
            self.validate_existing(raw, task)
            if raw_complete(raw, task, self.args.episodes):
                return
            observed = False
            while lock_held(raw/'.collection.lock') or external_collector(self.output, raw, task):
                observed = True
                self.validate_existing(raw, task)
                self.update(task, stage='waiting_existing_collector', raw=str(raw),
                            existing_collector=external_collector(self.output, raw, task),
                            collection_progress=read_json(raw/'progress.json', {}))
                self.pause()
            if raw_complete(raw, task, self.args.episodes):
                return
            progress = read_json(raw/'progress.json', {})
            if (observed or (progress and progress.get('status') != 'initialized')) and not self.args.retry_failed:
                raise RuntimeError(f'Existing collector stopped incomplete: {raw}; inspect before --retry-failed')
            self.update(task, stage='collecting', raw=str(raw), gpu=self.gpus[task], num_envs=self.envs[task],
                        workers_per_gpu=self.args.workers_per_gpu, target_episodes=self.args.episodes)
            command = self.collector_command(task, raw)
            if self.args.workers_per_gpu > 1:
                if not (raw/'manifest.json').is_file():
                    code = self.run_command(f'{task}_initialize', command+['--initialize-only'])
                    if code or not (raw/'manifest.json').is_file():
                        raise RuntimeError(f'{task} manifest initialization failed with exit {code}')
                    self.validate_existing(raw, task)
                command = [str(self.args.utility_python), '-u',
                           str(ROOT/'scripts/run_sharded_manipulation_collection.py'),
                           '--canonical', str(raw), '--gpus', str(self.gpus[task]),
                           '--workers-per-gpu', str(self.args.workers_per_gpu),
                           '--num-envs', str(self.envs[task]), '--max-attempts', str(self.args.max_attempts),
                           '--poll-seconds', str(self.args.poll_seconds),
                           '--sim-python', str(self.args.sim_python), '--utility-python', str(self.args.utility_python)]
                if self.args.retry_failed:
                    command += ['--retry-failed-workers']
            code = self.run_command(f'{task}_collect', command)
            if code or not raw_complete(raw, task, self.args.episodes):
                raise RuntimeError(f'{task} collector exit {code}; physical {self.args.episodes}-episode completion gate did not pass')

    def task(self, task):
        raw = self.output/f'{task}_raw'
        dataset = self.args.dataset_root/f'pressb_{task}_{self.args.episodes}_{self.args.dataset_suffix}'
        examples = self.output/f'{task}_examples'
        self.collect(task, raw)
        self.update(task, stage='raw_complete', accepted=self.args.episodes)
        # Export is CPU-bound and memory-heavy. Let the other GPU collect while
        # serializing two exporters, each of which internally audits four videos.
        with self.export_lock:
            if not export_complete(dataset, task, self.args.episodes):
                self.update(task, stage='exporting', dataset=str(dataset))
                code = self.run_command(f'{task}_export', [str(self.args.export_python), '-u',
                    str(ROOT/'scripts/export_manipulation_dataset.py'), '--raw', str(raw),
                    '--output', str(dataset), '--episodes', str(self.args.episodes)])
                if code or not export_complete(dataset, task, self.args.episodes):
                    raise RuntimeError(f'{task} export failed with exit {code}')
        self.update(task, stage='generating_examples', examples=str(examples))
        summary = read_json(examples/'summary.json', {})
        if (summary.get('episodes') != self.args.episodes or summary.get('task_id') != task
                or not (examples/'examples_1x.mp4').is_file()):
            code = self.run_command(f'{task}_examples', [str(self.args.utility_python), '-u',
                str(ROOT/'scripts/preview_manipulation_dataset.py'), '--raw', str(raw), '--output', str(examples)])
            summary = read_json(examples/'summary.json', {})
            if (code or summary.get('episodes') != self.args.episodes or summary.get('task_id') != task
                    or not (examples/'examples_1x.mp4').is_file()):
                raise RuntimeError(f'{task} example generation failed with exit {code}')
        self.update(task, stage='ready_for_sft', dataset=str(dataset), video=str(examples/'examples_1x.mp4'),
                    summary=summary)

    def deploy(self):
        deployment = self.args.config_dir/'deployment_status.json'
        while True:
            previous = int(self.state.get('deployment_attempts', 0))
            receipt = read_json(self.receipts/f'deploy_{previous:05d}.json', {})
            # Adopt an in-flight transfer after orchestrator restart. Two rsync
            # invocations must not update the same remote staging directory.
            attempt = previous if receipt.get('status') in ('launching', 'running') else previous+1
            self.update(stage='deploying_sft', deployment_attempts=attempt)
            command = [str(self.args.utility_python), '-u', str(ROOT/'scripts/launch_manipulation_sft.py'),
                       '--launch', '--host', self.args.host, '--dataset-root', str(self.args.dataset_root),
                       '--config-dir', str(self.args.config_dir), '--status', str(deployment)]
            code = self.run_command(f'deploy_{attempt:05d}', command)
            self.update(deployment=read_json(deployment, {}))
            if code == 0:
                if read_json(deployment, {}).get('status') != 'jobs_started':
                    raise RuntimeError('SFT launcher exited zero without jobs_started receipt')
                self.update(status='completed', stage='both_sft_jobs_started', finished_unix=time.time())
                return
            if code != 75:
                raise RuntimeError(f'SFT deployment failed with exit {code}; inspect {deployment}')
            self.update(stage='waiting_h200', retry_after_seconds=self.args.retry_seconds)
            self.pause(self.args.retry_seconds)

    def run(self):
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = {pool.submit(self.task, task): task for task in TASKS}
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception:
                        self.stop.set()
                        raise
            if self.args.prepare_only:
                self.update(status='completed', stage='datasets_and_videos_ready', finished_unix=time.time())
            else:
                self.deploy()
        except Exception:
            self.update(status='failed', error=traceback.format_exc())
            raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/task_datasets_20261110')
    parser.add_argument('--dataset-root', type=Path, default=Path('/home/pengguanqi/Datasets'))
    parser.add_argument('--dataset-suffix', default='20261110')
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--config-dir', type=Path)
    parser.add_argument('--cube-gpu', type=int, default=1)
    parser.add_argument('--cleanup-gpu', type=int, default=1)
    parser.add_argument('--num-envs', type=int, default=16)
    parser.add_argument('--workers-per-gpu', type=int, default=1,
                        help='Isolated collector processes per task GPU; num-envs is per process')
    parser.add_argument('--collection-config', type=Path, help='Optional JSON overrides for both task collectors')
    parser.add_argument('--trajectory-profile', choices=['compact', 'legacy'], default='compact')
    parser.add_argument('--cube-num-envs', type=int)
    parser.add_argument('--cleanup-num-envs', type=int)
    parser.add_argument('--seed', type=int, default=20261110)
    parser.add_argument('--noise-deg', type=float, default=.12)
    parser.add_argument('--max-attempts', type=int, default=5000)
    parser.add_argument('--poll-seconds', type=float, default=10.)
    parser.add_argument('--retry-seconds', type=float, default=120.)
    parser.add_argument('--host', default='h200')
    parser.add_argument('--sim-python', type=Path, default=ROOT/'.conda/envs/pressb/bin/python')
    parser.add_argument('--export-python', type=Path, default=ROOT/'.conda/envs/lerobot/bin/python')
    parser.add_argument('--utility-python', type=Path, default=ROOT/'.conda/envs/vlajepa-inference/bin/python')
    parser.add_argument('--prepare-only', action='store_true', help='Collect/export/video only; omit to launch SFT too')
    parser.add_argument('--retry-failed', action='store_true', help='Explicitly retry incomplete/failed prior stages after inspection')
    args = parser.parse_args(argv)
    args.output = args.output.resolve(); args.dataset_root = args.dataset_root.resolve()
    args.config_dir = (args.config_dir or args.output/'training_prepare').resolve()
    args.cube_num_envs = args.num_envs if args.cube_num_envs is None else args.cube_num_envs
    args.cleanup_num_envs = args.num_envs if args.cleanup_num_envs is None else args.cleanup_num_envs
    values = (args.episodes, args.workers_per_gpu, args.num_envs, args.cube_num_envs,
              args.cleanup_num_envs, args.max_attempts, args.poll_seconds, args.retry_seconds)
    if any(not math.isfinite(value) or value <= 0 for value in values):
        parser.error('Counts and intervals must be positive')
    if min(args.cube_gpu, args.cleanup_gpu) < 0:
        parser.error('GPU indices must be nonnegative')
    if not math.isfinite(args.noise_deg) or args.noise_deg < 0:
        parser.error('Noise amplitude must be finite and nonnegative')
    if not args.dataset_suffix or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.dataset_suffix):
        parser.error('Dataset suffix must contain only letters, digits, underscores or hyphens')
    if (args.episodes != 500 or args.dataset_suffix != '20261110') and not args.prepare_only:
        parser.error('Custom episodes/dataset suffix require --prepare-only; the SFT launcher expects 500_20261110')
    if args.collection_config:
        args.collection_config = args.collection_config.resolve()
        try:
            config = json.loads(args.collection_config.read_text())
            if not isinstance(config, dict) or not isinstance(config.get('backend', {}), dict):
                raise ValueError('Expected a JSON object')
            if not isinstance(config.get('backend', {}).get('config', config), dict):
                raise ValueError('Collection backend config must be a JSON object')
        except (OSError, ValueError) as exc:
            parser.error(f'Cannot read collection config: {exc}')
    return args


def main(argv=None):
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'.pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        Pipeline(args).run()
    return 0


if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1]=='worker':
        raise SystemExit(worker(sys.argv[2]))
    raise SystemExit(main())
