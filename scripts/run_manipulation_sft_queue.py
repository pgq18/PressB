#!/usr/bin/env python3
"""Sequential, audited SFT with optional verified working-copy rotation.

The collection/export pipeline must run with --prepare-only. This queue owns SFT
launches. Archive/release flags are opt-in: local datasets and complete model
archives remain durable; only matching remote working copies are removed.
"""
from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
TASKS = ('cube_stack', 'tabletop_cleanup')
SSH_OPTIONS = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', '-o',
               'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3']
ARCHIVE_MARKER = 'remote_payload_archive.json'


class RetryLater(RuntimeError):
    pass


def read_json(path, default=None):
    p = Path(path)
    return json.loads(p.read_text()) if p.is_file() else default


def write_json(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_name(p.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(p)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def tree_manifest(root, *, ignore=()):
    """Hash regular files without following links; reject external/broken links."""
    root = Path(root).resolve(strict=True)
    files, links = {}, {}
    for directory, dirs, names in os.walk(root, followlinks=False):
        for name in sorted(dirs + names):
            p = Path(directory) / name
            relative = p.relative_to(root).as_posix()
            if relative in ignore:
                continue
            if p.is_symlink():
                target = os.readlink(p)
                if Path(target).is_absolute() or not p.resolve(strict=True).is_relative_to(root):
                    raise ValueError(f'Unsafe archive symlink: {p}')
                links[relative] = target
            elif p.is_file():
                before = p.stat(); sha = digest(p); after = p.stat()
                if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                    raise RuntimeError(f'File changed while hashing: {p}')
                files[relative] = dict(bytes=after.st_size, sha256=sha)
            elif not p.is_dir():
                raise ValueError(f'Unsupported archive entry: {p}')
    return dict(files=files, symlinks=links)


def process_identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return fields[19] if fields[0] != 'Z' else None
    except (FileNotFoundError, PermissionError, IndexError):
        return None


def ensure_no_users(root):
    """Refuse removal while a same-user process opens, runs in, or names root."""
    root = Path(root).resolve(strict=True)
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            if proc.stat().st_uid != os.getuid() or process_identity(int(proc.name)) is None:
                continue
            command = (proc/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            if str(root) in command or (proc/'cwd').resolve().is_relative_to(root):
                raise RetryLater(f'Process {proc.name} still uses {root}')
            for fd in (proc/'fd').iterdir():
                try:
                    if fd.resolve().is_relative_to(root):
                        raise RetryLater(f'Process {proc.name} has an open file below {root}')
                except (FileNotFoundError, PermissionError, RuntimeError) as error:
                    if isinstance(error, RetryLater):
                        raise
        except (FileNotFoundError, PermissionError):
            continue


def checked_run(request):
    root = Path(request['remote_root']).resolve()
    task = request['task']
    if task not in TASKS:
        raise ValueError('Unknown task')
    receipt = read_json(root/'outputs/launch_receipts'/f'{task}.json')
    if receipt is None:
        return root, None, None
    run = Path(receipt['run']).resolve()
    if run.parent != root/'outputs/model_training' or receipt.get('task') != task:
        raise ValueError('Receipt run lies outside this deployment')
    return root, receipt, run


def remote_status(request):
    root, row, run = checked_run(request)
    if row is None:
        return dict(status='not_started', complete=False)
    expected_world = request.get('gpus_per_task', 2)
    if row.get('world_size', 1) != expected_world:
        raise ValueError('Existing job uses a different GPU count')
    if request.get('per_device_batch_size') is not None and row.get('per_device_batch_size') != request['per_device_batch_size']:
        raise ValueError('Existing job uses a different per-device batch')
    if request.get('gpu_indices') is not None and row.get('gpu_indices') != request['gpu_indices']:
        raise ValueError('Existing job uses a different GPU allocation')
    if row.get('shared_gpu_min_free_mib') != request.get('shared_gpu_min_free_mib'):
        raise ValueError('Existing job uses a different shared GPU admission policy')
    launch = row.get('launch_settings', {})
    for key, expected in (('gpus_per_task', expected_world), ('per_device_batch_size', request.get('per_device_batch_size')),
                          ('gpu_indices', request.get('gpu_indices')), ('run_suffix', request.get('run_suffix', ''))):
        if key in launch and expected is not None and launch[key] != expected:
            raise ValueError(f'Existing job launch settings have different {key}')
    suffix = request.get('run_suffix', '')
    if run.name != f'vlajepa_{request["task"]}_500_20261110'+('_'+suffix if suffix else ''):
        raise ValueError('Existing job uses a different run identity')
    state = row.get('status')
    if state in ('failed', 'cancelled', 'stopped', 'superseded'):
        raise RuntimeError(f'{request["task"]} is {state}; no automatic restart: {row.get("log")}')
    current = read_json(run/'status.json', {})
    complete = read_json(run/'completed.json', {})
    if state != 'completed':
        if state == 'launching' and time.time()-row.get('created_unix', 0) < 60:
            return dict(status=state, complete=False, receipt=row, training_status=current)
        pid = row.get('supervisor_pid')
        if process_identity(pid) != row.get('supervisor_start_ticks') or not row.get('supervisor_start_ticks'):
            raise RuntimeError('SFT supervisor disappeared before full completion')
        return dict(status=state, complete=False, receipt=row, training_status=current)
    expected_world = request.get('gpus_per_task', 2)
    geometry = read_json(run/'initialization.json', {})
    if (row.get('exit_code') != 0 or complete.get('success') is not True
            or complete.get('world_size') != expected_world or complete.get('probe')
            or current.get('status') != 'completed'
            or complete.get('step', 0) < 1 or current.get('step') != complete['step']
            or complete.get('step') != geometry.get('target_steps')):
        raise RuntimeError('Completed receipt lacks matching full-training completion evidence')
    return dict(status='completed', complete=True, receipt=row, training_status=current, completed=complete)


def require_completed(request):
    state = remote_status(request)
    if not state['complete']:
        raise RetryLater('Task training has not completed')
    _, _, run = checked_run(request)
    ensure_no_users(run)
    return state, run


def remote_run_inventory(request):
    state, run = require_completed(request)
    return dict(run=str(run), manifest=tree_manifest(run, ignore=(ARCHIVE_MARKER,)),
                marker=read_json(run/ARCHIVE_MARKER), completed=state['completed'])


def remove_checkpoint_payloads(request):
    """Idempotently remove only model.pt files whose full local archive was verified."""
    _, run = require_completed(request)
    expected = request['manifest']
    digest_expected = fingerprint(expected)
    if request.get('verified_local_manifest_sha256') != digest_expected:
        raise ValueError('Verified local archive digest is missing or different')
    payloads = [p for p in expected['files']
                if Path(p).name == 'model.pt' and len(Path(p).parts) >= 3 and Path(p).parts[0] == 'checkpoints']
    if not payloads:
        raise ValueError('No checkpoint payloads in verified archive')
    marker_path = run/ARCHIVE_MARKER
    marker = read_json(marker_path)
    current = tree_manifest(run, ignore=(ARCHIVE_MARKER,))
    if marker:
        if marker.get('manifest_sha256') != digest_expected or marker.get('payloads') != payloads:
            raise ValueError('Existing archive marker belongs to a different backup')
        missing = set(expected['files']) - set(current['files'])
        if not missing.issubset(payloads):
            raise ValueError('Non-payload files disappeared from remote run')
        trimmed = dict(files={k:v for k,v in expected['files'].items() if k not in missing}, symlinks=expected['symlinks'])
        if current != trimmed:
            raise ValueError('Remote run changed since archive verification')
    elif current != expected:
        raise ValueError('Remote run changed since archive verification')
    # Persist the permitted list before unlinking, so interruption is resumable.
    marker = dict(status='removing', archive_path=request['archive_path'], manifest_sha256=digest_expected,
                  payloads=payloads, updated_unix=time.time())
    write_json(marker_path, marker)
    for relative in payloads:
        p = run/relative
        if p.exists():
            if p.is_symlink() or p.resolve().parent.parent != run/'checkpoints':
                raise ValueError('Checkpoint payload has unexpected nesting or link')
            if p.stat().st_size != expected['files'][relative]['bytes'] or digest(p) != expected['files'][relative]['sha256']:
                raise ValueError('Checkpoint changed immediately before removal')
            p.unlink()
    marker.update(status='archived', updated_unix=time.time())
    write_json(marker_path, marker)
    return marker


def checked_dataset(request):
    task = request['task']
    root = Path(request['remote_dataset_root']).resolve(strict=True)
    p = Path(request['remote_dataset_path'])
    if task not in TASKS or p.name != f'pressb_{task}_500_20261110' or p.parent.resolve() != root or p.is_symlink():
        raise ValueError('Dataset removal restricted to exact task working directory')
    if p.exists() and p.resolve().parent != root:
        raise ValueError('Remote dataset escapes dataset root')
    return p


def remote_dataset_inventory(request):
    require_completed(request)
    p = checked_dataset(request)
    marker = read_json(Path(request['remote_root'])/'outputs/launch_receipts'/f'{request["task"]}.dataset_archived.json')
    if not p.exists():
        return dict(missing=True, marker=marker)
    ensure_no_users(p)
    manifest = tree_manifest(p)
    if manifest['symlinks']:
        raise ValueError('Dataset must contain regular files only')
    return dict(missing=False, path=str(p), manifest=manifest, marker=marker)


def release_dataset(request):
    _, receipt, run = checked_run(request)
    require_completed(request)
    p = checked_dataset(request)
    expected = request['manifest']; expected_sha = fingerprint(expected)
    if request.get('verified_local_manifest_sha256') != expected_sha or expected['symlinks']:
        raise ValueError('Verified local dataset digest is missing or different')
    marker_path = Path(request['remote_root'])/'outputs/launch_receipts'/f'{request["task"]}.dataset_archived.json'
    marker = read_json(marker_path)
    if marker and marker.get('manifest_sha256') != expected_sha:
        raise ValueError('Dataset release marker belongs to different local data')
    if not p.exists():
        if marker and marker.get('status') in ('removing','archived'):
            marker.update(status='archived'); write_json(marker_path, marker); return marker
        raise ValueError('Dataset disappeared without verified release receipt')
    ensure_no_users(p)
    current = tree_manifest(p)
    if marker:
        if current['symlinks'] or any(expected['files'].get(k) != v for k,v in current['files'].items()):
            raise ValueError('Remote dataset changed during release')
    elif current != expected:
        raise ValueError('Remote dataset differs from verified local copy')
    marker = dict(status='removing', local_path=request['local_path'], manifest_sha256=expected_sha,
                  remote_path=str(p), files=len(expected['files']), updated_unix=time.time())
    write_json(marker_path, marker)
    # Every file is accounted for by the verified manifest; remove no unrelated paths.
    for relative in expected['files']:
        target = p/relative
        if target.exists():
            if target.is_symlink() or not target.resolve().is_relative_to(p.resolve()):
                raise ValueError('Unsafe dataset entry during release')
            target.unlink()
    for directory, dirs, files in os.walk(p, topdown=False):
        Path(directory).rmdir()  # Unexpected new content prevents final removal.
    marker.update(status='archived', updated_unix=time.time()); write_json(marker_path, marker)
    return marker


def remote_call(args, mode, request):
    encoded = base64.b64encode(json.dumps(request).encode()).decode()
    command = ['ssh', *SSH_OPTIONS, args.host, shlex.join(['python3', '-', 'remote', mode, encoded])]
    result = subprocess.run(command, input=Path(__file__).read_text(), capture_output=True, text=True, timeout=7200)
    if result.returncode in (75,255):
        raise RetryLater(result.stderr[-2000:] or result.stdout[-2000:])
    if result.returncode:
        raise RuntimeError(result.stderr[-6000:] or result.stdout[-6000:])
    return json.loads(result.stdout.strip().splitlines()[-1])


def launch_worker(receipt_path):
    row = read_json(receipt_path)
    row.update(status='running', supervisor_pid=os.getpid(), start_ticks=process_identity(os.getpid()))
    write_json(receipt_path, row)
    with open(row['log'], 'a') as log:
        process = subprocess.Popen(row['command'], cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        row.update(child_pid=process.pid); write_json(receipt_path, row)
        code = process.wait()
    row.update(status='exited', exit_code=code, finished_unix=time.time()); write_json(receipt_path, row)
    return 0


def ensure_worker_live(row, label):
    if row.get('status') == 'launching' and time.time()-row.get('created_unix', 0) <= 60:
        return
    if (row.get('status') != 'running' or not row.get('start_ticks')
            or process_identity(row.get('supervisor_pid')) != row.get('start_ticks')):
        raise RuntimeError(f'{label} worker disappeared without a completion receipt')


def archive_worker(receipt_path):
    """Archive-only worker with its own Queue state and completion receipt."""
    row = read_json(receipt_path)
    values = dict(row['args'])
    path_keys = ('remote_dataset_root','dataset_root','config_dir','output','archive_root','python','task_overrides')
    for key in path_keys:
        if values.get(key) is not None:
            values[key] = Path(values[key])
    args = argparse.Namespace(**values)
    if args.archive_completed_remote_payloads or args.release_remote_dataset or args.release_final_remote:
        raise ValueError('Background archives may not remove remote working copies')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'.archive_worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        row.update(status='running', supervisor_pid=os.getpid(), start_ticks=process_identity(os.getpid()), started_unix=time.time())
        write_json(receipt_path, row)
        try:
            worker_queue = Queue(args)
            while True:
                try:
                    worker_queue.archive(row['task']); break
                except (RetryLater, subprocess.TimeoutExpired) as error:
                    worker_queue.update(row['task'], stage='waiting_archive', reason=str(error))
                    time.sleep(args.poll_seconds)
            row.update(status='completed', exit_code=0, finished_unix=time.time())
        except Exception as error:
            row.update(status='failed', exit_code=2, error=str(error), finished_unix=time.time())
        write_json(receipt_path, row)
        return row['exit_code']


class Queue:
    def __init__(self, args):
        self.args = args
        self.output = args.output.resolve(); self.output.mkdir(parents=True, exist_ok=True)
        self.state = read_json(self.output/'status.json', {})
        self.overrides = read_json(args.task_overrides, {}) if args.task_overrides else {}
        validate_overrides(self.overrides)
        if any('shared_gpu_min_free_mib' in values for values in self.overrides.values()) and args.gpu_indices is None:
            raise ValueError('Shared GPU admission requires explicit --gpu-indices')
        contract = {k:str(getattr(args,k)) for k in ('host','remote_root','remote_dataset_root','dataset_root','config_dir',
                    'archive_root','gpus_per_task','per_device_batch_size','run_suffix','gpu_indices','checkpoint_reserve_gib','storage_margin_gib','archive_completed_remote_payloads','release_remote_dataset','release_final_remote','background_archives','prestage_next')}
        contract['task_overrides'] = self.overrides
        if self.state and self.state.get('contract') != contract:
            raise ValueError('Queue settings changed; use a new output directory')
        self.state.update(contract=contract, status='running', pid=os.getpid(), tasks=self.state.get('tasks', {}))
        self.save()

    def save(self):
        self.state['updated_unix'] = time.time(); write_json(self.output/'status.json', self.state)

    def update(self, task, **fields):
        self.state['tasks'].setdefault(task, {}).update(fields); self.save()

    def task_setting(self, task, key):
        return self.overrides.get(task, {}).get(key, getattr(self.args, key, None))

    def request(self, task):
        dataset_root = Path(self.task_setting(task, 'remote_dataset_root'))
        return dict(task=task, remote_root=self.task_setting(task, 'remote_root'), gpus_per_task=self.args.gpus_per_task,
                    gpu_indices=self.args.gpu_indices, per_device_batch_size=self.args.per_device_batch_size, run_suffix=self.args.run_suffix,
                    shared_gpu_min_free_mib=self.task_setting(task, 'shared_gpu_min_free_mib'),
                    remote_dataset_root=str(dataset_root),
                    remote_dataset_path=str(dataset_root/f'pressb_{task}_500_20261110'))

    def launch_command(self, task, *, stage_only=False):
        args = self.args
        command = [str(args.python), '-u', str(ROOT/'scripts/launch_manipulation_sft.py'), '--stage-only' if stage_only else '--launch', '--host', args.host,
                   '--tasks', task, '--dataset-root', str(args.dataset_root), '--remote-dataset-root', str(self.task_setting(task, 'remote_dataset_root')),
                   '--config-dir', str(args.config_dir), '--remote-root', self.task_setting(task, 'remote_root'),
                   '--status', str(self.output/f'{task}.staging.json' if stage_only else self.output/f'{task}.deployment.json'), '--gpus-per-task', str(args.gpus_per_task),
                   '--per-device-batch-size', str(args.per_device_batch_size)]
        if args.run_suffix:
            command += ['--run-suffix', args.run_suffix]
        if args.gpu_indices is not None:
            command += ['--gpu-indices', *map(str,args.gpu_indices)]
        for key in ('checkpoint_reserve_gib', 'storage_margin_gib', 'shared_gpu_min_free_mib'):
            value = self.task_setting(task, key)
            if value is not None:
                command += ['--'+key.replace('_', '-'), str(value)]
        return command

    def prestage(self, task):
        """One nonblocking stage-only tick; adopt a durable transfer across restarts."""
        import launch_manipulation_sft as launcher
        current = self.state['tasks'].get(task, {})
        attempt = current.get('stage_attempt', 0)
        path = self.output/f'{task}.stage_{attempt:04d}.json'
        row = read_json(path)
        command = self.launch_command(task, stage_only=True)
        if row is not None and row.get('command') != command:
            raise ValueError('Existing staging command differs')
        if row and row.get('status') == 'exited':
            if row.get('exit_code') == 0:
                evidence = read_json(self.output/f'{task}.staging.json', {})
                if evidence.get('status') != 'staged':
                    raise RuntimeError('Stage worker exited zero without staged evidence')
                self.update(task, prestage_status='staged', staging=evidence)
                return True
            if row.get('exit_code') != 75:
                raise RuntimeError(f'Stage worker failed; inspect {row["log"]}')
            if time.time()-row.get('finished_unix', 0) < self.args.poll_seconds:
                self.update(task, prestage_status='retry_wait')
                return False
            row = None
        if row is not None:
            ensure_worker_live(row, 'Stage')
            self.update(task, prestage_status='running', staging=read_json(self.output/f'{task}.staging.json', {}))
            return False
        # Audit before spawning so collection/export can progress independently.
        try:
            launcher.audit_gate(self.args.dataset_root/f'pressb_{task}_500_20261110', task)
        except launcher.RetryLater as error:
            self.update(task, prestage_status='waiting_dataset', prestage_reason=str(error))
            return False
        attempt += 1; path = self.output/f'{task}.stage_{attempt:04d}.json'
        row = dict(status='launching', command=command, log=str(path.with_suffix('.log')), created_unix=time.time())
        write_json(path, row)
        self.update(task, stage_attempt=attempt, prestage_status='starting')
        subprocess.Popen([str(self.args.python), str(Path(__file__).resolve()), 'launch-worker', str(path)],
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        return False

    def prestage_next(self, current_task):
        if not self.args.prestage_next or current_task == TASKS[-1]:
            return
        self.prestage(TASKS[TASKS.index(current_task)+1])

    def wait_for_prestage(self, task):
        if not self.args.prestage_next or task == TASKS[0]:
            return
        while not self.prestage(task):
            self.poll_archives()
            self.update(task, stage='waiting_prestage')
            time.sleep(self.args.poll_seconds)

    def archive_worker_arguments(self, task):
        values = {key:str(value) if isinstance(value, Path) else value for key,value in vars(self.args).items()}
        worker_output = self.output/'archive_workers'/task
        values.update(output=str(worker_output), task_overrides=str(worker_output/'task_overrides.json') if self.overrides else None, background_archives=False, prestage_next=False,
                      archive_completed_remote_payloads=False, release_remote_dataset=False, release_final_remote=False)
        return values

    def start_archive(self, task):
        """Start/adopt an independent writer; the worker cannot overwrite queue state."""
        path = self.output/f'{task}.archive_worker.json'
        values = self.archive_worker_arguments(task)
        row = read_json(path)
        if row is not None:
            if row.get('task') != task or row.get('args') != values:
                raise ValueError('Existing archive worker belongs to different settings')
            if row.get('status') == 'failed':
                raise RuntimeError(f'Archive worker failed; inspect {row.get("log")}')
            if row.get('status') != 'completed':
                ensure_worker_live(row, 'Archive')
        else:
            worker_output = Path(values['output']); worker_output.mkdir(parents=True, exist_ok=True)
            if self.overrides:
                override_path = Path(values['task_overrides'])
                previous = read_json(override_path)
                if previous is not None and previous != self.overrides:
                    raise ValueError('Archive worker override snapshot differs')
                write_json(override_path, self.overrides)
            row = dict(task=task, status='launching', args=values, log=str(worker_output/'worker.log'), created_unix=time.time())
            write_json(path, row)
            with Path(row['log']).open('a') as log:
                subprocess.Popen([str(self.args.python), str(Path(__file__).resolve()), 'archive-worker', str(path)],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.update(task, stage='training_complete_archiving', archive_worker=read_json(path))

    def poll_archives(self, *, require_all=False):
        if not self.args.background_archives:
            return True
        complete = True
        for task in TASKS:
            row = read_json(self.output/f'{task}.archive_worker.json')
            if row is None:
                if require_all:
                    complete = False
                continue
            if row.get('status') == 'failed':
                raise RuntimeError(f'Archive worker failed for {task}: {row.get("error")}; inspect {row.get("log")}')
            if row.get('status') == 'completed':
                evidence = read_json(Path(row['args']['output'])/f'{task}.archive.json', {})
                if (row.get('exit_code') != 0 or not evidence.get('verified_unix')
                        or evidence.get('manifest_sha256') != fingerprint(evidence.get('manifest'))):
                    raise RuntimeError('Completed archive worker lacks verified archive evidence')
                self.update(task, stage='completed', archived=True, archive_path=evidence['archive_path'], archive_worker=row)
            else:
                ensure_worker_live(row, 'Archive')
                self.update(task, archive_worker=row)
                complete = False
        return complete

    def launch(self, task):
        """Adopt in-flight transfers; never run two launchers on the same dataset."""
        self.wait_for_prestage(task)
        attempt = self.state['tasks'].get(task, {}).get('launch_attempt', 0)
        path = self.output/f'{task}.launch_{attempt:04d}.json'
        row = read_json(path)
        if row and row.get('status') == 'exited':
            if row.get('exit_code') not in (0,75):
                raise RuntimeError(f'Launch failed; inspect {path}')
            row = None
        if row is None:
            attempt += 1; path = self.output/f'{task}.launch_{attempt:04d}.json'
            command = self.launch_command(task)
            row = dict(status='launching', command=command, log=str(path.with_suffix('.log')), created_unix=time.time())
            write_json(path, row); self.update(task, launch_attempt=attempt, stage='launching')
            subprocess.Popen([str(self.args.python), str(Path(__file__).resolve()), 'launch-worker', str(path)],
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        elif row['command'] != self.launch_command(task):
            raise ValueError('Existing launcher command differs')
        while True:
            row = read_json(path)
            if row.get('status') == 'exited':
                if row.get('exit_code') == 75:
                    raise RetryLater(f'Launcher pending; see {row["log"]}')
                if row.get('exit_code') != 0:
                    raise RuntimeError(f'Launcher failed; see {row["log"]}')
                return
            if row.get('status') == 'running' and process_identity(row.get('supervisor_pid')) != row.get('start_ticks'):
                raise RuntimeError('Launcher supervisor disappeared without exit receipt')
            if row.get('status') == 'launching' and time.time()-row['created_unix'] > 60:
                raise RuntimeError('Launcher supervisor did not start')
            self.update(task, deployment=read_json(self.output/f'{task}.deployment.json', {}))
            time.sleep(min(30., self.args.poll_seconds))

    def await_completion(self, task):
        import launch_manipulation_sft as launcher
        while True:
            try:
                self.poll_archives()
                self.prestage_next(task)
                # Do not wait for the next dataset before completing/archiving this task.
                state = remote_call(self.args, 'status', self.request(task))
                self.update(task, stage='training' if state['status'] != 'not_started' else 'waiting_dataset', remote=state)
                if state['complete']:
                    return state
                if state['status'] == 'not_started':
                    launcher.audit_gate(self.args.dataset_root/f'pressb_{task}_500_20261110', task)
                    self.launch(task)
            except (RetryLater, launcher.RetryLater, subprocess.TimeoutExpired) as error:
                self.update(task, stage='waiting', reason=str(error))
            time.sleep(self.args.poll_seconds)

    def archive(self, task):
        args = self.args; request = self.request(task)
        receipt_path = self.output/f'{task}.archive.json'
        record = read_json(receipt_path)
        if record is None:
            self.update(task, stage='backing_up_model')
            inventory = remote_call(args, 'run-inventory', request)
            if inventory['marker']:
                raise ValueError('Remote payloads were archived but local verification receipt is missing')
            destination = args.archive_root/Path(inventory['run']).name
            destination.mkdir(parents=True, exist_ok=True)
            command = ['rsync','-aH','--partial','--protect-args','-e',shlex.join(['ssh', *SSH_OPTIONS]),
                       f'{args.host}:{inventory["run"]}/', str(destination)+'/']
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode:
                raise RetryLater(f'Archive rsync failed: {result.stderr[-2000:]}')
            local = tree_manifest(destination)
            fresh = remote_call(args, 'run-inventory', request)
            if local != inventory['manifest'] or local != fresh['manifest']:
                raise RuntimeError('Full local model archive differs from completed remote run')
            record = dict(task=task, archive_path=str(destination), run=inventory['run'], manifest=local,
                          manifest_sha256=fingerprint(local), verified_unix=time.time())
            write_json(receipt_path, record)
        elif tree_manifest(record['archive_path']) != record['manifest']:
            raise RuntimeError('Previously verified local model archive changed')
        rotate = task != TASKS[-1] or args.release_final_remote
        if rotate and args.archive_completed_remote_payloads and not record.get('remote_payloads_archived'):
            self.update(task, stage='releasing_verified_remote_weights')
            marker = remote_call(args, 'prune-model', dict(request, manifest=record['manifest'],
                verified_local_manifest_sha256=record['manifest_sha256'], archive_path=record['archive_path']))
            record.update(remote_payloads_archived=marker); write_json(receipt_path, record)
        if rotate and args.release_remote_dataset and not record.get('remote_dataset_archived'):
            self.update(task, stage='verifying_dataset_before_release')
            local_path = args.dataset_root/f'pressb_{task}_500_20261110'
            local = tree_manifest(local_path)
            inventory = remote_call(args, 'dataset-inventory', request)
            if not inventory.get('missing') and not inventory.get('marker') and local != inventory['manifest']:
                raise RuntimeError('Remote dataset differs from durable local dataset')
            marker = remote_call(args, 'release-dataset', dict(request, manifest=local,
                verified_local_manifest_sha256=fingerprint(local), local_path=str(local_path)))
            record.update(remote_dataset_archived=marker); write_json(receipt_path, record)
        self.update(task, stage='completed', archive_path=record['archive_path'], archived=True)

    def run(self):
        try:
            for task in TASKS:
                self.await_completion(task)
                if self.args.background_archives:
                    self.start_archive(task)
                else:
                    while True:
                        try:
                            self.archive(task); break
                        except (RetryLater, subprocess.TimeoutExpired) as error:
                            self.update(task, stage='waiting_archive', reason=str(error))
                            time.sleep(self.args.poll_seconds)
            if self.args.background_archives:
                self.state.update(status='archiving'); self.save()
                while not self.poll_archives(require_all=True):
                    time.sleep(self.args.poll_seconds)
            self.state.update(status='completed', finished_unix=time.time()); self.save()
        except Exception as error:
            self.state.update(status='failed', error=str(error)); self.save(); raise


def validate_overrides(overrides):
    allowed = {'remote_root', 'remote_dataset_root', 'checkpoint_reserve_gib', 'storage_margin_gib', 'shared_gpu_min_free_mib'}
    if not isinstance(overrides, dict) or any(task not in TASKS for task in overrides):
        raise ValueError('Unknown task override')
    for task, values in overrides.items():
        if not isinstance(values, dict) or set(values) - allowed:
            raise ValueError('Unknown task override fields')
        for key, value in values.items():
            if key in ('remote_root', 'remote_dataset_root'):
                if not isinstance(value, str) or not Path(value).is_absolute():
                    raise ValueError('Task remote paths must be absolute')
            elif key == 'shared_gpu_min_free_mib':
                if type(value) is not int or value <= 0:
                    raise ValueError('shared_gpu_min_free_mib must be a positive integer')
            elif not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
                raise ValueError('Storage budgets must be nonnegative')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', default='h200')
    p.add_argument('--remote-root', default='/home/pengguanqi/Worksapce/Research/PressB-task-sft-ddp2-20261110')
    p.add_argument('--remote-dataset-root', type=Path, default=Path('/home/pengguanqi/Datasets'))
    p.add_argument('--dataset-root', type=Path, default=Path('/home/pengguanqi/Datasets'))
    p.add_argument('--config-dir', type=Path, default=ROOT/'outputs/task_datasets_20261110/training_prepare')
    p.add_argument('--output', type=Path, default=ROOT/'outputs/task_datasets_20261110/ddp2_sft_queue')
    p.add_argument('--archive-root', type=Path, default=Path('/home/pengguanqi/Models/VLA-JEPA/manipulation_joint7_500_ddp2_20261110'))
    p.add_argument('--python', type=Path, default=ROOT/'.conda/envs/vlajepa-inference/bin/python')
    p.add_argument('--gpus-per-task', type=int, default=2)
    p.add_argument('--per-device-batch-size', type=int, default=32)
    p.add_argument('--run-suffix', default='')
    p.add_argument('--gpu-indices', type=int, nargs='+', help='Explicit GPU indices for each sequential task')
    p.add_argument('--checkpoint-reserve-gib', type=float)
    p.add_argument('--storage-margin-gib', type=float)
    p.add_argument('--task-overrides', type=Path, help='JSON task map overriding remote paths, storage budgets and optional shared_gpu_min_free_mib admission')
    p.add_argument('--poll-seconds', type=float, default=60.)
    p.add_argument('--archive-completed-remote-payloads', action='store_true')
    p.add_argument('--release-remote-dataset', action='store_true')
    p.add_argument('--release-final-remote', action='store_true', help='Also release the final task working copies; default retains them for evaluation')
    p.add_argument('--background-archives', action='store_true', help='Archive completed models in independent workers while the next task trains; requires retaining remote copies')
    p.add_argument('--prestage-next', action='store_true', help='Stage the next audited dataset during current training without allocating GPUs')
    args = p.parse_args(argv)
    if args.host.startswith('-') or not Path(args.remote_root).is_absolute() or not args.remote_dataset_root.is_absolute():
        p.error('Host and absolute remote paths are required')
    if min(args.gpus_per_task, args.per_device_batch_size, args.poll_seconds) <= 0:
        p.error('Counts/intervals must be positive')
    if args.gpu_indices is not None and (len(args.gpu_indices) != args.gpus_per_task or len(set(args.gpu_indices)) != len(args.gpu_indices) or min(args.gpu_indices) < 0):
        p.error('--gpu-indices must contain one unique nonnegative index per requested GPU')
    if any(value is not None and value < 0 for value in (args.checkpoint_reserve_gib, args.storage_margin_gib)):
        p.error('Storage budgets must be nonnegative')
    if args.background_archives and (args.archive_completed_remote_payloads or args.release_remote_dataset or args.release_final_remote):
        p.error('--background-archives cannot be combined with remote release flags')
    args.dataset_root = args.dataset_root.absolute(); args.config_dir = args.config_dir.resolve()
    args.archive_root = args.archive_root.resolve()
    return args


def main(argv=None):
    args = parse_args(argv); args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'.queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        Queue(args).run()
    return 0


if __name__ == '__main__':
    try:
        if len(sys.argv)>1 and sys.argv[1]=='remote':
            request=json.loads(base64.b64decode(sys.argv[3]))
            handlers={'status':remote_status,'run-inventory':remote_run_inventory,'prune-model':remove_checkpoint_payloads,
                      'dataset-inventory':remote_dataset_inventory,'release-dataset':release_dataset}
            print(json.dumps(handlers[sys.argv[2]](request)));raise SystemExit(0)
        if len(sys.argv)>1 and sys.argv[1]=='launch-worker':
            raise SystemExit(launch_worker(sys.argv[2]))
        if len(sys.argv)>1 and sys.argv[1]=='archive-worker':
            raise SystemExit(archive_worker(sys.argv[2]))
        raise SystemExit(main())
    except RetryLater as error:
        print(str(error),file=sys.stderr);raise SystemExit(75)
