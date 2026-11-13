#!/usr/bin/env python3
"""Archive an explicitly selected remote run directory before removing its copies.

Default mode only creates a plan. --apply requires the same recorded plan and
copies up to --workers independent top-level entries at a time. Complete SHA256 manifests are checked at
both ends; a durable proof and remote tombstone precede deletion. Symbolic links
are preserved literally, including broken links, and are never followed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import stat
import subprocess
import sys
import threading
import time

CANONICAL_TASK = re.compile(r'(^|[^a-z0-9])(?:cube[_-]?stack|tabletop[_-]?cleanup)($|[^a-z0-9])')
GENERIC_TASK_DIRECTORY = re.compile(r'(^|[_-])(?:stack|cleanup)($|[_-])')
RECEIPTS = '.pressb_archive_receipts'
SSH = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3']
DIRECTORY = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


class RetryLater(RuntimeError):
    pass


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
        stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)
    fd = os.open(path.parent, DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def identity(st):
    return dict(dev=st.st_dev, ino=st.st_ino, size=st.st_size,
                mtime_ns=st.st_mtime_ns, ctime_ns=st.st_ctime_ns, mode=st.st_mode)


def protected(value):
    """Recognize task paths, not historical housekeeping names containing cleanup.

    Canonical task names remain protected anywhere in a path component. Generic
    stack/cleanup names protect directories and extensionless task run names;
    scripts/logs/receipts such as cleanup_services.py and *_cleanup_receipt.json
    are ordinary historical artifacts rather than evidence of a task scene.
    """
    for part in PurePosixPath(str(value).lower()).parts:
        if CANONICAL_TASK.search(part):
            return True
        if '.' not in part and GENERIC_TASK_DIRECTORY.search(part):
            return True
    return False


def valid_name(name):
    if not isinstance(name, str) or not name or name in ('.', '..', RECEIPTS) or '/' in name or '\x00' in name:
        raise ValueError('Entry must be one ordinary top-level basename')
    if protected(name):
        raise ValueError(f'Protected current task: {name}')
    return name


def validate_manifest(manifest):
    if not isinstance(manifest, dict) or set(manifest) != {'nodes'} or '.' not in manifest['nodes']:
        raise ValueError('Invalid manifest')
    for relative, row in manifest['nodes'].items():
        p = PurePosixPath(relative)
        if relative != '.' and (p.is_absolute() or '..' in p.parts or str(p) != relative):
            raise ValueError('Manifest path escapes entry')
        if protected(relative) or (row.get('kind') == 'symlink' and protected(row.get('target', ''))):
            raise ValueError('Manifest contains a protected task')
        if row.get('kind') not in ('file', 'dir', 'symlink'):
            raise ValueError('Unsupported manifest node')
        if row['kind'] == 'file' and (type(row.get('bytes')) is not int or row['bytes'] < 0 or
                not isinstance(row.get('sha256'), str) or len(row['sha256']) != 64):
            raise ValueError('File manifest must contain bytes and SHA256')


def inventory(path, *, hash_files=True):
    """Use directory FDs and O_NOFOLLOW throughout, including file hashing."""
    path = Path(path).absolute()
    parent = os.open(path.parent, DIRECTORY)
    nodes, identities, allocated = {}, {}, 0
    seen_inodes = set()

    def visit(parent_fd, name, relative):
        nonlocal allocated
        before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        ident = identity(before); identities[relative] = ident
        if stat.S_ISLNK(before.st_mode):
            nodes[relative] = dict(kind='symlink', target=os.readlink(name, dir_fd=parent_fd))
        elif stat.S_ISREG(before.st_mode):
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
            try:
                if identity(os.fstat(fd)) != ident:
                    raise RuntimeError('File changed before hashing')
                row = dict(kind='file', bytes=before.st_size)
                if hash_files:
                    h = hashlib.sha256()
                    while block := os.read(fd, 8 * 1024**2):
                        h.update(block)
                    row['sha256'] = h.hexdigest()
                if identity(os.fstat(fd)) != ident:
                    raise RuntimeError('File changed while hashing')
                nodes[relative] = row
                key = (before.st_dev, before.st_ino)
                if key not in seen_inodes:
                    allocated += before.st_blocks * 512; seen_inodes.add(key)
            finally:
                os.close(fd)
        elif stat.S_ISDIR(before.st_mode):
            nodes[relative] = dict(kind='dir')
            fd = os.open(name, DIRECTORY, dir_fd=parent_fd)
            try:
                if identity(os.fstat(fd)) != ident:
                    raise RuntimeError('Directory changed before inventory')
                for child in sorted(os.listdir(fd)):
                    visit(fd, child, child if relative == '.' else relative + '/' + child)
                if identity(os.fstat(fd)) != ident:
                    raise RuntimeError('Directory changed during inventory')
            finally:
                os.close(fd)
        else:
            raise ValueError(f'Unsupported special file: {path}/{relative}')
        if identity(os.stat(name, dir_fd=parent_fd, follow_symlinks=False)) != ident:
            raise RuntimeError('Entry changed during inventory')
    try:
        visit(parent, path.name, '.')
    finally:
        os.close(parent)
    return dict(manifest=dict(nodes=nodes), identities=identities, allocated_bytes=allocated,
                logical_bytes=sum(x.get('bytes', 0) for x in nodes.values()))


def root_checked(request):
    raw = Path(request['remote_root'])
    if not raw.is_absolute():
        raise ValueError('Remote root must be absolute')
    root = raw.resolve(strict=True)
    if not root.is_dir() or root in (Path('/'), Path.home(), Path('/data')):
        raise ValueError('An explicit run directory is required')
    expected = request.get('root_identity')
    st = root.stat()
    if expected and expected != dict(dev=st.st_dev, ino=st.st_ino):
        raise ValueError('Remote root identity changed')
    return root


def entry_checked(request):
    root = root_checked(request)
    return root, root / valid_name(request['name'])


def session_daemon(comm, command):
    return ((comm == 'systemd' and command.endswith(' --user')) or
            (comm in ('(sd-pam)', 'sd-pam') and command == '(sd-pam)') or
            (comm == 'sshd' and command.startswith('sshd: ')))


def ensure_no_users(path):
    """Check active same-user processes, including mapped files and config refs."""
    path = Path(path).absolute(); prefix = str(path)
    alias = path.parent.name + '/' + path.name
    def uses(value):
        value = str(value).removesuffix(' (deleted)')
        return value == prefix or value.startswith(prefix + '/')
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            fields = (proc/'stat').read_text().rsplit(')', 1)[1].split()
            if fields[0] == 'Z':
                continue
            comm = (proc/'comm').read_text().strip()
            command = (proc/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace').strip()
            if prefix in command or alias in command:
                raise RetryLater(f'Process {proc.name} references {path}')
            cwd = None
            try:
                cwd = os.readlink(proc/'cwd')
                if uses(cwd):
                    raise RetryLater(f'Process {proc.name} works inside {path}')
                for fd in (proc/'fd').iterdir():
                    try:
                        if uses(os.readlink(fd)):
                            raise RetryLater(f'Process {proc.name} has an open file inside {path}')
                    except FileNotFoundError:
                        continue
                maps = (proc/'maps').read_text()
                if prefix in maps:
                    raise RetryLater(f'Process {proc.name} maps a file inside {path}')
            except PermissionError as error:
                if not session_daemon(comm, command):
                    raise RetryLater(f'Cannot inspect process {proc.name}: {comm}') from error
            for token in command.split():
                candidate = token.split('=', 1)[-1]
                if candidate.endswith(('.yaml', '.yml', '.json')):
                    config = Path(candidate)
                    if not config.is_absolute():
                        if cwd is None:
                            raise RetryLater(f'Cannot resolve process {proc.name} configuration')
                        config = Path(cwd)/config
                    if config.is_file() and config.stat().st_size <= 2*1024**2:
                        contents = config.read_text(errors='replace')
                        if prefix in contents or alias in contents:
                            raise RetryLater(f'Process {proc.name} configuration references {path}')
        except FileNotFoundError:
            continue
        except PermissionError as error:
            raise RetryLater(f'Cannot inspect same-user process {proc.name}') from error


def remote_plan(request):
    root = root_checked(request); st = root.stat(); entries, excluded = [], []
    for path in sorted(root.iterdir()):
        if path.name == RECEIPTS or protected(path.name):
            excluded.append(dict(name=path.name, reason='protected task or archive receipts')); continue
        row = inventory(path, hash_files=False)
        if any(protected(k) or protected(v.get('target', '')) for k,v in row['manifest']['nodes'].items()):
            excluded.append(dict(name=path.name, reason='protected descendant or symlink target')); continue
        entries.append(dict(name=path.name, **row))
    return dict(remote_root=str(root), root_identity=dict(dev=st.st_dev, ino=st.st_ino),
                entries=sorted(entries, key=lambda r:(r['logical_bytes'],r['name'])), excluded=excluded,
                created_unix=time.time())


def remote_inventory(request):
    _, path = entry_checked(request)
    ensure_no_users(path)
    row = inventory(path)
    if row['identities'] != request['planned_identities']:
        raise ValueError('Source changed since the approved inventory')
    validate_manifest(row['manifest'])
    return row


def marker_path(root, name):
    folder = root/RECEIPTS
    if folder.is_symlink():
        raise ValueError('Archive receipt directory must not be a symlink')
    folder.mkdir(exist_ok=True)
    return folder/(hashlib.sha256(name.encode()).hexdigest()+'.json')


def delete_verified(request):
    root, path = entry_checked(request)
    expected = request['manifest']; validate_manifest(expected)
    sha = fingerprint(expected)
    if request.get('verified_local_manifest_sha256') != sha or not request.get('local_archive'):
        raise ValueError('Complete verified local archive proof is required')
    marker_file = marker_path(root, path.name); marker = read_json(marker_file)
    if marker and (marker.get('manifest_sha256') != sha or marker.get('name') != path.name or
                   marker.get('local_archive') != request['local_archive']):
        raise ValueError('Remote tombstone disagrees with backup proof')
    ensure_no_users(path)
    if not os.path.lexists(path):
        if not marker:
            raise ValueError('Source disappeared without an archive tombstone')
        marker.update(status='archived', updated_unix=time.time()); write_json(marker_file,marker); return marker
    current = inventory(path); validate_manifest(current['manifest'])
    actual = current['manifest']['nodes']; wanted = expected['nodes']
    if (not marker and actual != wanted) or any(wanted.get(k) != v for k,v in actual.items()):
        raise ValueError('Remote entry differs from the verified full backup')
    if not marker and current['identities'] != request['planned_identities']:
        raise ValueError('Source identity changed before removal')
    marker = dict(status='removing', name=path.name, manifest_sha256=sha,
                  local_archive=request['local_archive'], local_proof=request.get('local_proof'),
                  updated_unix=time.time())
    write_json(marker_file, marker)
    ensure_no_users(path)
    # FD-relative traversal prevents following any directory replaced with a link.
    root_fd = os.open(root, DIRECTORY)
    unlinked_inodes = set()
    try:
        for relative in sorted(actual, key=lambda p:(len(PurePosixPath(p).parts),p), reverse=True):
            parts = [path.name] + ([] if relative == '.' else list(PurePosixPath(relative).parts))
            fd = os.dup(root_fd)
            try:
                for part in parts[:-1]:
                    child = os.open(part, DIRECTORY, dir_fd=fd); os.close(fd); fd=child
                name = parts[-1]; st = os.stat(name,dir_fd=fd,follow_symlinks=False)
                old = current['identities'][relative]
                if (st.st_dev,st.st_ino,st.st_mode) != (old['dev'],old['ino'],old['mode']):
                    raise ValueError('Node identity changed immediately before removal')
                if actual[relative]['kind'] == 'dir':
                    os.rmdir(name,dir_fd=fd)
                else:
                    if identity(st) != old:
                        # Unlinking another name of a hardlinked file changes
                        # its ctime. Rehash this name instead of accepting that
                        # metadata change without checking its bytes.
                        if actual[relative]['kind'] != 'file' or (st.st_dev, st.st_ino) not in unlinked_inodes:
                            raise ValueError('File changed after full SHA verification')
                        handle = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
                        try:
                            before = identity(os.fstat(handle)); h = hashlib.sha256()
                            while block := os.read(handle, 8 * 1024**2): h.update(block)
                            if (identity(os.fstat(handle)) != before or before['size'] != wanted[relative]['bytes']
                                    or h.hexdigest() != wanted[relative]['sha256']):
                                raise ValueError('Hardlinked file changed after verification')
                        finally:
                            os.close(handle)
                    os.unlink(name,dir_fd=fd)
                    unlinked_inodes.add((st.st_dev, st.st_ino))
            finally:
                os.close(fd)
        os.fsync(root_fd)
    finally:
        os.close(root_fd)
    marker.update(status='archived', updated_unix=time.time()); write_json(marker_file,marker)
    return marker


def remote_call(args, mode, request):
    wrapper = '''import json,sys,os
p=json.load(sys.stdin);ns={'__name__':'archive_remote_runs'}
exec(p['source'],ns)
try:
 os.nice(10)
 value=ns[p['mode']](p['request'])
 print(json.dumps(value))
except ns['RetryLater'] as e:
 print(str(e),file=sys.stderr);sys.exit(75)
'''
    command = ['ssh',*SSH,args.host,shlex.join(['python3','-c',wrapper])]
    packet = dict(source=Path(__file__).read_text(),mode=mode,request=request)
    result = subprocess.run(command,input=json.dumps(packet),capture_output=True,text=True,timeout=21600)
    if result.returncode in (75,255):
        raise RetryLater(result.stderr[-3000:] or 'SSH unavailable')
    if result.returncode:
        raise RuntimeError(result.stderr[-6000:] or result.stdout[-6000:])
    return json.loads(result.stdout.strip().splitlines()[-1])


class Archive:
    def __init__(self,args):
        self.state_lock=threading.RLock()
        self.stop=threading.Event()
        self.workers=getattr(args,'workers',1)
        if type(self.workers) is not int or not 1<=self.workers<=8:
            raise ValueError('workers must be an integer from 1 to 8')
        self.args=args; self.output=args.output.resolve();self.output.mkdir(parents=True,exist_ok=True)
        self.local=args.local_root.absolute();self.local.mkdir(parents=True,exist_ok=True)
        if self.local.is_symlink():
            raise ValueError('Local archive root must not be a symlink')
        self.contract=dict(host=args.host,remote_root=args.remote_root,local_root=str(self.local))
        self.state=read_json(self.output/'status.json',{})
        if self.state and self.state.get('contract')!=self.contract:
            raise ValueError('Archive contract changed; use a new output directory')
        self.state.update(contract=self.contract,pid=os.getpid(),status='planning',
                          workers=self.workers,entries=self.state.get('entries',{}));self.save()

    def save(self):
        with self.state_lock:
            self.state['updated_unix']=time.time();write_json(self.output/'status.json',self.state)

    def update(self,name,**changes):
        with self.state_lock:
            self.state['entries'].setdefault(name,{}).update(changes);self.save()

    def entry_state(self,name):
        with self.state_lock:
            return dict(self.state['entries'].get(name,{}))

    def request(self,plan,entry):
        return dict(remote_root=plan['remote_root'],root_identity=plan['root_identity'],name=entry['name'],
                    planned_identities=entry['identities'])

    def copy_command(self,name):
        source=self.plan['remote_root']+'/'+name
        return ['rsync','-aH','--checksum','--partial','--protect-args','--info=progress2','--stats',
                 '-e',shlex.join(['ssh',*SSH]),f'{self.args.host}:{source}',str(self.local)+'/']

    def adopt_transfer(self,name):
        command=self.copy_command(name)
        previous=self.entry_state(name)
        oldpid=previous.get('rsync_pid')
        # A surviving rsync may still own its temporary destination files.
        # Wait for it before validating that tree, never run two writers.
        while oldpid:
            try:
                fields=Path(f'/proc/{oldpid}/stat').read_text().rsplit(')',1)[1].split()
                oldcommand=[x.decode() for x in Path(f'/proc/{oldpid}/cmdline').read_bytes().split(b'\0') if x]
                alive=(fields[0]!='Z' and fields[19]==previous.get('rsync_start_ticks') and oldcommand==command)
            except (FileNotFoundError,PermissionError):
                alive=False
            if not alive:break
            self.update(name,stage='adopting_existing_transfer',heartbeat_unix=time.time());time.sleep(5)

    def copy(self,name,manifest):
        destination=self.local/name
        command=self.copy_command(name)
        logfile=self.output/(hashlib.sha256(name.encode()).hexdigest()+'.rsync.log')
        self.adopt_transfer(name)
        # A deliberate preseed may contain aliases, but never write through an
        # unexpected alias. Rsync's default also replaces directory symlinks.
        if os.path.lexists(destination):
            existing=inventory(destination,hash_files=False)['manifest']['nodes']
            for relative,row in existing.items():
                expected=manifest['nodes'].get(relative)
                if expected is None or row['kind']!=expected['kind'] or (row['kind']=='symlink' and row!=expected):
                    raise ValueError(f'Unexpected pre-existing local entry: {name}/{relative}')
            if existing['.']['kind']=='symlink':
                return
        with logfile.open('ab') as stream:
            proc=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
            ticks=Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
            self.update(name,stage='copying',rsync_pid=proc.pid,rsync_start_ticks=ticks,rsync_log=str(logfile),command=command)
            while proc.poll() is None:
                self.update(name,stage='copying',heartbeat_unix=time.time())
                time.sleep(5)
            code=proc.wait()
        if code in (10,12,30,35,255):
            raise RetryLater(f'Rsync temporary failure {code}; inspect {logfile}')
        if code:
            raise RuntimeError(f'Rsync failed {code}; inspect {logfile}')

    def process(self,plan,entry):
        name=entry['name']; request=self.request(plan,entry)
        proofpath=self.output/(hashlib.sha256(name.encode()).hexdigest()+'.verified.json')
        proof=read_json(proofpath)
        if proof is None:
            # A resumed supervisor adopts its old writer before remote hashing;
            # otherwise the remote use guard would merely poll the sender until
            # it exits, obscuring that this transfer is already progressing.
            self.adopt_transfer(name)
            self.update(name,stage='hashing_remote',logical_bytes=entry['logical_bytes'])
            remote=remote_call(self.args,'remote_inventory',request)
            self.copy(name,remote['manifest'])
            self.update(name,stage='verifying_local')
            local=inventory(self.local/name)
            if local['manifest']!=remote['manifest']:
                raise RuntimeError(f'Local backup SHA manifest differs: {name}')
            proof=dict(name=name,manifest=local['manifest'],manifest_sha256=fingerprint(local['manifest']),
                       local_archive=str(self.local/name),verified_unix=time.time(),source=request)
            write_json(proofpath,proof)
        else:
            if proof['source']!=request or inventory(self.local/name)['manifest']!=proof['manifest']:
                raise RuntimeError('Previously verified backup changed')
        self.update(name,stage='verified',proof=str(proofpath))
        marker=remote_call(self.args,'delete_verified',dict(request,manifest=proof['manifest'],
            verified_local_manifest_sha256=proof['manifest_sha256'],local_archive=proof['local_archive'],local_proof=str(proofpath)))
        self.update(name,stage='archived',remote_tombstone=marker,completed_unix=time.time(),
                    logical_bytes=entry['logical_bytes'],estimated_freed_bytes=entry['allocated_bytes'])

    def process_until_complete(self,entry):
        name=entry['name']
        while not self.stop.is_set():
            try:
                self.process(self.plan,entry)
                return True
            except (RetryLater,subprocess.TimeoutExpired) as error:
                self.update(name,stage='retrying',reason=str(error))
                if self.stop.wait(self.args.poll_seconds):
                    return False
            except Exception as error:
                # Stop scheduling new entries immediately. Other workers may
                # finish their current verified operation, but cannot turn the
                # global result back into completed or retry indefinitely.
                self.stop.set()
                with self.state_lock:
                    self.state['entries'].setdefault(name,{}).update(stage='failed',error=str(error))
                    self.state.update(status='failed',error=str(error),failed_entry=name)
                    self.save()
                raise
        return False

    def run_workers(self,entries):
        pending=iter(entries)
        first_error=None
        with ThreadPoolExecutor(max_workers=self.workers,thread_name_prefix='archive') as pool:
            active={}
            def fill():
                while len(active)<self.workers and not self.stop.is_set():
                    entry=next(pending,None)
                    if entry is None:break
                    active[pool.submit(self.process_until_complete,entry)]=entry['name']
            fill()
            while active:
                done,_=wait(active,return_when=FIRST_COMPLETED)
                for future in done:
                    active.pop(future)
                    try:
                        future.result()
                    except Exception as error:
                        self.stop.set()
                        if first_error is None:first_error=error
                fill()
        if first_error is not None:
            raise first_error

    def run(self):
        try:
            self.plan=read_json(self.output/'plan.json')
            if self.plan is None:
                self.plan=remote_call(self.args,'remote_plan',dict(remote_root=self.args.remote_root))
                self.plan['contract']=self.contract;write_json(self.output/'plan.json',self.plan)
            elif self.plan.get('contract')!=self.contract:
                raise ValueError('Stored plan belongs to a different archive request')
            names=[entry['name'] for entry in self.plan['entries']]
            if len(names)!=len(set(names)):
                raise ValueError('Plan contains duplicate top-level entries')
            for name in names:valid_name(name)
            with self.state_lock:
                self.state.update(status='running' if self.args.apply else 'planned',plan=str(self.output/'plan.json'),
                    entries_total=len(self.plan['entries']),logical_bytes=sum(e['logical_bytes'] for e in self.plan['entries']),
                    excluded=self.plan['excluded']);self.save()
            if not self.args.apply:
                return
            remaining=[entry for entry in self.plan['entries'] if self.entry_state(entry['name']).get('stage')!='archived']
            self.run_workers(remaining)
            with self.state_lock:
                if self.stop.is_set() or any(self.state['entries'].get(name,{}).get('stage')!='archived' for name in names):
                    raise RuntimeError('Archive workers stopped before every entry completed')
                self.state.update(status='completed',finished_unix=time.time());self.save()
        except Exception as error:
            with self.state_lock:
                self.state.update(status='failed',error=str(error));self.save()
            raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default='h200')
    parser.add_argument('--remote-root',required=True)
    parser.add_argument('--local-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--workers',type=int,choices=range(1,9),default=1,
                        help='Concurrent independent top-level archives (1-8); changing this does not change the archive identity')
    parser.add_argument('--poll-seconds',type=float,default=60.)
    args=parser.parse_args(argv)
    if not args.host or args.host.startswith('-') or not Path(args.remote_root).is_absolute() or args.poll_seconds<=0:
        parser.error('Valid host, absolute remote root and positive polling interval required')
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        Archive(args).run()
    return 0


if __name__=='__main__':
    raise SystemExit(main())
