#!/usr/bin/env python3
"""Deploy the two audited 500-episode datasets and launch isolated H200 SFT jobs.

Local mode is idempotent. Exit 75 means retry later (SSH, resources, or first step);
exit 2 means a dataset/configuration/job failure needs inspection. Nothing is
removed and an existing failed run is never silently resumed or overwritten.
"""
from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

TASKS = ('cube_stack', 'tabletop_cleanup')
REMOTE_ROOT = '/home/pengguanqi/Worksapce/Research/PressB-task-sft-20261110'
BASE_SHA256 = 'fd929c79d9bbd0bda56c0b952c7acb470d93c6241a519013fe5248c3f3ea5fab'
GIB = 1024**3
SSH_OPTIONS = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3']
PYTHON_COMPATIBILITY_PROBE = r'''
import importlib,json,sys,traceback
checks=[]
for module,attributes in [
 ('torch',[]),('torchvision',[]),('transformers',['Qwen3VLForConditionalGeneration','AutoVideoProcessor','VJEPA2VideoProcessor']),
 ('omegaconf',[]),('pyarrow',[]),('av',[]),('numpy',[]),('PIL',[]),('diffusers',[]),
 ('accelerate',[]),('qwen_vl_utils',['process_vision_info']),('einops',[]),('timm',[]),('rich',[]),('scipy',[])]:
 try:
  imported=importlib.import_module(module)
  for attribute in attributes:getattr(imported,attribute)
  checks.append(dict(module=module,ok=True,version=getattr(imported,'__version__',None)))
 except Exception as error:
  checks.append(dict(module=module,ok=False,error=str(error),traceback=traceback.format_exc()[-2500:]))
ok=all(row['ok'] for row in checks)
print(json.dumps(dict(ok=ok,python=sys.version,executable=sys.executable,checks=checks)))
raise SystemExit(0 if ok else 1)
'''


class RetryLater(RuntimeError):
    pass


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def directory_bytes(root):
    return sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file())


def file_inventory(root):
    """Detect changes after rsync's checksum verification without reuploading data."""
    root=Path(root)
    return {str(path.relative_to(root)):[path.stat().st_size,path.stat().st_mtime_ns]
            for path in sorted(root.rglob('*')) if path.is_file()}


def staged_identity(request, task):
    dataset=next(data for data in request['datasets'] if data['task']==task)
    return dict(task=task,source_files=request['source_files'],launch_settings=launch_settings(request),
        run_root=storage_settings(request)['remote_run_root'],dataset_path=dataset['remote_path'],
        dataset_metadata_sha256=dataset['metadata_sha256'])


def verified_stages(request):
    """An unchanged staged dataset can launch immediately after GPUs are free."""
    root=Path(request['remote_root'])
    result={}
    for task in selected_tasks(request):
        path=root/'outputs/stage_receipts'/f'{task}.json'
        if not path.exists():
            result[task]=None;continue
        row=read_json(path)
        if row.get('identity')!=staged_identity(request,task):
            raise RuntimeError(f'Existing {task} staging uses a different source/config/dataset identity; use a fresh remote root')
        try:
            valid=(row.get('status')=='staged' and row['dataset_inventory']==file_inventory(row['identity']['dataset_path'])
                and digest(row['config'])==row['config_sha256']
                and all(digest(root/relative)==expected for relative,expected in row['identity']['source_files'].items())
                and all(digest(Path(row['identity']['dataset_path'])/'meta'/name)==expected
                        for name,expected in row['identity']['dataset_metadata_sha256'].items()))
            checkpoint=Path(row.get('initialization_checkpoint') or row['base_checkpoint'])
            checkpoint_stat=row.get('initialization_checkpoint_stat') or row['base_checkpoint_stat']
            valid=valid and checkpoint_stat==[checkpoint.stat().st_size,checkpoint.stat().st_mtime_ns]
            valid=valid and all([Path(path).stat().st_size,Path(path).stat().st_mtime_ns]==expected
                for path,expected in row.get('initialization_support_files',{}).items())
        except (FileNotFoundError,KeyError):
            valid=False
        result[task]=row if valid else None
    return result


def audit_gate(root, task):
    """Only the exporter's fully decoded, atomically published output is eligible."""
    root = Path(root)
    if not (root/'meta/audit.json').is_file():
        raise RetryLater(f'Dataset not finalized/audited: {root}')
    audit = read_json(root/'meta/audit.json')
    info = read_json(root/'meta/info.json')
    manifest = read_json(root/'meta/export_manifest.json')
    count, frames = info['total_episodes'], info['total_frames']
    if not (audit.get('success') is True and audit.get('full_video_decode') is True
            and audit.get('episodes') == count == 500 and count == len(manifest['episodes'])
            and audit.get('frames') == audit.get('numeric_rows_compared') == frames
            and info.get('codebase_version') == 'v3.0' and info.get('fps') == 30
            and manifest.get('task_id') == task
            and all(e.get('success') is True for e in manifest['episodes'])):
        raise ValueError(f'Incomplete or incompatible dataset audit: {root}')
    if any(audit.get('decoded_frames', {}).get('observation.images.'+v) != frames for v in ('up','wrist')):
        raise ValueError(f'Decoded camera frame counts disagree: {root}')
    return dict(task=task, local_path=str(root.resolve()), remote_path=str(root.absolute()), bytes=directory_bytes(root), frames=frames,
                metadata_sha256={name:digest(root/'meta'/name) for name in ('audit.json','info.json','export_manifest.json')})


def quota_rows(raw):
    """Parse quota -w -p KiB output; preserve soft limits instead of consuming grace."""
    result = {}
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0].startswith('/'):
            try:
                used, soft, hard = [int(v.rstrip('*')) for v in parts[1:4]]
            except ValueError:
                continue
            limits = [v for v in (soft, hard) if v > 0]
            result[parts[0]] = None if not limits else max(0, min(limits)-used)*1024
    return result


def existing_ancestor(path):
    path = Path(path)
    while not path.exists():
        path = path.parent
    return path


def process_matches(receipt):
    pid = receipt.get('supervisor_pid')
    if not isinstance(pid, int):
        return False
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
        return (fields[0] != 'Z' and 'remote-worker' in command and str(receipt['receipt_path']) in command
                and fields[19] == receipt.get('supervisor_start_ticks'))
    except (FileNotFoundError, PermissionError, IndexError):
        return False


def job_state(receipt_path):
    if not Path(receipt_path).exists():
        return None
    result = read_json(receipt_path)
    if result.get('status') in ('completed', 'failed'):
        return result
    if process_matches(result):
        return result
    # A freshly detached supervisor may not yet have written its /proc identity.
    if result.get('status') == 'launching' and time.time()-result['created_unix'] < 30:
        return result
    raise RuntimeError(f'SFT supervisor exited without a completion receipt; inspect {receipt_path}')


def launch_settings(request):
    settings=dict(gpus_per_task=1,per_device_batch_size=None,run_suffix='',gpu_indices=None)
    settings.update(request.get('launch_settings',{}))
    if type(settings['gpus_per_task']) is not int or settings['gpus_per_task'] not in (1,2,3):
        raise ValueError('gpus_per_task must be 1, 2, or 3')
    batch=settings['per_device_batch_size']
    if batch is not None and (type(batch) is not int or batch<1):
        raise ValueError('per_device_batch_size must be a positive integer')
    suffix=settings['run_suffix']
    if not isinstance(suffix,str) or (suffix and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',suffix)):
        raise ValueError('run_suffix must contain only letters, digits, underscores, or hyphens')
    indices=settings['gpu_indices']
    if indices is not None:
        if (not isinstance(indices,list) or any(type(index) is not int or index<0 for index in indices)
                or len(indices)!=len(set(indices))):
            raise ValueError('gpu_indices must be a list of distinct nonnegative integers')
        if len(indices)<settings['gpus_per_task']:
            raise ValueError('gpu_indices must include at least gpus_per_task devices')
    return settings


def task_run_id(task, suffix=''):
    return f'vlajepa_{task}_500_20261110'+('_'+suffix if suffix else '')


def storage_settings(request):
    reserve=request.get('checkpoint_reserve_gib',32.)
    margin=request.get('storage_margin_gib',1.)
    for name,value,positive in [('checkpoint_reserve_gib',reserve,True),('storage_margin_gib',margin,False)]:
        if type(value) not in (int,float) or not math.isfinite(value) or value<0 or (positive and value==0):
            raise ValueError(f'{name} must be a finite '+('positive' if positive else 'nonnegative')+' number')
    run_root=Path(request.get('remote_run_root') or (Path(request['remote_root'])/'outputs/model_training'))
    if not run_root.is_absolute():
        raise ValueError('remote_run_root must be absolute')
    return dict(checkpoint_reserve_gib=reserve,storage_margin_gib=margin,remote_run_root=str(run_root))


def receipt_gpu_uuids(receipt):
    return receipt.get('gpu_uuids') or ([receipt['gpu_uuid']] if receipt.get('gpu_uuid') else [])


def shared_gpu_min_free_mib(request):
    """Runtime admission policy, deliberately excluded from training identity."""
    minimum=request.get('shared_gpu_min_free_mib')
    if minimum is not None:
        if type(minimum) is not int or minimum<=0:
            raise ValueError('shared_gpu_min_free_mib must be a positive integer')
        if launch_settings(request)['gpu_indices'] is None:
            raise ValueError('Shared GPU admission requires an explicit --gpu-indices list')
    return minimum


def gpu_admitted(total, used, utilization, minimum=None):
    if not all(math.isfinite(value) for value in (total,used,utilization)) or not 0<=used<=total:
        return False
    if minimum is not None:
        return total-used>=minimum
    return used<=2048 and utilization<=10 and total-used>=80000


def selected_gpu_inventory(devices, settings):
    """Apply an ordered allowlist after all availability checks have passed."""
    indices=settings.get('gpu_indices')
    if indices is None:
        return devices
    eligible={gpu['index']:gpu for gpu in devices}
    return [eligible[index] for index in indices if index in eligible]


def allocate_gpu_groups(devices, pending, count):
    """Only start complete, disjoint groups; remaining tasks can retry later."""
    if len({gpu['uuid'] for gpu in devices})!=len(devices):
        raise ValueError('Duplicate GPU UUID in resource inventory')
    groups={}
    for index,task in enumerate(pending):
        group=devices[index*count:(index+1)*count]
        if len(group)<count:break
        groups[task]=group
    if pending and not groups:
        raise RetryLater(f'Need {count} eligible H200 GPUs per task; only {len(devices)} available')
    return groups


def checkpoint_reserve_bytes(root, receipts, tasks=TASKS, run_suffix='', *,
                             run_root=None, checkpoint_reserve_gib=32., storage_margin_gib=1.):
    """Budget selected new jobs plus every active sibling against shared quota."""
    reserve=0
    pending=False
    for task,row in receipts.items():
        selected_new=row is None and task in tasks
        active=row is not None and row.get('status') in ('launching','starting','running')
        if not selected_new and not active:
            continue
        pending=True
        base=Path(run_root) if run_root is not None else Path(root)/'outputs/model_training'
        run=Path(row['run']) if row and row.get('run') else base/task_run_id(task,run_suffix)
        reserve+=max(0,math.ceil(checkpoint_reserve_gib*GIB)-directory_bytes(run))
    return reserve+(math.ceil(storage_margin_gib*GIB) if pending else 0)


def checkpoint_storage_needs(root, receipts, tasks, settings, storage):
    """Keep each active run's reservation on its own output filesystem."""
    budgets={}
    for task,row in receipts.items():
        if not ((row is None and task in tasks) or (row and row.get('status') in ('launching','starting','running'))):
            continue
        base=Path(row['run']).parent if row and row.get('run') else Path(storage['remote_run_root'])
        # A later request may use a smaller budget for another task. Preserve
        # the active sibling's own reservation instead of silently shrinking it.
        own_storage=row.get('storage_settings',storage) if row else storage
        amount=checkpoint_reserve_bytes(root,{task:row},tasks,settings['run_suffix'],run_root=base,
            checkpoint_reserve_gib=own_storage['checkpoint_reserve_gib'],storage_margin_gib=0.)
        entry=budgets.setdefault(base,dict(size=0,margin=0))
        entry['size']+=amount
        entry['margin']=max(entry['margin'],math.ceil(own_storage['storage_margin_gib']*GIB))
    return [(base,entry['size']+entry['margin']) for base,entry in budgets.items()]


def training_readiness(receipt):
    """A live process is not ready until an optimizer update is on disk."""
    if receipt is None:
        return dict(ready=False,reason='process_not_started')
    if receipt.get('status')=='failed':
        raise RuntimeError(f'Existing {receipt.get("task")} job failed; inspect {receipt.get("log")}')
    run=Path(receipt['run'])
    pending=dict(ready=False,reason='awaiting_first_optimizer_step',run=str(run))
    path=run/'train.jsonl'
    if not path.is_file() or not (run/'status.json').is_file():
        if receipt.get('status')=='completed':
            raise RuntimeError(f'Completed job lacks optimizer-step evidence: {run}')
        return pending
    with path.open('rb') as stream:
        first=stream.readline()
        if not first.endswith(b'\n'):
            return pending
        stream.seek(0,os.SEEK_END)
        size=stream.tell()
        stream.seek(max(0,size-65536))
        complete_lines=stream.read().split(b'\n')[:-1]
    if not complete_lines:
        return pending
    records=[json.loads(first),json.loads(complete_lines[-1])]
    metrics=('action_loss','wm_loss_weighted_0_1','grad_norm')
    for record in records:
        if record.get('event')!='train' or type(record.get('step')) is not int or record['step']<1:
            raise RuntimeError(f'Invalid optimizer-step record: {path}')
        if any(type(record.get(k)) not in (int,float) or not math.isfinite(record[k]) for k in metrics):
            raise RuntimeError(f'Nonfinite or missing training loss/gradient: {path}')
    status=read_json(run/'status.json')
    if status.get('status')=='failed':
        raise RuntimeError(f'Training status reports failure: {run}')
    for key in metrics:
        if key in status and (type(status[key]) not in (int,float) or not math.isfinite(status[key])):
            raise RuntimeError(f'Nonfinite training status {key}: {run}')
    if (status.get('status') not in ('training','completed') or type(status.get('step')) is not int
            or status['step']<records[0]['step']):
        if receipt.get('status')=='completed':
            raise RuntimeError(f'Completed job has no verified training status: {run}')
        return pending
    return dict(ready=True,reason='optimizer_step_verified',first_verified_step=records[0]['step'],
                latest_log_step=records[-1]['step'],status_step=status['step'],
                first_step_metrics={k:records[0][k] for k in metrics},run=str(run))


def selected_tasks(request):
    tasks=request.get('tasks',list(TASKS))
    if not isinstance(tasks,list) or not tasks or len(tasks)!=len(set(tasks)) or any(t not in TASKS for t in tasks):
        raise ValueError('tasks must be a nonempty unique selection of supported tasks')
    return tasks


def task_receipts(root,tasks):
    receipts={}
    for task in TASKS:
        path=Path(root)/'outputs/launch_receipts'/f'{task}.json'
        # Unselected jobs still reserve their GPU and unwritten checkpoint
        # budget. Their readiness/failure must not gate the selected task.
        receipts[task]=job_state(path) if task in tasks else (read_json(path) if path.is_file() else None)
    return receipts


def readiness_summary(receipts,tasks=TASKS):
    readiness={task:training_readiness(receipts[task]) for task in tasks}
    verified=all(row['ready'] for row in readiness.values())
    return dict(selected_tasks=list(tasks),processes_started=all(receipts[task] is not None for task in tasks),
                first_steps_verified=verified,readiness=readiness,complete=verified)


def discover_model_python(home):
    candidates = [home/'miniconda3/envs/vlajepa-piper/bin/python',
                  home/'miniconda3/envs/vlajepa-inference/bin/python',
                  home/'Worksapce/Research/VLA-JEPA/.conda/envs/vlajepa-inference/bin/python']
    candidates += sorted((home/'miniconda3/envs').glob('*/bin/python'))
    diagnostics=[]
    for candidate in dict.fromkeys(candidates):
        if not candidate.is_file():
            continue
        try:
            checked=subprocess.run([str(candidate),'-c',PYTHON_COMPATIBILITY_PROBE],
                                   capture_output=True,text=True,timeout=120)
        except subprocess.TimeoutExpired as error:
            diagnostics.append(dict(python=str(candidate),ok=False,error=f'Import probe timeout: {error}'))
            continue
        try:
            report=json.loads(checked.stdout.strip().splitlines()[-1])
        except (IndexError,json.JSONDecodeError):
            report=dict(ok=False,error='Import probe did not return a JSON report',stdout=checked.stdout[-4000:])
        report.update(candidate=str(candidate),returncode=checked.returncode,stderr=checked.stderr[-4000:])
        diagnostics.append(report)
        if checked.returncode==0 and report.get('ok') is True:
            return str(candidate),diagnostics
    raise RuntimeError('No existing VLA-JEPA compatible Python environment found on H200. Import diagnostics: '+json.dumps(diagnostics))


def remote_probe(request):
    root = Path(request['remote_root'])
    tasks=selected_tasks(request)
    settings=launch_settings(request)
    shared_minimum=shared_gpu_min_free_mib(request)
    storage=storage_settings(request)
    receipts=task_receipts(root,tasks)
    for task, row in receipts.items():
        if task in tasks and row is not None and row.get('status') == 'failed':
            raise RuntimeError(f'Existing {task} job failed; inspect {row.get("log")}')
        if row is not None:
            data=next((d for d in request.get('datasets',[]) if d['task']==task),None)
            if data is not None and row.get('dataset_metadata_sha256')!=data['metadata_sha256']:
                raise RuntimeError(f'Existing {task} job belongs to different dataset metadata')
            if 'source_files' in request and row.get('source_files')!=request['source_files']:
                raise RuntimeError(f'Existing {task} job belongs to a different source/config snapshot')
            if task in tasks and 'launch_settings' in request:
                previous=launch_settings(dict(launch_settings=row.get('launch_settings',{})))
                if previous!=settings or row.get('world_size',1)!=settings['gpus_per_task']:
                    raise RuntimeError(f'Existing {task} job uses different GPU/batch/run settings; use a fresh remote root')
                expected=Path(storage['remote_run_root'])/task_run_id(task,settings['run_suffix'])
                if row.get('run')!=str(expected):
                    raise RuntimeError(f'Existing {task} job uses a different run directory')
    readiness=readiness_summary(receipts,tasks)
    if readiness['processes_started']:
        return dict(jobs=receipts,**readiness)
    home = Path.home()
    model_python,python_diagnostics=discover_model_python(home)
    assets = dict(qwen=home/'Models/Qwen3-VL-2B-Instruct', vjepa=home/'Models/vjepa2-vitl-fpc64-256',
                  checkpoint=home/'Models/VLA-JEPA/Pretrain/checkpoints/VLA-JEPA-pretrain.pt')
    for name, path in assets.items():
        if not path.exists():
            raise RuntimeError(f'Required pretrained asset missing: {name}: {path}')
    devices=[]
    groups={}
    if not request.get('stage_only',False):
        query = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'], text=True)
        claimed={uid for r in receipts.values() if r and r.get('status') != 'completed' for uid in receipt_gpu_uuids(r)}
        for line in query.splitlines():
            index, uid, name, total, used, util = [p.strip() for p in line.split(',')]
            if 'H200' in name and uid not in claimed and gpu_admitted(float(total),float(used),float(util),shared_minimum):
                devices.append(dict(index=int(index),uuid=uid,name=name,free_mib=float(total)-float(used)))
        devices=selected_gpu_inventory(devices,settings)
        groups=allocate_gpu_groups(devices,[task for task in tasks if receipts[task] is None],settings['gpus_per_task'])
    # Track both filesystem free space and the user's historically restrictive quota.
    quota = subprocess.run(['quota','-w','-p'],text=True,capture_output=True,timeout=30) if shutil.which('quota') else None
    if quota is None or quota.returncode not in (0,1):
        raise RetryLater('Cannot inspect user quota; storage availability is unverified')
    quotas = quota_rows(quota.stdout)
    needs = []
    for data in request['datasets']:
        remote_path=Path(data['remote_path'])
        already=directory_bytes(remote_path) if remote_path.exists() else 0
        needs.append((remote_path,max(0,data['bytes']-already)))
    # Unselected jobs that have never started do not reserve space. An active
    # sibling still keeps its unwritten checkpoint budget across retries.
    needs.extend(checkpoint_storage_needs(root,receipts,tasks,settings,storage))
    by_device={}
    for path, size in needs:
        ancestor=existing_ancestor(path)
        device=subprocess.check_output(['df','--output=source',str(ancestor)],text=True).splitlines()[-1].strip()
        if device not in quotas:
            raise RetryLater(f'No interpretable user quota row for {device}; quota output: {quota.stdout[-2000:]}')
        entry=by_device.setdefault(device,dict(required_bytes=0,filesystem_free_bytes=shutil.disk_usage(ancestor).free,
                                               quota_available_bytes=quotas.get(device)))
        entry['required_bytes']+=size
    for device,row in by_device.items():
        capacities=[row['filesystem_free_bytes']]
        if row['quota_available_bytes'] is not None:
            capacities.append(row['quota_available_bytes'])
        if min(capacities)<row['required_bytes']:
            raise RetryLater(f'Insufficient capacity on {device}: {row}; no deletion performed')
    stages=verified_stages(request) if 'source_files' in request else {task:None for task in tasks}
    return dict(jobs=receipts,**readiness,python=model_python,assets={k:str(v) for k,v in assets.items()},
                python_diagnostics=python_diagnostics,gpus=devices,gpu_groups=groups,storage=by_device,quota_raw=quota.stdout,
                stages=stages,all_staged=all(stages.values()))


def model_preflight(root, python, config, task, world_size=1):
    code=r'''
import json,sys
from pathlib import Path
from omegaconf import OmegaConf
from pressb.model_sources import activate_source
activate_source('vlajepa')
from pressb.training.vlajepa_joint7_data import Joint7LeRobotDataset
from starVLA.training.train_piper import epoch_geometry,sha256
c=OmegaConf.to_container(OmegaConf.load(sys.argv[1]),resolve=True)
d=Joint7LeRobotDataset(c['datasets']['vla_data']['data_root_dir'],split='train',seed=c['seed'],sample_stride=c['datasets']['vla_data']['sample_stride'])
v=Joint7LeRobotDataset(c['datasets']['vla_data']['data_root_dir'],split='val',seed=c['seed'])
assert d.source_episode_count==500 and len(v.source_episode_ids)==50 and d.task_id==sys.argv[2]
for x in (d,v):
 for i in (0,len(x)-1):
  s=x[i]; assert s['state'].shape==(1,7) and s['action'].shape==(7,7) and s['video'].shape==(2,8,224,224,3)
world=int(sys.argv[3]);batch=c['datasets']['vla_data']['per_device_batch_size'];accumulation=c['trainer']['gradient_accumulation_steps']
g=epoch_geometry(len(d),world,batch,accumulation,c['trainer']['target_epochs'])
if any(c['trainer'].get(key) is not None for key in ('warm_start_checkpoint','warm_start_sha256','warm_start_step')):
 from starVLA.training.train_table_buttons import inspect_warm_start
 warm=inspect_warm_start(c,schema='vlajepa.joint7.v1',provenance=dict(dataset_sha256=d.export_manifest_sha256,split_sha256=d.split_manifest_sha256,task_id=d.task_id),normalization=d.normalization)
 initialization=dict(initialization='warm_start_model_only',warm_start=warm)
else:
 assert sha256(c['trainer']['pretrained_checkpoint'])==c['trainer']['pretrained_sha256']
 initialization=dict(initialization='official_pretrained',base_sha256=c['trainer']['pretrained_sha256'])
print(json.dumps(dict(geometry=g,world_size=world,per_device_batch_size=batch,global_batch_size=world*batch*accumulation,task=d.task_id,instruction=d.instruction,train_episodes=450,val_episodes=50,frames=d.info['total_frames'],**initialization)))
'''
    env=dict(os.environ,PYTHONPATH=str(root/'src'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4')
    result=subprocess.run([python,'-c',code,str(config),task,str(world_size)],cwd=root,env=env,text=True,capture_output=True,timeout=900)
    if result.returncode:
        raise RuntimeError(f'{task} preflight failed:\n{result.stderr[-8000:]}\n{result.stdout[-1000:]}')
    return json.loads(result.stdout.strip().splitlines()[-1])


def verify_transferred_files(request):
    root=Path(request['remote_root'])
    for relative,expected in request['source_files'].items():
        if digest(root/relative)!=expected:
            raise RuntimeError(f'Deployed source differs: {relative}')
    for data in request['datasets']:
        for name,expected in data['metadata_sha256'].items():
            if digest(Path(data['remote_path'])/'meta'/name)!=expected:
                raise RuntimeError(f'Dataset metadata differs after transfer: {data["task"]}/{name}')


def prepare_task(request, probe, task):
    """Resolve local model/data paths and CPU-check a fresh task, without CUDA."""
    root=Path(request['remote_root'])
    settings,storage=launch_settings(request),storage_settings(request)
    template=root/'training_prepare'/f'{task}.yaml'
    config=root/'training_prepare'/f'{task}.resolved.yaml'
    dataset=next(data for data in request['datasets'] if data['task']==task)
    code='''import json,sys;from omegaconf import OmegaConf\nc=OmegaConf.load(sys.argv[1]);a=json.loads(sys.argv[3]);c.run_root_dir=sys.argv[4];c.datasets.vla_data.data_root_dir=sys.argv[5];c.framework.qwenvl.base_vlm=a['qwen'];c.framework.vj2_model.base_encoder=a['vjepa'];c.trainer.pretrained_checkpoint=a['checkpoint'];OmegaConf.save(c,sys.argv[2])'''
    subprocess.run([probe['python'],'-c',code,str(template),str(config),json.dumps(probe['assets']),
                    storage['remote_run_root'],dataset['remote_path']],check=True)
    return config,model_preflight(root,probe['python'],config,task,settings['gpus_per_task'])


def remote_stage(request):
    """Validate a transferred task while training GPUs are occupied; never launch."""
    request=dict(request,stage_only=True)
    root=Path(request['remote_root'])
    root.mkdir(parents=True,exist_ok=True)
    with (root/'.launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        probe=remote_probe(request)
        tasks=selected_tasks(request)
        if any(probe['jobs'][task] is not None for task in tasks):
            raise RuntimeError('stage-only requires unstarted tasks; existing training jobs are left untouched')
        verify_transferred_files(request)
        stages=probe.get('stages',{})
        for task in tasks:
            if stages.get(task):
                continue
            config,metadata=prepare_task(request,probe,task)
            identity=staged_identity(request,task)
            warm=metadata.get('warm_start') or {}
            checkpoint=Path(warm.get('checkpoint_path') or probe['assets']['checkpoint'])
            support_files={}
            if warm:
                source_run=Path(warm['source_run'])
                support=[checkpoint.parent/'manifest.json',*(Path(warm['processor_path']).rglob('*'))]
                support.extend(source_run/name for name in ('split.json','normalization.json','action_schema.json',
                    'training_contract.json','dataset_fingerprint.json'))
                support_files={str(path):[path.stat().st_size,path.stat().st_mtime_ns] for path in support if path.is_file()}
            row=dict(status='staged',staged_unix=time.time(),identity=identity,config=str(config),
                     config_sha256=digest(config),preflight=metadata,
                     dataset_inventory=file_inventory(identity['dataset_path']),initialization_checkpoint=str(checkpoint),
                     initialization_checkpoint_stat=[checkpoint.stat().st_size,checkpoint.stat().st_mtime_ns],
                     initialization_support_files=support_files)
            write_json(root/'outputs/stage_receipts'/f'{task}.json',row)
            stages[task]=row
        return dict(status='staged',stages=stages,all_staged=True,selected_tasks=tasks,
                    processes_started=False,first_steps_verified=False,storage=probe['storage'])


def remote_ensure(request):
    if request.get('stage_only',False):
        raise ValueError('stage-only cannot invoke the training launch operation')
    root=Path(request['remote_root'])
    tasks=selected_tasks(request)
    settings=launch_settings(request)
    shared_minimum=shared_gpu_min_free_mib(request)
    storage=storage_settings(request)
    root.mkdir(parents=True,exist_ok=True)
    with (root/'.launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        probe=remote_probe(request)
        if probe['processes_started']:
            return probe
        verify_transferred_files(request)
        pending=[t for t in tasks if probe['jobs'][t] is None]
        groups=allocate_gpu_groups(probe['gpus'],pending,settings['gpus_per_task'])
        for task in pending:
            group=groups.get(task)
            if group is None:
                break
            staged=probe.get('stages',{}).get(task)
            if staged:
                config,metadata=Path(staged['config']),staged['preflight']
            else:
                config,metadata=prepare_task(request,probe,task)
            # Preflight can take minutes. Recheck every rank's GPU before use.
            admission_samples=[]
            for gpu in group:
                current=subprocess.check_output(['nvidia-smi',f'--id={gpu["uuid"]}',
                    '--query-gpu=memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
                total,used,util=[float(v.strip()) for v in current.strip().split(',')]
                if not gpu_admitted(total,used,util,shared_minimum):
                    raise RetryLater(f'GPU {gpu["uuid"]} became busy during preflight')
                admission_samples.append(dict(index=gpu['index'],uuid=gpu['uuid'],
                    free_mib=total-used,used_mib=used,utilization=util))
            receipt_path=root/'outputs/launch_receipts'/f'{task}.json'
            run=Path(storage['remote_run_root'])/task_run_id(task,settings['run_suffix'])
            if run.exists() and any(run.iterdir()):
                raise RuntimeError(f'Existing run without launch receipt: {run}')
            receipt=dict(status='launching',task=task,created_unix=time.time(),receipt_path=str(receipt_path),
                         gpu_uuid=group[0]['uuid'],gpu_index=group[0]['index'],
                         gpu_uuids=[gpu['uuid'] for gpu in group],gpu_indices=[gpu['index'] for gpu in group],
                         world_size=len(group),per_device_batch_size=metadata['per_device_batch_size'],
                         global_batch_size=metadata['global_batch_size'],launch_settings=settings,storage_settings=storage,
                         shared_gpu_min_free_mib=shared_minimum,gpu_admission_samples=admission_samples,
                         python=probe['python'],config=str(config),
                         root=str(root),run=str(run),log=str(receipt_path.with_suffix('.log')),
                         config_sha256=digest(config),source_files=request['source_files'],preflight=metadata)
            receipt['dataset_metadata_sha256']=next(d['metadata_sha256'] for d in request['datasets'] if d['task']==task)
            write_json(receipt_path,receipt)
            with receipt_path.with_suffix('.supervisor.log').open('ab') as log:
                proc=subprocess.Popen([sys.executable,str(root/'scripts/launch_manipulation_sft.py'),'remote-worker',str(receipt_path)],
                                      stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            # Worker owns subsequent receipt updates. Wait only for startup acknowledgement.
            deadline=time.monotonic()+10
            while time.monotonic()<deadline:
                state=read_json(receipt_path)
                if state['status']!='launching':break
                if proc.poll() is not None:raise RuntimeError(f'Supervisor exited: {receipt_path}')
                time.sleep(.1)
            if state['status']=='failed':
                raise RuntimeError(f'Training exited during launch; inspect {state["log"]}')
            probe['jobs'][task]=state
        probe['jobs']=task_receipts(root,tasks)
        probe.update(readiness_summary(probe['jobs'],tasks))
        return probe


def training_command(receipt):
    root=Path(receipt['root'])
    devices=receipt_gpu_uuids(receipt)
    world=receipt.get('world_size',len(devices) or 1)
    if len(devices)!=world or len(set(devices))!=world:
        raise ValueError('Training receipt needs one distinct GPU UUID per rank')
    command=[receipt['python']]
    if world>1:
        command+=['-m','torch.distributed.run','--standalone',f'--nproc-per-node={world}']
    command+=[str(root/'scripts/train_model.py'),'--model','vlajepa','--profile','joint7_generic','--config',receipt['config']]
    return command,','.join(devices)


def remote_worker(receipt_path):
    receipt=read_json(receipt_path)
    root=Path(receipt['root'])
    receipt.update(status='starting',supervisor_pid=os.getpid(),supervisor_start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])
    write_json(receipt_path,receipt)
    command,visible_devices=training_command(receipt)
    env=dict(os.environ,PYTHONPATH=str(root/'src'),CUDA_VISIBLE_DEVICES=visible_devices,OMP_NUM_THREADS='4',
             TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    try:
        with Path(receipt['log']).open('xb') as log:
            proc=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            receipt.update(status='running',training_pid=proc.pid,command=command)
            write_json(receipt_path,receipt)
            code=proc.wait()
        completed=Path(receipt['run'])/'completed.json'
        receipt.update(status='completed' if code==0 and completed.exists() and read_json(completed).get('success') else 'failed',
                       exit_code=code,finished_unix=time.time())
    except Exception as error:
        receipt.update(status='failed',error=str(error),finished_unix=time.time())
    write_json(receipt_path,receipt)
    return receipt


def remote_call(host, mode, request):
    encoded=base64.b64encode(json.dumps(request).encode()).decode()
    cmd=shlex.join(['python3','-',mode,encoded])
    result=subprocess.run(['ssh',*SSH_OPTIONS,host,cmd],input=Path(__file__).read_text(),text=True,capture_output=True,timeout=2400)
    if result.returncode in (75,255):
        raise RetryLater(result.stderr[-2000:] or result.stdout[-2000:] or 'Remote temporarily unavailable')
    if result.returncode:
        raise RuntimeError(result.stderr[-8000:] or result.stdout[-8000:])
    return json.loads(result.stdout.strip().splitlines()[-1])


def snapshot(root, config_dir, output, settings=None):
    fixed=['src/pressb/__init__.py','src/pressb/model_sources/__init__.py','src/pressb/training/__init__.py',
           'src/pressb/training/cli.py','src/pressb/training/image_transforms.py','src/pressb/training/vlajepa_joint7.py',
           'src/pressb/training/vlajepa_joint7_data.py','scripts/train_model.py','scripts/launch_manipulation_sft.py']
    paths=[root/p for p in fixed]+[p for p in (root/'src/pressb/model_sources/vlajepa').rglob('*')
                                  if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo')]
    files={str(p.relative_to(root)):digest(p) for p in paths}
    generated={}
    for task in TASKS:
        relative=f'training_prepare/{task}.yaml'
        if settings is None:
            files[relative]=digest(config_dir/f'{task}.yaml')
        else:
            import yaml
            config=yaml.safe_load((config_dir/f'{task}.yaml').read_text())
            config['run_id']=task_run_id(task,settings['run_suffix'])
            if settings['per_device_batch_size'] is not None:
                config['datasets']['vla_data']['per_device_batch_size']=settings['per_device_batch_size']
            generated[relative]=yaml.safe_dump(config,sort_keys=False).encode()
            files[relative]=hashlib.sha256(generated[relative]).hexdigest()
    if settings is not None:
        relative='training_prepare/launch_settings.json'
        generated[relative]=(json.dumps(settings,sort_keys=True,indent=2)+'\n').encode()
        files[relative]=hashlib.sha256(generated[relative]).hexdigest()
    key=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    destination=output/key
    for relative in files:
        source=config_dir/Path(relative).name if relative.startswith('training_prepare/') else root/relative
        target=destination/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            if relative in generated:target.write_bytes(generated[relative])
            else:shutil.copyfile(source,target)
        if digest(target)!=files[relative]:raise RuntimeError(f'Snapshot changed: {target}')
    return destination,files,key


def local_main(argv):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default='h200')
    parser.add_argument('--tasks',nargs='+',choices=TASKS,default=list(TASKS),
                        help='Tasks to audit, transfer, launch and verify; default: both')
    parser.add_argument('--dataset-root',type=Path,default=Path('/home/pengguanqi/Datasets'))
    parser.add_argument('--remote-dataset-root',type=Path,default=Path('/home/pengguanqi/Datasets'),
                        help='H200 dataset directory; independent of local symlink targets')
    parser.add_argument('--config-dir',type=Path,default=Path('outputs/task_datasets_20261110/training_prepare'))
    parser.add_argument('--remote-root',default=REMOTE_ROOT)
    parser.add_argument('--gpus-per-task',type=int,choices=(1,2,3),default=1,
                        help='DDP ranks per model; each task receives a disjoint group of idle H200 GPUs')
    parser.add_argument('--gpu-indices',type=int,nargs='+',help='Ordered GPU index allowlist; busy GPUs remain unavailable')
    parser.add_argument('--shared-gpu-min-free-mib',type=int,
                        help='Explicitly allow shared GPUs from --gpu-indices when this much memory remains free; default requires idle GPUs')
    parser.add_argument('--per-device-batch-size',type=int,
                        help='Override each GPU batch; for two GPUs use 32 to preserve total batch 64')
    parser.add_argument('--run-suffix',default='',help='Suffix for a fresh run; existing runs are never overwritten')
    parser.add_argument('--remote-run-root',type=Path,help='H200 training output directory; defaults to <remote-root>/outputs/model_training')
    parser.add_argument('--checkpoint-reserve-gib',type=float,default=32.,
                        help='Total disk budget per active/new model run, independent of DDP rank count')
    parser.add_argument('--storage-margin-gib',type=float,default=1.,help='Additional free disk margin for training outputs')
    parser.add_argument('--status',type=Path,default=Path('outputs/task_datasets_20261110/training_prepare/deployment_status.json'))
    operation=parser.add_mutually_exclusive_group()
    operation.add_argument('--launch',action='store_true',help='Deploy/start and verify a finite optimizer step for selected jobs; otherwise inspect only')
    operation.add_argument('--stage-only',action='store_true',
                           help='Transfer and CPU-validate datasets/configs while GPUs are busy; never start training')
    args=parser.parse_args(argv)
    if not args.remote_dataset_root.is_absolute():
        parser.error('--remote-dataset-root must be an absolute remote path')
    tasks=selected_tasks(dict(tasks=args.tasks))
    try:
        settings=launch_settings(dict(launch_settings=dict(gpus_per_task=args.gpus_per_task,
            per_device_batch_size=args.per_device_batch_size,run_suffix=args.run_suffix,gpu_indices=args.gpu_indices)))
        shared_minimum=shared_gpu_min_free_mib(dict(launch_settings=settings,
            shared_gpu_min_free_mib=args.shared_gpu_min_free_mib))
        storage=storage_settings(dict(remote_root=args.remote_root,
            remote_run_root=str(args.remote_run_root) if args.remote_run_root else None,
            checkpoint_reserve_gib=args.checkpoint_reserve_gib,storage_margin_gib=args.storage_margin_gib))
    except ValueError as error:
        parser.error(str(error))
    status=dict(status='checking',updated_unix=time.time(),host=args.host,launch_requested=args.launch,
                stage_only=args.stage_only,selected_tasks=tasks)
    status['launch_settings']=settings
    status['storage_settings']=storage
    status['shared_gpu_min_free_mib']=shared_minimum
    try:
        datasets=[audit_gate(args.dataset_root/f'pressb_{task}_500_20261110',task) for task in tasks]
        for data in datasets:
            data['remote_path']=str(args.remote_dataset_root/f'pressb_{data["task"]}_500_20261110')
        root=Path(__file__).resolve().parents[1]
        payload,files,key=snapshot(root,args.config_dir.resolve(),args.status.parent/'source_snapshots',settings)
        request=dict(remote_root=args.remote_root,tasks=tasks,datasets=datasets,source_files=files,source_sha256=key,
                     launch_settings=settings,stage_only=args.stage_only,shared_gpu_min_free_mib=shared_minimum,**storage)
        status.update(datasets=datasets,status='remote_preflight')
        write_json(args.status,status)
        probe=remote_call(args.host,'remote-probe',request)
        status['remote']=probe
        if args.stage_only:
            if probe.get('processes_started') or any(probe.get('jobs',{}).get(task) is not None for task in tasks):
                raise RuntimeError('stage-only requires unstarted tasks; existing training jobs are left untouched')
            if probe.get('all_staged'):
                status.update(status='staged',updated_unix=time.time())
                write_json(args.status,status);print(json.dumps(status,indent=2));return 0
        if (not args.launch and not args.stage_only) or probe['complete']:
            status.update(status='jobs_started' if probe['complete'] else
                          ('waiting_first_step' if probe['processes_started'] else 'ready'),updated_unix=time.time())
            write_json(args.status,status);print(json.dumps(status,indent=2));return 0
        if probe['processes_started']:
            raise RetryLater('Selected SFT processes started; waiting for finite optimizer-step evidence from each task')
        # Source changes while a task is running must never mutate its imported modules.
        for existing in probe['jobs'].values():
            if existing and existing['source_files']!=files:
                raise RuntimeError('Running deployment has a different source snapshot; use a new remote root')
        if not probe.get('all_staged'):
            status.update(status='transferring',source_sha256=key,source_bytes=directory_bytes(payload))
            write_json(args.status,status)
            mkdir=shlex.join(['mkdir','-p',args.remote_root,str(args.remote_dataset_root)])
            subprocess.run(['ssh',*SSH_OPTIONS,args.host,mkdir],check=True,timeout=60)
            log=args.status.with_suffix('.transfer.log')
            transport=shlex.join(['ssh',*SSH_OPTIONS])
            with log.open('a') as stream:
                pairs=[(payload,args.remote_root)]+[(Path(d['local_path']),d['remote_path']) for d in datasets]
                for source,destination in pairs:
                    command=['rsync','-a','--checksum','--partial','--protect-args','--stats','-e',transport,str(source)+'/',f'{args.host}:{destination}/']
                    result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
                    if result.returncode:raise RetryLater(f'rsync exit {result.returncode}; see {log}')
        if args.stage_only:
            status.update(status='validating_stage',updated_unix=time.time());write_json(args.status,status)
            status['remote']=remote_call(args.host,'remote-stage',request)
            if not status['remote'].get('all_staged'):
                raise RetryLater('Staging has not finished validation')
            status.update(status='staged',updated_unix=time.time())
            write_json(args.status,status);print(json.dumps(status,indent=2));return 0
        status.update(status='launching',updated_unix=time.time());write_json(args.status,status)
        status['remote']=remote_call(args.host,'remote-ensure',request)
        if not status['remote']['complete']:
            raise RetryLater('SFT launch pending: waiting for available GPUs or verified finite first optimizer steps')
        status.update(status='jobs_started',updated_unix=time.time())
        write_json(args.status,status);print(json.dumps(status,indent=2));return 0
    except (RetryLater,subprocess.TimeoutExpired,subprocess.CalledProcessError) as error:
        status.update(status='retryable',error=str(error),retry_after_seconds=120,updated_unix=time.time())
        write_json(args.status,status);print(json.dumps(status,indent=2));return 75
    except Exception as error:
        status.update(status='failed',error=str(error),updated_unix=time.time())
        write_json(args.status,status);print(json.dumps(status,indent=2));return 2


if __name__=='__main__':
    try:
        if len(sys.argv)>1 and sys.argv[1].startswith('remote-'):
            mode=sys.argv[1]
            if mode=='remote-worker':result=remote_worker(sys.argv[2])
            else:
                request=json.loads(base64.b64decode(sys.argv[2]))
                result={'remote-probe':remote_probe,'remote-ensure':remote_ensure,'remote-stage':remote_stage}[mode](request)
            print(json.dumps(result));raise SystemExit(0)
        raise SystemExit(local_main(sys.argv[1:]))
    except RetryLater as error:
        print(str(error),file=sys.stderr);raise SystemExit(75)
