"""Audited-data, retry, and no-duplicate-start gates for the H200 SFT pipeline."""
import importlib.util
import json
from pathlib import Path

import pytest

SPEC=importlib.util.spec_from_file_location('launch_manipulation_sft',Path(__file__).resolve().parents[1]/'scripts/launch_manipulation_sft.py')
launcher=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def publish_metadata(root, task='cube_stack'):
    root=Path(root);(root/'meta').mkdir(parents=True)
    launcher.write_json(root/'meta/info.json',dict(codebase_version='v3.0',fps=30,total_episodes=500,total_frames=1000))
    launcher.write_json(root/'meta/export_manifest.json',dict(task_id=task,episodes=[dict(success=True) for _ in range(500)]))
    launcher.write_json(root/'meta/audit.json',dict(success=True,full_video_decode=True,episodes=500,frames=1000,
        numeric_rows_compared=1000,decoded_frames={f'observation.images.{c}':1000 for c in ('up','wrist')}))
    return root


def training_evidence(run, *, step=1, **metrics):
    run=Path(run);run.mkdir(parents=True,exist_ok=True)
    record=dict(event='train',step=step,action_loss=.04,wm_loss_weighted_0_1=.015,grad_norm=.8,**{})
    record.update(metrics)
    (run/'train.jsonl').write_text(json.dumps(record)+'\n')
    launcher.write_json(run/'status.json',dict(status='training',**record))
    return record


def test_audit_gate_requires_both_camera_counts_and_correct_task(tmp_path):
    root=publish_metadata(tmp_path/'dataset')
    result=launcher.audit_gate(root,'cube_stack')
    assert result['frames']==1000 and set(result['metadata_sha256'])=={'audit.json','info.json','export_manifest.json'}
    with pytest.raises(ValueError,match='audit'):
        launcher.audit_gate(root,'tabletop_cleanup')
    audit=launcher.read_json(root/'meta/audit.json')
    audit['decoded_frames']['observation.images.wrist']=999
    launcher.write_json(root/'meta/audit.json',audit)
    with pytest.raises(ValueError,match='camera'):
        launcher.audit_gate(root,'cube_stack')


def test_audit_records_local_resolved_path_without_resolving_remote_alias(tmp_path):
    physical=publish_metadata(tmp_path/'mounted_data'/'cube')
    alias=tmp_path/'home_datasets'/'cube';alias.parent.mkdir()
    alias.symlink_to(physical,target_is_directory=True)
    result=launcher.audit_gate(alias,'cube_stack')
    assert result['local_path']==str(physical)
    assert result['remote_path']==str(alias)


def test_python_probe_retains_module_error_and_stderr(tmp_path,monkeypatch):
    from subprocess import CompletedProcess
    candidate=tmp_path/'miniconda3/envs/vlajepa-piper/bin/python'
    candidate.parent.mkdir(parents=True);candidate.touch()
    report=dict(ok=False,checks=[dict(module='av',ok=False,error='PyAV ABI failure')])
    monkeypatch.setattr(launcher.subprocess,'run',lambda *args,**kwargs:CompletedProcess(args,1,json.dumps(report),'library diagnostic'))
    with pytest.raises(RuntimeError) as caught:
        launcher.discover_model_python(tmp_path)
    assert 'PyAV ABI failure' in str(caught.value) and 'library diagnostic' in str(caught.value)


def test_python_probe_uses_actual_av_reader_dependency_not_decord(tmp_path,monkeypatch):
    from subprocess import CompletedProcess
    candidate=tmp_path/'miniconda3/envs/vlajepa-piper/bin/python'
    candidate.parent.mkdir(parents=True);candidate.touch()
    def check(command,**kwargs):
        assert "('av',[])" in command[-1] and 'decord' not in command[-1]
        return CompletedProcess(command,0,json.dumps(dict(ok=True,checks=[dict(module='av',ok=True)])),'')
    monkeypatch.setattr(launcher.subprocess,'run',check)
    python,diagnostics=launcher.discover_model_python(tmp_path)
    assert python==str(candidate) and diagnostics[0]['ok']


def test_incomplete_export_is_retryable(tmp_path):
    with pytest.raises(launcher.RetryLater,match='not finalized'):
        launcher.audit_gate(tmp_path,'cube_stack')


def test_quota_uses_soft_limit_and_overquota_marker():
    text='''Disk quotas for user pengguanqi (uid 2002):
Filesystem blocks quota limit grace files quota limit grace
/dev/md0 292736056 293601280 314572800 0 201765 0 0 0
/dev/sda 100* 90 120 0 0 0 0 0
/dev/unlimited 400 0 0 0 0 0 0 0
'''
    assert launcher.quota_rows(text)=={'/dev/md0':885989376,'/dev/sda':0,'/dev/unlimited':None}


def test_combined_reservation_preserves_running_sibling_budget(tmp_path,monkeypatch):
    jobs={task:None for task in launcher.TASKS}
    assert launcher.checkpoint_reserve_bytes(tmp_path,jobs)==65*launcher.GIB
    jobs['cube_stack']={'status':'running'}
    monkeypatch.setattr(launcher,'directory_bytes',lambda p:8*launcher.GIB if 'cube_stack' in str(p) else 0)
    assert launcher.checkpoint_reserve_bytes(tmp_path,jobs)==57*launcher.GIB
    jobs['cube_stack']['status']='completed'
    assert launcher.checkpoint_reserve_bytes(tmp_path,jobs)==33*launcher.GIB


def test_selected_budget_excludes_unstarted_sibling_and_keeps_active_sibling(tmp_path,monkeypatch):
    jobs={task:None for task in launcher.TASKS}
    assert launcher.checkpoint_reserve_bytes(tmp_path,jobs,['cube_stack'])==33*launcher.GIB
    assert launcher.checkpoint_reserve_bytes(tmp_path,jobs)==65*launcher.GIB
    jobs['tabletop_cleanup']={'status':'running'}
    monkeypatch.setattr(launcher,'directory_bytes',lambda p:8*launcher.GIB if 'tabletop_cleanup' in str(p) else 0)
    assert launcher.checkpoint_reserve_bytes(tmp_path,jobs,['cube_stack'])==57*launcher.GIB
    for state in ('completed','failed'):
        jobs['tabletop_cleanup']['status']=state
        assert launcher.checkpoint_reserve_bytes(tmp_path,jobs,['cube_stack'])==33*launcher.GIB
    jobs['cube_stack']={'status':'completed'}
    assert launcher.checkpoint_reserve_bytes(tmp_path,jobs,['cube_stack'])==0


def test_existing_jobs_skip_resource_and_launch_actions(tmp_path,monkeypatch):
    receipts={task:dict(status='completed',task=task,run=str(tmp_path/task)) for task in launcher.TASKS}
    for task,row in receipts.items():
        training_evidence(row['run'])
        launcher.write_json(tmp_path/'outputs/launch_receipts'/f'{task}.json',row)
    def forbidden(*args,**kwargs):
        raise AssertionError('Existing jobs must not probe GPUs or launch subprocesses')
    monkeypatch.setattr(launcher.subprocess,'run',forbidden)
    result=launcher.remote_probe(dict(remote_root=str(tmp_path)))
    assert result['complete'] and result['first_steps_verified'] and result['processes_started']
    assert result['jobs']==receipts


def test_running_but_initializing_waits_without_resource_probe_or_relaunch(tmp_path,monkeypatch):
    for task in launcher.TASKS:
        run=tmp_path/task
        launcher.write_json(run/'status.json',dict(status='initializing'))
        launcher.write_json(tmp_path/'outputs/launch_receipts'/f'{task}.json',dict(status='running',task=task,run=str(run)))
    monkeypatch.setattr(launcher,'process_matches',lambda _:True)
    def forbidden(*args,**kwargs):
        raise AssertionError('Initializing jobs must not be duplicated or probe for new GPUs')
    monkeypatch.setattr(launcher.subprocess,'run',forbidden)
    request=dict(remote_root=str(tmp_path))
    for probe in (launcher.remote_probe(request),launcher.remote_ensure(request)):
        assert probe['processes_started'] and not probe['first_steps_verified'] and not probe['complete']
        assert all(row['reason']=='awaiting_first_optimizer_step' for row in probe['readiness'].values())


def test_first_finite_optimizer_step_and_status_are_both_required(tmp_path):
    receipt=dict(status='running',task='cube_stack',run=str(tmp_path))
    assert not launcher.training_readiness(receipt)['ready']
    record=training_evidence(tmp_path)
    assert launcher.training_readiness(receipt)['ready']
    launcher.write_json(tmp_path/'status.json',dict(status='initializing'))
    assert not launcher.training_readiness(receipt)['ready']
    launcher.write_json(tmp_path/'status.json',dict(status='training',**record))
    result=launcher.training_readiness(receipt)
    assert result['first_verified_step']==1 and result['first_step_metrics']['grad_norm']==.8


@pytest.mark.parametrize('metric',['action_loss','wm_loss_weighted_0_1','grad_norm'])
def test_nan_loss_or_gradient_is_hard_failure(tmp_path,metric):
    training_evidence(tmp_path,**{metric:float('nan')})
    with pytest.raises(RuntimeError,match='Nonfinite'):
        launcher.training_readiness(dict(status='running',run=str(tmp_path)))


def test_status_nan_after_first_good_step_is_hard_failure(tmp_path):
    record=training_evidence(tmp_path)
    record['grad_norm']=float('inf')
    launcher.write_json(tmp_path/'status.json',dict(status='training',**record))
    with pytest.raises(RuntimeError,match='Nonfinite'):
        launcher.training_readiness(dict(status='running',run=str(tmp_path)))


def test_failed_job_never_automatically_restarts(tmp_path):
    launcher.write_json(tmp_path/'outputs/launch_receipts/cube_stack.json',dict(status='failed',log='problem.log'))
    with pytest.raises(RuntimeError,match='problem.log'):
        launcher.remote_probe(dict(remote_root=str(tmp_path)))


def test_dead_supervisor_never_silently_restarts(tmp_path):
    path=tmp_path/'receipt.json'
    launcher.write_json(path,dict(status='running',supervisor_pid=999999999,receipt_path=str(path)))
    with pytest.raises(RuntimeError,match='without a completion'):
        launcher.job_state(path)


def test_ssh_failure_returns_retryable_status_without_transfer(tmp_path,monkeypatch):
    monkeypatch.setattr(launcher,'audit_gate',lambda root,task:dict(task=task,path=str(root),bytes=100))
    monkeypatch.setattr(launcher,'snapshot',lambda *args:(tmp_path,{},'hash'))
    def unavailable(*args):
        raise launcher.RetryLater('SSH banner timeout')
    monkeypatch.setattr(launcher,'remote_call',unavailable)
    status=tmp_path/'status.json'
    assert launcher.local_main(['--launch','--status',str(status)])==75
    result=json.loads(status.read_text())
    assert result['status']=='retryable' and result['retry_after_seconds']==120
    assert result['error']=='SSH banner timeout'


def test_initializing_processes_return_75_and_do_not_retransfer(tmp_path,monkeypatch):
    monkeypatch.setattr(launcher,'audit_gate',lambda root,task:dict(task=task,path=str(root),bytes=100))
    monkeypatch.setattr(launcher,'snapshot',lambda *args:(tmp_path,{},'hash'))
    monkeypatch.setattr(launcher,'remote_call',lambda *args:dict(processes_started=True,first_steps_verified=False,complete=False))
    status=tmp_path/'status.json'
    assert launcher.local_main(['--launch','--status',str(status)])==75
    result=json.loads(status.read_text())
    assert result['status']=='retryable' and result['remote']['processes_started']
    assert not result['remote']['first_steps_verified']


def test_selected_cube_does_not_require_cleanup_dataset(tmp_path,monkeypatch):
    root=tmp_path/'datasets'
    publish_metadata(root/'pressb_cube_stack_500_20261110')
    assert not (root/'pressb_tabletop_cleanup_500_20261110').exists()
    monkeypatch.setattr(launcher,'snapshot',lambda *args:(tmp_path,{},'same-two-config-snapshot'))
    calls=[]
    def checked_remote(host,mode,request):
        calls.append(mode)
        assert request['tasks']==['cube_stack']
        assert [d['task'] for d in request['datasets']]==['cube_stack']
        return dict(processes_started=True,first_steps_verified=True,complete=True,selected_tasks=['cube_stack'])
    monkeypatch.setattr(launcher,'remote_call',checked_remote)
    status=tmp_path/'status.json'
    assert launcher.local_main(['--tasks','cube_stack','--launch','--dataset-root',str(root),'--status',str(status)])==0
    assert calls==['remote-probe'] and launcher.read_json(status)['selected_tasks']==['cube_stack']


def test_resolved_local_dataset_never_leaks_into_remote_path(tmp_path,monkeypatch):
    mounted=tmp_path/'mnt/data/datasets'
    publish_metadata(mounted/'pressb_cube_stack_500_20261110')
    alias=tmp_path/'home/datasets';alias.parent.mkdir(parents=True);alias.symlink_to(mounted,target_is_directory=True)
    monkeypatch.setattr(launcher,'snapshot',lambda *args:(tmp_path,{},'snapshot'))
    def check(host,mode,request):
        data=request['datasets'][0]
        assert data['local_path']==str(mounted/'pressb_cube_stack_500_20261110')
        assert data['remote_path']=='/home/pengguanqi/Datasets/pressb_cube_stack_500_20261110'
        return dict(processes_started=True,complete=True)
    monkeypatch.setattr(launcher,'remote_call',check)
    # The background pipeline resolves its local root before calling this CLI.
    assert launcher.local_main(['--tasks','cube_stack','--launch','--dataset-root',str(alias.resolve()),
        '--status',str(tmp_path/'status.json')])==0


@pytest.mark.parametrize('cleanup_state',[None,'failed','running'])
def test_selected_readiness_keeps_unselected_receipt_but_does_not_gate_it(tmp_path,cleanup_state):
    run=tmp_path/'cube_run';training_evidence(run)
    cube=dict(status='completed',task='cube_stack',run=str(run))
    launcher.write_json(tmp_path/'outputs/launch_receipts/cube_stack.json',cube)
    cleanup=None
    if cleanup_state:
        cleanup=dict(status=cleanup_state,task='tabletop_cleanup',run='not-ready',supervisor_pid=999999999)
        launcher.write_json(tmp_path/'outputs/launch_receipts/tabletop_cleanup.json',cleanup)
    request=dict(remote_root=str(tmp_path),tasks=['cube_stack'])
    for result in (launcher.remote_probe(request),launcher.remote_ensure(request)):
        assert result['complete'] and result['selected_tasks']==['cube_stack']
        assert list(result['readiness'])==['cube_stack']
        assert result['jobs']['tabletop_cleanup']==cleanup
    # The default two-task request still reserves the combined budget.
    assert launcher.checkpoint_reserve_bytes(tmp_path,{'cube_stack':None,'tabletop_cleanup':None})==65*launcher.GIB


def test_existing_job_refuses_changed_dataset(tmp_path):
    launcher.write_json(tmp_path/'outputs/launch_receipts/cube_stack.json',
                        dict(status='completed',dataset_metadata_sha256={'audit.json':'old'}))
    with pytest.raises(RuntimeError,match='different dataset'):
        launcher.remote_probe(dict(remote_root=str(tmp_path),datasets=[dict(task='cube_stack',metadata_sha256={'audit.json':'new'})]))


def test_source_snapshot_includes_only_required_model(tmp_path):
    root=Path(__file__).resolve().parents[1]
    cfg=tmp_path/'configs';cfg.mkdir()
    for task in launcher.TASKS:(cfg/f'{task}.yaml').write_text('run_id: example\n')
    destination,files,key=launcher.snapshot(root,cfg,tmp_path/'snapshots')
    assert len(key)==64 and launcher.directory_bytes(destination)<8*1024**2
    assert 'src/pressb/training/vlajepa_joint7_data.py' in files
    assert 'src/pressb/model_sources/vlajepa/starVLA/training/train_joint7.py' in files
    assert not any('walloss/' in p or '/tasks/' in p or '__pycache__' in p for p in files)
    second,second_files,second_key=launcher.snapshot(root,cfg,tmp_path/'snapshots')
    assert (destination,files,key)==(second,second_files,second_key)


def test_ddp_gpu_groups_are_disjoint_and_never_partial():
    devices=[dict(index=i,uuid=f'GPU-{i}') for i in range(5)]
    groups=launcher.allocate_gpu_groups(devices,list(launcher.TASKS),2)
    assert [[gpu['index'] for gpu in groups[task]] for task in launcher.TASKS]==[[0,1],[2,3]]
    assert list(launcher.allocate_gpu_groups(devices[:3],list(launcher.TASKS),2))==['cube_stack']
    with pytest.raises(launcher.RetryLater,match='Need 2'):
        launcher.allocate_gpu_groups(devices[:1],['cube_stack'],2)
    with pytest.raises(ValueError,match='Duplicate'):
        launcher.allocate_gpu_groups([devices[0],devices[0]],['cube_stack'],2)


@pytest.mark.parametrize('world',[1,2,3])
def test_training_command_uses_all_ranks_and_uuid_mask(world):
    receipt=dict(root='/work/pressb',python='/env/bin/python',config='/work/config.yaml',
                 gpu_uuids=[f'GPU-{i}' for i in range(world)],world_size=world)
    command,mask=launcher.training_command(receipt)
    assert mask==','.join(receipt['gpu_uuids'])
    assert command[-2:]==['--config',receipt['config']]
    if world>1:
        assert command[:5]==['/env/bin/python','-m','torch.distributed.run','--standalone',f'--nproc-per-node={world}']
    else:
        assert command[:2]==['/env/bin/python','/work/pressb/scripts/train_model.py']
    receipt['gpu_uuids']*=2
    with pytest.raises(ValueError,match='distinct GPU'):
        launcher.training_command(receipt)


def test_legacy_single_gpu_receipt_keeps_direct_python_command():
    command,mask=launcher.training_command(dict(root='/work',python='python',config='config.yaml',gpu_uuid='GPU-old'))
    assert command[:2]==['python','/work/scripts/train_model.py'] and mask=='GPU-old'


@pytest.mark.parametrize('settings',[
    dict(gpus_per_task=0),dict(gpus_per_task=4),dict(gpus_per_task=True),
    dict(per_device_batch_size=0),dict(per_device_batch_size=-1),dict(run_suffix='../old'),
    dict(gpu_indices=[1,1]),dict(gpu_indices=[-1,2]),dict(gpu_indices=[True,2]),
    dict(gpu_indices=[]),dict(gpu_indices='1,2'),dict(gpus_per_task=2,gpu_indices=[1]),
])
def test_invalid_launch_geometry_is_rejected(settings):
    with pytest.raises(ValueError):launcher.launch_settings(dict(launch_settings=settings))


def test_existing_single_gpu_run_is_not_ddp_ready(tmp_path):
    run=tmp_path/'run';training_evidence(run)
    launcher.write_json(tmp_path/'outputs/launch_receipts/cube_stack.json',
        dict(status='completed',task='cube_stack',run=str(run)))
    with pytest.raises(RuntimeError,match='different GPU/batch/run'):
        launcher.remote_probe(dict(remote_root=str(tmp_path),tasks=['cube_stack'],
            launch_settings=dict(gpus_per_task=2,per_device_batch_size=32,run_suffix='ddp2')))


@pytest.mark.parametrize('changed',[dict(gpus_per_task=1),dict(per_device_batch_size=64),dict(run_suffix='other')])
def test_ddp_retry_requires_identical_launch_settings(tmp_path,changed):
    run=tmp_path/'outputs/model_training/vlajepa_cube_stack_500_20261110_ddp2';training_evidence(run)
    settings=dict(gpus_per_task=2,per_device_batch_size=32,run_suffix='ddp2')
    launcher.write_json(tmp_path/'outputs/launch_receipts/cube_stack.json',dict(status='completed',task='cube_stack',
        run=str(run),world_size=2,launch_settings=settings,gpu_uuids=['GPU-1','GPU-2']))
    request=dict(remote_root=str(tmp_path),tasks=['cube_stack'],launch_settings=settings)
    assert launcher.remote_probe(request)['complete']
    request['launch_settings']={**settings,**changed}
    with pytest.raises(RuntimeError,match='different GPU/batch/run'):launcher.remote_probe(request)


def test_ddp_checkpoint_budget_follows_actual_run_and_has_one_rank0_copy(tmp_path,monkeypatch):
    observed=[]
    def size(path):
        observed.append(Path(path));return 8*launcher.GIB
    monkeypatch.setattr(launcher,'directory_bytes',size)
    actual=tmp_path/'elsewhere/custom_ddp_run'
    jobs={'cube_stack':dict(status='running',run=str(actual),world_size=2),'tabletop_cleanup':None}
    assert launcher.checkpoint_reserve_bytes(tmp_path,jobs,['cube_stack'],'ddp2')==25*launcher.GIB
    assert observed==[actual]
    observed.clear();jobs['cube_stack']=None
    launcher.checkpoint_reserve_bytes(tmp_path,jobs,['cube_stack'],'ddp2')
    assert observed[0].name=='vlajepa_cube_stack_500_20261110_ddp2'


def test_ddp_snapshot_freezes_overridden_batch_run_and_settings(tmp_path):
    import yaml
    root=Path(__file__).resolve().parents[1]
    cfg=tmp_path/'configs';cfg.mkdir()
    original='run_id: original\ndatasets:\n  vla_data:\n    per_device_batch_size: 64\n'
    for task in launcher.TASKS:(cfg/f'{task}.yaml').write_text(original)
    settings=launcher.launch_settings(dict(launch_settings=dict(gpus_per_task=2,per_device_batch_size=32,run_suffix='ddp2')))
    target,files,key=launcher.snapshot(root,cfg,tmp_path/'snapshots',settings)
    assert launcher.read_json(target/'training_prepare/launch_settings.json')==settings
    for task in launcher.TASKS:
        config=yaml.safe_load((target/'training_prepare'/f'{task}.yaml').read_text())
        assert config['datasets']['vla_data']['per_device_batch_size']==32
        assert config['run_id']==launcher.task_run_id(task,'ddp2')
        assert (cfg/f'{task}.yaml').read_text()==original
    other,other_files,other_key=launcher.snapshot(root,cfg,tmp_path/'snapshots',{**settings,'gpus_per_task':3})
    assert key!=other_key and files!=other_files


def test_cli_forwards_ddp_settings_and_remote_run_root(tmp_path,monkeypatch):
    monkeypatch.setattr(launcher,'audit_gate',lambda root,task:dict(task=task,path=str(root),bytes=100))
    def capture_snapshot(root,config_dir,output,settings):
        assert settings==dict(gpus_per_task=2,per_device_batch_size=32,run_suffix='ddp2',gpu_indices=[1,2])
        return tmp_path,{},'snapshot'
    monkeypatch.setattr(launcher,'snapshot',capture_snapshot)
    def remote(host,mode,request):
        assert request['launch_settings']==dict(gpus_per_task=2,per_device_batch_size=32,run_suffix='ddp2',gpu_indices=[1,2])
        assert request['remote_root']=='/work/new-ddp-root'
        assert request['remote_run_root']=='/data/training'
        assert request['checkpoint_reserve_gib']==19.127 and request['storage_margin_gib']==.25
        assert request['shared_gpu_min_free_mib']==58000
        assert 'shared_gpu_min_free_mib' not in request['launch_settings']
        return dict(processes_started=True,complete=True)
    monkeypatch.setattr(launcher,'remote_call',remote)
    assert launcher.local_main(['--tasks','cube_stack','--launch','--gpus-per-task','2',
        '--gpu-indices','1','2',
        '--shared-gpu-min-free-mib','58000',
        '--per-device-batch-size','32','--run-suffix','ddp2','--remote-root','/work/new-ddp-root',
        '--remote-run-root','/data/training','--checkpoint-reserve-gib','19.127','--storage-margin-gib','.25',
        '--status',str(tmp_path/'status.json')])==0


def test_checkpoint_reservations_stay_on_each_runs_filesystem(tmp_path,monkeypatch):
    monkeypatch.setattr(launcher,'directory_bytes',lambda _:0)
    jobs={'cube_stack':dict(status='running',run='/home/user/run/cube'),'tabletop_cleanup':None}
    settings=launcher.launch_settings({})
    storage=launcher.storage_settings(dict(remote_root='/home/user/code',remote_run_root='/data/runs',
        checkpoint_reserve_gib=19.127,storage_margin_gib=.25))
    needs=dict(launcher.checkpoint_storage_needs(tmp_path,jobs,['tabletop_cleanup'],settings,storage))
    assert set(needs)=={Path('/home/user/run'),Path('/data/runs')}
    assert all(size==launcher.math.ceil(19.127*launcher.GIB)+launcher.GIB//4 for size in needs.values())


def test_smaller_cleanup_budget_does_not_shrink_running_cube_budget(tmp_path,monkeypatch):
    monkeypatch.setattr(launcher,'directory_bytes',lambda _:0)
    cube_storage=dict(checkpoint_reserve_gib=32,storage_margin_gib=1,remote_run_root='/home/runs')
    cleanup_storage=dict(checkpoint_reserve_gib=19.13,storage_margin_gib=.25,remote_run_root='/data/runs')
    jobs={'cube_stack':dict(status='running',run='/home/runs/cube',storage_settings=cube_storage),'tabletop_cleanup':None}
    needs=dict(launcher.checkpoint_storage_needs(tmp_path,jobs,['tabletop_cleanup'],launcher.launch_settings({}),cleanup_storage))
    assert needs[Path('/home/runs')]==33*launcher.GIB
    assert needs[Path('/data/runs')]==launcher.math.ceil(19.13*launcher.GIB)+launcher.GIB//4


@pytest.mark.parametrize('values',[dict(checkpoint_reserve_gib=0),dict(checkpoint_reserve_gib=float('nan')),
    dict(storage_margin_gib=-1),dict(remote_run_root='relative')])
def test_invalid_storage_options_are_rejected(values):
    with pytest.raises(ValueError):launcher.storage_settings(dict(remote_root='/work',**values))


@pytest.mark.parametrize('busy_second_gpu',[False,True])
@pytest.mark.parametrize('pre_staged',[False,True])
@pytest.mark.parametrize('shared_minimum',[None,58000])
def test_remote_ensure_launches_full_group_and_rechecks_every_gpu(tmp_path,monkeypatch,busy_second_gpu,pre_staged,shared_minimum):
    from types import SimpleNamespace
    settings=dict(gpus_per_task=2,per_device_batch_size=32,run_suffix='ddp2',gpu_indices=[1,2])
    request=dict(remote_root=str(tmp_path),remote_run_root=str(tmp_path/'separate_disk/runs'),tasks=['cube_stack'],
        launch_settings=settings,shared_gpu_min_free_mib=shared_minimum,
        source_files={},datasets=[dict(task='cube_stack',remote_path='/data/cube',metadata_sha256={})])
    devices=[dict(index=i,uuid=f'GPU-{i}') for i in (1,2)]
    staged={}
    if pre_staged:
        config=tmp_path/'training_prepare/cube_stack.resolved.yaml';config.parent.mkdir();config.write_text('staged config')
        staged['cube_stack']=dict(config=str(config),preflight=dict(per_device_batch_size=32,global_batch_size=64))
    monkeypatch.setattr(launcher,'remote_probe',lambda _:dict(processes_started=False,jobs={t:None for t in launcher.TASKS},
        python='python',assets={},gpus=devices,stages=staged))
    def resolve(command,**kwargs):
        assert not pre_staged,'A verified staged task must not repeat CPU preflight'
        path=Path(command[4]);path.parent.mkdir(parents=True,exist_ok=True);path.write_text('resolved config')
        assert command[6]==str(tmp_path/'separate_disk/runs')
    monkeypatch.setattr(launcher.subprocess,'run',resolve)
    def preflight(root,python,config,task,world):
        assert not pre_staged
        assert world==2
        return dict(per_device_batch_size=32,global_batch_size=64)
    monkeypatch.setattr(launcher,'model_preflight',preflight)
    inspected=[]
    def query(command,**kwargs):
        inspected.append(command[1])
        if busy_second_gpu and command[1]=='--id=GPU-2':return '143771, 100000, 90'
        return '143771, 70000, 99' if shared_minimum else '143771, 0, 0'
    monkeypatch.setattr(launcher.subprocess,'check_output',query)
    starts=[]
    def launch(command,**kwargs):
        starts.append(command)
        path=Path(command[-1]);row=launcher.read_json(path);row['status']='running';launcher.write_json(path,row)
        return SimpleNamespace(poll=lambda:None)
    monkeypatch.setattr(launcher.subprocess,'Popen',launch)
    monkeypatch.setattr(launcher,'process_matches',lambda _:True)
    if busy_second_gpu:
        with pytest.raises(launcher.RetryLater,match='GPU-2 became busy'):launcher.remote_ensure(request)
        assert not starts and not (tmp_path/'outputs/launch_receipts/cube_stack.json').exists()
    else:
        result=launcher.remote_ensure(request);row=result['jobs']['cube_stack']
        assert len(starts)==1 and row['gpu_uuids']==['GPU-1','GPU-2'] and row['world_size']==2
        assert row['global_batch_size']==64 and row['per_device_batch_size']==32
        assert row['shared_gpu_min_free_mib']==shared_minimum
        assert len(row['gpu_admission_samples'])==2
        assert row['run']==str(tmp_path/'separate_disk/runs/vlajepa_cube_stack_500_20261110_ddp2')
    assert inspected==['--id=GPU-1','--id=GPU-2']


def test_explicit_gpu_pool_preserves_order_and_never_reintroduces_busy_devices():
    available=[dict(index=i,uuid=f'GPU-{i}') for i in (0,1,2)]
    settings=launcher.launch_settings(dict(launch_settings=dict(gpus_per_task=2,gpu_indices=[2,1])))
    eligible=launcher.selected_gpu_inventory(available,settings)
    assert [gpu['index'] for gpu in eligible]==[2,1]
    groups=launcher.allocate_gpu_groups(eligible,['cube_stack'],2)
    assert [gpu['uuid'] for gpu in groups['cube_stack']]==['GPU-2','GPU-1']
    eligible=launcher.selected_gpu_inventory(available[:2],settings)
    assert [gpu['index'] for gpu in eligible]==[1]
    with pytest.raises(launcher.RetryLater,match='only 1 available'):
        launcher.allocate_gpu_groups(eligible,['cube_stack'],2)
    assert launcher.selected_gpu_inventory(available,launcher.launch_settings({}))==available


def test_different_gpu_pool_requires_fresh_launch_identity(tmp_path):
    run=tmp_path/'outputs/model_training/vlajepa_cube_stack_500_20261110_ddp2';training_evidence(run)
    settings=dict(gpus_per_task=2,per_device_batch_size=32,run_suffix='ddp2',gpu_indices=[1,2])
    launcher.write_json(tmp_path/'outputs/launch_receipts/cube_stack.json',dict(status='completed',task='cube_stack',
        run=str(run),world_size=2,launch_settings=settings,gpu_uuids=['GPU-1','GPU-2'],gpu_indices=[1,2]))
    request=dict(remote_root=str(tmp_path),tasks=['cube_stack'],launch_settings=settings)
    assert launcher.remote_probe(request)['complete']
    request['launch_settings']={**settings,'gpu_indices':[0,1]}
    with pytest.raises(RuntimeError,match='different GPU/batch/run'):
        launcher.remote_probe(request)


def test_stage_resource_probe_requires_quota_but_never_idle_gpus(tmp_path,monkeypatch):
    from types import SimpleNamespace
    home=tmp_path/'home'
    for relative in ('Models/Qwen3-VL-2B-Instruct','Models/vjepa2-vitl-fpc64-256',
                     'Models/VLA-JEPA/Pretrain/checkpoints/VLA-JEPA-pretrain.pt'):
        path=home/relative;path.parent.mkdir(parents=True,exist_ok=True);path.touch()
    monkeypatch.setattr(launcher.Path,'home',classmethod(lambda cls:home))
    monkeypatch.setattr(launcher,'discover_model_python',lambda _:('python',[]))
    monkeypatch.setattr(launcher.shutil,'which',lambda _:'/usr/bin/quota')
    monkeypatch.setattr(launcher.shutil,'disk_usage',lambda _:SimpleNamespace(free=1000*launcher.GIB))
    calls=[]
    def query(command,**kwargs):
        calls.append(command)
        if command[0]=='df':return 'Filesystem\n/dev/fake\n'
        assert command[0]=='nvidia-smi'
        return '1, GPU-1, NVIDIA H200, 143771, 100000, 100\n2, GPU-2, NVIDIA H200, 143771, 100000, 100\n'
    monkeypatch.setattr(launcher.subprocess,'check_output',query)
    def quota(command,**kwargs):
        assert command==['quota','-w','-p']
        return SimpleNamespace(returncode=0,stdout='/dev/fake 0 1073741824 1073741824')
    monkeypatch.setattr(launcher.subprocess,'run',quota)
    request=dict(remote_root=str(tmp_path/'deploy'),tasks=['cube_stack'],stage_only=True,
        launch_settings=dict(gpus_per_task=2,gpu_indices=[1,2]),source_files={},
        datasets=[dict(task='cube_stack',remote_path=str(tmp_path/'dataset'),bytes=100,metadata_sha256={})])
    result=launcher.remote_probe(request)
    assert result['gpus']==[] and result['gpu_groups']=={} and result['storage']
    assert not any(command[0]=='nvidia-smi' for command in calls)
    with pytest.raises(launcher.RetryLater,match='Need 2'):
        launcher.remote_probe({**request,'stage_only':False})


@pytest.mark.parametrize('warm_start',[False,True])
def test_stage_cpu_validates_without_launch_receipt_and_detects_later_changes(tmp_path,monkeypatch,warm_start):
    root=tmp_path/'deploy';root.mkdir()
    source=root/'code.py';source.write_text('stable source')
    dataset=publish_metadata(tmp_path/'dataset')
    (dataset/'video.mp4').write_bytes(b'reference video')
    audited=launcher.audit_gate(dataset,'cube_stack')
    checkpoint=tmp_path/'base.pt';checkpoint.write_bytes(b'base checkpoint')
    warm={}
    if warm_start:
        source_run=tmp_path/'previous_run'
        source_checkpoint=source_run/'checkpoints/last_step_019167/model.pt'
        source_checkpoint.parent.mkdir(parents=True)
        source_checkpoint.write_bytes(b'trained joint7 weights')
        (source_checkpoint.parent/'manifest.json').write_text('{}')
        processor=source_run/'processor';processor.mkdir()
        (processor/'tokenizer_config.json').write_text('{}')
        (source_run/'normalization.json').write_text('{}')
        (source_run/'split.json').write_text('{}')
        warm=dict(checkpoint_path=str(source_checkpoint),source_run=str(source_run),processor_path=str(processor),source_step=19167)
    request=dict(remote_root=str(root),tasks=['cube_stack'],source_files={'code.py':launcher.digest(source)},
        datasets=[audited],launch_settings=dict(gpus_per_task=2,per_device_batch_size=32,run_suffix='ddp2',gpu_indices=[1,2]))
    def probe(req):
        assert req['stage_only'] is True
        return dict(jobs={task:None for task in launcher.TASKS},python='python',assets={'checkpoint':str(checkpoint)},
            storage={'/dev/fake':dict(required_bytes=33*launcher.GIB)},stages=launcher.verified_stages(req))
    monkeypatch.setattr(launcher,'remote_probe',probe)
    calls=[]
    def resolve(command,**kwargs):
        config=Path(command[4]);config.parent.mkdir(parents=True,exist_ok=True);config.write_text('resolved')
    monkeypatch.setattr(launcher.subprocess,'run',resolve)
    def preflight(root,python,config,task,world):
        calls.append((task,world))
        return dict(world_size=world,per_device_batch_size=32,global_batch_size=64,
                    **(dict(warm_start=warm,initialization='warm_start_model_only') if warm else dict(base_sha256='checked')))
    monkeypatch.setattr(launcher,'model_preflight',preflight)
    def forbidden(*args,**kwargs):raise AssertionError('Staging must not launch any training process')
    monkeypatch.setattr(launcher.subprocess,'Popen',forbidden)
    first=launcher.remote_stage(request)
    assert first['status']=='staged' and first['all_staged'] and not first['processes_started']
    assert calls==[('cube_stack',2)]
    assert (root/'outputs/stage_receipts/cube_stack.json').exists()
    assert not (root/'outputs/launch_receipts').exists() and not (root/'outputs/model_training').exists()
    assert launcher.remote_stage(request)['all_staged'] and len(calls)==1
    shared_request={**request,'shared_gpu_min_free_mib':58000}
    assert launcher.staged_identity(shared_request,'cube_stack')==launcher.staged_identity(request,'cube_stack')
    assert launcher.verified_stages(shared_request)['cube_stack']['status']=='staged'
    if warm:
        row=launcher.read_json(root/'outputs/stage_receipts/cube_stack.json')
        assert row['initialization_checkpoint']==str(source_checkpoint)
        checkpoint.write_bytes(b'changed unused official base')
        assert launcher.verified_stages(request)['cube_stack'] is not None
        (processor/'tokenizer_config.json').write_text('{"changed":true}')
        assert launcher.verified_stages(request)['cube_stack'] is None
        assert launcher.remote_stage(request)['all_staged']
    (dataset/'video.mp4').write_bytes(b'changed video')
    assert launcher.verified_stages(request)['cube_stack'] is None
    changed={**request,'launch_settings':{**request['launch_settings'],'per_device_batch_size':64}}
    with pytest.raises(RuntimeError,match='different source/config/dataset'):
        launcher.verified_stages(changed)


def test_stage_cli_then_launch_reuses_transfer_and_keeps_same_snapshot_settings(tmp_path,monkeypatch):
    from types import SimpleNamespace
    dataset_root=tmp_path/'datasets';publish_metadata(dataset_root/'pressb_cube_stack_500_20261110')
    snapshots=[]
    def snapshot(root,config_dir,output,settings):
        snapshots.append(settings);return tmp_path,{},'same-source'
    monkeypatch.setattr(launcher,'snapshot',snapshot)
    remote_calls=[]
    def remote(host,mode,request):
        remote_calls.append(mode)
        if mode=='remote-stage':
            assert request['stage_only'] is True
            return dict(status='staged',all_staged=True,processes_started=False)
        if mode=='remote-ensure':
            assert request['stage_only'] is False
            return dict(complete=True,processes_started=True)
        assert mode=='remote-probe'
        return dict(complete=False,processes_started=False,all_staged=not request['stage_only'],jobs={task:None for task in launcher.TASKS})
    monkeypatch.setattr(launcher,'remote_call',remote)
    transfers=[]
    monkeypatch.setattr(launcher.subprocess,'run',lambda command,**kwargs:(transfers.append(command) or SimpleNamespace(returncode=0)))
    args=['--tasks','cube_stack','--dataset-root',str(dataset_root),'--status',str(tmp_path/'status.json'),
          '--gpus-per-task','2','--gpu-indices','1','2','--per-device-batch-size','32','--run-suffix','ddp2']
    assert launcher.local_main([*args,'--stage-only'])==0
    assert launcher.read_json(tmp_path/'status.json')['status']=='staged'
    transferred=len(transfers)
    assert sum(command[0]=='rsync' for command in transfers)==2
    assert launcher.local_main([*args,'--launch'])==0
    assert len(transfers)==transferred
    assert remote_calls==['remote-probe','remote-stage','remote-probe','remote-ensure']
    assert snapshots[0]==snapshots[1] and 'stage_only' not in snapshots[0]


def test_stage_only_cannot_be_combined_with_launch_or_call_ensure(tmp_path):
    with pytest.raises(SystemExit):launcher.local_main(['--stage-only','--launch'])
    with pytest.raises(ValueError,match='cannot invoke'):
        launcher.remote_ensure(dict(stage_only=True))


@pytest.mark.parametrize('minimum',[0,-1,True,1.5,float('nan')])
def test_invalid_shared_gpu_threshold_is_rejected(minimum):
    with pytest.raises(ValueError,match='positive integer'):
        launcher.shared_gpu_min_free_mib(dict(shared_gpu_min_free_mib=minimum,
            launch_settings=dict(gpu_indices=[1,2],gpus_per_task=2)))


def test_shared_gpu_requires_explicit_pool_and_respects_free_memory_floor():
    with pytest.raises(ValueError,match='explicit --gpu-indices'):
        launcher.shared_gpu_min_free_mib(dict(shared_gpu_min_free_mib=58000))
    assert launcher.gpu_admitted(143771,70000,99,58000)
    assert not launcher.gpu_admitted(143771,70000,99)
    assert launcher.gpu_admitted(143771,85771,100,58000)
    assert not launcher.gpu_admitted(143771,85772,100,58000)
    assert not launcher.gpu_admitted(float('nan'),0,0,58000)
    assert launcher.gpu_admitted(143771,0,0)


def test_shared_gpu_selection_keeps_same_deployment_claims_and_allowlist(tmp_path,monkeypatch):
    from types import SimpleNamespace
    home=tmp_path/'home'
    for relative in ('Models/Qwen3-VL-2B-Instruct','Models/vjepa2-vitl-fpc64-256',
                     'Models/VLA-JEPA/Pretrain/checkpoints/VLA-JEPA-pretrain.pt'):
        path=home/relative;path.parent.mkdir(parents=True,exist_ok=True);path.touch()
    monkeypatch.setattr(launcher.Path,'home',classmethod(lambda cls:home))
    monkeypatch.setattr(launcher,'discover_model_python',lambda _:('python',[]))
    monkeypatch.setattr(launcher.shutil,'which',lambda _:'/usr/bin/quota')
    monkeypatch.setattr(launcher.shutil,'disk_usage',lambda _:SimpleNamespace(free=1000*launcher.GIB))
    monkeypatch.setattr(launcher.subprocess,'run',lambda *args,**kwargs:
        SimpleNamespace(returncode=0,stdout='/dev/fake 0 1073741824 1073741824'))
    def query(command,**kwargs):
        if command[0]=='df':return 'Filesystem\n/dev/fake\n'
        assert command[0]=='nvidia-smi'
        return ('0, GPU-0, NVIDIA H200, 143771, 0, 0\n'
                '1, GPU-1, NVIDIA H200, 143771, 70000, 99\n'
                '2, GPU-2, NVIDIA H200, 143771, 70000, 99\n')
    monkeypatch.setattr(launcher.subprocess,'check_output',query)
    request=dict(remote_root=str(tmp_path/'deploy'),tasks=['tabletop_cleanup'],shared_gpu_min_free_mib=58000,
        launch_settings=dict(gpus_per_task=2,gpu_indices=[1,2]),source_files={},
        datasets=[dict(task='tabletop_cleanup',remote_path=str(tmp_path/'dataset'),bytes=100,metadata_sha256={})])
    result=launcher.remote_probe(request)
    assert [gpu['index'] for gpu in result['gpu_groups']['tabletop_cleanup']]==[1,2]
    with pytest.raises(launcher.RetryLater,match='only 0 available'):
        launcher.remote_probe({**request,'shared_gpu_min_free_mib':None})
    launcher.write_json(tmp_path/'deploy/outputs/launch_receipts/cube_stack.json',
        dict(status='running',task='cube_stack',run=str(tmp_path/'cube_run'),gpu_uuid='GPU-1',source_files={}))
    with pytest.raises(launcher.RetryLater,match='only 1 available'):
        launcher.remote_probe(request)


@pytest.mark.parametrize('warm_start',[False,True])
def test_model_preflight_identifies_actual_initialization_and_checks_warm_dataset(tmp_path,monkeypatch,warm_start):
    import contextlib
    import io
    import sys
    from types import SimpleNamespace
    cfg=dict(trainer=dict(pretrained_checkpoint='official.pt',pretrained_sha256='official-sha',
                         gradient_accumulation_steps=1,target_epochs=3),
             datasets=dict(vla_data=dict(data_root_dir='/dataset',sample_stride=1,per_device_batch_size=32)),seed=42)
    if warm_start:cfg['trainer'].update(warm_start_checkpoint='/parent/checkpoints/last/model.pt',
        warm_start_sha256='warm-sha',warm_start_step=19167)
    norm={'scale':[180]*6+[8]}
    class Dataset:
        source_episode_count=500
        task_id='cube_stack'
        instruction='Stack the cubes.'
        info={'total_frames':454426}
        export_manifest_sha256='dataset-sha'
        split_manifest_sha256='split-sha'
        normalization=norm
        def __init__(self,*args,**kwargs):self.source_episode_ids=list(range(50))
        def __len__(self):return 408881
        def __getitem__(self,index):return {key:SimpleNamespace(shape=shape) for key,shape in
            [('state',(1,7)),('action',(7,7)),('video',(2,8,224,224,3))]}
    hashes=[];inspections=[]
    def sha(path):hashes.append(path);return 'official-sha'
    def inspect(config,**kwargs):
        inspections.append(kwargs)
        assert config['trainer']['warm_start_checkpoint'].endswith('/model.pt')
        assert kwargs==dict(schema='vlajepa.joint7.v1',provenance=dict(dataset_sha256='dataset-sha',
            split_sha256='split-sha',task_id='cube_stack'),normalization=norm)
        return dict(checkpoint_path=config['trainer']['warm_start_checkpoint'],checkpoint_sha256='warm-sha',source_step=19167)
    monkeypatch.setitem(sys.modules,'omegaconf',SimpleNamespace(OmegaConf=SimpleNamespace(load=lambda _:cfg,to_container=lambda c,**kwargs:c)))
    monkeypatch.setitem(sys.modules,'pressb.model_sources',SimpleNamespace(activate_source=lambda _:None))
    monkeypatch.setitem(sys.modules,'pressb.training.vlajepa_joint7_data',SimpleNamespace(Joint7LeRobotDataset=Dataset))
    monkeypatch.setitem(sys.modules,'starVLA.training.train_piper',SimpleNamespace(sha256=sha,
        epoch_geometry=lambda count,world,batch,accum,epochs:dict(target_steps=19167,world_size=world)))
    monkeypatch.setitem(sys.modules,'starVLA.training.train_table_buttons',SimpleNamespace(inspect_warm_start=inspect))
    def run(command,**kwargs):
        stream=io.StringIO()
        with monkeypatch.context() as context:
            context.setattr(sys,'argv',['-c',*command[3:]])
            with contextlib.redirect_stdout(stream):exec(command[2],{})
        return SimpleNamespace(returncode=0,stdout=stream.getvalue(),stderr='')
    monkeypatch.setattr(launcher.subprocess,'run',run)
    result=launcher.model_preflight(tmp_path,'python','config.yaml','cube_stack',2)
    assert result['global_batch_size']==64
    if warm_start:
        assert len(inspections)==1 and not hashes
        assert result['initialization']=='warm_start_model_only' and result['warm_start']['checkpoint_sha256']=='warm-sha'
        assert 'base_sha256' not in result
    else:
        assert hashes==['official.pt'] and not inspections
        assert result['initialization']=='official_pretrained' and result['base_sha256']=='official-sha'
