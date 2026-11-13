"""Completion, backup, ownership and sequential scheduling for task SFT."""
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('manipulation_sft_queue', Path(__file__).resolve().parents[1]/'scripts/run_manipulation_sft_queue.py')
queue = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(queue)


def completed_run(tmp_path, monkeypatch, task='cube_stack'):
    root = tmp_path/'deployment'; run = root/'outputs/model_training'/f'vlajepa_{task}_500_20261110'
    checkpoint = run/'checkpoints/best_step_001000'
    checkpoint.mkdir(parents=True)
    (checkpoint/'model.pt').write_bytes(b'weights')
    (checkpoint/'manifest.json').write_text('{"metadata": true}')
    (run/'checkpoints/best').symlink_to('best_step_001000', target_is_directory=True)
    (run/'checkpoints/last').symlink_to('best_step_001000', target_is_directory=True)
    queue.write_json(run/'initialization.json', dict(target_steps=1000))
    queue.write_json(run/'completed.json', dict(success=True,step=1000,world_size=2))
    queue.write_json(run/'status.json', dict(status='completed',step=1000))
    queue.write_json(root/'outputs/launch_receipts'/f'{task}.json', dict(task=task,status='completed',exit_code=0,run=str(run),world_size=2,per_device_batch_size=32,gpu_indices=[1,2]))
    monkeypatch.setattr(queue, 'ensure_no_users', lambda root: None)
    return root, run, dict(task=task, remote_root=str(root), gpus_per_task=2)


def test_manifest_hashes_files_and_preserves_relative_aliases(tmp_path):
    (tmp_path/'payload').write_bytes(b'abc'); (tmp_path/'link').symlink_to('payload')
    first = queue.tree_manifest(tmp_path)
    assert first['files']['payload']['bytes'] == 3
    assert first['symlinks'] == {'link':'payload'}
    (tmp_path/'payload').write_bytes(b'abd')
    assert queue.tree_manifest(tmp_path) != first


def test_manifest_refuses_external_symlink(tmp_path):
    root=tmp_path/'run'; root.mkdir(); outside=tmp_path/'outside';outside.write_text('private')
    (root/'escape').symlink_to('../outside')
    with pytest.raises(ValueError, match='Unsafe'):
        queue.tree_manifest(root)


def test_new_remote_deployment_is_not_started(tmp_path):
    assert queue.remote_status(dict(remote_root=str(tmp_path/'not-yet-created'),task='cube_stack')) == dict(status='not_started', complete=False)


def test_completed_requires_actual_target_not_partial_training(tmp_path, monkeypatch):
    root, run, request=completed_run(tmp_path,monkeypatch)
    assert queue.remote_status(request)['complete']
    queue.write_json(run/'completed.json', dict(success=True,step=99,world_size=2))
    queue.write_json(run/'status.json', dict(status='completed',step=99))
    with pytest.raises(RuntimeError, match='full-training'):
        queue.remote_status(request)


def test_failed_or_stopped_training_never_restarts_silently(tmp_path,monkeypatch):
    root,run,request=completed_run(tmp_path,monkeypatch)
    p=root/'outputs/launch_receipts/cube_stack.json'
    row=queue.read_json(p)
    for state in ('failed','stopped','cancelled','superseded'):
        row['status']=state;queue.write_json(p,row)
        with pytest.raises(RuntimeError,match='no automatic restart'):
            queue.remote_status(request)


def test_running_job_requires_live_supervisor(tmp_path,monkeypatch):
    root,run,request=completed_run(tmp_path,monkeypatch)
    p=root/'outputs/launch_receipts/cube_stack.json';row=queue.read_json(p)
    row.update(status='running',supervisor_pid=123,supervisor_start_ticks='abc');queue.write_json(p,row)
    monkeypatch.setattr(queue,'process_identity',lambda pid:'abc')
    assert not queue.remote_status(request)['complete']
    monkeypatch.setattr(queue,'process_identity',lambda pid:None)
    with pytest.raises(RuntimeError,match='disappeared'):
        queue.remote_status(request)


def test_only_verified_model_payload_removed_and_metadata_retained(tmp_path,monkeypatch):
    root,run,request=completed_run(tmp_path,monkeypatch)
    (run/'other.pt').write_bytes(b'keep')
    manifest=queue.tree_manifest(run)
    request.update(manifest=manifest,verified_local_manifest_sha256=queue.fingerprint(manifest),archive_path='/local/verified')
    result=queue.remove_checkpoint_payloads(request)
    assert result['status']=='archived'
    assert not (run/'checkpoints/best/model.pt').exists()
    assert (run/'checkpoints/best/manifest.json').is_file()
    assert (run/'other.pt').read_bytes()==b'keep'
    assert queue.remove_checkpoint_payloads(request)['status']=='archived'


def test_changed_checkpoint_prevents_removal(tmp_path,monkeypatch):
    root,run,request=completed_run(tmp_path,monkeypatch)
    manifest=queue.tree_manifest(run)
    request.update(manifest=manifest,verified_local_manifest_sha256=queue.fingerprint(manifest),archive_path='/local/verified')
    (run/'checkpoints/best/model.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):
        queue.remove_checkpoint_payloads(request)
    assert (run/'checkpoints/best/model.pt').is_file()


def test_live_user_prevents_all_rotation(tmp_path,monkeypatch):
    root,run,request=completed_run(tmp_path,monkeypatch)
    def occupied(root):raise queue.RetryLater('still uses')
    monkeypatch.setattr(queue,'ensure_no_users',occupied)
    with pytest.raises(queue.RetryLater):queue.remote_run_inventory(request)


def dataset_request(tmp_path, monkeypatch):
    root,run,request=completed_run(tmp_path,monkeypatch)
    data_root=tmp_path/'datasets';data=data_root/'pressb_cube_stack_500_20261110';(data/'meta').mkdir(parents=True)
    (data/'meta/info.json').write_text('{"audited":true}')
    (data/'video.mp4').write_bytes(b'video')
    local=queue.tree_manifest(data)
    request.update(remote_dataset_root=str(data_root),remote_dataset_path=str(data),manifest=local,
        verified_local_manifest_sha256=queue.fingerprint(local),local_path='/durable/local/dataset')
    return root,data,request


def test_dataset_release_is_exact_verified_and_idempotent(tmp_path,monkeypatch):
    root,data,request=dataset_request(tmp_path,monkeypatch)
    sibling=data.parent/'unrelated';sibling.mkdir();(sibling/'keep').write_text('keep')
    result=queue.release_dataset(request)
    assert result['status']=='archived' and not data.exists()
    assert (sibling/'keep').read_text()=='keep'
    assert queue.release_dataset(request)['status']=='archived'


def test_dataset_changed_or_wrong_directory_is_not_removed(tmp_path,monkeypatch):
    root,data,request=dataset_request(tmp_path,monkeypatch)
    (data/'video.mp4').write_bytes(b'changed')
    with pytest.raises(ValueError,match='differs'):
        queue.release_dataset(request)
    request['remote_dataset_path']=str(data.parent/'unrelated')
    with pytest.raises(ValueError,match='exact task'):
        queue.release_dataset(request)
    assert data.exists()


def test_interrupted_dataset_release_continues_only_from_manifest_receipt(tmp_path,monkeypatch):
    root,data,request=dataset_request(tmp_path,monkeypatch)
    marker=dict(status='removing',manifest_sha256=queue.fingerprint(request['manifest']))
    queue.write_json(root/'outputs/launch_receipts/cube_stack.dataset_archived.json',marker)
    (data/'video.mp4').unlink()
    assert queue.remote_dataset_inventory(request)['marker']['status']=='removing'
    assert queue.release_dataset(request)['status']=='archived'


def arguments(tmp_path,extra=()):
    return queue.parse_args(['--output',str(tmp_path/'queue'),'--archive-root',str(tmp_path/'archive'),*extra])


def test_rotation_is_opt_in_and_ddp_defaults_keep_total_batch_64(tmp_path):
    args=arguments(tmp_path);q=queue.Queue(args);command=q.launch_command('cube_stack')
    assert not args.archive_completed_remote_payloads and not args.release_remote_dataset
    assert command[command.index('--gpus-per-task')+1]=='2'
    assert command[command.index('--per-device-batch-size')+1]=='32'


def test_per_task_storage_override_is_used_for_launch_and_remote_operations(tmp_path):
    p=tmp_path/'overrides.json'
    queue.write_json(p,dict(tabletop_cleanup=dict(remote_root='/data/cleanup',remote_dataset_root='/home/datasets',
        checkpoint_reserve_gib=19.13,storage_margin_gib=.25)))
    args=arguments(tmp_path,['--task-overrides',str(p)]);q=queue.Queue(args)
    assert q.request('tabletop_cleanup')['remote_root']=='/data/cleanup'
    assert q.request('tabletop_cleanup')['remote_dataset_path']=='/home/datasets/pressb_tabletop_cleanup_500_20261110'
    command=q.launch_command('tabletop_cleanup')
    assert command[command.index('--checkpoint-reserve-gib')+1]=='19.13'
    assert command[command.index('--storage-margin-gib')+1]=='0.25'
    assert q.request('cube_stack')['remote_root']==args.remote_root


def test_queue_waits_full_completion_and_archive_before_next_task(tmp_path,monkeypatch):
    q=queue.Queue(arguments(tmp_path));events=[]
    monkeypatch.setattr(q,'await_completion',lambda task:events.append(('completed',task)))
    monkeypatch.setattr(q,'archive',lambda task:events.append(('archived',task)))
    q.run()
    assert events==[('completed','cube_stack'),('archived','cube_stack'),('completed','tabletop_cleanup'),('archived','tabletop_cleanup')]
    assert queue.read_json(q.output/'status.json')['status']=='completed'


def test_failed_first_training_prevents_second_launch(tmp_path,monkeypatch):
    q=queue.Queue(arguments(tmp_path));events=[]
    def fail(task):events.append(task);raise RuntimeError('training failed')
    monkeypatch.setattr(q,'await_completion',fail)
    with pytest.raises(RuntimeError):q.run()
    assert events==['cube_stack'] and q.state['status']=='failed'


def test_resume_rejects_changed_storage_or_training_configuration(tmp_path):
    queue.Queue(arguments(tmp_path))
    with pytest.raises(ValueError,match='settings changed'):
        queue.Queue(arguments(tmp_path,['--per-device-batch-size','64']))


def test_queue_forwards_explicit_gpu_pair_and_global_storage_budget(tmp_path):
    q=queue.Queue(arguments(tmp_path,['--gpu-indices','1','2','--checkpoint-reserve-gib','22','--storage-margin-gib','1']))
    command=q.launch_command('cube_stack')
    assert command[command.index('--gpu-indices')+1:command.index('--gpu-indices')+3]==['1','2']
    assert command[command.index('--checkpoint-reserve-gib')+1]=='22.0'
    assert q.request('cube_stack')['gpu_indices']==[1,2]
    assert q.request('cube_stack')['per_device_batch_size']==32


@pytest.mark.parametrize('field,value',[('gpus_per_task',1),('per_device_batch_size',64),('gpu_indices',[0,1]),('run_suffix','other')])
def test_remote_status_does_not_adopt_a_different_training_configuration(tmp_path,monkeypatch,field,value):
    root,run,request=completed_run(tmp_path,monkeypatch);request[field]=value
    with pytest.raises(ValueError,match='different'):
        queue.remote_status(request)


def test_launching_receipt_can_precede_run_directory_creation(tmp_path):
    import time
    root=tmp_path/'new'
    run=root/'outputs/model_training/vlajepa_cube_stack_500_20261110'
    queue.write_json(root/'outputs/launch_receipts/cube_stack.json',dict(task='cube_stack',run=str(run),status='launching',
        created_unix=time.time(),world_size=2))
    assert queue.remote_status(dict(task='cube_stack',remote_root=str(root),gpus_per_task=2))['status']=='launching'


def test_receipt_launch_settings_must_agree_with_actual_requested_recipe(tmp_path,monkeypatch):
    root,run,request=completed_run(tmp_path,monkeypatch)
    p=root/'outputs/launch_receipts/cube_stack.json';row=queue.read_json(p)
    row['launch_settings']=dict(gpus_per_task=3);queue.write_json(p,row)
    with pytest.raises(ValueError,match='launch settings'):
        queue.remote_status(request)


@pytest.mark.parametrize('task,release_final,expected_modes',[
    ('cube_stack',False,['prune-model','dataset-inventory','release-dataset']),
    ('tabletop_cleanup',False,[]),
    ('tabletop_cleanup',True,['prune-model','dataset-inventory','release-dataset']),
])
def test_final_task_is_backed_up_but_retained_remote_unless_explicitly_released(tmp_path,monkeypatch,task,release_final,expected_modes):
    extra=['--archive-completed-remote-payloads','--release-remote-dataset','--dataset-root',str(tmp_path/'datasets')]
    if release_final:extra+=['--release-final-remote']
    args=arguments(tmp_path,extra);q=queue.Queue(args)
    archive=args.archive_root/task;archive.mkdir(parents=True);(archive/'model.pt').write_bytes(b'verified model')
    manifest=queue.tree_manifest(archive)
    queue.write_json(q.output/f'{task}.archive.json',dict(task=task,archive_path=str(archive),run='/remote/task',
        manifest=manifest,manifest_sha256=queue.fingerprint(manifest)))
    data=args.dataset_root/f'pressb_{task}_500_20261110';data.mkdir(parents=True)
    (data/'data').write_bytes(b'verified dataset');dataset_manifest=queue.tree_manifest(data)
    modes=[]
    def remote(args,mode,request):
        modes.append(mode)
        return dict(missing=False,manifest=dataset_manifest) if mode=='dataset-inventory' else dict(status='archived')
    monkeypatch.setattr(queue,'remote_call',remote)
    q.archive(task)
    assert modes==expected_modes
    assert q.state['tasks'][task]['stage']=='completed'
    assert (archive/'model.pt').read_bytes()==b'verified model'


@pytest.mark.parametrize('flag',['--archive-completed-remote-payloads','--release-remote-dataset','--release-final-remote'])
def test_background_archives_reject_remote_deletion_flags(tmp_path,flag):
    with pytest.raises(SystemExit):
        arguments(tmp_path,['--background-archives',flag])


def test_stage_command_does_not_request_training_and_has_separate_status(tmp_path):
    q=queue.Queue(arguments(tmp_path,['--prestage-next']))
    cmd=q.launch_command('tabletop_cleanup',stage_only=True)
    assert '--stage-only' in cmd and '--launch' not in cmd
    assert cmd[cmd.index('--status')+1].endswith('tabletop_cleanup.staging.json')
    assert q.launch_command('tabletop_cleanup')[q.launch_command('tabletop_cleanup').index('--status')+1].endswith('tabletop_cleanup.deployment.json')


def fake_audit(monkeypatch, *, ready=True):
    import sys
    from types import SimpleNamespace
    def audit(*args):
        if not ready:raise queue.RetryLater('dataset still exporting')
        return {'audited':True}
    monkeypatch.setitem(sys.modules,'launch_manipulation_sft',SimpleNamespace(audit_gate=audit,RetryLater=queue.RetryLater))


def test_prestage_waits_for_full_dataset_audit_without_spawning(tmp_path,monkeypatch):
    fake_audit(monkeypatch,ready=False);q=queue.Queue(arguments(tmp_path,['--prestage-next']))
    def bad_spawn(*args,**kwargs):raise AssertionError('must not spawn')
    monkeypatch.setattr(queue.subprocess,'Popen',bad_spawn)
    assert not q.prestage('tabletop_cleanup')
    assert q.state['tasks']['tabletop_cleanup']['prestage_status']=='waiting_dataset'


def test_prestage_is_nonblocking_and_restart_adopts_same_transfer(tmp_path,monkeypatch):
    fake_audit(monkeypatch);args=arguments(tmp_path,['--prestage-next']);q=queue.Queue(args);spawned=[]
    monkeypatch.setattr(queue.subprocess,'Popen',lambda *a,**kw:spawned.append(a))
    assert not q.prestage('tabletop_cleanup')
    assert len(spawned)==1
    p=q.output/'tabletop_cleanup.stage_0001.json';row=queue.read_json(p)
    row.update(status='running',supervisor_pid=999,start_ticks='alive');queue.write_json(p,row)
    monkeypatch.setattr(queue,'process_identity',lambda pid:'alive')
    restarted=queue.Queue(args)
    assert not restarted.prestage('tabletop_cleanup') and len(spawned)==1
    row.update(status='exited',exit_code=0);queue.write_json(p,row)
    queue.write_json(q.output/'tabletop_cleanup.staging.json',dict(status='staged'))
    assert restarted.prestage('tabletop_cleanup') and len(spawned)==1


def test_hard_stage_failure_is_not_silently_retried(tmp_path,monkeypatch):
    fake_audit(monkeypatch);q=queue.Queue(arguments(tmp_path,['--prestage-next']))
    q.update('tabletop_cleanup',stage_attempt=1)
    queue.write_json(q.output/'tabletop_cleanup.stage_0001.json',dict(status='exited',exit_code=2,
        command=q.launch_command('tabletop_cleanup',stage_only=True),log='failed.log'))
    with pytest.raises(RuntimeError,match='Stage worker failed'):
        q.prestage('tabletop_cleanup')


def test_actual_launch_waits_for_inflight_stage_before_starting(tmp_path,monkeypatch):
    q=queue.Queue(arguments(tmp_path,['--prestage-next']))
    events=[];attempts=iter([False,True])
    def stage(task):
        result=next(attempts);events.append(('stage_ready',result));return result
    monkeypatch.setattr(q,'prestage',stage)
    monkeypatch.setattr(queue.time,'sleep',lambda seconds:events.append(('wait',seconds)))
    def spawn(command,**kwargs):
        events.append(('launch_spawned',True));p=Path(command[-1]);row=queue.read_json(p)
        row.update(status='exited',exit_code=0);queue.write_json(p,row)
    monkeypatch.setattr(queue.subprocess,'Popen',spawn)
    q.launch('tabletop_cleanup')
    assert [event[0] for event in events]==['stage_ready','wait','stage_ready','launch_spawned']


def test_background_archive_does_not_delay_next_training(tmp_path,monkeypatch):
    q=queue.Queue(arguments(tmp_path,['--background-archives']));events=[]
    monkeypatch.setattr(q,'await_completion',lambda task:events.append(('training_done',task)))
    monkeypatch.setattr(q,'start_archive',lambda task:events.append(('archive_spawn',task)))
    monkeypatch.setattr(q,'poll_archives',lambda **kw:events.append(('archive_wait',True)) or True)
    monkeypatch.setattr(q,'archive',lambda task:(_ for _ in ()).throw(AssertionError('must not block on archive')))
    q.run()
    assert events==[('training_done','cube_stack'),('archive_spawn','cube_stack'),
                    ('training_done','tabletop_cleanup'),('archive_spawn','tabletop_cleanup'),('archive_wait',True)]
    assert q.state['status']=='completed'


def test_archive_worker_writes_own_state_then_parent_polls_verified_receipt(tmp_path,monkeypatch):
    q=queue.Queue(arguments(tmp_path,['--background-archives']));spawned=[]
    monkeypatch.setattr(queue.subprocess,'Popen',lambda *a,**kw:spawned.append(a))
    q.start_archive('cube_stack');parent_before=(q.output/'status.json').read_bytes()
    p=q.output/'cube_stack.archive_worker.json';row=queue.read_json(p)
    assert Path(row['args']['output']) != q.output
    def archive(worker,task):
        manifest=dict(files={},symlinks={})
        queue.write_json(worker.output/f'{task}.archive.json',dict(verified_unix=123,manifest=manifest,
            manifest_sha256=queue.fingerprint(manifest),archive_path='/verified/archive'))
        worker.update(task,stage='completed',archived=True)
    monkeypatch.setattr(queue.Queue,'archive',archive)
    assert queue.archive_worker(p)==0
    assert (q.output/'status.json').read_bytes()==parent_before
    assert q.poll_archives()
    assert q.state['tasks']['cube_stack']['archived']
    q.start_archive('cube_stack')
    assert len(spawned)==1


def test_archive_worker_failure_is_reported_without_restart(tmp_path,monkeypatch):
    q=queue.Queue(arguments(tmp_path,['--background-archives']))
    queue.write_json(q.output/'cube_stack.archive_worker.json',dict(status='failed',error='checksum mismatch',log='archive.log'))
    with pytest.raises(RuntimeError,match='checksum mismatch'):
        q.poll_archives()


def test_background_mode_changes_require_new_queue_identity(tmp_path):
    queue.Queue(arguments(tmp_path))
    with pytest.raises(ValueError,match='settings changed'):
        queue.Queue(arguments(tmp_path,['--background-archives']))


def test_task_only_shared_gpu_admission_is_forwarded_to_staging_and_launch(tmp_path):
    p=tmp_path/'overrides.json'
    queue.write_json(p,dict(tabletop_cleanup=dict(shared_gpu_min_free_mib=58000)))
    q=queue.Queue(arguments(tmp_path,['--task-overrides',str(p),'--gpu-indices','1','2']))
    for stage_only in (False,True):
        command=q.launch_command('tabletop_cleanup',stage_only=stage_only)
        assert command[command.index('--shared-gpu-min-free-mib')+1]=='58000'
        assert '--shared-gpu-min-free-mib' not in q.launch_command('cube_stack',stage_only=stage_only)
    assert q.request('tabletop_cleanup')['shared_gpu_min_free_mib']==58000
    assert q.request('cube_stack')['shared_gpu_min_free_mib'] is None
    assert q.state['contract']['task_overrides']['tabletop_cleanup']['shared_gpu_min_free_mib']==58000


@pytest.mark.parametrize('value',[0,-1,1.5,True,'58000',None])
def test_invalid_shared_gpu_free_memory_override_rejected(value):
    with pytest.raises(ValueError,match='positive integer'):
        queue.validate_overrides(dict(tabletop_cleanup=dict(shared_gpu_min_free_mib=value)))


def test_shared_gpu_admission_requires_explicit_gpu_indices(tmp_path):
    p=tmp_path/'overrides.json';queue.write_json(p,dict(tabletop_cleanup=dict(shared_gpu_min_free_mib=58000)))
    with pytest.raises(ValueError,match='explicit --gpu-indices'):
        queue.Queue(arguments(tmp_path,['--task-overrides',str(p)]))


def test_shared_gpu_receipt_policy_must_match_queue_request(tmp_path,monkeypatch):
    root,run,request=completed_run(tmp_path,monkeypatch)
    request['shared_gpu_min_free_mib']=58000
    with pytest.raises(ValueError,match='admission policy'):
        queue.remote_status(request)
    p=root/'outputs/launch_receipts/cube_stack.json';row=queue.read_json(p)
    row['shared_gpu_min_free_mib']=58000;queue.write_json(p,row)
    assert queue.remote_status(request)['complete']
    request['shared_gpu_min_free_mib']=None
    with pytest.raises(ValueError,match='admission policy'):
        queue.remote_status(request)
