"""Proof, scope, symlink and interrupted deletion contracts for remote archival."""
import importlib.util
import os
from pathlib import Path
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location('archive_remote_runs', Path(__file__).resolve().parents[1]/'scripts/archive_remote_runs.py')
archive = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(archive)


def fixture(tmp_path,monkeypatch,name='historical_run'):
    root=tmp_path/'remote'; root.mkdir()
    run=root/name; (run/'checkpoint').mkdir(parents=True)
    (run/'checkpoint/model.pt').write_bytes(b'model-data')
    (run/'config.yaml').write_text('old-model-config')
    (run/'empty').mkdir()
    (run/'best').symlink_to('checkpoint')
    monkeypatch.setattr(archive,'ensure_no_users',lambda path:None)
    plan=archive.remote_plan(dict(remote_root=str(root)))
    row=next(x for x in plan['entries'] if x['name']==name)
    request=dict(remote_root=str(root),root_identity=plan['root_identity'],name=name,planned_identities=row['identities'])
    checked=archive.remote_inventory(request)
    request.update(manifest=checked['manifest'],verified_local_manifest_sha256=archive.fingerprint(checked['manifest']),
                   local_archive='/local/complete-backup/'+name,local_proof='/local/proof.json')
    return root,run,request


def test_hashes_regular_files_and_preserves_broken_and_absolute_links(tmp_path):
    (tmp_path/'weights').write_bytes(b'abcd')
    (tmp_path/'broken').symlink_to('/missing/original/model.pt')
    (tmp_path/'directory_alias').symlink_to('/tmp',target_is_directory=True)
    result=archive.inventory(tmp_path)
    assert result['manifest']['nodes']['weights']['sha256']==archive.hashlib.sha256(b'abcd').hexdigest()
    assert result['manifest']['nodes']['broken']==dict(kind='symlink',target='/missing/original/model.pt')
    assert set(result['manifest']['nodes'])=={'.','weights','broken','directory_alias'}


def test_plan_protects_current_tasks_in_names_descendants_and_targets(tmp_path):
    for name in ('CUBE_STACK_now','new_CLEANUP','legacy','nested','aliased'):
        (tmp_path/name).mkdir()
    (tmp_path/'nested/cube_stack_future').write_text('protected')
    (tmp_path/'aliased/current').symlink_to('/other/cleanup/checkpoints')
    result=archive.remote_plan(dict(remote_root=str(tmp_path)))
    assert [x['name'] for x in result['entries']]==['legacy']
    assert len(result['excluded'])==4


def test_plan_sorts_small_entries_before_large(tmp_path):
    (tmp_path/'b').write_bytes(b'bb'); (tmp_path/'a').write_bytes(b'a')
    result=archive.remote_plan(dict(remote_root=str(tmp_path)))
    assert [x['name'] for x in result['entries']]==['a','b']


def test_legacy_cleanup_bookkeeping_does_not_exclude_historical_runs(tmp_path):
    for run,relative in (
        ('old_red','previous_red1000_cleanup_receipt.json'),
        ('old_multitask','previous_multitask3000_cleanup_receipt.json'),
        ('old_panel','service/cleanup_services.py'),
        ('old_panel','service/__pycache__/cleanup_services.cpython-310.pyc'),
    ):
        p=tmp_path/run/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'legacy bookkeeping')
    result=archive.remote_plan(dict(remote_root=str(tmp_path)))
    assert {x['name'] for x in result['entries']}=={'old_red','old_multitask','old_panel'}
    assert result['excluded']==[]
    for entry in result['entries']:
        archive.validate_manifest(archive.inventory(tmp_path/entry['name'])['manifest'])


@pytest.mark.parametrize('value',[
    'cube_stack','CUBE_STACK_500','vlajepa_cube_stack_500/config.yaml',
    'tabletop_cleanup/checkpoint/model.pt','vlajepa_tabletop_cleanup_500',
    'stack','stack_20261110','cleanup','cleanup_20261110',
    '/outside/cleanup_20261110/model.pt','/outside/tabletop_cleanup/config.yaml',
])
def test_canonical_and_generic_task_directories_are_protected(value):
    assert archive.protected(value)


def test_orphan_transfer_is_adopted_before_inventory(tmp_path,monkeypatch):
    args=SimpleNamespace(output=tmp_path/'status',local_root=tmp_path/'local',host='h200',remote_root='/fixed/root',apply=True,poll_seconds=1)
    runner=archive.Archive(args);runner.plan=dict(remote_root='/fixed/root')
    run=runner.local/'old';run.mkdir();(run/'.model.pt.temporary').write_text('being written')
    command=['rsync','-aH','--checksum','--partial','--protect-args','--info=progress2','--stats',
             '-e',archive.shlex.join(['ssh',*archive.SSH]),'h200:/fixed/root/old',str(runner.local)+'/']
    runner.state['entries']['old']=dict(rsync_pid=123,rsync_start_ticks='100')
    original_text=Path.read_text;original_bytes=Path.read_bytes
    alive=[True];events=[]
    def read_text(p,*args,**kwargs):
        if str(p)=='/proc/123/stat':
            if not alive[0]:raise FileNotFoundError
            fields=['S']+['0']*18+['100']
            return '123 (rsync) '+' '.join(fields)
        return original_text(p,*args,**kwargs)
    def read_bytes(p,*args,**kwargs):
        if str(p)=='/proc/123/cmdline':return b'\0'.join(x.encode() for x in command)+b'\0'
        return original_bytes(p,*args,**kwargs)
    def sleep(seconds):
        events.append('adopt');alive[0]=False;(run/'.model.pt.temporary').unlink();(run/'model.pt').write_bytes(b'done')
    original_inventory=archive.inventory
    def inventory(p,**kwargs):
        events.append('inventory');assert not alive[0]
        return original_inventory(p,**kwargs)
    class Stopped(Exception):pass
    def launch(*args,**kwargs):raise Stopped
    monkeypatch.setattr(Path,'read_text',read_text);monkeypatch.setattr(Path,'read_bytes',read_bytes)
    monkeypatch.setattr(archive.time,'sleep',sleep);monkeypatch.setattr(archive,'inventory',inventory)
    monkeypatch.setattr(archive.subprocess,'Popen',launch)
    expected=dict(nodes={'.':dict(kind='dir'),'model.pt':dict(kind='file',bytes=4,sha256=archive.hashlib.sha256(b'done').hexdigest())})
    with pytest.raises(Stopped):runner.copy('old',expected)
    assert events==['adopt','inventory']


def test_verified_removal_preserves_parent_and_siblings(tmp_path,monkeypatch):
    root,run,request=fixture(tmp_path,monkeypatch)
    (root/'unrelated').write_text('keep')
    result=archive.delete_verified(request)
    assert result['status']=='archived'
    assert not run.exists()
    assert (root/'unrelated').read_text()=='keep'
    assert root.is_dir() and (root/archive.RECEIPTS).is_dir()
    assert archive.delete_verified(request)['status']=='archived'


def test_symlink_targets_are_never_deleted(tmp_path,monkeypatch):
    root,run,request=fixture(tmp_path,monkeypatch)
    outside=tmp_path/'outside';outside.mkdir();(outside/'payload').write_text('keep')
    (run/'external').symlink_to(outside,target_is_directory=True)
    row=archive.inventory(run); request.update(planned_identities=row['identities'],manifest=row['manifest'],
        verified_local_manifest_sha256=archive.fingerprint(row['manifest']))
    archive.delete_verified(request)
    assert (outside/'payload').read_text()=='keep'


@pytest.mark.parametrize('kind',['file','symlink'])
def test_top_level_non_directories_can_be_archived(tmp_path,monkeypatch,kind):
    root=tmp_path/'remote';root.mkdir();p=root/'migration.log'
    if kind=='file':p.write_text('log')
    else:p.symlink_to('/missing/historical')
    monkeypatch.setattr(archive,'ensure_no_users',lambda path:None)
    row=archive.inventory(p); st=root.stat()
    req=dict(remote_root=str(root),root_identity=dict(dev=st.st_dev,ino=st.st_ino),name=p.name,
        planned_identities=row['identities'],manifest=row['manifest'],
        verified_local_manifest_sha256=archive.fingerprint(row['manifest']),local_archive='/local/archive')
    archive.delete_verified(req)
    assert not os.path.lexists(p)


def test_same_size_mutation_prevents_deletion(tmp_path,monkeypatch):
    root,run,request=fixture(tmp_path,monkeypatch)
    (run/'checkpoint/model.pt').write_bytes(b'wrong-data')
    with pytest.raises(ValueError,match='differs'):
        archive.delete_verified(request)
    assert (run/'checkpoint/model.pt').exists()


def test_missing_local_proof_prevents_deletion(tmp_path,monkeypatch):
    root,run,request=fixture(tmp_path,monkeypatch)
    request['verified_local_manifest_sha256']='bad'
    with pytest.raises(ValueError,match='proof'):
        archive.delete_verified(request)
    assert run.exists()


def test_extra_file_prevents_deletion(tmp_path,monkeypatch):
    root,run,request=fixture(tmp_path,monkeypatch)
    (run/'new-data').write_text('new')
    with pytest.raises(ValueError,match='differs'):
        archive.delete_verified(request)
    assert (run/'new-data').exists()


def test_changed_root_identity_refuses_scope(tmp_path,monkeypatch):
    root,run,request=fixture(tmp_path,monkeypatch)
    request['root_identity']['ino']+=1
    with pytest.raises(ValueError,match='root identity'):
        archive.delete_verified(request)


@pytest.mark.parametrize('name',['../other','nested/path','.', '..','cube_stack_current','cleanup_old'])
def test_entry_name_scope_rejects_traversal_and_protection(name,tmp_path):
    with pytest.raises(ValueError):
        archive.entry_checked(dict(remote_root=str(tmp_path),name=name))


def test_manifest_path_traversal_refused(tmp_path,monkeypatch):
    root,run,request=fixture(tmp_path,monkeypatch)
    request['manifest']['nodes']['../outside']={'kind':'dir'}
    request['verified_local_manifest_sha256']=archive.fingerprint(request['manifest'])
    with pytest.raises(ValueError,match='escapes'):
        archive.delete_verified(request)


def test_interrupted_deletion_resumes_only_with_matching_tombstone(tmp_path,monkeypatch):
    root,run,request=fixture(tmp_path,monkeypatch)
    marker=archive.marker_path(root,run.name)
    archive.write_json(marker,dict(status='removing',name=run.name,manifest_sha256=archive.fingerprint(request['manifest']),
                                  local_archive=request['local_archive']))
    (run/'checkpoint/model.pt').unlink()
    assert archive.delete_verified(request)['status']=='archived'
    assert not run.exists()


def test_unaccounted_missing_source_fails_closed(tmp_path,monkeypatch):
    root,run,request=fixture(tmp_path,monkeypatch)
    (run/'checkpoint/model.pt').unlink()
    with pytest.raises(ValueError,match='differs'):
        archive.delete_verified(request)


def test_partial_delete_with_new_file_refuses_to_continue(tmp_path,monkeypatch):
    root,run,request=fixture(tmp_path,monkeypatch)
    marker=archive.marker_path(root,run.name)
    archive.write_json(marker,dict(status='removing',name=run.name,manifest_sha256=archive.fingerprint(request['manifest']),local_archive=request['local_archive']))
    (run/'checkpoint/model.pt').unlink();(run/'new').write_text('keep')
    with pytest.raises(ValueError,match='differs'):
        archive.delete_verified(request)


def test_active_process_prevents_deletion(tmp_path,monkeypatch):
    root,run,request=fixture(tmp_path,monkeypatch)
    def active(path):raise archive.RetryLater('active training')
    monkeypatch.setattr(archive,'ensure_no_users',active)
    with pytest.raises(archive.RetryLater,match='active'):
        archive.delete_verified(request)
    assert run.exists()


def test_hardlinked_payloads_are_preserved_in_manifest_and_removed_safely(tmp_path,monkeypatch):
    root,run,request=fixture(tmp_path,monkeypatch)
    os.link(run/'checkpoint/model.pt',run/'checkpoint/last.pt')
    row=archive.inventory(run)
    request.update(planned_identities=row['identities'],manifest=row['manifest'],verified_local_manifest_sha256=archive.fingerprint(row['manifest']))
    assert row['manifest']['nodes']['checkpoint/model.pt']==row['manifest']['nodes']['checkpoint/last.pt']
    assert archive.delete_verified(request)['status']=='archived'


def test_symlink_receipt_directory_refused(tmp_path,monkeypatch):
    root,run,request=fixture(tmp_path,monkeypatch)
    outside=tmp_path/'receipt-outside';outside.mkdir()
    (root/archive.RECEIPTS).symlink_to(outside)
    with pytest.raises(ValueError,match='symlink'):
        archive.delete_verified(request)
    assert run.exists() and not list(outside.iterdir())


def test_source_changed_since_initial_plan_fails(tmp_path,monkeypatch):
    root,run,request=fixture(tmp_path,monkeypatch)
    (run/'config.yaml').write_text('changed-config')
    with pytest.raises(ValueError,match='since'):
        archive.remote_inventory(request)


def test_session_permission_exemption_is_exact():
    assert archive.session_daemon('systemd','/usr/lib/systemd/systemd --user')
    assert archive.session_daemon('sshd','sshd: user@notty')
    assert not archive.session_daemon('python','python trainer.py')
    assert not archive.session_daemon('sshd','python /tmp/sshd')


def test_apply_requires_same_plan_contract(tmp_path,monkeypatch):
    args=SimpleNamespace(output=tmp_path/'status',local_root=tmp_path/'local',host='h200',remote_root='/fixed/root',apply=False,poll_seconds=1)
    runner=archive.Archive(args)
    archive.write_json(runner.output/'plan.json',dict(contract={'host':'different'}))
    with pytest.raises(ValueError,match='different'):
        runner.run()


def make_runner(tmp_path,workers=3):
    args=SimpleNamespace(output=tmp_path/'status',local_root=tmp_path/'local',host='h200',
                         remote_root=str(tmp_path/'remote'),apply=True,poll_seconds=.01,workers=workers)
    return archive.Archive(args)


def test_actual_concurrent_verified_archives_and_completed_entries_skip(tmp_path,monkeypatch):
    root=tmp_path/'remote';root.mkdir()
    for index in range(3):
        run=root/f'old_{index}';run.mkdir()
        (run/'weights').write_bytes(bytes([index])*128)
        (run/'config').write_text('config '+str(index))
        (run/'best').symlink_to('weights')
    monkeypatch.setattr(archive,'ensure_no_users',lambda path:None)
    runner=make_runner(tmp_path)
    plan=archive.remote_plan(dict(remote_root=str(root)));plan['contract']=runner.contract
    for index in range(16):
        name=f'previously_archived_{index}'
        plan['entries'].append(dict(name=name,logical_bytes=0,allocated_bytes=0,identities={}))
        runner.update(name,stage='archived',original_receipt='preserve')
    archive.write_json(runner.output/'plan.json',plan)
    monkeypatch.setattr(archive,'remote_call',lambda args,mode,request:getattr(archive,mode)(request))
    barrier=threading.Barrier(3);guard=threading.Lock();running=0;peak=0;copied=[]
    def copy(name,manifest):
        nonlocal running,peak
        with guard:
            running+=1;peak=max(peak,running);copied.append(name)
        barrier.wait(timeout=5)
        shutil.copytree(root/name,runner.local/name,symlinks=True)
        with guard:running-=1
    monkeypatch.setattr(runner,'copy',copy)
    runner.run()
    assert peak==3 and sorted(copied)==['old_0','old_1','old_2']
    assert archive.read_json(runner.output/'status.json')['status']=='completed'
    assert all(runner.entry_state(f'old_{i}')['stage']=='archived' for i in range(3))
    assert all(runner.entry_state(f'previously_archived_{i}')['original_receipt']=='preserve' for i in range(16))
    for index in range(3):
        assert not (root/f'old_{index}').exists()
        assert (runner.local/f'old_{index}/best').read_bytes()==bytes([index])*128


def test_threaded_state_writes_are_atomic_and_keep_all_entries(tmp_path):
    runner=make_runner(tmp_path,workers=8)
    barrier=threading.Barrier(8)
    def updates(index):
        barrier.wait(timeout=5)
        for counter in range(15):
            runner.update(str(index),counter=counter,thread=index)
            snapshot=archive.read_json(runner.output/'status.json')
            assert snapshot['workers']==8
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(updates,range(8)))
    result=archive.read_json(runner.output/'status.json')
    assert len(result['entries'])==8
    assert all(row['counter']==14 for row in result['entries'].values())
    assert not list(runner.output.glob('*.tmp'))


def test_failure_stops_new_work_waits_for_active_work_and_never_completes(tmp_path,monkeypatch):
    runner=make_runner(tmp_path,workers=3)
    entries=[dict(name=name,logical_bytes=0,allocated_bytes=0,identities={}) for name in ('bad','active_a','active_b','must_not_start')]
    archive.write_json(runner.output/'plan.json',dict(contract=runner.contract,entries=entries,excluded=[],remote_root='/fixed/root'))
    barrier=threading.Barrier(3);release=threading.Event();raised=threading.Event();started=[];errors=[]
    def process(plan,entry):
        name=entry['name'];started.append(name);barrier.wait(timeout=5)
        if name=='bad':
            raised.set();raise ValueError('bad checksum')
        assert release.wait(timeout=5)
        runner.update(name,stage='archived')
    monkeypatch.setattr(runner,'process',process)
    def run():
        try:runner.run()
        except Exception as error:errors.append(error)
    thread=threading.Thread(target=run);thread.start()
    assert raised.wait(timeout=5)
    deadline=time.monotonic()+5
    while archive.read_json(runner.output/'status.json')['status']!='failed' and time.monotonic()<deadline:
        time.sleep(.005)
    assert archive.read_json(runner.output/'status.json')['status']=='failed'
    assert thread.is_alive()  # Already-running archives finish before supervisor exits.
    release.set();thread.join(timeout=5)
    assert not thread.is_alive() and len(errors)==1 and 'bad checksum' in str(errors[0])
    assert 'must_not_start' not in started
    result=archive.read_json(runner.output/'status.json')
    assert result['status']=='failed' and 'finished_unix' not in result
    assert result['entries']['active_a']['stage']=='archived'
    assert result['entries']['active_b']['stage']=='archived'


def test_retry_worker_exits_after_another_worker_fails(tmp_path,monkeypatch):
    runner=make_runner(tmp_path,workers=2)
    barrier=threading.Barrier(2)
    entries=[dict(name=name) for name in ('retry','fatal')]
    runner.plan={}
    def process(plan,entry):
        barrier.wait(timeout=5)
        if entry['name']=='retry':raise archive.RetryLater('temporarily busy')
        raise ValueError('fatal checksum')
    monkeypatch.setattr(runner,'process',process)
    with pytest.raises(ValueError,match='fatal'):
        runner.run_workers(entries)
    assert runner.state['status']=='failed'


def test_workers_can_change_without_changing_legacy_contract(tmp_path):
    runner=make_runner(tmp_path,workers=1)
    contract=dict(runner.contract)
    runner.update('done',stage='archived')
    legacy=archive.read_json(runner.output/'status.json');legacy.pop('workers')
    archive.write_json(runner.output/'status.json',legacy)
    restarted=make_runner(tmp_path,workers=3)
    assert restarted.contract==contract and restarted.state['workers']==3
    assert restarted.entry_state('done')['stage']=='archived'


def test_duplicate_planned_entries_rejected_before_concurrent_work(tmp_path,monkeypatch):
    runner=make_runner(tmp_path)
    entry=dict(name='duplicate',logical_bytes=0,allocated_bytes=0,identities={})
    archive.write_json(runner.output/'plan.json',dict(contract=runner.contract,entries=[entry,entry],excluded=[]))
    monkeypatch.setattr(runner,'process',lambda *args:pytest.fail('must not execute'))
    with pytest.raises(ValueError,match='duplicate'):
        runner.run()


@pytest.mark.parametrize('workers',[0,9,True,1.5])
def test_invalid_worker_counts_refused(tmp_path,workers):
    with pytest.raises(ValueError,match='workers'):
        make_runner(tmp_path,workers=workers)


def test_process_adopts_transfer_before_remote_inventory(tmp_path,monkeypatch):
    runner=make_runner(tmp_path);runner.plan=dict(remote_root='/fixed/root')
    events=[]
    monkeypatch.setattr(runner,'adopt_transfer',lambda name:events.append('adopt'))
    class StopAfterInventory(Exception):pass
    def remote(*args):
        events.append('remote_inventory');assert events==['adopt','remote_inventory'];raise StopAfterInventory
    monkeypatch.setattr(archive,'remote_call',remote)
    with pytest.raises(StopAfterInventory):
        runner.process(dict(remote_root='/fixed/root',root_identity={}),dict(name='old',identities={},logical_bytes=1))
