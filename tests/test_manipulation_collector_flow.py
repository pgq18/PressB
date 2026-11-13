"""CPU end-to-end collector scheduling, boundary alignment and resume checks."""
from copy import deepcopy
import gc
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from pressb.runtime import demonstrations
from pressb import registry

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('test_manipulation_collector_script',ROOT/'scripts/collect_manipulation.py')
collector=importlib.util.module_from_spec(spec);spec.loader.exec_module(collector)


class DummyVideo:
    def __init__(self,path,size,fps):self.path=Path(path);self.count=0
    def add(self,rgb):self.count+=1
    def close(self):self.path.write_text(str(self.count))


@pytest.fixture
def harness(monkeypatch):
    module=ModuleType('pressb.tasks.cube_stack.collection')
    module.INSTRUCTION='Stack the red cube on the blue cube.'
    module.collection_config=lambda base: {'scene_overrides':{}}
    module.plan_calls=[]
    module.sample_eids=[]
    module.reject_seeds=set()
    def plan(config,reset_spec,snapshot,base,**options):
        seed=reset_spec['seed'];module.plan_calls.append(seed)
        if seed in module.reject_seeds:raise ValueError('Synthetic unreachable sample')
        count=3+seed%3
        actions=[[.1+.01*i,0,0,0,0,0,.06] for i in range(count)]
        return dict(actions=actions,phases=['pregrasp']*(count-1)+['settled'],fps=30.,
                    expected_initial_state=[0,0,0,0,0,0,.06],seed=seed)
    module.plan_episode=plan
    module.assess_episode=lambda history,plan: dict(passed=(history[-1]['action_index']==len(plan['actions'])-1 and history[-1]['phase']=='settled'))
    engines=[]
    class Backend:
        def __init__(self,config):
            self.num_envs=config['num_envs'];self.ready=np.zeros(self.num_envs,bool)
            self.reset_specs=[None]*self.num_envs;self.states=np.zeros((self.num_envs,7));self.states[:,6]=.06
            self.task=SimpleNamespace(config=dict(control_hz=30.,scene={'initial_state_joint_rad_m':[0,0,0,0,0,0,.06]}),
                lower=np.array([-2]*6+[0]),upper=np.array([2]*6+[.07]),scene={'robot_base_position_world':[0,0,.766]})
            self.task.sample_reset=lambda eid,seed,opts: (module.sample_eids.append(eid) or dict(seed=seed,objects={'red':{'center_xy':[.3,.1],'yaw_deg':0.},'blue':{'center_xy':[.3,-.1],'yaw_deg':0.}}))
            self.engine=SimpleNamespace(offsets=np.zeros((self.num_envs,3)))
            self.timings={};self.step_ids=[];self.reset_ids=[];self.closed=False
            engines.append(self)
        def describe(self):return {'kind':'cpu_fixture'}
        def reset(self,ids,seed,options,env_seeds):
            self.reset_ids.append(list(ids))
            for eid in ids:
                self.reset_specs[eid]=dict(seed=env_seeds[eid],**options['by_env'][str(eid)])
                self.ready[eid]=True;self.states[eid]=[0,0,0,0,0,0,.06]
            return self.rows(ids)
        def rows(self,ids,items=None):
            commands={} if items is None else {i['env_id']:i['actions'][0] for i in items}
            return [dict(env_id=eid,observation={'state':self.states[eid].tolist()},
                         info={'applied_actions':[commands[eid]]} if items else {},terminated=False,truncated=False) for eid in ids]
        def end_episode(self,ids):
            assert all(self.ready[i] for i in ids)
            self.ready[ids]=False
        def step(self,items):
            ids=[x['env_id'] for x in items]
            assert set(ids)==set(np.flatnonzero(self.ready))
            for item in items:self.states[item['env_id']]=item['actions'][-1]
            return self.rows(ids)
        def capture_for_recording(self,ids):
            image=np.zeros((480,640,3),np.uint8)
            return {eid:{'up':image,'wrist':image} for eid in ids}
        def step_for_recording(self,items):
            ids=[x['env_id'] for x in items];self.step_ids.append(ids)
            assert set(ids)==set(np.flatnonzero(self.ready))
            for item in items:self.states[item['env_id']]=np.asarray(item['actions'][0])+np.array([.0005,0,0,0,0,0,0])
            return dict(items=self.rows(ids,items),rgb=self.capture_for_recording(ids),
                        snapshots={i:value for i,value in ((int(k),v) for k,v in self.snapshot(ids).items())})
        def snapshot(self,ids):
            return {str(eid):dict(objects={'red':{'position_world':[.3,.1,.78],'orientation_wxyz':[1,0,0,0]},'blue':{'position_world':[.3,-.1,.78],'orientation_wxyz':[1,0,0,0]}}) for eid in ids}
        def close(self):self.closed=True
    factory=ModuleType('_manipulation_test_backend');factory.create_backend=Backend
    monkeypatch.setitem(sys.modules,'pressb.tasks.cube_stack.collection',module)
    monkeypatch.setitem(sys.modules,'_manipulation_test_backend',factory)
    monkeypatch.setattr(registry,'resolve',lambda *args:'_manipulation_test_backend:create_backend')
    monkeypatch.setattr(demonstrations,'Video',DummyVideo)
    def run(output,episodes=7,num_envs=3,max_batches=0,task='cube_stack',extra_args=()):
        if task=='tabletop_cleanup':
            module.PAIRS=(('gamepad','speaker'),('gamepad','lidar'),('speaker','lidar'))
            module.sample_episode=lambda config,seed,episode_index,env_id: dict(
                objects={k:{'center_xy':[.3,.1*j],'yaw_deg':0.} for j,k in enumerate(module.PAIRS[episode_index%3])})
            monkeypatch.setitem(sys.modules,'pressb.tasks.tabletop_cleanup.collection',module)
        monkeypatch.setattr(sys,'argv',['collect_manipulation.py','--task',task,'--output',str(output),
            '--episodes',str(episodes),'--num-envs',str(num_envs),'--seed','10','--noise-deg','0',
            '--max-batches',str(max_batches),*map(str,extra_args)])
        collector.main()
        # The CLI normally exits between resumes. Clear traceback cycles from
        # rejected planning futures when exercising it repeatedly in one process.
        gc.collect()
    return SimpleNamespace(run=run,module=module,engines=engines)


def test_post_motion_verification_never_pads_training_samples(tmp_path,harness):
    original=harness.module.plan_episode
    def with_validation(*args,**kwargs):
        plan=original(*args,**kwargs)
        plan['validation_hold_seconds']=.6
        return plan
    harness.module.plan_episode=with_validation
    def audit(history,plan):
        motion=[row for row in history if row['recorded']]
        validation=[row for row in history if not row['recorded']]
        return dict(passed=(len(motion)==len(plan['actions']) and len(validation)==18
            and [row['validation_index'] for row in validation]==list(range(18))
            and all(row['action_index']==len(plan['actions'])-1 for row in validation)))
    harness.module.assess_episode=audit
    harness.run(tmp_path,episodes=5,num_envs=3)
    for path in (tmp_path/'episodes').glob('episode_*'):
        plan=json.loads((path/'plan.json').read_text())
        metadata=json.loads((path/'metadata.json').read_text())
        data=np.load(path/'data.npz')
        history=[json.loads(line) for line in (path/'physics.jsonl').read_text().splitlines()]
        assert len(data['state'])==len(plan['actions'])+1
        assert len(history)==len(plan['actions'])+18
        assert metadata['validation_steps']==18 and not metadata['validation_in_training_data']
        np.testing.assert_allclose(data['state'][1:,0],data['action'][:-1,0]+.0005)
        np.testing.assert_array_equal(data['action'][-1],data['action'][-2])
        assert int((path/'up.mp4').read_text())==len(data['state'])
    assert json.loads((tmp_path/'progress.json').read_text())['accepted']==5
    profile=json.loads((tmp_path/'collection_profile.json').read_text())
    batches=[json.loads(line) for line in (tmp_path/'batch_profile.jsonl').read_text().splitlines()]
    assert profile['completed_batches']==len(batches)==2
    assert profile['session_accepted']==profile['counts']['accepted_episodes']==5
    frame_count=sum(len(np.load(path/'data.npz')['state']) for path in (tmp_path/'episodes').glob('episode_*'))
    motion_steps=frame_count-5
    assert profile['counts']['accepted_recorded_frames']==profile['counts']['recorded_frames']==frame_count
    assert profile['counts']['motion_env_steps']==profile['counts']['accepted_motion_env_steps']==motion_steps
    assert profile['counts']['validation_env_steps']==18*5
    backend=harness.engines[-1]
    capacity=3*len(backend.step_ids)
    assert profile['counts']['execution_world_control_steps']==len(backend.step_ids)
    assert sum(map(len,backend.step_ids))==motion_steps+18*5
    assert profile['motion_slot_utilization']==pytest.approx(motion_steps/capacity)
    assert profile['active_slot_utilization']==pytest.approx((motion_steps+18*5)/capacity)
    assert sum(profile['exclusive_stage_seconds'].values())<=profile['wall_seconds']
    assert all(value>=0 for value in profile['exclusive_stage_seconds'].values())
    assert sum(batch['counts']['accepted_recorded_frames'] for batch in batches)==frame_count
    assert sum(batch['configured_slot_control_intervals'] for batch in batches)==capacity
    assert all(batch['session_id']==profile['session_id'] for batch in batches)
    assert 'layout_env_id' not in json.loads((tmp_path/'manifest.json').read_text())


@pytest.mark.parametrize('task,count,num_envs',[('cube_stack',7,3),('tabletop_cleanup',500,8)])
def test_unequal_trajectories_and_partial_final_batch_are_time_aligned(tmp_path,harness,task,count,num_envs):
    harness.run(tmp_path,episodes=count,num_envs=num_envs,task=task)
    assert json.loads((tmp_path/'progress.json').read_text())['status']=='complete'
    episodes=sorted((tmp_path/'episodes').glob('episode_*'))
    assert len(episodes)==count
    backend=harness.engines[-1]
    assert backend.reset_ids==[list(range(num_envs))]*(count//num_envs)+[list(range(count%num_envs))]
    assert backend.closed and not backend.ready.any()
    assert any(len(ids)<num_envs for ids in backend.step_ids)
    pair_counts={}
    for path in episodes:
        metadata=json.loads((path/'metadata.json').read_text())
        pair=tuple(sorted(metadata['reset_spec']['objects']))
        pair_counts[pair]=pair_counts.get(pair,0)+1
        data=np.load(path/'data.npz')
        state,action=data['state'],data['action']
        assert metadata['frame_count']==len(state)==len(action)
        np.testing.assert_allclose(state[1:,0],action[:-1,0]+.0005)
        np.testing.assert_array_equal(action[-1],action[-2])
        np.testing.assert_allclose(data['sim_time'],np.arange(len(state))/30)
        assert int((path/'up.mp4').read_text())==len(state)==int((path/'wrist.mp4').read_text())
    if task=='tabletop_cleanup':assert sorted(pair_counts.values())==[166,167,167]


def test_rejected_plans_retry_missing_slots_and_resume_keeps_committed_episodes(tmp_path,harness):
    # First seed is args.seed + first attempt * 1009.
    harness.module.reject_seeds.add(1019)
    harness.run(tmp_path,episodes=5,num_envs=2,max_batches=1)
    committed={p.name:p.joinpath('metadata.json').read_bytes() for p in (tmp_path/'episodes').glob('episode_*')}
    assert set(committed)=={'episode_000001'}
    assert (tmp_path/'rejected/attempt_000001.json').is_file()
    batch=json.loads((tmp_path/'batch_profile.jsonl').read_text().splitlines()[0])
    assert batch['counts']['planning_rejections']==batch['counts']['accepted_episodes']==1
    assert batch['assignments']['0']['seed']==1019
    assert batch['assignments']['0']['objects']==['blue','red']
    harness.run(tmp_path,episodes=5,num_envs=2)
    assert len(list((tmp_path/'episodes').glob('episode_*')))==5
    for name,content in committed.items():assert (tmp_path/'episodes'/name/'metadata.json').read_bytes()==content
    assert len(set(harness.module.plan_calls))==len(harness.module.plan_calls)
    before=len(harness.engines)
    harness.run(tmp_path,episodes=5,num_envs=2)
    assert len(harness.engines)==before


def test_old_manifest_is_legacy_and_keeps_its_bytes_on_explicit_resume(tmp_path,harness):
    harness.run(tmp_path,episodes=3,num_envs=2,max_batches=1,extra_args=('--trajectory-profile','legacy'))
    path=tmp_path/'manifest.json'
    old=json.loads(path.read_text());old.pop('trajectory_profile')
    path.write_text(json.dumps(old))
    before=path.read_bytes()
    with pytest.raises(ValueError,match='different parameters'):
        harness.run(tmp_path,episodes=3,num_envs=2)
    gc.collect()
    harness.run(tmp_path,episodes=3,num_envs=2,extra_args=('--trajectory-profile','legacy'))
    assert path.read_bytes()==before
    assert json.loads((tmp_path/'progress.json').read_text())['accepted']==3


def test_sparse_workers_keep_global_pairs_and_disjoint_attempts_across_retry_and_resume(tmp_path,harness):
    shards=([17,100,499],[0,118,302])
    arguments=[]
    for worker,indices in enumerate(shards):
        path=tmp_path/f'indices_{worker}.json';path.write_text(json.dumps(indices))
        arguments.append(('--episode-indices',path,'--attempt-offset',384+worker,'--attempt-stride',2))
    # First worker retries its first global index after resuming. Its second
    # completed episode must remain byte-identical, with no seed reuse.
    harness.module.reject_seeds.add(10+386*1009)
    first=tmp_path/'worker0';second=tmp_path/'worker1'
    harness.run(first,episodes=500,num_envs=2,max_batches=1,task='tabletop_cleanup',extra_args=arguments[0])
    progress=json.loads((first/'progress.json').read_text())
    assert (progress['accepted'],progress['target'],progress['attempts'])==(1,3,388)
    assert progress['global_target_episodes']==500
    committed=(first/'episodes/episode_000100/metadata.json').read_bytes()
    harness.run(second,episodes=500,num_envs=2,task='tabletop_cleanup',extra_args=arguments[1])
    harness.run(first,episodes=500,num_envs=2,task='tabletop_cleanup',extra_args=arguments[0])
    assert (first/'episodes/episode_000100/metadata.json').read_bytes()==committed
    schedule=np.arange(500)%3;np.random.default_rng(10).shuffle(schedule)
    attempts=[]
    for worker,(output,indices) in enumerate(zip((first,second),shards)):
        manifest=json.loads((output/'manifest.json').read_text())
        assert manifest['target_episodes']==3 and manifest['global_target_episodes']==500
        assert manifest['episode_indices']==indices
        assert manifest['pair_schedule']==schedule.tolist()
        assert json.loads((output/'progress.json').read_text())['status']=='complete'
        paths=list((output/'episodes').glob('episode_*'))
        assert {int(path.name.split('_')[-1]) for path in paths}==set(indices)
        for path in paths:
            metadata=json.loads((path/'metadata.json').read_text())
            index=metadata['episode_index'];attempt=metadata['attempt_index'];attempts.append(attempt)
            assert attempt>385 and attempt%2==worker
            assert metadata['seed']==10+attempt*1009
            assert set(metadata['reset_spec']['objects'])==set(harness.module.PAIRS[schedule[index]])
    assert len(attempts)==len(set(attempts))==6
    assert len(harness.module.plan_calls)==len(set(harness.module.plan_calls))
    before=len(harness.engines)
    harness.run(first,episodes=500,num_envs=2,task='tabletop_cleanup',extra_args=arguments[0])
    assert len(harness.engines)==before


@pytest.mark.parametrize('indices',[[0,0],[500],[-1],[True],[1.5],{'a':1}])
def test_sparse_indices_reject_invalid_global_ids_before_creating_backend(tmp_path,harness,indices):
    path=tmp_path/'indices.json';path.write_text(json.dumps(indices))
    with pytest.raises(SystemExit) as exc:
        harness.run(tmp_path/'output',episodes=500,extra_args=('--episode-indices',path))
    assert exc.value.code==2 and not harness.engines


@pytest.mark.parametrize('extra_args',[('--attempt-offset','-1'),('--attempt-stride','0')])
def test_attempt_partition_arguments_are_validated(tmp_path,harness,extra_args):
    with pytest.raises(SystemExit) as exc:harness.run(tmp_path,extra_args=extra_args)
    assert exc.value.code==2 and not harness.engines


def test_fixed_layout_sampler_id_does_not_change_execution_environment_ids(tmp_path,harness):
    harness.run(tmp_path,episodes=4,num_envs=4,extra_args=('--layout-env-id','0'))
    assert harness.module.sample_eids==[0,0,0,0]
    assert harness.engines[-1].reset_ids==[[0,1,2,3]]
    assert set(harness.engines[-1].step_ids[0])=={0,1,2,3}
    assert json.loads((tmp_path/'manifest.json').read_text())['layout_env_id']==0
    batch=json.loads((tmp_path/'batch_profile.jsonl').read_text().splitlines()[0])
    assert len(batch['assignments'])==4
    assert all(item['layout_env_id']==0 for item in batch['assignments'].values())
    assert all(item['selected_for_recording'] for item in batch['assignments'].values())


def test_fixed_layout_sampler_id_must_be_nonnegative(tmp_path,harness):
    with pytest.raises(SystemExit) as exc:harness.run(tmp_path,extra_args=('--layout-env-id','-1'))
    assert exc.value.code==2 and not harness.engines


def test_attempt_pair_schedule_survives_rejections_and_different_parallelism(tmp_path,harness):
    # A rejected first attempt makes successful episode slot and attempt diverge.
    harness.module.reject_seeds.add(1019)
    outputs=[]
    for num_envs in (2,4):
        output=tmp_path/f'envs_{num_envs}'
        harness.run(output,episodes=8,num_envs=num_envs,max_batches=8//num_envs,
            task='tabletop_cleanup',extra_args=('--layout-env-id','0','--pair-schedule-by','attempt'))
        batches=[json.loads(line) for line in (output/'batch_profile.jsonl').read_text().splitlines()]
        assignments={item['attempt_index']:item for batch in batches for item in batch['assignments'].values()}
        manifest=json.loads((output/'manifest.json').read_text())
        assert manifest['pair_schedule_by']=='attempt'
        assert len(assignments)==8
        for attempt,item in assignments.items():
            expected=harness.module.PAIRS[manifest['pair_schedule'][(attempt-1)%len(manifest['pair_schedule'])]]
            assert item['objects']==sorted(expected)
        outputs.append(assignments)
    for attempt in range(1,9):
        for field in ('seed','objects','sampled_objects','layout_env_id'):
            assert outputs[0][attempt][field]==outputs[1][attempt][field]
    # Retry slot schedules really differ, so this checks more than all-success sampling.
    assert any(outputs[0][attempt]['episode_index']!=outputs[1][attempt]['episode_index']
               for attempt in range(1,9))


def test_empty_sparse_worker_completes_without_starting_simulation(tmp_path,harness):
    path=tmp_path/'indices.json';path.write_text('[]')
    harness.run(tmp_path/'output',episodes=500,extra_args=('--episode-indices',path))
    progress=json.loads((tmp_path/'output/progress.json').read_text())
    assert progress['status']=='complete' and progress['accepted']==progress['target']==0
    assert progress['global_target_episodes']==500 and not harness.engines


@pytest.mark.parametrize('task', ['cube_stack', 'tabletop_cleanup'])
def test_initialize_only_writes_manifest_without_resolving_backend_and_resumes(tmp_path,harness,monkeypatch,task):
    def forbidden_backend(*args):
        pytest.fail('Manifest initialization resolved a simulator backend')
    with monkeypatch.context() as context:
        context.setattr(registry,'resolve',forbidden_backend)
        harness.run(tmp_path,episodes=100,num_envs=8,task=task,extra_args=('--initialize-only',))
        first_manifest=(tmp_path/'manifest.json').read_bytes()
        first_progress=(tmp_path/'progress.json').read_bytes()
        harness.run(tmp_path,episodes=100,num_envs=8,task=task,extra_args=('--initialize-only',))
    assert not harness.engines
    assert not list((tmp_path/'episodes').iterdir())
    assert not (tmp_path/'collection_profile.json').exists()
    assert (tmp_path/'manifest.json').read_bytes()==first_manifest
    assert (tmp_path/'progress.json').read_bytes()==first_progress
    manifest=json.loads(first_manifest)
    assert manifest['target_episodes']==100 and manifest['trajectory_profile']=='compact'
    assert manifest['config']['num_envs']==8
    progress=json.loads(first_progress)
    assert progress['status']=='initialized' and progress['accepted']==progress['attempts']==0
    if task=='tabletop_cleanup':
        assert sorted(manifest['pair_schedule'].count(i) for i in range(3))==[33,33,34]
    harness.run(tmp_path,episodes=100,num_envs=8,max_batches=1,task=task)
    assert len(harness.engines)==1
    committed_progress=(tmp_path/'progress.json').read_bytes()
    harness.run(tmp_path,episodes=100,num_envs=8,task=task,extra_args=('--initialize-only',))
    assert len(harness.engines)==1
    assert (tmp_path/'manifest.json').read_bytes()==first_manifest
    assert (tmp_path/'progress.json').read_bytes()==committed_progress
