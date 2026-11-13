#!/usr/bin/env python3
"""Parallel physical demonstrations using the shared runtime and task adapters."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import importlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


class CollectionProfile:
    """Session-local wall-clock accounting; nested stages report exclusive time.

    Engine physics/render timers are a nested breakdown of step/reset/capture,
    not additional work to add to these collector-stage totals.
    """
    def __init__(self, output, task_id, num_envs, initial_accepted):
        self.output=output;self.task_id=task_id;self.num_envs=num_envs
        self.session_id=str(time.time_ns());self.started=time.perf_counter()
        self.initial_accepted=initial_accepted;self.seconds=Counter();self.inclusive=Counter()
        self.counts=Counter();self.stack=[];self.batch=None;self.batches=0

    @contextmanager
    def timed(self, name):
        item=[name,time.perf_counter(),0.];self.stack.append(item)
        try:yield
        finally:
            elapsed=time.perf_counter()-item[1];self.stack.pop()
            self.seconds[name]+=elapsed-item[2];self.inclusive[name]+=elapsed
            if self.stack:self.stack[-1][2]+=elapsed

    def begin_batch(self, index):
        self.batch=dict(index=index,started=time.perf_counter(),seconds=dict(self.seconds),
                        counts=dict(self.counts),active_histogram=Counter(),motion_histogram=Counter())

    def count(self, name, value=1):
        self.counts[name]+=value

    def end_batch(self, assignments, plans, rejections, num_envs, engine_timings):
        if self.batch is None:return
        before=self.batch
        counts={k:v-before['counts'].get(k,0) for k,v in self.counts.items()}
        seconds={k:v-before['seconds'].get(k,0.) for k,v in self.seconds.items()}
        steps=counts.get('execution_world_control_steps',0)
        capacity=num_envs*steps
        record=dict(schema_version=1,session_id=self.session_id,batch_index=before['index'],
            task_id=self.task_id,num_envs=num_envs,wall_seconds=time.perf_counter()-before['started'],
            assignments=assignments,planned_envs=len(plans),planning_rejections=rejections,
            planned_motion_steps={str(eid):len(plan['actions']) for eid,plan in plans.items()},
            planned_validation_seconds={str(eid):float(plan.get('validation_hold_seconds',0.)) for eid,plan in plans.items()},
            exclusive_stage_seconds=seconds,counts=counts,
            active_env_count_histogram=dict(before['active_histogram']),
            motion_env_count_histogram=dict(before['motion_histogram']),
            configured_slot_control_intervals=capacity,
            motion_slot_utilization=counts.get('motion_env_steps',0)/capacity if capacity else None,
            active_slot_utilization=(counts.get('motion_env_steps',0)+counts.get('validation_env_steps',0))/capacity if capacity else None,
            engine_cumulative_timings=deepcopy(engine_timings))
        with (self.output/'batch_profile.jsonl').open('a') as stream:
            stream.write(json.dumps(record,allow_nan=False)+'\n')
        self.batch=None;self.batches+=1

    def save(self, status, accepted, engine_timings=None):
        from pressb.runtime.recording import write
        elapsed=time.perf_counter()-self.started
        counts=dict(self.counts)
        capacity=self.num_envs*counts.get('execution_world_control_steps',0)
        report=dict(schema_version=1,session_id=self.session_id,task_id=self.task_id,status=status,
            num_envs=self.num_envs,initial_accepted=self.initial_accepted,
            session_accepted=accepted-self.initial_accepted,canonical_accepted=accepted,
            completed_batches=self.batches,wall_seconds=elapsed,
            exclusive_stage_seconds=dict(self.seconds),inclusive_stage_seconds=dict(self.inclusive),
            uninstrumented_seconds=max(0.,elapsed-sum(self.seconds.values())),counts=counts,
            accepted_frames_per_wall_second=counts.get('accepted_recorded_frames',0)/elapsed if elapsed else 0.,
            accepted_episodes_per_hour=(accepted-self.initial_accepted)*3600/elapsed if elapsed else 0.,
            motion_slot_utilization=counts.get('motion_env_steps',0)/capacity if capacity else None,
            active_slot_utilization=(counts.get('motion_env_steps',0)+counts.get('validation_env_steps',0))/capacity if capacity else None,
            engine_cumulative_timings=deepcopy(engine_timings or {}),
            semantics={
                'timing':'Exclusive stage seconds sum without nested double counting; engine timers break down reset/settle/capture/step and must not be added again.',
                'frames':'A recorded frame is one state/action row plus both camera images; terminal duplicated-target row is included, validation-only steps are excluded.',
                'throughput':'Session newly accepted frames/episodes divided by wall time; excludes later LeRobot export and simulator teardown.',
                'utilization':'Executed motion or active (motion+validation) env-steps divided by configured slots times executed world control intervals; settling excluded.',
                'batch_log':'Append-only, partition by session_id when collection resumes.'})
        write(self.output/'collection_profile.json',report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',required=True,choices=['cube_stack','tabletop_cleanup'])
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--episodes',type=int,default=500)
    parser.add_argument('--episode-indices',type=Path,
        help='JSON list of global episode indices assigned to this isolated worker')
    parser.add_argument('--attempt-offset',type=int,default=0)
    parser.add_argument('--attempt-stride',type=int,default=1,
        help='Increment between global attempt IDs; the first is offset + stride')
    parser.add_argument('--num-envs',type=int,default=8)
    parser.add_argument('--gpu',type=int,default=1)
    parser.add_argument('--seed',type=int,default=20261110)
    parser.add_argument('--layout-env-id',type=int,
        help='Optional fixed sampler env ID for layout-matched parallelism benchmarks; execution keeps real environment IDs')
    parser.add_argument('--pair-schedule-by',choices=['episode','attempt'],default='episode',
        help='Assign object pairs by successful episode slot, or by attempt for layout-matched benchmarks')
    parser.add_argument('--noise-deg',type=float,default=.12)
    parser.add_argument('--trajectory-profile',choices=['compact','legacy'],default='compact',
        help='Compact continuous demonstrations, or the original planner for comparisons')
    parser.add_argument('--settle-seconds',type=float,default=1.)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--initialize-only',action='store_true',
        help='Write/validate the canonical manifest on CPU without starting the simulator or collecting episodes')
    parser.add_argument('--max-batches',type=int,default=0)
    parser.add_argument('--max-attempts',type=int,default=5000)
    args=parser.parse_args()
    if min(args.episodes,args.num_envs,args.max_attempts)<1:parser.error('Counts must be positive')
    if args.attempt_offset<0:parser.error('Attempt offset must be nonnegative')
    if args.attempt_stride<1:parser.error('Attempt stride must be positive')
    if args.layout_env_id is not None and args.layout_env_id<0:parser.error('Layout env ID must be nonnegative')
    selected=list(range(args.episodes))
    if args.episode_indices is not None:
        try:selected=json.loads(args.episode_indices.read_text())
        except (OSError,ValueError) as exc:parser.error(f'Cannot read episode indices: {exc}')
        if (not isinstance(selected,list) or
            any(type(index) is not int or not 0<=index<args.episodes for index in selected)):
            parser.error('Episode indices must be a JSON list of integers in [0, episodes)')
        if len(set(selected))!=len(selected):parser.error('Episode indices must be unique')
    selected_set=set(selected)
    target=len(selected)
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    lock=(output/'.collection.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import numpy as np
    from pressb.runtime.recording import write
    from pressb.runtime.demonstrations import RawEpisode,perturb_plan
    from pressb.runtime.trajectory_metrics import plan_motion_metrics,measured_motion_metrics
    from pressb.registry import resolve
    module=importlib.import_module(f'pressb.tasks.{args.task}.collection')
    base=json.loads(args.config.read_text()) if args.config else None
    config=module.collection_config(base)
    config.update(num_envs=args.num_envs,gpu=args.gpu,horizon=32,
        observation_render_frames=1,reset_render_frames=2,
        max_episode_seconds=150.,terminate_on_success=False,log_dir=str(output/'simulation'))
    config['scene_overrides']={**config.get('scene_overrides',{}),'rendering':{
        **config.get('scene_overrides',{}).get('rendering',{}),'warmup_frames':8}}
    pair_schedule=None
    if hasattr(module,'PAIRS'):
        pair_schedule=(np.arange(args.episodes)%len(module.PAIRS))
        np.random.default_rng(args.seed).shuffle(pair_schedule)
        pair_schedule=pair_schedule.tolist()
    manifest=dict(schema_version=1,task_id=args.task,instruction=module.INSTRUCTION,
        target_episodes=target,seed=args.seed,noise_deg=args.noise_deg,
        trajectory_profile=args.trajectory_profile,
        settle_seconds=args.settle_seconds,config=config,success_only=True,
        format='raw_boundary_aligned_joint7_rgb',physics_after_reset='contact_and_gravity_only',
        pair_schedule=pair_schedule)
    if args.episode_indices is not None:
        manifest.update(global_target_episodes=args.episodes,episode_indices=selected)
    if args.attempt_offset or args.attempt_stride!=1:
        manifest.update(attempt_offset=args.attempt_offset,attempt_stride=args.attempt_stride)
    if args.layout_env_id is not None:manifest['layout_env_id']=args.layout_env_id
    if args.pair_schedule_by!='episode':manifest['pair_schedule_by']=args.pair_schedule_by
    manifest_path=output/'manifest.json'
    if manifest_path.exists():
        existing=json.loads(manifest_path.read_text())
        existing.setdefault('trajectory_profile','legacy')
        if existing!=manifest:
            raise ValueError('Existing collection has different parameters; use a new output')
    else:
        write(manifest_path,manifest)
    for name in ('episodes','attempts','rejected'):(output/name).mkdir(exist_ok=True)
    completed={int(p.name.split('_')[-1]) for p in (output/'episodes').glob('episode_*')
               if (p/'metadata.json').is_file()} & selected_set
    progress_path=output/'progress.json'
    previous=json.loads(progress_path.read_text()) if progress_path.exists() else {}
    attempt=max(args.attempt_offset,int(previous.get('attempts',args.attempt_offset)))
    if (attempt-args.attempt_offset)%args.attempt_stride:
        raise ValueError('Saved attempt counter is not on this worker attempt stride')
    if args.initialize_only:
        # Initialization must not overwrite an existing worker's progress or
        # profiler receipts. The normal collector can resume this manifest.
        if not progress_path.exists():
            value=dict(task_id=args.task,status='initialized',accepted=len(completed),
                       target=target,attempts=attempt,updated_unix=time.time())
            if args.episode_indices is not None:value['global_target_episodes']=args.episodes
            write(progress_path,value)
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
        return
    batch=0;start=time.time();backend=None
    profile=CollectionProfile(output,args.task,args.num_envs,len(completed))
    writers={}
    def progress(status='running',**extra):
        value=dict(task_id=args.task,status=status,accepted=len(completed),target=target,
                   attempts=attempt,session_batches=batch,updated_unix=time.time(),session_elapsed_s=time.time()-start,
                   **extra)
        if args.episode_indices is not None:value['global_target_episodes']=args.episodes
        write(progress_path,value);print(json.dumps(value),flush=True)
        profile.save(status,len(completed),None if backend is None else backend.timings)
    try:
        if len(completed)>=target:progress('complete');return
        progress(stage='starting_simulator')
        path,attribute=resolve('task',args.task).split(':')
        with profile.timed('startup'):
            backend=getattr(importlib.import_module(path),attribute)(config)
            normalized=backend.task.config
            write(output/'runtime_metadata.json',backend.describe())
        fps=normalized['control_hz'];home=normalized['scene']['initial_state_joint_rad_m']
        settle_steps=round(args.settle_seconds*fps)
        if settle_steps<1:raise ValueError('Require a positive settling interval')
        while len(completed)<target and attempt<args.max_attempts:
            if args.max_batches and batch>=args.max_batches:progress('paused_batch_limit');break
            profile.begin_batch(batch+1)
            sampling_started=time.perf_counter()
            missing=[i for i in selected if i not in completed][:args.num_envs]
            ids=list(range(len(missing)))
            # Initial reset of a shared world includes every slot. Extra slots
            # on a short final batch are immediately retired without recording.
            reset_ids=list(range(args.num_envs)) if all(s is None for s in backend.reset_specs) else ids
            options={};seeds={};assignments={};specs={}
            for eid in reset_ids:
                episode=missing[eid] if eid<len(missing) else missing[0]
                attempt+=args.attempt_stride;seed=args.seed+attempt*1009
                sample_eid=eid if args.layout_env_id is None else args.layout_env_id
                if hasattr(module,'sample_episode'):
                    pair_index=episode if args.pair_schedule_by=='episode' else (attempt-1)%len(pair_schedule)
                    spec=module.sample_episode(normalized,seed,pair_schedule[pair_index],env_id=sample_eid)
                else:
                    spec=backend.task.sample_reset(sample_eid,seed,{})
                opts={'objects':{name:{key:pose[key] for key in ('center_xy','yaw_deg')}
                                 for name,pose in spec['objects'].items()}}
                options[str(eid)]=opts;seeds[eid]=seed
                assignments[eid]=dict(episode_index=episode,attempt_index=attempt,seed=seed)
            profile.seconds['sampling']+=time.perf_counter()-sampling_started
            profile.inclusive['sampling']=profile.seconds['sampling']
            profile.count('sampled_layouts',len(reset_ids))
            batch+=1;progress(stage='reset',batch_assignments=assignments)
            with profile.timed('reset'):
                rows=backend.reset(reset_ids,args.seed,{'by_env':options},env_seeds=seeds)
                for eid in reset_ids:
                    specs[eid]=deepcopy(backend.reset_specs[eid])
                extra=[eid for eid in reset_ids if eid not in ids]
                if extra:backend.end_episode(extra)
            profile_assignments={eid:{**value,'objects':sorted(specs[eid]['objects']),
                'sampled_objects':deepcopy(options[str(eid)]['objects']),
                'selected_for_recording':eid in ids,
                'layout_env_id':eid if args.layout_env_id is None else args.layout_env_id}
                for eid,value in assignments.items()}
            remaining=settle_steps
            with profile.timed('settle'):
                while remaining:
                    live=[i for i in ids if backend.ready[i]]
                    if not live:break
                    count=min(32,remaining)
                    rows=backend.step([dict(env_id=i,actions=[home]*count) for i in live]);remaining-=count
                    profile.count('settling_env_steps',len(live)*count)
            with profile.timed('snapshot'):
                snapshot=backend.snapshot(ids)
                write(output/f'batch_ending_attempt_{attempt:06d}_settled.json',snapshot)
            plans={};rejections=[]
            def plan_one(eid):
                if not backend.ready[eid]:raise ValueError('Terminated during settling')
                base_world=np.asarray(backend.task.scene['robot_base_position_world'])+backend.engine.offsets[eid]
                result=module.plan_episode(normalized,specs[eid],snapshot[str(eid)],base_world,
                                          trajectory_profile=args.trajectory_profile)
                result=perturb_plan(result,assignments[eid]['seed'],amplitude_deg=args.noise_deg,
                    joint_lower_rad=backend.task.lower[:6],joint_upper_rad=backend.task.upper[:6])
                # Store exactly the runtime contract's float32 targets, not
                # higher-precision labels that differ from dispatched values.
                result['actions']=np.asarray(result['actions'],dtype=np.float32).tolist()
                result['command_dtype']='float32'
                result['motion_metrics']=plan_motion_metrics(result)
                actions=np.asarray(result['actions'])
                if np.any(actions<backend.task.lower-1e-7) or np.any(actions>backend.task.upper+1e-7):
                    raise ValueError('Perturbed trajectory exceeds joint limits')
                return result
            planning_started=time.perf_counter()
            with ThreadPoolExecutor(max_workers=min(4,len(ids))) as pool:
                futures={eid:pool.submit(plan_one,eid) for eid in ids}
                for eid,future in futures.items():
                    try:plans[eid]=future.result()
                    except Exception as exc:
                        rejected={**assignments[eid],'stage':'planning','error':f'{type(exc).__name__}: {exc}'}
                        rejections.append(rejected)
                        write(output/'rejected'/f"attempt_{assignments[eid]['attempt_index']:06d}.json",rejected)
                        if backend.ready[eid]:backend.end_episode([eid])
            profile.seconds['planning']+=time.perf_counter()-planning_started
            profile.inclusive['planning']=profile.seconds['planning']
            profile.count('planning_rejections',len(rejections))
            profile.count('planned_episodes',len(plans))
            if not plans:
                profile.end_batch(profile_assignments,plans,rejections,args.num_envs,backend.timings)
                progress(stage='all_plans_rejected',rejections=rejections);continue
            live=sorted(plans)
            current={r['env_id']:r for r in rows if r['env_id'] in plans}
            with profile.timed('initial_capture'):rgb=backend.capture_for_recording(live)
            histories={eid:[] for eid in live};writers={}
            validation_steps={eid:int(np.ceil(float(plans[eid].get('validation_hold_seconds',0.))*fps))
                              for eid in live}
            if any(value<0 for value in validation_steps.values()):
                raise ValueError('Validation duration must be nonnegative')
            with profile.timed('writer_init'):
                for eid in live:
                    directory=output/'attempts'/f"attempt_{assignments[eid]['attempt_index']:06d}"
                    writers[eid]=RawEpisode(directory,fps=fps)
                    write(directory/'plan.json',plans[eid]);write(directory/'settled.json',snapshot[str(eid)])
            progress(stage='executing',planned=len(plans),rejections=rejections)
            frame=0
            terminal_recorded=set()
            while live:
                profile.batch['active_histogram'][len(live)]+=1
                motion_count=sum(frame<len(plans[eid]['actions']) for eid in live)
                profile.batch['motion_histogram'][motion_count]+=1
                profile.count('execution_world_control_steps')
                with profile.timed('record_frame'):
                    for eid in live:
                        if frame<len(plans[eid]['actions']):
                            writers[eid].frame(current[eid]['observation']['state'],plans[eid]['actions'][frame],rgb[eid])
                            profile.count('recorded_frames')
                # Finished trajectories keep their last target during physical
                # verification, without padding their training video/state/action
                # stream or deleting internal 30 Hz intervals.
                indices={eid:min(frame,len(plans[eid]['actions'])-1) for eid in live}
                with profile.timed('step_for_recording'):
                    result=backend.step_for_recording([dict(env_id=eid,actions=[plans[eid]['actions'][indices[eid]]]) for eid in live])
                snapshots=result['snapshots'];rgb=result['rgb']
                log_started=time.perf_counter()
                terminal_seconds=0.
                for row in result['items']:
                    eid=row['env_id'];current[eid]=row
                    if not np.allclose(row['info']['applied_actions'][0],plans[eid]['actions'][indices[eid]],rtol=0,atol=1e-9):
                        raise ValueError('Recorded target differs from actually applied action')
                    validating=frame>=len(plans[eid]['actions'])
                    profile.count('validation_env_steps' if validating else 'motion_env_steps')
                    record=dict(action_index=indices[eid],phase='validation' if validating else plans[eid]['phases'][frame],
                        recorded=not validating,
                        state=row['observation']['state'],objects=snapshots[eid]['objects'],
                        info=deepcopy(row['info']),terminated=row['terminated'],truncated=row['truncated'])
                    if validating:record['validation_index']=frame-len(plans[eid]['actions'])
                    writers[eid].physical(record);histories[eid].append(record)
                    if eid not in terminal_recorded and (frame+1==len(plans[eid]['actions']) or not backend.ready[eid]):
                        terminal_started=time.perf_counter()
                        writers[eid].frame(row['observation']['state'],plans[eid]['actions'][indices[eid]],rgb[eid])
                        terminal_seconds+=time.perf_counter()-terminal_started
                        profile.count('recorded_frames')
                        terminal_recorded.add(eid)
                profile.seconds['physical_log']+=time.perf_counter()-log_started-terminal_seconds
                profile.inclusive['physical_log']=profile.seconds['physical_log']
                profile.seconds['record_frame']+=terminal_seconds
                profile.inclusive['record_frame']+=terminal_seconds
                finished=[eid for eid in live if not backend.ready[eid]
                          or frame+1==len(plans[eid]['actions'])+validation_steps[eid]]
                for eid in finished:
                    writer=writers[eid]
                    audit_started=time.perf_counter()
                    audit=module.assess_episode(histories[eid],plans[eid])
                    metadata={**assignments[eid],'task_id':args.task,'instruction':module.INSTRUCTION,
                        'success':bool(audit['passed']),'audit':audit,'reset_spec':specs[eid],
                        'env_id':eid,'num_envs':args.num_envs,
                        'noise':plans[eid]['noise'],'settling_seconds':args.settle_seconds,
                        'trajectory_profile':args.trajectory_profile,
                        'validation_steps':sum(row['phase']=='validation' for row in histories[eid]),
                        'validation_in_training_data':False,
                        'motion_metrics':measured_motion_metrics(writer.states,fps)}
                    profile.seconds['audit']+=time.perf_counter()-audit_started
                    profile.inclusive['audit']=profile.seconds['audit']
                    with profile.timed('commit'):
                        if audit['passed']:
                            index=assignments[eid]['episode_index']
                            writer.commit(output/'episodes'/f'episode_{index:06d}',metadata);completed.add(index)
                            profile.count('accepted_recorded_frames',len(writer.states))
                            profile.count('accepted_motion_env_steps',sum(row['recorded'] for row in histories[eid]))
                            profile.count('accepted_episodes')
                        else:
                            writer.close();write(writer.directory/'metadata.json',metadata)
                            writer.directory.rename(output/'rejected'/writer.directory.name)
                            profile.count('execution_rejections')
                    if backend.ready[eid]:backend.end_episode([eid])
                    del writers[eid]
                    progress(stage='episode_finished',last_episode=metadata['episode_index'],last_success=metadata['success'])
                live=[eid for eid in live if eid not in finished];frame+=1
                if frame%300==0:progress(stage='executing',frame=frame,active=len(live))
            write(output/'timings.json',backend.timings)
            profile.end_batch(profile_assignments,plans,rejections,args.num_envs,backend.timings)
        if len(completed)>=target:progress('complete')
        elif attempt>=args.max_attempts:progress('attempt_limit')
    except Exception as exc:
        progress('failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        raise
    finally:
        for writer in writers.values():
            try:writer.close()
            except Exception:traceback.print_exc()
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
        profile.save(json.loads(progress_path.read_text()).get('status','unknown'),len(completed),
                     None if backend is None else backend.timings)
        if backend is not None:backend.close()


if __name__=='__main__':main()
