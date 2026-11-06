#!/usr/bin/env python3
"""Collect physical tabletop-button demonstrations into synchronized raw episodes.

A single Isaac application builds independent stages. Accepted episodes finish
at the first 30 Hz boundary after a contact-validated press. The offline LeRobot
exporter consumes the raw arrays and the two original-time RGB videos.
By default, exit code 75 requests a fresh process after 50 accepted episodes;
rerun the same command with --resume until exit code 0 confirms the full quota.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
import gc
import faulthandler
import json
import os
import signal
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from pressb.runtime.recording import Video, write
from pressb.runtime.isaac import build_piper_runtime, close_piper_runtime, make_camera, set_intrinsics

from pressb.table_buttons.execution import (
    contact_limited_command, in_press_window, press_phase_times,
)


def now():
    return datetime.now(timezone.utc).isoformat()



def collect_episode(app, args, cfg, path_times, path_commands, plan_report):
    import numpy as np
    from PIL import Image
    from pxr import UsdGeom, UsdPhysics, PhysicsSchemaTools
    import omni.physx
    from isaacsim.core.prims import SingleRigidPrim
    from isaacsim.core.utils.types import ArticulationAction
    from pressb.pick_place.scene import build_tripod
    from pressb.table_buttons.scene import build_scene
    from pressb.table_buttons.task import COLORS, TableButtonOutcome, task_instruction, project_base_points

    phase_times = press_phase_times(plan_report)
    runtime = build_piper_runtime(cfg, build_scene, asset_root=ROOT, camera_hardware_builder=build_tripod)
    world = runtime.world
    stage = runtime.stage
    scene = runtime.scene
    base = runtime.base
    robot = runtime.robot
    controller = runtime.controller
    indices = runtime.indices
    cameras = runtime.cameras
    camera_specs = runtime.camera_specs
    wrist_info = runtime.wrist_info
    initial = runtime.initial
    initial_state = runtime.initial_state
    full_command = runtime.full_command
    dt = runtime.dt
    fps = runtime.fps
    stride = runtime.stride
    warmup_frames = runtime.warmup_frames
    subframes = runtime.subframes
    caps = {color: SingleRigidPrim(prim_path=spec['body_path'], name=f'cap_{color}')
            for color, spec in scene['buttons'].items()}
    for cap in caps.values():
        cap.initialize()
    tick = [0]
    contact_forces = {color: 0. for color in COLORS}
    contact_events, unexpected_contacts = [], []

    def on_contact(headers, data):
        for header in headers:
            if not header.num_contact_data:
                continue
            actors = [str(PhysicsSchemaTools.intToSdfPath(value)) for value in (header.actor0, header.actor1)]
            force = sum(float(np.linalg.norm(data[i].impulse)) for i in
                        range(header.contact_data_offset, header.contact_data_offset+header.num_contact_data)) / dt
            fingers = [name for name in ('link7', 'link8') if f'/World/Piper/{name}' in actors]
            for color, spec in scene['buttons'].items():
                if spec['body_path'] in actors and fingers:
                    contact_forces[color] += force
                    contact_events.append(dict(tick=tick[0], color=color, fingers=fingers, force_n=force))
            moving_robot = [actor for actor in actors if actor.startswith('/World/Piper/')
                            and actor != '/World/Piper/base_link']
            self_contact = all(actor.startswith('/World/Piper/') for actor in actors) and actors[0] != actors[1]
            environment_contact = bool(moving_robot) and any(not actor.startswith('/World/Piper/') and
                not any(spec['body_path'] == actor for spec in scene['buttons'].values()) for actor in actors)
            non_finger_cap_contact = not fingers and any(spec['body_path'] in actors for spec in scene['buttons'].values())
            if force > .02 and moving_robot and (self_contact or environment_contact or non_finger_cap_contact):
                unexpected_contacts.append(dict(tick=tick[0], actors=actors, force_n=force))

    subscription = omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(on_contact)
    for _ in range(24):
        world.step(render=False)
    for _ in range(warmup_frames):
        world.render()
    for name, camera in cameras.items():
        Image.fromarray(np.asarray(camera.get_rgba())[..., :3]).save(args.output/f'initial_{name}.jpg', quality=95)
    for name, value in zip(robot.dof_names, initial):
        prim = stage.GetPrimAtPath(f'/World/Piper/joints/{name}')
        if prim.IsA(UsdPhysics.RevoluteJoint):
            UsdPhysics.DriveAPI(prim, 'angular').GetTargetPositionAttr().Set(float(np.rad2deg(value)))
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            UsdPhysics.DriveAPI(prim, 'linear').GetTargetPositionAttr().Set(float(value))
    report = dict(task='piper_table_buttons', mode='dataset_collection', config=cfg, scene=scene,
                  initial_q_target=initial[indices].tolist(), initial_q_actual=robot.get_joint_positions()[indices].tolist(),
                  wrist_camera=wrist_info, physics_dt=dt, video_fps=fps, seed=args.seed,
                  policy_or_RL_evaluation=False, controller='scripted bounded IK, 30 Hz zero-order hold',
                  button_motion='PhysX spring joint and contact only; no cap teleport or scripted press state')
    cache = UsdGeom.XformCache()
    report['cameras'] = {name: dict(path=path, resolution=size,
        intrinsics=np.asarray(cameras[name].get_intrinsics_matrix()).tolist(),
        world_from_camera_usd_at_initial=np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path))).T.tolist())
        for name, (path, size) in camera_specs.items()}
    centers = [np.asarray(scene['buttons'][color]['neutral_top_world'])-base for color in COLORS]
    projected, depth = project_base_points(centers, cfg['up_camera'])
    report['button_pixels_up'] = {color: dict(xy=projected[i].tolist(), depth_m=float(depth[i])) for i,color in enumerate(COLORS)}
    report['neutral_travel_m'] = {color: float(scene['buttons'][color]['cap_rest_world'][2]-cap.get_world_pose()[0][2]) for color,cap in caps.items()}

    from pressb.table_buttons.collection import measured_state, validate_raw_arrays
    # Static settling and rendering are excluded from the demonstration timeline.
    contact_events.clear()
    unexpected_contacts.clear()
    for name in COLORS:
        contact_forces[name] = 0.
    videos = {name: Video(args.output/f'{name}.mp4', size, fps)
              for name, (_,size) in camera_specs.items()}
    checker = TableButtonOutcome(args.target, **cfg['success'])
    frame_times, states, actions = [], [], []
    physics = []
    first_wrong_press = None
    wrong_contact = False
    held_command = None
    previous_command = initial_state.copy()
    total_intervals = round(path_times[-1]*fps)
    result = None
    maximum_depth = {name: 0. for name in COLORS}
    maximum_force = {name: 0. for name in COLORS}

    def physical_record(command):
        q_actual = robot.get_joint_positions()[indices].copy()
        travels = np.array([scene['buttons'][name]['cap_rest_world'][2]-caps[name].get_world_pose()[0][2]
                            for name in COLORS], dtype=float)
        forces = np.array([contact_forces[name] for name in COLORS], dtype=float)
        physics.append(dict(time_s=tick[0]*dt, q_actual=q_actual,
            q_target=full_command(command)[indices].copy(), travel_m=travels,
            finger_force_n=forces))
        return q_actual, dict(zip(COLORS, travels)), dict(zip(COLORS, forces))

    physical_record(initial_state)
    stop_reason = 'timeout'
    try:
        for frame_index in range(total_intervals+1):
            seconds = frame_index/fps
            terminal = bool(result and (result['terminated'] or unexpected_contacts or wrong_contact)) or frame_index == total_intervals
            if terminal:
                command = previous_command.copy()
            else:
                command = np.array([np.interp(seconds, path_times, path_commands[:,j]) for j in range(7)])
                command = contact_limited_command(
                    seconds, command, held_command, path_times, path_commands, phase_times)
            # World.render does not advance physics. Both RGB views, measured
            # state and the next applied action describe this same boundary.
            if frame_index:
                for _ in range(subframes):
                    world.render()
            frame_times.append(seconds)
            states.append(measured_state(robot.get_joint_positions()[indices]))
            actions.append(command.copy())
            for name, camera in cameras.items():
                rgb = np.asarray(camera.get_rgba())[..., :3]
                if rgb.shape != (480,640,3):
                    raise RuntimeError(f'Unexpected {name} RGB shape: {rgb.shape}')
                videos[name].add(np.ascontiguousarray(rgb))
            if terminal:
                if unexpected_contacts:
                    stop_reason = 'unexpected_contact'
                elif wrong_contact:
                    stop_reason = 'non_target_finger_contact'
                elif first_wrong_press is not None:
                    stop_reason = 'wrong_button'
                elif result and result['success']:
                    stop_reason = 'success'
                break
            previous_command = command.copy()
            # Exactly one action per 30 Hz interval. Contact feedback changes
            # only the next interval; a recorded action is never rewritten.
            q_target = full_command(command)
            controller.apply_action(ArticulationAction(joint_positions=q_target))
            for _ in range(stride):
                tick[0] += 1
                for name in COLORS:
                    contact_forces[name] = 0.
                world.step(render=False)
                _, travels, forces = physical_record(command)
                wrong_contact = wrong_contact or any(forces[name] > .02 for name in COLORS if name != args.target)
                measured_seconds = tick[0]*dt
                result = checker.update(measured_seconds, travels, forces)
                # Continue recording press events after the outcome latch, so
                # a non-target press in the terminal interval cannot be hidden.
                wrong = [name for name in result['press_events'] if name != args.target]
                if wrong and first_wrong_press is None:
                    first_wrong_press = dict(color=wrong[0], time_s=measured_seconds)
                if held_command is None and in_press_window(seconds, phase_times) and travels[args.target] >= cfg['success']['press_depth_m']+.0002:
                    held_command = command.copy()
                for name in COLORS:
                    maximum_depth[name] = max(maximum_depth[name], travels[name])
                    maximum_force[name] = max(maximum_force[name], forces[name])
        frames = dict(sim_time=np.asarray(frame_times), state=np.asarray(states), action=np.asarray(actions))
        arrays = {key: np.asarray([row[key] for row in physics]) for key in physics[0]}
        validate_raw_arrays(frames, arrays, fps=fps, physics_dt=dt)
        np.savez_compressed(args.output/'data.npz', **frames)
        np.savez_compressed(args.output/'physics.npz', **arrays)
        success = bool(result and result['success'] and first_wrong_press is None and not unexpected_contacts
                       and not wrong_contact)
        if not success and stop_reason == 'success':
            stop_reason = 'wrong_button'
        report.update(success=success, outcome=result, stop_reason=stop_reason,
            duration_s=float(frame_times[-1]), first_video_sample_time_s=0.,
            first_success_time_s=checker.first_success_time_s,
            video_frames={name:video.count for name,video in videos.items()},
            first_wrong_press=first_wrong_press, maximum_travel_m=maximum_depth,
            maximum_finger_force_n=maximum_force,
            unexpected_contact_count=len(unexpected_contacts), finger_contact_events=len(contact_events),
            maximum_arm_tracking_error_rad=float(np.max(np.abs(arrays['q_actual'][:,:6]-arrays['q_target'][:,:6]))),
            plan=plan_report, phase_times=phase_times,
            motion_template=plan_report.get('motion_template', {'mode': 'shared'}),
            action_timing='30 Hz next interval targets, each held for four physics steps',
            terminal_action='repeat_previous_interval_target',
            trajectory_end='first control boundary after physical success, without retreat')
        metadata = dict(schema_version='pressb.table_buttons.raw.v1',
            episode_index=args.episode_index, group=f'{args.profile}_{args.target}',
            spacing_profile=args.profile, target_color=args.target,
            instruction=task_instruction(args.target), seed=args.seed, fps=fps, physics_dt=dt,
            camera_resolution=[640,480], camera_names=['up','wrist'],
            state_units=['rad']*6+['m'], action_units=['rad']*6+['m'],
            gripper_representation='full_width',
            phase_times=phase_times,
            motion_template=plan_report.get('motion_template', {'mode': 'shared'}),
            positions_xy_m={name:cfg['buttons'][name]['center_xy'] for name in COLORS},
            spacing=cfg['sampling_domain']['sampling'],
            initial_state_target_rad_m=initial_state.tolist(),
            initial_state_measured_rad_m=frames['state'][0].tolist(),
            physics_joint_names=[f'joint{i}' for i in range(1,9)], button_color_order=list(COLORS),
            success=success, first_success_time_s=checker.first_success_time_s,
            duration_s=float(frame_times[-1]), frame_count=len(frames['sim_time']),
            terminal_action='repeat_previous_interval_target', stop_reason=stop_reason,
            capture_alignment='RGB and measured joints at identical world time; render-only subframes do not advance physics',
            controller='scripted bounded IK; not a learned-policy demonstration',
            physical_outcome_thresholds=cfg['success'])
        for name,camera in cameras.items():
            Image.fromarray(np.asarray(camera.get_rgba())[..., :3]).save(args.output/f'final_{name}.jpg',quality=95)
        write(args.output/'contacts.json',contact_events)
        write(args.output/'unexpected_contacts.json',unexpected_contacts)
        write(args.output/'metadata.json',metadata)
        write(args.output/'report.json',report)
        return report
    finally:
        for video in videos.values():
            video.close()
        subscription = None
        close_piper_runtime(runtime)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/piper_table_buttons.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--profiles',nargs='+',choices=('near','medium','far'),default=['near','medium','far'])
    parser.add_argument('--targets',nargs='+',choices=('red','yellow','blue'),default=['red','yellow','blue'])
    parser.add_argument('--episodes-per-group',type=int,default=100)
    parser.add_argument('--seed-start',type=int,default=0)
    parser.add_argument('--max-retries',type=int,default=50)
    parser.add_argument('--max-new-episodes',type=int,default=50,
                        help='Exit 75 after this many new accepted episodes to release stage caches; rerun with --resume until exit 0. Default 50; 0 disables')
    parser.add_argument('--gpu',type=int,default=1)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    if args.episodes_per_group < 1 or args.max_retries < 1 or args.seed_start < 0 or args.max_new_episodes < 0:
        parser.error('Episode quota, max retries and seeds must be nonnegative/positive')
    if len(set(args.profiles)) != len(args.profiles) or len(set(args.targets)) != len(args.targets):
        parser.error('Profiles and targets must not contain duplicates')
    cfg=json.loads(args.config.read_text())
    if cfg['fps'] != 30 or abs(cfg['physics_dt']-1/120)>1e-12:
        parser.error('This dataset contract requires 30 Hz cameras/actions and 120 Hz physics')
    args.output=args.output.resolve()
    if args.output.exists() and not args.resume:
        parser.error('Output already exists; use --resume to continue matching plan')
    args.output.mkdir(parents=True,exist_ok=True)
    plan=dict(schema_version='pressb.table_buttons.collection_plan.v1',
              config=cfg, profiles=args.profiles, targets=args.targets,
              episodes_per_group=args.episodes_per_group, seed_start=args.seed_start,
              max_retries=args.max_retries, fps=30, physics_hz=120,
              instructions={color:f'Press the {color} button.' for color in args.targets},
              seed_rule='first attempt seed_start+episode_index; retry 1000000+1000*(seed_start+episode_index)+retry',
              success_only=True, demonstration_end='first control boundary after physical success')
    plan_path=args.output/'plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            parser.error('Existing output plan differs; use a separate output root')
    else:
        write(plan_path,plan)
    status=dict(state='starting',started_at=now(),completed={},rejected=0,
                total_requested=len(args.profiles)*len(args.targets)*args.episodes_per_group,
                gpu=args.gpu)
    accepted_this_process=0
    stop_after_episode=[False]
    signal.signal(signal.SIGTERM, lambda *_: stop_after_episode.__setitem__(0,True))
    status_path=args.output/'status.json'
    write(status_path,status)
    from pressb.kinematics import PiperKinematics
    from pressb.table_buttons.collection import LayoutPlanner
    planner=LayoutPlanner(PiperKinematics(ROOT/cfg['robot_urdf'],tip_offset=.1358),cfg)
    # Import hpp-fcl/Assimp before Kit loads its own Assimp build. Reversing
    # this order resolves hpp-fcl against an incompatible shared library.
    for profile in args.profiles:
        planner.placement(profile)
    layout_dir=args.output/'layouts'
    layout_dir.mkdir(exist_ok=True)
    app=None
    os.environ.setdefault('OMNI_KIT_ACCEPT_EULA','YES')
    try:
        from isaacsim import SimulationApp
        app=SimulationApp({'headless':True,'active_gpu':args.gpu,'physics_gpu':args.gpu,
            'multi_gpu':False,'renderer':'RayTracedLighting','anti_aliasing':1,'fast_shutdown':True,
            'limit_cpu_threads':8,
            'create_new_stage':False,
            'extra_args':['--/app/asyncRendering=false','--/rtx/ecoMode/enabled=false',
                '--/validate/p2p/enabled=false','--/validate/iommu/enabled=false',
                '--/plugins/carb.tasking.plugin/threadCount=8','--/plugins/omni.tbb.globalcontrol/maxThreadCount=8']})
        from isaacsim.core.api import World
        from isaacsim.core.utils.stage import create_new_stage
        for profile in args.profiles:
            for target in args.targets:
                group=f'{profile}_{target}'
                group_dir=args.output/group
                group_dir.mkdir(exist_ok=True)
                status['completed'][group]=0
                for episode_index in range(args.episodes_per_group):
                    destination=group_dir/f'episode_{episode_index:06d}'
                    if destination.exists():
                        metadata=json.loads((destination/'metadata.json').read_text())
                        if not metadata['success'] or metadata['episode_index'] != episode_index or metadata['group'] != group:
                            raise RuntimeError(f'Existing accepted episode is invalid: {destination}')
                        status['completed'][group]+=1
                        continue
                    for retry in range(args.max_retries):
                        seed=args.seed_start+episode_index if retry==0 else 1000000+1000*(args.seed_start+episode_index)+retry
                        pending=args.output/'pending'/group/f'episode_{episode_index:06d}_seed{seed}'
                        rejected=args.output/'rejected'/group/pending.name
                        if rejected.exists():
                            continue
                        if pending.exists():
                            # An interrupted write is kept for diagnosis; it cannot
                            # be promoted into accepted data by a resumed process.
                            interrupted=args.output/'interrupted'/group/(pending.name+'_'+str(time.time_ns()))
                            interrupted.parent.mkdir(parents=True,exist_ok=True)
                            pending.rename(interrupted)
                        pending.mkdir(parents=True)
                        start=time.monotonic()
                        status.update(state='collecting',current=dict(group=group,episode_index=episode_index,seed=seed,retry=retry))
                        write(status_path,status)
                        print(json.dumps(dict(event='episode_start',**status['current'])),flush=True)
                        attempt=dict(group=group,episode_index=episode_index,seed=seed,retry=retry,started_at=now())
                        phase='planning'
                        try:
                            layout_path=layout_dir/f'{profile}_seed{seed}.json'
                            if layout_path.exists():
                                layout=json.loads(layout_path.read_text())
                            else:
                                layout=planner.sample(profile,seed)
                                write(layout_path,layout)
                            effective,times,commands,preflight=planner.prepare(profile,seed,target,layout)
                            write(pending/'config.json',effective)
                            write(pending/'preflight.json',preflight)
                            phase='physical_collection'
                            if World.instance() is not None:
                                old_world=World.instance()
                                old_world.stop()
                                World.clear_instance()
                                del old_world
                                gc.collect()
                            create_new_stage()
                            for _ in range(3):
                                app.update()
                            episode_args=SimpleNamespace(output=pending,seed=seed,target=target,
                                profile=profile,episode_index=episode_index)
                            report=collect_episode(app,episode_args,effective,times,commands,preflight)
                            accepted=bool(report['success'])
                            attempt.update(accepted=accepted,stop_reason=report['stop_reason'],
                                duration_s=report['duration_s'],frame_count=report['video_frames']['up'])
                        except Exception as error:
                            attempt.update(accepted=False,stop_reason='exception',error=repr(error))
                            traceback.print_exc()
                            write(pending/'error.json',attempt)
                            accepted=False
                            if phase != 'planning' or not isinstance(error, ValueError):
                                raise
                        attempt.update(finished_at=now(),wall_seconds=time.monotonic()-start)
                        with (args.output/'attempts.jsonl').open('a') as handle:
                            handle.write(json.dumps(attempt,ensure_ascii=False,allow_nan=False)+'\n')
                        if accepted:
                            pending.rename(destination)
                            status['completed'][group]+=1
                            accepted_this_process+=1
                        else:
                            rejected.parent.mkdir(parents=True,exist_ok=True)
                            pending.rename(rejected)
                            status['rejected']+=1
                        status['last_attempt']=attempt
                        write(status_path,status)
                        print(json.dumps(dict(event='episode_complete',**attempt,completed=sum(status['completed'].values()))),flush=True)
                        if (stop_after_episode[0] or (args.max_new_episodes and accepted_this_process >= args.max_new_episodes)) and sum(status['completed'].values()) < status['total_requested']:
                            status.update(state='restart_required',finished_at=now(),
                                          accepted_this_process=accepted_this_process,
                                          restart_reason='requested' if stop_after_episode[0] else 'release_cpu_stage_caches')
                            return 75
                        if accepted:
                            break
                    else:
                        raise RuntimeError(f'Failed to obtain {group} episode {episode_index} within retry budget')
        status.update(state='complete',finished_at=now())
    except BaseException as error:
        status.update(state='failed',finished_at=now(),error=repr(error))
        traceback.print_exc()
        raise
    finally:
        write(status_path,status)
        if app is not None:
            # Kit fast_shutdown may terminate before Python returns, so set
            # the native application's exit code as well as main's return.
            app.app.post_quit(75 if status['state']=='restart_required' else (0 if status['state']=='complete' else 1))
            app.close()


if __name__=='__main__':
    raise SystemExit(main())
