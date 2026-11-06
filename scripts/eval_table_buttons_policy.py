#!/usr/bin/env python3
"""Evaluate learned table-button policies in physical Isaac simulation.

Run one immutable evaluation-plan cell per process. Network calls pause physical
simulation; only model joint targets move the robot. Exit 75 after the configured
number of completed episodes to release Isaac stage caches, then --resume.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import faulthandler
import gc
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# These helpers do not create a world or generate any scripted robot motion.
from pressb.runtime.recording import Video, write
from pressb.runtime.isaac import build_piper_runtime, close_piper_runtime, make_camera, set_intrinsics


def now():
    return datetime.now(timezone.utc).isoformat()


def run_episode(app, args, cfg, policy):
    episode_wall_started = time.monotonic()
    policy_description = ('remote VLA-JEPA joint6 policy with fixed gripper'
                          if policy.model.get('native_action_dim') == 6
                          else 'remote VLA-JEPA joint7 policy')
    from pressb.table_buttons.policy_evaluation import is_allowed_finger_closure
    import numpy as np
    from PIL import Image
    from pxr import UsdGeom, UsdPhysics, PhysicsSchemaTools
    import omni.physx
    from isaacsim.core.prims import SingleRigidPrim
    from isaacsim.core.utils.types import ArticulationAction
    from pressb.pick_place.scene import build_tripod
    from pressb.table_buttons.scene import build_scene
    from pressb.table_buttons.task import COLORS, TableButtonOutcome, task_instruction, project_base_points

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
    finger_closure_contacts, blocking_contacts = [], []

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
                event = dict(tick=tick[0], actors=actors, force_n=force)
                unexpected_contacts.append(event)
                if is_allowed_finger_closure(actors, args.collision_policy):
                    finger_closure_contacts.append(event)
                else:
                    blocking_contacts.append(event)

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
    report = dict(task='piper_table_buttons', mode='policy_evaluation', config=cfg, scene=scene,
                  initial_q_target=initial[indices].tolist(), initial_q_actual=robot.get_joint_positions()[indices].tolist(),
                  wrist_camera=wrist_info, physics_dt=dt, video_fps=fps, seed=args.seed,
                  policy_or_RL_evaluation=True, controller=policy_description + ', 30 Hz zero-order hold',
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

    from pressb.table_buttons.collection import measured_state
    from pressb.table_buttons.policy_evaluation import (InvalidPolicyAction, JointPolicyCommands,
        command_limits, to_policy_units, validate_execution_arrays, canonical_sha, strict_outcome)
    # The world and sensors above deliberately use the collector's exact setup.
    # Reset all contact history after its 24 physical settling ticks and renders.
    contact_events.clear()
    unexpected_contacts.clear()
    finger_closure_contacts.clear()
    blocking_contacts.clear()
    for name in COLORS:
        contact_forces[name] = 0.
    dof_properties = robot.dof_properties
    physical_limits = np.column_stack((dof_properties['lower'], dof_properties['upper']))[indices]
    lower, upper = command_limits(physical_limits)
    commands = JointPolicyCommands(lower, upper,
        measured_state(robot.get_joint_positions()[indices]), smoothing=args.smoothing,
        alpha=args.smoothing_alpha)
    videos = ({name: Video(args.output / f'{name}.mp4', size, fps)
               for name, (_, size) in camera_specs.items()} if args.record_video else {})
    checker = TableButtonOutcome(args.target, **cfg['success'])
    frame_times, states, actions, physics = [], [], [], []
    first_wrong_press = first_wrong_contact = None
    wrong_contact = False
    previous_command = initial_state.copy()
    total_intervals = round(args.timeout_s * fps)
    result = None
    maximum_depth, maximum_force = ({name: 0. for name in COLORS} for _ in range(2))
    request_index, action_index = 0, 7
    chunk = None
    model_error = None
    total_network_seconds = 0.
    network_latencies = []
    request_file = (args.output / 'policy_requests.jsonl').open('w')
    control_file = (args.output / 'control.jsonl').open('w')

    def trace(handle, value):
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        handle.flush()

    def physical_record(command):
        q_actual = robot.get_joint_positions()[indices].copy()
        travels = np.array([scene['buttons'][name]['cap_rest_world'][2] - caps[name].get_world_pose()[0][2]
                            for name in COLORS], dtype=float)
        forces = np.array([contact_forces[name] for name in COLORS], dtype=float)
        physics.append(dict(time_s=tick[0] * dt, q_actual=q_actual,
            q_target=full_command(command)[indices].copy(), travel_m=travels, finger_force_n=forces))
        return dict(zip(COLORS, travels)), dict(zip(COLORS, forces))

    physical_record(initial_state)
    stop_reason = 'timeout'
    try:
        # Reset has no hidden simulated stepping; policy service is stateless.
        policy.reset(args.episode_id)
        for frame_index in range(total_intervals + 1):
            seconds = frame_index / fps
            if frame_index:
                for _ in range(subframes):
                    world.render()
            state = measured_state(robot.get_joint_positions()[indices])
            images = {name: np.ascontiguousarray(camera.get_rgba()[..., :3])
                      for name, camera in cameras.items()}
            terminal = bool((result and result['terminated']) or blocking_contacts or wrong_contact
                            or frame_index == total_intervals)
            command = previous_command.copy()
            if not terminal:
                if action_index == 7:
                    request = dict(episode_id=args.episode_id, request_id=request_index,
                        instruction=task_instruction(args.target), state=to_policy_units(state),
                        images=images, noise_seed=args.policy_seed + request_index)
                    request_trace = dict(frame_index=frame_index, sim_time_s=seconds,
                        episode_id=args.episode_id, request_id=request_index,
                        noise_seed=request['noise_seed'], instruction=request['instruction'],
                        state_deg_cm=request['state'].tolist(), model=policy.model)
                    try:
                        chunk, response = policy.predict(**request)
                    except InvalidPolicyAction as error:
                        model_error = dict(error=repr(error), frame_index=frame_index, request_id=request_index)
                        trace(request_file, dict(**request_trace, error=repr(error), status='invalid_policy_action'))
                        terminal = True
                        stop_reason = 'invalid_policy_action'
                    else:
                        response.update(frame_index=frame_index, sim_time_s=seconds, status='complete')
                        response['latency_s'] = response['round_trip_seconds']
                        trace(request_file, response)
                        total_network_seconds += response['round_trip_seconds']
                        network_latencies.append(response['round_trip_seconds'])
                        action_index = 0
                    request_index += 1
                if not terminal:
                    command, conversion = commands.apply(chunk[action_index])
                    trace(control_file, dict(frame_index=frame_index, sim_time_s=seconds,
                        request_id=request_index - 1, action_index=action_index, **conversion))
                    action_index += 1
            frame_times.append(seconds)
            states.append(state)
            actions.append(command.copy())
            for name, video in videos.items():
                video.add(images[name])
            if terminal:
                _, stop_reason = strict_outcome(result, wrong_press=first_wrong_press,
                    wrong_contact=wrong_contact, unexpected_contacts=blocking_contacts, model_error=model_error)
                break
            previous_command = command.copy()
            controller.apply_action(ArticulationAction(joint_positions=full_command(command)))
            # No per-tick interpolation: each actual policy command is held for
            # all four physics ticks. Always inspect the full terminal interval.
            for _ in range(stride):
                tick[0] += 1
                for name in COLORS:
                    contact_forces[name] = 0.
                world.step(render=False)
                travels, forces = physical_record(command)
                wrong = [name for name in COLORS if name != args.target and forces[name] > .02]
                if wrong and first_wrong_contact is None:
                    first_wrong_contact = dict(color=wrong[0], time_s=tick[0] * dt,
                                               force_n=float(forces[wrong[0]]))
                wrong_contact = wrong_contact or bool(wrong)
                result = checker.update(tick[0] * dt, travels, forces)
                pressed_wrong = [name for name in result['press_events'] if name != args.target]
                if pressed_wrong and first_wrong_press is None:
                    first_wrong_press = dict(color=pressed_wrong[0], time_s=tick[0] * dt)
                for name in COLORS:
                    maximum_depth[name] = max(maximum_depth[name], travels[name])
                    maximum_force[name] = max(maximum_force[name], forces[name])
        frames = dict(sim_time=np.asarray(frame_times), state=np.asarray(states), action=np.asarray(actions))
        arrays = {key: np.asarray([row[key] for row in physics]) for key in physics[0]}
        validate_execution_arrays(frames, arrays, fps=fps, physics_dt=dt)
        np.savez_compressed(args.output / 'data.npz', **frames)
        np.savez_compressed(args.output / 'physics.npz', **arrays)
        success, stop_reason = strict_outcome(result, wrong_press=first_wrong_press,
            wrong_contact=wrong_contact, unexpected_contacts=blocking_contacts, model_error=model_error)
        strict_collector_success, strict_collector_stop_reason = strict_outcome(result, wrong_press=first_wrong_press,
            wrong_contact=wrong_contact, unexpected_contacts=unexpected_contacts, model_error=model_error)
        report.update(evaluation_status='complete', cell_id=args.cell_id, model_id=args.model_id,
            model=policy.model, policy_seed=args.policy_seed, episode_index=args.episode_index,
            spacing_profile=args.profile, target_color=args.target, instruction=task_instruction(args.target),
            success=success, task_success=success, strict_collector_success=strict_collector_success,
            strict_collector_stop_reason=strict_collector_stop_reason, collision_policy=args.collision_policy,
            outcome=result, stop_reason=stop_reason, timeout_s=args.timeout_s,
            duration_s=float(frame_times[-1]), sim_duration_s=float(frame_times[-1]),
            first_success_time_s=checker.first_success_time_s,
            first_wrong_press=first_wrong_press, wrong_contact=wrong_contact, first_wrong_contact=first_wrong_contact,
            unexpected_contact_count=len(unexpected_contacts), finger_contact_events=len(contact_events),
            finger_closure_contact_count=len(finger_closure_contacts), blocking_contact_count=len(blocking_contacts),
            maximum_travel_m=maximum_depth, maximum_finger_force_n=maximum_force,
            model_error=model_error, policy_requests=request_index, network_wait_seconds=total_network_seconds,
            wall_duration_s=time.monotonic() - episode_wall_started,
            inference_latency_s=dict(count=len(network_latencies), sum=total_network_seconds,
                mean=float(np.mean(network_latencies)) if network_latencies else None,
                p95=float(np.percentile(network_latencies, 95)) if network_latencies else None),
            commands=commands.summary(), joint_limits_dof_rad_m=physical_limits.tolist(),
            control_frames=len(frame_times), video_frames={name: video.count for name, video in videos.items()},
            maximum_arm_tracking_error_rad=float(np.max(np.abs(arrays['q_actual'][:, :6] - arrays['q_target'][:, :6]))),
            first_video_sample_time_s=0., camera_render_hz=fps,
            action_timing='model next-interval target held for four 120 Hz physics steps; no interpolation',
            terminal_action='repeat_previous_interval_target',
            trajectory_end='first control boundary after physical success or strict failure, or timeout',
            network_wait_advances_simulation=False, expert_actions_used=False)
        metadata = dict(schema_version='pressb.table_buttons.policy_eval.v1',
            episode_index=args.episode_index, cell_id=args.cell_id, model_id=args.model_id,
            model=policy.model, model_sha256=policy.model['model_sha256'],
            spacing_profile=args.profile, target_color=args.target,
            instruction=task_instruction(args.target), seed=args.seed, policy_seed=args.policy_seed,
            fps=fps, physics_dt=dt, timeout_s=args.timeout_s,
            camera_resolution=[640, 480], camera_names=['up', 'wrist'],
            state_units=['rad'] * 6 + ['m'], action_units=['rad'] * 6 + ['m'],
            policy_state_action_units=['degree'] * 6 + ['cm'],
            policy_normalization_scale=[180.] * 6 + [8.], gripper_representation='full_width',
            positions_xy_m={name: cfg['buttons'][name]['center_xy'] for name in COLORS},
            spacing=cfg['sampling_domain']['sampling'],
            initial_state_target_rad_m=initial_state.tolist(), initial_state_measured_rad_m=frames['state'][0].tolist(),
            physics_joint_names=[f'joint{i}' for i in range(1, 9)], button_color_order=list(COLORS),
            success=success, task_success=success, strict_collector_success=strict_collector_success,
            strict_collector_stop_reason=strict_collector_stop_reason, collision_policy=args.collision_policy,
            first_success_time_s=checker.first_success_time_s,
            duration_s=float(frame_times[-1]), sim_duration_s=float(frame_times[-1]), frame_count=len(frame_times),
            terminal_action='repeat_previous_interval_target', stop_reason=stop_reason,
            physical_outcome_thresholds=cfg['success'], non_target_contact_failure_force_n=.02,
            unexpected_contact_failure_force_n=.02, smoothing=commands.postprocessor.summary(),
            joint_limits_dof_rad_m=physical_limits.tolist(), joint_limits_lower_rad_m=lower.tolist(),
            joint_limits_upper_rad_m=upper.tolist(), config_sha256=canonical_sha(cfg),
            layout_sha256=args.layout_sha256, plan_sha256=args.plan_sha256,
            capture_alignment='RGB and measured joints at identical world time; network waits do not step physics',
            controller=policy_description + '; no expert motion',
            expert_actions_used=False, evaluation_status='complete')
        for name, rgb in images.items():
            Image.fromarray(rgb).save(args.output / f'final_{name}.jpg', quality=95)
        write(args.output / 'contacts.json', contact_events)
        write(args.output / 'unexpected_contacts.json', unexpected_contacts)
        write(args.output / 'finger_closure_contacts.json', finger_closure_contacts)
        write(args.output / 'blocking_contacts.json', blocking_contacts)
        write(args.output / 'metadata.json', metadata)
        write(args.output / 'report.json', report)
        return report
    finally:
        request_file.close()
        control_file.close()
        for video in videos.values():
            video.close()
        subscription = None
        close_piper_runtime(runtime)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--cell', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inference', help='Override this cell\'s HTTP inference endpoint')
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-new-episodes', type=int, default=20)
    parser.add_argument('--record-video-first', type=int, default=2)
    parser.add_argument('--request-timeout', type=float, default=180.)
    parser.add_argument('--smoothing', choices=('none', 'joint_ema'))
    parser.add_argument('--smoothing-alpha', type=float)
    args = parser.parse_args()
    if args.max_new_episodes < 0 or args.record_video_first < 0 or args.request_timeout <= 0:
        parser.error('Invalid episode/video quota or request timeout')
    from pressb.table_buttons.policy_evaluation import (HTTPJointPolicy, canonical_sha, file_sha, load_cell,
                                                       episode_config, summarize, collision_settings)
    plan, cell = load_cell(args.plan, args.cell)
    defaults = plan['defaults']
    collision_policy = collision_settings(defaults.get('collision_policy'))
    smoothing = dict(defaults.get('smoothing', dict(mode='none', alpha=.8)))
    if args.smoothing is not None:
        smoothing['mode'] = args.smoothing
    if args.smoothing_alpha is not None:
        smoothing['alpha'] = args.smoothing_alpha
    from pressb.action_postprocessing import JointActionPostprocessor
    JointActionPostprocessor(smoothing['mode'], smoothing['alpha'])
    timeout_s = float(defaults['timeout_s'])
    if timeout_s <= 0 or abs(timeout_s * 30 - round(timeout_s * 30)) > 1e-8:
        parser.error('Timeout must be positive and lie on a 30 Hz boundary')
    args.output = args.output.resolve()
    if args.output.exists() and not args.resume:
        parser.error('Output exists; --resume preserves this same evaluation contract')
    args.output.mkdir(parents=True, exist_ok=True)
    endpoint = args.inference or cell['inference_endpoint']
    contract = dict(schema_version='pressb.table_buttons.policy_eval_cell.v1', plan_sha256=file_sha(args.plan),
        plan_path=str(args.plan.resolve()), cell=cell, timeout_s=timeout_s, smoothing=smoothing,
        collision_policy=collision_policy, inference_endpoint=endpoint,
        record_video_first=args.record_video_first, fps=30, physics_dt=1 / 120)
    contract_path = args.output / 'plan.json'
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        parser.error('Existing cell evaluation plan differs')
    write(contract_path, contract)
    status_path = args.output / 'status.json'
    status = dict(state='starting', started_at=now(), cell_id=cell['cell_id'])
    stop_after_episode = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop_after_episode.__setitem__(0, True))
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    completed_this_process, app = 0, None
    os.environ.setdefault('OMNI_KIT_ACCEPT_EULA', 'YES')
    try:
        summary = summarize(args.output, cell)
        write(args.output / 'summary.json', summary)
        if summary['complete']:
            status.update(state='complete', **summary)
            return 0
        # No layout planner or scripted motion is called by this evaluator.
        # Plans already contain independently checked frozen scene configs.
        if plan.get('evaluation_scope', {}).get('kind') in ('joint6_single_model', 'joint6_multitask_model'):
            from pressb.table_buttons.policy_evaluation_joint6 import HTTPJoint6Policy
            policy = HTTPJoint6Policy(endpoint, cell['model_id'], timeout=args.request_timeout,
                                      model_contract=cell['model_contract'])
        else:
            policy = HTTPJointPolicy(endpoint, cell['model_id'], timeout=args.request_timeout)
        model = policy.load()
        model_path = args.output / 'model.json'
        if model_path.exists():
            previous_model = json.loads(model_path.read_text())
            if any(previous_model.get(key) != model.get(key) for key in ('model_id', 'model_sha256', 'best_step')):
                raise RuntimeError('Model identity changed while resuming an evaluation cell')
        write(model_path, model)
        from isaacsim import SimulationApp
        app = SimulationApp(dict(headless=True, active_gpu=args.gpu, physics_gpu=args.gpu,
            multi_gpu=False, renderer='RayTracedLighting', anti_aliasing=1, fast_shutdown=True,
            limit_cpu_threads=8, create_new_stage=False,
            extra_args=['--/app/asyncRendering=false', '--/rtx/ecoMode/enabled=false',
                '--/validate/p2p/enabled=false', '--/validate/iommu/enabled=false',
                '--/plugins/carb.tasking.plugin/threadCount=8', '--/plugins/omni.tbb.globalcontrol/maxThreadCount=8']))
        from isaacsim.core.api import World
        from isaacsim.core.utils.stage import create_new_stage
        for row in cell['episodes']:
            destination = args.output / 'episodes' / f"episode_{row['episode_index']:06d}"
            if destination.exists():
                metadata = json.loads((destination / 'metadata.json').read_text())
                if (metadata.get('evaluation_status') != 'complete' or metadata['model_sha256'] != model['model_sha256']
                        or metadata['plan_sha256'] != contract['plan_sha256'] or metadata['seed'] != row['seed']):
                    raise RuntimeError(f'Existing episode identity mismatch: {destination}')
                continue
            pending = args.output / 'pending' / destination.name
            if pending.exists():
                interrupted = args.output / 'interrupted' / (pending.name + '_' + str(time.time_ns()))
                interrupted.parent.mkdir(parents=True, exist_ok=True)
                pending.rename(interrupted)
            pending.mkdir(parents=True)
            cfg, layout = episode_config(plan, cell, row)
            write(pending / 'config.json', cfg)
            write(pending / 'layout.json', layout)
            write(pending / 'preflight.json', json.loads(Path(row['preflight_file']).read_text()))
            status.update(state='evaluating', current=dict(episode_index=row['episode_index'], seed=row['seed']))
            write(status_path, status)
            started = time.monotonic()
            print(json.dumps(dict(event='episode_start', cell_id=cell['cell_id'], **status['current'])), flush=True)
            if World.instance() is not None:
                old_world = World.instance()
                old_world.stop()
                World.clear_instance()
                del old_world
                gc.collect()
            create_new_stage()
            for _ in range(3):
                app.update()
            episode_args = SimpleNamespace(output=pending, seed=row['seed'], target=cell['target_color'],
                profile=cell['spacing_profile'], episode_index=row['episode_index'],
                cell_id=cell['cell_id'], model_id=cell['model_id'],
                episode_id=f"{cell['cell_id']}:seed{row['seed']}", policy_seed=row['policy_seed'],
                timeout_s=timeout_s, smoothing=smoothing['mode'], smoothing_alpha=smoothing['alpha'],
                collision_policy=collision_policy,
                record_video=row['episode_index'] < args.record_video_first,
                layout_sha256=file_sha(row['layout_file']), plan_sha256=contract['plan_sha256'])
            try:
                report = run_episode(app, episode_args, cfg, policy)
            except BaseException as error:
                write(pending / 'error.json', dict(error=repr(error), traceback=traceback.format_exc(),
                    episode_index=row['episode_index'], seed=row['seed'], category='infrastructure_error'))
                raise
            destination.parent.mkdir(parents=True, exist_ok=True)
            pending.rename(destination)
            completed_this_process += 1
            summary = summarize(args.output, cell)
            write(args.output / 'summary.json', summary)
            status.update(**summary, last_episode=dict(episode_index=row['episode_index'], seed=row['seed'],
                success=report['success'], stop_reason=report['stop_reason'], wall_seconds=time.monotonic() - started))
            write(status_path, status)
            print(json.dumps(dict(event='episode_complete', **status['last_episode'],
                                  cell_id=cell['cell_id'], completed=summary['completed'])), flush=True)
            if (stop_after_episode[0] or args.max_new_episodes and completed_this_process >= args.max_new_episodes) and not summary['complete']:
                status.update(state='restart_required', restart_reason='requested' if stop_after_episode[0] else 'release_stage_caches')
                return 75
        status.update(state='complete', finished_at=now())
        return 0
    except BaseException as error:
        status.update(state='failed', finished_at=now(), error=repr(error))
        traceback.print_exc()
        raise
    finally:
        write(status_path, status)
        if app is not None:
            app.app.post_quit(75 if status['state'] == 'restart_required' else 0 if status['state'] == 'complete' else 1)
            app.close()


if __name__ == '__main__':
    raise SystemExit(main())
