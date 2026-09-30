#!/usr/bin/env python3
"""Independently audit physical execution of recorded absolute TCP actions.

Run with the LeRobot environment; this script never starts Isaac or a GPU.
--input is the extracted action directory; --output is the actual replay root.
The replay's success flags and reported maximum errors are not acceptance evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

from audit_lerobot import IndependentFK, quaternion_matrices

ROOT = Path(__file__).resolve().parents[1]
TCP_OFFSET = .1358
TIP_OFFSET = .24
FULL_CYCLE_END = 'full_cycle_return_home'
PRESS_ONLY_END = 'first_sampled_target_light_on'
THRESHOLDS = dict(home_rad=.025, joint_tracking_rad=.15, gripper_m=.00025,
                  fk_tip_m=.005, tcp_position_m=.005, tcp_rotation_rad=.01,
                  action_position_m=.005, action_rotation_rad=.01, contact_force_n=.02,
                  initial_joint_velocity_rad_s=.02, minimum_unique_video_fraction=.5,
                  minimum_mean_adjacent_rgb_change=.05)


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def recompute_lights(travel, force, press, release):
    """Infer momentary lamp state solely from recorded travel and contact force."""
    state = np.zeros(12, dtype=np.uint8)
    result, events = [], []
    for index, (distance, contact) in enumerate(zip(travel, force)):
        for button in range(12):
            if not state[button] and distance[button] >= press and contact[button] > .02:
                state[button] = 1
                events.append(('pressed', button + 24, index))
            elif state[button] and distance[button] <= release:
                state[button] = 0
                events.append(('released', button + 24, index))
        result.append(state.copy())
    return np.asarray(result), events


def rotation_errors(left, right):
    trace = np.einsum('nij,nij->n', left, right)
    return np.arccos(np.clip((trace - 1.) / 2., -1., 1.))


def normalized_rotations(wxyz):
    """Float32 unit quaternions need normalization before SO(3) angle errors."""
    values = np.asarray(wxyz, dtype=float)
    norms = np.linalg.norm(values, axis=1)
    require(np.isfinite(values).all() and np.allclose(norms, 1., atol=1e-5, rtol=0),
            'Invalid pose quaternion')
    return quaternion_matrices(values / norms[:, None])


def audit_physics(physics, metadata, source, source_metadata, source_config, fk):
    errors = []

    def check(condition, message):
        if not bool(condition):
            errors.append(message)

    cfg = metadata['config']
    episode_end = source_metadata.get('episode_end', FULL_CYCLE_END)
    require(episode_end in (FULL_CYCLE_END, PRESS_ONLY_END), 'Unsupported source episode_end')
    check(metadata.get('episode_end', FULL_CYCLE_END) == episode_end,
          'Replay episode termination differs from the recorded source')
    policy_source = source_metadata.get('episode_end_source_export_manifest_sha256')
    if policy_source is not None or episode_end == PRESS_ONLY_END:
        require(isinstance(policy_source, str) and len(policy_source) == 64
                and all(c in '0123456789abcdef' for c in policy_source),
                'Source termination lacks an export manifest SHA256')
        check(metadata.get('episode_end_source_export_manifest_sha256') == policy_source,
              'Replay termination export provenance differs from source')
    check(np.isclose(metadata.get('panel_offset_y_m', 0.), source_metadata.get('panel_offset_y_m', 0.),
                     rtol=0, atol=1e-12), 'Replay panel offset differs from recorded episode')
    check(np.isclose(metadata.get('panel_offset_x_m', 0.), source_metadata.get('panel_offset_x_m', 0.),
                     rtol=0, atol=1e-12), 'Replay panel depth offset differs from recorded episode')
    require(cfg == source_config, 'Replay configuration differs from recorded collection')
    for key in ('home_q', 'robot_urdf', 'robot_base_x', 'robot_base_y', 'table_height',
                'physics_dt', 'press_threshold', 'release_threshold', 'tip_offset',
                'gripper_joint_positions_m', 'home_tolerance_rad'):
        require(cfg[key] == source_config[key], f'Replay changed source configuration: {key}')
    require(abs(cfg['tip_offset'] - TIP_OFFSET) < 1e-12, 'Unexpected probe offset')
    require(metadata['fps'] == 30 and metadata['physics_hz'] == 120, 'Expected 30 Hz / 120 Hz replay')
    require(abs(cfg['physics_dt'] - 1/120) < 1e-12, 'Unexpected physics time step')
    floor = metadata['floor']
    require(metadata['source_episode_id'] == source_metadata['source_episode_id'], 'Source episode identity mismatch')
    require(floor == source_metadata['floor'] and metadata['task'] == f'Press {floor} floor.'
            and metadata['task'] == source_metadata['task'], 'Floor/task mismatch')
    count = len(physics['q_actual'])
    require(count > 1, 'Insufficient physical evidence')
    shapes = {'q_actual': (count, 6), 'q_command': (count, 6),
              'gripper_actual': (count, 2), 'gripper_command': (count, 2),
              'button_travel': (count, 12), 'contact_force': (count, 12), 'lights': (count, 12),
              'tcp_state': (count, 8), 'tip_position_world': (count, 3), 'time': (count,)}
    for key, shape in shapes.items():
        require(key in physics and physics[key].shape == shape, f'Wrong/missing physics shape: {key}')
        require(np.isfinite(physics[key]).all(), f'Nonfinite physics: {key}')
    indices = np.asarray(metadata['capture_physics_indices'])
    require(np.issubdtype(indices.dtype, np.integer), 'Capture indices must be integers')
    require(np.array_equal(indices, np.arange(0, count, 4)), 'Missing or irregular camera physics indices')
    require(indices[-1] == count-1, 'Final episode observation was not captured')
    require(len(indices) == metadata['num_frames'] == len(source['action']) == len(source['state']),
            'Replay/source/camera sample counts differ')
    require(source['action'].shape == source['state'].shape == (len(indices), 8), 'Wrong source pose dimensions')
    require(np.isfinite(source['action']).all() and np.isfinite(source['state']).all(), 'Nonfinite source poses')
    require(np.allclose(physics['time'], np.arange(count)/120, atol=1e-9, rtol=0), 'Invalid physics timestamps')
    require(np.allclose(source['sim_time'], np.arange(len(indices))/30, atol=1e-8, rtol=0), 'Invalid source timestamps')
    check(np.isin(physics['lights'], [0, 1]).all(), 'Lamp labels are not binary')
    inferred, events = recompute_lights(physics['button_travel'], physics['contact_force'],
                                       cfg['press_threshold'], cfg['release_threshold'])
    check(np.array_equal(inferred, physics['lights']), 'Recorded lamps disagree with contact/travel hysteresis')
    if episode_end == PRESS_ONLY_END:
        check([event[:2] for event in events] == [('pressed', floor)],
              'Target must press exactly once and remain pressed, with no other floor pressed')
        expected_terminal_lights = np.zeros(12, dtype=np.uint8)
        expected_terminal_lights[floor - 24] = 1
        check(np.array_equal(inferred[-1], expected_terminal_lights),
              'Press-only terminal observation must contain the target light only')
    else:
        check([event[:2] for event in events] == [('pressed', floor), ('released', floor)],
              'Target must press and release exactly once, with no other floor pressed')
    recorded_events = [(event['type'], event['floor'], event['physics_index']) for event in metadata['events']]
    check(events == recorded_events, 'Recorded event list differs from independent physical events')
    check(not metadata['unexpected_collisions'], 'Unexpected contact/collision monitor reported an event')
    if episode_end == PRESS_ONLY_END:
        check(np.max(np.abs(physics['button_travel'][0])) <= cfg['release_threshold'], 'Buttons not released at start')
        check(not inferred[0].any(), 'Lamp already active at start')
    else:
        check(np.max(np.abs(physics['button_travel'][[0, -1]])) <= cfg['release_threshold'], 'Buttons not released at start/end')
        check(not inferred[[0, -1]].any(), 'Lamp still active at start/end')
    home = np.asarray(cfg['home_q'])
    home_start = float(np.max(np.abs(physics['q_actual'][0] - home)))
    home_end = float(np.max(np.abs(physics['q_actual'][-1] - home)))
    tracking = float(np.max(np.abs(physics['q_actual'] - physics['q_command'])))
    grip_tracking = float(np.max(np.abs(physics['gripper_actual'] - physics['gripper_command'])))
    grip_closed = float(np.max(np.abs(physics['gripper_actual'] - np.asarray(cfg['gripper_joint_positions_m']))))
    if episode_end == PRESS_ONLY_END:
        check(home_start < min(THRESHOLDS['home_rad'], cfg['home_tolerance_rad']), 'Initial joint pose is not home')
    else:
        check(max(home_start, home_end) < min(THRESHOLDS['home_rad'], cfg['home_tolerance_rad']), 'Initial/final joint pose is not home')
    initial_velocity = np.asarray(metadata['initial_joint_velocity_rad_s'])
    require(initial_velocity.shape == (6,) and np.isfinite(initial_velocity).all(), 'Missing/invalid settled initial joint velocities')
    check(np.max(np.abs(initial_velocity)) < THRESHOLDS['initial_joint_velocity_rad_s'], 'Initial joints are not physically settled')
    check(tracking < THRESHOLDS['joint_tracking_rad'], 'Excessive physical joint tracking error')
    check(max(grip_tracking, grip_closed) < THRESHOLDS['gripper_m'], 'Gripper tracking/closed-opening error')
    check(np.all(physics['q_command'] >= fk.lower-1e-5) and np.all(physics['q_command'] <= fk.upper+1e-5),
          'Commands exceed official URDF joint limits')
    actual = fk.batch(physics['q_actual'])
    base = np.asarray(metadata['robot_base_world_m'])
    expected_base = np.asarray([cfg['robot_base_x'], cfg['robot_base_y'], cfg['table_height']]) + metadata['env_offset_m']
    require(base.shape == (3,) and np.allclose(base, expected_base, atol=1e-8, rtol=0), 'Wrong robot base/world offset')
    expected_tip = actual[:, :3, 3] + actual[:, :3, :3] @ np.array([0., 0., TIP_OFFSET-TCP_OFFSET]) + base
    tip_error = float(np.linalg.norm(expected_tip - physics['tip_position_world'], axis=1).max())
    check(tip_error <= THRESHOLDS['fk_tip_m'], 'Actual world probe disagrees with independent URDF FK')
    tcp = physics['tcp_state']
    require(np.allclose(np.linalg.norm(tcp[:, 3:7], axis=1), 1., atol=1e-5, rtol=0), 'Nonunit actual TCP quaternion')
    actual_rotation = normalized_rotations(tcp[:, 3:7])
    tcp_position_error = float(np.linalg.norm(actual[:, :3, 3] - tcp[:, :3], axis=1).max())
    tcp_rotation_error = float(rotation_errors(actual[:, :3, :3], actual_rotation).max())
    check(tcp_position_error <= THRESHOLDS['tcp_position_m'] and tcp_rotation_error <= THRESHOLDS['tcp_rotation_rad'],
          'Recorded TCP state disagrees with independent actual-joint FK')
    check(np.max(np.abs(tcp[:, 7] - (physics['gripper_actual'][:, 0]-physics['gripper_actual'][:, 1]))) < 1e-6,
          'TCP state gripper width differs from actual finger positions')
    # Replay convention: source action[k] is reached at the next 30 Hz knot k+1.
    commanded = fk.batch(physics['q_command'][indices[1:]])
    targets = source['action'][:-1].astype(float)
    require(np.allclose(np.linalg.norm(targets[:, 3:7], axis=1), 1., atol=1e-5, rtol=0), 'Nonunit source action quaternion')
    target_rotations = normalized_rotations(targets[:, 3:7])
    action_position_error = float(np.linalg.norm(commanded[:, :3, 3] - targets[:, :3], axis=1).max())
    action_rotation_error = float(rotation_errors(commanded[:, :3, :3], target_rotations).max())
    commanded_width = np.diff(-physics['gripper_command'][indices[1:]], axis=1)[:, 0]
    action_gripper_error = float(np.max(np.abs(commanded_width-targets[:, 7])))
    check(action_position_error <= THRESHOLDS['action_position_m'], 'Commands do not reach recorded absolute action positions')
    check(action_rotation_error <= THRESHOLDS['action_rotation_rad'], 'Commands do not reach recorded full action orientations')
    check(action_gripper_error < THRESHOLDS['gripper_m'], 'Commands do not reproduce recorded action gripper opening')
    observed = actual[indices]
    reference = source['state'].astype(float)
    state_position_delta = np.linalg.norm(observed[:, :3, 3] - reference[:, :3], axis=1)
    state_rotation_delta = rotation_errors(observed[:, :3, :3], normalized_rotations(reference[:, 3:7]))
    visible_lit = np.flatnonzero(inferred[indices, floor-24])
    check(len(visible_lit) > 0, 'No captured frame contains the target press')
    return dict(success=not errors, errors=errors, physics_steps=count, camera_frames=len(indices),
                episode_end=episode_end,
                independently_recomputed_events=[dict(type=t, floor=f, physics_index=i, time_s=i/120) for t,f,i in events],
                initial_home_error_rad=home_start, final_home_error_rad=home_end,
                initial_joint_velocity_max_rad_s=float(np.max(np.abs(initial_velocity))),
                first_step_position_difference_velocity_max_rad_s=float(np.max(np.abs(np.diff(physics['q_actual'][:2], axis=0))) * 120),
                max_joint_tracking_error_rad=tracking, max_gripper_tracking_error_m=grip_tracking,
                max_closed_gripper_error_m=grip_closed, max_independent_world_tip_error_m=tip_error,
                max_tcp_position_error_m=tcp_position_error, max_tcp_rotation_error_rad=tcp_rotation_error,
                action_knot_position_error_m=action_position_error, action_knot_rotation_error_rad=action_rotation_error,
                action_gripper_error_m=action_gripper_error,
                maximum_target_button_travel_m=float(physics['button_travel'][:,floor-24].max()),
                maximum_target_contact_force_n=float(physics['contact_force'][:,floor-24].max()),
                source_state_comparison=dict(acceptance_criterion=False,
                    max_position_delta_m=float(state_position_delta.max()), rms_position_delta_m=float(np.sqrt(np.mean(state_position_delta**2))),
                    max_rotation_delta_rad=float(state_rotation_delta.max())),
                sampled_lit_frame_indices=visible_lit.tolist())


def audit_video(path, expected_frames, times, keep):
    images, hashes = {}, set()
    adjacent_change, previous = [], None
    count = 0
    with av.open(str(path)) as container:
        require(len(container.streams.video) == 1, 'Expected exactly one video stream')
        stream = container.streams.video[0]
        require(stream.average_rate == Fraction(30, 1), f'Wrong video fps: {stream.average_rate}')
        stream.codec_context.thread_count = 1
        for index, frame in enumerate(container.decode(video=0)):
            require(index < expected_frames, 'Too many video frames')
            require(frame.pts is not None and frame.time_base is not None, 'Missing video PTS')
            require(abs(float(frame.pts*frame.time_base)-times[index]) <= 1e-8, 'Video PTS differs from physical sample time')
            rgb = frame.to_ndarray(format='rgb24')
            require(rgb.shape == (480, 640, 3), f'Wrong RGB dimensions: {rgb.shape}')
            hashes.add(hashlib.sha256(rgb.tobytes()).digest())
            if previous is not None:
                adjacent_change.append(float(np.mean(np.abs(rgb.astype(np.int16)-previous))))
            previous = rgb.astype(np.int16)
            if index in keep:
                images[index] = Image.fromarray(rgb)
            count += 1
    require(count == expected_frames, f'Decoded {count} frames, expected {expected_frames}')
    mean_change = float(np.mean(adjacent_change)) if adjacent_change else 0.
    require(len(hashes) > expected_frames*THRESHOLDS['minimum_unique_video_fraction']
            and mean_change > THRESHOLDS['minimum_mean_adjacent_rgb_change'],
            'Replay video is static or lacks sufficient motion evidence')
    return dict(success=True, decoded_frames=count, fps=30, pts_match_physics=True,
                unique_rgb_hashes=len(hashes), mean_adjacent_rgb_change=mean_change, dynamic_video=True,
                sha256=sha(path)), images


def save_overview(directory, eid, floor, phases, images):
    sheet = Image.new('RGB', (1280, 536), (24, 26, 30))
    draw = ImageDraw.Draw(sheet)
    for row, view in enumerate(('wrist', 'global')):
        for col, (phase, index) in enumerate(phases.items()):
            img = images[view][index]
            img.save(directory / f'episode_{eid:06d}_floor{floor}_{view}_{phase}.png')
            sheet.paste(img.resize((320, 240)), (col*320, row*268+28))
            draw.text((col*320+5, row*268+5), f'F{floor} {view} {phase} frame {index}', fill='white')
    path = directory / f'episode_{eid:06d}_floor{floor}_overview.png'
    sheet.save(path)
    return str(path)


def overview_phases(physical):
    lit = physical['sampled_lit_frame_indices']
    last = physical['camera_frames'] - 1
    if physical['episode_end'] == PRESS_ONLY_END:
        require(lit and lit[-1] == last, 'No target light in terminal press-only capture')
        return dict(initial=0, approach=max(0, lit[0]-1), pressed=lit[0], terminal=last)
    require(lit and lit[-1] < last, 'No captured press/release pair')
    return dict(initial=0, pressed=(lit[0]+lit[-1])//2, released=lit[-1]+1, home=last)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--allow-partial', action='store_true', help='Diagnostic only; not full 12-floor acceptance')
    parser.add_argument('--episode-ids', type=int, nargs='+', help='Select committed source IDs for a partial diagnostic')
    args = parser.parse_args()
    source_root, replay_root = args.input.resolve(), args.output.resolve()
    report_path = (args.report or replay_root / 'audit.json').resolve()
    if report_path.exists():
        parser.error('Report exists; choose a new --report to preserve evidence')
    images_root = report_path.parent / f'{report_path.stem}_images'
    images_root.mkdir(parents=True, exist_ok=True)
    manifest_path = source_root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    source_episode_end = manifest['semantics'].get('episode_end', FULL_CYCLE_END)
    config = manifest['collection_config']
    urdf = ROOT / config['robot_urdf']
    fk = IndependentFK(urdf)
    result = dict(success=False, replay_root=str(replay_root), input_root=str(source_root),
                  audited_at=datetime.now(timezone.utc).isoformat(), thresholds=THRESHOLDS,
                  input_manifest_sha256=sha(manifest_path), urdf_sha256=sha(urdf),
                  audit_source_sha256=sha(Path(__file__)), allow_partial=args.allow_partial,
                  independent_fk_source_sha256=sha(ROOT / 'scripts/audit_lerobot.py'),
                  limitations=['Collision acceptance uses the recorded collider-monitor unexpected_collisions list.',
                               'Deviation from the original recorded state is diagnostic, not a success criterion.',
                               'Video integrity/PTS and saved contact sheets are checked; this is not an automated geometric pixel audit.'],
                  episodes=[], errors=[])
    try:
        run_path = replay_root / 'replay_manifest.json'
        run = json.loads(run_path.read_text())
        require(run['input_manifest_sha256'] == sha(manifest_path), 'Run used a different extracted input manifest')
        require(run['prepared_manifest'] == manifest, 'Run embedded source identity differs from extracted input')
        require(run['config'] == config, 'Run configuration differs from source')
        require(run['export_manifest_sha256'] == manifest['dataset']['export_manifest_sha256']
                == sha(Path(run['dataset']) / 'meta/export_manifest.json'), 'Original LeRobot export manifest changed')
        exported = json.loads((Path(run['dataset']) / 'meta/export_manifest.json').read_text())
        require(source_episode_end in (FULL_CYCLE_END, PRESS_ONLY_END), 'Unsupported recorded episode_end')
        require(manifest['semantics'] == exported['semantics'], 'Prepared semantics differ from original LeRobot export')
        require(source_episode_end == manifest.get('episode_end', FULL_CYCLE_END)
                == run.get('episode_end', FULL_CYCLE_END), 'Replay termination differs from original source')
        if ('episode_end_source_export_manifest_sha256' in run or source_episode_end == PRESS_ONLY_END):
            require(run.get('episode_end_source_export_manifest_sha256') == run['export_manifest_sha256'],
                    'Replay termination is not bound to original export SHA256')
        require(run['recorded_joint_targets_used_for_control'] is False and run['recorded_light_events_used'] is False,
                'Run declares use of recorded joints/lights for control')
        checked_sources = {}
        for name, expected_sha in run['source_code_sha256'].items():
            actual_sha = sha(ROOT / name)
            require(actual_sha == expected_sha, f'Frozen replay source changed: {name}')
            checked_sources[name] = actual_sha
        require(bool(checked_sources), 'Run lacks source code hashes')
        require(sha(Path(run['scene'])) == run['scene_sha256'], 'Frozen scene changed')
        result['replay_provenance'] = dict(success=True, manifest_sha256=sha(run_path),
                                         source_code_sha256=checked_sources, scene_sha256=run['scene_sha256'])
    except Exception as exc:
        result['errors'].append(f'Replay provenance: {type(exc).__name__}: {exc}')
    entries = {entry['source_episode_id']: entry for entry in manifest['episodes']}
    if args.episode_ids is not None:
        if not args.allow_partial or len(set(args.episode_ids)) != len(args.episode_ids) or any(i not in entries for i in args.episode_ids):
            parser.error('--episode-ids requires --allow-partial and distinct IDs present in the input manifest')
        entries = {i: entries[i] for i in args.episode_ids}
    result['requested_episode_ids'] = args.episode_ids
    pressed_images = {}
    for eid, entry in sorted(entries.items()):
        directory = replay_root / f'episode_{eid:06d}'
        if args.allow_partial and not (directory / 'metadata.json').exists():
            continue
        row = dict(source_episode_id=eid, success=False, errors=[])
        try:
            source_json = source_root / f'episode_{eid:06d}.json'
            source_metadata = json.loads(source_json.read_text())
            require(sha(source_json) == entry['manifest_sha256'], 'Extracted episode JSON hash mismatch')
            require(source_metadata.get('episode_end', FULL_CYCLE_END) == source_episode_end,
                    'Extracted episode termination differs from original source')
            if ('episode_end_source_export_manifest_sha256' in source_metadata or source_episode_end == PRESS_ONLY_END):
                require(source_metadata.get('episode_end_source_export_manifest_sha256')
                        == manifest['dataset']['export_manifest_sha256'],
                        'Extracted episode termination is not bound to source export')
            source_npz = source_root / source_metadata['npz_file']
            require(source_npz.resolve().parent == source_root, 'Source NPZ escapes extraction root')
            require(sha(source_npz) == source_metadata['npz_sha256'], 'Extracted action/state hash mismatch')
            with np.load(source_npz, allow_pickle=False) as data:
                source = dict(data)
            metadata_path, physics_path = directory / 'metadata.json', directory / 'physics.npz'
            metadata = json.loads(metadata_path.read_text())
            require(metadata['input_sha256'] == sha(source_npz), 'Replay used different action arrays')
            with np.load(physics_path, allow_pickle=False) as data:
                physics = dict(data)
            control_path = replay_root / 'controls' / f'episode_{eid:06d}.npz'
            require(sha(control_path) == metadata['control_sha256'], 'Saved replay control plan hash differs')
            with np.load(control_path, allow_pickle=False) as data:
                require(np.allclose(data['joint_positions'], physics['q_command'], atol=1e-9, rtol=0),
                        'Executed commands differ from the saved IK/interpolation plan')
                require(np.allclose(data['times_s'], physics['time'], atol=1e-9, rtol=0), 'Executed control clock differs')
                require(np.allclose(data['gripper_widths'], np.diff(-physics['gripper_command'], axis=1)[:,0], atol=1e-9, rtol=0),
                        'Executed gripper commands differ from the saved control plan')
            row.update(floor=metadata['floor'], task=metadata['task'],
                       source_episode_sha256=sha(source_json), source_arrays_sha256=sha(source_npz),
                       physics_sha256=sha(physics_path), metadata_sha256=sha(metadata_path), control_sha256=sha(control_path))
            physical = audit_physics(physics, metadata, source, source_metadata, config, fk)
            row.update(physics=physical, errors=list(physical['errors']), videos={})
            frames_path = directory / 'frames.npz'
            with np.load(frames_path, allow_pickle=False) as data:
                frames = dict(data)
            indices = np.asarray(metadata['capture_physics_indices'])
            require(np.array_equal(frames['physics_index'], indices), 'Frame/physics indices differ')
            for name, expected in (('state', physics['tcp_state'][indices]), ('action', source['action']),
                                   ('q_actual', physics['q_actual'][indices]), ('lights', physics['lights'][indices]),
                                   ('sim_time', physics['time'][indices])):
                require(np.allclose(frames[name], expected, atol=1e-7, rtol=0), f'Captured {name} differs from physics/source')
            row['frames_sha256'] = sha(frames_path)
            for name, evidence in metadata['source_files'].items():
                require(sha(directory / name) == evidence['sha256'], f'Replay evidence hash mismatch: {name}')
            phases = overview_phases(physical)
            samples = {}
            for view in ('wrist', 'global'):
                row['videos'][view], samples[view] = audit_video(directory / f'{view}.mp4',
                    physical['camera_frames'], physics['time'][metadata['capture_physics_indices']], set(phases.values()))
            row['overview'] = save_overview(images_root, eid, metadata['floor'], phases, samples)
            row['image_phases'] = phases
            pressed_images[metadata['floor']] = {v: samples[v][phases['pressed']] for v in samples}
            row['success'] = not row['errors']
        except Exception as exc:
            row['errors'].append(f'{type(exc).__name__}: {exc}')
        result['episodes'].append(row)
        if not row['success']:
            result['errors'].append(f'episode_{eid:06d}: '+ '; '.join(row['errors']))
        print(json.dumps({k: row.get(k) for k in ('source_episode_id', 'floor', 'success', 'errors')}), flush=True)
    counts = Counter(row.get('floor') for row in result['episodes'])
    result['floor_counts'] = {str(f): counts[f] for f in range(24, 36)}
    if not result['episodes'] or (not args.allow_partial and (len(result['episodes']) != 12 or any(counts[f] != 1 for f in range(24,36)))):
        result['errors'].append('Expected exactly one independently replayed episode for each of 12 floors')
    if pressed_images:
        gallery = Image.new('RGB', (2048, 672), (24, 26, 30))
        draw = ImageDraw.Draw(gallery)
        for index, floor in enumerate(range(24, 36)):
            if floor not in pressed_images:
                continue
            x, y = (index % 4)*512, (index//4)*224
            draw.text((x+5,y+8), f'Floor {floor}: actual replay | wrist / global', fill='white')
            for col, view in enumerate(('wrist','global')):
                gallery.paste(pressed_images[floor][view].resize((256,192)), (x+col*256,y+32))
        gallery_path = images_root / 'all_floors_pressed.png'
        gallery.save(gallery_path)
        result['pressed_overview'] = str(gallery_path)
    result['total_episodes'] = len(result['episodes'])
    result['decoded_rgb_frames'] = sum(v['decoded_frames'] for row in result['episodes'] for v in row.get('videos',{}).values())
    result['success'] = not result['errors']
    with report_path.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k != 'episodes'}, indent=2))
    return 0 if result['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
