#!/usr/bin/env python3
"""Audit learned-policy execution from recorded physics, raw RPCs and RGB.

This never imports Isaac, the policy, or the controller. Report ``audit_pass``
means evidence is internally consistent; ``task_successes`` counts physical
button presses and can be zero in a passing audit.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
from itertools import product
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from audit_lerobot import IndependentFK, quaternion_matrices

ROOT = Path(__file__).resolve().parents[1]


def require(value, message):
    if not bool(value):
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def policy_identity(manifest, health):
    """Bind the recorded serving checkpoint to this run, without a fixed step."""
    expected = manifest['expected_checkpoint_sha256']
    require(isinstance(expected, str) and len(expected) == 64
            and all(c in '0123456789abcdef' for c in expected), 'Invalid expected checkpoint SHA256')
    require(health.get('status') == 'ready' and health.get('checkpoint_verified') is True,
            'Policy service did not verify a ready checkpoint')
    provenance = health.get('provenance', {})
    require(health.get('model_sha256') == provenance.get('model_sha256') == expected,
            'Policy service checkpoint hash differs from the evaluation manifest')
    arguments = manifest.get('arguments', {})
    require(arguments.get('expected_checkpoint_sha256', expected) == expected,
            'Expected checkpoint hash differs between manifest and arguments')
    step = health.get('checkpoint_step')
    require(type(step) is int and step >= 0, 'Invalid served checkpoint step')
    for source in (manifest, arguments):
        if 'expected_checkpoint_step' in source:
            require(type(source['expected_checkpoint_step']) is int
                    and source['expected_checkpoint_step'] == step,
                    'Policy service checkpoint step differs from the evaluation manifest')
    require(isinstance(provenance.get('checkpoint'), str) and bool(provenance['checkpoint']),
            'Policy service checkpoint path is missing')
    require(health.get('fps') == 30 and health.get('action_horizon') == 7
            and health.get('camera_order') == ['global', 'wrist'], 'Unsupported policy service timing/camera contract')
    return dict(checkpoint=provenance['checkpoint'], checkpoint_step=step, checkpoint_sha256=expected)


def scene_identity(manifest, collection):
    """Bind the training snapshot and the explicitly scheduled evaluation layouts."""
    cfg = manifest['config']
    require(cfg == collection['config'], 'Evaluation scene configuration differs from training collection')
    explicit_layouts = 'panel_layouts' in manifest
    if not explicit_layouts:
        require(not cfg.get('panel_randomization', {}).get('enabled', False)
                and all(cfg.get(f'panel_offset_{axis}_m', 0.) == 0. for axis in ('x', 'y')),
                'An unshifted legacy scene or an explicit panel layout schedule is required')
    layouts = panel_layout_plan(manifest)
    require(manifest['scene_sha256'] == collection['scene_sha256'] == sha(manifest['arguments']['snapshot']),
            'Evaluation did not use the training snapshot')
    if manifest['arguments'].get('asset_bundle') is not None:
        require(isinstance(manifest.get('scene_relocation'), dict),
                'Relocated evaluation is missing its runtime scene evidence')
    if 'scene_relocation' in manifest:
        import sys
        if str(ROOT / 'src') not in sys.path:
            sys.path.insert(0, str(ROOT / 'src'))
        from pressb.scene_portability import verify_runtime_snapshot
        relocation = manifest['scene_relocation']
        verified = verify_runtime_snapshot(Path(manifest['arguments']['snapshot']), Path(relocation['report_path']))
        require(verified == relocation, 'Runtime scene relocation differs from its recorded evidence')
    identity = collection['identity']
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    require(manifest['collection_fingerprint'] == collection['collection_fingerprint'] == fingerprint,
            'Evaluation collection fingerprint differs from the original collection identity')
    require(identity['config'] == cfg and identity['scene_sha256'] == manifest['scene_sha256'],
            'Training collection identity does not match its scene/configuration')
    require(manifest['recorded_actions_used'] is False and manifest['target_planner_used'] is False,
            'Evaluation declares expert action/goal-planner inputs')
    return dict(scene_sha256=manifest['scene_sha256'], collection_fingerprint=fingerprint,
                panel_offset_x_m=cfg.get('panel_offset_x_m', 0.), panel_offset_y_m=cfg.get('panel_offset_y_m', 0.),
                panel_randomization_enabled=cfg.get('panel_randomization', {}).get('enabled', False),
                panel_layout_mode=manifest.get('panel_layout_mode', 'fixed'), panel_layouts=layouts)


PANEL_FIELDS = ('panel_layout_index', 'panel_layout_name', 'panel_offset_x_m', 'panel_offset_y_m')


def panel_layout_plan(manifest):
    """Rebuild the layout table without importing the evaluation scheduler."""
    cfg = manifest['config']
    explicit = 'panel_layouts' in manifest
    mode = manifest.get('panel_layout_mode', 'fixed')
    require(mode in ('fixed', 'center_corners'), 'Unknown panel layout mode')
    if mode == 'fixed':
        points = [('fixed', cfg.get('panel_offset_x_m', 0.), cfg.get('panel_offset_y_m', 0.))]
    else:
        settings = cfg['panel_randomization']
        require(settings.get('enabled') is True, 'Center/corner layouts require recorded randomization bounds')
        xmin, xmax, ymin, ymax = [settings[f'{bound}_offset_{axis}_m'] for axis in 'xy' for bound in ('min', 'max')]
        require(np.isfinite([xmin, xmax, ymin, ymax]).all() and xmin < xmax and ymin < ymax,
                'Invalid center/corner panel rectangle')
        points = [('center', (xmin + xmax) / 2, (ymin + ymax) / 2),
                  ('xmin_ymin', xmin, ymin), ('xmin_ymax', xmin, ymax),
                  ('xmax_ymin', xmax, ymin), ('xmax_ymax', xmax, ymax)]
    layouts = [dict(panel_layout_index=i, panel_layout_name=name,
                    panel_offset_x_m=float(x), panel_offset_y_m=float(y)) for i, (name, x, y) in enumerate(points)]
    require(all(np.isfinite([row['panel_offset_x_m'], row['panel_offset_y_m']]).all() for row in layouts),
            'Nonfinite panel offsets')
    if explicit:
        require(manifest['panel_layouts'] == layouts, 'Recorded panel layout table differs from configured positions')
        require(manifest['arguments'].get('panel_layouts', mode) == mode, 'Panel layout CLI differs from manifest')
    else:
        require(mode == 'fixed', 'Shifted evaluation requires an explicit panel layout table')
    return layouts


def episode_schedule(manifest):
    """Check every requested floor/repeat/layout, including subsets used for smoke tests."""
    layouts = panel_layout_plan(manifest)
    args = manifest['arguments']
    floors, repeats = args['floors'], args['episodes_per_floor']
    require(isinstance(floors, list) and floors and len(floors) == len(set(floors))
            and all(type(f) is int and 24 <= f <= 35 for f in floors), 'Invalid requested floors')
    require(type(repeats) is int and repeats > 0, 'Invalid repeats per floor/layout')
    schedule = []
    for repeat in range(repeats):
        for layout in layouts:
            for floor in floors:
                schedule.append(dict(episode_id=len(schedule), floor=floor, repeat=repeat,
                    seed_episode_index=(repeat * len(layouts) + layout['panel_layout_index']) * 12 + floor - 24,
                    **layout))
    if 'panel_layouts' in manifest:
        require(manifest.get('episode_schedule') == schedule, 'Episode schedule differs from floor/repeat/layout contract')
    return schedule


def panel_episode_geometry(metadata, collection=None):
    """Verify measured panel geometry and return the target face in world/base frames.

    Layout offsets move only the panel. The robot base and per-clone origin
    keep their original meaning; a button body's rest centre is face +3 mm X.
    """
    cfg = metadata['config']
    require(type(metadata['floor']) is int and 24 <= metadata['floor'] <= 35, 'Invalid panel target floor')
    base = np.asarray(metadata['robot_base_world_m'], dtype=float)
    require(base.shape == (3,) and np.isfinite(base).all(), 'Invalid robot base position')
    nominal_base = np.array([cfg['robot_base_x'], cfg['robot_base_y'], cfg['table_height']])
    env = base - nominal_base
    explicit = any(key in metadata for key in (*PANEL_FIELDS, 'panel_layout'))
    if explicit:
        require(all(key in metadata for key in (*PANEL_FIELDS, 'panel_layout', 'env_offset_m')),
                'Incomplete episode panel geometry evidence')
        require(type(metadata['panel_layout_index']) is int and metadata['panel_layout_index'] >= 0
                and isinstance(metadata['panel_layout_name'], str) and bool(metadata['panel_layout_name']),
                'Invalid episode panel identity')
        panel = {key: metadata[key] for key in PANEL_FIELDS}
    else:
        require(not cfg.get('panel_randomization', {}).get('enabled', False)
                and all(cfg.get(f'panel_offset_{axis}_m', 0.) == 0. for axis in 'xy'),
                'Missing measured layout for shifted/randomized episode')
        panel = dict(panel_layout_index=0, panel_layout_name='fixed', panel_offset_x_m=0., panel_offset_y_m=0.)
    xy = np.array([panel['panel_offset_x_m'], panel['panel_offset_y_m']], dtype=float)
    require(np.isfinite(xy).all(), 'Invalid episode panel offset')
    if 'env_offset_m' in metadata:
        origin = np.asarray(metadata['env_offset_m'], dtype=float)
        require(origin.shape == (3,) and np.isfinite(origin).all() and np.allclose(origin, env, atol=1e-8, rtol=0),
                'Robot base was moved with the panel or clone origin differs')
    faces = np.array([[cfg['button_face_x'] + xy[0],
                       cfg['button_column_y'] * (1 if floor < 30 else -1) + xy[1],
                       cfg['button_bottom_z'] + ((floor - 24) % 6) * cfg['button_pitch_z']]
                      for floor in range(24, 36)]) + env
    proof = dict(**panel, measured_layout_verified=explicit, environment_origin_m=env.tolist(),
                 target_face_world_m=faces[metadata['floor'] - 24].tolist(),
                 target_face_base_m=(faces[metadata['floor'] - 24] - base).tolist())
    if not explicit:
        return proof
    layout = metadata['panel_layout']
    require(all(layout[f'panel_offset_{axis}_m'] == panel[f'panel_offset_{axis}_m'] for axis in 'xy'),
            'Measured layout offset differs from episode offset')
    require(layout['fixed_rig_unchanged'] is True, 'Panel move changed the fixed robot/camera rig')
    rests = layout['button_rest_positions_world_m']
    require(set(rests) == {str(f) for f in range(24, 36)}, 'Missing measured button rest positions')
    measured = np.asarray([rests[str(f)] for f in range(24, 36)], dtype=float)
    require(measured.shape == (12, 3) and np.isfinite(measured).all(), 'Invalid measured button positions')
    errors = np.linalg.norm(measured - faces - [0.003, 0., 0.], axis=1)
    require(errors.max() <= .0002 and abs(errors.max() - layout['max_button_rest_error_m']) < 1e-7,
            'Button rests do not match the actual episode panel position')
    require(0 < layout['rest_tolerance_m'] <= .0002 and errors.max() <= layout['rest_tolerance_m']
            and 0 <= layout['max_spring_anchor_error_m'] <= layout['rest_tolerance_m'], 'Invalid panel rest/spring tolerance evidence')
    bounds = np.asarray(layout['panel_world_bounds_m'], dtype=float)
    expected_bounds = np.array([[cfg['button_face_x'] + xy[0] - .000225, xy[1] - .1, cfg['button_bottom_z'] - .07],
                               [cfg['button_face_x'] + xy[0] + .049, xy[1] + .1, cfg['button_bottom_z'] + 5 * cfg['button_pitch_z'] + .07]]) + env
    require(bounds.shape == (2, 3) and np.isfinite(bounds).all()
            and np.allclose(bounds, expected_bounds, atol=.0002, rtol=0), 'Measured whole-panel bounds do not match the offset')
    wall = np.asarray(layout['back_wall_world_bounds_m'], dtype=float)
    require(wall.shape == (2, 3) and np.isfinite(wall).all() and np.all(wall[0] < wall[1]), 'Invalid back wall bounds')
    wall_error = abs(wall[0, 0] - (cfg.get('wall_x', cfg['button_face_x'] + .06) + xy[0] + env[0]))
    require(wall_error <= 1e-6 and abs(wall_error - layout['back_wall_alignment_error_m']) < 1e-9,
            'Back wall did not follow panel X')
    camera = layout['global_camera']
    pixels, depths = np.asarray(camera['corners_px']), np.asarray(camera['corner_depths_m'])
    margin = cfg.get('panel_randomization', {}).get('camera_margin_px', 24.)
    require(np.isfinite(margin) and 0 <= margin < 240, 'Invalid configured camera margin')
    require(camera['resolution'] == [640, 480] and camera['coverage_kind'] == 'frustum_only'
            and camera['all_inside'] is True and camera['margin_px'] == margin,
            'Wrong panel camera coverage declaration')
    require(pixels.shape == (8, 2) and depths.shape == (8,) and np.isfinite(pixels).all()
            and np.isfinite(depths).all() and (depths > .005).all(), 'Invalid panel camera corners')
    actual_margin = float(np.minimum(pixels, np.array([639, 479]) - pixels).min())
    require(actual_margin >= margin, 'Actual panel corners violate camera margin')
    require(collection is not None, 'Measured panel projection requires the bound collection calibration')
    calibration = collection['global']
    transform = np.asarray(calibration['sensor']['world_optical_transform'], dtype=float)
    intrinsics = np.asarray(calibration['K'], dtype=float)
    require(transform.shape == (4, 4) and intrinsics.shape == (3, 3)
            and np.isfinite(transform).all() and np.isfinite(intrinsics).all(), 'Invalid fixed camera calibration')
    require(np.allclose(transform[:3, 3], cfg['global_camera_eye'], atol=1e-8, rtol=0), 'Fixed camera eye differs from configuration')
    forward = np.asarray(cfg['global_camera_target'], dtype=float) - transform[:3, 3]
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.]); right /= np.linalg.norm(right)
    require(np.allclose(transform[:3, :3], np.column_stack((right, np.cross(right, forward), -forward)), atol=1e-8, rtol=0),
            'Fixed camera orientation differs from configuration')
    corners = np.array(list(product(*zip(bounds[0], bounds[1]))))
    local = (corners - env - transform[:3, 3]) @ transform[:3, :3]
    expected_depths = -local[:, 2]
    uv = np.column_stack((intrinsics[0, 0] * local[:, 0] / expected_depths + intrinsics[0, 2],
                          intrinsics[1, 2] - intrinsics[1, 1] * local[:, 1] / expected_depths))
    require(np.allclose(depths, expected_depths, atol=1e-6, rtol=0)
            and np.allclose(pixels, uv, atol=.001, rtol=0), 'Panel projection differs from independent camera geometry')
    proof.update(max_button_rest_error_m=float(errors.max()), minimum_camera_margin_px=actual_margin,
                 max_camera_projection_error_px=float(np.abs(uv - pixels).max()), camera_coverage_kind='frustum_only')
    return proof


def outcome_statistics(episodes):
    """Aggregate repeats without replacing earlier episodes for the same floor."""
    rows = list(episodes)
    identities = [row.get('episode_id', row.get('episode')) for row in rows]
    require(None not in identities and len(set(identities)) == len(rows), 'Duplicate or missing episode identity in outcome summary')
    def counts(selected):
        successes = sum(bool(row['success']) for row in selected)
        return dict(episodes=len(selected), successes=successes,
                    success_rate=successes / len(selected) if selected else None,
                    task_successes=sum(bool(row.get('task_success', row['success'])) for row in selected),
                    termination_counts=dict(Counter(row['termination'] for row in selected)))
    names = list(dict.fromkeys(row.get('panel_layout_name', 'fixed') for row in rows))
    floors = sorted({row['floor'] for row in rows})
    return dict(overall=counts(rows),
                by_floor={str(f): counts([row for row in rows if row['floor'] == f]) for f in floors},
                by_position={name: counts([row for row in rows if row.get('panel_layout_name', 'fixed') == name]) for name in names},
                by_floor_position={str(f): {name: counts([row for row in rows if row['floor'] == f
                                      and row.get('panel_layout_name', 'fixed') == name]) for name in names} for f in floors})


def rot6(values):
    values = np.asarray(values, dtype=float)
    require(values.shape[-1] == 6 and np.isfinite(values).all(), 'Invalid raw rotation6D')
    first = values[..., :3]
    norm = np.linalg.norm(first, axis=-1, keepdims=True)
    require(np.all(norm > 1e-10), 'Degenerate first rotation row')
    first = first / norm
    second = values[..., 3:] - np.sum(first * values[..., 3:], axis=-1, keepdims=True) * first
    norm = np.linalg.norm(second, axis=-1, keepdims=True)
    require(np.all(norm > 1e-10), 'Degenerate second rotation row')
    second = second / norm
    return np.stack((first, second, np.cross(first, second)), axis=-2)


def rotations(pose8):
    q = np.asarray(pose8, dtype=float)[..., 3:7]
    flat = q.reshape(-1, 4)
    norm = np.linalg.norm(flat, axis=1)
    require(np.isfinite(flat).all() and np.allclose(norm, 1, atol=1e-5, rtol=0), 'Invalid wxyz quaternion')
    return quaternion_matrices(flat / norm[:, None]).reshape(q.shape[:-1] + (3, 3))


def rotation_error(left, right):
    return np.arccos(np.clip((np.sum(left * right, axis=(-2, -1)) - 1) / 2, -1, 1))


def pose_consistency(pose8, pose9):
    p8, p9 = np.asarray(pose8, dtype=float), np.asarray(pose9, dtype=float)
    require(p8.shape[-1] == 8 and p9.shape == p8.shape[:-1] + (9,), 'Wrong pose shapes')
    require(np.isfinite(p8).all() and np.isfinite(p9).all(), 'Nonfinite raw model poses')
    require(np.array_equal(p8[..., :3], p9[..., :3]), 'Conversion changed raw model XYZ')
    require(np.allclose(p8[..., 7], .008, atol=1e-8, rtol=0), 'Learned/changed gripper opening')
    error = float(np.max(rotation_error(rotations(p8), rot6(p9[..., 3:]))))
    require(error < 2e-5, 'Quaternion is not the raw six-dimensional ROW rotation')
    return error


def lamp_evidence(travel, force, press, release):
    state = np.zeros(12, dtype=np.uint8)
    result, events = [], []
    for index, (displacement, contact) in enumerate(zip(travel, force)):
        for button in range(12):
            if not state[button] and displacement[button] >= press and contact[button] > .02:
                state[button] = 1
                events.append(dict(type='pressed', floor=button + 24, physics_index=index))
            elif state[button] and displacement[button] <= release:
                state[button] = 0
                events.append(dict(type='released', floor=button + 24, physics_index=index))
        result.append(state.copy())
    return np.asarray(result), events


def video_evidence(path, expected_frames):
    import av

    unique, previous, changes, count = set(), None, [], 0
    with av.open(str(path)) as container:
        require(len(container.streams.video) == 1, 'Expected one RGB video stream')
        stream = container.streams.video[0]
        require(stream.average_rate == Fraction(30, 1), 'Video rate is not 30 Hz')
        stream.codec_context.thread_count = 1
        for i, frame in enumerate(container.decode(video=0)):
            require(frame.pts is not None and frame.time_base is not None, 'Missing RGB timestamp')
            require(abs(float(frame.pts * frame.time_base) - i / 30) < 1e-7, 'Nonuniform encoded RGB timestamps')
            rgb = frame.to_ndarray(format='rgb24')
            require(rgb.shape == (480, 640, 3), 'Video is not RGB 640 x 480')
            unique.add(hashlib.sha256(rgb.tobytes()).digest())
            if previous is not None:
                changes.append(float(np.abs(rgb.astype(np.int16) - previous).mean()))
            previous, count = rgb.astype(np.int16), i + 1
    require(count == expected_frames, f'Video frame count {count} != {expected_frames}')
    return dict(frames=count, unique_frames=len(unique), sha256=sha(path),
                mean_adjacent_rgb_change=float(np.mean(changes)) if changes else 0.,
                encoded_fps=30, terminal_frame_physical_time_from_frames_npz=True)


def field(value, *names):
    for name in names:
        if name in value:
            return value[name]
    raise ValueError(f'Missing required field: {names}')


def image_path(directory, path):
    path = Path(path)
    return path if path.is_absolute() else directory / path


def smoothing_settings(record):
    """Validate the declared filter; historical runs mean window one."""
    declared = record.get('motion_smoothing')
    require(declared is None or isinstance(declared, dict), 'Invalid smoothing metadata')
    window = 1 if declared is None else declared.get('window')
    require(type(window) is int and window in (1, 3, 5, 7, 9, 11), 'Invalid smoothing window')
    expected = dict(kind='linear_joint_interpolation_then_causal_mean', window=window, physics_hz=120,
                    nominal_delay_s=(window - 1) / 240, initial_history='repeat_initial_command',
                    reset='per_episode', history='cross_chunk')
    if declared is not None:
        require(isinstance(declared, dict) and declared == expected, 'Unsupported smoothing metadata')
    return expected


def command_smoothing_evidence(metadata, physics):
    """Reconstruct from a padded offline input array, independently of the live filter."""
    settings = smoothing_settings(metadata)
    executed = np.asarray(physics['q_command'])
    require(executed.ndim == 2 and executed.shape[1] == 6 and len(executed) > 0
            and executed.dtype.kind == 'f' and np.isfinite(executed).all(), 'Invalid executed commands')
    explicit = 'motion_smoothing' in metadata
    require(('q_command_unsmoothed' in physics) == explicit,
            'Unsmoothed commands and smoothing declaration must be recorded together')
    raw = np.asarray(physics['q_command_unsmoothed']) if explicit else executed
    require(raw.shape == executed.shape and raw.dtype.kind == 'f' and np.isfinite(raw).all(),
            'Invalid unsmoothed command evidence')
    window = settings['window']
    padded = np.concatenate((np.repeat(raw[:1].astype(float), window - 1, axis=0), raw.astype(float)))
    reconstructed = np.array([padded[i:i + window].mean(axis=0) for i in range(len(raw))])
    # A distinct floating-point reduction order may round at the neighboring
    # float32 value. Permit only one output ULP, never a control-step lag.
    rounding = np.maximum(np.abs(np.spacing(executed)), 1e-12)
    error = np.abs(reconstructed.astype(executed.dtype) - executed)
    require(np.all(error <= rounding), 'Executed commands differ from the declared causal moving mean')
    require(np.array_equal(raw[0].astype(executed.dtype), executed[0]), 'Filter did not start at the initial command')
    return raw, settings, float(error.max())


def executed_action_evidence(action, raw_pose, commands, fk):
    """Use the actual last substep, including a partial terminal interval."""
    start, end = int(action['physics_start_index']), int(action['physics_end_index'])
    for key, index in (('q_executed_before', start), ('q_executed_endpoint', end)):
        require(np.array_equal(np.asarray(action[key]), commands[index]), f'{key} differs from the sent command')
    transform = fk.batch(commands[end:end + 1])[0]
    position = float(np.linalg.norm(transform[:3, 3] - raw_pose[:3]))
    angle = float(rotation_error(transform[:3, :3], rotations(raw_pose)))
    require(abs(position - action['executed_position_residual_m']) < 1e-7,
            'Incorrect executed endpoint position residual')
    require(abs(angle - action['executed_rotation_residual_rad']) < 2e-6,
            'Incorrect executed endpoint rotation residual')
    return position, angle


def request_seed_evidence(metadata, requests, base_seed=None, layout_count=1):
    if 'seed_episode_index' not in metadata:
        return None
    index, repeat, floor = metadata['seed_episode_index'], metadata['repeat'], metadata['floor']
    layout_index = metadata.get('panel_layout_index', 0)
    require(type(layout_count) is int and layout_count > 0 and type(layout_index) is int
            and 0 <= layout_index < layout_count, 'Invalid inference layout index/count')
    require(type(index) is int and type(repeat) is int and repeat >= 0
            and index == (repeat * layout_count + layout_index) * 12 + floor - 24,
            'Inference seed index does not preserve the floor/repeat/layout stream')
    if not requests:
        return base_seed
    if base_seed is None:
        base_seed = requests[0]['seed'] - index * 10000 - requests[0]['chunk_index']
    require(type(base_seed) is int, 'Invalid inference base seed')
    require(all(request['seed'] == base_seed + index * 10000 + request['chunk_index'] for request in requests),
            'Request inference seed differs from the declared floor/repeat stream')
    return base_seed


def audit_episode(directory, base_seed=None, collection=None, layout_count=1):
    metadata = json.loads((directory / 'metadata.json').read_text())
    cfg = metadata['config']
    floor = int(metadata['floor'])
    require(24 <= floor <= 35 and metadata['task'] == f'Press {floor} floor.', 'Wrong task text')
    panel_geometry = panel_episode_geometry(metadata, collection)
    require(metadata['fps'] == 30 and metadata['physics_hz'] == 120, 'Wrong control/physics frequency')
    require(abs(cfg['physics_dt'] - 1 / 120) < 1e-12, 'Wrong physics dt')
    fk = IndependentFK(ROOT / cfg['robot_urdf'])
    xml = ET.parse(ROOT / cfg['robot_urdf']).getroot()
    velocities = np.asarray([float(xml.find(f"joint[@name='joint{i}']/limit").get('velocity')) for i in range(1, 7)])
    with np.load(directory / 'physics.npz', allow_pickle=False) as data:
        physics = {key: data[key].copy() for key in data.files}
    with np.load(directory / 'frames.npz', allow_pickle=False) as data:
        frames = {key: data[key].copy() for key in data.files}
    n = len(physics['q_actual'])
    require(n > 0, 'No physical state evidence')
    shapes = dict(q_actual=(n, 6), q_command=(n, 6), gripper_actual=(n, 2),
                  button_travel=(n, 12), contact_force=(n, 12), lights=(n, 12),
                  state=(n, 8), tip_position_world=(n, 3), sim_time=(n,), physics_index=(n,))
    for key, shape in shapes.items():
        require(key in physics and physics[key].shape == shape and np.isfinite(physics[key]).all(), f'Invalid physics {key}')
    require(np.array_equal(physics['physics_index'], np.arange(n)), 'Nonconsecutive physics indices')
    require(np.allclose(physics['sim_time'], np.arange(n) / 120, atol=1e-9, rtol=0), 'Nonuniform physical time')
    expected_lights, events = lamp_evidence(physics['button_travel'], physics['contact_force'], cfg['press_threshold'], cfg['release_threshold'])
    require(np.array_equal(expected_lights, physics['lights']), 'Lamp states disagree with independent physical contact/travel')
    recorded_events = [{key: event[key] for key in ('type', 'floor', 'physics_index')} for event in metadata['events']]
    require(recorded_events == events, 'Reported button events differ from physical evidence')
    actual = fk.batch(physics['q_actual'])
    state = physics['state']
    tcp_error = float(np.linalg.norm(actual[:, :3, 3] - state[:, :3], axis=1).max())
    tcp_rotation_error = float(rotation_error(actual[:, :3, :3], rotations(state)).max())
    require(tcp_error < 2e-6 and tcp_rotation_error < 2e-5, 'Measured TCP is not actual-joint FK in base_link')
    require(np.allclose(state[:, 7], physics['gripper_actual'][:, 0] - physics['gripper_actual'][:, 1], atol=1e-8, rtol=0), 'State gripper is not measured')
    base = np.asarray(metadata['robot_base_world_m'], dtype=float)
    tip = actual[:, :3, 3] + actual[:, :3, :3] @ np.array([0., 0., .24 - .1358]) + base
    tip_error = float(np.linalg.norm(tip - physics['tip_position_world'], axis=1).max())
    require(tip_error < .005, 'Physical tool tip disagrees with independent world FK')
    button_center = np.asarray(panel_geometry['target_face_world_m'])
    tip_distances = np.linalg.norm(physics['tip_position_world'] - button_center, axis=1)
    require(np.allclose(tip_distances, physics['target_tip_distance_m'], atol=1e-9, rtol=0), 'Incorrect diagnostic target-tip distances')
    require(abs(tip_distances.min() - metadata['min_target_tip_distance_m']) < 1e-9, 'Incorrect reported minimum target distance')
    joint_tracking = float(np.max(np.abs(physics['q_actual'] - physics['q_command'])))
    gripper_error = float(np.max(np.abs(physics['gripper_actual'] - np.asarray(cfg['gripper_joint_positions_m']))))
    qcmd = physics['q_command']
    qunsmoothed, smoothing, reconstruction_error = command_smoothing_evidence(metadata, physics)
    require(np.all(qcmd >= fk.lower - 1e-7) and np.all(qcmd <= fk.upper + 1e-7), 'Command exceeded joint limits')
    require(np.all(qunsmoothed >= fk.lower - 1e-7) and np.all(qunsmoothed <= fk.upper + 1e-7), 'Unsmoothed command exceeded joint limits')
    # Isaac's articulation command buffers are float32. Account only for that
    # representation error; model targets and IK endpoints are checked below.
    command_rounding = 2 * np.finfo(qcmd.dtype).eps * max(1., float(np.abs(qcmd).max()))
    require(n == 1 or np.all(np.abs(np.diff(qcmd, axis=0)) * 120 <= velocities + command_rounding * 120), 'Command exceeded URDF joint speed')
    require(n == 1 or np.all(np.abs(np.diff(qunsmoothed, axis=0)) * 120 <= velocities + command_rounding * 120), 'Unsmoothed command exceeded URDF joint speed')
    initial_error = float(np.max(np.abs(physics['q_actual'][0] - cfg['home_q'])))
    require(initial_error < cfg['home_tolerance_rad'], 'Episode did not begin at folded home')
    require(np.max(np.abs(metadata['initial_joint_velocity_rad_s'])) < .02, 'Initial joints not settled')
    indices = np.asarray(frames['physics_index'], dtype=int)
    require(len(indices) > 0 and indices[0] == 0 and indices[-1] == n - 1, 'Missing initial/terminal RGB frame')
    expected_indices = list(range(0, n, 4))
    if expected_indices[-1] != n - 1:
        expected_indices.append(n - 1)
    require(indices.tolist() == expected_indices, 'Camera captures are not 30 Hz plus optional terminal partial frame')
    for key in ('state', 'q_actual', 'lights', 'sim_time'):
        require(np.array_equal(frames[key], physics[key][indices]), f'RGB-associated {key} not sampled from same physics state')

    requests = read_jsonl(directory / 'requests.jsonl')
    actions = read_jsonl(directory / 'actions.jsonl')
    checked_base_seed = request_seed_evidence(metadata, requests, base_seed, layout_count)
    by_chunk, request_errors, invalid_requests = {}, [], []
    for ordinal, request in enumerate(requests):
        chunk = int(request['chunk_index'])
        require(chunk == ordinal and chunk not in by_chunk, 'Request chunks not unique/consecutive')
        index = int(request['observation_physics_index'])
        require(index in indices and index < n, 'Model consumed an uncaptured physics state')
        observed = np.asarray(field(request, 'observed_state', 'observedstate', 'state'), dtype=float)
        require(np.array_equal(observed, state[index]), 'Model consumed a changed/unmeasured state')
        require(np.array_equal(np.asarray(request['q_actual']), physics['q_actual'][index]), 'Request measured joints disagree')
        require(request['task'] == metadata['task'] and type(request['seed']) is int, 'Request task/seed missing')
        reply = field(request, 'raw_reply', 'reply', 'response')
        require(reply.get('camera_order') == ['global', 'wrist'], 'Inference service camera order changed')
        require(reply.get('seed') == request['seed'] and reply.get('xyz_units') == 'metres', 'Inference response seed or position units differ')
        require(request.get('frozen_sim_time') is True, 'Inference was not recorded as physically paused')
        p8, p9, invalid = None, None, False
        try:
            p8, p9 = np.asarray(reply['actions_pose8']), np.asarray(reply['actions_pose9'])
            require(p8.shape == (7, 8) and p9.shape == (7, 9), 'Wrong model horizon')
            request_errors.append(pose_consistency(p8, p9))
            require(np.all(np.linalg.norm(p8[:, :3], axis=1) <= 1.5), 'Target outside 1.5 m controller bound')
            for key, expected in (('pose_frame', 'base_link'), ('pose_link', 'gripper_tcp'), ('gripper_is_learned', False)):
                require(key not in reply or reply[key] == expected, f'Wrong policy response {key}')
        except (KeyError, ValueError, TypeError) as error:
            require(metadata['termination'] == 'invalid_policy_action' and ordinal == len(requests) - 1,
                    f'Unclassified invalid policy response: {error}')
            invalid = True
            invalid_requests.append(dict(chunk=chunk, error=str(error)))
        if 'images' in request and isinstance(request['images'].get('global'), dict):
            image_hashes = {view: value['rgb_sha256'] for view, value in request['images'].items()}
            image_files = {view: value['path'] for view, value in request['images'].items()}
        else:
            image_hashes = field(request, 'image_sha256', 'image_rgb_sha256', 'image_sha256_rgb')
            image_files = field(request, 'image_paths', 'observation_images', 'images')
        for view in ('global', 'wrist'):
            path = image_path(directory, image_files[view])
            require(path.resolve().parent == (directory / 'observations').resolve(), 'Model RGB is not a saved live episode observation')
            with Image.open(path) as im:
                rgb = np.asarray(im.convert('RGB'))
            require(rgb.shape == (480, 640, 3), 'Policy input image wrong dimensions')
            require(hashlib.sha256(rgb.tobytes()).hexdigest() == image_hashes[view], f'{view} policy RGB hash mismatch')
            if isinstance(request.get('images', {}).get(view), dict):
                require(sha(path) == request['images'][view]['sha256'], f'{view} input PNG file hash mismatch')
        by_chunk[chunk] = dict(index=index, pose8=p8, pose9=p9, rows=[], invalid=invalid)

    covered, max_projection_m, max_projection_rad, projected_count, partial_actions = 0, 0., 0., 0, 0
    max_executed_m, max_executed_rad = 0., 0.
    for action in actions:
        chunk, row = int(action['chunk_index']), int(action['chunk_action_index'])
        require(chunk in by_chunk and 0 <= row < 7, 'Executed action absent from raw model reply')
        request = by_chunk[chunk]
        require(not request['invalid'], 'Executed an independently invalid model response')
        require(row == len(request['rows']), 'Skipped/repeated/reordered model chunk row')
        request['rows'].append(row)
        raw8 = np.asarray(field(action, 'raw_pose8', 'rawpose8', 'pose8', 'action_pose8'), dtype=float)
        raw9 = np.asarray(field(action, 'raw_pose9', 'rawpose9', 'pose9', 'action_pose9'), dtype=float)
        require(np.array_equal(raw8, request['pose8'][row]) and np.array_equal(raw9, request['pose9'][row]), 'Execution changed raw model target')
        pose_consistency(raw8, raw9)
        start, end = int(action['physics_start_index']), int(action['physics_end_index'])
        require(start == covered and start == request['index'] + row * 4, 'Target was executed at wrong time')
        require(1 <= end - start <= 4 and end < n, 'Invalid executed interval')
        require(end - start == 4 or end == n - 1, 'Partial nonterminal target interval')
        partial_actions += int(end - start < 4)
        before, target = np.asarray(action['q_before']), np.asarray(action['q_target'])
        require(before.shape == (6,) and target.shape == (6,) and np.isfinite(target).all(), 'Invalid commanded joint target')
        require(np.all(target >= fk.lower - 1e-7) and np.all(target <= fk.upper + 1e-7), 'Projected endpoint exceeds joint limits')
        require(np.all(np.abs(target - before) * 30 <= velocities + 1e-5), 'Projected endpoint exceeds 30 Hz joint speed')
        # Endpoint interpolation can cancel sub-picoradian joint values near
        # zero before the float32 assignment; this is not target projection.
        require(np.allclose(before.astype(qunsmoothed.dtype), qunsmoothed[start], atol=1e-10, rtol=0), 'Action does not start at preceding unsmoothed commanded joints')
        substeps = np.arange(1, end - start + 1)[:, None] / 4
        expected = before + substeps * (target - before)
        require(np.array_equal(expected.astype(qunsmoothed.dtype), qunsmoothed[start + 1:end + 1]), 'Unsmoothed command is not documented 120 Hz interpolation')
        command_tcp = fk.batch(target[None])[0]
        position_error = float(np.linalg.norm(command_tcp[:3, 3] - raw8[:3]))
        angle_error = float(rotation_error(command_tcp[:3, :3], rotations(raw8)))
        diagnostic = action if 'command_position_residual_m' in action else field(action, 'controller', 'controller_diagnostics', 'diagnostics')
        require(abs(position_error - diagnostic['command_position_residual_m']) < 1e-7, 'Incorrect controller position projection diagnostic')
        require(abs(angle_error - diagnostic['command_rotation_residual_rad']) < 2e-6, 'Incorrect controller rotation projection diagnostic')
        unrestricted = np.asarray(diagnostic['unrestricted_joint_target'], dtype=float)
        unrestricted_tcp = fk.batch(unrestricted[None])[0]
        unrestricted_position = float(np.linalg.norm(unrestricted_tcp[:3, 3] - raw8[:3]))
        unrestricted_angle = float(rotation_error(unrestricted_tcp[:3, :3], rotations(raw8)))
        require(abs(unrestricted_position - diagnostic['unrestricted_position_residual_m']) < 1e-7, 'Incorrect unrestricted IK position diagnostic')
        require(abs(unrestricted_angle - diagnostic['unrestricted_rotation_residual_rad']) < 2e-6, 'Incorrect unrestricted IK rotation diagnostic')
        if 'motion_smoothing' in metadata:
            executed_m, executed_rad = executed_action_evidence(action, raw8, qcmd, fk)
        else:
            sent_tcp = fk.batch(qcmd[end:end + 1])[0]
            executed_m = float(np.linalg.norm(sent_tcp[:3, 3] - raw8[:3]))
            executed_rad = float(rotation_error(sent_tcp[:3, :3], rotations(raw8)))
        max_executed_m, max_executed_rad = max(max_executed_m, executed_m), max(max_executed_rad, executed_rad)
        max_projection_m, max_projection_rad = max(max_projection_m, position_error), max(max_projection_rad, angle_error)
        projected_count += int(position_error > 1e-5 or angle_error > 1e-4)
        covered = end
    require(covered == n - 1, 'Physics has no corresponding raw policy target')
    for chunk, request in by_chunk.items():
        require(bool(request['rows']) or (request['invalid'] and request['index'] == n - 1),
                'Unused model request without explicit invalid-action classification')
        if chunk < len(requests) - 1:
            require(len(request['rows']) == metadata.get('action_chunk', 7), 'Replanned before configured chunk length')
    presses = [event for event in events if event['type'] == 'pressed']
    wrong = [event for event in presses if event['floor'] != floor]
    target = [event for event in presses if event['floor'] == floor]
    collisions = metadata['unexpected_collisions']
    task_success = bool(target) and not wrong
    safe_success = task_success and not collisions
    if presses:
        require(presses[0]['physics_index'] == n - 1, 'Continued motion after first physical button press')
    require(bool(metadata['task_success']) == task_success, 'Reported task success disagrees with physical evidence')
    require(bool(metadata['success']) == safe_success, 'Reported safe success disagrees with physical evidence')
    videos = {view: video_evidence(directory / f'{view}.mp4', len(indices)) for view in ('global', 'wrist')}
    return dict(episode=directory.name, episode_id=metadata['episode_id'], repeat=metadata['repeat'],
                floor=floor, termination=metadata['termination'], audit_pass=True, task_success=task_success,
                **{key: panel_geometry[key] for key in PANEL_FIELDS}, panel_geometry=panel_geometry,
                success=safe_success, physics_steps=n - 1, simulation_seconds=(n - 1) / 120,
                requests=len(requests), executed_actions=len(actions), discarded_chunk_rows=len(requests) * 7 - len(actions),
                fully_executed_actions=len(actions) - partial_actions, partially_executed_actions=partial_actions,
                invalid_requests=invalid_requests,
                camera_frames=len(indices), events=events, unexpected_collisions=len(collisions),
                initial_home_error_rad=initial_error, measured_tcp_fk_position_error_m=tcp_error,
                max_joint_tracking_error_rad=joint_tracking, max_closed_gripper_error_m=gripper_error,
                min_target_tip_distance_m=float(tip_distances.min()), final_target_tip_distance_m=float(tip_distances[-1]),
                measured_tcp_fk_rotation_error_rad=tcp_rotation_error, world_tip_fk_error_m=tip_error,
                max_rotation_conversion_error_rad=max(request_errors, default=0),
                projected_actions=projected_count, max_command_projection_m=max_projection_m,
                max_command_projection_deg=float(np.degrees(max_projection_rad)),
                motion_smoothing=smoothing, max_smoothing_reconstruction_error_rad=reconstruction_error,
                seed_episode_index=metadata.get('seed_episode_index'), checked_inference_base_seed=checked_base_seed,
                max_executed_endpoint_residual_m=max_executed_m,
                max_executed_endpoint_residual_deg=float(np.degrees(max_executed_rad)),
                target_max_travel_m=float(physics['button_travel'][:, floor - 24].max()),
                target_max_contact_force_n=float(physics['contact_force'][:, floor - 24].max()),
                videos=videos, files_sha256={name: sha(directory / name) for name in
                    ('metadata.json', 'physics.npz', 'frames.npz', 'requests.jsonl', 'actions.jsonl')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    run = args.run.resolve()
    report_path = (args.report or run / 'independent_audit.json').resolve()
    require(not report_path.exists(), f'Report already exists: {report_path}')
    result = dict(audit_pass=False, run=str(run), audited_at=datetime.now(timezone.utc).isoformat(),
                  audit_source_sha256=sha(Path(__file__)), independent_fk_source_sha256=sha(ROOT / 'scripts/audit_lerobot.py'),
                  allow_partial=args.allow_partial, episodes=[], errors=[],
                  limitations=['Collision classification uses recorded PhysX contact monitor events.',
                               'RGB input hashes/video timestamps are checked; image semantics are not automatically classified.',
                               'Controller feasibility projection is measured and reported, not counted as exact target execution.',
                               'Simulator is paused during remote inference; this does not measure real-time deployment.'])
    manifest = json.loads((run / 'eval_manifest.json').read_text())
    dataset = Path(manifest['arguments']['dataset'])
    collection = json.loads((dataset / 'meta/collection_metadata.json').read_text())
    scene = scene_identity(manifest, collection)
    schedule = episode_schedule(manifest)
    result['expected_episodes'] = len(schedule)
    identity = policy_identity(manifest, json.loads((run / 'policy_service.json').read_text()))
    result['motion_smoothing'] = smoothing_settings(manifest)
    result['provenance'] = dict(eval_manifest_sha256=sha(run / 'eval_manifest.json'),
                               **scene, **identity, policy_service_sha256=sha(run / 'policy_service.json'),
                               dataset_collection_sha256=sha(dataset / 'meta/collection_metadata.json'))
    for path in sorted(run.glob('episode_*/metadata.json')):
        try:
            metadata = json.loads(path.read_text())
            require(metadata['config'] == collection['config'], 'Episode scene differs from training collection')
            require(metadata.get('motion_smoothing') == manifest.get('motion_smoothing'), 'Episode smoothing differs from manifest')
            index = metadata['episode_id']
            require(type(index) is int and 0 <= index < len(schedule)
                    and path.parent.name == f'episode_{index:06d}', 'Unexpected episode directory/identity')
            keys = ['floor', 'repeat']
            if 'panel_layouts' in manifest:
                keys += [*PANEL_FIELDS, 'seed_episode_index']
            require(all(metadata.get(key) == schedule[index][key] for key in keys),
                    'Episode differs from the independently reconstructed schedule')
            result['episodes'].append(audit_episode(path.parent, base_seed=manifest['arguments'].get('seed'),
                                                    collection=collection, layout_count=len(scene['panel_layouts'])))
        except Exception as error:
            result['errors'].append(dict(episode=path.parent.name, type=type(error).__name__, error=str(error)))
    floors = [episode['floor'] for episode in result['episodes']]
    if not floors:
        result['errors'].append(dict(error='No completely audited episodes'))
    if sorted(row['episode_id'] for row in result['episodes']) != list(range(len(schedule))):
        result['errors'].append(dict(error='Incomplete requested floor/repeat/layout schedule'))
    if not args.allow_partial and sorted(set(floors)) != list(range(24, 36)):
        result['errors'].append(dict(error='Expected all 12 floor tasks'))
    result.update(audit_pass=not result['errors'], audited_episodes=len(floors),
                  outcomes=outcome_statistics(result['episodes']),
                  task_successes=sum(item['task_success'] for item in result['episodes']),
                  collision_free_successes=sum(item['success'] for item in result['episodes']))
    report_path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: result[key] for key in ('audit_pass', 'audited_episodes', 'task_successes', 'collision_free_successes', 'errors')}, indent=2))
    raise SystemExit(0 if result['audit_pass'] else 1)


if __name__ == '__main__':
    main()
