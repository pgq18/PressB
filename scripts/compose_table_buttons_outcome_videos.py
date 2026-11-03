#!/usr/bin/env python3
"""Compare one recorded success and failure from the same table-button policy.

These are different episodes and may have different layouts and random seeds.
Their identity is shown in each column. Original 640x480 up/wrist frames remain
uncropped and unresized, on their recorded 30 Hz simulation clock. Terminal
frames are held with explicit labels. No simulation or action processing runs.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont

import compose_rl_comparison_videos as common
import compose_table_buttons_policy_videos as policy


FPS, WIDTH, HEIGHT = 30, 1280, 1220
TOP, WRIST_Y, FOOTER_Y = 132, 644, 1124
BG, FG, MUTED = policy.BG, policy.FG, policy.MUTED
OUTCOMES = ('success', 'failure')


def shared_config(config):
    """Keep scene settings while removing only sampled-layout-derived values."""
    result = deepcopy(config)
    for button in result['buttons'].values():
        button.pop('center_xy')
    result.pop('press_approaches', None)
    domain = result.get('sampling_domain', {})
    domain.pop('layout_clearance', None)
    for field in ('pairwise_distances_m', 'attempts', 'layout_restarts', 'rejected_by_reason'):
        domain.get('sampling', {}).pop(field, None)
    return result


def validate_outcomes(cases):
    """Validate outcome examples without changing the paired-policy validator."""
    left, right = [case['report'] for case in cases]
    if left['task_success'] is not True or right['task_success'] is not False:
        raise ValueError('Columns require a task success followed by a task failure')
    if cases[0]['directory'] == cases[1]['directory']:
        raise ValueError('Outcome examples must be different episodes')
    for field in ('task', 'cell_id', 'model_id', 'spacing_profile', 'target_color',
                  'instruction', 'timeout_s', 'collision_policy', 'physics_dt',
                  'initial_q_target', 'controller', 'action_timing'):
        if left[field] != right[field]:
            raise ValueError(f'Outcome comparison changes {field}')
    for field in ('model_sha256', 'checkpoint_step', 'checkpoint_path',
                  'native_action_dim', 'fixed_gripper_full_width_m'):
        if left['model'][field] != right['model'][field]:
            raise ValueError(f'Outcome comparison changes model {field}')
    for field in ('camera_names', 'camera_resolution', 'plan_sha256',
                  'physical_outcome_thresholds', 'non_target_contact_failure_force_n',
                  'unexpected_contact_failure_force_n'):
        if cases[0]['metadata'][field] != cases[1]['metadata'][field]:
            raise ValueError(f'Outcome comparison changes metadata {field}')
    if shared_config(left['config']) != shared_config(right['config']):
        raise ValueError('Outcome comparison changes scene settings beyond sampled layouts')
    for case in cases:
        report, metadata = case['report'], case['metadata']
        for field in ('episode_index', 'model_id', 'seed', 'policy_seed', 'spacing_profile',
                      'target_color', 'instruction', 'task_success', 'strict_collector_success',
                      'duration_s', 'stop_reason', 'first_success_time_s'):
            if report[field] != metadata[field]:
                raise ValueError(f'Report and metadata differ: {field}')
        if report['model']['model_sha256'] != metadata['model_sha256']:
            raise ValueError('Report and metadata checkpoint identity differ')
        smoothing = report['commands']['smoothing']
        if smoothing != metadata['smoothing'] or smoothing['mode'] != 'none':
            raise ValueError('This compositor labels original unsmoothed policy rollouts')
        if smoothing['control_hz'] != FPS or report['video_fps'] != FPS:
            raise ValueError('Require 30 Hz control and video recording')
        if (metadata['frame_count'] != case['count']
                or report['control_frames'] != case['count']
                or report['video_frames'] != {'up': case['count'], 'wrist': case['count']}):
            raise ValueError('Source report, metadata, and decoded frame counts disagree')
        if not math.isclose(report['duration_s'], (case['count'] - 1) / FPS, abs_tol=1e-8):
            raise ValueError('Source terminal frame is not on its recorded 30 Hz clock')
    return dict(mode='same_policy_success_failure_examples', columns=list(OUTCOMES),
                matched_fields=['model/checkpoint', 'spacing', 'target', 'instruction',
                                'scene settings', 'initial target pose', 'evaluation plan'],
                allowed_differences=['episode_index', 'layout', 'seed', 'policy_seed', 'outcome'],
                same_layout=cases[0]['metadata']['layout_sha256'] == cases[1]['metadata']['layout_sha256'],
                causal_or_paired_model_comparison=False)


def selection_context(path, cases):
    if path is None:
        return None
    path = path.resolve(strict=True)
    selection = policy.read(path)
    profile = cases[0]['report']['spacing_profile']
    groups = [group for group in selection['groups'] if group['spacing_profile'] == profile]
    if len(groups) != 1:
        raise ValueError('Selection manifest must contain exactly one matching spacing')
    group = groups[0]
    for outcome, case in zip(OUTCOMES, cases):
        selected = group['cases'][outcome]
        if Path(selected['directory']).resolve(strict=True) != Path(case['directory']):
            raise ValueError('Input episode is different from selection manifest')
        for field in ('episode_index', 'seed', 'policy_seed', 'model_id', 'task_success',
                      'strict_collector_success', 'duration_s', 'first_success_time_s', 'stop_reason'):
            if selected[field] != case['report'][field]:
                raise ValueError(f'Selection differs from source report: {field}')
        for field in ('model_sha256', 'layout_sha256'):
            if selected[field] != case['metadata'][field]:
                raise ValueError(f'Selection differs from source metadata: {field}')
    if common.sha256(group['audit_file']) != group['audit_sha256']:
        raise ValueError('Selection audit file changed')
    return dict(path=str(path), selection_rule=selection['selection_rule'],
                audit_file=group['audit_file'], audit_sha256=group['audit_sha256'],
                task_successes=group['task_successes'], eval_episodes=group['eval_episodes'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--success', type=Path, required=True)
    parser.add_argument('--failure', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--selection-manifest', type=Path)
    parser.add_argument('--end-hold-seconds', type=float, default=2.)
    args = parser.parse_args()
    if not math.isfinite(args.end_hold_seconds) or not 0 <= args.end_hold_seconds <= 10:
        parser.error('End hold must be finite and in [0, 10] seconds')
    cases = [policy.load(path) for path in (args.success, args.failure)]
    comparison = validate_outcomes(cases)
    context = selection_context(args.selection_manifest, cases)
    args.output.mkdir(parents=True, exist_ok=False)
    _, font_path = common.load_fonts('zh')
    fonts = {name: ImageFont.truetype(font_path, size) for name, size in
             [('title', 28), ('column', 24), ('status', 23), ('note', 18), ('clock', 25), ('small', 16)]}
    left = cases[0]['report']
    profile, color = left['spacing_profile'], left['target_color']
    background = Image.new('RGB', (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(background)
    put = policy.text
    put(draw, (16, 10), f"{policy.PROFILE[profile]} · 按{policy.COLOR[color]}按键 · {left['model_id']}", fonts['title'])
    put(draw, (16, 49), '同一模型，不同回合 / 布局 / 随机种子   |   仿真原速 1× / 30 fps   |   未平滑', fonts['note'], MUTED)
    for col, case in enumerate(cases):
        x, report = col * 640 + 16, case['report']
        label, ink = ('成功示例', '#76d5a6') if col == 0 else ('失败示例', '#ffb67c')
        put(draw, (x, 78), f"{label} · Episode {report['episode_index']} · Seed {report['seed']}", fonts['column'], ink)
        put(draw, (x, 109), f"上方相机 up   |   Policy seed {report['policy_seed']}", fonts['note'], MUTED)
        put(draw, (x, 620), '腕部相机 wrist', fonts['note'], MUTED)
    put(draw, (16, 1197), '原始闭环录像；每帧对应原仿真时刻，结束后仅保持末帧。两列为独立回合。', fonts['small'], MUTED)
    last_count = max(case['count'] for case in cases)
    hold_frames = round(args.end_hold_seconds * FPS)
    total = last_count + hold_frames
    partial, final = args.output / 'execution.partial.mp4', args.output / 'execution.mp4'
    encoder = subprocess.Popen(['ffmpeg', '-v', 'error', '-nostdin', '-n',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s:v', f'{WIDTH}x{HEIGHT}',
        '-r', str(FPS), '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'fast',
        '-crf', '18', '-pix_fmt', 'yuv420p', '-threads', '2', '-movflags', '+faststart',
        str(partial)], stdin=subprocess.PIPE)
    snapshots = {}
    for index, name in ((0, 'initial.jpg'),
                        (cases[0]['count'] - 1, 'success_terminal.jpg'),
                        (total - 1, 'final.jpg')):
        snapshots.setdefault(index, []).append(name)
    if last_count > 12 * FPS:
        snapshots.setdefault(12 * FPS, []).append('at_12s.jpg')
    try:
        with ExitStack() as stack:
            readers = {(col, camera): stack.enter_context(common.FrameReader(
                item['path'], item['metadata'], case['times']))
                for col, case in enumerate(cases) for camera, item in case['videos'].items()}
            for index in range(total):
                seconds = index / FPS
                canvas = background.copy()
                draw = ImageDraw.Draw(canvas)
                put(draw, (WIDTH - 222, 12), f't = {min(index, last_count - 1) / FPS:05.2f} s', fonts['clock'])
                for col, case in enumerate(cases):
                    x, report = col * 640, case['report']
                    for camera, y in (('up', TOP), ('wrist', WRIST_Y)):
                        reader = readers[col, camera]
                        canvas.paste(reader.at(seconds), (x, y))
                        if reader.index != min(index, case['count'] - 1):
                            raise RuntimeError('Source frame index changed relative to simulation time')
                    ended = index >= case['count'] - 1
                    clip_time = min(seconds, report['duration_s'])
                    if not ended:
                        status, ink = f'执行中   t = {clip_time:05.2f} s', FG
                        note = '当前观测驱动模型动作'
                    elif report['task_success']:
                        status = f"任务成功 · 首次成功 {report['first_success_time_s']:.2f} s"
                        ink, note = '#76d5a6', f'已结束，画面保持   |   终止 t = {clip_time:.2f} s'
                    else:
                        reason = {'timeout': f"{report['timeout_s']:g} 秒超时",
                                  'non_target_finger_contact': '触碰非目标',
                                  'unexpected_contact': '异常接触', 'wrong_button': '按错按键'}.get(
                                      report['stop_reason'], report['stop_reason'])
                        status, ink = f'任务失败 · {reason}', '#ffb67c'
                        note = f'已结束，画面保持   |   终止 t = {clip_time:.2f} s'
                    put(draw, (x + 16, FOOTER_Y + 12), status, fonts['status'], ink)
                    put(draw, (x + 16, FOOTER_Y + 44), note, fonts['note'], MUTED)
                for snapshot in snapshots.get(index, []):
                    canvas.save(args.output / snapshot, quality=95)
                encoder.stdin.write(canvas.tobytes())
                if index % 150 == 0:
                    print(json.dumps(dict(profile=profile, frame=index, frames=total)), flush=True)
        encoder.stdin.close()
        if encoder.wait(timeout=60):
            raise RuntimeError('Video encoder failed')
    except BaseException:
        if encoder.poll() is None:
            encoder.terminate()
            encoder.wait(timeout=10)
        raise
    output = common.probe(partial)
    if (output['frame_count'] != total or output['fps'] != FPS
            or (output['width'], output['height']) != (WIDTH, HEIGHT)):
        raise RuntimeError('Encoded outcome comparison has incorrect timing or geometry')
    partial.replace(final)
    manifest = dict(output=str(final.resolve()), output_metadata=output,
        fps=FPS, frames=total, source_last_sim_time_s=(last_count - 1) / FPS,
        end_hold_seconds=hold_frames / FPS, appended_end_hold_frames=hold_frames,
        playback='1x simulation time; output frame i uses source frame min(i, source_count-1)',
        policy_evaluation=True, no_new_simulation=True, no_action_changes=True,
        source_rollouts='Original unsmoothed closed-loop learned-policy episodes',
        image_processing='Uncropped, unresized 640x480 views; labels outside image cells; H.264 encoding only',
        comparison=comparison, columns=list(OUTCOMES), rows=['up', 'wrist'], selection=context,
        source_rectangles={outcome: {camera: [col * 640, y, 640, 480]
            for camera, y in (('up', TOP), ('wrist', WRIST_Y))}
            for col, outcome in enumerate(OUTCOMES)},
        sources=[dict(outcome=outcome, directory=case['directory'],
            **{key: case['report'][key] for key in ('model_id', 'episode_index', 'seed',
                'policy_seed', 'spacing_profile', 'target_color', 'instruction', 'task_success',
                'strict_collector_success', 'first_success_time_s', 'duration_s', 'stop_reason')},
            model_sha256=case['metadata']['model_sha256'],
            checkpoint_step=case['report']['model']['checkpoint_step'],
            layout_sha256=case['metadata']['layout_sha256'],
            positions_xy_m=case['metadata']['positions_xy_m'],
            smoothing=case['report']['commands']['smoothing'], frame_count=case['count'],
            terminal_frame_index=case['count'] - 1, held_frames_after_terminal=total - case['count'],
            videos=case['videos']) for outcome, case in zip(OUTCOMES, cases)])
    common.write_json(args.output / 'manifest.json', manifest)
    print(json.dumps(dict(output=str(final), frames=total, fps=FPS)), flush=True)


if __name__ == '__main__':
    main()
