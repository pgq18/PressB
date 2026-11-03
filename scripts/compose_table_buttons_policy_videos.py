#!/usr/bin/env python3
"""Compose paired policy videos on their recorded simulation clock.

Use --single/--multitask for different models, or --before/--after for one
model with no smoothing versus joint EMA. The latter requires identical model,
layout, language, episode and policy noise seed and labels the actual EMA alpha.

Both camera images retain their full 640x480 pixels. One input frame is used
per output frame at 30 Hz. Early terminal images are held with an explicit label;
no action interpolation, retiming, smoothing, or simulator replay is performed.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont

import compose_rl_comparison_videos as common

FPS, WIDTH, HEIGHT = 30, 1280, 1180
TOP, WRIST_Y, FOOTER_Y = 100, 608, 1088
BG, FG, MUTED = '#101922', '#f3f6fa', '#b4c1cf'
PROFILE = {'near': '近距离', 'medium': '中距离', 'far': '远距离'}
COLOR = {'red': '红色', 'yellow': '黄色', 'blue': '蓝色'}


def read(path):
    return json.loads(Path(path).read_text())


def load(directory):
    directory = Path(directory).resolve(strict=True)
    report, metadata = read(directory / 'report.json'), read(directory / 'metadata.json')
    if report.get('mode') != 'policy_evaluation' or report.get('expert_actions_used') is not False:
        raise ValueError('Only recorded learned-policy episodes are supported')
    if metadata['evaluation_status'] != 'complete' or metadata['fps'] != FPS:
        raise ValueError('Require a completed 30 Hz episode')
    count = round(report['duration_s'] * FPS) + 1
    videos = {}
    for camera in ('up', 'wrist'):
        path = directory / f'{camera}.mp4'
        probe = common.probe(path)
        if ((probe['width'], probe['height']) != (640, 480)
                or probe['fps'] != FPS or probe['frame_count'] != count):
            raise ValueError(f'Source geometry/timing mismatch: {path}')
        videos[camera] = dict(path=str(path), metadata=probe)
    return dict(directory=str(directory), report=report, metadata=metadata,
                videos=videos, count=count, times=[i / FPS for i in range(count)])


def text(draw, xy, value, font, color=FG):
    draw.text(xy, value, font=font, fill=color, anchor='lt')


def validate_pair(cases, mode):
    """Reject misleading comparisons before creating any output files."""
    left, right = [case['report'] for case in cases]
    for field in ('seed', 'target_color', 'spacing_profile', 'instruction', 'config'):
        if left[field] != right[field]:
            raise ValueError(f'Comparison changes {field}')
    for field in ('layout_sha256', 'policy_seed'):
        if cases[0]['metadata'][field] != cases[1]['metadata'][field]:
            raise ValueError(f'Comparison changes {field}')
    if mode == 'single_multitask':
        if left['model']['task_type'] != 'single' or right['model']['task_type'] != 'multitask':
            raise ValueError('Column labels require single-task then multitask models')
        return dict(mode=mode, columns=['single', 'multitask'], labels=['单任务模型', '多任务模型'])
    if mode != 'smoothing':
        raise ValueError(f'Unknown comparison mode: {mode}')
    for field in ('cell_id', 'model_id', 'episode_index', 'policy_seed', 'timeout_s',
                  'collision_policy', 'physics_dt', 'initial_q_target'):
        if left[field] != right[field]:
            raise ValueError(f'Smoothing comparison changes {field}')
    for field in ('model_sha256', 'model_id', 'episode_index', 'config_sha256',
                  'positions_xy_m', 'camera_names', 'camera_resolution'):
        if cases[0]['metadata'][field] != cases[1]['metadata'][field]:
            raise ValueError(f'Smoothing comparison changes metadata {field}')
    if left['model']['model_sha256'] != right['model']['model_sha256']:
        raise ValueError('Smoothing comparison changes model SHA')
    smoothing = []
    for case in cases:
        report, metadata = case['report'], case['metadata']
        spec = report['commands']['smoothing']
        if spec != metadata['smoothing']:
            raise ValueError('Report and metadata smoothing differ')
        if (report['model']['model_sha256'] != metadata['model_sha256']
                or report['policy_seed'] != metadata['policy_seed']
                or report['episode_index'] != metadata['episode_index']):
            raise ValueError('Report and metadata identity differ')
        if spec['gripper_filtered'] is not False or spec['control_hz'] != FPS:
            raise ValueError('Require arm-only smoothing at the recorded 30 Hz control rate')
        smoothing.append(spec)
    if smoothing[0]['mode'] != 'none' or smoothing[1]['mode'] != 'joint_ema':
        raise ValueError('Smoothing columns must be none on the left, joint_ema on the right')
    alpha = smoothing[1]['alpha']
    if (isinstance(alpha, bool) or not isinstance(alpha, (int, float))
            or not math.isfinite(alpha) or not 0 < alpha <= 1):
        raise ValueError('EMA alpha must be finite and in (0, 1]')
    return dict(mode=mode, columns=['before', 'after'],
                labels=['未平滑（none）', f'六关节 EMA α={alpha:g}'],
                alpha=alpha, smoothing=smoothing, gripper_filtered=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--single', type=Path)
    parser.add_argument('--multitask', type=Path)
    parser.add_argument('--before', type=Path, help='Same-model episode without smoothing (left)')
    parser.add_argument('--after', type=Path, help='Same-model joint EMA episode (right)')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--end-hold-seconds', type=float, default=2.)
    args = parser.parse_args()
    if not 0 <= args.end_hold_seconds <= 10:
        parser.error('End hold must be in [0, 10] seconds')
    legacy, smoothing = (args.single, args.multitask), (args.before, args.after)
    if all(legacy) and not any(smoothing):
        directories, mode = legacy, 'single_multitask'
    elif all(smoothing) and not any(legacy):
        directories, mode = smoothing, 'smoothing'
    else:
        parser.error('Choose exactly one complete pair: --single/--multitask or --before/--after')
    cases = [load(directory) for directory in directories]
    left, right = [case['report'] for case in cases]
    comparison = validate_pair(cases, mode)
    args.output.mkdir(parents=True, exist_ok=False)
    _, font_path = common.load_fonts('zh')
    fonts = {key: ImageFont.truetype(font_path, size) for key, size in
             [('title', 28), ('column', 23), ('status', 23), ('note', 18), ('clock', 25)]}
    profile, color = left['spacing_profile'], left['target_color']
    background = Image.new('RGB', (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(background)
    title = f'桌面按键 · {PROFILE[profile]} · 按{COLOR[color]}按键'
    if mode == 'smoothing':
        title += f" · {left['model_id']}"
    text(draw, (16, 9), title, fonts['title'])
    text(draw, (16, 45), f"指令：{left['instruction']}   |   Seed {left['seed']}   |   仿真原速 1× / 30 fps", fonts['note'], MUTED)
    for column, (case, label) in enumerate(zip(cases, comparison['labels'])):
        x = column * 640 + 16
        heading = f"{label} · 第三视角" if mode == 'smoothing' else f"{label}  {case['report']['model_id']} · 第三视角"
        text(draw, (x, 73), heading, fonts['column'])
        text(draw, (x, 584), '腕部视角', fonts['note'], MUTED)
    footer = ('同一模型及场景；右侧仅平滑六个关节，夹爪不参与平滑。原速对齐；结束后保持画面。'
              if mode == 'smoothing' else '允许正常夹爪闭合的任务成功口径；原始录像，无额外动作平滑。')
    text(draw, (16, 1156), footer, fonts['note'], MUTED)
    last_count = max(case['count'] for case in cases)
    total = last_count + round(args.end_hold_seconds * FPS)
    partial, final = args.output / 'execution.partial.mp4', args.output / 'execution.mp4'
    command = ['ffmpeg', '-v', 'error', '-nostdin', '-n', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s:v', f'{WIDTH}x{HEIGHT}', '-r', str(FPS), '-i', 'pipe:0', '-an',
               '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
               '-threads', '2', '-movflags', '+faststart', str(partial)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    snapshots = {0: 'initial.jpg', 12 * FPS: 'at_12s.jpg',
                 max(case['count'] for case in cases) - 1: 'final.jpg'}
    try:
        with ExitStack() as stack:
            readers = {(col, camera): stack.enter_context(common.FrameReader(
                item['path'], item['metadata'], case['times']))
                for col, case in enumerate(cases) for camera, item in case['videos'].items()}
            for index in range(total):
                seconds = index / FPS
                canvas = background.copy()
                draw = ImageDraw.Draw(canvas)
                global_time = min(seconds, (last_count - 1) / FPS)
                text(draw, (WIDTH - 238, 12), f't = {global_time:05.2f} s', fonts['clock'])
                for col, case in enumerate(cases):
                    x, report = col * 640, case['report']
                    for camera, y in (('up', TOP), ('wrist', WRIST_Y)):
                        reader = readers[col, camera]
                        canvas.paste(reader.at(seconds), (x, y))
                        if reader.index != min(index, case['count'] - 1):
                            raise RuntimeError('Video frame index changed relative to physical clock')
                    ended = index >= case['count'] - 1
                    clip_time = min(seconds, report['duration_s'])
                    if not ended:
                        status, ink = f'执行中   t = {clip_time:05.2f} s', FG
                        note = '已超过原 12 秒时限，继续执行' if seconds >= 12 else '当前观测驱动模型动作'
                        if mode == 'smoothing' and col == 1:
                            note = f"六关节 EMA α={comparison['alpha']:g}；夹爪不参与平滑"
                    elif report['task_success']:
                        status = f"任务成功 · 首次成功 {report['first_success_time_s']:.2f} s"
                        ink, note = '#76d5a6', f'已结束，画面保持   |   终止 t = {clip_time:.2f} s'
                    else:
                        reason = {'timeout': f"{report.get('timeout_s', 30):g} 秒超时", 'non_target_finger_contact': '触碰非目标',
                                  'unexpected_contact': '异常接触', 'wrong_button': '按错按键'}.get(report['stop_reason'], report['stop_reason'])
                        status, ink = f'任务失败 · {reason}', '#ffb67c'
                        note = f'已结束，画面保持   |   终止 t = {clip_time:.2f} s'
                    text(draw, (x + 16, FOOTER_Y + 10), status, fonts['status'], ink)
                    text(draw, (x + 16, FOOTER_Y + 42), note, fonts['note'], MUTED)
                if index in snapshots:
                    canvas.save(args.output / snapshots[index], quality=95)
                encoder.stdin.write(canvas.tobytes())
                if index % 300 == 0:
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
        raise RuntimeError('Encoded comparison has incorrect timing or geometry')
    partial.replace(final)
    manifest = dict(output=str(final.resolve()), output_metadata=output, fps=FPS, frames=total,
        source_last_sim_time_s=(last_count-1)/FPS, end_hold_seconds=args.end_hold_seconds,
        playback='1x simulation time; original frame i at output i; explicit terminal-image holds only',
        policy_evaluation=True, no_new_simulation=True, no_action_changes=True,
        processing_scope='no_new_simulation and no_action_changes describe video composition only, not the source rollouts',
        source_rollouts=('Separate closed-loop physical episodes; joint EMA was applied during the after rollout, never by this compositor'
                         if mode == 'smoothing' else 'Separate single-task and multitask closed-loop physical episodes'),
        image_processing='Uncropped and unresized 640x480 views; labels outside image cells; H.264 re-encoding only',
        comparison=comparison, columns=comparison['columns'], rows=['up', 'wrist'],
        alignment=dict(seed=left['seed'], policy_seed=cases[0]['metadata']['policy_seed'],
                       layout_sha256=cases[0]['metadata']['layout_sha256'],
                       episode_index=left.get('episode_index'), instruction=left['instruction']),
        source_rectangles=dict(up=[0, TOP, 640, 480], wrist=[0, WRIST_Y, 640, 480]),
        sources=[dict(directory=case['directory'], model_id=case['report']['model_id'],
            model_sha256=case['report']['model']['model_sha256'], task_success=case['report']['task_success'],
            strict_collector_success=case['report']['strict_collector_success'],
            first_success_time_s=case['report']['first_success_time_s'],
            duration_s=case['report']['duration_s'], stop_reason=case['report']['stop_reason'],
            smoothing=case['report']['commands']['smoothing'],
            frame_count=case['count'], videos=case['videos']) for case in cases])
    common.write_json(args.output / 'manifest.json', manifest)
    print(json.dumps(dict(output=str(final), frames=total, fps=FPS)), flush=True)


if __name__ == '__main__':
    main()
