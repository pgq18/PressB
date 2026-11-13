#!/usr/bin/env python3
"""Summarize completed fixed-attempt collection benchmarks without loading Isaac.

The collector's engine timers overlap its exclusive stage timers. This report
keeps those two decompositions separate and checks nominal sampled layouts by
attempt ID before comparing parallelism settings.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


GIB = 1024**3
MIB_PER_GIB = 1024


def read(path):
    return json.loads(path.read_text())


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def stat(summary, key, which='mean', scale=1.):
    value = summary.get(key, {}).get(which)
    return value / scale if value is not None else None


def resource_metrics(summary):
    devices = list(summary.get('gpu', {}).values())
    if len(devices) != 1:
        raise ValueError('Benchmark resources must select exactly one GPU')
    gpu = devices[0]
    host = summary.get('host', {})
    return dict(samples=summary.get('selected_samples'), gpu=gpu.get('device'),
        gpu_utilization_mean_pct=stat(gpu, 'gpu_utilization_percent'),
        gpu_utilization_p95_pct=stat(gpu, 'gpu_utilization_percent', 'p95'),
        gpu_utilization_max_pct=stat(gpu, 'gpu_utilization_percent', 'max'),
        gpu_memory_mean_gib=stat(gpu, 'memory_used_mib', scale=MIB_PER_GIB),
        gpu_memory_peak_gib=stat(gpu, 'memory_used_mib', 'max', MIB_PER_GIB),
        gpu_power_mean_w=stat(gpu, 'power_w'), gpu_power_peak_w=stat(gpu, 'power_w', 'max'),
        gpu_temperature_peak_c=stat(gpu, 'temperature_c', 'max'),
        host_cpu_mean_pct=stat(host, 'cpu_percent'), host_cpu_peak_pct=stat(host, 'cpu_percent', 'max'),
        process_cpu_mean_cores=stat(summary, 'process_cpu_percent', scale=100.),
        process_cpu_peak_cores=stat(summary, 'process_cpu_percent', 'max', 100.),
        process_rss_mean_gib=stat(summary, 'process_rss_bytes', scale=GIB),
        process_rss_peak_gib=stat(summary, 'process_rss_bytes', 'max', GIB),
        host_available_memory_min_gib=stat(host, 'memory_available_bytes', 'min', GIB),
        host_swap_peak_gib=stat(host, 'swap_used_bytes', 'max', GIB))


def assignments_for(case, session_id):
    assignments = {}
    batches = []
    for line in (case/'raw/batch_profile.jsonl').read_text().splitlines():
        batch = json.loads(line)
        if batch['session_id'] != session_id:
            continue
        batches.append(batch)
        for slot, assignment in batch['assignments'].items():
            attempt = int(assignment['attempt_index'])
            if attempt in assignments:
                raise ValueError(f'Duplicate attempt {attempt} in {case}')
            assignments[attempt] = {**assignment, 'physical_env_id': int(slot)}
    return assignments, batches


def load_case(case, variant, expected_attempts, attempt_prefix=None):
    receipt = read(case/'receipt.json')
    scope_path = case/'benchmark_scope.json'
    scope = read(scope_path) if scope_path.exists() else None
    if scope is not None:
        scoped_attempts = scope.get('attempt_prefix')
        if (type(scoped_attempts) is not int or scoped_attempts < 1
                or not isinstance(scope.get('reason'), str) or not scope['reason'].strip()):
            raise ValueError(f'Explicit bounded benchmark scope is invalid: {scope_path}')
        if scoped_attempts > expected_attempts:
            raise ValueError(f'Scoped attempts exceed manifest expectation in {case}')
    if receipt.get('status') != 'finished' or (receipt.get('exit_code') != 0 and scope is None):
        raise ValueError(f'Case has not finished successfully: {case}')
    profile = read(case/'raw/collection_profile.json')
    counts = profile['counts']
    assignments, batches = assignments_for(case, profile['session_id'])
    original_sampled_attempts = counts.get('sampled_layouts', 0)
    attempts = original_sampled_attempts
    if scope is None and (attempts != expected_attempts or len(assignments) != attempts):
        raise ValueError(f'Expected {expected_attempts} recorded attempts in {case}, got {attempts}/{len(assignments)}')
    if len(batches) != profile['completed_batches']:
        raise ValueError(f'Incomplete batch accounting in {case}')
    chosen_prefix = attempt_prefix
    if scope is not None:
        if attempt_prefix is not None and attempt_prefix > scoped_attempts:
            raise ValueError(f'Requested prefix exceeds explicit benchmark scope in {case}')
        chosen_prefix = scoped_attempts if attempt_prefix is None else attempt_prefix
    prefix = chosen_prefix is not None
    if prefix:
        if not 0 < chosen_prefix <= attempts:
            raise ValueError(f'Attempt prefix {chosen_prefix} outside sampled count {attempts}')
        selected = []
        selected_count = 0
        for batch in batches:
            selected.append(batch)
            selected_count += len(batch['assignments'])
            if selected_count >= chosen_prefix:
                break
        if selected_count != chosen_prefix:
            raise ValueError('Attempt prefix must end on a full batch boundary for every compared case')
        batches = selected
        included = {int(a['attempt_index']) for b in batches for a in b['assignments'].values()}
        if included != set(range(1, chosen_prefix+1)):
            raise ValueError(f'Selected complete batches do not cover attempts 1..{chosen_prefix} exactly in {case}')
        assignments = {k:v for k,v in assignments.items() if k in included}
        counts = dict(sum((Counter(b['counts']) for b in batches), Counter()))
        stages = dict(sum((Counter(b['exclusive_stage_seconds']) for b in batches), Counter()))
        stages['startup'] = profile['exclusive_stage_seconds'].get('startup', 0.)
        wall = stages['startup'] + sum(b['wall_seconds'] for b in batches)
        capacity = profile['num_envs']*counts.get('execution_world_control_steps', 0)
        profile = {**profile, 'counts': counts, 'session_accepted': counts.get('accepted_episodes',0),
            'wall_seconds': wall, 'exclusive_stage_seconds': stages,
            'uninstrumented_seconds': max(0., wall-sum(stages.values())),
            'engine_cumulative_timings': batches[-1]['engine_cumulative_timings'],
            'motion_slot_utilization': ratio(counts.get('motion_env_steps',0), capacity),
            'active_slot_utilization': ratio(counts.get('motion_env_steps',0)+counts.get('validation_env_steps',0), capacity)}
        attempts = chosen_prefix
    wall = profile['wall_seconds']
    stages = profile['exclusive_stage_seconds']
    startup = stages.get('startup', 0.)
    post_startup = wall - startup
    frames = counts.get('accepted_recorded_frames', 0)
    accepted = counts.get('accepted_episodes', 0)
    if accepted != profile['session_accepted']:
        raise ValueError(f'Accepted episode counters disagree in {case}')
    engine = profile.get('engine_cumulative_timings', {})
    if prefix:
        import monitor_runtime_resources as monitor
        all_samples = [json.loads(line) for line in (case/'resources.jsonl').read_text().splitlines() if line.strip()]
        # Progress is sampled, so these resource boundaries have one sampling
        # interval of uncertainty. Throughput above uses exact batch timers.
        last_index = batches[-1]['batch_index']
        samples = [s for s in all_samples if ((s.get('status') or {}).get('data') or {}).get('session_batches',0) <= last_index]
        gpu_selectors = [str(receipt['gpu'])]
        resources = resource_metrics(monitor.summarize(samples,gpu_selectors=gpu_selectors,roles=['collector']))
        post_samples = [s for s in samples if s['wall_time'] >= receipt['start_unix']+startup]
        post_resources = resource_metrics(monitor.summarize(post_samples,gpu_selectors=gpu_selectors,roles=['collector']))
    else:
        resources = resource_metrics(read(case/'resources_summary.json'))
        post_resources_path = case/'resources_post_startup.json'
        post_resources = resource_metrics(read(post_resources_path)) if post_resources_path.exists() else None
    pairs = Counter('+'.join(item['objects']) for item in assignments.values())
    row = dict(variant=variant, case=case.name, path=str(case.resolve()),
        task_id=profile['task_id'], num_envs=profile['num_envs'], attempts=attempts,
        attempt_prefix=chosen_prefix, benchmark_scope=scope,
        original_process_exit_code=receipt.get('exit_code'), original_profile_status=read(case/'raw/collection_profile.json').get('status'),
        original_sampled_attempts=original_sampled_attempts,
        timing_scope='startup + sum of complete selected batch wall timers' if prefix else 'full collector session',
        resource_scope='sampled progress batch index, boundary uncertainty up to one sample interval' if prefix else 'full process / after startup',
        accepted_episodes=accepted, acceptance_fraction=ratio(accepted, attempts),
        planning_rejections=counts.get('planning_rejections', 0),
        execution_rejections=counts.get('execution_rejections', 0),
        nominal_object_pair_counts=dict(pairs),
        wall_seconds=wall, process_wall_seconds=receipt.get('process_wall_seconds'),
        startup_seconds=startup, wall_excluding_startup_seconds=post_startup,
        accepted_recorded_frames=frames, recorded_frames=counts.get('recorded_frames', 0),
        accepted_motion_env_steps=counts.get('accepted_motion_env_steps', 0),
        motion_env_steps=counts.get('motion_env_steps', 0),
        validation_env_steps=counts.get('validation_env_steps', 0),
        accepted_frames_per_second=ratio(frames, wall),
        accepted_frames_per_second_excluding_startup=ratio(frames, post_startup),
        accepted_episodes_per_hour=ratio(accepted*3600, wall),
        accepted_episodes_per_hour_excluding_startup=ratio(accepted*3600, post_startup),
        all_motion_env_steps_per_second=ratio(counts.get('motion_env_steps', 0), wall),
        all_motion_env_steps_per_second_excluding_startup=ratio(counts.get('motion_env_steps', 0), post_startup),
        motion_slot_utilization=profile.get('motion_slot_utilization'),
        active_slot_utilization=profile.get('active_slot_utilization'),
        exclusive_stage_seconds=stages,
        exclusive_stage_percent_of_wall={key: 100*value/wall for key, value in stages.items()},
        uninstrumented_seconds=profile.get('uninstrumented_seconds', 0.),
        engine_cumulative_timings=engine, resources_whole_process=resources,
        resources_excluding_startup=post_resources,
        engine_nested_shares=dict(
            contact_callback_fraction_of_physics=ratio(engine.get('contact_callback_seconds', 0.), engine.get('physics_seconds', 0.)),
            color_response_fraction_of_render=ratio(engine.get('color_response_seconds', 0.), engine.get('render_seconds', 0.))))
    return row, assignments


def compare_assignments(left, right, left_name, right_name):
    left_ids, right_ids = set(left), set(right)
    mismatches = []
    for attempt in sorted(left_ids & right_ids):
        for field in ('seed', 'objects', 'sampled_objects', 'layout_env_id', 'selected_for_recording'):
            if left[attempt].get(field) != right[attempt].get(field):
                mismatches.append(dict(attempt_index=attempt, field=field,
                    left=left[attempt].get(field), right=right[attempt].get(field)))
    return dict(left=left_name, right=right_name, common_attempts=len(left_ids & right_ids),
        missing_from_left=sorted(right_ids-left_ids), missing_from_right=sorted(left_ids-right_ids),
        nominal_layouts_identical=left_ids == right_ids and not mismatches,
        mismatches=mismatches,
        note='Nominal reset poses and object combinations match. Settled physical poses and outcomes may differ across world topologies.')


def fmt(value, digits=2):
    return '—' if value is None else f'{value:.{digits}f}'


def percent(value):
    return value*100 if value is not None else None


def markdown(report):
    rows = report['cases']
    out = ['# Parallel manipulation collection benchmark', '',
        'Fixed attempt sequences, latest configured scene, full two-camera recording. '
        'Only episodes that pass physical auditing count toward accepted throughput.', '',
        'One frame is one 30 Hz state/action row with both camera images; the terminal row is included. '
        'It is not a physics tick or an individual camera image. Validation holds do not enter the dataset. '
        'Wall time includes simulator startup but excludes teardown and later LeRobot export.', '',
        '| Variant / task | Envs | Accepted / attempts | Wall (s) | Startup (s) | Accepted frames/s | Excl. startup | Episodes/h | Excl. startup | Motion slot use |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    if report.get('attempt_prefix') is not None:
        out[2] += (f" Only the first {report['attempt_prefix']} attempts of each run are compared, ending at complete batch boundaries. "
                   'Prefix wall is startup plus the sum of selected batch timers; negligible bookkeeping between batches is excluded. '
                   'Resource windows follow sampled batch progress and can differ by one sampling interval at a boundary.')
    if any(r.get('benchmark_scope') is not None for r in rows):
        out[2] += (' Explicitly bounded cases use only the complete batches declared in benchmark_scope.json. '
                   'Their original interruption exit codes and any later sampled attempts are preserved in summary.json; '
                   'partial or out-of-scope batches do not contribute to throughput. Prefix resource windows have one sampling interval of boundary uncertainty.')
    for r in rows:
        out.append(f"| {r['variant']} / {r['task_id']} | {r['num_envs']} | {r['accepted_episodes']}/{r['attempts']} | "
            f"{fmt(r['wall_seconds'],1)} | {fmt(r['startup_seconds'],1)} | {fmt(r['accepted_frames_per_second'])} | "
            f"{fmt(r['accepted_frames_per_second_excluding_startup'])} | {fmt(r['accepted_episodes_per_hour'],1)} | "
            f"{fmt(r['accepted_episodes_per_hour_excluding_startup'],1)} | {fmt(percent(r['motion_slot_utilization']),1)}% |")
    out += ['', '## Hardware resources', '',
        'Values below use samples after startup where available; peak VRAM/RSS are sampled whole-process maxima. '
        'CPU cores are the collector process tree CPU percentage divided by 100; RSS sums may count shared pages more than once. '
        'GPU and host totals include any unrelated activity. Sampling may miss brief peaks.', '',
        '| Variant / task / envs | GPU mean / p95 (%) | VRAM peak (GiB) | Power mean / peak (W) | CPU mean / peak (cores) | Host CPU mean (%) | RSS peak (GiB) | Host available min (GiB) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        p = r['resources_excluding_startup'] or r['resources_whole_process']
        w = r['resources_whole_process']
        out.append(f"| {r['variant']} / {r['task_id']} / {r['num_envs']} | "
            f"{fmt(p['gpu_utilization_mean_pct'],1)} / {fmt(p['gpu_utilization_p95_pct'],1)} | {fmt(w['gpu_memory_peak_gib'])} | "
            f"{fmt(p['gpu_power_mean_w'],1)} / {fmt(w['gpu_power_peak_w'],1)} | "
            f"{fmt(p['process_cpu_mean_cores'])} / {fmt(w['process_cpu_peak_cores'])} | {fmt(p['host_cpu_mean_pct'],1)} | "
            f"{fmt(w['process_rss_peak_gib'])} | {fmt(w['host_available_memory_min_gib'])} |")
    out += ['', '## Exclusive collection stages', '',
        'Seconds in this table can be added. The engine breakdown in the following section is nested inside these stages.', '',
        '| Variant / task / envs | Startup | Reset + settle | Planning | Simulation + capture step | Video frame write | Physical log | Commit | Other |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        s = r['exclusive_stage_seconds']
        used = ['startup', 'reset', 'settle', 'planning', 'step_for_recording', 'record_frame', 'physical_log', 'commit']
        vals = [s.get('startup',0), s.get('reset',0)+s.get('settle',0), s.get('planning',0),
                s.get('step_for_recording',0), s.get('record_frame',0), s.get('physical_log',0), s.get('commit',0),
                sum(v for k,v in s.items() if k not in used)+r['uninstrumented_seconds']]
        out.append(f"| {r['variant']} / {r['task_id']} / {r['num_envs']} | "+' | '.join(fmt(v,1) for v in vals)+' |')
    out += ['', '## Nested engine breakdown', '',
        '`physics_seconds` includes contact callbacks; `render_seconds` includes color response. '
        '`after_tick_seconds` is task outcome computation outside the physics timer. '
        'These engine counters cover reset/settle/capture as well as trajectory execution; do not add them to collector stages.', '',
        '| Variant / task / envs | Physics (s) | Contact callback (s) | After tick (s) | Render (s) | Color response (s) | Contact headers | Color images |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        e = r['engine_cumulative_timings']
        vals = [e.get(k) for k in ('physics_seconds','contact_callback_seconds','after_tick_seconds','render_seconds','color_response_seconds')]
        out.append(f"| {r['variant']} / {r['task_id']} / {r['num_envs']} | "+' | '.join(fmt(v,1) for v in vals)
            +f" | {e.get('contact_callback_headers','—')} | {e.get('color_response_images','—')} |")
    out += ['', '## Workload matching', '']
    for c in report['layout_comparisons']:
        out.append(f"- {c['left']} ↔ {c['right']}: {c['common_attempts']} common attempts; "
                   f"nominal seeds / object combinations / positions / yaw identical: **{c['nominal_layouts_identical']}**.")
    out += ['', 'Settled geometry and contact outcomes can vary when the number of cloned environments changes. '
        'Compare both accepted throughput and all executed motion throughput when acceptance differs. '
        'This is a short throughput benchmark, not a full task success-rate evaluation.', '',
        '## Throughput ratios', '',
        '| Comparison | Accepted frames/s ratio (excl. startup) | All motion env-step/s ratio (excl. startup) | Episode/h ratio (excl. startup) |',
        '|---|---:|---:|---:|']
    for c in report['throughput_comparisons']:
        out.append(f"| {c['numerator']} / {c['denominator']} | {fmt(c['accepted_frame_speedup'])} | "
                   f"{fmt(c['motion_step_speedup'])} | {fmt(c['episode_speedup'])} |")
    return '\n'.join(out)+'\n'


def plot(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    rows = report['cases']
    labels = [f"{r['variant']}\n{r['task_id'].replace('tabletop_', '')}\n{r['num_envs']} envs" for r in rows]
    x = np.arange(len(rows))
    colors = ['#187b9e' if r['task_id']=='cube_stack' else '#dc8c35' for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(max(11., len(rows)*1.6), 8.5), constrained_layout=True)
    metrics = [([r['accepted_frames_per_second_excluding_startup'] for r in rows],
                'Accepted data frames / second (startup excluded)', 'frames/s'),
               ([percent(r['motion_slot_utilization']) or 0 for r in rows], 'Motion slot utilization', '%'),
               ([r['resources_whole_process']['gpu_memory_peak_gib'] for r in rows], 'Peak sampled VRAM', 'GiB'),
               ([(r['resources_excluding_startup'] or r['resources_whole_process'])['process_cpu_mean_cores'] for r in rows],
                'Mean process-tree CPU (startup excluded)', 'CPU cores')]
    for ax, (values, title, unit) in zip(axes.flat, metrics):
        bars=ax.bar(x, values, color=colors, width=.65)
        ax.bar_label(bars, fmt='%.1f', padding=3, fontsize=9)
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(unit)
        ax.set_ylim(0, max(values)*1.18 if max(values)>0 else 1)
        ax.grid(axis='y', alpha=.2);ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Stack / Cleanup collection: fixed-attempt workload, dual-camera recording', fontsize=13)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def build_report(inputs, allow_partial=False, attempt_prefix=None):
    rows, assignment_maps, manifests = [], {}, {}
    for directory in inputs:
        variant = directory.name
        if variant in manifests:
            raise ValueError('Variant directories must have unique names')
        manifest = read(directory/'manifest.json')
        manifests[variant] = manifest
        for task in manifest['tasks']:
            for n in manifest['envs']:
                case = directory/f'{task}_{n}'
                if allow_partial and not (case/'result.json').exists():
                    if not ((case/'benchmark_scope.json').exists() and (case/'receipt.json').exists()
                            and read(case/'receipt.json').get('status') == 'finished'):
                        continue
                expected = manifest.get('case_attempts', {}).get(case.name, manifest['attempts'])
                row, assignments = load_case(case, variant, expected, attempt_prefix)
                rows.append(row)
                assignment_maps[(variant, task, n)] = assignments
    if not rows:
        raise ValueError('No completed cases found')
    layout_comparisons, throughput_comparisons = [], []
    def compare(a, b):
        # b is the numerator: larger parallelism or later variant.
        an = f"{a['variant']}/{a['task_id']}/{a['num_envs']}"
        bn = f"{b['variant']}/{b['task_id']}/{b['num_envs']}"
        am = assignment_maps[(a['variant'],a['task_id'],a['num_envs'])]
        bm = assignment_maps[(b['variant'],b['task_id'],b['num_envs'])]
        matching = compare_assignments(am,bm,an,bn)
        layout_comparisons.append(matching)
        throughput_comparisons.append(dict(numerator=bn,denominator=an,
            workload_matched=matching['nominal_layouts_identical'],
            accepted_frame_speedup=ratio(b['accepted_frames_per_second_excluding_startup'],a['accepted_frames_per_second_excluding_startup']),
            motion_step_speedup=ratio(b['all_motion_env_steps_per_second_excluding_startup'],a['all_motion_env_steps_per_second_excluding_startup']),
            episode_speedup=ratio(b['accepted_episodes_per_hour_excluding_startup'],a['accepted_episodes_per_hour_excluding_startup'])))
    for i, left in enumerate(rows):
        for right in rows[i+1:]:
            if left['task_id'] != right['task_id']:
                continue
            same_variant = left['variant']==right['variant']
            same_envs = left['num_envs']==right['num_envs']
            if same_variant or same_envs:
                a,b=(left,right) if not same_variant or left['num_envs']<right['num_envs'] else (right,left)
                compare(a,b)
    return dict(schema_version=1, inputs=[str(p.resolve()) for p in inputs], attempt_prefix=attempt_prefix, cases=rows,
        layout_comparisons=layout_comparisons, throughput_comparisons=throughput_comparisons,
        semantics=dict(frame='One 30 Hz state/action row with both images, including one terminal row; excludes validation holds.',
            motion_env_step='One executed 30 Hz action in one environment; excludes validation, settling, physics substeps.',
            throughput='Accepted episodes/frames only after physical audit; excludes simulator teardown and LeRobot export.',
            resources='Sampled, not instantaneous peak; process RSS can double count shared pages.',
            timings='Collector stage times exclusive. Engine counters nested and not additive with collector times.',
            workload='Exact nominal reset layouts and pair choices checked by global attempt; physical settling not guaranteed identical.'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Baseline benchmark directory')
    parser.add_argument('--variant-dir', type=Path, action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--attempt-prefix', type=int,
                        help='Compare only first N attempts; must end at a full batch boundary in every case')
    parser.add_argument('--plot', action='store_true')
    args=parser.parse_args()
    report=build_report([args.input, *args.variant_dir], args.allow_partial, args.attempt_prefix)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (args.output/'report.md').write_text(markdown(report))
    if args.plot:
        plot(report,args.output/'comparison.png')
    print(json.dumps(dict(cases=len(report['cases']),output=str(args.output.resolve()),
        all_layout_comparisons_match=all(c['nominal_layouts_identical'] for c in report['layout_comparisons']))))


if __name__=='__main__':
    main()
