#!/usr/bin/env python3
"""Verify that a press-only dataset preserves an already simulated action prefix.

This reuses immutable physical evidence from the full-cycle replay. It does
not launch a new simulation and does not claim to have replayed all episodes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--replay', type=Path, default=ROOT / 'outputs/replay_edge_30hz/run_v1')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    dataset_root, replay_root = args.dataset.resolve(), args.replay.resolve()
    require(not args.report.exists(), 'Report already exists; preserve the existing evidence')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    os.environ.setdefault('HF_HOME', str(ROOT / '.cache/hf'))
    os.environ.setdefault('HF_DATASETS_DISABLE_PROGRESS_BARS', '1')
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    manifest_path = dataset_root / 'meta/export_manifest.json'
    manifest = read(manifest_path)
    dataset_audit = read(dataset_root / 'meta/audit.json')
    require(dataset_audit['success'] and not dataset_audit.get('errors'), 'Press dataset audit failed')
    require(dataset_audit['manifest_sha256'] == digest(manifest_path), 'Press dataset manifest changed after audit')
    replay_manifest = read(replay_root / 'replay_manifest.json')
    replay_audit = read(replay_root / 'audit.json')
    require(replay_audit['success'] and not replay_audit['errors'], 'Full-cycle replay evidence is not verified')
    require(replay_audit['replay_provenance']['manifest_sha256'] == digest(replay_root / 'replay_manifest.json'),
            'Replay manifest changed after its independent audit')
    # The archived controller is causal: command knot k+1 depends only on
    # action[k] and previous solutions; the last action is checked, not executed.
    for filename in ('src/pressb/replay_control.py', 'scripts/replay_dataset.py'):
        expected = replay_manifest['source_code_sha256'][filename]
        require(digest(ROOT / filename) == expected, f'Replay controller changed: {filename}')
        require(digest(replay_root / 'provenance' / filename) == expected,
                f'Archived replay controller changed: {filename}')
    dataset = LeRobotDataset('local/piper_elevator_press', root=dataset_root, video_backend='pyav')
    require(dataset.fps == 30, 'Unexpected replay sample rate')
    by_source = {entry['episode_id']: (index, entry) for index, entry in enumerate(manifest['episodes'])}
    physical_entries = {entry['source_episode_id']: entry for entry in replay_audit['episodes']}
    raw_root = Path(manifest['raw_root'])
    evidence = []
    for source_id in sorted(physical_entries):
        require(source_id in by_source, f'Missing previously replayed source episode {source_id}')
        index, exported = by_source[source_id]
        meta = dataset.meta.episodes[index]
        start, stop = int(meta['dataset_from_index']), int(meta['dataset_to_index'])
        count = stop - start
        replay_episode = replay_root / f'episode_{source_id:06d}'
        replay_meta = read(replay_episode / 'metadata.json')
        prior = physical_entries[source_id]
        require(digest(replay_episode / 'metadata.json') == prior['metadata_sha256'], 'Replay metadata changed')
        require(digest(replay_episode / 'physics.npz') == prior['physics_sha256'], 'Replay physics changed')
        prepared_path = Path(replay_meta['input_file'])
        require(digest(prepared_path) == replay_meta['input_sha256'], 'Original replay input changed')
        with np.load(prepared_path, allow_pickle=False) as original:
            expected_actions = original['action'][:count].copy()
            expected_actions[-1] = original['action'][count - 2]
            expected_states = original['state'][:count].copy()
            expected_initial_q = original['initial_joint_position'].copy()
        values = dataset.select_columns(['action', 'observation.state', 'observation.joint_position',
                                         'source_episode_id']).select(range(start, stop)).with_format('numpy')[:]
        action = np.asarray(values['action'], dtype=np.float32)
        state = np.asarray(values['observation.state'], dtype=np.float32)
        initial_q = np.asarray(values['observation.joint_position'][0], dtype=np.float32)
        require(np.array_equal(action, expected_actions), 'New actions differ from the verified simulated prefix')
        require(np.array_equal(state, expected_states), 'New states differ from the recorded prefix')
        require(np.array_equal(initial_q, expected_initial_q), 'New episode changed the single initial IK seed')
        require(np.all(np.asarray(values['source_episode_id']) == source_id), 'Source identity mismatch')
        raw_episode = raw_root / f'episode_{source_id:06d}'
        require(digest(raw_episode / 'frames.npz') == exported['source_sha256']['frames.npz'], 'Raw source changed')
        floor = exported['floor']
        with np.load(raw_episode / 'frames.npz', allow_pickle=False) as raw:
            first_lit = int(np.flatnonzero(raw['lights'][:, floor - 24])[0])
            require(count == first_lit + 1, 'New episode does not end at first recorded light-on sample')
        last_step = first_lit * 4
        with np.load(replay_episode / 'physics.npz', allow_pickle=False) as physics:
            lights = physics['lights'][:last_step + 1]
            require(lights[-1, floor - 24] == 1, 'Replayed target was not pressed at the new endpoint')
            require(not np.delete(lights, floor - 24, axis=1).any(), 'Another floor was triggered in the prefix')
            events = [event for event in replay_meta['events'] if event['physics_index'] <= last_step]
            require([(event['type'], event['floor']) for event in events] == [('pressed', floor)],
                    'Replay prefix did not contain exactly one target press')
            endpoint = {'button_travel_m': float(physics['button_travel'][last_step, floor - 24]),
                        'contact_force_n': float(physics['contact_force'][last_step, floor - 24])}
        require(not replay_meta['unexpected_collisions'], 'Original replay had unintended contacts')
        evidence.append({'source_episode_id': source_id, 'floor': floor, 'dataset_episode_index': index,
                         'kept_frames': count, 'last_physics_index': last_step,
                         'actions_equal_executed_prefix_with_terminal_clamp': True,
                         'states_equal_source_prefix': True, 'initial_seed_unchanged': True,
                         'target_lit_at_prefix_endpoint': True, 'endpoint': endpoint,
                         'full_replay_physics_sha256': prior['physics_sha256']})
    require(len(evidence) == 12 and {row['floor'] for row in evidence} == set(range(24, 36)),
            'Expected the 12 existing replay demonstrations')
    report = {'success': True, 'created_utc': datetime.now(timezone.utc).isoformat(),
              'dataset': str(dataset_root), 'lerobot_version': version('lerobot'),
              'manifest_sha256': digest(manifest_path), 'replay': str(replay_root),
              'replay_audit_sha256': digest(replay_root / 'audit.json'),
              'mode': 'reuse_verified_full_cycle_physics_prefix', 'new_simulation_executed': False,
              'episodes_checked': len(evidence), 'episodes': evidence,
              'original_replay_process_exit': read(replay_root / 'process_exit.json'),
              'limitations': ['This links 12 shortened action prefixes to existing physical replay evidence; it does not claim a new 1200-episode simulation.']}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'success': True, 'episodes_checked': len(evidence), 'report': str(args.report)}, indent=2))


if __name__ == '__main__':
    main()
