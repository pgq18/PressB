"""Global ownership, atomic publication and failure gates for multi-GPU capture."""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('sharded_manipulation', ROOT/'scripts/run_sharded_manipulation_collection.py')
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)


@pytest.fixture
def collection(tmp_path):
    raw = tmp_path/'canonical'; raw.mkdir()
    manifest = dict(task_id='tabletop_cleanup', instruction='Cleanup.', target_episodes=6,
                    seed=10, noise_deg=.12, settle_seconds=1., pair_schedule=[0,1,2,0,1,2],
                    config=dict(gpu=1, num_envs=16, log_dir=str(raw/'simulation')))
    runner.write_json(raw/'manifest.json', manifest)
    return raw, manifest


def episode(root, manifest, index, attempt):
    path = root/'episodes'/f'episode_{index:06d}'; path.mkdir(parents=True)
    metadata = dict(episode_index=index, attempt_index=attempt, seed=manifest['seed']+1009*attempt,
                    task_id=manifest['task_id'], instruction=manifest['instruction'], success=True,
                    audit={'passed':True}, frame_count=2, fps=30, noise={'amplitude_deg':manifest['noise_deg']},
                    camera_resolution=[640,480], state_units=['rad']*6+['m'], action_units=['rad']*6+['m'],
                    gripper_representation='full_width', terminal_action='repeat_previous_interval_target',
                    reset_spec={'objects':{kind:{} for kind in runner.PAIRS[manifest['pair_schedule'][index]]}})
    if 'trajectory_profile' in manifest:
        metadata['trajectory_profile'] = manifest['trajectory_profile']
    for name in runner.REQUIRED:
        (path/name).write_bytes(b'fixture')
    runner.write_json(path/'metadata.json', metadata)
    return path, metadata


def test_split_preserves_existing_and_attempts_are_disjoint(collection):
    raw, manifest = collection
    episode(raw, manifest, 0, 1); episode(raw, manifest, 2, 9)
    runner.write_json(raw/'progress.json', {'attempts':11})
    (raw/'rejected').mkdir(); (raw/'rejected/attempt_000013.json').write_text('{}')
    episodes = runner.scan_episodes(raw, manifest)
    plan = runner.make_plan(raw, manifest, episodes, [0,1], 16, 5000)
    assert plan['initial_accepted'] == 2
    assert plan['max_previous_attempt'] == 13
    assert [w['episode_indices'] for w in plan['workers']] == [[1,4],[3,5]]
    attempts = [{w['attempt_offset']+w['attempt_stride']*i for i in range(1,11)} for w in plan['workers']]
    assert not attempts[0] & attempts[1]
    assert min(attempts[0] | attempts[1]) > 13
    assert plan['schema_version'] == 2 and plan['trajectory_profile'] == 'legacy'
    runner.validate_plan(plan, raw, manifest, episodes, [0,1], 16, 5000)


@pytest.mark.parametrize('profile', ['legacy', 'compact'])
def test_worker_command_pins_canonical_profile(collection, profile):
    raw, manifest = collection
    if profile == 'compact':
        manifest['trajectory_profile'] = profile
        runner.write_json(raw/'manifest.json', manifest)
    plan = runner.make_plan(raw, manifest, {}, [0,1], 16, 5000)
    assert plan['trajectory_profile'] == profile
    for worker in plan['workers']:
        command = runner.worker_command(worker, plan, manifest, Path('/python'))
        assert command[command.index('--trajectory-profile')+1] == profile
    runner.validate_plan(plan, raw, manifest, {}, [0,1], 16, 5000)


def test_legacy_shard_plan_without_profile_remains_resumable(collection):
    raw, manifest = collection
    plan = runner.make_plan(raw, manifest, {}, [0,1], 16, 5000)
    plan['schema_version'] = 1
    del plan['trajectory_profile']
    runner.validate_plan(plan, raw, manifest, {}, [0,1], 16, 5000)
    command = runner.worker_command(plan['workers'][0], plan, manifest, Path('/python'))
    assert command[command.index('--trajectory-profile')+1] == 'legacy'


@pytest.mark.parametrize('profile', ['compact', None, 'unknown'])
def test_episode_profile_mismatch_is_rejected_for_legacy_manifest(collection, profile):
    raw, manifest = collection
    path, metadata = episode(raw, manifest, 0, 1)
    metadata['trajectory_profile'] = profile
    runner.write_json(path/'metadata.json', metadata)
    with pytest.raises(ValueError, match='trajectory profile'):
        runner.validate_episode(path, manifest)


def test_legacy_episode_cannot_enter_compact_dataset(collection):
    raw, manifest = collection
    path, _ = episode(raw, manifest, 0, 1)
    manifest['trajectory_profile'] = 'compact'
    with pytest.raises(ValueError, match='trajectory profile'):
        runner.validate_episode(path, manifest)
    metadata = runner.read_json(path/'metadata.json')
    metadata['trajectory_profile'] = 'compact'
    runner.write_json(path/'metadata.json', metadata)
    assert runner.validate_episode(path, manifest)['trajectory_profile'] == 'compact'


def test_shard_plan_profile_and_version_are_validated(collection):
    raw, manifest = collection
    plan = runner.make_plan(raw, manifest, {}, [0,1], 16, 5000)
    plan['trajectory_profile'] = 'compact'
    with pytest.raises(ValueError, match='configuration'):
        runner.validate_plan(plan, raw, manifest, {}, [0,1], 16, 5000)
    with pytest.raises(ValueError, match='trajectory profile'):
        runner.worker_command(plan['workers'][0], plan, manifest, Path('/python'))
    del plan['trajectory_profile']
    with pytest.raises(ValueError, match='explicit trajectory profile'):
        runner.validate_plan(plan, raw, manifest, {}, [0,1], 16, 5000)
    plan['schema_version'] = 3
    with pytest.raises(ValueError, match='schema version'):
        runner.validate_plan(plan, raw, manifest, {}, [0,1], 16, 5000)


def test_worker_manifest_must_match_compact_profile_before_publication(collection):
    raw, manifest = collection
    manifest['trajectory_profile'] = 'compact'
    runner.write_json(raw/'manifest.json', manifest)
    plan = runner.make_plan(raw, manifest, {}, [0,1], 16, 5000)
    worker = plan['workers'][0]
    config = dict(manifest['config'], gpu=worker['gpu'], num_envs=worker['num_envs'],
                  log_dir=str(Path(worker['output'])/'simulation'))
    worker_manifest = dict(manifest, config=config, episode_indices=worker['episode_indices'],
                           global_target_episodes=manifest['target_episodes'])
    del worker_manifest['trajectory_profile']
    runner.write_json(Path(worker['output'])/'manifest.json', worker_manifest)
    with pytest.raises(ValueError, match='semantics/config'):
        runner.merge_workers(raw, manifest, plan, {})
    worker_manifest['trajectory_profile'] = 'compact'
    runner.write_json(Path(worker['output'])/'manifest.json', worker_manifest)
    assert runner.merge_workers(raw, manifest, plan, {}) == 0


@pytest.mark.parametrize('retry', [False, True])
def test_old_worker_receipt_keeps_legacy_semantics_on_completion_or_retry(collection, monkeypatch, retry):
    raw, manifest = collection
    plan = runner.make_plan(raw, manifest, {}, [0,1], 16, 5000)
    worker = plan['workers'][0]
    command = runner.worker_command(worker, plan, manifest, Path('/python'))
    old_command = list(command)
    index = old_command.index('--trajectory-profile')
    del old_command[index:index+2]
    receipt = dict(status='failed' if retry else 'exited', exit_code=1 if retry else 0,
                   command=old_command)
    runner.write_json(worker['receipt'], receipt)
    runner.write_json(Path(worker['output'])/'progress.json', {'status':'failed' if retry else 'complete'})
    spawned = []
    monkeypatch.setattr(runner.subprocess, 'Popen', lambda *args, **kwargs: spawned.append(args))
    runner.start_worker(worker, plan, manifest, Path('/python'), Path('/utility'), retry)
    if retry:
        assert len(spawned) == 1
        assert runner.read_json(worker['receipt'])['command'] == command
        archived = list(Path(worker['receipt']).parent.glob('worker_00.prior_*.json'))
        assert len(archived) == 1 and runner.read_json(archived[0]) == receipt
    else:
        assert not spawned
        assert runner.read_json(worker['receipt']) == receipt


def test_merge_keeps_worker_source_and_is_idempotent(collection):
    raw, manifest = collection
    initial, _ = episode(raw, manifest, 0, 1)
    before = runner.digest(initial/'metadata.json')
    episodes = runner.scan_episodes(raw, manifest)
    plan = runner.make_plan(raw, manifest, episodes, [0,1], 16, 5000)
    worker = plan['workers'][0]; index = worker['episode_indices'][0]
    source, metadata = episode(Path(worker['output']), manifest, index, worker['attempt_offset']+2)
    assert runner.merge_workers(raw, manifest, plan, episodes) == 1
    destination = raw/'episodes'/source.name
    assert source.is_dir() and destination.is_dir()
    assert os.path.samestat((source/'up.mp4').stat(), (destination/'up.mp4').stat())
    assert runner.merge_workers(raw, manifest, plan, episodes) == 0
    assert runner.digest(initial/'metadata.json') == before
    assert not list((raw/'episodes').glob('.incoming_*'))


def test_wrong_global_pair_is_rejected(collection):
    raw, manifest = collection
    path, metadata = episode(raw, manifest, 0, 1)
    metadata['reset_spec']['objects'] = {'speaker':{}, 'lidar':{}}
    runner.write_json(path/'metadata.json', metadata)
    with pytest.raises(ValueError, match='schedule mismatch'):
        runner.scan_episodes(raw, manifest)


def test_wrong_worker_attempt_residue_is_rejected(collection):
    raw, manifest = collection
    plan = runner.make_plan(raw, manifest, {}, [0,1], 16, 5000)
    worker = plan['workers'][0]
    episode(Path(worker['output']), manifest, worker['episode_indices'][0], worker['attempt_offset']+1)
    with pytest.raises(ValueError, match='ownership'):
        runner.merge_workers(raw, manifest, plan, {})


def test_canonical_collision_never_overwritten(collection):
    raw, manifest = collection
    canonical, metadata = episode(raw, manifest, 0, 1)
    source, _ = episode(raw/'worker', manifest, 0, 3)
    before = runner.digest(canonical/'metadata.json')
    with pytest.raises(ValueError, match='overwrite'):
        runner.publish_episode(source, canonical, runner.read_json(source/'metadata.json'), metadata)
    assert runner.digest(canonical/'metadata.json') == before


def test_zero_exit_without_subset_completion_is_failed(collection):
    raw, manifest = collection
    plan = runner.make_plan(raw, manifest, {}, [0,1], 16, 5000)
    worker = plan['workers'][0]
    runner.write_json(worker['receipt'], {'status':'exited', 'exit_code':0})
    runner.write_json(Path(worker['output'])/'progress.json', {'status':'failed','accepted':1,'target':3})
    assert runner.worker_state(worker)['status'] == 'failed'


def test_end_to_end_fake_workers_publish_all_and_keep_manifest(collection, monkeypatch):
    raw, manifest = collection
    episode(raw, manifest, 0, 1); episode(raw, manifest, 4, 2)
    before = runner.digest(raw/'manifest.json')
    def start(worker, plan, model, *args):
        for local_index, index in enumerate(worker['episode_indices'], start=1):
            episode(Path(worker['output']), model, index, worker['attempt_offset']+2*local_index)
        count = len(worker['episode_indices'])
        runner.write_json(worker['receipt'], {'status':'exited','exit_code':0})
        runner.write_json(Path(worker['output'])/'progress.json', {'status':'complete','accepted':count,'target':count})
    monkeypatch.setattr(runner, 'start_worker', start)
    args = SimpleNamespace(canonical=raw, gpus=[0,1], num_envs=16, max_attempts=5000,
                           sim_python=Path('/unused'), utility_python=Path('/unused'),
                           retry_failed_workers=False, poll_seconds=.01)
    assert runner.run(args) == 0
    progress = runner.read_json(raw/'progress.json')
    assert progress['status'] == 'complete' and progress['accepted'] == 6
    assert runner.digest(raw/'manifest.json') == before
    assert sorted(progress['pair_counts'].values()) == [2,2,2]
    assert len(list((raw/'episodes').glob('episode_*'))) == 6
    assert len(list((raw/'shards/shard_00/episodes').glob('episode_*'))) == 2


def test_failed_workers_cannot_publish_complete(collection, monkeypatch):
    raw, manifest = collection
    def start(worker, *args):
        runner.write_json(worker['receipt'], {'status':'exited','exit_code':0})
        runner.write_json(Path(worker['output'])/'progress.json', {'status':'failed','accepted':0,'target':3})
    monkeypatch.setattr(runner, 'start_worker', start)
    args = SimpleNamespace(canonical=raw, gpus=[0,1], num_envs=16, max_attempts=5000,
                           sim_python=Path('/unused'), utility_python=Path('/unused'),
                           retry_failed_workers=False, poll_seconds=.01)
    with pytest.raises(RuntimeError, match='incomplete'):
        runner.run(args)
    assert runner.read_json(raw/'progress.json')['status'] == 'failed'
