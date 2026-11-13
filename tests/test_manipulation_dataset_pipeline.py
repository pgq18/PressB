"""Pipeline gates prevent training from launching on incomplete collections."""
import importlib.util
import json
from pathlib import Path
import sys
import threading

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('manipulation_pipeline', ROOT/'scripts/run_manipulation_dataset_pipeline.py')
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


def arguments(tmp_path, *extra):
    return pipeline.parse_args(['--output', str(tmp_path/'output'), '--dataset-root', str(tmp_path/'datasets'),
                                '--utility-python', sys.executable, '--poll-seconds', '.01', *extra])


def test_zero_exit_cannot_pass_missing_physical_completion(tmp_path, monkeypatch):
    flow = pipeline.Pipeline(arguments(tmp_path))
    monkeypatch.setattr(flow, 'run_command', lambda *args: 0)
    with pytest.raises(RuntimeError, match='physical 500-episode completion gate'):
        flow.collect('cube_stack', flow.output/'cube_stack_raw')


def test_complete_progress_requires_successful_episode_files(tmp_path):
    raw = tmp_path/'raw'
    pipeline.write_json(raw/'progress.json', {'status':'complete', 'accepted':1, 'target':1})
    pipeline.write_json(raw/'manifest.json', {'task_id':'cube_stack', 'target_episodes':1})
    episode = raw/'episodes/episode_000000'
    pipeline.write_json(episode/'metadata.json', {'task_id':'cube_stack', 'success':True,
                                                'audit':{'passed':False}, 'episode_index':0})
    for name in ('data.npz', 'up.mp4', 'wrist.mp4'):
        (episode/name).write_bytes(b'fixture')
    with pytest.raises(ValueError, match='physical acceptance'):
        pipeline.raw_complete(raw, 'cube_stack', count=1)


def test_active_external_collector_is_observed_without_duplicate(tmp_path, monkeypatch):
    flow = pipeline.Pipeline(arguments(tmp_path))
    locks = iter([True, False])
    completion = iter([False, True])
    monkeypatch.setattr(pipeline, 'lock_held', lambda p: next(locks))
    monkeypatch.setattr(pipeline, 'external_collector', lambda *a: None)
    monkeypatch.setattr(pipeline, 'raw_complete', lambda *a: next(completion))
    monkeypatch.setattr(flow, 'run_command', lambda *a: pytest.fail('Duplicate collector was started'))
    flow.collect('cube_stack', flow.output/'cube_stack_raw')
    assert flow.state['tasks']['cube_stack']['stage'] == 'waiting_existing_collector'


def test_task_failure_prevents_deployment(tmp_path, monkeypatch):
    flow = pipeline.Pipeline(arguments(tmp_path))
    def task(name):
        if name == 'tabletop_cleanup':
            raise RuntimeError('physical collection failed')
    monkeypatch.setattr(flow, 'task', task)
    monkeypatch.setattr(flow, 'deploy', lambda: pytest.fail('Training launched after collection failure'))
    with pytest.raises(RuntimeError, match='physical collection failed'):
        flow.run()
    assert flow.state['status'] == 'failed'


def test_both_tasks_export_video_before_training(tmp_path, monkeypatch):
    flow = pipeline.Pipeline(arguments(tmp_path, '--cube-gpu', '0', '--cleanup-gpu', '1'))
    events = []
    mutex = threading.Lock()
    def collect(task, raw):
        with mutex: events.append((task, 'collect'))
    def command(label, cmd):
        task = next(t for t in pipeline.TASKS if label.startswith(t))
        with mutex: events.append((task, 'export' if label.endswith('export') else 'video'))
        if label.endswith('export'):
            dataset = Path(cmd[cmd.index('--output')+1])
            pipeline.write_json(dataset/'done.json', {'task':task})
        else:
            output = Path(cmd[cmd.index('--output')+1]);output.mkdir(parents=True, exist_ok=True)
            pipeline.write_json(output/'summary.json', {'task_id':task, 'episodes':500})
            (output/'examples_1x.mp4').write_bytes(b'fixture')
        return 0
    monkeypatch.setattr(flow, 'collect', collect)
    monkeypatch.setattr(flow, 'run_command', command)
    monkeypatch.setattr(pipeline, 'export_complete', lambda path, task, count=500: (path/'done.json').is_file())
    def deploy():
        assert all(flow.state['tasks'][t]['stage'] == 'ready_for_sft' for t in pipeline.TASKS)
        events.append(('both', 'deploy'))
    monkeypatch.setattr(flow, 'deploy', deploy)
    flow.run()
    for task in pipeline.TASKS:
        assert [phase for t, phase in events if t == task] == ['collect', 'export', 'video']
    assert events[-1] == ('both', 'deploy')


def test_deployment_retries_only_temporary_failures(tmp_path, monkeypatch):
    flow = pipeline.Pipeline(arguments(tmp_path))
    codes = iter([75, 0]); pauses = []
    def command(label, cmd):
        code = next(codes)
        pipeline.write_json(flow.args.config_dir/'deployment_status.json',
                            {'status':'retryable' if code else 'jobs_started'})
        return code
    monkeypatch.setattr(flow, 'run_command', command)
    monkeypatch.setattr(flow, 'pause', lambda delay=None: pauses.append(delay))
    flow.deploy()
    assert pauses == [120.]
    assert flow.state['deployment_attempts'] == 2
    assert flow.state['stage'] == 'both_sft_jobs_started'


def test_detached_worker_receipt_is_idempotent(tmp_path):
    flow = pipeline.Pipeline(arguments(tmp_path))
    destination = tmp_path/'counter.txt'
    command = [sys.executable, '-c',
               'from pathlib import Path; p=Path(__import__("sys").argv[1]); '
               'p.write_text(p.read_text()+"x" if p.exists() else "x")', str(destination)]
    assert flow.run_command('cpu_probe', command) == 0
    assert flow.run_command('cpu_probe', command) == 0
    receipt = pipeline.read_json(flow.receipts/'cpu_probe.json')
    assert receipt['status'] == 'exited'
    assert receipt['exit_code'] == 0
    assert destination.read_text() == 'x'


def test_failed_shard_runner_retry_preserves_receipt_and_allows_retry_switch(tmp_path, monkeypatch):
    flow = pipeline.Pipeline(arguments(tmp_path, '--workers-per-gpu', '2', '--retry-failed'))
    command = [sys.executable, str(pipeline.ROOT/'scripts/run_sharded_manipulation_collection.py'),
               '--canonical', str(tmp_path/'raw')]
    path = flow.receipts/'cube_stack_collect.json'
    pipeline.write_json(path, dict(command=command, status='exited', exit_code=1))
    def launch(*args, **kwargs):
        row = pipeline.read_json(path)
        row.update(status='exited', exit_code=0)
        pipeline.write_json(path, row)
    monkeypatch.setattr(pipeline.subprocess, 'Popen', launch)
    assert flow.run_command('cube_stack_collect', command+['--retry-failed-workers']) == 0
    assert len(list(flow.receipts.glob('cube_stack_collect.*.json'))) == 1
    assert pipeline.read_json(path)['command'] == command+['--retry-failed-workers']


@pytest.mark.parametrize('flag', [
    '--episodes', '--workers-per-gpu', '--num-envs', '--cube-num-envs',
    '--cleanup-num-envs', '--max-attempts', '--poll-seconds', '--retry-seconds',
])
@pytest.mark.parametrize('value', ['0', '-1'])
def test_nonpositive_collection_settings_are_rejected(tmp_path, flag, value):
    with pytest.raises(SystemExit) as exc:
        arguments(tmp_path, '--prepare-only', flag, value)
    assert exc.value.code == 2


def test_historical_defaults_are_preserved(tmp_path):
    args = arguments(tmp_path)
    assert args.episodes == 500
    assert args.workers_per_gpu == 1
    assert args.dataset_suffix == '20261110'
    assert args.collection_config is None
    assert not args.prepare_only


@pytest.mark.parametrize('extra', [('--episodes', '100'), ('--dataset-suffix', 'fresh')])
def test_custom_dataset_requires_prepare_only_for_historical_sft_launcher(tmp_path, extra):
    with pytest.raises(SystemExit) as exc:
        arguments(tmp_path, *extra)
    assert exc.value.code == 2
    assert arguments(tmp_path, '--prepare-only', *extra).prepare_only


def test_prepare_100_episodes_exports_and_makes_examples_without_sft(tmp_path, monkeypatch):
    flow = pipeline.Pipeline(arguments(tmp_path, '--episodes', '100', '--prepare-only',
                                       '--dataset-suffix', 'fresh100'))
    exported = set()
    commands = {}
    gates = []

    def export_complete(path, task, count=500):
        gates.append((task, count))
        return task in exported

    def command(label, cmd):
        commands[label] = cmd
        task = next(t for t in pipeline.TASKS if label.startswith(t))
        if label.endswith('_export'):
            assert cmd[cmd.index('--episodes')+1] == '100'
            assert Path(cmd[cmd.index('--output')+1]).name == f'pressb_{task}_100_fresh100'
            exported.add(task)
        elif label.endswith('_examples'):
            output = Path(cmd[cmd.index('--output')+1])
            pipeline.write_json(output/'summary.json', {'task_id':task, 'episodes':100})
            (output/'examples_1x.mp4').write_bytes(b'fixture')
        else:
            pytest.fail(f'Unexpected command: {label}')
        return 0

    monkeypatch.setattr(flow, 'collect', lambda *args: None)
    monkeypatch.setattr(flow, 'run_command', command)
    monkeypatch.setattr(pipeline, 'export_complete', export_complete)
    monkeypatch.setattr(flow, 'deploy', lambda: pytest.fail('Prepare-only launched SFT'))
    flow.run()
    assert flow.state['status'] == 'completed'
    assert flow.state['stage'] == 'datasets_and_videos_ready'
    assert all(count == 100 for _, count in gates)
    for task in pipeline.TASKS:
        assert flow.state['tasks'][task]['accepted'] == 100
        assert f'{task}_export' in commands and f'{task}_examples' in commands


@pytest.mark.parametrize('task,gpu', [('cube_stack', '0'), ('tabletop_cleanup', '1')])
def test_two_workers_initialize_canonical_then_collect_on_task_gpu(tmp_path, monkeypatch, task, gpu):
    config = tmp_path/'fast.json'
    pipeline.write_json(config, {'color_response_workers':4})
    flow = pipeline.Pipeline(arguments(tmp_path, '--episodes', '100', '--prepare-only',
        '--workers-per-gpu', '2', '--num-envs', '8', '--cube-gpu', '0', '--cleanup-gpu', '1',
        '--collection-config', str(config)))
    raw = flow.output/f'{task}_raw'
    commands = []
    complete = False
    checked_counts = []

    def raw_complete(path, task_id, count=500):
        assert path == raw and task_id == task
        checked_counts.append(count)
        return complete

    def command(label, cmd):
        nonlocal complete
        commands.append(cmd)
        if any(part.endswith('collect_manipulation.py') for part in cmd):
            assert '--initialize-only' in cmd
            assert cmd[cmd.index('--task')+1] == task
            assert cmd[cmd.index('--episodes')+1] == '100'
            assert cmd[cmd.index('--num-envs')+1] == '8'
            assert cmd[cmd.index('--gpu')+1] == gpu
            assert Path(cmd[cmd.index('--config')+1]) == config
            pipeline.write_json(raw/'manifest.json', {
                'task_id':task, 'target_episodes':100, 'seed':flow.args.seed,
                'noise_deg':flow.args.noise_deg, 'trajectory_profile':'compact',
                'config':{'gpu':int(gpu), 'num_envs':8, 'color_response_workers':4},
            })
            pipeline.write_json(raw/'progress.json', {'status':'initialized', 'accepted':0, 'target':100})
        else:
            assert any(part.endswith('run_sharded_manipulation_collection.py') for part in cmd)
            assert Path(cmd[cmd.index('--canonical')+1]) == raw
            assert cmd[cmd.index('--gpus')+1] == gpu
            assert cmd[cmd.index('--workers-per-gpu')+1] == '2'
            assert cmd[cmd.index('--num-envs')+1] == '8'
            assert Path(cmd[cmd.index('--sim-python')+1]) == flow.args.sim_python
            assert Path(cmd[cmd.index('--utility-python')+1]) == flow.args.utility_python
            assert pipeline.read_json(raw/'manifest.json')['config']['color_response_workers'] == 4
            complete = True
        return 0

    monkeypatch.setattr(flow, 'run_command', command)
    monkeypatch.setattr(pipeline, 'raw_complete', raw_complete)
    monkeypatch.setattr(pipeline, 'external_collector', lambda *args: None)
    flow.collect(task, raw)
    assert len(commands) == 2 and complete
    assert checked_counts and all(count == 100 for count in checked_counts)


def test_existing_manifest_rejects_changed_collection_override(tmp_path):
    config = tmp_path/'fast.json'
    pipeline.write_json(config, {'color_response_workers':4})
    flow = pipeline.Pipeline(arguments(tmp_path, '--episodes', '100', '--prepare-only',
        '--num-envs', '8', '--collection-config', str(config)))
    raw = flow.output/'cube_stack_raw'
    pipeline.write_json(raw/'manifest.json', {
        'task_id':'cube_stack', 'target_episodes':100, 'seed':flow.args.seed,
        'noise_deg':flow.args.noise_deg, 'trajectory_profile':'compact',
        'config':{'gpu':flow.args.cube_gpu, 'num_envs':8, 'color_response_workers':2},
    })
    with pytest.raises(ValueError, match='differ'):
        flow.validate_existing(raw, 'cube_stack')
    manifest = pipeline.read_json(raw/'manifest.json')
    manifest['config']['color_response_workers'] = 4
    pipeline.write_json(raw/'manifest.json', manifest)
    flow.validate_existing(raw, 'cube_stack')


def test_initialized_canonical_resumes_without_bootstrap_or_retry_flag(tmp_path, monkeypatch):
    flow = pipeline.Pipeline(arguments(tmp_path, '--episodes', '100', '--prepare-only',
        '--workers-per-gpu', '2', '--num-envs', '8'))
    raw = flow.output/'cube_stack_raw'
    pipeline.write_json(raw/'manifest.json', {
        'task_id':'cube_stack', 'target_episodes':100, 'seed':flow.args.seed,
        'noise_deg':flow.args.noise_deg, 'trajectory_profile':'compact',
        'config':{'gpu':flow.args.cube_gpu, 'num_envs':8},
    })
    pipeline.write_json(raw/'progress.json', {'status':'initialized', 'accepted':0, 'target':100})
    complete = False

    def command(label, cmd):
        nonlocal complete
        assert any(part.endswith('run_sharded_manipulation_collection.py') for part in cmd)
        assert '--retry-failed-workers' not in cmd
        complete = True
        return 0

    monkeypatch.setattr(pipeline, 'raw_complete', lambda *args: complete)
    monkeypatch.setattr(flow, 'run_command', command)
    monkeypatch.setattr(pipeline, 'external_collector', lambda *args: None)
    flow.collect('cube_stack', raw)
    assert complete


def test_legacy_manifest_requires_explicit_legacy_profile(tmp_path):
    raw = tmp_path/'raw'
    flow = pipeline.Pipeline(arguments(tmp_path))
    pipeline.write_json(raw/'manifest.json', {
        'task_id':'cube_stack', 'target_episodes':500, 'seed':flow.args.seed,
        'noise_deg':flow.args.noise_deg,
        'config':{'gpu':flow.args.cube_gpu, 'num_envs':flow.args.cube_num_envs},
    })
    with pytest.raises(ValueError, match='differ'):
        flow.validate_existing(raw, 'cube_stack')
    legacy = pipeline.Pipeline(arguments(tmp_path, '--trajectory-profile', 'legacy'))
    legacy.validate_existing(raw, 'cube_stack')
