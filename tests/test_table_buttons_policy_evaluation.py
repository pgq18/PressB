"""Policy execution semantics without starting Isaac or a GPU model."""
import ast
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import threading
from urllib.error import HTTPError

import numpy as np
from PIL import Image
import pytest

from pressb.compat.table_buttons_policy import HTTPJointPolicy, png_base64
from pressb.tasks.table_buttons.policy_evaluation import (InvalidPolicyAction, JointPolicyCommands,
    command_limits, episode_config, load_cell, strict_outcome, summarize,
    to_policy_units, to_sim_units, validate_execution_arrays, collision_settings,
    is_allowed_finger_closure, DUAL_COLLISION_POLICY, STRICT_COLLISION_POLICY)
from pressb.tasks.table_buttons.task import COLORS, TableButtonOutcome


def test_joint_units_full_opening_and_inverse():
    physical = np.array([180, -90, 45, 0, 360, -180, 2.])
    expected = np.array([np.pi, -np.pi/2, np.pi/4, 0, 2*np.pi, -np.pi, .02])
    np.testing.assert_allclose(to_sim_units(physical), expected, atol=1e-15)
    np.testing.assert_allclose(to_policy_units(expected), physical, atol=1e-13)
    with pytest.raises(ValueError):
        to_sim_units([0] * 8)


def test_actual_finger_limits_and_model_gripper_is_not_fixed():
    limits = np.array([[-3, 3]] * 6 + [[0, .05], [-.05, 0]])
    lower, upper = command_limits(limits)
    assert lower[-1] == 0 and upper[-1] == .1
    controller = JointPolicyCommands(lower, upper, np.zeros(7))
    first, trace = controller.apply([0, 0, 0, 0, 0, 0, 4.])
    assert first[-1] == .04 and trace['clipped_dimensions'] == []
    negative, trace = controller.apply([400, 0, 0, 0, 0, 0, -1])
    assert negative[0] == 3 and negative[-1] == 0 and trace['clipped_dimensions'] == [0, 6]
    above, trace = controller.apply([0, 0, 0, 0, 0, 0, 12])
    assert above[-1] == .1 and trace['clipped_dimensions'] == [6]
    summary = controller.summary()
    assert summary['clipped_actions'] == 2 and summary['clipped_by_dimension'] == [1, 0, 0, 0, 0, 0, 2]
    assert summary['smoothing']['mode'] == 'none'
    with pytest.raises(ValueError):
        command_limits(np.zeros((8, 2)))


def test_optional_arm_smoothing_keeps_model_gripper():
    controller = JointPolicyCommands([-3] * 6 + [0], [3] * 6 + [.1], np.zeros(7), smoothing='joint_ema', alpha=.5)
    first, _ = controller.apply([90, 0, 0, 0, 0, 0, 2])
    second, _ = controller.apply([90, 0, 0, 0, 0, 0, 3])
    assert np.isclose(first[0], np.pi/4) and np.isclose(second[0], 3*np.pi/8)
    assert first[6] == .02 and second[6] == .03


def test_true_press_then_last_interval_wrong_contact_is_failure():
    checker = TableButtonOutcome('red')
    depth = dict.fromkeys(COLORS, 0.)
    force = dict.fromkeys(COLORS, 0.)
    depth['red'], force['red'] = .004, 1.
    for tick in range(1, 5):
        result = checker.update(tick / 120, depth, force)
    assert result['success']
    assert strict_outcome(result) == (True, 'success')
    # Success occurred within a control interval; a later tick remains decisive.
    force['blue'] = .021
    result = checker.update(5/120, depth, force)
    assert result['success']  # This is the task checker's latched state.
    assert strict_outcome(result, wrong_contact=True) == (False, 'non_target_finger_contact')
    assert strict_outcome(result, unexpected_contacts=[{'force_n': .03}]) == (False, 'unexpected_contact')
    assert strict_outcome(result, wrong_press={'color': 'blue'}) == (False, 'wrong_button')
    assert strict_outcome(None, model_error={'error': 'bad shape'}) == (False, 'invalid_policy_action')
    assert strict_outcome(None) == (False, 'timeout')


def test_no_success_without_force_and_dwell():
    checker = TableButtonOutcome('red')
    depth, force = dict.fromkeys(COLORS, 0.), dict.fromkeys(COLORS, 0.)
    depth['red'] = .004
    for tick in range(1, 20):
        result = checker.update(tick/120, depth, force)
    assert strict_outcome(result) == (False, 'timeout')
    force['red'] = 1.
    for tick in range(20, 23):
        result = checker.update(tick/120, depth, force)
    assert strict_outcome(result) == (False, 'timeout')
    result = checker.update(23/120, depth, force)
    assert strict_outcome(result) == (True, 'success')


def test_only_explicit_exact_finger_pair_is_allowed_for_task_metric():
    pair = ['/World/Piper/link7', '/World/Piper/link8']
    assert collision_settings() == STRICT_COLLISION_POLICY
    assert not is_allowed_finger_closure(pair, collision_settings())
    policy = collision_settings(DUAL_COLLISION_POLICY)
    assert is_allowed_finger_closure(pair, policy)
    assert is_allowed_finger_closure(pair[::-1], policy)
    for invalid in (['/World/Piper/link6', '/World/Piper/link7'],
                    ['/World/Piper/link7', '/World/Table/Body'],
                    ['/World/Piper/link7', '/World/Piper/link7'],
                    ['/World/Piper/link7/collision', '/World/Piper/link8'],
                    pair + ['/World/Table/Body']):
        assert not is_allowed_finger_closure(invalid, policy)
    malformed = dict(policy, allowed_task_contact_pairs=[pair, ['/World/Piper/link6', '/World/Piper/link7']])
    with pytest.raises(ValueError, match='exact finger-pair'):
        collision_settings(malformed)
    original = [dict(tick=1, actors=pair, force_n=.08)]
    blocking = [event for event in original if not is_allowed_finger_closure(event['actors'], policy)]
    target_result = dict(success=True)
    assert strict_outcome(target_result, unexpected_contacts=blocking) == (True, 'success')
    assert strict_outcome(target_result, unexpected_contacts=original) == (False, 'unexpected_contact')
    assert len(original) == 1  # Raw collision evidence is retained for the second metric.


def test_four_tick_zoh_alignment_and_zero_action_failure():
    command = np.array([.1, .2, .3, .4, .5, .6, .04])
    target = np.r_[command[:6], command[6]/2, -command[6]/2]
    physics = dict(time_s=np.arange(9)/120, q_actual=np.zeros((9, 8)),
                   q_target=np.repeat(target[None], 9, 0), travel_m=np.zeros((9, 3)), finger_force_n=np.zeros((9, 3)))
    frames = dict(sim_time=np.arange(3)/30, state=np.zeros((3, 7)), action=np.repeat(command[None], 3, 0))
    validate_execution_arrays(frames, physics)
    physics['q_target'][4, -1] = -.03
    with pytest.raises(ValueError, match='next control interval'):
        validate_execution_arrays(frames, physics)
    zero = {key: value[:1] for key, value in physics.items()}
    validate_execution_arrays({key: value[:1] for key, value in frames.items()}, zero)


def test_png_is_lossless_rgb():
    rgb = np.random.default_rng(3).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    decoded = Image.open(BytesIO(base64.b64decode(png_base64(rgb))))
    assert decoded.mode == 'RGB' and decoded.size == (640, 480)
    np.testing.assert_array_equal(np.asarray(decoded), rgb)
    with pytest.raises(ValueError):
        png_base64(rgb[:100])


@pytest.fixture
def server():
    captured = []
    model = dict(model_id='near_multitask', model_sha256='sha', best_step=123)
    control = {'mode': 'valid'}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            captured.append((self.path, body))
            code = 200
            if self.path == '/load':
                result = dict(model=model)
            elif self.path == '/reset':
                result = dict(ok=True)
            elif control['mode'] == 'typed_error':
                code, result = 422, dict(error_kind='invalid_policy_action', error='nonfinite model action')
            elif control['mode'] == 'transport_error':
                code, result = 503, dict(error='temporarily unavailable')
            else:
                sample = body['samples'][0]
                actions = np.tile([10., 20, 30, 40, 50, 60, .02], (7, 1))
                if control['mode'] == 'nonfinite': actions[0, 0] = float('nan')
                result = dict(model=model, inference_seconds=.01, predictions=[dict(
                    episode_id=sample['episode_id'], request_id=sample['request_id'], noise_seed=sample['noise_seed'],
                    actions=actions.tolist(), normalized_actions=(actions / np.array([180.]*6+[8.])).tolist(),
                    normalized_state=(np.array(sample['state']) / np.array([180.]*6+[8.])).tolist(),
                    initial_noise_sha256='noise', preprocessed_image_sha256={'up': 'up', 'wrist': 'wrist'})])
                if control['mode'] == 'wrong_identity': result['predictions'][0]['request_id'] += 1
            data = json.dumps(result).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    http = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{http.server_port}', control, captured
    http.shutdown()
    http.server_close()
    thread.join()


def test_http_identity_units_audit_evidence_and_typed_failures(server):
    endpoint, control, captured = server
    policy = HTTPJointPolicy(endpoint, 'near_multitask')
    policy.load()
    policy.reset('episode0')
    sample = dict(episode_id='episode0', request_id=0, instruction='Press the red button.',
                  state=[10, 20, 30, 40, 50, 60, .02], images={name: np.zeros((480, 640, 3), np.uint8)
                  for name in ('up', 'wrist')}, noise_seed=2000000)
    actions, trace = policy.predict(**sample)
    assert actions.shape == (7, 7) and actions[0, 6] == .02
    assert trace['initial_noise_sha256'] == 'noise' and trace['normalized_state'][0] == 10/180
    assert trace['preprocessed_image_sha256'] == {'up': 'up', 'wrist': 'wrist'}
    assert captured[-1][1]['samples'][0]['state'] == sample['state']
    assert set(captured[-1][1]['samples'][0]) == {'episode_id', 'request_id', 'instruction', 'state', 'images', 'noise_seed'}
    for mode in ('nonfinite', 'typed_error'):
        control['mode'] = mode
        with pytest.raises(InvalidPolicyAction):
            policy.predict(**sample)
    control['mode'] = 'wrong_identity'
    with pytest.raises(RuntimeError, match='identity'):
        policy.predict(**sample)
    control['mode'] = 'transport_error'
    with pytest.raises(HTTPError):
        policy.predict(**sample)


def test_plan_frozen_config_and_denominator(tmp_path):
    config = dict(position_sampling={}, buttons={color: {} for color in COLORS}, immutable_camera='same_as_training')
    layout = dict(positions={color: [.1, i*.1] for i, color in enumerate(COLORS)}, approaches={}, domain={})
    layout_file = tmp_path/'layout.json'
    layout_file.write_text(json.dumps(layout))
    effective = dict(config, position_sampling={'spacing_profile': 'near'}, task_color='red',
        task_instruction='Press the red button.', buttons={color: {'center_xy': layout['positions'][color]} for color in COLORS},
        press_approaches={}, sampling_domain={})
    config_file = tmp_path/'config.json'
    config_file.write_text(json.dumps(effective))
    row = dict(episode_index=0, seed=10000, policy_seed=2000000,
               layout_file=str(layout_file), effective_config_file=str(config_file))
    cell = dict(cell_id='near_multitask_red', model_id='near_multitask', spacing_profile='near',
                target_color='red', instruction='Press the red button.', episodes=[row])
    plan = dict(schema='pressb.table_buttons.policy_eval_plan.v1', config=config, cells=[cell],
                defaults=dict(fps=30, physics_dt=1/120, execute_steps=7))
    path = tmp_path/'plan.json'
    path.write_text(json.dumps(plan))
    loaded, selected = load_cell(path, cell['cell_id'])
    assert episode_config(loaded, selected, row)[0] == effective
    assert summarize(tmp_path, cell)['completed'] == 0
    episode = tmp_path/'episodes/episode_000000'
    episode.mkdir(parents=True)
    (episode/'report.json').write_text(json.dumps(dict(evaluation_status='complete', cell_id=cell['cell_id'],
        seed=10000, success=False, strict_collector_success=False, stop_reason='invalid_policy_action')))
    summary = summarize(tmp_path, cell)
    assert summary['complete'] and summary['failures'] == 1 and summary['success_rate'] == 0.
    assert summary['task_success_rate'] == summary['strict_collector_success_rate'] == 0.
    (episode/'report.json').write_text(json.dumps(dict(evaluation_status='complete', cell_id=cell['cell_id'],
        seed=10000, success=True, strict_collector_success=False, stop_reason='success')))
    summary = summarize(tmp_path, cell)
    assert summary['task_success_rate'] == 1. and summary['strict_collector_success_rate'] == 0.
    effective['immutable_camera'] = 'changed'
    config_file.write_text(json.dumps(effective))
    with pytest.raises(ValueError, match='frozen'):
        episode_config(loaded, selected, row)


def test_runner_has_no_expert_action_or_interpolation_calls():
    path = Path(__file__).resolve().parents[1]/'scripts/eval_table_buttons_policy.py'
    tree = ast.parse(path.read_text())
    names = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not any(name.rsplit('.', 1)[-1] in ('plan_press', 'prepare', 'interp', 'solve') for name in names)
    assert 'commands.apply' in names and 'policy.predict' in names
