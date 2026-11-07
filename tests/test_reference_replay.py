"""Original-client execution-queue parity; no policy preprocessing or hardware.

The original control_loop is compiled unchanged from its AST, with smoothing
disabled and deterministic observations/actions. SDK velocity projection, lost
feedback fallback and idle lifecycle are deliberately outside this comparison.
"""
import ast
import collections
import dataclasses
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import ModuleType, SimpleNamespace
from typing import Any, Optional

import numpy as np
import pytest

import pressb.control.real_replay as replay_module
from pressb.control.real_replay import ReplayError, ReplayPlan, execute_replay
from test_real_replay import Clock, Robot, plan
from test_grasp_replay import grasp_plan


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = Path(os.environ.get("PRESSB_REFERENCE_PIPER_CLIENT", str(
    ROOT.parents[1] / "Work/lerobot-piper/piper_scripts/piper_inference/piper_client.py")))
KEYS = [f"joint_{i}.pos" for i in range(1, 7)] + ["gripper.pos"]
TO_LOCAL = np.r_[np.full(6, 180 / np.pi), 100.0]


def load_original(clock):
    if not ORIGINAL.exists():
        pytest.skip("Original lerobot-piper source is unavailable; core tests still run")
    wanted = {"_ControlFlags", "FLAGS", "PIPER_JOINT_LIMITS", "DEFAULT_MAX_DELTA",
              "EXTERNAL_BUTTON_RETURN_POSE", "UNIT_CONVERSION", "piper_idle_action", "ROBOT_CONFIGS",
              "_clamp_action", "_limit_delta", "_get_conversion_factor", "_convert_action_units",
              "_linear_interp_prefix_blend", "_mean_filter_chunk", "ControlConfig", "control_loop"}
    selected = []
    for node in ast.parse(ORIGINAL.read_text()).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted:
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in wanted for t in node.targets):
            selected.append(node)
    module = ModuleType("pressb_original_client_queue_test")
    # dataclasses resolves the class module while creating its field metadata.
    sys.modules[module.__name__] = module
    module.__dict__.update(np=np, collections=collections, dataclass=dataclasses.dataclass,
        Any=Any, Optional=Optional, Robot=object, RobotConfig=object,
        logger=logging.getLogger(module.__name__),
        time=SimpleNamespace(perf_counter=clock.monotonic, sleep=clock.sleep),
        precise_sleep=clock.sleep, keyboard_listener=lambda: None,
        threading=SimpleNamespace(Thread=lambda **kwargs: SimpleNamespace(start=lambda: None)))
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(ORIGINAL), "exec"), module.__dict__)
    return module


class ReferenceRobot(Robot):
    def __init__(self, config, state, clock):
        super().__init__(config, state, clock)
        self.state_at = lambda now: self.state.copy()
        self.read_fault = None
        self.on_command = None

    def read_state(self):
        if self.read_fault and self.commands:
            raise RuntimeError(self.read_fault)
        return np.asarray(self.state_at(self.clock.monotonic())).copy()

    def command(self, target):
        target = np.asarray(target).copy()
        self.commands.append((self.clock.monotonic(), target))
        self.clock.sleep(self.command_duration)
        if self.on_command:
            self.on_command()
        return target


def reference_plan(value):
    value["replay"].update(execution_mode="reference", max_wall_time_s=40)
    return value


def run_reference(value, path, configure=lambda robot: None, **kwargs):
    clock = Clock()
    robot = ReferenceRobot(value["robot"], value["expected_initial_state"], clock)
    configure(robot)
    result = execute_replay(value, lambda _: robot, path, monotonic=clock.monotonic,
                            sleep=clock.sleep, wall_time=clock.monotonic, **kwargs)
    return result, robot, clock


class EndOfTrace(BaseException):
    """Stop the offline original loop at its next inference request."""


def original_trace(actions, state_at, *, chunk_size, command_duration=0):
    clock = Clock()
    original = load_original(clock)
    original.FLAGS.running = True
    chunks = [actions[start:start + chunk_size] for start in range(0, len(actions), chunk_size)]
    commands, infer_times = [], []
    class OriginalRobot:
        config = object()
        def get_observation(self):
            values = state_at(clock.monotonic()) * TO_LOCAL
            return {**dict(zip(KEYS, values)), "up": np.zeros((1, 1, 3), np.uint8),
                    "wrist": np.zeros((1, 1, 3), np.uint8)}
        def send_action(self, action):
            commands.append((clock.monotonic(), np.array([action[key] for key in KEYS]) / TO_LOCAL))
            clock.sleep(command_duration)
    class Policy:
        def infer(self, element):
            infer_times.append(clock.monotonic())
            if not chunks:
                raise EndOfTrace
            return {"action": [chunks.pop(0) * TO_LOCAL]}
    config = original.ControlConfig(robot=object(), fps=30, action_chunk_size=chunk_size,
        enable_action_smoothing=False, record_pose=False, info_interval=0)
    with pytest.raises(EndOfTrace):
        original.control_loop(OriginalRobot(), Policy(), 30, cfg=config, settle_time=.5)
    return commands, infer_times


@pytest.mark.parametrize("chunk_size", [None, 2])
@pytest.mark.parametrize("command_duration", [0, .005])
def test_original_queue_order_clamp_fresh_delta_and_settling_parity(plan, tmp_path, chunk_size, command_duration):
    reference_plan(plan)
    if chunk_size is not None:
        plan["replay"]["reference_chunk_size"] = chunk_size
    initial = np.asarray(plan["expected_initial_state"])
    actions = np.tile(initial, (4, 1))
    actions[:, 0] = [10., -10., 2., -2.]
    actions[:, 6] = [.2, -.1, .07, .0]
    # Original casts chunks to float32 before their execution queue. Prepare
    # the same queue values here; model preprocessing itself is not the core.
    actions = (actions * TO_LOCAL).astype(np.float32).astype(float) / TO_LOCAL
    plan["actions"] = actions.tolist()
    def state_at(now):
        state = initial.copy()
        state[0] += .01 * now
        state[6] += .001 * now
        return state
    def configure(robot):
        robot.state_at = state_at
        robot.command_duration = command_duration
    result, robot, _ = run_reference(plan, tmp_path, configure)
    expected, infer_times = original_trace(actions, state_at, chunk_size=chunk_size or len(actions),
                                           command_duration=command_duration)
    np.testing.assert_allclose([t for t, _ in robot.commands], [t for t, _ in expected], atol=1e-10)
    np.testing.assert_allclose([a for _, a in robot.commands], [a for _, a in expected], atol=2e-7)
    assert result["dispatch_complete"]
    assert len(infer_times) == math.ceil(len(actions) / (chunk_size or len(actions))) + 1


def test_reference_does_not_wait_five_seconds_for_soft_lag_or_loaded_gripper(plan, tmp_path):
    reference_plan(plan)
    initial = list(plan["expected_initial_state"])
    initial[6] = .0351
    plan["expected_initial_state"] = initial
    plan["actions"] = [[.08] + initial[1:6] + [.028166667] for _ in range(180)]
    result, robot, clock = run_reference(plan, tmp_path)
    assert result["dispatch_complete"]
    assert result["target_reached"] is None and result["task_success"] is None
    assert not robot.stopped and robot.closed
    assert 6 <= clock.now < 7
    assert len(robot.commands) >= 180
    # State is deliberately fixed: dispatch is not a successful physical task.
    assert robot.state[0] == 0 and robot.state[6] == .0351


def test_reference_does_not_construct_grasp_monitor_or_require_grasp_api(grasp_plan, tmp_path, monkeypatch):
    reference_plan(grasp_plan)
    monkeypatch.setattr(replay_module, "GraspMonitor", lambda *a, **kw: pytest.fail("no contact phase gate"))
    result, robot, _ = run_reference(grasp_plan, tmp_path)
    assert result["dispatch_complete"] and not hasattr(robot, "hold_grasp")
    assert result["task_success"] is None


@pytest.mark.parametrize("fault", ["stale feedback", "motor fault"])
def test_reference_feedback_failure_still_stops(plan, tmp_path, fault):
    reference_plan(plan)
    robots = []
    def configure(robot):
        robot.read_fault = fault
        robots.append(robot)
    with pytest.raises(ReplayError, match=fault):
        run_reference(plan, tmp_path, configure)
    assert robots[0].stopped and robots[0].closed
    result = json.loads((tmp_path / "summary.json").read_text())
    assert not result["dispatch_complete"]
    assert result["failure_kind"] == "hardware_or_control" and result["emergency_stop_requested"]


@pytest.mark.parametrize("method", ["event", "stop_file", "watchdog"])
def test_reference_operator_stop_and_watchdog_still_stop(plan, tmp_path, method):
    reference_plan(plan)
    event = threading.Event()
    stop = tmp_path / "STOP"
    robots = []
    def configure(robot):
        if method == "watchdog":
            robot.command_duration = .6
        else:
            robot.on_command = event.set if method == "event" else stop.touch
        robots.append(robot)
    with pytest.raises(ReplayError):
        run_reference(plan, tmp_path, configure, cancel=event, stop_file=stop)
    assert robots[0].stopped and robots[0].closed
    result = json.loads((tmp_path / "summary.json").read_text())
    assert result["failure_kind"] == "hardware_or_control" and result["emergency_stop_requested"]


@pytest.mark.parametrize("fault", ["hold", "close", "cleanup_watchdog"])
def test_reference_cleanup_fault_requests_stop_and_reports_hardware_failure(plan, tmp_path, fault):
    reference_plan(plan)
    robots = []
    def configure(robot):
        if fault == "hold":
            def hold():
                raise RuntimeError("fault during hold")
            robot.hold = hold
        elif fault == "close":
            robot.close_error = True
        else:
            original = robot.close
            def close():
                original()
                robot.clock.sleep(6)
            robot.close = close
        robots.append(robot)
    with pytest.raises(ReplayError):
        run_reference(plan, tmp_path, configure)
    result = json.loads((tmp_path / "summary.json").read_text())
    assert not result["success"] and robots[0].stopped and robots[0].closed
    assert result["failure_kind"] == "hardware_or_control" and result["emergency_stop_requested"]


def test_reference_settling_still_honors_operator_stop(plan, tmp_path):
    reference_plan(plan)
    event = threading.Event()
    robots = []
    def configure(robot):
        def stop_during_settle():
            if len(robot.commands) >= len(plan["actions"]) + 3:
                event.set()
        robot.on_command = stop_during_settle
        robots.append(robot)
    with pytest.raises(ReplayError):
        run_reference(plan, tmp_path, configure, cancel=event)
    assert robots[0].stopped and not robots[0].held
    assert not json.loads((tmp_path / "summary.json").read_text())["dispatch_complete"]


def test_reference_revalidation_keeps_mode_and_chunk_size(plan, tmp_path):
    reference_plan(plan)
    plan["replay"]["reference_chunk_size"] = 1
    validated = ReplayPlan.from_mapping(plan)
    clock = Clock()
    robot = ReferenceRobot(plan["robot"], plan["expected_initial_state"], clock)
    result = execute_replay(validated, lambda _: robot, tmp_path, monotonic=clock.monotonic,
                            sleep=clock.sleep, wall_time=clock.monotonic)
    assert result["execution_mode"] == "reference" and result["dispatch_complete"]
    assert clock.now >= 1  # Both one-action chunks include their fixed settle window.


def test_reference_does_not_silently_apply_native_recovery(plan, tmp_path):
    reference_plan(plan)
    plan["native_startup_recovery"] = {"verified_max_joint_speed_rad_s": [3.] * 6}
    with pytest.raises(ValueError):
        execute_replay(plan, lambda _: pytest.fail("reference cannot add native recovery"), tmp_path)


def test_original_unit_helpers_and_limits_match_si_contract():
    original = load_original(Clock())
    raw = dict(zip(KEYS, [.3, .1, -.2, .4, .5, -.6, 35.1]))
    converted = original._convert_action_units(raw, "radian", "mm")
    np.testing.assert_allclose([converted[k] for k in KEYS], np.r_[np.rad2deg(list(raw.values())[:6]), 3.51])
    np.testing.assert_allclose([original.DEFAULT_MAX_DELTA[k] for k in KEYS] / TO_LOCAL,
                              np.r_[np.full(6, np.pi / 4), .03])


@pytest.mark.parametrize("size", [0, -1, True, 1.5, "2"])
def test_reference_chunk_size_invalid_before_factory(plan, tmp_path, size):
    reference_plan(plan)
    plan["replay"]["reference_chunk_size"] = size
    with pytest.raises((ValueError, TypeError)):
        execute_replay(plan, lambda _: pytest.fail("invalid chunk size must not connect"), tmp_path)


def test_tracked_default_and_bounds_validation_remain(plan):
    assert ReplayPlan.from_mapping(plan).options.execution_mode == "tracked"
    plan["actions"][-1][0] = 10
    with pytest.raises(ValueError, match="bounds"):
        ReplayPlan.from_mapping(plan)


def cli_summary(plan, path, override=None):
    path.write_text(json.dumps(plan))
    args = ["--plan", str(path)] + ([] if override is None else ["--execution-mode", override])
    code = f'''import importlib.util, sys
spec = importlib.util.spec_from_file_location("reference_cli", {str(ROOT / "scripts/replay_real_robot.py")!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module.main({args!r}) == 0
assert "pressb.robots.piper.real" not in sys.modules
assert "piper_sdk" not in sys.modules
'''
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


@pytest.mark.parametrize("explicit,override,expected", [
    (None, None, "reference"), ("tracked", None, "tracked"),
    ("reference", "tracked", "tracked"), ("tracked", "reference", "reference"),
])
def test_cli_reference_default_plan_precedence_and_override_without_hardware(plan, tmp_path, explicit, override, expected):
    if explicit is not None:
        plan["replay"]["execution_mode"] = explicit
    result = cli_summary(plan, tmp_path / "plan.json", override)
    assert result["execution_mode"] == expected and not result["executed"]
