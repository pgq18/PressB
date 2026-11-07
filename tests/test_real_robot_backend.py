"""Real environment/runtime integration using explicitly fake hardware and labels.

No CAN interface, camera SDK, physical robot, model checkpoint or GPU is used.
The reset/step service, HTTP transport, inference service and CPU SAC are real.
"""
from contextlib import contextmanager
from copy import deepcopy
import json
import threading

import numpy as np
import pytest

from pressb.runtime.client import NodeClient
from pressb.runtime.contract import decode_image, negotiate
from pressb.runtime.inference import InferenceService
from pressb.runtime.real_robot import RealRobotBackend
from pressb.runtime.rpc import RPCServer
from pressb.runtime.simulation import SimulationService


class FakeClock:
    def __init__(self):
        self.now = 100.
        self.periods = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.periods.append(seconds)
        self.now += seconds


class FakeRobot:
    """Measured feedback lags a rate-limited, quantized fake controller."""
    def __init__(self, *, actuation_enabled=True, max_delta=.05):
        self.actuation_enabled = actuation_enabled
        self.max_delta = max_delta
        self.state = np.array([0.] * 6 + [.02])
        self.last_sent = self.state.copy()
        self.connected = self.closed = False
        self.requests, self.sent, self.holds = [], [], []
        self.stop_event = threading.Event()
        self.read_error = self.command_error = None
        self.invalid_state = False

    def describe(self):
        state = dict(names=[f"joint{i}" for i in range(1, 7)] + ["gripper_width"],
                     units=["rad"] * 6 + ["m"])
        return dict(robot_type="explicit_fake_piper", actuation_enabled=self.actuation_enabled,
                    state=state, action={**deepcopy(state), "representation": "joint_position",
                        "frame": "robot_base", "control_hz": 30.,
                        "lower": [-2.] * 6 + [0.], "upper": [2.] * 6 + [.07]})

    def connect(self):
        self.connected = True

    def read_state(self):
        if self.read_error:
            raise self.read_error
        assert self.connected
        return np.full(7, np.nan) if self.invalid_state else self.state.copy()

    def command(self, target):
        if self.command_error:
            raise self.command_error
        assert self.connected and not self.stop_event.is_set()
        target = np.asarray(target, dtype=float)
        self.requests.append(target.copy())
        self.last_sent = np.round(self.last_sent + np.clip(target - self.last_sent,
                                                         -self.max_delta, self.max_delta), 6)
        self.sent.append(self.last_sent.copy())
        self.state = self.last_sent.copy()
        self.state[:6] *= .8
        return self.last_sent.copy()

    def hold(self):
        assert self.connected
        self.last_sent = self.read_state()
        self.holds.append(self.last_sent.copy())
        return self.last_sent.copy()

    def stop(self):
        self.stop_event.set()

    def status(self):
        return dict(kind="explicit_fake_hardware", connected=self.connected,
                    stopped=self.stop_event.is_set())

    def close(self):
        self.closed = True


class FakeCameras:
    def __init__(self):
        self.config = {"up": "explicit-fake-up", "wrist": "explicit-fake-wrist"}
        self.metadata = {}
        self.connected = self.closed = False
        self.reads = 0
        self.error = None
        self.missing = self.invalid = False
        self.block_started = threading.Event()
        self.unblock = threading.Event()
        self.block = False

    def connect(self):
        self.connected = True

    def read(self):
        assert self.connected
        if self.block:
            self.block_started.set()
            if not self.unblock.wait(2):
                raise TimeoutError("fake camera was not released")
        if self.error:
            raise self.error
        self.reads += 1
        self.metadata = {name: {"source": "explicit_fake_camera", "frame_number": self.reads}
                         for name in self.config}
        dtype = np.float32 if self.invalid else np.uint8
        result = {name: np.full((4, 6, 3), self.reads, dtype=dtype) for name in self.config}
        if self.missing:
            result.pop("wrist")
        return result

    def close(self):
        self.closed = True
        self.unblock.set()


class FakeTask:
    """Explicit synthetic labels; these are never physical success evidence."""
    def __init__(self, robot, *, terminal_after=None):
        self.robot, self.terminal_after = robot, terminal_after
        self.resets, self.ends = [], []
        self.result = {}
        self.error = None
        self.closed = False
        self.start_commands = 0

    def describe(self):
        return dict(kind="explicit_fake_labels", reset="immediate_test_fixture",
                    outcomes="synthetic_test_signal")

    def reset(self, seed, options):
        self.resets.append((seed, deepcopy(options)))
        self.start_commands = len(self.robot.sent)
        self.result = {}

    def poll(self):
        if self.error:
            raise self.error
        if self.terminal_after is not None and len(self.robot.sent) - self.start_commands >= self.terminal_after:
            return dict(terminated=True, success=True, reward=1., reason="synthetic_test_success")
        return deepcopy(self.result)

    def end(self, reason):
        self.ends.append(reason)

    def status(self):
        return dict(kind="explicit_fake_labels", resets=len(self.resets), ends=len(self.ends))

    def close(self):
        self.closed = True


@contextmanager
def environment(*, task_ticks=None, robot=None, watchdog=False, **options):
    clock = FakeClock()
    robot = robot or FakeRobot()
    cameras = FakeCameras()
    task = FakeTask(robot, terminal_after=task_ticks)
    config = dict(instruction="Perform the explicitly synthetic test task.",
                  task_id="fake-hardware-only", horizon=4, control_hz=30., gamma=.9,
                  discount_timebase_hz=30., max_episode_steps=50, max_episode_seconds=120,
                  watchdog_seconds=10.)
    config.update(options)
    backend = RealRobotBackend(config, robot=robot, cameras=cameras, task=task,
                               clock=clock, sleep=clock.sleep, _start_watchdog=watchdog)
    try:
        yield backend, robot, cameras, task, clock
    finally:
        backend.close()


def actions(count=1, joint=.6, gripper=.03):
    return [[joint] * 6 + [gripper] for _ in range(count)]


def reset(service, *, request="reset", run="run", seed=7):
    return service.reset(dict(run_id=run, request_id=request, env_ids=[0], seed=seed))["items"][0]


def step_request(slot, *, request="step", run="run", count=1):
    return dict(run_id=run, request_id=request, items=[{
        "env_id": 0, "episode_id": slot["episode_id"], "step_id": slot["step_id"],
        "actions": actions(count),
    }])


def inference_metadata(backend):
    sim = backend.describe()
    return dict(schema_version=2, node_type="inference", model_id="explicit-fake-policy",
                observation=deepcopy(sim["observation"]), action=deepcopy(sim["action"]),
                noise=dict(shape=[sim["action"]["horizon"], 9]), feature_dim=3,
                capabilities=["encode", "decode", "features", "context_cache", "external_initial_noise"])


@pytest.mark.parametrize("method", ["base", "dsrl", "action_residual"])
def test_joint7_schema_negotiation_and_fresh_reset_observation(method):
    with environment() as (backend, robot, cameras, task, _):
        service = SimulationService(backend)
        contract = negotiate(inference_metadata(backend), service.health(), method, execution_horizon=2)
        assert contract["num_envs"] == 1
        assert (contract["action_dim"], contract["state_dim"], contract["control_state_dim"]) == (7, 7, 7)
        assert contract["noise_shape"] == [4, 9]
        assert contract["obs_dim"] == 17
        assert service.health()["environment_type"] == "real_robot"
        assert service.health()["reward_schema"]["kind"] == "explicit_fake_labels"
        assert not robot.connected and not cameras.connected
        row = reset(service)
        assert robot.connected and cameras.connected
        assert task.resets == [(7, {})]
        assert not robot.sent and len(robot.holds) == 1
        np.testing.assert_allclose(row["observation"]["state"], row["observation"]["control_state"])
        for name in cameras.config:
            image = decode_image(row["observation"]["images"][name], require_rgb=True)
            np.testing.assert_array_equal(np.asarray(image), np.full((4, 6, 3), 1, dtype=np.uint8))
        with pytest.raises(ValueError, match="terminal_state_only"):
            negotiate(inference_metadata(backend), service.health(), method,
                      terminal_observation_mode="state_only_if_no_bootstrap")


def test_applied_command_control_state_and_feedback_are_distinct():
    with environment() as (backend, robot, cameras, _, clock):
        service = SimulationService(backend)
        slot = reset(service)
        row = service.step(step_request(slot, count=2))["items"][0]
        assert not row["terminated"] and not row["truncated"]
        np.testing.assert_allclose(row["info"]["applied_actions"], robot.sent)
        np.testing.assert_allclose(row["observation"]["control_state"], robot.sent[-1])
        np.testing.assert_allclose(row["observation"]["state"], robot.state)
        assert row["observation"]["state"] != row["observation"]["control_state"]
        assert robot.requests[-1].tolist() != robot.sent[-1].tolist()
        assert row["info"]["elapsed_control_steps"] == row["info"]["physics_steps"] == 2
        assert row["discount"] == pytest.approx(.9 ** 2)
        assert cameras.reads == 2
        assert row["info"]["camera_metadata"]["up"]["frame_number"] == 2
        assert clock.periods == pytest.approx([1 / 30, 1 / 30])


def test_discount_uses_executed_ticks_and_timebase_excluding_inference_wait():
    with environment(discount_timebase_hz=15.) as (backend, _, _, _, clock):
        service = SimulationService(backend)
        slot = reset(service)
        clock.now += 5.  # Synthetic policy inference delay does not execute a control tick.
        row = service.step(step_request(slot, count=2))["items"][0]
        assert row["discount"] == pytest.approx(.9 ** (2 * 15 / 30))
        assert row["info"]["hardware_control_steps"] == 2
        assert row["info"]["wall_seconds"] == pytest.approx(2 / 30)


def test_smoothing_uses_previously_applied_controller_setpoint_across_chunks():
    with environment(smoothing={"mode": "joint_ema", "alpha": .5}) as (backend, robot, _, _, _):
        backend.reset([0], 1)
        backend.step([dict(env_id=0, actions=actions(joint=1.))])
        prior = robot.sent[-1].copy()
        backend.step([dict(env_id=0, actions=actions(joint=1.))])
        np.testing.assert_allclose(robot.requests[-1][:6], .5 + .5 * prior[:6])
        assert robot.requests[-1][6] == pytest.approx(.03)


def test_service_retries_and_stale_episode_guards_never_repeat_hardware():
    with environment() as (backend, robot, _, task, _):
        service = SimulationService(backend)
        slot = reset(service)
        assert reset(service) == slot
        assert len(task.resets) == 1
        payload = step_request(slot)
        result = service.step(payload)
        assert service.step(deepcopy(payload)) == result
        assert len(robot.sent) == 1
        with pytest.raises(ValueError, match="reused"):
            changed = deepcopy(payload)
            changed["items"][0]["actions"] = actions(joint=.4)
            service.step(changed)
        with pytest.raises(ValueError, match="Stale"):
            service.step(step_request(slot, request="stale"))
        with pytest.raises(ValueError, match="another run"):
            service.step(step_request(result["items"][0], request="foreign", run="foreign"))
        with pytest.raises(ValueError, match="active episode"):
            reset(service, request="active-reset")
        assert len(robot.sent) == 1 and len(task.resets) == 1


def test_terminal_signal_within_chunk_stops_remainder_and_preserves_terminal_images():
    with environment(task_ticks=2) as (backend, robot, cameras, task, _):
        service = SimulationService(backend)
        row = service.step(step_request(reset(service), count=4))["items"][0]
        assert row["terminated"] and not row["truncated"] and row["info"]["success"]
        assert len(robot.sent) == row["info"]["hardware_control_steps"] == 2
        assert row["reward"] == pytest.approx(.9)
        assert row["discount"] == pytest.approx(.9 ** 2)
        assert set(row["observation"]["images"]) == set(cameras.config)
        assert task.ends == ["synthetic_test_success"]
        with pytest.raises(ValueError, match="terminal"):
            service.step(step_request(row, request="after-terminal"))
        next_slot = reset(service, request="next-episode")
        assert next_slot["episode_id"] != row["episode_id"]
        assert len(task.resets) == 2 and len(robot.sent) == 2


@pytest.mark.parametrize("limits", [{"max_episode_steps": 2}, {"max_episode_seconds": .05}])
def test_time_limit_truncates_with_actual_tick_discount(limits):
    with environment(**limits) as (backend, robot, _, task, _):
        service = SimulationService(backend)
        row = service.step(step_request(reset(service), count=4))["items"][0]
        assert row["truncated"] and not row["terminated"] and not row["info"]["success"]
        assert len(robot.sent) == row["info"]["elapsed_control_steps"] == 2
        assert row["discount"] == pytest.approx(.9 ** 2)
        assert row["reward"] == 0 and task.ends == ["time_limit"]


def test_preaction_operator_terminal_uses_measured_hold_without_policy_motion():
    with environment() as (backend, robot, _, task, _):
        service = SimulationService(backend)
        slot = reset(service)
        state = robot.state.copy()
        task.result = dict(terminated=True, success=True, reward=2., reason="synthetic_operator_success")
        row = service.step(step_request(slot, count=4))["items"][0]
        assert not robot.sent
        np.testing.assert_allclose(row["info"]["applied_actions"], [state])
        np.testing.assert_allclose(row["observation"]["control_state"], state)
        assert row["info"]["hardware_control_steps"] == 1
        assert row["terminated"] and row["reward"] == 2.
        assert row["discount"] == pytest.approx(.9)


def test_nonterminal_reward_polled_before_action_is_preserved_once(monkeypatch):
    with environment() as (backend, robot, _, task, _):
        service = SimulationService(backend)
        slot = reset(service)
        events = iter([{"reward": 2.}, {}, {}, {}])
        monkeypatch.setattr(task, "poll", lambda: next(events))
        first = service.step(step_request(slot))["items"][0]
        assert first["reward"] == 2.
        assert not first["terminated"] and not first["truncated"]
        second = service.step(step_request(first, request="next-step"))["items"][0]
        assert second["reward"] == 0.
        assert len(robot.sent) == 2


@pytest.mark.parametrize("outcome", ["success", "timeout"])
def test_terminal_observation_is_recaptured_with_actual_measured_hold_command(outcome):
    options = {"task_ticks": 1} if outcome == "success" else {"max_episode_steps": 1}
    with environment(**options) as (backend, robot, cameras, _, _):
        service = SimulationService(backend)
        row = service.step(step_request(reset(service), count=4))["items"][0]
        assert row["terminated"] is (outcome == "success")
        assert row["truncated"] is (outcome == "timeout")
        assert len(robot.sent) == 1 and len(robot.holds) == 2
        held = robot.holds[-1]
        assert not np.allclose(held, robot.sent[-1]), "fake feedback must lag the policy setpoint"
        np.testing.assert_allclose(row["observation"]["control_state"], held)
        np.testing.assert_allclose(row["observation"]["state"], held)
        np.testing.assert_allclose(row["info"]["applied_actions"], robot.sent)
        # Reset, chunk boundary and the post-hold terminal observation each
        # acquire fresh frames. The returned pixels match the final metadata.
        assert cameras.reads == 3
        assert row["info"]["camera_metadata"]["up"]["frame_number"] == 3
        pixels = np.asarray(decode_image(row["observation"]["images"]["up"]))
        np.testing.assert_array_equal(pixels, np.full((4, 6, 3), 3, dtype=np.uint8))


@pytest.mark.parametrize("fault", ["camera", "missing_view", "bad_pixels", "state", "nonfinite_state", "abort", "command"])
def test_infrastructure_fault_latches_stop_and_prevents_further_motion(fault):
    with environment() as (backend, robot, cameras, task, _):
        service = SimulationService(backend)
        slot = reset(service)
        if fault == "camera":
            cameras.error = TimeoutError("synthetic camera timeout")
        elif fault == "missing_view":
            cameras.missing = True
        elif fault == "bad_pixels":
            cameras.invalid = True
        elif fault == "state":
            robot.read_error = RuntimeError("synthetic stale feedback")
        elif fault == "nonfinite_state":
            robot.invalid_state = True
        elif fault == "command":
            robot.command_error = RuntimeError("synthetic CAN write failure")
        else:
            task.error = RuntimeError("synthetic operator abort")
        with pytest.raises((RuntimeError, ValueError, TimeoutError)):
            service.step(step_request(slot))
        assert robot.stop_event.is_set()
        assert backend.status()["fault"] and not backend.status()["active"]
        count = len(robot.sent)
        with pytest.raises(RuntimeError, match="restart"):
            backend.step([dict(env_id=0, actions=actions())])
        with pytest.raises(RuntimeError, match="restart"):
            backend.reset([0], 2)
        with pytest.raises(RuntimeError, match="restart"):
            service.step(step_request(slot, request="after-fault"))
        assert len(robot.sent) == count
        if fault == "abort":
            assert not robot.sent


def test_inactivity_watchdog_stops_robot_without_another_rpc():
    with environment(watchdog=True, watchdog_seconds=.04) as (backend, robot, _, _, clock):
        backend.reset([0], 1)
        clock.now += .1
        assert robot.stop_event.wait(1), "watchdog did not stop fake hardware"
        assert "watchdog" in backend.status()["fault"].lower()
        assert not backend.status()["active"]
        with pytest.raises(RuntimeError, match="restart"):
            backend.step([dict(env_id=0, actions=actions())])
        assert not robot.sent


def test_watchdog_can_stop_while_camera_observation_is_blocked():
    with environment(watchdog=True, watchdog_seconds=.1) as (backend, robot, cameras, _, clock):
        backend.reset([0], 1)
        cameras.block = True
        errors = []
        def run_step():
            try:
                backend.step([dict(env_id=0, actions=actions())])
            except BaseException as error:
                errors.append(error)
        worker = threading.Thread(target=run_step)
        worker.start()
        try:
            assert cameras.block_started.wait(1)
            clock.now += .2
            assert robot.stop_event.wait(1), "camera read prevented the hardware watchdog"
        finally:
            cameras.unblock.set()
            worker.join(timeout=2)
        assert not worker.is_alive()
        assert len(errors) == 1 and "faulted" in str(errors[0])


def test_watchdog_faults_during_blocked_command_and_rejects_its_late_success(monkeypatch):
    with environment(watchdog=True, watchdog_seconds=.04) as (backend, robot, _, _, clock):
        service = SimulationService(backend)
        slot = reset(service)
        started, release = threading.Event(), threading.Event()
        responses, errors = [], []

        def delayed_command(target):
            # Model a driver call already in flight that returns a valid
            # setpoint after stop was requested. The runtime must reject it.
            started.set()
            if not release.wait(2):
                raise TimeoutError("synthetic blocked command was not released")
            robot.last_sent = np.asarray(target).copy()
            robot.sent.append(robot.last_sent.copy())
            return robot.last_sent.copy()

        def run_step():
            try:
                responses.append(service.step(step_request(slot)))
            except BaseException as error:
                errors.append(error)

        monkeypatch.setattr(robot, "command", delayed_command)
        worker = threading.Thread(target=run_step)
        worker.start()
        try:
            assert started.wait(1)
            lock_available = backend._lock.acquire(blocking=False)
            if lock_available:
                backend._lock.release()
            assert not lock_available, "test command must block while the backend operation lock is held"
            clock.now += .1
            assert robot.stop_event.wait(1), "watchdog waited for the blocked command lock"
            assert worker.is_alive()
            assert "watchdog" in backend.status()["fault"].lower()
            assert not backend.status()["active"]
        finally:
            release.set()
            worker.join(timeout=2)
        assert not worker.is_alive()
        assert not responses and len(errors) == 1
        assert "watchdog" in str(errors[0]).lower()
        assert "watchdog" in backend.status()["fault"].lower()
        assert backend.status()["executed_control_steps"] == 0
        assert service.fault and service.slots[0]["step_id"] == 0
        with pytest.raises(RuntimeError, match="restart"):
            service.step(step_request(slot, request="after-late-command"))


def test_failed_camera_cleanup_is_reported_and_can_be_retried(monkeypatch):
    with environment() as (backend, robot, cameras, task, _):
        backend.reset([0], 1)
        original_close = cameras.close
        attempts = []

        def fail_first_close():
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("synthetic camera stop still pending")
            original_close()

        monkeypatch.setattr(cameras, "close", fail_first_close)
        with pytest.raises(RuntimeError, match="cleanup incomplete.*camera stop still pending"):
            backend.close()
        assert robot.closed and task.closed and not cameras.closed
        assert "camera stop still pending" in backend.status()["fault"]
        with pytest.raises(RuntimeError, match="closed or faulted"):
            backend.step([dict(env_id=0, actions=actions())])
        backend.close()
        assert robot.closed and cameras.closed and task.closed
        assert len(attempts) == 2
        backend.close()
        assert len(attempts) == 2


def test_readonly_hardware_rejects_rl_reset_without_connecting_or_actuating():
    robot = FakeRobot(actuation_enabled=False)
    with environment(robot=robot) as (backend, _, cameras, task, _):
        assert backend.describe()["execution"]["actuation_enabled"] is False
        with pytest.raises(RuntimeError, match="Actuation is disabled"):
            backend.reset([0], 1)
        assert not robot.connected and not cameras.connected
        assert not robot.holds and not robot.sent and not task.resets


@pytest.mark.parametrize("num_envs", [0, 2, True, 1.0])
def test_only_one_physical_environment_is_supported(num_envs):
    with pytest.raises(ValueError, match="exactly one"):
        RealRobotBackend({"instruction": "fake", "num_envs": num_envs}, _start_watchdog=False)


def test_default_hardware_and_camera_factories_advertise_compatible_abi_without_connecting():
    # Actual component constructors are exercised, but serials are explicit
    # nonexistent test identities and no reset/connect operation is performed.
    from pressb.robots.piper.real import PiperHardware
    from pressb.sensors.realsense import RealSenseCameras

    config = dict(instruction="Unconnected metadata test.", cameras={"config": {
        "up": "UNCONNECTED-TEST-CAMERA-UP", "wrist": "UNCONNECTED-TEST-CAMERA-WRIST"}})
    backend = RealRobotBackend(config, task=FakeTask(None), _start_watchdog=False)
    try:
        assert isinstance(backend.robot, PiperHardware)
        assert isinstance(backend.cameras, RealSenseCameras)
        assert backend.describe()["execution"]["actuation_enabled"] is False
        assert backend.describe()["observation"]["cameras"] == ["up", "wrist"]
        assert not backend.status()["connected"]
        assert negotiate(inference_metadata(backend), backend.describe(), "base")["action_dim"] == 7
    finally:
        backend.close()


def test_close_run_holds_robot_and_allows_a_new_owner_and_reset():
    with environment() as (backend, robot, _, task, _):
        service = SimulationService(backend)
        slot = reset(service)
        service.step(step_request(slot))
        hold_count = len(robot.holds)
        with pytest.raises(ValueError, match="owning run"):
            service.close_run({"run_id": "foreign"})
        assert len(robot.holds) == hold_count
        assert service.close_run({"run_id": "run"}) == {"closed": True, "restart_required": False}
        assert len(robot.holds) == hold_count + 1
        assert not backend.status()["active"]
        assert task.ends[-1] == "run_closed"
        next_slot = reset(service, request="new-reset", run="new-owner")
        assert next_slot["episode_id"] != slot["episode_id"]
        assert backend.status()["active"]


def test_external_end_is_idempotent_preserves_step_response_and_allows_reset():
    with environment() as (backend, robot, _, task, _):
        service = SimulationService(backend)
        step_payload = step_request(reset(service))
        response = service.step(step_payload)
        slot = response["items"][0]
        hold_count, command_count = len(robot.holds), len(robot.sent)
        payload = dict(run_id="run", request_id="external-end", items=[{
            key: slot[key] for key in ("env_id", "episode_id", "step_id")}])
        result = service.end_episode(payload)
        assert result == {"ended": payload["items"]}
        assert service.end_episode(deepcopy(payload)) == result
        assert len(robot.holds) == hold_count + 1 and len(robot.sent) == command_count
        assert service.slots[0]["done"] and service.slots[0]["step_id"] == slot["step_id"]
        assert not backend.status()["active"] and task.ends[-1] == "run_closed"
        assert service.step(deepcopy(step_payload)) == response
        assert response["items"][0]["terminated"] is False
        status = service.status()
        assert status["backend"]["robot"]["kind"] == "explicit_fake_hardware"
        assert status["backend"]["active"] is False
        with pytest.raises(ValueError, match="current active episode"):
            service.end_episode({**payload, "request_id": "end-twice"})
        new_slot = reset(service, request="reset-after-external-end")
        assert new_slot["episode_id"] != slot["episode_id"]
        assert len(task.resets) == 2


@pytest.mark.parametrize("invalid", ["run", "episode", "step", "boolean_step", "env", "duplicate_env"])
def test_external_end_rejects_wrong_owner_or_identity_before_touching_hardware(invalid):
    with environment() as (backend, robot, _, task, _):
        service = SimulationService(backend)
        slot = service.step(step_request(reset(service)))["items"][0]
        payload = dict(run_id="run", request_id="external-end", items=[{
            key: slot[key] for key in ("env_id", "episode_id", "step_id")}])
        if invalid == "run":
            payload["run_id"] = "foreign-owner"
        elif invalid == "episode":
            payload["items"][0]["episode_id"] = "old-episode"
        elif invalid == "step":
            payload["items"][0]["step_id"] -= 1
        elif invalid == "boolean_step":
            payload["items"][0]["step_id"] = True
        elif invalid == "env":
            payload["items"][0]["env_id"] = 1
        else:
            payload["items"].append(deepcopy(payload["items"][0]))
        hold_count = len(robot.holds)
        with pytest.raises(ValueError):
            service.end_episode(payload)
        assert len(robot.holds) == hold_count and not task.ends
        assert backend.status()["active"] and not service.slots[0]["done"]
        assert not service.fault


@pytest.mark.parametrize("operation", ["close_run", "end_episode"])
def test_failed_hold_during_close_or_external_end_keeps_ownership_and_latches_fault(operation):
    with environment() as (backend, robot, _, _, _):
        service = SimulationService(backend)
        slot = reset(service)
        robot.read_error = RuntimeError("synthetic hold feedback failure")
        payload = dict(run_id="run", request_id="failed-end", items=[{
            key: slot[key] for key in ("env_id", "episode_id", "step_id")}])
        with pytest.raises(RuntimeError, match="hold feedback failure"):
            getattr(service, operation)(payload)
        assert service.run_id == "run" and not service.slots[0]["done"]
        assert service.fault.startswith(operation + ":")
        assert backend.status()["fault"] and robot.stop_event.is_set()
        with pytest.raises(RuntimeError, match="restart"):
            reset(service, request="after-failed-close")


class FakeInferenceBackend:
    """Bounded synthetic joint7 policy with a nine-channel noise interface."""
    def __init__(self, environment_backend):
        self.metadata = inference_metadata(environment_backend)
        self.encodes = self.decodes = 0

    def describe(self):
        return deepcopy(self.metadata)

    def encode_batch(self, observations, seeds):
        self.encodes += len(observations)
        return [(dict(state=np.asarray(obs["state"]), seed=seed),
                 np.array([obs["state"][0], .2, .3], dtype=np.float32))
                for obs, seed in zip(observations, seeds)]

    def decode_batch(self, contexts, initial_noises):
        self.decodes += len(contexts)
        output = []
        for context, noise in zip(contexts, initial_noises):
            command = np.asarray(actions(self.metadata["action"]["horizon"], joint=.1))
            if noise is not None:
                command[:, :6] += .01 * np.tanh(np.asarray(noise)[:, :6])
            output.append(command)
        return output


@contextmanager
def http_service(service):
    with RPCServer(("127.0.0.1", 0), service.handlers) as server:
        worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .005}, daemon=True)
        worker.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            worker.join(timeout=2)
            assert not worker.is_alive()


def test_real_http_client_replays_an_identical_request_without_repeating_actuation():
    with environment(task_ticks=2) as (backend, robot, _, _, _):
        service = SimulationService(backend)
        with http_service(service) as endpoint:
            client = NodeClient(endpoint, timeout=2)
            try:
                assert client.call("/health")["environment_type"] == "real_robot"
                slot = client.call("/reset", dict(run_id="run", request_id="reset", env_ids=[0], seed=8))["items"][0]
                payload = step_request(slot, count=4)
                result = client.call("/step", payload)
                assert client.call("/step", payload) == result
                assert result["items"][0]["terminated"] and len(robot.sent) == 2
                assert client.call("/close_run", {"run_id": "run"})["closed"]
            finally:
                client.close()


@pytest.mark.parametrize("method", ["base", "dsrl", "action_residual"])
def test_generic_trainer_runs_over_real_http_with_fake_hardware_and_labels(tmp_path, method):
    import torch
    from pressb.runtime.trainer import GenericTrainer
    from test_rl_nodes_trainer import config as trainer_config

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with environment(task_ticks=3) as (backend, robot, _, task, _):
            simulation = SimulationService(backend)
            fake_policy = FakeInferenceBackend(backend)
            inference = InferenceService(fake_policy, batch_size=1)
            with http_service(simulation) as simulation_endpoint, http_service(inference) as inference_endpoint:
                config = trainer_config(tmp_path / method, method=method, steps=6)
                config["endpoints"] = dict(inference=inference_endpoint, simulation=simulation_endpoint, timeout=3)
                if method == "base":
                    config["mode"] = "eval"
                    config["rollout"]["eval_episodes"] = 3
                if method == "action_residual":
                    config["learner"].update(residual_indices=[0, 2], residual_scale=[.02, .03])
                trainer = GenericTrainer(config)
                result = trainer.run()
                assert result["state"] == "completed"
                assert result["transitions"] == 6
                assert result["episodes"] == result["successes"] == 3
                assert len(robot.sent) == 9
                assert len(task.resets) == 3 and task.ends[-1] == "run_closed"
                assert not backend.status()["active"] and not backend.status()["fault"]
                assert simulation.run_id is None and not inference.contexts
                assert fake_policy.encodes > 0 and fake_policy.decodes > 0
                assert trainer.contract["num_envs"] == 1
                saved = json.loads((tmp_path / method / "negotiated.json").read_text())
                assert "explicit-fake-policy" in json.dumps(saved)
                if method == "base":
                    assert result["updates"] == 0
                else:
                    assert result["updates"] > 0 and len(trainer.replay) == 6
                    arrays = trainer.replay.arrays
                    np.testing.assert_array_equal(arrays["terminated"][:6, 0], [False, True] * 3)
                    assert not arrays["truncated"][:6].any()
                    assert (tmp_path / method / "checkpoints").is_dir()
    finally:
        torch.set_num_threads(previous_threads)
