"""Single physical environment behind the existing reset/step wire contract.

Hardware, cameras and task outcome/reset logic are independent factories. No
simulator, model implementation or LeRobot checkout is imported here. The wire
node_type stays ``simulation`` for compatibility; environment_type identifies
physical execution. Reset never teleports or automatically homes an arm.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
import inspect
import math
from pathlib import Path
import threading
import time

import numpy as np

from .contract import encode_image, finite_array, fingerprint, validate_actions, validate_metadata


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _component(spec, default):
    if not isinstance(spec, dict) or set(spec) - {"factory", "config"}:
        raise ValueError("Components support factory and config only")
    factory = spec.get("factory", default)
    if not isinstance(factory, str) or factory.count(":") != 1 or not isinstance(spec.get("config", {}), dict):
        raise ValueError("Component requires module:factory and an object config")
    module, name = factory.split(":")
    return getattr(importlib.import_module(module), name)(deepcopy(spec.get("config", {})))


def _outcome(value):
    if not isinstance(value, dict):
        raise ValueError("Task outcome must be a mapping")
    result = {"reward": 0., "terminated": False, "truncated": False, "success": False, "reason": "running", **value}
    if any(type(result[key]) is not bool for key in ("terminated", "truncated", "success")):
        raise ValueError("Task outcome flags must be booleans")
    if result["terminated"] and result["truncated"] or result["success"] and not result["terminated"]:
        raise ValueError("Success must terminate; terminated and truncated are exclusive")
    if isinstance(result["reward"], bool) or not isinstance(result["reward"], (int, float)) or not math.isfinite(result["reward"]):
        raise ValueError("Task reward must be finite")
    return result


class RealRobotBackend:
    """One robot, paced action prefixes, fresh observations and pluggable labels.

    Task hooks: reset(seed, options), poll(), describe(), end(reason), close().
    An optional evaluate(observation, context) can provide automated outcomes
    from the measured state/images at each chunk boundary. ``poll`` supplies
    asynchronous events such as an operator label or an external button sensor.
    All infrastructure faults latch and require restarting this backend.
    """

    def __init__(self, config, *, robot=None, cameras=None, task=None,
                 clock=time.monotonic, sleep=time.sleep, _start_watchdog=True):
        allowed = {"robot", "cameras", "task", "task_id", "instruction", "horizon", "control_hz",
                   "gamma", "discount_timebase_hz", "max_episode_steps", "max_episode_seconds",
                   "watchdog_seconds", "smoothing", "model_inputs", "num_envs"}
        if not isinstance(config, dict) or set(config) - allowed:
            raise ValueError(f"Unknown real robot configuration keys: {set(config) - allowed if isinstance(config, dict) else config}")
        self.config = deepcopy(config)
        if type(config.get("num_envs", 1)) is not int or config.get("num_envs", 1) != 1:
            raise ValueError("A real robot node has exactly one environment")
        self.instruction = config.get("instruction", "")
        if not isinstance(self.instruction, str) or not self.instruction.strip():
            raise ValueError("A nonempty real task instruction is required")
        self.task_id = config.get("task_id", "real_robot")
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("task_id must be nonempty")
        self.horizon = config.get("horizon", 32)
        self.max_steps = config.get("max_episode_steps", 1800)
        if any(type(x) is not int or x < 1 for x in (self.horizon, self.max_steps)):
            raise ValueError("horizon and max_episode_steps must be positive integers")
        self.hz = _positive(config.get("control_hz", 30), "control_hz")
        self.gamma = _positive(config.get("gamma", .995), "gamma")
        if self.gamma > 1:
            raise ValueError("gamma must be <= 1")
        self.discount_hz = _positive(config.get("discount_timebase_hz", self.hz), "discount_timebase_hz")
        self.max_seconds = _positive(config.get("max_episode_seconds", 120), "max_episode_seconds")
        self.watchdog_seconds = _positive(config.get("watchdog_seconds", 15), "watchdog_seconds")
        self.model_inputs = deepcopy(config.get("model_inputs", {}))
        if not isinstance(self.model_inputs, dict):
            raise ValueError("model_inputs must be an object")
        self.smoothing = {"mode": "none", "alpha": .8, **config.get("smoothing", {})}
        if set(self.smoothing) != {"mode", "alpha"}:
            raise ValueError("smoothing supports mode and alpha")
        from pressb.control.joint_actions import JointActionPostprocessor
        # Validate even when disabled; it is applied only for the seven-axis ABI.
        self.postprocessor = JointActionPostprocessor(**self.smoothing)
        self._clock, self._sleep = clock, sleep
        self._lock = threading.RLock()
        self._shutdown = threading.Event()
        self._watchdog = None
        self._closed = self._connected = self._active = False
        self._fault = None
        self._last_activity = self._clock()
        self._last_command = None
        self._steps = 0
        self._episode_started = None
        self.robot = self.cameras = self.task = None
        try:
            self.robot = robot if robot is not None else _component(config.get("robot", {}), "pressb.robots.piper.real:PiperHardware")
            self.cameras = cameras if cameras is not None else _component(config.get("cameras", {}), "pressb.sensors.realsense:RealSenseCameras")
            self.task = task if task is not None else _component(config.get("task", {}), "pressb.runtime.real_robot_operator:ManualTask")
            hardware = self.robot.describe()
            self.width = len(hardware["state"]["names"])
            self.action = {**deepcopy(hardware["action"]), "horizon": self.horizon}
            if self.action["control_hz"] != self.hz:
                raise ValueError("Hardware and environment control_hz must agree")
            if self.action["names"] != hardware["state"]["names"] or self.action["units"] != hardware["state"]["units"]:
                raise ValueError("Real joint-position commands and measured states must share coordinates")
            if self.smoothing["mode"] != "none" and self.width != 7:
                raise ValueError("joint_ema currently supports six arm joints plus gripper")
            camera_names = list(self.cameras.config)
            # Components advertise immutable identity, not live frames/counters.
            identity = deepcopy(config)
            if isinstance(identity.get("task", {}).get("config"), dict):
                identity["task"]["config"].pop("directory", None)
            sources = {}
            for obj in (self, self.robot, self.cameras, self.task):
                path = inspect.getsourcefile(type(obj))
                if path:
                    sources[type(obj).__module__] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            self.metadata = validate_metadata(dict(schema_version=2, node_type="simulation", environment_type="real_robot",
                task_id=self.task_id, num_envs=1, reset_mode="indexed", capabilities=[], terminal_observation_modes=["full"],
                task_config_sha256=fingerprint({"config": identity, "hardware": hardware, "task": self.task.describe()}),
                source_revision=fingerprint(sources), smoothing=deepcopy(self.smoothing),
                observation={"state": deepcopy(hardware["state"]), "cameras": camera_names}, action=self.action,
                control_state_dim=self.width, task_features_dim=0,
                discount={"gamma": self.gamma, "timebase_hz": self.discount_hz},
                execution={"backend": "physical_robot", "robot_type": hardware.get("robot_type"),
                    "actuation_enabled": hardware.get("actuation_enabled", False),
                    "max_episode_steps": self.max_steps, "max_episode_seconds": self.max_seconds,
                    "watchdog_seconds": self.watchdog_seconds, "physics_steps_semantics": "executed hardware control ticks",
                    "discount_semantics": "executed control ticks; inference and manual reset waits excluded"},
                reward_schema=self.task.describe()), "simulation")
            if _start_watchdog:
                self._watchdog = threading.Thread(target=self._watch, name="pressb-robot-watchdog", daemon=True)
                self._watchdog.start()
        except BaseException:
            self.close()
            raise

    def describe(self):
        return deepcopy(self.metadata)

    def status(self):
        return {"connected": self._connected, "active": self._active, "fault": self._fault,
                "executed_control_steps": self._steps, "robot": self.robot.status(), "task": self.task.status()}

    def _check(self):
        if self._closed or self._fault:
            raise RuntimeError(f"Real robot backend closed or faulted; restart required: {self._fault}")

    def _fail(self, error):
        # Latch before touching a potentially blocked transport/operation lock.
        # A command returning after its deadline must never be acknowledged.
        self._fault = f"{type(error).__name__}: {error}"
        self._active = False
        if self._connected:
            try:
                self.robot.stop()
            except Exception as stop_error:
                self._fault += f"; stop failed: {stop_error}"

    def _watch(self):
        while not self._shutdown.wait(min(.1, self.watchdog_seconds / 4)):
            if self._active:
                try:
                    check_abort = getattr(self.task, "check_abort", None)
                    if callable(check_abort):
                        check_abort()
                    if self._clock() - self._last_activity > self.watchdog_seconds:
                        raise TimeoutError("Control watchdog expired while waiting for commands/observations")
                except Exception as error:
                    self._fail(error)

    def _state(self):
        with self._lock:
            self._check()
            return finite_array(self.robot.read_state(), (self.width,), "measured robot state").copy()

    def _observation(self):
        images = self.cameras.read()
        expected = self.metadata["observation"]["cameras"]
        if set(images) != set(expected):
            raise RuntimeError("Camera source returned a different set of views")
        encoded = {}
        for name, pixels in images.items():
            array = np.asarray(pixels)
            if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
                raise ValueError(f"Camera {name} must supply RGB uint8 HWC pixels")
            encoded[name] = encode_image(array)
        state = self._state()
        return {"state": state.tolist(), "images": encoded, "task": self.instruction,
                "control_state": self._last_command.tolist(), "task_features": [],
                "model_inputs": deepcopy(self.model_inputs)}

    def reset(self, env_ids, seed, options=None):
        self._check()
        options = {} if options is None else options
        if env_ids != [0] or type(env_ids[0]) is not int:
            raise ValueError("Real robot reset accepts env_ids=[0] only")
        if type(seed) is not int or seed < 0 or not isinstance(options, dict):
            raise ValueError("Reset requires nonnegative seed and object options")
        if self._active:
            raise ValueError("Cannot reset an active physical episode")
        if not self.metadata["execution"]["actuation_enabled"]:
            raise RuntimeError("Actuation is disabled; use camera/feedback diagnostics, not RL reset/step")
        try:
            if not self._connected:
                self.robot.connect()
                self._connected = True
                self.cameras.connect()
            # Safe hold only; task hook asks the operator to restore the scene.
            with self._lock:
                self.robot.hold()
            self.task.reset(seed, deepcopy(options))
            self._last_command = self._state()
            if self.width == 7:
                self.postprocessor.reset(self._last_command)
            self._steps = 0
            observation = self._observation()
            with self._lock:
                self._check()
                self._episode_started = self._last_activity = self._clock()
                self._active = True
            return [{"env_id": 0, "observation": observation, "info": {
                "environment_type": "real_robot", "reset_seed": seed,
                "reset_semantics": "operator/task hook; no automatic homing or scene randomization",
                "task_status": self.task.status()}}]
        except BaseException as error:
            self._fail(error)
            raise

    def step(self, commands):
        self._check()
        if not self._active:
            raise RuntimeError("Physical episode requires reset before step")
        if not isinstance(commands, list) or len(commands) != 1 or commands[0].get("env_id") != 0:
            raise ValueError("Real robot step accepts one command for env_id=0")
        actions = validate_actions(commands[0].get("actions"), self.action, allow_prefix=True)
        result, reward, applied = _outcome({}), 0., []
        start = self._clock()
        try:
            for raw in actions:
                # A label arriving during inference must not trigger another
                # policy motion: execute a measured hold for its terminal tick.
                result = _outcome(self.task.poll())
                tick_reward = result["reward"]
                timed_out = self._steps >= self.max_steps or self._clock() - self._episode_started >= self.max_seconds
                terminal = result["terminated"] or result["truncated"] or timed_out
                tick_start = self._clock()
                with self._lock:
                    self._check()
                    if terminal:
                        sent = self.robot.hold()
                    else:
                        command = self.postprocessor.apply(raw) if self.width == 7 else raw
                        sent = self.robot.command(command)
                    self._check()
                    if self._clock() - self._last_activity > self.watchdog_seconds:
                        raise TimeoutError("Hardware operation exceeded the control watchdog deadline")
                    self._last_command = finite_array(sent, (self.width,), "sent robot command").copy()
                    if self.width == 7:
                        # Hardware rate projection is part of execution. EMA
                        # history must follow what was actually sent, too.
                        self.postprocessor.accept_applied(self._last_command)
                    self._last_activity = self._clock()
                applied.append(self._last_command.tolist())
                # No catch-up bursts after an overrun: each tick gets a period.
                self._sleep(max(0., tick_start + 1 / self.hz - self._clock()))
                self._state()  # every tick checks that measured feedback is fresh
                self._steps += 1
                if not terminal:
                    result = _outcome(self.task.poll())
                    tick_reward += result["reward"]
                reward += self.gamma ** ((len(applied) - 1) * self.discount_hz / self.hz) * tick_reward
                if result["terminated"] or result["truncated"]:
                    break
                if timed_out or self._steps >= self.max_steps or self._clock() - self._episode_started >= self.max_seconds:
                    result = _outcome({"truncated": True, "reason": "time_limit"})
                    break
            observation = self._observation()
            evaluate = getattr(self.task, "evaluate", None)
            if evaluate and not (result["terminated"] or result["truncated"]):
                result = _outcome(evaluate(observation, {"executed_control_steps": self._steps,
                    "elapsed_wall_seconds": self._clock() - self._episode_started}))
                reward += self.gamma ** ((len(applied) - 1) * self.discount_hz / self.hz) * result["reward"]
            if result["terminated"] or result["truncated"]:
                with self._lock:
                    self._check()
                    self._last_command = finite_array(self.robot.hold(), (self.width,), "held robot command").copy()
                    self._active = False
                self.task.end(result["reason"])
                observation = self._observation()
            with self._lock:
                self._check()
                self._last_activity = self._clock()
            return [{"env_id": 0, "observation": observation, "reward": reward,
                "discount": self.gamma ** (len(applied) * self.discount_hz / self.hz),
                "terminated": result["terminated"], "truncated": result["truncated"],
                "info": {"success": result["success"], "reason": result["reason"],
                    "physics_steps": len(applied), "elapsed_control_steps": len(applied),
                    "hardware_control_steps": len(applied), "applied_actions": applied,
                    "executed_control_steps": self._steps, "wall_seconds": self._clock() - start,
                    "camera_metadata": deepcopy(getattr(self.cameras, "metadata", {})),
                    "environment_type": "real_robot", "task_status": self.task.status()}}]
        except BaseException as error:
            self._fail(error)
            raise

    def end_episode(self, env_ids):
        if env_ids != [0] or type(env_ids[0]) is not int:
            raise ValueError("Real robot end_episode accepts env_ids=[0] only")
        self.close_run()

    def close_run(self):
        with self._lock:
            self._check()
            try:
                if self._connected:
                    self.robot.hold()
                self._active = False
                self.task.end("run_closed")
            except BaseException as error:
                self._fail(error)
                raise

    def close(self):
        if getattr(self, "_cleanup_complete", False):
            return
        self._shutdown.set()
        if self._watchdog and self._watchdog is not threading.current_thread():
            self._watchdog.join(timeout=2)
        with self._lock:
            self._active = False
            self._closed = True
            errors = []
            for component in (self.robot, self.cameras, self.task):
                if component is not None:
                    try:
                        component.close()
                    except Exception as error:
                        errors.append(str(error))
            if errors:
                self._fault = "; ".join(errors)
                raise RuntimeError("Real robot cleanup incomplete: " + self._fault)
            self._cleanup_complete = True


def create_backend(config):
    return RealRobotBackend(config)
