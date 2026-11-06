"""Task-independent reset/step service with indexed episode ownership.

The backend owns physics, rewards, action processing and observations. This
service owns sequencing, idempotency and wire validation. Default step never
auto-resets terminal states. The optional atomic step/indexed-reset operation
keeps old transitions and new initial observations separate. Neither operation
retries physics or turns an infrastructure error into a failed task episode.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import math
import uuid

from .contract import (finite_array, fingerprint, validate_actions, validate_metadata,
                       validate_observation, validate_terminal_observation_mode)


class SimulationService:
    def __init__(self, backend, request_cache_size=8):
        if type(request_cache_size) is not int or request_cache_size < 1:
            raise ValueError("request_cache_size must be positive")
        self.backend = backend
        self.metadata = deepcopy(validate_metadata(backend.describe(), "simulation"))
        self.request_cache_size = request_cache_size
        self.requests = OrderedDict()
        self.seen_requests = set()
        self.run_id = None
        self.slots = {}
        self.fault = None

    @property
    def handlers(self):
        return {"/health": self.health, "/status": self.status, "/reset": self.reset,
                "/step": self.step, "/step_and_reset": self.step_and_reset, "/close_run": self.close_run}

    def health(self, payload=None):
        return deepcopy(self.metadata)

    def status(self, payload=None):
        return {"run_id": self.run_id, "fault": self.fault, "environments": deepcopy(self.slots)}

    def _request(self, payload, operation):
        if self.fault:
            raise RuntimeError(f"Backend execution state is unknown; restart the node: {self.fault}")
        if not isinstance(payload, dict):
            raise ValueError("Request must be an object")
        for name in ("run_id", "request_id"):
            if not isinstance(payload.get(name), str) or not payload[name]:
                raise ValueError(f"{name} is required")
        if self.run_id is not None and payload["run_id"] != self.run_id:
            raise ValueError("Simulation is owned by another run; close that run first")
        key = (payload["run_id"], payload["request_id"])
        digest = fingerprint({"operation": operation, "payload": payload})
        if key in self.requests:
            previous_digest, response = self.requests[key]
            if previous_digest != digest:
                raise ValueError("request_id reused with different operation or content")
            self.requests.move_to_end(key)
            return key, digest, deepcopy(response)
        if key in self.seen_requests:
            raise ValueError("Request response has expired from the cache; physics was not repeated")
        return key, digest, None

    def _commit(self, key, digest, response):
        self.requests[key] = (digest, deepcopy(response))
        self.seen_requests.add(key)
        while len(self.requests) > self.request_cache_size:
            self.requests.popitem(last=False)
        return response

    def _ids(self, env_ids):
        if (not isinstance(env_ids, list) or not env_ids
                or any(type(i) is not int or not 0 <= i < self.metadata["num_envs"] for i in env_ids)
                or len(set(env_ids)) != len(env_ids)):
            raise ValueError("env_ids must contain distinct valid environment indices")

    def _observation(self, row, *, images_omitted=False):
        observation = validate_observation(row["observation"], self.metadata["observation"],
                                           allow_missing_images=images_omitted)
        for key, dim in (("control_state", self.metadata["control_state_dim"]),
                         ("task_features", self.metadata["task_features_dim"])):
            value = observation.get(key, [])
            finite_array(value, (dim,), f"observation.{key}")
        return observation

    @staticmethod
    def _rows(rows, env_ids):
        if not isinstance(rows, list) or len(rows) != len(env_ids):
            raise RuntimeError("Backend returned wrong number of environment rows")
        if [row.get("env_id") for row in rows] != env_ids:
            raise RuntimeError("Backend returned reordered or wrong environment indices")

    def reset(self, payload):
        key, digest, previous = self._request(payload, "reset")
        if previous is not None:
            return previous
        env_ids = payload.get("env_ids")
        self._ids(env_ids)
        seed = payload.get("seed")
        if type(seed) is not int or seed < 0:
            raise ValueError("Reset seed must be a nonnegative integer")
        options = payload.get("options", {})
        if not isinstance(options, dict):
            raise ValueError("Reset options must be an object")
        if self.run_id is None and self.metadata.get("initial_reset") == "all_envs" and set(env_ids) != set(range(self.metadata["num_envs"])):
            raise ValueError("This adapter requires all environments in its initial reset")
        if any(i in self.slots and not self.slots[i]["done"] for i in env_ids):
            raise ValueError("Cannot reset an active episode; close the run to abandon it explicitly")
        try:
            rows = self.backend.reset(env_ids, seed, options)
            self._rows(rows, env_ids)
            results, slots = [], {}
            for row in rows:
                observation = self._observation(row)
                env_id = row["env_id"]
                slot = {"episode_id": uuid.uuid4().hex, "step_id": 0, "done": False}
                slots[env_id] = slot
                results.append(dict(env_id=env_id, episode_id=slot["episode_id"], step_id=0,
                                    observation=observation, info=row.get("info", {})))
        except Exception as error:
            self.fault = f"reset: {type(error).__name__}: {error}"
            raise
        self.run_id = payload["run_id"]
        self.slots.update(slots)
        return self._commit(key, digest, {"items": results})

    def _step_commands(self, payload):
        if self.run_id is None:
            raise ValueError("Reset must precede step")
        terminal_mode = validate_terminal_observation_mode(
            payload.get("terminal_observation_mode", "full"), self.metadata)
        items = payload.get("items")
        if not isinstance(items, list) or not items:
            raise ValueError("step.items must be nonempty")
        env_ids = [item.get("env_id") for item in items]
        self._ids(env_ids)
        if self.metadata.get("scheduling") == "all_active" and set(env_ids) != {i for i, s in self.slots.items() if not s["done"]}:
            raise ValueError("This simulation adapter requires all active environments per step")
        commands = []
        for item in items:
            env_id = item["env_id"]
            slot = self.slots.get(env_id)
            if slot is None or slot["done"]:
                raise ValueError("Environment is uninitialized or terminal; reset it first")
            if item.get("episode_id") != slot["episode_id"] or type(item.get("step_id")) is not int or item["step_id"] != slot["step_id"]:
                raise ValueError("Stale episode_id or step_id")
            actions = validate_actions(item.get("actions"), self.metadata["action"], allow_prefix=True)
            if "execution_horizons" in self.metadata and len(actions) not in self.metadata["execution_horizons"]:
                raise ValueError("Unsupported execution horizon")
            commands.append({"env_id": env_id, "actions": actions.tolist()})
        if (self.metadata.get("equal_execution_horizon", False)
                and len({len(command["actions"]) for command in commands}) != 1):
            raise ValueError("This simulation adapter requires equal action prefix lengths")
        return commands, terminal_mode

    def _step_results(self, rows, commands, terminal_mode):
        self._rows(rows, [command["env_id"] for command in commands])
        results, slots = [], {}
        for row, command in zip(rows, commands):
            env_id = row["env_id"]
            terminated, truncated = row["terminated"], row["truncated"]
            if type(terminated) is not bool or type(truncated) is not bool or (terminated and truncated):
                raise RuntimeError("Backend terminated/truncated flags must be exclusive booleans")
            omitted = row.get("terminal_observation_omitted", False)
            expected_omission = terminal_mode != "full" and (terminated or truncated)
            if type(omitted) is not bool or omitted != expected_omission:
                raise RuntimeError("Backend terminal_observation_omitted disagrees with negotiated mode")
            observation = self._observation(row, images_omitted=omitted)
            reward = float(finite_array([row["reward"]], (1,), "reward")[0])
            discount = float(finite_array([row["discount"]], (1,), "discount")[0])
            info = row.get("info", {})
            if (not isinstance(info, dict) or type(info.get("success")) is not bool
                    or type(info.get("physics_steps")) is not int or info["physics_steps"] <= 0):
                raise RuntimeError("Backend must report boolean success and positive integer physics_steps")
            duration = info.get("elapsed_control_steps")
            if (isinstance(duration, bool) or not isinstance(duration, (int, float))
                    or not math.isfinite(duration) or not 0 < duration <= len(command["actions"])+1e-6):
                raise RuntimeError("Backend must report positive actual elapsed_control_steps")
            spec = self.metadata["discount"]
            expected = spec["gamma"] ** (duration * spec["timebase_hz"] / self.metadata["action"]["control_hz"])
            if not 0 <= discount <= 1 or not math.isclose(discount, expected, rel_tol=1e-5, abs_tol=1e-7):
                raise RuntimeError("Backend discount disagrees with actual execution duration")
            slot = {**self.slots[env_id], "step_id": self.slots[env_id]["step_id"]+1,
                    "done": terminated or truncated}
            slots[env_id] = slot
            results.append(dict(env_id=env_id, episode_id=slot["episode_id"], step_id=slot["step_id"],
                                observation=observation, reward=reward, discount=discount,
                                terminated=terminated, truncated=truncated, info=info))
            if omitted:
                results[-1]["terminal_observation_omitted"] = True
        return results, slots

    def step(self, payload):
        key, digest, previous = self._request(payload, "step")
        if previous is not None:
            return previous
        commands, terminal_mode = self._step_commands(payload)
        try:
            rows = (self.backend.step(commands) if terminal_mode == "full" else
                    self.backend.step(commands, terminal_observation_mode=terminal_mode))
            results, slots = self._step_results(rows, commands, terminal_mode)
        except Exception as error:
            self.fault = f"step: {type(error).__name__}: {error}"
            raise
        self.slots.update(slots)
        return self._commit(key, digest, {"items": results})

    def step_and_reset(self, payload):
        """Opt-in atomic step/indexed reset with one final observation boundary."""
        key, digest, previous = self._request(payload, "step_and_reset")
        if previous is not None:
            return previous
        if "step_and_reset" not in self.metadata.get("capabilities", []):
            raise ValueError("Simulation does not advertise step_and_reset support")
        commands, terminal_mode = self._step_commands(payload)
        if terminal_mode != "state_only_if_no_bootstrap":
            raise ValueError("step_and_reset requires state_only_if_no_bootstrap")
        seed, options = payload.get("reset_seed"), payload.get("reset_options", {})
        maximum = payload.get("max_active_envs")
        if type(seed) is not int or seed < 0 or not isinstance(options, dict):
            raise ValueError("step_and_reset requires a nonnegative reset_seed and reset_options object")
        if type(maximum) is not int or not 0 <= maximum <= self.metadata["num_envs"]:
            raise ValueError("max_active_envs must be an integer between zero and num_envs")
        try:
            response = self.backend.step_and_reset(commands, reset_seed=seed,
                                                  reset_options=options, max_active_envs=maximum)
            results, slots = self._step_results(response["items"], commands, terminal_mode)
            done = [row["env_id"] for row in results if row["terminated"] or row["truncated"]]
            remaining_active = sum(not slot["done"] for slot in self.slots.values()) - len(done)
            reset_ids = done[:max(0, maximum - remaining_active)]
            reset_rows = response["reset_items"]
            self._rows(reset_rows, reset_ids)
            reset_results = []
            for row in reset_rows:
                env_id = row["env_id"]
                observation = self._observation(row)
                slot = {"episode_id": uuid.uuid4().hex, "step_id": 0, "done": False}
                slots[env_id] = slot
                reset_results.append(dict(env_id=env_id, episode_id=slot["episode_id"], step_id=0,
                                          observation=observation, info=row.get("info", {})))
        except Exception as error:
            self.fault = f"step_and_reset: {type(error).__name__}: {error}"
            raise
        self.slots.update(slots)
        # One commit retains both old terminal identity and new initial identity.
        # An identical request retry returns these exact IDs without more physics.
        return self._commit(key, digest, {"items": results, "reset_items": reset_results})

    def close_run(self, payload):
        if self.fault:
            raise RuntimeError("Faulted simulation must be restarted")
        if self.run_id is not None and payload.get("run_id") != self.run_id:
            raise ValueError("Only the owning run can close its session")
        restart_required = (self.metadata.get("requires_restart_after_active_close", False)
                            and any(not slot["done"] for slot in self.slots.values()))
        self.run_id = None
        self.slots.clear()
        self.requests.clear()
        self.seen_requests.clear()
        if restart_required:
            self.fault = "Adapter requires a simulator restart after abandoning active episodes"
        # Physics is not advanced. The next run must reset its environments.
        return {"closed": True, "restart_required": bool(restart_required)}

    def close(self):
        close = getattr(self.backend, "close", None)
        if close:
            close()
