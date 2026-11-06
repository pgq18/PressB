"""Schema-v2 bridge to an already-running legacy fast elevator simulator.

The legacy simulator owns Isaac and its physical success/termination rules.
This adapter only translates observations, commands and indexed reset requests.
Its shared world requires all active slots in every seven-command step; it does
not silently fill omitted environments with holds or advance them without data.
"""
from __future__ import annotations

import copy
import math
import uuid

import numpy as np

from pressb.online_rl.protocol import pose8_to_pose9, pose9_to_pose8, task_floor
from pressb.runtime.rpc import RPCClient
from pressb.runtime.contract import finite_array, fingerprint, validate_actions, validate_metadata, validate_observation


class ElevatorBackend:
    def __init__(self, config, *, client=None):
        allowed = {"endpoint", "timeout", "token", "run_id", "floors", "panel_randomization", "expected_metadata"}
        if not isinstance(config, dict) or set(config) - allowed:
            raise ValueError(f"Elevator adapter configuration supports {sorted(allowed)}")
        if not isinstance(config.get("endpoint"), str) or not config["endpoint"].startswith(("http://", "https://")):
            raise ValueError("Elevator adapter requires the existing simulator HTTP(S) endpoint")
        timeout = config.get("timeout", 180.)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        self.floors = config.get("floors", list(range(24, 36)))
        if (not isinstance(self.floors, (list, tuple)) or not self.floors
                or any(type(f) is not int or not 24 <= f <= 35 for f in self.floors)
                or len(set(self.floors)) != len(self.floors)):
            raise ValueError("floors must be distinct integers in 24..35")
        self.randomize_panel = config.get("panel_randomization", True)
        if type(self.randomize_panel) is not bool:
            raise ValueError("panel_randomization must be boolean")
        self.run_id = config.get("run_id", "schema2-elevator-" + uuid.uuid4().hex)
        if not isinstance(self.run_id, str) or not self.run_id.strip() or len(self.run_id) > 256:
            raise ValueError("run_id must be a nonempty string of at most 256 characters")
        expected = config.get("expected_metadata", {})
        if not isinstance(expected, dict):
            raise ValueError("expected_metadata must map legacy health fields to expected values")
        self.client = client or RPCClient(config["endpoint"], timeout=timeout, token=config.get("token"))
        self._closed = False
        self._fault = None
        self._cohort = None
        self._step = 0
        self._items = {}
        self._source = self.client.call("/health")
        health = self._source
        required = {"service": "simulation", "protocol_version": 1, "ready": True,
                    "supports_indexed_reset": True, "physics_hz": 120, "action_fps": 30,
                    "action_horizon": 7, "pose_frame": "base_link", "pose_link": "gripper_tcp",
                    "quaternion_order": "wxyz", "camera_order": ["global", "wrist"],
                    "gripper_width_m": .008}
        for requirements in (required, expected):
            for key, value in requirements.items():
                if health.get(key) != value:
                    raise ValueError(f"Legacy elevator metadata mismatch for {key}: expected {value!r}")
        # A live or ended run belongs to its original caller; no implicit takeover.
        if health.get("run_id") is not None:
            raise ValueError("Legacy simulator is already owned by a run; close that run or restart it before attaching")
        self.num_envs = health.get("num_envs")
        if type(self.num_envs) is not int or self.num_envs < 1:
            raise ValueError("Legacy num_envs must be a positive integer")
        self._bounds = health.get("panel_bounds", {})
        for axis in "xy":
            bounds = finite_array([self._bounds.get(f"min_offset_{axis}_m"),
                                   self._bounds.get(f"max_offset_{axis}_m")], (2,), "panel_bounds", dtype=np.float64)
            if bounds[0] > bounds[1]:
                raise ValueError("Legacy panel bounds must be increasing")
        observation = dict(state=dict(names=["x", "y", "z", "qw", "qx", "qy", "qz", "gripper_width"],
                                      units=["m"] * 3 + ["unitless"] * 4 + ["m"]),
                           cameras=["global", "wrist"])
        action = dict(representation="pose9", frame="robot_base",
                      names=["x", "y", "z"] + [f"rot6d_{i}" for i in range(6)],
                      units=["m"] * 3 + ["unitless"] * 6, horizon=7, control_hz=30)
        self._metadata = dict(schema_version=2, node_type="simulation", task_id="elevator",
            num_envs=self.num_envs, observation=observation, action=action,
            control_state_dim=health.get("control_state_dim"), task_features_dim=12,
            discount=dict(gamma=health.get("single_gamma"), timebase_hz=30), reset_mode="indexed",
            scheduling="all_active", initial_reset="all_envs", execution_horizons=[7],
            requires_restart_after_active_close=True,
            fixed_gripper_width_m=.008, quaternion_order="wxyz", rotation6d="first_two_matrix_rows_row_major",
            smoothing=copy.deepcopy(health.get("motion_smoothing", {})),
            task_config_sha256=fingerprint({"legacy_config": health.get("config_sha256"),
                "scene": health.get("scene_sha256"), "max_seconds": health.get("max_seconds"),
                "bounds": self._bounds, "floors": self.floors, "panel_randomization": self.randomize_panel,
                "reward": health.get("reward_contract"), "smoothing": health.get("motion_smoothing")}),
            execution=dict(backend="legacy_fast_isaac_http", physics_hz=120,
                reset="indexed_zero_step_for_ended_slots", step="all_active_slots_full_seven_command_chunks",
                lifecycle="Restart the legacy simulator after ending a run with active slots"),
            reward=copy.deepcopy(health.get("reward_contract", {})),
            legacy_identity={key: copy.deepcopy(health[key]) for key in
                             ("config_sha256", "scene_sha256", "collection_fingerprint", "max_seconds") if key in health})
        validate_metadata(self._metadata, "simulation")

    def describe(self):
        return copy.deepcopy(self._metadata)

    def _available(self):
        if self._closed or self._fault is not None:
            raise RuntimeError(f"Elevator adapter is closed or faulted; remote execution state may be unknown: {self._fault}")

    def _ids(self, ids):
        self._available()
        if (not isinstance(ids, list) or not ids or any(type(i) is not int or not 0 <= i < self.num_envs for i in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError("env_ids must be distinct existing integer environment ids")

    def _request(self, path, payload):
        try:
            return self.client.call(path, {"run_id": self.run_id, "request_id": uuid.uuid4().hex, **payload})
        except BaseException as error:
            self._fault = str(error)
            raise

    def _episode(self, env_id, seed, options):
        if not isinstance(options, dict) or set(options) - {"floor", "task", "offset_x_m", "offset_y_m"}:
            raise ValueError("Elevator reset options support floor, task, offset_x_m and offset_y_m (plus by_env)")
        rng = np.random.default_rng(np.random.SeedSequence([seed, env_id]))
        floor = options.get("floor", int(rng.choice(self.floors)))
        if "task" in options:
            selected = task_floor(options["task"])
            if "floor" in options and floor != selected:
                raise ValueError("floor and task specify different elevator targets")
            floor = selected
        if type(floor) is not int or floor not in self.floors:
            raise ValueError("Reset floor is outside the configured floor set")
        result = {"floor": floor}
        for axis in "xy":
            low, high = (self._bounds[f"{which}_offset_{axis}_m"] for which in ("min", "max"))
            default = rng.uniform(low, high) if self.randomize_panel else self._source["config"].get(f"panel_offset_{axis}_m", 0.)
            value = options.get(f"offset_{axis}_m", float(default))
            offset = finite_array([value], (1,), f"offset_{axis}_m", dtype=np.float64)[0]
            if not low <= offset <= high:
                raise ValueError("Reset panel offset lies outside the existing simulator's frozen bounds")
            result[f"offset_{axis}_m"] = float(offset)
        return result

    def _row(self, item, *, step=False):
        observation = copy.deepcopy(item["observation"])
        state = finite_array(observation.get("state"), (8,), "state")
        pose8_to_pose9(state)  # Check the measured quaternion convention, without changing its representation.
        control = finite_array(observation.get("control_state"), (self._metadata["control_state_dim"],), "control_state")
        features = np.zeros(12, np.float32)
        features[task_floor(observation.get("task")) - 24] = 1.
        observation.update(state=state.tolist(), control_state=control.tolist(), task_features=features.tolist(),
            model_inputs={}, images={name: {"encoding": "png_base64", "data": observation["images"][name]}
                                     for name in ("global", "wrist")})
        validate_observation(observation, self._metadata["observation"])
        terminated, truncated = item.get("terminated"), item.get("truncated")
        if type(terminated) is not bool or type(truncated) is not bool or (terminated and truncated):
            raise ValueError("Invalid legacy termination flags")
        info = copy.deepcopy(item.get("info", {}))
        if not isinstance(info, dict):
            raise ValueError("Legacy transition info must be an object")
        result = dict(env_id=item["env_id"], observation=observation, terminated=terminated,
                      truncated=truncated, info=info)
        if step:
            ticks = item.get("executed_physics_steps")
            if type(ticks) is not int or not 1 <= ticks <= 28:
                raise ValueError("Legacy step must execute between 1 and 28 physics ticks")
            reward, discount = finite_array([item.get("reward"), item.get("discount")], (2,), "reward/discount", dtype=np.float64)
            expected_discount = self._metadata["discount"]["gamma"] ** (ticks / 4)
            if not math.isclose(discount, expected_discount, rel_tol=1e-7, abs_tol=1e-8):
                raise ValueError("Legacy discount does not match its reported elapsed physics ticks")
            result.update(reward=float(reward), discount=float(discount))
            info.update(elapsed_control_steps=ticks / 4, physics_steps=ticks,
                        executed_physics_steps=ticks, elapsed_seconds=ticks / 120)
        return result

    def _response(self, response, expected_ids):
        try:
            items = response["items"]
            if (not isinstance(items, list) or len(items) != len(expected_ids)
                    or any(not isinstance(x, dict) or type(x.get("env_id")) is not int for x in items)
                    or {x["env_id"] for x in items} != set(expected_ids)):
                raise ValueError("Legacy response returned incorrect environment ids")
            mapping = {item["env_id"]: item for item in items}
            for item in items:
                if not isinstance(item.get("episode_id"), str) or not item["episode_id"]:
                    raise ValueError("Legacy response lacks episode identity")
            return mapping
        except BaseException as error:
            self._fault = str(error)
            raise

    def reset(self, env_ids, seed, options=None):
        self._ids(env_ids)
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer in [0,2**63)")
        initial = self._cohort is None
        if initial and set(env_ids) != set(range(self.num_envs)):
            raise ValueError("Initial elevator reset must include all environments")
        if not initial and any(not (self._items[i]["terminated"] or self._items[i]["truncated"]) for i in env_ids):
            raise ValueError("Only ended elevator environments may be reset")
        options = {} if options is None else copy.deepcopy(options)
        if not isinstance(options, dict):
            raise ValueError("reset options must be an object")
        per_env = options.pop("by_env", {})
        if not isinstance(per_env, dict) or any(not isinstance(v, dict) for v in per_env.values()):
            raise ValueError("reset options.by_env must map ids to option objects")
        normalized = {}
        for key, value in per_env.items():
            if type(key) is int:
                env_id = key
            elif isinstance(key, str) and key.isdecimal() and str(int(key)) == key:
                env_id = int(key)
            else:
                raise ValueError("reset options.by_env keys must identify existing environments")
            if not 0 <= env_id < self.num_envs or env_id in normalized:
                raise ValueError("reset options.by_env keys contain a duplicate or unknown environment")
            normalized[env_id] = value
        episodes = {i: self._episode(i, seed, {**options, **normalized.get(i, {})}) for i in env_ids}
        if initial:
            response = self._request("/reset", dict(seed=seed, episodes=[episodes[i] for i in range(self.num_envs)]))
        else:
            response = self._request("/reset_envs", dict(seed=seed, cohort_id=self._cohort, step_id=self._step,
                episodes=[dict(env_id=i, **episodes[i]) for i in env_ids]))
        try:
            mapping = self._response(response, env_ids)
            if ((initial and (not isinstance(response.get("cohort_id"), str) or not response["cohort_id"]))
                    or (not initial and response.get("cohort_id") != self._cohort)
                    or type(response.get("step_id")) is not int or response["step_id"] != (0 if initial else self._step)):
                raise ValueError("Legacy reset returned an invalid cohort/step identity")
            rows = [self._row(mapping[i]) for i in env_ids]
            if any(row["terminated"] or row["truncated"] for row in rows):
                raise ValueError("Reset returned an already-ended environment")
            if any(task_floor(row["observation"]["task"]) != episodes[row["env_id"]]["floor"] for row in rows):
                raise ValueError("Legacy reset changed the requested target floor")
            self._items.update(copy.deepcopy(mapping))
            self._cohort = response["cohort_id"]
            return rows
        except BaseException as error:
            self._fault = str(error)
            raise

    def step(self, items):
        if not isinstance(items, list) or any(not isinstance(x, dict) for x in items):
            raise ValueError("step items must be a list of action objects")
        ids = [item.get("env_id") for item in items]
        self._ids(ids)
        active = {i for i, item in self._items.items() if not (item["terminated"] or item["truncated"])}
        if self._cohort is None or set(ids) != active:
            raise ValueError("Elevator step requires exactly all active environments after reset")
        actions = []
        for item in items:
            pose9 = validate_actions(item.get("actions"), self._metadata["action"])
            pose8 = pose9_to_pose8(pose9)
            if np.any(np.linalg.norm(pose8[:, :3], axis=1) > 1.5):
                raise ValueError("Elevator pose target exceeds 1.5 m from the robot base")
            actions.append(dict(env_id=item["env_id"], actions_pose8=pose8.tolist()))
        response = self._request("/step", dict(cohort_id=self._cohort, step_id=self._step, actions=actions))
        try:
            mapping = self._response(response, range(self.num_envs))
            if (response.get("cohort_id") != self._cohort or type(response.get("step_id")) is not int
                    or response["step_id"] != self._step + 1):
                raise ValueError("Legacy step returned a stale cohort/step identity")
            if any(mapping[i]["episode_id"] != self._items[i]["episode_id"] for i in range(self.num_envs)):
                raise ValueError("Legacy step unexpectedly changed episode identity")
            if any(mapping[i]["observation"].get("task") != self._items[i]["observation"].get("task")
                   for i in range(self.num_envs)):
                raise ValueError("Legacy step unexpectedly changed the episode task")
            for i in set(range(self.num_envs)) - active:
                if (mapping[i].get("terminated") != self._items[i]["terminated"]
                        or mapping[i].get("truncated") != self._items[i]["truncated"]
                        or mapping[i].get("executed_physics_steps") != 0):
                    raise ValueError("Legacy step advanced or reset an inactive environment")
            rows = [self._row(mapping[i], step=True) for i in ids]
            self._items, self._step = copy.deepcopy(mapping), response["step_id"]
            return rows
        except BaseException as error:
            self._fault = str(error)
            raise

    def close(self):
        self._closed = True
        close = getattr(self.client, "close", None)
        if close:
            close()


def create_backend(config):
    return ElevatorBackend(config)
