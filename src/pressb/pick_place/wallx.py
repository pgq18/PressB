"""Native Wall-X policy transport and Piper joint-action units for PAP.

This uses the ordinary ``{obs, is_rtc: false}`` WebSocket contract. It does not
use the VLA-JEPA encoder/flow RPC adapter. State normalization and action
denormalization belong to the Wall-X server; here only physical units change.
"""
from __future__ import annotations

import json
import inspect
from pathlib import Path
import time

import numpy as np


JOINT_KEYS = tuple(f"joint_{index}.pos" for index in range(1, 7)) + ("gripper.pos",)


def _pack_array(value):
    if isinstance(value, (np.ndarray, np.generic)):
        if value.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported MessagePack array dtype: {value.dtype}")
        if isinstance(value, np.ndarray):
            return {b"__ndarray__": True, b"data": value.tobytes(),
                    b"dtype": value.dtype.str, b"shape": value.shape}
        return {b"__npgeneric__": True, b"data": value.item(), b"dtype": value.dtype.str}
    raise TypeError(f"Cannot encode {type(value).__name__}")


def _unpack_array(value):
    if b"__ndarray__" in value or b"__npgeneric__" in value:
        dtype = np.dtype(value[b"dtype"])
        if dtype.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported MessagePack array dtype: {dtype}")
        if b"__ndarray__" in value:
            return np.ndarray(buffer=value[b"data"], dtype=dtype, shape=value[b"shape"])
        return dtype.type(value[b"data"])
    return value


def pack(value):
    """Wall-X/openpi_client NumPy wire format, without importing model code."""
    import msgpack
    return msgpack.packb(value, default=_pack_array)


def unpack(value):
    import msgpack
    return msgpack.unpackb(value, object_hook=_unpack_array)


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


class WallXClient:
    """One synchronous native policy connection, with no retry after failure."""

    def __init__(self, endpoint="ws://127.0.0.1:8094", *, timeout=180., token=None,
                 expected_horizon=32):
        if not endpoint.startswith(("ws://", "wss://")):
            raise ValueError("Wall-X endpoint must be a ws:// or wss:// URL")
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("Policy timeout must be positive and finite")
        if type(expected_horizon) is not int or expected_horizon < 1:
            raise ValueError("Expected policy horizon must be a positive integer")
        self.endpoint, self.timeout, self.token = endpoint, float(timeout), token
        self.expected_horizon = expected_horizon
        self.metadata = None
        self._connection = None
        self._failed = False
        self.last_latency_s = None

    def connect(self):
        from websockets.sync.client import connect

        if self._failed:
            raise RuntimeError("Policy connection failed or closed; automatic retry is disabled")
        if self._connection is not None:
            return self.metadata
        try:
            headers = {"Authorization": f"Api-Key {self.token}"} if self.token else None
            header_args = {}
            if headers:
                key = "additional_headers" if "additional_headers" in inspect.signature(connect).parameters else "extra_headers"
                header_args[key] = headers
            self._connection = connect(self.endpoint, **header_args, compression=None,
                max_size=64 * 1024 * 1024, open_timeout=self.timeout, close_timeout=2.)
            frame = self._connection.recv(timeout=self.timeout)
            if not isinstance(frame, bytes):
                raise ValueError("Expected binary Wall-X metadata")
            metadata = unpack(frame)
            if (not isinstance(metadata, dict) or metadata.get("action_dim") != 7
                    or metadata.get("pred_horizon") != self.expected_horizon):
                raise ValueError(f"Expected a 7D/{self.expected_horizon}-step Piper Wall-X server, got {metadata}")
            if "state_dim" in metadata and metadata["state_dim"] != 7:
                raise ValueError("This Piper policy requires seven raw joint/gripper state values")
            self.metadata = metadata
            return metadata
        except BaseException:
            self.close()
            raise

    def infer(self, observation, *, record_prefix=None):
        """Record exact request/response frames, including RGB, before validation."""
        self.connect()
        request = pack(dict(obs=observation, is_rtc=False))
        prefix = Path(record_prefix) if record_prefix is not None else None
        if prefix is not None:
            prefix.parent.mkdir(parents=True, exist_ok=True)
            prefix.with_suffix(".request.msgpack").write_bytes(request)
        started = time.perf_counter()
        try:
            self._connection.send(request)
            frame = self._connection.recv(timeout=self.timeout)
            if isinstance(frame, str):
                if prefix is not None:
                    prefix.with_suffix(".response_error.txt").write_text(frame)
                raise RuntimeError(f"Wall-X server error: {frame}")
            if prefix is not None:
                prefix.with_suffix(".response.msgpack").write_bytes(frame)
            response = unpack(frame)
            if not isinstance(response, dict) or "action" not in response:
                raise ValueError("Wall-X response is missing action")
            return response
        except BaseException:
            self.close()
            raise
        finally:
            self.last_latency_s = time.perf_counter() - started

    def close(self):
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        self._failed = True


def observation_from_sim(joints, up_rgb, wrist_rgb, prompt, dataset_name="lerobot/piper"):
    """Measured joint1..8 in rad/m → raw training state in degrees/full-width cm."""
    joints = np.asarray(joints, dtype=float)
    if joints.shape != (8,) or not np.isfinite(joints).all():
        raise ValueError("Measured Piper state must contain six arm and two finger joints")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Policy task prompt must be nonempty text")
    if not isinstance(dataset_name, str) or not dataset_name:
        raise ValueError("Wall-X dataset name must be nonempty")
    images = []
    for name, value in (("up", up_rgb), ("wrist", wrist_rgb)):
        array = np.asarray(value)
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3 or min(array.shape[:2]) <= 0:
            raise ValueError(f"{name} must be an HWC uint8 RGB image")
        images.append(np.ascontiguousarray(array))
    state = np.r_[np.rad2deg(joints[:6]), (joints[6] - joints[7]) * 100].astype(np.float32)
    return dict(face_view=images[0], left_wrist_view=images[1], state=state,
                prompt=prompt, dataset_names=[dataset_name])


def decode_actions(action, lower, upper, maximum_width_m, *, horizon=32,
                   arm_limit_mode="error", gripper_limit_mode="clip"):
    """Absolute degree/cm actions → rad/m commands; every saturation is reported.

    A small negative gripper position is a recorded closure command in this
    dataset. The default explicitly clips gripper travel, as PAP replay does.
    Arm bounds fail by default; optional clipping is an explicit run setting.
    """
    raw = np.asarray(action, dtype=float)
    if raw.shape == (1, horizon, 7):
        raw = raw[0]
    if raw.shape != (horizon, 7) or not np.isfinite(raw).all():
        raise ValueError(f"Expected finite Wall-X actions with shape (1,{horizon},7) or ({horizon},7)")
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if (lower.shape != (6,) or upper.shape != (6,) or not np.isfinite([lower, upper]).all()
            or np.any(lower >= upper) or not np.isfinite(maximum_width_m) or maximum_width_m <= 0):
        raise ValueError("Invalid physical joint/gripper bounds")
    if arm_limit_mode not in ("error", "clip") or gripper_limit_mode not in ("error", "clip"):
        raise ValueError("Policy limit mode must be error or clip")
    converted = np.column_stack((np.deg2rad(raw[:, :6]), raw[:, 6] * .01))
    clipped = converted.copy()
    clipped[:, :6] = np.clip(converted[:, :6], lower, upper)
    clipped[:, 6] = np.clip(converted[:, 6], 0., maximum_width_m)
    delta = np.abs(clipped - converted)
    if arm_limit_mode == "error" and np.any(delta[:, :6] > 1e-9):
        row, column = np.argwhere(delta[:, :6] > 1e-9)[0]
        raise ValueError(f"Policy arm limit exceeded at chunk row {row}, joint {column+1}: "
                         f"{converted[row, column]:.6f} rad not in [{lower[column]:.6f}, {upper[column]:.6f}]")
    if gripper_limit_mode == "error" and np.any(delta[:, 6] > 1e-9):
        raise ValueError(f"Policy gripper command exceeds [0, {maximum_width_m}] m")
    if arm_limit_mode == "error":
        clipped[:, :6] = converted[:, :6]
    if gripper_limit_mode == "error":
        clipped[:, 6] = converted[:, 6]
    return clipped, dict(
        arm_limit_mode=arm_limit_mode, gripper_limit_mode=gripper_limit_mode,
        arm_saturated_values=int(np.count_nonzero(delta[:, :6])) if arm_limit_mode == "clip" else 0,
        gripper_saturated_samples=int(np.count_nonzero(delta[:, 6])) if gripper_limit_mode == "clip" else 0,
        maximum_arm_saturation_rad=float(delta[:, :6].max()) if arm_limit_mode == "clip" else 0.,
        maximum_gripper_saturation_m=float(delta[:, 6].max()) if gripper_limit_mode == "clip" else 0.,
        raw_gripper_range_cm=[float(raw[:, 6].min()), float(raw[:, 6].max())])


def load_policy_initial(path, maximum_width_m):
    """Read only the first recorded observation; never load expert action targets."""
    with np.load(path, allow_pickle=False) as data:
        raw = np.asarray(data["state_raw"][0], dtype=float)
    if raw.shape != (7,) or not np.isfinite(raw).all():
        raise ValueError("Expected a finite seven-component initial dataset state")
    initial = np.r_[np.deg2rad(raw[:6]), raw[6] * .01]
    if not 0 <= initial[6] <= maximum_width_m:
        raise ValueError("Initial observed gripper width exceeds the configured travel")
    return dict(initial=initial, times=np.array([0.]), commands=np.empty((0, 7)),
                metadata=dict(source="first recorded observation only", source_angle_unit="degree",
                    source_width_unit="cm", width_unit_status="inferred from dataset and grasp geometry",
                    expert_action_targets_loaded=False, initial_state_only=True))


class WallXChunkPolicy:
    """Execute a configurable chunk prefix, then query fresh measured observations."""

    def __init__(self, client, kin, cfg, output, *, execute_steps=32, prompt=None,
                 dataset_name="lerobot/piper", arm_limit_mode="error", gripper_limit_mode="clip"):
        if type(execute_steps) is not int or not 1 <= execute_steps <= client.expected_horizon:
            raise ValueError("Executed chunk prefix must be between 1 and the trained horizon")
        self.client, self.kin, self.cfg = client, kin, cfg
        self.output = Path(output)
        self.execute_steps = execute_steps
        self.prompt = prompt or cfg["task_instruction"]
        self.dataset_name = dataset_name
        self.arm_limit_mode, self.gripper_limit_mode = arm_limit_mode, gripper_limit_mode
        self._buffer = np.empty((0, 7))
        self._cursor = 0
        self.requests = 0
        self.commands_selected = 0
        self.latencies = []

    @property
    def needs_observation(self):
        return self._cursor >= len(self._buffer)

    def command(self, seconds, *, joints=None, up_rgb=None, wrist_rgb=None):
        if self.needs_observation:
            observation = observation_from_sim(joints, up_rgb, wrist_rgb, self.prompt, self.dataset_name)
            index = self.requests
            self.requests += 1
            prefix = self.output / "policy_requests" / f"request_{index:06d}"
            record = dict(request_index=index, simulation_time_s=float(seconds),
                state_raw=observation["state"].tolist(), prompt=self.prompt,
                dataset_names=observation["dataset_names"], image_shapes={key: list(observation[key].shape)
                    for key in ("face_view", "left_wrist_view")}, wire_prefix=str(prefix.relative_to(self.output)))
            try:
                response = self.client.infer(observation, record_prefix=prefix)
                record.update(response=response, latency_s=self.client.last_latency_s)
                commands, checks = decode_actions(response["action"], self.kin.lower, self.kin.upper,
                    self.cfg["gripper_max_width_m"], horizon=self.client.expected_horizon,
                    arm_limit_mode=self.arm_limit_mode, gripper_limit_mode=self.gripper_limit_mode)
                self._buffer = commands[:self.execute_steps]
                self._cursor = 0
                self.latencies.append(self.client.last_latency_s)
                record.update(status="accepted", commands_rad_m=commands, checks=checks,
                              planned_execute_steps=len(self._buffer))
            except BaseException as error:
                # Preserve invalid numeric output in the exact binary frame;
                # JSON must remain valid and must not mask the validation error.
                if "response" in record:
                    record.pop("response")
                    record["response_note"] = "See exact response.msgpack for rejected response"
                record.update(status="failed", error=f"{type(error).__name__}: {error}",
                              latency_s=self.client.last_latency_s)
                raise
            finally:
                with (self.output / "policy_requests.jsonl").open("a") as stream:
                    stream.write(json.dumps(record, default=_json_value, allow_nan=False) + "\n")
        action_index = self._cursor
        value = self._buffer[action_index].copy()
        self._cursor += 1
        self.commands_selected += 1
        return value, dict(policy_request=self.requests-1, policy_action_index=action_index)

    def summary(self):
        return dict(endpoint=self.client.endpoint, server_metadata=self.client.metadata,
            prompt=self.prompt, dataset_name=self.dataset_name, state_action_keys=list(JOINT_KEYS),
            wire_units="absolute joint degrees and full gripper width centimetres",
            simulation_units="absolute joint radians and full gripper width metres",
            normalization="server state normalization and model action denormalization only",
            execute_steps=self.execute_steps, predicted_horizon=self.client.expected_horizon,
            control_hz=30, rtc=False, inference_requests=self.requests, commands_selected=self.commands_selected,
            latency_s=self.latencies, arm_limit_mode=self.arm_limit_mode,
            gripper_limit_mode=self.gripper_limit_mode, expert_actions_used=False,
            inference_wait_advances_simulation=False)
