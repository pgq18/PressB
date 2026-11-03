"""CPU-only contracts for real joint-space table-button policy rollouts.

The remote policy speaks degrees/full gripper centimetres; Isaac commands use
radians/full gripper metres. No IK, expert trajectory, velocity interpolation,
or goal-derived action appears in this module.
"""
from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import numpy as np
from PIL import Image

from pressb.action_postprocessing import JointActionPostprocessor
from .collection import validate_raw_arrays
from .task import COLORS, task_instruction


SCALE = np.asarray([180.] * 6 + [8.])
POLICY_TO_SIM = np.asarray([np.pi / 180] * 6 + [.01])
FINGER_PAIR = ('/World/Piper/link7', '/World/Piper/link8')
DUAL_COLLISION_POLICY = dict(version='table_buttons_dual_contact_v1',
    allowed_task_contact_pairs=[list(FINGER_PAIR)],
    task_success='physical_target_press_without_wrong_or_blocking_contact',
    strict_collector_success='task_success_and_no_original_unexpected_contact')
STRICT_COLLISION_POLICY = dict(version='collector_strict_v1', allowed_task_contact_pairs=[],
    task_success='physical_target_press_without_any_original_unexpected_contact',
    strict_collector_success='identical_to_task_success')


class InvalidPolicyAction(ValueError):
    """A completed response contains unusable model actions (policy failure)."""


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collision_settings(requested=None):
    """Only an explicit frozen plan can permit normal inter-finger closure."""
    if requested is None or requested == STRICT_COLLISION_POLICY:
        return deepcopy(STRICT_COLLISION_POLICY)
    if requested != DUAL_COLLISION_POLICY:
        raise ValueError('Unknown collision policy; only the exact finger-pair exception is supported')
    return deepcopy(DUAL_COLLISION_POLICY)


def is_allowed_finger_closure(actors, collision_policy):
    return bool(collision_policy == DUAL_COLLISION_POLICY and len(actors) == 2
                and set(actors) == set(FINGER_PAIR))


def joint7(value):
    result = np.asarray(value)
    if result.dtype.kind not in "fiu" or result.shape != (7,) or not np.isfinite(result).all():
        raise ValueError("Expected seven finite real joint/gripper values")
    return result.astype(np.float64, copy=True)


def to_policy_units(rad_m):
    return joint7(rad_m) / POLICY_TO_SIM


def to_sim_units(deg_cm):
    return joint7(deg_cm) * POLICY_TO_SIM


def command_limits(dof_limits):
    """Derive full-width bounds from the actual eight articulation DOFs."""
    bounds = np.asarray(dof_limits, dtype=float)
    if bounds.shape != (8, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("Expected finite increasing physical limits for joint1..8")
    lower_width = max(0., 2 * bounds[6, 0], -2 * bounds[7, 1])
    upper_width = min(2 * bounds[6, 1], -2 * bounds[7, 0])
    if upper_width <= lower_width:
        raise ValueError("Physical finger limits do not permit symmetric gripper motion")
    return np.r_[bounds[:6, 0], lower_width], np.r_[bounds[:6, 1], upper_width]


class JointPolicyCommands:
    """Explicit units -> physical limit clamp -> optional arm-only smoothing."""
    def __init__(self, lower, upper, initial, *, smoothing="none", alpha=.8):
        self.lower, self.upper = joint7(lower), joint7(upper)
        if np.any(self.lower >= self.upper):
            raise ValueError("Invalid joint7 physical limits")
        self.postprocessor = JointActionPostprocessor(smoothing, alpha)
        self.postprocessor.reset(np.clip(joint7(initial), self.lower, self.upper))
        self.clipped_by_dimension = np.zeros(7, dtype=int)
        self.clipped_actions = 0
        self.total_actions = 0

    def apply(self, raw):
        requested = joint7(raw)
        converted = to_sim_units(requested)
        clipped = np.clip(converted, self.lower, self.upper)
        mask = clipped != converted
        applied = self.postprocessor.apply(clipped)
        self.clipped_by_dimension += mask.astype(int)
        self.clipped_actions += int(mask.any())
        self.total_actions += 1
        return applied, dict(raw_command_deg_cm=requested.tolist(), converted_rad_m=converted.tolist(),
            clipped_rad_m=clipped.tolist(), applied_rad_m=applied.tolist(),
            clipped_dimensions=np.flatnonzero(mask).tolist())

    def summary(self):
        return dict(total_actions=self.total_actions, clipped_actions=self.clipped_actions,
            clipped_by_dimension=self.clipped_by_dimension.tolist(), lower_rad_m=self.lower.tolist(),
            upper_rad_m=self.upper.tolist(), smoothing=self.postprocessor.summary(),
            clipping_rule="physical joint limits only; any clamp is recorded, not automatically a failure")


def png_base64(rgb):
    rgb = np.asarray(rgb)
    if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
        raise ValueError("Policy cameras must provide RGB uint8 HWC 480x640")
    stream = BytesIO()
    Image.fromarray(rgb).save(stream, format="PNG", compress_level=1)
    return base64.b64encode(stream.getvalue()).decode("ascii")


class HTTPJointPolicy:
    """Synchronous JSON RPC; callers do not advance simulation while waiting."""
    def __init__(self, endpoint, model_id, *, timeout=180.):
        if not str(endpoint).startswith(("http://", "https://")):
            raise ValueError("Expected an HTTP(S) inference endpoint")
        self.endpoint = str(endpoint).rstrip("/")
        self.model_id, self.timeout = str(model_id), float(timeout)
        self.model = None

    def call(self, route, body=None):
        payload = None if body is None else json.dumps(body, allow_nan=False).encode()
        request = Request(self.endpoint + route, data=payload,
                          headers={"Content-Type": "application/json"}, method="GET" if body is None else "POST")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except HTTPError as error:
            if error.code == 422:
                details = json.loads(error.read())
                if details.get('error_kind') == 'invalid_policy_action':
                    raise InvalidPolicyAction(details.get('error', 'Server returned invalid model actions')) from error
            raise
        if not isinstance(result, dict):
            raise RuntimeError("Inference server did not return a JSON object")
        return result

    def load(self):
        body = self.call("/load", dict(model_id=self.model_id))
        model = body.get("model", body)
        if model.get("model_id") != self.model_id or not model.get("model_sha256"):
            raise RuntimeError("Loaded model identity/hash missing or mismatched")
        self.model = deepcopy(model)
        return deepcopy(self.model)

    def reset(self, episode_id):
        return self.call("/reset", dict(model_id=self.model_id, episode_id=episode_id))

    def predict(self, *, episode_id, request_id, instruction, state, images, noise_seed):
        sample = dict(episode_id=episode_id, request_id=request_id, instruction=instruction,
                      state=joint7(state).tolist(), images={name: png_base64(images[name]) for name in ("up", "wrist")},
                      noise_seed=int(noise_seed))
        started = time.perf_counter()
        body = self.call("/predict", dict(model_id=self.model_id, samples=[sample]))
        elapsed = time.perf_counter() - started
        model = body.get("model", {})
        if self.model is None or model.get("model_id") != self.model_id or model.get("model_sha256") != self.model["model_sha256"]:
            raise RuntimeError("Inference model changed during evaluation")
        predictions = body.get("predictions", [])
        if len(predictions) != 1:
            raise RuntimeError("Expected exactly one prediction per request sample")
        result = predictions[0]
        if (result.get("episode_id") != episode_id or result.get("request_id") != request_id
                or result.get("noise_seed") != noise_seed):
            raise RuntimeError("Inference response identity or noise seed mismatch")
        try:
            actions = np.asarray(result["actions"], dtype=np.float64)
            normalized = np.asarray(result["normalized_actions"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as error:
            raise InvalidPolicyAction("Response has no numeric joint7 action chunk") from error
        if (actions.shape != (7, 7) or normalized.shape != (7, 7)
                or not np.isfinite(actions).all() or not np.isfinite(normalized).all()):
            raise InvalidPolicyAction("Expected finite seven-step seven-channel model actions")
        if not np.allclose(actions, normalized * SCALE, atol=2e-5, rtol=2e-6):
            raise RuntimeError("Server actions do not invert the declared joint7 normalization")
        evidence = {key: deepcopy(result[key]) for key in
                    ('normalized_state', 'initial_noise_sha256', 'preprocessed_image_sha256') if key in result}
        if 'normalized_state' in evidence and not np.allclose(joint7(evidence['normalized_state']),
                                                              sample['state'] / SCALE, atol=2e-7, rtol=2e-6):
            raise RuntimeError("Server normalized a different state than this observation")
        return actions, dict(model=deepcopy(model), raw_actions_deg_cm=actions.tolist(),
            normalized_actions=normalized.tolist(), round_trip_seconds=elapsed,
            inference_seconds=body.get("inference_seconds"), episode_id=episode_id,
            request_id=request_id, noise_seed=int(noise_seed), instruction=instruction,
            state_deg_cm=sample["state"], **evidence)


def strict_outcome(result, *, wrong_press=None, wrong_contact=False, unexpected_contacts=(), model_error=None):
    """Apply collector-equivalent failures even after the success latch fired."""
    if model_error is not None:
        return False, 'invalid_policy_action'
    if unexpected_contacts:
        return False, 'unexpected_contact'
    if wrong_contact:
        return False, 'non_target_finger_contact'
    if wrong_press is not None:
        return False, 'wrong_button'
    if result and result['success']:
        return True, 'success'
    return False, 'timeout'


def load_cell(plan_path, cell_id):
    """Load immutable per-cell plans; single/multitask cells share layout files."""
    plan = json.loads(Path(plan_path).read_text())
    if plan.get("schema") != "pressb.table_buttons.policy_eval_plan.v1":
        raise ValueError("Unexpected policy evaluation plan schema")
    selected = [cell for cell in plan["cells"] if cell["cell_id"] == cell_id]
    if len(selected) != 1:
        raise ValueError("Evaluation plan must contain exactly one matching cell")
    cell = selected[0]
    defaults = plan["defaults"]
    collision_settings(defaults.get('collision_policy'))
    if defaults["fps"] != 30 or abs(defaults["physics_dt"] - 1 / 120) > 1e-12 or defaults["execute_steps"] != 7:
        raise ValueError("Evaluation requires seven actions per request, 30 Hz control, 120 Hz physics")
    if cell["instruction"] != task_instruction(cell["target_color"]) or cell["spacing_profile"] not in ("near", "medium", "far"):
        raise ValueError("Cell task identity is inconsistent")
    ids = [row["episode_index"] for row in cell["episodes"]]
    if ids != list(range(len(ids))) or len({row["seed"] for row in cell["episodes"]}) != len(ids):
        raise ValueError("Episode indices/seeds must be distinct and ordered")
    for row in cell["episodes"]:
        if type(row["seed"]) is not int or row["seed"] < 10000 or type(row["policy_seed"]) is not int:
            raise ValueError("Held-out layouts require new seeds >=10000 and explicit paired policy seeds")
    return plan, cell


def episode_config(plan, cell, row):
    """Verify scene config against common collection settings and paired layout."""
    for name in ('layout', 'effective_config', 'preflight'):
        if name + '_sha256' in row and file_sha(row[name + '_file']) != row[name + '_sha256']:
            raise ValueError(f"Frozen {name} file changed after evaluation planning")
    layout = json.loads(Path(row["layout_file"]).read_text())
    actual = json.loads(Path(row["effective_config_file"]).read_text())
    expected = deepcopy(plan["config"])
    expected["position_sampling"]["spacing_profile"] = cell["spacing_profile"]
    expected["task_color"] = cell["target_color"]
    expected["task_instruction"] = cell["instruction"]
    for color in COLORS:
        expected["buttons"][color]["center_xy"] = layout["positions"][color]
    expected["press_approaches"] = layout["approaches"]
    expected["sampling_domain"] = layout["domain"]
    if actual != expected:
        raise ValueError("Effective scene differs from frozen collector configuration and paired layout")
    return actual, layout


def validate_execution_arrays(frames, physics, *, fps=30, physics_dt=1 / 120):
    if len(frames["sim_time"]) == 1:
        if (np.asarray(frames["state"]).shape != (1, 7) or np.asarray(frames["action"]).shape != (1, 7)
                or np.asarray(physics["q_actual"]).shape != (1, 8)
                or np.asarray(physics["q_target"]).shape != (1, 8)
                or frames["sim_time"][0] != 0 or physics["time_s"].tolist() != [0.]):
            raise ValueError("Invalid zero-action episode arrays")
        for array in (*frames.values(), *physics.values()):
            if not np.isfinite(array).all():
                raise ValueError("Nonfinite zero-action episode arrays")
        return
    validate_raw_arrays(frames, physics, fps=fps, physics_dt=physics_dt)


def summarize(output, cell):
    reports = []
    for row in cell["episodes"]:
        path = Path(output) / "episodes" / f"episode_{row['episode_index']:06d}" / "report.json"
        if path.exists():
            report = json.loads(path.read_text())
            if report.get("evaluation_status") != "complete" or report["cell_id"] != cell["cell_id"] or report["seed"] != row["seed"]:
                raise ValueError(f"Existing episode identity/completion mismatch: {path}")
            reports.append(report)
    counts = {}
    for report in reports:
        counts[report["stop_reason"]] = counts.get(report["stop_reason"], 0) + 1
    successes = sum(report["success"] for report in reports)
    strict_successes = sum(report.get('strict_collector_success', report['success']) for report in reports)
    return dict(cell_id=cell["cell_id"], model_id=cell["model_id"], spacing_profile=cell["spacing_profile"],
                target_color=cell["target_color"], planned=len(cell["episodes"]), completed=len(reports),
                successes=successes, failures=len(reports) - successes,
                success_rate=successes / len(reports) if reports else None,
                task_successes=successes, task_success_rate=successes / len(reports) if reports else None,
                strict_collector_successes=strict_successes,
                strict_collector_success_rate=strict_successes / len(reports) if reports else None,
                stop_reasons=counts, complete=len(reports) == len(cell["episodes"]))
