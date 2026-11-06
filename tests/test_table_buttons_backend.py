"""CPU contract tests with measured cap/contact traces, never synthetic success."""
from copy import deepcopy
import sys

import numpy as np
import pytest

from pressb.runtime.contract import decode_image, validate_metadata
from pressb.runtime.simulation import SimulationService
from pressb.tasks.table_buttons.backend import TableButtonsBackend, load_config
from pressb.table_buttons.task import COLORS


LAYOUTS = {
    "near": [[.33, -.04], [.40, 0.], [.33, .04]],
    "medium": [[.30, -.07], [.42, 0.], [.30, .07]],
    "far": [[.25, -.10], [.44, 0.], [.25, .10]],
}


class TelemetryEnvironment:
    def __init__(self, config, trace=()):
        self.config, self.trace = config, trace
        self.closed = False
        self.total_ticks = 0

    def reset(self, spec):
        self.spec, self.tick_index = deepcopy(spec), 0
        self.measured = np.r_[self.config["scene"]["initial_joint_positions_rad"], .0003]
        self.sent = []

    def state(self):
        return self.measured.copy()

    def images(self):
        return {name: np.zeros((3, 4, 3), dtype=np.uint8) for name in ("up", "wrist")}

    def apply_action(self, command):
        self.sent.append(np.asarray(command).copy())

    def tick(self):
        empty = {"travels": dict.fromkeys(COLORS, 0.), "contact_forces": dict.fromkeys(COLORS, 0.)}
        sample = deepcopy(self.trace[self.tick_index] if self.tick_index < len(self.trace) else empty)
        self.tick_index += 1
        self.total_ticks += 1
        return sample

    def close(self):
        self.closed = True


def sample(*, color="red", depth=.0032, force=.2, blocking=False):
    return {"travels": {name: depth if name == color else 0. for name in COLORS},
            "contact_forces": {name: force if name == color else 0. for name in COLORS},
            "blocking_contacts": [{"actors": ["arm", "table"], "force_n": .1}] if blocking else []}


def backend(trace=(), **overrides):
    config = {"randomization": {"positions": False}, "max_episode_seconds": 2 / 30, **overrides}
    return TableButtonsBackend(config, environment_factory=lambda cfg: TelemetryEnvironment(cfg, trace))


def actions(node, count=7):
    values = np.r_[node.config["scene"]["initial_joint_positions_rad"], .03]
    return np.tile(values[:node.config["action_dim"]], (count, 1)).tolist()


def reset(node):
    return node.reset([0], 12, {"target_color": "red"})[0]


def test_joint6_uses_measured_state7_and_fixed_gripper_without_starting_isaac():
    node = backend()
    assert node.environment is None
    metadata = validate_metadata(node.describe(), "simulation")
    assert metadata["observation"]["state"]["units"] == ["rad"] * 6 + ["m"]
    assert metadata["action"]["units"] == ["rad"] * 6
    assert metadata["num_envs"] == 1 and metadata["execution"]["vectorized"] is False
    row = reset(node)
    assert len(row["observation"]["state"]) == 7
    assert row["observation"]["state"][-1] == .0003
    assert decode_image(row["observation"]["images"]["up"]).size == (4, 3)
    node.step([{"env_id": 0, "actions": actions(node, 1)}])
    assert node.environment.sent[0][6] == node.describe()["fixed_gripper_width_m"]


def test_joint7_executes_full_width_metres_without_degree_or_centimetre_conversion():
    node = backend(action_dim=7)
    reset(node)
    node.step([{"env_id": 0, "actions": actions(node, 1)}])
    np.testing.assert_allclose(node.environment.sent[0], actions(node, 1)[0], rtol=1e-6)
    assert "fixed_gripper_width_m" not in node.describe()


@pytest.mark.parametrize("depth,force", [(.0032, 0.), (0., .2)])
def test_displacement_without_finger_force_and_force_without_displacement_cannot_succeed(depth, force):
    node = backend([sample(depth=depth, force=force)] * 8)
    reset(node)
    row = node.step([{"env_id": 0, "actions": actions(node)}])[0]
    assert row["truncated"] and not row["terminated"]
    assert row["reward"] == 0 and not row["info"]["success"]
    assert row["info"]["physics_steps"] == 8
    assert row["info"]["elapsed_control_steps"] == 2
    assert row["discount"] == pytest.approx(.99**2)


def test_debounced_physical_press_ends_at_control_boundary_and_requires_reset():
    node = backend([sample()] * 8)
    reset(node)
    row = node.step([{"env_id": 0, "actions": actions(node)}])[0]
    assert row["terminated"] and not row["truncated"]
    assert row["reward"] == 1 and row["info"]["success"]
    assert row["info"]["physics_steps"] == 4
    assert row["info"]["elapsed_control_steps"] == 1
    assert row["info"]["physical_outcome"]["first_success_time_s"] == pytest.approx(4/120)
    with pytest.raises(ValueError, match="Reset"):
        node.step([{"env_id": 0, "actions": actions(node)}])
    reset(node)
    assert node.checker.first_success_time_s is None


def test_short_contact_pulse_does_not_satisfy_hold_duration():
    node = backend([sample()] * 3 + [sample(force=0.)] * 5)
    reset(node)
    row = node.step([{"env_id": 0, "actions": actions(node)}])[0]
    assert row["truncated"] and row["reward"] == 0


@pytest.mark.parametrize("failure", ["wrong_color", "blocking_contact"])
def test_contact_failure_after_success_in_same_interval_vetoes_latched_reward(failure):
    trace = [sample()] * 4
    trace[-1] = sample(color="yellow") if failure == "wrong_color" else sample(blocking=True)
    node = backend(trace, scene_overrides={"success": {"hold_seconds": 0.}})
    reset(node)
    row = node.step([{"env_id": 0, "actions": actions(node)}])[0]
    assert row["info"]["physical_outcome"]["success"]  # Latched at the first physics tick.
    assert row["terminated"] and not row["info"]["success"] and row["reward"] == 0
    assert row["info"]["physics_steps"] == 4  # All terminal-interval contacts were read.


def test_chunk_discount_accounts_for_actual_duration_of_successful_prefix():
    trace = [sample(force=0.)] * 4 + [sample()] * 4
    node = backend(trace)
    reset(node)
    row = node.step([{"env_id": 0, "actions": actions(node)}])[0]
    assert row["reward"] == pytest.approx(.99)
    assert row["discount"] == pytest.approx(.99**2)
    assert row["info"]["elapsed_control_steps"] == 2


def test_common_service_reset_step_retries_do_not_repeat_physics():
    node = backend([sample()] * 8)
    service = SimulationService(node)
    start = {"run_id": "test", "request_id": "reset", "env_ids": [0], "seed": 2,
             "options": {"target_color": "red"}}
    initial = service.reset(start)["items"][0]
    assert service.reset(start)["items"][0] == initial
    request = {"run_id": "test", "request_id": "step", "items": [{"env_id": 0,
        "episode_id": initial["episode_id"], "step_id": 0, "actions": actions(node)}]}
    result = service.step(request)
    assert service.step(request) == result and node.environment.total_ticks == 4
    service.close()
    assert node.environment.closed


def test_invalid_actions_rejected_without_advancing_physics_or_poisoning_backend():
    node = backend()
    reset(node)
    invalid = actions(node, 1)
    invalid[0][0] = 100
    with pytest.raises(ValueError, match="limit"):
        node.step([{"env_id": 0, "actions": invalid}])
    assert node.environment.total_ticks == 0
    node.step([{"env_id": 0, "actions": actions(node, 1)}])


def test_profile_and_target_sampling_reproducible_across_all_nine_combinations():
    calls = []
    def sampler(profile, seed):
        calls.append((profile, seed))
        return {"positions": dict(zip(COLORS, LAYOUTS[profile]))}
    node = TableButtonsBackend({}, layout_sampler=sampler)
    for profile in LAYOUTS:
        for color in COLORS:
            options = {"profile": profile, "target_color": color}
            first = node._reset_spec(42, options)
            assert first == node._reset_spec(42, options)
            assert first["task"] == f"Press the {color} button."
            assert first["profile"] == profile
            assert first["positions"] == dict(zip(COLORS, LAYOUTS[profile]))
    assert len(calls) == 18 and node.environment is None


def test_randomization_really_varies_profile_and_color_by_seed():
    sampler = lambda profile, seed: {"positions": dict(zip(COLORS, LAYOUTS[profile]))}
    node = TableButtonsBackend({}, layout_sampler=sampler)
    specs = [node._reset_spec(seed, {}) for seed in range(100)]
    assert {row["profile"] for row in specs} == set(LAYOUTS)
    assert {row["target_color"] for row in specs} == set(COLORS)


@pytest.mark.parametrize("config,match", [
    ({"num_envs": 64}, "num_envs=1"), ({"action_dim": 8}, "action_dim"),
    ({"fixed_gripper_width_m": .2}, "gripper"), ({"physics_dt": .01}, "divide"),
    ({"gamma": 1.1}, "gamma"), ({"randomization": {"profiles": ["unknown"]}}, "profiles"),
    ({"scene_overrides": {"success": {"min_contact_force_n": 0.}}}, "contact force"),
])
def test_invalid_configuration_rejected_before_creating_simulation(config, match):
    with pytest.raises(ValueError, match=match):
        load_config(config)


def test_identity_tracks_physical_outcome_and_action_semantics_but_not_gpu():
    a, b = backend(), backend(gpu=1)
    assert a.identity == b.identity
    assert a.identity != backend(action_dim=7).identity
    assert a.identity != backend(scene_overrides={"success": {"hold_seconds": .05}}).identity


def test_physics_error_latches_fault_and_is_never_retried(monkeypatch):
    node = backend()
    reset(node)
    def failed_tick():
        raise RuntimeError("physics failed after starting a tick")
    monkeypatch.setattr(node.environment, "tick", failed_tick)
    with pytest.raises(RuntimeError, match="physics failed"):
        node.step([{"env_id": 0, "actions": actions(node)}])
    with pytest.raises(RuntimeError, match="faulted"):
        node.step([{"env_id": 0, "actions": actions(node)}])
    with pytest.raises(RuntimeError, match="faulted"):
        reset(node)
    node.close()
    assert node.environment.closed


def test_camera_calibration_and_source_identity_are_bound_to_run():
    node = backend()
    assert "wrist_camera" not in node.config["scene"]  # Elevator registration remains the default.
    assert "runtime/isaac.py" in node.identity["source_identity"]
    for path in ("tasks/table_buttons/task.py", "tasks/table_buttons/scene.py",
                 "tasks/table_buttons/collision_policy.py", "runtime/scene.py", "runtime/workbench.py"):
        assert path in node.identity["source_identity"]
    changed = backend(scene_overrides={"up_camera": {"fx": 600.}})
    assert changed.identity["task_config_sha256"] != node.identity["task_config_sha256"]


def test_shared_camera_helpers_and_old_cli_reexports_load_without_isaac():
    import importlib.util
    from pathlib import Path
    from pressb.runtime.isaac import make_camera, set_intrinsics
    from pressb.runtime.recording import Video, write
    root = Path(__file__).resolve().parents[1]
    for script in ("collect_table_buttons", "eval_table_buttons_policy"):
        spec = importlib.util.spec_from_file_location(script, root / "scripts" / f"{script}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert (module.make_camera, module.set_intrinsics, module.Video, module.write) == (make_camera, set_intrinsics, Video, write)
    assert "isaacsim" not in sys.modules
