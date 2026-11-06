"""CPU coverage of vector PAP configuration and transition timing semantics."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from pressb.control.joint_actions import JointActionPostprocessor
from pressb.tasks.pap.backend import ROOT
from pressb.tasks.pap.vector import PAPVectorBackend, _path_rows, load_config, renderer_profile, _encode_png
from pressb.runtime.image_transport import walloss_resize_spec, resize_for_transport
from pressb.runtime.contract import decode_image


def base_config():
    return json.loads((ROOT / "configs/rl_nodes/pap_simulation.json").read_text())["backend"]["config"]


def test_shared_world_preserves_native_camera_aspect():
    config = load_config({**base_config(), "num_envs": 64, "render_resolution": [320, 240]})
    assert config["render_resolution"] == [320, 240]
    assert config["scene"]["up_camera"]["resolution"] == [640, 480]
    assert config["scene"]["wrist_camera"]["resolution"] == [640, 480]
    assert config["reset_render_frames"] == config["scene"]["rendering"]["warmup_frames"]
    assert config["render_subframes"] == config["scene"]["rendering"]["subframes_per_frame"]
    with pytest.raises(ValueError, match="aspect ratio"):
        load_config({**base_config(), "render_resolution": [256, 256]})


def test_fast_renderer_is_opt_in_and_preserves_camera_geometry():
    original = load_config(base_config())
    fast = load_config({**base_config(), "renderer_profile": "rl_fast"})
    assert original["renderer_profile"] == "pap_calibrated"
    assert renderer_profile(original["renderer_profile"]) == {"anti_aliasing": 1, "history_controls": {}}
    assert renderer_profile(fast["renderer_profile"])["anti_aliasing"] == 3
    assert renderer_profile("rl_fast")["reset_dlss_history"] is True
    assert renderer_profile("rl_fast")["history_controls"]["/rtx-transient/dldenoiser/enabled"] is True
    assert fast["observation_render_frames"] == 2
    assert fast["reset_render_frames"] == 2
    assert fast["render_subframes"] == 1
    assert fast["render_resolution"] == original["render_resolution"] == [640, 480]
    assert fast["scene"] == original["scene"]
    overridden = load_config({**base_config(), "renderer_profile": "rl_fast", "reset_render_frames": 8})
    assert overridden["reset_render_frames"] == 8
    with pytest.raises(ValueError, match="renderer_profile"):
        load_config({**base_config(), "renderer_profile": "unknown"})


def test_shared_world_rejects_multigpu_and_overlapping_room_floors():
    with pytest.raises(ValueError, match="one GPU"):
        load_config({**base_config(), "gpu_ids": [0, 1]})
    with pytest.raises(ValueError, match="separate"):
        load_config({**base_config(), "spacing_m": 2.})


def test_tensor_mapping_uses_actual_paths():
    desired = [f"/World/envs/env_{i}/Piper" for i in range(12)]
    physical = sorted(desired)
    rows = _path_rows(physical, desired)
    assert [physical[i] for i in rows] == desired
    with pytest.raises(RuntimeError, match="missing"):
        _path_rows(physical[:-1], desired)


def fake_backend():
    backend = object.__new__(PAPVectorBackend)
    backend.num_envs = 2
    backend._check = lambda: None
    backend.config = dict(horizon=32, physics_ticks_per_action=1, physics_dt=.1,
        gamma=.9, terminate_on_drop=False, scene={"table_height": .76},
        drop_below_table_m=.05, reward={"success": 1., "failure": -1.})
    backend.ready = np.ones(2, dtype=bool)
    backend.lower, backend.upper = np.full(7, -1.), np.ones(7)
    backend.ticks = np.zeros(2, dtype=np.int64)
    backend.last_sent = np.zeros((2, 7))
    backend.dofs = np.arange(8)
    backend.commands = np.zeros((2, 8))
    backend.arm_rows = np.array([1, 0])
    backend.object_rows = np.array([0, 1])
    backend.postprocessors = [JointActionPostprocessor() for _ in range(2)]
    backend.contact_counts = np.ones((2, 2), dtype=np.int64)
    backend.finger_contact = np.zeros(2, dtype=bool)
    backend._states = lambda: np.zeros((2, 7))
    backend.reset_specs = [{"task": "pick red cube", "model_inputs": {}} for _ in range(2)]
    backend._guard = SimpleNamespace(check=lambda: None)
    backend.world = SimpleNamespace(current_time=0., count=0)
    def step(render=False):
        backend.world.count += 1
        backend.world.current_time += .1
    backend.world.step = step
    backend.arms = SimpleNamespace(set_joint_position_targets=lambda *args: None)
    backend.bodies = SimpleNamespace(
        get_world_poses=lambda: (np.array([[backend.world.count, 0., 1.], [0., 0., 1.]]),
                                 np.tile([1., 0., 0., 0.], (2, 1))),
        get_velocities=lambda: np.zeros((2, 6)))
    backend.outcomes = [SimpleNamespace(update=lambda *a, **kw: {"success": True}),
                        SimpleNamespace(update=lambda *a, **kw: {"success": False})]
    backend.max_ticks = 3
    backend.timings = dict(physics_seconds=0., world_ticks=0, step_seconds=0.)
    backend._observations = lambda ids: {i: {"captured_at_world_tick": backend.world.count,
                                           "episode_tick": int(backend.ticks[i])} for i in ids}
    return backend


def actions(ids=(0, 1), prefix=4):
    return [dict(env_id=i, actions=np.zeros((prefix, 7)).tolist()) for i in ids]


def test_subset_and_mismatched_prefix_rejected_before_any_tick():
    backend = fake_backend()
    with pytest.raises(ValueError, match="every active"):
        backend.step(actions(ids=(0,)))
    commands = actions()
    commands[1]["actions"].pop()
    with pytest.raises(ValueError, match="equal action"):
        backend.step(commands)
    assert backend.world.count == 0


def test_terminal_rgb_and_object_snapshot_precede_other_slots_continuation():
    backend = fake_backend()
    rows = backend.step(actions(ids=(1, 0)))
    assert [row["env_id"] for row in rows] == [1, 0]
    live, success = rows
    assert success["terminated"] and not success["truncated"]
    assert success["observation"]["captured_at_world_tick"] == 1
    assert success["info"]["object_position_world"] == [1., 0., 1.]
    assert success["info"]["physics_steps"] == 1
    assert success["reward"] == 1.
    assert success["discount"] == .9
    assert live["truncated"] and not live["terminated"]
    assert live["observation"]["captured_at_world_tick"] == 3
    assert live["info"]["physics_steps"] == 3
    assert live["discount"] == pytest.approx(.9**3)
    assert backend.world.count == 3
    assert backend.ticks.tolist() == [1, 3]


def test_success_requires_both_finger_contact_histories():
    backend = fake_backend()
    backend.contact_counts[0, 1] = 0
    row = backend.step(actions())[0]
    assert row["truncated"]
    assert not row["info"]["success"]
    assert row["reward"] == 0.


def test_state_only_terminal_keeps_exact_state_and_reward_without_render():
    full = fake_backend().step(actions())
    backend = fake_backend()
    backend._states = lambda: np.tile([backend.world.count, 0., 0., 0., 0., 0., 0.], (2, 1))
    backend._observations = lambda ids: pytest.fail("All slots end; no RGB should be captured")
    reduced = backend.step(actions(), terminal_observation_mode="state_only_if_no_bootstrap")
    for expected, actual in zip(full, reduced):
        assert actual["terminal_observation_omitted"] is True
        assert actual["observation"]["images"] == {}
        assert actual["observation"]["state"][0] == expected["info"]["episode_physics_steps"]
        assert actual["observation"]["control_state"] == [0.] * 7
        assert actual["observation"]["task"] == "pick red cube"
        for field in ("reward", "discount", "terminated", "truncated", "info"):
            assert actual[field] == expected[field]


def test_state_only_mode_preserves_rgb_for_live_slots_and_rejects_bad_mode_before_step():
    backend = fake_backend()
    with pytest.raises(ValueError, match="terminal_observation_mode"):
        backend.step(actions(), terminal_observation_mode="skip_everything")
    assert backend.world.count == 0
    rows = backend.step(actions(prefix=1), terminal_observation_mode="state_only_if_no_bootstrap")
    assert rows[0]["terminal_observation_omitted"] is True
    assert "terminal_observation_omitted" not in rows[1]
    assert rows[1]["observation"]["captured_at_world_tick"] == 1


def fused_backend():
    backend = fake_backend()
    settings = load_config(base_config())
    settings.update({key: value for key, value in backend.config.items() if key != "scene"})
    settings.update(observation_render_frames=2, reset_render_frames=48)
    backend.config = settings
    backend.render_calls, backend.reset_calls = [], []
    backend._states = lambda: np.c_[backend.ticks, np.zeros((2, 6))]
    def reset(ids, specs, *, capture_images):
        assert capture_images is False
        before = backend.ticks.copy()
        backend.reset_calls.append(deepcopy(ids))
        backend.ticks[ids] = 0
        backend.ready[ids] = True
        backend.last_sent[ids] = 0.
        for i in ids:
            backend.reset_specs[i] = deepcopy(specs[i])
        for i in set(range(2))-set(ids):
            assert backend.ticks[i] == before[i]
        observations = backend._state_observations(ids)
        return [dict(env_id=i, observation=observations[i], info={"reset": deepcopy(specs[i])}) for i in ids]
    def observe(ids, *, render_frames=None):
        backend.render_calls.append({"ids": ids, "frames": render_frames, "ticks": backend.ticks.tolist()})
        observations = backend._state_observations(ids)
        for row in observations.values():
            row["images"] = {"up": "current", "wrist": "current"}
        return observations
    backend._reset, backend._observations = reset, observe
    return backend


def test_fused_mixed_terminal_preserves_old_state_and_renders_survivor_and_reset_together():
    backend = fused_backend()
    rows = backend.step_and_reset(actions(prefix=1), reset_seed=42, reset_options={}, max_active_envs=2)
    terminal, survivor = rows["items"]
    fresh, = rows["reset_items"]
    assert terminal["observation"]["state"][0] == 1
    assert terminal["observation"]["images"] == {}
    assert terminal["terminal_observation_omitted"] is True
    assert fresh["observation"]["state"][0] == 0
    assert fresh["observation"] is not terminal["observation"]
    assert survivor["observation"]["state"][0] == 1
    assert survivor["observation"]["images"] == fresh["observation"]["images"]
    assert backend.render_calls == [{"ids": [0, 1], "frames": 48, "ticks": [0, 1]}]
    assert backend.reset_calls == [[0]]
    assert backend.world.count == 1
    assert backend.timings["fused_steps"] == 1


@pytest.mark.parametrize("budget, expected_reset", [(0, []), (1, [1]), (2, [1, 0])])
def test_fused_all_terminal_success_and_timeout_reset_in_command_order(budget, expected_reset):
    backend = fused_backend()
    rows = backend.step_and_reset(actions(ids=(1, 0)), reset_seed=42, reset_options={}, max_active_envs=budget)
    assert rows["items"][0]["truncated"]
    assert rows["items"][1]["terminated"]
    assert [row["observation"]["state"][0] for row in rows["items"]] == [3, 1]
    assert [row["env_id"] for row in rows["reset_items"]] == expected_reset
    assert backend.reset_calls == ([expected_reset] if expected_reset else [])
    assert len(backend.render_calls) == bool(expected_reset)
    assert backend.world.count == 3


def test_fused_no_terminal_renders_once_without_reset_and_honors_observation_frames():
    backend = fused_backend()
    backend.contact_counts[:] = 0
    rows = backend.step_and_reset(actions(prefix=1), reset_seed=42, reset_options={}, max_active_envs=0)
    assert rows["reset_items"] == []
    assert backend.reset_calls == []
    assert backend.render_calls == [{"ids": [0, 1], "frames": 2, "ticks": [1, 1]}]
    assert all(not row.get("terminal_observation_omitted") for row in rows["items"])


@pytest.mark.parametrize("changes", [
    {"reset_options": {"unknown": True}}, {"reset_options": {"center_xy": [float("nan"), 0.]}},
    {"reset_options": []}, {"reset_options": {"by_env": []}}, {"reset_seed": -1},
    {"max_active_envs": True}, {"max_active_envs": 3},
])
def test_fused_invalid_reset_rejected_before_physics(changes):
    backend = fused_backend()
    kwargs = {"reset_seed": 42, "reset_options": {}, "max_active_envs": 2, **changes}
    with pytest.raises((TypeError, ValueError)):
        backend.step_and_reset(actions(), **kwargs)
    assert backend.world.count == 0
    assert backend.reset_calls == backend.render_calls == []


def test_fused_reset_failure_never_returns_partly_rendered_success():
    backend = fused_backend()
    def fail(*args, **kwargs):
        raise RuntimeError("reset failed")
    backend._reset = fail
    with pytest.raises(RuntimeError, match="reset failed"):
        backend.step_and_reset(actions(), reset_seed=42, reset_options={}, max_active_envs=2)
    assert backend.render_calls == []


def test_producer_resize_preserves_exact_model_pixels_and_legacy_png():
    from PIL import Image
    spec = walloss_resize_spec(256)
    source = np.random.default_rng(42).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    original = source.copy()
    encoded = _encode_png(source, spec)
    assert encoded["preprocessing"] == {"spec": spec, "original_size": [640, 480]}
    assert np.array_equal(np.asarray(decode_image(encoded)), np.asarray(resize_for_transport(Image.fromarray(source), spec)))
    assert np.array_equal(source, original)
    legacy = _encode_png(source)
    assert "preprocessing" not in legacy
    assert np.array_equal(np.asarray(decode_image(legacy)), source)
    loaded = load_config({**base_config(), "image_preprocessing": spec})
    assert loaded["image_preprocessing"] == spec
    assert loaded["image_preprocessing"] is not spec
    assert loaded["render_resolution"] == [640, 480]
    with pytest.raises(ValueError, match="specification"):
        load_config({**base_config(), "image_preprocessing": {"kind": "approximate"}})
