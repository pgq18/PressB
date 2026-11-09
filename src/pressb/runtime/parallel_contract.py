"""Task and robot hooks for the shared parallel simulation backend.

Only the runtime may step physics, render, maintain episode clocks, or schedule
slots. Task hooks author a single template, randomize task objects, and interpret
measured physics. Hooks must not advance the world's clock. Robot drivers own
joint mapping, IK, interpolation and filtering, independently of task outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SceneTemplate:
    """Authored scene before cloning; no per-camera render products are needed.

    ``world`` is created by the scene authoring helper, before any episode. The
    shared engine owns it thereafter. All cloned content must be under root_path
    and the physics scene must be outside that subtree. ``data`` carries only
    task/robot authoring information, not an independent simulation backend.
    """
    world: Any
    stage: Any
    robot_path: str
    camera_paths: dict[str, str]
    data: dict = field(default_factory=dict)
    root_path: str = "/World"


class ParallelTask:
    """Base hook defaults. Subclasses must supply config, execution and metadata.

    ``execution`` contains: num_envs, gpu, physics_dt, control_hz, horizon,
    max_episode_seconds, settling_ticks, spacing_m, render_resolution,
    render_subframes, observation_render_frames, reset_render_frames,
    cpu_threads, png_workers, log_dir, gamma. Optional settings include
    renderer_profile (calibrated/rl_fast), image_preprocessing,
    image_output_resolution, terminal_boundary (physics_tick/control_interval),
    timeout_reason and headless. Configuration is normalized before Kit starts.

    The engine context exposes world/stage, template, offsets/prefixes,
    robot_paths, camera_names, num_envs, simulation_view, robot,
    relocate(env_id, template_path), rigid_bodies(template_paths), and
    publish_transforms(). backend adds ready, ticks, reset_specs and timings.
    ``rigid_bodies`` returns (TensorRigidBodies, row_indices[N, object_count]).
    """
    def metadata(self) -> dict:
        raise NotImplementedError

    def build_template(self) -> SceneTemplate:
        raise NotImplementedError

    def create_robot(self, engine):
        """Return a driver implementing the methods documented below."""
        raise NotImplementedError

    def bind(self, engine):
        self.engine = engine

    def after_settle(self):
        """Cache measured task reset state after startup-only settling."""

    def sample_reset(self, env_id, seed, options) -> dict:
        """Pure validation/sampling; runtime has already merged by_env options."""
        raise NotImplementedError

    def reset_task(self, env_ids, specs) -> dict:
        """Zero-tick selected-object reset; return info mapping keyed by env ID."""
        return {i: {} for i in env_ids}

    def before_tick(self, env_ids):
        """Clear transient per-tick contacts, without touching episode state."""

    def on_contact(self, env_id, actor_paths, contacts):
        """Paths use the original template root; contacts are native contact data."""

    def after_tick(self, env_ids) -> dict:
        """Return per-ID outcome dicts from measured physics.

        Keys: reward (undiscounted), success, terminated, truncated, reason, info.
        For control_interval termination the runtime calls this every tick but
        accepts rewards/terminal flags only at the end of that control interval.
        Task updates must continue after an early success latch to catch failures.
        Runtime applies time limits and actual-duration reward/discount factors.
        """
        return {i: {} for i in env_ids}

    def observation_fields(self, env_ids) -> dict:
        """Per-ID task, model_inputs and task_features (no images/state)."""
        return {i: {"task": self.engine.backend.reset_specs[i]["task"],
                    "model_inputs": self.engine.backend.reset_specs[i].get("model_inputs", {}),
                    "task_features": []} for i in env_ids}

    def diagnostic_state(self, env_ids) -> dict:
        """Per-ID measured task state used for indexed-reset peer invariants."""
        return {i: {} for i in env_ids}

    def close(self):
        pass


class ParallelRobot:
    """Robot driver interface; never steps physics or owns episode scheduling.

    Constructed after world.reset and tensor-view creation. Constructor sets the
    nominal initial pose/targets. cache_initial_state runs after startup settling.
    reset may use specs[i]['initial_state_joint_rad_m']; otherwise restores cache.
    begin_action returns the actual high-level command per ID for action logs.
    before_tick receives a zero-based interpolation index in a control interval.
    observations returns {ID: {'state': [...], 'control_state': [...]}}.
    """
    def cache_initial_state(self):
        raise NotImplementedError

    def reset(self, env_ids, specs):
        raise NotImplementedError

    def begin_action(self, env_ids, actions) -> dict:
        raise NotImplementedError

    def before_tick(self, env_ids, substep, ticks_per_action):
        raise NotImplementedError

    def observations(self, env_ids) -> dict:
        raise NotImplementedError

    def diagnostic_state(self, env_ids) -> dict:
        raise NotImplementedError

    def close(self):
        pass
