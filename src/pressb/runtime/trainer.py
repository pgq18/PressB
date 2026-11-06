"""Task-independent chunk SAC orchestration for the three-node interface.

Only this process owns replay and trainable parameters. One I/O worker waits on
inference/simulation while the calling thread updates SAC; actor reads and
optimizer writes therefore never race. Physics requests are never retried.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import asdict, dataclass, field
import copy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Any
import uuid

import numpy as np
import torch

from .learner import (CHECKPOINT_VERSION, SOURCE_COMMIT, LearnerConfig,
                                    ReplayBuffer, SACLearner, _Actor)
from .client import NodeClient
from .contract import negotiate, validate_observation


@dataclass
class EndpointConfig:
    inference: str = "ws://127.0.0.1:19871"
    simulation: str = "http://127.0.0.1:19870"
    timeout: float = 120.
    token: str | None = None
    max_message_bytes: int = 64 * 1024 * 1024
    image_transport: str = "legacy"
    preprocess_images: bool = False
    image_workers: int = 1


@dataclass
class RolloutConfig:
    max_transitions: int = 100_000
    eval_episodes: int = 100
    execution_horizon: int | None = None
    num_envs: int | None = None
    reset_options: dict = field(default_factory=dict)
    recent_window: int = 100
    deterministic_eval: bool = True
    action_bounds: str = "error"
    terminal_observation_mode: str = "full"
    fused_step_reset: bool = False


@dataclass
class UpdateConfig:
    replay_capacity: int = 100_000
    learning_starts: int = 2_000
    utd: float = 1.
    max_backlog: int = 1_000
    max_updates_per_wait: int = 16
    progressive_exploration_steps: int = 0


@dataclass
class CheckpointConfig:
    interval: int = 100_000
    # None retains every checkpoint; positive integers explicitly opt into rotation.
    keep: int | None = None
    include_replay: bool = True
    resume: str | None = None


@dataclass
class FrozenDSRLConfig:
    """A fixed noise policy supplying the action-residual learner's base chunk."""

    checkpoint: str
    deterministic: bool = True
    checkpoint_sha256: str | None = None

    def __post_init__(self):
        if not isinstance(self.checkpoint, str) or not self.checkpoint:
            raise ValueError("frozen_dsrl.checkpoint must be a nonempty path")
        if type(self.deterministic) is not bool:
            raise ValueError("frozen_dsrl.deterministic must be boolean")
        digest = self.checkpoint_sha256
        if digest is not None and (not isinstance(digest, str) or len(digest) != 64
                                   or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("frozen_dsrl.checkpoint_sha256 must be a lowercase SHA256 digest")


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "pressb-rl-nodes"
    entity: str | None = None
    name: str | None = None
    mode: str = "online"
    tags: list[str] = field(default_factory=list)
    log_interval_seconds: float = 5.
    init_timeout_seconds: float = 15.


@dataclass
class TrainerConfig:
    method: str = "dsrl"
    mode: str = "train"
    device: str = "cpu"
    seed: int = 42
    endpoints: EndpointConfig = field(default_factory=EndpointConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    learner: dict = field(default_factory=dict)
    update: UpdateConfig = field(default_factory=UpdateConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    frozen_dsrl: FrozenDSRLConfig | None = None
    wandb: WandbConfig = field(default_factory=WandbConfig)
    output_dir: str = "outputs/rl_nodes_run"

    def __post_init__(self):
        for name, kind in (("endpoints", EndpointConfig), ("rollout", RolloutConfig),
                           ("update", UpdateConfig), ("checkpoint", CheckpointConfig),
                           ("wandb", WandbConfig)):
            value = getattr(self, name)
            if isinstance(value, dict):
                setattr(self, name, kind(**value))
            elif not isinstance(value, kind):
                raise ValueError(f"{name} must be a configuration object")
        if self.rollout.action_bounds not in ("error", "clip"):
            raise ValueError("rollout.action_bounds must be error or clip")
        if self.endpoints.image_transport not in ("legacy", "auto"):
            raise ValueError("endpoints.image_transport must be legacy or auto")
        if type(self.endpoints.preprocess_images) is not bool:
            raise ValueError("endpoints.preprocess_images must be boolean")
        if self.method not in ("dsrl", "action_residual", "base"):
            raise ValueError("method must be dsrl, action_residual or base")
        if self.frozen_dsrl is not None:
            if isinstance(self.frozen_dsrl, dict):
                self.frozen_dsrl = FrozenDSRLConfig(**self.frozen_dsrl)
            elif not isinstance(self.frozen_dsrl, FrozenDSRLConfig):
                raise ValueError("frozen_dsrl must be a configuration object or null")
            self.frozen_dsrl.__post_init__()
            if self.method != "action_residual":
                raise ValueError("frozen_dsrl requires method=action_residual")
        if self.mode not in ("train", "eval") or (self.method == "base" and self.mode != "eval"):
            raise ValueError("mode must be train or eval; base supports evaluation only")
        for name, value, minimum in (
                ("seed", self.seed, 0),
                ("endpoints.max_message_bytes", self.endpoints.max_message_bytes, 1),
                ("endpoints.image_workers", self.endpoints.image_workers, 1),
                ("max_transitions", self.rollout.max_transitions, 1),
                ("eval_episodes", self.rollout.eval_episodes, 1),
                ("recent_window", self.rollout.recent_window, 1),
                ("replay_capacity", self.update.replay_capacity, 1),
                ("learning_starts", self.update.learning_starts, 0),
                ("max_backlog", self.update.max_backlog, 1),
                ("max_updates_per_wait", self.update.max_updates_per_wait, 1),
                ("progressive_exploration_steps", self.update.progressive_exploration_steps, 0),
                ("checkpoint.interval", self.checkpoint.interval, 1)):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.checkpoint.keep is not None and (type(self.checkpoint.keep) is not int or self.checkpoint.keep < 1):
            raise ValueError("checkpoint.keep must be a positive integer or null (retain all)")
        for name in ("num_envs", "execution_horizon"):
            value = getattr(self.rollout, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer or null")
        if not math.isfinite(self.update.utd) or self.update.utd < 0:
            raise ValueError("utd must be finite and nonnegative")
        if not math.isfinite(self.endpoints.timeout) or self.endpoints.timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        if type(self.wandb.enabled) is not bool or self.wandb.mode not in ("online", "offline"):
            raise ValueError("wandb.enabled must be boolean and mode must be online or offline")
        for name in ("log_interval_seconds", "init_timeout_seconds"):
            value = getattr(self.wandb, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"wandb.{name} must be finite and positive")
        if (not isinstance(self.wandb.project, str) or not self.wandb.project
                or not isinstance(self.wandb.tags, list)
                or any(not isinstance(tag, str) for tag in self.wandb.tags)):
            raise ValueError("wandb requires a nonempty project and a list of string tags")
        if not isinstance(self.learner, dict) or not isinstance(self.rollout.reset_options, dict):
            raise ValueError("learner and reset_options must be objects")
        self.validate_terminal_mode()
        if self.method == "base" and self.checkpoint.resume:
            raise ValueError("base evaluation does not use an RL checkpoint")
        if self.mode == "eval" and self.method != "base" and not self.checkpoint.resume:
            raise ValueError("RL evaluation requires checkpoint.resume; use method=base for frozen-model evaluation")

    def validate_terminal_mode(self):
        mode = self.rollout.terminal_observation_mode
        if mode not in ("full", "state_only_if_no_bootstrap"):
            raise ValueError("Unknown rollout.terminal_observation_mode")
        if mode != "full" and (self.mode != "train" or self.learner.get("bootstrap_truncated", False) is not False):
            raise ValueError("State-only terminal observations require training with bootstrap_truncated=false")
        if type(self.rollout.fused_step_reset) is not bool:
            raise ValueError("rollout.fused_step_reset must be boolean")
        if self.rollout.fused_step_reset and mode != "state_only_if_no_bootstrap":
            raise ValueError("fused_step_reset requires state_only_if_no_bootstrap")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _schema_fingerprint(identity):
    stable = copy.deepcopy(identity)
    # Runtime capacity and replay-only storage choices are not policy identity.
    for key in ("num_envs", "fingerprint", "terminal_observation_mode", "fused_step_reset"):
        stable["contract"].pop(key, None)
    return hashlib.sha256(_canonical(stable).encode()).hexdigest()


def _module_sha256(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(_canonical([name, str(array.dtype), list(array.shape)]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


class _FrozenDSRLPolicy:
    """Load only the actor; never restore the previous learner's RNG or optimizer.

    Checkpoints are trusted local pickle files, as for SACLearner.load. The
    source replay and critic tensors stay on CPU and are discarded after load.
    """

    def __init__(self, settings, fingerprint, contract, device):
        path = Path(settings.checkpoint).resolve()
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        checkpoint_sha256 = digest.hexdigest()
        if settings.checkpoint_sha256 is not None and settings.checkpoint_sha256 != checkpoint_sha256:
            raise ValueError("Frozen DSRL checkpoint SHA256 mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        after = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError("Frozen DSRL checkpoint changed while loading")
        if payload.get("version") != CHECKPOINT_VERSION or payload.get("source_commit") != SOURCE_COMMIT:
            raise ValueError("Unsupported frozen DSRL checkpoint identity")
        if payload.get("extra", {}).get("schema_fingerprint") != fingerprint:
            raise ValueError("Frozen DSRL checkpoint schema/model/task fingerprint differs from connected nodes")
        self.config = cfg = LearnerConfig(**payload["config"])
        if cfg.method != "initial_noise":
            raise ValueError("Frozen DSRL checkpoint must contain an initial_noise learner")
        expected = {"obs_dim": contract["obs_dim"], "action_horizon": contract["action_horizon"],
                    "model_action_dim": contract["action_dim"], "noise_dim": contract["noise_shape"][1]}
        if any(getattr(cfg, key) != value for key, value in expected.items()):
            raise ValueError("Frozen DSRL learner geometry differs from negotiated observations/noise/actions")
        self.device = torch.device(device)
        # Initializers consume CPU randomness even though every weight is then
        # replaced. Preserve it so adding a frozen actor cannot perturb SAC.
        with torch.random.fork_rng(devices=[]):
            self.actor = _Actor(cfg.obs_dim, cfg.action_dim, cfg)
        self.actor.load_state_dict(payload["actor"], strict=True)
        if any(not torch.isfinite(value).all() for value in self.actor.state_dict().values()):
            raise ValueError("Frozen DSRL actor contains nonfinite parameters")
        self.actor.to(self.device).eval().requires_grad_(False)
        self.actor_sha256 = _module_sha256(self.actor)
        self.verification_count = 0
        self.deterministic = settings.deterministic
        self.identity = {"checkpoint_sha256": checkpoint_sha256, "schema_fingerprint": fingerprint,
                         "learner_config": asdict(cfg), "deterministic": self.deterministic}
        self.provenance = {**self.identity, "checkpoint": str(path), "updates": payload["updates"],
                           "transitions": payload.get("extra", {}).get("transitions"),
                           "actor_sha256": self.actor_sha256, "trainable_parameters": 0,
                           "optimizer_loaded": False, "replay_loaded": False}

    def verify(self):
        if (self.actor.training or any(parameter.requires_grad or parameter.grad is not None
                                       for parameter in self.actor.parameters())):
            raise RuntimeError("Frozen DSRL actor lost its eval/no-gradient invariant")
        current = _module_sha256(self.actor)
        if current != self.actor_sha256:
            raise RuntimeError("Frozen DSRL actor parameters changed during residual training")
        self.verification_count += 1
        return {"actor_sha256": current, "matches_initial": True, "count": self.verification_count}

    @torch.no_grad()
    def act(self, observations):
        observations = _finite(observations, (len(observations), self.config.obs_dim), "frozen DSRL observations")
        tensor = torch.as_tensor(observations, device=self.device, dtype=torch.float32)
        action = self.actor.deterministic(tensor) if self.deterministic else self.actor.sample(tensor)[0]
        return _finite(action.cpu().numpy(), (len(observations), self.config.action_dim), "frozen DSRL action")


def _finite(value, shape, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain numeric values")
    array = raw.astype(np.float32)
    if array.shape != tuple(shape) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {tuple(shape)}, got {array.shape}")
    return array


def _atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _wandb_payload(status):
    """Only measured scalars; recent SR counts completed episodes, not env slots."""
    payload = {}
    groups = {
        "train": ("transitions", "updates"),
        "rollout": ("episodes", "successes", "success_rate", "recent_sr", "recent_episodes"),
        "performance": ("elapsed_seconds", "transitions_per_second", "updates_per_second", "learner_seconds"),
        "progress": ("replay_size", "update_backlog", "dropped_update_credit"),
    }
    for group, names in groups.items():
        for name in names:
            value = status.get(name)
            if type(value) in (int, float) and math.isfinite(value):
                payload[f"{group}/{name}"] = value
    for name, value in status.get("last_update_metrics", {}).items():
        if type(value) in (int, float) and math.isfinite(value):
            payload[f"learner/{name}"] = value
    return payload


def _wandb_status(output, run_id, state, **extra):
    # SDK exceptions may include credentials: never record raw exception text.
    record = {"state": state, "run_id": run_id, **extra}
    try:
        _atomic_json(output / "wandb_status.json", record)
    except OSError:
        pass
    if state == "local_fallback":
        print(json.dumps({"event": "wandb_unavailable", **record}), flush=True)


class _WandbMonitor:
    """Separate uploader process; SDK exit hooks cannot block the trainer.

    The worker samples atomic status.json, while full history stays in
    metrics.jsonl. Authentication uses the existing W&B environment or login.
    """

    def __init__(self, output, config, saved_config, run_id):
        self.output, self.config = output, config
        self.saved_config, self.run_id = copy.deepcopy(saved_config), run_id
        # Endpoint URLs may carry userinfo or query credentials; no endpoint
        # addresses are needed to interpret uploaded experiment metrics.
        for name in ("inference", "simulation"):
            self.saved_config["endpoints"].pop(name, None)
        self.saved_config["endpoints"]["token"] = None
        self.config_path = output / f"wandb_config_{run_id}.json"
        self.stop_path = output / f".wandb_stop_{run_id}"
        self.process = None

    def start(self):
        _wandb_status(self.output, self.run_id, "starting")
        try:
            _atomic_json(self.config_path, self.saved_config)
            environment = os.environ.copy()
            source = str(Path(__file__).resolve().parents[2])
            environment["PYTHONPATH"] = source + os.pathsep + environment.get("PYTHONPATH", "")
            self.process = subprocess.Popen([sys.executable, "-c",
                "from pressb.runtime.trainer import _wandb_worker; import sys; _wandb_worker(*sys.argv[1:])",
                str(self.output.resolve()), str(self.config_path.resolve()), self.run_id, str(os.getpid())],
                env=environment, start_new_session=True)
        except OSError as error:
            _wandb_status(self.output, self.run_id, "local_fallback", stage="process_start",
                          error_type=type(error).__name__, local_metrics="metrics.jsonl")

    def close(self):
        if self.process is None:
            return
        try:
            self.stop_path.touch()
            self.process.wait(timeout=5.)
        except (OSError, subprocess.TimeoutExpired):
            # Kill the uploader's process group, including the SDK service.
            # The trainer never imports W&B, so it owns no SDK atexit hooks.
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(self.process.pid, sig)
                    self.process.wait(timeout=1.)
                    break
                except ProcessLookupError:
                    break
                except subprocess.TimeoutExpired:
                    continue
            _wandb_status(self.output, self.run_id, "local_fallback", stage="shutdown_timeout",
                          local_metrics="metrics.jsonl", local_status="status.json",
                          final_remote_metrics_may_be_missing=True)


def _wandb_worker(output_dir, config_path, run_id, parent_pid=None):
    """Subprocess entry point. Local terminal status survives interrupted upload."""
    output, run = Path(output_dir), None
    orphaned = threading.Event()
    if parent_pid is not None:
        def watch_parent():
            while os.getppid() == int(parent_pid):
                time.sleep(.2)
            orphaned.set()
            _wandb_status(output, run_id, "trainer_unavailable", stage="parent_exit",
                          local_metrics="metrics.jsonl", local_status="status.json")
            # This worker has its own process group. Bound SDK init/finish and
            # atexit even when the trainer cannot perform normal cleanup.
            time.sleep(5.)
            if os.getpgrp() == os.getpid():
                os.killpg(os.getpid(), signal.SIGKILL)

        threading.Thread(target=watch_parent, name="wandb-parent-watch", daemon=True).start()
    stage = "initialization"
    try:
        saved_config = json.loads(Path(config_path).read_text())
        cfg = WandbConfig(**saved_config["wandb"])
        wandb = importlib.import_module("wandb")
        run = wandb.init(project=cfg.project, entity=cfg.entity,
            name=cfg.name or output.name, id=run_id, mode=cfg.mode,
            dir=str(output), config=saved_config, tags=cfg.tags,
            settings=wandb.Settings(disable_git=True, disable_code=True, save_code=False,
                console="off", quiet=True, silent=True, x_disable_stats=True, x_disable_meta=True,
                init_timeout=cfg.init_timeout_seconds))
        if run is None:
            raise RuntimeError("W&B did not create a run")
        for axis in ("train/transitions", "train/updates"):
            run.define_metric(axis, hidden=True)
        for group in ("rollout", "performance", "progress"):
            run.define_metric(group + "/*", step_metric="train/transitions", step_sync=False)
        run.define_metric("learner/*", step_metric="train/updates", step_sync=False)
        registration = {"mode": cfg.mode, "url": run.url if cfg.mode == "online" else None}
        _wandb_status(output, run_id, "active", **registration)
        stage, last_log, previous, trainer_state = "logging", -math.inf, None, None
        stop_path = output / f".wandb_stop_{run_id}"
        while True:
            stopping = stop_path.exists() or orphaned.is_set()
            try:
                status = json.loads((output / "status.json").read_text())
            except FileNotFoundError:
                status = None
            if status and status.get("run_id") == run_id:
                identity = (status["transitions"], status["updates"], status["state"])
                terminal = status["state"] in ("completed", "stopped", "failed", "transition_limit")
                if identity != previous and (terminal or stopping or time.monotonic() - last_log >= cfg.log_interval_seconds):
                    run.log(_wandb_payload(status))
                    trainer_state = status["state"]
                    run.summary["trainer/state"] = trainer_state
                    previous, last_log = identity, time.monotonic()
            if stopping:
                if orphaned.is_set():
                    trainer_state = "trainer_unavailable"
                    run.summary["trainer/state"] = trainer_state
                break
            time.sleep(min(.2, cfg.log_interval_seconds))
        stage = "finish"
        run.finish(exit_code=1 if trainer_state in ("failed", "trainer_unavailable") else 0)
        run = None
        _wandb_status(output, run_id, "finished", trainer_state=trainer_state, **registration)
    except Exception as error:
        _wandb_status(output, run_id, "local_fallback", stage=stage,
                      error_type=type(error).__name__, local_metrics="metrics.jsonl", local_status="status.json")
    finally:
        if run is not None:
            try:
                run.finish(exit_code=1)
            except Exception:
                pass


class GenericTrainer:
    """Configurable DSRL/action-residual trainer, also supporting base evaluation.

    Resuming restores replay, optimizer and RNG state. Simulator episodes start
    afresh; the checkpoint explicitly does not claim exact physics continuation.
    Client injection is useful for CPU contract tests and custom transports.
    An optional observer receives validated rollout snapshots for collection or
    evaluation, without changing the policy, simulation, or checkpoint format.
    """

    def __init__(self, config: TrainerConfig | dict, *, inference_client=None,
                 simulation_client=None, observer=None):
        self.config = copy.deepcopy(config if isinstance(config, TrainerConfig)
                                    else TrainerConfig(**config))
        self.config.__post_init__()
        if observer is not None:
            for name in ("on_reset", "on_transition", "close"):
                if not callable(getattr(observer, name, None)):
                    raise TypeError(f"observer.{name} must be callable")
        self.observer = observer
        self._observer_closed = False
        cfg = self.config
        self.inference = inference_client or NodeClient(
            cfg.endpoints.inference, timeout=cfg.endpoints.timeout, token=cfg.endpoints.token,
            max_message_bytes=cfg.endpoints.max_message_bytes,
            image_transport=cfg.endpoints.image_transport, preprocess_images=cfg.endpoints.preprocess_images,
            image_workers=cfg.endpoints.image_workers)
        self.simulation = simulation_client or NodeClient(
            cfg.endpoints.simulation, timeout=cfg.endpoints.timeout, token=cfg.endpoints.token,
            max_message_bytes=cfg.endpoints.max_message_bytes,
            image_transport=cfg.endpoints.image_transport, preprocess_images=cfg.endpoints.preprocess_images,
            image_workers=cfg.endpoints.image_workers)
        self.output = Path(cfg.output_dir)
        self.run_id = uuid.uuid4().hex
        self.rng = np.random.default_rng(cfg.seed)
        self.stop_event = threading.Event()
        self.learner = None
        self.frozen_dsrl = None
        self.replay = None
        self.contract = None
        self.schema_fingerprint = None
        self.live_contexts: set[str] = set()
        self.slots: dict[int, dict] = {}
        self.transitions = self.episodes = self.successes = 0
        self.reset_envs = self.reset_calls = 0
        self.update_credit = self.dropped_update_credit = 0.
        self.recent = deque(maxlen=cfg.rollout.recent_window)
        self.latest_metrics = {}
        self.last_checkpoint = None
        self.last_checkpoint_step = 0
        self.last_saved_updates = -1
        self.started = None
        self._status = {"state": "created", "run_id": self.run_id}
        self._executor = None
        self._step_requests = 0
        self._run_initial_transitions = 0
        self._run_initial_updates = 0
        self._allow_updates = True
        self._simulation_claim_attempted = False
        self._physics_in_flight = False
        self.rpc_seconds = {}
        self.rpc_calls = {}
        self.learner_seconds = 0.
        self.projected_coordinates = self.projected_chunks = 0
        self.max_projection_delta = 0.
        self.projection_delta_by_coordinate = None
        self._wandb = None

    def stop(self, _payload=None):
        self.stop_event.set()
        return self.health()

    def health(self, _payload=None):
        return copy.deepcopy(self._status)

    def _observe(self, method, *values):
        if self.observer is not None:
            # Observers own their snapshots: neither mutation nor deferred
            # writing may alter the rollout's live observations and actions.
            getattr(self.observer, method)(*copy.deepcopy(values))

    def _close_observer(self, failure):
        if self.observer is None or self._observer_closed:
            return
        self._observer_closed = True
        try:
            self.observer.close(self.health())
        except BaseException as error:
            if failure is None:
                self._status_write("failed", error_type=type(error).__name__, error=str(error))
                raise
            # Preserve the original simulation/policy/write error; a secondary
            # close failure must not disguise which operation failed first.
            try:
                self._log("events.jsonl", {"event": "observer_close_failed", "error": str(error)})
            except Exception:
                pass

    def _log(self, name, data):
        with (self.output / name).open("a") as stream:
            stream.write(json.dumps(data, allow_nan=False) + "\n")

    def _status_write(self, state, **extra):
        elapsed = time.monotonic() - self.started if self.started else 0.
        updates = self.learner.updates if self.learner else 0
        self._status = {
            "state": state, "run_id": self.run_id, "method": self.config.method,
            "transitions": self.transitions, "episodes": self.episodes,
            "successes": self.successes,
            "success_rate": self.successes / self.episodes if self.episodes else None,
            "recent_sr": sum(self.recent) / len(self.recent) if self.recent else None,
            "recent_episodes": len(self.recent), "updates": updates,
            "replay_size": len(self.replay) if self.replay is not None else 0,
            "update_backlog": self.update_credit,
            "dropped_update_credit": self.dropped_update_credit,
            "elapsed_seconds": elapsed,
            "transitions_per_second": ((self.transitions - self._run_initial_transitions) / elapsed
                                       if elapsed else 0.),
            "updates_per_second": ((updates - self._run_initial_updates) / elapsed if elapsed else 0.),
            "last_checkpoint": self.last_checkpoint,
            "last_update_metrics": self.latest_metrics,
            "rpc_seconds": self.rpc_seconds.copy(), "rpc_calls": self.rpc_calls.copy(),
            "learner_seconds": self.learner_seconds,
            "action_bounds": self.config.rollout.action_bounds,
            "terminal_observation_mode": self.config.rollout.terminal_observation_mode,
            "fused_step_reset": self.config.rollout.fused_step_reset,
            "reset_envs": self.reset_envs, "reset_calls": self.reset_calls,
            "projected_action_coordinates": self.projected_coordinates,
            "projected_action_chunks": self.projected_chunks,
            "max_projection_delta": self.max_projection_delta,
            "projection_delta_by_coordinate": (self.projection_delta_by_coordinate.tolist()
                                               if self.projection_delta_by_coordinate is not None else None),
            "transition_budget_overshoot": max(0, self.transitions - self.config.rollout.max_transitions),
            "episode_budget_overshoot": (max(0, self.episodes - self.config.rollout.eval_episodes)
                                         if self.config.mode == "eval" else 0),
            "schema_fingerprint": self.schema_fingerprint, **extra,
        }
        _atomic_json(self.output / "status.json", self._status)

    def _can_update(self):
        return (self._allow_updates and self.config.mode == "train" and self.learner is not None
                and self.replay is not None and len(self.replay) >= self.learner.config.batch_size
                and self.transitions >= self.config.update.learning_starts
                and self.update_credit >= 1.)

    def _learn_one(self):
        started = time.monotonic()
        self.latest_metrics = self.learner.update(self.replay.sample(self.learner.config.batch_size))
        self.learner_seconds += time.monotonic() - started
        self.update_credit -= 1.

    def _call(self, client, path, payload=None):
        started = time.monotonic()
        node = "inference" if client is self.inference else "simulation"
        name = node + path
        try:
            result = self._call_pending(client, path, payload)
            # No request payloads, image bytes, endpoint URLs, or credentials.
            # Successful calls only: a failed RPC must not reuse stale timings.
            timing = {"node": node, "path": path, "method": path.removeprefix("/"),
                      "transition": self.transitions, "elapsed_seconds": time.monotonic() - started}
            for label, values in (("client", getattr(client, "last_call_timing", {})),
                                  ("server", result.get("timing", {}) if isinstance(result, dict) else {})):
                if isinstance(values, dict):
                    timing[label] = {key: value for key, value in values.items()
                                     if isinstance(key, str) and key.isascii() and key.replace("_", "").isalnum()
                                     and type(value) in (int, float) and math.isfinite(value)}
            self._log("rpc_timing.jsonl", timing)
            return result
        finally:
            self.rpc_seconds[name] = self.rpc_seconds.get(name, 0.) + time.monotonic() - started
            self.rpc_calls[name] = self.rpc_calls.get(name, 0) + 1

    def _call_pending(self, client, path, payload=None):
        # The I/O thread never touches learner or replay. No optimistic retry:
        # a lost /step response means the physical execution state is unknown.
        future = self._executor.submit(client.call, path, payload)
        remaining = self.config.update.max_updates_per_wait
        while not future.done():
            if remaining and self._can_update():
                self._learn_one()
                remaining -= 1
            else:
                try:
                    return future.result(timeout=.05)
                except FutureTimeout:
                    pass
        return future.result()

    def _negotiate(self):
        self.config.validate_terminal_mode()
        self.inference_metadata = self._call(self.inference, "/health")
        self.simulation_metadata = self._call(self.simulation, "/health")
        self.contract = negotiate(self.inference_metadata, self.simulation_metadata,
                                  self.config.method, self.config.rollout.execution_horizon,
                                  terminal_observation_mode=self.config.rollout.terminal_observation_mode,
                                  fused_step_reset=self.config.rollout.fused_step_reset)
        contract = self.contract
        noise_shape = tuple(contract["noise_shape"])
        if len(noise_shape) != 2 or noise_shape[0] != contract["action_horizon"]:
            raise ValueError("Trainer requires [horizon, latent_width] noise geometry")
        simulation_action = self.simulation_metadata["action"]
        self.action_lower = simulation_action.get("lower")
        self.action_upper = simulation_action.get("upper")
        if self.config.rollout.action_bounds == "clip" and (self.action_lower is None or self.action_upper is None):
            raise ValueError("action_bounds=clip requires simulation action.lower and action.upper")
        self.projection_delta_by_coordinate = np.zeros(contract["action_dim"], dtype=np.float64)
        identity = {"contract": contract, "action_projection": {
            "mode": self.config.rollout.action_bounds, "lower": self.action_lower, "upper": self.action_upper}}
        # Stable identities only; context counts, timestamps and hostnames must
        # not prevent a compatible resume on a different machine.
        for kind, metadata in (("inference", self.inference_metadata),
                               ("simulation", self.simulation_metadata)):
            identity[kind] = {k: metadata[k] for k in (
                "model_id", "checkpoint_sha256", "weights_sha256", "source_revision",
                "normalization_sha256", "task_id", "task_config_sha256", "identity",
                "schema_fingerprint", "action_schema", "observation_schema", "reward_schema")
                if k in metadata}
        if self.config.frozen_dsrl is not None:
            # Negotiate the original method separately, including external-noise
            # capability. The residual contract has a different method identity.
            noise_identity = copy.deepcopy(identity)
            noise_identity["contract"] = negotiate(
                self.inference_metadata, self.simulation_metadata, "dsrl",
                self.config.rollout.execution_horizon,
                terminal_observation_mode=self.config.rollout.terminal_observation_mode,
                fused_step_reset=self.config.rollout.fused_step_reset)
            self.frozen_dsrl = _FrozenDSRLPolicy(
                self.config.frozen_dsrl, _schema_fingerprint(noise_identity),
                noise_identity["contract"], self.config.device)
            identity["frozen_dsrl"] = self.frozen_dsrl.identity
            _atomic_json(self.output / "frozen_dsrl.json", self.frozen_dsrl.provenance)
        self.schema_fingerprint = _schema_fingerprint(identity)
        _atomic_json(self.output / "negotiated.json", identity)
        available = self.simulation_metadata.get("num_envs")
        if type(available) is not int or available < 1:
            raise ValueError("Simulation metadata must declare positive num_envs")
        self.num_envs = self.config.rollout.num_envs or available
        if self.num_envs > available:
            raise ValueError("Configured num_envs exceeds simulation capacity")
        self.all_active = self.simulation_metadata.get("scheduling") == "all_active"
        self.initial_reset_all = self.simulation_metadata.get("initial_reset") == "all_envs"
        if self.initial_reset_all and self.num_envs != available:
            raise ValueError("This simulation requires all environments in its initial reset")
        if self.config.method == "base":
            return
        values = copy.deepcopy(self.config.learner)
        expected = dict(method="initial_noise" if self.config.method == "dsrl" else "action_residual",
                        obs_dim=contract["obs_dim"], action_horizon=contract["action_horizon"],
                        model_action_dim=contract["action_dim"], noise_dim=noise_shape[1])
        for name, value in expected.items():
            if name in values and values[name] != value:
                raise ValueError(f"learner.{name} conflicts with negotiated metadata")
            values[name] = value
        for name, bounds in (("action_lower", self.action_lower), ("action_upper", self.action_upper)):
            desired_bounds = bounds if self.config.rollout.action_bounds == "clip" else None
            if name in values and values[name] is not None and not np.array_equal(values[name], desired_bounds):
                raise ValueError(f"learner.{name} conflicts with configured executable action projection")
            values[name] = desired_bounds
        values.setdefault("seed", self.config.seed)
        values.setdefault("residual_mode", "all")
        if self.config.method == "action_residual" and "residual_scale" not in values:
            raise ValueError("Specify residual_scale explicitly in executable action units")
        values.setdefault("residual_q_bounds", None)
        desired = LearnerConfig(**values)
        if self.config.mode == "train" and self.config.update.replay_capacity < desired.batch_size:
            raise ValueError("replay_capacity must be at least learner.batch_size")
        self.learner = SACLearner(desired, device=self.config.device)
        self.replay = ReplayBuffer(self.config.update.replay_capacity, desired.obs_dim,
                                   desired.action_dim, seed=self.config.seed,
                                   base_action_shape=desired.base_action_shape)
        if self.config.checkpoint.resume:
            self._resume(desired)
        else:
            initial = {"event": "learner_initialized", "method": self.config.method,
                      "updates": self.learner.updates, "transitions": self.transitions,
                      "replay_size": len(self.replay), "resume": None,
                      "actor_sha256": _module_sha256(self.learner.actor),
                      "optimizer_state_entries": {"actor": len(self.learner.actor_optimizer.state),
                                                  "critic": len(self.learner.q_optimizer.state),
                                                  "alpha": len(self.learner.alpha_optimizer.state)},
                      "frozen_dsrl": self.frozen_dsrl.provenance if self.frozen_dsrl is not None else None}
            self._log("events.jsonl", initial)
            _atomic_json(self.output / "initialization.json", initial)

    def _resume(self, desired):
        learner = SACLearner.load(self.config.checkpoint.resume, device=self.config.device)
        extra = learner.checkpoint_extra
        if extra.get("schema_fingerprint") != self.schema_fingerprint:
            raise ValueError("Checkpoint schema/model/task fingerprint differs from connected nodes")
        if _canonical(asdict(learner.config)) != _canonical(asdict(desired)):
            raise ValueError("Checkpoint learner configuration differs; use an explicit matching config")
        self.learner = learner
        self._run_initial_updates = learner.updates
        if self.config.mode == "eval":
            return
        if extra.get("terminal_observation_mode", "full") != self.config.rollout.terminal_observation_mode:
            raise ValueError("Checkpoint terminal observation mode differs; replay storage semantics cannot change on resume")
        if extra.get("fused_step_reset", False) != self.config.rollout.fused_step_reset:
            raise ValueError("Checkpoint fused_step_reset differs; reset scheduling cannot change on resume")
        if "replay" not in extra:
            raise ValueError("Full training resume requires a checkpoint saved with include_replay=true")
        self.replay.load_state_dict(extra["replay"])
        self.transitions = int(extra["transitions"])
        self.episodes, self.successes = int(extra["episodes"]), int(extra["successes"])
        self.recent.extend(extra["recent"])
        self.rng.bit_generator.state = extra["runner_rng"]
        self.update_credit = float(extra["update_credit"])
        self.dropped_update_credit = float(extra.get("dropped_update_credit", 0.))
        self.last_checkpoint_step = self.transitions
        self._run_initial_transitions = self.transitions
        self._run_initial_updates = learner.updates
        self._log("events.jsonl", {"event": "resumed", "checkpoint": self.config.checkpoint.resume,
                                  "simulator_continuation": "fresh episodes", "transitions": self.transitions})

    def _observation_vector(self, observation, encoded):
        parts = [_finite(encoded["features"], (self.contract["feature_dim"],), "features")]
        for name, width in (("state", "state_dim"), ("control_state", "control_state_dim"),
                            ("task_features", "task_features_dim")):
            parts.append(_finite(observation.get(name, []), (self.contract[width],), name))
        vector = np.concatenate(parts)
        return _finite(vector, (self.contract["obs_dim"],), "learner observation")

    def _encode(self, rows, *, seeds=None):
        if not rows:
            return []
        if seeds is None:
            seeds = self.rng.integers(0, 2**31 - 1, size=len(rows)).tolist()
        result = self._call(self.inference, "/encode", {
            "observations": [row["observation"] for row in rows], "seeds": seeds,
            "include_base_actions": self.config.method in ("base", "action_residual")
                                    and self.frozen_dsrl is None})
        items = result.get("items", [])
        previous_contexts = self.live_contexts.copy()
        # Track all server-created contexts before validating any item, so an
        # invalid response still gets best-effort cache cleanup.
        self.live_contexts.update(item["context_id"] for item in items
                                  if isinstance(item, dict) and isinstance(item.get("context_id"), str))
        if len(items) != len(rows):
            raise ValueError("Inference encode returned the wrong number of items")
        contexts = [item.get("context_id") for item in items]
        if (any(not isinstance(c, str) or not c for c in contexts)
                or len(set(contexts)) != len(contexts) or previous_contexts.intersection(contexts)):
            raise ValueError("Inference encode context ids must be unique nonempty strings")
        shape = (self.contract["action_horizon"], self.contract["action_dim"])
        slots = []
        for row, encoded in zip(rows, items, strict=True):
            slot = {**row, "encoded": encoded, "vector": self._observation_vector(row["observation"], encoded)}
            if self.config.method in ("base", "action_residual") and self.frozen_dsrl is None:
                slot["base"] = _finite(encoded["base_actions"], shape, "base_actions")
            slots.append(slot)
        if self.frozen_dsrl is not None:
            normalized = self.frozen_dsrl.act(np.stack([slot["vector"] for slot in slots]))
            bases = self._decode_noise(slots, normalized, self.frozen_dsrl.config)
            for slot, base in zip(slots, bases, strict=True):
                slot["base"] = base
        return slots

    def _next_slots(self, rows, *, reset_rows=None):
        if self.config.rollout.terminal_observation_mode == "full":
            return self._encode(rows), {}
        # Consume the same seed stream as the full path. No terminal image or
        # cache entry is needed when the Bellman continuation mask is exactly 0.
        seeds = self.rng.integers(0, 2**31 - 1, size=len(rows)).tolist()
        terminal_slots = {}
        to_encode = {}
        for row, seed in zip(rows, seeds, strict=True):
            if not row.get("terminal_observation_omitted", False):
                to_encode[row["env_id"]] = (row, seed)
                continue
            # Keep the exact terminal proprioception/control/task state; only
            # unavailable visual features and next base actions are placeholders.
            features = np.zeros(self.contract["feature_dim"], dtype=np.float32)
            slot = {**row, "vector": self._observation_vector(row["observation"], {"features": features})}
            if self.config.method == "action_residual":
                slot["base"] = np.zeros((self.contract["action_horizon"], self.contract["action_dim"]), dtype=np.float32)
            terminal_slots[row["env_id"]] = slot
        # Reset only ended environments, then encode their new initial images
        # together with continuing observations. This usually keeps one full
        # batch per rollout and avoids a second, small vision-model invocation.
        remaining = self.config.rollout.max_transitions - self.transitions - len(rows)
        continuing = len(self.slots) - len(terminal_slots)
        if reset_rows is None:
            reset_ids = list(terminal_slots)[:max(0, remaining - continuing)]
            if self.stop_event.is_set():
                reset_ids = []
            reset_rows = self._reset_rows(reset_ids) if reset_ids else []
        else:
            reset_ids = [row["env_id"] for row in reset_rows]
        if reset_rows:
            reset_seeds = self.rng.integers(0, 2**31 - 1, size=len(reset_rows)).tolist()
            to_encode.update({row["env_id"]: (row, seed)
                              for row, seed in zip(reset_rows, reset_seeds, strict=True)})
        ordered = [to_encode[eid] for eid in sorted(to_encode)]
        encoded = self._encode([item[0] for item in ordered], seeds=[item[1] for item in ordered])
        encoded_by_id = {slot["env_id"]: slot for slot in encoded}
        next_slots = [terminal_slots[row["env_id"]] if row["env_id"] in terminal_slots
                      else encoded_by_id[row["env_id"]] for row in rows]
        return next_slots, {eid: encoded_by_id[eid] for eid in reset_ids}

    def _release(self, contexts):
        contexts = sorted(set(contexts) & self.live_contexts)
        if contexts:
            self._call(self.inference, "/release", {"context_ids": contexts})
            self.live_contexts.difference_update(contexts)

    def _reset_rows(self, env_ids):
        self._simulation_claim_attempted = True
        self._physics_in_flight = True
        rows = self._call(self.simulation, "/reset", {
            "run_id": self.run_id, "request_id": uuid.uuid4().hex, "env_ids": env_ids,
            "seed": int(self.rng.integers(0, 2**31 - 1)),
            "options": self.config.rollout.reset_options}).get("items", [])
        self._validate_reset_rows(rows, env_ids)
        self.reset_envs += len(rows)
        self.reset_calls += 1
        self._physics_in_flight = False
        self._observe("on_reset", rows)
        return rows

    def _validate_reset_rows(self, rows, env_ids):
        if len(rows) != len(env_ids) or {r.get("env_id") for r in rows} != set(env_ids):
            raise ValueError("Simulation reset returned wrong environment ids")
        for row in rows:
            if (not isinstance(row.get("episode_id"), str) or not row["episode_id"]
                    or type(row.get("step_id")) is not int or row["step_id"] != 0):
                raise ValueError("Simulation reset must start an identified episode at step zero")
            prior = self.slots.get(row["env_id"])
            if prior is not None and row["episode_id"] == prior["episode_id"]:
                raise ValueError("Simulation reset reused the previous episode identity")
            validate_observation(row.get("observation"), self.simulation_metadata["observation"])

    def _reset(self, env_ids):
        rows = self._reset_rows(env_ids)
        self.slots.update({s["env_id"]: s for s in self._encode(rows)})

    def _actions(self, slots):
        if self.config.method == "base":
            return None, np.stack([s["base"] for s in slots])
        deterministic = self.config.mode == "eval" and self.config.rollout.deterministic_eval
        warmup = self.config.mode == "train" and self.transitions < self.config.update.learning_starts
        ramp = self.config.update.progressive_exploration_steps
        exploration = (min(1., max(0., (self.transitions - self.config.update.learning_starts) / ramp))
                       if ramp and self.config.mode == "train" else 1.)
        base = np.stack([s["base"] for s in slots]) if self.config.method == "action_residual" else None
        normalized = self.learner.act(np.stack([s["vector"] for s in slots]), base,
                                      deterministic=deterministic, warmup=warmup,
                                      exploration_probability=exploration)
        cfg = self.learner.config
        if self.config.method == "action_residual":
            actions = base.copy()
            delta = normalized.reshape(len(slots), cfg.action_horizon, len(cfg.residual_indices))
            actions[..., list(cfg.residual_indices)] += delta * np.asarray(cfg.residual_scale)
            return normalized, actions
        return normalized, self._decode_noise(slots, normalized, cfg)

    def _decode_noise(self, slots, normalized, cfg):
        noise = normalized.reshape(len(slots), cfg.noise_steps, cfg.noise_dim)
        noise = np.tile(noise, (1, cfg.action_horizon // cfg.noise_steps, 1)) * cfg.noise_scale
        contexts = [s["encoded"]["context_id"] for s in slots]
        rows = self._call(self.inference, "/decode", {"items": [
            {"context_id": context, "initial_noise": z.tolist()}
            for context, z in zip(contexts, noise, strict=True)]}).get("items", [])
        if len(rows) != len(slots) or [r.get("context_id") for r in rows] != contexts:
            raise ValueError("Inference decode response context ids/order mismatch")
        actions = np.stack([_finite(row["actions"], (cfg.action_horizon, cfg.model_action_dim),
                                    "decoded actions") for row in rows])
        return actions

    def _project_actions(self, actions):
        shape = (len(actions), self.contract["action_horizon"], self.contract["action_dim"])
        actions = _finite(actions, shape, "executable actions")
        prefix = self.contract["execution_horizon"]
        executed = actions[:, :prefix]
        lower = np.asarray(self.action_lower, dtype=np.float32) if self.action_lower is not None else None
        upper = np.asarray(self.action_upper, dtype=np.float32) if self.action_upper is not None else None
        if self.config.rollout.action_bounds == "error":
            if ((lower is not None and np.any(executed < lower - 1e-6))
                    or (upper is not None and np.any(executed > upper + 1e-6))):
                raise ValueError("Executable action exceeds simulation limits; explicitly configure action_bounds=clip if desired")
            return actions
        clipped = np.clip(actions, lower, upper)
        delta = np.abs(clipped[:, :prefix] - executed)
        changed = delta > 0
        if np.any(changed):
            count = int(changed.sum())
            chunks = int(changed.any(axis=(1, 2)).sum())
            maximum = float(delta.max())
            self.projected_coordinates += count
            self.projected_chunks += chunks
            self.max_projection_delta = max(self.max_projection_delta, maximum)
            by_coordinate = delta.max(axis=(0, 1))
            self.projection_delta_by_coordinate = np.maximum(self.projection_delta_by_coordinate, by_coordinate)
            self._log("events.jsonl", {"event": "actions_projected", "before_transition": self.transitions,
                                      "coordinates": count, "chunks": chunks, "max_delta": maximum,
                                      "delta_by_coordinate": by_coordinate.tolist()})
        return clipped

    def _validate_steps(self, before, result):
        rows = result.get("items", [])
        expected = {slot["env_id"]: slot for slot in before}
        if len(rows) != len(before) or {r.get("env_id") for r in rows} != set(expected):
            raise ValueError("Simulation step returned wrong environment ids")
        for row in rows:
            prior = expected[row["env_id"]]
            if (row.get("episode_id") != prior["episode_id"] or type(row.get("step_id")) is not int
                    or row["step_id"] != prior["step_id"] + 1):
                raise ValueError("Simulation step returned stale episode/step identity")
            ended, timed = row.get("terminated"), row.get("truncated")
            if type(ended) is not bool or type(timed) is not bool or (ended and timed):
                raise ValueError("Termination flags must be exclusive booleans")
            omitted = row.get("terminal_observation_omitted", False)
            expected_omission = self.config.rollout.terminal_observation_mode != "full" and (ended or timed)
            if type(omitted) is not bool or omitted != expected_omission:
                raise ValueError("Simulation terminal_observation_omitted disagrees with negotiated mode")
            validate_observation(row.get("observation"), self.simulation_metadata["observation"],
                                 allow_missing_images=omitted)
            _finite([row.get("reward"), row.get("discount")], (2,), "reward/discount")
            if not 0 <= row["discount"] <= 1:
                raise ValueError("Simulation discount must be in [0,1]")
            info = row.get("info", {})
            duration = info.get("elapsed_control_steps")
            if (isinstance(duration, bool) or not isinstance(duration, (int, float))
                    or not math.isfinite(duration) or duration <= 0):
                raise ValueError("Simulation must report actual positive elapsed_control_steps")
            if type(info.get("success")) is not bool:
                raise ValueError("Simulation info.success must be a boolean")
            if type(info.get("physics_steps")) is not int or info["physics_steps"] <= 0:
                raise ValueError("Simulation info.physics_steps must be a positive integer")
            if duration > self.contract["execution_horizon"] + 1e-6:
                raise ValueError("Simulation executed more control steps than the submitted prefix")
            discount = self.contract["discount"]
            exponent = duration * discount["timebase_hz"] / self.contract["action"]["control_hz"]
            if not math.isclose(row["discount"], discount["gamma"] ** exponent, rel_tol=1e-5, abs_tol=1e-7):
                raise ValueError("Simulation discount disagrees with gamma and actual duration")
        return [next(row for row in rows if row["env_id"] == slot["env_id"]) for slot in before]

    def _budget_remaining(self):
        if self.config.mode == "train":
            return self.config.rollout.max_transitions - self.transitions
        return min(self.config.rollout.eval_episodes - self.episodes,
                   self.config.rollout.max_transitions - self.transitions)

    def _save(self, *, final=False):
        if self.config.mode != "train" or self.learner is None:
            return
        frozen_verification = (self.frozen_dsrl.verify() if final and self.frozen_dsrl is not None else None)
        if frozen_verification is not None:
            _atomic_json(self.output / "frozen_dsrl_verification.json", {
                         **frozen_verification, "transitions": self.transitions, "final": final})
        if (self.transitions == self.last_checkpoint_step
                and self.last_saved_updates == self.learner.updates):
            return
        # Save at absolute interval crossings. Parallel batches may cross a
        # boundary by a few transitions; retain the true step in the filename
        # without shifting all later boundaries by that overshoot. The same
        # rule preserves the next boundary after an early final/resume.
        interval = self.config.checkpoint.interval
        if not final and self.transitions // interval <= self.last_checkpoint_step // interval:
            return
        if self.frozen_dsrl is not None and frozen_verification is None:
            frozen_verification = self.frozen_dsrl.verify()
        extra = {"schema_fingerprint": self.schema_fingerprint, "contract": self.contract,
                 "transitions": self.transitions, "episodes": self.episodes, "successes": self.successes,
                 "recent": list(self.recent), "runner_rng": self.rng.bit_generator.state,
                 "update_credit": self.update_credit, "dropped_update_credit": self.dropped_update_credit,
                 "terminal_observation_mode": self.config.rollout.terminal_observation_mode,
                 "fused_step_reset": self.config.rollout.fused_step_reset,
                 "simulator_continuation": "fresh episodes; replay contains terminal state with declared image mode"}
        if self.frozen_dsrl is not None:
            extra["frozen_dsrl"] = {**self.frozen_dsrl.provenance, "verification": frozen_verification}
            _atomic_json(self.output / "frozen_dsrl_verification.json", {
                         **frozen_verification, "transitions": self.transitions, "final": final})
        if self.config.checkpoint.include_replay:
            extra["replay"] = self.replay.state_dict()
        path = self.output / "checkpoints" / f"step_{self.transitions:012d}.pt"
        self.learner.save(path, extra=extra)
        self.last_checkpoint = str(path.resolve())
        self.last_checkpoint_step = self.transitions
        self.last_saved_updates = self.learner.updates
        if self.config.checkpoint.keep is not None:
            for stale in sorted(path.parent.glob("step_*.pt"))[:-self.config.checkpoint.keep]:
                stale.unlink()
        _atomic_json(self.output / "checkpoint.json", {"path": self.last_checkpoint,
                     "transitions": self.transitions, "updates": self.learner.updates,
                     "includes_replay": self.config.checkpoint.include_replay,
                     "simulator_continuation": "fresh episodes"})

    def run(self):
        if self._executor is not None:
            raise RuntimeError("A GenericTrainer instance can only run once")
        if self.output.exists() and any(self.output.iterdir()) and not self.config.checkpoint.resume:
            raise FileExistsError(f"Output directory is not empty: {self.output}")
        self.output.mkdir(parents=True, exist_ok=True)
        saved_config = asdict(self.config)
        saved_config["endpoints"]["token"] = None  # Never write credentials into artifacts.
        config_path = self.output / "config.json"
        if config_path.exists():
            config_path = self.output / f"config_resume_{self.run_id}.json"
        _atomic_json(config_path, saved_config)
        self.started = time.monotonic()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rl-node-io")
        if self.config.wandb.enabled:
            self._wandb = _WandbMonitor(self.output, self.config.wandb, saved_config, self.run_id)
            self._wandb.start()
        failure = None
        try:
            self._status_write("negotiating")
            self._negotiate()
            if self._budget_remaining() > 0:
                initial_count = self.num_envs if self.initial_reset_all else min(self.num_envs, self._budget_remaining())
                self._reset(list(range(initial_count)))
            self._status_write("running")
            while self._budget_remaining() > 0 and not self.stop_event.is_set():
                selected_ids = sorted(self.slots)
                if not self.all_active:
                    selected_ids = selected_ids[:self._budget_remaining()]
                slots = [self.slots[eid] for eid in selected_ids]
                if not slots:
                    raise RuntimeError("No active simulation slots remain")
                normalized, actions = self._actions(slots)
                actions = self._project_actions(actions)
                prefix = self.contract["execution_horizon"]
                payload = {"run_id": self.run_id, "request_id": uuid.uuid4().hex, "items": [
                    {"env_id": slot["env_id"], "episode_id": slot["episode_id"], "step_id": slot["step_id"],
                     "actions": action[:prefix].tolist()}
                    for slot, action in zip(slots, actions, strict=True)]}
                if self.config.rollout.terminal_observation_mode != "full":
                    payload["terminal_observation_mode"] = self.config.rollout.terminal_observation_mode
                step_path = "/step"
                if self.config.rollout.fused_step_reset:
                    step_path = "/step_and_reset"
                    maximum = min(self.num_envs, max(0, self._budget_remaining() - len(slots)))
                    payload.update(reset_seed=int(self.rng.integers(0, 2**31 - 1)),
                                   reset_options=self.config.rollout.reset_options,
                                   max_active_envs=0 if self.stop_event.is_set() else maximum)
                self._step_requests += 1
                self._physics_in_flight = True
                response = self._call(self.simulation, step_path, payload)
                rows = self._validate_steps(slots, response)
                fused_resets = None
                if self.config.rollout.fused_step_reset:
                    done_ids = [row["env_id"] for row in rows if row["terminated"] or row["truncated"]]
                    continuing = len(self.slots) - len(done_ids)
                    reset_ids = done_ids[:max(0, payload["max_active_envs"] - continuing)]
                    fused_resets = response.get("reset_items")
                    if not isinstance(fused_resets, list):
                        raise ValueError("step_and_reset must return a reset_items list")
                    self._validate_reset_rows(fused_resets, reset_ids)
                    self.reset_envs += len(fused_resets)
                    self.reset_calls += bool(fused_resets)
                self._physics_in_flight = False
                # The prefix is exactly what was sent after action projection.
                # Early termination can execute less: rows report the actual
                # duration in info.elapsed_control_steps (possibly fractional).
                self._observe("on_transition", slots, actions[:, :prefix], rows)
                if fused_resets:
                    self._observe("on_reset", fused_resets)
                # Snapshot all real terminal state before reset; skip visual
                # encoding only under the explicit zero-bootstrap contract.
                next_slots, preencoded_resets = self._next_slots(rows, reset_rows=fused_resets)
                reset_ids = []
                for index, (old, new) in enumerate(zip(slots, next_slots, strict=True)):
                    if self.config.mode == "train":
                        self.replay.add(old["vector"], normalized[index], new["reward"], new["discount"],
                                        new["vector"], new["terminated"], new["truncated"],
                                        old.get("base"), new.get("base"))
                    self.transitions += 1
                    self.slots[new["env_id"]] = new
                    if new["terminated"] or new["truncated"]:
                        success = bool(new.get("info", {}).get("success", False))
                        self.episodes += 1
                        self.successes += int(success)
                        self.recent.append(int(success))
                        self._log("episodes.jsonl", {"transition": self.transitions,
                                  **{k: new[k] for k in ("env_id", "episode_id", "step_id", "terminated", "truncated")},
                                  "success": success, "info": new.get("info", {})})
                        reset_ids.append(new["env_id"])
                if self.config.mode == "train" and self.transitions >= self.config.update.learning_starts:
                    total_credit = self.update_credit + len(slots) * self.config.update.utd
                    self.update_credit = min(total_credit, self.config.update.max_backlog)
                    self.dropped_update_credit += total_credit - self.update_credit
                self._release([old["encoded"]["context_id"] for old in slots])
                self._release([self.slots[eid]["encoded"]["context_id"] for eid in reset_ids
                               if "encoded" in self.slots[eid]])
                for eid in reset_ids:
                    del self.slots[eid]
                self.slots.update(preencoded_resets)
                remaining = self._budget_remaining()
                # Only reset ended environments. Continuing episodes never wait
                # for the whole cohort and are never reset because another ended.
                if (self.config.rollout.terminal_observation_mode == "full"
                        and remaining > 0 and not self.stop_event.is_set()):
                    selected_resets = reset_ids[:max(0, remaining - len(self.slots))]
                    if selected_resets:
                        self._reset(selected_resets)
                for _ in range(self.config.update.max_updates_per_wait):
                    if not self._can_update():
                        break
                    self._learn_one()
                self._save()
                self._status_write("running")
                self._log("metrics.jsonl", self.health())
            self._save(final=True)
            state = "stopped" if self.stop_event.is_set() else "completed"
            if self.config.mode == "eval" and self.episodes < self.config.rollout.eval_episodes and state == "completed":
                state = "transition_limit"
            self._status_write(state)
        except BaseException as error:
            failure = error
            self._status_write("failed", error_type=type(error).__name__, error=str(error),
                               execution_state=("unknown after failed reset/step RPC; never automatically retried"
                                                if self._physics_in_flight else
                                                "no automatic continuation; resume starts fresh episodes"))
            raise
        finally:
            self._allow_updates = False
            try:
                try:
                    self._release(self.live_contexts.copy())
                except Exception as error:
                    self._log("events.jsonl", {"event": "context_cleanup_failed", "error": str(error)})
                    if failure is None:
                        self._status_write(self._status["state"], context_cleanup_error=str(error))
                if self._simulation_claim_attempted:
                    try:
                        self._call(self.simulation, "/close_run", {"run_id": self.run_id})
                    except Exception as error:
                        self._log("events.jsonl", {"event": "simulation_close_failed", "error": str(error)})
                self._executor.shutdown(wait=True, cancel_futures=True)
                for client in (self.inference, self.simulation):
                    close = getattr(client, "close", None)
                    if close:
                        close()
                if self._wandb is not None:
                    self._wandb.close()
            except BaseException as error:
                if failure is None and self.observer is not None:
                    failure = error
                    self._status_write("failed", error_type=type(error).__name__, error=str(error))
                raise
            finally:
                self._close_observer(failure)
        return self.health()
