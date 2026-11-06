# Runtime, task plugins, model adapters and experiments

PressB uses one runtime for inference services, environment services, DSRL,
action-residual learning, policy evaluation and chunk trajectory recording.
Choose a task, a frozen model and execution settings in an experiment JSON.
Task physics and outcome rules remain explicit task code.
Wall-OSS and VLA-JEPA inference and supervised-training implementations are now
bundled in PressB; new experiments do not need their separate source clones.
See [bundled models](bundled_models.md) for training profiles and CUDA builds.

## Code ownership

```text
src/pressb/
  runtime/                 # No concrete task/model imports
    contract.py            # Dimensions, units, capabilities and identities
    server.py, client.py    # HTTP / binary MessagePack WebSocket services
    inference.py           # Batching and observation-context lifetime
    simulation.py          # reset/step identities, ordering, validation, retries
    trainer.py, learner.py  # Shared rollout, replay, SAC, evaluation, checkpoints
    physics.py, isaac.py    # PhysX tensors, PiPER/camera assembly
    scene.py, workbench.py # Shared USD primitives, materials, workbench and tripod
    recording.py           # Original-time camera video utilities
    trajectory.py          # Validated chunk transition recorder
  tasks/
    base.py                # Structural TaskPlugin interface
    pap/                   # Scene, outcomes, sampling, vector/process simulation and recording
    table_buttons/         # Scene, outcomes, placement, motion, expert collection and evaluation
    elevator/              # Native simulator and legacy HTTP compatibility
  models/
    base.py                # Structural ModelAdapter interface
    walloss.py             # Frozen Wall-OSS encoder / flow decoder
    vlajepa.py             # Original Cartesian VLA-JEPA checkpoint
    vlajepa_joint.py       # Native joint6 / joint7 VLA-JEPA checkpoints
  model_sources/           # Versioned model/data/training code and provenance
    walloss/               # Wall-X modules, native training and CUDA sources
    vlajepa/               # Shared model plus pose/joint and multi-task trainers
  training/                # Common supervised-training launcher
  compat/                  # Historical Wall-X and table-button policy clients
  experiments/
    config.py, cli.py      # Inheritance, resolved configs and common commands
  registry.py              # Lazy built-in plugin names; no weights loaded
```

PAP and table-button domain implementations now live entirely under their task
packages: scene props, measured success rules, placement, expert motion and
evaluation helpers. Both scenes use `runtime.scene` and `runtime.workbench` for
the shared table, room, mounting hardware, materials, lighting and camera tripod.
The table-button scene does not import the PAP scene. A task-specific planner or
success test is part of the plugin, not another training framework.

The older `pick_place` and `table_buttons` packages contain only compatibility
aliases. Old and canonical module imports share the same module objects so old
imports, monkeypatches and saved class paths continue to resolve. Historical
Wall-X WebSocket and table-button HTTP clients live in `compat`; normal task
backends do not depend on them. PAP observation-only initialization is separate
from the historical client. Existing collection/evaluation scripts now import
canonical task and shared-runtime modules, while retaining their original
frame alignment, dataset schema strings and LeRobot export commands. Collection
source archives and task fingerprints cover the moved implementations and shared
dependencies, not just forwarding files. Elevator scene/planner libraries and
`rl_nodes` aliases retain their earlier compatibility roles.

## One configuration, three processes

Templates live in `configs/experiments/`:

| Template | Task/model | Operation |
|---|---|---|
| `pap_walloss_dsrl.json` | PAP / Wall-OSS | DSRL, 64 environments, BF16 native batching |
| `pap_walloss_residual.json` | PAP / Wall-OSS | Fresh action residual |
| `pap_walloss_eval.json` | PAP / Wall-OSS | Frozen base-policy evaluation |
| `table_buttons_vlajepa_eval.json` | Medium table buttons / joint6 VLA-JEPA | Three-color evaluation, fixed gripper |
| `elevator_vlajepa_dsrl.json` | Elevator / original pose9 VLA-JEPA | DSRL |

A template is a local deployment example, not a request to launch training.
Set the weight paths, output directory, GPUs and network addresses before use.
Different nodes may use different Conda environments and different machines.
Run the generic `python` examples from the repository root with the `zprl`
environment activated; its local interpreter is
`/home/pengguanqi/miniconda3/envs/zprl/bin/python`. Inference and simulation use
their explicitly listed interpreters below.

```bash
# List plugins; this does not import Isaac, Torch or a model.
python scripts/run_experiment.py list

# Validate configuration and optionally write resolved per-node configs.
python scripts/run_experiment.py validate --config configs/experiments/pap_walloss_dsrl.json
python scripts/run_experiment.py prepare --config configs/experiments/pap_walloss_dsrl.json \
  --resolved-dir outputs/my_experiment/resolved

# Three terminals / nodes, each with its required environment.
.conda/envs/wallx-inference/bin/python scripts/run_experiment.py inference \
  --config configs/experiments/pap_walloss_dsrl.json
.conda/envs/pressb/bin/python scripts/run_experiment.py simulation \
  --config configs/experiments/pap_walloss_dsrl.json
/home/pengguanqi/miniconda3/envs/zprl/bin/python scripts/run_experiment.py train \
  --config configs/experiments/pap_walloss_dsrl.json --output outputs/my_dsrl/run
```

`task` and `model` each select exactly one `plugin` or explicit external
`factory="package.module:create_backend"`, plus a `config` object. `nodes`
contains server and service settings. `run` is the shared trainer configuration.
An experiment may `extends` another JSON file; objects merge recursively and
lists replace. `${project_root}`, `${experiment_dir}` and `${env:NAME}` are
expanded, including parent-file-relative variables. Unknown plugin names,
inheritance cycles and unresolved variables fail before loading a model.

`prepare` emits `simulation.json`, `inference.json`, `trainer.json` and a source
manifest. Existing different resolved files are never overwritten. The three
older service/trainer scripts can consume these files directly. On remote
machines, resolve the template there or override paths explicitly. Server bind
addresses and runner endpoint addresses are separate; use `run.endpoints` or
`--inference` / `--simulation` for cross-machine routing. Store service secrets
in the configured environment variables, not experiment files.

## Evaluate, collect and smooth

Use the same simulation service and model service for all operations:

```bash
python scripts/run_experiment.py eval --config configs/experiments/pap_walloss_eval.json \
  --episodes 100 --output outputs/my_base_eval/run
python scripts/run_experiment.py collect --config configs/experiments/pap_walloss_eval.json \
  --episodes 10 --output outputs/my_collection/run
```

For a learned policy, use its matching DSRL or action-residual experiment with
`eval --resume /absolute/checkpoint.pt`. Evaluation automatically selects full
terminal observations and disables the training-only fused reset optimization;
it does not update weights. The source model/task/normalization/action contract
must still match. Set `task.config.smoothing` to `{"mode":"joint_ema","alpha":0.8}`
for PAP/table joint smoothing, or `{"mode":"none"}` to disable it. Elevator
retains its task-specific pose/control smoothing configuration. There is no
implicit conversion between Cartesian and joint actions.

`collect` attaches a task-independent observer to the same validated rollout,
without adding another policy execution loop. It writes initial observations,
requested action prefixes, next/terminal observations, rewards, outcomes and
actual duration into `pressb.chunk_trajectory.v1`. Its default destination is
beside the run directory with suffix `_trajectories`. The format is explicitly
**chunk-level**, not a 30 Hz LeRobot export: a terminated action chunk may be
partially executed, and the recorder never invents missing intermediate camera
frames. Existing frame-synchronized expert collection and LeRobot conversion
remain available through their original commands. Set
`collection.include_images=false` for compact numeric records.

## Task and model contracts

A task plugin provides `describe`, `reset(env_ids, seed, options)`, `step` and
`close`. Optional `step_and_reset` must be explicitly advertised. It owns scene
objects, randomization, measured observations, physical success/reward and
termination. The runtime owns run/episode/step IDs, retry idempotency, validation,
networking and learner scheduling. Time discounts use actual simulated duration.

A model adapter provides `describe`, `encode_batch`, `decode_batch` and `close`.
It owns frozen weights, normalization, feature extraction and external initial
noise. Encoded contexts remain on the inference node; the trainer receives
features and context references. Noise dimensions and executable action
dimensions are independent. Joint adapters publish radians/metres at the node
boundary, even when a historical model was trained in degrees/centimetres.

| Plugin | Execution contract |
|---|---|
| `pap` | Shared world, indexed zero-tick resets, equal prefixes across active slots, tested 64-environment profile |
| `pap_process` | Independent Isaac processes for configured object variants |
| `pap_recording` | PAP vector backend with task-specific measured trajectory capture |
| `table_buttons` | One environment per node, near/medium/far and red/yellow/blue randomization, state7 with native action6 or action7 |
| `elevator` | Native shared-world execution, all active slots and full seven-action chunks |
| `elevator_remote` | Explicit bridge to an already running historical elevator service |

Table buttons currently validates `num_envs=1`; it does not silently pretend to
vectorize. PAP's calibrated cameras and table-buttons' elevator-aligned wrist
camera retain their distinct task configurations. Changing a task/model pair is
allowed only when the negotiated observation and action contracts agree.

## Compatibility and checkpoints

Old pure-generic modules re-export the same module objects as `runtime`, so old
imports, monkeypatches and checkpoint class references keep working. Mixed
historical helpers retain their task-specific code in the compatibility package.
Existing per-node `backend.factory` configs and historical CLI flags remain
supported. New code should import the canonical modules and select a plugin.

Source fingerprints intentionally change when implementation files move. A
checkpoint that binds the old source fingerprint is not silently accepted as a
new implementation. `scripts/migrate_rl_checkpoint.py` creates a separate,
audited derived checkpoint for a verified source-only migration; the original
checkpoint and its replay remain untouched. It rejects changes to task rules,
model weights, normalization, dimensions or learning configuration. Migration
must name the before/after negotiated identities and explicitly acknowledge the
source revision change with supporting verification evidence. Ordinary loading
and resume retain their strict checks. See the script's `--help` for frozen-DSRL
chain handling.

Future RL checkpoints default to **every 100k transitions, retaining all numbered
files**, plus a final checkpoint on a normal stop. Exact simulator continuation
is not claimed: optimizer/replay/RNG restore, then physics starts fresh episodes.
The 2026-11-06 PAP residual run was stopped by user request at **16,704 transitions**
with a full checkpoint; this refactor does not resume it automatically.

The frozen DSRL and paused residual checkpoints have verified derived copies in
`outputs/framework_refactor_20261106/migrated_checkpoints/`. Its
`verification.json` records exact model/optimizer/replay/RNG preservation and
successful strict loading. `resume_experiment.json` is a validated, unlaunched
configuration that continues to the original total budget of 200k transitions,
writing to a new output directory. The original checkpoints remain unchanged.

The subsequent model-source integration creates another verified set at
`outputs/model_internalization_20261106/migrated_checkpoints/`. Its
`resume_experiment.json` records the prepared configuration at that stage; the
earlier refactor-only configuration records its earlier source identity. Neither
configuration has been launched automatically.

After the task-directory consolidation, the current prepared resume configuration
is `outputs/task_consolidation_20261106/checkpoint_migration/resume_experiment.json`.
Its derived DSRL (100,022 transitions) and residual (16,704 transitions) checkpoints
bind the real, newly verified PAP64 service identity. Only the simulation source
revision changes; model, task semantics, learning settings, optimizer, replay and
RNG values are preserved. Original and earlier derived files remain untouched.
Strict CPU loading accepts the new copies and rejects the stale identities.
The resume configuration continues toward the original 200k total budget in a
new output directory; validation does not start or resume training.

## Verification

Verification evidence is in `outputs/framework_refactor_20261106/`; the portable
summary is in `docs/results/framework_refactor_20261106.json`. It distinguishes
CPU regression, real model equality checks, physical expert replay and short
online learning checks. Expert replay success and tiny training runs are
functional checks, not new learned-policy success-rate estimates.
The framework-only regression totals 2,049 passing tests across the main and dependency-
specific environments. Real three-node tests cover training, resume, evaluation
and recording, including a short PAP64 / Wall-OSS native batch64 run.
The later [model integration report](results/model_internalization_20261106.json)
records 2,106 main-project tests across the required environments, additional
bundled-source tests, installed-wheel checks and real-weight comparisons.

The [task consolidation report](results/task_consolidation_20261106.json) records
2,137 main-project tests across the required environments and 17 exact comparisons
of complete USD scene serialization and returned metadata against the prior
implementation. Fresh physical tests cover PAP grasp/place (2/2), tabletop-button
expert replay (9/9 across all spacing/color combinations), wrong contacts,
timeouts, gripper motion, and elevator replay (2/2). New builtin-Wall tests cover
the complete two-environment train/resume/eval/collect workflow and native batch64
with 64 distinct environments and indexed reset isolation. An isolated installed
wheel resolves old and canonical imports, all plugins and model sources; five
experiment configurations and 14 modified script entrypoints also pass checks.
These are regression and short functional runs, not a new long training campaign
or learned-policy success-rate comparison. Formal PAP training remains paused.
