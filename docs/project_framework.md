# Runtime, task plugins, model adapters and experiments

The calibrated [Playground](playground.md#derive-another-task-from-this-workspace)
also serves as a reusable workbench scene template. New tasks pin its versioned
scene configuration, add objects with scene overrides or a composed scene builder,
and define their own reset/reward/success logic in a task plugin. The shared
runtime, model adapters and training loop remain reusable. The supplied
`configs/rl_nodes/playground_objects_example.json` demonstrates scene composition;
it retains the workspace backend's zero-reward semantics.

PressB uses one runtime for inference services, environment services, DSRL,
action-residual learning, policy evaluation and chunk trajectory recording.
Choose a task, a frozen model and execution settings in an experiment JSON.
Task physics and outcome rules remain explicit task code.
Wall-OSS and VLA-JEPA inference and supervised-training implementations are now
bundled in PressB; new experiments do not need their separate source clones.
See [bundled models](bundled_models.md) for training profiles and CUDA builds.

## Code ownership

The package root contains only `__init__.py` and the lazy `registry.py` as
Python files. Implementation belongs to an explicit owner:

```text
src/pressb/
  __init__.py, registry.py  # Package metadata and lazy plugin names
  runtime/                 # Task-independent services and online learning
    contract.py            # Dimensions, units, capabilities and identities
    server.py, client.py    # HTTP / binary MessagePack WebSocket services
    inference.py           # Batching and observation-context lifetime
    simulation.py          # reset/step identities, ordering and request validation
    trainer.py, learner.py  # Rollout, replay, SAC, evaluation and checkpoints
    physics.py, assets.py  # PhysX access and verified USD asset relocation
    scene.py, workbench.py # Shared USD primitives, room, table and tripod
    recording.py           # Original-time camera video utilities
    trajectory.py          # Validated chunk transition recorder
  control/                 # Joint filters, finite-array validation and pose math
  sensors/                 # Camera calibration, RGB-D recording and tiled RGB
  robots/piper/            # PiPER FK/IK, pose control, batching and D435 hardware
    isaac.py               # PiPER articulation assembly and cleanup
  tasks/
    base.py                # Structural TaskPlugin interface
    elevator/              # Panel scene, floors, planning, collection and simulators
      wire_protocol.py     # Legacy fixed-640x480 RGB PNG helpers
    pap/                   # Object scene, outcomes, sampling and vector/process simulation
    table_buttons/         # Button scene, placement, motion, collection and evaluation
  models/                  # Frozen-model adapters and loading/inference wrappers
    walloss.py             # Frozen Wall-OSS encoder / flow decoder
    vlajepa.py             # Cartesian VLA-JEPA adapter
    vlajepa_backend.py     # Frozen pose-model loading and cached flow execution
    vlajepa_joint.py       # Native joint6 / joint7 VLA-JEPA adapters
  model_sources/           # Preserved upstream model/data/training source bundles
    walloss/               # Wall-X modules, native training and CUDA sources
    vlajepa/               # Model plus pose/joint and multi-task trainers
  training/                # Common supervised-training launcher
  experiments/             # Configuration inheritance, validation and commands
  compat/                  # Explicit historical application/protocol boundaries
    elevator_rl/           # Legacy elevator runners, inference bridge and logging
    elevator_policy.py     # Fixed-8-mm pose8/pose9 HTTP policy protocol
    wallx.py               # Historical PAP WebSocket client
    table_buttons_policy.py # Historical tabletop HTTP policy clients
```

A component's assumptions determine its owner. Floors, panel geometry, button
success rules, task prompts and fixed task camera placement belong to the task.
Joint filtering, camera intrinsics, RGB-D storage and USD asset relocation do
not need those assumptions and belong to shared packages. PiPER kinematics,
articulation setup and D435 mounting geometry belong to `robots/piper`.

Mixed modules are split at that boundary. `sensors.recording` writes calibrated
RGB-D streams; `tasks.elevator.camera_recording` supplies floor/phase fields and
historical preview names. `robots.piper.fixed_camera` builds hardware at an
explicit pose; `tasks.elevator.cameras` supplies elevator defaults and the +Y
placement requirement. `robots.piper.policy_control` performs bounded IK;
`compat.elevator_policy` validates the historical fixed-gripper HTTP protocol.
`control.pose_protocol` contains only finite-array validation and pose math:
`pose9_to_pose8(pose, *, gripper_width_m)` requires the caller to supply the width.
The legacy 8 mm setting stays with its callers, and fixed-640×480 RGB PNG helpers
live in `tasks.elevator.wire_protocol`. The split preserves task geometry,
units, camera registration and action timing.

### Dependency direction and extension rules

Tasks and model adapters consume the common contracts. `runtime`, `control`,
`sensors` and `robots` must not import concrete task packages, `models`,
`compat` or `model_sources`. The runtime receives backend objects through the registry or
configured factories; it does not select a task by importing its scene. Model
adapters own model-source loading. Upstream model bundles retain their module
names, parameter keys and provenance; framework integration belongs in `models`.

Add a new scene, success rule, planner or collection adapter under
`tasks/<task>/`. Reuse a shared helper only when its inputs and behavior are
independent of that task; keep task defaults in the task adapter. Add robot
hardware or kinematics under `robots/<robot>/`, and task-independent filtering
or sensors under `control` or `sensors`. Add model adapters under `models`, and
select task/model factories through `experiments` rather than branching the
trainer. Historical runners and wire conventions belong in `compat`; new task
implementations should not depend on them. The native elevator backend uses its
own task-local engines, not a historical runner.

### Task package paths

There are no forwarding packages for `pressb.online_rl`, `pressb.rl_nodes`,
`pressb.pick_place` or `pressb.table_buttons`, and no top-level task/helper
stubs. Update external imports and explicit `backend.factory` strings to the
canonical paths below. `configs/rl_nodes/` remains a directory of per-node JSON
templates; its name does not imply a Python package.

| Former module or symbol | Canonical module or symbol |
|---|---|
| `pressb.scene`, `.planning`, `.dataset_planning`, `.dataset_scene` | `pressb.tasks.elevator.<same module>` |
| `pressb.panel_randomization`, `.panel_metadata`, `.policy_layouts`, `.press_prefix`, `.recording` | `pressb.tasks.elevator.<same module>` |
| `pressb.fixed_camera.build_global_camera` | `pressb.tasks.elevator.cameras.build_global_camera` |
| `pressb.camera_recording.WristCameraRecorder` | `pressb.tasks.elevator.camera_recording.WristCameraRecorder` |
| Generic RGB-D writer extracted from `pressb.camera_recording` | `pressb.sensors.recording.RGBDCameraRecorder` |
| `pressb.dataset_scene.build_tiled_rgb`, `.split_tiled_rgb` | `pressb.sensors.tiled.<same function>` |
| `pressb.kinematics`, `.wrist_camera`, `.replay_control` | `pressb.robots.piper.kinematics`, `.wrist_camera`, `.pose_control` |
| `pressb.action_postprocessing`, `.motion_smoothing` | `pressb.control.joint_actions`, `.joint_smoothing` |
| `pressb.scene_portability` | `pressb.runtime.assets` |
| `pressb.runtime.isaac.build_piper_runtime`, `.close_piper_runtime` | `pressb.robots.piper.isaac.<same function>` |
| `pressb.runtime.isaac.make_camera`, `.set_intrinsics` | `pressb.sensors.camera.<same function>` |
| `pressb.policy_eval.PolicyPoseController` | `pressb.robots.piper.policy_control.PolicyPoseController` |
| `pressb.policy_eval.PolicyClient`, `.validate_policy_reply`, `.InvalidPolicyAction` | `pressb.compat.elevator_policy.<same symbol>` |
| `pressb.pick_place.scene`, `.task`, `.evaluation` | `pressb.tasks.pap.scene`, `.task`, `.evaluation` |
| `pressb.pick_place.wallx` | `pressb.compat.wallx` |
| `pressb.table_buttons.<module>` | `pressb.tasks.table_buttons.<module>` |
| Historical `HTTPJointPolicy` / `HTTPJoint6Policy` clients | `pressb.compat.table_buttons_policy` |
| Generic `pressb.rl_nodes` service/client/contract/trainer modules | `pressb.runtime.<same module>` |
| `pressb.rl_nodes.pap`, `.pap_vector`, `.pap_worker`, `.pap_recording`, `.elevator` | `pressb.tasks.pap.backend`, `.vector`, `.worker`, `.recording`, `pressb.tasks.elevator.backend` |
| `pressb.rl_nodes.walloss`, `.vlajepa` | `pressb.models.walloss`, `.vlajepa` |
| `pressb.online_rl.simulation`, `.fast_simulation`, `.trajectory_recording` | `pressb.tasks.elevator.<same module>` |
| `pressb.online_rl.batched_control` | `pressb.robots.piper.batched_control` |
| `pressb.online_rl.inference.FrozenPiperBackend` | `pressb.models.vlajepa_backend.FrozenPiperBackend` |
| Legacy `InferenceApplication`, runners, action postprocessing and log monitoring | `pressb.compat.elevator_rl.<same module>` |
| `pressb.online_rl.protocol.finite_array`, `.pose8_to_pose9` | `pressb.control.pose_protocol.<same function>` |
| Generic `pose9_to_pose8` | `pressb.control.pose_protocol.pose9_to_pose8(pose, *, gripper_width_m)`; width must be explicit |
| Legacy fixed-640×480 `encode_image`, `decode_image` | `pressb.tasks.elevator.wire_protocol.<same function>` |
| Elevator floor parsing / legacy learner protocol | `pressb.tasks.elevator.semantics` / `pressb.compat.elevator_rl.protocol` |
| `pressb.online_rl.learner`, `.rpc` and generic inference transport APIs | `pressb.runtime.learner`, `.rpc`, `.inference_transport` |
| Legacy `InferencePolicyAdapter` / `load_wallx_server` | `pressb.compat.elevator_rl.inference_transport` |

For example, new scripts import `PiperKinematics` from
`pressb.robots.piper.kinematics` and `JointActionPostprocessor` from
`pressb.control.joint_actions`. Task entry points and their existing CLI flags
remain available, but removed Python imports now fail. Custom pickle files that
embed old class paths need conversion under the earlier code version. Dataset
identifiers such as `pressb.table_buttons.raw.v1` are unchanged protocol strings,
not imports. Historical source archives and SHA manifests retain their original
paths and bytes; they are evidence of the code used for those runs.

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

Historical CLI entry points remain available through canonical implementations.
Removed Python packages have no aliases; explicit factory strings must use the
paths above. Supplied per-node templates retain their location under
`configs/rl_nodes`, while new experiments can select a named plugin. Historical
application code is maintained explicitly in `compat`, not at the package root.

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

The current verified migration is in
`outputs/component_ownership_20261107/checkpoint_migration/`. Its
`resume_experiment.json` (a local artifact in that directory)
is validated and **has not been launched**. It retains 64 environments, native
batch64 and the original **200,000 total transitions**, with **183,296 remaining**
after the paused 16,704-transition state. Formal residual training remains paused.

Strict CPU loading verified exact model parameters, optimizer, replay and RNG
preservation for the DSRL (100,022 transitions) and residual (16,704 transitions)
checkpoints. Original and prior derived files remain unchanged; ordinary strict
loading still rejects their stale source identities. Verification performed no
formal training updates. The new checkpoint SHA-256 values are:

| Derived checkpoint | SHA-256 |
|---|---|
| `dsrl_step_000000100022.pt` | `f4fa61b544fd41ac6c7303a2578983607e08813a9db0c18c45bef5eefdea56c0` |
| `residual_step_000000016704.pt` | `08bbe6d04a90c32eff00b90411fcee606095afdc931b8dcdae16c830cf962188` |

The [portable verification summary](results/component_ownership_20261107.json)
links the immutable evidence and its hashes. Earlier verified migration stages
below remain historical records, not the current resume configuration.

The initial framework refactor produced verified derived copies in
`outputs/framework_refactor_20261106/migrated_checkpoints/`. Its
`verification.json` records exact model/optimizer/replay/RNG preservation and
successful strict loading. `resume_experiment.json` is a validated, unlaunched
configuration that continues to the original total budget of 200k transitions,
writing to a new output directory. The original checkpoints remain unchanged.

The subsequent model-source integration created another verified set at
`outputs/model_internalization_20261106/migrated_checkpoints/`. Its
`resume_experiment.json` records the prepared configuration at that stage; the
earlier refactor-only configuration records its earlier source identity. Neither
configuration has been launched automatically.

The task-directory consolidation produced the historical resume configuration
`outputs/task_consolidation_20261106/checkpoint_migration/resume_experiment.json`,
bound to that stage's real PAP64 service identity.

The preceding removal of the top-level PAP/table-button packages produced the
then-prepared resume configuration
`outputs/task_directory_cleanup_20261106/checkpoint_migration/resume_experiment.json`.
Its derived DSRL (100,022 transitions) and residual (16,704 transitions) checkpoints
used that stage's updated PAP source identity. Model parameters, task semantics,
learning settings, optimizer, replay and RNG values were preserved; original and
earlier derived files remain untouched. Strict CPU loading then accepted those
copies and rejected stale identities. The source identity was verified in a new
real two-environment pipeline; offline loading used that actual metadata. The prepared
configuration retains 64 environments, batch64 and the original total budget of
200k transitions through the existing capacity-independent policy fingerprint.
It writes to a new output directory and has not been launched. This cleanup did
not repeat the prior 64-environment GPU test.

## Verification

The current ownership split has passed the following checks against the final
source snapshot `627ce9acdae9a6bf8fc27eeebe610728a58f71ef3d079939f5cd4cbeba6f96f6`:

- **2,210 unique tests passed**, with every dependency skip resolved: 2,135 in
  the main environment, 71 USD/collision, one LeRobot and four VLA tests, with
  one duplicate excluded from the total.
- A clean installed wheel passed **68 canonical imports and 50 CLI/config
  checks**. Removed packages are absent, and all **169 bundled model/training
  files** remain unchanged.
- **25 scene cases** produced identical complete USD text and metadata;
  RGB-D recording produced **29 byte-identical files**. The persisted pose/wire
  comparison covers **163 poses** with the explicit 0.008 m width and **three
  PNG cases**, preserving exact legacy outputs.
- Physical expert replay passed for **PAP 2/2**, **table buttons 9/9** across
  spacing/color combinations and **elevator 2/2**. Wrong-contact, timeout and
  joint7 gripper checks also passed.
- The real Wall two-environment pipeline passed DSRL training **40→48** with
  resume, fixed-DSRL residual training **40**, base/residual evaluation and
  **two complete collected episodes** with **26 transitions / 104 PNGs**.
  A final-source check confirms its 112 recorded dependencies were unchanged.
- Native **batch64** execution passed **192 transitions / 39 updates**, with
  four 64-item encoder calls and three 64-item decoder calls. All 64 object
  positions and 128 camera PNGs were independently checked; resetting one slot
  preserved the other 63. This short run spans 3.2 simulated seconds and does
  not reach the 20-second episode limit; completed terminal episodes are covered
  by the separate two-environment run. All services exited with code 0 and
  released their ports.
- Checkpoint migration and strict CPU restore passed as described above,
  preserving learning state and retaining all original/prior checkpoints.

The [portable summary](results/component_ownership_20261107.json) indexes the
local evidence in `outputs/component_ownership_20261107/`, including
`validation_ready.json` (local artifact).
These are functional regressions, not policy success-rate or throughput
measurements. Formal PAP residual training remains paused. The reports below
describe completed earlier revisions and keep their original identities and scope.

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

The subsequent [directory cleanup report](results/task_directory_cleanup_20261106.json)
records **2,127 unique passing tests**, with all dependency skips resolved. The
count changes because 25 old alias/pickle cases were replaced by 15 canonical
ownership, removed-package, dependency-boundary and archive checks. A clean
wheel build and isolated installation contain neither removed package; canonical
imports, both task scenes, nine plugins and 24 CLI/config checks pass. All 169
tracked model/training files remain byte-identical. A new real two-environment
Wall/Isaac workflow covers DSRL updates/resume, frozen-DSRL residual updates,
evaluation and collection, followed by strict offline verification of the paused
formal checkpoints. The final migration evidence also records an interrupted
attempt after a provenance correction and a validation-only retry after fixing a
PyTorch guard in the offline harness; neither required a product-code change.

## Physical environment and human outcomes

The fourth service role, `pressb real_robot`, supplies the same environment contract as simulation with `num_envs=1`. Hardware adapters belong to `robots`, capture drivers to `sensors`, physical execution to `runtime/real_robot.py`, and replaceable reset/reward hooks to a configured factory. The trainer optionally replaces environment outcomes with episode-scoped human labels and acknowledges external termination before reset. See [real robot setup and verification scope](real_robot.md). Existing simulation configuration keys remain compatible.
