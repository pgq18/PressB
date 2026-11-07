# Configurable inference, simulation and RL nodes

**2026-11-07 ownership update:** service and learning code lives in `runtime`, task implementations in `tasks/{elevator,pap,table_buttons}`, and model integration in `models`. PiPER hardware/control, generic filters and sensors live in `robots/piper`, `control` and `sensors`. The `pressb.rl_nodes` and `pressb.online_rl` packages are removed without aliases; `configs/rl_nodes/` remains the per-node template directory. New experiments use [the unified configuration/entry point and path map](project_framework.md). Historical elevator orchestration lives in `compat/elevator_rl`.

The current ownership refactor has passed 2,210 unique tests, installed-package checks, physical replay for all three tasks, the two-environment Wall train/resume/eval/collect pipeline and native batch64 checks. See the [current verification summary](results/component_ownership_20261107.json) for scope and evidence. Strict checkpoint migration passed; the local resume configuration at `outputs/component_ownership_20261107/checkpoint_migration/resume_experiment.json` is validated but unlaunched, retaining 64 environments, batch64 and the original 200,000 total budget (183,296 remaining). Formal PAP residual training remains paused at 16,704 transitions. Earlier test and GPU results below retain their original dates and scope.

This interface separates frozen model inference, physical environment execution
and online learning. Public `method="dsrl"` means learning the flow model's
initial noise; the legacy learner/checkpoints still call this `initial_noise`.
The existing elevator experiment entry points and the native Wall-X observation
and RTC interfaces remain available.

For the opt-in 64-environment PAP performance profile, see
[PAP DSRL pipeline optimization](pap_dsrl_pipeline_optimization.md). It adds
negotiated image preprocessing, terminal state-only handling, and atomic
`step_and_reset` without changing the default `reset`/`step` interfaces.

The implementation includes a Wall-OSS adapter and a PAP Isaac worker backend.
The original 2026-11-04 validation described below includes CPU tests for real sockets, independent worker processes, actual SAC updates,
checkpoint resume, and the Wall-OSS flow code's cached/uncached equivalence.
That validation also passed GPU integration with the full PAP Wall-OSS checkpoint and two real
Isaac worker processes. Both DSRL and action-residual runs collected 24 chunk
transitions and performed 22 SAC updates, including timeout/reset and checkpoint
saving. These short runs used a two-second episode limit to exercise lifecycle
handling; they do not measure task success or learning effectiveness.

## Responsibilities and flow

| Node | Owns | Interface |
|---|---|---|
| Inference | Frozen weights, preprocessing/normalization, observation features, encoded contexts, flow decoding | `health`, `encode`, `decode`, `infer`, `release` |
| Simulation | Task scene, parallel environments, reset randomization, executable commands, smoothing, physical rewards/outcomes, camera observations | `health`, `status`, `reset`, `step`, `close_run` |
| Training | DSRL or residual actor/critics, replay, exploration, sampling requests, asynchronous SAC updates, checkpoints and metrics | Configured outgoing node clients; JSONL/status files and SIGINT/SIGTERM stop |

```mermaid
sequenceDiagram
    participant T as Training node
    participant S as Simulation node
    participant I as Frozen inference node
    T->>S: health; reset selected environments
    S-->>T: observations + episode/step identities
    T->>I: health; encode observations
    I-->>T: features + context IDs (+ base actions for residual RL)
    Note over T: Choose initial noise or action residual
    T->>I: decode(context IDs, initial noise), for DSRL
    I-->>T: executable action chunks
    T->>S: step(selected environments, action prefixes)
    Note over T: Update SAC from replay while RPC is pending
    S-->>T: terminal/next observations, rewards, duration discounts, done flags
    T->>I: encode next observations before any reset
    Note over T: Store transitions; release old contexts; reset ended slots
```

The trainer initiates requests to the other two nodes. The simulator never calls
the model, and inference never advances physics. Network waiting does not add
simulation time. The three processes can live on different machines; only the
inference machine needs the model weights, and only simulation workers need
Isaac/assets/initial-state data. The training machine needs PyTorch and replay
storage.

### Action residual training on a frozen DSRL policy

Set `method="action_residual"` and add `frozen_dsrl` with a trusted
`checkpoint`, optional expected `checkpoint_sha256`, and explicit
`deterministic` sampling mode. `checkpoint.resume` belongs to the new residual
learner; leave it null to start its actor, critics, optimizers and replay afresh.
The noise-policy checkpoint is loaded separately and its original DSRL
model/task fingerprint must match the connected nodes. The residual checkpoint
also binds the noise checkpoint hash, learner configuration and sampling mode.

The trainer encodes each observation once without requesting ordinary base
actions, applies the frozen DSRL actor to the negotiated observation vector,
and decodes its noise once. That raw decoded chunk conditions the new residual
actor and is retained in replay. Each `next_base_action` is the same chunk used
for that next observation's execution. The residual is added before final action
clipping, matching the critic and actor-loss composition. Frozen actor weights
are checked at checkpoints and normal training completion. `initialization.json`
records the new learner's initial counters and parameter identities.

For the local 64-environment PAP experiment, use
`configs/rl_nodes/pap_residual_on_dsrl_local_fast.json` with the existing fast
simulation and inference configurations. It freezes the final 400k DSRL actor
in deterministic mode and trains 400k new residual transitions. PAP residuals
cover six joint angles and full gripper width, with scales
`[0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.005]` in radians/metres. The first 2k
transitions use zero residual; the probability of applying a sampled residual
then increases over 50k transitions. Gamma remains 0.995 per 30 Hz control step,
and no additional action smoothing is enabled.

## Capability negotiation

Each node publishes `schema_version=2`. The surrounding HTTP/MessagePack
transport envelope remains version 1, so these two version numbers describe
different things. The trainer checks:

- State component names/order/units and camera names.
- Action representation, reference frame, names/order/units, control rate and
  supported execution length.
- Model feature dimension and **independent** latent noise shape.
- Simulator control-state/task-feature dimensions, discount timebase and
  indexed-reset support.
- Model/configuration/task content identities when resuming a checkpoint.

There are no implicit degree/radian, quaternion/rotation6D, joint/TCP or camera
conversions in the trainer. A named adapter performs these conversions and
advertises the resulting semantics. A mismatch fails before rollout.

| Adapter pair | Observed state | Executable model action | Initial noise |
|---|---|---|---|
| PAP + Wall-OSS | 6 joint angles in rad + full gripper width in m | `32×7`, absolute joints/gripper in rad/m | `32×20` |
| Elevator + original VLA-JEPA | TCP xyz, wxyz quaternion, gripper width | `7×9`, xyz + row-major rotation6D; simulation adapter converts to pose8 | `7×9` |

Wall-OSS internally retains all 20 latent channels. The 13 channels not exposed
as robot actions are not set to zero: they can influence the learned flow.
Its adapter converts SI observations to the checkpoint's degrees/centimetres
and converts output actions back to SI. It uses the **checkpoint's**
`norm_stats.json`; it does not substitute the raw dataset's statistics.

Wall-OSS features pool valid pre-noise multimodal embeddings. Its cache stores
these embeddings and RoPE indices, **not Transformer KV states**. The adapter
uses true native model batches: one visual encode handles the observations in
each service microbatch, and each Euler step runs one action-transformer forward
for that microbatch. Different prompt lengths are padded and masked; each cache
entry owns its unpadded tensors and seeded baseline noise. Entries can be decoded
in a different order or grouping without borrowing another observation's state.
The elevator adapter uses the existing batched VLA-JEPA encoder/decoder.

Set `service.batch_size` to control the maximum observations per model batch.
For example, 64 observations with `batch_size=16` become four native batches of
16. Health metadata advertises `batching.native_model_batching=true` and the
configured `microbatch_size`; this setting is not a claim that every batch size
fits every GPU. The action transformer uses eager attention to preserve padded
sequence semantics. Batched vision attention groups observations by their total
visual token width and retains each observation's original image/window attention
mask. It avoids a dense attention matrix spanning all observations while keeping
the original per-observation softmax width. Batch size one uses the original
visual path. These APIs use diffusion prediction; the legacy singleton and RTC
interfaces remain available.

Identical initial noise does not guarantee bitwise-identical BF16 outputs across
model batch sizes. Real-checkpoint comparisons have shown action differences;
changing batch size therefore needs measured action-error comparisons and
closed-loop task-success evaluation. Preserving each context's seed and state
does not establish identical policy behavior across batch sizes.

## Launch PAP

Run commands from the PressB root. The supplied configuration paths describe the
current workstation; on another node, change paths to that machine's local
assets. The Wall-X repository contains model changes needed by this adapter:
`flow_context.py`, `vision_batch.py`, the Qwen vision/action models, `WallXPolicy`,
policy preprocessing and `csrc/window_index.cu` must travel together with PressB.
These files are included in the adapter's source identity. The CUDA extension
must also be rebuilt: the window-index kernel fixes a cross-block race when
there are more than 256 windows, and copying source alone leaves the old compiled
kernel in use. See the H200 rebuild command and deployment notes in the
[adapter notes](https://github.com/pgq18/wall-x/blob/base/inference/README_PressB_RL.md).

These commands start services and a **short connectivity smoke run**. The GPU
integration tests used separate copies of these configurations with the smaller
warmup and episode limits described below.

```bash
# Inference machine, dedicated model environment; example physical GPU 1.
CUDA_VISIBLE_DEVICES=1 .conda/envs/wallx-inference/bin/python \
  scripts/serve_inference_node.py --config configs/rl_nodes/pap_walloss_inference.json

# Simulation machine; two separate Isaac processes on GPU 0 in this example.
.conda/envs/pressb/bin/python scripts/serve_simulation_node.py \
  --config configs/rl_nodes/pap_simulation.json

# Training machine; example GPU 0. Explicit 20-transition smoke budget.
.conda/envs/pressb/bin/python scripts/run_rl_training.py \
  --config configs/rl_nodes/pap_dsrl.json \
  --max-transitions 20 --output outputs/pap_dsrl_smoke
```

The 20-transition command checks rollout and checkpoint wiring; the supplied
`learning_starts=2000` means it does **not** yet perform SAC updates. For a short
update test, copy the trainer JSON, set `learning_starts=4`, `batch_size=4`, and
use at least 20 transitions. Full example budgets are 400k chunk transitions;
the hyperparameters are starting configurations, not validated PAP settings.

The H200 deployment configuration is
[`pap_walloss_h200_inference.json`](../configs/rl_nodes/pap_walloss_h200_inference.json).
It sets a native microbatch of 64, a 256-entry cache, and a loopback WebSocket
endpoint on port 18130; its checkpoint path is local to H200. Launch it with the
same `scripts/serve_inference_node.py --config ...` entry point in the H200
Wall-OSS environment. The [H200 batch report](walloss_batch_inference.md) records
measured throughput, memory and BF16 action differences from batch 1 through 64.
Set the service batch size according to those measurements and closed-loop task
evaluation; the tested batch size does not establish unchanged success rates.
The loopback endpoint can be reached from another machine through an SSH port
forward, or the server's bind interface and authentication can be configured
explicitly.

Use `pap_action_residual.json` for residual RL or `pap_base_eval.json` for the
frozen model baseline. Base evaluation uses the same generic execution path,
action projection and configurable smoothing. Learned evaluation uses the
matching training config with `--mode eval --resume <checkpoint.pt>`; it never
updates parameters. Use `--output` to preserve previous results.

For separate machines, configure `server.host` to a reachable interface, change
the trainer's inference/simulation URLs, and keep the backend's asset paths local
to its own machine. Each server accepts optional `server.token_env`; the trainer
reads `PRESSB_RL_TOKEN` (or `--token-env`). WebSocket sends the existing Wall-X
`__pressb_rpc__` binary MessagePack envelope. HTTP remains available by setting
`server.transport="http"` and using an `http://` endpoint.

WebSocket message capacity is configurable on both sides: set inference
`server.max_message_bytes` and trainer `endpoints.max_message_bytes` to the same
positive integer. Both default to `67108864` (64 MiB), preserving the previous
limit. The 64-environment H200 launch uses `201326592` (192 MiB) for image-bearing
requests and responses. This setting changes the WebSocket limit; it does not
change the existing HTTP server's 64 MiB request-body limit or resize images.
Increasing the limit does not split the model batch or alter its precision.

The deployment record for the **fresh 400k-transition PAP DSRL run** is
[`outputs/pap_dsrl_h200_bf16_64env_400k_20261104/README.md`](../outputs/pap_dsrl_h200_bf16_64env_400k_20261104/README.md).
Its configuration places frozen BF16 Wall-OSS inference with native batches of
up to 64 on H200 GPU 0, 64 shared-World PAP environments on workstation GPU 0,
and SAC on workstation GPU 1. The local inference URL is an SSH forward to the
H200 loopback service. The run uses random red-cube positions inside the marker
edges, the existing PAP cameras, no temporal action smoothing, and independent
`32×20` initial-noise coordinates scaled by 1.5. Its discount is 0.995 per 30 Hz
control step, so a full 32-step chunk uses `0.995**32`. This is a training
configuration, not a claim of improved task success.

## Task and training configuration

Configuration files are in [`configs/rl_nodes`](../configs/rl_nodes/).

| Configuration | Important settings |
|---|---|
| Inference `backend.config` | Model/checkpoint/config/normalization paths, device/dtype, camera mapping, input resolution, flow steps, optional CUDA allocator memory fraction |
| Inference `service` | Context capacity/TTL and request microbatch size |
| PAP simulation `backend.config` | Scene/initial trajectory, environment count/GPU assignment, randomization, reward, timeout, smoothing |
| Trainer `rollout` | Budget, execution prefix, selected environment count, reset options, recent-SR window, action-bound handling |
| Trainer `learner` | Noise steps/scale, residual indices/scales, SAC hyperparameters, entropy, truncation bootstrap |
| Trainer `update` | Replay capacity, warmup, update-to-data ratio, bounded update backlog, overlap update limit |
| Trainer `checkpoint` | Save interval (default 100,000 transitions), retention (`keep: null` retains all), replay inclusion, resume path |
| Trainer `endpoints` | Inference/simulation URLs, timeout and WebSocket message capacity |
| Trainer `wandb` | Optional monitoring project/entity/name, online/offline mode, tags and upload interval |

PAP keeps its current dataset-aligned cameras and initial observed joints.
`randomization.position="markers"` samples inside the rectangle formed by the
four green marker **edges**. Colors can be `red`, `green`, `blue`, `white`,
`yellow`. An `objects` list can configure cube, cuboid or cylinder geometry;
`object_indices` selects reset candidates. Per-reset `options.by_env` can supply
different object positions/colors/tasks to individual environments.

PAP success retains the physical criterion: bilateral finger contact history,
actual lift, the entire object inside the box, released gripper, no current
finger contact and stable rest on the box floor. Success terminates; dropping
below the table can terminate; timeout truncates. Infrastructure errors are
reported as errors, never silently counted as task failures.

Optional smoothing lives **in simulation**, under
`"smoothing":{"mode":"joint_ema","alpha":0.8}`; default `mode="none"`.
It smooths the six arm joints and preserves gripper commands. Each worker owns
its filter state; the last sent target is part of `observation.control_state`,
so the learner observes the state that affects future execution.

The PAP residual example trains all seven outputs. Its scales are 0.05 radians
per arm joint and 0.005 metres of full gripper width. To keep the gripper frozen,
select `residual_indices=[0,1,2,3,4,5]` and provide six scales. These are joint
residuals, not the elevator's XYZ-metre residuals.

`rollout.action_bounds="clip"` explicitly projects commands to the simulator's
declared joint/gripper limits. The PAP examples select this mode. The residual
learner uses the same projection for current/target/actor critic inputs and
records projection counts and maximum displacement. Default `"error"` rejects
out-of-range actions. Projection is separate from temporal smoothing.

DSRL chooses the **entire initial flow state**: the actor output in `[-1,1]` is
scaled by `noise_scale`. `noise_steps=32` learns all `32×20=640` PAP latent
coordinates; `noise_steps=1` repeats one 20-dimensional vector across time.
The latter is a different latent parameterization, not independent Gaussian
noise. Warmup uses the learner's bounded normalized exploration, while the
frozen-model baseline uses seeded Gaussian noise. DSRL is not Gaussian noise
plus an additive learned perturbation in this implementation.

One transition is one executed action prefix, not one episode or physics tick.
The simulator reports the actual duration, including early termination:
`discount = gamma ** (duration_seconds * timebase_hz)`.
PAP uses 30 Hz control and 120 Hz physics; a complete 32-action chunk lasts
128 ticks and has discount `gamma**32`. Rewards are discounted at their event
time within the chunk.

## Parallelism, recovery and logs

PAP offers two simulation backends. The original `pap_simulation.json` runs
independent Isaac processes concurrently; each owns one environment
and renderer. Startup is serial to limit initialization pressure. Requests
dispatch to selected workers together and return after those workers finish.
An indexed reset advances only the reset worker during settling. This provides
parallel sampling and independent episode reset, but is not a shared-renderer
GPU vector backend or a fully asynchronous per-environment rollout scheduler.
Increasing `num_envs` increases Isaac memory usage per environment; no automatic
GPU-memory tuning is claimed.

For higher parallelism, [`pap_vector_simulation.json`](../configs/rl_nodes/pap_vector_simulation.json)
uses one Isaac process, a shared World, batched PhysX tensor access and tiled
cameras. **64 environments with 128 native 640×480 views passed a real GPU
capacity test**, using about 15 GiB of simulator GPU memory. A recorded successful
grasp-and-place trajectory also succeeded in all 64 environments; this checks
physical execution rather than learned-policy success rates. See the
[capacity report](pap_parallel_capacity.md) for settings, timings and limitations.

This backend requires every active environment to receive an equal-length action
prefix in each step. Episodes still reset independently, with zero physics ticks
during indexed reset. Startup-settled joints are restored and the object is placed
on its support surface; this reset contract has a different task fingerprint from
the process backend. One object geometry is configured per service, with position
and color randomization. The default keeps 48 render-only warmup captures after
reset; lowering this count can leave image-history artifacts. Physics currently
uses CPU PhysX with tensor access, not GPU-native dynamics. Wall-OSS model batch
size is configured separately from simulation parallelism, as described above.

The measured 64-environment native-RGB reset response was 63.12 MiB when JSON
serialized. That size is close to the default inference WebSocket limit;
the simulator's HTTP response does not itself enforce that limit. The forwarded
inference request must be measured separately. The original capacity tests called
the service directly; they did not qualify a complete network/inference/training
run. The later H200 launch configures 192 MiB on both WebSocket endpoints and
uses a separate 256-transition preflight with SAC updates before fresh long-run
training. The larger limit leaves room for changes in PNG sizes as views change,
while preserving native camera images and the batch-64 request grouping.

The trainer overlaps replay updates with blocking network requests. Actor
reads, optimizer writes and replay writes stay on one owner thread. Update
backlog is bounded and dropped credit is logged. Cached contexts are isolated,
bounded by count/TTL and released after transitions are stored. Provision cache
capacity for at least twice the active environment count because current and
next contexts coexist.

Every physics request has a run ID, request ID, episode ID and step ID. Repeated
cached requests return the original response; expired request IDs are rejected
without repeating physics. Lost responses are never automatically retried. An
unknown physics state requires a simulator restart. Checkpoint resume restores
learner/optimizer/replay/RNG, then starts **fresh physical episodes**; it does
not claim bitwise continuation of Isaac state.

`status.json` contains recent SR, lifetime SR, transition/update throughput,
replay size, update backlog, losses, RPC/learner time and projection statistics.
`metrics.jsonl`, `episodes.jsonl` and `events.jsonl` retain the history.
`negotiated.json` records both node identities and the negotiated contract.
`checkpoint.json` points to the latest checkpoint. New training runs default to
`checkpoint.interval=100000` and `checkpoint.keep=null`: save every 100,000
transitions and retain **all** `step_*.pt` files. A positive integer `keep` is an
explicit opt-in to pruning; it is no longer enabled in the supplied training
templates. Normal completion and SIGINT/SIGTERM also save a checkpoint, even
when the final transition count is between scheduled saves. SIGINT/SIGTERM
stops after the outstanding request/chunk, saves the learner, releases contexts
and closes the simulation session. Existing run configurations and previously
saved checkpoints are not rewritten by this default change.

Scheduled saves use absolute interval boundaries: a batch crossing 100k does
not push the next save beyond the 200k boundary. Filenames contain the actual
transition count. Backends requiring all active environments can slightly
exceed the final budget; `transition_budget_overshoot` records that difference.

The historical elevator runner in `compat/elevator_rl` and its templates also
default to 100,000
transitions and retain their numbered checkpoints. The older fast runner keeps
its existing 100,000-transition permanent checkpoints plus a separately
overwritten `last.pt` every 10,000 transitions for recovery; updating `last.pt`
does not remove the permanent checkpoints. Its logging remains unchanged.

### Optional W&B monitoring

The generic trainer now accepts an optional top-level `wandb` section:

```json
{
  "wandb": {
    "enabled": true,
    "project": "pressb-online-rl",
    "entity": null,
    "name": "pap-dsrl",
    "mode": "online",
    "tags": ["pap", "dsrl"],
    "log_interval_seconds": 5,
    "init_timeout_seconds": 20
  }
}
```

Monitoring is disabled by default. With `enabled=true`, the trainer starts a
separate uploader process in the learner's Python environment; install W&B and
authenticate there using the existing W&B login or environment. `mode="offline"`
records a local W&B run. The uploader samples atomic `status.json`; the complete
per-batch history remains in `metrics.jsonl`. W&B initialization, networking and
SDK exit hooks do not run in the learner process. Shutdown waits up to five
seconds for the uploader, then uses bounded termination if needed. An upload
failure leaves local training/logging available and is recorded in
`wandb_status.json`; final remote metrics can be missing if shutdown times out.
This file also records the online run URL when initialization succeeds.

The dashboard exposes `rollout/recent_sr` over the most recent completed
episodes (100 in the PAP example), lifetime SR, episode counts, transition/update
throughput, replay size, update backlog and dropped credits. `learner/*` contains
finite SAC metrics. Rollout curves use `train/transitions` as their horizontal
axis; learner curves use `train/updates`. No SR is reported before an episode has
finished. Endpoint URLs and the RPC token are excluded from uploaded configuration.

## Existing elevator and new adapters

The `elevator` experiment plugin selects the native task engine at
`pressb.tasks.elevator.native:create_backend`. The separate `elevator_remote`
plugin selects `pressb.tasks.elevator.backend:create_backend` for historical
service bridging. `elevator_inference.json` selects the canonical
`pressb.models.vlajepa:create_backend` adapter; its explicit source override, if
present, selects a historical model checkout instead of the bundled default.
The per-node `elevator_simulation.json` template connects to an already launched, unowned
[`serve_rl_fast_simulation.py`](../scripts/serve_rl_fast_simulation.py) service;
its endpoint must match that service's port. This compatibility adapter retains
the old simulation's requirements: full 7-action chunks, all active slots per
step, all environments in the initial reset. If a run abandons live episodes,
restart the legacy simulator and the generic simulation node before another
run. Legacy RL checkpoints should continue to use their original experiment
entry points; the new runner requires its negotiated schema fingerprint.

To add another task/model, implement a factory `module:function(config)` and
select it in JSON. Model backends provide `describe`, `encode_batch`,
`decode_batch`; environment backends provide `describe`, `reset`, `step`,
`close`. See [`contract.py`](../src/pressb/runtime/contract.py),
[`walloss.py`](../src/pressb/models/walloss.py) and
[`backend.py`](../src/pressb/tasks/pap/backend.py). Metadata cannot supply missing physics
or model decoding logic: a new task still needs its success/reward/reset adapter.
The tabletop-buttons six-joint checkpoint now uses the `vlajepa_joint` model
adapter and the `table_buttons` task plugin. Historical commands retain their
explicit implementations in `compat`; removed import paths are not supported.
Keep task defaults and reward/success rules in the task package. Shared runtime,
control, robot and sensor helpers must not import concrete tasks, `models`,
`model_sources` or legacy applications; model-source loading belongs to adapters.

## Wire reference

Observations contain `state`, `images` (camera-name →
`{"encoding":"png_base64","data":"..."}`), `task`, optional `model_inputs`,
`control_state` and `task_features`.

| Operation | Request | Response |
|---|---|---|
| `encode` | `observations`, one `seed` per item in `seeds`, `include_base_actions` | `items`: `context_id`, `features`, optional `base_actions` |
| `decode` | `items`: `context_id`, optional `initial_noise` | `items`: `context_id`, `actions` |
| `infer` | `observations`, `seeds`, optional `initial_noises` | `items`: `actions`; no persistent contexts |
| `release` | `context_ids` | Release acknowledgement |
| `reset` | `run_id`, `request_id`, `env_ids`, `seed`, `options` | `items`: environment/episode/step IDs, observation, info |
| `step` | `run_id`, `request_id`, `items`: IDs plus executable `actions` | `items`: IDs, observation, reward, discount, terminated, truncated, info |
| `close_run` | `run_id` | `closed`, `restart_required` |

Terminal observations remain terminal; no reset observation is substituted into
replay. `info.success`, `info.physics_steps` and `info.elapsed_control_steps` are
required, with task-specific diagnostics added by the environment backend.

## Validation on 2026-11-04

The PressB suite passes **1,491 tests, with 13 skipped**; the Wall-X native flow
test suite passes **3 tests**. After the integration runs, the inference cache
was empty, simulation ownership was released, and all smoke services/workers
were stopped normally. A compact [validation summary](results/rl_nodes_20261104.json)
is stored with this documentation.

Artifacts are in `outputs/rl_nodes_smoke_20261104/`:

- `walloss_real_model_checks.json`: full checkpoint, real Isaac camera images;
  cached versus native inference and repeated baseline decoding both have zero
  maximum absolute error. Changing initial noise changes actions (maximum change
  0.44615 in the mixed radian/metre output), while observation features remain
  unchanged. The model is frozen, with features `[2048]`, noise `[32,20]` and
  actions `[32,7]`.
- `pap_parallel_real_smoke_report.json`: two Isaac processes execute concurrently;
  resetting or advancing one environment does not reset or advance the other.
- `real_dsrl_run/` and `real_action_residual_run/`: each completed 24 transitions,
  22 SAC updates and two timeout episodes, with finite losses and checkpoints.
  Inference and simulation ran on GPU 0; learning ran on GPU 1. Episode duration
  was deliberately limited to two seconds, and each command executed at most
  eight control steps. These are functional checks, not throughput benchmarks or
  success-rate evaluations.
- `real_training_checks.json`: independent CPU reload of the actual GPU
  checkpoints, parameter-update and replay checks.

Long-run PAP learning performance and hyperparameter selection remain separate
experiments. The original elevator results do not establish PAP performance.

## Real robot endpoint and human labels

`pressb real_robot` can replace the simulation endpoint using the same schema/reset/step protocol, with one PiPER environment. The trainer can select environment outcomes or human success/failure labels; an externally ended episode is acknowledged through `/end_episode` before reset. See [real robot guide](real_robot.md) and the `pap_real_walloss_dsrl.json` / `pap_real_walloss_eval.json` experiments. No arm movement has been tested yet because the remote machine has no CAN connection.
