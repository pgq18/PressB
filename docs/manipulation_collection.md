# Playground manipulation datasets and SFT

The collection pipeline targets **500 physically accepted episodes per task**,
stored separately. It uses the shared Isaac parallel backend and task planning
hooks; the calibrated playground robot, table and cameras are retained.

New collection defaults use the [November 13 appearance preset](playground.md#appearance-calibration-2026-11-13).
Its fixed camera response is applied inside the simulation capture path, so
recorded RGB and model observations agree. Use a new output directory when
changing appearance; historical datasets retain their original pixels and labels.

| Task | Instruction | Randomized layout |
|---|---|---|
| `cube_stack` | Pick up the red cube and place it on top of the blue cube. | Both complete cube footprints inside base-frame X `[0.28, 0.40]` m, Y `[-0.14, 0.14]` m; yaw `[-45, 45]` degrees; centres at least 8 cm apart. |
| `tabletop_cleanup` | Put all objects on the table into the black box. | Exactly two different objects from gamepad, speaker and lidar; complete footprints inside X `[0.24, 0.46]` m, Y `[-0.22, 0.13]` m; yaw `[-30, 30]` degrees; 3 cm clearance. The box stays fixed. |

Cleanup balances the three object pairs across accepted dataset slots
(167/167/166 episodes), with a shuffled schedule. Failed attempts change the
layout seed without changing the required pair. Planning uses measured poses
after the objects settle. These are proposal ranges: rejected layouts and
unsuccessful executions are excluded, so the accepted distribution can differ
from a uniform distribution. Each dataset's example summary reports its actual
ranges and pair counts.

The default command perturbation is a smooth, bounded **0.12-degree** noise on
arm joints during free-space pregrasp/approach-open/transfer phases. It vanishes at phase
boundaries and scales down near joint limits. Descent, grasp, placement, release
and gripper commands are not perturbed. Recorded action labels contain the
commands actually dispatched. Acceptance requires the complete physical
trajectory, release/withdrawal and stable task outcome; objects are not
teleported along planned paths.

## Compact trajectories (2026-11-12)

New collections default to `--trajectory-profile compact`. Use a **new output
directory**: the previous 500-episode datasets retain their original motion and
60 mm (stack) / up-to-70 mm (cleanup) openings. Changing the scene's maximum
opening does not change old action labels or normalization statistics.

The compact planner uses the calibrated scene's full **100 mm** opening and
keeps the same speed limits (20 degrees/s joints, 40 mm/s Cartesian motion,
30 mm/s full gripper width). It removes fixed waypoint holds and final
return-home motions. Stack combines approach with opening, lifting/turning/
transport/lowering into a continuous arch, and finishes opening during retreat.
Cleanup combines approach with opening and reorients during the lift after the
held object's geometry clears the table. It goes directly to the second object
after clearing the box. Closure and release still take real gripper-motion time.

The last moving target is followed by **0.6 seconds of physical verification**.
Those verification steps are kept in `physics.jsonl` (`phase: validation`,
`recorded: false`), but excluded from training RGB/state/action samples.
Initial settling is likewise outside the recording. The recording itself
remains contiguous at 30 Hz: no internal frames are removed or retimed, and
the required terminal observation repeats the final action. Success still
requires physical grasp/lift, release, support and a stable outcome; the final
joint target is the retreat pose rather than home.

`plan.json.motion_metrics` reports duplicate command rows and planned holds;
`metadata.json.motion_metrics` reports actual motion. Near-static means all six
joint changes are below 0.05 degrees and full gripper-width change below 0.1 mm
per control interval. It includes slow movement and contact-constrained fingers,
so it must not be interpreted as proof of an unnecessary pause.

```bash
.conda/envs/pressb/bin/python scripts/collect_manipulation.py \
  --task cube_stack --output outputs/stack_compact_preview \
  --episodes 4 --num-envs 4 --gpu 0 --trajectory-profile compact
.conda/envs/pressb/bin/python scripts/preview_manipulation_dataset.py \
  --raw outputs/stack_compact_preview --output outputs/stack_compact_preview/videos
```

`--trajectory-profile legacy` retains the earlier planner for comparison. The
low-level planning functions also keep `legacy` as their default for existing
replay callers; dataset collection explicitly selects `compact`. Manifests and
sharded workers pin this choice and reject mixing profiles.

The 2026-11-12 simulator check used 0.12-degree bounded command noise: Stack
passed 8/8 executed layouts; Cleanup passed 3/3 (one per object pair), with one
additional layout rejected by IK before execution. Same-layout legacy CPU
plans averaged 29.98 s and 62.11 s, versus 16.57 s and 46.89 s for compact Stack
and Cleanup respectively. Compact planned commands had no duplicate rows;
measured near-static intervals averaged 0.60% and 2.01%. This is a small planner
validation, not a policy success-rate estimate. Both preview datasets passed
full LeRobot export/image/numeric/reader audits. Detailed evidence is in
[`results/compact_manipulation_20261112.json`](results/compact_manipulation_20261112.json);
original-rate examples are under
`outputs/compact_manipulation_20261112/{stack_noisy_examples,cleanup_examples}/`.

### Centred start poses

Stack and Cleanup now start over the midpoint of their configured X/Y spawn
rectangle, independently of each episode's sampled object locations. The
collection regions give Stack `(0.340, 0.000)` m and Cleanup `(0.350, -0.045)` m
in robot-base XY. The fingertip TCP starts **0.160 m above the table**, with
15-degree outward radial tilt and 100 mm full gripper opening. The tilt keeps
the pose within the existing joint limits; no camera or base-mount change is
needed. The two calibrated camera configurations and empty Playground preset
are retained; the wrist camera moves with the arm as before.

Task configuration accepts:

```json
"initial_pose": {
  "mode": "workspace_center",
  "height_above_table_m": 0.16,
  "radial_tilt_deg": 15.0,
  "tool_yaw_deg": 180.0,
  "gripper_width_m": 0.1
}
```

All four active task node presets select `workspace_center`. New collection
manifests explicitly pin its settings, preventing silent appends to older runs
with different starts. For historical replay, use `initial_pose.mode: scene`
and the saved `scene_overrides.initial_state_joint_rad_m`, including its gripper
coordinate. The `legacy` trajectory profile alone does not select an old home.
Default task starts may differ from collection starts if their configured
sampling rectangles differ. The scene's three initial-state representations
are kept identical for USD authoring, reset and planning.

The centred-start check passed 8/8 Stack executions and all three Cleanup
object pairs (3/3 executions, plus one IK-rejected layout). Relative to the
same layouts with the prior home, average trajectories changed from 16.57 to
14.30 seconds and 46.89 to 44.42 seconds respectively. All camera fields and
both source preset file hashes were unchanged, and 359 CPU tests passed.
See [`results/centered_manipulation_20261112.json`](results/centered_manipulation_20261112.json)
and `outputs/centered_manipulation_20261112/{stack_examples,cleanup_examples}/`
for measured initial poses, full physical results and original-rate videos.

## Run and inspect

From the repository root:

```bash
.conda/envs/vlajepa-inference/bin/python scripts/run_manipulation_dataset_pipeline.py \
  --num-envs 16 --cube-gpu 0 --cleanup-gpu 1
```

The pipeline collects, exports/audits, creates examples, then deploys both SFT
jobs. GPU and environment counts are configurable; tasks sharing one GPU are
scheduled accordingly. Add `--prepare-only` to stop after datasets/videos.
The orchestrator and its stage receipts support reattachment to running work;
failed stages require inspection before the explicit `--retry-failed` option.

For a fresh 100-episode dataset per task using the optimised settings, run:

```bash
.conda/envs/pressb/bin/python scripts/run_manipulation_dataset_pipeline.py \
  --output outputs/task_datasets_100_20261113 \
  --episodes 100 --dataset-suffix 20261113 --seed 20261113 \
  --cube-gpu 0 --cleanup-gpu 1 --workers-per-gpu 2 --num-envs 8 \
  --collection-config configs/collection/manipulation_fast.json --prepare-only
```

`--num-envs` is per worker: this command runs 32 environments across four
processes. Each task owns a separate canonical dataset, and its two workers
own 50 accepted episode IDs each. The pipeline initializes manifests without
starting Isaac, collects through the sharder, exports and audits LeRobot v3,
then creates original-rate examples. Cleanup preserves its global 34/33/33
object-pair schedule through retries. New exports are named
`pressb_{task}_100_20261113` under `--dataset-root` (default
`/home/pengguanqi/Datasets`). Custom counts or suffixes currently require
`--prepare-only`, because the existing SFT launcher targets the historical
500-episode datasets.

Progress lives under `outputs/task_datasets_20261110/`:

- `pipeline_status.json`: overall stage and errors.
- `{cube_stack,tabletop_cleanup}_raw/progress.json`: accepted count and attempts.
- The raw `manifest.json`, episode `metadata.json` and `physics.jsonl` retain
  configurations, seeds, settled poses, perturbations and physical acceptance.
- `{cube_stack,tabletop_cleanup}_examples/examples_1x.mp4`: selected successful
  examples, global view above wrist view, at the original 30 Hz. Shorter examples
  hold their final frame. `summary.json` records episode selection and coverage.

Individual tools are `scripts/collect_manipulation.py`,
`scripts/export_manipulation_dataset.py` and
`scripts/preview_manipulation_dataset.py`; use `--help` for their arguments.

### Continue one task on multiple GPUs

After stopping its original collector, an unfinished canonical collection can
be split across independent Isaac processes:

```bash
.conda/envs/vlajepa-inference/bin/python scripts/run_sharded_manipulation_collection.py \
  --canonical outputs/task_datasets_20261110/tabletop_cleanup_raw \
  --gpus 0 1 --num-envs 16
```

The runner retains committed episodes and assigns the missing global IDs and
disjoint attempt/seed sequences to isolated workers. It preserves the global
object-pair schedule and atomically publishes accepted episodes back into the
canonical dataset. `collection_shards.json` records ownership and provenance;
`shards/worker_*.log` and canonical `progress.json` report live status. Start the
main pipeline after the runner has acquired the canonical collection lock; it
will wait for all 500 episodes, then export, make videos and launch SFT normally.
On 2026-11-10 this resumed cleanup from 300 accepted episodes with two workers,
100 missing episodes each, using 16 environments per GPU.

### Measured collection settings (2026-11-13)

On the local RTX 5090, **8 environments per process** is the preferred setting.
With the latest appearance, centred start, compact trajectories and 100 mm
gripper, increasing one process from 8 to 16 environments brought little speed
gain while increasing memory use. Both 640×480 camera videos, state/action
records and physical audits were enabled throughout these measurements.

| Task | Environments | Accepted / attempts | Accepted frames/s | GPU mean | VRAM peak | Process-tree RSS peak |
|---|---:|---:|---:|---:|---:|---:|
| Stack | 8 | 30/32 | 36.08 | 36.7% | 5.06 GiB | 7.72 GiB |
| Stack | 16 | 30/32 | 37.19 | 40.3% | 6.98 GiB | 10.20 GiB |
| Cleanup | 8 | 12/16 | 14.09 | 17.3% | 4.97 GiB | 7.59 GiB |
| Cleanup | 16 | 12/16 | 14.33 | 18.7% | 6.71 GiB | 9.70 GiB |

Here one frame is a state/action row with **both** RGB images, not an episode
or physics tick. Throughput includes simulator startup and recording, excludes
teardown and LeRobot export, and counts only physically accepted episodes.
GPU means exclude startup. Resources were sampled every two seconds, so short
peaks can be missed; summed RSS can include shared pages more than once.
These are short throughput probes, not policy success-rate evaluations.

The main costs were CPU contact processing and camera colour processing.
The runtime now caches immutable contact-path classifications and removes
redundant per-contact NumPy dispatch while preserving the force calculation.
The optional [`manipulation_fast.json`](../configs/collection/manipulation_fast.json)
preset adds four CPU workers for independent colour-response operations.
Isaac rendering and camera readback remain on their owning thread.
On the same first 16 layouts, these changes improved startup-excluded
throughput from **37.76 to 50.68 frames/s** for Stack and **14.21 to 19.48**
for Cleanup. Actions, states, object poses, contacts and physical outcomes
matched exactly in the paired single-process traces. Unit tests also check
pixel-exact colour processing for identical inputs; MP4 files are not claimed
to be byte-identical.

With the optimised implementation, **two independent processes, each with
8 environments on the same GPU**, performed better than one process:

| Task | Accepted / attempts | Entire-program wall time | Accepted frames/s | GPU mean | VRAM peak | RSS peak |
|---|---:|---:|---:|---:|---:|---:|
| Stack | 14/16 | 88.77 s | 70.09 | 74.1% | 10.03 GiB | 14.99 GiB |
| Cleanup | 12/16 | 441.79 s | 35.71 | 47.8% | 9.99 GiB | 14.92 GiB |

This second table includes process startup **and teardown**; GPU means cover
the entire run. Matching single-process runs took 136.01 s and 824.05 s,
respectively: two workers improved throughput by 53.2% and 86.5%. Nominal
layouts and acceptance matched; independent physical settling added one
recorded frame in Cleanup, so cross-process traces are not bitwise identical.
Two GPUs with two workers each have not been benchmarked. Further candidates
are batched contact statistics and refilling completed environment slots
without waiting for the longest episode in a batch.

For a **new** dataset, initialise it with one batch using the fast preset,
then distribute its remaining episodes. This example uses GPU 1:

```bash
.conda/envs/pressb/bin/python scripts/collect_manipulation.py \
  --task cube_stack --output outputs/stack_fast_raw --episodes 500 \
  --num-envs 8 --gpu 1 --max-batches 1 \
  --config configs/collection/manipulation_fast.json
.conda/envs/pressb/bin/python scripts/run_sharded_manipulation_collection.py \
  --canonical outputs/stack_fast_raw --gpus 1 --workers-per-gpu 2 --num-envs 8
```

Use `--task tabletop_cleanup` and a separate output directory for Cleanup.
The sharder needs the canonical manifest from the first command and inherits
its configuration. `--workers-per-gpu` defaults to 1; episode ownership,
attempt seeds, files and resumable receipts remain separate for each worker.
Do not edit an existing dataset manifest to change its scene or capture setup.

Raw timings, commands, source snapshots, resource samples, trace comparisons
and the final report are under
`outputs/collection_benchmark_20261113/`, particularly
`report_final/report.md`. The original Cleanup-8 probe was bounded after
two complete batches (16 attempts); its explicit `benchmark_scope.json`
excludes the interrupted, unexecuted next batch. The original exit receipt
is preserved. The runtime/collector checks passed 537 tests and the sharder
checks passed 42 tests.

## Published datasets

Final destinations, published only after export audit:

```text
/home/pengguanqi/Datasets/pressb_cube_stack_500_20261110
/home/pengguanqi/Datasets/pressb_tabletop_cleanup_500_20261110
```

Exports use official **LeRobot 0.6.1 / v3.0**, two RGB cameras (`up`, `wrist`) at
640×480 and 30 Hz, seven-coordinate state/action (six joints in degrees plus full
gripper width in centimetres). Observation at time *t* is paired with the target
actually executed over the following 1/30-second interval; the terminal action
repeats the last applied target. `meta/audit.json` records complete numeric-row
comparison, full camera-video decoding and official-reader sample checks.
An optional v2.1 export is a separate compatibility copy, not a relabelled v3
dataset. The new SFT reader consumes v3 directly.

## H200 supervised training

After both audited datasets and videos are ready, the pipeline invokes:

```bash
.conda/envs/vlajepa-inference/bin/python scripts/launch_manipulation_sft.py \
  --launch --config-dir outputs/task_datasets_20261110/training_prepare
```

A task whose dataset is ready can start independently, without waiting for the
other export. For example, add `--tasks cube_stack` and optionally use
`--status outputs/task_datasets_20261110/training_prepare/cube_deployment_status.json`.
The default selects both tasks. Selection controls dataset auditing, transfer
and readiness checks; both configuration files remain in the same source
snapshot, and both tasks' receipts are retained. Space is reserved for selected
new jobs and for any already running sibling; an unselected task that has never
started does not reserve checkpoint capacity.
`--dataset-root` selects local input; `--remote-dataset-root` independently
defaults to `/home/pengguanqi/Datasets` on H200. Resolving a local dataset symlink
does not change the remote transfer destination or the training configuration.

Each dataset trains an independent VLA-JEPA model with `joint7_generic`: learned
six-joint and gripper outputs, 450/50 episode train/validation split, three
epochs, global batch 64 and bf16. The launcher defaults to one H200 per model;
two H200s with batch 32 per rank preserve global batch 64. Steps are derived from the final
training-frame count. Shared random image augmentation is train-only;
validation stays clean. Initialization uses the official base checkpoint with
four joint7 interface tensors reset, without resuming another task. Only best
and final model weights are retained. Offline validation metrics measure action
prediction; they do not constitute simulator success rates.

The launcher transfers the datasets and a small bundled PressB source snapshot,
verifies the base checkpoint digest and sample decoding, and probes current GPU
availability, filesystem space and user quota. It reserves 65 GiB for two fresh
jobs, or 33 GiB for one selected fresh job, plus the remaining dataset bytes and
any active sibling's unwritten checkpoint budget. Existing data
and jobs are not removed. Exit `75` means retry in 120 seconds; exit `2` requires
inspection. Persistent receipts prevent duplicate jobs or implicit resumes.
The launch gate reports `jobs_started` only after **each selected task has completed at
least one optimizer update**, with finite action loss, world-model loss and
gradient norm in `train.jsonl` and a matching advanced `status.json`.
`processes_started` and `first_steps_verified` are separate status fields;
initializing processes return exit `75` without being relaunched.
Python compatibility checks report individual import failures and stderr. The
joint7 reader requires PyAV; Decord is not a prerequisite for this profile.

Local deployment status is
`outputs/task_datasets_20261110/training_prepare/deployment_status.json`.
The remote root is
`/home/pengguanqi/Worksapce/Research/PressB-task-sft-20261110`; its
`outputs/launch_receipts/` contains task PIDs, logs and exit status, while
`outputs/model_training/` holds configurations, splits, metrics and weights.

### Sequential two-GPU training

Run the collection pipeline with `--prepare-only` when the SFT queue owns
training. It continues collection, export, full dataset checks and example
videos without starting the older single-GPU deployment workflow.

```bash
.conda/envs/vlajepa-inference/bin/python scripts/run_manipulation_sft_queue.py \
  --gpus-per-task 2 --gpu-indices 1 2 --per-device-batch-size 32 \
  --run-suffix ddp2 --checkpoint-reserve-gib 22 --storage-margin-gib 1 \
  --task-overrides outputs/task_datasets_20261110/training_prepare/ddp2_preserved_task_overrides.json \
  --background-archives --prestage-next \
  --output outputs/task_datasets_20261110/ddp2_sft_queue_immediate
```

The queue trains `cube_stack` and verifies full training completion before
starting `tabletop_cleanup`. With `--prestage-next`, the audited cleanup dataset
and source snapshot are transferred and checked while cube training is still
running; staging does not require idle GPUs and never starts training. With
`--background-archives`, complete-run backups run independently and do not delay
the next model. Without these options, staging happens at launch and verified
archival remains sequential. Both tasks use fresh official
pretrained weights. Existing single-GPU runs are not resumed: the model-only
checkpoint policy has no optimizer or RNG state, and native exact resume also
requires the original world size. An explicit GPU list is still checked for
availability; it does not permit sharing a busy device.

The current queue preserves both tasks' remote weights and datasets. Historical
runs are archived separately to reclaim server space. Working-copy rotation can
instead be enabled explicitly with `--archive-completed-remote-payloads` and
`--release-remote-dataset` in sequential archive mode. Background archival cannot
be combined with remote removal, so a backup worker cannot remove files needed
by a subsequent training job.
Every archived regular file is SHA256 checked against the remote run, and
internal symlinks are verified. Only matching checkpoint `model.pt` files are
removed remotely; configuration, logs and archive receipts remain. Dataset
release similarly requires a complete matching local dataset. The final task's
remote weights and dataset are retained by default, even when rotation is
enabled. `--release-final-remote` explicitly extends rotation to the final task.
Local datasets and model archives are always retained.

`--task-overrides` accepts per-task `remote_root`, `remote_dataset_root`,
`checkpoint_reserve_gib` and `storage_margin_gib`. These separate input storage
from training outputs when a server has different home/data quotas. The
launcher also supports `--remote-run-root` for standalone deployments. Reduced
budgets must be justified by the actual checkpoint writer: this model-only
policy holds at most two model payloads, with final best/last sharing an inode
when identical, and only rank zero writes weights. Quotas and filesystem space
are rechecked before launch; no unrelated artifacts are removed.

Queue progress is recorded in `outputs/task_datasets_20261110/ddp2_sft_queue_immediate/status.json`.
The queue distinguishes data preparation, model initialization, training,
verified archival and completion. Consult these live records for progress;
configured or initialized jobs are not evidence of finished training.

### Archiving historical runs

`scripts/archive_remote_runs.py` creates an inventory of an explicitly selected
remote run directory. Review that plan before using `--apply`, which copies and
fully SHA256-verifies each entry before removing its remote copy. Current stack
and cleanup task paths are protected; symbolic links are preserved without
following their targets.

Use `--workers 3` to archive three independent run directories concurrently
(default: one, maximum: eight). Each worker has a separate transfer and proof;
the supervisor serializes shared status updates. Restarting with the same
archive paths adopts existing transfers and skips completed entries. Changing
the worker count does not invalidate the archive plan. Tune concurrency to the
available network bandwidth; it does not change the verification requirements.
