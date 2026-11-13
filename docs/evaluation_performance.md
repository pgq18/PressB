# Evaluation performance

The [generic sharded runner](../scripts/run_sharded_evaluation.py) runs independent
simulator/evaluation processes through one frozen inference service. Workers
can simulate while another worker uses inference; there is no barrier between
action chunks. The inference service serializes requests and applies its
configured microbatch size. It does not merge requests into a larger batch.

By default, `--startup-order inference-first` loads and verifies the model before
starting simulator processes. This avoids overlapping model-loading and Isaac
initialization memory peaks. `--startup-order parallel` remains available when
host RAM is sufficient. Steady-state workers still run concurrently.

## Pi0.5 fast profile

Run from the repository root with an existing checkpoint and exported dataset:

```bash
.conda/envs/pressb/bin/python scripts/run_pi05_manipulation_eval.py evaluate \
  --task cube_stack \
  --checkpoint /path/to/pretrained_model \
  --dataset-suffix 100_20261113 \
  --performance-profile fast \
  --simulation-workers 2 --num-envs 8 --batch-size 8 \
  --episodes 100 --sim-gpu 0 --inference-gpu 1 \
  --output outputs/cube_stack_pi05_fast
```

`--num-envs` is **per simulator**: this example has 16 environment slots and one
model instance. The two simulators share GPU0 and use consecutive ports starting
at `--simulation-port`; inference uses GPU1 and `--inference-port`. The Pi0.5
launcher requires distinct simulation and inference GPUs.

The default profile remains `legacy`, with one simulator, eight environments,
batch size four, full terminal observations, and separate step/reset calls.
`fast` defaults to batch size eight, enables four CPU color-response workers and
inference timing, and selects `state_only_if_no_bootstrap` plus fused step/reset.
An explicit `--batch-size` overrides either profile's default. The fast profile
requires at least one allocated episode per environment slot.

With a service advertising `encode_release_contexts`, fast evaluation also
retires each worker's previous observation cache entries atomically inside the
next `/encode` request. This avoids a separate `/release` request waiting behind
another worker's inference. Replacement features/actions are validated before
old entries are retired; final cleanup still releases outstanding entries.
Older inference services automatically retain the separate-release behavior.

Dataset selection determines the scene: `--dataset` accepts an export directory
or its `meta/export_manifest.json`; `--dataset-name` selects a directory under
`--dataset-root`; `--dataset-suffix` selects `pressb_<task>_<suffix>`. The default
suffix remains `500_20261110`. The launcher reads the selected manifest's
`raw_manifest.config`, including its scene and initial pose, rather than
substituting the scene from the experiment template. Manifest paths and hashes
are recorded. Optional `--task-config` supplies explicit deep-merged overrides.

Both profiles retain the selected scene's physics, collision/contact settings,
randomization, camera resolution and render-frame settings, settling, smoothing,
and success criteria. Color-response threading uses the same pixel transform.
The launcher applies the same evaluation settings in both profiles: 30 Hz,
checkpoint chunk size, execution horizon `min(7, chunk_size)`, termination on
success, and time limits of 45 seconds for Stack or 120 seconds for Cleanup
(unless explicitly overridden). The profile does not change checkpoint weights,
processors, model precision, denoising steps, or compilation settings.

Append `--validate-only` (alias `--prepare-config-only`) to write and validate
configs without starting GPU processes. Use a separate output directory for
validation and execution; sharded outputs must be fresh.

## Generic runner and episode accounting

For another task/model experiment with `run.method=base` and `run.mode=eval`:

```bash
.conda/envs/pressb/bin/python scripts/run_sharded_evaluation.py \
  --config configs/experiments/cube_stack_pi05_eval.json \
  --workers 2 --num-envs 8 --batch-size 8 --episodes 100 \
  --sim-gpu 0 --inference-gpu 1 --output outputs/sharded_eval
```

Set the checkpoint in the source experiment first. The generic runner preserves
its rollout/terminal settings; it does not automatically select the Pi0.5 fast
profile. `--workers` also accepts the alias `--simulation-workers`.

The runner divides the requested episode count across workers, requires every
worker to finish its exact allocation, and aggregates **all completed episodes,
including failures**. It records globally unique episode IDs, success rate and
Wilson 95% interval, termination reasons, and node identities. Missing episodes,
duplicate identities, inconsistent success records, simulator faults, or cleanup
failures prevent a completed result. Failed steps/workers are not retried and
interrupted outputs are not resumed.

`--max-transitions-per-worker N` is an explicit throughput diagnostic: its result
is `diagnostic_completed` with `formal_evaluation=false`, and any success rate
covers completed episodes only. Separately, the Pi0.5 wrapper labels its standard
100-episode run `formal` and other episode counts `diagnostic`; those shorter
runs still require their exact requested count unless using the generic
transition-limited probe.

Worker `i` uses `seed + i`. Reset-layout and policy-noise draws share each
worker's RNG stream. Changing worker count or full/fused reset scheduling changes
later draws; equal starting seeds do **not** make runs paired by episode.
Global episode IDs follow completion order within each worker's allocation.

## Timing and measured inference result

Inspect `result.json`, merged `episodes.jsonl`, and per-worker
`run/rpc_timing.jsonl` and `simulation.final_status.json`. The Pi0.5 wrapper stores
these sharded artifacts under its output's `sharded/` directory.

`evaluation_wall_seconds` measures the concurrent evaluation interval after
startup; `startup_wall_seconds` is separate, and `total_wall_seconds` includes
finalization and cleanup. Worker elapsed times and RPC totals overlap across
processes. Per-call client/server timings describe successful calls; status RPC
totals also include failed attempts. Simulation counters nest:

- `contact_callback_seconds` is included in `physics_seconds`.
- `after_tick_seconds` measures outcome checks outside physics, inside the step.
- `color_response_seconds` is included in `render_seconds`.
- `encode_seconds` covers output resize/copy and PNG encoding; step/reset/fused
  totals enclose their component work.

Do not add these counters to reconstruct wall time. Inference service timings
are host wall measurements without extra CUDA synchronization; the standalone
inference benchmark below synchronizes CUDA around each measured request.

On 2026-11-13, one RTX 5090 ran the Stack step-22300 checkpoint on eight distinct
first-frame observations from the new Stack dataset. The same backend, PNGs and
per-observation noise seeds were used for two warmups and five interleaved
measurements per batch size, with contexts released after every request.

| Service microbatch | Median `/encode` time for 8 observations | Peak allocated VRAM |
| --- | ---: | ---: |
| 4 | 0.546016 s | 9.1626 GiB |
| 8 | 0.424925 s | 9.5730 GiB |

Batch eight reduced median inference latency by **22.18%** (1.285× throughput).
All predictions were finite; the checkpoint's bf16 configuration, ten inference
steps, and disabled compilation were retained. Outputs were not bitwise equal:
paired cross-batch maxima were 0.005567 rad for arm joints and 0.00011671 m for
gripper width. Same-batch repetitions also varied, so these differences cannot
be attributed solely to batch size. This measures inference, not closed-loop
success or simulator throughput.

Local evidence is retained under `outputs/eval_optimization_20261113/inference_batch/`:
`results.json`, `benchmark_pi05_batch.py`, and `analysis.json`. These raw artifacts
are not distributed in Git; the [portable summary](results/evaluation_performance_20261113.json)
records the measurements and comparison limits.

The subsequent closed-loop throughput probes executed 512 chunk transitions per
case, with seven actions per chunk. Each task's two configurations used identical
checkpoint, model-configuration and semantic task-configuration hashes:

| Task | 1 simulator × 8 envs, batch 4 | 2 simulators × 8 envs, batch 8 + fast | Throughput gain | Fast peak GPU0 memory |
| --- | ---: | ---: | ---: | ---: |
| Cube Stack | 60.84 s | 36.78 s | 1.65× | 21.36 GiB |
| Tabletop Cleanup | 180.51 s | 90.06 s | 2.00× | 21.81 GiB |

These timings exclude service startup and include initial reset and trainer
execution. Inference and simulation shared GPU0, while dataset collection
continued on GPU1. GPU memory is the sampled device total, including the desktop;
these are one-run throughput probes, with different worker/reset RNG streams,
not paired success-rate evaluations. All final inference caches were empty and
all owned processes exited cleanly. No complete episode finished within these
short probes. Results do not establish a 100-episode runtime or success rate.

Cleanup's single-simulator contact callbacks accounted for 124.88 s of the
180.51 s interval. Its physics time was 128.43 s, including those callbacks;
increasing model batch size alone cannot remove that CPU cost. Independent
simulator processes parallelize it. In Stack, merging cache retirement removed
most separate-release queueing: the first two-worker probe spent 14.81 s in
summed release RPCs, versus 0.124 s in the final probe.

The initial parallel-startup probe encountered severe host RAM pressure (minimum
sampled available memory 0.65 GiB, inference exited with SIGKILL). Inference-first startup passed
the subsequent probes. This changes startup scheduling, not rollout semantics.

Validation: 349 runtime/CLI/launcher tests and 121 adapter/expert tests passed;
two WebSocket expert tests were skipped because the environment lacks the
required `websockets.asyncio.server` API. GPU probes used HTTP transport.
See the [portable result summary](results/evaluation_performance_20261113.json).
Local detailed results remain in `outputs/eval_optimization_20261113/summary.json`.
