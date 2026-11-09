# Runtime resource monitoring

[`scripts/monitor_runtime_resources.py`](../scripts/monitor_runtime_resources.py)
is a standalone Linux sampler using Python, `psutil`, and `nvidia-smi`.
It reads counters only: it does not import CUDA/Isaac, call services, or signal
workloads. Copy the single script to each remote host and record its SHA256
before launch. Existing output files are never overwritten.

```bash
python3 scripts/monitor_runtime_resources.py collect \
  --output outputs/benchmark/resources.jsonl --interval 2 --duration 3600 \
  --status outputs/benchmark/train/status.json \
  --phase-file outputs/benchmark/train_benchmark/phase.json \
  --pid simulation=123 --pid inference=456 --pid trainer=789
```

Alternatively use `--pid-file trainer=trainer_process.json`. Receipts contain
an integer `pid`, optionally `start_ticks` or `create_time` for identity checking.
Files may appear after monitoring starts. Root PIDs bind to their first observed
process identity; a replacement process requires a new monitor. Descendants and
GPU compute PIDs are also sampled, with separate role labels. Stop the monitor
with SIGINT/SIGTERM; monitored processes are unaffected.

Each JSONL row (`schema_version=1`) includes:

- Hostname, UTC epoch, host monotonic time, sample duration and requested interval.
- GPU UUID/physical index, used/total MiB, GPU utilization, memory-controller
  utilization, power, temperature, and all compute allocations by PID.
- Host CPU, available RAM, swap and cumulative swap/network/disk counters.
- PID identity, roles, RSS, CPU time/utilization and cumulative IO counters.
- Selected trainer status fields, file mtime/age, and the complete phase marker.

Missing/unsupported readings are null or explicit errors. The first CPU reading
is null. Process CPU utilization of 100% represents one CPU core.

The driver should atomically write phase markers, e.g. `{"phase":"checkpoint"}`.
Trainer `state="updating"` alone may include checkpoint writes. Phase windows
split on marker-file replacement, including same-name re-entry between samples.
Do not use an inter-function `steady` marker as the full training throughput
window: use the benchmark's complete round boundary timestamps instead.

```bash
python3 scripts/monitor_runtime_resources.py summarize resources_h200.jsonl \
  --start-unix 1794145000 --end-unix 1794145600 \
  --gpu 1 --gpu 2 --role inference --role trainer \
  --output resources_h200_summary.json
```

Replace the illustrative timestamps with recorded post-warmup round boundaries.
Use GPU UUIDs instead of physical indices if needed. Role filters take a union,
so a process with two selected roles is counted once. On a separate simulation
host, summarize its file separately with its GPU and `--role simulation`.
Do not include other users' GPUs or `gpu_compute` as a task-owned role.

`memory_capacity_used_percent` means **used VRAM / total VRAM × 100**.
`memory_utilization_percent` means **memory-controller busy percentage** and is
not a capacity measurement. Device memory still includes other processes on the
selected GPU. Host totals remain whole-host background measurements; they cannot
be attributed to one job. RSS sums may double-count shared pages.

Resource extrema are sampled, so brief peaks can be missed. Status counters can
lag, and host UTC clocks must be synchronized for cross-host window alignment.
The summary records both observed and trainer-counter elapsed time; derive exact
end-to-end throughput from benchmark round events, not sparse resource samples.

## DSRL component timing

[`scripts/benchmark_dsrl_training.py`](../scripts/benchmark_dsrl_training.py)
runs the actual `GenericTrainer` and records RPC worker intervals, caller wait
envelopes, SAC updates, and completed rollout batches. It selects the source
checkout before importing PressB, and can verify the trainer and learner hashes:

```bash
python scripts/benchmark_dsrl_training.py \
  --source-root /path/to/frozen/src \
  --config outputs/benchmark/trainer.json \
  --phase-log outputs/benchmark/phases.jsonl \
  --phase-file outputs/benchmark/phase.json \
  --skip-checkpoints
```

Use the model's Python environment. The configuration must be a flat
`TrainerConfig` with `method=dsrl` and `mode=train`. Timing files must be outside
its `output_dir`. `--check-only` validates configuration and source imports
without contacting services. Checkpoints are enabled by default;
`--skip-checkpoints` explicitly suppresses interval **and final** saves, so this
mode does not produce a reusable trained policy.

DSRL can update SAC while an RPC is outstanding. Adding RPC time and update time
would therefore count some wall time twice. For offline accounting, use the
actual `start_monotonic_seconds` and `end_monotonic_seconds` in `phase_end` events
of `kind=rpc` and `kind=update`. Their interval unions and intersections partition
a window into RPC only, update only, both, and neither. This measures host-side
concurrency, not independently timed CUDA kernels. RPCs include serialization,
transport, and server work; their nested caller envelopes are separate evidence.

`phase_start` is a notification, and can precede the actual call. Worker and
update events are queued to reduce disturbance: JSONL event order need not be
timestamp order. Completed `batch_end` timestamps provide the throughput-window
boundaries. Phase markers are throttled to 100 ms; full interval events remain
in JSONL. Cumulative `recording_seconds` exposes logging overhead, which must not
be blindly subtracted from elapsed time because it may overlap RPC work.

Record actual optimizer steps and remaining/dropped update credit alongside
speed. Configured update-to-data ratio does not guarantee that all requested
updates were completed by the interaction-budget boundary. Initial and final
actor hashes and optimizer step receipts distinguish real training from a
rollout-only measurement.

For a completed 64-environment run, the offline analyzer validates the logs and
produces both full-loop and matched-window statistics:

```bash
python3 scripts/summarize_dsrl_benchmark.py \
  --training-dir outputs/benchmark/train \
  --phases outputs/benchmark/phases.jsonl \
  --start-transition 2304 --expected-transitions 10752 \
  --output outputs/benchmark/dsrl_summary.json
```

Choose boundaries present in both experiments. This example requires the exact
10,752-transition budget and reports the 2,304→10,752 interval separately. It
rejects incomplete runs, mixed run identities, missing optimizer receipts,
inconsistent update credit, and existing output files. Copied remote logs keep
their original output-directory identity; local artifact relocation is recorded.
