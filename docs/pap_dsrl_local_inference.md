# PAP DSRL with local Wall-OSS inference

Local inference is deployed and both 10,048-transition training comparisons
completed on 2026-11-05. The fastest measured placement uses **simulation and
Wall-OSS on GPU 0, with SAC on GPU 1**: **13.3434 transitions/s**, compared with
**12.5489/s** using remote H200 inference, a **6.33% throughput increase**.
The measured rate projects to **8.33 hours for 400k** or **20.82 hours for 1M**;
these are estimates, not completed long-training durations.

The fixed-input inference-only comparison measured 2.991 seconds per batch on
H200 versus 2.707 seconds on local RTX 5090, a 1.105× speedup. Local transport
saved time, while the local flow decoder took longer. This limits the overall
gain from moving inference onto the workstation.

This extends the [optimized PAP pipeline](pap_dsrl_pipeline_optimization.md)
without changing the task, frozen checkpoint, BF16 precision, or native batch
size. It does not report an independent task-success evaluation.

## Environment and configuration

| Component | H200 deployment | Local RTX 5090 deployment |
|---|---|---|
| Model environment | Remote project `.venv`, Python 3.10 | `.conda/envs/wallx-inference`, Python 3.10.20 |
| PyTorch | `2.6.0+cu124` | `2.7.0+cu128` |
| PyTorch CUDA runtime | 12.4 | 12.8 |
| Transformers | 4.49.0 | 4.49.0 |
| Pillow | 12.3.0 | 12.2.0 |
| CUDA extension target | H200 | `sm_120`, rebuilt with CUDA 12.8 |
| Inference endpoint | SSH-forwarded `ws://127.0.0.1:18130` | `ws://127.0.0.1:18131` |

The local model environment stays separate from Isaac's
`.conda/envs/pressb` and the learner's `zprl` environment. The recorded package
list and extension build details are in
`outputs/pap_dsrl_local_5090_20261105/environment.txt`,
`environment_identity.json`, and `extension_build.log`.

The local CUDA extension was rebuilt from the same corrected source used on
H200. In particular, an older `wallx_csrc` binary must not bypass the
window-index race fix described in the [batch guide](walloss_batch_inference.md).
The local extension passed 200 checks spanning batch sizes 1, 4, 32, and 64,
mixed-size image batches, and default/nondefault CUDA streams. To reproduce the
build on this workstation, stop its inference service first and run:

```bash
cd /home/pengguanqi/Workspace/Research/PressB/ref/wall-x
env -u TORCH_CUDA_ARCH_LIST CUDA_VISIBLE_DEVICES=1 \
  CUDA_HOME=/usr/local/cuda-12.8 PYTHONNOUSERSITE=1 \
  /home/pengguanqi/Workspace/Research/PressB/.conda/envs/wallx-inference/bin/python \
  setup.py build_ext --inplace --build-temp build_5090 --force
```

Here `setup.py` discovers the visible GPU's native architecture. The successful
build generated `compute_120`/`sm_120` code. Rebuild and restart when the CUDA
extension source or its PyTorch environment changes.

Use these public configurations together:

- [Local inference](../configs/rl_nodes/pap_walloss_local_fast_inference.json):
  frozen BF16 Wall-OSS, native batch 64, 32-action horizon, 10 flow steps,
  `flow_prefix_cache=true`, `flow_prefix_cache_mode="full_attention"`,
  256 cached contexts, 600-second TTL, and 192 MiB message limit.
- [Simulation](../configs/rl_nodes/pap_vector_fast_simulation.json):
  64 PAP environments on physical GPU 0, native 640×480 cameras, two-frame
  renderer-history refresh, fused step/reset, 20-second episode limit,
  randomized red-cube positions, and no action smoothing.
- [Local DSRL](../configs/rl_nodes/pap_dsrl_local_fast.json):
  WebSocket inference and HTTP simulation on loopback, negotiated binary PNG,
  eight image workers, noise scale 1.5, replay capacity 100k, requested UTD 1,
  and learning start at 2,000 transitions. Its default budget is 400k;
  the qualification commands below override that budget to 10,048.

The simulator applies the model's existing first PIL bicubic resize to
256×192 before lossless PNG transport. Native render dimensions, FOV and camera
calibration remain unchanged. The model retains its subsequent preprocessing.
The `full_attention` mode preserves attention shapes while reusing the invariant
prefix during flow decoding. It is the mode validated by the preceding pipeline
work; the experimental `suffix` mode is not selected here.

## Start the three local nodes

Run from the PressB root, using separate terminals. Adjust model, normalization,
trajectory and Python paths to the deployment. The existing reconstruction
trajectory referenced by the simulation configuration is required. Reuse an
already-running matching service instead of binding its port twice.

For placement A, simulation uses physical GPU 0; inference and SAC share
physical GPU 1:

```bash
cd /home/pengguanqi/Workspace/Research/PressB

env -u PYTHONPATH -u CUDA_VISIBLE_DEVICES PYTHONNOUSERSITE=1 \
  .conda/envs/pressb/bin/python scripts/serve_simulation_node.py \
  --config configs/rl_nodes/pap_vector_fast_simulation.json
```

```bash
cd /home/pengguanqi/Workspace/Research/PressB
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  .conda/envs/wallx-inference/bin/python scripts/serve_inference_node.py \
  --config configs/rl_nodes/pap_walloss_local_fast_inference.json
```

```bash
cd /home/pengguanqi/Workspace/Research/PressB
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=0,1 PYTHONNOUSERSITE=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/pengguanqi/miniconda3/envs/zprl/bin/python scripts/run_rl_training.py \
  --config configs/rl_nodes/pap_dsrl_local_fast.json \
  --max-transitions 10048 --output outputs/pap_dsrl_local_A_NEW_RUN/train
```

`CUDA_VISIBLE_DEVICES=1` maps the inference process's physical GPU 1 to logical
`cuda:0`. The trainer sees both devices and selects `device="cuda:1"` from its
configuration. The simulation worker selects physical GPU 0 through
`backend.config.gpu_ids=[0]`.

For placement B, restart the inference node with `CUDA_VISIBLE_DEVICES=0`
and start a fresh simulator for the comparison. The inference JSON still uses
logical `cuda:0`. Simulation and
inference then share physical GPU 0, while SAC remains on physical GPU 1.
Use a fresh output directory, for example
`outputs/pap_dsrl_local_B_NEW_RUN/train`, and no `--resume` for this comparison.
Check available memory before running the shared simulation/inference placement.
Placement B was faster in the completed comparison and is the current local
deployment. Both placement trials restarted the simulator before a fresh run.
The 10k runs are throughput qualifications; removing `--max-transitions` uses
the public 400k training budget.

## Paired inference measurement

Both nodes received the same saved batch of 64 independently randomized PAP
observations, including each observation's original instruction, state and two
camera images. The input file is
`outputs/pap_dsrl_optimization_20261104/model/real_64_observations.json`, with
SHA-256
`4742e7738fc91c25187d77c10a9e513eb1405d5f408b9b6645cf8dc5b1777f9e`.

Each node performed one warmup round followed by five timed rounds. Every round
encoded all 64 observations once, decoded the resulting contexts with each of
three fixed initial-noise arrays, and released all 64 contexts. Normal noise
used NumPy seed 20261105; clipped noise used the same normal array clipped to
±1.5; uniform noise used seed 20261106 and bounds ±1.5. Every noise array had
shape `[64, 32, 20]`. Final cache size was zero on both nodes.

The following means exclude warmup. RPC timing includes the client call; service
timing covers the service handler and is not a GPU-only measurement.

| Mean per batch of 64 | Remote H200 | Local RTX 5090 |
|---|---:|---:|
| Encode RPC | 1.812 s | 0.964 s |
| Encode service handler | 0.997 s | 0.929 s |
| Decode RPC, normal noise | 1.179 s | 1.743 s |
| Decode service handler, normal noise | 1.067 s | 1.739 s |
| Encode + decode, normal noise | 2.991 s | 2.707 s |
| Encode + decode, clipped normal | 2.972 s | 2.707 s |
| Encode + decode, uniform | 2.971 s | 2.717 s |
| Inference-only chunks/s, normal noise | 21.40 | 23.64 |

The sum of encode/decode exchange time minus service time was approximately
0.901 seconds remotely and 0.013 seconds locally for normal noise. This residual
includes serialization, queueing and transport; it does not isolate network
time. The local decode handler took about 0.672 seconds longer, offsetting much
of the transfer saving. These were sequential measurements of one fixed batch,
not randomized alternating trials or a simulator/learner benchmark.

The exact three-noise benchmark is saved as
`outputs/pap_dsrl_local_5090_20261105/paired_node_benchmark.py`. It can be rerun
against an idle service, with a fresh output file:

```bash
PYTHONPATH=src /home/pengguanqi/miniconda3/envs/zprl/bin/python \
  outputs/pap_dsrl_local_5090_20261105/paired_node_benchmark.py \
  --endpoint ws://127.0.0.1:18131 --label local5090 \
  --observations outputs/pap_dsrl_optimization_20261104/model/real_64_observations.json \
  --output outputs/pap_dsrl_local_5090_20261105/repeat_local5090.json
```

The artifact script and saved observations are local evidence, not packaged
assets in a clean clone. The tracked
[`benchmark_walloss_node_pair.py`](../scripts/benchmark_walloss_node_pair.py)
also supports fixed-input single-noise checks with `--repeats 5`; its interface
uses singular `--observation` and requires `--seed 20261105` to match this normal
noise array. Run H200 through an existing SSH tunnel or the remote deployment
instructions in the [pipeline guide](pap_dsrl_pipeline_optimization.md).

## Numerical identity and differences

The two service reports matched model ID, checkpoint content hash,
normalization hash, training-config hash, source revision, image-preprocessing
identity, noise/action schemas, full-attention mode and flow-step count.
Features had shape `[64, 2048]`; actions had shape `[64, 32, 7]`. All values were
finite, and repeated features/actions within each deployment were exactly
identical across the six rounds.

Across deployments, feature relative L2 difference was **0.004222**, with mean
row cosine similarity **0.999991** and maximum absolute difference **0.031963**.
The action differences were:

| Fixed initial noise | Arm max | Arm RMS | Gripper max | Gripper RMS |
|---|---:|---:|---:|---:|
| Normal N(0,1) | 2.625° | 0.166° | 14.922 mm | 1.007 mm |
| Same normal clipped to ±1.5 | 2.750° | 0.165° | 7.031 mm | 0.563 mm |
| Uniform ±1.5 | 5.000° | 0.156° | 5.156 mm | 0.429 mm |

These are absolute output-command differences over all 64×32 actions, before
the simulator clips commands to executable bounds. They are not physical
tracking errors. Both nodes used BF16 and batch 64; hardware, PyTorch/CUDA
kernels and Pillow versions differed. The comparison therefore measures these
two deployments, and cannot assign the differences solely to GPU hardware or
batch size. Within-node repeatability does not establish identical cross-node
closed-loop behavior.

## Complete training comparison

One transition means one environment's action chunk, containing up to 32
30 Hz control steps. It does not mean one episode. Compare the common
2,048-to-10,048 post-warmup window, including indexed resets, SAC updates and
checkpoint saving. Count actual elapsed time; SAC updates overlap pending RPCs
and must not be added to RPC time again.

| Placement | GPU 0 | GPU 1 | Model node | Post-warmup transitions/s |
|---|---|---|---|---:|
| Prior remote qualification | Simulation | SAC | Remote H200 GPU 2 | 12.5489 |
| Local A: inference with SAC | Simulation | Inference + SAC | Local GPU 1 | 12.9107 |
| Local B: inference with simulation | Simulation + inference | SAC | Local GPU 0 | 13.3434 |

The remote row comes from the completed
[pipeline qualification](pap_dsrl_pipeline_optimization.md). Local A completed 10,048 transitions and 8,017 SAC updates in 774.814 seconds.
Its post-warmup window completed 8,000 transitions in 619.643 seconds, with
8,000 updates and no dropped update credit. Its throughput was 2.88% above the
remote run. Local B completed 10,048 transitions and 8,017 SAC updates in
755.006 seconds. Its post-warmup window completed 8,000 transitions in 599.547
seconds, with 8,000 updates, no dropped credit, and 602 indexed environment
resets in 122 actual reset batches. It was 6.33% faster than the H200 run and
3.35% faster than local A.

| Post-warmup measurement | Remote H200 | Local A | Local B |
|---|---:|---:|---:|
| Encode RPC, seconds/call | 1.820 | 0.961 | 0.952 |
| Decode RPC, seconds/call | 1.166 | 1.933 | 1.770 |
| Simulation step/reset, seconds/call | 1.949 | 1.906 | 1.918 |
| Overlapping learner seconds in the window | 50.121 | 188.008 | 50.367 |
| Projected 400k training hours | 8.85 | 8.61 | 8.33 |
| Projected 1M training hours | 22.14 | 21.52 | 20.82 |

Sharing the inference GPU with SAC increased both decode and learner time.
Moving inference to the simulation GPU avoids concurrent SAC work on that
device; the trainer issues simulation and inference requests sequentially.
Local B still spends about 1.77 seconds decoding each batch, compared with
1.17 seconds remotely. Both computation and simulation therefore remain
substantial costs after reducing transport overhead.

The **15 transitions/s qualification target remains unmet** in all three runs;
all other throughput accounting checks passed. Both local final checkpoints
passed finite-value, optimizer, replay, terminal-state and actual actor-update
audits. Local A/B completed 751/755 episodes; their terminal outcomes, rewards,
discounts and replay entries matched their logs. Each final checkpoint retained
47 unconsumed update credits without dropping any credit.

Ten-second resource sampling found a GPU 0 peak of **29,852 MiB** for local B,
with a minimum of **2,235 MiB free**. GPU 1 peaked at 1,029 MiB. No OOM occurred;
host available memory remained above 41.57 GiB. These are sampled device totals,
including desktop and driver allocations, rather than continuous allocation
peaks. Local A peaked at 15,549/15,268 MiB on GPU 0/1. The same saved B64 inputs
produced exactly identical model features and actions on the two local GPUs.

These are single deployment runs, without confidence intervals. Runtime
versions and GPU hardware differ between H200 and local; closed-loop
trajectories and reset counts can also differ. The short runs do not fill the
100k-capacity replay, so later checkpoint saves can be larger. Startup and
pre-window warmup are excluded from the steady-state rate.

Neither the inference-only test nor a training run's `recent_sr` is an
independent frozen-policy success-rate evaluation. A controlled closed-loop
evaluation is needed to compare task success after changing deployment.

Use the existing analyzer for each completed run:

```bash
/home/pengguanqi/miniconda3/envs/zprl/bin/python \
  scripts/analyze_rl_node_throughput.py \
  --run outputs/pap_dsrl_local_A_NEW_RUN/train \
  --output outputs/pap_dsrl_local_A_NEW_RUN/throughput.json
```

## Evidence and service lifecycle

Detailed evidence is under `outputs/pap_dsrl_local_5090_20261105/`:

- `paired_h200.json` and `paired_local5090.json`, with corresponding `.npz`
  arrays: all per-round timings, identities, inputs and outputs.
- `paired_comparison.json` and `paired_independent_audit.json`: numerical
  differences, exact input matching and warmup exclusion checks.
- `window_kernel_checks.json`, `environment_identity.json` and
  `extension_build.log`: local CUDA-extension and runtime validation.
- `h200_pair_inference_process.json` and `h200_pair_shutdown.json`: temporary
  H200 service ownership and shutdown evidence. Its cache was empty before
  shutdown; subsequent checks found no listener and zero GPU 2 memory use.
- `throughput.json`, `checkpoint_audit.json`, `episode_audit.json`, and the
  same names under `sim_inference_colocated/`: complete local A/B measurements.
- `three_deployment_comparison.json`: matched configuration checks, all three
  completed windows, resource measurements and duration estimates.
- `sim_inference_colocated/same_inputs_cross_gpu_comparison.json`: exact
  feature/action comparison between physical GPU 1 and GPU 0.

Both qualification trainers have exited and their temporary simulation services
were stopped after checking that their run ownership was released. The local
inference node remains available on **GPU 0**, `ws://127.0.0.1:18131`, with an
empty context cache. Its PID/start-time receipt is
`sim_inference_colocated/inference_process.json`; verify that identity before
managing the process. The temporary H200 inference service is stopped. No new
400k training run was started.

See the [tracked result](results/pap_dsrl_local_5090_20261105.json),
[local A W&B run](https://wandb.ai/penggq2025-southern-university-of-science-technology/pressb-online-rl/runs/748e5ea0bc1d4ae88d38ff4166090f5f),
and [local B W&B run](https://wandb.ai/penggq2025-southern-university-of-science-technology/pressb-online-rl/runs/3f047d6e915742fcb3a98e57c1d545b4).
