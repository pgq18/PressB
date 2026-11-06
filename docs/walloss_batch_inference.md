# Native Wall-OSS batch inference on H200

On 2026-11-04, the PAP Wall-OSS checkpoint completed real model batches from 1
through 64 on an NVIDIA H200 NVL. At batch 64, preprocessing plus encoding and
10-step flow decoding took **3.972 seconds**, or **16.11 observations/s**, with
**11.64 GiB peak PyTorch allocation**. Module hooks confirmed that all 64 rows
participated in each action-transformer forward.

The BF16 batched outputs are **not numerically identical to the original BF16
singleton policy**. At batch 64 the maximum arm-command difference was
**0.135263 rad (7.75 degrees)** and the maximum full-gripper-width difference was
**8.125 mm**, using the same input observations and initial noise. These results
establish model batching and measured throughput; they do not establish unchanged
PAP success rates or end-to-end RL throughput.

## Implementation and batch evidence

`WallXPolicy.encode_batch` preprocesses a list of observations together, calls the
visual encoder once, and returns an independently owned context for each row.
`decode_batch` reassembles those contexts into one model batch and performs one
transformer forward per Euler step. `infer_batch` combines the two operations.
The PressB adapter calls these APIs once per configured service microbatch.

Different prompt lengths use left padding and attention masks. Each context
stores its unpadded embeddings, RoPE indices and observation features; it does
not store transformer KV states. Per-observation seeds generate independent
baseline noise, while externally supplied noise retains all `32 × 20` latent
coordinates. Each output is a `32 × 7` sequence of absolute arm joints in radians
and full gripper width in metres.

The visual attention implementation groups observations by their total visual
token width and preserves each observation's original image/window mask and
softmax width. This avoids a dense attention matrix spanning every observation.
Batch size one retains the original visual path. The action transformer uses
eager attention to support padded rows. Existing singleton inference, RTC and
the legacy WebSocket observation/RTC formats remain available; the new batch
methods do not implement batched RTC.

The batch-64 forward hooks recorded:

| Operation | Observed module calls |
|---|---|
| Encode | One visual forward containing **128 images** |
| Decode | **10 action-transformer forwards**, each with batch dimension **64** |
| Decoder input | `[64, 232, 2048]` after padding in this case |
| Visual work during cached decode | **Zero** visual forwards |

The same hook checks passed for all seven tested batch sizes. Cache fingerprints
were unchanged after reordered, subset, duplicate-context and alternate-noise
decodes. This establishes context immutability; it does not imply bitwise action
equivalence when regrouping changes a BF16 computation's batch shape.

## Measurement protocol

The benchmark used **three saved PAP camera observations**, each containing the
third-person and wrist views, cycled to form 64 input rows. Four instruction
variants exercise different prompt lengths; their valid token lengths were
218, 222 and 232. The 64 rows used independently generated, fixed external noise
arrays, and every comparison used the matching observation and noise row.
These are repeated saved inputs, not 64 independently sampled layouts or
64 completed task episodes.

Each batch size received one warmup and three measured repetitions. Timings use
CUDA synchronization and include CPU image/token preprocessing, model execution,
context assembly and output copies. They **exclude network transport, Isaac
simulation, resets and RL updates**. Weights remained frozen. The input image
resolution setting was 256, the horizon was 32, and flow decoding used 10 Euler
steps. The supplied PNGs pass through the checkpoint's existing image processor.

The numerical reference runs the original singleton Python `encode`/`decode`
path once for each of the 64 observation/noise pairs. No numerical acceptance
thresholds were supplied. JSON fields such as `all_supplied_thresholds_passed`
therefore do not certify that the action differences are acceptable for PAP.

## BF16 throughput and memory

Source: [complete BF16 benchmark JSON](../outputs/walloss_batch_h200_20261104/batch_1_to_64.json).
Times are mean seconds per batch, not seconds per individual observation.

| Batch | Prepare + encode (s) | Decode (s) | Total (s) | Observations/s | Peak allocated (GiB) | Peak reserved (GiB) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.055 | 0.339 | 0.394 | 2.54 | 8.06 | 8.52 |
| 2 | 0.065 | 0.336 | 0.401 | 4.99 | 8.13 | 8.59 |
| 4 | 0.117 | 0.352 | 0.469 | 8.53 | 8.24 | 8.77 |
| 8 | 0.221 | 0.388 | 0.609 | 13.13 | 8.47 | 9.12 |
| 16 | 0.415 | 0.666 | 1.082 | 14.79 | 8.92 | 9.73 |
| 32 | 0.862 | 1.223 | 2.085 | 15.35 | 9.83 | 10.95 |
| 64 | 1.657 | 2.315 | 3.972 | 16.11 | 11.64 | 13.42 |

Memory values are PyTorch allocator statistics including model weights, not
total GPU usage reported by `nvidia-smi`. Reserved memory includes allocator
cache. The results are specific to this H200, checkpoint and input workload;
changing image size, token count or context-cache occupancy changes the memory
requirements. Observations/s counts one predicted action chunk per observation,
not episodes/s or completed simulation transitions/s.

## Numerical differences and FP32 comparison

BF16 differences against the matching original singleton execution were:

| Batch | Maximum arm error (rad) | Maximum arm error (degrees) | Maximum gripper-width error (mm) |
|---:|---:|---:|---:|
| 1 | 0 | 0 | 0 |
| 2 | 0.008727 | 0.50 | 4.375 |
| 4 | 0.021817 | 1.25 | 8.516 |
| 8 | 0.082903 | 4.75 | 4.375 |
| 16 | 0.119991 | 6.875 | 9.922 |
| 32 | 0.135263 | 7.75 | 8.125 |
| 64 | 0.135263 | 7.75 | 8.125 |

For batch 64, arm RMS error was 0.004083 rad and gripper-width RMS error was
0.620 mm. These are errors over action coordinates and timesteps, not Cartesian
end-effector displacement. The maxima are large enough that closed-loop
evaluation is necessary before claiming that batching preserves task success.

The [visual diagnosis](../outputs/walloss_batch_h200_20261104/vision_diagnosis_v2.json)
verified matching image pixels, grid geometry, fused RoPE values, window indices
and cumulative window offsets for the checked batch sizes. BF16 differences
also appeared in a dense batched visual path, so replacing the observation-grouped
attention with a dense cross-observation matrix did not restore singleton
equivalence. The FP32 visual comparison reduced the discrepancy substantially.

The [FP32 model benchmark](../outputs/walloss_batch_h200_20261104/batch_fp32.json)
compares FP32 batches against **their own FP32 singleton reference**:

| Batch | Total (s) | Observations/s | Peak allocated / reserved (GiB) | Maximum arm error (rad) | Maximum gripper-width error (m) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.519 | 1.93 | 15.82 / 19.55 | 0 | 0 |
| 8 | 2.935 | 2.73 | 16.27 / 19.92 | 4.9472e-6 | 1.7937e-6 |
| 64 | 20.607 | 3.11 | 19.78 / 24.59 | 3.7372e-5 | 2.2501e-6 |

At FP32 batch 64, the maximum error corresponds to about 0.00214 degrees and
0.00225 mm. This is strong numerical agreement within the FP32 comparison, with
a substantial throughput cost. **It does not show that FP32 reproduces the old
BF16 policy**: the reference dtype changed too, and no closed-loop equivalence
claim follows from these numbers. A direct inspection of this PAP checkpoint's
`model.safetensors` header found **1,049 tensors, all stored as BF16**
(4,224,041,152 elements); its config also declares `torch_dtype=bfloat16`.
The current loader converts weights to BF16 on CPU, then the FP32 adapter casts
those values to FP32. For this checkpoint, that intermediate BF16 conversion
does not discard additional precision from the stored weights. FP32 execution
reduces subsequent arithmetic rounding; it cannot reconstruct precision absent
from the saved BF16 values. A future FP32 checkpoint would require changing the
loader to avoid its unconditional intermediate BF16 conversion.

An additional [BF16 full-reduction run](../outputs/walloss_batch_h200_20261104/batch_bf16_full_reduction.json)
still showed a batch-64 maximum arm error of 0.104720 rad (6 degrees) and gripper
error of 5.234 mm, at 16.00 observations/s. It did not eliminate the numerical
differences. The deployment configuration remains **BF16, batch 64**; this is a
tested performance configuration, not an acceptance decision on PAP success.

## Cross-machine WebSocket check

The [64-observation network check](../outputs/walloss_batch_h200_20261104/network_64.json)
passed from the workstation through an SSH tunnel to the H200 service. Health
reported `batching.native_model_batching=true` and `microbatch_size=64`. The
following are individual end-to-end RPC measurements, including transport and
server validation, rather than the direct-model timings in the earlier table:

| Request | Time (s) | MessagePack request size (MiB) | Result |
|---|---:|---:|---|
| `encode`, 64 observations | 6.235 | 28.643 | 64 independent context IDs and features |
| `decode`, 64 cached contexts | 2.445 | 0.361 | Actions shaped `[64, 32, 7]` |
| Repeated `decode` | 2.402 | 0.361 | Exactly equal to the previous decode |
| One-shot `infer`, 64 observations | 8.203 | 29.001 | Exactly equal to cached encode/decode |

Both action comparisons had maximum absolute error **0** with the same
64-row order, service microbatch size and external noise. Release removed all
64 contexts, and health then reported cache size zero. These checks do not
contradict the cross-batch BF16 differences: the network comparison kept the
batch shape and grouping unchanged.

The encode timing includes cross-machine image upload, service-side validation,
preprocessing and model encoding. Its inputs were the same three saved camera
observations repeated to fill 64 rows. The measured message sizes therefore do
not bound the payload size for 64 independently rendered simulator observations;
PNG size depends on image content. The earlier PAP reset response of 63.12 MiB
was measured on a different image set and JSON representation. A complete
64-environment PAP/DSRL rollout, including resets, inference, replay and learner
updates, still requires its own integration validation.

## Deployment identity and CUDA extension

| Component | Recorded value |
|---|---|
| Remote project | `/home/pengguanqi/Worksapce/Research/PressB-walloss` |
| Python environment | Project `.venv/bin/python`, Python 3.10 |
| GPU | NVIDIA H200 NVL, logical `cuda:0` |
| PyTorch / runtime CUDA | `2.6.0+cu124` / `12.4` |
| Extension build toolkit | Project `.cuda-12.8` |
| Transformers | `4.49.0` |
| CPU threads | 8 |
| Configured model directory | `/home/pengguanqi/walloss_models/piper_pap_20260606` |
| Resolved model storage | `/data/scratch/pengguanqi/Models/piper_pap_20260606` |
| Inference config | [`pap_walloss_h200_inference.json`](../configs/rl_nodes/pap_walloss_h200_inference.json) |

The local and remote `model.safetensors` file SHA-256 matched:
`ce0b8c9963195f8725610c29bf022223ff2ab5c85d839add4a72284c9ee33f19`.
The adapter's content identity includes file names as well as bytes and therefore
has a different weights hash:
`c5509660ace0ea5d100290781e7ae165884bdf34308fccb7692b041feeeb887d`.
The benchmark normalization hash was
`0382bb75f230ab115a795343b13d3580bf96401f5d5d17b237cd390084b14f7a`.
The [environment package list](../outputs/walloss_batch_h200_20261104/environment.txt)
and benchmark JSON retain the remaining configuration and source identities.

Deployment must include `flow_context.py`, `vision_batch.py`, both Qwen vision
and action models, the policy and preprocessing utilities, and
`csrc/window_index.cu`. The window-index kernel fixes a cross-block race when
there are more than 256 windows. Updating source alone does not replace an old
extension binary. Rebuild from the remote Wall-X directory:

```bash
cd /home/pengguanqi/Worksapce/Research/PressB-walloss/ref/wall-x
CUDA_HOME=/home/pengguanqi/Worksapce/Research/PressB-walloss/.cuda-12.8 \
  /home/pengguanqi/Worksapce/Research/PressB-walloss/.venv/bin/python \
  setup.py build_ext --inplace --build-temp build_h200 --force
```

Restart inference after rebuilding. The rebuilt kernel passed repeated default-
and nondefault-stream checks, including PAP batch 64 with 768 windows and a
mixed-size 128-image case; see
[kernel validation](../outputs/walloss_batch_h200_20261104/window_kernel_checks.json).
These CUDA checks are distinct from the BF16 rounding differences above.

## Reproduce the model benchmark

Run from the remote project root after deployment and extension rebuilding:

```bash
cd /home/pengguanqi/Worksapce/Research/PressB-walloss
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 .venv/bin/python \
  scripts/benchmark_walloss_batch.py \
  --config configs/rl_nodes/pap_walloss_h200_inference.json \
  --observation outputs/walloss_batch_h200_20261104/real_observations.json \
  --output outputs/walloss_batch_repeat/bf16.json \
  --batch-sizes 1,2,4,8,16,32,64 --warmups 1 --repeats 3
```

The script defaults to mixed-length prompts and seed `20261104`. The recorded
input JSON hash is
`a29f7f5676c6222f220cf298b58d4edc6ad72acb0d217102951ca68d5ca50784`;
the 64-row external-noise array hash is
`429b41fe49fa30e079cfac8874d02db76a3b18579b2b7192647c2cc582ff5a98`.
For FP32, copy the inference config, set `backend.config.dtype` to `float32`, and
run the same command with `--batch-sizes 1,8,64`. The original FP32 configuration
is retained beside its report as `fp32_config.json`.

The service configuration binds WebSocket to `127.0.0.1:18130`, uses a maximum
native batch of 64, and retains at most 256 contexts with a 600-second TTL.
The tested service is on H200 GPU 0, with a workstation SSH tunnel forwarding
local port 18130 to the remote loopback endpoint. Its remote log is
`/home/pengguanqi/Worksapce/Research/PressB-walloss/outputs/walloss_batch_h200_20261104/service.log`.
The network check can be repeated from the workstation with:

```bash
.conda/envs/wallx-inference/bin/python scripts/check_walloss_batch_service.py \
  --endpoint ws://127.0.0.1:18130 \
  --observation outputs/walloss_batch_h200_20261104/real_observations.json \
  --output outputs/walloss_batch_repeat/network_64.json --count 64
```

The [checked-in result summary](results/walloss_batch_h200_20261104.json)
records the direct-model and network evidence. Network inference passed;
complete 64-environment PAP/DSRL execution and closed-loop success-rate
comparisons remain separate validation work.
