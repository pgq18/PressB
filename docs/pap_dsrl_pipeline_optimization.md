# PAP DSRL pipeline optimization

The subsequent [local RTX 5090 comparison](pap_dsrl_local_inference.md) completed
on 2026-11-05: simulation and inference sharing GPU 0 measured 13.3434
transitions/s, versus the 12.5489/s H200 deployment documented below.

This profile retains 64 PAP environments, native 640×480 calibrated cameras,
32 actions per transition, ten BF16 flow steps, the 20-second episode limit,
the physical grasp/place success rule, and requested UTD 1. It optimizes the
work between these operations. The original configurations remain available.

The initial training run was stopped at the user's request at 3,520 transitions
and 1,532 updates. Its full checkpoint, replay and logs are preserved at
`outputs/pap_dsrl_h200_bf16_64env_400k_20261104/`. Do not resume that replay into
this profile: rendering, sampling scheduling and model runtime identities differ.

## Configuration and deployment

Use these three configurations together:

- [PAP simulation](../configs/rl_nodes/pap_vector_fast_simulation.json): local
  RTX 5090 GPU 0, 64 environments, HTTP port 18101.
- [Frozen Wall-OSS inference](../configs/rl_nodes/pap_walloss_h200_fast_inference.json):
  H200, WebSocket port 18130, native batch 64.
- [DSRL trainer](../configs/rl_nodes/pap_dsrl_fast.json): local RTX 5090 GPU 1,
  400k budget, warmup 2,000, replay capacity 100k, UTD 1. W&B is optional.

Paths in these examples describe the current deployment. Adjust checkpoint,
scene, trajectory and Python paths on another machine. Keep the inference
endpoint private, for example through the existing SSH forward. Launch each
service with its own model/Isaac environment; launch the trainer with the
`zprl` environment. No model parameters are trained by the inference service.

After deploying the matching source and rebuilding Wall's CUDA extension on
H200 as described in [the batch guide](walloss_batch_inference.md), launch the
inference service there:

```bash
cd /home/pengguanqi/Worksapce/Research/PressB-walloss
CUDA_VISIBLE_DEVICES=2 .venv/bin/python scripts/serve_inference_node.py \
  --config configs/rl_nodes/pap_walloss_h200_fast_inference.json
```

On the simulation/training machine, use separate terminals for the services
and a fresh output directory for each training run:

```bash
ssh -N -L 18130:127.0.0.1:18130 h200

.conda/envs/pressb/bin/python scripts/serve_simulation_node.py \
  --config configs/rl_nodes/pap_vector_fast_simulation.json

/home/pengguanqi/miniconda3/envs/zprl/bin/python scripts/run_rl_training.py \
  --config configs/rl_nodes/pap_dsrl_fast.json \
  --output outputs/pap_dsrl_fast_NEW_RUN
```

Reuse an existing healthy service or tunnel instead of starting a duplicate on
the same port. W&B can be enabled in the trainer configuration. SIGTERM to the
trainer requests an orderly stop and saves a full checkpoint; inspect its final
status before terminating the simulation or inference service.

The optimization validation uses physical H200 GPU 2, exposed as logical
`cuda:0` with `CUDA_VISIBLE_DEVICES=2`; another user's job occupies H200 GPU 0.
The original model service on GPU 0 was stopped after confirming its cache was
empty. Other users' processes were not changed. This placement change means
the end-to-end before/after numbers are not an isolated single-factor benchmark.

## What changes

1. **Clear the actual renderer history.** `renderer_profile="rl_fast"` retains
   DLSS/Ray Reconstruction and explicitly clears its temporal history at each
   observation boundary. Ordinary USD accumulation reset did not clear it.
   Two zero-physics-time render frames replace the 48-frame reset warmup.
   Effective renderer settings are checked after every capture. Camera optics,
   aspect ratio and native render resolution are unchanged.
2. **Resize before serializing the simulation images.** The simulator performs
   exactly Wall's existing first PIL bicubic resize, then sends a small lossless
   RGB PNG with its original dimensions and preprocessing identity. The trainer
   forwards binary PNG over WebSocket. Matching model/simulation preprocessing
   is negotiated before reset. Full PNG validation remains on the model node.
3. **Fuse action execution and indexed reset.** The opt-in `step_and_reset`
   operation records real terminal state/reward/time first, resets only ended
   slots without advancing physics, and captures continuing/new observations
   together. Old terminal `items` and new `reset_items` have distinct episode
   identities. One request-cache commit prevents accidental repeated physics.
   The default `step` and `reset` APIs remain unchanged.
4. **Skip unused terminal visual inference.** With
   `terminal_observation_mode="state_only_if_no_bootstrap"`, successful and
   truncated transitions keep their real terminal proprioception and duration.
   Only unused visual features and next residual bases use finite zero
   placeholders. This mode requires `bootstrap_truncated=false`; the SAC target
   is unchanged because continuation is zero. Reset observations never replace
   terminal observations in replay.
5. **Reuse the invariant flow prefix.**
   `flow_prefix_cache=true, flow_prefix_cache_mode="full_attention"` caches
   prefix layer states within one decode and skips their repeated MLP work.
   Attention shapes remain identical to the original path. This is separate
   from the persistent observation-embedding cache, and no state crosses decode
   requests. The faster experimental `suffix` mode changes BF16 arithmetic and
   is not selected by this profile.
6. **Reduce outcome-check allocations.** Fixed box transforms and geometry are
   cached, and quaternion transforms are reused. Contact requirements, lifting,
   floor-rest, velocity, release and stability thresholds are unchanged.

The trainer still initiates all requests. SAC updates overlap pending RPCs;
the simulation and model do not run arbitrary chunks asynchronously with stale
observations. Fused reset removes duplicate rendering, without claiming a fully
overlapped physics/model pipeline.

## Correctness evidence

- Real 64-layout Wall checks used each layout's original instruction and three
  initial-noise families. Features and actions with `full_attention` matched the
  original **batch-64 BF16** path exactly in these samples; repeated/reordered
  decoding matched and encoded contexts were unchanged. This does not remove
  the previously measured BF16 batch-versus-singleton differences. Model decode
  was about 2.24–2.29 → 1.07 seconds.
- The original first resize produced identical RGB hashes across Pillow 9.2.0,
  11.2.1 and 12.3.0 for six real camera images. The final processor's ten input
  fields also matched exactly. Render resolution is not model input resolution.
  The simulator-version check is in `sim_producer_preprocessing_cpu.json`;
  the training/inference-version check is in `model/resize_equivalence.json`
  under the optimization evidence directory.
- Renderer audits found no red-object residue at its old position after reset;
  the previous history-reset path retained 43.86% of the object-mask area in
  the wrist view. Indexed reset preserved peers' physical states and clocks.
- A native-640×480, 64-environment recorded expert replay succeeded in all
  environments. Mixed batches with 57 and then 7 terminal environments each
  used only one shared two-frame capture; terminal/reset states remained
  distinct and continuing environments did not advance during reset. An
  all-terminal batch with zero remaining budget performed no reset or render.
  This verifies physical execution and success checks, not learned-policy
  success. Evidence: `pap_dsrl_optimize_20261104/fused_640_64env_v3/report.json`.
- The final CPU regression passed 295 tests in the trainer environment and
  seven elevator compatibility tests in its Python 3.11 environment. Wall's
  separate prefix suite passed 14 tests, in addition to the real GPU checks.
- Outcome optimization was compared field by field on 12,000 random poses and
  threshold cases, plus contact/lift/stability/latching sequences. In the Isaac
  Python environment, 8,192 cube checks took 0.428 → 0.116 seconds.

Raw evidence is under `outputs/pap_dsrl_optimization_20261104/`, renderer evidence
under `outputs/pap_dsrl_optimize_20261104/`, and the original timing profile under
`outputs/pap_dsrl_profile_20261104/`. Numerical equivalence tests do not establish
the success rate or convergence of a newly trained noise policy.

## End-to-end measurement

One transition is one environment action chunk, not an episode or video frame.
Count actual wall time, random resets, successful and timeout episodes, SAC
updates, cache lifecycle, and checkpoint saving. Exclude warmup from the main
steady-state comparison. Do not add overlapping learner time to RPC time.

```bash
/home/pengguanqi/miniconda3/envs/zprl/bin/python \
  scripts/analyze_rl_node_throughput.py \
  --run OUTPUT/train --output OUTPUT/throughput.json
```

The tool also checks update credit accounting and actual reset counts. A fused
RPC call without any reset is not accepted as reset evidence. Its default target
is 15 transitions/s over at least 300 seconds and 5,000 post-warmup transitions.
Projected 400k/1M durations are estimates from a measured window, not completed
long-training results. The replay buffer has not reached its 100k capacity in
the 10k validation; later full-replay checkpoints are larger. Model/server
startup, network contention and long-run behavior can also change actual time.

Historical elevator runs actually took 7.43 hours for 400k XYZ residual
transitions, 15.16 hours for 1M initial-noise transitions, and 19.81 hours for
1M action-residual transitions. An elevator transition executed at most seven
actions, versus up to 32 for PAP; equal transition counts are not equal amounts
of simulated control time.

The original post-warmup run measured 1.4406 transitions/s. The intermediate
4,096-transition candidate measured 10.0874 transitions/s after warmup, but used
the experimental suffix model path and still rendered step/reset separately;
it is not the final recommended profile.

The final native-resolution, full-attention run completed **10,048 transitions
and 8,018 SAC updates in 799.935 seconds**. The full run average was 12.561/s.
The post-warmup window (2,048 → 10,048) measured **8,000 transitions in 637.506
seconds, or 12.5489/s**, including periodic and final checkpoint saving. This is
**8.71×** the original post-warmup throughput. The window completed 8,001 updates
(consuming one previously outstanding update credit), dropped no credit, and
performed 580 actual environment resets in 123 batches. Both successful and
timeout episodes occurred.

| Pipeline | Measured transitions/s after warmup | Projected 400k | Projected 1M |
|---|---:|---:|---:|
| Original PAP | 1.4406 | 77.13 h | 192.83 h |
| Optimized native-640 PAP | 12.5489 | 8.85 h | 22.14 h |

The **15/s performance target was not met**; the other qualification checks
passed. The result reduces the projected duration from days to hours, but does
not establish equal speed to the historical elevator runs. A new 400k or 1M
training run has not been started.

Of the measured post-warmup wall time, simulation step plus fused reset occupied
38.2% (1.949 s/call), observation encode RPC 35.7% (1.820 s/call), and action
decode RPC 22.9% (1.166 s/call). Encode includes image validation and transfer,
not just GPU work. SAC consumed 50.12 seconds overlapping these RPC waits.

An ongoing H200 model backup shared the inference network path. A separate
fixed-batch A/B test briefly paused only its receiving rsync process for 10.87
seconds and then verified resumed writes. Encode medians were 1.841 s with the
backup, 1.718 s while paused, and 1.827 s after resuming; all feature hashes
matched and the model cache was empty afterward. This explains about 0.11–0.12
seconds per round, not the entire remaining bottleneck. The measured training
throughput above includes the ongoing backup.

Local sampled peak device use was 15,549 MiB on simulation GPU 0 and 1,029 MiB
on learner GPU 1 (including other allocations on those devices). Process-tree
RSS peaked at 14.69 GiB; at least 42.72 GiB host memory remained available.
These measurements do not include H200 memory.

The final full checkpoint passed finite-value, optimizer, replay and actor-update
checks. It preserves 46 outstanding update credits; no credits were discarded.
Terminal features are zero only under the declared no-bootstrap mode, while
terminal state fields remain present. The validation trainer exited and W&B
finished. Validation simulation and inference services were then stopped to
release their GPUs; the old training run remains stopped.

See the [tracked result summary](results/pap_dsrl_pipeline_optimization_20261104.json),
`fused_native_throughput.json`, `fused_native_checkpoint_audit.json` and
`network_contention_ab.json` in the optimization evidence directory, and the
[validation W&B run](https://wandb.ai/penggq2025-southern-university-of-science-technology/pressb-online-rl/runs/1781eecaceed49f08508cd3f1b8a301c).
