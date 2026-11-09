# Diffusion Policy and action-expert reinforcement learning

PressB supports `diffusion_policy` alongside `vlajepa`, `vlajepa_joint` and
`walloss`. Its `dppo` runner updates the model's action expert, using the existing
inference and simulation nodes. DSRL and action-residual SAC remain separate
methods that leave the base model frozen.

## Implementation and provenance

The bundled temporal UNet, cosine schedule and DDPM equations come from
[irom-princeton/dppo, revision cc7234ad](https://github.com/irom-princeton/dppo/tree/cc7234ad7ff39a8f32de3af903606723a16f0648).
The upstream MIT license and per-file provenance are retained in
[`model_sources/diffusion_policy`](../src/pressb/model_sources/diffusion_policy/).
PressB adds a ResNet18/GroupNorm and spatial-softmax observation encoder,
LeRobot readers, SI-unit conversion, the task-instruction embedding, and its
distributed runner. This is an integrated visual Diffusion Policy, not an
unchanged copy of the upstream benchmark configuration.

The original [DPPO method](https://diffusion-ppo.github.io/) treats each Gaussian
denoising transition as an inner control step and the environment as an outer
control process. Its
[PPO implementation](https://github.com/irom-princeton/dppo/blob/cc7234ad7ff39a8f32de3af903606723a16f0648/model/diffusion/diffusion_ppo.py)
provides the denoising discount and clipping schedule used here.

| Model | Optimized parameters | Stochastic kernel used for RL |
|---|---|---|
| Diffusion Policy | Conditional temporal UNet; visual/state/task encoding frozen | Cosine DDPM or subsampled DDIM reverse transitions |
| VLA-JEPA | Existing flow action head, including its state/action projections | Stochastic Euler flow extension |
| Wall-OSS | Action-route MoE MLPs and action input/output projections | Stochastic Euler flow extension |

Wall's visual encoder, text/shared transformer parameters, proprioception
projection and normalizers stay frozen. Its RL velocity evaluation recomputes
the transformer without reusing KV state from another set of weights.

**Flow matching is not DDPM.** VLA-JEPA and Wall-OSS originally use deterministic
Euler integration. Their extension samples
`x[k+1] = x[k] + dt[k] * velocity_theta(x[k], t[k], condition) + std[k] * epsilon`.
It scores that explicitly stochastic kernel; it does not assign a Gaussian
density to the original ODE or claim to preserve its action distribution.

## Sampling and optimization contracts

- `GaussianChainPolicy` retains the raw latent chain `[B,K+1,H,D]` and raw
  elementwise transition log probabilities `[B,K,H,D]`. Sampling and scoring
  use the same standard deviations, including a positive final-step floor.
  Clipping the reconstructed clean prediction inside the DDPM/DDIM mean is
  distinct from clipping the sampled latent, which this sampler does not do.
- Neural evaluation may use BF16; latent arithmetic and Gaussian densities use
  FP32. BF16/FP16 optimization uses FP32 master parameters and optimizer state,
  then copies rounded weights to both the active learner and inference expert.
  Diffusion Policy transition scoring explicitly disables TF32 and outer
  autocast in its action expert, then restores the caller settings. This keeps
  rollout and shuffled-minibatch log probabilities consistent across CUDA batch
  shapes; the raw-likelihood checks remain enabled.
- Wall uses canonical transformer shapes to prevent BF16 kernel changes from
  producing artificial PPO ratios. The PAP example sets
  `transition_microbatch_size: 8` and `transition_context_length: 256` in
  `nodes.inference.service.action_expert`: each request, including single rows
  and incomplete learner minibatches, is padded to the same batch and token
  shape. Unused duplicate rows are discarded. Contexts exceeding the explicit
  token bound are rejected, never truncated or silently sent through another
  kernel. The default remains canonical single-row evaluation. Changing these
  settings requires a new matching baseline. The [real BF16 probe](dppo_validation.md)
  found zero raw log-probability differences across all ten denoising steps,
  shuffled rows, variable-length contexts, incomplete batches and mixed steps;
  it also verified a differentiable backward pass.
- `dppo.logprob_reduction: upstream_mean` applies the upstream elementwise
  clamp (default `[-5,2]`) and mean over the executed action prefix. This is a tempered
  surrogate, not an exact joint-density ratio. `exact_sum` uses the complete
  horizon and all latent channels without this clamp; unexecuted latent
  coordinates can still influence later denoising steps.
  The elevator flow example explicitly uses `logprob_max: 10`: its SI latent
  transition standard deviation is `0.002`, giving a Gaussian peak log density
  near `5.30`. The upstream upper bound `2` would zero almost all near-mean
  likelihood gradients in these units. This is an explicit flow configuration
  choice, not a change to the default DDPM/DDIM objective.
- Every denoising step is currently trained. Omit `ft_denoising_steps` or set it
  equal to the chain length. A smaller value is rejected: supporting it requires
  a separately frozen prefix policy, not just omitting early loss terms.
- Rollouts are on-policy. One rollout uses one immutable expert version, with
  GAE computed from actual chunk-duration discounts. Terminal transitions do
  not bootstrap; time limits optionally bootstrap the terminal observation.
  Both episode boundaries cut GAE recursion. Old rollouts are discarded after
  the update; there is no SAC replay buffer.
- Before optimization, learner/inference base identities and sampled transition
  probabilities are checked. Raw log-probability drift is checked separately
  from the clipped PPO surrogate, so clipping cannot hide a mismatch.

Deterministic evaluation runs the selected kernel's means with no transition
noise and returns no likelihood. For DDIM, changing `eta` also changes the mean;
therefore use the same sampler configuration for baseline and trained-policy
comparisons. A stochastic evaluation is also possible through
`rollout.deterministic_eval: false`.

The PAP Diffusion Policy template uses `eta: 0` with a positive `min_std: 0.02`:
the reverse mean stays equal to the SFT DDIM sampler, while training draws
Gaussian transition noise with that floor. This is distinct from setting
`eta: 1` and merely omitting its noise during evaluation. Upstream DPPO resets
DDIM eta to zero for deterministic evaluation; this runner instead keeps the
configured mean fixed. The template's `logprob_max: 5` avoids clipping most
coordinates of this narrow Gaussian; its standard deviation gives a maximum
elementwise log density of about 2.99. Both choices are explicit experiment
settings, not the upstream benchmark defaults.

The DP and VLA templates use one PPO epoch, actor LR `1e-6`, and
`critic_warmup_iterations: 2`. A warmup iteration collects fresh on-policy
rollouts and verifies their probabilities, then updates only the value network;
it does not publish unchanged action weights. The count is in complete
rollout/update rounds, not transitions or optimizer minibatches. With
`critic_continue_after_kl: true`, reaching the actor KL threshold stops actor
updates for that round while the critic finishes its minibatches. Defaults of
zero warmup and no continuation preserve the original runner behavior.
Checkpoints retain the warmup cursor and separate actor/critic update counts.
Logs distinguish accepted-minibatch KL from the rejected KL that caused a stop.

The VLA template uses its pretrained head's native **four** Euler steps. Wall's
template retains the tested ten-step, canonical-B8 configuration, one PPO epoch,
minibatch 32, actor LR `1e-6`, and the upstream `logprob_max: 2`. Its normalized
noise scale is `0.03`, so this clamp flattens some near-mean likelihood gradients;
changing the bound changes the optimization objective and needs its own trial.
These are starting configurations, not a claim that all three models improve.
Completed comparisons and unsuccessful pilots are recorded in
[integration validation](dppo_validation.md).

## Three-node data flow

1. The learner obtains observations from simulation `reset/step` and requests
   inference `/sample_chain` with observations, per-row seeds and the expected
   policy version.
2. Inference returns executable actions, frozen conditioning, critic features,
   the sampled latent chain and its log probabilities. The learner sends the
   configured action prefix to the simulator. Simulation applies any configured
   execution bounds/smoothing and returns reward, terminal flags and duration.
3. The learner optimizes the action expert and its value network. After the PPO
   update round, it sends actual expert tensor bytes using `/weights_begin`,
   `/weights_append`, then `/weights_commit`.
4. Inference validates keys, shapes, dtypes, finiteness, immutable buffers and
   checksums, installs the complete update under its inference lock, invalidates
   cached contexts and acknowledges the new version/hash. The next rollout
   starts only after that acknowledgement. No shared filesystem is required.

The published payload excludes the observation encoder. Sampler and
normalization buffers cannot be changed through a weight update. Existing
`encode/decode/infer` calls share the updated action expert, while retaining their
own configured sampling behavior. DPPO currently accepts simulation backends;
it does not enable physical-robot training.

Simulation backends advertising `per_env_reset_seeds` receive one subsequent
`/reset` request for all environments ending in the same sampling step. Its
top-level `env_seeds` maps each environment ID to the same reset seed previously
sent in its singleton request; reset options and each environment's episode
counter are preserved. The required scalar `seed` remains present, while the
explicit per-environment seeds take precedence. Older backends keep singleton
requests. Initial resets retain their previous behavior, including one shared
seed when `initial_reset: all_envs` is required. This reduces repeated reset
calls without changing policy seeds, episode identities or GAE boundaries.
The negotiated seed-schedule description records the chosen path. Source and
checkpoint identity checks remain strict; this change does not silently migrate
an existing run or make old/new-source evaluation reports interchangeable.

## Supervised training

Use a model-training environment with Torch, torchvision, einops, PyArrow and
PyAV. Do not install these into the Isaac environment just to launch training.
Edit the dataset/output/device paths in
[`diffusion_policy_pap.yaml`](../configs/model_training/diffusion_policy_pap.yaml),
then run:

```bash
python scripts/train_model.py --model diffusion_policy \
  --config configs/model_training/diffusion_policy_pap.yaml --dry-run
python scripts/train_model.py --model diffusion_policy \
  --config configs/model_training/diffusion_policy_pap.yaml
```

The reader uses episode-disjoint train/validation splits, fits normalization
only on training episodes, repeats tail actions within the same episode and
converts the declared dataset units to radians/metres. The PAP template uses
seven joint/gripper dimensions and a 16-action horizon. Task strings select
trained instruction embeddings; this model does not provide an unrestricted
language encoder.

Training images use the shared PressB `image_transforms` preset. Validation
images are unaugmented, with fixed validation diffusion noise/timesteps.
`best.pt` is selected by validation denoising loss; `last.pt` and configured
milestone checkpoints are also saved. Denoising loss alone is not task success.

For the existing PAP recordings, `video_timestamp_mode: packed_validated` checks
packed video frame counts/PTS and the explicit metadata offset before matching
images to episode rows. This prevents frames from crossing episode boundaries.
The 0.03-second correction establishes episode membership; it is **not a
measurement of camera-to-robot synchronization**. Source data remain unchanged.

## Launching RL and evaluation

The experiment templates are:

- [`pap_diffusion_policy_dppo.json`](../configs/experiments/pap_diffusion_policy_dppo.json)
- [`pap_walloss_dppo.json`](../configs/experiments/pap_walloss_dppo.json)
- [`elevator_vlajepa_dppo.json`](../configs/experiments/elevator_vlajepa_dppo.json)

Set the trained checkpoint, devices, addresses and output paths for your hosts.
The PAP templates use an initial-state trajectory extracted from the dataset.
Prepare it once with the dataset dependencies installed:

```bash
python scripts/analyze_pap_dataset.py \
  --dataset /home/pengguanqi/Datasets/piper_pap_20260606_offset \
  --output outputs/pap_dataset_audit --episodes 0
```

The inference `service.action_expert` and trainer `action_expert` configurations
must match. Wall-OSS additionally needs its matching CUDA extension. Run the
three commands in separate processes, using the model environment for inference
and training and the Isaac environment for simulation:

```bash
python scripts/run_experiment.py validate --config configs/experiments/pap_diffusion_policy_dppo.json
python scripts/run_experiment.py simulation --config configs/experiments/pap_diffusion_policy_dppo.json
python scripts/run_experiment.py inference --config configs/experiments/pap_diffusion_policy_dppo.json
python scripts/run_experiment.py train --config configs/experiments/pap_diffusion_policy_dppo.json --seed 42
```

For a baseline, start fresh inference from the SFT checkpoint and use `eval`
with a new output directory. For a trained policy, pass its full DPPO checkpoint
using `--resume`; evaluation restores/publishes the expert without optimization:

```bash
python scripts/run_experiment.py eval --config configs/experiments/pap_diffusion_policy_dppo.json \
  --episodes 100 --seed 43 --output outputs/dppo_eval_baseline
python scripts/run_experiment.py eval --config configs/experiments/pap_diffusion_policy_dppo.json \
  --episodes 100 --seed 43 --resume /path/to/dppo/checkpoints/step_000100000.pt \
  --output outputs/dppo_eval_trained
```

Use a held-out evaluation seed distinct from training. Keep task randomization,
evaluation seeds, environment count, action horizon, timeout,
sampler and smoothing identical for the comparison. Allow enough
`max_transitions` for all requested episodes. Check `status.json`,
`metrics.jsonl`, `episodes.jsonl`, `events.jsonl` and `rpc_timing.jsonl` for
completion, outcome counts, version acknowledgements and timing.

## Checkpoint storage on another machine

DPPO checkpoints include the action expert, FP32 master parameters, Adam state,
value network and RNG state. When the learner's storage quota is insufficient,
set `run.checkpoint_sink` to an HTTP endpoint and a run-specific namespace.
Checkpoints then stream directly to the storage host, with SHA256 verification
and atomic publication; no learner-side `.pt` file or temporary disk cache is
created. All interval checkpoints are retained. The optional
[`serve_checkpoint_sink.py`](../scripts/serve_checkpoint_sink.py) receiver can
run on local loopback behind a reverse SSH tunnel.

Both a filesystem checkpoint path and a sink HTTP(S) URL can be passed to
`--resume`. URL reads are verified and loaded into RAM, so available host memory
must accommodate the checkpoint. Only the configured learner GPU's RNG state
is captured or restored. See [streaming checkpoint storage](checkpoint_sink.md)
for receiver commands, authentication and configuration.

Passing optimizer, protocol or small-model tests establishes implementation
behavior only. A performance claim requires completed Isaac evaluations of
the SFT baseline and RL checkpoint under matched settings, with episode counts
and uncertainty reported. Training recent success rate is a diagnostic, not a
replacement for that evaluation.

## Measuring throughput and component time

[`benchmark_dppo_training.py`](../scripts/benchmark_dppo_training.py) runs the
real trainer from an explicitly selected source tree. Pass a resolved, flat
`DPPOTrainerConfig` JSON, rather than an experiment template:

```bash
python scripts/benchmark_dppo_training.py \
  --source-root /path/to/frozen/pressb/src \
  --expected-trainer-sha256 TRAINER_SHA256 \
  --config /path/to/resolved/trainer.json \
  --output outputs/dppo_benchmark/train \
  --skip-checkpoints
```

The wrapper records initialization, collection, reset, old-policy likelihood
verification, optimization, expert publication and checkpoint calls. Nested
events carry both inclusive and exclusive durations. Combine them with the
trainer's `rpc_timing.jsonl` to split simulation step/reset and inference
sample/bootstrap requests. RPC durations include transport and server work;
they are not CUDA kernel measurements. Actor and critic optimization share one
timed phase, and observation encoding and denoising share the `sample_chain`
RPC. Do not report their internal costs as separately measured values.

Use complete post-warmup round boundaries for throughput, including episode
resets and expert publication. The marker named `steady` represents gaps
between methods, not the complete training interval. When changing environment
count, report both per-environment rollout length and the global PPO batch;
keeping the latter constant still changes per-environment GAE truncation.

`--skip-checkpoints` explicitly disables all checkpoint writes, including the
final checkpoint. Optimization and publication to inference remain active.
Omit the flag when checkpoint overhead or a reusable final policy is required.
Record this choice in comparisons with earlier runs.

Run the independent [resource monitor](runtime_resource_monitor.md) before
starting services, on each participating host. Align resource summaries to the
same round timestamps and select only the experiment's GPUs and process roles.
