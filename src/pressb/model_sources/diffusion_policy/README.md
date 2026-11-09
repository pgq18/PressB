# Bundled visual Diffusion Policy

The temporal conditional `Unet1D`, cosine DDPM schedule, denoising formulas and
their helper modules come from the MIT-licensed
[DPPO implementation](https://github.com/irom-princeton/dppo/tree/cc7234ad7ff39a8f32de3af903606723a16f0648).
`source_manifest.json` records original and bundled file hashes. Vendored files
only change package-relative imports; `LICENSE` retains the original notice.

`policy.py` is the PressB integration. It adds a shared ResNet18 with GroupNorm
and spatial softmax over each camera, measured joint state, and a learned task
embedding. Instruction strings are an explicit finite training vocabulary;
this policy does not claim a pretrained language model or unseen-language
generalization. The conditional temporal UNet predicts epsilon on normalized
absolute action chunks. Its architecture and DDPM training loss are upstream
implementations, not a state-only controller or a residual actor.

`pressb-model-train --model diffusion_policy --config
configs/model_training/diffusion_policy_pap.yaml` trains from LeRobot v3.0 data.
Units are configured explicitly; the public backend uses SI units. Images use
the shared train-only SFT augmentation, while validation stays unaugmented.
Episode splits and train-only min/max statistics are persisted in checkpoints.
PAP's documented 0.03-second metadata discrepancy is handled by an explicit
`packed_validated` mode that checks every video PTS and packed episode length.
RGB caches are stored outside the immutable source dataset.

SFT trains both the visual encoder and UNet. For DPPO, the encoder stays frozen
and the UNet action expert is trainable. The expert returns differentiable
DDPM/DDIM Gaussian means and fixed standard deviations for the shared rollout
and likelihood implementation. Raw sampled latents are never clipped. The
clean reconstruction used inside a transition mean is clipped to [-1, 1], as
in upstream. DDIM policy-gradient sampling includes explicit `eta` and positive
`min_std`, including the final step. Baseline deterministic DDIM and stochastic
DPPO-kernel evaluations are distinct experimental settings and must be labeled.
Expert forward passes use IEEE float32 with autocast and TF32 disabled, even
when the caller enables them. Small rollout batches and larger mixed-step PPO
batches otherwise select different approximate convolution kernels, producing
material likelihood errors at small standard deviations. The precision scope
preserves caller settings and autograd; it does not impose serial inference.

Resized decoded images and encoded observation features are reusable. The
action expert registers only its UNet and immutable sampler/normalization
buffers; updates do not include the image encoder. Source weights are external
inputs and are not packaged in this repository.
