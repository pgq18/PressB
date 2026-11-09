# Model inference and supervised training inside PressB

PressB includes model code for Wall-OSS, VLA-JEPA and Diffusion Policy experiments.
The default model adapters no longer require `ref/wall-x`, `src/VLA-JEPA`, or
access to the H200 workspace. Weights, backbone assets, datasets, Python packages
and CUDA toolkits remain separately provisioned inputs.

## Layout and provenance

| Location | Contents |
|---|---|
| `src/pressb/model_sources/walloss` | Wall-X model, native batches, external noise, context/prefix caches, data readers, supervised trainer, original serving protocol, CUDA sources |
| `src/pressb/model_sources/vlajepa` | PiPER VLA-JEPA model, pose9 and joint6/joint7 inference, data readers, single-task and multi-task supervised trainers |
| `src/pressb/model_sources/diffusion_policy` | Pinned DPPO denoiser/diffusion sources and visual conditioning |
| `src/pressb/models` | Model adapters and trainable action-expert wrappers |
| `src/pressb/training` | Common supervised-training launcher and image augmentation |
| `configs/model_training` | Supervised-training YAML templates |
| `configs/model_metadata` | Unchanged historical configuration paired with the deployed PAP checkpoint |

The bundles preserve upstream `wall_x`, `openpi_client` and `starVLA` module names
and tensor keys. Their `source_manifest.json` files record upstream repository
revisions, original file hashes, imported working-tree changes, and local edits.
In particular, the VLA-JEPA bundle includes the previously H200-local joint and
multi-task training modules. Original source notices are retained; see
[THIRD_PARTY.md](../THIRD_PARTY.md). No model or dataset is included in Git.

The ownership refactor keeps these upstream bundles separate from framework
integration. Frozen pose-model loading and cached flow execution now live in
`models/vlajepa_backend.py`; `models/vlajepa.py` implements the generic node
adapter. The historical elevator application and Wall-X hosting bridge belong
to `compat/elevator_rl`. Shared runtime, robot, control and sensor helpers do not
import concrete `models` implementations or `model_sources`. Keep new model adapters in `models` and supervised
launcher configuration in `training`, without moving task rules into model code.

The source loader rejects a model family already imported from another tree.
Use separate processes/environments for Wall-OSS and VLA-JEPA. An explicit
`model.config.repo` (Wall-OSS), `model.config.vla_repo` (VLA-JEPA), or supervised
`--source-root` still selects a historical checkout. The default is bundled code.
Source identities hash live model files; a source migration does not silently
bypass the normal RL checkpoint identity checks.

Diffusion Policy uses `--model diffusion_policy` with
`configs/model_training/diffusion_policy_pap.yaml`. Its native LeRobot reader
uses episode-disjoint validation, training-only normalization and the shared
image augmentation pipeline. The visual encoder and diffusion denoiser are
trained by SFT; online DPPO subsequently freezes the encoder and updates only
the denoiser. See [DPPO setup and contracts](dppo.md).

## Inference

Use `scripts/run_experiment.py inference --config ...` with a template from
`configs/experiments/`. Templates select the bundled implementation by omitting
external source paths. Supply local weights and normalization/backbone paths in
the model configuration. Simulation, rewards and RL remain in the common
runtime and task plugins.

Wall-OSS requires `wallx_csrc` built for the selected Torch/CUDA/GPU stack. The
supported builder uses the bundled CUDA source and a pinned CUTLASS dependency:

```bash
CUDA_HOME=/usr/local/cuda-12.8 .conda/envs/wallx-inference/bin/python \
  src/pressb/model_sources/walloss/build_extension.py --arch 12.0 --jobs 4
```

Use `--arch 9.0` for H200 with its matching toolkit and model environment. Build
files and binaries stay in the user cache, outside the repository. The successful
build records hashes and a cache pointer. `PRESSB_WALLX_EXTENSION_DIR` or the
adapter's `extension_dir` can select an explicit build. Source selection never
downloads packages or compiles code automatically. See the bundle's
[build and environment guide](../src/pressb/model_sources/walloss/README.md).

PAP inference uses `configs/model_metadata/walloss_pap_20260606.yaml`, whose bytes
match the historical checkpoint's training config. Its original training paths
are retained as metadata. Do not launch a new training run with that metadata;
use the fresh template below, which has no inherited resume setting.

## Supervised training

Run in the appropriate model environment. The shared launcher supports the
native training parameters after `--`; it does not alter the underlying loss,
normalization, optimizer or checkpoint rules. `--dry-run` reports the chosen
source/entrypoint and arguments without importing the model or starting training;
the native trainer performs dataset and training-contract validation at launch.

For Wall-OSS, fill the checkpoint, dataset, normalization and output paths in a
copy of `configs/model_training/walloss_pap.yaml`, then run:

```bash
python scripts/train_model.py --model walloss --config /path/to/walloss_job.yaml --dry-run
python scripts/train_model.py --model walloss --config /path/to/walloss_job.yaml -- --seed 42
```

The original Accelerate trainer is included. Training additionally needs the
bundle's `requirements-training.txt`; an inference-only environment does not
provide all dataset dependencies. CUDA Torch versions must match the chosen
hardware profile. Avoid installing training dependencies into Isaac Sim.

For VLA-JEPA, set the four local input paths and optional output root:

```bash
export PRESSB_QWEN_ROOT=/path/to/Qwen3-VL-2B-Instruct
export PRESSB_VJEPA_ROOT=/path/to/vjepa2-vitl-fpc64-256
export PRESSB_PRETRAINED_CHECKPOINT=/path/to/pretrained-model.pt
export PRESSB_DATASET_ROOT=/path/to/lerobot-dataset
export PRESSB_RUN_ROOT=/path/to/new-runs
python scripts/train_model.py --model vlajepa --profile joint6_multitask \
  --config configs/model_training/vlajepa_joint6_multitask.yaml --dry-run
torchrun --standalone --nproc_per_node=1 scripts/train_model.py \
  --model vlajepa --profile joint6_multitask \
  --config configs/model_training/vlajepa_joint6_multitask.yaml \
  -- --run-id my_table_buttons_run
```

Profiles are `pose9`, `joint6`, `joint7`, `joint6_multitask`, and
`joint7_multitask`; use the matching YAML. Increase `--nproc_per_node` only for a
planned multi-GPU run. The existing schema checks still apply: the current
joint6 multi-task trainer expects the established 3,000-episode, three-color
dataset structure, and six-action checkpoints retain a seven-dimensional
measured state. This import does not generalize those datasets implicitly.

`python -m pressb.training` and the installed `pressb-model-train` command are
equivalent to `scripts/train_model.py`. DSRL and action-residual RL continue to
use `scripts/run_experiment.py train`; supervised model training and online RL
have separate configurations and entrypoints.

### Shared SFT image augmentation

All bundled SFT profiles use `pressb.training.image_transforms`. Augmentation
defaults to enabled for training, including older configs without the block;
validation, test and evaluation splits always disable it. Configure the
top-level `image_transforms` block in a training YAML. The historical Wall-OSS
`data.lerobot_config.image_transforms` block remains supported when the top-level
block is absent. To opt out, set `image_transforms: {enable: false}`.

The default preset samples up to three transforms without replacement, with
equal weights, and applies the selected transforms in configuration order:

| Transform | Default range |
|---|---|
| Brightness, contrast, saturation | Each `[0.8, 1.2]` |
| Hue | `[-0.05, 0.05]` |
| Sharpness | `[0.8, 1.2]` |
| Affine | Rotation `[-3, 3]` degrees; translation up to `0.05` of each image axis |

Set `max_num_transforms`, `random_order`, and each entry's `weight`, `type` and
`kwargs` under `tfs` to customize the preset; a supplied `tfs` mapping replaces
the candidate set. The training templates list all six entries, and trainers
save the resolved configuration with each run.
Transforms operate on decoded images after cache retrieval and before model
image/video processors; one sample's temporal frames and current image share
the sampled transforms. Parameters are keyed by training seed, epoch, sample
and camera so worker scheduling does not change a sample's augmentation.
State and action labels retain their original values.
