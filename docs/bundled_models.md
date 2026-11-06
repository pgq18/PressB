# Model inference and supervised training inside PressB

PressB includes the model code needed for its Wall-OSS and VLA-JEPA experiments.
The default model adapters no longer require `ref/wall-x`, `src/VLA-JEPA`, or
access to the H200 workspace. Weights, backbone assets, datasets, Python packages
and CUDA toolkits remain separately provisioned inputs.

## Layout and provenance

| Location | Contents |
|---|---|
| `src/pressb/model_sources/walloss` | Wall-X model, native batches, external noise, context/prefix caches, data readers, supervised trainer, original serving protocol, CUDA sources |
| `src/pressb/model_sources/vlajepa` | PiPER VLA-JEPA model, pose9 and joint6/joint7 inference, data readers, single-task and multi-task supervised trainers |
| `src/pressb/models` | Frozen-model adapters used by the common three-node runtime |
| `src/pressb/training` | Common supervised-training launcher |
| `configs/model_training` | Six native YAML training templates |
| `configs/model_metadata` | Unchanged historical configuration paired with the deployed PAP checkpoint |

The bundles preserve upstream `wall_x`, `openpi_client` and `starVLA` module names
and tensor keys. Their `source_manifest.json` files record upstream repository
revisions, original file hashes, imported working-tree changes, and local edits.
In particular, the VLA-JEPA bundle includes the previously H200-local joint and
multi-task training modules. Original source notices are retained; see
[THIRD_PARTY.md](../THIRD_PARTY.md). No model or dataset is included in Git.

The source loader rejects a model family already imported from another tree.
Use separate processes/environments for Wall-OSS and VLA-JEPA. An explicit
`model.config.repo` (Wall-OSS), `model.config.vla_repo` (VLA-JEPA), or supervised
`--source-root` still selects a historical checkout. The default is bundled code.
Source identities hash live model files; a source migration does not silently
bypass the normal RL checkpoint identity checks.

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
