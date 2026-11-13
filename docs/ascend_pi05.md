# π₀.₅ supervised training on Ascend 910C

PressB's `pi05` plugin uses the same dataset, augmentation, training and
checkpoint interfaces on CUDA and Ascend. NPU execution selects HCCL, maps each
worker to its visible device, and computes the model's timestep embedding in
FP32. ARM CPU sharpness augmentation uses a scoped native convolution fallback.
Validation has no random image augmentation. See [the model guide](pi05.md).

## Validated environment

The November 13 deployment used an **aarch64 host, Python 3.11, CANN 9.0.0,
driver 25.5.5, PyTorch 2.10.0, torch_npu 2.10.0 and torchvision 0.25.0**.
The PyTorch/torchvision wheels came from the CPU index; `torch_npu` supplies the
accelerator backend. [Pinned Python constraints](../scripts/ascend/constraints-910c.txt)
also preserve NumPy 1.26.4, Transformers 5.3.0, SentencePiece 0.2.1 and
protobuf 5.29.6. These pins describe the tested environment, not a compatibility
claim for other CANN/driver combinations.

Create a dedicated conda prefix on the server's large data/workspace volume.
Run the following **on the Ascend server**, with a PressB checkout there:

```bash
export PRESSB_ASCEND_ROOT=/path/to/large-volume/pressb-ascend
export PRESSB_REPO=/path/to/PressB
mkdir -p "$PRESSB_ASCEND_ROOT"/{tmp,cache,models,datasets,runs,logs}
conda create -y -p "$PRESSB_ASCEND_ROOT/env" -c conda-forge python=3.11 pip
conda activate "$PRESSB_ASCEND_ROOT/env"
source /usr/local/Ascend/cann/set_env.sh  # use the installed toolkit's path
export TMPDIR="$PRESSB_ASCEND_ROOT/tmp"
export PIP_CACHE_DIR="$PRESSB_ASCEND_ROOT/cache/pip"
bash "$PRESSB_REPO/scripts/ascend/install_pi05.sh"
```

The installer refuses a missing/base conda environment and installs into that
prefix only. It reuses `scripts/install_pi05.py` for the checksum-pinned LeRobot
source and does not install drivers or change the system Python. Preserve any
`PYTHONPATH` entries established by CANN when adding PressB:

```bash
export PYTHONPATH="$PRESSB_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=8
export HCCL_EXEC_TIMEOUT=1800 PYTHONUNBUFFERED=1
export HF_HOME="$PRESSB_ASCEND_ROOT/cache/huggingface"
```

## Download inputs on the server

Use the server's existing Hugging Face login, or run `hf auth login` there.
When changing `HF_HOME`, either log in after setting it or set `HF_TOKEN_PATH`
to the existing token file. Never put tokens in configs or command-line
arguments. The two example datasets are private and require repository access.
Large datasets and weights download directly to the server; no workstation-to-
server weight/data transfer is needed.

```bash
export HF_HUB_OFFLINE=0
hf download pgq18/pressb_cube_stack_100_20261113 --repo-type dataset \
  --revision 1a41e724c8d99df2d54fc5af57540788f526cf5a \
  --local-dir "$PRESSB_ASCEND_ROOT/datasets/cube_stack"
hf download pgq18/pressb_tabletop_cleanup_100_20261113 --repo-type dataset \
  --revision 27f29c73e9d5911e9f6ebe81f813e534e9aac208 \
  --local-dir "$PRESSB_ASCEND_ROOT/datasets/tabletop_cleanup"
hf download lerobot/pi05_base \
  --revision 22c30e5b34f9e7ab738b1b34fc2fc5ecfe2d28f0 \
  --local-dir "$PRESSB_ASCEND_ROOT/models/pi05_base"
python "$PRESSB_REPO/scripts/prepare_pi05_tokenizer.py" \
  --destination "$PRESSB_ASCEND_ROOT/models/paligemma-tokenizer"
export HF_HUB_OFFLINE=1
```

The tokenizer helper downloads the official public PaliGemma SentencePiece
artifact, checks its SHA256, and compares the resulting Hugging Face token IDs
against SentencePiece on 104 prompts. It refuses to overwrite an existing
directory. `--source-model` permits an offline rebuild and `--reference` adds
comparison against an existing tokenizer. The source/model hashes and tested
input revisions are recorded in [the validation receipt](results/pi05_ascend_20261113.json).

## Six-device sequential training

The [Stack](../configs/model_training/pi05_ascend/cube_stack.yaml) and
[Cleanup](../configs/model_training/pi05_ascend/tabletop_cleanup.yaml) configs
use paths relative to `PRESSB_ASCEND_ROOT`. Both independently initialize from
the same base model, train only the action expert, and use:

| Setting | Value |
|---|---:|
| Optimizer updates per task | 40,000 |
| Action chunk / prediction length | 16 |
| Batch per device / accumulation | 24 / 1 |
| Six-device global batch | 144 |
| Precision / gradient checkpointing | BF16 / enabled |
| Episode split | 90 train / 10 validation |
| Validation / scheduled save interval | 1,000 / 5,000 updates |
| Checkpoint retention | latest two generations plus best validation |

Gradient checkpointing stays enabled: the tested six-device batch-24 run
without it exhausted NPU memory. Check device availability with `npu-smi info`
before choosing the visibility list; IDs 2–7 were the idle devices in the
recorded deployment, not a mandatory selection for other servers.

```bash
cd "$PRESSB_ASCEND_ROOT"
export ASCEND_RT_VISIBLE_DEVICES=2,3,4,5,6,7
python "$PRESSB_REPO/scripts/ascend/torchrun_local.py" \
  --nnodes=1 --nproc-per-node=6 --master-addr=127.0.0.1 --master-port=29551 \
  "$PRESSB_REPO/scripts/ascend/hccl_probe.py"
python -m pressb.training.pi05 \
  --config "$PRESSB_REPO/configs/model_training/pi05_ascend/cube_stack.yaml" --prepare-only
python -m pressb.training.pi05 \
  --config "$PRESSB_REPO/configs/model_training/pi05_ascend/tabletop_cleanup.yaml" --prepare-only
```

Before a long run on a new installation, make a copy of the training config
with `steps: 2`, `log_every: 1`, `eval_every: 2`, `save_every: 2`,
`validation_samples: 6` and a separate `output_dir`. Run it using the same
six-device command below and check that training, validation and native
checkpoint export complete. HCCL and dataset preparation alone do not exercise
the full policy. Never reuse the short-test output directory for production.

Run the two production jobs sequentially in a persistent terminal/session:

```bash
set -euo pipefail
cd "$PRESSB_ASCEND_ROOT"
for task in cube_stack tabletop_cleanup; do
  python "$PRESSB_REPO/scripts/ascend/torchrun_local.py" \
    --nnodes=1 --nproc-per-node=6 --master-addr=127.0.0.1 --master-port=29551 \
    "$PRESSB_REPO/scripts/train_model.py" --model pi05 \
    --config "$PRESSB_REPO/configs/model_training/pi05_ascend/$task.yaml" \
    2>&1 | tee "logs/train_$task.log"
done
```

The launcher is for a **single host** whose own hostname does not resolve.
It changes only its parent process's local-host FQDN lookup; it does not edit
system DNS. Use ordinary `torchrun` on a normally configured host. The six
workers address the selected devices as `npu:0` through `npu:5`.

`metrics.jsonl` under each configured run contains optimizer step, zero-based
epoch, loss, validation loss, gradient norm, learning rate, batch geometry and
elapsed time. Final step 40,000 is always saved. These are **model-only** native
LeRobot checkpoints with processors, tokenizer and PressB action metadata;
optimizer-state resume is not provided. Training refuses an existing output
directory. Archive a failed attempt or choose a fresh output path deliberately.

## Recorded validation

The November 13 deployment passed single-device pretrained-policy forward/
backward and chunk inference, six-device HCCL/BF16 backward, and a two-update
real-data SFT run including validation and native checkpoint export. A separate
eight-worker test read 480 real augmented samples after the ARM sharpness fix.
The six-device run used a global batch of 144 and produced finite losses and
gradients. The immutable [receipt](results/pi05_ascend_20261113.json) distinguishes
these checks from production training and simulation task-success evaluation.
