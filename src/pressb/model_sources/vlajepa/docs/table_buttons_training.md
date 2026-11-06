# Table-buttons joint-space fine-tuning

This suite trains nine independent VLA-JEPA models: `near`, `medium`, and `far`
spacing, each with `red`, `yellow`, and `blue` as the target. Every source
dataset contains 100 successful scripted Isaac demonstrations. The language
instruction is identical across spacing profiles for the same color, for
example, `Press the red button.`

The H200 repository is `/home/pengguanqi/Worksapce/Research/VLA-JEPA` (the
directory is spelled `Worksapce`). Data lives in
`/home/pengguanqi/Datasets/piper_table_buttons_20261023/<group>` and is read
without modification.

## Data and action contract

`starVLA/dataloader/table_buttons_lerobot.py` implements
`TableButtonsLeRobotDataset`. Camera order is **up, wrist**. RGB images and
numeric observations are sampled at 30 Hz; the model receives an eight-frame
video and predicts seven consecutive action targets. Source images are
640 × 480 and are resized to the configured model resolution.

Both state and action have seven channels:

| Channels | Source value | Model normalization | Inverse |
|---|---|---|---|
| 0–5 | Six joint angles, degrees | `value / 180` | `prediction * 180` |
| 6 | Full gripper opening, centimetres | `value / 8` | `prediction * 8` |

State is the measured joint configuration. Action is the absolute commanded
target for the following 1/30-second interval. There is no centering, clipping,
dataset-fitted normalization, Cartesian pose conversion, or removed gripper
channel. The near-constant gripper remains a valid seventh coordinate.
`normalize_joint7` and `denormalize_joint7` implement these conversions;
`normalization.json` and `action_schema.json` preserve them with each run.

The final stored action repeats the previous interval target. Future chunks
clamp to their own episode's final row, with explicit padding masks; they never
cross into the next episode. `sample_stride` changes anchor selection only,
not the 30 Hz future timestep.

## Shared experiment settings

- Each model uses 90 complete training episodes and 10 complete validation
  episodes. A NumPy permutation of IDs 0–99 with seed 42 selects the first ten
  IDs for validation. All nine datasets use exactly the same episode-ID split.
- Every run starts independently from the same official VLA-JEPA checkpoint.
  Four action/state interface tensors are reset for joint-space semantics,
  including action tensors whose shapes already match. A model trained on a
  different color or spacing is not the initialization for the next run.
- The suite targets three full training epochs with batch size 64 and gradient
  accumulation 1 per model. These settings are recorded in each resolved
  config and `initialization.json`. The last batch of an epoch can be smaller.
- Three GPUs run three independent single-GPU models concurrently, with the
  remaining models queued. This is not a three-GPU distributed run of one
  model. Each group has its own parameters, optimizer, logs, and checkpoint.

Dataset and training fingerprints bind the source manifests, numeric files,
split, fixed normalization, task identity, and training settings to a run.
Validation uses 50 fixed anchors: five anchors from each held-out episode.

## Commands and monitoring

Activate the existing H200 `vlajepa-piper` Conda environment and run commands
from the repository root. The suite configuration directory is
`runs/table_buttons_joint7_20261023/configs`; each group has its own YAML.

```bash
# Resource probe: two optimizer updates, without checkpoint publication.
CUDA_VISIBLE_DEVICES=0 python -m starVLA.training.train_table_buttons \
  --config runs/table_buttons_joint7_20261023/configs/near_red.yaml \
  --run-id near_red_probe --probe-steps 2

# Nine-run controller; resume/relaunch behavior is governed by suite state.
python scripts/train_table_buttons_suite.py \
  --suite-root runs/table_buttons_joint7_20261023 \
  --config-dir runs/table_buttons_joint7_20261023/configs

# Direct single-model launch, when no controller owns this group.
CUDA_VISIBLE_DEVICES=0 python -m starVLA.training.train_table_buttons \
  --config runs/table_buttons_joint7_20261023/configs/near_red.yaml
```

Active runs live in `/tmp/pengguanqi-table-buttons-20261023/<group>`.
For an interrupted individual run, append
`--resume /tmp/pengguanqi-table-buttons-20261023/near_red/checkpoints/last`
for the corresponding group. Resume restores that
run's optimizer, scheduler, RNG state, and epoch cursor. It requires the same
data/training contract and a complete resumable checkpoint. Use a fresh run ID
for a new experiment; a model-only best archive cannot resume optimization.

At the suite root, `suite_status.json` reports the queue and individual run
status, and `logs/<group>.log` captures each process's output. In an active run,
`status.json` is updated atomically and `train.jsonl` records every optimizer
update, including action loss, world-model loss, gradient norm, learning rates,
step duration, and peak GPU memory. `validation_<step>.json` records offline
evaluation. `epoch_geometry.json` contains the actual number of batches and
optimizer updates; `completed.json` marks successful training completion.

## Model selection, storage, and loading

Best-model selection minimizes
`first_joint_mae_deg + 22.5 * first_gripper_mae_cm` on the fixed validation
anchors. These are held-out action-prediction errors, **not simulator task
success rates**. Closed-loop Isaac evaluation is a separate experiment.

During training, `checkpoints/last` contains a resumable model plus optimizer
state, and `checkpoints/best` contains the best model. Superseded generations
are removed by the trainer. The suite archives the best model and run metadata,
then releases temporary optimizer checkpoints. To fit the available storage
quotas, `near_red`, `near_yellow`, `near_blue`, and `medium_red` keep their best
models at `<suite-root>/<group>/checkpoints/best` on scratch storage. The other
five use `/home/pengguanqi/vlajepa_table_buttons_models_20261023/<group>/best`,
with a symlink at the same suite-relative `checkpoints/best` path.
`archive.json` records the actual model path, step,
size, and digest; use that recorded path rather than assuming every model file
is on the same filesystem.

Keep the model together with its saved `config.yaml`, `processor/`,
`normalization.json`, `action_schema.json`, and provenance manifests. Reload the
saved processor and joint7 architecture; denormalize predicted joints to
degrees and full gripper width to centimetres before converting to an executor's
units. The existing elevator TCP/rotation-6D policy is not a compatible loader
for these joint-space checkpoints.

```bash
# Inspect completion and archive contracts without loading a GPU model.
python scripts/validate_table_buttons_suite.py \
  --suite-root runs/table_buttons_joint7_20261023

# Explicitly reload archived models and check finite joint7 predictions.
# Select an idle GPU; this check does not execute an Isaac task.
CUDA_VISIBLE_DEVICES=0 python scripts/validate_table_buttons_suite.py \
  --suite-root runs/table_buttons_joint7_20261023 \
  --check-models --device cuda:0
```

The validator supports `--groups near_red near_yellow near_blue` and an explicit
`--output` report path. Otherwise it writes timestamped reports under the suite's
`validation/` directory. Successful checkpoint reload and finite predictions do
not establish task execution success.
