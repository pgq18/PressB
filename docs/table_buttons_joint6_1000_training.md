# Three red-button models from 1,000-episode datasets

**Completed experiment.** All three models finished training on 2026-10-26,
and their original-precision best/final checkpoints finished local archival on
2026-10-27. Subsequent 100-episode evaluations per spacing profile achieved
near **3/100**, medium **6/100** and far **44/100** task successes; the strict
collector counts were 3, 5 and 38. Evaluation used a 12-second timeout and no
extra smoothing. See the [policy evaluation guide](table_buttons_policy_eval.md)
for the two outcome definitions. These models used the earlier shared-motion
datasets, not the later `target_relative_v2` demonstrations.

The 2026-10-26 experiment trains one VLA-JEPA model for each table-button spacing profile. All three models receive the instruction **“Press the red button.”** Each model uses its own expanded 1,000-episode dataset and predicts six arm joints. The measured input state retains the gripper coordinate (seven values); execution appends the fixed full gripper width of approximately 0.2 mm.

| Model | Dataset | H200 GPU | Train / validation episodes | Updates |
| --- | --- | --- | --- | --- |
| `near_red_joint6_1000` | `near_red` | 0 | 900 / 100 | 4,896 |
| `medium_red_joint6_1000` | `medium_red` | 1 | 900 / 100 | 4,896 |
| `far_red_joint6_1000` | `far_red` | 2 | 900 / 100 | 4,896 |

All runs initialize independently from the same official VLA-JEPA pretraining checkpoint, with the four state/action interfaces reinitialized for this task. They do not resume the earlier 100-episode fine-tuned models. Each run uses three complete epochs, batch size 64, 245 warmup updates, and the previous joint6 learning rates. The near/medium training sets contain 104,404 frame anchors and far contains 104,406, giving 1,632 updates per epoch in each case.

The split uses a shared seeded permutation (`seed=42`) of episode IDs 0–999: 100 validation episodes and 900 training episodes. Validation selects five fixed anchors per held-out episode (500 anchors), every 100 updates and at epoch ends. Best-checkpoint selection uses first-action six-joint MAE in degrees. These offline validation episodes differ from the old 100-episode experiment, so its MAE and the new MAE are not measurements on the same holdout. Physical evaluation seeds 10000–10099 remain excluded from training and validation; no new simulation success rate is claimed by training or the post-training inference check.

## Recorded workflow and outputs

The following paths describe the completed local experiment. Its training and
archive orchestration scripts under `outputs/` and the VLA-JEPA training code
belong to separate local workspaces; they are not shipped with this PressB
checkout. The records are not instructions to restart the completed workflow.

The local run directory is `outputs/table_buttons_red_joint6_1000_20261026`. Read its `workflow_status.json` for the current state before starting or restarting anything. The expanded collection passed its complete 9,000-episode audit at 2026-10-26 02:53 Beijing time. A separate backup workflow moves the nine older 100-episode single-task models to this workstation and removes each remote model file only after verifying its local backup. The training workflow waits for sufficient persistent storage, transfers the three red datasets to H200, runs actual 1,000-episode CPU reader checks, and launches the three independent trainers after the old-model cleanup finishes. Each model's final inference check uses its own training GPU. Original-precision best/last model files are then archived to this workstation, one model at a time.

Useful paths within that run directory:

- `workflow_plan.json`: frozen paths, group/GPU mapping and launch commands.
- `workflow.log` and `workflow_status.json`: overall progress and failures.
- `groups/<model>/train.jsonl` and `train.log`: mirrored training metrics and log.
- `groups/<model>/archive_status.json`: durable model transfer status.
- `configs/<model>.yaml`: exact model/training configuration.
- `reports/training_geometry.json`: split and update-count derivation.

The workflow entry point is `code/scripts/run_workflow.py` within the run directory; it requires `--run` to perform the scheduled work. A running workflow owns a lock, and the remote launch checks process identity to avoid duplicate training. A dataset audit failure prevents launch. The old-model backup/cleanup status is under `outputs/table_buttons_joint7_retirement_20261026`.

## Checkpoint retention

This experiment uses `best_and_final_model_only`. A validation improvement atomically replaces the one best model; ordinary non-improving validations write metrics without another model. The end of training additionally saves the final model. If final and best are the same generation, they share model bytes through a hard link. Optimizer and RNG state are not written to checkpoints, so these new runs cannot resume an interrupted training step exactly. The old 100-episode checkpoint/resume behavior remains supported.

The earlier approximately 177 GiB peak estimate included a resumable last checkpoint, optimizer state and atomic replacement for three simultaneous trainers. Keeping only best/final model weights bounds the new peak to two full model files per run: approximately **57.4 GiB across three runs**, plus approximately 5.9 GiB for the datasets. This changes checkpoint retention, not the loss, training schedule, precision or model architecture.

## Storage

The H200 repository is `/home/pengguanqi/Worksapce/Research/VLA-JEPA` (the spelling is intentional). Small supervision files live under `runs/table_buttons_red_joint6_1000_20261026/<model>`.

The new dataset path `/home/pengguanqi/Datasets/piper_table_buttons_1000_20261025` resolves to the persistent `/data/scratch/pengguanqi/Datasets/piper_table_buttons_1000_20261025` directory. Only the three red groups are transferred for this experiment. The near and medium runs write under `/home/pengguanqi/vlajepa_red_joint6_1000_20261026/runs`; the far run writes under `/data/scratch/pengguanqi/VLA-JEPA-red-joint6-1000-20261026/runs`. New datasets and model files do not use `/tmp`.

Backing up the older nine single-task weights releases approximately 47.8 GiB of home quota and 38.3 GiB of scratch quota. Their local archive root is `/home/pengguanqi/Models/VLA-JEPA/table_buttons_joint7_20261023`. Remote configuration, processor, split, metrics and manifests remain available, together with a receipt pointing to the local model backup. Original datasets, the three multitask models and the earlier joint6 model are outside this cleanup. Before training, the workflow verifies the remaining home and scratch quotas rather than using filesystem-wide free space.

Durable models are also saved locally at `/home/pengguanqi/Models/VLA-JEPA/table_buttons_red_joint6_1000_20261026/<model>`. A completed `durable_archive.json` receipt confirms the original model bytes and supporting metadata were transferred. Both the H200 and local final checkpoints are model-only; this experiment does not produce a resumable optimizer checkpoint.
