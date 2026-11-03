# Three multi-task models from 3,000-episode datasets

**Completed experiment.** All three models finished training, inference checks
and local checkpoint archival on 2026-10-28. The subsequent 900-episode
evaluation achieved **85/900 task successes (9.44%)** and **78/900 strict
collector successes (8.67%)**. Per-spacing totals were near 11/300, medium
18/300 and far 56/300, with a 12-second timeout and no extra smoothing.
See the [policy evaluation guide](table_buttons_policy_eval.md) for scoring
definitions. Training used the earlier shared-motion demonstrations, not the
new `target_relative_v2` collection.

This experiment trains an independent VLA-JEPA model for each table-button
spacing profile. Each dataset contains 1,000 demonstrations each of
`Press the red button.`, `Press the yellow button.`, and `Press the blue button.`
The model receives the instruction belonging to the sampled episode.

| Model | H200 GPU | Training / validation episodes | Updates |
| --- | --- | --- | --- |
| `near_multitask_joint6_3000` | 0 | 2,700 / 300 | 14,682 |
| `medium_multitask_joint6_3000` | 1 | 2,700 / 300 | 14,682 |
| `far_multitask_joint6_3000` | 2 | 2,700 / 300 | 14,685 |

All runs start independently from the official pretrained checkpoint, resetting
the four task-specific state/action interfaces. They predict six arm joint
angles, retain all seven measured state coordinates, and leave the execution
gripper at the dataset's fixed full width of approximately 0.2 mm. The datasets
themselves retain their original joint7 values.

The schedule is three complete epochs with batch size 64, 734 warmup updates,
and the learning rates used for the previous red-button joint6 experiments.
Each profile uses 900 training layouts and 100 held-out layouts. The split
groups actual colored-button positions, keeping all three tasks for a layout
together, and preserves the earlier seed-42 split by red source episode ID.
Physical evaluation seeds 10000–10099 are excluded.

Validation runs every 300 updates and at epoch ends. It uses five fixed anchors
per held-out episode: 1,500 total, 500 per color. Best-checkpoint selection uses
the equal-weight mean of the three first-action six-joint MAEs in degrees.
These are offline prediction metrics, not simulator task success rates.

## Runtime and outputs

The paths below are records of the completed experiment. Model training,
inference and archive orchestration depend on the separate VLA-JEPA workspace
and local scripts under ignored `outputs/`; this PressB checkout contains the
simulator, dataset tools and evaluation client, not those training scripts.

The local supervision directory is
`outputs/table_buttons_multitask_joint6_3000_20261027`. Read
`workflow_status.json` before starting or restarting any process. The exact
configs are in `configs/`; mirrored logs are in
`groups/<model>/train.jsonl` and `train.log`.

H200 uses `/home/pengguanqi/Worksapce/Research/VLA-JEPA`, with supervisor records
under `runs/table_buttons_multitask_joint6_3000_20261027`. The datasets are at
`/home/pengguanqi/Datasets/piper_table_buttons_multitask_3000_20261027`, backed by
the persistent `/data/scratch/pengguanqi/Datasets` directory. Identical existing
red videos are shared as immutable hard links; all transferred dataset files
were checked against the local originals.

Near and medium checkpoints use
`/home/pengguanqi/vlajepa_multitask_joint6_3000_20261027/runs`; far uses
`/data/scratch/pengguanqi/VLA-JEPA-multitask-joint6-3000-20261027/runs`.
Only best and final model weights are retained. Atomic best replacement needs
at most two full model payloads per run. Optimizer/RNG state is not saved, so
these checkpoints do not support exact training resume.

After training, each supervisor reloads its best checkpoint on its own GPU and
checks predictions for all three language tasks. A separate workstation process
archives original-precision best/last weights and metadata to
`/home/pengguanqi/Models/VLA-JEPA/table_buttons_multitask_joint6_3000_20261027`.
`durable_archive.json` is the completion receipt for each local archive.

With explicit user approval, the previous three red-task 1,000-episode models'
remote model payloads were removed after verifying their local backups. Their
local models and all remote metadata remain. The cleanup receipt is
`outputs/table_buttons_multitask_joint6_3000_20261027/storage/checkpoint_cleanup_receipt.json`.
This released approximately 47.8 GiB of H200 user quota; no new data or weights
are placed under `/tmp`.
