# Table-buttons multi-task training

The dataset merger below belongs to the PressB workstation repository at
`/home/pengguanqi/Workspace/Research/PressB`; run that command there.
Training and model verification run in this H200 VLA-JEPA repository.

The nine source datasets are also available as three independent multi-task
datasets under `/home/pengguanqi/Datasets/piper_table_buttons_multitask_20261023`
on both the workstation and H200:

| Dataset | Red episodes | Yellow episodes | Blue episodes | Total frames |
| --- | ---: | ---: | ---: | ---: |
| `near_multitask` | 100 | 100 | 100 | 34,802 |
| `medium_multitask` | 100 | 100 | 100 | 34,804 |
| `far_multitask` | 100 | 100 | 100 | 34,804 |

These are regroupings of the same 900 demonstrations, not additional collection.
Each keeps `Press the red button.`, `Press the yellow button.`, and
`Press the blue button.` as separate tasks. Episode IDs are red 0–99,
yellow 100–199, and blue 200–299. The manifest preserves source group, original
episode ID, layout seed, and button positions. Global frame/task/episode indices
and statistics are updated; the original joint7 samples, 30 Hz timing and RGB
video bytes are retained. Original dataset files remain unchanged.

The merger uses the official LeRobot 0.6.1 aggregator without concatenating or
re-encoding video. It audits every numeric row, video-copy content, source-file
immutability, and official-reader image/language alignment across color boundaries.
Use a fresh destination to reproduce:

```bash
.conda/envs/lerobot/bin/python scripts/merge_table_buttons_datasets.py \
  --source /home/pengguanqi/Datasets/piper_table_buttons_20261023 \
  --destination /home/pengguanqi/Datasets/piper_table_buttons_multitask_20261023
```

Three independent VLA-JEPA models are queued on H200 after all nine single-task
runs and their checkpoint verification. Each starts from the same official
pretrained weights with newly initialized joint7 interfaces, uses batch 64,
and trains for three epochs (1,470 updates). For each color, 90 episodes train
and 10 validate: 270/30 per model. All colors of the same layout stay on the
same side of the split, matching the original seed-42 held-out episode IDs.
The model reads the instruction belonging to each episode. Validation reports
150 fixed anchors, 50 per color, and selects the best model by the equal-weight
mean of the three per-color action-error scores. This is offline prediction
validation, not a measured Isaac task success rate.

Remote configs, logs, and status are under
`/home/pengguanqi/Worksapce/Research/VLA-JEPA/runs/table_buttons_multitask_joint7_20261023`.
The remote repository's `scripts/train_table_buttons_multitask_suite.py` checks
the old suite's completion before using GPUs 0/1/2 for near/medium/far. Read
`suite_status.json` before starting another controller.

H200's remaining persistent quota is reserved for the original nine models.
The new best weights stay temporarily under
`/tmp/pengguanqi-table-buttons-multitask-20261023` and, after strict model reload
and inference checks, are automatically copied to the workstation's
`/home/pengguanqi/Models/VLA-JEPA/table_buttons_multitask_joint7_20261023`.
The local archive preserves model bytes, processor, configuration, split,
normalization, and logs. H200 retains its temporary best copy; its run directory
records the durable workstation location. Local queue/transfer evidence is in
`outputs/table_buttons_multitask_h200_training_20261023`, including
`local_archive_status.json`. Completed archival is recorded in
`durable_archive_completion.json` under the model root.


## Queue and archive commands

Inspect the current suite status before relaunching these controllers.

```bash
# H200, from the VLA-JEPA repository and its vlajepa-piper environment:
python scripts/train_table_buttons_multitask_suite.py

# Metadata-only audit after training; add --check-models to use a GPU.
python scripts/validate_table_buttons_multitask_suite.py \
  --suite-root runs/table_buttons_multitask_joint7_20261023

# Workstation, from PressB (automatically started for this experiment):
python3 outputs/table_buttons_multitask_h200_training_20261023/code/scripts/pull_table_buttons_multitask_models.py \
  --status-path outputs/table_buttons_multitask_h200_training_20261023/local_archive_status.json
```
