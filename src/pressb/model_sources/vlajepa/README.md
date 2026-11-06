# Bundled VLA-JEPA source

This directory contains the PiPER model, supervised-training, data-reader and
inference code used by PressB. `starVLA` retains its original module and tensor
names so existing checkpoints remain loadable. Neither a separate VLA-JEPA
checkout nor access to H200 is needed to import these sources.

`source_manifest.json` records every imported file's hash, its upstream hash,
and whether it came from the local tracked checkout or the additional H200
working-tree modules. Both checkouts shared commit
`9c9dc71c8e199c7428b25d103ab666395832c001`. The model implementation is shared
by all five profiles; there is no fork of the model for joint-space policies.

| Profile | Native supervised-training module | State / action |
|---|---|---|
| `pose9` | `starVLA.training.train_piper` | 9 / 9 |
| `joint6` | `starVLA.training.train_table_buttons` | 7 / 6 |
| `joint7` | `starVLA.training.train_table_buttons` | 7 / 7 |
| `joint6_multitask` | `starVLA.training.train_table_buttons_multitask_joint6` | 7 / 6 |
| `joint7_multitask` | `starVLA.training.train_table_buttons_multitask` | 7 / 7 |

Use PressB's model-training entrypoint to select a profile and its YAML in
`configs/`. Native modules accept `--config`, `--run-id`, `--resume`,
and repeatable `--set KEY=VALUE`. Joint-space trainers also accept
`--probe-steps`; pose9 instead provides `--max-steps`, `--target-epochs`,
`--val-max-samples`, and `--continue-from`. Multi-GPU training uses
`torchrun`; the trainer initializes the distributed process group. Model-only
checkpoints cannot restore optimizer progress. The existing dataset schema,
normalization, layout-group splitting and checkpoint-provenance checks remain
in force. In particular, the joint6 multi-task reader requires the established
3,000-episode, three-color dataset format.

Portable YAML examples read `PRESSB_QWEN_ROOT`, `PRESSB_VJEPA_ROOT`,
`PRESSB_DATASET_ROOT`, and `PRESSB_PRETRAINED_CHECKPOINT` from the environment.
`PRESSB_RUN_ROOT` defaults to `outputs/model_training`. Weights, backbone
configuration/tokenizer assets and datasets are external inputs; none are
included in this source bundle. Required package versions are recorded in
`requirements-piper.txt` and `requirements-piper-inference-cu128.txt`; use a
separate model environment from Isaac Sim. The former preserves the H200
training stack, while the latter preserves the RTX 5090 inference stack.

This is the PiPER/Qwen3-VL/V-JEPA2 dependency closure, not the upstream LIBERO,
DROID or unrelated model families. Historical suite-launch/backup scripts are
not part of this bundle. The retained native policy-server scripts support
regression tests and old clients; PressB's model adapter is the normal service
entrypoint.

Original copyright/SPDX notices are preserved, including Apache-2.0 notices
in the NVIDIA flow-matching modules. Neither checked-out source contained the
`LICENSE` file referenced by its package metadata. `UPSTREAM_PYPROJECT.toml`
retains that metadata's MIT declaration; no missing license text is invented.

Run source regression tests in the VLA-JEPA environment with this directory on
`PYTHONPATH`. Some archived real-dataset tests need the named datasets; the
portable unit tests use synthetic Parquet/video fixtures. PressB's
`tests/test_bundled_vlajepa.py` additionally verifies source hashes and performs
a real CPU optimizer update and checkpoint round trip through a reduced-size
flow head for all three action/state interfaces. This CPU check does not stand
in for full pretrained GPU inference or full-model supervised training.
