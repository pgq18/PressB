# Target-relative color demonstrations: nine completed datasets

Collection, LeRobot export and the full independent audit completed on
**2026-11-01 at 05:22:49 Asia/Shanghai**. This run contains **9,000 newly collected
scripted demonstrations**, with no episodes copied from previous datasets.
These are accepted generator demonstrations, not learned-policy success rates.

The [`target_relative_v2` configuration](../../../configs/piper_table_buttons_color_templates.json)
uses three approaches toward each target: red travels directly, yellow rises
while advancing and then approaches directly, and blue reaches above the target
before descending. The [motion guide](../../table_buttons_color_templates.md)
explains timing, geometry and collection commands. Earlier shared-motion
datasets remain separate and unchanged.

| Dataset | Episodes | Observation/action rows | Instruction |
|---|---:|---:|---|
| `near_red` | 1,000 | 143,006 | Press the red button. |
| `near_yellow` | 1,000 | 142,924 | Press the yellow button. |
| `near_blue` | 1,000 | 142,963 | Press the blue button. |
| `medium_red` | 1,000 | 143,004 | Press the red button. |
| `medium_yellow` | 1,000 | 142,924 | Press the yellow button. |
| `medium_blue` | 1,000 | 142,957 | Press the blue button. |
| `far_red` | 1,000 | 143,006 | Press the red button. |
| `far_yellow` | 1,000 | 142,900 | Press the yellow button. |
| `far_blue` | 1,000 | 142,907 | Press the blue button. |
| **Total** | **9,000** | **1,286,591** | Three instructions |

Every row has synchronized fixed-camera and wrist-camera RGB at 640×480,
30 Hz. Frame totals above count synchronized observations, not the sum of both
video streams. State and action retain six joint coordinates and gripper width;
the gripper stays fixed during these demonstrations. Export uses the existing
LeRobot v3 schema with degrees for joint angles and centimetres for full gripper
width. Collection stops on physical success; these clips do not include the
full release-and-return motion shown by the separate demonstration renderer.

## Validation

- All nine datasets passed physical trace, action/state timing, export and
  official LeRobot reader checks.
- All **18,000 raw camera videos** were fully decoded.
- Each group uses seeds **0–999**. All **3,000 profile/episode indices** have
  matching red/yellow/blue button layouts across the three target tasks; there
  are **zero pairing mismatches**.
- The collector recorded zero episode-level planning rejections, physical
  rejections or retries in the formal run. Internal candidate filtering during
  layout sampling is not counted as an episode-level attempt.
- The separate nine-episode smoke test is excluded from all formal counts.

The compact [machine-readable summary](summary.json) was extracted from the
completed run's `completion_summary.json`, `audit.json`, `pairing_report.json`
and `dataset_plan.json`; it records their local paths and hashes. The original
reports, raw data and videos remain under the ignored local output directory
`outputs/table_buttons_target_relative_1000_20261030/run`.

The nine standalone exported datasets are stored locally at
`/home/pengguanqi/Datasets/piper_table_buttons_target_relative_1000_20261030`.
This directory and the full source reports are not included in Git. No model
has been trained or evaluated on this new collection as part of this run.
