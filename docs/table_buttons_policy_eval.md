# Table-button policy evaluation

## Runtime scope and external inference service

PressB includes the simulation runner, HTTP client, evaluation supervisor,
independent outcome audit and video composition tools. The dedicated
table-button VLA-JEPA model server is a separate dependency: the historical
H200 workspace uses `scripts/serve_table_buttons_policies.py`,
`scripts/serve_table_buttons_joint6_policy.py` and
`starVLA/inference/table_buttons*_policy.py` under
`/home/pengguanqi/Worksapce/Research/VLA-JEPA`. As checked on 2026-11-03, these
server additions are local uncommitted files in that workspace, not included
in the VLA-JEPA `9c9dc71` checkout or this PressB repository. The baseline run's
`source_archive/remote_inference/` retains a local source snapshot. A fresh
PressB clone therefore also needs that separately provisioned model service
and checkpoint files before running these evaluations.

The client sends JSON over HTTP(S): `/load` selects `model_id` and returns its
identity; `/reset` begins an episode; `/predict` receives a one-sample batch
with instruction, measured joint7 state, base64 PNG `up`/`wrist` images and
an explicit noise seed. Responses must preserve model/episode/request
identity and provide both normalized and physical seven-step action chunks.
State and physical action units are degrees and centimetres. The joint6
adapter checks the six native predicted arm joints and verifies that the
server appended the declared fixed gripper width as the seventh wire channel.
The simulation does not advance while waiting for a response.

`scripts/prepare_table_buttons_policy_eval.py` prepares the original 18-cell
joint7 baseline from the 2026-10-23 collection: it requires 100 training
episodes per group and seeds starting at zero. `--reuse-plan` derives scoring
variants from a saved plan without regenerating its layouts. Later joint6
runs use their own frozen derived plans; this preparation CLI does **not**
directly prepare an evaluation of the new 1,000-episode `target_relative_v2`
datasets. Collection and export of those datasets are independent of this
evaluation preparation restriction.

Video composition requires FFmpeg, DejaVu and Noto Sans CJK fonts, as listed
in the main installation instructions. Detailed run paths below refer to
local evidence under ignored `outputs/`, not files distributed with Git.

## Original 100-episode joint7 baseline

The evaluation of twelve independently trained baseline VLA-JEPA models is complete.
All **1,800 episodes** passed the independent trajectory audit, with **zero missing
or invalid episodes**. Primary task success is **37/1800 (2.06%)**; the stricter
collector metric is **13/1800 (0.72%)**, using the same denominator. Single-task
models achieved **14/900 (1.56%)**, and multi-task models **23/900 (2.56%)** on the
primary metric.

The table reports primary successes out of 100 episodes in every color cell:

| Spacing | Model type | Red | Yellow | Blue |
|---|---|---:|---:|---:|
| Near | Single-task | 0/100 | 4/100 | 0/100 |
| Near | Multi-task | 0/100 | 0/100 | 0/100 |
| Medium | Single-task | 1/100 | 1/100 | 0/100 |
| Medium | Multi-task | 11/100 | 5/100 | 6/100 |
| Far | Single-task | 0/100 | 8/100 | 0/100 |
| Far | Multi-task | 1/100 | 0/100 | 0/100 |

Evaluation finished at **2026-10-25 03:16:54 Asia/Shanghai**
(2026-10-24 19:16:54 UTC), after 17,387.99 seconds, approximately 4 hours
50 minutes. Final trajectory and video auditing completed afterward, and the
three dedicated inference services were stopped automatically.

The local [result report](../outputs/table_buttons_policy_eval_20261024/run/results/README.md),
[per-cell CSV](../outputs/table_buttons_policy_eval_20261024/run/results/per_cell.csv)
and [machine-readable summary](../outputs/table_buttons_policy_eval_20261024/run/results/summary.json)
are derived from the final independent audit and include both success metrics.

The run directory is
`/home/pengguanqi/Workspace/Research/PressB/outputs/table_buttons_policy_eval_20261024/run`.
Use these files to inspect the completed run:

| File | Purpose |
|---|---|
| `plan.json` | Frozen formal plan, model/task cells, layouts and scoring rules |
| `workflow_status.json` | Evaluation, final audit and cleanup state |
| `suite_status.json` | Worker status, rotations and per-cell progress |
| `summary.json` | Both success metrics, completed counts and missing episodes |
| `cells/<cell_id>/episodes/` | Observations, actions, physics and contact evidence |
| `source_archive/` | Local runtime and remote inference source snapshots |
| `audit/summary.json` | Final independently reconstructed outcomes; `audit_passed=true`, `complete=true` |
| `service_cleanup.json` | Successful termination of all three dedicated H200 services |

The workflow's historical PID was `3915546`; both workflow and suite status now
report `state=complete`. Three Isaac workers shared local GPU1 and rotated after
20 completed episodes per process to release stage caches. Three dedicated H200
services used GPU0/1/2 and ports 18110/18111/18112 for near/medium/far respectively,
forwarded over SSH. Each cached its three single-task models and one multi-task
model. The workflow audited all completed trajectories, fully decoded recorded
videos, and cleaned up its services and tunnel after the audit succeeded.

Each model was evaluated in its matching collection configuration. The 100 new
layout seeds are `10000` through `10099`; paired single-task and multi-task cells
reuse the same scene and request random seeds. The simulator executes seven
joint7 targets per inference request at 30 Hz, with physics at 120 Hz and a
12-second simulation timeout. Inference waiting time does not advance physics.
The model's six joint targets and full gripper-width prediction are retained;
the execution layer only clips to physical joint limits. Extra action smoothing
is disabled and the gripper is not replaced with a fixed command.

Two outcome definitions were frozen before the formal run:

| Metric | Definition |
|---|---|
| `task_success` | Physical target-button success, permitting only the exact `/World/Piper/link7`–`/World/Piper/link8` finger-closure contact pair |
| `strict_collector_success` | Task success with no contact rejected by the original collector, including that finger-closure pair |
| `finger_closure_episode_rate` | Fraction of episodes containing the declared finger-closure contact |

Both success metrics use the same valid physical episodes as their denominator.
Wrong presses, non-target button contacts, other unexpected self-contact and
environment collisions remain failures. All contacts remain in the evidence;
an allowed finger closure can let the task continue while the strict outcome
remains failed. Missing and invalid records are tracked separately; this complete
run has neither. Both final rates use all 1,800 audited physical outcomes.

Timeout was the dominant recorded termination reason: 1,722 episodes. Another
30 ended on unexpected contact and 11 on non-target finger contact. A read-only
diagnostic of the first three red-task episodes in each spacing profile (nine
episodes total) found hovering, wandering or insufficient downward motion; it
did not find numeric bridge mismatches in that subset. The diagnostic is saved
at `outputs/table_buttons_policy_eval_20261024/diagnostics/first_nine_single_red.json`.
It describes those nine trajectories, not every failure, and does not establish
an underlying training cause.

A separate 30-second timeout diagnostic completed on 2026-10-25. It reused the
first two planned episodes of each of the 18 cells (36 rollouts, six distinct
layouts across the three spacing profiles), with identical checkpoints, scene
configurations, policy seeds and execution settings. Independent scoring of the
same long trajectories at 12/20/30 seconds yielded **0/36, 0/36 and 1/36** task
successes; all three strict collector counts were **0/36**. The added success was
`near_yellow`, episode 0, at 24.242 simulated seconds. At 30 seconds, 33 episodes
still timed out and two ended on non-target contact. All 36 trajectories and
recorded videos passed audit, and the dedicated inference services were stopped.
The original 1,800-episode evaluation is unchanged. This small diagnostic shows
one recoverable timeout, not a full-suite success-rate estimate. RTX reruns did
not reproduce identical image inputs, so the comparison uses cutoffs within each
long rollout rather than differences from the previous run. Details are in the
[timeout report](../outputs/table_buttons_timeout_probe_20261025/results/README.md).

A subsequent read-only training/data diagnosis rechecked all 900 successful
demonstrations and 104,410 exported numeric rows. No omitted press-success step
or numeric alignment error was found. It identified sparse contact/recovery
coverage, substantial held-out action errors, and a checkpoint-selection tradeoff
between arm and gripper errors. These findings do not establish that longer
training alone will restore closed-loop success. See the
[diagnosis and proposed controlled comparisons](../outputs/table_buttons_failure_diagnosis_20261025/README.md).

The original strict plan remains at
`outputs/table_buttons_policy_eval_20261024/plan.json`; the formal `run/plan.json`
records this parent and reuses its layouts without regeneration. The three
original `smoke_<spacing>_multitask_red` checks are preserved. The later
`smoke2_<spacing>_multitask_red/audit.json` files independently validate the dual
metrics: medium produced a physical task success but failed the strict metric,
while near and far timed out. These preflight episodes are outside the formal
1,800-episode run and are not a formal success-rate estimate.

`outputs/table_buttons_policy_eval_20261024/service_reports/summary.json` verifies
all twelve models against independent offline strict-reload outputs. All 33
single-sample HTTP comparisons are exactly equal; the simulator uses batch size
one. Separate batched checks report their numerical differences and are not used
to claim batch-size-independent equality.

## Completed joint6 experiments

The expanded shared-motion datasets were used for the following later runs.
Each color cell contains 100 evaluated episodes; a dash means that color was
not part of the single-task model's evaluation.

| Training dataset / model | Spacing | Red | Yellow | Blue |
|---|---|---:|---:|---:|
| 1,000 episodes / red single-task joint6 | Near | 3/100 | — | — |
| 1,000 episodes / red single-task joint6 | Medium | 6/100 | — | — |
| 1,000 episodes / red single-task joint6 | Far | 44/100 | — | — |
| 3,000 episodes / multi-task joint6 | Near | 4/100 | 3/100 | 4/100 |
| 3,000 episodes / multi-task joint6 | Medium | 8/100 | 5/100 | 5/100 |
| 3,000 episodes / multi-task joint6 | Far | 17/100 | 22/100 | 17/100 |

The red single-task total is **53/300 task successes**, with **46/300 strict
collector successes**. Multi-task totals are **85/900** and **78/900**,
respectively. All physical outcomes and videos passed the corresponding run
audits. Both experiments kept the 12-second timeout, seven-step execution,
fixed gripper and disabled extra smoothing. They reused the held-out baseline
layouts; joint7 comparisons also change the number of training demonstrations
and the predicted action dimension.

Local summaries are
`outputs/table_buttons_red_joint6_1000_eval_20261026/run/results/summary.json`
and
`outputs/table_buttons_multitask_joint6_3000_eval_20261028/run/results/summary.json`.
Training settings and archive locations are recorded in the
[single-task](table_buttons_joint6_1000_training.md) and
[multi-task](table_buttons_multitask_joint6_3000_training.md) guides.
These results do not evaluate the later color-specific `target_relative_v2`
demonstrations, whose new collection is reported
[separately](results/table_buttons_target_relative_20261030/README.md).
