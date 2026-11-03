# Color-specific tabletop-button demonstrations

The [color template configuration](../configs/piper_table_buttons_color_templates.json)
now uses **`target_relative_v2`**. Each color advances toward its own target;
height profiles distinguish the routes. The default `piper_table_buttons.json`
continues to use shared motion. Existing collected datasets are unchanged.

## Target-relative paths

All positions use the robot base frame and the task's link6 +Z 0.1358 m
planning reference point. The common initial tip height is approximately
16.34 cm. The routes end at the original near-contact pose, 1.8 cm above the
button top, before the existing contact-limited press starts.

| Color | Cartesian approach |
|---|---|
| Red | Straight line from the initial tip position to the near-contact pose. |
| Yellow | Advance 40% of the horizontal distance while rising 6 cm above the initial height; then follow a straight line to the near-contact pose. |
| Blue | Move directly above the target at 18 cm height (or the initial height if higher); then descend vertically to the near-contact pose. |

The horizontal projection remains on the initial-to-target line. There are no
fixed left/right base-frame waypoints. Each Cartesian leg uses quintic timing.
Radial yaw follows Cartesian position while local tool orientation transitions
to the selected pressing orientation; bounded IK provides joint targets.

The prefix lasts 3.6 seconds for each color, after the common initial 0.4-second
hold. Yellow uses two equal-duration legs; blue uses 2.4 seconds to reach the
overhead point and 1.2 seconds to descend. Press starts at 4.0 seconds, release
at 5.35 seconds, and retreat reaches the original hover at 6.45 seconds. Full
demos include return home and finish at 8.7 seconds. Collection still stops at
the first 30 Hz observation after physical success (4.733 seconds in the smoke
test). The gripper remains fixed.

Configuration options under `motion_templates`:

| Option | Current value | Meaning |
|---|---:|---|
| `mode` | `target_relative_v2` | Select these target-oriented routes. |
| `transit_duration_s` | 3.6 | Initial-to-near prefix duration, excluding initial hold. |
| `mid_lift_m` | 0.06 | Yellow midpoint lift above the higher endpoint. |
| `overhead_z_m` | 0.18 | Blue overhead height floor in the base frame. |
| `min_transit_separation_m` | 0.03 | Required mean synchronous tip distance for each color pair. |

The separation criterion is an average across the prefix, not a minimum at
every instant. It has been reduced from the historical detour configuration's
10 cm so that route selection respects the requested forward movement.
The generator also checks joint speeds, sampled collisions and FK direction
between command knots. Rejection thresholds are 1 mm lateral deviation or
overshoot, and 0.5 mm backward movement per interpolated sample. IK/clearance
failures reject candidates; the generator does not switch to the old detours.
New layouts carry their color, target, exact joint prefix and phase metadata.
A cached layout from another mode or configuration is rejected.

## Collect

From the PressB root, choose an available GPU and a fresh output directory:

```bash
env -u PYTHONPATH OMP_NUM_THREADS=8 .conda/envs/pressb/bin/python \
  scripts/collect_table_buttons.py \
  --config configs/piper_table_buttons_color_templates.json \
  --output outputs/table_buttons_target_routes_new \
  --profiles near --targets red yellow blue --episodes-per-group 3 --gpu 0
```

Use `--profiles near medium far` for all spacing configurations. Large
collections can use the [collection supervisor](../scripts/run_table_buttons_collection_suite.py)
to collect all nine groups, rotate Isaac processes, export completed groups and
run the final audit:

```bash
python3 scripts/run_table_buttons_collection_suite.py \
  --config configs/piper_table_buttons_color_templates.json \
  --output outputs/table_buttons_target_relative_new \
  --dataset-root /path/to/datasets/piper_table_buttons_target_relative_new \
  --episodes-per-group 1000 --reuse-per-group 0 \
  --profile-gpus 0 1 1 --max-new-episodes 50 --run
```

Use a new output and dataset directory. Omitting `--run` prepares the frozen
plan without launching collection. `--profile-gpus` maps near/medium/far to
physical GPU indices; this run used three collectors on two RTX 5090 GPUs.
Each collector rotates after 50 accepted episodes, and at most two CPU exports
run concurrently. Repeating the same supervisor command resumes incomplete
work against the same frozen configuration. `suite_status.json` records live
progress; `completion_summary.json` and `audit.json` record final completion.

For a manually managed collector, exit 75 means resume the identical command
with `--resume`. Each raw group uses the existing [LeRobot exporter](../scripts/export_table_buttons.py),
action units and camera schema. Language remains `Press the red/yellow/blue
button.` across profiles. Physical retries are per color; the supervisor's
`pairing_report.json` compares actual seeds and positions across colors.

To render all three colors from a frozen layout and record the full external
view, use `scripts/run_table_buttons.py --config <saved-episode/config.json>
--mode demo --target all --gpu 0 --headless --output <new-directory>`, without
`--seed`. The saved `overview.mp4` includes approach, pressing and return.
The visualization demo interpolates at 120 Hz; collection holds targets at
30 Hz. They are separately validated executions, not frame-identical replays.

## Completed collection — 2026-11-01

The full run completed at **05:22:49 Asia/Shanghai on 2026-11-01**:
**nine fresh groups of 1,000 episodes, 9,000 total**, containing **1,286,591
observation/action rows**. All groups passed physical validation, LeRobot export
and the independent audit, including full decoding of 18,000 camera videos.
All 3,000 profile/episode indices retain matching layouts across the three
colors, with zero mismatches. The nine-group smoke is separate from these
counts. No old episodes were reused.

The [committed collection report](results/table_buttons_target_relative_20261030/README.md)
and [compact JSON summary](results/table_buttons_target_relative_20261030/summary.json)
contain per-group counts and source provenance. Full datasets remain local at
`/home/pengguanqi/Datasets/piper_table_buttons_target_relative_1000_20261030`.
These are physical scripted demonstrations; they do not measure a learned
policy's task success rate. New model training with these routes has not been
performed.

## Earlier revision validation — 2026-10-30

The following measurements describe the smaller development checks before the
full collection. They are not full-dataset path statistics.

- **19/20 predetermined near layouts (seeds 0–19) passed all three new routes**
  using their original pressing approaches. Seed 9 was rejected because the
  yellow lift is infeasible near the base; attempting alternative pressing
  orientations also failed. This is a reduced feasible subset for this route
  family, not a claim that every previously valid layout remains valid.
- All 57 accepted routes passed dense arm/table/button preflight. Their maximum
  horizontal corridor error was 0.040 mm; maximum backward interpolation step
  was 0.021 mm. Mean pairwise pre-press tip distance was 7.95 cm, with minimum
  per-layout color-pair mean 4.68 cm. Rejected layouts are excluded from these
  path statistics and retained in the report.
- Newly sampled medium and far seed-0 layouts also passed all three dense
  preflights; these two layouts have not been physically executed in this test.
- Near seeds 0–2, all three colors: **9/9 first-attempt physical presses passed**.
  An independent audit reconstructed success, contact and action/state timing,
  and fully decoded all 18 videos. Actual FK lateral error was at most
  **0.247 mm**, and maximum backward physical step was **0.022 mm**. The mean
  pairwise prefix distance was **7.80 cm**. These are generator checks, not
  learned-policy success rates.
- The full third-person seed-0 demonstrations all passed press, release and
  return checks. Every separate video has 261 frames at 30 fps, duration 8.7 s.
- Related CPU tests: **160 passed, 1 dependency-dependent test skipped**.

The detailed development report, scripts and videos remain local at
`outputs/table_buttons_target_relative_20261030/README.md`; they are not included
in a clean clone.

## Historical fixed-waypoint mode

`color_detours_v1` is retained only to reproduce the earlier left/right/high
waypoint experiment. Its measured 4.18× moving-stage separation and 10.8-second
full videos describe that older route, not `target_relative_v2`.
To reproduce it, use the saved configuration in
`outputs/table_buttons_color_templates_20261030/physical/near_red/episode_000000/config.json`.
The historical report is local at
`outputs/table_buttons_color_templates_20261030/README.md`.
