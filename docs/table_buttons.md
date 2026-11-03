# Piper tabletop color buttons

This independent task places red, yellow and blue spring-loaded buttons on the
pick-and-place workbench. It reuses the PAP table, Piper arm, fixed
camera registrations, materials and lighting, while removing the box, loose
objects and tape markers. The elevator and pick-and-place tasks remain available.
The wrist camera uses the elevator task's intrinsics and optical pose relative
to the wrist; PAP keeps its dataset-fitted camera registration.

The [configuration](../configs/piper_table_buttons.json) embeds episode 461's
initial joint positions and gripper width, so launching this scene does not
require the source dataset. It does require the existing Piper and camera assets
and the Isaac Sim 5.0 environment. Button dimensions are task design choices.

The default uses a shared motion pattern for all colors. An optional
[color-specific configuration](table_buttons_color_templates.md) uses direct,
forward-lifting and above-target approaches for red, yellow and blue. Its
[completed nine-group collection](results/table_buttons_target_relative_20261030/README.md)
contains 1,000 fresh episodes per spacing/color combination. Existing
shared-motion datasets are retained separately. Learned-policy execution and
its external inference dependency are documented in the
[evaluation guide](table_buttons_policy_eval.md).

## Wrist camera

The default configuration uses the same `attach_wrist_camera` setup as the
elevator task: 640 × 480 RGB, `fx = fy ≈ 625.2214 px`, principal point
`(320, 240)`, and a **54.2084° × 42°** field of view. The optical centre in the
`link6` frame is approximately **(−73.122, 3.494, 32.250) mm**, with the original
official-assembly optical orientation. No PAP-fitted `wrist_camera` override is
applied. The housing and bracket use the shared official assets.

The fixed third-person camera and overview retain the PAP configuration, so
the button sampling region is unchanged. Historical output configurations may
still contain the earlier PAP wrist registration; passing one explicitly to
`--config` preserves that recorded view. Use the default configuration for the
new elevator-aligned wrist view. The historical videos below predate this change.

The [updated seed-73 wrist preview](../outputs/piper_table_buttons_20261022/elevator_wrist/initial_wrist.jpg)
was rendered in Isaac. Comparing its saved USD camera with the frozen elevator
training snapshot gives identical intrinsics and a relative-pose matrix maximum
difference of `6.11e-8` (numerical serialization precision). The fixed camera,
sampling domain and seed-73 layout match the preceding run. See the
[alignment report](../outputs/piper_table_buttons_20261022/elevator_wrist/camera_alignment_validation.json).
The relevant existing tests passed **66/66**; this check validates camera setup,
not learned-policy performance.

## Run

For color-dependent routes that remain distinct at near spacing, see
[color-specific motion templates](table_buttons_color_templates.md). This is an
optional generator configuration; the shared route remains the default.

Run from the PressB root and choose a new output directory each time.

```bash
# Open the initialized scene and keep the interactive window open.
env -u PYTHONPATH .conda/envs/pressb/bin/python scripts/run_table_buttons.py \
  --mode snapshot --hold --gpu 1 --output outputs/table_buttons_gui

# Validate physical pressing and release for all three colors in sequence.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
env -u PYTHONPATH .conda/envs/pressb/bin/python scripts/run_table_buttons.py \
  --headless --gpu 1 --mode demo --target all \
  --output outputs/table_buttons_demo

# A single instruction selects one target; English instructions also work.
env -u PYTHONPATH .conda/envs/pressb/bin/python scripts/run_table_buttons.py \
  --headless --gpu 1 --mode demo --instruction '按下红色按键' \
  --output outputs/table_buttons_red

# Sample the camera-visible, executable press region and validate all colors.
env -u PYTHONPATH .conda/envs/pressb/bin/python scripts/run_table_buttons.py \
  --headless --gpu 1 --mode demo --target all --seed 73 \
  --output outputs/table_buttons_expanded_seed73

# Choose near / medium / far pairwise spacing for a reproducible random layout.
env -u PYTHONPATH .conda/envs/pressb/bin/python scripts/run_table_buttons.py \
  --headless --gpu 1 --mode snapshot --seed 73 --spacing near \
  --output outputs/table_buttons_near_seed73

# Map the region on a 1 cm diagnostic grid; this CPU check does not start Isaac.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.conda/envs/pressb/bin/python scripts/map_table_button_workspace.py \
  --step .01 --output outputs/table_buttons_workspace_map_01 \
  --image outputs/piper_table_buttons_20261022/workspace_expansion/random_seed73/initial_up.jpg
```

`--config` defaults to `configs/piper_table_buttons.json`. Omit `--seed` to use
the configured positions. `--target` accepts `red`, `yellow`, `blue` or `all`;
`all` runs a separate validation episode per color. `--instruction` accepts a
single-color command such as `Press the blue button.` or `按下蓝色按键` through
an explicit parser. A conflicting `--target` is rejected. This parser selects
the task; it is not a language-model inference call.

The default `position_sampling.domain` is `camera_reachable`. Both seeded and
configured layouts are checked before Isaac starts; demo mode additionally
checks the generated command path at the configured physics rate. The first
use builds and caches radial IK profiles in the current process, which adds
some CPU preparation time before the simulation window appears.

## Layout and physical success

Positions use metres in the robot base frame: +X points forward, +Y left and
Z=0 is the table surface.

| Button | Default centre XY (m) |
|---|---|
| Red | (0.33, 0.12) |
| Yellow | (0.38, 0.00) |
| Blue | (0.33, −0.12) |

Each colored cap has a 32 mm diameter inside a 44 mm diameter housing,
a neutral top height of 25 mm above the
table and 4 mm of downward travel. A dynamic cap moves on a PhysX prismatic
joint with a return spring. The gripper fingers press it through contact.

A valid press requires **at least 3 mm of measured cap depression and at least
0.1 N of measured finger contact force simultaneously for 25 ms**. Pressing the
target succeeds; a valid press on another color fails, including simultaneous
target/wrong-color presses. Release below 1 mm rearms the cap. The demo also
checks spring return after retreat. These conditions use measured physical
state, not the commanded end-effector position.

### Expanded placement

Seeded placement now proposes positions over the full fixed-camera/table
intersection and retains positions that admit the task's scripted press motion.
Without a spacing profile, the minimum button-centre separation remains
0.10 m with no additional upper distance limit. All profiles retain a 16-pixel
image margin around the full housing. The three stages are:

1. [`camera_center_domain`](../src/pressb/table_buttons/camera_domain.py)
   intersects the tabletop with the configured camera frustum, accounting for
   the housing radius, button height and image margin. With the current
   configuration, this polygon has area **0.21462185 m²** and approximate
   bounding intervals **X∈[0.0806, 0.5217] m**, **Y∈[−0.2994, 0.3046] m**.
   These intervals enclose the camera polygon; they are not an executable
   rectangular workspace or a promise that every enclosed point is visible.
2. [`PressWorkspace`](../src/pressb/table_buttons/workspace.py) tests actual URDF
   joint limits and a sampled hover-to-press path at each proposed position.
   It tries multiple gripper pitches and hover heights, retaining alternatives
   when the preferred approach fails. Radial profiles reuse the first joint's
   Z-axis symmetry to narrow the search; the proposed radius is still solved
   individually and its resulting poses checked by forward kinematics.
3. [`ReachablePlacement`](../src/pressb/table_buttons/placement.py) uses
   [`RobotClearance`](../src/pressb/table_buttons/collision.py) to screen the
   sampled initial-pose transit, approach, press and retreat against the finite
   tabletop, nonadjacent robot bodies, wrist-camera housing and bracket, and
   button geometry. After placing all three colors, it checks each selected
   path against the entire layout. Intended finger contact with the target cap
   is allowed; housing collisions and contacts with other caps are rejected.

### Near, medium and far spacing

Use `--spacing near`, `--spacing medium` or `--spacing far` together with
`--seed`. Each profile constrains **all three pairwise centre distances** in
the table plane, including the most distant pair:

| Profile | Centre-to-centre distance for every pair | Housing edge gap (44 mm diameter) |
|---|---|---|
| `near` (近) | 6–10 cm | 1.6–5.6 cm |
| `medium` (中) | 10–18 cm | 5.6–13.6 cm |
| `far` (远) | 18–30 cm | 13.6–25.6 cm |

The three button positions remain random; the sampler does not impose a rigid
triangle or a fixed color order on the table. The entire camera-visible,
reachable region remains available, subject to the selected distance band and
the same IK and collision checks. All three complete press paths must still
pass the layout check. Infeasible custom limits raise an error after the bounded
proposal budget; the sampler does not silently relax the band.

The equivalent JSON setting is `position_sampling.spacing_profile`, with
`null` preserving the original minimum-only sampling. Each profile's limits
can be adjusted in `position_sampling.spacing_profiles` using
`min_separation` and `max_separation` **in metres**. Both limits must be present
for a customized profile, and the lower bound must keep the housings disjoint.
A CLI `--spacing` overrides the JSON profile. Keep `--seed` to randomize;
without it, configured positions are replayed and checked against the configured
distance limits. The CLI rejects `--spacing` without `--seed` to avoid silently
changing a fixed-layout replay.

The saved `config.json` records the selected profile and actual positions.
For the default reachable-domain sampler, `sampling_domain.sampling` also
records the resolved limits and all three `pairwise_distances_m`. Reusing the
same seed, profile, configuration and implementation reproduces the layout.

The initial spacing validation sampled seeds 73, 0, 1, 2 and 3 for each profile
using the actual Piper URDF: all 15 layouts passed. For seed 73, the three
commanded press paths in each profile also passed table, robot and complete
button-layout collision checks at 120 Hz (937 samples per path, nine paths).
These are kinematic/geometric checks, not new physical rollout success rates.
The [local validation records](../outputs/table_buttons_spacing_20261023/sampling_validation.json)
include distances and proposal counts. The related automated suite passed
115 tests; one USD scene-construction test requires the Isaac runtime and was
skipped by the CPU test environment.

### Workspace and motion checks

The `press_workspace` configuration controls candidate pitches, hover heights,
press offset and IK profile resolution. Defaults try pitches
`[160, 150, 140, 130, 170]` degrees with hover heights `[0.12, 0.08]` m, preferring
the original 160-degree approach where it passes. A more distant or nearer
position may therefore use a different approach instead of being rejected by
one fixed pose. The press target is 6 mm below the neutral top; the physical
controller stops further descent from measured cap displacement.

These are **discrete geometric and motion checks**, followed by physical Isaac
validation for the recorded runs. They do not establish continuous collision
freedom or the maximum workspace of the whole robot under arbitrary motions.
The CPU collision model does not simulate servo tracking, contact offsets,
compliance, room geometry, table clamps or the camera tripod. Camera-frustum
visibility also does not account for temporary occlusion by the arm.

`map_table_button_workspace.py` writes `workspace_map.npz`, `report.json`,
`config.json`, and PNG/PDF views of the checked region. Its area is a grid
estimate for single-button paths, distinct from the exact camera-polygon area
above. It does not imply that arbitrary combinations of three feasible points
form a clear layout: actual sampling checks all three selected paths together.
The map is diagnostic; sampling evaluates the proposed coordinates directly
instead of accepting a point from a precomputed mask.

The [1 cm map](../outputs/piper_table_buttons_20261022/workspace_expansion/map_1cm/expanded_sampling_range.png)
checked **2,160 visible grid positions**, all of which admitted a screened
single-button approach. Normalizing the accepted grid fraction by the exact
visible polygon gives approximately **0.2146 m² (2,146 cm²), 8.52 times** the
old 252 cm² window. This is a sampled workspace estimate. The analytic camera
polygon and its enclosing rectangle remain distinct; points outside the
polygon are rejected. The [map report](../outputs/piper_table_buttons_20261022/workspace_expansion/map_1cm/report.json)
records the estimator, grid and collision-model details.

CPU workspace screening uses Pinocchio 2.7.0 and hpp-fcl 2.4.4, already available
in the current PressB environment. For another compatible environment, these
dependencies are declared by `pip install -e '.[table-buttons]'`. Before demo
execution, the runner also screens the actual planned commands interpolated at
the 120 Hz physics rate against the robot, table and complete button layout.

A seed is reproducible with the same configuration and implementation. Changing
from the old rectangle to this expanded domain changes the proposal sequence
and accepted positions, so the same seed does **not** reproduce an old layout.
To replay an existing layout, use that run's saved `config.json` and omit
`--seed`. Legacy configurations containing `position_sampling.bounds_xy` and
without `domain: camera_reachable` retain the original rectangle sampler when
an explicit seed is supplied.

## Collect separate LeRobot datasets

`scripts/collect_table_buttons.py` collects successful physical demonstrations
for the Cartesian product of spacing profiles and target colors. The default is
100 episodes in each of nine groups: `near_red`, `near_yellow`, `near_blue`,
`medium_red`, `medium_yellow`, `medium_blue`, `far_red`, `far_yellow`, `far_blue`.
Every scene contains all three colors. A color's instruction is identical in
all spacing groups: `Press the red button.`, `Press the yellow button.` or
`Press the blue button.`

```bash
# Collect 900 successful episodes; recycle Isaac after each 50 to release caches.
# Exit 75 requests a restart; other errors stop this loop for diagnosis.
while true; do
  collection_exit=0
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  env -u PYTHONPATH .conda/envs/pressb/bin/python scripts/collect_table_buttons.py \
    --output outputs/table_buttons_collection --gpu 1 --episodes-per-group 100 \
    --max-new-episodes 50 --resume || collection_exit=$?
  if [ "$collection_exit" -eq 0 ]; then break; fi
  if [ "$collection_exit" -ne 75 ]; then exit "$collection_exit"; fi
done

# Install the separate CPU export environment once; leave Isaac's packages intact.
bash scripts/setup_dataset_env.sh

# Export one group. Repeat for the other eight group names.
env -u PYTHONPATH .conda/envs/lerobot/bin/python scripts/export_table_buttons.py \
  --raw outputs/table_buttons_collection/near_red \
  --output /path/to/datasets/near_red --episodes 100

# Audit all raw captures and their corresponding nine exported datasets.
env -u PYTHONPATH .conda/envs/lerobot/bin/python scripts/audit_table_buttons_dataset.py \
  --raw-root outputs/table_buttons_collection --expected-per-group 100 \
  --lerobot-root /path/to/datasets
```

Use `--profiles near` or `--targets red` to collect a subset. Independent
collectors must use separate output roots. `--resume` continues an identical
saved collection plan; accepted episodes are retained. The exporter can wait
for a group's full quota with `--watch`. Exported directories become visible
only after validation completes. Neither script uploads data to the Hub.
Long collections should use the restart loop above: Isaac retains CPU stage
caches across episodes even when GPU memory stays stable. `--max-new-episodes`
limits newly accepted episodes per process without changing the dataset plan.
Exit 75 with `restart_required` means the saved dataset should be resumed; it
does not mean the requested quota is complete.

The controller uses bounded IK and quintic motion with commands held at 30 Hz,
while physics runs at 120 Hz. Each trajectory starts with the initial observation
at t=0 and ends at the first 30 Hz boundary after a physically measured target
press. Retraction is excluded. Failed attempts, wrong-color contacts above
0.02 N and unexpected robot contacts are excluded from the successful quota and
retained separately. Infrastructure errors abort the worker for diagnosis.
Layout seeds, all button positions, the chosen approach, contact forces and
complete physical samples remain in the raw records.

The exported **LeRobot v3.0** interface matches PAP's joint-space convention:

| Feature | Content |
|---|---|
| `observation.images.up`, `observation.images.wrist` | 640 × 480 RGB video at 30 Hz |
| `observation.state` | Six measured joint angles in degrees, full gripper opening in cm |
| `action` | Six commanded joint targets in degrees, full gripper opening in cm |
| Task | One fixed color instruction per dataset |

An action applies during the following 1/30-second interval; the terminal action
repeats the preceding target. Raw joint values remain in radians and metres.
The official LeRobot writer and reader validate all numeric rows, fully decode
the videos, and compare first/middle/last camera frames in every episode against
the source captures. The independent audit also reconstructs the physical
success condition and checks action/state alignment at every physics tick.
These are scripted demonstrations for imitation learning, not learned-policy
evaluation results.

### Completed collection — 2026-10-23

The nine groups were collected and exported to
`/home/pengguanqi/Datasets/piper_table_buttons_20261023`: **100 successful episodes
per group, 900 total, 104,410 observation/action frames** (the same count for
each RGB stream). Each group contains 100 distinct button layouts. Instructions
are identical across profiles for each color. Observed durations are
3.8333–3.9 seconds, ending at measured success as specified above.

All 900 physical traces passed independent checks, and all nine datasets passed
official LeRobot writer/reader validation. The formal collection accepted 891
new episodes plus nine previously audited pilot episodes; no completed physical
attempt failed. Pre-capture process interruptions were resumed at the same seed
and did not contribute incomplete or duplicate episodes. Long-running CPU cache
growth was handled by rotating each worker after 50 new accepted episodes.

The [collection summary](../outputs/table_buttons_dataset_20261023/collection_summary.json)
lists all dataset directories and counts. The
[independent audit](../outputs/table_buttons_dataset_20261023/raw/audit.json)
contains per-episode physical, timing and video checks. Raw captures, layouts,
process logs and source snapshots remain under
`outputs/table_buttons_dataset_20261023`. Data and full recordings are not
committed to Git.

## Expanded multi-task datasets — 2026-10-27

The expanded single-task datasets are merged by spacing profile under
`/home/pengguanqi/Datasets/piper_table_buttons_multitask_3000_20261027`:

| Dataset | Red episodes | Yellow episodes | Blue episodes | Total episodes | Frames |
| --- | ---: | ---: | ---: | ---: | ---: |
| `near_multitask` | 1,000 | 1,000 | 1,000 | 3,000 | 348,008 |
| `medium_multitask` | 1,000 | 1,000 | 1,000 | 3,000 | 348,016 |
| `far_multitask` | 1,000 | 1,000 | 1,000 | 3,000 | 348,036 |

Each dataset uses the three instructions `Press the red button.`,
`Press the yellow button.`, and `Press the blue button.`; instructions are identical
across spacing profiles. Episode IDs are red 0–999, yellow 1000–1999, and blue
2000–2999. The original nine 1,000-episode datasets and the older 100/300-episode
versions remain unchanged. This merge preserves the original joint7 data,
30 Hz timing and both RGB video streams, with no video re-encoding.

Each profile contains 1,000 layouts shared across the three tasks. The manifest
records source episode IDs, actual layout positions and `layout_group_id`; keep
all tasks of a layout together when making a training/validation split.
Numeric statistics are recomputed after remapping global frame, episode and task
indices. Each dataset's `meta/audit.json` records full numeric comparisons,
video-copy checks and official-reader samples; the root `merge_summary.json`
reports all three groups. Run logs and the final-path reader audit are under
`outputs/table_buttons_multitask_3000_20261027`.

```bash
.conda/envs/lerobot/bin/python scripts/merge_table_buttons_datasets.py \
  --source /home/pengguanqi/Datasets/piper_table_buttons_1000_20261025 \
  --destination /home/pengguanqi/Datasets/piper_table_buttons_multitask_3000_20261027 \
  --expected-episodes-per-color 1000
```

The destination must be new; the merger refuses to overwrite a published dataset.

## Multi-task datasets — 2026-10-23

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

## Outputs and scope

Expanded-domain physical validation covered **10 scripted presses** across a
random layout and near/side/far boundary layouts. Every target press and spring
return passed, with no recorded unintended contacts. This includes a near-base
position `(0.10, 0.00)` m using a 170-degree pitch / 8 cm hover and the far corner
`(0.52, 0.20)` m using a 150-degree pitch / 12 cm hover. See the
[validation summary](../outputs/piper_table_buttons_20261022/workspace_expansion/validation_summary.json)
and [expanded random-layout video](../outputs/piper_table_buttons_20261022/workspace_expansion/random_seed73/table_buttons_demo.mp4).
The relevant automated suite passed **154 tests**. These physical spot checks
do not substitute for exhaustive testing of all continuous layouts.

Snapshot mode writes the effective `config.json`, `scene.usda`, `report.json`,
`status.json` and `initial_up.jpg`, `initial_wrist.jpg`, `initial_overview.jpg`.
The expanded configuration also records the selected per-color
`press_approaches` and `sampling_domain`, including placement diagnostics.
Demo mode additionally records the three corresponding MP4 streams,
`physics.npz`, `episodes.json`, finger contacts, unexpected contacts and final
images. Videos use the configured 30 Hz rate; physics runs at 120 Hz.

The runner uses scripted bounded IK to check the scene's reachability, contact
and spring response. It does not evaluate a trained VLA or RL policy. This task
has not yet been connected to the online simulation RPC service or trained with
the three-node RL framework. The existing elevator results do not describe this
new task.

## Validation — 2026-10-22

### Expanded camera/press region

The [expanded seed-73 layout](../outputs/piper_table_buttons_20261022/workspace_expansion/random_seed73/initial_up.jpg)
passed **3/3 physical press-and-return checks** in Isaac, with no wrong-button
presses or recorded unintended table/housing contacts. Its positions differ
from the earlier rectangle-based seed-73 example below. See the
[physical report](../outputs/piper_table_buttons_20261022/workspace_expansion/random_seed73/report.json),
[per-color outcomes](../outputs/piper_table_buttons_20261022/workspace_expansion/random_seed73/episodes.json)
and [fixed-camera execution](../outputs/piper_table_buttons_20261022/workspace_expansion/random_seed73/up.mp4).
This validates that sampled layout; it is not a success-rate estimate over the
entire expanded region.

### Earlier rectangle-based layouts

The earlier sampler used **X∈[0.30, 0.39] m and Y∈[−0.14, 0.14] m**: a
9 × 28 cm rectangle with area 0.0252 m². The runs in this section predate the
expanded-domain sampler and remain useful historical scene checks.

The current 32 mm caps / 44 mm housings were reduced from the initial
48 mm / 64 mm design. All three colors passed physical press/return checks
again, with no wrong-button presses or unintended moving-arm contacts;
the 32 tabletop-task tests passed. See the updated
[physical report](../outputs/piper_table_buttons_20261022/smaller_buttons/report.json),
[fixed-camera preview](../outputs/piper_table_buttons_20261022/smaller_buttons/initial_up.jpg)
and [overview](../outputs/piper_table_buttons_20261022/smaller_buttons/initial_overview.jpg).
The updated [fixed/wrist execution video](../outputs/piper_table_buttons_20261022/smaller_buttons/table_buttons_demo.mp4)
shows red, yellow and blue presses at original speed (23.4 s, 702 frames at 30 fps).

A [randomized layout with seed 73](../outputs/piper_table_buttons_20261022/smaller_buttons_seed73/initial_up.jpg)
also passed all three physical press/return checks using the current smaller
buttons. The minimum center spacing was 10.28 cm, and every housing remained
within the fixed camera image. See its
[report](../outputs/piper_table_buttons_20261022/smaller_buttons_seed73/report.json)
and [original-speed execution video](../outputs/piper_table_buttons_20261022/smaller_buttons_seed73/table_buttons_demo.mp4).
Replay these exact saved positions with
`--config outputs/piper_table_buttons_20261022/smaller_buttons_seed73/config.json --mode demo --target all`
and a new output directory, omitting `--seed`.

The original 48 mm-cap default-layout demonstration physically pressed and released each
of the red, yellow and blue buttons, with no wrong-button presses or unintended
moving-arm contacts with the table or button housings. See the
[physical report](../outputs/piper_table_buttons_20261022/final/report.json),
[initial overview](../outputs/piper_table_buttons_20261022/final/initial_overview.jpg)
and [fixed-camera view](../outputs/piper_table_buttons_20261022/final/initial_up.jpg).
The [side-by-side fixed/wrist video](../outputs/piper_table_buttons_20261022/final/table_buttons_demo.mp4)
contains 702 frames at 30 fps (23.4 s), preserving simulation playback speed.

Peak finger forces were approximately **35–47 N**, and the caps reached their
full 4 mm travel. The scripted controller can reach the lower mechanical stop;
this demonstration has not optimized contact force.

A separate [seed-42 run](../outputs/piper_table_buttons_20261022/random_seed42/report.json)
used `请按下蓝色按键。`, kept all button housings within the fixed camera view,
and passed the blue-button press/return check without unintended contacts.
The relevant automated checks passed **98 tests**. These are scene and scripted
controller checks, not learned-policy success-rate estimates. Generated artifacts
are local outputs and are not distributed in Git.
