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
The minimum button-centre separation remains 0.10 m, with a 16-pixel image
margin around the full housing. The three stages are:

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
