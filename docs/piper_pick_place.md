# Piper tabletop pick-and-place reconstruction

This independent Isaac Sim task reconstructs the real `piper_pap_20260606_offset`
setup: a black Piper at the edge of a green workbench grasps a tabletop object and drops
it into an open cardboard box. The elevator scene, snapshots, controller settings,
and trained policy interfaces are preserved.

The initial reconstruction uses the supplied photographs and 500 real episodes
(162,953 frames, 30 Hz; `up` and `wrist` RGB, both 640×480). The instruction is
`Grab the stuff into the box.` No VLA or RL training is included in this task yet.

The [published 200-episode Wall-X evaluation](results/pap_rectangle_20261023/README.md)
contains the corrected rectangle results, plots and per-episode CSV. A
[smoothed execution video](media/pap-wallx-smoothed.mp4) is included in Git.
Other links to `outputs/` below refer to local reconstruction evidence and are
not downloadable from a clean clone. Dataset/model paths are workspace examples;
use your local copies when reproducing them.

## Run

Use the existing Isaac Sim 5.0 environment and downloaded Piper/wrist assets.
Install the PAP client and dataset-reading dependencies into the intended Python
environment with `python -m pip install -e '.[pap]'` (PyAV, PyArrow, WebSocket and
MessagePack). Older workspace runs borrowed the Arrow wheel from a separate
environment; a fresh setup should install these dependencies explicitly.

```bash
env -u PYTHONPATH .conda/envs/pressb/bin/python scripts/analyze_pap_dataset.py \
  --dataset /home/pengguanqi/Datasets/piper_pap_20260606_offset \
  --output outputs/pap_dataset_audit --episodes 0 10 100

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
env -u PYTHONPATH .conda/envs/pressb/bin/python scripts/run_pap_scene.py \
  --config configs/piper_pick_place.json \
  --trajectory outputs/pap_dataset_audit/episode_000000.npz \
  --headless --gpu 1 --mode replay --duration 10.5 \
  --output outputs/pap_replay

# Inspect the initialized scene interactively; omit --headless.
env -u PYTHONPATH .conda/envs/pressb/bin/python scripts/run_pap_scene.py \
  --trajectory outputs/pap_dataset_audit/episode_000000.npz \
  --gpu 1 --mode snapshot --hold --output outputs/pap_gui

env -u PYTHONPATH .conda/envs/pressb/bin/python scripts/compose_pap_comparison.py \
  --dataset /home/pengguanqi/Datasets/piper_pap_20260606_offset --episode 0 \
  --simulation outputs/pap_replay --output outputs/pap_comparison
```

The runner requires a new output directory. It writes `scene.usda`, initial RGB
views, three videos, measured `physics.npz`, finger contact events, and a report.
The overview camera is for inspection; `up` and `wrist` reproduce the dataset views.
For another source episode, adjust the cube position in a separate configuration
to match that episode before replay. Merely changing the trajectory does not
automatically relocate the cube.

### Episode-specific objects and positions

The dataset contains red, blue and green cubes and yellow/pink irregular toys.
Their initial positions and orientations vary. Object identity/pose labels are
not supplied by the dataset; each replay needs a matching scene configuration.
The initial red cube is one example, not an assumption about all episodes.

`configs/piper_pick_place_episode_461.json` reproduces episode 461's **blue cube**.
Its single estimated edge is 27.846 mm, base-frame XY is (0.40550, −0.04323) m,
and yaw is −2.30°. The user's cube-shape clarification is enforced exactly;
the approximate 30 mm edge remains a weak prior. Initial third-person and wrist
images constrain the edge and pose while camera settings remain fixed. These
parameters are selected from images before physical replay, without using its
success result. The `object` entry supports strict cubes (`shape: "cube"`, `edge`),
cuboids (`size`) and upright cylinders (`radius`, `height`) with episode-specific
colour, mass, initial position and yaw. Conflicting dimensions on a cube are
rejected. The original `cube` configuration remains supported. Irregular toys
require suitable geometry; they have not been validated by this cube replay.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
env -u PYTHONPATH .conda/envs/pressb/bin/python scripts/run_pap_scene.py \
  --config configs/piper_pick_place_episode_461.json \
  --headless --gpu 1 --mode replay --duration 10 \
  --output outputs/pap_episode461
```

This preset selects its own `episode_000461.npz` when `--trajectory` is omitted;
an explicit trajectory must match the configured episode. The duration caps at
the recorded endpoint: all 290 source frames at 30 Hz, through t=9.6333 s.
Episode 461 physically lifts and releases the blue cube into the box, reaching
the stable placement condition at 7.1083 s. This is a recorded-trajectory replay,
not a learned-policy evaluation. Source-arm observations drive joint targets;
the gripper uses recorded commands, and the object moves freely under contact.

The first replay's artifacts are under `outputs/piper_pap_episode_replay_20261022/`:
`dataset_scan/` contains the 500 initial-frame survey, `geometry_fit/` records
the selected image fit, `replay_461_final/` contains the physical replay, and
`comparison_461/real_vs_sim.mp4` shows both real and simulated camera streams.
That archived run used an unconstrained rectangular block. The corrected cube
and box run is under `outputs/piper_pap_geometry_revision_20261022/` (see below).

## Wall-X policy inference

The local Wall-X checkpoint at `/home/pengguanqi/Models/piper_pap_20260606`
can use a separate `.conda/envs/wallx-inference` Conda environment. The
[installer](https://github.com/pgq18/wall-x/blob/base/inference/install_rtx5090.sh) installs the inference
dependencies and compiles the repository's CUDA extension for the selected
RTX 5090; it does not modify the Isaac or VLA-JEPA environments. See the
[Wall-X deployment notes](https://github.com/pgq18/wall-x/blob/base/inference/README_rtx5090.md).
Clone Wall-X separately at `ref/wall-x` and use the commit recorded in
[`repos.lock.json`](../repos.lock.json); PressB does not embed that repository.

```bash
# One-time inference setup (CUDA 12.8 toolkit required for the extension).
bash ref/wall-x/inference/install_rtx5090.sh

# Keep this server running on GPU 0 in a separate terminal.
bash ref/wall-x/inference/launch_server_piper_local.sh

# Model-controlled red-cube rollout: Isaac on GPU 1, 12 simulation seconds.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
env -u PYTHONPATH .conda/envs/pressb/bin/python scripts/run_pap_scene.py \
  --config configs/piper_pick_place.json --mode policy \
  --trajectory outputs/pap_dataset_audit/episode_000000.npz \
  --policy-endpoint ws://127.0.0.1:8094 --policy-execute-steps 32 \
  --headless --gpu 1 --duration 12 --output outputs/pap_wallx_red

# Use configs/piper_pick_place_episode_461.json for the blue-cube layout,
# with a separate output directory.
```

The simulator sends its current `up`/`wrist` RGB as `face_view`/
`left_wrist_view`, the literal instruction `Grab the stuff into the box.`,
`dataset_names=["lerobot/piper"]`, and measured joint/gripper state. This is the
native Wall-X WebSocket protocol, not the VLA-JEPA split-inference RPC.
Wall-X predicts 32 absolute targets: six joint angles in degrees plus total
gripper opening in centimetres. The simulator converts these to radians/metres
and applies one row every 1/30 simulation second, four physics ticks per row.
State is normalized once by the server; returned actions are already
unnormalized. Images retain the model's own aspect-preserving preprocessing.

Policy mode reads the recorded episode only for initialization; subsequent
commands come from the model and observations from the running simulator.
Inference waits pause simulation time. `--policy-execute-steps` sets the number
of predicted rows applied before observing and replanning; 32 executes the full
chunk. The existing camera calibration, objects, materials, contact physics,
snapshot mode and expert replay remain available.

### Optional execution smoothing

The execution layer includes the shared
[`JointActionPostprocessor`](../src/pressb/action_postprocessing.py), reusable
by tasks with six absolute arm-joint targets and one full gripper-width target.
PAP policy mode supports both task configuration and command-line overrides:

```json
"policy_execution": {
  "smoothing": {"mode": "none", "alpha": 0.8}
}
```

Append one of the following options to the policy rollout command above:

```bash
# Explicitly disable smoothing (the default).
--policy-smoothing none

# Enable arm-joint EMA, retaining 80% of each new target.
--policy-smoothing joint_ema --policy-smoothing-alpha 0.8
```

Command-line values override the corresponding configuration fields. Older
configs without `policy_execution` retain disabled smoothing. The selected
settings are recorded in `report.json` under `policy.smoothing`.

The EMA applies once per **30 Hz action**, after unit conversion and joint-limit
validation: `sent = alpha * raw + (1-alpha) * previous_sent` for the six arm
joints. Its initial history is the settled measured arm pose; it persists
across chunk boundaries and resets only for a new rollout. The gripper keeps
its original, limit-checked target. Each resulting target is held for four
120 Hz physics ticks. It must not be filtered again at every physics tick.

`alpha` must be in `(0, 1]`; lower values give stronger smoothing and more lag.
`none` and `alpha=1` preserve the original targets exactly. There is no extra
interpolation, speed limiter or model-side change. `physics.npz` stores both
`q_target_raw` (after unit conversion/limits, before smoothing) and `q_target`
(actually sent), while the original model responses remain in
`policy_requests/`. Smoothing can change grasp/contact timing, so its task
success must be checked separately. The historical demonstrations below used
`none`.

On 2026-10-23, `joint_ema` with `alpha=0.8` completed both the red-cube and
blue-cube layouts, reaching stable placement at 7.4333 s and 7.1417 s of
simulation time. The red run selected smoothing through the CLI; the blue run
selected it through the task configuration. Each executed 360 actions over
12 seconds. The recorded targets match the EMA recurrence at every control
step, including chunk boundaries, and preserve both finger targets exactly.
These are two functional checks, not a randomized success-rate measurement or
a controlled comparison of smoothing quality. See the
[smoothing validation summary](../outputs/wallx_pap_smoothing_20261023/summary.json).

### Randomized policy evaluation

This is the fixed 3 cm cube / 25 g / episode-0 experiment preset, not a generic
evaluator for every reconstructed object. Use `configs/piper_pick_place.json`
and the episode-0 initialization NPZ prepared above; an episode-461 scene has
different geometry and is rejected by this planner. The protocol records the
checkpoint and normalization paths used for this experiment. Other deployments
must supply their own model files and preserve matching inference settings.

`eval_pap_policy.py` prepares two groups with paired random positions: fixed red
cubes, followed by randomly ordered red/green/blue/white/yellow cubes (20 of each
color in a 100-episode group). Each green tape marker identifies a rectangle
edge: its long-axis line through the tape centre forms one boundary. The four
line intersections define the rectangle corners; the tape centres are not
corners. Positions are uniform by area inside this rectangle, inset by the cube
footprint so the entire cube stays inside. There is no additional reachability rejection. The
current scene uses a 3 cm cube at fixed yaw, with unchanged mass, friction,
lighting, box, cameras and initial robot pose. Blue uses the existing dataset
reconstruction's blue/cyan material color; all palette values are saved in the
plan as linear RGB.

For the current scene, the outer rectangle is X = 22.70–49.72 cm and
Y = −14.64–7.67 cm in the robot base frame (27.02 × 22.31 cm). For a 3 cm
axis-aligned cube, the centre range is X = 24.20–48.22 cm and
Y = −13.14–6.17 cm (24.02 × 19.31 cm). Version 2 plans record the four edge
lines and the footprint-inset region. This centre area is 2.77 times the
incorrect marker-centre hull used in the first evaluation.

![Corrected placement boundary](../outputs/wallx_pap_rectangle_correction_20261023/corrected_sampling_region.png)

```bash
.conda/envs/pressb/bin/python scripts/eval_pap_policy.py prepare \
  --config configs/piper_pick_place.json \
  --output outputs/wallx_pap_eval_rectangle_20261023 \
  --seed 20261023 --episodes-per-group 100

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 env -u PYTHONPATH \
  .conda/envs/pressb/bin/python scripts/run_pap_scene.py \
  --mode policy --headless --gpu 1 \
  --config outputs/wallx_pap_eval_rectangle_20261023/effective_config.json \
  --evaluation-plan outputs/wallx_pap_eval_rectangle_20261023/plan.json \
  --trajectory outputs/pap_dataset_audit/episode_000000.npz \
  --policy-endpoint ws://127.0.0.1:8094 --policy-execute-steps 32 \
  --output outputs/wallx_pap_eval_rectangle_20261023/run

.conda/envs/pressb/bin/python scripts/eval_pap_policy.py summarize \
  --output outputs/wallx_pap_eval_rectangle_20261023
```

Preparation selects joint EMA at `alpha=0.8`. A persistent simulator resets
joint/object poses and velocities, action buffers, EMA history, contact counts
and outcome state between episodes. Control and observation rendering stay at
30 Hz, with 120 Hz physics and the existing camera warmup. Each episode ends
after physical success or 20 simulated seconds. Success requires at least 4 cm
of object-bottom lift, full containment in the box, release, resting on the box
floor and low velocity for 0.5 s, plus contact evidence from both fingers.
Timeouts and invalid model actions count as failures. Network or simulator
errors interrupt evaluation and are reported separately.

The runner writes `run/progress.json`, `run/results.jsonl` and full per-episode
reports/trajectories; seven prespecified episodes also record videos. Summaries
can be regenerated during execution and show completed versus planned counts,
per-color results, Wilson 95% intervals and spatial outcomes. `--resume` resumes
an identical plan/configuration, preserving completed successes and failures.
Placement and color seeds are fixed; the native diffusion sampler is stochastic
and does not use paired noise across the two groups.

With complete local recordings, run the offline analyses without restarting
Isaac or the model:

```bash
.conda/envs/pressb/bin/python scripts/audit_pap_eval.py \
  --root outputs/wallx_pap_eval_rectangle_20261023
.conda/envs/pressb/bin/python scripts/analyze_pap_eval_failures.py \
  --root outputs/wallx_pap_eval_rectangle_20261023
.conda/envs/pressb/bin/python scripts/analyze_pap_eval_locations.py \
  --root outputs/wallx_pap_eval_rectangle_20261023
.conda/envs/pressb/bin/python scripts/plot_pap_eval.py \
  --summary outputs/wallx_pap_eval_rectangle_20261023/summary.json
```

#### Corrected rectangle evaluation — 2026-10-23

The corrected run completed both groups (200 episodes total) in 40.34 minutes,
using the rectangle formed by the four tape-edge lines. All failures reached
the 20-second limit; no infrastructure errors or invalid model actions occurred.
Recomputing outcomes from all 200 saved physics/contact records agreed with the
reports. The command audit also verified initial placements, native model
responses, EMA recurrence, control timing and per-episode resets.

| Evaluation | Success | Rate | Wilson 95% interval |
|---|---:|---:|---:|
| Random position, fixed red | 65/100 | 65% | 55.3–73.6% |
| Random position and five balanced colors | 54/100 | 54% | 44.3–63.4% |

| Color in second group | Success | Rate |
|---|---:|---:|
| Red | 13/20 | 65% |
| Green | 9/20 | 45% |
| Blue | 12/20 | 60% |
| White | 8/20 | 40% |
| Yellow | 12/20 | 60% |

For each color's same 20 positions, the fixed-red baseline versus second-group
success counts were red 12→13, green 15→9, blue 12→12, white 11→8 and yellow
15→12. Green, white and yellow subsets had lower counts in this run. With 20
trials per color and unpaired diffusion draws, the aggregate difference should
not be attributed entirely to color.

In the fixed-red group, dividing the valid centre rectangle into three equal
bands along each axis gives the highest failure rates in the middle-left and
lower-left cells of the fixed camera view: 9/13 (69.2%) each. Across the three
left-to-right bands, failure rates are 20/33 (60.6%), 12/41 (29.3%) and 3/26
(11.5%). Across the top-to-bottom bands, they are 5/27 (18.5%), 12/37 (32.4%)
and 18/36 (50.0%). These are descriptive spatial associations from one trial per
position, not a diagnosis of the physical failure mechanism. The grid uses
equal geometric intervals, without adjusting boundaries to the outcomes.

Within this new run, the 37 positions inside the old, incorrect centre domain
had 6 failures (16.2%); the other 63 positions had 29 failures (46.0%). This is
an exploratory comparison of different positions in the same run, not a
paired comparison or a reuse of the old evaluation outcomes.

Artifacts: [summary](../outputs/wallx_pap_eval_rectangle_20261023/summary.md),
[episode CSV](../outputs/wallx_pap_eval_rectangle_20261023/episodes.csv),
[success-rate figure](../outputs/wallx_pap_eval_rectangle_20261023/success_rates.png),
[fixed-red failure-location figure](../outputs/wallx_pap_eval_rectangle_20261023/location_analysis/failure_locations.png),
[failure-location CSV](../outputs/wallx_pap_eval_rectangle_20261023/location_analysis/failed_initial_positions.csv),
[placement audit](../outputs/wallx_pap_eval_rectangle_20261023/placement_audit.json),
[paired color results](../outputs/wallx_pap_eval_rectangle_20261023/paired_color_breakdown.json),
and [full rollout audit](../outputs/wallx_pap_eval_rectangle_20261023/rollout_audit.json).

#### Historical evaluation with the incorrect region — 2026-10-23

**The following results used the smaller inner quadrilateral obtained by
joining the four marker centres. They do not measure success over the requested
rectangle. The original plans, episodes and counts are retained as historical
evidence. The earlier failure-location analysis also applies only to that
incorrect region.** The corrected run uses the separate
`outputs/wallx_pap_eval_rectangle_20261023` directory.

The historical run completed all 200 episodes in 30.73 minutes. Recomputing the
physical outcomes from every saved trajectory and contact record agreed with
all 200 reports; there were no infrastructure errors or invalid model actions.

| Evaluation | Success | Rate | Wilson 95% interval |
|---|---:|---:|---:|
| Random position, fixed red | 88/100 | 88% | 80.2–93.0% |
| Random position and five balanced colors | 85/100 | 85% | 76.7–90.7% |

| Color in second group | Success | Rate |
|---|---:|---:|
| Red | 14/20 | 70% |
| Green | 18/20 | 90% |
| Blue | 17/20 | 85% |
| White | 18/20 | 90% |
| Yellow | 18/20 | 90% |

All 27 failures reached the 20-second limit. All 12 fixed-red failures lifted
the cube but failed during transport/placement. Color effects cannot be inferred
from the aggregate three-point difference: on the 20 positions assigned red in
both groups, successes changed from 17 to 14. The other four position subsets
each had the same success count before and after changing color, although some
individual outcomes differed. These are single stochastic trials per condition,
not evidence that color never affects the policy.

Artifacts: [summary](../outputs/wallx_pap_eval_20261023/summary.md),
[episode CSV](../outputs/wallx_pap_eval_20261023/episodes.csv),
[success-rate figure](../outputs/wallx_pap_eval_20261023/success_rates.png),
[spatial outcomes](../outputs/wallx_pap_eval_20261023/success_by_position.png),
[paired color results](../outputs/wallx_pap_eval_20261023/paired_color_breakdown.json),
and [full rollout audit](../outputs/wallx_pap_eval_20261023/rollout_audit.json).

### Initial closed-loop check — 2026-10-22

The supplied checkpoint completed two model-controlled 12-second runs with the
current reconstructed scenes. Each run made 12 inference requests and executed
360 predicted actions, using 32 rows before each replan; the final chunk used
only 8 rows before the 12-second cutoff.

| Scene preset | Physical pick-and-place outcome | First stable placement (simulation s) | Mean request latency (s) |
|---|---|---:|---:|
| Red cube / episode 0 layout | Success | 7.5083 | 0.2116 |
| Blue cube / episode 461 layout | Success | 7.2167 | 0.2139 |

These are two single demonstrations, not a success-rate estimate over randomized
layouts. Success requires an actual lift, released placement and settling in
the box. It latches once achieved; the gripper may subsequently close again
after returning home. No arm commands were clipped; negative gripper closure
commands were explicitly saturated at zero opening and recorded. The largest
transient arm tracking errors were 0.503 rad (red) and 0.059 rad (blue); these
runs do not establish smoothness or suitability for real-robot control.

The [run summary](../outputs/wallx_pap_deployment_20261022/summary.json),
[red report](../outputs/wallx_pap_deployment_20261022/red_chunk32/report.json),
[blue report](../outputs/wallx_pap_deployment_20261022/blue_chunk32/report.json)
and [execution video](../outputs/wallx_pap_deployment_20261022/execution_red_blue.mp4)
record the results. The video shows red above blue, third-person on the left
and wrist on the right, at 1× simulation time. Each stream includes t=0 and
t=12 seconds: 361 frames at 30 fps, encoded as 12.0333 seconds. Network waiting
time is excluded because the simulator pauses while waiting for inference.

The native interface also passed four requests using audited real and simulated
initial observations. Prewarmed inference took approximately 0.212 seconds per
32-row chunk. CPU topology/header comparison found no unexpected, mismatched,
or critical missing checkpoint tensors; only the intentionally rebuilt
normalizers and tied output embedding were absent from the loaded-key set.

### Normalization provenance

The supplied training YAML points to
`piper_pap_20260606_20260629_offset_merged`, while the available reconstruction
dataset contains 500 episodes from `piper_pap_20260606_offset`. Their statistics
differ: some joint quantiles differ by about 12 degrees. The launcher therefore
uses **`/home/pengguanqi/Models/piper_pap_20260606/norm_stats.json`**, whose state
quantiles match the checkpoint's normalizer buffers after BF16 rounding,
instead of regenerating statistics from the 500-episode dataset. The original
training YAML, dataset and model files are not edited.

The upstream customized-robot loader uses the same joint names for action and
state statistics, so the latter overwrite the former in its shared mapping.
Both checkpoint normalizers contain those state statistics. This deployment
preserves that existing behavior; it does not silently reinterpret the trained
model's output. The [contract audit](../outputs/wallx_pap_deployment_20261022/contract_audit.json)
records the arrays, source locations, units and image preprocessing.

## Geometry and cameras

Coordinates are metres, Z up. The robot base faces +X; +Y is its left. Task XY
values are relative to the base. The table surface is base Z=0; its absolute
height of 0.76 m is a scene convention, not an inferred real measurement.

| Quantity | Current configuration | Basis |
|---|---|---|
| Cube edge | 3 cm (episode 0); 2.785 cm (episode 461) | Approximate reference / image fit with three equal sides |
| Box outer dimensions | 13.18 × 11.00 × 6.35 cm | Revised fixed-camera image fit; approximate 15 × 10 × 6 cm reference is not a measurement |
| Box centre XY | (0.31556, 0.20464) m | Revised image/kinematic reconstruction |
| Cube centre XY, episode 0 | (0.3158, 0.0135) m | Initial image projection at nominal 3 cm edge |
| Fixed camera, relative to base | (0.6745, 0.0295, 0.5089) m | Multi-frame fit; 75 cm distance / 45 cm height used only as rough context |
| Fixed / wrist focal length | 667.73 / 655.80 px | Approximate square-pixel pinhole fits, centred at (320,240); wrist refined with near-field landmarks |
| Gripper total travel | 100 mm | PAP-only joint-limit overlay using an official Piper variant |

Both cameras use the **same official D435 housing mesh**, with a tabletop tripod
and the existing official wrist bracket. The optical frames are registered
separately; the wrist fit is not a measured mechanical calibration of the bracket.
The full matrices and all geometry are in `configs/piper_pick_place.json`.
These dimensions, intrinsics, lens distortion, lighting, object mass (25 g), and
friction are reconstruction assumptions, not a calibrated digital twin.

`scripts/fit_pap_cameras.py` records explicit manual image correspondences and
joint-derived 3D gripper landmarks. User dimensions are soft priors with 15%
relative tolerance. The initial fit had 43 correspondences, 4.28 px coordinate
RMS (6.05 px point-distance RMS); the largest error is 15.8 px on a moving gripper
landmark. These are fit residuals, not independent calibration accuracy. Initial
wrist views from episodes 10, 100, 400, and 499 were also inspected as held-out
visual checks. The source data provides no calibrated camera matrices.

### Wrist view refinement

The current wrist configuration adds near-field cube and open/closed fingertip
landmarks to the original tabletop-marker fit. It moves the optical centre about
13.8 mm along link6 +Z, adjusts its orientation, and changes fx=fy from 684.94 to
655.80 px. This corrects depth-dependent image displacement; it does not crop or
digitally enlarge the video. All object geometry, joint controls, physics, and
the third-person camera are unchanged.

The four manually annotated fingertip coordinates have fitting RMS of 0.86 px,
down from 17.43 px. Initial-pose tape checks from four other episodes change from
4.58 to 5.00 px RMS per coordinate. These are image-registration diagnostics,
not instrument-calibration accuracy; the stock finger silhouette and estimated
object sizes still differ from reality. The original joint-camera fit metrics
above describe the first reconstruction, before this wrist-only refinement.

```bash
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 .conda/envs/pressb/bin/python \
  scripts/refine_pap_wrist_camera.py --output outputs/pap_wrist_refit

env -u PYTHONPATH .conda/envs/pressb/bin/python scripts/compare_pap_wrist_views.py \
  --dataset /home/pengguanqi/Datasets/piper_pap_20260606_offset --episode 0 \
  --before outputs/piper_pap_reconstruction_20261022/final_scene \
  --after outputs/piper_pap_reconstruction_20261022/wrist_revision/replay \
  --output outputs/pap_wrist_comparison
```

The refiner starts from the frozen initial `final_scene/config.json`, not the
current configuration. Its output is a candidate; the exact applied camera and
annotation details are in `wrist_revision/refinement_report.json`. The new full
physical replay is in `wrist_revision/replay/`; all six physical-state arrays and
633 contact records are identical to the initial run, and success still occurs
at 7.425 s. `wrist_revision/comparison/wrist_before_after.mp4` shows real, previous,
and revised wrist views at 1× nominal speed. `wrist_revision/real_vs_sim/` contains
the updated two-camera real/simulation comparison. Both retain 640×480 source
pixels without digital zoom or retiming.

### Cube and box correction

The current presets enforce cubic blocks and use the same revised cardboard box.
The first blue-block fit allowed three independent dimensions and misidentified
one third-person top corner as a side-face point. Rechecking the source pixels
and imposing a single edge fixes both issues. `scripts/fit_pap_episode_cube.py`
fits XY, yaw and one edge only; it preserves the existing 1:0.25 up/wrist residual
weights. The revised cube's coordinate RMS is 1.14 px in up and 3.32 px in wrist.
These are fitting residuals, not held-out calibration accuracy.

The box moves +0.24 mm in base X and +5.49 mm in base Y. This moves it outward
toward the right edge in the fixed-camera image and left edge in the initial
wrist image. Its outer dimensions change from 126.48 × 108.46 × 56.72 mm to
131.81 × 109.99 × 63.51 mm, and yaw changes from −3.44° to −1.39°. Wall thickness
stays 5 mm. Five wrist poses plus the fixed view constrain the revision, with
extra weight on the initial wrist corner highlighted by the visual mismatch.

For that corner, horizontal discrepancy decreases from 36.05 to 17.45 px.
Across the selected box landmarks, wrist coordinate RMS decreases from 19.56
to 10.03 px; up increases from 2.15 to 7.69 px. This is a compromise under the
frozen approximate cameras, not an improvement at every point. Real cardboard
walls are bent and tapered; a diagnostic tapered fit did not substantially
remove the residual. Camera intrinsics/extrinsics, controls and contact physics
are not retuned to obtain successful replay.

The reproducible fits, manual annotations, candidate comparisons and final runs
are in `outputs/piper_pap_geometry_revision_20261022/`:

- `cube_annotation_check/` records corrected source-pixel corner annotations.
- `cube_fit_refined/` contains the equal-edge fit and sensitivity candidates.
- `box_fit/fit_report.json` records both box candidates, the selected
  `initial_balanced` result and remaining discrepancies.
- `replay_461/` contains the full 290-frame physical replay; stable placement is
  reached at 7.1083 s, with maximum lowest-surface clearance of 16.38 cm.
- `replay_000/` reruns the original red-cube demonstration through 10.5 s with
  the revised shared box; stable placement still occurs at 7.425 s.
- `comparison_461/real_vs_sim.mp4` shows the revised real/simulation comparison
  at nominal 1× speed; source images are neither cropped nor digitally enlarged.
- `initial_before_after.jpg` compares real, previous, and revised initial views.

These outputs supersede the archived unconstrained-block replay for the current
episode 461 preset. Neither replay is a learned-policy evaluation.

```bash
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 .conda/envs/pressb/bin/python \
  scripts/fit_pap_episode_cube.py \
  --source-config outputs/piper_pap_episode_replay_20261022/replay_461_final/config.json \
  --trajectory outputs/piper_pap_episode_replay_20261022/episode_audit/episode_000461.npz \
  --annotations outputs/piper_pap_geometry_revision_20261022/cube_annotation_check/annotations.json \
  --output outputs/pap_cube_refit
```

### Materials and lighting

The current presets also use image-guided PBR materials and directional area
lighting. The block is a smooth non-metallic surface (roughness 0.18), the green
tabletop is moderately rough (0.56), and the cardboard interior/exterior are
rougher (0.86/0.88). The cardboard interior is warm brown; a separate muted
exterior material reproduces the different outer paper. Four visual-only quads
sit 50 micrometres outside the walls; the five original collision faces and
the box cavity remain identical.

`materials` overrides named material colour, roughness, metallic and specular
values. Colours are linear RGB. `object.material` can override the object's
surface properties; its existing `object.color` remains the colour override.
The original `material_colors` configuration remains supported. Omitting the
new appearance options preserves the previous authored USD scene.

Three rectangular lights provide a broad main source from base +X/+Y, soft
overhead fill and a side fill. This corrects the previous reversal of the blue
block's lit and dark faces. The final main light's centre is (0.69, 0.30, 1.10) m
relative to the robot base, with a 0.50 × 0.35 m emitting surface. Every area
light faces its configured `target_base_m`; the main emitter is fully in front
of the reconstructed rear wall. `lighting.ambient` controls the remaining dome
fill. Intensities and reflectance values are rendering estimates, not measured
real-room photometry.

Using 16 manually selected initial-view surface patches, display-colour
diagnostic RMSE drops from 0.1428 to 0.0493 in inverse-sRGB coordinates. The
visible +X/−X block-face luminance ratio changes from 0.19 to 3.30 (real: 2.97),
and the paper-box floor RGB changes from approximately (155,119,70) to
(213,148,120), compared with real (213,144,116). These are **in-sample image
matching diagnostics**, not measured BRDF accuracy. Camera response and
exposure, local table wear, cardboard fibres/printing and robot surface details
still differ; those fine textures have not been reconstructed.

`rendering` controls RTX sampling and convergence independently of the physical
clock. These presets use four direct/reflection samples per pixel, 48 warmup
renders and two subframes per video frame. Extra renders do not advance physics
or change the 30 fps playback speed. The runner's defaults remain 8 warmup
renders and one subframe if these options are omitted.

Artifacts are in `outputs/piper_pap_appearance_revision_20261022/`:

- `reference_analysis/` records source-pixel observations and reproducible
  patch statistics; candidate snapshots retain the tuning history.
- `replay_461/` and `replay_000/` contain the updated blue/red physical replays.
  All six saved physical-state arrays match their corresponding geometry-only
  runs exactly; stable placement remains 7.1083 s and 7.425 s, respectively.
- `comparison_threeway/real_before_after.mp4` compares real, previous simulation,
  and revised simulation, with up/wrist rows and unresized 640×480 cells.

```bash
env -u PYTHONPATH .conda/envs/pressb/bin/python scripts/compose_pap_before_after.py \
  --dataset /home/pengguanqi/Datasets/piper_pap_20260606_offset --episode 461 \
  --before outputs/piper_pap_geometry_revision_20261022/replay_461 \
  --after outputs/piper_pap_appearance_revision_20261022/replay_461 \
  --output outputs/pap_appearance_comparison
```

## Dataset semantics and physical replay

The first six state/action values are interpreted as degrees, then converted to
radians. This agrees with every observed arm state and the bundled joint limits.
Gripper values are interpreted as centimetres of total opening: the observed
maximum is 9.11, and a held cube typically gives about 3.6 while its command
continues closing. The acquisition driver is unavailable, so units remain inferred.
No extra joint offset is invented from the dataset directory name.

The default `observed_arm_action_gripper` mode drives the six physical arm joints
toward their recorded **observed** positions and uses the recorded **gripper
command** to produce closing force. Episode 0 begins with 18 all-zero action rows
while its measured arm remains still; blindly executing those rows would create
a spurious return to zero. `--source action` retains that raw behaviour for
diagnostics, and `--source observation` uses observed positions for all joints.
Initial pose always comes from the first observation. Negative gripper commands
are explicitly bounded at zero, with affected counts and maximum change reported.

Controls are interpolated from 30 Hz onto 120 Hz physics. A cube remains a free
rigid body: there is no grasp attachment, suction constraint, or object teleport.
The box uses a floor and four separate static walls, leaving its cavity open.
Success requires a prior lift, both fingers having contacted the cube, no current
finger contact, the entire rotated cube inside the box, support at the box floor,
and low object velocities continuously for 0.5 seconds. Success is latched at
completion; later return motion may close the empty gripper again.

For the generic object path, containment and support use the full rotated
cuboid or cylindrical surface. Prior lift requires the object's lowest point
to clear the tabletop by 4 cm, so merely tipping a long object upright cannot
produce a false lift. The legacy centre-rise diagnostic is retained separately.

The packed source videos start at PTS=0, but episode video metadata carries a
uniform +0.03 s offset. After checking file frame counts and episode lengths,
extraction uses packed frame indices to avoid reading the next episode's first
frame. This establishes episode membership; it does not establish exact
camera/joint synchronization. The comparison video preserves nominal 30 Hz time
and documents the remaining approximately one-frame timing uncertainty.

## Validation and artifacts

The completed 2026-10-22 reconstruction run is in
`outputs/piper_pap_reconstruction_20261022/final_scene`. Episode 0 physically
grasped, lifted, released, and settled the cube inside the box, reaching the
success condition at 7.425 s with a maximum centre lift of 13.54 cm. This is one
dataset-driven scene validation, not a learned-policy success rate. The run
continues through 10.5 s to show the return motion.

The same output root holds `dataset_audit/`, `camera_fit/`, and the real/simulation
comparison. Source data and generated videos remain outside Git. CPU checks:

```bash
PYTHONPATH=.cache/usd-inspect:src .conda/envs/pressb/bin/python -m pytest -q \
  tests/test_pick_place.py tests/test_pick_place_objects.py tests/test_pick_place_appearance.py
```

## Asset and model references

The original robot and wrist assets remain those listed in `THIRD_PARTY.md`.
PAP overlays only the gripper travel; it does not edit the downloaded USD/URDF.
Official references supporting the 100 mm variant:

- [AgileX Piper SDK: 70 mm / 100 mm gripper configuration](https://github.com/agilexrobotics/piper_sdk/blob/b847e5393a7ce6ff8508d713eca7bdd3a42421b4/piper_sdk/piper_msgs/msg_v2/transmit/arm_gripper_teaching_param_config.py).
- [Official 100 mm Piper gripper URDF](https://github.com/agilexrobotics/Agilex-College/blob/c2688be41e1bc99a9addd555f237a16bc839d936/isaac_sim/agx_arm_IsaacLab/IsaacLab_Data_Collection/source/agx_teleop/agx_description/agx_arm_urdf/piper/urdf/piper_gripper.urdf).
- [Piper control-message unit: 0.001 mm at the SDK protocol boundary](https://github.com/agilexrobotics/piper_sdk/blob/b847e5393a7ce6ff8508d713eca7bdd3a42421b4/piper_sdk/piper_msgs/msg_v2/transmit/arm_gripper_ctrl.py). This does not itself verify the dataset's conversion.
