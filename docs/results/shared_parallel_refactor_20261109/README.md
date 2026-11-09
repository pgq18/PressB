# Shared parallel simulation refactor — 2026-11-09

The four native simulation tasks now construct one task-independent `SharedWorldBackend`. Task hooks supply the scene, reset sampling and physical outcomes; PiPER drivers supply joint/pose control. The common runtime owns cloning, physics stepping, contact dispatch, tiled rendering and episode scheduling. [Architecture and interfaces](../../project_framework.md#shared-world-simulation-core).

**Validation completed.** The table below lists completed checks at their recorded source revisions. This page does not establish a learned-policy success rate, training speedup or 64-environment capacity for all four tasks.

## Completed checks

| Check | Recorded result | Scope |
|---|---|---|
| Final main CPU suite | 3,600 passed, 27 skipped | No CUDA, hardware access or external training-service calls |
| Final CPU USD supplement | 156 passed; 69 additional passing cases | 3,669 unique passing cases across the two runs |
| PAP, 2 environments | Passed | Shared backend, observations/actions, seeded indexed reset, zero reset ticks, unchanged peer snapshot |
| Elevator, 2 environments | Passed | Same common smoke contract, pose-action driver |
| Playground, 2 environments | Passed | Same common smoke contract, empty workspace; zero reward/no task-success detector |
| Table buttons, 2 environments | Passed after reset correction | Independent layout sampling and common smoke contract |
| Elevator physical replay, 2 environments | Both reached `target_pressed` at tick 148 | Recorded successful trajectory; timeout control terminated both at tick 240 |
| PAP observer recording, 2 environments | Recording audit passed | Two complete episodes at ticks `[0,4,8]`; two partial post-reset episodes at tick `[0]`; no extra physics/render calls |
| PAP physical replay and fused reset, 2 environments | Both completed grasp and placement | Bilateral contact history, lift, release and stable rest in the box |
| Playground, 8 environments | Passed | Same generic hold/reset/seed smoke contract; no task-success claim |
| Table-button physical presses, 2 environments × 2 layouts | Four of four scripted presses succeeded | Red/blue targets, measured finger force and cap travel; no learned-policy evaluation |
| Table buttons, 8 environments | Passed | Same generic hold/reset/seed smoke contract; default 48-frame reset rendering |

The CPU total comes from distinct test identities in the main and supplemental receipts. Individual targeted test runs overlap with these suites and are not added again. All 752 Python source files under `src`, `scripts` and `tests` had matching start/end manifests during this final validation. The [release receipt](../../../outputs/parallel_core_regression_20261109/cpu_release_summary.json) retains skips and source snapshots. CUDA kernel parity, the `torchdiffeq` native-expert check and the official LeRobot writer/reader integration were not exercised in this CPU environment. Earlier validation runs remain historical evidence rather than additional passing cases.

The real simulator smoke checks used GPU 1, with two or eight scene copies in one Isaac World. They exercise reset/step semantics and image delivery rather than a trained model. A passing indexed reset preserves the unselected environment snapshot and advances no physics ticks. Camera configuration stays task-specific; these checks do not recalibrate the cameras or claim pixel-identical output to earlier render sequences.

The elevator contact test reused a recorded successful trajectory in both slots: 148 ticks at 120 Hz is 1.2333 simulated seconds. Its hold-action timeout control ended at exactly 240 ticks, or 2 seconds. The report records `operations_status=passed` and a successful service `close_run`. Kit fast shutdown left the outer report at `status=closing`, `closed=false`; the operator confirmed process exit code 0. Those fields are preserved instead of rewritten as a completed cleanup report.

The PAP observer test recorded two completed short timeout episodes and the initial states of two fresh episodes created by fused reset. Its audit confirms eight world ticks and six tiled captures, with no additional work introduced by recording. This tests timestamp/terminal-state capture and observer behavior, not grasp success.

The separate completed `pap_fused_2env_fixed` replay reached physical grasp-and-place success in both slots, at ticks 894 and 898. Both had contact history on both fingers and a maximum lift of approximately 0.125 m. Four transmitted images matched the producer's expected image-preprocessing result exactly. A diagnostic one-tick timeout with reset budget zero produced two terminal rows without rendering or resetting either environment. The initial incomplete `pap_fused_2env` attempt is retained but not counted as a pass.

The table-button contact regression used two distinct layouts and red/blue targets. All four episodes succeeded at 460 ticks (3.8333 seconds), with 3.9187–3.9999 mm cap travel and nonzero measured finger force. Fixture and spring-anchor drift were zero; the largest recorded anchor/rest or lateral-position error was 1.45 × 10⁻⁷ m. This physics regression explicitly used two reset-render frames. The task defaults separately inherit the scene's render warmup (48 frames in the supplied table-button/Playground scenes); restoring that default does not alter the camera or lighting configuration.

The final eight-environment table-button smoke used that inherited 48-frame warmup. It recorded 156 tiled captures and eight world ticks overall; indexed resets still advanced zero physics ticks. One saved global-camera image was visually checked for stale button remnants. This is a spot check, not a pixel-equivalence or temporal-image-quality benchmark.

## Compatibility and evidence

`pap_process` and `elevator_remote` retain the explicit process backend and remote bridge. The public `pap.vector` and `elevator.native` paths now construct the common core. `pap_recording` observes that core at reset, control boundaries and exact terminal ticks; it does not add physics or rendering calls. The previous native elevator implementation remains at `tasks.elevator.legacy_native`.

- [Machine-readable summary and artifact identities](summary.json).
- [Final main CPU and USD evidence](../../../outputs/parallel_core_regression_20261109/cpu_release_summary.json); [earlier validation receipt](../../../outputs/parallel_core_regression_20261109/summary.json).
- Generic smoke reports: [PAP](../../../outputs/shared_parallel_refactor_20261109/pap_2env/result.json), [elevator](../../../outputs/shared_parallel_refactor_20261109/elevator_2env/result.json), [Playground](../../../outputs/shared_parallel_refactor_20261109/playground_2env/result.json), [table buttons](../../../outputs/shared_parallel_refactor_20261109/table_buttons_2env_fixed/result.json).
- [Elevator contact/timeout report](../../../outputs/shared_parallel_refactor_20261109/elevator_contact_2env/report.json).
- [PAP observer recording audit](../../../outputs/shared_parallel_refactor_20261109/pap_recording_2env/recording_audit.json).
- [PAP physical replay/fused-reset report](../../../outputs/shared_parallel_refactor_20261109/pap_fused_2env_fixed/report.json).
- [Eight-environment Playground smoke](../../../outputs/shared_parallel_refactor_20261109/playground_8env/result.json).
- [Table-button physical-contact regression](../../../outputs/shared_parallel_refactor_20261109/table_contact_2env/report.json).
- [Eight-environment table-button smoke](../../../outputs/shared_parallel_refactor_20261109/table_buttons_8env/result.json).
- Validation entry points: [generic task smoke](../../../scripts/smoke_parallel_tasks.py), [elevator physical smoke](../../../scripts/smoke_elevator_task_backend.py), [PAP recording smoke](../../../scripts/smoke_pap_recording_shared.py).

Earlier table-button attempts remain in the output directory; the successful row refers specifically to `table_buttons_2env_fixed`. Every result remains tied to its report and metadata fingerprint. Existing DP/Wall training processes and old checkpoint identities are outside this refactor validation; no checkpoint migration or training resume is implied.
