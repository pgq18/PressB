# PAP parallel simulation capacity

PAP now has a shared-World simulation backend alongside its original process-per-environment backend. Both expose the configurable RL `reset/step` service. **64 environments with 128 native-resolution cameras have run successfully on one RTX 5090**, including a recorded grasp-and-place action sequence that completed successfully in all 64 copies. Each environment shares the Isaac application and renderer.

## Execution model

| Property | `pap:create_backend` | `pap_vector:create_backend` |
|---|---|---|
| Isaac applications | One per environment | One per node |
| Scene organization | Independent worlds | Separate, collision-filtered scene copies in one World |
| Robot and object state | Per-process calls | Batched PhysX tensor views |
| Physics solver | CPU PhysX | CPU PhysX; tensor views do not imply GPU physics |
| RGB | Independent renderers | One tiled render product containing both cameras of every environment |
| Step scheduling | Selected environments | All active environments, equal action-prefix lengths |
| Reset | Selected workers settle their scenes | Selected slots restore a cached settled arm and supported object with zero physics ticks |
| Object geometry | Configurable object variants | One configured geometry per node; positions and colors can vary |

The vector backend retains the calibrated third-person and wrist camera poses, optical offsets and native **640×480** images. A smaller resolution is optional and must preserve the calibrated aspect ratio. Environment copies are separated by six metres and collision-filtered; this world-space translation does not change their robot-relative actions or observations.

Indexed resets do not advance other environments' physical state or episode clocks. Terminal observations are captured at their actual terminating physics tick before other slots continue. Backend diagnostics expose joint state, velocities, object poses and episode clocks so the benchmark can check reset isolation directly.

## Configuration and launch

The service configuration is [`configs/rl_nodes/pap_vector_simulation.json`](../configs/rl_nodes/pap_vector_simulation.json). It requests 64 environments, two 640×480 RGB cameras per environment, a 32-command maximum action chunk, 30 Hz control and 120 Hz physics. The configuration retains **48 reset render frames** and **2 render subframes** from the calibrated PAP image pipeline. GPU placement, environment count, randomization, episode duration, rewards and optional action smoothing remain configurable.

Start a persistent simulation node with:

```bash
.conda/envs/pressb/bin/python scripts/serve_simulation_node.py \
  --config configs/rl_nodes/pap_vector_simulation.json
```

The inference and learner nodes connect through the existing task-neutral node protocol; see [configurable RL nodes](configurable_rl_nodes.md). Changing the simulation backend does not enable true Wall-OSS batch inference by itself.

## Bounded capacity probes

Run probes from the repository root. Every output directory must be new. The harness launches its own isolated child, records GPU memory, process-tree RSS and host available memory, and stops only its own process tree if free GPU memory falls below **2 GiB**, host available memory falls below **8 GiB**, or the time limit expires. Resource samples and cleanup results are saved in `report.json` and `resources.jsonl`; stdout from the simulator is in `child.log`.

The following command tests 64 environments with the normal 48-frame reset history:

```bash
.conda/envs/pressb/bin/python scripts/benchmark_pap_parallel.py \
  --config configs/rl_nodes/pap_vector_simulation.json \
  --num-envs 64 --gpu-id 0 --chunks 10 \
  --action-horizon 32 --max-episode-seconds 2 \
  --output-dir outputs/pap_capacity_64_default_history
```

For a shorter **capacity-only** probe, explicitly reduce reset history to eight render frames:

```bash
.conda/envs/pressb/bin/python scripts/benchmark_pap_parallel.py \
  --config configs/rl_nodes/pap_vector_simulation.json \
  --num-envs 64 --gpu-id 0 --chunks 10 \
  --action-horizon 32 --max-episode-seconds 2 \
  --reset-render-frames 8 --render-subframes 2 \
  --output-dir outputs/pap_capacity_64_eight_reset_frames
```

Eight frames are an explicit benchmark override, not the recommended replacement for PAP's 48-frame reset history. Faint gray remnants of the previous object position were visible with the shorter history. Temporal rendering history affects observation convergence and reset cost; compare throughput only with these settings stated. The capacity trajectory holds the initial pose with a small environment-specific first-joint offset. It validates images, distinct randomized object positions, action execution, timeouts and indexed resets; it does not test task success.

One transition means **one environment's action-chunk response**. The report separates startup, initial reset, pure `step` time and rollout time including subsequent resets. These rates exclude model inference, network communication and learner updates. The harness also measures JSON request/response sizes against a 64 MiB budget; the instrumentation's serialization time is recorded and excluded from throughput timings. Process-tree RSS is a sum of process RSS, so shared pages can be counted more than once. GPU allocation samples are not an exclusive reservation of the device.

## Physical replay qualification

Replay a previously recorded successful PAP execution to check that grasping, lift, transport and release still work in all scene copies:

```bash
.conda/envs/pressb/bin/python scripts/benchmark_pap_parallel.py \
  --config configs/rl_nodes/pap_vector_simulation.json \
  --num-envs 64 --gpu-id 0 \
  --replay-episode outputs/wallx_pap_eval_20261023/run/episodes/red_000 \
  --max-episode-seconds 20 \
  --output-dir outputs/pap_replay_64_default_history
```

Replay mode uses the recorded scene and cube position, extracts the already-applied targets from `physics.npz` at the control rate, and converts eight simulated joint values to six arm joints plus gripper width. It disables additional smoothing. All slots receive the same recorded commands in their own robot frames; the last command is held until success or timeout. The report includes bilateral finger contact counts, lift, containment, release, stability and final object positions for each environment, plus initial and final camera contact sheets.

This verifies physical replay consistency. It is **not a fresh model evaluation, randomized-task success rate, Wall-OSS batch-64 inference test or end-to-end DSRL throughput measurement**.

## Validation record: 2026-11-04

Artifacts are under `outputs/pap_parallel_capacity_20261104/`. Completed evidence at this revision:

| Probe | Reset render frames | Result | Peak owned GPU memory | Peak process-tree RSS |
|---|---:|---|---:|---:|
| Original process backend, 2 environments | 48 | 20 transitions, 10 timeout episodes | 8.65 GiB | 10.84 GiB |
| Shared World, 16 environments | 8 | 64 transitions, 32 timeout episodes; independent resets verified | 6.88 GiB | 7.17 GiB |
| Shared World, 2 environments, recorded action replay | 8 | Both copies grasped, lifted and deposited the cube successfully | 4.73 GiB | 5.31 GiB |
| Shared World, 64 environments | 8 | 640 transitions, 320 timeout episodes; 10 independent-reset checks passed | 15.07 GiB | 12.46 GiB |
| Shared World, 64 environments, recorded action replay | 48 | 64/64 physical replay successes with bilateral contacts and stable placement | 15.10 GiB | 12.65 GiB |
| Shared World, 64 environments, randomized reset | 48 | 128 transitions, 64 timeout episodes; two independent-reset checks passed | 15.10 GiB | 12.23 GiB |

The 16-environment probe returned 29.41 transitions/s during `step` and 9.15 transitions/s including its frequent two-second episode resets. The 64-environment eight-frame probe returned **35.82 transitions/s during `step`**, or **9.27 transitions/s including resets**. It produced 64 distinct randomized positions and valid RGB from all 128 cameras; all checks confirmed that indexed resets left the other slots' physical state and clocks unchanged. These are short simulation-only measurements. The process baseline used a different reset-render history; its overall throughput is not a controlled backend speed comparison.

The 48-frame 64-environment replay reached physical success after 7.325–7.5 simulated seconds, with approximately 12.49 cm of lift in every copy. All 64 reported both finger contacts, lift, full containment, release, stability and resting on the box floor. The final camera grids were inspected, and the completed probes left no owned simulator processes running. Replaying one fixed sequence successfully in 64 copies does not establish a policy success rate across randomized placements.

The default 48-frame randomized-reset probe retained 64 distinct positions and valid 640×480 images from all 128 cameras. The camera grid was inspected and showed a clean reset history. Its short step-only rate was **39.10 transitions/s**; including deliberately frequent two-second timeouts and two separate indexed-reset calls, the rate was **2.27 transitions/s**. Initial reset took 26.90 seconds. This stress test resets one environment and then the other 63 to verify isolation; it does not estimate throughput for the normal 20-second task horizon.

The largest measured JSON response was **66,183,538 bytes (63.12 MiB)**, leaving only 0.88 MiB below the current 64 MiB budget. All five measured responses fit in this run, but this is direct JSON sizing, not a network transport qualification. More complex images or another scene can exceed that budget. Transport batching, image compression or configurable message limits need explicit qualification before claiming a complete 64-environment inference-and-training pipeline.

Machine-readable counts, resources, configuration identities and artifact paths are recorded in [`results/pap_parallel_capacity_20261104.json`](results/pap_parallel_capacity_20261104.json).
