# PiPER control-path audit — 2026-11-07

The corrected controller has **successfully reached PAP episode 144's initial pose and subsequently replayed the complete episode** on an empty table. Initial positioning took 16.237 seconds; the separate full replay completed 342/342 actions in 135.524 seconds, with maximum per-action joint arrival error 0.499°. Latest feedback is `arm_status = 0`, `error_code = 0`, all six motors enabled and no reported motor fault. The arm now holds the episode's final pose. This verifies motion replay, not object-transfer success or online RL. See the [replay evidence snapshot](piper_replay_20261107.json).

The reviewed reference and PressB use the same CAN joint-control mode and equivalent position units. The material difference is how startup targets are constructed and constrained. This audit does not establish a firmware mode problem or validate a replacement physical controller.

## Actual reference entry path

The path requested by the user is:

```text
piper_client.sh
  → piper_client.py control_loop
  → PiperFollower.send_action(manual_control=True)
  → PiperMotorsBus.write
  → MotionCtrl_2 / JointCtrl / GripperCtrl
```

The alternative `piper_client_numbers.sh` selects `piper_client_numbers.py` and the same `PiperFollower` / `PiperMotorsBus` driver with manual control enabled. Its fixed elevator idle target is different: `[15.00, 8.00, -45.32, 6.48, 32.62, -6.74]` degrees, gripper closed ([source](/home/pengguanqi/Workspace/Work/lerobot-piper/piper_scripts/piper_inference/piper_client_numbers.py:162)). This is another task-specific pose; neither client's fixed idle should replace the PAP dataset's initial pose implicitly.

The [shell entry, line 78](/home/pengguanqi/Workspace/Work/lerobot-piper/piper_scripts/piper_inference/piper_client.sh:78) selects `piper_follower`, `can0` and `--robot.manual_control=true`. The [follower's send_action, line 155](/home/pengguanqi/Workspace/Work/lerobot-piper/src/lerobot/robots/piper_follower/piper_follower.py:155) converts the named fields to joint order and calls `bus.write` when manual control is enabled. Its additional `max_relative_target` clamp defaults to `None` in the [configuration, line 38](/home/pengguanqi/Workspace/Work/lerobot-piper/src/lerobot/robots/piper_follower/config_piper_follower.py:38); the shell does not override it.

At [PiperMotorsBus.write, line 305](/home/pengguanqi/Workspace/Work/lerobot-piper/src/lerobot/motors/piper/piper_motors_bus.py:305), the reference scales, rounds and sends each target with:

```python
MotionCtrl_2(0x01, 0x01, 100, 0x00)
JointCtrl(*joints)
GripperCtrl(abs(gripper), 1000, 0x01, 0)
```

The first two mode arguments select CAN control and MOVE J; the final argument disables MIT mode. PressB uses the same mode fields with configurable speed, set to **5%** for the attempted approach. The reference's **100%** speed argument is a real configuration difference, not a different motion mode. The actual second-attempt sender is preserved at [executed_home_piper_real.py, line 525](/home/pengguanqi/Workspace/Research/PressB/outputs/piper_replay_20261107/executed_home_piper_real.py:525).

## Units match; startup behavior differs

| Quantity | Reference conversion | PressB conversion | SDK units |
| --- | --- | --- | --- |
| Six arm joints | degrees × 1,000 | radians × 180,000 / π | 0.001 degree |
| Full gripper opening | centimetres × 10,000 | metres × 1,000,000 | micrometres |

These conversions are equivalent. The reference factors are defined at [bus driver, line 76](/home/pengguanqi/Workspace/Work/lerobot-piper/src/lerobot/motors/piper/piper_motors_bus.py:76); its model-facing defaults are degree/cm at [client, line 565](/home/pengguanqi/Workspace/Work/lerobot-piper/piper_scripts/piper_inference/piper_client.py:565). Neither the source path nor the dataset name justifies adding a joint offset.

A hardware-free comparison loaded the actual reference bus with a stub SDK and compared three targets: reference idle, PAP episode 144 action 0 and action 20. At an explicitly matched speed of 100%, the resulting SDK arguments and call order were identical in all three cases. [`reference_encoding_audit.json`](/home/pengguanqi/Workspace/Research/PressB/outputs/piper_replay_20261107/reference_encoding_audit.json:1) records the calls and source hashes. This verifies encoding and ordering for those cases, not startup/feedback policy equivalence or physical execution.

The reference starts with [`running = False`, line 109](/home/pengguanqi/Workspace/Work/lerobot-piper/piper_scripts/piper_inference/piper_client.py:109). Its [idle branch, line 981](/home/pengguanqi/Workspace/Work/lerobot-piper/piper_scripts/piper_inference/piper_client.py:981) directly sends the fixed idle target:

```text
J1 −0.04306525°, J2 +1.40575743°, J3 −0.75656450°,
J4 +1.79226434°, J5 +2.44608521°, J6 −2.91705346°,
gripper 0 cm
```

All six targets are within the queried firmware limits. In particular, the first J2/J3 target is already on the nominally valid side of zero. The values come from [piper_idle_action, line 175](/home/pengguanqi/Workspace/Work/lerobot-piper/piper_scripts/piper_inference/piper_client.py:175), whose source comment describes an elevator dataset mean; they are **not** PAP episode 144's initial pose. This branch does not implement PressB's startup feedback envelope or slew-limited recovery from an out-of-range measured pose. It also moves the other four joints to their fixed idle targets.

For policy actions, the reference has [45° per-joint and 3 cm gripper delta bounds, line 143](/home/pengguanqi/Workspace/Work/lerobot-piper/piper_scripts/piper_inference/piper_client.py:143). The [policy branch, line 952](/home/pengguanqi/Workspace/Work/lerobot-piper/piper_scripts/piper_inference/piper_client.py:952) applies these against a freshly measured pose, falling back to stored state if observation fails. These are **per-update target differences**, not degrees/second or centimetres/second. The idle branch bypasses this policy clamp. They should not be equated with PressB's 5°/s target-slew cap or copied as an equivalent speed setting.

## What the second PressB attempt actually sent

The seven successful send records in [home_run_01/replay.jsonl](/home/pengguanqi/Workspace/Research/PressB/outputs/piper_replay_20261107/home_run_01/replay.jsonl:1) reconstruct the following SDK joint targets. Values are integer millidegrees, derived from the logged `applied` SI state:

| Tick | J2 raw target | J3 raw target |
| --- | ---: | ---: |
| 1 | −1430 | +2263 |
| 2 | −1264 | +2097 |
| 3 | −1098 | +1931 |
| 4 | −932 | +1765 |
| 5 | −766 | +1599 |
| 6 | −600 | +1433 |
| 7 | −434 | +1267 |

Every J2 target remained negative and every J3 target positive: PressB was still generating small inward reference steps from the out-of-range measured start. Its feedback-dependent envelope contracts as the measured arm moves inward, while its reference can advance by only about 0.166° per 30 Hz tick. These two constraints can conflict when measured motion gets ahead of the limited reference. The resulting exception was `PiPER projected setpoint exceeds the active startup recovery envelope`, generated by PressB's command projection check. It was not a reported hardware collision.

The exact first rejected sample was not stored in this run. The log establishes the issued targets and the failed guard; it does not justify reconstructing an exact rejected joint/value from feedback sampled after the stop. The main chain difference above explains why merely matching SDK mode fields does not reproduce the reference client's startup behavior.

## ConnectPort, initialization and unrelated helpers

The reference [bus constructor, line 69](/home/pengguanqi/Workspace/Work/lerobot-piper/src/lerobot/motors/piper/piper_motors_bus.py:69) creates `C_PiperInterface_V2` and calls `ConnectPort()` with defaults. In the **0.6.2 SDK inspected for this test**, [ConnectPort, line 590](/home/pengguanqi/Workspace/Research/PressB/outputs/piper_connection_20261107/sdk_0_6_2/interface/piper_interface_v2.py:590) defaults `piper_init=True`; [PiperInit, line 723](/home/pengguanqi/Workspace/Research/PressB/outputs/piper_connection_20261107/sdk_0_6_2/interface/piper_interface_v2.py:723) queries angle/speed limits, acceleration limits and firmware version. It does not home the arm. The reference's subsequent [connect, line 100](/home/pengguanqi/Workspace/Work/lerobot-piper/src/lerobot/motors/piper/piper_motors_bus.py:100) explicitly enables the arm and gripper.

The separate [reset_to_pose helper, line 186](/home/pengguanqi/Workspace/Work/lerobot-piper/src/lerobot/motors/piper/piper_motors_bus.py:186) is not called by this inference client. It includes interpolation and later `ReqMasterArmMoveToHome(0)` restoration for master/slave use. Similarly, [setup_piper_master_slave.sh, line 76](/home/pengguanqi/Workspace/Work/lerobot-piper/piper_scripts/setup_piper_master_slave.sh:76) configures a different master/slave setup. Neither should be copied into this single-arm inference connection just because it exists in the same repository. The available evidence does not call for changing master/slave or SDK motion modes.

## Firmware speed-unit correction

The read-only version query returned **S-V1.8-7**. Each motor's speed-limit field was raw `300` (`0x012c`). The official changelog specifies 0.01 rad/s per raw unit for firmware V1.5-8 and later, so this arm's reported maximum is **3.0 rad/s**, not 0.3 rad/s. [AgileX SDK changelog](https://github.com/agilexrobotics/piper_sdk/blob/master/CHANGELOG.MD).

The first decoding used the older 0.001 rad/s unit from stale SDK comments. [`firmware_limits.json`](/home/pengguanqi/Workspace/Research/PressB/outputs/piper_replay_20261107/firmware_limits.json:1) remains unchanged as original evidence, including that incorrect derived value. [`firmware_limits_corrected.json`](/home/pengguanqi/Workspace/Research/PressB/outputs/piper_replay_20261107/firmware_limits_corrected.json:1) records the corrected interpretation and preserves the same raw response bytes; [`firmware_version.json`](/home/pengguanqi/Workspace/Research/PressB/outputs/piper_replay_20261107/firmware_version.json:1) preserves the version response. No firmware parameter was changed. The actual 5°/s software reference cap is distinct from the 5% SDK speed field; multiplying the latter by 3 rad/s is only a nominal comparison, not a measured physical-speed guarantee.

## Provenance and remaining validation

The local reference checkout HEAD was `8fff0fde7c79f23a93d845d1a50e985de01f8b8a`. The table fingerprints the **files actually inspected**, including any local modifications. The isolated test environment's SDK version is known (`piper_sdk 0.6.2`, `python-can 4.6.1`); the SDK version used by historical executions of the reference client is not established by its source checkout or shell environment name.

| Inspected file | SHA-256 |
| --- | --- |
| `piper_scripts/piper_inference/piper_client.sh` | `96a380107250881ecb8b0aab7c2912222c09179fa3588ebb49cf0e3da3235768` |
| `piper_scripts/piper_inference/piper_client.py` | `04bf989456259e62b621406eb4563747394bf6df2c60331fb2effb8dfa8f7763` |
| `src/lerobot/robots/piper_follower/piper_follower.py` | `5dc50551d543b813c5f3d021e51c57adbbf7fd8c6c527d5046d16e7e1c892329` |
| `src/lerobot/robots/piper_follower/config_piper_follower.py` | `f1905487becaeba642b2a1bfb6c7dc118b32f927f300c7ad8308cedef8302ba9` |
| `src/lerobot/motors/piper/piper_motors_bus.py` | `000f1128f249d9f5704f984ce1bcb2519cc0826d7cf6720881a562619e7144ca` |
| Archived test SDK `interface/piper_interface_v2.py` | `1dbdbfd8eb133542634c40356e2096efe08c4c4c47501b13808cdd20661800e5` |
| Actual second-attempt `executed_home_piper_real.py` | `330f6209852fd5dd6718fe5e66dafb19d9358b8432e1e30a283d520ce5c86861` |
| Actual second-attempt `executed_home_real_replay.py` | `cb6da3d7562ca3e93b58e1a295bf162931dd0310dee053a02fc53c854186edb3` |
| Third-attempt native adapter `src/pressb/robots/piper/real.py` | `273a4cc4963b3176330f2c859cc1ab4e0790d5d04044ea312381ba8e7beb3423` |
| Third-attempt native runner `src/pressb/control/real_replay.py` | `cf86140cab64688fbc949096b09effea6e87978772a27142099df7de3f65a82b` |
| Deployed boundary-fix adapter `src/pressb/robots/piper/real.py` | `be47a32482d073f9477d4f8b66aefbc9ac339c39b51b4b9952ca69f11f271166` |

The code preceding the second physical attempt passed 208 software tests. Afterward, additional diagnostic logging and regression coverage passed **213 tests**, recorded in [`replay_regression_after_diagnostics.log`](/home/pengguanqi/Workspace/Research/PressB/outputs/piper_replay_20261107/replay_regression_after_diagnostics.log:1). The later count is not evidence that the later code ran successfully on hardware.

The initial native-startup draft was withdrawn for review. The subsequently completed implementation is now available through the default-off `native_startup_recovery` plan block and `recover_startup_native`. It adopts the reference's complete legal-target/MOVE J mechanism with a separate 1% device-speed recovery phase. From fresh feedback it changes only initially out-of-range J2/J3, targeting at most 1° inside the limits and holding other axes/gripper. Limits are 4° travel, 20 seconds, a 5°/s measured-speed guard, 1° drift on nonrecovering joints and 1 mm gripper drift; fresh in-range feedback must remain within 0.25° of target for at least 0.5 seconds. Callers must supply firmware speed limits whose units they have verified. This phase deliberately uses device planning instead of the ordinary reference-slew rule; it does not reset emergency stop. See [the integrated replay documentation](../real_robot.md#bounded-dataset-replay).

The completed implementation passed **332 related software tests** in [`native_final_regression.log`](/home/pengguanqi/Workspace/Research/PressB/outputs/piper_replay_20261107/native_final_regression.log:1), plus an [independent fake-MCU integration](/home/pengguanqi/Workspace/Research/PressB/outputs/piper_replay_20261107/native_offline_integration/validation.json:1). The integration exercised the actual adapter and runner but no real hardware; its target-clipping model and idealized speed integration are assumptions, not observations of this firmware. The local and remote native-plan validations match, both with `executed: false`. The code hashes above were deployed for the prepared plan, which uses verified 3 rad/s maxima, 1% SDK speed for native recovery and ordinary approach, and 0.5° ordinary arrival tolerance.

The prepared native plan was subsequently refreshed after the separately authorized second reset and physically attempted after hands-clear confirmation. [`home_native_run_01/summary.json`](/home/pengguanqi/Workspace/Research/PressB/outputs/piper_replay_20261107/home_native_run_01/summary.json:1) records failure after 1.234 seconds, with 43 native-feedback records, no ordinary ticks and no completed actions. The shared startup-feedback guard retired recovery on the first nominal-boundary reading; raw CAN shows J2 reaching 0° at `1794041962.9158902`, then −0.052° at `1794041962.9413748`. That second reading was rejected as outside nominal limits after J2's recovery allowance had already been cleared. Emergency-stop feedback first appears afterward, at `1794041962.965425`. This identifies a boundary-retirement bug in the executed implementation.

Enable-before-motion was verified, not inferred from the final enable flags. [`native_enable_feedback_transitions.json`](/home/pengguanqi/Workspace/Research/PressB/outputs/piper_replay_20261107/native_enable_feedback_transitions.json:1) records six false-to-true driver transitions between 16:59:21.937 and 16:59:22.017 CST, before the first native-feedback record at 16:59:22.044. The concise [enable audit](/home/pengguanqi/Workspace/Research/PressB/outputs/piper_replay_20261107/native_enable_audit.json:1) preserves the decoding and chronology. Reference connection sends `EnableArm(7)` and gripper enable; PressB additionally awaits fresh enabled feedback. The SDK's [CAN bus construction, line 158](/home/pengguanqi/Workspace/Research/PressB/outputs/piper_connection_20261107/sdk_0_6_2/hardware_port/can_encapsulation_v0_4_0.py:158) sets `local_loopback=False`; consequently, a separate passive socket's missing SDK transmit frames cannot establish that enable or motion commands were absent.

The native-only correction is now implemented, validated offline and deployed. It retains only the original recovering J2/J3 flags until stable interior arrival or native-phase exit, without increasing the 0.25° feedback allowance or relaxing ordinary strict limits; it also rejects reversed recovery targets before sending. [339 final tests passed](/home/pengguanqi/Workspace/Research/PressB/outputs/piper_replay_20261107/native_boundary_regression_tests_final.log:1). An independent fake-MCU run reuses the 43 recorded samples and appends the observed −0.052° return: [old code reproduces the failure](/home/pengguanqi/Workspace/Research/PressB/outputs/piper_replay_20261107/native_boundary_regression/before_recorded/summary.json:1), while [final corrected code completes the simulated home](/home/pengguanqi/Workspace/Research/PressB/outputs/piper_replay_20261107/native_boundary_regression/current_recorded_3/independent_summary.json:1) with zero illegal targets, 108 native samples, 416 ordinary ticks, 16.003 seconds of virtual time and maximum final joint error 0.451°. The [−0.251° return case still stops](/home/pengguanqi/Workspace/Research/PressB/outputs/piper_replay_20261107/native_boundary_regression/current_excess_return/independent_summary.json:1). These tests do not use physical hardware. [Deployment evidence](/home/pengguanqi/Workspace/Research/PressB/outputs/piper_replay_20261107/native_boundary_deployment.json:1) confirms matching local/remote hashes and a remote dry run with `executed: false`. That artifact describes the status at deployment, before the later separately authorized reset; it remains unchanged.

After the operator authorized a third reset only, [reset_run_03](/home/pengguanqi/Workspace/Research/PressB/outputs/piper_replay_20261107/reset_run_03/report.json:1) sent one reset frame with no enable/motion commands. The [post-reset read-only probe](/home/pengguanqi/Workspace/Research/PressB/outputs/piper_replay_20261107/after_reset_03_feedback.json:1) confirmed a cleared stop and disabled motors. The refreshed `home_plan_native_after_reset_03.json` passed matching local/remote dry runs before execution; those reports' `executed: false` values describe the validation step.

After separate hands-clear confirmation, the corrected controller's [fourth physical approach succeeded](/home/pengguanqi/Workspace/Research/PressB/outputs/piper_replay_20261107/home_native_run_02/summary.json:1): 16.237 seconds total, 1.951 seconds of native recovery, 93 native-feedback records, 408 ordinary ticks and one completed action. The [success audit](/home/pengguanqi/Workspace/Research/PressB/outputs/piper_replay_20261107/home_native_success_audit.json:1) verifies maximum arrival error 0.454° and all six motor enable transitions before native recovery. Post-home feedback was `[-1.691, 24.917, -24.717, -4.792, 40.323, 5.047]` degrees, zero gripper opening, `arm_status = 0`, `error_code = 0`, six motors enabled and no faults. Its 100-sample probe transmitted zero frames and the home-run recordings closed.

The user next requested one complete episode (144, 342 frames), rather than all 500 dataset episodes. The [full physical replay](/home/pengguanqi/Workspace/Research/PressB/outputs/piper_replay_20261107/full_episode144_run_01/summary.json:1) completed 342/342 actions and 3,944 ticks in 135.524 seconds, with no additional reset/native recovery and all startup allowances zero. Device speed was 1%, joint-reference slew 5°/s and gripper-reference slew 10 mm/s. The [full-replay audit](/home/pengguanqi/Workspace/Research/PressB/outputs/piper_replay_20261107/full_episode144_audit.json:1) verifies every action index and maximum arrival errors of 0.499° / 1 mm. The source timestamps span 11.367 seconds; execution was deliberately retimed to tracking and speed limits.

**Initial positioning and one complete empty-table trajectory replay are physically verified.** Latest final-pose feedback is `[-10.621, 27.116, -26.308, -6.519, 42.856, 4.695]` degrees, closed gripper, `arm_status = 0`, `error_code = 0`, six motors enabled and no faults; the subsequent 100-sample probe sent zero frames. The arm remains enabled at this final pose, not the initial pose. Three earlier failures and three authorized resets remain historical evidence; no new reset was needed for the full episode. Object grasp/place outcomes, task success rates and physical online RL remain unverified.


## Contact replay stop-policy comparison after the 14 mm retry

The reference online path (`piper_client.sh` → `piper_client.py` →
`PiperFollower.send_action` → `PiperMotorsBus.write`) does not implement PressB's
per-target arrival wait, five-second tracking timeout, or `MotionCtrl_1` stop
request. Client lines 943–994 dequeue and send one target each control cycle,
with target joint clipping and per-update delta limits; block-end settling is
a fixed delay. Bus lines 132–157 read position without the additional feedback
freshness/fault checks. Its disconnect at lines 112–129 intentionally does not
send DisableArm or Standby; client cleanup first sends its idle pose. The
separate `reset_to_pose` helper has a 15-second wait with warning on timeout,
and is not used by this online client. These statements concern the reviewed
Python path, not all SDK or firmware protections.

PressB added a different execution policy: feedback-dependent target advance,
startup bounds, freshness and fault checks, watchdogs, and a latched stop for
control exceptions. In the latest physical trial, release target 28.167 mm was
below loaded feedback 35.1 mm; the added soft-arrival rule prevented advancement
and then triggered the five-second timeout. This was a software control defect,
not evidence of a reported motor fault. Archived old code reproduces action
939 failure with a dynamic MCU; the local release-transition fix completes
all 1188 actions in that same test. It has not been deployed or physically
validated. The operator separately confirmed that the real cube was gripped
but not lifted. See [the trial record](playground_mount14_grasp_retry_20261107.json).
