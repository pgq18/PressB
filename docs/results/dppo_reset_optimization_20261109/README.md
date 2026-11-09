# DPPO batched-reset validation — 2026-11-09

Batching simultaneous resets reduced measured PAP reset-service time by **7.70×, 28.08× and 49.94×** for sets of 8, 32 and 64 environments. These are in-process reset-service measurements, not end-to-end training speedups.

## Protocol

Both branches use the same new code and one real Isaac simulator. Hiding `per_env_reset_seeds` selects the old singleton schedule; advertising it selects the batched schedule. Each k has two paired measurements, in AB then BA order, with the same selected environments and mixed episode counters. Per-environment reset seeds retain the existing formula.

Hardware and rendering: GPU 1, NVIDIA GeForce RTX 5090; 64 PAP environments; 128 camera views at 640 × 480; two full tiled render captures per reset request; `rl_fast`, eight CPU threads and eight PNG workers. Initial-layout preparation, rendering warmup, image checks and the extra diagnostic capture are outside the reported service timers. No policy inference, RL update or network transport is timed.

## Reset service results

| Environments reset (k) | Singleton mean (s) | Batched mean (s) | Ratio of means | Service calls, singleton → batch | Tiled captures, singleton → batch |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.946 | 0.963 | 0.98× | 1 → 1 | 2 → 2 |
| 8 | 7.583 | 0.985 | 7.70× | 8 → 1 | 16 → 2 |
| 32 | 30.748 | 1.095 | 28.08× | 32 → 1 | 64 → 2 |
| 64 | 61.899 | 1.240 | 49.94× | 64 → 1 | 128 → 2 |

Ratio = mean singleton service seconds ÷ mean batched service seconds. It is not an average of per-pair ratios. Each capture renders the shared 128-view tile; returned images are encoded only for the requested environments. At k=1 both schedules still make one call, so that row provides a same-schedule variation reference without assigning a cause to the difference. Two repeats describe this run; they do not establish a timing confidence interval.

## Correctness and image checks

**598 CPU tests passed** across 19 test files, with zero failures, errors or skips and unchanged tested sources. Coverage includes protocol validation, old-backend fallback, per-environment seeds, initial reset rules, DPPO collection/episode boundaries and GAE.

The real Isaac benchmark checked all eight pairs at runtime: matching reset seeds, physical snapshots and non-image observations; unchanged non-reset peer states and service slots; zero advanced physics ticks; preserved initial shared seed. All **840 measured returned images** matched their corresponding current live camera tile exactly. These checks guard against stale or swapped tiles. Full physical snapshots and every image were not saved, so the report records passed runtime assertions rather than claiming an independent offline reconstruction.

**Images from separate rendering sequences were not pixel-identical.** Below, MAE averages absolute RGB-channel differences on the 0–255 scale, equally over the same-sized images and both repeats. “Common final scene” applies the same extra diagnostic capture after all selected resets; it is excluded from service timings.

| k | Returned-image MAE | Common-final-scene MAE |
|---:|---:|---:|
| 1 | 3.939 | 4.351 |
| 8 | 3.877 | 3.844 |
| 32 | 4.117 | 4.109 |
| 64 | 4.226 | 4.037 |

Equal audited physical state and correct current-tile selection do not establish exact visual equivalence, unchanged policy success or a corresponding whole-training speedup. Reset-group frequency, network transfer, inference and PPO updates still contribute to training time.

## Evidence

- [Machine-readable summary, raw per-repeat times and source hashes](summary.json).
- [Completed Isaac report](../../../outputs/dppo_reset_optimization_20261109/isaac64/report.json).
- [CPU validation receipt](../../../outputs/dppo_reset_optimization_20261109/cpu_validation/validation.json).
- [Benchmark implementation](../../../scripts/benchmark_pap_reset_batching.py).

## Stopped run and deployment

The Wall-OSS PAP 200k run stopped after **7,424 transitions, 2,320 updates and policy version 29**. Its complete 6,526,982,982-byte checkpoint was committed locally; a full-file SHA256 reread and CPU expert-state check matched the sink receipt and last acknowledged policy publication. All owned local and H200 services exited. The checkpoint and old source/configuration archive are retained; no checkpoint identity was rewritten.

The reset changes were synchronized to H200 only after those checks. Its isolated runtime also passed 52 DPPO tests (a subset of the 598 local tests). Model code/configuration and algorithm settings were preserved. **Training has not restarted.** Operational receipts are under `outputs/pap_walloss_dppo_200k_20261109/stop_evidence/` and `outputs/dppo_reset_optimization_20261109/deployment/`.
