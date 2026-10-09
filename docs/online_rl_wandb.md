# Live W&B monitoring

`scripts/watch_online_rl_wandb.py` attaches to an existing residual-on-frozen-noise
experiment. It reads `train/metrics.jsonl`, `train/episodes.jsonl`,
`train/status.json`, and supervisor state without restarting training. It writes
only its own `wandb_monitor/` directory. The monitor imports no model or simulator
and sends scalar metrics, configuration, and experiment state to W&B; it does not
upload checkpoints or trajectory files.

The running gamma 0.995 experiment uses a separate Python environment with
`wandb==0.30.0`, so the trainer's dependencies are unchanged:

```bash
env -u PYTHONPATH .cache/wandb-monitor-venv/bin/python \
  scripts/watch_online_rl_wandb.py \
  --output outputs/online_rl_residual_on_noise_gamma0995_400k_20261007 \
  --project pressb-online-rl --interval 5
```

Credentials come from the user's existing W&B login. The run URL and stable ID
are in `OUTPUT/wandb_monitor/run.json`; local monitor status and console output
are `status.json` and `monitor.log` in that directory. A lock prevents duplicate
monitors. To restart, use the same command after the old monitor exits; history
resumes from the prefix acknowledged by W&B. An interrupted upload can replay a
point, but unsent history is not skipped. Stopping this monitor does not stop the
trainer. W&B stays active through the supervisor's subsequent evaluation.

| Section | Contents and axes |
| --- | --- |
| `rollout` | Exact recent 100/1000 completed-episode SR, actual window counts, cumulative SR; transitions axis |
| `live` | Status every 5 seconds: SR, progress, throughput, episode duration, pending updates; independent transitions axis |
| `loss`, `value`, `policy` | Actor/critic/alpha losses, Q target/prediction, entropy, gradient norms, reward and discount; optimizer updates axis |
| `performance`, `progress` | Cumulative and recent throughput, recorded rate window, completed episodes, training/drain transitions, replay size |
| `episode` | Mean whole-episode simulated duration and termination frequencies over the recent 1000 completed episodes |
| `buttons_recent_1000` | Each button's SR and denominator within that same global window |
| `hardware` | GPU utilization, memory, power and temperature sampled every 30 seconds |
| `timing`, `latency` | Cumulative measured times and mean RPC latency; timings overlap due to pipelined training |
| `eval` | Subsequent fixed-condition evaluation results, separate from training SR |

SR is a fraction in `[0, 1]`, so `0.85` means 85%. Before 100/1000 completed
episodes, the denominator is the number available, not a padded window. Each
historical point uses **its own recorded episode count** to select a prefix of
the episode log. The newer log tail cannot change an older point's SR. Whole
episode duration is `sim_seconds`, not the final chunk's duration. One transition
is one environment action chunk, up to seven control steps; throughput is not
episodes per second. Recent throughput uses a historical difference spanning at
least 60 seconds when available and reports the actual span.

Historical metrics are backfilled from the start of training; new history follows
the trainer's existing interval of 256 optimizer updates. Status is sampled
every five seconds. W&B's network/UI refresh may add some delay. Losses are not
interpolated between logged updates. Training uses exploration and randomized
tasks, so its SR is not interchangeable with the later fixed-condition eval SR.
