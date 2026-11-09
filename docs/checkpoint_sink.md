# Streaming checkpoint storage

An optional storage endpoint lets an RL learner save complete checkpoints on
another machine without first writing a temporary checkpoint to its own disk.
The DPPO integration streams the expert, FP32 master parameters, optimizer
states, value network and RNG state; it does not reduce checkpoints to inference
weights. Default local checkpoint behavior is unchanged.

Start the receiver on the machine with available storage:

```bash
python scripts/serve_checkpoint_sink.py \
  --directory /absolute/path/to/checkpoints --port 18391
```

It binds to loopback by default and only requires the standard library. For an
H200 learner reached with `ssh h200`, a reverse tunnel makes that local receiver
available on the learner's loopback address:

```bash
ssh -N -R 18391:127.0.0.1:18391 h200
```

Add this optional section to the DPPO runner configuration (`run` in an
experiment file):

```json
"checkpoint_sink": {
  "endpoint": "http://127.0.0.1:18391",
  "namespace": "my_training_run",
  "timeout": 1800,
  "max_bytes": 34359738368,
  "token_env": "PRESSB_CHECKPOINT_TOKEN"
}
```

The receiver retains every checkpoint under
`<directory>/<namespace>/step_<transitions>.pt`. It writes incoming bytes to a
temporary file in that destination directory, verifies the completed stream's
SHA256 and byte count, fsyncs, then publishes the checkpoint atomically without
overwriting an existing file. Partial or corrupt uploads are removed. The
learner verifies the acknowledgement and stores the resulting URL/checksum in
`checkpoints/latest.json`; it writes no local `.pt` file. Uploads are not retried
automatically if their acknowledgement is lost.

Only single-component namespace and `.pt` filenames are accepted. The receiver
limits individual chunks and complete checkpoint size. Optional bearer
authentication reads `PRESSB_CHECKPOINT_TOKEN` on both hosts; the token itself is
never written into training configs. Prefer loopback plus SSH forwarding for
cross-machine use.

A checkpoint can be resumed or evaluated from its HTTP(S) URL:

```bash
python scripts/run_experiment.py train --config /path/to/experiment.json \
  --resume http://127.0.0.1:18391/checkpoints/my_training_run/step_000100000.pt
```

The learner downloads into RAM, verifies the byte count and SHA256, and loads
with `weights_only=True`. This uses host memory roughly proportional to the
checkpoint size; it creates no disk cache. Model/task/sampler identity checks
still apply. A normal local checkpoint path also remains supported.

This sink is task-independent. DPPO currently calls it directly; other training
backends can use `save_checkpoint_stream` and `load_checkpoint_stream` from
`pressb.runtime.checkpoint_sink` without duplicating the transport.
