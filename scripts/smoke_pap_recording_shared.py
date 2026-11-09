#!/usr/bin/env python3
"""Run the shared PAP hold/timeout/fused-reset probe with trajectory recording.

Uses the same two-control-period smoke test as smoke_pap_task_backend.py. Before
Isaac shutdown, checks that the recording observer saved exact terminal ticks,
separate reset episodes, measured clocks, and no extra render or physics work.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from smoke_pap_task_backend import parser as common_parser, require, run


class RecordingAudit:
    """Audit after the recorder flushes, before SimulationApp fast shutdown."""
    def __init__(self, backend, output):
        self.backend, self.output = backend, output
        self.startup_captures = backend.timings["captures"]

    def close(self):
        backend, cfg = self.backend, self.backend.execution
        report = {"status": "failed", "kind": "pap_shared_recording_audit", "episodes": []}
        try:
            stride = cfg["physics_ticks_per_action"]
            require(backend.timings["world_ticks"] == 2*stride, "Recording altered shared physics work")
            expected_captures = cfg["reset_render_frames"] + cfg["observation_render_frames"] + max(
                cfg["observation_render_frames"], cfg["reset_render_frames"])
            captures = backend.timings["captures"]-self.startup_captures
            require(captures == expected_captures, "Recording introduced additional camera captures")
            directory = backend.recorder.directory
            for eid in range(backend.num_envs):
                for episode in range(2):
                    path = directory/f"env_{eid:03d}_episode_{episode:03d}.json"
                    meta = json.loads(path.read_text())
                    expected_ticks = [0, stride, 2*stride] if episode == 0 else [0]
                    require(meta["complete"] is (episode == 0), "Terminal/reset recording completeness differs")
                    require(meta["final_physics_tick"] == expected_ticks[-1], "Incorrect final recorded tick")
                    reason = "timeout" if episode == 0 else "recorder_closed_before_terminal"
                    require(meta["info"]["termination_reason"] == reason, "Incorrect recorded termination reason")
                    with np.load(path.with_suffix(".npz")) as data:
                        require(data["episode_physics_steps"].tolist() == expected_ticks,
                                "Recording omitted a control boundary or exact terminal")
                        clocks = data["world_time"]-data["world_time"][0]
                        require(np.allclose(clocks, np.array(expected_ticks)*cfg["physics_dt"], atol=1e-8, rtol=1e-5),
                                "Recorded world clocks disagree with executed ticks")
                        require(not data["success"].any(), "Hold probe unexpectedly succeeded")
                    report["episodes"].append({"env_id": eid, "episode_index": episode,
                        "complete": meta["complete"], "ticks": expected_ticks, "metadata": str(path)})
            index = [json.loads(line) for line in (directory/"index.jsonl").read_text().splitlines()]
            require(len(index) == backend.num_envs*2, "Recording index omitted or duplicated episodes")
            require(not backend.recorder.slots, "Recorder close did not flush active reset episodes")
            report.update(status="passed", world_ticks=backend.timings["world_ticks"],
                captures=captures, expected_captures=expected_captures, additional_physics_or_render_calls=0)
        except BaseException as error:
            report["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            (self.output/"recording_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
            print(json.dumps({"event": "recording_audit", **report}), flush=True)


def main(argv=None):
    arguments = common_parser()
    arguments.description = __doc__
    args = arguments.parse_args(argv)
    if args.output_dir is not None and args.output_dir.expanduser().exists():
        arguments.error("--output-dir must be a new directory")

    def factory(config):
        from pressb.tasks.pap.recording import PAPRecordingVectorBackend
        output = Path(config["log_dir"]).parent
        config["trajectory_recording"] = {"directory": str(output/"trajectories"),
            "env_ids": list(range(args.num_envs)), "max_episodes_per_env": 2}
        (output/"recording_config.json").write_text(json.dumps(config, indent=2)+"\n")
        backend = PAPRecordingVectorBackend(config)
        backend.add_observer(RecordingAudit(backend, output))
        return backend

    return run(args, backend_factory=factory)


if __name__ == "__main__":
    raise SystemExit(main())
