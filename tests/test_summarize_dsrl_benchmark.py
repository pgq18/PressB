"""Pure log fixtures test accounting; they are not training/speed evidence."""
import importlib.util
import json
from pathlib import Path
import random

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/summarize_dsrl_benchmark.py"
SPEC = importlib.util.spec_from_file_location("summarize_dsrl_test", SCRIPT)
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def fixture_logs(tmp_path):
    training = tmp_path / "training"
    training.mkdir()
    phases, metrics = [], []
    serial = 0
    coordinates = dict(transitions=0, updates=0, episodes=0, update_credit=0., dropped_update_credit=0.)

    def event(name, when, **values):
        row = dict(event=name, session_id="fixture", run_id="fixture-run", monotonic_seconds=when,
                   unix_seconds=1000000+when, recording_seconds=when/10000,
                   **coordinates)
        row.update(values)
        phases.append(row)
        return row

    def span(kind, method, phase, left, right, **extra):
        nonlocal serial
        serial += 1
        identity = dict(phase_id=serial, kind=kind, method=method, phase=phase, **extra)
        # Notification starts intentionally precede original-call work times.
        event("phase_start", left-.2, **identity, start_monotonic_seconds=left-100)
        return event("phase_end", right+.01, **identity, start_monotonic_seconds=left,
                     end_monotonic_seconds=right, duration_seconds=right-left,
                     outcome="completed")

    def optimizer(steps):
        return dict(state_entries=10 if steps else 0, step_entries=10 if steps else 0,
                    step_min=steps if steps else None, step_max=steps if steps else None)

    def identity(updates, actor_hash):
        return dict(updates=updates, actor_sha256=actor_hash,
                    optimizers={name:optimizer(count) for name,count in
                    (("q_optimizer",updates),("actor_optimizer",updates//2),("alpha_optimizer",updates//2))})

    config = dict(method="dsrl",mode="train",output_dir=str(training),rollout=dict(num_envs=64,max_transitions=2112,
                  fused_step_reset=False),checkpoint=dict(resume=None),
                  learner=dict(batch_size=256,policy_freq=2,auto_alpha=True,num_qs=5),
                  update=dict(learning_starts=2000,utd=1,max_backlog=1000))
    event("benchmark_start",0,run_id=None,provenance=dict(skip_checkpoints=True,output_dir=str(training)))
    span("rpc","client.call","reset",1,2,node="simulation",path="/reset")
    span("rpc_wait","_call","rpc_wait",.9,2.1,node="simulation",path="/reset")
    event("learner_initial",3,identity=identity(0,"a"*64))
    for index in range(1,34):
        start = index*10
        before_u = coordinates["updates"]
        event("batch_start",start,selected_envs=64)
        span("rpc","client.call","step",start+1,start+7,node="simulation",path="/step")
        span("rpc_wait","_call","rpc_wait",start+.9,start+8,node="simulation",path="/step")
        if index>=32:
            for left,right in ((start+2,start+3),(start+6,start+7.5)):
                span("update","_learn_one","update",left,right,updates_delta=1)
            coordinates["updates"] += 2
            coordinates["update_credit"] += 62
        coordinates["transitions"] += 64
        span("method","_save","checkpoint",start+8.2,start+8.3,skipped=True)
        event("checkpoint_skipped",start+8.25,final=False)
        event("batch_end",start+9,batch_index=index,selected_envs=64,transitions_delta=64,
              updates_delta=coordinates["updates"]-before_u,start_monotonic_seconds=start,
              outcome="completed")
        metrics.append(dict(state="running",run_id="fixture-run",transitions=coordinates["transitions"],
            updates=coordinates["updates"],episodes=0,successes=0,
            update_backlog=coordinates["update_credit"],dropped_update_credit=0.,
            reset_calls=1,reset_envs=64,elapsed_seconds=start+9))
    span("method","_save","checkpoint",340,340.1,skipped=True)
    event("checkpoint_skipped",340.05,final=True)
    event("learner_final",341,identity=identity(4,"b"*64))
    event("benchmark_end",342,outcome="completed",incomplete_batch=False,
          result=dict(state="completed",run_id="fixture-run",transitions=2112))
    status=dict(metrics[-1],last_checkpoint=None)
    status["state"]="completed"
    for name,value in (("config.json",config),("status.json",status)):
        (training/name).write_text(json.dumps(value))
    (training/"metrics.jsonl").write_text("".join(json.dumps(row)+"\n" for row in metrics))
    path=tmp_path/"phases.jsonl"
    write_phases(path,phases)
    return training,path,phases


def write_phases(path,rows):
    path.write_text("".join(json.dumps(row)+"\n" for row in rows))


def test_completed_run_uses_real_spans_and_disjoint_overlap(tmp_path):
    training,path,rows=fixture_logs(tmp_path)
    report=summary.summarize(training,path,2048,2112)
    assert report["matched_window"]["elapsed_seconds"] == 10
    assert report["matched_window"]["start_unix"] == 1000329
    assert report["full_loop"]["elapsed_seconds"] == 339
    assert report["async_partition"] == dict(rpc_only_seconds=4.,update_only_seconds=.5,
        overlap_seconds=2.,neither_seconds=3.5,duration_seconds=10.,rpc_union_seconds=6.,update_union_seconds=2.5)
    route=report["rpc_routes"]["simulation:/step"]
    assert route["count"] == 1 and route["total_seconds"] == route["mean_seconds"] == 6
    assert report["caller_waits"]["inclusive_seconds"] == pytest.approx(7.1)
    full=report["learning_work"]["full"]
    assert full["generated_update_credit"] == 128 == full["q_optimizer_steps"]+full["backlog_delta"]
    assert report["learning_work"]["matched_window"]["actor_optimizer_steps"] == 1
    assert report["learning_work"]["random_warmup_transitions"] == 2048
    assert report["audit"]["checkpoint_saves_skipped"] == 34
    assert report["workload"]["full"]["reset_envs"] == 64
    assert not report["recording_overhead"]["subtracted_from_wall_time"]
    assert str(path) in report["source_sha256"]


def test_jsonl_order_does_not_assign_concurrent_work_to_wrong_batch(tmp_path):
    training,path,rows=fixture_logs(tmp_path)
    original=summary.summarize(training,path,2048,2112)
    random.Random(4).shuffle(rows)
    write_phases(path,rows)
    changed=summary.summarize(training,path,2048,2112)
    for key in ("full_loop","matched_window","async_partition","rpc_routes","learning_work","workload"):
        assert changed[key] == original[key]


def test_clip_partial_rpc_and_union_update_intervals():
    rpc=[dict(kind="rpc",node="simulation",path="/step",start_monotonic_seconds=-1.,end_monotonic_seconds=3.)]
    updates=[dict(start_monotonic_seconds=2.,end_monotonic_seconds=5.)]
    part,routes,_=summary.timing_window(0.,4.,rpc,updates,[])
    assert part["rpc_only_seconds"] == 2 and part["overlap_seconds"] == 1
    assert part["update_only_seconds"] == 1 and part["neither_seconds"] == 0
    assert routes["simulation:/step"]["total_seconds"] == 3
    assert routes["simulation:/step"]["partial_calls"] == 1


@pytest.mark.parametrize("mutation,match",[
    ("credit","credit identity"),("missing_end","Missing phase"),
    ("missing_benchmark_end","exactly one benchmark_end"),("failure","Failed phase"),
    ("unchanged_actor","Actor did not change"),("optimizer","minimum step"),
    ("incomplete","incomplete final batch"),("checkpoint","checkpoint payload"),
    ("missing_batch","Missing or extra complete batches"),("overlapping_rpc","Overlapping rpc")])
def test_invalid_evidence_rejected(tmp_path,mutation,match):
    training,path,rows=fixture_logs(tmp_path)
    if mutation=="credit":
        next(row for row in rows if row["event"]=="batch_end" and row["transitions"]==2048)["update_credit"]+=1
    elif mutation=="missing_end":
        rows.remove(next(row for row in rows if row["event"]=="phase_end"))
    elif mutation=="missing_benchmark_end":
        rows.remove(next(row for row in rows if row["event"]=="benchmark_end"))
    elif mutation=="failure":
        next(row for row in rows if row["event"]=="phase_end")["outcome"]="failed"
    elif mutation=="unchanged_actor":
        next(row for row in rows if row["event"]=="learner_final")["identity"]["actor_sha256"]="a"*64
    elif mutation=="optimizer":
        next(row for row in rows if row["event"]=="learner_final")["identity"]["optimizers"]["q_optimizer"]["step_min"]=3
    elif mutation=="incomplete":
        next(row for row in rows if row["event"]=="benchmark_end")["incomplete_batch"]=True
    elif mutation=="checkpoint":
        (training/"unwanted.pt").write_bytes(b"checkpoint fixture")
    elif mutation=="missing_batch":
        rows.remove(next(row for row in rows if row["event"]=="batch_end"))
    elif mutation=="overlapping_rpc":
        target=next(row for row in rows if row["event"]=="phase_end" and row["kind"]=="rpc")
        target["end_monotonic_seconds"]=12
        target["duration_seconds"]=11
    write_phases(path,rows)
    with pytest.raises(summary.AuditError,match=match):
        summary.summarize(training,path,2048,2112)


def test_nonexistent_common_boundary_rejected(tmp_path):
    training,path,_=fixture_logs(tmp_path)
    with pytest.raises(summary.AuditError,match="64-transition boundaries"):
        summary.summarize(training,path,2000,2112)


@pytest.mark.parametrize("location",["phase","metrics","result","output_dir"])
def test_run_identity_mismatch_rejected(tmp_path,location):
    training,path,rows=fixture_logs(tmp_path)
    if location=="phase":
        next(row for row in rows if row["event"]=="learner_initial")["run_id"]="other-run"
    elif location=="result":
        next(row for row in rows if row["event"]=="benchmark_end")["result"]["run_id"]="other-run"
    elif location=="output_dir":
        next(row for row in rows if row["event"]=="benchmark_start")["provenance"]["output_dir"]="/other/output"
    elif location=="metrics":
        metrics=training/"metrics.jsonl"
        data=[json.loads(line) for line in metrics.read_text().splitlines()]
        data[-1]["run_id"]="other-run"
        write_phases(metrics,data)
    write_phases(path,rows)
    with pytest.raises(summary.AuditError,match="run_id|identity|output_dir"):
        summary.summarize(training,path,2048,2112)


def test_remote_output_artifact_mirror_allowed_when_provenance_agrees(tmp_path):
    training,path,rows=fixture_logs(tmp_path)
    config=json.loads((training/"config.json").read_text())
    config["output_dir"]="/remote/host/benchmark/training"
    (training/"config.json").write_text(json.dumps(config))
    next(row for row in rows if row["event"]=="benchmark_start")["provenance"]["output_dir"]=config["output_dir"]
    write_phases(path,rows)
    report=summary.summarize(training,path,2048,2112)
    assert report["provenance"]["local_artifact_copy"]
    assert report["provenance"]["recorded_training_output_dir"]==config["output_dir"]


def test_existing_local_symlink_output_identity_is_proven(tmp_path):
    training,path,rows=fixture_logs(tmp_path)
    alias=tmp_path/"training_alias"
    alias.symlink_to(training,target_is_directory=True)
    config=json.loads((training/"config.json").read_text())
    config["output_dir"]=str(alias)
    (training/"config.json").write_text(json.dumps(config))
    report=summary.summarize(training,path,2048,2112)
    binding=report["provenance"]["output_dir_binding"]
    assert binding["method"]=="existing_same_directory"
    assert binding["configured_output_dir"]==str(alias)
    assert binding["resolved_directory"]==str(training.resolve())
    assert not report["provenance"]["local_artifact_copy"]


@pytest.mark.parametrize("exists",[True,False])
def test_different_or_unverifiable_output_alias_rejected(tmp_path,exists):
    training,path,_=fixture_logs(tmp_path)
    alias=tmp_path/"other_training"
    if exists:
        alias.mkdir()
    config=json.loads((training/"config.json").read_text())
    config["output_dir"]=str(alias)
    (training/"config.json").write_text(json.dumps(config))
    with pytest.raises(summary.AuditError,match="output_dir differs"):
        summary.summarize(training,path,2048,2112)


@pytest.mark.parametrize("target",["phases","result"])
def test_cli_never_overwrites_input_or_existing_summary(tmp_path,target):
    training,path,_=fixture_logs(tmp_path)
    output=path if target=="phases" else tmp_path/"result.json"
    if target=="result":
        output.write_bytes(b"sealed prior result\n")
    before=output.read_bytes()
    with pytest.raises(FileExistsError):
        summary.main(["--training-dir",str(training),"--phases",str(path),
                      "--start-transition","2048","--expected-transitions","2112",
                      "--output",str(output)])
    assert output.read_bytes()==before
