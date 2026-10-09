"""Read-only live-log contracts, independent of Torch, NumPy and W&B."""
import json

import pytest

from pressb.online_rl.log_monitor import LogConsistencyError, TrainingLogReader


def episode(index):
    success = index % 3 != 0
    return dict(success=success, sim_seconds=index % 7 + .25,
                executed_sim_seconds=.025, executed_action_steps=1,
                termination="target_pressed" if success else
                            "wrong_button_pressed" if index % 2 else "time_limit",
                layout={"floor": 24 + index % 12}, events=[{"discard": "large detail"}])


def metric(count, wall=0, transitions=0, updates=0):
    return dict(episodes=count, successes=sum(episode(i)["success"] for i in range(count)),
                wall_seconds=wall, transitions=transitions, updates=updates,
                critic_loss=.125, timing_seconds={"simulation_step": 3.},
                rpc_counts={"simulation_step": 12})


def lines(path, rows, *, mode="w"):
    with path.open(mode) as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def status(path, row):
    path.write_text(json.dumps(row))


def test_backfill_and_status_use_each_completed_prefix_and_whole_episode_duration(tmp_path):
    completed = [episode(i) for i in range(1800)]
    rows = [metric(n, index * 65, index * 100, index * 80)
            for index, n in enumerate([0, 75, 101, 1005, 1100])]
    latest = metric(1150, 300, 600, 500)
    lines(tmp_path / "episodes.jsonl", completed)
    lines(tmp_path / "metrics.jsonl", rows)
    status(tmp_path / "status.json", latest)
    reader = TrainingLogReader(tmp_path)
    history, current = reader.poll()
    assert len(history) == len(rows)
    for row, original in zip(history + [current], rows + [latest]):
        n = original["episodes"]
        for window in [100, 1000]:
            subset = completed[max(0, n - window):n]
            expected = sum(e["success"] for e in subset) / len(subset) if subset else None
            assert row[f"recent_count_{window}"] == len(subset)
            assert row[f"recent_sr_{window}"] == expected
        subset = completed[max(0, n - 1000):n]
        duration = sum(e["sim_seconds"] for e in subset) / len(subset) if subset else None
        assert row["episode_seconds_recent_1000"] == duration
        for reason in ["target_pressed", "wrong_button_pressed", "time_limit"]:
            frequency = sum(e["termination"] == reason for e in subset) / len(subset) if subset else None
            assert row[f"termination_recent_{reason}"] == frequency
        for floor in range(24, 36):
            floor_rows = [e for e in subset if e["layout"]["floor"] == floor]
            expected = sum(e["success"] for e in floor_rows) / len(floor_rows) if floor_rows else None
            assert row[f"floor_recent_sr_{floor}"] == expected
            assert row[f"floor_recent_count_{floor}"] == len(floor_rows)
        for key in original:
            assert row[key] == original[key]
    assert history[1]["episode_seconds_recent_1000"] > 1
    assert reader.poll() == ([], current)
    assert all(len(record) == 4 for record in reader.completed)


def test_incomplete_lines_are_held_until_newline_and_never_duplicated(tmp_path):
    first, second = metric(1, 10, 100, 50), metric(2, 20, 200, 150)
    lines(tmp_path / "episodes.jsonl", [episode(0)])
    lines(tmp_path / "metrics.jsonl", [first])
    with (tmp_path / "episodes.jsonl").open("a") as stream:
        stream.write(json.dumps(episode(1)))
    with (tmp_path / "metrics.jsonl").open("a") as stream:
        stream.write(json.dumps(second))
    reader = TrainingLogReader(tmp_path)
    history, current = reader.poll()
    assert [row["episodes"] for row in history] == [1] and current is None
    assert reader.poll() == ([], None)
    with (tmp_path / "metrics.jsonl").open("a") as stream:
        stream.write("\n")
    status(tmp_path / "status.json", second)
    # A complete metrics/status snapshot can precede its episode append.
    assert reader.poll() == ([], None)
    with (tmp_path / "episodes.jsonl").open("a") as stream:
        stream.write("\n")
    history, current = reader.poll()
    assert [row["episodes"] for row in history] == [2]
    assert history[0]["recent_sr_100"] == current["recent_sr_100"] == .5
    assert reader.poll() == ([], current)


@pytest.mark.parametrize("location", ["history", "status"])
def test_success_counter_must_match_its_exact_prefix(tmp_path, location):
    lines(tmp_path / "episodes.jsonl", [episode(i) for i in range(5)])
    row = metric(2)
    row["successes"] = 2  # Prefix has one success; the tail cannot repair it.
    if location == "history":
        lines(tmp_path / "metrics.jsonl", [row])
    else:
        status(tmp_path / "status.json", row)
    with pytest.raises(LogConsistencyError, match="Episode prefix mismatch"):
        TrainingLogReader(tmp_path).poll()


def test_rates_use_actual_historical_deltas_with_at_least_sixty_seconds(tmp_path):
    times, transitions, updates = [10, 25, 70, 88, 150], [100, 250, 900, 1000, 1700], [0, 10, 90, 100, 170]
    rows = [metric(0, *values) for values in zip(times, transitions, updates)]
    lines(tmp_path / "metrics.jsonl", rows)
    status(tmp_path / "status.json", metric(0, 155, 1850, 190))
    history, latest = TrainingLogReader(tmp_path).poll()
    assert history[0]["transitions_per_second_recent_60s"] is None
    assert history[0]["speed_window_seconds"] == 0
    for index, base in [(1, 0), (2, 0), (3, 1), (4, 3)]:
        span = times[index] - times[base]
        assert history[index]["speed_window_seconds"] == span
        assert history[index]["transitions_per_second_recent_60s"] == (transitions[index] - transitions[base]) / span
        assert history[index]["updates_per_second_recent_60s"] == (updates[index] - updates[base]) / span
    assert latest["speed_window_seconds"] == 67
    assert latest["transitions_per_second_recent_60s"] == 850 / 67
    assert latest["updates_per_second_recent_60s"] == 90 / 67


def test_status_older_than_last_history_uses_only_earlier_rate_baseline(tmp_path):
    lines(tmp_path / "metrics.jsonl", [metric(0, 10, 100, 20), metric(0, 80, 800, 300)])
    status(tmp_path / "status.json", metric(0, 75, 750, 280))
    _, current = TrainingLogReader(tmp_path).poll()
    assert current["speed_window_seconds"] == 65
    assert current["transitions_per_second_recent_60s"] == 10


def test_missing_history_and_zero_episodes_are_defined(tmp_path):
    reader = TrainingLogReader(tmp_path)
    assert reader.poll() == ([], None)
    status(tmp_path / "status.json", metric(0))
    history, current = reader.poll()
    assert history == []
    assert current["recent_count_100"] == current["recent_count_1000"] == 0
    assert current["recent_sr_100"] is current["episode_seconds_recent_1000"] is None
    assert current["floor_recent_count_24"] == 0 and current["floor_recent_sr_24"] is None
    assert current["transitions_per_second_recent_60s"] is None


@pytest.mark.parametrize("filename", ["metrics.jsonl", "episodes.jsonl"])
@pytest.mark.parametrize("line", ["{broken}\n", "[]\n", "\n"])
def test_complete_malformed_jsonl_is_never_silently_skipped(tmp_path, filename, line):
    (tmp_path / filename).write_text(line)
    with pytest.raises(LogConsistencyError):
        TrainingLogReader(tmp_path).poll()


@pytest.mark.parametrize("mutation", ["truncate", "replace", "same_size_rewrite", "disappear"])
def test_log_truncation_and_rotation_cannot_silently_duplicate_history(tmp_path, mutation):
    path = tmp_path / "metrics.jsonl"
    lines(path, [metric(0, 10, 100, 20)])
    reader = TrainingLogReader(tmp_path)
    assert len(reader.poll()[0]) == 1
    if mutation == "truncate":
        path.write_text("")
    elif mutation == "replace":
        replacement = tmp_path / "replacement.jsonl"
        replacement.write_text(path.read_text())
        replacement.replace(path)
    elif mutation == "same_size_rewrite":
        path.write_text(path.read_text().replace('"simulation_step": 12', '"simulation_step": 13'))
    else:
        path.unlink()
    with pytest.raises(LogConsistencyError):
        reader.poll()


def test_history_counter_reset_is_reported(tmp_path):
    lines(tmp_path / "metrics.jsonl", [metric(0, 20, 200, 20), metric(0, 30, 100, 30)])
    with pytest.raises(LogConsistencyError, match="counters decreased"):
        TrainingLogReader(tmp_path).poll()
