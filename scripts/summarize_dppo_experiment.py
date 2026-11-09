#!/usr/bin/env python3
"""Audit actual DPPO/base-policy logs and write JSON, Markdown and optional plots.

Example:
  python scripts/summarize_dppo_experiment.py --baseline outputs/before \
    --after outputs/after --training outputs/train --sft outputs/sft --output outputs/report

Incomplete runs can be inspected but cannot establish a completed comparison.
Pairing requires matching logged seeds, settings and actual reset information;
equal requested run seeds alone are insufficient. No simulator/model is loaded.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def wilson_interval(successes, episodes):
    """Two-sided 95% Wilson score interval; undefined for zero observations."""
    if type(successes) is not int or type(episodes) is not int or not 0 <= successes <= episodes:
        raise ValueError("successes and episodes must be nonnegative integer counts")
    if not episodes:
        return None
    z = 1.959963984540054
    p, z2 = successes / episodes, z * z
    denominator = 1 + z2 / episodes
    centre = (p + z2 / (2 * episodes)) / denominator
    radius = z * math.sqrt(p * (1 - p) / episodes + z2 / (4 * episodes**2)) / denominator
    # At 0/n and n/n these endpoints are exactly 0 and 1. Subtracting
    # rounded centre/radius can otherwise put the interval just above 0
    # (or below 1), producing a negative plotting error-bar length.
    lower = 0. if successes == 0 else max(0., centre - radius)
    upper = 1. if successes == episodes else min(1., centre + radius)
    return [lower, upper]


def mcnemar_exact(improved, worsened):
    """Two-sided exact binomial test on discordant matched outcomes."""
    if any(type(n) is not int or n < 0 for n in (improved, worsened)):
        raise ValueError("discordant counts must be nonnegative integers")
    n = improved + worsened
    return min(1., (2 * sum(math.comb(n, k) for k in range(min(improved, worsened) + 1))) / 2**n)


def _json(raw):
    def bad(value):
        raise ValueError(f"Nonfinite JSON value: {value}")
    return json.loads(raw, parse_constant=bad)


def _jsonl(raw, name):
    rows = []
    for index, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = _json(line)
        except (ValueError, TypeError) as error:
            raise ValueError(f"{name}:{index}: invalid JSON; snapshot again after the writer finishes") from error
        if not isinstance(row, dict):
            raise ValueError(f"{name}:{index}: expected an object")
        rows.append(row)
    return rows


def load_run(path):
    """Load an auditable log snapshot; never infer outcomes from training loss."""
    path = Path(path).resolve()
    required = ("config.json", "status.json", "negotiated.json")
    raw = {name: (path / name).read_bytes() for name in required}
    for name in ("episodes.jsonl", "resets.jsonl", "metrics.jsonl"):
        if (path / name).exists():
            raw[name] = (path / name).read_bytes()
    config, status, negotiated = (_json(raw[name]) for name in required)
    episodes = _jsonl(raw.get("episodes.jsonl", b""), "episodes.jsonl")
    resets = _jsonl(raw.get("resets.jsonl", b""), "resets.jsonl")
    metrics = _jsonl(raw.get("metrics.jsonl", b""), "metrics.jsonl")
    ids, previous_transition = set(), -1
    for row in episodes:
        if not isinstance(row.get("episode_id"), str) or not row["episode_id"] or row["episode_id"] in ids:
            raise ValueError("Episode IDs must be nonempty and unique")
        ids.add(row["episode_id"])
        if type(row.get("success")) is not bool or not (row.get("terminated") is True or row.get("truncated") is True):
            raise ValueError("Only completed episodes with boolean success may enter SR")
        transition = row.get("transition")
        if type(transition) is not int or transition < previous_transition:
            raise ValueError("Episode transition counts must be monotonically nondecreasing integers")
        previous_transition = transition
    successes, count = sum(row["success"] for row in episodes), len(episodes)
    notes = []
    consistent = status.get("episodes") == count and status.get("successes") == successes
    if not consistent:
        notes.append("Status episode/success counts differ from the episode log snapshot.")
    if status.get("success_rate") is not None and count and not math.isclose(
            status["success_rate"], successes / count, abs_tol=1e-12, rel_tol=0):
        consistent = False
        notes.append("Status success_rate differs from completed episode outcomes.")
    if (path / "status.json").read_bytes() != raw["status.json"]:
        consistent = False
        notes.append("Status changed while taking this snapshot; rerun for final reporting.")
    expected = config.get("rollout", {}).get("eval_episodes") if config.get("mode") == "eval" else None
    completed = status.get("state") == "completed" and consistent
    if completed and expected is not None and count != expected:
        raise ValueError(f"Completed evaluation has {count} episodes, configured target is {expected}")
    if status.get("state") == "completed" and not consistent:
        raise ValueError("Completed status contradicts logged episode outcomes")
    if not completed:
        notes.append("Partial/stopped/failed run: this is not a completed evaluation result.")
    versions = sorted({row["policy_version"] for row in episodes if "policy_version" in row})
    if config.get("mode") == "eval" and len(versions) > 1:
        raise ValueError("Evaluation episodes contain multiple policy versions")
    return dict(path=str(path), method=config.get("method"), mode=config.get("mode"),
                state=status.get("state"), completed=completed, configured_episodes=expected,
                episodes=count, successes=successes, success_rate=successes / count if count else None,
                wilson_95=wilson_interval(successes, count), policy_versions=versions,
                transitions=status.get("transitions"), updates=status.get("updates"),
                elapsed_seconds=status.get("elapsed_seconds"), notes=notes,
                termination_counts=dict(Counter(
                    row.get("info", {}).get("termination_reason") or
                    row.get("info", {}).get("termination") or "unknown" for row in episodes)),
                sources={name: hashlib.sha256(value).hexdigest() for name, value in raw.items()},
                _config=config, _negotiated=negotiated, _episodes=episodes, _resets=resets, _metrics=metrics)


def evaluation_settings(run):
    config, negotiated = run["_config"], run["_negotiated"]
    contract = negotiated.get("contract", {})
    identity = contract.get("identity", {})
    inference, simulation = identity.get("inference", {}), identity.get("simulation", {})
    rollout = config.get("rollout", {})
    return dict(
        rollout={key: rollout.get(key) for key in ("execution_horizon", "num_envs", "reset_options",
                 "deterministic_eval", "action_bounds", "terminal_observation_mode", "fused_step_reset")},
        action=contract.get("action"), observation=contract.get("observation"),
        simulation=simulation,
        inference={key: inference.get(key) for key in ("model_id", "source_revision", "normalization_sha256",
                   "model_config_sha256", "image_preprocessing")},
        action_expert=negotiated.get("action_expert"), action_expert_config=negotiated.get("action_expert_config"))


def _differences(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        return [item for key in sorted(set(left) | set(right))
                for item in _differences(left.get(key), right.get(key), f"{prefix}.{key}".strip("."))]
    return [] if left == right else [prefix]


def _pair_rows(run):
    """Use stable rollout coordinates, never completion order or random UUIDs."""
    result = {}
    resets = {row.get("episode_id"): row for row in run["_resets"]}
    if len(resets) != len(run["_resets"]):
        raise ValueError("Duplicate reset episode IDs")
    for episode in run["_episodes"]:
        key = (episode.get("env_id"), episode.get("episode_index"))
        if any(type(value) is not int or value < 0 for value in key):
            raise ValueError("Missing logged env_id/episode_index pairing keys")
        if key in result:
            raise ValueError("Duplicate pairing key")
        if type(episode.get("reset_seed")) is not int:
            raise ValueError("Missing logged reset_seed")
        reset = resets.get(episode["episode_id"])
        if reset is None or reset.get("seed") != episode["reset_seed"] or (
                reset.get("env_id"), reset.get("episode_index")) != key:
            raise ValueError("Missing or inconsistent reset record for completed episode")
        actual = reset.get("info", {})
        if not actual:
            raise ValueError("Reset records do not contain actual initial-condition evidence")
        # Episode UUIDs are transport identities, not physical initial states.
        result[key] = dict(success=episode["success"], reset_seed=episode["reset_seed"],
                           reset_info=actual, policy_seed=episode.get("policy_seed"),
                           policy_seed_at_step0=reset.get("policy_seed_at_step0"))
    return result


def compare_runs(baseline, after):
    settings_before, settings_after = evaluation_settings(baseline), evaluation_settings(after)
    differences = _differences(settings_before, settings_after)
    reasons = []
    if any(run["mode"] != "eval" or not run["completed"] for run in (baseline, after)):
        reasons.append("Both inputs must be completed evaluations, not training or partial runs.")
    for label, settings in (("baseline", settings_before), ("after", settings_after)):
        sim = settings["simulation"]
        if not sim.get("task_config_sha256") or not sim.get("source_revision"):
            reasons.append(f"{label}: missing simulation configuration/source identity.")
        if settings["action"] is None or settings["observation"] is None:
            reasons.append(f"{label}: missing negotiated action/observation settings.")
    if differences:
        reasons.append("Evaluation settings differ: " + ", ".join(differences))
    comparable = not reasons
    result = dict(settings_equal=not differences, setting_differences=differences,
                  completed_comparison_supported=comparable, paired=False, reasons=reasons,
                  observed_delta_percentage_points=(100 * (after["success_rate"] - baseline["success_rate"])
                     if baseline["success_rate"] is not None and after["success_rate"] is not None else None))
    if not comparable:
        return result
    try:
        before_rows, after_rows = _pair_rows(baseline), _pair_rows(after)
        if not before_rows or set(before_rows) != set(after_rows):
            raise ValueError("Completed episode pairing-key sets differ or are empty")
        # The schedule marker documents how policy initial noise depends on
        # env/episode/step; deterministic evaluation still draws initial noise.
        before_schedule = baseline["_negotiated"].get("rollout_seed_schedule", baseline["_negotiated"].get("seed_schedule"))
        after_schedule = after["_negotiated"].get("rollout_seed_schedule", after["_negotiated"].get("seed_schedule"))
        shared_schedule = bool(before_schedule) and before_schedule == after_schedule and (
            baseline["_config"].get("seed") == after["_config"].get("seed"))
        for key in before_rows:
            before, after_row = before_rows[key], after_rows[key]
            if before["reset_seed"] != after_row["reset_seed"] or before["reset_info"] != after_row["reset_info"]:
                raise ValueError(f"Reset seed/actual initial conditions differ at {key}")
            explicit_seeds = type(before["policy_seed"]) is int and before["policy_seed"] == after_row["policy_seed"]
            if not shared_schedule and not explicit_seeds:
                raise ValueError("No matching logged policy-noise seed or seed_schedule evidence")
            if shared_schedule and isinstance(before_schedule, dict) and before_schedule.get("algorithm") == "blake2b":
                base_seed = baseline["_config"].get("seed")
                if before_schedule.get("base_seed") != base_seed:
                    raise ValueError("Logged seed schedule disagrees with configured base seed")
                if (before_schedule.get("digest_size"), before_schedule.get("byteorder"),
                        before_schedule.get("mask"), before_schedule.get("preimage")) != (
                        4, "little", 0x7fffffff, "{seed}:{env_id}:{episode_index}:{stream}:{step}"):
                    raise ValueError("Unsupported policy seed schedule; cannot independently verify seeds")
                preimage = f"{base_seed}:{key[0]}:{key[1]}:policy:0".encode("utf-8")
                expected = int.from_bytes(hashlib.blake2b(preimage, digest_size=4).digest(), "little") & 0x7fffffff
                if before["policy_seed_at_step0"] != expected or after_row["policy_seed_at_step0"] != expected:
                    raise ValueError(f"Logged policy seed does not match declared schedule at {key}")
        counts = Counter((before_rows[key]["success"], after_rows[key]["success"]) for key in before_rows)
        improved, worsened = counts[(False, True)], counts[(True, False)]
        result.update(paired=True, matched_episodes=len(before_rows),
                      both_success=counts[(True, True)], both_failure=counts[(False, False)],
                      improved=improved, worsened=worsened,
                      mcnemar_exact_two_sided_p=mcnemar_exact(improved, worsened),
                      pairing_evidence="same settings, stable episode keys, reset seeds, actual reset info and policy seeds/schedule")
    except ValueError as error:
        result["reasons"].append(str(error))
    return result


def _public(run):
    return {key: value for key, value in run.items() if not key.startswith("_")}


def load_sft(path):
    path = Path(path).resolve()
    if path.is_dir():
        path /= "metrics.jsonl"
    raw = path.read_bytes()
    rows = _jsonl(raw, str(path))
    if any(type(row.get("step")) is not int or row["step"] < 0 for row in rows) or any(
            b["step"] < a["step"] for a, b in zip(rows, rows[1:])):
        raise ValueError("SFT steps must be nonnegative and monotonically nondecreasing")
    config_path = path.parent / "resolved_config.json"
    config_raw = config_path.read_bytes() if config_path.exists() else None
    config = _json(config_raw) if config_raw else {}
    interval, total = config.get("eval_every"), config.get("max_steps")
    known_schedule = type(interval) is int and interval > 0 and type(total) is int
    validation = [{"step": row["step"], "validation_loss": row["validation_loss"]}
                  for row in rows if row.get("validation_loss") is not None and (
                      not known_schedule or row["step"] % interval == 0 or row["step"] == total)]
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                logged_steps=rows[-1]["step"] if rows else None, validation=validation,
                validation_points="scheduled evaluations" if known_schedule else "last reported validation values",
                resolved_config_sha256=hashlib.sha256(config_raw).hexdigest() if config_raw else None,
                metric="supervised denoising validation loss; not task success rate", _rows=rows)


def write_plots(output, baseline, after, training, sft):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    artifacts = []
    evaluations = [("Before", baseline)] + ([("After", after)] if after else [])
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for index, (label, run) in enumerate(evaluations):
        if run["success_rate"] is None:
            continue
        sr, (lo, hi) = run["success_rate"], run["wilson_95"]
        ax.errorbar(index, sr, yerr=[[max(0., sr - lo)], [max(0., hi - sr)]],
                    fmt="o", capsize=6, color=("#2474A8" if index == 0 else "#D17A22"))
        ax.annotate(f"{run['successes']}/{run['episodes']}" + (" (partial)" if not run["completed"] else ""),
                    (index, sr), xytext=(10, 0), textcoords="offset points")
    ax.set(xticks=list(range(len(evaluations))), xticklabels=[name for name, _ in evaluations],
           xlim=(-.6, len(evaluations) - .4), ylim=(-.03, 1.08), ylabel="Task success rate",
           title="Completed episode outcomes · Wilson 95% intervals")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    fig.tight_layout()
    fig.savefig(output / "evaluation_success.png", dpi=160)
    plt.close(fig)
    artifacts.append("evaluation_success.png")
    if training:
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        outcomes, transitions = [], []
        cumulative, recent = [], []
        width = training["_config"].get("rollout", {}).get("recent_window", 100)
        total = 0
        for row in training["_episodes"]:
            outcomes.append(int(row["success"]))
            total += int(row["success"])
            transitions.append(row["transition"])
            cumulative.append(total / len(outcomes))
            recent.append(sum(outcomes[-width:]) / min(width, len(outcomes)))
        axes[0].plot(transitions, recent, label=f"Recent {width} episodes", alpha=.8)
        axes[0].plot(transitions, cumulative, label="Cumulative", linewidth=1.5)
        axes[0].set(ylabel="Training rollout success", ylim=(-.03, 1.03))
        axes[0].yaxis.set_major_formatter(PercentFormatter(1))
        axes[0].legend()
        metrics = {row.get("updates"): row for row in training["_metrics"] if row.get("last_update_metrics")}
        for name in ("policy_loss", "value_loss", "approx_kl"):
            points = [(row["transitions"], row["last_update_metrics"][name]) for row in metrics.values()
                      if name in row["last_update_metrics"]]
            if points:
                axes[1].plot([x for x, _ in points], [y for _, y in points], label=name)
        if axes[1].lines:
            axes[1].legend()
        axes[1].set(xlabel="Environment transitions", ylabel="Logged optimization metrics")
        fig.suptitle("On-policy training diagnostics (not held-out evaluation)")
        fig.tight_layout()
        fig.savefig(output / "training_curves.png", dpi=160)
        plt.close(fig)
        artifacts.append("training_curves.png")
    if sft:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        rows, validation = sft["_rows"], sft["validation"]
        train = [row for row in rows if row.get("loss") is not None]
        if train:
            ax.plot([row["step"] for row in train], [row["loss"] for row in train], label="Training loss", alpha=.45)
        if validation:
            ax.plot([row["step"] for row in validation], [row["validation_loss"] for row in validation],
                    label="Validation loss", marker=".")
        if ax.lines:
            ax.legend()
        ax.set(xlabel="SFT optimizer steps", ylabel="Supervised denoising loss",
               title="SFT loss (separate from task success rate)")
        fig.tight_layout()
        fig.savefig(output / "sft_loss.png", dpi=160)
        plt.close(fig)
        artifacts.append("sft_loss.png")
    return artifacts


def write_report(baseline, output, *, after=None, training=None, sft=None, plots=True):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = dict(schema="pressb.dppo.experiment_summary.v1", baseline=_public(baseline))
    for key, value in (("after", after), ("training", training), ("sft", sft)):
        if value is not None:
            report[key] = _public(value)
    if after:
        report["comparison"] = compare_runs(baseline, after)
    report["artifacts"] = write_plots(output, baseline, after, training, sft) if plots else []
    lines = ["# DPPO experiment evidence", "", "| Run | State | Successes / episodes | SR | Wilson 95% CI |",
             "|---|---|---:|---:|---:|"]
    for label, run in (("Before", baseline), ("After", after)):
        if run is None:
            continue
        sr = "undefined" if run["success_rate"] is None else f"{run['success_rate']:.1%}"
        ci = "undefined" if run["wilson_95"] is None else "–".join(f"{v:.1%}" for v in run["wilson_95"])
        lines.append(f"| {label} | {run['state']} | {run['successes']} / {run['episodes']} | {sr} | {ci} |")
    if after:
        comparison = report["comparison"]
        lines += ["", f"Evaluation settings equal: **{comparison['settings_equal']}**. "
                  f"Completed comparison supported: **{comparison['completed_comparison_supported']}**."]
        if comparison["paired"]:
            lines += ["", f"Verified pairs: {comparison['matched_episodes']}; improved {comparison['improved']}, "
                      f"worsened {comparison['worsened']}, both succeeded {comparison['both_success']}, "
                      f"both failed {comparison['both_failure']}. Exact two-sided McNemar "
                      f"p = {comparison['mcnemar_exact_two_sided_p']:.6g}."]
        else:
            lines += ["", "Paired inference is unavailable; no McNemar result is reported."]
        for reason in comparison["reasons"]:
            lines += ["", "- " + reason]
    for label, run in (("Before", baseline), ("After", after), ("Training", training)):
        if run:
            lines += ["", f"{label} source: `{run['path']}`."]
            lines += ["- " + note for note in run["notes"]]
    lines += ["", "Training rollout SR, supervised validation loss and held-out task SR are distinct measurements. "
              "A protocol smoke test or a decrease in SFT loss does not establish an RL task-success improvement.", "",
              "Wilson intervals describe episode outcomes, not variability across independent training seeds. "
              "Source SHA256 digests and comparison diagnostics are recorded in summary.json."]
    for artifact in report["artifacts"]:
        lines += ["", f"![{artifact}]({artifact})"]
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "README.md").write_text("\n".join(lines) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--after", type=Path)
    parser.add_argument("--training", type=Path)
    parser.add_argument("--sft", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true", help="JSON/Markdown only; no matplotlib import")
    args = parser.parse_args(argv)
    report = write_report(load_run(args.baseline), args.output,
                         after=load_run(args.after) if args.after else None,
                         training=load_run(args.training) if args.training else None,
                         sft=load_sft(args.sft) if args.sft else None, plots=not args.no_plots)
    print(json.dumps({"output": str(args.output.resolve()), "baseline": report["baseline"]["success_rate"],
                      "paired": report.get("comparison", {}).get("paired", False)}, allow_nan=False))


if __name__ == "__main__":
    main()
