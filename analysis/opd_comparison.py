# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compare OPD runs by cost, response position, and paired benchmark accuracy."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from verl.trainer.ppo.metric_archive import METRIC_COMMIT_FILENAME, inspect_metric_archive
from verl.utils.evaluation.math_benchmark import MATH5_BENCHMARKS

POSITION_METRIC = re.compile(r"opd/student_only/(gap_abs|top16)_bin/(\d+)_(\d+)/(sum|count)")
TIMING_METRICS = {"timing_s/step", "timing_s/testing", "timing_s/test_benchmarks"}


def read_training_metrics(root: Path) -> tuple[dict, dict[int, dict[str, float]]]:
    manifest = inspect_metric_archive(root)
    if manifest is None:
        raise ValueError(f"No metric archive at {root}.")
    steps = defaultdict(dict)
    for shard in manifest["shards"]:
        if shard.get("checkpoint_step") is None or shard["state"] == "ORPHANED":
            continue
        path = root / "shards" / Path(shard["path"]).name
        table = pq.read_table(
            path,
            columns=["global_step", "metric_key", "value"],
            filters=[("event_kind", "=", "train")],
        )
        for step, key, value in zip(*(column.to_pylist() for column in table.columns), strict=True):
            if key not in TIMING_METRICS and not key.startswith(("opd_cost/", "opd/student_only/")):
                continue
            if key in steps[step] or not math.isfinite(value):
                raise ValueError(f"Duplicate or invalid metric {key!r} at step {step} in {root}.")
            steps[step][key] = value
    if not steps:
        raise ValueError(f"No committed training metrics at {root}.")
    return manifest, dict(steps)


def _cumulative_seconds(steps: dict[int, dict], through: int, devices: int) -> dict[int, float]:
    cumulative = {0: 0.0}
    for step in range(1, through + 1):
        metrics = steps.get(step, {})
        duration = metrics.get("timing_s/step")
        if duration is None or not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"Missing or invalid timing_s/step at step {step}.")
        if metrics.get("opd_cost/allocated_devices") != devices:
            raise ValueError(f"Expected {devices} allocated devices at step {step}.")
        cumulative[step] = cumulative[step - 1] + duration
    return cumulative


def _committed_checkpoints(root: Path, logical_run_id: str) -> dict[int, str]:
    checkpoints = {}
    for path in root.glob("global_step_*"):
        match = re.fullmatch(r"global_step_(\d+)", path.name)
        marker_path = path / METRIC_COMMIT_FILENAME
        if not match or not marker_path.is_file() or not (path / "actor").is_dir():
            continue
        step = int(match[1])
        marker = json.loads(marker_path.read_text())
        if marker["logical_run_id"] != logical_run_id or marker["checkpoint_step"] != step:
            raise ValueError(f"Checkpoint commit identity mismatch at {path}.")
        checkpoints[step] = str(path.resolve())
    return checkpoints


def _cost_summary(steps: dict[int, dict], cumulative: dict[int, float], step: int, devices: int) -> dict:
    usage = Counter()
    for index in range(1, step + 1):
        usage.update(
            {
                key: val
                for key, val in steps[index].items()
                if key.startswith("opd_cost/") and "/allocated_devices" not in key
            }
        )
    return dict(
        step=step,
        train_seconds=cumulative[step],
        npu_hours=cumulative[step] * devices / 3600,
        evaluation_seconds=sum(
            steps[index].get(key, 0.0)
            for index in range(1, step + 1)
            for key in ("timing_s/testing", "timing_s/test_benchmarks")
        ),
        usage=dict(sorted(usage.items())),
    )


def budget_comparison(baseline: Path, candidate: Path, checkpoint_root: Path, *, step: int, devices: int) -> dict:
    if step <= 0 or devices <= 0:
        raise ValueError("Budget comparison requires a positive step and device count.")
    _, reference = read_training_metrics(baseline)
    manifest, trial = read_training_metrics(candidate)
    checkpoints = _committed_checkpoints(checkpoint_root, manifest["logical_run_id"])
    reference_time = _cumulative_seconds(reference, step, devices)
    budget = reference_time[step]
    trial_time = _cumulative_seconds(trial, max(trial), devices)
    eligible = [index for index in checkpoints if index > 0 and trial_time.get(index, math.inf) <= budget]
    if not eligible:
        raise ValueError("No committed candidate checkpoint with measured cost within the budget.")
    matched = max(eligible)
    baseline_cost = _cost_summary(reference, reference_time, step, devices)
    matched_cost = _cost_summary(trial, trial_time, matched, devices)
    fixed_cost = _cost_summary(trial, trial_time, step, devices) if step in trial_time else None
    result = dict(
        baseline=baseline_cost,
        candidate_fixed_step=fixed_cost,
        candidate_budget_checkpoint=checkpoints[matched],
        candidate_budget_cost=matched_cost,
        unused_npu_hours=(budget - trial_time[matched]) * devices / 3600,
    )
    if fixed_cost is not None:
        result["fixed_step_overhead"] = dict(
            train_seconds=trial_time[step] - budget,
            npu_hours=(trial_time[step] - budget) * devices / 3600,
            fraction=trial_time[step] / budget - 1,
        )
    last_step = max(trial_time)
    if step <= last_step < 434 and trial_time[last_step] < budget and last_step in checkpoints:
        start = max(last_step - 32, 0)
        seconds_per_step = (trial_time[last_step] - trial_time[start]) / (last_step - start)
        estimate = last_step + (budget - trial_time[last_step]) / seconds_per_step
        result["estimated_resume_target"] = min(434, 10 * math.ceil(estimate / 10))
    return result


def _position_bins(metrics: dict[str, float], name: str) -> dict[tuple[int, int], tuple[float, float]]:
    bins = defaultdict(dict)
    for key, value in metrics.items():
        match = POSITION_METRIC.fullmatch(key)
        if match and match[1] == name:
            bins[int(match[2]), int(match[3])][match[4]] = value
    if any(fields.keys() != {"sum", "count"} for fields in bins.values()):
        raise ValueError(f"Incomplete position bins for {name}.")
    return {interval: (fields["sum"], fields["count"]) for interval, fields in bins.items()}


def matched_diagnostics(baseline: dict[int, dict], candidate: dict[int, dict], *, through: int) -> dict:
    result = {}
    for name in ("gap_abs", "top16"):
        rows = []
        for step in sorted(baseline.keys() & candidate.keys()):
            if step > through:
                break
            ref = _position_bins(baseline[step], name)
            trial = _position_bins(candidate[step], name)
            common = sorted(key for key in ref.keys() & trial.keys() if min(ref[key][1], trial[key][1]) > 0)
            if not common:
                continue
            ref_count = sum(ref[key][1] for key in common)
            rows.append(
                dict(
                    step=step,
                    baseline=sum(ref[key][0] for key in common) / ref_count,
                    candidate=sum(trial[key][0] * ref[key][1] / trial[key][1] for key in common) / ref_count,
                    common_bins=len(common),
                    baseline_tokens=int(ref_count),
                    candidate_tokens=int(sum(trial[key][1] for key in common)),
                    baseline_total_tokens=int(sum(count for _, count in ref.values())),
                    candidate_total_tokens=int(sum(count for _, count in trial.values())),
                )
            )
        if not rows:
            raise ValueError(f"No shared steps and response positions for {name}.")
        summary = {}
        for method in ("baseline", "candidate"):
            if name == "gap_abs" and len(rows) > 1:
                values = [row[method] for row in rows]
                steps = [row["step"] for row in rows]
                summary[method] = float(np.trapz(values, steps) / (steps[-1] - steps[0]))
            else:
                summary[method] = statistics.mean(row[method] for row in rows)
        summary["delta"] = summary["candidate"] - summary["baseline"]
        if name == "gap_abs" and summary["baseline"] > 0:
            summary["relative_reduction"] = -summary["delta"] / summary["baseline"]
        if name == "top16":
            summary["delta_percentage_points"] = summary["delta"] * 100
        result[name] = dict(
            aggregation="normalized_step_auc" if name == "gap_abs" and len(rows) > 1 else "mean_over_steps",
            summary=summary,
            steps=rows,
        )
    return result


def read_evaluation(path: Path) -> tuple[dict, dict[str, float]]:
    data = json.loads(path.read_text())
    protocol = data["protocol"]
    if (
        data["schema_version"] != 1
        or protocol["suite"] != "math5"
        or tuple(protocol["benchmarks"]) != MATH5_BENCHMARKS
        or protocol["rollout_n"] != 16
    ):
        raise ValueError(f"Expected math5 mean@16 evaluation records in {path}.")
    expected = set(protocol["question_hashes"])
    draws = defaultdict(dict)
    for record in data["records"]:
        uid, index, acc = (record[key] for key in ("uid", "draw", "acc"))
        if uid not in expected or uid.split(":", 1)[0] != record["benchmark"]:
            raise ValueError(f"Unexpected evaluation question {uid!r} in {path}.")
        if type(index) is not int or index not in range(16) or index in draws[uid] or acc not in (0, 1):
            raise ValueError(f"Invalid or duplicate draw for {uid!r} in {path}.")
        draws[uid][index] = acc
    if draws.keys() != expected or any(len(values) != 16 for values in draws.values()):
        raise ValueError(f"Missing questions or draws in {path}.")
    return protocol, {uid: sum(values.values()) / 16 for uid, values in draws.items()}


def paired_evaluation(baseline: Path, candidate: Path, *, resamples: int, seed: int) -> dict:
    protocol, reference = read_evaluation(baseline)
    trial_protocol, trial = read_evaluation(candidate)
    if protocol != trial_protocol:
        raise ValueError("Paired evaluation requires identical question sets, prompts, and sampling protocols.")
    if resamples < 1:
        raise ValueError("Bootstrap resamples must be positive.")
    rng = np.random.default_rng(seed)
    result = {}
    samples = []
    for benchmark in MATH5_BENCHMARKS:
        uids = sorted(uid for uid in reference if uid.startswith(benchmark + ":"))
        if not uids:
            raise ValueError(f"Evaluation contains no questions for {benchmark}.")
        ref_values = np.asarray([reference[uid] for uid in uids])
        trial_values = np.asarray([trial[uid] for uid in uids])
        changes = trial_values - ref_values
        draws = changes[rng.integers(len(changes), size=(resamples, len(changes)))].mean(axis=1)
        samples.append(draws)
        result[benchmark] = dict(
            questions=len(uids),
            baseline=float(ref_values.mean()),
            candidate=float(trial_values.mean()),
            delta=float(changes.mean()),
            ci95=np.quantile(draws, [0.025, 0.975]).tolist(),
        )
    result["macro"] = {
        key: statistics.mean(row[key] for row in result.values()) for key in ("baseline", "candidate", "delta")
    }
    result["macro"]["ci95"] = np.quantile(np.mean(samples, axis=0), [0.025, 0.975]).tolist()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    budget = commands.add_parser("budget", help="Select a checkpoint using measured cost.")
    diagnostics = commands.add_parser("diagnostics", help="Compare student-only metrics on shared positions.")
    for command in (budget, diagnostics):
        command.add_argument("baseline_archive", type=Path)
        command.add_argument("candidate_archive", type=Path)
        command.add_argument("--step", type=int, default=217)
    budget.add_argument("candidate_checkpoints", type=Path)
    budget.add_argument("--devices", type=int, default=40)
    paired = commands.add_parser("paired-eval", help="Compute paired intervals and optional between-run SDs.")
    paired.add_argument("--pair", nargs=2, action="append", type=Path, required=True, metavar=("OPD", "COMPARISON"))
    paired.add_argument("--resamples", type=int, default=10_000)
    paired.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "budget":
        result = budget_comparison(
            args.baseline_archive,
            args.candidate_archive,
            args.candidate_checkpoints,
            step=args.step,
            devices=args.devices,
        )
    elif args.command == "diagnostics":
        _, baseline = read_training_metrics(args.baseline_archive)
        _, candidate = read_training_metrics(args.candidate_archive)
        result = matched_diagnostics(baseline, candidate, through=args.step)
    else:
        pairs = [paired_evaluation(*pair, resamples=args.resamples, seed=args.seed) for pair in args.pair]
        result = dict(pairs=pairs)
        if len(pairs) > 1:
            result["between_run_sample_sd"] = {
                name: {
                    key: statistics.stdev(pair[name][key] for pair in pairs)
                    for key in ("baseline", "candidate", "delta")
                }
                for name in pairs[0]
            }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
