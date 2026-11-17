#!/usr/bin/env python3
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

"""Migrate selected DMPD runs from local logs to SwanLab and a private HF dataset."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import gzip
import json
import math
import re
import sys
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from zoneinfo import ZoneInfo

SHARED_MOUNT = Path("/opt/huawei/dataset/jnguan")
DEFAULT_ROOT = Path("/home/ma-user/work/dataset/jnguan/code/DMPD")
DEFAULT_MIRROR = Path(__file__).resolve().parents[1] / "analysis_artifacts/jubgjf/DMPD/runs"
DEFAULT_SINCE = dt.date(2026, 11, 1)
TRAINING_STEP = "training/global_step"
TIMESTAMP = re.compile(r"(\d{8}-\d{6})$")
EVALUATION_FILE = re.compile(r"([a-zA-Z0-9][a-zA-Z0-9_-]*)--step_(\d+)\.json")
EVALUATION_MARKER = "EVAL_RESULT_JSON="

# Main table, 7B comparison, ablations, and sensitivity runs. Names also locate
# legacy output directories that predate the metric archive.
PAPER_RUNS = {
    # DeepSeek and Nemotron 1.5B: OPD, ATI-OPD, and ESR.
    "9bbeb176889b42aa847665d03274fd68": "vanilla-opd-20260824-124656",
    "802c696019c1449aa3328bca0e1806eb": "dmpd--train_on_teacher--loss_opd--prune_future-20260902-085200",
    "48b5708e981c4d17aacc398ffa8d043c": "topd--horizon_100-20260922-164001",
    "040b92fbe33a41e6be75ffcaf8f07f27": "vanilla-opd-20260901-020541",
    "d60c26f9321046f181400c86ee096e79": "dmpd--train_on_teacher--loss_opd--prune_future-20260904-180001",
    "8ea23b41d0ef497abe3ead9d05413655": "topd--horizon_100-20260922-162100",
    # DeepSeek 7B.
    "50042337c98c461e9400930a6843eb5a": "vanilla-opd-20260914-103255",
    "a704eaf642fc42979843c533cfaae26a": "dmpd--train_on_teacher--loss_opd--prune_future-20260914-130458",
    # Masked, SFT, Full M=1 and M=4, Prune, and Skip incorrect.
    "39ce8dd19e6b4451bd5175609358110d": "dmpd--prune_future-20260917-180951",
    "f7c7a430d8de464992d9de225413b3a1": "dmpd--train_on_teacher--loss_sft--coef_0.1--prune_future-20260917-124323",
    "38457043f2534dbbb700eeb7ce4b6542": "dmpd--train_on_teacher--loss_opd--max_int_1-20260917-175345",
    "08806c2c3e8547449299f0fc3e62c766": "dmpd--train_on_teacher--loss_opd-20260915-114359",
    "c7c88aa34a4a4097b73968039a8ccc89": "dmpd--train_on_teacher--loss_opd--prune-20260824-124704",
    "323d9444cf2246409d023fed76381653": "dmpd--skip_correct--train_on_teacher--loss_opd--prune_future-20260901-170101",
    # Teacher segment length and trigger threshold.
    "7187a1dc4bec46e3a37ff984dec7d1b2": (
        "dmpd--teacher_len_512--train_on_teacher--loss_opd--prune_future-20260914-173437"
    ),
    "6ca6d5692c294ffaa9c0ba928bafa993": (
        "dmpd--teacher_len_1024--train_on_teacher--loss_opd--prune_future-20260914-130830"
    ),
    "e720fe93a0c94f6da8f28feaca7639f0": "dmpd--k3_0.1--train_on_teacher--loss_opd--prune_future-20260915-085701",
    "8591eaa49198422ab8268632040b559d": "dmpd--k3_0.3--train_on_teacher--loss_opd--prune_future-20260915-114405",
}

EVALUATION_FILES = ["evaluations/*.json"]
OUTPUT_FILES = [
    "diagnostics/manifest.json",
    "diagnostics/sessions/*.json",
    "diagnostics/shards/*",
    "diagnostics/attachments/*",
    *EVALUATION_FILES,
    "latest_checkpointed_iteration.txt",
    "global_step_*/metrics_commit.json",
]
MIRROR_FILES = ["run.json", "history/*.jsonl.gz", "files/media/*", "logged_artifacts/training-diagnostics/*"]
History = dict[int, dict[str, float]]


@dataclass(frozen=True)
class Experiment:
    run_id: str
    name: str
    output: Path | None = None
    mirror: Path | None = None


class ArchiveBusyError(RuntimeError):
    """The training driver still holds the archive's writer lock."""


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _experiment_date(name: str, manifest: dict) -> dt.date | None:
    match = TIMESTAMP.search(name)
    if match:
        return dt.datetime.strptime(match[1], "%Y%m%d-%H%M%S").date()
    if created_at := manifest.get("created_at"):
        timestamp = dt.datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        return timestamp.astimezone(ZoneInfo("Asia/Shanghai")).date()
    return None


def discover_experiments(root: Path, mirror_root: Path, *, since: dt.date, scope: str) -> list[Experiment]:
    """Select exact historical IDs and new experiment manifests by original launch date."""
    outputs = root / "outputs"
    selected = {}
    if scope != "recent":
        for run_id, name in PAPER_RUNS.items():
            output = outputs / name
            selected[run_id] = Experiment(run_id, name, output if output.is_dir() else None)

    manifest_dirs: dict[str, Path] = {}
    for manifest_path in sorted(outputs.glob("*/diagnostics/manifest.json")):
        manifest = _read_json(manifest_path)
        run_id = manifest["logical_run_id"]
        output = manifest_path.parents[1]
        name = manifest.get("experiment") or output.name
        if run_id in PAPER_RUNS:
            if scope == "recent":
                continue
        else:
            if scope == "paper":
                continue
            launched = _experiment_date(name, manifest)
            if launched is None or launched < since:
                continue
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}", run_id):
            raise ValueError(f"Invalid run ID {run_id!r} in {manifest_path}")
        if run_id in manifest_dirs:
            raise ValueError(f"Run {run_id} has two output directories: {manifest_dirs[run_id]} and {output}")
        manifest_dirs[run_id] = output
        selected[run_id] = Experiment(run_id, name, output)

    result = []
    for experiment in selected.values():
        mirror = mirror_root / experiment.run_id
        result.append(
            Experiment(
                experiment.run_id,
                experiment.name,
                experiment.output,
                mirror if (mirror / "run.json").is_file() else None,
            )
        )
    return result


def _indexed_segments(directories: list[Path]) -> dict[str, list[Path]]:
    segments = defaultdict(list)
    for directory in dict.fromkeys(directories):
        for pattern in ("offline-run-*/run-*.wandb", "run-*/run-*.wandb"):
            for path in directory.glob(pattern):
                if path.is_file():
                    segments[path.stem.removeprefix("run-")].append(path.resolve())
    return {run_id: sorted(set(paths), key=lambda path: path.parent.name) for run_id, paths in segments.items()}


def _session_segments(run_id: str, manifest: dict, indexed: list[Path], root: Path) -> list[Path]:
    if not manifest.get("sessions"):
        return indexed

    # Computation and jump hosts see the same repository under different mounts.
    compute_root = SHARED_MOUNT / "code/DMPD"
    by_directory = {path.parent.name: path for path in indexed}
    paths = []
    sessions = sorted(
        manifest["sessions"].values(),
        key=lambda session: (session.get("session_order", 0), session.get("started_at", "")),
    )
    for session in sessions:
        if session.get("wandb_mode") == "disabled":
            continue
        recorded = session.get("wandb_sync_file")
        if not recorded and session.get("wandb_sync_dir"):
            recorded = Path(session["wandb_sync_dir"]) / f"run-{run_id}.wandb"
        if not recorded:
            continue
        path = Path(recorded)
        if path.is_relative_to(compute_root):
            path = root / path.relative_to(compute_root)
        if not path.is_file():
            path = by_directory.get(path.parent.name, path)
        if path.name == f"run-{run_id}.wandb" and path.is_file():
            paths.append(path.resolve())

    # Preserve old offline files that predate session registration.
    registered = set(paths)
    return [path for path in indexed if path not in registered] + list(dict.fromkeys(paths))


@contextmanager
def _snapshot_lock(output: Path | None) -> Iterator[None]:
    """Read a stable archive, including interrupted runs whose writer has exited."""
    lock = output / "diagnostics/.writer.lock" if output is not None else None
    if lock is None or not lock.is_file():
        yield
        return
    with lock.open("rb") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ArchiveBusyError(lock) from exc
        yield


def _add_row(history: History, row: dict) -> None:
    step = row.get(TRAINING_STEP, row.get("_step"))
    if not isinstance(step, int | float) or not math.isfinite(step) or step < 0 or int(step) != step:
        return
    metrics = {
        key: float(value)
        for key, value in row.items()
        if key and not key.startswith("_") and isinstance(value, int | float) and math.isfinite(value)
    }
    if metrics:
        history.setdefault(int(step), {}).update(metrics)


def _read_wandb_file(path: Path, run_id: str, history: History, config: dict) -> None:
    from swanlab.converter.wb.utils import proto_items_to_dict
    from swanlab.converter.wb.wbdatastore import DataStoreError, WBDataStoreReader
    from wandb.proto.wandb_internal_pb2 import Record

    with WBDataStoreReader() as reader:
        try:
            reader.open(path)
            while raw := reader.scan():
                record = Record.FromString(raw)
                kind = record.WhichOneof("record_type")
                if kind == "run":
                    if record.run.run_id and record.run.run_id != run_id:
                        raise ValueError(f"W&B run ID mismatch in {path}: {record.run.run_id}")
                    config.update(proto_items_to_dict(record.run.config.update))
                elif kind == "config":
                    config.update(proto_items_to_dict(record.config.update))
                elif kind == "history":
                    row = {
                        item.key or "/".join(item.nested_key): json.loads(item.value_json)
                        for item in record.history.item
                        if item.value_json
                    }
                    _add_row(history, row)
        except DataStoreError as exc:
            print(f"  W&B read stopped in {path}: {exc}. Using complete records and any committed archive.")


def _read_mirror(mirror: Path, run_id: str, history: History) -> dict:
    metadata = _read_json(mirror / "run.json")
    if metadata["id"] != run_id:
        raise ValueError(f"Run ID mismatch in {mirror / 'run.json'}")
    for path in sorted((mirror / "history").glob("steps_*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                _add_row(history, json.loads(line))
    return metadata.get("config") or {}


def _parquet_history(paths: list[Path], *, steps: set[int] | None = None) -> History:
    if not paths or (steps is not None and not steps):
        return {}

    import pyarrow.parquet as pq

    history: History = {}
    filters: list[tuple[str, str, str | list[int]]] = [("delivery", "=", "live")]
    if steps is not None:
        filters.append(("global_step", "in", sorted(steps)))
    for path in paths:
        table = pq.read_table(
            path,
            columns=["global_step", "metric_key", "value"],
            filters=filters,
        )
        for step, key, value in zip(*(column.to_pylist() for column in table.columns), strict=True):
            if key.startswith("_") or not math.isfinite(value):
                continue
            row = history.setdefault(step, {})
            if key in row and row[key] != value:
                raise ValueError(f"Conflicting archived metric {key!r} at step {step} in {path}")
            row[key] = value
    return history


def _mirrored_parquet(experiment: Experiment) -> list[Path]:
    if experiment.mirror is None:
        return []
    archive = experiment.mirror / "logged_artifacts/training-diagnostics"
    path = archive / "manifest.json"
    if not path.is_file():
        return []
    manifest = _read_json(path)
    if manifest["run_id"] != experiment.run_id:
        raise ValueError(f"Run ID mismatch in {path}")
    paths = []
    for version in manifest["versions"]:
        name = version["version"]
        for entry in version["entries"]:
            relative = Path(entry)
            if not re.fullmatch(r"v\d+", name) or relative.parts != ("shards", relative.name):
                raise ValueError(f"Unexpected W&B artifact path {name}/{entry} in {path}")
            paths.append(archive / name / relative)
    return paths


def read_history(experiment: Experiment, manifest: dict, segments: list[Path]) -> tuple[History, dict]:
    """Merge native logs and cached history, then reconcile against committed rows."""
    history: History = {}
    config = _read_mirror(experiment.mirror, experiment.run_id, history) if experiment.mirror else {}
    for path in segments:
        _read_wandb_file(path, experiment.run_id, history, config)

    missing = _missing_training_steps(experiment, history)
    if experiment.output is not None and manifest:
        paths = [
            experiment.output / "diagnostics/shards" / Path(shard["path"]).name
            for shard in manifest.get("shards", [])
            if shard.get("checkpoint_step") is not None and shard["state"] != "ORPHANED"
        ]
        committed = _parquet_history(paths)
        if committed:
            # A rollback may leave old values at steps the resumed session did
            # not emit. Replace the archived part of the curve, preserving only
            # the native prefix from before balanced logging was first enabled.
            legacy_step = manifest.get("legacy_before_step") or 0
            history = {step: row for step, row in history.items() if step <= legacy_step}
            history.update(committed)
            missing = {step for step in missing if step <= legacy_step}
    if missing:
        for step, row in _parquet_history(_mirrored_parquet(experiment), steps=missing).items():
            history[step] = {**row, **history.get(step, {})}
    return history, config


def _missing_training_steps(experiment: Experiment, history: History) -> set[int]:
    training_steps = {step for step, row in history.items() if step > 0 and TRAINING_STEP in row}
    end = max(training_steps, default=0)
    if experiment.output:
        tracker = experiment.output / "latest_checkpointed_iteration.txt"
        if tracker.is_file():
            end = int(tracker.read_text(encoding="utf-8").strip())
            return set(range(1, end + 1)) - training_steps
    if experiment.mirror:
        summary = _read_json(experiment.mirror / "run.json").get("summary") or {}
        step = summary.get(TRAINING_STEP)
        if isinstance(step, int):
            end = max(end, step)
    return set(range(1, end + 1)) - training_steps


def _checkpoint_identity(path: str) -> tuple[str, int]:
    parts = Path(path).parts
    for index, part in enumerate(parts[:-2]):
        if part == "outputs" and (match := re.fullmatch(r"global_step_(\d+)", parts[index + 2])):
            return parts[index + 1], int(match[1])
    raise ValueError(f"Expected outputs/<experiment>/global_step_<step> in checkpoint {path!r}")


def _summary_suite(config: dict) -> str:
    suite = config.get("suite") or ("all" if config.get("eval_all") else "hmmt")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", suite):
        raise ValueError(f"Invalid evaluation suite {suite!r}")
    return suite


def _read_evaluation_stdout(paths: list[Path]) -> dict[str, dict[tuple[str, int], dict]]:
    summaries: dict[str, dict[tuple[str, int], dict]] = defaultdict(dict)
    files = set()
    for path in paths:
        path = path.expanduser()
        if path.is_dir():
            files.update(
                file
                for file in path.rglob("*")
                if file.is_file() and (file.name == "stdout" or file.suffix in {".log", ".out", ".txt", ".gz"})
            )
        elif path.is_file():
            files.add(path)
        else:
            raise ValueError(f"Evaluation log path does not exist: {path}")
    for path in sorted(files):
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, 1):
                _, marker, text = line.partition(EVALUATION_MARKER)
                if not marker or not text.startswith("{"):
                    continue
                try:
                    result = json.loads(text)
                    config = result["config"]
                    name, step = _checkpoint_identity(config["model_path"])
                    suite = _summary_suite(config)
                    if not result["metrics"] or set(result["benchmarks"]) != set(config["benchmarks"]):
                        raise ValueError("Incomplete benchmark summary")
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid evaluation result at {path}:{line_number}: {exc}") from exc
                previous = summaries[name].setdefault((suite, step), result)
                if previous != result:
                    raise ValueError(f"Conflicting {suite} summaries for {name} step {step} at {path}:{line_number}")
    return summaries


def _evaluation_history(experiment: Experiment, summaries: dict[tuple[str, int], dict]) -> History:
    """Compute standalone metrics from existing draws and recover format metrics from stdout."""
    history: History = {}
    records = {}
    if experiment.output is not None:
        for path in sorted((experiment.output / "evaluations").glob("*.json")):
            if match := EVALUATION_FILE.fullmatch(path.name):
                records[match[1], int(match[2])] = _read_json(path)

    for suite, step in sorted(records.keys() | summaries.keys()):
        draw_archive = records.get((suite, step))
        summary = summaries.get((suite, step))
        if draw_archive is not None:
            protocol = draw_archive["protocol"]
            n = protocol["rollout_n"]
            name, recorded_step = _checkpoint_identity(draw_archive["checkpoint"])
            if name != experiment.name or recorded_step != step or protocol["suite"] != suite:
                raise ValueError(
                    f"Evaluation record belongs to another checkpoint or suite: {experiment.name} step {step}"
                )
            expected = set(protocol["question_hashes"])
            seen = set()
            by_benchmark = defaultdict(list)
            for record in draw_archive["records"]:
                uid, draw = record["uid"], record["draw"]
                if (
                    uid not in expected
                    or uid.split(":", 1)[0] != record["benchmark"]
                    or type(draw) is not int
                    or draw not in range(n)
                    or (uid, draw) in seen
                    or record["acc"] not in (0, 1)
                    or type(record["response_tokens"]) is not int
                    or record["response_tokens"] < 0
                    or record["finish_reason"] not in {"stop", "length"}
                ):
                    raise ValueError(f"Invalid or duplicate evaluation draw for {uid} at step {step}")
                seen.add((uid, draw))
                by_benchmark[record["benchmark"]].append(record)
            if len(seen) != len(expected) * n or set(by_benchmark) != set(protocol["benchmarks"]):
                raise ValueError(f"Incomplete evaluation records for {experiment.name} at step {step}")
            benchmarks = {
                benchmark: dict(
                    accuracy=sum(r["acc"] for r in draws) / len(draws),
                    responses=len(draws),
                    response_tokens_mean=sum(r["response_tokens"] for r in draws) / len(draws),
                    length_finish_rate=sum(r["finish_reason"] == "length" for r in draws) / len(draws),
                )
                for benchmark, draws in by_benchmark.items()
            }
            if summary is not None:
                config = summary["config"]
                compatible = all(config[key] == protocol[key] for key in ("rollout_n", "seed", "enable_thinking"))
                if (
                    not compatible
                    or config["effective_sampling"] != protocol["sampling"]
                    or set(summary["benchmarks"]) != set(benchmarks)
                    or any(
                        not math.isclose(summary["benchmarks"][b][k], value, rel_tol=1e-8, abs_tol=1e-9)
                        for b, stats in benchmarks.items()
                        for k, value in stats.items()
                    )
                ):
                    raise ValueError(f"Evaluation stdout differs from saved draws for {experiment.name} step {step}")
        else:
            n = summary["config"]["rollout_n"]
            benchmarks = summary["benchmarks"]

        row = history.setdefault(step, {})
        # SwanLab resumes each metric after its last step. Keep checkpoint metrics
        # distinct so an older checkpoint can be archived after a newer one.
        prefix = f"standalone-{suite}--step_{step}"
        row[f"{prefix}/macro/acc/mean@{n}"] = sum(s["accuracy"] for s in benchmarks.values()) / len(benchmarks)
        for benchmark, stats in benchmarks.items():
            row[f"{prefix}/{benchmark}/acc/mean@{n}"] = stats["accuracy"]
            for key in ("response_tokens_mean", "length_finish_rate"):
                row[f"{prefix}/{benchmark}/{key}"] = stats[key]
            if draw_archive is not None:
                draws = by_benchmark[benchmark]
                for label, acc in (("wrong", 0), ("correct", 1)):
                    long_draws = sum(r["acc"] == acc and r["response_tokens"] > 12000 for r in draws)
                    row[f"{prefix}/{benchmark}/{label}_over_12000_tokens_rate"] = long_draws / len(draws)
        if summary is not None:
            for key, value in summary["metrics"].items():
                _, _, metric = key.partition("/")
                if metric.endswith((f"/acc/maj@{n}/mean", f"/invalid_pred/mean@{n}", f"/has_boxed_pattern/mean@{n}")):
                    row[f"{prefix}/{metric}"] = value
        if not all(isinstance(value, float | int) and math.isfinite(value) for value in row.values()):
            raise ValueError(f"Non-finite evaluation metric for {experiment.name} step {step}")
    return history


def _import_swanlab(experiment: Experiment, history: History, config: dict, workspace: str, project: str) -> None:
    import swanlab

    run = swanlab.init(
        workspace=workspace,
        project=project,
        name=experiment.name,
        id=experiment.run_id,
        resume="allow",
        mode="online",
        on_online_error="raise",
        public=False,
        config={**config, "wandb_run_id": experiment.run_id},
    )
    try:
        for step, metrics in sorted(history.items()):
            run.log(metrics, step=step)
    finally:
        run.finish()

    keys = [key for key in (TRAINING_STEP, "timing_s/step") if any(key in row for row in history.values())]
    if keys:
        remote = swanlab.Api().run(path=f"{workspace}/{project}/{experiment.run_id}")
        result = remote.metrics(keys=keys, all=True)
        by_key = {metric["key"]: metric["metrics"] for metric in result["list"]}
        for key in keys:
            expected = {step: row[key] for step, row in history.items() if key in row}
            actual = {
                int(point["step"]): float(point["value"])
                for point in by_key.get(key, [])
                if point.get("value") is not None
            }
            if actual.keys() != expected.keys() or any(
                not math.isclose(actual[step], value, rel_tol=1e-6, abs_tol=1e-9) for step, value in expected.items()
            ):
                raise ValueError(
                    f"SwanLab differs from local history for {key!r}. Check for partial uploads or previous imports."
                )


def _upload_swanlab_evaluations(experiment: Experiment, history: History, workspace: str, project: str) -> None:
    import swanlab

    path = f"{workspace}/{project}/{experiment.run_id}"
    api = swanlab.Api()
    keys = sorted({key for metrics in history.values() for key in metrics})

    def pending_metrics() -> History:
        result = api.run(path=path).metrics(keys=keys, all=True)
        existing = {
            (int(point["step"]), metric["key"]): float(point["value"])
            for metric in result["list"]
            for point in metric["metrics"]
            if point.get("value") is not None
        }
        pending: History = {}
        for step, metrics in history.items():
            for key, value in metrics.items():
                if (step, key) not in existing:
                    pending.setdefault(step, {})[key] = value
                elif not math.isclose(existing[step, key], value, rel_tol=1e-6, abs_tol=1e-9):
                    raise ValueError(f"SwanLab already has a different value for {key!r} at step {step}")
        return pending

    pending = pending_metrics()
    if not pending:
        print("  SwanLab already has these standalone evaluation metrics")
        return
    run = swanlab.init(
        workspace=workspace,
        project=project,
        id=experiment.run_id,
        resume="must",
        mode="online",
        on_online_error="raise",
    )
    try:
        for step, metrics in sorted(pending.items()):
            run.log(metrics, step=step)
    finally:
        run.finish()
    if pending_metrics():
        raise ValueError("SwanLab is missing standalone evaluation metrics after upload")
    print(f"  SwanLab verified {len(keys)} standalone evaluation metrics on {path}")


def _upload_hf(
    api: Any,
    repo_id: str,
    experiment: Experiment,
    *,
    eval_only: bool = False,
    summaries: dict[tuple[str, int], dict] | None = None,
) -> None:
    destinations = []
    prefix = f"runs/{experiment.run_id}"
    if experiment.output:
        destinations.append((experiment.output, prefix, EVALUATION_FILES if eval_only else OUTPUT_FILES))
    if experiment.mirror and not eval_only:
        destinations.append((experiment.mirror, f"{prefix}/wandb_mirror", MIRROR_FILES))

    uploaded = False
    for path, destination, patterns in destinations:
        if not any(file.is_file() for pattern in patterns for file in path.glob(pattern)):
            continue
        commit = api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=path,
            path_in_repo=destination,
            allow_patterns=patterns,
            ignore_patterns=["*.tmp"],
            commit_message=f"Archive {experiment.name}",
        )
        uploaded = True
        print(f"  HF {destination} @ {commit.oid}")
    for (suite, step), summary in sorted((summaries or {}).items()):
        destination = f"{prefix}/evaluations/{suite}--step_{step}--summary.json"
        content = json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n"
        commit = api.upload_file(
            repo_id=repo_id,
            repo_type="dataset",
            path_or_fileobj=content.encode("utf-8"),
            path_in_repo=destination,
            commit_message=f"Archive standalone {suite} evaluation for {experiment.name} step {step}",
        )
        uploaded = True
        print(f"  HF {destination} @ {commit.oid}")
    if not uploaded:
        print("  No archive files to upload")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="DMPD shared workspace containing outputs/")
    parser.add_argument(
        "--mirror-root", type=Path, default=DEFAULT_MIRROR, help="Existing W&B download's runs/ directory"
    )
    parser.add_argument("--wandb-dir", type=Path, action="append", default=[], help="Additional W&B session directory")
    parser.add_argument("--scope", choices=["all", "paper", "recent"], default="all")
    parser.add_argument(
        "--since", type=dt.date.fromisoformat, default=DEFAULT_SINCE, help="First recent date (inclusive)"
    )
    parser.add_argument("--swanlab-workspace", help="SwanLab workspace or workspace/project")
    parser.add_argument("--swanlab-project", help="SwanLab project name (defaults to DMPD)")
    parser.add_argument("--hf-repo", help="Upload archives to this existing private HF dataset")
    parser.add_argument("--eval-only", action="store_true", help="Migrate evaluations without replaying training logs")
    parser.add_argument(
        "--eval-stdout", type=Path, action="append", default=[], help="Standalone evaluation stdout file or directory"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the selected runs without uploading")
    args = parser.parse_args()
    if args.swanlab_workspace is not None:
        workspace, separator, project = args.swanlab_workspace.partition("/")
        if separator:
            if args.swanlab_project not in (None, project):
                parser.error("--swanlab-project conflicts with the project in --swanlab-workspace")
        else:
            project = args.swanlab_project if args.swanlab_project is not None else "DMPD"
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,25}", workspace):
            parser.error("Invalid SwanLab workspace. Use a workspace name such as jubgjf.")
        if not re.fullmatch(r"[a-zA-Z0-9_.+\-]{1,100}", project):
            parser.error("Invalid SwanLab project. Use a project name such as ATI-OPD.")
        args.swanlab_workspace, args.swanlab_project = workspace, project
        print(f"SwanLab project {workspace}/{project}")

    root = args.root.expanduser().resolve()
    mirror_root = args.mirror_root.expanduser().resolve()
    preview = args.dry_run or not (args.swanlab_workspace or args.hf_repo)
    if args.scope != "paper" and not (root / "outputs").is_dir():
        parser.error(f"Cannot find {root / 'outputs'}. Run on the jump host or use --root or --scope paper.")
    experiments = discover_experiments(root, mirror_root, since=args.since, scope=args.scope)
    try:
        stdout = _read_evaluation_stdout(args.eval_stdout)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.eval_stdout and not stdout:
        print("No completed EVAL_RESULT_JSON records found in the supplied logs", file=sys.stderr)
    unselected = stdout.keys() - {experiment.name for experiment in experiments}
    for name in sorted(unselected):
        print(f"  Skipping evaluation stdout for unselected run {name}")
    indexed = {}
    if not args.eval_only:
        wandb_dirs = [root / "wandb_logs/wandb", root / "wandb", *(path.expanduser() for path in args.wandb_dir)]
        indexed = _indexed_segments(wandb_dirs)

    api = None
    if not preview and args.hf_repo:
        from huggingface_hub import HfApi

        api = HfApi()
        if api.repo_info(args.hf_repo, repo_type="dataset").private is not True:
            parser.error(f"HF dataset {args.hf_repo!r} must be private.")

    failures = missing = active = 0
    for experiment in experiments:
        print(f"{experiment.run_id}  {experiment.name}", flush=True)
        if experiment.output:
            print(f"  Output {experiment.output}")
        if experiment.mirror:
            print(f"  Mirror {experiment.mirror}")
        segments = indexed.get(experiment.run_id, [])
        try:
            with _snapshot_lock(experiment.output):
                summaries = stdout.get(experiment.name, {})
                evaluation_history = {}
                if args.eval_only or summaries:
                    evaluation_history = _evaluation_history(experiment, summaries)
                    if evaluation_history:
                        steps = ", ".join(str(step) for step in evaluation_history)
                        print(f"  Standalone evaluations at step(s) {steps}, {len(summaries)} stdout summaries")
                    elif args.eval_only:
                        print("  No standalone evaluations")
                        missing += 1
                        continue
                if args.eval_only:
                    if api is not None:
                        _upload_hf(api, args.hf_repo, experiment, eval_only=True, summaries=summaries)
                    if not preview and args.swanlab_workspace:
                        _upload_swanlab_evaluations(
                            experiment, evaluation_history, args.swanlab_workspace, args.swanlab_project
                        )
                    continue
                manifest = {}
                if experiment.output:
                    path = experiment.output / "diagnostics/manifest.json"
                    if path.is_file():
                        manifest = _read_json(path)
                        if manifest["logical_run_id"] != experiment.run_id:
                            raise ValueError(f"Run ID mismatch in {path}")
                segments = _session_segments(experiment.run_id, manifest, segments, root)
                for path in segments:
                    print(f"  W&B {path}")
                if not (experiment.output or experiment.mirror or segments):
                    print("  Missing local files")
                    missing += 1
                    continue
                if preview:
                    continue
                if api is not None:
                    _upload_hf(api, args.hf_repo, experiment, summaries=summaries)
                if args.swanlab_workspace:
                    history, config = read_history(experiment, manifest, segments)
                    if not history:
                        print("  No scalar history to import")
                    else:
                        missing_steps = _missing_training_steps(experiment, history)
                        if missing_steps:
                            print(f"  Local history is missing {len(missing_steps)} training steps.")
                        steps = sorted(history)
                        print(f"  Importing {len(steps)} steps ({steps[0]}–{steps[-1]}) into SwanLab")
                        _import_swanlab(experiment, history, config, args.swanlab_workspace, args.swanlab_project)
                    if evaluation_history:
                        _upload_swanlab_evaluations(
                            experiment, evaluation_history, args.swanlab_workspace, args.swanlab_project
                        )
        except ArchiveBusyError:
            print("  Archive is currently locked by the trainer")
            active += 1
        except Exception as exc:
            print(f"  Failed {experiment.run_id}: {exc}", file=sys.stderr)
            failures += 1
    print(f"Selected {len(experiments)} runs, {active} active, {missing} without local files, {failures} failed")
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
