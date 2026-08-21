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

"""Idempotently publish standalone OPD benchmark results to Weights & Biases.

This file is intentionally standalone: outside Python's standard library, only
``wandb`` is imported, and that import happens only when a real sync starts.
Execute this file by path instead of importing the ``verl`` package.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 2
METADATA_FILENAME = "metadata.json"
ROLLOUTS_FILENAME = "rollouts.jsonl.gz"
SUMMARY_FILENAME = "summary.json"
PER_PROBLEM_FILENAME = "per_problem.csv"
LENGTH_BINS_FILENAME = "length_bins.csv"
COMPLETION_FILENAME = "COMPLETED.json"
BENCHMARKS_DIRNAME = "benchmarks"
WORK_DIRNAME = ".work"
REQUIRED_RESULT_FILES = (
    METADATA_FILENAME,
    ROLLOUTS_FILENAME,
    SUMMARY_FILENAME,
    PER_PROBLEM_FILENAME,
    LENGTH_BINS_FILENAME,
)
SYNC_FILENAME = "wandb-sync.json"
OFFLINE_SYNC_COMMANDS_FILENAME = "wandb-offline-sync.sh"
SYNC_UNIT_KEY = "sync/unit_key"
SYNC_FINGERPRINT_KEY = "sync/result_fingerprint"
LEGACY_MTIME_MIGRATION_REASON = "legacy model manifest upgrade; merged-weight differences were limited to mtime"
INPUT_MANIFEST_GROUPS = (
    "dataset",
    "student_model_metadata",
    "student_model_weights",
    "student_tokenizer",
    "chat_template",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=_json_default)


def payload_fingerprint(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def serialized_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n").encode("utf-8")


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def absolute_path(path: str | Path) -> Path:
    """Return an absolute path without resolving portable shared-storage symlinks."""

    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def normalize_benchmark_path(raw_path: str) -> str:
    stripped = str(raw_path).strip()
    path = Path(stripped)
    if not stripped or path.is_absolute() or path == Path(".") or ".." in path.parts:
        raise ValueError(f"Benchmark must be a non-empty path relative to the dataset root, got {raw_path!r}")
    return path.as_posix()


def _validate_completed_unit(output_dir: Path) -> tuple[bool, str]:
    marker_path = output_dir / COMPLETION_FILENAME
    metadata_path = output_dir / METADATA_FILENAME
    if not marker_path.is_file() or not metadata_path.is_file():
        return False, "completion marker or metadata is missing"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"completion metadata is unreadable: {exc}"
    if marker.get("schema_version") != SCHEMA_VERSION or metadata.get("schema_version") != SCHEMA_VERSION:
        return False, "schema version differs"
    if not metadata.get("generation_complete"):
        return False, "generation_complete is false"
    if marker.get("generation_fingerprint") != metadata.get("generation_fingerprint"):
        return False, "completion marker and metadata fingerprints differ"

    file_entries = marker.get("files") or {}
    for name in REQUIRED_RESULT_FILES:
        path = output_dir / name
        entry = file_entries.get(name)
        if not path.is_file() or not isinstance(entry, dict):
            return False, f"required result file is missing: {name}"
        if path.stat().st_size != entry.get("size"):
            return False, f"result file size changed: {name}"
    return True, "complete"


def discover_units(
    result_root: str | Path,
    *,
    global_steps: set[int] | None = None,
    benchmarks: set[str] | None = None,
) -> dict[tuple[str, int], Path]:
    root = absolute_path(result_root)
    units: dict[tuple[str, int], Path] = {}
    benchmarks_root = root / BENCHMARKS_DIRNAME
    if not benchmarks_root.is_dir():
        return units
    for directory in sorted(benchmarks_root.iterdir()):
        benchmark_metadata_path = directory / "benchmark.json"
        if not directory.is_dir() or not benchmark_metadata_path.is_file():
            continue
        try:
            benchmark_metadata = json.loads(benchmark_metadata_path.read_text(encoding="utf-8"))
            normalized = normalize_benchmark_path(str(benchmark_metadata["path"]))
        except Exception:
            continue
        if benchmarks is not None and normalized not in benchmarks:
            continue
        for step_dir in sorted(directory.glob("global_step_*")):
            match = re.fullmatch(r"global_step_(\d+)", step_dir.name)
            if not match:
                continue
            global_step = int(match.group(1))
            if global_steps is not None and global_step not in global_steps:
                continue
            valid, _ = _validate_completed_unit(step_dir)
            if valid:
                units[(normalized, global_step)] = step_dir
    return dict(sorted(units.items()))


def stable_run_id(entity: str, project: str, experiment: str) -> str:
    digest = hashlib.sha256(f"{entity}\0{project}\0{experiment}".encode()).hexdigest()[:20]
    return f"eval-{digest}"


def sanitize_name(value: str, *, max_length: int = 120) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-") or "evaluation"
    if len(normalized) <= max_length:
        return normalized
    digest = hashlib.sha256(normalized.encode()).hexdigest()[:10]
    return f"{normalized[: max_length - 12].rstrip('.-')}--{digest}"


def unit_sync_key(benchmark_path: str, global_step: int) -> str:
    return f"{benchmark_path}@global_step_{global_step}"


def artifact_name(experiment: str, benchmark_id: str, global_step: int) -> str:
    experiment_digest = hashlib.sha256(experiment.encode()).hexdigest()[:10]
    return sanitize_name(f"eval-{experiment_digest}-{benchmark_id}-step-{global_step}")


def _finite_metric(payload: dict[str, Any], key: str, value: Any) -> None:
    parsed = finite_float(value)
    if parsed is not None:
        payload[key] = parsed


def unit_metric_payload(
    summary: dict[str, Any],
    result_fingerprint: str,
    *,
    metadata: dict[str, Any] | None = None,
    checkpoint_timing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    benchmark = summary.get("benchmark") or {}
    benchmark_id = str(benchmark["id"])
    benchmark_path = str(benchmark["path"])
    benchmark_name = str(benchmark.get("name") or Path(benchmark_path).name)
    global_step = int(summary["global_step"])
    metrics = summary.get("metrics") or {}
    response_tokens = metrics.get("response_tokens") or {}
    prefix = f"benchmark/{benchmark_id}"
    rollout_n = int(summary.get("rollout_n") or 16)
    test_prefix = f"test-core/{benchmark_name}"
    payload: dict[str, Any] = {
        "eval/global_step": global_step,
        SYNC_UNIT_KEY: unit_sync_key(benchmark_path, global_step),
        SYNC_FINGERPRINT_KEY: result_fingerprint,
    }
    direct_metrics = {
        "acc": metrics.get("acc"),
        "eos_success_rate": metrics.get("eos_success_rate"),
        "answer_extractable_rate": metrics.get("answer_extractable_rate"),
        "answer_pattern_rate": metrics.get("answer_pattern_rate"),
        "accepted_answer_format_rate": metrics.get("accepted_answer_format_rate"),
        "truncation_rate": metrics.get("truncation_rate"),
        "repetition_text_zlib_rate": metrics.get("repetition_text_zlib_rate"),
        "repetition_token_zlib_rate": metrics.get("repetition_token_zlib_rate"),
        "tail_ngram_repeat_fraction_mean": metrics.get("tail_ngram_repeat_fraction_mean"),
        "question_count": metrics.get("question_count"),
        "rollout_count": metrics.get("rollout_count"),
        "response_tokens_mean": response_tokens.get("mean"),
        "response_tokens_p50": response_tokens.get("p50"),
        "response_tokens_p90": response_tokens.get("p90"),
        "response_tokens_p99": response_tokens.get("p99"),
    }
    for name, value in direct_metrics.items():
        _finite_metric(payload, f"{prefix}/{name}", value)

    raw_accuracy = summary.get("accuracy_at_n") or {}
    for name in ("mean", "maj", "best"):
        _finite_metric(payload, f"{test_prefix}/acc/{name}@{rollout_n}", raw_accuracy.get(name))
    for name, value in {
        "response_length/mean": response_tokens.get("mean"),
        "response_length/p90": response_tokens.get("p90"),
        "response_length/p99": response_tokens.get("p99"),
        "clip_ratio": metrics.get("truncation_rate"),
        "eos_success_rate": metrics.get("eos_success_rate"),
        "repetition/token_zlib_rate": metrics.get("repetition_token_zlib_rate"),
        "repetition/text_zlib_rate": metrics.get("repetition_text_zlib_rate"),
        "repetition/tail_ngram_fraction": metrics.get("tail_ngram_repeat_fraction_mean"),
    }.items():
        _finite_metric(payload, f"{test_prefix}/{name}", value)

    tail_gap = (metrics.get("gap_regions") or {}).get("last_window") or {}
    for name, value in {
        "tail_sampled_forward_kl": tail_gap.get("sampled_forward_kl_token_weighted"),
        "tail_gap_abs": tail_gap.get("abs_mean_token_weighted"),
        "tail_high_gap_rate": tail_gap.get("high_gap_rate_token_weighted"),
    }.items():
        _finite_metric(payload, f"{prefix}/{name}", value)
        _finite_metric(payload, f"{test_prefix}/teacher_student_gap/{name}", value)

    acc_ci = (summary.get("question_cluster_bootstrap_ci95") or {}).get("acc") or {}
    _finite_metric(payload, f"{prefix}/acc_ci95_low", acc_ci.get("ci95_low"))
    _finite_metric(payload, f"{prefix}/acc_ci95_high", acc_ci.get("ci95_high"))

    repair = summary.get("repair") or {}
    repair_metrics = repair.get("metrics") or {}
    repair_accuracy = repair.get("accuracy_at_n") or {}
    paired = repair.get("paired") or {}
    if repair:
        repair_prefix = f"test-repair-core/{benchmark_name}"
        for name in ("mean", "maj", "best"):
            _finite_metric(payload, f"{repair_prefix}/acc/{name}@{rollout_n}", repair_accuracy.get(name))
        _finite_metric(payload, f"{repair_prefix}/acc/paired_delta@{rollout_n}", paired.get("paired_delta"))
        repair_response_tokens = repair_metrics.get("response_tokens") or {}
        for name, value in {
            "repair/execution_rate": paired.get("repair_execution_rate"),
            "repair/trigger_rate": paired.get("trigger_rate"),
            "repair/intervention_count_mean": (paired.get("intervention_count") or {}).get("mean"),
            "repair/first_position_mean": (paired.get("first_intervention_position") or {}).get("mean"),
            "teacher_student_gap/initial_mean": (paired.get("initial_gap_mean") or {}).get("mean"),
            "teacher_student_gap/initial_max": (paired.get("initial_gap_max") or {}).get("mean"),
            "teacher_student_gap/initial_p95": (paired.get("initial_gap_p95") or {}).get("mean"),
            "teacher_student_gap/selected_segment_score": (paired.get("selected_segment_score") or {}).get("mean"),
            "teacher/nonfinite_token_fraction": paired.get("teacher_nonfinite_token_fraction"),
            "response_length/mean": repair_response_tokens.get("mean"),
            "response_length/p90": repair_response_tokens.get("p90"),
            "clip_ratio": repair_metrics.get("truncation_rate"),
            "eos_success_rate": repair_metrics.get("eos_success_rate"),
            "repetition/token_zlib_rate": repair_metrics.get("repetition_token_zlib_rate"),
            "repetition/text_zlib_rate": repair_metrics.get("repetition_text_zlib_rate"),
            "repetition/tail_ngram_fraction": repair_metrics.get("tail_ngram_repeat_fraction_mean"),
        }.items():
            _finite_metric(payload, f"{repair_prefix}/{name}", value)
        for name, value in (paired.get("transitions") or {}).items():
            _finite_metric(payload, f"{repair_prefix}/transition/{name}", value)

    metadata = metadata or {}
    for name, value in {
        "raw_generation_seconds": metadata.get("generation_elapsed_seconds"),
        "teacher_scoring_seconds": metadata.get("teacher_scoring_elapsed_seconds"),
        "repair_seconds": metadata.get("repair_elapsed_seconds"),
    }.items():
        _finite_metric(payload, f"eval/timing/{benchmark_name}/{name}", value)
    checkpoint_timing = checkpoint_timing or {}
    for name, value in {
        "model_load_seconds": checkpoint_timing.get("model_load_seconds"),
        "teacher_model_load_seconds": checkpoint_timing.get("teacher_model_load_seconds"),
        "checkpoint_total_seconds": checkpoint_timing.get("checkpoint_total_seconds"),
        "failure_retry_count": checkpoint_timing.get("failure_retry_count"),
    }.items():
        _finite_metric(payload, f"eval/timing/{name}", value)
    return payload


def _load_experiment_name(result_root: Path) -> str:
    manifest_path = result_root / "experiment.json"
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        experiment = str(payload.get("experiment") or "").strip()
        if experiment:
            return experiment
    return result_root.name


def discover_experiment_roots(base_dir: Path, experiments: list[str] | None) -> list[Path]:
    root = absolute_path(base_dir)
    if experiments:
        result = [absolute_path(root / experiment) for experiment in experiments]
    else:
        result = sorted(
            absolute_path(path) for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")
        )
    missing = [path for path in result if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Evaluation experiment directories do not exist: {missing}")
    return result


def _remote_history_state(api: Any, entity: str, project: str, run_id: str) -> dict[str, str]:
    remote_run = api.run(f"{entity}/{project}/{run_id}")
    rows = remote_run.scan_history(keys=[SYNC_UNIT_KEY, SYNC_FINGERPRINT_KEY], page_size=1000)
    result: dict[str, str] = {}
    for row in rows:
        key = row.get(SYNC_UNIT_KEY)
        fingerprint = row.get(SYNC_FINGERPRINT_KEY)
        if key and fingerprint:
            result[str(key)] = str(fingerprint)
    return result


def _is_not_found_error(error: Exception) -> bool:
    """Return whether a W&B API exception was caused by an HTTP 404."""

    pending: list[BaseException] = [error]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        response = getattr(current, "response", None)
        if getattr(response, "status_code", None) == 404:
            return True
        message = str(current).lower()
        if any(
            phrase in message
            for phrase in ("http 404", "not found", "could not find", "unable to find", "does not exist")
        ):
            return True
        for candidate in (
            getattr(current, "exc", None),
            getattr(current, "__cause__", None),
            getattr(current, "__context__", None),
        ):
            if isinstance(candidate, BaseException):
                pending.append(candidate)
    return False


def resolve_run_id(api: Any, entity: str, project: str, experiment: str, *, adopt_existing_name: bool) -> str:
    candidate = stable_run_id(entity, project, experiment)
    try:
        api.run(f"{entity}/{project}/{candidate}")
    except Exception as exc:
        if not _is_not_found_error(exc):
            raise RuntimeError(f"Could not query candidate W&B run {entity}/{project}/{candidate}") from exc
    else:
        return candidate
    if not adopt_existing_name:
        return candidate

    matches = list(
        api.runs(
            path=f"{entity}/{project}",
            filters={"display_name": experiment},
            order="-created_at",
            per_page=50,
        )
    )
    if not matches:
        return candidate
    if len(matches) > 1:
        ids = [str(run.id) for run in matches]
        raise RuntimeError(
            f"Found multiple W&B runs named {experiment!r} in {entity}/{project}: {ids}. "
            "Pass --run-id to select the intended run explicitly."
        )
    adopted = str(matches[0].id)
    print(f"[wandb] adopting existing run by exact name: experiment={experiment} run_id={adopted}")
    return adopted


def _remote_artifact_matches(api: Any, entity: str, project: str, name: str, fingerprint: str) -> bool:
    try:
        artifact = api.artifact(f"{entity}/{project}/{name}:latest")
    except Exception as exc:
        if _is_not_found_error(exc):
            return False
        raise RuntimeError(f"Could not query W&B artifact {entity}/{project}/{name}:latest") from exc
    metadata = artifact.metadata or {}
    return str(metadata.get("artifact_fingerprint")) == fingerprint


def _iter_artifact_files(unit_dir: Path, *, include_rollouts: bool) -> Iterable[Path]:
    for path in sorted(unit_dir.rglob("*")):
        if not path.is_file() or WORK_DIRNAME in path.parts or path.name in {".lock"} or path.name.endswith(".tmp"):
            continue
        if not include_rollouts and path.name == "rollouts.jsonl.gz":
            continue
        yield path


def artifact_fingerprint(unit_dir: Path, *, include_rollouts: bool) -> str:
    files = {
        str(path.relative_to(unit_dir)): {"size": path.stat().st_size, "sha256": _sha256_file(path)}
        for path in _iter_artifact_files(unit_dir, include_rollouts=include_rollouts)
    }
    return hashlib.sha256(canonical_json(files).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_unit(unit_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    marker = json.loads((unit_dir / COMPLETION_FILENAME).read_text(encoding="utf-8"))
    summary = json.loads((unit_dir / SUMMARY_FILENAME).read_text(encoding="utf-8"))
    metadata = json.loads((unit_dir / METADATA_FILENAME).read_text(encoding="utf-8"))
    return marker, summary, metadata


def _checkpoint_timing(result_root: Path, global_step: int) -> dict[str, Any]:
    path = result_root / "checkpoint_timings" / f"global_step_{global_step}.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _empty_sync_manifest(
    *,
    entity: str,
    project: str,
    run_id: str,
    experiment: str,
    source_training_run_id: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "entity": entity,
        "project": project,
        "run_id": run_id,
        "experiment": experiment,
        "source_training_run_id": source_training_run_id,
        "units": {},
        "compatible_units": {},
        "artifacts": {},
        "offline_segments": [],
    }


def _load_local_sync_manifest(
    result_root: Path,
    *,
    entity: str,
    project: str,
    run_id: str,
    experiment: str,
    source_training_run_id: str | None,
) -> dict[str, Any]:
    expected = _empty_sync_manifest(
        entity=entity,
        project=project,
        run_id=run_id,
        experiment=experiment,
        source_training_run_id=source_training_run_id,
    )
    path = result_root / SYNC_FILENAME
    if not path.is_file():
        return expected
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("entity", "project", "run_id", "experiment"):
        actual = payload.get(key)
        if actual is not None and str(actual) != str(expected[key]):
            raise RuntimeError(
                f"Local W&B sync manifest identity changed at {path}: {key}={actual!r}, expected={expected[key]!r}."
            )
    stored_source = payload.get("source_training_run_id")
    if stored_source and source_training_run_id and str(stored_source) != str(source_training_run_id):
        raise RuntimeError(
            "Local W&B sync manifest points at another source training run: "
            f"stored={stored_source!r}, requested={source_training_run_id!r}."
        )
    expected["units"] = {str(key): str(value) for key, value in dict(payload.get("units") or {}).items()}
    expected["compatible_units"] = {
        str(key): dict(value)
        for key, value in dict(payload.get("compatible_units") or {}).items()
        if isinstance(value, dict)
    }
    expected["artifacts"] = {str(key): str(value) for key, value in dict(payload.get("artifacts") or {}).items()}
    expected["offline_segments"] = [str(value) for value in list(payload.get("offline_segments") or [])]
    return expected


def _write_offline_sync_commands(result_root: Path, manifest: dict[str, Any]) -> None:
    segments = [str(absolute_path(value)) for value in manifest.get("offline_segments") or []]
    command_path = result_root / OFFLINE_SYNC_COMMANDS_FILENAME
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "command -v wandb >/dev/null 2>&1 || { echo 'wandb CLI is required on the sync host.' >&2; exit 127; }",
        "",
    ]
    for index, segment in enumerate(segments):
        synced_marker = segment if segment.endswith(".synced") else f"{segment}.synced"
        parts = ["wandb", "sync"]
        if index:
            parts.append("--append")
        if manifest.get("entity"):
            parts.extend(["--entity", str(manifest["entity"])])
        parts.extend(["--project", str(manifest["project"]), "--id", str(manifest["run_id"]), segment])
        lines.append(f"if [[ ! -f {shlex.quote(synced_marker)} ]]; then")
        lines.append(f"    if [[ ! -f {shlex.quote(segment)} ]]; then")
        lines.append(
            "        echo "
            + shlex.quote("Offline W&B segment is not visible at its recorded absolute path: " + segment)
            + " >&2"
        )
        lines.append(
            "        echo 'Keep EVAL_WANDB_ROOT on shared storage. The NPU launcher configures its "
            "jump-host-visible spelling through WANDB_PORTABLE_DATASET_DIR.' >&2"
        )
        lines.append("        exit 2")
        lines.append("    fi")
        lines.append(f"    {' '.join(shlex.quote(part) for part in parts)}")
        lines.append("fi")
    temporary = command_path.with_name(command_path.name + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.chmod(0o755)
    temporary.replace(command_path)


def _offline_sync_target(run_directory: Path) -> Path:
    candidates = sorted(run_directory.glob("*.wandb"))
    if not candidates:
        candidates = sorted(run_directory.glob("*.wandb.synced"))
    if not candidates:
        raise RuntimeError(f"Offline W&B run did not produce a sync file under {run_directory}.")
    return absolute_path(candidates[-1])


def _legacy_manifest_before_mtime_migration(
    metadata: dict[str, Any],
    migration: dict[str, Any],
) -> dict[str, Any] | None:
    compatibility = metadata.get("generation_compatibility") or {}
    current_manifest = metadata.get("input_manifest") or {}
    if (
        metadata.get("generation_schema_version") != 2
        or compatibility.get("schema_version") != 1
        or current_manifest.get("schema_version") != 2
        or migration.get("reason") != LEGACY_MTIME_MIGRATION_REASON
        or migration.get("legacy_generation_fingerprint") != metadata.get("generation_fingerprint")
        or migration.get("stable_input_manifest_fingerprint") != current_manifest.get("fingerprint")
        or migration.get("student_model_identity") != current_manifest.get("student_model_identity")
        or migration.get("compatible_generation_fingerprint")
        not in set(compatibility.get("compatible_fingerprints") or [])
    ):
        return None

    changes: dict[str, dict[str, Any]] = {}
    for raw_change in migration.get("mtime_changes") or []:
        if not isinstance(raw_change, dict) or not raw_change.get("path"):
            return None
        path = str(raw_change["path"])
        if path in changes:
            return None
        changes[path] = raw_change

    restored: dict[str, Any] = {}
    consumed_changes: set[str] = set()
    for group in INPUT_MANIFEST_GROUPS:
        restored_entries: list[dict[str, Any]] = []
        for raw_entry in current_manifest.get(group) or []:
            if not isinstance(raw_entry, dict):
                return None
            entry = {str(key): value for key, value in raw_entry.items() if key != "logical_path"}
            if group == "student_model_weights":
                path = str(entry.get("path") or "")
                change = changes.get(path)
                if change is not None:
                    if (
                        entry.get("size") != change.get("size")
                        or entry.get("mtime_ns") != change.get("current_mtime_ns")
                        or change.get("existing_mtime_ns") is None
                    ):
                        return None
                    entry["mtime_ns"] = change["existing_mtime_ns"]
                    consumed_changes.add(path)
            restored_entries.append(entry)
        restored[group] = restored_entries
    if consumed_changes != set(changes):
        return None

    restored["fingerprint"] = payload_fingerprint(restored)
    if restored["fingerprint"] != migration.get("legacy_input_manifest_fingerprint"):
        return None
    return restored


def compatible_legacy_result_fingerprints(
    unit_dir: Path,
    marker: dict[str, Any],
    metadata: dict[str, Any],
) -> set[str]:
    """Recover result fingerprints changed only by the v2-to-v3 manifest migration.

    A short-lived evaluator version persisted a compatibility-only manifest
    upgrade into completed results. Since ``metadata.json`` participates in the
    immutable result fingerprint, those units no longer matched an identical
    result already published by an older evaluator. Reconstruct the exact old
    metadata only for this narrowly validated migration so the publisher can
    recognize the remote unit without uploading a revision.
    """

    if not metadata.get("generation_complete") or not metadata.get("repair_complete"):
        return set()
    files = marker.get("files") or {}
    metadata_entry = files.get(METADATA_FILENAME)
    if not isinstance(metadata_entry, dict):
        return set()
    try:
        current_metadata_bytes = (unit_dir / METADATA_FILENAME).read_bytes()
    except OSError:
        return set()
    if (
        len(current_metadata_bytes) != metadata_entry.get("size")
        or hashlib.sha256(current_metadata_bytes).hexdigest() != metadata_entry.get("sha256")
        or marker.get("result_fingerprint") != payload_fingerprint(files)
    ):
        return set()

    compatibility = metadata.get("generation_compatibility") or {}
    migrations = compatibility.get("migrations") or []
    if not migrations or any(
        not isinstance(migration, dict) or migration.get("reason") != LEGACY_MTIME_MIGRATION_REASON
        for migration in migrations
    ):
        return set()

    result: set[str] = set()
    for migration in migrations:
        legacy_manifest = _legacy_manifest_before_mtime_migration(metadata, migration)
        if legacy_manifest is None:
            continue
        legacy_metadata = copy.deepcopy(metadata)
        legacy_metadata.pop("generation_compatibility", None)
        legacy_metadata["input_manifest"] = legacy_manifest
        legacy_metadata_bytes = serialized_json_bytes(legacy_metadata)
        legacy_files = copy.deepcopy(files)
        legacy_files[METADATA_FILENAME] = {
            "size": len(legacy_metadata_bytes),
            "sha256": hashlib.sha256(legacy_metadata_bytes).hexdigest(),
        }
        result.add(payload_fingerprint(legacy_files))
    return result


def _check_remote_conflict(
    remote_state: dict[str, str],
    key: str,
    fingerprint: str,
    *,
    compatible_fingerprints: set[str] | None = None,
    state_label: str = "W&B",
) -> str | None:
    existing = remote_state.get(key)
    if existing is None:
        return None
    if existing != fingerprint:
        if existing in (compatible_fingerprints or set()):
            return existing
        raise RuntimeError(
            f"{state_label} already contains {key} with fingerprint {existing}, "
            f"but local results have {fingerprint}. "
            "Completed evaluation units are immutable; use a new --run-id for a deliberate revision."
        )
    return existing


def _record_compatible_unit(
    manifest: dict[str, Any],
    *,
    key: str,
    remote_fingerprint: str,
    local_fingerprint: str,
) -> None:
    manifest.setdefault("compatible_units", {})[key] = {
        "reason": "completed result changed only by the legacy merged-weight mtime manifest migration",
        "remote_result_fingerprint": remote_fingerprint,
        "local_result_fingerprint": local_fingerprint,
    }


def _wandb_settings(wandb_module: Any) -> Any | None:
    settings_class = getattr(wandb_module, "Settings", None)
    if settings_class is None:
        return None
    fields = getattr(settings_class, "model_fields", {})
    kwargs: dict[str, Any] = {"disable_git": True}
    if "x_disable_stats" in fields:
        kwargs["x_disable_stats"] = True
    elif "_disable_stats" in fields:
        kwargs["_disable_stats"] = True
    return settings_class(**kwargs)


def sync_experiment(args: argparse.Namespace, result_root: Path) -> int:
    requested_steps = getattr(args, "global_step", None)
    global_steps = None if not requested_steps else {int(step) for step in requested_steps}
    requested_benchmarks = getattr(args, "benchmark", None)
    benchmarks = (
        None
        if not requested_benchmarks
        else {normalize_benchmark_path(str(benchmark)) for benchmark in requested_benchmarks}
    )
    units = discover_units(result_root, global_steps=global_steps, benchmarks=benchmarks)
    if not units:
        if global_steps is not None:
            requested = ", ".join(str(step) for step in sorted(global_steps))
            raise RuntimeError(
                f"No completed benchmark units for requested global step(s) {requested} under {result_root}"
            )
        print(f"[wandb] no completed benchmark units under {result_root}; skipping")
        return 0
    experiment = _load_experiment_name(result_root)
    candidate_run_id = args.run_id or stable_run_id(args.entity, args.project, experiment)
    mode = str(args.mode).lower()
    if mode == "disabled":
        print(f"[wandb] disabled: experiment={experiment}")
        return 0
    if args.dry_run:
        print(
            canonical_json(
                {
                    "experiment": experiment,
                    "result_root": result_root,
                    "run_id": candidate_run_id,
                    "mode": mode,
                    "run_id_note": (
                        "explicit" if args.run_id else "stable candidate; dry-run does not query legacy runs"
                    ),
                    "units": [unit_sync_key(path, step) for path, step in units],
                }
            )
        )
        return 0

    api = None
    if mode == "online":
        import wandb

        api = wandb.Api()
        run_id = args.run_id or resolve_run_id(
            api,
            args.entity,
            args.project,
            experiment,
            adopt_existing_name=args.adopt_existing_name,
        )
    elif mode == "offline":
        # Offline operation must not contact the Public API. Every restart uses
        # the same deterministic ID and creates an appendable local segment.
        run_id = candidate_run_id
    else:
        raise ValueError(f"Unsupported W&B mode: {mode!r}")

    local_manifest = _load_local_sync_manifest(
        result_root,
        entity=args.entity,
        project=args.project,
        run_id=run_id,
        experiment=experiment,
        source_training_run_id=args.source_training_run_id,
    )
    if mode == "offline":
        pending = False
        manifest_changed = False
        for (benchmark_path, global_step), unit_dir in units.items():
            marker, summary, metadata = _load_unit(unit_dir)
            fingerprint = str(marker["result_fingerprint"])
            key = unit_sync_key(benchmark_path, global_step)
            compatible_fingerprints = compatible_legacy_result_fingerprints(unit_dir, marker, metadata)
            staged_fingerprint = _check_remote_conflict(
                local_manifest["units"],
                key,
                fingerprint,
                compatible_fingerprints=compatible_fingerprints,
                state_label="Local W&B staging manifest",
            )
            metrics_done = staged_fingerprint is not None
            compatibility_only = staged_fingerprint is not None and staged_fingerprint != fingerprint
            if compatibility_only:
                _record_compatible_unit(
                    local_manifest,
                    key=key,
                    remote_fingerprint=staged_fingerprint,
                    local_fingerprint=fingerprint,
                )
                manifest_changed = True
            benchmark = summary.get("benchmark") or {}
            name = artifact_name(experiment, str(benchmark["id"]), global_step)
            artifact_done = compatibility_only
            if not artifact_done:
                content_fingerprint = artifact_fingerprint(unit_dir, include_rollouts=args.include_rollouts)
                artifact_done = local_manifest["artifacts"].get(name) == content_fingerprint
            pending = pending or not (metrics_done and artifact_done)
        if not pending:
            if manifest_changed:
                write_json_atomic(result_root / SYNC_FILENAME, local_manifest)
            print(f"[wandb] all offline units already staged: experiment={experiment} run_id={run_id}")
            _write_offline_sync_commands(result_root, local_manifest)
            return 0

    wandb_dir = absolute_path(args.wandb_dir) if args.wandb_dir else result_root / ".wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    if mode == "offline":
        # Keep every file needed by ``wandb sync`` on shared storage. In
        # particular, Artifact staging must never fall back to the NPU node's
        # unshared HOME directory.
        os.environ["WANDB_DIR"] = str(wandb_dir)
        os.environ["WANDB_DATA_DIR"] = str(
            absolute_path(args.wandb_data_dir) if args.wandb_data_dir else result_root / ".wandb-data"
        )
        os.environ["WANDB_CACHE_DIR"] = str(
            absolute_path(args.wandb_cache_dir) if args.wandb_cache_dir else result_root / ".wandb-cache"
        )
        Path(os.environ["WANDB_DATA_DIR"]).mkdir(parents=True, exist_ok=True)
        Path(os.environ["WANDB_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
        import wandb

    init_kwargs = {
        "entity": args.entity,
        "project": args.project,
        "id": run_id,
        "resume": "allow",
        "name": experiment,
        "job_type": "standalone-benchmark-evaluation",
        "config": {
            "evaluation_experiment": experiment,
            "evaluation_schema_version": 2,
            "source_training_run_id": args.source_training_run_id,
        },
        "mode": mode,
        "dir": str(wandb_dir),
        "reinit": True,
    }
    settings = _wandb_settings(wandb)
    if settings is not None:
        init_kwargs["settings"] = settings
    run = wandb.init(
        **init_kwargs,
    )
    if run is None:
        raise RuntimeError("wandb.init returned no run")
    offline_run_directory = absolute_path(Path(run.dir).parent) if mode == "offline" else None
    synced = 0
    skipped = 0
    artifacts_updated = 0
    try:
        if mode == "online":
            assert api is not None
            remote_state = _remote_history_state(api, args.entity, args.project, run_id) if run.resumed else {}
            summary_state = {
                str(key): str(value) for key, value in dict(run.summary.get("benchmark_sync_manifest") or {}).items()
            }
            for key, fingerprint in summary_state.items():
                existing = remote_state.get(key)
                if existing is not None and existing != fingerprint:
                    raise RuntimeError(f"W&B history and run summary disagree for {key}: {existing} != {fingerprint}")
                remote_state[key] = fingerprint
            sync_manifest = dict(remote_state)
            artifact_manifest = dict(local_manifest["artifacts"])
        else:
            remote_state = dict(local_manifest["units"])
            sync_manifest = dict(remote_state)
            artifact_manifest = dict(local_manifest["artifacts"])
        run.define_metric("eval/global_step")
        run.define_metric("benchmark/*", step_metric="eval/global_step")
        run.define_metric("test-core/*", step_metric="eval/global_step")
        run.define_metric("test-repair-core/*", step_metric="eval/global_step")
        run.define_metric("eval/timing/*", step_metric="eval/global_step")
        for (benchmark_path, global_step), unit_dir in units.items():
            marker, summary, metadata = _load_unit(unit_dir)
            fingerprint = str(marker["result_fingerprint"])
            key = unit_sync_key(benchmark_path, global_step)
            compatible_fingerprints = compatible_legacy_result_fingerprints(unit_dir, marker, metadata)
            synced_fingerprint = _check_remote_conflict(
                remote_state,
                key,
                fingerprint,
                compatible_fingerprints=compatible_fingerprints,
                state_label="W&B" if mode == "online" else "Local W&B staging manifest",
            )
            metrics_already_synced = synced_fingerprint is not None
            compatibility_only = synced_fingerprint is not None and synced_fingerprint != fingerprint
            benchmark = summary.get("benchmark") or {}
            name = artifact_name(experiment, str(benchmark["id"]), global_step)
            if compatibility_only:
                artifact_already_synced = True
            else:
                content_fingerprint = artifact_fingerprint(unit_dir, include_rollouts=args.include_rollouts)
                if mode == "online":
                    assert api is not None
                    artifact_already_synced = _remote_artifact_matches(
                        api, args.entity, args.project, name, content_fingerprint
                    )
                else:
                    artifact_already_synced = artifact_manifest.get(name) == content_fingerprint
            if not artifact_already_synced:
                artifact = wandb.Artifact(
                    name=name,
                    type="benchmark-evaluation",
                    metadata={
                        "experiment": experiment,
                        "benchmark": benchmark_path,
                        "global_step": global_step,
                        "result_fingerprint": fingerprint,
                        "artifact_fingerprint": content_fingerprint,
                        "includes_rollouts": args.include_rollouts,
                    },
                )
                for file_path in _iter_artifact_files(unit_dir, include_rollouts=args.include_rollouts):
                    artifact.add_file(str(file_path), name=str(file_path.relative_to(unit_dir)))
                logged_artifact = run.log_artifact(artifact, aliases=["latest", f"step-{global_step}"])
                if mode == "online":
                    logged_artifact.wait()
                artifact_manifest[name] = content_fingerprint
                local_manifest["artifacts"] = artifact_manifest
                write_json_atomic(result_root / SYNC_FILENAME, local_manifest)
                artifacts_updated += 1
                print(f"[wandb] artifact {'uploaded' if mode == 'online' else 'staged'}: {key}")
            if metrics_already_synced:
                assert synced_fingerprint is not None
                sync_manifest[key] = synced_fingerprint
                run.summary["benchmark_sync_manifest"] = sync_manifest
                local_manifest["units"] = sync_manifest
                local_manifest["artifacts"] = artifact_manifest
                if compatibility_only:
                    _record_compatible_unit(
                        local_manifest,
                        key=key,
                        remote_fingerprint=synced_fingerprint,
                        local_fingerprint=fingerprint,
                    )
                write_json_atomic(
                    result_root / SYNC_FILENAME,
                    local_manifest,
                )
                skipped += 1
                if compatibility_only:
                    print(
                        f"[wandb] compatible legacy result already "
                        f"{'synced' if mode == 'online' else 'staged'}: {key} "
                        f"remote={synced_fingerprint} local={fingerprint}"
                    )
                else:
                    print(f"[wandb] metrics already {'synced' if mode == 'online' else 'staged'}: {key}")
                continue
            run.log(
                unit_metric_payload(
                    summary,
                    fingerprint,
                    metadata=metadata,
                    checkpoint_timing=_checkpoint_timing(result_root, global_step),
                )
            )
            remote_state[key] = fingerprint
            sync_manifest[key] = fingerprint
            run.summary["benchmark_sync_manifest"] = sync_manifest
            local_manifest["units"] = sync_manifest
            local_manifest["artifacts"] = artifact_manifest
            write_json_atomic(
                result_root / SYNC_FILENAME,
                local_manifest,
            )
            synced += 1
            print(f"[wandb] {'synced' if mode == 'online' else 'staged'}: {key}")
    finally:
        run.finish()
        if mode == "offline":
            assert offline_run_directory is not None
            segment = str(_offline_sync_target(offline_run_directory))
            if segment not in local_manifest["offline_segments"]:
                local_manifest["offline_segments"].append(segment)
            write_json_atomic(result_root / SYNC_FILENAME, local_manifest)
            _write_offline_sync_commands(result_root, local_manifest)
    print(
        f"[wandb] mode={mode} experiment={experiment} run_id={run_id} "
        f"metrics_synced={synced} metrics_skipped={skipped} artifacts_updated={artifacts_updated}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default="./eval-results")
    parser.add_argument("--experiments", nargs="+", default=None)
    parser.add_argument("--entity", default="jubgjf")
    parser.add_argument("--project", default="DMPD-eval")
    parser.add_argument(
        "--mode",
        choices=["online", "offline", "disabled"],
        default=os.environ.get("WANDB_MODE", "offline"),
        help="Offline mode never calls the W&B Public API and emits ordered wandb sync commands.",
    )
    parser.add_argument(
        "--wandb-dir",
        default=None,
        help="Local W&B staging root; defaults to <experiment-result-root>/.wandb.",
    )
    parser.add_argument("--wandb-data-dir", default=None)
    parser.add_argument("--wandb-cache-dir", default=None)
    parser.add_argument(
        "--source-training-run-id",
        default=None,
        help="Source training run ID recorded in config; evaluation still uses an independent run.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional explicit run ID. Only valid when syncing exactly one experiment.",
    )
    parser.add_argument(
        "--global-step",
        action="append",
        type=int,
        default=None,
        help="Only publish units at this checkpoint step; repeat to select multiple steps.",
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        default=None,
        help="Only publish this benchmark path; repeat to select multiple benchmarks.",
    )
    parser.add_argument("--include-rollouts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--adopt-existing-name",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume a unique legacy run whose display name exactly matches the experiment.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    roots = discover_experiment_roots(Path(args.base_dir), args.experiments)
    if args.run_id and len(roots) != 1:
        raise ValueError("--run-id requires exactly one selected experiment")
    failures: list[str] = []
    for root in roots:
        try:
            sync_experiment(args, root)
        except Exception as exc:
            failures.append(f"{root}: {type(exc).__name__}: {exc}")
            print(f"ERROR: {failures[-1]}", file=sys.stderr)
    if failures:
        print(f"W&B sync failed for {len(failures)} experiment(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
