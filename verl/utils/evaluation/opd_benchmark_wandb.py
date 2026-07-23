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

"""Idempotently publish standalone OPD benchmark results to Weights & Biases."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

from verl.utils.evaluation.opd_benchmark import (
    COMPLETION_FILENAME,
    SUMMARY_FILENAME,
    WORK_DIRNAME,
    canonical_json,
    discover_units,
    finite_float,
    write_json_atomic,
)

SYNC_FILENAME = "wandb-sync.json"
SYNC_UNIT_KEY = "sync/unit_key"
SYNC_FINGERPRINT_KEY = "sync/result_fingerprint"


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


def unit_metric_payload(summary: dict[str, Any], result_fingerprint: str) -> dict[str, Any]:
    benchmark = summary.get("benchmark") or {}
    benchmark_id = str(benchmark["id"])
    benchmark_path = str(benchmark["path"])
    global_step = int(summary["global_step"])
    metrics = summary.get("metrics") or {}
    response_tokens = metrics.get("response_tokens") or {}
    prefix = f"benchmark/{benchmark_id}"
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

    tail_gap = (metrics.get("gap_regions") or {}).get("last_window") or {}
    for name, value in {
        "tail_sampled_forward_kl": tail_gap.get("sampled_forward_kl_token_weighted"),
        "tail_gap_abs": tail_gap.get("abs_mean_token_weighted"),
        "tail_high_gap_rate": tail_gap.get("high_gap_rate_token_weighted"),
    }.items():
        _finite_metric(payload, f"{prefix}/{name}", value)

    acc_ci = (summary.get("question_cluster_bootstrap_ci95") or {}).get("acc") or {}
    _finite_metric(payload, f"{prefix}/acc_ci95_low", acc_ci.get("ci95_low"))
    _finite_metric(payload, f"{prefix}/acc_ci95_high", acc_ci.get("ci95_high"))
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
    root = base_dir.expanduser().resolve()
    if experiments:
        result = [(root / experiment).resolve() for experiment in experiments]
    else:
        result = sorted(path.resolve() for path in root.iterdir() if path.is_dir() and not path.name.startswith("."))
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


def _load_unit(unit_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    marker = json.loads((unit_dir / COMPLETION_FILENAME).read_text(encoding="utf-8"))
    summary = json.loads((unit_dir / SUMMARY_FILENAME).read_text(encoding="utf-8"))
    return marker, summary


def _check_remote_conflict(remote_state: dict[str, str], key: str, fingerprint: str) -> bool:
    existing = remote_state.get(key)
    if existing is None:
        return False
    if existing != fingerprint:
        raise RuntimeError(
            f"W&B already contains {key} with fingerprint {existing}, but local results have {fingerprint}. "
            "Completed evaluation units are immutable; use a new --run-id for a deliberate revision."
        )
    return True


def sync_experiment(args: argparse.Namespace, result_root: Path) -> int:
    units = discover_units(result_root)
    if not units:
        print(f"[wandb] no completed benchmark units under {result_root}; skipping")
        return 0
    experiment = _load_experiment_name(result_root)
    candidate_run_id = args.run_id or stable_run_id(args.entity, args.project, experiment)
    if args.dry_run:
        print(
            canonical_json(
                {
                    "experiment": experiment,
                    "result_root": result_root,
                    "run_id": candidate_run_id,
                    "run_id_note": (
                        "explicit" if args.run_id else "stable candidate; dry-run does not query legacy runs"
                    ),
                    "units": [unit_sync_key(path, step) for path, step in units],
                }
            )
        )
        return 0

    import wandb

    api = wandb.Api()
    run_id = args.run_id or resolve_run_id(
        api,
        args.entity,
        args.project,
        experiment,
        adopt_existing_name=args.adopt_existing_name,
    )
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        id=run_id,
        resume="allow",
        name=experiment,
        job_type="standalone-benchmark-evaluation",
        config={"evaluation_experiment": experiment, "evaluation_schema_version": 2},
        reinit=True,
    )
    if run is None:
        raise RuntimeError("wandb.init returned no run")
    synced = 0
    skipped = 0
    artifacts_updated = 0
    try:
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
        run.define_metric("eval/global_step")
        run.define_metric("benchmark/*", step_metric="eval/global_step")
        for (benchmark_path, global_step), unit_dir in units.items():
            marker, summary = _load_unit(unit_dir)
            fingerprint = str(marker["result_fingerprint"])
            key = unit_sync_key(benchmark_path, global_step)
            metrics_already_synced = _check_remote_conflict(remote_state, key, fingerprint)
            benchmark = summary.get("benchmark") or {}
            name = artifact_name(experiment, str(benchmark["id"]), global_step)
            content_fingerprint = artifact_fingerprint(unit_dir, include_rollouts=args.include_rollouts)
            if not _remote_artifact_matches(api, args.entity, args.project, name, content_fingerprint):
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
                logged_artifact.wait()
                artifacts_updated += 1
                print(f"[wandb] artifact uploaded: {key}")
            if metrics_already_synced:
                sync_manifest[key] = fingerprint
                run.summary["benchmark_sync_manifest"] = sync_manifest
                write_json_atomic(
                    result_root / SYNC_FILENAME,
                    {
                        "entity": args.entity,
                        "project": args.project,
                        "run_id": run_id,
                        "experiment": experiment,
                        "units": sync_manifest,
                    },
                )
                skipped += 1
                print(f"[wandb] metrics already synced: {key}")
                continue
            run.log(unit_metric_payload(summary, fingerprint))
            remote_state[key] = fingerprint
            sync_manifest[key] = fingerprint
            run.summary["benchmark_sync_manifest"] = sync_manifest
            write_json_atomic(
                result_root / SYNC_FILENAME,
                {
                    "entity": args.entity,
                    "project": args.project,
                    "run_id": run_id,
                    "experiment": experiment,
                    "units": sync_manifest,
                },
            )
            synced += 1
            print(f"[wandb] synced: {key}")
    finally:
        run.finish()
    print(
        f"[wandb] experiment={experiment} run_id={run_id} "
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
        "--run-id",
        default=None,
        help="Optional explicit run ID. Only valid when syncing exactly one experiment.",
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
