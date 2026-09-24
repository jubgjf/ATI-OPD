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

"""Incrementally mirror W&B run records for local analysis.

By default this script downloads ``jubgjf/DMPD`` into
``./analysis_artifacts``. Each run contains:

* ``run.json``: run metadata, config, and summary;
* ``history/steps_<start>_<end>.jsonl.gz``: complete, unsampled history;
* ``files/``: files attached directly to the W&B run;
* optionally, raw Parquet shards from logged ``training-diagnostics`` artifacts;
* ``sync.json`` and ``files-manifest.json``: local incremental-sync state.

W&B's public API supports step-bounded history scans. Existing runs therefore
fetch only steps newer than the last local history shard. If W&B reports that
history changed without advancing its last step, this script deliberately
falls back to refreshing that run's history instead of maintaining a complex
row-level reconciliation database.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1
DEFAULT_PROJECTS = ("jubgjf/DMPD",)
HISTORY_SHARD_RE = re.compile(r"steps_(\d{12})_(\d{12})\.jsonl\.gz$")
METADATA_FILE_NAMES = {
    "config.yaml",
    "diff.patch",
    "output.log",
    "requirements.txt",
    "wandb-metadata.json",
    "wandb-summary.json",
}
TRAINING_DIAGNOSTICS_TYPE = "training-diagnostics"
TRAINING_DIAGNOSTICS_SHARD_PREFIX = "shards/"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dt.date | dt.datetime):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=json_default)


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=json_default)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_json_if_changed(path: Path, payload: Any) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n"
    if path.is_file() and path.read_text(encoding="utf-8") == encoded:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def split_project(project_path: str) -> tuple[str, str]:
    parts = project_path.strip("/").split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"W&B project must have the form ENTITY/PROJECT, got {project_path!r}.")
    return parts[0], parts[1]


def safe_remote_path(name: str) -> Path:
    path = Path(name)
    if not name or path.is_absolute() or path == Path(".") or ".." in path.parts:
        raise ValueError(f"Unsafe W&B file path: {name!r}")
    return path


def summary_dict(run: Any) -> dict[str, Any]:
    summary = run.summary
    raw = getattr(summary, "_json_dict", None)
    if isinstance(raw, dict):
        return dict(raw)
    try:
        return dict(summary)
    except Exception:
        return {}


def run_probe(run: Any) -> dict[str, Any]:
    attrs = dict(getattr(run, "_attrs", {}) or {})
    try:
        last_history_step = int(run.lastHistoryStep)
    except (TypeError, ValueError):
        last_history_step = -1
    line_count = attrs.get("historyLineCount")
    try:
        line_count = int(line_count) if line_count is not None else None
    except (TypeError, ValueError):
        line_count = None
    return {
        "state": str(getattr(run, "state", attrs.get("state", "unknown"))),
        "last_history_step": last_history_step,
        "history_line_count": line_count,
        "heartbeat_at": attrs.get("heartbeatAt"),
    }


def run_record(run: Any, entity: str, project: str, probe: dict[str, Any]) -> dict[str, Any]:
    attrs = dict(getattr(run, "_attrs", {}) or {})
    sweep = getattr(run, "sweep", None)
    return {
        "schema_version": SCHEMA_VERSION,
        "entity": entity,
        "project": project,
        "id": str(run.id),
        "name": str(run.name),
        "url": str(run.url),
        "state": probe["state"],
        "created_at": attrs.get("createdAt"),
        "heartbeat_at": probe["heartbeat_at"],
        "last_history_step": probe["last_history_step"],
        "history_line_count": probe["history_line_count"],
        "group": attrs.get("group"),
        "job_type": attrs.get("jobType"),
        "tags": list(getattr(run, "tags", None) or attrs.get("tags") or []),
        "notes": getattr(run, "notes", None),
        "description": attrs.get("description"),
        "sweep_id": str(getattr(sweep, "id", "")) or None,
        "config": dict(run.config or {}),
        "summary": summary_dict(run),
        "downloaded_at": utc_now(),
    }


def history_shards(history_dir: Path) -> list[dict[str, Any]]:
    manifest = read_json(history_dir / "manifest.json", {})
    known = {
        str(item.get("file")): item
        for item in manifest.get("shards", [])
        if isinstance(item, dict) and item.get("file")
    }
    shards: list[dict[str, Any]] = []
    if not history_dir.is_dir():
        return shards
    for path in sorted(history_dir.glob("steps_*.jsonl.gz")):
        match = HISTORY_SHARD_RE.fullmatch(path.name)
        if match is None:
            continue
        item = dict(known.get(path.name, {}))
        item.update(
            {
                "file": path.name,
                "min_step": int(match.group(1)),
                "max_step": int(match.group(2)),
                "size": path.stat().st_size,
            }
        )
        shards.append(item)
    return shards


def history_local_max(history_dir: Path) -> int:
    shards = history_shards(history_dir)
    return max((int(item["max_step"]) for item in shards), default=-1)


def write_history_shard(
    run: Any,
    destination: Path,
    *,
    min_step: int,
    max_step: int,
    page_size: int,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    row_count = 0
    observed_min: int | None = None
    observed_max: int | None = None
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        rows = run.scan_history(
            page_size=page_size,
            min_step=min_step,
            max_step=max_step + 1,
            use_cache=False,
        )
        for row in rows:
            step = row.get("_step") if isinstance(row, dict) else None
            try:
                parsed_step = int(step)
            except (TypeError, ValueError):
                parsed_step = None
            if parsed_step is not None:
                observed_min = parsed_step if observed_min is None else min(observed_min, parsed_step)
                observed_max = parsed_step if observed_max is None else max(observed_max, parsed_step)
            handle.write(json.dumps(row, ensure_ascii=False, default=json_default))
            handle.write("\n")
            row_count += 1
    temporary.replace(destination)
    return {
        "file": destination.name,
        "min_step": min_step,
        "max_step": max_step,
        "observed_min_step": observed_min,
        "observed_max_step": observed_max,
        "row_count": row_count,
        "size": destination.stat().st_size,
        "downloaded_at": utc_now(),
    }


def write_history_manifest(
    history_dir: Path,
    *,
    remote_last_step: int,
    remote_line_count: int | None,
) -> dict[str, Any]:
    shards = history_shards(history_dir)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "format": "gzip-json-lines",
        "remote_last_history_step": remote_last_step,
        "remote_history_line_count": remote_line_count,
        "local_max_step": max((int(item["max_step"]) for item in shards), default=-1),
        "row_count": sum(int(item.get("row_count", 0)) for item in shards),
        "shards": shards,
        "updated_at": utc_now(),
    }
    write_json_atomic(history_dir / "manifest.json", payload)
    return payload


def refresh_history(
    run: Any,
    run_dir: Path,
    *,
    remote_last_step: int,
    remote_line_count: int | None,
    page_size: int,
) -> dict[str, Any]:
    history_dir = run_dir / "history"
    temporary_dir = Path(tempfile.mkdtemp(prefix=".history-refresh-", dir=run_dir))
    try:
        if remote_last_step >= 0:
            shard_name = f"steps_{0:012d}_{remote_last_step:012d}.jsonl.gz"
            shard = write_history_shard(
                run,
                temporary_dir / shard_name,
                min_step=0,
                max_step=remote_last_step,
                page_size=page_size,
            )
            write_json_atomic(
                temporary_dir / "manifest.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "format": "gzip-json-lines",
                    "remote_last_history_step": remote_last_step,
                    "remote_history_line_count": remote_line_count,
                    "local_max_step": remote_last_step,
                    "row_count": shard["row_count"],
                    "shards": [shard],
                    "updated_at": utc_now(),
                },
            )
        else:
            write_json_atomic(
                temporary_dir / "manifest.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "format": "gzip-json-lines",
                    "remote_last_history_step": -1,
                    "remote_history_line_count": remote_line_count,
                    "local_max_step": -1,
                    "row_count": 0,
                    "shards": [],
                    "updated_at": utc_now(),
                },
            )

        backup = run_dir / ".history-backup"
        if backup.exists():
            shutil.rmtree(backup)
        if history_dir.exists():
            history_dir.replace(backup)
        temporary_dir.replace(history_dir)
        if backup.exists():
            shutil.rmtree(backup)
        return read_json(history_dir / "manifest.json", {})
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)


def sync_history(
    run: Any,
    run_dir: Path,
    *,
    probe: dict[str, Any],
    previous_sync: dict[str, Any],
    page_size: int,
    force: bool,
) -> tuple[str, dict[str, Any]]:
    history_dir = run_dir / "history"
    remote_last = int(probe["last_history_step"])
    remote_lines = probe.get("history_line_count")
    local_max = history_local_max(history_dir)
    previous_probe = previous_sync.get("remote_probe") or {}
    previous_lines = previous_probe.get("history_line_count")

    requires_refresh = force or not (history_dir / "manifest.json").is_file() or remote_last < local_max
    if remote_last == local_max and previous_lines is not None and remote_lines != previous_lines:
        # W&B has no row cursor for late writes at an existing step. Refreshing
        # this one run is safer and intentionally simpler than local row merging.
        requires_refresh = True

    if requires_refresh:
        return "refreshed", refresh_history(
            run,
            run_dir,
            remote_last_step=remote_last,
            remote_line_count=remote_lines,
            page_size=page_size,
        )

    if remote_last > local_max:
        min_step = local_max + 1
        shard_name = f"steps_{min_step:012d}_{remote_last:012d}.jsonl.gz"
        shard = write_history_shard(
            run,
            history_dir / shard_name,
            min_step=min_step,
            max_step=remote_last,
            page_size=page_size,
        )
        manifest = read_json(history_dir / "manifest.json", {})
        old_shards = [item for item in manifest.get("shards", []) if item.get("file") != shard_name]
        manifest["shards"] = [*old_shards, shard]
        write_json_atomic(history_dir / "manifest.json", manifest)
        return "appended", write_history_manifest(
            history_dir,
            remote_last_step=remote_last,
            remote_line_count=remote_lines,
        )

    return "unchanged", read_json(history_dir / "manifest.json", {})


def file_signature(remote_file: Any) -> dict[str, Any]:
    attrs = dict(getattr(remote_file, "_attrs", {}) or {})
    return {
        "size": int(getattr(remote_file, "size", 0) or 0),
        "md5": attrs.get("md5"),
        "updated_at": attrs.get("updatedAt"),
    }


def include_run_file(name: str, mode: str) -> bool:
    if mode == "none":
        return False
    if mode == "all":
        return True
    path = Path(name)
    return path.name in METADATA_FILE_NAMES or path.suffix == ".log"


def local_md5_base64(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode("ascii")


def local_file_matches(path: Path, signature: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    expected_size = int(signature.get("size") or 0)
    remote_md5 = signature.get("md5")
    # A few W&B-generated files are reported with size=0 and md5=None even
    # though their signed download contains data. In that case updatedAt is
    # the only useful remote change marker, so merely require a local file.
    if expected_size > 0 and path.stat().st_size != expected_size:
        return False
    return not remote_md5 or local_md5_base64(path) == remote_md5


def download_run_file(remote_file: Any, api: Any, files_dir: Path) -> None:
    relative_path = safe_remote_path(str(remote_file.name))
    destination = files_dir / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=".file-download-", dir=files_dir.parent))
    try:
        downloaded = remote_file.download(root=str(temporary_root), replace=True, api=api)
        downloaded.close()
        temporary_path = temporary_root / relative_path
        if not temporary_path.is_file():
            raise RuntimeError(f"W&B reported a successful download but {temporary_path} is missing.")
        temporary_path.replace(destination)
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


def sync_run_files(run: Any, api: Any, run_dir: Path, *, mode: str) -> tuple[int, int]:
    if mode == "none":
        return 0, 0
    files_dir = run_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "files-manifest.json"
    old_manifest = read_json(manifest_path, {})
    old_files = dict(old_manifest.get("files") or {})
    new_files: dict[str, Any] = {}
    downloaded_count = 0
    skipped_count = 0

    for remote_file in run.files(per_page=100):
        name = str(remote_file.name)
        if not include_run_file(name, mode):
            continue
        signature = file_signature(remote_file)
        destination = files_dir / safe_remote_path(name)
        if old_files.get(name) == signature and local_file_matches(destination, signature):
            skipped_count += 1
        else:
            download_run_file(remote_file, api, files_dir)
            if not local_file_matches(destination, signature):
                raise RuntimeError(f"Downloaded W&B file failed size/checksum validation: {destination}")
            downloaded_count += 1
        new_files[name] = signature

    write_json_atomic(
        manifest_path,
        {
            "schema_version": SCHEMA_VERSION,
            "mode": mode,
            "files": new_files,
            "updated_at": utc_now(),
        },
    )
    return downloaded_count, skipped_count


def artifact_entry_signature(entry: Any) -> dict[str, Any]:
    """Return the stable fields needed to validate one artifact entry."""
    return {
        "size": int(getattr(entry, "size", 0) or 0),
        "digest": getattr(entry, "digest", None),
    }


def local_artifact_entry_matches(path: Path, signature: dict[str, Any]) -> bool:
    """Check a local artifact file against its W&B manifest entry."""
    if not path.is_file():
        return False
    expected_size = int(signature.get("size") or 0)
    if expected_size > 0 and path.stat().st_size != expected_size:
        return False
    digest = signature.get("digest")
    return not digest or local_md5_base64(path) == digest


def training_diagnostics_artifacts(run: Any) -> list[Any]:
    """Return this run's diagnostics artifacts in numeric version order."""
    expected_name = f"{run.id}-{TRAINING_DIAGNOSTICS_TYPE}"
    artifacts = []
    for artifact in run.logged_artifacts():
        if str(getattr(artifact, "type", "")) != TRAINING_DIAGNOSTICS_TYPE:
            continue
        name = str(getattr(artifact, "name", ""))
        if name.split(":", maxsplit=1)[0] != expected_name:
            raise RuntimeError(f"Unexpected {TRAINING_DIAGNOSTICS_TYPE} artifact for run {run.id}: {name!r}")
        version = str(getattr(artifact, "version", ""))
        if re.fullmatch(r"v\d+", version) is None:
            raise RuntimeError(f"Invalid artifact version for {name!r}: {version!r}")
        artifacts.append(artifact)
    return sorted(artifacts, key=lambda artifact: int(str(artifact.version)[1:]))


def download_artifact_entry(artifact: Any, entry_name: str, destination_root: Path) -> None:
    """Download one artifact entry and atomically place it under destination_root."""
    relative_path = safe_remote_path(entry_name)
    destination = destination_root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=".artifact-download-", dir=destination_root.parent))
    try:
        entry = artifact.get_entry(entry_name)
        downloaded_path = Path(entry.download(root=str(temporary_root)))
        if not downloaded_path.is_file():
            downloaded_path = temporary_root / relative_path
        if not downloaded_path.is_file():
            raise RuntimeError(f"W&B reported a successful download but {downloaded_path} is missing.")
        downloaded_path.replace(destination)
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


def sync_training_diagnostics_shards(
    run: Any,
    run_dir: Path,
    *,
    force: bool,
    dry_run: bool,
) -> tuple[int, int, dict[str, Any]]:
    """Mirror only raw Parquet shards from logged diagnostics artifacts."""
    root = run_dir / "logged_artifacts" / TRAINING_DIAGNOSTICS_TYPE
    versions: list[dict[str, Any]] = []
    downloaded_count = 0
    skipped_count = 0
    artifacts = training_diagnostics_artifacts(run)
    if not artifacts:
        raise RuntimeError(f"Run {run.id} has no logged {TRAINING_DIAGNOSTICS_TYPE} artifacts.")

    for artifact in artifacts:
        entries = [
            (name, entry)
            for name, entry in sorted(artifact.manifest.entries.items())
            if name.startswith(TRAINING_DIAGNOSTICS_SHARD_PREFIX) and name.endswith(".parquet")
        ]
        if not entries:
            raise RuntimeError(f"Artifact {artifact.name} contains no Parquet metric shards.")
        version_record: dict[str, Any] = {
            "name": str(artifact.name),
            "version": str(artifact.version),
            "digest": str(artifact.digest),
            "entries": {},
        }
        version_root = root / str(artifact.version)
        for entry_name, entry in entries:
            relative_path = safe_remote_path(entry_name)
            signature = artifact_entry_signature(entry)
            destination = version_root / relative_path
            if not force and local_artifact_entry_matches(destination, signature):
                skipped_count += 1
            elif dry_run:
                downloaded_count += 1
            else:
                download_artifact_entry(artifact, entry_name, version_root)
                if not local_artifact_entry_matches(destination, signature):
                    raise RuntimeError(f"Downloaded artifact entry failed validation: {destination}")
                downloaded_count += 1
            version_record["entries"][str(relative_path)] = signature
        versions.append(version_record)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": TRAINING_DIAGNOSTICS_TYPE,
        "run_id": str(run.id),
        "versions": versions,
        "updated_at": utc_now(),
    }
    if not dry_run:
        write_json_atomic(root / "manifest.json", manifest)
    return downloaded_count, skipped_count, manifest


def file_mode_covers(previous_mode: str | None, requested_mode: str) -> bool:
    order = {"none": 0, "metadata": 1, "all": 2}
    return order.get(str(previous_mode), -1) >= order[requested_mode]


def sync_one_run(
    run: Any,
    api: Any,
    project_dir: Path,
    *,
    entity: str,
    project: str,
    page_size: int,
    files_mode: str,
    force: bool,
    dry_run: bool,
    training_diagnostics_shards: bool = False,
) -> dict[str, Any]:
    run_dir = project_dir / "runs" / str(run.id)
    previous_sync = read_json(run_dir / "sync.json", {})
    probe = run_probe(run)
    same_probe = previous_sync.get("remote_probe") == probe
    files_covered = file_mode_covers(previous_sync.get("files_mode"), files_mode)

    if same_probe and files_covered and not training_diagnostics_shards and not force:
        print(f"[skip] {entity}/{project}/{run.id} {run.name}: unchanged")
        return {
            "id": str(run.id),
            "name": str(run.name),
            "state": probe["state"],
            "url": str(run.url),
            "last_history_step": probe["last_history_step"],
            "status": "unchanged",
        }

    if dry_run:
        action = "new" if not previous_sync else "update"
        print(f"[dry-run:{action}] {entity}/{project}/{run.id} {run.name} probe={canonical_json(probe)}")
        if training_diagnostics_shards:
            diagnostics_downloaded, diagnostics_skipped, _ = sync_training_diagnostics_shards(
                run,
                run_dir,
                force=force,
                dry_run=True,
            )
            print(
                f"[dry-run:artifacts] {entity}/{project}/{run.id}: "
                f"diagnostic_shards_download={diagnostics_downloaded}, "
                f"diagnostic_shards_unchanged={diagnostics_skipped}"
            )
        return {
            "id": str(run.id),
            "name": str(run.name),
            "state": probe["state"],
            "url": str(run.url),
            "last_history_step": probe["last_history_step"],
            "status": f"dry-run-{action}",
        }

    run_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(run_dir / "run.json", run_record(run, entity, project, probe))
    history_action, history_manifest = sync_history(
        run,
        run_dir,
        probe=probe,
        previous_sync=previous_sync,
        page_size=page_size,
        force=force,
    )
    files_downloaded, files_skipped = sync_run_files(run, api, run_dir, mode=files_mode)
    diagnostics_downloaded = 0
    diagnostics_skipped = 0
    diagnostics_manifest = previous_sync.get("training_diagnostics")
    if training_diagnostics_shards:
        diagnostics_downloaded, diagnostics_skipped, diagnostics_manifest = sync_training_diagnostics_shards(
            run,
            run_dir,
            force=force,
            dry_run=False,
        )
    write_json_atomic(
        run_dir / "sync.json",
        {
            "schema_version": SCHEMA_VERSION,
            "remote_probe": probe,
            "files_mode": files_mode,
            "history": history_manifest,
            "training_diagnostics": diagnostics_manifest,
            "synced_at": utc_now(),
        },
    )
    print(
        f"[sync] {entity}/{project}/{run.id} {run.name}: history={history_action}, "
        f"files_downloaded={files_downloaded}, files_unchanged={files_skipped}, "
        f"diagnostic_shards_downloaded={diagnostics_downloaded}, "
        f"diagnostic_shards_unchanged={diagnostics_skipped}"
    )
    return {
        "id": str(run.id),
        "name": str(run.name),
        "state": probe["state"],
        "url": str(run.url),
        "last_history_step": probe["last_history_step"],
        "status": history_action,
    }


def selected_runs(api: Any, project_path: str, run_ids: set[str], max_runs: int | None) -> Iterable[Any]:
    if run_ids:
        found = 0
        for run_id in sorted(run_ids):
            try:
                run = api.run(f"{project_path}/{run_id}")
            except Exception:
                continue
            yield run
            found += 1
            if max_runs is not None and found >= max_runs:
                return
        return

    for index, run in enumerate(api.runs(project_path, order="+created_at", per_page=100)):
        if max_runs is not None and index >= max_runs:
            return
        yield run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project",
        action="append",
        dest="projects",
        help="W&B project as ENTITY/PROJECT. Repeatable; defaults to jubgjf/DMPD.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "analysis_artifacts",
        help="Local mirror root (default: ./analysis_artifacts).",
    )
    parser.add_argument(
        "--files",
        choices=("none", "metadata", "all"),
        default="all",
        help="Run files to mirror in addition to config/summary/history (default: all).",
    )
    parser.add_argument("--page-size", type=int, default=1000, help="W&B history scan page size (default: 1000).")
    parser.add_argument("--timeout", type=int, default=60, help="W&B API timeout in seconds (default: 60).")
    parser.add_argument("--run-id", action="append", default=[], help="Only sync a run ID; repeatable.")
    parser.add_argument(
        "--training-diagnostics-shards",
        action="store_true",
        help="Also mirror raw Parquet shards from logged training-diagnostics artifacts.",
    )
    parser.add_argument("--max-runs", type=int, default=None, help="Limit runs per project, primarily for smoke tests.")
    parser.add_argument("--force", action="store_true", help="Refresh history and re-check all selected run files.")
    parser.add_argument("--dry-run", action="store_true", help="List new/changed runs without writing files.")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.page_size <= 0:
        raise ValueError("--page-size must be positive.")
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive.")
    if args.max_runs is not None and args.max_runs <= 0:
        raise ValueError("--max-runs must be positive when provided.")


def main() -> int:
    args = build_parser().parse_args()
    try:
        validate_args(args)
        projects = args.projects or list(DEFAULT_PROJECTS)
        parsed_projects = [(value, *split_project(value)) for value in projects]
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    output_dir = args.output_dir.expanduser().resolve()
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_DIR", str(output_dir / ".wandb"))
    os.environ.setdefault("WANDB_CACHE_DIR", str(output_dir / ".wandb-cache"))
    os.environ.setdefault("WANDB_DATA_DIR", str(output_dir / ".wandb-data"))

    try:
        import wandb
    except ImportError:
        print("error: wandb is not installed; activate the project environment or run with uv.", file=sys.stderr)
        return 2

    api = wandb.Api(timeout=args.timeout)
    run_ids = {str(value) for value in args.run_id}
    failures: list[str] = []
    root_index: list[dict[str, Any]] = []

    for project_path, entity, project in parsed_projects:
        project_dir = output_dir / entity / project
        if not args.dry_run:
            project_dir.mkdir(parents=True, exist_ok=True)
        project_runs: list[dict[str, Any]] = []
        print(f"[project] {project_path}")
        try:
            runs = list(selected_runs(api, project_path, run_ids, args.max_runs))
        except Exception as exc:
            failures.append(f"{project_path}: listing failed: {exc}")
            print(f"[error] {failures[-1]}", file=sys.stderr)
            continue

        for run in runs:
            try:
                project_runs.append(
                    sync_one_run(
                        run,
                        api,
                        project_dir,
                        entity=entity,
                        project=project,
                        page_size=args.page_size,
                        files_mode=args.files,
                        training_diagnostics_shards=args.training_diagnostics_shards,
                        force=args.force,
                        dry_run=args.dry_run,
                    )
                )
            except Exception as exc:
                message = f"{project_path}/{getattr(run, 'id', 'unknown')}: {exc}"
                failures.append(message)
                print(f"[error] {message}", file=sys.stderr)

        project_index = {
            "schema_version": SCHEMA_VERSION,
            "entity": entity,
            "project": project,
            "run_count": len(project_runs),
            "runs": project_runs,
            "updated_at": utc_now(),
        }
        if not args.dry_run:
            write_json_if_changed(project_dir / "runs.json", project_index)
        root_index.append(
            {
                "entity": entity,
                "project": project,
                "run_count": len(project_runs),
                "path": str(project_dir.relative_to(output_dir)),
            }
        )

    if not args.dry_run:
        write_json_atomic(
            output_dir / "manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "projects": root_index,
                "failures": failures,
                "updated_at": utc_now(),
            },
        )
    print(
        f"[done] output={output_dir} projects={len(root_index)} failures={len(failures)}"
        + (" (dry-run)" if args.dry_run else "")
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
