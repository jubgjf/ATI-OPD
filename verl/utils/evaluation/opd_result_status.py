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

"""Lightweight completed-result validation for the standalone OPD watcher.

This file intentionally uses only the Python standard library and can be run by
path. Importing :mod:`verl` loads the training runtime (including torch), which
is far too expensive for every checkpoint inspected by the queue scanner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2
BENCHMARKS_DIRNAME = "benchmarks"
COMPLETION_FILENAME = "COMPLETED.json"
METADATA_FILENAME = "metadata.json"
REQUIRED_RESULT_FILES = (
    METADATA_FILENAME,
    "rollouts.jsonl.gz",
    "summary.json",
    "per_problem.csv",
    "length_bins.csv",
)


def normalize_benchmark_path(raw_path: str) -> str:
    stripped = str(raw_path).strip()
    path = Path(stripped)
    if not stripped or path.is_absolute() or path == Path(".") or ".." in path.parts:
        raise ValueError(f"Benchmark must be a non-empty path relative to the dataset root, got {raw_path!r}")
    return path.as_posix()


def benchmark_id(raw_path: str) -> str:
    normalized = normalize_benchmark_path(raw_path)
    readable = re.sub(r"[^A-Za-z0-9._-]+", "__", normalized).strip("._-") or "benchmark"
    readable = readable[:80].rstrip("._-")
    return f"{readable}--{hashlib.sha256(normalized.encode()).hexdigest()[:10]}"


def unit_output_dir(result_root: str | Path, raw_path: str, global_step: int) -> Path:
    if global_step < 0:
        raise ValueError(f"global_step must be non-negative, got {global_step}")
    return (
        Path(result_root).expanduser().resolve()
        / BENCHMARKS_DIRNAME
        / benchmark_id(raw_path)
        / f"global_step_{global_step}"
    )


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_completed_unit(
    output_dir: str | Path,
    *,
    verify_checksums: bool = True,
    require_repair: bool = True,
) -> tuple[bool, str]:
    """Validate the subset used by ``opd_benchmark status`` in the watcher.

    The launcher deliberately disables mutable input-manifest validation and
    supplies no request expectations, so this function can stay dependency-free
    while preserving its completion, checksum, and paired-repair checks.
    """

    output_path = Path(output_dir)
    marker_path = output_path / COMPLETION_FILENAME
    metadata_path = output_path / METADATA_FILENAME
    marker = _read_json(marker_path) if marker_path.is_file() else None
    metadata = _read_json(metadata_path) if metadata_path.is_file() else None
    if marker is None or metadata is None:
        return False, "completion marker or metadata is missing or unreadable"
    if marker.get("schema_version") != SCHEMA_VERSION or metadata.get("schema_version") != SCHEMA_VERSION:
        return False, "schema version differs"
    if not metadata.get("generation_complete"):
        return False, "generation_complete is false"
    if marker.get("generation_fingerprint") != metadata.get("generation_fingerprint"):
        return False, "completion marker and metadata fingerprints differ"

    file_entries = marker.get("files") or {}
    for name in REQUIRED_RESULT_FILES:
        path = output_path / name
        entry = file_entries.get(name)
        if not path.is_file() or not isinstance(entry, dict):
            return False, f"required result file is missing: {name}"
        try:
            size = path.stat().st_size
        except OSError as exc:
            return False, f"required result file is unreadable: {name}: {exc}"
        if size != entry.get("size"):
            return False, f"result file size changed: {name}"
        if verify_checksums:
            try:
                checksum = _sha256_file(path)
            except OSError as exc:
                return False, f"required result file is unreadable: {name}: {exc}"
            if checksum != entry.get("sha256"):
                return False, f"result file checksum changed: {name}"
    if require_repair and not metadata.get("repair_complete"):
        return False, "paired repair is incomplete"
    return True, "complete"


def validate_checkpoint_results(
    result_root: str | Path,
    benchmarks: list[str],
    global_step: int,
    *,
    verify_checksums: bool = True,
    require_repair: bool = True,
) -> tuple[bool, str]:
    for benchmark in benchmarks:
        output_dir = unit_output_dir(result_root, benchmark, global_step)
        valid, reason = validate_completed_unit(
            output_dir,
            verify_checksums=verify_checksums,
            require_repair=require_repair,
        )
        if not valid:
            return False, f"{benchmark}: {reason}"
    return True, "complete"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--benchmark", action="append", required=True)
    parser.add_argument("--global-step", type=int, required=True)
    parser.add_argument("--no-verify-checksums", dest="verify_checksums", action="store_false")
    parser.add_argument("--no-require-repair", dest="require_repair", action="store_false")
    parser.set_defaults(verify_checksums=True, require_repair=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    valid, reason = validate_checkpoint_results(
        args.result_root,
        args.benchmark,
        args.global_step,
        verify_checksums=args.verify_checksums,
        require_repair=args.require_repair,
    )
    if valid:
        print("complete")
        return 0
    print(f"incomplete: {reason}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
