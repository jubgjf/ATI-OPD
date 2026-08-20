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

"""Standalone, resumable OPD benchmark evaluation.

The durable unit of work is one ``(experiment, benchmark, global_step)`` tuple.
Each unit has its own rollout, summary, and completion artifacts so adding a new
benchmark never invalidates already completed benchmark runs.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import copy
import csv
import fcntl
import gzip
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
import uuid
import zlib
from array import array
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator

from transformers import AutoTokenizer

from verl.utils.evaluation.math_benchmark import (
    build_math_chat_messages,
    find_benchmark_data_files,
    load_benchmark_samples,
    score_completion,
)

if TYPE_CHECKING:
    from openai import AsyncOpenAI

SCHEMA_VERSION = 2
GENERATION_SCHEMA_VERSION = 3
INPUT_MANIFEST_SCHEMA_VERSION = 2
GENERATION_COMPATIBILITY_SCHEMA_VERSION = 1
DEFAULT_BENCHMARKS = [
    "math-ai/aime24",
    "math-ai/aime25",
    "math-ai/amc23",
    "meituan-longcat/AMO-Bench",
]
DEFAULT_LENGTH_BIN_EDGES = [0, 1024, 2048, 4096, 8192, 12000, 16000, 20000, 23000]
DEFAULT_BOOTSTRAP_METRICS = [
    "acc",
    "eos_success",
    "answer_extractable",
    "accepted_answer_format",
    "truncated",
    "repetition_text_zlib",
    "repetition_token_zlib",
    "response_tokens",
    "tail_ngram_repeat_fraction",
    "gap_last_window_sampled_forward_kl",
    "gap_last_window_abs_mean",
    "gap_tail_12000_plus_sampled_forward_kl",
    "gap_tail_12000_plus_abs_mean",
]
GAP_REGION_NAMES = [
    "all",
    "head_0_4096",
    "mid_4096_12000",
    "tail_12000_plus",
    "deep_tail_16000_plus",
    "extreme_tail_20000_plus",
    "last_window",
]
INVALID_LOGPROB_FLOOR = -9990.0
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
REQUEST_PATH_KEYS = {
    "dataset_root",
    "student_model_path",
    "student_tokenizer_model",
    "chat_template_model",
}


@dataclass(frozen=True)
class Problem:
    benchmark_id: str
    benchmark_name: str
    benchmark_path: str
    question_index: int
    sample_id: str
    problem: str
    ground_truth: str
    prompt_token_ids: list[int]
    question_id: int | str | None = None
    answer_type: str | None = None
    grader: str | None = None


@dataclass(frozen=True)
class RolloutTask:
    flat_index: int
    task_id: str
    benchmark_id: str
    benchmark_name: str
    benchmark_path: str
    question_index: int
    sample_id: str
    problem: str
    ground_truth: str
    prompt_token_ids: list[int]
    rollout_index: int
    seed: int | None
    question_id: int | str | None = None
    answer_type: str | None = None
    grader: str | None = None


@dataclass(frozen=True)
class GenerationPreparation:
    tokenizer: Any
    problems: list[Problem]
    tasks: list[RolloutTask]
    dataset_fingerprint: str
    input_manifest: dict[str, Any]
    tokenizer_info: dict[str, Any]
    request: dict[str, Any]
    analysis: dict[str, Any]
    fingerprint: str


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "__dataclass_fields__") and not isinstance(value, type):
        return asdict(value)
    return str(value)


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=json_default)


def payload_fingerprint(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _atomic_temporary_path(path)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=json_default)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_csv_atomic(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _atomic_temporary_path(path)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: "" if value is None else value for key, value in row.items()})
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_jsonl_gzip(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, compresslevel=6, mtime=0) as gzip_handle:
            with io.TextIOWrapper(gzip_handle, encoding="utf-8") as text_handle:
                for record in records:
                    text_handle.write(canonical_json(record))
                    text_handle.write("\n")
        raw_handle.flush()
        os.fsync(raw_handle.fileno())


def write_jsonl_gz_atomic(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _atomic_temporary_path(path)
    try:
        _write_jsonl_gzip(temporary, records)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_gz_atomic(path: Path, payload: dict[str, Any]) -> None:
    write_jsonl_gz_atomic(path, [payload])


def iter_jsonl_gz(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}, line {line_number}") from exc


def read_single_json_gz(path: Path) -> dict[str, Any] | None:
    try:
        records = list(iter_jsonl_gz(path))
    except (OSError, ValueError):
        return None
    if len(records) != 1 or not isinstance(records[0], dict):
        return None
    return records[0]


def normalize_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if not normalized.endswith("/v1"):
        normalized += "/v1"
    return normalized + "/"


def object_field(value: Any, name: str, default: Any = None) -> Any:
    if value is None:
        return default
    direct = getattr(value, name, None)
    if direct is not None:
        return direct
    model_extra = getattr(value, "model_extra", None)
    if isinstance(model_extra, dict) and name in model_extra:
        return model_extra[name]
    if isinstance(value, dict):
        return value.get(name, default)
    return default


def stable_seed(*parts: Any) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFFFFFF


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


def benchmark_root(result_root: str | Path, raw_path: str) -> Path:
    return Path(result_root).expanduser().resolve() / BENCHMARKS_DIRNAME / benchmark_id(raw_path)


def unit_output_dir(result_root: str | Path, raw_path: str, global_step: int) -> Path:
    if global_step < 0:
        raise ValueError(f"global_step must be non-negative, got {global_step}")
    return benchmark_root(result_root, raw_path) / f"global_step_{global_step}"


@contextlib.contextmanager
def unit_lock(output_dir: Path) -> Iterator[None]:
    # The standalone launcher uses atomic checkpoint claims for cross-host exclusion.
    # This flock remains a local safeguard for direct module invocations only.
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _weight_files(model_dir: Path) -> list[Path]:
    patterns = ("*.safetensors", "pytorch_model*.bin", "model*.bin")
    return sorted({path for pattern in patterns for path in model_dir.glob(pattern) if path.is_file()})


def validate_hf_checkpoint(model_dir: str | Path) -> tuple[bool, str]:
    path = Path(model_dir).expanduser().resolve()
    config_path = path / "config.json"
    if not config_path.is_file():
        return False, f"missing {config_path}"

    index_paths = [path / "model.safetensors.index.json", path / "pytorch_model.bin.index.json"]
    existing_indexes = [index_path for index_path in index_paths if index_path.is_file()]
    if existing_indexes:
        referenced: set[str] = set()
        for index_path in existing_indexes:
            try:
                payload = json.loads(index_path.read_text(encoding="utf-8"))
                referenced.update(str(value) for value in (payload.get("weight_map") or {}).values())
            except Exception as exc:
                return False, f"invalid weight index {index_path}: {exc}"
        if not referenced:
            return False, f"weight index under {path} contains no shards"
        missing = [
            name for name in sorted(referenced) if not (path / name).is_file() or (path / name).stat().st_size <= 0
        ]
        if missing:
            return False, f"missing or empty weight shards under {path}: {missing[:5]}"
    else:
        weights = _weight_files(path)
        if not weights or any(weight.stat().st_size <= 0 for weight in weights):
            return False, f"no complete model weights found under {path}"

    tokenizer_config = path / "tokenizer_config.json"
    tokenizer_assets = [
        path / "tokenizer.json",
        path / "tokenizer.model",
        path / "spiece.model",
        path / "vocab.json",
        path / "vocab.txt",
    ]
    if not tokenizer_config.is_file() or not any(asset.is_file() for asset in tokenizer_assets):
        return False, f"tokenizer files are incomplete under {path}"
    return True, "ready"


def _file_descriptor(
    path: Path,
    *,
    content_hash: bool,
    logical_path: str | None = None,
) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    descriptor: dict[str, Any] = {
        "path": str(resolved),
        "size": stat.st_size,
        "mode": "sha256" if content_hash else "stat",
    }
    if logical_path is not None:
        descriptor["logical_path"] = str(logical_path)
    if content_hash:
        descriptor["sha256"] = sha256_file(resolved)
    else:
        descriptor["mtime_ns"] = stat.st_mtime_ns
    return descriptor


def _dataset_files(dataset_root: str | Path, raw_benchmark_path: str) -> list[Path]:
    return [
        path.resolve() for path in find_benchmark_data_files(dataset_root, normalize_benchmark_path(raw_benchmark_path))
    ]


def _tokenizer_files(model_dir: str | Path) -> list[Path]:
    directory = Path(model_dir).expanduser().resolve()
    names = (
        "tokenizer_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "spiece.model",
        "vocab.json",
        "vocab.txt",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
    )
    return [directory / name for name in names if (directory / name).is_file()]


def _relative_file_descriptor(path: Path, root: Path, *, content_hash: bool) -> dict[str, Any]:
    return _file_descriptor(
        path,
        content_hash=content_hash,
        logical_path=path.expanduser().resolve().relative_to(root.expanduser().resolve()).as_posix(),
    )


def _semantic_file_descriptor(entry: Mapping[str, Any]) -> dict[str, Any]:
    descriptor = {
        "logical_path": str(entry["logical_path"]),
        "size": int(entry["size"]),
        "mode": str(entry["mode"]),
    }
    if entry.get("mode") == "sha256":
        descriptor["sha256"] = str(entry["sha256"])
    return descriptor


def _stable_input_manifest_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    groups = (
        "dataset",
        "student_model_metadata",
        "student_model_weights",
        "student_tokenizer",
        "chat_template",
    )
    return {
        "schema_version": INPUT_MANIFEST_SCHEMA_VERSION,
        "student_model_identity": manifest["student_model_identity"],
        **{
            group: sorted(
                (_semantic_file_descriptor(entry) for entry in manifest.get(group, [])),
                key=lambda entry: entry["logical_path"],
            )
            for group in groups
        },
    }


def stable_input_manifest_fingerprint(manifest: Mapping[str, Any]) -> str:
    """Fingerprint semantic inputs without deployment paths or merged-file mtimes."""

    return payload_fingerprint(_stable_input_manifest_payload(manifest))


def _weight_content_identity(weight_files: list[Path], model_dir: Path) -> dict[str, Any]:
    files = [
        {
            "logical_path": path.resolve().relative_to(model_dir.resolve()).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in weight_files
    ]
    return {
        "kind": "weight_content_sha256",
        "fingerprint": payload_fingerprint(files),
        "files": files,
    }


def capture_input_manifest(
    *,
    dataset_root: str | Path,
    raw_benchmark_path: str,
    student_model_path: str | Path,
    student_tokenizer_model: str | Path,
    chat_template_model: str | Path,
    student_model_identity: str | None = None,
) -> dict[str, Any]:
    model_dir = Path(student_model_path).expanduser().resolve()
    ready, reason = validate_hf_checkpoint(model_dir)
    if not ready:
        raise RuntimeError(f"Student Hugging Face checkpoint is not ready: {reason}")

    model_small_files = [
        path
        for name in (
            "config.json",
            "generation_config.json",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        )
        if (path := model_dir / name).is_file()
    ]
    dataset_dir = Path(dataset_root).expanduser().resolve()
    student_tokenizer_dir = Path(student_tokenizer_model).expanduser().resolve()
    chat_template_dir = Path(chat_template_model).expanduser().resolve()
    weight_files = _weight_files(model_dir)
    if student_model_identity:
        model_identity = {
            "kind": "committed_checkpoint",
            "fingerprint": str(student_model_identity),
        }
    else:
        model_identity = _weight_content_identity(weight_files, model_dir)
    manifest = {
        "schema_version": INPUT_MANIFEST_SCHEMA_VERSION,
        "student_model_identity": model_identity,
        "dataset": [
            _relative_file_descriptor(path, dataset_dir, content_hash=True)
            for path in _dataset_files(dataset_root, raw_benchmark_path)
        ],
        "student_model_metadata": [
            _relative_file_descriptor(path, model_dir, content_hash=True) for path in model_small_files
        ],
        "student_model_weights": [
            _relative_file_descriptor(path, model_dir, content_hash=False) for path in weight_files
        ],
        "student_tokenizer": [
            _relative_file_descriptor(path, student_tokenizer_dir, content_hash=True)
            for path in _tokenizer_files(student_tokenizer_model)
        ],
        "chat_template": [
            _relative_file_descriptor(path, chat_template_dir, content_hash=True)
            for path in _tokenizer_files(chat_template_model)
        ],
    }
    manifest["fingerprint"] = stable_input_manifest_fingerprint(manifest)
    return manifest


def capture_model_manifest(model_dir: str | Path) -> dict[str, Any]:
    path = Path(model_dir).expanduser().resolve()
    ready, reason = validate_hf_checkpoint(path)
    if not ready:
        raise RuntimeError(f"Hugging Face checkpoint is not ready: {reason}")
    metadata_files = [
        path / name
        for name in (
            "config.json",
            "generation_config.json",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        )
        if (path / name).is_file()
    ]
    manifest = {
        "model_metadata": [_file_descriptor(file_path, content_hash=True) for file_path in metadata_files],
        "model_weights": [_file_descriptor(file_path, content_hash=False) for file_path in _weight_files(path)],
        "tokenizer": [_file_descriptor(file_path, content_hash=True) for file_path in _tokenizer_files(path)],
    }
    manifest["fingerprint"] = payload_fingerprint(manifest)
    return manifest


def validate_input_manifest(manifest: dict[str, Any]) -> tuple[bool, str]:
    expected_fingerprint = manifest.get("fingerprint")
    if manifest.get("schema_version") == INPUT_MANIFEST_SCHEMA_VERSION:
        refreshed: dict[str, Any] = {
            "schema_version": INPUT_MANIFEST_SCHEMA_VERSION,
            "student_model_identity": dict(manifest.get("student_model_identity") or {}),
        }
        try:
            for group_name in (
                "dataset",
                "student_model_metadata",
                "student_model_weights",
                "student_tokenizer",
                "chat_template",
            ):
                refreshed[group_name] = [
                    _file_descriptor(
                        Path(entry["path"]),
                        content_hash=entry.get("mode") == "sha256",
                        logical_path=str(entry["logical_path"]),
                    )
                    for entry in manifest.get(group_name, [])
                ]
            identity = refreshed["student_model_identity"]
            if identity.get("kind") == "weight_content_sha256":
                weight_entries = refreshed["student_model_weights"]
                files = [
                    {
                        "logical_path": str(entry["logical_path"]),
                        "size": int(entry["size"]),
                        "sha256": sha256_file(Path(entry["path"])),
                    }
                    for entry in weight_entries
                ]
                refreshed["student_model_identity"] = {
                    "kind": "weight_content_sha256",
                    "fingerprint": payload_fingerprint(files),
                    "files": files,
                }
        except (KeyError, OSError, ValueError) as exc:
            return False, f"input file is missing or unreadable: {exc}"
        actual_fingerprint = stable_input_manifest_fingerprint(refreshed)
        if actual_fingerprint != expected_fingerprint:
            return False, "input dataset/model/tokenizer fingerprint changed"
        return True, "valid"

    refreshed: dict[str, Any] = {}
    try:
        for group_name, entries in manifest.items():
            if group_name == "fingerprint":
                continue
            refreshed[group_name] = [
                _file_descriptor(Path(entry["path"]), content_hash=entry.get("mode") == "sha256") for entry in entries
            ]
    except (KeyError, OSError) as exc:
        return False, f"input file is missing or unreadable: {exc}"
    actual_fingerprint = payload_fingerprint(refreshed)
    if actual_fingerprint != expected_fingerprint:
        return False, "input dataset/model/tokenizer fingerprint changed"
    return True, "valid"


def tokenizer_fingerprint(tokenizer: Any) -> str:
    digest = hashlib.sha256()
    for token, token_id in sorted(tokenizer.get_vocab().items(), key=lambda item: (int(item[1]), str(item[0]))):
        digest.update(str(int(token_id)).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(token).encode("utf-8", errors="surrogatepass"))
        digest.update(b"\n")
    return digest.hexdigest()


def tokenizer_metadata(tokenizer: Any, model_path: str) -> dict[str, Any]:
    chat_template = str(getattr(tokenizer, "chat_template", "") or "")
    return {
        "model_path": str(Path(model_path).expanduser().resolve()),
        "vocab_size": len(tokenizer),
        "fingerprint_sha256": tokenizer_fingerprint(tokenizer),
        "chat_template_sha256": hashlib.sha256(chat_template.encode("utf-8")).hexdigest(),
        "special_token_ids": {
            name: getattr(tokenizer, name, None)
            for name in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id")
        },
    }


def source_metadata() -> dict[str, Any]:
    module_path = Path(__file__).resolve()
    result: dict[str, Any] = {"module": str(module_path), "module_sha256": sha256_file(module_path)}
    try:
        root = Path(__file__).resolve().parents[3]
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(["git", "diff", "--quiet"], cwd=root, check=False).returncode != 0
        result.update({"git_revision": revision, "git_dirty": dirty})
    except Exception:
        result.update({"git_revision": None, "git_dirty": None})
    return result


def load_eval_tokenizer(args: argparse.Namespace) -> Any:
    student_tokenizer = AutoTokenizer.from_pretrained(
        args.student_tokenizer_model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=True,
    )
    chat_template_tokenizer = AutoTokenizer.from_pretrained(
        args.chat_template_model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=True,
    )
    chat_template = getattr(chat_template_tokenizer, "chat_template", None)
    if not chat_template:
        raise ValueError(f"Tokenizer at {args.chat_template_model!r} has no chat_template")
    student_tokenizer.chat_template = chat_template
    return student_tokenizer


def apply_chat_template(tokenizer: Any, messages: list[dict[str, str]], enable_thinking: bool) -> list[int]:
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
    }
    try:
        token_ids = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        token_ids = tokenizer.apply_chat_template(messages, **kwargs)
    if isinstance(token_ids, Mapping):
        if "input_ids" not in token_ids:
            raise ValueError(f"Chat template returned a mapping without input_ids: {sorted(token_ids)}")
        token_ids = token_ids["input_ids"]
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        if len(token_ids) != 1:
            raise ValueError(f"Unexpected chat-template batch size: {len(token_ids)}")
        token_ids = token_ids[0]
    normalized = [int(token_id) for token_id in token_ids]
    if not normalized:
        raise ValueError("Chat template produced an empty prompt")
    return normalized


def prepare_tasks(args: argparse.Namespace, tokenizer: Any) -> tuple[list[Problem], list[RolloutTask], str]:
    normalized_path = normalize_benchmark_path(args.benchmark)
    identity = benchmark_id(normalized_path)
    name = Path(normalized_path).name
    max_samples = None if args.max_samples_per_benchmark < 0 else args.max_samples_per_benchmark
    samples = load_benchmark_samples(args.dataset_root, normalized_path, max_samples=max_samples)
    if not samples:
        raise ValueError(f"No samples found for benchmark {normalized_path!r}")
    sample_ids = [str(sample.sample_id) for sample in samples]
    duplicates = sorted(sample_id for sample_id in set(sample_ids) if sample_ids.count(sample_id) > 1)
    if duplicates:
        raise ValueError(f"Benchmark {normalized_path!r} has duplicate sample IDs: {duplicates[:10]}")

    problems: list[Problem] = []
    tasks: list[RolloutTask] = []
    dataset_records: list[dict[str, Any]] = []
    for question_index, sample in enumerate(samples):
        messages = build_math_chat_messages(
            sample.problem,
            idx=question_index,
            use_raw_prompt=bool(getattr(sample, "use_raw_prompt", False)),
        )
        prompt_token_ids = apply_chat_template(tokenizer, messages, args.enable_thinking)
        if len(prompt_token_ids) > args.max_prompt_tokens:
            raise ValueError(
                f"{normalized_path}/{sample.sample_id} has {len(prompt_token_ids)} prompt tokens, "
                f"exceeding --max-prompt-tokens={args.max_prompt_tokens}"
            )
        problem = Problem(
            benchmark_id=identity,
            benchmark_name=name,
            benchmark_path=normalized_path,
            question_index=question_index,
            sample_id=str(sample.sample_id),
            problem=str(sample.problem),
            ground_truth=str(sample.ground_truth),
            prompt_token_ids=prompt_token_ids,
            question_id=getattr(sample, "question_id", None),
            answer_type=getattr(sample, "answer_type", None),
            grader=getattr(sample, "grader", None),
        )
        problems.append(problem)
        dataset_records.append(
            {
                "sample_id": problem.sample_id,
                "problem": problem.problem,
                "ground_truth": problem.ground_truth,
                "question_id": problem.question_id,
                "answer_type": problem.answer_type,
                "grader": problem.grader,
            }
        )
        for rollout_index in range(args.rollout_n):
            seed = (
                stable_seed(args.base_seed, normalized_path, problem.sample_id, rollout_index)
                if args.base_seed is not None
                else None
            )
            task_id = hashlib.sha256(f"{normalized_path}\0{problem.sample_id}\0{rollout_index}".encode()).hexdigest()[
                :24
            ]
            tasks.append(
                RolloutTask(
                    flat_index=len(tasks),
                    task_id=task_id,
                    benchmark_id=identity,
                    benchmark_name=name,
                    benchmark_path=normalized_path,
                    question_index=question_index,
                    sample_id=problem.sample_id,
                    problem=problem.problem,
                    ground_truth=problem.ground_truth,
                    prompt_token_ids=prompt_token_ids,
                    rollout_index=rollout_index,
                    seed=seed,
                    question_id=problem.question_id,
                    answer_type=problem.answer_type,
                    grader=problem.grader,
                )
            )
    return problems, tasks, payload_fingerprint(dataset_records)


def generation_request(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "experiment": args.experiment,
        "global_step": args.global_step,
        "benchmark": normalize_benchmark_path(args.benchmark),
        "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
        "student_model_path": str(Path(args.student_model_path).expanduser().resolve()),
        "student_tokenizer_model": str(Path(args.student_tokenizer_model).expanduser().resolve()),
        "chat_template_model": str(Path(args.chat_template_model).expanduser().resolve()),
        "rollout_n": args.rollout_n,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "max_prompt_tokens": args.max_prompt_tokens,
        "base_seed": args.base_seed,
        "enable_thinking": args.enable_thinking,
        "verifier": args.verifier,
        "score_student_logprobs": args.score_student_logprobs,
        "require_exact_token_ids": args.require_exact_token_ids,
        "max_samples_per_benchmark": args.max_samples_per_benchmark,
    }


def analysis_request(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "length_bin_edges": normalize_length_edges(args.length_bin_edges),
        "repetition_tail_chars": args.repetition_tail_chars,
        "repetition_tail_token_window": args.repetition_tail_token_window,
        "repetition_threshold": args.repetition_threshold,
        "ngram_size": args.ngram_size,
        "high_gap_threshold": args.high_gap_threshold,
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.bootstrap_seed,
    }


def generation_request_identity(request: Mapping[str, Any]) -> dict[str, Any]:
    """Return sampling semantics without host-specific deployment paths."""

    return {str(key): value for key, value in request.items() if key not in REQUEST_PATH_KEYS}


def tokenizer_identity(tokenizer_info: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in tokenizer_info.items() if key != "model_path"}


def generation_fingerprint(
    *,
    request: dict[str, Any],
    input_manifest: dict[str, Any],
    dataset_fingerprint: str,
    tokenizer_info: dict[str, Any],
) -> str:
    return payload_fingerprint(
        {
            "generation_schema_version": GENERATION_SCHEMA_VERSION,
            "request": generation_request_identity(request),
            "input_manifest_fingerprint": input_manifest["fingerprint"],
            "dataset_fingerprint": dataset_fingerprint,
            "tokenizer": tokenizer_identity(tokenizer_info),
        }
    )


def percentile(values: list[float], probability: float) -> float | None:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return None
    if len(finite) == 1:
        return finite[0]
    position = min(max(probability, 0.0), 1.0) * (len(finite) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return finite[lower]
    fraction = position - lower
    return finite[lower] * (1.0 - fraction) + finite[upper] * fraction


def finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def mean_present(values: Iterable[Any]) -> float | None:
    finite = [value for raw in values if (value := finite_float(raw)) is not None]
    return statistics.fmean(finite) if finite else None


def distribution(values: Iterable[Any]) -> dict[str, float | int | None]:
    finite = [value for raw in values if (value := finite_float(raw)) is not None]
    if not finite:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": len(finite),
        "mean": statistics.fmean(finite),
        "std": statistics.pstdev(finite) if len(finite) > 1 else 0.0,
        "min": min(finite),
        "p50": percentile(finite, 0.50),
        "p75": percentile(finite, 0.75),
        "p90": percentile(finite, 0.90),
        "p95": percentile(finite, 0.95),
        "p99": percentile(finite, 0.99),
        "max": max(finite),
    }


def compression_ratio(payload: bytes) -> float | None:
    if not payload:
        return None
    compressed = zlib.compress(payload)
    return len(payload) / len(compressed) if compressed else None


def text_repetition_metrics(text: str, tail_chars: int, threshold: float) -> dict[str, Any]:
    eligible = len(text) >= tail_chars
    ratio = compression_ratio(text[-tail_chars:].encode("utf-8")) if eligible else None
    return {
        "repetition_text_zlib": bool(eligible and ratio is not None and ratio > threshold),
        "repetition_text_zlib_eligible": eligible,
        "repetition_text_zlib_ratio": ratio,
    }


def token_repetition_metrics(
    token_ids: list[int], tail_token_window: int, threshold: float, ngram_size: int
) -> dict[str, Any]:
    eligible = len(token_ids) >= tail_token_window
    ratio: float | None = None
    if eligible:
        ratio = compression_ratio(array("q", token_ids[-tail_token_window:]).tobytes())
    ngram_tail = token_ids[-min(len(token_ids), tail_token_window) :]
    ngram_count = max(len(ngram_tail) - ngram_size + 1, 0)
    if ngram_count:
        unique_count = len({tuple(ngram_tail[index : index + ngram_size]) for index in range(ngram_count)})
        repeat_fraction = 1.0 - unique_count / ngram_count
    else:
        repeat_fraction = None
    return {
        "repetition_token_zlib": bool(eligible and ratio is not None and ratio > threshold),
        "repetition_token_zlib_eligible": eligible,
        "repetition_token_zlib_ratio": ratio,
        "tail_ngram_repeat_fraction": repeat_fraction,
        "ngram_size": ngram_size,
    }


def is_valid_logprob(value: Any) -> bool:
    parsed = finite_float(value)
    return parsed is not None and INVALID_LOGPROB_FLOOR < parsed <= 1e-5


def gap_region(
    student_logprobs: list[Any],
    teacher_logprobs: list[Any],
    start: int,
    end: int,
    high_gap_threshold: float,
) -> dict[str, Any]:
    start = max(0, start)
    end = max(start, min(end, len(student_logprobs), len(teacher_logprobs)))
    signed_gaps: list[float] = []
    absolute_gaps: list[float] = []
    student_probs: list[float] = []
    teacher_probs: list[float] = []
    invalid_count = 0
    high_gap_count = 0
    for position in range(start, end):
        student = student_logprobs[position]
        teacher = teacher_logprobs[position]
        if not is_valid_logprob(student) or not is_valid_logprob(teacher):
            invalid_count += 1
            continue
        student_value = float(student)
        teacher_value = float(teacher)
        gap = student_value - teacher_value
        absolute = abs(gap)
        signed_gaps.append(gap)
        absolute_gaps.append(absolute)
        student_probs.append(math.exp(max(student_value, -100.0)))
        teacher_probs.append(math.exp(max(teacher_value, -100.0)))
        high_gap_count += int(absolute > high_gap_threshold)
    token_count = len(signed_gaps)
    return {
        "requested_token_count": end - start,
        "valid_token_count": token_count,
        "invalid_token_count": invalid_count,
        "sampled_forward_kl": statistics.fmean(signed_gaps) if signed_gaps else None,
        "abs_mean": statistics.fmean(absolute_gaps) if absolute_gaps else None,
        "abs_p90": percentile(absolute_gaps, 0.90),
        "high_gap_rate": high_gap_count / token_count if token_count else None,
        "student_sampled_prob_mean": statistics.fmean(student_probs) if student_probs else None,
        "teacher_sampled_prob_mean": statistics.fmean(teacher_probs) if teacher_probs else None,
    }


def compute_gap_metrics(
    student_logprobs: list[Any],
    teacher_logprobs: list[Any],
    high_gap_threshold: float,
    tail_token_window: int,
) -> dict[str, Any]:
    if len(student_logprobs) != len(teacher_logprobs):
        raise ValueError(
            f"Student and teacher response logprobs do not align: {len(student_logprobs)} != {len(teacher_logprobs)}"
        )
    length = len(student_logprobs)
    regions = {
        "all": (0, length),
        "head_0_4096": (0, min(4096, length)),
        "mid_4096_12000": (min(4096, length), min(12000, length)),
        "tail_12000_plus": (min(12000, length), length),
        "deep_tail_16000_plus": (min(16000, length), length),
        "extreme_tail_20000_plus": (min(20000, length), length),
        "last_window": (max(0, length - tail_token_window), length),
    }
    return {
        name: gap_region(student_logprobs, teacher_logprobs, start, end, high_gap_threshold)
        for name, (start, end) in regions.items()
    }


async def check_served_model(client: AsyncOpenAI, served_model_name: str) -> None:
    response = await client.models.list()
    available = [model.id for model in response.data]
    if served_model_name not in available:
        raise RuntimeError(f"Model {served_model_name!r} is not served; available={available}")


async def score_sequence_logprobs(
    *,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    served_model_name: str,
    prompt_token_ids: list[int],
    response_token_ids: list[int],
) -> tuple[list[float | None], bool]:
    if not response_token_ids:
        return [], False
    sequence_ids = [*prompt_token_ids, *response_token_ids]

    async def request(extra_body: dict[str, Any] | None) -> Any:
        kwargs: dict[str, Any] = {
            "model": served_model_name,
            "prompt": sequence_ids,
            "n": 1,
            "max_tokens": 1,
            "temperature": 0.0,
            "top_p": 1.0,
            "echo": True,
            "logprobs": 1,
            "stream": False,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body
        async with semaphore:
            return await client.completions.create(**kwargs)

    fallback_used = False
    try:
        response = await request({"return_token_ids": True})
    except Exception:
        fallback_used = True
        response = await request(None)
    if len(response.choices) != 1:
        raise RuntimeError(f"Expected one scoring choice, got {len(response.choices)}")
    logprobs_object = object_field(response.choices[0], "logprobs")
    token_logprobs = object_field(logprobs_object, "token_logprobs")
    if token_logprobs is None:
        raise RuntimeError("vLLM echo scoring did not return token_logprobs")
    values = list(token_logprobs)
    if len(values) < len(sequence_ids):
        raise RuntimeError(
            f"vLLM echo scoring returned {len(values)} token logprobs for a {len(sequence_ids)}-token sequence"
        )
    response_values = values[len(prompt_token_ids) : len(prompt_token_ids) + len(response_token_ids)]
    if len(response_values) != len(response_token_ids):
        raise RuntimeError("Could not align response token logprobs")
    return [None if value is None else float(value) for value in response_values], fallback_used


async def _request_completion(
    *,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    args: argparse.Namespace,
    task: RolloutTask,
) -> tuple[Any, bool]:
    extra_body: dict[str, Any] = {"top_k": args.top_k, "return_token_ids": True}
    if args.score_student_logprobs:
        extra_body["return_tokens_as_token_ids"] = True
    if task.seed is not None:
        extra_body["seed"] = task.seed

    async def request(body: dict[str, Any]) -> Any:
        async with semaphore:
            request_kwargs: dict[str, Any] = {
                "model": args.served_model_name,
                "prompt": task.prompt_token_ids,
                "n": 1,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "stream": False,
                "extra_body": body,
            }
            if args.score_student_logprobs:
                # vLLM returns the realized-token probability as part of the
                # generation response. This avoids a second echo forward in the
                # normal standalone-repair path.
                request_kwargs["logprobs"] = 1
            return await client.completions.create(
                **request_kwargs,
            )

    try:
        return await request(extra_body), False
    except Exception as exc:
        if args.require_exact_token_ids:
            raise RuntimeError(
                "vLLM rejected or failed the completion request with return_token_ids enabled; "
                "exact generated token IDs are required"
            ) from exc
        fallback_body = dict(extra_body)
        fallback_body.pop("return_token_ids", None)
        return await request(fallback_body), True


async def evaluate_one(
    *,
    client: AsyncOpenAI,
    generation_semaphore: asyncio.Semaphore,
    scoring_semaphore: asyncio.Semaphore,
    args: argparse.Namespace,
    tokenizer: Any,
    task: RolloutTask,
    fingerprint: str,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    base_record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generation_fingerprint": fingerprint,
        "task_id": task.task_id,
        "flat_index": task.flat_index,
        "benchmark_id": task.benchmark_id,
        "benchmark": task.benchmark_name,
        "benchmark_path": task.benchmark_path,
        "question_index": task.question_index,
        "sample_id": task.sample_id,
        "question_id": task.question_id,
        "answer_type": task.answer_type,
        "grader": task.grader,
        "rollout_index": task.rollout_index,
        "seed": task.seed,
        "problem": task.problem,
        "ground_truth": task.ground_truth,
        "problem_fingerprint": payload_fingerprint(
            {
                "sample_id": task.sample_id,
                "problem": task.problem,
                "ground_truth": task.ground_truth,
                "question_id": task.question_id,
                "answer_type": task.answer_type,
                "grader": task.grader,
            }
        ),
        "prompt_token_ids": task.prompt_token_ids,
    }
    try:
        response, request_fallback = await _request_completion(
            client=client,
            semaphore=generation_semaphore,
            args=args,
            task=task,
        )
        if len(response.choices) != 1:
            raise RuntimeError(f"Expected one completion, got {len(response.choices)}")
        choice = response.choices[0]
        text = str(choice.text or "")
        finish_reason_raw = object_field(choice, "finish_reason")
        finish_reason = str(finish_reason_raw).lower() if finish_reason_raw is not None else None
        stop_reason = object_field(choice, "stop_reason")
        usage = object_field(response, "usage")
        usage_completion_tokens = object_field(usage, "completion_tokens")
        if usage_completion_tokens is not None:
            usage_completion_tokens = int(usage_completion_tokens)

        returned_token_ids = object_field(choice, "token_ids")
        if returned_token_ids is None:
            if args.require_exact_token_ids:
                raise RuntimeError("vLLM completion response omitted token_ids while exact token IDs are required")
            response_token_ids = [int(token_id) for token_id in tokenizer.encode(text, add_special_tokens=False)]
            token_ids_source = "retokenized_text"
        else:
            response_token_ids = [int(token_id) for token_id in returned_token_ids]
            token_ids_source = "vllm"

        student_logprobs: list[float | None] | None = None
        scoring_fallback: bool | None = None
        student_logprobs_source: str | None = None
        if args.score_student_logprobs:
            choice_logprobs = object_field(choice, "logprobs")
            direct_values = object_field(choice_logprobs, "token_logprobs")
            values = list(direct_values) if direct_values is not None else []
            if direct_values is not None and len(values) >= len(response_token_ids):
                values = values[-len(response_token_ids) :] if response_token_ids else []
                student_logprobs = [None if value is None else float(value) for value in values]
                scoring_fallback = False
                student_logprobs_source = "generation_response"
            else:
                # Compatibility fallback for older vLLM builds that accept the
                # request but omit generation logprobs. It is recorded so the
                # watcher can surface the extra forward instead of hiding it.
                student_logprobs, scoring_fallback = await score_sequence_logprobs(
                    client=client,
                    semaphore=scoring_semaphore,
                    served_model_name=args.served_model_name,
                    prompt_token_ids=task.prompt_token_ids,
                    response_token_ids=response_token_ids,
                )
                student_logprobs_source = "echo_scoring_fallback"
        response_tokens = usage_completion_tokens if usage_completion_tokens is not None else len(response_token_ids)
        scored = score_completion(
            text,
            task.ground_truth,
            finish_reason=finish_reason,
            response_token_count=response_tokens,
            verifier=args.verifier,
            extra_info={
                "benchmark": task.benchmark_name,
                "benchmark_path": task.benchmark_path,
                "question_id": task.question_id,
                "answer_type": task.answer_type,
                "grader": task.grader,
            },
        )
        eos_success = finish_reason in {"stop", "eos"}
        repetition = text_repetition_metrics(
            text,
            tail_chars=args.repetition_tail_chars,
            threshold=args.repetition_threshold,
        )
        repetition.update(
            token_repetition_metrics(
                response_token_ids,
                tail_token_window=args.repetition_tail_token_window,
                threshold=args.repetition_threshold,
                ngram_size=args.ngram_size,
            )
        )
        return {
            **base_record,
            "text": text,
            "response_token_ids": response_token_ids,
            "response_token_ids_source": token_ids_source,
            "student_logprobs": student_logprobs,
            "student_logprobs_source": student_logprobs_source,
            "teacher_logprobs": None,
            "gap_metrics": None,
            "finish_reason": finish_reason,
            "stop_reason": stop_reason,
            "usage_completion_tokens": usage_completion_tokens,
            "response_tokens": response_tokens,
            "response_token_id_count": len(response_token_ids),
            "token_id_count_minus_usage_tokens": (
                len(response_token_ids) - usage_completion_tokens if usage_completion_tokens is not None else None
            ),
            "eos_success": eos_success,
            "truncated": finish_reason == "length",
            "acc": bool(scored.acc),
            "pred": str(scored.pred),
            "final_content": str(scored.final_content),
            "invalid_pred": bool(scored.invalid_pred),
            "answer_extractable": not bool(scored.invalid_pred),
            "has_answer_pattern": bool(scored.has_answer_pattern),
            "has_boxed_pattern": bool(scored.has_boxed_pattern),
            "accepted_answer_format": bool(scored.has_accepted_answer_format),
            "verified_by": str(scored.verified_by),
            "math_verify_score": scored.math_verify_score,
            "repetition": repetition,
            "generation_return_token_ids_fallback": request_fallback,
            "student_scoring_return_token_ids_fallback": scoring_fallback,
            "elapsed_seconds": time.perf_counter() - started_at,
            "error": None,
        }
    except Exception as exc:
        return {
            **base_record,
            "elapsed_seconds": time.perf_counter() - started_at,
            "error": f"{type(exc).__name__}: {exc}",
        }


async def bounded_results(tasks: Iterable[RolloutTask], worker: Any, max_pending: int) -> Any:
    iterator = iter(tasks)
    pending: set[asyncio.Task[Any]] = set()

    def fill() -> None:
        while len(pending) < max_pending:
            try:
                task = next(iterator)
            except StopIteration:
                return
            pending.add(asyncio.create_task(worker(task)))

    fill()
    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for future in done:
            yield await future
        fill()


def lightweight_record(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("error"):
        return {
            "benchmark": record.get("benchmark"),
            "sample_id": record.get("sample_id"),
            "rollout_index": record.get("rollout_index"),
            "error": record.get("error"),
        }
    repetition = record.get("repetition") or {}
    light: dict[str, Any] = {
        "benchmark": record["benchmark"],
        "benchmark_path": record.get("benchmark_path"),
        "question_index": record["question_index"],
        "sample_id": str(record["sample_id"]),
        "problem_fingerprint": record.get("problem_fingerprint"),
        "rollout_index": record["rollout_index"],
        "acc": float(bool(record.get("acc"))),
        "eos_success": float(bool(record.get("eos_success"))),
        "answer_extractable": float(bool(record.get("answer_extractable"))),
        "has_answer_pattern": float(bool(record.get("has_answer_pattern"))),
        "accepted_answer_format": float(bool(record.get("accepted_answer_format"))),
        "has_boxed_pattern": float(bool(record.get("has_boxed_pattern"))),
        "truncated": float(bool(record.get("truncated"))),
        "response_tokens": int(record.get("response_tokens", 0)),
        "repetition_text_zlib": float(bool(repetition.get("repetition_text_zlib"))),
        "repetition_text_zlib_eligible": float(bool(repetition.get("repetition_text_zlib_eligible"))),
        "repetition_text_zlib_ratio": repetition.get("repetition_text_zlib_ratio"),
        "repetition_token_zlib": float(bool(repetition.get("repetition_token_zlib"))),
        "repetition_token_zlib_eligible": float(bool(repetition.get("repetition_token_zlib_eligible"))),
        "repetition_token_zlib_ratio": repetition.get("repetition_token_zlib_ratio"),
        "tail_ngram_repeat_fraction": repetition.get("tail_ngram_repeat_fraction"),
        "error": None,
    }
    for region_name, region in (record.get("gap_metrics") or {}).items():
        prefix = f"gap_{region_name}"
        light[f"{prefix}_valid_token_count"] = region.get("valid_token_count")
        light[f"{prefix}_sampled_forward_kl"] = region.get("sampled_forward_kl")
        light[f"{prefix}_abs_mean"] = region.get("abs_mean")
        light[f"{prefix}_high_gap_rate"] = region.get("high_gap_rate")
        light[f"{prefix}_student_sampled_prob_mean"] = region.get("student_sampled_prob_mean")
        light[f"{prefix}_teacher_sampled_prob_mean"] = region.get("teacher_sampled_prob_mean")
    return light


def load_lightweight_records(rollouts_path: Path) -> list[dict[str, Any]]:
    return [lightweight_record(record) for record in iter_jsonl_gz(rollouts_path)]


def repair_lightweight_record(record: dict[str, Any]) -> dict[str, Any] | None:
    repair = record.get("repair")
    if not isinstance(repair, dict):
        return None
    if repair.get("error"):
        return {
            "benchmark": record.get("benchmark"),
            "sample_id": record.get("sample_id"),
            "rollout_index": record.get("rollout_index"),
            "error": repair.get("error"),
        }
    repetition = repair.get("repetition") or {}
    return {
        "benchmark": record.get("benchmark"),
        "benchmark_path": record.get("benchmark_path"),
        "question_index": record.get("question_index"),
        "sample_id": str(record.get("sample_id")),
        "problem_fingerprint": record.get("problem_fingerprint"),
        "rollout_index": record.get("rollout_index"),
        "acc": float(bool(repair.get("acc"))),
        "eos_success": float(bool(repair.get("eos_success"))),
        "answer_extractable": float(bool(repair.get("answer_extractable"))),
        "has_answer_pattern": float(bool(repair.get("has_answer_pattern"))),
        "accepted_answer_format": float(bool(repair.get("accepted_answer_format"))),
        "has_boxed_pattern": float(bool(repair.get("has_boxed_pattern"))),
        "truncated": float(bool(repair.get("truncated"))),
        "response_tokens": int(repair.get("response_tokens", 0)),
        "repetition_text_zlib": float(bool(repetition.get("repetition_text_zlib"))),
        "repetition_text_zlib_eligible": float(bool(repetition.get("repetition_text_zlib_eligible"))),
        "repetition_text_zlib_ratio": repetition.get("repetition_text_zlib_ratio"),
        "repetition_token_zlib": float(bool(repetition.get("repetition_token_zlib"))),
        "repetition_token_zlib_eligible": float(bool(repetition.get("repetition_token_zlib_eligible"))),
        "repetition_token_zlib_ratio": repetition.get("repetition_token_zlib_ratio"),
        "tail_ngram_repeat_fraction": repetition.get("tail_ngram_repeat_fraction"),
        "repair_executed": float(bool(repair.get("repair_executed"))),
        "intervention_count": int(repair.get("intervention_count", 0) or 0),
        "first_intervention_position": repair.get("first_intervention_position"),
        "error": None,
    }


def _accuracy_at_n(records: list[dict[str, Any]], *, repair_branch: bool) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        branch = record.get("repair") if repair_branch else record
        if not isinstance(branch, dict) or branch.get("error"):
            continue
        grouped[str(record.get("sample_id"))].append(branch)
    if not grouped:
        return {"mean": None, "maj": None, "best": None, "question_count": 0}

    all_correct: list[float] = []
    majority_correct: list[float] = []
    best_correct: list[float] = []
    for group in grouped.values():
        all_correct.extend(float(bool(item.get("acc"))) for item in group)
        best_correct.append(float(any(bool(item.get("acc")) for item in group)))
        by_prediction: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in group:
            by_prediction[str(item.get("pred", ""))].append(item)
        # Stable tie-breaking avoids depending on request completion order.
        majority_prediction = min(by_prediction, key=lambda pred: (-len(by_prediction[pred]), pred))
        majority_group = by_prediction[majority_prediction]
        majority_correct.append(float(any(bool(item.get("acc")) for item in majority_group)))
    return {
        "mean": statistics.fmean(all_correct),
        "maj": statistics.fmean(majority_correct),
        "best": statistics.fmean(best_correct),
        "question_count": len(grouped),
    }


def _paired_repair_metrics(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    paired = [
        record for record in records if isinstance(record.get("repair"), dict) and not record["repair"].get("error")
    ]
    if not paired:
        return None
    deltas = [float(bool(record["repair"].get("acc"))) - float(bool(record.get("acc"))) for record in paired]
    transitions = {
        "pre_correct_post_correct": 0,
        "pre_correct_post_wrong": 0,
        "pre_wrong_post_correct": 0,
        "pre_wrong_post_wrong": 0,
    }
    intervention_counts: list[int] = []
    first_positions: list[int] = []
    initial_gap_means: list[float] = []
    initial_gap_maxima: list[float] = []
    initial_gap_p95: list[float] = []
    selected_segment_scores: list[float] = []
    teacher_nonfinite_tokens = 0
    teacher_scored_tokens = 0
    trigger_count = 0
    repair_count = 0
    for record in paired:
        repair = record["repair"]
        raw_correct = bool(record.get("acc"))
        final_correct = bool(repair.get("acc"))
        transition = (
            "pre_correct_post_correct"
            if raw_correct and final_correct
            else "pre_correct_post_wrong"
            if raw_correct
            else "pre_wrong_post_correct"
            if final_correct
            else "pre_wrong_post_wrong"
        )
        transitions[transition] += 1
        intervention_count = int(repair.get("intervention_count", 0) or 0)
        intervention_counts.append(intervention_count)
        repair_count += int(intervention_count > 0)
        trigger_count += int(bool(repair.get("trigger_detected")))
        first_position = repair.get("first_intervention_position")
        if first_position is not None:
            first_positions.append(int(first_position))
        diagnostics = repair.get("gap_diagnostics") or []
        if diagnostics:
            initial = diagnostics[0]
            for destination, key in (
                (initial_gap_means, "mean"),
                (initial_gap_maxima, "max"),
                (initial_gap_p95, "p95"),
                (selected_segment_scores, "selected_segment_score"),
            ):
                value = finite_float(initial.get(key))
                if value is not None:
                    destination.append(value)
            teacher_nonfinite_tokens += int(initial.get("teacher_nonfinite_token_count", 0) or 0)
            teacher_scored_tokens += int(initial.get("valid_token_count", 0) or 0)
    count = len(paired)
    return {
        "rollout_count": count,
        "paired_delta": statistics.fmean(deltas),
        "repair_execution_rate": repair_count / count,
        "trigger_rate": trigger_count / count,
        "transitions": {name: value / count for name, value in transitions.items()},
        "intervention_count": distribution(intervention_counts),
        "first_intervention_position": distribution(first_positions),
        "initial_gap_mean": distribution(initial_gap_means),
        "initial_gap_max": distribution(initial_gap_maxima),
        "initial_gap_p95": distribution(initial_gap_p95),
        "selected_segment_score": distribution(selected_segment_scores),
        "teacher_nonfinite_token_fraction": (
            teacher_nonfinite_tokens / teacher_scored_tokens if teacher_scored_tokens else None
        ),
    }


def aggregate_gap_region(records: list[dict[str, Any]], region_name: str) -> dict[str, Any] | None:
    count_key = f"gap_{region_name}_valid_token_count"
    valid_records = [record for record in records if finite_float(record.get(count_key)) not in {None, 0.0}]
    if not valid_records:
        return None
    total_count = sum(float(record[count_key]) for record in valid_records)

    def weighted(metric_suffix: str) -> float | None:
        numerator = 0.0
        denominator = 0.0
        metric_key = f"gap_{region_name}_{metric_suffix}"
        for record in valid_records:
            count = finite_float(record.get(count_key))
            value = finite_float(record.get(metric_key))
            if count is not None and count > 0 and value is not None:
                numerator += count * value
                denominator += count
        return numerator / denominator if denominator else None

    return {
        "valid_token_count": int(total_count),
        "rollout_count": len(valid_records),
        "sampled_forward_kl_token_weighted": weighted("sampled_forward_kl"),
        "sampled_forward_kl_rollout_mean": mean_present(
            record.get(f"gap_{region_name}_sampled_forward_kl") for record in valid_records
        ),
        "abs_mean_token_weighted": weighted("abs_mean"),
        "high_gap_rate_token_weighted": weighted("high_gap_rate"),
        "student_sampled_prob_mean_token_weighted": weighted("student_sampled_prob_mean"),
        "teacher_sampled_prob_mean_token_weighted": weighted("teacher_sampled_prob_mean"),
    }


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [record for record in records if not record.get("error")]
    question_keys = {str(record.get("sample_id")) for record in successful}
    result: dict[str, Any] = {
        "rollout_count": len(successful),
        "failed_rollout_count": len(records) - len(successful),
        "question_count": len(question_keys),
        "acc": mean_present(record.get("acc") for record in successful),
        "eos_success_rate": mean_present(record.get("eos_success") for record in successful),
        "answer_extractable_rate": mean_present(record.get("answer_extractable") for record in successful),
        "answer_pattern_rate": mean_present(record.get("has_answer_pattern") for record in successful),
        "accepted_answer_format_rate": mean_present(record.get("accepted_answer_format") for record in successful),
        "boxed_answer_rate": mean_present(record.get("has_boxed_pattern") for record in successful),
        "truncation_rate": mean_present(record.get("truncated") for record in successful),
        "repetition_text_zlib_rate": mean_present(record.get("repetition_text_zlib") for record in successful),
        "repetition_text_zlib_eligible_rate": mean_present(
            record.get("repetition_text_zlib_eligible") for record in successful
        ),
        "repetition_token_zlib_rate": mean_present(record.get("repetition_token_zlib") for record in successful),
        "repetition_token_zlib_eligible_rate": mean_present(
            record.get("repetition_token_zlib_eligible") for record in successful
        ),
        "tail_ngram_repeat_fraction_mean": mean_present(
            record.get("tail_ngram_repeat_fraction") for record in successful
        ),
        "response_tokens": distribution(record.get("response_tokens") for record in successful),
        "repetition_text_zlib_ratio": distribution(record.get("repetition_text_zlib_ratio") for record in successful),
        "repetition_token_zlib_ratio": distribution(record.get("repetition_token_zlib_ratio") for record in successful),
    }
    gap_regions: dict[str, Any] = {}
    for region_name in GAP_REGION_NAMES:
        region = aggregate_gap_region(successful, region_name)
        if region is not None:
            gap_regions[region_name] = region
    result["gap_regions"] = gap_regions
    return result


def question_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if not record.get("error"):
            grouped[str(record["sample_id"])].append(record)
    scalar_fields = [
        "acc",
        "eos_success",
        "answer_extractable",
        "has_answer_pattern",
        "accepted_answer_format",
        "has_boxed_pattern",
        "truncated",
        "response_tokens",
        "repetition_text_zlib",
        "repetition_text_zlib_eligible",
        "repetition_token_zlib",
        "repetition_token_zlib_eligible",
        "tail_ngram_repeat_fraction",
        "gap_all_sampled_forward_kl",
        "gap_all_abs_mean",
        "gap_last_window_sampled_forward_kl",
        "gap_last_window_abs_mean",
        "gap_tail_12000_plus_sampled_forward_kl",
        "gap_tail_12000_plus_abs_mean",
        "gap_deep_tail_16000_plus_sampled_forward_kl",
        "gap_deep_tail_16000_plus_abs_mean",
        "gap_extreme_tail_20000_plus_sampled_forward_kl",
        "gap_extreme_tail_20000_plus_abs_mean",
    ]
    rows: list[dict[str, Any]] = []
    for sample_id, group in sorted(grouped.items()):
        problem_fingerprints = {
            str(record.get("problem_fingerprint")) for record in group if record.get("problem_fingerprint")
        }
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "rollout_count": len(group),
            "problem_fingerprint": next(iter(problem_fingerprints)) if len(problem_fingerprints) == 1 else None,
        }
        for field in scalar_fields:
            row[field] = mean_present(record.get(field) for record in group)
        rows.append(row)
    return rows


def bootstrap_mean_ci(values: list[float], resamples: int, seed: int) -> dict[str, Any] | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None
    estimate = statistics.fmean(finite)
    if resamples <= 0 or len(finite) == 1:
        return {
            "estimate": estimate,
            "ci95_low": estimate,
            "ci95_high": estimate,
            "question_count": len(finite),
            "resamples": 0,
        }
    rng = random.Random(seed)
    size = len(finite)
    bootstrap_values = [statistics.fmean(finite[rng.randrange(size)] for _ in range(size)) for _ in range(resamples)]
    return {
        "estimate": estimate,
        "ci95_low": percentile(bootstrap_values, 0.025),
        "ci95_high": percentile(bootstrap_values, 0.975),
        "question_count": size,
        "resamples": resamples,
    }


def bootstrap_metrics(
    per_problem: list[dict[str, Any]], metrics: list[str], resamples: int, bootstrap_seed: int, identity: str
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in metrics:
        values = [value for row in per_problem if (value := finite_float(row.get(metric))) is not None]
        ci = bootstrap_mean_ci(values, resamples=resamples, seed=stable_seed(bootstrap_seed, identity, metric))
        if ci is not None:
            result[metric] = ci
    return result


def normalize_length_edges(edges: list[int]) -> list[int]:
    normalized = sorted(set(int(edge) for edge in edges))
    if not normalized or normalized[0] != 0:
        normalized.insert(0, 0)
    if any(edge < 0 for edge in normalized):
        raise ValueError("Length-bin edges must be non-negative")
    if len(normalized) < 2:
        raise ValueError("At least two distinct length-bin edges are required")
    return normalized


def length_bin_label(length: int, edges: list[int]) -> str:
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        if lower <= length < upper:
            return f"{lower:05d}_{upper:05d}"
    return f"{edges[-1]:05d}_plus"


def build_length_bin_rows(records: list[dict[str, Any]], edges: list[int]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if not record.get("error"):
            grouped[length_bin_label(int(record.get("response_tokens", 0)), edges)].append(record)
    rows: list[dict[str, Any]] = []
    for label, group in sorted(grouped.items()):
        summary = aggregate_records(group)
        tail_gap = (summary.get("gap_regions") or {}).get("last_window") or {}
        rows.append(
            {
                "length_bin": label,
                "rollout_count": summary["rollout_count"],
                "question_count": summary["question_count"],
                "acc": summary["acc"],
                "eos_success_rate": summary["eos_success_rate"],
                "answer_extractable_rate": summary["answer_extractable_rate"],
                "answer_pattern_rate": summary["answer_pattern_rate"],
                "accepted_answer_format_rate": summary["accepted_answer_format_rate"],
                "truncation_rate": summary["truncation_rate"],
                "repetition_text_zlib_rate": summary["repetition_text_zlib_rate"],
                "repetition_token_zlib_rate": summary["repetition_token_zlib_rate"],
                "tail_ngram_repeat_fraction_mean": summary["tail_ngram_repeat_fraction_mean"],
                "response_tokens_mean": summary["response_tokens"]["mean"],
                "last_window_sampled_forward_kl_token_weighted": tail_gap.get("sampled_forward_kl_token_weighted"),
                "last_window_abs_gap_token_weighted": tail_gap.get("abs_mean_token_weighted"),
                "last_window_high_gap_rate": tail_gap.get("high_gap_rate_token_weighted"),
            }
        )
    return rows


def write_summary_artifacts(output_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    full_records = list(iter_jsonl_gz(output_dir / ROLLOUTS_FILENAME))
    records = [lightweight_record(record) for record in full_records]
    failures = [record for record in records if record.get("error")]
    successful = [record for record in records if not record.get("error")]
    per_problem = question_rows(successful)
    identity = str(metadata["benchmark"]["id"])
    bootstrap = bootstrap_metrics(
        per_problem,
        metrics=DEFAULT_BOOTSTRAP_METRICS,
        resamples=int(metadata["analysis"]["bootstrap_resamples"]),
        bootstrap_seed=int(metadata["analysis"]["bootstrap_seed"]),
        identity=identity,
    )
    edges = normalize_length_edges([int(value) for value in metadata["analysis"]["length_bin_edges"]])
    length_rows = build_length_bin_rows(successful, edges)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at_unix": time.time(),
        "experiment": metadata.get("experiment"),
        "global_step": metadata.get("global_step"),
        "benchmark": metadata.get("benchmark"),
        "teacher_scored": bool(metadata.get("teacher_scored")),
        "teacher_served_model_name": metadata.get("teacher_served_model_name"),
        "rollout_n": metadata.get("request", {}).get("rollout_n"),
        "rollout_count": len(records),
        "failed_rollout_count": len(failures),
        "metrics": aggregate_records(records),
        "accuracy_at_n": _accuracy_at_n(full_records, repair_branch=False),
        "question_cluster_bootstrap_ci95": bootstrap,
        "metric_definitions": {
            "eos_success": "finish_reason is stop/eos; this is reported separately from length truncation.",
            "answer_extractable": "math_dapo produced a non-invalid prediction, irrespective of correctness.",
            "repetition_text_zlib": "Configured text tail exceeds the configured raw/compressed zlib threshold.",
            "repetition_token_zlib": "Configured token tail exceeds the configured int64-byte zlib threshold.",
            "sampled_forward_kl": (
                "Mean log p_student(y_t|prefix)-log p_teacher(y_t|prefix) on realized response tokens. "
                "With temperature/top-p sampling this is a signed sampled log-prob gap, not exact vocabulary KL."
            ),
            "bootstrap": "Confidence intervals resample questions, preserving dependence among rollouts per question.",
        },
    }
    repair_records = [item for record in full_records if (item := repair_lightweight_record(record)) is not None]
    if repair_records:
        repair_successful = [record for record in repair_records if not record.get("error")]
        repair_per_problem = question_rows(repair_successful)
        repair_records_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in repair_successful:
            repair_records_by_sample[str(record["sample_id"])].append(record)
        for row in repair_per_problem:
            group = repair_records_by_sample[str(row["sample_id"])]
            row["repair_executed"] = mean_present(record.get("repair_executed") for record in group)
            row["intervention_count"] = mean_present(record.get("intervention_count") for record in group)
            row["first_intervention_position"] = mean_present(
                record.get("first_intervention_position") for record in group
            )
        repair_by_sample = {str(row["sample_id"]): row for row in repair_per_problem}
        for row in per_problem:
            repair_row = repair_by_sample.get(str(row["sample_id"]))
            if repair_row is None:
                continue
            for key, value in repair_row.items():
                if key not in {"sample_id", "problem_fingerprint"}:
                    target = key if key == "repair_executed" else f"repair_{key}"
                    row[target] = value
            raw_acc = finite_float(row.get("acc"))
            repaired_acc = finite_float(repair_row.get("acc"))
            row["repair_acc_delta"] = (
                repaired_acc - raw_acc if raw_acc is not None and repaired_acc is not None else None
            )
        summary["repair"] = {
            "complete": bool(metadata.get("repair_complete")),
            "rollout_count": len(repair_records),
            "failed_rollout_count": len(repair_records) - len(repair_successful),
            "metrics": aggregate_records(repair_records),
            "accuracy_at_n": _accuracy_at_n(full_records, repair_branch=True),
            "paired": _paired_repair_metrics(full_records),
            "question_cluster_bootstrap_ci95": bootstrap_metrics(
                repair_per_problem,
                metrics=DEFAULT_BOOTSTRAP_METRICS,
                resamples=int(metadata["analysis"]["bootstrap_resamples"]),
                bootstrap_seed=int(metadata["analysis"]["bootstrap_seed"]),
                identity=f"{identity}:repair",
            ),
        }
    write_json_atomic(output_dir / SUMMARY_FILENAME, summary)
    per_problem_fields = sorted({key for row in per_problem for key in row})
    preferred = ["sample_id", "rollout_count", "problem_fingerprint"]
    per_problem_fields = preferred + [field for field in per_problem_fields if field not in preferred]
    write_csv_atomic(output_dir / PER_PROBLEM_FILENAME, per_problem, per_problem_fields)
    length_fields = [
        "length_bin",
        "rollout_count",
        "question_count",
        "acc",
        "eos_success_rate",
        "answer_extractable_rate",
        "answer_pattern_rate",
        "accepted_answer_format_rate",
        "truncation_rate",
        "repetition_text_zlib_rate",
        "repetition_token_zlib_rate",
        "tail_ngram_repeat_fraction_mean",
        "response_tokens_mean",
        "last_window_sampled_forward_kl_token_weighted",
        "last_window_abs_gap_token_weighted",
        "last_window_high_gap_rate",
    ]
    write_csv_atomic(output_dir / LENGTH_BINS_FILENAME, length_rows, length_fields)
    print_unit_metrics(summary)
    return summary


def print_unit_metrics(summary: dict[str, Any]) -> None:
    metrics = summary.get("metrics") or {}
    response_tokens = metrics.get("response_tokens") or {}

    def fmt(value: Any, *, percentage: bool = False) -> str:
        parsed = finite_float(value)
        if parsed is None:
            return "N/A"
        return f"{parsed * 100:.2f}%" if percentage else f"{parsed:.1f}"

    benchmark = summary.get("benchmark") or {}
    print(
        "benchmark_summary "
        f"benchmark={benchmark.get('path')} step={summary.get('global_step')} "
        f"acc={fmt(metrics.get('acc'), percentage=True)} "
        f"eos={fmt(metrics.get('eos_success_rate'), percentage=True)} "
        f"trunc={fmt(metrics.get('truncation_rate'), percentage=True)} "
        f"length_mean={fmt(response_tokens.get('mean'))}",
        flush=True,
    )


def _completion_file_entry(path: Path) -> dict[str, Any]:
    return {"size": path.stat().st_size, "sha256": sha256_file(path)}


def write_completion_marker(output_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    files = {name: _completion_file_entry(output_dir / name) for name in REQUIRED_RESULT_FILES}
    marker = {
        "schema_version": SCHEMA_VERSION,
        "completed_at_unix": time.time(),
        "experiment": metadata["experiment"],
        "global_step": metadata["global_step"],
        "benchmark": metadata["benchmark"],
        "generation_fingerprint": metadata["generation_fingerprint"],
        "rollout_count": metadata["rollout_count"],
        "files": files,
    }
    marker["result_fingerprint"] = payload_fingerprint(files)
    write_json_atomic(output_dir / COMPLETION_FILENAME, marker)
    return marker


def parse_expectations(values: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected KEY=VALUE for --expect, got {value!r}")
        key, raw = value.split("=", 1)
        if not key:
            raise ValueError(f"Expected a non-empty key in --expect {value!r}")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = raw
        result[key] = parsed
    return result


def _comparable_expectation_value(key: str, value: Any) -> Any:
    if key in REQUEST_PATH_KEYS and isinstance(value, str):
        return str(Path(value).expanduser().resolve())
    if key == "benchmark" and isinstance(value, str):
        return normalize_benchmark_path(value)
    return value


def validate_completed_unit(
    output_dir: Path,
    *,
    expectations: dict[str, Any] | None = None,
    validate_inputs: bool = True,
    verify_checksums: bool = False,
    allow_legacy: bool = False,
) -> tuple[bool, str]:
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
    if expectations:
        request = metadata.get("request") or {}
        analysis = metadata.get("analysis") or {}
        is_legacy = bool(metadata.get("legacy_migrated"))
        mismatches: dict[str, dict[str, Any]] = {}
        for key, expected in expectations.items():
            if key.startswith("analysis."):
                analysis_key = key.removeprefix("analysis.")
                if allow_legacy and is_legacy and analysis_key not in analysis:
                    continue
                actual = analysis.get(analysis_key)
            else:
                if allow_legacy and is_legacy and key not in request:
                    continue
                actual = request.get(key)
            normalized_expected = _comparable_expectation_value(key, expected)
            normalized_actual = _comparable_expectation_value(key, actual)
            if normalized_actual != normalized_expected:
                mismatches[key] = {"expected": normalized_expected, "actual": normalized_actual}
        if mismatches:
            return False, f"requested evaluation configuration changed: {mismatches}"
    if validate_inputs and not (allow_legacy and metadata.get("legacy_migrated")):
        valid, reason = validate_input_manifest(metadata.get("input_manifest") or {})
        if not valid:
            return False, reason
    file_entries = marker.get("files") or {}
    for name in REQUIRED_RESULT_FILES:
        path = output_dir / name
        entry = file_entries.get(name)
        if not path.is_file() or not isinstance(entry, dict):
            return False, f"required result file is missing: {name}"
        if path.stat().st_size != entry.get("size"):
            return False, f"result file size changed: {name}"
        if verify_checksums and sha256_file(path) != entry.get("sha256"):
            return False, f"result file checksum changed: {name}"
    return True, "complete"


def _part_path(parts_dir: Path, task: RolloutTask) -> Path:
    return parts_dir / f"{task.flat_index:08d}-{task.task_id}.json.gz"


def _load_completed_part(path: Path, task: RolloutTask, fingerprint: str) -> dict[str, Any] | None:
    record = read_single_json_gz(path) if path.is_file() else None
    if not record or record.get("error"):
        return None
    if record.get("task_id") != task.task_id or record.get("generation_fingerprint") != fingerprint:
        return None
    return record


def _clean_unit_for_overwrite(output_dir: Path) -> None:
    for name in (*REQUIRED_RESULT_FILES, COMPLETION_FILENAME):
        path = output_dir / name
        if path.is_file():
            path.unlink()
    work_dir = output_dir / WORK_DIRNAME
    if work_dir.is_dir():
        shutil.rmtree(work_dir)


def _write_layout_metadata(result_root: Path, args: argparse.Namespace) -> None:
    experiment_path = result_root / "experiment.json"
    experiment_payload = {"schema_version": SCHEMA_VERSION, "experiment": args.experiment}
    if experiment_path.is_file():
        existing = json.loads(experiment_path.read_text(encoding="utf-8"))
        if existing.get("experiment") != args.experiment:
            raise RuntimeError(
                f"Result root {result_root} already belongs to experiment {existing.get('experiment')!r}, "
                f"not {args.experiment!r}"
            )
    else:
        write_json_atomic(experiment_path, experiment_payload)

    normalized = normalize_benchmark_path(args.benchmark)
    root = benchmark_root(result_root, normalized)
    root.mkdir(parents=True, exist_ok=True)
    benchmark_path = root / "benchmark.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "id": benchmark_id(normalized),
        "name": Path(normalized).name,
        "path": normalized,
    }
    if benchmark_path.is_file():
        existing = json.loads(benchmark_path.read_text(encoding="utf-8"))
        if existing.get("path") != normalized:
            raise RuntimeError(f"Benchmark ID collision under {root}: {existing.get('path')!r} vs {normalized!r}")
    else:
        write_json_atomic(benchmark_path, payload)


def validate_generate_args(args: argparse.Namespace) -> None:
    positive_names = [
        "rollout_n",
        "max_tokens",
        "max_prompt_tokens",
        "dp_size",
        "max_concurrency_per_dp",
        "score_max_in_flight",
        "repetition_tail_chars",
        "repetition_tail_token_window",
        "ngram_size",
    ]
    for name in positive_names:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.max_pending is not None and args.max_pending <= 0:
        raise ValueError("--max-pending must be positive when specified")
    if args.max_samples_per_benchmark == 0 or args.max_samples_per_benchmark < -1:
        raise ValueError("--max-samples-per-benchmark must be -1 or positive")
    if args.max_retries < 0:
        raise ValueError("--max-retries cannot be negative")
    if args.print_every < 0:
        raise ValueError("--print-every cannot be negative")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        raise ValueError("--temperature must be finite and non-negative")
    if not math.isfinite(args.top_p) or not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1]")
    if not math.isfinite(args.repetition_threshold):
        raise ValueError("--repetition-threshold must be finite")
    if not math.isfinite(args.high_gap_threshold) or args.high_gap_threshold < 0:
        raise ValueError("--high-gap-threshold must be finite and non-negative")


def prepare_generation(args: argparse.Namespace) -> GenerationPreparation:
    tokenizer = load_eval_tokenizer(args)
    problems, tasks, dataset_fingerprint = prepare_tasks(args, tokenizer)
    input_manifest = capture_input_manifest(
        dataset_root=args.dataset_root,
        raw_benchmark_path=args.benchmark,
        student_model_path=args.student_model_path,
        student_tokenizer_model=args.student_tokenizer_model,
        chat_template_model=args.chat_template_model,
        student_model_identity=getattr(args, "student_model_identity", None),
    )
    tokenizer_info = tokenizer_metadata(tokenizer, args.student_tokenizer_model)
    request = generation_request(args)
    analysis = analysis_request(args)
    fingerprint = generation_fingerprint(
        request=request,
        input_manifest=input_manifest,
        dataset_fingerprint=dataset_fingerprint,
        tokenizer_info=tokenizer_info,
    )
    return GenerationPreparation(
        tokenizer=tokenizer,
        problems=problems,
        tasks=tasks,
        dataset_fingerprint=dataset_fingerprint,
        input_manifest=input_manifest,
        tokenizer_info=tokenizer_info,
        request=request,
        analysis=analysis,
        fingerprint=fingerprint,
    )


def _metadata_for_generation(
    *,
    args: argparse.Namespace,
    problems: list[Problem],
    tasks: list[RolloutTask],
    dataset_fingerprint: str,
    tokenizer_info: dict[str, Any],
    input_manifest: dict[str, Any],
    fingerprint: str,
    generation_schema_version: int = GENERATION_SCHEMA_VERSION,
    generation_compatibility: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    prompt_lengths = [len(problem.prompt_token_ids) for problem in problems]
    normalized = normalize_benchmark_path(args.benchmark)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "generation_schema_version": int(generation_schema_version),
        "experiment": args.experiment,
        "global_step": args.global_step,
        "benchmark": {
            "id": benchmark_id(normalized),
            "name": Path(normalized).name,
            "path": normalized,
            "dataset_fingerprint": dataset_fingerprint,
        },
        "served_model_name": args.served_model_name,
        "teacher_scored": False,
        "teacher_served_model_name": None,
        "generation_complete": False,
        "request": generation_request(args),
        "analysis": analysis_request(args),
        "generation_fingerprint": fingerprint,
        "input_manifest": input_manifest,
        "student_tokenizer": tokenizer_info,
        "problem_count": len(problems),
        "rollout_count": len(tasks),
        "prompt_tokens_min": min(prompt_lengths),
        "prompt_tokens_max": max(prompt_lengths),
        "source": source_metadata(),
    }
    if generation_compatibility:
        metadata["generation_compatibility"] = dict(generation_compatibility)
    return metadata


def _read_metadata(output_dir: Path) -> dict[str, Any] | None:
    path = output_dir / METADATA_FILENAME
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _mapping_differences(existing: Mapping[str, Any], current: Mapping[str, Any]) -> dict[str, Any]:
    differences: dict[str, Any] = {}
    for key in sorted(set(existing) | set(current)):
        if existing.get(key) != current.get(key):
            differences[str(key)] = {"existing": existing.get(key), "current": current.get(key)}
    return differences


def _operational_descriptor(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in entry.items() if key != "logical_path"}


def _descriptor_map(entries: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(entry.get("path") or entry.get("logical_path")): _operational_descriptor(entry) for entry in entries}


def _manifest_group_difference(
    existing_entries: Iterable[Mapping[str, Any]],
    current_entries: Iterable[Mapping[str, Any]],
) -> dict[str, Any] | None:
    existing = _descriptor_map(existing_entries)
    current = _descriptor_map(current_entries)
    added = sorted(set(current) - set(existing))
    removed = sorted(set(existing) - set(current))
    changed = {
        path: _mapping_differences(existing[path], current[path])
        for path in sorted(set(existing) & set(current))
        if existing[path] != current[path]
    }
    if not added and not removed and not changed:
        return None
    return {"added": added, "removed": removed, "changed": changed}


def generation_compatibility_differences(
    existing: Mapping[str, Any],
    preparation: GenerationPreparation,
) -> dict[str, Any]:
    existing_manifest = existing.get("input_manifest") or {}
    current_manifest = preparation.input_manifest
    manifest_groups: dict[str, Any] = {}
    for group in (
        "dataset",
        "student_model_metadata",
        "student_model_weights",
        "student_tokenizer",
        "chat_template",
    ):
        difference = _manifest_group_difference(
            existing_manifest.get(group, []),
            current_manifest.get(group, []),
        )
        if difference is not None:
            manifest_groups[group] = difference
    differences: dict[str, Any] = {
        "generation_fingerprint": {
            "existing": existing.get("generation_fingerprint"),
            "current": preparation.fingerprint,
        },
        "generation_schema_version": {
            "existing": existing.get("generation_schema_version"),
            "current": GENERATION_SCHEMA_VERSION,
        },
    }
    request_difference = _mapping_differences(existing.get("request") or {}, preparation.request)
    if request_difference:
        differences["request"] = request_difference
    existing_dataset_fingerprint = (existing.get("benchmark") or {}).get("dataset_fingerprint")
    if existing_dataset_fingerprint != preparation.dataset_fingerprint:
        differences["dataset_fingerprint"] = {
            "existing": existing_dataset_fingerprint,
            "current": preparation.dataset_fingerprint,
        }
    tokenizer_difference = _mapping_differences(
        existing.get("student_tokenizer") or {},
        preparation.tokenizer_info,
    )
    if tokenizer_difference:
        differences["student_tokenizer"] = tokenizer_difference
    existing_model_identity = existing_manifest.get("student_model_identity")
    current_model_identity = current_manifest.get("student_model_identity")
    if existing_model_identity != current_model_identity:
        differences["student_model_identity"] = {
            "existing": existing_model_identity,
            "current": current_model_identity,
        }
    if manifest_groups:
        differences["input_manifest_groups"] = manifest_groups
    return differences


def _legacy_generation_fingerprint(existing: Mapping[str, Any]) -> str | None:
    try:
        return payload_fingerprint(
            {
                "generation_schema_version": int(existing["generation_schema_version"]),
                "request": existing["request"],
                "input_manifest_fingerprint": existing["input_manifest"]["fingerprint"],
                "dataset_fingerprint": existing["benchmark"]["dataset_fingerprint"],
                "tokenizer": existing["student_tokenizer"],
            }
        )
    except (KeyError, TypeError, ValueError):
        return None


def _legacy_manifest_is_self_consistent(manifest: Mapping[str, Any]) -> bool:
    if manifest.get("schema_version") == INPUT_MANIFEST_SCHEMA_VERSION:
        return False
    payload = {str(key): value for key, value in manifest.items() if key != "fingerprint"}
    return manifest.get("fingerprint") == payload_fingerprint(payload)


def _legacy_mtime_only_compatibility(
    existing: Mapping[str, Any],
    preparation: GenerationPreparation,
) -> tuple[bool, list[dict[str, Any]]]:
    if existing.get("generation_schema_version") != 2:
        return False, []
    model_identity = preparation.input_manifest.get("student_model_identity") or {}
    if model_identity.get("kind") != "committed_checkpoint" or not model_identity.get("fingerprint"):
        return False, []
    if existing.get("generation_fingerprint") != _legacy_generation_fingerprint(existing):
        return False, []
    existing_manifest = existing.get("input_manifest") or {}
    if not _legacy_manifest_is_self_consistent(existing_manifest):
        return False, []
    if existing.get("request") != preparation.request:
        return False, []
    if (existing.get("benchmark") or {}).get("dataset_fingerprint") != preparation.dataset_fingerprint:
        return False, []
    if existing.get("student_tokenizer") != preparation.tokenizer_info:
        return False, []

    current_manifest = preparation.input_manifest
    for group in ("dataset", "student_model_metadata", "student_tokenizer", "chat_template"):
        if _descriptor_map(existing_manifest.get(group, [])) != _descriptor_map(current_manifest.get(group, [])):
            return False, []

    existing_weights = _descriptor_map(existing_manifest.get("student_model_weights", []))
    current_weights = _descriptor_map(current_manifest.get("student_model_weights", []))
    if set(existing_weights) != set(current_weights) or not existing_weights:
        return False, []
    changes: list[dict[str, Any]] = []
    for path in sorted(existing_weights):
        previous = existing_weights[path]
        current = current_weights[path]
        previous_without_mtime = {key: value for key, value in previous.items() if key != "mtime_ns"}
        current_without_mtime = {key: value for key, value in current.items() if key != "mtime_ns"}
        if previous_without_mtime != current_without_mtime:
            return False, []
        if previous.get("mtime_ns") != current.get("mtime_ns"):
            changes.append(
                {
                    "path": path,
                    "size": current.get("size"),
                    "existing_mtime_ns": previous.get("mtime_ns"),
                    "current_mtime_ns": current.get("mtime_ns"),
                }
            )
    return True, changes


def assess_generation_compatibility(
    existing: Mapping[str, Any] | None,
    preparation: GenerationPreparation,
) -> dict[str, Any]:
    if existing is None:
        return {
            "status": "new",
            "compatible": True,
            "effective_fingerprint": preparation.fingerprint,
            "differences": {},
        }
    existing_fingerprint = str(existing.get("generation_fingerprint") or "")
    if existing_fingerprint == preparation.fingerprint:
        return {
            "status": "exact",
            "compatible": True,
            "effective_fingerprint": preparation.fingerprint,
            "differences": {},
        }
    compatibility = existing.get("generation_compatibility") or {}
    compatible_fingerprints = compatibility.get("compatible_fingerprints") or []
    if preparation.fingerprint in compatible_fingerprints:
        return {
            "status": "compatible_alias",
            "compatible": True,
            "effective_fingerprint": existing_fingerprint,
            "differences": {},
        }
    legacy_compatible, mtime_changes = _legacy_mtime_only_compatibility(existing, preparation)
    if legacy_compatible:
        return {
            "status": "legacy_mtime_only",
            "compatible": True,
            "effective_fingerprint": existing_fingerprint,
            "differences": generation_compatibility_differences(existing, preparation),
            "mtime_changes": mtime_changes,
        }
    return {
        "status": "incompatible",
        "compatible": False,
        "effective_fingerprint": None,
        "differences": generation_compatibility_differences(existing, preparation),
    }


def _apply_generation_compatibility_migration(
    output_dir: Path,
    existing: dict[str, Any],
    preparation: GenerationPreparation,
    assessment: Mapping[str, Any],
) -> dict[str, Any]:
    if assessment.get("status") != "legacy_mtime_only":
        return existing
    migrated = copy.deepcopy(existing)
    compatibility = dict(migrated.get("generation_compatibility") or {})
    aliases = [str(value) for value in compatibility.get("compatible_fingerprints") or []]
    if preparation.fingerprint not in aliases:
        aliases.append(preparation.fingerprint)
    migrations = list(compatibility.get("migrations") or [])
    migrations.append(
        {
            "migrated_at_unix": time.time(),
            "reason": "legacy model manifest upgrade; merged-weight differences were limited to mtime",
            "legacy_generation_fingerprint": migrated.get("generation_fingerprint"),
            "compatible_generation_fingerprint": preparation.fingerprint,
            "legacy_input_manifest_fingerprint": (migrated.get("input_manifest") or {}).get("fingerprint"),
            "stable_input_manifest_fingerprint": preparation.input_manifest.get("fingerprint"),
            "student_model_identity": preparation.input_manifest.get("student_model_identity"),
            "mtime_changes": list(assessment.get("mtime_changes") or []),
        }
    )
    compatibility.update(
        {
            "schema_version": GENERATION_COMPATIBILITY_SCHEMA_VERSION,
            "compatible_fingerprints": aliases,
            "migrations": migrations,
        }
    )
    migrated["generation_compatibility"] = compatibility
    migrated["input_manifest"] = preparation.input_manifest
    write_json_atomic(output_dir / METADATA_FILENAME, migrated)
    if migrated.get("generation_complete") and all((output_dir / name).is_file() for name in REQUIRED_RESULT_FILES):
        write_completion_marker(output_dir, migrated)
    return migrated


def _reanalyze_completed_unit(output_dir: Path, metadata: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    metadata["analysis"] = analysis
    metadata["generation_complete"] = True
    write_json_atomic(output_dir / METADATA_FILENAME, metadata)
    write_summary_artifacts(output_dir, metadata)
    marker = write_completion_marker(output_dir, metadata)
    print(f"[analysis] refreshed derived artifacts: {output_dir} fingerprint={marker['result_fingerprint']}")
    return metadata


async def generate_main(args: argparse.Namespace) -> int:
    validate_generate_args(args)
    result_root = Path(args.result_root).expanduser().resolve()
    output_dir = unit_output_dir(result_root, args.benchmark, args.global_step)
    with unit_lock(output_dir):
        _write_layout_metadata(result_root, args)
        preparation = prepare_generation(args)
        existing = _read_metadata(output_dir)

        if args.overwrite:
            assessment = {
                "status": "overwrite",
                "compatible": True,
                "effective_fingerprint": preparation.fingerprint,
                "differences": generation_compatibility_differences(existing, preparation) if existing else {},
            }
        else:
            assessment = assess_generation_compatibility(existing, preparation)
        if not assessment["compatible"]:
            raise RuntimeError(
                "Generation preflight rejected an existing evaluation. No model server is required to diagnose this "
                "conflict:\n"
                + json.dumps(
                    {
                        "event": "generation_preflight",
                        "status": assessment["status"],
                        "output_dir": str(output_dir),
                        "differences": assessment["differences"],
                    },
                    ensure_ascii=False,
                    indent=2,
                    default=json_default,
                )
            )
        if existing is not None and assessment["status"] == "legacy_mtime_only":
            existing = _apply_generation_compatibility_migration(output_dir, existing, preparation, assessment)
        print(
            canonical_json(
                {
                    "event": "generation_preflight",
                    "status": assessment["status"],
                    "output_dir": output_dir,
                    "existing_generation_fingerprint": None
                    if existing is None
                    else existing.get("generation_fingerprint"),
                    "current_generation_fingerprint": preparation.fingerprint,
                    "effective_generation_fingerprint": assessment["effective_fingerprint"],
                    "mtime_change_count": len(assessment.get("mtime_changes") or []),
                }
            ),
            flush=True,
        )
        if getattr(args, "preflight_only", False):
            return 0

        if args.overwrite:
            _clean_unit_for_overwrite(output_dir)
            existing = None
        fingerprint = str(assessment["effective_fingerprint"])
        if existing and existing.get("generation_complete") and (output_dir / ROLLOUTS_FILENAME).is_file():
            if existing.get("analysis") != preparation.analysis or not (output_dir / COMPLETION_FILENAME).is_file():
                _reanalyze_completed_unit(output_dir, existing, preparation.analysis)
                return 0
            else:
                valid, reason = validate_completed_unit(output_dir, validate_inputs=True)
                if valid:
                    print(f"[generate] already complete: {output_dir}")
                    return 0
                if "result file" in reason or "completion marker" in reason:
                    _reanalyze_completed_unit(output_dir, existing, preparation.analysis)
                    return 0
                raise RuntimeError(f"Existing evaluation is no longer compatible: {reason}")

        metadata = _metadata_for_generation(
            args=args,
            problems=preparation.problems,
            tasks=preparation.tasks,
            dataset_fingerprint=preparation.dataset_fingerprint,
            tokenizer_info=preparation.tokenizer_info,
            input_manifest=preparation.input_manifest,
            fingerprint=fingerprint,
            generation_schema_version=(
                int(existing.get("generation_schema_version", GENERATION_SCHEMA_VERSION))
                if existing is not None
                else GENERATION_SCHEMA_VERSION
            ),
            generation_compatibility=(existing or {}).get("generation_compatibility"),
        )
        work_dir = output_dir / WORK_DIRNAME
        parts_dir = work_dir / "generation"
        failures_dir = work_dir / "generation_failures"
        parts_dir.mkdir(parents=True, exist_ok=True)
        failures_dir.mkdir(parents=True, exist_ok=True)

        completed_records: dict[str, dict[str, Any]] = {}
        pending_tasks: list[RolloutTask] = []
        for task in preparation.tasks:
            record = _load_completed_part(_part_path(parts_dir, task), task, fingerprint)
            if record is None:
                pending_tasks.append(task)
            else:
                completed_records[task.task_id] = record
        metadata["generation_resumed_rollout_count"] = len(completed_records)
        metadata["generation_pending_rollout_count"] = len(pending_tasks)
        write_json_atomic(output_dir / METADATA_FILENAME, metadata)
        print(
            canonical_json(
                {
                    "event": "generation_start",
                    "output_dir": output_dir,
                    "benchmark": metadata["benchmark"],
                    "rollout_count": len(preparation.tasks),
                    "resumed": len(completed_records),
                    "pending": len(pending_tasks),
                }
            ),
            flush=True,
        )

        failures: list[dict[str, Any]] = []
        started_at = time.perf_counter()
        if pending_tasks:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "The standalone vLLM evaluator requires the 'openai' Python client in the active environment"
                ) from exc
            client = AsyncOpenAI(
                base_url=normalize_base_url(args.base_url),
                api_key=args.api_key,
                timeout=args.request_timeout_seconds,
                max_retries=args.max_retries,
            )
            generation_semaphore = asyncio.Semaphore(args.dp_size * args.max_concurrency_per_dp)
            scoring_semaphore = asyncio.Semaphore(args.score_max_in_flight)

            async def worker(task: RolloutTask) -> dict[str, Any]:
                return await evaluate_one(
                    client=client,
                    generation_semaphore=generation_semaphore,
                    scoring_semaphore=scoring_semaphore,
                    args=args,
                    tokenizer=preparation.tokenizer,
                    task=task,
                    fingerprint=fingerprint,
                )

            completed_now = 0
            max_pending = args.max_pending or max(args.dp_size * args.max_concurrency_per_dp * 2, 1)
            try:
                await check_served_model(client, args.served_model_name)
                async for record in bounded_results(pending_tasks, worker, max_pending=max_pending):
                    task = preparation.tasks[int(record["flat_index"])]
                    if record.get("error"):
                        failures.append(record)
                        write_json_atomic(failures_dir / f"{task.task_id}.json", record)
                    else:
                        write_json_gz_atomic(_part_path(parts_dir, task), record)
                        completed_records[task.task_id] = record
                        failure_path = failures_dir / f"{task.task_id}.json"
                        if failure_path.is_file():
                            failure_path.unlink()
                    completed_now += 1
                    total_done = len(completed_records)
                    if args.print_every and (
                        completed_now % args.print_every == 0 or completed_now == len(pending_tasks)
                    ):
                        print(
                            f"progress={total_done}/{len(preparation.tasks)} current_failures={len(failures)} "
                            f"elapsed_seconds={time.perf_counter() - started_at:.1f}",
                            flush=True,
                        )
            finally:
                await client.close()

        if failures:
            metadata["generation_error_count"] = len(failures)
            metadata["generation_completed_rollout_count"] = len(completed_records)
            metadata["generation_elapsed_seconds"] = time.perf_counter() - started_at
            write_json_atomic(output_dir / METADATA_FILENAME, metadata)
            print(
                f"ERROR: {len(failures)} rollout tasks failed; successful task shards were retained for resume",
                file=sys.stderr,
            )
            for failure in failures[:20]:
                print(
                    f"  {failure.get('benchmark_path')}/{failure.get('sample_id')} "
                    f"rollout={failure.get('rollout_index')}: {failure.get('error')}",
                    file=sys.stderr,
                )
            return 1

        ordered_records: list[dict[str, Any]] = []
        for task in preparation.tasks:
            record = completed_records.get(task.task_id) or _load_completed_part(
                _part_path(parts_dir, task), task, fingerprint
            )
            if record is None:
                raise RuntimeError(f"Missing completed rollout shard for task {task.task_id}")
            ordered_records.append(record)
        write_jsonl_gz_atomic(output_dir / ROLLOUTS_FILENAME, ordered_records)
        metadata["generation_complete"] = True
        metadata["generation_error_count"] = 0
        metadata["generation_completed_rollout_count"] = len(ordered_records)
        metadata["generation_elapsed_seconds"] = time.perf_counter() - started_at
        write_json_atomic(output_dir / METADATA_FILENAME, metadata)
        write_summary_artifacts(output_dir, metadata)
        marker = write_completion_marker(output_dir, metadata)
        if work_dir.is_dir():
            shutil.rmtree(work_dir)
        print(f"[generate] complete: {output_dir} fingerprint={marker['result_fingerprint']}")
        return 0


async def score_teacher_record(
    *,
    record: dict[str, Any],
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    args: argparse.Namespace,
    scoring_fingerprint: str,
) -> dict[str, Any]:
    try:
        teacher_logprobs, fallback = await score_sequence_logprobs(
            client=client,
            semaphore=semaphore,
            served_model_name=args.served_model_name,
            prompt_token_ids=[int(value) for value in record["prompt_token_ids"]],
            response_token_ids=[int(value) for value in record["response_token_ids"]],
        )
        student_logprobs = list(record.get("student_logprobs") or [])
        if len(student_logprobs) != len(teacher_logprobs):
            raise RuntimeError(
                f"student/teacher logprob lengths differ: {len(student_logprobs)} != {len(teacher_logprobs)}"
            )
        result = dict(record)
        result["teacher_logprobs"] = teacher_logprobs
        result["gap_metrics"] = compute_gap_metrics(
            student_logprobs,
            teacher_logprobs,
            high_gap_threshold=args.high_gap_threshold,
            tail_token_window=args.repetition_tail_token_window,
        )
        result["teacher_scoring_return_token_ids_fallback"] = fallback
        result["teacher_scoring_fingerprint"] = scoring_fingerprint
        result["teacher_scoring_error"] = None
        return result
    except Exception as exc:
        return {
            "task_id": record.get("task_id"),
            "benchmark_path": record.get("benchmark_path"),
            "sample_id": record.get("sample_id"),
            "rollout_index": record.get("rollout_index"),
            "teacher_scoring_error": f"{type(exc).__name__}: {exc}",
        }


def batched(iterator: Iterator[dict[str, Any]], batch_size: int) -> Iterator[list[dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    batch: list[dict[str, Any]] = []
    for item in iterator:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


async def teacher_score_main(args: argparse.Namespace) -> int:
    if args.score_max_in_flight <= 0 or args.score_batch_size <= 0:
        raise ValueError("--score-max-in-flight and --score-batch-size must be positive")
    output_dir = unit_output_dir(args.result_root, args.benchmark, args.global_step)
    with unit_lock(output_dir):
        teacher_work_root = output_dir / WORK_DIRNAME
        teacher_state_path = teacher_work_root / "teacher-state.json"
        valid, reason = validate_completed_unit(output_dir, validate_inputs=False)
        if not valid and not teacher_state_path.is_file():
            raise RuntimeError(f"Student generation is not complete under {output_dir}: {reason}")
        metadata = _read_metadata(output_dir)
        generation_is_recoverable = (
            metadata is not None and metadata.get("generation_complete") and (output_dir / ROLLOUTS_FILENAME).is_file()
        )
        if not generation_is_recoverable:
            raise RuntimeError(f"Student generation is not recoverable under {output_dir}: {reason}")
        assert metadata is not None
        if not metadata.get("request", {}).get("score_student_logprobs"):
            raise RuntimeError("Teacher scoring requires generation with --score-student-logprobs")

        teacher_tokenizer = AutoTokenizer.from_pretrained(
            args.teacher_tokenizer_model,
            trust_remote_code=args.trust_remote_code,
            local_files_only=True,
        )
        teacher_info = tokenizer_metadata(teacher_tokenizer, args.teacher_tokenizer_model)
        student_info = metadata.get("student_tokenizer") or {}
        fingerprints_match = teacher_info.get("fingerprint_sha256") == student_info.get(
            "fingerprint_sha256"
        ) and teacher_info.get("vocab_size") == student_info.get("vocab_size")
        if not fingerprints_match and not args.allow_tokenizer_mismatch:
            raise RuntimeError(
                "Student and teacher tokenizer vocabularies/IDs differ; direct token-ID scoring would be invalid"
            )

        teacher_model_path = Path(args.teacher_model_path or args.teacher_tokenizer_model).expanduser().resolve()
        teacher_input_manifest = capture_model_manifest(teacher_model_path)
        scoring_request = {
            "served_model_name": args.served_model_name,
            "teacher_model_path": str(teacher_model_path),
            "teacher_tokenizer_model": str(Path(args.teacher_tokenizer_model).expanduser().resolve()),
            "allow_tokenizer_mismatch": args.allow_tokenizer_mismatch,
            "high_gap_threshold": args.high_gap_threshold,
            "repetition_tail_token_window": args.repetition_tail_token_window,
        }
        scoring_fingerprint = payload_fingerprint(
            {
                "generation_fingerprint": metadata["generation_fingerprint"],
                "request": scoring_request,
                "teacher_input_manifest_fingerprint": teacher_input_manifest["fingerprint"],
                "teacher_tokenizer": teacher_info,
            }
        )
        existing_scoring_fingerprint = metadata.get("teacher_scoring_fingerprint")
        if metadata.get("teacher_scored") and not args.overwrite:
            if existing_scoring_fingerprint != scoring_fingerprint:
                raise RuntimeError(
                    "The completed teacher scoring used a different model or scoring configuration; "
                    "pass --overwrite only if replacing it is intentional"
                )
            analysis_changed = any(
                metadata["analysis"].get(key) != value
                for key, value in {
                    "bootstrap_resamples": args.bootstrap_resamples,
                    "bootstrap_seed": args.bootstrap_seed,
                }.items()
            )
            if analysis_changed or not valid:
                write_json_atomic(
                    teacher_state_path,
                    {
                        "teacher_scoring_fingerprint": scoring_fingerprint,
                        "mode": "analysis_refresh",
                    },
                )
                metadata["analysis"]["bootstrap_resamples"] = args.bootstrap_resamples
                metadata["analysis"]["bootstrap_seed"] = args.bootstrap_seed
                write_json_atomic(output_dir / METADATA_FILENAME, metadata)
                write_summary_artifacts(output_dir, metadata)
                write_completion_marker(output_dir, metadata)
                print(f"[teacher-score] refreshed derived artifacts: {output_dir}")
            else:
                print(f"[teacher-score] already complete: {output_dir}")
            if teacher_work_root.is_dir():
                shutil.rmtree(teacher_work_root)
            return 0

        if args.overwrite and teacher_work_root.is_dir():
            shutil.rmtree(teacher_work_root)
        existing_state = (
            json.loads(teacher_state_path.read_text(encoding="utf-8")) if teacher_state_path.is_file() else {}
        )
        resumable_fingerprint = existing_state.get("teacher_scoring_fingerprint")
        if resumable_fingerprint is not None and resumable_fingerprint != scoring_fingerprint:
            raise RuntimeError(
                "Resumable teacher shards were created with a different model or scoring configuration; "
                "pass --overwrite to discard those shards"
            )
        write_json_atomic(
            teacher_state_path,
            {
                "teacher_scoring_fingerprint": scoring_fingerprint,
                "teacher_scoring_request": scoring_request,
                "teacher_input_manifest": teacher_input_manifest,
                "teacher_tokenizer": teacher_info,
                "teacher_tokenizer_matches_student": fingerprints_match,
            },
        )

        source_records = list(iter_jsonl_gz(output_dir / ROLLOUTS_FILENAME))
        work_dir = teacher_work_root / "teacher"
        work_dir.mkdir(parents=True, exist_ok=True)
        completed: dict[str, dict[str, Any]] = {}
        pending: list[dict[str, Any]] = []
        for record in source_records:
            task_id = str(record["task_id"])
            part_path = work_dir / f"{task_id}.json.gz"
            existing = read_single_json_gz(part_path) if part_path.is_file() else None
            if (
                existing
                and not existing.get("teacher_scoring_error")
                and existing.get("teacher_scoring_fingerprint") == scoring_fingerprint
            ):
                completed[task_id] = existing
            else:
                pending.append(record)

        errors: list[dict[str, Any]] = []
        started_at = time.perf_counter()
        if pending:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "Teacher scoring requires the 'openai' Python client in the active environment"
                ) from exc
            client = AsyncOpenAI(
                base_url=normalize_base_url(args.base_url),
                api_key=args.api_key,
                timeout=args.request_timeout_seconds,
                max_retries=args.max_retries,
            )
            semaphore = asyncio.Semaphore(args.score_max_in_flight)
            try:
                await check_served_model(client, args.served_model_name)
                processed = 0
                for batch in batched(iter(pending), args.score_batch_size):
                    scored_batch = await asyncio.gather(
                        *[
                            score_teacher_record(
                                record=record,
                                client=client,
                                semaphore=semaphore,
                                args=args,
                                scoring_fingerprint=scoring_fingerprint,
                            )
                            for record in batch
                        ]
                    )
                    for record in scored_batch:
                        task_id = str(record["task_id"])
                        if record.get("teacher_scoring_error"):
                            errors.append(record)
                            write_json_atomic(work_dir / f"{task_id}.error.json", record)
                        else:
                            write_json_gz_atomic(work_dir / f"{task_id}.json.gz", record)
                            completed[task_id] = record
                        processed += 1
                    if args.print_every and (processed % args.print_every < len(batch) or processed == len(pending)):
                        print(
                            f"teacher_score_progress={len(completed)}/{len(source_records)} errors={len(errors)} "
                            f"elapsed_seconds={time.perf_counter() - started_at:.1f}",
                            flush=True,
                        )
            finally:
                await client.close()

        if errors:
            state = json.loads(teacher_state_path.read_text(encoding="utf-8"))
            state["teacher_scoring_error_count"] = len(errors)
            write_json_atomic(teacher_state_path, state)
            print(
                f"ERROR: teacher scoring failed for {len(errors)} rollouts; resumable shards retained",
                file=sys.stderr,
            )
            return 1

        ordered = [completed[str(record["task_id"])] for record in source_records]
        staged_rollouts = teacher_work_root / ROLLOUTS_FILENAME
        write_jsonl_gz_atomic(staged_rollouts, ordered)
        staged_rollouts.replace(output_dir / ROLLOUTS_FILENAME)
        metadata["teacher_scoring_request"] = scoring_request
        metadata["teacher_scoring_fingerprint"] = scoring_fingerprint
        metadata["teacher_input_manifest"] = teacher_input_manifest
        metadata["teacher_tokenizer"] = teacher_info
        metadata["teacher_tokenizer_matches_student"] = fingerprints_match
        metadata["teacher_served_model_name"] = args.served_model_name
        metadata["teacher_scored"] = True
        metadata["teacher_scoring_complete"] = True
        metadata["teacher_scoring_elapsed_seconds"] = time.perf_counter() - started_at
        metadata["teacher_scoring_rollout_count"] = len(ordered)
        metadata["teacher_scoring_error_count"] = 0
        metadata["analysis"]["high_gap_threshold"] = args.high_gap_threshold
        metadata["analysis"]["repetition_tail_token_window"] = args.repetition_tail_token_window
        metadata["analysis"]["bootstrap_resamples"] = args.bootstrap_resamples
        metadata["analysis"]["bootstrap_seed"] = args.bootstrap_seed
        write_json_atomic(output_dir / METADATA_FILENAME, metadata)
        write_summary_artifacts(output_dir, metadata)
        write_completion_marker(output_dir, metadata)
        if teacher_work_root.is_dir():
            shutil.rmtree(teacher_work_root)
        print(f"[teacher-score] complete: {output_dir}")
        return 0


def _repair_part_path(parts_dir: Path, task_id: str) -> Path:
    return parts_dir / f"{task_id}.json.gz"


def _load_repair_part(path: Path, *, task_id: str, repair_fingerprint: str) -> dict[str, Any] | None:
    record = read_single_json_gz(path) if path.is_file() else None
    repair = record.get("repair") if isinstance(record, dict) else None
    if (
        not isinstance(repair, dict)
        or repair.get("error")
        or record.get("task_id") != task_id
        or repair.get("fingerprint") != repair_fingerprint
    ):
        return None
    return record


def _repair_finish_reason(raw_record: dict[str, Any], interventions: list[dict[str, Any]]) -> str | None:
    if not interventions:
        value = raw_record.get("finish_reason")
        return str(value).lower() if value is not None else None
    last = interventions[-1]
    value = last.get("student_suffix_finish_reason") or last.get("teacher_finish_reason")
    if value is None:
        value = raw_record.get("finish_reason")
    return str(value).lower() if value is not None else None


def _sanitize_raw_student_logprobs(values: list[Any], *, floor: float) -> tuple[list[float], list[int]]:
    result: list[float] = []
    replaced: list[int] = []
    for index, value in enumerate(values):
        parsed = finite_float(value)
        if parsed is None:
            parsed = floor
            replaced.append(index)
        if parsed > 1e-5:
            raise RuntimeError(f"Raw student logprob at response position {index} is positive: {parsed!r}.")
        result.append(float(parsed))
    return result, replaced


async def repair_one_record(
    *,
    record: dict[str, Any],
    runner: Any,
    tokenizer: Any,
    metadata: dict[str, Any],
    repair_fingerprint: str,
    logprob_floor: float,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    try:
        prompt_ids = [int(value) for value in record["prompt_token_ids"]]
        response_ids = [int(value) for value in record["response_token_ids"]]
        raw_student_logprobs = list(record.get("student_logprobs") or [])
        if len(raw_student_logprobs) != len(response_ids):
            raise RuntimeError(
                "Raw generation did not retain one student logprob per token: "
                f"response_tokens={len(response_ids)}, student_logprobs={len(raw_student_logprobs)}."
            )
        student_logprobs, replaced_positions = _sanitize_raw_student_logprobs(
            raw_student_logprobs,
            floor=logprob_floor,
        )
        request = metadata.get("request") or {}
        sampling_params = {
            "temperature": float(request.get("temperature", 0.7)),
            "top_p": float(request.get("top_p", 0.95)),
            "top_k": int(request.get("top_k", -1)),
            "max_tokens": int(request.get("max_tokens", len(response_ids) or 1)),
        }
        raw_acc = float(bool(record.get("acc")))
        output = await runner.repair(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_logprobs=student_logprobs,
            raw_reward_score=raw_acc,
            raw_reward_extra_info={"acc": raw_acc},
            raw_correctness_value=raw_acc,
            raw_correct=bool(record.get("acc")),
            finish_reason=record.get("finish_reason"),
            sampling_params=sampling_params,
            request_id=f"standalone-repair:{record['task_id']}",
            sample_kwargs={
                "data_source": record.get("benchmark_path"),
                "index": record.get("sample_id"),
                "global_steps": metadata.get("global_step"),
            },
        )
        repaired_ids = [int(value) for value in output.response_ids]
        repaired_text = tokenizer.decode(repaired_ids, skip_special_tokens=True)
        interventions = [dict(item) for item in output.extra_fields.get("teacher_interventions", [])]
        proposals = [dict(item) for item in output.extra_fields.get("teacher_intervention_proposals", [])]
        skipped_interventions = [dict(item) for item in output.extra_fields.get("teacher_intervention_skipped", [])]
        gap_diagnostics = [dict(item) for item in output.extra_fields.get("teacher_intervention_gap_diagnostics", [])]
        finish_reason = _repair_finish_reason(record, interventions)
        scored = score_completion(
            repaired_text,
            record["ground_truth"],
            finish_reason=finish_reason,
            response_token_count=len(repaired_ids),
            verifier=str(request.get("verifier", "math_verify")),
            extra_info={
                "benchmark": record.get("benchmark"),
                "benchmark_path": record.get("benchmark_path"),
                "question_id": record.get("question_id"),
                "answer_type": record.get("answer_type"),
                "grader": record.get("grader"),
            },
        )
        analysis = metadata.get("analysis") or {}
        repetition = text_repetition_metrics(
            repaired_text,
            tail_chars=int(analysis.get("repetition_tail_chars", 10000)),
            threshold=float(analysis.get("repetition_threshold", 10.0)),
        )
        repetition.update(
            token_repetition_metrics(
                repaired_ids,
                tail_token_window=int(analysis.get("repetition_tail_token_window", 4096)),
                threshold=float(analysis.get("repetition_threshold", 10.0)),
                ngram_size=int(analysis.get("ngram_size", 4)),
            )
        )
        intervention_count = len(interventions)
        final_correct = bool(scored.acc)
        raw_correct = bool(record.get("acc"))
        transition = (
            "pre_correct_post_correct"
            if raw_correct and final_correct
            else "pre_correct_post_wrong"
            if raw_correct
            else "pre_wrong_post_correct"
            if final_correct
            else "pre_wrong_post_wrong"
        )
        outcome = dict(output.extra_fields.get("teacher_intervention_outcome") or {})
        outcome.update(
            {
                "final_reward_score": float(final_correct),
                "final_reward_extra_info": {"acc": float(final_correct)},
                "final_correctness_value": float(final_correct),
                "final_correct": final_correct,
                "final_response_length": len(repaired_ids),
                "final_finish_reason": finish_reason,
                "intervention_count": intervention_count,
                "repair_executed": intervention_count > 0,
                "repair_effective": intervention_count > 0 and not raw_correct and final_correct,
                "repair_ineffective": intervention_count > 0 and not raw_correct and not final_correct,
                "transition": transition,
            }
        )
        if interventions:
            interventions[0].update(
                {
                    "sample_outcome_summary": True,
                    "raw_correctness_value": float(raw_correct),
                    "raw_correct": raw_correct,
                    "final_correctness_value": float(final_correct),
                    "final_correct": final_correct,
                    "repair_effective": outcome["repair_effective"],
                    "repair_ineffective": outcome["repair_ineffective"],
                    "transition": transition,
                }
            )
        first_position = min((int(item["start"]) for item in interventions), default=None)
        repaired = dict(record)
        repaired["repair"] = {
            "fingerprint": repair_fingerprint,
            "text": repaired_text,
            "response_token_ids": repaired_ids,
            "response_tokens": len(repaired_ids),
            "response_token_id_count": len(repaired_ids),
            "finish_reason": finish_reason,
            "stop_reason": None,
            "eos_success": finish_reason in {"stop", "eos"},
            "truncated": finish_reason == "length",
            "acc": final_correct,
            "pred": str(scored.pred),
            "final_content": str(scored.final_content),
            "invalid_pred": bool(scored.invalid_pred),
            "answer_extractable": not bool(scored.invalid_pred),
            "has_answer_pattern": bool(scored.has_answer_pattern),
            "has_boxed_pattern": bool(scored.has_boxed_pattern),
            "accepted_answer_format": bool(scored.has_accepted_answer_format),
            "verified_by": str(scored.verified_by),
            "math_verify_score": scored.math_verify_score,
            "repetition": repetition,
            "trigger_detected": bool(outcome.get("trigger_detected")),
            "repair_executed": intervention_count > 0,
            "intervention_count": intervention_count,
            "first_intervention_position": first_position,
            "teacher_interventions": interventions,
            "teacher_intervention_proposals": proposals,
            "teacher_intervention_skipped": skipped_interventions,
            "gap_diagnostics": gap_diagnostics,
            "outcome": outcome,
            "raw_student_logprob_nonfinite_count": len(replaced_positions),
            "raw_student_logprob_nonfinite_positions": replaced_positions,
            "raw_student_logprob_nonfinite_replacement": logprob_floor if replaced_positions else None,
            "elapsed_seconds": time.perf_counter() - started_at,
            "error": None,
        }
        return repaired
    except Exception as exc:
        failed = dict(record)
        failed["repair"] = {
            "fingerprint": repair_fingerprint,
            "elapsed_seconds": time.perf_counter() - started_at,
            "error": f"{type(exc).__name__}: {exc}",
        }
        return failed


def _validate_repair_spec_against_unit(
    *,
    spec: dict[str, Any],
    state: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    semantic = spec.get("semantic_config") or {}
    if str(semantic.get("experiment_name")) != str(metadata.get("experiment")):
        raise RuntimeError(
            "Evaluation spec experiment does not match the generated unit: "
            f"spec={semantic.get('experiment_name')!r}, unit={metadata.get('experiment')!r}."
        )
    if int(state.get("checkpoint_step", -1)) != int(metadata.get("global_step", -2)):
        raise RuntimeError("Evaluation state checkpoint step does not match the generated unit.")
    intervention = (semantic.get("distillation") or {}).get("intervention") or {}
    if intervention.get("outcome_aware_candidate_selection_enabled") and state.get("teacher_nll_calibration") is None:
        raise RuntimeError(
            "Outcome-aware repair requires the checkpoint's saved teacher-NLL calibration; "
            "evaluation will not fit a threshold on test data."
        )


async def repair_main(args: argparse.Namespace) -> int:
    if args.max_in_flight <= 0 or args.max_pending <= 0:
        raise ValueError("--max-in-flight and --max-pending must be positive.")
    output_dir = unit_output_dir(args.result_root, args.benchmark, args.global_step)
    with unit_lock(output_dir):
        metadata = _read_metadata(output_dir)
        if (
            metadata is None
            or not metadata.get("generation_complete")
            or not (output_dir / ROLLOUTS_FILENAME).is_file()
        ):
            raise RuntimeError(f"Raw student generation is incomplete under {output_dir}.")
        if not metadata.get("request", {}).get("score_student_logprobs"):
            raise RuntimeError("Paired repair requires raw generation with --score-student-logprobs.")

        from verl.utils.evaluation.checkpoint_evaluation import load_evaluation_spec, load_evaluation_state
        from verl.utils.evaluation.teacher_intervention import (
            OpenAICompletionAdapter,
            OpenAICompletionAdapterConfig,
            StandaloneTeacherInterventionRunner,
        )

        spec_path = Path(args.evaluation_spec).expanduser().resolve()
        state_path = Path(args.evaluation_state).expanduser().resolve()
        spec = load_evaluation_spec(spec_path.parent)
        state = load_evaluation_state(
            state_path.parent,
            expected_step=args.global_step,
            expected_spec_fingerprint=str(spec["spec_fingerprint"]),
        )
        _validate_repair_spec_against_unit(spec=spec, state=state, metadata=metadata)
        semantic = spec["semantic_config"]
        intervention_config = dict((semantic.get("distillation") or {}).get("intervention") or {})
        logprob_floor = float((semantic.get("distillation") or {}).get("log_prob_min_clamp", -10.0))

        tokenizer = AutoTokenizer.from_pretrained(
            args.student_tokenizer_model,
            trust_remote_code=args.trust_remote_code,
            local_files_only=True,
        )
        student_tokenizer_info = tokenizer_metadata(tokenizer, args.student_tokenizer_model)
        unit_tokenizer_info = metadata.get("student_tokenizer") or {}
        if student_tokenizer_info.get("fingerprint_sha256") != unit_tokenizer_info.get(
            "fingerprint_sha256"
        ) or student_tokenizer_info.get("vocab_size") != unit_tokenizer_info.get("vocab_size"):
            raise RuntimeError("Repair tokenizer vocabulary differs from the raw-generation tokenizer.")
        teacher_tokenizer = AutoTokenizer.from_pretrained(
            args.teacher_tokenizer_model,
            trust_remote_code=args.trust_remote_code,
            local_files_only=True,
        )
        teacher_tokenizer_info = tokenizer_metadata(teacher_tokenizer, args.teacher_tokenizer_model)
        tokenizers_match = teacher_tokenizer_info.get("fingerprint_sha256") == student_tokenizer_info.get(
            "fingerprint_sha256"
        ) and teacher_tokenizer_info.get("vocab_size") == student_tokenizer_info.get("vocab_size")
        if not tokenizers_match and not args.allow_tokenizer_mismatch:
            raise RuntimeError(
                "Student and teacher tokenizer vocabularies/IDs differ; token-ID segment splicing would be invalid."
            )

        repair_request = {
            "generation_fingerprint": metadata["generation_fingerprint"],
            "spec_fingerprint": spec["spec_fingerprint"],
            "state_fingerprint": state["state_fingerprint"],
            "student_served_model_name": args.student_served_model_name,
            "teacher_served_model_name": args.teacher_served_model_name,
            "teacher_model_fingerprint": args.teacher_model_fingerprint,
            "student_tokenizer": student_tokenizer_info,
            "teacher_tokenizer": teacher_tokenizer_info,
            "intervention": intervention_config,
            "logprob_floor": logprob_floor,
        }
        repair_fingerprint = payload_fingerprint(repair_request)
        existing_fingerprint = metadata.get("repair_fingerprint")
        if metadata.get("repair_complete") and not args.overwrite:
            if existing_fingerprint != repair_fingerprint:
                raise RuntimeError(
                    "Completed paired repair used a different configuration; pass --overwrite only if replacing it "
                    "is intentional."
                )
            valid, reason = validate_completed_unit(output_dir, validate_inputs=False)
            if valid:
                print(f"[repair] already complete: {output_dir}")
                return 0
            raise RuntimeError(f"Completed paired repair artifacts are invalid: {reason}")

        work_root = output_dir / WORK_DIRNAME / "repair"
        if args.overwrite and work_root.is_dir():
            shutil.rmtree(work_root)
        parts_dir = work_root / "parts"
        failures_dir = work_root / "failures"
        parts_dir.mkdir(parents=True, exist_ok=True)
        failures_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            work_root / "repair-state.json",
            {"repair_fingerprint": repair_fingerprint, "repair_request": repair_request},
        )

        source_records = list(iter_jsonl_gz(output_dir / ROLLOUTS_FILENAME))
        completed: dict[str, dict[str, Any]] = {}
        pending: list[dict[str, Any]] = []
        for record in source_records:
            task_id = str(record["task_id"])
            completed_record = _load_repair_part(
                _repair_part_path(parts_dir, task_id),
                task_id=task_id,
                repair_fingerprint=repair_fingerprint,
            )
            if completed_record is None:
                existing_repair = record.get("repair")
                if isinstance(existing_repair, dict) and existing_repair.get("fingerprint") == repair_fingerprint:
                    if not existing_repair.get("error"):
                        completed_record = record
            if completed_record is None:
                pending.append(record)
            else:
                completed[task_id] = completed_record

        started_at = time.perf_counter()
        failures: list[dict[str, Any]] = []
        if pending:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise RuntimeError("Paired repair requires the 'openai' Python client.") from exc
            student_openai = AsyncOpenAI(
                base_url=normalize_base_url(args.student_base_url),
                api_key=args.student_api_key,
                timeout=args.request_timeout_seconds,
                max_retries=args.max_retries,
            )
            teacher_openai = AsyncOpenAI(
                base_url=normalize_base_url(args.teacher_base_url),
                api_key=args.teacher_api_key,
                timeout=args.request_timeout_seconds,
                max_retries=args.max_retries,
            )
            student_adapter = OpenAICompletionAdapter(
                client=student_openai,
                config=OpenAICompletionAdapterConfig(
                    served_model_name=args.student_served_model_name,
                    logprob_floor=logprob_floor,
                ),
                semaphore=asyncio.Semaphore(args.max_in_flight),
            )
            teacher_adapter = OpenAICompletionAdapter(
                client=teacher_openai,
                config=OpenAICompletionAdapterConfig(
                    served_model_name=args.teacher_served_model_name,
                    logprob_floor=logprob_floor,
                ),
                semaphore=asyncio.Semaphore(args.max_in_flight),
            )
            runner = StandaloneTeacherInterventionRunner(
                tokenizer=tokenizer,
                intervention_config=intervention_config,
                max_response_length=int((semantic.get("sampling") or {}).get("max_response_length")),
                logprob_floor=logprob_floor,
                student_client=student_adapter,
                teacher_client=teacher_adapter,
                teacher_nll_calibration=state.get("teacher_nll_calibration"),
                teacher_key=str((semantic.get("teacher") or {}).get("teacher_key") or "default"),
            )

            async def worker(record: dict[str, Any]) -> dict[str, Any]:
                return await repair_one_record(
                    record=record,
                    runner=runner,
                    tokenizer=tokenizer,
                    metadata=metadata,
                    repair_fingerprint=repair_fingerprint,
                    logprob_floor=logprob_floor,
                )

            processed = 0
            try:
                await check_served_model(student_openai, args.student_served_model_name)
                await check_served_model(teacher_openai, args.teacher_served_model_name)
                async for repaired_record in bounded_results(pending, worker, max_pending=args.max_pending):
                    task_id = str(repaired_record["task_id"])
                    repair = repaired_record["repair"]
                    if repair.get("error"):
                        failures.append(repaired_record)
                        write_json_atomic(failures_dir / f"{task_id}.json", repaired_record)
                    else:
                        write_json_gz_atomic(_repair_part_path(parts_dir, task_id), repaired_record)
                        completed[task_id] = repaired_record
                        (failures_dir / f"{task_id}.json").unlink(missing_ok=True)
                    processed += 1
                    if args.print_every and (processed % args.print_every == 0 or processed == len(pending)):
                        print(
                            f"repair_progress={len(completed)}/{len(source_records)} errors={len(failures)} "
                            f"elapsed_seconds={time.perf_counter() - started_at:.1f}",
                            flush=True,
                        )
            finally:
                await student_openai.close()
                await teacher_openai.close()

        if failures:
            repair_state = json.loads((work_root / "repair-state.json").read_text(encoding="utf-8"))
            repair_state.update(
                {
                    "repair_complete": False,
                    "repair_error_count": len(failures),
                    "repair_completed_rollout_count": len(completed),
                    "repair_elapsed_seconds": time.perf_counter() - started_at,
                }
            )
            write_json_atomic(work_root / "repair-state.json", repair_state)
            print(
                f"ERROR: paired repair failed for {len(failures)} rollouts; resumable shards retained",
                file=sys.stderr,
            )
            return 1

        ordered: list[dict[str, Any]] = []
        for source_record in source_records:
            task_id = str(source_record["task_id"])
            repaired_record = completed.get(task_id) or _load_repair_part(
                _repair_part_path(parts_dir, task_id),
                task_id=task_id,
                repair_fingerprint=repair_fingerprint,
            )
            if repaired_record is None:
                raise RuntimeError(f"Missing completed repair shard for task {task_id}.")
            ordered.append(repaired_record)
        staged_rollouts = work_root / ROLLOUTS_FILENAME
        write_jsonl_gz_atomic(staged_rollouts, ordered)
        staged_rollouts.replace(output_dir / ROLLOUTS_FILENAME)
        metadata.update(
            {
                "repair_complete": True,
                "repair_request": repair_request,
                "repair_fingerprint": repair_fingerprint,
                "repair_error_count": 0,
                "repair_completed_rollout_count": len(ordered),
                "repair_elapsed_seconds": time.perf_counter() - started_at,
                "repair_student_served_model_name": args.student_served_model_name,
                "repair_teacher_served_model_name": args.teacher_served_model_name,
                "repair_teacher_tokenizer_matches_student": tokenizers_match,
            }
        )
        write_json_atomic(output_dir / METADATA_FILENAME, metadata)
        write_summary_artifacts(output_dir, metadata)
        marker = write_completion_marker(output_dir, metadata)
        shutil.rmtree(work_root)
        print(f"[repair] complete: {output_dir} fingerprint={marker['result_fingerprint']}")
        return 0


def discover_units(result_root: str | Path) -> dict[tuple[str, int], Path]:
    root = Path(result_root).expanduser().resolve()
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
        for step_dir in sorted(directory.glob("global_step_*")):
            match = re.fullmatch(r"global_step_(\d+)", step_dir.name)
            if not match:
                continue
            valid, _ = validate_completed_unit(step_dir, validate_inputs=False)
            if valid:
                units[(normalized, int(match.group(1)))] = step_dir
    return dict(sorted(units.items()))


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _index_problem_rows(rows: list[dict[str, Any]], *, context: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id:
            raise ValueError(f"Missing sample_id in {context}")
        if sample_id in result:
            raise ValueError(f"Duplicate sample_id {sample_id!r} in {context}")
        result[sample_id] = row
    return result


def _comparison_compatibility(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    left_benchmark = left.get("benchmark") or {}
    right_benchmark = right.get("benchmark") or {}
    if left_benchmark.get("path") != right_benchmark.get("path"):
        issues.append("benchmark paths differ")
    if left_benchmark.get("dataset_fingerprint") != right_benchmark.get("dataset_fingerprint"):
        issues.append("dataset fingerprints differ")
    comparable_fields = (
        "rollout_n",
        "temperature",
        "top_p",
        "top_k",
        "max_tokens",
        "max_prompt_tokens",
        "base_seed",
        "enable_thinking",
        "verifier",
        "max_samples_per_benchmark",
    )
    left_request = left.get("request") or {}
    right_request = right.get("request") or {}
    for field in comparable_fields:
        if field in left_request and field in right_request and left_request[field] != right_request[field]:
            issues.append(f"request field {field!r} differs: {left_request[field]!r} != {right_request[field]!r}")
    left_tokenizer = left.get("student_tokenizer") or {}
    right_tokenizer = right.get("student_tokenizer") or {}
    if left_tokenizer.get("fingerprint_sha256") != right_tokenizer.get("fingerprint_sha256"):
        issues.append("student tokenizer vocabularies differ")
    if left_tokenizer.get("chat_template_sha256") != right_tokenizer.get("chat_template_sha256"):
        issues.append("chat templates differ")
    return issues


def paired_bootstrap(
    pairs: list[tuple[float, float]],
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    if not pairs:
        raise ValueError("paired_bootstrap received no pairs")
    differences_per_question = [right - left for left, right in pairs]
    observed = statistics.fmean(differences_per_question)
    if len(pairs) == 1 or resamples <= 0:
        return {
            "left_mean": statistics.fmean(left for left, _ in pairs),
            "right_mean": statistics.fmean(right for _, right in pairs),
            "difference_right_minus_left": observed,
            "ci95_low": observed,
            "ci95_high": observed,
            "probability_difference_gt_zero": float(observed > 0),
            "paired_randomization_p": None,
            "paired_question_count": len(pairs),
            "resamples": 0,
        }
    rng = random.Random(seed)
    size = len(pairs)
    bootstrap_values = [
        statistics.fmean(differences_per_question[rng.randrange(size)] for _ in range(size)) for _ in range(resamples)
    ]
    randomization_values = [
        abs(statistics.fmean(value if rng.randrange(2) else -value for value in differences_per_question))
        for _ in range(resamples)
    ]
    randomization_p = (sum(value >= abs(observed) for value in randomization_values) + 1) / (resamples + 1)
    return {
        "left_mean": statistics.fmean(left for left, _ in pairs),
        "right_mean": statistics.fmean(right for _, right in pairs),
        "difference_right_minus_left": observed,
        "ci95_low": percentile(bootstrap_values, 0.025),
        "ci95_high": percentile(bootstrap_values, 0.975),
        "probability_difference_gt_zero": sum(value > 0 for value in bootstrap_values) / resamples,
        "paired_randomization_p": randomization_p,
        "paired_question_count": len(pairs),
        "resamples": resamples,
    }


def compare_main(args: argparse.Namespace) -> int:
    left_root = Path(args.left_root).expanduser().resolve()
    right_root = Path(args.right_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    left_units = discover_units(left_root)
    right_units = discover_units(right_root)
    requested = {normalize_benchmark_path(value) for value in args.benchmarks} if args.benchmarks else None
    if requested is not None:
        left_units = {key: value for key, value in left_units.items() if key[0] in requested}
        right_units = {key: value for key, value in right_units.items() if key[0] in requested}
    left_keys = set(left_units)
    right_keys = set(right_units)
    missing_left = sorted(right_keys - left_keys)
    missing_right = sorted(left_keys - right_keys)
    if (missing_left or missing_right) and not args.allow_missing_units:
        raise RuntimeError(
            "Evaluation roots do not contain the same completed benchmark-step units. "
            f"missing_left={missing_left}, missing_right={missing_right}"
        )
    common_keys = sorted(left_keys & right_keys)
    if not common_keys:
        raise RuntimeError(f"No common completed benchmark-step units in {left_root} and {right_root}")

    metrics = args.metrics or DEFAULT_BOOTSTRAP_METRICS
    rows: list[dict[str, Any]] = []
    nested: dict[str, Any] = {}
    coverage: list[dict[str, Any]] = []
    for benchmark_path, step in common_keys:
        left_dir = left_units[(benchmark_path, step)]
        right_dir = right_units[(benchmark_path, step)]
        left_metadata = _read_metadata(left_dir) or {}
        right_metadata = _read_metadata(right_dir) or {}
        compatibility_issues = _comparison_compatibility(left_metadata, right_metadata)
        if compatibility_issues and not args.allow_incompatible:
            raise RuntimeError(
                f"Cannot compare benchmark={benchmark_path} step={step}: " + "; ".join(compatibility_issues)
            )
        left_index = _index_problem_rows(
            read_csv_rows(left_dir / PER_PROBLEM_FILENAME), context=f"{left_dir / PER_PROBLEM_FILENAME}"
        )
        right_index = _index_problem_rows(
            read_csv_rows(right_dir / PER_PROBLEM_FILENAME), context=f"{right_dir / PER_PROBLEM_FILENAME}"
        )
        left_samples = set(left_index)
        right_samples = set(right_index)
        left_only = sorted(left_samples - right_samples)
        right_only = sorted(right_samples - left_samples)
        if (left_only or right_only) and not args.allow_partial_samples:
            raise RuntimeError(
                f"Sample coverage differs for benchmark={benchmark_path} step={step}: "
                f"left_only={left_only[:10]}, right_only={right_only[:10]}"
            )
        sample_ids = sorted(left_samples & right_samples)
        for sample_id in sample_ids:
            left_problem = left_index[sample_id].get("problem_fingerprint")
            right_problem = right_index[sample_id].get("problem_fingerprint")
            if left_problem and right_problem and left_problem != right_problem:
                raise RuntimeError(
                    f"sample_id={sample_id!r} identifies different problems for benchmark={benchmark_path} step={step}"
                )
        unit_result: dict[str, Any] = {}
        for metric in metrics:
            pairs: list[tuple[float, float]] = []
            for sample_id in sample_ids:
                left_value = finite_float(left_index[sample_id].get(metric))
                right_value = finite_float(right_index[sample_id].get(metric))
                if left_value is not None and right_value is not None:
                    pairs.append((left_value, right_value))
            if not pairs:
                continue
            result = paired_bootstrap(
                pairs,
                resamples=args.bootstrap_resamples,
                seed=stable_seed(args.bootstrap_seed, step, benchmark_path, metric),
            )
            unit_result[metric] = result
            rows.append({"step": step, "benchmark": benchmark_path, "metric": metric, **result})
        nested.setdefault(benchmark_path, {})[str(step)] = unit_result
        coverage.append(
            {
                "benchmark": benchmark_path,
                "step": step,
                "paired_sample_count": len(sample_ids),
                "left_only_sample_count": len(left_only),
                "right_only_sample_count": len(right_only),
                "compatibility_issues": compatibility_issues,
            }
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "left_root": str(left_root),
        "right_root": str(right_root),
        "difference_definition": "right_minus_left",
        "bootstrap_unit": "paired question/sample_id",
        "bootstrap_resamples": args.bootstrap_resamples,
        "missing_left_units": missing_left,
        "missing_right_units": missing_right,
        "coverage": coverage,
        "results": nested,
    }
    write_json_atomic(output_dir / "paired_bootstrap.json", payload)
    fields = [
        "step",
        "benchmark",
        "metric",
        "left_mean",
        "right_mean",
        "difference_right_minus_left",
        "ci95_low",
        "ci95_high",
        "probability_difference_gt_zero",
        "paired_randomization_p",
        "paired_question_count",
        "resamples",
    ]
    write_csv_atomic(output_dir / "paired_bootstrap.csv", rows, fields)
    for row in rows:
        if row["metric"] == "acc":
            print(
                f"step={row['step']} benchmark={row['benchmark']} metric=acc "
                f"right_minus_left={row['difference_right_minus_left']:.6f} "
                f"ci95=[{row['ci95_low']:.6f},{row['ci95_high']:.6f}] "
                f"p_randomization={row['paired_randomization_p']}",
                flush=True,
            )
    print(f"Paired comparison outputs: {output_dir}")
    return 0


def report_main(args: argparse.Namespace) -> int:
    units = discover_units(args.result_root)
    if not units:
        raise RuntimeError(f"No completed benchmark-step units under {args.result_root}")
    print("=" * 150)
    print(f"Benchmark diagnostic report: {Path(args.result_root).expanduser().resolve()}")
    print("=" * 150)
    print(
        f"{'step':>6} {'benchmark':<32} {'acc':>9} {'eos':>9} {'trunc':>9} "
        f"{'extract':>9} {'rep-token':>11} {'len-mean':>11} {'len-p90':>11}"
    )
    for (benchmark_path, step), unit_dir in units.items():
        summary = json.loads((unit_dir / SUMMARY_FILENAME).read_text(encoding="utf-8"))
        metrics = summary.get("metrics") or {}
        lengths = metrics.get("response_tokens") or {}

        def fmt(value: Any, *, percentage: bool = False, width: int = 9) -> str:
            parsed = finite_float(value)
            if parsed is None:
                return f"{'N/A':>{width}}"
            if percentage:
                return f"{parsed * 100:{width}.2f}%"
            return f"{parsed:{width}.1f}"

        print(
            f"{step:6d} {benchmark_path:<32} "
            f"{fmt(metrics.get('acc'), percentage=True)} "
            f"{fmt(metrics.get('eos_success_rate'), percentage=True)} "
            f"{fmt(metrics.get('truncation_rate'), percentage=True)} "
            f"{fmt(metrics.get('answer_extractable_rate'), percentage=True)} "
            f"{fmt(metrics.get('repetition_token_zlib_rate'), percentage=True, width=11)} "
            f"{fmt(lengths.get('mean'), width=11)} "
            f"{fmt(lengths.get('p90'), width=11)}"
        )
    print("=" * 150)
    return 0


def status_main(args: argparse.Namespace) -> int:
    output_dir = unit_output_dir(args.result_root, args.benchmark, args.global_step)
    expectations = parse_expectations(args.expect)
    valid, reason = validate_completed_unit(
        output_dir,
        expectations=expectations,
        validate_inputs=args.validate_inputs,
        verify_checksums=args.verify_checksums,
        allow_legacy=args.allow_legacy,
    )
    if valid:
        metadata = _read_metadata(output_dir) or {}
        if args.require_teacher:
            if not metadata.get("teacher_scored") or not metadata.get("teacher_scoring_complete"):
                print(f"incomplete output_dir={output_dir} reason=teacher scoring is incomplete", file=sys.stderr)
                return 1
            teacher_manifest = metadata.get("teacher_input_manifest")
            if args.validate_inputs and teacher_manifest:
                teacher_valid, teacher_reason = validate_input_manifest(teacher_manifest)
                if not teacher_valid:
                    print(
                        f"incomplete output_dir={output_dir} reason=teacher {teacher_reason}",
                        file=sys.stderr,
                    )
                    return 1
        if args.require_repair and not metadata.get("repair_complete"):
            print(f"incomplete output_dir={output_dir} reason=paired repair is incomplete", file=sys.stderr)
            return 1
        print(f"complete output_dir={output_dir}")
        return 0
    print(f"incomplete output_dir={output_dir} reason={reason}", file=sys.stderr)
    return 1


def _legacy_request(metadata: dict[str, Any], benchmark_path: str) -> dict[str, Any]:
    generation = metadata.get("generation") or {}
    result = {
        "experiment": metadata.get("experiment"),
        "global_step": metadata.get("global_step"),
        "benchmark": benchmark_path,
    }
    for field in (
        "dataset_root",
        "rollout_n",
        "temperature",
        "top_p",
        "top_k",
        "max_tokens",
        "max_prompt_tokens",
        "base_seed",
        "enable_thinking",
        "verifier",
        "score_student_logprobs",
    ):
        if field in generation:
            result[field] = generation[field]
    return result


def _legacy_analysis(metadata: dict[str, Any]) -> dict[str, Any]:
    analysis = {
        "length_bin_edges": list(DEFAULT_LENGTH_BIN_EDGES),
        "repetition_tail_chars": 10000,
        "repetition_tail_token_window": 4096,
        "repetition_threshold": 10.0,
        "ngram_size": 4,
        "high_gap_threshold": 1.0,
        "bootstrap_resamples": 5000,
        "bootstrap_seed": 20260722,
    }
    analysis.update(metadata.get("analysis") or {})
    analysis["length_bin_edges"] = normalize_length_edges([int(value) for value in analysis["length_bin_edges"]])
    return analysis


def _normalize_legacy_record(record: dict[str, Any], benchmark_path: str, identity: str) -> dict[str, Any]:
    result = dict(record)
    sample_id = str(result.get("sample_id"))
    rollout_index = int(result.get("rollout_index", 0))
    result["benchmark_id"] = identity
    result["benchmark_path"] = benchmark_path
    result["task_id"] = hashlib.sha256(f"{benchmark_path}\0{sample_id}\0{rollout_index}".encode()).hexdigest()[:24]
    result["problem_fingerprint"] = payload_fingerprint(
        {
            "sample_id": sample_id,
            "problem": result.get("problem"),
            "ground_truth": result.get("ground_truth"),
        }
    )
    repetition = result.get("repetition")
    if isinstance(repetition, dict) and "tail_ngram_repeat_fraction" not in repetition:
        repetition["tail_ngram_repeat_fraction"] = repetition.get("tail_4gram_repeat_fraction")
        repetition.setdefault("ngram_size", 4)
    gaps = result.get("gap_metrics")
    if isinstance(gaps, dict) and "last_window" not in gaps and "last_4096" in gaps:
        gaps["last_window"] = gaps.pop("last_4096")
    return result


def migrate_legacy_main(args: argparse.Namespace) -> int:
    result_root = Path(args.result_root).expanduser().resolve()
    migrated = 0
    skipped = 0
    for legacy_dir in sorted(result_root.glob("global_step_*")):
        match = re.fullmatch(r"global_step_(\d+)", legacy_dir.name)
        metadata_path = legacy_dir / METADATA_FILENAME
        rollouts_path = legacy_dir / ROLLOUTS_FILENAME
        if not match or not metadata_path.is_file() or not rollouts_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not metadata.get("generation_complete"):
            continue
        paths = [normalize_benchmark_path(value) for value in (metadata.get("generation") or {}).get("benchmarks", [])]
        by_name: dict[str, str] = {}
        for path in paths:
            name = Path(path).name
            if name in by_name:
                raise RuntimeError(f"Cannot migrate legacy results with duplicate benchmark basename {name!r}")
            by_name[name] = path
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in iter_jsonl_gz(rollouts_path):
            name = str(record.get("benchmark"))
            if name not in by_name:
                raise RuntimeError(f"Legacy record benchmark {name!r} has no path in metadata")
            grouped[by_name[name]].append(record)
        step = int(match.group(1))
        for path, raw_records in grouped.items():
            output_dir = unit_output_dir(result_root, path, step)
            if (output_dir / COMPLETION_FILENAME).is_file() and not args.overwrite:
                skipped += 1
                continue
            if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
                raise RuntimeError(f"Migration target is not empty: {output_dir}")
            output_dir.mkdir(parents=True, exist_ok=True)
            identity = benchmark_id(path)
            records = [_normalize_legacy_record(record, path, identity) for record in raw_records]
            records.sort(key=lambda record: (int(record.get("question_index", 0)), int(record.get("rollout_index", 0))))
            dataset_records: dict[str, dict[str, Any]] = {}
            for record in records:
                dataset_records[str(record["sample_id"])] = {
                    "sample_id": str(record["sample_id"]),
                    "problem": record.get("problem"),
                    "ground_truth": record.get("ground_truth"),
                }
            dataset_fingerprint = payload_fingerprint([dataset_records[key] for key in sorted(dataset_records)])
            request = _legacy_request(metadata, path)
            legacy_fingerprint = "legacy-" + payload_fingerprint(
                {"source": str(legacy_dir), "benchmark": path, "request": request}
            )
            for record in records:
                record["generation_fingerprint"] = legacy_fingerprint
            migrated_metadata = {
                "schema_version": SCHEMA_VERSION,
                "generation_schema_version": 1,
                "legacy_migrated": True,
                "legacy_source_dir": str(legacy_dir),
                "experiment": metadata.get("experiment") or result_root.name,
                "global_step": step,
                "benchmark": {
                    "id": identity,
                    "name": Path(path).name,
                    "path": path,
                    "dataset_fingerprint": dataset_fingerprint,
                },
                "served_model_name": metadata.get("served_model_name"),
                "teacher_scored": bool(metadata.get("teacher_scored")),
                "teacher_served_model_name": metadata.get("teacher_served_model_name"),
                "generation_complete": True,
                "request": request,
                "analysis": _legacy_analysis(metadata),
                "generation_fingerprint": legacy_fingerprint,
                "input_manifest": {"fingerprint": payload_fingerprint({})},
                "student_tokenizer": metadata.get("student_tokenizer") or {},
                "problem_count": len(dataset_records),
                "rollout_count": len(records),
                "source": {"legacy_migration": True},
            }
            experiment_path = result_root / "experiment.json"
            if not experiment_path.is_file():
                write_json_atomic(
                    experiment_path,
                    {"schema_version": SCHEMA_VERSION, "experiment": migrated_metadata["experiment"]},
                )
            root = benchmark_root(result_root, path)
            root.mkdir(parents=True, exist_ok=True)
            benchmark_metadata_path = root / "benchmark.json"
            if not benchmark_metadata_path.is_file():
                write_json_atomic(
                    benchmark_metadata_path,
                    {"schema_version": SCHEMA_VERSION, "id": identity, "name": Path(path).name, "path": path},
                )
            write_jsonl_gz_atomic(output_dir / ROLLOUTS_FILENAME, records)
            write_json_atomic(output_dir / METADATA_FILENAME, migrated_metadata)
            write_summary_artifacts(output_dir, migrated_metadata)
            write_completion_marker(output_dir, migrated_metadata)
            migrated += 1
            print(f"[migrate] {legacy_dir} benchmark={path} -> {output_dir}")
    print(f"Legacy migration complete: migrated={migrated} skipped={skipped}")
    return 0


def add_server_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--request-timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--max-retries", type=int, default=2)


def add_analysis_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--length-bin-edges", nargs="+", type=int, default=DEFAULT_LENGTH_BIN_EDGES)
    parser.add_argument("--repetition-tail-chars", type=int, default=10000)
    parser.add_argument("--repetition-tail-token-window", type=int, default=4096)
    parser.add_argument("--repetition-threshold", type=float, default=10.0)
    parser.add_argument("--ngram-size", type=int, default=4)
    parser.add_argument("--high-gap-threshold", type=float, default=1.0)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260722)


def add_unit_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--global-step", type=int, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate one benchmark for one student checkpoint")
    add_server_arguments(generate)
    add_unit_arguments(generate)
    generate.add_argument("--experiment", required=True)
    generate.add_argument("--dataset-root", required=True)
    generate.add_argument("--student-model-path", required=True)
    generate.add_argument(
        "--student-model-identity",
        default=None,
        help=(
            "Stable identity of the immutable source checkpoint. When omitted, the evaluator hashes model weights "
            "to avoid relying on merged-file mtimes."
        ),
    )
    generate.add_argument("--student-tokenizer-model", required=True)
    generate.add_argument("--chat-template-model", required=True)
    generate.add_argument("--trust-remote-code", action="store_true")
    generate.add_argument("--rollout-n", type=int, default=16)
    generate.add_argument("--temperature", type=float, default=0.7)
    generate.add_argument("--top-p", type=float, default=0.95)
    generate.add_argument("--top-k", type=int, default=-1)
    generate.add_argument("--max-tokens", type=int, default=23 * 1024)
    generate.add_argument("--max-prompt-tokens", type=int, default=1024)
    generate.add_argument("--dp-size", type=int, default=8)
    generate.add_argument("--max-concurrency-per-dp", type=int, default=64)
    generate.add_argument("--max-pending", type=int, default=None)
    generate.add_argument("--score-max-in-flight", type=int, default=128)
    generate.add_argument(
        "--score-student-logprobs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Prerequisite for optional teacher gap scoring.",
    )
    generate.add_argument(
        "--require-exact-token-ids",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail instead of retokenizing completion text when vLLM does not return generated token IDs.",
    )
    generate.add_argument("--max-samples-per-benchmark", type=int, default=-1)
    generate.add_argument("--base-seed", type=int, default=42)
    generate.add_argument("--print-every", type=int, default=50)
    generate.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    generate.add_argument("--verifier", choices=["math_dapo", "math_verify", "hybrid"], default="math_verify")
    add_analysis_arguments(generate)
    generate.add_argument("--overwrite", action="store_true")
    generate.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate or migrate result compatibility without contacting the model server.",
    )

    teacher = subparsers.add_parser("teacher-score", help="Attach teacher logprobs to one completed unit")
    add_server_arguments(teacher)
    add_unit_arguments(teacher)
    teacher.add_argument("--teacher-tokenizer-model", required=True)
    teacher.add_argument(
        "--teacher-model-path",
        default=None,
        help="Teacher checkpoint used by the server; defaults to --teacher-tokenizer-model.",
    )
    teacher.add_argument("--trust-remote-code", action="store_true")
    teacher.add_argument("--allow-tokenizer-mismatch", action="store_true")
    teacher.add_argument("--score-max-in-flight", type=int, default=128)
    teacher.add_argument("--score-batch-size", type=int, default=256)
    teacher.add_argument("--print-every", type=int, default=50)
    teacher.add_argument("--repetition-tail-token-window", type=int, default=4096)
    teacher.add_argument("--high-gap-threshold", type=float, default=1.0)
    teacher.add_argument("--bootstrap-resamples", type=int, default=5000)
    teacher.add_argument("--bootstrap-seed", type=int, default=20260722)
    teacher.add_argument("--overwrite", action="store_true")

    repair = subparsers.add_parser("repair", help="Run paired DMPD repair on completed raw rollouts")
    add_unit_arguments(repair)
    repair.add_argument("--evaluation-spec", required=True)
    repair.add_argument("--evaluation-state", required=True)
    repair.add_argument("--student-base-url", required=True)
    repair.add_argument("--student-api-key", default="EMPTY")
    repair.add_argument("--student-served-model-name", required=True)
    repair.add_argument("--teacher-base-url", required=True)
    repair.add_argument("--teacher-api-key", default="EMPTY")
    repair.add_argument("--teacher-served-model-name", required=True)
    repair.add_argument("--teacher-model-fingerprint", required=True)
    repair.add_argument("--student-tokenizer-model", required=True)
    repair.add_argument("--teacher-tokenizer-model", required=True)
    repair.add_argument("--trust-remote-code", action="store_true")
    repair.add_argument("--allow-tokenizer-mismatch", action="store_true")
    repair.add_argument("--request-timeout-seconds", type=float, default=7200.0)
    repair.add_argument("--max-retries", type=int, default=2)
    repair.add_argument("--max-in-flight", type=int, default=128)
    repair.add_argument("--max-pending", type=int, default=256)
    repair.add_argument("--print-every", type=int, default=50)
    repair.add_argument("--overwrite", action="store_true")

    status = subparsers.add_parser("status", help="Validate one benchmark-step completion marker")
    add_unit_arguments(status)
    status.add_argument("--expect", action="append", default=[], help="Expected request value in KEY=JSON_VALUE form")
    status.add_argument("--validate-inputs", action=argparse.BooleanOptionalAction, default=True)
    status.add_argument("--verify-checksums", action="store_true")
    status.add_argument("--allow-legacy", action="store_true")
    status.add_argument("--require-teacher", action="store_true")
    status.add_argument("--require-repair", action="store_true")

    report = subparsers.add_parser("report", help="Print all completed benchmark-step summaries")
    report.add_argument("--result-root", required=True)

    compare = subparsers.add_parser("compare", help="Question-level paired comparison across experiment roots")
    compare.add_argument("--left-root", required=True, help="Baseline result root")
    compare.add_argument("--right-root", required=True, help="Candidate result root; differences are right-left")
    compare.add_argument("--output-dir", required=True)
    compare.add_argument("--benchmarks", nargs="+", default=None)
    compare.add_argument("--metrics", nargs="+", default=None)
    compare.add_argument("--bootstrap-resamples", type=int, default=10000)
    compare.add_argument("--bootstrap-seed", type=int, default=20260722)
    compare.add_argument("--allow-missing-units", action="store_true")
    compare.add_argument("--allow-partial-samples", action="store_true")
    compare.add_argument("--allow-incompatible", action="store_true")

    migrate = subparsers.add_parser("migrate-legacy", help="Split legacy mixed-benchmark step outputs")
    migrate.add_argument("--result-root", required=True)
    migrate.add_argument("--overwrite", action="store_true")

    identity = subparsers.add_parser("benchmark-id", help="Print the stable directory ID for a benchmark path")
    identity.add_argument("--benchmark", required=True)

    unit_path = subparsers.add_parser("unit-path", help="Print the output directory for one benchmark-step unit")
    add_unit_arguments(unit_path)

    checkpoint = subparsers.add_parser("checkpoint-ready", help="Validate a merged Hugging Face checkpoint")
    checkpoint.add_argument("--path", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "generate":
        return asyncio.run(generate_main(args))
    if args.command == "teacher-score":
        return asyncio.run(teacher_score_main(args))
    if args.command == "repair":
        return asyncio.run(repair_main(args))
    if args.command == "status":
        return status_main(args)
    if args.command == "report":
        return report_main(args)
    if args.command == "compare":
        return compare_main(args)
    if args.command == "migrate-legacy":
        return migrate_legacy_main(args)
    if args.command == "benchmark-id":
        print(benchmark_id(args.benchmark))
        return 0
    if args.command == "unit-path":
        print(unit_output_dir(args.result_root, args.benchmark, args.global_step))
        return 0
    if args.command == "checkpoint-ready":
        ready, reason = validate_hf_checkpoint(args.path)
        print(reason)
        return 0 if ready else 1
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
