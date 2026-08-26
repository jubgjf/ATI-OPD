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
"""Preprocess NVIDIA Nemotron-Cascade-2 MOPD prompts into verl's RL schema.

The public MOPD file contains prompts and verifier metadata, but no model
responses. This converter deliberately preserves ``responses_create_params.input``
without adding a system prompt, applying a chat template, truncating, or filtering
by token length. It removes the unsupported Workplace tool-use domain and the
known, programmatically detectable instruction-following contradictions before
creating an exact, domain-stratified 95/5 split.

The source identity is pinned because NVIDIA currently publishes MOPD as one
mutable ``train.jsonl`` file. A different revision, byte size, SHA256, row count,
domain mixture, or cleaning result fails loudly instead of silently changing the
experiment. Network access is opt-in through ``--download``.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import datasets

DATA_SOURCE = "nvidia/Nemotron-Cascade-2-RL-data"
DATASET_CONFIG = "MOPD"
DATASET_REVISION = "05bbaf03bac608e804efc86c5f8bf99844f2d197"
SOURCE_FILENAME = "MOPD/train.jsonl"
SOURCE_SHA256 = "a0a5e7619832317519ef593645a182d65a2806ef4408d7426ca18507bc0588dc"
SOURCE_BYTES = 28_496_971
SOURCE_ROWS = 6_171

REWARD_DATA_SOURCE = "nemotron_cascade_mopd"
DEFAULT_LOCAL_DATASET_PATH = os.environ.get(
    "NEMOTRON_CASCADE_2_RL_MOPD_LOCAL_PATH",
    "dataset/Nemotron-Cascade-2-RL-data",
)
DEFAULT_LOCAL_SAVE_DIR = os.environ.get(
    "NEMOTRON_CASCADE_2_RL_MOPD_SAVE_PATH",
    "dataset/Nemotron-Cascade-2-RL-data-MOPD-verl--qwen3",
)
DEFAULT_TEST_FRACTION = 0.05
DEFAULT_SEED = 42

ACEREASON = "acereason"
INSTRUCTION_FOLLOWING = "nano-v3-RL-instruction_following"
STRUCTURED_OUTPUT = "nano_v3_sft_profiled_structured_outputs"
STEM_MCQA = "nano_v3_sft_profiled_stem_mcqa"
WORKPLACE = "nano_v3_sft_profiled_workbench"
IMPOSSIBLE_PARAGRAPH_CONSTRAINT = "length_constraints:nth_paragraph_first_word"

DOMAIN_ORDER = (ACEREASON, INSTRUCTION_FOLLOWING, STRUCTURED_OUTPUT, STEM_MCQA)
DOMAIN_TO_ABILITY = {
    ACEREASON: "math",
    INSTRUCTION_FOLLOWING: "instruction_following",
    STRUCTURED_OUTPUT: "structured_output",
    STEM_MCQA: "stem_mcqa",
}
EXPECTED_RAW_DOMAIN_COUNTS = {
    ACEREASON: 1_853,
    INSTRUCTION_FOLLOWING: 1_854,
    STRUCTURED_OUTPUT: 927,
    STEM_MCQA: 927,
    WORKPLACE: 610,
}
EXPECTED_CLEAN_DOMAIN_COUNTS = {
    ACEREASON: 1_853,
    INSTRUCTION_FOLLOWING: 1_830,
    STRUCTURED_OUTPUT: 927,
    STEM_MCQA: 927,
}
EXPECTED_TRAIN_DOMAIN_COUNTS = {
    ACEREASON: 1_760,
    INSTRUCTION_FOLLOWING: 1_738,
    STRUCTURED_OUTPUT: 881,
    STEM_MCQA: 881,
}
EXPECTED_TEST_DOMAIN_COUNTS = {
    ACEREASON: 93,
    INSTRUCTION_FOLLOWING: 92,
    STRUCTURED_OUTPUT: 46,
    STEM_MCQA: 46,
}
EXPECTED_WORKPLACE_DROPS = 610
EXPECTED_IMPOSSIBLE_IF_DROPS = 24
EXPECTED_CLEAN_ROWS = 5_537
EXPECTED_TRAIN_ROWS = 5_260
EXPECTED_TEST_ROWS = 277


def _expand_path(path: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(path))))


def _ordered_counts(counts: Counter[str] | dict[str, int], include_workplace: bool = False) -> dict[str, int]:
    domains = (*DOMAIN_ORDER, WORKPLACE) if include_workplace else DOMAIN_ORDER
    return {domain: int(counts.get(domain, 0)) for domain in domains}


def _validate_revision(revision: str) -> None:
    if revision != DATASET_REVISION:
        raise ValueError(
            f"This converter is pinned to {DATA_SOURCE}@{DATASET_REVISION}; got revision={revision!r}. "
            "Audit a new source file and update its revision/hash/count constants together."
        )


def resolve_source_file(
    local_dataset_path: str | Path,
    download: bool = False,
    revision: str = DATASET_REVISION,
) -> Path:
    """Resolve a local MOPD JSONL file, downloading the pinned file only on request."""
    _validate_revision(revision)
    local_path = _expand_path(local_dataset_path)
    candidates = [local_path] if local_path.is_file() else [local_path / SOURCE_FILENAME, local_path / "train.jsonl"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    if download:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=DATA_SOURCE,
            repo_type="dataset",
            filename=SOURCE_FILENAME,
            revision=revision,
        )
        return Path(downloaded)

    raise FileNotFoundError(
        f"Could not find {SOURCE_FILENAME} under local path {local_path}. Download the dataset first or rerun "
        "with --download; network access is intentionally disabled by default."
    )


def load_validated_source_rows(
    source_file: str | Path,
    *,
    expected_sha256: str = SOURCE_SHA256,
    expected_bytes: int = SOURCE_BYTES,
    expected_rows: int = SOURCE_ROWS,
    revision: str = DATASET_REVISION,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parse JSONL while validating the pinned byte hash, byte size, and physical row count."""
    _validate_revision(revision)
    source_path = _expand_path(source_file)
    digest = hashlib.sha256()
    rows: list[dict[str, Any]] = []
    byte_count = 0

    with source_path.open("rb") as file:
        for line_number, raw_line in enumerate(file, start=1):
            digest.update(raw_line)
            byte_count += len(raw_line)
            if not raw_line.strip():
                raise ValueError(f"{source_path} contains a blank physical line at line {line_number}.")
            try:
                row = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"{source_path} contains invalid JSON at line {line_number}.") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{source_path} line {line_number} must be a JSON object.")
            rows.append(row)

    actual_sha256 = digest.hexdigest()
    mismatches = []
    if byte_count != expected_bytes:
        mismatches.append(f"bytes={byte_count} (expected {expected_bytes})")
    if actual_sha256 != expected_sha256:
        mismatches.append(f"sha256={actual_sha256} (expected {expected_sha256})")
    if len(rows) != expected_rows:
        mismatches.append(f"rows={len(rows)} (expected {expected_rows})")
    if mismatches:
        raise ValueError(
            f"MOPD source identity mismatch for {source_path}: {'; '.join(mismatches)}. "
            "Do not preprocess an unaudited source revision."
        )

    identity = {
        "dataset": DATA_SOURCE,
        "config": DATASET_CONFIG,
        "revision": revision,
        "source_filename": SOURCE_FILENAME,
        "resolved_source_file": str(source_path.resolve()),
        "bytes": byte_count,
        "sha256": actual_sha256,
        "rows": len(rows),
    }
    return rows, identity


def _require_positive_integer(value: Any, field_name: str, source_index: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"MOPD row {source_index} has invalid {IMPOSSIBLE_PARAGRAPH_CONSTRAINT}.{field_name}={value!r}."
        )
    return value


def detect_impossible_if_paragraph_constraints(row: dict[str, Any], source_index: int = -1) -> list[dict[str, Any]]:
    """Detect constraints that request paragraph ``n`` while allowing fewer than ``n`` paragraphs."""
    if row.get("dataset") != INSTRUCTION_FOLLOWING:
        return []

    instruction_ids = row.get("instruction_id_list")
    constraint_kwargs = row.get("kwargs")
    if not isinstance(instruction_ids, list) or not isinstance(constraint_kwargs, list):
        raise ValueError(f"MOPD IF row {source_index} must contain list instruction_id_list and kwargs fields.")
    if len(instruction_ids) != len(constraint_kwargs):
        raise ValueError(
            f"MOPD IF row {source_index} has misaligned instruction_id_list/kwargs lengths: "
            f"{len(instruction_ids)} != {len(constraint_kwargs)}."
        )

    contradictions = []
    for instruction_id, kwargs in zip(instruction_ids, constraint_kwargs, strict=True):
        if instruction_id != IMPOSSIBLE_PARAGRAPH_CONSTRAINT:
            continue
        if not isinstance(kwargs, dict):
            raise ValueError(f"MOPD IF row {source_index} has non-object kwargs for {IMPOSSIBLE_PARAGRAPH_CONSTRAINT}.")
        num_paragraphs = _require_positive_integer(kwargs.get("num_paragraphs"), "num_paragraphs", source_index)
        nth_paragraph = _require_positive_integer(kwargs.get("nth_paragraph"), "nth_paragraph", source_index)
        if nth_paragraph > num_paragraphs:
            contradictions.append(
                {
                    "constraint_id": instruction_id,
                    "num_paragraphs": num_paragraphs,
                    "nth_paragraph": nth_paragraph,
                    "first_word": kwargs.get("first_word"),
                }
            )
    return contradictions


def _validate_prompt(row: dict[str, Any], source_index: int) -> list[dict[str, Any]]:
    create_params = row.get("responses_create_params")
    if not isinstance(create_params, dict):
        raise ValueError(f"MOPD row {source_index} must contain object responses_create_params.")
    prompt = create_params.get("input")
    if not isinstance(prompt, list) or not prompt:
        raise ValueError(f"MOPD row {source_index} must contain a non-empty responses_create_params.input list.")
    for message_index, message in enumerate(prompt):
        if not isinstance(message, dict):
            raise ValueError(f"MOPD row {source_index} prompt message {message_index} must be an object.")
        if not isinstance(message.get("role"), str) or not isinstance(message.get("content"), str):
            raise ValueError(
                f"MOPD row {source_index} prompt message {message_index} must have string role/content fields."
            )
    return prompt


def _audit_sample_id(row: dict[str, Any], source_index: int) -> str:
    """Return the public UUID when available, otherwise a pinned-source row ID."""
    source_id = row.get("uuid")
    if isinstance(source_id, str) and source_id.strip():
        return source_id
    return f"{DATASET_REVISION[:12]}:{source_index}"


def clean_source_rows(
    rows: list[dict[str, Any]],
    *,
    strict_current_source: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Drop Workplace and the detected impossible IF constraints, recording source indices."""
    raw_counts = Counter(row.get("dataset") for row in rows)
    if strict_current_source and _ordered_counts(raw_counts, include_workplace=True) != EXPECTED_RAW_DOMAIN_COUNTS:
        raise ValueError(
            f"Unexpected raw MOPD domain counts: {dict(raw_counts)}; expected {EXPECTED_RAW_DOMAIN_COUNTS}."
        )

    clean_rows: list[dict[str, Any]] = []
    workplace_indices: list[int] = []
    workplace_rows: list[dict[str, Any]] = []
    impossible_if_rows: list[dict[str, Any]] = []
    for source_index, row in enumerate(rows):
        source_domain = row.get("dataset")
        if source_domain == WORKPLACE:
            workplace_indices.append(source_index)
            workplace_rows.append({"source_index": source_index, "sample_id": _audit_sample_id(row, source_index)})
            continue
        if source_domain not in DOMAIN_TO_ABILITY:
            raise ValueError(f"MOPD row {source_index} has unsupported dataset domain {source_domain!r}.")

        _validate_prompt(row, source_index)
        contradictions = detect_impossible_if_paragraph_constraints(row, source_index)
        if contradictions:
            impossible_if_rows.append(
                {
                    "source_index": source_index,
                    "sample_id": _audit_sample_id(row, source_index),
                    "constraints": contradictions,
                }
            )
            continue

        indexed_row = dict(row)
        indexed_row["_source_index"] = source_index
        clean_rows.append(indexed_row)

    clean_counts = Counter(row["dataset"] for row in clean_rows)
    if strict_current_source:
        errors = []
        if len(workplace_indices) != EXPECTED_WORKPLACE_DROPS:
            errors.append(f"workplace_drops={len(workplace_indices)}")
        if len(impossible_if_rows) != EXPECTED_IMPOSSIBLE_IF_DROPS:
            errors.append(f"impossible_if_drops={len(impossible_if_rows)}")
        if len(clean_rows) != EXPECTED_CLEAN_ROWS:
            errors.append(f"clean_rows={len(clean_rows)}")
        if _ordered_counts(clean_counts) != EXPECTED_CLEAN_DOMAIN_COUNTS:
            errors.append(f"clean_domain_counts={dict(clean_counts)}")
        if errors:
            raise ValueError(f"Unexpected MOPD cleaning result: {', '.join(errors)}.")

    audit = {
        "raw_rows": len(rows),
        "raw_domain_counts": _ordered_counts(raw_counts, include_workplace=True),
        "dropped_rows": len(workplace_indices) + len(impossible_if_rows),
        "dropped_by_reason": {
            "unsupported_workplace_tool_environment": len(workplace_indices),
            "impossible_if_paragraph_constraint": len(impossible_if_rows),
        },
        "workplace_source_indices": workplace_indices,
        "workplace_rows": workplace_rows,
        "impossible_if_rows": impossible_if_rows,
        "clean_rows": len(clean_rows),
        "clean_domain_counts": _ordered_counts(clean_counts),
    }
    return clean_rows, audit


def _stable_seed(seed: int, *parts: str) -> int:
    payload = "\0".join((str(seed), *parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big")


def resolve_test_count(total_rows: int, test_fraction: float = DEFAULT_TEST_FRACTION) -> int:
    if total_rows < 2:
        raise ValueError(f"At least two clean rows are required to split MOPD, got {total_rows}.")
    if not 0 < test_fraction < 1:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction!r}.")
    test_rows = math.floor(total_rows * test_fraction + 0.5)
    if not 0 < test_rows < total_rows:
        raise ValueError(f"test_fraction={test_fraction} resolves to invalid test_rows={test_rows}.")
    return test_rows


def allocate_domain_test_counts(domain_sizes: dict[str, int], test_rows: int) -> dict[str, int]:
    """Use largest remainders to obtain an exact proportional test allocation."""
    total_rows = sum(domain_sizes.values())
    if not 0 < test_rows < total_rows:
        raise ValueError(f"test_rows must be between 1 and {total_rows - 1}, got {test_rows}.")
    if any(size <= 0 for size in domain_sizes.values()):
        raise ValueError(f"All domain strata must be non-empty, got {domain_sizes}.")

    ideals = {domain: size * test_rows / total_rows for domain, size in domain_sizes.items()}
    allocations = {domain: min(math.floor(ideals[domain]), size - 1) for domain, size in domain_sizes.items()}
    remaining = test_rows - sum(allocations.values())
    domain_rank = {domain: index for index, domain in enumerate(DOMAIN_ORDER)}

    while remaining:
        candidates = [domain for domain, size in domain_sizes.items() if allocations[domain] < size - 1]
        if not candidates:
            raise ValueError(f"Cannot allocate {test_rows} test rows while retaining one train row per domain.")
        candidates.sort(
            key=lambda domain: (
                -(ideals[domain] - allocations[domain]),
                domain_rank.get(domain, len(domain_rank)),
                domain,
            )
        )
        allocations[candidates[0]] += 1
        remaining -= 1
    return {domain: allocations[domain] for domain in domain_sizes}


def build_domain_stratified_split(
    clean_rows: list[dict[str, Any]],
    *,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    seed: int = DEFAULT_SEED,
    strict_current_source: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Create a deterministic exact split, sampling independently inside each source domain."""
    test_rows = resolve_test_count(len(clean_rows), test_fraction)
    rows_by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in clean_rows:
        rows_by_domain[row["dataset"]].append(row)
    domain_sizes = {domain: len(rows_by_domain[domain]) for domain in DOMAIN_ORDER if rows_by_domain[domain]}
    allocations = allocate_domain_test_counts(domain_sizes, test_rows)

    train: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for domain in DOMAIN_ORDER:
        domain_rows = list(rows_by_domain.get(domain, []))
        if not domain_rows:
            continue
        random.Random(_stable_seed(seed, domain)).shuffle(domain_rows)
        domain_test_rows = allocations[domain]
        test.extend(domain_rows[:domain_test_rows])
        train.extend(domain_rows[domain_test_rows:])

    random.Random(_stable_seed(seed, "train")).shuffle(train)
    random.Random(_stable_seed(seed, "test")).shuffle(test)
    train_counts = Counter(row["dataset"] for row in train)
    test_counts = Counter(row["dataset"] for row in test)

    if strict_current_source:
        errors = []
        if len(train) != EXPECTED_TRAIN_ROWS or len(test) != EXPECTED_TEST_ROWS:
            errors.append(f"train/test={len(train)}/{len(test)}")
        if _ordered_counts(train_counts) != EXPECTED_TRAIN_DOMAIN_COUNTS:
            errors.append(f"train_domain_counts={dict(train_counts)}")
        if _ordered_counts(test_counts) != EXPECTED_TEST_DOMAIN_COUNTS:
            errors.append(f"test_domain_counts={dict(test_counts)}")
        if errors:
            raise ValueError(f"Unexpected MOPD split result: {', '.join(errors)}.")

    split_audit = {
        "strategy": "domain_stratified_largest_remainder",
        "seed": seed,
        "test_fraction_requested": test_fraction,
        "clean_rows": len(clean_rows),
        "train_rows": len(train),
        "test_rows": len(test),
        "test_fraction_actual": len(test) / len(clean_rows),
        "domain_counts": {
            domain: {
                "clean": domain_sizes[domain],
                "train": int(train_counts[domain]),
                "test": int(test_counts[domain]),
            }
            for domain in DOMAIN_ORDER
            if domain in domain_sizes
        },
    }
    return train, test, split_audit


def build_ground_truth_payload(row: dict[str, Any], source_index: int) -> dict[str, Any]:
    """Build a JSON-serializable verifier payload without claiming a shared reward implementation."""
    source_domain = row["dataset"]
    if source_domain == ACEREASON:
        expected_answer = row.get("expected_answer")
        if not isinstance(expected_answer, str) or not expected_answer.strip():
            raise ValueError(f"MOPD AceReason row {source_index} has an empty expected_answer.")
        return {"verifier": "math_answer", "expected_answer": expected_answer}

    if source_domain == INSTRUCTION_FOLLOWING:
        instruction_ids = row.get("instruction_id_list")
        constraint_kwargs = row.get("kwargs")
        if not isinstance(instruction_ids, list) or not isinstance(constraint_kwargs, list):
            raise ValueError(f"MOPD IF row {source_index} has invalid verifier metadata.")
        if len(instruction_ids) != len(constraint_kwargs):
            raise ValueError(f"MOPD IF row {source_index} has misaligned verifier metadata.")
        return {
            "verifier": "instruction_following",
            "instruction_id_list": copy.deepcopy(instruction_ids),
            "kwargs": copy.deepcopy(constraint_kwargs),
        }

    if source_domain == STRUCTURED_OUTPUT:
        schema_str = row.get("schema_str")
        if not isinstance(schema_str, str) or not schema_str.strip():
            raise ValueError(f"MOPD structured-output row {source_index} has an empty schema_str.")
        try:
            schema = json.loads(schema_str)
        except json.JSONDecodeError as exc:
            raise ValueError(f"MOPD structured-output row {source_index} has invalid schema_str JSON.") from exc
        return {
            "verifier": "json_schema",
            "schema_type": row.get("schema_type"),
            "schema": schema,
        }

    if source_domain == STEM_MCQA:
        expected_answer = row.get("expected_answer")
        template_metadata = row.get("template_metadata")
        if not isinstance(expected_answer, str) or not expected_answer.strip():
            raise ValueError(f"MOPD STEM row {source_index} has an empty expected_answer.")
        if not isinstance(template_metadata, dict) or not isinstance(template_metadata.get("output_regex"), str):
            raise ValueError(f"MOPD STEM row {source_index} has invalid template_metadata.output_regex.")
        return {
            "verifier": "mcqa_regex",
            "expected_answer": expected_answer,
            "output_regex": template_metadata["output_regex"],
        }

    raise ValueError(f"MOPD row {source_index} has unsupported retained domain {source_domain!r}.")


def to_verl_record(row: dict[str, Any], split: str) -> dict[str, Any]:
    """Convert one clean source row while preserving its input messages exactly."""
    source_index = int(row["_source_index"])
    source_domain = row["dataset"]
    prompt = _validate_prompt(row, source_index)
    ground_truth = json.dumps(
        build_ground_truth_payload(row, source_index),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    pass_rate = row.get("pass_rate")
    if pass_rate is not None:
        pass_rate = float(pass_rate)

    return {
        "data_source": REWARD_DATA_SOURCE,
        "prompt": copy.deepcopy(prompt),
        "ability": DOMAIN_TO_ABILITY[source_domain],
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {
            "index": str(source_index),
            "source_index": source_index,
            "split": split,
            "source_dataset": DATA_SOURCE,
            "source_config": DATASET_CONFIG,
            "source_revision": DATASET_REVISION,
            "source_domain": source_domain,
            "source_id": row.get("uuid"),
            "source_pass_rate": pass_rate,
            "reward_data_source": REWARD_DATA_SOURCE,
        },
    }


def to_verl_dataset(rows: list[dict[str, Any]], split: str) -> datasets.Dataset:
    if not rows:
        raise ValueError(f"Cannot emit an empty MOPD {split} split.")
    return datasets.Dataset.from_list([to_verl_record(row, split=split) for row in rows])


def preprocess_source_file(
    source_file: str | Path,
    *,
    revision: str = DATASET_REVISION,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    seed: int = DEFAULT_SEED,
) -> tuple[datasets.Dataset, datasets.Dataset, dict[str, Any]]:
    """Validate, clean, split, and convert the pinned public MOPD source file."""
    source_rows, source_identity = load_validated_source_rows(source_file, revision=revision)
    clean_rows, cleaning_audit = clean_source_rows(source_rows, strict_current_source=True)
    train_rows, test_rows, split_audit = build_domain_stratified_split(
        clean_rows,
        test_fraction=test_fraction,
        seed=seed,
        strict_current_source=True,
    )
    train_dataset = to_verl_dataset(train_rows, split="train")
    test_dataset = to_verl_dataset(test_rows, split="test")
    audit = {
        "source": source_identity,
        "cleaning": cleaning_audit,
        "split": split_audit,
        "output": {
            "reward_data_source": REWARD_DATA_SOURCE,
            "schema": ["data_source", "prompt", "ability", "reward_model", "extra_info"],
            "ground_truth_encoding": "canonical_json_string",
            "prompt_source": "responses_create_params.input",
            "prompt_preserved_exactly": True,
            "token_length_filtering": False,
            "truncation": False,
        },
    }
    return train_dataset, test_dataset, audit


def _normalize_token_ids(tokenized: Any) -> list[int]:
    input_ids = tokenized.input_ids if hasattr(tokenized, "input_ids") else tokenized
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
        if len(input_ids) != 1:
            raise ValueError(f"Expected one tokenized prompt, got batch size {len(input_ids)}.")
        input_ids = input_ids[0]
    if not isinstance(input_ids, list):
        raise TypeError(f"Unsupported chat-template token output type: {type(input_ids).__name__}.")
    return input_ids


def _percentile(sorted_values: list[int], fraction: float) -> float:
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _token_length_summary(lengths: list[int]) -> dict[str, Any]:
    if not lengths:
        raise ValueError("Cannot summarize an empty token-length collection.")
    ordered = sorted(lengths)
    return {
        "rows": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "p50": _percentile(ordered, 0.50),
        "p90": _percentile(ordered, 0.90),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
        "max": ordered[-1],
        "over_threshold": {
            str(threshold): sum(length > threshold for length in ordered) for threshold in (1_024, 2_048, 4_096, 8_192)
        },
    }


def build_qwen_chat_template_token_audit(
    train_dataset: datasets.Dataset,
    test_dataset: datasets.Dataset,
    tokenizer: Any,
    *,
    tokenizer_path: str,
    tokenizer_revision: str | None = None,
) -> dict[str, Any]:
    """Measure Qwen chat-template prompt lengths without selecting or modifying any row."""
    grouped_lengths: dict[str, list[int]] = defaultdict(list)
    split_lengths: dict[str, list[int]] = defaultdict(list)
    all_lengths: list[int] = []
    for split, dataset in (("train", train_dataset), ("test", test_dataset)):
        for row in dataset:
            tokenized = tokenizer.apply_chat_template(
                row["prompt"],
                add_generation_prompt=True,
                tokenize=True,
                enable_thinking=True,
            )
            length = len(_normalize_token_ids(tokenized))
            all_lengths.append(length)
            split_lengths[split].append(length)
            grouped_lengths[row["ability"]].append(length)

    return {
        "tokenizer_path": tokenizer_path,
        "tokenizer_revision": tokenizer_revision,
        "chat_template_kwargs": {"add_generation_prompt": True, "tokenize": True, "enable_thinking": True},
        "used_for_filtering": False,
        "overall": _token_length_summary(all_lengths),
        "by_split": {split: _token_length_summary(lengths) for split, lengths in sorted(split_lengths.items())},
        "by_domain": {domain: _token_length_summary(lengths) for domain, lengths in sorted(grouped_lengths.items())},
    }


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert the pinned Nemotron-Cascade-2 MOPD JSONL to clean, stratified verl parquet files."
    )
    parser.add_argument(
        "--local_dataset_path",
        default=DEFAULT_LOCAL_DATASET_PATH,
        help=f"Local dataset root, MOPD directory, or {SOURCE_FILENAME} file.",
    )
    parser.add_argument(
        "--local_save_dir",
        default=DEFAULT_LOCAL_SAVE_DIR,
        help="Directory for train.parquet, test.parquet, and audit.json.",
    )
    parser.add_argument("--revision", default=DATASET_REVISION, help="Pinned Hugging Face dataset revision.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Within-domain split seed. Default: 42.")
    parser.add_argument(
        "--test_fraction",
        type=float,
        default=DEFAULT_TEST_FRACTION,
        help="Exact domain-stratified held-out fraction. The audited default resolves to 277 rows.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the pinned source file only when it is unavailable locally.",
    )
    parser.add_argument(
        "--qwen_tokenizer_path",
        default=None,
        help="Optional local/cached Qwen tokenizer for a non-filtering chat-template token audit.",
    )
    parser.add_argument("--qwen_tokenizer_revision", default=None)
    parser.add_argument(
        "--allow_tokenizer_download",
        action="store_true",
        help="Allow the optional tokenizer audit to fetch missing tokenizer files.",
    )
    args = parser.parse_args()

    source_file = resolve_source_file(args.local_dataset_path, download=args.download, revision=args.revision)
    train_dataset, test_dataset, audit = preprocess_source_file(
        source_file,
        revision=args.revision,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )

    save_dir = _expand_path(args.local_save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    train_path = save_dir / "train.parquet"
    test_path = save_dir / "test.parquet"
    train_dataset.to_parquet(str(train_path))
    test_dataset.to_parquet(str(test_path))

    if args.qwen_tokenizer_path:
        from transformers import AutoTokenizer

        tokenizer_kwargs: dict[str, Any] = {"local_files_only": not args.allow_tokenizer_download}
        if args.qwen_tokenizer_revision:
            tokenizer_kwargs["revision"] = args.qwen_tokenizer_revision
        tokenizer = AutoTokenizer.from_pretrained(args.qwen_tokenizer_path, **tokenizer_kwargs)
        token_audit = build_qwen_chat_template_token_audit(
            train_dataset,
            test_dataset,
            tokenizer,
            tokenizer_path=args.qwen_tokenizer_path,
            tokenizer_revision=args.qwen_tokenizer_revision,
        )
        token_audit_path = save_dir / "qwen_token_audit.json"
        _write_json(token_audit_path, token_audit)
        audit["output"]["optional_token_audit"] = token_audit_path.name

    audit["output"]["train_parquet"] = train_path.name
    audit["output"]["test_parquet"] = test_path.name
    _write_json(save_dir / "audit.json", audit)

    print(f"Saved MOPD train split to {train_path} ({len(train_dataset)} rows)", flush=True)
    print(f"Saved MOPD test split to {test_path} ({len(test_dataset)} rows)", flush=True)
    print(f"Saved source, cleaning, and split audit to {save_dir / 'audit.json'}", flush=True)


if __name__ == "__main__":
    main()
