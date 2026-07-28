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
"""Preprocess zwhe99/DeepMath-103K into verl's RL parquet schema.

DeepMath-103K only provides a train split. This script creates a deterministic
held-out split stratified jointly by difficulty and broad mathematical topic.
Rare singleton strata stay in the training split. The output keeps only the
fields used by verl, so the three long R1 solution columns are not duplicated.
"""

import argparse
import hashlib
import heapq
import json
import math
import numbers
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import datasets

DATA_SOURCE = "zwhe99/DeepMath-103K"
DATASET_REVISION = os.environ.get("DEEPMATH_103K_REVISION", "5cf055d1fe3d7a2eb19719ac020211469736ae44")
REWARD_DATA_SOURCE = "math_verify"
DEFAULT_LOCAL_DATASET_PATH = os.environ.get("DEEPMATH_103K_LOCAL_PATH", "~/data/DeepMath-103K")
DEFAULT_LOCAL_SAVE_DIR = os.environ.get(
    "DEEPMATH_103K_SAVE_PATH",
    "~/data/DeepMath-103K-Processed-verl",
)
DEFAULT_VALIDATION_SIZE = 0.05
DEFAULT_SEED = 42
PROMPT_STYLE_BOXED = "boxed"
PROMPT_STYLE_RAW = "raw"
BOXED_MATH_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
REQUIRED_COLUMNS = ("question", "final_answer", "difficulty", "topic")
INVALID_ROW_POLICIES = ("drop", "error")
STRATIFICATION_FIELDS = ("difficulty", "topic_group")


def _expand_path(path: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(path)))


def difficulty_bucket(difficulty: Any) -> str:
    """Return a stable string representation for a DeepMath difficulty value."""
    if isinstance(difficulty, bool) or not isinstance(difficulty, numbers.Real):
        raise ValueError(f"DeepMath difficulty must be a finite number, got {difficulty!r}.")
    value = float(difficulty)
    if not math.isfinite(value):
        raise ValueError(f"DeepMath difficulty must be finite, got {difficulty!r}.")
    return f"{value:g}"


def topic_group(topic: Any) -> str:
    """Collapse DeepMath's topic hierarchy to its broad mathematical topic."""
    if not isinstance(topic, str) or not topic.strip():
        raise ValueError(f"DeepMath topic must be a non-empty string, got {topic!r}.")

    parts = [part.strip() for part in topic.split("->") if part.strip()]
    if not parts:
        raise ValueError(f"DeepMath topic must contain a non-empty hierarchy, got {topic!r}.")
    if parts[0].casefold() == "mathematics":
        return parts[1] if len(parts) > 1 else "Other"
    return parts[0]


def stratification_key(difficulty: Any, topic: Any) -> tuple[str, str]:
    return difficulty_bucket(difficulty), topic_group(topic)


def format_stratification_key(key: tuple[str, str]) -> str:
    return f"difficulty={key[0]}|topic_group={key[1]}"


def build_prompt(question: Any, prompt_style: str = PROMPT_STYLE_BOXED) -> list[dict[str, str]]:
    """Build a single-user-message prompt without injecting a system message."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"DeepMath question must be a non-empty string, got {question!r}.")
    question = question.strip()

    if prompt_style == PROMPT_STYLE_BOXED:
        content = question if question.endswith(BOXED_MATH_INSTRUCTION) else f"{question}\n\n{BOXED_MATH_INSTRUCTION}"
    elif prompt_style == PROMPT_STYLE_RAW:
        content = question
    else:
        raise ValueError(
            f"prompt_style must be one of {PROMPT_STYLE_BOXED!r}/{PROMPT_STYLE_RAW!r}, got {prompt_style!r}."
        )

    return [{"role": "user", "content": content}]


def _load_parquet_files(path: Path) -> datasets.DatasetDict:
    if path.is_file():
        parquet_files = [str(path)]
    else:
        preferred_files = sorted((path / "data").glob("train-*.parquet"))
        if not preferred_files:
            preferred_files = sorted(path.glob("train-*.parquet"))
        parquet_files = [str(file_path) for file_path in preferred_files or sorted(path.rglob("*.parquet"))]

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under local DeepMath-103K path: {path}")
    return datasets.load_dataset("parquet", data_files={"train": parquet_files})


def load_deepmath_dataset(
    local_dataset_path: str,
    download: bool = False,
    revision: str = DATASET_REVISION,
) -> datasets.DatasetDict:
    """Load DeepMath-103K locally by default and download only when requested."""
    dataset_path = _expand_path(local_dataset_path)
    if dataset_path.exists():
        if (dataset_path / "dataset_dict.json").exists() or (dataset_path / "state.json").exists():
            loaded = datasets.load_from_disk(str(dataset_path))
            return loaded if isinstance(loaded, datasets.DatasetDict) else datasets.DatasetDict({"train": loaded})

        if dataset_path.is_file() or any(dataset_path.rglob("*.parquet")):
            return _load_parquet_files(dataset_path)

        try:
            loaded = datasets.load_dataset(str(dataset_path))
            return loaded if isinstance(loaded, datasets.DatasetDict) else datasets.DatasetDict({"train": loaded})
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load local DeepMath-103K from {dataset_path}. Expected a Hugging Face dataset "
                "directory, a dataset saved with save_to_disk, or raw parquet shards."
            ) from exc

    if download:
        hf_endpoint = os.environ.get("HF_ENDPOINT")
        if hf_endpoint:
            print(f"HF_ENDPOINT={hf_endpoint}", flush=True)
        return datasets.load_dataset(DATA_SOURCE, revision=revision)

    raise FileNotFoundError(
        f"Local DeepMath-103K path does not exist: {dataset_path}. Download it first, or rerun with --download. "
        "For mirror downloads, export HF_ENDPOINT=https://hf-mirror.com."
    )


def _row_validation_errors(example: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    question = example.get("question")
    if not isinstance(question, str) or not question.strip():
        errors.append("empty question")

    final_answer = example.get("final_answer")
    if not isinstance(final_answer, str) or not final_answer.strip():
        errors.append("empty final_answer")

    try:
        difficulty_bucket(example.get("difficulty"))
    except ValueError as exc:
        errors.append(str(exc))

    try:
        topic_group(example.get("topic"))
    except ValueError as exc:
        errors.append(str(exc))
    return errors


def validate_source_dataset(
    dataset: datasets.Dataset,
    invalid_row_policy: str = "drop",
) -> tuple[datasets.Dataset, list[dict[str, Any]]]:
    """Validate raw rows, remove unused solution columns, and audit invalid rows."""
    if invalid_row_policy not in INVALID_ROW_POLICIES:
        raise ValueError(f"invalid_row_policy must be one of {INVALID_ROW_POLICIES}, got {invalid_row_policy!r}.")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataset.column_names]
    if missing_columns:
        raise ValueError(
            f"{DATA_SOURCE} is missing required columns {missing_columns}; available columns: {dataset.column_names}"
        )

    selected_columns = list(REQUIRED_COLUMNS)
    if "source_index" in dataset.column_names:
        selected_columns.append("source_index")
    projected = dataset.select_columns(selected_columns)
    if "source_index" not in projected.column_names:
        projected = projected.add_column("source_index", list(range(len(projected))))

    valid_indices: list[int] = []
    dropped_rows: list[dict[str, Any]] = []
    for local_index, example in enumerate(projected):
        errors = _row_validation_errors(example)
        if not errors:
            valid_indices.append(local_index)
            continue

        question = example.get("question")
        dropped_rows.append(
            {
                "source_index": int(example["source_index"]),
                "reasons": errors,
                "question": question if isinstance(question, str) else repr(question),
                "difficulty": example.get("difficulty"),
                "topic": example.get("topic"),
            }
        )

    if dropped_rows and invalid_row_policy == "error":
        first = dropped_rows[0]
        raise ValueError(
            f"{DATA_SOURCE} contains {len(dropped_rows)} invalid rows. First invalid source_index="
            f"{first['source_index']}: {', '.join(first['reasons'])}. Use --invalid_row_policy=drop to audit "
            "and exclude them."
        )
    if not valid_indices:
        raise ValueError(f"{DATA_SOURCE} contains no valid rows after schema validation.")

    return projected.select(valid_indices), dropped_rows


def parse_validation_size(value: str) -> int | float:
    """Parse an integer row count or a fractional validation size."""
    stripped = value.strip()
    try:
        if any(character in stripped.casefold() for character in (".", "e")):
            return float(stripped)
        return int(stripped)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"validation_size must be an integer row count or a fraction in (0, 1), got {value!r}."
        ) from exc


def resolve_validation_count(total_rows: int, validation_size: int | float) -> int:
    if total_rows < 2:
        raise ValueError(f"At least two valid rows are required to create a split, got {total_rows}.")

    if isinstance(validation_size, bool):
        raise ValueError(f"validation_size must be an integer or fraction, got {validation_size!r}.")
    if isinstance(validation_size, numbers.Integral):
        validation_rows = int(validation_size)
    elif isinstance(validation_size, numbers.Real) and 0 < float(validation_size) < 1:
        validation_rows = int(total_rows * float(validation_size) + 0.5)
    else:
        raise ValueError(
            f"validation_size must be an integer row count or a fraction in (0, 1), got {validation_size!r}."
        )

    if not 0 < validation_rows < total_rows:
        raise ValueError(
            f"validation_size resolves to {validation_rows} rows for a {total_rows}-row dataset; "
            "the validation split must be non-empty and smaller than the dataset."
        )
    return validation_rows


def _stable_seed(seed: int, *parts: str) -> int:
    payload = "\0".join((str(seed), *parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big")


def _difficulty_sort_key(value: str) -> tuple[float, str]:
    try:
        return float(value), value
    except ValueError:
        return math.inf, value


def _stratum_sort_key(key: tuple[str, str]) -> tuple[float, str, str]:
    numeric_difficulty, difficulty_text = _difficulty_sort_key(key[0])
    return numeric_difficulty, difficulty_text, key[1]


def allocate_validation_counts(
    stratum_sizes: dict[tuple[str, str], int],
    validation_rows: int,
) -> dict[tuple[str, str], int]:
    """Allocate an exact validation size while retaining one train row per stratum."""
    total_rows = sum(stratum_sizes.values())
    capacity = sum(max(size - 1, 0) for size in stratum_sizes.values())
    if validation_rows > capacity:
        raise ValueError(
            f"Cannot allocate {validation_rows} validation rows while retaining one training example in each "
            f"stratum; maximum supported validation size is {capacity}."
        )

    ideal_counts = {key: size * validation_rows / total_rows for key, size in stratum_sizes.items()}
    allocations = {key: min(math.floor(ideal_counts[key]), max(size - 1, 0)) for key, size in stratum_sizes.items()}
    remaining = validation_rows - sum(allocations.values())

    candidates: list[tuple[float, tuple[float, str, str], tuple[str, str]]] = []
    for key, size in stratum_sizes.items():
        if allocations[key] < size - 1:
            marginal_cost = 2 * (allocations[key] - ideal_counts[key]) + 1
            heapq.heappush(candidates, (marginal_cost, _stratum_sort_key(key), key))

    while remaining:
        if not candidates:
            raise RuntimeError("Internal error: exhausted stratified allocation capacity.")
        _, _, key = heapq.heappop(candidates)
        allocations[key] += 1
        remaining -= 1
        if allocations[key] < stratum_sizes[key] - 1:
            marginal_cost = 2 * (allocations[key] - ideal_counts[key]) + 1
            heapq.heappush(candidates, (marginal_cost, _stratum_sort_key(key), key))

    return allocations


def _build_distribution(
    stratum_sizes: dict[tuple[str, str], int],
    allocations: dict[tuple[str, str], int],
    field_index: int,
) -> list[dict[str, Any]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "train": 0, "validation": 0})
    for key, total in stratum_sizes.items():
        value = key[field_index]
        validation = allocations[key]
        counts[value]["total"] += total
        counts[value]["train"] += total - validation
        counts[value]["validation"] += validation

    if field_index == 0:
        values = sorted(counts, key=_difficulty_sort_key)
    else:
        values = sorted(counts)
    return [{"value": value, **counts[value]} for value in values]


def build_stratified_split(
    dataset: datasets.Dataset,
    validation_size: int | float = DEFAULT_VALIDATION_SIZE,
    seed: int = DEFAULT_SEED,
) -> tuple[datasets.Dataset, datasets.Dataset, dict[str, Any]]:
    """Split valid DeepMath rows by joint difficulty and broad topic strata."""
    validation_rows = resolve_validation_count(len(dataset), validation_size)
    stratum_indices: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, example in enumerate(dataset):
        key = stratification_key(example["difficulty"], example["topic"])
        stratum_indices[key].append(index)

    stratum_sizes = {key: len(indices) for key, indices in stratum_indices.items()}
    allocations = allocate_validation_counts(stratum_sizes, validation_rows)
    train_indices: list[int] = []
    validation_indices: list[int] = []

    for key in sorted(stratum_indices, key=_stratum_sort_key):
        indices = list(stratum_indices[key])
        rng = random.Random(_stable_seed(seed, *key))
        rng.shuffle(indices)
        stratum_validation_rows = allocations[key]
        validation_indices.extend(indices[:stratum_validation_rows])
        train_indices.extend(indices[stratum_validation_rows:])

    random.Random(_stable_seed(seed, "train")).shuffle(train_indices)
    random.Random(_stable_seed(seed, "validation")).shuffle(validation_indices)

    strata = []
    for key in sorted(stratum_sizes, key=_stratum_sort_key):
        total = stratum_sizes[key]
        validation = allocations[key]
        strata.append(
            {
                "difficulty": key[0],
                "topic_group": key[1],
                "total": total,
                "train": total - validation,
                "validation": validation,
            }
        )

    report = {
        "strategy": "joint_stratified_exact_allocation",
        "stratification_fields": list(STRATIFICATION_FIELDS),
        "seed": seed,
        "validation_size": validation_size,
        "total_rows": len(dataset),
        "train_rows": len(train_indices),
        "validation_rows": len(validation_indices),
        "validation_fraction": len(validation_indices) / len(dataset),
        "singleton_strata_kept_in_train": sum(size == 1 for size in stratum_sizes.values()),
        "strata": strata,
        "marginals": {
            "difficulty": _build_distribution(stratum_sizes, allocations, field_index=0),
            "topic_group": _build_distribution(stratum_sizes, allocations, field_index=1),
        },
    }
    return dataset.select(train_indices), dataset.select(validation_indices), report


def to_verl_dataset(
    dataset: datasets.Dataset,
    split: str,
    prompt_style: str = PROMPT_STYLE_BOXED,
) -> datasets.Dataset:
    """Convert validated source rows to verl's five-column RL schema."""

    def process_fn(example: dict[str, Any]) -> dict[str, Any]:
        source_index = int(example["source_index"])
        difficulty = float(example["difficulty"])
        topic = example["topic"].strip()
        key = stratification_key(difficulty, topic)
        return {
            "data_source": REWARD_DATA_SOURCE,
            "prompt": build_prompt(example["question"], prompt_style=prompt_style),
            "ability": "MATH",
            "reward_model": {
                "style": "rule",
                "ground_truth": example["final_answer"].strip(),
            },
            "extra_info": {
                "index": str(source_index),
                "source_index": source_index,
                "split": split,
                "source_dataset": DATA_SOURCE,
                "prompt_style": prompt_style,
                "difficulty": difficulty,
                "difficulty_bucket": key[0],
                "topic": topic,
                "topic_group": key[1],
                "stratification_key": format_stratification_key(key),
                "reward_data_source": REWARD_DATA_SOURCE,
            },
        }

    return dataset.map(
        function=process_fn,
        remove_columns=dataset.column_names,
        desc=f"Converting DeepMath-103K {split} split",
    )


def build_train_test_split(
    dataset_dict: datasets.DatasetDict,
    validation_size: int | float = DEFAULT_VALIDATION_SIZE,
    seed: int = DEFAULT_SEED,
    prompt_style: str = PROMPT_STYLE_BOXED,
    invalid_row_policy: str = "drop",
) -> tuple[datasets.Dataset, datasets.Dataset, dict[str, Any], list[dict[str, Any]]]:
    if "train" not in dataset_dict:
        raise ValueError(f"{DATA_SOURCE} must contain a train split; found splits: {list(dataset_dict.keys())}")

    valid_source, dropped_rows = validate_source_dataset(
        dataset_dict["train"],
        invalid_row_policy=invalid_row_policy,
    )
    train_source, validation_source, split_report = build_stratified_split(
        valid_source,
        validation_size=validation_size,
        seed=seed,
    )
    split_report["raw_rows"] = len(dataset_dict["train"])
    split_report["valid_rows"] = len(valid_source)
    split_report["dropped_rows"] = len(dropped_rows)

    return (
        to_verl_dataset(train_source, split="train", prompt_style=prompt_style),
        to_verl_dataset(validation_source, split="test", prompt_style=prompt_style),
        split_report,
        dropped_rows,
    )


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)
        file.write("\n")


def write_artifacts(
    local_save_dir: Path,
    train_dataset: datasets.Dataset,
    test_dataset: datasets.Dataset,
    split_report: dict[str, Any],
    dropped_rows: list[dict[str, Any]],
    local_dataset_path: str,
    revision: str,
    prompt_style: str,
    invalid_row_policy: str,
) -> None:
    _write_json(local_save_dir / "train_example.json", train_dataset[0])
    _write_json(local_save_dir / "test_example.json", test_dataset[0])
    _write_json(local_save_dir / "split_report.json", split_report)
    _write_json(local_save_dir / "dropped_rows.json", dropped_rows)
    _write_json(
        local_save_dir / "dataset_info.json",
        {
            "source": DATA_SOURCE,
            "requested_revision": revision,
            "local_dataset_path": str(_expand_path(local_dataset_path)),
            "prompt_style": prompt_style,
            "invalid_row_policy": invalid_row_policy,
            "train_rows": len(train_dataset),
            "test_rows": len(test_dataset),
            "schema": list(train_dataset.features.keys()),
            "split_report": "split_report.json",
            "dropped_rows_report": "dropped_rows.json",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert DeepMath-103K to verl parquet with a deterministic stratified validation split."
    )
    parser.add_argument("--local_dir", default=None, help="Deprecated. Use --local_save_dir instead.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--local_dataset_path",
        default=DEFAULT_LOCAL_DATASET_PATH,
        help="Path to raw DeepMath-103K parquet shards or a dataset saved with save_to_disk.",
    )
    parser.add_argument(
        "--local_save_dir",
        default=DEFAULT_LOCAL_SAVE_DIR,
        help="Directory for train.parquet, test.parquet, and split audit reports.",
    )
    parser.add_argument(
        "--validation_size",
        type=parse_validation_size,
        default=DEFAULT_VALIDATION_SIZE,
        help="Held-out fraction in (0, 1), or an exact integer number of validation rows. Default: 0.05.",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="Seed for deterministic within-stratum sampling."
    )
    parser.add_argument(
        "--prompt_style",
        choices=(PROMPT_STYLE_BOXED, PROMPT_STYLE_RAW),
        default=PROMPT_STYLE_BOXED,
        help="'boxed' emits one user message with the boxed-answer instruction; 'raw' emits the question unchanged.",
    )
    parser.add_argument(
        "--invalid_row_policy",
        choices=INVALID_ROW_POLICIES,
        default="drop",
        help="Drop and audit malformed rows, or stop with an error. The source revision has two empty answers.",
    )
    parser.add_argument(
        "--revision",
        default=DATASET_REVISION,
        help="Hugging Face revision used with --download (pinned to the revision audited by this script).",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Explicitly download from Hugging Face if --local_dataset_path does not exist.",
    )
    args = parser.parse_args()

    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: --local_dir is deprecated. Use --local_save_dir instead.", flush=True)
    else:
        local_save_dir = args.local_save_dir
    local_save_path = _expand_path(local_save_dir)
    local_save_path.mkdir(parents=True, exist_ok=True)

    dataset = load_deepmath_dataset(
        args.local_dataset_path,
        download=args.download,
        revision=args.revision,
    )
    train_dataset, test_dataset, split_report, dropped_rows = build_train_test_split(
        dataset,
        validation_size=args.validation_size,
        seed=args.seed,
        prompt_style=args.prompt_style,
        invalid_row_policy=args.invalid_row_policy,
    )

    train_path = local_save_path / "train.parquet"
    test_path = local_save_path / "test.parquet"
    train_dataset.to_parquet(str(train_path))
    test_dataset.to_parquet(str(test_path))
    write_artifacts(
        local_save_path,
        train_dataset,
        test_dataset,
        split_report,
        dropped_rows,
        args.local_dataset_path,
        args.revision,
        args.prompt_style,
        args.invalid_row_policy,
    )

    print(
        f"Saved DeepMath-103K train split to {train_path} ({len(train_dataset)} rows)",
        flush=True,
    )
    print(
        f"Saved stratified validation split to {test_path} ({len(test_dataset)} rows)",
        flush=True,
    )
    print(
        f"Dropped {len(dropped_rows)} invalid source rows; audit reports are in {local_save_path}",
        flush=True,
    )

    if args.hdfs_dir is not None:
        from verl.utils.hdfs_io import copy, makedirs

        makedirs(args.hdfs_dir)
        copy(src=str(local_save_path), dst=args.hdfs_dir)


if __name__ == "__main__":
    main()
