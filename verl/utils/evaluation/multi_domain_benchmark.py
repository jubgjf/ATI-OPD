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

"""Local, deterministic adapters for the Nemotron multi-domain benchmark suite."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from verl.utils.evaluation.math_benchmark import build_math_chat_messages, load_benchmark_samples
from verl.utils.tokenizer import normalize_token_ids

SUPPORTED_ADAPTERS = frozenset({"aime", "ifbench", "mmlu_pro", "gpqa_diamond", "longbench_v2"})
BENCHMARK_SAMPLING_PROFILES = ("legacy", "train_sampling")
CHOICE_LABELS = tuple("ABCDEFGHIJ")
LONG_BENCH_METRIC_NAME = "longbench-v2-32k-middle-truncated"


@dataclass(frozen=True)
class BenchmarkProfile:
    n: int


@dataclass(frozen=True)
class BenchmarkSpec:
    """Validated public configuration for one benchmark entry."""

    name: str
    adapter: str
    path: str
    batch_size: int
    max_samples: int
    prompt_max_length: int
    generation_max_tokens: int
    profiles: Mapping[str, BenchmarkProfile]
    adapter_kwargs: Mapping[str, Any] = field(default_factory=dict)

    @property
    def metric_name(self) -> str:
        return LONG_BENCH_METRIC_NAME if self.adapter == "longbench_v2" else self.name


@dataclass(frozen=True)
class BenchmarkRecord:
    uid: str
    messages: list[dict[str, str]]
    ground_truth: str
    payload: dict[str, Any]


def _as_plain_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "items"):
        return {str(key): item for key, item in value.items()}
    raise TypeError(f"{field_name} must be a mapping, got {type(value).__name__}.")


def _normalize_relative_path(value: Any, *, field_name: str) -> str:
    path = Path(str(value).strip())
    if not str(value).strip() or path == Path("."):
        raise ValueError(f"{field_name} must be a non-empty relative path.")
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field_name} must stay below data.test_benchmarks.root, got {value!r}.")
    return path.as_posix()


def parse_benchmark_specs(entries: Sequence[Any]) -> tuple[BenchmarkSpec, ...]:
    """Parse and validate ``data.test_benchmarks.entries``."""
    specs: list[BenchmarkSpec] = []
    for index, raw_entry in enumerate(entries):
        entry = _as_plain_mapping(raw_entry, field_name=f"test_benchmarks.entries[{index}]")
        name = str(entry.get("name", "")).strip()
        adapter = str(entry.get("adapter", "")).strip()
        if not name:
            raise ValueError(f"test_benchmarks.entries[{index}].name must be non-empty.")
        if adapter not in SUPPORTED_ADAPTERS:
            raise ValueError(
                f"test_benchmarks.entries[{index}].adapter must be one of {sorted(SUPPORTED_ADAPTERS)}, "
                f"got {adapter!r}."
            )
        path = _normalize_relative_path(entry.get("path"), field_name=f"test_benchmarks.entries[{index}].path")
        batch_size = int(entry.get("batch_size", 1))
        max_samples = int(entry.get("max_samples", -1))
        prompt_max_length = int(entry.get("prompt_max_length", 0))
        generation_max_tokens = int(entry.get("generation_max_tokens", 0))
        if batch_size <= 0:
            raise ValueError(f"Benchmark {name!r} batch_size must be positive, got {batch_size}.")
        if max_samples == 0 or max_samples < -1:
            raise ValueError(f"Benchmark {name!r} max_samples must be -1 or positive, got {max_samples}.")
        if prompt_max_length <= 0 or generation_max_tokens <= 0:
            raise ValueError(f"Benchmark {name!r} prompt_max_length and generation_max_tokens must both be positive.")

        raw_profiles = _as_plain_mapping(entry.get("profiles"), field_name=f"benchmark {name!r} profiles")
        missing = [profile for profile in BENCHMARK_SAMPLING_PROFILES if profile not in raw_profiles]
        extra = sorted(set(raw_profiles) - set(BENCHMARK_SAMPLING_PROFILES))
        if missing or extra:
            raise ValueError(
                f"Benchmark {name!r} profiles must be exactly {BENCHMARK_SAMPLING_PROFILES}; "
                f"missing={missing}, extra={extra}."
            )
        profiles: dict[str, BenchmarkProfile] = {}
        for profile_name in BENCHMARK_SAMPLING_PROFILES:
            raw_profile = _as_plain_mapping(
                raw_profiles[profile_name], field_name=f"benchmark {name!r} profile {profile_name!r}"
            )
            n = int(raw_profile.get("n", 0))
            if n <= 0:
                raise ValueError(f"Benchmark {name!r} profile {profile_name!r} n must be positive, got {n}.")
            profiles[profile_name] = BenchmarkProfile(n=n)

        specs.append(
            BenchmarkSpec(
                name=name,
                adapter=adapter,
                path=path,
                batch_size=batch_size,
                max_samples=max_samples,
                prompt_max_length=prompt_max_length,
                generation_max_tokens=generation_max_tokens,
                profiles=profiles,
                adapter_kwargs=_as_plain_mapping(
                    entry.get("adapter_kwargs"), field_name=f"benchmark {name!r} adapter_kwargs"
                ),
            )
        )

    duplicate_names = sorted({spec.name for spec in specs if sum(item.name == spec.name for item in specs) > 1})
    if duplicate_names:
        raise ValueError(f"Benchmark entry names must be unique, got duplicates {duplicate_names}.")
    return tuple(specs)


def stable_int_seed(*parts: Any) -> int:
    digest = hashlib.sha256("\x1f".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big", signed=False)


def _read_parquet_rows(directory: Path, *, split: str | None = None) -> list[dict[str, Any]]:
    if not directory.exists():
        raise FileNotFoundError(f"Benchmark path does not exist: {directory}")
    candidates = sorted(directory.rglob("*.parquet"))
    if split:
        split_candidates = [path for path in candidates if split.casefold() in path.as_posix().casefold()]
        if not split_candidates:
            raise FileNotFoundError(f"No parquet files found under {directory} for required split={split!r}.")
        candidates = split_candidates
    if not candidates:
        raise FileNotFoundError(f"No parquet files found under {directory} for split={split!r}.")
    rows: list[dict[str, Any]] = []
    for path in candidates:
        rows.extend(pd.read_parquet(path).to_dict(orient="records"))
    return rows


def _parse_list(value: Any, *, field_name: str) -> list[Any]:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple | list):
        return list(value)
    if isinstance(value, str):
        stripped = value.strip()
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(stripped)
            except (SyntaxError, ValueError) as exc:
                raise ValueError(f"Could not parse {field_name} as a list: {value[:200]!r}") from exc
        if isinstance(parsed, list):
            return parsed
    raise TypeError(f"{field_name} must be a list or serialized list, got {type(value).__name__}.")


def _json_native(value: Any) -> Any:
    """Convert Arrow/numpy containers into deterministic JSON-compatible values."""
    if isinstance(value, np.ndarray):
        return [_json_native(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_native(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_native(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value):
            return None
        # Arrow promotes nullable integer fields in IFBench's union-like
        # kwargs struct to floats. Restore their JSON integer representation;
        # several official verifiers use these values in range() or slices.
        if math.isfinite(value) and value.is_integer():
            return int(value)
    return value


def _answer_label(value: Any, *, option_count: int) -> str:
    if isinstance(value, bool):
        raise ValueError(f"Multiple-choice answer cannot be boolean: {value!r}.")
    if isinstance(value, int | np.integer):
        index = int(value)
        if 0 <= index < option_count:
            return CHOICE_LABELS[index]
    text = str(value).strip().upper()
    if text.isdigit() and 0 <= int(text) < option_count:
        return CHOICE_LABELS[int(text)]
    match = re.fullmatch(r"\(?([A-J])\)?", text)
    if match and CHOICE_LABELS.index(match.group(1)) < option_count:
        return match.group(1)
    raise ValueError(f"Invalid multiple-choice answer {value!r} for {option_count} options.")


def _choice_prompt(question: str, choices: Sequence[str], *, prefix: str = "") -> str:
    rendered = "\n".join(f"{CHOICE_LABELS[index]}. {choice}" for index, choice in enumerate(choices))
    parts = [part for part in (prefix.strip(), str(question).strip(), rendered) if part]
    return "\n\n".join(parts)


def _largest_remainder_allocations(counts: Mapping[str, int], total: int) -> dict[str, int]:
    if total <= 0 or total > sum(counts.values()):
        raise ValueError(f"Invalid stratified sample total {total} for {sum(counts.values())} rows.")
    exact = {key: total * count / sum(counts.values()) for key, count in counts.items()}
    result = {key: min(counts[key], math.floor(value)) for key, value in exact.items()}
    remaining = total - sum(result.values())
    order = sorted(counts, key=lambda key: (-(exact[key] - result[key]), key))
    for key in order:
        if remaining <= 0:
            break
        if result[key] < counts[key]:
            result[key] += 1
            remaining -= 1
    if remaining:
        raise RuntimeError(f"Could not allocate {total} stratified samples across {counts}.")
    return result


def stratified_sample_rows(
    rows: Sequence[dict[str, Any]], *, category_field: str, max_samples: int, seed: int
) -> list[dict[str, Any]]:
    """Select an exact, deterministic proportional sample and preserve source order."""
    if max_samples < 0 or max_samples >= len(rows):
        return list(rows)
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        category = str(row.get(category_field, "")).strip()
        if not category:
            raise ValueError(f"Row {index} is missing non-empty category field {category_field!r}.")
        grouped[category].append((index, row))
    allocations = _largest_remainder_allocations({key: len(value) for key, value in grouped.items()}, max_samples)
    selected: list[tuple[int, dict[str, Any]]] = []
    for category, values in grouped.items():
        shuffled = list(values)
        random.Random(stable_int_seed(seed, category)).shuffle(shuffled)
        selected.extend(shuffled[: allocations[category]])
    selected.sort(key=lambda item: item[0])
    return [row for _, row in selected]


def _chat_token_ids(tokenizer: Any, messages: list[dict[str, str]], kwargs: Mapping[str, Any]) -> list[int]:
    tokenized = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        **dict(kwargs),
    )
    return normalize_token_ids(tokenized)


def _validate_prompt_budget(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    max_length: int,
    chat_template_kwargs: Mapping[str, Any],
    benchmark: str,
) -> int:
    length = len(_chat_token_ids(tokenizer, messages, chat_template_kwargs))
    if length > max_length:
        raise ValueError(
            f"Formatted prompt for benchmark {benchmark!r} has {length} tokens, exceeding prompt_max_length="
            f"{max_length}. Only LongBench-v2 permits deterministic context truncation."
        )
    return length


def _load_aime_records(root: Path, spec: BenchmarkSpec) -> list[BenchmarkRecord]:
    max_samples = None if spec.max_samples < 0 else spec.max_samples
    samples = load_benchmark_samples(root, spec.path, max_samples=max_samples)
    return [
        BenchmarkRecord(
            uid=f"{spec.name}:{sample.sample_id}",
            messages=build_math_chat_messages(sample.problem, idx=index, use_raw_prompt=sample.use_raw_prompt),
            ground_truth=sample.ground_truth,
            payload={"task_type": "aime", "expected_answer": sample.ground_truth},
        )
        for index, sample in enumerate(samples)
    ]


def _load_ifbench_records(directory: Path, spec: BenchmarkSpec) -> list[BenchmarkRecord]:
    rows = _read_parquet_rows(directory)
    if spec.max_samples > 0:
        rows = rows[: spec.max_samples]
    records = []
    for index, row in enumerate(rows):
        prompt = str(row.get("prompt", "")).strip()
        instruction_ids = [
            str(value) for value in _parse_list(row.get("instruction_id_list"), field_name="instruction_id_list")
        ]
        raw_kwargs = _parse_list(row.get("kwargs"), field_name="kwargs")
        kwargs = []
        for constraint_index, raw_constraint_kwargs in enumerate(raw_kwargs):
            if not isinstance(raw_constraint_kwargs, Mapping):
                raise TypeError(
                    f"Invalid IFBench row {index}: kwargs[{constraint_index}] must be a mapping, "
                    f"got {type(raw_constraint_kwargs).__name__}."
                )
            # Arrow expands this dataset's struct union into dozens of null-valued
            # keys. Drop those fields before embedding the verifier payload.
            kwargs.append(
                {
                    str(key): native_value
                    for key, value in raw_constraint_kwargs.items()
                    if (native_value := _json_native(value)) is not None
                }
            )
        if not prompt or not instruction_ids or len(instruction_ids) != len(kwargs):
            raise ValueError(f"Invalid IFBench row {index}: prompt/constraint metadata are not aligned.")
        key = str(row.get("key", index))
        records.append(
            BenchmarkRecord(
                uid=f"{spec.name}:{key}",
                messages=[{"role": "user", "content": prompt}],
                ground_truth="",
                payload={
                    "task_type": "ifbench",
                    "prompt": prompt,
                    "instruction_id_list": instruction_ids,
                    "kwargs": kwargs,
                },
            )
        )
    return records


def _mmlu_example_text(row: Mapping[str, Any], *, include_answer: bool) -> str:
    question = str(row.get("question", "")).strip()
    options = [str(value).strip() for value in _parse_list(row.get("options"), field_name="MMLU-Pro options")]
    if not question or not 2 <= len(options) <= len(CHOICE_LABELS):
        raise ValueError("MMLU-Pro rows require a question and 2-10 options.")
    prompt = _choice_prompt(question, options)
    if include_answer:
        answer = _answer_label(row.get("answer", row.get("answer_index")), option_count=len(options))
        cot = str(row.get("cot_content", "")).strip()
        answer_text = cot if cot else f"The correct answer is ({answer})."
        if f"({answer})" not in answer_text and f" {answer}" not in answer_text:
            answer_text = f"{answer_text}\nThe correct answer is ({answer})."
        prompt = f"{prompt}\n\nAnswer: Let's think step by step. {answer_text}"
    else:
        prompt = (
            f"{prompt}\n\nAnswer: Let's think step by step. Finish with "
            '"The correct answer is (X)", where X is the option letter.'
        )
    return prompt


def _load_mmlu_records(directory: Path, spec: BenchmarkSpec, *, seed: int) -> list[BenchmarkRecord]:
    validation_rows = _read_parquet_rows(directory, split="validation")
    test_rows = _read_parquet_rows(directory, split="test")
    category_field = "category"
    categories = {str(row.get(category_field, "")).strip() for row in test_rows}
    categories.discard("")
    expected_category_count = int(spec.adapter_kwargs.get("expected_category_count", 14))
    if len(categories) != expected_category_count:
        raise ValueError(
            f"MMLU-Pro expected {expected_category_count} test categories, found {len(categories)}: "
            f"{sorted(categories)}."
        )
    test_rows = stratified_sample_rows(
        test_rows,
        category_field=category_field,
        max_samples=spec.max_samples,
        seed=seed,
    )
    shots: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in validation_rows:
        category = str(row.get(category_field, "")).strip()
        if category and len(shots[category]) < int(spec.adapter_kwargs.get("num_few_shot", 5)):
            shots[category].append(row)

    records = []
    for index, row in enumerate(test_rows):
        category = str(row.get(category_field, "")).strip()
        if not category or len(shots[category]) < int(spec.adapter_kwargs.get("num_few_shot", 5)):
            raise ValueError(f"MMLU-Pro category {category!r} does not have the configured validation demonstrations.")
        options = [str(value).strip() for value in _parse_list(row.get("options"), field_name="MMLU-Pro options")]
        answer = _answer_label(row.get("answer", row.get("answer_index")), option_count=len(options))
        intro = f"The following are multiple choice questions (with answers) about {category}."
        demonstrations = "\n\n".join(_mmlu_example_text(item, include_answer=True) for item in shots[category])
        prompt = f"{intro}\n\n{demonstrations}\n\n{_mmlu_example_text(row, include_answer=False)}"
        sample_id = str(row.get("question_id", row.get("id", index)))
        records.append(
            BenchmarkRecord(
                uid=f"{spec.name}:{sample_id}",
                messages=[{"role": "user", "content": prompt}],
                ground_truth=answer,
                payload={
                    "task_type": "multiple_choice",
                    "benchmark": "mmlu_pro",
                    "expected_answer": answer,
                    "valid_labels": CHOICE_LABELS[: len(options)],
                    "category": category,
                },
            )
        )
    return records


def _load_gpqa_records(directory: Path, spec: BenchmarkSpec, *, seed: int) -> list[BenchmarkRecord]:
    csv_candidates = sorted(directory.rglob("gpqa_diamond.csv"))
    if not csv_candidates:
        raise FileNotFoundError(f"No gpqa_diamond.csv found under {directory}.")
    rows = pd.read_csv(csv_candidates[0]).to_dict(orient="records")
    if spec.max_samples > 0:
        rows = rows[: spec.max_samples]
    records = []
    for index, row in enumerate(rows):
        sample_id = str(row.get("Record ID", index))
        question = str(row.get("Question", "")).strip()
        answer_pairs = [
            (str(row.get("Correct Answer", "")).strip(), True),
            (str(row.get("Incorrect Answer 1", "")).strip(), False),
            (str(row.get("Incorrect Answer 2", "")).strip(), False),
            (str(row.get("Incorrect Answer 3", "")).strip(), False),
        ]
        if not question or any(not answer for answer, _ in answer_pairs):
            raise ValueError(f"Invalid GPQA-Diamond row {index}.")
        random.Random(stable_int_seed(seed, sample_id)).shuffle(answer_pairs)
        choices = [answer for answer, _ in answer_pairs]
        answer = CHOICE_LABELS[next(i for i, (_, correct) in enumerate(answer_pairs) if correct)]
        prompt = _choice_prompt(
            question,
            choices,
            prefix=(
                "Choose the best answer. Think step by step, then finish with exactly "
                '"The correct answer is (X)", where X is the option letter.'
            ),
        )
        records.append(
            BenchmarkRecord(
                uid=f"{spec.name}:{sample_id}",
                messages=[{"role": "user", "content": prompt}],
                ground_truth=answer,
                payload={
                    "task_type": "multiple_choice",
                    "benchmark": "gpqa_diamond",
                    "expected_answer": answer,
                    "valid_labels": CHOICE_LABELS[:4],
                    "category": str(row.get("High-level domain", "unknown")),
                },
            )
        )
    return records


def _longbench_prompt(row: Mapping[str, Any], context: str) -> str:
    return (
        "Please read the following text and answer the question below.\n\n"
        f"<text>\n{context.strip()}\n</text>\n\n"
        f"What is the correct answer to this question: {str(row['question']).strip()}\n"
        "Choices:\n"
        f"(A) {str(row['choice_A']).strip()}\n"
        f"(B) {str(row['choice_B']).strip()}\n"
        f"(C) {str(row['choice_C']).strip()}\n"
        f"(D) {str(row['choice_D']).strip()}\n\n"
        'Let\'s think step by step. Finish with "The correct answer is (X)", where X is the option letter.'
    )


def middle_truncate_longbench_context(
    row: Mapping[str, Any],
    *,
    tokenizer: Any,
    prompt_max_length: int,
    chat_template_kwargs: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Truncate only the document while retaining both ends and fixed question/options."""
    context = str(row.get("context", "")).strip()
    if not context:
        raise ValueError("LongBench-v2 row has an empty context.")
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    empty_messages = [{"role": "user", "content": _longbench_prompt(row, "")}]
    fixed_tokens = len(_chat_token_ids(tokenizer, empty_messages, chat_template_kwargs))
    # Every released LongBench-v2 document is large. Avoid tokenizing its entire
    # 445 MB corpus twice: context tokenization is exact, while the additive
    # full-prompt estimate can differ only at the two insertion boundaries.
    estimated_full_prompt_tokens = len(context_ids) + fixed_tokens
    full_prompt_tokens_exact = estimated_full_prompt_tokens <= prompt_max_length + 8
    if full_prompt_tokens_exact:
        full_messages = [{"role": "user", "content": _longbench_prompt(row, context)}]
        full_prompt_tokens = len(_chat_token_ids(tokenizer, full_messages, chat_template_kwargs))
    else:
        full_prompt_tokens = estimated_full_prompt_tokens
    if full_prompt_tokens <= prompt_max_length:
        return context, {
            "original_prompt_tokens": full_prompt_tokens,
            "original_prompt_tokens_exact": True,
            "formatted_prompt_tokens": full_prompt_tokens,
            "original_context_tokens": len(context_ids),
            "retained_context_tokens": len(context_ids),
            "retained_ratio": 1.0,
            "truncated": False,
        }

    marker = "\n\n... [middle of context truncated for the 32K evaluation protocol] ...\n\n"
    marker_ids = tokenizer.encode(marker, add_special_tokens=False)
    keep = prompt_max_length - fixed_tokens - len(marker_ids)
    if keep < 2:
        raise ValueError(
            f"LongBench-v2 fixed question/options require {fixed_tokens} tokens, leaving no context budget under "
            f"prompt_max_length={prompt_max_length}."
        )

    while keep >= 2:
        first = keep // 2
        last = keep - first
        retained_ids = list(context_ids[:first]) + list(marker_ids) + list(context_ids[-last:])
        truncated_context = tokenizer.decode(retained_ids, skip_special_tokens=True)
        messages = [{"role": "user", "content": _longbench_prompt(row, truncated_context)}]
        actual_tokens = len(_chat_token_ids(tokenizer, messages, chat_template_kwargs))
        if actual_tokens <= prompt_max_length:
            retained = first + last
            return truncated_context, {
                "original_prompt_tokens": full_prompt_tokens,
                "original_prompt_tokens_exact": full_prompt_tokens_exact,
                "formatted_prompt_tokens": actual_tokens,
                "original_context_tokens": len(context_ids),
                "retained_context_tokens": retained,
                "retained_ratio": retained / len(context_ids),
                "truncated": True,
            }
        keep -= max(1, actual_tokens - prompt_max_length)
    raise RuntimeError("Could not fit LongBench-v2 context into the configured prompt budget.")


def _load_longbench_records(
    directory: Path,
    spec: BenchmarkSpec,
    *,
    tokenizer: Any,
    chat_template_kwargs: Mapping[str, Any],
) -> list[BenchmarkRecord]:
    data_path = directory / "data.json"
    if not data_path.is_file():
        candidates = sorted(directory.rglob("data.json"))
        if not candidates:
            raise FileNotFoundError(f"No LongBench-v2 data.json found under {directory}.")
        data_path = candidates[0]
    with data_path.open(encoding="utf-8") as file:
        rows = json.load(file)
    if not isinstance(rows, list):
        raise ValueError(f"LongBench-v2 data must be a JSON list, got {type(rows).__name__}.")
    if spec.max_samples > 0:
        rows = rows[: spec.max_samples]
    records = []
    for index, row in enumerate(rows):
        context, length_info = middle_truncate_longbench_context(
            row,
            tokenizer=tokenizer,
            prompt_max_length=spec.prompt_max_length,
            chat_template_kwargs=chat_template_kwargs,
        )
        answer = _answer_label(row.get("answer"), option_count=4)
        sample_id = str(row.get("_id", index))
        payload = {
            "task_type": "multiple_choice",
            "benchmark": "longbench_v2",
            "expected_answer": answer,
            "valid_labels": CHOICE_LABELS[:4],
            "category": str(row.get("domain", "unknown")),
            "length_bucket": str(row.get("length", "unknown")),
            **length_info,
        }
        records.append(
            BenchmarkRecord(
                uid=f"{spec.name}:{sample_id}",
                messages=[{"role": "user", "content": _longbench_prompt(row, context)}],
                ground_truth=answer,
                payload=payload,
            )
        )
    return records


def load_benchmark_records(
    *,
    dataset_root: str | Path,
    spec: BenchmarkSpec,
    tokenizer: Any,
    seed: int,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> list[BenchmarkRecord]:
    root = Path(dataset_root).expanduser()
    directory = root / spec.path
    template_kwargs = dict(chat_template_kwargs or {})
    if spec.adapter == "aime":
        records = _load_aime_records(root, spec)
    elif spec.adapter == "ifbench":
        records = _load_ifbench_records(directory, spec)
    elif spec.adapter == "mmlu_pro":
        records = _load_mmlu_records(directory, spec, seed=seed)
    elif spec.adapter == "gpqa_diamond":
        records = _load_gpqa_records(directory, spec, seed=seed)
    elif spec.adapter == "longbench_v2":
        records = _load_longbench_records(
            directory,
            spec,
            tokenizer=tokenizer,
            chat_template_kwargs=template_kwargs,
        )
    else:  # pragma: no cover - parse_benchmark_specs validates this.
        raise AssertionError(f"Unhandled benchmark adapter {spec.adapter!r}.")
    if not records:
        raise ValueError(f"Benchmark {spec.name!r} loaded no records from {directory}.")
    if spec.adapter != "longbench_v2":
        for record in records:
            _validate_prompt_budget(
                tokenizer,
                record.messages,
                max_length=spec.prompt_max_length,
                chat_template_kwargs=template_kwargs,
                benchmark=spec.name,
            )
    return records


def aggregate_benchmark_details(
    *,
    adapter: str,
    metric_name: str,
    metric_prefix: str,
    sample_uids: Sequence[str],
    reward_extra_infos: Mapping[str, Sequence[Any]],
) -> dict[str, float]:
    """Compute adapter-level metrics that cannot be expressed by generic mean@k aggregation."""
    acc_values = reward_extra_infos.get("acc", reward_extra_infos.get("reward", ()))
    if not acc_values or len(acc_values) != len(sample_uids):
        return {}
    per_uid: dict[str, list[int]] = defaultdict(list)
    uid_category: dict[str, str] = {}
    uid_length: dict[str, str] = {}
    categories = reward_extra_infos.get("category", ())
    length_buckets = reward_extra_infos.get("length_bucket", ())
    for index, uid in enumerate(sample_uids):
        per_uid[str(uid)].append(index)
        if index < len(categories) and categories[index] is not None:
            uid_category[str(uid)] = str(categories[index])
        if index < len(length_buckets) and length_buckets[index] is not None:
            uid_length[str(uid)] = str(length_buckets[index])

    uid_acc = {uid: float(np.mean([float(acc_values[index]) for index in indices])) for uid, indices in per_uid.items()}
    metrics: dict[str, float] = {}
    if adapter == "mmlu_pro":
        grouped: dict[str, list[float]] = defaultdict(list)
        for uid, value in uid_acc.items():
            grouped[uid_category.get(uid, "unknown")].append(value)
        category_means = {}
        for category, values in sorted(grouped.items()):
            slug = re.sub(r"[^a-z0-9]+", "-", category.casefold()).strip("-") or "unknown"
            category_means[category] = float(np.mean(values))
            metrics[f"{metric_prefix}-aux/{metric_name}/category/{slug}/acc"] = category_means[category]
        metrics[f"{metric_prefix}-aux/{metric_name}/category-macro/acc"] = float(np.mean(list(category_means.values())))
        metrics[f"{metric_prefix}-core/{metric_name}/overall-micro/acc"] = float(np.mean(list(uid_acc.values())))
    elif adapter == "longbench_v2":
        grouped = defaultdict(list)
        for uid, value in uid_acc.items():
            grouped[uid_length.get(uid, "unknown")].append(value)
        for bucket, values in sorted(grouped.items()):
            slug = re.sub(r"[^a-z0-9]+", "-", bucket.casefold()).strip("-") or "unknown"
            metrics[f"{metric_prefix}-aux/{metric_name}/length/{slug}/acc"] = float(np.mean(values))
        for source_key, metric_key in (
            ("truncated", "truncation-rate"),
            ("retained_ratio", "retained-ratio"),
            ("original_prompt_tokens", "original-prompt-tokens"),
            ("original_context_tokens", "original-context-tokens"),
            ("retained_context_tokens", "retained-context-tokens"),
        ):
            values = reward_extra_infos.get(source_key, ())
            if values:
                metrics[f"{metric_prefix}-aux/{metric_name}/{metric_key}/mean"] = float(
                    np.mean([float(value) for value in values])
                )
    return metrics


def expected_generation_count(specs: Iterable[BenchmarkSpec]) -> int:
    return sum(
        spec.profiles[profile].n * (spec.max_samples if spec.max_samples > 0 else 0)
        for spec in specs
        for profile in BENCHMARK_SAMPLING_PROFILES
    )


__all__ = [
    "BENCHMARK_SAMPLING_PROFILES",
    "BenchmarkProfile",
    "BenchmarkRecord",
    "BenchmarkSpec",
    "LONG_BENCH_METRIC_NAME",
    "aggregate_benchmark_details",
    "load_benchmark_records",
    "middle_truncate_longbench_context",
    "parse_benchmark_specs",
    "stable_int_seed",
    "stratified_sample_rows",
]
