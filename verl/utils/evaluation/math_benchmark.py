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

"""Reusable math dataset and response-scoring helpers."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from verl.utils.reward_score import default_compute_score, math_dapo, math_verify

try:
    from examples.data_preprocess.dapo_math_17k import build_qwen3_boxed_prompt
except Exception:  # pragma: no cover - defensive fallback for stripped installs.
    QWEN3_MATH_BOXED_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."

    def build_qwen3_boxed_prompt(prompt: Any, idx: int) -> list[dict[str, str]]:
        del idx
        return [{"role": "user", "content": f"{str(prompt).strip()}\n\n{QWEN3_MATH_BOXED_INSTRUCTION}"}]


PROMPT_FIELDS = ("problem", "question", "prompt", "input")
ANSWER_FIELDS = ("answer", "solution", "ground_truth", "final_answer")
THINK_END_TAGS = ("</think>", "<|end_of_thought|>")
AMO_BENCH_P_SIZE = 39
AMO_BENCH_P_ANSWER_TYPES = frozenset({"number", "set", "variable"})


@dataclass(frozen=True)
class EvalSample:
    benchmark: str
    sample_id: str
    problem: str
    ground_truth: str
    question_id: int | str | None = None
    answer_type: str | None = None
    grader: str | None = None
    use_raw_prompt: bool = False


@dataclass(frozen=True)
class ScoredCompletion:
    text: str
    final_content: str
    reasoning_content: str | None
    content: str | None
    finish_reason: str | None
    response_token_count: int | None
    acc: bool
    pred: str
    invalid_pred: bool
    has_answer_pattern: bool
    has_boxed_pattern: bool
    has_accepted_answer_format: bool
    verified_by: str
    math_verify_score: float | None


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _normalize_numeric_answer(value: Any) -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return str(int(value))
        return str(value)
    text = str(value).strip()
    if re.fullmatch(r"-?\d+\.0+", text):
        return text.split(".", 1)[0]
    return None


def normalize_ground_truth(value: Any) -> str:
    numeric = _normalize_numeric_answer(value)
    if numeric is not None:
        return numeric
    text = str(value).strip()
    boxed = math_dapo.last_boxed_only_string(text)
    return math_dapo.remove_boxed(boxed).strip() if boxed is not None else text


def _extract_prompt(row: dict[str, Any], idx: int) -> str:
    for field_name in PROMPT_FIELDS:
        value = row.get(field_name)
        if not _is_missing(value) and str(value).strip():
            return str(value).strip()
    raise ValueError(f"Benchmark row {idx} must contain one of prompt fields: {', '.join(PROMPT_FIELDS)}")


def _extract_answer(row: dict[str, Any], idx: int) -> str:
    reward_model = row.get("reward_model")
    if isinstance(reward_model, str) and reward_model.strip().startswith("{"):
        try:
            reward_model = json.loads(reward_model)
        except json.JSONDecodeError:
            pass
    if isinstance(reward_model, dict) and not _is_missing(reward_model.get("ground_truth")):
        return normalize_ground_truth(reward_model["ground_truth"])

    for field_name in ANSWER_FIELDS:
        value = row.get(field_name)
        if not _is_missing(value) and str(value).strip():
            return normalize_ground_truth(value)
    raise ValueError(f"Benchmark row {idx} must contain one of answer fields: {', '.join(ANSWER_FIELDS)}")


def _row_to_sample(row: dict[str, Any], benchmark: str, idx: int, *, is_amo_bench: bool) -> EvalSample:
    question_id = row.get("question_id")
    if _is_missing(question_id):
        question_id = None
    sample_id = row.get("id", question_id if question_id is not None else idx)
    if _is_missing(sample_id):
        sample_id = idx
    answer_type = row.get("answer_type")
    if _is_missing(answer_type):
        answer_type = None
    elif answer_type is not None:
        answer_type = str(answer_type).strip().lower()
    return EvalSample(
        benchmark=benchmark,
        sample_id=str(sample_id),
        problem=_extract_prompt(row, idx),
        ground_truth=_extract_answer(row, idx),
        question_id=question_id,
        answer_type=answer_type,
        grader="amo_bench" if is_amo_bench else None,
        use_raw_prompt=is_amo_bench,
    )


def _load_jsonl_dataset(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_no}: {stripped[:200]!r}") from exc
    return rows


def is_amo_bench(benchmark: str | Path) -> bool:
    """Return whether a benchmark path identifies the official AMO-Bench dataset."""
    normalized_name = Path(str(benchmark).rstrip("/")).name.casefold().replace("_", "-")
    return normalized_name == "amo-bench"


def find_benchmark_data_files(dataset_root: str | Path, benchmark: str) -> list[Path]:
    """Find local benchmark files, including Hugging Face's common ``data/`` layout."""
    benchmark_dir = Path(dataset_root).expanduser() / benchmark
    if not benchmark_dir.exists():
        raise FileNotFoundError(f"Benchmark dataset directory does not exist: {benchmark_dir}")

    for relative_jsonl in (Path("test.jsonl"), Path("data/test.jsonl")):
        jsonl_path = benchmark_dir / relative_jsonl
        if jsonl_path.is_file():
            return [jsonl_path]

    search_patterns = ("test*.parquet", "*.parquet")
    for relative_dir in (Path("."), Path("data")):
        directory = benchmark_dir / relative_dir
        if not directory.is_dir():
            continue
        for pattern in search_patterns:
            parquet_files = sorted(directory.glob(pattern))
            if parquet_files:
                return parquet_files

    raise FileNotFoundError(f"No local jsonl/parquet benchmark files found under {benchmark_dir}")


def _filter_amo_bench_p_rows(rows: list[dict[str, Any]], benchmark: str) -> list[dict[str, Any]]:
    """Select and validate the official 39-problem parser-based AMO-Bench-P subset."""
    missing_answer_type = [idx for idx, row in enumerate(rows) if _is_missing(row.get("answer_type"))]
    if missing_answer_type:
        raise ValueError(
            f"{benchmark} must provide answer_type for every row so AMO-Bench-P can exclude description questions; "
            f"missing at row indices {missing_answer_type[:10]}."
        )

    parser_rows = [row for row in rows if str(row["answer_type"]).strip().lower() in AMO_BENCH_P_ANSWER_TYPES]
    unsupported_types = sorted(
        {
            str(row["answer_type"]).strip().lower()
            for row in rows
            if str(row["answer_type"]).strip().lower() not in AMO_BENCH_P_ANSWER_TYPES | {"description"}
        }
    )
    if unsupported_types:
        raise ValueError(f"{benchmark} contains unsupported answer types: {unsupported_types!r}.")
    if len(parser_rows) != AMO_BENCH_P_SIZE:
        raise ValueError(
            f"{benchmark} must contain exactly {AMO_BENCH_P_SIZE} parser-graded questions after excluding "
            f"answer_type='description', but found {len(parser_rows)}."
        )
    return parser_rows


def load_benchmark_samples(
    dataset_root: str | Path,
    benchmark: str,
    *,
    max_samples: int | None = None,
) -> list[EvalSample]:
    data_files = find_benchmark_data_files(dataset_root, benchmark)
    rows: list[dict[str, Any]] = []
    for data_file in data_files:
        if data_file.suffix == ".jsonl":
            rows.extend(_load_jsonl_dataset(data_file))
        else:
            rows.extend(pd.read_parquet(data_file).to_dict(orient="records"))

    amo_bench = is_amo_bench(benchmark)
    if amo_bench:
        rows = _filter_amo_bench_p_rows(rows, benchmark)

    samples = [
        _row_to_sample(row, benchmark=benchmark, idx=idx, is_amo_bench=amo_bench) for idx, row in enumerate(rows)
    ]
    if max_samples is not None and max_samples >= 0:
        samples = samples[:max_samples]
    return samples


def build_math_chat_messages(
    problem: str,
    idx: int = 0,
    *,
    use_raw_prompt: bool = False,
) -> list[dict[str, str]]:
    if use_raw_prompt:
        return [{"role": "user", "content": str(problem).strip()}]
    return build_qwen3_boxed_prompt(problem, idx)


def split_final_content(
    text: str | None,
    *,
    reasoning_content: str | None = None,
    content: str | None = None,
) -> str:
    raw_text = str(content).strip() if content is not None and str(content).strip() else str(text or "")
    for tag in THINK_END_TAGS:
        if tag in raw_text:
            return raw_text.rsplit(tag, 1)[-1].strip()
    if reasoning_content:
        reasoning = str(reasoning_content)
        if raw_text.startswith(reasoning):
            return raw_text[len(reasoning) :].strip()
    return raw_text.strip()


def score_completion(
    text: str,
    ground_truth: str,
    *,
    finish_reason: str | None = None,
    response_token_count: int | None = None,
    reasoning_content: str | None = None,
    content: str | None = None,
    verifier: str = "hybrid",
    math_verify_timeout: float = 30.0,
    extra_info: dict[str, Any] | None = None,
) -> ScoredCompletion:
    final_content = split_final_content(text, reasoning_content=reasoning_content, content=content)
    if verifier not in {"math_dapo", "math_verify", "hybrid"}:
        raise ValueError(f"verifier must be one of math_dapo/math_verify/hybrid, got {verifier!r}")

    dapo_score = default_compute_score("math_dapo", final_content, ground_truth)
    if not isinstance(dapo_score, dict):
        dapo_score = math_dapo.compute_score(final_content, ground_truth)

    acc = bool(dapo_score["acc"])
    verified_by = "math_dapo" if acc else "none"
    math_verify_score = None
    if verifier == "math_verify" or (verifier == "hybrid" and not acc):
        math_verify_score = float(
            math_verify.compute_score(
                final_content,
                ground_truth,
                timeout_score=0.0,
                timeout=math_verify_timeout,
                extra_info=extra_info,
            )
        )
        if verifier == "math_verify":
            acc = math_verify_score > 0.0
            verified_by = "math_verify" if acc else "none"
        elif math_verify_score > 0.0:
            acc = True
            verified_by = "math_verify"

    return ScoredCompletion(
        text=text,
        final_content=final_content,
        reasoning_content=reasoning_content,
        content=content,
        finish_reason=finish_reason,
        response_token_count=response_token_count,
        acc=acc,
        pred=str(dapo_score["pred"]),
        invalid_pred=bool(dapo_score["invalid_pred"]) and not acc,
        has_answer_pattern=bool(dapo_score["has_answer_pattern"]),
        has_boxed_pattern=bool(dapo_score["has_boxed_pattern"]),
        has_accepted_answer_format=bool(dapo_score["has_accepted_answer_format"]),
        verified_by=verified_by,
        math_verify_score=math_verify_score,
    )
