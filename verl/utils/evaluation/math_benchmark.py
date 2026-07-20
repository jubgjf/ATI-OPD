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
import os
import re
from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class EvalSample:
    benchmark: str
    sample_id: str
    problem: str
    ground_truth: str


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


@dataclass(frozen=True)
class CheckpointInfo:
    step: int
    path: Path
    actor_dir: Path
    hf_dir: Path


def _json_default(value: Any) -> Any:
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


def append_jsonl_record(path: str | Path, record: dict[str, Any], *, fsync: bool = True) -> None:
    jsonl_path = Path(path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, default=_json_default))
        file.write("\n")
        file.flush()
        if fsync:
            os.fsync(file.fileno())


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, default=_json_default)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    tmp_path.replace(output_path)


def parse_step_from_checkpoint_name(name: str) -> int | None:
    match = re.fullmatch(r"global_step_(\d+)", name)
    return None if match is None else int(match.group(1))


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


def _row_to_sample(row: dict[str, Any], benchmark: str, idx: int) -> EvalSample:
    sample_id = row.get("id", idx)
    if _is_missing(sample_id):
        sample_id = idx
    return EvalSample(
        benchmark=benchmark,
        sample_id=str(sample_id),
        problem=_extract_prompt(row, idx),
        ground_truth=_extract_answer(row, idx),
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


def load_benchmark_samples(
    dataset_root: str | Path,
    benchmark: str,
    *,
    max_samples: int | None = None,
) -> list[EvalSample]:
    benchmark_dir = Path(dataset_root).expanduser() / benchmark
    if not benchmark_dir.exists():
        raise FileNotFoundError(f"Benchmark dataset directory does not exist: {benchmark_dir}")

    jsonl_path = benchmark_dir / "test.jsonl"
    if jsonl_path.exists():
        rows = _load_jsonl_dataset(jsonl_path)
    else:
        parquet_files = sorted(benchmark_dir.glob("test*.parquet")) or sorted(benchmark_dir.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No local jsonl/parquet benchmark files found under {benchmark_dir}")
        rows = []
        for parquet_file in parquet_files:
            rows.extend(pd.read_parquet(parquet_file).to_dict(orient="records"))

    samples = [_row_to_sample(row, benchmark=benchmark, idx=idx) for idx, row in enumerate(rows)]
    if max_samples is not None and max_samples >= 0:
        samples = samples[:max_samples]
    return samples


def build_math_chat_messages(problem: str, idx: int = 0) -> list[dict[str, str]]:
    return build_qwen3_boxed_prompt(problem, idx)


def render_prompt_with_chat_template(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    enable_thinking: bool | None = True,
) -> str:
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


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
            math_verify.compute_score(final_content, ground_truth, timeout_score=0.0, timeout=math_verify_timeout)
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


def hf_model_ready(hf_dir: str | Path) -> bool:
    path = Path(hf_dir)
    if not (path / "config.json").exists():
        return False
    patterns = ("*.safetensors", "pytorch_model*.bin", "model*.bin")
    return any(any(path.glob(pattern)) for pattern in patterns)
