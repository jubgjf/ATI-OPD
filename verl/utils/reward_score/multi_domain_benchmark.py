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

"""Rule-based scoring for the typed multi-domain benchmark adapters."""

from __future__ import annotations

import copy
import json
import math
import re
from typing import Any, Mapping

from verl.utils.evaluation.math_benchmark import split_final_content

_CHOICE_PATTERNS = (
    re.compile(r"the\s+correct\s+answer\s+is\s*\(?\s*([A-J])\s*\)?", re.IGNORECASE),
    re.compile(r"(?:final\s+)?answer\s*[:：]\s*\(?\s*([A-J])\s*\)?", re.IGNORECASE),
    re.compile(r"\\boxed\s*\{\s*([A-J])\s*\}", re.IGNORECASE),
    re.compile(r"\*\*\s*([A-J])\s*\*\*", re.IGNORECASE),
)
_THINK_OPEN_RE = re.compile(r"<think(?:\s[^>]*)?>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think\s*>", re.IGNORECASE)
_MULTIPLE_CHOICE_METADATA_KEYS = (
    "benchmark",
    "category",
    "length_bucket",
    "original_prompt_tokens",
    "original_prompt_tokens_exact",
    "formatted_prompt_tokens",
    "original_context_tokens",
    "retained_context_tokens",
    "retained_ratio",
    "truncated",
)


def _parse_payload(ground_truth: Any) -> dict[str, Any]:
    if isinstance(ground_truth, str):
        try:
            payload = json.loads(ground_truth)
        except json.JSONDecodeError as exc:
            raise ValueError("multi_domain_benchmark ground_truth must be valid JSON.") from exc
    elif isinstance(ground_truth, Mapping):
        payload = dict(ground_truth)
    else:
        raise TypeError(
            f"multi_domain_benchmark ground_truth must be a JSON string or mapping, got {type(ground_truth).__name__}."
        )
    if not isinstance(payload, dict) or not str(payload.get("task_type", "")).strip():
        raise ValueError("multi_domain_benchmark payload requires a non-empty task_type.")
    return payload


def _visible_answer(solution_str: Any) -> str:
    text = str(solution_str or "")
    if _THINK_OPEN_RE.search(text) and not _THINK_CLOSE_RE.search(text):
        return ""
    return split_final_content(text)


def extract_choice(text: str, valid_labels: tuple[str, ...]) -> str | None:
    """Extract the final multiple-choice letter without guessing on malformed output."""
    matches: list[tuple[int, str]] = []
    for pattern in _CHOICE_PATTERNS:
        matches.extend((match.start(), match.group(1).upper()) for match in pattern.finditer(text))
    if matches:
        answer = max(matches, key=lambda item: item[0])[1]
        return answer if answer in valid_labels else None
    stripped = text.strip().upper()
    match = re.fullmatch(r"\(?\s*([A-J])\s*\)?[.!]?", stripped)
    if match and match.group(1) in valid_labels:
        return match.group(1)
    return None


def _normalize_ifbench_kwarg(value: Any) -> Any:
    """Restore integer-valued numbers lost to Arrow/JSON schema widening."""
    if isinstance(value, Mapping):
        return {str(key): _normalize_ifbench_kwarg(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_normalize_ifbench_kwarg(item) for item in value]
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return value


def _check_ifbench_instruction(
    *,
    instruction_id: str,
    kwargs: Mapping[str, Any],
    prompt: str,
    responses: list[str],
) -> bool:
    try:
        from verl.third_party.ifbench import instructions_registry
    except ImportError as exc:  # pragma: no cover - exercised by launcher preflight on minimal environments.
        raise RuntimeError(
            "IFBench evaluation dependencies are missing. Install verl's base dependencies before starting the Ray job."
        ) from exc
    if instruction_id not in instructions_registry.INSTRUCTION_DICT:
        raise ValueError(f"Unknown IFBench instruction_id {instruction_id!r}.")
    instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
    for response in responses:
        instruction = instruction_cls(instruction_id)
        clean_kwargs = {key: _normalize_ifbench_kwarg(value) for key, value in kwargs.items() if value is not None}
        instruction.build_description(**clean_kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=prompt)
        if response.strip() and bool(instruction.check_following(response)):
            return True
    return False


def _validate_ifbench_verifier_payload(payload: Mapping[str, Any]) -> None:
    """Fail on corrupt official-verifier metadata even for an empty completion."""
    from verl.third_party.ifbench import instructions_registry, require_nltk_resources

    require_nltk_resources()
    prompt = str(payload["prompt"])
    for instruction_id, kwargs in zip(payload["instruction_id_list"], payload["kwargs"], strict=True):
        instruction_id = str(instruction_id)
        instruction_cls = instructions_registry.INSTRUCTION_DICT.get(instruction_id)
        if instruction_cls is None:
            raise ValueError(f"Unknown IFBench instruction_id {instruction_id!r}.")
        clean_kwargs = {
            key: _normalize_ifbench_kwarg(value) for key, value in dict(kwargs).items() if value is not None
        }
        try:
            instruction = instruction_cls(instruction_id)
            instruction.build_description(**clean_kwargs)
            args = instruction.get_instruction_args()
            if args and "prompt" in args:
                instruction.build_description(prompt=prompt)
        except Exception as exc:
            raise ValueError(f"Invalid IFBench verifier kwargs for {instruction_id!r}: {exc}") from exc


def _score_ifbench(solution: str, payload: dict[str, Any]) -> dict[str, Any]:
    prompt = payload.get("prompt")
    instruction_ids = payload.get("instruction_id_list")
    instruction_kwargs = payload.get("kwargs")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("IFBench payload requires a non-empty prompt.")
    if not isinstance(instruction_ids, list) or not isinstance(instruction_kwargs, list):
        raise ValueError("IFBench payload requires instruction_id_list and kwargs lists.")
    if not instruction_ids or len(instruction_ids) != len(instruction_kwargs):
        raise ValueError("IFBench instruction_id_list and kwargs must be non-empty and aligned.")
    if not all(isinstance(item, Mapping) for item in instruction_kwargs):
        raise ValueError("IFBench kwargs entries must be mappings.")

    response_lines = solution.split("\n")
    without_first = "\n".join(response_lines[1:]).strip()
    without_last = "\n".join(response_lines[:-1]).strip()
    without_both = "\n".join(response_lines[1:-1]).strip()
    loose_responses = [
        solution,
        solution.replace("*", ""),
        without_first,
        without_last,
        without_both,
        without_first.replace("*", ""),
        without_last.replace("*", ""),
        without_both.replace("*", ""),
    ]
    strict: list[bool] = []
    loose: list[bool] = []
    for instruction_id, kwargs in zip(instruction_ids, instruction_kwargs, strict=True):
        strict.append(
            _check_ifbench_instruction(
                instruction_id=str(instruction_id),
                kwargs=copy.deepcopy(dict(kwargs)),
                prompt=prompt,
                responses=[solution],
            )
        )
        loose.append(
            _check_ifbench_instruction(
                instruction_id=str(instruction_id),
                kwargs=copy.deepcopy(dict(kwargs)),
                prompt=prompt,
                responses=loose_responses,
            )
        )
    prompt_loose = all(loose)
    return {
        "score": float(prompt_loose),
        "acc": prompt_loose,
        "pred": "PASS" if prompt_loose else "FAIL",
        "parse_failure": False,
        "prompt_loose_acc": float(prompt_loose),
        "prompt_strict_acc": float(all(strict)),
        "instruction_loose_acc": float(sum(loose) / len(loose)),
        "instruction_strict_acc": float(sum(strict) / len(strict)),
        "instruction_count": len(instruction_ids),
    }


def _score_aime(solution: str, payload: dict[str, Any], extra_info: dict[str, Any] | None) -> dict[str, Any]:
    expected = payload.get("expected_answer")
    if expected is None or not str(expected).strip():
        raise ValueError("AIME benchmark payload requires expected_answer.")
    from verl.utils.reward_score import math_verify

    score = float(math_verify.compute_score(solution, str(expected), timeout_score=0.0, extra_info=extra_info))
    return {
        "score": score,
        "acc": bool(score > 0.0),
        "pred": "[VERIFIED]" if score > 0.0 else "[INVALID]",
        "parse_failure": False,
        "math_verify_score": score,
    }


def _score_multiple_choice(solution: str, payload: dict[str, Any]) -> dict[str, Any]:
    expected = str(payload.get("expected_answer", "")).strip().upper()
    raw_labels = payload.get("valid_labels")
    if not isinstance(raw_labels, list | tuple) or not raw_labels:
        raise ValueError("Multiple-choice payload requires non-empty valid_labels.")
    valid_labels = tuple(str(label).upper() for label in raw_labels)
    if expected not in valid_labels:
        raise ValueError(f"Expected answer {expected!r} is not in valid_labels={valid_labels}.")
    pred = extract_choice(solution, valid_labels)
    acc = pred == expected
    result: dict[str, Any] = {
        "score": float(acc),
        "acc": acc,
        "pred": pred if pred is not None else "[INVALID]",
        "parse_failure": pred is None,
    }
    for key in _MULTIPLE_CHOICE_METADATA_KEYS:
        if key in payload:
            result[key] = payload[key]
    return result


def _validate_task_payload(task_type: str, payload: dict[str, Any]) -> None:
    if task_type == "aime":
        expected = payload.get("expected_answer")
        if expected is None or not str(expected).strip():
            raise ValueError("AIME benchmark payload requires expected_answer.")
        return
    if task_type == "ifbench":
        prompt = payload.get("prompt")
        instruction_ids = payload.get("instruction_id_list")
        instruction_kwargs = payload.get("kwargs")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("IFBench payload requires a non-empty prompt.")
        if not isinstance(instruction_ids, list) or not instruction_ids:
            raise ValueError("IFBench payload requires a non-empty instruction_id_list.")
        if not isinstance(instruction_kwargs, list) or len(instruction_ids) != len(instruction_kwargs):
            raise ValueError("IFBench instruction_id_list and kwargs must be aligned lists.")
        if not all(isinstance(item, Mapping) for item in instruction_kwargs):
            raise ValueError("IFBench kwargs entries must be mappings.")
        return
    if task_type == "multiple_choice":
        expected = str(payload.get("expected_answer", "")).strip().upper()
        labels = payload.get("valid_labels")
        if not isinstance(labels, list | tuple) or not labels:
            raise ValueError("Multiple-choice payload requires non-empty valid_labels.")
        if expected not in {str(label).strip().upper() for label in labels}:
            raise ValueError("Multiple-choice expected_answer must be present in valid_labels.")
        return
    raise ValueError(f"Unsupported multi-domain benchmark task_type {task_type!r}.")


def compute_score(
    solution_str: str,
    ground_truth: Any,
    *,
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Score one visible completion using its typed benchmark payload."""
    payload = _parse_payload(ground_truth)
    visible = _visible_answer(solution_str)
    task_type = str(payload["task_type"])
    _validate_task_payload(task_type, payload)
    if task_type == "ifbench":
        _validate_ifbench_verifier_payload(payload)
    if not visible:
        result: dict[str, Any] = {
            "score": 0.0,
            "acc": False,
            "pred": "[INVALID]",
            "parse_failure": True,
        }
        if task_type == "aime":
            result["math_verify_score"] = 0.0
        elif task_type == "ifbench":
            instruction_ids = payload["instruction_id_list"]
            result.update(
                {
                    "prompt_loose_acc": 0.0,
                    "prompt_strict_acc": 0.0,
                    "instruction_loose_acc": 0.0,
                    "instruction_strict_acc": 0.0,
                    "instruction_count": len(instruction_ids),
                }
            )
        elif task_type == "multiple_choice":
            for key in _MULTIPLE_CHOICE_METADATA_KEYS:
                if key in payload:
                    result[key] = payload[key]
    elif task_type == "aime":
        result = _score_aime(visible, payload, extra_info)
    elif task_type == "ifbench":
        result = _score_ifbench(visible, payload)
    elif task_type == "multiple_choice":
        result = _score_multiple_choice(visible, payload)
    else:
        raise ValueError(f"Unsupported multi-domain benchmark task_type {task_type!r}.")
    result["visible_answer_chars"] = len(visible)
    return result


__all__ = ["compute_score", "extract_choice"]
