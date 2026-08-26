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

"""Reward scorer for the Nemotron Cascade 2 MOPD subset.

Instruction-following samples are scored by the repository's pinned, vendored
Open-Instruct IFEvalG verifier. Missing offline resources fail explicitly; the
reward worker never downloads data and never substitutes approximate semantics.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from . import math_verify

_MATH_ANSWER_RE = re.compile(
    r"(?:\\boxed\s*\{|(?:final\s+)?answer\s*(?::|=|is)|[-+]?\d+(?:[.,/]\d+)?)",
    flags=re.IGNORECASE,
)
_TASK_ALIASES = {
    "acereason": "acereason",
    "math_answer": "acereason",
    "instruction_following": "instruction_following",
    "nano-v3-rl-instruction_following": "instruction_following",
    "json_schema": "structured_output",
    "structured_output": "structured_output",
    "structured_outputs": "structured_output",
    "nano_v3_sft_profiled_structured_outputs": "structured_output",
    "mcqa_regex": "stem_mcqa",
    "stem_mcqa": "stem_mcqa",
    "nano_v3_sft_profiled_stem_mcqa": "stem_mcqa",
}


def _parse_payload(ground_truth: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(ground_truth, str):
        try:
            payload = json.loads(ground_truth)
        except json.JSONDecodeError as exc:
            raise ValueError("MOPD ground_truth must be a valid JSON object") from exc
    elif isinstance(ground_truth, Mapping):
        payload = dict(ground_truth)
    else:
        raise TypeError("MOPD ground_truth must be a JSON string or mapping")

    if not isinstance(payload, dict):
        raise ValueError("MOPD ground_truth JSON must contain an object")
    task_type = payload.get("task_type")
    verifier = payload.get("verifier")
    discriminator = task_type if task_type is not None else verifier
    if not isinstance(discriminator, str) or not discriminator.strip():
        raise ValueError("MOPD ground_truth is missing a non-empty task_type/verifier")
    normalized = _TASK_ALIASES.get(discriminator.strip().lower())
    if normalized is None:
        raise ValueError(f"Unsupported MOPD task_type/verifier: {discriminator!r}")
    if task_type is not None and verifier is not None:
        normalized_verifier = _TASK_ALIASES.get(verifier.strip().lower()) if isinstance(verifier, str) else None
        if normalized_verifier != normalized:
            raise ValueError(f"Conflicting MOPD task_type={task_type!r} and verifier={verifier!r}")
    payload["task_type"] = normalized
    return payload


def _malformed_answer(task_type: str, reason: str) -> dict[str, Any]:
    return {
        "score": 0.0,
        "acc": False,
        "parse_failure": True,
        "task_type": task_type,
        "error": reason,
    }


def _strip_qwen_thinking(value: str) -> tuple[str, bool, bool]:
    """Return visible answer, whether thinking was removed, and an unclosed-tag flag."""
    closing_tags = list(re.finditer(r"</think\s*>", value, flags=re.IGNORECASE))
    if closing_tags:
        return value[closing_tags[-1].end() :].strip(), True, False
    if re.search(r"<think(?:\s[^>]*)?>", value, flags=re.IGNORECASE):
        return "", True, True
    return value.strip(), False, False


def _strip_markdown_json_fence(value: str) -> str:
    match = re.fullmatch(r"\s*```(?:json)?[ \t]*\r?\n?(.*?)\r?\n?```\s*", value, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else value.strip()


def _expected_answer(payload: Mapping[str, Any]) -> str:
    answer = payload.get("expected_answer", payload.get("answer"))
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("MOPD payload requires a non-empty string expected_answer")
    return answer.strip()


def _score_acereason(solution_str: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
    expected = _expected_answer(payload)
    if not isinstance(solution_str, str) or not solution_str.strip():
        return _malformed_answer("acereason", "empty_or_non_string_model_answer")
    visible_answer, thinking_stripped, unclosed_think = _strip_qwen_thinking(solution_str)
    if unclosed_think or not visible_answer:
        return _malformed_answer("acereason", "unclosed_think_tag_or_empty_visible_answer")

    verified_score = float(math_verify.compute_score(visible_answer, expected))
    acc = verified_score > 0.0
    return {
        "score": verified_score,
        "acc": bool(acc),
        "parse_failure": bool(not acc and not _MATH_ANSWER_RE.search(visible_answer)),
        "task_type": "acereason",
        "math_verify_score": verified_score,
        "expected_answer": expected,
        "thinking_stripped": thinking_stripped,
    }


def _load_json_schema(payload: Mapping[str, Any]) -> dict[str, Any]:
    schema_type = payload.get("schema_type")
    if schema_type is not None and (not isinstance(schema_type, str) or schema_type.lower() != "json"):
        raise ValueError(f"structured_output only supports schema_type='json', got {schema_type!r}")

    schema = None
    for key in ("schema", "json_schema", "response_schema", "schema_str"):
        if key in payload:
            schema = payload[key]
            break
    if schema is None:
        raise ValueError("structured_output payload is missing schema/schema_str")
    if isinstance(schema, str):
        try:
            schema = json.loads(schema)
        except json.JSONDecodeError as exc:
            raise ValueError("structured_output schema_str is not valid JSON") from exc
    if not isinstance(schema, dict):
        raise ValueError("structured_output JSON Schema must be an object")
    return schema


_SCHEMA_MAP_CONTAINERS = frozenset({"$defs", "definitions", "dependentSchemas", "patternProperties"})
_SCHEMA_SINGLE_CONTAINERS = frozenset(
    {
        "additionalItems",
        "additionalProperties",
        "contains",
        "contentSchema",
        "else",
        "if",
        "not",
        "propertyNames",
        "then",
        "unevaluatedItems",
        "unevaluatedProperties",
    }
)
_SCHEMA_LIST_CONTAINERS = frozenset({"allOf", "anyOf", "oneOf", "prefixItems"})


def _misplaced_schema_keyword(name: str, value: Any) -> bool:
    """Recognize schema keywords serialized as entries of ``properties`` by MOPD."""
    integer_keywords = {"maxItems", "maxProperties", "minItems", "minProperties"}
    if name == "required":
        return isinstance(value, list)
    if name in {"additionalProperties", "uniqueItems"}:
        return isinstance(value, bool)
    if name in integer_keywords:
        return isinstance(value, int) and not isinstance(value, bool)
    return name == "description" and isinstance(value, str)


def normalize_nvidia_schema_dialect(schema: Mapping[str, Any]) -> tuple[dict[str, Any], int]:
    """Normalize deterministic schema serialization defects in NVIDIA MOPD.

    The released structured-output rows mix standard JSON Schema with several
    OpenAI-style or mis-nested keywords.  This adapter performs only repairs
    whose intended standard location/type is unambiguous.  Unknown malformed
    constructs are intentionally left for ``jsonschema.check_schema`` to reject.

    Returns:
        A fresh normalized schema and the number of applied repairs.
    """

    if not isinstance(schema, Mapping):
        raise ValueError("NVIDIA schema dialect root must be an object")

    def visit(node: Any, path: tuple[str | int, ...]) -> tuple[Any, int]:
        if isinstance(node, bool):
            return node, 0
        if not isinstance(node, Mapping):
            return node, 0

        normalized = dict(node)
        repairs = 0
        promoted_required: list[str] = []
        require_all_properties = False

        required = normalized.get("required")
        if isinstance(required, bool):
            normalized.pop("required")
            repairs += 1
            if required:
                properties = normalized.get("properties")
                if not isinstance(properties, Mapping):
                    location = ".".join(str(part) for part in path) or "<root>"
                    raise ValueError(f"Boolean required=true at {location} has no properties to require")
                require_all_properties = True

        description = normalized.get("description")
        if description is not None and not isinstance(description, str):
            if isinstance(description, Mapping) and not description:
                normalized.pop("description")
                repairs += 1
            else:
                location = ".".join(str(part) for part in (*path, "description"))
                raise ValueError(f"Cannot normalize non-string schema description at {location}")

        schema_id = normalized.get("$id")
        if isinstance(schema_id, str) and schema_id.startswith("#/"):
            # A non-empty fragment is not a valid base URI and none of the MOPD
            # rows use these fragment-only identifiers as references.
            normalized.pop("$id")
            repairs += 1

        for exclusive_key, inclusive_key in (
            ("exclusiveMinimum", "minimum"),
            ("exclusiveMaximum", "maximum"),
        ):
            exclusive = normalized.get(exclusive_key)
            if not isinstance(exclusive, bool):
                continue
            normalized.pop(exclusive_key)
            repairs += 1
            if exclusive:
                inclusive = normalized.pop(inclusive_key, None)
                if isinstance(inclusive, bool) or not isinstance(inclusive, int | float):
                    location = ".".join(str(part) for part in (*path, exclusive_key))
                    raise ValueError(
                        f"Cannot normalize {exclusive_key}=true without numeric {inclusive_key} at {location}"
                    )
                normalized[exclusive_key] = inclusive

        properties = normalized.get("properties")
        if isinstance(properties, Mapping):
            normalized_properties: dict[str, Any] = {}
            hoisted_keywords: dict[str, Any] = {}
            for property_name, property_schema in properties.items():
                if _misplaced_schema_keyword(str(property_name), property_schema):
                    hoisted_keywords[str(property_name)] = property_schema
                    repairs += 1
                    continue

                if isinstance(property_schema, Mapping) and isinstance(property_schema.get("required"), bool):
                    property_schema = dict(property_schema)
                    property_required = property_schema.pop("required")
                    repairs += 1
                    if property_required:
                        promoted_required.append(str(property_name))
                normalized_property, property_repairs = visit(
                    property_schema, (*path, "properties", str(property_name))
                )
                normalized_properties[str(property_name)] = normalized_property
                repairs += property_repairs
            normalized["properties"] = normalized_properties
            if require_all_properties:
                promoted_required.extend(normalized_properties)

            for keyword, keyword_value in hoisted_keywords.items():
                if keyword == "required":
                    promoted_required.extend(keyword_value)
                    continue
                if keyword in normalized and normalized[keyword] != keyword_value:
                    location = ".".join(str(part) for part in path) or "<root>"
                    raise ValueError(f"Conflicting {keyword!r} values while normalizing schema at {location}")
                normalized[keyword] = keyword_value

        for container in _SCHEMA_MAP_CONTAINERS:
            children = normalized.get(container)
            if not isinstance(children, Mapping):
                continue
            normalized_children = {}
            for name, child in children.items():
                normalized_child, child_repairs = visit(child, (*path, container, str(name)))
                normalized_children[str(name)] = normalized_child
                repairs += child_repairs
            normalized[container] = normalized_children

        items = normalized.get("items")
        if isinstance(items, Mapping | bool):
            normalized["items"], item_repairs = visit(items, (*path, "items"))
            repairs += item_repairs
        elif isinstance(items, list):
            normalized_items = []
            for index, item in enumerate(items):
                normalized_item, item_repairs = visit(item, (*path, "items", index))
                normalized_items.append(normalized_item)
                repairs += item_repairs
            normalized["items"] = normalized_items

        for container in _SCHEMA_SINGLE_CONTAINERS:
            child = normalized.get(container)
            if not isinstance(child, Mapping | bool):
                continue
            normalized[container], child_repairs = visit(child, (*path, container))
            repairs += child_repairs

        for container in _SCHEMA_LIST_CONTAINERS:
            children = normalized.get(container)
            if not isinstance(children, list):
                continue
            normalized_children = []
            for index, child in enumerate(children):
                normalized_child, child_repairs = visit(child, (*path, container, index))
                normalized_children.append(normalized_child)
                repairs += child_repairs
            normalized[container] = normalized_children

        if promoted_required:
            existing_required = normalized.get("required", [])
            if not isinstance(existing_required, list):
                location = ".".join(str(part) for part in path) or "<root>"
                raise ValueError(f"Cannot merge required properties at {location}")
            normalized["required"] = list(dict.fromkeys([*existing_required, *promoted_required]))

        return normalized, repairs

    normalized_schema, repair_count = visit(schema, ())
    return dict(normalized_schema), repair_count


def _score_structured_output(solution_str: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
    schema, schema_normalization_count = normalize_nvidia_schema_dialect(_load_json_schema(payload))
    schema_normalized = schema_normalization_count > 0
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover - jsonschema is part of the supported runtime
        raise RuntimeError("structured_output scoring requires the jsonschema package") from exc

    validator_cls = jsonschema.validators.validator_for(schema)
    try:
        validator_cls.check_schema(schema)
    except jsonschema.exceptions.SchemaError as exc:
        raise ValueError(f"Invalid structured_output JSON Schema: {exc.message}") from exc

    if not isinstance(solution_str, str) or not solution_str.strip():
        result = _malformed_answer("structured_output", "empty_or_non_string_model_answer")
        result.update(
            {"schema_normalized": schema_normalized, "schema_normalization_count": schema_normalization_count}
        )
        return result
    visible_answer, thinking_stripped, unclosed_think = _strip_qwen_thinking(solution_str)
    if unclosed_think or not visible_answer:
        result = _malformed_answer("structured_output", "unclosed_think_tag_or_empty_visible_answer")
        result.update(
            {"schema_normalized": schema_normalized, "schema_normalization_count": schema_normalization_count}
        )
        return result
    json_text = _strip_markdown_json_fence(visible_answer)
    try:
        instance = json.loads(json_text)
    except json.JSONDecodeError as exc:
        result = _malformed_answer("structured_output", "invalid_json_model_answer")
        result.update(
            {
                "json_parsed": False,
                "schema_valid": False,
                "schema_normalized": schema_normalized,
                "schema_normalization_count": schema_normalization_count,
                "thinking_stripped": thinking_stripped,
                "validation_error": f"{exc.msg} at line {exc.lineno} column {exc.colno}",
            }
        )
        return result

    validator = validator_cls(schema, format_checker=jsonschema.FormatChecker())
    first_error = next(validator.iter_errors(instance), None)
    acc = first_error is None
    result = {
        "score": float(acc),
        "acc": bool(acc),
        "parse_failure": False,
        "task_type": "structured_output",
        "json_parsed": True,
        "schema_valid": bool(acc),
        "schema_normalized": schema_normalized,
        "schema_normalization_count": schema_normalization_count,
        "thinking_stripped": thinking_stripped,
    }
    if first_error is not None:
        path = ".".join(str(part) for part in first_error.absolute_path) or "<root>"
        result["validation_error"] = f"{path}: {first_error.message}"
    return result


def _score_stem_mcqa(solution_str: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
    expected = _expected_answer(payload)
    output_regex = payload.get("output_regex")
    if output_regex is None and isinstance(payload.get("template_metadata"), Mapping):
        output_regex = payload["template_metadata"].get("output_regex")
    if not isinstance(output_regex, str) or not output_regex:
        raise ValueError("stem_mcqa payload requires a non-empty output_regex")
    try:
        pattern = re.compile(output_regex)
    except re.error as exc:
        raise ValueError(f"Invalid stem_mcqa output_regex: {exc}") from exc
    if pattern.groups < 1:
        raise ValueError("stem_mcqa output_regex must expose the answer in capture group 1")

    if not isinstance(solution_str, str) or not solution_str.strip():
        return _malformed_answer("stem_mcqa", "empty_or_non_string_model_answer")
    visible_answer, thinking_stripped, unclosed_think = _strip_qwen_thinking(solution_str)
    if unclosed_think or not visible_answer:
        return _malformed_answer("stem_mcqa", "unclosed_think_tag_or_empty_visible_answer")
    matches = list(pattern.finditer(visible_answer))
    if not matches:
        result = _malformed_answer("stem_mcqa", "output_regex_did_not_match")
        result.update({"expected_answer": expected, "thinking_stripped": thinking_stripped})
        return result

    captured_answer = matches[-1].group(1)
    if captured_answer is None:
        result = _malformed_answer("stem_mcqa", "output_regex_capture_group_was_empty")
        result.update({"expected_answer": expected, "thinking_stripped": thinking_stripped})
        return result
    predicted = captured_answer.strip()
    acc = predicted.casefold() == expected.casefold()
    return {
        "score": float(acc),
        "acc": bool(acc),
        "parse_failure": False,
        "task_type": "stem_mcqa",
        "predicted_answer": predicted,
        "expected_answer": expected,
        "regex_match_count": len(matches),
        "thinking_stripped": thinking_stripped,
    }


SUPPORTED_INSTRUCTION_IDS = frozenset(
    {
        "change_case:capital_word_frequency",
        "change_case:english_capital",
        "change_case:english_lowercase",
        "combination:two_responses",
        "copy:repeat_phrase",
        "count:count_increment_word",
        "count:count_unique",
        "count:counting_composition",
        "count:lowercase_counting",
        "detectable_content:number_placeholders",
        "detectable_content:postscript",
        "detectable_format:bigram_wrapping",
        "detectable_format:constrained_response",
        "detectable_format:json_format",
        "detectable_format:multiple_sections",
        "detectable_format:number_bullet_lists",
        "detectable_format:number_highlighted_sections",
        "detectable_format:sentence_hyphens",
        "detectable_format:square_brackets",
        "detectable_format:title",
        "first_word:first_word_answer",
        "first_word:first_word_sent",
        "keywords:existence",
        "keywords:forbidden_words",
        "keywords:frequency",
        "keywords:keyword_specific_position",
        "keywords:letter_frequency",
        "keywords:no_adjacent_consecutive",
        "keywords:palindrome",
        "keywords:start_end",
        "keywords:word_count_different_numbers",
        "keywords:word_once",
        "language:response_language",
        "last_word:last_word_answer",
        "last_word:last_word_sent",
        "length_constraints:nth_paragraph_first_word",
        "length_constraints:number_paragraphs",
        "length_constraints:number_sentences",
        "length_constraints:number_words",
        "letters:letter_counting",
        "letters:letter_counting2",
        "paragraphs:paragraphs",
        "paragraphs:paragraphs2",
        "punctuation:no_comma",
        "punctuation:punctuation_dot",
        "punctuation:punctuation_exclamation",
        "startend:end_checker",
        "startend:quotation",
    }
)


def _external_instruction_class(instruction_id: str) -> type[Any]:
    from verl.third_party.ifevalg import instructions_registry

    instruction_cls = instructions_registry.INSTRUCTION_DICT.get(instruction_id)
    if instruction_cls is None:
        raise ValueError(f"Unsupported instruction-following constraint: {instruction_id!r}")
    return instruction_cls


def _validate_instruction_config(instruction_id: str, args: Mapping[str, Any]) -> type[Any]:
    if instruction_id not in SUPPORTED_INSTRUCTION_IDS:
        raise ValueError(f"Unsupported instruction-following constraint: {instruction_id!r}")
    instruction_cls = _external_instruction_class(instruction_id)
    try:
        instruction = instruction_cls(instruction_id)
        required_keys = set(instruction.get_instruction_args_keys())
        actual_keys = set(args)
        if actual_keys != required_keys:
            raise ValueError(
                f"missing={sorted(required_keys - actual_keys)}, extra={sorted(actual_keys - required_keys)}"
            )
        none_keys = sorted(key for key, value in args.items() if value is None)
        if none_keys:
            raise ValueError(f"null values for keys={none_keys}")
        instruction.build_description(**args)
        normalized_args = instruction.get_instruction_args()
        if required_keys and normalized_args != dict(args):
            raise ValueError(f"verifier normalized supplied kwargs from {dict(args)!r} to {normalized_args!r}")
    except Exception as exc:
        raise ValueError(f"Invalid kwargs for instruction {instruction_id!r}: {exc}") from exc
    return instruction_cls


def _evaluate_validated_instruction(
    value: str, instruction_id: str, args: Mapping[str, Any], instruction_cls: type[Any]
) -> tuple[bool, str]:
    instruction = instruction_cls(instruction_id)
    instruction.build_description(**args)
    return bool(instruction.check_following(value)), "vendored_ifevalg"


def _instruction_items(payload: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    if "instructions" in payload:
        instructions = payload["instructions"]
        if isinstance(instructions, str | bytes) or not isinstance(instructions, Sequence) or not instructions:
            raise ValueError("instructions must be a non-empty sequence")
        items: list[tuple[str, Mapping[str, Any]]] = []
        for item in instructions:
            if not isinstance(item, Mapping):
                raise ValueError("Each instructions entry must be an object")
            instruction_id = item.get("instruction_id", item.get("id"))
            kwargs = item.get("kwargs")
            if not isinstance(instruction_id, str) or not instruction_id:
                raise ValueError("Each instructions entry requires an instruction_id")
            if kwargs is None:
                kwargs = {}
            if not isinstance(kwargs, Mapping):
                raise ValueError(f"kwargs for {instruction_id!r} must be an object or null")
            items.append((instruction_id, dict(kwargs)))
        return items

    instruction_ids = payload.get("instruction_id_list", payload.get("instruction_ids"))
    kwargs_list = payload.get("kwargs", payload.get("instruction_kwargs"))
    if isinstance(instruction_ids, str | bytes) or not isinstance(instruction_ids, Sequence) or not instruction_ids:
        raise ValueError("instruction_following payload requires a non-empty instruction_id_list")
    if kwargs_list is None:
        kwargs_list = [None] * len(instruction_ids)
    if isinstance(kwargs_list, str | bytes) or not isinstance(kwargs_list, Sequence):
        raise ValueError("instruction_following kwargs must be a sequence")
    if len(instruction_ids) != len(kwargs_list):
        raise ValueError("instruction_id_list and kwargs must have the same length")

    items = []
    for instruction_id, kwargs in zip(instruction_ids, kwargs_list, strict=True):
        if not isinstance(instruction_id, str) or not instruction_id:
            raise ValueError("instruction_id_list must contain only non-empty strings")
        if kwargs is None:
            kwargs = {}
        if not isinstance(kwargs, Mapping):
            raise ValueError(f"kwargs for {instruction_id!r} must be an object or null")
        items.append((instruction_id, dict(kwargs)))
    return items


def _score_instruction_following(solution_str: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
    from verl.third_party.ifevalg import require_nltk_resources

    require_nltk_resources()
    items = _instruction_items(payload)
    if not isinstance(solution_str, str) or not solution_str.strip():
        # Evaluate once to ensure malformed configuration still raises.
        for instruction_id, args in items:
            _validate_instruction_config(instruction_id, args)
        return _malformed_answer("instruction_following", "empty_or_non_string_model_answer")

    visible_answer, thinking_stripped, unclosed_think = _strip_qwen_thinking(solution_str)
    if unclosed_think or not visible_answer:
        for instruction_id, args in items:
            _validate_instruction_config(instruction_id, args)
        return _malformed_answer("instruction_following", "unclosed_think_tag_or_empty_visible_answer")

    results: list[tuple[str, bool, str]] = []
    evaluator_errors: list[str] = []
    for instruction_id, args in items:
        # Validate ground-truth/configuration outside the answer-evaluation
        # guard. Invalid verifier metadata must fail the job, while a legal but
        # unparseable model answer is a zero-score sample.
        instruction_cls = _validate_instruction_config(instruction_id, args)
        try:
            passed, backend = _evaluate_validated_instruction(visible_answer, instruction_id, args, instruction_cls)
        except (ImportError, RuntimeError):
            raise
        except Exception as exc:
            passed, backend = False, "evaluator_error"
            evaluator_errors.append(f"{instruction_id}:{type(exc).__name__}")
        results.append((instruction_id, passed, backend))

    passed_count = sum(passed for _, passed, _ in results)
    acc = passed_count == len(results)
    pass_rate = passed_count / len(results)
    return {
        "score": float(pass_rate),
        "acc": bool(acc),
        "parse_failure": bool(evaluator_errors),
        "task_type": "instruction_following",
        "constraint_count": len(results),
        "constraints_satisfied": passed_count,
        "constraint_pass_rate": pass_rate,
        "failed_instruction_ids": ",".join(instruction_id for instruction_id, passed, _ in results if not passed),
        "instruction_backends": ",".join(sorted({backend for _, _, backend in results})),
        "evaluator_errors": ",".join(evaluator_errors),
        "thinking_stripped": thinking_stripped,
    }


def compute_score(solution_str: str, ground_truth: str | Mapping[str, Any]) -> dict[str, Any]:
    """Score one MOPD response from its self-describing JSON ground-truth payload."""
    payload = _parse_payload(ground_truth)
    task_type = payload["task_type"]
    if task_type == "acereason":
        return _score_acereason(solution_str, payload)
    if task_type == "instruction_following":
        return _score_instruction_following(solution_str, payload)
    if task_type == "structured_output":
        return _score_structured_output(solution_str, payload)
    if task_type == "stem_mcqa":
        return _score_stem_mcqa(solution_str, payload)
    raise AssertionError(f"Unreachable normalized task_type: {task_type}")


__all__ = [
    "SUPPORTED_INSTRUCTION_IDS",
    "compute_score",
    "normalize_nvidia_schema_dialect",
]
