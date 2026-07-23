# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import multiprocessing
import re
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any

try:
    from math_verify.errors import TimeoutException
except ImportError:

    class TimeoutException(Exception):
        pass

    print("To use Math-Verify, please install it first by running `pip install math-verify`.")

_pool = None
_pool_lock = threading.Lock()

# AMO-Bench's MIT-licensed parser protocol:
# https://github.com/meituan-longcat/AMO-Bench/blob/main/utils.py
_AMO_ANSWER_PREFIXES = (
    "### the final answer is:",
    "### the final answer:",
    "### final answer is:",
    "### final answer:",
    "### the final answer is",
    "### the final answer",
    "### final answer is",
    "### final answer",
)
_AMO_THINK_END_TAGS = ("</think>", "</longcat_think>")
_AMO_CUT_MARKERS = ("\\medskip", "\n---")
_AMO_SIZE_COMMANDS = (
    "\\bigl",
    "\\bigr",
    "\\Bigl",
    "\\Bigr",
    "\\biggl",
    "\\biggr",
    "\\Biggl",
    "\\Biggr",
    "\\bigg",
    "\\Bigg",
    "\\big",
    "\\Big",
    "\\left",
    "\\right",
)
_AMO_REPLACEMENTS = (
    ("‘", "'"),
    ("’", "'"),
    ("“", '"'),
    ("”", '"'),
    ("（", "("),
    ("）", ")"),
    ("，", ", "),
    ("：", ": "),
    ("；", "; "),
    ("。", ". "),
    ("！", "! "),
    ("？", "? "),
    ("…", "..."),
    ("–", "-"),
    ("−", "-"),
)
_AMO_VARIABLE_TRIALS = {
    5: tuple(f"n={value}" for value in range(1, 21)),
    37: tuple(f"a={value},b={value + 1},c={value + 2}" for value in range(2, 19)),
}
_NUMERIC_PERCENTAGE_PATTERN = re.compile(r"(?<![\w.])([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(?:\\%|%)")


def _get_pool():
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn"))
    return _pool


def _normalize_percentage_for_math_verify(text: str) -> str:
    """Work around Math-Verify 0.9.0 parsing escaped percentages inside ``\\boxed{}``."""
    return _NUMERIC_PERCENTAGE_PATTERN.sub(lambda match: f"({match.group(1)}/100)", text)


def _extract_amo_prediction(model_output: str, answer_type: str) -> str:
    prediction = _normalize_percentage_for_math_verify(model_output).replace("：", ": ")
    for end_tag in _AMO_THINK_END_TAGS:
        prediction = prediction.split(end_tag)[-1].strip()

    lowered = prediction.lower()
    for prefix in (*_AMO_ANSWER_PREFIXES, *(prefix[4:] for prefix in _AMO_ANSWER_PREFIXES)):
        if prefix in lowered:
            lowered_suffix = lowered.split(prefix)[-1]
            prediction = prediction[-len(lowered_suffix) :].strip()
            break

    for command in _AMO_SIZE_COMMANDS:
        prediction = prediction.replace(command, "")
    for old, new in _AMO_REPLACEMENTS:
        prediction = prediction.replace(old, new)
    while " }" in prediction:
        prediction = prediction.replace(" }", "}")
    while ".}" in prediction:
        prediction = prediction.replace(".}", "}")

    prediction = prediction.replace(r"\,", "").replace(r"\;", "").replace("\n", " ")
    if answer_type in {"number", "variable"}:
        prediction = (
            prediction.replace(",", "").replace(r"\{", "(").replace(r"\}", ")").replace(r"\[", "(").replace(r"\]", ")")
        )
    return prediction.strip()


def _cut_amo_prediction(prediction: str) -> str:
    for marker in _AMO_CUT_MARKERS:
        prediction = prediction.split(marker)[0].strip()
    return prediction


def _verify_amo_number_or_set(prediction: str, ground_truth_boxed: str) -> bool:
    from math_verify import parse, verify

    parsed_prediction = parse(prediction)
    parsed_gold = parse(ground_truth_boxed)
    result = bool(
        verify(parsed_gold, parsed_prediction, float_rounding=4)
        or verify(parsed_prediction, parsed_gold, float_rounding=4)
    )

    if parsed_prediction and isinstance(parsed_prediction[-1], str) and "=" in parsed_prediction[-1]:
        final_expression = parsed_prediction[-1].split("=")[-1]
        parsed_final_expression = parse("\\boxed{" + final_expression + "}")
        result = result or bool(
            verify(parsed_gold, parsed_final_expression, float_rounding=4)
            or verify(parsed_final_expression, parsed_gold, float_rounding=4)
        )
    return result


def _extract_variable_expression(parsed: list[Any]) -> str | None:
    if not parsed:
        return None
    expression = str(parsed[-1])
    if "\\qquad" in expression:
        expression = expression.split("\\qquad")[-2].strip()
    if "\\quad" in expression:
        expression = expression.split("\\quad")[-2].strip()
    return expression.split("=")[-1]


def _solution_value(solution: Any, variable_name: str) -> Any | None:
    if isinstance(solution, list):
        if not solution:
            return None
        solution = solution[0]
    if not isinstance(solution, dict):
        return None
    for symbol, value in solution.items():
        if str(symbol) == variable_name:
            return value
    return None


def _verify_amo_variable(
    prediction: str,
    ground_truth_boxed: str,
    *,
    question_id: int,
) -> bool:
    from math_verify import parse, verify
    from sympy import solve

    trials = _AMO_VARIABLE_TRIALS.get(question_id)
    if not trials:
        raise ValueError(f"AMO-Bench variable question {question_id} has no verification trials.")

    predicted_expression = _extract_variable_expression(parse(prediction))
    gold_expression = _extract_variable_expression(parse(ground_truth_boxed))
    if predicted_expression is None or gold_expression is None:
        return False

    for trial in trials:
        parsed_prediction = parse(f"\\boxed{{{trial}, y={predicted_expression}}}")
        parsed_gold = parse(f"\\boxed{{{trial}, y={gold_expression}}}")
        if not parsed_prediction or not parsed_gold:
            return False
        try:
            predicted_solution = solve(parsed_prediction[0])
            gold_solution = solve(parsed_gold[0])
        except Exception:
            return False

        predicted_y = _solution_value(predicted_solution, "y")
        gold_y = _solution_value(gold_solution, "y")
        if predicted_y is None or gold_y is None:
            return False
        predicted_y = predicted_y.evalf()
        gold_y = gold_y.evalf()
        if not (verify(gold_y, predicted_y, float_rounding=8) or verify(predicted_y, gold_y, float_rounding=8)):
            return False
    return True


def _verify_amo_parser_answer(
    *,
    ground_truth_boxed: str,
    model_output: str,
    answer_type: str,
    question_id: int,
) -> float:
    answer_type = answer_type.strip().lower()
    if answer_type not in {"number", "set", "variable"}:
        raise ValueError(f"AMO-Bench-P only supports parser-graded number/set/variable answers, got {answer_type!r}.")

    ground_truth_boxed = _normalize_percentage_for_math_verify(ground_truth_boxed)
    prediction = _extract_amo_prediction(model_output, answer_type)
    candidates = (prediction, _cut_amo_prediction(prediction))
    for candidate in dict.fromkeys(candidates):
        if answer_type == "variable":
            verified = _verify_amo_variable(
                candidate,
                ground_truth_boxed,
                question_id=question_id,
            )
        else:
            verified = _verify_amo_number_or_set(candidate, ground_truth_boxed)
        if verified:
            return 1.0
    return 0.0


def _verify_in_subprocess(
    ground_truth_boxed: str,
    model_output: str,
    grader: str | None = None,
    answer_type: str | None = None,
    question_id: int | None = None,
) -> float:
    """Run math_verify in a subprocess where signal.alarm() works."""
    ground_truth_boxed = _normalize_percentage_for_math_verify(ground_truth_boxed)
    model_output = _normalize_percentage_for_math_verify(model_output)
    if grader == "amo_bench":
        if answer_type is None or question_id is None:
            raise ValueError("AMO-Bench scoring requires answer_type and question_id metadata.")
        return _verify_amo_parser_answer(
            ground_truth_boxed=ground_truth_boxed,
            model_output=model_output,
            answer_type=answer_type,
            question_id=int(question_id),
        )

    from math_verify.grader import verify
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, parse

    gold_targets = (LatexExtractionConfig(),)
    pred_targets = (ExprExtractionConfig(), LatexExtractionConfig())

    extracted_gold = parse(ground_truth_boxed, gold_targets)
    extracted_pred = parse(model_output, pred_targets)
    if extracted_gold and extracted_pred:
        return max(1.0 if any(verify(g, p) for g in extracted_gold) else 0.0 for p in extracted_pred)
    return 0.0


def compute_score(
    model_output: str,
    ground_truth: str,
    timeout_score: float = 0,
    timeout: float = 30.0,
    extra_info: dict[str, Any] | None = None,
) -> float:
    ret_score = 0.0
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    grader = extra_info.get("grader") if isinstance(extra_info, dict) else None
    answer_type = extra_info.get("answer_type") if isinstance(extra_info, dict) else None
    question_id = extra_info.get("question_id") if isinstance(extra_info, dict) else None
    try:
        future = _get_pool().submit(
            _verify_in_subprocess,
            ground_truth_boxed,
            model_output,
            grader,
            answer_type,
            question_id,
        )
        ret_score = future.result(timeout=timeout)
    except (FuturesTimeoutError, TimeoutException):
        ret_score = timeout_score
    except Exception as e:
        print(f"Error in math_verify compute_score: {e}")
    return ret_score
