# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py

import re
from typing import Optional

SOLUTION_TAIL_CHARS = 2048
DEFAULT_ANSWER_PATTERN = r"(?:^|[\s*_`>])(?:final\s+answer|最终答案|answer|答案)\s*(?:[*_`]+\s*)?[:：]\s*([^\n]+)"
_ANSWER_LABEL_RE = re.compile(DEFAULT_ANSWER_PATTERN, re.IGNORECASE | re.MULTILINE)
_LEADING_ANSWER_LABEL_RE = re.compile(r"^(?:final\s+answer|最终答案|answer|答案)\s*[:：]\s*", re.IGNORECASE)
_TEX_TEXT_ANSWER_LABEL_RE = re.compile(
    r"\\(?:text|textbf|mathrm)\s*\{\s*(?:final\s+answer|最终答案|answer|答案)\s*[:：]\s*([^}]*)\}\s*([^\n$}]*)",
    re.IGNORECASE | re.MULTILINE,
)
_BOXED_COMMAND_RE = re.compile(r"\\(?:boxed|fbox)\s*")


def _iter_boxed_expressions(string: str):
    for match in _BOXED_COMMAND_RE.finditer(string):
        content_start = match.end()
        if content_start >= len(string):
            continue

        if string[content_start] == "{":
            i = content_start
            right_brace_idx = None
            num_left_braces_open = 0

            while i < len(string):
                if string[i] == "{":
                    num_left_braces_open += 1
                if string[i] == "}":
                    num_left_braces_open -= 1
                    if num_left_braces_open == 0:
                        right_brace_idx = i
                        break
                i += 1

            if right_brace_idx is not None:
                yield match.start(), right_brace_idx + 1, string[match.start() : right_brace_idx + 1]
            continue

        token_end = content_start
        while token_end < len(string) and not string[token_end].isspace():
            token_end += 1
        if token_end > content_start:
            yield match.start(), token_end, string[match.start() : token_end]


def _last_boxed_match(string: str) -> Optional[tuple[int, int, str]]:
    last_match = None
    for boxed_match in _iter_boxed_expressions(string):
        last_match = boxed_match
    return last_match


def last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last LaTeX boxed/fboxed expression from a string.

    Args:
        string: Input string containing LaTeX code

    Returns:
        The last boxed expression or None if not found. Supports
        ``\\boxed{...}``, ``\\fbox{...}``, and Qwen-style ``\\boxed ...``.
    """
    boxed_match = _last_boxed_match(string)
    if boxed_match is None:
        return None
    return boxed_match[2]


def remove_boxed(s: str) -> str:
    """Remove the LaTeX boxed/fboxed command from a string.

    Args:
        s: String with format "\\boxed{content}", "\\fbox{content}", or "\\boxed content"

    Returns:
        The content inside the boxed command
    """
    match = _BOXED_COMMAND_RE.match(s)
    assert match is not None, f"box error: {s}"
    content_start = match.end()
    if content_start < len(s) and s[content_start] == "{":
        assert s[-1] == "}", f"box error: {s}"
        return s[content_start + 1 : -1]
    return s[content_start:].strip()


def _clean_answer_candidate(candidate: str) -> str:
    candidate = str(candidate).strip()
    candidate = candidate.strip("*_` \t")
    candidate = candidate.strip()
    candidate = re.sub(r"^\$+\s*", "", candidate)
    candidate = re.sub(r"\s*\$+$", "", candidate)
    candidate = candidate.strip("*_` \t")
    candidate = candidate.strip()
    if candidate.endswith(("。", ".", ",", ";", ":")):
        candidate = candidate[:-1].strip()
    return candidate


def _get_answer_label_candidate(match: re.Match) -> str:
    if match.groups():
        return next((part for part in reversed(match.groups()) if part), "")
    return match.group(0)


def _get_tex_text_answer_label_candidate(match: re.Match) -> str:
    in_text = _clean_answer_candidate(match.group(1))
    if in_text:
        candidate = in_text
    else:
        candidate = _clean_answer_candidate(match.group(2))

    # Keep this path conservative: regex cannot reliably capture nested
    # LaTeX commands after \text{Answer: }, so let the boxed extractor handle
    # those instead of returning partial strings such as "\frac{25".
    if re.search(r"\\|[{}]", candidate):
        return ""
    return candidate


def _iter_answer_label_match_candidates(solution_str: str, answer_pattern: str = DEFAULT_ANSWER_PATTERN):
    if answer_pattern == DEFAULT_ANSWER_PATTERN:
        matches = [
            (_get_answer_label_candidate(match), match.start(), match.end())
            for match in _ANSWER_LABEL_RE.finditer(solution_str)
        ]
        matches.extend(
            (_get_tex_text_answer_label_candidate(match), match.start(), match.end())
            for match in _TEX_TEXT_ANSWER_LABEL_RE.finditer(solution_str)
        )
        yield from sorted(matches, key=lambda item: item[1])
        return

    for match in re.finditer(answer_pattern, solution_str):
        yield _get_answer_label_candidate(match), match.start(), match.end()


def _iter_answer_label_matches(solution_str: str, answer_pattern: str = DEFAULT_ANSWER_PATTERN):
    for candidate, _, _ in _iter_answer_label_match_candidates(solution_str, answer_pattern=answer_pattern):
        yield candidate


def _is_compact_answer_candidate(candidate: str) -> bool:
    """Return whether an answer-label capture looks like the final answer itself.

    Qwen3 sometimes writes ``Final Answer:`` followed by a full explanatory
    sentence and then puts the scalar answer in ``\\boxed{...}`` on the next line.
    In that case the boxed answer is more reliable than the label capture.
    """
    candidate = _clean_answer_candidate(candidate)
    if not candidate:
        return False
    if len(candidate) > 80:
        return False
    if re.search(r"[.!?。]\s+\S", candidate):
        return False

    words = re.findall(r"[A-Za-z]{2,}", candidate)
    latex_words = {"frac", "sqrt", "text", "mathrm", "left", "right", "overline", "boxed", "fbox"}
    unit_words = {expr.replace("\\text{", "").replace("}", "").lower() for expr in REMOVED_EXPRESSIONS}
    prose_words = [word for word in words if word.lower() not in latex_words and word.lower() not in unit_words]
    return len(prose_words) <= 3


def extract_final_answer(solution_str: str, answer_pattern: str = DEFAULT_ANSWER_PATTERN) -> str:
    """Extract a final answer accepted by DAPO-Math reward/validation."""
    boxed_match = _last_boxed_match(solution_str)
    boxed_start = boxed_match[0] if boxed_match is not None else -1
    boxed_value = _clean_answer_candidate(remove_boxed(boxed_match[2])) if boxed_match is not None else None

    for candidate, _, label_end in reversed(
        list(_iter_answer_label_match_candidates(solution_str, answer_pattern=answer_pattern))
    ):
        candidate = _clean_answer_candidate(candidate)
        if not candidate:
            continue
        boxed_candidate = last_boxed_only_string(candidate)
        if boxed_candidate is not None:
            return _clean_answer_candidate(remove_boxed(boxed_candidate))
        if boxed_value is not None and boxed_start >= label_end and not _is_compact_answer_candidate(candidate):
            return boxed_value
        return candidate

    if boxed_value is not None:
        return boxed_value
    return "[INVALID]"


def has_answer_label(text: object) -> bool:
    return any(_clean_answer_candidate(match) for match in _iter_answer_label_matches(str(text or "")))


def has_boxed_answer(text: object) -> bool:
    return last_boxed_only_string(str(text or "")) is not None


def has_accepted_answer_format(text: object) -> bool:
    return has_answer_label(text) or has_boxed_answer(text)


# Constants for normalization
SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question.

    Args:
        final_answer: The answer string to normalize

    Returns:
        Normalized answer string
    """
    final_answer = final_answer.split("=")[-1]
    final_answer = _LEADING_ANSWER_LABEL_RE.sub("", final_answer.strip())

    # Apply substitutions and removals
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract and normalize LaTeX math
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\(?:boxed|fbox)\{)(.*)(\})", "\\2", final_answer)
    final_answer = re.sub(r"\\boxed\s+([^\s]+)", "\\1", final_answer)
    final_answer = re.sub(r"\\(?:displaystyle|textstyle|scriptstyle|scriptscriptstyle)\s*", "", final_answer)
    final_answer = final_answer.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    final_answer = final_answer.replace("\\left", "").replace("\\right", "")
    final_answer = _LEADING_ANSWER_LABEL_RE.sub("", final_answer.strip())

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize numbers
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer.strip()


def is_correct_minerva(
    solution_str: str, gt: str, gt_need_extract: bool = False, answer_pattern: str = DEFAULT_ANSWER_PATTERN
) -> tuple[bool, str]:
    """Check if the solution is correct according to Minerva criteria.

    Args:
        solution_str: The solution string to check
        gt: The ground truth answer
        gt_need_extract: Whether the ground truth needs extraction
        answer_pattern: Regex pattern to extract the answer

    Returns:
        Tuple of (is_correct, normalized_prediction)
    """
    extracted_answer = extract_final_answer(solution_str, answer_pattern=answer_pattern)
    pred = normalize_final_answer(extracted_answer)

    # Process ground truth
    if gt_need_extract:
        gt = normalize_final_answer(remove_boxed(last_boxed_only_string(gt)))
    else:
        gt = normalize_final_answer(gt)

    return (pred == gt), pred


def is_correct_strict_box(
    pred: str, gt: str, pause_tokens_index: Optional[list[int]] = None
) -> tuple[int, Optional[str]]:
    """Check if the prediction is correct using strict boxed answer criteria.

    Args:
        pred: The prediction string
        gt: The ground truth answer
        pause_tokens_index: Indices of pause tokens

    Returns:
        Tuple of (score, extracted_prediction)
    """
    # Extract the relevant part of the prediction
    if pause_tokens_index is not None:
        assert len(pause_tokens_index) == 4
        pred = pred[pause_tokens_index[-1] - 100 :]
    else:
        pred = pred[-100:]

    # Extract and check the boxed answer
    boxed_pred = last_boxed_only_string(pred)
    extracted_pred = remove_boxed(boxed_pred) if boxed_pred is not None else None

    return 1 if (extracted_pred == gt) else -1, extracted_pred


def verify(
    solution_str: str, answer: str, strict_box_verify: bool = False, pause_tokens_index: Optional[list[int]] = None
) -> bool:
    """Verify if the solution is correct.

    Args:
        solution_str: The solution string to verify
        answer: The ground truth answer
        strict_box_verify: Whether to use strict box verification
        pause_tokens_index: Indices of pause tokens

    Returns:
        True if the solution is correct, False otherwise
    """
    if strict_box_verify:
        correct, pred = is_correct_strict_box(solution_str, answer, pause_tokens_index)
        return correct == 1, pred

    correct, pred = is_correct_minerva(solution_str, answer)
    return correct, pred


def compute_score(
    solution_str: str,
    ground_truth: str,
    strict_box_verify: bool = False,
    pause_tokens_index: Optional[list[int]] = None,
) -> float:
    """Compute the reward score for a solution.

    Args:
        solution_str: The solution string
        ground_truth: The ground truth answer
        strict_box_verify: Whether to use strict box verification
        pause_tokens_index: Indices of pause tokens

    Returns:
        Reward score (1.0 for correct, -1.0 for incorrect)
    """
    # Limit solution length for efficiency. Keep reward and validation format
    # diagnostics aligned with the exact text used for answer extraction.
    solution_str = solution_str[-SOLUTION_TAIL_CHARS:]
    has_answer_pattern = has_answer_label(solution_str)
    has_boxed_pattern = has_boxed_answer(solution_str)
    has_accepted_format = has_answer_pattern or has_boxed_pattern

    # Verify the solution
    correct, pred = verify(solution_str, ground_truth, strict_box_verify, pause_tokens_index)

    reward = 1.0 if correct else -1.0
    acc = correct

    return {
        "score": reward,
        "acc": acc,
        "pred": pred,
        "invalid_pred": pred == "[INVALID]",
        "has_answer_pattern": has_answer_pattern,
        "has_boxed_pattern": has_boxed_pattern,
        "has_accepted_answer_format": has_accepted_format,
    }
