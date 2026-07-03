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

from verl.utils.reward_score import math_dapo


def _assert_correct(solution: str, ground_truth: str = "36"):
    score = math_dapo.compute_score(solution, ground_truth)

    assert score["acc"] is True
    assert score["pred"] == ground_truth


def test_math_dapo_accepts_answer_labels():
    _assert_correct("Work\n\nAnswer: 36")
    _assert_correct("Work\n\n**Answer:** 36")
    _assert_correct("Work\n\n**Answer**: 36")
    _assert_correct("Work\n\nFinal Answer: 36")
    _assert_correct("Work\n\n**Final Answer**: 36")
    _assert_correct("Work\n\n答案：36")


def test_math_dapo_accepts_boxed_answers():
    _assert_correct("Work\n\n### Final Answer\n\n$$\n\\boxed{36}\n$$")
    _assert_correct("Work\n\nThe final result is \\boxed 36.")
    _assert_correct("Work\n\nThe final result is \\fbox{36}.")


def test_math_dapo_normalizes_qwen_latex_variants():
    _assert_correct(r"Work\n\n$$\boxed{\displaystyle 36}$$")
    _assert_correct(r"Work\n\n$$\boxed{\left(36\right)}$$", "(36)")
    _assert_correct(r"Work\n\n$$\boxed{\dfrac{1}{2}}$$", r"\frac{1}{2}")


def test_math_dapo_accepts_answer_label_with_boxed_answer():
    _assert_correct("Work\n\nAnswer: \\boxed{36}")


def test_math_dapo_accepts_latex_text_answer_labels():
    _assert_correct(r"Work\n\n### Final Answer\n\n$$ \text{Answer: } 36 $$")
    _assert_correct(r"Work\n\n### Final Answer\n\n$$ \mathrm{Answer: 36} $$")
    _assert_correct(r"Work\n\n$$\boxed{\text{Answer: } 36}$$")
    _assert_correct(r"Work\n\n$$ \text{Answer: } \boxed{36} $$")


def test_math_dapo_uses_complete_boxed_answer_after_latex_text_label():
    _assert_correct(r"Work\n\n$$ \text{Answer: } \frac{25}{216}, \quad m+n=\boxed{36} $$")
    _assert_correct(r"Work\n\n$$ \text{Answer: } \boxed{36} $$")


def test_math_dapo_prefers_later_boxed_answer_when_final_answer_label_is_verbose():
    _assert_correct("Work\n\nFinal Answer: There are 36 valid arrangements.\n\n$$\\boxed{36}$$")
    _assert_correct("Work\n\n**Answer:** After simplifying the expression, the answer is 36.\n\n\\fbox{36}")


def test_math_dapo_uses_late_boxed_answer_outside_previous_300_char_tail():
    _assert_correct("Work\n\n\\boxed{36}\n" + ("trailing explanation\n" * 30))


def test_math_dapo_keeps_invalid_without_answer():
    score = math_dapo.compute_score("Work only", "36")

    assert score["acc"] is False
    assert score["pred"] == "[INVALID]"


def test_math_dapo_returns_format_diagnostics():
    score = math_dapo.compute_score("Work\n\nAnswer: \\boxed{36}", "36")

    assert score["invalid_pred"] is False
    assert score["has_answer_pattern"] is True
    assert score["has_boxed_pattern"] is True
    assert score["has_accepted_answer_format"] is True
