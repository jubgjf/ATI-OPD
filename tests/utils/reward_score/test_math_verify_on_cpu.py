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

import pytest

from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score import math_verify as math_verify_reward


@pytest.mark.parametrize(
    ("answer_type", "question_id", "ground_truth", "model_output"),
    [
        ("number", 1, "42", r"### The final answer is: $\boxed{42}$"),
        ("set", 43, r"\{40,64,75,100,150\}", r"### The final answer is: $\boxed{\{40,64,75,100,150\}}$"),
    ],
)
def test_amo_bench_number_and_set_use_official_math_verify_protocol_on_cpu(
    answer_type,
    question_id,
    ground_truth,
    model_output,
):
    score = math_verify_reward._verify_in_subprocess(
        "\\boxed{" + ground_truth + "}",
        model_output,
        "amo_bench",
        answer_type,
        question_id,
    )

    assert score == 1.0


@pytest.mark.parametrize(
    ("question_id", "answer"),
    [
        (5, r"\dfrac{(2+\sqrt{3})^{2^{2-n}}+(2-\sqrt{3})^{2^{2-n}}}{2}"),
        (
            37,
            (
                r"\frac{(a-b)(a-c)(b-c)}{8}\cdot "
                r"\left(\frac{(a+b-c)(a-b+c)(-a+b+c)}"
                r"{2abc-(a+b-c)(a-b+c)(-a+b+c)}\right)^3"
            ),
        ),
    ],
)
def test_amo_bench_variable_answers_use_official_trials_on_cpu(question_id, answer):
    score = math_verify_reward._verify_in_subprocess(
        "\\boxed{" + answer + "}",
        f"### The final answer is: $\\boxed{{{answer}}}$",
        "amo_bench",
        "variable",
        question_id,
    )

    assert score == 1.0


def test_default_reward_route_keeps_math_verify_as_accuracy_source_on_cpu(monkeypatch):
    monkeypatch.setattr(math_verify_reward, "compute_score", lambda *args, **kwargs: 1.0)

    result = default_compute_score(
        data_source="math-ai/aime24",
        solution_str=r"The final answer is \boxed{27}.",
        ground_truth="27",
        extra_info={"reward_data_source": "math_verify"},
    )

    assert result["score"] == 1.0
    assert result["acc"] is True
    assert result["math_verify_score"] == 1.0


@pytest.mark.parametrize(
    ("grader", "answer_type", "question_id"),
    [
        (None, None, None),
        ("amo_bench", "number", 1),
    ],
)
def test_math_verify_090_accepts_escaped_boxed_percentage_on_cpu(grader, answer_type, question_id):
    score = math_verify_reward._verify_in_subprocess(
        r"\boxed{0.09}",
        r"### The final answer is: $\boxed{9\%}$",
        grader,
        answer_type,
        question_id,
    )

    assert score == 1.0


def test_math_verify_percentage_compatibility_does_not_confuse_percent_with_integer_on_cpu():
    score = math_verify_reward._verify_in_subprocess(
        r"\boxed{9}",
        r"### The final answer is: $\boxed{9\%}$",
    )

    assert score == 0.0
