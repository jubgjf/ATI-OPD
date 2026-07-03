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

import json

import datasets
import pytest

from examples.data_preprocess.dapo_math_17k import (
    DATA_SOURCE,
    QWEN3_MATH_BOXED_INSTRUCTION,
    build_qwen3_boxed_prompt,
    clean_qwen3_math_problem_text,
    normalize_extra_info,
    normalize_prompt,
    normalize_reward_model,
    validate_and_tag_dataset,
)


def test_normalize_prompt_accepts_list_dict_and_json_on_cpu():
    expected = [{"role": "user", "content": "Solve x."}]

    assert normalize_prompt(expected, idx=0) == expected
    assert normalize_prompt({"role": "user", "content": "Solve x."}, idx=0) == expected
    assert normalize_prompt(json.dumps(expected), idx=0) == expected


def test_normalize_prompt_rejects_assistant_or_empty_content_on_cpu():
    with pytest.raises(ValueError, match="invalid prompt message"):
        normalize_prompt({"role": "assistant", "content": "No."}, idx=0)
    with pytest.raises(ValueError, match="invalid prompt message"):
        normalize_prompt({"role": "user", "content": ""}, idx=0)


def test_build_qwen3_boxed_prompt_appends_instruction_on_cpu():
    prompt = build_qwen3_boxed_prompt("What is 1+1?", idx=0)

    assert prompt == [
        {
            "role": "user",
            "content": f"What is 1+1?\n\n{QWEN3_MATH_BOXED_INSTRUCTION}",
        }
    ]


def test_build_qwen3_boxed_prompt_allows_bracket_prefixed_problem_text_on_cpu():
    prompt = build_qwen3_boxed_prompt("[i]Greedy Algorithms[/i] asks a question.", idx=0)

    assert prompt[0]["content"].startswith("[i]Greedy Algorithms[/i] asks a question.")
    assert prompt[0]["content"].endswith(QWEN3_MATH_BOXED_INSTRUCTION)


def test_clean_qwen3_math_problem_text_removes_legacy_answer_suffix_on_cpu():
    text = (
        "求 1+1。 "
        "让我们一步一步地思考。请以“Answer: \\boxed{<final_answer>}”的格式输出最终答案。"
        "如果是选择题，请按顺序输出正确的选项，不带任何标点或空格。"
        "对于其他类型的问题，请只输出最终答案的数值。"
    )

    assert clean_qwen3_math_problem_text(text) == "求 1+1。"


def test_normalize_reward_model_accepts_dict_and_json_on_cpu():
    expected = {"style": "rule", "ground_truth": "42"}

    assert normalize_reward_model(expected, idx=0) == expected
    assert normalize_reward_model(json.dumps(expected), idx=0) == expected


def test_normalize_extra_info_accepts_dict_json_and_none_on_cpu():
    expected = {"index": "abc"}

    assert normalize_extra_info(expected, idx=0) == expected
    assert normalize_extra_info(json.dumps(expected), idx=0) == expected
    assert normalize_extra_info(None, idx=0) == {}


def test_validate_and_tag_dataset_outputs_verl_rl_schema_on_cpu():
    dataset = datasets.Dataset.from_list(
        [
            {
                "data_source": "math_dapo",
                "prompt": "What is 6*7?",
                "source_prompt": [{"role": "user", "content": "Solve step by step. What is 6*7?"}],
                "ability": "MATH",
                "reward_model": json.dumps({"style": "rule", "ground_truth": "42"}),
                "extra_info": {"original_index": 7},
            }
        ]
    )

    processed = validate_and_tag_dataset(dataset, split="train")
    row = processed[0]

    assert set(processed.column_names) == {"data_source", "prompt", "ability", "reward_model", "extra_info"}
    assert row["data_source"] == "math_dapo"
    assert row["prompt"] == [{"role": "user", "content": "Solve step by step. What is 6*7?"}]
    assert row["reward_model"] == {"style": "rule", "ground_truth": "42"}
    assert row["extra_info"]["split"] == "train"
    assert row["extra_info"]["source_dataset"] == DATA_SOURCE
    assert row["extra_info"]["source_data_source"] == "math_dapo"
    assert row["extra_info"]["prompt_style"] == "dapo"


def test_validate_and_tag_dataset_qwen3_boxed_uses_raw_prompt_column_on_cpu():
    dataset = datasets.Dataset.from_list(
        [
            {
                "data_source": "math_dapo",
                "prompt": "What is 6*7?",
                "source_prompt": [
                    {
                        "role": "user",
                        "content": (
                            "Solve step by step. What is 6*7?\n"
                            'Remember to put your answer on its own line after "Answer:".'
                        ),
                    }
                ],
                "ability": "MATH",
                "reward_model": json.dumps({"style": "rule", "ground_truth": "42"}),
                "extra_info": {"original_index": 7},
            }
        ]
    )

    processed = validate_and_tag_dataset(dataset, split="train", prompt_style="qwen3_boxed")
    row = processed[0]
    content = row["prompt"][0]["content"]

    assert row["prompt"][0]["role"] == "user"
    assert content == f"What is 6*7?\n\n{QWEN3_MATH_BOXED_INSTRUCTION}"
    assert "Answer:" not in content
    assert row["extra_info"]["prompt_style"] == "qwen3_boxed"


def test_validate_and_tag_dataset_accepts_problem_and_solution_fallback_on_cpu():
    dataset = datasets.Dataset.from_list(
        [
            {
                "problem": "What is 0?",
                "solution": 0,
                "extra_info": json.dumps({"original_index": "zero-answer"}),
            }
        ]
    )

    processed = validate_and_tag_dataset(dataset, split="test")
    row = processed[0]

    assert set(processed.column_names) == {"data_source", "prompt", "ability", "reward_model", "extra_info"}
    assert row["data_source"] == "math_dapo"
    assert row["prompt"] == [{"role": "user", "content": "What is 0?"}]
    assert row["reward_model"] == {"style": "rule", "ground_truth": "0"}
    assert row["extra_info"]["original_index"] == "zero-answer"
    assert row["extra_info"]["source_dataset"] == DATA_SOURCE
