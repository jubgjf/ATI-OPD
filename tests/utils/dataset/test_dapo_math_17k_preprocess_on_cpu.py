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


def test_normalize_reward_model_accepts_dict_and_json_on_cpu():
    expected = {"style": "rule", "ground_truth": "42"}

    assert normalize_reward_model(expected, idx=0) == expected
    assert normalize_reward_model(json.dumps(expected), idx=0) == expected


def test_validate_and_tag_dataset_outputs_verl_rl_schema_on_cpu():
    dataset = datasets.Dataset.from_list(
        [
            {
                "data_source": "math_dapo",
                "prompt": json.dumps({"role": "user", "content": "What is 6*7?"}),
                "ability": "MATH",
                "reward_model": json.dumps({"style": "rule", "ground_truth": "42"}),
                "extra_info": {"original_index": 7},
            }
        ]
    )

    processed = validate_and_tag_dataset(dataset, split="train")
    row = processed[0]

    assert row["prompt"] == [{"role": "user", "content": "What is 6*7?"}]
    assert row["reward_model"] == {"style": "rule", "ground_truth": "42"}
    assert row["extra_info"]["split"] == "train"
    assert row["extra_info"]["source_dataset"] == "BytedTsinghua-SIA/DAPO-Math-17k"
