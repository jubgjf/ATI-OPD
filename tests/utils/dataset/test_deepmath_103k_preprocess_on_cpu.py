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

import datasets
import pytest

from examples.data_preprocess.deepmath_103k import (
    BOXED_MATH_INSTRUCTION,
    DATA_SOURCE,
    build_prompt,
    build_stratified_split,
    build_train_test_split,
    parse_validation_size,
    resolve_validation_count,
    topic_group,
    validate_source_dataset,
)


def _source_row(
    source_index: int,
    difficulty: float,
    broad_topic: str,
    final_answer: str | None = None,
) -> dict:
    return {
        "question": f"Question {source_index}",
        "final_answer": final_answer if final_answer is not None else str(source_index),
        "difficulty": difficulty,
        "topic": f"Mathematics -> {broad_topic} -> Synthetic",
        "r1_solution_1": f"long solution {source_index}",
        "r1_solution_2": f"long solution {source_index}",
        "r1_solution_3": f"long solution {source_index}",
    }


def _synthetic_source_dataset(include_invalid_row: bool = False) -> datasets.Dataset:
    rows = []
    source_index = 0
    for size, difficulty, broad_topic in (
        (20, 3.0, "Algebra"),
        (10, 3.0, "Geometry"),
        (8, 7.0, "Algebra"),
        (1, 9.0, "Number Theory"),
    ):
        for _ in range(size):
            rows.append(_source_row(source_index, difficulty, broad_topic))
            source_index += 1
    if include_invalid_row:
        rows.append(_source_row(source_index, 5.0, "Number Theory", final_answer=""))
    return datasets.Dataset.from_list(rows)


def _source_indices(dataset: datasets.Dataset) -> list[int]:
    return [int(value) for value in dataset["source_index"]]


def test_build_prompt_uses_one_user_message_without_system_prompt_on_cpu():
    prompt = build_prompt("What is 6 * 7?")

    assert prompt == [
        {
            "role": "user",
            "content": f"What is 6 * 7?\n\n{BOXED_MATH_INSTRUCTION}",
        }
    ]
    assert build_prompt(prompt[0]["content"]) == prompt
    assert build_prompt("What is 6 * 7?", prompt_style="raw") == [{"role": "user", "content": "What is 6 * 7?"}]


def test_topic_group_uses_broad_topic_level_on_cpu():
    assert topic_group("Mathematics -> Number Theory -> Prime Numbers") == "Number Theory"
    assert topic_group("Mathematics") == "Other"
    assert topic_group("Physics -> Mechanics") == "Physics"


def test_validate_source_dataset_drops_and_audits_invalid_answers_on_cpu():
    raw_dataset = _synthetic_source_dataset(include_invalid_row=True)

    valid_dataset, dropped_rows = validate_source_dataset(raw_dataset, invalid_row_policy="drop")

    assert len(valid_dataset) == 39
    assert set(valid_dataset.column_names) == {
        "question",
        "final_answer",
        "difficulty",
        "topic",
        "source_index",
    }
    assert dropped_rows == [
        {
            "source_index": 39,
            "reasons": ["empty final_answer"],
            "question": "Question 39",
            "difficulty": 5.0,
            "topic": "Mathematics -> Number Theory -> Synthetic",
        }
    ]


def test_validate_source_dataset_error_policy_reports_source_index_on_cpu():
    raw_dataset = _synthetic_source_dataset(include_invalid_row=True)

    with pytest.raises(ValueError, match=r"1 invalid rows.*source_index=39.*empty final_answer"):
        validate_source_dataset(raw_dataset, invalid_row_policy="error")


def test_stratified_split_is_exact_deterministic_and_keeps_singletons_in_train_on_cpu():
    valid_dataset, _ = validate_source_dataset(_synthetic_source_dataset())

    train_a, validation_a, report_a = build_stratified_split(valid_dataset, validation_size=8, seed=17)
    train_b, validation_b, report_b = build_stratified_split(valid_dataset, validation_size=8, seed=17)
    _, validation_c, _ = build_stratified_split(valid_dataset, validation_size=8, seed=18)

    train_indices = set(_source_indices(train_a))
    validation_indices = set(_source_indices(validation_a))
    assert len(train_a) == 31
    assert len(validation_a) == 8
    assert train_indices.isdisjoint(validation_indices)
    assert train_indices | validation_indices == set(range(39))
    assert _source_indices(train_a) == _source_indices(train_b)
    assert _source_indices(validation_a) == _source_indices(validation_b)
    assert report_a == report_b
    assert set(_source_indices(validation_c)) != validation_indices

    singleton_index = 38
    assert singleton_index in train_indices
    assert singleton_index not in validation_indices
    assert report_a["singleton_strata_kept_in_train"] == 1
    assert report_a["validation_rows"] == 8
    assert {(row["difficulty"], row["topic_group"]): row["validation"] for row in report_a["strata"]} == {
        ("3", "Algebra"): 4,
        ("3", "Geometry"): 2,
        ("7", "Algebra"): 2,
        ("9", "Number Theory"): 0,
    }


def test_build_train_test_split_emits_verl_schema_and_split_metadata_on_cpu():
    dataset_dict = datasets.DatasetDict({"train": _synthetic_source_dataset(include_invalid_row=True)})

    train_dataset, test_dataset, report, dropped_rows = build_train_test_split(
        dataset_dict,
        validation_size=8,
        seed=17,
    )

    assert len(train_dataset) == 31
    assert len(test_dataset) == 8
    assert len(dropped_rows) == 1
    assert report["raw_rows"] == 40
    assert report["valid_rows"] == 39
    assert report["dropped_rows"] == 1
    assert set(train_dataset.column_names) == {
        "data_source",
        "prompt",
        "ability",
        "reward_model",
        "extra_info",
    }

    row = train_dataset[0]
    assert row["data_source"] == "math_verify"
    assert row["ability"] == "MATH"
    assert row["prompt"][0]["role"] == "user"
    assert row["prompt"][0]["content"].endswith(BOXED_MATH_INSTRUCTION)
    assert row["reward_model"]["style"] == "rule"
    assert row["reward_model"]["ground_truth"]
    assert row["extra_info"]["split"] == "train"
    assert row["extra_info"]["source_dataset"] == DATA_SOURCE
    assert row["extra_info"]["reward_data_source"] == "math_verify"
    assert row["extra_info"]["stratification_key"].startswith("difficulty=")
    assert all(example["extra_info"]["split"] == "test" for example in test_dataset)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("5000", 5000),
        ("0.05", 0.05),
        ("5e-2", 0.05),
    ],
)
def test_parse_validation_size_accepts_row_count_or_fraction_on_cpu(value, expected):
    assert parse_validation_size(value) == expected


def test_resolve_validation_count_rounds_fraction_and_validates_bounds_on_cpu():
    assert resolve_validation_count(103_020, 0.05) == 5_151
    assert resolve_validation_count(103_020, 5_000) == 5_000

    with pytest.raises(ValueError, match="must be non-empty"):
        resolve_validation_count(10, 0)
    with pytest.raises(ValueError, match="must be an integer row count or a fraction"):
        resolve_validation_count(10, 1.5)
