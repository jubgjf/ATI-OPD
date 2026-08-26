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

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from examples.data_preprocess.nemotron_cascade_2_rl_mopd import (
    ACEREASON,
    DATASET_REVISION,
    EXPECTED_CLEAN_DOMAIN_COUNTS,
    EXPECTED_CLEAN_ROWS,
    EXPECTED_IMPOSSIBLE_IF_DROPS,
    EXPECTED_TEST_DOMAIN_COUNTS,
    EXPECTED_TEST_ROWS,
    EXPECTED_TRAIN_DOMAIN_COUNTS,
    EXPECTED_TRAIN_ROWS,
    EXPECTED_WORKPLACE_DROPS,
    INSTRUCTION_FOLLOWING,
    REWARD_DATA_SOURCE,
    SOURCE_BYTES,
    SOURCE_ROWS,
    SOURCE_SHA256,
    STEM_MCQA,
    STRUCTURED_OUTPUT,
    WORKPLACE,
    build_domain_stratified_split,
    clean_source_rows,
    detect_impossible_if_paragraph_constraints,
    load_validated_source_rows,
    preprocess_source_file,
    to_verl_record,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
LOCAL_SOURCE_FILE = REPO_ROOT / "dataset/Nemotron-Cascade-2-RL-data/MOPD/train.jsonl"


def _prompt(content: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "System text that must remain byte-for-byte unchanged."},
        {"role": "user", "content": content},
    ]


def _if_row(num_paragraphs: int, nth_paragraph: int, content: str = "Answer carefully.") -> dict:
    return {
        "dataset": INSTRUCTION_FOLLOWING,
        "instruction_id_list": ["length_constraints:nth_paragraph_first_word"],
        "kwargs": [
            {
                "num_paragraphs": num_paragraphs,
                "nth_paragraph": nth_paragraph,
                "first_word": "start",
            }
        ],
        "responses_create_params": {"input": _prompt(content), "tools": [], "parallel_tool_calls": False},
    }


def _source_row(domain: str, index: int, content: str | None = None) -> dict:
    prompt = _prompt(content or f"Prompt {domain} {index}")
    common = {"dataset": domain, "responses_create_params": {"input": prompt, "tools": []}}
    if domain == ACEREASON:
        return {**common, "expected_answer": str(index)}
    if domain == INSTRUCTION_FOLLOWING:
        return {
            **common,
            "instruction_id_list": ["keywords:existence"],
            "kwargs": [{"keywords": ["required"]}],
        }
    if domain == STRUCTURED_OUTPUT:
        return {**common, "schema_str": '{"type":"object"}', "schema_type": "json"}
    if domain == STEM_MCQA:
        return {
            **common,
            "expected_answer": "A",
            "uuid": f"uuid-{index}",
            "template_metadata": {"output_regex": "Answer: ([A-Z])"},
        }
    if domain == WORKPLACE:
        return common
    raise AssertionError(f"Unsupported synthetic domain: {domain}")


def test_detect_impossible_if_paragraph_constraint_uses_structured_metadata_on_cpu():
    impossible = _if_row(num_paragraphs=3, nth_paragraph=4)
    satisfiable = _if_row(num_paragraphs=4, nth_paragraph=4)

    assert detect_impossible_if_paragraph_constraints(impossible, source_index=7) == [
        {
            "constraint_id": "length_constraints:nth_paragraph_first_word",
            "num_paragraphs": 3,
            "nth_paragraph": 4,
            "first_word": "start",
        }
    ]
    assert detect_impossible_if_paragraph_constraints(satisfiable, source_index=8) == []
    assert detect_impossible_if_paragraph_constraints(_source_row(ACEREASON, 0), source_index=9) == []


def test_cleaning_drops_only_workplace_and_detected_impossible_if_on_cpu():
    rows = [
        _source_row(ACEREASON, 0),
        _source_row(WORKPLACE, 1),
        _if_row(num_paragraphs=2, nth_paragraph=3),
        _if_row(num_paragraphs=2, nth_paragraph=2),
        _source_row(STRUCTURED_OUTPUT, 4),
    ]

    clean, audit = clean_source_rows(rows, strict_current_source=False)

    assert [row["_source_index"] for row in clean] == [0, 3, 4]
    assert audit["dropped_by_reason"] == {
        "unsupported_workplace_tool_environment": 1,
        "impossible_if_paragraph_constraint": 1,
    }
    assert audit["impossible_if_rows"][0]["source_index"] == 2
    assert audit["impossible_if_rows"][0]["sample_id"].endswith(":2")
    assert audit["workplace_rows"][0]["source_index"] == 1
    assert audit["workplace_rows"][0]["sample_id"].endswith(":1")


def test_verl_record_preserves_prompt_and_encodes_domain_ground_truth_as_json_on_cpu():
    long_content = "prefix\n" + "x" * 50_000 + "\nsuffix"
    source_rows = [
        _source_row(ACEREASON, 0, content=long_content),
        _source_row(INSTRUCTION_FOLLOWING, 1),
        _source_row(STRUCTURED_OUTPUT, 2),
        _source_row(STEM_MCQA, 3),
    ]

    for source_index, source_row in enumerate(source_rows):
        indexed_row = {**source_row, "_source_index": source_index}
        record = to_verl_record(indexed_row, split="train")
        payload = json.loads(record["reward_model"]["ground_truth"])

        assert record["data_source"] == REWARD_DATA_SOURCE
        assert record["prompt"] == source_row["responses_create_params"]["input"]
        assert record["extra_info"]["reward_data_source"] == REWARD_DATA_SOURCE
        assert record["extra_info"]["source_revision"] == DATASET_REVISION
        assert payload["verifier"] in {"math_answer", "instruction_following", "json_schema", "mcqa_regex"}

    assert source_rows[0]["responses_create_params"]["input"][1]["content"] == long_content


def test_domain_stratified_split_is_exact_and_deterministic_on_cpu():
    clean_rows = []
    source_index = 0
    for domain, size in ((ACEREASON, 20), (INSTRUCTION_FOLLOWING, 20), (STRUCTURED_OUTPUT, 10), (STEM_MCQA, 10)):
        for _ in range(size):
            clean_rows.append({**_source_row(domain, source_index), "_source_index": source_index})
            source_index += 1

    train_a, test_a, report_a = build_domain_stratified_split(
        clean_rows,
        test_fraction=0.2,
        seed=42,
        strict_current_source=False,
    )
    train_b, test_b, report_b = build_domain_stratified_split(
        clean_rows,
        test_fraction=0.2,
        seed=42,
        strict_current_source=False,
    )
    _, test_c, _ = build_domain_stratified_split(
        clean_rows,
        test_fraction=0.2,
        seed=43,
        strict_current_source=False,
    )

    assert len(train_a) == 48
    assert len(test_a) == 12
    assert [row["_source_index"] for row in train_a] == [row["_source_index"] for row in train_b]
    assert [row["_source_index"] for row in test_a] == [row["_source_index"] for row in test_b]
    assert [row["_source_index"] for row in test_a] != [row["_source_index"] for row in test_c]
    assert report_a == report_b
    assert {domain: values["test"] for domain, values in report_a["domain_counts"].items()} == {
        ACEREASON: 4,
        INSTRUCTION_FOLLOWING: 4,
        STRUCTURED_OUTPUT: 2,
        STEM_MCQA: 2,
    }


def test_source_identity_validation_checks_hash_bytes_and_physical_rows_on_cpu(tmp_path):
    source_file = tmp_path / "train.jsonl"
    payload = b'{"dataset":"synthetic"}\n'
    source_file.write_bytes(payload)
    expected_hash = hashlib.sha256(payload).hexdigest()

    rows, identity = load_validated_source_rows(
        source_file,
        expected_sha256=expected_hash,
        expected_bytes=len(payload),
        expected_rows=1,
    )
    assert rows == [{"dataset": "synthetic"}]
    assert identity["sha256"] == expected_hash

    with pytest.raises(ValueError, match="source identity mismatch"):
        load_validated_source_rows(
            source_file,
            expected_sha256="0" * 64,
            expected_bytes=len(payload),
            expected_rows=1,
        )


@pytest.mark.skipif(not LOCAL_SOURCE_FILE.is_file(), reason="local Nemotron MOPD source file is not available")
def test_pinned_local_source_has_exact_clean_split_and_prompt_preservation_on_cpu():
    train_dataset, test_dataset, audit = preprocess_source_file(LOCAL_SOURCE_FILE)

    assert audit["source"]["revision"] == DATASET_REVISION
    assert audit["source"]["sha256"] == SOURCE_SHA256
    assert audit["source"]["bytes"] == SOURCE_BYTES
    assert audit["source"]["rows"] == SOURCE_ROWS
    assert audit["cleaning"]["clean_rows"] == EXPECTED_CLEAN_ROWS
    assert audit["cleaning"]["clean_domain_counts"] == EXPECTED_CLEAN_DOMAIN_COUNTS
    assert audit["cleaning"]["dropped_by_reason"] == {
        "unsupported_workplace_tool_environment": EXPECTED_WORKPLACE_DROPS,
        "impossible_if_paragraph_constraint": EXPECTED_IMPOSSIBLE_IF_DROPS,
    }
    assert len(train_dataset) == EXPECTED_TRAIN_ROWS
    assert len(test_dataset) == EXPECTED_TEST_ROWS
    assert {domain: values["train"] for domain, values in audit["split"]["domain_counts"].items()} == (
        EXPECTED_TRAIN_DOMAIN_COUNTS
    )
    assert {domain: values["test"] for domain, values in audit["split"]["domain_counts"].items()} == (
        EXPECTED_TEST_DOMAIN_COUNTS
    )
    assert set(train_dataset.column_names) == {
        "data_source",
        "prompt",
        "ability",
        "reward_model",
        "extra_info",
    }
    assert audit["output"]["token_length_filtering"] is False
    assert audit["output"]["truncation"] is False

    raw_rows, _ = load_validated_source_rows(LOCAL_SOURCE_FILE)
    emitted = list(train_dataset) + list(test_dataset)
    emitted_counts = Counter(row["extra_info"]["source_domain"] for row in emitted)
    assert emitted_counts == Counter(EXPECTED_CLEAN_DOMAIN_COUNTS)
    assert all(
        row["prompt"] == raw_rows[row["extra_info"]["source_index"]]["responses_create_params"]["input"]
        for row in emitted
    )
