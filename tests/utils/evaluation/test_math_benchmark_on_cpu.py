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

import pandas as pd
import pytest

from verl.utils.evaluation.math_benchmark import (
    AMO_BENCH_P_SIZE,
    build_math_chat_messages,
    load_benchmark_samples,
    normalize_ground_truth,
    score_completion,
    split_final_content,
)


def test_load_benchmark_samples_normalizes_local_schemas_on_cpu(tmp_path):
    dataset_root = tmp_path / "dataset"
    (dataset_root / "aime24").mkdir(parents=True)
    (dataset_root / "amc23").mkdir()
    (dataset_root / "aime24" / "test.jsonl").write_text(
        json.dumps({"id": "0", "problem": "AIME problem", "answer": 70}) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame([{"id": 3, "question": "AMC problem", "solution": r"\boxed{27}"}]).to_parquet(
        dataset_root / "amc23" / "test-00000-of-00001.parquet"
    )

    assert load_benchmark_samples(dataset_root, "aime24")[0].ground_truth == "70"
    amc_sample = load_benchmark_samples(dataset_root, "amc23")[0]
    assert amc_sample.problem == "AMC problem"
    assert amc_sample.ground_truth == "27"


def test_load_amo_bench_uses_official_parser_subset_from_hf_data_layout_on_cpu(tmp_path):
    dataset_root = tmp_path / "dataset"
    benchmark_dir = dataset_root / "meituan-longcat" / "AMO-Bench" / "data"
    benchmark_dir.mkdir(parents=True)
    description_ids = {2, 24, 25, 26, 27, 38, 39, 40, 41, 42, 46}
    rows = [
        {
            "question_id": question_id,
            "prompt": f"Official AMO prompt {question_id}",
            "answer": rf"\boxed{{{question_id}}}",
            "answer_type": "description" if question_id in description_ids else "number",
        }
        for question_id in range(1, 51)
    ]
    pd.DataFrame(rows).to_parquet(benchmark_dir / "test-00000-of-00001.parquet")

    samples = load_benchmark_samples(dataset_root, "meituan-longcat/AMO-Bench")

    assert len(samples) == AMO_BENCH_P_SIZE
    assert {int(sample.question_id) for sample in samples}.isdisjoint(description_ids)
    assert samples[0].sample_id == "1"
    assert samples[0].grader == "amo_bench"
    assert samples[0].use_raw_prompt is True
    assert build_math_chat_messages(samples[0].problem, use_raw_prompt=True) == [
        {"role": "user", "content": "Official AMO prompt 1"}
    ]


def test_load_amo_bench_rejects_noncanonical_parser_subset_size_on_cpu(tmp_path):
    benchmark_dir = tmp_path / "AMO-Bench"
    benchmark_dir.mkdir()
    pd.DataFrame(
        [
            {
                "question_id": 1,
                "prompt": "Incomplete AMO dataset",
                "answer": r"\boxed{1}",
                "answer_type": "number",
            }
        ]
    ).to_parquet(benchmark_dir / "test.parquet")

    with pytest.raises(ValueError, match="exactly 39"):
        load_benchmark_samples(tmp_path, "AMO-Bench")


def test_math_response_helpers_ignore_thinking_answer_on_cpu():
    text = r"<think>bad early answer \boxed{0}</think> final answer is \boxed{204}"

    assert split_final_content(text) == r"final answer is \boxed{204}"
    scored = score_completion(text, "204", verifier="math_dapo")
    assert scored.acc is True
    assert scored.pred == "204"
    assert normalize_ground_truth("70.0") == "70"
