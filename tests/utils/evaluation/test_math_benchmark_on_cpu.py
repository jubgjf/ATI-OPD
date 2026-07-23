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

from verl.utils.evaluation.math_benchmark import (
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


def test_math_response_helpers_ignore_thinking_answer_on_cpu():
    text = r"<think>bad early answer \boxed{0}</think> final answer is \boxed{204}"

    assert split_final_content(text) == r"final answer is \boxed{204}"
    scored = score_completion(text, "204", verifier="math_dapo")
    assert scored.acc is True
    assert scored.pred == "204"
    assert normalize_ground_truth("70.0") == "70"
