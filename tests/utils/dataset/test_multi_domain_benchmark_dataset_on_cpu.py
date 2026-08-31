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

from verl.third_party import ifbench as vendored_ifbench
from verl.trainer.ppo.metric_utils import process_validation_metrics
from verl.utils.dataset.multi_domain_benchmark_dataset import MultiDomainBenchmarkDataset
from verl.utils.evaluation.multi_domain_benchmark import (
    LONG_BENCH_METRIC_NAME,
    BenchmarkProfile,
    BenchmarkSpec,
    aggregate_benchmark_details,
    load_benchmark_records,
    parse_benchmark_specs,
)
from verl.utils.reward_score.multi_domain_benchmark import compute_score


class _CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) for character in text]

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(chr(int(token_id)) for token_id in token_ids)

    def apply_chat_template(self, messages, add_generation_prompt, tokenize, **kwargs):
        del kwargs
        rendered = "".join(f"<{item['role']}>{item['content']}" for item in messages)
        if add_generation_prompt:
            rendered += "<assistant>"
        return self.encode(rendered) if tokenize else rendered


def _spec(adapter, *, name=None, path=None, max_samples=-1, prompt=4096, response=512):
    return BenchmarkSpec(
        name=name or adapter.replace("_", "-"),
        adapter=adapter,
        path=path or adapter,
        batch_size=8,
        max_samples=max_samples,
        prompt_max_length=prompt,
        generation_max_tokens=response,
        profiles={"legacy": BenchmarkProfile(n=1), "train_sampling": BenchmarkProfile(n=4)},
    )


def test_parse_typed_benchmark_specs_on_cpu():
    specs = parse_benchmark_specs(
        [
            {
                "name": "gpqa-diamond",
                "adapter": "gpqa_diamond",
                "path": "Idavidrein/gpqa",
                "batch_size": 16,
                "max_samples": -1,
                "prompt_max_length": 4096,
                "generation_max_tokens": 8192,
                "profiles": {"legacy": {"n": 1}, "train_sampling": {"n": 8}},
            }
        ]
    )

    assert specs[0].path == "Idavidrein/gpqa"
    assert specs[0].profiles["train_sampling"].n == 8


def test_ifbench_adapter_preserves_constraints_in_reward_payload_on_cpu(tmp_path):
    benchmark_dir = tmp_path / "allenai" / "IFBench_test" / "data"
    benchmark_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "key": "case-1",
                "prompt": "Include the word lake.",
                "instruction_id_list": "['keywords:existence']",
                "kwargs": "[{'keywords': ['lake']}]",
            }
        ]
    ).to_parquet(benchmark_dir / "train.parquet")
    spec = _spec("ifbench", name="ifbench", path="allenai/IFBench_test")

    dataset = MultiDomainBenchmarkDataset(
        dataset_root=tmp_path,
        spec=spec,
        tokenizer=_CharacterTokenizer(),
    )
    row = dataset[0]
    payload = json.loads(row["reward_model"]["ground_truth"])

    assert row["uid"] == "ifbench:case-1"
    assert row["raw_prompt"] == [{"role": "user", "content": "Include the word lake."}]
    assert payload["instruction_id_list"] == ["keywords:existence"]
    assert payload["kwargs"] == [{"keywords": ["lake"]}]


def test_ifbench_uses_vendored_official_verifier_and_reports_all_four_metrics_on_cpu(monkeypatch):
    monkeypatch.setattr(vendored_ifbench, "require_nltk_resources", lambda: None)
    payload = json.dumps(
        {
            "task_type": "ifbench",
            "prompt": "Include the word lake.",
            "instruction_id_list": ["keywords:existence"],
            "kwargs": [{"keywords": ["lake"]}],
        }
    )

    result = compute_score("The lake is calm.", payload)

    assert result["acc"] is True
    assert result["prompt_loose_acc"] == 1.0
    assert result["prompt_strict_acc"] == 1.0
    assert result["instruction_loose_acc"] == 1.0
    assert result["instruction_strict_acc"] == 1.0


def test_ifbench_restores_arrow_promoted_integer_kwargs_on_cpu(tmp_path, monkeypatch):
    benchmark_dir = tmp_path / "allenai" / "IFBench_test" / "data"
    benchmark_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "key": "repeat-span",
                "prompt": "Copy the requested span.",
                "instruction_id_list": "['repeat:repeat_span']",
                "kwargs": "[{'prompt_to_repeat': 'abcdef', 'n_start': 1.0, 'n_end': 3.0}]",
            }
        ]
    ).to_parquet(benchmark_dir / "train.parquet")
    dataset = MultiDomainBenchmarkDataset(
        dataset_root=tmp_path,
        spec=_spec("ifbench", name="ifbench", path="allenai/IFBench_test"),
        tokenizer=_CharacterTokenizer(),
    )
    payload = json.loads(dataset[0]["reward_model"]["ground_truth"])

    assert payload["kwargs"] == [{"prompt_to_repeat": "abcdef", "n_start": 1, "n_end": 3}]

    monkeypatch.setattr(vendored_ifbench, "require_nltk_resources", lambda: None)
    result = compute_score("bcd", json.dumps(payload))

    assert result["acc"] is True
    assert result["prompt_loose_acc"] == 1.0


def test_ifbench_scorer_defensively_normalizes_integer_valued_float_kwargs_on_cpu(monkeypatch):
    monkeypatch.setattr(vendored_ifbench, "require_nltk_resources", lambda: None)
    payload = json.dumps(
        {
            "task_type": "ifbench",
            "prompt": "Copy the requested span.",
            "instruction_id_list": ["repeat:repeat_span"],
            "kwargs": [{"prompt_to_repeat": "abcdef", "n_start": 1.0, "n_end": 3.0}],
        }
    )

    result = compute_score("bcd", payload)

    assert result["acc"] is True
    assert result["prompt_strict_acc"] == 1.0


def test_mmlu_pro_selects_exact_stratified_thousand_and_uses_five_shots_on_cpu(tmp_path):
    benchmark_dir = tmp_path / "TIGER-Lab" / "MMLU-Pro" / "data"
    benchmark_dir.mkdir(parents=True)
    categories = [f"category-{index}" for index in range(14)]
    validation = []
    test = []
    for category_index, category in enumerate(categories):
        for shot in range(5):
            validation.append(
                {
                    "question": f"Validation {category} {shot}",
                    "options": ["wrong", "right"],
                    "answer": 1,
                    "cot_content": "Reasoning. The correct answer is (B).",
                    "category": category,
                }
            )
        for item in range(86 if category_index < 10 else 85):
            test.append(
                {
                    "question_id": f"{category}-{item}",
                    "question": f"Test {category} {item}",
                    "options": ["wrong", "right"],
                    "answer": 1,
                    "cot_content": "",
                    "category": category,
                }
            )
    pd.DataFrame(validation).to_parquet(benchmark_dir / "validation-00000.parquet")
    pd.DataFrame(test).to_parquet(benchmark_dir / "test-00000.parquet")
    spec = _spec("mmlu_pro", name="mmlu-pro", path="TIGER-Lab/MMLU-Pro", max_samples=1000, prompt=24576)

    records = load_benchmark_records(
        dataset_root=tmp_path,
        spec=spec,
        tokenizer=_CharacterTokenizer(),
        seed=42,
    )

    assert len(records) == 1000
    assert records[0].messages[0]["content"].count("Validation category-") == 5
    assert set(record.payload["category"] for record in records) == set(categories)


def test_gpqa_choice_shuffle_is_stable_and_answer_tracks_correct_text_on_cpu(tmp_path):
    benchmark_dir = tmp_path / "Idavidrein" / "gpqa"
    benchmark_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "Record ID": "record-1",
                "Question": "Which option is correct?",
                "Correct Answer": "correct text",
                "Incorrect Answer 1": "wrong one",
                "Incorrect Answer 2": "wrong two",
                "Incorrect Answer 3": "wrong three",
                "High-level domain": "Physics",
            }
        ]
    ).to_csv(benchmark_dir / "gpqa_diamond.csv", index=False)
    spec = _spec("gpqa_diamond", name="gpqa-diamond", path="Idavidrein/gpqa")

    first = load_benchmark_records(
        dataset_root=tmp_path,
        spec=spec,
        tokenizer=_CharacterTokenizer(),
        seed=42,
    )[0]
    second = load_benchmark_records(
        dataset_root=tmp_path,
        spec=spec,
        tokenizer=_CharacterTokenizer(),
        seed=42,
    )[0]

    assert first.messages == second.messages
    expected = first.payload["expected_answer"]
    expected_line = next(line for line in first.messages[0]["content"].splitlines() if line.startswith(f"{expected}."))
    assert expected_line.endswith("correct text")


def test_longbench_middle_truncates_only_context_and_labels_metric_on_cpu(tmp_path):
    benchmark_dir = tmp_path / "THUDM" / "LongBench-v2"
    benchmark_dir.mkdir(parents=True)
    context = "BEGIN-" + "m" * 4000 + "-END"
    (benchmark_dir / "data.json").write_text(
        json.dumps(
            [
                {
                    "_id": "long-1",
                    "domain": "Long document QA",
                    "sub_domain": "synthetic",
                    "difficulty": "hard",
                    "length": "long",
                    "question": "Question remains intact?",
                    "choice_A": "alpha",
                    "choice_B": "beta",
                    "choice_C": "gamma",
                    "choice_D": "delta",
                    "answer": "D",
                    "context": context,
                }
            ]
        ),
        encoding="utf-8",
    )
    spec = _spec(
        "longbench_v2",
        name="longbench-v2",
        path="THUDM/LongBench-v2",
        prompt=800,
        response=128,
    )

    dataset = MultiDomainBenchmarkDataset(
        dataset_root=tmp_path,
        spec=spec,
        tokenizer=_CharacterTokenizer(),
    )
    row = dataset[0]
    payload = json.loads(row["reward_model"]["ground_truth"])
    prompt = row["raw_prompt"][0]["content"]

    assert row["data_source"] == LONG_BENCH_METRIC_NAME
    assert payload["truncated"] is True
    assert payload["retained_ratio"] < 1.0
    assert "BEGIN-" in prompt and "-END" in prompt
    assert "Question remains intact?" in prompt and "(D) delta" in prompt


def test_multiple_choice_score_strips_qwen_thinking_and_custom_aggregates_on_cpu():
    payload = json.dumps(
        {
            "task_type": "multiple_choice",
            "benchmark": "mmlu_pro",
            "expected_answer": "B",
            "valid_labels": ["A", "B", "C"],
            "category": "physics",
        }
    )

    result = compute_score("<think>A long thought.</think> The correct answer is (B).", payload)
    metrics = aggregate_benchmark_details(
        adapter="mmlu_pro",
        metric_name="mmlu-pro",
        metric_prefix="test-legacy",
        sample_uids=["one", "two"],
        reward_extra_infos={"acc": [1.0, 0.0], "category": ["physics", "chemistry"]},
    )

    assert result["acc"] is True
    assert result["pred"] == "B"
    assert metrics["test-legacy-aux/mmlu-pro/category/physics/acc"] == 1.0
    assert metrics["test-legacy-aux/mmlu-pro/category-macro/acc"] == 0.5
    assert metrics["test-legacy-core/mmlu-pro/overall-micro/acc"] == 0.5


def test_unclosed_qwen_thinking_cannot_leak_a_hidden_benchmark_answer_on_cpu():
    payload = json.dumps(
        {
            "task_type": "multiple_choice",
            "benchmark": "mmlu_pro",
            "expected_answer": "B",
            "valid_labels": ["A", "B", "C"],
            "category": "physics",
        }
    )

    result = compute_score("<think>The correct answer is (B).", payload)

    assert result["score"] == 0.0
    assert result["parse_failure"] is True
    assert result["benchmark"] == "mmlu_pro"
    assert result["category"] == "physics"


def test_unclosed_qwen_thinking_returns_zero_aime_diagnostics_on_cpu():
    payload = json.dumps(
        {
            "task_type": "aime",
            "expected_answer": "42",
        }
    )

    result = compute_score("<think>The answer might be 42.", payload)

    assert result["score"] == 0.0
    assert result["acc"] is False
    assert result["parse_failure"] is True
    assert result["math_verify_score"] == 0.0


def test_aime_valid_and_unclosed_outputs_aggregate_without_missing_values_on_cpu(monkeypatch):
    from verl.utils.reward_score import math_verify

    monkeypatch.setattr(math_verify, "compute_score", lambda *args, **kwargs: 1.0)
    payload = json.dumps(
        {
            "task_type": "aime",
            "expected_answer": "42",
        }
    )
    results = [
        compute_score("The answer is 42.", payload),
        compute_score("<think>The answer might be 42.", payload),
    ]
    info_keys = set().union(*(result.keys() for result in results))
    infos_dict = {key: [result.get(key) for result in results] for key in info_keys}

    metrics = process_validation_metrics(
        ["aime25", "aime25"],
        ["aime25:1", "aime25:1"],
        infos_dict,
        seed=42,
    )

    assert metrics["aime25"]["score"]["mean@2"] == 0.5
    assert metrics["aime25"]["math_verify_score"]["mean@2"] == 0.5
