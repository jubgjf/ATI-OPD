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
from types import SimpleNamespace

import pandas as pd
import pytest

from verl.trainer import main_math_benchmark_eval as eval_cli
from verl.trainer.main_math_benchmark_eval import extract_choice_generation
from verl.utils.evaluation import math_benchmark
from verl.utils.evaluation.math_benchmark import (
    CheckpointInfo,
    append_jsonl_record,
    apply_path_remaps,
    completed_sample_ids,
    discover_checkpoints,
    find_baseline_model_path_in_wandb_config,
    find_default_local_dir_in_wandb_config,
    has_length_finish_reason,
    load_benchmark_samples,
    load_jsonl_records,
    normalize_ground_truth,
    parse_path_remaps,
    score_completion,
    split_final_content,
    summarize_records,
)


def test_load_benchmark_samples_normalizes_local_schemas_on_cpu(tmp_path):
    dataset_root = tmp_path / "dataset"
    (dataset_root / "aime24").mkdir(parents=True)
    (dataset_root / "aime25").mkdir()
    (dataset_root / "amc23").mkdir()

    pd.DataFrame(
        [{"id": 60, "problem": "AIME problem", "solution": r"\boxed{204}", "url": "https://example.test"}]
    ).to_parquet(dataset_root / "aime24" / "test-00000-of-00001.parquet")
    (dataset_root / "aime25" / "test.jsonl").write_text(
        json.dumps({"id": "0", "problem": "AIME 2025 problem", "answer": 70}) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame([{"id": 0, "question": "AMC problem", "answer": 27.0}]).to_parquet(
        dataset_root / "amc23" / "test-00000-of-00001.parquet"
    )

    assert load_benchmark_samples(dataset_root, "aime24")[0].ground_truth == "204"
    assert load_benchmark_samples(dataset_root, "aime25")[0].ground_truth == "70"
    amc_sample = load_benchmark_samples(dataset_root, "amc23")[0]
    assert amc_sample.problem == "AMC problem"
    assert amc_sample.ground_truth == "27"


def test_final_content_split_ignores_thinking_boxed_answer_on_cpu():
    text = r"<think>bad early answer \boxed{0}</think> final answer is \boxed{204}"

    assert split_final_content(text) == r"final answer is \boxed{204}"
    assert (
        split_final_content(None, content=r"<think>bad early answer \boxed{0}</think> final answer is \boxed{204}")
        == r"final answer is \boxed{204}"
    )
    assert split_final_content(None, reasoning_content="bad early answer", content=r"final answer is \boxed{204}") == (
        r"final answer is \boxed{204}"
    )
    scored = score_completion(text, "204", verifier="math_dapo")

    assert scored.acc is True
    assert scored.pred == "204"
    assert scored.verified_by == "math_dapo"


def test_hybrid_verifier_falls_back_to_math_verify_on_cpu(monkeypatch):
    monkeypatch.setattr(math_benchmark.math_verify, "compute_score", lambda *args, **kwargs: 1.0)

    scored = score_completion("Final Answer: not the same string", "42", verifier="hybrid")

    assert scored.acc is True
    assert scored.verified_by == "math_verify"
    assert scored.math_verify_score == 1.0
    assert scored.invalid_pred is False


def test_normalize_ground_truth_handles_boxed_and_integer_float_on_cpu():
    assert normalize_ground_truth(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert normalize_ground_truth(27.0) == "27"
    assert normalize_ground_truth("70.0") == "70"


def test_jsonl_resume_helpers_and_summary_on_cpu(tmp_path):
    records_path = tmp_path / "records.jsonl"
    skipped_path = tmp_path / "skipped.jsonl"
    append_jsonl_record(
        records_path,
        {
            "sample_id": "a",
            "correct_count": 1,
            "avg@2": 0.5,
            "pass@2": 1.0,
            "completions": [
                {
                    "acc": True,
                    "invalid_pred": False,
                    "has_answer_pattern": False,
                    "has_boxed_pattern": True,
                    "has_accepted_answer_format": True,
                },
                {
                    "acc": False,
                    "invalid_pred": True,
                    "has_answer_pattern": False,
                    "has_boxed_pattern": False,
                    "has_accepted_answer_format": False,
                },
            ],
        },
        fsync=False,
    )
    append_jsonl_record(skipped_path, {"sample_id": "b", "reason": "length"}, fsync=False)

    records = load_jsonl_records(records_path)
    skipped = load_jsonl_records(skipped_path)
    summary = summarize_records(records, skipped, benchmark="aime24", checkpoint_step=10, rollout_n=2)

    assert completed_sample_ids(records) == {"a"}
    assert summary.valid_samples == 1
    assert summary.skipped_samples == 1
    assert summary.avg_at_n == pytest.approx(0.5)
    assert summary.pass_at_n == pytest.approx(1.0)
    assert summary.length_skip_count == 1


def test_discover_checkpoints_requires_complete_fsdp_actor_on_cpu(tmp_path):
    root = tmp_path / "outputs"
    complete_actor = root / "global_step_10" / "actor"
    incomplete_actor = root / "global_step_20" / "actor"
    complete_actor.mkdir(parents=True)
    incomplete_actor.mkdir(parents=True)
    (complete_actor / "fsdp_config.json").write_text('{"world_size": 2}', encoding="utf-8")
    (complete_actor / "model_world_size_2_rank_0.pt").write_bytes(b"")
    (complete_actor / "model_world_size_2_rank_1.pt").write_bytes(b"")
    (incomplete_actor / "fsdp_config.json").write_text('{"world_size": 2}', encoding="utf-8")
    (incomplete_actor / "model_world_size_2_rank_0.pt").write_bytes(b"")

    checkpoints = discover_checkpoints(root)

    assert [checkpoint.step for checkpoint in checkpoints] == [10]
    assert checkpoints[0].hf_dir == complete_actor / "huggingface"


def test_wandb_config_lookup_and_path_remap_on_cpu():
    assert (
        find_default_local_dir_in_wandb_config({"trainer": {"default_local_dir": "/npu/outputs/run"}})
        == "/npu/outputs/run"
    )
    assert find_default_local_dir_in_wandb_config({"trainer.default_local_dir": "/flat/run"}) == "/flat/run"
    assert (
        find_baseline_model_path_in_wandb_config({"actor_rollout_ref": {"model": {"path": "/models/Qwen3-1.7B"}}})
        == "/models/Qwen3-1.7B"
    )

    remaps = parse_path_remaps(["/npu/outputs=/gpu/outputs"])
    assert apply_path_remaps("/npu/outputs/run/global_step_1", remaps) == "/gpu/outputs/run/global_step_1"


def test_discover_evaluation_checkpoints_includes_baseline_step_zero_on_cpu(tmp_path):
    baseline = tmp_path / "Qwen3-1.7B"
    baseline.mkdir()
    (baseline / "config.json").write_text("{}", encoding="utf-8")
    (baseline / "model.safetensors").write_bytes(b"")

    root = tmp_path / "outputs"
    actor = root / "global_step_20" / "actor"
    actor.mkdir(parents=True)
    (actor / "fsdp_config.json").write_text('{"world_size": 1}', encoding="utf-8")
    (actor / "model_world_size_1_rank_0.pt").write_bytes(b"")

    args = SimpleNamespace(model_path=None, model_step=0)

    checkpoints = eval_cli.discover_evaluation_checkpoints(
        default_local_dir=str(root),
        checkpoint_steps=None,
        baseline_model_path=str(baseline),
        args=args,
    )

    assert [checkpoint.step for checkpoint in checkpoints] == [0, 20]
    assert checkpoints[0].hf_dir == baseline


def test_wandb_eval_logging_uses_benchmark_series_without_all_or_step_summary_on_cpu(tmp_path):
    class DummyRun:
        def __init__(self):
            self.payloads = []
            self.summary = {}

        def log(self, payload):
            self.payloads.append(payload)

    run = DummyRun()
    summary = {
        "benchmark": "aime24",
        "checkpoint_step": 40,
        "rollout_n": 16,
        "avg_at_n": 0.25,
        "pass_at_n": 0.5,
        "valid_samples": 30,
        "skipped_samples": 0,
        "correct_completions": 120,
        "total_completions": 480,
        "invalid_pred_rate": 0.0,
        "answer_pattern_rate": 1.0,
        "boxed_pattern_rate": 1.0,
        "accepted_answer_format_rate": 1.0,
        "length_skip_count": 0,
    }

    eval_cli.log_summary_to_wandb_once(
        wandb_run=run,
        summary=summary,
        logged_path=tmp_path / "global_step_40" / "aime24" / "wandb_logged.json",
    )

    assert run.payloads == [
        {
            "eval/checkpoint_step": 40,
            "eval/aime24/avg@16": 0.25,
            "eval/aime24/pass@16": 0.5,
            "eval/aime24/valid_samples": 30,
            "eval/aime24/skipped_samples": 0,
            "eval/aime24/correct_completions": 120,
            "eval/aime24/total_completions": 480,
            "eval/aime24/invalid_pred_rate": 0.0,
            "eval/aime24/answer_pattern_rate": 1.0,
            "eval/aime24/boxed_pattern_rate": 1.0,
            "eval/aime24/accepted_answer_format_rate": 1.0,
            "eval/aime24/length_skip_count": 0,
        }
    ]
    assert run.summary == {}
    assert all(not key.startswith("eval/all/") for key in run.payloads[0])
    assert all("global_step_40" not in key for key in run.payloads[0])


def test_has_length_finish_reason_on_cpu():
    assert has_length_finish_reason(["stop", "length"]) is True
    assert has_length_finish_reason(["stop", None]) is False


def test_extract_chat_choice_generation_on_cpu():
    choice = {
        "message": {
            "role": "assistant",
            "reasoning_content": "work",
            "content": r"final answer is \boxed{204}",
        },
        "finish_reason": "stop",
    }

    generation = extract_choice_generation(choice)

    assert generation["text"] == r"work\nfinal answer is \boxed{204}".replace(r"\n", "\n")
    assert generation["reasoning_content"] == "work"
    assert generation["content"] == r"final answer is \boxed{204}"


def test_evaluate_benchmark_expands_rollouts_into_individual_requests_on_cpu(tmp_path, monkeypatch):
    dataset_root = tmp_path / "dataset"
    (dataset_root / "amc23").mkdir(parents=True)
    (dataset_root / "amc23" / "test.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"id": "a", "problem": "one plus zero", "answer": "1"}),
                json.dumps({"id": "b", "problem": "two plus zero", "answer": "2"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    answer_by_problem = {"one plus zero": "1", "two plus zero": "2"}
    calls = []

    def fake_request_generations(*, messages, n, **kwargs):
        del kwargs
        calls.append((messages[0]["content"], n))
        problem = messages[0]["content"].split("\n\n", 1)[0]
        return [
            {
                "message": {"content": rf"final answer is \boxed{{{answer_by_problem[problem]}}}"},
                "finish_reason": "stop",
            }
        ]

    monkeypatch.setattr(eval_cli, "request_generations", fake_request_generations)

    class DummyTokenizer:
        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return text.split()

    class DummyServer:
        base_url = "http://127.0.0.1:1"

    args = SimpleNamespace(
        num_samples=3,
        max_samples=-1,
        retry_skipped=False,
        stop_after_samples=None,
        max_concurrency=2,
        max_concurrency_per_dp=128,
        data_parallel_size=1,
        max_num_seqs=128,
        length_retries=2,
        served_model_name="dummy",
        verifier="math_dapo",
        math_verify_timeout=1.0,
        store_outputs=True,
    )
    checkpoint = CheckpointInfo(step=0, path=tmp_path, actor_dir=tmp_path, hf_dir=tmp_path)

    summary, stopped = eval_cli.evaluate_benchmark(
        checkpoint=checkpoint,
        benchmark="amc23",
        dataset_root=dataset_root,
        results_dir=tmp_path / "results",
        tokenizer=DummyTokenizer(),
        server=DummyServer(),
        args=args,
        processed_counter={"count": 0},
    )

    assert stopped is False
    assert len(calls) == 6
    assert {n for _, n in calls} == {1}
    assert summary["valid_samples"] == 2
    assert summary["total_completions"] == 6
    assert summary["avg_at_n"] == pytest.approx(1.0)


def test_effective_max_concurrency_defaults_to_dp_times_per_dp_on_cpu():
    args = SimpleNamespace(max_concurrency=None, data_parallel_size=4, max_concurrency_per_dp=128, max_num_seqs=128)

    assert eval_cli.effective_max_concurrency(args) == 512


def test_evaluate_benchmark_retries_length_finish_reason_on_cpu(tmp_path, monkeypatch):
    dataset_root = tmp_path / "dataset"
    (dataset_root / "amc23").mkdir(parents=True)
    (dataset_root / "amc23" / "test.jsonl").write_text(
        json.dumps({"id": "a", "problem": "one plus zero", "answer": "1"}) + "\n",
        encoding="utf-8",
    )
    calls = {"count": 0}

    def fake_request_generations(*, n, **kwargs):
        del kwargs
        calls["count"] += 1
        finish_reason = "length" if calls["count"] <= 4 else "stop"
        return [{"message": {"content": r"final answer is \boxed{1}"}, "finish_reason": finish_reason}]

    monkeypatch.setattr(eval_cli, "request_generations", fake_request_generations)

    class DummyTokenizer:
        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return text.split()

    class DummyServer:
        base_url = "http://127.0.0.1:1"

    args = SimpleNamespace(
        num_samples=2,
        max_samples=-1,
        retry_skipped=False,
        stop_after_samples=None,
        max_concurrency=1,
        max_concurrency_per_dp=128,
        data_parallel_size=1,
        max_num_seqs=128,
        length_retries=2,
        served_model_name="dummy",
        verifier="math_dapo",
        math_verify_timeout=1.0,
        store_outputs=True,
    )
    checkpoint = CheckpointInfo(step=0, path=tmp_path, actor_dir=tmp_path, hf_dir=tmp_path)
    results_dir = tmp_path / "results"

    summary, stopped = eval_cli.evaluate_benchmark(
        checkpoint=checkpoint,
        benchmark="amc23",
        dataset_root=dataset_root,
        results_dir=results_dir,
        tokenizer=DummyTokenizer(),
        server=DummyServer(),
        args=args,
        processed_counter={"count": 0},
    )

    skipped = load_jsonl_records(results_dir / "global_step_0" / "amc23" / "skipped.jsonl")

    assert stopped is False
    assert calls["count"] == 6
    assert [record["status"] for record in skipped] == ["retry", "retry"]
    assert summary["valid_samples"] == 1
    assert summary["skipped_samples"] == 0
    assert summary["avg_at_n"] == pytest.approx(1.0)


def test_evaluate_benchmark_resumes_length_retry_attempts_on_cpu(tmp_path, monkeypatch):
    dataset_root = tmp_path / "dataset"
    (dataset_root / "amc23").mkdir(parents=True)
    (dataset_root / "amc23" / "test.jsonl").write_text(
        json.dumps({"id": "a", "problem": "one plus zero", "answer": "1"}) + "\n",
        encoding="utf-8",
    )
    results_dir = tmp_path / "results"
    skipped_path = results_dir / "global_step_0" / "amc23" / "skipped.jsonl"
    append_jsonl_record(
        skipped_path,
        {
            "status": "retry",
            "reason": "length",
            "sample_id": "a",
            "attempt_idx": 0,
            "attempt_number": 1,
            "max_attempts": 3,
        },
        fsync=False,
    )
    calls = {"count": 0}

    def fake_request_generations(*, n, **kwargs):
        del kwargs
        calls["count"] += 1
        return [{"message": {"content": r"final answer is \boxed{1}"}, "finish_reason": "stop"}]

    monkeypatch.setattr(eval_cli, "request_generations", fake_request_generations)

    class DummyTokenizer:
        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return text.split()

    class DummyServer:
        base_url = "http://127.0.0.1:1"

    args = SimpleNamespace(
        num_samples=2,
        max_samples=-1,
        retry_skipped=False,
        stop_after_samples=None,
        max_concurrency=1,
        max_concurrency_per_dp=128,
        data_parallel_size=1,
        max_num_seqs=128,
        length_retries=2,
        served_model_name="dummy",
        verifier="math_dapo",
        math_verify_timeout=1.0,
        store_outputs=True,
    )
    checkpoint = CheckpointInfo(step=0, path=tmp_path, actor_dir=tmp_path, hf_dir=tmp_path)

    summary, stopped = eval_cli.evaluate_benchmark(
        checkpoint=checkpoint,
        benchmark="amc23",
        dataset_root=dataset_root,
        results_dir=results_dir,
        tokenizer=DummyTokenizer(),
        server=DummyServer(),
        args=args,
        processed_counter={"count": 0},
    )

    assert stopped is False
    assert calls["count"] == 2
    assert summary["valid_samples"] == 1
    assert summary["skipped_samples"] == 0
