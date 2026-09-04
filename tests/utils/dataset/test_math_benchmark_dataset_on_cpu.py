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
from omegaconf import OmegaConf

from verl.trainer.main_ppo import create_math_benchmark_dataset
from verl.utils.dataset.math_benchmark_dataset import MathBenchmarkDataset
from verl.utils.reward_score import default_compute_score


def _write_benchmarks(dataset_root):
    (dataset_root / "aime24").mkdir(parents=True)
    (dataset_root / "amc23").mkdir()
    (dataset_root / "aime24" / "test.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"id": "a", "problem": "AIME one", "answer": 1}),
                json.dumps({"id": "b", "problem": "AIME two", "answer": 2}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    pd.DataFrame([{"id": 7, "question": "AMC question", "solution": r"\boxed{27}"}]).to_parquet(
        dataset_root / "amc23" / "test-00000-of-00001.parquet"
    )


def _write_amo_bench(dataset_root):
    description_ids = {2, 24, 25, 26, 27, 38, 39, 40, 41, 42, 46}
    benchmark_dir = dataset_root / "meituan-longcat" / "AMO-Bench" / "data"
    benchmark_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "question_id": question_id,
                "prompt": f"AMO prompt {question_id} with its own final-answer instruction.",
                "answer": rf"\boxed{{{question_id}}}",
                "answer_type": "description" if question_id in description_ids else "number",
            }
            for question_id in range(1, 51)
        ]
    ).to_parquet(benchmark_dir / "test-00000-of-00001.parquet")


def test_math_benchmark_dataset_builds_stable_validation_rows_on_cpu(tmp_path):
    dataset_root = tmp_path / "datasets"
    _write_benchmarks(dataset_root)

    dataset = MathBenchmarkDataset(
        dataset_root=dataset_root,
        benchmarks=["aime24", "amc23"],
        max_samples_per_benchmark=1,
    )

    assert len(dataset) == 2
    aime_row = dataset[0]
    amc_row = dataset[1]
    assert aime_row["uid"] == "aime24:a"
    assert amc_row["uid"] == "amc23:7"
    assert aime_row["data_source"] == "aime24"
    assert amc_row["data_source"] == "amc23"
    assert amc_row["reward_model"]["ground_truth"] == "27"
    assert amc_row["extra_info"]["reward_data_source"] == "math_verify"
    assert "put your final answer within \\boxed{}" in aime_row["raw_prompt"][0]["content"]


def test_math_benchmark_dataset_preserves_tool_agent_and_system_prompt_on_cpu(tmp_path):
    dataset_root = tmp_path / "datasets"
    _write_benchmarks(dataset_root)
    system_prompt_path = tmp_path / "tool-system.txt"
    system_prompt_path.write_text("You may call code_interpreter.", encoding="utf-8")

    dataset = MathBenchmarkDataset(
        dataset_root=dataset_root,
        benchmarks=["aime24"],
        max_samples_per_benchmark=1,
        system_prompt_path=system_prompt_path,
        agent_name="tool_agent",
    )

    row = dataset[0]
    assert row["agent_name"] == "tool_agent"
    assert row["raw_prompt"][0] == {"role": "system", "content": "You may call code_interpreter."}
    assert row["raw_prompt"][1]["role"] == "user"


def test_math_benchmark_dataset_appends_user_suffix_and_preserves_system_prompt_on_cpu(tmp_path):
    dataset_root = tmp_path / "datasets"
    _write_benchmarks(dataset_root)
    system_prompt_path = tmp_path / "tool-system.txt"
    suffix_path = tmp_path / "tool-user-suffix.txt"
    system_prompt_path.write_text("System policy", encoding="utf-8")
    suffix_path.write_text("Plan first, then use code_interpreter.", encoding="utf-8")

    dataset = MathBenchmarkDataset(
        dataset_root=dataset_root,
        benchmarks=["aime24"],
        max_samples_per_benchmark=1,
        system_prompt_path=system_prompt_path,
        user_prompt_suffix_path=suffix_path,
        agent_name="tool_agent",
    )

    row = dataset[0]
    assert row["raw_prompt"][0] == {"role": "system", "content": "System policy"}
    assert row["raw_prompt"][1]["content"].endswith("\n\nPlan first, then use code_interpreter.")


def test_create_math_benchmark_dataset_uses_training_tool_environment_on_cpu(tmp_path):
    dataset_root = tmp_path / "datasets"
    _write_benchmarks(dataset_root)
    system_prompt_path = tmp_path / "tool-system.txt"
    system_prompt_path.write_text("Use the configured code tool when useful.", encoding="utf-8")
    config = OmegaConf.create(
        {
            "trainer": {"test_before_train": False, "test_benchmarks_freq": 20},
            "data": {
                "system_prompt_path": str(system_prompt_path),
                "test_benchmarks": {
                    "root": str(dataset_root),
                    "names": ["aime24"],
                    "reward_data_source": "math_verify",
                    "max_samples_per_benchmark": 1,
                },
            },
            "actor_rollout_ref": {
                "rollout": {
                    "agent": {"default_agent_loop": "tool_agent"},
                    "multi_turn": {
                        "enable": True,
                        "tool_config_path": None,
                        "function_tool_path": "verl/tools/code_interpreter_tool.py",
                    },
                }
            },
        }
    )

    dataset = create_math_benchmark_dataset(config)
    row = dataset[0]

    assert row["agent_name"] == "tool_agent"
    assert row["raw_prompt"][0]["content"] == "Use the configured code tool when useful."


def test_create_math_benchmark_dataset_passes_user_suffix_on_cpu(tmp_path):
    dataset_root = tmp_path / "datasets"
    _write_benchmarks(dataset_root)
    suffix_path = tmp_path / "tool-user-suffix.txt"
    suffix_path.write_text("Plan first, then use code_interpreter.", encoding="utf-8")
    config = OmegaConf.create(
        {
            "trainer": {"test_before_train": False, "test_benchmarks_freq": 20},
            "data": {
                "system_prompt_path": None,
                "user_prompt_suffix_path": str(suffix_path),
                "test_benchmarks": {
                    "root": str(dataset_root),
                    "names": ["aime24"],
                    "reward_data_source": "math_verify",
                    "max_samples_per_benchmark": 1,
                },
            },
            "actor_rollout_ref": {
                "rollout": {
                    "agent": {"default_agent_loop": "tool_agent"},
                    "multi_turn": {
                        "enable": True,
                        "tool_config_path": None,
                        "function_tool_path": "verl/tools/sod_code_interpreter_tool.py",
                    },
                }
            },
        }
    )

    row = create_math_benchmark_dataset(config)[0]

    assert row["raw_prompt"][0]["role"] == "user"
    assert row["raw_prompt"][0]["content"].endswith("\n\nPlan first, then use code_interpreter.")


def test_create_math_benchmark_dataset_rejects_tool_agent_without_tools_on_cpu(tmp_path):
    dataset_root = tmp_path / "datasets"
    _write_benchmarks(dataset_root)
    config = OmegaConf.create(
        {
            "trainer": {"test_before_train": False, "test_benchmarks_freq": 20},
            "data": {"test_benchmarks": {"root": str(dataset_root), "names": ["aime24"]}},
            "actor_rollout_ref": {
                "rollout": {
                    "agent": {"default_agent_loop": "tool_agent"},
                    "multi_turn": {"enable": True, "tool_config_path": None, "function_tool_path": None},
                }
            },
        }
    )

    with pytest.raises(ValueError, match="requires multi_turn.tool_config_path or multi_turn.function_tool_path"):
        create_math_benchmark_dataset(config)


def test_math_benchmark_dataset_preserves_amo_metadata_and_prompt_on_cpu(tmp_path):
    dataset_root = tmp_path / "datasets"
    _write_amo_bench(dataset_root)

    dataset = MathBenchmarkDataset(
        dataset_root=dataset_root,
        benchmarks=["meituan-longcat/AMO-Bench"],
    )

    assert len(dataset) == 39
    row = dataset[0]
    assert row["uid"] == "AMO-Bench:1"
    assert row["raw_prompt"] == [{"role": "user", "content": "AMO prompt 1 with its own final-answer instruction."}]
    assert row["extra_info"]["question_id"] == 1
    assert row["extra_info"]["answer_type"] == "number"
    assert row["extra_info"]["grader"] == "amo_bench"
    assert row["extra_info"]["reward_data_source"] == "math_verify"


def test_math_benchmark_dataset_uses_nested_directory_basename_for_metrics_on_cpu(tmp_path):
    dataset_root = tmp_path / "datasets"
    benchmark_dir = dataset_root / "math-ai" / "aime24"
    benchmark_dir.mkdir(parents=True)
    (benchmark_dir / "test.jsonl").write_text(
        json.dumps({"id": "nested", "problem": "Nested AIME problem", "answer": 42}) + "\n",
        encoding="utf-8",
    )

    dataset = MathBenchmarkDataset(dataset_root=dataset_root, benchmarks=["math-ai/aime24/"])

    row = dataset[0]
    assert dataset.benchmarks == ("math-ai/aime24",)
    assert dataset.benchmark_names == ("aime24",)
    assert row["data_source"] == "aime24"
    assert row["uid"] == "aime24:nested"
    assert row["extra_info"]["benchmark"] == "aime24"
    assert row["extra_info"]["benchmark_path"] == "math-ai/aime24"


def test_math_benchmark_dataset_rejects_duplicate_metric_names_on_cpu(tmp_path):
    with pytest.raises(ValueError, match="unique final directory names"):
        MathBenchmarkDataset(
            dataset_root=tmp_path,
            benchmarks=["math-ai/aime24", "another-source/aime24"],
        )


@pytest.mark.parametrize("benchmark_path", [".", "../aime24", "/datasets/aime24"])
def test_math_benchmark_dataset_rejects_paths_outside_root_on_cpu(tmp_path, benchmark_path):
    with pytest.raises(ValueError, match="under dataset_root|relative to dataset_root"):
        MathBenchmarkDataset(dataset_root=tmp_path, benchmarks=[benchmark_path])


def test_math_benchmark_reward_route_keeps_metric_data_source_on_cpu():
    result = default_compute_score(
        data_source="amc23",
        solution_str=r"The final answer is \boxed{27}.",
        ground_truth="27",
        extra_info={"reward_data_source": "math_dapo"},
    )

    assert result["acc"] is True
    assert result["pred"] == "27"


@pytest.mark.parametrize("max_samples", [0, -2])
def test_math_benchmark_dataset_rejects_invalid_sample_limit_on_cpu(tmp_path, max_samples):
    with pytest.raises(ValueError, match="max_samples_per_benchmark"):
        MathBenchmarkDataset(
            dataset_root=tmp_path,
            benchmarks=["aime24"],
            max_samples_per_benchmark=max_samples,
        )
