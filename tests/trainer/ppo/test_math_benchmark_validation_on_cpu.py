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

import numpy as np
import torch
from omegaconf import OmegaConf

from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    _as_sample_values,
    _extend_validation_tool_metric_values,
    _summarize_validation_tool_metrics,
    _with_reward_data_source,
)


def test_trainer_defaults_disable_validation_and_pretrain_benchmarks_on_cpu():
    config = OmegaConf.load("verl/trainer/config/ppo_trainer.yaml")

    assert config.trainer.val_before_train is False
    assert config.trainer.teacher_val_before_train is False
    assert config.trainer.test_before_train is False
    assert config.trainer.test_freq == -1


def test_validation_reward_override_copies_metadata_and_uses_math_verify_on_cpu():
    original = np.array([{"index": 1}, None], dtype=object)

    overridden = _with_reward_data_source(
        original,
        count=2,
        reward_data_source="math_verify",
    )

    assert overridden.tolist() == [
        {"index": 1, "reward_data_source": "math_verify"},
        {"reward_data_source": "math_verify"},
    ]
    assert original[0] == {"index": 1}


def test_evaluation_reward_field_normalization_accepts_scalar_singleton_on_cpu():
    assert _as_sample_values(np.asarray(1.0), sample_count=1, field_name="acc") == [1.0]
    assert _as_sample_values(torch.tensor(0.5), sample_count=1, field_name="acc") == [0.5]


def test_math_benchmark_profiles_have_independent_mean_at_16_namespaces_on_cpu():
    trainer = object.__new__(RayPPOTrainer)
    data_sources = np.array(["aime24"] * 16, dtype=object)
    sample_uids = ["aime24:0"] * 16
    acc = [1.0] * 4 + [0.0] * 12

    legacy_metrics = trainer._val_metrics_update(
        data_sources=data_sources,
        sample_uids=sample_uids,
        reward_extra_infos_dict={"acc": acc, "reward": acc},
        sample_turns=[],
        metric_prefix="test-legacy",
    )
    train_sampling_metrics = trainer._val_metrics_update(
        data_sources=data_sources,
        sample_uids=sample_uids,
        reward_extra_infos_dict={"acc": acc, "reward": acc},
        sample_turns=[],
        metric_prefix="test-train-sampling",
    )

    assert legacy_metrics["test-legacy-core/aime24/acc/mean@16"] == 0.25
    assert train_sampling_metrics["test-train-sampling-core/aime24/acc/mean@16"] == 0.25
    assert legacy_metrics.keys().isdisjoint(train_sampling_metrics)


def test_math_benchmark_tool_metrics_are_scoped_to_sampling_profile_on_cpu():
    collected = {}
    _extend_validation_tool_metric_values(
        collected,
        {
            "tool_call_counts": np.asarray([0, 2], dtype=object),
            "tool_call/attempt_rate": np.asarray([0.0, 1.0], dtype=object),
            "tool_call/execution_success_rate": np.asarray([0.0, 0.5], dtype=object),
        },
    )

    metrics = _summarize_validation_tool_metrics(collected, metric_prefix="test-train-sampling")

    assert metrics["test-train-sampling-tool/tool_call_counts/min"] == 0.0
    assert metrics["test-train-sampling-tool/tool_call_counts/max"] == 2.0
    assert metrics["test-train-sampling-tool/tool_call_counts/mean"] == 1.0
    assert metrics["test-train-sampling-tool/tool_call/attempt_rate"] == 0.5
    assert metrics["test-train-sampling-tool/tool_call/execution_success_rate"] == 0.25


def test_validate_test_benchmarks_runs_both_raw_sampling_profiles_on_cpu():
    trainer = object.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "test_benchmarks_max_response_length": 128,
                "log_test_generations": 3,
                "test_data_dir": None,
            },
            "actor_rollout_ref": {
                "rollout": {
                    "response_length": 256,
                    "benchmark_sampling_profiles": {
                        "legacy": {
                            "temperature": 0.7,
                            "top_p": 0.95,
                            "top_k": -1,
                            "do_sample": True,
                            "n": 16,
                        },
                        "train_sampling": {
                            "temperature": 1.0,
                            "top_p": 1.0,
                            "top_k": -1,
                            "do_sample": True,
                            "n": 16,
                        },
                    },
                }
            },
        }
    )
    trainer.test_benchmark_dataloader = object()
    trainer.test_benchmark_generations_loggers = {"legacy": object(), "train_sampling": object()}
    captured = []

    def _validate(**kwargs):
        captured.append(kwargs)
        return {f"{kwargs['metric_prefix']}-core/aime24/acc/mean@16": 0.5}

    trainer._validate = _validate

    metrics = trainer._validate_test_benchmarks()

    assert set(metrics) == {
        "test-legacy-core/aime24/acc/mean@16",
        "test-train-sampling-core/aime24/acc/mean@16",
    }
    assert [call["sampling_profile_name"] for call in captured] == ["legacy", "train_sampling"]
    assert [call["metric_prefix"] for call in captured] == ["test-legacy", "test-train-sampling"]
    assert all(call["dataloader"] is trainer.test_benchmark_dataloader for call in captured)
    assert all(call["evaluation_mode"] == "test" for call in captured)
    assert all(int(call["sampling_config"].n) == 16 for call in captured)
    assert all(call["generation_max_tokens"] == 128 for call in captured)
    assert all(call["log_generations"] == 3 for call in captured)
    assert all(call["validation_data_dir"] == "" for call in captured)
    assert [call["dump_filename_prefix"] for call in captured] == [
        "test_benchmarks_legacy",
        "test_benchmarks_train_sampling",
    ]
