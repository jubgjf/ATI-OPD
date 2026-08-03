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

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    _as_sample_values,
    _cap_ranked_outcome_cases,
    _with_reward_data_source,
)


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


def test_evaluation_case_cap_is_global_across_accumulated_batches_on_cpu():
    cases = [
        {"transition": "pre_wrong_post_correct", "case_rank": 0.8},
        {"transition": "pre_wrong_post_correct", "case_rank": 0.2},
        {"transition": "pre_correct_post_wrong", "case_rank": 0.6},
        {"transition": "pre_correct_post_wrong", "case_rank": 0.1},
    ]

    capped = _cap_ranked_outcome_cases(cases, case_limit=1)

    assert [index for index, case in enumerate(capped) if case is not None] == [1, 3]


def test_math_benchmark_metrics_use_test_core_and_mean_at_16_on_cpu():
    trainer = object.__new__(RayPPOTrainer)
    data_sources = np.array(["aime24"] * 16 + ["amc23"] * 16, dtype=object)
    sample_uids = ["aime24:0"] * 16 + ["amc23:0"] * 16
    acc = [1.0] * 4 + [0.0] * 12 + [1.0] * 8 + [0.0] * 8

    metrics = trainer._val_metrics_update(
        data_sources=data_sources,
        sample_uids=sample_uids,
        reward_extra_infos_dict={"acc": acc, "reward": acc},
        sample_turns=[],
        metric_prefix="test",
    )

    assert metrics["test-core/aime24/acc/mean@16"] == 0.25
    assert metrics["test-core/amc23/acc/mean@16"] == 0.5


def test_validate_test_benchmarks_uses_its_own_dataloader_and_sampling_mode_on_cpu():
    trainer = object.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "test_benchmarks_max_response_length": 128,
                "log_test_generations": 3,
                "test_data_dir": None,
            },
            "actor_rollout_ref": {"rollout": {"response_length": 256}},
        }
    )
    trainer.test_benchmark_dataloader = object()
    trainer.test_benchmark_generations_logger = object()
    captured = {}

    def _validate(**kwargs):
        captured.update(kwargs)
        return {"test-core/aime24/acc/mean@16": 0.5}

    trainer._validate = _validate

    metrics = trainer._validate_test_benchmarks()

    assert metrics == {"test-core/aime24/acc/mean@16": 0.5}
    assert captured["dataloader"] is trainer.test_benchmark_dataloader
    assert captured["evaluation_mode"] == "test"
    assert captured["metric_prefix"] == "test"
    assert captured["generation_max_tokens"] == 128
    assert captured["log_generations"] == 3
    assert captured["validation_data_dir"] == ""


class _PairedEvaluationTokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        values = token_ids.detach().cpu().tolist() if isinstance(token_ids, torch.Tensor) else list(token_ids)
        return " ".join(str(value) for value in values if value != self.pad_token_id)


class _PairedEvaluationGenerationManager:
    def __init__(self):
        self.calls = 0
        self.meta_infos = []

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        self.calls += 1
        self.meta_infos.append(dict(prompts.meta_info))
        assert prompts.meta_info.get("repair_for_evaluation") is True
        return DataProto.from_single_dict(
            {
                "responses": torch.tensor([[31, 0], [31, 0]], dtype=torch.long),
                "response_mask": torch.tensor([[1, 0], [1, 0]], dtype=torch.long),
                "rm_scores": torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32),
                "acc": np.array([1.0, 1.0], dtype=object),
                "teacher_intervention_outcome": np.array(
                    [
                        {
                            "enabled": True,
                            "evaluation_pair": True,
                            "raw_response": "raw-wrong",
                            "raw_reward_score": 0.0,
                            "raw_reward_extra_info": {"acc": 0.0},
                            "raw_response_length": 2,
                            "raw_finish_reason": "length",
                            "raw_correct": False,
                            "final_response_length": 1,
                            "final_finish_reason": "stop",
                            "final_correct": True,
                            "repair_executed": True,
                            "intervention_count": 1,
                            "transition": "pre_wrong_post_correct",
                        },
                        {
                            "enabled": True,
                            "evaluation_pair": True,
                            "raw_response": "raw-correct",
                            "raw_reward_score": 1.0,
                            "raw_reward_extra_info": {"acc": 1.0},
                            "raw_response_length": 1,
                            "raw_finish_reason": "stop",
                            "raw_correct": True,
                            "final_response_length": 1,
                            "final_finish_reason": "stop",
                            "final_correct": True,
                            "repair_executed": False,
                            "intervention_count": 0,
                            "transition": "not_repaired",
                        },
                    ],
                    dtype=object,
                ),
                "teacher_interventions": np.array(
                    [[{"start": 0, "teacher_tokens": 1}], []],
                    dtype=object,
                ),
                "teacher_intervention_outcome_case": np.array(
                    [
                        {
                            "transition": "pre_wrong_post_correct",
                            "case_rank": 0.1,
                            "repair_segments": [{"teacher_segment": "31"}],
                        },
                        None,
                    ],
                    dtype=object,
                ),
            },
            meta_info={"reward_extra_keys": ["acc"]},
        )


def _paired_evaluation_trainer(manager):
    trainer = object.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {"log_val_generations": 0, "validation_data_dir": None},
            "data": {"validation_reward_data_source": None},
            "actor_rollout_ref": {
                "rollout": {
                    "response_length": 2,
                    "agent": {"num_workers": 1},
                    "val_kwargs": {"n": 1, "do_sample": False},
                    "test_kwargs": {"n": 1, "do_sample": False},
                }
            },
        }
    )
    trainer.distillation_config = SimpleNamespace(
        intervention=SimpleNamespace(
            enabled=True,
            repair_for_evaluation=True,
            outcome_aware_candidate_selection_enabled=False,
        ),
        diagnostics=SimpleNamespace(outcome_case_samples_per_transition=8),
    )
    trainer.async_rollout_manager = manager
    trainer.tokenizer = _PairedEvaluationTokenizer()
    trainer.use_rm = False
    trainer.global_steps = 0
    trainer.val_dataloader = [
        {
            "prompts": torch.tensor([[101, 102], [103, 104]], dtype=torch.long),
            "uid": np.array(["uid-1", "uid-2"], dtype=object),
            "data_source": np.array(["math", "math"], dtype=object),
            "reward_model": np.array(
                [{"ground_truth": "one"}, {"ground_truth": "two"}],
                dtype=object,
            ),
        }
    ]
    trainer.validation_generations_logger = object()
    trainer._metric_logging_settings = SimpleNamespace(mode="balanced")
    trainer._metric_archive_writer = object()
    trainer._pending_event_record_archives = []
    trainer._event_record_archive_warning_emitted = False
    return trainer


def test_paired_repair_validation_uses_one_raw_rollout_and_preserves_raw_namespace_on_cpu():
    manager = _PairedEvaluationGenerationManager()
    trainer = _paired_evaluation_trainer(manager)

    metrics = trainer._validate()

    assert manager.calls == 1
    assert metrics["val-core/math/acc/mean@1"] == 0.5
    assert metrics["val-repair-core/math/acc/mean@1"] == 1.0
    assert metrics["val-repair-core/math/acc/paired_delta@1"] == 0.5
    assert "val-repair-aux/math/outcome/pre_wrong_post_correct_count" in metrics
    assert len(trainer._pending_event_record_archives) == 1
    pending = trainer._pending_event_record_archives[0]
    assert pending.metric_key == "val-repair/paired_outcomes"
    assert len(pending.records) == 2
    assert pending.records[0]["raw_output"] == "raw-wrong"
    assert pending.records[0]["final_output"] == "31"
    assert "raw_output" not in pending.records[1]


def test_paired_repair_evaluation_rejects_teacher_or_baseline_manager_on_cpu():
    manager = _PairedEvaluationGenerationManager()
    trainer = _paired_evaluation_trainer(manager)
    other_manager = object()

    with pytest.raises(ValueError, match="student generation manager"):
        trainer._validate(generation_manager=other_manager, repair_for_evaluation=True)


def test_outcome_aware_repair_evaluation_uses_explicit_oracle_namespace_on_cpu():
    manager = _PairedEvaluationGenerationManager()
    trainer = _paired_evaluation_trainer(manager)
    trainer.distillation_config.intervention.outcome_aware_candidate_selection_enabled = True

    metrics = trainer._validate()

    assert manager.calls == 1
    assert metrics["val-oracle-repair-core/math/acc/mean@1"] == 1.0
    assert metrics["val-oracle-repair-core/math/acc/paired_delta@1"] == 0.5
    assert "val-repair-core/math/acc/mean@1" not in metrics
    assert trainer._pending_event_record_archives[0].metric_key == "val-oracle-repair/paired_outcomes"
