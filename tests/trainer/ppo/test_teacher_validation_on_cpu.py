# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    _build_generation_archive_records,
    _build_training_intervention_outcome_archive_records,
    _build_validation_sample_records,
    _compute_reward_format_metrics,
    _compute_validation_format_metrics,
    _TeacherValidationGenerationManager,
)
from verl.utils.tracking import ValidationGenerationsLogger


def test_teacher_validation_format_metrics_on_cpu():
    metrics = _compute_validation_format_metrics(
        ["**Answer:** 42", r"The final result is \boxed{7}.", "No final answer"],
        {"pred": ["42", "7", None]},
    )

    assert metrics["format/answer_pattern_rate"] == pytest.approx(1 / 3)
    assert metrics["format/boxed_pattern_rate"] == pytest.approx(1 / 3)
    assert metrics["format/accepted_answer_format_rate"] == pytest.approx(2 / 3)
    assert metrics["format/invalid_pred_rate"] == pytest.approx(1 / 3)


def test_teacher_validation_format_metrics_without_pred_on_cpu():
    metrics = _compute_validation_format_metrics(["Answer: 42"], {})

    assert metrics["format/answer_pattern_rate"] == 1.0
    assert metrics["format/boxed_pattern_rate"] == 0.0
    assert metrics["format/accepted_answer_format_rate"] == 1.0
    assert math.isnan(metrics["format/invalid_pred_rate"])


def test_training_reward_format_metrics_on_cpu():
    metrics = _compute_reward_format_metrics(
        {
            "has_answer_pattern": np.array([True, False, False], dtype=object),
            "has_boxed_pattern": np.array([False, True, False], dtype=object),
            "has_accepted_answer_format": np.array([True, True, False], dtype=object),
            "invalid_pred": np.array([False, False, True], dtype=object),
        }
    )

    assert metrics["format/answer_pattern_rate"] == pytest.approx(1 / 3)
    assert metrics["format/boxed_pattern_rate"] == pytest.approx(1 / 3)
    assert metrics["format/accepted_answer_format_rate"] == pytest.approx(2 / 3)
    assert metrics["format/invalid_pred_rate"] == pytest.approx(1 / 3)


def test_training_reward_format_metrics_can_fallback_to_pred_on_cpu():
    metrics = _compute_reward_format_metrics({"pred": np.array(["42", "[INVALID]", ""], dtype=object)})

    assert metrics == {"format/invalid_pred_rate": pytest.approx(2 / 3)}


def test_teacher_validation_sample_records_include_format_fields_on_cpu():
    records = _build_validation_sample_records(
        inputs=["question"],
        outputs=[r"Answer: \boxed{42}"],
        gts=["42"],
        scores=[1.0],
        reward_extra_infos_dict={"acc": [1.0], "pred": ["42"]},
    )

    assert records == [
        {
            "input": "question",
            "output": r"Answer: \boxed{42}",
            "ground_truth": "42",
            "score": 1.0,
            "acc": 1.0,
            "pred": "42",
            "has_answer_pattern": True,
            "has_boxed_pattern": True,
            "has_accepted_answer_format": True,
        }
    ]


def test_generation_archive_records_preserve_complete_evaluation_fields_on_cpu():
    records = _build_generation_archive_records(
        evaluation_scope="test",
        global_step=8,
        inputs=["question-1", "question-2"],
        outputs=["answer-1", "answer-2"],
        gts=["one", "two"],
        scores=[1.0, 0.0],
        reward_extra_infos_dict={"acc": [1.0, 0.0], "pred": ["one", "wrong"], "ignored": [1.0]},
        response_lengths=[11, 22],
        finish_reasons=["stop", "length"],
        data_sources=["aime", "amc"],
        sample_uids=np.array(["uid-1", "uid-2"], dtype=object),
    )

    assert records == [
        {
            "evaluation_scope": "test",
            "global_step": 8,
            "sample_index": 0,
            "input": "question-1",
            "output": "answer-1",
            "ground_truth": "one",
            "score": 1.0,
            "response_token_count": 11,
            "finish_reason": "stop",
            "data_source": "aime",
            "uid": "uid-1",
            "acc": 1.0,
            "pred": "one",
        },
        {
            "evaluation_scope": "test",
            "global_step": 8,
            "sample_index": 1,
            "input": "question-2",
            "output": "answer-2",
            "ground_truth": "two",
            "score": 0.0,
            "response_token_count": 22,
            "finish_reason": "length",
            "data_source": "amc",
            "uid": "uid-2",
            "acc": 0.0,
            "pred": "wrong",
        },
    ]


def test_training_outcome_archive_keeps_all_compact_rows_and_only_selected_case_text_on_cpu():
    records = _build_training_intervention_outcome_archive_records(
        global_step=7,
        outcomes=[
            {
                "enabled": True,
                "raw_correct": False,
                "final_correct": True,
                "transition": "pre_wrong_post_correct",
                "raw_reward_score": 0.0,
                "final_reward_score": 1.0,
                "_raw_response_ids": [1, 2],
            },
            {
                "enabled": True,
                "raw_correct": True,
                "final_correct": True,
                "transition": "not_repaired",
                "raw_reward_score": 1.0,
                "final_reward_score": 1.0,
            },
        ],
        interventions=[[{"start": 4, "teacher_tokens": 2}], []],
        proposals=[[], []],
        skipped_interventions=[[], [{"reason": "below_threshold"}]],
        cases=[
            {
                "transition": "pre_wrong_post_correct",
                "raw_response": "wrong reasoning",
                "final_response": "fixed reasoning",
                "repair_segments": [{"teacher_segment": "fix"}],
            },
            None,
        ],
        sample_uids=["uid-1", "uid-2"],
        data_sources=["math", "math"],
        raw_prompts=[[{"role": "user", "content": "question 1"}], [{"role": "user", "content": "question 2"}]],
        reward_models=[{"ground_truth": "one"}, {"ground_truth": "two"}],
    )

    assert len(records) == 2
    assert records[0]["case_included"] is True
    assert records[0]["raw_response"] == "wrong reasoning"
    assert records[0]["final_response"] == "fixed reasoning"
    assert records[0]["ground_truth"] == "one"
    assert "_raw_response_ids" not in records[0]["outcome"]
    assert records[1]["case_included"] is False
    assert records[1]["skipped_interventions"] == [{"reason": "below_threshold"}]
    assert "raw_response" not in records[1]
    assert "raw_prompt" not in records[1]


def test_training_outcome_archive_queue_is_independent_of_diagnostics_event_interval_on_cpu():
    trainer = object.__new__(RayPPOTrainer)
    trainer.global_steps = 3
    trainer.distillation_config = SimpleNamespace(
        diagnostics=SimpleNamespace(
            enabled=True,
            collect_intervention_metrics=True,
            collect_intervention_outcome_transitions=True,
            log_interval=2,
        )
    )
    trainer._metric_logging_settings = SimpleNamespace(mode="balanced")
    trainer._metric_archive_writer = object()
    trainer._pending_event_record_archives = []
    trainer._event_record_archive_warning_emitted = False
    batch = DataProto.from_single_dict(
        {
            "dummy": torch.ones((2, 1)),
            "teacher_intervention_outcome": np.array(
                [
                    {"enabled": True, "transition": "pre_wrong_post_correct"},
                    {"enabled": True, "transition": "not_repaired"},
                ],
                dtype=object,
            ),
            "teacher_interventions": np.array([[{"start": 1}], []], dtype=object),
            "teacher_intervention_outcome_case": np.array([None, None], dtype=object),
            "uid": np.array(["uid-1", "uid-2"], dtype=object),
            "data_source": np.array(["math", "math"], dtype=object),
        }
    )

    trainer._queue_training_intervention_outcome_archive(batch=batch, global_steps=3)
    assert len(trainer._pending_event_record_archives) == 1
    pending = trainer._pending_event_record_archives[0]
    assert pending.global_step == 3
    assert pending.metric_key == "opd_intervention/outcome_pairs"
    assert len(pending.records) == 2


def test_teacher_validation_metric_prefix_does_not_overwrite_student_on_cpu():
    trainer = object.__new__(RayPPOTrainer)
    metrics = trainer._val_metrics_update(
        data_sources=np.array(["math_dapo", "math_dapo"], dtype=object),
        sample_uids=["uid-1", "uid-2"],
        reward_extra_infos_dict={"acc": [1.0, 0.0], "reward": [1.0, -1.0]},
        sample_turns=[],
        metric_prefix="teacher-val",
    )

    assert "teacher-val-core/math_dapo/acc/mean@1" in metrics
    assert "val-core/math_dapo/acc/mean@1" not in metrics


def test_teacher_validation_before_train_uses_val_kwargs_on_cpu():
    trainer = object.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "teacher_val_before_train": True,
                "teacher_val_max_samples": 7,
                "log_teacher_val_generations": 0,
                "validation_data_dir": None,
            },
            "distillation": {"enabled": True},
            "actor_rollout_ref": {"rollout": {"response_length": 128, "val_kwargs": {"n": 5, "do_sample": True}}},
        }
    )
    trainer.use_teacher_policy = True
    trainer.distillation_config = object()
    trainer.teacher_validation_generations_logger = object()

    class _Manager:
        pass

    manager = _Manager()
    trainer._get_teacher_validation_manager = lambda: manager
    captured = {}

    def _validate(**kwargs):
        captured.update(kwargs)
        return {"teacher-val-core/math_dapo/acc/mean@5": 1.0}

    trainer._validate = _validate

    metrics = trainer._validate_teacher_before_train()

    assert metrics == {"teacher-val-core/math_dapo/acc/mean@5": 1.0}
    assert captured["generation_manager"] is manager
    assert captured["metric_prefix"] == "teacher-val"
    assert captured["max_samples"] == 7
    assert "repeat_times" not in captured
    assert "do_sample" not in captured
    assert captured["generation_max_tokens"] == 128


def test_validation_generation_logger_supports_teacher_record_shape_on_cpu():
    sample = {
        "input": "question",
        "output": "Answer: 42",
        "ground_truth": "42",
        "score": 1.0,
        "acc": 1.0,
        "pred": "42",
        "has_answer_pattern": True,
        "has_boxed_pattern": False,
        "has_accepted_answer_format": True,
    }
    fields = ValidationGenerationsLogger._sample_fields([sample])

    assert fields == list(sample.keys())
    assert ValidationGenerationsLogger._sample_to_values(sample, fields) == list(sample.values())


class _FakeAgentLoopManager:
    def __init__(self, marker: int):
        self.marker = marker

    @classmethod
    def create(cls, *, llm_client, **_kwargs):
        return cls(marker=llm_client)

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        ids = prompts.batch["ids"]
        return DataProto.from_single_dict(
            {"responses": ids * 10 + self.marker},
            meta_info={"timing": {"agent_loop/generate_sequences/mean": float(self.marker)}},
        )


def test_teacher_validation_manager_routes_multi_teacher_batches_on_cpu():
    manager = _TeacherValidationGenerationManager(
        config={},
        agent_loop_manager_cls=_FakeAgentLoopManager,
        teacher_clients={"a": 1, "b": 2},
        teacher_key="data_source",
    )
    prompts = DataProto.from_single_dict(
        {
            "ids": torch.tensor([[0], [1], [2], [3]], dtype=torch.long),
            "data_source": np.array(["b", "a", "b", "a"], dtype=object),
        }
    )

    output = manager.generate_sequences(prompts)

    assert output.batch["responses"].tolist() == [[2], [11], [22], [31]]
    assert output.meta_info["timing"]["agent_loop/generate_sequences/mean"] == pytest.approx(1.5)
