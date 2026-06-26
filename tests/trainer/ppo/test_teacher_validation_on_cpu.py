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

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    _build_validation_sample_records,
    _compute_validation_format_metrics,
    _TeacherValidationGenerationManager,
)
from verl.utils.tracking import ValidationGenerationsLogger


def test_teacher_validation_format_metrics_on_cpu():
    metrics = _compute_validation_format_metrics(
        ["Answer: 42", r"The final result is \boxed{7}.", "No final answer"],
        {"pred": ["42", "[INVALID]", None]},
    )

    assert metrics["format/answer_pattern_rate"] == pytest.approx(1 / 3)
    assert metrics["format/boxed_pattern_rate"] == pytest.approx(1 / 3)
    assert metrics["format/invalid_pred_rate"] == pytest.approx(2 / 3)


def test_teacher_validation_format_metrics_without_pred_on_cpu():
    metrics = _compute_validation_format_metrics(["Answer: 42"], {})

    assert metrics["format/answer_pattern_rate"] == 1.0
    assert metrics["format/boxed_pattern_rate"] == 0.0
    assert math.isnan(metrics["format/invalid_pred_rate"])


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
        }
    ]


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
