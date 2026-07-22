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

import pytest
from omegaconf import OmegaConf

from verl.trainer.distillation.config_validation import validate_distillation_intervention_training_config
from verl.workers.config import DistillationInterventionConfig


def test_teacher_segment_loss_config_defaults_to_legacy_opd_on_cpu():
    config = DistillationInterventionConfig()

    assert config.teacher_segment_loss_mode == "opd"
    assert config.teacher_segment_sft_loss_coef == 1.0
    assert config.gap_metric == "k3"
    assert config.gap_threshold == 1.0
    assert config.teacher_rejection_token_cap == 10.0
    assert config.teacher_rejection_persistence_enabled is False
    assert config.teacher_rejection_persistence_window_tokens == 64
    assert config.teacher_rejection_persistence_threshold is None
    assert config.teacher_rejection_persistence_min_valid_tokens == 32


def test_teacher_segment_loss_config_validates_mode_and_coefficient_on_cpu():
    with pytest.raises(ValueError, match="teacher_segment_loss_mode"):
        DistillationInterventionConfig(teacher_segment_loss_mode="mixed")

    for invalid_coef in (-1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="teacher_segment_sft_loss_coef"):
            DistillationInterventionConfig(teacher_segment_sft_loss_coef=invalid_coef)


def test_teacher_rejection_intervention_config_validates_token_cap_on_cpu():
    config = DistillationInterventionConfig(gap_metric="teacher_rejection", teacher_rejection_token_cap=5.0)
    assert config.teacher_rejection_token_cap == 5.0

    for invalid_cap in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="teacher_rejection_token_cap"):
            DistillationInterventionConfig(
                gap_metric="teacher_rejection",
                teacher_rejection_token_cap=invalid_cap,
            )


def test_teacher_rejection_persistence_config_validates_on_cpu():
    config = DistillationInterventionConfig(
        gap_metric="teacher_rejection",
        teacher_rejection_persistence_enabled=True,
        teacher_rejection_persistence_window_tokens=32,
        teacher_rejection_persistence_threshold=0.5,
        teacher_rejection_persistence_min_valid_tokens=16,
    )
    assert config.teacher_rejection_persistence_enabled is True

    with pytest.raises(ValueError, match="requires gap_metric='teacher_rejection'"):
        DistillationInterventionConfig(teacher_rejection_persistence_enabled=True)

    for field, value in (
        ("teacher_rejection_persistence_window_tokens", 0),
        ("teacher_rejection_persistence_min_valid_tokens", 0),
        ("teacher_rejection_persistence_threshold", -0.1),
        ("teacher_rejection_persistence_threshold", float("inf")),
        ("teacher_rejection_persistence_threshold", float("nan")),
    ):
        with pytest.raises(ValueError, match=field):
            DistillationInterventionConfig(gap_metric="teacher_rejection", **{field: value})

    with pytest.raises(ValueError, match="must not exceed"):
        DistillationInterventionConfig(
            gap_metric="teacher_rejection",
            teacher_rejection_persistence_window_tokens=16,
            teacher_rejection_persistence_min_valid_tokens=17,
        )


def test_distillation_intervention_rejects_train_teacher_segments_with_bypass_mode_on_cpu():
    config = OmegaConf.create(
        {
            "distillation": {
                "enabled": True,
                "intervention": {
                    "enabled": True,
                    "train_on_teacher_segments": True,
                },
            },
            "algorithm": {
                "rollout_correction": {
                    "bypass_mode": True,
                },
            },
        }
    )

    with pytest.raises(ValueError, match="train_on_teacher_segments=true"):
        validate_distillation_intervention_training_config(config)


def test_distillation_intervention_allows_default_decoupled_rollout_correction_on_cpu():
    config = OmegaConf.create(
        {
            "distillation": {
                "enabled": True,
                "intervention": {
                    "enabled": True,
                    "train_on_teacher_segments": True,
                },
            },
            "algorithm": {
                "rollout_correction": {
                    "bypass_mode": False,
                },
            },
        }
    )

    validate_distillation_intervention_training_config(config)
