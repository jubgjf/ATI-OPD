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


def test_teacher_segment_loss_config_validates_mode_and_coefficient_on_cpu():
    with pytest.raises(ValueError, match="teacher_segment_loss_mode"):
        DistillationInterventionConfig(teacher_segment_loss_mode="mixed")

    for invalid_coef in (-1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="teacher_segment_sft_loss_coef"):
            DistillationInterventionConfig(teacher_segment_sft_loss_coef=invalid_coef)


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
