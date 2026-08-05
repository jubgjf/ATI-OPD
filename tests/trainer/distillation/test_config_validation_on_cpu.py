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
from verl.workers.config import DistillationDiagnosticsConfig, DistillationInterventionConfig, DistillationLossConfig


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
    assert config.outcome_aware_candidate_selection_enabled is False
    assert config.outcome_correctness_key == "acc"
    assert config.teacher_candidate_count == 4
    assert config.teacher_candidate_max_concurrency_per_worker == 4
    assert config.teacher_nll_calibration_quantile == 0.75
    assert config.teacher_nll_calibration_window == 256
    assert config.teacher_nll_calibration_min_samples == 16
    assert config.teacher_nll_initial_threshold is None
    assert config.teacher_nll_healthy_probe_probability == 0.1
    assert config.repair_for_evaluation is False


def test_intervention_outcome_observation_config_defaults_and_validation_on_cpu():
    diagnostics = DistillationDiagnosticsConfig()

    assert diagnostics.collect_intervention_outcome_transitions is False
    assert diagnostics.outcome_case_samples_per_transition == 8
    assert diagnostics.collect_intervention_boundary_metrics is False
    assert diagnostics.intervention_boundary_text_samples_per_step == 8
    assert diagnostics.intervention_boundary_text_context_tokens == 32

    with pytest.raises(ValueError, match="outcome_case_samples_per_transition"):
        DistillationDiagnosticsConfig(outcome_case_samples_per_transition=-1)
    with pytest.raises(ValueError, match="intervention_boundary_text_samples_per_step"):
        DistillationDiagnosticsConfig(intervention_boundary_text_samples_per_step=-1)
    with pytest.raises(ValueError, match="intervention_boundary_text_context_tokens"):
        DistillationDiagnosticsConfig(intervention_boundary_text_context_tokens=0)

    with pytest.raises(ValueError, match="repair_for_evaluation=true"):
        DistillationInterventionConfig(repair_for_evaluation=True)

    enabled = DistillationInterventionConfig(enabled=True, repair_for_evaluation=True)
    assert enabled.repair_for_evaluation is True

    config = OmegaConf.create(
        {
            "distillation": {
                "enabled": True,
                "intervention": {"enabled": True},
                "diagnostics": {"outcome_case_samples_per_transition": -1},
            }
        }
    )
    with pytest.raises(ValueError, match="outcome_case_samples_per_transition"):
        validate_distillation_intervention_training_config(config)


@pytest.mark.parametrize(
    ("intervention_enabled", "diagnostics_enabled", "collect_intervention_metrics", "error"),
    [
        (False, True, True, "intervention.enabled=true"),
        (True, False, True, "diagnostics.enabled=true"),
        (True, True, False, "collect_intervention_metrics=true"),
    ],
)
def test_outcome_transition_cross_config_requires_intervention_diagnostics_and_metrics_on_cpu(
    intervention_enabled,
    diagnostics_enabled,
    collect_intervention_metrics,
    error,
):
    config = OmegaConf.create(
        {
            "distillation": {
                "enabled": True,
                "intervention": {"enabled": intervention_enabled},
                "diagnostics": {
                    "enabled": diagnostics_enabled,
                    "collect_intervention_metrics": collect_intervention_metrics,
                    "collect_intervention_outcome_transitions": True,
                },
            }
        }
    )

    with pytest.raises(ValueError, match=error):
        validate_distillation_intervention_training_config(config)


def test_distillation_loss_config_preserves_legacy_default_and_accepts_student_support_strategies_on_cpu():
    default_config = DistillationLossConfig()
    assert default_config.top_k_strategy == "only_stu"
    assert default_config.teacher_topk == 128
    assert (
        DistillationLossConfig(top_k_strategy="only_stu_rethinking_style").top_k_strategy == "only_stu_rethinking_style"
    )
    teacher_topk_config = DistillationLossConfig(
        topk=16,
        teacher_topk=128,
        top_k_strategy="only_stu_teacher_topk",
    )
    assert teacher_topk_config.topk == 16
    assert teacher_topk_config.teacher_topk == 128

    with pytest.raises(ValueError, match="top_k_strategy"):
        DistillationLossConfig(top_k_strategy="unknown")
    for invalid_teacher_topk in (0, -1):
        with pytest.raises(ValueError, match="teacher_topk must be positive"):
            DistillationLossConfig(teacher_topk=invalid_teacher_topk)
    with pytest.raises(ValueError, match="greater than or equal to topk"):
        DistillationLossConfig(topk=16, teacher_topk=8, top_k_strategy="only_stu_teacher_topk")


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


def test_outcome_aware_candidate_selection_config_validates_on_cpu():
    config = DistillationInterventionConfig(
        enabled=True,
        outcome_aware_candidate_selection_enabled=True,
        gap_metric="teacher_rejection",
        teacher_candidate_count=4,
        teacher_candidate_max_concurrency_per_worker=2,
        teacher_nll_calibration_quantile=0.8,
        teacher_nll_calibration_window=64,
        teacher_nll_calibration_min_samples=8,
        teacher_nll_initial_threshold=2.0,
        teacher_nll_healthy_probe_probability=0.25,
    )
    assert config.outcome_aware_candidate_selection_enabled is True

    with pytest.raises(ValueError, match="requires distillation.intervention.enabled=true"):
        DistillationInterventionConfig(outcome_aware_candidate_selection_enabled=True)
    with pytest.raises(ValueError, match="outcome_correctness_key"):
        DistillationInterventionConfig(outcome_correctness_key="")
    with pytest.raises(ValueError, match="teacher_candidate_count"):
        DistillationInterventionConfig(teacher_candidate_count=0)
    with pytest.raises(ValueError, match="teacher_candidate_max_concurrency_per_worker"):
        DistillationInterventionConfig(teacher_candidate_max_concurrency_per_worker=0)
    for quantile in (0.0, 1.0, float("nan")):
        with pytest.raises(ValueError, match="teacher_nll_calibration_quantile"):
            DistillationInterventionConfig(teacher_nll_calibration_quantile=quantile)
    with pytest.raises(ValueError, match="teacher_nll_calibration_window"):
        DistillationInterventionConfig(teacher_nll_calibration_window=0)
    with pytest.raises(ValueError, match="must not exceed"):
        DistillationInterventionConfig(
            teacher_nll_calibration_window=4,
            teacher_nll_calibration_min_samples=5,
        )
    for threshold in (-0.1, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="teacher_nll_initial_threshold"):
            DistillationInterventionConfig(teacher_nll_initial_threshold=threshold)
    for probability in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="teacher_nll_healthy_probe_probability"):
            DistillationInterventionConfig(teacher_nll_healthy_probe_probability=probability)


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


@pytest.mark.parametrize(
    "intervention",
    [
        {"enabled": False, "gap_metric": "k3"},
        {"enabled": True, "gap_metric": "k3"},
        {"enabled": True, "gap_metric": "teacher_rejection"},
        {
            "enabled": True,
            "gap_metric": "teacher_rejection",
            "teacher_rejection_persistence_enabled": True,
            "teacher_rejection_persistence_window_tokens": 32,
            "teacher_rejection_persistence_min_valid_tokens": 16,
        },
        {"enabled": True, "gap_metric": "topk_distribution_kl"},
        {
            "enabled": True,
            "gap_metric": "k3",
            "outcome_aware_candidate_selection_enabled": True,
        },
        {
            "enabled": True,
            "gap_metric": "teacher_rejection",
            "train_on_teacher_segments": True,
            "teacher_segment_loss_mode": "sft",
        },
    ],
)
@pytest.mark.parametrize("top_k_strategy", ["only_stu_rethinking_style", "only_stu_teacher_topk"])
def test_student_support_strategies_are_compatible_with_dmpd_settings_on_cpu(intervention, top_k_strategy):
    # Instantiate the intervention dataclass as well as the cross-feature
    # validator, so persistence and rejection-specific invariants are covered.
    DistillationInterventionConfig(**intervention)
    config = OmegaConf.create(
        {
            "distillation": {
                "enabled": True,
                "use_actor_loss": False,
                "distillation_loss": {"top_k_strategy": top_k_strategy},
                "intervention": intervention,
            },
            "algorithm": {
                "adv_estimator": "token_reward_direct",
                "rollout_correction": {"bypass_mode": False},
            },
        }
    )

    validate_distillation_intervention_training_config(config)


@pytest.mark.parametrize(
    ("use_actor_loss", "adv_estimator", "bypass_mode", "error"),
    [
        (True, "token_reward_direct", False, "use_actor_loss=false"),
        (False, "gae", False, "token_reward_direct"),
        (False, "token_reward_direct", True, "bypass_mode=false"),
    ],
)
@pytest.mark.parametrize("top_k_strategy", ["only_stu_rethinking_style", "only_stu_teacher_topk"])
def test_student_support_strategies_reject_silent_fallback_training_paths_on_cpu(
    use_actor_loss,
    adv_estimator,
    bypass_mode,
    error,
    top_k_strategy,
):
    config = OmegaConf.create(
        {
            "distillation": {
                "enabled": True,
                "use_actor_loss": use_actor_loss,
                "distillation_loss": {"top_k_strategy": top_k_strategy},
                "intervention": {"enabled": False},
            },
            "algorithm": {
                "adv_estimator": adv_estimator,
                "rollout_correction": {"bypass_mode": bypass_mode},
            },
        }
    )

    with pytest.raises(ValueError, match=error):
        validate_distillation_intervention_training_config(config)
