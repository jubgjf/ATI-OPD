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

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SKIP_LAUNCHER = REPO_ROOT / "run" / "dmpd--skip_correct--train_on_teacher--loss_opd--prune_future.sh"
PLAIN_LAUNCHER = REPO_ROOT / "run" / "dmpd--train_on_teacher--loss_opd--prune_future.sh"


def test_prune_future_launchers_pin_method_semantics_and_expose_all_adaptive_overrides_on_cpu():
    skip_source = SKIP_LAUNCHER.read_text(encoding="utf-8")
    plain_source = PLAIN_LAUNCHER.read_text(encoding="utf-8")

    assert "dmpd--skip_correct--train_on_teacher--loss_opd--prune_future" in skip_source
    assert "dmpd--train_on_teacher--loss_opd--prune_future" in plain_source
    assert "TEACHER_INTERVENTION_SKIP_CORRECT=${TEACHER_INTERVENTION_SKIP_CORRECT:-True}" in skip_source
    assert "TEACHER_INTERVENTION_SKIP_CORRECT=${TEACHER_INTERVENTION_SKIP_CORRECT:-False}" in plain_source

    expected_overrides = (
        "distillation.intervention.enabled=True",
        "distillation.intervention.max_interventions_per_sample=${teacher_intervention_max_count}",
        "distillation.intervention.confirmation_mode=${teacher_intervention_confirmation_mode}",
        "distillation.intervention.outcome_aware_candidate_selection_enabled=False",
        "distillation.intervention.train_on_teacher_segments=${teacher_intervention_train_teacher}",
        "distillation.intervention.teacher_segment_loss_mode=${teacher_intervention_loss_mode}",
        "distillation.intervention.prune_mode=${teacher_intervention_prune_mode}",
        "distillation.intervention.prune_future_window_tokens=${teacher_intervention_prune_future_window_tokens}",
        "distillation.intervention.prune_future_min_valid_tokens=${teacher_intervention_prune_future_min_valid_tokens}",
        "distillation.intervention.prune_future_bad_window_persistence=${teacher_intervention_prune_future_bad_window_persistence}",
        "distillation.intervention.prune_future_healthy_quantile=${teacher_intervention_prune_future_healthy_quantile}",
        "distillation.intervention.prune_future_calibration_window_blocks=${teacher_intervention_prune_future_calibration_window_blocks}",
        "distillation.intervention.prune_future_calibration_min_blocks=${teacher_intervention_prune_future_calibration_min_blocks}",
        "distillation.intervention.prune_future_max_healthy_blocks_per_sample=${teacher_intervention_prune_future_max_healthy_blocks_per_sample}",
        "distillation.intervention.prune_future_initial_threshold=${teacher_intervention_prune_future_initial_threshold}",
    )
    for source in (skip_source, plain_source):
        assert "teacher_intervention_max_count=${TEACHER_INTERVENTION_MAX_COUNT:-1}" in source
        assert "teacher_intervention_confirmation_mode=${TEACHER_INTERVENTION_CONFIRMATION_MODE:-none}" in source
        assert "teacher_intervention_train_teacher=${TEACHER_INTERVENTION_TRAIN_TEACHER:-True}" in source
        assert "teacher_intervention_loss_mode=${TEACHER_INTERVENTION_LOSS_MODE:-opd}" in source
        assert (
            "teacher_intervention_prune_mode=${TEACHER_INTERVENTION_PRUNE_FUTURE_MODE:-"
            "${TEACHER_INTERVENTION_PRUNE_MODE:-adaptive_future}}" in source
        )
        assert "TEACHER_INTERVENTION_PRUNE=${TEACHER_INTERVENTION_PRUNE:-True}" in source
        for override in expected_overrides:
            assert override in source
