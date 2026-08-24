#!/usr/bin/env bash

set -x

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

DEFAULT_EXPERIMENT_NAME=${EXPERIMENT_NAME:-dmpd--teacher_rejection--future_gap--train_on_teacher--loss_opd}

teacher_intervention_segment_length=${TEACHER_INTERVENTION_SEGMENT_LENGTH:-128}
teacher_intervention_teacher_length=${TEACHER_INTERVENTION_TEACHER_LENGTH:-128}
teacher_intervention_gap_metric=${TEACHER_INTERVENTION_GAP_METRIC:-teacher_rejection}
teacher_intervention_gap_aggregation=${TEACHER_INTERVENTION_GAP_AGGREGATION:-mean}
teacher_intervention_gap_threshold=${TEACHER_INTERVENTION_GAP_THRESHOLD:-0.3}
teacher_intervention_max_count=${TEACHER_INTERVENTION_MAX_COUNT:-1}
teacher_intervention_train_teacher=${TEACHER_INTERVENTION_TRAIN_TEACHER:-True}
teacher_intervention_loss_mode=${TEACHER_INTERVENTION_LOSS_MODE:-opd}
teacher_intervention_future_gap_window=${TEACHER_INTERVENTION_FUTURE_GAP_WINDOW:-64}
teacher_intervention_future_gap_min_valid=${TEACHER_INTERVENTION_FUTURE_GAP_MIN_VALID:-32}
teacher_intervention_future_gap_max_threshold=${TEACHER_INTERVENTION_FUTURE_GAP_MAX_THRESHOLD:-null}
teacher_intervention_future_gap_min_reduction=${TEACHER_INTERVENTION_FUTURE_GAP_MIN_REDUCTION:-0.0}
teacher_intervention_min_trainable_tokens=${TEACHER_INTERVENTION_MIN_TRAINABLE_TOKENS:-1}
teacher_intervention_temperature=${TEACHER_INTERVENTION_TEMPERATURE:-0.6}
teacher_intervention_top_p=${TEACHER_INTERVENTION_TOP_P:-0.95}
teacher_intervention_top_k=${TEACHER_INTERVENTION_TOP_K:-20}
dmpd_collect_outcome_transitions=${DMPD_COLLECT_OUTCOME_TRANSITIONS:-True}
dmpd_outcome_cases_per_transition=${DMPD_OUTCOME_CASES_PER_TRANSITION:-8}

METHOD_HYDRA_OVERRIDES=(
    "distillation.intervention.enabled=True"
    "distillation.intervention.student_segment_length=${teacher_intervention_segment_length}"
    "distillation.intervention.teacher_segment_length=${teacher_intervention_teacher_length}"
    "distillation.intervention.gap_metric=${teacher_intervention_gap_metric}"
    "distillation.intervention.gap_aggregation=${teacher_intervention_gap_aggregation}"
    "distillation.intervention.gap_threshold=${teacher_intervention_gap_threshold}"
    "distillation.intervention.max_interventions_per_sample=${teacher_intervention_max_count}"
    "distillation.intervention.confirmation_mode=future_gap"
    "distillation.intervention.future_gap_window_tokens=${teacher_intervention_future_gap_window}"
    "distillation.intervention.future_gap_min_valid_tokens=${teacher_intervention_future_gap_min_valid}"
    "distillation.intervention.future_gap_max_threshold=${teacher_intervention_future_gap_max_threshold}"
    "distillation.intervention.future_gap_min_reduction=${teacher_intervention_future_gap_min_reduction}"
    "distillation.intervention.train_on_teacher_segments=${teacher_intervention_train_teacher}"
    "distillation.intervention.teacher_segment_loss_mode=${teacher_intervention_loss_mode}"
    "distillation.intervention.min_trainable_tokens_after_repair=${teacher_intervention_min_trainable_tokens}"
    "distillation.intervention.teacher_temperature=${teacher_intervention_temperature}"
    "distillation.intervention.teacher_top_p=${teacher_intervention_top_p}"
    "distillation.intervention.teacher_top_k=${teacher_intervention_top_k}"
    "distillation.diagnostics.collect_intervention_outcome_transitions=${dmpd_collect_outcome_transitions}"
    "distillation.diagnostics.outcome_case_samples_per_transition=${dmpd_outcome_cases_per_transition}"
)

# shellcheck source=run/common.sh
source "$SCRIPT_DIR/common.sh"
run_npu_opd_job
