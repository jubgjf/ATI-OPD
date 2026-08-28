#!/usr/bin/env bash

set -x

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

DEFAULT_EXPERIMENT_NAME=${EXPERIMENT_NAME:-dmpd--train_on_teacher--loss_opd--prune_future}

teacher_intervention_segment_length=${TEACHER_INTERVENTION_SEGMENT_LENGTH:-128}
teacher_intervention_teacher_length=${TEACHER_INTERVENTION_TEACHER_LENGTH:-128}
teacher_intervention_gap_metric=${TEACHER_INTERVENTION_GAP_METRIC:-k3}
teacher_intervention_gap_distribution_topk=${TEACHER_INTERVENTION_GAP_DISTRIBUTION_TOPK:-16}
teacher_intervention_gap_aggregation=${TEACHER_INTERVENTION_GAP_AGGREGATION:-mean}
teacher_intervention_gap_threshold=${TEACHER_INTERVENTION_GAP_THRESHOLD:-0.15}
teacher_intervention_max_count=${TEACHER_INTERVENTION_MAX_COUNT:-1}
teacher_intervention_confirmation_mode=${TEACHER_INTERVENTION_CONFIRMATION_MODE:-none}
teacher_intervention_train_teacher=${TEACHER_INTERVENTION_TRAIN_TEACHER:-True}
teacher_intervention_loss_mode=${TEACHER_INTERVENTION_LOSS_MODE:-opd}
teacher_intervention_min_trainable_tokens=${TEACHER_INTERVENTION_MIN_TRAINABLE_TOKENS:-1}
teacher_intervention_temperature=${TEACHER_INTERVENTION_TEMPERATURE:-0.6}
teacher_intervention_top_p=${TEACHER_INTERVENTION_TOP_P:-0.95}
teacher_intervention_top_k=${TEACHER_INTERVENTION_TOP_K:-20}
teacher_intervention_prune_mode=${TEACHER_INTERVENTION_PRUNE_FUTURE_MODE:-${TEACHER_INTERVENTION_PRUNE_MODE:-adaptive_future}}
teacher_intervention_prune_future_window_tokens=${TEACHER_INTERVENTION_PRUNE_FUTURE_WINDOW_TOKENS:-64}
teacher_intervention_prune_future_min_valid_tokens=${TEACHER_INTERVENTION_PRUNE_FUTURE_MIN_VALID_TOKENS:-32}
teacher_intervention_prune_future_bad_window_persistence=${TEACHER_INTERVENTION_PRUNE_FUTURE_BAD_WINDOW_PERSISTENCE:-2}
teacher_intervention_prune_future_healthy_quantile=${TEACHER_INTERVENTION_PRUNE_FUTURE_HEALTHY_QUANTILE:-0.90}
teacher_intervention_prune_future_calibration_window_blocks=${TEACHER_INTERVENTION_PRUNE_FUTURE_CALIBRATION_WINDOW_BLOCKS:-2048}
teacher_intervention_prune_future_calibration_min_blocks=${TEACHER_INTERVENTION_PRUNE_FUTURE_CALIBRATION_MIN_BLOCKS:-128}
teacher_intervention_prune_future_max_healthy_blocks_per_sample=${TEACHER_INTERVENTION_PRUNE_FUTURE_MAX_HEALTHY_BLOCKS_PER_SAMPLE:-8}
teacher_intervention_prune_future_initial_threshold=${TEACHER_INTERVENTION_PRUNE_FUTURE_INITIAL_THRESHOLD:-null}
dmpd_collect_outcome_transitions=${DMPD_COLLECT_OUTCOME_TRANSITIONS:-True}
dmpd_outcome_cases_per_transition=${DMPD_OUTCOME_CASES_PER_TRANSITION:-8}
TEACHER_INTERVENTION_SKIP_CORRECT=${TEACHER_INTERVENTION_SKIP_CORRECT:-False}
TEACHER_INTERVENTION_PRUNE=${TEACHER_INTERVENTION_PRUNE:-True}

METHOD_HYDRA_OVERRIDES=(
    "distillation.intervention.enabled=True"
    "distillation.intervention.student_segment_length=${teacher_intervention_segment_length}"
    "distillation.intervention.teacher_segment_length=${teacher_intervention_teacher_length}"
    "distillation.intervention.gap_metric=${teacher_intervention_gap_metric}"
    "distillation.intervention.gap_distribution_topk=${teacher_intervention_gap_distribution_topk}"
    "distillation.intervention.gap_aggregation=${teacher_intervention_gap_aggregation}"
    "distillation.intervention.gap_threshold=${teacher_intervention_gap_threshold}"
    "distillation.intervention.max_interventions_per_sample=${teacher_intervention_max_count}"
    "distillation.intervention.confirmation_mode=${teacher_intervention_confirmation_mode}"
    "distillation.intervention.outcome_aware_candidate_selection_enabled=False"
    "distillation.intervention.train_on_teacher_segments=${teacher_intervention_train_teacher}"
    "distillation.intervention.teacher_segment_loss_mode=${teacher_intervention_loss_mode}"
    "distillation.intervention.min_trainable_tokens_after_repair=${teacher_intervention_min_trainable_tokens}"
    "distillation.intervention.teacher_temperature=${teacher_intervention_temperature}"
    "distillation.intervention.teacher_top_p=${teacher_intervention_top_p}"
    "distillation.intervention.teacher_top_k=${teacher_intervention_top_k}"
    "distillation.intervention.prune_mode=${teacher_intervention_prune_mode}"
    "distillation.intervention.prune_future_window_tokens=${teacher_intervention_prune_future_window_tokens}"
    "distillation.intervention.prune_future_min_valid_tokens=${teacher_intervention_prune_future_min_valid_tokens}"
    "distillation.intervention.prune_future_bad_window_persistence=${teacher_intervention_prune_future_bad_window_persistence}"
    "distillation.intervention.prune_future_healthy_quantile=${teacher_intervention_prune_future_healthy_quantile}"
    "distillation.intervention.prune_future_calibration_window_blocks=${teacher_intervention_prune_future_calibration_window_blocks}"
    "distillation.intervention.prune_future_calibration_min_blocks=${teacher_intervention_prune_future_calibration_min_blocks}"
    "distillation.intervention.prune_future_max_healthy_blocks_per_sample=${teacher_intervention_prune_future_max_healthy_blocks_per_sample}"
    "distillation.intervention.prune_future_initial_threshold=${teacher_intervention_prune_future_initial_threshold}"
    "distillation.diagnostics.collect_intervention_outcome_transitions=${dmpd_collect_outcome_transitions}"
    "distillation.diagnostics.outcome_case_samples_per_transition=${dmpd_outcome_cases_per_transition}"
)

# shellcheck source=run/common.sh
source "$SCRIPT_DIR/common.sh"
run_npu_opd_job
