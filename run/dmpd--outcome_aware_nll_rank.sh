#!/usr/bin/env bash

set -x

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

DEFAULT_EXPERIMENT_NAME=${EXPERIMENT_NAME:-dmpd--teacher_rejection--outcome_aware_nll_rank}

teacher_intervention_segment_length=${TEACHER_INTERVENTION_SEGMENT_LENGTH:-128}
teacher_intervention_teacher_length=${TEACHER_INTERVENTION_TEACHER_LENGTH:-128}
teacher_intervention_gap_metric=${TEACHER_INTERVENTION_GAP_METRIC:-teacher_rejection}
teacher_intervention_gap_aggregation=${TEACHER_INTERVENTION_GAP_AGGREGATION:-mean}
teacher_intervention_gap_threshold=${TEACHER_INTERVENTION_GAP_THRESHOLD:-1.0}
teacher_intervention_persistence_enabled=${TEACHER_INTERVENTION_PERSISTENCE_ENABLED:-False}
teacher_intervention_persistence_window=${TEACHER_INTERVENTION_PERSISTENCE_WINDOW:-64}
teacher_intervention_persistence_threshold=${TEACHER_INTERVENTION_PERSISTENCE_THRESHOLD:-null}
teacher_intervention_persistence_min_tokens=${TEACHER_INTERVENTION_PERSISTENCE_MIN_TOKENS:-32}
teacher_intervention_max_count=${TEACHER_INTERVENTION_MAX_COUNT:-4}
teacher_intervention_train_teacher=${TEACHER_INTERVENTION_TRAIN_TEACHER:-False}
teacher_intervention_min_trainable_tokens=${TEACHER_INTERVENTION_MIN_TRAINABLE_TOKENS:-1}
teacher_intervention_temperature=${TEACHER_INTERVENTION_TEMPERATURE:-0.6}
teacher_intervention_top_p=${TEACHER_INTERVENTION_TOP_P:-0.95}
teacher_intervention_top_k=${TEACHER_INTERVENTION_TOP_K:-20}

outcome_correctness_key=${OUTCOME_CORRECTNESS_KEY:-acc}
outcome_correctness_threshold=${OUTCOME_CORRECTNESS_THRESHOLD:-0.5}
teacher_candidate_count=${TEACHER_CANDIDATE_COUNT:-4}
teacher_nll_calibration_quantile=${TEACHER_NLL_CALIBRATION_QUANTILE:-0.75}
teacher_nll_calibration_window=${TEACHER_NLL_CALIBRATION_WINDOW:-256}
teacher_nll_calibration_min_samples=${TEACHER_NLL_CALIBRATION_MIN_SAMPLES:-16}
teacher_nll_initial_threshold=${TEACHER_NLL_INITIAL_THRESHOLD:-null}
teacher_nll_healthy_probe_probability=${TEACHER_NLL_HEALTHY_PROBE_PROBABILITY:-0.1}

METHOD_HYDRA_OVERRIDES=(
    "distillation.intervention.enabled=True"
    "distillation.intervention.student_segment_length=${teacher_intervention_segment_length}"
    "distillation.intervention.teacher_segment_length=${teacher_intervention_teacher_length}"
    "distillation.intervention.gap_metric=${teacher_intervention_gap_metric}"
    "distillation.intervention.gap_aggregation=${teacher_intervention_gap_aggregation}"
    "distillation.intervention.gap_threshold=${teacher_intervention_gap_threshold}"
    "distillation.intervention.teacher_rejection_persistence_enabled=${teacher_intervention_persistence_enabled}"
    "distillation.intervention.teacher_rejection_persistence_window_tokens=${teacher_intervention_persistence_window}"
    "distillation.intervention.teacher_rejection_persistence_threshold=${teacher_intervention_persistence_threshold}"
    "distillation.intervention.teacher_rejection_persistence_min_valid_tokens=${teacher_intervention_persistence_min_tokens}"
    "distillation.intervention.max_interventions_per_sample=${teacher_intervention_max_count}"
    "distillation.intervention.train_on_teacher_segments=${teacher_intervention_train_teacher}"
    "distillation.intervention.min_trainable_tokens_after_repair=${teacher_intervention_min_trainable_tokens}"
    "distillation.intervention.teacher_temperature=${teacher_intervention_temperature}"
    "distillation.intervention.teacher_top_p=${teacher_intervention_top_p}"
    "distillation.intervention.teacher_top_k=${teacher_intervention_top_k}"
    "distillation.intervention.outcome_aware_candidate_selection_enabled=True"
    "distillation.intervention.outcome_correctness_key=${outcome_correctness_key}"
    "distillation.intervention.outcome_correctness_threshold=${outcome_correctness_threshold}"
    "distillation.intervention.teacher_candidate_count=${teacher_candidate_count}"
    "distillation.intervention.teacher_nll_calibration_quantile=${teacher_nll_calibration_quantile}"
    "distillation.intervention.teacher_nll_calibration_window=${teacher_nll_calibration_window}"
    "distillation.intervention.teacher_nll_calibration_min_samples=${teacher_nll_calibration_min_samples}"
    "distillation.intervention.teacher_nll_initial_threshold=${teacher_nll_initial_threshold}"
    "distillation.intervention.teacher_nll_healthy_probe_probability=${teacher_nll_healthy_probe_probability}"
)

# shellcheck source=run/common.sh
source "$SCRIPT_DIR/common.sh"
run_npu_opd_job
