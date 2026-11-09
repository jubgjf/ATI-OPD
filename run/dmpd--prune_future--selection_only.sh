#!/usr/bin/env bash

set -x

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

DEFAULT_EXPERIMENT_NAME=${EXPERIMENT_NAME:-dmpd--prune_future--selection_only}

teacher_intervention_segment_length=${TEACHER_INTERVENTION_SEGMENT_LENGTH:-128}
teacher_intervention_gap_aggregation=${TEACHER_INTERVENTION_GAP_AGGREGATION:-mean}
teacher_intervention_gap_threshold=${TEACHER_INTERVENTION_GAP_THRESHOLD:-0.15}
teacher_intervention_prune_future_window_tokens=${TEACHER_INTERVENTION_PRUNE_FUTURE_WINDOW_TOKENS:-64}
teacher_intervention_prune_future_min_valid_tokens=${TEACHER_INTERVENTION_PRUNE_FUTURE_MIN_VALID_TOKENS:-32}
teacher_intervention_prune_future_bad_window_persistence=${TEACHER_INTERVENTION_PRUNE_FUTURE_BAD_WINDOW_PERSISTENCE:-2}
teacher_intervention_prune_future_healthy_quantile=${TEACHER_INTERVENTION_PRUNE_FUTURE_HEALTHY_QUANTILE:-0.90}
teacher_intervention_prune_future_calibration_window_blocks=${TEACHER_INTERVENTION_PRUNE_FUTURE_CALIBRATION_WINDOW_BLOCKS:-2048}
teacher_intervention_prune_future_calibration_min_blocks=${TEACHER_INTERVENTION_PRUNE_FUTURE_CALIBRATION_MIN_BLOCKS:-128}
teacher_intervention_prune_future_max_healthy_blocks_per_sample=${TEACHER_INTERVENTION_PRUNE_FUTURE_MAX_HEALTHY_BLOCKS_PER_SAMPLE:-8}
teacher_intervention_prune_future_initial_threshold=${TEACHER_INTERVENTION_PRUNE_FUTURE_INITIAL_THRESHOLD:-null}

METHOD_HYDRA_OVERRIDES=(
    "distillation.intervention.enabled=False"
    "distillation.intervention.prune=True"
    "distillation.intervention.prune_mode=selection_only"
    "distillation.intervention.student_segment_length=${teacher_intervention_segment_length}"
    "distillation.intervention.gap_metric=k3"
    "distillation.intervention.gap_aggregation=${teacher_intervention_gap_aggregation}"
    "distillation.intervention.gap_threshold=${teacher_intervention_gap_threshold}"
    "distillation.intervention.prune_future_window_tokens=${teacher_intervention_prune_future_window_tokens}"
    "distillation.intervention.prune_future_min_valid_tokens=${teacher_intervention_prune_future_min_valid_tokens}"
    "distillation.intervention.prune_future_bad_window_persistence=${teacher_intervention_prune_future_bad_window_persistence}"
    "distillation.intervention.prune_future_healthy_quantile=${teacher_intervention_prune_future_healthy_quantile}"
    "distillation.intervention.prune_future_calibration_window_blocks=${teacher_intervention_prune_future_calibration_window_blocks}"
    "distillation.intervention.prune_future_calibration_min_blocks=${teacher_intervention_prune_future_calibration_min_blocks}"
    "distillation.intervention.prune_future_max_healthy_blocks_per_sample=${teacher_intervention_prune_future_max_healthy_blocks_per_sample}"
    "distillation.intervention.prune_future_initial_threshold=${teacher_intervention_prune_future_initial_threshold}"
)

# shellcheck source=run/common.sh
source "$SCRIPT_DIR/common.sh"
run_npu_opd_job
