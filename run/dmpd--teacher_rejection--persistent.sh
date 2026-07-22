#!/usr/bin/env bash

set -x

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

DEFAULT_EXPERIMENT_NAME=${EXPERIMENT_NAME:-dmpd--teacher_rejection--persistent}

teacher_intervention_segment_length=${TEACHER_INTERVENTION_SEGMENT_LENGTH:-128}
teacher_intervention_teacher_length=${TEACHER_INTERVENTION_TEACHER_LENGTH:-128}
teacher_intervention_gap_metric=${TEACHER_INTERVENTION_GAP_METRIC:-teacher_rejection}
teacher_intervention_gap_aggregation=${TEACHER_INTERVENTION_GAP_AGGREGATION:-mean}
teacher_intervention_gap_threshold=${TEACHER_INTERVENTION_GAP_THRESHOLD:-1.0}
teacher_intervention_persistence_enabled=${TEACHER_INTERVENTION_PERSISTENCE_ENABLED:-True}
teacher_intervention_persistence_window=${TEACHER_INTERVENTION_PERSISTENCE_WINDOW:-64}
teacher_intervention_persistence_threshold=${TEACHER_INTERVENTION_PERSISTENCE_THRESHOLD:-${teacher_intervention_gap_threshold}}
teacher_intervention_persistence_min_tokens=${TEACHER_INTERVENTION_PERSISTENCE_MIN_TOKENS:-$(((teacher_intervention_persistence_window + 1) / 2))}
teacher_intervention_max_count=${TEACHER_INTERVENTION_MAX_COUNT:-4}
teacher_intervention_train_teacher=${TEACHER_INTERVENTION_TRAIN_TEACHER:-False}
teacher_intervention_min_trainable_tokens=${TEACHER_INTERVENTION_MIN_TRAINABLE_TOKENS:-1}
teacher_intervention_temperature=${TEACHER_INTERVENTION_TEMPERATURE:-0.6}
teacher_intervention_top_p=${TEACHER_INTERVENTION_TOP_P:-0.95}
teacher_intervention_top_k=${TEACHER_INTERVENTION_TOP_K:-20}

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
)

# shellcheck source=run/common.sh
source "$SCRIPT_DIR/common.sh"
run_npu_opd_job
