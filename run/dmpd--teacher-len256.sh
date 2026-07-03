#!/usr/bin/env bash

set -x

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

DEFAULT_EXPERIMENT_NAME=${EXPERIMENT_NAME:-dmpd--teacher-len256}

teacher_intervention_segment_length=${TEACHER_INTERVENTION_SEGMENT_LENGTH:-128}
teacher_intervention_teacher_length=${TEACHER_INTERVENTION_TEACHER_LENGTH:-256}
teacher_intervention_gap_metric=${TEACHER_INTERVENTION_GAP_METRIC:-k3}
teacher_intervention_gap_threshold=${TEACHER_INTERVENTION_GAP_THRESHOLD:-0.3}
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
    "distillation.intervention.gap_threshold=${teacher_intervention_gap_threshold}"
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
