#!/usr/bin/env bash

set -x

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

DEFAULT_EXPERIMENT_NAME=${EXPERIMENT_NAME:-relay-opd}

teacher_intervention_max_count=${TEACHER_INTERVENTION_MAX_COUNT:-2}
teacher_intervention_temperature=${TEACHER_INTERVENTION_TEMPERATURE:-1.0}
teacher_intervention_top_p=${TEACHER_INTERVENTION_TOP_P:-1.0}
teacher_intervention_top_k=${TEACHER_INTERVENTION_TOP_K:--1}

# Relay-OPD eab21451f99e1a40fbb244f556de766d153c88f5, emitted-action objective.
METHOD_HYDRA_OVERRIDES=(
    "algorithm.adv_estimator=grpo"
    "distillation.use_actor_loss=True"
    "distillation.distillation_loss.loss_mode=k1"
    "distillation.distillation_loss.top_k_strategy=only_stu"
    "distillation.distillation_loss.use_task_rewards=False"
    "distillation.distillation_loss.use_policy_gradient=True"
    "distillation.distillation_loss.loss_max_clamp=null"
    "distillation.intervention.enabled=True"
    "distillation.intervention.strategy=relay"
    "distillation.intervention.max_interventions_per_sample=${teacher_intervention_max_count}"
    "distillation.intervention.train_on_teacher_segments=True"
    "distillation.intervention.teacher_temperature=${teacher_intervention_temperature}"
    "distillation.intervention.teacher_top_p=${teacher_intervention_top_p}"
    "distillation.intervention.teacher_top_k=${teacher_intervention_top_k}"
    "distillation.diagnostics.collect_topk_overlap=False"
    "distillation.diagnostics.collect_rethinking_opd_metrics=False"
    "distillation.diagnostics.collect_source_position_metrics=False"
    "distillation.diagnostics.collect_intervention_boundary_metrics=False"
)

# shellcheck source=run/common.sh
source "$SCRIPT_DIR/common.sh"
run_npu_opd_job
