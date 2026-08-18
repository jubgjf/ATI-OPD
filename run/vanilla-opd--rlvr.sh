#!/usr/bin/env bash

set -x

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

DEFAULT_EXPERIMENT_NAME=${EXPERIMENT_NAME:-vanilla-opd--rlvr}

USE_TASK_REWARDS=True
ADV_ESTIMATOR=token_reward_direct
DISTILLATION_USE_ACTOR_LOSS=False
USE_KL_IN_REWARD=False
RLVR_LOSS_COEF=${RLVR_LOSS_COEF:-0.5}
RLVR_ADV_ESTIMATOR=${RLVR_ADV_ESTIMATOR:-grpo}
RLVR_CREDIT_MODE=${RLVR_CREDIT_MODE:-no_intervention_only}

METHOD_HYDRA_OVERRIDES=()

# shellcheck source=run/common.sh
source "$SCRIPT_DIR/common.sh"
run_npu_opd_job
