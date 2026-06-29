#!/usr/bin/env bash

set -x

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

DEFAULT_EXPERIMENT_NAME=${EXPERIMENT_NAME:-topd--no_use_task_rewards}
USE_TASK_REWARDS=False

distillation_horizon_tokens=${DISTILLATION_HORIZON_TOKENS:-4096}

METHOD_HYDRA_OVERRIDES=(
    "distillation.distillation_loss.horizon.mode=fixed"
    "distillation.distillation_loss.horizon.tokens=${distillation_horizon_tokens}"
)

# shellcheck source=run/common.sh
source "$SCRIPT_DIR/common.sh"
run_npu_opd_job
