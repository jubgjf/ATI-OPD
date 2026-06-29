#!/usr/bin/env bash

set -x

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

DEFAULT_EXPERIMENT_NAME=${EXPERIMENT_NAME:-popd}

distillation_horizon_start_tokens=${DISTILLATION_HORIZON_START_TOKENS:-2048}
distillation_horizon_end_tokens=${DISTILLATION_HORIZON_END_TOKENS:-4096}
distillation_horizon_increment_tokens=${DISTILLATION_HORIZON_INCREMENT_TOKENS:-512}
distillation_horizon_interval_steps=${DISTILLATION_HORIZON_INTERVAL_STEPS:-50}

METHOD_HYDRA_OVERRIDES=(
    "distillation.distillation_loss.horizon.mode=progressive"
    "distillation.distillation_loss.horizon.start_tokens=${distillation_horizon_start_tokens}"
    "distillation.distillation_loss.horizon.end_tokens=${distillation_horizon_end_tokens}"
    "distillation.distillation_loss.horizon.increment_tokens=${distillation_horizon_increment_tokens}"
    "distillation.distillation_loss.horizon.interval_steps=${distillation_horizon_interval_steps}"
)

# shellcheck source=run/common.sh
source "$SCRIPT_DIR/common.sh"
run_npu_opd_job
