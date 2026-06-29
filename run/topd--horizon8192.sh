#!/usr/bin/env bash

set -x

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

DEFAULT_EXPERIMENT_NAME=${EXPERIMENT_NAME:-topd}

distillation_horizon_tokens=${DISTILLATION_HORIZON_TOKENS:-8192}

METHOD_HYDRA_OVERRIDES=(
    "distillation.distillation_loss.horizon.mode=fixed"
    "distillation.distillation_loss.horizon.tokens=${distillation_horizon_tokens}"
)

# shellcheck source=run/common.sh
source "$SCRIPT_DIR/common.sh"
run_npu_opd_job
