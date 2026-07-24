#!/usr/bin/env bash

set -x

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

DEFAULT_EXPERIMENT_NAME=${EXPERIMENT_NAME:-vanilla-opd--only_stu_rethinking_style}

TOP_K_STRATEGY=${TOP_K_STRATEGY:-only_stu_rethinking_style}
DISTILLATION_LOG_PROB_MIN_CLAMP=${DISTILLATION_LOG_PROB_MIN_CLAMP:-null}

METHOD_HYDRA_OVERRIDES=()

# shellcheck source=run/common.sh
source "$SCRIPT_DIR/common.sh"
run_npu_opd_job
