#!/usr/bin/env bash

set -x

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

DEFAULT_EXPERIMENT_NAME=${EXPERIMENT_NAME:-vanilla-opd--only_stu_teacher_topk--teacher_topk_128}

TOP_K_STRATEGY=${TOP_K_STRATEGY:-only_stu_teacher_topk}
DISTILLATION_TOPK=${DISTILLATION_TOPK:-16}
TEACHER_TOPK=${TEACHER_TOPK:-128}

METHOD_HYDRA_OVERRIDES=()

# shellcheck source=run/common.sh
source "$SCRIPT_DIR/common.sh"
run_npu_opd_job
