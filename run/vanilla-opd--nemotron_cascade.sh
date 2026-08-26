#!/usr/bin/env bash

set -x

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

DEFAULT_EXPERIMENT_NAME=${EXPERIMENT_NAME:-vanilla-opd--nemotron_cascade}

TRAIN_DATASET=${TRAIN_DATASET:-nemotron_cascade_2_rl_mopd}

METHOD_HYDRA_OVERRIDES=()

# shellcheck source=run/common.sh
source "$SCRIPT_DIR/common.sh"
run_npu_opd_job
