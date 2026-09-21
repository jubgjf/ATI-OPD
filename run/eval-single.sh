#!/usr/bin/env bash
# Standalone math evaluation in the existing Ascend/verl environment; results use stdout only.
#
# Required: MODEL_PATH=/path/to/global_step_N, its actor or actor/huggingface,
# or a complete HF model. Relative paths are resolved from the launch directory.
# Megatron exports at global_step_N/model/huggingface also accept the step or model directory.
# These exports require complete HF weights readable by the evaluation user; no Megatron merge is run.
# FSDP weights are merged into actor/huggingface and reused on later runs.
# Set EVAL_FORCE_MERGE=True to regenerate existing HF weights from FSDP shards.
# EVAL_ALL=0 (default): HMMT Feb 2026 and Nov 2025. EVAL_ALL=1 additionally evaluates
# AIME24, AIME25, AMC23 and the 39-question AMO-Bench-P subset, in that order.
# Optional: TEST_BENCHMARK_ROOT (default: $DATASET_DIR/datasets), CHAT_TEMPLATE_MODEL,
# Provider directories, flat layouts and the older datasets/MathArena root are supported.
# ENABLE_THINKING=True, TEST_TEMPERATURE=0.7, TEST_TOP_P=0.95, TEST_TOP_K=-1,
# TEST_BENCHMARK_N=16, TEST_BENCHMARKS_MAX_RESPONSE_LENGTH=23552, STUDENT_TP=2.
# For a smoke test, set TEST_BENCHMARK_MAX_SAMPLES=1 and TEST_BENCHMARK_N=2.
# Results are printed only; checkpoint conversion writes actor/huggingface,
# and vLLM/Ascend may still create their own runtime caches.

set -x

_load_external_train_config() {
    local train_config_path=${TRAIN_CONFIG_PATH:-/opt/huawei/algorithm/train.config}
    if [[ ! -s "$train_config_path" ]]; then
        echo "No non-empty train config found at $train_config_path; skip external env loading."
        return 0
    fi

    echo "Loading external train config from $train_config_path"
    local allexport_was_set=0
    case "$-" in
        *a*) allexport_was_set=1 ;;
    esac

    set -o allexport
    # shellcheck disable=SC1090
    if ! . "$train_config_path"; then
        if [[ "$allexport_was_set" -eq 0 ]]; then
            set +o allexport
        fi
        echo "Failed to load external train config: $train_config_path" >&2
        return 1
    fi
    if [[ "$allexport_was_set" -eq 0 ]]; then
        set +o allexport
    fi
}

DATASET_DIR=/opt/huawei/dataset/jnguan
WORKSPACE=$DATASET_DIR/code/DMPD

_load_external_train_config || exit 1

export NODE_RANK=${NODE_RANK:-${VC_TASK_INDEX:-0}}
if [[ ! "$NODE_RANK" =~ ^[0-9]+$ ]]; then
    echo "NODE_RANK must be a non-negative integer, got: $NODE_RANK" >&2
    exit 2
fi
if (( 10#$NODE_RANK != 0 )); then
    echo "Skipping standalone evaluation on NODE_RANK=$NODE_RANK; only node 0 evaluates."
    exit 0
fi
if [[ -z "${MODEL_PATH:-}" || ! -d "$MODEL_PATH" ]]; then
    echo "MODEL_PATH must name an existing local HF or FSDP checkpoint directory." >&2
    exit 2
fi
MODEL_PATH=$(cd -- "$MODEL_PATH" && pwd -P) || exit 2

export MODEL_PATH WORKSPACE
export EVAL_ALL=${EVAL_ALL:-0}
export TEST_BENCHMARK_ROOT=${TEST_BENCHMARK_ROOT:-$DATASET_DIR/datasets}
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1

npu-smi info

. /usr/local/Ascend/ascend-toolkit/set_env.sh
. /usr/local/Ascend/nnal/atb/set_env.sh

ls -l /usr/lib64/libjemalloc.so.2
export LD_PRELOAD=/usr/lib64/libjemalloc.so.2:${LD_PRELOAD}

. "$HOME"/anaconda3/etc/profile.d/conda.sh
conda activate verl

cd "$WORKSPACE" || exit 1
pip install -e .

export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-3600}
export HCCL_LOG_LEVEL=${HCCL_LOG_LEVEL:-0}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-3600}
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-2000}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export VLLM_ASCEND_ENABLE_NZ=${VLLM_ASCEND_ENABLE_NZ:-0}
export HCCL_ASYNC_ERROR_HANDLING=${HCCL_ASYNC_ERROR_HANDLING:-0}
export CLOSE_MATMUL_K_SHIFT=${CLOSE_MATMUL_K_SHIFT:-1}
export ATB_MATMUL_SHUFFLE_K_ENABLE=${ATB_MATMUL_SHUFFLE_K_ENABLE:-0}
export HCCL_DETERMINISTIC=${HCCL_DETERMINISTIC:-"true"}
export VLLM_ENABLE_V1_MULTIPROCESSING=${VLLM_ENABLE_V1_MULTIPROCESSING:-0}
export VLLM_USE_V1=${VLLM_USE_V1:-1}
ulimit -n "${ULIMIT_NOFILE:-65536}"

python -m verl.utils.evaluation.standalone_math
