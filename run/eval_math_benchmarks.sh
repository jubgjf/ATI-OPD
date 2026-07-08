#!/usr/bin/env bash
# Evaluate math benchmarks for verl checkpoints.
#
# Examples:
#   WANDB_PROJECT=DMPD-testing WANDB_RUN_NAME=dmpd-test-xxx bash run/eval_math_benchmarks.sh
#   SKIP_WANDB=1 DEFAULT_LOCAL_DIR_OVERRIDE=/share/home/jnguan/code/verl-for-remote-debug/DMPD-outputs/run \
#     MAX_SAMPLES=1 ROLLOUT_N=1 MAX_TOKENS=512 bash run/eval_math_benchmarks.sh

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

if [[ -x "$REPO_ROOT/.venv/bin/ray" ]]; then
    RAY_BIN=${RAY_BIN:-"$REPO_ROOT/.venv/bin/ray"}
else
    RAY_BIN=${RAY_BIN:-ray}
fi

RAY_API_SERVER_ADDRESS=${RAY_API_SERVER_ADDRESS:-http://127.0.0.1:8191}
RAY_NO_WAIT=${RAY_NO_WAIT:-0}
DRY_RUN=${DRY_RUN:-0}
EVAL_LOCAL=${EVAL_LOCAL:-0}
REMOTE_PYTHON=${REMOTE_PYTHON:-python3}
INSTALL_MATH_VERIFY=${INSTALL_MATH_VERIFY:-1}
CUDA_VISIBLE_DEVICES_OVERRIDE=${CUDA_VISIBLE_DEVICES_OVERRIDE:-}

if [[ "$EVAL_LOCAL" == "1" ]]; then
    DATASET_ROOT=${DATASET_ROOT:-"$REPO_ROOT/dataset"}
else
    DATASET_ROOT=${DATASET_ROOT:-/share/home/jnguan/share/datasets}
fi

WANDB_ENTITY=${WANDB_ENTITY:-jubgjf}
WANDB_PROJECT=${WANDB_PROJECT:-DMPD}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-}
WANDB_RUN_ID=${WANDB_RUN_ID:-}
WANDB_INIT_TIMEOUT=${WANDB_INIT_TIMEOUT:-60}
WANDB_PREFLIGHT_TIMEOUT=${WANDB_PREFLIGHT_TIMEOUT:-10}
SKIP_WANDB=${SKIP_WANDB:-0}
MODEL_PATH=${MODEL_PATH:-}
MODEL_STEP=${MODEL_STEP:-0}
BASELINE_MODEL_PATH=${BASELINE_MODEL_PATH:-${BASE_MODEL_PATH:-}}
INCLUDE_BASELINE_MODEL=${INCLUDE_BASELINE_MODEL:-1}
DEFAULT_LOCAL_DIR_OVERRIDE=${DEFAULT_LOCAL_DIR_OVERRIDE:-${DEFAULT_LOCAL_DIR:-}}
BENCHMARKS=${BENCHMARKS:-aime24,aime25,amc23}
RESULTS_DIR=${RESULTS_DIR:-}
CHECKPOINT_STEPS=${CHECKPOINT_STEPS:-}
CHECKPOINT_PATH_REMAPS=${CHECKPOINT_PATH_REMAPS:-}

MAX_SAMPLES=${MAX_SAMPLES:--1}
ROLLOUT_N=${ROLLOUT_N:-16}
MAX_TOKENS=${MAX_TOKENS:-32768}
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-0.95}
TOP_K=${TOP_K:-}
GENERATION_ENDPOINT=${GENERATION_ENDPOINT:-chat}
ENABLE_THINKING=${ENABLE_THINKING:-1}
VERIFIER=${VERIFIER:-hybrid}
MATH_VERIFY_TIMEOUT=${MATH_VERIFY_TIMEOUT:-30}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-}
STORE_OUTPUTS=${STORE_OUTPUTS:-1}
RETRY_SKIPPED=${RETRY_SKIPPED:-0}
STOP_AFTER_SAMPLES=${STOP_AFTER_SAMPLES:-}
LENGTH_RETRIES=${LENGTH_RETRIES:-2}

TP_SIZE=${TP_SIZE:-1}
DP_SIZE=${DP_SIZE:-1}
LOCAL_DP_SIZE=${LOCAL_DP_SIZE:-}
DP_ADDRESS=${DP_ADDRESS:-}
DP_RPC_PORT=${DP_RPC_PORT:-}
DP_START_RANK=${DP_START_RANK:-}
API_SERVER_COUNT=${API_SERVER_COUNT:-}
MAX_CONCURRENCY_PER_DP=${MAX_CONCURRENCY_PER_DP:-128}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-128}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-32768}
VLLM_PORT=${VLLM_PORT:-0}
VLLM_HOST=${VLLM_HOST:-127.0.0.1}
VLLM_BIN=${VLLM_BIN:-vllm}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-eval-model}
SERVER_START_TIMEOUT=${SERVER_START_TIMEOUT:-900}
ENABLE_REASONING_PARSER=${ENABLE_REASONING_PARSER:-0}
REASONING_PARSER=${REASONING_PARSER:-deepseek_r1}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-0}
VLLM_EXTRA_ARGS=${VLLM_EXTRA_ARGS:-}

SKIP_MERGE=${SKIP_MERGE:-0}
FORCE_MERGE=${FORCE_MERGE:-0}
MERGE_TIMEOUT=${MERGE_TIMEOUT:-}
WATCH=${WATCH:-0}
POLL_INTERVAL=${POLL_INTERVAL:-300}
IDLE_TIMEOUT=${IDLE_TIMEOUT:-0}
LOG_LEVEL=${LOG_LEVEL:-INFO}

if [[ "$SKIP_WANDB" != "1" && -z "$WANDB_RUN_NAME" && -z "$WANDB_RUN_ID" ]]; then
    echo "WANDB_RUN_NAME or WANDB_RUN_ID is required unless SKIP_WANDB=1." >&2
    exit 2
fi

if [[ "$SKIP_WANDB" == "1" && -z "$DEFAULT_LOCAL_DIR_OVERRIDE" && -z "$MODEL_PATH" && -z "$BASELINE_MODEL_PATH" ]]; then
    echo "DEFAULT_LOCAL_DIR_OVERRIDE, MODEL_PATH, or BASELINE_MODEL_PATH is required when SKIP_WANDB=1." >&2
    exit 2
fi

ARGS=(
    "--wandb-entity" "$WANDB_ENTITY"
    "--wandb-project" "$WANDB_PROJECT"
    "--wandb-init-timeout" "$WANDB_INIT_TIMEOUT"
    "--wandb-preflight-timeout" "$WANDB_PREFLIGHT_TIMEOUT"
    "--dataset-root" "$DATASET_ROOT"
    "--benchmarks" "$BENCHMARKS"
    "--max-samples" "$MAX_SAMPLES"
    "--num-samples" "$ROLLOUT_N"
    "--max-tokens" "$MAX_TOKENS"
    "--temperature" "$TEMPERATURE"
    "--top-p" "$TOP_P"
    "--generation-endpoint" "$GENERATION_ENDPOINT"
    "--verifier" "$VERIFIER"
    "--math-verify-timeout" "$MATH_VERIFY_TIMEOUT"
    "--length-retries" "$LENGTH_RETRIES"
    "--tensor-parallel-size" "$TP_SIZE"
    "--data-parallel-size" "$DP_SIZE"
    "--max-concurrency-per-dp" "$MAX_CONCURRENCY_PER_DP"
    "--gpu-memory-utilization" "$GPU_MEMORY_UTILIZATION"
    "--host" "$VLLM_HOST"
    "--port" "$VLLM_PORT"
    "--vllm-bin" "$VLLM_BIN"
    "--served-model-name" "$SERVED_MODEL_NAME"
    "--server-start-timeout" "$SERVER_START_TIMEOUT"
    "--reasoning-parser" "$REASONING_PARSER"
    "--poll-interval" "$POLL_INTERVAL"
    "--idle-timeout" "$IDLE_TIMEOUT"
    "--log-level" "$LOG_LEVEL"
)

add_optional_arg() {
    local flag=$1
    local value=$2
    if [[ -n "$value" ]]; then
        ARGS+=("$flag" "$value")
    fi
}

add_bool_arg() {
    local flag=$1
    local value=$2
    if [[ "$value" == "1" || "$value" == "true" || "$value" == "True" ]]; then
        ARGS+=("$flag")
    else
        ARGS+=("--no-${flag#--}")
    fi
}

add_optional_arg "--wandb-run-name" "$WANDB_RUN_NAME"
add_optional_arg "--wandb-run-id" "$WANDB_RUN_ID"
add_optional_arg "--model-path" "$MODEL_PATH"
add_optional_arg "--model-step" "$MODEL_STEP"
add_optional_arg "--baseline-model-path" "$BASELINE_MODEL_PATH"
add_optional_arg "--default-local-dir" "$DEFAULT_LOCAL_DIR_OVERRIDE"
add_optional_arg "--results-dir" "$RESULTS_DIR"
add_optional_arg "--checkpoint-steps" "$CHECKPOINT_STEPS"
add_optional_arg "--top-k" "$TOP_K"
add_optional_arg "--request-timeout" "$REQUEST_TIMEOUT"
add_optional_arg "--stop-after-samples" "$STOP_AFTER_SAMPLES"
add_optional_arg "--data-parallel-size-local" "$LOCAL_DP_SIZE"
add_optional_arg "--data-parallel-address" "$DP_ADDRESS"
add_optional_arg "--data-parallel-rpc-port" "$DP_RPC_PORT"
add_optional_arg "--data-parallel-start-rank" "$DP_START_RANK"
add_optional_arg "--api-server-count" "$API_SERVER_COUNT"
add_optional_arg "--max-concurrency" "$MAX_CONCURRENCY"
add_optional_arg "--max-model-len" "$MAX_MODEL_LEN"
add_optional_arg "--max-num-seqs" "$MAX_NUM_SEQS"
add_optional_arg "--max-num-batched-tokens" "$MAX_NUM_BATCHED_TOKENS"
add_optional_arg "--merge-timeout" "$MERGE_TIMEOUT"

if [[ -n "$CHECKPOINT_PATH_REMAPS" ]]; then
    IFS=',' read -r -a REMAP_ARRAY <<< "$CHECKPOINT_PATH_REMAPS"
    for remap in "${REMAP_ARRAY[@]}"; do
        if [[ -n "$remap" ]]; then
            ARGS+=("--path-remap" "$remap")
        fi
    done
fi

if [[ -n "$VLLM_EXTRA_ARGS" ]]; then
    IFS=' ' read -r -a EXTRA_ARG_ARRAY <<< "$VLLM_EXTRA_ARGS"
    for extra_arg in "${EXTRA_ARG_ARRAY[@]}"; do
        if [[ -n "$extra_arg" ]]; then
            ARGS+=("--vllm-extra-arg" "$extra_arg")
        fi
    done
fi

add_bool_arg "--skip-wandb" "$SKIP_WANDB"
add_bool_arg "--include-baseline-model" "$INCLUDE_BASELINE_MODEL"
add_bool_arg "--store-outputs" "$STORE_OUTPUTS"
add_bool_arg "--retry-skipped" "$RETRY_SKIPPED"
add_bool_arg "--enable-thinking" "$ENABLE_THINKING"
add_bool_arg "--enable-reasoning-parser" "$ENABLE_REASONING_PARSER"
add_bool_arg "--trust-remote-code" "$TRUST_REMOTE_CODE"
add_bool_arg "--skip-merge" "$SKIP_MERGE"
add_bool_arg "--force-merge" "$FORCE_MERGE"
add_bool_arg "--watch" "$WATCH"
add_bool_arg "--dry-run" "$DRY_RUN"

printf -v ARGS_QUOTED '%q ' "${ARGS[@]}"

REMOTE_CMD='set -euo pipefail
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0
export RAY_USAGE_STATS_ENABLED=0
if [[ -n '"$(printf "%q" "$CUDA_VISIBLE_DEVICES_OVERRIDE")"' ]]; then
  export CUDA_VISIBLE_DEVICES='"$(printf "%q" "$CUDA_VISIBLE_DEVICES_OVERRIDE")"'
  echo "[eval] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi
echo "[eval] remote pwd=$PWD"
if [[ '"$(printf "%q" "$INSTALL_MATH_VERIFY")"' == "1" ]]; then
  if ! '"$(printf "%q" "$REMOTE_PYTHON")"' -c "import math_verify" >/dev/null 2>&1; then
    if command -v uv >/dev/null 2>&1; then
      uv pip install math-verify
    else
      '"$(printf "%q" "$REMOTE_PYTHON")"' -m pip install math-verify
    fi
  fi
fi
'"$(printf "%q" "$REMOTE_PYTHON")"' -m verl.trainer.main_math_benchmark_eval '"$ARGS_QUOTED"

cat <<EOF
[eval] math benchmark evaluation
  ray_api:      ${RAY_API_SERVER_ADDRESS}
  local:        ${EVAL_LOCAL}
  wandb:        entity=${WANDB_ENTITY} project=${WANDB_PROJECT} run_name=${WANDB_RUN_NAME:-<none>} run_id=${WANDB_RUN_ID:-<none>} skip=${SKIP_WANDB}
  model path:   ${MODEL_PATH:-<none>}
  baseline:     ${BASELINE_MODEL_PATH:-<auto-or-none>} include=${INCLUDE_BASELINE_MODEL}
  ckpt root:    ${DEFAULT_LOCAL_DIR_OVERRIDE:-<from wandb>}
  data root:    ${DATASET_ROOT}
  benchmarks:   ${BENCHMARKS}
  samples:      max_samples=${MAX_SAMPLES} rollout_n=${ROLLOUT_N} max_tokens=${MAX_TOKENS}
  retries:      length_retries=${LENGTH_RETRIES}
  endpoint:     ${GENERATION_ENDPOINT} enable_thinking=${ENABLE_THINKING}
  verifier:     ${VERIFIER}
  vllm:         tp=${TP_SIZE} dp=${DP_SIZE} local_dp=${LOCAL_DP_SIZE:-auto} max_concurrency=${MAX_CONCURRENCY:-auto} per_dp=${MAX_CONCURRENCY_PER_DP} mem=${GPU_MEMORY_UTILIZATION} max_model_len=${MAX_MODEL_LEN:-auto}
  cuda devices: ${CUDA_VISIBLE_DEVICES_OVERRIDE:-<ray default>}
EOF

if [[ "$DRY_RUN" == "1" ]]; then
    printf "\n[eval] command:\n%s\n" "$REMOTE_CMD"
fi

cd "$REPO_ROOT"
if [[ "$EVAL_LOCAL" == "1" ]]; then
    bash -lc "$REMOTE_CMD"
else
    RAY_JOB_ARGS=(job submit --working-dir .)
    if [[ "$RAY_NO_WAIT" == "1" ]]; then
        RAY_JOB_ARGS+=(--no-wait)
    fi
    RAY_API_SERVER_ADDRESS="$RAY_API_SERVER_ADDRESS" "$RAY_BIN" "${RAY_JOB_ARGS[@]}" -- bash -lc "$REMOTE_CMD"
fi
