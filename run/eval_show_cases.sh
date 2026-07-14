#!/usr/bin/env bash
# Show model generations from a verl checkpoint on a small train/validation split.
#
# Edit the variables below on the server, then run:
#   bash run/eval_show_cases.sh

set -euo pipefail

export TOKENIZERS_PARALLELISM=false

CHECKPOINT_PATH=${CHECKPOINT_PATH:-outputs/vanilla-opd-20260707-151135/global_step_160}
DATA_FILE=${DATA_FILE:-/opt/huawei/dataset/jnguan/datasets/open-r1/DAPO-Math-17k-Processed-verl--qwen3/test.parquet}

NUM_CASES=${NUM_CASES:-10}
SAMPLE_OFFSET=${SAMPLE_OFFSET:-0}
SHUFFLE=${SHUFFLE:-0}
SEED=${SEED:-42}

# "auto" means the checkpoint model's maximum context length.
MAX_COMPLETION_TOKENS=${MAX_COMPLETION_TOKENS:-auto}
FIT_CONTEXT=${FIT_CONTEXT:-1}
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-0.95}

TP_SIZE=${TP_SIZE:-1}
DP_SIZE=${DP_SIZE:-8}
LOCAL_DP_SIZE=${LOCAL_DP_SIZE:-8}
API_SERVER_COUNT=${API_SERVER_COUNT:-8}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-128}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-32768}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}

OUTPUT_JSONL=${OUTPUT_JSONL:-}
PRINT_MAX_CHARS=${PRINT_MAX_CHARS:-1200}

CMD=(
    python3 -m verl.trainer.main_eval_show_cases
    --checkpoint-path "$CHECKPOINT_PATH"
    --data-file "$DATA_FILE"
    --num-cases "$NUM_CASES"
    --sample-offset "$SAMPLE_OFFSET"
    --seed "$SEED"
    --max-completion-tokens "$MAX_COMPLETION_TOKENS"
    --temperature "$TEMPERATURE"
    --top-p "$TOP_P"
    --tensor-parallel-size "$TP_SIZE"
    --data-parallel-size "$DP_SIZE"
    --data-parallel-size-local "$LOCAL_DP_SIZE"
    --api-server-count "$API_SERVER_COUNT"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --print-max-chars "$PRINT_MAX_CHARS"
)

if [[ "$SHUFFLE" == "1" ]]; then
    CMD+=(--shuffle)
fi
if [[ "$FIT_CONTEXT" == "0" ]]; then
    CMD+=(--no-fit-context)
fi
if [[ -n "$OUTPUT_JSONL" ]]; then
    CMD+=(--output-jsonl "$OUTPUT_JSONL")
fi

echo "[show-cases] checkpoint: ${CHECKPOINT_PATH}"
echo "[show-cases] data: ${DATA_FILE}"
echo "[show-cases] cases: n=${NUM_CASES} offset=${SAMPLE_OFFSET} shuffle=${SHUFFLE}"
echo "[show-cases] vllm: tp=${TP_SIZE} dp=${DP_SIZE} local_dp=${LOCAL_DP_SIZE}"
echo "[show-cases] max_completion_tokens=${MAX_COMPLETION_TOKENS} fit_context=${FIT_CONTEXT}"
echo "[show-cases] command: ${CMD[*]}"

"${CMD[@]}"
