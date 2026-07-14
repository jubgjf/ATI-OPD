#!/usr/bin/env bash
# Run math benchmark evaluation on the current server.
#
# Edit the variables below, activate the verl environment, then run:
#   bash run/eval_math_benchmarks.sh

export TOKENIZERS_PARALLELISM=false

# W&B run. Set SKIP_WANDB=1 only for local/debug runs.
WANDB_ENTITY=jubgjf
WANDB_PROJECT=DMPD
WANDB_RUN_NAME=
WANDB_RUN_ID=
SKIP_WANDB=0

# If DEFAULT_LOCAL_DIR is empty, it is read from the W&B run config.
DEFAULT_LOCAL_DIR=
RESULTS_DIR=

# BASE_MODEL_PATH is evaluated as global_step_0 when CHECKPOINT_STEPS contains 0.
BASE_MODEL_PATH=/share/home/jnguan/share/models/Qwen3-1.7B
CHECKPOINT_STEPS=0,20,40,60

DATASET_ROOT=/share/home/jnguan/share/datasets
BENCHMARKS=aime24,aime25,amc23

ROLLOUT_N=16
MAX_TOKENS=32768
STORE_OUTPUTS=1

TP_SIZE=2
DP_SIZE=2

CMD=(
    python3 -m verl.trainer.main_math_benchmark_eval
    --wandb-entity "$WANDB_ENTITY"
    --wandb-project "$WANDB_PROJECT"
    --dataset-root "$DATASET_ROOT"
    --benchmarks "$BENCHMARKS"
    --num-samples "$ROLLOUT_N"
    --max-tokens "$MAX_TOKENS"
    --tensor-parallel-size "$TP_SIZE"
    --data-parallel-size "$DP_SIZE"
)

if [[ -n "$WANDB_RUN_NAME" ]]; then
    CMD+=(--wandb-run-name "$WANDB_RUN_NAME")
fi
if [[ -n "$WANDB_RUN_ID" ]]; then
    CMD+=(--wandb-run-id "$WANDB_RUN_ID")
fi
if [[ -n "$DEFAULT_LOCAL_DIR" ]]; then
    CMD+=(--default-local-dir "$DEFAULT_LOCAL_DIR")
fi
if [[ -n "$RESULTS_DIR" ]]; then
    CMD+=(--results-dir "$RESULTS_DIR")
fi
if [[ -n "$BASE_MODEL_PATH" ]]; then
    CMD+=(--baseline-model-path "$BASE_MODEL_PATH")
fi
if [[ -n "$CHECKPOINT_STEPS" ]]; then
    CMD+=(--checkpoint-steps "$CHECKPOINT_STEPS")
fi
if [[ "$SKIP_WANDB" == "1" ]]; then
    CMD+=(--skip-wandb)
fi
if [[ "$STORE_OUTPUTS" == "0" ]]; then
    CMD+=(--no-store-outputs)
fi

echo "[eval] running on current server"
echo "[eval] wandb: ${WANDB_ENTITY}/${WANDB_PROJECT} run_name=${WANDB_RUN_NAME} run_id=${WANDB_RUN_ID} skip=${SKIP_WANDB}"
echo "[eval] checkpoints: root=${DEFAULT_LOCAL_DIR} steps=${CHECKPOINT_STEPS}"
echo "[eval] baseline: ${BASE_MODEL_PATH}"
echo "[eval] datasets: ${DATASET_ROOT} benchmarks=${BENCHMARKS}"
echo "[eval] vllm: tp=${TP_SIZE} dp=${DP_SIZE}"
echo "[eval] command: ${CMD[*]}"

"${CMD[@]}"
