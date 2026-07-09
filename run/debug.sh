#!/usr/bin/env bash
# GPU Ray debug launcher for OPD/TOPD/POPD/DMPD.
#
# This script is intended to be run locally after forwarding the Ray dashboard:
#   ssh -N -L 8191:gpu05:8191 hitsc
#
# Examples:
#   bash run/debug.sh
#   DEBUG_MODE=popd bash run/debug.sh
#   DEBUG_MODE=vanilla TOTAL_TRAINING_STEPS=1 bash run/debug.sh
#   DEBUG_MODE=popd TEACHER_GPUS=1 TEACHER_TP=1 bash run/debug.sh
#
# The job disables W&B and checkpoint saves by default.

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
RUN_PREFLIGHT=${RUN_PREFLIGHT:-1}
REMOTE_PYTHON=${REMOTE_PYTHON:-python3}

DEBUG_MODE=${DEBUG_MODE:-topd}
MODEL_ROOT=${MODEL_ROOT:-/share/home/jnguan/share/models}
DATASET_ROOT=${DATASET_ROOT:-/share/home/jnguan/share/datasets}
DEFAULT_STUDENT_MODEL="$MODEL_ROOT/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DEFAULT_TEACHER_MODEL="$MODEL_ROOT/hbx/JustRL-DeepSeek-1.5B"
if [[ ! -d "$DEFAULT_STUDENT_MODEL" && -d "$MODEL_ROOT/DeepSeek-R1-Distill-Qwen-1.5B" ]]; then
    DEFAULT_STUDENT_MODEL="$MODEL_ROOT/DeepSeek-R1-Distill-Qwen-1.5B"
fi
if [[ ! -d "$DEFAULT_TEACHER_MODEL" && -d "$MODEL_ROOT/JustRL-DeepSeek-1.5B" ]]; then
    DEFAULT_TEACHER_MODEL="$MODEL_ROOT/JustRL-DeepSeek-1.5B"
fi
STUDENT_MODEL=${STUDENT_MODEL:-"$DEFAULT_STUDENT_MODEL"}
TEACHER_MODEL=${TEACHER_MODEL:-"$DEFAULT_TEACHER_MODEL"}
TRAIN_FILE=${TRAIN_FILE:-"$DATASET_ROOT/DAPO-Math-17k-Processed-verl--qwen3/train.parquet"}
VAL_FILE=${VAL_FILE:-"$DATASET_ROOT/DAPO-Math-17k-Processed-verl--qwen3/test.parquet"}
CHAT_TEMPLATE_MODEL=${CHAT_TEMPLATE_MODEL:-$TEACHER_MODEL}

STUDENT_GPUS=${STUDENT_GPUS:-4}
TEACHER_GPUS=${TEACHER_GPUS:-4}
STUDENT_NNODES=${STUDENT_NNODES:-1}
TEACHER_NNODES=${TEACHER_NNODES:-1}
STUDENT_TP=${STUDENT_TP:-$STUDENT_GPUS}
TEACHER_TP=${TEACHER_TP:-$TEACHER_GPUS}
TEACHER_REPLICAS=${TEACHER_REPLICAS:-1}
SP_SIZE=${SP_SIZE:-1}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-15360}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1))}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-1024}
TEACHER_MAX_NUM_BATCHED_TOKENS=${TEACHER_MAX_NUM_BATCHED_TOKENS:-1024}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-8}
TEACHER_MAX_NUM_SEQS=${TEACHER_MAX_NUM_SEQS:-8}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.45}
TEACHER_GPU_MEMORY_UTILIZATION=${TEACHER_GPU_MEMORY_UTILIZATION:-0.45}
ROLLOUT_N=${ROLLOUT_N:-4}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-1.0}
ROLLOUT_TOP_K=${ROLLOUT_TOP_K:--1}
ROLLOUT_DO_SAMPLE=${ROLLOUT_DO_SAMPLE:-True}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.7}
VAL_TOP_P=${VAL_TOP_P:-0.95}
VAL_TOP_K=${VAL_TOP_K:-$ROLLOUT_TOP_K}
VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-True}
VAL_N=${VAL_N:-2}
ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-False}
TEACHER_ENFORCE_EAGER=${TEACHER_ENFORCE_EAGER:-False}
SKIP_TOKENIZER_INIT=${SKIP_TOKENIZER_INIT:-True}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-flash_attention_2}
REMOTE_DISABLE_LD_PRELOAD=${REMOTE_DISABLE_LD_PRELOAD:-0}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-$TRAIN_BATCH_SIZE}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-8}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-16}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-256}
DATA_SEED=${DATA_SEED:--42}
DATA_TRUNCATION=${DATA_TRUNCATION:-error}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-2}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
TEST_FREQ=${TEST_FREQ:--1}
SAVE_FREQ=${SAVE_FREQ:--1}
ACTOR_LR=${ACTOR_LR:-1e-6}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-$(( (MAX_MODEL_LEN + 127) / 128 * 128 ))}

ADV_ESTIMATOR=${ADV_ESTIMATOR:-token_reward_direct}
if [[ "$ADV_ESTIMATOR" == "token_reward_direct" ]]; then
    DISTILLATION_USE_ACTOR_LOSS=${DISTILLATION_USE_ACTOR_LOSS:-False}
else
    DISTILLATION_USE_ACTOR_LOSS=${DISTILLATION_USE_ACTOR_LOSS:-True}
fi
DISTILLATION_LOSS_MODE=${DISTILLATION_LOSS_MODE:-k1}
DISTILLATION_TOPK=${DISTILLATION_TOPK:-16}
TOP_K_STRATEGY=${TOP_K_STRATEGY:-only_stu}
REWARD_WEIGHT_MODE=${REWARD_WEIGHT_MODE:-student_p}
DISTILLATION_USE_TASK_REWARDS=${DISTILLATION_USE_TASK_REWARDS:-False}
DISTILLATION_USE_POLICY_GRADIENT=${DISTILLATION_USE_POLICY_GRADIENT:-True}
DISTILLATION_LOSS_MAX_CLAMP=${DISTILLATION_LOSS_MAX_CLAMP:-10.0}
DISTILLATION_LOG_PROB_MIN_CLAMP=${DISTILLATION_LOG_PROB_MIN_CLAMP:--10.0}
DISTILLATION_HORIZON_TOKENS=${DISTILLATION_HORIZON_TOKENS:-64}
DISTILLATION_HORIZON_START_TOKENS=${DISTILLATION_HORIZON_START_TOKENS:-32}
DISTILLATION_HORIZON_END_TOKENS=${DISTILLATION_HORIZON_END_TOKENS:-64}
DISTILLATION_HORIZON_INCREMENT_TOKENS=${DISTILLATION_HORIZON_INCREMENT_TOKENS:-32}
DISTILLATION_HORIZON_INTERVAL_STEPS=${DISTILLATION_HORIZON_INTERVAL_STEPS:-1}

OPD_DIAGNOSTICS_ENABLED=${OPD_DIAGNOSTICS_ENABLED:-True}
OPD_DIAGNOSTICS_COLLECT_TOPK_OVERLAP=${OPD_DIAGNOSTICS_COLLECT_TOPK_OVERLAP:-True}
OPD_DIAGNOSTICS_COLLECT_RETHINKING_OPD=${OPD_DIAGNOSTICS_COLLECT_RETHINKING_OPD:-True}
OPD_DIAGNOSTICS_TOPK=${OPD_DIAGNOSTICS_TOPK:-$DISTILLATION_TOPK}
OPD_DIAGNOSTICS_POSITION_BIN_SIZE=${OPD_DIAGNOSTICS_POSITION_BIN_SIZE:-32}
OPD_DIAGNOSTICS_LOG_INTERVAL=${OPD_DIAGNOSTICS_LOG_INTERVAL:-1}
OPD_DIAGNOSTICS_TOPK_LOG_INTERVAL=${OPD_DIAGNOSTICS_TOPK_LOG_INTERVAL:-1}
OPD_DIAGNOSTICS_ASYNC_METRICS=${OPD_DIAGNOSTICS_ASYNC_METRICS:-False}
OPD_DIAGNOSTICS_ASYNC_MAX_PENDING=${OPD_DIAGNOSTICS_ASYNC_MAX_PENDING:-1}
OPD_DIAGNOSTICS_HEATMAP_INTERVAL=${OPD_DIAGNOSTICS_HEATMAP_INTERVAL:-1}
OPD_DIAGNOSTICS_HEATMAP_HISTORY_STEPS=${OPD_DIAGNOSTICS_HEATMAP_HISTORY_STEPS:-8}
OPD_DIAGNOSTICS_LOG_WANDB_HEATMAP=${OPD_DIAGNOSTICS_LOG_WANDB_HEATMAP:-False}
OPD_DIAGNOSTICS_HIGH_GAP_THRESHOLD=${OPD_DIAGNOSTICS_HIGH_GAP_THRESHOLD:-1.0}
OPD_DIAGNOSTICS_BAD_TAIL_MIN_POSITION=${OPD_DIAGNOSTICS_BAD_TAIL_MIN_POSITION:-12000}
OPD_DIAGNOSTICS_BAD_TAIL_DEEP_POSITION=${OPD_DIAGNOSTICS_BAD_TAIL_DEEP_POSITION:-16000}
OPD_DIAGNOSTICS_BAD_TAIL_EXTREME_POSITION=${OPD_DIAGNOSTICS_BAD_TAIL_EXTREME_POSITION:-20000}
OPD_DIAGNOSTICS_BAD_TAIL_OVERLAP_THRESHOLD=${OPD_DIAGNOSTICS_BAD_TAIL_OVERLAP_THRESHOLD:-0.5}
OPD_DIAGNOSTICS_BAD_TAIL_TEACHER_PROB_THRESHOLD=${OPD_DIAGNOSTICS_BAD_TAIL_TEACHER_PROB_THRESHOLD:-0.2}
OPD_DIAGNOSTICS_BAD_TAIL_GAP_THRESHOLD=${OPD_DIAGNOSTICS_BAD_TAIL_GAP_THRESHOLD:-1.0}
OPD_DIAGNOSTICS_BAD_TAIL_REWARD_THRESHOLD=${OPD_DIAGNOSTICS_BAD_TAIL_REWARD_THRESHOLD:--2.0}
OPD_DIAGNOSTICS_BAD_TAIL_ENTROPY_THRESHOLD=${OPD_DIAGNOSTICS_BAD_TAIL_ENTROPY_THRESHOLD:-4.0}

TEACHER_INTERVENTION_SEGMENT_LENGTH=${TEACHER_INTERVENTION_SEGMENT_LENGTH:-32}
TEACHER_INTERVENTION_TEACHER_LENGTH=${TEACHER_INTERVENTION_TEACHER_LENGTH:-32}
TEACHER_INTERVENTION_GAP_METRIC=${TEACHER_INTERVENTION_GAP_METRIC:-k3}
TEACHER_INTERVENTION_GAP_DISTRIBUTION_TOPK=${TEACHER_INTERVENTION_GAP_DISTRIBUTION_TOPK:-16}
TEACHER_INTERVENTION_GAP_THRESHOLD=${TEACHER_INTERVENTION_GAP_THRESHOLD:-0.3}
TEACHER_INTERVENTION_MAX_COUNT=${TEACHER_INTERVENTION_MAX_COUNT:-2}
TEACHER_INTERVENTION_TRAIN_TEACHER=${TEACHER_INTERVENTION_TRAIN_TEACHER:-False}
TEACHER_INTERVENTION_MIN_TRAINABLE_TOKENS=${TEACHER_INTERVENTION_MIN_TRAINABLE_TOKENS:-1}
TEACHER_INTERVENTION_TEMPERATURE=${TEACHER_INTERVENTION_TEMPERATURE:-0.6}
TEACHER_INTERVENTION_TOP_P=${TEACHER_INTERVENTION_TOP_P:-0.95}
TEACHER_INTERVENTION_TOP_K=${TEACHER_INTERVENTION_TOP_K:-20}

VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
TEACHER_VAL_BEFORE_TRAIN=${TEACHER_VAL_BEFORE_TRAIN:-False}
TRAINER_LOGGER=${TRAINER_LOGGER:-'["console"]'}
PROJECT_NAME=${PROJECT_NAME:-DMPD-debug}
WANDB_MODE=${WANDB_MODE:-disabled}
WANDB_DISABLED=${WANDB_DISABLED:-true}

EXPERIMENT_TIMESTAMP=${EXPERIMENT_TIMESTAMP:-$(date +"%Y%m%d-%H%M%S")}
EXPERIMENT_BASE_NAME=${EXPERIMENT_NAME:-"debug-${DEBUG_MODE}"}
EXPERIMENT_FULL_NAME=${EXPERIMENT_FULL_NAME:-"${EXPERIMENT_BASE_NAME}-${EXPERIMENT_TIMESTAMP}"}
DEFAULT_LOCAL_DIR=${DEFAULT_LOCAL_DIR:-"/tmp/verl_debug/${EXPERIMENT_FULL_NAME}"}

case "$DEBUG_MODE" in
    vanilla | opd)
        MODE_NAME=vanilla-opd
        MODE_OVERRIDES=(
            "distillation.distillation_loss.horizon.mode=full"
            "distillation.intervention.enabled=False"
        )
        ;;
    topd)
        MODE_NAME=topd
        MODE_OVERRIDES=(
            "distillation.distillation_loss.horizon.mode=fixed"
            "distillation.distillation_loss.horizon.tokens=${DISTILLATION_HORIZON_TOKENS}"
            "distillation.intervention.enabled=False"
        )
        ;;
    popd)
        MODE_NAME=popd
        MODE_OVERRIDES=(
            "distillation.distillation_loss.horizon.mode=progressive"
            "distillation.distillation_loss.horizon.start_tokens=${DISTILLATION_HORIZON_START_TOKENS}"
            "distillation.distillation_loss.horizon.end_tokens=${DISTILLATION_HORIZON_END_TOKENS}"
            "distillation.distillation_loss.horizon.increment_tokens=${DISTILLATION_HORIZON_INCREMENT_TOKENS}"
            "distillation.distillation_loss.horizon.interval_steps=${DISTILLATION_HORIZON_INTERVAL_STEPS}"
            "distillation.intervention.enabled=False"
        )
        ;;
    dmpd)
        MODE_NAME=dmpd
        MODE_OVERRIDES=(
            "distillation.distillation_loss.horizon.mode=full"
            "distillation.intervention.enabled=True"
            "distillation.intervention.student_segment_length=${TEACHER_INTERVENTION_SEGMENT_LENGTH}"
            "distillation.intervention.teacher_segment_length=${TEACHER_INTERVENTION_TEACHER_LENGTH}"
            "distillation.intervention.gap_metric=${TEACHER_INTERVENTION_GAP_METRIC}"
            "distillation.intervention.gap_distribution_topk=${TEACHER_INTERVENTION_GAP_DISTRIBUTION_TOPK}"
            "distillation.intervention.gap_threshold=${TEACHER_INTERVENTION_GAP_THRESHOLD}"
            "distillation.intervention.max_interventions_per_sample=${TEACHER_INTERVENTION_MAX_COUNT}"
            "distillation.intervention.train_on_teacher_segments=${TEACHER_INTERVENTION_TRAIN_TEACHER}"
            "distillation.intervention.min_trainable_tokens_after_repair=${TEACHER_INTERVENTION_MIN_TRAINABLE_TOKENS}"
            "distillation.intervention.teacher_temperature=${TEACHER_INTERVENTION_TEMPERATURE}"
            "distillation.intervention.teacher_top_p=${TEACHER_INTERVENTION_TOP_P}"
            "distillation.intervention.teacher_top_k=${TEACHER_INTERVENTION_TOP_K}"
        )
        ;;
    *)
        echo "Unsupported DEBUG_MODE=${DEBUG_MODE}. Use vanilla, topd, popd, or dmpd." >&2
        exit 2
        ;;
esac

COMMON_OVERRIDES=(
    "algorithm.adv_estimator=${ADV_ESTIMATOR}"
    "algorithm.use_kl_in_reward=False"
    "algorithm.rollout_correction.bypass_mode=False"
    "data.train_files=[\"${TRAIN_FILE}\"]"
    "data.val_files=[\"${VAL_FILE}\"]"
    "data.train_batch_size=${TRAIN_BATCH_SIZE}"
    "data.val_batch_size=${VAL_BATCH_SIZE}"
    "data.train_max_samples=${TRAIN_MAX_SAMPLES}"
    "data.val_max_samples=${VAL_MAX_SAMPLES}"
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
    "data.max_response_length=${MAX_RESPONSE_LENGTH}"
    "data.filter_overlong_prompts=True"
    "data.truncation=${DATA_TRUNCATION}"
    "data.shuffle=False"
    "data.seed=${DATA_SEED}"
    "data.dataloader_num_workers=0"
    "actor_rollout_ref.model.path=${STUDENT_MODEL}"
    "actor_rollout_ref.model.chat_template_from_model=${CHAT_TEMPLATE_MODEL}"
    "actor_rollout_ref.model.use_remove_padding=True"
    "actor_rollout_ref.model.enable_gradient_checkpointing=True"
    "+actor_rollout_ref.model.override_config.attn_implementation=${ATTN_IMPLEMENTATION}"
    "actor_rollout_ref.actor.use_torch_compile=False"
    "actor_rollout_ref.actor.optim.lr=${ACTOR_LR}"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
    "actor_rollout_ref.actor.use_dynamic_bsz=True"
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}"
    "actor_rollout_ref.actor.ulysses_sequence_parallel_size=${SP_SIZE}"
    "actor_rollout_ref.actor.fsdp_config.param_offload=True"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True"
    "actor_rollout_ref.actor.fsdp_config.forward_prefetch=True"
    "actor_rollout_ref.ref.fsdp_config.forward_prefetch=True"
    "actor_rollout_ref.rollout.name=vllm"
    "actor_rollout_ref.rollout.mode=async"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${STUDENT_TP}"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}"
    "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
    "actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE}"
    "actor_rollout_ref.rollout.top_p=${ROLLOUT_TOP_P}"
    "actor_rollout_ref.rollout.top_k=${ROLLOUT_TOP_K}"
    "actor_rollout_ref.rollout.do_sample=${ROLLOUT_DO_SAMPLE}"
    "actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}"
    "actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS}"
    "actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}"
    "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True"
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}"
    "actor_rollout_ref.rollout.enforce_eager=${ROLLOUT_ENFORCE_EAGER}"
    "actor_rollout_ref.rollout.skip_tokenizer_init=${SKIP_TOKENIZER_INIT}"
    "actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE}"
    "actor_rollout_ref.rollout.val_kwargs.do_sample=${VAL_DO_SAMPLE}"
    "actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P}"
    "actor_rollout_ref.rollout.val_kwargs.top_k=${VAL_TOP_K}"
    "actor_rollout_ref.rollout.val_kwargs.n=${VAL_N}"
    "reward.reward_manager.name=naive"
    "critic.enable=False"
    "trainer.balance_batch=True"
    "trainer.logger=${TRAINER_LOGGER}"
    "trainer.project_name=${PROJECT_NAME}"
    "trainer.experiment_name=${EXPERIMENT_FULL_NAME}"
    "trainer.n_gpus_per_node=${STUDENT_GPUS}"
    "trainer.nnodes=${STUDENT_NNODES}"
    "trainer.device=cuda"
    "trainer.default_local_dir=${DEFAULT_LOCAL_DIR}"
    "trainer.val_before_train=${VAL_BEFORE_TRAIN}"
    "trainer.teacher_val_before_train=${TEACHER_VAL_BEFORE_TRAIN}"
    "trainer.teacher_val_max_samples=${VAL_MAX_SAMPLES}"
    "trainer.save_freq=${SAVE_FREQ}"
    "trainer.test_freq=${TEST_FREQ}"
    "trainer.total_epochs=${TOTAL_EPOCHS}"
    "trainer.total_training_steps=${TOTAL_TRAINING_STEPS}"
    "distillation.enabled=True"
    "distillation.use_actor_loss=${DISTILLATION_USE_ACTOR_LOSS}"
    "distillation.n_gpus_per_node=${TEACHER_GPUS}"
    "distillation.nnodes=${TEACHER_NNODES}"
    "distillation.teacher_models.teacher_model.model_path=${TEACHER_MODEL}"
    "distillation.teacher_models.teacher_model.num_replicas=${TEACHER_REPLICAS}"
    "distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${TEACHER_TP}"
    "distillation.teacher_models.teacher_model.inference.name=vllm"
    "distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${TEACHER_GPU_MEMORY_UTILIZATION}"
    "distillation.teacher_models.teacher_model.inference.max_model_len=${MAX_MODEL_LEN}"
    "distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=${TEACHER_MAX_NUM_BATCHED_TOKENS}"
    "distillation.teacher_models.teacher_model.inference.max_num_seqs=${TEACHER_MAX_NUM_SEQS}"
    "distillation.teacher_models.teacher_model.inference.enforce_eager=${TEACHER_ENFORCE_EAGER}"
    "distillation.teacher_models.teacher_model.inference.skip_tokenizer_init=${SKIP_TOKENIZER_INIT}"
    "distillation.distillation_loss.loss_mode=${DISTILLATION_LOSS_MODE}"
    "distillation.distillation_loss.topk=${DISTILLATION_TOPK}"
    "distillation.distillation_loss.top_k_strategy=${TOP_K_STRATEGY}"
    "distillation.distillation_loss.reward_weight_mode=${REWARD_WEIGHT_MODE}"
    "distillation.distillation_loss.use_task_rewards=${DISTILLATION_USE_TASK_REWARDS}"
    "distillation.distillation_loss.use_policy_gradient=${DISTILLATION_USE_POLICY_GRADIENT}"
    "distillation.distillation_loss.loss_max_clamp=${DISTILLATION_LOSS_MAX_CLAMP}"
    "distillation.distillation_loss.log_prob_min_clamp=${DISTILLATION_LOG_PROB_MIN_CLAMP}"
    "distillation.diagnostics.enabled=${OPD_DIAGNOSTICS_ENABLED}"
    "distillation.diagnostics.collect_topk_overlap=${OPD_DIAGNOSTICS_COLLECT_TOPK_OVERLAP}"
    "distillation.diagnostics.collect_rethinking_opd_metrics=${OPD_DIAGNOSTICS_COLLECT_RETHINKING_OPD}"
    "distillation.diagnostics.topk=${OPD_DIAGNOSTICS_TOPK}"
    "distillation.diagnostics.position_bin_size=${OPD_DIAGNOSTICS_POSITION_BIN_SIZE}"
    "distillation.diagnostics.log_interval=${OPD_DIAGNOSTICS_LOG_INTERVAL}"
    "distillation.diagnostics.topk_log_interval=${OPD_DIAGNOSTICS_TOPK_LOG_INTERVAL}"
    "distillation.diagnostics.async_metrics=${OPD_DIAGNOSTICS_ASYNC_METRICS}"
    "distillation.diagnostics.async_max_pending=${OPD_DIAGNOSTICS_ASYNC_MAX_PENDING}"
    "distillation.diagnostics.heatmap_interval=${OPD_DIAGNOSTICS_HEATMAP_INTERVAL}"
    "distillation.diagnostics.heatmap_history_steps=${OPD_DIAGNOSTICS_HEATMAP_HISTORY_STEPS}"
    "distillation.diagnostics.log_wandb_heatmap=${OPD_DIAGNOSTICS_LOG_WANDB_HEATMAP}"
    "distillation.diagnostics.high_gap_threshold=${OPD_DIAGNOSTICS_HIGH_GAP_THRESHOLD}"
    "distillation.diagnostics.bad_tail_min_position=${OPD_DIAGNOSTICS_BAD_TAIL_MIN_POSITION}"
    "distillation.diagnostics.bad_tail_deep_position=${OPD_DIAGNOSTICS_BAD_TAIL_DEEP_POSITION}"
    "distillation.diagnostics.bad_tail_extreme_position=${OPD_DIAGNOSTICS_BAD_TAIL_EXTREME_POSITION}"
    "distillation.diagnostics.bad_tail_overlap_threshold=${OPD_DIAGNOSTICS_BAD_TAIL_OVERLAP_THRESHOLD}"
    "distillation.diagnostics.bad_tail_teacher_prob_threshold=${OPD_DIAGNOSTICS_BAD_TAIL_TEACHER_PROB_THRESHOLD}"
    "distillation.diagnostics.bad_tail_gap_threshold=${OPD_DIAGNOSTICS_BAD_TAIL_GAP_THRESHOLD}"
    "distillation.diagnostics.bad_tail_reward_threshold=${OPD_DIAGNOSTICS_BAD_TAIL_REWARD_THRESHOLD}"
    "distillation.diagnostics.bad_tail_entropy_threshold=${OPD_DIAGNOSTICS_BAD_TAIL_ENTROPY_THRESHOLD}"
)

OVERRIDES=("${COMMON_OVERRIDES[@]}" "${MODE_OVERRIDES[@]}")

PREFLIGHT_CODE='import numpy as np, numba, torch, flash_attn, vllm
from numba import njit

@njit
def f(x):
    return x + 1

print({
    "numpy": np.__version__,
    "numba": numba.__version__,
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "flash_attn": flash_attn.__version__,
    "vllm": vllm.__version__,
    "numba_njit_result": int(f(41)),
}, flush=True)'

if [[ "$REMOTE_DISABLE_LD_PRELOAD" == "1" ]]; then
    REMOTE_LD_PRELOAD_LINE='unset LD_PRELOAD'
else
    REMOTE_LD_PRELOAD_LINE='export LD_PRELOAD="/lib64/libgomp.so.1:/lib64/libstdc++.so.6:/lib64/libgcc_s.so.1:${LD_PRELOAD:-}"'
fi

REMOTE_CMD='set -euo pipefail
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export WANDB_MODE='"$WANDB_MODE"'
export WANDB_DISABLED='"$WANDB_DISABLED"'
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export RAY_USAGE_STATS_ENABLED=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_DEBUG=WARN
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND='"${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"'
'"${REMOTE_LD_PRELOAD_LINE}"'
echo "[debug] remote pwd=$PWD"
test -f '"$(printf "%q" "$TRAIN_FILE")"' || { echo "Missing TRAIN_FILE='"$TRAIN_FILE"'" >&2; exit 2; }
test -f '"$(printf "%q" "$VAL_FILE")"' || { echo "Missing VAL_FILE='"$VAL_FILE"'" >&2; exit 2; }
test -d '"$(printf "%q" "$STUDENT_MODEL")"' || { echo "Missing STUDENT_MODEL='"$STUDENT_MODEL"'" >&2; exit 2; }
test -d '"$(printf "%q" "$TEACHER_MODEL")"' || { echo "Missing TEACHER_MODEL='"$TEACHER_MODEL"'" >&2; exit 2; }
test -d '"$(printf "%q" "$CHAT_TEMPLATE_MODEL")"' || { echo "Missing CHAT_TEMPLATE_MODEL='"$CHAT_TEMPLATE_MODEL"'" >&2; exit 2; }
'

if [[ "$RUN_PREFLIGHT" == "1" ]]; then
    REMOTE_CMD+="${REMOTE_PYTHON} -c $(printf "%q" "$PREFLIGHT_CODE")
"
fi

REMOTE_CMD+="TRAIN_CMD=($(printf "%q" "$REMOTE_PYTHON") -m verl.trainer.main_ppo)
"
for override in "${OVERRIDES[@]}"; do
    REMOTE_CMD+="TRAIN_CMD+=($(printf "%q" "$override"))
"
done
REMOTE_CMD+='echo "[debug] train command overrides=\$(( \${#TRAIN_CMD[@]} - 3 ))"
"\${TRAIN_CMD[@]}"
'

cat <<EOF
[debug] submitting GPU debug job
  ray_api:      ${RAY_API_SERVER_ADDRESS}
  mode:         ${MODE_NAME}
  student:      ${STUDENT_MODEL}
  teacher:      ${TEACHER_MODEL}
  data train:   ${TRAIN_FILE}
  data val:     ${VAL_FILE}
  gpu split:    student ${STUDENT_GPUS}g tp=${STUDENT_TP}, teacher ${TEACHER_GPUS}g tp=${TEACHER_TP}
  steps:        ${TOTAL_TRAINING_STEPS}
  max lengths:  prompt ${MAX_PROMPT_LENGTH}, response ${MAX_RESPONSE_LENGTH}, model ${MAX_MODEL_LEN}
  chat template:${CHAT_TEMPLATE_MODEL:-disabled}
  attention:    ${ATTN_IMPLEMENTATION}, vllm backend ${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
  diagnostics:  log every ${OPD_DIAGNOSTICS_LOG_INTERVAL}, topk every ${OPD_DIAGNOSTICS_TOPK_LOG_INTERVAL}, async=${OPD_DIAGNOSTICS_ASYNC_METRICS}
  wandb/ckpt:   WANDB_MODE=${WANDB_MODE}, save_freq=${SAVE_FREQ}
EOF

if [[ "$DRY_RUN" == "1" ]]; then
    printf "\n[debug] remote command:\n%s\n" "$REMOTE_CMD"
    exit 0
fi

RAY_JOB_ARGS=(job submit --working-dir .)
if [[ "$RAY_NO_WAIT" == "1" ]]; then
    RAY_JOB_ARGS+=(--no-wait)
fi

cd "$REPO_ROOT"
RAY_API_SERVER_ADDRESS="$RAY_API_SERVER_ADDRESS" "$RAY_BIN" "${RAY_JOB_ARGS[@]}" -- bash -lc "$REMOTE_CMD"
