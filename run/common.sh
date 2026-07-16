#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "run/common.sh is a shared launcher. Source it from a method script instead of executing it directly." >&2
    exit 1
fi

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

_check_resume_required() {
    if [[ "${resume_required:-False}" != "True" ]]; then
        return 0
    fi

    if [[ -z "${WANDB_RUN_ID:-}" ]]; then
        echo "RESUME_REQUIRED=True requires WANDB_RUN_ID to resume the same W&B run." >&2
        return 1
    fi

    local tracker_file="$default_local_dir/latest_checkpointed_iteration.txt"
    if [[ ! -f "$tracker_file" ]]; then
        echo "RESUME_REQUIRED=True but checkpoint tracker is missing: $tracker_file" >&2
        return 1
    fi
    local latest_step
    latest_step=$(cat "$tracker_file")
    local latest_ckpt="$default_local_dir/global_step_${latest_step}"
    if [[ ! -d "$latest_ckpt" ]]; then
        echo "RESUME_REQUIRED=True but latest checkpoint directory is missing: $latest_ckpt" >&2
        return 1
    fi
    echo "Resume preflight passed: $latest_ckpt"
}

run_npu_opd_job() {
    HOME=/home/ma-user
    DATASET_DIR=/opt/huawei/dataset/jnguan
    WORKSPACE=$DATASET_DIR/code/DMPD
    NETRC_PATH=$DATASET_DIR/apps/configs/dotnetrc

    _load_external_train_config || exit 1

    npu-smi info
    echo ============================

    . /usr/local/Ascend/ascend-toolkit/set_env.sh
    . /usr/local/Ascend/nnal/atb/set_env.sh
    sudo chown -R ma-user:ma-group /usr/local/Ascend/ascend-toolkit
    sudo chown -R ma-user:ma-group /usr/local/Ascend/nnal
    python -c "import acl;print(acl.get_soc_name())"
    echo ============================

    sudo yum install jemalloc jq tmux -y
    ls -l /usr/lib64/libjemalloc.so.2
    export LD_PRELOAD=/usr/lib64/libjemalloc.so.2:${LD_PRELOAD}
    echo ============================

    rm -rf "$HOME"/vllm*
    tar -xf "$DATASET_DIR"/libs/vllm-build.tar.gz -C "$HOME"
    . "$HOME"/anaconda3/etc/profile.d/conda.sh
    mkdir -p "$HOME"/anaconda3/envs/verl
    tar -xf "$DATASET_DIR"/libs/verl-conda-env-20260617.tar.gz -C "$HOME"/anaconda3/envs/verl
    conda env list
    conda activate verl
    echo ============================

    cd "$WORKSPACE"
    pip install -r requirements-npu.txt
    pip install -e .
    echo ============================

    ln -sfnT "$NETRC_PATH" "$HOME"/.netrc

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

    STUDENT_MODEL=${STUDENT_MODEL:-$DATASET_DIR/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}
    TEACHER_MODEL=${TEACHER_MODEL:-$DATASET_DIR/models/hbx/JustRL-DeepSeek-1.5B}
    CHAT_TEMPLATE_MODEL=${CHAT_TEMPLATE_MODEL:-$TEACHER_MODEL}

    dapo_math_train=${DAPO_MATH_TRAIN:-$DATASET_DIR/datasets/open-r1/DAPO-Math-17k-Processed-verl--qwen3/train.parquet}
    dapo_math_valid=${DAPO_MATH_VALID:-$DATASET_DIR/datasets/open-r1/DAPO-Math-17k-Processed-verl--qwen3/test.parquet}
    train_files=${TRAIN_FILES:-"['$dapo_math_train']"}
    valid_files=${VALID_FILES:-"['$dapo_math_valid']"}

    NNODES=${NNODES:-$VC_WORKER_NUM}
    STUDENT_NNODES=${STUDENT_NNODES:-$((VC_WORKER_NUM - 1))}
    TEACHER_NNODES=${TEACHER_NNODES:-1}
    STUDENT_NGPUS_PER_NODE=${STUDENT_NGPUS_PER_NODE:-8}
    TEACHER_NGPUS_PER_NODE=${TEACHER_NGPUS_PER_NODE:-8}
    student_tp=${STUDENT_TP:-4}
    student_gpu_mem_util=${STUDENT_GPU_MEM_UTIL:-0.6}
    teacher_tp=${TEACHER_TP:-4}
    teacher_gpu_mem_util=${TEACHER_GPU_MEM_UTIL:-0.6}
    sp_size=${SP_SIZE:-4}
    TEACHER_REPLICAS=${TEACHER_REPLICAS:-$((TEACHER_NNODES * TEACHER_NGPUS_PER_NODE / teacher_tp))}

    max_prompt_length=${MAX_PROMPT_LENGTH:-$((1 * 1024))}
    max_response_length=${MAX_RESPONSE_LENGTH:-$((23 * 1024))}
    max_num_tokens=${MAX_NUM_TOKENS:-$((max_prompt_length + max_response_length + 1))}
    rollout_max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-2048}
    teacher_max_num_batched_tokens=${TEACHER_MAX_NUM_BATCHED_TOKENS:-2048}
    rollout_max_num_seqs=${ROLLOUT_MAX_NUM_SEQS:-128}
    teacher_max_num_seqs=${TEACHER_MAX_NUM_SEQS:-128}
    rollout_n=${ROLLOUT_N:-4}
    rollout_temperature=${ROLLOUT_TEMPERATURE:-1.0}
    rollout_top_p=${ROLLOUT_TOP_P:-1.0}
    rollout_top_k=${ROLLOUT_TOP_K:--1}
    rollout_do_sample=${ROLLOUT_DO_SAMPLE:-True}
    val_temperature=${VAL_TEMPERATURE:-0.7}
    val_top_p=${VAL_TOP_P:-0.95}
    val_top_k=${VAL_TOP_K:-$rollout_top_k}
    val_do_sample=${VAL_DO_SAMPLE:-True}
    val_n=${VAL_N:-16}

    actor_lr=${ACTOR_LR:-1e-6}
    train_batch_size=${TRAIN_BATCH_SIZE:-64}
    ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-64}
    ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-$(( ((max_num_tokens + sp_size - 1) / sp_size + 127) / 128 * 128 ))}
    train_max_samples=${TRAIN_MAX_SAMPLES:--1}
    val_max_samples=${VAL_MAX_SAMPLES:-256}
    val_batch_size=${VAL_BATCH_SIZE:-256}
    data_seed=${DATA_SEED:--42}

    distillation_loss_mode=${DISTILLATION_LOSS_MODE:-k1}
    top_k_strategy=${TOP_K_STRATEGY:-only_stu}
    reward_weight_mode=${REWARD_WEIGHT_MODE:-student_p}
    use_task_rewards=${USE_TASK_REWARDS:-False}
    use_policy_gradient=${USE_POLICY_GRADIENT:-True}
    distillation_topk=${DISTILLATION_TOPK:-16}
    distillation_loss_max_clamp=${DISTILLATION_LOSS_MAX_CLAMP:-10.0}
    distillation_log_prob_min_clamp=${DISTILLATION_LOG_PROB_MIN_CLAMP:--10.0}
    opd_diagnostics_enabled=${OPD_DIAGNOSTICS_ENABLED:-True}
    opd_diagnostics_collect_topk_overlap=${OPD_DIAGNOSTICS_COLLECT_TOPK_OVERLAP:-True}
    opd_diagnostics_collect_rethinking_opd=${OPD_DIAGNOSTICS_COLLECT_RETHINKING_OPD:-True}
    opd_diagnostics_collect_repetition=${OPD_DIAGNOSTICS_COLLECT_REPETITION:-True}
    opd_diagnostics_collect_train_text_zlib_repetition=${OPD_DIAGNOSTICS_COLLECT_TRAIN_TEXT_ZLIB_REPETITION:-True}
    opd_diagnostics_topk=${OPD_DIAGNOSTICS_TOPK:-16}
    opd_diagnostics_repetition_tail_chars=${OPD_DIAGNOSTICS_REPETITION_TAIL_CHARS:-10000}
    opd_diagnostics_repetition_comp_ratio_threshold=${OPD_DIAGNOSTICS_REPETITION_COMP_RATIO_THRESHOLD:-10.0}
    opd_diagnostics_repetition_tail_token_window=${OPD_DIAGNOSTICS_REPETITION_TAIL_TOKEN_WINDOW:-4096}
    opd_diagnostics_position_bin_size=${OPD_DIAGNOSTICS_POSITION_BIN_SIZE:-512}
    opd_diagnostics_log_interval=${OPD_DIAGNOSTICS_LOG_INTERVAL:-5}
    opd_diagnostics_topk_log_interval=${OPD_DIAGNOSTICS_TOPK_LOG_INTERVAL:-10}
    opd_diagnostics_async_metrics=${OPD_DIAGNOSTICS_ASYNC_METRICS:-True}
    opd_diagnostics_async_max_pending=${OPD_DIAGNOSTICS_ASYNC_MAX_PENDING:-1}
    opd_diagnostics_heatmap_interval=${OPD_DIAGNOSTICS_HEATMAP_INTERVAL:-10}
    opd_diagnostics_heatmap_history_steps=${OPD_DIAGNOSTICS_HEATMAP_HISTORY_STEPS:-80}
    opd_diagnostics_log_heatmap=${OPD_DIAGNOSTICS_LOG_HEATMAP:-True}
    opd_diagnostics_high_gap_threshold=${OPD_DIAGNOSTICS_HIGH_GAP_THRESHOLD:-1.0}
    opd_diagnostics_bad_tail_min_position=${OPD_DIAGNOSTICS_BAD_TAIL_MIN_POSITION:-12000}
    opd_diagnostics_bad_tail_deep_position=${OPD_DIAGNOSTICS_BAD_TAIL_DEEP_POSITION:-16000}
    opd_diagnostics_bad_tail_extreme_position=${OPD_DIAGNOSTICS_BAD_TAIL_EXTREME_POSITION:-20000}
    opd_diagnostics_bad_tail_overlap_threshold=${OPD_DIAGNOSTICS_BAD_TAIL_OVERLAP_THRESHOLD:-0.5}
    opd_diagnostics_bad_tail_teacher_prob_threshold=${OPD_DIAGNOSTICS_BAD_TAIL_TEACHER_PROB_THRESHOLD:-0.2}
    opd_diagnostics_bad_tail_gap_threshold=${OPD_DIAGNOSTICS_BAD_TAIL_GAP_THRESHOLD:-1.0}
    opd_diagnostics_bad_tail_reward_threshold=${OPD_DIAGNOSTICS_BAD_TAIL_REWARD_THRESHOLD:--2.0}
    opd_diagnostics_bad_tail_entropy_threshold=${OPD_DIAGNOSTICS_BAD_TAIL_ENTROPY_THRESHOLD:-4.0}

    adv_estimator=${ADV_ESTIMATOR:-token_reward_direct}
    distillation_use_actor_loss=${DISTILLATION_USE_ACTOR_LOSS:-False}
    use_kl_in_reward=${USE_KL_IN_REWARD:-False}
    filter_overlong_prompts=${FILTER_OVERLONG_PROMPTS:-True}
    data_truncation=${DATA_TRUNCATION:-error}
    data_shuffle=${DATA_SHUFFLE:-False}
    enable_gradient_checkpointing=${ENABLE_GRADIENT_CHECKPOINTING:-True}
    rollout_name=${ROLLOUT_NAME:-vllm}
    reward_manager_name=${REWARD_MANAGER_NAME:-naive}
    trainer_logger=${TRAINER_LOGGER:-'["console","wandb"]'}
    trainer_device=${TRAINER_DEVICE:-npu}
    val_before_train=${VAL_BEFORE_TRAIN:-True}
    teacher_val_before_train=${TEACHER_VAL_BEFORE_TRAIN:-False}
    distillation_enabled=${DISTILLATION_ENABLED:-True}

    project_name=${PROJECT_NAME:-DMPD}
    experiment_timestamp=${EXPERIMENT_TIMESTAMP:-$(date +"%Y%m%d-%H%M%S")}
    experiment_base_name=${EXPERIMENT_NAME:-${DEFAULT_EXPERIMENT_NAME:-opd}}
    experiment_name=${EXPERIMENT_FULL_NAME:-"${experiment_base_name}-${experiment_timestamp}"}
    default_local_dir=${DEFAULT_LOCAL_DIR:-$WORKSPACE/outputs/${experiment_name}}
    total_epochs=${TOTAL_EPOCHS:-1}
    save_freq=${SAVE_FREQ:-20}
    test_freq=${TEST_FREQ:-10}
    resume_required=${RESUME_REQUIRED:-False}

    export WANDB_MODE=${WANDB_MODE:-online}
    export WANDB_START_METHOD=${WANDB_START_METHOD:-thread}
    case "$resume_required" in
        True) export WANDB_RESUME=must ;;
        False) ;;
        *)
            echo "RESUME_REQUIRED must be True or False, got: $resume_required" >&2
            exit 1
            ;;
    esac

    export NODE_RANK=${NODE_RANK:-$VC_TASK_INDEX}
    export MASTER_ADDR=${MASTER_ADDR:-"${MA_VJ_NAME}-${MA_TASK_NAME}-0.${MA_VJ_NAME}"}
    export TMPDIR=${TMPDIR:-/cache/ray_tmp}
    mkdir -p "$TMPDIR"

    export MASTER_PORT=${MASTER_PORT:-6167}
    export DASHBOARD_PORT=${DASHBOARD_PORT:-8191}
    export NPUS_PER_NODE=${NPUS_PER_NODE:-8}
    export WORLD_SIZE=$((NPUS_PER_NODE * NNODES))

    SOCKET_IFNAME=${SOCKET_IFNAME:-eth0}
    export CURRENT_IP=$(ifconfig "$SOCKET_IFNAME" | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')

    echo "Cleaning up old Ray processes..."
    ray stop --force
    sleep "${RAY_CLEANUP_SLEEP_SECONDS:-5}"
    rm -rf "$TMPDIR"/*
    pkill -9 -f raylet || true
    pkill -9 -f plasma_store || true
    pkill -9 -f gcs_server || true
    echo "Waiting ${NPU_RELEASE_SLEEP_SECONDS:-20}s for NPU to release resources..."
    npu-smi info
    sleep "${NPU_RELEASE_SLEEP_SECONDS:-20}"

    export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-0}
    export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-eth0}
    export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
    export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-eth0}
    export ASCEND_GLOBAL_LOG_LEVEL=${ASCEND_GLOBAL_LOG_LEVEL:-3}
    export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
    export RAY_gcs_server_rpc_server_thread_num=${RAY_gcs_server_rpc_server_thread_num:-32}
    export RAY_gcs_server_request_timeout_seconds=${RAY_gcs_server_request_timeout_seconds:-600}
    export RAY_timeout_ms=${RAY_timeout_ms:-600000}
    export RAY_worker_register_timeout_seconds=${RAY_worker_register_timeout_seconds:-600}
    export RAY_USAGE_STATS_ENABLED=${RAY_USAGE_STATS_ENABLED:-0}

    if [ "$NODE_RANK" -eq 0 ]; then
        _run_npu_head_node
    else
        _run_npu_worker_node
    fi
}

_run_npu_head_node() {
    echo "Starting Ray HEAD node on $CURRENT_IP..."
    ray start --head \
        --node-ip-address="$MA_CURRENT_IP" \
        --port="$MASTER_PORT" \
        --dashboard-host=0.0.0.0 \
        --dashboard-port="$DASHBOARD_PORT" \
        --resources='{"NPU": '"$NPUS_PER_NODE"'}' \
        --block &
    local ray_head_pid=$!

    echo "Waiting 10s for GCS to initialize..."
    sleep "${RAY_GCS_INIT_SLEEP_SECONDS:-10}"

    echo "Waiting for worker nodes to join..."
    target_npus=$((NNODES * NPUS_PER_NODE))

    while true; do
        ray_status_output=$(ray status 2>&1)
        current_total_npus=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*NPU)' | awk '{s+=$1} END {print int(s)}')
        if [ -z "$current_total_npus" ]; then current_total_npus=0; fi

        echo "Check Resources: Found $current_total_npus / $target_npus NPUs"

        if [ "$current_total_npus" -ge "$target_npus" ]; then
            echo "Ray cluster is READY! (Found $current_total_npus NPUs)."
            break
        else
            echo "Waiting for workers... Retrying in 5s..."
            sleep 5
        fi
    done

    local job_exit_code=0
    local errexit_was_set=0
    case "$-" in
        *e*) errexit_was_set=1 ;;
    esac
    set +e
    _submit_npu_verl_job
    job_exit_code=$?
    if [[ "$errexit_was_set" -eq 1 ]]; then
        set -e
    else
        set +e
    fi

    echo "Ray job finished with exit code: $job_exit_code"
    echo "Check Dashboard at http://$CURRENT_IP:$DASHBOARD_PORT"
    df -h || true
    echo "Stopping Ray cluster before exiting head node..."
    ray stop --force || true
    wait "$ray_head_pid" 2>/dev/null || true
    exit "$job_exit_code"
}

_run_npu_worker_node() {
    echo "Starting Ray WORKER node connecting to $MASTER_ADDR..."

    local restart_on_exit=${RAY_WORKER_RESTART_ON_EXIT:-False}
    while true; do
        echo "Attempting to connect to Ray Head ($MASTER_ADDR:$MASTER_PORT) from $MA_CURRENT_IP..."

        ray start --address="$MASTER_ADDR:$MASTER_PORT" \
            --node-ip-address="$MA_CURRENT_IP" \
            --resources='{"NPU": '"$NPUS_PER_NODE"'}' \
            --block

        EXIT_CODE=$?
        echo "Ray worker exited with code $EXIT_CODE."
        if [[ "$restart_on_exit" != "True" && "$restart_on_exit" != "true" && "$restart_on_exit" != "1" ]]; then
            exit "$EXIT_CODE"
        fi
        echo "RAY_WORKER_RESTART_ON_EXIT=${restart_on_exit}; retrying in 15s..."
        sleep 15
    done
}

_submit_npu_verl_job() {
    _check_resume_required || return 1

    local -a hydra_args=(
        "algorithm.adv_estimator=${adv_estimator}"
        "algorithm.use_kl_in_reward=${use_kl_in_reward}"
        "data.train_files=${train_files}"
        "data.val_files=${valid_files}"
        "data.train_batch_size=${train_batch_size}"
        "data.val_batch_size=${val_batch_size}"
        "data.train_max_samples=${train_max_samples}"
        "data.val_max_samples=${val_max_samples}"
        "data.max_prompt_length=${max_prompt_length}"
        "data.max_response_length=${max_response_length}"
        "data.filter_overlong_prompts=${filter_overlong_prompts}"
        "data.truncation=${data_truncation}"
        "data.shuffle=${data_shuffle}"
        "data.seed=${data_seed}"
        "actor_rollout_ref.model.path=${STUDENT_MODEL}"
        "actor_rollout_ref.model.chat_template_from_model=${CHAT_TEMPLATE_MODEL}"
        "actor_rollout_ref.model.use_remove_padding=True"
        "actor_rollout_ref.model.enable_gradient_checkpointing=${enable_gradient_checkpointing}"
        "actor_rollout_ref.actor.use_torch_compile=False"
        "actor_rollout_ref.actor.optim.lr=${actor_lr}"
        "actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}"
        "actor_rollout_ref.actor.use_dynamic_bsz=True"
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}"
        "actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size}"
        "actor_rollout_ref.actor.fsdp_config.param_offload=True"
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True"
        "actor_rollout_ref.actor.fsdp_config.forward_prefetch=True"
        "actor_rollout_ref.ref.fsdp_config.forward_prefetch=True"
        "actor_rollout_ref.rollout.name=${rollout_name}"
        "actor_rollout_ref.rollout.tensor_model_parallel_size=${student_tp}"
        "actor_rollout_ref.rollout.gpu_memory_utilization=${student_gpu_mem_util}"
        "actor_rollout_ref.rollout.n=${rollout_n}"
        "actor_rollout_ref.rollout.temperature=${rollout_temperature}"
        "actor_rollout_ref.rollout.top_p=${rollout_top_p}"
        "actor_rollout_ref.rollout.top_k=${rollout_top_k}"
        "actor_rollout_ref.rollout.do_sample=${rollout_do_sample}"
        "actor_rollout_ref.rollout.max_model_len=${max_num_tokens}"
        "actor_rollout_ref.rollout.max_num_batched_tokens=${rollout_max_num_batched_tokens}"
        "actor_rollout_ref.rollout.max_num_seqs=${rollout_max_num_seqs}"
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True"
        "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}"
        "actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature}"
        "actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p}"
        "actor_rollout_ref.rollout.val_kwargs.top_k=${val_top_k}"
        "actor_rollout_ref.rollout.val_kwargs.do_sample=${val_do_sample}"
        "actor_rollout_ref.rollout.val_kwargs.n=${val_n}"
        "reward.reward_manager.name=${reward_manager_name}"
        "trainer.balance_batch=True"
        "trainer.logger=${trainer_logger}"
        "trainer.project_name=${project_name}"
        "trainer.experiment_name=${experiment_name}"
        "trainer.n_gpus_per_node=${STUDENT_NGPUS_PER_NODE}"
        "trainer.nnodes=${STUDENT_NNODES}"
        "trainer.device=${trainer_device}"
        "trainer.default_local_dir=${default_local_dir}"
        "trainer.val_before_train=${val_before_train}"
        "trainer.teacher_val_before_train=${teacher_val_before_train}"
        "trainer.teacher_val_max_samples=${val_max_samples}"
        "trainer.log_teacher_val_generations=${val_batch_size}"
        "trainer.save_freq=${save_freq}"
        "trainer.test_freq=${test_freq}"
        "trainer.total_epochs=${total_epochs}"
        "distillation.enabled=${distillation_enabled}"
        "distillation.use_actor_loss=${distillation_use_actor_loss}"
        "distillation.n_gpus_per_node=${TEACHER_NGPUS_PER_NODE}"
        "distillation.nnodes=${TEACHER_NNODES}"
        "distillation.teacher_models.teacher_model.model_path=${TEACHER_MODEL}"
        "distillation.teacher_models.teacher_model.num_replicas=${TEACHER_REPLICAS}"
        "distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${teacher_tp}"
        "distillation.teacher_models.teacher_model.inference.name=vllm"
        "distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${teacher_gpu_mem_util}"
        "distillation.teacher_models.teacher_model.inference.max_model_len=${max_num_tokens}"
        "distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=${teacher_max_num_batched_tokens}"
        "distillation.teacher_models.teacher_model.inference.max_num_seqs=${teacher_max_num_seqs}"
        "distillation.distillation_loss.loss_mode=${distillation_loss_mode}"
        "distillation.distillation_loss.topk=${distillation_topk}"
        "distillation.distillation_loss.top_k_strategy=${top_k_strategy}"
        "distillation.distillation_loss.reward_weight_mode=${reward_weight_mode}"
        "distillation.distillation_loss.use_task_rewards=${use_task_rewards}"
        "distillation.distillation_loss.use_policy_gradient=${use_policy_gradient}"
        "distillation.distillation_loss.loss_max_clamp=${distillation_loss_max_clamp}"
        "distillation.distillation_loss.log_prob_min_clamp=${distillation_log_prob_min_clamp}"
        "distillation.diagnostics.enabled=${opd_diagnostics_enabled}"
        "distillation.diagnostics.collect_topk_overlap=${opd_diagnostics_collect_topk_overlap}"
        "distillation.diagnostics.collect_rethinking_opd_metrics=${opd_diagnostics_collect_rethinking_opd}"
        "distillation.diagnostics.collect_repetition_metrics=${opd_diagnostics_collect_repetition}"
        "distillation.diagnostics.collect_train_text_zlib_repetition=${opd_diagnostics_collect_train_text_zlib_repetition}"
        "distillation.diagnostics.topk=${opd_diagnostics_topk}"
        "distillation.diagnostics.repetition_tail_chars=${opd_diagnostics_repetition_tail_chars}"
        "distillation.diagnostics.repetition_comp_ratio_threshold=${opd_diagnostics_repetition_comp_ratio_threshold}"
        "distillation.diagnostics.repetition_tail_token_window=${opd_diagnostics_repetition_tail_token_window}"
        "distillation.diagnostics.position_bin_size=${opd_diagnostics_position_bin_size}"
        "distillation.diagnostics.log_interval=${opd_diagnostics_log_interval}"
        "distillation.diagnostics.topk_log_interval=${opd_diagnostics_topk_log_interval}"
        "distillation.diagnostics.async_metrics=${opd_diagnostics_async_metrics}"
        "distillation.diagnostics.async_max_pending=${opd_diagnostics_async_max_pending}"
        "distillation.diagnostics.heatmap_interval=${opd_diagnostics_heatmap_interval}"
        "distillation.diagnostics.heatmap_history_steps=${opd_diagnostics_heatmap_history_steps}"
        "distillation.diagnostics.log_wandb_heatmap=${opd_diagnostics_log_heatmap}"
        "distillation.diagnostics.high_gap_threshold=${opd_diagnostics_high_gap_threshold}"
        "distillation.diagnostics.bad_tail_min_position=${opd_diagnostics_bad_tail_min_position}"
        "distillation.diagnostics.bad_tail_deep_position=${opd_diagnostics_bad_tail_deep_position}"
        "distillation.diagnostics.bad_tail_extreme_position=${opd_diagnostics_bad_tail_extreme_position}"
        "distillation.diagnostics.bad_tail_overlap_threshold=${opd_diagnostics_bad_tail_overlap_threshold}"
        "distillation.diagnostics.bad_tail_teacher_prob_threshold=${opd_diagnostics_bad_tail_teacher_prob_threshold}"
        "distillation.diagnostics.bad_tail_gap_threshold=${opd_diagnostics_bad_tail_gap_threshold}"
        "distillation.diagnostics.bad_tail_reward_threshold=${opd_diagnostics_bad_tail_reward_threshold}"
        "distillation.diagnostics.bad_tail_entropy_threshold=${opd_diagnostics_bad_tail_entropy_threshold}"
    )

    hydra_args+=("${METHOD_HYDRA_OVERRIDES[@]}")

    local runtime_env_json
    if [[ "$resume_required" == "True" ]]; then
        runtime_env_json='{
            "working_dir": "./",
            "excludes": ["/.git/"],
            "env_vars": {
                "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "HCCL_HOST_SOCKET_PORT_RANGE": "auto",
                "HCCL_NPU_SOCKET_PORT_RANGE": "auto",
                "WANDB_RUN_ID": "'"$WANDB_RUN_ID"'",
                "WANDB_RESUME": "'"$WANDB_RESUME"'"
            }
        }'
    else
        runtime_env_json='{
            "working_dir": "./",
            "excludes": ["/.git/"],
            "env_vars": {
                "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "HCCL_HOST_SOCKET_PORT_RANGE": "auto",
                "HCCL_NPU_SOCKET_PORT_RANGE": "auto"
            }
        }'
    fi

    ray job submit \
        --address="http://127.0.0.1:$DASHBOARD_PORT" \
        --runtime-env-json="$runtime_env_json" \
        -- \
        python3 -m verl.trainer.main_ppo "${hydra_args[@]}"
}
