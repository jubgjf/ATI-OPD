#!/usr/bin/env bash
# On-policy distillation | NPU (Ascend) | vLLM rollout | FSDP training

set -x

HOME=${HOME:-/home/ma-user}
DATASET_DIR=/opt/huawei/dataset/jnguan
WORKSPACE=$DATASET_DIR/code/DMPD

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

. $HOME/anaconda3/etc/profile.d/conda.sh
mkdir -p $HOME/anaconda3/envs/verl
tar -xf $DATASET_DIR/libs/verl-conda-env.tar.gz -C $HOME/anaconda3/envs/verl
conda env list
conda activate verl
echo ============================

cp -r "${DATASET_DIR}/libs/vllm" "${HOME}/vllm-src"
cp -r "${DATASET_DIR}/libs/vllm-ascend" "${HOME}/vllm-ascend-src"
cd "${HOME}/vllm-src" || exit
git checkout v0.11.0
VLLM_TARGET_DEVICE=empty pip install -v -e .
cd "${HOME}/vllm-ascend-src" || exit
git checkout v0.11.0
git submodule update
pip install -v -e .
echo ============================

cd $WORKSPACE
pip install -r requirements-npu.txt
pip install -e .
echo ============================

ma_vj_name=`echo ${MA_VJ_NAME} | sed 's:ma-job:modelarts-job:g'`
task_name="worker-${VC_TASK_INDEX}"
task_plog_path=${MA_LOG_DIR}/${ma_vj_name}/${task_name}
mkdir -p ${task_plog_path}
export ASCEND_PROCESS_LOG_PATH=${ASCEND_PROCESS_LOG_PATH}/${VC_TASK_INDEX}
echo "plog path: ${ASCEND_PROCESS_LOG_PATH}"

MASTER_ADDR="${MA_VJ_NAME}-${MA_TASK_NAME}-${VC_TASK_INDEX}.${MA_VJ_NAME}"
MASTER_PORT=${PORT}
MA_CURRENT_INSTANCE_NAME="$MA_CURRENT_INSTANCE_NAME"

export HCCL_EXEC_TIMEOUT=3600
export HCCL_LOG_LEVEL=0
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_BUFFSIZE=2000
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_ASCEND_ENABLE_NZ=0
export HCCL_ASYNC_ERROR_HANDLING=0  # 关闭 watchdog，防止通信超时被误杀
ulimit -n 65536                     # 句柄上限提升到 65536

export CLOSE_MATMUL_K_SHIFT=1
export ATB_MATMUL_SHUFFLE_K_ENABLE=0
export HCCL_DETERMINISTIC="true"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_V1=1

STUDENT_MODEL=$DATASET_DIR/models/Qwen/Qwen3-8B
TEACHER_MODEL=$DATASET_DIR/models/Qwen/Qwen3-32B

dapo_math_train=$DATASET_DIR/datasets/BytedTsinghua-SIA/DAPO-Math-17k-verl/train.parquet
dapo_math_valid=$DATASET_DIR/datasets/BytedTsinghua-SIA/DAPO-Math-17k-verl/test.parquet

train_files="['$dapo_math_train']"
valid_files="['$dapo_math_valid']"

# 训练基本参数
max_prompt_length=2048
max_response_length=8192
max_num_tokens=$(( max_prompt_length + max_response_length + 1 ))
rollout_max_num_batched_tokens=$max_num_tokens
teacher_max_num_batched_tokens=$max_num_tokens
rollout_max_num_seqs=256
teacher_max_num_seqs=256

actor_lr=1e-6
train_batch_size=128
ppo_mini_batch_size=128
ppo_max_token_len_per_gpu=12288
train_max_samples=-1
val_max_samples=4096
val_batch_size=256
data_seed=-42

# 分布式与并行参数
NNODES=${NNODES:-4}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-4}
rollout_tp=${ROLLOUT_TP:-2}
rollout_gpu_mem_util=0.9
teacher_tp=${TEACHER_TP:-2}
teacher_gpu_mem_util=0.9

# 蒸馏参数
distillation_loss_mode=k1
use_policy_gradient=True
distillation_topk=64
teacher_intervention_enabled=${TEACHER_INTERVENTION_ENABLED:-True}
teacher_intervention_segment_length=${TEACHER_INTERVENTION_SEGMENT_LENGTH:-128}
teacher_intervention_teacher_length=${TEACHER_INTERVENTION_TEACHER_LENGTH:-128}
teacher_intervention_gap_metric=${TEACHER_INTERVENTION_GAP_METRIC:-k3}
teacher_intervention_gap_threshold=${TEACHER_INTERVENTION_GAP_THRESHOLD:-1.0}
teacher_intervention_max_count=${TEACHER_INTERVENTION_MAX_COUNT:-4}
teacher_intervention_train_teacher=${TEACHER_INTERVENTION_TRAIN_TEACHER:-True}
teacher_intervention_temperature=${TEACHER_INTERVENTION_TEMPERATURE:-0.7}
opd_diagnostics_enabled=${OPD_DIAGNOSTICS_ENABLED:-True}
opd_diagnostics_position_bin_size=${OPD_DIAGNOSTICS_POSITION_BIN_SIZE:-512}
opd_diagnostics_heatmap_interval=${OPD_DIAGNOSTICS_HEATMAP_INTERVAL:-10}
opd_diagnostics_heatmap_history_steps=${OPD_DIAGNOSTICS_HEATMAP_HISTORY_STEPS:-80}
opd_diagnostics_log_heatmap=${OPD_DIAGNOSTICS_LOG_HEATMAP:-True}
opd_diagnostics_high_gap_threshold=${OPD_DIAGNOSTICS_HIGH_GAP_THRESHOLD:-1.0}

# 日志参数
project_name="DMPD"
experiment_timestamp=${EXPERIMENT_TIMESTAMP:-$(date +"%Y%m%d-%H%M%S")}
experiment_base_name=${EXPERIMENT_NAME:-"dmpd"}
experiment_name=${EXPERIMENT_FULL_NAME:-"${experiment_base_name}-${experiment_timestamp}"}
default_local_dir=$WORKSPACE/outputs/${experiment_name}
total_epochs=3
save_freq=200
test_freq=10

mkdir -p ${default_local_dir}

export WANDB_MODE=offline
export WANDB_DIR=$WORKSPACE
export WANDB_START_METHOD=thread
mkdir -p ${WANDB_DIR}

### 启动分布式网络配置
export NODE_RANK=${VC_TASK_INDEX}
export MASTER_ADDR="${MA_VJ_NAME}-${MA_TASK_NAME}-0.${MA_VJ_NAME}"
export MASTER_PORT=${PORT}
export TMPDIR="/cache/ray_tmp"
mkdir -p /cache/ray_tmp

export MASTER_PORT=6167
export DASHBOARD_PORT=8191
export NPUS_PER_NODE=8
export WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

readonly SOCKET_IFNAME="eth0"
export CURRENT_IP=$(ifconfig $SOCKET_IFNAME | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')

echo "Cleaning up old Ray processes..."
ray stop --force
sleep 5
rm -rf /cache/ray_tmp/*
pkill -9 -f raylet || true
pkill -9 -f plasma_store || true
pkill -9 -f gcs_server || true
echo "Waiting 20s for NPU to release resources..."
npu-smi info
sleep 20

export RAY_DEDUP_LOGS=0
export HCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_PROCESS_LOG_PATH=${default_local_dir}/ascend_logs
export HYDRA_FULL_ERROR=1
export RAY_gcs_server_rpc_server_thread_num=32
export RAY_gcs_server_request_timeout_seconds=600
export RAY_timeout_ms=600000
export RAY_worker_register_timeout_seconds=600
export RAY_USAGE_STATS_ENABLED=0

mkdir -p ${default_local_dir}/ascend_logs

cp /opt/huawei/dataset/jnguan/code/dotnetrc $HOME/.netrc


if [ "$NODE_RANK" -eq 0 ]; then
    # ---------- 主节点 (Head Node) ----------
    echo "【防守型策略】主节点休眠 300 秒，等待所有节点物理环境就绪..."
    sleep 300
    
    echo "Starting Ray HEAD node on $CURRENT_IP..."
    # 显式绑定 node-ip-address 为 MA_CURRENT_IP
    ray start --head \
        --node-ip-address=$MA_CURRENT_IP \
        --port=$MASTER_PORT \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=$DASHBOARD_PORT \
        --resources='{"NPU": '$NPUS_PER_NODE'}' \
        --block &

    echo "Waiting 10s for GCS to initialize..."
    sleep 10
    
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
    
    # ---------- 提交 OPD 蒸馏任务 ----------
    echo "Submitting Verl OPD Distillation job to the cluster..."
    ray job submit \
        --address="http://127.0.0.1:$DASHBOARD_PORT" \
        --runtime-env=verl/trainer/runtime_env.yaml \
        -- \
        python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        algorithm.use_kl_in_reward=False \
        data.train_files="$train_files" \
        data.val_files="$valid_files" \
        data.train_batch_size=$train_batch_size \
        data.val_batch_size=$val_batch_size \
        data.train_max_samples=$train_max_samples \
        data.val_max_samples=$val_max_samples \
        data.max_prompt_length=$max_prompt_length \
        data.max_response_length=$max_response_length \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        data.shuffle=False \
        data.seed=${data_seed} \
        actor_rollout_ref.model.path="$STUDENT_MODEL" \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.use_torch_compile=False \
        actor_rollout_ref.actor.optim.lr=${actor_lr} \
        actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
        actor_rollout_ref.ref.fsdp_config.forward_prefetch=True \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp} \
        actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util} \
        actor_rollout_ref.rollout.n=1 \
        actor_rollout_ref.rollout.max_model_len=${max_num_tokens} \
        actor_rollout_ref.rollout.max_num_batched_tokens=${rollout_max_num_batched_tokens} \
        actor_rollout_ref.rollout.max_num_seqs=${rollout_max_num_seqs} \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
        reward.reward_manager.name=naive \
        trainer.balance_batch=True \
        trainer.logger='["console","wandb"]' \
        trainer.project_name=${project_name} \
        trainer.experiment_name=${experiment_name} \
        trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
        trainer.nnodes=${NNODES} \
        trainer.device=npu \
        trainer.default_local_dir=${default_local_dir} \
        trainer.val_before_train=False \
        trainer.save_freq=${save_freq} \
        trainer.test_freq=${test_freq} \
        trainer.total_epochs=${total_epochs} \
        distillation.enabled=True \
        distillation.n_gpus_per_node=${TEACHER_WORLD_SIZE} \
        distillation.nnodes=${NNODES} \
        distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL" \
        distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${teacher_tp} \
        distillation.teacher_models.teacher_model.inference.name=vllm \
        distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${teacher_gpu_mem_util} \
        distillation.teacher_models.teacher_model.inference.max_model_len=${max_num_tokens} \
        distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=${teacher_max_num_batched_tokens} \
        distillation.teacher_models.teacher_model.inference.max_num_seqs=${teacher_max_num_seqs} \
        distillation.distillation_loss.loss_mode=${distillation_loss_mode} \
        distillation.distillation_loss.topk=${distillation_topk} \
        distillation.distillation_loss.use_task_rewards=False \
        distillation.distillation_loss.use_policy_gradient=${use_policy_gradient} \
        distillation.distillation_loss.loss_max_clamp=10.0 \
        distillation.distillation_loss.log_prob_min_clamp=-10.0 \
        distillation.intervention.enabled=${teacher_intervention_enabled} \
        distillation.intervention.student_segment_length=${teacher_intervention_segment_length} \
        distillation.intervention.teacher_segment_length=${teacher_intervention_teacher_length} \
        distillation.intervention.gap_metric=${teacher_intervention_gap_metric} \
        distillation.intervention.gap_threshold=${teacher_intervention_gap_threshold} \
        distillation.intervention.max_interventions_per_sample=${teacher_intervention_max_count} \
        distillation.intervention.train_on_teacher_segments=${teacher_intervention_train_teacher} \
        distillation.intervention.teacher_temperature=${teacher_intervention_temperature} \
        distillation.diagnostics.enabled=${opd_diagnostics_enabled} \
        distillation.diagnostics.position_bin_size=${opd_diagnostics_position_bin_size} \
        distillation.diagnostics.heatmap_interval=${opd_diagnostics_heatmap_interval} \
        distillation.diagnostics.heatmap_history_steps=${opd_diagnostics_heatmap_history_steps} \
        distillation.diagnostics.log_wandb_heatmap=${opd_diagnostics_log_heatmap} \
        distillation.diagnostics.high_gap_threshold=${opd_diagnostics_high_gap_threshold}

    echo "Job submitted. Check Dashboard at http://$CURRENT_IP:$DASHBOARD_PORT"
    df -h
    exit 0
    
else
    # ---------- 从节点 (Worker Node) ----------
    RANDOM_DELAY=$((RANDOM % 15 + 1))
    echo "Waiting $RANDOM_DELAY seconds to stagger worker starts..."
    sleep $RANDOM_DELAY
    echo "Starting Ray WORKER node connecting to $MASTER_ADDR..."
    
    while true; do
        echo "Attempting to connect to Ray Head ($MASTER_ADDR:$MASTER_PORT) from $MA_CURRENT_IP..."
        
        # 绑定自身的 MA_CURRENT_IP
        ray start --address="$MASTER_ADDR:$MASTER_PORT" \
            --node-ip-address=$MA_CURRENT_IP \
            --resources='{"NPU": '$NPUS_PER_NODE'}' \
            --block
            
        EXIT_CODE=$?
        echo "Ray worker 异常退出 (Exit Code: $EXIT_CODE)。可能是连接超时或网络波动，15秒后自动重试..."
        sleep 15
    done
fi
