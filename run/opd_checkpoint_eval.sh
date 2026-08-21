#!/usr/bin/env bash
# Persistent two-role checkpoint evaluation for every DMPD/OPD experiment below one output root.
# Independent two-node jobs coordinate through atomic directory claims on shared storage.

if [[ "${EVAL_SHELL_TRACE:-0}" == "1" ]]; then
    set -x
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
OPD_RESULT_STATUS=$REPO_ROOT/verl/utils/evaluation/opd_result_status.py
DATASET_DIR=/opt/huawei/dataset/jnguan
if command -v npu-smi >/dev/null 2>&1 && [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    npu_environment_available=1
    HOME=/home/ma-user
    WORKSPACE=$DATASET_DIR/code/DMPD
    export HOME
else
    npu_environment_available=0
    WORKSPACE=${WORKSPACE:-$REPO_ROOT}
fi

load_external_eval_config() {
    local config_path=${TRAIN_CONFIG_PATH:-/opt/huawei/algorithm/train.config}
    if [[ ! -s "$config_path" ]]; then
        return 0
    fi
    local allexport_was_set=0
    case "$-" in
        *a*) allexport_was_set=1 ;;
    esac
    set -o allexport
    # shellcheck disable=SC1090
    . "$config_path"
    if [[ "$allexport_was_set" -eq 0 ]]; then
        set +o allexport
    fi
}

setup_npu_eval_environment() {
    local netrc_path=${NETRC_PATH:-$DATASET_DIR/apps/configs/dotnetrc}

    npu-smi info
    # shellcheck disable=SC1091
    . /usr/local/Ascend/ascend-toolkit/set_env.sh
    # shellcheck disable=SC1091
    . /usr/local/Ascend/nnal/atb/set_env.sh
    sudo chown -R ma-user:ma-group /usr/local/Ascend/ascend-toolkit
    sudo chown -R ma-user:ma-group /usr/local/Ascend/nnal
    python3 -c "import acl; print(acl.get_soc_name())"

    sudo yum install curl jemalloc jq tmux util-linux -y
    export LD_PRELOAD=/usr/lib64/libjemalloc.so.2:${LD_PRELOAD:-}

    if [[ "${EVAL_INSTALL_DEPENDENCIES:-1}" == "1" ]]; then
        rm -rf "$HOME"/vllm*
        tar -xf "$DATASET_DIR/libs/vllm-build.tar.gz" -C "$HOME"
        mkdir -p "$HOME/anaconda3/envs/verl"
        tar -xf "$DATASET_DIR/libs/verl-conda-env-20260617.tar.gz" -C "$HOME/anaconda3/envs/verl"
    fi
    # shellcheck disable=SC1091
    . "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate verl
    if [[ "${EVAL_INSTALL_DEPENDENCIES:-1}" == "1" ]]; then
        cd "$WORKSPACE"
        pip install -r requirements-npu.txt
        pip install -e .
    fi
    if [[ -f "$netrc_path" ]]; then
        ln -sfnT "$netrc_path" "$HOME/.netrc"
    fi

    export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-3600}
    export HCCL_LOG_LEVEL=${HCCL_LOG_LEVEL:-0}
    export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-3600}
    export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-2000}
    export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
    export VLLM_ASCEND_ENABLE_NZ=${VLLM_ASCEND_ENABLE_NZ:-0}
    export HCCL_ASYNC_ERROR_HANDLING=${HCCL_ASYNC_ERROR_HANDLING:-0}
    export CLOSE_MATMUL_K_SHIFT=${CLOSE_MATMUL_K_SHIFT:-1}
    export ATB_MATMUL_SHUFFLE_K_ENABLE=${ATB_MATMUL_SHUFFLE_K_ENABLE:-0}
    export HCCL_DETERMINISTIC=${HCCL_DETERMINISTIC:-true}
    export VLLM_ENABLE_V1_MULTIPROCESSING=${VLLM_ENABLE_V1_MULTIPROCESSING:-0}
    export VLLM_USE_V1=${VLLM_USE_V1:-1}
    ulimit -n "${ULIMIT_NOFILE:-65536}"
}

load_external_eval_config
EVAL_SETUP_MODE=${EVAL_SETUP_MODE:-auto}
if [[ "$EVAL_SETUP_MODE" == "auto" ]]; then
    if [[ "$npu_environment_available" == "1" ]]; then
        EVAL_SETUP_MODE=npu
    else
        EVAL_SETUP_MODE=skip
    fi
fi
case "$EVAL_SETUP_MODE" in
    npu) setup_npu_eval_environment ;;
    skip) ;;
    *)
        echo "EVAL_SETUP_MODE must be auto, npu, or skip; got $EVAL_SETUP_MODE." >&2
        exit 2
        ;;
esac

EVAL_ROLE=${EVAL_ROLE:-auto}
if [[ "$EVAL_ROLE" == "auto" ]]; then
    if [[ "${VC_WORKER_NUM:-}" != "2" ]]; then
        echo "EVAL_ROLE=auto requires a two-node task (VC_WORKER_NUM=2)." >&2
        exit 2
    fi
    case "${VC_TASK_INDEX:-}" in
        0) EVAL_ROLE=controller ;;
        1) EVAL_ROLE=teacher ;;
        *)
            echo "EVAL_ROLE=auto requires VC_TASK_INDEX=0 or 1, got ${VC_TASK_INDEX:-unset}." >&2
            exit 2
            ;;
    esac
fi
if [[ "$EVAL_ROLE" != "controller" && "$EVAL_ROLE" != "teacher" ]]; then
    echo "EVAL_ROLE must be auto, controller, or teacher; got $EVAL_ROLE." >&2
    exit 2
fi

resolve_workspace_path() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *) printf '%s/%s\n' "${WORKSPACE%/}" "$1" ;;
    esac
}

if [[ -n "${CKPT_ROOT:-}" ]]; then
    # Backward-compatible single-experiment mode. The default watcher mode below
    # intentionally scans the parent shared by every experiment instead.
    CHECKPOINT_SEARCH_ROOT=$CKPT_ROOT
elif [[ -n "${CHECKPOINT_SEARCH_ROOT:-}" ]]; then
    CHECKPOINT_SEARCH_ROOT=$CHECKPOINT_SEARCH_ROOT
elif [[ -n "${DEFAULT_LOCAL_DIR:-}" ]]; then
    CHECKPOINT_SEARCH_ROOT=$(dirname -- "$(resolve_workspace_path "$DEFAULT_LOCAL_DIR")")
else
    CHECKPOINT_SEARCH_ROOT=$WORKSPACE/outputs
fi
CHECKPOINT_SEARCH_ROOT=$(resolve_workspace_path "$CHECKPOINT_SEARCH_ROOT")
mkdir -p "$CHECKPOINT_SEARCH_ROOT"

EVAL_RESULTS_ROOT=$(resolve_workspace_path "${EVAL_RESULTS_ROOT:-eval-results}")
mkdir -p "$EVAL_RESULTS_ROOT"
WANDB_MODE=${WANDB_MODE:-offline}
WANDB_ENTITY=${WANDB_ENTITY:-jubgjf}
WANDB_PROJECT=${WANDB_PROJECT:-DMPD-eval}

configure_eval_wandb_storage() {
    local wandb_workspace=$WORKSPACE
    if [[ "$WANDB_MODE" == "offline" || "$WANDB_MODE" == "dryrun" ]]; then
        local portable_dataset_dir=${WANDB_PORTABLE_DATASET_DIR:-/home/ma-user/work/dataset/jnguan}
        local portable_workspace=
        portable_dataset_dir=${portable_dataset_dir%/}
        if [[ "$WORKSPACE" == "$DATASET_DIR"/* ]]; then
            if [[ "$portable_dataset_dir" != "${DATASET_DIR%/}" ]]; then
                sudo mkdir -p "$(dirname -- "$portable_dataset_dir")"
                if [[ -e "$portable_dataset_dir" || -L "$portable_dataset_dir" ]]; then
                    if [[ ! "$portable_dataset_dir" -ef "$DATASET_DIR" ]]; then
                        echo "WANDB_PORTABLE_DATASET_DIR does not resolve to DATASET_DIR:" >&2
                        echo "  portable=$portable_dataset_dir" >&2
                        echo "  dataset=$DATASET_DIR" >&2
                        return 1
                    fi
                else
                    sudo ln -s "$DATASET_DIR" "$portable_dataset_dir"
                fi
            fi
            portable_workspace="$portable_dataset_dir/${WORKSPACE#"${DATASET_DIR%/}"/}"
            wandb_workspace=$portable_workspace
        elif [[ "$EVAL_SETUP_MODE" == "npu" ]]; then
            echo "NPU evaluation workspace must be below the shared dataset mount:" >&2
            echo "  workspace=$WORKSPACE" >&2
            echo "  dataset_mount=$DATASET_DIR" >&2
            return 1
        fi
    fi
    if [[ -z "${EVAL_WANDB_ROOT:-}" ]]; then
        EVAL_WANDB_ROOT=$wandb_workspace/wandb_logs/DMPD-eval
    elif [[ "$EVAL_WANDB_ROOT" != /* ]]; then
        EVAL_WANDB_ROOT=${wandb_workspace%/}/$EVAL_WANDB_ROOT
    fi
    mkdir -p "$EVAL_WANDB_ROOT"
}
configure_eval_wandb_storage

if [[ -n "${EVAL_INSTANCE_ID:-}" ]]; then
    EVAL_INSTANCE_ID=$EVAL_INSTANCE_ID
elif [[ -n "${MA_VJ_NAME:-}" && -n "${MA_TASK_NAME:-}" ]]; then
    EVAL_INSTANCE_ID=${MA_VJ_NAME}--${MA_TASK_NAME}
else
    echo "EVAL_INSTANCE_ID is required when MA_VJ_NAME and MA_TASK_NAME are unavailable." >&2
    echo "Set the same EVAL_INSTANCE_ID on the controller and teacher nodes of one evaluation task." >&2
    exit 2
fi
eval_instance_dirname=$(python -c '
import hashlib, re, sys
raw = sys.argv[1]
slug = re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("-.")[:64] or "instance"
print(f"{slug}--{hashlib.sha256(raw.encode()).hexdigest()[:12]}")
' "$EVAL_INSTANCE_ID")

shared_runtime_root=$CHECKPOINT_SEARCH_ROOT/.standalone-eval
instance_runtime_root=$shared_runtime_root/instances/$eval_instance_dirname
mkdir -p "$shared_runtime_root" "$instance_runtime_root"
teacher_request_file=$instance_runtime_root/teacher-request.json
teacher_endpoint_file=$instance_runtime_root/teacher-endpoint.json
teacher_stop_file=$instance_runtime_root/STOP_TEACHER
eval_cache_base=$shared_runtime_root/merged-checkpoints
eval_log_base=$instance_runtime_root/logs
mkdir -p "$eval_cache_base" "$eval_log_base"

POLL_SECONDS=${POLL_SECONDS:-600}
TEACHER_REQUEST_POLL_SECONDS=${TEACHER_REQUEST_POLL_SECONDS:-2}
RETRY_SECONDS=${RETRY_SECONDS:-300}
WATCH_ONCE=${WATCH_ONCE:-0}
EVAL_CLAIM_HEARTBEAT_SECONDS=${EVAL_CLAIM_HEARTBEAT_SECONDS:-30}
EVAL_CLAIM_STALE_SECONDS=${EVAL_CLAIM_STALE_SECONDS:-300}
EVAL_CLAIM_POLL_SECONDS=${EVAL_CLAIM_POLL_SECONDS:-30}
EVAL_WANDB_LOCK_WAIT_SECONDS=${EVAL_WANDB_LOCK_WAIT_SECONDS:-3600}
EVAL_QUEUE_MAX_ROWS=${EVAL_QUEUE_MAX_ROWS:-0}
if ! python -c '
import sys
heartbeat, stale, poll, wandb_wait = map(float, sys.argv[1:5])
queue_max_rows = int(sys.argv[5])
if heartbeat <= 0 or stale <= heartbeat or poll <= 0 or wandb_wait <= 0 or queue_max_rows < 0:
    raise SystemExit(1)
' "$EVAL_CLAIM_HEARTBEAT_SECONDS" "$EVAL_CLAIM_STALE_SECONDS" "$EVAL_CLAIM_POLL_SECONDS" \
    "$EVAL_WANDB_LOCK_WAIT_SECONDS" "$EVAL_QUEUE_MAX_ROWS"
then
    echo "Invalid claim settings: require heartbeat>0, stale>heartbeat, poll>0, W&B wait>0, queue rows>=0." >&2
    exit 2
fi
EVAL_STEPS_TEXT=${EVAL_STEPS_TEXT:-}
EVAL_EXPERIMENTS_TEXT=${EVAL_EXPERIMENTS_TEXT:-}
KEEP_MERGED_CHECKPOINTS=${KEEP_MERGED_CHECKPOINTS:-0}
OVERWRITE_RESULTS=${OVERWRITE_RESULTS:-0}
CONTINUE_ON_EVAL_ERROR=${CONTINUE_ON_EVAL_ERROR:-1}
ALLOW_DEBUG_OVERRIDES=${ALLOW_DEBUG_OVERRIDES:-0}

SERVER_READY_TIMEOUT=${SERVER_READY_TIMEOUT:-3600}
SERVER_STOP_TIMEOUT=${SERVER_STOP_TIMEOUT:-180}
TEACHER_PORT=${TEACHER_PORT:-8001}
STUDENT_PORT=${STUDENT_PORT:-8000}
TEACHER_BIND_HOST=${TEACHER_BIND_HOST:-0.0.0.0}
STUDENT_BIND_HOST=${STUDENT_BIND_HOST:-127.0.0.1}
TEACHER_ADVERTISE_HOST=${TEACHER_ADVERTISE_HOST:-}
if [[ -z "$TEACHER_ADVERTISE_HOST" ]]; then
    TEACHER_ADVERTISE_HOST=$(hostname -I 2>/dev/null | awk '{print $1}' || true)
fi
TEACHER_ADVERTISE_HOST=${TEACHER_ADVERTISE_HOST:-127.0.0.1}

EVAL_DEVICES_PER_NODE=${EVAL_DEVICES_PER_NODE:-8}
STUDENT_TP_SIZE=${STUDENT_TP_SIZE:-1}
STUDENT_DP_SIZE=${STUDENT_DP_SIZE:-$EVAL_DEVICES_PER_NODE}
TEACHER_TP_SIZE=${TEACHER_TP_SIZE:-1}
TEACHER_DP_SIZE=${TEACHER_DP_SIZE:-$EVAL_DEVICES_PER_NODE}
STUDENT_MAX_NUM_SEQS=${STUDENT_MAX_NUM_SEQS:-64}
TEACHER_MAX_NUM_SEQS=${TEACHER_MAX_NUM_SEQS:-64}
STUDENT_GPU_MEMORY_UTILIZATION=${STUDENT_GPU_MEMORY_UTILIZATION:-0.9}
TEACHER_GPU_MEMORY_UTILIZATION=${TEACHER_GPU_MEMORY_UTILIZATION:-0.9}
STUDENT_DP_RPC_PORT=${STUDENT_DP_RPC_PORT:-13389}
TEACHER_DP_RPC_PORT=${TEACHER_DP_RPC_PORT:-13390}
STUDENT_SCORE_MAX_IN_FLIGHT=${STUDENT_SCORE_MAX_IN_FLIGHT:-128}
REPAIR_MAX_IN_FLIGHT=${REPAIR_MAX_IN_FLIGHT:-128}
REPAIR_MAX_PENDING=${REPAIR_MAX_PENDING:-256}
MAX_CONCURRENCY_PER_DP=${MAX_CONCURRENCY_PER_DP:-64}
server_max_logprobs=${SERVER_MAX_LOGPROBS:-20}

json_value() {
    local expression=$1
    python -c '
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
value = payload
for part in sys.argv[2].split("."):
    value = value[part]
if isinstance(value, (dict, list)):
    print(json.dumps(value, separators=(",", ":")))
elif value is None:
    print("")
else:
    print(value)
' "$EVALUATION_SPEC" "$expression"
}

sampling_value() {
    local key=$1
    local fallback=$2
    python -c '
import json, sys
payload = json.loads(sys.argv[1])
print(payload.get(sys.argv[2], sys.argv[3]))
' "$sampling_json" "$key" "$fallback"
}

load_experiment_context() {
    local candidate_json=$1
    CKPT_ROOT=$(python -c 'import json,sys; print(json.loads(sys.argv[1])["checkpoint_root"])' "$candidate_json")
    EVALUATION_SPEC=$(python -c 'import json,sys; print(json.loads(sys.argv[1])["evaluation_spec"])' "$candidate_json")
    source_checkpoint_identity=$(python -c \
        'import json,sys; print(json.loads(sys.argv[1])["source_checkpoint_identity"])' "$candidate_json")
    spec_fingerprint=$(json_value spec_fingerprint)
    experiment_name=$(json_value semantic_config.experiment_name)
    training_project=$(json_value semantic_config.training_project)
    training_run_id=$(json_value training_run_id)
    dataset_root=$(json_value semantic_config.benchmarks.root)
    student_tokenizer_from_spec=$(json_value semantic_config.student.tokenizer_path)
    chat_template_model=$(json_value semantic_config.student.chat_template_from_model)
    trust_remote_code=$(json_value semantic_config.student.trust_remote_code)
    max_prompt_tokens=$(json_value semantic_config.sampling.max_prompt_length)
    max_completion_tokens=$(json_value semantic_config.sampling.max_response_length)
    max_model_len=$(json_value semantic_config.sampling.max_model_len)
    base_seed=$(json_value semantic_config.sampling.base_seed)
    sampling_json=$(json_value semantic_config.sampling.test_kwargs)
    benchmark_json=$(json_value semantic_config.benchmarks.names)
    max_samples_per_benchmark=$(json_value semantic_config.benchmarks.max_samples_per_benchmark)
    verifier=$(json_value semantic_config.reward.benchmark_reward_data_source)
    intervention_json=$(json_value semantic_config.distillation.intervention)
    if [[ -z "$max_model_len" || "$max_model_len" == "None" || "$max_model_len" == "null" ]]; then
        max_model_len=$((max_prompt_tokens + max_completion_tokens + 1))
    fi

    local teacher_models teacher_count teacher_model_from_spec
    teacher_models=$(json_value semantic_config.teacher.models)
    teacher_count=$(python -c 'import json,sys; print(len(json.loads(sys.argv[1])))' "$teacher_models")
    if [[ "$teacher_count" != "1" ]]; then
        echo "Experiment $experiment_name requires exactly one Teacher model; found $teacher_count." >&2
        return 2
    fi
    teacher_model_from_spec=$(python -c '
import json, sys
value = next(iter(json.loads(sys.argv[1]).values()))
print(value.get("model_path") or value.get("path") or "")
' "$teacher_models")
    TEACHER_MODEL=$teacher_model_from_spec
    TEACHER_TOKENIZER_MODEL=$TEACHER_MODEL
    STUDENT_TOKENIZER_MODEL=$student_tokenizer_from_spec

    if [[ -z "$TEACHER_MODEL" || ! -d "$TEACHER_MODEL" ]]; then
        echo "Teacher model directory is unavailable for $experiment_name: $TEACHER_MODEL" >&2
        return 2
    fi
    if [[ -n "$STUDENT_TOKENIZER_MODEL" && ! -d "$STUDENT_TOKENIZER_MODEL" ]]; then
        echo "Student tokenizer directory is unavailable for $experiment_name: $STUDENT_TOKENIZER_MODEL" >&2
        return 2
    fi
    if [[ -n "$chat_template_model" && ! -d "$chat_template_model" ]]; then
        echo "Chat-template tokenizer directory is unavailable for $experiment_name: $chat_template_model" >&2
        return 2
    fi
    if [[ -z "$dataset_root" || ! -d "$dataset_root" ]]; then
        echo "Benchmark dataset root is unavailable for $experiment_name: $dataset_root" >&2
        return 2
    fi

    readarray -t benchmarks < <(
        python -c 'import json,sys; print("\n".join(str(x) for x in json.loads(sys.argv[1])))' "$benchmark_json"
    )
    if [[ -n "${EVAL_BENCHMARKS_TEXT:-}" ]]; then
        read -r -a benchmarks <<< "$EVAL_BENCHMARKS_TEXT"
    fi
    if ((${#benchmarks[@]} == 0)); then
        echo "Experiment $experiment_name contains no benchmark names." >&2
        return 2
    fi

    rollout_n=${EVAL_ROLLOUT_N:-$(sampling_value n 16)}
    temperature=${EVAL_TEMPERATURE:-$(sampling_value temperature 0.7)}
    top_p=${EVAL_TOP_P:-$(sampling_value top_p 0.95)}
    top_k=${EVAL_TOP_K:-$(sampling_value top_k -1)}
    if [[ "$rollout_n" != "16" && "$ALLOW_DEBUG_OVERRIDES" != "1" ]]; then
        echo "Formal checkpoint evaluation requires n=16 for $experiment_name; got $rollout_n." >&2
        return 2
    fi
    enable_thinking=$(python -c '
import json, sys
student = json.load(open(sys.argv[1], encoding="utf-8"))["semantic_config"]["student"]
print(str(bool((student.get("apply_chat_template_kwargs") or {}).get("enable_thinking", True))).lower())
' "$EVALUATION_SPEC")

    gap_metric=$(python -c \
        'import json,sys; print(str(json.loads(sys.argv[1]).get("gap_metric", "k3")).lower())' \
        "$intervention_json")
    case "$gap_metric" in
        full_kl|full_distribution_kl|distribution_kl)
            echo "HTTP evaluation refuses exact full-vocabulary KL for $experiment_name." >&2
            return 2
            ;;
    esac
    gap_distribution_topk=$(python -c \
        'import json,sys; print(int(json.loads(sys.argv[1]).get("gap_distribution_topk", 16)))' \
        "$intervention_json")
    SERVER_MAX_LOGPROBS=$server_max_logprobs
    case "$gap_metric" in
        topk_kl|topk_distribution_kl|topk_dist_kl)
            if ((gap_distribution_topk > SERVER_MAX_LOGPROBS)); then
                SERVER_MAX_LOGPROBS=$gap_distribution_topk
            fi
            ;;
    esac

    RESULT_ROOT="$EVAL_RESULTS_ROOT/$training_project/$experiment_name"
    EVAL_CACHE_ROOT="$eval_cache_base/$spec_fingerprint"
    EVAL_LOG_ROOT="$eval_log_base/$experiment_name"
    CURRENT_WANDB_ROOT="$EVAL_WANDB_ROOT/$training_project/$experiment_name"
    mkdir -p "$RESULT_ROOT" "$EVAL_CACHE_ROOT" "$EVAL_LOG_ROOT" "$CURRENT_WANDB_ROOT"
}

load_queue_context() {
    local candidate_json=$1
    local context_output
    if ! context_output=$(python -c '
import json, sys
candidate = json.loads(sys.argv[1])
with open(candidate["evaluation_spec"], encoding="utf-8") as handle:
    spec = json.load(handle)
semantic = spec["semantic_config"]
benchmarks = sys.argv[2].split() or [str(value) for value in semantic["benchmarks"]["names"]]
fields = [
    candidate["checkpoint_root"],
    candidate["evaluation_spec"],
    candidate["spec_fingerprint"],
    candidate["experiment_name"],
    candidate["training_project"],
]
if any("\t" in str(value) or "\n" in str(value) for value in [*fields, *benchmarks]):
    raise ValueError("Queue metadata may not contain tabs or newlines")
print("\t".join(str(value) for value in fields))
print("\n".join(benchmarks))
' "$candidate_json" "${EVAL_BENCHMARKS_TEXT:-}")
    then
        return 2
    fi

    local -a context_lines=()
    mapfile -t context_lines <<< "$context_output"
    IFS=$'\t' read -r CKPT_ROOT EVALUATION_SPEC spec_fingerprint experiment_name training_project <<< \
        "${context_lines[0]}"
    benchmarks=("${context_lines[@]:1}")
    if ((${#benchmarks[@]} == 0)); then
        echo "Experiment $experiment_name contains no benchmark names." >&2
        return 2
    fi
    RESULT_ROOT="$EVAL_RESULTS_ROOT/$training_project/$experiment_name"
}

VLLM_PID=""
VLLM_PGID=""
VLLM_LOG=""
ROLE_CLAIM_PATH=""
ROLE_CLAIM_TOKEN=""
ROLE_CLAIM_HEARTBEAT_PID=""
ACTIVE_CLAIM_PATH=""
ACTIVE_CLAIM_TOKEN=""
ACTIVE_CLAIM_HEARTBEAT_PID=""
ACTIVE_CLAIM_LABEL=""
WANDB_CLAIM_PATH=""
WANDB_CLAIM_TOKEN=""
WANDB_CLAIM_HEARTBEAT_PID=""
STARTED_HEARTBEAT_PID=""
QUEUE_CANDIDATES=()
WATCH_ONCE_ATTEMPTED_KEYS=()
REPORTED_STALE_CLAIM_PATHS=()
LAST_QUEUE_SIGNATURE=""

claim_json_value() {
    local payload=$1
    local key=$2
    python -c 'import json,sys; print(json.loads(sys.argv[1])[sys.argv[2]])' "$payload" "$key"
}

start_claim_heartbeat() {
    local claim_path=$1
    local claim_token=$2
    python -m verl.utils.evaluation.checkpoint_watcher heartbeat-claim \
        --claim-path "$claim_path" \
        --claim-token "$claim_token" \
        --loop-seconds "$EVAL_CLAIM_HEARTBEAT_SECONDS" >/dev/null &
    STARTED_HEARTBEAT_PID=$!
}

stop_claim_heartbeat() {
    local heartbeat_pid=${1:-}
    if [[ -z "$heartbeat_pid" ]]; then
        return 0
    fi
    kill "$heartbeat_pid" 2>/dev/null || true
    wait "$heartbeat_pid" 2>/dev/null || true
}

release_claim_safely() {
    local claim_path=${1:-}
    local claim_token=${2:-}
    if [[ -z "$claim_path" || -z "$claim_token" ]]; then
        return 0
    fi
    if ! python -m verl.utils.evaluation.checkpoint_watcher release-claim \
        --claim-path "$claim_path" \
        --claim-token "$claim_token" >/dev/null
    then
        echo "[claim] failed to release $claim_path; inspect it with:" >&2
        echo "  python -m verl.utils.evaluation.checkpoint_watcher list-claims --runtime-root $shared_runtime_root" >&2
        return 1
    fi
}

register_role_claim() {
    local claim_output claim_status
    claim_output=$(python -m verl.utils.evaluation.checkpoint_watcher acquire-claim \
        --runtime-root "$shared_runtime_root" \
        --claim-type role \
        --instance-id "$EVAL_INSTANCE_ID" \
        --owner-pid "$$" \
        --phase "$EVAL_ROLE" \
        --role "$EVAL_ROLE" \
        --stale-after-seconds "$EVAL_CLAIM_STALE_SECONDS")
    claim_status=$?
    if [[ "$claim_status" != "0" ]]; then
        echo "[claim] another $EVAL_ROLE already uses EVAL_INSTANCE_ID=$EVAL_INSTANCE_ID" >&2
        echo "$claim_output" >&2
        local claim_state=UNKNOWN
        if [[ "$claim_status" == "3" ]]; then
            claim_state=$(python -c 'import json,sys; print(json.loads(sys.argv[1]).get("state", "UNKNOWN"))' \
                "$claim_output")
        fi
        if [[ "$claim_state" == "STALE" ]]; then
            print_stale_claim_recovery "$claim_output"
        else
            echo "Inspect claims with:" >&2
            echo "  python -m verl.utils.evaluation.checkpoint_watcher list-claims --runtime-root $shared_runtime_root" >&2
        fi
        return 2
    fi
    ROLE_CLAIM_PATH=$(claim_json_value "$claim_output" claim_path)
    ROLE_CLAIM_TOKEN=$(claim_json_value "$claim_output" claim_token)
    start_claim_heartbeat "$ROLE_CLAIM_PATH" "$ROLE_CLAIM_TOKEN"
    ROLE_CLAIM_HEARTBEAT_PID=$STARTED_HEARTBEAT_PID
}

update_active_claim_phase() {
    local phase=$1
    if [[ -z "$ACTIVE_CLAIM_PATH" || -z "$ACTIVE_CLAIM_TOKEN" ]]; then
        return 0
    fi
    python -m verl.utils.evaluation.checkpoint_watcher heartbeat-claim \
        --claim-path "$ACTIVE_CLAIM_PATH" \
        --claim-token "$ACTIVE_CLAIM_TOKEN" \
        --phase "$phase" >/dev/null
    echo "[current] instance=$EVAL_INSTANCE_ID checkpoint=${ACTIVE_CLAIM_LABEL:-unknown} phase=$phase"
}

release_active_claim() {
    stop_claim_heartbeat "$ACTIVE_CLAIM_HEARTBEAT_PID"
    ACTIVE_CLAIM_HEARTBEAT_PID=""
    release_claim_safely "$ACTIVE_CLAIM_PATH" "$ACTIVE_CLAIM_TOKEN" || true
    ACTIVE_CLAIM_PATH=""
    ACTIVE_CLAIM_TOKEN=""
    ACTIVE_CLAIM_LABEL=""
}

release_wandb_claim() {
    stop_claim_heartbeat "$WANDB_CLAIM_HEARTBEAT_PID"
    WANDB_CLAIM_HEARTBEAT_PID=""
    release_claim_safely "$WANDB_CLAIM_PATH" "$WANDB_CLAIM_TOKEN" || true
    WANDB_CLAIM_PATH=""
    WANDB_CLAIM_TOKEN=""
}

port_is_open() {
    local host=$1
    local port=$2
    python -c '
import socket, sys
sock = socket.socket()
sock.settimeout(1)
code = sock.connect_ex((sys.argv[1], int(sys.argv[2])))
sock.close()
raise SystemExit(0 if code == 0 else 1)
' "$host" "$port" >/dev/null 2>&1
}

model_is_ready() {
    local base_url=$1
    local api_key=$2
    local served_model=$3
    curl --silent --show-error --fail --connect-timeout 2 --max-time 10 \
        --header "Authorization: Bearer $api_key" "$base_url/models" | \
        python -c '
import json,sys
expected=sys.argv[1]
payload=json.load(sys.stdin)
raise SystemExit(0 if expected in {str(item.get("id")) for item in payload.get("data", [])} else 1)
' "$served_model" >/dev/null 2>&1
}

wait_for_model() {
    local base_url=$1
    local api_key=$2
    local served_model=$3
    local deadline=$((SECONDS + SERVER_READY_TIMEOUT))
    while ((SECONDS < deadline)); do
        if model_is_ready "$base_url" "$api_key" "$served_model"; then
            return 0
        fi
        if [[ -n "$VLLM_PID" ]] && ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "vLLM exited before serving $served_model; log=$VLLM_LOG" >&2
            tail -n 200 "$VLLM_LOG" >&2 || true
            return 1
        fi
        sleep 5
    done
    echo "Timed out waiting for vLLM model $served_model at $base_url." >&2
    return 1
}

start_vllm() {
    local role=$1
    local model_path=$2
    local tokenizer_path=$3
    local served_model=$4
    local bind_host=$5
    local port=$6
    local tp_size=$7
    local dp_size=$8
    local max_num_seqs=$9
    local gpu_memory_utilization=${10}
    local dp_rpc_port=${11}
    local api_key=${12}

    if port_is_open 127.0.0.1 "$port"; then
        echo "$role vLLM port is already occupied: $port" >&2
        return 1
    fi
    VLLM_LOG="$EVAL_LOG_ROOT/${role}-$(date +%Y%m%d-%H%M%S).log"
    local -a command=(
        vllm serve "$model_path"
        --served-model-name "$served_model"
        --tokenizer "$tokenizer_path"
        --host "$bind_host"
        --port "$port"
        --api-key "$api_key"
        --tensor-parallel-size "$tp_size"
        --max-num-seqs "$max_num_seqs"
        --max-model-len "$max_model_len"
        --gpu-memory-utilization "$gpu_memory_utilization"
        --max-logprobs "$SERVER_MAX_LOGPROBS"
        --enable-prefix-caching
        --generation-config vllm
    )
    if ((dp_size > 1)); then
        command+=(
            --data-parallel-size "$dp_size"
            --data-parallel-size-local "$dp_size"
            --data-parallel-address 127.0.0.1
            --data-parallel-rpc-port "$dp_rpc_port"
            --data-parallel-start-rank 0
        )
    fi
    if [[ "$trust_remote_code" == "True" || "$trust_remote_code" == "true" ]]; then
        command+=(--trust-remote-code)
    fi
    echo "[vllm] starting role=$role model=$served_model tp=$tp_size dp=$dp_size log=$VLLM_LOG"
    setsid "${command[@]}" </dev/null >"$VLLM_LOG" 2>&1 &
    VLLM_PID=$!
    VLLM_PGID=$VLLM_PID
}

stop_vllm() {
    if [[ -z "$VLLM_PID" ]]; then
        return 0
    fi
    local pid=$VLLM_PID
    local pgid=$VLLM_PGID
    local deadline=$((SECONDS + SERVER_STOP_TIMEOUT))
    kill -TERM -- "-$pgid" 2>/dev/null || true
    while kill -0 "$pid" 2>/dev/null; do
        if ((SECONDS >= deadline)); then
            kill -KILL -- "-$pgid" 2>/dev/null || true
            break
        fi
        sleep 2
    done
    wait "$pid" 2>/dev/null || true
    VLLM_PID=""
    VLLM_PGID=""
}

cleanup() {
    local status=$?
    trap - EXIT
    set +e
    stop_vllm
    release_wandb_claim
    release_active_claim
    if [[ "$EVAL_ROLE" == "controller" && -n "$ROLE_CLAIM_TOKEN" ]]; then
        touch "$teacher_stop_file"
    fi
    stop_claim_heartbeat "$ROLE_CLAIM_HEARTBEAT_PID"
    ROLE_CLAIM_HEARTBEAT_PID=""
    release_claim_safely "$ROLE_CLAIM_PATH" "$ROLE_CLAIM_TOKEN" || true
    ROLE_CLAIM_PATH=""
    ROLE_CLAIM_TOKEN=""
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

teacher_role_main() {
    local teacher_api_key=${TEACHER_API_KEY:-EMPTY}
    local base_url="http://$TEACHER_ADVERTISE_HOST:$TEACHER_PORT/v1"
    rm -f -- "$teacher_stop_file"
    rm -f -- "$teacher_endpoint_file"
    local active_request_id=
    while [[ ! -f "$teacher_stop_file" ]]; do
        if [[ -s "$teacher_request_file" ]]; then
            local -a request_fields=()
            if mapfile -t request_fields < <(python -c '
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
for key in ("request_id", "model_path", "tokenizer_path", "served_model_name", "max_model_len", "max_logprobs", "trust_remote_code"):
    print(payload[key])
' "$teacher_request_file") && ((${#request_fields[@]} == 7)); then
                local requested_id=${request_fields[0]}
                if [[ "$requested_id" != "$active_request_id" ]]; then
                    local requested_model=${request_fields[1]}
                    local requested_tokenizer=${request_fields[2]}
                    local requested_served_model=${request_fields[3]}
                    max_model_len=${request_fields[4]}
                    SERVER_MAX_LOGPROBS=${request_fields[5]}
                    trust_remote_code=${request_fields[6]}
                    if [[ ! -d "$requested_model" || ! -d "$requested_tokenizer" ]]; then
                        echo "[teacher] requested model/tokenizer is unavailable: $requested_model / $requested_tokenizer" >&2
                        sleep "$TEACHER_REQUEST_POLL_SECONDS"
                        continue
                    fi
                    stop_vllm
                    rm -f -- "$teacher_endpoint_file"
                    EVAL_LOG_ROOT="$eval_log_base/teacher"
                    mkdir -p "$EVAL_LOG_ROOT"
                    local load_started=$SECONDS
                    if ! start_vllm teacher "$requested_model" "$requested_tokenizer" "$requested_served_model" \
                        "$TEACHER_BIND_HOST" "$TEACHER_PORT" "$TEACHER_TP_SIZE" "$TEACHER_DP_SIZE" \
                        "$TEACHER_MAX_NUM_SEQS" "$TEACHER_GPU_MEMORY_UTILIZATION" "$TEACHER_DP_RPC_PORT" \
                        "$teacher_api_key" || ! wait_for_model "$base_url" "$teacher_api_key" "$requested_served_model"
                    then
                        stop_vllm
                        sleep "$RETRY_SECONDS"
                        continue
                    fi
                    local load_seconds=$((SECONDS - load_started))
                    python -m verl.utils.evaluation.checkpoint_watcher write-endpoint \
                        --path "$teacher_endpoint_file" \
                        --base-url "$base_url" \
                        --served-model-name "$requested_served_model" \
                        --model-path "$requested_model" \
                        --request-id "$requested_id" \
                        --model-load-seconds "$load_seconds" >/dev/null
                    active_request_id=$requested_id
                    echo "[teacher] ready request=$active_request_id model=$requested_model"
                fi
            fi
        fi
        if [[ -n "$VLLM_PID" ]] && ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "Teacher vLLM exited unexpectedly; it will be restarted from the active request. log=$VLLM_LOG" >&2
            tail -n 200 "$VLLM_LOG" >&2 || true
            VLLM_PID=""
            VLLM_PGID=""
            active_request_id=
            rm -f -- "$teacher_endpoint_file"
        fi
        sleep "$TEACHER_REQUEST_POLL_SECONDS"
    done
    echo "[teacher] controller requested shutdown"
}

request_teacher_endpoint() {
    local model_path
    local tokenizer_path
    model_path=$(cd -- "$TEACHER_MODEL" && pwd)
    tokenizer_path=$(cd -- "$TEACHER_TOKENIZER_MODEL" && pwd)
    local request_id
    request_id=$(python -c '
import hashlib, json, os, sys
path, model, tokenizer = sys.argv[1:4]
payload = {
    "model_path": model,
    "tokenizer_path": tokenizer,
    "max_model_len": int(sys.argv[4]),
    "max_logprobs": int(sys.argv[5]),
    "trust_remote_code": sys.argv[6].lower() == "true",
}
request_id = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
payload.update({"schema_version": 1, "request_id": request_id, "served_model_name": "dmpd-teacher-" + request_id[:12]})
tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
os.replace(tmp, path)
print(request_id)
' "$teacher_request_file" "$model_path" "$tokenizer_path" "$max_model_len" "$SERVER_MAX_LOGPROBS" \
        "$trust_remote_code")

    local deadline=$((SECONDS + SERVER_READY_TIMEOUT))
    while ((SECONDS < deadline)); do
        if [[ -s "$teacher_endpoint_file" ]]; then
            local endpoint_request_id
            endpoint_request_id=$(python -c \
                'import json,sys; print(json.load(open(sys.argv[1])).get("request_id", ""))' \
                "$teacher_endpoint_file")
            if [[ "$endpoint_request_id" == "$request_id" ]]; then
                teacher_base_url=$(python -c \
                    'import json,sys; print(json.load(open(sys.argv[1]))["base_url"])' "$teacher_endpoint_file")
                teacher_served_model=$(python -c \
                    'import json,sys; print(json.load(open(sys.argv[1]))["served_model_name"])' "$teacher_endpoint_file")
                teacher_model_fingerprint=$(python -c \
                    'import json,sys; print(json.load(open(sys.argv[1]))["model_fingerprint"])' \
                    "$teacher_endpoint_file")
                teacher_model_load_seconds=$(python -c \
                    'import json,sys; print(float(json.load(open(sys.argv[1])).get("model_load_seconds", 0.0)))' \
                    "$teacher_endpoint_file")
                if model_is_ready "$teacher_base_url" "${TEACHER_API_KEY:-EMPTY}" "$teacher_served_model"; then
                    return 0
                fi
            fi
        fi
        sleep 5
    done
    echo "Timed out waiting for teacher request $request_id at $teacher_endpoint_file" >&2
    return 1
}

hf_checkpoint_ready() {
    python -m verl.utils.evaluation.opd_benchmark checkpoint-ready --path "$1" >/dev/null 2>&1
}

ensure_hf_checkpoint() {
    local step=$1
    local actor_dir=$2
    local direct_hf=$3
    if [[ -n "$direct_hf" ]] && hf_checkpoint_ready "$direct_hf"; then
        printf '%s\n' "$direct_hf"
        return 0
    fi
    local cache_step="$EVAL_CACHE_ROOT/global_step_$step"
    local target="$cache_step/huggingface"
    if hf_checkpoint_ready "$target"; then
        printf '%s\n' "$target"
        return 0
    fi
    # Match the proven legacy standalone evaluator's direct-merge behavior:
    # some NPU shared filesystems reject renaming a populated directory even
    # within the same parent. Keep the output in an independent cache rather
    # than mutating actor/huggingface; the checkpoint claim guarantees one writer.
    if ! mkdir -p "$cache_step" || ! rm -rf -- "$target" "$cache_step"/.huggingface.tmp.*; then
        echo "Failed to prepare merged-checkpoint cache: $target" >&2
        return 1
    fi
    if ! python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "$actor_dir" \
        --target_dir "$target" >&2
    then
        echo "Failed to merge FSDP checkpoint: $actor_dir" >&2
        return 1
    fi
    if ! hf_checkpoint_ready "$target"; then
        echo "Merged Hugging Face checkpoint is incomplete: $target" >&2
        return 1
    fi
    printf '%s\n' "$target"
}

checkpoint_complete() {
    local step=$1
    local benchmark
    local -a status_args=(
        --result-root "$RESULT_ROOT"
        --global-step "$step"
    )
    for benchmark in "${benchmarks[@]}"; do
        status_args+=(--benchmark "$benchmark")
    done
    python "$OPD_RESULT_STATUS" "${status_args[@]}" >/dev/null 2>&1
}

checkpoint_wandb_is_staged() {
    local step=$1
    if [[ "$WANDB_MODE" == "disabled" ]]; then
        return 0
    fi
    python -c '
import json, sys
try:
    units = json.load(open(sys.argv[1], encoding="utf-8")).get("units", {})
except Exception:
    raise SystemExit(1)
step = int(sys.argv[2])
expected = {f"{benchmark}@global_step_{step}" for benchmark in sys.argv[3:]}
raise SystemExit(0 if expected.issubset(units) else 1)
' "$RESULT_ROOT/wandb-sync.json" "$step" "${benchmarks[@]}"
}

step_is_selected() {
    local candidate=$1
    if [[ -z "$EVAL_STEPS_TEXT" ]]; then
        return 0
    fi
    local selected
    for selected in $EVAL_STEPS_TEXT; do
        if [[ "$candidate" == "$selected" ]]; then
            return 0
        fi
    done
    return 1
}

experiment_is_selected() {
    local candidate=$1
    if [[ -z "$EVAL_EXPERIMENTS_TEXT" ]]; then
        return 0
    fi
    local selected
    for selected in $EVAL_EXPERIMENTS_TEXT; do
        if [[ "$candidate" == "$selected" ]]; then
            return 0
        fi
    done
    return 1
}

run_raw_unit() {
    local step=$1
    local checkpoint=$2
    local benchmark=$3
    local served_model=$4
    local base_url=$5
    local tokenizer_model=$6
    local preflight_only=${7:-0}
    local effective_chat_template_model=${chat_template_model:-$tokenizer_model}
    local -a flags=(
        generate
        --base-url "$base_url"
        --api-key "${STUDENT_API_KEY:-EMPTY}"
        --served-model-name "$served_model"
        --result-root "$RESULT_ROOT"
        --benchmark "$benchmark"
        --global-step "$step"
        --experiment "$experiment_name"
        --dataset-root "$dataset_root"
        --student-model-path "$checkpoint"
        --student-model-identity "$source_checkpoint_identity"
        --student-tokenizer-model "$tokenizer_model"
        --chat-template-model "$effective_chat_template_model"
        --rollout-n "$rollout_n"
        --temperature "$temperature"
        --top-p "$top_p"
        --top-k "$top_k"
        --max-tokens "$max_completion_tokens"
        --max-prompt-tokens "$max_prompt_tokens"
        --dp-size "$STUDENT_DP_SIZE"
        --max-concurrency-per-dp "$MAX_CONCURRENCY_PER_DP"
        --score-max-in-flight "$STUDENT_SCORE_MAX_IN_FLIGHT"
        --score-student-logprobs
        --require-exact-token-ids
        --max-samples-per-benchmark "$max_samples_per_benchmark"
        --base-seed "$base_seed"
        --verifier "$verifier"
    )
    if [[ "$enable_thinking" == "true" ]]; then
        flags+=(--enable-thinking)
    else
        flags+=(--no-enable-thinking)
    fi
    if [[ "$trust_remote_code" == "True" || "$trust_remote_code" == "true" ]]; then
        flags+=(--trust-remote-code)
    fi
    if [[ "$OVERWRITE_RESULTS" == "1" ]]; then
        flags+=(--overwrite)
    fi
    if [[ "$preflight_only" == "1" ]]; then
        flags+=(--preflight-only)
    fi
    python -m verl.utils.evaluation.opd_benchmark "${flags[@]}"
}

run_repair_unit() {
    local step=$1
    local benchmark=$2
    local student_served_model=$3
    local student_base_url=$4
    local tokenizer_model=$5
    local -a flags=(
        repair
        --result-root "$RESULT_ROOT"
        --benchmark "$benchmark"
        --global-step "$step"
        --evaluation-spec "$EVALUATION_SPEC"
        --evaluation-state "$CKPT_ROOT/global_step_$step/evaluation_state.json"
        --student-base-url "$student_base_url"
        --student-api-key "${STUDENT_API_KEY:-EMPTY}"
        --student-served-model-name "$student_served_model"
        --teacher-base-url "$teacher_base_url"
        --teacher-api-key "${TEACHER_API_KEY:-EMPTY}"
        --teacher-served-model-name "$teacher_served_model"
        --teacher-model-fingerprint "$teacher_model_fingerprint"
        --student-tokenizer-model "$tokenizer_model"
        --teacher-tokenizer-model "$TEACHER_TOKENIZER_MODEL"
        --max-in-flight "$REPAIR_MAX_IN_FLIGHT"
        --max-pending "$REPAIR_MAX_PENDING"
    )
    if [[ "$trust_remote_code" == "True" || "$trust_remote_code" == "true" ]]; then
        flags+=(--trust-remote-code)
    fi
    if [[ "${ALLOW_TOKENIZER_MISMATCH:-0}" == "1" ]]; then
        flags+=(--allow-tokenizer-mismatch)
    fi
    if [[ "$OVERWRITE_RESULTS" == "1" ]]; then
        flags+=(--overwrite)
    fi
    python -m verl.utils.evaluation.opd_benchmark "${flags[@]}"
}

sync_wandb() {
    local step=$1
    local wandb_dir="$CURRENT_WANDB_ROOT/wandb"
    local wandb_data_dir="$CURRENT_WANDB_ROOT/wandb-data"
    local wandb_cache_dir="$CURRENT_WANDB_ROOT/wandb-cache"
    mkdir -p "$wandb_dir" "$wandb_data_dir" "$wandb_cache_dir"
    local -a flags=(
        --base-dir "$(dirname -- "$RESULT_ROOT")"
        --experiments "$(basename -- "$RESULT_ROOT")"
        --entity "$WANDB_ENTITY"
        --project "$WANDB_PROJECT"
        --mode "$WANDB_MODE"
        --wandb-dir "$wandb_dir"
        --wandb-data-dir "$wandb_data_dir"
        --wandb-cache-dir "$wandb_cache_dir"
        --source-training-run-id "$training_run_id"
        --global-step "$step"
    )
    python "$SCRIPT_DIR/upload_opd_benchmark_wandb.py" "${flags[@]}"
}

sync_wandb_serialized() {
    local step=$1
    if [[ "$WANDB_MODE" == "disabled" ]] || checkpoint_wandb_is_staged "$step"; then
        return 0
    fi
    update_active_claim_phase "wandb:publish"
    local deadline=$((SECONDS + EVAL_WANDB_LOCK_WAIT_SECONDS))
    local reported_stale=""
    while ((SECONDS < deadline)); do
        local claim_output claim_status
        claim_output=$(python -m verl.utils.evaluation.checkpoint_watcher acquire-claim \
            --runtime-root "$shared_runtime_root" \
            --claim-type wandb \
            --instance-id "$EVAL_INSTANCE_ID" \
            --owner-pid "$$" \
            --phase "wandb:publish" \
            --result-root "$RESULT_ROOT" \
            --experiment-name "$experiment_name" \
            --training-project "$training_project" \
            --spec-fingerprint "$spec_fingerprint" \
            --stale-after-seconds "$EVAL_CLAIM_STALE_SECONDS")
        claim_status=$?
        if [[ "$claim_status" == "0" ]]; then
            WANDB_CLAIM_PATH=$(claim_json_value "$claim_output" claim_path)
            WANDB_CLAIM_TOKEN=$(claim_json_value "$claim_output" claim_token)
            start_claim_heartbeat "$WANDB_CLAIM_PATH" "$WANDB_CLAIM_TOKEN"
            WANDB_CLAIM_HEARTBEAT_PID=$STARTED_HEARTBEAT_PID
            if checkpoint_wandb_is_staged "$step"; then
                release_wandb_claim
                return 0
            fi
            if ! sync_wandb "$step"; then
                release_wandb_claim
                return 1
            fi
            release_wandb_claim
            return 0
        fi
        if [[ "$claim_status" != "3" ]]; then
            echo "[wandb] could not acquire publish claim: $claim_output" >&2
            return 1
        fi
        local claim_state
        claim_state=$(python -c 'import json,sys; print(json.loads(sys.argv[1]).get("state", "UNKNOWN"))' \
            "$claim_output")
        if [[ "$claim_state" == "STALE" && "$reported_stale" != "1" ]]; then
            reported_stale=1
            echo "[wandb] stale publish claim blocks $training_project/$experiment_name:" >&2
            print_stale_claim_recovery "$claim_output"
        fi
        sleep "$EVAL_CLAIM_POLL_SECONDS"
    done
    echo "[wandb] timed out waiting ${EVAL_WANDB_LOCK_WAIT_SECONDS}s for $training_project/$experiment_name" >&2
    return 1
}

register_checkpoint_attempt() {
    local step=$1
    python -c '
import json, os, sys
path = sys.argv[1]
os.makedirs(os.path.dirname(path), exist_ok=True)
try:
    payload = json.load(open(path, encoding="utf-8"))
except Exception:
    payload = {"attempt_count": 0}
payload["attempt_count"] = int(payload.get("attempt_count", 0)) + 1
tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
os.replace(tmp, path)
print(payload["attempt_count"])
' "$RESULT_ROOT/checkpoint_timings/global_step_$step.attempts.json"
}

evaluate_checkpoint() {
    local candidate_json=$1
    local step actor_dir direct_hf checkpoint
    step=$(python -c 'import json,sys; print(json.loads(sys.argv[1])["step"])' "$candidate_json")
    actor_dir=$(python -c 'import json,sys; print(json.loads(sys.argv[1])["actor_dir"])' "$candidate_json")
    direct_hf=$(python -c 'import json,sys; print(json.loads(sys.argv[1]).get("hf_checkpoint") or "")' "$candidate_json")
    local attempt_count
    attempt_count=$(register_checkpoint_attempt "$step")
    local checkpoint_started=$SECONDS
    local load_started=$SECONDS
    update_active_claim_phase "merge"
    if ! checkpoint=$(ensure_hf_checkpoint "$step" "$actor_dir" "$direct_hf"); then
        return 1
    fi
    local tokenizer_model=${STUDENT_TOKENIZER_MODEL:-$checkpoint}
    local served_model="dmpd-student-step-$step"
    local student_base_url="http://127.0.0.1:$STUDENT_PORT/v1"
    local benchmark
    for benchmark in "${benchmarks[@]}"; do
        update_active_claim_phase "preflight:$benchmark"
        if ! run_raw_unit "$step" "$checkpoint" "$benchmark" "$served_model" "$student_base_url" \
            "$tokenizer_model" 1
        then
            return 1
        fi
    done
    update_active_claim_phase "teacher:request"
    if ! request_teacher_endpoint; then
        return 1
    fi
    update_active_claim_phase "load_student"
    if ! start_vllm "student-step-$step" "$checkpoint" "$tokenizer_model" "$served_model" \
        "$STUDENT_BIND_HOST" "$STUDENT_PORT" "$STUDENT_TP_SIZE" "$STUDENT_DP_SIZE" \
        "$STUDENT_MAX_NUM_SEQS" "$STUDENT_GPU_MEMORY_UTILIZATION" "$STUDENT_DP_RPC_PORT" \
        "${STUDENT_API_KEY:-EMPTY}"
    then
        return 1
    fi
    if ! wait_for_model "$student_base_url" "${STUDENT_API_KEY:-EMPTY}" "$served_model"; then
        return 1
    fi
    local load_seconds=$((SECONDS - load_started))
    for benchmark in "${benchmarks[@]}"; do
        update_active_claim_phase "raw:$benchmark"
        if ! run_raw_unit "$step" "$checkpoint" "$benchmark" "$served_model" "$student_base_url" \
            "$tokenizer_model"
        then
            return 1
        fi
        update_active_claim_phase "repair:$benchmark"
        if ! run_repair_unit "$step" "$benchmark" "$served_model" "$student_base_url" "$tokenizer_model"; then
            return 1
        fi
    done
    if ! stop_vllm; then
        return 1
    fi
    if ! python -c '
import json, os, sys, time
path = sys.argv[1]
payload = {
    "global_step": int(sys.argv[2]),
    "model_load_seconds": float(sys.argv[3]),
    "checkpoint_total_seconds": float(sys.argv[4]),
    "failure_retry_count": max(int(sys.argv[5]) - 1, 0),
    "teacher_model_load_seconds": float(sys.argv[6]),
    "completed_at_unix": time.time(),
}
os.makedirs(os.path.dirname(path), exist_ok=True)
tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
os.replace(tmp, path)
' "$RESULT_ROOT/checkpoint_timings/global_step_$step.json" "$step" "$load_seconds" \
        "$((SECONDS - checkpoint_started))" "$attempt_count" "${teacher_model_load_seconds:-0}"
    then
        return 1
    fi
    if ! sync_wandb_serialized "$step"; then
        return 1
    fi
    if [[ "$KEEP_MERGED_CHECKPOINTS" != "1" && -z "$direct_hf" ]]; then
        rm -rf -- "$EVAL_CACHE_ROOT/global_step_$step"
    fi
    echo "[watcher] completed global_step_$step"
}

array_contains() {
    local expected=$1
    shift
    local value
    for value in "$@"; do
        if [[ "$value" == "$expected" ]]; then
            return 0
        fi
    done
    return 1
}

candidate_key() {
    local candidate_json=$1
    python -c '
import json, sys
candidate = json.loads(sys.argv[1])
print("{}\x1f{}".format(candidate["checkpoint_root"], candidate["step"]))
' "$candidate_json"
}

print_queue_snapshot() {
    local force=${1:-0}
    local -a flags=(
        --runtime-root "$shared_runtime_root"
        --instance-id "$EVAL_INSTANCE_ID"
        --stale-after-seconds "$EVAL_CLAIM_STALE_SECONDS"
        --max-rows "$EVAL_QUEUE_MAX_ROWS"
        --include-signature
    )
    if [[ -n "$ACTIVE_CLAIM_PATH" ]]; then
        flags+=(--current-claim-path "$ACTIVE_CLAIM_PATH")
    fi
    local snapshot signature table
    if ! snapshot=$(printf '%s\n' "${QUEUE_CANDIDATES[@]}" | \
        python -m verl.utils.evaluation.checkpoint_watcher render-queue "${flags[@]}")
    then
        echo "[queue] failed to render checkpoint queue" >&2
        return 1
    fi
    signature=${snapshot%%$'\n'*}
    signature=${signature#__QUEUE_SIGNATURE__=}
    table=${snapshot#*$'\n'}
    if [[ "$force" == "1" || "$signature" != "$LAST_QUEUE_SIGNATURE" ]]; then
        echo "$table"
        LAST_QUEUE_SIGNATURE=$signature
    fi
}

print_stale_claim_recovery() {
    local claim_output=$1
    local claim_path claim_token owner_missing runtime_root_quoted claim_path_quoted claim_token_quoted
    claim_path=$(python -c 'import json,sys; print(json.loads(sys.argv[1])["claim_path"])' "$claim_output")
    if array_contains "$claim_path" "${REPORTED_STALE_CLAIM_PATHS[@]}"; then
        return 0
    fi
    REPORTED_STALE_CLAIM_PATHS+=("$claim_path")
    claim_token=$(python -c '
import json, sys
owner = json.loads(sys.argv[1]).get("owner") or {}
print(owner.get("claim_token") or "")
' "$claim_output")
    owner_missing=$(python -c '
import json, sys
print(str(json.loads(sys.argv[1]).get("owner") is None).lower())
' "$claim_output")
    printf -v runtime_root_quoted '%q' "$shared_runtime_root"
    printf -v claim_path_quoted '%q' "$claim_path"
    printf -v claim_token_quoted '%q' "$claim_token"
    echo "[claim] stale claim blocks a managed resource: $claim_path" >&2
    echo "Confirm its scheduler task has exited, then release it explicitly:" >&2
    if [[ "$owner_missing" == "true" ]]; then
        echo "  python -m verl.utils.evaluation.checkpoint_watcher release-stale-claim \
--runtime-root $runtime_root_quoted --claim-path $claim_path_quoted \
--stale-after-seconds $EVAL_CLAIM_STALE_SECONDS --allow-ownerless" >&2
    else
        echo "  python -m verl.utils.evaluation.checkpoint_watcher release-stale-claim \
--runtime-root $runtime_root_quoted --claim-path $claim_path_quoted --claim-token $claim_token_quoted \
--stale-after-seconds $EVAL_CLAIM_STALE_SECONDS" >&2
    fi
}

try_acquire_checkpoint_claim() {
    local candidate_json=$1
    local checkpoint_root step candidate_experiment candidate_project candidate_fingerprint
    checkpoint_root=$(python -c 'import json,sys; print(json.loads(sys.argv[1])["checkpoint_root"])' "$candidate_json")
    step=$(python -c 'import json,sys; print(json.loads(sys.argv[1])["step"])' "$candidate_json")
    candidate_experiment=$(python -c \
        'import json,sys; print(json.loads(sys.argv[1])["experiment_name"])' "$candidate_json")
    candidate_project=$(python -c \
        'import json,sys; print(json.loads(sys.argv[1])["training_project"])' "$candidate_json")
    candidate_fingerprint=$(python -c \
        'import json,sys; print(json.loads(sys.argv[1])["spec_fingerprint"])' "$candidate_json")

    local claim_output claim_status
    claim_output=$(python -m verl.utils.evaluation.checkpoint_watcher acquire-claim \
        --runtime-root "$shared_runtime_root" \
        --claim-type checkpoint \
        --instance-id "$EVAL_INSTANCE_ID" \
        --owner-pid "$$" \
        --phase "claimed" \
        --checkpoint-root "$checkpoint_root" \
        --global-step "$step" \
        --experiment-name "$candidate_experiment" \
        --training-project "$candidate_project" \
        --spec-fingerprint "$candidate_fingerprint" \
        --stale-after-seconds "$EVAL_CLAIM_STALE_SECONDS")
    claim_status=$?
    if [[ "$claim_status" == "0" ]]; then
        ACTIVE_CLAIM_PATH=$(claim_json_value "$claim_output" claim_path)
        ACTIVE_CLAIM_TOKEN=$(claim_json_value "$claim_output" claim_token)
        ACTIVE_CLAIM_LABEL=$candidate_experiment/global_step_$step
        start_claim_heartbeat "$ACTIVE_CLAIM_PATH" "$ACTIVE_CLAIM_TOKEN"
        ACTIVE_CLAIM_HEARTBEAT_PID=$STARTED_HEARTBEAT_PID
        return 0
    fi
    if [[ "$claim_status" == "3" ]]; then
        local claim_state
        claim_state=$(python -c 'import json,sys; print(json.loads(sys.argv[1]).get("state", "UNKNOWN"))' \
            "$claim_output")
        if [[ "$claim_state" == "STALE" ]]; then
            print_stale_claim_recovery "$claim_output"
        fi
        return 3
    fi
    echo "[claim] failed to acquire $candidate_experiment/global_step_$step: $claim_output" >&2
    return 1
}

controller_role_main() {
    rm -f -- "$teacher_stop_file" "$teacher_request_file"
    while true; do
        local candidate_output
        echo "[watcher] discovering committed checkpoints under $CHECKPOINT_SEARCH_ROOT"
        if ! candidate_output=$(python -m verl.utils.evaluation.checkpoint_watcher discover-tree \
            --search-root "$CHECKPOINT_SEARCH_ROOT"); then
            return 1
        fi
        local -a candidates=()
        if [[ -n "$candidate_output" ]]; then
            mapfile -t candidates <<< "$candidate_output"
        fi
        echo "[watcher] discovered ${#candidates[@]} committed checkpoint(s); checking result state"

        local -a pending_candidates=()
        local candidate scan_index=0
        for candidate in "${candidates[@]}"; do
            local step candidate_experiment key
            scan_index=$((scan_index + 1))
            step=$(python -c 'import json,sys; print(json.loads(sys.argv[1])["step"])' "$candidate")
            candidate_experiment=$(python -c \
                'import json,sys; print(json.loads(sys.argv[1])["experiment_name"])' "$candidate")
            if ! step_is_selected "$step" || ! experiment_is_selected "$candidate_experiment"; then
                continue
            fi
            key=$(candidate_key "$candidate")
            if [[ "$WATCH_ONCE" == "1" ]] && array_contains "$key" "${WATCH_ONCE_ATTEMPTED_KEYS[@]}"; then
                continue
            fi
            echo "[watcher] scan $scan_index/${#candidates[@]}: $candidate_experiment/global_step_$step"
            if ! load_queue_context "$candidate"; then
                echo "[watcher] failed to load queue context for $candidate_experiment/global_step_$step" >&2
                if [[ "$CONTINUE_ON_EVAL_ERROR" != "1" ]]; then
                    return 1
                fi
                continue
            fi
            if checkpoint_complete "$step" && checkpoint_wandb_is_staged "$step"; then
                continue
            fi
            pending_candidates+=("$candidate")
        done
        echo "[watcher] result-state scan complete: ${#pending_candidates[@]} checkpoint(s) remain"
        QUEUE_CANDIDATES=("${pending_candidates[@]}")

        local selected_candidate=""
        for candidate in "${pending_candidates[@]}"; do
            try_acquire_checkpoint_claim "$candidate"
            local claim_status=$?
            if [[ "$claim_status" == "0" ]]; then
                selected_candidate=$candidate
                if [[ "$WATCH_ONCE" == "1" ]]; then
                    WATCH_ONCE_ATTEMPTED_KEYS+=("$(candidate_key "$candidate")")
                fi
                break
            fi
            if [[ "$claim_status" != "3" && "$CONTINUE_ON_EVAL_ERROR" != "1" ]]; then
                return 1
            fi
        done
        print_queue_snapshot 0

        if [[ -z "$selected_candidate" ]]; then
            if [[ "$WATCH_ONCE" == "1" ]]; then
                echo "[watcher] no unclaimed checkpoint remains for this WATCH_ONCE worker"
                break
            fi
            if ((${#pending_candidates[@]} == 0)); then
                echo "[watcher] no pending committed checkpoint; sleeping ${POLL_SECONDS}s"
                sleep "$POLL_SECONDS"
            else
                echo "[watcher] all pending checkpoints are claimed or stale; sleeping ${EVAL_CLAIM_POLL_SECONDS}s"
                sleep "$EVAL_CLAIM_POLL_SECONDS"
            fi
            continue
        fi

        local step candidate_experiment work_failed=0
        step=$(python -c 'import json,sys; print(json.loads(sys.argv[1])["step"])' "$selected_candidate")
        candidate_experiment=$(python -c \
            'import json,sys; print(json.loads(sys.argv[1])["experiment_name"])' "$selected_candidate")
        if ! load_experiment_context "$selected_candidate"; then
            echo "[watcher] failed to reload evaluation context for $candidate_experiment/global_step_$step" >&2
            work_failed=1
        elif checkpoint_complete "$step"; then
            if checkpoint_wandb_is_staged "$step"; then
                echo "[watcher] $experiment_name/global_step_$step was completed by another worker before recheck"
            else
                echo "[watcher] staging recovered W&B results for $experiment_name/global_step_$step"
                if ! sync_wandb_serialized "$step"; then
                    echo "[watcher] W&B staging failed for $experiment_name/global_step_$step" >&2
                    work_failed=1
                fi
            fi
        else
            echo "[watcher] evaluating experiment=$experiment_name step=$step root=$CKPT_ROOT"
            if ! evaluate_checkpoint "$selected_candidate"; then
                stop_vllm || true
                echo "[watcher] $experiment_name/global_step_$step failed; resumable shards were retained." >&2
                work_failed=1
            fi
        fi

        if [[ "$work_failed" == "1" ]]; then
            if [[ "$CONTINUE_ON_EVAL_ERROR" != "1" ]]; then
                release_active_claim
                return 1
            fi
            echo "[watcher] retaining the checkpoint claim during the ${RETRY_SECONDS}s retry backoff"
            sleep "$RETRY_SECONDS"
        fi
        release_active_claim
        echo "[watcher] released $candidate_experiment/global_step_$step; rescanning immediately"
    done
    touch "$teacher_stop_file"
}

echo "================================================================================"
echo "Standalone checkpoint evaluation"
echo "  role:             $EVAL_ROLE"
echo "  instance:         $EVAL_INSTANCE_ID"
echo "  instance runtime: $instance_runtime_root"
echo "  checkpoint search: $CHECKPOINT_SEARCH_ROOT"
echo "  result root:       $EVAL_RESULTS_ROOT"
echo "  W&B root:          $EVAL_WANDB_ROOT"
echo "  setup mode:        $EVAL_SETUP_MODE"
if [[ "$WANDB_MODE" == "offline" ]]; then
    echo "  offline sync:      run the wandb-offline-sync.sh files below $EVAL_RESULTS_ROOT"
fi
echo "  experiment filter: ${EVAL_EXPERIMENTS_TEXT:-all}"
echo "  step filter:       ${EVAL_STEPS_TEXT:-all}"
echo "  claim heartbeat:   ${EVAL_CLAIM_HEARTBEAT_SECONDS}s"
echo "  claim stale after: ${EVAL_CLAIM_STALE_SECONDS}s (manual release only)"
echo "================================================================================"

cd "$REPO_ROOT"
if ! register_role_claim; then
    exit 2
fi
if [[ "$EVAL_ROLE" == "teacher" ]]; then
    teacher_role_main
else
    controller_role_main
fi
