#!/usr/bin/env bash
# Standalone NPU launcher for benchmark-level OPD diagnostics.

# ==============================================================================
# Machine/bootstrap environment
# ==============================================================================

_load_external_train_config() {
    local train_config_path=${TRAIN_CONFIG_PATH:-/opt/huawei/algorithm/train.config}
    if [[ ! -s "${train_config_path}" ]]; then
        echo "No non-empty train config found at ${train_config_path}; skip external env loading."
        return 0
    fi

    echo "Loading external train config from ${train_config_path}"
    local allexport_was_set=0
    case "$-" in
        *a*) allexport_was_set=1 ;;
    esac
    set -o allexport
    # shellcheck disable=SC1090
    . "${train_config_path}"
    if [[ "${allexport_was_set}" -eq 0 ]]; then
        set +o allexport
    fi
}

HOME=/home/ma-user
DATASET_DIR=/opt/huawei/dataset/jnguan
WORKSPACE=$DATASET_DIR/code/DMPD

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

export PYTHONDONTWRITEBYTECODE=1

# ==============================================================================
# Ascend/vLLM environment
# ==============================================================================

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


# ==============================================================================
# Experiment and model configuration
# ==============================================================================

EVAL_MODULE=${EVAL_MODULE:-verl.utils.evaluation.opd_benchmark}
REPO_ROOT=${REPO_ROOT:-${WORKSPACE}}
REPO_ROOT=$(cd -- "${REPO_ROOT}" && pwd)
EXP_NAME=${EXP_NAME:-vanilla-opd-20260716-122946}
CKPT_ROOT=${CKPT_ROOT:-${REPO_ROOT}/outputs/${EXP_NAME}}

STEPS_TEXT=${STEPS_TEXT:-"20 40 60 80 100 120 140 160 180 200 217"}
read -r -a STEPS <<< "${STEPS_TEXT}"

DATASET_ROOT=${DATASET_ROOT:-${DATASET_DIR}/datasets}
TEACHER_MODEL=${TEACHER_MODEL:-${DATASET_DIR}/models/hbx/JustRL-DeepSeek-1.5B}
CHAT_TEMPLATE_MODEL=${CHAT_TEMPLATE_MODEL:-${TEACHER_MODEL}}

BENCHMARKS_TEXT=${BENCHMARKS_TEXT:-"math-ai/aime24 math-ai/aime25 math-ai/amc23 meituan-longcat/AMO-Bench"}
read -r -a BENCHMARKS <<< "${BENCHMARKS_TEXT}"

RESULT_ROOT=${RESULT_ROOT:-${WORKSPACE}/eval-results/${EXP_NAME}}
COMPARE_RESULT_ROOT=${COMPARE_RESULT_ROOT:-}
COMPARE_OUTPUT_DIR=${COMPARE_OUTPUT_DIR:-${RESULT_ROOT}/comparison}


# ==============================================================================
# Generation and diagnostic configuration
# ==============================================================================

ROLLOUT_N=${ROLLOUT_N:-16}
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-0.95}
TOP_K=${TOP_K:--1}
MAX_PROMPT_TOKENS=${MAX_PROMPT_TOKENS:-1024}
MAX_COMPLETION_TOKENS=${MAX_COMPLETION_TOKENS:-$((23 * 1024))}
BASE_SEED=${BASE_SEED:-42}
VERIFIER=${VERIFIER:-math_verify}
ENABLE_THINKING=${ENABLE_THINKING:-1}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-0}
REQUIRE_EXACT_TOKEN_IDS=${REQUIRE_EXACT_TOKEN_IDS:-1}
MAX_SAMPLES_PER_BENCHMARK=${MAX_SAMPLES_PER_BENCHMARK:--1}

LENGTH_BIN_EDGES_TEXT=${LENGTH_BIN_EDGES_TEXT:-"0 1024 2048 4096 8192 12000 16000 20000 23000"}
read -r -a LENGTH_BIN_EDGES <<< "${LENGTH_BIN_EDGES_TEXT}"
LENGTH_BIN_EDGES_JSON=$(IFS=,; echo "[${LENGTH_BIN_EDGES[*]}]")

REPETITION_TAIL_CHARS=${REPETITION_TAIL_CHARS:-10000}
REPETITION_TAIL_TOKEN_WINDOW=${REPETITION_TAIL_TOKEN_WINDOW:-4096}
REPETITION_THRESHOLD=${REPETITION_THRESHOLD:-10.0}
NGRAM_SIZE=${NGRAM_SIZE:-4}
HIGH_GAP_THRESHOLD=${HIGH_GAP_THRESHOLD:-1.0}
BOOTSTRAP_RESAMPLES=${BOOTSTRAP_RESAMPLES:-5000}
COMPARE_BOOTSTRAP_RESAMPLES=${COMPARE_BOOTSTRAP_RESAMPLES:-10000}
BOOTSTRAP_SEED=${BOOTSTRAP_SEED:-20260722}

ENABLE_TEACHER_SCORING=${ENABLE_TEACHER_SCORING:-0}
ALLOW_TOKENIZER_MISMATCH=${ALLOW_TOKENIZER_MISMATCH:-0}


# ==============================================================================
# vLLM/client configuration
# ==============================================================================

HTTP_HOST=${HTTP_HOST:-127.0.0.1}
HTTP_PORT=${HTTP_PORT:-8000}
BASE_URL="http://${HTTP_HOST}:${HTTP_PORT}/v1"
API_KEY=${API_KEY:-EMPTY}

STUDENT_TP_SIZE=${STUDENT_TP_SIZE:-1}
STUDENT_DP_SIZE=${STUDENT_DP_SIZE:-8}
STUDENT_MAX_CONCURRENCY_PER_DP=${STUDENT_MAX_CONCURRENCY_PER_DP:-64}
STUDENT_DP_RPC_PORT=${STUDENT_DP_RPC_PORT:-13389}
STUDENT_MAX_PENDING=${STUDENT_MAX_PENDING:-}

TEACHER_TP_SIZE=${TEACHER_TP_SIZE:-1}
TEACHER_DP_SIZE=${TEACHER_DP_SIZE:-8}
TEACHER_MAX_CONCURRENCY_PER_DP=${TEACHER_MAX_CONCURRENCY_PER_DP:-64}
TEACHER_DP_RPC_PORT=${TEACHER_DP_RPC_PORT:-13389}

STUDENT_SCORE_MAX_IN_FLIGHT=${STUDENT_SCORE_MAX_IN_FLIGHT:-128}
TEACHER_SCORE_MAX_IN_FLIGHT=${TEACHER_SCORE_MAX_IN_FLIGHT:-128}
TEACHER_SCORE_BATCH_SIZE=${TEACHER_SCORE_BATCH_SIZE:-256}

MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
SERVER_READY_TIMEOUT=${SERVER_READY_TIMEOUT:-3600}
SERVER_STOP_TIMEOUT=${SERVER_STOP_TIMEOUT:-180}
NPU_RELEASE_WAIT=${NPU_RELEASE_WAIT:-10}
REQUEST_TIMEOUT_SECONDS=${REQUEST_TIMEOUT_SECONDS:-7200}
REQUEST_MAX_RETRIES=${REQUEST_MAX_RETRIES:-2}
PRINT_EVERY=${PRINT_EVERY:-50}


# ==============================================================================
# Workflow configuration
# ==============================================================================

MERGE_MISSING_CHECKPOINTS=${MERGE_MISSING_CHECKPOINTS:-1}
CONTINUE_ON_EVAL_ERROR=${CONTINUE_ON_EVAL_ERROR:-1}
SKIP_COMPLETED=${SKIP_COMPLETED:-1}
OVERWRITE_RESULTS=${OVERWRITE_RESULTS:-0}
KEEP_RUNTIME_LOGS=${KEEP_RUNTIME_LOGS:-0}
MIGRATE_LEGACY_RESULTS=${MIGRATE_LEGACY_RESULTS:-1}
ACCEPT_MIGRATED_LEGACY_RESULTS=${ACCEPT_MIGRATED_LEGACY_RESULTS:-1}
COMPARE_ALLOW_MISSING_UNITS=${COMPARE_ALLOW_MISSING_UNITS:-0}

cd "${REPO_ROOT}"
mkdir -p "${RESULT_ROOT}"

command -v setsid >/dev/null 2>&1 || {
    echo "ERROR: setsid is required" >&2
    exit 1
}

RUNTIME_DIR=$(mktemp -d "${RESULT_ROOT}/runtime.XXXXXX")
VLLM_PID=""
VLLM_PGID=""
CURRENT_VLLM_LOG=""
CURRENT_SERVER_ROLE=""

declare -a PENDING_STEPS=()
declare -A PENDING_BENCHMARKS=()
declare -a FAILED_UNITS=()


# ==============================================================================
# Checkpoint handling
# ==============================================================================

hf_checkpoint_ready() {
    local hf_dir=$1
    python -m "${EVAL_MODULE}" checkpoint-ready --path "${hf_dir}" >/dev/null 2>&1
}


ensure_hf_checkpoint() {
    local step=$1
    local actor_dir="${CKPT_ROOT}/global_step_${step}/actor"
    local hf_dir="${actor_dir}/huggingface"
    if hf_checkpoint_ready "${hf_dir}"; then
        echo "[checkpoint] step=${step}: Hugging Face checkpoint ready"
        return 0
    fi
    if [[ "${MERGE_MISSING_CHECKPOINTS}" != "1" ]]; then
        echo "ERROR: missing or incomplete Hugging Face checkpoint: ${hf_dir}" >&2
        return 1
    fi
    if [[ ! -d "${actor_dir}" ]]; then
        echo "ERROR: actor checkpoint does not exist: ${actor_dir}" >&2
        return 1
    fi

    local merge_dir="${actor_dir}/.huggingface.merge.${BASHPID}"
    if [[ -e "${merge_dir}" ]]; then
        echo "ERROR: temporary merge target already exists: ${merge_dir}" >&2
        return 1
    fi
    echo "[checkpoint] converting step=${step}: ${actor_dir} -> ${hf_dir}"
    if ! python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "${actor_dir}" \
        --target_dir "${merge_dir}"
    then
        echo "ERROR: checkpoint conversion failed; partial output retained at ${merge_dir}" >&2
        return 1
    fi
    if ! hf_checkpoint_ready "${merge_dir}"; then
        echo "ERROR: converted checkpoint is incomplete; retained at ${merge_dir}" >&2
        return 1
    fi
    if [[ -e "${hf_dir}" ]]; then
        local backup_dir="${hf_dir}.incomplete.$(date +%Y%m%d-%H%M%S)"
        mv -- "${hf_dir}" "${backup_dir}"
        echo "[checkpoint] moved incomplete checkpoint to ${backup_dir}"
    fi
    mv -- "${merge_dir}" "${hf_dir}"
    hf_checkpoint_ready "${hf_dir}"
}


# ==============================================================================
# Port/process helpers and vLLM lifecycle
# ==============================================================================

port_is_open() {
    python -c '
import socket
import sys
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(1.0)
result = sock.connect_ex((sys.argv[1], int(sys.argv[2])))
sock.close()
raise SystemExit(0 if result == 0 else 1)
' "${HTTP_HOST}" "${HTTP_PORT}" >/dev/null 2>&1
}


process_group_has_live_members() {
    local pgid=$1
    ps -eo pgid=,stat= | awk -v target="${pgid}" '
        $1 == target && $2 !~ /^Z/ { found = 1 }
        END { exit(found ? 0 : 1) }
    '
}


print_vllm_log_tail() {
    if [[ -n "${CURRENT_VLLM_LOG}" && -f "${CURRENT_VLLM_LOG}" ]]; then
        echo "---------------- ${CURRENT_SERVER_ROLE} vLLM log tail ----------------" >&2
        tail -n 200 "${CURRENT_VLLM_LOG}" >&2
        echo "-----------------------------------------------------------------------" >&2
    fi
}


start_vllm() {
    local role=$1
    local model_path=$2
    local tokenizer_path=$3
    local served_model_name=$4
    local tp_size=$5
    local dp_size=$6
    local max_num_seqs=$7
    local dp_rpc_port=$8

    if port_is_open; then
        echo "ERROR: ${HTTP_HOST}:${HTTP_PORT} is already occupied" >&2
        return 1
    fi

    CURRENT_SERVER_ROLE=${role}
    CURRENT_VLLM_LOG="${RUNTIME_DIR}/vllm_${role}.log"
    echo "================================================================================"
    echo "[vllm] starting role=${role} model=${served_model_name}"
    echo "[vllm] path=${model_path} tp=${tp_size} dp=${dp_size}"
    echo "================================================================================"

    setsid vllm serve \
        "${model_path}" \
        --served-model-name "${served_model_name}" \
        --tokenizer "${tokenizer_path}" \
        --host "${HTTP_HOST}" \
        --port "${HTTP_PORT}" \
        --api-key "${API_KEY}" \
        --tensor-parallel-size "${tp_size}" \
        --data-parallel-size "${dp_size}" \
        --data-parallel-size-local "${dp_size}" \
        --data-parallel-address 127.0.0.1 \
        --data-parallel-rpc-port "${dp_rpc_port}" \
        --data-parallel-start-rank 0 \
        --max-num-seqs "${max_num_seqs}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --enable-prefix-caching \
        --generation-config vllm \
        </dev/null >"${CURRENT_VLLM_LOG}" 2>&1 &

    VLLM_PID=$!
    VLLM_PGID=${VLLM_PID}
    echo "[vllm] role=${role} pid=${VLLM_PID} pgid=${VLLM_PGID}"
}


wait_for_vllm() {
    local served_model_name=$1
    local deadline=$((SECONDS + SERVER_READY_TIMEOUT))
    local next_progress_log=${SECONDS}
    while ((SECONDS < deadline)); do
        if [[ -z "${VLLM_PID}" ]] || ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            echo "ERROR: vLLM exited before readiness: ${served_model_name}" >&2
            return 1
        fi
        if curl \
            --silent \
            --show-error \
            --fail \
            --connect-timeout 2 \
            --max-time 5 \
            --header "Authorization: Bearer ${API_KEY}" \
            "${BASE_URL}/models" 2>/dev/null | \
            python -c '
import json
import sys
expected = sys.argv[1]
payload = json.load(sys.stdin)
raise SystemExit(0 if expected in {item.get("id") for item in payload.get("data", [])} else 1)
' "${served_model_name}" >/dev/null 2>&1
        then
            echo "[vllm] API ready: ${served_model_name}"
            return 0
        fi
        if ((SECONDS >= next_progress_log)); then
            echo "[vllm] still loading ${served_model_name} ..."
            next_progress_log=$((SECONDS + 60))
        fi
        sleep 5
    done
    echo "ERROR: timed out waiting for ${served_model_name}" >&2
    return 1
}


stop_vllm() {
    if [[ -z "${VLLM_PID}" ]]; then
        return 0
    fi
    local pid=${VLLM_PID}
    local pgid=${VLLM_PGID}
    local deadline=$((SECONDS + SERVER_STOP_TIMEOUT))
    echo "[vllm] stopping role=${CURRENT_SERVER_ROLE} pid=${pid} pgid=${pgid}"
    if process_group_has_live_members "${pgid}"; then
        kill -TERM -- "-${pgid}" 2>/dev/null || true
    fi
    while process_group_has_live_members "${pgid}"; do
        if ((SECONDS >= deadline)); then
            echo "[vllm] graceful shutdown timed out; killing remaining group" >&2
            kill -KILL -- "-${pgid}" 2>/dev/null || true
            break
        fi
        sleep 2
    done
    wait "${pid}" 2>/dev/null || true
    VLLM_PID=""
    VLLM_PGID=""
    CURRENT_SERVER_ROLE=""

    local port_deadline=$((SECONDS + 30))
    while port_is_open; do
        if ((SECONDS >= port_deadline)); then
            echo "ERROR: port ${HTTP_PORT} remains open after vLLM stop" >&2
            return 1
        fi
        sleep 1
    done
    echo "[vllm] stopped; waiting ${NPU_RELEASE_WAIT}s for NPU/HCCL cleanup"
    sleep "${NPU_RELEASE_WAIT}"
}


cleanup() {
    local exit_code=$?
    trap - EXIT
    set +e
    stop_vllm
    local stop_status=$?
    if [[ "${KEEP_RUNTIME_LOGS}" == "1" ]]; then
        echo "[logs] retained at ${RUNTIME_DIR}"
    elif [[ -d "${RUNTIME_DIR}" ]]; then
        rm -rf -- "${RUNTIME_DIR}"
    fi
    if ((exit_code == 0 && stop_status != 0)); then
        exit_code=${stop_status}
    fi
    exit "${exit_code}"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM


# ==============================================================================
# Reusable flags and status checks
# ==============================================================================

TOKENIZER_FLAGS=()
if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
    TOKENIZER_FLAGS+=(--trust-remote-code)
fi

THINKING_FLAGS=(--enable-thinking)
THINKING_JSON=true
if [[ "${ENABLE_THINKING}" != "1" ]]; then
    THINKING_FLAGS=(--no-enable-thinking)
    THINKING_JSON=false
fi

EXACT_TOKEN_FLAGS=(--require-exact-token-ids)
EXACT_TOKEN_JSON=true
if [[ "${REQUIRE_EXACT_TOKEN_IDS}" != "1" ]]; then
    EXACT_TOKEN_FLAGS=(--no-require-exact-token-ids)
    EXACT_TOKEN_JSON=false
fi

OVERWRITE_FLAGS=()
if [[ "${OVERWRITE_RESULTS}" == "1" ]]; then
    OVERWRITE_FLAGS+=(--overwrite)
fi

MISMATCH_FLAGS=()
if [[ "${ALLOW_TOKENIZER_MISMATCH}" == "1" ]]; then
    MISMATCH_FLAGS+=(--allow-tokenizer-mismatch)
fi

STUDENT_LOGPROB_FLAGS=(--no-score-student-logprobs)
STUDENT_LOGPROB_JSON=false
if [[ "${ENABLE_TEACHER_SCORING}" == "1" ]]; then
    STUDENT_LOGPROB_FLAGS=(--score-student-logprobs)
    STUDENT_LOGPROB_JSON=true
fi

MAX_PENDING_FLAGS=()
if [[ -n "${STUDENT_MAX_PENDING}" ]]; then
    MAX_PENDING_FLAGS+=(--max-pending "${STUDENT_MAX_PENDING}")
fi

LEGACY_STATUS_FLAGS=()
if [[ "${ACCEPT_MIGRATED_LEGACY_RESULTS}" == "1" ]]; then
    LEGACY_STATUS_FLAGS+=(--allow-legacy)
fi


unit_status() {
    local step=$1
    local benchmark=$2
    local checkpoint_path="${CKPT_ROOT}/global_step_${step}/actor/huggingface"
    python -m "${EVAL_MODULE}" status \
        --result-root "${RESULT_ROOT}" \
        --benchmark "${benchmark}" \
        --global-step "${step}" \
        --expect "experiment=${EXP_NAME}" \
        --expect "global_step=${step}" \
        --expect "benchmark=${benchmark}" \
        --expect "dataset_root=${DATASET_ROOT}" \
        --expect "student_model_path=${checkpoint_path}" \
        --expect "student_tokenizer_model=${checkpoint_path}" \
        --expect "chat_template_model=${CHAT_TEMPLATE_MODEL}" \
        --expect "rollout_n=${ROLLOUT_N}" \
        --expect "temperature=${TEMPERATURE}" \
        --expect "top_p=${TOP_P}" \
        --expect "top_k=${TOP_K}" \
        --expect "max_tokens=${MAX_COMPLETION_TOKENS}" \
        --expect "max_prompt_tokens=${MAX_PROMPT_TOKENS}" \
        --expect "base_seed=${BASE_SEED}" \
        --expect "enable_thinking=${THINKING_JSON}" \
        --expect "verifier=${VERIFIER}" \
        --expect "score_student_logprobs=${STUDENT_LOGPROB_JSON}" \
        --expect "require_exact_token_ids=${EXACT_TOKEN_JSON}" \
        --expect "max_samples_per_benchmark=${MAX_SAMPLES_PER_BENCHMARK}" \
        --expect "analysis.length_bin_edges=${LENGTH_BIN_EDGES_JSON}" \
        --expect "analysis.repetition_tail_chars=${REPETITION_TAIL_CHARS}" \
        --expect "analysis.repetition_tail_token_window=${REPETITION_TAIL_TOKEN_WINDOW}" \
        --expect "analysis.repetition_threshold=${REPETITION_THRESHOLD}" \
        --expect "analysis.ngram_size=${NGRAM_SIZE}" \
        --expect "analysis.high_gap_threshold=${HIGH_GAP_THRESHOLD}" \
        --expect "analysis.bootstrap_resamples=${BOOTSTRAP_RESAMPLES}" \
        --expect "analysis.bootstrap_seed=${BOOTSTRAP_SEED}" \
        "${LEGACY_STATUS_FLAGS[@]}"
}


teacher_status() {
    local step=$1
    local benchmark=$2
    python -m "${EVAL_MODULE}" status \
        --result-root "${RESULT_ROOT}" \
        --benchmark "${benchmark}" \
        --global-step "${step}" \
        --require-teacher
}


mark_failed() {
    FAILED_UNITS+=("$1")
}


# ==============================================================================
# Phase 0: migrate legacy mixed-benchmark outputs without deleting them
# ==============================================================================

if [[ "${MIGRATE_LEGACY_RESULTS}" == "1" ]]; then
    python -m "${EVAL_MODULE}" migrate-legacy --result-root "${RESULT_ROOT}"
fi


# ==============================================================================
# Phase 1: plan pending benchmark-step units, then verify only needed checkpoints
# ==============================================================================

echo "================================================================================"
echo "Planning benchmark units for ${EXP_NAME}"
echo "================================================================================"

for step in "${STEPS[@]}"; do
    pending=()
    for benchmark in "${BENCHMARKS[@]}"; do
        if [[ "${SKIP_COMPLETED}" == "1" && "${OVERWRITE_RESULTS}" != "1" ]] && \
            unit_status "${step}" "${benchmark}" >/dev/null 2>&1
        then
            echo "[plan] step=${step} benchmark=${benchmark}: complete"
        else
            pending+=("${benchmark}")
            echo "[plan] step=${step} benchmark=${benchmark}: pending"
        fi
    done
    if ((${#pending[@]} > 0)); then
        if ensure_hf_checkpoint "${step}"; then
            PENDING_STEPS+=("${step}")
            PENDING_BENCHMARKS["${step}"]="${pending[*]}"
        else
            for benchmark in "${pending[@]}"; do
                mark_failed "${step}/${benchmark}"
            done
            if [[ "${CONTINUE_ON_EVAL_ERROR}" != "1" ]]; then
                exit 1
            fi
        fi
    fi
done


# ==============================================================================
# Phase 2: load each pending student checkpoint once and run only missing benchmarks
# ==============================================================================

for step in "${PENDING_STEPS[@]}"; do
    checkpoint_path="${CKPT_ROOT}/global_step_${step}/actor/huggingface"
    served_model_name="${EXP_NAME}--global_step_${step}"
    read -r -a step_benchmarks <<< "${PENDING_BENCHMARKS[${step}]}"

    if ! start_vllm \
        "student_step_${step}" \
        "${checkpoint_path}" \
        "${checkpoint_path}" \
        "${served_model_name}" \
        "${STUDENT_TP_SIZE}" \
        "${STUDENT_DP_SIZE}" \
        "${STUDENT_MAX_CONCURRENCY_PER_DP}" \
        "${STUDENT_DP_RPC_PORT}" || \
        ! wait_for_vllm "${served_model_name}"
    then
        print_vllm_log_tail
        stop_vllm || true
        for benchmark in "${step_benchmarks[@]}"; do
            mark_failed "${step}/${benchmark}"
        done
        if [[ "${CONTINUE_ON_EVAL_ERROR}" == "1" ]]; then
            continue
        fi
        exit 1
    fi

    for benchmark in "${step_benchmarks[@]}"; do
        unit_dir=$(python -m "${EVAL_MODULE}" unit-path \
            --result-root "${RESULT_ROOT}" --benchmark "${benchmark}" --global-step "${step}")
        mkdir -p "${unit_dir}"
        echo "################################################################################"
        echo "# Generating ${EXP_NAME}, ${benchmark}, global_step_${step}"
        echo "################################################################################"

        set +e
        python -u -m "${EVAL_MODULE}" generate \
            --result-root "${RESULT_ROOT}" \
            --experiment "${EXP_NAME}" \
            --global-step "${step}" \
            --benchmark "${benchmark}" \
            --served-model-name "${served_model_name}" \
            --student-model-path "${checkpoint_path}" \
            --student-tokenizer-model "${checkpoint_path}" \
            --chat-template-model "${CHAT_TEMPLATE_MODEL}" \
            --dataset-root "${DATASET_ROOT}" \
            --base-url "${BASE_URL}" \
            --api-key "${API_KEY}" \
            --rollout-n "${ROLLOUT_N}" \
            --temperature "${TEMPERATURE}" \
            --top-p "${TOP_P}" \
            --top-k "${TOP_K}" \
            --max-prompt-tokens "${MAX_PROMPT_TOKENS}" \
            --max-tokens "${MAX_COMPLETION_TOKENS}" \
            --dp-size "${STUDENT_DP_SIZE}" \
            --max-concurrency-per-dp "${STUDENT_MAX_CONCURRENCY_PER_DP}" \
            --score-max-in-flight "${STUDENT_SCORE_MAX_IN_FLIGHT}" \
            --request-timeout-seconds "${REQUEST_TIMEOUT_SECONDS}" \
            --max-retries "${REQUEST_MAX_RETRIES}" \
            --max-samples-per-benchmark "${MAX_SAMPLES_PER_BENCHMARK}" \
            --base-seed "${BASE_SEED}" \
            --print-every "${PRINT_EVERY}" \
            --verifier "${VERIFIER}" \
            --length-bin-edges "${LENGTH_BIN_EDGES[@]}" \
            --repetition-tail-chars "${REPETITION_TAIL_CHARS}" \
            --repetition-tail-token-window "${REPETITION_TAIL_TOKEN_WINDOW}" \
            --repetition-threshold "${REPETITION_THRESHOLD}" \
            --ngram-size "${NGRAM_SIZE}" \
            --high-gap-threshold "${HIGH_GAP_THRESHOLD}" \
            --bootstrap-resamples "${BOOTSTRAP_RESAMPLES}" \
            --bootstrap-seed "${BOOTSTRAP_SEED}" \
            "${THINKING_FLAGS[@]}" \
            "${TOKENIZER_FLAGS[@]}" \
            "${STUDENT_LOGPROB_FLAGS[@]}" \
            "${EXACT_TOKEN_FLAGS[@]}" \
            "${MAX_PENDING_FLAGS[@]}" \
            "${OVERWRITE_FLAGS[@]}" \
            2>&1 | tee "${unit_dir}/generate.stdout.log"
        pipeline_status=("${PIPESTATUS[@]}")
        set -e
        eval_status=${pipeline_status[0]:-1}
        tee_status=${pipeline_status[1]:-1}
        if ((eval_status == 0 && tee_status != 0)); then
            eval_status=${tee_status}
        fi
        if ((eval_status != 0)); then
            mark_failed "${step}/${benchmark}"
            echo "ERROR: generation failed at step=${step}, benchmark=${benchmark}, status=${eval_status}" >&2
            if [[ "${CONTINUE_ON_EVAL_ERROR}" != "1" ]]; then
                stop_vllm || true
                exit "${eval_status}"
            fi
        fi
    done

    if ! stop_vllm; then
        print_vllm_log_tail
        exit 1
    fi
done


# ==============================================================================
# Phase 3: optionally load the fixed teacher once and score pending units
# ==============================================================================

if [[ "${ENABLE_TEACHER_SCORING}" == "1" ]]; then
    PENDING_TEACHER_UNITS=()
    for step in "${STEPS[@]}"; do
        for benchmark in "${BENCHMARKS[@]}"; do
            if unit_status "${step}" "${benchmark}" >/dev/null 2>&1 && \
                ! teacher_status "${step}" "${benchmark}" >/dev/null 2>&1
            then
                PENDING_TEACHER_UNITS+=("${step}|${benchmark}")
            fi
        done
    done

    if ((${#PENDING_TEACHER_UNITS[@]} > 0)); then
        teacher_served_model_name="${EXP_NAME}--diagnostic-teacher"
        if ! start_vllm \
            teacher \
            "${TEACHER_MODEL}" \
            "${TEACHER_MODEL}" \
            "${teacher_served_model_name}" \
            "${TEACHER_TP_SIZE}" \
            "${TEACHER_DP_SIZE}" \
            "${TEACHER_MAX_CONCURRENCY_PER_DP}" \
            "${TEACHER_DP_RPC_PORT}" || \
            ! wait_for_vllm "${teacher_served_model_name}"
        then
            print_vllm_log_tail
            stop_vllm || true
            exit 1
        fi

        for unit in "${PENDING_TEACHER_UNITS[@]}"; do
            step=${unit%%|*}
            benchmark=${unit#*|}
            unit_dir=$(python -m "${EVAL_MODULE}" unit-path \
                --result-root "${RESULT_ROOT}" --benchmark "${benchmark}" --global-step "${step}")
            set +e
            python -u -m "${EVAL_MODULE}" teacher-score \
                --result-root "${RESULT_ROOT}" \
                --global-step "${step}" \
                --benchmark "${benchmark}" \
                --served-model-name "${teacher_served_model_name}" \
                --teacher-model-path "${TEACHER_MODEL}" \
                --teacher-tokenizer-model "${TEACHER_MODEL}" \
                --base-url "${BASE_URL}" \
                --api-key "${API_KEY}" \
                --score-max-in-flight "${TEACHER_SCORE_MAX_IN_FLIGHT}" \
                --score-batch-size "${TEACHER_SCORE_BATCH_SIZE}" \
                --request-timeout-seconds "${REQUEST_TIMEOUT_SECONDS}" \
                --max-retries "${REQUEST_MAX_RETRIES}" \
                --print-every "${PRINT_EVERY}" \
                --repetition-tail-token-window "${REPETITION_TAIL_TOKEN_WINDOW}" \
                --high-gap-threshold "${HIGH_GAP_THRESHOLD}" \
                --bootstrap-resamples "${BOOTSTRAP_RESAMPLES}" \
                --bootstrap-seed "${BOOTSTRAP_SEED}" \
                "${TOKENIZER_FLAGS[@]}" \
                "${MISMATCH_FLAGS[@]}" \
                "${OVERWRITE_FLAGS[@]}" \
                2>&1 | tee "${unit_dir}/teacher-score.stdout.log"
            pipeline_status=("${PIPESTATUS[@]}")
            set -e
            eval_status=${pipeline_status[0]:-1}
            tee_status=${pipeline_status[1]:-1}
            if ((eval_status == 0 && tee_status != 0)); then
                eval_status=${tee_status}
            fi
            if ((eval_status != 0)); then
                mark_failed "teacher:${step}/${benchmark}"
                if [[ "${CONTINUE_ON_EVAL_ERROR}" != "1" ]]; then
                    stop_vllm || true
                    exit "${eval_status}"
                fi
            fi
        done
        stop_vllm
    fi
fi


# ==============================================================================
# Phase 4: dynamic report and optional paired comparison
# ==============================================================================

python -u -m "${EVAL_MODULE}" report --result-root "${RESULT_ROOT}"

if [[ -n "${COMPARE_RESULT_ROOT}" ]]; then
    if [[ ! -d "${COMPARE_RESULT_ROOT}" ]]; then
        echo "ERROR: COMPARE_RESULT_ROOT does not exist: ${COMPARE_RESULT_ROOT}" >&2
        exit 1
    fi
    COMPARE_FLAGS=()
    if [[ "${COMPARE_ALLOW_MISSING_UNITS}" == "1" ]]; then
        COMPARE_FLAGS+=(--allow-missing-units)
    fi
    python -u -m "${EVAL_MODULE}" compare \
        --left-root "${COMPARE_RESULT_ROOT}" \
        --right-root "${RESULT_ROOT}" \
        --output-dir "${COMPARE_OUTPUT_DIR}" \
        --benchmarks "${BENCHMARKS[@]}" \
        --bootstrap-resamples "${COMPARE_BOOTSTRAP_RESAMPLES}" \
        --bootstrap-seed "${BOOTSTRAP_SEED}" \
        "${COMPARE_FLAGS[@]}"
fi

echo "================================================================================"
echo "experiment=${EXP_NAME}"
echo "result_root=${RESULT_ROOT}"
echo "pending_steps=${PENDING_STEPS[*]:-}"
echo "failed_units=${FAILED_UNITS[*]:-}"
echo "================================================================================"

if ((${#FAILED_UNITS[@]} > 0)); then
    exit 1
fi
