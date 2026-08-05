#!/usr/bin/env bash
# Persistent two-role checkpoint evaluation for DMPD/OPD experiments.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
PYTHON=${PYTHON:-python3}
EVAL_MODULE=${EVAL_MODULE:-verl.utils.evaluation.opd_benchmark}
WATCHER_MODULE=${WATCHER_MODULE:-verl.utils.evaluation.checkpoint_watcher}

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

CKPT_ROOT=${CKPT_ROOT:?Set CKPT_ROOT to the training checkpoint root.}
CKPT_ROOT=$(cd -- "$CKPT_ROOT" && pwd)
EVALUATION_SPEC=${EVALUATION_SPEC:-$CKPT_ROOT/evaluation_spec.json}
if [[ ! -f "$EVALUATION_SPEC" ]]; then
    echo "Missing evaluation spec: $EVALUATION_SPEC" >&2
    exit 2
fi

shared_runtime_root=${EVAL_SHARED_RUNTIME_ROOT:-$CKPT_ROOT/.standalone-eval}
mkdir -p "$shared_runtime_root"
teacher_endpoint_file=${TEACHER_ENDPOINT_FILE:-$shared_runtime_root/teacher-endpoint.json}
teacher_stop_file=${TEACHER_STOP_FILE:-$shared_runtime_root/STOP_TEACHER}
teacher_timing_file=${TEACHER_TIMING_FILE:-$shared_runtime_root/teacher-timing.json}

json_value() {
    local expression=$1
    "$PYTHON" -c '
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

experiment_name=$(json_value semantic_config.experiment_name)
training_run_id=$(json_value training_run_id)
dataset_root=${DATASET_ROOT:-$(json_value semantic_config.benchmarks.root)}
teacher_key=$(json_value semantic_config.teacher.teacher_key)
student_tokenizer_from_spec=$(json_value semantic_config.student.tokenizer_path)
chat_template_model=${CHAT_TEMPLATE_MODEL:-$(json_value semantic_config.student.chat_template_from_model)}
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

teacher_count=$(
    "$PYTHON" -c 'import json,sys; print(len(json.loads(sys.argv[1])))' \
        "$(json_value semantic_config.teacher.models)"
)
if [[ "$teacher_count" != "1" ]]; then
    echo "Standalone two-node evaluation currently requires exactly one Teacher model; found $teacher_count." >&2
    exit 2
fi
teacher_model_from_spec=$(
    "$PYTHON" -c '
import json, sys
models = json.loads(sys.argv[1])
value = next(iter(models.values()))
print(value.get("model_path") or value.get("path") or "")
' "$(json_value semantic_config.teacher.models)"
)
TEACHER_MODEL=${TEACHER_MODEL:-$teacher_model_from_spec}
TEACHER_TOKENIZER_MODEL=${TEACHER_TOKENIZER_MODEL:-$TEACHER_MODEL}
STUDENT_TOKENIZER_MODEL=${STUDENT_TOKENIZER_MODEL:-$student_tokenizer_from_spec}
if [[ -z "$TEACHER_MODEL" || ! -d "$TEACHER_MODEL" ]]; then
    echo "Teacher model directory is unavailable: $TEACHER_MODEL" >&2
    exit 2
fi
if [[ -n "$STUDENT_TOKENIZER_MODEL" && ! -d "$STUDENT_TOKENIZER_MODEL" ]]; then
    echo "Student tokenizer directory from evaluation spec is unavailable: $STUDENT_TOKENIZER_MODEL" >&2
    exit 2
fi
if [[ -n "$chat_template_model" && ! -d "$chat_template_model" ]]; then
    echo "Chat-template tokenizer directory is unavailable: $chat_template_model" >&2
    exit 2
fi
if [[ -z "$dataset_root" || ! -d "$dataset_root" ]]; then
    echo "Benchmark dataset root is unavailable: $dataset_root" >&2
    exit 2
fi

readarray -t benchmarks < <(
    "$PYTHON" -c 'import json,sys; print("\n".join(str(x) for x in json.loads(sys.argv[1])))' "$benchmark_json"
)
if [[ -n "${EVAL_BENCHMARKS_TEXT:-}" ]]; then
    read -r -a benchmarks <<< "$EVAL_BENCHMARKS_TEXT"
fi
if ((${#benchmarks[@]} == 0)); then
    echo "evaluation_spec.json contains no benchmark names." >&2
    exit 2
fi

sampling_value() {
    local key=$1
    local fallback=$2
    "$PYTHON" -c '
import json, sys
payload = json.loads(sys.argv[1])
value = payload.get(sys.argv[2], sys.argv[3])
print(value)
' "$sampling_json" "$key" "$fallback"
}

rollout_n=${EVAL_ROLLOUT_N:-$(sampling_value n 16)}
temperature=${EVAL_TEMPERATURE:-$(sampling_value temperature 0.7)}
top_p=${EVAL_TOP_P:-$(sampling_value top_p 0.95)}
top_k=${EVAL_TOP_K:-$(sampling_value top_k -1)}
enable_thinking=$(
    "$PYTHON" -c '
import json, sys
student = json.load(open(sys.argv[1], encoding="utf-8"))["semantic_config"]["student"]
print(str(bool((student.get("apply_chat_template_kwargs") or {}).get("enable_thinking", True))).lower())
' "$EVALUATION_SPEC"
)

ALLOW_DEBUG_OVERRIDES=${ALLOW_DEBUG_OVERRIDES:-0}
if [[ "$rollout_n" != "16" && "$ALLOW_DEBUG_OVERRIDES" != "1" ]]; then
    echo "Formal checkpoint evaluation requires n=16; got $rollout_n. Set ALLOW_DEBUG_OVERRIDES=1 only for a smoke test." >&2
    exit 2
fi

RESULT_ROOT=${RESULT_ROOT:-$REPO_ROOT/eval-results/$experiment_name}
case "$RESULT_ROOT" in
    /*) ;;
    *) RESULT_ROOT=$REPO_ROOT/$RESULT_ROOT ;;
esac
if [[ -n "${EVAL_PORTABLE_RESULT_ROOT:-}" ]]; then
    case "$EVAL_PORTABLE_RESULT_ROOT" in
        /*) ;;
        *)
            echo "EVAL_PORTABLE_RESULT_ROOT must be absolute: $EVAL_PORTABLE_RESULT_ROOT" >&2
            exit 2
            ;;
    esac
    physical_result_root=$RESULT_ROOT
    mkdir -p "$physical_result_root" "$(dirname -- "$EVAL_PORTABLE_RESULT_ROOT")"
    if [[ -e "$EVAL_PORTABLE_RESULT_ROOT" || -L "$EVAL_PORTABLE_RESULT_ROOT" ]]; then
        if [[ ! "$EVAL_PORTABLE_RESULT_ROOT" -ef "$physical_result_root" ]]; then
            echo "EVAL_PORTABLE_RESULT_ROOT does not resolve to RESULT_ROOT:" >&2
            echo "  portable=$EVAL_PORTABLE_RESULT_ROOT" >&2
            echo "  physical=$physical_result_root" >&2
            exit 2
        fi
    else
        ln -s "$physical_result_root" "$EVAL_PORTABLE_RESULT_ROOT"
    fi
    # Keep the portable spelling: offline W&B Artifact metadata contains
    # absolute paths which must also exist on the networked jump host.
    RESULT_ROOT=$EVAL_PORTABLE_RESULT_ROOT
fi
EVAL_CACHE_ROOT=${EVAL_CACHE_ROOT:-$CKPT_ROOT/.eval-cache}
EVAL_LOG_ROOT=${EVAL_LOG_ROOT:-$shared_runtime_root/logs}
mkdir -p "$RESULT_ROOT" "$EVAL_CACHE_ROOT" "$EVAL_LOG_ROOT"

POLL_SECONDS=${POLL_SECONDS:-1800}
RETRY_SECONDS=${RETRY_SECONDS:-300}
WATCH_ONCE=${WATCH_ONCE:-0}
EVAL_STEPS_TEXT=${EVAL_STEPS_TEXT:-}
KEEP_MERGED_CHECKPOINTS=${KEEP_MERGED_CHECKPOINTS:-0}
OVERWRITE_RESULTS=${OVERWRITE_RESULTS:-0}
CONTINUE_ON_EVAL_ERROR=${CONTINUE_ON_EVAL_ERROR:-1}
WANDB_MODE=${WANDB_MODE:-offline}
WANDB_ENTITY=${WANDB_ENTITY:-jubgjf}
WANDB_PROJECT=${WANDB_PROJECT:-DMPD-eval}

SERVER_READY_TIMEOUT=${SERVER_READY_TIMEOUT:-3600}
SERVER_STOP_TIMEOUT=${SERVER_STOP_TIMEOUT:-180}
TEACHER_PORT=${TEACHER_PORT:-8001}
STUDENT_PORT=${STUDENT_PORT:-8000}
TEACHER_BIND_HOST=${TEACHER_BIND_HOST:-0.0.0.0}
STUDENT_BIND_HOST=${STUDENT_BIND_HOST:-127.0.0.1}
TEACHER_ADVERTISE_HOST=${TEACHER_ADVERTISE_HOST:-}
if [[ -z "$TEACHER_ADVERTISE_HOST" ]]; then
    TEACHER_ADVERTISE_HOST=$(hostname -I 2>/dev/null | awk '{print $1}')
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

gap_metric=$(
    "$PYTHON" -c 'import json,sys; print(str(json.loads(sys.argv[1]).get("gap_metric", "k3")).lower())' \
        "$intervention_json"
)
case "$gap_metric" in
    full_kl|full_distribution_kl|distribution_kl)
        echo "HTTP checkpoint evaluation refuses exact full-vocabulary KL; no approximation will be used." >&2
        exit 2
        ;;
esac
gap_distribution_topk=$(
    "$PYTHON" -c 'import json,sys; print(int(json.loads(sys.argv[1]).get("gap_distribution_topk", 16)))' \
        "$intervention_json"
)
SERVER_MAX_LOGPROBS=${SERVER_MAX_LOGPROBS:-20}
case "$gap_metric" in
    topk_kl|topk_distribution_kl|topk_dist_kl)
        if ((gap_distribution_topk > SERVER_MAX_LOGPROBS)); then
            SERVER_MAX_LOGPROBS=$gap_distribution_topk
        fi
        ;;
esac

VLLM_PID=""
VLLM_PGID=""
VLLM_LOG=""

port_is_open() {
    local host=$1
    local port=$2
    "$PYTHON" -c '
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
        "$PYTHON" -c '
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
    if [[ "$EVAL_ROLE" == "controller" ]]; then
        touch "$teacher_stop_file"
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

teacher_role_main() {
    local teacher_api_key=${TEACHER_API_KEY:-EMPTY}
    local served_model=${TEACHER_SERVED_MODEL_NAME:-dmpd-teacher}
    local base_url="http://$TEACHER_ADVERTISE_HOST:$TEACHER_PORT/v1"
    local load_started=$SECONDS
    rm -f -- "$teacher_stop_file"
    rm -f -- "$teacher_endpoint_file"
    rm -f -- "$teacher_timing_file"
    start_vllm teacher "$TEACHER_MODEL" "$TEACHER_TOKENIZER_MODEL" "$served_model" \
        "$TEACHER_BIND_HOST" "$TEACHER_PORT" "$TEACHER_TP_SIZE" "$TEACHER_DP_SIZE" \
        "$TEACHER_MAX_NUM_SEQS" "$TEACHER_GPU_MEMORY_UTILIZATION" "$TEACHER_DP_RPC_PORT" "$teacher_api_key"
    wait_for_model "$base_url" "$teacher_api_key" "$served_model"
    "$PYTHON" -c '
import json, os, sys, time
path = sys.argv[1]
payload = {"teacher_model_load_seconds": float(sys.argv[2]), "ready_at_unix": time.time()}
tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
os.replace(tmp, path)
' "$teacher_timing_file" "$((SECONDS - load_started))"
    "$PYTHON" -m "$WATCHER_MODULE" write-endpoint \
        --path "$teacher_endpoint_file" \
        --base-url "$base_url" \
        --served-model-name "$served_model" \
        --model-path "$TEACHER_MODEL" >/dev/null
    echo "[teacher] ready endpoint=$teacher_endpoint_file"
    while [[ ! -f "$teacher_stop_file" ]]; do
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "Teacher vLLM exited unexpectedly; log=$VLLM_LOG" >&2
            tail -n 200 "$VLLM_LOG" >&2 || true
            return 1
        fi
        sleep 10
    done
    echo "[teacher] controller requested shutdown"
}

wait_for_teacher_endpoint() {
    local deadline=$((SECONDS + SERVER_READY_TIMEOUT))
    while ((SECONDS < deadline)); do
        if [[ -s "$teacher_endpoint_file" ]]; then
            teacher_base_url=$(
                "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["base_url"])' \
                    "$teacher_endpoint_file"
            )
            teacher_served_model=$(
                "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["served_model_name"])' \
                    "$teacher_endpoint_file"
            )
            endpoint_model=$(
                "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_path"])' \
                    "$teacher_endpoint_file"
            )
            teacher_model_fingerprint=$(
                "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_fingerprint"])' \
                    "$teacher_endpoint_file"
            )
            if [[ "$(cd -- "$endpoint_model" && pwd)" != "$(cd -- "$TEACHER_MODEL" && pwd)" ]]; then
                echo "Teacher endpoint model differs from evaluation spec: $endpoint_model vs $TEACHER_MODEL" >&2
                return 1
            fi
            if model_is_ready "$teacher_base_url" "${TEACHER_API_KEY:-EMPTY}" "$teacher_served_model"; then
                teacher_model_load_seconds=$(
                    "$PYTHON" -c '
import json, sys
try:
    print(float(json.load(open(sys.argv[1], encoding="utf-8"))["teacher_model_load_seconds"]))
except Exception:
    print(0.0)
' "$teacher_timing_file"
                )
                return 0
            fi
        fi
        sleep 5
    done
    echo "Timed out waiting for teacher endpoint descriptor: $teacher_endpoint_file" >&2
    return 1
}

hf_checkpoint_ready() {
    "$PYTHON" -m "$EVAL_MODULE" checkpoint-ready --path "$1" >/dev/null 2>&1
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
    mkdir -p "$cache_step"
    local temporary="$cache_step/.huggingface.tmp.$$"
    rm -rf -- "$temporary"
    "$PYTHON" -m verl.model_merger merge --backend fsdp --local_dir "$actor_dir" --target_dir "$temporary" >&2
    if ! hf_checkpoint_ready "$temporary"; then
        echo "Merged Hugging Face checkpoint is incomplete: $temporary" >&2
        return 1
    fi
    rm -rf -- "$target"
    mv -- "$temporary" "$target"
    printf '%s\n' "$target"
}

checkpoint_complete() {
    local step=$1
    local benchmark
    for benchmark in "${benchmarks[@]}"; do
        if ! "$PYTHON" -m "$EVAL_MODULE" status \
            --result-root "$RESULT_ROOT" \
            --benchmark "$benchmark" \
            --global-step "$step" \
            --no-validate-inputs \
            --verify-checksums \
            --require-repair >/dev/null 2>&1
        then
            return 1
        fi
    done
    return 0
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

run_raw_unit() {
    local step=$1
    local checkpoint=$2
    local benchmark=$3
    local served_model=$4
    local base_url=$5
    local tokenizer_model=$6
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
    "$PYTHON" -m "$EVAL_MODULE" "${flags[@]}"
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
    "$PYTHON" -m "$EVAL_MODULE" "${flags[@]}"
}

sync_wandb() {
    local -a flags=(
        --base-dir "$(dirname -- "$RESULT_ROOT")"
        --experiments "$(basename -- "$RESULT_ROOT")"
        --entity "$WANDB_ENTITY"
        --project "$WANDB_PROJECT"
        --mode "$WANDB_MODE"
        --wandb-dir "$RESULT_ROOT/.wandb"
        --wandb-data-dir "$RESULT_ROOT/.wandb-data"
        --wandb-cache-dir "$RESULT_ROOT/.wandb-cache"
        --source-training-run-id "$training_run_id"
    )
    "$PYTHON" "$SCRIPT_DIR/upload_opd_benchmark_wandb.py" "${flags[@]}"
}

register_checkpoint_attempt() {
    local step=$1
    "$PYTHON" -c '
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
    step=$("$PYTHON" -c 'import json,sys; print(json.loads(sys.argv[1])["step"])' "$candidate_json")
    actor_dir=$("$PYTHON" -c 'import json,sys; print(json.loads(sys.argv[1])["actor_dir"])' "$candidate_json")
    direct_hf=$("$PYTHON" -c 'import json,sys; print(json.loads(sys.argv[1]).get("hf_checkpoint") or "")' "$candidate_json")
    if checkpoint_complete "$step"; then
        echo "[watcher] global_step_$step already complete"
        return 0
    fi
    local attempt_count
    attempt_count=$(register_checkpoint_attempt "$step")
    local checkpoint_started=$SECONDS
    local load_started=$SECONDS
    if ! checkpoint=$(ensure_hf_checkpoint "$step" "$actor_dir" "$direct_hf"); then
        return 1
    fi
    local tokenizer_model=${STUDENT_TOKENIZER_MODEL:-$checkpoint}
    local served_model="dmpd-student-step-$step"
    local student_base_url="http://127.0.0.1:$STUDENT_PORT/v1"
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
    local benchmark
    for benchmark in "${benchmarks[@]}"; do
        if ! run_raw_unit "$step" "$checkpoint" "$benchmark" "$served_model" "$student_base_url" \
            "$tokenizer_model"
        then
            return 1
        fi
        if ! run_repair_unit "$step" "$benchmark" "$served_model" "$student_base_url" "$tokenizer_model"; then
            return 1
        fi
    done
    if ! stop_vllm; then
        return 1
    fi
    if ! "$PYTHON" -c '
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
    if ! sync_wandb; then
        return 1
    fi
    if [[ "$KEEP_MERGED_CHECKPOINTS" != "1" && -z "$direct_hf" ]]; then
        rm -rf -- "$EVAL_CACHE_ROOT/global_step_$step"
    fi
    echo "[watcher] completed global_step_$step"
}

controller_role_main() {
    command -v flock >/dev/null 2>&1 || {
        echo "The controller requires flock for the checkpoint-root consumer lock." >&2
        return 2
    }
    exec 9>"$shared_runtime_root/controller.lock"
    if ! flock -n 9; then
        echo "Another checkpoint evaluator already owns $shared_runtime_root/controller.lock." >&2
        return 2
    fi
    wait_for_teacher_endpoint
    while true; do
        local candidate_output
        if ! candidate_output=$("$PYTHON" -m "$WATCHER_MODULE" discover --checkpoint-root "$CKPT_ROOT"); then
            return 1
        fi
        local -a candidates=()
        if [[ -n "$candidate_output" ]]; then
            mapfile -t candidates <<< "$candidate_output"
        fi
        local processed=0
        local candidate
        for candidate in "${candidates[@]}"; do
            local step
            step=$("$PYTHON" -c 'import json,sys; print(json.loads(sys.argv[1])["step"])' "$candidate")
            if ! step_is_selected "$step"; then
                continue
            fi
            if checkpoint_complete "$step"; then
                continue
            fi
            processed=$((processed + 1))
            if ! evaluate_checkpoint "$candidate"; then
                stop_vllm || true
                echo "[watcher] global_step_$step failed; resumable shards were retained." >&2
                if [[ "$CONTINUE_ON_EVAL_ERROR" != "1" ]]; then
                    return 1
                fi
                sleep "$RETRY_SECONDS"
            fi
        done
        if [[ "$WATCH_ONCE" == "1" ]]; then
            break
        fi
        if ((processed == 0)); then
            echo "[watcher] no pending committed checkpoint; sleeping ${POLL_SECONDS}s"
        fi
        sleep "$POLL_SECONDS"
    done
    touch "$teacher_stop_file"
}

echo "================================================================================"
echo "Standalone checkpoint evaluation"
echo "  role:             $EVAL_ROLE"
echo "  experiment:       $experiment_name"
echo "  source run id:    ${training_run_id:-unset}"
echo "  checkpoint root:  $CKPT_ROOT"
echo "  result root:      $RESULT_ROOT"
if [[ "$WANDB_MODE" == "offline" ]]; then
    echo "  offline sync:     $RESULT_ROOT/wandb-offline-sync.sh"
    if [[ -z "${EVAL_PORTABLE_RESULT_ROOT:-}" ]]; then
        echo "  path requirement: RESULT_ROOT must have this identical spelling on the jump host"
    fi
fi
echo "  rollout n:        $rollout_n"
echo "  benchmarks:       ${benchmarks[*]}"
echo "  teacher key:      $teacher_key"
echo "  gap metric:       $gap_metric"
echo "================================================================================"

cd "$REPO_ROOT"
if [[ "$EVAL_ROLE" == "teacher" ]]; then
    teacher_role_main
else
    controller_role_main
fi
