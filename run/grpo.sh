#!/usr/bin/env bash

# DAPO-Math-17K GRPO baseline for the Ascend verl image documented below.
# Supported models:
#   - deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
#   - nvidia/OpenMath-Nemotron-1.5B
#
# The script intentionally targets the image's /verl checkout instead of this
# repository. MODEL_PATH is the only required model-specific setting.

# Re-enter Bash when a scheduler invokes this file with `sh run/grpo.sh`.
if [ -z "${BASH_VERSION:-}" ]; then
    exec /bin/bash "$0" "$@"
fi

set -Eeuo pipefail
if [[ "${VERBOSE:-0}" == "1" ]]; then
    set -x
fi

die() {
    echo "ERROR: $*" >&2
    exit 1
}

warn() {
    echo "WARNING: $*" >&2
}

is_true() {
    case "${1:-}" in
        1 | true | True | TRUE | yes | Yes | YES) return 0 ;;
        *) return 1 ;;
    esac
}

check_positive_integer() {
    local name=$1
    local value=$2
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "$name must be a positive integer, got: $value"
}

ALLOW_VERSION_MISMATCH=${ALLOW_VERSION_MISMATCH:-0}
SKIP_RUNTIME_PREFLIGHT=${SKIP_RUNTIME_PREFLIGHT:-0}
PREPARE_ONLY=${PREPARE_ONLY:-0}
DRY_RUN=${DRY_RUN:-0}

if is_true "$SKIP_RUNTIME_PREFLIGHT" && ! is_true "$PREPARE_ONLY" && ! is_true "$DRY_RUN"; then
    die "SKIP_RUNTIME_PREFLIGHT is permitted only with PREPARE_ONLY=1 or DRY_RUN=1."
fi

compatibility_mismatch() {
    if is_true "$ALLOW_VERSION_MISMATCH"; then
        warn "$* (continuing because ALLOW_VERSION_MISMATCH=1)"
    else
        die "$* Set ALLOW_VERSION_MISMATCH=1 only after reviewing the incompatibility."
    fi
}

check_git_revision() {
    local label=$1
    local repo=$2
    local expected=$3
    local actual

    if [[ ! -d "$repo/.git" ]]; then
        compatibility_mismatch "$label checkout is missing at $repo or has no .git directory."
        return
    fi
    actual=$(git -c safe.directory="$repo" -C "$repo" rev-parse HEAD 2>/dev/null) || {
        compatibility_mismatch "Cannot read $label revision from $repo."
        return
    }
    echo "$label revision: $actual"
    if [[ "$actual" != "$expected"* ]]; then
        compatibility_mismatch "$label revision $actual does not match expected prefix $expected."
    fi
}

check_cann_version() {
    local expected=${EXPECTED_CANN_VERSION:-9.1.0}
    local report
    local status

    set +e
    report=$(python3 - "$expected" "${ASCEND_HOME_PATH:-}" <<'PY'
import glob
import re
import sys
from pathlib import Path

expected, ascend_home = sys.argv[1:]
candidates = [
    "/usr/local/Ascend/ascend-toolkit/latest/version.cfg",
    "/usr/local/Ascend/ascend-toolkit/latest/ascend_toolkit_install.info",
    "/etc/Ascend/ascend_cann_install.info",
    "/etc/ascend_install.info",
]
if ascend_home:
    candidates.extend(
        [
            str(Path(ascend_home) / "version.cfg"),
            str(Path(ascend_home) / "ascend_toolkit_install.info"),
        ]
    )
candidates.extend(glob.glob("/usr/local/Ascend/ascend-toolkit/latest/*/ascend_toolkit_install.info"))

seen = set()
reports = []
for candidate in candidates:
    path = Path(candidate)
    if path in seen or not path.is_file():
        continue
    seen.add(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    reports.append(f"{path}: {text[:2000].strip()}")

if not reports:
    print("No CANN version metadata file was found")
    raise SystemExit(2)

report = "\n".join(reports)
print(report)
if re.search(rf"(?<![0-9]){re.escape(expected)}(?![0-9])", report) is None:
    raise SystemExit(3)
PY
    )
    status=$?
    set -e

    if ((status == 2)); then
        compatibility_mismatch "Unable to determine the installed CANN version."
    elif ((status != 0)); then
        compatibility_mismatch "CANN metadata does not contain expected version $expected. Detected metadata: $report"
    else
        echo "CANN version metadata verified: $expected"
    fi
}

setup_ascend_environment() {
    local cann_env=${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}
    if [[ -f "$cann_env" ]]; then
        set +u
        # shellcheck disable=SC1090
        source "$cann_env"
        set -u
    elif ! is_true "$PREPARE_ONLY" && ! is_true "$DRY_RUN"; then
        die "CANN environment script not found: $cann_env"
    else
        warn "CANN environment script not found: $cann_env"
    fi

    export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1800}
    export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-auto}
    export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-auto}
    unset HCCL_IF_BASE_PORT || true

    export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
    export VLLM_USE_V1=${VLLM_USE_V1:-1}
    export VLLM_ALLREDUCE_USE_SYMM_MEM=${VLLM_ALLREDUCE_USE_SYMM_MEM:-0}
    export VLLM_ASCEND_ENABLE_PREFETCH_MLP=${VLLM_ASCEND_ENABLE_PREFETCH_MLP:-1}
    export VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE=${VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE:-1}
    export VLLM_ASCEND_ENABLE_FLASHCOMM1=${VLLM_ASCEND_ENABLE_FLASHCOMM1:-1}
    export CPU_AFFINITY_CONF=${CPU_AFFINITY_CONF:-1}
    export TOKENIZERS_PARALLELISM=false
    export HYDRA_FULL_ERROR=1
    ulimit -n "${ULIMIT_NOFILE:-65535}" || true
}

check_runtime_versions() {
    if is_true "$SKIP_RUNTIME_PREFLIGHT"; then
        warn "Skipping image revision and package-version checks."
        return
    fi

    check_cann_version
    check_git_revision verl "$VERL_ROOT" ea53291385ce764019a2b40733605f21d8317583
    check_git_revision Megatron-LM /Megatron-LM ddc0d67
    check_git_revision MindSpeed /MindSpeed 084cf58a
    check_git_revision transformers /transformers cc7ab9b
    check_git_revision vLLM /vllm 0fc695f
    check_git_revision vLLM-Ascend /vllm-ascend 84f60b825

    python3 - "$ALLOW_VERSION_MISMATCH" <<'PY'
import sys
from importlib.metadata import PackageNotFoundError, version

allow_mismatch = sys.argv[1].lower() in {"1", "true", "yes"}
requirements = {
    "mbridge": ("0.15.1", "exact"),
    "vllm": ("0.23.0", "prefix"),
    "vllm-ascend": ("0.23.1.dev15", "prefix"),
}

errors = []
reported = {}
for package, (expected, match_mode) in requirements.items():
    try:
        actual = version(package)
    except PackageNotFoundError:
        actual = "missing"
    reported[package] = actual
    matches = actual == expected if match_mode == "exact" else actual.startswith(expected)
    if not matches:
        errors.append(f"{package}={actual!r}, expected {match_mode} match {expected!r}")

for package in ("torch", "torch-npu", "transformers", "ray", "pyarrow", "ijson"):
    try:
        reported[package] = version(package)
    except PackageNotFoundError:
        reported[package] = "missing"

print("Runtime packages:", reported)
if errors and not allow_mismatch:
    raise RuntimeError("; ".join(errors))
if errors:
    print("WARNING: package mismatches accepted:", "; ".join(errors), file=sys.stderr)
PY
}

print_command() {
    printf 'Command:'
    printf ' %q' "$@"
    printf '\n'
}

prepare_deepseek_distill_agent_loop_compat() {
    local compat_dir=${GRPO_COMPAT_DIR:-$OUTPUT_ROOT/runtime_compat/verl-ea532913-grpo}
    local module_path=$compat_dir/verl_grpo_deepseek_distill_compat.py
    local config_path=$compat_dir/deepseek_distill_agent_loop.yaml

    mkdir -p "$compat_dir"
    python3 - "$module_path" "$config_path" <<'PY'
import os
import sys
from pathlib import Path

module_path = Path(sys.argv[1])
config_path = Path(sys.argv[2])

module_source = '''\
"""DeepSeek-R1-Distill-Qwen adapter for verl's Continuous Token dispatch."""

from typing import Any

from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop


class DeepSeekDistillSingleTurnAgentLoop(SingleTurnAgentLoop):
    """Select DeepSeek boundaries for a Qwen2-architecture distillation model."""

    def __init__(self, *args: Any, hf_model_type: str | None = None, **kwargs: Any) -> None:
        del hf_model_type
        super().__init__(*args, hf_model_type="deepseek_v3", **kwargs)
'''

config_source = '''\
- name: deepseek_distill_single_turn_agent
  _target_: verl_grpo_deepseek_distill_compat.DeepSeekDistillSingleTurnAgentLoop
'''


def write_if_changed(path: Path, content: str) -> None:
    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary_path.write_text(content, encoding="utf-8")
    os.replace(temporary_path, path)


write_if_changed(module_path, module_source)
write_if_changed(config_path, config_source)
PY

    DEEPSEEK_AGENT_LOOP_CONFIG=$config_path
    GRPO_RAY_PYTHONPATH=$compat_dir
    if [[ -n "${PYTHONPATH:-}" ]]; then
        GRPO_RAY_PYTHONPATH+=":$PYTHONPATH"
    fi
    export PYTHONPATH=$GRPO_RAY_PYTHONPATH

    python3 - "$MODEL_PATH" "$config_path" <<'PY'
import hydra
import importlib
import sys

from omegaconf import OmegaConf
from transformers import AutoTokenizer
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.utils.tokenizer.continuous_token import DeepSeekContinuousTokenBuilder
from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder

model_path, agent_loop_config_path = sys.argv[1:]
compat_module = importlib.import_module("verl_grpo_deepseek_distill_compat")
loop_cls = compat_module.DeepSeekDistillSingleTurnAgentLoop

captured = {}
original_init = SingleTurnAgentLoop.__init__


def capture_init(self, *args, **kwargs):
    del self, args
    captured.update(kwargs)


SingleTurnAgentLoop.__init__ = capture_init
try:
    agent_loop_configs = OmegaConf.load(agent_loop_config_path)
    if len(agent_loop_configs) != 1:
        raise RuntimeError(f"Expected exactly one compatibility agent loop, got {len(agent_loop_configs)}")
    configured_loop = hydra.utils.instantiate(agent_loop_configs[0], hf_model_type="qwen2")
finally:
    SingleTurnAgentLoop.__init__ = original_init

if not isinstance(configured_loop, loop_cls):
    raise RuntimeError(f"Unexpected compatibility loop class: {type(configured_loop).__name__}")
if captured.get("hf_model_type") != "deepseek_v3":
    raise RuntimeError(f"DeepSeek agent-loop adapter did not override builder routing: {captured}")

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
try:
    create_continuous_token_builder(tokenizer, hf_model_type="qwen2")
except ValueError as error:
    if "<|im_end|>" not in str(error):
        raise
else:
    raise RuntimeError("Pinned verl unexpectedly accepted the DeepSeek tokenizer as Qwen ChatML")

builder = create_continuous_token_builder(tokenizer, hf_model_type=captured["hf_model_type"])
if not isinstance(builder, DeepSeekContinuousTokenBuilder):
    raise RuntimeError(f"Expected DeepSeekContinuousTokenBuilder, got {type(builder).__name__}")

messages = [{"role": "user", "content": "Continuous-token compatibility probe"}]
expected = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
if hasattr(expected, "input_ids"):
    expected = expected.input_ids
if hasattr(expected, "tolist"):
    expected = expected.tolist()
if expected and isinstance(expected[0], list):
    expected = expected[0]
actual = builder.build_initial_tokens(messages)
if actual != list(expected):
    raise RuntimeError("DeepSeek compatibility builder changed the initial chat-template token ids")

print(
    "GRPO continuous-token compatibility:",
    {
        "agent_loop": "deepseek_distill_single_turn_agent",
        "builder": type(builder).__name__,
        "forced_protocol_model_type": captured["hf_model_type"],
        "prompt_tokens": len(actual),
    },
)
PY
}

verify_chatml_continuous_token_builder() {
    python3 - "$MODEL_PATH" <<'PY'
import sys

from transformers import AutoTokenizer
from verl.utils.tokenizer.continuous_token import QwenContinuousTokenBuilder
from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder

tokenizer = AutoTokenizer.from_pretrained(sys.argv[1], trust_remote_code=False)
builder = create_continuous_token_builder(tokenizer, hf_model_type="qwen2")
if not isinstance(builder, QwenContinuousTokenBuilder):
    raise RuntimeError(f"Expected QwenContinuousTokenBuilder, got {type(builder).__name__}")

messages = [{"role": "user", "content": "Continuous-token compatibility probe"}]
expected = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
if hasattr(expected, "input_ids"):
    expected = expected.input_ids
if hasattr(expected, "tolist"):
    expected = expected.tolist()
if expected and isinstance(expected[0], list):
    expected = expected[0]
actual = builder.build_initial_tokens(messages)
if actual != list(expected):
    raise RuntimeError("Qwen compatibility builder changed the initial chat-template token ids")

print(
    "GRPO continuous-token compatibility:",
    {"agent_loop": "single_turn_agent", "builder": type(builder).__name__, "prompt_tokens": len(actual)},
)
PY
}

################################ paths and experiment identity

VERL_ROOT=${VERL_ROOT:-/verl}
[[ -d "$VERL_ROOT" ]] || die "VERL_ROOT does not exist: $VERL_ROOT"

DATA_ROOT=${DATA_ROOT:-/opt/huawei/dataset/jnguan}
RL_DATA_DIR=${RL_DATA_DIR:-$DATA_ROOT/datasets/open-r1/DAPO-Math-17k-Processed-verl--qwen3}
TRAIN_FILE=${TRAIN_FILE:-$RL_DATA_DIR/train.parquet}
VAL_FILE=${VAL_FILE:-$RL_DATA_DIR/test.parquet}

MODEL_PATH=${MODEL_PATH:-$DATA_ROOT/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}
MODEL_NAME=${MODEL_NAME:-$(basename -- "$MODEL_PATH")}
MODEL_SLUG=$(printf '%s' "$MODEL_NAME" | tr '[:upper:]' '[:lower:]' | tr -cs '[:alnum:]' '-')
MODEL_SLUG=${MODEL_SLUG#-}
MODEL_SLUG=${MODEL_SLUG%-}
[[ -n "$MODEL_SLUG" ]] || die "MODEL_NAME must contain at least one alphanumeric character."

OUTPUT_ROOT=${OUTPUT_ROOT:-$DATA_ROOT/outputs/dapo_math_17k_baselines}
PROJECT_NAME=${PROJECT_NAME:-DMPD}
EXPERIMENT_TIMESTAMP=${EXPERIMENT_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-${MODEL_SLUG}-grpo-${EXPERIMENT_TIMESTAMP}}
CKPT_DIR=${CKPT_DIR:-$OUTPUT_ROOT/checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME}
LOG_DIR=${LOG_DIR:-$OUTPUT_ROOT/logs}
LOG_FILE=${LOG_FILE:-$LOG_DIR/${EXPERIMENT_NAME}.log}

export WANDB_ENTITY=${WANDB_ENTITY:-jubgjf}
export WANDB_PROJECT=${WANDB_PROJECT:-$PROJECT_NAME}
export WANDB_NAME=${WANDB_NAME:-$EXPERIMENT_NAME}

mkdir -p "$CKPT_DIR" "$LOG_DIR"

[[ -d "$MODEL_PATH" ]] || die "Missing model directory: $MODEL_PATH"
[[ -f "$TRAIN_FILE" ]] || die "Missing training parquet: $TRAIN_FILE"
[[ -f "$VAL_FILE" ]] || die "Missing validation parquet: $VAL_FILE"

################################ cluster and parallel topology

NNODES=${NNODES:-${VC_WORKER_NUM:-1}}
NPUS_PER_NODE=${NPUS_PER_NODE:-8}
NODE_RANK=${NODE_RANK:-${VC_TASK_INDEX:-0}}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-6167}
export MASTER_ADDR MASTER_PORT

check_positive_integer NNODES "$NNODES"
check_positive_integer NPUS_PER_NODE "$NPUS_PER_NODE"
[[ "$NODE_RANK" =~ ^[0-9]+$ ]] || die "NODE_RANK must be a non-negative integer, got: $NODE_RANK"
((NODE_RANK < NNODES)) || die "NODE_RANK=$NODE_RANK must be smaller than NNODES=$NNODES"

if ((NNODES > 1)); then
    [[ "$NODE_RANK" == "0" ]] || die "Submit multi-node GRPO only once from NODE_RANK=0; workers must already belong to Ray."
    [[ -n "${RAY_ADDRESS:-}" ]] || die "NNODES=$NNODES requires a pre-existing Ray cluster and RAY_ADDRESS."
fi

WORLD_SIZE=$((NNODES * NPUS_PER_NODE))
ACTOR_TP=${ACTOR_TP:-1}
ACTOR_PP=${ACTOR_PP:-1}
ACTOR_CP=${ACTOR_CP:-2}
ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_PP=${ROLLOUT_PP:-1}
ROLLOUT_DP=${ROLLOUT_DP:-1}

for setting in ACTOR_TP ACTOR_PP ACTOR_CP ROLLOUT_TP ROLLOUT_PP ROLLOUT_DP; do
    check_positive_integer "$setting" "${!setting}"
done

MODEL_PARALLEL_SIZE=$((ACTOR_TP * ACTOR_PP * ACTOR_CP))
((WORLD_SIZE % MODEL_PARALLEL_SIZE == 0)) || die \
    "WORLD_SIZE=$WORLD_SIZE must be divisible by actor TP*PP*CP=$MODEL_PARALLEL_SIZE."
ACTOR_DP=$((WORLD_SIZE / MODEL_PARALLEL_SIZE))

((ROLLOUT_DP == 1)) || die \
    "This verl/vLLM-Ascend combination requires ROLLOUT_DP=1; use multiple rollout replicas instead."
ROLLOUT_FOOTPRINT=$((ROLLOUT_TP * ROLLOUT_PP * ROLLOUT_DP))
((WORLD_SIZE % ROLLOUT_FOOTPRINT == 0)) || die \
    "WORLD_SIZE=$WORLD_SIZE must be divisible by rollout TP*PP*DP=$ROLLOUT_FOOTPRINT."
ROLLOUT_REPLICAS=$((WORLD_SIZE / ROLLOUT_FOOTPRINT))

################################ lengths, batches, and optimizer

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-23552}
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-256}
ROLLOUT_N=${ROLLOUT_N:-4}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}

for setting in \
    MAX_PROMPT_LENGTH \
    MAX_RESPONSE_LENGTH \
    PPO_MAX_TOKEN_LEN_PER_GPU \
    TRAIN_BATCH_SIZE \
    VAL_BATCH_SIZE \
    ROLLOUT_N \
    PPO_MINI_BATCH_SIZE \
    PPO_MICRO_BATCH_SIZE_PER_GPU; do
    check_positive_integer "$setting" "${!setting}"
done

((ACTOR_CP == 1 || MAX_MODEL_LEN % (2 * ACTOR_CP) == 0)) || die \
    "MAX_MODEL_LEN=$MAX_MODEL_LEN must be divisible by 2*ACTOR_CP=$((2 * ACTOR_CP))."
((TRAIN_BATCH_SIZE % ACTOR_DP == 0)) || die \
    "TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE must be divisible by ACTOR_DP=$ACTOR_DP."
TOTAL_ROLLOUT_SAMPLES=$((TRAIN_BATCH_SIZE * ROLLOUT_N))
((TRAIN_BATCH_SIZE % PPO_MINI_BATCH_SIZE == 0)) || die \
    "TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE must be divisible by prompt-level PPO_MINI_BATCH_SIZE=$PPO_MINI_BATCH_SIZE."

ACTOR_LR=${ACTOR_LR:-1e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-null}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-20}
SEED=${SEED:-42}

ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.50}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-8}

EXPECTED_TRAIN_ROWS=${EXPECTED_TRAIN_ROWS:-13918}
EXPECTED_VAL_ROWS=${EXPECTED_VAL_ROWS:-3480}
EXPECTED_ELIGIBLE_TRAIN_ROWS=${EXPECTED_ELIGIBLE_TRAIN_ROWS:-13914}
EXPECTED_ELIGIBLE_VAL_ROWS=${EXPECTED_ELIGIBLE_VAL_ROWS:-3476}
STRICT_EXPECTED_COUNTS=${STRICT_EXPECTED_COUNTS:-1}
for setting in EXPECTED_TRAIN_ROWS EXPECTED_VAL_ROWS EXPECTED_ELIGIBLE_TRAIN_ROWS EXPECTED_ELIGIBLE_VAL_ROWS; do
    check_positive_integer "$setting" "${!setting}"
done
EXPECTED_STEPS_PER_EPOCH=$((EXPECTED_ELIGIBLE_TRAIN_ROWS / TRAIN_BATCH_SIZE))

setup_ascend_environment
cd "$VERL_ROOT"
check_runtime_versions

if [[ "${CLEAN_STALE_RAY:-0}" == "1" ]]; then
    ray stop --force || true
fi

################################ model and data compatibility preflight

REQUIRE_NPU=1
if is_true "$PREPARE_ONLY" || is_true "$DRY_RUN"; then
    REQUIRE_NPU=0
fi

python3 - \
    "$MODEL_PATH" \
    "$TRAIN_FILE" \
    "$VAL_FILE" \
    "$MAX_PROMPT_LENGTH" \
    "$MAX_MODEL_LEN" \
    "$ACTOR_TP" \
    "$ACTOR_PP" \
    "$ROLLOUT_TP" \
    "$NPUS_PER_NODE" \
    "$EXPECTED_TRAIN_ROWS" \
    "$EXPECTED_VAL_ROWS" \
    "$EXPECTED_ELIGIBLE_TRAIN_ROWS" \
    "$EXPECTED_ELIGIBLE_VAL_ROWS" \
    "$STRICT_EXPECTED_COUNTS" \
    "$REQUIRE_NPU" \
    "$SKIP_RUNTIME_PREFLIGHT" <<'PY'
import hashlib
import json
import sys
from collections.abc import Mapping

import pyarrow.parquet as pq
import torch
from transformers import AutoConfig, AutoTokenizer

(
    model_path,
    train_file,
    val_file,
    max_prompt_length,
    max_model_len,
    actor_tp,
    actor_pp,
    rollout_tp,
    expected_local_npus,
    expected_train_rows,
    expected_val_rows,
    expected_eligible_train_rows,
    expected_eligible_val_rows,
    strict_expected_counts,
    require_npu,
    skip_runtime_preflight,
) = sys.argv[1:]

max_prompt_length = int(max_prompt_length)
max_model_len = int(max_model_len)
actor_tp = int(actor_tp)
actor_pp = int(actor_pp)
rollout_tp = int(rollout_tp)
expected_local_npus = int(expected_local_npus)
expected_train_rows = int(expected_train_rows)
expected_val_rows = int(expected_val_rows)
expected_eligible_train_rows = int(expected_eligible_train_rows)
expected_eligible_val_rows = int(expected_eligible_val_rows)
strict_expected_counts = strict_expected_counts.lower() in {"1", "true", "yes"}
require_npu = require_npu.lower() in {"1", "true", "yes"}
skip_runtime_preflight = skip_runtime_preflight.lower() in {"1", "true", "yes"}


def as_python(value):
    if hasattr(value, "as_py"):
        return value.as_py()
    return value


def normalize_token_ids(token_ids):
    if hasattr(token_ids, "input_ids"):
        token_ids = token_ids.input_ids
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        if len(token_ids) != 1:
            raise ValueError("Unexpected batched tokenization result")
        token_ids = token_ids[0]
    return list(token_ids)


def parse_json_like(value, field_name, row_index):
    value = as_python(value)
    if isinstance(value, str) and value.lstrip().startswith(("[", "{")):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"row {row_index} has invalid JSON in {field_name}") from exc
    return value


def normalize_messages(value, row_index):
    value = parse_json_like(value, "prompt", row_index)
    if isinstance(value, str):
        messages = [{"role": "user", "content": value}]
    elif isinstance(value, Mapping):
        messages = [dict(value)]
    elif isinstance(value, list):
        messages = [dict(item) for item in value]
    else:
        raise TypeError(f"row {row_index} has unsupported prompt type {type(value).__name__}")
    if not messages or messages[-1].get("role") != "user":
        raise ValueError(f"row {row_index} must end with a user message")
    for message in messages:
        if message.get("role") not in {"system", "user"}:
            raise ValueError(f"row {row_index} has unsupported prompt role {message.get('role')!r}")
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError(f"row {row_index} has an empty prompt message")
    return messages


def ground_truth(row, row_index):
    reward_model = parse_json_like(row.get("reward_model"), "reward_model", row_index)
    if isinstance(reward_model, Mapping):
        answer = reward_model.get("ground_truth")
    else:
        answer = None
    if answer is None:
        answer = row.get("solution")
    if answer is None or not str(answer).strip():
        raise ValueError(f"row {row_index} has no reward_model.ground_truth")
    return str(answer).strip()


def load_split(path, split):
    rows = pq.read_table(path).to_pylist()
    normalized = []
    key_rows = {}
    for index, row in enumerate(rows):
        messages = normalize_messages(row.get("prompt"), index)
        answer = ground_truth(row, index)
        data_source = row.get("data_source")
        if data_source not in (None, "math_dapo"):
            raise ValueError(f"{split} row {index} has data_source={data_source!r}, expected 'math_dapo'")
        key = hashlib.sha256(
            json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        key_rows.setdefault(key, []).append((index, answer))
        normalized.append((messages, answer))
    duplicate_keys = {key: values for key, values in key_rows.items() if len(values) > 1}
    conflict_keys = {
        key: values
        for key, values in duplicate_keys.items()
        if len({answer for _index, answer in values}) > 1
    }
    stats = {
        "rows": len(rows),
        "unique_prompt_keys": len(key_rows),
        "duplicate_prompt_keys": len(duplicate_keys),
        "duplicate_rows_beyond_first": sum(len(values) - 1 for values in duplicate_keys.values()),
        "ground_truth_conflict_keys": len(conflict_keys),
    }
    return normalized, key_rows, stats


train_rows, train_key_rows, train_stats = load_split(train_file, "train")
val_rows, val_key_rows, val_stats = load_split(val_file, "validation")
cross_split_keys = set(train_key_rows) & set(val_key_rows)
cross_split_ground_truth_conflicts = {
    key
    for key in cross_split_keys
    if {answer for _index, answer in train_key_rows[key]}
    != {answer for _index, answer in val_key_rows[key]}
}
if train_stats["duplicate_prompt_keys"] or val_stats["duplicate_prompt_keys"]:
    print(
        "WARNING: preserving duplicate rows from the authoritative RL parquets: "
        f"train_keys={train_stats['duplicate_prompt_keys']}, "
        f"validation_keys={val_stats['duplicate_prompt_keys']}",
        file=sys.stderr,
    )
if cross_split_keys:
    print(
        "WARNING: authoritative RL parquets contain prompts in both splits; "
        f"preserving their memberships exactly (keys={len(cross_split_keys)}).",
        file=sys.stderr,
    )

count_errors = []
if len(train_rows) != expected_train_rows:
    count_errors.append(f"train rows={len(train_rows)}, expected={expected_train_rows}")
if len(val_rows) != expected_val_rows:
    count_errors.append(f"validation rows={len(val_rows)}, expected={expected_val_rows}")
if count_errors and strict_expected_counts:
    raise ValueError("; ".join(count_errors))
if count_errors:
    print("WARNING:", "; ".join(count_errors), file=sys.stderr)

config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)

if config.model_type != "qwen2":
    raise ValueError(f"Expected model_type='qwen2', got {config.model_type!r}")
if list(getattr(config, "architectures", [])) != ["Qwen2ForCausalLM"]:
    raise ValueError(f"Expected Qwen2ForCausalLM, got architectures={config.architectures!r}")
if int(config.max_position_embeddings) < max_model_len:
    raise ValueError(
        f"max_position_embeddings={config.max_position_embeddings} is below requested {max_model_len}"
    )
if int(config.num_hidden_layers) % actor_pp:
    raise ValueError("num_hidden_layers must be divisible by ACTOR_PP")

num_kv_heads = int(getattr(config, "num_key_value_heads", config.num_attention_heads))
for name, parallel_size in (("ACTOR_TP", actor_tp), ("ROLLOUT_TP", rollout_tp)):
    if int(config.num_attention_heads) % parallel_size:
        raise ValueError(f"num_attention_heads must be divisible by {name}={parallel_size}")
    if num_kv_heads % parallel_size:
        raise ValueError(f"num_key_value_heads must be divisible by {name}={parallel_size}")
if len(tokenizer) > int(config.vocab_size):
    raise ValueError(f"tokenizer size {len(tokenizer)} exceeds model vocab_size {config.vocab_size}")
special_ids = set(tokenizer.all_special_ids)
special_ids.update(
    token_id
    for token_id in (getattr(config, "bos_token_id", None), getattr(config, "eos_token_id", None))
    if isinstance(token_id, int)
)
if special_ids and max(special_ids) >= int(config.vocab_size):
    raise ValueError(f"special token ids exceed model vocab_size: {sorted(special_ids)}")
tokenizer_model_max_length = int(tokenizer.model_max_length)
if tokenizer_model_max_length < max_model_len:
    print(
        "WARNING: tokenizer metadata advertises model_max_length="
        f"{tokenizer_model_max_length}, while model config supports "
        f"{config.max_position_embeddings}; rollout.max_model_len={max_model_len} is explicit.",
        file=sys.stderr,
    )

generation_probe = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Template preflight"}],
    tokenize=False,
    add_generation_prompt=True,
)
if generation_probe.endswith("<｜Assistant｜><think>\n"):
    generation_prefix = "deepseek_native_thinking"
    protocol_eos = "<｜end▁of▁sentence｜>"
    protocol_eos_id = tokenizer.convert_tokens_to_ids(protocol_eos)
    if protocol_eos_id is None or protocol_eos_id == tokenizer.unk_token_id:
        raise ValueError(f"DeepSeek tokenizer does not define required token {protocol_eos!r}")
elif generation_probe.endswith("<|im_start|>assistant\n"):
    generation_prefix = "native_chatml"
    protocol_eos = "<|im_end|>"
    protocol_eos_id = tokenizer.convert_tokens_to_ids(protocol_eos)
    if protocol_eos_id is None or protocol_eos_id == tokenizer.unk_token_id:
        raise ValueError(f"ChatML tokenizer does not define required token {protocol_eos!r}")
else:
    raise ValueError(f"Unsupported generation prefix: {generation_probe[-200:]!r}")

overlong = []
overlong_counts = {"train": 0, "validation": 0}
max_seen = 0
for split, rows in (("train", train_rows), ("validation", val_rows)):
    for index, (messages, _answer) in enumerate(rows):
        token_ids = normalize_token_ids(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        )
        length = len(token_ids)
        max_seen = max(max_seen, length)
        if length > max_prompt_length:
            overlong_counts[split] += 1
            if len(overlong) < 20:
                overlong.append((split, index, length))
if overlong:
    print(
        f"WARNING: verl will filter prompts longer than max_prompt_length={max_prompt_length}; "
        f"counts={overlong_counts}, examples={overlong}",
        file=sys.stderr,
    )
eligible_train_rows = len(train_rows) - overlong_counts["train"]
eligible_val_rows = len(val_rows) - overlong_counts["validation"]
eligible_count_errors = []
if eligible_train_rows != expected_eligible_train_rows:
    eligible_count_errors.append(
        f"eligible train rows={eligible_train_rows}, expected={expected_eligible_train_rows}"
    )
if eligible_val_rows != expected_eligible_val_rows:
    eligible_count_errors.append(
        f"eligible validation rows={eligible_val_rows}, expected={expected_eligible_val_rows}"
    )
if eligible_count_errors and strict_expected_counts:
    raise ValueError("; ".join(eligible_count_errors))
if eligible_count_errors:
    print("WARNING:", "; ".join(eligible_count_errors), file=sys.stderr)

bridge_name = "skipped"
engine_name = "skipped"
visible_npus = "not checked"
if not skip_runtime_preflight:
    import verl.workers.engine  # noqa: F401
    from verl.models.mcore.mbridge import AutoBridge
    from verl.utils.device import get_device_name
    from verl.workers.engine.base import EngineRegistry
    from mbridge.models import Qwen2Bridge

    supported_models = AutoBridge.list_supported_models()
    if config.model_type not in supported_models:
        raise RuntimeError(
            f"MBridge does not register model_type={config.model_type!r}; supported={supported_models}"
        )
    bridge_name = f"{Qwen2Bridge.__module__}.{Qwen2Bridge.__name__} (registered:{config.model_type})"

    engine_cls = EngineRegistry.get_engine_cls(model_type="language_model", backend="megatron")
    engine_name = f"{engine_cls.__module__}.{engine_cls.__name__}"
    if require_npu:
        import torch_npu  # noqa: F401

        if not torch.npu.is_available():
            raise RuntimeError("torch.npu is unavailable")
        if get_device_name() != "npu":
            raise RuntimeError(f"verl selected device={get_device_name()!r}, expected 'npu'")
        visible_npus = torch.npu.device_count()
        if visible_npus < expected_local_npus:
            raise RuntimeError(
                f"Only {visible_npus} NPUs are visible, expected at least {expected_local_npus}"
            )
        if "mindspeed" not in engine_cls.__module__:
            raise RuntimeError(f"verl did not select the MindSpeed engine: {engine_name}")

print(
    "GRPO preflight:",
    {
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "train_integrity": train_stats,
        "validation_integrity": val_stats,
        "cross_split_prompt_keys": len(cross_split_keys),
        "cross_split_ground_truth_conflict_keys": len(cross_split_ground_truth_conflicts),
        "max_prompt_tokens": max_seen,
        "overlong_prompt_rows": overlong_counts,
        "eligible_train_rows": eligible_train_rows,
        "eligible_validation_rows": eligible_val_rows,
        "generation_prefix": generation_prefix,
        "protocol_eos": protocol_eos,
        "protocol_eos_id": protocol_eos_id,
        "model_type": config.model_type,
        "architectures": config.architectures,
        "tie_word_embeddings": bool(getattr(config, "tie_word_embeddings", False)),
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": num_kv_heads,
        "vocab_size": config.vocab_size,
        "tokenizer_size": len(tokenizer),
        "special_token_ids": sorted(special_ids),
        "max_position_embeddings": config.max_position_embeddings,
        "tokenizer_model_max_length": tokenizer_model_max_length,
        "bridge": bridge_name,
        "engine": engine_name,
        "visible_npus": visible_npus,
    },
)
PY

MODEL_CHAT_PROTOCOL=$(python3 - "$MODEL_PATH/tokenizer_config.json" <<'PY'
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
template = config.get("chat_template")
if not isinstance(template, str):
    raise RuntimeError("tokenizer_config.json must define a string chat_template")
if "<｜Assistant｜><think>" in template and "<｜end▁of▁sentence｜>" in template:
    print("deepseek_distill")
elif "<|im_start|>assistant" in template and "<|im_end|>" in template:
    print("chatml")
else:
    raise RuntimeError("Unsupported chat-template protocol for this GRPO script")
PY
)

AGENT_LOOP_CONFIG_PATH=
AGENT_LOOP_NAME=
RAY_RUNTIME_PYTHONPATH=
# The pinned verl resolves Continuous Token builders from config.model_type. DeepSeek-R1-Distill-Qwen
# reports qwen2 for its architecture but uses DeepSeek's special-token protocol, so route only its
# single-turn agent loop through the DeepSeek builder. OpenMath is genuine Qwen/ChatML and stays native.
case "$MODEL_CHAT_PROTOCOL" in
    deepseek_distill)
        prepare_deepseek_distill_agent_loop_compat
        AGENT_LOOP_CONFIG_PATH=$DEEPSEEK_AGENT_LOOP_CONFIG
        AGENT_LOOP_NAME=deepseek_distill_single_turn_agent
        RAY_RUNTIME_PYTHONPATH=$GRPO_RAY_PYTHONPATH
        ;;
    chatml)
        verify_chatml_continuous_token_builder
        ;;
    *)
        die "Unsupported MODEL_CHAT_PROTOCOL=$MODEL_CHAT_PROTOCOL"
        ;;
esac

if is_true "$PREPARE_ONLY"; then
    echo "PREPARE_ONLY complete; no GRPO process was launched."
    exit 0
fi

################################ Hydra arguments

DATA_ARGS=(
    "data.train_files=${TRAIN_FILE}"
    "data.val_files=${VAL_FILE}"
    "data.prompt_key=prompt"
    "data.train_batch_size=${TRAIN_BATCH_SIZE}"
    "data.val_batch_size=${VAL_BATCH_SIZE}"
    "data.train_max_samples=-1"
    "data.val_max_samples=-1"
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
    "data.max_response_length=${MAX_RESPONSE_LENGTH}"
    # Four source train rows exceed 1,024 tokens with both supported tokenizers.
    # Filtering preserves the exact text of every retained prompt; truncation stays forbidden.
    "data.filter_overlong_prompts=True"
    "data.truncation=error"
    "data.shuffle=True"
    "data.seed=${SEED}"
    "data.validation_shuffle=False"
)

MODEL_ARGS=(
    "actor_rollout_ref.model.path=${MODEL_PATH}"
    "actor_rollout_ref.model.trust_remote_code=False"
    "actor_rollout_ref.model.use_remove_padding=True"
)

ALGORITHM_ARGS=(
    "algorithm.adv_estimator=grpo"
    "algorithm.norm_adv_by_std_in_grpo=True"
    "algorithm.use_kl_in_reward=False"
    "algorithm.kl_ctrl.kl_coef=0.0"
)

ACTOR_ARGS=(
    "actor_rollout_ref.actor.strategy=megatron"
    "actor_rollout_ref.actor.use_torch_compile=False"
    "actor_rollout_ref.actor.use_dynamic_bsz=True"
    "actor_rollout_ref.actor.ppo_epochs=1"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}"
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}"
    "actor_rollout_ref.actor.loss_agg_mode=token-mean"
    "actor_rollout_ref.actor.clip_ratio=0.2"
    "actor_rollout_ref.actor.clip_ratio_low=0.2"
    "actor_rollout_ref.actor.clip_ratio_high=0.2"
    "actor_rollout_ref.actor.optim.lr=${ACTOR_LR}"
    "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05"
    "actor_rollout_ref.actor.optim.weight_decay=0.01"
    "actor_rollout_ref.actor.optim.clip_grad=1.0"
    "actor_rollout_ref.actor.use_kl_loss=True"
    "actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}"
    "actor_rollout_ref.actor.kl_loss_type=low_var_kl"
    "actor_rollout_ref.actor.entropy_coeff=0"
    "actor_rollout_ref.actor.data_loader_seed=${SEED}"
    "actor_rollout_ref.actor.megatron.use_mbridge=True"
    "actor_rollout_ref.actor.megatron.vanilla_mbridge=True"
    "actor_rollout_ref.actor.megatron.use_remove_padding=True"
    "actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${ACTOR_TP}"
    "actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${ACTOR_PP}"
    "actor_rollout_ref.actor.megatron.context_parallel_size=${ACTOR_CP}"
    "actor_rollout_ref.actor.megatron.sequence_parallel=True"
    "actor_rollout_ref.actor.megatron.expert_model_parallel_size=1"
    "actor_rollout_ref.actor.megatron.use_distributed_optimizer=True"
    "actor_rollout_ref.actor.megatron.use_megatron_fsdp=False"
    "actor_rollout_ref.actor.megatron.use_dist_checkpointing=False"
    "actor_rollout_ref.actor.megatron.param_offload=False"
    "actor_rollout_ref.actor.megatron.optimizer_offload=False"
    "actor_rollout_ref.actor.megatron.dtype=bfloat16"
    "+actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_algo=megatron_cp_algo"
    "+actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True"
    "actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform"
    "actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full"
    "actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1"
    "+actor_rollout_ref.actor.checkpoint.mbridge_config.distributed_filesystem=True"
    "+actor_rollout_ref.actor.checkpoint.mbridge_config.memory_efficient=True"
)

REF_ARGS=(
    "actor_rollout_ref.ref.strategy=megatron"
    "actor_rollout_ref.ref.use_torch_compile=False"
    "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True"
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1"
    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}"
    "actor_rollout_ref.ref.megatron.use_mbridge=True"
    "actor_rollout_ref.ref.megatron.vanilla_mbridge=True"
    "actor_rollout_ref.ref.megatron.use_remove_padding=True"
    "actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${ACTOR_TP}"
    "actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${ACTOR_PP}"
    "actor_rollout_ref.ref.megatron.context_parallel_size=${ACTOR_CP}"
    "actor_rollout_ref.ref.megatron.sequence_parallel=True"
    "actor_rollout_ref.ref.megatron.expert_model_parallel_size=1"
    "actor_rollout_ref.ref.megatron.use_megatron_fsdp=False"
    "actor_rollout_ref.ref.megatron.use_dist_checkpointing=False"
    "actor_rollout_ref.ref.megatron.param_offload=True"
)

ROLLOUT_ARGS=(
    "actor_rollout_ref.rollout.name=vllm"
    "actor_rollout_ref.rollout.mode=async"
    "actor_rollout_ref.rollout.dtype=bfloat16"
    "actor_rollout_ref.rollout.load_format=dummy"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}"
    "actor_rollout_ref.rollout.pipeline_model_parallel_size=${ROLLOUT_PP}"
    "actor_rollout_ref.rollout.data_parallel_size=${ROLLOUT_DP}"
    "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
    "actor_rollout_ref.rollout.temperature=1.0"
    "actor_rollout_ref.rollout.top_p=1.0"
    "actor_rollout_ref.rollout.top_k=-1"
    "actor_rollout_ref.rollout.do_sample=True"
    "actor_rollout_ref.rollout.ignore_eos=False"
    "actor_rollout_ref.rollout.seed=${SEED}"
    "actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}"
    "actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS}"
    "actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}"
    "actor_rollout_ref.rollout.calculate_log_probs=True"
    "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1"
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}"
    "actor_rollout_ref.rollout.enable_chunked_prefill=True"
    "actor_rollout_ref.rollout.enable_prefix_caching=True"
    "actor_rollout_ref.rollout.free_cache_engine=True"
    "actor_rollout_ref.rollout.enforce_eager=False"
    "actor_rollout_ref.rollout.cudagraph_capture_sizes=[1,2,4,8]"
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode=FULL_DECODE_ONLY"
    "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=1024"
    "actor_rollout_ref.rollout.val_kwargs.n=1"
    "actor_rollout_ref.rollout.val_kwargs.do_sample=False"
    "actor_rollout_ref.rollout.val_kwargs.temperature=0.0"
    "actor_rollout_ref.rollout.val_kwargs.top_p=1.0"
    "actor_rollout_ref.rollout.val_kwargs.top_k=-1"
)

TRAINER_ARGS=(
    "actor_rollout_ref.nccl_timeout=7200"
    "reward.reward_manager.name=naive"
    "trainer.device=npu"
    "trainer.nnodes=${NNODES}"
    "trainer.n_gpus_per_node=${NPUS_PER_NODE}"
    "trainer.balance_batch=True"
    'trainer.logger=["console","wandb"]'
    "trainer.project_name=${PROJECT_NAME}"
    "trainer.experiment_name=${EXPERIMENT_NAME}"
    "trainer.total_epochs=${TOTAL_EPOCHS}"
    "trainer.total_training_steps=${TOTAL_TRAINING_STEPS}"
    "trainer.val_before_train=False"
    "trainer.test_freq=${TEST_FREQ}"
    "trainer.save_freq=${SAVE_FREQ}"
    "trainer.log_val_generations=0"
    "trainer.default_local_dir=${CKPT_DIR}"
    "trainer.resume_mode=auto"
    "trainer.max_actor_ckpt_to_keep=2"
)

COMMAND=(
    python3 -m verl.trainer.main_ppo
    model_engine=megatron
    "${DATA_ARGS[@]}"
    "${MODEL_ARGS[@]}"
    "${ALGORITHM_ARGS[@]}"
    "${ACTOR_ARGS[@]}"
    "${REF_ARGS[@]}"
    "${ROLLOUT_ARGS[@]}"
    "${TRAINER_ARGS[@]}"
)
if [[ -n "$AGENT_LOOP_CONFIG_PATH" ]]; then
    COMMAND+=(
        "actor_rollout_ref.rollout.agent.agent_loop_config_path=${AGENT_LOOP_CONFIG_PATH}"
        "actor_rollout_ref.rollout.agent.default_agent_loop=${AGENT_LOOP_NAME}"
        "+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=${RAY_RUNTIME_PYTHONPATH}"
    )
fi
if [[ -n "${RAY_ADDRESS:-}" ]]; then
    COMMAND+=("+ray_kwargs.ray_init.address=${RAY_ADDRESS}")
fi
COMMAND+=("$@")

echo "Experiment: $WANDB_ENTITY/$PROJECT_NAME/$EXPERIMENT_NAME"
echo "Model: $MODEL_PATH"
echo "Data: train=$TRAIN_FILE validation=$VAL_FILE"
echo "Actor topology: TP=$ACTOR_TP PP=$ACTOR_PP CP=$ACTOR_CP DP=$ACTOR_DP"
echo "Rollout topology: replicas=$ROLLOUT_REPLICAS TP=$ROLLOUT_TP PP=$ROLLOUT_PP internal-DP=$ROLLOUT_DP"
echo "Batching: prompts=$TRAIN_BATCH_SIZE rollouts_per_prompt=$ROLLOUT_N responses=$TOTAL_ROLLOUT_SAMPLES ppo_prompt_minibatch=$PPO_MINI_BATCH_SIZE"
echo "Sequence lengths: prompt=$MAX_PROMPT_LENGTH response=$MAX_RESPONSE_LENGTH total=$MAX_MODEL_LEN"
echo "Schedule: epochs=$TOTAL_EPOCHS explicit_steps=$TOTAL_TRAINING_STEPS save_every=$SAVE_FREQ validate_every=$TEST_FREQ"
echo "Audited eligible rows: train=$EXPECTED_ELIGIBLE_TRAIN_ROWS validation=$EXPECTED_ELIGIBLE_VAL_ROWS expected_steps_per_epoch=$EXPECTED_STEPS_PER_EPOCH"
echo "Checkpoint directory: $CKPT_DIR"
echo "Log file: $LOG_FILE"
print_command "${COMMAND[@]}"

if is_true "$DRY_RUN"; then
    echo "DRY_RUN complete; no GRPO process was launched."
    exit 0
fi

"${COMMAND[@]}" 2>&1 | tee "$LOG_FILE"
