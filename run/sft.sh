#!/usr/bin/env bash

# DAPO-Math-17K SFT baseline for the Ascend verl image documented below.
# Supported models:
#   - deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
#   - nvidia/OpenMath-Nemotron-1.5B
#
# The embedded preprocessing step joins Yang-Zhou's rejection-distilled
# responses against the exact RL train/test membership. Missing targets are
# reported and skipped; RL validation prompts are never moved into SFT train.

# Re-enter Bash when a scheduler invokes this file with `sh run/sft.sh`.
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

################################ paths and experiment identity

VERL_ROOT=${VERL_ROOT:-/verl}
[[ -d "$VERL_ROOT" ]] || die "VERL_ROOT does not exist: $VERL_ROOT"

DATA_ROOT=${DATA_ROOT:-/opt/huawei/dataset/jnguan}
RL_DATA_DIR=${RL_DATA_DIR:-$DATA_ROOT/datasets/open-r1/DAPO-Math-17k-Processed-verl--qwen3}
RL_TRAIN_FILE=${RL_TRAIN_FILE:-$RL_DATA_DIR/train.parquet}
RL_VAL_FILE=${RL_VAL_FILE:-$RL_DATA_DIR/test.parquet}
SFT_SOURCE_JSON=${SFT_SOURCE_JSON:-$DATA_ROOT/datasets/Yang-Zhou/DAPO-Math-17k-Qwen3-235B-A22B-Thinking-2507-rejection-distill/dapo_distill_boxed.json}

MODEL_PATH=${MODEL_PATH:-$DATA_ROOT/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}
MODEL_NAME=${MODEL_NAME:-$(basename -- "$MODEL_PATH")}
MODEL_SLUG=$(printf '%s' "$MODEL_NAME" | tr '[:upper:]' '[:lower:]' | tr -cs '[:alnum:]' '-')
MODEL_SLUG=${MODEL_SLUG#-}
MODEL_SLUG=${MODEL_SLUG%-}
[[ -n "$MODEL_SLUG" ]] || die "MODEL_NAME must contain at least one alphanumeric character."

OUTPUT_ROOT=${OUTPUT_ROOT:-$DATA_ROOT/outputs/dapo_math_17k_baselines}
SFT_CACHE_DIR=${SFT_CACHE_DIR:-$OUTPUT_ROOT/prepared_data/dapo_math_17k_sft}
PROJECT_NAME=${PROJECT_NAME:-DMPD}
EXPERIMENT_TIMESTAMP=${EXPERIMENT_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-${MODEL_SLUG}-sft-${EXPERIMENT_TIMESTAMP}}
CKPT_DIR=${CKPT_DIR:-$OUTPUT_ROOT/checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME}
LOG_DIR=${LOG_DIR:-$OUTPUT_ROOT/logs}

export WANDB_ENTITY=${WANDB_ENTITY:-jubgjf}
export WANDB_PROJECT=${WANDB_PROJECT:-$PROJECT_NAME}
export WANDB_NAME=${WANDB_NAME:-$EXPERIMENT_NAME}

[[ -d "$MODEL_PATH" ]] || die "Missing model directory: $MODEL_PATH"
[[ -f "$RL_TRAIN_FILE" ]] || die "Missing RL training parquet: $RL_TRAIN_FILE"
[[ -f "$RL_VAL_FILE" ]] || die "Missing RL validation parquet: $RL_VAL_FILE"
[[ -f "$SFT_SOURCE_JSON" ]] || die "Missing Yang-Zhou SFT JSON: $SFT_SOURCE_JSON"

mkdir -p "$CKPT_DIR" "$LOG_DIR" "$SFT_CACHE_DIR"

################################ cluster and parallel topology

NNODES=${NNODES:-${VC_WORKER_NUM:-1}}
NPUS_PER_NODE=${NPUS_PER_NODE:-8}
NODE_RANK=${NODE_RANK:-${VC_TASK_INDEX:-0}}
MASTER_PORT=${MASTER_PORT:-29500}

check_positive_integer NNODES "$NNODES"
check_positive_integer NPUS_PER_NODE "$NPUS_PER_NODE"
[[ "$NODE_RANK" =~ ^[0-9]+$ ]] || die "NODE_RANK must be a non-negative integer, got: $NODE_RANK"
((NODE_RANK < NNODES)) || die "NODE_RANK=$NODE_RANK must be smaller than NNODES=$NNODES"

if ((NNODES > 1)); then
    [[ -n "${MASTER_ADDR:-}" ]] || die "Multi-node SFT requires MASTER_ADDR."
else
    MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
fi

WORLD_SIZE=$((NNODES * NPUS_PER_NODE))
ACTOR_TP=${ACTOR_TP:-1}
ACTOR_PP=${ACTOR_PP:-1}
ACTOR_CP=${ACTOR_CP:-2}

for setting in ACTOR_TP ACTOR_PP ACTOR_CP; do
    check_positive_integer "$setting" "${!setting}"
done

MODEL_PARALLEL_SIZE=$((ACTOR_TP * ACTOR_PP * ACTOR_CP))
((WORLD_SIZE % MODEL_PARALLEL_SIZE == 0)) || die \
    "WORLD_SIZE=$WORLD_SIZE must be divisible by TP*PP*CP=$MODEL_PARALLEL_SIZE."
ACTOR_DP=$((WORLD_SIZE / MODEL_PARALLEL_SIZE))

################################ lengths, batches, and optimizer

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-23552}
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-1}
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-32768}

for setting in \
    MAX_PROMPT_LENGTH \
    MAX_RESPONSE_LENGTH \
    TRAIN_BATCH_SIZE \
    MICRO_BATCH_SIZE_PER_GPU \
    MAX_TOKEN_LEN_PER_GPU; do
    check_positive_integer "$setting" "${!setting}"
done

((ACTOR_CP == 1 || MAX_MODEL_LEN % (2 * ACTOR_CP) == 0)) || die \
    "MAX_MODEL_LEN=$MAX_MODEL_LEN must be divisible by 2*ACTOR_CP=$((2 * ACTOR_CP))."
((TRAIN_BATCH_SIZE % ACTOR_DP == 0)) || die \
    "TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE must be divisible by ACTOR_DP=$ACTOR_DP."

SFT_LR=${SFT_LR:-1e-5}
MIN_LR=${MIN_LR:-1e-6}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-null}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-20}
SEED=${SEED:-42}

EXPECTED_RL_TRAIN_ROWS=${EXPECTED_RL_TRAIN_ROWS:-13918}
EXPECTED_RL_VAL_ROWS=${EXPECTED_RL_VAL_ROWS:-3480}
# The current upstream files produce 12,217/3,028 rows.  These counts include
# repeated rows already present in the authoritative RL parquet split.  The
# earlier 11,708/2,927 estimate is not reproduced by the requested exact join
# over the complete 15,129-row Yang file.
EXPECTED_SFT_TRAIN_ROWS=${EXPECTED_SFT_TRAIN_ROWS:-12217}
EXPECTED_SFT_VAL_ROWS=${EXPECTED_SFT_VAL_ROWS:-3028}
STRICT_EXPECTED_COUNTS=${STRICT_EXPECTED_COUNTS:-1}
PREPARE_TIMEOUT_SECONDS=${PREPARE_TIMEOUT_SECONDS:-3600}
check_positive_integer PREPARE_TIMEOUT_SECONDS "$PREPARE_TIMEOUT_SECONDS"

setup_ascend_environment
cd "$VERL_ROOT"
check_runtime_versions

################################ exact split join, tokenizer patch, and token audit

TOKENIZER_PATH_FILE=$SFT_CACHE_DIR/tokenizer_path.${MODEL_SLUG}.txt

python3 - \
    "$MODEL_PATH" \
    "$RL_TRAIN_FILE" \
    "$RL_VAL_FILE" \
    "$SFT_SOURCE_JSON" \
    "$SFT_CACHE_DIR" \
    "$MODEL_SLUG" \
    "$MAX_MODEL_LEN" \
    "$EXPECTED_RL_TRAIN_ROWS" \
    "$EXPECTED_RL_VAL_ROWS" \
    "$EXPECTED_SFT_TRAIN_ROWS" \
    "$EXPECTED_SFT_VAL_ROWS" \
    "$STRICT_EXPECTED_COUNTS" \
    "$PREPARE_TIMEOUT_SECONDS" \
    "$TOKENIZER_PATH_FILE" <<'PY'
import hashlib
import json
import os
import re
import shutil
import socket
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import ijson
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoConfig, AutoTokenizer

(
    model_path,
    rl_train_file,
    rl_val_file,
    sft_source_json,
    cache_dir,
    model_slug,
    max_model_len,
    expected_rl_train_rows,
    expected_rl_val_rows,
    expected_sft_train_rows,
    expected_sft_val_rows,
    strict_expected_counts,
    prepare_timeout_seconds,
    tokenizer_path_file,
) = sys.argv[1:]

cache_dir = Path(cache_dir)
max_model_len = int(max_model_len)
expected_rl_train_rows = int(expected_rl_train_rows)
expected_rl_val_rows = int(expected_rl_val_rows)
expected_sft_train_rows = int(expected_sft_train_rows)
expected_sft_val_rows = int(expected_sft_val_rows)
strict_expected_counts = strict_expected_counts.lower() in {"1", "true", "yes"}
prepare_timeout_seconds = int(prepare_timeout_seconds)
tokenizer_path_file = Path(tokenizer_path_file)

RULES_VERSION = 7
KNOWN_FORMAT_SUFFIXES = (
    "Please reason step by step, and put your final answer within \\boxed{}.",
    "Let's think step by step and output the final answer within \\boxed{}.",
    'Remember to put your answer on its own line after "Answer:".',
)

cache_dir.mkdir(parents=True, exist_ok=True)
lock_path = cache_dir / ".prepare.lock"
lock_fd = None
deadline = time.monotonic() + prepare_timeout_seconds
last_notice = 0.0
while lock_fd is None:
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.write(
            lock_fd,
            json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "started": time.time()}).encode(),
        )
        os.fsync(lock_fd)
    except FileExistsError:
        now = time.monotonic()
        if now >= deadline:
            owner = lock_path.read_text(errors="replace") if lock_path.exists() else "unknown"
            raise TimeoutError(f"Timed out waiting for {lock_path}; recorded owner={owner}")
        if now - last_notice >= 60:
            print(f"Waiting for shared SFT preparation lock: {lock_path}", flush=True)
            last_notice = now
        time.sleep(5)


def atomic_json(path, payload):
    path = Path(path)
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def atomic_text(path, text):
    path = Path(path)
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temp.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def file_stat(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_json_like(value, field_name, row_index):
    if hasattr(value, "as_py"):
        value = value.as_py()
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


def normalize_ground_truth(row, row_index):
    reward_model = parse_json_like(row.get("reward_model"), "reward_model", row_index)
    answer = reward_model.get("ground_truth") if isinstance(reward_model, Mapping) else None
    if answer is None:
        answer = row.get("solution")
    if answer is None or not str(answer).strip():
        raise ValueError(f"row {row_index} has no reward_model.ground_truth")
    answer = str(answer).strip()
    if answer.startswith("\\boxed{") and answer.endswith("}"):
        answer = answer[len("\\boxed{") : -1].strip()
    return answer


def prompt_match_key(text):
    text = text.strip()
    changed = True
    while changed:
        changed = False
        for suffix in KNOWN_FORMAT_SUFFIXES:
            if text.endswith(suffix):
                text = text[: -len(suffix)].rstrip()
                changed = True
    if not text:
        raise ValueError("Prompt became empty after removing a known formatting suffix")
    return text


def prompt_format(messages, row_index):
    text = messages[-1]["content"]
    if "\\boxed" in text or re.search(r"\bboxed(?:[- ]answer)?\b", text, flags=re.IGNORECASE):
        return "boxed"
    if "Answer:" in text:
        return "answer"
    raise ValueError(f"row {row_index} has an unknown answer format in its exact RL prompt")


def row_identifier(row, split, row_index):
    extra_info = parse_json_like(row.get("extra_info"), "extra_info", row_index)
    if isinstance(extra_info, Mapping) and extra_info.get("index") is not None:
        return str(extra_info["index"])
    return f"{split}:{row_index}"


def load_rl_split(path, split):
    rows = pq.read_table(path).to_pylist()
    index = {}
    for row_index, row in enumerate(rows):
        messages = normalize_messages(row.get("prompt"), row_index)
        key = prompt_match_key(messages[-1]["content"])
        entry = {
            "messages": messages,
            "ground_truth": normalize_ground_truth(row, row_index),
            "prompt_format": prompt_format(messages, row_index),
            "rl_index": row_identifier(row, split, row_index),
            "row_index": row_index,
        }
        index.setdefault(key, []).append(entry)
    duplicate_keys = {key: entries for key, entries in index.items() if len(entries) > 1}
    ground_truth_conflicts = {
        key: entries
        for key, entries in duplicate_keys.items()
        if len({entry["ground_truth"] for entry in entries}) > 1
    }
    stats = {
        "rows": len(rows),
        "unique_prompt_keys": len(index),
        "duplicate_prompt_keys": len(duplicate_keys),
        "duplicate_rows_beyond_first": sum(len(entries) - 1 for entries in duplicate_keys.values()),
        "ground_truth_conflict_keys": len(ground_truth_conflicts),
        "ground_truth_conflicts": [
            {
                "prompt_sha256": hashlib.sha256(key.encode("utf-8")).hexdigest(),
                "rows": [
                    {
                        "row_index": entry["row_index"],
                        "rl_index": entry["rl_index"],
                        "ground_truth": entry["ground_truth"],
                    }
                    for entry in entries
                ],
            }
            for key, entries in ground_truth_conflicts.items()
        ],
    }
    return rows, index, stats


def append_matching_rows(destination, matched, split, key, rl_rows, sample, source_index):
    ground_truths = {row["ground_truth"] for row in rl_rows}
    if len(ground_truths) > 1:
        row_summary = [(row["row_index"], row["ground_truth"]) for row in rl_rows]
        raise ValueError(
            f"Yang row {source_index} matches conflicting {split} RL rows: {row_summary}"
        )
    for rl_row in rl_rows:
        target = normalize_target(
            sample.get("output"),
            rl_row["prompt_format"],
            rl_row["ground_truth"],
            source_index,
        )
        messages = [dict(message) for message in rl_row["messages"]]
        messages.append({"role": "assistant", "content": target})
        destination.append(
            {
                "messages": messages,
                "ground_truth": rl_row["ground_truth"],
                "split": split,
                "rl_index": rl_row["rl_index"],
                "prompt_format": rl_row["prompt_format"],
            }
        )
    matched.add(key)


def has_canonical_terminal(text, prompt_style, answer):
    escaped = re.escape(answer)
    if prompt_style == "boxed":
        pattern = rf"(?:\$\$|\$)?\s*\\boxed\{{{escaped}\}}\s*(?:\$\$|\$)?[.!]?\s*$"
    else:
        pattern = rf"Answer:\s*{escaped}\s*[.!]?\s*$"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def normalize_target(output, prompt_style, answer, source_index):
    if not isinstance(output, str) or not output.strip():
        raise ValueError(f"Yang row {source_index} has an empty output")
    if "<think>" not in output or "</think>" not in output:
        raise ValueError(f"Yang row {source_index} does not contain a complete <think> block")
    if output.index("<think>") > output.index("</think>"):
        raise ValueError(f"Yang row {source_index} has malformed thinking tags")
    output = output.rstrip()
    if not has_canonical_terminal(output, prompt_style, answer):
        terminal = f"\\boxed{{{answer}}}" if prompt_style == "boxed" else f"Answer: {answer}"
        output = f"{output}\n\n{terminal}"
    return output


def write_parquet_atomic(path, rows):
    message_type = pa.struct([pa.field("role", pa.string()), pa.field("content", pa.string())])
    schema = pa.schema(
        [
            pa.field("messages", pa.list_(message_type)),
            pa.field("ground_truth", pa.string()),
            pa.field("split", pa.string()),
            pa.field("rl_index", pa.string()),
            pa.field("prompt_format", pa.string()),
        ]
    )
    table = pa.Table.from_pylist(rows, schema=schema)
    temp = Path(path).with_name(f".{Path(path).name}.tmp.{os.getpid()}")
    pq.write_table(table, temp, compression="zstd")
    os.replace(temp, path)


def count_check(label, actual, expected):
    if actual == expected:
        return
    message = f"{label}={actual}, expected={expected}"
    if strict_expected_counts:
        raise ValueError(message)
    print(f"WARNING: {message}", file=sys.stderr)


try:
    train_output = cache_dir / "train.parquet"
    val_output = cache_dir / "validation.parquet"
    dataset_manifest_path = cache_dir / "dataset_manifest.json"
    source_hashes = {
        "rl_train": sha256_file(rl_train_file),
        "rl_validation": sha256_file(rl_val_file),
        "yang_source": sha256_file(sft_source_json),
    }
    source_signature = {
        "rules_version": RULES_VERSION,
        "rl_train": {**file_stat(rl_train_file), "sha256": source_hashes["rl_train"]},
        "rl_validation": {
            **file_stat(rl_val_file),
            "sha256": source_hashes["rl_validation"],
        },
        "yang_source": {**file_stat(sft_source_json), "sha256": source_hashes["yang_source"]},
    }

    dataset_manifest = None
    if dataset_manifest_path.exists() and train_output.exists() and val_output.exists():
        candidate = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
        if candidate.get("source_signature") == source_signature:
            dataset_manifest = candidate
            print("Reusing cached SFT parquet files.", flush=True)

    if dataset_manifest is None:
        train_rows_raw, train_index, train_stats = load_rl_split(rl_train_file, "train")
        val_rows_raw, val_index, val_stats = load_rl_split(rl_val_file, "validation")
        train_ground_truth_conflict_keys = {
            key
            for key, rows in train_index.items()
            if len({row["ground_truth"] for row in rows}) > 1
        }
        val_ground_truth_conflict_keys = {
            key
            for key, rows in val_index.items()
            if len({row["ground_truth"] for row in rows}) > 1
        }
        rl_cross_split_keys = set(train_index) & set(val_index)
        rl_cross_split_ground_truth_conflicts = {
            key
            for key in rl_cross_split_keys
            if {row["ground_truth"] for row in train_index[key]}
            != {row["ground_truth"] for row in val_index[key]}
        }
        if train_stats["duplicate_prompt_keys"] or val_stats["duplicate_prompt_keys"]:
            print(
                "WARNING: preserving duplicate rows from the authoritative RL parquets: "
                f"train_keys={train_stats['duplicate_prompt_keys']}, "
                f"validation_keys={val_stats['duplicate_prompt_keys']}",
                file=sys.stderr,
            )
        if rl_cross_split_keys:
            print(
                "WARNING: authoritative RL parquets contain normalized prompts in both splits; "
                f"preserving their memberships exactly (keys={len(rl_cross_split_keys)}).",
                file=sys.stderr,
            )

        count_check("RL train rows", len(train_rows_raw), expected_rl_train_rows)
        count_check("RL validation rows", len(val_rows_raw), expected_rl_val_rows)

        output_train = []
        output_val = []
        matched_train = set()
        matched_val = set()
        seen_yang_keys = set()
        unmatched_yang = 0
        source_rows = 0

        with open(sft_source_json, "rb") as handle:
            for source_index, sample in enumerate(ijson.items(handle, "item")):
                source_rows += 1
                instruction = sample.get("instruction")
                input_text = sample.get("input")
                if not isinstance(instruction, str) or not instruction.strip():
                    raise ValueError(f"Yang row {source_index} has an empty instruction")
                if input_text not in (None, ""):
                    raise ValueError(
                        f"Yang row {source_index} has a non-empty input field; exact matching rule is undefined"
                    )
                key = prompt_match_key(instruction)
                if key in seen_yang_keys:
                    raise ValueError(f"Yang source has a duplicate normalized prompt at row {source_index}")
                seen_yang_keys.add(key)

                destinations = 0
                if key in train_index:
                    append_matching_rows(
                        output_train,
                        matched_train,
                        "train",
                        key,
                        train_index[key],
                        sample,
                        source_index,
                    )
                    destinations += 1
                if key in val_index:
                    append_matching_rows(
                        output_val,
                        matched_val,
                        "validation",
                        key,
                        val_index[key],
                        sample,
                        source_index,
                    )
                    destinations += 1
                if destinations == 0:
                    unmatched_yang += 1

        count_check("SFT train rows", len(output_train), expected_sft_train_rows)
        count_check("SFT validation rows", len(output_val), expected_sft_val_rows)
        sft_cross_split_keys = matched_train & matched_val
        matched_conflict_keys = (
            train_ground_truth_conflict_keys
            | val_ground_truth_conflict_keys
            | rl_cross_split_ground_truth_conflicts
        ) & (matched_train | matched_val)
        if matched_conflict_keys:
            raise AssertionError("Matched ground-truth conflicts should have failed during row expansion")

        write_parquet_atomic(train_output, output_train)
        write_parquet_atomic(val_output, output_val)

        dataset_manifest = {
            "source_signature": source_signature,
            "source_sha256": source_hashes,
            "counts": {
                "rl_train": len(train_rows_raw),
                "rl_validation": len(val_rows_raw),
                "yang_source": source_rows,
                "sft_train": len(output_train),
                "sft_validation": len(output_val),
                "matched_rl_train_prompt_keys": len(matched_train),
                "matched_rl_validation_prompt_keys": len(matched_val),
                "missing_rl_train_prompt_keys": len(train_index) - len(matched_train),
                "missing_rl_validation_prompt_keys": len(val_index) - len(matched_val),
                "missing_rl_train_rows": sum(
                    len(rows) for key, rows in train_index.items() if key not in matched_train
                ),
                "missing_rl_validation_rows": sum(
                    len(rows) for key, rows in val_index.items() if key not in matched_val
                ),
                "unmatched_yang_rows": unmatched_yang,
            },
            "integrity": {
                "rl_train": train_stats,
                "rl_validation": val_stats,
                "rl_cross_split_prompt_keys": len(rl_cross_split_keys),
                "rl_cross_split_ground_truth_conflict_keys": len(
                    rl_cross_split_ground_truth_conflicts
                ),
                "sft_cross_split_prompt_keys": len(sft_cross_split_keys),
                "cross_split_policy": "preserve_authoritative_rl_membership",
            },
            "matching": {
                "mode": "exact_after_allowlisted_suffix_removal",
                "known_suffixes": list(KNOWN_FORMAT_SUFFIXES),
                "fuzzy_matching": False,
                "missing_target_policy": "skip",
                "duplicate_rl_row_policy": "repeat_the_Yang_target_for_each_consistent_RL_row",
                "matched_ground_truth_conflict_policy": "fail",
                "split_authority": "RL parquet membership",
            },
            "outputs": {
                "train": str(train_output.resolve()),
                "validation": str(val_output.resolve()),
            },
        }
        atomic_json(dataset_manifest_path, dataset_manifest)

    counts = dataset_manifest["counts"]
    count_check("RL train rows", counts["rl_train"], expected_rl_train_rows)
    count_check("RL validation rows", counts["rl_validation"], expected_rl_val_rows)
    count_check("SFT train rows", counts["sft_train"], expected_sft_train_rows)
    count_check("SFT validation rows", counts["sft_validation"], expected_sft_val_rows)

    source_tokenizer_files = []
    tokenizer_digest = hashlib.sha256()
    for name in (
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "added_tokens.json",
    ):
        path = Path(model_path) / name
        if path.exists():
            source_tokenizer_files.append(name)
            tokenizer_digest.update(name.encode())
            tokenizer_digest.update(path.read_bytes())
    if not source_tokenizer_files:
        raise FileNotFoundError(f"No tokenizer/config files found under {model_path}")
    source_tokenizer_sha256 = tokenizer_digest.hexdigest()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
    source_tokenizer_model_max_length = int(tokenizer.model_max_length)
    if source_tokenizer_model_max_length < max_model_len:
        print(
            "WARNING: tokenizer metadata advertises model_max_length="
            f"{source_tokenizer_model_max_length}, but model config supports "
            f"{config.max_position_embeddings}; the temporary tokenizer copy will use {max_model_len}.",
            file=sys.stderr,
        )
        tokenizer.model_max_length = max_model_len
    probe_marker = "SFT_PRIVATE_REASONING_MARKER_7F2A"
    probe_messages = [
        {"role": "user", "content": "What is 1+1?"},
        {
            "role": "assistant",
            "content": f"<think>\n{probe_marker}\n</think>\n\\boxed{{2}}",
        },
    ]
    probe_render = tokenizer.apply_chat_template(
        probe_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    tokenizer_was_patched = probe_marker not in probe_render
    tokenizer_path = model_path

    if tokenizer_was_patched:
        template = tokenizer.chat_template
        if not isinstance(template, str) or not template:
            raise ValueError("Tokenizer strips thinking but has no editable chat_template")
        stripping_pattern = re.compile(
            r"\{%-?\s*if\s+'</think>'\s+in\s+content\s*-?%\}"
            r"\{%-?\s*set\s+content\s*=\s*content\.split\('</think>'\)\[-1\]\s*-?%\}"
            r"\{%-?\s*endif\s*-?%\}"
        )
        patched_template, replacements = stripping_pattern.subn("", template)
        if replacements != 1:
            raise ValueError(
                "Tokenizer removes thinking, but the known DeepSeek stripping block was not found exactly once"
            )
        tokenizer.chat_template = patched_template
        patched_render = tokenizer.apply_chat_template(
            probe_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        if probe_marker not in patched_render:
            raise ValueError("Patched chat template still removes the thinking content")

        patch_digest = hashlib.sha256(
            f"{source_tokenizer_sha256}:{max_model_len}:{patched_template}".encode("utf-8")
        ).hexdigest()[:16]
        tokenizer_dir = cache_dir / "tokenizers" / f"{model_slug}-{patch_digest}"
        if not tokenizer_dir.exists():
            tokenizer_dir.parent.mkdir(parents=True, exist_ok=True)
            temp_dir = tokenizer_dir.with_name(f".{tokenizer_dir.name}.tmp.{os.getpid()}")
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            tokenizer.save_pretrained(temp_dir)
            os.replace(temp_dir, tokenizer_dir)
        tokenizer_path = str(tokenizer_dir.resolve())

    audit_path = cache_dir / f"tokenizer_audit.{model_slug}.json"
    dataset_manifest_sha256 = hashlib.sha256(
        json.dumps(dataset_manifest, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    audit_signature = {
        "rules_version": RULES_VERSION,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "source_tokenizer_sha256": source_tokenizer_sha256,
        "tokenizer_path": tokenizer_path,
        "max_model_len": max_model_len,
    }

    audit = None
    if audit_path.exists():
        candidate = json.loads(audit_path.read_text(encoding="utf-8"))
        if candidate.get("audit_signature") == audit_signature and Path(tokenizer_path).exists():
            audit = candidate
            print("Reusing cached full-tokenization audit.", flush=True)

    if audit is None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False)

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

        def render_ids(messages, add_generation_prompt=False):
            return normalize_token_ids(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=add_generation_prompt,
                )
            )

        empty_user = [{"role": "user", "content": ""}]
        once = render_ids(empty_user)
        twice = render_ids(empty_user * 2)
        added_length = len(twice) - len(once)
        if added_length <= 0:
            raise ValueError("Could not infer the tokenizer system prompt")
        system_prompt = once[:-added_length]
        with_generation = render_ids(empty_user, add_generation_prompt=True)
        generation_prompt = with_generation[len(once) :]
        if not generation_prompt:
            raise ValueError("Tokenizer produced an empty assistant generation prompt")

        audited_rows = 0
        max_tokens = 0
        min_trainable_tokens = None
        overlong = []
        mismatch = []
        thinking_loss = []

        for split, parquet_path in (("train", train_output), ("validation", val_output)):
            for row_index, row in enumerate(pq.read_table(parquet_path).to_pylist()):
                messages = [dict(message) for message in row["messages"]]
                concatenated = []
                loss_mask = []
                for message_index, message in enumerate(messages):
                    message_ids = render_ids([message])
                    if message_index != 0 and message["role"] != "system":
                        if message_ids[: len(system_prompt)] != system_prompt:
                            raise ValueError(
                                f"{split} row {row_index} does not start isolated turns with the inferred system prefix"
                            )
                        message_ids = message_ids[len(system_prompt) :]
                    message_loss = [1 if message["role"] == "assistant" else 0] * len(message_ids)
                    if message["role"] == "assistant":
                        message_loss[: len(generation_prompt)] = [0] * min(
                            len(generation_prompt), len(message_loss)
                        )
                    concatenated.extend(message_ids)
                    loss_mask.extend(message_loss)

                full_ids = render_ids(messages)
                if concatenated != full_ids:
                    mismatch.append((split, row_index, len(concatenated), len(full_ids)))
                    if len(mismatch) >= 10:
                        break

                token_count = len(concatenated)
                trainable_tokens = sum(loss_mask)
                max_tokens = max(max_tokens, token_count)
                min_trainable_tokens = (
                    trainable_tokens
                    if min_trainable_tokens is None
                    else min(min_trainable_tokens, trainable_tokens)
                )
                if token_count > max_model_len:
                    overlong.append((split, row_index, token_count))
                    if len(overlong) >= 20:
                        break
                if trainable_tokens <= 0:
                    thinking_loss.append((split, row_index, "no trainable assistant tokens"))
                rendered = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                assistant_text = messages[-1]["content"]
                if "<think>" not in rendered or "</think>" not in rendered:
                    thinking_loss.append((split, row_index, "thinking tags absent from rendered conversation"))
                if assistant_text not in rendered:
                    thinking_loss.append((split, row_index, "assistant text changed by chat template"))
                if not has_canonical_terminal(
                    assistant_text,
                    row["prompt_format"],
                    row["ground_truth"],
                ):
                    thinking_loss.append((split, row_index, "assistant answer is not canonical"))
                audited_rows += 1
            if mismatch or overlong:
                break

        if mismatch:
            raise ValueError(f"Separate-turn/full-template token mismatch: {mismatch}")
        if overlong:
            raise ValueError(f"SFT examples exceed max_model_len={max_model_len}: {overlong}")
        if thinking_loss:
            raise ValueError(f"Thinking/loss-mask audit failed: {thinking_loss[:20]}")

        audit = {
            "audit_signature": audit_signature,
            "model": {
                "model_type": config.model_type,
                "architectures": config.architectures,
                "tie_word_embeddings": bool(getattr(config, "tie_word_embeddings", False)),
                "tokenizer_size": len(tokenizer),
                "vocab_size": config.vocab_size,
                "max_position_embeddings": config.max_position_embeddings,
            },
            "tokenizer_was_patched": tokenizer_was_patched,
            "source_tokenizer_model_max_length": source_tokenizer_model_max_length,
            "audited_rows": audited_rows,
            "max_tokens": max_tokens,
            "min_trainable_tokens": min_trainable_tokens,
        }
        atomic_json(audit_path, audit)

    atomic_text(tokenizer_path_file, f"{tokenizer_path}\n")
    print("SFT dataset manifest:", dataset_manifest, flush=True)
    print("SFT tokenizer audit:", audit, flush=True)
finally:
    if lock_fd is not None:
        os.close(lock_fd)
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass
PY

[[ -s "$TOKENIZER_PATH_FILE" ]] || die "Tokenizer preparation did not create $TOKENIZER_PATH_FILE"
TOKENIZER_PATH=$(<"$TOKENIZER_PATH_FILE")
[[ -d "$TOKENIZER_PATH" ]] || die "Prepared tokenizer path does not exist: $TOKENIZER_PATH"

SFT_TRAIN_FILE=$SFT_CACHE_DIR/train.parquet
SFT_VAL_FILE=$SFT_CACHE_DIR/validation.parquet
[[ -f "$SFT_TRAIN_FILE" ]] || die "Prepared SFT training parquet is missing: $SFT_TRAIN_FILE"
[[ -f "$SFT_VAL_FILE" ]] || die "Prepared SFT validation parquet is missing: $SFT_VAL_FILE"

read -r SFT_TRAIN_ROWS SFT_VAL_ROWS < <(
    python3 - "$SFT_TRAIN_FILE" "$SFT_VAL_FILE" <<'PY'
import sys

import pyarrow.parquet as pq

print(*(pq.ParquetFile(path).metadata.num_rows for path in sys.argv[1:]))
PY
)
SFT_STEPS_PER_EPOCH=$((SFT_TRAIN_ROWS / TRAIN_BATCH_SIZE))
SFT_VAL_BATCHES=$((SFT_VAL_ROWS / TRAIN_BATCH_SIZE))

################################ model/MBridge/NPU compatibility preflight

REQUIRE_NPU=1
if is_true "$PREPARE_ONLY" || is_true "$DRY_RUN"; then
    REQUIRE_NPU=0
fi

python3 - \
    "$MODEL_PATH" \
    "$TOKENIZER_PATH" \
    "$MAX_MODEL_LEN" \
    "$ACTOR_TP" \
    "$ACTOR_PP" \
    "$NPUS_PER_NODE" \
    "$REQUIRE_NPU" \
    "$SKIP_RUNTIME_PREFLIGHT" <<'PY'
import sys

import torch
from transformers import AutoConfig, AutoTokenizer

(
    model_path,
    tokenizer_path,
    max_model_len,
    actor_tp,
    actor_pp,
    expected_local_npus,
    require_npu,
    skip_runtime_preflight,
) = sys.argv[1:]

max_model_len = int(max_model_len)
actor_tp = int(actor_tp)
actor_pp = int(actor_pp)
expected_local_npus = int(expected_local_npus)
require_npu = require_npu.lower() in {"1", "true", "yes"}
skip_runtime_preflight = skip_runtime_preflight.lower() in {"1", "true", "yes"}

config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False)

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
if int(config.num_attention_heads) % actor_tp:
    raise ValueError("num_attention_heads must be divisible by ACTOR_TP")
if num_kv_heads % actor_tp:
    raise ValueError("num_key_value_heads must be divisible by ACTOR_TP")
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

bridge_name = "skipped"
engine_name = "skipped"
visible_npus = "not checked"
if not skip_runtime_preflight:
    import verl.workers.engine  # noqa: F401
    from verl.models.mcore.mbridge import AutoBridge
    from verl.utils.device import get_device_name
    from verl.workers.engine.base import EngineRegistry

    bridge = AutoBridge.from_config(config, dtype=torch.bfloat16)
    bridge_name = f"{bridge.__class__.__module__}.{bridge.__class__.__name__}"
    if "qwen2" not in bridge_name.lower():
        raise RuntimeError(f"Legacy MBridge did not select a Qwen2 bridge: {bridge_name}")

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
    "SFT model preflight:",
    {
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
        "bridge": bridge_name,
        "engine": engine_name,
        "visible_npus": visible_npus,
    },
)
PY

if is_true "$PREPARE_ONLY"; then
    echo "PREPARE_ONLY complete; prepared train=$SFT_TRAIN_FILE validation=$SFT_VAL_FILE tokenizer=$TOKENIZER_PATH"
    exit 0
fi

################################ torchrun and Hydra arguments

DATA_ARGS=(
    "data.train_files=${SFT_TRAIN_FILE}"
    "data.val_files=${SFT_VAL_FILE}"
    "data.train_batch_size=${TRAIN_BATCH_SIZE}"
    "data.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU}"
    "data.max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU}"
    "data.use_dynamic_bsz=True"
    "data.train_max_samples=-1"
    "data.val_max_samples=-1"
    "data.messages_key=messages"
    "data.pad_mode=no_padding"
    "data.max_length=${MAX_MODEL_LEN}"
    "data.truncation=error"
    "data.ignore_input_ids_mismatch=False"
    "+data.shuffle=True"
    "+data.seed=${SEED}"
)

MODEL_ARGS=(
    "model.path=${MODEL_PATH}"
    "model.hf_config_path=${MODEL_PATH}"
    "model.tokenizer_path=${TOKENIZER_PATH}"
    "model.trust_remote_code=False"
    "model.use_remove_padding=True"
)

ENGINE_ARGS=(
    "engine.use_mbridge=True"
    "engine.vanilla_mbridge=True"
    "engine.use_remove_padding=True"
    "engine.tensor_model_parallel_size=${ACTOR_TP}"
    "engine.pipeline_model_parallel_size=${ACTOR_PP}"
    "engine.context_parallel_size=${ACTOR_CP}"
    "engine.sequence_parallel=True"
    "engine.expert_model_parallel_size=1"
    "engine.use_distributed_optimizer=True"
    "engine.use_megatron_fsdp=False"
    "engine.use_dist_checkpointing=False"
    "engine.param_offload=False"
    "engine.optimizer_offload=False"
    "engine.dtype=bfloat16"
    "+engine.override_transformer_config.context_parallel_algo=megatron_cp_algo"
    "+engine.override_transformer_config.use_flash_attn=True"
    "engine.override_transformer_config.recompute_method=uniform"
    "engine.override_transformer_config.recompute_granularity=full"
    "engine.override_transformer_config.recompute_num_layers=1"
)

OPTIM_ARGS=(
    "optim.lr=${SFT_LR}"
    "optim.lr_warmup_steps_ratio=0.05"
    "optim.weight_decay=0.01"
    'optim.betas=[0.9,0.95]'
    "optim.clip_grad=1.0"
    "optim.lr_decay_style=cosine"
    "optim.min_lr=${MIN_LR}"
)

TRAINER_ARGS=(
    'checkpoint.save_contents=["model","optimizer","extra"]'
    "trainer.device=npu"
    "trainer.nnodes=${NNODES}"
    "trainer.n_gpus_per_node=${NPUS_PER_NODE}"
    "trainer.project_name=${PROJECT_NAME}"
    "trainer.experiment_name=${EXPERIMENT_NAME}"
    'trainer.logger=["console","wandb"]'
    "trainer.seed=${SEED}"
    "trainer.total_epochs=${TOTAL_EPOCHS}"
    "trainer.total_training_steps=${TOTAL_TRAINING_STEPS}"
    "trainer.save_freq=${SAVE_FREQ}"
    "trainer.test_freq=${TEST_FREQ}"
    "trainer.max_ckpt_to_keep=2"
    "trainer.default_local_dir=${CKPT_DIR}"
    "trainer.resume_mode=auto"
)

TORCHRUN_ARGS=(
    --nnodes="$NNODES"
    --nproc-per-node="$NPUS_PER_NODE"
    --node-rank="$NODE_RANK"
    --master-addr="$MASTER_ADDR"
    --master-port="$MASTER_PORT"
)

COMMAND=(
    torchrun
    "${TORCHRUN_ARGS[@]}"
    -m verl.trainer.sft_trainer
    engine=megatron
    optim=megatron
    "${DATA_ARGS[@]}"
    "${MODEL_ARGS[@]}"
    "${ENGINE_ARGS[@]}"
    "${OPTIM_ARGS[@]}"
    "${TRAINER_ARGS[@]}"
    "$@"
)

LOG_FILE=${LOG_FILE:-$LOG_DIR/${EXPERIMENT_NAME}.node${NODE_RANK}.log}

echo "Experiment: $WANDB_ENTITY/$PROJECT_NAME/$EXPERIMENT_NAME"
echo "Model: $MODEL_PATH"
echo "Tokenizer: $TOKENIZER_PATH"
echo "Prepared data: train=$SFT_TRAIN_FILE validation=$SFT_VAL_FILE"
echo "Prepared rows: train=$SFT_TRAIN_ROWS validation=$SFT_VAL_ROWS"
echo "Topology: TP=$ACTOR_TP PP=$ACTOR_PP CP=$ACTOR_CP DP=$ACTOR_DP"
echo "Sequence lengths: prompt_budget=$MAX_PROMPT_LENGTH response_budget=$MAX_RESPONSE_LENGTH total=$MAX_MODEL_LEN"
echo "Schedule: epochs=$TOTAL_EPOCHS explicit_steps=$TOTAL_TRAINING_STEPS save_every=$SAVE_FREQ validate_every=$TEST_FREQ"
echo "Native drop-last schedule: train_steps_per_epoch=$SFT_STEPS_PER_EPOCH validation_batches=$SFT_VAL_BATCHES"
echo "Checkpoint directory: $CKPT_DIR"
echo "Log file: $LOG_FILE"
print_command "${COMMAND[@]}"

if is_true "$DRY_RUN"; then
    echo "DRY_RUN complete; no SFT process was launched."
    exit 0
fi

"${COMMAND[@]}" 2>&1 | tee "$LOG_FILE"
