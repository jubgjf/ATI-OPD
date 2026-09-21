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

check_port() {
    local name=$1
    local value=$2
    [[ "$value" =~ ^[0-9]+$ ]] || die "$name must be an integer TCP port, got: $value"
    ((value >= 1 && value <= 65535)) || die "$name must be between 1 and 65535, got: $value"
}

load_external_train_config() {
    local train_config_path=${TRAIN_CONFIG_PATH:-/opt/huawei/algorithm/train.config}
    if [[ ! -s "$train_config_path" ]]; then
        echo "No non-empty train config found at $train_config_path; skip external env loading."
        return 0
    fi

    echo "Loading external train config from $train_config_path"
    local allexport_was_set=0
    local nounset_was_set=0
    case "$-" in
        *a*) allexport_was_set=1 ;;
    esac
    case "$-" in
        *u*)
            nounset_was_set=1
            set +u
            ;;
    esac

    set -o allexport
    # shellcheck disable=SC1090
    if ! source "$train_config_path"; then
        if ((allexport_was_set == 0)); then
            set +o allexport
        fi
        if ((nounset_was_set == 1)); then
            set -u
        fi
        die "Failed to load external train config: $train_config_path"
    fi
    if ((allexport_was_set == 0)); then
        set +o allexport
    fi
    if ((nounset_was_set == 1)); then
        set -u
    fi
}

load_external_train_config
if [[ "${VERBOSE:-0}" == "1" ]]; then
    set -x
fi

resolve_cluster_addresses() {
    if ((NNODES > 1)); then
        if [[ -z "${MASTER_ADDR:-}" ]]; then
            if [[ -n "${MA_VJ_NAME:-}" && -n "${MA_TASK_NAME:-}" ]]; then
                MASTER_ADDR="${MA_VJ_NAME}-${MA_TASK_NAME}-0.${MA_VJ_NAME}"
            else
                die \
                    "Multi-node launch requires MASTER_ADDR, or both MA_VJ_NAME and MA_TASK_NAME " \
                    "for the common.sh rank-0 DNS convention."
            fi
        fi
    else
        MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
    fi

    SOCKET_IFNAME=${SOCKET_IFNAME:-eth0}
    if [[ -z "${CURRENT_IP:-}" ]]; then
        if [[ -n "${MA_CURRENT_IP:-}" ]]; then
            CURRENT_IP=$MA_CURRENT_IP
        else
            CURRENT_IP=$(ifconfig "$SOCKET_IFNAME" | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')
        fi
    fi
    [[ -n "$CURRENT_IP" ]] || die "Failed to discover CURRENT_IP from SOCKET_IFNAME=$SOCKET_IFNAME."
    export MASTER_ADDR CURRENT_IP SOCKET_IFNAME
    echo "Cluster network: master=$MASTER_ADDR local=$CURRENT_IP interface=$SOCKET_IFNAME"
}

synchronize_experiment_identity() {
    local method=$1
    local requested_timestamp=${EXPERIMENT_TIMESTAMP:-}
    local requested_full_name=${EXPERIMENT_FULL_NAME:-}
    local identity

    if ((NNODES == 1)) || is_true "$PREPARE_ONLY" || is_true "$DRY_RUN"; then
        if ((NNODES > 1)) && [[ -z "$requested_timestamp" && -z "$requested_full_name" ]]; then
            warn \
                "Multi-node PREPARE_ONLY/DRY_RUN does not contact peer nodes; its generated timestamp is local. " \
                "Set EXPERIMENT_TIMESTAMP for reproducible dry-run output."
        fi
        EXPERIMENT_TIMESTAMP=${requested_timestamp:-$(date +%Y%m%d-%H%M%S)}
        EXPERIMENT_FULL_NAME=${requested_full_name:-${EXPERIMENT_NAME}-${EXPERIMENT_TIMESTAMP}}
        export EXPERIMENT_TIMESTAMP EXPERIMENT_FULL_NAME
        return
    fi

    identity=$(python3 - \
        "$method" \
        "$MODEL_SLUG" \
        "$MODEL_PATH" \
        "$EXPERIMENT_SYNC_FINGERPRINT" \
        "$EXPERIMENT_NAME" \
        "$PROJECT_NAME" \
        "$OUTPUT_ROOT" \
        "$requested_timestamp" \
        "$requested_full_name" \
        "${CKPT_DIR:-}" \
        "$LOG_DIR" \
        "$NNODES" \
        "$NODE_RANK" \
        "$MASTER_ADDR" \
        "$CURRENT_IP" \
        "$EXPERIMENT_SYNC_PORT" \
        "$EXPERIMENT_SYNC_TIMEOUT_SECONDS" <<'PY'
import json
import socket
import sys
import time

(
    method,
    model_slug,
    model_path,
    launch_fingerprint,
    experiment_name,
    project_name,
    output_root,
    requested_timestamp,
    requested_full_name,
    checkpoint_override,
    log_dir,
    nnodes,
    node_rank,
    master_addr,
    current_ip,
    sync_port,
    timeout_seconds,
) = sys.argv[1:]

nnodes = int(nnodes)
node_rank = int(node_rank)
sync_port = int(sync_port)
timeout_seconds = int(timeout_seconds)
deadline = time.monotonic() + timeout_seconds
configuration = {
    "method": method,
    "model_slug": model_slug,
    "model_path": model_path,
    "launch_fingerprint": launch_fingerprint,
    "experiment_name": experiment_name,
    "project_name": project_name,
    "output_root": output_root,
    "requested_timestamp": requested_timestamp,
    "requested_full_name": requested_full_name,
    "checkpoint_override": checkpoint_override,
    "log_dir": log_dir,
    "nnodes": nnodes,
}


def receive_line(connection):
    chunks = []
    size = 0
    while True:
        chunk = connection.recv(4096)
        if not chunk:
            raise RuntimeError("peer closed the experiment-identity connection before sending a newline")
        chunks.append(chunk)
        size += len(chunk)
        if size > 65536:
            raise RuntimeError("experiment-identity message exceeds 64 KiB")
        if b"\n" in chunk:
            return b"".join(chunks).split(b"\n", 1)[0]


def send_json(connection, payload):
    connection.sendall(json.dumps(payload, sort_keys=True).encode("utf-8") + b"\n")


if node_rank == 0:
    timestamp = requested_timestamp or time.strftime("%Y%m%d-%H%M%S")
    full_name = requested_full_name or f"{experiment_name}-{timestamp}"
    connections = {}
    all_connections = []
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind((current_ip, sync_port))
        server.listen(max(1, nnodes - 1))
        while len(connections) < nnodes - 1:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                missing = sorted(set(range(1, nnodes)) - set(connections))
                raise TimeoutError(f"timed out waiting for experiment-identity peers; missing ranks={missing}")
            server.settimeout(min(1.0, remaining))
            try:
                connection, _ = server.accept()
            except TimeoutError:
                continue
            all_connections.append(connection)
            connection.settimeout(max(1.0, remaining))
            message = json.loads(receive_line(connection))
            peer_rank = int(message.pop("node_rank"))
            if peer_rank <= 0 or peer_rank >= nnodes:
                raise RuntimeError(f"invalid peer NODE_RANK={peer_rank}")
            if peer_rank in connections:
                raise RuntimeError(f"duplicate experiment-identity peer NODE_RANK={peer_rank}")
            if message != configuration:
                differing = {
                    key: {"rank0": configuration.get(key), f"rank{peer_rank}": message.get(key)}
                    for key in sorted(set(configuration) | set(message))
                    if configuration.get(key) != message.get(key)
                }
                raise RuntimeError(f"distributed launch configuration mismatch: {differing}")
            connections[peer_rank] = connection
        response = {"ok": True, "timestamp": timestamp, "full_name": full_name}
        for connection in connections.values():
            send_json(connection, response)
    except Exception as error:
        response = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        for connection in all_connections:
            try:
                send_json(connection, response)
            except OSError:
                pass
        raise
    finally:
        for connection in all_connections:
            connection.close()
        server.close()
else:
    request = {**configuration, "node_rank": node_rank}
    last_error = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"timed out connecting to experiment-identity server at {master_addr}:{sync_port}; "
                f"last_error={last_error!r}"
            )
        try:
            connection = socket.create_connection((master_addr, sync_port), timeout=min(5.0, remaining))
            break
        except OSError as error:
            last_error = error
            time.sleep(min(1.0, remaining))
    with connection:
        connection.settimeout(max(1.0, deadline - time.monotonic()))
        send_json(connection, request)
        response = json.loads(receive_line(connection))
    if not response.get("ok"):
        raise RuntimeError(response.get("error", "rank 0 rejected experiment-identity request"))
    timestamp = response["timestamp"]
    full_name = response["full_name"]

if not timestamp or not full_name or any(character in timestamp + full_name for character in "\r\n\t"):
    raise RuntimeError(f"invalid synchronized experiment identity: timestamp={timestamp!r}, full_name={full_name!r}")
print(f"{timestamp}\t{full_name}")
PY
    ) || die "Failed to synchronize the multi-node experiment identity."

    IFS=$'\t' read -r EXPERIMENT_TIMESTAMP EXPERIMENT_FULL_NAME <<<"$identity"
    [[ -n "$EXPERIMENT_TIMESTAMP" && -n "$EXPERIMENT_FULL_NAME" ]] || die \
        "Experiment identity synchronization returned an invalid result: $identity"
    export EXPERIMENT_TIMESTAMP EXPERIMENT_FULL_NAME
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

    export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-3600}
    export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1800}
    export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-auto}
    export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-auto}
    unset HCCL_IF_BASE_PORT || true
    export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-$SOCKET_IFNAME}
    export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-$SOCKET_IFNAME}
    export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-$SOCKET_IFNAME}

    export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
    export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-0}
    export RAY_USAGE_STATS_ENABLED=${RAY_USAGE_STATS_ENABLED:-0}
    export RAY_gcs_server_rpc_server_thread_num=${RAY_gcs_server_rpc_server_thread_num:-32}
    export RAY_gcs_server_request_timeout_seconds=${RAY_gcs_server_request_timeout_seconds:-600}
    export RAY_timeout_ms=${RAY_timeout_ms:-600000}
    export RAY_worker_register_timeout_seconds=${RAY_worker_register_timeout_seconds:-600}
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

RAY_BLOCK_PID=
RAY_LIFECYCLE_SERVER_PID=
RAY_LIFECYCLE_DIR=
RAY_LIFECYCLE_READY_FILE=
MANAGED_RAY_STARTED=0

cleanup_managed_ray() {
    if [[ -n "$RAY_LIFECYCLE_SERVER_PID" ]]; then
        kill "$RAY_LIFECYCLE_SERVER_PID" 2>/dev/null || true
        wait "$RAY_LIFECYCLE_SERVER_PID" 2>/dev/null || true
        RAY_LIFECYCLE_SERVER_PID=
    fi
    if ((MANAGED_RAY_STARTED == 0)) && [[ -z "$RAY_LIFECYCLE_DIR" ]]; then
        return
    fi
    if ((MANAGED_RAY_STARTED == 1)); then
        MANAGED_RAY_STARTED=0
        echo "Stopping the managed Ray runtime on node rank $NODE_RANK..."
        ray stop --force >/dev/null 2>&1 || true
    fi
    if [[ -n "$RAY_BLOCK_PID" ]]; then
        kill "$RAY_BLOCK_PID" 2>/dev/null || true
        wait "$RAY_BLOCK_PID" 2>/dev/null || true
        RAY_BLOCK_PID=
    fi
    if [[ -n "$RAY_LIFECYCLE_DIR" ]]; then
        rm -f "$RAY_LIFECYCLE_READY_FILE"
        rmdir "$RAY_LIFECYCLE_DIR" 2>/dev/null || true
        RAY_LIFECYCLE_DIR=
        RAY_LIFECYCLE_READY_FILE=
    fi
}

start_ray_lifecycle_server() {
    RAY_LIFECYCLE_DIR=$(mktemp -d)
    RAY_LIFECYCLE_READY_FILE=$RAY_LIFECYCLE_DIR/workers.ready
    python3 -u - \
        "$CURRENT_IP" \
        "$EXPERIMENT_SYNC_PORT" \
        "$NNODES" \
        "$EXPERIMENT_FULL_NAME" \
        "$RAY_WORKER_CONNECT_TIMEOUT_SECONDS" \
        "$RAY_LIFECYCLE_READY_FILE" <<'PY' &
import json
import socket
import sys
import time
from pathlib import Path

bind_ip, port, nnodes, experiment_name, timeout_seconds, ready_file = sys.argv[1:]
port = int(port)
nnodes = int(nnodes)
deadline = time.monotonic() + int(timeout_seconds)
connections = {}
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((bind_ip, port))
server.listen(max(1, nnodes - 1))
print(f"Ray lifecycle server listening at {bind_ip}:{port}", flush=True)

while len(connections) < nnodes - 1:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        missing = sorted(set(range(1, nnodes)) - set(connections))
        raise TimeoutError(f"timed out waiting for Ray worker lifecycle clients; missing ranks={missing}")
    server.settimeout(min(1.0, remaining))
    try:
        connection, _ = server.accept()
    except TimeoutError:
        continue
    connection.settimeout(max(1.0, remaining))
    chunks = []
    size = 0
    while True:
        chunk = connection.recv(4096)
        if not chunk:
            raise RuntimeError("Ray worker closed its lifecycle connection before registration")
        chunks.append(chunk)
        size += len(chunk)
        if size > 65536:
            raise RuntimeError("Ray worker lifecycle registration exceeds 64 KiB")
        if b"\n" in chunk:
            break
    message = json.loads(b"".join(chunks).split(b"\n", 1)[0])
    worker_rank = int(message.get("node_rank", -1))
    if worker_rank <= 0 or worker_rank >= nnodes:
        raise RuntimeError(f"invalid Ray worker lifecycle NODE_RANK={worker_rank}")
    if worker_rank in connections:
        raise RuntimeError(f"duplicate Ray worker lifecycle NODE_RANK={worker_rank}")
    if message.get("experiment_name") != experiment_name:
        raise RuntimeError(
            "Ray worker lifecycle experiment mismatch: "
            f"rank0={experiment_name!r}, rank{worker_rank}={message.get('experiment_name')!r}"
        )
    connection.settimeout(None)
    connections[worker_rank] = connection
    print(f"Ray worker lifecycle registered: rank={worker_rank}", flush=True)

Path(ready_file).write_text(
    json.dumps({"experiment_name": experiment_name, "worker_ranks": sorted(connections)}) + "\n",
    encoding="utf-8",
)
print(f"All {nnodes - 1} Ray worker lifecycle clients are registered", flush=True)

# The parent terminates this process during cleanup. Closing these sockets is
# the shutdown signal that makes every worker stop its own local Ray runtime.
while True:
    time.sleep(3600)
PY
    RAY_LIFECYCLE_SERVER_PID=$!
    trap cleanup_managed_ray EXIT
}

wait_for_ray_lifecycle_workers() {
    local deadline=$((SECONDS + RAY_WORKER_CONNECT_TIMEOUT_SECONDS))
    while [[ ! -s "$RAY_LIFECYCLE_READY_FILE" ]]; do
        if ! kill -0 "$RAY_LIFECYCLE_SERVER_PID" 2>/dev/null; then
            wait "$RAY_LIFECYCLE_SERVER_PID" || true
            die "Ray lifecycle server exited before every worker registered."
        fi
        ((SECONDS < deadline)) || die "Timed out waiting for Ray worker lifecycle registration."
        sleep 1
    done
    echo "Ray lifecycle workers: $(<"$RAY_LIFECYCLE_READY_FILE")"
}

wait_for_ray_lifecycle_shutdown() {
    python3 - \
        "$MASTER_ADDR" \
        "$EXPERIMENT_SYNC_PORT" \
        "$NODE_RANK" \
        "$EXPERIMENT_FULL_NAME" \
        "$RAY_WORKER_CONNECT_TIMEOUT_SECONDS" <<'PY'
import json
import socket
import sys
import time

host, port, node_rank, experiment_name, timeout_seconds = sys.argv[1:]
port = int(port)
deadline = time.monotonic() + int(timeout_seconds)
last_error = None
while True:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError(
            f"timed out connecting to Ray lifecycle server at {host}:{port}; last_error={last_error!r}"
        )
    try:
        connection = socket.create_connection((host, port), timeout=min(5.0, remaining))
        break
    except OSError as error:
        last_error = error
        time.sleep(min(1.0, remaining))

with connection:
    payload = {"node_rank": int(node_rank), "experiment_name": experiment_name}
    connection.sendall(json.dumps(payload, sort_keys=True).encode("utf-8") + b"\n")
    connection.settimeout(None)
    while connection.recv(4096):
        pass
print("Ray lifecycle shutdown signal received from rank 0")
PY
}

wait_for_tcp_endpoint() {
    local host=$1
    local port=$2
    local timeout_seconds=$3
    python3 - "$host" "$port" "$timeout_seconds" <<'PY'
import socket
import sys
import time

host, port, timeout_seconds = sys.argv[1:]
port = int(port)
deadline = time.monotonic() + int(timeout_seconds)
last_error = None
while time.monotonic() < deadline:
    try:
        with socket.create_connection((host, port), timeout=min(5.0, max(0.1, deadline - time.monotonic()))):
            print(f"Ray head endpoint is reachable: {host}:{port}")
            raise SystemExit(0)
    except OSError as error:
        last_error = error
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
raise TimeoutError(f"timed out waiting for {host}:{port}; last_error={last_error!r}")
PY
}

wait_for_ray_cluster() {
    python3 - \
        "$RAY_GCS_ADDRESS" \
        "$NNODES" \
        "$NPUS_PER_NODE" \
        "$RAY_CLUSTER_READY_TIMEOUT_SECONDS" <<'PY'
import sys
import time

import ray

address, expected_nodes, expected_npus_per_node, timeout_seconds = sys.argv[1:]
expected_nodes = int(expected_nodes)
expected_npus_per_node = int(expected_npus_per_node)
timeout_seconds = int(timeout_seconds)
expected_total_npus = expected_nodes * expected_npus_per_node
deadline = time.monotonic() + timeout_seconds
last_report = None
last_error = None

while time.monotonic() < deadline:
    try:
        if not ray.is_initialized():
            ray.init(address=address, ignore_reinit_error=True, logging_level="ERROR")
        alive_nodes = [node for node in ray.nodes() if node.get("Alive")]
        npu_counts = [float(node.get("Resources", {}).get("NPU", 0.0)) for node in alive_nodes]
        total_npus = sum(npu_counts)
        report = (len(alive_nodes), tuple(sorted(npu_counts)), total_npus)
        if report != last_report:
            print(
                "Ray cluster resources: "
                f"nodes={len(alive_nodes)}/{expected_nodes}, "
                f"per_node_npus={npu_counts}, total_npus={total_npus:g}/{expected_total_npus}"
            )
            last_report = report
        if (
            len(alive_nodes) == expected_nodes
            and all(count == expected_npus_per_node for count in npu_counts)
            and total_npus == expected_total_npus
        ):
            ray.shutdown()
            raise SystemExit(0)
    except Exception as error:
        last_error = error
        if ray.is_initialized():
            ray.shutdown()
    time.sleep(2)

if ray.is_initialized():
    ray.shutdown()
raise TimeoutError(
    f"Ray cluster did not reach {expected_nodes} nodes and {expected_total_npus} NPUs "
    f"within {timeout_seconds}s; last_report={last_report!r}, last_error={last_error!r}"
)
PY
}

start_managed_ray_head() {
    local ray_resources
    printf -v ray_resources '{"NPU": %d}' "$NPUS_PER_NODE"
    echo "Starting managed Ray head at $CURRENT_IP:$MASTER_PORT..."
    ray start \
        --head \
        --node-ip-address="$CURRENT_IP" \
        --port="$MASTER_PORT" \
        --dashboard-host=0.0.0.0 \
        --dashboard-port="$DASHBOARD_PORT" \
        --resources="$ray_resources" \
        --disable-usage-stats \
        --block &
    RAY_BLOCK_PID=$!

    sleep 1
    if ! kill -0 "$RAY_BLOCK_PID" 2>/dev/null; then
        wait "$RAY_BLOCK_PID" || true
        RAY_BLOCK_PID=
        die "Managed Ray head exited before the cluster became ready."
    fi
    MANAGED_RAY_STARTED=1
    trap cleanup_managed_ray EXIT
}

run_managed_ray_worker() {
    local ray_resources
    printf -v ray_resources '{"NPU": %d}' "$NPUS_PER_NODE"
    echo "Waiting for managed Ray head at $RAY_GCS_ADDRESS..."
    wait_for_tcp_endpoint "$MASTER_ADDR" "$MASTER_PORT" "$RAY_WORKER_CONNECT_TIMEOUT_SECONDS"
    echo "Starting managed Ray worker rank $NODE_RANK from $CURRENT_IP..."
    ray start \
        --address="$RAY_GCS_ADDRESS" \
        --node-ip-address="$CURRENT_IP" \
        --resources="$ray_resources" \
        --disable-usage-stats \
        --block &
    RAY_BLOCK_PID=$!

    sleep 1
    if ! kill -0 "$RAY_BLOCK_PID" 2>/dev/null; then
        wait "$RAY_BLOCK_PID" || true
        RAY_BLOCK_PID=
        die "Managed Ray worker rank $NODE_RANK exited before lifecycle registration."
    fi
    MANAGED_RAY_STARTED=1
    trap cleanup_managed_ray EXIT
    wait_for_ray_lifecycle_shutdown
    cleanup_managed_ray
    trap - EXIT
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
EXPERIMENT_NAME=${EXPERIMENT_NAME:-${MODEL_SLUG}-grpo}
LOG_DIR=${LOG_DIR:-$OUTPUT_ROOT/logs}

unset WANDB_ENTITY WANDB_PROJECT WANDB_NAME WANDB_RUN_ID WANDB_RESUME
export WANDB_MODE=disabled
export WANDB_DISABLED=true

[[ -d "$MODEL_PATH" ]] || die "Missing model directory: $MODEL_PATH"
[[ -f "$TRAIN_FILE" ]] || die "Missing training parquet: $TRAIN_FILE"
[[ -f "$VAL_FILE" ]] || die "Missing validation parquet: $VAL_FILE"

# mbridge's memory-efficient HF export reuses the source checkpoint's shard
# layout and therefore requires model.safetensors.index.json.  Both supported
# 1.5B models are distributed as a single model.safetensors file, so default to
# the regular export path for them.  An explicit opt-in fails here instead of
# wasting training time before failing at the first checkpoint.
MBRIDGE_MEMORY_EFFICIENT_SAVE=${MBRIDGE_MEMORY_EFFICIENT_SAVE:-auto}
MBRIDGE_SAFETENSORS_INDEX=$MODEL_PATH/model.safetensors.index.json
case "$MBRIDGE_MEMORY_EFFICIENT_SAVE" in
    auto | Auto | AUTO)
        if [[ -f "$MBRIDGE_SAFETENSORS_INDEX" ]]; then
            MBRIDGE_MEMORY_EFFICIENT_SAVE_ENABLED=True
        else
            MBRIDGE_MEMORY_EFFICIENT_SAVE_ENABLED=False
        fi
        ;;
    1 | true | True | TRUE | yes | Yes | YES)
        [[ -f "$MBRIDGE_SAFETENSORS_INDEX" ]] || die \
            "MBRIDGE_MEMORY_EFFICIENT_SAVE=1 requires $MBRIDGE_SAFETENSORS_INDEX."
        MBRIDGE_MEMORY_EFFICIENT_SAVE_ENABLED=True
        ;;
    0 | false | False | FALSE | no | No | NO)
        MBRIDGE_MEMORY_EFFICIENT_SAVE_ENABLED=False
        ;;
    *)
        die "MBRIDGE_MEMORY_EFFICIENT_SAVE must be auto, 0, or 1; got: $MBRIDGE_MEMORY_EFFICIENT_SAVE"
        ;;
esac

################################ cluster and parallel topology

NNODES=${NNODES:-${VC_WORKER_NUM:-1}}
NPUS_PER_NODE=${NPUS_PER_NODE:-8}
NODE_RANK=${NODE_RANK:-${VC_TASK_INDEX:-0}}
MASTER_PORT=${MASTER_PORT:-6167}
DASHBOARD_PORT=${DASHBOARD_PORT:-8191}
EXPERIMENT_SYNC_TIMEOUT_SECONDS=${EXPERIMENT_SYNC_TIMEOUT_SECONDS:-600}
RAY_CLUSTER_READY_TIMEOUT_SECONDS=${RAY_CLUSTER_READY_TIMEOUT_SECONDS:-600}
RAY_WORKER_CONNECT_TIMEOUT_SECONDS=${RAY_WORKER_CONNECT_TIMEOUT_SECONDS:-600}

check_positive_integer NNODES "$NNODES"
check_positive_integer NPUS_PER_NODE "$NPUS_PER_NODE"
[[ "$NODE_RANK" =~ ^[0-9]+$ ]] || die "NODE_RANK must be a non-negative integer, got: $NODE_RANK"
((NODE_RANK < NNODES)) || die "NODE_RANK=$NODE_RANK must be smaller than NNODES=$NNODES"
check_port MASTER_PORT "$MASTER_PORT"
check_port DASHBOARD_PORT "$DASHBOARD_PORT"
EXPERIMENT_SYNC_PORT=${EXPERIMENT_SYNC_PORT:-$((MASTER_PORT + 1))}
check_port EXPERIMENT_SYNC_PORT "$EXPERIMENT_SYNC_PORT"
((EXPERIMENT_SYNC_PORT != MASTER_PORT)) || die "EXPERIMENT_SYNC_PORT must differ from MASTER_PORT."
((EXPERIMENT_SYNC_PORT != DASHBOARD_PORT)) || die "EXPERIMENT_SYNC_PORT must differ from DASHBOARD_PORT."
((MASTER_PORT != DASHBOARD_PORT)) || die "MASTER_PORT must differ from DASHBOARD_PORT."
check_positive_integer EXPERIMENT_SYNC_TIMEOUT_SECONDS "$EXPERIMENT_SYNC_TIMEOUT_SECONDS"
check_positive_integer RAY_CLUSTER_READY_TIMEOUT_SECONDS "$RAY_CLUSTER_READY_TIMEOUT_SECONDS"
check_positive_integer RAY_WORKER_CONNECT_TIMEOUT_SECONDS "$RAY_WORKER_CONNECT_TIMEOUT_SECONDS"

if [[ -n "${RAY_ADDRESS:-}" ]]; then
    if ! is_true "$PREPARE_ONLY" && ! is_true "$DRY_RUN"; then
        die "External RAY_ADDRESS is not supported; unset it so grpo.sh can manage the Ray cluster."
    fi
    warn "Ignoring external RAY_ADDRESS during PREPARE_ONLY/DRY_RUN."
fi
unset RAY_ADDRESS

printf -v EXPERIMENT_SYNC_FINGERPRINT \
    'data=%s,%s;npus=%s;master_port=%s;socket=%s;tp=%s;pp=%s;cp=%s;rollout_tp=%s;rollout_pp=%s;rollout_dp=%s;batch=%s;prompt=%s;response=%s' \
    "$TRAIN_FILE" "$VAL_FILE" "$NPUS_PER_NODE" "$MASTER_PORT" "${SOCKET_IFNAME:-eth0}" \
    "${ACTOR_TP:-1}" "${ACTOR_PP:-1}" "${ACTOR_CP:-2}" \
    "${ROLLOUT_TP:-1}" "${ROLLOUT_PP:-1}" "${ROLLOUT_DP:-1}" "${TRAIN_BATCH_SIZE:-64}" \
    "${MAX_PROMPT_LENGTH:-1024}" "${MAX_RESPONSE_LENGTH:-23552}"
resolve_cluster_addresses
synchronize_experiment_identity grpo

CKPT_DIR=${CKPT_DIR:-$OUTPUT_ROOT/checkpoints/$PROJECT_NAME/$EXPERIMENT_FULL_NAME}
LOG_FILE=${LOG_FILE:-$LOG_DIR/${EXPERIMENT_FULL_NAME}.log}
RAY_GCS_ADDRESS=${MASTER_ADDR}:${MASTER_PORT}
export MASTER_PORT

mkdir -p "$CKPT_DIR" "$LOG_DIR"

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
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-6272}

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

if is_true "${CLEAN_STALE_RAY:-0}" && ! is_true "$PREPARE_ONLY" && ! is_true "$DRY_RUN"; then
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
)

if is_true "$MBRIDGE_MEMORY_EFFICIENT_SAVE_ENABLED"; then
    ACTOR_ARGS+=(
        "+actor_rollout_ref.actor.checkpoint.mbridge_config.distributed_filesystem=True"
        "+actor_rollout_ref.actor.checkpoint.mbridge_config.memory_efficient=True"
    )
fi

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
    'trainer.logger=["console"]'
    "trainer.project_name=${PROJECT_NAME}"
    "trainer.experiment_name=${EXPERIMENT_FULL_NAME}"
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
if ((NNODES > 1)); then
    COMMAND+=("+ray_kwargs.ray_init.address=${RAY_GCS_ADDRESS}")
fi
COMMAND+=("$@")

echo "Experiment: $PROJECT_NAME/$EXPERIMENT_FULL_NAME (base=$EXPERIMENT_NAME timestamp=$EXPERIMENT_TIMESTAMP)"
echo "Model: $MODEL_PATH"
echo "Data: train=$TRAIN_FILE validation=$VAL_FILE"
echo "Cluster: node_rank=$NODE_RANK nnodes=$NNODES local_ip=$CURRENT_IP master=$MASTER_ADDR:$MASTER_PORT"
echo "Actor topology: TP=$ACTOR_TP PP=$ACTOR_PP CP=$ACTOR_CP DP=$ACTOR_DP"
echo "Rollout topology: replicas=$ROLLOUT_REPLICAS TP=$ROLLOUT_TP PP=$ROLLOUT_PP internal-DP=$ROLLOUT_DP"
echo "Batching: prompts=$TRAIN_BATCH_SIZE rollouts_per_prompt=$ROLLOUT_N responses=$TOTAL_ROLLOUT_SAMPLES ppo_prompt_minibatch=$PPO_MINI_BATCH_SIZE"
echo "Sequence lengths: prompt=$MAX_PROMPT_LENGTH response=$MAX_RESPONSE_LENGTH total=$MAX_MODEL_LEN"
echo "Checkpoint export: mbridge_memory_efficient=$MBRIDGE_MEMORY_EFFICIENT_SAVE_ENABLED source_index=$MBRIDGE_SAFETENSORS_INDEX"
echo "Schedule: epochs=$TOTAL_EPOCHS explicit_steps=$TOTAL_TRAINING_STEPS save_every=$SAVE_FREQ validate_every=$TEST_FREQ"
echo "Audited eligible rows: train=$EXPECTED_ELIGIBLE_TRAIN_ROWS validation=$EXPECTED_ELIGIBLE_VAL_ROWS expected_steps_per_epoch=$EXPECTED_STEPS_PER_EPOCH"
echo "Checkpoint directory: $CKPT_DIR"
echo "Log file: $LOG_FILE"
print_command "${COMMAND[@]}"

if is_true "$DRY_RUN"; then
    if ((NNODES > 1)); then
        ray_resources=$(printf '{"NPU": %d}' "$NPUS_PER_NODE")
        if ((NODE_RANK == 0)); then
            print_command ray start \
                --head \
                --node-ip-address="$CURRENT_IP" \
                --port="$MASTER_PORT" \
                --dashboard-host=0.0.0.0 \
                --dashboard-port="$DASHBOARD_PORT" \
                --resources="$ray_resources" \
                --disable-usage-stats \
                --block
        else
            print_command ray start \
                --address="$RAY_GCS_ADDRESS" \
                --node-ip-address="$CURRENT_IP" \
                --resources="$ray_resources" \
                --disable-usage-stats \
                --block
        fi
    fi
    echo "DRY_RUN complete; no GRPO process was launched."
    exit 0
fi

if ((NNODES == 1)); then
    "${COMMAND[@]}" 2>&1 | tee "$LOG_FILE"
elif ((NODE_RANK == 0)); then
    start_ray_lifecycle_server
    start_managed_ray_head
    wait_for_ray_cluster
    wait_for_ray_lifecycle_workers
    training_exit_code=0
    "${COMMAND[@]}" 2>&1 | tee "$LOG_FILE" || training_exit_code=$?
    cleanup_managed_ray
    trap - EXIT
    exit "$training_exit_code"
else
    run_managed_ray_worker
fi
