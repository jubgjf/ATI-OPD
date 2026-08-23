#!/usr/bin/env bash

# Shared lifecycle helpers for the lightweight code-interpreter sandbox.
# Callers own their EXIT trap so this file can be reused by launchers that
# already have broader cleanup responsibilities.

_check_verl_python_sandbox_health() {
    local sandbox_url=$1
    local sandbox_python=${PYTHON_SANDBOX_PYTHON:-python}
    local health_timeout=${2:-5}

    "$sandbox_python" -c \
        'import sys, urllib.request; url=sys.argv[1].rstrip("/")+"/health"; assert urllib.request.urlopen(url, timeout=float(sys.argv[2])).read().decode().strip()=="OK"' \
        "$sandbox_url" "$health_timeout"
}

_stop_verl_python_sandbox() {
    local sandbox_pid=${verl_python_sandbox_pid:-}
    if [[ -z "$sandbox_pid" ]]; then
        return 0
    fi

    if kill -0 "$sandbox_pid" 2>/dev/null; then
        echo "Stopping local Python sandbox (pid=$sandbox_pid)..."
        kill "$sandbox_pid" 2>/dev/null || true
        wait "$sandbox_pid" 2>/dev/null || true
    fi
    verl_python_sandbox_pid=
}

_start_verl_python_sandbox() {
    local sandbox_url=$1
    local sandbox_bind_host=$2
    local sandbox_port=$3
    local sandbox_max_concurrency=$4
    local sandbox_timeout=$5
    local sandbox_startup_timeout=$6
    local sandbox_log_path=$7
    local sandbox_python=${PYTHON_SANDBOX_PYTHON:-python}

    mkdir -p "$(dirname -- "$sandbox_log_path")"
    "$sandbox_python" -m verl.tools.python_sandbox \
        --host "$sandbox_bind_host" \
        --port "$sandbox_port" \
        --max-concurrency "$sandbox_max_concurrency" \
        --max-timeout "$sandbox_timeout" \
        >"$sandbox_log_path" 2>&1 &
    verl_python_sandbox_pid=$!

    local attempt
    for ((attempt = 1; attempt <= sandbox_startup_timeout; attempt++)); do
        if ! kill -0 "$verl_python_sandbox_pid" 2>/dev/null; then
            echo "Local Python sandbox exited during startup. Log: $sandbox_log_path" >&2
            tail -n 80 "$sandbox_log_path" >&2 || true
            wait "$verl_python_sandbox_pid" 2>/dev/null || true
            verl_python_sandbox_pid=
            return 1
        fi
        if _check_verl_python_sandbox_health "$sandbox_url" 2 >/dev/null 2>&1; then
            echo "Local Python sandbox ready: url=$sandbox_url, pid=$verl_python_sandbox_pid, log=$sandbox_log_path"
            return 0
        fi
        sleep 1
    done

    echo "Local Python sandbox health check timed out after ${sandbox_startup_timeout}s: $sandbox_url" >&2
    tail -n 80 "$sandbox_log_path" >&2 || true
    _stop_verl_python_sandbox
    return 1
}
