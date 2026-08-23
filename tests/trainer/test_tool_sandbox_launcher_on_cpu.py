# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shlex
import socket
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMON_SH = REPO_ROOT / "run" / "common.sh"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _launcher_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "AUTO_START_PYTHON_SANDBOX",
        "PYTHON_SANDBOX_URL",
        "PYTHON_SANDBOX_PORT",
        "PYTHON_SANDBOX_TIMEOUT",
        "PYTHON_SANDBOX_MAX_CONCURRENCY",
        "PYTHON_SANDBOX_STARTUP_TIMEOUT",
        "PYTHON_SANDBOX_LOG_PATH",
    ):
        env.pop(key, None)
    env["PYTHON_SANDBOX_PYTHON"] = sys.executable
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def test_common_sh_auto_starts_and_stops_one_local_sandbox_on_cpu(tmp_path):
    port = _free_port()
    output_dir = tmp_path / "run"
    script = f"""
source {shlex.quote(str(COMMON_SH))}
enable_tool_call=True
NODE_RANK=3
default_local_dir={shlex.quote(str(output_dir))}
PYTHON_SANDBOX_PORT={port}
_configure_verl_python_sandbox || exit $?
saved_pid=$verl_python_sandbox_pid
printf '%s\n%s\n%s\n' "$PYTHON_SANDBOX_URL" "$python_sandbox_log_path" "$saved_pid"
_check_verl_python_sandbox_health "$PYTHON_SANDBOX_URL" 2 || exit $?
"""

    result = subprocess.run(
        ["bash", "-c", script],
        cwd=REPO_ROOT,
        env=_launcher_env(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    output_lines = [line for line in result.stdout.splitlines() if not line.startswith(("Local ", "Stopping "))]
    assert output_lines[0] == f"http://127.0.0.1:{port}"
    assert Path(output_lines[1]) == output_dir / "python_sandbox" / "node_3.log"
    assert output_lines[2].isdigit()
    assert "Starting Python sandbox" in Path(output_lines[1]).read_text(encoding="utf-8")
    try:
        os.kill(int(output_lines[2]), 0)
    except ProcessLookupError:
        pass
    else:
        raise AssertionError("sandbox process survived the common.sh EXIT trap")


def test_common_sh_external_sandbox_mode_requires_a_url_on_cpu(tmp_path):
    script = f"""
source {shlex.quote(str(COMMON_SH))}
enable_tool_call=True
NODE_RANK=0
default_local_dir={shlex.quote(str(tmp_path))}
AUTO_START_PYTHON_SANDBOX=False
_configure_verl_python_sandbox
"""

    result = subprocess.run(
        ["bash", "-c", script],
        cwd=REPO_ROOT,
        env=_launcher_env(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "AUTO_START_PYTHON_SANDBOX=False requires PYTHON_SANDBOX_URL" in result.stderr
