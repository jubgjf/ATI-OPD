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
"""FunctionTool client for the training-time Python sandbox."""

from __future__ import annotations

import os
import re
from typing import Any

import aiohttp

from verl.tools.function_tool import function_tool
from verl.tools.schemas import ToolResponse

CODE_INTERPRETER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "code_interpreter",
        "description": "A tool for executing code.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The code to execute.",
                }
            },
            "required": ["code"],
        },
    },
}

_FENCED_CODE_RE = re.compile(r"```(?P<language>[^\n`]*)\n(?P<code>.*?)```", re.DOTALL)


def extract_executable_code(code: str) -> tuple[str, bool]:
    """Extract the best Python/py/generic Markdown code block, if present."""
    matches = list(_FENCED_CODE_RE.finditer(code))
    if not matches:
        return code, False
    for accepted_languages in ({"python", "py"}, {""}):
        for match in matches:
            language = match.group("language").strip().lower()
            if language in accepted_languages:
                return match.group("code"), True
    # A model occasionally labels otherwise valid Python as ``Python3`` or
    # ``code``.  Executing the first fenced block is more useful than sending
    # Markdown fences to ast.parse; the status remains observable in metrics.
    return matches[0].group("code"), True


def _classify_status(payload: dict[str, Any]) -> str:
    status = str(payload.get("status", "error"))
    run_result = payload.get("run_result") or {}
    run_status = str(run_result.get("status", ""))
    if status == "success":
        return "success"
    if status == "timeout" or run_status == "TimeLimitExceeded":
        return "timeout"
    if run_status == "SyntaxError":
        return "syntax_error"
    if status == "failed":
        return "runtime_error"
    return "http_error"


def _observation(payload: dict[str, Any], status: str) -> str:
    run_result = payload.get("run_result") or {}
    stdout = str(run_result.get("stdout") or "")
    stderr = str(run_result.get("stderr") or "")
    parts = [f"[code_interpreter status={status}]"]
    if stdout:
        parts.append(stdout.rstrip())
    if stderr:
        parts.append(f"stderr:\n{stderr.rstrip()}")
    if len(parts) == 1:
        parts.append(str(payload.get("error") or "No output."))
    return "\n".join(parts)


@function_tool("code_interpreter", schema=CODE_INTERPRETER_SCHEMA)
async def code_interpreter(code: str) -> tuple[ToolResponse, float, dict[str, Any]]:
    """Execute a self-contained Python snippet in the configured sandbox."""
    executable_code, markdown_extracted = extract_executable_code(code)
    base_url = os.environ.get("PYTHON_SANDBOX_URL", "http://127.0.0.1:8080").rstrip("/")
    timeout_seconds = int(os.environ.get("PYTHON_SANDBOX_TIMEOUT", "10"))
    client_timeout = aiohttp.ClientTimeout(total=timeout_seconds + 5)
    payload: dict[str, Any]
    http_error = False
    try:
        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            async with session.post(
                f"{base_url}/run_code",
                json={"code": executable_code, "timeout": timeout_seconds, "auto_display": True},
            ) as response:
                if response.status != 200:
                    http_error = True
                    payload = {"status": "error", "error": f"HTTP {response.status}: {await response.text()}"}
                else:
                    payload = await response.json(content_type=None)
    except Exception as error:
        http_error = True
        payload = {"status": "error", "error": f"Sandbox request failed: {type(error).__name__}: {error}"}

    status = "http_error" if http_error else _classify_status(payload)
    metrics = {
        "tool_execution_status": status,
        "tool_status_success": float(status == "success"),
        "tool_status_syntax_error": float(status == "syntax_error"),
        "tool_status_runtime_error": float(status == "runtime_error"),
        "tool_status_timeout": float(status == "timeout"),
        "tool_status_http_error": float(status == "http_error"),
        "tool_markdown_extracted": float(markdown_extracted),
    }
    return ToolResponse(text=_observation(payload, status)), 0.0, metrics
