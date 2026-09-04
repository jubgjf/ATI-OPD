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

from typing import Any

from verl.tools.function_tool import function_tool
from verl.tools.python_sandbox_client import (
    CODE_INTERPRETER_SCHEMA,
    PythonSandboxResult,
    build_tool_metrics,
    execute_python_sandbox,
    extract_executable_code,
)
from verl.tools.schemas import ToolResponse


def _observation(result: PythonSandboxResult) -> str:
    run_result = result.payload.get("run_result") or {}
    stdout = str(run_result.get("stdout") or "")
    stderr = str(run_result.get("stderr") or "")
    parts = [f"[code_interpreter status={result.status}]"]
    if stdout:
        parts.append(stdout.rstrip())
    if stderr:
        parts.append(f"stderr:\n{stderr.rstrip()}")
    if len(parts) == 1:
        parts.append(str(result.payload.get("error") or "No output."))
    return "\n".join(parts)


@function_tool("code_interpreter", schema=CODE_INTERPRETER_SCHEMA)
async def code_interpreter(code: str) -> tuple[ToolResponse, float, dict[str, Any]]:
    """Execute a self-contained Python snippet in the configured sandbox."""
    result = await execute_python_sandbox(code)
    return ToolResponse(text=_observation(result)), 0.0, build_tool_metrics(result)


__all__ = ["CODE_INTERPRETER_SCHEMA", "code_interpreter", "extract_executable_code"]
