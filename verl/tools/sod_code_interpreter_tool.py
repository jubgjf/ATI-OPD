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
"""SOD-compatible observation wrapper for the training-time Python sandbox."""

from __future__ import annotations

from typing import Any

from verl.tools.function_tool import function_tool
from verl.tools.python_sandbox_client import (
    CODE_INTERPRETER_SCHEMA,
    PythonSandboxResult,
    build_tool_metrics,
    execute_python_sandbox,
)
from verl.tools.schemas import ToolResponse


def _sod_observation(result: PythonSandboxResult) -> str:
    """Match the text visible to the model in SOD while retaining local metrics."""
    if result.status != "success":
        return "no stdout here"
    run_result = result.payload.get("run_result") or {}
    stdout = str(run_result.get("stdout") or "")
    stderr = str(run_result.get("stderr") or "")
    return stdout + stderr


@function_tool("code_interpreter", schema=CODE_INTERPRETER_SCHEMA)
async def code_interpreter(code: str) -> tuple[ToolResponse, float, dict[str, Any]]:
    """Execute code and expose SOD-compatible raw output to the model."""
    result = await execute_python_sandbox(code)
    return ToolResponse(text=_sod_observation(result)), 0.0, build_tool_metrics(result)


__all__ = ["CODE_INTERPRETER_SCHEMA", "code_interpreter"]
