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

from __future__ import annotations

import asyncio
import importlib
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from verl.tools.code_interpreter_tool import code_interpreter, extract_executable_code
from verl.tools.python_sandbox import SandboxHTTPServer, execute_request
from verl.tools.python_sandbox_client import PythonSandboxResult


@pytest.fixture
def sod_tool_module():
    registry_module = importlib.import_module("verl.tools.function_tool")
    module_name = "verl.tools.sod_code_interpreter_tool"
    previous = registry_module.FUNCTION_TOOL_REGISTRY.pop("code_interpreter", None)
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    try:
        yield module
    finally:
        registry_module.FUNCTION_TOOL_REGISTRY.pop("code_interpreter", None)
        sys.modules.pop(module_name, None)
        if previous is not None:
            registry_module.FUNCTION_TOOL_REGISTRY["code_interpreter"] = previous


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("1 + 2", "3\n"),
        ("answer = 42", "42\n"),
        ("print(7)", "7\n"),
    ],
)
def test_sandbox_auto_display_on_cpu(code: str, expected: str):
    response = execute_request(code, timeout_seconds=5)
    assert response["status"] == "success"
    assert response["run_result"]["stdout"] == expected


def test_sandbox_errors_timeout_and_truncation_on_cpu():
    syntax = execute_request("for", timeout_seconds=5)
    runtime = execute_request("raise RuntimeError('boom')", timeout_seconds=5)
    timeout = execute_request("import time; time.sleep(2)", timeout_seconds=1)
    truncated = execute_request("print('x' * 1000)", timeout_seconds=5, max_output_chars=32)

    assert syntax["run_result"]["status"] == "SyntaxError"
    assert runtime["run_result"]["status"] == "RuntimeError"
    assert "boom" in runtime["run_result"]["stderr"]
    assert timeout["status"] == "timeout"
    assert "truncated" in truncated["run_result"]["stdout"]
    assert "original length=" in truncated["run_result"]["stdout"]


def test_sandbox_spawn_requests_can_run_concurrently_on_cpu():
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(lambda value: execute_request(f"{value} ** 2", timeout_seconds=5), range(4)))
    assert [response["run_result"]["stdout"].strip() for response in responses] == ["0", "1", "4", "9"]


@pytest.mark.parametrize(
    ("source", "expected", "extracted"),
    [
        ("x = 1", "x = 1", False),
        ("```python\nprint(1)\n```", "print(1)\n", True),
        ("```py\nprint(2)\n```", "print(2)\n", True),
        ("```\nprint(3)\n```", "print(3)\n", True),
    ],
)
def test_markdown_code_extraction_on_cpu(source: str, expected: str, extracted: bool):
    assert extract_executable_code(source) == (expected, extracted)


@pytest.mark.asyncio
async def test_code_interpreter_http_contract_on_cpu(monkeypatch):
    server = SandboxHTTPServer(
        ("127.0.0.1", 0),
        max_concurrency=4,
        max_timeout_seconds=15,
        max_output_chars=1000,
        max_request_chars=10_000,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("PYTHON_SANDBOX_URL", f"http://127.0.0.1:{server.server_port}")
    # Process spawn can be noticeably slower while the full pytest suite is
    # importing distributed-training modules in parallel. This test verifies
    # the HTTP/tool contract, not the timeout boundary.
    monkeypatch.setenv("PYTHON_SANDBOX_TIMEOUT", "10")
    try:
        response, reward, metrics = await code_interpreter("```python\n6 * 7\n```")
    finally:
        await asyncio.to_thread(server.shutdown)
        server.server_close()
        thread.join(timeout=5)

    assert reward == 0.0
    assert response.text == "[code_interpreter status=success]\n42"
    assert metrics["tool_execution_status"] == "success"
    assert metrics["tool_markdown_extracted"] == 1.0


def test_sod_observation_exposes_raw_output_without_status_prefix_on_cpu(sod_tool_module):
    success = PythonSandboxResult(
        payload={"status": "success", "run_result": {"stdout": "42\n", "stderr": "warning\n"}},
        status="success",
        markdown_extracted=False,
    )
    empty = PythonSandboxResult(
        payload={"status": "success", "run_result": {"stdout": "", "stderr": ""}},
        status="success",
        markdown_extracted=False,
    )
    failed = PythonSandboxResult(
        payload={"status": "failed", "run_result": {"stderr": "traceback"}},
        status="runtime_error",
        markdown_extracted=False,
    )

    assert sod_tool_module._sod_observation(success) == "42\nwarning\n"
    assert sod_tool_module._sod_observation(empty) == ""
    assert sod_tool_module._sod_observation(failed) == "no stdout here"


@pytest.mark.asyncio
async def test_sod_tool_preserves_execution_status_as_out_of_band_metrics_on_cpu(monkeypatch, sod_tool_module):
    async def execute(_code: str) -> PythonSandboxResult:
        return PythonSandboxResult(
            payload={"status": "failed", "run_result": {"status": "RuntimeError", "stderr": "boom"}},
            status="runtime_error",
            markdown_extracted=True,
        )

    monkeypatch.setattr(sod_tool_module, "execute_python_sandbox", execute)
    response, reward, metrics = await sod_tool_module.code_interpreter("raise RuntimeError('boom')")

    assert response.text == "no stdout here"
    assert reward == 0.0
    assert metrics["tool_execution_status"] == "runtime_error"
    assert metrics["tool_status_runtime_error"] == 1.0
    assert metrics["tool_markdown_extracted"] == 1.0
