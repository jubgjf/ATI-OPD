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
"""Small HTTP Python sandbox used by the math tool-call experiments.

This module intentionally provides *process isolation and timeouts*, not a
security boundary.  It is suitable for trusted mathematical model outputs on a
dedicated training network.  Do not expose it to untrusted users or the public
internet.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import multiprocessing as mp
import queue
import threading
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

DEFAULT_TIMEOUT_SECONDS = 10
DEFAULT_MAX_OUTPUT_CHARS = 10_000
DEFAULT_MAX_REQUEST_CHARS = 200_000
DEFAULT_MAX_CONCURRENCY = 32


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n... (truncated, original length={len(text)})"


def _safe_repr(value: Any, max_chars: int) -> str:
    try:
        text = repr(value)
    except Exception:
        try:
            text = str(value)
        except Exception:
            text = "<Unable to represent result>"
    return _truncate_text(text, max_chars)


def _simple_assignment_name(node: ast.stmt) -> str | None:
    if isinstance(node, ast.Assign):
        for target in reversed(node.targets):
            if isinstance(target, ast.Name):
                return target.id
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return None


def _compile_with_auto_display(code: str) -> tuple[Any, str | None, str | None]:
    tree = ast.parse(code, filename="<sandbox>", mode="exec")
    if not tree.body:
        return compile(tree, "<sandbox>", "exec"), None, None

    last_node = tree.body[-1]
    if isinstance(last_node, ast.Expr):
        capture_name = "__verl_sandbox_last_expression_value__"
        capture_node = ast.Assign(
            targets=[ast.Name(id=capture_name, ctx=ast.Store())],
            value=last_node.value,
        )
        ast.copy_location(capture_node, last_node)
        tree.body[-1] = capture_node
        ast.fix_missing_locations(tree)
        return compile(tree, "<sandbox>", "exec"), capture_name, "last_expression"

    assignment_name = _simple_assignment_name(last_node)
    ast.fix_missing_locations(tree)
    return compile(tree, "<sandbox>", "exec"), assignment_name, "last_assignment" if assignment_name else None


def _execute_code(
    code: str,
    result_queue: mp.Queue,
    started_event: Any,
    auto_display: bool,
    max_output_chars: int,
) -> None:
    # The parent starts the user-code timeout from this point. Importing the
    # spawned Python worker itself can be slow on a busy training/debug host,
    # but that startup latency must not consume the code's execution budget.
    started_event.set()
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    exec_globals: dict[str, Any] = {"__name__": "__main__"}
    result: dict[str, Any] = {
        "status": "success",
        "run_status": "Finished",
        "stdout": "",
        "stderr": "",
        "return_code": 0,
        "auto_displayed": False,
        "auto_display_source": None,
    }

    try:
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            if auto_display:
                compiled, result_name, result_source = _compile_with_auto_display(code)
            else:
                compiled = compile(code, "<sandbox>", "exec")
                result_name = None
                result_source = None
            exec(compiled, exec_globals, exec_globals)
            if result_name is not None and result_name in exec_globals:
                value = exec_globals[result_name]
                # A final print(...) already made its value observable.
                if value is not None:
                    print(_safe_repr(value, max_output_chars))
                    result["auto_displayed"] = True
                    result["auto_display_source"] = result_source
    except SyntaxError:
        result.update(status="failed", run_status="SyntaxError", return_code=1)
        stderr_capture.write(traceback.format_exc())
    except BaseException:
        result.update(status="failed", run_status="RuntimeError", return_code=1)
        stderr_capture.write(traceback.format_exc())

    result["stdout"] = _truncate_text(stdout_capture.getvalue(), max_output_chars)
    result["stderr"] = _truncate_text(stderr_capture.getvalue(), max_output_chars)
    try:
        result_queue.put(result)
    except Exception:
        pass


def execute_request(
    code: str,
    *,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    auto_display: bool = True,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
) -> dict[str, Any]:
    """Execute one snippet in a fresh spawned process and return the wire response."""
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(maxsize=1)
    started_event = ctx.Event()
    process = ctx.Process(
        target=_execute_code,
        args=(code, result_queue, started_event, auto_display, max_output_chars),
        daemon=True,
    )
    process.start()
    startup_timeout_seconds = max(30, timeout_seconds * 2)
    if not started_event.wait(timeout=startup_timeout_seconds):
        if process.is_alive():
            process.terminate()
        process.join(timeout=2)
        try:
            result_queue.close()
            result_queue.join_thread()
        except Exception:
            pass
        return {
            "status": "failed",
            "run_result": {
                "status": "Error",
                "stdout": "",
                "stderr": f"Error: Sandbox worker failed to start after {startup_timeout_seconds} seconds.",
                "return_code": 1,
                "auto_displayed": False,
                "auto_display_source": None,
            },
            "error": "Sandbox worker failed to start.",
        }
    try:
        result = result_queue.get(timeout=timeout_seconds)
    except queue.Empty:
        if process.is_alive():
            process.terminate()
        process.join(timeout=2)
        return {
            "status": "timeout",
            "run_result": {
                "status": "TimeLimitExceeded",
                "stdout": "",
                "stderr": f"Error: Execution timed out after {timeout_seconds} seconds.",
                "return_code": 1,
                "auto_displayed": False,
                "auto_display_source": None,
            },
            "error": "Execution timed out.",
        }
    finally:
        try:
            result_queue.close()
            result_queue.join_thread()
        except Exception:
            pass

    process.join(timeout=2)
    if process.is_alive():
        process.terminate()
        process.join(timeout=2)
    return {
        "status": result["status"],
        "run_result": {
            "status": result["run_status"],
            "stdout": result["stdout"],
            "stderr": result["stderr"],
            "return_code": result["return_code"],
            "auto_displayed": result.get("auto_displayed", False),
            "auto_display_source": result.get("auto_display_source"),
        },
        "error": result["stderr"] if result["status"] != "success" else None,
    }


class SandboxHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        max_concurrency: int,
        max_timeout_seconds: int,
        max_output_chars: int,
        max_request_chars: int,
    ) -> None:
        super().__init__(server_address, SandboxRequestHandler)
        self.execution_slots = threading.BoundedSemaphore(max_concurrency)
        self.max_timeout_seconds = max_timeout_seconds
        self.max_output_chars = max_output_chars
        self.max_request_chars = max_request_chars


class SandboxRequestHandler(BaseHTTPRequestHandler):
    server: SandboxHTTPServer

    def log_message(self, format: str, *args: Any) -> None:
        # Keep Ray logs useful during large rollout batches.
        return

    def _write_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") == "/health":
            body = b"OK"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self._write_json(HTTPStatus.NOT_FOUND, {"status": "error", "error": "Not found."})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/run_code":
            self._write_json(HTTPStatus.NOT_FOUND, {"status": "error", "error": "Not found."})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0
        if content_length <= 0 or content_length > self.server.max_request_chars:
            self._write_json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                {"status": "error", "error": "Invalid or oversized request body."},
            )
            return
        try:
            payload = json.loads(self.rfile.read(content_length))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._write_json(HTTPStatus.BAD_REQUEST, {"status": "error", "error": "Invalid JSON body."})
            return

        code = payload.get("code", "")
        if isinstance(code, list):
            code = code[0] if code else ""
        if not isinstance(code, str):
            code = str(code)
        if not code.strip():
            self._write_json(
                HTTPStatus.OK,
                {
                    "status": "failed",
                    "run_result": {
                        "status": "Error",
                        "stdout": "",
                        "stderr": "Error: Empty code.",
                        "return_code": 1,
                        "auto_displayed": False,
                        "auto_display_source": None,
                    },
                    "error": "Empty code.",
                },
            )
            return

        try:
            timeout_seconds = int(payload.get("timeout", DEFAULT_TIMEOUT_SECONDS))
        except (TypeError, ValueError):
            timeout_seconds = DEFAULT_TIMEOUT_SECONDS
        timeout_seconds = min(max(timeout_seconds, 1), self.server.max_timeout_seconds)

        if not self.server.execution_slots.acquire(timeout=1):
            self._write_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"status": "error", "error": "Sandbox concurrency limit reached."},
            )
            return
        try:
            result = execute_request(
                code,
                timeout_seconds=timeout_seconds,
                auto_display=bool(payload.get("auto_display", True)),
                max_output_chars=self.server.max_output_chars,
            )
        except Exception as error:
            result = {"status": "error", "error": str(error), "run_result": None}
        finally:
            self.server.execution_slots.release()
        self._write_json(HTTPStatus.OK, result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max-concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY)
    parser.add_argument("--max-timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--max-output-chars", type=int, default=DEFAULT_MAX_OUTPUT_CHARS)
    parser.add_argument("--max-request-chars", type=int, default=DEFAULT_MAX_REQUEST_CHARS)
    args = parser.parse_args()

    server = SandboxHTTPServer(
        (args.host, args.port),
        max_concurrency=max(1, args.max_concurrency),
        max_timeout_seconds=max(1, args.max_timeout),
        max_output_chars=max(1, args.max_output_chars),
        max_request_chars=max(1, args.max_request_chars),
    )
    print(
        f"Starting Python sandbox on {args.host}:{server.server_port} "
        f"(max_concurrency={args.max_concurrency}, max_timeout={args.max_timeout}s)",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
