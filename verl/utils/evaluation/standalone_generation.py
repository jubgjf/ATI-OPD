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

"""Helpers for small standalone generation tools such as eval_show_cases."""

from __future__ import annotations

import argparse
import logging
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import requests

from verl.utils.evaluation.math_benchmark import CheckpointInfo, hf_model_ready

logger = logging.getLogger(__name__)
_CHAT_TEMPLATE_KWARGS_SUPPORTED: bool | None = None
_CHAT_TEMPLATE_KWARGS_LOCK = threading.Lock()


def find_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def run_subprocess(cmd: list[str], *, timeout: float | None = None) -> None:
    logger.info("Running command: %s", " ".join(cmd))
    completed = subprocess.run(cmd, check=False, timeout=timeout)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(cmd)}")


def ensure_checkpoint_merged(checkpoint: CheckpointInfo, args: argparse.Namespace) -> None:
    if args.force_merge:
        logger.info("Force merging checkpoint global_step_%s", checkpoint.step)
    elif hf_model_ready(checkpoint.hf_dir):
        logger.info("Merged HuggingFace model already exists: %s", checkpoint.hf_dir)
        return

    if args.skip_merge:
        raise FileNotFoundError(f"HuggingFace weights are missing in {checkpoint.hf_dir}, but --skip-merge was set.")

    checkpoint.hf_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str(checkpoint.actor_dir),
        "--target_dir",
        str(checkpoint.hf_dir),
    ]
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    run_subprocess(cmd, timeout=args.merge_timeout)

    if not hf_model_ready(checkpoint.hf_dir):
        raise RuntimeError(f"Model merger finished but no HuggingFace weights were found in {checkpoint.hf_dir}")


class VLLMServer:
    def __init__(self, model_dir: Path, checkpoint_step: int, results_dir: Path, args: argparse.Namespace):
        self.model_dir = model_dir
        self.checkpoint_step = checkpoint_step
        self.results_dir = results_dir
        self.args = args
        self.host = args.host
        self.port = args.port or find_free_port(args.host)
        self.process: subprocess.Popen | None = None
        self.log_file = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _command(self) -> list[str]:
        vllm_bin = shutil.which(self.args.vllm_bin) or self.args.vllm_bin
        cmd = [
            vllm_bin,
            "serve",
            str(self.model_dir),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--served-model-name",
            self.args.served_model_name,
            "--tensor-parallel-size",
            str(self.args.tensor_parallel_size),
            "--gpu-memory-utilization",
            str(self.args.gpu_memory_utilization),
        ]
        if self.args.data_parallel_size > 1 or any(
            value is not None
            for value in (
                self.args.data_parallel_size_local,
                self.args.data_parallel_address,
                self.args.data_parallel_rpc_port,
                self.args.data_parallel_start_rank,
                self.args.api_server_count,
            )
        ):
            dp_size = self.args.data_parallel_size
            local_dp_size = self.args.data_parallel_size_local or dp_size
            dp_address = self.args.data_parallel_address or self.host
            dp_rpc_port = self.args.data_parallel_rpc_port or find_free_port(self.host)
            api_server_count = self.args.api_server_count or dp_size
            cmd.extend(
                [
                    "--data-parallel-size",
                    str(dp_size),
                    "--data-parallel-size-local",
                    str(local_dp_size),
                    "--data-parallel-address",
                    dp_address,
                    "--data-parallel-rpc-port",
                    str(dp_rpc_port),
                    "--api-server-count",
                    str(api_server_count),
                ]
            )
            if self.args.data_parallel_start_rank is not None:
                cmd.extend(["--data-parallel-start-rank", str(self.args.data_parallel_start_rank)])
        if self.args.max_model_len is not None:
            cmd.extend(["--max-model-len", str(self.args.max_model_len)])
        if self.args.max_num_seqs is not None:
            cmd.extend(["--max-num-seqs", str(self.args.max_num_seqs)])
        if self.args.max_num_batched_tokens is not None:
            cmd.extend(["--max-num-batched-tokens", str(self.args.max_num_batched_tokens)])
        if self.args.trust_remote_code:
            cmd.append("--trust-remote-code")
        if self.args.enable_reasoning_parser:
            cmd.extend(["--enable-reasoning", "--reasoning-parser", self.args.reasoning_parser])
        cmd.extend(self.args.vllm_extra_arg or [])
        return cmd

    def start(self) -> None:
        log_dir = self.results_dir / "vllm_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"global_step_{self.checkpoint_step}.log"
        self.log_file = log_path.open("a", encoding="utf-8")
        cmd = self._command()
        logger.info("Starting vLLM server for global_step_%s on %s", self.checkpoint_step, self.base_url)
        logger.info("vLLM log file: %s", log_path)
        self.log_file.write(f"\n===== start {time.ctime()} =====\n")
        self.log_file.write(" ".join(cmd) + "\n")
        self.log_file.flush()
        self.process = subprocess.Popen(cmd, stdout=self.log_file, stderr=subprocess.STDOUT)
        self.wait_ready()

    def wait_ready(self) -> None:
        assert self.process is not None
        deadline = time.time() + self.args.server_start_timeout
        last_error = None
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"vLLM server exited early with code {self.process.returncode}")
            try:
                response = requests.get(f"{self.base_url}/v1/models", timeout=5)
                if response.status_code == 200:
                    logger.info("vLLM server is ready: %s", self.base_url)
                    return
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            except requests.RequestException as exc:
                last_error = str(exc)
            time.sleep(5)
        raise TimeoutError(f"Timed out waiting for vLLM server at {self.base_url}. Last error: {last_error}")

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            logger.info("Stopping vLLM server for global_step_%s", self.checkpoint_step)
            self.process.terminate()
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=30)
        if self.log_file is not None:
            self.log_file.write(f"===== stop {time.ctime()} =====\n")
            self.log_file.close()

    def __enter__(self) -> VLLMServer:
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()


def request_chat_completions(
    *,
    base_url: str,
    model_name: str,
    messages: list[dict[str, str]],
    n: int,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    global _CHAT_TEMPLATE_KWARGS_SUPPORTED

    payload: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "n": n,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }
    with _CHAT_TEMPLATE_KWARGS_LOCK:
        include_chat_template_kwargs = _CHAT_TEMPLATE_KWARGS_SUPPORTED is not False
    if include_chat_template_kwargs:
        payload["chat_template_kwargs"] = {"enable_thinking": args.enable_thinking}
    if args.top_k is not None and args.top_k >= 0:
        payload["top_k"] = args.top_k

    response = requests.post(
        f"{base_url}/v1/chat/completions",
        headers={"Authorization": "Bearer EMPTY"},
        json=payload,
        timeout=args.request_timeout,
    )
    if response.status_code in {400, 422} and "chat_template_kwargs" in response.text:
        logger.warning("vLLM rejected chat_template_kwargs; retrying chat request without enable_thinking override.")
        with _CHAT_TEMPLATE_KWARGS_LOCK:
            _CHAT_TEMPLATE_KWARGS_SUPPORTED = False
        payload.pop("chat_template_kwargs", None)
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            headers={"Authorization": "Bearer EMPTY"},
            json=payload,
            timeout=args.request_timeout,
        )
    if response.status_code != 200:
        raise RuntimeError(f"vLLM chat request failed with HTTP {response.status_code}: {response.text[:500]}")
    choices = response.json().get("choices", [])
    if len(choices) != n:
        raise RuntimeError(f"Expected {n} chat completions, got {len(choices)}")
    return choices


def extract_choice_generation(choice: dict[str, Any]) -> dict[str, str | None]:
    message = choice.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        reasoning_content = message.get("reasoning_content")
        content_text = None if content is None else str(content)
        reasoning_text = None if reasoning_content is None else str(reasoning_content)
        if reasoning_text and reasoning_text.strip():
            text = f"{reasoning_text}\n{content_text or ''}".strip()
        else:
            text = content_text or ""
        return {"text": text, "reasoning_content": reasoning_text, "content": content_text}

    return {"text": str(choice.get("text", "")), "reasoning_content": None, "content": None}


def count_tokens(tokenizer: Any, text: str) -> int | None:
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return None
