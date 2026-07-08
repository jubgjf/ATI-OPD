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

"""Evaluate math benchmarks for verl checkpoints with a vLLM OpenAI server."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hashlib
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from transformers import AutoTokenizer

from verl.utils.evaluation.math_benchmark import (
    BENCHMARK_NAMES,
    CheckpointInfo,
    append_jsonl_record,
    apply_path_remaps,
    build_math_chat_messages,
    completed_sample_ids,
    discover_checkpoints,
    evaluate_sample,
    find_baseline_model_path_in_wandb_config,
    find_default_local_dir_in_wandb_config,
    has_length_finish_reason,
    hf_model_ready,
    load_benchmark_samples,
    load_jsonl_records,
    parse_csv,
    parse_path_remaps,
    render_prompt_with_chat_template,
    sample_record_from_evaluation,
    score_completion,
    summarize_records,
    write_json_atomic,
)

logger = logging.getLogger(__name__)
_CHAT_TEMPLATE_KWARGS_SUPPORTED: bool | None = None
_CHAT_TEMPLATE_KWARGS_LOCK = threading.Lock()


@dataclass(frozen=True)
class RolloutTask:
    benchmark: str
    sample_idx: int
    sample_id: str
    attempt_idx: int
    rollout_idx: int
    messages: list[dict[str, str]]


@dataclass(frozen=True)
class RolloutResult:
    task: RolloutTask
    choice: dict[str, Any] | None = None
    error: str | None = None


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_float(name: str, default: float | None = None) -> float | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", "jubgjf"))
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "DMPD"))
    parser.add_argument("--wandb-run-name", default=os.environ.get("WANDB_RUN_NAME"))
    parser.add_argument("--wandb-run-id", default=os.environ.get("WANDB_RUN_ID"))
    parser.add_argument("--wandb-init-mode", default=os.environ.get("EVAL_WANDB_INIT_MODE"))
    parser.add_argument("--wandb-init-timeout", type=float, default=_env_float("WANDB_INIT_TIMEOUT", 60.0))
    parser.add_argument("--wandb-preflight-timeout", type=float, default=_env_float("WANDB_PREFLIGHT_TIMEOUT", 10.0))
    parser.add_argument("--skip-wandb", action=argparse.BooleanOptionalAction, default=_env_bool("SKIP_WANDB", False))

    parser.add_argument(
        "--model-path",
        default=os.environ.get("MODEL_PATH"),
        help="Evaluate a merged HuggingFace model directory directly instead of scanning verl checkpoints.",
    )
    parser.add_argument("--model-step", type=int, default=_env_int("MODEL_STEP", 0))
    parser.add_argument(
        "--baseline-model-path",
        default=os.environ.get("BASELINE_MODEL_PATH") or os.environ.get("BASE_MODEL_PATH"),
        help=(
            "Original HuggingFace model directory to evaluate as global_step_0 before verl checkpoints. "
            "If omitted, the evaluator tries actor_rollout_ref.model.path from the matched W&B run config."
        ),
    )
    parser.add_argument(
        "--include-baseline-model",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("INCLUDE_BASELINE_MODEL", True),
        help="Include the original model as global_step_0 when --baseline-model-path or W&B config provides one.",
    )
    parser.add_argument(
        "--default-local-dir",
        default=os.environ.get("DEFAULT_LOCAL_DIR_OVERRIDE") or os.environ.get("DEFAULT_LOCAL_DIR"),
        help="Checkpoint root. If omitted, it is read from the matched W&B run config.",
    )
    parser.add_argument(
        "--path-remap",
        action="append",
        default=parse_csv(os.environ.get("CHECKPOINT_PATH_REMAPS")),
        help="Rewrite checkpoint paths from W&B config, in FROM=TO format. Can be repeated.",
    )
    parser.add_argument(
        "--dataset-root",
        default=os.environ.get("DATASET_ROOT", "dataset"),
        help="Directory containing aime24/aime25/amc23 local datasets.",
    )
    parser.add_argument(
        "--benchmarks",
        default=os.environ.get("BENCHMARKS", ",".join(BENCHMARK_NAMES)),
        help="Comma-separated benchmark names.",
    )
    parser.add_argument("--results-dir", default=os.environ.get("EVAL_RESULTS_DIR"))
    parser.add_argument("--checkpoint-steps", default=os.environ.get("CHECKPOINT_STEPS"))
    parser.add_argument("--max-samples", type=int, default=_env_int("MAX_SAMPLES", -1))
    parser.add_argument("--num-samples", type=int, default=_env_int("ROLLOUT_N", 16))
    parser.add_argument("--max-tokens", type=int, default=_env_int("MAX_TOKENS", 32768))
    parser.add_argument("--temperature", type=float, default=_env_float("TEMPERATURE", 0.7))
    parser.add_argument("--top-p", type=float, default=_env_float("TOP_P", 0.95))
    parser.add_argument("--top-k", type=int, default=_env_int("TOP_K"))
    parser.add_argument(
        "--generation-endpoint",
        choices=("chat", "completions"),
        default=os.environ.get("GENERATION_ENDPOINT", "chat"),
        help="Use vLLM's chat endpoint by default. The completions endpoint is kept for raw-prompt debugging.",
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("ENABLE_THINKING", True),
        help="Pass enable_thinking to the tokenizer chat template. Applies to chat_template_kwargs or local rendering.",
    )
    parser.add_argument(
        "--verifier",
        choices=("math_dapo", "math_verify", "hybrid"),
        default=os.environ.get("VERIFIER", "hybrid"),
        help="Answer verifier. hybrid keeps math_dapo diagnostics and falls back to math_verify for equivalence.",
    )
    parser.add_argument("--math-verify-timeout", type=float, default=_env_float("MATH_VERIFY_TIMEOUT", 30.0))
    parser.add_argument("--request-timeout", type=float, default=_env_float("REQUEST_TIMEOUT", None))
    parser.add_argument(
        "--store-outputs",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("STORE_OUTPUTS", True),
    )
    parser.add_argument(
        "--retry-skipped",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("RETRY_SKIPPED", False),
    )
    parser.add_argument(
        "--stop-after-samples",
        type=int,
        default=_env_int("STOP_AFTER_SAMPLES"),
        help="Debug hook for resume testing. Stops after this many newly processed samples.",
    )
    parser.add_argument(
        "--length-retries",
        type=int,
        default=_env_int("LENGTH_RETRIES", 2),
        help="Retry a whole sample this many extra times when any rollout finishes with reason 'length'.",
    )

    parser.add_argument("--vllm-bin", default=os.environ.get("VLLM_BIN", "vllm"))
    parser.add_argument("--served-model-name", default=os.environ.get("SERVED_MODEL_NAME", "eval-model"))
    parser.add_argument("--host", default=os.environ.get("VLLM_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=_env_int("VLLM_PORT", 0) or 0)
    parser.add_argument("--server-start-timeout", type=float, default=_env_float("SERVER_START_TIMEOUT", 900.0))
    parser.add_argument("--tensor-parallel-size", type=int, default=_env_int("TP_SIZE", 1))
    parser.add_argument("--data-parallel-size", type=int, default=_env_int("DP_SIZE", 1))
    parser.add_argument("--data-parallel-size-local", type=int, default=_env_int("LOCAL_DP_SIZE"))
    parser.add_argument("--data-parallel-address", default=os.environ.get("DP_ADDRESS"))
    parser.add_argument("--data-parallel-rpc-port", type=int, default=_env_int("DP_RPC_PORT"))
    parser.add_argument("--data-parallel-start-rank", type=int, default=_env_int("DP_START_RANK"))
    parser.add_argument("--api-server-count", type=int, default=_env_int("API_SERVER_COUNT"))
    parser.add_argument("--max-concurrency-per-dp", type=int, default=_env_int("MAX_CONCURRENCY_PER_DP", 128))
    parser.add_argument("--max-concurrency", type=int, default=_env_int("MAX_CONCURRENCY"))
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=_env_float("GPU_MEMORY_UTILIZATION", 0.9),
    )
    parser.add_argument("--max-model-len", type=int, default=_env_int("MAX_MODEL_LEN"))
    parser.add_argument("--max-num-seqs", type=int, default=_env_int("MAX_NUM_SEQS", 128))
    parser.add_argument("--max-num-batched-tokens", type=int, default=_env_int("MAX_NUM_BATCHED_TOKENS", 32768))
    parser.add_argument(
        "--enable-reasoning-parser",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("ENABLE_REASONING_PARSER", False),
    )
    parser.add_argument("--reasoning-parser", default=os.environ.get("REASONING_PARSER", "deepseek_r1"))
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("TRUST_REMOTE_CODE", False),
    )
    parser.add_argument("--vllm-extra-arg", action="append", default=parse_csv(os.environ.get("VLLM_EXTRA_ARGS")))

    parser.add_argument("--skip-merge", action=argparse.BooleanOptionalAction, default=_env_bool("SKIP_MERGE", False))
    parser.add_argument("--force-merge", action=argparse.BooleanOptionalAction, default=_env_bool("FORCE_MERGE", False))
    parser.add_argument("--merge-timeout", type=float, default=_env_float("MERGE_TIMEOUT", None))

    parser.add_argument("--watch", action=argparse.BooleanOptionalAction, default=_env_bool("WATCH", False))
    parser.add_argument("--poll-interval", type=float, default=_env_float("POLL_INTERVAL", 300.0))
    parser.add_argument("--idle-timeout", type=float, default=_env_float("IDLE_TIMEOUT", 0.0))
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=_env_bool("DRY_RUN", False))
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"))

    return parser.parse_args()


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


def request_completions(
    *,
    base_url: str,
    model_name: str,
    prompt: str,
    n: int,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    payload: dict[str, Any] = {
        "model": model_name,
        "prompt": prompt,
        "n": n,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }
    if args.top_k is not None and args.top_k >= 0:
        payload["top_k"] = args.top_k
    response = requests.post(
        f"{base_url}/v1/completions",
        headers={"Authorization": "Bearer EMPTY"},
        json=payload,
        timeout=args.request_timeout,
    )
    if response.status_code != 200:
        raise RuntimeError(f"vLLM completion request failed with HTTP {response.status_code}: {response.text[:500]}")
    data = response.json()
    choices = data.get("choices", [])
    if len(choices) != n:
        raise RuntimeError(f"Expected {n} completions, got {len(choices)}")
    return choices


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
    data = response.json()
    choices = data.get("choices", [])
    if len(choices) != n:
        raise RuntimeError(f"Expected {n} chat completions, got {len(choices)}")
    return choices


def request_generations(
    *,
    base_url: str,
    model_name: str,
    tokenizer: Any,
    messages: list[dict[str, str]],
    n: int,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if args.generation_endpoint == "chat":
        return request_chat_completions(base_url=base_url, model_name=model_name, messages=messages, n=n, args=args)

    prompt = render_prompt_with_chat_template(tokenizer, messages, enable_thinking=args.enable_thinking)
    return request_completions(base_url=base_url, model_name=model_name, prompt=prompt, n=n, args=args)


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


def effective_max_concurrency(args: argparse.Namespace) -> int:
    if args.max_concurrency is not None and args.max_concurrency > 0:
        return args.max_concurrency
    per_dp = args.max_concurrency_per_dp or args.max_num_seqs or 128
    return max(1, args.data_parallel_size * per_dp)


def request_rollout(
    *,
    task: RolloutTask,
    base_url: str,
    model_name: str,
    tokenizer: Any,
    args: argparse.Namespace,
) -> RolloutResult:
    try:
        choices = request_generations(
            base_url=base_url,
            model_name=model_name,
            tokenizer=tokenizer,
            messages=task.messages,
            n=1,
            args=args,
        )
        return RolloutResult(task=task, choice=choices[0])
    except Exception as exc:
        return RolloutResult(task=task, error=repr(exc))


def records_paths(results_dir: Path, checkpoint: CheckpointInfo, benchmark: str) -> tuple[Path, Path, Path, Path]:
    benchmark_dir = results_dir / f"global_step_{checkpoint.step}" / benchmark
    return (
        benchmark_dir / "records.jsonl",
        benchmark_dir / "skipped.jsonl",
        benchmark_dir / "summary.json",
        benchmark_dir / "wandb_logged.json",
    )


def max_length_attempts(args: argparse.Namespace) -> int:
    return max(1, int(args.length_retries) + 1)


def length_attempt_counts(skipped_records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in skipped_records:
        if record.get("reason") != "length" or "sample_id" not in record:
            continue
        sample_id = str(record["sample_id"])
        counts[sample_id] = counts.get(sample_id, 0) + 1
    return counts


def terminal_skipped_sample_ids(skipped_records: list[dict[str, Any]], args: argparse.Namespace) -> set[str]:
    if args.retry_skipped:
        return set()

    length_counts = length_attempt_counts(skipped_records)
    exhausted_length_ids = {
        sample_id for sample_id, attempt_count in length_counts.items() if attempt_count >= max_length_attempts(args)
    }
    terminal_ids = set(exhausted_length_ids)
    for record in skipped_records:
        if "sample_id" not in record or record.get("status") == "retry":
            continue
        if record.get("reason") != "length":
            terminal_ids.add(str(record["sample_id"]))
    return terminal_ids


def filtered_skipped_records(
    skipped_records: list[dict[str, Any]], ok_records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    ok_ids = completed_sample_ids(ok_records)
    latest_by_sample_id: dict[str, dict[str, Any]] = {}
    for record in skipped_records:
        sample_id = str(record.get("sample_id"))
        if sample_id in ok_ids or record.get("status") == "retry":
            continue
        latest_by_sample_id[sample_id] = record
    return list(latest_by_sample_id.values())


def benchmark_has_remaining(
    dataset_root: Path,
    benchmark: str,
    records_path: Path,
    skipped_path: Path,
    args: argparse.Namespace,
) -> bool:
    samples = load_benchmark_samples(dataset_root, benchmark, max_samples=args.max_samples)
    ok_records = load_jsonl_records(records_path)
    skipped_records = load_jsonl_records(skipped_path)
    done_ids = completed_sample_ids(ok_records)
    done_ids.update(terminal_skipped_sample_ids(skipped_records, args))
    return any(sample.sample_id not in done_ids for sample in samples)


def evaluate_benchmarks(
    *,
    checkpoint: CheckpointInfo,
    benchmarks: list[str],
    dataset_root: Path,
    results_dir: Path,
    tokenizer: Any,
    server: VLLMServer | None,
    args: argparse.Namespace,
    processed_counter: dict[str, int],
) -> tuple[list[dict[str, Any]], bool]:
    stopped_early = False
    benchmark_paths: dict[str, tuple[Path, Path, Path, Path]] = {}
    pending_samples: list[tuple[str, int, Any, int]] = []

    for benchmark in benchmarks:
        records_path, skipped_path, summary_path, logged_path = records_paths(results_dir, checkpoint, benchmark)
        benchmark_paths[benchmark] = (records_path, skipped_path, summary_path, logged_path)
        records = load_jsonl_records(records_path)
        skipped_records = load_jsonl_records(skipped_path)
        done_ids = completed_sample_ids(records)
        done_ids.update(terminal_skipped_sample_ids(skipped_records, args))
        length_counts = length_attempt_counts(skipped_records)

        samples = load_benchmark_samples(dataset_root, benchmark, max_samples=args.max_samples)
        logger.info(
            "Evaluating global_step_%s %s: %s samples, %s already done",
            checkpoint.step,
            benchmark,
            len(samples),
            len(done_ids),
        )
        pending_samples.extend(
            (benchmark, idx, sample, 0 if args.retry_skipped else length_counts.get(sample.sample_id, 0))
            for idx, sample in enumerate(samples)
            if sample.sample_id not in done_ids
        )

    if args.stop_after_samples is not None:
        remaining_budget = args.stop_after_samples - processed_counter["count"]
        if remaining_budget <= 0:
            stopped_early = True
            pending_samples = []
        elif len(pending_samples) > remaining_budget:
            pending_samples = pending_samples[:remaining_budget]
            stopped_early = True
    if pending_samples and server is None:
        stopped_early = True
        pending_samples = []

    rollout_tasks: deque[RolloutTask] = deque()
    sample_states: dict[tuple[str, str], dict[str, Any]] = {}
    for benchmark, idx, sample, attempt_idx in pending_samples:
        messages = build_math_chat_messages(sample.problem, idx=idx)
        sample_key = (benchmark, sample.sample_id)
        sample_states[sample_key] = {
            "benchmark": benchmark,
            "idx": idx,
            "sample": sample,
            "messages": messages,
            "attempt_idx": attempt_idx,
            "choices": {},
            "errors": [],
        }

        for rollout_idx in range(args.num_samples):
            rollout_tasks.append(
                RolloutTask(
                    benchmark=benchmark,
                    sample_idx=idx,
                    sample_id=sample.sample_id,
                    attempt_idx=attempt_idx,
                    rollout_idx=rollout_idx,
                    messages=messages,
                )
            )

    if rollout_tasks:
        max_concurrency = min(effective_max_concurrency(args), len(rollout_tasks))
        logger.info(
            "Submitting %s rollout requests for global_step_%s across %s benchmarks with max_concurrency=%s",
            len(rollout_tasks),
            checkpoint.step,
            len(benchmarks),
            max_concurrency,
        )

        in_flight: dict[concurrent.futures.Future[RolloutResult], RolloutTask] = {}

        def submit_next(executor: concurrent.futures.ThreadPoolExecutor) -> bool:
            if not rollout_tasks:
                return False
            task = rollout_tasks.popleft()
            assert server is not None
            future = executor.submit(
                request_rollout,
                task=task,
                base_url=server.base_url,
                model_name=args.served_model_name,
                tokenizer=tokenizer,
                args=args,
            )
            in_flight[future] = task
            return True

        def enqueue_attempt(state: dict[str, Any]) -> None:
            sample = state["sample"]
            for rollout_idx in range(args.num_samples):
                rollout_tasks.append(
                    RolloutTask(
                        benchmark=state["benchmark"],
                        sample_idx=state["idx"],
                        sample_id=sample.sample_id,
                        attempt_idx=state["attempt_idx"],
                        rollout_idx=rollout_idx,
                        messages=state["messages"],
                    )
                )

        def finalize_sample_if_ready(task: RolloutTask) -> None:
            state = sample_states[(task.benchmark, task.sample_id)]
            if task.attempt_idx != state["attempt_idx"]:
                return
            choices_by_rollout: dict[int, dict[str, Any]] = state["choices"]
            errors: list[str] = state["errors"]
            if len(choices_by_rollout) + len(errors) < args.num_samples:
                return

            benchmark = state["benchmark"]
            sample = state["sample"]
            records_path, skipped_path, _, _ = benchmark_paths[benchmark]
            choices = [choices_by_rollout[idx] for idx in sorted(choices_by_rollout)]
            finish_reasons = [choice.get("finish_reason") for choice in choices]
            if errors:
                append_jsonl_record(
                    skipped_path,
                    {
                        "status": "skipped",
                        "reason": "error",
                        "checkpoint_step": checkpoint.step,
                        "benchmark": benchmark,
                        "sample_id": sample.sample_id,
                        "problem": sample.problem,
                        "ground_truth": sample.ground_truth,
                        "errors": errors,
                        "finish_reasons": finish_reasons,
                        "timestamp": time.time(),
                    },
                )
                logger.warning(
                    "Skipping global_step_%s %s sample %s because %s/%s rollouts errored.",
                    checkpoint.step,
                    benchmark,
                    sample.sample_id,
                    len(errors),
                    args.num_samples,
                )
                processed_counter["count"] += 1
                return
            if has_length_finish_reason(finish_reasons):
                next_attempt_idx = state["attempt_idx"] + 1
                has_retry = next_attempt_idx < max_length_attempts(args)
                append_jsonl_record(
                    skipped_path,
                    {
                        "status": "retry" if has_retry else "skipped",
                        "reason": "length",
                        "checkpoint_step": checkpoint.step,
                        "benchmark": benchmark,
                        "sample_id": sample.sample_id,
                        "problem": sample.problem,
                        "ground_truth": sample.ground_truth,
                        "attempt_idx": state["attempt_idx"],
                        "attempt_number": state["attempt_idx"] + 1,
                        "max_attempts": max_length_attempts(args),
                        "finish_reasons": finish_reasons,
                        "timestamp": time.time(),
                    },
                )
                if has_retry:
                    logger.warning(
                        "Retrying global_step_%s %s sample %s after length finish_reason on attempt %s/%s: %s",
                        checkpoint.step,
                        benchmark,
                        sample.sample_id,
                        state["attempt_idx"] + 1,
                        max_length_attempts(args),
                        finish_reasons,
                    )
                    state["attempt_idx"] = next_attempt_idx
                    state["choices"] = {}
                    state["errors"] = []
                    enqueue_attempt(state)
                else:
                    logger.warning(
                        "Skipping global_step_%s %s sample %s after length retries were exhausted: %s",
                        checkpoint.step,
                        benchmark,
                        sample.sample_id,
                        finish_reasons,
                    )
                    processed_counter["count"] += 1
                return

            scored = []
            for choice in choices:
                generation = extract_choice_generation(choice)
                text = generation["text"] or ""
                scored.append(
                    score_completion(
                        text,
                        sample.ground_truth,
                        finish_reason=choice.get("finish_reason"),
                        response_token_count=count_tokens(tokenizer, text),
                        reasoning_content=generation["reasoning_content"],
                        content=generation["content"],
                        verifier=args.verifier,
                        math_verify_timeout=args.math_verify_timeout,
                    )
                )
            evaluation = evaluate_sample(sample, scored, rollout_n=args.num_samples)
            append_jsonl_record(
                records_path,
                sample_record_from_evaluation(
                    evaluation,
                    checkpoint_step=checkpoint.step,
                    include_outputs=args.store_outputs,
                ),
            )
            processed_counter["count"] += 1
            logger.info(
                "Recorded global_step_%s %s sample %s: correct=%s/%s",
                checkpoint.step,
                benchmark,
                sample.sample_id,
                evaluation.correct_count,
                args.num_samples,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            for _ in range(max_concurrency):
                submit_next(executor)
            while in_flight:
                done, _ = concurrent.futures.wait(
                    in_flight,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    task = in_flight.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:  # pragma: no cover - request_rollout normally captures exceptions.
                        result = RolloutResult(task=task, error=repr(exc))
                    state = sample_states[(result.task.benchmark, result.task.sample_id)]
                    if result.error is not None:
                        state["errors"].append(result.error)
                    elif result.choice is not None:
                        state["choices"][result.task.rollout_idx] = result.choice
                    finalize_sample_if_ready(result.task)
                while len(in_flight) < max_concurrency and submit_next(executor):
                    pass

    summaries = []
    for benchmark in benchmarks:
        records_path, skipped_path, summary_path, _ = benchmark_paths[benchmark]
        records = load_jsonl_records(records_path)
        skipped_records = filtered_skipped_records(load_jsonl_records(skipped_path), records)
        summary = summarize_records(
            records,
            skipped_records,
            benchmark=benchmark,
            checkpoint_step=checkpoint.step,
            rollout_n=args.num_samples,
        )
        summary_payload = asdict(summary)
        write_json_atomic(summary_path, summary_payload)
        summaries.append(summary_payload)
    return summaries, stopped_early


def evaluate_benchmark(
    *,
    checkpoint: CheckpointInfo,
    benchmark: str,
    dataset_root: Path,
    results_dir: Path,
    tokenizer: Any,
    server: VLLMServer | None,
    args: argparse.Namespace,
    processed_counter: dict[str, int],
) -> tuple[dict[str, Any], bool]:
    summaries, stopped_early = evaluate_benchmarks(
        checkpoint=checkpoint,
        benchmarks=[benchmark],
        dataset_root=dataset_root,
        results_dir=results_dir,
        tokenizer=tokenizer,
        server=server,
        args=args,
        processed_counter=processed_counter,
    )
    return summaries[0], stopped_early


def find_wandb_run(args: argparse.Namespace) -> Any:
    import wandb

    api = wandb.Api()
    if args.wandb_run_id:
        return api.run(f"{args.wandb_entity}/{args.wandb_project}/{args.wandb_run_id}")
    if not args.wandb_run_name:
        raise ValueError("--wandb-run-name or --wandb-run-id is required unless --skip-wandb is set.")

    runs = list(
        api.runs(
            f"{args.wandb_entity}/{args.wandb_project}",
            filters={"display_name": args.wandb_run_name},
        )
    )
    exact = [run for run in runs if run.name == args.wandb_run_name]
    if len(exact) != 1:
        raise RuntimeError(
            f"Expected exactly one W&B run named {args.wandb_run_name!r} in "
            f"{args.wandb_entity}/{args.wandb_project}, found {len(exact)}."
        )
    return exact[0]


def check_wandb_reachable(args: argparse.Namespace) -> None:
    if args.wandb_preflight_timeout is None or args.wandb_preflight_timeout <= 0:
        return
    base_url = os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai")
    parsed = urlparse(base_url)
    hostname = parsed.hostname
    if not hostname:
        raise RuntimeError(f"Invalid WANDB_BASE_URL: {base_url!r}")
    try:
        socket.getaddrinfo(hostname, parsed.port or 443)
    except socket.gaierror as exc:
        raise RuntimeError(
            f"W&B host {hostname!r} is not resolvable from this machine. "
            "Use --skip-wandb for dry debugging, configure DNS/proxy, or run evaluation where W&B is reachable."
        ) from exc

    try:
        requests.get(base_url, timeout=args.wandb_preflight_timeout)
    except requests.RequestException as exc:
        raise RuntimeError(
            f"W&B base URL {base_url!r} is not reachable within {args.wandb_preflight_timeout}s. "
            "Use --skip-wandb for dry debugging or configure WANDB_BASE_URL/proxy."
        ) from exc


def init_wandb_for_eval(public_run: Any, args: argparse.Namespace) -> Any:
    import wandb

    init_kwargs: dict[str, Any] = {
        "entity": args.wandb_entity,
        "project": args.wandb_project,
        "id": public_run.id,
        "name": public_run.name,
        "resume": "must",
        "job_type": "eval",
        "reinit": True,
        "settings": wandb.Settings(init_timeout=args.wandb_init_timeout),
    }
    init_mode = args.wandb_init_mode
    if init_mode is None and getattr(public_run, "state", None) == "running":
        init_mode = "shared"
    if init_mode:
        init_kwargs["mode"] = init_mode

    run = wandb.init(**init_kwargs)
    wandb.define_metric("eval/*", step_metric="eval/checkpoint_step")
    return run


def log_summary_to_wandb_once(
    *,
    wandb_run: Any,
    summary: dict[str, Any],
    logged_path: Path,
) -> None:
    metrics = summary_to_metrics_payload(summary)
    fingerprint = hashlib.sha256(json.dumps(metrics, sort_keys=True).encode("utf-8")).hexdigest()
    if logged_path.exists():
        try:
            logged = json.loads(logged_path.read_text(encoding="utf-8"))
            if logged.get("fingerprint") == fingerprint:
                logger.info("W&B metrics already logged for %s", logged_path.parent)
                return
        except json.JSONDecodeError:
            pass

    benchmark = summary["benchmark"]
    step = int(summary["checkpoint_step"])
    payload = {"eval/checkpoint_step": step}
    for key, value in metrics.items():
        payload[f"eval/{benchmark}/{key}"] = value
    wandb_run.log(payload)
    write_json_atomic(logged_path, {"fingerprint": fingerprint, "metrics": metrics, "logged_at": time.time()})


def summary_to_metrics_payload(summary: dict[str, Any]) -> dict[str, float | int]:
    rollout_n = int(summary["rollout_n"])
    return {
        f"avg@{rollout_n}": float(summary["avg_at_n"]),
        f"pass@{rollout_n}": float(summary["pass_at_n"]),
        "valid_samples": int(summary["valid_samples"]),
        "skipped_samples": int(summary["skipped_samples"]),
        "correct_completions": int(summary["correct_completions"]),
        "total_completions": int(summary["total_completions"]),
        "invalid_pred_rate": float(summary["invalid_pred_rate"]),
        "answer_pattern_rate": float(summary["answer_pattern_rate"]),
        "boxed_pattern_rate": float(summary["boxed_pattern_rate"]),
        "accepted_answer_format_rate": float(summary["accepted_answer_format_rate"]),
        "length_skip_count": int(summary["length_skip_count"]),
    }


def resolve_default_local_dir(args: argparse.Namespace) -> tuple[str, Any | None, Any | None]:
    if args.skip_wandb:
        if not args.default_local_dir and not args.model_path and not args.baseline_model_path:
            raise ValueError(
                "--default-local-dir, --model-path, or --baseline-model-path is required when --skip-wandb is set."
            )
        return args.default_local_dir or "", None, None

    check_wandb_reachable(args)
    public_run = find_wandb_run(args)
    default_local_dir = args.default_local_dir or find_default_local_dir_in_wandb_config(dict(public_run.config))
    if not default_local_dir and not args.model_path:
        raise RuntimeError(
            f"Could not find trainer.default_local_dir in W&B run config for {public_run.path}. "
            "Pass --default-local-dir explicitly."
        )
    wandb_run = init_wandb_for_eval(public_run, args)
    return default_local_dir or "", public_run, wandb_run


def checkpoint_from_hf_model_path(model_path: str | Path, *, step: int, flag_name: str) -> CheckpointInfo:
    path = Path(model_path).expanduser()
    if not hf_model_ready(path):
        raise FileNotFoundError(f"{flag_name} does not look like a ready HuggingFace model: {path}")
    return CheckpointInfo(step=step, path=path, actor_dir=path, hf_dir=path)


def resolve_baseline_model_path(
    *,
    args: argparse.Namespace,
    public_run: Any | None,
    remaps: list[tuple[str, str]],
) -> str | None:
    if not args.include_baseline_model or args.model_path:
        return None

    baseline_model_path = args.baseline_model_path
    explicit = bool(baseline_model_path)
    source = "--baseline-model-path"
    if not baseline_model_path and public_run is not None:
        baseline_model_path = find_baseline_model_path_in_wandb_config(dict(public_run.config))
        source = "W&B config actor_rollout_ref.model.path"
    if not baseline_model_path:
        return None

    baseline_model_path = apply_path_remaps(baseline_model_path, remaps)
    if hf_model_ready(baseline_model_path):
        logger.info("Including baseline model as global_step_0 from %s: %s", source, baseline_model_path)
        return baseline_model_path
    if explicit:
        raise FileNotFoundError(f"{source} does not look like a ready HuggingFace model: {baseline_model_path}")
    logger.warning(
        "Skipping baseline model from %s because it is not a ready local HuggingFace directory: %s",
        source,
        baseline_model_path,
    )
    return None


def discover_evaluation_checkpoints(
    *,
    default_local_dir: str,
    checkpoint_steps: list[int] | None,
    baseline_model_path: str | None,
    args: argparse.Namespace,
) -> list[CheckpointInfo]:
    if args.model_path:
        return [checkpoint_from_hf_model_path(args.model_path, step=args.model_step, flag_name="--model-path")]

    step_filter = set(checkpoint_steps) if checkpoint_steps is not None else None
    checkpoints: list[CheckpointInfo] = []
    if baseline_model_path and (step_filter is None or 0 in step_filter):
        checkpoints.append(
            checkpoint_from_hf_model_path(baseline_model_path, step=0, flag_name="--baseline-model-path")
        )
    elif baseline_model_path:
        logger.info("Skipping baseline global_step_0 because --checkpoint-steps excludes 0.")

    if default_local_dir:
        nonzero_steps = None
        if step_filter is not None:
            nonzero_steps = sorted(step for step in step_filter if step != 0)
        checkpoints.extend(discover_checkpoints(default_local_dir, checkpoint_steps=nonzero_steps))
    elif not checkpoints:
        raise ValueError("--default-local-dir is required unless --model-path or --baseline-model-path supplies work.")

    by_step: dict[int, CheckpointInfo] = {}
    for checkpoint in checkpoints:
        if checkpoint.step in by_step:
            logger.warning(
                "Duplicate global_step_%s discovered; keeping %s and skipping %s",
                checkpoint.step,
                by_step[checkpoint.step].hf_dir,
                checkpoint.hf_dir,
            )
            continue
        by_step[checkpoint.step] = checkpoint
    return [by_step[step] for step in sorted(by_step)]


def checkpoint_has_remaining_work(
    checkpoint: CheckpointInfo,
    *,
    benchmarks: list[str],
    dataset_root: Path,
    results_dir: Path,
    args: argparse.Namespace,
) -> bool:
    for benchmark in benchmarks:
        records_path, skipped_path, _, _ = records_paths(results_dir, checkpoint, benchmark)
        if benchmark_has_remaining(dataset_root, benchmark, records_path, skipped_path, args):
            return True
    return False


def evaluate_checkpoint(
    *,
    checkpoint: CheckpointInfo,
    benchmarks: list[str],
    dataset_root: Path,
    results_dir: Path,
    wandb_run: Any | None,
    args: argparse.Namespace,
    processed_counter: dict[str, int],
) -> bool:
    needs_server = checkpoint_has_remaining_work(
        checkpoint,
        benchmarks=benchmarks,
        dataset_root=dataset_root,
        results_dir=results_dir,
        args=args,
    )
    if needs_server:
        ensure_checkpoint_merged(checkpoint, args)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint.hf_dir, trust_remote_code=args.trust_remote_code)
    else:
        logger.info("No remaining samples for global_step_%s; recomputing summaries only.", checkpoint.step)
        tokenizer = None

    server_context = (
        VLLMServer(checkpoint.hf_dir, checkpoint.step, results_dir, args)
        if needs_server and not args.dry_run
        else contextlib.nullcontext(None)
    )
    with server_context as server:
        summaries, stopped_early = evaluate_benchmarks(
            checkpoint=checkpoint,
            benchmarks=benchmarks,
            dataset_root=dataset_root,
            results_dir=results_dir,
            tokenizer=tokenizer,
            server=server,
            args=args,
            processed_counter=processed_counter,
        )
        for summary in summaries:
            benchmark = str(summary["benchmark"])
            records_path, _, _, logged_path = records_paths(results_dir, checkpoint, benchmark)
            if wandb_run is not None and not args.dry_run and records_path.exists():
                log_summary_to_wandb_once(wandb_run=wandb_run, summary=summary, logged_path=logged_path)

    return stopped_early


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")

    default_local_dir, public_run, wandb_run = resolve_default_local_dir(args)
    remaps = parse_path_remaps(args.path_remap)
    default_local_dir = apply_path_remaps(default_local_dir, remaps)
    baseline_model_path = resolve_baseline_model_path(args=args, public_run=public_run, remaps=remaps)
    dataset_root = Path(args.dataset_root).expanduser()
    if args.results_dir:
        results_dir = Path(args.results_dir).expanduser()
    elif args.model_path:
        results_dir = Path.cwd() / "eval_math_benchmarks" / Path(args.model_path).expanduser().name
    else:
        results_dir = Path(default_local_dir) / "eval_math_benchmarks"
    benchmarks = parse_csv(args.benchmarks, BENCHMARK_NAMES)
    checkpoint_steps = [int(step) for step in parse_csv(args.checkpoint_steps)] if args.checkpoint_steps else None

    logger.info("Evaluation target default_local_dir=%s", default_local_dir or "<direct-model>")
    if args.model_path:
        logger.info("Direct HuggingFace model path=%s", args.model_path)
    if baseline_model_path:
        logger.info("Baseline HuggingFace model path=%s", baseline_model_path)
    logger.info("Results directory=%s", results_dir)
    if public_run is not None:
        logger.info(
            "W&B run: entity=%s project=%s name=%s id=%s",
            args.wandb_entity,
            args.wandb_project,
            public_run.name,
            public_run.id,
        )

    processed_counter = {"count": 0}
    idle_since = time.time()
    try:
        while True:
            checkpoints = discover_evaluation_checkpoints(
                default_local_dir=default_local_dir,
                checkpoint_steps=checkpoint_steps,
                baseline_model_path=baseline_model_path,
                args=args,
            )
            loop_start_count = processed_counter["count"]
            if args.dry_run:
                logger.info("Discovered checkpoints: %s", [checkpoint.step for checkpoint in checkpoints])
                for benchmark in benchmarks:
                    samples = load_benchmark_samples(dataset_root, benchmark, max_samples=args.max_samples)
                    logger.info("Benchmark %s samples=%s", benchmark, len(samples))
                return

            did_work = False
            for checkpoint in checkpoints:
                if args.stop_after_samples is not None and processed_counter["count"] >= args.stop_after_samples:
                    logger.info("Reached --stop-after-samples=%s", args.stop_after_samples)
                    return
                stopped = evaluate_checkpoint(
                    checkpoint=checkpoint,
                    benchmarks=benchmarks,
                    dataset_root=dataset_root,
                    results_dir=results_dir,
                    wandb_run=wandb_run,
                    args=args,
                    processed_counter=processed_counter,
                )
                did_work = did_work or stopped or processed_counter["count"] > loop_start_count
                if stopped:
                    logger.info(
                        "Stopping early for resume/debug after processing %s samples.",
                        processed_counter["count"],
                    )
                    return

            if not args.watch:
                return
            if did_work:
                idle_since = time.time()
            elif args.idle_timeout > 0 and time.time() - idle_since >= args.idle_timeout:
                logger.info("No new work for %.1fs; exiting watch mode.", args.idle_timeout)
                return
            logger.info("Watch mode sleeping %.1fs before rescanning checkpoints.", args.poll_interval)
            time.sleep(args.poll_interval)
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
