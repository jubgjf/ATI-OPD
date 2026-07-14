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

"""Show a small set of model generations from a verl checkpoint."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import AutoConfig, AutoTokenizer

from verl.trainer.main_math_benchmark_eval import (
    VLLMServer,
    count_tokens,
    ensure_checkpoint_merged,
    extract_choice_generation,
    request_chat_completions,
)
from verl.utils.evaluation.math_benchmark import (
    CheckpointInfo,
    append_jsonl_record,
    hf_model_ready,
    normalize_ground_truth,
    parse_step_from_checkpoint_name,
    render_prompt_with_chat_template,
    score_completion,
    write_json_atomic,
)

logger = logging.getLogger(__name__)

PROMPT_FIELDS = ("prompt", "messages", "chat", "question", "problem", "input")
ANSWER_FIELDS = ("answer", "ground_truth", "final_answer", "solution")
MODEL_MAX_LENGTH_FIELDS = (
    "max_position_embeddings",
    "seq_length",
    "max_sequence_length",
    "n_positions",
    "model_max_length",
)


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

    parser.add_argument("--checkpoint-path", default=os.environ.get("CHECKPOINT_PATH"), required=False)
    parser.add_argument("--data-file", default=os.environ.get("DATA_FILE"), required=False)
    parser.add_argument("--num-cases", type=int, default=_env_int("NUM_CASES", 10))
    parser.add_argument("--sample-offset", type=int, default=_env_int("SAMPLE_OFFSET", 0))
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=_env_bool("SHUFFLE", False))
    parser.add_argument("--seed", type=int, default=_env_int("SEED", 42))
    parser.add_argument("--output-jsonl", default=os.environ.get("OUTPUT_JSONL"))
    parser.add_argument("--print-max-chars", type=int, default=_env_int("PRINT_MAX_CHARS", 12000))
    parser.add_argument("--max-completion-tokens", default=os.environ.get("MAX_COMPLETION_TOKENS", "auto"))
    parser.add_argument(
        "--fit-context",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("FIT_CONTEXT", True),
        help="Reduce per-request max_tokens so prompt tokens plus completion tokens fit the model context.",
    )
    parser.add_argument("--temperature", type=float, default=_env_float("TEMPERATURE", 0.7))
    parser.add_argument("--top-p", type=float, default=_env_float("TOP_P", 0.95))
    parser.add_argument("--top-k", type=int, default=_env_int("TOP_K"))
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("ENABLE_THINKING", True),
    )
    parser.add_argument(
        "--verifier",
        choices=("math_dapo", "math_verify", "hybrid"),
        default=os.environ.get("VERIFIER", "hybrid"),
    )
    parser.add_argument("--math-verify-timeout", type=float, default=_env_float("MATH_VERIFY_TIMEOUT", 30.0))
    parser.add_argument("--request-timeout", type=float, default=_env_float("REQUEST_TIMEOUT", None))

    parser.add_argument("--vllm-bin", default=os.environ.get("VLLM_BIN", "vllm"))
    parser.add_argument("--served-model-name", default=os.environ.get("SERVED_MODEL_NAME", "case-model"))
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
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=_env_float("GPU_MEMORY_UTILIZATION", 0.9),
    )
    parser.add_argument("--max-model-len", type=int, default=_env_int("MAX_MODEL_LEN"))
    parser.add_argument("--max-num-seqs", type=int, default=_env_int("MAX_NUM_SEQS", 16))
    parser.add_argument("--max-num-batched-tokens", type=int, default=_env_int("MAX_NUM_BATCHED_TOKENS", 32768))
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("TRUST_REMOTE_CODE", False),
    )
    parser.add_argument("--vllm-extra-arg", action="append", default=[])
    parser.add_argument(
        "--enable-reasoning-parser",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("ENABLE_REASONING_PARSER", False),
    )
    parser.add_argument("--reasoning-parser", default=os.environ.get("REASONING_PARSER", "deepseek_r1"))

    parser.add_argument("--skip-merge", action=argparse.BooleanOptionalAction, default=_env_bool("SKIP_MERGE", False))
    parser.add_argument("--force-merge", action=argparse.BooleanOptionalAction, default=_env_bool("FORCE_MERGE", False))
    parser.add_argument("--merge-timeout", type=float, default=_env_float("MERGE_TIMEOUT", None))
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=_env_bool("DRY_RUN", False))
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"))

    args = parser.parse_args()
    if not args.checkpoint_path:
        raise ValueError("--checkpoint-path or CHECKPOINT_PATH is required")
    if not args.data_file:
        raise ValueError("--data-file or DATA_FILE is required")
    if args.num_cases <= 0:
        raise ValueError("--num-cases must be positive")
    return args


def _loads_json_like(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("{", "[")):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


def _to_builtin(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _normalize_messages(value: Any) -> list[dict[str, str]]:
    value = _to_builtin(_loads_json_like(value))
    if isinstance(value, dict) and "role" in value and "content" in value:
        value = [value]
    elif isinstance(value, str):
        value = [{"role": "user", "content": value}]
    if not isinstance(value, list):
        raise ValueError(f"Unsupported prompt value type: {type(value).__name__}")

    messages = []
    for message in value:
        message = _to_builtin(message)
        if not isinstance(message, dict):
            raise ValueError(f"Prompt message must be a dict, got {type(message).__name__}")
        role = str(message.get("role", "user"))
        content = message.get("content")
        if content is None:
            raise ValueError(f"Prompt message is missing content: {message!r}")
        messages.append({"role": role, "content": str(content)})
    if not messages:
        raise ValueError("Prompt contains no messages")
    return messages


def extract_messages(row: dict[str, Any], idx: int) -> list[dict[str, str]]:
    for field in PROMPT_FIELDS:
        value = row.get(field)
        if value is not None and not (isinstance(value, float) and pd.isna(value)):
            return _normalize_messages(value)
    raise ValueError(f"Row {idx} must contain one of prompt fields: {', '.join(PROMPT_FIELDS)}")


def extract_ground_truth(row: dict[str, Any]) -> str | None:
    reward_model = _to_builtin(_loads_json_like(row.get("reward_model")))
    if isinstance(reward_model, dict) and reward_model.get("ground_truth") is not None:
        return normalize_ground_truth(reward_model["ground_truth"])

    for field in ANSWER_FIELDS:
        value = row.get(field)
        if value is not None and not (isinstance(value, float) and pd.isna(value)):
            text = str(value).strip()
            if text:
                return normalize_ground_truth(text)
    return None


def load_rows(data_file: str | Path) -> list[dict[str, Any]]:
    path = Path(data_file).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Data file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        dataframe = pd.read_parquet(path)
        return dataframe.to_dict(orient="records")
    if suffix == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    rows.append(json.loads(stripped))
        return rows
    if suffix == ".json":
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("data", "train", "test", "validation"):
                value = data.get(key)
                if isinstance(value, list):
                    return value
        raise ValueError(f"Unsupported JSON dataset shape in {path}")
    raise ValueError(f"Unsupported data file format: {path}")


def select_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[tuple[int, dict[str, Any]]]:
    indexed_rows = list(enumerate(rows))
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(indexed_rows)
        return indexed_rows[: args.num_cases]

    start = max(0, args.sample_offset)
    end = min(len(indexed_rows), start + args.num_cases)
    return indexed_rows[start:end]


def _checkpoint_step_from_path(path: Path) -> int:
    for candidate in (path.name, path.parent.name):
        step = parse_step_from_checkpoint_name(candidate)
        if step is not None:
            return step
    return 0


def resolve_checkpoint(checkpoint_path: str | Path) -> CheckpointInfo:
    path = Path(checkpoint_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
    step = _checkpoint_step_from_path(path)

    if hf_model_ready(path):
        return CheckpointInfo(step=step, path=path, actor_dir=path, hf_dir=path)
    if (path / "actor").is_dir():
        actor_dir = path / "actor"
        return CheckpointInfo(step=step, path=path, actor_dir=actor_dir, hf_dir=actor_dir / "huggingface")
    if (path / "fsdp_config.json").exists():
        actor_dir = path
        return CheckpointInfo(step=step, path=path.parent, actor_dir=actor_dir, hf_dir=actor_dir / "huggingface")
    raise ValueError(
        f"Checkpoint path must be a HuggingFace model, a global_step_* directory, or an actor directory: {path}"
    )


def model_context_length(model_dir: str | Path, tokenizer: Any, *, trust_remote_code: bool = False) -> int:
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
    candidates = []
    for field in MODEL_MAX_LENGTH_FIELDS:
        value = getattr(config, field, None)
        if isinstance(value, int) and 0 < value < 1_000_000_000:
            candidates.append(value)
    tokenizer_length = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_length, int) and 0 < tokenizer_length < 1_000_000_000:
        candidates.append(tokenizer_length)
    if not candidates:
        raise ValueError(f"Could not infer model maximum context length from {model_dir}")
    return max(candidates)


def resolve_max_completion_tokens(args: argparse.Namespace, model_dir: Path, tokenizer: Any) -> int:
    value = str(args.max_completion_tokens).strip().lower()
    if value == "auto":
        return model_context_length(model_dir, tokenizer, trust_remote_code=args.trust_remote_code)
    resolved = int(value)
    if resolved <= 0:
        raise ValueError("--max-completion-tokens must be positive or auto")
    return resolved


def prompt_token_count(tokenizer: Any, messages: list[dict[str, str]], *, enable_thinking: bool) -> int | None:
    try:
        prompt = render_prompt_with_chat_template(tokenizer, messages, enable_thinking=enable_thinking)
        return count_tokens(tokenizer, prompt)
    except Exception as exc:
        logger.warning("Failed to count chat prompt tokens; using unadjusted max_tokens: %r", exc)
        return None


def request_args_for_case(
    args: argparse.Namespace,
    *,
    tokenizer: Any,
    messages: list[dict[str, str]],
) -> tuple[argparse.Namespace, int | None]:
    prompt_tokens = prompt_token_count(tokenizer, messages, enable_thinking=args.enable_thinking)
    request_args = argparse.Namespace(**vars(args))
    if args.fit_context and prompt_tokens is not None:
        request_args.max_tokens = max(1, args.max_tokens - prompt_tokens)
    return request_args, prompt_tokens


def short_text(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    keep = max(0, limit - 200)
    return text[:keep] + f"\n\n...[truncated {len(text) - keep} chars]..."


def print_case(record: dict[str, Any], *, print_max_chars: int) -> None:
    print("\n" + "=" * 100, flush=True)
    print(f"case_index={record['case_index']} row_index={record['row_index']}", flush=True)
    print(f"finish_reason={record['finish_reason']} response_tokens={record['response_token_count']}", flush=True)
    if record.get("ground_truth") is not None:
        print(
            "ground_truth={ground_truth} pred={pred!r} acc={acc} verified_by={verified_by}".format(**record),
            flush=True,
        )
    print("\n--- prompt ---", flush=True)
    print(short_text(record["prompt_text"], print_max_chars), flush=True)
    print("\n--- final content ---", flush=True)
    print(short_text(record["final_content"], print_max_chars), flush=True)
    if record.get("reasoning_content"):
        print("\n--- reasoning content ---", flush=True)
        print(short_text(record["reasoning_content"], print_max_chars), flush=True)


def run_cases(args: argparse.Namespace) -> None:
    checkpoint = resolve_checkpoint(args.checkpoint_path)
    ensure_checkpoint_merged(checkpoint, args)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint.hf_dir, trust_remote_code=args.trust_remote_code)
    args.max_tokens = resolve_max_completion_tokens(args, checkpoint.hf_dir, tokenizer)
    logger.info("Using max_completion_tokens=%s", args.max_tokens)

    rows = load_rows(args.data_file)
    selected_rows = select_rows(rows, args)
    if not selected_rows:
        raise ValueError(f"No rows selected from {args.data_file}")

    if args.output_jsonl:
        output_jsonl = Path(args.output_jsonl).expanduser()
    else:
        output_jsonl = (
            checkpoint.path
            / "case_analysis"
            / f"{Path(args.data_file).stem}_offset{args.sample_offset}_n{len(selected_rows)}.jsonl"
        )
    summary_path = output_jsonl.with_suffix(".summary.json")
    logger.info("Output JSONL=%s", output_jsonl)

    if args.dry_run:
        logger.info("Dry run selected row indices: %s", [idx for idx, _ in selected_rows])
        return

    records = []
    with VLLMServer(checkpoint.hf_dir, checkpoint.step, output_jsonl.parent, args) as server:
        for case_index, (row_index, row) in enumerate(selected_rows):
            messages = extract_messages(row, row_index)
            ground_truth = extract_ground_truth(row)
            request_args, prompt_tokens = request_args_for_case(args, tokenizer=tokenizer, messages=messages)
            choice = request_chat_completions(
                base_url=server.base_url,
                model_name=args.served_model_name,
                messages=messages,
                n=1,
                args=request_args,
            )[0]
            generation = extract_choice_generation(choice)
            text = generation["text"] or ""
            scored = score_completion(
                text,
                ground_truth or "",
                finish_reason=choice.get("finish_reason"),
                response_token_count=count_tokens(tokenizer, text),
                reasoning_content=generation["reasoning_content"],
                content=generation["content"],
                verifier=args.verifier if ground_truth is not None else "math_dapo",
                math_verify_timeout=args.math_verify_timeout,
            )
            record = {
                "status": "ok",
                "case_index": case_index,
                "row_index": row_index,
                "checkpoint_step": checkpoint.step,
                "checkpoint_path": str(checkpoint.path),
                "model_path": str(checkpoint.hf_dir),
                "data_file": str(Path(args.data_file).expanduser()),
                "messages": messages,
                "prompt_text": "\n".join(str(message.get("content", "")) for message in messages),
                "ground_truth": ground_truth,
                "finish_reason": choice.get("finish_reason"),
                "response_token_count": count_tokens(tokenizer, text),
                "text": text,
                "content": generation["content"],
                "reasoning_content": generation["reasoning_content"],
                "final_content": scored.final_content,
                "acc": scored.acc if ground_truth is not None else None,
                "pred": scored.pred if ground_truth is not None else "",
                "invalid_pred": scored.invalid_pred if ground_truth is not None else None,
                "has_answer_pattern": scored.has_answer_pattern,
                "has_boxed_pattern": scored.has_boxed_pattern,
                "has_accepted_answer_format": scored.has_accepted_answer_format,
                "verified_by": scored.verified_by if ground_truth is not None else "",
                "math_verify_score": scored.math_verify_score if ground_truth is not None else None,
                "model_max_completion_tokens": args.max_tokens,
                "request_max_completion_tokens": request_args.max_tokens,
                "prompt_token_count": prompt_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "timestamp": time.time(),
            }
            append_jsonl_record(output_jsonl, record)
            records.append(record)
            print_case(record, print_max_chars=args.print_max_chars)

    scored_records = [record for record in records if record["acc"] is not None]
    summary = {
        "checkpoint_step": checkpoint.step,
        "checkpoint_path": str(checkpoint.path),
        "model_path": str(checkpoint.hf_dir),
        "data_file": str(Path(args.data_file).expanduser()),
        "num_cases": len(records),
        "num_scored_cases": len(scored_records),
        "accuracy": (
            sum(1 for record in scored_records if record["acc"]) / len(scored_records) if scored_records else None
        ),
        "boxed_pattern_rate": (
            sum(1 for record in records if record["has_boxed_pattern"]) / len(records) if records else None
        ),
        "accepted_answer_format_rate": (
            sum(1 for record in records if record["has_accepted_answer_format"]) / len(records) if records else None
        ),
        "model_max_completion_tokens": args.max_tokens,
        "output_jsonl": str(output_jsonl),
    }
    write_json_atomic(summary_path, summary)
    print("\n" + "=" * 100, flush=True)
    print("summary", json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_cases(args)


if __name__ == "__main__":
    main()
