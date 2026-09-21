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

"""Evaluate local HF or FSDP checkpoints with the training legacy math protocol.

Run with ``MODEL_PATH=/local/checkpoint python -B -u -m verl.utils.evaluation.standalone_math``
in an existing vLLM environment. The Ascend launcher is ``run/eval-single.sh``.
FSDP inputs are merged into actor/huggingface when weights are missing; set
EVAL_FORCE_MERGE=True to rebuild existing weights. Evaluation output uses only stdout.
Megatron checkpoints already exported under model/huggingface are loaded directly.
Framework runtime caches are allowed.
By default only HMMT is evaluated; EVAL_ALL=1 adds AIME24/25, AMC23 and AMO-Bench-P.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from verl.trainer.ppo.metric_utils import process_validation_metrics
from verl.utils.chat_template import apply_chat_template
from verl.utils.evaluation.math_benchmark import (
    AMO_BENCH_P_ANSWER_TYPES,
    EvalSample,
    build_math_chat_messages,
    load_benchmark_samples,
)
from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score.reward_extra import collate_reward_extra_infos
from verl.utils.tokenizer import hf_tokenizer, normalize_token_ids

BENCHMARKS = ("hmmt_feb_2026", "hmmt_nov_2025")
BENCHMARK_PATHS = {
    "hmmt_feb_2026": "MathArena/hmmt_feb_2026",
    "hmmt_nov_2025": "MathArena/hmmt_nov_2025",
    "aime24": "math-ai/aime24",
    "aime25": "math-ai/aime25",
    "amc23": "math-ai/amc23",
    "AMO-Bench": "meituan-longcat/AMO-Bench",
}
METRIC_PREFIX = "test-legacy"
HF_WEIGHT_FILES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)


def _env_value(env: Mapping[str, str], *names: str, default: Any = None) -> Any:
    # Match the launchers' ${NAME:-default}, including empty-string handling.
    return next((env[name] for name in names if env.get(name)), default)


def _env_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    value = str(_env_value(env, name, default=str(default))).lower()
    if value not in {"true", "false", "1", "0"}:
        raise ValueError(f"{name} must be True or False, got {value!r}.")
    return value in {"true", "1"}


@dataclass(frozen=True)
class EvaluationConfig:
    """Resolved standalone settings; sampling defaults match run/common.sh's legacy profile."""

    model_path: str
    dataset_root: str
    chat_template_model: str | None = None
    enable_thinking: bool = True
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = -1
    do_sample: bool = True
    rollout_n: int = 16
    prompt_length: int = 1024
    response_length: int = 23552
    generation_max_tokens: int = 23552
    max_model_len: int = 24577
    tensor_parallel_size: int = 2
    gpu_memory_utilization: float = 0.6
    max_num_batched_tokens: int = 2048
    max_num_seqs: int = 128
    batch_size: int = 256
    max_samples: int = -1
    seed: int = 0
    enforce_eager: bool = False
    force_merge: bool = False
    eval_all: bool = False

    @property
    def benchmarks(self) -> tuple[str, ...]:
        """Keep HMMT first, optionally followed by the four training math benchmarks."""
        return tuple(BENCHMARK_PATHS) if self.eval_all else BENCHMARKS

    def __post_init__(self) -> None:
        for name in (
            "rollout_n",
            "prompt_length",
            "response_length",
            "generation_max_tokens",
            "max_model_len",
            "tensor_parallel_size",
            "max_num_batched_tokens",
            "max_num_seqs",
            "batch_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}.")
        if self.generation_max_tokens > self.response_length:
            raise ValueError("TEST_BENCHMARKS_MAX_RESPONSE_LENGTH cannot exceed MAX_RESPONSE_LENGTH.")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("TEST_TEMPERATURE must be finite and non-negative.")
        if not 0 < self.top_p <= 1:
            raise ValueError("TEST_TOP_P must be in (0, 1].")
        if self.top_k != -1 and self.top_k <= 0:
            raise ValueError("TEST_TOP_K must be -1 or positive.")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("STUDENT_GPU_MEM_UTIL must be in (0, 1].")
        if self.max_samples != -1 and self.max_samples <= 0:
            raise ValueError("TEST_BENCHMARK_MAX_SAMPLES must be -1 (all questions) or positive.")
        if not 0 <= self.seed < 2**32:
            raise ValueError("EVAL_SEED must be in [0, 2**32).")

    @classmethod
    def from_environment(cls, env: Mapping[str, str] | None = None) -> EvaluationConfig:
        """Resolve the existing test/rollout environment variables without loading a training config."""
        env = os.environ if env is None else env
        model_path = _env_value(env, "MODEL_PATH")
        if not model_path:
            raise ValueError("MODEL_PATH is required and must name a local HF or FSDP checkpoint directory.")
        if _env_bool(env, "ENABLE_TOOL_CALL", False) or _env_bool(env, "SOD_STYLE", False):
            raise ValueError("Standalone math evaluation requires ENABLE_TOOL_CALL=False and SOD_STYLE=False.")
        if _env_value(env, "TEST_BENCHMARK_REWARD_DATA_SOURCE", default="math_verify") != "math_verify":
            raise ValueError("Standalone math evaluation requires TEST_BENCHMARK_REWARD_DATA_SOURCE=math_verify.")
        prompt_length = int(_env_value(env, "MAX_PROMPT_LENGTH", default=1024))
        response_length = int(_env_value(env, "MAX_RESPONSE_LENGTH", default=23552))
        default_context = prompt_length + response_length + 1
        if env.get("TRAIN_DATASET") == "nemotron_cascade_2_rl_mopd":
            default_context = 32768
        template_path = _env_value(env, "CHAT_TEMPLATE_MODEL")
        repo_root = Path(__file__).resolve().parents[3]
        return cls(
            model_path=str(Path(model_path).expanduser().resolve()),
            dataset_root=str(
                Path(_env_value(env, "TEST_BENCHMARK_ROOT", default=repo_root / "dataset")).expanduser().resolve()
            ),
            chat_template_model=str(Path(template_path).expanduser().resolve()) if template_path else None,
            enable_thinking=_env_bool(env, "ENABLE_THINKING", True),
            temperature=float(_env_value(env, "TEST_TEMPERATURE", default=0.7)),
            top_p=float(_env_value(env, "TEST_TOP_P", default=0.95)),
            top_k=int(_env_value(env, "TEST_TOP_K", "ROLLOUT_TOP_K", default=-1)),
            do_sample=_env_bool(env, "TEST_DO_SAMPLE", True),
            rollout_n=int(_env_value(env, "TEST_BENCHMARK_N", "TEST_N", default=16)),
            prompt_length=prompt_length,
            response_length=response_length,
            generation_max_tokens=int(_env_value(env, "TEST_BENCHMARKS_MAX_RESPONSE_LENGTH", default=response_length)),
            max_model_len=int(
                _env_value(
                    env,
                    "ROLLOUT_MAX_MODEL_LEN",
                    "STUDENT_MAX_MODEL_LEN",
                    "MAX_MODEL_LEN",
                    "MAX_NUM_TOKENS",
                    default=default_context,
                )
            ),
            tensor_parallel_size=int(_env_value(env, "STUDENT_TP", default=2)),
            gpu_memory_utilization=float(_env_value(env, "STUDENT_GPU_MEM_UTIL", default=0.6)),
            max_num_batched_tokens=int(_env_value(env, "ROLLOUT_MAX_NUM_BATCHED_TOKENS", default=2048)),
            max_num_seqs=int(_env_value(env, "ROLLOUT_MAX_NUM_SEQS", default=128)),
            batch_size=int(_env_value(env, "TEST_BENCHMARK_BATCH_SIZE", default=256)),
            max_samples=int(_env_value(env, "TEST_BENCHMARK_MAX_SAMPLES", default=-1)),
            seed=int(_env_value(env, "EVAL_SEED", default=0)),
            enforce_eager=_env_bool(env, "ROLLOUT_ENFORCE_EAGER", False),
            force_merge=_env_bool(env, "EVAL_FORCE_MERGE", False),
            eval_all=_env_bool(env, "EVAL_ALL", False),
        )


def _read_checkpoint_json(path: Path) -> dict[str, Any]:
    """Read checkpoint metadata with enough context to identify a damaged file."""
    try:
        contents = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Cannot read checkpoint JSON {path}: {exc}") from exc
    try:
        value = json.loads(contents.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        detail = " Empty file." if not contents else " Whitespace-only file." if not contents.strip() else ""
        raise ValueError(f"Invalid checkpoint JSON {path} ({len(contents)} bytes): {exc}.{detail}") from exc
    if not isinstance(value, dict):
        raise ValueError(
            f"Invalid checkpoint JSON {path} ({len(contents)} bytes): "
            f"expected a JSON object, got {type(value).__name__}."
        )
    return value


def _validate_hf_metadata(root: Path) -> None:
    """Check JSON assets before Transformers loads them or the merger replaces weights."""
    _read_checkpoint_json(root / "config.json")
    for filename in (
        "generation_config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "added_tokens.json",
        "vocab.json",
    ):
        path = root / filename
        if path.exists():
            _read_checkpoint_json(path)


def _check_readable_weights(paths: list[Path]) -> None:
    """Check read access without loading tensors, before inference or destructive merge cleanup."""
    for path in paths:
        try:
            with path.open("rb") as stream:
                stream.read(1)
        except OSError as exc:
            raise OSError(
                f"Cannot read checkpoint weights {path}: {exc}. "
                "Check file ownership and read permissions for the evaluation user."
            ) from exc


def validate_checkpoint(model_path: str) -> None:
    """Require valid local HF metadata and complete, nonempty weights, including indexed shards."""
    root = Path(model_path)
    if not root.is_dir():
        raise ValueError(f"MODEL_PATH must be a local checkpoint directory: {root}")
    _validate_hf_metadata(root)
    for filename in HF_WEIGHT_FILES:
        path = root / filename
        if not path.is_file():
            continue
        if filename.endswith(".index.json"):
            index = _read_checkpoint_json(path)
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError(f"Missing or empty weight_map in {path}")
            names = list(weight_map.values())
            if any(not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts for name in names):
                raise ValueError(f"Invalid local shard name in {path}")
            weights = [root / name for name in sorted(set(names))]
        else:
            weights = [path]
        missing = [str(weight) for weight in weights if not weight.is_file() or weight.stat().st_size == 0]
        if missing:
            raise ValueError(f"Missing or empty checkpoint weights: {missing}")
        _check_readable_weights(weights)
        return
    raise ValueError(f"No complete HF weights found in {root}; export/merge the checkpoint before evaluation.")


def _resolve_checkpoint_paths(model_path: str) -> tuple[Path, Path | None]:
    root = Path(model_path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"MODEL_PATH must be a local checkpoint directory: {root}")
    if (root / "config.json").is_file() or root.name == "huggingface":
        actor = None
        if root.name == "huggingface" and (root.parent.name == "actor" or (root.parent / "fsdp_config.json").is_file()):
            actor = root.parent
        return root, actor
    if (root / "actor").is_dir():
        actor = root / "actor"
    elif (root / "model" / "huggingface").is_dir():
        # Megatron SFT can export complete HF weights alongside its distributed checkpoint.
        return root / "model" / "huggingface", None
    elif root.name == "model" and (root / "huggingface").is_dir() and not (root / "fsdp_config.json").is_file():
        return root / "huggingface", None
    elif root.name == "actor" or (root / "fsdp_config.json").is_file() or (root / "huggingface").is_dir():
        actor = root
    else:
        raise ValueError(
            f"Unrecognized checkpoint directory: {root}. Set MODEL_PATH to a specific global_step_* directory, "
            "its actor or actor/huggingface directory, a Megatron model or model/huggingface directory "
            "with exported HF weights, or a complete HF model."
        )
    return actor / "huggingface", actor


def _validate_merge_source(actor: Path, hf_dir: Path) -> None:
    from transformers import AutoConfig

    config_path = actor / "fsdp_config.json"
    if not config_path.is_file():
        raise ValueError(f"Missing FSDP config: {config_path}")
    fsdp_config = _read_checkpoint_json(config_path)
    world_size = fsdp_config.get("world_size")
    if type(world_size) is not int or world_size <= 0:
        raise ValueError(f"Expected a positive integer world_size in {config_path}, got {world_size!r}.")
    shards = [actor / f"model_world_size_{world_size}_rank_{rank}.pt" for rank in range(world_size)]
    missing = [str(path) for path in shards if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise ValueError(f"Missing or empty FSDP model shards: {missing}")
    _check_readable_weights(shards)
    # Validate assets before removing any old HF weights. No model weights are
    # allocated here, and optimizer / RNG checkpoints are deliberately unused.
    _validate_hf_metadata(hf_dir)
    AutoConfig.from_pretrained(str(hf_dir), local_files_only=True, trust_remote_code=False)
    hf_tokenizer(str(hf_dir), local_files_only=True, trust_remote_code=False)
    if not os.access(hf_dir, os.W_OK | os.X_OK):
        raise PermissionError(f"HF output directory is not writable: {hf_dir}")


def _remove_hf_weights(hf_dir: Path) -> None:
    for path in hf_dir.iterdir():
        if path.name in HF_WEIGHT_FILES or re.fullmatch(
            r"(?:model-\d+-of-\d+\.safetensors|pytorch_model-\d+-of-\d+\.bin)", path.name
        ):
            path.unlink()


def prepare_checkpoint(model_path: str, *, force_merge: bool = False) -> str:
    """Resolve a local checkpoint, reusing HF weights or merging its FSDP actor in a subprocess."""
    hf_dir, actor = _resolve_checkpoint_paths(model_path)
    print(f"[checkpoint] input={model_path}, huggingface={hf_dir}", flush=True)
    if not force_merge:
        try:
            validate_checkpoint(str(hf_dir))
        except ValueError:
            if actor is None:
                raise
        else:
            print("[checkpoint] Reusing complete HF weights.", flush=True)
            return str(hf_dir)
    if actor is None:
        raise ValueError(f"EVAL_FORCE_MERGE requires a corresponding FSDP actor directory: {model_path}")
    _validate_merge_source(actor, hf_dir)
    print(f"[checkpoint] Merging FSDP model shards from {actor} into {hf_dir}.", flush=True)
    _remove_hf_weights(hf_dir)
    command = [
        sys.executable,
        "-B",
        "-u",
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str(actor),
        "--target_dir",
        str(hf_dir),
    ]
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    try:
        subprocess.run(command, check=True, env=env)
        validate_checkpoint(str(hf_dir))
    except BaseException:
        # A failed save may leave nonempty, truncated weights. Do not let a
        # subsequent evaluation mistake those files for a completed conversion.
        try:
            _remove_hf_weights(hf_dir)
        except OSError as exc:
            print(f"[checkpoint] Could not remove incomplete HF weights: {exc}", file=sys.stderr, flush=True)
        raise
    print(f"[checkpoint] HF conversion complete: {hf_dir}", flush=True)
    return str(hf_dir)


def load_tokenizer(config: EvaluationConfig):
    """Load only local tokenizer assets, optionally replacing the chat template in memory."""
    tokenizer = hf_tokenizer(config.model_path, local_files_only=True, trust_remote_code=False)
    if config.chat_template_model:
        if not Path(config.chat_template_model).is_dir():
            raise ValueError(f"CHAT_TEMPLATE_MODEL must be a local directory: {config.chat_template_model}")
        source = hf_tokenizer(config.chat_template_model, local_files_only=True, trust_remote_code=False)
        tokenizer.chat_template = source.chat_template
    if not tokenizer.chat_template:
        raise ValueError("The tokenizer must define a chat_template; set CHAT_TEMPLATE_MODEL to its training source.")
    return tokenizer


@dataclass(frozen=True)
class PreparedSample:
    """A benchmark question with its exact model input and available generation budget."""

    sample: EvalSample
    prompt_ids: list[int]
    max_tokens: int

    @property
    def uid(self) -> str:
        """Use the same stable question identity as MathBenchmarkDataset."""
        return f"{self.sample.benchmark}:{self.sample.sample_id}"


def resolve_benchmark_dirs(config: EvaluationConfig) -> dict[str, Path]:
    """Resolve selected datasets in provider, flat, or legacy MathArena-root layouts."""
    root = Path(config.dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"TEST_BENCHMARK_ROOT must be a local dataset directory: {root}")
    # Older standalone launchers pointed directly at datasets/MathArena.
    common_root = root.parent if root.name == "MathArena" else root
    directories = {}
    for benchmark in config.benchmarks:
        candidates = list(dict.fromkeys((common_root / BENCHMARK_PATHS[benchmark], root / benchmark)))
        directory = next((path for path in candidates if path.is_dir()), None)
        if directory is None:
            raise FileNotFoundError(f"Missing benchmark {benchmark}; tried: {', '.join(map(str, candidates))}")
        directories[benchmark] = directory
    return directories


def _validate_scoring_metadata(sample: EvalSample) -> None:
    if sample.benchmark == "AMO-Bench" or sample.grader == "amo_bench":
        question_id = str(sample.question_id)
        if (
            sample.grader != "amo_bench"
            or sample.answer_type not in AMO_BENCH_P_ANSWER_TYPES
            or not re.fullmatch(r"[0-9]+", question_id)
            or int(question_id) <= 0
        ):
            raise ValueError(
                f"AMO-Bench:{sample.sample_id} requires grader='amo_bench', a positive integer question_id "
                "and answer_type=number/set/variable."
            )


def prepare_samples(
    config: EvaluationConfig, tokenizer, *, benchmark_dirs: Mapping[str, Path] | None = None
) -> dict[str, list[PreparedSample]]:
    """Read and tokenize selected datasets before starting inference, without truncating questions."""
    benchmark_dirs = resolve_benchmark_dirs(config) if benchmark_dirs is None else benchmark_dirs
    prepared = {}
    for benchmark in config.benchmarks:
        # Pass the short name to the loader so metrics and question IDs match training.
        samples = load_benchmark_samples(benchmark_dirs[benchmark].parent, benchmark, max_samples=config.max_samples)
        if not samples:
            raise ValueError(f"No evaluation questions found for {benchmark}.")
        if len({sample.sample_id for sample in samples}) != len(samples):
            raise ValueError(f"Duplicate question IDs in {benchmark}.")
        rows = []
        for index, sample in enumerate(samples):
            _validate_scoring_metadata(sample)
            messages = build_math_chat_messages(sample.problem, index, use_raw_prompt=sample.use_raw_prompt)
            prompt_ids = normalize_token_ids(
                apply_chat_template(
                    tokenizer,
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=config.enable_thinking,
                )
            )
            if not prompt_ids or len(prompt_ids) > config.prompt_length:
                raise ValueError(
                    f"{benchmark}:{sample.sample_id} has {len(prompt_ids)} prompt tokens; "
                    f"expected 1..{config.prompt_length}. Increase MAX_PROMPT_LENGTH."
                )
            max_tokens = min(config.generation_max_tokens, config.max_model_len - len(prompt_ids))
            if max_tokens <= 0:
                raise ValueError(f"No generation space for {benchmark}:{sample.sample_id} within max_model_len.")
            rows.append(PreparedSample(sample, prompt_ids, max_tokens))
        prepared[benchmark] = rows
    return prepared


def sampling_kwargs(config: EvaluationConfig, max_tokens: int) -> dict[str, Any]:
    """Match one training validation request; repeat questions externally for rollout_n answers."""
    return dict(
        n=1,
        temperature=config.temperature if config.do_sample else 0.0,
        top_p=config.top_p if config.do_sample else 1.0,
        top_k=config.top_k if config.do_sample else -1,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        max_tokens=max_tokens,
        ignore_eos=False,
        logprobs=None,
        detokenize=False,
    )


def engine_kwargs(config: EvaluationConfig) -> dict[str, Any]:
    """Build a local multiprocess engine that loads real HF weights and preserves model EOS settings."""
    return dict(
        model=config.model_path,
        tokenizer=config.model_path,
        load_format="auto",
        dtype="bfloat16",
        trust_remote_code=False,
        distributed_executor_backend="mp",
        tensor_parallel_size=config.tensor_parallel_size,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_model_len=config.max_model_len,
        max_num_batched_tokens=config.max_num_batched_tokens,
        max_num_seqs=config.max_num_seqs,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        enforce_eager=config.enforce_eager,
        disable_log_stats=True,
        seed=config.seed,
        generation_config="auto",
        override_generation_config=dict(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=1.0,
            max_new_tokens=config.response_length,
        ),
    )


def score_response(tokenizer, token_ids: list[int], sample: EvalSample) -> dict[str, Any]:
    """Score the complete generated text with the same math_verify dispatcher as training."""
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    extra_info: dict[str, Any] = {"reward_data_source": "math_verify"}
    for name in ("question_id", "answer_type", "grader"):
        value = getattr(sample, name)
        if value is not None:
            extra_info[name] = value
    result = default_compute_score(
        data_source=sample.benchmark,
        solution_str=text,
        ground_truth=sample.ground_truth,
        extra_info=extra_info,
    )
    score = float(result["score"])
    if not math.isfinite(score):
        raise RuntimeError(f"Non-finite score for {sample.benchmark}:{sample.sample_id}")
    return {**result, "reward": score}


def aggregate_metrics(uids: list[str], benchmark: str, infos: list[dict[str, Any]]) -> dict[str, float]:
    """Reuse the training statistics and its core/aux metric naming convention."""
    _, columns = collate_reward_extra_infos(infos)
    nested = process_validation_metrics([benchmark] * len(uids), uids, columns)
    metrics = {}
    for source, variables in nested.items():
        core_var = "acc" if "acc" in variables else "reward"
        for variable, values in variables.items():
            n_max = max(int(name.split("@")[-1].split("/")[0]) for name in values)
            for name, value in values.items():
                core = variable == core_var and name.startswith(("mean", "maj", "best")) and f"@{n_max}" in name
                section = "core" if core else "aux"
                metrics[f"{METRIC_PREFIX}-{section}/{source}/{variable}/{name}"] = float(value)
    return metrics


def evaluate_benchmark(
    config: EvaluationConfig, tokenizer, engine, sampling_params_cls, rows: list[PreparedSample]
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    """Generate and score in memory; reject incomplete, reordered, or aborted engine responses."""
    benchmark = rows[0].sample.benchmark
    uids, infos, lengths = [], [], []
    length_finishes = 0
    for start in range(0, len(rows), config.batch_size):
        batch = rows[start : start + config.batch_size]
        requests = [row for row in batch for _ in range(config.rollout_n)]
        params = [sampling_params_cls(**sampling_kwargs(config, row.max_tokens)) for row in requests]
        outputs = engine.generate(
            [{"prompt_token_ids": list(row.prompt_ids)} for row in requests],
            params,
            use_tqdm=False,
        )
        if len(outputs) != len(requests):
            raise RuntimeError(f"{benchmark}: expected {len(requests)} request outputs, got {len(outputs)}.")
        for row, output in zip(requests, outputs, strict=True):
            if output.prompt_token_ids != row.prompt_ids or not output.finished or len(output.outputs) != 1:
                raise RuntimeError(f"Invalid or mismatched generation output for {row.uid}.")
            completion = output.outputs[0]
            if completion.finish_reason not in {"stop", "length"}:
                raise RuntimeError(f"Generation failed for {row.uid}: finish_reason={completion.finish_reason!r}.")
            token_ids = normalize_token_ids(completion.token_ids)
            if len(token_ids) > row.max_tokens:
                raise RuntimeError(f"Generation exceeded its token budget for {row.uid}.")
            infos.append(score_response(tokenizer, token_ids, row.sample))
            uids.append(row.uid)
            lengths.append(len(token_ids))
            length_finishes += completion.finish_reason == "length"
        del outputs
        print(
            f"[eval] {benchmark}: scored {min(start + len(batch), len(rows))}/{len(rows)} questions "
            f"({len(infos)} responses).",
            flush=True,
        )
    return (
        uids,
        infos,
        dict(
            questions=len(rows),
            responses=len(infos),
            response_tokens_mean=sum(lengths) / len(lengths),
            response_tokens_min=min(lengths),
            response_tokens_max=max(lengths),
            length_finish_rate=length_finishes / len(lengths),
            context_capped_questions=sum(row.max_tokens < config.generation_max_tokens for row in rows),
        ),
    )


def run_evaluation(config: EvaluationConfig) -> dict[str, Any]:
    """Evaluate the selected suite and print a final JSON result only after every benchmark completes."""
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "VLLM_NO_USAGE_STATS"):
        os.environ[name] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    # Import explicitly before allocating the engine: the legacy scorer otherwise
    # converts a missing Math-Verify dependency into zero scores.
    from math_verify.grader import verify  # noqa: F401
    from math_verify.parser import parse  # noqa: F401

    config = replace(config, model_path=prepare_checkpoint(config.model_path, force_merge=config.force_merge))
    tokenizer = load_tokenizer(config)
    benchmark_dirs = resolve_benchmark_dirs(config)
    prepared = prepare_samples(config, tokenizer, benchmark_dirs=benchmark_dirs)
    resolved = {
        **asdict(config),
        "benchmarks": list(config.benchmarks),
        "benchmark_dirs": {name: str(path) for name, path in benchmark_dirs.items()},
        "dtype": "bfloat16",
        "repetition_penalty": 1.0,
        "reward_data_source": "math_verify",
        "bootstrap_seed": 42,
        "effective_sampling": sampling_kwargs(config, config.generation_max_tokens),
    }
    print("[eval] config=" + json.dumps(resolved, sort_keys=True), flush=True)
    for name, rows in prepared.items():
        print(
            f"[eval] {name}: {len(rows)} questions, {len(rows) * config.rollout_n} responses, "
            f"prompt_tokens={min(len(row.prompt_ids) for row in rows)}.."
            f"{max(len(row.prompt_ids) for row in rows)}.",
            flush=True,
        )

    from vllm import LLM, SamplingParams

    engine = LLM(**engine_kwargs(config))
    scores = {}
    try:
        for name, rows in prepared.items():
            scores[name] = evaluate_benchmark(config, tokenizer, engine, SamplingParams, rows)
    finally:
        del engine

    result = {"profile": "legacy", "config": resolved, "benchmarks": {}, "metrics": {}}
    for name, (uids, infos, summary) in scores.items():
        metrics = aggregate_metrics(uids, name, infos)
        summary["accuracy"] = metrics[f"{METRIC_PREFIX}-core/{name}/acc/mean@{config.rollout_n}"]
        result["benchmarks"][name] = summary
        result["metrics"].update(metrics)
    for name, summary in result["benchmarks"].items():
        print(
            f"[result] {name}: questions={summary['questions']}, responses={summary['responses']}, "
            f"mean@{config.rollout_n}={summary['accuracy']:.6f} ({summary['accuracy']:.2%}), "
            f"response_tokens_mean={summary['response_tokens_mean']:.1f}, "
            f"length_finish_rate={summary['length_finish_rate']:.2%}",
            flush=True,
        )
        for key, value in sorted(result["metrics"].items()):
            if key.startswith(f"{METRIC_PREFIX}-core/{name}/acc/"):
                print(f"{key} = {value:.8f}", flush=True)
    print("EVAL_RESULT_JSON=" + json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return result


def main() -> int:
    """Read environment configuration and return a nonzero status on any evaluation failure."""
    try:
        node_rank = _env_value(os.environ, "NODE_RANK", "VC_TASK_INDEX", default="0")
        if not re.fullmatch(r"[0-9]+", node_rank):
            raise ValueError(f"NODE_RANK must be a non-negative integer, got {node_rank!r}.")
        if int(node_rank) != 0:
            print(f"[eval] Skipping standalone evaluation on NODE_RANK={node_rank}; only node 0 evaluates.", flush=True)
            return 0
        run_evaluation(EvaluationConfig.from_environment())
    except Exception as exc:
        print(f"[eval] ERROR: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
