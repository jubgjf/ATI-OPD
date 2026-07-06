# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Agent framework for multi-turn rollout and agentic reinforcement learning.
- AgentLoopBase: coroutine based abstract base class for agent loop.
  - SingleTurnAgentLoop: single turn agent loop.
  - ToolAgentLoop: ReAct agent loop with tool calling, with user defined tools.
- AgentLoopWorker: worker class for running agent loop coroutines in parallel.
- AgentLoopManager: manager class for running agent loop workers in parallel.

AgentLoopManager is one specific agent-framework implementation in verl,
and is designed to be fully replaceable by other agent frameworks such as:
- NVIDIA Nemo-Gym
- AWS Bedrock AgentCore
- SWE-agent
- ...
"""

import asyncio
import logging
import math
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional
from uuid import uuid4

import hydra
import numpy as np
import ray
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from pydantic import BaseModel, ConfigDict
from tensordict import TensorDict
from transformers import AutoProcessor, AutoTokenizer

from verl.experimental.agent_loop.utils import resolve_config_path
from verl.experimental.teacher_loop.teacher_manager import (
    DEFAULT_NONFINITE_LOGPROB_REPLACEMENT,
    TeacherGenerationResult,
    TeacherLogprobScoringResult,
)
from verl.protocol import DataProto
from verl.tools.tool_registry import load_all_tools
from verl.trainer.distillation import is_distillation_enabled
from verl.trainer.distillation.diagnostics import compute_rewrite_overlap_metrics
from verl.utils.chat_template import apply_chat_template, initialize_system_prompt
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.dataset.rl_dataset import RLHFDataset, get_dataset_class
from verl.utils.model import compute_position_id_with_mask
from verl.utils.profiler import simple_timer
from verl.utils.ray_utils import auto_await, get_event_loop
from verl.utils.rollout_trace import (
    RolloutTraceConfig,
    rollout_trace_attr,
)
from verl.utils.tokenizer import (
    build_multimodal_processor_inputs,
    get_processor_token_id,
    normalize_token_ids,
)
from verl.workers.config import (
    DistillationInterventionConfig,
    HFModelConfig,
    RolloutConfig,
)
from verl.workers.rollout.llm_server import LLMServerClient

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

DEFAULT_ROUTING_CACHE_SIZE = 10000
INTERVENTION_TOPK_DISTRIBUTION_GAP_METRICS = {"topk_kl", "topk_distribution_kl", "topk_dist_kl"}
INTERVENTION_FULL_DISTRIBUTION_GAP_METRICS = {"full_kl", "full_distribution_kl", "distribution_kl"}


@dataclass
class _StudentSuffixGenerationResult:
    token_ids: list[int]
    logprobs: list[float]
    stop_reason: Optional[str] = None
    finish_reason: Optional[str] = None
    terminal: bool = False
    eos_truncated: bool = False


@dataclass
class _ThinkingBlockState:
    valid: bool = True
    open: bool = False
    reason: Optional[str] = None


def _compute_intervention_gap(
    student_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    metric: str,
) -> torch.Tensor:
    """Compute per-token teacher/student gap for intervention gating."""
    metric = metric.lower()
    student_logprobs = student_logprobs.float()
    teacher_logprobs = teacher_logprobs.float()
    if metric in ("kl", "k1"):
        return teacher_logprobs - student_logprobs
    if metric == "abs":
        return (student_logprobs - teacher_logprobs).abs()
    if metric in ("mse", "k2"):
        return 0.5 * (student_logprobs - teacher_logprobs).square()
    if metric in ("low_var_kl", "k3"):
        kl = (teacher_logprobs - student_logprobs).clamp(min=-20, max=20)
        ratio = torch.exp(kl)
        return (ratio - kl - 1).clamp(min=-10, max=10)
    raise NotImplementedError(f"Unsupported teacher intervention gap metric: {metric}")


def _teacher_response_logprobs(
    teacher_logprobs: torch.Tensor,
    prompt_len: int,
    response_len: int,
) -> torch.Tensor:
    """Slice teacher prompt_logprobs so they align with response tokens."""
    if response_len == 0:
        return torch.empty(0, dtype=teacher_logprobs.dtype)
    if prompt_len <= 0:
        raise ValueError("Teacher intervention requires a non-empty prompt.")
    if teacher_logprobs.ndim != 2 or teacher_logprobs.shape[1] != 1:
        raise RuntimeError(
            "Teacher intervention scoring expects sampled-token prompt logprobs with shape "
            f"[sequence_len, 1], but got teacher_logprobs.shape={tuple(teacher_logprobs.shape)}. "
            "Check that the teacher backend returns sampled_prompt_logprobs when top-k scoring is requested."
        )
    required_len = prompt_len + response_len
    if teacher_logprobs.shape[0] < required_len:
        raise RuntimeError(
            "Teacher intervention scoring returned too few prompt logprobs: "
            f"teacher_logprobs.shape={tuple(teacher_logprobs.shape)}, prompt_len={prompt_len}, "
            f"response_len={response_len}, required_sequence_len={required_len}. "
            "If this is a local vLLM Metal run, vllm_metal currently does not populate prompt_logprobs."
        )
    sampled_logprobs = teacher_logprobs[:, 0]
    response_logprobs = sampled_logprobs[prompt_len - 1 : prompt_len + response_len - 1]
    if response_logprobs.numel() != response_len:
        raise RuntimeError(
            "Teacher intervention scoring failed to align teacher logprobs with response tokens: "
            f"got {response_logprobs.numel()} values for response_len={response_len}."
        )
    if not torch.isfinite(response_logprobs.float()).all():
        raise RuntimeError("Teacher intervention scoring produced NaN/Inf response logprobs.")
    return response_logprobs


def _uses_intervention_distribution_gap(metric: str) -> bool:
    metric = metric.lower()
    return metric in INTERVENTION_TOPK_DISTRIBUTION_GAP_METRICS | INTERVENTION_FULL_DISTRIBUTION_GAP_METRICS


def _uses_intervention_full_distribution_gap(metric: str) -> bool:
    return metric.lower() in INTERVENTION_FULL_DISTRIBUTION_GAP_METRICS


def _response_distribution_scoring_rows(
    *,
    result: TeacherLogprobScoringResult,
    prompt_len: int,
    response_len: int,
    context: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice prompt-logprob top-k/full rows so they align with response tokens."""
    if response_len == 0:
        return torch.empty((0, 0), dtype=torch.long), torch.empty((0, 0), dtype=torch.float32)
    if prompt_len <= 0:
        raise ValueError(f"{context} intervention scoring requires a non-empty prompt.")
    ids = result.ids
    logprobs = result.logprobs
    if ids.ndim != 2 or logprobs.ndim != 2 or ids.shape != logprobs.shape:
        raise RuntimeError(
            f"{context} intervention distribution scoring expects aligned rank-2 ids/logprobs, "
            f"got ids.shape={tuple(ids.shape)} and logprobs.shape={tuple(logprobs.shape)}."
        )
    required_len = prompt_len + response_len
    if ids.shape[0] < required_len:
        raise RuntimeError(
            f"{context} intervention scoring returned too few prompt-logprob rows: "
            f"ids.shape={tuple(ids.shape)}, prompt_len={prompt_len}, response_len={response_len}, "
            f"required_sequence_len={required_len}."
        )
    start = prompt_len - 1
    end = start + response_len
    return ids[start:end].long(), logprobs[start:end].float()


def _logprob_floor_from_distillation_config(distillation_config: Any) -> float:
    loss_config = getattr(distillation_config, "distillation_loss", None)
    floor = getattr(loss_config, "log_prob_min_clamp", None)
    if floor is None:
        floor = DEFAULT_NONFINITE_LOGPROB_REPLACEMENT
    floor = float(floor)
    if not math.isfinite(floor) or floor > 0:
        raise ValueError(f"distillation_loss.log_prob_min_clamp must be finite and <= 0, got {floor!r}.")
    return floor


def _logprobs_on_support(
    *,
    source_ids: torch.Tensor,
    source_logprobs: torch.Tensor,
    support_ids: torch.Tensor,
    log_prob_floor: float,
) -> torch.Tensor:
    source_logprobs = torch.nan_to_num(
        source_logprobs.float(),
        nan=log_prob_floor,
        neginf=log_prob_floor,
        posinf=0.0,
    ).clamp_min(log_prob_floor)
    matches = support_ids.unsqueeze(-1) == source_ids.long().unsqueeze(0)
    floor_values = torch.full(
        (support_ids.shape[0], source_ids.shape[0]),
        log_prob_floor,
        dtype=source_logprobs.dtype,
        device=source_logprobs.device,
    )
    matched_logprobs = torch.where(matches, source_logprobs.unsqueeze(0).expand_as(floor_values), floor_values)
    return matched_logprobs.max(dim=-1).values


def _compute_distribution_intervention_gap(
    *,
    student_result: TeacherLogprobScoringResult,
    teacher_result: TeacherLogprobScoringResult,
    prompt_len: int,
    response_len: int,
    metric: str,
    tokenizer: Any,
    log_prob_floor: float,
) -> torch.Tensor:
    """Compute per-token KL(student || teacher) on top-k union support or exact full support."""
    student_ids, student_logprobs = _response_distribution_scoring_rows(
        result=student_result,
        prompt_len=prompt_len,
        response_len=response_len,
        context="Student",
    )
    teacher_ids, teacher_logprobs = _response_distribution_scoring_rows(
        result=teacher_result,
        prompt_len=prompt_len,
        response_len=response_len,
        context="Teacher",
    )
    if student_ids.shape[0] != teacher_ids.shape[0]:
        raise RuntimeError(
            "Student and teacher intervention distribution scoring returned different response lengths: "
            f"student_ids.shape={tuple(student_ids.shape)}, teacher_ids.shape={tuple(teacher_ids.shape)}."
        )

    exact_full_kl = _uses_intervention_full_distribution_gap(metric)
    vocab_size = _tokenizer_vocab_size(tokenizer) if exact_full_kl else None
    if exact_full_kl and vocab_size is None:
        raise RuntimeError("gap_metric=full_kl requires a tokenizer with a known vocabulary size.")

    gaps = []
    for pos in range(response_len):
        support_ids = torch.unique(torch.cat([student_ids[pos], teacher_ids[pos]], dim=0))
        support_ids = support_ids[support_ids >= 0]
        if vocab_size is not None:
            support_ids = support_ids[support_ids < vocab_size]
            if support_ids.numel() < vocab_size:
                raise RuntimeError(
                    "gap_metric=full_kl requires exact full-vocabulary prompt logprobs from both student and "
                    "teacher, but the returned union support is incomplete at response position "
                    f"{pos}: support={support_ids.numel()}, vocab_size={vocab_size}. "
                    "Use gap_metric=topk_distribution_kl for the top-k approximation."
                )
        if support_ids.numel() == 0:
            gaps.append(torch.tensor(0.0, dtype=torch.float32))
            continue

        support_ids = support_ids.to(student_ids.device)
        student_lp = _logprobs_on_support(
            source_ids=student_ids[pos].to(student_ids.device),
            source_logprobs=student_logprobs[pos].to(student_ids.device),
            support_ids=support_ids,
            log_prob_floor=log_prob_floor,
        )
        teacher_lp = _logprobs_on_support(
            source_ids=teacher_ids[pos].to(student_ids.device),
            source_logprobs=teacher_logprobs[pos].to(student_ids.device),
            support_ids=support_ids,
            log_prob_floor=log_prob_floor,
        )
        student_log_dist = student_lp - torch.logsumexp(student_lp, dim=-1)
        teacher_log_dist = teacher_lp - torch.logsumexp(teacher_lp, dim=-1)
        kl = (student_log_dist.exp() * (student_log_dist - teacher_log_dist)).sum()
        gaps.append(torch.clamp(torch.nan_to_num(kl.float(), nan=0.0, posinf=1e4, neginf=0.0), min=0.0))
    if not gaps:
        return torch.empty(0, dtype=torch.float32)
    return torch.stack(gaps).cpu()


def _teacher_gap_logprobs(result: TeacherLogprobScoringResult) -> torch.Tensor:
    if result.sampled_logprobs is not None:
        return result.sampled_logprobs
    return result.logprobs


def _coerce_teacher_logprob_scoring_result(result: Any) -> TeacherLogprobScoringResult:
    if isinstance(result, TeacherLogprobScoringResult):
        return result
    if isinstance(result, tuple) and len(result) == 2:
        teacher_ids, teacher_logprobs = result
        return TeacherLogprobScoringResult(ids=teacher_ids, logprobs=teacher_logprobs)
    raise TypeError(
        "Teacher logprob scoring must return TeacherLogprobScoringResult or a legacy "
        f"(teacher_ids, teacher_logprobs) tuple, got {type(result).__name__}."
    )


def _coerce_prompt_logprob_scoring_output(
    output: Any,
    *,
    sequence_len: int,
    expected_width: int,
    context: str,
) -> TeacherLogprobScoringResult:
    extra_fields = getattr(output, "extra_fields", None) or {}
    missing_keys = {"prompt_ids", "prompt_logprobs"} - set(extra_fields)
    if missing_keys:
        raise RuntimeError(
            f"{context} logprob scoring did not return required prompt scoring fields {sorted(missing_keys)}. "
            f"Returned extra_fields keys: {sorted(extra_fields.keys())}."
        )
    try:
        ids = torch.tensor(extra_fields["prompt_ids"], dtype=torch.int32)
        logprobs = torch.tensor(extra_fields["prompt_logprobs"])
        sampled_ids = (
            torch.tensor(extra_fields["sampled_prompt_ids"], dtype=torch.int32)
            if "sampled_prompt_ids" in extra_fields
            else None
        )
        sampled_logprobs = (
            torch.tensor(extra_fields["sampled_prompt_logprobs"]) if "sampled_prompt_logprobs" in extra_fields else None
        )
    except Exception as exc:
        raise RuntimeError(
            f"{context} logprob scoring returned prompt scoring fields that could not be converted to tensors. "
            f"prompt_ids type={type(extra_fields['prompt_ids']).__name__}, "
            f"prompt_logprobs type={type(extra_fields['prompt_logprobs']).__name__}."
        ) from exc
    if ids.ndim != 2 or logprobs.ndim != 2 or ids.shape != logprobs.shape:
        raise RuntimeError(
            f"{context} logprob scoring expected aligned rank-2 prompt ids/logprobs, "
            f"got ids.shape={tuple(ids.shape)} and logprobs.shape={tuple(logprobs.shape)}."
        )
    if ids.shape != (sequence_len, expected_width):
        raise RuntimeError(
            f"{context} logprob scoring returned unexpected prompt scoring shape: "
            f"expected {(sequence_len, expected_width)}, got ids.shape={tuple(ids.shape)} and "
            f"logprobs.shape={tuple(logprobs.shape)}."
        )
    if (sampled_ids is None) != (sampled_logprobs is None):
        raise RuntimeError(
            f"{context} logprob scoring must return sampled_prompt_ids and sampled_prompt_logprobs together, "
            f"but got extra_fields keys: {sorted(extra_fields.keys())}."
        )
    if sampled_ids is not None and (
        sampled_ids.ndim != 2
        or sampled_logprobs.ndim != 2
        or sampled_ids.shape != sampled_logprobs.shape
        or sampled_ids.shape != (sequence_len, 1)
    ):
        raise RuntimeError(
            f"{context} sampled prompt scoring fields must have shape [sequence_len, 1], "
            f"got sampled_ids.shape={tuple(sampled_ids.shape)} and "
            f"sampled_logprobs.shape={tuple(sampled_logprobs.shape)}."
        )
    if logprobs.numel() and torch.nan_to_num(logprobs.float(), posinf=1.0).max().item() > 1e-6:
        raise RuntimeError(f"{context} prompt_logprobs contained positive log probabilities.")
    return TeacherLogprobScoringResult(
        ids=ids,
        logprobs=logprobs,
        sampled_ids=sampled_ids,
        sampled_logprobs=sampled_logprobs,
    )


def _coerce_teacher_generation_result(result: Any) -> TeacherGenerationResult:
    if isinstance(result, TeacherGenerationResult):
        return result
    if isinstance(result, list):
        return TeacherGenerationResult(token_ids=list(result))
    if hasattr(result, "token_ids"):
        extra_fields = getattr(result, "extra_fields", None) or {}
        return TeacherGenerationResult(
            token_ids=list(result.token_ids),
            stop_reason=getattr(result, "stop_reason", None),
            finish_reason=extra_fields.get("finish_reason"),
        )
    raise TypeError(
        "Teacher segment generation must return TeacherGenerationResult or a legacy token id list, "
        f"got {type(result).__name__}."
    )


def _normalize_finish_reason(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("type") or value.get("reason") or value
    return str(value).strip().lower()


def _is_abort_reason(value: Any) -> bool:
    return _normalize_finish_reason(value) in {"abort", "aborted", "cancelled", "canceled"}


def _is_length_reason(value: Any) -> bool:
    return _normalize_finish_reason(value) in {"length", "max_tokens", "max_new_tokens"}


def _apply_generation_max_tokens(sampling_params: dict[str, Any], meta_info: dict[str, Any]) -> None:
    """Forward an explicit generation token budget from DataProto metadata."""
    if "generation_max_tokens" not in meta_info:
        return
    max_tokens = meta_info["generation_max_tokens"]
    max_tokens = max_tokens.item() if hasattr(max_tokens, "item") else max_tokens
    max_tokens = int(max_tokens)
    if max_tokens <= 0:
        raise ValueError(f"generation_max_tokens must be positive, got {max_tokens}.")
    sampling_params["max_tokens"] = max_tokens


def _is_stop_reason(value: Any) -> bool:
    return _normalize_finish_reason(value) in {"stop", "eos", "end", "stop_sequence"}


def _generation_was_aborted(*, stop_reason: Any, finish_reason: Any) -> bool:
    return _is_abort_reason(stop_reason) or _is_abort_reason(finish_reason)


def _tokenizer_eos_token_ids(tokenizer: Any) -> set[int]:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, int):
        return {eos_token_id}
    if isinstance(eos_token_id, list | tuple | set):
        return {int(token_id) for token_id in eos_token_id if token_id is not None}
    return {int(eos_token_id)}


def _tokenizer_vocab_size(tokenizer: Any) -> Optional[int]:
    try:
        size = len(tokenizer)
    except TypeError:
        size = getattr(tokenizer, "vocab_size", None)
    except Exception:
        size = None
    return int(size) if size is not None and int(size) > 0 else None


def _invalid_token_ids_for_tokenizer(token_ids: list[int], tokenizer: Any) -> list[int]:
    vocab_size = _tokenizer_vocab_size(tokenizer)
    if vocab_size is None:
        return []
    return [int(token_id) for token_id in token_ids if int(token_id) < 0 or int(token_id) >= vocab_size]


def _truncate_generation_at_eos(
    token_ids: list[int],
    logprobs: Optional[list[float]],
    *,
    eos_token_ids: set[int],
) -> tuple[list[int], Optional[list[float]], bool]:
    if not eos_token_ids:
        return token_ids, logprobs, False
    for idx, token_id in enumerate(token_ids):
        if token_id in eos_token_ids:
            end = idx + 1
            return token_ids[:end], logprobs[:end] if logprobs is not None else None, True
    return token_ids, logprobs, False


def _encode_token_pattern(tokenizer: Any, text: str) -> list[int]:
    if not text or tokenizer is None or not hasattr(tokenizer, "encode"):
        return []
    try:
        return normalize_token_ids(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        logger.warning(
            "Failed to tokenize teacher intervention thinking tag %r; protection disabled for this tag.",
            text,
        )
        return []


def _match_token_pattern(token_ids: list[int], pattern: list[int], position: int) -> bool:
    return bool(pattern) and token_ids[position : position + len(pattern)] == pattern


def _thinking_block_state(
    token_ids: list[int],
    *,
    start_pattern: list[int],
    end_pattern: list[int],
) -> _ThinkingBlockState:
    if not start_pattern or not end_pattern:
        return _ThinkingBlockState()
    seen_start = False
    seen_closed = False
    open_block = False
    position = 0
    while position < len(token_ids):
        if _match_token_pattern(token_ids, start_pattern, position):
            if open_block:
                return _ThinkingBlockState(valid=False, open=True, reason="nested_start")
            if seen_closed:
                return _ThinkingBlockState(valid=False, open=False, reason="reopened_after_close")
            seen_start = True
            open_block = True
            position += len(start_pattern)
            continue
        if _match_token_pattern(token_ids, end_pattern, position):
            if not open_block:
                return _ThinkingBlockState(valid=False, open=False, reason="unmatched_close")
            open_block = False
            seen_closed = True
            position += len(end_pattern)
            continue
        position += 1
    return _ThinkingBlockState(valid=True, open=open_block if seen_start else False)


def _expand_segment_to_token_patterns(
    token_ids: list[int],
    start: int,
    end: int,
    *,
    patterns: list[list[int]],
) -> tuple[int, int]:
    adjusted_start = start
    adjusted_end = end
    for pattern in patterns:
        if not pattern or len(pattern) <= 1:
            continue
        for position in range(0, max(len(token_ids) - len(pattern) + 1, 0)):
            pattern_end = position + len(pattern)
            if token_ids[position:pattern_end] != pattern:
                continue
            if position < adjusted_start < pattern_end:
                adjusted_start = position
            if position < adjusted_end < pattern_end:
                adjusted_end = pattern_end
    return adjusted_start, min(adjusted_end, len(token_ids))


def _generation_is_terminal(
    *,
    token_ids: list[int],
    max_tokens: int,
    stop_reason: Any,
    finish_reason: Any,
    eos_truncated: bool,
) -> bool:
    if eos_truncated or _is_stop_reason(finish_reason):
        return True
    if _is_length_reason(finish_reason) or _is_abort_reason(finish_reason):
        return False
    if _is_stop_reason(stop_reason):
        return True
    if _is_abort_reason(stop_reason):
        return False
    # Older rollout adapters exposed only stop_reason="completed"; keep the
    # previous short-generation-is-terminal heuristic when raw finish_reason is absent.
    if finish_reason is None and _normalize_finish_reason(stop_reason) == "completed" and len(token_ids) < max_tokens:
        return True
    return False


def _record_skipped_intervention(
    skipped_interventions: list[dict[str, Any]],
    *,
    reason: str,
    attempt: int,
    start: int,
    end: int,
    candidate_response_ids: Optional[list[int]] = None,
    candidate_response_mask: Optional[list[int]] = None,
    teacher_tokens: int = 0,
    student_suffix_tokens: int = 0,
) -> None:
    skipped_interventions.append(
        {
            "reason": reason,
            "attempt": int(attempt),
            "start": int(start),
            "end": int(end),
            "candidate_tokens": 0 if candidate_response_ids is None else len(candidate_response_ids),
            "candidate_trainable_tokens": 0 if candidate_response_mask is None else int(sum(candidate_response_mask)),
            "teacher_tokens": int(teacher_tokens),
            "student_suffix_tokens": int(student_suffix_tokens),
        }
    )


_MOE_CONFIG_ATTRIBUTES = (
    "num_experts",
    "num_local_experts",
    "num_experts_per_tok",
    "n_routed_experts",
    "n_shared_experts",
    "moe_intermediate_size",
    "shared_expert_intermediate_size",
    "first_k_dense_replace",
)


def _get_config_value(config: Any, name: str) -> Any:
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)


def _is_positive_config_value(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, int | float):
        return value > 0
    return True


def _is_moe_like_hf_config(hf_config: Any) -> bool:
    configs = [hf_config]
    text_config = _get_config_value(hf_config, "text_config")
    if text_config is not None:
        configs.append(text_config)
    for config in configs:
        model_type = _get_config_value(config, "model_type")
        if model_type is not None and "moe" in str(model_type).lower():
            return True
        architectures = _get_config_value(config, "architectures") or []
        if isinstance(architectures, str):
            architectures = [architectures]
        if any("moe" in str(architecture).lower() for architecture in architectures):
            return True
        for attr_name in _MOE_CONFIG_ATTRIBUTES:
            if _is_positive_config_value(_get_config_value(config, attr_name)):
                return True
    return False


def _response_relative_nonfinite_positions(
    result: TeacherLogprobScoringResult,
    *,
    prompt_len: int,
    response_len: int,
) -> list[int]:
    if prompt_len <= 0:
        raise ValueError("Teacher logprob diagnostics require a non-empty prompt.")
    response_start = prompt_len - 1
    response_end = response_start + response_len
    positions = []
    for sequence_position in result.nonfinite_sequence_positions or []:
        if response_start <= sequence_position < response_end:
            positions.append(int(sequence_position - response_start))
    return positions


def _store_teacher_scoring_result(
    output: "AgentLoopOutput",
    result: TeacherLogprobScoringResult,
    *,
    prompt_len: int,
    response_len: int,
) -> None:
    nonfinite_positions = _response_relative_nonfinite_positions(
        result,
        prompt_len=prompt_len,
        response_len=response_len,
    )
    output.extra_fields["teacher_ids"] = result.ids
    output.extra_fields["teacher_logprobs"] = result.logprobs
    if result.sampled_ids is not None:
        output.extra_fields["teacher_sampled_ids"] = result.sampled_ids
    if result.sampled_logprobs is not None:
        output.extra_fields["teacher_sampled_logprobs"] = result.sampled_logprobs
    output.extra_fields["teacher_logprob_nonfinite_count"] = len(nonfinite_positions)
    output.extra_fields["teacher_logprob_nonfinite_positions"] = nonfinite_positions
    output.extra_fields["teacher_logprob_nonfinite_replacement"] = result.replacement_value


def _select_intervention_segment(
    gaps: torch.Tensor,
    response_mask: list[int],
    intervention_config: DistillationInterventionConfig,
) -> Optional[tuple[int, int, float]]:
    """Return the first fixed-token segment whose aggregate gap crosses the threshold."""
    segment_len = intervention_config.student_segment_length
    if segment_len <= 0:
        raise ValueError(f"student_segment_length must be positive, got {segment_len}.")
    if len(response_mask) != gaps.numel():
        raise RuntimeError(
            "Teacher intervention gap mask is not aligned with gap values: "
            f"len(response_mask)={len(response_mask)}, num_gaps={gaps.numel()}."
        )
    invalid_mask_values = sorted({mask for mask in response_mask if mask not in (0, 1)})
    if invalid_mask_values:
        raise ValueError(f"Teacher intervention response_mask contains non-binary values: {invalid_mask_values}.")
    if not torch.isfinite(gaps.float()).all():
        raise RuntimeError("Teacher intervention gap computation produced NaN/Inf values.")
    response_mask_tensor = torch.tensor(response_mask, dtype=torch.bool)
    for start in range(0, gaps.numel(), segment_len):
        end = min(start + segment_len, gaps.numel())
        segment_mask = response_mask_tensor[start:end]
        if not segment_mask.any():
            continue
        segment_gaps = gaps[start:end][segment_mask]
        if intervention_config.gap_aggregation == "max":
            score = segment_gaps.max().item()
        else:
            score = segment_gaps.mean().item()
        if score > intervention_config.gap_threshold:
            return start, end, score
    return None


def _validate_intervention_state(
    *,
    context: str,
    response_ids: list[int],
    response_mask: list[int],
    response_logprobs: Optional[list[float]],
    gap_mask: list[int],
    response_length_limit: int,
) -> None:
    if len(response_ids) > response_length_limit:
        raise RuntimeError(
            f"Teacher intervention state {context} exceeded rollout response length: "
            f"len(response_ids)={len(response_ids)}, response_length_limit={response_length_limit}."
        )
    if len(response_mask) != len(response_ids):
        raise RuntimeError(
            f"Teacher intervention state {context} has misaligned response_mask: "
            f"len(response_mask)={len(response_mask)}, len(response_ids)={len(response_ids)}."
        )
    if len(gap_mask) != len(response_ids):
        raise RuntimeError(
            f"Teacher intervention state {context} has misaligned gap_mask: "
            f"len(gap_mask)={len(gap_mask)}, len(response_ids)={len(response_ids)}."
        )
    invalid_response_mask = sorted({mask for mask in response_mask if mask not in (0, 1)})
    if invalid_response_mask:
        raise ValueError(
            f"Teacher intervention state {context} has non-binary response_mask values: {invalid_response_mask}."
        )
    invalid_gap_mask = sorted({mask for mask in gap_mask if mask not in (0, 1)})
    if invalid_gap_mask:
        raise ValueError(f"Teacher intervention state {context} has non-binary gap_mask values: {invalid_gap_mask}.")
    if response_logprobs is not None:
        if len(response_logprobs) != len(response_ids):
            raise RuntimeError(
                f"Teacher intervention state {context} has misaligned response_logprobs: "
                f"len(response_logprobs)={len(response_logprobs)}, len(response_ids)={len(response_ids)}."
            )
        if response_logprobs and not torch.isfinite(torch.tensor(response_logprobs, dtype=torch.float32)).all():
            raise RuntimeError(f"Teacher intervention state {context} has NaN/Inf response_logprobs.")


def _require_consistent_optional_field(inputs: list["_InternalAgentLoopOutput"], field_name: str) -> bool:
    present = [getattr(input_item, field_name) is not None for input_item in inputs]
    if any(present) and not all(present):
        bad_indices = [idx for idx, is_present in enumerate(present) if not is_present]
        raise RuntimeError(
            f"Inconsistent optional field {field_name!r} across agent loop outputs; missing at batch indices "
            f"{bad_indices}. This usually means only some samples produced that tensor. "
            "Keep the field present for every sample in the batch, or disable the feature that produces mixed outputs."
        )
    return all(present)


class AgentLoopMetrics(BaseModel):
    """Agent loop performance metrics."""

    generate_sequences: float = 0.0
    tool_calls: float = 0.0
    compute_score: float = 0.0
    num_preempted: int = -1  # -1 means not available


class AgentLoopOutput(BaseModel):
    """Agent loop output."""

    prompt_ids: list[int]
    """Prompt token ids."""
    response_ids: list[int]
    """Response token ids including LLM generated token, tool response token."""
    response_mask: list[int]
    """Response mask, 1 for LLM generated token, 0 for tool response token."""
    response_logprobs: Optional[list[float]] = None
    """Log probabilities for the response tokens."""
    routed_experts: Optional[Any] = None
    """Routed experts for the total tokens."""
    multi_modal_data: Optional[dict[str, Any]] = None
    """Multi-modal data for multi-modal tools."""
    reward_score: Optional[float] = None
    """Reward score for the trajectory."""
    num_turns: int = 0
    """Number of chat turns, including user, assistant, tool."""
    metrics: AgentLoopMetrics
    """Auxiliary performance metrics"""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""
    mm_processor_kwargs: Optional[dict[str, Any]] = None
    """Processor/backend kwargs that must stay aligned across rollout and training paths."""

    def as_dict(self) -> dict[str, Any]:
        """Convert agent loop output to a dictionary."""
        output = self.model_dump(exclude_unset=True)

        output["prompts"] = torch.tensor(output.pop("prompt_ids"), dtype=torch.int64)
        output["responses"] = torch.tensor(output.pop("response_ids"), dtype=torch.int64)
        output["response_mask"] = torch.tensor(output.pop("response_mask"), dtype=torch.int64)

        response_logprobs = output.pop("response_logprobs", None)
        if response_logprobs is not None:
            output["rollout_log_probs"] = torch.tensor(response_logprobs, dtype=torch.float32)

        routed_experts = output.pop("routed_experts", None)
        if routed_experts is not None:
            output["routed_experts"] = torch.tensor(routed_experts, dtype=torch.int64)

        # rm_scores: reward score for each token
        reward_score = output.pop("reward_score", None)
        if reward_score is not None:
            rm_scores = torch.zeros_like(output["response_mask"], dtype=torch.float32)
            rm_scores[-1] = reward_score
            output["rm_scores"] = rm_scores

        teacher_ids, teacher_logprobs = (
            output["extra_fields"].pop("teacher_ids", None),
            output["extra_fields"].pop("teacher_logprobs", None),
        )
        teacher_sampled_ids, teacher_sampled_logprobs = (
            output["extra_fields"].pop("teacher_sampled_ids", None),
            output["extra_fields"].pop("teacher_sampled_logprobs", None),
        )
        if teacher_ids is not None:
            output["teacher_ids"] = teacher_ids
        if teacher_logprobs is not None:
            output["teacher_logprobs"] = teacher_logprobs
        if teacher_sampled_ids is not None:
            output["teacher_sampled_ids"] = teacher_sampled_ids
        if teacher_sampled_logprobs is not None:
            output["teacher_sampled_logprobs"] = teacher_sampled_logprobs
        return output


class _InternalAgentLoopOutput(AgentLoopOutput):
    """Internal agent loop output with padded sequences."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_ids: torch.Tensor
    """Padded prompt token ids."""
    response_ids: torch.Tensor
    """Padded response token ids."""
    input_ids: torch.Tensor
    """Padded input ids(prompt_ids + response_ids)."""
    position_ids: torch.Tensor
    """Padded position ids."""
    response_mask: torch.Tensor
    """Padded response mask."""
    attention_mask: torch.Tensor
    """Padded attention mask."""
    response_logprobs: Optional[torch.Tensor] = None
    """Padded log probabilities for the response tokens."""
    teacher_logprobs: Optional[torch.Tensor] = None
    """Padded log probabilities from teacher model for prompt/response tokens."""
    teacher_ids: Optional[torch.Tensor] = None
    """Padded token ids corresponding to the teacher log probabilities."""
    teacher_sampled_logprobs: Optional[torch.Tensor] = None
    """Padded sampled-token teacher log probabilities for prompt/response tokens."""
    teacher_sampled_ids: Optional[torch.Tensor] = None
    """Padded sampled token ids corresponding to teacher_sampled_logprobs."""
    routed_experts: Optional[torch.Tensor] = None
    """Padded routed experts for the total tokens."""
    multi_modal_inputs: Optional[dict[str, torch.Tensor]] = None
    """Multi-modal inputs for processors (e.g. pixel_values, image_grid_thw, video_grid_thw)."""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


class DictConfigWrap:
    """Wrapper for DictConfig to avoid hydra.utils.instantiate recursive resolve."""

    def __init__(self, config: DictConfig):
        self.config = config


class ToolListWrap:
    """Wraps a tool list so ``hydra.utils.instantiate`` doesn't recursively
    resolve its elements (which would demote them to ``DictConfig``)."""

    def __init__(self, tools: list):
        self.tools = tools


class AgentLoopBase(ABC):
    """An agent loop takes an input message, chat with OpenAI compatible LLM server and interact with various
    environments.

    Args:
        trainer_config (DictConfig): whole config for main entrypoint.
        server_manager (LLMServerClient): OpenAI compatible LLM server manager.
        tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
        processor (AutoProcessor): Processor for process messages.
        dataset_cls (type[Dataset]): Dataset class for creating dataset, Defaults to RLHFDataset.
        data_config (DictConfigWrap): Dataset config.
    """

    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: LLMServerClient,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        dataset_cls: type[RLHFDataset],
        data_config: DictConfigWrap,
        **kwargs,
    ):
        self.config = trainer_config.config
        self.rollout_config = self.config.actor_rollout_ref.rollout
        self.server_manager = server_manager
        self.tokenizer = tokenizer
        self.processor = processor
        self.dataset_cls = dataset_cls
        self.data_config = data_config.config
        self.apply_chat_template_kwargs = self.data_config.get("apply_chat_template_kwargs", {})
        self.mm_processor_kwargs = self.data_config.get("mm_processor_kwargs", {})
        processing_class = self.processor if self.processor is not None else self.tokenizer
        self.system_prompt = initialize_system_prompt(processing_class, **self.apply_chat_template_kwargs)
        self.loop = get_event_loop()

    def _get_mm_processor_kwargs(self, audio_data: Optional[list[Any]] = None) -> dict[str, Any]:
        mm_processor_kwargs = dict(self.mm_processor_kwargs or {})
        if audio_data is not None and "sampling_rate" not in mm_processor_kwargs:
            sampling_rate = getattr(getattr(self.processor, "feature_extractor", None), "sampling_rate", None)
            if sampling_rate is not None:
                mm_processor_kwargs["sampling_rate"] = int(sampling_rate)
        return mm_processor_kwargs

    async def process_vision_info(self, messages: list[dict]) -> dict:
        """Backward-compatible wrapper for multi-modal extraction."""
        return await self.process_multi_modal_info(messages)

    async def process_multi_modal_info(self, messages: list[dict]) -> dict:
        """Extract images, videos and audios from messages.

        Args:
            messages (list[dict]): Input messages.

        Returns:
            dict: Multi-modal data with keys like "images", "videos" and "audios".
        """
        multi_modal_data = {}
        if self.processor is not None:
            image_patch_size = getattr(getattr(self.processor, "image_processor", None), "patch_size", 14)
            if hasattr(self.dataset_cls, "process_multi_modal_info"):
                images, videos, audios = await self.dataset_cls.process_multi_modal_info(
                    messages, image_patch_size=image_patch_size, config=self.data_config
                )
            else:
                images, videos = await self.dataset_cls.process_vision_info(
                    messages, image_patch_size=image_patch_size, config=self.data_config
                )
                audios = None
            if images is not None:
                multi_modal_data["images"] = images
            if videos is not None:
                multi_modal_data["videos"] = videos
            if audios is not None:
                multi_modal_data["audios"] = audios

        return multi_modal_data

    async def apply_chat_template(
        self,
        messages: list[dict],
        tools: list[dict] = None,
        images: list[Image.Image] = None,
        videos: list[tuple[torch.Tensor, dict]] = None,
        audios: list[Any] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        remove_system_prompt: bool = False,
    ):
        """Apply chat template to messages with optional tools, images, and videos.

        Args:
            messages (list[dict]): Input messages.
            tools (list[dict], optional): Tools schemas. Defaults to None.
            images (list[Image.Image], optional): Input images. Defaults to None.
            videos (list[tuple[torch.Tensor, dict]], optional): Input videos. Defaults to None.
            remove_system_prompt (bool, optional): Whether to remove system prompt. Defaults to False.

        Returns:
            list[int]: Prompt token ids.
        """
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: apply_chat_template(
                    self.processor,
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )

            model_inputs = build_multimodal_processor_inputs(
                self.processor,
                text=[raw_prompt],
                images=images,
                videos=videos,
                audio=audios,
                mm_processor_kwargs=mm_processor_kwargs
                if mm_processor_kwargs is not None
                else self._get_mm_processor_kwargs(audios),
            )
            prompt_ids = normalize_token_ids(model_inputs.pop("input_ids"))
        else:
            tokenized_prompt = await self.loop.run_in_executor(
                None,
                lambda: apply_chat_template(
                    self.tokenizer,
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
            prompt_ids = normalize_token_ids(tokenized_prompt)

        if remove_system_prompt:
            prompt_ids = prompt_ids[len(self.system_prompt) :]

        return prompt_ids

    @abstractmethod
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run agent loop to interact with LLM server and environment.

        Args:
            sampling_params (Dict[str, Any]): LLM sampling params.
            **kwargs: dataset fields from `verl.utils.dataset.RLHFDataset`.

        Returns:
            AgentLoopOutput: Agent loop output.
        """
        raise NotImplementedError


"""Agent loop registry: key is agent_name, value is a dict of agent loop config
used by hydra.utils.instantiate to initialize agent loop instance.

https://hydra.cc/docs/advanced/instantiate_objects/overview/
"""
_agent_loop_registry: dict[str, dict] = {}


def register(agent_name: str):
    """Register agent loop class."""

    def decorator(subclass: type[AgentLoopBase]) -> type[AgentLoopBase]:
        fqdn = f"{subclass.__module__}.{subclass.__qualname__}"
        _agent_loop_registry[agent_name] = {"_target_": fqdn}
        return subclass

    return decorator


class AgentLoopWorker:
    """Agent loop worker takes a batch of messages and run each message in an agent loop.

    Args:
        config (DictConfig): whole config for main entrypoint.
        llm_client (LLMServerClient): Client for the LLM server.
        teacher_client (dict[str, LLMServerClient]): Client for multiple teacher servers.
        reward_loop_worker_handles (List[ray.actor.ActorHandle]): Actor handles for streaming reward computation.
    """

    def __init__(
        self,
        config: DictConfig,
        llm_client: LLMServerClient,
        teacher_client: dict[str, LLMServerClient] = None,
        reward_loop_worker_handles: list[ray.actor.ActorHandle] = None,
    ):
        self.config = config
        self.llm_client = llm_client
        self.teacher_client = teacher_client
        self.reward_loop_worker_handles = reward_loop_worker_handles

        rollout_config, model_config = config.actor_rollout_ref.rollout, config.actor_rollout_ref.model
        self.rollout_config: RolloutConfig = omega_conf_to_dataclass(rollout_config)
        self.model_config: HFModelConfig = omega_conf_to_dataclass(model_config)

        self.dataset_cls = get_dataset_class(config.data)
        self.tokenizer = self.model_config.tokenizer
        self.processor = self.model_config.processor
        self.mm_processor_kwargs = config.data.get("mm_processor_kwargs", {})

        # Online policy distillation
        self.distillation_enabled = is_distillation_enabled(config.distillation)
        self.teacher_intervention_enabled = False
        self.teacher_intervention_config = None
        self._teacher_intervention_logged_samples = 0
        self._teacher_intervention_logged_step = None
        self._teacher_intervention_thinking_start_pattern: list[int] = []
        self._teacher_intervention_thinking_end_pattern: list[int] = []
        if self.distillation_enabled:
            from verl.experimental.teacher_loop.teacher_manager import AsyncTeacherLLMServerManager

            self.teacher_key: str = config.distillation.teacher_key
            self.teacher_server_manager = AsyncTeacherLLMServerManager(
                config=config,
                teacher_client=teacher_client,
            )
            self.distillation_config = self.teacher_server_manager.distillation_config
            self.teacher_intervention_config = self.distillation_config.intervention
            self.teacher_intervention_enabled = self.teacher_intervention_config.enabled
            self._validate_teacher_intervention_compatibility()
            self._init_teacher_intervention_token_patterns()

        # Load tools once per worker; each trajectory just reuses self.tools.
        tool_config_path = self.rollout_config.multi_turn.tool_config_path
        function_tool_path = self.rollout_config.multi_turn.function_tool_path
        self.tools = load_all_tools(
            tool_config_path=resolve_config_path(tool_config_path) if tool_config_path else None,
            function_tool_path=resolve_config_path(function_tool_path) if function_tool_path else None,
        )

        # Load custom agent loop implementations from config path
        agent_loop_config_path = self.rollout_config.agent.agent_loop_config_path
        if agent_loop_config_path:
            resolved_path = resolve_config_path(agent_loop_config_path)
            agent_loop_configs = OmegaConf.load(resolved_path)
            for agent_loop_config in agent_loop_configs:
                _agent_loop_registry[agent_loop_config.name] = agent_loop_config
        if self.model_config.get("custom_chat_template", None) is not None:
            if self.model_config.processor is not None:
                self.model_config.processor.chat_template = self.model_config.custom_chat_template
            self.model_config.tokenizer.chat_template = self.model_config.custom_chat_template

        trace_config = self.rollout_config.trace
        RolloutTraceConfig.init(
            self.rollout_config.trace.project_name,
            self.rollout_config.trace.experiment_name,
            trace_config.get("backend"),
            trace_config.get("token2text", False),
            trace_config.get("max_samples_per_step_per_worker", None),
        )

    def _validate_teacher_intervention_compatibility(self) -> None:
        if not self.teacher_intervention_enabled:
            return
        if getattr(self.rollout_config, "enable_rollout_routing_replay", False):
            raise NotImplementedError(
                "distillation.intervention.enabled is incompatible with "
                "actor_rollout_ref.rollout.enable_rollout_routing_replay. Repaired trajectories rewrite tokens after "
                "rollout, so captured routed experts would be stale."
            )
        if _is_moe_like_hf_config(getattr(self.model_config, "hf_config", None)):
            raise NotImplementedError(
                "distillation.intervention.enabled does not currently support MoE models. Repaired trajectories "
                "need fresh router decisions for inserted teacher and regenerated student tokens."
            )

    def _init_teacher_intervention_token_patterns(self) -> None:
        config = self.teacher_intervention_config
        if config is None or not getattr(config, "protect_thinking_blocks", True):
            self._teacher_intervention_thinking_start_pattern = []
            self._teacher_intervention_thinking_end_pattern = []
            return
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None:
            self._teacher_intervention_thinking_start_pattern = []
            self._teacher_intervention_thinking_end_pattern = []
            return
        self._teacher_intervention_thinking_start_pattern = _encode_token_pattern(
            tokenizer,
            getattr(config, "thinking_start_text", "<think>"),
        )
        self._teacher_intervention_thinking_end_pattern = _encode_token_pattern(
            tokenizer,
            getattr(config, "thinking_end_text", "</think>"),
        )

    def _ensure_teacher_intervention_token_patterns(self) -> None:
        if not hasattr(self, "_teacher_intervention_thinking_start_pattern"):
            AgentLoopWorker._init_teacher_intervention_token_patterns(self)

    def _teacher_intervention_thinking_state(
        self,
        response_ids: list[int],
        *,
        prompt_ids: Optional[list[int]] = None,
    ) -> _ThinkingBlockState:
        AgentLoopWorker._ensure_teacher_intervention_token_patterns(self)
        token_ids = list(response_ids)
        if prompt_ids:
            token_ids = list(prompt_ids) + token_ids
        return _thinking_block_state(
            token_ids,
            start_pattern=getattr(self, "_teacher_intervention_thinking_start_pattern", []),
            end_pattern=getattr(self, "_teacher_intervention_thinking_end_pattern", []),
        )

    def _expand_teacher_intervention_segment_for_token_patterns(self, response_ids: list[int], start: int, end: int):
        AgentLoopWorker._ensure_teacher_intervention_token_patterns(self)
        return _expand_segment_to_token_patterns(
            response_ids,
            start,
            end,
            patterns=[
                getattr(self, "_teacher_intervention_thinking_start_pattern", []),
                getattr(self, "_teacher_intervention_thinking_end_pattern", []),
            ],
        )

    def _teacher_prompt_logprobs(self) -> int:
        distillation_config = getattr(self, "distillation_config", None)
        loss_config = getattr(distillation_config, "distillation_loss", None)
        loss_settings = getattr(loss_config, "loss_settings", None)
        prompt_logprobs = 0
        if loss_config is not None and getattr(loss_settings, "use_topk", False):
            prompt_logprobs = max(prompt_logprobs, int(loss_config.topk))
        config = getattr(self, "config", None)
        algorithm_config = getattr(config, "algorithm", None)
        adv_estimator = getattr(algorithm_config, "adv_estimator", "")
        adv_estimator = getattr(adv_estimator, "value", adv_estimator)
        if loss_config is not None and str(adv_estimator) == "token_reward_direct":
            prompt_logprobs = max(prompt_logprobs, int(loss_config.topk or 0))
        diagnostics_config = getattr(distillation_config, "diagnostics", None)
        if diagnostics_config is not None and (
            getattr(diagnostics_config, "collect_topk_overlap", False)
            or getattr(diagnostics_config, "collect_rethinking_opd_metrics", False)
        ):
            prompt_logprobs = max(prompt_logprobs, int(getattr(diagnostics_config, "topk", 64) or 64))
        return prompt_logprobs

    def _intervention_prompt_logprobs(self) -> int:
        prompt_logprobs = AgentLoopWorker._teacher_prompt_logprobs(self)
        intervention_config = getattr(self, "teacher_intervention_config", None)
        if intervention_config is None or not getattr(intervention_config, "enabled", False):
            return prompt_logprobs

        metric = str(getattr(intervention_config, "gap_metric", "")).lower()
        if metric in INTERVENTION_TOPK_DISTRIBUTION_GAP_METRICS:
            prompt_logprobs = max(prompt_logprobs, int(getattr(intervention_config, "gap_distribution_topk", 16)))
        elif metric in INTERVENTION_FULL_DISTRIBUTION_GAP_METRICS:
            vocab_size = _tokenizer_vocab_size(getattr(self, "tokenizer", None))
            if vocab_size is None:
                raise RuntimeError("gap_metric=full_kl requires a tokenizer with a known vocabulary size.")
            prompt_logprobs = max(prompt_logprobs, vocab_size)
        return prompt_logprobs

    def _maybe_reset_teacher_intervention_log_counter(self, sample_kwargs: Optional[dict[str, Any]]) -> None:
        if not sample_kwargs or "global_steps" not in sample_kwargs:
            return
        global_steps = sample_kwargs["global_steps"]
        global_steps = global_steps.item() if hasattr(global_steps, "item") else global_steps
        if global_steps != getattr(self, "_teacher_intervention_logged_step", None):
            self._teacher_intervention_logged_step = global_steps
            self._teacher_intervention_logged_samples = 0

    def _get_mm_processor_kwargs(self, audio_data: Optional[list[Any]] = None) -> dict[str, Any]:
        """Return multimodal processor kwargs with audio sampling-rate defaults."""
        mm_processor_kwargs = dict(self.mm_processor_kwargs or {})
        if audio_data is not None and "sampling_rate" not in mm_processor_kwargs:
            sampling_rate = getattr(getattr(self.processor, "feature_extractor", None), "sampling_rate", None)
            if sampling_rate is not None:
                mm_processor_kwargs["sampling_rate"] = int(sampling_rate)
        return mm_processor_kwargs

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        """Generate sequences from agent loop.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        config = self.rollout_config
        validate = batch.meta_info.get("validate", False)
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs or (self.teacher_intervention_enabled and not validate),
        )

        def apply_greedy_sampling_params(params: dict[str, Any]) -> None:
            params["top_p"] = 1.0
            params["top_k"] = -1
            params["temperature"] = 0

        # override sampling params for validation
        if validate:
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["top_k"] = config.val_kwargs.top_k
            sampling_params["temperature"] = config.val_kwargs.temperature
        _apply_generation_max_tokens(sampling_params, batch.meta_info)

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch.non_tensor_batch:
            default_agent_loop = config.agent.default_agent_loop
            batch.non_tensor_batch["agent_name"] = np.array([default_agent_loop] * len(batch), dtype=object)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        max_samples_per_worker = RolloutTraceConfig.get_instance().max_samples_per_step_per_worker

        # For n rollouts per sample, we trace all n rollouts for selected samples
        # Note: This sampling happens per-worker, so total traces = max_samples_per_worker * num_workers * n
        if max_samples_per_worker is not None:
            unique_sample_indices = np.unique(index)
            if max_samples_per_worker < len(unique_sample_indices):
                selected_samples = set(
                    np.random.choice(unique_sample_indices, max_samples_per_worker, replace=False).tolist()
                )
                traced_indices = set(i for i in range(len(batch)) if index[i] in selected_samples)
            else:
                traced_indices = set(range(len(batch)))
        else:
            traced_indices = set(range(len(batch)))

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index.tolist(), batch.meta_info.get("validate", False)
        )

        # NOTE: __do_sample__ is an internal per-sample override used by REMAX combined rollout.
        # Do not forward it to concrete agent loops, which may reject unknown kwargs.
        per_sample_do_sample = batch.non_tensor_batch.get("__do_sample__")
        has_global_steps = "global_steps" in batch.meta_info
        global_steps = batch.meta_info.get("global_steps")
        tasks = []
        for i in range(len(batch)):
            trace_this_sample = i in traced_indices
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items() if k != "__do_sample__"}
            if has_global_steps and "global_steps" not in kwargs:
                kwargs["global_steps"] = global_steps
            sample_sampling_params = dict(sampling_params)
            if not validate and per_sample_do_sample is not None and not bool(per_sample_do_sample[i]):
                apply_greedy_sampling_params(sample_sampling_params)
            tasks.append(
                asyncio.create_task(
                    self._run_agent_loop(sample_sampling_params, trajectory_info[i], trace=trace_this_sample, **kwargs)
                )
            )
        outputs = await asyncio.gather(*tasks)

        output = self._postprocess(
            outputs, input_non_tensor_batch=batch.non_tensor_batch, validate=batch.meta_info.get("validate", False)
        )
        return output

    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        trace: bool = True,
        **kwargs,
    ) -> _InternalAgentLoopOutput:
        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_loop",
            trace=trace,
        ):
            assert agent_name in _agent_loop_registry, (
                f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
            )

            agent_loop_config = _agent_loop_registry[agent_name]
            agent_loop = hydra.utils.instantiate(
                config=agent_loop_config,
                trainer_config=DictConfigWrap(config=self.config),
                server_manager=self.llm_client,
                tokenizer=self.tokenizer,
                processor=self.processor,
                dataset_cls=self.dataset_cls,
                data_config=DictConfigWrap(self.config.data),
                tools=ToolListWrap(self.tools),
            )
            output: AgentLoopOutput = await agent_loop.run(sampling_params, **kwargs)
            return await self._agent_loop_postprocess(
                output,
                trajectory["validate"],
                sampling_params=sampling_params,
                **kwargs,
            )

    async def _agent_loop_postprocess(
        self,
        output,
        validate,
        sampling_params: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> _InternalAgentLoopOutput:
        """Perform post-processing operations on the output of each individual agent loop."""
        output.extra_fields["raw_prompt"] = kwargs["raw_prompt"]
        if getattr(self, "teacher_intervention_enabled", False):
            await self._maybe_repair_with_teacher(
                output,
                validate=validate,
                sampling_params=sampling_params,
                sample_kwargs=kwargs,
            )
            if (
                not validate
                and self.teacher_intervention_config is not None
                and self.teacher_intervention_config.train_on_teacher_segments
            ):
                # The intervention path needs rollout logprobs to score gaps, but
                # after teacher tokens are inserted there is no valid rollout
                # policy logprob for every trainable token. Drop this optional
                # field for every sample so the trainer recomputes old_log_probs
                # from the actor over the final sequence instead of receiving a
                # mixed per-sample schema.
                output.response_logprobs = None

        # Some AgentLoop may have already computed the reward score, e.g SWE-agent.

        # NOTE: consistent with the legacy batch version of generate_sequences that existed in the
        # deprecated vLLM SPMD rollout implementation.
        # prompt_ids: left padded with zeros (e.g., [0,0,0,0,1,2,3,4])
        # response_ids: right padded with zeros (e.g., [5,6,7,8,0,0,0,0])
        # input_ids: concatenation of prompt + response
        # Mask:
        # For example, if the prompt is [1,2,3,4] and the response is [5,6,7,(tool start)8,9(tool end),10,11,12]
        # - prompt_attention_mask: 0s for padding, 1s for tokens
        #   e.g., [0,0,0,0,1,1,1,1]
        # - response_attention_mask: 0s for padding, 1s for tokens
        #   e.g., [1,1,1,1,1,1,1,1,1,1,1,0,0,0,0]
        # attention_mask: concatenation of prompt_attention_mask and response_attention_mask
        #   e.g., [0,0,0,0,1,1,1,1(prompt),1,1,1,1,1,1,1,1,1,1,1,0,0,0,0(response)]
        # - response_mask: 1s for LLM generated tokens, 0 for tool response/padding tokens
        #   e.g., [1,1,1,1,1,1,1,(tool start),0,0(tool end),1,1,0,0,0,0]
        # - position_ids: sequential positions for tokens, starting at 0
        #   e.g., [0,0,0,0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,0,0,0,0]

        # TODO(wuxibin): remove padding and use tensordict.
        self.tokenizer.padding_side = "left"
        prompt_output = self.tokenizer.pad(
            {"input_ids": output.prompt_ids},
            padding="max_length",
            max_length=self.rollout_config.prompt_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if prompt_output["input_ids"].dim() == 1:
            prompt_output["input_ids"] = prompt_output["input_ids"].unsqueeze(0)
            prompt_output["attention_mask"] = prompt_output["attention_mask"].unsqueeze(0)

        self.tokenizer.padding_side = "right"
        response_output = self.tokenizer.pad(
            {"input_ids": output.response_ids},
            padding="max_length",
            max_length=self.rollout_config.response_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if response_output["input_ids"].dim() == 1:
            response_output["input_ids"] = response_output["input_ids"].unsqueeze(0)
            response_output["attention_mask"] = response_output["attention_mask"].unsqueeze(0)

        response_mask_output = self.tokenizer.pad(
            {"input_ids": output.response_mask},
            padding="max_length",
            max_length=self.rollout_config.response_length,
            return_tensors="pt",
            return_attention_mask=False,
        )
        if response_mask_output["input_ids"].dim() == 1:
            response_mask_output["input_ids"] = response_mask_output["input_ids"].unsqueeze(0)

        response_logprobs = None
        if output.response_logprobs is not None:
            pad_size = self.rollout_config.response_length - len(output.response_logprobs)
            response_logprobs = torch.tensor(output.response_logprobs + [0.0] * pad_size).unsqueeze(0)

        response_mask = response_mask_output["input_ids"] * response_output["attention_mask"]
        attention_mask = torch.cat([prompt_output["attention_mask"], response_output["attention_mask"]], dim=1)
        input_ids = torch.cat([prompt_output["input_ids"], response_output["input_ids"]], dim=1)

        routed_experts = None
        if output.routed_experts is not None:
            total_length = input_ids.shape[1]
            length, layer_num, topk_num = output.routed_experts.shape
            if isinstance(output.routed_experts, np.ndarray):
                routed_experts_array = output.routed_experts
                if not routed_experts_array.flags.writeable:
                    routed_experts_array = routed_experts_array.copy()
                experts_tensor = torch.from_numpy(routed_experts_array)
            elif isinstance(output.routed_experts, torch.Tensor):
                experts_tensor = output.routed_experts
            else:
                raise TypeError(f"Unsupported type for routed_experts: {type(output.routed_experts)}")
            routed_experts = torch.zeros(1, total_length, layer_num, topk_num, dtype=experts_tensor.dtype)

            # Calculate start position: left padding means original prompt starts at the end
            start_pos = prompt_output["input_ids"].shape[1] - len(output.prompt_ids)
            end_pos = min(start_pos + length, total_length)

            # Add boundary checks for robustness
            if start_pos < 0 or end_pos > total_length:
                raise ValueError(
                    f"Invalid position range: start_pos={start_pos}, end_pos={end_pos}, total_length={total_length}"
                )

            routed_experts[:, start_pos:end_pos] = experts_tensor.unsqueeze(0)

        multi_modal_inputs = self._compute_multi_modal_inputs(output, input_ids)
        position_ids = self._compute_position_ids(
            input_ids,
            attention_mask,
            multi_modal_inputs,
            output.mm_processor_kwargs
            if output.mm_processor_kwargs is not None
            else self._get_mm_processor_kwargs(
                output.multi_modal_data.get("audios") if output.multi_modal_data else None
            ),
        )
        await self._compute_score([output], kwargs=kwargs)
        await self._compute_teacher_logprobs(
            output,
            prompt_ids=output.prompt_ids,
            response_ids=output.response_ids,
            validate=validate,
            sample_kwargs=kwargs,
        )
        teacher_ids, teacher_logprobs = (
            output.extra_fields.pop("teacher_ids", None),
            output.extra_fields.pop("teacher_logprobs", None),
        )
        teacher_sampled_ids, teacher_sampled_logprobs = (
            output.extra_fields.pop("teacher_sampled_ids", None),
            output.extra_fields.pop("teacher_sampled_logprobs", None),
        )
        if teacher_ids is not None and teacher_logprobs is not None:
            # TODO(wuxibin): remove padding and use tensordict.
            from verl.experimental.teacher_loop.teacher_manager import _pad_teacher_outputs

            teacher_ids, teacher_logprobs = _pad_teacher_outputs(
                teacher_ids,
                teacher_logprobs,
                prompt_width=prompt_output["input_ids"].shape[1],
                response_width=response_output["input_ids"].shape[1],
                prompt_length=len(output.prompt_ids),
                response_length=len(output.response_ids),
                pad_token_id=self.tokenizer.pad_token_id,
            )
        if teacher_sampled_ids is not None and teacher_sampled_logprobs is not None:
            from verl.experimental.teacher_loop.teacher_manager import _pad_teacher_outputs

            teacher_sampled_ids, teacher_sampled_logprobs = _pad_teacher_outputs(
                teacher_sampled_ids,
                teacher_sampled_logprobs,
                prompt_width=prompt_output["input_ids"].shape[1],
                response_width=response_output["input_ids"].shape[1],
                prompt_length=len(output.prompt_ids),
                response_length=len(output.response_ids),
                pad_token_id=self.tokenizer.pad_token_id,
            )

        return _InternalAgentLoopOutput(
            prompt_ids=prompt_output["input_ids"],
            response_ids=response_output["input_ids"],
            input_ids=input_ids,
            position_ids=position_ids,
            response_mask=response_mask,
            attention_mask=attention_mask,
            response_logprobs=response_logprobs,
            routed_experts=routed_experts,
            multi_modal_inputs=multi_modal_inputs,
            multi_modal_data=output.multi_modal_data,
            mm_processor_kwargs=output.mm_processor_kwargs,
            teacher_logprobs=teacher_logprobs,
            teacher_ids=teacher_ids,
            teacher_sampled_logprobs=teacher_sampled_logprobs,
            teacher_sampled_ids=teacher_sampled_ids,
            reward_score=output.reward_score,
            num_turns=output.num_turns,
            metrics=output.metrics,
            extra_fields=output.extra_fields,
        )

    def _compute_multi_modal_inputs(self, output, input_ids) -> dict[str, torch.Tensor]:
        """Compute multi-modal inputs with image, video and audio."""
        multi_modal_inputs = {}
        if self.processor is None:
            return multi_modal_inputs

        multi_modal_data = output.multi_modal_data or {}
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        current_text = self.tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)

        multi_modal_inputs = build_multimodal_processor_inputs(
            self.processor,
            text=[current_text],
            images=images,
            videos=videos,
            audio=audios,
            mm_processor_kwargs=output.mm_processor_kwargs
            if output.mm_processor_kwargs is not None
            else self._get_mm_processor_kwargs(audios),
        )
        multi_modal_inputs.pop("input_ids", None)
        multi_modal_inputs.pop("attention_mask", None)

        # We must use dict(multi_modal_inputs) to convert BatchFeature values to a new dict
        # because np.array() only keeps the keys for BatchFeature.
        multi_modal_inputs = dict(multi_modal_inputs.convert_to_tensors("pt"))
        image_grid_thw = multi_modal_inputs.get("image_grid_thw")
        if image_grid_thw is not None:
            images_seqlens = torch.repeat_interleave(image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0])
            multi_modal_inputs["images_seqlens"] = images_seqlens
        return multi_modal_inputs

    def _compute_position_ids(
        self,
        input_ids,
        attention_mask,
        multi_modal_inputs,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Compute position ids for multi-modal inputs."""
        if self.processor is None:
            return compute_position_id_with_mask(attention_mask)  # (1, seq_len)

        multi_modal_kwargs = {
            "image_grid_thw": multi_modal_inputs.get("image_grid_thw"),
            "video_grid_thw": multi_modal_inputs.get("video_grid_thw"),
        }
        # For transformers>=5.3.0, mm_token_type_ids is only used to calculate position ids.
        if multi_modal_inputs.pop("mm_token_type_ids", None) is not None:
            mm_token_type_ids = torch.zeros_like(input_ids)
            image_token_id = get_processor_token_id(self.processor, "image")
            video_token_id = get_processor_token_id(self.processor, "video")
            if image_token_id is not None:
                mm_token_type_ids[0][input_ids[0] == image_token_id] = 1
            if video_token_id is not None:
                mm_token_type_ids[0][input_ids[0] == video_token_id] = 2
            multi_modal_kwargs["mm_token_type_ids"] = mm_token_type_ids

        # Model's get_rope_index has been dynamically bind to the processor.
        vision_position_ids, _ = self.processor.get_rope_index(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **multi_modal_kwargs,
        )
        vision_position_ids = vision_position_ids.transpose(0, 1)  # (3, 1, seq_len) => (1, 3, seq_len)

        valid_mask = attention_mask[0].bool()
        text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
        text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
        text_position_ids = text_position_ids.unsqueeze(0)
        position_ids = torch.cat((text_position_ids, vision_position_ids), dim=1)  # (1, 4, seq_length)
        return position_ids

    async def _compute_score(self, outputs: list[AgentLoopOutput], kwargs: dict) -> None:
        """Compute reward score for all outputs in a trajectory; assigns result to outputs[-1]."""
        enable_async_reward = self.reward_loop_worker_handles is not None

        final_output = outputs[-1]
        if final_output.reward_score is None and enable_async_reward:
            timing = {}
            with simple_timer("compute_score", timing):
                all_prompts, all_responses, all_input_ids, all_attention_mask, all_position_ids = [], [], [], [], []
                for output in outputs:
                    prompts = torch.tensor(output.prompt_ids, dtype=torch.int64)
                    responses = torch.tensor(output.response_ids, dtype=torch.int64)
                    input_ids = torch.cat([prompts, responses], dim=0)
                    attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
                    multi_modal_inputs = self._compute_multi_modal_inputs(output, input_ids)
                    position_ids = self._compute_position_ids(
                        input_ids.unsqueeze(0),
                        attention_mask.unsqueeze(0),
                        multi_modal_inputs,
                        output.mm_processor_kwargs
                        if output.mm_processor_kwargs is not None
                        else self._get_mm_processor_kwargs(
                            output.multi_modal_data.get("audios") if output.multi_modal_data else None
                        ),
                    ).squeeze(0)
                    all_prompts.append(prompts)
                    all_responses.append(responses)
                    all_input_ids.append(input_ids)
                    all_attention_mask.append(attention_mask)
                    all_position_ids.append(position_ids)

                n = len(outputs)
                batch = TensorDict(
                    {
                        "prompts": torch.nn.utils.rnn.pad_sequence(all_prompts, batch_first=True, padding_value=0),
                        "responses": torch.nn.utils.rnn.pad_sequence(all_responses, batch_first=True, padding_value=0),
                        "attention_mask": torch.nn.utils.rnn.pad_sequence(
                            all_attention_mask, batch_first=True, padding_value=0
                        ),
                        "input_ids": torch.nn.utils.rnn.pad_sequence(all_input_ids, batch_first=True, padding_value=0),
                        "position_ids": torch.nn.utils.rnn.pad_sequence(
                            all_position_ids, batch_first=True, padding_value=0
                        ),
                    },
                    batch_size=n,
                )
                non_tensor_batch = {
                    **{k: np.array([v] * n) for k, v in kwargs.items()},
                    "__num_turns__": np.array([o.num_turns for o in outputs]),
                    "tool_extra_fields": np.array([o.extra_fields for o in outputs], dtype=object),
                    "prompt_len": np.array([len(o.prompt_ids) for o in outputs]),
                    "response_len": np.array([len(o.response_ids) for o in outputs]),
                }

                data = DataProto(
                    batch=batch,
                    non_tensor_batch=non_tensor_batch,
                )
                selected_reward_loop_worker_handle = random.choice(self.reward_loop_worker_handles)
                result = await selected_reward_loop_worker_handle.compute_score.remote(data)
                final_output.reward_score = result["reward_score"]
                final_output.extra_fields["reward_extra_info"] = result["reward_extra_info"]
            final_output.metrics.compute_score = timing["compute_score"]

    def _teacher_routing_key(self, sample_kwargs: Optional[dict[str, Any]]) -> Optional[str]:
        routing_key = None
        if sample_kwargs is not None:
            routing_value = sample_kwargs.get(self.teacher_key)
            if routing_value is not None:
                # Non-tensor batch values arrive as 0-d numpy objects / arrays; normalize to Python.
                routing_key = routing_value.item() if hasattr(routing_value, "item") else routing_value
        return routing_key

    async def _score_sequence_for_intervention(
        self,
        *,
        request_id: str,
        prompt_ids: list[int],
        response_ids: list[int],
        output: AgentLoopOutput,
        routing_key: Optional[str],
    ) -> TeacherLogprobScoringResult:
        result = await self.teacher_server_manager.compute_teacher_logprobs_single(
            sequence_ids=prompt_ids + response_ids,
            multi_modal_data=output.multi_modal_data,
            mm_processor_kwargs=output.mm_processor_kwargs,
            routing_key=routing_key,
            request_id=request_id,
            prompt_logprobs=AgentLoopWorker._intervention_prompt_logprobs(self),
        )
        return _coerce_teacher_logprob_scoring_result(result)

    async def _score_student_sequence_for_intervention(
        self,
        *,
        request_id: str,
        prompt_ids: list[int],
        response_ids: list[int],
        output: AgentLoopOutput,
    ) -> TeacherLogprobScoringResult:
        sequence_ids = prompt_ids + response_ids
        prompt_logprobs = AgentLoopWorker._intervention_prompt_logprobs(self)
        if prompt_logprobs < 0:
            raise ValueError(f"Student intervention scoring requires prompt_logprobs >= 0, got {prompt_logprobs}.")
        multi_modal_data = output.multi_modal_data or {}
        student_output = await self.llm_client.generate(
            request_id=request_id,
            prompt_ids=sequence_ids,
            sampling_params={
                "max_tokens": 1,
                "temperature": 1.0,
                "prompt_logprobs": prompt_logprobs,
                "logprobs": None,
            },
            image_data=multi_modal_data.get("images"),
            video_data=multi_modal_data.get("videos"),
            audio_data=multi_modal_data.get("audios"),
            mm_processor_kwargs=output.mm_processor_kwargs,
        )
        return _coerce_prompt_logprob_scoring_output(
            student_output,
            sequence_len=len(sequence_ids),
            expected_width=max(prompt_logprobs, 1),
            context="Student intervention",
        )

    async def _generate_student_suffix(
        self,
        *,
        request_id: str,
        prefix_ids: list[int],
        max_tokens: int,
        sampling_params: Optional[dict[str, Any]],
        output: AgentLoopOutput,
    ) -> _StudentSuffixGenerationResult:
        if max_tokens <= 0:
            return _StudentSuffixGenerationResult(token_ids=[], logprobs=[])
        repair_sampling_params = dict(sampling_params or {})
        repair_sampling_params["max_tokens"] = max_tokens
        repair_sampling_params["logprobs"] = True
        multi_modal_data = output.multi_modal_data or {}
        student_output = await self.llm_client.generate(
            request_id=request_id,
            prompt_ids=prefix_ids,
            sampling_params=repair_sampling_params,
            image_data=multi_modal_data.get("images"),
            video_data=multi_modal_data.get("videos"),
            audio_data=multi_modal_data.get("audios"),
            mm_processor_kwargs=output.mm_processor_kwargs,
        )
        suffix_ids = list(student_output.token_ids[:max_tokens])
        suffix_logprobs = list((student_output.log_probs or [])[: len(suffix_ids)])
        if len(suffix_logprobs) != len(suffix_ids):
            raise RuntimeError(
                "Teacher intervention requires student suffix token logprobs for re-scoring, but the "
                f"student server returned len(token_ids)={len(suffix_ids)} and "
                f"len(log_probs)={len(suffix_logprobs)} for request_id={request_id!r}. "
                "Check that rollout sampling_params['logprobs'] is enabled and the rollout backend returns "
                "one logprob per generated token."
            )
        if suffix_logprobs and not torch.isfinite(torch.tensor(suffix_logprobs, dtype=torch.float32)).all():
            raise RuntimeError(f"Student suffix generation returned NaN/Inf logprobs for request_id={request_id!r}.")
        finish_reason = (student_output.extra_fields or {}).get("finish_reason")
        suffix_ids, suffix_logprobs, eos_truncated = _truncate_generation_at_eos(
            suffix_ids,
            suffix_logprobs,
            eos_token_ids=_tokenizer_eos_token_ids(self.tokenizer),
        )
        return _StudentSuffixGenerationResult(
            token_ids=suffix_ids,
            logprobs=suffix_logprobs,
            stop_reason=student_output.stop_reason,
            finish_reason=finish_reason,
            terminal=_generation_is_terminal(
                token_ids=suffix_ids,
                max_tokens=max_tokens,
                stop_reason=student_output.stop_reason,
                finish_reason=finish_reason,
                eos_truncated=eos_truncated,
            )
            or _generation_was_aborted(stop_reason=student_output.stop_reason, finish_reason=finish_reason),
            eos_truncated=eos_truncated,
        )

    async def _maybe_repair_with_teacher(
        self,
        output: AgentLoopOutput,
        *,
        validate: bool,
        sampling_params: Optional[dict[str, Any]],
        sample_kwargs: Optional[dict[str, Any]],
    ) -> None:
        """Repair high-gap response segments with teacher generation before padding/training."""
        if not self.teacher_intervention_enabled or validate:
            return

        AgentLoopWorker._maybe_reset_teacher_intervention_log_counter(self, sample_kwargs)
        AgentLoopWorker._ensure_teacher_intervention_token_patterns(self)
        output.extra_fields["teacher_intervention_count"] = 0
        output.extra_fields["teacher_interventions"] = []
        output.extra_fields["teacher_intervention_skipped"] = []
        interventions: list[dict[str, Any]] = output.extra_fields["teacher_interventions"]
        skipped_interventions: list[dict[str, Any]] = output.extra_fields["teacher_intervention_skipped"]

        if output.routed_experts is not None:
            raise NotImplementedError(
                "distillation.intervention.enabled cannot repair outputs that already contain routed_experts. "
                "Repaired tokens would need fresh router decisions; disable intervention or routing replay."
            )

        intervention_config: DistillationInterventionConfig = self.teacher_intervention_config
        min_trainable_tokens = int(getattr(intervention_config, "min_trainable_tokens_after_repair", 1))
        response_ids = list(output.response_ids[: self.rollout_config.response_length])
        if len(output.response_mask) < len(response_ids):
            raise RuntimeError(
                "Teacher intervention requires response_mask to cover every response token, but got "
                f"len(response_mask)={len(output.response_mask)} and len(response_ids)={len(response_ids)}."
            )
        response_mask = list(output.response_mask[: len(response_ids)])
        if not response_ids:
            return
        if any(mask != 1 for mask in response_mask):
            first_zero = next(idx for idx, mask in enumerate(response_mask) if mask != 1)
            raise NotImplementedError(
                "Teacher intervention currently supports only pure single-turn student-generated responses. "
                f"Found response_mask[{first_zero}]={response_mask[first_zero]!r}; this usually indicates a "
                "tool/multi-turn response or pre-existing non-trainable span. Disable "
                "distillation.intervention.enabled for this agent loop, or add explicit support for segmented "
                "multi-turn repair."
            )
        if output.response_logprobs is None:
            raise RuntimeError(
                "Teacher intervention is enabled but the student rollout did not return response_logprobs. "
                "AgentLoopWorker.generate_sequences should set sampling_params['logprobs']=True when "
                "intervention is enabled; check custom agent loops or rollout backends that may drop log_probs."
            )

        response_logprobs = list(output.response_logprobs[: len(response_ids)])
        gap_mask = list(response_mask)
        if len(response_logprobs) < len(response_ids):
            raise RuntimeError(
                "Teacher intervention requires one student logprob per response token, but got "
                f"len(response_logprobs)={len(response_logprobs)} and len(response_ids)={len(response_ids)}."
            )
        _validate_intervention_state(
            context="before scoring",
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            gap_mask=gap_mask,
            response_length_limit=self.rollout_config.response_length,
        )

        prompt_ids = list(output.prompt_ids)
        if not prompt_ids:
            raise RuntimeError("Teacher intervention requires a non-empty prompt_ids list.")
        routing_key = self._teacher_routing_key(sample_kwargs)
        base_request_id = output.extra_fields.get("rollout_request_id") or uuid4().hex
        cached_teacher_result = None
        eos_token_ids = _tokenizer_eos_token_ids(self.tokenizer)

        for attempt in range(intervention_config.max_interventions_per_sample + 1):
            teacher_result = await self._score_sequence_for_intervention(
                request_id=f"{base_request_id}:teacher-score:{attempt}",
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                output=output,
                routing_key=routing_key,
            )
            cached_teacher_result = teacher_result
            if _uses_intervention_distribution_gap(intervention_config.gap_metric):
                student_result = await self._score_student_sequence_for_intervention(
                    request_id=f"{base_request_id}:student-score:{attempt}",
                    prompt_ids=prompt_ids,
                    response_ids=response_ids,
                    output=output,
                )
                gaps = _compute_distribution_intervention_gap(
                    student_result=student_result,
                    teacher_result=teacher_result,
                    prompt_len=len(prompt_ids),
                    response_len=len(response_ids),
                    metric=intervention_config.gap_metric,
                    tokenizer=self.tokenizer,
                    log_prob_floor=_logprob_floor_from_distillation_config(self.distillation_config),
                )
            else:
                teacher_response_logprobs = _teacher_response_logprobs(
                    teacher_logprobs=_teacher_gap_logprobs(teacher_result),
                    prompt_len=len(prompt_ids),
                    response_len=len(response_ids),
                )
                student_response_logprobs = torch.tensor(response_logprobs[: len(response_ids)], dtype=torch.float32)
                if teacher_response_logprobs.shape != student_response_logprobs.shape:
                    raise ValueError(
                        "Teacher intervention scoring produced misaligned logprobs: "
                        f"{teacher_response_logprobs.shape=} vs {student_response_logprobs.shape=}, "
                        f"prompt_len={len(prompt_ids)}, response_len={len(response_ids)}, attempt={attempt}."
                    )
                gaps = _compute_intervention_gap(
                    student_logprobs=student_response_logprobs,
                    teacher_logprobs=teacher_response_logprobs,
                    metric=intervention_config.gap_metric,
                )
            bad_segment = _select_intervention_segment(gaps, gap_mask, intervention_config)
            if bad_segment is None or len(interventions) >= intervention_config.max_interventions_per_sample:
                break

            start, end, score = bad_segment
            start, end = AgentLoopWorker._expand_teacher_intervention_segment_for_token_patterns(
                self,
                response_ids,
                start,
                end,
            )
            original_student_segment_ids = response_ids[start:end]
            prefix_response_ids = response_ids[:start]
            prefix_response_mask = response_mask[:start]
            prefix_response_logprobs = response_logprobs[:start]
            prefix_gap_mask = gap_mask[:start]
            teacher_max_tokens = min(
                intervention_config.teacher_segment_length,
                self.rollout_config.response_length - len(prefix_response_ids),
            )
            if teacher_max_tokens <= 0:
                _record_skipped_intervention(
                    skipped_interventions,
                    reason="teacher_token_budget_exhausted",
                    attempt=attempt,
                    start=start,
                    end=end,
                    candidate_response_ids=response_ids,
                    candidate_response_mask=response_mask,
                )
                break

            teacher_generation = _coerce_teacher_generation_result(
                await self.teacher_server_manager.generate_teacher_segment_single(
                    prefix_ids=prompt_ids + prefix_response_ids,
                    max_tokens=teacher_max_tokens,
                    multi_modal_data=output.multi_modal_data,
                    mm_processor_kwargs=output.mm_processor_kwargs,
                    routing_key=routing_key,
                    request_id=f"{base_request_id}:teacher-generate:{attempt}",
                )
            )
            if _generation_was_aborted(
                stop_reason=teacher_generation.stop_reason,
                finish_reason=teacher_generation.finish_reason,
            ):
                _record_skipped_intervention(
                    skipped_interventions,
                    reason="teacher_generation_aborted",
                    attempt=attempt,
                    start=start,
                    end=end,
                    candidate_response_ids=prefix_response_ids,
                    candidate_response_mask=prefix_response_mask,
                    teacher_tokens=len(teacher_generation.token_ids),
                )
                break
            teacher_segment_ids, _, teacher_eos_truncated = _truncate_generation_at_eos(
                list(teacher_generation.token_ids),
                None,
                eos_token_ids=eos_token_ids,
            )
            if len(teacher_segment_ids) > teacher_max_tokens:
                raise RuntimeError(
                    "Teacher intervention generation returned more tokens than requested: "
                    f"len(teacher_segment_ids)={len(teacher_segment_ids)}, "
                    f"teacher_max_tokens={teacher_max_tokens}. "
                    "The rollout backend should return generated tokens only and respect sampling max_tokens."
                )
            if not teacher_segment_ids:
                _record_skipped_intervention(
                    skipped_interventions,
                    reason="teacher_generation_empty",
                    attempt=attempt,
                    start=start,
                    end=end,
                    candidate_response_ids=prefix_response_ids,
                    candidate_response_mask=prefix_response_mask,
                )
                break
            invalid_teacher_token_ids = _invalid_token_ids_for_tokenizer(teacher_segment_ids, self.tokenizer)
            if invalid_teacher_token_ids:
                _record_skipped_intervention(
                    skipped_interventions,
                    reason="teacher_generation_invalid_token_ids",
                    attempt=attempt,
                    start=start,
                    end=end,
                    candidate_response_ids=prefix_response_ids,
                    candidate_response_mask=prefix_response_mask,
                    teacher_tokens=len(teacher_segment_ids),
                )
                skipped_interventions[-1]["invalid_token_ids"] = invalid_teacher_token_ids[:8]
                skipped_interventions[-1]["tokenizer_vocab_size"] = _tokenizer_vocab_size(self.tokenizer)
                break
            rewrite_overlap_metrics = compute_rewrite_overlap_metrics(original_student_segment_ids, teacher_segment_ids)

            teacher_terminal = _generation_is_terminal(
                token_ids=teacher_segment_ids,
                max_tokens=teacher_max_tokens,
                stop_reason=teacher_generation.stop_reason,
                finish_reason=teacher_generation.finish_reason,
                eos_truncated=teacher_eos_truncated,
            )
            teacher_segment_mask_value = 1 if intervention_config.train_on_teacher_segments else 0
            teacher_segment_mask = [teacher_segment_mask_value] * len(teacher_segment_ids)
            # Placeholder values keep response_logprobs aligned. They are ignored when teacher tokens are masked,
            # and train_on_teacher_segments=True drops rollout logprobs so the actor recomputes them.
            teacher_segment_logprobs = [0.0] * len(teacher_segment_ids)
            remaining_tokens = self.rollout_config.response_length - len(prefix_response_ids) - len(teacher_segment_ids)
            student_suffix_result = _StudentSuffixGenerationResult(token_ids=[], logprobs=[])
            if not teacher_terminal:
                student_suffix_result = await self._generate_student_suffix(
                    request_id=base_request_id,
                    prefix_ids=prompt_ids + prefix_response_ids + teacher_segment_ids,
                    max_tokens=remaining_tokens,
                    sampling_params=sampling_params,
                    output=output,
                )
            candidate_response_ids = prefix_response_ids + teacher_segment_ids + student_suffix_result.token_ids
            candidate_response_mask = (
                prefix_response_mask + teacher_segment_mask + [1] * len(student_suffix_result.token_ids)
            )
            candidate_response_logprobs = (
                prefix_response_logprobs + teacher_segment_logprobs + student_suffix_result.logprobs
            )
            candidate_gap_mask = (
                prefix_gap_mask + [0] * len(teacher_segment_ids) + [1] * len(student_suffix_result.token_ids)
            )
            _validate_intervention_state(
                context=f"after repair attempt {attempt}",
                response_ids=candidate_response_ids,
                response_mask=candidate_response_mask,
                response_logprobs=candidate_response_logprobs,
                gap_mask=candidate_gap_mask,
                response_length_limit=self.rollout_config.response_length,
            )
            thinking_state = AgentLoopWorker._teacher_intervention_thinking_state(
                self,
                candidate_response_ids,
                prompt_ids=prompt_ids,
            )
            if not thinking_state.valid:
                _record_skipped_intervention(
                    skipped_interventions,
                    reason="invalid_thinking_block",
                    attempt=attempt,
                    start=start,
                    end=end,
                    candidate_response_ids=candidate_response_ids,
                    candidate_response_mask=candidate_response_mask,
                    teacher_tokens=len(teacher_segment_ids),
                    student_suffix_tokens=len(student_suffix_result.token_ids),
                )
                skipped_interventions[-1]["thinking_reason"] = thinking_state.reason
                break
            candidate_terminal = teacher_terminal or student_suffix_result.terminal
            if (
                candidate_terminal
                and thinking_state.open
                and getattr(intervention_config, "require_closed_thinking_on_terminal", True)
            ):
                _record_skipped_intervention(
                    skipped_interventions,
                    reason="terminal_open_thinking_block",
                    attempt=attempt,
                    start=start,
                    end=end,
                    candidate_response_ids=candidate_response_ids,
                    candidate_response_mask=candidate_response_mask,
                    teacher_tokens=len(teacher_segment_ids),
                    student_suffix_tokens=len(student_suffix_result.token_ids),
                )
                break
            if sum(candidate_response_mask) < min_trainable_tokens:
                _record_skipped_intervention(
                    skipped_interventions,
                    reason="insufficient_trainable_tokens",
                    attempt=attempt,
                    start=start,
                    end=end,
                    candidate_response_ids=candidate_response_ids,
                    candidate_response_mask=candidate_response_mask,
                    teacher_tokens=len(teacher_segment_ids),
                    student_suffix_tokens=len(student_suffix_result.token_ids),
                )
                break

            response_ids = candidate_response_ids
            response_mask = candidate_response_mask
            response_logprobs = candidate_response_logprobs
            gap_mask = candidate_gap_mask
            cached_teacher_result = None
            interventions.append(
                {
                    "attempt": attempt,
                    "start": start,
                    "end": end,
                    "score": score,
                    "teacher_tokens": len(teacher_segment_ids),
                    "student_suffix_tokens": len(student_suffix_result.token_ids),
                    "teacher_stop_reason": teacher_generation.stop_reason,
                    "teacher_finish_reason": teacher_generation.finish_reason,
                    "student_suffix_stop_reason": student_suffix_result.stop_reason,
                    "student_suffix_finish_reason": student_suffix_result.finish_reason,
                    "terminated": candidate_terminal,
                    **rewrite_overlap_metrics,
                }
            )
            if candidate_terminal:
                break

        output.response_ids = response_ids[: self.rollout_config.response_length]
        output.response_mask = response_mask[: len(output.response_ids)]
        output.response_logprobs = response_logprobs[: len(output.response_ids)]
        _validate_intervention_state(
            context="final",
            response_ids=output.response_ids,
            response_mask=output.response_mask,
            response_logprobs=output.response_logprobs,
            gap_mask=gap_mask[: len(output.response_ids)],
            response_length_limit=self.rollout_config.response_length,
        )
        output.extra_fields["teacher_intervention_count"] = len(interventions)
        output.extra_fields["teacher_interventions"] = interventions
        output.extra_fields["teacher_intervention_skipped"] = skipped_interventions
        if cached_teacher_result is not None:
            _store_teacher_scoring_result(
                output,
                cached_teacher_result,
                prompt_len=len(prompt_ids),
                response_len=len(output.response_ids),
            )
        if interventions and self._teacher_intervention_logged_samples < intervention_config.log_examples_per_worker:
            logger.info("Teacher interventions applied: %s", interventions)
            self._teacher_intervention_logged_samples += 1

    async def _compute_teacher_logprobs(
        self,
        output: AgentLoopOutput,
        prompt_ids: list[int],
        response_ids: list[int],
        validate: bool,
        sample_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Compute teacher logprobs for single sample."""
        if self.distillation_enabled and not validate:
            prompt_logprobs = AgentLoopWorker._teacher_prompt_logprobs(self)
            expected_width = max(prompt_logprobs, 1)
            existing_ids = output.extra_fields.get("teacher_ids")
            existing_logprobs = output.extra_fields.get("teacher_logprobs")
            existing_sampled_ids = output.extra_fields.get("teacher_sampled_ids")
            existing_sampled_logprobs = output.extra_fields.get("teacher_sampled_logprobs")
            has_required_sampled_fields = expected_width == 1 or (
                existing_sampled_ids is not None
                and existing_sampled_logprobs is not None
                and existing_sampled_ids.shape[0]
                == existing_sampled_logprobs.shape[0]
                == len(prompt_ids) + len(response_ids)
                and existing_sampled_logprobs.shape[-1] == 1
            )
            if (
                existing_ids is not None
                and existing_logprobs is not None
                and existing_ids.shape[0] == existing_logprobs.shape[0] == len(prompt_ids) + len(response_ids)
                and existing_logprobs.shape[-1] == expected_width
                and has_required_sampled_fields
            ):
                output.extra_fields.setdefault("teacher_logprob_nonfinite_count", 0)
                output.extra_fields.setdefault("teacher_logprob_nonfinite_positions", [])
                output.extra_fields.setdefault(
                    "teacher_logprob_nonfinite_replacement", DEFAULT_NONFINITE_LOGPROB_REPLACEMENT
                )
                return
            routing_key = self._teacher_routing_key(sample_kwargs)
            teacher_result = await self.teacher_server_manager.compute_teacher_logprobs_single(
                sequence_ids=prompt_ids + response_ids,
                multi_modal_data=output.multi_modal_data,
                mm_processor_kwargs=output.mm_processor_kwargs,
                routing_key=routing_key,
                request_id=output.extra_fields.get("rollout_request_id"),
                prompt_logprobs=prompt_logprobs,
            )
            _store_teacher_scoring_result(
                output,
                _coerce_teacher_logprob_scoring_result(teacher_result),
                prompt_len=len(prompt_ids),
                response_len=len(response_ids),
            )

    def _postprocess(
        self,
        inputs: list[_InternalAgentLoopOutput],
        input_non_tensor_batch: dict | None = None,
        validate: bool = False,
    ) -> DataProto:
        """Process the padded outputs from _run_agent_loop and combine them into a batch."""
        # Convert lists back to tensors and stack them to create a batch.
        prompt_ids = torch.cat([input.prompt_ids for input in inputs], dim=0)
        response_ids = torch.cat([input.response_ids for input in inputs], dim=0)
        response_mask = torch.cat([input.response_mask for input in inputs], dim=0)
        attention_mask = torch.cat([input.attention_mask for input in inputs], dim=0)
        input_ids = torch.cat([input.input_ids for input in inputs], dim=0)
        position_ids = torch.cat([input.position_ids for input in inputs], dim=0)
        optional_outputs = {}
        has_response_logprobs = _require_consistent_optional_field(inputs, "response_logprobs")
        has_routed_experts = _require_consistent_optional_field(inputs, "routed_experts")
        has_teacher_logprobs = _require_consistent_optional_field(inputs, "teacher_logprobs")
        has_teacher_ids = _require_consistent_optional_field(inputs, "teacher_ids")
        has_teacher_sampled_logprobs = _require_consistent_optional_field(inputs, "teacher_sampled_logprobs")
        has_teacher_sampled_ids = _require_consistent_optional_field(inputs, "teacher_sampled_ids")
        if has_teacher_logprobs != has_teacher_ids:
            raise RuntimeError(
                "Inconsistent teacher scoring fields across agent loop outputs: teacher_logprobs and teacher_ids "
                "must both be present or both be absent for every sample."
            )
        if has_teacher_sampled_logprobs != has_teacher_sampled_ids:
            raise RuntimeError(
                "Inconsistent teacher sampled scoring fields across agent loop outputs: teacher_sampled_logprobs "
                "and teacher_sampled_ids must both be present or both be absent for every sample."
            )
        if has_response_logprobs:
            optional_outputs["rollout_log_probs"] = torch.cat([input.response_logprobs for input in inputs], dim=0)
        if has_routed_experts:
            optional_outputs["routed_experts"] = torch.cat([input.routed_experts for input in inputs], dim=0)
        if has_teacher_logprobs and has_teacher_ids:
            optional_outputs["teacher_logprobs"] = torch.cat([input.teacher_logprobs for input in inputs], dim=0)
            optional_outputs["teacher_ids"] = torch.cat([input.teacher_ids for input in inputs], dim=0)
        if has_teacher_sampled_logprobs and has_teacher_sampled_ids:
            optional_outputs["teacher_sampled_logprobs"] = torch.cat(
                [input.teacher_sampled_logprobs for input in inputs], dim=0
            )
            optional_outputs["teacher_sampled_ids"] = torch.cat([input.teacher_sampled_ids for input in inputs], dim=0)
        batch = TensorDict(
            {
                "prompts": prompt_ids,  # [bsz, prompt_length]
                "responses": response_ids,  # [bsz, response_length]
                "response_mask": response_mask,  # [bsz, response_length]
                "input_ids": input_ids,  # [bsz, prompt_length + response_length]
                "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
                # position_ids: [bsz, 3, prompt_length + response_length] or [bsz, prompt_length + response_length]
                "position_ids": position_ids,
                **optional_outputs,
            },
            batch_size=len(inputs),
        )

        scores = [input.reward_score for input in inputs]
        if all(score is not None for score in scores):
            prompt_length = prompt_ids.size(1)
            response_length = attention_mask[:, prompt_length:].sum(dim=1) - 1
            rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
            rm_scores[torch.arange(response_mask.size(0)), response_length] = torch.tensor(scores, dtype=torch.float32)
            batch["rm_scores"] = rm_scores

        non_tensor_batch = {
            "__num_turns__": np.array([input.num_turns for input in inputs], dtype=np.int32),
        }
        if self.reward_loop_worker_handles is None and input_non_tensor_batch:
            non_tensor_batch.update(input_non_tensor_batch)

        # add reward_extra_info to non_tensor_batch
        reward_extra_infos = [input.extra_fields.get("reward_extra_info", {}) for input in inputs]
        reward_extra_keys = list(reward_extra_infos[0].keys())
        for key in reward_extra_keys:
            non_tensor_batch[key] = np.array([info[key] for info in reward_extra_infos])

        # Add multi_modal_inputs to non_tensor_batch if any samples have them
        multi_modal_inputs_list = [input.multi_modal_inputs for input in inputs]
        if any(mmi is not None for mmi in multi_modal_inputs_list):
            non_tensor_batch["multi_modal_inputs"] = np.array(multi_modal_inputs_list, dtype=object)

        metrics = [input.metrics.model_dump() for input in inputs]
        # Collect extra fields from all inputs and convert them to np.ndarray
        # Keep a stable set of keys so downstream batch concat stays consistent across agent loops.
        extra_fields = {}
        default_extra_keys = {
            "turn_scores",
            "tool_rewards",
            "min_global_steps",
            "max_global_steps",
            "extras",
        }
        all_keys = set(key for input_item in inputs for key in input_item.extra_fields) | default_extra_keys
        for key in all_keys:
            temp_arr = np.empty(len(inputs), dtype=object)
            temp_arr[:] = [input.extra_fields.get(key) for input in inputs]
            extra_fields[key] = temp_arr

        non_tensor_batch.update(extra_fields)

        # Only include reward_extra_keys in meta_info if rm_scores is in batch
        # This avoids conflicts when reward_tensor is merged later in ray_trainer.py
        if "rm_scores" in batch.keys():
            meta_info = {"metrics": metrics, "reward_extra_keys": reward_extra_keys}
        else:
            meta_info = {"metrics": metrics}

        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info=meta_info,
        )


async def get_trajectory_info(step, index, validate):
    """Get trajectory info.

    Args:
        step (int): global steps in the trainer.
        index (list): form datastore extra_info.index column.
        validate (bool): whether is a validate step.

    Returns:
        list: trajectory.
    """
    trajectory_info = []
    rollout_n = 0
    for i in range(len(index)):
        if i > 0 and index[i - 1] == index[i]:
            rollout_n += 1
        else:
            rollout_n = 0
        trajectory_info.append({"step": step, "sample_index": index[i], "rollout_n": rollout_n, "validate": validate})
    return trajectory_info


class AgentLoopManager:
    """Agent loop manager that manages a group of agent loop workers.

    Args:
        config (DictConfig): whole config for main entrypoint.
        llm_client (LLMServerClient): Client for the LLM server.
        teacher_client (dict[str, LLMServerClient]): Client for multiple teacher servers.
        reward_loop_worker_handles (List[ray.actor.ActorHandle]): Actor handles for streaming reward computation.
    """

    def __init__(
        self,
        config: DictConfig,
        llm_client: LLMServerClient,
        teacher_client: dict[str, LLMServerClient] = None,
        reward_loop_worker_handles: list[ray.actor.ActorHandle] = None,
    ):
        self.config = config
        self.rollout_config = config.actor_rollout_ref.rollout
        self.model_config = config.actor_rollout_ref.model
        self.llm_client = llm_client
        self.teacher_client = teacher_client
        self.reward_loop_worker_handles = reward_loop_worker_handles

        if not hasattr(self, "agent_loop_workers_class"):
            self.agent_loop_workers_class = ray.remote(AgentLoopWorker)

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        """Create agent loop manager."""
        instance = cls(*args, **kwargs)
        await instance._init_agent_loop_workers()
        return instance

    async def _init_agent_loop_workers(self):
        self.agent_loop_workers = []
        num_workers = self.rollout_config.agent.num_workers

        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
        for i in range(num_workers):
            # Round-robin scheduling over the all nodes
            node_id = node_ids[i % len(node_ids)]
            self.agent_loop_workers.append(
                self.agent_loop_workers_class.options(
                    name=f"agent_loop_worker_{i}" + f"_{uuid4().hex[:8]}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=True
                    ),
                ).remote(
                    self.config,
                    self.llm_client,
                    self.teacher_client,
                    self.reward_loop_worker_handles,
                )
            )

    @auto_await
    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Split input batch and dispatch to agent loop workers.

        Args:
            prompts (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
        """
        chunkes = prompts.chunk(len(self.agent_loop_workers))
        outputs = await asyncio.gather(
            *[
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=True)
            ]
        )
        output = DataProto.concat(outputs)

        # calculate performance metrics
        metrics = [output.meta_info.pop("metrics") for output in outputs]  # List[List[Dict[str, str]]]
        timing = self._performance_metrics(metrics, output)

        output.meta_info = {"timing": timing, **outputs[0].meta_info}
        return output

    def _performance_metrics(self, metrics: list[list[dict[str, str]]], output: DataProto) -> dict[str, float]:
        timing = {}
        t_generate_sequences = np.array([metric["generate_sequences"] for chunk in metrics for metric in chunk])
        t_tool_calls = np.array([metric["tool_calls"] for chunk in metrics for metric in chunk])
        t_compute_score = np.array([metric["compute_score"] for chunk in metrics for metric in chunk])
        num_preempted = np.array([metric["num_preempted"] for chunk in metrics for metric in chunk])
        timing["agent_loop/num_preempted/min"] = num_preempted.min()
        timing["agent_loop/num_preempted/max"] = num_preempted.max()
        timing["agent_loop/num_preempted/mean"] = num_preempted.mean()
        timing["agent_loop/generate_sequences/min"] = t_generate_sequences.min()
        timing["agent_loop/generate_sequences/max"] = t_generate_sequences.max()
        timing["agent_loop/generate_sequences/mean"] = t_generate_sequences.mean()
        timing["agent_loop/tool_calls/min"] = t_tool_calls.min()
        timing["agent_loop/tool_calls/max"] = t_tool_calls.max()
        timing["agent_loop/tool_calls/mean"] = t_tool_calls.mean()
        timing["agent_loop/compute_score/min"] = t_compute_score.min()
        timing["agent_loop/compute_score/max"] = t_compute_score.max()
        timing["agent_loop/compute_score/mean"] = t_compute_score.mean()

        # batch sequence generation is bounded by the slowest sample
        slowest = np.argmax(t_generate_sequences + t_tool_calls + t_compute_score)
        prompt_length = output.batch["prompts"].shape[1]
        timing["agent_loop/slowest/generate_sequences"] = t_generate_sequences[slowest]
        timing["agent_loop/slowest/tool_calls"] = t_tool_calls[slowest]
        timing["agent_loop/slowest/compute_score"] = t_compute_score[slowest]
        timing["agent_loop/slowest/num_preempted"] = num_preempted[slowest]

        if "attention_mask" in output.batch:
            attention_mask = output.batch["attention_mask"][slowest]
            timing["agent_loop/slowest/prompt_length"] = attention_mask[:prompt_length].sum().item()
            timing["agent_loop/slowest/response_length"] = attention_mask[prompt_length:].sum().item()

        return timing
