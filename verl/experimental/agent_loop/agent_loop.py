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
import copy
import hashlib
import logging
import math
import os
import random
import time
from abc import ABC, abstractmethod
from collections import deque
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
MAX_NONFINITE_STUDENT_SUFFIX_LOGPROB_WARNINGS = 10
INTERVENTION_TOPK_DISTRIBUTION_GAP_METRICS = {"topk_kl", "topk_distribution_kl", "topk_dist_kl"}
INTERVENTION_FULL_DISTRIBUTION_GAP_METRICS = {"full_kl", "full_distribution_kl", "distribution_kl"}
INTERVENTION_LOCAL_EFFECT_V2_WINDOWS = (1, 8, 16, 32, 64, 128)


@dataclass
class _StudentSuffixGenerationResult:
    token_ids: list[int]
    logprobs: list[float]
    stop_reason: Optional[str] = None
    finish_reason: Optional[str] = None
    terminal: bool = False
    eos_truncated: bool = False
    nonfinite_logprob_offsets: tuple[int, ...] = ()
    nonfinite_logprob_replacement: Optional[float] = None


@dataclass(frozen=True)
class _InterventionScoreResult:
    """Active per-token risk plus reusable named feature components."""

    active_scores: torch.Tensor
    components: dict[str, torch.Tensor]


@dataclass
class _PendingFutureGapConfirmation:
    """One short-horizon repair candidate awaiting its already-scheduled score."""

    proposal: dict[str, Any]
    intervention: dict[str, Any]
    after_intervention_start: int
    baseline_values: list[float]
    raw_response_ids: list[int]
    raw_response_mask: list[int]
    raw_response_logprobs: list[float]
    raw_gap_mask: list[int]
    raw_teacher_segment_mask: Optional[list[int]]
    raw_teacher_result: TeacherLogprobScoringResult
    confirmation_suffix_result: _StudentSuffixGenerationResult


@dataclass
class _ThinkingBlockState:
    valid: bool = True
    open: bool = False
    reason: Optional[str] = None


@dataclass
class _TeacherRepairCandidate:
    index: int
    generation: TeacherGenerationResult
    token_ids: list[int]
    eos_truncated: bool
    teacher_nll: Optional[float] = None
    student_nll: Optional[float] = None
    teacher_rank: Optional[int] = None
    student_rank: Optional[int] = None
    rank_sum: Optional[int] = None
    invalid_reason: Optional[str] = None


def _merge_student_suffix_results(
    confirmation: _StudentSuffixGenerationResult,
    continuation: _StudentSuffixGenerationResult,
) -> _StudentSuffixGenerationResult:
    """Combine two token-ID continuation calls into one logical student suffix."""
    confirmation_length = len(confirmation.token_ids)
    return _StudentSuffixGenerationResult(
        token_ids=confirmation.token_ids + continuation.token_ids,
        logprobs=confirmation.logprobs + continuation.logprobs,
        stop_reason=continuation.stop_reason,
        finish_reason=continuation.finish_reason,
        terminal=continuation.terminal,
        eos_truncated=continuation.eos_truncated,
        nonfinite_logprob_offsets=(
            confirmation.nonfinite_logprob_offsets
            + tuple(confirmation_length + offset for offset in continuation.nonfinite_logprob_offsets)
        ),
        nonfinite_logprob_replacement=(
            confirmation.nonfinite_logprob_replacement
            if confirmation.nonfinite_logprob_replacement is not None
            else continuation.nonfinite_logprob_replacement
        ),
    )


def _student_suffix_logprob_metadata(
    result: _StudentSuffixGenerationResult,
    *,
    response_start: int,
) -> dict[str, Any]:
    """Build response-relative diagnostics for sanitized student suffix logprobs."""
    if not result.nonfinite_logprob_offsets:
        return {}
    response_positions = [response_start + offset for offset in result.nonfinite_logprob_offsets]
    return {
        "student_suffix_logprob_nonfinite_count": len(response_positions),
        "student_suffix_logprob_nonfinite_offsets": list(result.nonfinite_logprob_offsets),
        "student_suffix_logprob_nonfinite_response_positions": response_positions,
        "student_suffix_logprob_nonfinite_replacement": result.nonfinite_logprob_replacement,
    }


@dataclass
class _TeacherCandidateSelectionResult:
    candidate: Optional[_TeacherRepairCandidate]
    metadata: dict[str, Any]
    skip_reason: Optional[str] = None


@dataclass(frozen=True)
class _TeacherNLLCalibrationSnapshot:
    """Read-only global teacher-NLL calibration state broadcast to rollout workers."""

    threshold: Optional[float]
    source: str
    count: int
    version: int


class _GlobalTeacherNLLCalibration:
    """Manager-owned rolling calibration history shared by all agent-loop workers."""

    def __init__(self, config: DistillationInterventionConfig):
        self.quantile = float(config.teacher_nll_calibration_quantile)
        self.min_samples = int(config.teacher_nll_calibration_min_samples)
        self.initial_threshold = (
            None if config.teacher_nll_initial_threshold is None else float(config.teacher_nll_initial_threshold)
        )
        self.history: deque[float] = deque(maxlen=int(config.teacher_nll_calibration_window))
        self.version = 0

    def snapshot(self) -> _TeacherNLLCalibrationSnapshot:
        """Return the threshold state derived from the current global history."""
        count = len(self.history)
        if count >= self.min_samples:
            threshold = float(np.quantile(np.asarray(self.history, dtype=np.float64), self.quantile))
            source = "rolling_healthy_quantile"
        elif self.initial_threshold is not None:
            threshold = self.initial_threshold
            source = "initial"
        else:
            threshold = None
            source = "unavailable"
        return _TeacherNLLCalibrationSnapshot(
            threshold=threshold,
            source=source,
            count=count,
            version=self.version,
        )

    def add_updates(self, worker_updates: list[list[float]]) -> _TeacherNLLCalibrationSnapshot:
        """Merge one batch of per-worker healthy NLL updates and return the new snapshot."""
        updates: list[float] = []
        for worker_index, values in enumerate(worker_updates):
            for value_index, value in enumerate(values):
                value = float(value)
                if not math.isfinite(value) or value < 0:
                    raise ValueError(
                        "Global teacher-NLL calibration updates must be finite and non-negative, "
                        f"but worker {worker_index} update {value_index} was {value!r}."
                    )
                updates.append(value)
        if updates:
            self.history.extend(updates)
            self.version += 1
        return self.snapshot()

    def export_state(self) -> dict[str, Any]:
        """Return a JSON-safe snapshot for read-only standalone evaluation."""

        snapshot = self.snapshot()
        return {
            "quantile": self.quantile,
            "window": self.history.maxlen,
            "min_samples": self.min_samples,
            "initial_threshold": self.initial_threshold,
            "history": list(self.history),
            "threshold": snapshot.threshold,
            "source": snapshot.source,
            "count": snapshot.count,
            "version": snapshot.version,
        }


def _as_single_float(value: Any, *, context: str) -> float:
    """Normalize a reward-manager scalar without silently reducing a batch."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise RuntimeError(f"{context} must contain exactly one value, got shape={tuple(value.shape)}.")
        value = value.detach().cpu().item()
    elif isinstance(value, np.ndarray):
        if value.size != 1:
            raise RuntimeError(f"{context} must contain exactly one value, got shape={value.shape}.")
        value = value.reshape(-1)[0].item()
    elif isinstance(value, list | tuple):
        if len(value) != 1:
            raise RuntimeError(f"{context} must contain exactly one value, got len={len(value)}.")
        return _as_single_float(value[0], context=context)
    elif isinstance(value, np.generic):
        value = value.item()
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{context} must be numeric, got {value!r}.") from exc
    if not math.isfinite(scalar):
        raise RuntimeError(f"{context} must be finite, got {scalar!r}.")
    return scalar


def _extract_outcome_correctness(
    output: "AgentLoopOutput",
    intervention_config: DistillationInterventionConfig,
) -> tuple[bool, float]:
    reward_extra_info = output.extra_fields.get("reward_extra_info")
    if not isinstance(reward_extra_info, dict):
        raise RuntimeError(
            "Teacher-intervention outcome observation requires reward_extra_info from the reward manager, "
            "but it was unavailable. Configure an async reward loop that returns the correctness field "
            f"{intervention_config.outcome_correctness_key!r}."
        )
    correctness_key = str(intervention_config.outcome_correctness_key)
    if correctness_key not in reward_extra_info:
        raise RuntimeError(
            "Teacher-intervention outcome observation could not find its correctness signal: "
            f"reward_extra_info has keys={sorted(reward_extra_info)}, expected {correctness_key!r}."
        )
    correctness_value = _as_single_float(
        reward_extra_info[correctness_key],
        context=f"reward_extra_info[{correctness_key!r}]",
    )
    return correctness_value > float(intervention_config.outcome_correctness_threshold), correctness_value


def _outcome_transition(raw_correct: bool, final_correct: bool) -> str:
    """Return the canonical sample-level pre/post correctness transition."""
    if raw_correct:
        return "pre_correct_post_correct" if final_correct else "pre_correct_post_wrong"
    return "pre_wrong_post_correct" if final_correct else "pre_wrong_post_wrong"


def _token_ids_sha256(token_ids: list[int]) -> str:
    return hashlib.sha256(np.asarray(token_ids, dtype=np.int64).tobytes()).hexdigest()


def _metadata_scalar(value: Any) -> Any:
    """Normalize scalar rollout metadata for JSON-safe diagnostic records."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return value.detach().cpu().item()
    if isinstance(value, np.ndarray) and value.size == 1:
        return value.reshape(-1)[0].item()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return str(value)


def _stable_fraction(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) / float(1 << 64)


def _boundary_structure_tags(left_text: str, right_text: str) -> list[str]:
    """Describe lightweight surface structure at a decoded token seam."""
    tags: list[str] = []
    if not right_text:
        tags.append("terminal")
    if left_text.endswith(("\n", "\r")) or right_text.startswith(("\n", "\r")):
        tags.append("newline")
    left_stripped = left_text.rstrip()
    if left_stripped.endswith((".", "?", "!", "。", "？", "！")):
        tags.append("sentence_end")
    if left_stripped.endswith((",", ";", ":", "，", "；", "：", "、")):
        tags.append("clause_delimiter")
    if (left_text and left_text[-1].isspace()) or (right_text and right_text[0].isspace()):
        tags.append("whitespace")
    return tags or ["none"]


def _rank_min(values: dict[int, float]) -> dict[int, int]:
    """Rank smaller finite values first, assigning equal values the same minimum rank."""
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    ranks: dict[int, int] = {}
    previous_value: Optional[float] = None
    previous_rank = 0
    for ordinal, (index, value) in enumerate(ordered, start=1):
        rank = previous_rank if previous_value is not None and value == previous_value else ordinal
        ranks[index] = rank
        previous_value = value
        previous_rank = rank
    return ranks


def _stable_choice(indices: list[int], *, key: str) -> int:
    if not indices:
        raise ValueError("_stable_choice requires at least one index.")
    ordered = sorted(indices)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return ordered[int.from_bytes(digest[:8], byteorder="big", signed=False) % len(ordered)]


def _compute_intervention_gap(
    student_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    metric: str,
    *,
    teacher_rejection_token_cap: float = 10.0,
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
    if metric == "teacher_rejection":
        if not math.isfinite(teacher_rejection_token_cap) or teacher_rejection_token_cap <= 0:
            raise ValueError(
                f"teacher_rejection_token_cap must be finite and positive, got {teacher_rejection_token_cap!r}."
            )
        return (student_logprobs - teacher_logprobs).clamp(min=0, max=teacher_rejection_token_cap)
    raise NotImplementedError(f"Unsupported teacher intervention gap metric: {metric}")


def _compose_intervention_score(
    *,
    student_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    metric: str,
    teacher_rejection_token_cap: float,
    distribution_scores: Optional[torch.Tensor] = None,
) -> _InterventionScoreResult:
    """Build the active intervention risk and named components from aligned token scores."""
    teacher_rejection = _compute_intervention_gap(
        student_logprobs=student_logprobs,
        teacher_logprobs=teacher_logprobs,
        metric="teacher_rejection",
        teacher_rejection_token_cap=teacher_rejection_token_cap,
    )
    metric = metric.lower()
    if distribution_scores is None:
        active_scores = _compute_intervention_gap(
            student_logprobs=student_logprobs,
            teacher_logprobs=teacher_logprobs,
            metric=metric,
            teacher_rejection_token_cap=teacher_rejection_token_cap,
        )
    else:
        active_scores = distribution_scores.float()
    if active_scores.shape != teacher_rejection.shape:
        raise RuntimeError(
            "Teacher intervention active risk and teacher-rejection component are misaligned: "
            f"active_scores.shape={tuple(active_scores.shape)}, "
            f"teacher_rejection.shape={tuple(teacher_rejection.shape)}."
        )
    components = {
        "teacher_rejection": teacher_rejection,
        "student_nll": -student_logprobs.float(),
        "teacher_nll": -teacher_logprobs.float(),
    }
    components.setdefault(metric, active_scores)
    return _InterventionScoreResult(active_scores=active_scores, components=components)


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
    """Return the first fixed-token segment that passes local and optional persistence gates."""
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
    for start in range(0, gaps.numel(), segment_len):
        end = min(start + segment_len, gaps.numel())
        score, _ = _aggregate_intervention_span(
            scores=gaps,
            response_mask=response_mask,
            start=start,
            end=end,
            aggregation=intervention_config.gap_aggregation,
        )
        if score is None:
            continue
        if score <= intervention_config.gap_threshold:
            continue
        if bool(getattr(intervention_config, "teacher_rejection_persistence_enabled", False)):
            persistence_score, _ = _compute_teacher_rejection_persistence(
                gaps=gaps,
                response_mask=response_mask,
                candidate_end=end,
                intervention_config=intervention_config,
            )
            persistence_threshold = _teacher_rejection_persistence_threshold(intervention_config)
            if persistence_score is None or persistence_score <= persistence_threshold:
                continue
        return start, end, score
    return None


def _aggregate_intervention_span(
    *,
    scores: torch.Tensor,
    response_mask: list[int],
    start: int,
    end: int,
    aggregation: str,
) -> tuple[Optional[float], int]:
    """Aggregate valid token scores within one response span."""
    if len(response_mask) != scores.numel():
        raise RuntimeError(
            "Teacher intervention span mask is not aligned with score values: "
            f"len(response_mask)={len(response_mask)}, num_scores={scores.numel()}."
        )
    if start < 0 or end < start or end > scores.numel():
        raise ValueError(f"Invalid teacher intervention span [{start}, {end}) for {scores.numel()} token scores.")
    if aggregation not in {"mean", "max"}:
        raise ValueError(f"Unsupported teacher intervention score aggregation: {aggregation!r}.")
    span_mask = torch.tensor(response_mask[start:end], dtype=torch.bool, device=scores.device)
    valid_count = int(span_mask.sum().item())
    if valid_count == 0:
        return None, 0
    values = scores[start:end][span_mask]
    score = values.max() if aggregation == "max" else values.mean()
    return float(score.item()), valid_count


def _teacher_rejection_persistence_threshold(intervention_config: DistillationInterventionConfig) -> float:
    threshold = intervention_config.teacher_rejection_persistence_threshold
    return float(intervention_config.gap_threshold if threshold is None else threshold)


def _compute_teacher_rejection_persistence(
    *,
    gaps: torch.Tensor,
    response_mask: list[int],
    candidate_end: int,
    intervention_config: DistillationInterventionConfig,
) -> tuple[Optional[float], int]:
    """Aggregate realized future teacher rejection after a local candidate segment."""
    future_end = min(
        candidate_end + intervention_config.teacher_rejection_persistence_window_tokens,
        gaps.numel(),
    )
    score, valid_tokens = _aggregate_intervention_span(
        scores=gaps,
        response_mask=response_mask,
        start=candidate_end,
        end=future_end,
        aggregation="mean",
    )
    if valid_tokens < intervention_config.teacher_rejection_persistence_min_valid_tokens:
        return None, valid_tokens
    return score, valid_tokens


def _future_gap_threshold(intervention_config: DistillationInterventionConfig) -> float:
    threshold = intervention_config.future_gap_max_threshold
    return float(intervention_config.gap_threshold if threshold is None else threshold)


def _collect_future_gap_values(
    *,
    scores: torch.Tensor,
    response_mask: list[int],
    start: int,
    window_tokens: int,
    teacher_nonfinite_positions: Optional[list[int]] = None,
) -> tuple[list[float], dict[str, int]]:
    """Copy at most one small score window to CPU and retain finite trainable values."""
    if len(response_mask) != scores.numel():
        raise RuntimeError(
            "Future-gap response mask is not aligned with score values: "
            f"len(response_mask)={len(response_mask)}, num_scores={scores.numel()}."
        )
    if start < 0 or start > scores.numel():
        raise ValueError(f"Invalid future-gap start={start} for {scores.numel()} token scores.")
    end = min(start + window_tokens, scores.numel())
    invalid_counts = {"masked": 0, "teacher_nonfinite": 0, "value_nonfinite": 0}
    if end <= start:
        return [], invalid_counts

    span = scores[start:end].detach().float().cpu()
    mask = torch.tensor(response_mask[start:end], dtype=torch.bool)
    invalid_counts["masked"] = int((~mask).sum().item())

    teacher_nonfinite_mask = torch.zeros(end - start, dtype=torch.bool)
    for position in teacher_nonfinite_positions or ():
        if start <= position < end:
            teacher_nonfinite_mask[position - start] = True
    invalid_counts["teacher_nonfinite"] = int((mask & teacher_nonfinite_mask).sum().item())

    finite_mask = torch.isfinite(span)
    invalid_counts["value_nonfinite"] = int((mask & ~teacher_nonfinite_mask & ~finite_mask).sum().item())
    valid_mask = mask & ~teacher_nonfinite_mask & finite_mask
    return [float(value) for value in span[valid_mask].tolist()], invalid_counts


def _evaluate_future_gap_confirmation(
    *,
    proposal: dict[str, Any],
    baseline_values: list[float],
    repaired_values: list[float],
    intervention_config: DistillationInterventionConfig,
) -> bool:
    """Populate scalar gate diagnostics and return whether the repair is accepted."""
    started_at = time.perf_counter()
    effective_tokens = min(
        intervention_config.future_gap_window_tokens,
        len(baseline_values),
        len(repaired_values),
    )
    threshold = _future_gap_threshold(intervention_config)
    proposal.update(
        {
            "future_gap_window_tokens": int(intervention_config.future_gap_window_tokens),
            "future_gap_min_valid_tokens": int(intervention_config.future_gap_min_valid_tokens),
            "future_gap_max_threshold": threshold,
            "future_gap_min_reduction": float(intervention_config.future_gap_min_reduction),
            "future_gap_effective_tokens": effective_tokens,
        }
    )
    baseline_invalid_counts = proposal.get("future_gap_baseline_invalid_counts") or {}
    repaired_invalid_counts = proposal.get("future_gap_repaired_invalid_counts") or {}
    nonfinite_count = sum(
        int(invalid_counts.get(key, 0) or 0)
        for invalid_counts in (baseline_invalid_counts, repaired_invalid_counts)
        for key in ("teacher_nonfinite", "value_nonfinite")
    )
    if nonfinite_count:
        proposal.update(
            {
                "future_gap_pair_available": False,
                "future_gap_nonfinite_count": nonfinite_count,
                "accepted": False,
                "status": "rejected",
                "rejection_stage": "future_gap_confirmation",
                "reason": "future_gap_nonfinite_pair",
                "future_gap_gate_cpu_seconds": time.perf_counter() - started_at,
            }
        )
        return False
    if effective_tokens < intervention_config.future_gap_min_valid_tokens:
        proposal.update(
            {
                "future_gap_pair_available": False,
                "accepted": False,
                "status": "rejected",
                "rejection_stage": "future_gap_confirmation",
                "reason": "future_gap_insufficient_valid_tokens",
                "future_gap_gate_cpu_seconds": time.perf_counter() - started_at,
            }
        )
        return False

    before = math.fsum(baseline_values[:effective_tokens]) / effective_tokens
    after = math.fsum(repaired_values[:effective_tokens]) / effective_tokens
    reduction = before - after
    finite = all(math.isfinite(value) for value in (before, after, reduction))
    passes_absolute = finite and after < threshold
    passes_reduction = finite and reduction > intervention_config.future_gap_min_reduction
    accepted = passes_absolute and passes_reduction
    if not finite:
        reason = "future_gap_nonfinite_pair"
    elif not passes_absolute:
        reason = "future_gap_after_threshold"
    elif not passes_reduction:
        reason = "future_gap_insufficient_reduction"
    else:
        reason = None
    proposal.update(
        {
            "future_gap_pair_available": finite,
            "future_gap_before": before if finite else None,
            "future_gap_after": after if finite else None,
            "future_gap_reduction": reduction if finite else None,
            "future_gap_passes_absolute_threshold": passes_absolute,
            "future_gap_passes_min_reduction": passes_reduction,
            "accepted": accepted,
            "status": "accepted" if accepted else "rejected",
            "rejection_stage": None if accepted else "future_gap_confirmation",
            "reason": reason,
            "future_gap_gate_cpu_seconds": time.perf_counter() - started_at,
        }
    )
    return accepted


def _initialize_local_effect_metadata(
    *,
    score_result: _InterventionScoreResult,
    response_mask: list[int],
    no_intervention_start: int,
    segment_length: int,
    gap_metric: str,
    gap_aggregation: str,
    collect_v2: bool = False,
    teacher_nonfinite_positions: Optional[list[int]] = None,
) -> dict[str, Any]:
    """Capture the next-segment baseline before applying the current intervention."""
    no_intervention_end = min(no_intervention_start + segment_length, score_result.active_scores.numel())
    active_gap, valid_tokens = _aggregate_intervention_span(
        scores=score_result.active_scores,
        response_mask=response_mask,
        start=no_intervention_start,
        end=no_intervention_end,
        aggregation=gap_aggregation,
    )
    teacher_rejection, rejection_tokens = _aggregate_intervention_span(
        scores=score_result.components["teacher_rejection"],
        response_mask=response_mask,
        start=no_intervention_start,
        end=no_intervention_end,
        aggregation="mean",
    )
    if rejection_tokens != valid_tokens:
        raise RuntimeError(
            "Teacher intervention local-effect components used different baseline token counts: "
            f"active={valid_tokens}, teacher_rejection={rejection_tokens}."
        )
    metadata: dict[str, Any] = {
        "local_effect_gap_metric": str(gap_metric).lower(),
        "local_effect_pair_available": False,
        "local_effect_no_current_intervention_start": no_intervention_start,
        "local_effect_no_current_intervention_tokens": valid_tokens,
    }
    if active_gap is not None:
        metadata["local_effect_no_current_intervention_gap"] = active_gap
    if teacher_rejection is not None:
        metadata["local_effect_no_current_intervention_d"] = teacher_rejection
    if valid_tokens == 0:
        metadata["local_effect_unavailable_reason"] = "no_no_current_intervention_tokens"
    if collect_v2:
        invalid_positions = set(teacher_nonfinite_positions or ())
        window_end = min(
            no_intervention_start + max(INTERVENTION_LOCAL_EFFECT_V2_WINDOWS),
            score_result.active_scores.numel(),
        )
        component_names = ("active", "teacher_rejection", "student_nll", "teacher_nll")
        component_tensors = (
            score_result.active_scores,
            score_result.components["teacher_rejection"],
            score_result.components["student_nll"],
            score_result.components["teacher_nll"],
        )
        component_values = {name: [] for name in component_names}
        invalid_counts = {"masked": 0, "teacher_nonfinite": 0, "value_nonfinite": 0}
        for position in range(no_intervention_start, window_end):
            if not response_mask[position]:
                invalid_counts["masked"] += 1
                continue
            if position in invalid_positions:
                invalid_counts["teacher_nonfinite"] += 1
                continue
            values = [float(tensor[position].item()) for tensor in component_tensors]
            if not all(math.isfinite(value) for value in values):
                invalid_counts["value_nonfinite"] += 1
                continue
            for name, value in zip(component_names, values, strict=True):
                component_values[name].append(value)
        metadata.update(
            {
                "_local_effect_v2_baseline_active_scores": component_values["active"],
                "_local_effect_v2_baseline_teacher_rejection_scores": component_values["teacher_rejection"],
                "_local_effect_v2_baseline_student_nll": component_values["student_nll"],
                "_local_effect_v2_baseline_teacher_nll": component_values["teacher_nll"],
                "local_effect_v2_baseline_available_tokens": len(component_values["active"]),
                "local_effect_v2_baseline_span_tokens": max(window_end - no_intervention_start, 0),
                "local_effect_v2_baseline_invalid_counts": invalid_counts,
            }
        )
        if not component_values["active"]:
            metadata["local_effect_v2_baseline_unavailable_reason"] = "no_finite_baseline_tokens"
    return metadata


def _finalize_local_effect_metadata(
    *,
    metadata: dict[str, Any],
    score_result: _InterventionScoreResult,
    response_mask: list[int],
    after_intervention_start: int,
    segment_length: int,
    gap_aggregation: str,
    teacher_nonfinite_positions: Optional[list[int]] = None,
) -> None:
    """Pair a repaired suffix segment with its already-recorded pre-intervention baseline."""
    after_intervention_end = min(after_intervention_start + segment_length, score_result.active_scores.numel())
    active_gap, valid_tokens = _aggregate_intervention_span(
        scores=score_result.active_scores,
        response_mask=response_mask,
        start=after_intervention_start,
        end=after_intervention_end,
        aggregation=gap_aggregation,
    )
    teacher_rejection, rejection_tokens = _aggregate_intervention_span(
        scores=score_result.components["teacher_rejection"],
        response_mask=response_mask,
        start=after_intervention_start,
        end=after_intervention_end,
        aggregation="mean",
    )
    if rejection_tokens != valid_tokens:
        raise RuntimeError(
            "Teacher intervention local-effect components used different repaired token counts: "
            f"active={valid_tokens}, teacher_rejection={rejection_tokens}."
        )
    metadata["local_effect_after_intervention_start"] = after_intervention_start
    metadata["local_effect_after_intervention_tokens"] = valid_tokens
    if active_gap is not None:
        metadata["local_effect_after_intervention_gap"] = active_gap
    if teacher_rejection is not None:
        metadata["local_effect_after_intervention_d"] = teacher_rejection

    before_gap = metadata.get("local_effect_no_current_intervention_gap")
    before_d = metadata.get("local_effect_no_current_intervention_d")
    if before_gap is not None and active_gap is not None:
        metadata["local_effect_gap_reduction"] = float(before_gap) - active_gap
    if before_d is not None and teacher_rejection is not None:
        metadata["local_effect_d_reduction"] = float(before_d) - teacher_rejection

    pair_available = "local_effect_gap_reduction" in metadata and "local_effect_d_reduction" in metadata
    metadata["local_effect_pair_available"] = pair_available
    if pair_available:
        metadata.pop("local_effect_unavailable_reason", None)
    elif valid_tokens == 0:
        metadata["local_effect_unavailable_reason"] = "no_after_intervention_tokens"
    else:
        metadata.setdefault("local_effect_unavailable_reason", "incomplete_local_effect_pair")

    if "_local_effect_v2_baseline_active_scores" in metadata:
        invalid_positions = set(teacher_nonfinite_positions or ())
        window_end = min(
            after_intervention_start + max(INTERVENTION_LOCAL_EFFECT_V2_WINDOWS),
            score_result.active_scores.numel(),
        )
        active_values: list[float] = []
        valid_offsets: list[int] = []
        invalid_counts = {"masked": 0, "teacher_nonfinite": 0, "value_nonfinite": 0}
        for position in range(after_intervention_start, window_end):
            if not response_mask[position]:
                invalid_counts["masked"] += 1
                continue
            if position in invalid_positions:
                invalid_counts["teacher_nonfinite"] += 1
                continue
            value = float(score_result.active_scores[position].item())
            if not math.isfinite(value):
                invalid_counts["value_nonfinite"] += 1
                continue
            active_values.append(value)
            valid_offsets.append(position - after_intervention_start)
        metadata.update(
            {
                "_local_effect_v2_scored_after_active_scores": active_values,
                "_local_effect_v2_scored_after_offsets": valid_offsets,
                "local_effect_v2_scored_after_available_tokens": len(active_values),
                "local_effect_v2_scored_after_invalid_counts": invalid_counts,
            }
        )


def _mark_local_effect_unavailable(metadata: dict[str, Any], reason: str) -> None:
    metadata["local_effect_pair_available"] = False
    metadata.setdefault("local_effect_after_intervention_tokens", 0)
    metadata["local_effect_unavailable_reason"] = reason


def _validate_intervention_state(
    *,
    context: str,
    response_ids: list[int],
    response_mask: list[int],
    response_logprobs: Optional[list[float]],
    gap_mask: list[int],
    response_length_limit: int,
    teacher_segment_mask: Optional[list[int]] = None,
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
    if teacher_segment_mask is not None:
        if len(teacher_segment_mask) != len(response_ids):
            raise RuntimeError(
                f"Teacher intervention state {context} has misaligned teacher_segment_mask: "
                f"len(teacher_segment_mask)={len(teacher_segment_mask)}, len(response_ids)={len(response_ids)}."
            )
        invalid_teacher_segment_mask = sorted({mask for mask in teacher_segment_mask if mask not in (0, 1)})
        if invalid_teacher_segment_mask:
            raise ValueError(
                f"Teacher intervention state {context} has non-binary teacher_segment_mask values: "
                f"{invalid_teacher_segment_mask}."
            )
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
    teacher_segment_mask: Optional[list[int]] = None
    """Response-aligned mask, 1 only for teacher-inserted tokens supervised with SFT."""
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

        teacher_segment_mask = output.pop("teacher_segment_mask", None)
        if teacher_segment_mask is not None:
            output["teacher_segment_mask"] = torch.tensor(teacher_segment_mask, dtype=torch.int64)

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
    teacher_segment_mask: Optional[torch.Tensor] = None
    """Padded response-aligned teacher SFT supervision mask."""
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
        self._teacher_nll_calibration_snapshot = _TeacherNLLCalibrationSnapshot(
            threshold=None,
            source="unavailable",
            count=0,
            version=0,
        )
        self._teacher_nll_pending_updates: list[float] = []
        self._teacher_candidate_rpc_semaphore: Optional[asyncio.Semaphore] = None
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
            if self.teacher_intervention_config.outcome_aware_candidate_selection_enabled:
                initial_threshold = self.teacher_intervention_config.teacher_nll_initial_threshold
                self._teacher_nll_calibration_snapshot = _TeacherNLLCalibrationSnapshot(
                    threshold=None if initial_threshold is None else float(initial_threshold),
                    source="unavailable" if initial_threshold is None else "initial",
                    count=0,
                    version=0,
                )

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
        top_k_strategy = getattr(loss_config, "top_k_strategy", "only_stu")
        if (
            loss_config is not None
            and str(adv_estimator) == "token_reward_direct"
            and top_k_strategy in {"only_stu", "only_stu_teacher_topk"}
        ):
            strategy_topk = loss_config.teacher_topk if top_k_strategy == "only_stu_teacher_topk" else loss_config.topk
            prompt_logprobs = max(prompt_logprobs, int(strategy_topk or 0))
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

    def _collect_training_intervention_outcomes(
        self,
        *,
        validate: bool,
    ) -> bool:
        """Return whether sample-level training outcome observation is enabled."""
        if validate:
            return False
        distillation_config = getattr(self, "distillation_config", None)
        diagnostics_config = getattr(distillation_config, "diagnostics", None)
        if not (
            diagnostics_config is not None
            and getattr(diagnostics_config, "enabled", False)
            and getattr(diagnostics_config, "collect_intervention_metrics", False)
            and getattr(diagnostics_config, "collect_intervention_outcome_transitions", False)
        ):
            return False
        return True

    def _collect_intervention_boundary_metrics(self, *, validate: bool) -> bool:
        """Return whether zero-extra-forward repair-boundary diagnostics are active."""
        if validate:
            return False
        diagnostics_config = getattr(getattr(self, "distillation_config", None), "diagnostics", None)
        return bool(
            diagnostics_config is not None
            and getattr(diagnostics_config, "enabled", False)
            and getattr(diagnostics_config, "collect_intervention_metrics", False)
            and getattr(diagnostics_config, "collect_intervention_boundary_metrics", False)
        )

    def _collect_intervention_boundary_text(self, *, validate: bool) -> bool:
        """Return whether bounded text cases can be attached to the balanced archive."""
        if not AgentLoopWorker._collect_intervention_boundary_metrics(self, validate=validate):
            return False
        diagnostics_config = getattr(self.distillation_config, "diagnostics", None)
        if int(getattr(diagnostics_config, "intervention_boundary_text_samples_per_step", 0) or 0) <= 0:
            return False
        trainer_config = getattr(getattr(self, "config", None), "trainer", None)
        metric_logging = _get_config_value(trainer_config, "metric_logging")
        archive_config = _get_config_value(metric_logging, "archive")
        return bool(
            str(_get_config_value(metric_logging, "mode") or "full").lower() == "balanced"
            and _get_config_value(archive_config, "enabled")
        )

    def _outcome_case_limit(self) -> int:
        diagnostics_config = getattr(getattr(self, "distillation_config", None), "diagnostics", None)
        if diagnostics_config is None:
            return 0
        return int(getattr(diagnostics_config, "outcome_case_samples_per_transition", 0) or 0)

    def _prepare_local_intervention_boundary_cases(self, inputs: list["_InternalAgentLoopOutput"]) -> None:
        """Decode only the worker-local stable top-K repair-boundary cases."""
        diagnostics_config = getattr(getattr(self, "distillation_config", None), "diagnostics", None)
        case_limit = int(getattr(diagnostics_config, "intervention_boundary_text_samples_per_step", 0) or 0)
        context_tokens = int(getattr(diagnostics_config, "intervention_boundary_text_context_tokens", 32) or 32)
        candidates: list[tuple[float, int, int]] = []
        for sample_index, input_item in enumerate(inputs):
            interventions = input_item.extra_fields.get("teacher_interventions") or []
            for intervention_index, intervention in enumerate(interventions):
                if not isinstance(intervention, dict) or not intervention.get("boundary_final_span_available", True):
                    continue
                rank = intervention.get("boundary_case_rank")
                if rank is not None and intervention.get("_teacher_segment_ids") is not None:
                    candidates.append((float(rank), sample_index, intervention_index))

        cases_by_sample: list[list[dict[str, Any]]] = [[] for _ in inputs]
        for _, sample_index, intervention_index in sorted(candidates)[:case_limit]:
            input_item = inputs[sample_index]
            interventions = input_item.extra_fields.get("teacher_interventions") or []
            intervention = interventions[intervention_index]
            response_width = int(input_item.response_ids.shape[-1])
            response_attention = input_item.attention_mask[0, -response_width:].detach().bool().cpu()
            final_length = int(response_attention.sum().item())
            final_ids = input_item.response_ids[0, :final_length].detach().cpu().tolist()
            start = max(int(intervention.get("start", 0) or 0), 0)
            teacher_ids = list(intervention.get("_teacher_segment_ids") or [])
            original_ids = list(intervention.get("_original_student_segment_ids") or [])
            teacher_end = min(start + len(teacher_ids), final_length)
            later_starts = [
                int(item.get("start", final_length) or final_length)
                for item in interventions[intervention_index + 1 :]
                if isinstance(item, dict)
                and item.get("boundary_final_span_available", True)
                and int(item.get("start", final_length) or final_length) >= teacher_end
            ]
            suffix_limit = min(later_starts, default=final_length)

            prefix_tail_ids = final_ids[max(0, start - context_tokens) : start]
            suffix_head_ids = final_ids[teacher_end : min(teacher_end + context_tokens, suffix_limit)]

            def bounded_head_tail(token_ids: list[int]) -> tuple[list[int], list[int], bool]:
                if len(token_ids) <= context_tokens:
                    return token_ids, token_ids, False
                if len(token_ids) <= 2 * context_tokens:
                    return token_ids[:context_tokens], token_ids[context_tokens:], False
                return token_ids[:context_tokens], token_ids[-context_tokens:], True

            original_head_ids, original_tail_ids, original_truncated = bounded_head_tail(original_ids)
            teacher_head_ids, teacher_tail_ids, teacher_truncated = bounded_head_tail(teacher_ids)

            def decode(token_ids: list[int]) -> str:
                return self.tokenizer.decode(token_ids, skip_special_tokens=False)

            prefix_tail = decode(prefix_tail_ids)
            suffix_head = decode(suffix_head_ids)
            original_head = decode(original_head_ids)
            original_tail = decode(original_tail_ids)
            teacher_head = decode(teacher_head_ids)
            teacher_tail = decode(teacher_tail_ids)
            cases_by_sample[sample_index].append(
                {
                    "case_rank": float(intervention["boundary_case_rank"]),
                    "attempt": intervention.get("attempt"),
                    "start": start,
                    "prefix_tail": prefix_tail,
                    "original_student_segment_head": original_head,
                    "original_student_segment_tail": original_tail,
                    "original_student_segment_truncated": original_truncated,
                    "teacher_segment_head": teacher_head,
                    "teacher_segment_tail": teacher_tail,
                    "teacher_segment_truncated": teacher_truncated,
                    "student_suffix_head": suffix_head,
                    "entry_structure_tags": _boundary_structure_tags(prefix_tail, teacher_head),
                    "exit_structure_tags": _boundary_structure_tags(teacher_tail, suffix_head),
                    "context_token_limit": context_tokens,
                    "prefix_tokens_saved": len(prefix_tail_ids),
                    "suffix_tokens_saved": len(suffix_head_ids),
                }
            )

        for input_item, cases in zip(inputs, cases_by_sample, strict=True):
            input_item.extra_fields["teacher_intervention_boundary_cases"] = cases or None

    def _prepare_local_intervention_outcome_cases(self, inputs: list["_InternalAgentLoopOutput"]) -> None:
        """Keep only the local top-K full-text cases per transition.

        The manager performs the same stable-rank reduction after concatenating
        worker outputs, which yields a global cap without transmitting every
        long raw response between processes.
        """
        case_limit = AgentLoopWorker._outcome_case_limit(self)
        candidates: dict[str, list[tuple[float, int]]] = {}
        for index, input_item in enumerate(inputs):
            outcome = input_item.extra_fields.get("teacher_intervention_outcome")
            if not (
                isinstance(outcome, dict)
                and outcome.get("transition_observation_event")
                and outcome.get("repair_executed")
            ):
                continue
            transition = outcome.get("transition")
            rank = outcome.get("case_rank")
            if transition is None or rank is None:
                continue
            candidates.setdefault(str(transition), []).append((float(rank), index))

        selected = {
            index
            for transition_candidates in candidates.values()
            for _, index in sorted(transition_candidates)[:case_limit]
        }
        for index, input_item in enumerate(inputs):
            outcome = input_item.extra_fields.get("teacher_intervention_outcome")
            raw_response_ids = outcome.pop("_raw_response_ids", None) if isinstance(outcome, dict) else None
            interventions = input_item.extra_fields.get("teacher_interventions") or []
            decoded_segments = []
            for intervention in interventions:
                if not isinstance(intervention, dict):
                    continue
                original_ids = intervention.pop("_original_student_segment_ids", None)
                teacher_ids = intervention.pop("_teacher_segment_ids", None)
                if index in selected:
                    decoded_segments.append(
                        {
                            "attempt": intervention.get("attempt"),
                            "start": intervention.get("start"),
                            "end": intervention.get("end"),
                            "original_student_segment": self.tokenizer.decode(
                                original_ids or [], skip_special_tokens=True
                            ),
                            "teacher_segment": self.tokenizer.decode(teacher_ids or [], skip_special_tokens=True),
                        }
                    )
            if not (isinstance(outcome, dict) and outcome.get("transition_observation_event")):
                input_item.extra_fields.pop("teacher_intervention_outcome_case", None)
                continue
            if index not in selected:
                input_item.extra_fields["teacher_intervention_outcome_case"] = None
                continue

            final_length = int(outcome.get("final_response_length", 0) or 0)
            final_response_ids = input_item.response_ids[0, :final_length].detach().cpu().tolist()
            raw_response = outcome.get("raw_response")
            if raw_response is None:
                raw_response = self.tokenizer.decode(raw_response_ids or [], skip_special_tokens=True)
            input_item.extra_fields["teacher_intervention_outcome_case"] = {
                "case_rank": float(outcome["case_rank"]),
                "transition": outcome["transition"],
                "raw_response": raw_response,
                "final_response": self.tokenizer.decode(final_response_ids, skip_special_tokens=True),
                "repair_segments": decoded_segments,
            }

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
        validate = bool(batch.meta_info.get("validate", False))
        repair_for_evaluation = bool(batch.meta_info.get("repair_for_evaluation", False))
        intervention_config = getattr(self, "teacher_intervention_config", None)
        configured_repair_for_evaluation = bool(
            intervention_config is not None and getattr(intervention_config, "repair_for_evaluation", False)
        )
        if repair_for_evaluation and not (validate and configured_repair_for_evaluation):
            raise ValueError(
                "repair_for_evaluation may be requested only for validation/test when "
                "distillation.intervention.repair_for_evaluation=true."
            )
        intervention_active = self.teacher_intervention_enabled and (not validate or repair_for_evaluation)
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs or intervention_active,
        )

        def apply_greedy_sampling_params(params: dict[str, Any]) -> None:
            params["top_p"] = 1.0
            params["top_k"] = -1
            params["temperature"] = 0

        # override sampling params for validation
        if validate:
            evaluation_mode = batch.meta_info.get("evaluation_mode", "val")
            if evaluation_mode not in {"val", "test"}:
                raise ValueError(f"evaluation_mode must be 'val' or 'test', got {evaluation_mode!r}.")
            evaluation_config = config.test_kwargs if evaluation_mode == "test" else config.val_kwargs
            sampling_params["top_p"] = evaluation_config.top_p
            sampling_params["top_k"] = evaluation_config.top_k
            sampling_params["temperature"] = evaluation_config.temperature
            do_sample = batch.meta_info.get("do_sample", evaluation_config.do_sample)
            if not bool(do_sample):
                apply_greedy_sampling_params(sampling_params)
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
                    self._run_agent_loop(
                        sample_sampling_params,
                        trajectory_info[i],
                        trace=trace_this_sample,
                        repair_for_evaluation=repair_for_evaluation,
                        **kwargs,
                    )
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
        repair_for_evaluation: bool = False,
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
                repair_for_evaluation=repair_for_evaluation,
                **kwargs,
            )

    async def _agent_loop_postprocess(
        self,
        output,
        validate,
        sampling_params: Optional[dict[str, Any]] = None,
        repair_for_evaluation: bool = False,
        **kwargs,
    ) -> _InternalAgentLoopOutput:
        """Perform post-processing operations on the output of each individual agent loop."""
        output.extra_fields["raw_prompt"] = kwargs["raw_prompt"]
        if getattr(self, "teacher_intervention_enabled", False):
            skip_correct = bool(
                self.teacher_intervention_config is not None and self.teacher_intervention_config.skip_correct
            )
            outcome_aware_selection = bool(
                self.teacher_intervention_config is not None
                and self.teacher_intervention_config.outcome_aware_candidate_selection_enabled
            )
            collect_training_outcomes = AgentLoopWorker._collect_training_intervention_outcomes(
                self,
                validate=bool(validate),
            )
            observe_outcome = bool(
                skip_correct
                or collect_training_outcomes
                or (outcome_aware_selection and not validate)
                or repair_for_evaluation
            )
            if observe_outcome:
                # The raw student answer must be judged before any token is
                # rewritten. This call is a no-op if a custom AgentLoop already
                # populated the reward.
                await self._compute_score([output], kwargs=kwargs)
                raw_correct, raw_correctness_value = _extract_outcome_correctness(
                    output,
                    self.teacher_intervention_config,
                )
                transition_observation_event = bool(collect_training_outcomes or repair_for_evaluation)
                outcome_metadata = {
                    "enabled": True,
                    "correctness_key": self.teacher_intervention_config.outcome_correctness_key,
                    "correctness_threshold": float(self.teacher_intervention_config.outcome_correctness_threshold),
                    "raw_reward_score": output.reward_score,
                    "raw_correctness_value": raw_correctness_value,
                    "raw_correct": raw_correct,
                    "trigger_detected": False,
                    "repair_executed": False,
                    "skip_correct_enabled": skip_correct,
                }
                if transition_observation_event:
                    outcome_metadata.update(
                        {
                            "raw_reward_extra_info": copy.deepcopy(output.extra_fields.get("reward_extra_info", {})),
                            "raw_response_length": len(output.response_ids),
                            "raw_output_token_sha256": _token_ids_sha256(output.response_ids),
                            "raw_finish_reason": output.extra_fields.get("finish_reason"),
                            "_raw_response_ids": list(output.response_ids),
                            "oracle_gate_used": bool(outcome_aware_selection and repair_for_evaluation),
                            "evaluation_pair": bool(repair_for_evaluation),
                            "transition_observation_event": True,
                        }
                    )
                output.extra_fields["teacher_intervention_outcome"] = outcome_metadata
                if repair_for_evaluation:
                    outcome_metadata["raw_response"] = self.tokenizer.decode(
                        output.response_ids,
                        skip_special_tokens=True,
                    )
            await self._maybe_repair_with_teacher(
                output,
                validate=validate,
                repair_for_evaluation=repair_for_evaluation,
                retain_rejected_candidate_for_outcome=(
                    repair_for_evaluation and self.teacher_intervention_config.confirmation_mode == "future_gap"
                ),
                sampling_params=sampling_params,
                sample_kwargs=kwargs,
            )
            if repair_for_evaluation and self.teacher_intervention_config.confirmation_mode == "future_gap":
                await self._score_rejected_future_gap_candidate(output, kwargs=kwargs)
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
            elif repair_for_evaluation:
                # Evaluation needed rollout logprobs only while detecting and
                # executing repairs. Avoid returning dense student or teacher
                # scoring tensors that the validation aggregator never consumes.
                output.response_logprobs = None
                output.teacher_segment_mask = None
                for scoring_key in (
                    "teacher_ids",
                    "teacher_logprobs",
                    "teacher_sampled_ids",
                    "teacher_sampled_logprobs",
                    "teacher_logprob_nonfinite_count",
                    "teacher_logprob_nonfinite_positions",
                    "teacher_logprob_nonfinite_replacement",
                ):
                    output.extra_fields.pop(scoring_key, None)

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

        teacher_segment_mask = None
        if output.teacher_segment_mask is not None:
            teacher_segment_mask_output = self.tokenizer.pad(
                {"input_ids": output.teacher_segment_mask},
                padding="max_length",
                max_length=self.rollout_config.response_length,
                return_tensors="pt",
                return_attention_mask=False,
            )
            teacher_segment_mask = teacher_segment_mask_output["input_ids"]
            if teacher_segment_mask.dim() == 1:
                teacher_segment_mask = teacher_segment_mask.unsqueeze(0)
            teacher_segment_mask = teacher_segment_mask * response_output["attention_mask"]

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
        outcome_metadata = output.extra_fields.get("teacher_intervention_outcome")
        if isinstance(outcome_metadata, dict):
            final_correct, final_correctness_value = _extract_outcome_correctness(
                output,
                self.teacher_intervention_config,
            )
            intervention_count = int(output.extra_fields.get("teacher_intervention_count", 0) or 0)
            raw_correct = bool(outcome_metadata["raw_correct"])
            repair_executed = intervention_count > 0
            interventions = output.extra_fields.get("teacher_interventions") or []
            outcome_metadata.update(
                {
                    "final_reward_score": output.reward_score,
                    "final_correctness_value": final_correctness_value,
                    "final_correct": final_correct,
                    "repair_executed": repair_executed,
                    "repair_effective": repair_executed and not raw_correct and final_correct,
                    "repair_ineffective": repair_executed and not raw_correct and not final_correct,
                }
            )
            transition = None
            if outcome_metadata.get("transition_observation_event"):
                transition = _outcome_transition(raw_correct, final_correct) if repair_executed else "not_repaired"
                final_output_token_sha256 = _token_ids_sha256(output.response_ids)
                rank_key = ":".join(
                    str(value)
                    for value in (
                        kwargs.get("data_source"),
                        kwargs.get("index"),
                        kwargs.get("global_steps"),
                        transition,
                        outcome_metadata.get("raw_output_token_sha256"),
                        final_output_token_sha256,
                    )
                )
                final_finish_reason = output.extra_fields.get("finish_reason")
                if interventions:
                    last_intervention = interventions[-1]
                    final_finish_reason = (
                        last_intervention.get("student_suffix_finish_reason")
                        or last_intervention.get("teacher_finish_reason")
                        or final_finish_reason
                    )
                outcome_metadata.update(
                    {
                        "final_reward_extra_info": copy.deepcopy(output.extra_fields.get("reward_extra_info", {})),
                        "final_response_length": len(output.response_ids),
                        "final_output_token_sha256": final_output_token_sha256,
                        "final_finish_reason": final_finish_reason,
                        "intervention_count": intervention_count,
                        "transition": transition,
                        "case_rank": _stable_fraction(rank_key),
                    }
                )
            if interventions:
                # Outcome effectiveness is sample-level. Attach it only to the
                # first intervention so aggregation does not double-count
                # samples with multiple repairs.
                interventions[0].update(
                    {
                        "sample_outcome_summary": True,
                        "raw_correctness_value": outcome_metadata["raw_correctness_value"],
                        "raw_correct": raw_correct,
                        "final_correctness_value": final_correctness_value,
                        "final_correct": final_correct,
                        "repair_effective": outcome_metadata["repair_effective"],
                        "repair_ineffective": outcome_metadata["repair_ineffective"],
                    }
                )
                if transition is not None:
                    interventions[0]["transition"] = transition
            proposals = output.extra_fields.get("teacher_intervention_proposals") or []
            for proposal in proposals:
                if not isinstance(proposal, dict):
                    continue
                if proposal.get("accepted"):
                    proposal.update(
                        {
                            "candidate_outcome_available": True,
                            "candidate_reward_score": output.reward_score,
                            "candidate_correctness_value": final_correctness_value,
                            "candidate_correct": final_correct,
                            "candidate_response_length": len(output.response_ids),
                            "candidate_output_token_sha256": _token_ids_sha256(output.response_ids),
                            "candidate_is_final": True,
                            "outcome_scope": (
                                "evaluation_candidate_pair"
                                if outcome_metadata.get("evaluation_pair")
                                else "accepted_observational"
                            ),
                        }
                    )
                candidate_correct = proposal.get("candidate_correct")
                if candidate_correct is None:
                    continue
                proposal.update(
                    {
                        "raw_correctness_value": outcome_metadata["raw_correctness_value"],
                        "raw_correct": raw_correct,
                        "gated_correctness_value": final_correctness_value,
                        "gated_correct": final_correct,
                        "candidate_correctness_delta": int(bool(candidate_correct)) - int(raw_correct),
                        "candidate_transition": _outcome_transition(raw_correct, bool(candidate_correct)),
                    }
                )
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
            teacher_segment_mask=teacher_segment_mask,
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

    async def _score_rejected_future_gap_candidate(
        self,
        output: AgentLoopOutput,
        *,
        kwargs: dict[str, Any],
    ) -> None:
        """Evaluate an already-generated rejected candidate without retaining private token arrays."""
        proposals = output.extra_fields.get("teacher_intervention_proposals") or []
        for proposal in proposals:
            if not isinstance(proposal, dict):
                continue
            candidate_ids = proposal.pop("_candidate_response_ids", None)
            candidate_mask = proposal.pop("_candidate_response_mask", None)
            if candidate_ids is None:
                continue
            if proposal.get("accepted"):
                raise RuntimeError("Accepted future-gap proposals must reuse the final evaluation reward.")
            candidate_ids = list(candidate_ids)
            candidate_mask = list(candidate_mask or [1] * len(candidate_ids))
            safe_extra_fields = {
                key: value
                for key, value in output.extra_fields.items()
                if not str(key).startswith("teacher_intervention") and key != "reward_extra_info"
            }
            candidate_output = AgentLoopOutput(
                prompt_ids=list(output.prompt_ids),
                response_ids=candidate_ids,
                response_mask=candidate_mask,
                response_logprobs=None,
                routed_experts=None,
                multi_modal_data=output.multi_modal_data,
                reward_score=None,
                num_turns=output.num_turns,
                metrics=AgentLoopMetrics(),
                extra_fields=safe_extra_fields,
                mm_processor_kwargs=output.mm_processor_kwargs,
            )
            await self._compute_score([candidate_output], kwargs=kwargs)
            if candidate_output.reward_score is None:
                proposal["candidate_outcome_available"] = False
                proposal["candidate_outcome_unavailable_reason"] = "reward_score_unavailable"
                continue
            candidate_correct, candidate_correctness_value = _extract_outcome_correctness(
                candidate_output,
                self.teacher_intervention_config,
            )
            proposal.update(
                {
                    "candidate_outcome_available": True,
                    "candidate_reward_score": float(candidate_output.reward_score),
                    "candidate_correctness_value": candidate_correctness_value,
                    "candidate_correct": candidate_correct,
                    "candidate_response": self.tokenizer.decode(candidate_ids, skip_special_tokens=True),
                    "candidate_response_length": len(candidate_ids),
                    "candidate_output_token_sha256": _token_ids_sha256(candidate_ids),
                    "outcome_scope": "evaluation_candidate_pair",
                }
            )

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

    def _teacher_nll_calibration_state(self) -> _TeacherNLLCalibrationSnapshot:
        snapshot = getattr(self, "_teacher_nll_calibration_snapshot", None)
        if isinstance(snapshot, _TeacherNLLCalibrationSnapshot):
            return snapshot

        config: DistillationInterventionConfig = self.teacher_intervention_config
        initial_threshold = config.teacher_nll_initial_threshold
        snapshot = _TeacherNLLCalibrationSnapshot(
            threshold=None if initial_threshold is None else float(initial_threshold),
            source="unavailable" if initial_threshold is None else "initial",
            count=0,
            version=0,
        )
        self._teacher_nll_calibration_snapshot = snapshot
        return snapshot

    def set_teacher_nll_calibration_state(self, snapshot: _TeacherNLLCalibrationSnapshot) -> None:
        """Install one manager-computed calibration snapshot for the next rollout batch."""
        if not isinstance(snapshot, _TeacherNLLCalibrationSnapshot):
            raise TypeError(
                "Teacher-NLL calibration broadcasts must use _TeacherNLLCalibrationSnapshot, "
                f"got {type(snapshot).__name__}."
            )
        if snapshot.threshold is not None and (not math.isfinite(snapshot.threshold) or snapshot.threshold < 0):
            raise ValueError(
                "Teacher-NLL calibration threshold must be finite and non-negative when set, "
                f"got {snapshot.threshold!r}."
            )
        if snapshot.count < 0 or snapshot.version < 0:
            raise ValueError(
                "Teacher-NLL calibration count and version must be non-negative, "
                f"got count={snapshot.count}, version={snapshot.version}."
            )
        self._teacher_nll_calibration_snapshot = snapshot

    def drain_teacher_nll_calibration_updates(self) -> list[float]:
        """Return and clear healthy-NLL updates accumulated during the previous rollout batch."""
        updates = list(getattr(self, "_teacher_nll_pending_updates", []))
        self._teacher_nll_pending_updates = []
        return updates

    def _queue_teacher_nll_calibration_update(self, value: float) -> int:
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Healthy teacher-NLL calibration update must be finite and non-negative, got {value!r}.")
        pending = getattr(self, "_teacher_nll_pending_updates", None)
        if not isinstance(pending, list):
            pending = []
            self._teacher_nll_pending_updates = pending
        pending.append(value)
        return len(pending)

    def _current_teacher_nll_threshold(self) -> tuple[Optional[float], str, int]:
        snapshot = AgentLoopWorker._teacher_nll_calibration_state(self)
        return snapshot.threshold, snapshot.source, snapshot.count

    def _teacher_candidate_rpc_limiter(self) -> asyncio.Semaphore:
        """Return the worker-wide limiter shared by all candidate RPC stages."""
        limiter = getattr(self, "_teacher_candidate_rpc_semaphore", None)
        if limiter is None:
            config: DistillationInterventionConfig = self.teacher_intervention_config
            limiter = asyncio.Semaphore(int(config.teacher_candidate_max_concurrency_per_worker))
            self._teacher_candidate_rpc_semaphore = limiter
        return limiter

    @staticmethod
    def _candidate_mean_nll(
        result: TeacherLogprobScoringResult,
        *,
        prefix_len: int,
        candidate_ids: list[int],
        context: str,
    ) -> float:
        if not candidate_ids:
            raise ValueError(f"{context} cannot score an empty candidate.")
        sampled_logprobs = _teacher_response_logprobs(
            teacher_logprobs=_teacher_gap_logprobs(result),
            prompt_len=prefix_len,
            response_len=len(candidate_ids),
        )
        sampled_ids = result.sampled_ids
        if sampled_ids is None and result.ids.ndim == 2 and result.ids.shape[-1] == 1:
            sampled_ids = result.ids
        if sampled_ids is not None:
            start = prefix_len - 1
            end = start + len(candidate_ids)
            aligned_ids = sampled_ids[start:end, 0].detach().cpu().tolist()
            if aligned_ids != candidate_ids:
                raise RuntimeError(
                    f"{context} prompt-logprob token ids are misaligned with the candidate: "
                    f"expected={candidate_ids[:8]}, got={aligned_ids[:8]}, candidate_len={len(candidate_ids)}."
                )
        nll = float((-sampled_logprobs.float()).mean().item())
        if not math.isfinite(nll):
            raise RuntimeError(f"{context} produced a non-finite mean NLL: {nll!r}.")
        return max(nll, 0.0)

    async def _generate_and_select_teacher_candidate(
        self,
        *,
        prefix_ids: list[int],
        max_tokens: int,
        output: AgentLoopOutput,
        routing_key: Optional[str],
        base_request_id: str,
        attempt: int,
        calibration_only: bool,
    ) -> _TeacherCandidateSelectionResult:
        """Generate K candidates, NLL-filter them, then select by teacher/student rank sum."""
        config: DistillationInterventionConfig = self.teacher_intervention_config
        candidate_count = int(config.teacher_candidate_count)
        calibration_snapshot = AgentLoopWorker._teacher_nll_calibration_state(self)
        threshold = calibration_snapshot.threshold
        threshold_source = calibration_snapshot.source
        calibration_count_before = calibration_snapshot.count
        candidate_rpc_limiter = AgentLoopWorker._teacher_candidate_rpc_limiter(self)

        async def generate_candidate(candidate_index: int) -> TeacherGenerationResult:
            async with candidate_rpc_limiter:
                return await self.teacher_server_manager.generate_teacher_segment_single(
                    prefix_ids=prefix_ids,
                    max_tokens=max_tokens,
                    multi_modal_data=output.multi_modal_data,
                    mm_processor_kwargs=output.mm_processor_kwargs,
                    routing_key=routing_key,
                    request_id=f"{base_request_id}:teacher-candidate:{attempt}:{candidate_index}",
                )

        generation_results = await asyncio.gather(
            *(generate_candidate(candidate_index) for candidate_index in range(candidate_count)),
            return_exceptions=True,
        )

        eos_token_ids = _tokenizer_eos_token_ids(self.tokenizer)
        candidates: list[_TeacherRepairCandidate] = []
        generation_failures: dict[int, str] = {}
        for candidate_index, raw_result in enumerate(generation_results):
            if isinstance(raw_result, BaseException):
                generation_failures[candidate_index] = f"generation_exception:{type(raw_result).__name__}"
                logger.warning(
                    "Teacher candidate generation failed for request_id=%s, attempt=%d, candidate=%d: %s",
                    base_request_id,
                    attempt,
                    candidate_index,
                    raw_result,
                )
                continue
            generation = _coerce_teacher_generation_result(raw_result)
            candidate = _TeacherRepairCandidate(
                index=candidate_index,
                generation=generation,
                token_ids=[],
                eos_truncated=False,
            )
            if _generation_was_aborted(
                stop_reason=generation.stop_reason,
                finish_reason=generation.finish_reason,
            ):
                candidate.invalid_reason = "generation_aborted"
            else:
                candidate.token_ids, _, candidate.eos_truncated = _truncate_generation_at_eos(
                    list(generation.token_ids),
                    None,
                    eos_token_ids=eos_token_ids,
                )
                if len(candidate.token_ids) > max_tokens:
                    candidate.invalid_reason = "generation_exceeded_token_budget"
                elif not candidate.token_ids:
                    candidate.invalid_reason = "generation_empty"
                elif _invalid_token_ids_for_tokenizer(candidate.token_ids, self.tokenizer):
                    candidate.invalid_reason = "generation_invalid_token_ids"
            candidates.append(candidate)

        async def score_teacher_nll(candidate: _TeacherRepairCandidate) -> None:
            if candidate.invalid_reason is not None:
                return
            try:
                async with candidate_rpc_limiter:
                    teacher_result = await self._score_sequence_for_intervention(
                        request_id=f"{base_request_id}:teacher-candidate-score:{attempt}:{candidate.index}",
                        prompt_ids=prefix_ids,
                        response_ids=candidate.token_ids,
                        output=output,
                        routing_key=routing_key,
                    )
                candidate.teacher_nll = AgentLoopWorker._candidate_mean_nll(
                    teacher_result,
                    prefix_len=len(prefix_ids),
                    candidate_ids=candidate.token_ids,
                    context=f"Teacher candidate {candidate.index}",
                )
            except Exception as exc:
                candidate.invalid_reason = f"teacher_nll_scoring_failed:{type(exc).__name__}"
                logger.warning(
                    "Teacher candidate NLL scoring failed for request_id=%s, attempt=%d, candidate=%d: %s",
                    base_request_id,
                    attempt,
                    candidate.index,
                    exc,
                )

        await asyncio.gather(*(score_teacher_nll(candidate) for candidate in candidates))
        teacher_scored_candidates = [
            candidate
            for candidate in candidates
            if candidate.invalid_reason is None and candidate.teacher_nll is not None
        ]
        unique_candidate_count = len({tuple(candidate.token_ids) for candidate in teacher_scored_candidates})

        metadata: dict[str, Any] = {
            "candidate_selection_enabled": True,
            "candidate_count_requested": candidate_count,
            "candidate_rpc_max_concurrency_per_worker": int(config.teacher_candidate_max_concurrency_per_worker),
            "candidate_count_generated": candidate_count - len(generation_failures),
            "candidate_count_teacher_nll_valid": len(teacher_scored_candidates),
            "candidate_unique_count": unique_candidate_count,
            "candidate_duplicate_fraction": (
                1.0 - unique_candidate_count / len(teacher_scored_candidates) if teacher_scored_candidates else 0.0
            ),
            "teacher_nll_threshold": threshold,
            "teacher_nll_threshold_source": threshold_source,
            "teacher_nll_calibration_quantile": float(config.teacher_nll_calibration_quantile),
            "teacher_nll_calibration_count_before": calibration_count_before,
            "teacher_nll_calibration_scope": "global",
            "teacher_nll_calibration_version": calibration_snapshot.version,
            "candidate_generation_failures": generation_failures,
        }

        if calibration_only:
            if teacher_scored_candidates:
                healthy_min_nll = min(candidate.teacher_nll for candidate in teacher_scored_candidates)
                pending_count = AgentLoopWorker._queue_teacher_nll_calibration_update(
                    self,
                    healthy_min_nll,
                )
                metadata.update(
                    {
                        "healthy_calibration_updated": True,
                        "healthy_calibration_min_teacher_nll": float(healthy_min_nll),
                        "teacher_nll_calibration_count_after": calibration_count_before,
                        "teacher_nll_calibration_pending_count_after": pending_count,
                        "teacher_nll_global_update_pending": True,
                        "teacher_nll_threshold_after": threshold,
                        "teacher_nll_threshold_source_after": threshold_source,
                    }
                )
            else:
                metadata.update(
                    {
                        "healthy_calibration_updated": False,
                        "teacher_nll_calibration_count_after": calibration_count_before,
                        "teacher_nll_calibration_pending_count_after": len(
                            getattr(self, "_teacher_nll_pending_updates", [])
                        ),
                        "teacher_nll_global_update_pending": False,
                    }
                )
            metadata["teacher_candidates"] = [
                {
                    "index": candidate.index,
                    "token_count": len(candidate.token_ids),
                    "teacher_nll": candidate.teacher_nll,
                    "invalid_reason": candidate.invalid_reason,
                }
                for candidate in candidates
            ]
            return _TeacherCandidateSelectionResult(candidate=None, metadata=metadata)

        if not teacher_scored_candidates:
            metadata["candidate_count_threshold_eligible"] = 0
            metadata["teacher_candidates"] = [
                {
                    "index": candidate.index,
                    "token_count": len(candidate.token_ids),
                    "teacher_nll": candidate.teacher_nll,
                    "invalid_reason": candidate.invalid_reason,
                }
                for candidate in candidates
            ]
            return _TeacherCandidateSelectionResult(
                candidate=None,
                metadata=metadata,
                skip_reason="teacher_candidates_all_invalid",
            )
        if threshold is None:
            metadata["candidate_count_threshold_eligible"] = 0
            metadata["teacher_candidates"] = [
                {
                    "index": candidate.index,
                    "token_count": len(candidate.token_ids),
                    "teacher_nll": candidate.teacher_nll,
                    "invalid_reason": candidate.invalid_reason,
                }
                for candidate in candidates
            ]
            return _TeacherCandidateSelectionResult(
                candidate=None,
                metadata=metadata,
                skip_reason="teacher_nll_calibration_not_ready",
            )

        threshold_eligible = [
            candidate for candidate in teacher_scored_candidates if candidate.teacher_nll <= threshold
        ]
        metadata["candidate_count_threshold_eligible"] = len(threshold_eligible)
        if not threshold_eligible:
            metadata["teacher_candidates"] = [
                {
                    "index": candidate.index,
                    "token_count": len(candidate.token_ids),
                    "teacher_nll": candidate.teacher_nll,
                    "invalid_reason": candidate.invalid_reason,
                }
                for candidate in candidates
            ]
            return _TeacherCandidateSelectionResult(
                candidate=None,
                metadata=metadata,
                skip_reason="teacher_candidates_above_nll_threshold",
            )

        async def score_student_nll(candidate: _TeacherRepairCandidate) -> None:
            try:
                async with candidate_rpc_limiter:
                    student_result = await self._score_student_sequence_for_intervention(
                        request_id=f"{base_request_id}:student-candidate-score:{attempt}:{candidate.index}",
                        prompt_ids=prefix_ids,
                        response_ids=candidate.token_ids,
                        output=output,
                    )
                candidate.student_nll = AgentLoopWorker._candidate_mean_nll(
                    student_result,
                    prefix_len=len(prefix_ids),
                    candidate_ids=candidate.token_ids,
                    context=f"Student candidate {candidate.index}",
                )
            except Exception as exc:
                candidate.invalid_reason = f"student_nll_scoring_failed:{type(exc).__name__}"
                logger.warning(
                    "Student candidate NLL scoring failed for request_id=%s, attempt=%d, candidate=%d: %s",
                    base_request_id,
                    attempt,
                    candidate.index,
                    exc,
                )

        await asyncio.gather(*(score_student_nll(candidate) for candidate in threshold_eligible))
        rankable = [
            candidate
            for candidate in threshold_eligible
            if candidate.invalid_reason is None and candidate.student_nll is not None
        ]
        metadata["candidate_count_rankable"] = len(rankable)
        if not rankable:
            metadata["teacher_candidates"] = [
                {
                    "index": candidate.index,
                    "token_count": len(candidate.token_ids),
                    "teacher_nll": candidate.teacher_nll,
                    "student_nll": candidate.student_nll,
                    "invalid_reason": candidate.invalid_reason,
                }
                for candidate in candidates
            ]
            return _TeacherCandidateSelectionResult(
                candidate=None,
                metadata=metadata,
                skip_reason="teacher_candidates_student_nll_invalid",
            )

        teacher_ranks = _rank_min({candidate.index: float(candidate.teacher_nll) for candidate in rankable})
        student_ranks = _rank_min({candidate.index: float(candidate.student_nll) for candidate in rankable})
        for candidate in rankable:
            candidate.teacher_rank = teacher_ranks[candidate.index]
            candidate.student_rank = student_ranks[candidate.index]
            candidate.rank_sum = candidate.teacher_rank + candidate.student_rank
        best_rank_sum = min(candidate.rank_sum for candidate in rankable)
        tied_indices = [candidate.index for candidate in rankable if candidate.rank_sum == best_rank_sum]
        selected_index = _stable_choice(
            tied_indices,
            key=f"{base_request_id}:teacher-candidate-tie:{attempt}",
        )
        selected_candidate = next(candidate for candidate in rankable if candidate.index == selected_index)
        metadata.update(
            {
                "selected_candidate_index": selected_index,
                "selected_teacher_nll": selected_candidate.teacher_nll,
                "selected_student_nll": selected_candidate.student_nll,
                "selected_teacher_rank": selected_candidate.teacher_rank,
                "selected_student_rank": selected_candidate.student_rank,
                "selected_rank_sum": selected_candidate.rank_sum,
                "selected_tie_count": len(tied_indices),
                "teacher_candidates": [
                    {
                        "index": candidate.index,
                        "token_count": len(candidate.token_ids),
                        "teacher_nll": candidate.teacher_nll,
                        "student_nll": candidate.student_nll,
                        "teacher_rank": candidate.teacher_rank,
                        "student_rank": candidate.student_rank,
                        "rank_sum": candidate.rank_sum,
                        "threshold_eligible": (
                            candidate.teacher_nll is not None and candidate.teacher_nll <= threshold
                        ),
                        "invalid_reason": candidate.invalid_reason,
                    }
                    for candidate in candidates
                ],
            }
        )
        return _TeacherCandidateSelectionResult(candidate=selected_candidate, metadata=metadata)

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
        nonfinite_logprob_offsets: tuple[int, ...] = ()
        nonfinite_logprob_replacement = None
        if suffix_logprobs:
            suffix_logprobs_tensor = torch.tensor(suffix_logprobs, dtype=torch.float32)
            finite_mask = torch.isfinite(suffix_logprobs_tensor)
            if not finite_mask.all():
                nonfinite_logprob_offsets = tuple(
                    int(offset) for offset in torch.nonzero(~finite_mask, as_tuple=False).flatten().tolist()
                )
                nonfinite_logprob_replacement = _logprob_floor_from_distillation_config(self.distillation_config)
                suffix_logprobs_tensor = torch.nan_to_num(
                    suffix_logprobs_tensor,
                    nan=nonfinite_logprob_replacement,
                    neginf=nonfinite_logprob_replacement,
                    posinf=0.0,
                )
                if not torch.isfinite(suffix_logprobs_tensor).all():
                    raise RuntimeError(
                        "Student suffix logprob sanitization failed to produce finite values for "
                        f"request_id={request_id!r}."
                    )
                suffix_logprobs = suffix_logprobs_tensor.tolist()
                warning_count = getattr(self, "_student_suffix_nonfinite_logprob_warnings", 0)
                if warning_count < MAX_NONFINITE_STUDENT_SUFFIX_LOGPROB_WARNINGS:
                    logger.warning(
                        "Student suffix generation returned NaN/Inf logprobs at %d token positions for "
                        "request_id=%r, suffix_len=%d; replaced them with finite logprobs and excluded those "
                        "positions from subsequent intervention-gap detection.",
                        len(nonfinite_logprob_offsets),
                        request_id,
                        len(suffix_ids),
                    )
                    self._student_suffix_nonfinite_logprob_warnings = warning_count + 1
        finish_reason = (student_output.extra_fields or {}).get("finish_reason")
        suffix_ids, suffix_logprobs, eos_truncated = _truncate_generation_at_eos(
            suffix_ids,
            suffix_logprobs,
            eos_token_ids=_tokenizer_eos_token_ids(self.tokenizer),
        )
        nonfinite_logprob_offsets = tuple(offset for offset in nonfinite_logprob_offsets if offset < len(suffix_ids))
        if not nonfinite_logprob_offsets:
            nonfinite_logprob_replacement = None
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
            nonfinite_logprob_offsets=nonfinite_logprob_offsets,
            nonfinite_logprob_replacement=nonfinite_logprob_replacement,
        )

    async def _maybe_repair_with_teacher(
        self,
        output: AgentLoopOutput,
        *,
        validate: bool,
        repair_for_evaluation: bool = False,
        retain_rejected_candidate_for_outcome: bool = False,
        sampling_params: Optional[dict[str, Any]],
        sample_kwargs: Optional[dict[str, Any]],
    ) -> None:
        """Repair high-gap response segments with teacher generation before padding/training."""
        if not self.teacher_intervention_enabled or (validate and not repair_for_evaluation):
            return
        if repair_for_evaluation and not validate:
            raise ValueError("repair_for_evaluation is valid only when validate=true.")

        AgentLoopWorker._maybe_reset_teacher_intervention_log_counter(self, sample_kwargs)
        AgentLoopWorker._ensure_teacher_intervention_token_patterns(self)
        output.extra_fields["teacher_intervention_count"] = 0
        output.extra_fields["teacher_interventions"] = []
        output.extra_fields["teacher_intervention_skipped"] = []
        output.extra_fields["teacher_intervention_gap_diagnostics"] = []
        interventions: list[dict[str, Any]] = output.extra_fields["teacher_interventions"]
        skipped_interventions: list[dict[str, Any]] = output.extra_fields["teacher_intervention_skipped"]
        gap_diagnostics: list[dict[str, Any]] = output.extra_fields["teacher_intervention_gap_diagnostics"]

        intervention_config: DistillationInterventionConfig = self.teacher_intervention_config
        future_gap_confirmation = intervention_config.confirmation_mode == "future_gap"
        proposals: list[dict[str, Any]] = []
        if future_gap_confirmation:
            output.extra_fields["teacher_intervention_proposals"] = proposals
        outcome_metadata = output.extra_fields.get("teacher_intervention_outcome")
        if intervention_config.skip_correct:
            if not isinstance(outcome_metadata, dict) or "raw_correct" not in outcome_metadata:
                raise RuntimeError(
                    "skip_correct requires the raw on-policy reward to be computed before teacher intervention. "
                    "Call _agent_loop_postprocess instead of invoking _maybe_repair_with_teacher directly."
                )
            if bool(outcome_metadata["raw_correct"]):
                response_length = min(len(output.response_ids), self.rollout_config.response_length)
                if (
                    intervention_config.train_on_teacher_segments
                    and intervention_config.teacher_segment_loss_mode == "sft"
                ):
                    output.teacher_segment_mask = [0] * response_length
                _record_skipped_intervention(
                    skipped_interventions,
                    reason="skip_correct",
                    attempt=0,
                    start=0,
                    end=0,
                    candidate_response_ids=list(output.response_ids[:response_length]),
                    candidate_response_mask=list(output.response_mask[:response_length]),
                )
                outcome_metadata.update(
                    {
                        "skip_correct_applied": True,
                        "skip_reason": "skip_correct",
                        "trigger_detected": False,
                        "repair_executed": False,
                    }
                )
                return

        if output.routed_experts is not None:
            raise NotImplementedError(
                "distillation.intervention.enabled cannot repair outputs that already contain routed_experts. "
                "Repaired tokens would need fresh router decisions; disable intervention or routing replay."
            )

        outcome_aware_selection = bool(intervention_config.outcome_aware_candidate_selection_enabled)
        collect_boundary_metrics = AgentLoopWorker._collect_intervention_boundary_metrics(self, validate=validate)
        collect_boundary_text = AgentLoopWorker._collect_intervention_boundary_text(self, validate=validate)
        if outcome_aware_selection and not isinstance(outcome_metadata, dict):
            raise RuntimeError(
                "Outcome-aware teacher intervention requires the raw outcome gate to run before repair. "
                "Call _agent_loop_postprocess instead of invoking _maybe_repair_with_teacher directly, or provide "
                "teacher_intervention_outcome metadata in a focused test."
            )
        use_teacher_segment_sft = bool(
            intervention_config.train_on_teacher_segments and intervention_config.teacher_segment_loss_mode == "sft"
        )
        min_trainable_tokens = int(getattr(intervention_config, "min_trainable_tokens_after_repair", 1))
        response_ids = list(output.response_ids[: self.rollout_config.response_length])
        teacher_segment_mask = [0] * len(response_ids) if use_teacher_segment_sft else None
        output.teacher_segment_mask = teacher_segment_mask
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
            teacher_segment_mask=teacher_segment_mask,
        )

        prompt_ids = list(output.prompt_ids)
        if not prompt_ids:
            raise RuntimeError("Teacher intervention requires a non-empty prompt_ids list.")
        routing_key = self._teacher_routing_key(sample_kwargs)
        base_request_id = output.extra_fields.get("rollout_request_id") or uuid4().hex
        cached_teacher_result = None
        eos_token_ids = _tokenizer_eos_token_ids(self.tokenizer)
        pending_local_effect: Optional[tuple[dict[str, Any], int]] = None
        pending_future_gap: Optional[_PendingFutureGapConfirmation] = None

        for attempt in range(intervention_config.max_interventions_per_sample + 1):
            teacher_result = await self._score_sequence_for_intervention(
                request_id=f"{base_request_id}:teacher-score:{attempt}",
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                output=output,
                routing_key=routing_key,
            )
            cached_teacher_result = teacher_result
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
            distribution_scores = None
            if _uses_intervention_distribution_gap(intervention_config.gap_metric):
                student_result = await self._score_student_sequence_for_intervention(
                    request_id=f"{base_request_id}:student-score:{attempt}",
                    prompt_ids=prompt_ids,
                    response_ids=response_ids,
                    output=output,
                )
                distribution_scores = _compute_distribution_intervention_gap(
                    student_result=student_result,
                    teacher_result=teacher_result,
                    prompt_len=len(prompt_ids),
                    response_len=len(response_ids),
                    metric=intervention_config.gap_metric,
                    tokenizer=self.tokenizer,
                    log_prob_floor=_logprob_floor_from_distillation_config(self.distillation_config),
                )
            score_result = _compose_intervention_score(
                student_logprobs=student_response_logprobs,
                teacher_logprobs=teacher_response_logprobs,
                metric=intervention_config.gap_metric,
                teacher_rejection_token_cap=float(getattr(intervention_config, "teacher_rejection_token_cap", 10.0)),
                distribution_scores=distribution_scores,
            )
            teacher_nonfinite_positions = _response_relative_nonfinite_positions(
                teacher_result,
                prompt_len=len(prompt_ids),
                response_len=len(response_ids),
            )
            if pending_local_effect is not None:
                pending_metadata, after_intervention_start = pending_local_effect
                _finalize_local_effect_metadata(
                    metadata=pending_metadata,
                    score_result=score_result,
                    response_mask=gap_mask,
                    after_intervention_start=after_intervention_start,
                    segment_length=intervention_config.student_segment_length,
                    gap_aggregation=intervention_config.gap_aggregation,
                    teacher_nonfinite_positions=teacher_nonfinite_positions,
                )
                pending_local_effect = None
            gaps = score_result.active_scores
            bad_segment = _select_intervention_segment(gaps, gap_mask, intervention_config)
            if validate and repair_for_evaluation:
                # Keep standalone/evaluation diagnostics device-agnostic. In
                # particular, torch.quantile need not be implemented by the
                # NPU backend because this tiny summary is computed on CPU.
                valid_mask = torch.tensor(gap_mask, dtype=torch.bool)
                valid_gaps = gaps.detach().float().cpu()[valid_mask]
                finite_gaps = valid_gaps[torch.isfinite(valid_gaps)]
                gap_diagnostic = {
                    "attempt": attempt,
                    "gap_metric": str(intervention_config.gap_metric).lower(),
                    "gap_threshold": float(intervention_config.gap_threshold),
                    "gap_aggregation": str(intervention_config.gap_aggregation),
                    "valid_token_count": int(valid_gaps.numel()),
                    "finite_token_count": int(finite_gaps.numel()),
                    "mean": float(finite_gaps.mean().item()) if finite_gaps.numel() else None,
                    "max": float(finite_gaps.max().item()) if finite_gaps.numel() else None,
                    "p95": float(torch.quantile(finite_gaps, 0.95).item()) if finite_gaps.numel() else None,
                    "selected_segment_start": None if bad_segment is None else int(bad_segment[0]),
                    "selected_segment_end": None if bad_segment is None else int(bad_segment[1]),
                    "selected_segment_score": None if bad_segment is None else float(bad_segment[2]),
                    "teacher_nonfinite_token_count": len(teacher_nonfinite_positions),
                }
                gap_diagnostics.append(gap_diagnostic)
            if pending_future_gap is not None:
                pending = pending_future_gap
                repaired_values, repaired_invalid_counts = _collect_future_gap_values(
                    scores=score_result.active_scores,
                    response_mask=gap_mask,
                    start=pending.after_intervention_start,
                    window_tokens=intervention_config.future_gap_window_tokens,
                    teacher_nonfinite_positions=teacher_nonfinite_positions,
                )
                proposal = pending.proposal
                proposal["future_gap_repaired_available_tokens"] = len(repaired_values)
                proposal["future_gap_repaired_invalid_counts"] = repaired_invalid_counts
                accepted = _evaluate_future_gap_confirmation(
                    proposal=proposal,
                    baseline_values=pending.baseline_values,
                    repaired_values=repaired_values,
                    intervention_config=intervention_config,
                )
                proposal["future_gap_confirmation_accepted"] = accepted
                if accepted:
                    confirmation_terminal = bool(proposal.get("future_gap_confirmation_terminal", False))
                    continuation_budget = 0
                    combined_suffix_result = pending.confirmation_suffix_result
                    if not confirmation_terminal:
                        continuation_budget = max(self.rollout_config.response_length - len(response_ids), 0)
                    proposal["future_gap_continuation_requested_tokens"] = continuation_budget
                    if continuation_budget > 0:
                        continuation_result = await self._generate_student_suffix(
                            request_id=(
                                f"{base_request_id}:future-gap-continue:{int(proposal.get('attempt', attempt - 1))}"
                            ),
                            prefix_ids=prompt_ids + response_ids,
                            max_tokens=continuation_budget,
                            sampling_params=sampling_params,
                            output=output,
                        )
                        continuation_gap_mask = [1] * len(continuation_result.token_ids)
                        for suffix_offset in continuation_result.nonfinite_logprob_offsets:
                            continuation_gap_mask[suffix_offset] = 0
                        response_ids += continuation_result.token_ids
                        response_mask += [1] * len(continuation_result.token_ids)
                        response_logprobs += continuation_result.logprobs
                        gap_mask += continuation_gap_mask
                        if teacher_segment_mask is not None:
                            teacher_segment_mask += [0] * len(continuation_result.token_ids)
                        combined_suffix_result = _merge_student_suffix_results(
                            pending.confirmation_suffix_result,
                            continuation_result,
                        )
                        cached_teacher_result = None
                        _validate_intervention_state(
                            context="after accepted future-gap continuation",
                            response_ids=response_ids,
                            response_mask=response_mask,
                            response_logprobs=response_logprobs,
                            gap_mask=gap_mask,
                            response_length_limit=self.rollout_config.response_length,
                            teacher_segment_mask=teacher_segment_mask,
                        )
                        final_thinking_state = AgentLoopWorker._teacher_intervention_thinking_state(
                            self,
                            response_ids,
                            prompt_ids=prompt_ids,
                        )
                        final_terminal = continuation_result.terminal
                        proposal.update(
                            {
                                "future_gap_continuation_generated_tokens": len(continuation_result.token_ids),
                                "future_gap_final_candidate_complete": (
                                    final_terminal or len(response_ids) >= self.rollout_config.response_length
                                ),
                                "thinking_state_valid": final_thinking_state.valid,
                                "thinking_state_open": final_thinking_state.open,
                                "thinking_state_reason": final_thinking_state.reason,
                            }
                        )
                        if not final_thinking_state.valid:
                            accepted = False
                            proposal.update(
                                {
                                    "accepted": False,
                                    "status": "rejected",
                                    "rejection_stage": "post_confirmation_continuation",
                                    "reason": "invalid_thinking_block",
                                }
                            )
                        elif (
                            final_terminal
                            and final_thinking_state.open
                            and getattr(intervention_config, "require_closed_thinking_on_terminal", True)
                        ):
                            accepted = False
                            proposal.update(
                                {
                                    "accepted": False,
                                    "status": "rejected",
                                    "rejection_stage": "post_confirmation_continuation",
                                    "reason": "terminal_open_thinking_block",
                                }
                            )
                    else:
                        final_terminal = confirmation_terminal
                        proposal["future_gap_final_candidate_complete"] = (
                            final_terminal or len(response_ids) >= self.rollout_config.response_length
                        )

                    combined_logprob_metadata = _student_suffix_logprob_metadata(
                        combined_suffix_result,
                        response_start=pending.after_intervention_start,
                    )
                    final_candidate_metadata = {
                        "student_suffix_tokens": len(combined_suffix_result.token_ids),
                        "student_suffix_sha256": _token_ids_sha256(combined_suffix_result.token_ids),
                        "student_suffix_stop_reason": combined_suffix_result.stop_reason,
                        "student_suffix_finish_reason": combined_suffix_result.finish_reason,
                        "candidate_response_tokens": len(response_ids),
                        "candidate_response_sha256": _token_ids_sha256(response_ids),
                        "candidate_terminal": final_terminal,
                        **combined_logprob_metadata,
                    }
                    proposal.update(final_candidate_metadata)
                    pending.intervention.update(
                        {
                            "student_suffix_tokens": len(combined_suffix_result.token_ids),
                            "student_suffix_stop_reason": combined_suffix_result.stop_reason,
                            "student_suffix_finish_reason": combined_suffix_result.finish_reason,
                            "terminated": final_terminal,
                            **combined_logprob_metadata,
                        }
                    )
                    candidate_trainable_tokens = sum(response_mask)
                    if teacher_segment_mask is not None:
                        candidate_trainable_tokens += sum(teacher_segment_mask)
                    if accepted and candidate_trainable_tokens < min_trainable_tokens:
                        accepted = False
                        proposal.update(
                            {
                                "accepted": False,
                                "status": "rejected",
                                "rejection_stage": "post_confirmation_candidate_validation",
                                "reason": "insufficient_trainable_tokens",
                            }
                        )

                if accepted:
                    interventions.append(pending.intervention)
                    if isinstance(outcome_metadata, dict):
                        outcome_metadata["repair_executed"] = True
                    proposal["intervention_index"] = len(interventions) - 1
                else:
                    if validate and repair_for_evaluation and retain_rejected_candidate_for_outcome:
                        outcome_candidate_ids = response_ids
                        outcome_candidate_mask = response_mask
                        outcome_candidate_complete = bool(proposal.get("future_gap_final_candidate_complete", False))
                        diagnostic_continuation_budget = 0
                        if not outcome_candidate_complete:
                            diagnostic_continuation_budget = max(
                                self.rollout_config.response_length - len(response_ids),
                                0,
                            )
                        proposal["future_gap_rejected_eval_continuation_requested_tokens"] = (
                            diagnostic_continuation_budget
                        )
                        if diagnostic_continuation_budget > 0:
                            diagnostic_continuation = await self._generate_student_suffix(
                                request_id=(
                                    f"{base_request_id}:future-gap-rejected-eval-continue:"
                                    f"{int(proposal.get('attempt', attempt - 1))}"
                                ),
                                prefix_ids=prompt_ids + response_ids,
                                max_tokens=diagnostic_continuation_budget,
                                sampling_params=sampling_params,
                                output=output,
                            )
                            outcome_candidate_ids = response_ids + diagnostic_continuation.token_ids
                            outcome_candidate_mask = response_mask + [1] * len(diagnostic_continuation.token_ids)
                            outcome_candidate_complete = (
                                diagnostic_continuation.terminal
                                or len(outcome_candidate_ids) >= self.rollout_config.response_length
                            )
                            proposal["future_gap_rejected_eval_continuation_generated_tokens"] = len(
                                diagnostic_continuation.token_ids
                            )
                        if outcome_candidate_complete:
                            # These references are consumed and removed before
                            # the worker output is padded or transferred.
                            proposal["_candidate_response_ids"] = outcome_candidate_ids
                            proposal["_candidate_response_mask"] = outcome_candidate_mask
                        else:
                            proposal["candidate_outcome_available"] = False
                            proposal["candidate_outcome_unavailable_reason"] = (
                                "short_horizon_counterfactual_not_completed"
                            )
                    _record_skipped_intervention(
                        skipped_interventions,
                        reason=str(proposal.get("reason") or "future_gap_rejected"),
                        attempt=int(proposal.get("attempt", attempt - 1)),
                        start=int(proposal.get("start", 0)),
                        end=int(proposal.get("end", 0)),
                        candidate_response_ids=response_ids,
                        candidate_response_mask=response_mask,
                        teacher_tokens=int(proposal.get("teacher_tokens", 0)),
                        student_suffix_tokens=int(proposal.get("student_suffix_tokens", 0)),
                    )
                    response_ids = pending.raw_response_ids
                    response_mask = pending.raw_response_mask
                    response_logprobs = pending.raw_response_logprobs
                    gap_mask = pending.raw_gap_mask
                    teacher_segment_mask = pending.raw_teacher_segment_mask
                    cached_teacher_result = pending.raw_teacher_result
                    if isinstance(outcome_metadata, dict):
                        outcome_metadata["repair_executed"] = False
                        outcome_metadata["future_gap_rejected"] = True
                pending_future_gap = None
                break
            if bad_segment is None or len(interventions) >= intervention_config.max_interventions_per_sample:
                break

            start, end, score = bad_segment
            if isinstance(outcome_metadata, dict):
                outcome_metadata["trigger_detected"] = True
                outcome_metadata["trigger_count"] = int(outcome_metadata.get("trigger_count", 0)) + 1
            persistence_metadata: dict[str, Any] = {}
            if bool(getattr(intervention_config, "teacher_rejection_persistence_enabled", False)):
                persistence_score, persistence_tokens = _compute_teacher_rejection_persistence(
                    gaps=gaps,
                    response_mask=gap_mask,
                    candidate_end=end,
                    intervention_config=intervention_config,
                )
                if persistence_score is None:
                    raise RuntimeError(
                        "Teacher intervention selected a segment without enough valid persistence tokens."
                    )
                persistence_metadata = {
                    "teacher_rejection_persistence_score": persistence_score,
                    "teacher_rejection_persistence_tokens": persistence_tokens,
                    "teacher_rejection_persistence_window_tokens": (
                        intervention_config.teacher_rejection_persistence_window_tokens
                    ),
                    "teacher_rejection_persistence_threshold": _teacher_rejection_persistence_threshold(
                        intervention_config
                    ),
                }
            start, end = AgentLoopWorker._expand_teacher_intervention_segment_for_token_patterns(
                self,
                response_ids,
                start,
                end,
            )
            future_gap_proposal: Optional[dict[str, Any]] = None
            future_gap_baseline_values: list[float] = []
            if future_gap_confirmation:
                future_gap_baseline_values, baseline_invalid_counts = _collect_future_gap_values(
                    scores=score_result.active_scores,
                    response_mask=gap_mask,
                    start=end,
                    window_tokens=intervention_config.future_gap_window_tokens,
                    teacher_nonfinite_positions=teacher_nonfinite_positions,
                )
                future_gap_proposal = {
                    "schema_version": 2,
                    "confirmation_mode": "future_gap",
                    "global_step": _metadata_scalar((sample_kwargs or {}).get("global_steps")),
                    "uid": _metadata_scalar((sample_kwargs or {}).get("uid")),
                    "sample_index": _metadata_scalar((sample_kwargs or {}).get("index")),
                    "data_source": _metadata_scalar((sample_kwargs or {}).get("data_source")),
                    "attempt": attempt,
                    "start": start,
                    "end": end,
                    "start_fraction": start / max(len(response_ids), 1),
                    "trigger_gap_metric": str(intervention_config.gap_metric).lower(),
                    "trigger_gap_aggregation": str(intervention_config.gap_aggregation),
                    "trigger_score": float(score),
                    "trigger_threshold": float(intervention_config.gap_threshold),
                    "future_gap_window_tokens": int(intervention_config.future_gap_window_tokens),
                    "future_gap_min_valid_tokens": int(intervention_config.future_gap_min_valid_tokens),
                    "future_gap_max_threshold": _future_gap_threshold(intervention_config),
                    "future_gap_min_reduction": float(intervention_config.future_gap_min_reduction),
                    "future_gap_baseline_available_tokens": len(future_gap_baseline_values),
                    "future_gap_baseline_invalid_counts": baseline_invalid_counts,
                    "skip_correct": bool(intervention_config.skip_correct),
                    "train_on_teacher_segments": bool(intervention_config.train_on_teacher_segments),
                    "teacher_segment_loss_mode": str(intervention_config.teacher_segment_loss_mode),
                    "accepted": False,
                    "status": "triggered",
                    **persistence_metadata,
                }
                proposals.append(future_gap_proposal)
            local_effect_metadata = _initialize_local_effect_metadata(
                score_result=score_result,
                response_mask=gap_mask,
                no_intervention_start=end,
                segment_length=intervention_config.student_segment_length,
                gap_metric=intervention_config.gap_metric,
                gap_aggregation=intervention_config.gap_aggregation,
                collect_v2=collect_boundary_metrics,
                teacher_nonfinite_positions=teacher_nonfinite_positions,
            )
            original_student_segment_ids = response_ids[start:end]
            prefix_response_ids = response_ids[:start]
            prefix_response_mask = response_mask[:start]
            prefix_response_logprobs = response_logprobs[:start]
            prefix_gap_mask = gap_mask[:start]
            prefix_teacher_segment_mask = teacher_segment_mask[:start] if teacher_segment_mask is not None else None
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

            candidate_selection_metadata: dict[str, Any] = {}
            if outcome_aware_selection:
                raw_correct = bool(outcome_metadata["raw_correct"])
                if raw_correct:
                    configured_probe_probability = float(intervention_config.teacher_nll_healthy_probe_probability)
                    _, _, calibration_count = AgentLoopWorker._current_teacher_nll_threshold(self)
                    calibration_warmup = calibration_count < intervention_config.teacher_nll_calibration_min_samples
                    probe_probability = (
                        0.0 if validate else (1.0 if calibration_warmup else configured_probe_probability)
                    )
                    should_probe = not validate and (
                        _stable_fraction(f"{base_request_id}:healthy-teacher-nll-probe:{attempt}") < probe_probability
                    )
                    candidate_selection_metadata = {
                        "healthy_calibration_probe": should_probe,
                        "healthy_calibration_warmup": calibration_warmup,
                        "healthy_calibration_probe_probability": configured_probe_probability,
                        "healthy_calibration_effective_probe_probability": probe_probability,
                        "teacher_nll_calibration_read_only": bool(validate),
                    }
                    if should_probe:
                        calibration_result = await self._generate_and_select_teacher_candidate(
                            prefix_ids=prompt_ids + prefix_response_ids,
                            max_tokens=teacher_max_tokens,
                            output=output,
                            routing_key=routing_key,
                            base_request_id=base_request_id,
                            attempt=attempt,
                            calibration_only=True,
                        )
                        candidate_selection_metadata.update(calibration_result.metadata)
                    _record_skipped_intervention(
                        skipped_interventions,
                        reason="outcome_correct",
                        attempt=attempt,
                        start=start,
                        end=end,
                        candidate_response_ids=response_ids,
                        candidate_response_mask=response_mask,
                    )
                    skipped_interventions[-1].update(candidate_selection_metadata)
                    outcome_metadata["skip_reason"] = "outcome_correct"
                    outcome_metadata["healthy_calibration_probe"] = should_probe
                    break

                selection_result = await self._generate_and_select_teacher_candidate(
                    prefix_ids=prompt_ids + prefix_response_ids,
                    max_tokens=teacher_max_tokens,
                    output=output,
                    routing_key=routing_key,
                    base_request_id=base_request_id,
                    attempt=attempt,
                    calibration_only=False,
                )
                candidate_selection_metadata = selection_result.metadata
                candidate_selection_metadata["teacher_nll_calibration_read_only"] = bool(validate)
                if selection_result.candidate is None:
                    skip_reason = selection_result.skip_reason or "teacher_candidate_selection_failed"
                    _record_skipped_intervention(
                        skipped_interventions,
                        reason=skip_reason,
                        attempt=attempt,
                        start=start,
                        end=end,
                        candidate_response_ids=prefix_response_ids,
                        candidate_response_mask=prefix_response_mask,
                    )
                    skipped_interventions[-1].update(candidate_selection_metadata)
                    outcome_metadata["skip_reason"] = skip_reason
                    break
                teacher_generation = selection_result.candidate.generation
                teacher_segment_ids = list(selection_result.candidate.token_ids)
                teacher_eos_truncated = selection_result.candidate.eos_truncated
            else:
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
            if future_gap_proposal is not None:
                future_gap_proposal.update(
                    {
                        "original_student_tokens": len(original_student_segment_ids),
                        "original_student_segment_sha256": _token_ids_sha256(original_student_segment_ids),
                        "teacher_tokens": len(teacher_segment_ids),
                        "teacher_segment_sha256": _token_ids_sha256(teacher_segment_ids),
                        **rewrite_overlap_metrics,
                    }
                )

            teacher_terminal = _generation_is_terminal(
                token_ids=teacher_segment_ids,
                max_tokens=teacher_max_tokens,
                stop_reason=teacher_generation.stop_reason,
                finish_reason=teacher_generation.finish_reason,
                eos_truncated=teacher_eos_truncated,
            )
            teacher_response_mask_value = int(
                intervention_config.train_on_teacher_segments and intervention_config.teacher_segment_loss_mode == "opd"
            )
            teacher_response_mask = [teacher_response_mask_value] * len(teacher_segment_ids)
            # Placeholder values keep response_logprobs aligned. They are ignored when teacher tokens are masked,
            # and train_on_teacher_segments=True drops rollout logprobs so the actor recomputes them.
            teacher_segment_logprobs = [0.0] * len(teacher_segment_ids)
            remaining_tokens = self.rollout_config.response_length - len(prefix_response_ids) - len(teacher_segment_ids)
            student_suffix_max_tokens = remaining_tokens
            if future_gap_confirmation:
                # Generate only the counterfactual horizon needed by the gate. A
                # candidate that passes is continued from these exact token IDs
                # after the teacher has scored the short suffix.
                student_suffix_max_tokens = min(
                    remaining_tokens,
                    intervention_config.future_gap_window_tokens,
                )
            student_suffix_result = _StudentSuffixGenerationResult(token_ids=[], logprobs=[])
            student_suffix_requested_tokens = 0
            if not teacher_terminal:
                student_suffix_requested_tokens = student_suffix_max_tokens
                student_suffix_result = await self._generate_student_suffix(
                    request_id=(
                        f"{base_request_id}:future-gap-confirm:{attempt}"
                        if future_gap_confirmation
                        else base_request_id
                    ),
                    prefix_ids=prompt_ids + prefix_response_ids + teacher_segment_ids,
                    max_tokens=student_suffix_max_tokens,
                    sampling_params=sampling_params,
                    output=output,
                )
            candidate_response_ids = prefix_response_ids + teacher_segment_ids + student_suffix_result.token_ids
            candidate_response_mask = (
                prefix_response_mask + teacher_response_mask + [1] * len(student_suffix_result.token_ids)
            )
            candidate_response_logprobs = (
                prefix_response_logprobs + teacher_segment_logprobs + student_suffix_result.logprobs
            )
            suffix_gap_mask = [1] * len(student_suffix_result.token_ids)
            for suffix_offset in student_suffix_result.nonfinite_logprob_offsets:
                suffix_gap_mask[suffix_offset] = 0
            candidate_gap_mask = prefix_gap_mask + [0] * len(teacher_segment_ids) + suffix_gap_mask
            student_suffix_logprob_metadata = _student_suffix_logprob_metadata(
                student_suffix_result,
                response_start=len(prefix_response_ids) + len(teacher_segment_ids),
            )
            candidate_teacher_segment_mask = None
            if prefix_teacher_segment_mask is not None:
                candidate_teacher_segment_mask = (
                    prefix_teacher_segment_mask
                    + [1] * len(teacher_segment_ids)
                    + [0] * len(student_suffix_result.token_ids)
                )
            _validate_intervention_state(
                context=f"after repair attempt {attempt}",
                response_ids=candidate_response_ids,
                response_mask=candidate_response_mask,
                response_logprobs=candidate_response_logprobs,
                gap_mask=candidate_gap_mask,
                response_length_limit=self.rollout_config.response_length,
                teacher_segment_mask=candidate_teacher_segment_mask,
            )
            thinking_state = AgentLoopWorker._teacher_intervention_thinking_state(
                self,
                candidate_response_ids,
                prompt_ids=prompt_ids,
            )
            if future_gap_proposal is not None:
                future_gap_proposal.update(
                    {
                        "thinking_state_valid": thinking_state.valid,
                        "thinking_state_open": thinking_state.open,
                        "thinking_state_reason": thinking_state.reason,
                    }
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
                skipped_interventions[-1].update(student_suffix_logprob_metadata)
                break
            candidate_terminal = teacher_terminal or student_suffix_result.terminal
            if future_gap_proposal is not None:
                future_gap_proposal.update(
                    {
                        "future_gap_rollout_mode": "short_horizon_counterfactual",
                        "future_gap_confirmation_requested_tokens": int(student_suffix_requested_tokens),
                        "future_gap_confirmation_generated_tokens": len(student_suffix_result.token_ids),
                        "future_gap_confirmation_terminal": candidate_terminal,
                        "future_gap_confirmation_candidate_tokens": len(candidate_response_ids),
                        "future_gap_confirmation_candidate_sha256": _token_ids_sha256(candidate_response_ids),
                        "future_gap_confirmation_thinking_state_valid": thinking_state.valid,
                        "future_gap_confirmation_thinking_state_open": thinking_state.open,
                        "future_gap_confirmation_thinking_state_reason": thinking_state.reason,
                        "future_gap_continuation_requested_tokens": 0,
                        "future_gap_continuation_generated_tokens": 0,
                        "future_gap_final_candidate_complete": (
                            candidate_terminal or len(candidate_response_ids) >= self.rollout_config.response_length
                        ),
                        "student_suffix_tokens": len(student_suffix_result.token_ids),
                        "student_suffix_sha256": _token_ids_sha256(student_suffix_result.token_ids),
                        "candidate_response_tokens": len(candidate_response_ids),
                        "candidate_response_sha256": _token_ids_sha256(candidate_response_ids),
                        "teacher_terminal": teacher_terminal,
                        "candidate_terminal": candidate_terminal,
                        "teacher_stop_reason": teacher_generation.stop_reason,
                        "teacher_finish_reason": teacher_generation.finish_reason,
                        "student_suffix_stop_reason": student_suffix_result.stop_reason,
                        "student_suffix_finish_reason": student_suffix_result.finish_reason,
                        **student_suffix_logprob_metadata,
                    }
                )
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
                skipped_interventions[-1].update(student_suffix_logprob_metadata)
                break
            candidate_trainable_tokens = sum(candidate_response_mask)
            if candidate_teacher_segment_mask is not None:
                candidate_trainable_tokens += sum(candidate_teacher_segment_mask)
            if not future_gap_confirmation and candidate_trainable_tokens < min_trainable_tokens:
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
                skipped_interventions[-1].update(student_suffix_logprob_metadata)
                break

            raw_response_ids = response_ids
            raw_response_mask = response_mask
            raw_response_logprobs = response_logprobs
            raw_gap_mask = gap_mask
            raw_teacher_segment_mask = teacher_segment_mask
            raw_teacher_result = teacher_result
            response_ids = candidate_response_ids
            response_mask = candidate_response_mask
            response_logprobs = candidate_response_logprobs
            gap_mask = candidate_gap_mask
            teacher_segment_mask = candidate_teacher_segment_mask
            cached_teacher_result = None
            boundary_metadata: dict[str, Any] = {}
            if collect_boundary_metrics:
                teacher_segment_sha256 = _token_ids_sha256(teacher_segment_ids)
                original_segment_sha256 = _token_ids_sha256(original_student_segment_ids)
                stable_sample_parts = (
                    (sample_kwargs or {}).get("data_source"),
                    (sample_kwargs or {}).get("index"),
                    (sample_kwargs or {}).get("global_steps"),
                )
                stable_sample_key = ":".join(str(value) for value in stable_sample_parts)
                if all(value is None for value in stable_sample_parts):
                    stable_sample_key = base_request_id
                boundary_rank_key = (
                    f"{stable_sample_key}:{attempt}:{start}:{original_segment_sha256}:{teacher_segment_sha256}"
                )
                boundary_metadata = {
                    "gap_metric": str(intervention_config.gap_metric).lower(),
                    "gap_threshold": float(intervention_config.gap_threshold),
                    "gap_aggregation": str(intervention_config.gap_aggregation),
                    "original_student_tokens": len(original_student_segment_ids),
                    "original_student_segment_sha256": original_segment_sha256,
                    "teacher_segment_sha256": teacher_segment_sha256,
                    "teacher_terminal": teacher_terminal,
                    "boundary_final_span_available": True,
                    "boundary_case_rank": _stable_fraction(boundary_rank_key),
                }
                for previous_intervention in interventions:
                    previous_start = int(previous_intervention.get("start", -1))
                    previous_teacher_end = previous_start + int(previous_intervention.get("teacher_tokens", 0) or 0)
                    if start <= previous_start or start < previous_teacher_end:
                        previous_intervention["boundary_final_span_available"] = False
                        previous_intervention["boundary_final_span_unavailable_reason"] = (
                            "overwritten_by_later_intervention"
                        )
            intervention_item = {
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
                **persistence_metadata,
                **rewrite_overlap_metrics,
                **local_effect_metadata,
                **candidate_selection_metadata,
                **student_suffix_logprob_metadata,
                **boundary_metadata,
            }
            if collect_boundary_text or (
                isinstance(outcome_metadata, dict) and outcome_metadata.get("transition_observation_event")
            ):
                intervention_item.update(
                    {
                        "_original_student_segment_ids": list(original_student_segment_ids),
                        "_teacher_segment_ids": list(teacher_segment_ids),
                    }
                )
            if future_gap_confirmation:
                if future_gap_proposal is None:
                    raise RuntimeError("future_gap confirmation lost its proposal metadata before commit.")
                pending_local_effect = (intervention_item, start + len(teacher_segment_ids))
                pending_future_gap = _PendingFutureGapConfirmation(
                    proposal=future_gap_proposal,
                    intervention=intervention_item,
                    after_intervention_start=start + len(teacher_segment_ids),
                    baseline_values=future_gap_baseline_values,
                    raw_response_ids=raw_response_ids,
                    raw_response_mask=raw_response_mask,
                    raw_response_logprobs=raw_response_logprobs,
                    raw_gap_mask=raw_gap_mask,
                    raw_teacher_segment_mask=raw_teacher_segment_mask,
                    raw_teacher_result=raw_teacher_result,
                    confirmation_suffix_result=student_suffix_result,
                )
            else:
                interventions.append(intervention_item)
                if isinstance(outcome_metadata, dict):
                    outcome_metadata["repair_executed"] = True
                if candidate_terminal:
                    _mark_local_effect_unavailable(intervention_item, "terminal_response")
                    break
                pending_local_effect = (intervention_item, start + len(teacher_segment_ids))

        if pending_local_effect is not None:
            pending_metadata, _ = pending_local_effect
            _mark_local_effect_unavailable(pending_metadata, "repaired_response_not_rescored")
        if pending_future_gap is not None:
            proposal = pending_future_gap.proposal
            proposal.update(
                {
                    "future_gap_pair_available": False,
                    "accepted": False,
                    "status": "rejected",
                    "rejection_stage": "future_gap_confirmation",
                    "reason": "future_gap_repaired_response_not_rescored",
                }
            )
            response_ids = pending_future_gap.raw_response_ids
            response_mask = pending_future_gap.raw_response_mask
            response_logprobs = pending_future_gap.raw_response_logprobs
            gap_mask = pending_future_gap.raw_gap_mask
            teacher_segment_mask = pending_future_gap.raw_teacher_segment_mask
            cached_teacher_result = pending_future_gap.raw_teacher_result
            pending_future_gap = None

        if future_gap_confirmation:
            for proposal in proposals:
                if proposal.get("status") != "triggered":
                    continue
                reason = "future_gap_candidate_unavailable"
                if skipped_interventions:
                    reason = str(skipped_interventions[-1].get("reason") or reason)
                proposal.update(
                    {
                        "future_gap_pair_available": False,
                        "accepted": False,
                        "status": "rejected",
                        "rejection_stage": "candidate_construction",
                        "reason": reason,
                    }
                )

        if isinstance(outcome_metadata, dict) and interventions:
            # The reward currently belongs to the untouched on-policy answer.
            # Force _agent_loop_postprocess to score the repaired response.
            output.reward_score = None
            output.extra_fields.pop("reward_extra_info", None)
            outcome_metadata["reward_recompute_required"] = True

        output.response_ids = response_ids[: self.rollout_config.response_length]
        output.response_mask = response_mask[: len(output.response_ids)]
        output.response_logprobs = response_logprobs[: len(output.response_ids)]
        output.teacher_segment_mask = (
            teacher_segment_mask[: len(output.response_ids)] if teacher_segment_mask is not None else None
        )
        _validate_intervention_state(
            context="final",
            response_ids=output.response_ids,
            response_mask=output.response_mask,
            response_logprobs=output.response_logprobs,
            gap_mask=gap_mask[: len(output.response_ids)],
            response_length_limit=self.rollout_config.response_length,
            teacher_segment_mask=output.teacher_segment_mask,
        )
        output.extra_fields["teacher_intervention_count"] = len(interventions)
        output.extra_fields["teacher_interventions"] = interventions
        output.extra_fields["teacher_intervention_skipped"] = skipped_interventions
        output.extra_fields["teacher_intervention_gap_diagnostics"] = gap_diagnostics
        if future_gap_confirmation:
            output.extra_fields["teacher_intervention_proposals"] = proposals
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
        if AgentLoopWorker._collect_intervention_boundary_text(self, validate=validate):
            AgentLoopWorker._prepare_local_intervention_boundary_cases(self, inputs)
        AgentLoopWorker._prepare_local_intervention_outcome_cases(self, inputs)
        # Convert lists back to tensors and stack them to create a batch.
        prompt_ids = torch.cat([input.prompt_ids for input in inputs], dim=0)
        response_ids = torch.cat([input.response_ids for input in inputs], dim=0)
        response_mask = torch.cat([input.response_mask for input in inputs], dim=0)
        attention_mask = torch.cat([input.attention_mask for input in inputs], dim=0)
        input_ids = torch.cat([input.input_ids for input in inputs], dim=0)
        position_ids = torch.cat([input.position_ids for input in inputs], dim=0)
        optional_outputs = {}
        has_response_logprobs = _require_consistent_optional_field(inputs, "response_logprobs")
        has_teacher_segment_mask = _require_consistent_optional_field(inputs, "teacher_segment_mask")
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
        if has_teacher_segment_mask:
            optional_outputs["teacher_segment_mask"] = torch.cat(
                [input.teacher_segment_mask for input in inputs], dim=0
            )
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
        self._global_teacher_nll_calibration: Optional[_GlobalTeacherNLLCalibration] = None

        distillation_config = getattr(config, "distillation", None)
        if is_distillation_enabled(distillation_config):
            intervention_config: DistillationInterventionConfig = omega_conf_to_dataclass(
                distillation_config.intervention,
                dataclass_type=DistillationInterventionConfig,
            )
            if intervention_config.outcome_aware_candidate_selection_enabled:
                self._global_teacher_nll_calibration = _GlobalTeacherNLLCalibration(intervention_config)

        if not hasattr(self, "agent_loop_workers_class"):
            self.agent_loop_workers_class = ray.remote(AgentLoopWorker)

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        """Create agent loop manager."""
        instance = cls(*args, **kwargs)
        await instance._init_agent_loop_workers()
        return instance

    def _apply_global_teacher_nll_calibration_updates(
        self,
        worker_updates: list[list[float]],
    ) -> Optional[_TeacherNLLCalibrationSnapshot]:
        if self._global_teacher_nll_calibration is None:
            return None
        return self._global_teacher_nll_calibration.add_updates(worker_updates)

    def export_teacher_nll_calibration_state(self) -> Optional[dict[str, Any]]:
        """Export the manager-owned calibration without mutating evaluation state."""

        if self._global_teacher_nll_calibration is None:
            return None
        return self._global_teacher_nll_calibration.export_state()

    async def _refresh_global_teacher_nll_calibration(self) -> None:
        """Aggregate the previous batch's updates and broadcast one snapshot before the next batch."""
        if self._global_teacher_nll_calibration is None:
            return
        worker_updates = await asyncio.gather(
            *[worker.drain_teacher_nll_calibration_updates.remote() for worker in self.agent_loop_workers]
        )
        snapshot = self._apply_global_teacher_nll_calibration_updates(worker_updates)
        await asyncio.gather(
            *[worker.set_teacher_nll_calibration_state.remote(snapshot) for worker in self.agent_loop_workers]
        )

    def _cap_intervention_outcome_cases(self, output: DataProto) -> None:
        """Apply the configured global per-transition full-text case cap."""
        key = "teacher_intervention_outcome_case"
        if key not in output.non_tensor_batch:
            return
        diagnostics_config = getattr(getattr(self.config, "distillation", None), "diagnostics", None)
        case_limit = int(getattr(diagnostics_config, "outcome_case_samples_per_transition", 0) or 0)
        cases = output.non_tensor_batch[key]
        candidates: dict[str, list[tuple[float, int]]] = {}
        for index, case in enumerate(cases):
            if not isinstance(case, dict):
                continue
            candidates.setdefault(str(case["transition"]), []).append((float(case["case_rank"]), index))
        selected = {
            index
            for transition_candidates in candidates.values()
            for _, index in sorted(transition_candidates)[:case_limit]
        }
        for index in range(len(cases)):
            if index not in selected:
                cases[index] = None

    def _cap_intervention_boundary_cases(self, output: DataProto) -> None:
        """Apply the global stable cap after worker-local boundary-case selection."""
        key = "teacher_intervention_boundary_cases"
        if key not in output.non_tensor_batch:
            return
        diagnostics_config = getattr(getattr(self.config, "distillation", None), "diagnostics", None)
        case_limit = int(getattr(diagnostics_config, "intervention_boundary_text_samples_per_step", 0) or 0)
        sample_cases = output.non_tensor_batch[key]
        candidates: list[tuple[float, int, int]] = []
        for sample_index, cases in enumerate(sample_cases):
            if not isinstance(cases, list):
                continue
            for case_index, case in enumerate(cases):
                if isinstance(case, dict) and case.get("case_rank") is not None:
                    candidates.append((float(case["case_rank"]), sample_index, case_index))
        selected = {(sample_index, case_index) for _, sample_index, case_index in sorted(candidates)[:case_limit]}
        for sample_index, cases in enumerate(sample_cases):
            if not isinstance(cases, list):
                continue
            retained = [case for case_index, case in enumerate(cases) if (sample_index, case_index) in selected]
            sample_cases[sample_index] = retained or None

    def _refresh_global_teacher_nll_calibration_blocking(self) -> None:
        """Blocking equivalent used by the TransferQueue rollout manager."""
        if self._global_teacher_nll_calibration is None:
            return
        worker_updates = ray.get(
            [worker.drain_teacher_nll_calibration_updates.remote() for worker in self.agent_loop_workers]
        )
        snapshot = self._apply_global_teacher_nll_calibration_updates(worker_updates)
        ray.get([worker.set_teacher_nll_calibration_state.remote(snapshot) for worker in self.agent_loop_workers])

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
        await self._refresh_global_teacher_nll_calibration()
        chunkes = prompts.chunk(len(self.agent_loop_workers))
        outputs = await asyncio.gather(
            *[
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=True)
            ]
        )
        output = DataProto.concat(outputs)
        self._cap_intervention_outcome_cases(output)
        self._cap_intervention_boundary_cases(output)

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
