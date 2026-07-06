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

from typing import Optional

import torch


def _validate_topk_inputs(
    *,
    student_topk_ids: torch.Tensor,
    student_topk_logprobs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    teacher_topk_logprobs: torch.Tensor,
    response_mask: torch.Tensor,
) -> None:
    if student_topk_ids.shape != student_topk_logprobs.shape:
        raise ValueError(
            "student_topk_ids and student_topk_logprobs must align, "
            f"got {student_topk_ids.shape=} and {student_topk_logprobs.shape=}."
        )
    if teacher_topk_ids.shape != teacher_topk_logprobs.shape:
        raise ValueError(
            "teacher_topk_ids and teacher_topk_logprobs must align, "
            f"got {teacher_topk_ids.shape=} and {teacher_topk_logprobs.shape=}."
        )
    if student_topk_ids.ndim != 3 or teacher_topk_ids.ndim != 3:
        raise ValueError(
            "top-k tensors must have shape [batch, response_len, topk], "
            f"got {student_topk_ids.shape=} and {teacher_topk_ids.shape=}."
        )
    if student_topk_ids.shape[:2] != response_mask.shape or teacher_topk_ids.shape[:2] != response_mask.shape:
        raise ValueError(
            "top-k tensors must align with response_mask, "
            f"got {student_topk_ids.shape=}, {teacher_topk_ids.shape=}, and {response_mask.shape=}."
        )


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    selected = values[mask]
    if selected.numel() == 0:
        return 0.0
    return float(selected.float().mean().item())


def compute_only_student_topk_token_reward(
    *,
    student_topk_ids: torch.Tensor,
    student_topk_logprobs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    teacher_topk_logprobs: torch.Tensor,
    response_mask: torch.Tensor,
    reward_mask: Optional[torch.Tensor] = None,
    topk: Optional[int] = None,
    reward_weight_mode: str = "student_p",
    log_prob_min_clamp: Optional[float] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute per-candidate token rewards from student top-k support.

    This implements the repository's vLLM/SGLang-compatible approximation of
    TOP_K_STRATEGY=only_stu: the student top-k support is used as the candidate
    set, and teacher probabilities are gathered when those ids are present in
    the returned teacher top-k list. Missing teacher entries are masked instead
    of being assigned fabricated log probabilities.

    Returns:
        topk_rewards: Tensor with shape [batch, response_len, topk]. The final
            dimension is kept so the actor update can apply gradients to the same
            top-k candidate set, matching the OPD paper's Student Top-K update
            shape as closely as the current teacher backend allows.
    """
    _validate_topk_inputs(
        student_topk_ids=student_topk_ids,
        student_topk_logprobs=student_topk_logprobs,
        teacher_topk_ids=teacher_topk_ids,
        teacher_topk_logprobs=teacher_topk_logprobs,
        response_mask=response_mask,
    )
    available_topk = min(student_topk_ids.shape[-1], teacher_topk_ids.shape[-1])
    if topk is not None:
        if topk <= 0:
            raise ValueError(f"topk must be positive, got {topk}.")
        available_topk = min(available_topk, int(topk))
    if available_topk <= 0:
        empty_rewards = response_mask.new_zeros((*response_mask.shape, 0), dtype=torch.float32)
        return empty_rewards, {
            "distillation/token_reward_direct/available_topk": 0.0,
            "distillation/token_reward_direct/teacher_on_student_topk_coverage": 0.0,
            "distillation/token_reward_direct/nonempty_token_fraction": 0.0,
        }

    response_mask = response_mask.bool()
    if reward_mask is None:
        reward_mask = response_mask
    else:
        reward_mask = reward_mask.bool() & response_mask

    student_ids = student_topk_ids[..., :available_topk]
    student_logprobs = student_topk_logprobs[..., :available_topk].float()
    teacher_ids = teacher_topk_ids[..., :available_topk]
    teacher_logprobs = teacher_topk_logprobs[..., :available_topk].float()

    matches = student_ids.unsqueeze(-1) == teacher_ids.unsqueeze(-2)
    teacher_on_student = torch.where(
        matches,
        teacher_logprobs.unsqueeze(-2),
        torch.full_like(matches, float("-inf"), dtype=teacher_logprobs.dtype),
    ).amax(dim=-1)

    valid_topk = reward_mask.unsqueeze(-1) & torch.isfinite(student_logprobs) & torch.isfinite(teacher_on_student)
    if log_prob_min_clamp is not None:
        clamp_value = float(log_prob_min_clamp)
        student_logprobs = student_logprobs.clamp_min(clamp_value)
        teacher_on_student = teacher_on_student.clamp_min(clamp_value)

    if reward_weight_mode == "student_p":
        weight_logits = student_logprobs
    elif reward_weight_mode == "teacher_p":
        weight_logits = teacher_on_student
    elif reward_weight_mode == "none":
        weight_logits = torch.zeros_like(student_logprobs)
    else:
        raise ValueError(
            f"reward_weight_mode must be one of 'student_p', 'teacher_p', or 'none', got {reward_weight_mode!r}."
        )

    weight_logits = torch.where(valid_topk, weight_logits, torch.full_like(weight_logits, float("-inf")))
    has_valid = valid_topk.any(dim=-1)
    normalized_weight_logits = weight_logits - torch.logsumexp(weight_logits, dim=-1, keepdim=True)
    weights = torch.where(
        valid_topk & has_valid.unsqueeze(-1),
        normalized_weight_logits.exp(),
        torch.zeros_like(normalized_weight_logits),
    )
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

    topk_advantages = teacher_on_student - student_logprobs
    topk_rewards = torch.where(valid_topk, topk_advantages, torch.zeros_like(topk_advantages)) * weights
    topk_rewards = torch.where(reward_mask.unsqueeze(-1), topk_rewards, torch.zeros_like(topk_rewards))
    scalar_token_rewards = topk_rewards.sum(dim=-1)

    active_tokens = reward_mask
    active_token_count = int(active_tokens.sum().item())
    coverage_denom = max(active_token_count * available_topk, 1)
    coverage = float(valid_topk.sum().item()) / coverage_denom
    nonempty_fraction = _masked_mean(has_valid.float(), active_tokens)

    metrics = {
        "distillation/token_reward_direct/available_topk": float(available_topk),
        "distillation/token_reward_direct/teacher_on_student_topk_coverage": coverage,
        "distillation/token_reward_direct/nonempty_token_fraction": nonempty_fraction,
        "distillation/token_reward_direct/reward_mean": _masked_mean(scalar_token_rewards, active_tokens),
        "distillation/token_reward_direct/reward_abs_mean": _masked_mean(scalar_token_rewards.abs(), active_tokens),
    }
    if response_mask.any():
        metrics["distillation/token_reward_direct/active_token_fraction"] = float(active_tokens.sum().item()) / float(
            response_mask.sum().item()
        )
    else:
        metrics["distillation/token_reward_direct/active_token_fraction"] = 0.0

    return topk_rewards, metrics
