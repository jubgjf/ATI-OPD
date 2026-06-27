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

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch


def _to_cpu_float(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().float().cpu()


def _masked_values(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    values = _to_cpu_float(values)
    mask = mask.detach().bool().cpu()
    valid = mask & torch.isfinite(values)
    return values[valid]


def _safe_mean(values: torch.Tensor) -> Optional[float]:
    if values.numel() == 0:
        return None
    return float(values.mean().item())


def _safe_quantile(values: torch.Tensor, q: float) -> Optional[float]:
    if values.numel() == 0:
        return None
    return float(torch.quantile(values, q).item())


def _put_if_present(metrics: dict[str, float], key: str, value: Optional[float]) -> None:
    if value is not None and np.isfinite(value):
        metrics[key] = value


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> Optional[float]:
    return _safe_mean(_masked_values(values, mask))


def _masked_quantile(values: torch.Tensor, mask: torch.Tensor, q: float) -> Optional[float]:
    return _safe_quantile(_masked_values(values, mask), q)


def _bin_key(prefix: str, start: int, end: int) -> str:
    return f"{prefix}/{start:04d}_{end:04d}"


def compute_binned_token_means(
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    bin_size: int,
    prefix: str,
) -> dict[str, float]:
    """Compute position-binned token means for a [batch, response_len] tensor."""
    if bin_size <= 0:
        raise ValueError(f"bin_size must be positive, got {bin_size}.")
    if values.shape != mask.shape:
        raise ValueError(f"values and mask must have the same shape, got {values.shape=} and {mask.shape=}.")

    metrics: dict[str, float] = {}
    response_len = values.shape[1]
    for start in range(0, response_len, bin_size):
        end = min(start + bin_size, response_len)
        mean = _masked_mean(values[:, start:end], mask[:, start:end])
        _put_if_present(metrics, _bin_key(prefix, start, end), mean)
    return metrics


def compute_binned_event_rates(
    event_mask: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    bin_size: int,
    prefix: str,
) -> dict[str, float]:
    """Compute position-binned event rates for [batch, response_len] boolean tensors."""
    if bin_size <= 0:
        raise ValueError(f"bin_size must be positive, got {bin_size}.")
    if event_mask.shape != response_mask.shape:
        raise ValueError(
            "event_mask and response_mask must have the same shape, "
            f"got {event_mask.shape=} and {response_mask.shape=}."
        )

    event_mask = event_mask.detach().bool().cpu()
    response_mask = response_mask.detach().bool().cpu()
    metrics: dict[str, float] = {}
    response_len = response_mask.shape[1]
    for start in range(0, response_len, bin_size):
        end = min(start + bin_size, response_len)
        valid = response_mask[:, start:end]
        denom = int(valid.sum().item())
        if denom == 0:
            continue
        count = int((event_mask[:, start:end] & valid).sum().item())
        metrics[_bin_key(prefix, start, end)] = count / denom
    return metrics


def _event_metric_key(prefix: str, separator: str, suffix: str) -> str:
    return f"{prefix}{separator}{suffix}"


def _normalize_position_list(value: Any, *, context: str, sample_idx: int) -> list[int]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    elif isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, int | np.integer):
        return [int(value)]
    if not isinstance(value, list | tuple):
        raise TypeError(
            f"{context} positions for sample {sample_idx} must be a list/tuple of integers, got {type(value).__name__}."
        )

    positions = []
    for raw_position in value:
        if raw_position is None:
            continue
        if isinstance(raw_position, float | np.floating) and not np.isfinite(raw_position):
            raise ValueError(f"{context} positions for sample {sample_idx} contain non-finite value {raw_position!r}.")
        positions.append(int(raw_position))
    return positions


def positions_to_event_mask(
    event_positions: Optional[list[Any]],
    response_mask: torch.Tensor,
    *,
    context: str,
) -> torch.Tensor:
    """Convert per-sample response-relative positions to a boolean event mask."""
    response_mask_cpu = response_mask.detach().bool().cpu()
    event_mask = torch.zeros_like(response_mask_cpu, dtype=torch.bool)
    if event_positions is None:
        return event_mask
    if hasattr(event_positions, "tolist"):
        event_positions = event_positions.tolist()
    else:
        event_positions = list(event_positions)
    if len(event_positions) == 0:
        return event_mask
    if len(event_positions) != response_mask_cpu.shape[0]:
        raise ValueError(
            f"{context} positions batch-size mismatch: got {len(event_positions)} samples, "
            f"expected {response_mask_cpu.shape[0]} from response_mask.shape={tuple(response_mask_cpu.shape)}."
        )

    response_width = response_mask_cpu.shape[1]
    for sample_idx, sample_positions in enumerate(event_positions):
        for position in _normalize_position_list(sample_positions, context=context, sample_idx=sample_idx):
            if position < 0 or position >= response_width:
                raise ValueError(
                    f"{context} position {position} for sample {sample_idx} is outside response width {response_width}."
                )
            event_mask[sample_idx, position] = True
    return event_mask & response_mask_cpu


def compute_event_position_metrics(
    *,
    event_mask: torch.Tensor,
    response_mask: torch.Tensor,
    metric_prefix: str,
    suffix_separator: str,
    token_fraction_suffix: str,
    bin_prefix: str,
    bin_size: int,
) -> dict[str, float]:
    """Aggregate event frequency and location metrics."""
    if event_mask.shape != response_mask.shape:
        raise ValueError(
            f"event_mask and response_mask must align, got {event_mask.shape=} and {response_mask.shape=}."
        )
    if suffix_separator not in {"/", "_"}:
        raise ValueError(f"suffix_separator must be '/' or '_', got {suffix_separator!r}.")

    event_mask = event_mask.detach().bool().cpu()
    response_mask = response_mask.detach().bool().cpu()
    event_mask = event_mask & response_mask
    valid_lengths = response_mask.sum(dim=1)
    event_counts = event_mask.sum(dim=1)
    total_valid_tokens = int(response_mask.sum().item())
    total_events = int(event_counts.sum().item())
    batch_size = int(response_mask.shape[0])

    metrics = {
        _event_metric_key(metric_prefix, suffix_separator, token_fraction_suffix): total_events
        / max(total_valid_tokens, 1),
        _event_metric_key(metric_prefix, suffix_separator, "sample_fraction"): float((event_counts > 0).float().mean())
        if batch_size
        else 0.0,
        _event_metric_key(metric_prefix, suffix_separator, "count_mean"): float(event_counts.float().mean().item())
        if batch_size
        else 0.0,
        _event_metric_key(metric_prefix, suffix_separator, "count_max"): float(event_counts.max().item())
        if batch_size
        else 0.0,
    }

    first_positions = []
    first_position_fracs = []
    for sample_idx in range(batch_size):
        sample_positions = torch.nonzero(event_mask[sample_idx], as_tuple=False).flatten()
        if sample_positions.numel() == 0:
            continue
        first_position = int(sample_positions[0].item())
        first_positions.append(first_position)
        first_position_fracs.append(first_position / max(int(valid_lengths[sample_idx].item()), 1))
    if first_positions:
        metrics[_event_metric_key(metric_prefix, suffix_separator, "first_position_mean")] = float(
            np.mean(first_positions)
        )
        metrics[_event_metric_key(metric_prefix, suffix_separator, "first_position_frac_mean")] = float(
            np.mean(first_position_fracs)
        )

    metrics.update(
        compute_binned_event_rates(
            event_mask=event_mask,
            response_mask=response_mask,
            bin_size=bin_size,
            prefix=bin_prefix,
        )
    )
    return metrics


def _head_tail_metrics(
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    prefix: str,
    bin_size: int,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if values.shape[1] == 0:
        return metrics
    head_end = min(bin_size, values.shape[1])
    tail_start = max(values.shape[1] - bin_size, 0)
    head = _masked_mean(values[:, :head_end], mask[:, :head_end])
    tail = _masked_mean(values[:, tail_start:], mask[:, tail_start:])
    _put_if_present(metrics, f"{prefix}/head_mean", head)
    _put_if_present(metrics, f"{prefix}/tail_mean", tail)
    if head is not None and tail is not None:
        metrics[f"{prefix}/tail_minus_head"] = tail - head
        if abs(head) > 1e-8:
            metrics[f"{prefix}/tail_over_head"] = tail / head
    return metrics


def compute_topk_ladder(topk: int) -> list[int]:
    """Return powers-of-two coverage checkpoints ending at ``topk``."""
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}.")
    ladder = []
    value = 1
    while value < topk:
        ladder.append(value)
        value *= 2
    if not ladder or ladder[-1] != topk:
        ladder.append(topk)
    return ladder


def extract_response_topk_scores(
    *,
    sequence_ids: torch.Tensor,
    sequence_logprobs: torch.Tensor,
    response_mask: torch.Tensor,
    prompt_width: int,
    context: str = "teacher_topk",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract response-aligned top-k ids/logprobs from padded prompt+response teacher scoring output."""
    if sequence_ids.ndim != 3 or sequence_logprobs.ndim != 3:
        raise RuntimeError(
            f"{context} ids/logprobs must have rank 3 for OPD overlap diagnostics, "
            f"got ids.shape={tuple(sequence_ids.shape)} and logprobs.shape={tuple(sequence_logprobs.shape)}."
        )
    if sequence_ids.shape != sequence_logprobs.shape:
        raise RuntimeError(
            f"{context} ids/logprobs shape mismatch: "
            f"ids.shape={tuple(sequence_ids.shape)}, logprobs.shape={tuple(sequence_logprobs.shape)}."
        )
    if sequence_ids.shape[0] != response_mask.shape[0]:
        raise RuntimeError(
            f"{context} batch-size mismatch: ids.shape={tuple(sequence_ids.shape)}, "
            f"response_mask.shape={tuple(response_mask.shape)}."
        )
    if sequence_ids.shape[:2] == response_mask.shape:
        response_ids = sequence_ids
        response_logprobs = sequence_logprobs
    else:
        if prompt_width <= 0:
            raise RuntimeError("OPD overlap diagnostics require a positive padded prompt width.")
        response_width = response_mask.shape[1]
        start = prompt_width - 1
        end = start + response_width
        if sequence_ids.shape[1] < end:
            raise RuntimeError(
                f"Cannot align {context} with response tokens: ids.shape={tuple(sequence_ids.shape)}, "
                f"prompt_width={prompt_width}, response_width={response_width}, expected at least {end} positions."
            )
        response_ids = sequence_ids[:, start:end]
        response_logprobs = sequence_logprobs[:, start:end]
    if response_ids.shape[:2] != response_mask.shape:
        raise RuntimeError(
            f"OPD overlap diagnostics produced misaligned {context}: "
            f"response_ids.shape={tuple(response_ids.shape)}, response_mask.shape={tuple(response_mask.shape)}."
        )
    return response_ids, response_logprobs


def binary_auc(scores: torch.Tensor, labels: torch.Tensor) -> Optional[float]:
    """Compute ROC AUC with average ranks; returns None if only one class exists."""
    scores = _to_cpu_float(scores).flatten()
    labels = labels.detach().bool().cpu().flatten()
    finite = torch.isfinite(scores)
    scores = scores[finite]
    labels = labels[finite]
    n_pos = int(labels.sum().item())
    n_total = int(labels.numel())
    n_neg = n_total - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    order = torch.argsort(scores)
    sorted_scores = scores[order]
    ranks = torch.empty_like(scores)
    start = 0
    while start < n_total:
        end = start + 1
        while end < n_total and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end

    rank_sum_pos = ranks[labels].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc.item())


def extract_sampled_response_logprobs(
    *,
    sequence_logprobs: torch.Tensor,
    response_mask: torch.Tensor,
    prompt_width: int,
    context: str = "sequence_logprobs",
) -> torch.Tensor | None:
    """Extract response-token sampled logprobs from padded prompt+response prompt scoring output.

    Returns None when the tensor contains top-k logprobs instead of one sampled-token logprob per position.
    """
    if sequence_logprobs.ndim not in (2, 3):
        raise RuntimeError(
            f"{context} must have rank 2 or 3 for OPD diagnostics, got shape={tuple(sequence_logprobs.shape)}."
        )
    if sequence_logprobs.shape[0] != response_mask.shape[0]:
        raise RuntimeError(
            f"{context} batch-size mismatch for OPD diagnostics: "
            f"sequence_logprobs.shape={tuple(sequence_logprobs.shape)}, "
            f"response_mask.shape={tuple(response_mask.shape)}."
        )
    if sequence_logprobs.ndim == 3 and sequence_logprobs.shape[-1] != 1:
        return None

    if sequence_logprobs.shape[:2] == response_mask.shape:
        response_logprobs = sequence_logprobs
    else:
        if prompt_width <= 0:
            raise RuntimeError("OPD diagnostics require a positive padded prompt width.")
        response_width = response_mask.shape[1]
        start = prompt_width - 1
        end = start + response_width
        if sequence_logprobs.shape[1] < end:
            raise RuntimeError(
                f"Cannot align {context} with response tokens for OPD diagnostics: "
                f"sequence_logprobs.shape={tuple(sequence_logprobs.shape)}, prompt_width={prompt_width}, "
                f"response_width={response_width}, expected at least {end} scored positions."
            )
        response_logprobs = sequence_logprobs[:, start:end]

    if response_logprobs.ndim == 3:
        response_logprobs = response_logprobs.squeeze(-1)
    if response_logprobs.shape != response_mask.shape:
        raise RuntimeError(
            f"OPD diagnostics produced misaligned {context}: "
            f"response_logprobs.shape={tuple(response_logprobs.shape)}, "
            f"response_mask.shape={tuple(response_mask.shape)}."
        )
    return response_logprobs


def compute_overlap_diagnostics(
    *,
    response_ids: torch.Tensor,
    response_mask: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    teacher_topk_logprobs: torch.Tensor,
    sampled_teacher_logprobs: Optional[torch.Tensor] = None,
    bin_size: int,
    topk: int,
) -> dict[str, float]:
    """Compute teacher top-k support diagnostics for student response tokens."""
    if response_ids.shape != response_mask.shape:
        raise ValueError(
            f"response_ids and response_mask must align, got {response_ids.shape=} and {response_mask.shape=}."
        )
    if teacher_topk_ids.shape != teacher_topk_logprobs.shape:
        raise ValueError(
            "teacher_topk_ids and teacher_topk_logprobs must align, "
            f"got {teacher_topk_ids.shape=} and {teacher_topk_logprobs.shape=}."
        )
    if teacher_topk_ids.ndim != 3 or teacher_topk_ids.shape[:2] != response_mask.shape:
        raise ValueError(
            "teacher_topk_ids must have shape [batch, response_len, k], "
            f"got {teacher_topk_ids.shape=} and response_mask.shape={response_mask.shape}."
        )
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}.")

    response_ids = response_ids.detach().cpu()
    response_mask = response_mask.detach().bool().cpu()
    teacher_topk_ids = teacher_topk_ids.detach().cpu()
    teacher_topk_logprobs = _to_cpu_float(teacher_topk_logprobs)
    finite_topk = torch.isfinite(teacher_topk_logprobs)
    valid_topk_rows = response_mask & finite_topk.any(dim=-1)
    available_topk = min(int(topk), int(teacher_topk_ids.shape[-1]))
    metrics: dict[str, float] = {}

    for k in compute_topk_ladder(topk):
        if k > available_topk:
            continue
        coverage = (teacher_topk_ids[..., :k] == response_ids.unsqueeze(-1)).any(dim=-1)
        coverage = coverage & valid_topk_rows
        _put_if_present(metrics, f"opd/overlap/top{k}_rate", _safe_mean(coverage[valid_topk_rows].float()))
        metrics.update(
            compute_binned_event_rates(
                event_mask=coverage,
                response_mask=valid_topk_rows,
                bin_size=bin_size,
                prefix=f"opd/overlap/top{k}_bin",
            )
        )
        metrics.update(
            _head_tail_metrics(
                coverage.float(),
                valid_topk_rows,
                prefix=f"opd/overlap/top{k}",
                bin_size=bin_size,
            )
        )

    if sampled_teacher_logprobs is not None:
        if sampled_teacher_logprobs.ndim == 3 and sampled_teacher_logprobs.shape[-1] == 1:
            sampled_teacher_logprobs = sampled_teacher_logprobs.squeeze(-1)
        if sampled_teacher_logprobs.shape != response_mask.shape:
            raise ValueError(
                "sampled_teacher_logprobs and response_mask must align for OPD overlap diagnostics, "
                f"got {sampled_teacher_logprobs.shape=} and {response_mask.shape=}."
            )
        sampled_teacher_logprobs = _to_cpu_float(sampled_teacher_logprobs)
        sampled_prob = sampled_teacher_logprobs.exp()
        sampled_valid = response_mask & torch.isfinite(sampled_prob)
        sampled_values = sampled_prob[sampled_valid]
        _put_if_present(metrics, "opd/overlap/teacher_sampled_prob_mean", _safe_mean(sampled_values))
        _put_if_present(metrics, "opd/overlap/teacher_sampled_prob_p10", _safe_quantile(sampled_values, 0.1))
        _put_if_present(metrics, "opd/overlap/teacher_sampled_prob_p50", _safe_quantile(sampled_values, 0.5))
        metrics.update(
            compute_binned_token_means(
                sampled_prob,
                sampled_valid,
                bin_size=bin_size,
                prefix="opd/overlap/teacher_sampled_prob_bin",
            )
        )

    topk_probs = torch.where(finite_topk, teacher_topk_logprobs.exp(), torch.zeros_like(teacher_topk_logprobs))
    topk_mass = topk_probs[:, :, :available_topk].sum(dim=-1)
    _put_if_present(metrics, "opd/overlap/teacher_topk_mass_mean", _masked_mean(topk_mass, valid_topk_rows))
    return metrics


def compute_opd_diagnostics_metrics(
    *,
    student_logprobs: torch.Tensor,
    response_mask: torch.Tensor,
    bin_size: int,
    response_ids: Optional[torch.Tensor] = None,
    teacher_logprobs: Optional[torch.Tensor] = None,
    teacher_topk_ids: Optional[torch.Tensor] = None,
    teacher_topk_logprobs: Optional[torch.Tensor] = None,
    teacher_logprob_nonfinite_positions: Optional[list[Any]] = None,
    entropy: Optional[torch.Tensor] = None,
    rm_scores: Optional[torch.Tensor] = None,
    high_gap_threshold: float = 1.0,
    overlap_topk: Optional[int] = None,
) -> dict[str, float]:
    """Compute low-overhead OPD diagnostics from sampled-token tensors."""
    if student_logprobs.shape != response_mask.shape:
        raise ValueError(
            f"student_logprobs and response_mask must align, got {student_logprobs.shape=} and {response_mask.shape=}."
        )

    response_mask = response_mask.bool()
    metrics: dict[str, float] = {}
    response_lens = response_mask.sum(dim=1).detach().float().cpu()
    _put_if_present(metrics, "opd/response_len/mean", _safe_mean(response_lens))
    _put_if_present(metrics, "opd/response_len/p90", _safe_quantile(response_lens, 0.9))
    response_len_max = float(response_lens.max().item()) if response_lens.numel() else None
    _put_if_present(metrics, "opd/response_len/max", response_len_max)

    if teacher_logprob_nonfinite_positions is not None:
        nonfinite_event_mask = positions_to_event_mask(
            teacher_logprob_nonfinite_positions,
            response_mask,
            context="teacher_logprob_nonfinite",
        )
        metrics.update(
            compute_event_position_metrics(
                event_mask=nonfinite_event_mask,
                response_mask=response_mask,
                metric_prefix="opd/teacher_logprob_nonfinite",
                suffix_separator="/",
                token_fraction_suffix="response_token_fraction",
                bin_prefix="opd/teacher_logprob_nonfinite_bin",
                bin_size=bin_size,
            )
        )

    if teacher_logprobs is not None:
        if teacher_logprobs.ndim == 3 and teacher_logprobs.shape[-1] == 1:
            teacher_logprobs = teacher_logprobs.squeeze(-1)
        if teacher_logprobs.shape == student_logprobs.shape:
            token_reward = teacher_logprobs - student_logprobs
            abs_gap = token_reward.abs()
            reward_values = _masked_values(token_reward, response_mask)
            abs_gap_values = _masked_values(abs_gap, response_mask)
            _put_if_present(metrics, "opd/reward/mean", _safe_mean(reward_values))
            _put_if_present(metrics, "opd/reward/p10", _safe_quantile(reward_values, 0.1))
            _put_if_present(metrics, "opd/reward/p90", _safe_quantile(reward_values, 0.9))
            _put_if_present(metrics, "opd/gap_abs/mean", _safe_mean(abs_gap_values))
            _put_if_present(metrics, "opd/gap_abs/p90", _safe_quantile(abs_gap_values, 0.9))
            abs_gap_max = float(abs_gap_values.max().item()) if abs_gap_values.numel() else None
            _put_if_present(metrics, "opd/gap_abs/max", abs_gap_max)
            if abs_gap_values.numel():
                high_gap_mask = (abs_gap > high_gap_threshold) & response_mask & torch.isfinite(abs_gap.float())
                high_gap_metrics = compute_event_position_metrics(
                    event_mask=high_gap_mask,
                    response_mask=response_mask,
                    metric_prefix="opd/gap_abs/high",
                    suffix_separator="_",
                    token_fraction_suffix="token_fraction",
                    bin_prefix="opd/gap_abs/high_bin",
                    bin_size=bin_size,
                )
                metrics.update(high_gap_metrics)
                metrics["opd/gap_abs/high_fraction"] = high_gap_metrics["opd/gap_abs/high_token_fraction"]
            metrics.update(
                compute_binned_token_means(abs_gap, response_mask, bin_size=bin_size, prefix="opd/gap_abs_bin")
            )
            metrics.update(
                compute_binned_token_means(token_reward, response_mask, bin_size=bin_size, prefix="opd/reward_bin")
            )
            metrics.update(_head_tail_metrics(abs_gap, response_mask, prefix="opd/gap_abs", bin_size=bin_size))
            metrics.update(_head_tail_metrics(token_reward, response_mask, prefix="opd/reward", bin_size=bin_size))

            if rm_scores is not None:
                seq_reward = (token_reward * response_mask).sum(dim=1) / response_mask.sum(dim=1).clamp_min(1)
                labels = rm_scores.detach().float().sum(dim=1) > 0
                auc = binary_auc(seq_reward, labels)
                _put_if_present(metrics, "opd/reward/correct_auc", auc)
                correct_mean = _safe_mean(seq_reward.detach().float().cpu()[labels.detach().cpu()])
                incorrect_mean = _safe_mean(seq_reward.detach().float().cpu()[~labels.detach().cpu()])
                _put_if_present(metrics, "opd/reward/correct_mean", correct_mean)
                _put_if_present(metrics, "opd/reward/incorrect_mean", incorrect_mean)

    if (
        overlap_topk is not None
        and response_ids is not None
        and teacher_topk_ids is not None
        and teacher_topk_logprobs is not None
    ):
        metrics.update(
            compute_overlap_diagnostics(
                response_ids=response_ids,
                response_mask=response_mask,
                teacher_topk_ids=teacher_topk_ids,
                teacher_topk_logprobs=teacher_topk_logprobs,
                sampled_teacher_logprobs=teacher_logprobs,
                bin_size=bin_size,
                topk=overlap_topk,
            )
        )

    if entropy is not None:
        if entropy.shape != response_mask.shape:
            raise ValueError(f"entropy and response_mask must align, got {entropy.shape=} and {response_mask.shape=}.")
        entropy_values = _masked_values(entropy, response_mask)
        _put_if_present(metrics, "opd/entropy/mean", _safe_mean(entropy_values))
        _put_if_present(metrics, "opd/entropy/p90", _safe_quantile(entropy_values, 0.9))
        metrics.update(compute_binned_token_means(entropy, response_mask, bin_size=bin_size, prefix="opd/entropy_bin"))
        metrics.update(_head_tail_metrics(entropy, response_mask, prefix="opd/entropy", bin_size=bin_size))

    return metrics


def compute_entropy_heatmap_row(
    entropy: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    bin_size: int,
) -> np.ndarray:
    """Return one heatmap row: mean entropy per response-position bin."""
    return compute_value_heatmap_row(
        values=entropy,
        response_mask=response_mask,
        bin_size=bin_size,
        value_name="entropy",
    )


def compute_gap_heatmap_row(
    *,
    student_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    response_mask: torch.Tensor,
    bin_size: int,
) -> np.ndarray:
    """Return one heatmap row: mean sampled-token teacher/student absolute gap per response-position bin."""
    if teacher_logprobs.ndim == 3 and teacher_logprobs.shape[-1] == 1:
        teacher_logprobs = teacher_logprobs.squeeze(-1)
    if student_logprobs.shape != teacher_logprobs.shape:
        raise ValueError(
            "student_logprobs and teacher_logprobs must align for gap heatmap, "
            f"got {student_logprobs.shape=} and {teacher_logprobs.shape=}."
        )
    return compute_value_heatmap_row(
        values=(teacher_logprobs - student_logprobs).abs(),
        response_mask=response_mask,
        bin_size=bin_size,
        value_name="gap_abs",
    )


def compute_gap_high_heatmap_row(
    *,
    student_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    response_mask: torch.Tensor,
    bin_size: int,
    high_gap_threshold: float,
) -> np.ndarray:
    """Return one heatmap row: high-gap token rate per response-position bin."""
    if teacher_logprobs.ndim == 3 and teacher_logprobs.shape[-1] == 1:
        teacher_logprobs = teacher_logprobs.squeeze(-1)
    if student_logprobs.shape != teacher_logprobs.shape:
        raise ValueError(
            "student_logprobs and teacher_logprobs must align for high-gap heatmap, "
            f"got {student_logprobs.shape=} and {teacher_logprobs.shape=}."
        )
    event_mask = (teacher_logprobs.float() - student_logprobs.float()).abs() > high_gap_threshold
    return compute_event_heatmap_row(
        event_mask=event_mask,
        response_mask=response_mask,
        bin_size=bin_size,
        event_name="gap_abs_high",
    )


def compute_teacher_logprob_nonfinite_heatmap_row(
    *,
    teacher_logprob_nonfinite_positions: Optional[list[Any]],
    response_mask: torch.Tensor,
    bin_size: int,
) -> np.ndarray:
    """Return one heatmap row: nonfinite teacher scoring event rate per response-position bin."""
    event_mask = positions_to_event_mask(
        teacher_logprob_nonfinite_positions,
        response_mask,
        context="teacher_logprob_nonfinite",
    )
    return compute_event_heatmap_row(
        event_mask=event_mask,
        response_mask=response_mask,
        bin_size=bin_size,
        event_name="teacher_logprob_nonfinite",
    )


def compute_event_heatmap_row(
    *,
    event_mask: torch.Tensor,
    response_mask: torch.Tensor,
    bin_size: int,
    event_name: str,
) -> np.ndarray:
    """Return one heatmap row: event rate per response-position bin."""
    if event_mask.shape != response_mask.shape:
        raise ValueError(
            f"{event_name} and response_mask must align, got event_mask.shape={event_mask.shape} "
            f"and {response_mask.shape=}."
        )
    return compute_value_heatmap_row(
        values=event_mask.detach().float(),
        response_mask=response_mask,
        bin_size=bin_size,
        value_name=event_name,
    )


def compute_value_heatmap_row(
    *,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    bin_size: int,
    value_name: str,
) -> np.ndarray:
    """Return one heatmap row: mean value per response-position bin."""
    if bin_size <= 0:
        raise ValueError(f"bin_size must be positive, got {bin_size}.")
    if values.shape != response_mask.shape:
        raise ValueError(
            f"{value_name} and response_mask must align, got values.shape={values.shape} and {response_mask.shape=}."
        )
    values = _to_cpu_float(values)
    mask = response_mask.detach().bool().cpu()
    row = []
    for start in range(0, values.shape[1], bin_size):
        end = min(start + bin_size, values.shape[1])
        mean = _masked_mean(values[:, start:end], mask[:, start:end])
        row.append(np.nan if mean is None else mean)
    return np.asarray(row, dtype=np.float32)


def _as_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    elif isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, list | tuple):
        return []
    return [int(item) for item in value]


def _lcs_length(left: list[int], right: list[int]) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for left_item in left:
        current = [0]
        for idx, right_item in enumerate(right, start=1):
            if left_item == right_item:
                current.append(previous[idx - 1] + 1)
            else:
                current.append(max(previous[idx], current[-1]))
        previous = current
    return previous[-1]


def compute_rewrite_overlap_metrics(original_student_segment_ids: Any, teacher_segment_ids: Any) -> dict[str, float]:
    """Compare a replaced student segment with the teacher rewrite."""
    original_ids = _as_int_list(original_student_segment_ids)
    teacher_ids = _as_int_list(teacher_segment_ids)
    max_len = max(len(original_ids), len(teacher_ids), 1)
    min_len = min(len(original_ids), len(teacher_ids))
    exact_matches = sum(1 for idx in range(min_len) if original_ids[idx] == teacher_ids[idx])
    return {
        "rewrite_exact_position_overlap": exact_matches / max_len,
        "rewrite_lcs_ratio": _lcs_length(original_ids, teacher_ids) / max_len,
        "rewrite_length_ratio": min_len / max_len,
    }


def compute_intervention_metrics(
    *,
    intervention_counts: list[Any],
    interventions: list[Any],
    intervention_skipped: list[Any] | None = None,
    response_mask: torch.Tensor,
) -> dict[str, float]:
    """Aggregate teacher-intervention metadata collected during rollout."""
    counts = np.asarray([0 if value is None else int(value) for value in intervention_counts], dtype=np.float32)
    response_mask_bool = response_mask.detach().bool().cpu()
    valid_lengths_tensor = response_mask_bool.sum(dim=1)
    metrics = {
        "opd_intervention/rate": float((counts > 0).mean()) if counts.size else 0.0,
        "opd_intervention/count_mean": float(counts.mean()) if counts.size else 0.0,
        "opd_intervention/count_max": float(counts.max()) if counts.size else 0.0,
        "opd_intervention/zero_trainable_sample_fraction": float((valid_lengths_tensor == 0).float().mean().item())
        if valid_lengths_tensor.numel()
        else 0.0,
    }

    total_valid_tokens = int(valid_lengths_tensor.sum().item())
    metrics["opd_intervention/trainable_token_fraction"] = total_valid_tokens / max(response_mask.numel(), 1)
    teacher_tokens = 0
    suffix_tokens = 0
    first_positions = []
    first_position_fracs = []
    scores = []
    rewrite_exact_position_overlaps = []
    rewrite_lcs_ratios = []
    rewrite_length_ratios = []
    valid_lengths = valid_lengths_tensor.tolist()
    for sample_idx, sample_interventions in enumerate(interventions):
        if sample_interventions is None:
            continue
        for item_idx, item in enumerate(sample_interventions):
            if not isinstance(item, dict):
                continue
            teacher_tokens += int(item.get("teacher_tokens", 0) or 0)
            suffix_tokens += int(item.get("student_suffix_tokens", 0) or 0)
            score = item.get("score")
            if score is not None:
                scores.append(float(score))
            if {
                "rewrite_exact_position_overlap",
                "rewrite_lcs_ratio",
                "rewrite_length_ratio",
            }.issubset(item):
                rewrite_exact_position_overlaps.append(float(item["rewrite_exact_position_overlap"]))
                rewrite_lcs_ratios.append(float(item["rewrite_lcs_ratio"]))
                rewrite_length_ratios.append(float(item["rewrite_length_ratio"]))
            elif "original_student_segment_ids" in item and "teacher_segment_ids" in item:
                rewrite_metrics = compute_rewrite_overlap_metrics(
                    item["original_student_segment_ids"],
                    item["teacher_segment_ids"],
                )
                rewrite_exact_position_overlaps.append(rewrite_metrics["rewrite_exact_position_overlap"])
                rewrite_lcs_ratios.append(rewrite_metrics["rewrite_lcs_ratio"])
                rewrite_length_ratios.append(rewrite_metrics["rewrite_length_ratio"])
            if item_idx == 0:
                start = int(item.get("start", 0) or 0)
                first_positions.append(start)
                length = max(int(valid_lengths[sample_idx]), 1)
                first_position_fracs.append(start / length)

    denom = max(total_valid_tokens, 1)
    metrics["opd_intervention/teacher_token_fraction"] = teacher_tokens / denom
    metrics["opd_intervention/student_suffix_token_fraction"] = suffix_tokens / denom
    if first_positions:
        metrics["opd_intervention/first_position_mean"] = float(np.mean(first_positions))
        metrics["opd_intervention/first_position_frac_mean"] = float(np.mean(first_position_fracs))
    if scores:
        metrics["opd_intervention/gap_score_mean"] = float(np.mean(scores))
        metrics["opd_intervention/gap_score_max"] = float(np.max(scores))
    if rewrite_exact_position_overlaps:
        metrics["opd_intervention/rewrite_exact_position_overlap_mean"] = float(
            np.mean(rewrite_exact_position_overlaps)
        )
    if rewrite_lcs_ratios:
        metrics["opd_intervention/rewrite_lcs_ratio_mean"] = float(np.mean(rewrite_lcs_ratios))
    if rewrite_length_ratios:
        metrics["opd_intervention/rewrite_length_ratio_mean"] = float(np.mean(rewrite_length_ratios))

    skipped_counts = []
    skipped_reason_counts: dict[str, int] = {}
    for sample_skipped in intervention_skipped or []:
        if sample_skipped is None:
            skipped_counts.append(0)
            continue
        if isinstance(sample_skipped, dict):
            skipped_items = [sample_skipped]
        elif isinstance(sample_skipped, list | tuple):
            skipped_items = [item for item in sample_skipped if isinstance(item, dict)]
        else:
            skipped_items = []
        skipped_counts.append(len(skipped_items))
        for item in skipped_items:
            reason = item.get("reason")
            if reason is not None:
                skipped_reason_counts[str(reason)] = skipped_reason_counts.get(str(reason), 0) + 1
    skipped_array = np.asarray(skipped_counts, dtype=np.float32)
    metrics["opd_intervention/skipped_count_mean"] = float(skipped_array.mean()) if skipped_array.size else 0.0
    metrics["opd_intervention/skipped_count_max"] = float(skipped_array.max()) if skipped_array.size else 0.0
    metrics["opd_intervention/skipped_sample_fraction"] = (
        float((skipped_array > 0).mean()) if skipped_array.size else 0.0
    )
    skipped_denom = max(int(skipped_array.sum()), 1)
    for reason, count in skipped_reason_counts.items():
        metrics[f"opd_intervention/skipped_reason/{reason}"] = count / skipped_denom
    return metrics


@dataclass
class OpdDiagnosticsTracker:
    """Keeps a bounded history for OPD diagnostic heatmaps."""

    history_steps: int
    bin_size: int
    rows: deque[np.ndarray] = field(init=False)
    steps: deque[int] = field(init=False)

    def __post_init__(self):
        self.rows = deque(maxlen=self.history_steps)
        self.steps = deque(maxlen=self.history_steps)

    def add_entropy_row(self, step: int, row: np.ndarray) -> None:
        self.add_row(step, row)

    def add_gap_row(self, step: int, row: np.ndarray) -> None:
        self.add_row(step, row)

    def add_gap_high_row(self, step: int, row: np.ndarray) -> None:
        self.add_row(step, row)

    def add_teacher_logprob_nonfinite_row(self, step: int, row: np.ndarray) -> None:
        self.add_row(step, row)

    def add_row(self, step: int, row: np.ndarray) -> None:
        self.steps.append(step)
        self.rows.append(row)

    def render_entropy_heatmap(self):
        return self.render_heatmap(
            title="Student Entropy by Response Position",
            colorbar_label="Mean entropy",
        )

    def render_gap_heatmap(self):
        return self.render_heatmap(
            title="Teacher-Student Absolute Gap by Response Position",
            colorbar_label="Mean absolute logprob gap",
        )

    def render_gap_high_heatmap(self):
        return self.render_heatmap(
            title="High Teacher-Student Gap Rate by Response Position",
            colorbar_label="High-gap token rate",
        )

    def render_teacher_logprob_nonfinite_heatmap(self):
        return self.render_heatmap(
            title="Nonfinite Teacher Logprob Rate by Response Position",
            colorbar_label="Nonfinite teacher logprob rate",
        )

    def render_heatmap(self, *, title: str, colorbar_label: str):
        if not self.rows:
            return None

        try:
            import matplotlib
        except ImportError:
            return None

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        width = max(row.shape[0] for row in self.rows)
        heatmap = np.full((len(self.rows), width), np.nan, dtype=np.float32)
        for idx, row in enumerate(self.rows):
            heatmap[idx, : row.shape[0]] = row

        fig, ax = plt.subplots(figsize=(max(6.0, width * 0.8), max(3.0, len(self.rows) * 0.18)))
        image = ax.imshow(heatmap, aspect="auto", interpolation="nearest", origin="lower")
        ax.set_title(title)
        ax.set_xlabel("Response token position bin")
        ax.set_ylabel("Training step")
        ax.set_xticks(range(width))
        ax.set_xticklabels([f"{i * self.bin_size}-{(i + 1) * self.bin_size}" for i in range(width)], rotation=45)
        step_values = list(self.steps)
        if step_values:
            tick_count = min(8, len(step_values))
            tick_positions = np.linspace(0, len(step_values) - 1, tick_count, dtype=int)
            ax.set_yticks(tick_positions)
            ax.set_yticklabels([str(step_values[i]) for i in tick_positions])
        fig.colorbar(image, ax=ax, label=colorbar_label)
        fig.tight_layout()
        return fig
