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


def compute_opd_diagnostics_metrics(
    *,
    student_logprobs: torch.Tensor,
    response_mask: torch.Tensor,
    bin_size: int,
    teacher_logprobs: Optional[torch.Tensor] = None,
    teacher_logprob_nonfinite_positions: Optional[list[Any]] = None,
    entropy: Optional[torch.Tensor] = None,
    rm_scores: Optional[torch.Tensor] = None,
    high_gap_threshold: float = 1.0,
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


def compute_intervention_metrics(
    *,
    intervention_counts: list[Any],
    interventions: list[Any],
    response_mask: torch.Tensor,
) -> dict[str, float]:
    """Aggregate teacher-intervention metadata collected during rollout."""
    counts = np.asarray([0 if value is None else int(value) for value in intervention_counts], dtype=np.float32)
    metrics = {
        "opd_intervention/rate": float((counts > 0).mean()) if counts.size else 0.0,
        "opd_intervention/count_mean": float(counts.mean()) if counts.size else 0.0,
        "opd_intervention/count_max": float(counts.max()) if counts.size else 0.0,
    }

    total_valid_tokens = int(response_mask.detach().bool().cpu().sum().item())
    teacher_tokens = 0
    suffix_tokens = 0
    first_positions = []
    first_position_fracs = []
    scores = []
    valid_lengths = response_mask.detach().bool().cpu().sum(dim=1).tolist()
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
