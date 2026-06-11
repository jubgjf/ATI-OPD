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


def compute_opd_diagnostics_metrics(
    *,
    student_logprobs: torch.Tensor,
    response_mask: torch.Tensor,
    bin_size: int,
    teacher_logprobs: Optional[torch.Tensor] = None,
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
                high_gap_fraction = (abs_gap_values > high_gap_threshold).float().mean()
                metrics["opd/gap_abs/high_fraction"] = float(high_gap_fraction.item())
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
    if entropy.shape != response_mask.shape:
        raise ValueError(f"entropy and response_mask must align, got {entropy.shape=} and {response_mask.shape=}.")
    values = _to_cpu_float(entropy)
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
        self.steps.append(step)
        self.rows.append(row)

    def render_entropy_heatmap(self):
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
        ax.set_title("Student Entropy by Response Position")
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
        fig.colorbar(image, ax=ax, label="Mean entropy")
        fig.tight_layout()
        return fig
