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
"""CPU-only ranking diagnostics on existing, causally aligned top-k scores.

Candidate lists must be true descending top-k lists, not target-scoring slots.
Exact teacher-on-student scores, when available, must follow student candidate
IDs and must precede any loss clamping or replacement of missing scores.
"""

from __future__ import annotations

import time

import numpy as np
import torch

MAX_RANK_TOPK = 1024
_PAIR_ELEMENT_BUDGET = MAX_RANK_TOPK**2
_MAX_CHUNK_TOKENS = 4096
RANK_TEACHER_LOGPROBS_KEY = "opd_rank_teacher_on_student_logprobs"
_SCOPES = ("all", "student_only", "all_top1_agree", "student_only_top1_agree")
_METRICS = {
    "lists": ("top1_agreement", "top1_unique", "topk_overlap"),
    "intersection": ("candidate_count", "student_mass", "teacher_mass", "kendall_tau_b", "spearman_rho"),
    "student_topk_exact": (
        "kendall_tau_b",
        "spearman_rho",
        "student_mass",
        "teacher_mass",
        "tv_residual_bucket",
        "js_residual_bucket",
        "mass_roundoff_corrected",
    ),
}


def _rank_correlations(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise tau-b and average-rank Spearman; undefined rows remain NaN."""
    x = np.where(mask, x, 0.0)
    y = np.where(mask, y, 0.0)
    pairs = mask[:, :, None] & mask[:, None, :]
    dx = np.sign(x[:, :, None] - x[:, None, :]).astype(np.int8)
    dy = np.sign(y[:, :, None] - y[:, None, :]).astype(np.int8)
    upper = pairs & np.triu(np.ones((x.shape[1], x.shape[1]), dtype=bool), k=1)
    numerator = np.sum(dx * dy * upper, axis=(1, 2), dtype=np.float64)
    denominator = np.sqrt(
        np.sum((dx != 0) & upper, axis=(1, 2), dtype=np.float64)
        * np.sum((dy != 0) & upper, axis=(1, 2), dtype=np.float64)
    )
    tau = np.divide(numerator, denominator, out=np.full(len(x), np.nan), where=denominator > 0)

    n = mask.sum(axis=1, keepdims=True)
    rx = 1 + ((dx < 0) & pairs).sum(axis=2) + 0.5 * (((dx == 0) & pairs).sum(axis=2) - 1)
    ry = 1 + ((dy < 0) & pairs).sum(axis=2) + 0.5 * (((dy == 0) & pairs).sum(axis=2) - 1)
    rx = np.where(mask, rx - (n + 1) / 2, 0)
    ry = np.where(mask, ry - (n + 1) / 2, 0)
    denominator = np.sqrt(np.sum(rx * rx, axis=1) * np.sum(ry * ry, axis=1))
    rho = np.divide(np.sum(rx * ry, axis=1), denominator, out=np.full(len(x), np.nan), where=denominator > 0)
    return np.clip(tau, -1, 1), np.clip(rho, -1, 1)


def _valid_scores(logprobs: np.ndarray) -> np.ndarray:
    return (
        np.isfinite(logprobs).all(axis=1)
        & (logprobs <= 1e-6).all(axis=1)
        & (logprobs >= -1e4).all(axis=1)
        & (np.exp(np.minimum(logprobs, 0)).sum(axis=1) <= 1.001)
    )


def _valid_list(ids: np.ndarray, logprobs: np.ndarray) -> np.ndarray:
    sorted_ids = np.sort(ids, axis=1)
    return (
        _valid_scores(logprobs)
        & (ids >= 0).all(axis=1)
        & (sorted_ids[:, 1:] != sorted_ids[:, :-1]).all(axis=1)
        & (logprobs[:, :-1] >= logprobs[:, 1:]).all(axis=1)
    )


def _residual_distribution(logprobs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.exp(np.minimum(logprobs, 0))
    mass = probabilities.sum(axis=1, keepdims=True)
    # Only correct the small mass overflow admitted by _valid_scores. Do not
    # renormalize a truncated distribution whose missing mass is legitimate.
    probabilities = probabilities / np.maximum(mass, 1)
    tail = np.maximum(1 - mass, 0)
    return np.concatenate((probabilities, tail), axis=1), (mass[:, 0] > 1)


class _RankTotals:
    def __init__(self, topk_values: list[int], bin_count: int):
        self.eligible = {scope: np.zeros(bin_count, dtype=np.int64) for scope in _SCOPES}
        self.values = {
            (scope, k, basis, metric): (np.zeros(bin_count, dtype=np.float64), np.zeros(bin_count, dtype=np.int64))
            for scope in _SCOPES
            for k in topk_values
            for basis, metrics in _METRICS.items()
            for metric in metrics
        }

    def add(
        self,
        k: int,
        basis: str,
        metric: str,
        values: np.ndarray,
        bins: np.ndarray,
        scopes: dict[str, np.ndarray],
    ) -> None:
        for scope, eligible in scopes.items():
            valid = eligible & np.isfinite(values)
            if not valid.any():
                continue
            sums, counts = self.values[scope, k, basis, metric]
            sums += np.bincount(bins[valid], weights=values[valid], minlength=len(sums))
            counts += np.bincount(bins[valid], minlength=len(counts))

    def metrics(self, bin_size: int) -> dict[str, float]:
        result = {f"opd/rank/{scope}/token_count": float(counts.sum()) for scope, counts in self.eligible.items()}
        for (scope, k, basis, metric), (sums, counts) in self.values.items():
            prefix = f"opd/rank/{scope}/{basis}/top{k}/{metric}"
            total = float(sums.sum())
            count = int(counts.sum())
            eligible = int(self.eligible[scope].sum())
            result[prefix + "/sum"] = total
            result[prefix + "/count"] = float(count)
            result[prefix + "/eligible_count"] = float(eligible)
            result[prefix + "/unavailable_count"] = float(eligible - count)
            if count:
                result[prefix + "/mean"] = total / count
            if eligible:
                result[prefix + "/valid_fraction"] = count / eligible
            for index, (bin_sum, bin_count) in enumerate(zip(sums, counts, strict=True)):
                start = index * bin_size
                bin_prefix = f"{prefix}_bin/{start}_{start + bin_size}"
                result[bin_prefix + "/sum"] = float(bin_sum)
                result[bin_prefix + "/count"] = float(bin_count)
                result[bin_prefix + "/eligible_count"] = float(self.eligible[scope][index])
        return result


def compute_rank_diagnostics(
    *,
    response_mask: torch.Tensor,
    student_only_mask: torch.Tensor,
    student_topk_ids: torch.Tensor,
    student_topk_logprobs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    teacher_topk_logprobs: torch.Tensor,
    topk_values: list[int],
    bin_size: int,
    teacher_on_student_topk_logprobs: torch.Tensor | None = None,
    invalid_position_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Compare existing candidates without scoring models or changing training.

    All averages are token-weighted and carry additive position-bin totals.
    ``student_topk_exact`` is never reconstructed from incomplete teacher lists.
    TV and JS use student candidates plus a single residual bucket; JS is in nats.
    Conditional scopes require two visible, unequal head scores in both lists.
    """
    started = time.perf_counter()
    if response_mask.ndim != 2 or student_only_mask.shape != response_mask.shape:
        raise ValueError("Ranking response_mask and student_only_mask must have matching [batch, response_len] shapes.")
    if bin_size <= 0 or not topk_values or any(k <= 0 for k in topk_values):
        raise ValueError("Ranking bin_size and topk_values must be positive.")
    for name, ids, scores in (
        ("student", student_topk_ids, student_topk_logprobs),
        ("teacher", teacher_topk_ids, teacher_topk_logprobs),
    ):
        if ids.ndim != 3 or ids.shape != scores.shape or ids.shape[:2] != response_mask.shape or ids.shape[-1] == 0:
            raise ValueError(f"Ranking {name} IDs and logprobs must align as [batch, response_len, k].")
        if ids.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"Ranking {name} IDs must be integer token IDs.")
    if (
        teacher_on_student_topk_logprobs is not None
        and teacher_on_student_topk_logprobs.shape != student_topk_ids.shape
    ):
        raise ValueError("Exact teacher-on-student logprobs must follow all student top-k IDs with the same shape.")
    if invalid_position_mask is not None and invalid_position_mask.shape != response_mask.shape:
        raise ValueError("Ranking invalid_position_mask must align with response_mask.")

    response = response_mask.detach().bool().cpu().numpy()
    student_only = student_only_mask.detach().bool().cpu().numpy()
    if np.any(student_only & ~response):
        raise ValueError("Ranking student_only_mask must be a subset of response_mask.")
    available = min(student_topk_ids.shape[-1], teacher_topk_ids.shape[-1])
    ks = sorted({int(k) for k in topk_values if k <= available})
    width = response.shape[1]
    totals = _RankTotals(ks, (width + bin_size - 1) // bin_size)

    def flat(tensor: torch.Tensor, *, integer: bool = False) -> np.ndarray:
        cpu = tensor.detach().cpu()
        if not integer:
            cpu = cpu.float()
        return cpu.numpy().reshape(-1, tensor.shape[-1])

    student_ids = flat(student_topk_ids, integer=True)
    teacher_ids = flat(teacher_topk_ids, integer=True)
    student_lp = flat(student_topk_logprobs)
    teacher_lp = flat(teacher_topk_logprobs)
    exact_lp = None if teacher_on_student_topk_logprobs is None else flat(teacher_on_student_topk_logprobs)
    invalid = (
        np.zeros(response.size, dtype=bool)
        if invalid_position_mask is None
        else invalid_position_mask.detach().bool().cpu().numpy().reshape(-1)
    )
    positions = np.flatnonzero(response)
    largest_k = max(ks, default=1)
    if largest_k * largest_k > _PAIR_ELEMENT_BUDGET:
        raise ValueError("Ranking k exceeds the per-position pairwise memory budget; reduce diagnostics.topk.")
    chunk_size = max(1, min(_MAX_CHUNK_TOKENS, _PAIR_ELEMENT_BUDGET // (largest_k * largest_k)))
    for start in range(0, len(positions), chunk_size):
        indexes = positions[start : start + chunk_size]
        bins = (indexes % width) // bin_size
        # Head uniqueness needs the second candidate even when reporting k=1.
        sw = min(student_lp.shape[-1], max(largest_k, 2))
        tw = min(teacher_lp.shape[-1], max(largest_k, 2))
        si, ti = student_ids[indexes, :sw], teacher_ids[indexes, :tw]
        sl, tl = student_lp[indexes, :sw].astype(np.float64), teacher_lp[indexes, :tw].astype(np.float64)
        usable = ~invalid[indexes]
        head_valid = _valid_list(si[:, :1], sl[:, :1]) & _valid_list(ti[:, :1], tl[:, :1]) & usable
        agree = si[:, 0] == ti[:, 0]
        unique_known = np.zeros(len(indexes), dtype=bool)
        unique = np.zeros(len(indexes), dtype=bool)
        if sw >= 2 and tw >= 2:
            unique_known = _valid_list(si[:, :2], sl[:, :2]) & _valid_list(ti[:, :2], tl[:, :2]) & usable
            unique = unique_known & (sl[:, 0] > sl[:, 1]) & (tl[:, 0] > tl[:, 1])
        student_scope = student_only.reshape(-1)[indexes]
        scopes = {
            "all": np.ones(len(indexes), dtype=bool),
            "student_only": student_scope,
            "all_top1_agree": unique & agree,
            "student_only_top1_agree": student_scope & unique & agree,
        }
        for scope, eligible in scopes.items():
            totals.eligible[scope] += np.bincount(bins[eligible], minlength=len(totals.eligible[scope]))

        for k in ks:
            s, t = sl[:, :k], tl[:, :k]
            student_valid = _valid_list(si[:, :k], s) & usable
            list_valid = student_valid & _valid_list(ti[:, :k], t)
            matches = si[:, :k, None] == ti[:, None, :k]
            intersection = matches.any(axis=2) & list_valid[:, None]
            aligned_teacher = np.take_along_axis(t, matches.argmax(axis=2), axis=1)

            def add(
                basis: str,
                metric: str,
                values: np.ndarray,
                valid: np.ndarray | None = None,
                *,
                _k: int = k,
                _bins: np.ndarray = bins,
                _scopes: dict[str, np.ndarray] = scopes,
            ) -> None:
                if valid is not None:
                    values = np.where(valid, values, np.nan)
                totals.add(_k, basis, metric, values, _bins, _scopes)

            add("lists", "top1_agreement", agree, head_valid)
            add("lists", "top1_unique", unique, unique_known)
            size = intersection.sum(axis=1)
            add("lists", "topk_overlap", size / k, list_valid)
            add("intersection", "candidate_count", size, list_valid)
            add(
                "intersection",
                "student_mass",
                np.where(intersection, np.exp(np.minimum(s, 0)), 0).sum(axis=1),
                list_valid,
            )
            add(
                "intersection",
                "teacher_mass",
                np.where(intersection, np.exp(np.minimum(aligned_teacher, 0)), 0).sum(axis=1),
                list_valid,
            )
            tau, rho = _rank_correlations(s, aligned_teacher, intersection)
            add("intersection", "kendall_tau_b", tau)
            add("intersection", "spearman_rho", rho)

            if exact_lp is None:
                continue
            exact = exact_lp[indexes, :k].astype(np.float64)
            exact_valid = student_valid & _valid_scores(exact)
            mask = np.broadcast_to(exact_valid[:, None], s.shape)
            tau, rho = _rank_correlations(s, exact, mask)
            add("student_topk_exact", "kendall_tau_b", tau)
            add("student_topk_exact", "spearman_rho", rho)
            # Sanitize invalid rows before exp/log; they still have zero count.
            q, q_corrected = _residual_distribution(np.where(exact_valid[:, None], s, -np.inf))
            p, p_corrected = _residual_distribution(np.where(exact_valid[:, None], exact, -np.inf))
            midpoint = (p + q) / 2
            tiny = np.finfo(np.float64).tiny
            js = 0.5 * np.sum(
                p * (np.log(np.maximum(p, tiny)) - np.log(np.maximum(midpoint, tiny)))
                + q * (np.log(np.maximum(q, tiny)) - np.log(np.maximum(midpoint, tiny))),
                axis=1,
            )
            add("student_topk_exact", "student_mass", q[:, :-1].sum(axis=1), exact_valid)
            add("student_topk_exact", "teacher_mass", p[:, :-1].sum(axis=1), exact_valid)
            add("student_topk_exact", "tv_residual_bucket", np.abs(p - q).sum(axis=1) / 2, exact_valid)
            add("student_topk_exact", "js_residual_bucket", np.maximum(js, 0), exact_valid)
            add("student_topk_exact", "mass_roundoff_corrected", q_corrected | p_corrected, exact_valid)

    metrics = totals.metrics(bin_size)
    metrics["opd/rank/exact_scores_available"] = float(exact_lp is not None)
    metrics["opd/rank/available_topk"] = float(available)
    metrics["opd/rank/requested_topk"] = float(max(topk_values))
    metrics["opd/rank/compute_seconds"] = time.perf_counter() - started
    return metrics
