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

import math

import numpy as np
import pytest
import torch

from verl.trainer.distillation import rank_diagnostics as ranking
from verl.trainer.distillation.diagnostics import compute_opd_diagnostics_metrics


def _inputs(p=(0.5, 0.25, 0.15, 0.1), q=None, *, k=None, length=1):
    p, q = torch.tensor(p), torch.tensor(p if q is None else q)
    k = len(p) if k is None else k
    sl, si = q.log().topk(k)
    tl, ti = p.log().topk(k)

    def batch(tensor):
        return tensor[None, None, :].expand(1, length, -1).clone()

    mask = torch.ones((1, length), dtype=torch.bool)
    return dict(
        response_mask=mask,
        student_only_mask=mask.clone(),
        student_topk_ids=batch(si),
        student_topk_logprobs=batch(sl),
        teacher_topk_ids=batch(ti),
        teacher_topk_logprobs=batch(tl),
        teacher_on_student_topk_logprobs=batch(p.log()[si]),
        topk_values=[k],
        bin_size=2,
    )


def _key(metric, *, basis="student_topk_exact", scope="student_only", k=4, stat="mean"):
    return f"opd/rank/{scope}/{basis}/top{k}/{metric}/{stat}"


@pytest.mark.parametrize(
    "p,q,tau,rho",
    [
        ((0.5, 0.25, 0.15, 0.1), (0.5, 0.25, 0.15, 0.1), 1.0, 1.0),
        ((0.1, 0.15, 0.25, 0.5), (0.5, 0.25, 0.15, 0.1), -1.0, -1.0),
        ((0.5, 0.25, 0.15, 0.1), (0.5, 0.1, 0.15, 0.25), 0.0, 0.2),
        # Three concordant pairs; two x-only ties and one y-only tie.
        # Average ranks: (1.5, 1.5, 3.5, 3.5) and (1, 2.5, 2.5, 4).
        ((0.4, 0.4, 0.1, 0.1), (0.35, 0.25, 0.25, 0.15), 3 / math.sqrt(20), 1 / math.sqrt(2)),
    ],
)
def test_rank_correlations_match_hand_calculated_reference(p, q, tau, rho):
    inputs = _inputs(p, q)
    metrics = ranking.compute_rank_diagnostics(**inputs)
    for basis in ("intersection", "student_topk_exact"):
        assert metrics[_key("kendall_tau_b", basis=basis)] == pytest.approx(tau)
        assert metrics[_key("spearman_rho", basis=basis)] == pytest.approx(rho)
    p, q = np.asarray(p), np.asarray(q)
    midpoint = (p + q) / 2
    assert metrics[_key("tv_residual_bucket")] == pytest.approx(np.abs(p - q).sum() / 2, abs=1e-7)
    assert metrics[_key("js_residual_bucket")] == pytest.approx(
        (np.sum(p * np.log(p / midpoint)) + np.sum(q * np.log(q / midpoint))) / 2, abs=1e-7
    )


def test_same_top1_and_set_can_have_different_internal_ranks():
    inputs = _inputs(q=(0.5, 0.1, 0.15, 0.25))
    metrics = ranking.compute_rank_diagnostics(**inputs)
    assert metrics[_key("top1_agreement", basis="lists")] == 1
    assert metrics[_key("topk_overlap", basis="lists")] == 1
    assert metrics[_key("kendall_tau_b")] == 0
    assert metrics[_key("spearman_rho")] == pytest.approx(0.2)
    assert metrics[_key("tv_residual_bucket")] == pytest.approx(0.15)
    assert metrics[_key("kendall_tau_b", scope="student_only_top1_agree", stat="count")] == 1


def test_intersection_aligns_token_ids_and_is_separate_from_exact_scores():
    inputs = _inputs(p=(0.1, 0.4, 0.3, 0.2), q=(0.4, 0.35, 0.15, 0.1), k=3)
    metrics = ranking.compute_rank_diagnostics(**inputs)
    assert metrics[_key("topk_overlap", basis="lists", k=3)] == pytest.approx(2 / 3)
    assert metrics[_key("candidate_count", basis="intersection", k=3)] == 2
    assert metrics[_key("kendall_tau_b", basis="intersection", k=3)] == 1
    assert metrics[_key("kendall_tau_b", k=3)] == pytest.approx(-1 / 3)
    assert metrics[_key("student_mass", k=3)] == pytest.approx(0.9)
    assert metrics[_key("teacher_mass", k=3)] == pytest.approx(0.8)


def test_empty_intersection_and_missing_exact_scores_are_not_zero_correlations():
    inputs = _inputs()
    inputs["teacher_topk_ids"] += 20
    inputs["teacher_on_student_topk_logprobs"] = None
    metrics = ranking.compute_rank_diagnostics(**inputs)
    assert metrics[_key("topk_overlap", basis="lists")] == 0
    assert metrics[_key("candidate_count", basis="intersection")] == 0
    for basis in ("intersection", "student_topk_exact"):
        assert _key("kendall_tau_b", basis=basis) not in metrics
        assert metrics[_key("kendall_tau_b", basis=basis, stat="count")] == 0
        assert metrics[_key("kendall_tau_b", basis=basis, stat="unavailable_count")] == 1
        assert metrics[_key("kendall_tau_b", basis=basis, stat="valid_fraction")] == 0
    assert metrics["opd/rank/exact_scores_available"] == 0


@pytest.mark.parametrize("p", [(0.25, 0.25, 0.25, 0.25), (1.0,)])
def test_undefined_correlations_and_unknown_or_tied_top1(p):
    metrics = ranking.compute_rank_diagnostics(**_inputs(p))
    k = len(p)
    assert _key("kendall_tau_b", k=k) not in metrics
    assert _key("spearman_rho", k=k) not in metrics
    assert metrics[_key("kendall_tau_b", k=k, stat="count")] == 0
    assert metrics["opd/rank/student_only_top1_agree/token_count"] == 0
    assert metrics[_key("tv_residual_bucket", k=k)] == pytest.approx(0)


def test_residual_bucket_preserves_missing_mass_instead_of_renormalizing_topk():
    metrics = ranking.compute_rank_diagnostics(**_inputs(p=(0.4, 0.2, 0.2, 0.2), q=(0.6, 0.3, 0.05, 0.05), k=2))
    # Heads have the same normalized 2:1 ratio, but mass differs by 0.3.
    assert metrics[_key("tv_residual_bucket", k=2)] == pytest.approx(0.3, abs=1e-7)
    assert metrics[_key("js_residual_bucket", k=2)] > 0
    assert metrics[_key("teacher_mass", k=2)] == pytest.approx(0.6)
    assert metrics[_key("student_mass", k=2)] == pytest.approx(0.9)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), 5.0, -20000.0])
def test_invalid_exact_scores_have_counts_but_no_fabricated_mean(bad):
    inputs = _inputs(length=2)
    inputs["teacher_on_student_topk_logprobs"][0, 1, 2] = bad
    metrics = ranking.compute_rank_diagnostics(**inputs)
    assert metrics[_key("kendall_tau_b", stat="count")] == 1
    assert metrics[_key("kendall_tau_b", stat="valid_fraction")] == 0.5
    assert metrics[_key("kendall_tau_b")] == 1
    assert metrics[_key("kendall_tau_b", basis="intersection", stat="count")] == 2
    assert all(math.isfinite(value) for value in metrics.values())


def test_invalid_lists_events_and_duplicate_ids_are_excluded_with_denominators():
    inputs = _inputs(length=4)
    inputs["teacher_topk_ids"][0, 0, 1] = inputs["teacher_topk_ids"][0, 0, 0]
    inputs["student_topk_logprobs"][0, 1, 2] = float("nan")
    inputs["invalid_position_mask"] = torch.tensor([[False, False, True, False]])
    metrics = ranking.compute_rank_diagnostics(**inputs)
    assert metrics[_key("topk_overlap", basis="lists", stat="count")] == 1
    assert metrics[_key("topk_overlap", basis="lists", stat="eligible_count")] == 4
    assert metrics[_key("topk_overlap", basis="lists", stat="unavailable_count")] == 3


def test_zero_response_and_single_overlap_do_not_produce_correlation_means():
    inputs = _inputs()
    inputs["teacher_topk_ids"][0, 0, 1:] += 20
    metrics = ranking.compute_rank_diagnostics(**inputs)
    assert metrics[_key("candidate_count", basis="intersection")] == 1
    assert _key("kendall_tau_b", basis="intersection") not in metrics
    inputs["response_mask"][:] = False
    inputs["student_only_mask"][:] = False
    metrics = ranking.compute_rank_diagnostics(**inputs)
    assert metrics["opd/rank/all/token_count"] == 0
    assert not any(key.endswith("/mean") for key in metrics)


def test_diagnostics_wrapper_preserves_positions_and_loss_mask_excluded_students():
    inputs = _inputs(length=7)
    inputs["response_mask"][0, 6] = False
    payload = {
        key: value
        for key, value in inputs.items()
        if key not in {"response_mask", "student_only_mask", "topk_values", "bin_size"}
    }
    payload.update(topk=4, interventions=[[{"start": 1, "teacher_tokens": 1}, {"start": 3, "teacher_tokens": 1}]])
    args = dict(
        student_logprobs=torch.full((1, 7), -1.0),
        response_mask=inputs["response_mask"],
        bin_size=2,
        collect_repetition_metrics=False,
        teacher_logprob_nonfinite_positions=[[4]],
    )
    old = compute_opd_diagnostics_metrics(**args)
    metrics = compute_opd_diagnostics_metrics(**args, rank_inputs=payload)
    assert {key: value for key, value in metrics.items() if not key.startswith("opd/rank/")} == old
    assert metrics["opd/rank/all/token_count"] == 6
    assert metrics["opd/rank/student_only/token_count"] == 4
    prefix = _key("kendall_tau_b").removesuffix("/mean")
    # Student positions 0, 2, 4, 5 remain in place. Position 4 is an invalid
    # scoring event; position 5 remains eligible regardless of loss masking.
    assert metrics[prefix + "/count"] == 3
    for start in (0, 2, 4):
        assert metrics[f"{prefix}_bin/{start}_{start + 2}/count"] == 1
    assert sum(metrics[f"{prefix}_bin/{start}_{start + 2}/sum"] for start in (0, 2, 4, 6)) == metrics[prefix + "/sum"]


def test_chunking_is_equivalent_and_bounds_pairwise_memory(monkeypatch):
    inputs = _inputs(p=(0.4, 0.3, 0.2, 0.1), q=(0.35, 0.4, 0.1, 0.15), length=1025)
    inputs["topk_values"] = [1, 2, 4]
    expected = ranking.compute_rank_diagnostics(**inputs)
    original = ranking._rank_correlations
    shapes = []

    def checked(x, y, mask):
        shapes.append(x.shape)
        assert x.shape[0] * x.shape[1] ** 2 <= 128
        return original(x, y, mask)

    monkeypatch.setattr(ranking, "_PAIR_ELEMENT_BUDGET", 128)
    monkeypatch.setattr(ranking, "_rank_correlations", checked)
    actual = ranking.compute_rank_diagnostics(**inputs)
    assert len(shapes) > 100
    for key, value in expected.items():
        if key != "opd/rank/compute_seconds":
            assert actual[key] == pytest.approx(value, abs=1e-10)


def test_input_contracts_and_inputs_are_not_modified():
    inputs = _inputs()
    snapshots = {key: value.clone() for key, value in inputs.items() if isinstance(value, torch.Tensor)}
    ranking.compute_rank_diagnostics(**inputs)
    for key, value in snapshots.items():
        torch.testing.assert_close(inputs[key], value)
    inputs["teacher_on_student_topk_logprobs"] = torch.zeros((1, 1, 3))
    with pytest.raises(ValueError, match="same shape"):
        ranking.compute_rank_diagnostics(**inputs)
    inputs["teacher_on_student_topk_logprobs"] = None
    inputs["response_mask"][:] = False
    with pytest.raises(ValueError, match="subset"):
        ranking.compute_rank_diagnostics(**inputs)
