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

import numpy as np
import pytest
import torch

from verl.trainer.distillation.diagnostics import (
    OpdDiagnosticsTracker,
    binary_auc,
    compute_binned_token_means,
    compute_entropy_heatmap_row,
    compute_intervention_metrics,
    compute_opd_diagnostics_metrics,
)


def test_compute_binned_token_means_respects_mask_on_cpu():
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0]], dtype=torch.bool)

    metrics = compute_binned_token_means(values, mask, bin_size=2, prefix="opd/test_bin")

    assert metrics["opd/test_bin/0000_0002"] == pytest.approx((1.0 + 2.0 + 10.0) / 3.0)
    assert metrics["opd/test_bin/0002_0004"] == 3.0


def test_compute_opd_diagnostics_metrics_logs_late_gap_and_entropy_on_cpu():
    student_logprobs = torch.tensor([[-1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0]])
    teacher_logprobs = torch.tensor([[-1.1, -1.1, -3.0, -3.0], [-1.0, -1.0, -1.0, -1.0]])
    entropy = torch.tensor([[0.1, 0.2, 1.2, 1.4], [0.2, 0.2, 0.3, 0.3]])
    response_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    rm_scores = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]])

    metrics = compute_opd_diagnostics_metrics(
        student_logprobs=student_logprobs,
        teacher_logprobs=teacher_logprobs,
        entropy=entropy,
        response_mask=response_mask,
        rm_scores=rm_scores,
        bin_size=2,
        high_gap_threshold=1.0,
    )

    assert metrics["opd/response_len/mean"] == 3.0
    assert metrics["opd/gap_abs_bin/0000_0002"] < metrics["opd/gap_abs_bin/0002_0004"]
    assert metrics["opd/entropy/tail_minus_head"] > 0
    assert metrics["opd/gap_abs/high_fraction"] > 0
    assert "opd/reward/correct_auc" in metrics


def test_binary_auc_handles_ties_and_single_class_on_cpu():
    assert binary_auc(torch.tensor([0.0, 1.0]), torch.tensor([False, True])) == 1.0
    assert binary_auc(torch.tensor([0.0, 1.0]), torch.tensor([True, True])) is None
    assert binary_auc(torch.tensor([1.0, 1.0]), torch.tensor([False, True])) == 0.5


def test_compute_entropy_heatmap_row_and_tracker_on_cpu():
    entropy = torch.tensor([[0.0, 2.0, 4.0, 6.0]])
    response_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)

    row = compute_entropy_heatmap_row(entropy, response_mask, bin_size=2)
    np.testing.assert_allclose(row, np.array([1.0, 4.0], dtype=np.float32))

    tracker = OpdDiagnosticsTracker(history_steps=2, bin_size=2)
    tracker.add_entropy_row(1, row)
    tracker.add_entropy_row(2, np.array([2.0, 3.0, 4.0], dtype=np.float32))
    tracker.add_entropy_row(3, np.array([5.0], dtype=np.float32))

    assert list(tracker.steps) == [2, 3]
    fig = tracker.render_entropy_heatmap()
    if fig is not None:
        fig.clear()


def test_compute_intervention_metrics_on_cpu():
    counts = [0, 2]
    interventions = [
        [],
        [
            {"start": 4, "score": 1.5, "teacher_tokens": 2, "student_suffix_tokens": 8},
            {"start": 16, "score": 2.0, "teacher_tokens": 2, "student_suffix_tokens": 4},
        ],
    ]
    response_mask = torch.ones((2, 20), dtype=torch.bool)

    metrics = compute_intervention_metrics(
        intervention_counts=counts,
        interventions=interventions,
        response_mask=response_mask,
    )

    assert metrics["opd_intervention/rate"] == 0.5
    assert metrics["opd_intervention/count_mean"] == 1.0
    assert metrics["opd_intervention/teacher_token_fraction"] == 4 / 40
    assert metrics["opd_intervention/student_suffix_token_fraction"] == 12 / 40
    assert metrics["opd_intervention/first_position_mean"] == 4
