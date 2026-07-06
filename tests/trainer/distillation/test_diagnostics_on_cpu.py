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
    compute_binned_event_rates,
    compute_binned_token_means,
    compute_entropy_heatmap_row,
    compute_gap_heatmap_row,
    compute_gap_high_heatmap_row,
    compute_intervention_metrics,
    compute_opd_diagnostics_metrics,
    compute_overlap_diagnostics,
    compute_rethinking_opd_diagnostics,
    compute_teacher_logprob_nonfinite_heatmap_row,
    compute_topk_ladder,
    extract_response_topk_scores,
    extract_sampled_response_logprobs,
    positions_to_event_mask,
)
from verl.workers.config import (
    DistillationConfig,
    DistillationDiagnosticsConfig,
    DistillationLossConfig,
    DistillationTeacherModelConfig,
    RolloutConfig,
)


def test_compute_binned_token_means_respects_mask_on_cpu():
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0]], dtype=torch.bool)

    metrics = compute_binned_token_means(values, mask, bin_size=2, prefix="opd/test_bin")

    assert metrics["opd/test_bin/0000_0002"] == pytest.approx((1.0 + 2.0 + 10.0) / 3.0)
    assert metrics["opd/test_bin/0002_0004"] == 3.0


def test_compute_binned_event_rates_respects_mask_on_cpu():
    event_mask = torch.tensor([[0, 1, 1, 1], [1, 0, 1, 0]], dtype=torch.bool)
    response_mask = torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0]], dtype=torch.bool)

    metrics = compute_binned_event_rates(event_mask, response_mask, bin_size=2, prefix="opd/event_bin")

    assert metrics["opd/event_bin/0000_0002"] == pytest.approx(2 / 3)
    assert metrics["opd/event_bin/0002_0004"] == 1.0


def test_distillation_overlap_diagnostics_topk_defaults_to_loss_topk_on_cpu():
    config = DistillationConfig(
        enabled=True,
        n_gpus_per_node=1,
        nnodes=1,
        distillation_loss=DistillationLossConfig(loss_mode="k3", topk=64),
        diagnostics=DistillationDiagnosticsConfig(collect_topk_overlap=True, topk=None),
        teacher_models={
            "teacher_model": DistillationTeacherModelConfig(
                key="teacher",
                model_path="dummy",
                num_replicas=1,
                inference=RolloutConfig(
                    name="vllm",
                    prompt_length=4,
                    response_length=8,
                    tensor_model_parallel_size=1,
                    data_parallel_size=1,
                    pipeline_model_parallel_size=1,
                    engine_kwargs={"vllm": {}},
                ),
            )
        },
    )

    teacher_config = next(iter(config.teacher_models.values()))
    assert config.diagnostics.topk == 64
    assert teacher_config.inference.engine_kwargs["vllm"]["max_logprobs"] == 64


def test_distillation_rethinking_opd_diagnostics_requests_teacher_topk_on_cpu():
    config = DistillationConfig(
        enabled=True,
        n_gpus_per_node=1,
        nnodes=1,
        distillation_loss=DistillationLossConfig(loss_mode="k1", topk=16),
        diagnostics=DistillationDiagnosticsConfig(
            collect_topk_overlap=False,
            collect_rethinking_opd_metrics=True,
            topk=16,
        ),
        teacher_models={
            "teacher_model": DistillationTeacherModelConfig(
                key="teacher",
                model_path="dummy",
                num_replicas=1,
                inference=RolloutConfig(
                    name="vllm",
                    prompt_length=4,
                    response_length=8,
                    tensor_model_parallel_size=1,
                    data_parallel_size=1,
                    pipeline_model_parallel_size=1,
                    engine_kwargs={"vllm": {}},
                ),
            )
        },
    )

    teacher_config = next(iter(config.teacher_models.values()))
    assert teacher_config.inference.engine_kwargs["vllm"]["max_logprobs"] == 16


def test_compute_opd_diagnostics_metrics_logs_late_gap_and_entropy_on_cpu():
    student_logprobs = torch.tensor([[-1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0]])
    teacher_logprobs = torch.tensor([[-1.1, -1.1, -3.0, -3.0], [-1.0, -1.0, -1.0, -1.0]])
    entropy = torch.tensor([[0.1, 0.2, 1.2, 1.4], [0.2, 0.2, 0.3, 0.3]])
    response_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    rm_scores = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]])

    metrics = compute_opd_diagnostics_metrics(
        student_logprobs=student_logprobs,
        teacher_logprobs=teacher_logprobs,
        teacher_logprob_nonfinite_positions=[[2, 3], [1]],
        entropy=entropy,
        response_mask=response_mask,
        rm_scores=rm_scores,
        bin_size=2,
        high_gap_threshold=1.0,
    )

    assert metrics["opd/response_len/mean"] == 3.0
    assert metrics["opd/gap_abs_bin/0000_0002"] < metrics["opd/gap_abs_bin/0002_0004"]
    assert metrics["opd/entropy/tail_minus_head"] > 0
    assert metrics["opd/gap_abs/high_fraction"] == pytest.approx(2 / 6)
    assert metrics["opd/gap_abs/high_token_fraction"] == pytest.approx(2 / 6)
    assert metrics["opd/gap_abs/high_sample_fraction"] == pytest.approx(0.5)
    assert metrics["opd/gap_abs/high_count_mean"] == pytest.approx(1.0)
    assert metrics["opd/gap_abs/high_count_max"] == pytest.approx(2.0)
    assert metrics["opd/gap_abs/high_first_position_mean"] == pytest.approx(2.0)
    assert metrics["opd/gap_abs/high_first_position_frac_mean"] == pytest.approx(0.5)
    assert metrics["opd/gap_abs/high_bin/0000_0002"] == 0.0
    assert metrics["opd/gap_abs/high_bin/0002_0004"] == 1.0
    assert metrics["opd/teacher_logprob_nonfinite/response_token_fraction"] == pytest.approx(3 / 6)
    assert metrics["opd/teacher_logprob_nonfinite/sample_fraction"] == pytest.approx(1.0)
    assert metrics["opd/teacher_logprob_nonfinite/count_mean"] == pytest.approx(1.5)
    assert metrics["opd/teacher_logprob_nonfinite/count_max"] == pytest.approx(2.0)
    assert metrics["opd/teacher_logprob_nonfinite/first_position_mean"] == pytest.approx(1.5)
    assert metrics["opd/teacher_logprob_nonfinite/first_position_frac_mean"] == pytest.approx(0.5)
    assert metrics["opd/teacher_logprob_nonfinite_bin/0000_0002"] == pytest.approx(1 / 4)
    assert metrics["opd/teacher_logprob_nonfinite_bin/0002_0004"] == 1.0
    assert "opd/reward/correct_auc" in metrics


def test_positions_to_event_mask_handles_missing_positions_on_cpu():
    response_mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)

    event_mask = positions_to_event_mask(None, response_mask, context="teacher_logprob_nonfinite")
    assert not event_mask.any()

    event_mask = positions_to_event_mask([[1, 2], []], response_mask, context="teacher_logprob_nonfinite")
    torch.testing.assert_close(event_mask, torch.tensor([[0, 1, 0], [0, 0, 0]], dtype=torch.bool))

    with pytest.raises(ValueError, match="outside response width"):
        positions_to_event_mask([[3], []], response_mask, context="teacher_logprob_nonfinite")


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


def test_opd_diagnostics_filters_sentinel_outliers_on_cpu():
    student_logprobs = torch.tensor([[-1.0, -1.0, -1.0, -1.0]])
    teacher_logprobs = torch.tensor([[-1.2, -1.0e38, -2.0, -3.0]])
    entropy = torch.tensor([[0.1, 1.0e38, 0.3, 0.5]])
    response_mask = torch.ones((1, 4), dtype=torch.bool)

    metrics = compute_opd_diagnostics_metrics(
        student_logprobs=student_logprobs,
        teacher_logprobs=teacher_logprobs,
        entropy=entropy,
        response_mask=response_mask,
        bin_size=2,
        high_gap_threshold=0.5,
    )

    assert metrics["opd/gap_abs/outlier_token_fraction"] == pytest.approx(1 / 4)
    assert metrics["opd/gap_abs/mean"] == pytest.approx((0.2 + 1.0 + 2.0) / 3)
    assert metrics["opd/gap_abs_bin/0000_0002"] == pytest.approx(0.2)
    assert metrics["opd/gap_abs_bin/0002_0004"] == pytest.approx(1.5)
    assert metrics["opd/gap_abs/high_bin/0000_0002"] == pytest.approx(0.0)
    assert metrics["opd/gap_abs/high_bin/0002_0004"] == pytest.approx(1.0)
    assert metrics["opd/entropy/outlier_token_fraction"] == pytest.approx(1 / 4)
    assert metrics["opd/entropy/mean"] == pytest.approx((0.1 + 0.3 + 0.5) / 3)

    entropy_row = compute_entropy_heatmap_row(entropy, response_mask, bin_size=2)
    gap_row = compute_gap_heatmap_row(
        student_logprobs=student_logprobs,
        teacher_logprobs=teacher_logprobs,
        response_mask=response_mask,
        bin_size=2,
    )
    high_gap_row = compute_gap_high_heatmap_row(
        student_logprobs=student_logprobs,
        teacher_logprobs=teacher_logprobs,
        response_mask=response_mask,
        bin_size=2,
        high_gap_threshold=0.5,
    )

    np.testing.assert_allclose(entropy_row, np.array([0.1, 0.4], dtype=np.float32))
    np.testing.assert_allclose(gap_row, np.array([0.2, 1.5], dtype=np.float32), rtol=1e-6)
    np.testing.assert_allclose(high_gap_row, np.array([0.0, 1.0], dtype=np.float32))


def test_compute_gap_heatmap_row_and_tracker_on_cpu():
    student_logprobs = torch.tensor([[-1.0, -1.0, -1.0, -1.0]])
    teacher_logprobs = torch.tensor([[[-1.0], [-2.0], [-3.0], [-5.0]]])
    response_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)

    row = compute_gap_heatmap_row(
        student_logprobs=student_logprobs,
        teacher_logprobs=teacher_logprobs,
        response_mask=response_mask,
        bin_size=2,
    )
    np.testing.assert_allclose(row, np.array([0.5, 2.0], dtype=np.float32))

    tracker = OpdDiagnosticsTracker(history_steps=2, bin_size=2)
    tracker.add_gap_row(1, row)
    fig = tracker.render_gap_heatmap()
    if fig is not None:
        fig.clear()


def test_compute_event_heatmap_rows_on_cpu():
    student_logprobs = torch.tensor([[-1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0]])
    teacher_logprobs = torch.tensor([[-1.0, -3.0, -1.0, -4.0], [-1.0, -1.0, -1.0, -1.0]])
    response_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)

    high_gap_row = compute_gap_high_heatmap_row(
        student_logprobs=student_logprobs,
        teacher_logprobs=teacher_logprobs,
        response_mask=response_mask,
        bin_size=2,
        high_gap_threshold=1.0,
    )
    np.testing.assert_allclose(high_gap_row, np.array([1 / 4, 0.5], dtype=np.float32))

    nonfinite_row = compute_teacher_logprob_nonfinite_heatmap_row(
        teacher_logprob_nonfinite_positions=[[1, 3], []],
        response_mask=response_mask,
        bin_size=2,
    )
    np.testing.assert_allclose(nonfinite_row, np.array([1 / 4, 0.5], dtype=np.float32))

    tracker = OpdDiagnosticsTracker(history_steps=2, bin_size=2)
    tracker.add_gap_high_row(1, high_gap_row)
    tracker.add_teacher_logprob_nonfinite_row(2, nonfinite_row)
    fig = tracker.render_gap_high_heatmap()
    if fig is not None:
        fig.clear()
    fig = tracker.render_teacher_logprob_nonfinite_heatmap()
    if fig is not None:
        fig.clear()


def test_compute_topk_ladder_on_cpu():
    assert compute_topk_ladder(64) == [1, 2, 4, 8, 16, 32, 64]
    assert compute_topk_ladder(20) == [1, 2, 4, 8, 16, 20]


def test_compute_overlap_diagnostics_topk_rates_and_sampled_prob_on_cpu():
    response_ids = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]])
    response_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    teacher_topk_ids = torch.tensor(
        [
            [[10, 99, 98, 97], [99, 11, 98, 97], [99, 98, 97, 12], [13, 99, 98, 97]],
            [[99, 98, 20, 97], [99, 98, 97, 96], [22, 99, 98, 97], [23, 99, 98, 97]],
        ]
    )
    teacher_topk_logprobs = torch.log(
        torch.tensor(
            [
                [[0.5, 0.2, 0.1, 0.05], [0.4, 0.3, 0.1, 0.05], [0.4, 0.2, 0.1, 0.05], [0.5, 0.2, 0.1, 0.05]],
                [[0.4, 0.2, 0.1, 0.05], [0.4, 0.2, 0.1, 0.05], [0.5, 0.2, 0.1, 0.05], [0.5, 0.2, 0.1, 0.05]],
            ]
        )
    )
    sampled_teacher_logprobs = torch.log(torch.tensor([[0.50, 0.30, 0.05, 0.50], [0.01, 0.02, 0.50, 0.50]]))

    metrics = compute_overlap_diagnostics(
        response_ids=response_ids,
        response_mask=response_mask,
        teacher_topk_ids=teacher_topk_ids,
        teacher_topk_logprobs=teacher_topk_logprobs,
        sampled_teacher_logprobs=sampled_teacher_logprobs,
        bin_size=2,
        topk=4,
    )

    assert metrics["opd/overlap/top1_rate"] == pytest.approx(1 / 5)
    assert metrics["opd/overlap/top2_rate"] == pytest.approx(2 / 5)
    assert metrics["opd/overlap/top4_rate"] == pytest.approx(4 / 5)
    assert metrics["opd/overlap/top4_bin/0000_0002"] == pytest.approx(3 / 4)
    assert metrics["opd/overlap/top4_bin/0002_0004"] == 1.0
    assert metrics["opd/overlap/top4/head_mean"] == pytest.approx(3 / 4)
    assert metrics["opd/overlap/top4/tail_mean"] == 1.0
    assert metrics["opd/overlap/teacher_sampled_prob_mean"] == pytest.approx((0.50 + 0.30 + 0.05 + 0.01 + 0.02) / 5)
    assert metrics["opd/overlap/teacher_sampled_prob_p50"] == pytest.approx(0.05)
    assert metrics["opd/overlap/teacher_topk_mass_mean"] == pytest.approx(0.79)


def test_compute_overlap_diagnostics_skips_unavailable_k_on_cpu():
    response_ids = torch.tensor([[3, 4, 5]])
    response_mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    teacher_topk_ids = torch.tensor([[[3, 9], [9, 4], [5, 9]]])
    teacher_topk_logprobs = torch.log(torch.tensor([[[0.6, 0.2], [0.6, 0.2], [0.6, 0.2]]]))

    metrics = compute_overlap_diagnostics(
        response_ids=response_ids,
        response_mask=response_mask,
        teacher_topk_ids=teacher_topk_ids,
        teacher_topk_logprobs=teacher_topk_logprobs,
        sampled_teacher_logprobs=torch.log(torch.tensor([[0.6, 0.01, 0.6]])),
        bin_size=2,
        topk=4,
    )

    assert metrics["opd/overlap/top1_rate"] == 0.5
    assert metrics["opd/overlap/top2_rate"] == 1.0
    assert "opd/overlap/top4_rate" not in metrics
    assert metrics["opd/overlap/teacher_sampled_prob_mean"] == pytest.approx((0.6 + 0.01) / 2)


def test_compute_rethinking_opd_diagnostics_logs_support_intersection_metrics_on_cpu():
    response_mask = torch.tensor([[1, 1]], dtype=torch.bool)
    student_topk_ids = torch.tensor([[[1, 2, 3, 4], [10, 11, 12, 13]]])
    teacher_topk_ids = torch.tensor([[[2, 3, 5, 6], [14, 15, 16, 17]]])
    student_topk_logprobs = torch.log(torch.tensor([[[0.40, 0.30, 0.20, 0.05], [0.40, 0.30, 0.20, 0.05]]]))
    teacher_topk_logprobs = torch.log(torch.tensor([[[0.50, 0.25, 0.10, 0.05], [0.50, 0.25, 0.10, 0.05]]]))

    metrics = compute_rethinking_opd_diagnostics(
        response_mask=response_mask,
        student_topk_ids=student_topk_ids,
        student_topk_logprobs=student_topk_logprobs,
        teacher_topk_ids=teacher_topk_ids,
        teacher_topk_logprobs=teacher_topk_logprobs,
        bin_size=1,
        topk=4,
    )

    expected_advantage = (0.6 * np.log((2 / 3) / 0.6) + 0.4 * np.log((1 / 3) / 0.4)) / 2
    assert metrics["distillation/overlap_ratio"] == pytest.approx(0.25)
    assert metrics["distillation/overlap_mass_student"] == pytest.approx(0.25)
    assert metrics["distillation/overlap_mass_teacher"] == pytest.approx(0.375)
    assert metrics["distillation/empty_overlap_fraction"] == pytest.approx(0.5)
    assert metrics["distillation/overlap_token_advantage"] == pytest.approx(expected_advantage, rel=1e-5)
    assert "distillation/entropy_gap" in metrics
    assert metrics["distillation/entropy_gap_residual_bucket_approximation"] == 1.0
    assert metrics["distillation/overlap_ratio_top1"] == pytest.approx(0.0)


def test_extract_response_topk_scores_from_padded_teacher_scoring_on_cpu():
    response_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
    sequence_ids = torch.arange(12).reshape(1, 6, 2)
    sequence_logprobs = torch.arange(12, dtype=torch.float32).reshape(1, 6, 2)

    response_ids, response_logprobs = extract_response_topk_scores(
        sequence_ids=sequence_ids,
        sequence_logprobs=sequence_logprobs,
        response_mask=response_mask,
        prompt_width=3,
        context="teacher_logprobs",
    )

    torch.testing.assert_close(response_ids, sequence_ids[:, 2:6])
    torch.testing.assert_close(response_logprobs, sequence_logprobs[:, 2:6])


def test_extract_sampled_response_logprobs_from_padded_teacher_scoring_on_cpu():
    response_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
    sequence_logprobs = torch.tensor([[0.0, -0.1, -1.0, -2.0, -3.0, 0.0]]).unsqueeze(-1)

    response_logprobs = extract_sampled_response_logprobs(
        sequence_logprobs=sequence_logprobs,
        response_mask=response_mask,
        prompt_width=3,
        context="teacher_logprobs",
    )

    assert response_logprobs is not None
    torch.testing.assert_close(response_logprobs, torch.tensor([[-1.0, -2.0, -3.0, 0.0]]))


def test_extract_sampled_response_logprobs_skips_topk_and_errors_on_short_sequence_on_cpu():
    response_mask = torch.tensor([[1, 1]], dtype=torch.bool)
    topk_logprobs = torch.zeros((1, 4, 2))

    assert (
        extract_sampled_response_logprobs(
            sequence_logprobs=topk_logprobs,
            response_mask=response_mask,
            prompt_width=2,
            context="teacher_logprobs",
        )
        is None
    )

    with pytest.raises(RuntimeError, match="Cannot align teacher_logprobs"):
        extract_sampled_response_logprobs(
            sequence_logprobs=torch.zeros((1, 3, 1)),
            response_mask=response_mask,
            prompt_width=3,
            context="teacher_logprobs",
        )


def test_compute_intervention_metrics_on_cpu():
    counts = [0, 2]
    interventions = [
        [],
        [
            {
                "start": 4,
                "score": 1.5,
                "teacher_tokens": 2,
                "student_suffix_tokens": 8,
                "rewrite_exact_position_overlap": 0.5,
                "rewrite_lcs_ratio": 0.75,
                "rewrite_length_ratio": 1.0,
            },
            {
                "start": 16,
                "score": 2.0,
                "teacher_tokens": 2,
                "student_suffix_tokens": 4,
                "original_student_segment_ids": [1, 2, 3],
                "teacher_segment_ids": [2, 3],
            },
        ],
    ]
    skipped = [
        [],
        [{"reason": "insufficient_trainable_tokens", "attempt": 0, "candidate_trainable_tokens": 0}],
    ]
    response_mask = torch.ones((2, 20), dtype=torch.bool)

    metrics = compute_intervention_metrics(
        intervention_counts=counts,
        interventions=interventions,
        intervention_skipped=skipped,
        response_mask=response_mask,
    )

    assert metrics["opd_intervention/rate"] == 0.5
    assert metrics["opd_intervention/count_mean"] == 1.0
    assert metrics["opd_intervention/teacher_token_fraction"] == 4 / 40
    assert metrics["opd_intervention/student_suffix_token_fraction"] == 12 / 40
    assert metrics["opd_intervention/first_position_mean"] == 4
    assert metrics["opd_intervention/zero_trainable_sample_fraction"] == 0.0
    assert metrics["opd_intervention/trainable_token_fraction"] == 1.0
    assert metrics["opd_intervention/skipped_count_mean"] == 0.5
    assert metrics["opd_intervention/skipped_count_max"] == 1.0
    assert metrics["opd_intervention/skipped_sample_fraction"] == 0.5
    assert metrics["opd_intervention/skipped_reason/insufficient_trainable_tokens"] == 1.0
    assert metrics["opd_intervention/rewrite_exact_position_overlap_mean"] == pytest.approx(0.25)
    assert metrics["opd_intervention/rewrite_lcs_ratio_mean"] == pytest.approx((0.75 + 2 / 3) / 2)
    assert metrics["opd_intervention/rewrite_length_ratio_mean"] == pytest.approx((1.0 + 2 / 3) / 2)


def test_compute_intervention_metrics_counts_zero_trainable_samples_on_cpu():
    response_mask = torch.tensor([[0, 0, 0], [1, 0, 0]], dtype=torch.bool)

    metrics = compute_intervention_metrics(
        intervention_counts=[0, 0],
        interventions=[[], []],
        intervention_skipped=[[], []],
        response_mask=response_mask,
    )

    assert metrics["opd_intervention/zero_trainable_sample_fraction"] == 0.5
    assert metrics["opd_intervention/trainable_token_fraction"] == 1 / 6
