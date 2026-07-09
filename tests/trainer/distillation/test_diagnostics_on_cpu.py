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
    SpecialTokenIds,
    binary_auc,
    compute_binned_event_rates,
    compute_binned_token_means,
    compute_bucket_denominator_metrics,
    compute_coarse_token_stats,
    compute_entropy_heatmap_row,
    compute_gap_heatmap_row,
    compute_gap_high_heatmap_row,
    compute_intervention_metrics,
    compute_opd_diagnostics_metrics,
    compute_overlap_diagnostics,
    compute_phase_and_special_token_metrics,
    compute_rethinking_opd_diagnostics,
    compute_teacher_logprob_nonfinite_heatmap_row,
    compute_topk_ladder,
    extract_response_topk_scores,
    extract_sampled_response_logprobs,
    positions_to_event_mask,
    resolve_special_token_ids,
)
from verl.trainer.ppo.metric_utils import _response_length_tail_metrics
from verl.workers.config import (
    DistillationConfig,
    DistillationDiagnosticsConfig,
    DistillationLossConfig,
    DistillationTeacherModelConfig,
    RolloutConfig,
)


class _DummyQwenTokenizer:
    eos_token_id = 151645

    _token_to_id = {
        "<|im_start|>": 151644,
        "<|im_end|>": 151645,
        "<|endoftext|>": 151643,
        "<think>": 151667,
        "</think>": 151668,
        "<tool_call>": 151657,
        "</tool_call>": 151658,
    }

    def convert_tokens_to_ids(self, token):
        return self._token_to_id.get(token, -1)

    def encode(self, text, add_special_tokens=False):
        token_id = self._token_to_id.get(text)
        return [] if token_id is None else [token_id]


class _DummyDeepSeekStyleTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 2
    bos_token = "<｜begin▁of▁sentence｜>"
    eos_token = "<｜end▁of▁sentence｜>"
    pad_token = "<｜end▁of▁sentence｜>"

    _token_to_id = {
        "<｜begin▁of▁sentence｜>": 1,
        "<｜end▁of▁sentence｜>": 2,
        "<｜User｜>": 10,
        "<｜Assistant｜>": 11,
        "<|EOT|>": 12,
    }
    _text_to_ids = {
        "<think>": [101, 102],
        "</think>": [103, 104],
    }

    def get_vocab(self):
        return dict(self._token_to_id)

    def convert_tokens_to_ids(self, token):
        return self._token_to_id.get(token, -1)

    def encode(self, text, add_special_tokens=False):
        if text in self._text_to_ids:
            return list(self._text_to_ids[text])
        token_id = self._token_to_id.get(text)
        return [] if token_id is None else [token_id]


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


def test_resolve_special_token_ids_finds_qwen3_tokens_on_cpu():
    special = resolve_special_token_ids(_DummyQwenTokenizer())

    assert special.im_end == 151645
    assert special.endoftext == 151643
    assert special.think_start == 151667
    assert special.think_end == 151668
    assert special.think_start_sequence == (151667,)
    assert special.think_end_sequence == (151668,)
    assert 151645 in special.generation_stop_ids
    assert 151657 in special.foreign_control_ids


def test_resolve_special_token_ids_handles_deepseek_style_multitoken_think_on_cpu():
    special = resolve_special_token_ids(_DummyDeepSeekStyleTokenizer())

    assert special.endoftext == 2
    assert special.eot == 12
    assert special.user_role == 10
    assert special.assistant_role == 11
    assert special.bos == 1
    assert special.think_start is None
    assert special.think_end is None
    assert special.think_start_sequence == (101, 102)
    assert special.think_end_sequence == (103, 104)
    assert (2,) in special.generation_stop_sequences
    assert 10 in special.foreign_control_ids
    assert 11 in special.foreign_control_ids


def test_bucket_denominators_and_coarse_empty_denominator_on_cpu():
    response_mask = torch.zeros((2, 20001), dtype=torch.bool)
    response_mask[0, :512] = True
    response_mask[1, :16000] = True

    metrics = compute_bucket_denominator_metrics(response_mask, bin_size=512)

    assert metrics["bucket/bin/0000_0512/token_count"] == 1024
    assert metrics["bucket/bin/0000_0512/sample_count"] == 2
    assert metrics["bucket/bin/0512_1024/token_count"] == 512
    assert metrics["bucket/coarse/16k_20k/token_count"] == 0
    assert metrics["bucket/coarse/16k_20k/sample_count"] == 0

    values = torch.ones_like(response_mask, dtype=torch.float32)
    coarse = compute_coarse_token_stats(values, response_mask, prefix="opd/test")

    assert coarse["opd/test/coarse/16k_20k/token_count"] == 0
    assert "opd/test/coarse/16k_20k/mean" not in coarse


def test_response_length_tail_metrics_separates_nonclipped_main_mass_on_cpu():
    lengths = torch.tensor([7000.0, 6800.0, 23552.0, 6900.0])

    metrics = _response_length_tail_metrics(lengths, max_response_length=23552)

    assert metrics["response_length/clipped_count"] == 1
    assert metrics["response_length/nonclipped_count"] == 3
    assert metrics["response_length/nonclipped_mean"] == pytest.approx((7000 + 6800 + 6900) / 3)
    assert metrics["response_length/main_mass_mean"] == pytest.approx((7000 + 6800 + 6900) / 3)
    assert metrics["response_length/clipped_token_share"] == pytest.approx(23552 / lengths.sum().item())
    assert metrics["response_length/survival_gt_16k"] == pytest.approx(0.25)
    assert metrics["response_length/clip_given_gt_16k"] == pytest.approx(1.0)


def test_response_length_tail_metrics_omits_main_mass_when_all_clipped_on_cpu():
    metrics = _response_length_tail_metrics(torch.tensor([512.0, 512.0]), max_response_length=512)

    assert metrics["response_length/clipped_count"] == 2
    assert metrics["response_length/nonclipped_count"] == 0
    assert "response_length/main_mass_mean" not in metrics


def test_phase_parser_handles_qwen_think_cases_and_loss_mask_on_cpu():
    special = SpecialTokenIds(
        im_end=151645,
        endoftext=151643,
        think_start=151667,
        think_end=151668,
        im_start=151644,
        generation_stop_ids=(151645, 151643),
    )
    responses = torch.tensor(
        [
            [151667, 10, 151668, 20, 151645, 0],
            [151667, 10, 11, 12, 13, 14],
            [20, 151645, 21, 0, 0, 0],
            [151668, 20, 151645, 0, 0, 0],
            [151667, 10, 151668, 151667, 20, 0],
        ]
    )
    response_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 0],
        ],
        dtype=torch.bool,
    )

    info = compute_phase_and_special_token_metrics(
        response_ids=responses,
        response_mask=response_mask,
        special_token_ids=special,
        max_response_length=6,
    )

    assert info.metrics["format/has_think_end_ratio"] == pytest.approx(3 / 5)
    assert info.metrics["format/unclosed_think_ratio"] == pytest.approx(1 / 5)
    assert info.metrics["format/think_end_without_think_start_ratio"] == pytest.approx(1 / 5)
    assert info.metrics["format/think_reopened_after_think_end_ratio"] == pytest.approx(1 / 5)
    assert info.metrics["format/clipped_without_think_end_ratio"] == pytest.approx(1 / 5)
    assert info.metrics["loss_mask/generation_stop_included_count"] == 3
    assert info.metrics["response_end/ended_by_generation_stop_count"] == 2
    assert info.metrics["loss_mask/think_end_included_count"] == 3
    assert info.thinking_mask[0, :3].all()
    assert info.answering_mask[0, 3:5].all()


def test_phase_parser_handles_deepseek_style_multitoken_think_on_cpu():
    special = resolve_special_token_ids(_DummyDeepSeekStyleTokenizer())
    responses = torch.tensor(
        [
            [101, 102, 9, 103, 104, 20, 2, 0],
            [101, 102, 9, 12, 13, 14, 15, 16],
            [20, 103, 104, 2, 0, 0, 0, 0],
            [101, 102, 9, 103, 104, 101, 102, 2],
        ]
    )
    response_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )

    info = compute_phase_and_special_token_metrics(
        response_ids=responses,
        response_mask=response_mask,
        special_token_ids=special,
        max_response_length=8,
    )

    assert info.metrics["format/has_think_start_ratio"] == pytest.approx(3 / 4)
    assert info.metrics["format/has_think_end_ratio"] == pytest.approx(3 / 4)
    assert info.metrics["format/unclosed_think_ratio"] == pytest.approx(1 / 4)
    assert info.metrics["format/think_end_without_think_start_ratio"] == pytest.approx(1 / 4)
    assert info.metrics["format/think_reopened_after_think_end_ratio"] == pytest.approx(1 / 4)
    assert info.metrics["response_end/ended_by_generation_stop_count"] == 3
    assert info.thinking_mask[0, :5].all()
    assert info.answering_mask[0, 5:7].all()


def test_distillation_diagnostics_bad_tail_threshold_validation_on_cpu():
    config = DistillationDiagnosticsConfig(
        bad_tail_min_position=4,
        bad_tail_deep_position=8,
        bad_tail_extreme_position=12,
        bad_tail_overlap_threshold=0.25,
    )

    assert config.bad_tail_min_position == 4
    assert config.bad_tail_overlap_threshold == 0.25
    with pytest.raises(ValueError, match="bad_tail_deep_position"):
        DistillationDiagnosticsConfig(bad_tail_min_position=8, bad_tail_deep_position=4)
    with pytest.raises(ValueError, match="topk_log_interval"):
        DistillationDiagnosticsConfig(topk_log_interval=0)
    with pytest.raises(ValueError, match="async_max_pending"):
        DistillationDiagnosticsConfig(async_max_pending=0)


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
