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

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopOutput,
    AgentLoopWorker,
    _GlobalPruneFutureCalibration,
    _StudentSuffixGenerationResult,
)
from verl.experimental.agent_loop.relay import first_reflection_trigger
from verl.experimental.teacher_loop.teacher_manager import TeacherGenerationResult, TeacherLogprobScoringResult
from verl.workers.config.distillation import DistillationInterventionConfig, DistillationLossConfig


class Tokenizer:
    eos_token_id = 99

    def encode(self, text, *, add_special_tokens=False):
        assert not add_special_tokens
        return {"Wait": [70], "But": [71]}.get(text, [0, 0])

    def decode(self, ids, *, skip_special_tokens=False):
        return "\n\n" if ids == [12] else "x"


def make_worker(config):
    worker = object.__new__(AgentLoopWorker)
    worker.teacher_key = "teacher"
    worker.tokenizer = Tokenizer()
    worker.rollout_config = SimpleNamespace(response_length=512)
    worker.teacher_intervention_config = config
    worker.teacher_intervention_enabled = config.enabled
    worker.distillation_enabled = True
    worker.distillation_config = SimpleNamespace(
        distillation_loss=DistillationLossConfig(loss_mode="k1", use_policy_gradient=True), diagnostics=None
    )
    worker.teacher_server_manager = SimpleNamespace(
        compute_teacher_logprobs_single=AsyncMock(), generate_teacher_segment_single=AsyncMock()
    )
    return worker


def make_output(ids):
    return AgentLoopOutput(
        prompt_ids=[100, 101],
        response_ids=ids,
        response_mask=[1] * len(ids),
        response_logprobs=[-3.0] * len(ids),
        reward_score=1.0,
        num_turns=2,
        metrics={"generate_sequences": 0.0, "num_preempted": 0},
        extra_fields={"finish_reason": "stop", "reward_extra_info": {"acc": True}},
    )


def make_scores(prompt, response, width, *, teacher_position=None, response_logprobs=None):
    sequence = prompt + response
    sampled_ids = torch.tensor(sequence[1:] + [0]).unsqueeze(-1)
    ids = torch.arange(width).repeat(len(sequence), 1) + 30
    if width == 5:
        ids[:, 0] = 71  # A different reflection token does not veto the published trigger.
    else:
        ids[0, 0] = 70  # The prompt-only row and the final dummy row cannot trigger.
        ids[-1, 0] = 70
    if teacher_position is not None:
        ids[len(prompt) - 1 + teacher_position, 0] = 70
    logprobs = torch.full((len(sequence), width), -3.0)
    sampled_logprobs = torch.full((len(sequence), 1), -3.0)
    if response_logprobs is not None:
        sampled_logprobs[len(prompt) - 1 : -1, 0] = torch.tensor(response_logprobs)
    return TeacherLogprobScoringResult(ids, logprobs, sampled_ids, sampled_logprobs)


def test_relay_vetoes_emitted_or_student_top5_reflection():
    prompt = [100, 101]
    candidate = [70, 2, 3, 99]
    teacher = make_scores(prompt, candidate, 1)
    teacher.ids[1:4, 0] = 70
    student = make_scores(prompt, candidate, 5)
    student.ids[2, 4] = 70
    assert first_reflection_trigger(
        teacher=teacher, student=student, prefix_len=2, candidate=candidate, reflection_ids={70, 71}
    ) == (2, 70)


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ["none", "one", "two", "teacher_eos"])
async def test_relay_commits_only_causal_tokens_and_hands_off_at_paragraph_boundary(scenario):
    worker = make_worker(
        DistillationInterventionConfig(
            enabled=True, strategy="relay", train_on_teacher_segments=True, max_interventions_per_sample=2
        )
    )
    output = make_output([1, 2, 3, 99])
    prompt = output.prompt_ids.copy()
    expected_prefix = prompt + [1, 70, 12, 13, 12, 14, 12]

    async def score_teacher(*, prompt_ids, response_ids, **kwargs):
        trigger = 1 if (prompt_ids == prompt and scenario != "none") or scenario == "two" else None
        return make_scores(prompt_ids, response_ids, 1, teacher_position=trigger)

    async def score_student(*, prompt_ids, response_ids, **kwargs):
        assert kwargs["prompt_logprobs"] == 5
        return make_scores(prompt_ids, response_ids, 5)

    worker._score_sequence_for_intervention = AsyncMock(side_effect=score_teacher)
    worker._score_student_sequence_for_intervention = AsyncMock(side_effect=score_student)
    worker._generate_student_suffix = AsyncMock(
        return_value=_StudentSuffixGenerationResult(token_ids=[20, 21, 99], logprobs=[-1.0] * 3, finish_reason="stop")
    )
    worker.teacher_server_manager.generate_teacher_segment_single.side_effect = [
        TeacherGenerationResult(
            token_ids=[15, 99] if scenario == "teacher_eos" else [12, 13, 12, 14, 12, 15, 99], finish_reason="stop"
        ),
        TeacherGenerationResult(token_ids=[16, 12, 12, 12, 17], finish_reason="length"),
    ]
    await worker._maybe_repair_with_teacher(output, validate=False, sampling_params={}, sample_kwargs={})

    teacher_generate = worker.teacher_server_manager.generate_teacher_segment_single
    if scenario == "none":
        assert output.response_ids == [1, 2, 3, 99]
        assert output.reward_score == 1.0
        teacher_generate.assert_not_awaited()
        assert output.extra_fields["teacher_ids"][len(prompt) - 1 : -1, 0].tolist() == output.response_ids
    else:
        assert teacher_generate.await_args_list[0].kwargs["prefix_ids"] == prompt + [1, 70]
        assert teacher_generate.await_args_list[0].kwargs["max_tokens"] == 256
        assert output.reward_score is None
        assert output.response_logprobs is None
        assert output.extra_fields["teacher_interventions"][0]["start"] == 1
        assert output.response_mask == [1] * len(output.response_ids)
    if scenario == "teacher_eos":
        assert output.response_ids == [1, 70, 15, 99]
        worker._generate_student_suffix.assert_not_awaited()
    elif scenario in {"one", "two"}:
        worker._generate_student_suffix.assert_awaited_once()
        assert worker._generate_student_suffix.await_args.kwargs["prefix_ids"] == expected_prefix
        assert worker._score_sequence_for_intervention.await_args.kwargs["prompt_ids"] == expected_prefix
        if scenario == "one":
            assert output.response_ids == expected_prefix[len(prompt) :] + [20, 21, 99]
            assert output.extra_fields["teacher_intervention_count"] == 1
        else:
            assert teacher_generate.await_args_list[1].kwargs["prefix_ids"] == expected_prefix + [20, 70]
            assert output.response_ids == expected_prefix[len(prompt) :] + [20, 70, 16, 12, 12, 12]
            assert output.extra_fields["teacher_intervention_count"] == 2
            assert output.extra_fields["finish_reason"] == "length"


@pytest.mark.asyncio
async def test_selection_only_reuses_scoring_and_keeps_raw_trajectory():
    config = DistillationInterventionConfig(
        prune=True,
        prune_mode="selection_only",
        gap_threshold=0.15,
        student_segment_length=4,
        prune_future_window_tokens=2,
        prune_future_min_valid_tokens=2,
        prune_future_calibration_min_blocks=2,
    )
    worker = make_worker(config)
    output = make_output([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 99])
    original_ids = output.response_ids.copy()
    teacher_logprobs = [-1.0] * 4 + [-3.0] * 2 + [-1.0] * 4 + [-3.0] * 2
    worker._score_sequence_for_intervention = AsyncMock(
        return_value=make_scores(output.prompt_ids, output.response_ids, 1, response_logprobs=teacher_logprobs)
    )
    await worker._select_student_supervision(output, sample_kwargs={})
    await worker._compute_teacher_logprobs(output, output.prompt_ids, output.response_ids, False)
    assert output.extra_fields["opd_selection_cutoff"] == 6
    assert output.extra_fields["opd_selection_threshold"] == 0.15
    assert output.response_ids == original_ids
    assert output.response_mask == [1] * len(original_ids)
    assert output.reward_score == 1.0
    worker.teacher_server_manager.compute_teacher_logprobs_single.assert_not_awaited()
    worker.teacher_server_manager.generate_teacher_segment_single.assert_not_awaited()
    assert worker._prune_future_calibration_state().threshold == 0.15

    manager = _GlobalPruneFutureCalibration(config)
    updated = manager.add_updates([worker.drain_prune_future_calibration_updates()])
    assert updated.threshold > 0.15
    restored = _GlobalPruneFutureCalibration(config)
    restored.load_state_dict(manager.state_dict())
    assert restored.snapshot() == updated
