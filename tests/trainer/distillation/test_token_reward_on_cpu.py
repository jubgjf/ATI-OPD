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

import pytest
import torch

from verl.trainer.distillation.token_reward import compute_only_student_topk_token_reward


def test_only_student_topk_token_reward_matches_teacher_support_on_cpu():
    student_topk_ids = torch.tensor([[[1, 2, 3], [10, 11, 12]]])
    student_topk_logprobs = torch.log(torch.tensor([[[0.5, 0.3, 0.2], [0.6, 0.3, 0.1]]]))
    teacher_topk_ids = torch.tensor([[[2, 4, 1], [20, 21, 22]]])
    teacher_topk_logprobs = torch.log(torch.tensor([[[0.4, 0.35, 0.25], [0.7, 0.2, 0.1]]]))
    response_mask = torch.tensor([[1, 1]], dtype=torch.bool)
    reward_mask = torch.tensor([[1, 0]], dtype=torch.bool)

    token_rewards, metrics = compute_only_student_topk_token_reward(
        student_topk_ids=student_topk_ids,
        student_topk_logprobs=student_topk_logprobs,
        teacher_topk_ids=teacher_topk_ids,
        teacher_topk_logprobs=teacher_topk_logprobs,
        response_mask=response_mask,
        reward_mask=reward_mask,
        topk=3,
        reward_weight_mode="student_p",
    )

    expected_token0 = torch.tensor([(0.5 / 0.8) * math.log(0.25 / 0.5), (0.3 / 0.8) * math.log(0.4 / 0.3), 0.0])
    assert token_rewards.shape == (*response_mask.shape, 3)
    torch.testing.assert_close(token_rewards[0, 0], expected_token0)
    assert token_rewards[0, 0].sum().item() == pytest.approx(expected_token0.sum().item())
    assert torch.count_nonzero(token_rewards[0, 1]).item() == 0
    assert metrics["distillation/token_reward_direct/available_topk"] == 3.0
    assert metrics["distillation/token_reward_direct/teacher_on_student_topk_coverage"] == pytest.approx(2 / 3)
    assert metrics["distillation/token_reward_direct/nonempty_token_fraction"] == 1.0
    assert metrics["distillation/token_reward_direct/active_token_fraction"] == 0.5


def test_only_student_topk_token_reward_masks_missing_teacher_ids_on_cpu():
    student_topk_ids = torch.tensor([[[1, 2]]])
    student_topk_logprobs = torch.log(torch.tensor([[[0.7, 0.3]]]))
    teacher_topk_ids = torch.tensor([[[10, 11]]])
    teacher_topk_logprobs = torch.log(torch.tensor([[[0.6, 0.4]]]))
    response_mask = torch.tensor([[1]], dtype=torch.bool)

    token_rewards, metrics = compute_only_student_topk_token_reward(
        student_topk_ids=student_topk_ids,
        student_topk_logprobs=student_topk_logprobs,
        teacher_topk_ids=teacher_topk_ids,
        teacher_topk_logprobs=teacher_topk_logprobs,
        response_mask=response_mask,
        reward_weight_mode="none",
    )

    assert token_rewards.shape == (*response_mask.shape, 2)
    assert torch.count_nonzero(token_rewards).item() == 0
    assert metrics["distillation/token_reward_direct/teacher_on_student_topk_coverage"] == 0.0
    assert metrics["distillation/token_reward_direct/nonempty_token_fraction"] == 0.0
