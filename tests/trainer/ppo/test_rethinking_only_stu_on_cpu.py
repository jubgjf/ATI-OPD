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
from types import SimpleNamespace

import torch

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def test_trainer_rethinking_strategy_uses_exact_teacher_on_student_tensor_on_cpu():
    student_logprobs = torch.log(torch.tensor([[[0.7, 0.3]]]))
    teacher_on_student = torch.log(torch.tensor([[[0.2, 0.6]]]))
    batch = DataProto.from_dict(
        tensors={
            "student_topk_ids": torch.tensor([[[11, 12]]]),
            "student_topk_logprobs": student_logprobs,
            "response_mask": torch.ones(1, 1, dtype=torch.bool),
        }
    )
    trainer = object.__new__(RayPPOTrainer)
    trainer.distillation_config = SimpleNamespace(
        distillation_loss=SimpleNamespace(
            top_k_strategy="only_stu_rethinking_style",
            reward_weight_mode="student_p",
            log_prob_min_clamp=None,
        )
    )
    trainer._token_reward_direct_topk = lambda: 2
    calls = []

    def score_targets(received_batch):
        calls.append(received_batch)
        return teacher_on_student

    trainer._compute_rethinking_teacher_on_student_topk = score_targets

    rewards, metrics = trainer._compute_token_reward_direct_scores(batch)

    expected = (teacher_on_student - student_logprobs) * torch.tensor([[[0.7, 0.3]]])
    torch.testing.assert_close(rewards, expected)
    assert calls == [batch]
    assert metrics["distillation/token_reward_direct/rethinking_style"] == 1.0


def test_trainer_teacher_topk_strategy_reuses_ordinary_teacher_scores_on_cpu():
    student_logprobs = torch.log(torch.tensor([[[0.7, 0.3]]]))
    teacher_ids = torch.tensor([[[99, 98, 12]]])
    teacher_logprobs = torch.log(torch.tensor([[[0.5, 0.3, 0.2]]]))
    batch = DataProto.from_dict(
        tensors={
            "student_topk_ids": torch.tensor([[[11, 12]]]),
            "student_topk_logprobs": student_logprobs,
            "response_mask": torch.ones(1, 1, dtype=torch.bool),
        }
    )
    trainer = object.__new__(RayPPOTrainer)
    trainer.global_steps = 1
    trainer.distillation_config = SimpleNamespace(
        diagnostics=SimpleNamespace(enabled=False),
        distillation_loss=SimpleNamespace(
            top_k_strategy="only_stu_teacher_topk",
            teacher_topk=3,
            reward_weight_mode="student_p",
            log_prob_min_clamp=None,
        ),
    )
    trainer._token_reward_direct_topk = lambda: 2
    trainer._extract_teacher_response_topk_for_diagnostics = lambda received_batch: (
        teacher_ids,
        teacher_logprobs,
    )

    def fail_targeted_scoring(_):
        raise AssertionError("only_stu_teacher_topk must not call targeted teacher scoring")

    trainer._compute_rethinking_teacher_on_student_topk = fail_targeted_scoring

    rewards, metrics = trainer._compute_token_reward_direct_scores(batch)

    expected = torch.tensor([[[0.0, math.log(0.2 / 0.3)]]])
    torch.testing.assert_close(rewards, expected)
    assert rewards.shape == (1, 1, 2)
    assert metrics["distillation/only_stu_teacher_topk/teacher_topk"] == 3.0
    assert metrics["distillation/only_stu_teacher_topk/overlap_ratio"] == 0.5
