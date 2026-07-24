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

import torch

from verl.trainer.distillation.teacher_target_scoring import (
    assemble_teacher_target_logprobs,
    build_teacher_target_scoring_requests,
)


def test_teacher_target_request_aligns_left_padded_prompt_and_repaired_response_on_cpu():
    # Prompt width 4: [pad, pad, 101, 102].
    # Response width 4: three real tokens, then right padding. The middle real
    # token can be a teacher-inserted DMPD token; attention alignment must still
    # include it even when a later reward mask excludes it.
    input_ids = torch.tensor([[0, 0, 101, 102, 201, 301, 202, 0]])
    attention_mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 0]])
    student_topk_ids = torch.tensor(
        [
            [
                [11, 12],
                [21, 22],
                [31, 32],
                [0, 0],
            ]
        ]
    )

    requests = build_teacher_target_scoring_requests(
        input_ids=input_ids,
        attention_mask=attention_mask,
        student_topk_ids=student_topk_ids,
        prompt_width=4,
        routing_keys=["math"],
    )

    assert len(requests) == 1
    request = requests[0]
    assert request.sequence_ids == [101, 102, 201, 301, 202]
    assert request.routing_key == "math"
    assert request.response_row_mapping == ((0, 1), (1, 2), (2, 3))
    assert request.target_ids == [[0, 1], [11, 12], [21, 22], [31, 32], [0, 1]]


def test_teacher_target_result_is_verified_and_repadded_on_cpu():
    input_ids = torch.tensor([[0, 101, 102, 201, 202, 0]])
    attention_mask = torch.tensor([[0, 1, 1, 1, 1, 0]])
    student_topk_ids = torch.tensor([[[11, 12], [21, 22], [0, 0]]])
    requests = build_teacher_target_scoring_requests(
        input_ids=input_ids,
        attention_mask=attention_mask,
        student_topk_ids=student_topk_ids,
        prompt_width=3,
    )
    result = SimpleNamespace(
        ids=torch.tensor([[0, 1], [11, 12], [21, 22], [0, 0]], dtype=torch.int32),
        logprobs=torch.tensor([[0.0, 0.0], [-1.1, -1.2], [-2.1, -2.2], [0.0, 0.0]]),
    )

    teacher_logprobs = assemble_teacher_target_logprobs(
        requests=requests,
        scoring_results=[result],
        student_topk_ids=student_topk_ids,
    )

    torch.testing.assert_close(
        teacher_logprobs,
        torch.tensor([[[-1.1, -1.2], [-2.1, -2.2], [0.0, 0.0]]]),
    )
