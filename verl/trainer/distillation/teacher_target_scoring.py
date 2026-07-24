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

"""Build and assemble exact teacher-on-student Top-K scoring requests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass(frozen=True)
class TeacherTargetScoringRequest:
    """One unpadded teacher prompt and its per-position target candidates."""

    sequence_ids: list[int]
    target_ids: list[list[int]]
    # (padded response column, unpadded teacher prompt-logprob row)
    response_row_mapping: tuple[tuple[int, int], ...]
    routing_key: Optional[str] = None
    multi_modal_data: Optional[dict[str, Any]] = None
    mm_processor_kwargs: Optional[dict[str, Any]] = None


def build_teacher_target_scoring_requests(
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    student_topk_ids: torch.Tensor,
    prompt_width: int,
    routing_keys: Optional[list[Any]] = None,
    multi_modal_data: Optional[list[Any]] = None,
    mm_processor_kwargs: Optional[list[Any]] = None,
) -> list[TeacherTargetScoringRequest]:
    """Convert a padded training batch to exact teacher target requests.

    Returned target rows follow verl's prompt-logprob contract: row ``r``
    contains scores from the hidden state that predicts the next prompt token.
    Therefore response column ``t`` maps to
    ``unpadded_prompt_len - 1 + response_ordinal``.
    """
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError(
            "input_ids and attention_mask must have the same [batch, sequence_len] shape, "
            f"got input_ids={tuple(input_ids.shape)}, attention_mask={tuple(attention_mask.shape)}."
        )
    if student_topk_ids.ndim != 3 or student_topk_ids.shape[0] != input_ids.shape[0]:
        raise ValueError(
            "student_topk_ids must have shape [batch, response_len, topk], "
            f"got {tuple(student_topk_ids.shape)} for batch_size={input_ids.shape[0]}."
        )
    if prompt_width <= 0 or prompt_width > input_ids.shape[1]:
        raise ValueError(f"prompt_width must be in [1, {input_ids.shape[1]}], got {prompt_width}.")
    response_width = input_ids.shape[1] - prompt_width
    if student_topk_ids.shape[1] != response_width:
        raise ValueError(
            "student_topk_ids response dimension must match the padded response width, "
            f"got student_response_len={student_topk_ids.shape[1]}, response_width={response_width}."
        )
    if student_topk_ids.shape[-1] <= 0:
        raise ValueError("student_topk_ids must contain at least one candidate per response position.")

    batch_size = input_ids.shape[0]
    optional_fields = {
        "routing_keys": routing_keys,
        "multi_modal_data": multi_modal_data,
        "mm_processor_kwargs": mm_processor_kwargs,
    }
    for name, values in optional_fields.items():
        if values is not None and len(values) != batch_size:
            raise ValueError(f"{name} must have batch_size={batch_size} entries, got {len(values)}.")

    requests = []
    for batch_index in range(batch_size):
        row_attention = attention_mask[batch_index].bool()
        prompt_positions = torch.nonzero(row_attention[:prompt_width], as_tuple=False).flatten()
        response_positions = torch.nonzero(row_attention[prompt_width:], as_tuple=False).flatten()
        if prompt_positions.numel() == 0:
            raise ValueError(f"Teacher target scoring sample {batch_index} has no valid prompt tokens.")
        sequence_ids = input_ids[batch_index, row_attention].detach().cpu().long().tolist()
        unpadded_prompt_len = int(prompt_positions.numel())
        topk = student_topk_ids.shape[-1]
        # These rows are ignored by the trainer, but they still travel through
        # vLLM's dict-shaped prompt-logprob output. Keep ids unique so the dict
        # conversion cannot collapse K entries into one (and create -inf
        # padding that would look like an NPU numerical failure).
        dummy_target_ids = list(range(topk))
        target_rows = [dummy_target_ids.copy() for _ in sequence_ids]
        response_row_mapping = []

        for response_ordinal, padded_response_position_tensor in enumerate(response_positions):
            padded_response_position = int(padded_response_position_tensor.item())
            score_row = unpadded_prompt_len - 1 + response_ordinal
            if score_row < 0 or score_row >= len(sequence_ids) - 1:
                raise RuntimeError(
                    "Teacher target scoring response alignment escaped the scoreable prompt rows: "
                    f"sample={batch_index}, response_position={padded_response_position}, "
                    f"score_row={score_row}, sequence_len={len(sequence_ids)}."
                )
            candidates = student_topk_ids[batch_index, padded_response_position].detach().cpu().long().tolist()
            target_rows[score_row] = candidates
            response_row_mapping.append((padded_response_position, score_row))

        routing_key = None if routing_keys is None else routing_keys[batch_index]
        if hasattr(routing_key, "item"):
            routing_key = routing_key.item()
        requests.append(
            TeacherTargetScoringRequest(
                sequence_ids=sequence_ids,
                target_ids=target_rows,
                response_row_mapping=tuple(response_row_mapping),
                routing_key=routing_key,
                multi_modal_data=None if multi_modal_data is None else multi_modal_data[batch_index],
                mm_processor_kwargs=None if mm_processor_kwargs is None else mm_processor_kwargs[batch_index],
            )
        )
    return requests


def assemble_teacher_target_logprobs(
    *,
    requests: list[TeacherTargetScoringRequest],
    scoring_results: list[Any],
    student_topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Verify target identity and repad exact teacher log-probabilities."""
    if len(requests) != len(scoring_results) or len(requests) != student_topk_ids.shape[0]:
        raise ValueError(
            "Teacher target scoring request/result counts must match the student batch, "
            f"got requests={len(requests)}, results={len(scoring_results)}, "
            f"student_batch={student_topk_ids.shape[0]}."
        )
    teacher_on_student = torch.zeros_like(student_topk_ids, dtype=torch.float32)
    for batch_index, (request, result) in enumerate(zip(requests, scoring_results, strict=True)):
        result_ids = result.ids.detach().cpu()
        result_logprobs = result.logprobs.detach().cpu().float()
        expected_shape = (len(request.sequence_ids), student_topk_ids.shape[-1])
        if result_ids.shape != expected_shape or result_logprobs.shape != expected_shape:
            raise RuntimeError(
                "Teacher target result shape does not match its request: "
                f"sample={batch_index}, expected={expected_shape}, ids={tuple(result_ids.shape)}, "
                f"logprobs={tuple(result_logprobs.shape)}."
            )
        for padded_response_position, score_row in request.response_row_mapping:
            expected_ids = student_topk_ids[batch_index, padded_response_position].detach().cpu().long()
            returned_ids = result_ids[score_row].long()
            if not torch.equal(returned_ids, expected_ids):
                mismatch = int(torch.nonzero(returned_ids != expected_ids, as_tuple=False)[0].item())
                raise RuntimeError(
                    "Teacher target result ids do not match student Top-K candidates: "
                    f"sample={batch_index}, response_position={padded_response_position}, candidate={mismatch}, "
                    f"expected_id={int(expected_ids[mismatch])}, returned_id={int(returned_ids[mismatch])}."
                )
            teacher_on_student[batch_index, padded_response_position] = result_logprobs[score_row]
    return teacher_on_student.to(device=student_topk_ids.device)
