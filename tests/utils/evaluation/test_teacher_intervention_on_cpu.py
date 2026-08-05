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

from __future__ import annotations

from types import SimpleNamespace

import pytest

from verl.utils.evaluation.teacher_intervention import (
    StandaloneTeacherInterventionRunner,
    parse_echo_prompt_scoring,
    validate_http_intervention_config,
)
from verl.workers.config import DistillationInterventionConfig
from verl.workers.rollout.replica import TokenOutput


class _Tokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def __len__(self) -> int:
        return 256

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(character) % 256 for character in text]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(str(token_id) for token_id in token_ids)


def _echo_response(*, decoded_topk_key: bool = False) -> SimpleNamespace:
    key = "twenty" if decoded_topk_key else "token_id:20"
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                logprobs=SimpleNamespace(
                    token_logprobs=[None, -0.2, -0.3, -0.4],
                    top_logprobs=[
                        None,
                        {key: -0.2, "token_id:21": -1.2},
                        {"token_id:30": -0.3, "token_id:31": -1.3},
                        {"token_id:40": -0.4, "token_id:41": -1.4},
                    ],
                )
            )
        ]
    )


def test_parse_echo_prompt_scoring_aligns_exact_topk_ids() -> None:
    output = parse_echo_prompt_scoring(
        _echo_response(),
        sequence_ids=[10, 20, 30],
        prompt_logprobs=2,
        logprob_floor=-10.0,
    )
    assert output.extra_fields["prompt_ids"] == [[20, 21], [30, 31], [0, 0]]
    assert output.extra_fields["prompt_logprobs"] == [[-0.2, -1.2], [-0.3, -1.3], [0.0, 0.0]]
    assert output.extra_fields["sampled_prompt_ids"] == [[20], [30], [0]]
    assert output.extra_fields["sampled_prompt_logprobs"] == [[-0.2], [-0.3], [0.0]]


def test_parse_echo_prompt_scoring_rejects_decoded_topk_keys() -> None:
    with pytest.raises(RuntimeError, match="exact token IDs"):
        parse_echo_prompt_scoring(
            _echo_response(decoded_topk_key=True),
            sequence_ids=[10, 20, 30],
            prompt_logprobs=2,
            logprob_floor=-10.0,
        )


class _ScriptedTeacherClient:
    async def generate(self, request_id: str, *, prompt_ids: list[int], sampling_params: dict, **kwargs) -> TokenOutput:
        del request_id, kwargs
        if "prompt_logprobs" in sampling_params:
            width = max(int(sampling_params["prompt_logprobs"]), 1)
            sampled_ids = [[int(prompt_ids[index + 1])] for index in range(len(prompt_ids) - 1)] + [[0]]
            sampled_logprobs = [[-5.0] for _ in range(len(prompt_ids) - 1)] + [[0.0]]
            ids = [row * width for row in sampled_ids]
            logprobs = [row * width for row in sampled_logprobs]
            return TokenOutput(
                token_ids=[],
                log_probs=[],
                extra_fields={
                    "prompt_ids": ids,
                    "prompt_logprobs": logprobs,
                    "sampled_prompt_ids": sampled_ids,
                    "sampled_prompt_logprobs": sampled_logprobs,
                },
            )
        return TokenOutput(
            token_ids=[50],
            stop_reason="completed",
            extra_fields={"finish_reason": "length"},
        )


class _ScriptedStudentClient:
    def __init__(self) -> None:
        self.prompt_calls: list[list[int]] = []

    async def generate(self, *args, **kwargs) -> TokenOutput:
        del args
        self.prompt_calls.append(list(kwargs["prompt_ids"]))
        return TokenOutput(
            token_ids=[60],
            log_probs=[-0.2],
            stop_reason="completed",
            extra_fields={"finish_reason": "stop"},
        )


@pytest.mark.asyncio
async def test_standalone_runner_reuses_agent_loop_repair_state_machine() -> None:
    student_client = _ScriptedStudentClient()
    runner = StandaloneTeacherInterventionRunner(
        tokenizer=_Tokenizer(),
        intervention_config={
            "_target_": "verl.workers.config.DistillationInterventionConfig",
            "enabled": True,
            "student_segment_length": 2,
            "teacher_segment_length": 1,
            "gap_metric": "k3",
            "gap_threshold": 0.1,
            "max_interventions_per_sample": 1,
            "protect_thinking_blocks": False,
        },
        max_response_length=8,
        logprob_floor=-10.0,
        student_client=student_client,
        teacher_client=_ScriptedTeacherClient(),
    )
    output = await runner.repair(
        prompt_ids=[1, 3],
        response_ids=[10, 11],
        response_logprobs=[-0.1, -0.1],
        raw_reward_score=0.0,
        raw_reward_extra_info={"acc": 0.0},
        raw_correctness_value=0.0,
        raw_correct=False,
        finish_reason="length",
        sampling_params={"temperature": 0.7, "top_p": 0.95, "top_k": -1},
        request_id="test-request",
    )
    assert output.response_ids == [50, 60]
    assert output.extra_fields["teacher_intervention_count"] == 1
    assert output.extra_fields["teacher_interventions"][0]["start"] == 0
    gap = output.extra_fields["teacher_intervention_gap_diagnostics"][0]
    assert gap["gap_metric"] == "k3"
    assert gap["valid_token_count"] == 2
    assert gap["selected_segment_score"] is not None
    # Continuation is a raw token-ID prefix; no chat template is applied again.
    assert student_client.prompt_calls == [[1, 3, 50]]


def test_full_vocabulary_kl_is_rejected_before_http_requests() -> None:
    config = DistillationInterventionConfig(enabled=True, gap_metric="full_kl")
    with pytest.raises(NotImplementedError, match="full-vocabulary KL"):
        validate_http_intervention_config(config)
