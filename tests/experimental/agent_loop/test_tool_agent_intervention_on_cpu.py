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
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopMetrics,
    AgentLoopOutput,
    AgentLoopWorker,
    DictConfigWrap,
    ToolListWrap,
    _select_intervention_segment,
)
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
from verl.experimental.teacher_loop.teacher_manager import TeacherGenerationResult, TeacherLogprobScoringResult
from verl.tools.function_tool import FunctionTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.workers.config import DistillationInterventionConfig
from verl.workers.rollout.replica import TokenOutput


class _CharTokenizer:
    eos_token_id = 0
    pad_token = "<pad>"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, token_ids, *args, **kwargs) -> str:
        return "".join(chr(int(token_id)) for token_id in token_ids if int(token_id) != 0)

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        tokenize: bool,
        tools=None,
        **kwargs,
    ):
        text = "".join(f"<{message['role']}>{message.get('content', '')}" for message in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return self.encode(text) if tokenize else text


class _QueuedServer:
    def __init__(self, tokenizer: _CharTokenizer, outputs: list[str]):
        self.tokenizer = tokenizer
        self.outputs = list(outputs)

    async def generate(self, **kwargs) -> TokenOutput:
        text = self.outputs.pop(0)
        token_ids = self.tokenizer.encode(text)
        return TokenOutput(
            token_ids=token_ids,
            log_probs=[-0.1] * len(token_ids),
            stop_reason="completed",
            extra_fields={"finish_reason": "stop"},
        )


class _ToolDMPDTeacherManager:
    def __init__(
        self,
        tokenizer: _CharTokenizer,
        *,
        prompt_length: int,
        high_gap_start: int | None = None,
        high_gap_markers: list[str | None] | None = None,
        teacher_text: str | list[str],
    ):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.high_gap_start = high_gap_start
        self.high_gap_markers = high_gap_markers
        self.teacher_texts = [teacher_text] if isinstance(teacher_text, str) else list(teacher_text)
        self.score_calls = 0
        self.generated_prefixes: list[list[int]] = []

    async def compute_teacher_logprobs_single(self, sequence_ids: list[int], **kwargs) -> TeacherLogprobScoringResult:
        del kwargs
        sampled_logprobs = torch.full((len(sequence_ids), 1), -0.1, dtype=torch.float32)
        high_gap_start = self.high_gap_start
        if self.high_gap_markers is not None:
            marker_index = min(self.score_calls, len(self.high_gap_markers) - 1)
            marker = self.high_gap_markers[marker_index]
            response_text = self.tokenizer.decode(sequence_ids[self.prompt_length :])
            high_gap_start = response_text.find(marker) if marker is not None else None
        self.score_calls += 1
        sequence_index = -1 if high_gap_start is None else self.prompt_length - 1 + high_gap_start
        if 0 <= sequence_index < len(sequence_ids):
            sampled_logprobs[sequence_index, 0] = -5.0
        sampled_ids = torch.zeros((len(sequence_ids), 1), dtype=torch.int32)
        return TeacherLogprobScoringResult(
            ids=sampled_ids,
            logprobs=sampled_logprobs,
            sampled_ids=sampled_ids,
            sampled_logprobs=sampled_logprobs,
        )

    async def generate_teacher_segment_single(
        self, prefix_ids: list[int], max_tokens: int, **kwargs
    ) -> TeacherGenerationResult:
        del kwargs
        self.generated_prefixes.append(list(prefix_ids))
        text_index = min(len(self.generated_prefixes) - 1, len(self.teacher_texts) - 1)
        token_ids = self.tokenizer.encode(self.teacher_texts[text_index])
        assert len(token_ids) <= max_tokens
        return TeacherGenerationResult(token_ids=token_ids, stop_reason="completed", finish_reason="stop")


class _ToolDMPDWorker:
    _score_sequence_for_intervention = AgentLoopWorker._score_sequence_for_intervention
    _teacher_routing_key = AgentLoopWorker._teacher_routing_key
    teacher_key = "data_source"

    def __init__(
        self,
        tokenizer: _CharTokenizer,
        teacher_manager: _ToolDMPDTeacherManager,
        *,
        max_interventions_per_sample: int = 1,
        skip_correct: bool = False,
    ):
        self.tokenizer = tokenizer
        self.teacher_server_manager = teacher_manager
        self.teacher_intervention_enabled = True
        self.teacher_intervention_config = DistillationInterventionConfig(
            enabled=True,
            student_segment_length=2,
            teacher_segment_length=512,
            gap_metric="abs",
            gap_threshold=1.0,
            max_interventions_per_sample=max_interventions_per_sample,
            skip_correct=skip_correct,
            train_on_teacher_segments=True,
            teacher_segment_loss_mode="sft",
        )
        self.rollout_config = OmegaConf.create({"prompt_length": 512, "response_length": 2048})
        self.distillation_config = SimpleNamespace(
            distillation_loss=SimpleNamespace(
                topk=64,
                loss_settings=SimpleNamespace(use_topk=False),
                log_prob_min_clamp=-10.0,
            ),
            diagnostics=SimpleNamespace(enabled=False),
        )
        self._teacher_intervention_logged_samples = 0
        self._teacher_intervention_logged_step = None


def _tool_call(code: str = "6 * 7") -> str:
    return f'<tool_call>{{"name":"code_interpreter","arguments":{{"code":"{code}"}}}}</tool_call>'


def _make_tool(counter: dict[str, int]) -> FunctionTool:
    async def execute(code: str):
        counter["calls"] = counter.get("calls", 0) + 1
        return (
            ToolResponse(text=f"[code_interpreter status=success]\n{code}"),
            0.0,
            {"tool_execution_status": "success", "tool_markdown_extracted": 0.0},
        )

    schema = OpenAIFunctionToolSchema.model_validate(
        {
            "type": "function",
            "function": {
                "name": "code_interpreter",
                "description": "A tool for executing code.",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string", "description": "code"}},
                    "required": ["code"],
                },
            },
        }
    )
    return FunctionTool(name="code_interpreter", fn=execute, tool_schema=schema, is_async=True)


def _make_loop(outputs: list[str], *, response_length: int = 4096, counter: dict[str, int] | None = None):
    tokenizer = _CharTokenizer()
    counter = counter if counter is not None else {}
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "prompt_length": 512,
                    "response_length": response_length,
                    "multi_turn": {
                        "max_user_turns": None,
                        "max_assistant_turns": None,
                        "max_parallel_calls": 1,
                        "max_tool_response_length": 4096,
                        "tool_response_truncate_side": "middle",
                        "format": "hermes",
                    },
                }
            },
            "data": {"apply_chat_template_kwargs": {}, "mm_processor_kwargs": {}},
        }
    )
    loop = ToolAgentLoop(
        trainer_config=DictConfigWrap(config),
        server_manager=_QueuedServer(tokenizer, outputs),
        tokenizer=tokenizer,
        processor=None,
        dataset_cls=RLHFDataset,
        data_config=DictConfigWrap(config.data),
        tools=ToolListWrap([_make_tool(counter)]),
    )
    return loop, tokenizer, counter


@pytest.mark.asyncio
async def test_tool_agent_records_masks_spans_events_and_final_answer_on_cpu():
    final_answer = "<think>done</think>\\boxed{42}"
    loop, tokenizer, counter = _make_loop([_tool_call(), final_answer])
    output = await loop.run(
        {"max_tokens": 4096},
        raw_prompt=[{"role": "user", "content": "what is six times seven?"}],
    )

    assert counter["calls"] == 1
    spans = output.extra_fields["trajectory_spans"]
    assert [span["kind"] for span in spans] == ["assistant", "tool_observation", "assistant"]
    assert [span.get("source") for span in spans if span["kind"] == "assistant"] == ["student", "student"]
    observation = spans[1]
    assert output.response_mask[observation["start"] : observation["end"]] == [0] * (
        observation["end"] - observation["start"]
    )
    assert output.extra_fields["tool_events"][0]["execution_status"] == "success"
    assert output.extra_fields["tool_call_counts"] == 1
    assert output.extra_fields["reward_response_ids"] == tokenizer.encode(final_answer)


@pytest.mark.asyncio
async def test_multiple_calls_produce_protocol_observation_instead_of_silent_drop_on_cpu():
    loop, _, counter = _make_loop([_tool_call("1") + _tool_call("2"), "\\boxed{3}"])
    output = await loop.run({"max_tokens": 4096}, raw_prompt=[{"role": "user", "content": "q"}])

    assert counter.get("calls", 0) == 0
    assert len(output.extra_fields["tool_events"]) == 2
    assert {event["execution_status"] for event in output.extra_fields["tool_events"]} == {"protocol_error"}
    assert any(span.get("tool_name") == "protocol_error" for span in output.extra_fields["trajectory_spans"])


@pytest.mark.asyncio
async def test_multiple_tool_turns_are_unbounded_except_by_response_budget_on_cpu():
    loop, tokenizer, counter = _make_loop([_tool_call("1"), _tool_call("2"), "\\boxed{3}"])
    output = await loop.run({"max_tokens": 4096}, raw_prompt=[{"role": "user", "content": "q"}])

    assert counter["calls"] == 2
    assert output.extra_fields["tool_call_counts"] == 2
    assert len([span for span in output.extra_fields["trajectory_spans"] if span["kind"] == "tool_observation"]) == 2
    assert output.extra_fields["reward_response_ids"] == tokenizer.encode("\\boxed{3}")


@pytest.mark.asyncio
async def test_response_budget_can_end_immediately_after_truncated_observation_on_cpu():
    call = _tool_call()
    loop, _, counter = _make_loop([call], response_length=len(call) + 5)
    output = await loop.run({"max_tokens": 4096}, raw_prompt=[{"role": "user", "content": "q"}])

    assert counter["calls"] == 1
    assert len(output.response_ids) == len(call) + 5
    assert output.response_mask[-5:] == [0] * 5
    assert output.extra_fields["tool_call/ended_after_observation"] == 1.0
    assert output.extra_fields["reward_response_ids"] == []


@pytest.mark.asyncio
async def test_malformed_tool_json_is_tracked_and_not_executed_on_cpu():
    malformed = '<tool_call>{"name":"code_interpreter","arguments":</tool_call>'
    loop, _, counter = _make_loop([malformed])
    output = await loop.run({"max_tokens": 4096}, raw_prompt=[{"role": "user", "content": "q"}])

    assert counter.get("calls", 0) == 0
    assert output.extra_fields["tool_call_attempts"] == 1
    assert output.extra_fields["tool_call/parse_success_rate"] == 0.0
    assert output.extra_fields["reward_response_ids"] == []


@pytest.mark.asyncio
async def test_tool_exception_becomes_runtime_error_observation_and_model_can_continue_on_cpu():
    final_answer = "\\boxed{0}"
    loop, tokenizer, _ = _make_loop([_tool_call("raise RuntimeError()"), final_answer])

    async def fail(code: str):
        del code
        raise RuntimeError("fixture failure")

    loop.tools["code_interpreter"].fn = fail
    loop.tools["code_interpreter"].is_async = True
    output = await loop.run({"max_tokens": 4096}, raw_prompt=[{"role": "user", "content": "q"}])

    assert output.extra_fields["tool_events"][0]["execution_status"] == "runtime_error"
    assert output.extra_fields["tool_call/runtime_error_rate"] == 1.0
    assert output.extra_fields["reward_response_ids"] == tokenizer.encode(final_answer)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "teacher_response_value", "use_sft_mask"),
    [("false", 0, False), ("opd", 1, False), ("sft", 0, True)],
)
async def test_teacher_tool_call_is_executed_and_loss_masks_match_mode_on_cpu(
    mode: str,
    teacher_response_value: int,
    use_sft_mask: bool,
):
    final_answer = "\\boxed{42}"
    loop, tokenizer, counter = _make_loop([final_answer])
    prompt_ids = tokenizer.encode("prompt")
    prefix_ids = tokenizer.encode("student-prefix")
    teacher_ids = tokenizer.encode(_tool_call())
    original = AgentLoopOutput(
        prompt_ids=prompt_ids,
        response_ids=prefix_ids + tokenizer.encode("stale suffix"),
        response_mask=[1] * (len(prefix_ids) + len("stale suffix")),
        response_logprobs=[-0.2] * (len(prefix_ids) + len("stale suffix")),
        metrics=AgentLoopMetrics(),
        extra_fields={
            "trajectory_spans": [
                {
                    "kind": "assistant",
                    "source": "student",
                    "turn": 1,
                    "start": 0,
                    "end": len(prefix_ids) + len("stale suffix"),
                    "tool_call_ranges": [],
                }
            ],
            "tool_events": [],
        },
    )
    prefix_sft_mask = [0] * len(prefix_ids) if use_sft_mask else None
    resumed = await loop.resume_after_teacher_intervention(
        output=original,
        prefix_response_ids=prefix_ids,
        prefix_response_mask=[1] * len(prefix_ids),
        prefix_response_logprobs=[-0.2] * len(prefix_ids),
        prefix_teacher_segment_mask=prefix_sft_mask,
        teacher_segment_ids=teacher_ids,
        teacher_response_mask=[teacher_response_value] * len(teacher_ids),
        teacher_segment_logprobs=[0.0] * len(teacher_ids),
        teacher_segment_sft_mask=[1] * len(teacher_ids) if use_sft_mask else None,
        teacher_terminal=True,
        sampling_params={"max_tokens": 4096},
        sample_kwargs={"raw_prompt": [{"role": "user", "content": "q"}]},
        request_id=f"resume-{mode}",
    )

    assert counter["calls"] == 1
    assert tokenizer.decode(resumed.response_ids).startswith("student-prefix<tool_call>")
    assert "stale suffix" not in tokenizer.decode(resumed.response_ids)
    teacher_start = len(prefix_ids)
    teacher_end = teacher_start + len(teacher_ids)
    assert resumed.response_mask[teacher_start:teacher_end] == [teacher_response_value] * len(teacher_ids)
    observation_span = next(
        span for span in resumed.extra_fields["trajectory_spans"] if span["kind"] == "tool_observation"
    )
    assert resumed.response_mask[observation_span["start"] : observation_span["end"]] == [0] * (
        observation_span["end"] - observation_span["start"]
    )
    if use_sft_mask:
        assert resumed.teacher_segment_mask[teacher_start:teacher_end] == [1] * len(teacher_ids)
        assert sum(resumed.teacher_segment_mask) == len(teacher_ids)
    else:
        assert resumed.teacher_segment_mask is None
    assert resumed.extra_fields["tool_events"][-1]["source"] == "teacher"
    assert [span["source"] for span in resumed.extra_fields["trajectory_spans"] if span["kind"] == "assistant"] == [
        "student",
        "teacher",
        "student",
    ]
    assert resumed.extra_fields["reward_response_ids"] == tokenizer.encode(final_answer)


@pytest.mark.asyncio
async def test_dmpd_repair_executes_teacher_tool_and_discards_stale_environment_suffix_on_cpu():
    teacher_call = _tool_call("6 * 7")
    final_answer = "<think>verified</think>\\boxed{42}"
    loop, tokenizer, counter = _make_loop([final_answer])

    prompt_ids = tokenizer.encode("prompt")
    prefix = "sound-prefix:"
    stale_call = _tool_call("0")
    stale_observation = "<tool>[old-observation]"
    stale_answer = "\\boxed{0}"
    response_text = prefix + stale_call + stale_observation + stale_answer
    response_ids = tokenizer.encode(response_text)
    stale_call_start = len(prefix)
    stale_call_end = stale_call_start + len(stale_call)
    stale_observation_end = stale_call_end + len(stale_observation)

    teacher_manager = _ToolDMPDTeacherManager(
        tokenizer,
        prompt_length=len(prompt_ids),
        high_gap_start=stale_call_start + stale_call.index('"code"'),
        teacher_text=teacher_call,
    )
    worker = _ToolDMPDWorker(tokenizer, teacher_manager)
    output = AgentLoopOutput(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        response_mask=([1] * stale_call_end + [0] * len(stale_observation) + [1] * len(stale_answer)),
        response_logprobs=[-0.1] * len(response_ids),
        metrics=AgentLoopMetrics(),
        extra_fields={
            "rollout_request_id": "tool-dmpd-e2e",
            "trajectory_spans": [
                {
                    "kind": "assistant",
                    "source": "student",
                    "turn": 1,
                    "start": 0,
                    "end": stale_call_end,
                    "tool_call_ranges": [[stale_call_start, stale_call_end]],
                },
                {
                    "kind": "tool_observation",
                    "turn": 1,
                    "start": stale_call_end,
                    "end": stale_observation_end,
                    "tool_name": "code_interpreter",
                    "execution_status": "success",
                },
                {
                    "kind": "assistant",
                    "source": "student",
                    "turn": 2,
                    "start": stale_observation_end,
                    "end": len(response_ids),
                    "tool_call_ranges": [],
                },
            ],
            "tool_events": [
                {
                    "source": "student",
                    "turn": 1,
                    "name": "code_interpreter",
                    "parse_status": "success",
                    "execution_status": "success",
                    "call_start": stale_call_start,
                    "call_end": stale_call_end,
                    "observation_start": stale_call_end,
                    "observation_end": stale_observation_end,
                }
            ],
        },
    )

    await AgentLoopWorker._maybe_repair_with_teacher(
        worker,
        output,
        validate=False,
        sampling_params={"max_tokens": 2048},
        sample_kwargs={"raw_prompt": [{"role": "user", "content": "q"}]},
        agent_loop=loop,
    )

    repaired_text = tokenizer.decode(output.response_ids)
    assert counter["calls"] == 1
    assert repaired_text.startswith(prefix + teacher_call)
    assert stale_observation not in repaired_text
    assert stale_answer not in repaired_text
    assert repaired_text.endswith(final_answer)
    assert teacher_manager.generated_prefixes == [prompt_ids + tokenizer.encode(prefix)]
    assert output.extra_fields["teacher_intervention_count"] == 1
    intervention = output.extra_fields["teacher_interventions"][0]
    assert intervention["trigger_start"] > stale_call_start
    assert intervention["applied_start"] == stale_call_start
    assert intervention["applied_end"] == stale_call_end
    assert intervention["safe_boundary_expanded"] is True

    teacher_span = next(
        span
        for span in output.extra_fields["trajectory_spans"]
        if span["kind"] == "assistant" and span["source"] == "teacher"
    )
    observation_span = next(
        span for span in output.extra_fields["trajectory_spans"] if span["kind"] == "tool_observation"
    )
    assert output.response_mask[teacher_span["start"] : teacher_span["end"]] == [0] * (
        teacher_span["end"] - teacher_span["start"]
    )
    assert output.teacher_segment_mask[teacher_span["start"] : teacher_span["end"]] == [1] * (
        teacher_span["end"] - teacher_span["start"]
    )
    assert output.response_mask[observation_span["start"] : observation_span["end"]] == [0] * (
        observation_span["end"] - observation_span["start"]
    )
    assert output.teacher_segment_mask[observation_span["start"] : observation_span["end"]] == [0] * (
        observation_span["end"] - observation_span["start"]
    )
    assert output.extra_fields["tool_call/dmpd_repair_count"] == 1
    assert output.extra_fields["tool_call/dmpd_safe_boundary_expansion_count"] == 1
    assert output.extra_fields["reward_response_ids"] == tokenizer.encode(final_answer)


@pytest.mark.asyncio
async def test_dmpd_multiple_repairs_rescore_latest_tool_trajectory_on_cpu():
    first_teacher_call = _tool_call("1 + 1")
    second_teacher_call = _tool_call("6 * 7")
    first_student_suffix = "second-gap\\boxed{0}"
    final_answer = "<think>checked twice</think>\\boxed{42}"
    loop, tokenizer, counter = _make_loop([first_student_suffix, final_answer])
    prompt_ids = tokenizer.encode("prompt")
    original_text = "first-gap stale answer"
    original_ids = tokenizer.encode(original_text)
    teacher_manager = _ToolDMPDTeacherManager(
        tokenizer,
        prompt_length=len(prompt_ids),
        high_gap_markers=["first-gap", "second-gap", None],
        teacher_text=[first_teacher_call, second_teacher_call],
    )
    worker = _ToolDMPDWorker(tokenizer, teacher_manager, max_interventions_per_sample=2)
    output = AgentLoopOutput(
        prompt_ids=prompt_ids,
        response_ids=original_ids,
        response_mask=[1] * len(original_ids),
        response_logprobs=[-0.1] * len(original_ids),
        metrics=AgentLoopMetrics(),
        extra_fields={
            "rollout_request_id": "tool-dmpd-multiple",
            "trajectory_spans": [
                {
                    "kind": "assistant",
                    "source": "student",
                    "turn": 1,
                    "start": 0,
                    "end": len(original_ids),
                    "tool_call_ranges": [],
                }
            ],
            "tool_events": [],
        },
    )

    await AgentLoopWorker._maybe_repair_with_teacher(
        worker,
        output,
        validate=False,
        sampling_params={"max_tokens": 2048},
        sample_kwargs={"raw_prompt": [{"role": "user", "content": "q"}]},
        agent_loop=loop,
    )

    assert counter["calls"] == 2
    assert output.extra_fields["teacher_intervention_count"] == 2
    assert output.extra_fields["tool_call/dmpd_repair_count"] == 2
    assert [event["source"] for event in output.extra_fields["tool_events"]] == ["teacher", "teacher"]
    teacher_spans = [
        span
        for span in output.extra_fields["trajectory_spans"]
        if span["kind"] == "assistant" and span["source"] == "teacher"
    ]
    assert len(teacher_spans) == 2
    for span in teacher_spans:
        assert output.response_mask[span["start"] : span["end"]] == [0] * (span["end"] - span["start"])
        assert output.teacher_segment_mask[span["start"] : span["end"]] == [1] * (span["end"] - span["start"])
    assert tokenizer.decode(output.response_ids).endswith(final_answer)
    assert output.extra_fields["reward_response_ids"] == tokenizer.encode(final_answer)


@pytest.mark.asyncio
async def test_dmpd_malformed_teacher_tool_call_preserves_original_trajectory_on_cpu():
    loop, tokenizer, counter = _make_loop([])
    prompt_ids = tokenizer.encode("prompt")
    original_ids = tokenizer.encode("gap original answer")
    teacher_manager = _ToolDMPDTeacherManager(
        tokenizer,
        prompt_length=len(prompt_ids),
        high_gap_start=0,
        teacher_text='<tool_call>{"name":"code_interpreter","arguments":',
    )
    worker = _ToolDMPDWorker(tokenizer, teacher_manager)
    original_spans = [
        {
            "kind": "assistant",
            "source": "student",
            "turn": 1,
            "start": 0,
            "end": len(original_ids),
            "tool_call_ranges": [],
        }
    ]
    output = AgentLoopOutput(
        prompt_ids=prompt_ids,
        response_ids=list(original_ids),
        response_mask=[1] * len(original_ids),
        response_logprobs=[-0.1] * len(original_ids),
        metrics=AgentLoopMetrics(),
        extra_fields={
            "rollout_request_id": "tool-dmpd-malformed",
            "trajectory_spans": original_spans,
            "tool_events": [],
        },
    )

    await AgentLoopWorker._maybe_repair_with_teacher(
        worker,
        output,
        validate=False,
        sampling_params={"max_tokens": 2048},
        sample_kwargs={"raw_prompt": [{"role": "user", "content": "q"}]},
        agent_loop=loop,
    )

    assert counter.get("calls", 0) == 0
    assert output.response_ids == original_ids
    assert output.response_mask == [1] * len(original_ids)
    assert output.extra_fields["trajectory_spans"] == original_spans
    assert output.extra_fields["teacher_intervention_count"] == 0
    assert output.extra_fields["teacher_intervention_skipped"][0]["reason"] == (
        "teacher_tool_repair_incomplete_or_unbalanced_tool_call"
    )
    assert output.extra_fields["tool_call/dmpd_repair_rejection_count"] == 1


@pytest.mark.asyncio
async def test_dmpd_skip_correct_scores_only_final_assistant_and_preserves_tool_trajectory_on_cpu():
    final_answer = "\\boxed{42}"
    loop, tokenizer, counter = _make_loop([_tool_call("6 * 7"), final_answer])
    output = await loop.run({"max_tokens": 4096}, raw_prompt=[{"role": "user", "content": "q"}])
    original_ids = list(output.response_ids)
    original_mask = list(output.response_mask)
    output.extra_fields["teacher_intervention_outcome"] = {"raw_correct": True}
    teacher_manager = _ToolDMPDTeacherManager(
        tokenizer,
        prompt_length=len(output.prompt_ids),
        high_gap_start=0,
        teacher_text=_tool_call("0"),
    )
    worker = _ToolDMPDWorker(tokenizer, teacher_manager, skip_correct=True)

    await AgentLoopWorker._maybe_repair_with_teacher(
        worker,
        output,
        validate=False,
        sampling_params={"max_tokens": 2048},
        sample_kwargs={"raw_prompt": [{"role": "user", "content": "q"}]},
        agent_loop=loop,
    )

    assert counter["calls"] == 1
    assert teacher_manager.score_calls == 0
    assert output.response_ids == original_ids
    assert output.response_mask == original_mask
    assert output.teacher_segment_mask == [0] * len(original_ids)
    assert output.extra_fields["teacher_intervention_count"] == 0
    assert output.extra_fields["teacher_intervention_skipped"][0]["reason"] == "skip_correct"
    assert output.extra_fields["reward_response_ids"] == tokenizer.encode(final_answer)


@pytest.mark.asyncio
async def test_teacher_malformed_and_multiple_calls_are_rejected_on_cpu():
    loop, tokenizer, _ = _make_loop([])
    incomplete = tokenizer.encode('<tool_call>{"name":"code_interpreter"}')
    multiple = tokenizer.encode(_tool_call("1") + _tool_call("2"))
    unknown = tokenizer.encode('<tool_call>{"name":"other","arguments":{"code":"1"}}</tool_call>')
    missing_code = tokenizer.encode('<tool_call>{"name":"code_interpreter","arguments":{}}</tool_call>')
    non_string_code = tokenizer.encode('<tool_call>{"name":"code_interpreter","arguments":{"code":123}}</tool_call>')

    assert await loop.validate_teacher_intervention_segment(incomplete) == "incomplete_or_unbalanced_tool_call"
    assert await loop.validate_teacher_intervention_segment(multiple) == "multiple_tool_calls_in_assistant_turn"
    assert await loop.validate_teacher_intervention_segment(unknown) == "unknown_tool"
    assert await loop.validate_teacher_intervention_segment(missing_code) == "missing_required_tool_argument"
    assert await loop.validate_teacher_intervention_segment(non_string_code) == "invalid_tool_argument_type"


def test_observation_tokens_do_not_contribute_to_gap_trigger_on_cpu():
    config = DistillationInterventionConfig(
        enabled=True,
        student_segment_length=2,
        gap_threshold=1.0,
        gap_aggregation="max",
    )
    gaps = torch.tensor([0.0, 100.0, 100.0, 0.0])

    assert _select_intervention_segment(gaps, [1, 0, 0, 1], config) is None


@pytest.mark.parametrize(("open_tag", "close_tag"), [("<think>", "</think>"), ("<tool_call>", "</tool_call>")])
def test_intervention_boundaries_expand_outside_think_and_tool_call_blocks_on_cpu(open_tag: str, close_tag: str):
    tokenizer = _CharTokenizer()
    text = f"aa{open_tag}payload{close_tag}zz"
    token_ids = tokenizer.encode(text)
    block_start = text.index(open_tag)
    block_end = text.index(close_tag) + len(close_tag)
    dummy = SimpleNamespace(
        _teacher_intervention_thinking_start_pattern=tokenizer.encode("<think>"),
        _teacher_intervention_thinking_end_pattern=tokenizer.encode("</think>"),
        _teacher_intervention_tool_call_start_pattern=tokenizer.encode("<tool_call>"),
        _teacher_intervention_tool_call_end_pattern=tokenizer.encode("</tool_call>"),
    )

    start, end = AgentLoopWorker._expand_teacher_intervention_segment_for_token_patterns(
        dummy,
        token_ids,
        text.index("payload") + 1,
        text.index("payload") + 3,
    )

    assert (start, end) == (block_start, block_end)
