# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import asyncio
import copy
import json
import logging
import os
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

import torch
from PIL import Image

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    ToolListWrap,
    register,
)
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.experimental.agent_loop.utils import build_gpt_oss_tool_response_text
from verl.tools.function_tool import FunctionTool, normalize_function_tool_return
from verl.tools.schemas import ToolResponse
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"


class AgentData:
    """Encapsulates all state variables for the agent loop. AgentData is passed to tool calling in case that
    tool may need to access full history state. User can store any tool session data in `extra_fields`."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        image_data: list[Image.Image],
        video_data: list[tuple[torch.Tensor, dict[str, Any]]],
        audio_data: Optional[list[Any]],
        mm_processor_kwargs: Optional[dict[str, Any]],
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
    ):
        self.messages = messages
        self.image_data = image_data
        self.video_data = video_data
        self.audio_data = audio_data
        self.mm_processor_kwargs = mm_processor_kwargs or {}
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs

        # State variables
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn_scores: list[float] = []
        self.tool_rewards: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0

        # Temporary state for tool calls
        self.tool_calls: list[FunctionCall] = []
        self.pending_tool_event_indices: list[int] = []

        # Response-relative structured trajectory metadata.  It survives DMPD
        # suffix rewrites and is also used to derive tool-specific metrics.
        self.trajectory_spans: list[dict[str, Any]] = []
        self.tool_events: list[dict[str, Any]] = []
        self.assistant_source = "student"
        self.last_final_assistant_ids: list[int] = []
        self.ended_after_observation = False

        self.routed_experts = None

        # Extra fields for dynamic addition, e.g., tool session data
        self.extra_fields: dict[str, Any] = {}


@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    supports_intervention_resume = True

    def __init__(self, *args, tools: Optional[ToolListWrap] = None, **kwargs):
        """Initialize the tool agent loop.

        Args:
            tools: Tools to use for the tool agent loop.
        """
        super().__init__(*args, **kwargs)

        self.max_user_turns = self.rollout_config.multi_turn.max_user_turns
        self.max_assistant_turns = self.rollout_config.multi_turn.max_assistant_turns
        self.max_parallel_calls = self.rollout_config.multi_turn.max_parallel_calls
        self.max_tool_response_length = self.rollout_config.multi_turn.max_tool_response_length
        self.tool_response_truncate_side = self.rollout_config.multi_turn.tool_response_truncate_side

        tool_list = tools.tools if tools else []
        self.tools = {tool.name: tool for tool in tool_list}
        self.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        self.tool_parser = ToolParser.get_tool_parser(self.rollout_config.multi_turn.format, self.tokenizer)
        self.tool_parser_name = self.rollout_config.multi_turn.format

        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        self.tool_call_start_pattern = self.tokenizer.encode("<tool_call>", add_special_tokens=False)
        self.tool_call_end_pattern = self.tokenizer.encode("</tool_call>", add_special_tokens=False)

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = copy.deepcopy(kwargs["raw_prompt"])

        # extract multimodal inputs from messages
        multi_modal_data = await self.process_multi_modal_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        mm_processor_kwargs = self._get_mm_processor_kwargs(audios)

        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

        agent_data = AgentData(
            messages=messages,
            image_data=images,
            video_data=videos,
            audio_data=audios,
            mm_processor_kwargs=mm_processor_kwargs,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
        )

        self._configure_active_tools(agent_data, kwargs.get("extra_info", {}) or {})
        await self._run_state_machine(agent_data, sampling_params, AgentState.PENDING)
        return self._finalize_output(agent_data)

    def _configure_active_tools(self, agent_data: AgentData, extra_info: dict[str, Any]) -> None:
        """Apply an optional per-sample allowlist to the worker's tool set."""
        tool_selection = extra_info.get("tool_selection")
        if tool_selection and self.tools:
            selected = {name: self.tools[name] for name in tool_selection if name in self.tools}
            agent_data._active_tools = selected
            agent_data._active_tool_schemas = [
                tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in selected.values()
            ]
        else:
            agent_data._active_tools = self.tools
            agent_data._active_tool_schemas = self.tool_schemas

    async def _run_state_machine(
        self,
        agent_data: AgentData,
        sampling_params: dict[str, Any],
        state: AgentState,
    ) -> None:
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

    def _finalize_output(self, agent_data: AgentData) -> AgentLoopOutput:
        """Convert mutable loop state into the common response contract."""
        response_length = len(agent_data.response_mask)
        if response_length:
            response_ids = agent_data.prompt_ids[-response_length:]
            prompt_ids = agent_data.prompt_ids[:-response_length]
        else:
            response_ids = []
            prompt_ids = list(agent_data.prompt_ids)
        multi_modal_data = {}
        if agent_data.image_data is not None:
            multi_modal_data["images"] = agent_data.image_data
        if agent_data.video_data is not None:
            multi_modal_data["videos"] = agent_data.video_data
        if agent_data.audio_data is not None:
            multi_modal_data["audios"] = agent_data.audio_data

        output: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=agent_data.mm_processor_kwargs,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            routed_experts=(
                agent_data.routed_experts[: len(prompt_ids) + self.response_length]
                if agent_data.routed_experts is not None
                else None
            ),
            extra_fields=agent_data.extra_fields,
        )
        tool_metrics = self._summarize_tool_events(agent_data, response_ids)
        output.extra_fields.update(
            {
                "turn_scores": agent_data.turn_scores,
                "tool_rewards": agent_data.tool_rewards,
                "trajectory_spans": agent_data.trajectory_spans,
                "tool_events": agent_data.tool_events,
                "reward_response_ids": list(agent_data.last_final_assistant_ids),
                **tool_metrics,
            }
        )
        return output

    @staticmethod
    def _summarize_tool_events(agent_data: AgentData, response_ids: list[int]) -> dict[str, Any]:
        events = agent_data.tool_events
        attempts = len(events)
        parsed = sum(event.get("parse_status") == "success" for event in events)
        executed = sum(event.get("execution_status") == "success" for event in events)
        student_calls = sum(
            event.get("source") == "student" and event.get("parse_status") == "success" for event in events
        )
        teacher_calls = sum(
            event.get("source") == "teacher" and event.get("parse_status") == "success" for event in events
        )
        observations = [span for span in agent_data.trajectory_spans if span.get("kind") == "tool_observation"]
        observation_tokens = sum(int(span["end"]) - int(span["start"]) for span in observations)
        status_counts = {
            status: sum(event.get("execution_status") == status for event in events)
            for status in ("timeout", "runtime_error", "http_error", "syntax_error", "protocol_error")
        }
        markdown_count = sum(bool(event.get("markdown_extracted")) for event in events)
        any_tool = attempts > 0
        final_after_tool = bool(any_tool and agent_data.last_final_assistant_ids)
        return {
            "tool_call_counts": int(parsed),
            "tool_call_attempts": int(attempts),
            "tool_call/attempt_rate": float(any_tool),
            "tool_call/parse_success_rate": float(parsed / attempts) if attempts else 0.0,
            "tool_call/parse_failure_rate": float((attempts - parsed) / attempts) if attempts else 0.0,
            "tool_call/execution_success_rate": float(executed / parsed) if parsed else 0.0,
            "tool_call/protocol_error_rate": float(status_counts["protocol_error"] / parsed) if parsed else 0.0,
            "tool_call/timeout_rate": float(status_counts["timeout"] / parsed) if parsed else 0.0,
            "tool_call/runtime_error_rate": float(status_counts["runtime_error"] / parsed) if parsed else 0.0,
            "tool_call/http_error_rate": float(status_counts["http_error"] / parsed) if parsed else 0.0,
            "tool_call/syntax_error_rate": float(status_counts["syntax_error"] / parsed) if parsed else 0.0,
            "tool_call/markdown_extraction_rate": float(markdown_count / parsed) if parsed else 0.0,
            "tool_call/student_call_rate": float(student_calls > 0),
            "tool_call/teacher_call_rate": float(teacher_calls > 0),
            "tool_call/student_calls": int(student_calls),
            "tool_call/teacher_calls": int(teacher_calls),
            "tool_call/calls_per_trajectory": int(parsed),
            "tool_call/observation_tokens": int(observation_tokens),
            "tool_call/final_answer_after_tool": float(final_after_tool),
            "tool_call/ended_after_observation": float(agent_data.ended_after_observation),
            "tool_call/response_tokens": len(response_ids),
        }

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the pending state: prepare the prompt and start generation."""
        schemas = getattr(agent_data, "_active_tool_schemas", self.tool_schemas)
        prompt_ids = await self.apply_chat_template(
            agent_data.messages,
            tools=schemas,
            images=agent_data.image_data,
            videos=agent_data.video_data,
            audios=agent_data.audio_data,
            mm_processor_kwargs=agent_data.mm_processor_kwargs,
        )
        agent_data.prompt_ids = prompt_ids
        return AgentState.GENERATING

    @staticmethod
    def _find_pattern(token_ids: list[int], pattern: list[int], start: int = 0) -> int:
        if not pattern:
            return -1
        for index in range(start, len(token_ids) - len(pattern) + 1):
            if token_ids[index : index + len(pattern)] == pattern:
                return index
        return -1

    def _tool_call_ranges(self, token_ids: list[int]) -> tuple[list[tuple[int, int]], bool]:
        """Return complete tag ranges and whether the tag stream is malformed."""
        ranges: list[tuple[int, int]] = []
        cursor = 0
        malformed = False
        while True:
            start = self._find_pattern(token_ids, self.tool_call_start_pattern, cursor)
            end_without_start = self._find_pattern(token_ids, self.tool_call_end_pattern, cursor)
            if start < 0:
                if end_without_start >= 0:
                    malformed = True
                break
            if 0 <= end_without_start < start:
                malformed = True
                cursor = end_without_start + len(self.tool_call_end_pattern)
                continue
            end = self._find_pattern(
                token_ids,
                self.tool_call_end_pattern,
                start + len(self.tool_call_start_pattern),
            )
            if end < 0:
                malformed = True
                break
            nested_start = self._find_pattern(
                token_ids,
                self.tool_call_start_pattern,
                start + len(self.tool_call_start_pattern),
            )
            if 0 <= nested_start < end:
                malformed = True
            block_end = end + len(self.tool_call_end_pattern)
            ranges.append((start, block_end))
            cursor = block_end
        return ranges, malformed

    async def _inspect_tool_segment(self, token_ids: list[int]) -> dict[str, Any]:
        active_tools = [tool.tool_schema for tool in self.tools.values()]
        _, calls = await self.tool_parser.extract_tool_calls(token_ids, active_tools)
        ranges, malformed_tags = self._tool_call_ranges(token_ids)
        has_any_tag = bool(ranges or malformed_tags)
        if malformed_tags:
            reason = "incomplete_or_unbalanced_tool_call"
        elif len(ranges) > 1 or len(calls) > 1:
            reason = "multiple_tool_calls_in_assistant_turn"
        elif ranges and not calls:
            reason = "malformed_teacher_tool_call_json"
        elif calls and not ranges and self.tool_parser_name == "hermes":
            reason = "tool_call_missing_complete_tags"
        else:
            reason = None
        return {
            "calls": calls,
            "ranges": ranges,
            "has_any_tag": has_any_tag,
            "reason": reason,
        }

    @staticmethod
    def _matches_json_type(value: Any, expected_type: str | list[str]) -> bool:
        expected_types = {expected_type} if isinstance(expected_type, str) else set(expected_type)
        checks = {
            "string": lambda item: isinstance(item, str),
            "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
            "number": lambda item: isinstance(item, int | float) and not isinstance(item, bool),
            "boolean": lambda item: isinstance(item, bool),
            "object": lambda item: isinstance(item, dict),
            "array": lambda item: isinstance(item, list),
            "null": lambda item: item is None,
        }
        return any(checks.get(type_name, lambda item: True)(value) for type_name in expected_types)

    async def validate_teacher_intervention_segment(
        self,
        token_ids: list[int],
        *,
        sample_kwargs: Optional[dict[str, Any]] = None,
    ) -> Optional[str]:
        inspection = await self._inspect_tool_segment(token_ids)
        if inspection["reason"] is not None:
            return inspection["reason"]
        calls: list[FunctionCall] = inspection["calls"]
        if not calls:
            return None

        tool_selection = ((sample_kwargs or {}).get("extra_info") or {}).get("tool_selection")
        active_tools = self.tools
        if tool_selection and self.tools:
            active_tools = {name: self.tools[name] for name in tool_selection if name in self.tools}
        call = calls[0]
        if call.name not in active_tools:
            return "unknown_tool"
        try:
            arguments = json.loads(call.arguments)
        except (json.JSONDecodeError, TypeError):
            return "malformed_teacher_tool_call_json"
        if not isinstance(arguments, dict):
            return "tool_arguments_not_object"

        parameters = active_tools[call.name].tool_schema.function.parameters
        if any(required_name not in arguments for required_name in parameters.required):
            return "missing_required_tool_argument"
        for argument_name, argument_value in arguments.items():
            property_schema = parameters.properties.get(argument_name)
            if property_schema is None:
                continue
            if not self._matches_json_type(argument_value, property_schema.type):
                return "invalid_tool_argument_type"
            if property_schema.enum is not None and argument_value not in property_schema.enum:
                return "invalid_tool_argument_enum"
        return None

    async def _record_tool_parse(
        self,
        agent_data: AgentData,
        token_ids: list[int],
        *,
        response_start: int,
        source: str,
        assistant_span: dict[str, Any],
    ) -> dict[str, Any]:
        inspection = await self._inspect_tool_segment(token_ids)
        calls: list[FunctionCall] = inspection["calls"]
        ranges: list[tuple[int, int]] = inspection["ranges"]
        assistant_span["tool_call_ranges"] = [
            [response_start + range_start, response_start + range_end] for range_start, range_end in ranges
        ]

        agent_data.pending_tool_event_indices = []
        attempt_count = max(len(ranges), len(calls), int(inspection["has_any_tag"]))
        for call_index in range(attempt_count):
            call = calls[call_index] if call_index < len(calls) else None
            call_range = ranges[call_index] if call_index < len(ranges) else None
            parse_status = "success" if call is not None else str(inspection["reason"] or "malformed_json")
            event = {
                "source": source,
                "turn": agent_data.assistant_turns,
                "name": call.name if call is not None else None,
                "parse_status": parse_status,
                "execution_status": None,
                "call_start": response_start + call_range[0] if call_range else None,
                "call_end": response_start + call_range[1] if call_range else None,
            }
            agent_data.tool_events.append(event)
            agent_data.pending_tool_event_indices.append(len(agent_data.tool_events) - 1)

        agent_data.tool_calls = calls
        return inspection

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""
        remaining_tokens = self.response_length - len(agent_data.response_mask)
        if remaining_tokens <= 0:
            return AgentState.TERMINATED
        turn_sampling_params = dict(sampling_params)
        configured_max_tokens = int(turn_sampling_params.get("max_tokens", remaining_tokens))
        turn_sampling_params["max_tokens"] = min(configured_max_tokens, remaining_tokens)
        with simple_timer("generate_sequences", agent_data.metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=turn_sampling_params,
                image_data=agent_data.image_data,
                video_data=agent_data.video_data,
                audio_data=agent_data.audio_data,
                mm_processor_kwargs=agent_data.mm_processor_kwargs,
            )
        # first time to set num_preempted
        if agent_data.metrics.get("num_preempted") is None:
            agent_data.metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        # then add num_preempted to the metrics
        else:
            agent_data.metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

        if not agent_data.extra_fields:
            agent_data.extra_fields.update(output.extra_fields)
        else:
            # Multi-round calls, only update the maximum max_global_steps.
            max_global_steps = output.extra_fields.get("max_global_steps", None)
            if max_global_steps:
                agent_data.extra_fields["max_global_steps"] = max_global_steps
        agent_data.extra_fields["stop_reason"] = output.stop_reason
        if "finish_reason" in output.extra_fields:
            agent_data.extra_fields["finish_reason"] = output.extra_fields["finish_reason"]

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        response_start = len(agent_data.response_mask)
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        assistant_span = {
            "kind": "assistant",
            "source": agent_data.assistant_source,
            "turn": agent_data.assistant_turns,
            "start": response_start,
            "end": response_start + len(agent_data.response_ids),
            "tool_call_ranges": [],
        }
        agent_data.trajectory_spans.append(assistant_span)
        agent_data.ended_after_observation = False

        inspection = await self._record_tool_parse(
            agent_data,
            agent_data.response_ids,
            response_start=response_start,
            source=agent_data.assistant_source,
            assistant_span=assistant_span,
        )
        has_valid_calls = bool(agent_data.tool_calls)
        if not has_valid_calls and not inspection["has_any_tag"]:
            agent_data.last_final_assistant_ids = list(agent_data.response_ids)

        # Check termination conditions
        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED

        if has_valid_calls:
            return AgentState.PROCESSING_TOOLS
        return AgentState.TERMINATED

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []  # Local variable instead of agent_data attribute

        # This experiment intentionally permits one call per assistant turn.
        # Multiple calls are made observable to the model instead of silently
        # dropping everything after max_parallel_calls.
        protocol_error = len(agent_data.tool_calls) != 1
        tool_call_names = [tool_call.name for tool_call in agent_data.tool_calls]
        if protocol_error:
            responses = [
                (
                    ToolResponse(
                        text=(
                            "[tool protocol_error] Exactly one tool call is allowed in each assistant turn. "
                            "Please issue one call and try again."
                        )
                    ),
                    0.0,
                    {"tool_execution_status": "protocol_error"},
                )
            ]
        else:
            with simple_timer("tool_calls", agent_data.metrics):
                responses = await asyncio.gather(
                    self._call_tool(agent_data.tool_calls[0], agent_data.tools_kwargs, agent_data)
                )

        # Process tool responses and update multi_modal_data
        # Removed: agent_data.new_images_this_turn = []
        response_metrics: list[dict[str, Any]] = []
        for tool_response, tool_reward, tool_metrics in responses:
            response_metrics.append(tool_metrics)
            # Create message from tool response
            if tool_response.image or tool_response.video:
                # Multi-modal content with structured format
                if not getattr(self.processor, "image_processor", None):
                    raise ValueError(
                        "Multimedia data can only be processed by `processor`, but the processor is None. "
                        "This error is often caused if you are using a LLM model but your tool returns multimodal "
                        "data. Plase use a vlm as the base model."
                    )
                content = []
                if tool_response.image:
                    content.append({"type": "image"})
                if tool_response.video:
                    content.append({"type": "video"})
                if tool_response.text:
                    content.append({"type": "text", "text": tool_response.text})
                message = {"role": "tool", "content": content}
            else:
                # Text-only content
                message = {"role": "tool", "content": tool_response.text or ""}

            add_messages.append(message)

            # Handle image data
            if tool_response.image:
                # Add new image data
                if isinstance(tool_response.image, list):
                    # Ensure all elements in the list are valid image objects
                    for img in tool_response.image:
                        if img is not None:  # Add a check to ensure the image is not None
                            new_images_this_turn.append(img)  # Using local variable
                else:
                    # Ensure the image is not None
                    if tool_response.image is not None:
                        new_images_this_turn.append(tool_response.image)  # Using local variable

            # Handle video data
            if tool_response.video:
                # Currently not supported, raise informative error
                logger.warning("Multimedia type 'video' is not currently supported. Only 'image' is supported.")
                raise NotImplementedError(
                    "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                )

            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

        if protocol_error:
            for event_index in agent_data.pending_tool_event_indices:
                agent_data.tool_events[event_index]["execution_status"] = "protocol_error"
        elif agent_data.pending_tool_event_indices:
            event = agent_data.tool_events[agent_data.pending_tool_event_indices[0]]
            metrics = response_metrics[0] if response_metrics else {}
            event["execution_status"] = metrics.get("tool_execution_status", "success")
            event["markdown_extracted"] = bool(metrics.get("tool_markdown_extracted", 0.0))

        agent_data.messages.extend(add_messages)

        if self.tool_parser_name == "gpt-oss":
            logger.info("manually format tool responses for gpt-oss")
            tool_response_text = build_gpt_oss_tool_response_text(add_messages, tool_call_names)
            response_ids = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
            )
        else:
            # Note that we have to pass None to the images and videos if there are no new images / videos
            # to stay compatible with downstream image processing logic!
            images = new_images_this_turn if new_images_this_turn else None
            videos = None
            response_ids = await self.apply_chat_template(
                add_messages,
                images=images,
                videos=videos,
                remove_system_prompt=True,
            )

        if new_images_this_turn:
            if agent_data.image_data is None:
                agent_data.image_data = []
            elif not isinstance(agent_data.image_data, list):
                agent_data.image_data = [agent_data.image_data]
            for img in new_images_this_turn:
                agent_data.image_data.append(img)

        observation_start = len(agent_data.response_mask)
        remaining_tokens = max(self.response_length - observation_start, 0)
        response_ids = response_ids[:remaining_tokens]
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        execution_status = response_metrics[0].get("tool_execution_status", "success") if response_metrics else "error"
        observation_span = {
            "kind": "tool_observation",
            "turn": agent_data.user_turns + 1,
            "start": observation_start,
            "end": observation_start + len(response_ids),
            "tool_name": tool_call_names[0] if len(tool_call_names) == 1 else "protocol_error",
            "execution_status": execution_status,
        }
        agent_data.trajectory_spans.append(observation_span)
        for event_index in agent_data.pending_tool_event_indices:
            agent_data.tool_events[event_index]["observation_start"] = observation_span["start"]
            agent_data.tool_events[event_index]["observation_end"] = observation_span["end"]
        agent_data.user_turns += 1
        agent_data.tool_calls = []
        agent_data.pending_tool_event_indices = []
        if len(agent_data.response_mask) >= self.response_length:
            agent_data.ended_after_observation = True
            return AgentState.TERMINATED
        # Tool observations always hand control back to the student rollout
        # server.  This matters after a teacher intervention: the repaired
        # tool-call turn is attributed to the teacher, while every assistant
        # turn generated after its observation must be attributed to the
        # student again.
        agent_data.assistant_source = "student"
        return AgentState.GENERATING

    @staticmethod
    def _trim_trajectory_metadata(
        spans: list[dict[str, Any]],
        events: list[dict[str, Any]],
        response_end: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        kept_spans: list[dict[str, Any]] = []
        for original_span in spans:
            span_start = int(original_span.get("start", 0))
            span_end = int(original_span.get("end", span_start))
            if span_start >= response_end:
                continue
            span = copy.deepcopy(original_span)
            span["end"] = min(span_end, response_end)
            span["tool_call_ranges"] = [
                list(token_range)
                for token_range in span.get("tool_call_ranges", [])
                if len(token_range) == 2 and int(token_range[1]) <= response_end
            ]
            if int(span["end"]) > span_start:
                kept_spans.append(span)

        kept_events: list[dict[str, Any]] = []
        for original_event in events:
            call_end = original_event.get("call_end")
            observation_end = original_event.get("observation_end")
            if call_end is None or int(call_end) > response_end:
                continue
            if observation_end is not None and int(observation_end) > response_end:
                continue
            kept_events.append(copy.deepcopy(original_event))
        return kept_spans, kept_events

    async def resume_after_teacher_intervention(
        self,
        *,
        output: AgentLoopOutput,
        prefix_response_ids: list[int],
        prefix_response_mask: list[int],
        prefix_response_logprobs: list[float],
        prefix_teacher_segment_mask: Optional[list[int]],
        teacher_segment_ids: list[int],
        teacher_response_mask: list[int],
        teacher_segment_logprobs: list[float],
        teacher_segment_sft_mask: Optional[list[int]],
        teacher_terminal: bool,
        sampling_params: dict[str, Any],
        sample_kwargs: dict[str, Any],
        request_id: str,
    ) -> AgentLoopOutput:
        """Discard the stale suffix, apply a teacher segment, and resume tool execution."""
        skip_reason = await self.validate_teacher_intervention_segment(
            teacher_segment_ids,
            sample_kwargs=sample_kwargs,
        )
        if skip_reason is not None:
            raise ValueError(skip_reason)

        multi_modal_data = output.multi_modal_data or {}
        agent_data = AgentData(
            messages=copy.deepcopy(sample_kwargs.get("raw_prompt", [])),
            image_data=multi_modal_data.get("images"),
            video_data=multi_modal_data.get("videos"),
            audio_data=multi_modal_data.get("audios"),
            mm_processor_kwargs=output.mm_processor_kwargs,
            metrics=output.metrics.model_dump() if hasattr(output.metrics, "model_dump") else dict(output.metrics),
            request_id=request_id,
            tools_kwargs=sample_kwargs.get("tools_kwargs", {}) or {},
        )
        self._configure_active_tools(agent_data, sample_kwargs.get("extra_info", {}) or {})

        agent_data.prompt_ids = list(output.prompt_ids) + list(prefix_response_ids) + list(teacher_segment_ids)
        agent_data.response_mask = list(prefix_response_mask) + list(teacher_response_mask)
        agent_data.response_logprobs = list(prefix_response_logprobs) + list(teacher_segment_logprobs)
        agent_data.extra_fields = copy.deepcopy(output.extra_fields)
        for metric_key in list(agent_data.extra_fields):
            if metric_key.startswith("tool_call/") or metric_key in {
                "tool_call_counts",
                "tool_call_attempts",
                "reward_response_ids",
            }:
                agent_data.extra_fields.pop(metric_key, None)

        agent_data.trajectory_spans, agent_data.tool_events = self._trim_trajectory_metadata(
            output.extra_fields.get("trajectory_spans", []),
            output.extra_fields.get("tool_events", []),
            len(prefix_response_ids),
        )
        assistant_turns = [
            int(span.get("turn", 0)) for span in agent_data.trajectory_spans if span.get("kind") == "assistant"
        ]
        user_turns = [
            int(span.get("turn", 0)) for span in agent_data.trajectory_spans if span.get("kind") == "tool_observation"
        ]
        agent_data.assistant_turns = max(assistant_turns, default=0) + 1
        agent_data.user_turns = max(user_turns, default=0)
        agent_data.assistant_source = "teacher"

        teacher_start = len(prefix_response_ids)
        teacher_span = {
            "kind": "assistant",
            "source": "teacher",
            "turn": agent_data.assistant_turns,
            "start": teacher_start,
            "end": teacher_start + len(teacher_segment_ids),
            "tool_call_ranges": [],
        }
        agent_data.trajectory_spans.append(teacher_span)
        inspection = await self._record_tool_parse(
            agent_data,
            teacher_segment_ids,
            response_start=teacher_start,
            source="teacher",
            assistant_span=teacher_span,
        )

        if agent_data.tool_calls:
            state = AgentState.PROCESSING_TOOLS
        elif teacher_terminal:
            if not inspection["has_any_tag"]:
                agent_data.last_final_assistant_ids = list(teacher_segment_ids)
            state = AgentState.TERMINATED
        else:
            agent_data.assistant_source = "student"
            state = AgentState.GENERATING

        await self._run_state_machine(agent_data, sampling_params, state)
        resumed_output = self._finalize_output(agent_data)
        if prefix_teacher_segment_mask is not None:
            resumed_output.teacher_segment_mask = (
                list(prefix_teacher_segment_mask)
                + list(teacher_segment_sft_mask or [0] * len(teacher_segment_ids))
                + [0] * (len(resumed_output.response_ids) - len(prefix_response_ids) - len(teacher_segment_ids))
            )[: len(resumed_output.response_ids)]
        return resumed_output

    async def _call_tool(
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], agent_data: AgentData
    ) -> tuple[ToolResponse, float, dict]:
        """Call tool and return tool response.

        Dispatches between two contracts:
        - ``FunctionTool``: stateless function-based tool. Invoked directly with
          parsed arguments; no lifecycle.
        - ``BaseTool`` subclass: stateful tool with full lifecycle.
        """
        active_tools = getattr(agent_data, "_active_tools", self.tools)

        # Validate tool name
        tool_name = tool_call.name
        if tool_name not in active_tools:
            available = list(active_tools.keys())
            msg = f"Unknown function '{tool_name}'. Available tools: {available}"
            logger.warning(msg)
            return ToolResponse(text=msg), 0.0, {"tool_execution_status": "protocol_error"}

        # Validate tool arguments
        try:
            tool_args = json.loads(tool_call.arguments)
        except (json.JSONDecodeError, TypeError) as e:
            msg = f"Invalid JSON in arguments for '{tool_name}': {e}"
            logger.warning(msg)
            return ToolResponse(text=msg), 0.0, {"tool_execution_status": "protocol_error"}

        # Execute tool
        tool, instance_id = None, None
        try:
            tool = active_tools[tool_name]

            if isinstance(tool, FunctionTool):
                # Function-based tools have no lifecycle; call directly.
                # Note: tools_kwargs (create_kwargs / release_kwargs) is intentionally
                # ignored here. Function tools are stateless and per-trajectory state
                # injection is not supported by design; use a BaseTool subclass instead.
                raw = await tool.call(tool_args)
                tool_execution_response, tool_reward, res = normalize_function_tool_return(raw)
            else:
                # BaseTool subclass
                kwargs = tools_kwargs.get(tool_name, {})
                instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
                tool_execution_response, tool_reward, res = await tool.execute(
                    instance_id, tool_args, agent_data=agent_data
                )
        except Exception as e:
            logger.warning(f"Error executing tool '{tool_name}': {e}")
            return (
                ToolResponse(text=f"Error executing tool '{tool_name}': {e}"),
                0.0,
                {"tool_execution_status": "runtime_error"},
            )
        finally:
            # Only BaseTool instances need release (function tools never set instance_id).
            if tool and instance_id and not isinstance(tool, FunctionTool):
                await tool.release(instance_id)

        tool_response_text = tool_execution_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        # Create ToolResponse from tool execution result
        tool_response_kwargs = {"text": tool_response_text}

        # Add multimedia data if present
        for attr_name in ["image", "video"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value

        res = dict(res or {})
        res.setdefault("tool_execution_status", "success")
        return ToolResponse(**tool_response_kwargs), tool_reward, res
