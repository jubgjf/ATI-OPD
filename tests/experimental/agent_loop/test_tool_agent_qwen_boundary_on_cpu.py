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

import os
from typing import Any

import pytest
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from verl.experimental.agent_loop.agent_loop import DictConfigWrap, ToolListWrap
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
from verl.tools.function_tool import FunctionTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.chat_template import apply_chat_template
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.rollout.replica import TokenOutput


def _load_local_qwen3_tokenizer():
    candidates = [
        os.getenv("QWEN3_TOKENIZER_PATH"),
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-0.6B",
    ]
    failures = []
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return AutoTokenizer.from_pretrained(candidate, local_files_only=True)
        except (OSError, ValueError) as error:
            failures.append(f"{candidate}: {error}")
    pytest.skip("A local Qwen3 tokenizer is required for parity testing: " + "; ".join(failures))


class _RecordingServer:
    def __init__(self, outputs: list[list[int]]):
        self.outputs = list(outputs)
        self.prompt_history: list[list[int]] = []

    async def generate(self, *, prompt_ids: list[int], **kwargs) -> TokenOutput:
        del kwargs
        self.prompt_history.append(list(prompt_ids))
        token_ids = self.outputs.pop(0)
        return TokenOutput(
            token_ids=token_ids,
            log_probs=[-0.1] * len(token_ids),
            stop_reason="completed",
            extra_fields={"finish_reason": "stop"},
        )


def _make_code_interpreter(observation: str) -> FunctionTool:
    async def execute(code: str):
        assert code == "6 * 7"
        return ToolResponse(text=observation), 0.0, {"tool_execution_status": "success"}

    schema = OpenAIFunctionToolSchema.model_validate(
        {
            "type": "function",
            "function": {
                "name": "code_interpreter",
                "description": "A tool for executing code.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The code to execute.",
                        }
                    },
                    "required": ["code"],
                },
            },
        }
    )
    return FunctionTool(name="code_interpreter", fn=execute, tool_schema=schema, is_async=True)


@pytest.mark.asyncio
async def test_qwen3_incremental_tool_context_has_exact_canonical_token_parity_on_cpu():
    tokenizer = _load_local_qwen3_tokenizer()
    assistant_tool_text = (
        "<think>\nI should calculate this exactly.\n</think>\n\n"
        '<tool_call>\n{"name":"code_interpreter","arguments":{"code":"6 * 7"}}\n</tool_call>'
    )
    final_assistant_text = "<think>\nThe result is 42.\n</think>\n\n\\boxed{42}"
    observation = "[code_interpreter status=success]\n42"
    server = _RecordingServer(
        [
            tokenizer.encode(assistant_tool_text, add_special_tokens=False),
            tokenizer.encode(final_assistant_text, add_special_tokens=False),
        ]
    )
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "prompt_length": 2048,
                    "response_length": 4096,
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
            "data": {
                "apply_chat_template_kwargs": {"enable_thinking": True},
                "mm_processor_kwargs": {},
            },
        }
    )
    loop = ToolAgentLoop(
        trainer_config=DictConfigWrap(config),
        server_manager=server,
        tokenizer=tokenizer,
        processor=None,
        dataset_cls=RLHFDataset,
        data_config=DictConfigWrap(config.data),
        tools=ToolListWrap([_make_code_interpreter(observation)]),
    )
    raw_prompt: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is six times seven?"},
    ]

    output = await loop.run({"max_tokens": 4096}, raw_prompt=raw_prompt)

    canonical_messages = raw_prompt + [
        {"role": "assistant", "content": assistant_tool_text},
        {"role": "tool", "content": observation},
    ]
    canonical_second_prompt = normalize_token_ids(
        apply_chat_template(
            tokenizer,
            canonical_messages,
            tools=loop.tool_schemas,
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=True,
        )
    )
    assert len(server.prompt_history) == 2
    assert server.prompt_history[1] == canonical_second_prompt
    assert tokenizer.decode(loop.assistant_turn_boundary_ids) == "<|im_end|>\n"

    first_turn_end = len(tokenizer.encode(assistant_tool_text, add_special_tokens=False))
    boundary_end = first_turn_end + len(loop.assistant_turn_boundary_ids)
    assert output.response_ids[first_turn_end:boundary_end] == loop.assistant_turn_boundary_ids
    assert output.response_mask[first_turn_end:boundary_end] == [0] * len(loop.assistant_turn_boundary_ids)
    assert output.response_logprobs[first_turn_end:boundary_end] == [0.0] * len(loop.assistant_turn_boundary_ids)
