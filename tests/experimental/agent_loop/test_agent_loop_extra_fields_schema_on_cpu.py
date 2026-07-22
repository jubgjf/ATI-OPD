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

import warnings
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopMetrics,
    AgentLoopOutput,
    AgentLoopWorker,
    DictConfigWrap,
    _apply_generation_max_tokens,
    _compute_distribution_intervention_gap,
    _compute_intervention_gap,
    _InternalAgentLoopOutput,
    _is_moe_like_hf_config,
    _select_intervention_segment,
)
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.experimental.teacher_loop.teacher_manager import (
    AsyncTeacherLLMServerManager,
    TeacherGenerationResult,
    TeacherLogprobScoringResult,
)
from verl.protocol import DataProto
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.rollout_trace import RolloutTraceConfig
from verl.workers.config import DistillationInterventionConfig
from verl.workers.rollout.replica import TokenOutput


def test_apply_generation_max_tokens_adds_sampling_param_on_cpu():
    sampling_params = {"temperature": 0, "top_p": 1.0}

    _apply_generation_max_tokens(sampling_params, {"generation_max_tokens": 128})

    assert sampling_params["max_tokens"] == 128
    assert sampling_params["temperature"] == 0
    assert sampling_params["top_p"] == 1.0


def test_apply_generation_max_tokens_missing_meta_keeps_sampling_param_on_cpu():
    sampling_params = {"temperature": 0}

    _apply_generation_max_tokens(sampling_params, {})

    assert sampling_params == {"temperature": 0}


def test_apply_generation_max_tokens_rejects_non_positive_value_on_cpu():
    with pytest.raises(ValueError, match="generation_max_tokens must be positive"):
        _apply_generation_max_tokens({}, {"generation_max_tokens": 0})


class _FakeServerManager:
    async def generate(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        audio_data: Optional[list[Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
    ) -> TokenOutput:
        del request_id, sampling_params, image_data, video_data, audio_data, mm_processor_kwargs
        # Return a short, deterministic "generation" for testing.
        return TokenOutput(token_ids=prompt_ids[-1:] + [11, 12, 13], log_probs=[0.0, 0.0, 0.0, 0.0])

    async def generate_for_partial(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        audio_data: Optional[list[Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[list[int], list[float], bool]:
        del request_id, sampling_params, image_data, video_data, audio_data, mm_processor_kwargs
        # Return a short partial generation and "not cancelled".
        response_ids = prompt_ids[-1:] + [21, 22]
        response_logprobs = [0.0] * len(response_ids)
        return response_ids, response_logprobs, False


class _FakeTokenizer:
    padding_side = "right"
    pad_token_id = 0
    eos_token_id = 99
    text_tokens = {
        "<think>": [91],
        "</think>": [92],
        "<multi_think>": [71, 72],
        "</multi_think>": [73, 74],
    }

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: Optional[list[dict]] = None,
        add_generation_prompt: bool = True,
        tokenize: bool = True,
        **kwargs,
    ) -> list[int]:
        del messages, tools, add_generation_prompt, tokenize, kwargs
        # Minimal tokenization: return a small prompt.
        return [101, 102]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(self.text_tokens.get(text, []))

    def __len__(self) -> int:
        return 100

    def pad(
        self,
        encoded_inputs: dict[str, list[int]],
        *,
        padding: str,
        max_length: int,
        return_tensors: str,
        return_attention_mask: bool,
    ) -> dict[str, torch.Tensor]:
        del padding, return_tensors
        input_ids = encoded_inputs["input_ids"]
        if len(input_ids) > max_length:
            if self.padding_side == "left":
                input_ids = input_ids[-max_length:]
            else:
                input_ids = input_ids[:max_length]

        pad_len = max_length - len(input_ids)
        if self.padding_side == "left":
            padded_ids = [0] * pad_len + input_ids
            attention_mask = [0] * pad_len + [1] * len(input_ids)
        else:
            padded_ids = input_ids + [0] * pad_len
            attention_mask = [1] * len(input_ids) + [0] * pad_len

        output = {"input_ids": torch.tensor([padded_ids], dtype=torch.long)}
        if return_attention_mask:
            output["attention_mask"] = torch.tensor([attention_mask], dtype=torch.long)
        return output

    def decode(self, ids: list[int] | torch.Tensor, skip_special_tokens: bool = True) -> str:
        del ids, skip_special_tokens
        return "<decoded>"


def _pad_1d(ids: list[int], *, length: int, pad_id: int = 0) -> list[int]:
    if len(ids) > length:
        return ids[:length]
    return ids + [pad_id] * (length - len(ids))


def _to_internal(
    *,
    output_prompt_ids: list[int],
    output_response_ids: list[int],
    output_response_mask: list[int],
    metrics: AgentLoopMetrics,
    extra_fields: dict[str, Any],
    num_turns: int,
    prompt_len: int,
    response_len: int,
) -> _InternalAgentLoopOutput:
    prompt_ids = _pad_1d(output_prompt_ids, length=prompt_len, pad_id=0)
    response_ids = _pad_1d(output_response_ids, length=response_len, pad_id=0)
    response_mask = _pad_1d(output_response_mask, length=response_len, pad_id=0)

    seq_len = prompt_len + response_len
    attention_mask = _pad_1d([1] * len(output_prompt_ids), length=prompt_len, pad_id=0) + _pad_1d(
        [1] * len(output_response_ids),
        length=response_len,
        pad_id=0,
    )
    input_ids = prompt_ids + response_ids
    position_ids = list(range(seq_len))

    def t(x: list[int]) -> torch.Tensor:
        return torch.tensor([x], dtype=torch.long)

    return _InternalAgentLoopOutput(
        prompt_ids=t(prompt_ids),
        response_ids=t(response_ids),
        response_mask=t(response_mask),
        attention_mask=t(attention_mask),
        input_ids=t(input_ids),
        position_ids=t(position_ids),
        response_logprobs=None,
        routed_experts=None,
        multi_modal_inputs=None,
        multi_modal_data=None,
        reward_score=None,
        num_turns=num_turns,
        metrics=metrics,
        extra_fields=extra_fields,
    )


@pytest.mark.asyncio
async def test_agent_loop_extra_fields_schema_stable_for_training_concat_on_cpu():
    # Minimal config surface used by the agent loops.
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {"prompt_length": 16, "response_length": 16, "multi_turn": {"tool_config_path": None}},
                "model": {},
            },
            "data": {
                "tool_config_path": None,
                "apply_chat_template_kwargs": {},
            },
        }
    )

    server_manager = _FakeServerManager()
    tokenizer = _FakeTokenizer()
    processor = None

    trainer_config = DictConfigWrap(config)
    data_config = DictConfigWrap(config.data)

    single_turn = SingleTurnAgentLoop(
        trainer_config=trainer_config,
        server_manager=server_manager,
        tokenizer=tokenizer,
        processor=processor,
        dataset_cls=RLHFDataset,
        data_config=data_config,
    )

    raw_prompt = [{"role": "user", "content": "hi"}]
    sampling_params: dict[str, Any] = {}

    out = await single_turn.run(sampling_params=sampling_params, raw_prompt=raw_prompt)

    # Agent loop outputs should always contain these fields with consistent types.
    assert out.extra_fields["turn_scores"] == []
    assert out.extra_fields["tool_rewards"] == []

    internal_a = _to_internal(
        output_prompt_ids=out.prompt_ids,
        output_response_ids=out.response_ids,
        output_response_mask=out.response_mask,
        metrics=out.metrics,
        extra_fields=out.extra_fields,
        num_turns=out.num_turns,
        prompt_len=len(out.prompt_ids),
        response_len=len(out.response_ids),
    )

    # Mimic two "worker chunks" and concatenate as in training.
    dummy_worker = type(
        "_DummyWorker",
        (),
        {"reward_loop_worker_handles": None, "distillation_enabled": False},
    )()
    merged = AgentLoopWorker._postprocess(
        dummy_worker,
        inputs=[internal_a],
        input_non_tensor_batch={
            "index": np.array([0], dtype=object),
            "agent_name": np.array(["single_turn_agent"], dtype=object),
        },
    )

    # Stable schema: present regardless of which loop produced a sample.
    stable_keys = (
        "turn_scores",
        "tool_rewards",
        "min_global_steps",
        "max_global_steps",
        "extras",
    )
    for key in stable_keys:
        assert key in merged.non_tensor_batch, f"missing key in merged batch: {key}"
        assert merged.non_tensor_batch[key].shape == (1,), (
            f"invalid shape for {key}: {merged.non_tensor_batch[key].shape}"
        )

    # And the list-typed fields are actually lists (not missing / scalar).
    assert merged.non_tensor_batch["turn_scores"][0] == []
    assert merged.non_tensor_batch["tool_rewards"][0] == []


@pytest.mark.asyncio
async def test_agent_loop_generate_sequences_passes_meta_global_steps_to_sample_kwargs_on_cpu():
    class _DummyWorker:
        generate_sequences = AgentLoopWorker.generate_sequences

        def __init__(self):
            self.rollout_config = OmegaConf.create(
                {
                    "temperature": 0.8,
                    "top_p": 0.95,
                    "top_k": -1,
                    "calculate_log_probs": False,
                    "val_kwargs": {"top_p": 1.0, "top_k": -1, "temperature": 0.0},
                    "agent": {"default_agent_loop": "single_turn_agent"},
                }
            )
            self.teacher_intervention_enabled = True
            self.seen_kwargs: list[dict[str, Any]] = []
            self.seen_trajectories: list[dict[str, Any]] = []
            self.postprocess_input_non_tensor_batch = None

        async def _run_agent_loop(
            self,
            sampling_params: dict[str, Any],
            trajectory: dict[str, Any],
            trace: bool = False,
            **kwargs,
        ) -> AgentLoopOutput:
            del sampling_params, trace
            self.seen_kwargs.append(kwargs)
            self.seen_trajectories.append(trajectory)
            return AgentLoopOutput(
                prompt_ids=[101],
                response_ids=[11],
                response_mask=[1],
                response_logprobs=[-0.1],
                metrics=AgentLoopMetrics(),
                extra_fields={"rollout_request_id": kwargs["uid"]},
            )

        def _postprocess(self, outputs, input_non_tensor_batch=None, validate=False):
            del outputs, validate
            self.postprocess_input_non_tensor_batch = input_non_tensor_batch
            return DataProto.from_dict(tensors={"responses": torch.zeros((2, 1), dtype=torch.long)})

    RolloutTraceConfig.get_instance().max_samples_per_step_per_worker = None
    worker = _DummyWorker()
    batch = DataProto.from_dict(
        tensors={"input_ids": torch.zeros((2, 1), dtype=torch.long)},
        non_tensors={
            "raw_prompt": np.array(
                [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]],
                dtype=object,
            ),
            "uid": np.array(["sample-a", "sample-b"], dtype=object),
        },
        meta_info={"global_steps": 17},
    )

    await AgentLoopWorker.generate_sequences(worker, batch)

    assert [kwargs["global_steps"] for kwargs in worker.seen_kwargs] == [17, 17]
    assert [trajectory["step"] for trajectory in worker.seen_trajectories] == [17, 17]
    assert "global_steps" not in worker.postprocess_input_non_tensor_batch


@pytest.mark.asyncio
async def test_agent_loop_postprocess_accepts_read_only_routed_experts_on_cpu():
    class _DummyWorker:
        _compute_multi_modal_inputs = AgentLoopWorker._compute_multi_modal_inputs
        _compute_position_ids = AgentLoopWorker._compute_position_ids
        _get_mm_processor_kwargs = AgentLoopWorker._get_mm_processor_kwargs
        _compute_score = AgentLoopWorker._compute_score
        _compute_teacher_logprobs = AgentLoopWorker._compute_teacher_logprobs
        distillation_enabled = False

        def __init__(self):
            self.tokenizer = _FakeTokenizer()
            self.rollout_config = OmegaConf.create({"prompt_length": 4, "response_length": 4})
            self.processor = None
            self.mm_processor_kwargs = {}
            self.reward_loop_worker_handles = None

    routed_experts = np.arange(8, dtype=np.int64).reshape(4, 2, 1)
    routed_experts.setflags(write=False)
    assert not routed_experts.flags.writeable

    output = AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[11, 12],
        response_mask=[1, 1],
        routed_experts=routed_experts,
        metrics=AgentLoopMetrics(),
        extra_fields={},
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message="The given NumPy array is not writable.*",
            category=UserWarning,
        )
        internal = await AgentLoopWorker._agent_loop_postprocess(
            _DummyWorker(),
            output,
            validate=False,
            raw_prompt=[{"role": "user", "content": "hi"}],
        )

    expected = torch.tensor(routed_experts.copy()).unsqueeze(0)
    assert internal.routed_experts is not None
    assert internal.routed_experts.shape == (1, 8, 2, 1)
    torch.testing.assert_close(internal.routed_experts[:, 2:6], expected)
    assert torch.count_nonzero(internal.routed_experts[:, :2]) == 0
    assert torch.count_nonzero(internal.routed_experts[:, 6:]) == 0


class _FakeInterventionTeacherManager:
    def __init__(self):
        self.scored_sequences: list[list[int]] = []
        self.generated_prefixes: list[list[int]] = []

    async def compute_teacher_logprobs_single(
        self,
        sequence_ids: list[int],
        multi_modal_data: Optional[dict[str, Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        routing_key: Optional[str] = None,
        request_id: Optional[str] = None,
        prompt_logprobs: Optional[int] = None,
    ) -> TeacherLogprobScoringResult:
        del multi_modal_data, mm_processor_kwargs, routing_key, request_id
        assert prompt_logprobs == 0
        self.scored_sequences.append(list(sequence_ids))
        teacher_ids = torch.zeros(len(sequence_ids), 1, dtype=torch.int32)
        teacher_logprobs = torch.full((len(sequence_ids), 1), -0.1, dtype=torch.float32)

        # prompt_len=2. The first scoring sees response [1, 2, 3, 4, 5, 6].
        # Response token j aligns to teacher_logprobs[prompt_len + j - 1].
        response = sequence_ids[2:]
        if response[:6] == [1, 2, 3, 4, 5, 6]:
            teacher_logprobs[3, 0] = -5.0
            teacher_logprobs[4, 0] = -5.0
        call_idx = len(self.scored_sequences) - 1
        return TeacherLogprobScoringResult(
            ids=teacher_ids,
            logprobs=teacher_logprobs,
            nonfinite_count=1,
            nonfinite_sequence_positions=[3 if call_idx == 0 else 5],
            replacement_value=-10.0,
        )

    async def generate_teacher_segment_single(
        self,
        prefix_ids: list[int],
        max_tokens: int,
        multi_modal_data: Optional[dict[str, Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        routing_key: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> list[int]:
        del max_tokens, multi_modal_data, mm_processor_kwargs, routing_key, request_id
        self.generated_prefixes.append(list(prefix_ids))
        return [30, 31]


class _FakeRepairStudentClient:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        audio_data: Optional[list[Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
    ) -> TokenOutput:
        del image_data, video_data, audio_data, mm_processor_kwargs
        self.calls.append(
            {"request_id": request_id, "prompt_ids": list(prompt_ids), "sampling_params": sampling_params}
        )
        return TokenOutput(token_ids=[40, 41], log_probs=[-0.2, -0.2])


class _DistributionGapStudentClient(_FakeRepairStudentClient):
    async def generate(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        audio_data: Optional[list[Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
    ) -> TokenOutput:
        if "prompt_logprobs" not in sampling_params:
            return await super().generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                video_data=video_data,
                audio_data=audio_data,
                mm_processor_kwargs=mm_processor_kwargs,
            )
        self.calls.append(
            {"request_id": request_id, "prompt_ids": list(prompt_ids), "sampling_params": sampling_params}
        )
        width = int(sampling_params["prompt_logprobs"])
        assert width == 2
        ids = [[1, 2] for _ in prompt_ids]
        logprobs = torch.log(torch.tensor([[0.9, 0.1] for _ in prompt_ids], dtype=torch.float32)).tolist()
        sampled_ids = [[int(token_id)] for token_id in prompt_ids]
        sampled_logprobs = [[-0.1] for _ in prompt_ids]
        return TokenOutput(
            token_ids=[0],
            extra_fields={
                "prompt_ids": ids,
                "prompt_logprobs": logprobs,
                "sampled_prompt_ids": sampled_ids,
                "sampled_prompt_logprobs": sampled_logprobs,
            },
        )


class _DistributionGapTeacherManager:
    def __init__(self):
        self.scored_sequences: list[list[int]] = []
        self.generated_prefixes: list[list[int]] = []
        self.prompt_logprobs_calls: list[Optional[int]] = []

    async def compute_teacher_logprobs_single(
        self,
        sequence_ids: list[int],
        multi_modal_data: Optional[dict[str, Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        routing_key: Optional[str] = None,
        request_id: Optional[str] = None,
        prompt_logprobs: Optional[int] = None,
    ) -> TeacherLogprobScoringResult:
        del multi_modal_data, mm_processor_kwargs, routing_key, request_id
        self.scored_sequences.append(list(sequence_ids))
        self.prompt_logprobs_calls.append(prompt_logprobs)
        assert prompt_logprobs == 2
        ids = torch.tensor([[1, 2] for _ in sequence_ids], dtype=torch.int32)
        probs = torch.tensor([[0.9, 0.1] for _ in sequence_ids], dtype=torch.float32)
        # prompt_len=2, so row 1 scores the first response token.
        if len(sequence_ids) > 2:
            probs[1] = torch.tensor([0.1, 0.9])
        sampled_ids = torch.tensor([[int(token_id)] for token_id in sequence_ids], dtype=torch.int32)
        sampled_logprobs = torch.full((len(sequence_ids), 1), -0.1, dtype=torch.float32)
        return TeacherLogprobScoringResult(
            ids=ids,
            logprobs=torch.log(probs),
            sampled_ids=sampled_ids,
            sampled_logprobs=sampled_logprobs,
        )

    async def generate_teacher_segment_single(
        self,
        prefix_ids: list[int],
        max_tokens: int,
        multi_modal_data: Optional[dict[str, Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        routing_key: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> TeacherGenerationResult:
        del max_tokens, multi_modal_data, mm_processor_kwargs, routing_key, request_id
        self.generated_prefixes.append(list(prefix_ids))
        return TeacherGenerationResult(token_ids=[30], finish_reason="length")


class _FixedGapTeacherManager:
    def __init__(
        self,
        generation_result: TeacherGenerationResult | list[int],
        *,
        high_gap_start: int = 0,
        high_gap_width: int = 2,
        prompt_len: int = 2,
        topk_width: int = 1,
    ):
        self.generation_result = generation_result
        self.high_gap_start = high_gap_start
        self.high_gap_width = high_gap_width
        self.prompt_len = prompt_len
        self.topk_width = topk_width
        self.scored_sequences: list[list[int]] = []
        self.generated_prefixes: list[list[int]] = []
        self.prompt_logprobs_calls: list[Optional[int]] = []

    async def compute_teacher_logprobs_single(
        self,
        sequence_ids: list[int],
        multi_modal_data: Optional[dict[str, Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        routing_key: Optional[str] = None,
        request_id: Optional[str] = None,
        prompt_logprobs: Optional[int] = None,
    ) -> TeacherLogprobScoringResult:
        del multi_modal_data, mm_processor_kwargs, routing_key, request_id
        self.scored_sequences.append(list(sequence_ids))
        self.prompt_logprobs_calls.append(prompt_logprobs)
        width = self.topk_width if prompt_logprobs and prompt_logprobs > 0 else 1
        teacher_ids = torch.zeros(len(sequence_ids), width, dtype=torch.int32)
        teacher_logprobs = torch.full((len(sequence_ids), width), -0.1, dtype=torch.float32)
        sampled_teacher_ids = torch.zeros(len(sequence_ids), 1, dtype=torch.int32)
        sampled_teacher_logprobs = torch.full((len(sequence_ids), 1), -0.1, dtype=torch.float32)
        response_len = max(len(sequence_ids) - self.prompt_len, 0)
        for response_idx in range(
            self.high_gap_start,
            min(self.high_gap_start + self.high_gap_width, response_len),
        ):
            sequence_idx = self.prompt_len - 1 + response_idx
            teacher_logprobs[sequence_idx, 0] = -5.0
            sampled_teacher_logprobs[sequence_idx, 0] = -5.0
        return TeacherLogprobScoringResult(
            ids=teacher_ids,
            logprobs=teacher_logprobs,
            sampled_ids=sampled_teacher_ids,
            sampled_logprobs=sampled_teacher_logprobs,
        )

    async def generate_teacher_segment_single(
        self,
        prefix_ids: list[int],
        max_tokens: int,
        multi_modal_data: Optional[dict[str, Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        routing_key: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> TeacherGenerationResult | list[int]:
        del max_tokens, multi_modal_data, mm_processor_kwargs, routing_key, request_id
        self.generated_prefixes.append(list(prefix_ids))
        return self.generation_result


class _DirectInterventionWorker:
    _teacher_routing_key = AgentLoopWorker._teacher_routing_key
    _score_sequence_for_intervention = AgentLoopWorker._score_sequence_for_intervention
    _score_student_sequence_for_intervention = AgentLoopWorker._score_student_sequence_for_intervention
    _generate_student_suffix = AgentLoopWorker._generate_student_suffix
    _maybe_repair_with_teacher = AgentLoopWorker._maybe_repair_with_teacher
    teacher_key = "data_source"
    distillation_enabled = True

    def __init__(
        self,
        *,
        teacher_manager: _FixedGapTeacherManager,
        response_length: int = 8,
        student_segment_length: int = 2,
        teacher_segment_length: int = 2,
        max_interventions_per_sample: int = 1,
        train_on_teacher_segments: bool = False,
        teacher_segment_loss_mode: str = "opd",
        min_trainable_tokens_after_repair: int = 1,
        gap_metric: str = "abs",
        gap_threshold: float = 1.0,
        gap_distribution_topk: int = 16,
        teacher_rejection_token_cap: float = 10.0,
        topk: Optional[int] = None,
        thinking_start_text: str = "<think>",
        thinking_end_text: str = "</think>",
        log_examples_per_worker: int = 0,
        diagnostics_topk: Optional[int] = None,
    ):
        self.tokenizer = _FakeTokenizer()
        self.rollout_config = OmegaConf.create({"response_length": response_length})
        self.teacher_intervention_enabled = True
        self.teacher_intervention_config = DistillationInterventionConfig(
            enabled=True,
            student_segment_length=student_segment_length,
            teacher_segment_length=teacher_segment_length,
            gap_metric=gap_metric,
            gap_distribution_topk=gap_distribution_topk,
            teacher_rejection_token_cap=teacher_rejection_token_cap,
            gap_threshold=gap_threshold,
            max_interventions_per_sample=max_interventions_per_sample,
            train_on_teacher_segments=train_on_teacher_segments,
            teacher_segment_loss_mode=teacher_segment_loss_mode,
            min_trainable_tokens_after_repair=min_trainable_tokens_after_repair,
            thinking_start_text=thinking_start_text,
            thinking_end_text=thinking_end_text,
            log_examples_per_worker=log_examples_per_worker,
        )
        self.teacher_server_manager = teacher_manager
        self.llm_client = _FakeRepairStudentClient()
        self.distillation_config = SimpleNamespace(
            distillation_loss=SimpleNamespace(
                topk=topk or 64,
                loss_settings=SimpleNamespace(use_topk=topk is not None),
                log_prob_min_clamp=-10.0,
            ),
            diagnostics=SimpleNamespace(
                collect_topk_overlap=diagnostics_topk is not None,
                topk=diagnostics_topk or 64,
            ),
        )
        self._teacher_intervention_logged_samples = 0
        self._teacher_intervention_logged_step = None


@pytest.mark.asyncio
async def test_agent_loop_postprocess_repairs_high_gap_segment_on_cpu():
    class _DummyWorker:
        _compute_multi_modal_inputs = AgentLoopWorker._compute_multi_modal_inputs
        _compute_position_ids = AgentLoopWorker._compute_position_ids
        _get_mm_processor_kwargs = AgentLoopWorker._get_mm_processor_kwargs
        _compute_score = AgentLoopWorker._compute_score
        _teacher_routing_key = AgentLoopWorker._teacher_routing_key
        _score_sequence_for_intervention = AgentLoopWorker._score_sequence_for_intervention
        _generate_student_suffix = AgentLoopWorker._generate_student_suffix
        _maybe_repair_with_teacher = AgentLoopWorker._maybe_repair_with_teacher
        _compute_teacher_logprobs = AgentLoopWorker._compute_teacher_logprobs
        distillation_enabled = True
        teacher_key = "data_source"

        def __init__(self):
            self.tokenizer = _FakeTokenizer()
            self.rollout_config = OmegaConf.create({"prompt_length": 4, "response_length": 8})
            self.processor = None
            self.mm_processor_kwargs = {}
            self.reward_loop_worker_handles = None
            self.teacher_intervention_enabled = True
            self.teacher_intervention_config = DistillationInterventionConfig(
                enabled=True,
                student_segment_length=2,
                teacher_segment_length=2,
                gap_metric="abs",
                gap_threshold=1.0,
                max_interventions_per_sample=1,
                train_on_teacher_segments=False,
                log_examples_per_worker=0,
            )
            self.teacher_server_manager = _FakeInterventionTeacherManager()
            self.llm_client = _FakeRepairStudentClient()
            self.distillation_config = SimpleNamespace(
                distillation_loss=SimpleNamespace(topk=64, loss_settings=SimpleNamespace(use_topk=False))
            )
            self._teacher_intervention_logged_samples = 0

    worker = _DummyWorker()
    output = AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[1, 2, 3, 4, 5, 6],
        response_mask=[1, 1, 1, 1, 1, 1],
        response_logprobs=[-0.1] * 6,
        metrics=AgentLoopMetrics(),
        extra_fields={"rollout_request_id": "sample-1"},
    )

    internal = await AgentLoopWorker._agent_loop_postprocess(
        worker,
        output,
        validate=False,
        sampling_params={"temperature": 0.8, "top_p": 0.95, "logprobs": True},
        raw_prompt=[{"role": "user", "content": "hi"}],
        data_source="openai/gsm8k",
    )

    torch.testing.assert_close(
        internal.response_ids[0],
        torch.tensor([1, 2, 30, 31, 40, 41, 0, 0], dtype=torch.long),
    )
    torch.testing.assert_close(
        internal.response_mask[0],
        torch.tensor([1, 1, 0, 0, 1, 1, 0, 0], dtype=torch.long),
    )
    assert internal.teacher_logprobs is not None
    assert worker.teacher_server_manager.scored_sequences == [
        [101, 102, 1, 2, 3, 4, 5, 6],
        [101, 102, 1, 2, 30, 31, 40, 41],
    ]
    assert worker.teacher_server_manager.generated_prefixes == [[101, 102, 1, 2]]
    assert worker.llm_client.calls == [
        {
            "request_id": "sample-1",
            "prompt_ids": [101, 102, 1, 2, 30, 31],
            "sampling_params": {"temperature": 0.8, "top_p": 0.95, "logprobs": True, "max_tokens": 4},
        }
    ]
    assert internal.extra_fields["teacher_intervention_count"] == 1
    assert internal.extra_fields["teacher_logprob_nonfinite_count"] == 1
    assert internal.extra_fields["teacher_logprob_nonfinite_positions"] == [4]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("teacher_segment_loss_mode", "expected_response_mask", "expected_teacher_segment_mask"),
    [
        ("opd", [1, 1, 1, 1, 1, 1, 0, 0], None),
        ("sft", [1, 1, 0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0]),
    ],
)
async def test_teacher_intervention_train_teacher_segments_drops_rollout_logprobs_batch_wide_on_cpu(
    teacher_segment_loss_mode,
    expected_response_mask,
    expected_teacher_segment_mask,
):
    class _DummyWorker:
        _compute_multi_modal_inputs = AgentLoopWorker._compute_multi_modal_inputs
        _compute_position_ids = AgentLoopWorker._compute_position_ids
        _get_mm_processor_kwargs = AgentLoopWorker._get_mm_processor_kwargs
        _compute_score = AgentLoopWorker._compute_score
        _teacher_routing_key = AgentLoopWorker._teacher_routing_key
        _score_sequence_for_intervention = AgentLoopWorker._score_sequence_for_intervention
        _generate_student_suffix = AgentLoopWorker._generate_student_suffix
        _maybe_repair_with_teacher = AgentLoopWorker._maybe_repair_with_teacher
        _compute_teacher_logprobs = AgentLoopWorker._compute_teacher_logprobs
        distillation_enabled = True
        teacher_key = "data_source"

        def __init__(self):
            self.tokenizer = _FakeTokenizer()
            self.rollout_config = OmegaConf.create({"prompt_length": 4, "response_length": 8})
            self.processor = None
            self.mm_processor_kwargs = {}
            self.reward_loop_worker_handles = None
            self.teacher_intervention_enabled = True
            self.teacher_intervention_config = DistillationInterventionConfig(
                enabled=True,
                student_segment_length=2,
                teacher_segment_length=2,
                gap_metric="abs",
                gap_threshold=1.0,
                max_interventions_per_sample=1,
                train_on_teacher_segments=True,
                teacher_segment_loss_mode=teacher_segment_loss_mode,
                log_examples_per_worker=0,
            )
            self.teacher_server_manager = _FakeInterventionTeacherManager()
            self.llm_client = _FakeRepairStudentClient()
            self.distillation_config = SimpleNamespace(
                distillation_loss=SimpleNamespace(topk=64, loss_settings=SimpleNamespace(use_topk=False))
            )
            self._teacher_intervention_logged_samples = 0

    worker = _DummyWorker()
    intervened = AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[1, 2, 3, 4, 5, 6],
        response_mask=[1, 1, 1, 1, 1, 1],
        response_logprobs=[-0.1] * 6,
        metrics=AgentLoopMetrics(),
        extra_fields={"rollout_request_id": "intervened"},
    )
    untouched = AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[7, 8, 9, 10],
        response_mask=[1, 1, 1, 1],
        response_logprobs=[-0.1] * 4,
        metrics=AgentLoopMetrics(),
        extra_fields={"rollout_request_id": "untouched"},
    )

    internal_intervened = await AgentLoopWorker._agent_loop_postprocess(
        worker,
        intervened,
        validate=False,
        sampling_params={"temperature": 0.8, "top_p": 0.95, "logprobs": True},
        raw_prompt=[{"role": "user", "content": "hi"}],
        data_source="openai/gsm8k",
    )
    internal_untouched = await AgentLoopWorker._agent_loop_postprocess(
        worker,
        untouched,
        validate=False,
        sampling_params={"temperature": 0.8, "top_p": 0.95, "logprobs": True},
        raw_prompt=[{"role": "user", "content": "hi"}],
        data_source="openai/gsm8k",
    )

    assert internal_intervened.response_logprobs is None
    assert internal_untouched.response_logprobs is None
    assert internal_intervened.extra_fields["teacher_intervention_count"] == 1
    assert internal_untouched.extra_fields["teacher_intervention_count"] == 0
    torch.testing.assert_close(
        internal_intervened.response_mask[0],
        torch.tensor(expected_response_mask, dtype=torch.long),
    )
    if expected_teacher_segment_mask is None:
        assert internal_intervened.teacher_segment_mask is None
        assert internal_untouched.teacher_segment_mask is None
    else:
        torch.testing.assert_close(
            internal_intervened.teacher_segment_mask[0],
            torch.tensor(expected_teacher_segment_mask, dtype=torch.long),
        )
        torch.testing.assert_close(internal_untouched.teacher_segment_mask, torch.zeros(1, 8, dtype=torch.long))

    merged = AgentLoopWorker._postprocess(worker, inputs=[internal_intervened, internal_untouched])
    assert "rollout_log_probs" not in merged.batch.keys()
    assert ("teacher_segment_mask" in merged.batch.keys()) is (expected_teacher_segment_mask is not None)


@pytest.mark.asyncio
async def test_teacher_intervention_empty_response_emits_metadata_on_cpu():
    worker = _DirectInterventionWorker(
        teacher_manager=_FixedGapTeacherManager(TeacherGenerationResult(token_ids=[30], finish_reason="stop"))
    )
    output = AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[],
        response_mask=[],
        response_logprobs=None,
        metrics=AgentLoopMetrics(),
        extra_fields={"rollout_request_id": "empty-response"},
    )

    await AgentLoopWorker._maybe_repair_with_teacher(
        worker,
        output,
        validate=False,
        sampling_params={"temperature": 0.8},
        sample_kwargs={"data_source": "openai/gsm8k"},
    )

    assert output.extra_fields["teacher_intervention_count"] == 0
    assert output.extra_fields["teacher_interventions"] == []
    assert output.extra_fields["teacher_intervention_skipped"] == []


@pytest.mark.asyncio
async def test_teacher_intervention_overlap_diagnostics_requests_topk_and_keeps_sampled_fields_on_cpu():
    teacher_manager = _FixedGapTeacherManager(
        TeacherGenerationResult(token_ids=[30, 31], stop_reason="completed", finish_reason="length"),
        high_gap_start=0,
        topk_width=4,
    )
    worker = _DirectInterventionWorker(
        teacher_manager=teacher_manager,
        response_length=6,
        diagnostics_topk=4,
    )
    output = AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[1, 2, 3, 4, 5, 6],
        response_mask=[1, 1, 1, 1, 1, 1],
        response_logprobs=[-0.1] * 6,
        metrics=AgentLoopMetrics(),
        extra_fields={"rollout_request_id": "overlap-diagnostics"},
    )

    await AgentLoopWorker._maybe_repair_with_teacher(
        worker,
        output,
        validate=False,
        sampling_params={"temperature": 0.8, "top_p": 0.95, "logprobs": True},
        sample_kwargs={"data_source": "openai/gsm8k"},
    )

    assert teacher_manager.prompt_logprobs_calls == [4, 4]
    assert output.extra_fields["teacher_ids"].shape[-1] == 4
    assert output.extra_fields["teacher_logprobs"].shape[-1] == 4
    assert output.extra_fields["teacher_sampled_ids"].shape[-1] == 1
    assert output.extra_fields["teacher_sampled_logprobs"].shape[-1] == 1
    assert output.extra_fields["teacher_intervention_count"] == 1
    intervention = output.extra_fields["teacher_interventions"][0]
    assert intervention["rewrite_exact_position_overlap"] == 0.0
    assert intervention["rewrite_lcs_ratio"] == 0.0
    assert intervention["rewrite_length_ratio"] == 1.0


@pytest.mark.asyncio
async def test_teacher_intervention_skips_teacher_generation_with_tokenizer_out_of_range_ids_on_cpu():
    teacher_manager = _FixedGapTeacherManager(
        TeacherGenerationResult(token_ids=[30, 100], stop_reason="completed", finish_reason="length"),
        high_gap_start=0,
    )
    worker = _DirectInterventionWorker(teacher_manager=teacher_manager, response_length=4)
    output = AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[1, 2, 3, 4],
        response_mask=[1, 1, 1, 1],
        response_logprobs=[-0.1] * 4,
        metrics=AgentLoopMetrics(),
        extra_fields={"rollout_request_id": "invalid-teacher-token"},
    )

    await AgentLoopWorker._maybe_repair_with_teacher(
        worker,
        output,
        validate=False,
        sampling_params={"temperature": 0.8, "logprobs": True},
        sample_kwargs={"data_source": "openai/gsm8k"},
    )

    assert output.response_ids == [1, 2, 3, 4]
    assert output.extra_fields["teacher_intervention_count"] == 0
    assert output.extra_fields["teacher_intervention_skipped"][0]["reason"] == "teacher_generation_invalid_token_ids"
    assert output.extra_fields["teacher_intervention_skipped"][0]["invalid_token_ids"] == [100]
    assert output.extra_fields["teacher_intervention_skipped"][0]["tokenizer_vocab_size"] == 100
    assert worker.llm_client.calls == []


@pytest.mark.asyncio
async def test_teacher_intervention_skips_zero_trainable_repair_and_preserves_original_on_cpu():
    teacher_manager = _FixedGapTeacherManager(
        TeacherGenerationResult(token_ids=[30], stop_reason="completed", finish_reason="stop"),
        high_gap_start=0,
    )
    worker = _DirectInterventionWorker(teacher_manager=teacher_manager, response_length=4)
    output = AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[1, 2, 3, 4],
        response_mask=[1, 1, 1, 1],
        response_logprobs=[-0.1] * 4,
        metrics=AgentLoopMetrics(),
        extra_fields={"rollout_request_id": "zero-trainable"},
    )

    await AgentLoopWorker._maybe_repair_with_teacher(
        worker,
        output,
        validate=False,
        sampling_params={"temperature": 0.8, "logprobs": True},
        sample_kwargs={"data_source": "openai/gsm8k"},
    )

    assert output.response_ids == [1, 2, 3, 4]
    assert output.response_mask == [1, 1, 1, 1]
    assert output.extra_fields["teacher_intervention_count"] == 0
    assert output.extra_fields["teacher_intervention_skipped"] == [
        {
            "reason": "insufficient_trainable_tokens",
            "attempt": 0,
            "start": 0,
            "end": 2,
            "candidate_tokens": 1,
            "candidate_trainable_tokens": 0,
            "teacher_tokens": 1,
            "student_suffix_tokens": 0,
        }
    ]
    assert worker.llm_client.calls == []


@pytest.mark.asyncio
async def test_teacher_intervention_teacher_stop_prevents_student_suffix_on_cpu():
    teacher_manager = _FixedGapTeacherManager(
        TeacherGenerationResult(token_ids=[30], stop_reason="completed", finish_reason="stop"),
        high_gap_start=2,
    )
    worker = _DirectInterventionWorker(teacher_manager=teacher_manager, response_length=6)
    output = AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[1, 2, 3, 4, 5, 6],
        response_mask=[1, 1, 1, 1, 1, 1],
        response_logprobs=[-0.1] * 6,
        metrics=AgentLoopMetrics(),
        extra_fields={"rollout_request_id": "teacher-stop"},
    )

    await AgentLoopWorker._maybe_repair_with_teacher(
        worker,
        output,
        validate=False,
        sampling_params={"temperature": 0.8, "logprobs": True},
        sample_kwargs={"data_source": "openai/gsm8k"},
    )

    assert output.response_ids == [1, 2, 30]
    assert output.response_mask == [1, 1, 0]
    assert worker.llm_client.calls == []
    assert output.extra_fields["teacher_intervention_count"] == 1
    assert output.extra_fields["teacher_interventions"][0]["terminated"] is True
    assert output.extra_fields["teacher_interventions"][0]["student_suffix_tokens"] == 0
    assert output.extra_fields["teacher_interventions"][0]["local_effect_pair_available"] is False
    assert output.extra_fields["teacher_interventions"][0]["local_effect_unavailable_reason"] == "terminal_response"


@pytest.mark.asyncio
async def test_teacher_intervention_length_stop_allows_student_suffix_on_cpu():
    teacher_manager = _FixedGapTeacherManager(
        TeacherGenerationResult(token_ids=[30, 31], stop_reason="completed", finish_reason="length"),
        high_gap_start=2,
    )
    worker = _DirectInterventionWorker(teacher_manager=teacher_manager, response_length=6)
    output = AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[1, 2, 3, 4, 5, 6],
        response_mask=[1, 1, 1, 1, 1, 1],
        response_logprobs=[-0.1] * 6,
        metrics=AgentLoopMetrics(),
        extra_fields={"rollout_request_id": "teacher-length"},
    )

    await AgentLoopWorker._maybe_repair_with_teacher(
        worker,
        output,
        validate=False,
        sampling_params={"temperature": 0.8, "top_p": 0.95, "logprobs": True},
        sample_kwargs={"data_source": "openai/gsm8k"},
    )

    assert output.response_ids == [1, 2, 30, 31, 40, 41]
    assert output.response_mask == [1, 1, 0, 0, 1, 1]
    assert worker.llm_client.calls == [
        {
            "request_id": "teacher-length",
            "prompt_ids": [101, 102, 1, 2, 30, 31],
            "sampling_params": {"temperature": 0.8, "top_p": 0.95, "logprobs": True, "max_tokens": 2},
        }
    ]
    assert output.extra_fields["teacher_interventions"][0]["terminated"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("generation_result", "reason"),
    [
        (
            TeacherGenerationResult(token_ids=[], stop_reason="completed", finish_reason="stop"),
            "teacher_generation_empty",
        ),
        (
            TeacherGenerationResult(token_ids=[], stop_reason="aborted", finish_reason="abort"),
            "teacher_generation_aborted",
        ),
    ],
)
async def test_teacher_intervention_empty_or_aborted_teacher_generation_skips_repair_on_cpu(
    generation_result: TeacherGenerationResult,
    reason: str,
):
    teacher_manager = _FixedGapTeacherManager(generation_result, high_gap_start=0)
    worker = _DirectInterventionWorker(teacher_manager=teacher_manager, response_length=4)
    output = AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[1, 2, 3, 4],
        response_mask=[1, 1, 1, 1],
        response_logprobs=[-0.1] * 4,
        metrics=AgentLoopMetrics(),
        extra_fields={"rollout_request_id": "bad-teacher-generation"},
    )

    await AgentLoopWorker._maybe_repair_with_teacher(
        worker,
        output,
        validate=False,
        sampling_params={"temperature": 0.8, "logprobs": True},
        sample_kwargs={"data_source": "openai/gsm8k"},
    )

    assert output.response_ids == [1, 2, 3, 4]
    assert output.response_mask == [1, 1, 1, 1]
    assert output.extra_fields["teacher_intervention_count"] == 0
    assert output.extra_fields["teacher_intervention_skipped"][0]["reason"] == reason
    assert worker.llm_client.calls == []


@pytest.mark.asyncio
async def test_teacher_intervention_rejects_routed_experts_on_cpu():
    worker = _DirectInterventionWorker(
        teacher_manager=_FixedGapTeacherManager(TeacherGenerationResult(token_ids=[30], finish_reason="stop"))
    )
    output = AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[1, 2],
        response_mask=[1, 1],
        response_logprobs=[-0.1, -0.1],
        routed_experts=torch.zeros((4, 2, 1), dtype=torch.int64),
        metrics=AgentLoopMetrics(),
        extra_fields={"rollout_request_id": "routed-experts"},
    )

    with pytest.raises(NotImplementedError, match="routed_experts"):
        await AgentLoopWorker._maybe_repair_with_teacher(
            worker,
            output,
            validate=False,
            sampling_params={"temperature": 0.8, "logprobs": True},
            sample_kwargs={"data_source": "openai/gsm8k"},
        )


def test_teacher_intervention_rejects_moe_config_and_routing_replay_on_cpu():
    assert _is_moe_like_hf_config(SimpleNamespace(model_type="qwen3_moe", num_experts_per_tok=2))

    worker = object.__new__(AgentLoopWorker)
    worker.teacher_intervention_enabled = True
    worker.rollout_config = SimpleNamespace(enable_rollout_routing_replay=False)
    worker.model_config = SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen3_moe", num_experts_per_tok=2))
    with pytest.raises(NotImplementedError, match="MoE"):
        AgentLoopWorker._validate_teacher_intervention_compatibility(worker)

    worker.model_config = SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen3"))
    worker.rollout_config = SimpleNamespace(enable_rollout_routing_replay=True)
    with pytest.raises(NotImplementedError, match="enable_rollout_routing_replay"):
        AgentLoopWorker._validate_teacher_intervention_compatibility(worker)


def test_teacher_intervention_k1_gap_uses_teacher_minus_student_on_cpu():
    gaps = _compute_intervention_gap(
        student_logprobs=torch.tensor([-5.0, -4.0]),
        teacher_logprobs=torch.tensor([-0.5, -0.25]),
        metric="k1",
    )
    bad_segment = _select_intervention_segment(
        gaps,
        response_mask=[1, 1],
        intervention_config=DistillationInterventionConfig(
            enabled=True,
            student_segment_length=2,
            gap_metric="k1",
            gap_threshold=1.0,
        ),
    )

    torch.testing.assert_close(gaps, torch.tensor([4.5, 3.75]))
    assert bad_segment == (0, 2, pytest.approx(4.125))


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        ("kl", [1.0, -0.5]),
        ("k1", [1.0, -0.5]),
        ("abs", [1.0, 0.5]),
        ("mse", [0.5, 0.125]),
        ("k2", [0.5, 0.125]),
        ("low_var_kl", [np.e - 2.0, np.exp(-0.5) - 0.5]),
        ("k3", [np.e - 2.0, np.exp(-0.5) - 0.5]),
    ],
)
def test_teacher_intervention_legacy_sampled_gap_metrics_are_unchanged_on_cpu(metric, expected):
    gaps = _compute_intervention_gap(
        student_logprobs=torch.tensor([-2.0, -0.5]),
        teacher_logprobs=torch.tensor([-1.0, -1.0]),
        metric=metric,
    )

    torch.testing.assert_close(gaps, torch.tensor(expected, dtype=torch.float32))


def test_teacher_intervention_teacher_rejection_gap_is_directional_and_capped_on_cpu():
    gaps = _compute_intervention_gap(
        student_logprobs=torch.tensor([-0.1, -5.0, -0.1]),
        teacher_logprobs=torch.tensor([-5.0, -0.1, -100.0]),
        metric="teacher_rejection",
        teacher_rejection_token_cap=10.0,
    )

    torch.testing.assert_close(gaps, torch.tensor([4.9, 0.0, 10.0]))


def test_teacher_intervention_teacher_rejection_respects_aggregation_and_strict_threshold_on_cpu():
    gaps = torch.tensor([1.0, 3.0])
    mean_config = DistillationInterventionConfig(
        enabled=True,
        student_segment_length=2,
        gap_metric="teacher_rejection",
        gap_aggregation="mean",
        gap_threshold=2.0,
    )
    max_config = DistillationInterventionConfig(
        enabled=True,
        student_segment_length=2,
        gap_metric="teacher_rejection",
        gap_aggregation="max",
        gap_threshold=2.0,
    )

    assert _select_intervention_segment(gaps, [1, 1], mean_config) is None
    assert _select_intervention_segment(gaps, [1, 1], max_config) == (0, 2, pytest.approx(3.0))


def test_teacher_rejection_without_persistence_keeps_legacy_local_trigger_on_cpu():
    config = DistillationInterventionConfig(
        enabled=True,
        student_segment_length=2,
        gap_metric="teacher_rejection",
        gap_threshold=1.0,
        teacher_rejection_persistence_enabled=False,
    )

    assert _select_intervention_segment(torch.tensor([2.0, 2.0]), [1, 1], config) == (
        0,
        2,
        pytest.approx(2.0),
    )


def test_persistent_teacher_rejection_skips_transient_alert_and_finds_later_drift_on_cpu():
    config = DistillationInterventionConfig(
        enabled=True,
        student_segment_length=2,
        gap_metric="teacher_rejection",
        gap_threshold=1.0,
        teacher_rejection_persistence_enabled=True,
        teacher_rejection_persistence_window_tokens=2,
        teacher_rejection_persistence_threshold=1.0,
        teacher_rejection_persistence_min_valid_tokens=2,
    )
    gaps = torch.tensor([2.0, 2.0, 0.0, 0.0, 3.0, 3.0, 2.0, 2.0])

    assert _select_intervention_segment(gaps, [1] * 8, config) == (4, 6, pytest.approx(3.0))


def test_persistent_teacher_rejection_requires_strict_future_threshold_on_cpu():
    config = DistillationInterventionConfig(
        enabled=True,
        student_segment_length=2,
        gap_metric="teacher_rejection",
        gap_threshold=1.0,
        teacher_rejection_persistence_enabled=True,
        teacher_rejection_persistence_window_tokens=2,
        teacher_rejection_persistence_threshold=None,
        teacher_rejection_persistence_min_valid_tokens=2,
    )

    assert _select_intervention_segment(torch.tensor([2.0, 2.0, 1.0, 1.0]), [1] * 4, config) is None
    assert _select_intervention_segment(torch.tensor([2.0, 2.0, 1.1, 1.1]), [1] * 4, config) == (
        0,
        2,
        pytest.approx(2.0),
    )


def test_persistent_teacher_rejection_requires_minimum_valid_future_tokens_on_cpu():
    config = DistillationInterventionConfig(
        enabled=True,
        student_segment_length=2,
        gap_metric="teacher_rejection",
        gap_threshold=1.0,
        teacher_rejection_persistence_enabled=True,
        teacher_rejection_persistence_window_tokens=2,
        teacher_rejection_persistence_threshold=1.0,
        teacher_rejection_persistence_min_valid_tokens=2,
    )
    gaps = torch.tensor([2.0, 2.0, 3.0, 3.0])

    assert _select_intervention_segment(gaps, [1, 1, 1, 0], config) is None


@pytest.mark.asyncio
async def test_teacher_rejection_records_boundary_aligned_on_policy_local_effect_on_cpu():
    class _ImprovingTeacherManager(_FixedGapTeacherManager):
        async def compute_teacher_logprobs_single(self, *args, **kwargs):
            if not self.scored_sequences:
                self.high_gap_start = 0
                self.high_gap_width = 4
            else:
                self.high_gap_start = 99
            return await super().compute_teacher_logprobs_single(*args, **kwargs)

    teacher_manager = _ImprovingTeacherManager(
        TeacherGenerationResult(token_ids=[30], stop_reason="completed", finish_reason="length")
    )
    worker = _DirectInterventionWorker(
        teacher_manager=teacher_manager,
        response_length=6,
        student_segment_length=2,
        teacher_segment_length=1,
        gap_metric="teacher_rejection",
        gap_threshold=1.0,
    )
    output = AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[1, 2, 3, 4],
        response_mask=[1, 1, 1, 1],
        response_logprobs=[-0.1] * 4,
        metrics=AgentLoopMetrics(),
        extra_fields={"rollout_request_id": "teacher-rejection-local-effect"},
    )

    await AgentLoopWorker._maybe_repair_with_teacher(
        worker,
        output,
        validate=False,
        sampling_params={"temperature": 0.8, "logprobs": True},
        sample_kwargs={"data_source": "openai/gsm8k"},
    )

    intervention = output.extra_fields["teacher_interventions"][0]
    assert output.response_ids == [30, 40, 41]
    assert len(teacher_manager.scored_sequences) == 2
    assert len(worker.llm_client.calls) == 1
    assert intervention["local_effect_pair_available"] is True
    assert intervention["local_effect_no_current_intervention_start"] == 2
    assert intervention["local_effect_after_intervention_start"] == 1
    assert intervention["local_effect_no_current_intervention_tokens"] == 2
    assert intervention["local_effect_after_intervention_tokens"] == 2
    assert intervention["local_effect_no_current_intervention_gap"] == pytest.approx(4.9)
    assert intervention["local_effect_after_intervention_gap"] == pytest.approx(0.0)
    assert intervention["local_effect_gap_reduction"] == pytest.approx(4.9)
    assert intervention["local_effect_no_current_intervention_d"] == pytest.approx(4.9)
    assert intervention["local_effect_after_intervention_d"] == pytest.approx(0.0)
    assert intervention["local_effect_d_reduction"] == pytest.approx(4.9)


def test_teacher_intervention_full_kl_uses_full_vocab_support_on_cpu():
    class _TinyTokenizer:
        def __len__(self):
            return 3

    ids = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.int32)
    student_probs = torch.tensor([[0.5, 0.3, 0.2], [0.5, 0.3, 0.2]], dtype=torch.float32)
    teacher_probs = torch.tensor([[0.2, 0.3, 0.5], [0.2, 0.3, 0.5]], dtype=torch.float32)
    gaps = _compute_distribution_intervention_gap(
        student_result=TeacherLogprobScoringResult(ids=ids, logprobs=torch.log(student_probs)),
        teacher_result=TeacherLogprobScoringResult(ids=ids, logprobs=torch.log(teacher_probs)),
        prompt_len=1,
        response_len=1,
        metric="full_kl",
        tokenizer=_TinyTokenizer(),
        log_prob_floor=-10.0,
    )

    expected = (student_probs[0] * (student_probs[0].log() - teacher_probs[0].log())).sum()
    torch.testing.assert_close(gaps, expected.unsqueeze(0))


@pytest.mark.asyncio
async def test_teacher_intervention_topk_distribution_kl_triggers_repair_on_cpu():
    teacher_manager = _DistributionGapTeacherManager()
    worker = _DirectInterventionWorker(
        teacher_manager=teacher_manager,
        response_length=4,
        student_segment_length=1,
        teacher_segment_length=1,
        gap_metric="topk_distribution_kl",
        gap_distribution_topk=2,
        gap_threshold=1.0,
    )
    worker.llm_client = _DistributionGapStudentClient()
    output = AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[1, 2, 3, 4],
        response_mask=[1, 1, 1, 1],
        response_logprobs=[-0.1] * 4,
        metrics=AgentLoopMetrics(),
        extra_fields={"rollout_request_id": "distribution-kl"},
    )

    await AgentLoopWorker._maybe_repair_with_teacher(
        worker,
        output,
        validate=False,
        sampling_params={"temperature": 0.8, "logprobs": True},
        sample_kwargs={"data_source": "openai/gsm8k"},
    )

    assert teacher_manager.prompt_logprobs_calls[0] == 2
    assert worker.llm_client.calls[0]["sampling_params"]["prompt_logprobs"] == 2
    assert teacher_manager.generated_prefixes == [[101, 102]]
    assert output.response_ids == [30, 40, 41]
    assert output.response_mask == [0, 1, 1]
    assert output.extra_fields["teacher_intervention_count"] == 1
    assert output.extra_fields["teacher_interventions"][0]["start"] == 0


@pytest.mark.asyncio
async def test_teacher_intervention_topk_no_repair_reuses_scoring_for_training_logprobs_on_cpu():
    teacher_manager = _FixedGapTeacherManager(
        TeacherGenerationResult(token_ids=[30], finish_reason="stop"),
        topk_width=3,
    )
    worker = _DirectInterventionWorker(
        teacher_manager=teacher_manager,
        response_length=4,
        gap_threshold=999.0,
        topk=3,
    )
    output = AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[1, 2, 3, 4],
        response_mask=[1, 1, 1, 1],
        response_logprobs=[-0.1] * 4,
        metrics=AgentLoopMetrics(),
        extra_fields={"rollout_request_id": "topk-no-repair"},
    )

    await AgentLoopWorker._maybe_repair_with_teacher(
        worker,
        output,
        validate=False,
        sampling_params={"temperature": 0.8, "logprobs": True},
        sample_kwargs={"data_source": "openai/gsm8k"},
    )
    assert teacher_manager.prompt_logprobs_calls == [3]
    assert output.extra_fields["teacher_logprobs"].shape[-1] == 3

    await AgentLoopWorker._compute_teacher_logprobs(
        worker,
        output,
        prompt_ids=output.prompt_ids,
        response_ids=output.response_ids,
        validate=False,
        sample_kwargs={"data_source": "openai/gsm8k"},
    )

    assert teacher_manager.prompt_logprobs_calls == [3]


@pytest.mark.asyncio
async def test_teacher_intervention_topk_repair_reuses_final_scoring_for_training_logprobs_on_cpu():
    teacher_manager = _FixedGapTeacherManager(
        TeacherGenerationResult(token_ids=[30, 31], stop_reason="completed", finish_reason="length"),
        high_gap_start=2,
        topk_width=2,
    )
    worker = _DirectInterventionWorker(teacher_manager=teacher_manager, response_length=6, topk=2)
    output = AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[1, 2, 3, 4, 5, 6],
        response_mask=[1, 1, 1, 1, 1, 1],
        response_logprobs=[-0.1] * 6,
        metrics=AgentLoopMetrics(),
        extra_fields={"rollout_request_id": "topk-repair"},
    )

    await AgentLoopWorker._maybe_repair_with_teacher(
        worker,
        output,
        validate=False,
        sampling_params={"temperature": 0.8, "logprobs": True},
        sample_kwargs={"data_source": "openai/gsm8k"},
    )
    assert teacher_manager.prompt_logprobs_calls == [2, 2]
    assert output.extra_fields["teacher_logprobs"].shape[-1] == 2

    await AgentLoopWorker._compute_teacher_logprobs(
        worker,
        output,
        prompt_ids=output.prompt_ids,
        response_ids=output.response_ids,
        validate=False,
        sample_kwargs={"data_source": "openai/gsm8k"},
    )

    assert teacher_manager.prompt_logprobs_calls == [2, 2]


@pytest.mark.asyncio
async def test_teacher_intervention_skips_invalid_thinking_block_repair_on_cpu():
    teacher_manager = _FixedGapTeacherManager(
        TeacherGenerationResult(token_ids=[92], stop_reason="completed", finish_reason="stop"),
        high_gap_start=0,
    )
    worker = _DirectInterventionWorker(teacher_manager=teacher_manager, response_length=4)
    output = AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[1, 2, 3, 4],
        response_mask=[1, 1, 1, 1],
        response_logprobs=[-0.1] * 4,
        metrics=AgentLoopMetrics(),
        extra_fields={"rollout_request_id": "invalid-thinking"},
    )

    await AgentLoopWorker._maybe_repair_with_teacher(
        worker,
        output,
        validate=False,
        sampling_params={"temperature": 0.8, "logprobs": True},
        sample_kwargs={"data_source": "openai/gsm8k"},
    )

    assert output.response_ids == [1, 2, 3, 4]
    assert output.extra_fields["teacher_intervention_count"] == 0
    assert output.extra_fields["teacher_intervention_skipped"][0]["reason"] == "invalid_thinking_block"
    assert output.extra_fields["teacher_intervention_skipped"][0]["thinking_reason"] == "unmatched_close"


@pytest.mark.asyncio
async def test_teacher_intervention_allows_response_close_for_prompt_open_thinking_on_cpu():
    teacher_manager = _FixedGapTeacherManager(
        TeacherGenerationResult(token_ids=[92], stop_reason="completed", finish_reason="stop"),
        high_gap_start=0,
    )
    worker = _DirectInterventionWorker(
        teacher_manager=teacher_manager,
        response_length=4,
        teacher_segment_length=1,
        train_on_teacher_segments=True,
    )
    output = AgentLoopOutput(
        prompt_ids=[101, 91],
        response_ids=[1, 2, 3, 4],
        response_mask=[1, 1, 1, 1],
        response_logprobs=[-0.1] * 4,
        metrics=AgentLoopMetrics(),
        extra_fields={"rollout_request_id": "prompt-open-thinking"},
    )

    await AgentLoopWorker._maybe_repair_with_teacher(
        worker,
        output,
        validate=False,
        sampling_params={"temperature": 0.8, "logprobs": True},
        sample_kwargs={"data_source": "openai/gsm8k"},
    )

    assert output.response_ids == [92]
    assert output.extra_fields["teacher_intervention_count"] == 1
    assert output.extra_fields["teacher_intervention_skipped"] == []


@pytest.mark.asyncio
async def test_teacher_intervention_skips_terminal_open_thinking_block_on_cpu():
    teacher_manager = _FixedGapTeacherManager(
        TeacherGenerationResult(token_ids=[91, 30], stop_reason="completed", finish_reason="stop"),
        high_gap_start=0,
    )
    worker = _DirectInterventionWorker(teacher_manager=teacher_manager, response_length=4)
    output = AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[1, 2, 3, 4],
        response_mask=[1, 1, 1, 1],
        response_logprobs=[-0.1] * 4,
        metrics=AgentLoopMetrics(),
        extra_fields={"rollout_request_id": "terminal-open-thinking"},
    )

    await AgentLoopWorker._maybe_repair_with_teacher(
        worker,
        output,
        validate=False,
        sampling_params={"temperature": 0.8, "logprobs": True},
        sample_kwargs={"data_source": "openai/gsm8k"},
    )

    assert output.response_ids == [1, 2, 3, 4]
    assert output.extra_fields["teacher_intervention_count"] == 0
    assert output.extra_fields["teacher_intervention_skipped"][0]["reason"] == "terminal_open_thinking_block"


@pytest.mark.asyncio
async def test_teacher_intervention_length_open_thinking_allows_student_suffix_on_cpu():
    teacher_manager = _FixedGapTeacherManager(
        TeacherGenerationResult(token_ids=[91], stop_reason="completed", finish_reason="length"),
        high_gap_start=2,
    )
    worker = _DirectInterventionWorker(teacher_manager=teacher_manager, response_length=6)
    output = AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[1, 2, 3, 4, 5, 6],
        response_mask=[1, 1, 1, 1, 1, 1],
        response_logprobs=[-0.1] * 6,
        metrics=AgentLoopMetrics(),
        extra_fields={"rollout_request_id": "length-open-thinking"},
    )

    await AgentLoopWorker._maybe_repair_with_teacher(
        worker,
        output,
        validate=False,
        sampling_params={"temperature": 0.8, "logprobs": True},
        sample_kwargs={"data_source": "openai/gsm8k"},
    )

    assert output.response_ids == [1, 2, 91, 40, 41]
    assert output.extra_fields["teacher_intervention_count"] == 1
    assert output.extra_fields["teacher_interventions"][0]["terminated"] is False


@pytest.mark.asyncio
async def test_teacher_intervention_sft_masks_only_inserted_teacher_tokens_on_cpu():
    teacher_manager = _FixedGapTeacherManager(
        TeacherGenerationResult(token_ids=[30], stop_reason="completed", finish_reason="length"),
        high_gap_start=2,
    )
    worker = _DirectInterventionWorker(
        teacher_manager=teacher_manager,
        response_length=6,
        train_on_teacher_segments=True,
        teacher_segment_loss_mode="sft",
    )
    output = AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[1, 2, 3, 4, 5, 6],
        response_mask=[1, 1, 1, 1, 1, 1],
        response_logprobs=[-0.1] * 6,
        metrics=AgentLoopMetrics(),
        extra_fields={"rollout_request_id": "teacher-sft-mask"},
    )

    await AgentLoopWorker._maybe_repair_with_teacher(
        worker,
        output,
        validate=False,
        sampling_params={"temperature": 0.8, "logprobs": True},
        sample_kwargs={"data_source": "openai/gsm8k"},
    )

    assert output.response_ids == [1, 2, 30, 40, 41]
    assert output.response_mask == [1, 1, 0, 1, 1]
    assert output.teacher_segment_mask == [0, 0, 1, 0, 0]
    assert output.extra_fields["teacher_intervention_count"] == 1


@pytest.mark.asyncio
async def test_teacher_intervention_sft_emits_zero_mask_without_intervention_on_cpu():
    teacher_manager = _FixedGapTeacherManager(
        TeacherGenerationResult(token_ids=[30], stop_reason="completed", finish_reason="length"),
        high_gap_start=99,
    )
    worker = _DirectInterventionWorker(
        teacher_manager=teacher_manager,
        response_length=4,
        train_on_teacher_segments=True,
        teacher_segment_loss_mode="sft",
    )
    output = AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[1, 2, 3, 4],
        response_mask=[1, 1, 1, 1],
        response_logprobs=[-0.1] * 4,
        metrics=AgentLoopMetrics(),
        extra_fields={"rollout_request_id": "teacher-sft-no-intervention"},
    )

    await AgentLoopWorker._maybe_repair_with_teacher(
        worker,
        output,
        validate=False,
        sampling_params={"temperature": 0.8, "logprobs": True},
        sample_kwargs={"data_source": "openai/gsm8k"},
    )

    assert output.response_mask == [1, 1, 1, 1]
    assert output.teacher_segment_mask == [0, 0, 0, 0]
    assert output.extra_fields["teacher_intervention_count"] == 0


@pytest.mark.asyncio
async def test_teacher_intervention_sft_preserves_provenance_across_multiple_repairs_on_cpu():
    class _SequentialGapTeacherManager(_FixedGapTeacherManager):
        async def compute_teacher_logprobs_single(self, *args, **kwargs):
            gap_starts = [2, 4, 99]
            self.high_gap_start = gap_starts[min(len(self.scored_sequences), len(gap_starts) - 1)]
            return await super().compute_teacher_logprobs_single(*args, **kwargs)

    teacher_manager = _SequentialGapTeacherManager(
        TeacherGenerationResult(token_ids=[30], stop_reason="completed", finish_reason="length"),
        high_gap_width=1,
    )
    worker = _DirectInterventionWorker(
        teacher_manager=teacher_manager,
        response_length=8,
        student_segment_length=1,
        teacher_segment_length=1,
        max_interventions_per_sample=2,
        train_on_teacher_segments=True,
        teacher_segment_loss_mode="sft",
    )
    output = AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[1, 2, 3, 4, 5, 6],
        response_mask=[1, 1, 1, 1, 1, 1],
        response_logprobs=[-0.1] * 6,
        metrics=AgentLoopMetrics(),
        extra_fields={"rollout_request_id": "teacher-sft-multiple"},
    )

    await AgentLoopWorker._maybe_repair_with_teacher(
        worker,
        output,
        validate=False,
        sampling_params={"temperature": 0.8, "logprobs": True},
        sample_kwargs={"data_source": "openai/gsm8k"},
    )

    assert output.response_ids == [1, 2, 30, 40, 30, 40, 41]
    assert output.response_mask == [1, 1, 0, 1, 0, 1, 1]
    assert output.teacher_segment_mask == [0, 0, 1, 0, 1, 0, 0]
    assert output.extra_fields["teacher_intervention_count"] == 2
    first_intervention, second_intervention = output.extra_fields["teacher_interventions"]
    assert first_intervention["local_effect_pair_available"] is True
    assert second_intervention["local_effect_pair_available"] is False
    assert second_intervention["local_effect_unavailable_reason"] == "no_no_current_intervention_tokens"


@pytest.mark.asyncio
async def test_teacher_intervention_expands_segment_to_avoid_splitting_multitoken_thinking_tag_on_cpu():
    teacher_manager = _FixedGapTeacherManager(
        TeacherGenerationResult(token_ids=[30], stop_reason="completed", finish_reason="length"),
        high_gap_start=1,
    )
    worker = _DirectInterventionWorker(
        teacher_manager=teacher_manager,
        response_length=4,
        student_segment_length=1,
        thinking_start_text="<multi_think>",
        thinking_end_text="</multi_think>",
    )
    output = AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[71, 72, 11, 12],
        response_mask=[1, 1, 1, 1],
        response_logprobs=[-0.1] * 4,
        metrics=AgentLoopMetrics(),
        extra_fields={"rollout_request_id": "multitoken-thinking"},
    )

    await AgentLoopWorker._maybe_repair_with_teacher(
        worker,
        output,
        validate=False,
        sampling_params={"temperature": 0.8, "logprobs": True},
        sample_kwargs={"data_source": "openai/gsm8k"},
    )

    assert teacher_manager.generated_prefixes == [[101, 102]]
    assert output.extra_fields["teacher_interventions"][0]["start"] == 0


@pytest.mark.asyncio
async def test_teacher_intervention_logging_counter_resets_per_global_step_on_cpu():
    teacher_manager = _FixedGapTeacherManager(
        TeacherGenerationResult(token_ids=[30, 31], stop_reason="completed", finish_reason="length"),
        high_gap_start=2,
    )
    worker = _DirectInterventionWorker(
        teacher_manager=teacher_manager,
        response_length=6,
        log_examples_per_worker=1,
    )

    for request_id, global_steps in (("step-1-a", 1), ("step-1-b", 1), ("step-2-a", 2)):
        output = AgentLoopOutput(
            prompt_ids=[101, 102],
            response_ids=[1, 2, 3, 4, 5, 6],
            response_mask=[1, 1, 1, 1, 1, 1],
            response_logprobs=[-0.1] * 6,
            metrics=AgentLoopMetrics(),
            extra_fields={"rollout_request_id": request_id},
        )
        await AgentLoopWorker._maybe_repair_with_teacher(
            worker,
            output,
            validate=False,
            sampling_params={"temperature": 0.8, "logprobs": True},
            sample_kwargs={"data_source": "openai/gsm8k", "global_steps": global_steps},
        )

    assert worker._teacher_intervention_logged_step == 2
    assert worker._teacher_intervention_logged_samples == 1


@pytest.mark.asyncio
async def test_teacher_manager_rejects_missing_prompt_logprobs_on_cpu():
    class _BadTeacherClient:
        async def generate(
            self,
            request_id: str,
            *,
            prompt_ids: list[int],
            sampling_params: dict[str, Any],
            image_data: Optional[list[Any]] = None,
            video_data: Optional[list[Any]] = None,
            audio_data: Optional[list[Any]] = None,
            mm_processor_kwargs: Optional[dict[str, Any]] = None,
        ) -> TokenOutput:
            del request_id, prompt_ids, sampling_params, image_data, video_data, audio_data, mm_processor_kwargs
            return TokenOutput(token_ids=[1], extra_fields={})

    manager = object.__new__(AsyncTeacherLLMServerManager)
    manager.distillation_loss_config = SimpleNamespace(
        topk=64,
        loss_settings=SimpleNamespace(use_topk=False),
    )
    manager.teacher_key = "data_source"
    manager.teacher_model_configs = {
        "teacher": SimpleNamespace(inference=SimpleNamespace(temperature=1.0)),
    }
    manager.teacher_client = {"teacher": _BadTeacherClient()}

    with pytest.raises(RuntimeError, match="required prompt scoring fields"):
        await manager.compute_teacher_logprobs_single(
            sequence_ids=[101, 102, 1, 2],
            routing_key="teacher",
            request_id="bad-teacher-score",
            prompt_logprobs=0,
        )


@pytest.mark.asyncio
async def test_teacher_manager_sanitizes_nonfinite_prompt_logprobs_on_cpu(caplog):
    class _NonfiniteTeacherClient:
        async def generate(
            self,
            request_id: str,
            *,
            prompt_ids: list[int],
            sampling_params: dict[str, Any],
            image_data: Optional[list[Any]] = None,
            video_data: Optional[list[Any]] = None,
            audio_data: Optional[list[Any]] = None,
            mm_processor_kwargs: Optional[dict[str, Any]] = None,
        ) -> TokenOutput:
            del request_id, image_data, video_data, audio_data, mm_processor_kwargs
            assert prompt_ids == [101, 102, 1, 2]
            assert sampling_params["prompt_logprobs"] == 0
            return TokenOutput(
                token_ids=[1],
                extra_fields={
                    "prompt_ids": [[101], [102], [1], [2]],
                    "prompt_logprobs": [[-0.1], [float("-inf")], [float("nan")], [float("inf")]],
                },
            )

    manager = object.__new__(AsyncTeacherLLMServerManager)
    manager.distillation_loss_config = SimpleNamespace(
        topk=64,
        log_prob_min_clamp=-10.0,
        loss_settings=SimpleNamespace(use_topk=False),
    )
    manager.teacher_key = "data_source"
    manager.teacher_model_configs = {
        "teacher": SimpleNamespace(inference=SimpleNamespace(temperature=1.0)),
    }
    manager.teacher_client = {"teacher": _NonfiniteTeacherClient()}
    manager._nonfinite_teacher_logprob_warnings = 0

    with caplog.at_level("WARNING"):
        result = await manager.compute_teacher_logprobs_single(
            sequence_ids=[101, 102, 1, 2],
            routing_key="teacher",
            request_id="nonfinite-teacher-score",
            prompt_logprobs=0,
        )

    teacher_ids, teacher_logprobs = result.ids, result.logprobs
    torch.testing.assert_close(teacher_ids[:, 0], torch.tensor([101, 102, 1, 2], dtype=torch.int32))
    assert torch.isfinite(teacher_logprobs).all()
    torch.testing.assert_close(
        teacher_logprobs[:, 0],
        torch.tensor([-0.1, -10.0, -10.0, 0.0], dtype=teacher_logprobs.dtype),
    )
    assert result.nonfinite_count == 3
    assert result.nonfinite_sequence_positions == [1, 2, 3]
    assert result.replacement_value == -10.0
    assert "returned NaN/Inf values at 3 sequence positions" in caplog.text


@pytest.mark.asyncio
async def test_teacher_manager_parses_topk_and_sampled_prompt_logprobs_on_cpu():
    class _TopKTeacherClient:
        async def generate(
            self,
            request_id: str,
            *,
            prompt_ids: list[int],
            sampling_params: dict[str, Any],
            image_data: Optional[list[Any]] = None,
            video_data: Optional[list[Any]] = None,
            audio_data: Optional[list[Any]] = None,
            mm_processor_kwargs: Optional[dict[str, Any]] = None,
        ) -> TokenOutput:
            del request_id, image_data, video_data, audio_data, mm_processor_kwargs
            assert prompt_ids == [101, 102, 1, 2]
            assert sampling_params["prompt_logprobs"] == 2
            return TokenOutput(
                token_ids=[1],
                extra_fields={
                    "prompt_ids": [[101, 0], [102, 0], [1, 0], [2, 0]],
                    "prompt_logprobs": [[-0.1, -0.2], [-0.3, -0.4], [-0.5, -0.6], [0.0, 0.0]],
                    "sampled_prompt_ids": [[101], [102], [1], [2]],
                    "sampled_prompt_logprobs": [[-0.1], [float("-inf")], [-0.5], [0.0]],
                },
            )

    manager = object.__new__(AsyncTeacherLLMServerManager)
    manager.distillation_loss_config = SimpleNamespace(
        topk=2,
        log_prob_min_clamp=-10.0,
        loss_settings=SimpleNamespace(use_topk=True),
    )
    manager.teacher_key = "data_source"
    manager.teacher_model_configs = {
        "teacher": SimpleNamespace(inference=SimpleNamespace(temperature=1.0)),
    }
    manager.teacher_client = {"teacher": _TopKTeacherClient()}
    manager._nonfinite_teacher_logprob_warnings = 0

    result = await manager.compute_teacher_logprobs_single(
        sequence_ids=[101, 102, 1, 2],
        routing_key="teacher",
        request_id="topk-teacher-score",
        prompt_logprobs=2,
    )

    assert result.ids.shape == (4, 2)
    assert result.logprobs.shape == (4, 2)
    assert result.sampled_ids is not None
    assert result.sampled_logprobs is not None
    assert result.sampled_ids.shape == (4, 1)
    assert result.sampled_logprobs.shape == (4, 1)
    assert torch.isfinite(result.sampled_logprobs).all()
    torch.testing.assert_close(
        result.sampled_logprobs[:, 0],
        torch.tensor([-0.1, -10.0, -0.5, 0.0], dtype=result.sampled_logprobs.dtype),
    )
    assert result.nonfinite_count == 1
    assert result.nonfinite_sequence_positions == [1]


@pytest.mark.asyncio
async def test_teacher_intervention_requires_student_logprobs_on_cpu():
    class _DummyWorker:
        _teacher_routing_key = AgentLoopWorker._teacher_routing_key
        _maybe_repair_with_teacher = AgentLoopWorker._maybe_repair_with_teacher
        teacher_key = "data_source"

        def __init__(self):
            self.rollout_config = OmegaConf.create({"response_length": 8})
            self.teacher_intervention_enabled = True
            self.teacher_intervention_config = DistillationInterventionConfig(
                enabled=True,
                student_segment_length=2,
                teacher_segment_length=2,
                gap_metric="abs",
                gap_threshold=1.0,
                max_interventions_per_sample=1,
                log_examples_per_worker=0,
            )
            self.teacher_server_manager = _FakeInterventionTeacherManager()
            self.llm_client = _FakeRepairStudentClient()
            self._teacher_intervention_logged_samples = 0

    output = AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[1, 2, 3, 4],
        response_mask=[1, 1, 1, 1],
        response_logprobs=None,
        metrics=AgentLoopMetrics(),
        extra_fields={"rollout_request_id": "missing-logprobs"},
    )

    with pytest.raises(RuntimeError, match="student rollout did not return response_logprobs"):
        await AgentLoopWorker._maybe_repair_with_teacher(
            _DummyWorker(),
            output,
            validate=False,
            sampling_params={"temperature": 0.8},
            sample_kwargs={"data_source": "openai/gsm8k"},
        )


def test_agent_loop_postprocess_rejects_mixed_optional_response_logprobs_on_cpu():
    metrics = AgentLoopMetrics()
    internal_a = _to_internal(
        output_prompt_ids=[101, 102],
        output_response_ids=[1, 2],
        output_response_mask=[1, 1],
        metrics=metrics,
        extra_fields={},
        num_turns=2,
        prompt_len=4,
        response_len=4,
    )
    internal_b = _to_internal(
        output_prompt_ids=[101, 102],
        output_response_ids=[3, 4],
        output_response_mask=[1, 1],
        metrics=metrics,
        extra_fields={},
        num_turns=2,
        prompt_len=4,
        response_len=4,
    )
    internal_a.response_logprobs = torch.zeros((1, 4), dtype=torch.float32)

    dummy_worker = type(
        "_DummyWorker",
        (),
        {"reward_loop_worker_handles": None, "distillation_enabled": False},
    )()

    with pytest.raises(RuntimeError, match="Inconsistent optional field 'response_logprobs'"):
        AgentLoopWorker._postprocess(dummy_worker, inputs=[internal_a, internal_b])
