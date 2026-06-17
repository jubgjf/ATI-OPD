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
    _InternalAgentLoopOutput,
)
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.experimental.teacher_loop.teacher_manager import AsyncTeacherLLMServerManager, TeacherLogprobScoringResult
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.workers.config import DistillationInterventionConfig
from verl.workers.rollout.replica import TokenOutput


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
async def test_teacher_intervention_train_teacher_segments_drops_rollout_logprobs_batch_wide_on_cpu():
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
        torch.tensor([1, 1, 1, 1, 1, 1, 0, 0], dtype=torch.long),
    )

    merged = AgentLoopWorker._postprocess(worker, inputs=[internal_intervened, internal_untouched])
    assert "rollout_log_probs" not in merged.batch.keys()


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
