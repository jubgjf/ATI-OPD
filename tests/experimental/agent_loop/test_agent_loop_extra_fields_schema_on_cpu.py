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

import asyncio
import logging
import warnings
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopManager,
    AgentLoopMetrics,
    AgentLoopOutput,
    AgentLoopWorker,
    DictConfigWrap,
    _apply_generation_max_tokens,
    _compute_distribution_intervention_gap,
    _compute_intervention_gap,
    _evaluate_future_gap_confirmation,
    _GlobalTeacherNLLCalibration,
    _InternalAgentLoopOutput,
    _is_moe_like_hf_config,
    _select_intervention_segment,
    _TeacherNLLCalibrationSnapshot,
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


def test_future_gap_confirmation_uses_strict_threshold_and_reduction_boundaries_on_cpu():
    threshold_config = DistillationInterventionConfig(
        enabled=True,
        max_interventions_per_sample=1,
        confirmation_mode="future_gap",
        future_gap_window_tokens=2,
        future_gap_min_valid_tokens=2,
        future_gap_max_threshold=0.1,
    )
    threshold_proposal = {}
    assert not _evaluate_future_gap_confirmation(
        proposal=threshold_proposal,
        baseline_values=[0.2, 0.2],
        repaired_values=[0.1, 0.1],
        intervention_config=threshold_config,
    )
    assert threshold_proposal["reason"] == "future_gap_after_threshold"

    reduction_config = DistillationInterventionConfig(
        enabled=True,
        max_interventions_per_sample=1,
        confirmation_mode="future_gap",
        future_gap_window_tokens=2,
        future_gap_min_valid_tokens=2,
        future_gap_max_threshold=1.0,
        future_gap_min_reduction=0.25,
    )
    reduction_proposal = {}
    assert not _evaluate_future_gap_confirmation(
        proposal=reduction_proposal,
        baseline_values=[0.5, 0.5],
        repaired_values=[0.25, 0.25],
        intervention_config=reduction_config,
    )
    assert reduction_proposal["reason"] == "future_gap_insufficient_reduction"


def test_future_gap_confirmation_rejects_tracked_nonfinite_tokens_even_with_enough_finite_tokens_on_cpu():
    config = DistillationInterventionConfig(
        enabled=True,
        max_interventions_per_sample=1,
        confirmation_mode="future_gap",
        future_gap_window_tokens=2,
        future_gap_min_valid_tokens=2,
        future_gap_max_threshold=1.0,
    )
    proposal = {
        "future_gap_baseline_invalid_counts": {"teacher_nonfinite": 1},
        "future_gap_repaired_invalid_counts": {"value_nonfinite": 0},
    }

    assert not _evaluate_future_gap_confirmation(
        proposal=proposal,
        baseline_values=[0.5, 0.5],
        repaired_values=[0.1, 0.1],
        intervention_config=config,
    )
    assert proposal["future_gap_pair_available"] is False
    assert proposal["future_gap_nonfinite_count"] == 1
    assert proposal["reason"] == "future_gap_nonfinite_pair"


def test_rethinking_only_stu_defers_student_support_scoring_to_exact_teacher_pass_on_cpu():
    worker = object.__new__(AgentLoopWorker)
    worker.config = SimpleNamespace(algorithm=SimpleNamespace(adv_estimator="token_reward_direct"))
    worker.distillation_config = SimpleNamespace(
        distillation_loss=SimpleNamespace(
            topk=16,
            teacher_topk=128,
            top_k_strategy="only_stu_rethinking_style",
            loss_settings=SimpleNamespace(use_topk=False),
        ),
        diagnostics=SimpleNamespace(
            collect_topk_overlap=False,
            collect_rethinking_opd_metrics=False,
        ),
    )

    assert AgentLoopWorker._teacher_prompt_logprobs(worker) == 0

    # The asymmetric approximation expands only the ordinary teacher support.
    worker.distillation_config.distillation_loss.top_k_strategy = "only_stu_teacher_topk"
    assert AgentLoopWorker._teacher_prompt_logprobs(worker) == 128

    # The legacy strategy still requests teacher Top-K during rollout, which
    # protects the exact behavior of existing only_stu experiments.
    worker.distillation_config.distillation_loss.top_k_strategy = "only_stu"
    assert AgentLoopWorker._teacher_prompt_logprobs(worker) == 16


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


class _NonfiniteRepairStudentClient(_FakeRepairStudentClient):
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
        return TokenOutput(token_ids=[40, 41], log_probs=[float("-inf"), float("nan")])


class _CandidateSelectionStudentClient(_FakeRepairStudentClient):
    def __init__(self, student_nll_by_token: Optional[dict[int, float]] = None):
        super().__init__()
        self.student_nll_by_token = student_nll_by_token or {}

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
        del image_data, video_data, audio_data, mm_processor_kwargs
        self.calls.append(
            {"request_id": request_id, "prompt_ids": list(prompt_ids), "sampling_params": sampling_params}
        )
        width = max(int(sampling_params["prompt_logprobs"]), 1)
        ids = torch.zeros((len(prompt_ids), width), dtype=torch.int32)
        logprobs = torch.full((len(prompt_ids), width), -0.1, dtype=torch.float32)
        sampled_ids = torch.zeros((len(prompt_ids), 1), dtype=torch.int32)
        sampled_logprobs = torch.full((len(prompt_ids), 1), -0.1, dtype=torch.float32)
        for row in range(max(len(prompt_ids) - 1, 0)):
            target_id = int(prompt_ids[row + 1])
            sampled_ids[row, 0] = target_id
            ids[row, 0] = target_id
            sampled_logprobs[row, 0] = -float(self.student_nll_by_token.get(target_id, 0.1))
            logprobs[row, 0] = sampled_logprobs[row, 0]
        return TokenOutput(
            token_ids=[0],
            extra_fields={
                "prompt_ids": ids.tolist(),
                "prompt_logprobs": logprobs.tolist(),
                "sampled_prompt_ids": sampled_ids.tolist(),
                "sampled_prompt_logprobs": sampled_logprobs.tolist(),
            },
        )


class _CandidateSelectionTeacherManager:
    def __init__(
        self,
        candidates: list[list[int]],
        *,
        teacher_nll_by_token: Optional[dict[int, float]] = None,
        high_gap_start: int = 0,
        high_gap_width: int = 2,
    ):
        self.candidates = candidates
        self.teacher_nll_by_token = teacher_nll_by_token or {}
        self.high_gap_start = high_gap_start
        self.high_gap_width = high_gap_width
        self.generated_prefixes: list[list[int]] = []
        self.generation_request_ids: list[str] = []
        self.score_request_ids: list[str] = []

    async def generate_teacher_segment_single(
        self,
        prefix_ids: list[int],
        max_tokens: int,
        multi_modal_data: Optional[dict[str, Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        routing_key: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> TeacherGenerationResult:
        del multi_modal_data, mm_processor_kwargs, routing_key
        request_id = request_id or ""
        candidate_index = int(request_id.rsplit(":", 1)[-1])
        self.generated_prefixes.append(list(prefix_ids))
        self.generation_request_ids.append(request_id)
        return TeacherGenerationResult(
            token_ids=list(self.candidates[candidate_index][:max_tokens]),
            finish_reason="length",
        )

    async def compute_teacher_logprobs_single(
        self,
        sequence_ids: list[int],
        multi_modal_data: Optional[dict[str, Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        routing_key: Optional[str] = None,
        request_id: Optional[str] = None,
        prompt_logprobs: Optional[int] = None,
    ) -> TeacherLogprobScoringResult:
        del multi_modal_data, mm_processor_kwargs, routing_key
        request_id = request_id or ""
        self.score_request_ids.append(request_id)
        width = max(int(prompt_logprobs or 0), 1)
        ids = torch.zeros((len(sequence_ids), width), dtype=torch.int32)
        logprobs = torch.full((len(sequence_ids), width), -0.1, dtype=torch.float32)
        sampled_ids = torch.zeros((len(sequence_ids), 1), dtype=torch.int32)
        sampled_logprobs = torch.full((len(sequence_ids), 1), -0.1, dtype=torch.float32)
        for row in range(max(len(sequence_ids) - 1, 0)):
            target_id = int(sequence_ids[row + 1])
            sampled_ids[row, 0] = target_id
            ids[row, 0] = target_id
            sampled_logprobs[row, 0] = -float(self.teacher_nll_by_token.get(target_id, 0.1))
            logprobs[row, 0] = sampled_logprobs[row, 0]
        if ":teacher-score:" in request_id:
            response_start = 1
            for response_index in range(
                self.high_gap_start,
                min(self.high_gap_start + self.high_gap_width, len(sequence_ids) - 2),
            ):
                sampled_logprobs[response_start + response_index, 0] = -5.0
                logprobs[response_start + response_index, 0] = -5.0
        return TeacherLogprobScoringResult(
            ids=ids,
            logprobs=logprobs,
            sampled_ids=sampled_ids,
            sampled_logprobs=sampled_logprobs,
        )


class _ConcurrencyTrackingCandidateTeacherManager(_CandidateSelectionTeacherManager):
    def __init__(self, candidates: list[list[int]]):
        super().__init__(candidates)
        self.active_candidate_rpcs = 0
        self.max_active_candidate_rpcs = 0

    def _rpc_started(self) -> None:
        self.active_candidate_rpcs += 1
        self.max_active_candidate_rpcs = max(
            self.max_active_candidate_rpcs,
            self.active_candidate_rpcs,
        )

    async def generate_teacher_segment_single(self, *args: Any, **kwargs: Any) -> TeacherGenerationResult:
        self._rpc_started()
        try:
            await asyncio.sleep(0.01)
            return await super().generate_teacher_segment_single(*args, **kwargs)
        finally:
            self.active_candidate_rpcs -= 1

    async def compute_teacher_logprobs_single(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> TeacherLogprobScoringResult:
        self._rpc_started()
        try:
            await asyncio.sleep(0.01)
            return await super().compute_teacher_logprobs_single(*args, **kwargs)
        finally:
            self.active_candidate_rpcs -= 1


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
    _compute_multi_modal_inputs = AgentLoopWorker._compute_multi_modal_inputs
    _compute_position_ids = AgentLoopWorker._compute_position_ids
    _get_mm_processor_kwargs = AgentLoopWorker._get_mm_processor_kwargs
    _compute_score = AgentLoopWorker._compute_score
    _score_rejected_future_gap_candidate = AgentLoopWorker._score_rejected_future_gap_candidate
    _teacher_routing_key = AgentLoopWorker._teacher_routing_key
    _score_sequence_for_intervention = AgentLoopWorker._score_sequence_for_intervention
    _score_student_sequence_for_intervention = AgentLoopWorker._score_student_sequence_for_intervention
    _current_teacher_nll_threshold = AgentLoopWorker._current_teacher_nll_threshold
    _teacher_candidate_rpc_limiter = AgentLoopWorker._teacher_candidate_rpc_limiter
    _generate_and_select_teacher_candidate = AgentLoopWorker._generate_and_select_teacher_candidate
    _generate_student_suffix = AgentLoopWorker._generate_student_suffix
    _maybe_repair_with_teacher = AgentLoopWorker._maybe_repair_with_teacher
    _compute_teacher_logprobs = AgentLoopWorker._compute_teacher_logprobs
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
        skip_correct: bool = False,
        confirmation_mode: str = "none",
        future_gap_window_tokens: int = 64,
        future_gap_min_valid_tokens: int = 32,
        future_gap_max_threshold: Optional[float] = None,
        future_gap_min_reduction: float = 0.0,
        train_on_teacher_segments: bool = False,
        teacher_segment_loss_mode: str = "opd",
        min_trainable_tokens_after_repair: int = 1,
        gap_metric: str = "abs",
        gap_threshold: float = 1.0,
        gap_distribution_topk: int = 16,
        teacher_rejection_token_cap: float = 10.0,
        teacher_rejection_persistence_enabled: bool = False,
        teacher_rejection_persistence_window_tokens: int = 64,
        teacher_rejection_persistence_threshold: Optional[float] = None,
        teacher_rejection_persistence_min_valid_tokens: int = 32,
        topk: Optional[int] = None,
        thinking_start_text: str = "<think>",
        thinking_end_text: str = "</think>",
        log_examples_per_worker: int = 0,
        diagnostics_topk: Optional[int] = None,
        outcome_aware_candidate_selection_enabled: bool = False,
        teacher_candidate_count: int = 4,
        teacher_candidate_max_concurrency_per_worker: int = 4,
        teacher_nll_calibration_quantile: float = 0.75,
        teacher_nll_calibration_window: int = 16,
        teacher_nll_calibration_min_samples: int = 2,
        teacher_nll_initial_threshold: Optional[float] = None,
        teacher_nll_healthy_probe_probability: float = 0.0,
        repair_for_evaluation: bool = False,
        collect_intervention_outcome_transitions: bool = False,
        collect_intervention_boundary_metrics: bool = False,
        outcome_case_samples_per_transition: int = 8,
    ):
        self.tokenizer = _FakeTokenizer()
        self.rollout_config = OmegaConf.create({"prompt_length": 4, "response_length": response_length})
        self.processor = None
        self.mm_processor_kwargs = {}
        self.reward_loop_worker_handles = None
        self.teacher_intervention_enabled = True
        self.teacher_intervention_config = DistillationInterventionConfig(
            enabled=True,
            student_segment_length=student_segment_length,
            teacher_segment_length=teacher_segment_length,
            gap_metric=gap_metric,
            gap_distribution_topk=gap_distribution_topk,
            teacher_rejection_token_cap=teacher_rejection_token_cap,
            teacher_rejection_persistence_enabled=teacher_rejection_persistence_enabled,
            teacher_rejection_persistence_window_tokens=teacher_rejection_persistence_window_tokens,
            teacher_rejection_persistence_threshold=teacher_rejection_persistence_threshold,
            teacher_rejection_persistence_min_valid_tokens=teacher_rejection_persistence_min_valid_tokens,
            gap_threshold=gap_threshold,
            max_interventions_per_sample=max_interventions_per_sample,
            skip_correct=skip_correct,
            confirmation_mode=confirmation_mode,
            future_gap_window_tokens=future_gap_window_tokens,
            future_gap_min_valid_tokens=future_gap_min_valid_tokens,
            future_gap_max_threshold=future_gap_max_threshold,
            future_gap_min_reduction=future_gap_min_reduction,
            train_on_teacher_segments=train_on_teacher_segments,
            teacher_segment_loss_mode=teacher_segment_loss_mode,
            min_trainable_tokens_after_repair=min_trainable_tokens_after_repair,
            thinking_start_text=thinking_start_text,
            thinking_end_text=thinking_end_text,
            outcome_aware_candidate_selection_enabled=outcome_aware_candidate_selection_enabled,
            teacher_candidate_count=teacher_candidate_count,
            teacher_candidate_max_concurrency_per_worker=teacher_candidate_max_concurrency_per_worker,
            teacher_nll_calibration_quantile=teacher_nll_calibration_quantile,
            teacher_nll_calibration_window=teacher_nll_calibration_window,
            teacher_nll_calibration_min_samples=teacher_nll_calibration_min_samples,
            teacher_nll_initial_threshold=teacher_nll_initial_threshold,
            teacher_nll_healthy_probe_probability=teacher_nll_healthy_probe_probability,
            repair_for_evaluation=repair_for_evaluation,
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
                enabled=collect_intervention_outcome_transitions or collect_intervention_boundary_metrics,
                collect_intervention_metrics=True,
                collect_intervention_boundary_metrics=collect_intervention_boundary_metrics,
                collect_intervention_outcome_transitions=collect_intervention_outcome_transitions,
                outcome_case_samples_per_transition=outcome_case_samples_per_transition,
                log_interval=1,
                collect_topk_overlap=diagnostics_topk is not None,
                topk=diagnostics_topk or 64,
            ),
        )
        self._teacher_intervention_logged_samples = 0
        self._teacher_intervention_logged_step = None
        self._teacher_nll_calibration_snapshot = _TeacherNLLCalibrationSnapshot(
            threshold=teacher_nll_initial_threshold,
            source="unavailable" if teacher_nll_initial_threshold is None else "initial",
            count=0,
            version=0,
        )
        self._teacher_nll_pending_updates = []
        self._teacher_candidate_rpc_semaphore = None


def _outcome_gated_intervention_output(*, request_id: str, raw_correct: bool) -> AgentLoopOutput:
    return AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[1, 2, 3, 4, 5, 6],
        response_mask=[1, 1, 1, 1, 1, 1],
        response_logprobs=[-0.1] * 6,
        reward_score=1.0 if raw_correct else 0.0,
        metrics=AgentLoopMetrics(),
        extra_fields={
            "rollout_request_id": request_id,
            "teacher_intervention_outcome": {
                "enabled": True,
                "raw_correct": raw_correct,
                "trigger_detected": False,
                "repair_executed": False,
            },
        },
    )


class _FakeRemoteMethod:
    def __init__(self, function):
        self.function = function

    def remote(self, *args):
        async def invoke():
            return self.function(*args)

        return invoke()


class _FakeCalibrationWorkerHandle:
    def __init__(self, updates: list[float]):
        self.updates = list(updates)
        self.snapshot: Optional[_TeacherNLLCalibrationSnapshot] = None
        self.drain_teacher_nll_calibration_updates = _FakeRemoteMethod(self._drain)
        self.set_teacher_nll_calibration_state = _FakeRemoteMethod(self._set)

    def _drain(self) -> list[float]:
        updates = self.updates
        self.updates = []
        return updates

    def _set(self, snapshot: _TeacherNLLCalibrationSnapshot) -> None:
        self.snapshot = snapshot


def test_agent_loop_manager_builds_global_teacher_nll_calibration_from_plain_config_on_cpu():
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {"agent": {"num_workers": 2}},
                "model": {},
            },
            "distillation": {
                "enabled": True,
                "intervention": {
                    "enabled": True,
                    "outcome_aware_candidate_selection_enabled": True,
                    "teacher_nll_calibration_min_samples": 4,
                },
            },
        }
    )

    manager = AgentLoopManager(config, llm_client=None)

    assert manager._global_teacher_nll_calibration is not None
    assert manager._global_teacher_nll_calibration.min_samples == 4
    assert manager._global_teacher_nll_calibration.snapshot().source == "unavailable"


@pytest.mark.asyncio
async def test_agent_loop_manager_globally_aggregates_and_broadcasts_teacher_nll_on_cpu():
    config = DistillationInterventionConfig(
        enabled=True,
        outcome_aware_candidate_selection_enabled=True,
        teacher_nll_calibration_quantile=0.75,
        teacher_nll_calibration_window=8,
        teacher_nll_calibration_min_samples=4,
    )
    manager = object.__new__(AgentLoopManager)
    manager._global_teacher_nll_calibration = _GlobalTeacherNLLCalibration(config)
    workers = [
        _FakeCalibrationWorkerHandle([0.1, 0.3]),
        _FakeCalibrationWorkerHandle([0.2, 0.4]),
    ]
    manager.agent_loop_workers = workers

    await AgentLoopManager._refresh_global_teacher_nll_calibration(manager)

    expected_threshold = float(np.quantile(np.asarray([0.1, 0.3, 0.2, 0.4]), 0.75))
    assert manager._global_teacher_nll_calibration.history == pytest.approx([0.1, 0.3, 0.2, 0.4])
    assert all(worker.snapshot is not None for worker in workers)
    assert all(worker.snapshot.threshold == pytest.approx(expected_threshold) for worker in workers)
    assert all(worker.snapshot.source == "rolling_healthy_quantile" for worker in workers)
    assert all(worker.snapshot.count == 4 for worker in workers)
    assert all(worker.snapshot.version == 1 for worker in workers)
    assert all(worker.updates == [] for worker in workers)

    workers[0].updates = [0.5]
    await AgentLoopManager._refresh_global_teacher_nll_calibration(manager)

    assert all(worker.snapshot.count == 5 for worker in workers)
    assert all(worker.snapshot.version == 2 for worker in workers)


def test_agent_loop_manager_caps_full_text_outcome_cases_per_transition_by_stable_rank_on_cpu():
    manager = object.__new__(AgentLoopManager)
    manager.config = OmegaConf.create({"distillation": {"diagnostics": {"outcome_case_samples_per_transition": 1}}})
    output = DataProto.from_single_dict(
        {
            "dummy": torch.arange(5).unsqueeze(-1),
            "teacher_intervention_outcome_case": np.array(
                [
                    {"transition": "pre_wrong_post_correct", "case_rank": 0.8},
                    {"transition": "pre_wrong_post_correct", "case_rank": 0.2},
                    {"transition": "pre_wrong_post_correct", "case_rank": 0.5},
                    {"transition": "pre_correct_post_wrong", "case_rank": 0.7},
                    {"transition": "pre_correct_post_wrong", "case_rank": 0.1},
                ],
                dtype=object,
            ),
        }
    )

    AgentLoopManager._cap_intervention_outcome_cases(manager, output)

    cases = output.non_tensor_batch["teacher_intervention_outcome_case"].tolist()
    assert [index for index, case in enumerate(cases) if case is not None] == [1, 4]


def test_agent_loop_manager_globally_caps_boundary_cases_by_stable_rank_on_cpu():
    manager = object.__new__(AgentLoopManager)
    manager.config = OmegaConf.create(
        {"distillation": {"diagnostics": {"intervention_boundary_text_samples_per_step": 2}}}
    )
    output = DataProto.from_single_dict(
        {
            "dummy": torch.arange(3).unsqueeze(-1),
            "teacher_intervention_boundary_cases": np.array(
                [
                    [{"case_rank": 0.8}, {"case_rank": 0.2}],
                    [{"case_rank": 0.1}],
                    [{"case_rank": 0.5}],
                ],
                dtype=object,
            ),
        }
    )

    AgentLoopManager._cap_intervention_boundary_cases(manager, output)

    cases = output.non_tensor_batch["teacher_intervention_boundary_cases"].tolist()
    retained_ranks = [case["case_rank"] for sample_cases in cases if sample_cases for case in sample_cases]
    assert retained_ranks == [0.2, 0.1]


def test_agent_loop_worker_decodes_only_local_boundary_candidates_and_drops_private_token_ids_on_cpu():
    worker = object.__new__(AgentLoopWorker)
    worker.tokenizer = _FakeTokenizer()
    worker.distillation_config = SimpleNamespace(
        diagnostics=SimpleNamespace(
            intervention_boundary_text_samples_per_step=1,
            intervention_boundary_text_context_tokens=2,
            outcome_case_samples_per_transition=0,
        )
    )
    inputs = []
    for rank, start in ((0.8, 1), (0.2, 2)):
        inputs.append(
            _to_internal(
                output_prompt_ids=[101, 102],
                output_response_ids=[1, 2, 30, 31, 40, 41],
                output_response_mask=[1, 1, 0, 0, 1, 1],
                metrics=AgentLoopMetrics(),
                extra_fields={
                    "teacher_interventions": [
                        {
                            "attempt": 0,
                            "start": start,
                            "teacher_tokens": 2,
                            "boundary_case_rank": rank,
                            "boundary_final_span_available": True,
                            "_original_student_segment_ids": [7, 8, 9, 10, 11],
                            "_teacher_segment_ids": [30, 31],
                        }
                    ]
                },
                num_turns=1,
                prompt_len=4,
                response_len=8,
            )
        )

    AgentLoopWorker._prepare_local_intervention_boundary_cases(worker, inputs)
    AgentLoopWorker._prepare_local_intervention_outcome_cases(worker, inputs)

    assert inputs[0].extra_fields["teacher_intervention_boundary_cases"] is None
    selected_cases = inputs[1].extra_fields["teacher_intervention_boundary_cases"]
    assert isinstance(selected_cases, list) and len(selected_cases) == 1
    assert selected_cases[0]["case_rank"] == 0.2
    assert selected_cases[0]["context_token_limit"] == 2
    for input_item in inputs:
        intervention = input_item.extra_fields["teacher_interventions"][0]
        assert "_original_student_segment_ids" not in intervention
        assert "_teacher_segment_ids" not in intervention


class _SequenceAwareRewardMethod:
    def __init__(self):
        self.responses: list[list[int]] = []

    async def remote(self, data: DataProto) -> dict[str, Any]:
        response_ids = data.batch["responses"][0].tolist()
        self.responses.append(response_ids)
        correct = float(31 in response_ids)
        return {
            "reward_score": correct,
            "reward_extra_info": {"acc": correct},
        }


class _SequenceAwareRewardHandle:
    def __init__(self):
        self.compute_score = _SequenceAwareRewardMethod()


@pytest.mark.asyncio
async def test_outcome_aware_postprocess_scores_raw_then_repaired_answer_on_cpu():
    teacher_manager = _CandidateSelectionTeacherManager(
        [[30, 30], [31, 31], [32, 32], [33, 33]],
        teacher_nll_by_token={30: 0.1, 31: 0.2, 32: 0.3, 33: 0.8},
    )
    worker = _DirectInterventionWorker(
        teacher_manager=teacher_manager,
        response_length=6,
        outcome_aware_candidate_selection_enabled=True,
    )
    AgentLoopWorker.set_teacher_nll_calibration_state(
        worker,
        _TeacherNLLCalibrationSnapshot(
            threshold=0.5,
            source="rolling_healthy_quantile",
            count=16,
            version=3,
        ),
    )
    worker.llm_client = _CandidateSelectionStudentClient(student_nll_by_token={30: 0.4, 31: 0.1, 32: 0.2, 33: 0.05})
    reward_handle = _SequenceAwareRewardHandle()
    worker.reward_loop_worker_handles = [reward_handle]
    output = AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[1, 2, 3, 4, 5, 6],
        response_mask=[1, 1, 1, 1, 1, 1],
        response_logprobs=[-0.1] * 6,
        metrics=AgentLoopMetrics(),
        extra_fields={"rollout_request_id": "raw-final-reward"},
    )

    internal = await AgentLoopWorker._agent_loop_postprocess(
        worker,
        output,
        validate=False,
        sampling_params={"temperature": 0.8, "logprobs": True},
        raw_prompt=[{"role": "user", "content": "hi"}],
        data_source="openai/gsm8k",
    )

    assert reward_handle.compute_score.responses == [[1, 2, 3, 4, 5, 6], [31, 31, 40, 41]]
    assert internal.reward_score == 1.0
    outcome = internal.extra_fields["teacher_intervention_outcome"]
    assert outcome["raw_correct"] is False
    assert outcome["final_correct"] is True
    assert outcome["repair_effective"] is True
    assert outcome["repair_ineffective"] is False
    assert "transition_observation_event" not in outcome
    assert "raw_reward_extra_info" not in outcome
    assert "transition" not in outcome
    assert "_teacher_segment_ids" not in internal.extra_fields["teacher_interventions"][0]
    assert internal.extra_fields["teacher_interventions"][0]["sample_outcome_summary"] is True


@pytest.mark.asyncio
async def test_intervention_outcome_diagnostics_scores_raw_and_final_without_outcome_gate_on_cpu():
    teacher_manager = _FixedGapTeacherManager(
        TeacherGenerationResult(token_ids=[31, 31], finish_reason="length"),
    )
    worker = _DirectInterventionWorker(
        teacher_manager=teacher_manager,
        response_length=6,
        collect_intervention_outcome_transitions=True,
    )
    # Collection covers every training sample; log_interval only controls
    # when the aggregate diagnostics are emitted.
    worker.distillation_config.diagnostics.log_interval = 100
    reward_handle = _SequenceAwareRewardHandle()
    worker.reward_loop_worker_handles = [reward_handle]
    output = AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[1, 2, 3, 4, 5, 6],
        response_mask=[1, 1, 1, 1, 1, 1],
        response_logprobs=[-0.1] * 6,
        metrics=AgentLoopMetrics(),
        extra_fields={"rollout_request_id": "diagnostic-raw-final"},
    )

    internal = await AgentLoopWorker._agent_loop_postprocess(
        worker,
        output,
        validate=False,
        sampling_params={"temperature": 0.8, "logprobs": True},
        raw_prompt=[{"role": "user", "content": "hi"}],
        data_source="openai/gsm8k",
        uid="uid-diagnostic",
        global_steps=4,
    )

    assert reward_handle.compute_score.responses == [[1, 2, 3, 4, 5, 6], [31, 31, 40, 41]]
    assert internal.reward_score == 1.0
    outcome = internal.extra_fields["teacher_intervention_outcome"]
    assert outcome["raw_reward_score"] == 0.0
    assert outcome["final_reward_score"] == 1.0
    assert outcome["transition"] == "pre_wrong_post_correct"
    assert outcome["intervention_count"] == 1
    assert outcome["repair_executed"] is True
    training_batch = AgentLoopWorker._postprocess(worker, [internal], validate=False)
    assert training_batch.batch["rm_scores"].sum().item() == 1.0
    assert "raw_reward_score" not in training_batch.batch
    assert training_batch.non_tensor_batch["teacher_intervention_outcome"][0]["raw_reward_score"] == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw_answer_token", "teacher_answer_token", "expected_transition"),
    [
        (31, 31, "pre_correct_post_correct"),
        (31, 30, "pre_correct_post_wrong"),
        (1, 31, "pre_wrong_post_correct"),
        (1, 30, "pre_wrong_post_wrong"),
    ],
)
async def test_intervention_outcome_diagnostics_classifies_all_four_transitions_on_cpu(
    raw_answer_token,
    teacher_answer_token,
    expected_transition,
):
    teacher_manager = _FixedGapTeacherManager(
        TeacherGenerationResult(token_ids=[teacher_answer_token, teacher_answer_token], finish_reason="length"),
    )
    worker = _DirectInterventionWorker(
        teacher_manager=teacher_manager,
        response_length=6,
        collect_intervention_outcome_transitions=True,
    )
    reward_handle = _SequenceAwareRewardHandle()
    worker.reward_loop_worker_handles = [reward_handle]
    output = AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[raw_answer_token, 2, 3, 4, 5, 6],
        response_mask=[1, 1, 1, 1, 1, 1],
        response_logprobs=[-0.1] * 6,
        metrics=AgentLoopMetrics(),
        extra_fields={"rollout_request_id": f"transition-{expected_transition}"},
    )

    internal = await AgentLoopWorker._agent_loop_postprocess(
        worker,
        output,
        validate=False,
        sampling_params={"temperature": 0.8, "logprobs": True},
        raw_prompt=[{"role": "user", "content": "hi"}],
        data_source="openai/gsm8k",
        global_steps=3,
    )

    assert len(reward_handle.compute_score.responses) == 2
    assert internal.extra_fields["teacher_intervention_outcome"]["transition"] == expected_transition


@pytest.mark.asyncio
async def test_intervention_outcome_diagnostics_reuses_raw_reward_when_no_repair_on_cpu():
    teacher_manager = _FixedGapTeacherManager(
        TeacherGenerationResult(token_ids=[31, 31], finish_reason="length"),
    )
    worker = _DirectInterventionWorker(
        teacher_manager=teacher_manager,
        response_length=6,
        gap_threshold=10.0,
        collect_intervention_outcome_transitions=True,
    )
    reward_handle = _SequenceAwareRewardHandle()
    worker.reward_loop_worker_handles = [reward_handle]
    output = AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[31, 2, 3, 4, 5, 6],
        response_mask=[1, 1, 1, 1, 1, 1],
        response_logprobs=[-0.1] * 6,
        metrics=AgentLoopMetrics(),
        extra_fields={"rollout_request_id": "diagnostic-no-repair"},
    )

    internal = await AgentLoopWorker._agent_loop_postprocess(
        worker,
        output,
        validate=False,
        sampling_params={"temperature": 0.8, "logprobs": True},
        raw_prompt=[{"role": "user", "content": "hi"}],
        data_source="openai/gsm8k",
        global_steps=4,
    )

    assert reward_handle.compute_score.responses == [[31, 2, 3, 4, 5, 6]]
    assert internal.reward_score == 1.0
    outcome = internal.extra_fields["teacher_intervention_outcome"]
    assert outcome["transition"] == "not_repaired"
    assert outcome["repair_executed"] is False
    assert outcome["raw_reward_score"] == outcome["final_reward_score"] == 1.0


@pytest.mark.asyncio
async def test_repair_evaluation_forks_one_raw_response_and_discards_detection_logprobs_on_cpu():
    teacher_manager = _FixedGapTeacherManager(
        TeacherGenerationResult(token_ids=[31, 31], finish_reason="length"),
    )
    worker = _DirectInterventionWorker(
        teacher_manager=teacher_manager,
        response_length=6,
        repair_for_evaluation=True,
    )
    reward_handle = _SequenceAwareRewardHandle()
    worker.reward_loop_worker_handles = [reward_handle]
    output = AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[1, 2, 3, 4, 5, 6],
        response_mask=[1, 1, 1, 1, 1, 1],
        response_logprobs=[-0.1] * 6,
        metrics=AgentLoopMetrics(),
        extra_fields={"rollout_request_id": "paired-evaluation"},
    )

    internal = await AgentLoopWorker._agent_loop_postprocess(
        worker,
        output,
        validate=True,
        repair_for_evaluation=True,
        sampling_params={"temperature": 0.8, "logprobs": True},
        raw_prompt=[{"role": "user", "content": "hi"}],
        data_source="openai/gsm8k",
        uid="paired-uid",
        global_steps=0,
    )

    assert reward_handle.compute_score.responses == [[1, 2, 3, 4, 5, 6], [31, 31, 40, 41]]
    assert internal.response_logprobs is None
    assert internal.teacher_segment_mask is None
    assert internal.teacher_ids is None
    assert internal.teacher_logprobs is None
    assert internal.teacher_sampled_ids is None
    assert internal.teacher_sampled_logprobs is None
    assert "teacher_logprob_nonfinite_positions" not in internal.extra_fields
    outcome = internal.extra_fields["teacher_intervention_outcome"]
    assert outcome["evaluation_pair"] is True
    assert outcome["raw_response"] == "<decoded>"
    assert outcome["transition"] == "pre_wrong_post_correct"


@pytest.mark.asyncio
async def test_validation_without_repair_flag_preserves_legacy_no_teacher_path_on_cpu():
    teacher_manager = _FixedGapTeacherManager(
        TeacherGenerationResult(token_ids=[31, 31], finish_reason="length"),
    )
    worker = _DirectInterventionWorker(
        teacher_manager=teacher_manager,
        response_length=6,
        repair_for_evaluation=True,
    )
    reward_handle = _SequenceAwareRewardHandle()
    worker.reward_loop_worker_handles = [reward_handle]
    output = AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[1, 2, 3, 4, 5, 6],
        response_mask=[1, 1, 1, 1, 1, 1],
        response_logprobs=[-0.1] * 6,
        metrics=AgentLoopMetrics(),
        extra_fields={"rollout_request_id": "legacy-validation"},
    )

    internal = await AgentLoopWorker._agent_loop_postprocess(
        worker,
        output,
        validate=True,
        repair_for_evaluation=False,
        sampling_params={"temperature": 0.8, "logprobs": True},
        raw_prompt=[{"role": "user", "content": "hi"}],
        data_source="openai/gsm8k",
    )

    assert reward_handle.compute_score.responses == [[1, 2, 3, 4, 5, 6]]
    assert teacher_manager.scored_sequences == []
    assert teacher_manager.generated_prefixes == []
    assert "teacher_intervention_outcome" not in internal.extra_fields
    assert "teacher_intervention_count" not in internal.extra_fields


@pytest.mark.asyncio
async def test_outcome_aware_repair_evaluation_reads_but_does_not_update_calibration_on_cpu():
    teacher_manager = _CandidateSelectionTeacherManager(
        [[30, 30], [31, 31], [32, 32], [33, 33]],
        teacher_nll_by_token={30: 0.1, 31: 0.2, 32: 0.3, 33: 0.8},
    )
    worker = _DirectInterventionWorker(
        teacher_manager=teacher_manager,
        response_length=6,
        outcome_aware_candidate_selection_enabled=True,
        repair_for_evaluation=True,
    )
    AgentLoopWorker.set_teacher_nll_calibration_state(
        worker,
        _TeacherNLLCalibrationSnapshot(
            threshold=0.5,
            source="rolling_healthy_quantile",
            count=16,
            version=3,
        ),
    )
    worker.llm_client = _CandidateSelectionStudentClient(student_nll_by_token={30: 0.4, 31: 0.1, 32: 0.2, 33: 0.05})
    reward_handle = _SequenceAwareRewardHandle()
    worker.reward_loop_worker_handles = [reward_handle]
    output = AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[1, 2, 3, 4, 5, 6],
        response_mask=[1, 1, 1, 1, 1, 1],
        response_logprobs=[-0.1] * 6,
        metrics=AgentLoopMetrics(),
        extra_fields={"rollout_request_id": "oracle-evaluation"},
    )

    internal = await AgentLoopWorker._agent_loop_postprocess(
        worker,
        output,
        validate=True,
        repair_for_evaluation=True,
        sampling_params={"temperature": 0.8, "logprobs": True},
        raw_prompt=[{"role": "user", "content": "hi"}],
        data_source="openai/gsm8k",
        global_steps=0,
    )

    assert worker._teacher_nll_pending_updates == []
    outcome = internal.extra_fields["teacher_intervention_outcome"]
    assert outcome["oracle_gate_used"] is True
    assert outcome["transition"] == "pre_wrong_post_correct"
    intervention = internal.extra_fields["teacher_interventions"][0]
    assert intervention["teacher_nll_calibration_read_only"] is True
    assert intervention["teacher_nll_calibration_count_before"] == 16
    assert intervention["teacher_nll_calibration_version"] == 3


@pytest.mark.asyncio
async def test_outcome_aware_candidate_selection_filters_then_uses_rank_sum_on_cpu():
    teacher_manager = _CandidateSelectionTeacherManager(
        [[30, 30], [31, 31], [32, 32], [33, 33]],
        teacher_nll_by_token={30: 0.1, 31: 0.2, 32: 0.3, 33: 0.8},
    )
    worker = _DirectInterventionWorker(
        teacher_manager=teacher_manager,
        response_length=6,
        outcome_aware_candidate_selection_enabled=True,
    )
    AgentLoopWorker.set_teacher_nll_calibration_state(
        worker,
        _TeacherNLLCalibrationSnapshot(
            threshold=0.5,
            source="rolling_healthy_quantile",
            count=16,
            version=3,
        ),
    )
    worker.llm_client = _CandidateSelectionStudentClient(student_nll_by_token={30: 0.4, 31: 0.1, 32: 0.2, 33: 0.05})
    output = _outcome_gated_intervention_output(request_id="candidate-rank", raw_correct=False)

    await AgentLoopWorker._maybe_repair_with_teacher(
        worker,
        output,
        validate=False,
        sampling_params={"temperature": 0.8, "logprobs": True},
        sample_kwargs={"data_source": "openai/gsm8k"},
    )

    assert output.response_ids == [31, 31, 40, 41]
    assert len(teacher_manager.generation_request_ids) == 4
    student_candidate_calls = [
        call for call in worker.llm_client.calls if ":student-candidate-score:" in call["request_id"]
    ]
    assert len(student_candidate_calls) == 3
    intervention = output.extra_fields["teacher_interventions"][0]
    assert intervention["candidate_count_requested"] == 4
    assert intervention["candidate_count_teacher_nll_valid"] == 4
    assert intervention["candidate_count_threshold_eligible"] == 3
    assert intervention["candidate_count_rankable"] == 3
    assert intervention["teacher_nll_calibration_scope"] == "global"
    assert intervention["teacher_nll_calibration_count_before"] == 16
    assert intervention["teacher_nll_calibration_version"] == 3
    assert intervention["selected_candidate_index"] == 1
    assert intervention["selected_teacher_rank"] == 2
    assert intervention["selected_student_rank"] == 1
    assert intervention["selected_rank_sum"] == 3
    assert output.reward_score is None
    assert output.extra_fields["teacher_intervention_outcome"]["reward_recompute_required"] is True


@pytest.mark.asyncio
async def test_outcome_aware_candidate_rpcs_are_bounded_across_concurrent_samples_on_cpu():
    teacher_manager = _ConcurrencyTrackingCandidateTeacherManager([[30], [31], [32], [33]])
    worker = _DirectInterventionWorker(
        teacher_manager=teacher_manager,
        outcome_aware_candidate_selection_enabled=True,
        teacher_candidate_count=4,
        teacher_candidate_max_concurrency_per_worker=2,
    )
    outputs = [
        _outcome_gated_intervention_output(request_id=f"bounded-candidates-{index}", raw_correct=True)
        for index in range(2)
    ]

    results = await asyncio.gather(
        *[
            AgentLoopWorker._generate_and_select_teacher_candidate(
                worker,
                prefix_ids=[101, 102],
                max_tokens=1,
                output=output,
                routing_key=None,
                base_request_id=f"bounded-candidates-{index}",
                attempt=0,
                calibration_only=True,
            )
            for index, output in enumerate(outputs)
        ]
    )

    assert teacher_manager.max_active_candidate_rpcs == 2
    assert all(result.metadata["candidate_rpc_max_concurrency_per_worker"] == 2 for result in results)
    assert all(result.metadata["candidate_count_teacher_nll_valid"] == 4 for result in results)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("gap_metric", "persistence_enabled"),
    [
        ("k3", False),
        ("teacher_rejection", False),
        ("teacher_rejection", True),
    ],
)
async def test_outcome_aware_selection_preserves_existing_trigger_modes_on_cpu(
    gap_metric: str,
    persistence_enabled: bool,
):
    teacher_manager = _CandidateSelectionTeacherManager(
        [[30], [31], [32], [33]],
        teacher_nll_by_token={30: 0.1, 31: 0.2, 32: 0.3, 33: 0.4},
        high_gap_width=4,
    )
    worker = _DirectInterventionWorker(
        teacher_manager=teacher_manager,
        response_length=6,
        gap_metric=gap_metric,
        outcome_aware_candidate_selection_enabled=True,
        teacher_nll_initial_threshold=1.0,
        teacher_rejection_persistence_enabled=persistence_enabled,
        teacher_rejection_persistence_window_tokens=2,
        teacher_rejection_persistence_threshold=1.0,
        teacher_rejection_persistence_min_valid_tokens=2,
    )
    worker.llm_client = _CandidateSelectionStudentClient(student_nll_by_token={30: 0.1, 31: 0.2, 32: 0.3, 33: 0.4})
    output = _outcome_gated_intervention_output(
        request_id=f"trigger-{gap_metric}-{persistence_enabled}",
        raw_correct=False,
    )

    await AgentLoopWorker._maybe_repair_with_teacher(
        worker,
        output,
        validate=False,
        sampling_params={"temperature": 0.8, "logprobs": True},
        sample_kwargs={"data_source": "openai/gsm8k"},
    )

    assert output.extra_fields["teacher_intervention_count"] == 1
    if persistence_enabled:
        assert output.extra_fields["teacher_interventions"][0]["teacher_rejection_persistence_tokens"] == 2


@pytest.mark.asyncio
async def test_outcome_aware_candidate_selection_waits_for_healthy_calibration_on_cpu():
    teacher_manager = _CandidateSelectionTeacherManager(
        [[30], [31], [32], [33]],
        teacher_nll_by_token={30: 0.1, 31: 0.2, 32: 0.3, 33: 0.4},
    )
    worker = _DirectInterventionWorker(
        teacher_manager=teacher_manager,
        response_length=6,
        outcome_aware_candidate_selection_enabled=True,
        teacher_nll_initial_threshold=None,
    )
    worker.llm_client = _CandidateSelectionStudentClient()
    output = _outcome_gated_intervention_output(request_id="calibration-not-ready", raw_correct=False)

    await AgentLoopWorker._maybe_repair_with_teacher(
        worker,
        output,
        validate=False,
        sampling_params={"temperature": 0.8, "logprobs": True},
        sample_kwargs={"data_source": "openai/gsm8k"},
    )

    assert output.response_ids == [1, 2, 3, 4, 5, 6]
    assert output.extra_fields["teacher_intervention_count"] == 0
    skipped = output.extra_fields["teacher_intervention_skipped"][0]
    assert skipped["reason"] == "teacher_nll_calibration_not_ready"
    assert skipped["teacher_nll_threshold"] is None
    assert skipped["teacher_nll_threshold_source"] == "unavailable"
    assert not any(":student-candidate-score:" in call["request_id"] for call in worker.llm_client.calls)


@pytest.mark.asyncio
async def test_outcome_aware_candidate_selection_skips_when_all_candidates_fail_nll_gate_on_cpu():
    teacher_manager = _CandidateSelectionTeacherManager(
        [[30], [31], [32], [33]],
        teacher_nll_by_token={30: 0.2, 31: 0.3, 32: 0.4, 33: 0.5},
    )
    worker = _DirectInterventionWorker(
        teacher_manager=teacher_manager,
        response_length=6,
        outcome_aware_candidate_selection_enabled=True,
        teacher_nll_initial_threshold=0.1,
    )
    worker.llm_client = _CandidateSelectionStudentClient()
    output = _outcome_gated_intervention_output(request_id="all-above-threshold", raw_correct=False)

    await AgentLoopWorker._maybe_repair_with_teacher(
        worker,
        output,
        validate=False,
        sampling_params={"temperature": 0.8, "logprobs": True},
        sample_kwargs={"data_source": "openai/gsm8k"},
    )

    assert output.extra_fields["teacher_intervention_count"] == 0
    skipped = output.extra_fields["teacher_intervention_skipped"][0]
    assert skipped["reason"] == "teacher_candidates_above_nll_threshold"
    assert skipped["candidate_count_threshold_eligible"] == 0
    assert not any(":student-candidate-score:" in call["request_id"] for call in worker.llm_client.calls)


@pytest.mark.asyncio
async def test_outcome_correct_sample_is_never_rewritten_and_can_shadow_calibrate_on_cpu():
    teacher_manager = _CandidateSelectionTeacherManager(
        [[30], [31], [32], [33]],
        teacher_nll_by_token={30: 0.4, 31: 0.2, 32: 0.3, 33: 0.5},
    )
    worker = _DirectInterventionWorker(
        teacher_manager=teacher_manager,
        response_length=6,
        outcome_aware_candidate_selection_enabled=True,
        teacher_nll_healthy_probe_probability=1.0,
    )
    worker.llm_client = _CandidateSelectionStudentClient()
    output = _outcome_gated_intervention_output(request_id="healthy-shadow", raw_correct=True)

    await AgentLoopWorker._maybe_repair_with_teacher(
        worker,
        output,
        validate=False,
        sampling_params={"temperature": 0.8, "logprobs": True},
        sample_kwargs={"data_source": "openai/gsm8k"},
    )

    assert output.response_ids == [1, 2, 3, 4, 5, 6]
    assert output.extra_fields["teacher_intervention_count"] == 0
    skipped = output.extra_fields["teacher_intervention_skipped"][0]
    assert skipped["reason"] == "outcome_correct"
    assert skipped["healthy_calibration_probe"] is True
    assert skipped["healthy_calibration_updated"] is True
    assert skipped["teacher_nll_calibration_scope"] == "global"
    assert skipped["teacher_nll_global_update_pending"] is True
    assert AgentLoopWorker.drain_teacher_nll_calibration_updates(worker) == pytest.approx([0.2])
    assert AgentLoopWorker.drain_teacher_nll_calibration_updates(worker) == []
    assert not any(":student-candidate-score:" in call["request_id"] for call in worker.llm_client.calls)


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
    training_batch = AgentLoopWorker._postprocess(worker, [internal], validate=False)
    serialized_intervention = training_batch.non_tensor_batch["teacher_interventions"][0][0]
    assert "_original_student_segment_ids" not in serialized_intervention
    assert "_teacher_segment_ids" not in serialized_intervention
    assert "teacher_intervention_outcome_case" not in training_batch.non_tensor_batch


@pytest.mark.asyncio
async def test_teacher_intervention_sanitizes_nonfinite_student_suffix_logprobs_on_cpu(caplog):
    teacher_manager = _FixedGapTeacherManager(
        TeacherGenerationResult(token_ids=[30, 31], finish_reason="length"),
    )
    worker = _DirectInterventionWorker(
        teacher_manager=teacher_manager,
        response_length=6,
        max_interventions_per_sample=2,
    )
    worker.llm_client = _NonfiniteRepairStudentClient()
    output = AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[1, 2, 3, 4, 5, 6],
        response_mask=[1, 1, 1, 1, 1, 1],
        response_logprobs=[-0.1] * 6,
        metrics=AgentLoopMetrics(),
        extra_fields={"rollout_request_id": "nonfinite-student-suffix"},
    )

    with caplog.at_level(logging.WARNING):
        await AgentLoopWorker._maybe_repair_with_teacher(
            worker,
            output,
            validate=False,
            sampling_params={"temperature": 0.8, "logprobs": True},
            sample_kwargs={"data_source": "openai/gsm8k"},
        )

    assert output.response_ids == [30, 31, 40, 41]
    assert output.response_logprobs == pytest.approx([0.0, 0.0, -10.0, -10.0])
    assert output.extra_fields["teacher_intervention_count"] == 1
    intervention = output.extra_fields["teacher_interventions"][0]
    assert intervention["student_suffix_logprob_nonfinite_count"] == 2
    assert intervention["student_suffix_logprob_nonfinite_offsets"] == [0, 1]
    assert intervention["student_suffix_logprob_nonfinite_response_positions"] == [2, 3]
    assert intervention["student_suffix_logprob_nonfinite_replacement"] == -10.0
    assert len(teacher_manager.scored_sequences) == 2
    assert "excluded those positions from subsequent intervention-gap detection" in caplog.text


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


@pytest.mark.asyncio
async def test_boundary_metrics_do_not_add_generation_or_scoring_calls_on_cpu():
    async def run(collect_boundary_metrics: bool):
        teacher_manager = _FixedGapTeacherManager(
            TeacherGenerationResult(token_ids=[30], stop_reason="completed", finish_reason="length")
        )
        worker = _DirectInterventionWorker(
            teacher_manager=teacher_manager,
            response_length=6,
            student_segment_length=2,
            teacher_segment_length=1,
            gap_metric="abs",
            gap_threshold=1.0,
            collect_intervention_boundary_metrics=collect_boundary_metrics,
        )
        output = AgentLoopOutput(
            prompt_ids=[101, 102],
            response_ids=[1, 2, 3, 4],
            response_mask=[1, 1, 1, 1],
            response_logprobs=[-0.1] * 4,
            metrics=AgentLoopMetrics(),
            extra_fields={"rollout_request_id": f"boundary-{collect_boundary_metrics}"},
        )
        await AgentLoopWorker._maybe_repair_with_teacher(
            worker,
            output,
            validate=False,
            sampling_params={"temperature": 0.8, "logprobs": True},
            sample_kwargs={"data_source": "openai/gsm8k"},
        )
        return worker, teacher_manager, output

    disabled_worker, disabled_teacher, disabled_output = await run(False)
    enabled_worker, enabled_teacher, enabled_output = await run(True)

    assert len(disabled_teacher.scored_sequences) == len(enabled_teacher.scored_sequences)
    assert len(disabled_teacher.generated_prefixes) == len(enabled_teacher.generated_prefixes)
    assert len(disabled_worker.llm_client.calls) == len(enabled_worker.llm_client.calls)
    assert disabled_output.response_ids == enabled_output.response_ids
    assert disabled_output.response_mask == enabled_output.response_mask
    assert disabled_output.response_logprobs == enabled_output.response_logprobs
    disabled_intervention = disabled_output.extra_fields["teacher_interventions"][0]
    enabled_intervention = enabled_output.extra_fields["teacher_interventions"][0]
    assert "_local_effect_v2_baseline_active_scores" not in disabled_intervention
    assert "_local_effect_v2_baseline_active_scores" in enabled_intervention


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
async def test_only_stu_teacher_topk_reuses_expanded_intervention_scoring_on_cpu():
    teacher_manager = _FixedGapTeacherManager(
        TeacherGenerationResult(token_ids=[30], finish_reason="stop"),
        topk_width=128,
    )
    worker = _DirectInterventionWorker(
        teacher_manager=teacher_manager,
        response_length=4,
        gap_threshold=999.0,
    )
    worker.config = SimpleNamespace(algorithm=SimpleNamespace(adv_estimator="token_reward_direct"))
    worker.distillation_config.distillation_loss.topk = 16
    worker.distillation_config.distillation_loss.teacher_topk = 128
    worker.distillation_config.distillation_loss.top_k_strategy = "only_stu_teacher_topk"
    worker.distillation_config.distillation_loss.loss_settings = SimpleNamespace(use_topk=False)
    output = AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[1, 2, 3, 4],
        response_mask=[1, 1, 1, 1],
        response_logprobs=[-0.1] * 4,
        metrics=AgentLoopMetrics(),
        extra_fields={"rollout_request_id": "teacher-topk-no-repair"},
    )

    await AgentLoopWorker._maybe_repair_with_teacher(
        worker,
        output,
        validate=False,
        sampling_params={"temperature": 0.8, "logprobs": True},
        sample_kwargs={"data_source": "openai/gsm8k"},
    )
    assert teacher_manager.prompt_logprobs_calls == [128]
    assert output.extra_fields["teacher_logprobs"].shape[-1] == 128

    await AgentLoopWorker._compute_teacher_logprobs(
        worker,
        output,
        prompt_ids=output.prompt_ids,
        response_ids=output.response_ids,
        validate=False,
        sample_kwargs={"data_source": "openai/gsm8k"},
    )

    assert teacher_manager.prompt_logprobs_calls == [128]


@pytest.mark.asyncio
async def test_only_stu_teacher_topk_repair_reuses_expanded_final_scoring_on_cpu():
    teacher_manager = _FixedGapTeacherManager(
        TeacherGenerationResult(token_ids=[30, 31], stop_reason="completed", finish_reason="length"),
        high_gap_start=2,
        topk_width=128,
    )
    worker = _DirectInterventionWorker(teacher_manager=teacher_manager, response_length=6)
    worker.config = SimpleNamespace(algorithm=SimpleNamespace(adv_estimator="token_reward_direct"))
    worker.distillation_config.distillation_loss.topk = 16
    worker.distillation_config.distillation_loss.teacher_topk = 128
    worker.distillation_config.distillation_loss.top_k_strategy = "only_stu_teacher_topk"
    worker.distillation_config.distillation_loss.loss_settings = SimpleNamespace(use_topk=False)
    output = AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[1, 2, 3, 4, 5, 6],
        response_mask=[1, 1, 1, 1, 1, 1],
        response_logprobs=[-0.1] * 6,
        metrics=AgentLoopMetrics(),
        extra_fields={"rollout_request_id": "teacher-topk-repair"},
    )

    await AgentLoopWorker._maybe_repair_with_teacher(
        worker,
        output,
        validate=False,
        sampling_params={"temperature": 0.8, "logprobs": True},
        sample_kwargs={"data_source": "openai/gsm8k"},
    )

    assert teacher_manager.prompt_logprobs_calls == [128, 128]
    assert output.extra_fields["teacher_logprobs"].shape[-1] == 128

    await AgentLoopWorker._compute_teacher_logprobs(
        worker,
        output,
        prompt_ids=output.prompt_ids,
        response_ids=output.response_ids,
        validate=False,
        sample_kwargs={"data_source": "openai/gsm8k"},
    )

    assert teacher_manager.prompt_logprobs_calls == [128, 128]


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
async def test_multiple_repairs_produce_one_sample_transition_and_one_final_reward_recompute_on_cpu():
    class _SequentialGapTeacherManager(_FixedGapTeacherManager):
        async def compute_teacher_logprobs_single(self, *args, **kwargs):
            gap_starts = [2, 4, 99]
            self.high_gap_start = gap_starts[min(len(self.scored_sequences), len(gap_starts) - 1)]
            return await super().compute_teacher_logprobs_single(*args, **kwargs)

    teacher_manager = _SequentialGapTeacherManager(
        TeacherGenerationResult(token_ids=[31], stop_reason="completed", finish_reason="length"),
        high_gap_width=1,
    )
    worker = _DirectInterventionWorker(
        teacher_manager=teacher_manager,
        response_length=8,
        student_segment_length=1,
        teacher_segment_length=1,
        max_interventions_per_sample=2,
        collect_intervention_outcome_transitions=True,
    )
    reward_handle = _SequenceAwareRewardHandle()
    worker.reward_loop_worker_handles = [reward_handle]
    output = AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[1, 2, 3, 4, 5, 6],
        response_mask=[1, 1, 1, 1, 1, 1],
        response_logprobs=[-0.1] * 6,
        metrics=AgentLoopMetrics(),
        extra_fields={"rollout_request_id": "multiple-repair-outcome"},
    )

    internal = await AgentLoopWorker._agent_loop_postprocess(
        worker,
        output,
        validate=False,
        sampling_params={"temperature": 0.8, "logprobs": True},
        raw_prompt=[{"role": "user", "content": "hi"}],
        data_source="openai/gsm8k",
        global_steps=2,
    )

    assert len(reward_handle.compute_score.responses) == 2
    assert internal.extra_fields["teacher_intervention_count"] == 2
    outcome = internal.extra_fields["teacher_intervention_outcome"]
    assert outcome["intervention_count"] == 2
    assert outcome["transition"] == "pre_wrong_post_correct"
    assert sum(bool(item.get("sample_outcome_summary")) for item in internal.extra_fields["teacher_interventions"]) == 1


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
async def test_teacher_manager_scores_exact_per_position_target_ids_on_cpu():
    target_ids = [[7, 8], [11, 12], [21, 22], [0, 0]]

    class _TargetTeacherClient:
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
            target_prompt_token_ids: Optional[list[list[int]]] = None,
        ) -> TokenOutput:
            del request_id, image_data, video_data, audio_data, mm_processor_kwargs
            assert prompt_ids == [101, 102, 1, 2]
            assert sampling_params["prompt_logprobs"] == 2
            assert target_prompt_token_ids == target_ids
            return TokenOutput(
                token_ids=[1],
                extra_fields={
                    "prompt_ids": [[7, 8], [11, 12], [21, 22], [0, 0]],
                    "prompt_logprobs": [[-1.0, -2.0], [-3.0, -4.0], [-5.0, -6.0], [0.0, 0.0]],
                    "sampled_prompt_ids": [[102], [1], [2], [0]],
                    "sampled_prompt_logprobs": [[-0.1], [-0.2], [-0.3], [0.0]],
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
        "teacher": SimpleNamespace(inference=SimpleNamespace(name="vllm", temperature=1.0)),
    }
    manager.teacher_client = {"teacher": _TargetTeacherClient()}

    result = await manager.compute_teacher_logprobs_for_target_ids_single(
        sequence_ids=[101, 102, 1, 2],
        target_ids=target_ids,
        routing_key="teacher",
        request_id="target-teacher-score",
    )

    torch.testing.assert_close(result.ids, torch.tensor(target_ids, dtype=torch.int32))
    torch.testing.assert_close(
        result.logprobs,
        torch.tensor([[-1.0, -2.0], [-3.0, -4.0], [-5.0, -6.0], [0.0, 0.0]]),
    )


@pytest.mark.asyncio
async def test_teacher_manager_rejects_teacher_topk_fallback_for_target_scoring_on_cpu():
    class _WrongSupportTeacherClient:
        async def generate(self, **kwargs) -> TokenOutput:
            del kwargs
            return TokenOutput(
                token_ids=[1],
                extra_fields={
                    "prompt_ids": [[90, 91], [92, 93], [94, 95]],
                    "prompt_logprobs": [[-1.0, -2.0], [-3.0, -4.0], [0.0, 0.0]],
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
        "teacher": SimpleNamespace(inference=SimpleNamespace(name="vllm", temperature=1.0)),
    }
    manager.teacher_client = {"teacher": _WrongSupportTeacherClient()}

    with pytest.raises(RuntimeError, match="did not return the requested per-position target ids"):
        await manager.compute_teacher_logprobs_for_target_ids_single(
            sequence_ids=[101, 102, 1],
            target_ids=[[7, 8], [11, 12], [0, 0]],
            routing_key="teacher",
        )


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


class _FutureGapTeacherManager(_FixedGapTeacherManager):
    def __init__(self, *, improved: bool):
        super().__init__(
            TeacherGenerationResult(token_ids=[30, 31], stop_reason="completed", finish_reason="length"),
            high_gap_start=0,
            high_gap_width=4 if improved else 2,
        )
        self.improved = improved

    async def compute_teacher_logprobs_single(self, *args, **kwargs):
        if self.scored_sequences:
            self.high_gap_start = 99
            self.high_gap_width = 0
        return await super().compute_teacher_logprobs_single(*args, **kwargs)


class _FutureGapCandidateSelectionTeacherManager(_CandidateSelectionTeacherManager):
    def __init__(self):
        super().__init__(
            [[30, 30], [31, 31], [32, 32], [33, 33]],
            teacher_nll_by_token={30: 0.1, 31: 0.2, 32: 0.3, 33: 0.4},
            high_gap_width=4,
        )
        self.intervention_score_calls = 0

    async def compute_teacher_logprobs_single(self, *args, **kwargs):
        if ":teacher-score:" in str(kwargs.get("request_id", "")):
            if self.intervention_score_calls:
                self.high_gap_width = 0
            self.intervention_score_calls += 1
        return await super().compute_teacher_logprobs_single(*args, **kwargs)


def _future_gap_output(request_id: str) -> AgentLoopOutput:
    return AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[1, 2, 3, 4, 5, 6],
        response_mask=[1, 1, 1, 1, 1, 1],
        response_logprobs=[-0.1] * 6,
        metrics=AgentLoopMetrics(),
        extra_fields={"rollout_request_id": request_id},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("gap_metric", "gap_threshold"), [("k3", 0.15), ("teacher_rejection", 0.3)])
async def test_future_gap_accepts_improved_candidate_for_supported_trigger_metrics_on_cpu(
    gap_metric,
    gap_threshold,
):
    teacher_manager = _FutureGapTeacherManager(improved=True)
    worker = _DirectInterventionWorker(
        teacher_manager=teacher_manager,
        response_length=6,
        student_segment_length=2,
        teacher_segment_length=2,
        gap_metric=gap_metric,
        gap_threshold=gap_threshold,
        teacher_rejection_persistence_enabled=gap_metric == "teacher_rejection",
        teacher_rejection_persistence_window_tokens=2,
        teacher_rejection_persistence_threshold=gap_threshold,
        teacher_rejection_persistence_min_valid_tokens=2,
        confirmation_mode="future_gap",
        future_gap_window_tokens=2,
        future_gap_min_valid_tokens=2,
        train_on_teacher_segments=True,
        teacher_segment_loss_mode="opd",
    )
    output = _future_gap_output(f"future-gap-accept-{gap_metric}")

    await AgentLoopWorker._maybe_repair_with_teacher(
        worker,
        output,
        validate=False,
        sampling_params={"temperature": 0.8, "logprobs": True},
        sample_kwargs={"data_source": "openai/gsm8k", "global_steps": 3, "uid": "sample"},
    )

    assert output.response_ids == [30, 31, 40, 41]
    assert output.response_mask == [1, 1, 1, 1]
    assert output.extra_fields["teacher_intervention_count"] == 1
    proposal = output.extra_fields["teacher_intervention_proposals"][0]
    assert proposal["accepted"] is True
    assert proposal["future_gap_pair_available"] is True
    assert proposal["future_gap_effective_tokens"] == 2
    assert proposal["future_gap_after"] < proposal["future_gap_max_threshold"]
    assert proposal["future_gap_reduction"] > 0
    assert len(teacher_manager.scored_sequences) == 2
    assert len(teacher_manager.generated_prefixes) == 1
    assert len(worker.llm_client.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("train_on_teacher_segments", "teacher_segment_loss_mode", "expected_response_mask", "expected_teacher_mask"),
    [
        (False, "opd", [0, 0, 1, 1], None),
        (True, "opd", [1, 1, 1, 1], None),
        (True, "sft", [0, 0, 1, 1], [1, 1, 0, 0]),
    ],
)
async def test_future_gap_acceptance_preserves_teacher_segment_training_mode_on_cpu(
    train_on_teacher_segments,
    teacher_segment_loss_mode,
    expected_response_mask,
    expected_teacher_mask,
):
    worker = _DirectInterventionWorker(
        teacher_manager=_FutureGapTeacherManager(improved=True),
        response_length=6,
        gap_metric="k3",
        gap_threshold=0.15,
        confirmation_mode="future_gap",
        future_gap_window_tokens=2,
        future_gap_min_valid_tokens=2,
        train_on_teacher_segments=train_on_teacher_segments,
        teacher_segment_loss_mode=teacher_segment_loss_mode,
    )
    output = _future_gap_output(f"future-gap-training-mode-{train_on_teacher_segments}-{teacher_segment_loss_mode}")

    await AgentLoopWorker._maybe_repair_with_teacher(
        worker,
        output,
        validate=False,
        sampling_params={"temperature": 0.8, "logprobs": True},
        sample_kwargs={"data_source": "openai/gsm8k"},
    )

    assert output.extra_fields["teacher_intervention_count"] == 1
    assert output.response_mask == expected_response_mask
    assert output.teacher_segment_mask == expected_teacher_mask


@pytest.mark.asyncio
async def test_future_gap_rejection_restores_exact_raw_state_and_teacher_score_on_cpu():
    teacher_manager = _FutureGapTeacherManager(improved=False)
    worker = _DirectInterventionWorker(
        teacher_manager=teacher_manager,
        response_length=6,
        student_segment_length=2,
        teacher_segment_length=2,
        gap_metric="k3",
        gap_threshold=0.15,
        confirmation_mode="future_gap",
        future_gap_window_tokens=2,
        future_gap_min_valid_tokens=2,
        train_on_teacher_segments=True,
        teacher_segment_loss_mode="sft",
    )
    output = _future_gap_output("future-gap-reject")
    original_ids = list(output.response_ids)
    original_mask = list(output.response_mask)
    original_logprobs = list(output.response_logprobs)

    await AgentLoopWorker._maybe_repair_with_teacher(
        worker,
        output,
        validate=False,
        sampling_params={"temperature": 0.8, "logprobs": True},
        sample_kwargs={"data_source": "openai/gsm8k"},
    )

    assert output.response_ids == original_ids
    assert output.response_mask == original_mask
    assert output.response_logprobs == original_logprobs
    assert output.teacher_segment_mask == [0, 0, 0, 0, 0, 0]
    assert output.extra_fields["teacher_intervention_count"] == 0
    assert output.extra_fields["teacher_interventions"] == []
    proposal = output.extra_fields["teacher_intervention_proposals"][0]
    assert proposal["accepted"] is False
    assert proposal["reason"] == "future_gap_insufficient_reduction"
    assert not any(str(key).startswith("_candidate") for key in proposal)
    assert output.extra_fields["teacher_logprobs"].shape[0] == len(output.prompt_ids) + len(original_ids)
    assert len(teacher_manager.scored_sequences) == 2
    assert len(teacher_manager.generated_prefixes) == 1
    assert len(worker.llm_client.calls) == 1


@pytest.mark.asyncio
async def test_future_gap_rejects_terminal_candidate_without_extra_scoring_on_cpu():
    teacher_manager = _FixedGapTeacherManager(
        TeacherGenerationResult(token_ids=[30], stop_reason="completed", finish_reason="stop"),
        high_gap_start=0,
        high_gap_width=4,
    )
    worker = _DirectInterventionWorker(
        teacher_manager=teacher_manager,
        response_length=6,
        gap_metric="k3",
        gap_threshold=0.15,
        confirmation_mode="future_gap",
        future_gap_window_tokens=2,
        future_gap_min_valid_tokens=2,
        min_trainable_tokens_after_repair=0,
    )
    output = _future_gap_output("future-gap-terminal")

    await AgentLoopWorker._maybe_repair_with_teacher(
        worker,
        output,
        validate=False,
        sampling_params={"temperature": 0.8, "logprobs": True},
        sample_kwargs={"data_source": "openai/gsm8k"},
    )

    assert output.response_ids == [1, 2, 3, 4, 5, 6]
    assert output.extra_fields["teacher_intervention_count"] == 0
    assert output.extra_fields["teacher_intervention_proposals"][0]["reason"] == "future_gap_terminal_response"
    assert len(teacher_manager.scored_sequences) == 1
    assert worker.llm_client.calls == []


@pytest.mark.asyncio
async def test_future_gap_rejects_candidate_with_too_few_matched_tokens_on_cpu():
    teacher_manager = _FutureGapTeacherManager(improved=True)
    worker = _DirectInterventionWorker(
        teacher_manager=teacher_manager,
        response_length=6,
        gap_metric="k3",
        gap_threshold=0.15,
        confirmation_mode="future_gap",
        future_gap_window_tokens=3,
        future_gap_min_valid_tokens=3,
    )
    output = _future_gap_output("future-gap-short")

    await AgentLoopWorker._maybe_repair_with_teacher(
        worker,
        output,
        validate=False,
        sampling_params={"temperature": 0.8, "logprobs": True},
        sample_kwargs={"data_source": "openai/gsm8k"},
    )

    proposal = output.extra_fields["teacher_intervention_proposals"][0]
    assert proposal["accepted"] is False
    assert proposal["future_gap_effective_tokens"] == 2
    assert proposal["reason"] == "future_gap_insufficient_valid_tokens"
    assert output.response_ids == [1, 2, 3, 4, 5, 6]


@pytest.mark.asyncio
async def test_future_gap_does_not_add_generation_or_scoring_calls_vs_single_hard_repair_on_cpu():
    call_counts = {}
    for confirmation_mode in ("none", "future_gap"):
        teacher_manager = _FutureGapTeacherManager(improved=True)
        worker = _DirectInterventionWorker(
            teacher_manager=teacher_manager,
            response_length=6,
            gap_metric="k3",
            gap_threshold=0.15,
            confirmation_mode=confirmation_mode,
            future_gap_window_tokens=2,
            future_gap_min_valid_tokens=2,
        )
        output = _future_gap_output(f"call-count-{confirmation_mode}")
        await AgentLoopWorker._maybe_repair_with_teacher(
            worker,
            output,
            validate=False,
            sampling_params={"temperature": 0.8, "logprobs": True},
            sample_kwargs={"data_source": "openai/gsm8k"},
        )
        call_counts[confirmation_mode] = (
            len(teacher_manager.scored_sequences),
            len(teacher_manager.generated_prefixes),
            len(worker.llm_client.calls),
        )

    assert call_counts == {"none": (2, 1, 1), "future_gap": (2, 1, 1)}


@pytest.mark.asyncio
async def test_future_gap_runs_after_existing_outcome_aware_candidate_selection_on_cpu():
    teacher_manager = _FutureGapCandidateSelectionTeacherManager()
    worker = _DirectInterventionWorker(
        teacher_manager=teacher_manager,
        response_length=6,
        outcome_aware_candidate_selection_enabled=True,
        teacher_nll_initial_threshold=1.0,
        confirmation_mode="future_gap",
        future_gap_window_tokens=2,
        future_gap_min_valid_tokens=2,
    )
    worker.llm_client = _CandidateSelectionStudentClient(student_nll_by_token={30: 0.4, 31: 0.1, 32: 0.2, 33: 0.05})
    output = _outcome_gated_intervention_output(request_id="outcome-aware-future-gap", raw_correct=False)

    await AgentLoopWorker._maybe_repair_with_teacher(
        worker,
        output,
        validate=False,
        sampling_params={"temperature": 0.8, "logprobs": True},
        sample_kwargs={"data_source": "openai/gsm8k"},
    )

    assert output.extra_fields["teacher_intervention_count"] == 1
    proposal = output.extra_fields["teacher_intervention_proposals"][0]
    assert proposal["accepted"] is True
    assert output.extra_fields["teacher_interventions"][0]["candidate_selection_enabled"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("confirmation_mode", ["none", "future_gap"])
async def test_future_gap_does_not_add_training_reward_calls_on_cpu(confirmation_mode):
    worker = _DirectInterventionWorker(
        teacher_manager=_FutureGapTeacherManager(improved=True),
        response_length=6,
        gap_metric="k3",
        gap_threshold=0.15,
        confirmation_mode=confirmation_mode,
        future_gap_window_tokens=2,
        future_gap_min_valid_tokens=2,
    )
    reward_handle = _SequenceAwareRewardHandle()
    worker.reward_loop_worker_handles = [reward_handle]

    await AgentLoopWorker._agent_loop_postprocess(
        worker,
        _future_gap_output(f"future-gap-reward-count-{confirmation_mode}"),
        validate=False,
        sampling_params={"temperature": 0.8, "logprobs": True},
        raw_prompt=[{"role": "user", "content": "hi"}],
        data_source="openai/gsm8k",
    )

    assert reward_handle.compute_score.responses == [[30, 31, 40, 41]]


@pytest.mark.asyncio
@pytest.mark.parametrize("confirmation_mode", ["none", "future_gap"])
@pytest.mark.parametrize("gap_metric", ["k3", "teacher_rejection"])
async def test_skip_correct_bypasses_all_intervention_model_calls_on_cpu(confirmation_mode, gap_metric):
    teacher_manager = _FutureGapTeacherManager(improved=True)
    worker = _DirectInterventionWorker(
        teacher_manager=teacher_manager,
        response_length=6,
        gap_metric=gap_metric,
        gap_threshold=0.15,
        skip_correct=True,
        confirmation_mode=confirmation_mode,
        future_gap_window_tokens=2,
        future_gap_min_valid_tokens=2,
        train_on_teacher_segments=True,
        teacher_segment_loss_mode="sft",
    )
    output = _future_gap_output(f"skip-correct-{confirmation_mode}-{gap_metric}")
    output.extra_fields["teacher_intervention_outcome"] = {
        "enabled": True,
        "raw_correct": True,
        "trigger_detected": False,
        "repair_executed": False,
    }

    await AgentLoopWorker._maybe_repair_with_teacher(
        worker,
        output,
        validate=False,
        sampling_params={"temperature": 0.8, "logprobs": True},
        sample_kwargs={"data_source": "openai/gsm8k"},
    )

    assert teacher_manager.scored_sequences == []
    assert teacher_manager.generated_prefixes == []
    assert worker.llm_client.calls == []
    assert output.response_ids == [1, 2, 3, 4, 5, 6]
    assert output.teacher_segment_mask == [0, 0, 0, 0, 0, 0]
    assert output.extra_fields["teacher_intervention_skipped"][0]["reason"] == "skip_correct"
    if confirmation_mode == "future_gap":
        assert output.extra_fields["teacher_intervention_proposals"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("confirmation_mode", ["none", "future_gap"])
async def test_skip_correct_allows_incorrect_on_policy_answer_to_continue_through_intervention_on_cpu(
    confirmation_mode,
):
    teacher_manager = _FutureGapTeacherManager(improved=True)
    worker = _DirectInterventionWorker(
        teacher_manager=teacher_manager,
        response_length=6,
        gap_metric="k3",
        gap_threshold=0.15,
        skip_correct=True,
        confirmation_mode=confirmation_mode,
        future_gap_window_tokens=2,
        future_gap_min_valid_tokens=2,
    )
    output = _future_gap_output(f"skip-incorrect-{confirmation_mode}")
    output.extra_fields["teacher_intervention_outcome"] = {
        "enabled": True,
        "raw_correct": False,
        "trigger_detected": False,
        "repair_executed": False,
    }

    await AgentLoopWorker._maybe_repair_with_teacher(
        worker,
        output,
        validate=False,
        sampling_params={"temperature": 0.8, "logprobs": True},
        sample_kwargs={"data_source": "openai/gsm8k"},
    )

    assert output.response_ids == [30, 31, 40, 41]
    assert output.extra_fields["teacher_intervention_count"] == 1
    assert len(teacher_manager.scored_sequences) == 2
    assert len(teacher_manager.generated_prefixes) == 1
    assert len(worker.llm_client.calls) == 1


@pytest.mark.asyncio
async def test_future_gap_rejected_candidate_gets_evaluation_only_outcome_pair_on_cpu():
    teacher_manager = _FutureGapTeacherManager(improved=False)
    worker = _DirectInterventionWorker(
        teacher_manager=teacher_manager,
        response_length=6,
        student_segment_length=2,
        teacher_segment_length=2,
        gap_metric="k3",
        gap_threshold=0.15,
        confirmation_mode="future_gap",
        future_gap_window_tokens=2,
        future_gap_min_valid_tokens=2,
        repair_for_evaluation=True,
    )
    reward_handle = _SequenceAwareRewardHandle()
    worker.reward_loop_worker_handles = [reward_handle]
    output = _future_gap_output("future-gap-eval-rejected")

    internal = await AgentLoopWorker._agent_loop_postprocess(
        worker,
        output,
        validate=True,
        repair_for_evaluation=True,
        sampling_params={"temperature": 0.8, "logprobs": True},
        raw_prompt=[{"role": "user", "content": "hi"}],
        data_source="openai/gsm8k",
        global_steps=5,
    )

    assert reward_handle.compute_score.responses == [[1, 2, 3, 4, 5, 6], [30, 31, 40, 41]]
    assert internal.response_ids[0, :6].tolist() == [1, 2, 3, 4, 5, 6]
    proposal = internal.extra_fields["teacher_intervention_proposals"][0]
    assert proposal["accepted"] is False
    assert proposal["candidate_correct"] is True
    assert proposal["gated_correct"] is False
    assert proposal["candidate_correctness_delta"] == 1
    assert proposal["candidate_transition"] == "pre_wrong_post_correct"
    assert proposal["outcome_scope"] == "evaluation_candidate_pair"
    assert proposal["candidate_response"]
    assert not any(str(key).startswith("_candidate") for key in proposal)
