# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
from typing import Any, Optional
from uuid import uuid4

import torch
from omegaconf import DictConfig
from torch.nn import functional as F

from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import (
    DistillationConfig,
    DistillationInterventionConfig,
    DistillationLossConfig,
    DistillationTeacherModelConfig,
)
from verl.workers.rollout.llm_server import LLMServerClient


def _get_teacher_sampling_params(
    teacher_model_config: DistillationTeacherModelConfig,
    distillation_loss_config: DistillationLossConfig,
    prompt_logprobs: Optional[int] = None,
) -> dict[str, Any]:
    """Get sampling parameters for teacher model when computing log probabilities for distillation."""
    if teacher_model_config.inference.temperature != 1.0:
        raise NotImplementedError("vLLM does not support temperature for prompt_logprobs.")

    num_logprobs = (
        prompt_logprobs
        if prompt_logprobs is not None
        else distillation_loss_config.topk
        if distillation_loss_config.loss_settings.use_topk
        else 0
    )
    if num_logprobs is None or num_logprobs < 0:
        raise ValueError(
            f"Teacher logprob scoring requires prompt_logprobs to be a non-negative integer, but got {num_logprobs!r}."
        )
    return {
        "max_tokens": 1,
        "temperature": teacher_model_config.inference.temperature,
        "prompt_logprobs": num_logprobs,
    }


def _get_teacher_generation_sampling_params(
    intervention_config: DistillationInterventionConfig,
    max_tokens: int,
) -> dict[str, Any]:
    """Get sampling parameters for teacher intervention generation."""
    if max_tokens <= 0:
        raise ValueError(f"Teacher intervention generation requires max_tokens > 0, got {max_tokens}.")
    return {
        "max_tokens": max_tokens,
        "temperature": intervention_config.teacher_temperature,
        "top_p": intervention_config.teacher_top_p,
        "top_k": intervention_config.teacher_top_k,
        "logprobs": False,
    }


def _expected_prompt_logprob_width(
    distillation_loss_config: DistillationLossConfig,
    prompt_logprobs: Optional[int],
) -> int:
    if prompt_logprobs is not None:
        return max(prompt_logprobs, 1)
    if distillation_loss_config.loss_settings.use_topk:
        if distillation_loss_config.topk is None:
            raise ValueError("distillation_loss.topk must be set when the selected loss uses top-k logprobs.")
        return distillation_loss_config.topk
    return 1


def _pad_teacher_outputs(
    teacher_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    prompt_width: int,
    response_width: int,
    prompt_length: int,
    response_length: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO(wuxibin): remove padding and use tensordict.
    left_pad_size = prompt_width - prompt_length
    right_pad_size = response_width - response_length
    padding = (0, 0, left_pad_size, right_pad_size)
    return (
        F.pad(teacher_ids, padding, value=pad_token_id).unsqueeze(0),
        F.pad(teacher_logprobs, padding, value=0.0).unsqueeze(0),
    )


class AsyncTeacherLLMServerManager:
    """Teacher-specific async client used for distillation logprob computation."""

    def __init__(
        self,
        config: DictConfig,
        teacher_client: dict[str, LLMServerClient],
    ):
        self.distillation_config: DistillationConfig = omega_conf_to_dataclass(config.distillation)
        self.distillation_loss_config: DistillationLossConfig = self.distillation_config.distillation_loss
        self.teacher_key: str = self.distillation_config.teacher_key

        self.teacher_model_configs: dict[str, DistillationTeacherModelConfig] = self.distillation_config.teacher_models
        expected = set(self.teacher_model_configs)
        if set(teacher_client.keys()) != expected:
            raise ValueError(
                f"teacher client keys {sorted(teacher_client.keys())} "
                f"do not match teacher routing keys {sorted(expected)}."
            )
        self.teacher_client: dict[str, LLMServerClient] = teacher_client

    def _resolve_teacher_key(self, routing_key: Optional[str]) -> str:
        if len(self.teacher_model_configs) == 1:
            # Single-teacher path: route everything to the one teacher regardless of the sample's key.
            return next(iter(self.teacher_model_configs))
        if routing_key is None:
            raise ValueError(
                f"Routing key is required for multi-teacher distillation "
                f"(configured via distillation.teacher_key={self.teacher_key!r})."
            )
        if routing_key not in self.teacher_model_configs:
            raise ValueError(
                f"No teacher configured for routing key {routing_key!r}. "
                f"Configured teachers: {sorted(self.teacher_model_configs)}."
            )
        return routing_key

    async def compute_teacher_logprobs_single(
        self,
        sequence_ids: list[int],
        multi_modal_data: Optional[dict[str, Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        routing_key: Optional[str] = None,
        request_id: Optional[str] = None,
        prompt_logprobs: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute teacher log probabilities for a single unpadded sequence."""
        if not sequence_ids:
            raise ValueError("compute_teacher_logprobs_single received an empty sequence.")
        multi_modal_data = multi_modal_data or {}
        teacher_key = self._resolve_teacher_key(routing_key)
        teacher_model_config = self.teacher_model_configs[teacher_key]
        client = self.teacher_client[teacher_key]
        resolved_request_id = request_id or uuid4().hex
        sampling_params = _get_teacher_sampling_params(
            teacher_model_config,
            self.distillation_loss_config,
            prompt_logprobs=prompt_logprobs,
        )
        teacher_output = await client.generate(
            request_id=resolved_request_id,
            prompt_ids=sequence_ids,
            sampling_params=sampling_params,
            image_data=multi_modal_data.get("images"),
            video_data=multi_modal_data.get("videos"),
            audio_data=multi_modal_data.get("audios"),
            mm_processor_kwargs=mm_processor_kwargs,
        )
        # Shapes: # S, (1 or K), where S is the response length, K is either 1 or topk depending on
        # the distillation loss settings.
        extra_fields = teacher_output.extra_fields or {}
        missing_keys = {"prompt_ids", "prompt_logprobs"} - set(extra_fields)
        if missing_keys:
            raise RuntimeError(
                "Teacher logprob scoring did not return the required prompt scoring fields "
                f"{sorted(missing_keys)} for teacher={teacher_key!r}, request_id={resolved_request_id!r}, "
                f"sequence_len={len(sequence_ids)}, sampling_params={sampling_params}. "
                f"Returned extra_fields keys: {sorted(extra_fields.keys())}. "
                "If this is a local vLLM Metal run, note that vllm_metal currently does not populate "
                "prompt_logprobs; validate teacher scoring on the CUDA vLLM/SGLang backend used for training."
            )
        try:
            teacher_ids = torch.tensor(extra_fields["prompt_ids"], dtype=torch.int32)
            teacher_logprobs = torch.tensor(extra_fields["prompt_logprobs"])
        except Exception as exc:
            raise RuntimeError(
                "Teacher logprob scoring returned prompt scoring fields that could not be converted to tensors for "
                f"teacher={teacher_key!r}, request_id={resolved_request_id!r}. "
                f"prompt_ids type={type(extra_fields['prompt_ids']).__name__}, "
                f"prompt_logprobs type={type(extra_fields['prompt_logprobs']).__name__}. "
                "Check that the teacher backend returns rectangular numeric lists."
            ) from exc
        expected_width = _expected_prompt_logprob_width(self.distillation_loss_config, prompt_logprobs)
        if teacher_ids.ndim != 2 or teacher_logprobs.ndim != 2:
            raise RuntimeError(
                "Teacher logprob scoring returned tensors with invalid ranks for "
                f"teacher={teacher_key!r}, request_id={resolved_request_id!r}: "
                f"teacher_ids.shape={tuple(teacher_ids.shape)}, "
                f"teacher_logprobs.shape={tuple(teacher_logprobs.shape)}. "
                "Expected both tensors to be rank-2 with shape [sequence_len, width]."
            )
        if teacher_ids.shape[0] != len(sequence_ids) or teacher_logprobs.shape[0] != len(sequence_ids):
            raise RuntimeError(
                "Teacher logprob scoring returned a sequence length mismatch for "
                f"teacher={teacher_key!r}, request_id={resolved_request_id!r}: "
                f"input sequence_len={len(sequence_ids)}, teacher_ids.shape={tuple(teacher_ids.shape)}, "
                f"teacher_logprobs.shape={tuple(teacher_logprobs.shape)}."
            )
        if teacher_ids.shape[1] != expected_width or teacher_logprobs.shape[1] != expected_width:
            raise RuntimeError(
                "Teacher logprob scoring returned an unexpected prompt_logprobs width for "
                f"teacher={teacher_key!r}, request_id={resolved_request_id!r}: "
                f"expected width={expected_width}, teacher_ids.shape={tuple(teacher_ids.shape)}, "
                f"teacher_logprobs.shape={tuple(teacher_logprobs.shape)}, sampling_params={sampling_params}."
            )
        if not torch.isfinite(teacher_logprobs.float()).all():
            raise RuntimeError(
                "Teacher logprob scoring returned NaN/Inf values for "
                f"teacher={teacher_key!r}, request_id={resolved_request_id!r}."
            )
        return teacher_ids, teacher_logprobs

    async def generate_teacher_segment_single(
        self,
        prefix_ids: list[int],
        max_tokens: int,
        multi_modal_data: Optional[dict[str, Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        routing_key: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> list[int]:
        """Generate a short teacher intervention segment from a single unpadded prefix."""
        if not prefix_ids:
            raise ValueError("generate_teacher_segment_single received an empty prefix.")
        multi_modal_data = multi_modal_data or {}
        teacher_key = self._resolve_teacher_key(routing_key)
        client = self.teacher_client[teacher_key]
        resolved_request_id = request_id or uuid4().hex
        teacher_output = await client.generate(
            request_id=resolved_request_id,
            prompt_ids=prefix_ids,
            sampling_params=_get_teacher_generation_sampling_params(
                self.distillation_config.intervention,
                max_tokens=max_tokens,
            ),
            image_data=multi_modal_data.get("images"),
            video_data=multi_modal_data.get("videos"),
            audio_data=multi_modal_data.get("audios"),
            mm_processor_kwargs=mm_processor_kwargs,
        )
        if teacher_output.token_ids is None:
            raise RuntimeError(
                f"Teacher intervention generation returned no token_ids for "
                f"teacher={teacher_key!r}, request_id={resolved_request_id!r}."
            )
        return teacher_output.token_ids
