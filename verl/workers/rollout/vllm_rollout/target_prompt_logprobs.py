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

"""Exact, per-position target prompt log-probabilities for vLLM V1.

vLLM's public ``prompt_logprobs=K`` API returns the teacher's own top-K
tokens.  Rethinking-OPD ``only_stu`` instead needs the teacher probability of
each *student* top-K token, and that target set is different at every sequence
position.  The public API cannot express that contract.

This module installs a narrowly gated wrapper around vLLM V1's prompt-logprob
routine.  Normal requests call the original routine unchanged.  A request
whose ``SamplingParams.extra_args`` contains
``VERL_TARGET_PROMPT_TOKEN_IDS_KEY`` gathers the supplied token ids from the
teacher's full normalized distribution at each prompt position.
"""

from __future__ import annotations

from types import MethodType
from typing import Any

import torch

from verl.utils.device import get_torch_device

VERL_TARGET_PROMPT_TOKEN_IDS_KEY = "verl_target_prompt_token_ids"
_PATCHED_ATTRIBUTE = "_verl_target_prompt_logprobs_patched"


def _build_target_logprob_rows(
    *,
    normalized_logprobs: torch.Tensor,
    target_token_ids: torch.Tensor,
    sampled_token_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build vLLM ``LogprobsTensors`` rows for exact target token ids.

    vLLM allocates ``K + 1`` output columns for ``prompt_logprobs=K``.  Its
    serialization contract requires the actual prompt token in column zero
    and assigns ranks ``1..K`` to the following K columns.  The separate
    ``selected_token_ranks`` tensor is one-dimensional: it contains only the
    full-vocabulary rank of the actual prompt token.

    We put the K per-position student targets in columns ``1..K`` and use
    ``K + 1`` as a sentinel selected-token rank.  If the actual prompt token
    is also a student target, vLLM's dict conversion overwrites the duplicate
    column-zero entry with its synthetic rank in ``1..K``.  Otherwise the
    sentinel keeps the extra actual-token entry outside the K target ranks.
    """
    if normalized_logprobs.ndim != 2:
        raise ValueError(
            f"normalized_logprobs must have shape [num_positions, vocab_size], got {tuple(normalized_logprobs.shape)}."
        )
    if target_token_ids.ndim != 2:
        raise ValueError(
            f"target_token_ids must have shape [num_positions, topk], got {tuple(target_token_ids.shape)}."
        )
    if sampled_token_ids.ndim != 1:
        raise ValueError(f"sampled_token_ids must have shape [num_positions], got {tuple(sampled_token_ids.shape)}.")
    if normalized_logprobs.shape[0] != target_token_ids.shape[0] or target_token_ids.shape[0] != len(sampled_token_ids):
        raise ValueError(
            "Target prompt-logprob tensors must share their position dimension, "
            f"got normalized_logprobs={tuple(normalized_logprobs.shape)}, "
            f"target_token_ids={tuple(target_token_ids.shape)}, "
            f"sampled_token_ids={tuple(sampled_token_ids.shape)}."
        )

    # ``gather`` below enforces the token-id range. Do not call ``min().item()``
    # or ``max().item()`` here: this function runs once per request per
    # chunk, and each scalar read would serialize the entire NPU stream.
    target_logprobs = normalized_logprobs.gather(dim=-1, index=target_token_ids)
    target_width = target_token_ids.shape[-1]
    sampled_logprobs = normalized_logprobs.gather(dim=-1, index=sampled_token_ids.unsqueeze(-1))
    output_ids = torch.cat((sampled_token_ids.unsqueeze(-1), target_token_ids), dim=-1)
    output_logprobs = torch.cat((sampled_logprobs, target_logprobs), dim=-1)
    selected_token_ranks = torch.full(
        (target_token_ids.shape[0],),
        target_width + 1,
        dtype=torch.long,
        device=target_token_ids.device,
    )
    return output_ids, output_logprobs, selected_token_ranks


def _num_scheduled_tokens_by_request(scheduling_state: Any) -> dict[str, int]:
    if isinstance(scheduling_state, dict):
        return scheduling_state
    num_scheduled_tokens = getattr(scheduling_state, "num_scheduled_tokens", None)
    if not isinstance(num_scheduled_tokens, dict):
        raise RuntimeError(
            "Unsupported vLLM prompt-logprob scheduling state: expected a request-token dict or an object "
            f"with num_scheduled_tokens, got {type(scheduling_state).__name__}."
        )
    return num_scheduled_tokens


def _prompt_logprob_request_widths(model_runner: Any) -> dict[str, int]:
    widths = getattr(model_runner, "num_prompt_logprobs", None)
    if widths is None:
        widths = getattr(model_runner.input_batch, "num_prompt_logprobs", None)
    if not isinstance(widths, dict):
        raise RuntimeError("The active vLLM model runner does not expose prompt-logprob request widths.")
    return widths


def _query_offset(model_runner: Any, request_index: int) -> int:
    query_start_loc = getattr(model_runner, "query_start_loc", None)
    if query_start_loc is not None and hasattr(query_start_loc, "np"):
        return int(query_start_loc.np[request_index].item())
    query_start_loc_np = getattr(model_runner, "query_start_loc_np", None)
    if query_start_loc_np is not None:
        return int(query_start_loc_np[request_index].item())
    raise RuntimeError("The active vLLM model runner does not expose prompt query offsets.")


def _compute_logits(model_runner: Any, hidden_states: torch.Tensor) -> torch.Tensor:
    try:
        return model_runner.model.compute_logits(hidden_states)
    except TypeError:
        # vLLM 0.8-0.10 kept a second sampling-metadata argument.
        return model_runner.model.compute_logits(hidden_states, None)


def _sync_model_runner_device(model_runner: Any) -> None:
    sync_device = getattr(model_runner, "_sync_device", None)
    if sync_device is not None:
        sync_device()
        return

    runner_device = getattr(model_runner, "device", None)
    runner_device_type = getattr(runner_device, "type", None)
    torch_device = getattr(torch, runner_device_type, None) if runner_device_type is not None else None
    if torch_device is None:
        torch_device = get_torch_device()
    synchronize = getattr(torch_device, "synchronize", None)
    if synchronize is not None:
        synchronize()


def patch_vllm_model_runner_target_prompt_logprobs(
    model_runner: Any,
    *,
    logprobs_tensors_cls: type | None = None,
) -> None:
    """Install exact target prompt-logprob support on one vLLM V1 runner.

    The wrapper delegates byte-for-byte normal requests to vLLM's original
    method.  Target and normal prompt-logprob requests may not share a model
    runner batch; verl invokes target scoring only after rollout requests have
    drained.
    """
    if getattr(model_runner, _PATCHED_ATTRIBUTE, False):
        return
    original_method = getattr(model_runner, "_get_prompt_logprobs_dict", None)
    if original_method is None:
        raise RuntimeError(
            "only_stu_rethinking_style requires a vLLM V1 model runner with _get_prompt_logprobs_dict support."
        )
    if logprobs_tensors_cls is None:
        from vllm.v1.outputs import LogprobsTensors

        logprobs_tensors_cls = LogprobsTensors

    def _get_prompt_logprobs_dict_with_targets(
        self,
        hidden_states: torch.Tensor,
        scheduling_state: Any,
    ):
        request_widths = _prompt_logprob_request_widths(self)
        if not request_widths:
            return original_method(hidden_states, scheduling_state)

        target_request_ids = []
        normal_request_ids = []
        for request_id in request_widths:
            request = self.requests[request_id]
            sampling_params = getattr(request, "sampling_params", None)
            extra_args = getattr(sampling_params, "extra_args", None) or {}
            if VERL_TARGET_PROMPT_TOKEN_IDS_KEY in extra_args:
                target_request_ids.append(request_id)
            else:
                normal_request_ids.append(request_id)

        if not target_request_ids:
            # Reproducibility guard: legacy only_stu and every ordinary vLLM
            # request execute the unmodified upstream implementation.
            return original_method(hidden_states, scheduling_state)
        if normal_request_ids:
            raise RuntimeError(
                "Target and ordinary prompt-logprob requests were mixed in one vLLM model-runner batch. "
                "Run only_stu_rethinking_style target scoring after rollout requests have drained. "
                f"target_request_count={len(target_request_ids)}, normal_request_count={len(normal_request_ids)}."
            )

        num_scheduled_tokens = _num_scheduled_tokens_by_request(scheduling_state)
        in_progress = self.input_batch.in_progress_prompt_logprobs_cpu
        target_tensor_cache = getattr(self, "_verl_target_prompt_token_ids_tensors", None)
        if target_tensor_cache is None:
            target_tensor_cache = {}
            self._verl_target_prompt_token_ids_tensors = target_tensor_cache
        for cached_request_id in tuple(target_tensor_cache):
            if cached_request_id not in request_widths or cached_request_id not in self.requests:
                target_tensor_cache.pop(cached_request_id, None)
        completed_outputs = {}
        completed_request_ids = []

        for request_id in target_request_ids:
            num_tokens = num_scheduled_tokens.get(request_id)
            if num_tokens is None:
                # A preempted request will be resumed in a later engine step.
                continue

            request = self.requests[request_id]
            prompt_token_ids_list = request.prompt_token_ids
            if prompt_token_ids_list is None:
                raise RuntimeError("Target prompt logprobs do not support prompt embeddings without token ids.")
            sampling_params = request.sampling_params
            extra_args = sampling_params.extra_args or {}
            target_rows = extra_args[VERL_TARGET_PROMPT_TOKEN_IDS_KEY]
            target_tensor = target_tensor_cache.get(request_id)
            if target_tensor is None:
                target_tensor = torch.as_tensor(target_rows, dtype=torch.long).to(self.device, non_blocking=True)
                if target_tensor.ndim != 2:
                    raise ValueError(
                        "Target prompt token ids must have shape [prompt_len, topk], "
                        f"got {tuple(target_tensor.shape)} for request {request_id!r}."
                    )
                if target_tensor.shape[0] != len(prompt_token_ids_list):
                    raise ValueError(
                        "Target prompt token ids must contain one row per prompt token (the final row is a dummy), "
                        f"got target_rows={target_tensor.shape[0]}, prompt_len={len(prompt_token_ids_list)} "
                        f"for request {request_id!r}."
                    )
                expected_width = int(request_widths[request_id])
                if target_tensor.shape[1] != expected_width:
                    raise ValueError(
                        "Target prompt token-id width must equal prompt_logprobs, "
                        f"got target_width={target_tensor.shape[1]}, prompt_logprobs={expected_width} "
                        f"for request {request_id!r}."
                    )
                target_tensor_cache[request_id] = target_tensor

            prompt_token_ids = torch.as_tensor(prompt_token_ids_list, dtype=torch.long).to(
                self.device, non_blocking=True
            )
            prompt_len = len(prompt_token_ids_list)
            target_width = target_tensor.shape[-1]
            logprobs_tensors = in_progress.get(request_id)
            if not logprobs_tensors:
                logprobs_tensors = logprobs_tensors_cls.empty_cpu(prompt_len - 1, target_width + 1)
                in_progress[request_id] = logprobs_tensors

            start_index = request.num_computed_tokens
            start_token = start_index + 1
            remaining_tokens = prompt_len - start_token
            if num_tokens <= remaining_tokens:
                num_logits = num_tokens
            else:
                num_logits = remaining_tokens
                completed_request_ids.append(request_id)
                completed_outputs[request_id] = logprobs_tensors
            if num_logits <= 0:
                continue

            request_index = self.input_batch.req_id_to_index[request_id]
            offset = _query_offset(self, request_index)
            prompt_hidden_states = hidden_states[offset : offset + num_logits]
            logits = _compute_logits(self, prompt_hidden_states)
            normalized_logprobs = self.sampler.compute_logprobs(logits)
            target_ids = target_tensor[start_index : start_index + num_logits]
            sampled_ids = prompt_token_ids[start_token : start_token + num_logits]
            token_ids, logprobs, ranks = _build_target_logprob_rows(
                normalized_logprobs=normalized_logprobs,
                target_token_ids=target_ids,
                sampled_token_ids=sampled_ids,
            )

            chunk = slice(start_index, start_index + num_logits)
            logprobs_tensors.logprob_token_ids[chunk].copy_(token_ids, non_blocking=True)
            logprobs_tensors.logprobs[chunk].copy_(logprobs, non_blocking=True)
            logprobs_tensors.selected_token_ranks[chunk].copy_(ranks, non_blocking=True)

        for request_id in completed_request_ids:
            del request_widths[request_id]
            del in_progress[request_id]
            target_tensor_cache.pop(request_id, None)
        if completed_outputs:
            _sync_model_runner_device(self)
        return completed_outputs

    model_runner._get_prompt_logprobs_dict = MethodType(_get_prompt_logprobs_dict_with_targets, model_runner)
    setattr(model_runner, _PATCHED_ATTRIBUTE, True)
