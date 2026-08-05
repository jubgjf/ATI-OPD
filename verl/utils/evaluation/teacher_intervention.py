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

"""Backend-neutral teacher intervention with OpenAI-compatible HTTP adapters.

The repair policy itself intentionally remains in :class:`AgentLoopWorker`.
This module only adapts standalone vLLM HTTP responses to the same token-level
contracts used by training, then invokes the existing state machine directly.
"""

from __future__ import annotations

import asyncio
import dataclasses
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Mapping
from uuid import uuid4

import torch

from verl.experimental.agent_loop.agent_loop import (
    INTERVENTION_FULL_DISTRIBUTION_GAP_METRICS,
    AgentLoopMetrics,
    AgentLoopOutput,
    AgentLoopWorker,
    _TeacherNLLCalibrationSnapshot,
)
from verl.experimental.teacher_loop.teacher_manager import (
    TeacherGenerationResult,
    TeacherLogprobScoringResult,
    _sanitize_nonfinite_teacher_logprobs,
)
from verl.workers.config import DistillationInterventionConfig
from verl.workers.rollout.replica import TokenOutput

if TYPE_CHECKING:
    from openai import AsyncOpenAI


def object_field(value: Any, name: str, default: Any = None) -> Any:
    """Read one field from a Pydantic response, mapping, or plain object."""

    if value is None:
        return default
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _finite_logprob(value: Any, *, floor: float, context: str) -> float:
    if value is None:
        return floor
    parsed = float(value)
    if math.isnan(parsed) or parsed == float("-inf"):
        return floor
    if parsed == float("inf"):
        return 0.0
    if parsed > 1e-5:
        raise RuntimeError(f"{context} returned a positive log probability: {parsed!r}.")
    return parsed


def _token_id_from_logprob_key(value: Any) -> int:
    """Decode the exact token-ID representation requested from vLLM.

    ``return_tokens_as_token_ids=true`` makes vLLM serialize logprob keys as
    ``token_id:<integer>``. Numeric keys are accepted as well for clients that
    preserve the native response mapping. Decoded token text is deliberately
    rejected because retokenization can be ambiguous.
    """

    if isinstance(value, int):
        return value
    text = str(value)
    if text.startswith("token_id:"):
        text = text.removeprefix("token_id:")
    elif not text.lstrip("-").isdigit():
        raise RuntimeError(
            "vLLM did not return exact token IDs for prompt Top-K logprobs. "
            "The standalone repair evaluator requires return_tokens_as_token_ids support and will not "
            "retokenize decoded Top-K strings."
        )
    try:
        return int(text)
    except ValueError as exc:
        raise RuntimeError(f"Invalid token-ID logprob key: {value!r}.") from exc


def _normalize_top_logprob_row(row: Any, *, width: int, floor: float) -> tuple[list[int], list[float]]:
    if not isinstance(row, Mapping):
        raise RuntimeError(
            "vLLM prompt scoring did not return a Top-K mapping. "
            "Check that the server max_logprobs is at least the requested intervention support width."
        )
    candidates: dict[int, float] = {}
    for raw_token, raw_logprob in row.items():
        token_id = _token_id_from_logprob_key(raw_token)
        logprob_value = object_field(raw_logprob, "logprob", raw_logprob)
        logprob = _finite_logprob(logprob_value, floor=floor, context="vLLM prompt Top-K scoring")
        previous = candidates.get(token_id)
        if previous is None or logprob > previous:
            candidates[token_id] = logprob
    ordered = sorted(candidates.items(), key=lambda item: (item[1], -item[0]), reverse=True)[:width]
    ids = [token_id for token_id, _ in ordered]
    logprobs = [logprob for _, logprob in ordered]
    if len(ids) < width:
        ids.extend([0] * (width - len(ids)))
        logprobs.extend([floor] * (width - len(logprobs)))
    return ids, logprobs


def parse_echo_prompt_scoring(
    response: Any,
    *,
    sequence_ids: list[int],
    prompt_logprobs: int,
    logprob_floor: float,
) -> TokenOutput:
    """Convert a vLLM echo completion into verl's prompt-scoring contract."""

    choices = list(object_field(response, "choices", []) or [])
    if len(choices) != 1:
        raise RuntimeError(f"Expected one vLLM prompt-scoring choice, got {len(choices)}.")
    logprobs_object = object_field(choices[0], "logprobs")
    token_logprobs = list(object_field(logprobs_object, "token_logprobs", []) or [])
    top_logprobs = list(object_field(logprobs_object, "top_logprobs", []) or [])
    if len(token_logprobs) < len(sequence_ids):
        raise RuntimeError(
            "vLLM echo prompt scoring returned too few token logprobs: "
            f"sequence_len={len(sequence_ids)}, returned={len(token_logprobs)}."
        )
    if prompt_logprobs > 0 and len(top_logprobs) < len(sequence_ids):
        raise RuntimeError(
            "vLLM echo prompt scoring returned too few Top-K rows: "
            f"sequence_len={len(sequence_ids)}, returned={len(top_logprobs)}."
        )

    width = max(prompt_logprobs, 1)
    ids_rows: list[list[int]] = []
    logprob_rows: list[list[float]] = []
    sampled_ids_rows: list[list[int]] = []
    sampled_logprob_rows: list[list[float]] = []
    # Prompt position p contains log p(sequence[p] | sequence[:p]); verl row
    # p-1 predicts that token. The final row is a contract-preserving dummy.
    for row_index in range(max(len(sequence_ids) - 1, 0)):
        prompt_position = row_index + 1
        sampled_id = int(sequence_ids[prompt_position])
        sampled_logprob = _finite_logprob(
            token_logprobs[prompt_position],
            floor=logprob_floor,
            context="vLLM sampled prompt scoring",
        )
        sampled_ids_rows.append([sampled_id])
        sampled_logprob_rows.append([sampled_logprob])
        if prompt_logprobs == 0:
            ids_rows.append([sampled_id])
            logprob_rows.append([sampled_logprob])
        else:
            row_ids, row_logprobs = _normalize_top_logprob_row(
                top_logprobs[prompt_position],
                width=prompt_logprobs,
                floor=logprob_floor,
            )
            ids_rows.append(row_ids)
            logprob_rows.append(row_logprobs)

    ids_rows.append([0] * width)
    logprob_rows.append([0.0] * width)
    sampled_ids_rows.append([0])
    sampled_logprob_rows.append([0.0])
    if len(ids_rows) != len(sequence_ids):
        raise RuntimeError(
            f"Internal prompt-scoring alignment error: rows={len(ids_rows)}, sequence_len={len(sequence_ids)}."
        )
    return TokenOutput(
        token_ids=[],
        log_probs=[],
        extra_fields={
            "prompt_ids": ids_rows,
            "prompt_logprobs": logprob_rows,
            "sampled_prompt_ids": sampled_ids_rows,
            "sampled_prompt_logprobs": sampled_logprob_rows,
        },
    )


@dataclass(frozen=True)
class OpenAICompletionAdapterConfig:
    served_model_name: str
    logprob_floor: float = -10.0
    request_timeout_seconds: float = 7200.0

    def __post_init__(self) -> None:
        if not self.served_model_name:
            raise ValueError("served_model_name must be non-empty.")
        if not math.isfinite(self.logprob_floor) or self.logprob_floor > 0:
            raise ValueError(f"logprob_floor must be finite and <= 0, got {self.logprob_floor!r}.")


class OpenAICompletionAdapter:
    """Expose an OpenAI-compatible vLLM endpoint as verl's token client."""

    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        config: OpenAICompletionAdapterConfig,
        semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        self.client = client
        self.config = config
        self.semaphore = semaphore or asyncio.Semaphore(128)

    async def _create(self, **kwargs: Any) -> Any:
        async with self.semaphore:
            return await self.client.completions.create(**kwargs)

    async def _score_prompt(self, prompt_ids: list[int], prompt_logprobs: int) -> TokenOutput:
        if not prompt_ids:
            raise ValueError("Prompt scoring requires at least one token.")
        try:
            response = await self._create(
                model=self.config.served_model_name,
                prompt=prompt_ids,
                n=1,
                max_tokens=1,
                temperature=0.0,
                top_p=1.0,
                echo=True,
                logprobs=max(prompt_logprobs, 1),
                stream=False,
                extra_body={
                    "return_token_ids": True,
                    "return_tokens_as_token_ids": True,
                },
            )
        except Exception as exc:
            raise RuntimeError(
                "Exact standalone prompt scoring failed. The vLLM endpoint must support echo logprobs, "
                "return_token_ids, and return_tokens_as_token_ids; no decoded-token approximation is used."
            ) from exc
        return parse_echo_prompt_scoring(
            response,
            sequence_ids=prompt_ids,
            prompt_logprobs=prompt_logprobs,
            logprob_floor=self.config.logprob_floor,
        )

    async def _generate(self, *, prompt_ids: list[int], sampling_params: Mapping[str, Any]) -> TokenOutput:
        extra_body: dict[str, Any] = {"return_token_ids": True}
        for key in ("top_k", "seed", "min_p", "repetition_penalty", "stop_token_ids", "ignore_eos"):
            value = sampling_params.get(key)
            if value is not None:
                extra_body[key] = value
        wants_logprobs = bool(sampling_params.get("logprobs"))
        if wants_logprobs:
            extra_body["return_tokens_as_token_ids"] = True
        response = await self._create(
            model=self.config.served_model_name,
            prompt=prompt_ids,
            n=1,
            max_tokens=int(sampling_params.get("max_tokens", 1)),
            temperature=float(sampling_params.get("temperature", 1.0)),
            top_p=float(sampling_params.get("top_p", 1.0)),
            logprobs=1 if wants_logprobs else None,
            stream=False,
            extra_body=extra_body,
        )
        choices = list(object_field(response, "choices", []) or [])
        if len(choices) != 1:
            raise RuntimeError(f"Expected one vLLM completion, got {len(choices)}.")
        choice = choices[0]
        raw_token_ids = object_field(choice, "token_ids")
        if raw_token_ids is None:
            raise RuntimeError(
                "vLLM completion omitted exact generated token_ids. Standalone repair never retokenizes text."
            )
        token_ids = [int(token_id) for token_id in raw_token_ids]
        log_probs: list[float] | None = None
        if wants_logprobs:
            logprobs_object = object_field(choice, "logprobs")
            raw_values = list(object_field(logprobs_object, "token_logprobs", []) or [])
            if len(raw_values) < len(token_ids):
                raise RuntimeError(
                    "vLLM completion returned fewer sampled-token logprobs than token IDs: "
                    f"token_ids={len(token_ids)}, logprobs={len(raw_values)}."
                )
            raw_values = raw_values[-len(token_ids) :] if token_ids else []
            log_probs = [
                _finite_logprob(value, floor=self.config.logprob_floor, context="vLLM generation")
                for value in raw_values
            ]
        finish_reason_raw = object_field(choice, "finish_reason")
        finish_reason = str(finish_reason_raw).lower() if finish_reason_raw is not None else None
        stop_reason_raw = object_field(choice, "stop_reason")
        stop_reason = str(stop_reason_raw).lower() if stop_reason_raw is not None else None
        return TokenOutput(
            token_ids=token_ids,
            log_probs=log_probs,
            stop_reason=stop_reason,
            extra_fields={"finish_reason": finish_reason},
        )

    async def generate(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Any = None,
        video_data: Any = None,
        audio_data: Any = None,
        mm_processor_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> TokenOutput:
        del request_id, kwargs
        if any(value is not None for value in (image_data, video_data, audio_data)) or mm_processor_kwargs:
            raise NotImplementedError("Standalone checkpoint repair currently supports text-only trajectories.")
        prompt_logprobs = sampling_params.get("prompt_logprobs")
        if prompt_logprobs is not None:
            return await self._score_prompt(prompt_ids, int(prompt_logprobs))
        return await self._generate(prompt_ids=prompt_ids, sampling_params=sampling_params)


class OpenAITeacherManagerAdapter:
    """Teacher manager surface consumed by the shared AgentLoop repair state machine."""

    def __init__(
        self,
        *,
        client: OpenAICompletionAdapter,
        intervention_config: DistillationInterventionConfig,
        logprob_floor: float,
    ) -> None:
        self.client = client
        self.intervention_config = intervention_config
        self.logprob_floor = float(logprob_floor)

    async def compute_teacher_logprobs_single(
        self,
        sequence_ids: list[int],
        multi_modal_data: Mapping[str, Any] | None = None,
        mm_processor_kwargs: Mapping[str, Any] | None = None,
        routing_key: str | None = None,
        request_id: str | None = None,
        prompt_logprobs: int | None = None,
    ) -> TeacherLogprobScoringResult:
        del routing_key
        output = await self.client.generate(
            request_id=request_id or uuid4().hex,
            prompt_ids=sequence_ids,
            sampling_params={"max_tokens": 1, "temperature": 1.0, "prompt_logprobs": prompt_logprobs or 0},
            image_data=(multi_modal_data or {}).get("images"),
            video_data=(multi_modal_data or {}).get("videos"),
            audio_data=(multi_modal_data or {}).get("audios"),
            mm_processor_kwargs=mm_processor_kwargs,
        )
        extra = output.extra_fields
        ids = torch.tensor(extra["prompt_ids"], dtype=torch.int32)
        logprobs = torch.tensor(extra["prompt_logprobs"], dtype=torch.float32)
        sampled_ids = torch.tensor(extra["sampled_prompt_ids"], dtype=torch.int32)
        sampled_logprobs = torch.tensor(extra["sampled_prompt_logprobs"], dtype=torch.float32)
        logprobs, nonfinite_positions = _sanitize_nonfinite_teacher_logprobs(
            logprobs,
            replacement=self.logprob_floor,
        )
        sampled_logprobs, sampled_nonfinite = _sanitize_nonfinite_teacher_logprobs(
            sampled_logprobs,
            replacement=self.logprob_floor,
        )
        nonfinite_positions = sorted(set(nonfinite_positions).union(sampled_nonfinite))
        return TeacherLogprobScoringResult(
            ids=ids,
            logprobs=logprobs,
            sampled_ids=sampled_ids,
            sampled_logprobs=sampled_logprobs,
            nonfinite_count=len(nonfinite_positions),
            nonfinite_sequence_positions=nonfinite_positions,
            replacement_value=self.logprob_floor,
        )

    async def generate_teacher_segment_single(
        self,
        prefix_ids: list[int],
        max_tokens: int,
        multi_modal_data: Mapping[str, Any] | None = None,
        mm_processor_kwargs: Mapping[str, Any] | None = None,
        routing_key: str | None = None,
        request_id: str | None = None,
    ) -> TeacherGenerationResult:
        del routing_key
        output = await self.client.generate(
            request_id=request_id or uuid4().hex,
            prompt_ids=prefix_ids,
            sampling_params={
                "max_tokens": max_tokens,
                "temperature": self.intervention_config.teacher_temperature,
                "top_p": self.intervention_config.teacher_top_p,
                "top_k": self.intervention_config.teacher_top_k,
                "logprobs": None,
            },
            image_data=(multi_modal_data or {}).get("images"),
            video_data=(multi_modal_data or {}).get("videos"),
            audio_data=(multi_modal_data or {}).get("audios"),
            mm_processor_kwargs=mm_processor_kwargs,
        )
        return TeacherGenerationResult(
            token_ids=list(output.token_ids),
            stop_reason=output.stop_reason,
            finish_reason=(output.extra_fields or {}).get("finish_reason"),
        )


def intervention_config_from_mapping(config: Mapping[str, Any]) -> DistillationInterventionConfig:
    """Construct the typed intervention config while rejecting stale unknown keys."""

    config = {key: value for key, value in config.items() if key != "_target_"}
    allowed = {field.name for field in dataclasses.fields(DistillationInterventionConfig)}
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError(f"Unknown standalone intervention config fields: {unknown}.")
    return DistillationInterventionConfig(**dict(config))


def validate_http_intervention_config(config: DistillationInterventionConfig) -> None:
    metric = str(config.gap_metric).lower()
    if metric in INTERVENTION_FULL_DISTRIBUTION_GAP_METRICS:
        raise NotImplementedError(
            "Standalone HTTP repair does not support exact full-vocabulary KL. "
            "Use sampled-token or Top-K gap metrics; no approximate replacement is applied."
        )


class StandaloneTeacherInterventionRunner:
    """Run the training repair state machine against two persistent HTTP models."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        intervention_config: Mapping[str, Any] | DistillationInterventionConfig,
        max_response_length: int,
        logprob_floor: float,
        student_client: OpenAICompletionAdapter,
        teacher_client: OpenAICompletionAdapter,
        teacher_nll_calibration: Mapping[str, Any] | None = None,
        teacher_key: str = "default",
    ) -> None:
        if max_response_length <= 0:
            raise ValueError("max_response_length must be positive.")
        typed_config = (
            intervention_config
            if isinstance(intervention_config, DistillationInterventionConfig)
            else intervention_config_from_mapping(intervention_config)
        )
        validate_http_intervention_config(typed_config)
        if typed_config.outcome_aware_candidate_selection_enabled and teacher_nll_calibration is None:
            raise RuntimeError(
                "Outcome-aware standalone repair requires checkpoint evaluation_state.json teacher-NLL calibration."
            )

        worker = AgentLoopWorker.__new__(AgentLoopWorker)
        worker.tokenizer = tokenizer
        worker.processor = None
        worker.rollout_config = SimpleNamespace(response_length=int(max_response_length))
        worker.teacher_key = str(teacher_key)
        worker.teacher_intervention_enabled = bool(typed_config.enabled)
        worker.teacher_intervention_config = typed_config
        worker._teacher_intervention_logged_samples = 0
        worker._teacher_intervention_logged_step = None
        worker._teacher_nll_pending_updates = []
        worker._teacher_candidate_rpc_semaphore = None
        worker._student_suffix_nonfinite_logprob_warnings = 0
        diagnostics = SimpleNamespace(
            enabled=False,
            collect_topk_overlap=False,
            collect_rethinking_opd_metrics=False,
            collect_intervention_metrics=False,
            collect_intervention_boundary_metrics=False,
        )
        loss_settings = SimpleNamespace(use_topk=False)
        loss_config = SimpleNamespace(
            log_prob_min_clamp=float(logprob_floor),
            loss_settings=loss_settings,
            topk=1,
            top_k_strategy="only_stu",
        )
        worker.distillation_config = SimpleNamespace(
            intervention=typed_config,
            distillation_loss=loss_config,
            diagnostics=diagnostics,
        )
        worker.config = SimpleNamespace(
            algorithm=SimpleNamespace(adv_estimator=""),
            trainer=SimpleNamespace(metric_logging=SimpleNamespace(mode="full", archive=None)),
        )
        worker.llm_client = student_client
        worker.teacher_server_manager = OpenAITeacherManagerAdapter(
            client=teacher_client,
            intervention_config=typed_config,
            logprob_floor=logprob_floor,
        )
        worker._init_teacher_intervention_token_patterns()
        if teacher_nll_calibration is not None:
            worker._teacher_nll_calibration_snapshot = _TeacherNLLCalibrationSnapshot(
                threshold=(
                    None
                    if teacher_nll_calibration.get("threshold") is None
                    else float(teacher_nll_calibration["threshold"])
                ),
                source=str(teacher_nll_calibration.get("source", "unavailable")),
                count=int(teacher_nll_calibration.get("count", 0)),
                version=int(teacher_nll_calibration.get("version", 0)),
            )
        else:
            initial = typed_config.teacher_nll_initial_threshold
            worker._teacher_nll_calibration_snapshot = _TeacherNLLCalibrationSnapshot(
                threshold=None if initial is None else float(initial),
                source="unavailable" if initial is None else "initial",
                count=0,
                version=0,
            )
        self.worker = worker
        self.intervention_config = typed_config

    async def repair(
        self,
        *,
        prompt_ids: list[int],
        response_ids: list[int],
        response_logprobs: list[float],
        raw_reward_score: float,
        raw_reward_extra_info: Mapping[str, Any],
        raw_correctness_value: float,
        raw_correct: bool,
        finish_reason: str | None,
        sampling_params: Mapping[str, Any],
        request_id: str | None = None,
        sample_kwargs: Mapping[str, Any] | None = None,
    ) -> AgentLoopOutput:
        if len(response_ids) != len(response_logprobs):
            raise ValueError(
                "Standalone repair requires one sampled-token logprob per raw response token: "
                f"response_ids={len(response_ids)}, response_logprobs={len(response_logprobs)}."
            )
        output = AgentLoopOutput(
            prompt_ids=[int(value) for value in prompt_ids],
            response_ids=[int(value) for value in response_ids],
            response_mask=[1] * len(response_ids),
            response_logprobs=[float(value) for value in response_logprobs],
            reward_score=float(raw_reward_score),
            metrics=AgentLoopMetrics(),
            extra_fields={
                "rollout_request_id": request_id or uuid4().hex,
                "finish_reason": finish_reason,
                "reward_extra_info": dict(raw_reward_extra_info),
                "teacher_intervention_outcome": {
                    "enabled": True,
                    "correctness_key": self.intervention_config.outcome_correctness_key,
                    "correctness_threshold": float(self.intervention_config.outcome_correctness_threshold),
                    "raw_reward_score": float(raw_reward_score),
                    "raw_reward_extra_info": dict(raw_reward_extra_info),
                    "raw_correctness_value": float(raw_correctness_value),
                    "raw_correct": bool(raw_correct),
                    "raw_response_length": len(response_ids),
                    "raw_finish_reason": finish_reason,
                    "trigger_detected": False,
                    "repair_executed": False,
                    "oracle_gate_used": bool(self.intervention_config.outcome_aware_candidate_selection_enabled),
                    "evaluation_pair": True,
                    "transition_observation_event": True,
                },
            },
        )
        await AgentLoopWorker._maybe_repair_with_teacher(
            self.worker,
            output,
            validate=True,
            repair_for_evaluation=True,
            sampling_params=dict(sampling_params),
            sample_kwargs=dict(sample_kwargs or {}),
        )
        return output
