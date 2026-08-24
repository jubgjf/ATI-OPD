# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import asyncio
import copy
import json
import math
import os
import shlex
import time
import uuid
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pprint import pprint
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, ResourcePoolManager
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.distillation.batch_filter import to_tensordict_without_distillation_supervision
from verl.trainer.distillation.config_validation import validate_distillation_intervention_training_config
from verl.trainer.distillation.diagnostics import (
    OpdDiagnosticsTracker,
    build_future_gap_proposal_records,
    compute_entropy_heatmap_row,
    compute_gap_heatmap_row,
    compute_gap_high_heatmap_row,
    compute_intervention_boundary_diagnostics,
    compute_intervention_metrics,
    compute_opd_diagnostics_metrics,
    compute_teacher_logprob_nonfinite_heatmap_row,
    extract_response_topk_scores,
    extract_sampled_response_logprobs,
    resolve_special_token_ids,
)
from verl.trainer.distillation.losses import is_distillation_enabled, is_teacher_segment_sft_enabled
from verl.trainer.distillation.teacher_target_scoring import (
    assemble_teacher_target_logprobs,
    build_teacher_target_scoring_requests,
)
from verl.trainer.distillation.token_reward import (
    compute_only_student_teacher_topk_token_reward,
    compute_only_student_topk_token_reward,
    compute_rethinking_only_student_topk_token_reward,
)
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_text_repetition_and_truncation_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import extract_reward
from verl.trainer.ppo.utils import (
    Role,
    WorkerType,
    need_critic,
    need_reference_policy,
    need_reward_model,
    need_teacher_policy,
)
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.ray_utils import auto_await
from verl.utils.reward_score import math_dapo
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.config import DistillationConfig, EngineConfig
from verl.workers.rollout.llm_server import LLMServerManager
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding


@auto_await
async def _execute_teacher_target_scoring_requests(
    teacher_scorer,
    requests,
    *,
    request_id_prefix: str,
):
    tasks = [
        teacher_scorer.compute_teacher_logprobs_for_target_ids_single(
            sequence_ids=request.sequence_ids,
            target_ids=request.target_ids,
            multi_modal_data=request.multi_modal_data,
            mm_processor_kwargs=request.mm_processor_kwargs,
            routing_key=request.routing_key,
            request_id=f"{request_id_prefix}-{sample_index}-{uuid.uuid4().hex}",
        )
        for sample_index, request in enumerate(requests)
    ]
    return await asyncio.gather(*tasks)


def _with_reward_data_source(
    extra_infos: Any,
    *,
    count: int,
    reward_data_source: str,
) -> np.ndarray:
    """Copy per-sample metadata and override only the evaluation reward implementation."""
    if extra_infos is None:
        values = [{} for _ in range(count)]
    else:
        values = list(extra_infos)
        if len(values) != count:
            raise ValueError(f"Expected {count} extra_info entries, got {len(values)}.")

    copied = []
    for index, value in enumerate(values):
        if value is None:
            info = {}
        elif isinstance(value, dict):
            info = dict(value)
        else:
            raise TypeError(f"extra_info entry {index} must be a dict or None, got {type(value).__name__}.")
        info["reward_data_source"] = reward_data_source
        copied.append(info)
    return np.asarray(copied, dtype=object)


def _has_answer_pattern(text: Any) -> bool:
    return math_dapo.has_answer_label(text)


def _snapshot_diagnostics_value(value: Any, memo: dict[int, Any] | None = None) -> Any:
    """Snapshot diagnostics data once, preserving shared tensor references."""
    memo = {} if memo is None else memo
    value_id = id(value)
    if value_id in memo:
        return memo[value_id]
    if isinstance(value, torch.Tensor):
        detached = value.detach()
        snapshot = detached.clone() if detached.device.type == "cpu" else detached.cpu()
        memo[value_id] = snapshot
        return snapshot
    if isinstance(value, dict):
        snapshot = {key: _snapshot_diagnostics_value(item, memo) for key, item in value.items()}
        memo[value_id] = snapshot
        return snapshot
    if isinstance(value, list):
        snapshot = [_snapshot_diagnostics_value(item, memo) for item in value]
        memo[value_id] = snapshot
        return snapshot
    if isinstance(value, tuple):
        snapshot = tuple(_snapshot_diagnostics_value(item, memo) for item in value)
        memo[value_id] = snapshot
        return snapshot
    snapshot = copy.deepcopy(value)
    memo[value_id] = snapshot
    return snapshot


def _snapshot_diagnostics_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return _snapshot_diagnostics_value(kwargs)


def _snapshot_diagnostics_payloads(*payloads: dict[str, Any] | None) -> tuple[dict[str, Any] | None, ...]:
    """Snapshot related payloads with one device-to-host copy per shared tensor."""
    memo: dict[int, Any] = {}
    return tuple(None if payload is None else _snapshot_diagnostics_value(payload, memo) for payload in payloads)


@dataclass(frozen=True, slots=True)
class _OpdDiagnosticsResult:
    metrics: dict[str, float]
    intervention_records: tuple[dict[str, Any], ...] = ()
    future_gap_records: tuple[dict[str, Any], ...] = ()


def _compute_intervention_boundary_from_payloads(
    diagnostics_kwargs: dict[str, Any],
    intervention_kwargs: dict[str, Any] | None,
    boundary_kwargs: dict[str, Any] | None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    if boundary_kwargs is None or intervention_kwargs is None:
        return {}, []
    return compute_intervention_boundary_diagnostics(
        global_step=boundary_kwargs["global_step"],
        interventions=intervention_kwargs["interventions"],
        intervention_outcomes=intervention_kwargs.get("intervention_outcomes"),
        response_ids=diagnostics_kwargs["response_ids"],
        response_mask=intervention_kwargs["response_mask"],
        student_logprobs=diagnostics_kwargs["student_logprobs"],
        teacher_logprobs=diagnostics_kwargs["teacher_logprobs"],
        entropy=diagnostics_kwargs["entropy"],
        student_topk_ids=diagnostics_kwargs["student_topk_ids"],
        teacher_logprob_nonfinite_positions=diagnostics_kwargs["teacher_logprob_nonfinite_positions"],
        boundary_text_cases=boundary_kwargs.get("boundary_text_cases"),
        sample_uids=boundary_kwargs.get("sample_uids"),
        data_sources=boundary_kwargs.get("data_sources"),
        gap_metric=boundary_kwargs["gap_metric"],
        teacher_rejection_token_cap=boundary_kwargs["teacher_rejection_token_cap"],
        student_teacher_span_scores_available=bool(boundary_kwargs.get("student_teacher_span_scores_available", True)),
    )


def _compute_opd_diagnostics_background_metrics(
    diagnostics_kwargs: dict[str, Any],
    intervention_kwargs: dict[str, Any] | None,
    boundary_kwargs: dict[str, Any] | None = None,
) -> _OpdDiagnosticsResult:
    start_time = time.perf_counter()
    metrics = compute_opd_diagnostics_metrics(**diagnostics_kwargs)
    if intervention_kwargs is not None:
        metrics.update(compute_intervention_metrics(**intervention_kwargs))
    intervention_records: list[dict[str, Any]] = []
    if boundary_kwargs is not None and intervention_kwargs is not None:
        boundary_metrics, intervention_records = _compute_intervention_boundary_from_payloads(
            diagnostics_kwargs,
            intervention_kwargs,
            boundary_kwargs,
        )
        metrics.update(boundary_metrics)
    future_gap_records = build_future_gap_proposal_records(
        intervention_kwargs.get("intervention_proposals") if intervention_kwargs is not None else None
    )
    metrics["opd/diagnostics_async/worker_compute_seconds"] = time.perf_counter() - start_time
    return _OpdDiagnosticsResult(
        metrics=metrics,
        intervention_records=tuple(intervention_records),
        future_gap_records=tuple(future_gap_records),
    )


@dataclass(slots=True)
class _PendingOpdDiagnostics:
    source_step: int
    future: Future[_OpdDiagnosticsResult]
    submitted_at: float


@dataclass(frozen=True, slots=True)
class _PendingEventRecordsArchive:
    global_step: int
    metric_key: str
    records: tuple[dict[str, Any], ...]


_OPD_HEATMAP_METRIC_KEYS = frozenset(
    {
        "opd/entropy_heatmap",
        "opd/gap_abs_heatmap",
        "opd/gap_abs_high_heatmap",
        "opd/teacher_logprob_nonfinite_heatmap",
    }
)


@dataclass(frozen=True, slots=True)
class _ResolvedCheckpointContext:
    path: str | None
    step: int

    @property
    def is_resume(self) -> bool:
        """Return whether a concrete checkpoint directory was selected."""
        return self.path is not None


def _normalize_wandb_mode(mode: str | None) -> str:
    normalized = str(mode or "online").strip().lower()
    return "offline" if normalized == "dryrun" else normalized


def _query_wandb_remote_run_step(*, project: str, run_id: str, entity: str | None) -> int:
    """Return the last remote W&B history step, raising when identity cannot be verified."""
    import wandb

    api = wandb.Api()
    resolved_entity = entity or api.default_entity
    if not resolved_entity:
        raise RuntimeError("Unable to resolve the W&B entity for strict online resume.")
    remote_run = api.run(f"{resolved_entity}/{project}/{run_id}")
    remote_run_id = str(getattr(remote_run, "id", ""))
    if remote_run_id != run_id:
        raise RuntimeError(f"W&B remote run ID mismatch: requested={run_id!r}, remote={remote_run_id!r}.")
    last_step = getattr(remote_run, "lastHistoryStep", None)
    if last_step is None:
        return -1
    return int(last_step)


def _manifest_wandb_sync_file(session: dict[str, Any]) -> str | None:
    sync_file_value = session.get("wandb_sync_file")
    sync_file = os.path.abspath(sync_file_value) if sync_file_value else None
    if sync_file is None and session.get("wandb_sync_dir"):
        sync_dir = os.path.abspath(session["wandb_sync_dir"])
        if os.path.isdir(sync_dir):
            candidates = sorted(
                os.path.join(sync_dir, name)
                for name in os.listdir(sync_dir)
                if name.endswith((".wandb", ".wandb.synced"))
            )
            sync_file = candidates[-1] if candidates else None
    return sync_file


def _manifest_wandb_sync_available(session: dict[str, Any]) -> bool:
    sync_file = _manifest_wandb_sync_file(session)
    return bool(sync_file and (os.path.exists(sync_file) or os.path.exists(f"{sync_file}.synced")))


def _manifest_offline_session_has_payload(manifest: dict[str, Any], session: dict[str, Any]) -> bool:
    has_history = bool(session.get("wandb_committed_ranges"))
    recovered_by = session.get("wandb_history_recovered_by_session")
    if has_history and recovered_by:
        recovered_session = manifest.get("sessions", {}).get(recovered_by, {})
        if _manifest_wandb_sync_available(recovered_session):
            has_history = False
    session_id = session.get("session_id")
    has_staged_artifact = any(
        shard.get("state") == "STAGED_OFFLINE"
        and shard.get("wandb_delivery_session_id", shard.get("session_id")) == session_id
        for shard in manifest.get("shards", ())
    )
    return has_history or has_staged_artifact


def _unsynced_offline_sessions(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    sessions = sorted(
        manifest.get("sessions", {}).values(),
        key=lambda item: (
            int(item.get("session_order", 0)),
            str(item.get("started_at", "")),
            str(item.get("session_id", "")),
        ),
    )
    for session in sessions:
        if session.get("wandb_mode") != "offline":
            continue
        if not _manifest_offline_session_has_payload(manifest, session):
            continue
        sync_file = _manifest_wandb_sync_file(session)
        synced = bool(
            sync_file
            and ((sync_file.endswith(".synced") and os.path.exists(sync_file)) or os.path.exists(f"{sync_file}.synced"))
        )
        if not synced:
            item = dict(session)
            item["resolved_sync_file"] = sync_file
            result.append(item)
    return result


def _offline_sync_commands(
    manifest: dict[str, Any],
    *,
    entity: str | None,
    project: str,
) -> tuple[str, ...]:
    unsynced = {item["session_id"]: item for item in _unsynced_offline_sessions(manifest)}
    sessions = sorted(
        manifest.get("sessions", {}).values(),
        key=lambda item: (
            int(item.get("session_order", 0)),
            str(item.get("started_at", "")),
            str(item.get("session_id", "")),
        ),
    )
    commands = []
    prior_remote_segment = any(item.get("wandb_mode") == "online" for item in sessions)
    for session in sessions:
        if session.get("wandb_mode") != "offline":
            continue
        if not _manifest_offline_session_has_payload(manifest, session):
            continue
        pending = unsynced.get(session["session_id"])
        if pending is not None and pending.get("resolved_sync_file"):
            parts = ["wandb", "sync"]
            if prior_remote_segment:
                parts.append("--append")
            if entity:
                parts.extend(["--entity", entity])
            parts.extend(
                [
                    "--project",
                    project,
                    "--id",
                    str(manifest["logical_run_id"]),
                    str(pending["resolved_sync_file"]),
                ]
            )
            commands.append(" ".join(shlex.quote(part) for part in parts))
        prior_remote_segment = True
    return tuple(commands)


def _has_boxed_pattern(text: Any) -> bool:
    return math_dapo.has_boxed_answer(text)


def _has_accepted_answer_format(text: Any) -> bool:
    return math_dapo.has_accepted_answer_format(text)


def _is_invalid_pred(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        normalized = value.strip()
        return normalized == "" or normalized == "[INVALID]"
    return False


def _get_reward_extra_value(reward_extra_infos_dict: dict[str, list], key: str, idx: int, default: Any = None) -> Any:
    values = reward_extra_infos_dict.get(key)
    if values is None or idx >= len(values):
        return default
    return values[idx]


def _compute_validation_format_metrics(
    outputs: list[str], reward_extra_infos_dict: dict[str, list]
) -> dict[str, float]:
    if not outputs:
        return {
            "format/answer_pattern_rate": 0.0,
            "format/boxed_pattern_rate": 0.0,
            "format/accepted_answer_format_rate": 0.0,
            "format/invalid_pred_rate": 0.0,
        }

    metrics = {
        "format/answer_pattern_rate": float(np.mean([_has_answer_pattern(output) for output in outputs])),
        "format/boxed_pattern_rate": float(np.mean([_has_boxed_pattern(output) for output in outputs])),
        "format/accepted_answer_format_rate": float(
            np.mean([_has_accepted_answer_format(output) for output in outputs])
        ),
    }
    preds = reward_extra_infos_dict.get("pred")
    if preds is None:
        metrics["format/invalid_pred_rate"] = float("nan")
    else:
        metrics["format/invalid_pred_rate"] = float(np.mean([_is_invalid_pred(pred) for pred in preds]))
    return metrics


def _as_metric_values(values: Any) -> list[Any]:
    if values is None:
        return []
    if isinstance(values, np.ndarray):
        return values.tolist()
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().tolist()
    if isinstance(values, list):
        return values
    if isinstance(values, tuple):
        return list(values)
    return [values]


def _to_bool_metric(value: Any) -> bool:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _mean_bool_metric(values: Any) -> float | None:
    metric_values = _as_metric_values(values)
    if not metric_values:
        return None
    return float(np.mean([_to_bool_metric(value) for value in metric_values]))


def _compute_reward_format_metrics(reward_extra_infos_dict: dict[str, Any]) -> dict[str, float]:
    """Aggregate train-time reward parser diagnostics for wandb/console logging."""
    metrics = {}
    key_to_metric = {
        "has_answer_pattern": "format/answer_pattern_rate",
        "has_boxed_pattern": "format/boxed_pattern_rate",
        "has_accepted_answer_format": "format/accepted_answer_format_rate",
        "invalid_pred": "format/invalid_pred_rate",
    }
    for key, metric_name in key_to_metric.items():
        value = _mean_bool_metric(reward_extra_infos_dict.get(key))
        if value is not None:
            metrics[metric_name] = value

    if "format/invalid_pred_rate" not in metrics and "pred" in reward_extra_infos_dict:
        preds = _as_metric_values(reward_extra_infos_dict.get("pred"))
        if preds:
            metrics["format/invalid_pred_rate"] = float(np.mean([_is_invalid_pred(pred) for pred in preds]))
    return metrics


def _build_validation_sample_records(
    inputs: list[str],
    outputs: list[str],
    gts: list[Any],
    scores: list[float],
    reward_extra_infos_dict: dict[str, list],
    response_lengths: list[int] | None = None,
    finish_reasons: list[Any] | None = None,
) -> list[dict[str, Any]]:
    records = []
    for idx, (input_text, output_text, gt, score) in enumerate(zip(inputs, outputs, gts, scores, strict=True)):
        record = {
            "input": input_text,
            "output": output_text,
            "ground_truth": gt,
            "score": score,
            "acc": _get_reward_extra_value(reward_extra_infos_dict, "acc", idx),
            "pred": _get_reward_extra_value(reward_extra_infos_dict, "pred", idx),
            "has_answer_pattern": _has_answer_pattern(output_text),
            "has_boxed_pattern": _has_boxed_pattern(output_text),
            "has_accepted_answer_format": _has_accepted_answer_format(output_text),
        }
        if response_lengths is not None:
            record["response_token_count"] = response_lengths[idx] if idx < len(response_lengths) else None
        if finish_reasons is not None:
            record["finish_reason"] = finish_reasons[idx] if idx < len(finish_reasons) else None
        records.append(record)
    return records


def _build_generation_archive_records(
    *,
    evaluation_scope: str,
    global_step: int,
    inputs: list[str],
    outputs: list[str],
    gts: list[Any],
    scores: list[float],
    reward_extra_infos_dict: dict[str, list],
    response_lengths: list[int] | None = None,
    finish_reasons: list[Any] | None = None,
    data_sources: list[Any] | None = None,
    sample_uids: list[Any] | None = None,
) -> list[dict[str, Any]]:
    """Build complete per-sample text records for the diagnostics Artifact."""

    sample_count = len(inputs)
    required_lengths = {
        "outputs": len(outputs),
        "ground_truths": len(gts),
        "scores": len(scores),
    }
    mismatched = {name: length for name, length in required_lengths.items() if length != sample_count}
    if mismatched:
        raise ValueError(
            "Generation archive fields must have identical sample counts: "
            f"inputs={sample_count}, mismatched={mismatched}."
        )

    records = []
    for sample_index in range(sample_count):
        record = {
            "evaluation_scope": evaluation_scope,
            "global_step": int(global_step),
            "sample_index": sample_index,
            "input": inputs[sample_index],
            "output": outputs[sample_index],
            "ground_truth": gts[sample_index],
            "score": scores[sample_index],
        }
        if response_lengths is not None and sample_index < len(response_lengths):
            record["response_token_count"] = response_lengths[sample_index]
        if finish_reasons is not None and sample_index < len(finish_reasons):
            record["finish_reason"] = finish_reasons[sample_index]
        if data_sources is not None and sample_index < len(data_sources):
            record["data_source"] = data_sources[sample_index]
        if sample_uids is not None and sample_index < len(sample_uids):
            record["uid"] = sample_uids[sample_index]
        for key, values in reward_extra_infos_dict.items():
            if key not in record and len(values) == sample_count:
                record[key] = values[sample_index]
        records.append(record)
    return records


def _as_sample_values(values: Any, *, sample_count: int, field_name: str) -> list[Any]:
    if isinstance(values, np.ndarray):
        result = values.tolist()
    elif isinstance(values, torch.Tensor):
        result = values.detach().cpu().tolist()
    elif isinstance(values, list | tuple):
        result = list(values)
    else:
        result = [values]
    if not isinstance(result, list):
        result = [result]
    if len(result) != sample_count:
        raise ValueError(
            f"Evaluation reward field {field_name!r} must contain one value per sample: "
            f"expected={sample_count}, got={len(result)}."
        )
    return result


def _extend_reward_extra_infos(
    destination: dict[str, list],
    *,
    scores: list[float],
    per_sample_infos: list[dict[str, Any]],
) -> None:
    """Append a batch of aligned reward records while preserving missing keys."""
    if len(scores) != len(per_sample_infos):
        raise ValueError(
            "Evaluation reward scores and per-sample extra info must align: "
            f"scores={len(scores)}, extra_infos={len(per_sample_infos)}."
        )
    prior_count = len(destination.get("reward", []))
    new_keys = {key for info in per_sample_infos for key in info} - {"reward"}
    for key in new_keys:
        destination.setdefault(key, [None] * prior_count)
    destination.setdefault("reward", []).extend(scores)
    known_keys = set(destination) - {"reward"}
    for info in per_sample_infos:
        for key in known_keys:
            destination[key].append(info.get(key))


_VALIDATION_TOOL_METRIC_KEYS = (
    "tool_call_counts",
    "tool_call_attempts",
    "tool_call/attempt_rate",
    "tool_call/parse_success_rate",
    "tool_call/parse_failure_rate",
    "tool_call/execution_success_rate",
    "tool_call/protocol_error_rate",
    "tool_call/timeout_rate",
    "tool_call/runtime_error_rate",
    "tool_call/http_error_rate",
    "tool_call/syntax_error_rate",
    "tool_call/markdown_extraction_rate",
    "tool_call/student_call_rate",
    "tool_call/teacher_call_rate",
    "tool_call/student_calls",
    "tool_call/teacher_calls",
    "tool_call/calls_per_trajectory",
    "tool_call/observation_tokens",
    "tool_call/final_answer_after_tool",
    "tool_call/ended_after_observation",
)


def _extend_validation_tool_metric_values(
    destination: dict[str, list[float]],
    non_tensor_batch: dict[str, Any],
) -> None:
    """Collect numeric per-trajectory tool metrics from a generated validation batch."""
    for key in _VALIDATION_TOOL_METRIC_KEYS:
        raw_values = non_tensor_batch.get(key)
        if raw_values is None:
            continue
        values = np.asarray(raw_values, dtype=object).reshape(-1)
        for raw_value in values:
            if raw_value is None:
                continue
            value = float(raw_value)
            if math.isfinite(value):
                destination.setdefault(key, []).append(value)


def _summarize_validation_tool_metrics(
    values: dict[str, list[float]],
    *,
    metric_prefix: str,
) -> dict[str, float]:
    """Summarize raw tool usage under the active validation sampling profile."""
    metrics: dict[str, float] = {}
    for key, key_values in values.items():
        if not key_values:
            continue
        value_array = np.asarray(key_values, dtype=np.float64)
        if key in {"tool_call_counts", "tool_call_attempts"}:
            metrics[f"{metric_prefix}-tool/{key}/min"] = float(value_array.min())
            metrics[f"{metric_prefix}-tool/{key}/max"] = float(value_array.max())
            metrics[f"{metric_prefix}-tool/{key}/mean"] = float(value_array.mean())
        else:
            metrics[f"{metric_prefix}-tool/{key}"] = float(value_array.mean())
    return metrics


def _reward_info_dict_to_samples(reward_extra_info: dict[str, Any], *, sample_count: int) -> list[dict[str, Any]]:
    result = [{} for _ in range(sample_count)]
    for key, values in reward_extra_info.items():
        normalized = _as_sample_values(values, sample_count=sample_count, field_name=key)
        for index, value in enumerate(normalized):
            result[index][key] = value
    return result


def _without_boundary_private_fields(value: Any) -> Any:
    """Remove transient numeric window buffers before unrelated archives are serialized."""
    if isinstance(value, dict):
        return {
            key: _without_boundary_private_fields(item)
            for key, item in value.items()
            if not str(key).startswith("_local_effect_v2_")
        }
    if isinstance(value, list | tuple):
        return [_without_boundary_private_fields(item) for item in value]
    return copy.deepcopy(value)


def _build_training_intervention_outcome_archive_records(
    *,
    global_step: int,
    outcomes: list[Any],
    interventions: list[Any],
    proposals: list[Any],
    skipped_interventions: list[Any],
    cases: list[Any],
    sample_uids: list[Any],
    data_sources: list[Any],
    raw_prompts: list[Any],
    reward_models: list[Any],
) -> list[dict[str, Any]]:
    """Build all-sample compact train records with bounded full-text cases."""
    sample_count = len(outcomes)
    fields = {
        "interventions": interventions,
        "proposals": proposals,
        "skipped_interventions": skipped_interventions,
        "cases": cases,
        "sample_uids": sample_uids,
        "data_sources": data_sources,
        "raw_prompts": raw_prompts,
        "reward_models": reward_models,
    }
    mismatched = {name: len(values) for name, values in fields.items() if len(values) != sample_count}
    if mismatched:
        raise ValueError(
            "Training intervention archive fields must have identical sample counts: "
            f"outcomes={sample_count}, mismatched={mismatched}."
        )

    records = []
    for index, raw_outcome in enumerate(outcomes):
        if not isinstance(raw_outcome, dict) or not raw_outcome.get("enabled"):
            continue
        outcome = copy.deepcopy(raw_outcome)
        outcome.pop("raw_response", None)
        outcome.pop("_raw_response_ids", None)
        record = {
            "global_step": int(global_step),
            "sample_index": index,
            "uid": sample_uids[index],
            "data_source": data_sources[index],
            "outcome": outcome,
            "interventions": _without_boundary_private_fields(interventions[index]),
            "future_gap_proposals": build_future_gap_proposal_records([proposals[index]]),
            "skipped_interventions": skipped_interventions[index],
            "case_included": isinstance(cases[index], dict),
        }
        if isinstance(cases[index], dict):
            reward_model = reward_models[index]
            ground_truth = reward_model.get("ground_truth") if isinstance(reward_model, dict) else None
            record.update(
                {
                    "raw_prompt": raw_prompts[index],
                    "ground_truth": ground_truth,
                    **dict(cases[index]),
                }
            )
        records.append(record)
    return records


def _aggregate_generation_timing(timings: list[dict[str, Any]]) -> dict[str, float]:
    merged: dict[str, float] = {}
    keys = sorted({key for timing in timings for key in timing})
    for key in keys:
        values = [timing[key] for timing in timings if key in timing]
        if not values:
            continue
        if key.endswith("/min"):
            merged[key] = float(np.min(values))
        elif key.endswith("/max"):
            merged[key] = float(np.max(values))
        else:
            merged[key] = float(np.mean(values))
    return merged


class _TeacherValidationGenerationManager:
    """Route teacher validation generation to one or more existing teacher LLM clients."""

    def __init__(
        self,
        *,
        config,
        agent_loop_manager_cls,
        teacher_clients: dict[str, Any],
        teacher_key: str,
        reward_loop_worker_handles: list[Any] | None = None,
    ):
        if not teacher_clients:
            raise ValueError("teacher validation requires at least one teacher LLM client")

        self.teacher_key = teacher_key
        self.teacher_clients = teacher_clients
        self.manager_by_teacher_key = {
            teacher_name: agent_loop_manager_cls.create(
                config=config,
                llm_client=teacher_client,
                teacher_client=teacher_clients,
                reward_loop_worker_handles=reward_loop_worker_handles,
            )
            for teacher_name, teacher_client in teacher_clients.items()
        }

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Generate validation responses with teacher clients, preserving the input sample order."""
        if len(self.manager_by_teacher_key) == 1:
            manager = next(iter(self.manager_by_teacher_key.values()))
            return manager.generate_sequences(prompts)

        if self.teacher_key not in prompts.non_tensor_batch:
            raise ValueError(
                f"teacher validation has multiple teacher clients but validation samples do not contain "
                f"routing key {self.teacher_key!r} (configured by distillation.teacher_key)"
            )

        route_values = prompts.non_tensor_batch[self.teacher_key]
        ordered_indices: list[int] = []
        outputs = []
        timings = []
        for teacher_name, manager in self.manager_by_teacher_key.items():
            idxs = [
                idx
                for idx, value in enumerate(route_values)
                if str(value.item() if hasattr(value, "item") else value) == teacher_name
            ]
            if not idxs:
                continue
            ordered_indices.extend(idxs)
            output = manager.generate_sequences(prompts[idxs])
            timing = output.meta_info.pop("timing", None)
            if timing is not None:
                timings.append(timing)
            outputs.append(output)

        missing_routes = sorted(
            {
                str(value.item() if hasattr(value, "item") else value)
                for value in route_values
                if str(value.item() if hasattr(value, "item") else value) not in self.manager_by_teacher_key
            }
        )
        if missing_routes:
            raise ValueError(
                f"teacher validation routing key {self.teacher_key!r} contains unknown teacher values "
                f"{missing_routes}; available teachers are {sorted(self.manager_by_teacher_key)}"
            )
        if not outputs:
            raise ValueError("teacher validation received an empty routed batch")

        output = DataProto.concat(outputs)
        restore_indices = torch.tensor(np.argsort(np.array(ordered_indices)), dtype=torch.long)
        output.reorder(restore_indices)
        if timings:
            output.meta_info["timing"] = _aggregate_generation_timing(timings)
        return output


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def _get_config_value(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _get_horizon_config_value(horizon_config: Any, key: str, default: Any = None) -> Any:
    return _get_config_value(horizon_config, key, default)


def compute_distillation_horizon_tokens(horizon_config: Any, global_steps: int) -> Optional[int]:
    """Return the active OPD supervision horizon for TOPD/POPD masking."""
    mode = str(_get_horizon_config_value(horizon_config, "mode", "full")).lower()
    if mode == "full":
        return None
    if mode == "fixed":
        return int(_get_horizon_config_value(horizon_config, "tokens"))
    if mode == "progressive":
        start_tokens = int(_get_horizon_config_value(horizon_config, "start_tokens"))
        end_tokens = int(_get_horizon_config_value(horizon_config, "end_tokens"))
        increment_tokens = int(_get_horizon_config_value(horizon_config, "increment_tokens"))
        interval_steps = int(_get_horizon_config_value(horizon_config, "interval_steps"))
        increments = max(int(global_steps), 0) // interval_steps
        return min(end_tokens, start_tokens + increments * increment_tokens)
    raise ValueError(
        f"distillation.distillation_loss.horizon.mode must be one of 'full', 'fixed', or 'progressive', got {mode!r}."
    )


def build_distillation_prefix_mask(response_mask: torch.Tensor, horizon_tokens: int) -> torch.Tensor:
    """Build an OPD-only prefix mask without changing the task reward response_mask."""
    response_mask_bool = response_mask.bool()
    valid_positions = response_mask_bool.long().cumsum(dim=-1)
    prefix_mask = response_mask_bool & (valid_positions <= horizon_tokens)
    return prefix_mask.to(dtype=response_mask.dtype)


def apply_distillation_horizon_mask(
    data: DataProto,
    horizon_config: Any,
    global_steps: int,
) -> dict[str, float]:
    """Attach distillation_loss_mask to the batch when TOPD/POPD prefix masking is enabled."""
    # The trainer's logged global step is 1-based inside the training loop. Keep the
    # POPD schedule 0-based so the first optimizer step uses start_tokens.
    schedule_step = max(int(global_steps) - 1, 0)
    horizon_tokens = compute_distillation_horizon_tokens(horizon_config, schedule_step)
    if horizon_tokens is None:
        return {}

    response_mask = data.batch["response_mask"]
    loss_mask = build_distillation_prefix_mask(response_mask=response_mask, horizon_tokens=horizon_tokens)
    data.batch["distillation_loss_mask"] = loss_mask

    response_mask_bool = response_mask.bool()
    loss_mask_bool = loss_mask.bool()
    response_tokens = response_mask_bool.sum(dim=-1)
    loss_tokens = loss_mask_bool.sum(dim=-1)
    total_response_tokens = response_tokens.sum().item()
    total_loss_tokens = loss_tokens.sum().item()
    token_fraction = total_loss_tokens / total_response_tokens if total_response_tokens > 0 else 0.0
    clipped_sample_fraction = ((response_tokens > horizon_tokens) & (response_tokens > 0)).float().mean().item()

    return {
        "distillation/horizon_tokens": float(horizon_tokens),
        "distillation/mask_token_fraction": float(token_fraction),
        "distillation/mask_tokens/mean": loss_tokens.float().mean().item(),
        "distillation/mask_clipped_sample_fraction": float(clipped_sample_fraction),
    }


def apply_intervention_prune_mask(
    data: DataProto,
    intervention_config: Any,
) -> dict[str, float]:
    """Limit distillation supervision to the first repaired prefix and teacher span."""
    if not bool(_get_config_value(intervention_config, "prune", False)):
        return {}
    if not bool(_get_config_value(intervention_config, "enabled", False)):
        raise ValueError("distillation.intervention.prune=true requires intervention.enabled=true.")
    if "teacher_interventions" not in data.non_tensor_batch:
        raise RuntimeError(
            "DMPD prune requires teacher_interventions metadata for every rollout sample, but the batch has no "
            "teacher_interventions field."
        )
    if "response_mask" not in data.batch or "attention_mask" not in data.batch:
        raise RuntimeError("DMPD prune requires response_mask and attention_mask tensors.")

    response_mask = data.batch["response_mask"]
    if response_mask.ndim != 2:
        raise ValueError(f"DMPD prune expects response_mask with shape [B,T], got {tuple(response_mask.shape)}.")
    batch_size, response_width = response_mask.shape
    attention_mask = data.batch["attention_mask"]
    if attention_mask.ndim != 2 or attention_mask.shape[0] != batch_size or attention_mask.shape[1] < response_width:
        raise ValueError(
            "DMPD prune expects attention_mask to contain the response suffix, but got "
            f"response_mask.shape={tuple(response_mask.shape)} and attention_mask.shape={tuple(attention_mask.shape)}."
        )

    raw_interventions = data.non_tensor_batch["teacher_interventions"]
    if raw_interventions is None:
        raise RuntimeError("DMPD prune found a null teacher_interventions batch field; expected one list per sample.")
    if isinstance(raw_interventions, np.ndarray):
        sample_interventions = raw_interventions.tolist()
    else:
        sample_interventions = list(raw_interventions)
    if len(sample_interventions) != batch_size:
        raise RuntimeError(
            "DMPD prune requires one teacher_interventions entry per sample, "
            f"got {len(sample_interventions)} entries for batch size {batch_size}."
        )

    response_mask_bool = response_mask.bool()
    base_loss_mask = data.batch.get("distillation_loss_mask", response_mask)
    if base_loss_mask.shape != response_mask.shape:
        raise ValueError(
            "DMPD prune requires distillation_loss_mask to be response-aligned, "
            f"got {tuple(base_loss_mask.shape)} and {tuple(response_mask.shape)}."
        )
    base_loss_mask_bool = base_loss_mask.bool()
    if (base_loss_mask_bool & ~response_mask_bool).any().item():
        raise ValueError("distillation_loss_mask must be a subset of response_mask before applying DMPD prune.")

    response_attention_mask = attention_mask[:, -response_width:].to(device=response_mask.device).bool()
    valid_response_lengths = response_attention_mask.sum(dim=-1)
    cutoffs = torch.full((batch_size,), response_width, dtype=torch.long, device=response_mask.device)
    repaired_samples = torch.zeros(batch_size, dtype=torch.bool, device=response_mask.device)
    teacher_token_positions = torch.zeros_like(response_mask_bool)

    for sample_idx, raw_sample_records in enumerate(sample_interventions):
        if raw_sample_records is None:
            raise RuntimeError(
                f"DMPD prune found null teacher_interventions metadata for sample {sample_idx}; use [] when no "
                "intervention succeeded."
            )
        elif isinstance(raw_sample_records, dict):
            records = [raw_sample_records]
        elif isinstance(raw_sample_records, np.ndarray):
            records = raw_sample_records.tolist()
        elif isinstance(raw_sample_records, list | tuple):
            records = list(raw_sample_records)
        else:
            raise RuntimeError(
                "DMPD prune teacher_interventions entries must be lists of records, "
                f"but sample {sample_idx} has {type(raw_sample_records).__name__}."
            )
        if any(not isinstance(record, dict) for record in records):
            raise RuntimeError(f"DMPD prune found a non-dict intervention record for sample {sample_idx}.")
        if not records:
            continue

        valid_response_length = int(valid_response_lengths[sample_idx].item())
        for record_idx, record in enumerate(records):
            start = record.get("start")
            teacher_tokens = record.get("teacher_tokens")
            if isinstance(start, bool) or not isinstance(start, int | np.integer) or int(start) < 0:
                raise RuntimeError(
                    "DMPD prune requires a non-negative integer start for "
                    f"sample {sample_idx} intervention {record_idx}, got {start!r}."
                )
            if (
                isinstance(teacher_tokens, bool)
                or not isinstance(teacher_tokens, int | np.integer)
                or int(teacher_tokens) <= 0
            ):
                raise RuntimeError(
                    "DMPD prune requires a positive integer teacher_tokens value for "
                    f"sample {sample_idx} intervention {record_idx}, got {teacher_tokens!r}."
                )
            teacher_end = int(start) + int(teacher_tokens)
            if teacher_end > valid_response_length:
                raise RuntimeError(
                    "DMPD prune teacher span exceeds the non-padding response length for "
                    f"sample {sample_idx} intervention {record_idx}: end={teacher_end}, "
                    f"response_length={valid_response_length}."
                )
            teacher_token_positions[sample_idx, int(start) : teacher_end] = True
            if record_idx == 0:
                cutoffs[sample_idx] = teacher_end
        repaired_samples[sample_idx] = True

    positions = torch.arange(response_width, device=response_mask.device).unsqueeze(0)
    positional_mask = positions < cutoffs.unsqueeze(1)
    loss_mask_bool = base_loss_mask_bool & positional_mask
    data.batch["distillation_loss_mask"] = loss_mask_bool.to(dtype=response_mask.dtype)

    teacher_segment_mask = data.batch.get("teacher_segment_mask")
    if teacher_segment_mask is not None:
        if teacher_segment_mask.shape != response_mask.shape:
            raise ValueError(
                "DMPD prune requires teacher_segment_mask to be response-aligned, "
                f"got {tuple(teacher_segment_mask.shape)} and {tuple(response_mask.shape)}."
            )
        data.batch["teacher_segment_mask"] = (teacher_segment_mask.bool() & positional_mask).to(
            dtype=teacher_segment_mask.dtype
        )

    total_response_tokens = int(response_mask_bool.sum().item())
    total_loss_tokens = int(loss_mask_bool.sum().item())
    repaired_count = int(repaired_samples.sum().item())
    excluded_student_tokens = (response_mask_bool & ~positional_mask & ~teacher_token_positions).sum(dim=-1).float()
    if repaired_count > 0:
        repaired_cutoffs = cutoffs[repaired_samples].float()
        repaired_lengths = valid_response_lengths[repaired_samples].float()
        cutoff_position_mean = repaired_cutoffs.mean().item()
        cutoff_position_frac_mean = (repaired_cutoffs / repaired_lengths.clamp_min(1.0)).mean().item()
        excluded_tokens_mean = excluded_student_tokens[repaired_samples].mean().item()
    else:
        cutoff_position_mean = 0.0
        cutoff_position_frac_mean = 0.0
        excluded_tokens_mean = 0.0

    return {
        "opd_intervention/prune/opd_token_fraction": (
            float(total_loss_tokens) / float(total_response_tokens) if total_response_tokens > 0 else 0.0
        ),
        "opd_intervention/prune/excluded_student_tokens_mean_among_repaired": float(excluded_tokens_mean),
        "opd_intervention/prune/cutoff_position_mean": float(cutoff_position_mean),
        "opd_intervention/prune/cutoff_position_frac_mean": float(cutoff_position_frac_mean),
        "opd_intervention/prune/empty_opd_sample_fraction": float(
            (loss_mask_bool.sum(dim=-1) == 0).float().mean().item()
        ),
    }


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]
        # GDPO: pass raw data for per-dimension reward extraction
        if adv_estimator in (AdvantageEstimator.GDPO, "gdpo"):
            adv_kwargs["non_tensor_batch"] = data.non_tensor_batch
            adv_kwargs["batch"] = data.batch
        # Add sum_pi_squared for Optimal Token Baseline
        if adv_estimator in (AdvantageEstimator.OPTIMAL_TOKEN_BASELINE, AdvantageEstimator.TIR_OPTIMAL_TOKEN_BASELINE):
            # Check if sum_pi_squared is available
            assert "sum_pi_squared" in data.batch, (
                "Step-dependent optimal baseline requires sum_pi_squared from actor. "
                "Please set actor.calculate_sum_pi_squared=True in config."
            )
            adv_kwargs["sum_pi_squared"] = data.batch["sum_pi_squared"]
            # old_log_probs needed for path-variance proxy: w_t = 1 - 2*exp(old_log_probs) + sum_pi_squared
            adv_kwargs["old_log_probs"] = data.batch["old_log_probs"]
            # Get pre-computed rollout IS weights if available
            rollout_is_weights = data.batch.get("rollout_is_weights", None)
            adv_kwargs["rollout_is_weights"] = rollout_is_weights

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


def build_joint_rlvr_training_tensors(
    *,
    task_reward_tensor: torch.Tensor,
    response_mask: torch.Tensor,
    intervention_outcomes: Optional[list[Any]] = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Build raw-reward RLVR inputs without crediting teacher-repaired trajectories.

    When intervention outcomes are present, raw student rewards are used for
    GRPO group statistics. Samples whose response was repaired remain in those
    statistics but receive an all-zero RLVR response mask, because their raw
    reward no longer aligns with the repaired token sequence optimized by the
    actor.
    """
    if task_reward_tensor.ndim != 2 or response_mask.ndim != 2:
        raise ValueError(
            "Joint RLVR expects task_reward_tensor and response_mask with shape [B,T], "
            f"got {tuple(task_reward_tensor.shape)} and {tuple(response_mask.shape)}."
        )
    if task_reward_tensor.shape != response_mask.shape:
        raise ValueError(
            "Joint RLVR task rewards must align with response_mask, "
            f"got {tuple(task_reward_tensor.shape)} and {tuple(response_mask.shape)}."
        )

    response_mask_bool = response_mask.bool()
    # Reward managers may place the scalar outcome on a teacher-generated
    # terminal token, which is intentionally absent from response_mask.  The
    # task-reward tensor is already response-aligned, so sum it without the
    # policy mask when recovering the per-response outcome.
    final_scores = task_reward_tensor.sum(dim=-1)
    if intervention_outcomes is None:
        active_samples = response_mask_bool.any(dim=-1)
        metrics = {
            "distillation/joint_rlvr/raw_reward_mean": final_scores.float().mean().item(),
            "distillation/joint_rlvr/final_reward_mean": final_scores.float().mean().item(),
            "distillation/joint_rlvr/repair_reward_delta_mean": 0.0,
            "distillation/joint_rlvr/active_sample_fraction": active_samples.float().mean().item(),
            "distillation/joint_rlvr/intervention_excluded_fraction": 0.0,
        }
        return task_reward_tensor, response_mask, metrics

    if len(intervention_outcomes) != response_mask.shape[0]:
        raise RuntimeError(
            "Joint RLVR intervention metadata must contain one item per response, "
            f"got {len(intervention_outcomes)} items for batch size {response_mask.shape[0]}."
        )

    raw_scores: list[float] = []
    repaired: list[bool] = []
    for sample_idx, outcome in enumerate(intervention_outcomes):
        if not isinstance(outcome, dict) or "raw_reward_score" not in outcome:
            raise RuntimeError(
                "Joint token-reward OPD + RLVR requires teacher_intervention_outcome.raw_reward_score "
                f"for every DMPD sample, but sample {sample_idx} is missing it."
            )
        raw_score = outcome["raw_reward_score"]
        if raw_score is None or not math.isfinite(float(raw_score)):
            raise RuntimeError(
                "Joint token-reward OPD + RLVR requires finite raw_reward_score values, "
                f"got {raw_score!r} for sample {sample_idx}."
            )
        raw_scores.append(float(raw_score))
        repaired.append(bool(outcome.get("repair_executed", False)))

    raw_scores_tensor = task_reward_tensor.new_tensor(raw_scores)
    repaired_tensor = torch.tensor(repaired, dtype=torch.bool, device=response_mask.device)
    has_trainable_response = response_mask_bool.any(dim=-1)
    active_samples = ~repaired_tensor & has_trainable_response
    rlvr_response_mask = response_mask * active_samples.unsqueeze(-1).to(response_mask.dtype)

    # GRPO consumes only the per-response reward sum. Place the raw score on
    # the final valid response position so the tensor remains response-aligned.
    rlvr_token_rewards = torch.zeros_like(task_reward_tensor)
    if response_mask.shape[1] == 0 and raw_scores_tensor.numel() > 0:
        raise RuntimeError("Joint RLVR cannot represent response rewards when the response width is zero.")
    if response_mask_bool.any():
        positions = torch.arange(response_mask.shape[1], device=response_mask.device).unsqueeze(0)
        last_positions = positions.masked_fill(~response_mask_bool, -1).max(dim=-1).values
        valid_samples = last_positions >= 0
        valid_rows = torch.arange(response_mask.shape[0], device=response_mask.device)[valid_samples]
        rlvr_token_rewards[valid_rows, last_positions[valid_samples]] = raw_scores_tensor[valid_samples]
        invalid_rows = torch.arange(response_mask.shape[0], device=response_mask.device)[~valid_samples]
        rlvr_token_rewards[invalid_rows, -1] = raw_scores_tensor[~valid_samples]
    elif raw_scores_tensor.numel() > 0:
        # Keep empty responses in the GRPO group statistics while their zero
        # policy mask prevents any token-level gradient.
        rlvr_token_rewards[:, -1] = raw_scores_tensor

    metrics = {
        "distillation/joint_rlvr/raw_reward_mean": raw_scores_tensor.float().mean().item(),
        "distillation/joint_rlvr/final_reward_mean": final_scores.float().mean().item(),
        "distillation/joint_rlvr/repair_reward_delta_mean": (final_scores - raw_scores_tensor).float().mean().item(),
        "distillation/joint_rlvr/active_sample_fraction": active_samples.float().mean().item(),
        "distillation/joint_rlvr/intervention_excluded_fraction": repaired_tensor.float().mean().item(),
    }
    return rlvr_token_rewards, rlvr_response_mask, metrics


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        test_benchmark_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        from verl.trainer.ppo.metric_archive import MetricLoggingSettings

        self._metric_logging_settings = MetricLoggingSettings.from_mapping(
            self.config.trainer.get("metric_logging", None)
        )
        if self._metric_logging_settings.mode == "balanced":
            checkpoint_config = self.config.actor_rollout_ref.actor.checkpoint
            if bool(checkpoint_config.get("async_save", False)):
                raise ValueError(
                    "Balanced metric logging currently requires "
                    "actor_rollout_ref.actor.checkpoint.async_save=false so checkpoint payload, "
                    "metric shard, commit marker, and tracker can be committed in order."
                )
        validate_distillation_intervention_training_config(self.config)

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)
        self.use_teacher_policy = need_teacher_policy(self.config)

        self.use_rm = need_reward_model(self.config)

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.distillation_config: DistillationConfig | None = None
        cumulative_generation_tables = self._metric_logging_settings.mode != "balanced"
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            cumulative_wandb_table=cumulative_generation_tables,
        )
        self.test_benchmark_generations_loggers = {
            profile_name: ValidationGenerationsLogger(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                table_name=f"test/{profile_name}/generations",
                cumulative_wandb_table=cumulative_generation_tables,
            )
            for profile_name in ("legacy", "train_sampling")
        }
        self.teacher_validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            table_name="teacher-val/generations",
        )
        self.teacher_validation_manager = None
        self.agent_loop_manager_cls = None
        self.reward_loop_worker_handles = None
        self.opd_entropy_tracker: OpdDiagnosticsTracker | None = None
        self.opd_gap_tracker: OpdDiagnosticsTracker | None = None
        self.opd_gap_high_tracker: OpdDiagnosticsTracker | None = None
        self.opd_teacher_logprob_nonfinite_tracker: OpdDiagnosticsTracker | None = None
        self.opd_special_token_ids = resolve_special_token_ids(self.tokenizer)
        self._opd_diagnostics_executor: ThreadPoolExecutor | None = None
        self._opd_diagnostics_task: _PendingOpdDiagnostics | None = None

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)

        self._create_dataloader(
            train_dataset,
            val_dataset,
            collate_fn,
            train_sampler,
            test_benchmark_dataset=test_benchmark_dataset,
        )

        self.checkpoint_manager = None
        self._resolved_checkpoint_context: _ResolvedCheckpointContext | None = None
        self._logical_run_id: str | None = None
        self._metric_archive_writer = None
        self._metric_router = None
        self._metric_archive_session_id: str | None = None
        self._metric_resume_step: int | None = None
        self._metric_artifact_uploader = None
        self._offline_live_metric_buffer: list[tuple[int, dict[str, Any]]] = []
        self._pending_event_record_archives: list[_PendingEventRecordsArchive] = []
        self._event_record_archive_warning_emitted = False

    def _create_dataloader(
        self,
        train_dataset,
        val_dataset,
        collate_fn,
        train_sampler: Optional[Sampler],
        test_benchmark_dataset=None,
    ):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        self.test_benchmark_dataset = test_benchmark_dataset
        self.test_benchmark_dataloader = None
        if test_benchmark_dataset is not None:
            benchmark_config = self.config.data.test_benchmarks
            test_batch_size = benchmark_config.get("batch_size", None)
            if test_batch_size is None:
                test_batch_size = len(test_benchmark_dataset)
            test_batch_size = int(test_batch_size)
            if test_batch_size <= 0:
                raise ValueError(f"data.test_benchmarks.batch_size must be positive or null, got {test_batch_size}.")
            self.test_benchmark_dataloader = StatefulDataLoader(
                dataset=test_benchmark_dataset,
                batch_size=test_batch_size,
                num_workers=num_workers,
                shuffle=False,
                drop_last=False,
                collate_fn=collate_fn,
            )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )
        if self.test_benchmark_dataloader is not None:
            print(
                "Size of test benchmark dataloader: "
                f"{len(self.test_benchmark_dataloader)} ({len(self.test_benchmark_dataset)} problems)"
            )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(
        self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path, filename_prefix: str | None = None
    ):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename_stem = f"{filename_prefix}_step{self.global_steps}" if filename_prefix else str(self.global_steps)
        filename = os.path.join(dump_path, f"{filename_stem}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False, default=str))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in reward_extra_infos_dict.items()
            }
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_to_dump.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(
        self,
        inputs,
        outputs,
        scores,
        generations_to_log: int | None = None,
        generation_logger: ValidationGenerationsLogger | None = None,
        sample_records: list[dict[str, Any]] | None = None,
        defer_wandb: bool = False,
    ) -> dict[str, Any]:
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = (
            self.config.trainer.log_val_generations if generations_to_log is None else generations_to_log
        )
        generation_logger = generation_logger or self.validation_generations_logger

        if generations_to_log == 0:
            return {}

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = sample_records if sample_records is not None else list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x["input"] if isinstance(x, dict) else x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        return generation_logger.log(
            self.config.trainer.logger,
            samples,
            self.global_steps,
            defer_wandb=defer_wandb,
        )

    def _queue_generation_archive(
        self,
        *,
        evaluation_scope: str,
        records: list[dict[str, Any]],
    ) -> None:
        """Queue complete val/test records for the metric event at this step."""

        if evaluation_scope not in {"val", "test"} and not evaluation_scope.startswith("test-"):
            return
        # Preserve the historical generation-logging behavior outside balanced
        # mode; the explicit warning below is reserved for newly requested
        # intervention/paired-evaluation diagnostics.
        if (
            not records
            or self._metric_logging_settings.mode != "balanced"
            or getattr(self, "_metric_archive_writer", None) is None
        ):
            return
        self._queue_event_records_archive(
            metric_key=f"{evaluation_scope}/generations_full",
            records=records,
        )

    def _queue_event_records_archive(
        self,
        *,
        metric_key: str,
        records: list[dict[str, Any]],
    ) -> None:
        """Queue arbitrary detailed records for the current metric event."""
        if not records:
            return
        if self._metric_logging_settings.mode != "balanced" or getattr(self, "_metric_archive_writer", None) is None:
            if not getattr(self, "_event_record_archive_warning_emitted", False):
                print(
                    "WARNING: detailed intervention/evaluation event records require "
                    "trainer.metric_logging.mode=balanced with metric archival enabled. Aggregated metrics "
                    "will still be logged, but JSONL/Artifact records are unavailable for this run."
                )
                self._event_record_archive_warning_emitted = True
            return
        pending = getattr(self, "_pending_event_record_archives", None)
        if pending is None:
            pending = []
            self._pending_event_record_archives = pending
        pending.append(
            _PendingEventRecordsArchive(
                global_step=int(self.global_steps),
                metric_key=metric_key,
                records=tuple(copy.deepcopy(record) for record in records),
            )
        )

    def _queue_training_intervention_outcome_archive(
        self,
        *,
        batch: DataProto,
        global_steps: int,
    ) -> None:
        """Queue compact train outcomes and the bounded full-text case subset."""
        diagnostics_config = self._distillation_diagnostics_config()
        if not (
            diagnostics_config is not None
            and getattr(diagnostics_config, "collect_intervention_metrics", False)
            and getattr(diagnostics_config, "collect_intervention_outcome_transitions", False)
        ):
            return

        outcome_values = batch.non_tensor_batch.get("teacher_intervention_outcome")
        if outcome_values is None:
            raise RuntimeError(
                "Intervention outcome diagnostics are enabled, but rollout output did not include "
                "teacher_intervention_outcome. Check that the async DMPD agent-loop path is active."
            )

        sample_count = len(batch)

        def values_or_default(key: str, default: Any) -> list[Any]:
            values = batch.non_tensor_batch.get(key)
            if values is None:
                return [copy.deepcopy(default) for _ in range(sample_count)]
            result = self._non_tensor_to_list(values)
            if len(result) != sample_count:
                raise RuntimeError(
                    f"Training intervention archive field {key!r} has {len(result)} values for "
                    f"batch_size={sample_count}."
                )
            return result

        outcomes = self._non_tensor_to_list(outcome_values)
        if len(outcomes) != sample_count or not all(isinstance(item, dict) for item in outcomes):
            raise RuntimeError(
                "Intervention outcome diagnostics require one complete outcome dictionary per training sample."
            )
        records = _build_training_intervention_outcome_archive_records(
            global_step=global_steps,
            outcomes=outcomes,
            interventions=values_or_default("teacher_interventions", []),
            proposals=values_or_default("teacher_intervention_proposals", []),
            skipped_interventions=values_or_default("teacher_intervention_skipped", []),
            cases=values_or_default("teacher_intervention_outcome_case", None),
            sample_uids=values_or_default("uid", None),
            data_sources=values_or_default("data_source", "unknown"),
            raw_prompts=values_or_default("raw_prompt", None),
            reward_models=values_or_default("reward_model", None),
        )
        self._queue_event_records_archive(
            metric_key="opd_intervention/outcome_pairs",
            records=records,
        )

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = []
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _compute_reward_colocate(self, batch: DataProto) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor:
        """
        compute reward use colocate reward model
        """
        assert self.reward_loop_manager is not None, "RewardLoopManager is None"
        batch_reward = self.reward_loop_manager.compute_rm_score(batch)
        return batch_reward

    def _validate(
        self,
        merged: bool = False,
        dataloader=None,
        generation_manager=None,
        metric_prefix: str = "val",
        evaluation_mode: str = "val",
        max_samples: int | None = None,
        log_generations: int | None = None,
        validation_generations_logger: ValidationGenerationsLogger | None = None,
        validation_data_dir: str | None = None,
        dump_filename_prefix: str | None = None,
        include_format_metrics: bool = False,
        sampling_config=None,
        sampling_profile_name: str | None = None,
        repeat_times: int | None = None,
        do_sample: bool | None = None,
        generation_max_tokens: int | None = None,
        suppress_tracking: bool = False,
    ):
        if evaluation_mode not in {"val", "test"}:
            raise ValueError(f"evaluation_mode must be 'val' or 'test', got {evaluation_mode!r}.")
        dataloader = self.val_dataloader if dataloader is None else dataloader
        default_generation_manager = getattr(self, "async_rollout_manager", None)
        generation_manager = generation_manager or default_generation_manager
        if generation_manager is None:
            raise RuntimeError("Validation requires a generation manager.")
        if sampling_config is None:
            sampling_config = (
                self.config.actor_rollout_ref.rollout.test_kwargs
                if evaluation_mode == "test"
                else self.config.actor_rollout_ref.rollout.val_kwargs
            )
        repeat_times = int(sampling_config.n) if repeat_times is None else int(repeat_times)
        if repeat_times <= 0:
            raise ValueError(f"Evaluation repeat_times must be positive, got {repeat_times}.")
        resolved_do_sample = bool(sampling_config.do_sample if do_sample is None else do_sample)
        evaluation_sampling_params = {
            "temperature": float(sampling_config.temperature),
            "top_p": float(sampling_config.top_p),
            "top_k": int(sampling_config.top_k),
            "do_sample": resolved_do_sample,
        }
        if generation_max_tokens is not None:
            generation_max_tokens = int(generation_max_tokens)
            if generation_max_tokens <= 0:
                raise ValueError(f"generation_max_tokens must be positive, got {generation_max_tokens}.")
        validation_data_dir = (
            self.config.trainer.get("validation_data_dir", None) if validation_data_dir is None else validation_data_dir
        )
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []
        sample_response_lengths = []
        sample_finish_reasons = []
        validation_tool_metric_values: dict[str, list[float]] = defaultdict(list)
        processed_samples = 0

        for test_data in dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            if max_samples is not None:
                remaining = max_samples - processed_samples
                if remaining <= 0:
                    break
                if len(test_batch) > remaining:
                    test_batch = test_batch[:remaining]
                processed_samples += len(test_batch)

            if evaluation_mode == "val":
                validation_reward_data_source = self.config.data.get("validation_reward_data_source", None)
                if validation_reward_data_source:
                    test_batch.non_tensor_batch["extra_info"] = _with_reward_data_source(
                        test_batch.non_tensor_batch.get("extra_info"),
                        count=len(test_batch),
                        reward_data_source=str(validation_reward_data_source),
                    )

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": resolved_do_sample,
                "validate": True,
                "evaluation_mode": evaluation_mode,
                "evaluation_sampling_params": evaluation_sampling_params,
                "global_steps": self.global_steps,
            }
            if sampling_profile_name is not None:
                test_gen_batch.meta_info["evaluation_sampling_profile"] = sampling_profile_name
            if generation_max_tokens is not None:
                test_gen_batch.meta_info["generation_max_tokens"] = generation_max_tokens
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = self.config.actor_rollout_ref.rollout.agent.num_workers
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            test_output_gen_batch_padded = generation_manager.generate_sequences(test_gen_batch_padded)

            if self.use_rm and "rm_scores" not in test_output_gen_batch_padded.batch.keys():
                # for colocate reward models, we need to sleep rollout model
                # to spare GPU memory for reward model
                self.checkpoint_manager.sleep_replicas()
                batch_reward = self._compute_reward_colocate(test_output_gen_batch_padded)
                test_output_gen_batch_padded = test_output_gen_batch_padded.union(batch_reward)
                # wake up rollout model
                # replace with wake_up method once supported
                self.checkpoint_manager.update_weights(self.global_steps)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            _extend_validation_tool_metric_values(
                validation_tool_metric_values,
                test_output_gen_batch.non_tensor_batch,
            )

            print("validation generation end")

            final_output_ids = test_output_gen_batch.batch["responses"]
            final_output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in final_output_ids]
            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # Store original inputs
            input_ids = test_batch.batch["prompts"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            reward_tensor, reward_extra_info = extract_reward(test_batch)
            final_scores = reward_tensor.sum(-1).cpu().tolist()
            final_reward_infos = _reward_info_dict_to_samples(
                reward_extra_info,
                sample_count=len(final_scores),
            )
            prior_output_count = len(sample_outputs)
            output_count = len(final_output_texts)
            sample_outputs.extend(final_output_texts)
            sample_scores.extend(final_scores)
            if "response_mask" in test_output_gen_batch.batch:
                sample_response_lengths.extend(
                    [int(length) for length in test_output_gen_batch.batch["response_mask"].sum(dim=-1).cpu().tolist()]
                )
            if "finish_reason" in test_output_gen_batch.non_tensor_batch:
                if len(sample_finish_reasons) < prior_output_count:
                    sample_finish_reasons.extend([None] * (prior_output_count - len(sample_finish_reasons)))
                sample_finish_reasons.extend(test_output_gen_batch.non_tensor_batch["finish_reason"].tolist())
            elif sample_finish_reasons:
                sample_finish_reasons.extend([None] * output_count)
            _extend_reward_extra_infos(
                reward_extra_infos_dict,
                scores=final_scores,
                per_sample_infos=final_reward_infos,
            )

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        sample_records = (
            _build_validation_sample_records(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                response_lengths=sample_response_lengths,
                finish_reasons=sample_finish_reasons,
            )
            if include_format_metrics
            else None
        )
        validation_media = {}
        if not suppress_tracking:
            generations_to_log = int(
                self.config.trainer.log_val_generations if log_generations is None else log_generations
            )
            if generations_to_log < 0:
                raise ValueError(f"generations_to_log must be non-negative, got {generations_to_log}.")
            if generations_to_log > 0:
                flattened_data_sources = np.concatenate(data_source_lst, axis=0).tolist() if data_source_lst else None
                generation_archive_records = _build_generation_archive_records(
                    evaluation_scope=metric_prefix,
                    global_step=self.global_steps,
                    inputs=sample_inputs,
                    outputs=sample_outputs,
                    gts=sample_gts,
                    scores=sample_scores,
                    reward_extra_infos_dict=reward_extra_infos_dict,
                    response_lengths=sample_response_lengths or None,
                    finish_reasons=sample_finish_reasons or None,
                    data_sources=flattened_data_sources,
                    sample_uids=sample_uids,
                )
                self._queue_generation_archive(
                    evaluation_scope=metric_prefix,
                    records=generation_archive_records,
                )
            validation_media = self._maybe_log_val_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                generations_to_log=generations_to_log,
                generation_logger=validation_generations_logger,
                sample_records=sample_records,
                defer_wandb=not merged,
            )

        # dump generations
        if validation_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=validation_data_dir,
                filename_prefix=dump_filename_prefix,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        if merged:
            print("_merge_validation_results validate result will be merged")
            return {
                "data_sources": data_source_lst,
                "sample_uids": sample_uids,
                "sample_turns": sample_turns,
                "reward_extra_infos_dict": reward_extra_infos_dict,
            }
        data_sources = np.concatenate(data_source_lst, axis=0)
        metrics = self._val_metrics_update(
            data_sources, sample_uids, reward_extra_infos_dict, sample_turns, metric_prefix=metric_prefix
        )
        if include_format_metrics:
            for metric_name, value in _compute_validation_format_metrics(
                sample_outputs, reward_extra_infos_dict
            ).items():
                metrics[f"{metric_prefix}-aux/{metric_name}"] = value
            validation_max_response_length = int(
                generation_max_tokens
                if generation_max_tokens is not None
                else self.config.actor_rollout_ref.rollout.response_length
            )
            metrics.update(
                compute_text_repetition_and_truncation_metrics(
                    outputs=sample_outputs,
                    response_lengths=sample_response_lengths,
                    max_response_length=validation_max_response_length,
                    finish_reasons=sample_finish_reasons,
                    prefix=f"{metric_prefix}-aux",
                )
            )
        metrics.update(
            _summarize_validation_tool_metrics(
                validation_tool_metric_values,
                metric_prefix=metric_prefix,
            )
        )
        metrics.update(validation_media)
        return metrics

    def _val_metrics_update(
        self,
        data_sources,
        sample_uids,
        reward_extra_infos_dict,
        sample_turns,
        metric_prefix="val",
        core_mean_only: bool = False,
    ):
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(
                            metric_name.startswith(pfx)
                            for pfx in (["mean"] if core_mean_only else ["mean", "maj", "best"])
                        )
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = f"{metric_prefix}-core"
                    else:
                        metric_sec = f"{metric_prefix}-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict[f"{metric_prefix}-aux/num_turns/min"] = sample_turns.min()
            metric_dict[f"{metric_prefix}-aux/num_turns/max"] = sample_turns.max()
            metric_dict[f"{metric_prefix}-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def _validate_test_benchmarks(self, *, suppress_tracking: bool = False) -> dict[str, Any]:
        """Run the configured benchmark datasets through the student validation path."""
        if self.test_benchmark_dataloader is None:
            raise RuntimeError(
                "Test benchmark evaluation was requested, but the benchmark dataloader is not initialized."
            )

        generation_max_tokens = self.config.trainer.get("test_benchmarks_max_response_length", None)
        if generation_max_tokens is None:
            generation_max_tokens = int(self.config.actor_rollout_ref.rollout.response_length)
        else:
            generation_max_tokens = int(generation_max_tokens)
        if generation_max_tokens > int(self.config.actor_rollout_ref.rollout.response_length):
            raise ValueError(
                "trainer.test_benchmarks_max_response_length cannot exceed "
                "actor_rollout_ref.rollout.response_length; increase the rollout/model limits together. "
                f"Got {generation_max_tokens} > {self.config.actor_rollout_ref.rollout.response_length}."
            )

        profiles = self.config.actor_rollout_ref.rollout.get("benchmark_sampling_profiles", None)
        if not profiles:
            raise ValueError("Benchmark evaluation requires actor_rollout_ref.rollout.benchmark_sampling_profiles.")
        expected_profiles = ("legacy", "train_sampling")
        missing_profiles = [name for name in expected_profiles if name not in profiles]
        extra_profiles = [name for name in profiles if name not in expected_profiles]
        if missing_profiles or extra_profiles:
            raise ValueError(
                "Benchmark sampling profiles must be exactly legacy and train_sampling; "
                f"missing={missing_profiles}, extra={extra_profiles}."
            )

        test_data_dir = self.config.trainer.get("test_data_dir", None)
        metrics: dict[str, Any] = {}
        for profile_name in expected_profiles:
            sampling_config = profiles[profile_name]
            metric_profile_name = profile_name.replace("_", "-")
            profile_metrics = self._validate(
                dataloader=self.test_benchmark_dataloader,
                metric_prefix=f"test-{metric_profile_name}",
                evaluation_mode="test",
                sampling_config=sampling_config,
                sampling_profile_name=profile_name,
                log_generations=int(self.config.trainer.get("log_test_generations", 0)),
                validation_generations_logger=self.test_benchmark_generations_loggers[profile_name],
                validation_data_dir=test_data_dir or "",
                dump_filename_prefix=f"test_benchmarks_{profile_name}",
                include_format_metrics=True,
                generation_max_tokens=generation_max_tokens,
                suppress_tracking=suppress_tracking,
            )
            duplicate_keys = metrics.keys() & profile_metrics.keys()
            if duplicate_keys:
                raise RuntimeError(f"Benchmark sampling profiles produced duplicate metrics: {sorted(duplicate_keys)}")
            metrics.update(profile_metrics)
        assert metrics, f"{metrics=}"
        return metrics

    def _make_teacher_validation_config(self):
        return copy.deepcopy(self.config)

    def _get_teacher_validation_manager(self):
        if self.teacher_validation_manager is not None:
            return self.teacher_validation_manager
        if self.teacher_model_manager is None:
            raise RuntimeError("teacher validation requested but teacher_model_manager is not initialized")
        if self.agent_loop_manager_cls is None:
            raise RuntimeError("teacher validation requested before AgentLoopManager class is initialized")

        teacher_clients = self.teacher_model_manager.get_client()
        teacher_config = self._make_teacher_validation_config()
        teacher_key = self.config.distillation.teacher_key
        self.teacher_validation_manager = _TeacherValidationGenerationManager(
            config=teacher_config,
            agent_loop_manager_cls=self.agent_loop_manager_cls,
            teacher_clients=teacher_clients,
            teacher_key=teacher_key,
            reward_loop_worker_handles=self.reward_loop_worker_handles,
        )
        return self.teacher_validation_manager

    def _validate_teacher_before_train(self, *, suppress_tracking: bool = False):
        if not self.config.trainer.get("teacher_val_before_train", False):
            return {}
        if not self.use_teacher_policy or self.distillation_config is None:
            print("Skipping teacher validation before train because distillation teacher policy is not enabled.")
            return {}
        if not is_distillation_enabled(self.config.distillation):
            print("Skipping teacher validation before train because distillation.enabled is false.")
            return {}

        teacher_val_max_samples = self.config.trainer.get("teacher_val_max_samples", 256)
        if teacher_val_max_samples is not None:
            teacher_val_max_samples = int(teacher_val_max_samples)
            if teacher_val_max_samples == 0:
                print("Skipping teacher validation before train because trainer.teacher_val_max_samples=0.")
                return {}
            if teacher_val_max_samples < 0:
                teacher_val_max_samples = None

        teacher_log_generations = int(self.config.trainer.get("log_teacher_val_generations", 0))
        teacher_generation_manager = self._get_teacher_validation_manager()
        teacher_generation_max_tokens = int(self.config.actor_rollout_ref.rollout.response_length)
        teacher_metrics = self._validate(
            generation_manager=teacher_generation_manager,
            metric_prefix="teacher-val",
            max_samples=teacher_val_max_samples,
            log_generations=teacher_log_generations,
            validation_generations_logger=self.teacher_validation_generations_logger,
            validation_data_dir=self.config.trainer.get("validation_data_dir", None),
            dump_filename_prefix="teacher_val",
            include_format_metrics=True,
            generation_max_tokens=teacher_generation_max_tokens,
            suppress_tracking=suppress_tracking,
        )
        assert teacher_metrics, f"{teacher_metrics=}"
        return teacher_metrics

    def _merge_validation_results(self, result_a, result_b):
        if result_a is None and result_b is None:
            return {}
        if result_a is None:
            result_a = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}
        if result_b is None:
            result_b = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}

        if not result_a.get("data_sources") and not result_b.get("data_sources"):
            return {}

        data_sources = np.concatenate(result_a["data_sources"] + result_b["data_sources"], axis=0)
        sample_uids = result_a["sample_uids"] + result_b["sample_uids"]
        sample_turns = result_a["sample_turns"] + result_b["sample_turns"]

        reward_extra_infos_dict = {}
        all_keys = set(result_a["reward_extra_infos_dict"].keys()) | set(result_b["reward_extra_infos_dict"].keys())
        for key in all_keys:
            list_a = result_a["reward_extra_infos_dict"].get(key, [])
            list_b = result_b["reward_extra_infos_dict"].get(key, [])
            reward_extra_infos_dict[key] = list_a + list_b

        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                distillation_config=self.config.get("distillation"),
                role=str(actor_role),
            )
            self.resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

            from verl.workers.config import CriticConfig

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)

            # convert critic_cfg into TrainingWorkerConfig for the unified model engine worker
            from verl.workers.engine_workers import TrainingWorkerConfig

            orig_critic_cfg = critic_cfg
            engine_config: EngineConfig = orig_critic_cfg.engine
            engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
            engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu

            critic_cfg = TrainingWorkerConfig(
                model_type="value_model",
                model_config=orig_critic_cfg.model,
                engine_config=engine_config,
                optimizer_config=orig_critic_cfg.optim,
                checkpoint_config=orig_critic_cfg.checkpoint,
            )

            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/verl-project/verl/blob/master/examples/tutorial/ray/tutorial.ipynb
        # for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            if not class_dict:
                continue
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.reset()
            # assign critic loss
            from functools import partial

            from verl.workers.utils.losses import value_loss

            value_loss_ = partial(value_loss, config=orig_critic_cfg)
            self.critic_wg.set_loss_fn(value_loss_)

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # create reward loop manager
        from verl.experimental.reward_loop import RewardLoopManager

        # initalize reward loop manager
        # reward model (colocate or standalone): get resource_pool
        # no reward model: resource_pool = None
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel) if self.use_rm else None
        self.reward_loop_manager = RewardLoopManager(
            config=self.config,
            rm_resource_pool=resource_pool,
        )

        # create async rollout manager and request scheduler
        # Note: mode is always "async" since sync mode is deprecated
        self.async_rollout_mode = True

        # initialize teacher loop manager
        if self.use_teacher_policy:
            from verl.experimental.teacher_loop import MultiTeacherModelManager

            teacher_resource_pool = self.resource_pool_manager.get_resource_pool(Role.TeacherModel)
            self.teacher_model_manager = MultiTeacherModelManager(
                config=self.config,
                resource_pool=teacher_resource_pool,
            )
            self.distillation_config: DistillationConfig = omega_conf_to_dataclass(self.config.distillation)
        else:
            self.teacher_model_manager = None
            self.distillation_config = None

        # Support custom AgentLoopManager via config
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager
        self.agent_loop_manager_cls = AgentLoopManager

        # infrastructure overview: https://verl.readthedocs.io/en/latest/advance/reward_loop.html#architecture-design
        # agent_reward_loop: streaming reward computation with actor rollout
        # two conditions satisfied: (1) no reward model, or (2) reward model with extra resource pool
        enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool

        self.llm_server_manager = LLMServerManager.create(
            config=self.config, worker_group=self.actor_rollout_wg, rollout_resource_pool=actor_rollout_resource_pool
        )

        # if enable_agent_reward_loop, we directly pass reward_loop_workers to agent loop manager
        # to stream reward computation with actor rollout
        # To stream teacher computation with actor rollout, we instead pass the full manager so that the
        # teacher loop workers can sleep/wake together with rollout workers
        reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None
        self.reward_loop_worker_handles = reward_loop_worker_handles
        self.async_rollout_manager = AgentLoopManager.create(
            config=self.config,
            llm_client=self.llm_server_manager.get_client(),
            teacher_client=self.teacher_model_manager.get_client() if self.use_teacher_policy else None,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )

        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        # Support custom CheckpointEngineManager via config
        checkpoint_manager_class_fqn = self.config.actor_rollout_ref.rollout.get("checkpoint_manager_class")
        if checkpoint_manager_class_fqn:
            CheckpointEngineManager = load_class_from_fqn(checkpoint_manager_class_fqn, "CheckpointEngineManager")
        else:
            from verl.checkpoint_engine import CheckpointEngineManager
        self.checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            trainer=self.actor_rollout_wg,
            replicas=self.llm_server_manager.get_replicas(),
        )

        # sleep all replicas to load checkpoint
        self.checkpoint_manager.sleep_replicas()

    def _checkpoint_async_save_enabled(self) -> bool:
        checkpoint_config = self.config.actor_rollout_ref.actor.checkpoint
        return bool(checkpoint_config.get("async_save", False))

    def _save_checkpoint_payload(self) -> str:
        """Save checkpoint payloads without advancing the committed-checkpoint tracker."""
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        if self._metric_logging_mode() == "balanced":
            # The root tracker is intentionally committed after the metric
            # marker. Preserve its currently selected payload throughout that
            # window, even when the requested retention count is one.
            if (
                isinstance(max_actor_ckpt_to_keep, int)
                and not isinstance(max_actor_ckpt_to_keep, bool)
                and max_actor_ckpt_to_keep > 0
            ):
                max_actor_ckpt_to_keep += 1
            if (
                isinstance(max_critic_ckpt_to_keep, int)
                and not isinstance(max_critic_ckpt_to_keep, bool)
                and max_critic_ckpt_to_keep > 0
            ):
                max_critic_ckpt_to_keep += 1

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)
        return local_global_step_folder

    def _commit_checkpoint_tracker(self, checkpoint_step: int) -> None:
        """Atomically make a fully written checkpoint visible to resume discovery."""
        if self._checkpoint_async_save_enabled():
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return

        checkpoint_root = os.path.abspath(self.config.trainer.default_local_dir)
        os.makedirs(checkpoint_root, exist_ok=True)
        local_latest_checkpointed_iteration = os.path.join(checkpoint_root, "latest_checkpointed_iteration.txt")
        temporary_tracker = f"{local_latest_checkpointed_iteration}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        with open(temporary_tracker, "w", encoding="utf-8") as tracker_file:
            tracker_file.write(str(int(checkpoint_step)))
            tracker_file.flush()
            os.fsync(tracker_file.fileno())
        os.replace(temporary_tracker, local_latest_checkpointed_iteration)
        directory_fd = os.open(checkpoint_root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def _save_checkpoint(self) -> str:
        """Backward-compatible checkpoint save with an immediate tracker commit."""
        local_global_step_folder = self._save_checkpoint_payload()
        self._commit_checkpoint_tracker(self.global_steps)
        return local_global_step_folder

    def _metric_logging_mode(self) -> str:
        metric_logging = self.config.trainer.get("metric_logging", None)
        return "full" if metric_logging is None else str(metric_logging.get("mode", "full")).lower()

    def _resolve_resume_context(self) -> _ResolvedCheckpointContext:
        """Resolve the checkpoint once, before W&B initialization and worker loading."""
        if self._resolved_checkpoint_context is not None:
            return self._resolved_checkpoint_context

        resume_mode = str(self.config.trainer.resume_mode)
        if resume_mode not in {"disable", "auto", "resume_path"}:
            raise ValueError(
                f"trainer.resume_mode must be one of 'disable', 'auto', or 'resume_path', got {resume_mode!r}."
            )

        checkpoint_root = os.path.abspath(self.config.trainer.default_local_dir)
        tracker_path = os.path.join(checkpoint_root, "latest_checkpointed_iteration.txt")
        metric_logging = self.config.trainer.get("metric_logging", {})
        archive_config = metric_logging.get("archive", {}) if metric_logging is not None else {}
        archive_dir = archive_config.get("local_dir", None)
        if archive_dir is None:
            archive_dir = os.path.join(checkpoint_root, "diagnostics")
        else:
            archive_dir = os.path.abspath(os.path.expanduser(str(archive_dir)))
        manifest_path = os.path.join(archive_dir, "manifest.json")
        strict_balanced = (
            self._metric_logging_mode() == "balanced"
            and str(metric_logging.get("resume_policy", "strict")).lower() == "strict"
        )

        if resume_mode == "disable":
            if strict_balanced:
                checkpoint_entries = []
                if os.path.isdir(checkpoint_root):
                    checkpoint_entries = [
                        entry.name
                        for entry in os.scandir(checkpoint_root)
                        if entry.is_dir() and entry.name.startswith("global_step_")
                    ]
                if os.path.exists(tracker_path) or os.path.exists(manifest_path) or checkpoint_entries:
                    raise RuntimeError(
                        "trainer.resume_mode=disable cannot reuse an output directory containing checkpoint "
                        "or metric archive state in balanced/strict mode. Choose a new trainer.default_local_dir "
                        f"or enable resume. checkpoint_root={checkpoint_root!r}, manifest={manifest_path!r}."
                    )
            self._resolved_checkpoint_context = _ResolvedCheckpointContext(path=None, step=0)
            return self._resolved_checkpoint_context

        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")

        if resume_mode == "auto":
            global_step_folder = find_latest_ckpt_path(checkpoint_root)
            if global_step_folder is None:
                if strict_balanced and os.path.exists(tracker_path):
                    raise RuntimeError(
                        "The checkpoint tracker exists but does not resolve to a checkpoint directory: "
                        f"tracker={tracker_path!r}."
                    )
                print("Training from scratch")
                self._resolved_checkpoint_context = _ResolvedCheckpointContext(path=None, step=0)
                return self._resolved_checkpoint_context
        else:
            resume_from_path = self.config.trainer.resume_from_path
            if not isinstance(resume_from_path, str):
                raise TypeError("trainer.resume_from_path must be a string when resume_mode='resume_path'.")
            global_step_folder = os.path.abspath(os.path.expanduser(resume_from_path))
            if not os.path.isdir(global_step_folder):
                raise FileNotFoundError(f"Resume checkpoint directory does not exist: {global_step_folder}")

        checkpoint_name = os.path.basename(os.path.normpath(global_step_folder))
        if not checkpoint_name.startswith("global_step_"):
            raise ValueError(f"Resume checkpoint must be named global_step_<integer>, got {global_step_folder!r}.")
        try:
            checkpoint_step = int(checkpoint_name.removeprefix("global_step_"))
        except ValueError as exc:
            raise ValueError(
                f"Resume checkpoint must be named global_step_<integer>, got {global_step_folder!r}."
            ) from exc

        if strict_balanced and os.path.exists(tracker_path):
            with open(tracker_path, encoding="utf-8") as tracker_file:
                tracker_step = int(tracker_file.read().strip())
            if tracker_step != checkpoint_step:
                raise RuntimeError(
                    "Checkpoint identity mismatch in balanced/strict resume: "
                    f"resolved_checkpoint_step={checkpoint_step}, tracker_step={tracker_step}, "
                    f"checkpoint={global_step_folder!r}, tracker={tracker_path!r}."
                )

        self._resolved_checkpoint_context = _ResolvedCheckpointContext(
            path=os.path.abspath(global_step_folder),
            step=checkpoint_step,
        )
        return self._resolved_checkpoint_context

    def _load_checkpoint(self):
        resume_context = self._resolve_resume_context()
        if not resume_context.is_resume:
            return 0

        global_step_folder = resume_context.path
        assert global_step_folder is not None
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = resume_context.step

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            steps_per_epoch = len(self.train_dataloader)
            at_epoch_boundary = steps_per_epoch > 0 and self.global_steps % steps_per_epoch == 0
            if at_epoch_boundary:
                print(
                    f"Skipping dataloader state restore: global_steps={self.global_steps} "
                    f"is at an epoch boundary (steps_per_epoch={steps_per_epoch}). "
                    f"The saved state marks the dataloader as exhausted. "
                    f"Next epoch will iterate from scratch."
                )
            else:
                dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
                self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()

    def _get_dp_size(self, worker_group, role: str) -> int:
        """Get data parallel size from worker group dispatch info.

        This method retrieves the data parallel size by querying the dispatch info
        for the specified role. The dispatch info is cached for subsequent calls.

        Args:
            worker_group: The worker group to query dispatch info from.
            role: The role name (e.g., "actor", "critic") to get DP size for.

        Returns:
            The data parallel size (number of DP ranks).
        """
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        return max(dp_rank_mapping) + 1

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens.

        When use_prefix_grouper is enabled, uses group-level balancing to keep samples with
        the same uid together on the same rank for prefix sharing optimization.
        """
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)
        # Get dp_size from dispatch info to correctly balance across data parallel ranks
        # Note: world_size may include tensor/pipeline parallel dimensions, but we only want DP
        dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")

        # Use group-level balancing for PrefixGrouper to keep same-uid samples together
        if getattr(self, "use_prefix_grouper", False) and "uid" in batch.non_tensor_batch:
            from verl.utils.seqlen_balancing import get_group_balanced_partitions

            uid_list = list(batch.non_tensor_batch["uid"])
            seqlen_list = global_seqlen_lst.tolist()

            # Count number of uid groups
            num_groups = len(set(uid_list))

            if num_groups % dp_size != 0:
                raise ValueError(
                    f"PrefixGrouper with balance_batch requires num_uid_groups ({num_groups}) "
                    f"% dp_size ({dp_size}) == 0. "
                    f"This ensures each rank gets equal number of groups. "
                    f"Current batch_size={batch_size}, adjust batch_size to be a multiple of "
                    f"dp_size * rollout.n."
                )

            global_partition_lst = get_group_balanced_partitions(
                seqlen_list=seqlen_list,
                uid_list=uid_list,
                k_partitions=dp_size,
            )

        elif keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(dp_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=dp_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        # Skip reordering within partitions for PrefixGrouper to maintain uid grouping
        if not getattr(self, "use_prefix_grouper", False):
            for idx, partition in enumerate(global_partition_lst):
                partition.sort(key=lambda x: (workload_lst[x], x))
                ordered_partition = partition[::2] + partition[1::2][::-1]
                global_partition_lst[idx] = ordered_partition

        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(), partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _compute_values(self, batch: DataProto) -> DataProto:
        batch_td = self._to_tensordict_without_distillation_supervision(batch)
        # step 2: convert from padding to nopadding
        batch_td = left_right_2_no_padding(batch_td)
        # step 3: add meta info
        tu.assign_non_tensor(batch_td, compute_loss=False)
        output = self.critic_wg.infer_batch(batch_td)
        output = output.get()
        values = tu.get(output, "values")
        values = no_padding_2_padding(values, batch_td)
        values = tu.get_tensordict({"values": values.float()})
        values = DataProto.from_tensordict(values)
        return values

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        # step 1: convert dataproto to tensordict.
        batch_td = self._to_tensordict_without_distillation_supervision(batch)
        # step 2: convert from padding to nopadding
        batch_td = left_right_2_no_padding(batch_td)
        # step 3: add meta info
        metadata = {"calculate_entropy": False, "compute_loss": False}
        if self.ref_in_actor:
            metadata["no_lora_adapter"] = True
        tu.assign_non_tensor(batch_td, **metadata)
        if self.ref_in_actor:
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
        else:
            output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
        # gather output
        log_probs = tu.get(output, "log_probs")
        # step 4. No padding to padding
        log_probs = no_padding_2_padding(log_probs, batch_td)
        # step 5: rebuild a tensordict and convert to dataproto
        ref_log_prob = tu.get_tensordict({"ref_log_prob": log_probs.float()})
        ref_log_prob = DataProto.from_tensordict(ref_log_prob)

        return ref_log_prob

    def _compute_old_log_prob(self, batch: DataProto, global_steps: int | None = None):
        # TODO: remove step 1, 2, 4 after we make the whole training tensordict and padding free
        # step 1: convert dataproto to tensordict.
        batch_td = self._to_tensordict_without_distillation_supervision(batch)
        # step 2: convert from padding to nopadding
        batch_td = left_right_2_no_padding(batch_td)
        # step 3: add meta info
        calculate_sum_pi_squared = self.config.actor_rollout_ref.actor.get("calculate_sum_pi_squared", False)
        diagnostics_config = self._distillation_diagnostics_config()
        diagnostics_student_topk = self._token_reward_direct_topk()
        if (
            diagnostics_config is not None
            and getattr(diagnostics_config, "collect_rethinking_opd_metrics", False)
            and is_distillation_enabled(self.config.get("distillation"))
            and (global_steps is None or global_steps % diagnostics_config.log_interval == 0)
        ):
            diagnostics_student_topk = max(
                diagnostics_student_topk,
                int(diagnostics_config.topk or self.distillation_config.distillation_loss.topk or 64),
            )
        tu.assign_non_tensor(
            batch_td,
            calculate_entropy=True,
            calculate_sum_pi_squared=calculate_sum_pi_squared,
            diagnostics_student_topk=diagnostics_student_topk,
            compute_loss=False,
        )
        output = self.actor_rollout_wg.compute_log_prob(batch_td)
        # gather output
        entropy = tu.get(output, "entropy")
        log_probs = tu.get(output, "log_probs")
        routed_experts = tu.get(output, "routed_experts")
        sum_pi_squared = tu.get(output, "sum_pi_squared") if calculate_sum_pi_squared else None
        student_topk_ids = tu.get(output, "student_topk_ids") if diagnostics_student_topk > 0 else None
        student_topk_logprobs = tu.get(output, "student_topk_logprobs") if diagnostics_student_topk > 0 else None

        old_log_prob_mfu = tu.get(output, "metrics")["mfu"]
        # step 4. No padding to padding
        entropy = no_padding_2_padding(entropy, batch_td)
        log_probs = no_padding_2_padding(log_probs, batch_td)
        if sum_pi_squared is not None:
            sum_pi_squared = no_padding_2_padding(sum_pi_squared, batch_td)
        if student_topk_ids is not None and student_topk_logprobs is not None:
            student_topk_ids = no_padding_2_padding(student_topk_ids, batch_td)
            student_topk_logprobs = no_padding_2_padding(student_topk_logprobs, batch_td)
        # step 5: rebuild a tensordict and convert to dataproto
        result = {"old_log_probs": log_probs.float(), "entropys": entropy.float()}
        if routed_experts is not None:
            result["routed_experts"] = routed_experts
        if sum_pi_squared is not None:
            result["sum_pi_squared"] = sum_pi_squared.float()
        if student_topk_ids is not None and student_topk_logprobs is not None:
            result["student_topk_ids"] = student_topk_ids.long()
            result["student_topk_logprobs"] = student_topk_logprobs.float()
        old_log_prob = tu.get_tensordict(result)
        old_log_prob = DataProto.from_tensordict(old_log_prob)
        return old_log_prob, old_log_prob_mfu

    @staticmethod
    def _to_tensordict_without_distillation_supervision(batch: DataProto):
        return to_tensordict_without_distillation_supervision(batch)

    def _update_actor(self, batch: DataProto) -> DataProto:
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        # TODO: Make "temperature" single source of truth from generation.
        batch.meta_info["temperature"] = rollout_config.temperature
        # update actor
        batch_td = batch.to_tensordict()
        if self._uses_distillation_policy_loss_mask():
            if "distillation_loss_mask" not in batch_td:
                raise RuntimeError(
                    "Masked token-reward distillation requires distillation_loss_mask to be attached "
                    "before the actor update."
                )
            # Keep the supervision mask actor-local. response_mask remains the full rollout mask for
            # old/reference log-probabilities, entropy, KL, rewards, and diagnostics.
            batch_td["policy_loss_mask"] = batch_td["distillation_loss_mask"].clone()
        # step 2: convert from padding to no-padding
        batch_td = left_right_2_no_padding(batch_td)
        calculate_entropy = self.config.actor_rollout_ref.actor.calculate_entropy or (
            self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
        )
        token_reward_direct_topk = self._token_reward_direct_topk()
        teacher_segment_sft_enabled = is_teacher_segment_sft_enabled(self.distillation_config)
        distillation_use_topk = (
            self.distillation_config.distillation_loss.loss_settings.use_topk
            and getattr(self.distillation_config, "use_actor_loss", True)
            if is_distillation_enabled(self.config.get("distillation"))
            else False
        )
        ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
        ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
        seed = self.config.actor_rollout_ref.actor.data_loader_seed
        shuffle = self.config.actor_rollout_ref.actor.shuffle
        tu.assign_non_tensor(
            batch_td,
            calculate_entropy=calculate_entropy,
            distillation_use_topk=distillation_use_topk,
            token_reward_direct_topk=token_reward_direct_topk,
            joint_token_reward_rlvr=self._uses_joint_token_reward_rlvr(),
            teacher_segment_sft_enabled=teacher_segment_sft_enabled,
            global_batch_size=ppo_mini_batch_size,
            mini_batch_size=ppo_mini_batch_size,
            epochs=ppo_epochs,
            seed=seed,
            dataloader_kwargs={"shuffle": shuffle},
            compute_loss=True,
        )
        actor_output = self.actor_rollout_wg.update_actor(batch_td)
        actor_output = tu.get(actor_output, "metrics")
        actor_output = rename_dict(actor_output, "actor/")
        # modify key name
        actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
        actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})

        return actor_output

    def _distillation_diagnostics_config(self):
        if self.distillation_config is None:
            return None
        diagnostics_config = getattr(self.distillation_config, "diagnostics", None)
        if diagnostics_config is None or not diagnostics_config.enabled:
            return None
        return diagnostics_config

    def _uses_token_reward_direct(self) -> bool:
        if self.distillation_config is None or not is_distillation_enabled(self.config.get("distillation")):
            return False
        adv_estimator = self.config.algorithm.adv_estimator
        adv_estimator = getattr(adv_estimator, "value", adv_estimator)
        return str(adv_estimator) == "token_reward_direct"

    def _uses_joint_token_reward_rlvr(self) -> bool:
        if not self._uses_token_reward_direct():
            return False
        return bool(self.distillation_config.distillation_loss.use_task_rewards)

    def _uses_distillation_policy_loss_mask(self) -> bool:
        """Return whether token-reward OPD needs an actor-local supervision mask."""
        distillation_config = getattr(self, "distillation_config", None)
        if not is_distillation_enabled(distillation_config):
            return False

        intervention_config = getattr(distillation_config, "intervention", None)
        intervention_enabled = bool(_get_config_value(intervention_config, "enabled", False))
        prune_enabled = intervention_enabled and bool(_get_config_value(intervention_config, "prune", False))

        loss_config = distillation_config.distillation_loss
        horizon_mode = str(_get_horizon_config_value(loss_config.horizon, "mode", "full")).lower()
        horizon_enabled = not intervention_enabled and horizon_mode in {"fixed", "progressive"}
        if not (prune_enabled or horizon_enabled):
            return False

        adv_estimator = self.config.algorithm.adv_estimator
        adv_estimator = getattr(adv_estimator, "value", adv_estimator)
        if str(adv_estimator) != "token_reward_direct":
            return False

        if getattr(distillation_config, "use_actor_loss", True):
            raise ValueError(
                "Masked token-reward distillation requires distillation.use_actor_loss=False when "
                "ADV_ESTIMATOR=token_reward_direct."
            )

        actor_config = self.config.actor_rollout_ref.actor
        policy_loss_config = actor_config.policy_loss
        policy_loss_mode = policy_loss_config.get("loss_mode", "vanilla")
        if policy_loss_mode != "vanilla":
            raise NotImplementedError(
                "Masked token-reward distillation currently supports only "
                f"actor.policy_loss.loss_mode='vanilla', got {policy_loss_mode!r}."
            )
        if actor_config.loss_agg_mode != "token-mean":
            raise NotImplementedError(
                "Masked token-reward distillation currently supports only "
                f"actor.loss_agg_mode='token-mean', got {actor_config.loss_agg_mode!r}."
            )
        return True

    def _token_reward_direct_topk(self) -> int:
        if not self._uses_token_reward_direct():
            return 0
        if getattr(self.distillation_config, "use_actor_loss", True):
            raise ValueError(
                "ADV_ESTIMATOR=token_reward_direct requires distillation.use_actor_loss=False so the teacher "
                "signal is applied through token rewards instead of the actor-side distillation loss."
            )
        loss_config = self.distillation_config.distillation_loss
        supported_strategies = {"only_stu", "only_stu_rethinking_style", "only_stu_teacher_topk"}
        if loss_config.top_k_strategy not in supported_strategies:
            raise NotImplementedError(
                "ADV_ESTIMATOR=token_reward_direct supports top-k strategies "
                f"{sorted(supported_strategies)}, got {loss_config.top_k_strategy!r}."
            )
        if loss_config.topk is None or int(loss_config.topk) <= 0:
            raise ValueError("ADV_ESTIMATOR=token_reward_direct requires distillation.distillation_loss.topk > 0.")
        return int(loss_config.topk)

    def _extract_teacher_response_logprobs_for_diagnostics(self, batch: DataProto) -> torch.Tensor | None:
        teacher_logprobs = batch.batch.get("teacher_sampled_logprobs", None)
        context = "teacher_sampled_logprobs"
        if teacher_logprobs is None:
            teacher_logprobs = batch.batch.get("teacher_logprobs", None)
            context = "teacher_logprobs"
        if teacher_logprobs is None:
            return None
        return extract_sampled_response_logprobs(
            sequence_logprobs=teacher_logprobs,
            response_mask=batch.batch["response_mask"],
            prompt_width=batch.batch["prompts"].shape[1],
            context=context,
        )

    def _extract_teacher_response_topk_for_diagnostics(
        self,
        batch: DataProto,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        teacher_ids = batch.batch.get("teacher_ids", None)
        teacher_logprobs = batch.batch.get("teacher_logprobs", None)
        if teacher_ids is None or teacher_logprobs is None:
            return None, None
        if teacher_ids.ndim != 3 or teacher_ids.shape[-1] <= 1:
            return None, None
        return extract_response_topk_scores(
            sequence_ids=teacher_ids,
            sequence_logprobs=teacher_logprobs,
            response_mask=batch.batch["response_mask"],
            prompt_width=batch.batch["prompts"].shape[1],
            context="teacher_logprobs",
        )

    def _compute_rethinking_teacher_on_student_topk(self, batch: DataProto) -> torch.Tensor:
        from verl.experimental.teacher_loop.teacher_manager import AsyncTeacherLLMServerManager

        target_scorer = getattr(self, "_rethinking_teacher_target_scorer", None)
        if target_scorer is None:
            if self.teacher_model_manager is None:
                raise RuntimeError(
                    "only_stu_rethinking_style requires an initialized distillation teacher model manager."
                )
            target_scorer = AsyncTeacherLLMServerManager(
                config=self.config,
                teacher_client=self.teacher_model_manager.get_client(),
            )
            self._rethinking_teacher_target_scorer = target_scorer

        teacher_key = self.distillation_config.teacher_key
        routing_values = batch.non_tensor_batch.get(teacher_key)
        routing_keys = None if routing_values is None else list(routing_values)
        raw_multi_modal_data = batch.non_tensor_batch.get("multi_modal_data")
        raw_mm_processor_kwargs = batch.non_tensor_batch.get("mm_processor_kwargs")
        requests = build_teacher_target_scoring_requests(
            input_ids=batch.batch["input_ids"],
            attention_mask=batch.batch["attention_mask"],
            student_topk_ids=batch.batch["student_topk_ids"],
            prompt_width=batch.batch["prompts"].shape[1],
            routing_keys=routing_keys,
            multi_modal_data=None if raw_multi_modal_data is None else list(raw_multi_modal_data),
            mm_processor_kwargs=None if raw_mm_processor_kwargs is None else list(raw_mm_processor_kwargs),
        )
        scoring_results = _execute_teacher_target_scoring_requests(
            target_scorer,
            requests,
            request_id_prefix=f"only-stu-rethinking-step-{getattr(self, 'global_steps', -1)}",
        )
        return assemble_teacher_target_logprobs(
            requests=requests,
            scoring_results=scoring_results,
            student_topk_ids=batch.batch["student_topk_ids"],
        )

    def _compute_token_reward_direct_scores(self, batch: DataProto) -> tuple[torch.Tensor, dict[str, float]]:
        topk = self._token_reward_direct_topk()
        missing_keys = {
            key for key in ("student_topk_ids", "student_topk_logprobs", "response_mask") if key not in batch.batch
        }
        if missing_keys:
            raise RuntimeError(
                "ADV_ESTIMATOR=token_reward_direct requires student top-k logprobs from actor recomputation, "
                f"but batch is missing {sorted(missing_keys)}. Make sure rollout_correction.bypass_mode=False."
            )

        loss_config = self.distillation_config.distillation_loss
        if loss_config.top_k_strategy == "only_stu":
            teacher_topk_ids, teacher_topk_logprobs = self._extract_teacher_response_topk_for_diagnostics(batch)
            if teacher_topk_ids is None or teacher_topk_logprobs is None:
                raise RuntimeError(
                    "ADV_ESTIMATOR=token_reward_direct with legacy only_stu requires teacher top-k prompt "
                    "logprobs. Check distillation.distillation_loss.topk and teacher inference settings."
                )
            token_rewards, reward_metrics = compute_only_student_topk_token_reward(
                student_topk_ids=batch.batch["student_topk_ids"],
                student_topk_logprobs=batch.batch["student_topk_logprobs"],
                teacher_topk_ids=teacher_topk_ids,
                teacher_topk_logprobs=teacher_topk_logprobs,
                response_mask=batch.batch["response_mask"],
                reward_mask=batch.batch.get("distillation_loss_mask", batch.batch["response_mask"]),
                topk=topk,
                reward_weight_mode=loss_config.reward_weight_mode,
                log_prob_min_clamp=loss_config.log_prob_min_clamp,
            )
        elif loss_config.top_k_strategy == "only_stu_teacher_topk":
            teacher_topk_ids, teacher_topk_logprobs = self._extract_teacher_response_topk_for_diagnostics(batch)
            if teacher_topk_ids is None or teacher_topk_logprobs is None:
                raise RuntimeError(
                    "ADV_ESTIMATOR=token_reward_direct with only_stu_teacher_topk requires expanded teacher "
                    "top-k prompt logprobs. Check distillation.distillation_loss.teacher_topk and teacher "
                    "inference settings."
                )
            diagnostics_config = self._distillation_diagnostics_config()
            position_bin_size = None
            if diagnostics_config is not None and self.global_steps % diagnostics_config.topk_log_interval == 0:
                position_bin_size = diagnostics_config.position_bin_size
            token_rewards, reward_metrics = compute_only_student_teacher_topk_token_reward(
                student_topk_ids=batch.batch["student_topk_ids"],
                student_topk_logprobs=batch.batch["student_topk_logprobs"],
                teacher_topk_ids=teacher_topk_ids,
                teacher_topk_logprobs=teacher_topk_logprobs,
                response_mask=batch.batch["response_mask"],
                reward_mask=batch.batch.get("distillation_loss_mask", batch.batch["response_mask"]),
                topk=topk,
                teacher_topk=loss_config.teacher_topk,
                reward_weight_mode=loss_config.reward_weight_mode,
                log_prob_min_clamp=loss_config.log_prob_min_clamp,
                position_bin_size=position_bin_size,
            )
        elif loss_config.top_k_strategy == "only_stu_rethinking_style":
            teacher_on_student_topk_logprobs = self._compute_rethinking_teacher_on_student_topk(batch)
            token_rewards, reward_metrics = compute_rethinking_only_student_topk_token_reward(
                student_topk_logprobs=batch.batch["student_topk_logprobs"],
                teacher_on_student_topk_logprobs=teacher_on_student_topk_logprobs,
                response_mask=batch.batch["response_mask"],
                reward_mask=batch.batch.get("distillation_loss_mask", batch.batch["response_mask"]),
                topk=topk,
                reward_weight_mode=loss_config.reward_weight_mode,
                log_prob_min_clamp=loss_config.log_prob_min_clamp,
            )
        else:
            raise AssertionError(f"Unexpected top_k_strategy after validation: {loss_config.top_k_strategy!r}.")
        return token_rewards, reward_metrics

    @staticmethod
    def _non_tensor_to_list(value) -> list[Any]:
        if value is None:
            return []
        if hasattr(value, "tolist"):
            return value.tolist()
        return list(value)

    def _decode_response_texts_for_diagnostics(
        self,
        responses: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> list[str]:
        response_texts = []
        response_mask = response_mask.detach().bool().cpu()
        responses = responses.detach().cpu()
        for response_ids, mask in zip(responses, response_mask, strict=True):
            response_texts.append(self.tokenizer.decode(response_ids[mask].tolist(), skip_special_tokens=True))
        return response_texts

    def _maybe_log_opd_heatmaps(
        self,
        *,
        global_steps: int,
        entropy: torch.Tensor | None,
        student_logprobs: torch.Tensor,
        teacher_logprobs: torch.Tensor | None,
        teacher_logprob_nonfinite_positions: list[Any] | None,
        response_mask: torch.Tensor,
    ) -> dict[str, Any]:
        diagnostics_config = self._distillation_diagnostics_config()
        if diagnostics_config is None or not diagnostics_config.log_wandb_heatmap:
            return {}
        if global_steps % diagnostics_config.heatmap_interval != 0:
            return {}
        if getattr(self, "logger", None) is None or "wandb" not in getattr(self.logger, "logger", {}):
            return {}

        import matplotlib.pyplot as plt
        import wandb

        metrics: dict[str, Any] = {}

        if entropy is not None:
            if self.opd_entropy_tracker is None:
                self.opd_entropy_tracker = OpdDiagnosticsTracker(
                    history_steps=diagnostics_config.heatmap_history_steps,
                    bin_size=diagnostics_config.position_bin_size,
                )
            entropy_row = compute_entropy_heatmap_row(
                entropy=entropy,
                response_mask=response_mask,
                bin_size=diagnostics_config.position_bin_size,
            )
            self.opd_entropy_tracker.add_entropy_row(global_steps, entropy_row)
            fig = self.opd_entropy_tracker.render_entropy_heatmap()
            if fig is not None:
                try:
                    metrics["opd/entropy_heatmap"] = wandb.Image(fig)
                finally:
                    plt.close(fig)

        if teacher_logprobs is not None:
            if self.opd_gap_tracker is None:
                self.opd_gap_tracker = OpdDiagnosticsTracker(
                    history_steps=diagnostics_config.heatmap_history_steps,
                    bin_size=diagnostics_config.position_bin_size,
                )
            gap_row = compute_gap_heatmap_row(
                student_logprobs=student_logprobs,
                teacher_logprobs=teacher_logprobs,
                response_mask=response_mask,
                bin_size=diagnostics_config.position_bin_size,
            )
            self.opd_gap_tracker.add_gap_row(global_steps, gap_row)
            fig = self.opd_gap_tracker.render_gap_heatmap()
            if fig is not None:
                try:
                    metrics["opd/gap_abs_heatmap"] = wandb.Image(fig)
                finally:
                    plt.close(fig)

            if self.opd_gap_high_tracker is None:
                self.opd_gap_high_tracker = OpdDiagnosticsTracker(
                    history_steps=diagnostics_config.heatmap_history_steps,
                    bin_size=diagnostics_config.position_bin_size,
                )
            gap_high_row = compute_gap_high_heatmap_row(
                student_logprobs=student_logprobs,
                teacher_logprobs=teacher_logprobs,
                response_mask=response_mask,
                bin_size=diagnostics_config.position_bin_size,
                high_gap_threshold=diagnostics_config.high_gap_threshold,
            )
            self.opd_gap_high_tracker.add_gap_high_row(global_steps, gap_high_row)
            fig = self.opd_gap_high_tracker.render_gap_high_heatmap()
            if fig is not None:
                try:
                    metrics["opd/gap_abs_high_heatmap"] = wandb.Image(fig)
                finally:
                    plt.close(fig)

        if teacher_logprob_nonfinite_positions is not None:
            if self.opd_teacher_logprob_nonfinite_tracker is None:
                self.opd_teacher_logprob_nonfinite_tracker = OpdDiagnosticsTracker(
                    history_steps=diagnostics_config.heatmap_history_steps,
                    bin_size=diagnostics_config.position_bin_size,
                )
            nonfinite_row = compute_teacher_logprob_nonfinite_heatmap_row(
                teacher_logprob_nonfinite_positions=teacher_logprob_nonfinite_positions,
                response_mask=response_mask,
                bin_size=diagnostics_config.position_bin_size,
            )
            self.opd_teacher_logprob_nonfinite_tracker.add_teacher_logprob_nonfinite_row(global_steps, nonfinite_row)
            fig = self.opd_teacher_logprob_nonfinite_tracker.render_teacher_logprob_nonfinite_heatmap()
            if fig is not None:
                try:
                    metrics["opd/teacher_logprob_nonfinite_heatmap"] = wandb.Image(fig)
                finally:
                    plt.close(fig)

        return metrics

    def _build_opd_diagnostics_inputs(
        self,
        *,
        batch: DataProto,
        entropy: torch.Tensor | None,
        global_steps: int,
    ) -> (
        tuple[
            dict[str, Any],
            dict[str, Any] | None,
            dict[str, Any] | None,
            torch.Tensor | None,
            list[Any] | None,
        ]
        | None
    ):
        diagnostics_config = self._distillation_diagnostics_config()
        if (
            diagnostics_config is None
            or not is_distillation_enabled(self.config.get("distillation"))
            or global_steps % diagnostics_config.log_interval != 0
        ):
            return None

        required_keys = {"old_log_probs", "response_mask"}
        missing_keys = required_keys - set(batch.batch.keys())
        if missing_keys:
            raise RuntimeError(f"OPD diagnostics missing required batch tensors: {sorted(missing_keys)}.")

        response_mask = batch.batch["response_mask"]
        student_logprobs = batch.batch["old_log_probs"]
        if student_logprobs.shape != response_mask.shape:
            raise RuntimeError(
                "OPD diagnostics expected old_log_probs and response_mask to align, "
                f"got old_log_probs.shape={tuple(student_logprobs.shape)} and "
                f"response_mask.shape={tuple(response_mask.shape)}."
            )
        if entropy is not None and entropy.shape != response_mask.shape:
            raise RuntimeError(
                "OPD diagnostics expected entropy and response_mask to align, "
                f"got entropy.shape={tuple(entropy.shape)} and response_mask.shape={tuple(response_mask.shape)}."
            )

        collect_topk_diagnostics = (
            diagnostics_config.collect_topk_overlap
            or getattr(diagnostics_config, "collect_rethinking_opd_metrics", False)
        ) and global_steps % diagnostics_config.topk_log_interval == 0
        collect_boundary_diagnostics = bool(
            diagnostics_config.collect_intervention_metrics
            and getattr(diagnostics_config, "collect_intervention_boundary_metrics", False)
        )
        teacher_response_logprobs = self._extract_teacher_response_logprobs_for_diagnostics(batch)
        teacher_response_topk_ids, teacher_response_topk_logprobs = (
            self._extract_teacher_response_topk_for_diagnostics(batch) if collect_topk_diagnostics else (None, None)
        )
        teacher_logprob_nonfinite_positions = None
        if "teacher_logprob_nonfinite_positions" in batch.non_tensor_batch:
            teacher_logprob_nonfinite_positions = self._non_tensor_to_list(
                batch.non_tensor_batch["teacher_logprob_nonfinite_positions"]
            )
        response_texts = None
        if getattr(diagnostics_config, "collect_train_text_zlib_repetition", False) and "responses" in batch.batch:
            response_texts = self._decode_response_texts_for_diagnostics(batch.batch["responses"], response_mask)
        diagnostics_kwargs = {
            "student_logprobs": student_logprobs,
            "response_ids": batch.batch.get("responses", None),
            "response_texts": response_texts,
            "teacher_logprobs": teacher_response_logprobs,
            "student_topk_ids": (
                batch.batch.get("student_topk_ids", None)
                if collect_topk_diagnostics or collect_boundary_diagnostics
                else None
            ),
            "student_topk_logprobs": batch.batch.get("student_topk_logprobs", None)
            if collect_topk_diagnostics
            else None,
            "teacher_topk_ids": teacher_response_topk_ids,
            "teacher_topk_logprobs": teacher_response_topk_logprobs,
            "teacher_logprob_nonfinite_positions": teacher_logprob_nonfinite_positions,
            "entropy": entropy,
            "response_mask": response_mask,
            "rm_scores": batch.batch.get("rm_scores", None),
            "bin_size": diagnostics_config.position_bin_size,
            "high_gap_threshold": diagnostics_config.high_gap_threshold,
            "overlap_topk": (
                int(diagnostics_config.topk or self.distillation_config.distillation_loss.topk or 64)
                if collect_topk_diagnostics
                else None
            ),
            "special_token_ids": self.opd_special_token_ids,
            "max_response_length": batch.batch["responses"].shape[-1] if "responses" in batch.batch else None,
            "collect_repetition_metrics": diagnostics_config.collect_repetition_metrics,
            "collect_train_text_zlib_repetition": diagnostics_config.collect_train_text_zlib_repetition,
            "repetition_tail_chars": diagnostics_config.repetition_tail_chars,
            "repetition_comp_ratio_threshold": diagnostics_config.repetition_comp_ratio_threshold,
            "repetition_tail_token_window": diagnostics_config.repetition_tail_token_window,
            "bad_tail_min_position": diagnostics_config.bad_tail_min_position,
            "bad_tail_deep_position": diagnostics_config.bad_tail_deep_position,
            "bad_tail_extreme_position": diagnostics_config.bad_tail_extreme_position,
            "bad_tail_overlap_threshold": diagnostics_config.bad_tail_overlap_threshold,
            "bad_tail_teacher_prob_threshold": diagnostics_config.bad_tail_teacher_prob_threshold,
            "bad_tail_gap_threshold": diagnostics_config.bad_tail_gap_threshold,
            "bad_tail_reward_threshold": diagnostics_config.bad_tail_reward_threshold,
            "bad_tail_entropy_threshold": diagnostics_config.bad_tail_entropy_threshold,
        }
        intervention_kwargs = None
        boundary_kwargs = None

        if diagnostics_config.collect_intervention_metrics:
            intervention_config = getattr(self.distillation_config, "intervention", None)
            intervention_enabled = intervention_config is not None and intervention_config.enabled
            has_intervention_fields = (
                "teacher_intervention_count" in batch.non_tensor_batch
                and "teacher_interventions" in batch.non_tensor_batch
                and "teacher_intervention_skipped" in batch.non_tensor_batch
            )
            future_gap_enabled = bool(
                intervention_enabled and getattr(intervention_config, "confirmation_mode", "none") == "future_gap"
            )
            has_future_gap_field = "teacher_intervention_proposals" in batch.non_tensor_batch
            if intervention_enabled and not has_intervention_fields:
                raise RuntimeError(
                    "OPD intervention diagnostics are enabled, but rollout output did not include "
                    "'teacher_intervention_count', 'teacher_interventions', and 'teacher_intervention_skipped'. "
                    "Check that the DMPD agent-loop intervention path is active for every training sample."
                )
            if future_gap_enabled and not has_future_gap_field:
                raise RuntimeError(
                    "future_gap intervention diagnostics require 'teacher_intervention_proposals' from every "
                    "rollout sample. Check that the updated DMPD agent-loop path is active."
                )
            if has_intervention_fields:
                intervention_kwargs = {
                    "intervention_counts": self._non_tensor_to_list(
                        batch.non_tensor_batch["teacher_intervention_count"]
                    ),
                    "interventions": self._non_tensor_to_list(batch.non_tensor_batch["teacher_interventions"]),
                    "intervention_proposals": (
                        self._non_tensor_to_list(batch.non_tensor_batch["teacher_intervention_proposals"])
                        if has_future_gap_field
                        else None
                    ),
                    "intervention_skipped": self._non_tensor_to_list(
                        batch.non_tensor_batch["teacher_intervention_skipped"]
                    ),
                    "intervention_outcomes": (
                        self._non_tensor_to_list(batch.non_tensor_batch["teacher_intervention_outcome"])
                        if "teacher_intervention_outcome" in batch.non_tensor_batch
                        else None
                    ),
                    "response_mask": response_mask,
                }

                if collect_boundary_diagnostics:
                    if diagnostics_kwargs["response_ids"] is None:
                        raise RuntimeError(
                            "Repair-boundary diagnostics require the existing response token tensor, but "
                            "batch.batch['responses'] was unavailable."
                        )
                    boundary_kwargs = {
                        "global_step": int(global_steps),
                        "boundary_text_cases": (
                            self._non_tensor_to_list(batch.non_tensor_batch["teacher_intervention_boundary_cases"])
                            if "teacher_intervention_boundary_cases" in batch.non_tensor_batch
                            else None
                        ),
                        "sample_uids": (
                            self._non_tensor_to_list(batch.non_tensor_batch["uid"])
                            if "uid" in batch.non_tensor_batch
                            else None
                        ),
                        "data_sources": (
                            self._non_tensor_to_list(batch.non_tensor_batch["data_source"])
                            if "data_source" in batch.non_tensor_batch
                            else None
                        ),
                        "gap_metric": str(intervention_config.gap_metric),
                        "teacher_rejection_token_cap": float(intervention_config.teacher_rejection_token_cap),
                        "student_teacher_span_scores_available": entropy is not None,
                    }

        return (
            diagnostics_kwargs,
            intervention_kwargs,
            boundary_kwargs,
            teacher_response_logprobs,
            teacher_logprob_nonfinite_positions,
        )

    def _compute_opd_diagnostics_metrics(
        self,
        *,
        batch: DataProto,
        entropy: torch.Tensor | None,
        global_steps: int,
    ) -> dict[str, Any]:
        diagnostics_inputs = self._build_opd_diagnostics_inputs(
            batch=batch,
            entropy=entropy,
            global_steps=global_steps,
        )
        if diagnostics_inputs is None:
            return {}

        (
            diagnostics_kwargs,
            intervention_kwargs,
            boundary_kwargs,
            teacher_response_logprobs,
            teacher_logprob_nonfinite_positions,
        ) = diagnostics_inputs
        if boundary_kwargs is not None:
            diagnostics_snapshot, intervention_snapshot, boundary_snapshot = _snapshot_diagnostics_payloads(
                diagnostics_kwargs,
                intervention_kwargs,
                boundary_kwargs,
            )
            if diagnostics_snapshot is None:
                raise RuntimeError("OPD diagnostics snapshot unexpectedly dropped the required diagnostics payload.")
            diagnostics_kwargs = diagnostics_snapshot
            intervention_kwargs = intervention_snapshot
            boundary_kwargs = boundary_snapshot
            teacher_response_logprobs = diagnostics_kwargs["teacher_logprobs"]
            teacher_logprob_nonfinite_positions = diagnostics_kwargs["teacher_logprob_nonfinite_positions"]
        metrics = compute_opd_diagnostics_metrics(**diagnostics_kwargs)
        metrics["opd/gap_abs/available"] = 1.0 if teacher_response_logprobs is not None else 0.0
        if intervention_kwargs is not None:
            metrics.update(compute_intervention_metrics(**intervention_kwargs))
        if boundary_kwargs is not None and intervention_kwargs is not None:
            boundary_metrics, records = _compute_intervention_boundary_from_payloads(
                diagnostics_kwargs,
                intervention_kwargs,
                boundary_kwargs,
            )
            metrics.update(boundary_metrics)
            self._queue_event_records_archive(
                metric_key="opd_intervention/repair_boundary_records",
                records=records,
            )
        if intervention_kwargs is not None:
            future_gap_records = build_future_gap_proposal_records(intervention_kwargs.get("intervention_proposals"))
            if future_gap_records:
                self._queue_event_records_archive(
                    metric_key="opd_intervention/future_gap_records",
                    records=future_gap_records,
                )

        metrics.update(
            self._maybe_log_opd_heatmaps(
                global_steps=global_steps,
                entropy=diagnostics_kwargs["entropy"],
                student_logprobs=diagnostics_kwargs["student_logprobs"],
                teacher_logprobs=teacher_response_logprobs,
                teacher_logprob_nonfinite_positions=teacher_logprob_nonfinite_positions,
                response_mask=diagnostics_kwargs["response_mask"],
            )
        )
        return metrics

    def _ensure_opd_diagnostics_executor(self) -> ThreadPoolExecutor:
        if self._opd_diagnostics_executor is None:
            self._opd_diagnostics_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="opd-diagnostics")
        return self._opd_diagnostics_executor

    def _finish_opd_diagnostics_for_step(
        self,
        task: _PendingOpdDiagnostics | None,
        *,
        expected_step: int,
    ) -> dict[str, float]:
        if task is None:
            return {}
        if task.source_step != int(expected_step):
            raise RuntimeError(
                "Async OPD diagnostics source step does not match the step being committed: "
                f"source_step={task.source_step}, expected_step={expected_step}."
            )
        active_task = getattr(self, "_opd_diagnostics_task", None)
        if active_task is not task:
            raise RuntimeError("Async OPD diagnostics task is not the trainer's active task.")

        join_started_at = time.perf_counter()
        try:
            raw_result = task.future.result()
        except Exception as exc:
            raise RuntimeError(f"Async OPD diagnostics failed for source step {task.source_step}.") from exc
        finally:
            self._opd_diagnostics_task = None

        finished_at = time.perf_counter()
        result = (
            raw_result
            if isinstance(raw_result, _OpdDiagnosticsResult)
            else _OpdDiagnosticsResult(metrics=dict(raw_result))
        )
        metrics = dict(result.metrics)
        self._queue_event_records_archive(
            metric_key="opd_intervention/repair_boundary_records",
            records=list(result.intervention_records),
        )
        if result.future_gap_records:
            self._queue_event_records_archive(
                metric_key="opd_intervention/future_gap_records",
                records=list(result.future_gap_records),
            )
        metrics.update(
            {
                "opd/diagnostics_async/completed": 1.0,
                "opd/diagnostics_async/failed": 0.0,
                "opd/diagnostics_async/pending": 0.0,
                "opd/diagnostics_async/source_step": float(task.source_step),
                "opd/diagnostics_async/result_delay_steps": 0.0,
                "opd/diagnostics_async/wall_seconds": finished_at - task.submitted_at,
                "opd/diagnostics_async/join_wait_seconds": finished_at - join_started_at,
            }
        )
        return metrics

    def _shutdown_opd_diagnostics_executor(self) -> None:
        if self._opd_diagnostics_executor is not None:
            self._opd_diagnostics_executor.shutdown(wait=True)
            self._opd_diagnostics_executor = None
        self._opd_diagnostics_task = None

    def _submit_or_compute_opd_diagnostics_metrics(
        self,
        *,
        batch: DataProto,
        entropy: torch.Tensor | None,
        global_steps: int,
    ) -> tuple[dict[str, Any], _PendingOpdDiagnostics | None]:
        diagnostics_config = self._distillation_diagnostics_config()
        if diagnostics_config is None or not getattr(diagnostics_config, "async_metrics", False):
            return self._compute_opd_diagnostics_metrics(batch=batch, entropy=entropy, global_steps=global_steps), None

        diagnostics_inputs = self._build_opd_diagnostics_inputs(
            batch=batch,
            entropy=entropy,
            global_steps=global_steps,
        )
        if diagnostics_inputs is None:
            return {}, None

        if getattr(self, "_opd_diagnostics_task", None) is not None:
            active_step = self._opd_diagnostics_task.source_step
            raise RuntimeError(
                "Cannot submit OPD diagnostics while another source step is pending: "
                f"active_source_step={active_step}, requested_source_step={global_steps}."
            )

        (
            diagnostics_kwargs,
            intervention_kwargs,
            boundary_kwargs,
            teacher_response_logprobs,
            teacher_logprob_nonfinite_positions,
        ) = diagnostics_inputs
        metrics = self._maybe_log_opd_heatmaps(
            global_steps=global_steps,
            entropy=diagnostics_kwargs["entropy"],
            student_logprobs=diagnostics_kwargs["student_logprobs"],
            teacher_logprobs=teacher_response_logprobs,
            teacher_logprob_nonfinite_positions=teacher_logprob_nonfinite_positions,
            response_mask=diagnostics_kwargs["response_mask"],
        )
        metrics["opd/gap_abs/available"] = 1.0 if teacher_response_logprobs is not None else 0.0

        snapshot_start = time.perf_counter()
        snapshot_kwargs, snapshot_intervention_kwargs, snapshot_boundary_kwargs = _snapshot_diagnostics_payloads(
            diagnostics_kwargs,
            intervention_kwargs,
            boundary_kwargs,
        )
        if snapshot_kwargs is None:
            raise RuntimeError("OPD diagnostics snapshot unexpectedly dropped the required diagnostics payload.")
        snapshot_seconds = time.perf_counter() - snapshot_start
        submitted_at = time.perf_counter()
        future = self._ensure_opd_diagnostics_executor().submit(
            _compute_opd_diagnostics_background_metrics,
            snapshot_kwargs,
            snapshot_intervention_kwargs,
            snapshot_boundary_kwargs,
        )
        task = _PendingOpdDiagnostics(
            source_step=int(global_steps),
            future=future,
            submitted_at=submitted_at,
        )
        self._opd_diagnostics_task = task
        metrics.update(
            {
                "opd/diagnostics_async/submitted": 1.0,
                "opd/diagnostics_async/skipped_pending_full": 0.0,
                "opd/diagnostics_async/pending_after_submit": 1.0,
                "opd/diagnostics_async/snapshot_seconds": snapshot_seconds,
            }
        )
        return metrics, task

    def _update_critic(self, batch: DataProto) -> DataProto:
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to no-padding
        batch_td = left_right_2_no_padding(batch_td)
        ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
        ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        ppo_epochs = self.config.critic.ppo_epochs
        seed = self.config.critic.data_loader_seed
        shuffle = self.config.critic.shuffle
        tu.assign_non_tensor(
            batch_td,
            global_batch_size=ppo_mini_batch_size,
            mini_batch_size=ppo_mini_batch_size,
            epochs=ppo_epochs,
            seed=seed,
            dataloader_kwargs={"shuffle": shuffle},
        )

        output = self.critic_wg.train_mini_batch(batch_td)
        output = output.get()
        output = tu.get(output, "metrics")
        output = rename_dict(output, "critic/")
        # modify key name
        output["perf/mfu/critic"] = output.pop("critic/mfu")
        critic_output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
        return critic_output

    def _uses_wandb_backend(self) -> bool:
        logger_config = self.config.trainer.logger
        backends = [logger_config] if isinstance(logger_config, str) else list(logger_config)
        return "wandb" in backends or "tracking" in backends

    def _initialize_metric_logging(self) -> None:
        if self._metric_logging_settings.mode != "balanced" or self._metric_archive_writer is not None:
            return

        from verl.trainer.ppo.metric_archive import (
            METRIC_COMMIT_FILENAME,
            MetricArchiveWriter,
            MetricRouter,
            inspect_metric_archive,
        )

        resume_context = self._resolve_resume_context()
        archive_config = self._metric_logging_settings.archive
        archive_root = archive_config.local_dir
        if archive_root is None:
            archive_root = os.path.join(
                os.path.abspath(self.config.trainer.default_local_dir),
                "diagnostics",
            )
        archive_root = os.path.abspath(os.path.expanduser(str(archive_root)))
        existing_manifest = inspect_metric_archive(archive_root)
        requested_wandb_mode = (
            _normalize_wandb_mode(os.environ.get("WANDB_MODE")) if self._uses_wandb_backend() else "disabled"
        )
        if existing_manifest is not None and requested_wandb_mode == "online":
            unsynced_sessions = _unsynced_offline_sessions(existing_manifest)
            if unsynced_sessions:
                commands = _offline_sync_commands(
                    existing_manifest,
                    entity=os.environ.get("WANDB_ENTITY"),
                    project=str(self.config.trainer.project_name),
                )
                raise RuntimeError(
                    "Cannot resume online while prior offline W&B sessions remain unsynced. "
                    f"sessions={[item['session_id'] for item in unsynced_sessions]}, "
                    f"sync_commands={commands}."
                )
        if (
            existing_manifest is not None
            and resume_context.is_resume
            and not os.path.isfile(os.path.join(resume_context.path, METRIC_COMMIT_FILENAME))
        ):
            legacy_before_step = existing_manifest.get("legacy_before_step")
            is_declared_legacy_checkpoint = legacy_before_step is not None and resume_context.step <= int(
                legacy_before_step
            )
            if not is_declared_legacy_checkpoint:
                raise RuntimeError(
                    "The selected checkpoint belongs to a metric-archive run but has no metrics commit marker: "
                    f"checkpoint={resume_context.path!r}. "
                    "The tracker must point to the last fully committed checkpoint."
                )

        environment_run_id = os.environ.get("WANDB_RUN_ID")
        manifest_run_id = (
            str(existing_manifest["logical_run_id"])
            if existing_manifest is not None and existing_manifest.get("logical_run_id")
            else None
        )
        logical_run_id = environment_run_id or manifest_run_id or uuid.uuid4().hex
        session_id = uuid.uuid4().hex
        archive_resume_step = (
            resume_context.step if resume_context.is_resume else (0 if existing_manifest is not None else None)
        )
        writer = MetricArchiveWriter(
            archive_root,
            logical_run_id=logical_run_id,
            project=str(self.config.trainer.project_name),
            experiment=str(self.config.trainer.experiment_name),
            session_id=session_id,
            resume_checkpoint_step=archive_resume_step,
            checkpoint_dir=resume_context.path,
            strict_identity=True,
        )
        try:
            writer.validate_wandb_transition(requested_wandb_mode)
        except BaseException:
            writer.close(interrupted=True)
            raise

        configured_topk = None
        distillation = self.config.get("distillation", None)
        if distillation is not None:
            diagnostics = distillation.get("diagnostics", None)
            loss_config = distillation.get("distillation_loss", None)
            if diagnostics is not None:
                configured_topk = diagnostics.get("topk", None)
            if configured_topk is None and loss_config is not None:
                configured_topk = loss_config.get("topk", None)
        self._logical_run_id = writer.logical_run_id
        self._metric_archive_session_id = session_id
        self._metric_resume_step = archive_resume_step
        self._metric_archive_writer = writer
        self._metric_router = MetricRouter(
            mode="balanced",
            configured_topk=int(configured_topk) if configured_topk is not None else None,
            live_key_warning_threshold=self._metric_logging_settings.live_key_warning_threshold,
        )

    def _build_wandb_resume_spec(self):
        if self._metric_logging_mode() != "balanced" or not self._uses_wandb_backend():
            return None

        from verl.utils.tracking import WandbResumeSpec

        resume_context = self._resolve_resume_context()
        resume_step = getattr(self, "_metric_resume_step", None)
        is_resume = resume_context.is_resume or resume_step is not None
        run_id = self._logical_run_id or os.environ.get("WANDB_RUN_ID")
        if not run_id:
            run_id = uuid.uuid4().hex
        self._logical_run_id = str(run_id)

        wandb_mode = _normalize_wandb_mode(os.environ.get("WANDB_MODE"))
        if not is_resume:
            return WandbResumeSpec(run_id=self._logical_run_id, resume="never")
        if wandb_mode in {"offline", "disabled"}:
            return WandbResumeSpec(run_id=self._logical_run_id, resume="must")

        try:
            remote_step = _query_wandb_remote_run_step(
                project=str(self.config.trainer.project_name),
                run_id=self._logical_run_id,
                entity=os.environ.get("WANDB_ENTITY"),
            )
        except Exception as exc:
            raise RuntimeError(
                "Strict online W&B resume could not verify the existing remote run. "
                f"project={self.config.trainer.project_name!r}, run_id={self._logical_run_id!r}, "
                f"checkpoint_step={resume_step if resume_step is not None else resume_context.step}."
            ) from exc

        checkpoint_step = resume_step if resume_step is not None else resume_context.step
        if remote_step <= checkpoint_step:
            return WandbResumeSpec(run_id=self._logical_run_id, resume="must")
        return WandbResumeSpec(
            run_id=self._logical_run_id,
            resume_from=f"{self._logical_run_id}?_step={checkpoint_step}",
        )

    def _recover_missing_offline_wandb_history(self, logger) -> None:
        if self._metric_archive_writer is None:
            return
        run_info = logger.wandb_run_info
        resume_context = self._resolve_resume_context()
        resume_step = getattr(self, "_metric_resume_step", None)
        if run_info is None or not run_info.is_offline or (not resume_context.is_resume and resume_step is None):
            return
        events = self._metric_archive_writer.unrecorded_offline_committed_live_events(
            checkpoint_step=resume_step if resume_step is not None else resume_context.step
        )
        if not events:
            return

        import wandb

        ordered_events = sorted(events, key=lambda item: (item.global_step, item.session_id, item.event_seq))
        current_step = None
        current_event_ids: list[str] = []
        recovered_source_session_ids: set[str] = set()
        for event in ordered_events:
            if current_step is not None and event.global_step != current_step:
                self._metric_archive_writer.record_wandb_committed_range(
                    step_start=current_step,
                    step_end=current_step,
                    recovered_event_ids=tuple(current_event_ids),
                )
                current_event_ids = []
            metrics = dict(event.metrics)
            for event_file in event.event_files:
                if event_file.metric_key in _OPD_HEATMAP_METRIC_KEYS and os.path.isfile(event_file.path):
                    metrics[event_file.metric_key] = wandb.Image(event_file.path)
            logger.log(data=metrics, step=event.global_step, backend=["wandb"])
            current_step = event.global_step
            current_event_ids.append(event.event_id)
            recovered_source_session_ids.add(event.session_id)
        assert current_step is not None
        self._metric_archive_writer.record_wandb_committed_range(
            step_start=current_step,
            step_end=current_step,
            recovered_event_ids=tuple(current_event_ids),
        )
        self._metric_archive_writer.mark_wandb_history_recovered(tuple(sorted(recovered_source_session_ids)))
        print(f"Recovered {len(ordered_events)} committed metric event(s) missing from prior offline W&B segments.")

    def init_logger(self):
        """Initialize and return the trainer logger."""
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        if getattr(self, "logger", None) is not None:
            return self.logger

        self._resolve_resume_context()
        logger = None
        try:
            self._initialize_metric_logging()
            wandb_resume = self._build_wandb_resume_spec()
            logger = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
                config=OmegaConf.to_container(self.config, resolve=True),
                wandb_resume=wandb_resume,
            )
            if self._metric_archive_writer is not None:
                run_info = logger.wandb_run_info
                self._metric_archive_writer.register_wandb_session(
                    mode=run_info.mode if run_info is not None else "disabled",
                    run_id=run_info.run_id if run_info is not None else None,
                    sync_dir=run_info.sync_dir if run_info is not None else None,
                    sync_file=run_info.sync_file if run_info is not None else None,
                )
                self._recover_missing_offline_wandb_history(logger)
                if (
                    self._metric_logging_settings.archive.upload_to_wandb == "auto"
                    and run_info is not None
                    and run_info.mode != "disabled"
                ):
                    from verl.trainer.ppo.metric_artifact import MetricArtifactUploader

                    self._metric_artifact_uploader = MetricArtifactUploader(
                        writer=self._metric_archive_writer,
                        tracking=logger,
                        project=str(self.config.trainer.project_name),
                        entity=os.environ.get("WANDB_ENTITY"),
                        finish_timeout_seconds=self._metric_logging_settings.archive.finish_timeout_seconds,
                    )
                    self._metric_artifact_uploader.submit_pending()
        except BaseException:
            if getattr(self, "_metric_artifact_uploader", None) is not None:
                self._metric_artifact_uploader.finish()
                self._metric_artifact_uploader = None
            if getattr(self, "_metric_archive_writer", None) is not None:
                self._metric_archive_writer.close(interrupted=True)
                self._metric_archive_writer = None
                self._metric_router = None
            if logger is not None:
                logger.finish(exit_code=1)
            raise
        self.logger = logger
        return logger

    def _archive_metric_event_files(self, metrics: dict[str, Any], *, global_step: int) -> None:
        if self._metric_archive_writer is None:
            return
        for metric_key in _OPD_HEATMAP_METRIC_KEYS & metrics.keys():
            media_path = getattr(metrics[metric_key], "_path", None)
            if media_path and os.path.isfile(media_path):
                self._metric_archive_writer.add_event_file(
                    media_path,
                    global_step=global_step,
                    metric_key=metric_key,
                )

    def _archive_pending_event_records(self, *, global_step: int) -> None:
        """Attach queued detailed records to the active metric shard."""

        pending = list(getattr(self, "_pending_event_record_archives", ()))
        if not pending:
            return
        stale = [item for item in pending if item.global_step < global_step]
        if stale:
            descriptions = [f"{item.metric_key}@{item.global_step}" for item in stale]
            raise RuntimeError(
                "Detailed archive records were not attached to their metric event before the trainer advanced: "
                f"current_step={global_step}, pending={descriptions}."
            )
        ready = [item for item in pending if item.global_step == global_step]
        future_gap_serialization_seconds = 0.0
        for item in ready:
            started_at = time.perf_counter()
            self._metric_archive_writer.add_event_records(
                item.records,
                global_step=global_step,
                metric_key=item.metric_key,
            )
            if item.metric_key == "opd_intervention/future_gap_records":
                future_gap_serialization_seconds += time.perf_counter() - started_at
        if future_gap_serialization_seconds:
            previous_value = getattr(self, "_future_gap_archive_serialization_seconds_pending", None)
            self._future_gap_archive_serialization_seconds_pending = future_gap_serialization_seconds + float(
                previous_value or 0.0
            )
        self._pending_event_record_archives = [item for item in pending if item.global_step > global_step]

    def _log_balanced_live_metrics(self, logger, live_metrics: dict[str, Any], *, step: int) -> None:
        run_info = logger.wandb_run_info
        if run_info is None or run_info.is_online:
            logger.log(data=live_metrics, step=step)
            return

        non_wandb_backends = [backend for backend in logger.logger if backend != "wandb"]
        if non_wandb_backends:
            logger.log(data=live_metrics, step=step, backend=non_wandb_backends)
        if run_info.is_offline:
            self._offline_live_metric_buffer.append((int(step), dict(live_metrics)))

    def _flush_offline_wandb_history(self, logger, *, checkpoint_step: int) -> None:
        run_info = logger.wandb_run_info
        if run_info is None or not run_info.is_offline:
            return
        committed = []
        remaining = []
        for step, metrics in self._offline_live_metric_buffer:
            if step <= checkpoint_step:
                committed.append((step, metrics))
            else:
                remaining.append((step, metrics))
        current_step = None
        for step, metrics in committed:
            if current_step is not None and step != current_step and self._metric_archive_writer is not None:
                self._metric_archive_writer.record_wandb_committed_range(
                    step_start=current_step,
                    step_end=current_step,
                )
            logger.log(data=metrics, step=step, backend=["wandb"])
            current_step = step
        if current_step is not None and self._metric_archive_writer is not None:
            self._metric_archive_writer.record_wandb_committed_range(
                step_start=current_step,
                step_end=current_step,
            )
        self._offline_live_metric_buffer = remaining

    def _record_metric_event(
        self,
        logger,
        metrics: dict[str, Any],
        *,
        event_kind: str,
        checkpoint_payload_path: str | None = None,
    ) -> None:
        if self._metric_logging_settings.mode == "full":
            logger.log(data=metrics, step=self.global_steps)
            return

        if self._metric_archive_writer is None or self._metric_router is None:
            raise RuntimeError("Balanced metric logging was not initialized before the first metric event.")
        if self._metric_artifact_uploader is not None:
            self._metric_artifact_uploader.drain_completed()

        routed_input = dict(metrics)
        routed_input.update(self._metric_archive_writer.health_metrics())
        future_gap_serialization_seconds = getattr(
            self,
            "_future_gap_archive_serialization_seconds_pending",
            None,
        )
        if future_gap_serialization_seconds is not None:
            routed_input["opd_intervention/future_gap/archive_serialization_cpu_seconds_previous_event"] = float(
                future_gap_serialization_seconds
            )
            self._future_gap_archive_serialization_seconds_pending = None
        if checkpoint_payload_path is not None:
            routed_input["metric_archive/last_committed_step"] = float(self.global_steps)
        route = self._metric_router.route(routed_input)
        self._metric_archive_writer.append_event(
            route,
            global_step=self.global_steps,
            event_kind=event_kind,
            checkpoint_step=self.global_steps if checkpoint_payload_path is not None else None,
        )
        self._archive_metric_event_files(route.live_metrics, global_step=self.global_steps)
        self._archive_pending_event_records(global_step=self.global_steps)

        if checkpoint_payload_path is not None:
            self._metric_archive_writer.commit_checkpoint(
                checkpoint_payload_path,
                checkpoint_step=self.global_steps,
            )
            self._commit_checkpoint_tracker(self.global_steps)
            self._log_balanced_live_metrics(logger, route.live_metrics, step=self.global_steps)
            self._flush_offline_wandb_history(logger, checkpoint_step=self.global_steps)
            if self._metric_artifact_uploader is not None:
                self._metric_artifact_uploader.submit_pending()
        elif (
            self.global_steps > 0
            and self.global_steps % self._metric_logging_settings.archive.shard_interval_steps == 0
        ):
            self._metric_archive_writer.close_shard()
            self._log_balanced_live_metrics(logger, route.live_metrics, step=self.global_steps)
        else:
            self._log_balanced_live_metrics(logger, route.live_metrics, step=self.global_steps)

    def _finalize_metric_logging(self, logger, *, interrupted: bool) -> None:
        writer = getattr(self, "_metric_archive_writer", None)
        if writer is None:
            return
        try:
            if not interrupted:
                writer.close_shard()
                if self._metric_artifact_uploader is not None:
                    self._metric_artifact_uploader.submit_pending()
            if self._metric_artifact_uploader is not None:
                self._metric_artifact_uploader.finish()
            if self._offline_live_metric_buffer:
                first_step = self._offline_live_metric_buffer[0][0]
                last_step = self._offline_live_metric_buffer[-1][0]
                print(
                    f"Leaving uncommitted offline W&B history out of the sync segment: steps={first_step}-{last_step}."
                )
            sync_commands = writer.standard_wandb_sync_commands(
                entity=os.environ.get("WANDB_ENTITY"),
                project=str(self.config.trainer.project_name),
            )
            if sync_commands:
                print("Offline W&B segments can be synchronized in manifest order with:")
                for command in sync_commands:
                    print(f"  {command}")
        finally:
            writer.close(interrupted=interrupted)

    def fit(self):
        """Run training with deterministic logger and diagnostics cleanup."""
        logger = self.init_logger()
        exit_code = 0
        try:
            return self._fit_impl(logger)
        except BaseException:
            exit_code = 1
            raise
        finally:
            self._shutdown_opd_diagnostics_executor()
            try:
                self._finalize_metric_logging(logger, interrupted=exit_code != 0)
            finally:
                logger.finish(exit_code=exit_code)

    def _fit_impl(self, logger):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        self.global_steps = 0

        # load checkpoint and update weights before doing anything
        self._load_checkpoint()
        self.checkpoint_manager.update_weights(self.global_steps)

        current_epoch = self.global_steps // len(self.train_dataloader)
        resumed_initial_evaluation = self.global_steps > 0 or getattr(self, "_metric_resume_step", None) is not None

        # Perform validation and optional benchmark testing before training.
        initial_metrics = {}
        if self.config.trainer.get("val_before_train", False):
            val_metrics = self._validate(
                include_format_metrics=True,
                suppress_tracking=resumed_initial_evaluation,
            )
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            initial_metrics.update(val_metrics)
            teacher_val_metrics = self._validate_teacher_before_train(
                suppress_tracking=resumed_initial_evaluation,
            )
            if teacher_val_metrics:
                pprint(f"Initial teacher validation metrics: {teacher_val_metrics}")
                initial_metrics.update(teacher_val_metrics)

        if self.config.trainer.get("test_before_train", False):
            test_benchmark_metrics = self._validate_test_benchmarks(
                suppress_tracking=resumed_initial_evaluation,
            )
            pprint(f"Initial test benchmark metrics: {test_benchmark_metrics}")
            initial_metrics.update(test_benchmark_metrics)

        if initial_metrics and not resumed_initial_evaluation:
            self._record_metric_event(
                logger,
                initial_metrics,
                event_kind="initial_validation",
            )
        elif initial_metrics:
            print(
                f"Resume validation at checkpoint step {self.global_steps} is intentionally excluded "
                "from realtime history and the metric ledger."
            )
        if self.config.trainer.get("val_only", False):
            if not initial_metrics:
                raise ValueError(
                    "trainer.val_only=True requires trainer.val_before_train or trainer.test_before_train."
                )
            if not resumed_initial_evaluation and self._metric_logging_settings.mode == "balanced":
                # A fresh explicit evaluation is a terminal event rather than
                # an uncheckpointed training tail, so preserve its offline
                # history even though no model checkpoint is created.
                self._flush_offline_wandb_history(logger, checkpoint_step=self.global_steps)
            return

        if self.config.actor_rollout_ref.rollout.skip.get("enable", False):
            rollout_skip = RolloutSkip(self.config, self.async_rollout_manager)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        last_test_benchmark_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}
                opd_entropy = None
                opd_diagnostics_task = None
                checkpoint_payload_path = None

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                rollout_n = self.config.actor_rollout_ref.rollout.n
                gen_batch_output = gen_batch.repeat(repeat_times=rollout_n, interleave=True)

                if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                    # NOTE: REMAX needs one sampled rollout plus one greedy baseline per prompt.
                    # Keep them in a single agent-loop/vLLM request to avoid sending a second
                    # rollout after replicas have been put to sleep, which can leave async vLLM
                    # engines in an invalid state for multi-turn agent workloads.
                    gen_batch_output.non_tensor_batch["__do_sample__"] = np.ones(len(gen_batch_output), dtype=bool)
                    gen_baseline_batch = gen_batch.slice(0, None)
                    gen_baseline_batch.non_tensor_batch["__do_sample__"] = np.zeros(len(gen_baseline_batch), dtype=bool)
                    combined_gen_batch = DataProto.concat([gen_batch_output, gen_baseline_batch])
                    num_sampled_prompts = len(gen_batch_output)
                else:
                    combined_gen_batch = gen_batch_output
                    num_sampled_prompts = len(gen_batch_output)

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if curr_step_profile:
                            self.llm_server_manager.start_profile()
                        combined_gen_output = self.async_rollout_manager.generate_sequences(combined_gen_batch)
                        self.checkpoint_manager.sleep_replicas()
                        if curr_step_profile:
                            self.llm_server_manager.stop_profile()

                        timing_raw.update(combined_gen_output.meta_info["timing"])
                        combined_gen_output.meta_info.pop("timing", None)

                    gen_batch_output = combined_gen_output.slice(0, num_sampled_prompts)
                    if "__do_sample__" in gen_batch_output.non_tensor_batch:
                        gen_batch_output.pop(non_tensor_batch_keys=["__do_sample__"])

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        gen_baseline_output = combined_gen_output.slice(num_sampled_prompts, None)
                        if "__do_sample__" in gen_baseline_output.non_tensor_batch:
                            gen_baseline_output.pop(non_tensor_batch_keys=["__do_sample__"])

                        if self.use_rm and "rm_scores" not in gen_baseline_output.batch.keys():
                            baseline_reward = self._compute_reward_colocate(gen_baseline_output)
                            gen_baseline_output = gen_baseline_output.union(baseline_reward)

                        reward_baseline_tensor = gen_baseline_output.batch["rm_scores"].sum(dim=-1)
                        batch.batch["reward_baselines"] = reward_baseline_tensor

                        del gen_baseline_output
                    del combined_gen_batch, combined_gen_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    distillation_enabled = self.distillation_config is not None and is_distillation_enabled(
                        self.config.get("distillation")
                    )
                    if distillation_enabled:
                        metrics.update(
                            apply_distillation_horizon_mask(
                                data=batch,
                                horizon_config=self.distillation_config.distillation_loss.horizon,
                                global_steps=self.global_steps,
                            )
                        )
                        metrics.update(
                            apply_intervention_prune_mask(
                                data=batch,
                                intervention_config=self.distillation_config.intervention,
                            )
                        )

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    # get images_seqlens
                    images_seqlens_all = []
                    for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
                        if "image_grid_thw" not in multi_modal_input.keys():
                            continue
                        images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                    batch.meta_info["images_seqlens"] = images_seqlens_all
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            batch_reward = self._compute_reward_colocate(batch)
                            batch = batch.union(batch_reward)

                        # extract reward_tensor and reward_extra_infos_dict for training
                        if "rm_scores" in batch.batch:
                            reward_tensor, reward_extra_infos_dict = extract_reward(batch)
                            metrics.update(_compute_reward_format_metrics(reward_extra_infos_dict))
                        elif self._uses_token_reward_direct():
                            reward_tensor = torch.zeros_like(batch.batch["response_mask"], dtype=torch.float32)
                            reward_extra_infos_dict = {}
                        else:
                            raise RuntimeError(
                                "Training reward is missing rm_scores. Enable a reward source or use "
                                "ADV_ESTIMATOR=token_reward_direct with distillation top-k rewards."
                            )

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    opd_entropy = None
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(
                                batch, global_steps=self.global_steps
                            )
                            entropys = old_log_prob.batch["entropys"]
                            opd_entropy = entropys.detach()
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            if "routed_experts" in batch.batch and "routed_experts" in old_log_prob.batch:
                                raise ValueError(
                                    "Detected conflicting router replay configuration: "
                                    "router_replay.mode='R2' and enable_rollout_routing_replay=True "
                                    "cannot be enabled simultaneously. "
                                    "The enable_rollout_routing_replay option is only used in R3 mode; "
                                    "it should not be set when using R2 mode."
                                )
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self._uses_token_reward_direct():
                            if self.config.algorithm.use_kl_in_reward:
                                raise NotImplementedError(
                                    "ADV_ESTIMATOR=token_reward_direct now keeps top-k rewards as [B,T,K], "
                                    "which is incompatible with sampled-token in-reward KL. This matches the "
                                    "OPD paper setting where KL coefficient is 0; set algorithm.use_kl_in_reward=False."
                                )
                            token_reward_tensor, token_reward_metrics = self._compute_token_reward_direct_scores(batch)
                            metrics.update(token_reward_metrics)
                            loss_config = self.distillation_config.distillation_loss
                            if loss_config.use_task_rewards:
                                intervention_outcomes = None
                                if bool(getattr(self.distillation_config.intervention, "enabled", False)):
                                    if "teacher_intervention_outcome" not in batch.non_tensor_batch:
                                        raise RuntimeError(
                                            "Joint token-reward OPD + RLVR requires teacher_intervention_outcome "
                                            "metadata when DMPD intervention is enabled."
                                        )
                                    intervention_outcomes = self._non_tensor_to_list(
                                        batch.non_tensor_batch["teacher_intervention_outcome"]
                                    )
                                rlvr_token_rewards, rlvr_response_mask, rlvr_metrics = (
                                    build_joint_rlvr_training_tensors(
                                        task_reward_tensor=reward_tensor,
                                        response_mask=batch.batch["response_mask"],
                                        intervention_outcomes=intervention_outcomes,
                                    )
                                )
                                batch.batch["rlvr_token_level_rewards"] = rlvr_token_rewards
                                batch.batch["rlvr_response_mask"] = rlvr_response_mask
                                metrics.update(rlvr_metrics)
                                response_tokens = batch.batch["response_mask"].sum().item()
                                active_rlvr_tokens = rlvr_response_mask.sum().item()
                                metrics["distillation/joint_rlvr/active_token_fraction"] = (
                                    float(active_rlvr_tokens) / float(response_tokens) if response_tokens > 0 else 0.0
                                )
                                metrics["distillation/token_reward_direct/combined_with_task_rewards"] = 1.0
                            else:
                                metrics["distillation/token_reward_direct/combined_with_task_rewards"] = 0.0
                            reward_tensor = token_reward_tensor
                            batch.batch["rm_scores"] = reward_tensor

                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        with marked_timer("opd_diagnostics_submit", timing_raw, color="green"):
                            diagnostics_metrics, opd_diagnostics_task = self._submit_or_compute_opd_diagnostics_metrics(
                                batch=batch, entropy=opd_entropy, global_steps=self.global_steps
                            )
                            metrics.update(diagnostics_metrics)
                            self._queue_training_intervention_outcome_archive(
                                batch=batch,
                                global_steps=self.global_steps,
                            )
                        for diagnostics_key in ("student_topk_ids", "student_topk_logprobs"):
                            if diagnostics_key in batch.batch and not self._uses_token_reward_direct():
                                batch.batch.pop(diagnostics_key)

                        # Compute rollout correction while old_log_probs is still the sampled-token
                        # [B, T] proximal-policy logprob. token_reward_direct later swaps old_log_probs
                        # to [B, T, K] top-k logprobs for the actor loss.
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        if self._uses_token_reward_direct():
                            missing_topk_training_keys = {
                                key for key in ("student_topk_ids", "student_topk_logprobs") if key not in batch.batch
                            }
                            if missing_topk_training_keys:
                                raise RuntimeError(
                                    "ADV_ESTIMATOR=token_reward_direct requires student top-k tensors through "
                                    f"actor update, but missing {sorted(missing_topk_training_keys)}."
                                )
                            if self._uses_joint_token_reward_rlvr():
                                batch.batch["rlvr_old_log_probs"] = batch.batch["old_log_probs"].clone()
                            batch.batch["old_log_probs"] = batch.batch["student_topk_logprobs"]

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )
                        if self._uses_joint_token_reward_rlvr():
                            rlvr_advantages, rlvr_returns = core_algos.compute_grpo_outcome_advantage(
                                token_level_rewards=batch.batch["rlvr_token_level_rewards"],
                                response_mask=batch.batch["rlvr_response_mask"],
                                index=batch.non_tensor_batch["uid"],
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            )
                            batch.batch["rlvr_advantages"] = rlvr_advantages
                            batch.batch["rlvr_returns"] = rlvr_returns
                            rlvr_active = batch.batch["rlvr_response_mask"].bool()
                            if rlvr_active.any():
                                metrics["distillation/joint_rlvr/adv_abs_mean"] = (
                                    rlvr_advantages[rlvr_active].abs().float().mean().item()
                                )
                            else:
                                metrics["distillation/joint_rlvr/adv_abs_mean"] = 0.0
                            opd_active = (
                                batch.batch["response_mask"].bool().unsqueeze(-1).expand_as(batch.batch["advantages"])
                            )
                            metrics["distillation/joint_rlvr/opd_adv_abs_mean"] = (
                                batch.batch["advantages"][opd_active].abs().float().mean().item()
                                if opd_active.any()
                                else 0.0
                            )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup > self.global_steps:
                        # Still in critic warmup, only update weights to wake up rollout replicas.
                        self.checkpoint_manager.update_weights(self.global_steps)
                    else:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)

                        # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                        esi_close_to_expiration = should_save_ckpt_esi(
                            max_steps_duration=self.max_steps_duration,
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                        # Check if the conditions for saving a checkpoint are met.
                        # The conditions include a mandatory condition (1) and
                        # one of the following optional conditions (2/3/4):
                        # 1. The save frequency is set to a positive value.
                        # 2. It's the last training step.
                        # 3. The current step number is a multiple of the save frequency.
                        # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                        if self.config.trainer.save_freq > 0 and (
                            is_last_step
                            or self.global_steps % self.config.trainer.save_freq == 0
                            or esi_close_to_expiration
                        ):
                            if esi_close_to_expiration:
                                print("Force saving checkpoint: ESI instance expiration approaching.")
                            with marked_timer("save_checkpoint", timing_raw, color="green"):
                                if self._metric_logging_settings.mode == "balanced":
                                    checkpoint_payload_path = self._save_checkpoint_payload()
                                else:
                                    self._save_checkpoint()

                        # update weights from trainer to rollout
                        with marked_timer("update_weights", timing_raw, color="red"):
                            self.checkpoint_manager.update_weights(self.global_steps)

                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate(include_format_metrics=True)
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                test_benchmarks_freq = int(self.config.trainer.get("test_benchmarks_freq", -1))
                if test_benchmarks_freq > 0 and (is_last_step or self.global_steps % test_benchmarks_freq == 0):
                    with marked_timer("test_benchmarks", timing_raw, color="green"):
                        test_benchmark_metrics = self._validate_test_benchmarks()
                        if is_last_step:
                            last_test_benchmark_metrics = test_benchmark_metrics
                    metrics.update(test_benchmark_metrics)

                if opd_diagnostics_task is not None:
                    with marked_timer("opd_diagnostics_join", timing_raw, color="green"):
                        metrics.update(
                            self._finish_opd_diagnostics_for_step(
                                opd_diagnostics_task,
                                expected_step=self.global_steps,
                            )
                        )

                diagnostics_submit_seconds = timing_raw.get("opd_diagnostics_submit", 0.0)
                diagnostics_join_seconds = timing_raw.get("opd_diagnostics_join", 0.0)
                timing_raw["opd_diagnostics"] = diagnostics_submit_seconds + diagnostics_join_seconds
                # The submit work is already inside timing_raw["step"]. The join happens after
                # the core step/validation overlap and is the only additional blocking cost.
                timing_raw["step"] += diagnostics_join_seconds

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                # GDPO per-component reward metrics
                gdpo_reward_keys = self.config.algorithm.get("gdpo_reward_keys", None)
                if gdpo_reward_keys and self.config.algorithm.adv_estimator in ("gdpo", AdvantageEstimator.GDPO):
                    for key in gdpo_reward_keys:
                        if key in batch.non_tensor_batch:
                            vals = np.asarray(batch.non_tensor_batch[key], dtype=np.float32)
                            metrics[f"gdpo/{key}/mean"] = float(np.mean(vals))
                            metrics[f"gdpo/{key}/std"] = float(np.std(vals))
                            metrics[f"gdpo/{key}/max"] = float(np.max(vals))
                            metrics[f"gdpo/{key}/min"] = float(np.min(vals))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # compute variance proxy metrics
                gradient_norm = metrics.get("actor/grad_norm", None)
                metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                self._record_metric_event(
                    logger,
                    metrics,
                    event_kind="train",
                    checkpoint_payload_path=checkpoint_payload_path,
                )

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    pprint(f"Final test benchmark metrics: {last_test_benchmark_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
