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

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch

from verl.trainer.ppo.metric_utils import (
    compute_text_repetition_and_truncation_metrics_with_flags,
    compute_token_repetition_and_truncation_metrics,
    repetition_tail_mask_from_sample_flags,
)

MAX_REASONABLE_ENTROPY = 32.0
MAX_REASONABLE_LOGPROB_ABS = 1.0e4
DISTILLATION_DIAGNOSTICS_PREFIX = "distillation"

LONG_SEQUENCE_THRESHOLDS = (1024, 2048, 4096, 8192, 12000, 16000, 20000, 23000)
BAD_TAIL_MIN_POSITION = 12000
BAD_TAIL_DEEP_POSITION = 16000
BAD_TAIL_EXTREME_POSITION = 20000
BAD_TAIL_OVERLAP_THRESHOLD = 0.5
BAD_TAIL_TEACHER_PROB_THRESHOLD = 0.2
BAD_TAIL_GAP_THRESHOLD = 1.0
BAD_TAIL_REWARD_THRESHOLD = -2.0
BAD_TAIL_ENTROPY_THRESHOLD = 4.0


@dataclass(frozen=True)
class SpecialTokenIds:
    """Resolved tokenizer/control token ids used by long-sequence diagnostics."""

    im_end: Optional[int] = None
    endoftext: Optional[int] = None
    eot: Optional[int] = None
    think_start: Optional[int] = None
    think_end: Optional[int] = None
    im_start: Optional[int] = None
    user_role: Optional[int] = None
    assistant_role: Optional[int] = None
    bos: Optional[int] = None
    generation_stop_ids: tuple[int, ...] = ()
    foreign_control_ids: tuple[int, ...] = ()
    think_start_sequence: tuple[int, ...] = ()
    think_end_sequence: tuple[int, ...] = ()
    im_start_sequence: tuple[int, ...] = ()
    generation_stop_sequences: tuple[tuple[int, ...], ...] = ()
    foreign_control_sequences: tuple[tuple[int, ...], ...] = ()

    def token_items(self) -> tuple[tuple[str, Optional[int]], ...]:
        return (
            ("im_end", self.im_end),
            ("endoftext", self.endoftext),
            ("eot", self.eot),
            ("think_start", self.think_start),
            ("think_end", self.think_end),
        )


@dataclass
class ResponsePhaseInfo:
    """Parsed response phase state and masks."""

    metrics: dict[str, float]
    thinking_mask: torch.Tensor
    answering_mask: torch.Tensor
    malformed_mask: torch.Tensor
    clipped_mask: torch.Tensor
    carrier_mask: torch.Tensor


def _topk_distillation_metric_key(metric_name: str, *, k: int, configured_topk: int) -> str:
    suffix = "" if int(k) == int(configured_topk) else f"_top{k}"
    return f"{DISTILLATION_DIAGNOSTICS_PREFIX}/{metric_name}{suffix}"


def _to_cpu_float(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().float().cpu()


def _masked_values(values: torch.Tensor, mask: torch.Tensor, value_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    values = _to_cpu_float(values)
    mask = mask.detach().bool().cpu()
    valid = mask & torch.isfinite(values)
    if value_mask is not None:
        valid = valid & value_mask.detach().bool().cpu()
    return values[valid]


def _safe_mean(values: torch.Tensor) -> Optional[float]:
    if values.numel() == 0:
        return None
    return float(values.mean().item())


def _safe_quantile(values: torch.Tensor, q: float) -> Optional[float]:
    if values.numel() == 0:
        return None
    return float(torch.quantile(values, q).item())


def _safe_std(values: torch.Tensor) -> Optional[float]:
    if values.numel() == 0:
        return None
    return float(values.float().std(unbiased=False).item())


def _put_if_present(metrics: dict[str, float], key: str, value: Optional[float]) -> None:
    if value is not None and np.isfinite(value):
        metrics[key] = value


def _put_distribution_metrics(metrics: dict[str, float], prefix: str, values: torch.Tensor) -> None:
    values = _to_cpu_float(values).flatten()
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return
    metrics[f"{prefix}/count"] = float(values.numel())
    _put_if_present(metrics, f"{prefix}/mean", _safe_mean(values))
    _put_if_present(metrics, f"{prefix}/std", _safe_std(values))
    _put_if_present(metrics, f"{prefix}/min", float(values.min().item()))
    _put_if_present(metrics, f"{prefix}/p50", _safe_quantile(values, 0.50))
    _put_if_present(metrics, f"{prefix}/p75", _safe_quantile(values, 0.75))
    _put_if_present(metrics, f"{prefix}/p90", _safe_quantile(values, 0.90))
    _put_if_present(metrics, f"{prefix}/p95", _safe_quantile(values, 0.95))
    _put_if_present(metrics, f"{prefix}/p99", _safe_quantile(values, 0.99))
    _put_if_present(metrics, f"{prefix}/max", float(values.max().item()))


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> Optional[float]:
    return _safe_mean(_masked_values(values, mask))


def _masked_quantile(values: torch.Tensor, mask: torch.Tensor, q: float) -> Optional[float]:
    return _safe_quantile(_masked_values(values, mask), q)


def _masked_event_fraction(event_mask: torch.Tensor, mask: torch.Tensor) -> Optional[float]:
    mask = mask.detach().bool().cpu()
    denom = int(mask.sum().item())
    if denom == 0:
        return None
    event_mask = event_mask.detach().bool().cpu()
    return int((event_mask & mask).sum().item()) / denom


def _topk_residual_bucket_entropy(logprobs: torch.Tensor, finite_mask: torch.Tensor) -> torch.Tensor:
    probs = torch.where(finite_mask, logprobs.exp(), torch.zeros_like(logprobs))
    topk_entropy = torch.where(finite_mask, -(probs * logprobs), torch.zeros_like(logprobs)).sum(dim=-1)
    residual_mass = (1.0 - probs.sum(dim=-1)).clamp(min=0.0, max=1.0)
    residual_entropy = torch.where(
        residual_mass > 0,
        -(residual_mass * residual_mass.clamp_min(torch.finfo(probs.dtype).tiny).log()),
        torch.zeros_like(residual_mass),
    )
    return topk_entropy + residual_entropy


def _bin_key(prefix: str, start: int, end: int) -> str:
    return f"{prefix}/{start:04d}_{end:04d}"


def _coarse_bucket_specs(response_len: int) -> list[tuple[str, int, int]]:
    """Return overlapping coarse buckets used for long-sequence OPD diagnostics."""
    response_len = max(int(response_len), 0)
    specs = [
        ("0_4k", 0, min(4096, response_len)),
        ("4k_12k", min(4096, response_len), min(12000, response_len)),
        ("12k_16k", min(12000, response_len), min(16000, response_len)),
        ("16k_20k", min(16000, response_len), min(20000, response_len)),
        ("20k_max", min(20000, response_len), response_len),
        ("12k_max", min(12000, response_len), response_len),
        ("16k_max", min(16000, response_len), response_len),
        ("head", 0, min(4096, response_len)),
        ("mid", min(4096, response_len), min(12000, response_len)),
        ("tail", min(12000, response_len), response_len),
        ("deep_tail", min(16000, response_len), response_len),
        ("extreme_tail", min(20000, response_len), response_len),
    ]
    return [(name, start, end) for name, start, end in specs if end > start]


def _single_token_id(tokenizer: Any, text: str) -> Optional[int]:
    if tokenizer is None:
        return None
    try:
        vocab = tokenizer.get_vocab()
    except Exception:
        vocab = None
    if isinstance(vocab, dict) and text in vocab:
        try:
            return int(vocab[text])
        except (TypeError, ValueError):
            pass
    try:
        token_id = tokenizer.convert_tokens_to_ids(text)
    except Exception:
        token_id = None
    if token_id is not None:
        try:
            token_id_int = int(token_id)
            unk_token_id = getattr(tokenizer, "unk_token_id", None)
            unk_token = getattr(tokenizer, "unk_token", None)
            is_unknown = unk_token_id is not None and token_id_int == int(unk_token_id) and text != unk_token
            if token_id_int >= 0 and not is_unknown:
                return token_id_int
        except (TypeError, ValueError):
            pass
    try:
        encoded = tokenizer.encode(text, add_special_tokens=False)
    except Exception:
        encoded = []
    if len(encoded) == 1:
        return int(encoded[0])
    return None


def _token_sequence(tokenizer: Any, text: str) -> tuple[int, ...]:
    token_id = _single_token_id(tokenizer, text)
    if token_id is not None:
        return (token_id,)
    if tokenizer is None:
        return ()
    try:
        encoded = tokenizer.encode(text, add_special_tokens=False)
    except Exception:
        encoded = []
    return tuple(int(value) for value in encoded if value is not None)


def _token_content(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    content = getattr(value, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, str):
            return content
    return str(value)


def _token_ids_from_value(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int | np.integer):
        return [int(value)]
    if isinstance(value, list | tuple | set):
        result = []
        for item in value:
            result.extend(_token_ids_from_value(item))
        return result
    return []


def _unique_int_tuple(values: list[int | None]) -> tuple[int, ...]:
    result = []
    seen = set()
    for value in values:
        if value is None:
            continue
        token_id = int(value)
        if token_id < 0 or token_id in seen:
            continue
        seen.add(token_id)
        result.append(token_id)
    return tuple(result)


def _unique_sequence_tuple(sequences: list[tuple[int, ...]]) -> tuple[tuple[int, ...], ...]:
    result = []
    seen = set()
    for sequence in sequences:
        sequence = tuple(int(value) for value in sequence if value is not None)
        if not sequence or sequence in seen:
            continue
        seen.add(sequence)
        result.append(sequence)
    return tuple(result)


def resolve_special_token_ids(tokenizer: Any, generation_stop_ids: Optional[Any] = None) -> SpecialTokenIds:
    """Resolve Qwen/chat/control token ids without assuming where the tokenizer stores them."""
    eos_ids = _token_ids_from_value(getattr(tokenizer, "eos_token_id", None) if tokenizer is not None else None)
    pad_ids = _token_ids_from_value(getattr(tokenizer, "pad_token_id", None) if tokenizer is not None else None)
    generation_stop_values = _token_ids_from_value(generation_stop_ids)

    endoftext = _single_token_id(tokenizer, "<|endoftext|>")
    deepseek_end = _single_token_id(tokenizer, "<｜end▁of▁sentence｜>")
    im_end = _single_token_id(tokenizer, "<|im_end|>")
    eot = _single_token_id(tokenizer, "<|EOT|>")
    think_start = _single_token_id(tokenizer, "<think>")
    think_end = _single_token_id(tokenizer, "</think>")
    im_start = _single_token_id(tokenizer, "<|im_start|>")
    user_role = _single_token_id(tokenizer, "<｜User｜>")
    assistant_role = _single_token_id(tokenizer, "<｜Assistant｜>")
    bos = _single_token_id(tokenizer, "<｜begin▁of▁sentence｜>")
    endoftext = endoftext if endoftext is not None else deepseek_end

    stop_texts = [
        _token_content(getattr(tokenizer, "eos_token", None) if tokenizer is not None else None),
        _token_content(getattr(tokenizer, "pad_token", None) if tokenizer is not None else None),
        "<|im_end|>",
        "<|endoftext|>",
        "<｜end▁of▁sentence｜>",
        "<|EOT|>",
    ]
    generation_stop_sequences = _unique_sequence_tuple(
        [
            *(
                (int(token_id),)
                for token_id in [*eos_ids, *generation_stop_values, im_end, endoftext, eot]
                if token_id is not None
            ),
            *(_token_sequence(tokenizer, text) for text in stop_texts if text),
        ]
    )

    control_texts = (
        "<|object_ref_start|>",
        "<|object_ref_end|>",
        "<|box_start|>",
        "<|box_end|>",
        "<|quad_start|>",
        "<|quad_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|vision_pad|>",
        "<|image_pad|>",
        "<|video_pad|>",
        "<tool_call>",
        "</tool_call>",
        "<tool_response>",
        "</tool_response>",
        "<|fim_prefix|>",
        "<|fim_middle|>",
        "<|fim_suffix|>",
        "<|fim_pad|>",
        "<｜User｜>",
        "<｜Assistant｜>",
        "<｜begin▁of▁sentence｜>",
    )
    foreign_control_sequences = _unique_sequence_tuple([_token_sequence(tokenizer, token) for token in control_texts])
    foreign_control_ids = _unique_int_tuple(
        [sequence[0] for sequence in foreign_control_sequences if len(sequence) == 1]
    )
    stop_ids = _unique_int_tuple([*eos_ids, *generation_stop_values, im_end, endoftext, eot, *pad_ids])
    return SpecialTokenIds(
        im_end=im_end,
        endoftext=endoftext,
        eot=eot,
        think_start=think_start,
        think_end=think_end,
        im_start=im_start,
        user_role=user_role,
        assistant_role=assistant_role,
        bos=bos,
        generation_stop_ids=stop_ids,
        foreign_control_ids=foreign_control_ids,
        think_start_sequence=_token_sequence(tokenizer, "<think>"),
        think_end_sequence=_token_sequence(tokenizer, "</think>"),
        im_start_sequence=_token_sequence(tokenizer, "<|im_start|>"),
        generation_stop_sequences=generation_stop_sequences,
        foreign_control_sequences=foreign_control_sequences,
    )


def special_token_resolution_metrics(special_token_ids: SpecialTokenIds) -> dict[str, float]:
    metrics: dict[str, float] = {
        "special_tokens/resolved/generation_stop_ids_count": float(len(special_token_ids.generation_stop_ids)),
        "special_tokens/resolved/generation_stop_sequences_count": float(
            len(special_token_ids.generation_stop_sequences)
        ),
        "special_tokens/resolved/foreign_control_ids_count": float(len(special_token_ids.foreign_control_ids)),
        "special_tokens/resolved/foreign_control_sequences_count": float(
            len(special_token_ids.foreign_control_sequences)
        ),
        "special_tokens/resolved/think_start_sequence_length": float(len(special_token_ids.think_start_sequence)),
        "special_tokens/resolved/think_end_sequence_length": float(len(special_token_ids.think_end_sequence)),
    }
    for name, token_id in (
        ("im_end_id", special_token_ids.im_end),
        ("endoftext_id", special_token_ids.endoftext),
        ("eot_id", special_token_ids.eot),
        ("think_start_id", special_token_ids.think_start),
        ("think_end_id", special_token_ids.think_end),
        ("im_start_id", special_token_ids.im_start),
        ("user_role_id", special_token_ids.user_role),
        ("assistant_role_id", special_token_ids.assistant_role),
        ("bos_id", special_token_ids.bos),
    ):
        if token_id is not None:
            metrics[f"special_tokens/resolved/{name}"] = float(token_id)
    return metrics


def compute_binned_token_means(
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    bin_size: int,
    prefix: str,
    value_mask: Optional[torch.Tensor] = None,
) -> dict[str, float]:
    """Compute position-binned token means for a [batch, response_len] tensor."""
    if bin_size <= 0:
        raise ValueError(f"bin_size must be positive, got {bin_size}.")
    if values.shape != mask.shape:
        raise ValueError(f"values and mask must have the same shape, got {values.shape=} and {mask.shape=}.")

    metrics: dict[str, float] = {}
    response_len = values.shape[1]
    value_mask_cpu = value_mask.detach().bool().cpu() if value_mask is not None else None
    for start in range(0, response_len, bin_size):
        end = min(start + bin_size, response_len)
        bin_value_mask = value_mask_cpu[:, start:end] if value_mask_cpu is not None else None
        mean = _safe_mean(_masked_values(values[:, start:end], mask[:, start:end], bin_value_mask))
        _put_if_present(metrics, _bin_key(prefix, start, end), mean)
    return metrics


def compute_bucket_denominator_metrics(
    response_mask: torch.Tensor,
    *,
    bin_size: int,
) -> dict[str, float]:
    """Record denominator metrics for fine and coarse response-position buckets."""
    if bin_size <= 0:
        raise ValueError(f"bin_size must be positive, got {bin_size}.")
    response_mask_cpu = response_mask.detach().bool().cpu()
    response_len = response_mask_cpu.shape[1]
    response_lengths = response_mask_cpu.sum(dim=1)
    clipped_mask = response_lengths == response_len
    metrics: dict[str, float] = {}

    def add_denominators(prefix: str, bucket_mask: torch.Tensor) -> None:
        sample_mask = bucket_mask.any(dim=1)
        metrics[f"{prefix}/token_count"] = float(bucket_mask.sum().item())
        metrics[f"{prefix}/sample_count"] = float(sample_mask.sum().item())
        metrics[f"{prefix}/clipped_sample_count"] = float((sample_mask & clipped_mask).sum().item())
        metrics[f"{prefix}/nonclipped_sample_count"] = float((sample_mask & ~clipped_mask).sum().item())
        if sample_mask.any():
            metrics[f"{prefix}/avg_response_length_of_samples_in_bucket"] = float(
                response_lengths[sample_mask].float().mean().item()
            )

    for start in range(0, response_len, bin_size):
        end = min(start + bin_size, response_len)
        add_denominators(f"bucket/bin/{start:04d}_{end:04d}", response_mask_cpu[:, start:end])

    for name, start, end in _coarse_bucket_specs(response_len):
        add_denominators(f"bucket/coarse/{name}", response_mask_cpu[:, start:end])
    return metrics


def compute_coarse_token_stats(
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    prefix: str,
    value_mask: Optional[torch.Tensor] = None,
    include_distribution: bool = False,
) -> dict[str, float]:
    """Compute overlapping coarse-bucket token statistics and tail-head deltas."""
    if values.shape != mask.shape:
        raise ValueError(f"values and mask must have the same shape, got {values.shape=} and {mask.shape=}.")
    values_cpu = _to_cpu_float(values)
    mask_cpu = mask.detach().bool().cpu()
    value_mask_cpu = value_mask.detach().bool().cpu() if value_mask is not None else None
    response_len = values_cpu.shape[1]
    metrics: dict[str, float] = {}
    means: dict[str, float] = {}
    for name, start, end in _coarse_bucket_specs(response_len):
        current_mask = mask_cpu[:, start:end]
        if value_mask_cpu is not None:
            current_value_mask = value_mask_cpu[:, start:end]
        else:
            current_value_mask = None
        values_flat = _masked_values(values_cpu[:, start:end], current_mask, current_value_mask)
        token_count = (
            int((current_mask & current_value_mask).sum().item())
            if current_value_mask is not None
            else int(current_mask.sum().item())
        )
        metrics[f"{prefix}/coarse/{name}/token_count"] = float(token_count)
        if values_flat.numel() == 0:
            continue
        bucket_prefix = f"{prefix}/coarse/{name}"
        mean = _safe_mean(values_flat)
        _put_if_present(metrics, f"{bucket_prefix}/mean", mean)
        if mean is not None:
            means[name] = mean
        if include_distribution:
            _put_if_present(metrics, f"{bucket_prefix}/std", _safe_std(values_flat))
            _put_if_present(metrics, f"{bucket_prefix}/p50", _safe_quantile(values_flat, 0.50))
            _put_if_present(metrics, f"{bucket_prefix}/p90", _safe_quantile(values_flat, 0.90))
            _put_if_present(metrics, f"{bucket_prefix}/p95", _safe_quantile(values_flat, 0.95))
            _put_if_present(metrics, f"{bucket_prefix}/p99", _safe_quantile(values_flat, 0.99))
            _put_if_present(metrics, f"{bucket_prefix}/max", float(values_flat.max().item()))
    for tail_name, metric_name in (
        ("tail", "tail_minus_head"),
        ("deep_tail", "deep_tail_minus_head"),
        ("extreme_tail", "extreme_tail_minus_head"),
        ("16k_max", "16k_max_minus_0_4k"),
        ("20k_max", "20k_max_minus_0_4k"),
    ):
        if "head" in means and tail_name in means:
            metrics[f"{prefix}/coarse/{metric_name}"] = means[tail_name] - means["head"]
    return metrics


def compute_binned_event_rates(
    event_mask: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    bin_size: int,
    prefix: str,
) -> dict[str, float]:
    """Compute position-binned event rates for [batch, response_len] boolean tensors."""
    if bin_size <= 0:
        raise ValueError(f"bin_size must be positive, got {bin_size}.")
    if event_mask.shape != response_mask.shape:
        raise ValueError(
            "event_mask and response_mask must have the same shape, "
            f"got {event_mask.shape=} and {response_mask.shape=}."
        )

    event_mask = event_mask.detach().bool().cpu()
    response_mask = response_mask.detach().bool().cpu()
    metrics: dict[str, float] = {}
    response_len = response_mask.shape[1]
    for start in range(0, response_len, bin_size):
        end = min(start + bin_size, response_len)
        valid = response_mask[:, start:end]
        denom = int(valid.sum().item())
        if denom == 0:
            continue
        count = int((event_mask[:, start:end] & valid).sum().item())
        metrics[_bin_key(prefix, start, end)] = count / denom
    return metrics


def _event_metric_key(prefix: str, separator: str, suffix: str) -> str:
    return f"{prefix}{separator}{suffix}"


def _normalize_position_list(value: Any, *, context: str, sample_idx: int) -> list[int]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    elif isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, int | np.integer):
        return [int(value)]
    if not isinstance(value, list | tuple):
        raise TypeError(
            f"{context} positions for sample {sample_idx} must be a list/tuple of integers, got {type(value).__name__}."
        )

    positions = []
    for raw_position in value:
        if raw_position is None:
            continue
        if isinstance(raw_position, float | np.floating) and not np.isfinite(raw_position):
            raise ValueError(f"{context} positions for sample {sample_idx} contain non-finite value {raw_position!r}.")
        positions.append(int(raw_position))
    return positions


def positions_to_event_mask(
    event_positions: Optional[list[Any]],
    response_mask: torch.Tensor,
    *,
    context: str,
) -> torch.Tensor:
    """Convert per-sample response-relative positions to a boolean event mask."""
    response_mask_cpu = response_mask.detach().bool().cpu()
    event_mask = torch.zeros_like(response_mask_cpu, dtype=torch.bool)
    if event_positions is None:
        return event_mask
    if hasattr(event_positions, "tolist"):
        event_positions = event_positions.tolist()
    else:
        event_positions = list(event_positions)
    if len(event_positions) == 0:
        return event_mask
    if len(event_positions) != response_mask_cpu.shape[0]:
        raise ValueError(
            f"{context} positions batch-size mismatch: got {len(event_positions)} samples, "
            f"expected {response_mask_cpu.shape[0]} from response_mask.shape={tuple(response_mask_cpu.shape)}."
        )

    response_width = response_mask_cpu.shape[1]
    for sample_idx, sample_positions in enumerate(event_positions):
        for position in _normalize_position_list(sample_positions, context=context, sample_idx=sample_idx):
            if position < 0 or position >= response_width:
                raise ValueError(
                    f"{context} position {position} for sample {sample_idx} is outside response width {response_width}."
                )
            event_mask[sample_idx, position] = True
    return event_mask & response_mask_cpu


def compute_event_position_metrics(
    *,
    event_mask: torch.Tensor,
    response_mask: torch.Tensor,
    metric_prefix: str,
    suffix_separator: str,
    token_fraction_suffix: str,
    bin_prefix: str,
    bin_size: int,
) -> dict[str, float]:
    """Aggregate event frequency and location metrics."""
    if event_mask.shape != response_mask.shape:
        raise ValueError(
            f"event_mask and response_mask must align, got {event_mask.shape=} and {response_mask.shape=}."
        )
    if suffix_separator not in {"/", "_"}:
        raise ValueError(f"suffix_separator must be '/' or '_', got {suffix_separator!r}.")

    event_mask = event_mask.detach().bool().cpu()
    response_mask = response_mask.detach().bool().cpu()
    event_mask = event_mask & response_mask
    valid_lengths = response_mask.sum(dim=1)
    event_counts = event_mask.sum(dim=1)
    total_valid_tokens = int(response_mask.sum().item())
    total_events = int(event_counts.sum().item())
    batch_size = int(response_mask.shape[0])

    metrics = {
        _event_metric_key(metric_prefix, suffix_separator, token_fraction_suffix): total_events
        / max(total_valid_tokens, 1),
        _event_metric_key(metric_prefix, suffix_separator, "sample_fraction"): float((event_counts > 0).float().mean())
        if batch_size
        else 0.0,
        _event_metric_key(metric_prefix, suffix_separator, "count_mean"): float(event_counts.float().mean().item())
        if batch_size
        else 0.0,
        _event_metric_key(metric_prefix, suffix_separator, "count_max"): float(event_counts.max().item())
        if batch_size
        else 0.0,
    }

    first_positions = []
    first_position_fracs = []
    for sample_idx in range(batch_size):
        sample_positions = torch.nonzero(event_mask[sample_idx], as_tuple=False).flatten()
        if sample_positions.numel() == 0:
            continue
        first_position = int(sample_positions[0].item())
        first_positions.append(first_position)
        first_position_fracs.append(first_position / max(int(valid_lengths[sample_idx].item()), 1))
    if first_positions:
        metrics[_event_metric_key(metric_prefix, suffix_separator, "first_position_mean")] = float(
            np.mean(first_positions)
        )
        metrics[_event_metric_key(metric_prefix, suffix_separator, "first_position_frac_mean")] = float(
            np.mean(first_position_fracs)
        )

    metrics.update(
        compute_binned_event_rates(
            event_mask=event_mask,
            response_mask=response_mask,
            bin_size=bin_size,
            prefix=bin_prefix,
        )
    )
    return metrics


def _first_position(sequence: list[int], token_id: Optional[int]) -> Optional[int]:
    if token_id is None:
        return None
    for idx, value in enumerate(sequence):
        if int(value) == int(token_id):
            return idx
    return None


def _count_token(sequence: list[int], token_id: Optional[int]) -> int:
    if token_id is None:
        return 0
    return sum(1 for value in sequence if int(value) == int(token_id))


def _contains_any(sequence: list[int], token_ids: tuple[int, ...]) -> bool:
    if not token_ids:
        return False
    token_set = set(token_ids)
    return any(int(value) in token_set for value in sequence)


def _last_is_any(sequence: list[int], token_ids: tuple[int, ...]) -> bool:
    if not sequence or not token_ids:
        return False
    return int(sequence[-1]) in set(token_ids)


def _find_sequence_positions(sequence: list[int], pattern: tuple[int, ...]) -> list[int]:
    if not pattern or len(pattern) > len(sequence):
        return []
    pattern_len = len(pattern)
    return [
        idx
        for idx in range(0, len(sequence) - pattern_len + 1)
        if tuple(int(value) for value in sequence[idx : idx + pattern_len]) == pattern
    ]


def _first_sequence_position(sequence: list[int], pattern: tuple[int, ...]) -> Optional[int]:
    positions = _find_sequence_positions(sequence, pattern)
    return positions[0] if positions else None


def _contains_any_sequence(sequence: list[int], patterns: tuple[tuple[int, ...], ...]) -> bool:
    return any(_first_sequence_position(sequence, pattern) is not None for pattern in patterns)


def _ends_with_any_sequence(sequence: list[int], patterns: tuple[tuple[int, ...], ...]) -> bool:
    if not sequence:
        return False
    return any(len(pattern) <= len(sequence) and tuple(sequence[-len(pattern) :]) == pattern for pattern in patterns)


def compute_phase_and_special_token_metrics(
    *,
    response_ids: torch.Tensor,
    response_mask: torch.Tensor,
    special_token_ids: SpecialTokenIds,
    max_response_length: Optional[int] = None,
) -> ResponsePhaseInfo:
    """Parse thinking/answering phases and terminal-token inclusion from generated responses."""
    if response_ids.shape != response_mask.shape:
        raise ValueError(
            f"response_ids and response_mask must align, got {response_ids.shape=} and {response_mask.shape=}."
        )
    response_ids_cpu = response_ids.detach().cpu()
    response_mask_cpu = response_mask.detach().bool().cpu()
    batch_size, response_width = response_mask_cpu.shape
    if max_response_length is None:
        max_response_length = response_width

    thinking_mask = torch.zeros_like(response_mask_cpu, dtype=torch.bool)
    answering_mask = torch.zeros_like(response_mask_cpu, dtype=torch.bool)
    malformed_mask = torch.zeros_like(response_mask_cpu, dtype=torch.bool)
    carrier_mask = torch.zeros(batch_size, dtype=torch.bool)

    lengths = response_mask_cpu.sum(dim=1).long()
    clipped_mask = lengths >= int(max_response_length)
    has_think_start = []
    has_think_end = []
    unclosed_think = []
    repeated_think_start = []
    repeated_think_end = []
    think_end_without_start = []
    think_reopened = []
    generated_im_start = []
    foreign_control_sample = []
    malformed = []
    contains_im_end = []
    contains_endoftext = []
    contains_generation_stop = []
    ends_with_im_end = []
    ends_with_endoftext = []
    ends_with_generation_stop = []
    im_end_included = []
    endoftext_included = []
    generation_stop_included = []
    think_end_included = []
    clipped_without_think_end = []
    clipped_after_think_end = []
    clipped_before_think_end = []
    think_start_positions = []
    think_end_positions = []
    tokens_before_think_end = []
    tokens_after_think_end = []

    for sample_idx in range(batch_size):
        length = int(lengths[sample_idx].item())
        valid_ids = [int(value) for value in response_ids_cpu[sample_idx, :length].tolist()]
        think_start_pattern = special_token_ids.think_start_sequence
        think_end_pattern = special_token_ids.think_end_sequence
        start_positions = _find_sequence_positions(valid_ids, think_start_pattern)
        end_positions = _find_sequence_positions(valid_ids, think_end_pattern)
        start_count = (
            len(start_positions) if think_start_pattern else _count_token(valid_ids, special_token_ids.think_start)
        )
        end_count = len(end_positions) if think_end_pattern else _count_token(valid_ids, special_token_ids.think_end)
        first_start = (
            start_positions[0] if start_positions else _first_position(valid_ids, special_token_ids.think_start)
        )
        first_end = end_positions[0] if end_positions else _first_position(valid_ids, special_token_ids.think_end)
        think_end_width = max(len(think_end_pattern), 1)
        first_end_exclusive = (first_end + think_end_width) if first_end is not None else None
        has_start = first_start is not None
        has_end = first_end is not None
        end_without_start = has_end and not has_start
        reopened = False
        if has_end and first_end_exclusive is not None:
            suffix_after_first_end = valid_ids[first_end_exclusive:]
            if think_start_pattern:
                reopened = _first_sequence_position(suffix_after_first_end, think_start_pattern) is not None
            elif special_token_ids.think_start is not None:
                reopened = any(int(value) == int(special_token_ids.think_start) for value in suffix_after_first_end)
        im_start_patterns = _unique_sequence_tuple(
            [special_token_ids.im_start_sequence, _unique_int_tuple([special_token_ids.im_start])]
        )
        sample_generated_im_start = _contains_any_sequence(valid_ids, im_start_patterns)
        foreign_control_patterns = special_token_ids.foreign_control_sequences or tuple(
            (token_id,) for token_id in special_token_ids.foreign_control_ids
        )
        sample_foreign_control = _contains_any_sequence(valid_ids, foreign_control_patterns)
        sample_malformed = (
            end_without_start
            or reopened
            or start_count > 1
            or end_count > 1
            or sample_generated_im_start
            or sample_foreign_control
        )

        if has_start:
            think_start_positions.append(first_start)
        if has_end and first_end_exclusive is not None:
            think_end_positions.append(first_end)
            tokens_before_think_end.append(first_end_exclusive)
            tokens_after_think_end.append(max(length - first_end_exclusive, 0))
        if has_start and not has_end:
            unclosed = True
        else:
            unclosed = False

        if (
            has_start
            and has_end
            and first_start is not None
            and first_end is not None
            and first_end_exclusive is not None
            and first_start < first_end
        ):
            thinking_mask[sample_idx, first_start : min(first_end_exclusive, length)] = True
            if first_end_exclusive < length:
                answering_mask[sample_idx, first_end_exclusive:length] = True
        elif has_start and first_start is not None and not has_end:
            thinking_mask[sample_idx, first_start:length] = True
        elif not has_start and has_end and first_end is not None:
            malformed_mask[sample_idx, :length] = True
        if sample_malformed:
            malformed_mask[sample_idx, :length] = True
        carrier_mask[sample_idx] = bool(sample_malformed or unclosed or (clipped_mask[sample_idx] and not has_end))

        has_think_start.append(has_start)
        has_think_end.append(has_end)
        unclosed_think.append(unclosed)
        repeated_think_start.append(start_count > 1)
        repeated_think_end.append(end_count > 1)
        think_end_without_start.append(end_without_start)
        think_reopened.append(reopened)
        generated_im_start.append(sample_generated_im_start)
        foreign_control_sample.append(sample_foreign_control)
        malformed.append(sample_malformed)
        im_end_patterns = _unique_sequence_tuple([_unique_int_tuple([special_token_ids.im_end])])
        endoftext_patterns = _unique_sequence_tuple([_unique_int_tuple([special_token_ids.endoftext])])
        generation_stop_patterns = special_token_ids.generation_stop_sequences or tuple(
            (token_id,) for token_id in special_token_ids.generation_stop_ids
        )
        contains_im_end.append(_contains_any_sequence(valid_ids, im_end_patterns))
        contains_endoftext.append(_contains_any_sequence(valid_ids, endoftext_patterns))
        contains_generation_stop.append(_contains_any_sequence(valid_ids, generation_stop_patterns))
        ends_with_im_end.append(_ends_with_any_sequence(valid_ids, im_end_patterns))
        ends_with_endoftext.append(_ends_with_any_sequence(valid_ids, endoftext_patterns))
        ends_with_generation_stop.append(_ends_with_any_sequence(valid_ids, generation_stop_patterns))
        im_end_included.append(contains_im_end[-1])
        endoftext_included.append(contains_endoftext[-1])
        generation_stop_included.append(contains_generation_stop[-1])
        think_end_included.append(has_end)
        is_clipped = bool(clipped_mask[sample_idx].item())
        clipped_without_think_end.append(is_clipped and not has_end)
        clipped_after_think_end.append(is_clipped and has_end)
        clipped_before_think_end.append(is_clipped and not has_end and has_start)

    metrics: dict[str, float] = {}

    def ratio(name: str, values: list[bool]) -> None:
        metrics[name] = float(np.mean(values)) if values else 0.0

    ratio("format/has_think_start_ratio", has_think_start)
    ratio("format/has_think_end_ratio", has_think_end)
    ratio("format/unclosed_think_ratio", unclosed_think)
    ratio("format/repeated_think_start_ratio", repeated_think_start)
    ratio("format/repeated_think_end_ratio", repeated_think_end)
    ratio("format/think_end_without_think_start_ratio", think_end_without_start)
    ratio("format/think_reopened_after_think_end_ratio", think_reopened)
    ratio("format/generated_im_start_inside_response_ratio", generated_im_start)
    ratio("format/foreign_control_token_sample_ratio", foreign_control_sample)
    ratio("format/malformed_ratio", malformed)
    ratio("format/clipped_without_think_end_ratio", clipped_without_think_end)
    ratio("format/clipped_after_think_end_ratio", clipped_after_think_end)
    ratio("format/clipped_before_think_end_ratio", clipped_before_think_end)

    metrics["response_end/ended_by_max_length_count"] = float(clipped_mask.sum().item())
    metrics["response_end/ended_by_max_length_ratio"] = float(clipped_mask.float().mean().item()) if batch_size else 0.0
    metrics["response_end/ended_by_generation_stop_count"] = float(sum(ends_with_generation_stop))
    metrics["response_end/ended_by_generation_stop_ratio"] = (
        float(np.mean(ends_with_generation_stop)) if batch_size else 0.0
    )
    metrics["response_end/ended_by_im_end_count"] = float(sum(ends_with_im_end))
    metrics["response_end/ended_by_im_end_ratio"] = float(np.mean(ends_with_im_end)) if batch_size else 0.0
    metrics["response_end/ended_by_endoftext_count"] = float(sum(ends_with_endoftext))
    metrics["response_end/ended_by_endoftext_ratio"] = float(np.mean(ends_with_endoftext)) if batch_size else 0.0
    normal_ended = [stop and not bool(clipped_mask[idx].item()) for idx, stop in enumerate(ends_with_generation_stop)]
    metrics["response_end/ended_normally_ratio"] = float(np.mean(normal_ended)) if batch_size else 0.0

    for prefix, values in (
        ("format/think_start_position", think_start_positions),
        ("format/think_end_position", think_end_positions),
        ("format/tokens_before_think_end", tokens_before_think_end),
        ("format/tokens_after_think_end", tokens_after_think_end),
        ("format/tokens_after_think_end_until_stop", tokens_after_think_end),
    ):
        if values:
            _put_distribution_metrics(metrics, prefix, torch.tensor(values, dtype=torch.float32))

    metrics["loss_mask/response_contains_im_end_count"] = float(sum(contains_im_end))
    metrics["loss_mask/response_contains_endoftext_count"] = float(sum(contains_endoftext))
    metrics["loss_mask/response_contains_generation_stop_count"] = float(sum(contains_generation_stop))
    metrics["loss_mask/im_end_included_count"] = float(sum(im_end_included))
    metrics["loss_mask/endoftext_included_count"] = float(sum(endoftext_included))
    metrics["loss_mask/generation_stop_included_count"] = float(sum(generation_stop_included))
    metrics["loss_mask/think_end_included_count"] = float(sum(think_end_included))
    ended_count = max(sum(ends_with_generation_stop), 1)
    metrics["loss_mask/generation_stop_included_ratio_among_ended"] = (
        float(
            sum(
                stop and included
                for stop, included in zip(ends_with_generation_stop, generation_stop_included, strict=True)
            )
        )
        / ended_count
    )
    im_end_count = max(sum(ends_with_im_end), 1)
    metrics["loss_mask/im_end_included_ratio_among_ended"] = (
        float(sum(stop and included for stop, included in zip(ends_with_im_end, im_end_included, strict=True)))
        / im_end_count
    )
    think_end_count = max(sum(has_think_end), 1)
    metrics["loss_mask/think_end_included_ratio_among_has_think_end"] = float(sum(think_end_included)) / think_end_count

    total_loss_tokens = int(response_mask_cpu.sum().item())
    clipped_token_count = int(response_mask_cpu[clipped_mask].sum().item()) if clipped_mask.any() else 0
    deep_tail_mask = torch.zeros_like(response_mask_cpu, dtype=torch.bool)
    if response_width > 16000:
        deep_tail_mask[:, 16000:] = response_mask_cpu[:, 16000:]
    near_clipped_16k = lengths > 16000
    metrics["loss_mask/clipped_token_count"] = float(clipped_token_count)
    metrics["loss_mask/nonclipped_token_count"] = float(total_loss_tokens - clipped_token_count)
    metrics["loss_mask/clipped_token_share"] = clipped_token_count / max(total_loss_tokens, 1)
    metrics["loss_mask/near_clipped_16k_token_share"] = (
        int(response_mask_cpu[near_clipped_16k].sum().item()) / max(total_loss_tokens, 1)
        if near_clipped_16k.any()
        else 0.0
    )
    metrics["loss_mask/deep_tail_16k_token_share"] = int(deep_tail_mask.sum().item()) / max(total_loss_tokens, 1)

    for threshold in (4096, 8192, 12000, 16000, 20000):
        no_think_end = torch.tensor([not value for value in has_think_end], dtype=torch.bool)
        already_answering = torch.tensor(has_think_end, dtype=torch.bool)
        reached = lengths > threshold
        metrics[f"response/survival_no_think_end_by_{threshold // 1000}k"] = float(
            (no_think_end & reached).sum().item()
        ) / max(batch_size, 1)
        if reached.any():
            metrics[f"response/clip_given_no_think_end_by_{threshold // 1000}k"] = float(
                (clipped_mask & no_think_end & reached).sum().item()
            ) / max(int((no_think_end & reached).sum().item()), 1)
            metrics[f"response/clip_given_already_answering_by_{threshold // 1000}k"] = float(
                (clipped_mask & already_answering & reached).sum().item()
            ) / max(int((already_answering & reached).sum().item()), 1)

    return ResponsePhaseInfo(
        metrics=metrics,
        thinking_mask=thinking_mask & response_mask_cpu,
        answering_mask=answering_mask & response_mask_cpu,
        malformed_mask=malformed_mask & response_mask_cpu,
        clipped_mask=clipped_mask,
        carrier_mask=carrier_mask,
    )


def _head_tail_metrics(
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    prefix: str,
    bin_size: int,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if values.shape[1] == 0:
        return metrics
    head_end = min(bin_size, values.shape[1])
    tail_start = max(values.shape[1] - bin_size, 0)
    head = _masked_mean(values[:, :head_end], mask[:, :head_end])
    tail = _masked_mean(values[:, tail_start:], mask[:, tail_start:])
    _put_if_present(metrics, f"{prefix}/head_mean", head)
    _put_if_present(metrics, f"{prefix}/tail_mean", tail)
    if head is not None and tail is not None:
        metrics[f"{prefix}/tail_minus_head"] = tail - head
        if abs(head) > 1e-8:
            metrics[f"{prefix}/tail_over_head"] = tail / head
    return metrics


def compute_topk_ladder(topk: int) -> list[int]:
    """Return powers-of-two coverage checkpoints ending at ``topk``."""
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}.")
    ladder = []
    value = 1
    while value < topk:
        ladder.append(value)
        value *= 2
    if not ladder or ladder[-1] != topk:
        ladder.append(topk)
    return ladder


def extract_response_topk_scores(
    *,
    sequence_ids: torch.Tensor,
    sequence_logprobs: torch.Tensor,
    response_mask: torch.Tensor,
    prompt_width: int,
    context: str = "teacher_topk",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract response-aligned top-k ids/logprobs from padded prompt+response teacher scoring output."""
    if sequence_ids.ndim != 3 or sequence_logprobs.ndim != 3:
        raise RuntimeError(
            f"{context} ids/logprobs must have rank 3 for OPD overlap diagnostics, "
            f"got ids.shape={tuple(sequence_ids.shape)} and logprobs.shape={tuple(sequence_logprobs.shape)}."
        )
    if sequence_ids.shape != sequence_logprobs.shape:
        raise RuntimeError(
            f"{context} ids/logprobs shape mismatch: "
            f"ids.shape={tuple(sequence_ids.shape)}, logprobs.shape={tuple(sequence_logprobs.shape)}."
        )
    if sequence_ids.shape[0] != response_mask.shape[0]:
        raise RuntimeError(
            f"{context} batch-size mismatch: ids.shape={tuple(sequence_ids.shape)}, "
            f"response_mask.shape={tuple(response_mask.shape)}."
        )
    if sequence_ids.shape[:2] == response_mask.shape:
        response_ids = sequence_ids
        response_logprobs = sequence_logprobs
    else:
        if prompt_width <= 0:
            raise RuntimeError("OPD overlap diagnostics require a positive padded prompt width.")
        response_width = response_mask.shape[1]
        start = prompt_width - 1
        end = start + response_width
        if sequence_ids.shape[1] < end:
            raise RuntimeError(
                f"Cannot align {context} with response tokens: ids.shape={tuple(sequence_ids.shape)}, "
                f"prompt_width={prompt_width}, response_width={response_width}, expected at least {end} positions."
            )
        response_ids = sequence_ids[:, start:end]
        response_logprobs = sequence_logprobs[:, start:end]
    if response_ids.shape[:2] != response_mask.shape:
        raise RuntimeError(
            f"OPD overlap diagnostics produced misaligned {context}: "
            f"response_ids.shape={tuple(response_ids.shape)}, response_mask.shape={tuple(response_mask.shape)}."
        )
    return response_ids, response_logprobs


def binary_auc(scores: torch.Tensor, labels: torch.Tensor) -> Optional[float]:
    """Compute ROC AUC with average ranks; returns None if only one class exists."""
    scores = _to_cpu_float(scores).flatten()
    labels = labels.detach().bool().cpu().flatten()
    finite = torch.isfinite(scores)
    scores = scores[finite]
    labels = labels[finite]
    n_pos = int(labels.sum().item())
    n_total = int(labels.numel())
    n_neg = n_total - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    order = torch.argsort(scores)
    sorted_scores = scores[order]
    ranks = torch.empty_like(scores)
    start = 0
    while start < n_total:
        end = start + 1
        while end < n_total and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end

    rank_sum_pos = ranks[labels].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc.item())


def extract_sampled_response_logprobs(
    *,
    sequence_logprobs: torch.Tensor,
    response_mask: torch.Tensor,
    prompt_width: int,
    context: str = "sequence_logprobs",
) -> torch.Tensor | None:
    """Extract response-token sampled logprobs from padded prompt+response prompt scoring output.

    Returns None when the tensor contains top-k logprobs instead of one sampled-token logprob per position.
    """
    if sequence_logprobs.ndim not in (2, 3):
        raise RuntimeError(
            f"{context} must have rank 2 or 3 for OPD diagnostics, got shape={tuple(sequence_logprobs.shape)}."
        )
    if sequence_logprobs.shape[0] != response_mask.shape[0]:
        raise RuntimeError(
            f"{context} batch-size mismatch for OPD diagnostics: "
            f"sequence_logprobs.shape={tuple(sequence_logprobs.shape)}, "
            f"response_mask.shape={tuple(response_mask.shape)}."
        )
    if sequence_logprobs.ndim == 3 and sequence_logprobs.shape[-1] != 1:
        return None

    if sequence_logprobs.shape[:2] == response_mask.shape:
        response_logprobs = sequence_logprobs
    else:
        if prompt_width <= 0:
            raise RuntimeError("OPD diagnostics require a positive padded prompt width.")
        response_width = response_mask.shape[1]
        start = prompt_width - 1
        end = start + response_width
        if sequence_logprobs.shape[1] < end:
            raise RuntimeError(
                f"Cannot align {context} with response tokens for OPD diagnostics: "
                f"sequence_logprobs.shape={tuple(sequence_logprobs.shape)}, prompt_width={prompt_width}, "
                f"response_width={response_width}, expected at least {end} scored positions."
            )
        response_logprobs = sequence_logprobs[:, start:end]

    if response_logprobs.ndim == 3:
        response_logprobs = response_logprobs.squeeze(-1)
    if response_logprobs.shape != response_mask.shape:
        raise RuntimeError(
            f"OPD diagnostics produced misaligned {context}: "
            f"response_logprobs.shape={tuple(response_logprobs.shape)}, "
            f"response_mask.shape={tuple(response_mask.shape)}."
        )
    return response_logprobs


def compute_overlap_diagnostics(
    *,
    response_ids: torch.Tensor,
    response_mask: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    teacher_topk_logprobs: torch.Tensor,
    sampled_teacher_logprobs: Optional[torch.Tensor] = None,
    bin_size: int,
    topk: int,
) -> dict[str, float]:
    """Compute teacher top-k support diagnostics for student response tokens."""
    if response_ids.shape != response_mask.shape:
        raise ValueError(
            f"response_ids and response_mask must align, got {response_ids.shape=} and {response_mask.shape=}."
        )
    if teacher_topk_ids.shape != teacher_topk_logprobs.shape:
        raise ValueError(
            "teacher_topk_ids and teacher_topk_logprobs must align, "
            f"got {teacher_topk_ids.shape=} and {teacher_topk_logprobs.shape=}."
        )
    if teacher_topk_ids.ndim != 3 or teacher_topk_ids.shape[:2] != response_mask.shape:
        raise ValueError(
            "teacher_topk_ids must have shape [batch, response_len, k], "
            f"got {teacher_topk_ids.shape=} and response_mask.shape={response_mask.shape}."
        )
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}.")

    response_ids = response_ids.detach().cpu()
    response_mask = response_mask.detach().bool().cpu()
    teacher_topk_ids = teacher_topk_ids.detach().cpu()
    teacher_topk_logprobs = _to_cpu_float(teacher_topk_logprobs)
    finite_topk = torch.isfinite(teacher_topk_logprobs)
    valid_topk_rows = response_mask & finite_topk.any(dim=-1)
    available_topk = min(int(topk), int(teacher_topk_ids.shape[-1]))
    metrics: dict[str, float] = {}

    for k in compute_topk_ladder(topk):
        if k > available_topk:
            continue
        coverage = (teacher_topk_ids[..., :k] == response_ids.unsqueeze(-1)).any(dim=-1)
        coverage = coverage & valid_topk_rows
        _put_if_present(metrics, f"opd/overlap/top{k}_rate", _safe_mean(coverage[valid_topk_rows].float()))
        metrics.update(
            compute_binned_event_rates(
                event_mask=coverage,
                response_mask=valid_topk_rows,
                bin_size=bin_size,
                prefix=f"opd/overlap/top{k}_bin",
            )
        )
        metrics.update(
            _head_tail_metrics(
                coverage.float(),
                valid_topk_rows,
                prefix=f"opd/overlap/top{k}",
                bin_size=bin_size,
            )
        )
        metrics.update(
            compute_coarse_token_stats(
                coverage.float(),
                valid_topk_rows,
                prefix=f"opd/overlap/top{k}",
            )
        )

    if sampled_teacher_logprobs is not None:
        if sampled_teacher_logprobs.ndim == 3 and sampled_teacher_logprobs.shape[-1] == 1:
            sampled_teacher_logprobs = sampled_teacher_logprobs.squeeze(-1)
        if sampled_teacher_logprobs.shape != response_mask.shape:
            raise ValueError(
                "sampled_teacher_logprobs and response_mask must align for OPD overlap diagnostics, "
                f"got {sampled_teacher_logprobs.shape=} and {response_mask.shape=}."
            )
        sampled_teacher_logprobs = _to_cpu_float(sampled_teacher_logprobs)
        sampled_prob = sampled_teacher_logprobs.exp()
        sampled_valid = response_mask & torch.isfinite(sampled_prob)
        sampled_values = sampled_prob[sampled_valid]
        _put_if_present(metrics, "opd/overlap/teacher_sampled_prob_mean", _safe_mean(sampled_values))
        _put_if_present(metrics, "opd/overlap/teacher_sampled_prob_p01", _safe_quantile(sampled_values, 0.01))
        _put_if_present(metrics, "opd/overlap/teacher_sampled_prob_p10", _safe_quantile(sampled_values, 0.1))
        _put_if_present(metrics, "opd/overlap/teacher_sampled_prob_p50", _safe_quantile(sampled_values, 0.5))
        _put_if_present(metrics, "distillation/teacher_sampled_prob/mean", _safe_mean(sampled_values))
        _put_if_present(metrics, "distillation/teacher_sampled_prob/p01", _safe_quantile(sampled_values, 0.01))
        _put_if_present(metrics, "distillation/teacher_sampled_prob/p10", _safe_quantile(sampled_values, 0.1))
        _put_if_present(metrics, "distillation/teacher_sampled_prob/p50", _safe_quantile(sampled_values, 0.5))
        metrics.update(
            compute_binned_token_means(
                sampled_prob,
                sampled_valid,
                bin_size=bin_size,
                prefix="opd/overlap/teacher_sampled_prob_bin",
            )
        )
        metrics.update(
            compute_coarse_token_stats(
                sampled_prob,
                sampled_valid,
                prefix="distillation/teacher_sampled_prob",
                include_distribution=True,
            )
        )

    topk_probs = torch.where(finite_topk, teacher_topk_logprobs.exp(), torch.zeros_like(teacher_topk_logprobs))
    topk_mass = topk_probs[:, :, :available_topk].sum(dim=-1)
    _put_if_present(metrics, "opd/overlap/teacher_topk_mass_mean", _masked_mean(topk_mass, valid_topk_rows))
    return metrics


def compute_rethinking_opd_diagnostics(
    *,
    response_mask: torch.Tensor,
    student_topk_ids: torch.Tensor,
    student_topk_logprobs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    teacher_topk_logprobs: torch.Tensor,
    bin_size: int,
    topk: int,
) -> dict[str, float]:
    """Compute the top-k OPD dynamics from arXiv:2604.13016 on student-visited states."""
    if student_topk_ids.shape != student_topk_logprobs.shape:
        raise ValueError(
            "student_topk_ids and student_topk_logprobs must align, "
            f"got {student_topk_ids.shape=} and {student_topk_logprobs.shape=}."
        )
    if teacher_topk_ids.shape != teacher_topk_logprobs.shape:
        raise ValueError(
            "teacher_topk_ids and teacher_topk_logprobs must align, "
            f"got {teacher_topk_ids.shape=} and {teacher_topk_logprobs.shape=}."
        )
    if student_topk_ids.ndim != 3 or student_topk_ids.shape[:2] != response_mask.shape:
        raise ValueError(
            "student_topk_ids must have shape [batch, response_len, k], "
            f"got {student_topk_ids.shape=} and response_mask.shape={response_mask.shape}."
        )
    if teacher_topk_ids.ndim != 3 or teacher_topk_ids.shape[:2] != response_mask.shape:
        raise ValueError(
            "teacher_topk_ids must have shape [batch, response_len, k], "
            f"got {teacher_topk_ids.shape=} and response_mask.shape={response_mask.shape}."
        )
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}.")

    response_mask = response_mask.detach().bool().cpu()
    student_topk_ids = student_topk_ids.detach().cpu()
    teacher_topk_ids = teacher_topk_ids.detach().cpu()
    student_topk_logprobs = _to_cpu_float(student_topk_logprobs)
    teacher_topk_logprobs = _to_cpu_float(teacher_topk_logprobs)

    available_topk = min(int(topk), int(student_topk_ids.shape[-1]), int(teacher_topk_ids.shape[-1]))
    metrics: dict[str, float] = {}
    if available_topk <= 0:
        return metrics

    for k in compute_topk_ladder(topk):
        if k > available_topk:
            continue

        student_ids = student_topk_ids[..., :k]
        teacher_ids = teacher_topk_ids[..., :k]
        student_logprobs = student_topk_logprobs[..., :k]
        teacher_logprobs = teacher_topk_logprobs[..., :k]
        student_finite = torch.isfinite(student_logprobs)
        teacher_finite = torch.isfinite(teacher_logprobs)
        valid_rows = response_mask & (student_finite.sum(dim=-1) >= k) & (teacher_finite.sum(dim=-1) >= k)

        student_probs = torch.where(student_finite, student_logprobs.exp(), torch.zeros_like(student_logprobs))
        teacher_probs = torch.where(teacher_finite, teacher_logprobs.exp(), torch.zeros_like(teacher_logprobs))
        student_topk_mass = student_probs.sum(dim=-1)
        teacher_topk_mass = teacher_probs.sum(dim=-1)
        valid_rows = valid_rows & (student_topk_mass <= 1.001) & (teacher_topk_mass <= 1.001)

        matches = student_ids.unsqueeze(-1) == teacher_ids.unsqueeze(-2)
        matches = matches & student_finite.unsqueeze(-1) & teacher_finite.unsqueeze(-2)
        student_overlap_mask = matches.any(dim=-1)
        teacher_overlap_mask = matches.any(dim=-2)
        overlap_count = student_overlap_mask.sum(dim=-1)
        nonempty_overlap = overlap_count > 0
        valid_overlap_rows = valid_rows & nonempty_overlap

        overlap_ratio = overlap_count.detach().float() / float(k)
        overlap_mass_student = torch.where(student_overlap_mask, student_probs, torch.zeros_like(student_probs)).sum(
            dim=-1
        )
        overlap_mass_teacher = torch.where(teacher_overlap_mask, teacher_probs, torch.zeros_like(teacher_probs)).sum(
            dim=-1
        )

        _put_if_present(
            metrics,
            _topk_distillation_metric_key("overlap_ratio", k=k, configured_topk=topk),
            _masked_mean(overlap_ratio, valid_rows),
        )
        _put_if_present(
            metrics,
            _topk_distillation_metric_key("overlap_mass_student", k=k, configured_topk=topk),
            _masked_mean(overlap_mass_student, valid_rows),
        )
        _put_if_present(
            metrics,
            _topk_distillation_metric_key("overlap_mass_teacher", k=k, configured_topk=topk),
            _masked_mean(overlap_mass_teacher, valid_rows),
        )
        _put_if_present(
            metrics,
            _topk_distillation_metric_key("empty_overlap_fraction", k=k, configured_topk=topk),
            _masked_event_fraction(~nonempty_overlap, valid_rows),
        )
        metrics.update(
            compute_binned_token_means(
                overlap_ratio,
                valid_rows,
                bin_size=bin_size,
                prefix=_topk_distillation_metric_key("overlap_ratio_bin", k=k, configured_topk=topk),
            )
        )
        metrics.update(
            _head_tail_metrics(
                overlap_ratio,
                valid_rows,
                prefix=_topk_distillation_metric_key("overlap_ratio", k=k, configured_topk=topk),
                bin_size=bin_size,
            )
        )
        metrics.update(
            compute_coarse_token_stats(
                overlap_ratio,
                valid_rows,
                prefix=_topk_distillation_metric_key("overlap_ratio", k=k, configured_topk=topk),
            )
        )

        matched_teacher_logprobs = torch.where(
            matches,
            teacher_logprobs.unsqueeze(-2),
            torch.full_like(matches, float("-inf"), dtype=teacher_logprobs.dtype),
        ).amax(dim=-1)
        overlap_student_logprobs = torch.where(
            student_overlap_mask,
            student_logprobs,
            torch.full_like(student_logprobs, float("-inf")),
        )
        overlap_teacher_logprobs = torch.where(
            student_overlap_mask,
            matched_teacher_logprobs,
            torch.full_like(matched_teacher_logprobs, float("-inf")),
        )
        matched_student_logprobs = torch.where(
            matches,
            student_logprobs.unsqueeze(-1),
            torch.full_like(matches, float("-inf"), dtype=student_logprobs.dtype),
        ).amax(dim=-2)
        teacher_on_student_coverage = torch.where(
            torch.isfinite(matched_teacher_logprobs),
            matched_teacher_logprobs.exp(),
            torch.zeros_like(matched_teacher_logprobs),
        ).sum(dim=-1)
        student_on_teacher_coverage = torch.where(
            torch.isfinite(matched_student_logprobs),
            matched_student_logprobs.exp(),
            torch.zeros_like(matched_student_logprobs),
        ).sum(dim=-1)
        for coverage_name, coverage_value in (
            (f"distillation/coverage/teacher_on_student_top{k}", teacher_on_student_coverage),
            (f"distillation/coverage/student_on_teacher_top{k}", student_on_teacher_coverage),
        ):
            _put_if_present(metrics, f"{coverage_name}/mean", _masked_mean(coverage_value, valid_rows))
            metrics.update(
                compute_coarse_token_stats(
                    coverage_value,
                    valid_rows,
                    prefix=coverage_name,
                    include_distribution=True,
                )
            )

        student_overlap_log_mass = torch.logsumexp(overlap_student_logprobs, dim=-1, keepdim=True)
        teacher_overlap_log_mass = torch.logsumexp(overlap_teacher_logprobs, dim=-1, keepdim=True)
        student_logp_bar = overlap_student_logprobs - student_overlap_log_mass
        teacher_logp_bar = overlap_teacher_logprobs - teacher_overlap_log_mass
        student_p_bar = torch.where(student_overlap_mask, student_logp_bar.exp(), torch.zeros_like(student_logp_bar))
        advantage_values = torch.where(
            student_overlap_mask,
            student_p_bar * (teacher_logp_bar - student_logp_bar),
            torch.zeros_like(student_p_bar),
        )
        overlap_token_advantage = advantage_values.sum(dim=-1) / overlap_count.clamp_min(1).float()
        _put_if_present(
            metrics,
            _topk_distillation_metric_key("overlap_token_advantage", k=k, configured_topk=topk),
            _masked_mean(overlap_token_advantage, valid_overlap_rows),
        )
        metrics.update(
            compute_binned_token_means(
                overlap_token_advantage,
                valid_overlap_rows,
                bin_size=bin_size,
                prefix=_topk_distillation_metric_key("overlap_token_advantage_bin", k=k, configured_topk=topk),
            )
        )

        student_entropy_approx = _topk_residual_bucket_entropy(student_logprobs, student_finite)
        teacher_entropy_approx = _topk_residual_bucket_entropy(teacher_logprobs, teacher_finite)
        entropy_gap_approx = (teacher_entropy_approx - student_entropy_approx).abs()
        entropy_gap_signed = teacher_entropy_approx - student_entropy_approx
        _put_if_present(
            metrics,
            _topk_distillation_metric_key("entropy_student", k=k, configured_topk=topk),
            _masked_mean(student_entropy_approx, valid_rows),
        )
        _put_if_present(
            metrics,
            _topk_distillation_metric_key("entropy_teacher", k=k, configured_topk=topk),
            _masked_mean(teacher_entropy_approx, valid_rows),
        )
        _put_if_present(
            metrics,
            _topk_distillation_metric_key("entropy_gap", k=k, configured_topk=topk),
            _masked_mean(entropy_gap_approx, valid_rows),
        )
        metrics.update(
            compute_binned_token_means(
                entropy_gap_approx,
                valid_rows,
                bin_size=bin_size,
                prefix=_topk_distillation_metric_key("entropy_gap_bin", k=k, configured_topk=topk),
            )
        )
        entropy_suffix = "" if int(k) == int(topk) else f"_top{k}"
        for entropy_name, entropy_values in (
            (f"opd/entropy/student{entropy_suffix}", student_entropy_approx),
            (f"opd/entropy/teacher{entropy_suffix}", teacher_entropy_approx),
            (f"opd/entropy/gap_abs{entropy_suffix}", entropy_gap_approx),
            (f"opd/entropy/gap_signed{entropy_suffix}", entropy_gap_signed),
        ):
            metrics.update(
                compute_binned_token_means(
                    entropy_values,
                    valid_rows,
                    bin_size=bin_size,
                    prefix=f"{entropy_name}_bin",
                )
            )
            metrics.update(
                compute_coarse_token_stats(
                    entropy_values,
                    valid_rows,
                    prefix=entropy_name,
                    include_distribution=True,
                )
            )

    metrics[f"{DISTILLATION_DIAGNOSTICS_PREFIX}/entropy_gap_residual_bucket_approximation"] = 1.0
    return metrics


def _gather_visible_topk_probability(
    topk_ids: torch.Tensor,
    topk_logprobs: torch.Tensor,
    token_ids: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sum probability and visibility mask for token ids present in returned top-k rows."""
    if not token_ids:
        probs = torch.zeros(topk_ids.shape[:2], dtype=topk_logprobs.dtype, device=topk_logprobs.device)
        return probs, torch.zeros_like(probs, dtype=torch.bool)
    target = torch.tensor(token_ids, dtype=topk_ids.dtype, device=topk_ids.device)
    matches = (topk_ids.unsqueeze(-1) == target).any(dim=-1) & torch.isfinite(topk_logprobs)
    probs = torch.where(matches, topk_logprobs.exp(), torch.zeros_like(topk_logprobs)).sum(dim=-1)
    return probs, matches.any(dim=-1)


def compute_special_token_topk_metrics(
    *,
    response_mask: torch.Tensor,
    special_token_ids: SpecialTokenIds,
    bin_size: int,
    student_topk_ids: Optional[torch.Tensor] = None,
    student_topk_logprobs: Optional[torch.Tensor] = None,
    teacher_topk_ids: Optional[torch.Tensor] = None,
    teacher_topk_logprobs: Optional[torch.Tensor] = None,
    thinking_mask: Optional[torch.Tensor] = None,
    answering_mask: Optional[torch.Tensor] = None,
) -> dict[str, float]:
    """Compute special-token probabilities when tokens are visible in returned top-k rows."""
    metrics: dict[str, float] = {}
    response_mask_cpu = response_mask.detach().bool().cpu()

    def add_model_metrics(
        *,
        model_prefix: str,
        topk_ids: Optional[torch.Tensor],
        topk_logprobs: Optional[torch.Tensor],
    ) -> None:
        if topk_ids is None or topk_logprobs is None:
            return
        topk_ids_cpu = topk_ids.detach().cpu()
        topk_logprobs_cpu = _to_cpu_float(topk_logprobs)
        if topk_ids_cpu.shape != topk_logprobs_cpu.shape or topk_ids_cpu.shape[:2] != response_mask_cpu.shape:
            raise ValueError(
                f"{model_prefix} top-k tensors must align with response_mask, got "
                f"{topk_ids_cpu.shape=}, {topk_logprobs_cpu.shape=}, {response_mask_cpu.shape=}."
            )
        token_groups = [(name, _unique_int_tuple([token_id])) for name, token_id in special_token_ids.token_items()]
        token_groups.extend(
            [
                ("generation_stop_sum", special_token_ids.generation_stop_ids),
                ("foreign_control_sum", special_token_ids.foreign_control_ids),
            ]
        )
        for name, token_ids in token_groups:
            if not token_ids:
                continue
            probs, visible = _gather_visible_topk_probability(topk_ids_cpu, topk_logprobs_cpu, token_ids)
            valid = response_mask_cpu & visible
            if name in {"generation_stop_sum", "foreign_control_sum"}:
                base = f"{model_prefix}/prob_topk/group/{name}"
            else:
                base = f"{model_prefix}/prob_topk/token/{name}"
            metrics[f"{base}/visible_token_fraction"] = float(valid.sum().item()) / max(
                int(response_mask_cpu.sum().item()), 1
            )
            _put_if_present(metrics, f"{base}/mean", _masked_mean(probs, valid))
            metrics.update(compute_binned_token_means(probs, valid, bin_size=bin_size, prefix=f"{base}_bin"))
            metrics.update(compute_coarse_token_stats(probs, valid, prefix=base, include_distribution=True))

        for phase_name, phase_mask in (("thinking", thinking_mask), ("answering", answering_mask)):
            if phase_mask is None:
                continue
            phase_mask_cpu = phase_mask.detach().bool().cpu() & response_mask_cpu
            if phase_name == "thinking" and special_token_ids.think_end is not None:
                probs, visible = _gather_visible_topk_probability(
                    topk_ids_cpu, topk_logprobs_cpu, _unique_int_tuple([special_token_ids.think_end])
                )
                valid = phase_mask_cpu & visible
                _put_if_present(
                    metrics,
                    f"{model_prefix}/prob_phase/thinking/think_end/mean",
                    _masked_mean(probs, valid),
                )
            if phase_name == "answering" and special_token_ids.generation_stop_ids:
                probs, visible = _gather_visible_topk_probability(
                    topk_ids_cpu, topk_logprobs_cpu, special_token_ids.generation_stop_ids
                )
                valid = phase_mask_cpu & visible
                _put_if_present(
                    metrics,
                    f"{model_prefix}/prob_phase/answering/generation_stop_sum/mean",
                    _masked_mean(probs, valid),
                )
                if special_token_ids.im_end is not None:
                    im_end_probs, im_end_visible = _gather_visible_topk_probability(
                        topk_ids_cpu, topk_logprobs_cpu, _unique_int_tuple([special_token_ids.im_end])
                    )
                    _put_if_present(
                        metrics,
                        f"{model_prefix}/prob_phase/answering/im_end/mean",
                        _masked_mean(im_end_probs, phase_mask_cpu & im_end_visible),
                    )

    add_model_metrics(model_prefix="student", topk_ids=student_topk_ids, topk_logprobs=student_topk_logprobs)
    add_model_metrics(model_prefix="teacher", topk_ids=teacher_topk_ids, topk_logprobs=teacher_topk_logprobs)
    return metrics


def compute_bad_tail_metrics(
    *,
    response_mask: torch.Tensor,
    clipped_mask: torch.Tensor,
    phase_carrier_mask: Optional[torch.Tensor] = None,
    abs_gap: Optional[torch.Tensor] = None,
    token_reward: Optional[torch.Tensor] = None,
    teacher_sampled_prob: Optional[torch.Tensor] = None,
    top16_overlap: Optional[torch.Tensor] = None,
    student_entropy: Optional[torch.Tensor] = None,
    teacher_entropy: Optional[torch.Tensor] = None,
    min_position: int = BAD_TAIL_MIN_POSITION,
    deep_position: int = BAD_TAIL_DEEP_POSITION,
    extreme_position: int = BAD_TAIL_EXTREME_POSITION,
    overlap_threshold: float = BAD_TAIL_OVERLAP_THRESHOLD,
    teacher_prob_threshold: float = BAD_TAIL_TEACHER_PROB_THRESHOLD,
    gap_threshold: float = BAD_TAIL_GAP_THRESHOLD,
    reward_threshold: float = BAD_TAIL_REWARD_THRESHOLD,
    entropy_threshold: float = BAD_TAIL_ENTROPY_THRESHOLD,
) -> dict[str, float]:
    """Aggregate bad-tail token mass and carrier-sample metrics."""
    response_mask_cpu = response_mask.detach().bool().cpu()
    batch_size, response_width = response_mask_cpu.shape
    position = torch.arange(response_width).unsqueeze(0).expand(batch_size, -1)
    tail_min = response_mask_cpu & (position >= int(min_position))
    tail_deep = response_mask_cpu & (position >= int(deep_position))
    tail_extreme = response_mask_cpu & (position >= int(extreme_position))
    condition_masks: list[torch.Tensor] = []
    severity = torch.zeros_like(response_mask_cpu, dtype=torch.float32)

    def add_condition(condition: Optional[torch.Tensor]) -> None:
        nonlocal severity
        if condition is None:
            return
        condition_cpu = condition.detach().bool().cpu() & response_mask_cpu
        condition_masks.append(condition_cpu)
        severity = severity + condition_cpu.float()

    if abs_gap is not None:
        add_condition(_to_cpu_float(abs_gap) > float(gap_threshold))
    if token_reward is not None:
        add_condition(_to_cpu_float(token_reward) < float(reward_threshold))
    if teacher_sampled_prob is not None:
        add_condition(_to_cpu_float(teacher_sampled_prob) < float(teacher_prob_threshold))
    if top16_overlap is not None:
        add_condition(_to_cpu_float(top16_overlap) < float(overlap_threshold))
    if student_entropy is not None:
        add_condition(_to_cpu_float(student_entropy) > float(entropy_threshold))
    if teacher_entropy is not None:
        add_condition(_to_cpu_float(teacher_entropy) > float(entropy_threshold))

    if condition_masks:
        bad_condition = torch.stack(condition_masks, dim=0).any(dim=0)
    else:
        bad_condition = torch.zeros_like(response_mask_cpu, dtype=torch.bool)
    bad_tail = bad_condition & tail_min
    bad_tail_deep = bad_condition & tail_deep
    bad_tail_extreme = bad_condition & tail_extreme
    bad_tail_by_sample = bad_tail.any(dim=1)
    clipped_mask_cpu = clipped_mask.detach().bool().cpu()
    carrier_mask = bad_tail_by_sample.clone()
    if phase_carrier_mask is not None:
        carrier_mask = carrier_mask | phase_carrier_mask.detach().bool().cpu()
    total_tokens = int(response_mask_cpu.sum().item())
    bad_tail_tokens = int(bad_tail.sum().item())
    carrier_tokens = int(response_mask_cpu[carrier_mask].sum().item()) if carrier_mask.any() else 0

    metrics: dict[str, float] = {
        "contagion/bad_tail_min_position": float(min_position),
        "contagion/bad_tail_deep_position": float(deep_position),
        "contagion/bad_tail_extreme_position": float(extreme_position),
        "contagion/bad_tail_gap_threshold": float(gap_threshold),
        "contagion/bad_tail_reward_threshold": float(reward_threshold),
        "contagion/bad_tail_teacher_prob_threshold": float(teacher_prob_threshold),
        "contagion/bad_tail_overlap_threshold": float(overlap_threshold),
        "contagion/bad_tail_entropy_threshold": float(entropy_threshold),
        "contagion/bad_tail_token_count": float(bad_tail_tokens),
        "contagion/bad_tail_token_fraction": bad_tail_tokens / max(total_tokens, 1),
        "contagion/bad_tail_loss_token_fraction": bad_tail_tokens / max(total_tokens, 1),
        "contagion/bad_tail_12k_token_count": float(bad_tail_tokens),
        "contagion/bad_tail_16k_token_count": float(bad_tail_deep.sum().item()),
        "contagion/bad_tail_20k_token_count": float(bad_tail_extreme.sum().item()),
        "contagion/bad_tail_sample_count": float(bad_tail_by_sample.sum().item()),
        "contagion/bad_tail_sample_ratio": float(bad_tail_by_sample.float().mean().item()) if batch_size else 0.0,
        "contagion/bad_tail_clipped_sample_count": float((bad_tail_by_sample & clipped_mask_cpu).sum().item()),
        "contagion/bad_tail_nonclipped_sample_count": float((bad_tail_by_sample & ~clipped_mask_cpu).sum().item()),
        "contagion/bad_tail_near_clipped_sample_count": float(
            (bad_tail_by_sample & (response_mask_cpu.sum(dim=1) > int(deep_position))).sum().item()
        ),
        "contagion/carrier_sample_count": float(carrier_mask.sum().item()),
        "contagion/carrier_sample_ratio": float(carrier_mask.float().mean().item()) if batch_size else 0.0,
        "contagion/carrier_token_share": carrier_tokens / max(total_tokens, 1),
        "contagion/carrier_loss_share": carrier_tokens / max(total_tokens, 1),
        "contagion/current_bad_tail_mass": bad_tail_tokens / max(total_tokens, 1),
        "contagion/current_bad_tail_loss_share": bad_tail_tokens / max(total_tokens, 1),
        "contagion/current_carrier_ratio": float(carrier_mask.float().mean().item()) if batch_size else 0.0,
    }
    score_values = severity[bad_tail]
    _put_if_present(metrics, "contagion/bad_tail_score_mean", _safe_mean(score_values))
    _put_if_present(metrics, "contagion/bad_tail_score_p90", _safe_quantile(score_values, 0.90))
    _put_if_present(metrics, "contagion/bad_tail_score_p99", _safe_quantile(score_values, 0.99))
    if score_values.numel():
        metrics["contagion/bad_tail_score_max"] = float(score_values.max().item())

    susceptible = (~clipped_mask_cpu) & (response_mask_cpu.sum(dim=1) < int(min_position))
    metrics["contagion/susceptible_sample_count"] = float(susceptible.sum().item())
    if susceptible.any():
        metrics["contagion/susceptible_length_mean"] = float(response_mask_cpu.sum(dim=1)[susceptible].float().mean())
    return metrics


def compute_repetition_advantage_metrics(
    *,
    repetition_tail_mask: torch.Tensor,
    response_mask: torch.Tensor,
    student_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    value_mask: Optional[torch.Tensor] = None,
) -> dict[str, float]:
    """Compare reverse-KL advantage/logprobs on repetitive and non-repetitive response tokens."""
    if teacher_logprobs.ndim == 3 and teacher_logprobs.shape[-1] == 1:
        teacher_logprobs = teacher_logprobs.squeeze(-1)
    if not (repetition_tail_mask.shape == response_mask.shape == student_logprobs.shape == teacher_logprobs.shape):
        raise ValueError(
            "repetition_tail_mask, response_mask, student_logprobs, and teacher_logprobs must align, got "
            f"{repetition_tail_mask.shape=}, {response_mask.shape=}, {student_logprobs.shape=}, "
            f"{teacher_logprobs.shape=}."
        )
    response_mask_cpu = response_mask.detach().bool().cpu()
    repetition_tail_mask_cpu = repetition_tail_mask.detach().bool().cpu()
    student_logprobs_cpu = _to_cpu_float(student_logprobs)
    teacher_logprobs_cpu = _to_cpu_float(teacher_logprobs)
    valid = response_mask_cpu & torch.isfinite(student_logprobs_cpu) & torch.isfinite(teacher_logprobs_cpu)
    valid = (
        valid
        & (student_logprobs_cpu.abs() <= MAX_REASONABLE_LOGPROB_ABS)
        & (teacher_logprobs_cpu.abs() <= MAX_REASONABLE_LOGPROB_ABS)
    )
    if value_mask is not None:
        valid = valid & value_mask.detach().bool().cpu()
    token_reward = teacher_logprobs_cpu - student_logprobs_cpu
    repetition_mask = valid & repetition_tail_mask_cpu
    nonrepetition_mask = valid & ~repetition_tail_mask_cpu
    total_valid = int(valid.sum().item())
    metrics: dict[str, float] = {}
    if total_valid == 0:
        return metrics

    metrics["opd/repetition/token_fraction"] = int(repetition_mask.sum().item()) / total_valid
    repetition_values = token_reward[repetition_mask]
    nonrepetition_values = token_reward[nonrepetition_mask]
    repetition_mean = _safe_mean(repetition_values)
    nonrepetition_mean = _safe_mean(nonrepetition_values)
    _put_if_present(metrics, "opd/repetition/advantage_mean", repetition_mean)
    _put_if_present(metrics, "opd/nonrepetition/advantage_mean", nonrepetition_mean)
    if repetition_mean is not None and nonrepetition_mean is not None and abs(nonrepetition_mean) > 1.0e-12:
        metrics["opd/repetition/advantage_ratio"] = repetition_mean / nonrepetition_mean

    total_abs_advantage = token_reward[valid].abs().sum()
    if torch.isfinite(total_abs_advantage) and float(total_abs_advantage.item()) > 0:
        metrics["opd/repetition/advantage_mass_share"] = float(
            repetition_values.abs().sum().item() / total_abs_advantage.item()
        )
    _put_if_present(
        metrics,
        "opd/repetition/student_logprob_mean",
        _safe_mean(student_logprobs_cpu[repetition_mask]),
    )
    _put_if_present(
        metrics,
        "opd/repetition/teacher_logprob_mean",
        _safe_mean(teacher_logprobs_cpu[repetition_mask]),
    )
    return metrics


def compute_opd_diagnostics_metrics(
    *,
    student_logprobs: torch.Tensor,
    response_mask: torch.Tensor,
    bin_size: int,
    response_ids: Optional[torch.Tensor] = None,
    teacher_logprobs: Optional[torch.Tensor] = None,
    student_topk_ids: Optional[torch.Tensor] = None,
    student_topk_logprobs: Optional[torch.Tensor] = None,
    teacher_topk_ids: Optional[torch.Tensor] = None,
    teacher_topk_logprobs: Optional[torch.Tensor] = None,
    teacher_logprob_nonfinite_positions: Optional[list[Any]] = None,
    entropy: Optional[torch.Tensor] = None,
    rm_scores: Optional[torch.Tensor] = None,
    response_texts: Optional[list[Any]] = None,
    high_gap_threshold: float = 1.0,
    overlap_topk: Optional[int] = None,
    special_token_ids: Optional[SpecialTokenIds] = None,
    max_response_length: Optional[int] = None,
    collect_repetition_metrics: bool = True,
    collect_train_text_zlib_repetition: bool = False,
    repetition_tail_chars: int = 10000,
    repetition_comp_ratio_threshold: float = 10.0,
    repetition_tail_token_window: int = 4096,
    bad_tail_min_position: int = BAD_TAIL_MIN_POSITION,
    bad_tail_deep_position: int = BAD_TAIL_DEEP_POSITION,
    bad_tail_extreme_position: int = BAD_TAIL_EXTREME_POSITION,
    bad_tail_overlap_threshold: float = BAD_TAIL_OVERLAP_THRESHOLD,
    bad_tail_teacher_prob_threshold: float = BAD_TAIL_TEACHER_PROB_THRESHOLD,
    bad_tail_gap_threshold: float = BAD_TAIL_GAP_THRESHOLD,
    bad_tail_reward_threshold: float = BAD_TAIL_REWARD_THRESHOLD,
    bad_tail_entropy_threshold: float = BAD_TAIL_ENTROPY_THRESHOLD,
) -> dict[str, float]:
    """Compute low-overhead OPD diagnostics from sampled-token tensors."""
    if student_logprobs.shape != response_mask.shape:
        raise ValueError(
            f"student_logprobs and response_mask must align, got {student_logprobs.shape=} and {response_mask.shape=}."
        )

    response_mask = response_mask.bool()
    metrics: dict[str, float] = {}
    response_lens = response_mask.sum(dim=1).detach().float().cpu()
    max_response_length = int(max_response_length or response_mask.shape[1])
    metrics.update(compute_bucket_denominator_metrics(response_mask, bin_size=bin_size))
    _put_if_present(metrics, "opd/response_len/mean", _safe_mean(response_lens))
    _put_if_present(metrics, "opd/response_len/p90", _safe_quantile(response_lens, 0.9))
    response_len_max = float(response_lens.max().item()) if response_lens.numel() else None
    _put_if_present(metrics, "opd/response_len/max", response_len_max)
    repetition_tail_mask: torch.Tensor | None = None
    if collect_repetition_metrics and response_ids is not None:
        repetition_metrics, repetition_tail_mask = compute_token_repetition_and_truncation_metrics(
            response_ids=response_ids,
            response_mask=response_mask,
            max_response_length=max_response_length,
            prefix="rollout",
            tail_token_window=repetition_tail_token_window,
            threshold=repetition_comp_ratio_threshold,
        )
        metrics.update(repetition_metrics)
        if collect_train_text_zlib_repetition and response_texts is not None:
            text_repetition_metrics, text_repetition_flags, _, _ = (
                compute_text_repetition_and_truncation_metrics_with_flags(
                    outputs=response_texts,
                    response_lengths=[int(length) for length in response_lens.tolist()],
                    max_response_length=max_response_length,
                    prefix="rollout",
                    tail_chars=repetition_tail_chars,
                    threshold=repetition_comp_ratio_threshold,
                )
            )
            metrics.update(text_repetition_metrics)
            if text_repetition_flags.size == response_mask.shape[0]:
                repetition_tail_mask = repetition_tail_mask_from_sample_flags(
                    response_mask=response_mask,
                    repetition_flags=text_repetition_flags,
                    tail_token_window=repetition_tail_token_window,
                )
    phase_info: ResponsePhaseInfo | None = None
    if response_ids is not None and special_token_ids is not None:
        phase_info = compute_phase_and_special_token_metrics(
            response_ids=response_ids,
            response_mask=response_mask,
            special_token_ids=special_token_ids,
            max_response_length=max_response_length,
        )
        metrics.update(special_token_resolution_metrics(special_token_ids))
        metrics.update(phase_info.metrics)

    student_logprobs_cpu = _to_cpu_float(student_logprobs)
    student_prob = student_logprobs_cpu.exp()
    student_prob_valid = response_mask.detach().bool().cpu() & torch.isfinite(student_prob)
    _put_if_present(metrics, "distillation/student_sampled_prob/mean", _masked_mean(student_prob, student_prob_valid))
    _put_if_present(
        metrics,
        "distillation/student_sampled_prob/p10",
        _masked_quantile(student_prob, student_prob_valid, 0.10),
    )
    _put_if_present(
        metrics,
        "distillation/student_sampled_prob/p50",
        _masked_quantile(student_prob, student_prob_valid, 0.50),
    )
    metrics.update(
        compute_coarse_token_stats(
            student_prob,
            student_prob_valid,
            prefix="distillation/student_sampled_prob",
            include_distribution=True,
        )
    )

    abs_gap_for_bad_tail: torch.Tensor | None = None
    token_reward_for_bad_tail: torch.Tensor | None = None
    teacher_prob_for_bad_tail: torch.Tensor | None = None
    top16_overlap_for_bad_tail: torch.Tensor | None = None
    student_entropy_for_bad_tail: torch.Tensor | None = None
    teacher_entropy_for_bad_tail: torch.Tensor | None = None

    if teacher_logprob_nonfinite_positions is not None:
        nonfinite_event_mask = positions_to_event_mask(
            teacher_logprob_nonfinite_positions,
            response_mask,
            context="teacher_logprob_nonfinite",
        )
        metrics.update(
            compute_event_position_metrics(
                event_mask=nonfinite_event_mask,
                response_mask=response_mask,
                metric_prefix="opd/teacher_logprob_nonfinite",
                suffix_separator="/",
                token_fraction_suffix="response_token_fraction",
                bin_prefix="opd/teacher_logprob_nonfinite_bin",
                bin_size=bin_size,
            )
        )

    if teacher_logprobs is not None:
        if teacher_logprobs.ndim == 3 and teacher_logprobs.shape[-1] == 1:
            teacher_logprobs = teacher_logprobs.squeeze(-1)
        if teacher_logprobs.shape == student_logprobs.shape:
            teacher_logprobs_cpu = _to_cpu_float(teacher_logprobs)
            response_mask_cpu = response_mask.detach().bool().cpu()
            gap_value_mask = (
                torch.isfinite(student_logprobs_cpu)
                & torch.isfinite(teacher_logprobs_cpu)
                & (student_logprobs_cpu.abs() <= MAX_REASONABLE_LOGPROB_ABS)
                & (teacher_logprobs_cpu.abs() <= MAX_REASONABLE_LOGPROB_ABS)
            )
            _put_if_present(
                metrics,
                "opd/gap_abs/outlier_token_fraction",
                _masked_event_fraction(~gap_value_mask, response_mask_cpu),
            )
            valid_gap_response_mask = response_mask_cpu & gap_value_mask
            token_reward = teacher_logprobs_cpu - student_logprobs_cpu
            abs_gap = token_reward.abs()
            abs_gap_for_bad_tail = abs_gap
            token_reward_for_bad_tail = token_reward
            teacher_prob_for_bad_tail = teacher_logprobs_cpu.exp()
            reward_values = _masked_values(token_reward, response_mask_cpu, gap_value_mask)
            abs_gap_values = _masked_values(abs_gap, response_mask_cpu, gap_value_mask)
            _put_if_present(metrics, "opd/reward/mean", _safe_mean(reward_values))
            _put_if_present(metrics, "opd/reward/p10", _safe_quantile(reward_values, 0.1))
            _put_if_present(metrics, "opd/reward/p90", _safe_quantile(reward_values, 0.9))
            _put_if_present(metrics, "opd/gap_abs/mean", _safe_mean(abs_gap_values))
            _put_if_present(metrics, "opd/gap_abs/p90", _safe_quantile(abs_gap_values, 0.9))
            abs_gap_max = float(abs_gap_values.max().item()) if abs_gap_values.numel() else None
            _put_if_present(metrics, "opd/gap_abs/max", abs_gap_max)
            if abs_gap_values.numel():
                high_gap_mask = (abs_gap > high_gap_threshold) & valid_gap_response_mask
                high_gap_metrics = compute_event_position_metrics(
                    event_mask=high_gap_mask,
                    response_mask=valid_gap_response_mask,
                    metric_prefix="opd/gap_abs/high",
                    suffix_separator="_",
                    token_fraction_suffix="token_fraction",
                    bin_prefix="opd/gap_abs/high_bin",
                    bin_size=bin_size,
                )
                metrics.update(high_gap_metrics)
                metrics["opd/gap_abs/high_fraction"] = high_gap_metrics["opd/gap_abs/high_token_fraction"]
            metrics.update(
                compute_binned_token_means(
                    abs_gap,
                    response_mask_cpu,
                    bin_size=bin_size,
                    prefix="opd/gap_abs_bin",
                    value_mask=gap_value_mask,
                )
            )
            metrics.update(
                compute_coarse_token_stats(
                    abs_gap,
                    response_mask_cpu,
                    prefix="opd/gap_abs",
                    value_mask=gap_value_mask,
                    include_distribution=True,
                )
            )
            metrics.update(
                compute_binned_token_means(
                    token_reward,
                    response_mask_cpu,
                    bin_size=bin_size,
                    prefix="opd/reward_bin",
                    value_mask=gap_value_mask,
                )
            )
            metrics.update(
                compute_coarse_token_stats(
                    token_reward,
                    response_mask_cpu,
                    prefix="opd/reward",
                    value_mask=gap_value_mask,
                    include_distribution=True,
                )
            )
            metrics.update(
                _head_tail_metrics(abs_gap, valid_gap_response_mask, prefix="opd/gap_abs", bin_size=bin_size)
            )
            metrics.update(
                _head_tail_metrics(token_reward, valid_gap_response_mask, prefix="opd/reward", bin_size=bin_size)
            )
            if repetition_tail_mask is not None:
                metrics.update(
                    compute_repetition_advantage_metrics(
                        repetition_tail_mask=repetition_tail_mask,
                        response_mask=response_mask,
                        student_logprobs=student_logprobs,
                        teacher_logprobs=teacher_logprobs,
                        value_mask=gap_value_mask,
                    )
                )

            if rm_scores is not None:
                seq_reward = (token_reward * valid_gap_response_mask).sum(dim=1) / valid_gap_response_mask.sum(
                    dim=1
                ).clamp_min(1)
                if rm_scores.ndim == response_mask.ndim + 1:
                    rm_scores = rm_scores.sum(dim=-1)
                labels = rm_scores.detach().float().sum(dim=1) > 0
                auc = binary_auc(seq_reward, labels)
                _put_if_present(metrics, "opd/reward/correct_auc", auc)
                correct_mean = _safe_mean(seq_reward.detach().float().cpu()[labels.detach().cpu()])
                incorrect_mean = _safe_mean(seq_reward.detach().float().cpu()[~labels.detach().cpu()])
                _put_if_present(metrics, "opd/reward/correct_mean", correct_mean)
                _put_if_present(metrics, "opd/reward/incorrect_mean", incorrect_mean)

    if rm_scores is not None:
        scalar_token_reward = rm_scores.detach().float()
        if scalar_token_reward.ndim == response_mask.ndim + 1:
            scalar_token_reward = scalar_token_reward.sum(dim=-1)
        if scalar_token_reward.shape == response_mask.shape:
            scalar_reward_cpu = _to_cpu_float(scalar_token_reward)
            reward_valid = response_mask.detach().bool().cpu() & torch.isfinite(scalar_reward_cpu)
            reward_values = scalar_reward_cpu[reward_valid]
            _put_distribution_metrics(metrics, "distillation/token_reward_direct", reward_values)
            if reward_values.numel():
                metrics["distillation/token_reward_direct/negative_ratio"] = float((reward_values < 0).float().mean())
            metrics.update(
                compute_coarse_token_stats(
                    scalar_reward_cpu,
                    reward_valid,
                    prefix="distillation/token_reward_direct",
                    include_distribution=True,
                )
            )
            if token_reward_for_bad_tail is None:
                token_reward_for_bad_tail = scalar_reward_cpu

    if (
        overlap_topk is not None
        and response_ids is not None
        and teacher_topk_ids is not None
        and teacher_topk_logprobs is not None
    ):
        metrics.update(
            compute_overlap_diagnostics(
                response_ids=response_ids,
                response_mask=response_mask,
                teacher_topk_ids=teacher_topk_ids,
                teacher_topk_logprobs=teacher_topk_logprobs,
                sampled_teacher_logprobs=teacher_logprobs,
                bin_size=bin_size,
                topk=overlap_topk,
            )
        )

    if (
        overlap_topk is not None
        and student_topk_ids is not None
        and student_topk_logprobs is not None
        and teacher_topk_ids is not None
        and teacher_topk_logprobs is not None
    ):
        available_for_bad_tail = min(16, student_topk_ids.shape[-1], teacher_topk_ids.shape[-1])
        if available_for_bad_tail > 0:
            student_ids16 = student_topk_ids[..., :available_for_bad_tail].detach().cpu()
            teacher_ids16 = teacher_topk_ids[..., :available_for_bad_tail].detach().cpu()
            student_lp16 = _to_cpu_float(student_topk_logprobs[..., :available_for_bad_tail])
            teacher_lp16 = _to_cpu_float(teacher_topk_logprobs[..., :available_for_bad_tail])
            student_finite16 = torch.isfinite(student_lp16)
            teacher_finite16 = torch.isfinite(teacher_lp16)
            matches16 = student_ids16.unsqueeze(-1) == teacher_ids16.unsqueeze(-2)
            matches16 = matches16 & student_finite16.unsqueeze(-1) & teacher_finite16.unsqueeze(-2)
            top16_overlap_for_bad_tail = matches16.any(dim=-1).sum(dim=-1).float() / float(available_for_bad_tail)
            if entropy is None:
                student_entropy_for_bad_tail = _topk_residual_bucket_entropy(student_lp16, student_finite16)
            teacher_entropy_for_bad_tail = _topk_residual_bucket_entropy(teacher_lp16, teacher_finite16)
        metrics.update(
            compute_rethinking_opd_diagnostics(
                response_mask=response_mask,
                student_topk_ids=student_topk_ids,
                student_topk_logprobs=student_topk_logprobs,
                teacher_topk_ids=teacher_topk_ids,
                teacher_topk_logprobs=teacher_topk_logprobs,
                bin_size=bin_size,
                topk=overlap_topk,
            )
        )

    if entropy is not None:
        if entropy.shape != response_mask.shape:
            raise ValueError(f"entropy and response_mask must align, got {entropy.shape=} and {response_mask.shape=}.")
        entropy_cpu = _to_cpu_float(entropy)
        response_mask_cpu = response_mask.detach().bool().cpu()
        entropy_value_mask = (
            torch.isfinite(entropy_cpu) & (entropy_cpu >= 0.0) & (entropy_cpu <= MAX_REASONABLE_ENTROPY)
        )
        _put_if_present(
            metrics,
            "opd/entropy/outlier_token_fraction",
            _masked_event_fraction(~entropy_value_mask, response_mask_cpu),
        )
        valid_entropy_response_mask = response_mask_cpu & entropy_value_mask
        entropy_values = _masked_values(entropy_cpu, response_mask_cpu, entropy_value_mask)
        _put_if_present(metrics, "opd/entropy/mean", _safe_mean(entropy_values))
        _put_if_present(metrics, "opd/entropy/p90", _safe_quantile(entropy_values, 0.9))
        metrics.update(
            compute_binned_token_means(
                entropy_cpu,
                response_mask_cpu,
                bin_size=bin_size,
                prefix="opd/entropy_bin",
                value_mask=entropy_value_mask,
            )
        )
        metrics.update(
            _head_tail_metrics(entropy_cpu, valid_entropy_response_mask, prefix="opd/entropy", bin_size=bin_size)
        )
        metrics.update(
            compute_coarse_token_stats(
                entropy_cpu,
                valid_entropy_response_mask,
                prefix="opd/entropy/student",
                include_distribution=True,
            )
        )
        student_entropy_for_bad_tail = entropy_cpu

    if special_token_ids is not None:
        metrics.update(
            compute_special_token_topk_metrics(
                response_mask=response_mask,
                special_token_ids=special_token_ids,
                bin_size=bin_size,
                student_topk_ids=student_topk_ids,
                student_topk_logprobs=student_topk_logprobs,
                teacher_topk_ids=teacher_topk_ids,
                teacher_topk_logprobs=teacher_topk_logprobs,
                thinking_mask=None if phase_info is None else phase_info.thinking_mask,
                answering_mask=None if phase_info is None else phase_info.answering_mask,
            )
        )

    metrics.update(
        compute_bad_tail_metrics(
            response_mask=response_mask,
            clipped_mask=response_lens >= max_response_length,
            phase_carrier_mask=None if phase_info is None else phase_info.carrier_mask,
            abs_gap=abs_gap_for_bad_tail,
            token_reward=token_reward_for_bad_tail,
            teacher_sampled_prob=teacher_prob_for_bad_tail,
            top16_overlap=top16_overlap_for_bad_tail,
            student_entropy=student_entropy_for_bad_tail,
            teacher_entropy=teacher_entropy_for_bad_tail,
            min_position=bad_tail_min_position,
            deep_position=bad_tail_deep_position,
            extreme_position=bad_tail_extreme_position,
            overlap_threshold=bad_tail_overlap_threshold,
            teacher_prob_threshold=bad_tail_teacher_prob_threshold,
            gap_threshold=bad_tail_gap_threshold,
            reward_threshold=bad_tail_reward_threshold,
            entropy_threshold=bad_tail_entropy_threshold,
        )
    )

    return metrics


def compute_entropy_heatmap_row(
    entropy: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    bin_size: int,
) -> np.ndarray:
    """Return one heatmap row: mean entropy per response-position bin."""
    entropy_cpu = _to_cpu_float(entropy)
    entropy_value_mask = torch.isfinite(entropy_cpu) & (entropy_cpu >= 0.0) & (entropy_cpu <= MAX_REASONABLE_ENTROPY)
    return compute_value_heatmap_row(
        values=entropy_cpu,
        response_mask=response_mask,
        bin_size=bin_size,
        value_name="entropy",
        value_mask=entropy_value_mask,
    )


def compute_gap_heatmap_row(
    *,
    student_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    response_mask: torch.Tensor,
    bin_size: int,
) -> np.ndarray:
    """Return one heatmap row: mean sampled-token teacher/student absolute gap per response-position bin."""
    if teacher_logprobs.ndim == 3 and teacher_logprobs.shape[-1] == 1:
        teacher_logprobs = teacher_logprobs.squeeze(-1)
    if student_logprobs.shape != teacher_logprobs.shape:
        raise ValueError(
            "student_logprobs and teacher_logprobs must align for gap heatmap, "
            f"got {student_logprobs.shape=} and {teacher_logprobs.shape=}."
        )
    student_logprobs_cpu = _to_cpu_float(student_logprobs)
    teacher_logprobs_cpu = _to_cpu_float(teacher_logprobs)
    gap_value_mask = (
        torch.isfinite(student_logprobs_cpu)
        & torch.isfinite(teacher_logprobs_cpu)
        & (student_logprobs_cpu.abs() <= MAX_REASONABLE_LOGPROB_ABS)
        & (teacher_logprobs_cpu.abs() <= MAX_REASONABLE_LOGPROB_ABS)
    )
    return compute_value_heatmap_row(
        values=(teacher_logprobs_cpu - student_logprobs_cpu).abs(),
        response_mask=response_mask,
        bin_size=bin_size,
        value_name="gap_abs",
        value_mask=gap_value_mask,
    )


def compute_gap_high_heatmap_row(
    *,
    student_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    response_mask: torch.Tensor,
    bin_size: int,
    high_gap_threshold: float,
) -> np.ndarray:
    """Return one heatmap row: high-gap token rate per response-position bin."""
    if teacher_logprobs.ndim == 3 and teacher_logprobs.shape[-1] == 1:
        teacher_logprobs = teacher_logprobs.squeeze(-1)
    if student_logprobs.shape != teacher_logprobs.shape:
        raise ValueError(
            "student_logprobs and teacher_logprobs must align for high-gap heatmap, "
            f"got {student_logprobs.shape=} and {teacher_logprobs.shape=}."
        )
    student_logprobs_cpu = _to_cpu_float(student_logprobs)
    teacher_logprobs_cpu = _to_cpu_float(teacher_logprobs)
    gap_value_mask = (
        torch.isfinite(student_logprobs_cpu)
        & torch.isfinite(teacher_logprobs_cpu)
        & (student_logprobs_cpu.abs() <= MAX_REASONABLE_LOGPROB_ABS)
        & (teacher_logprobs_cpu.abs() <= MAX_REASONABLE_LOGPROB_ABS)
    )
    response_mask_cpu = response_mask.detach().bool().cpu()
    event_mask = ((teacher_logprobs_cpu - student_logprobs_cpu).abs() > high_gap_threshold) & gap_value_mask
    return compute_event_heatmap_row(
        event_mask=event_mask,
        response_mask=response_mask_cpu & gap_value_mask,
        bin_size=bin_size,
        event_name="gap_abs_high",
    )


def compute_teacher_logprob_nonfinite_heatmap_row(
    *,
    teacher_logprob_nonfinite_positions: Optional[list[Any]],
    response_mask: torch.Tensor,
    bin_size: int,
) -> np.ndarray:
    """Return one heatmap row: nonfinite teacher scoring event rate per response-position bin."""
    event_mask = positions_to_event_mask(
        teacher_logprob_nonfinite_positions,
        response_mask,
        context="teacher_logprob_nonfinite",
    )
    return compute_event_heatmap_row(
        event_mask=event_mask,
        response_mask=response_mask,
        bin_size=bin_size,
        event_name="teacher_logprob_nonfinite",
    )


def compute_event_heatmap_row(
    *,
    event_mask: torch.Tensor,
    response_mask: torch.Tensor,
    bin_size: int,
    event_name: str,
) -> np.ndarray:
    """Return one heatmap row: event rate per response-position bin."""
    if event_mask.shape != response_mask.shape:
        raise ValueError(
            f"{event_name} and response_mask must align, got event_mask.shape={event_mask.shape} "
            f"and {response_mask.shape=}."
        )
    return compute_value_heatmap_row(
        values=event_mask.detach().float(),
        response_mask=response_mask,
        bin_size=bin_size,
        value_name=event_name,
    )


def compute_value_heatmap_row(
    *,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    bin_size: int,
    value_name: str,
    value_mask: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """Return one heatmap row: mean value per response-position bin."""
    if bin_size <= 0:
        raise ValueError(f"bin_size must be positive, got {bin_size}.")
    if values.shape != response_mask.shape:
        raise ValueError(
            f"{value_name} and response_mask must align, got values.shape={values.shape} and {response_mask.shape=}."
        )
    values = _to_cpu_float(values)
    mask = response_mask.detach().bool().cpu()
    value_mask_cpu = value_mask.detach().bool().cpu() if value_mask is not None else None
    row = []
    for start in range(0, values.shape[1], bin_size):
        end = min(start + bin_size, values.shape[1])
        bin_value_mask = value_mask_cpu[:, start:end] if value_mask_cpu is not None else None
        mean = _safe_mean(_masked_values(values[:, start:end], mask[:, start:end], bin_value_mask))
        row.append(np.nan if mean is None else mean)
    return np.asarray(row, dtype=np.float32)


def _as_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    elif isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, list | tuple):
        return []
    return [int(item) for item in value]


def _lcs_length(left: list[int], right: list[int]) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for left_item in left:
        current = [0]
        for idx, right_item in enumerate(right, start=1):
            if left_item == right_item:
                current.append(previous[idx - 1] + 1)
            else:
                current.append(max(previous[idx], current[-1]))
        previous = current
    return previous[-1]


def compute_rewrite_overlap_metrics(original_student_segment_ids: Any, teacher_segment_ids: Any) -> dict[str, float]:
    """Compare a replaced student segment with the teacher rewrite."""
    original_ids = _as_int_list(original_student_segment_ids)
    teacher_ids = _as_int_list(teacher_segment_ids)
    max_len = max(len(original_ids), len(teacher_ids), 1)
    min_len = min(len(original_ids), len(teacher_ids))
    exact_matches = sum(1 for idx in range(min_len) if original_ids[idx] == teacher_ids[idx])
    return {
        "rewrite_exact_position_overlap": exact_matches / max_len,
        "rewrite_lcs_ratio": _lcs_length(original_ids, teacher_ids) / max_len,
        "rewrite_length_ratio": min_len / max_len,
    }


def compute_intervention_metrics(
    *,
    intervention_counts: list[Any],
    interventions: list[Any],
    intervention_skipped: list[Any] | None = None,
    response_mask: torch.Tensor,
) -> dict[str, float]:
    """Aggregate teacher-intervention metadata collected during rollout."""
    counts = np.asarray([0 if value is None else int(value) for value in intervention_counts], dtype=np.float32)
    response_mask_bool = response_mask.detach().bool().cpu()
    valid_lengths_tensor = response_mask_bool.sum(dim=1)
    metrics = {
        "opd_intervention/rate": float((counts > 0).mean()) if counts.size else 0.0,
        "opd_intervention/count_mean": float(counts.mean()) if counts.size else 0.0,
        "opd_intervention/count_max": float(counts.max()) if counts.size else 0.0,
        "opd_intervention/zero_trainable_sample_fraction": float((valid_lengths_tensor == 0).float().mean().item())
        if valid_lengths_tensor.numel()
        else 0.0,
    }

    total_valid_tokens = int(valid_lengths_tensor.sum().item())
    metrics["opd_intervention/trainable_token_fraction"] = total_valid_tokens / max(response_mask.numel(), 1)
    teacher_tokens = 0
    suffix_tokens = 0
    first_positions = []
    first_position_fracs = []
    scores = []
    rewrite_exact_position_overlaps = []
    rewrite_lcs_ratios = []
    rewrite_length_ratios = []
    valid_lengths = valid_lengths_tensor.tolist()
    for sample_idx, sample_interventions in enumerate(interventions):
        if sample_interventions is None:
            continue
        for item_idx, item in enumerate(sample_interventions):
            if not isinstance(item, dict):
                continue
            teacher_tokens += int(item.get("teacher_tokens", 0) or 0)
            suffix_tokens += int(item.get("student_suffix_tokens", 0) or 0)
            score = item.get("score")
            if score is not None:
                scores.append(float(score))
            if {
                "rewrite_exact_position_overlap",
                "rewrite_lcs_ratio",
                "rewrite_length_ratio",
            }.issubset(item):
                rewrite_exact_position_overlaps.append(float(item["rewrite_exact_position_overlap"]))
                rewrite_lcs_ratios.append(float(item["rewrite_lcs_ratio"]))
                rewrite_length_ratios.append(float(item["rewrite_length_ratio"]))
            elif "original_student_segment_ids" in item and "teacher_segment_ids" in item:
                rewrite_metrics = compute_rewrite_overlap_metrics(
                    item["original_student_segment_ids"],
                    item["teacher_segment_ids"],
                )
                rewrite_exact_position_overlaps.append(rewrite_metrics["rewrite_exact_position_overlap"])
                rewrite_lcs_ratios.append(rewrite_metrics["rewrite_lcs_ratio"])
                rewrite_length_ratios.append(rewrite_metrics["rewrite_length_ratio"])
            if item_idx == 0:
                start = int(item.get("start", 0) or 0)
                first_positions.append(start)
                length = max(int(valid_lengths[sample_idx]), 1)
                first_position_fracs.append(start / length)

    denom = max(total_valid_tokens, 1)
    metrics["opd_intervention/teacher_token_fraction"] = teacher_tokens / denom
    metrics["opd_intervention/student_suffix_token_fraction"] = suffix_tokens / denom
    if first_positions:
        metrics["opd_intervention/first_position_mean"] = float(np.mean(first_positions))
        metrics["opd_intervention/first_position_frac_mean"] = float(np.mean(first_position_fracs))
    if scores:
        metrics["opd_intervention/gap_score_mean"] = float(np.mean(scores))
        metrics["opd_intervention/gap_score_max"] = float(np.max(scores))
    if rewrite_exact_position_overlaps:
        metrics["opd_intervention/rewrite_exact_position_overlap_mean"] = float(
            np.mean(rewrite_exact_position_overlaps)
        )
    if rewrite_lcs_ratios:
        metrics["opd_intervention/rewrite_lcs_ratio_mean"] = float(np.mean(rewrite_lcs_ratios))
    if rewrite_length_ratios:
        metrics["opd_intervention/rewrite_length_ratio_mean"] = float(np.mean(rewrite_length_ratios))

    skipped_counts = []
    skipped_reason_counts: dict[str, int] = {}
    for sample_skipped in intervention_skipped or []:
        if sample_skipped is None:
            skipped_counts.append(0)
            continue
        if isinstance(sample_skipped, dict):
            skipped_items = [sample_skipped]
        elif isinstance(sample_skipped, list | tuple):
            skipped_items = [item for item in sample_skipped if isinstance(item, dict)]
        else:
            skipped_items = []
        skipped_counts.append(len(skipped_items))
        for item in skipped_items:
            reason = item.get("reason")
            if reason is not None:
                skipped_reason_counts[str(reason)] = skipped_reason_counts.get(str(reason), 0) + 1
    skipped_array = np.asarray(skipped_counts, dtype=np.float32)
    metrics["opd_intervention/skipped_count_mean"] = float(skipped_array.mean()) if skipped_array.size else 0.0
    metrics["opd_intervention/skipped_count_max"] = float(skipped_array.max()) if skipped_array.size else 0.0
    metrics["opd_intervention/skipped_sample_fraction"] = (
        float((skipped_array > 0).mean()) if skipped_array.size else 0.0
    )
    skipped_denom = max(int(skipped_array.sum()), 1)
    for reason, count in skipped_reason_counts.items():
        metrics[f"opd_intervention/skipped_reason/{reason}"] = count / skipped_denom
    return metrics


@dataclass
class OpdDiagnosticsTracker:
    """Keeps a bounded history for OPD diagnostic heatmaps."""

    history_steps: int
    bin_size: int
    rows: deque[np.ndarray] = field(init=False)
    steps: deque[int] = field(init=False)

    def __post_init__(self):
        self.rows = deque(maxlen=self.history_steps)
        self.steps = deque(maxlen=self.history_steps)

    def add_entropy_row(self, step: int, row: np.ndarray) -> None:
        self.add_row(step, row)

    def add_gap_row(self, step: int, row: np.ndarray) -> None:
        self.add_row(step, row)

    def add_gap_high_row(self, step: int, row: np.ndarray) -> None:
        self.add_row(step, row)

    def add_teacher_logprob_nonfinite_row(self, step: int, row: np.ndarray) -> None:
        self.add_row(step, row)

    def add_row(self, step: int, row: np.ndarray) -> None:
        self.steps.append(step)
        self.rows.append(row)

    def render_entropy_heatmap(self):
        return self.render_heatmap(
            title="Student Entropy by Response Position",
            colorbar_label="Mean entropy",
        )

    def render_gap_heatmap(self):
        return self.render_heatmap(
            title="Teacher-Student Absolute Gap by Response Position",
            colorbar_label="Mean absolute logprob gap",
        )

    def render_gap_high_heatmap(self):
        return self.render_heatmap(
            title="High Teacher-Student Gap Rate by Response Position",
            colorbar_label="High-gap token rate",
        )

    def render_teacher_logprob_nonfinite_heatmap(self):
        return self.render_heatmap(
            title="Nonfinite Teacher Logprob Rate by Response Position",
            colorbar_label="Nonfinite teacher logprob rate",
        )

    def render_heatmap(self, *, title: str, colorbar_label: str):
        if not self.rows:
            return None

        try:
            import matplotlib
        except ImportError:
            return None

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        width = max(row.shape[0] for row in self.rows)
        heatmap = np.full((len(self.rows), width), np.nan, dtype=np.float32)
        for idx, row in enumerate(self.rows):
            heatmap[idx, : row.shape[0]] = row

        fig, ax = plt.subplots(figsize=(max(6.0, width * 0.8), max(3.0, len(self.rows) * 0.18)))
        image = ax.imshow(heatmap, aspect="auto", interpolation="nearest", origin="lower")
        ax.set_title(title)
        ax.set_xlabel("Response token position bin")
        ax.set_ylabel("Training step")
        ax.set_xticks(range(width))
        ax.set_xticklabels([f"{i * self.bin_size}-{(i + 1) * self.bin_size}" for i in range(width)], rotation=45)
        step_values = list(self.steps)
        if step_values:
            tick_count = min(8, len(step_values))
            tick_positions = np.linspace(0, len(step_values) - 1, tick_count, dtype=int)
            ax.set_yticks(tick_positions)
            ax.set_yticklabels([str(step_values[i]) for i in tick_positions])
        fig.colorbar(image, ax=ax, label=colorbar_label)
        fig.tight_layout()
        return fig
