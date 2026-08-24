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
"""
Metrics related to the PPO trainer.
"""

import logging
import zlib
from collections import defaultdict
from functools import partial
from typing import Any, Callable

import numpy as np
import torch

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.utils.import_utils import deprecated

logger = logging.getLogger(__name__)

_FINISH_REASON_LENGTH = "length"
_FINISH_REASON_STOP = "stop"
_FINISH_REASON_EOS = "eos"
_FINISH_REASON_ABORT = "abort"
_FINISH_REASON_OTHER = "other"
_FINISH_REASON_CATEGORIES = (
    _FINISH_REASON_LENGTH,
    _FINISH_REASON_EOS,
    _FINISH_REASON_STOP,
    _FINISH_REASON_ABORT,
    _FINISH_REASON_OTHER,
)


@deprecated("verl.utils.metric.reduce_metrics")
def reduce_metrics(metrics: dict[str, list[Any]]) -> dict[str, Any]:
    """
    Reduces a dictionary of metric lists by computing the mean of each list.

    Args:
        metrics: A dictionary mapping metric names to lists of metric values.

    Returns:
        A dictionary with the same keys but with each list replaced by its mean value.

    Example:
        >>> metrics = {"loss": [1.0, 2.0, 3.0], "accuracy": [0.8, 0.9, 0.7]}
        >>> reduce_metrics(metrics)
        {"loss": 2.0, "accuracy": 0.8}
    """
    from verl.utils.metric import reduce_metrics

    return reduce_metrics(metrics)


def _compute_response_info(batch: DataProto) -> dict[str, Any]:
    """
    Computes information about prompts and responses from a batch.

    This is an internal helper function that extracts masks and lengths for prompts and responses.

    Args:
        batch: A DataProto object containing batch data with responses and attention masks.

    Returns:
        A dictionary containing:
            - response_mask: Attention mask for the response tokens
            - prompt_length: Tensor of prompt lengths for each item in the batch
            - response_length: Tensor of response lengths for each item in the batch
    """
    if "prompt_length" in batch.batch and "response_length" in batch.batch:
        return dict(
            prompt_length=batch.batch["prompt_length"],
            response_length=batch.batch["response_length"],
        )

    response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-response_length]
    response_mask = batch.batch["attention_mask"][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        prompt_length=prompt_length,
        response_length=response_length,
    )


def _safe_tensor_quantile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return float("nan")
    return torch.quantile(values.float(), q).detach().item()


def _safe_np_quantile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values.astype(np.float32), q))


def _length_distribution_metrics(prefix: str, lengths: torch.Tensor) -> dict[str, Any]:
    lengths = lengths.detach().float()
    if lengths.numel() == 0:
        return {}
    return {
        f"{prefix}/count": float(lengths.numel()),
        f"{prefix}/mean": torch.mean(lengths).detach().item(),
        f"{prefix}/std": torch.std(lengths, unbiased=False).detach().item(),
        f"{prefix}/min": torch.min(lengths).detach().item(),
        f"{prefix}/p50": _safe_tensor_quantile(lengths, 0.50),
        f"{prefix}/p75": _safe_tensor_quantile(lengths, 0.75),
        f"{prefix}/p90": _safe_tensor_quantile(lengths, 0.90),
        f"{prefix}/p95": _safe_tensor_quantile(lengths, 0.95),
        f"{prefix}/p99": _safe_tensor_quantile(lengths, 0.99),
        f"{prefix}/max": torch.max(lengths).detach().item(),
    }


def _length_distribution_metrics_from_values(prefix: str, lengths: list[int] | np.ndarray) -> dict[str, Any]:
    values = np.asarray(lengths, dtype=np.float32)
    if values.size == 0:
        return {}
    return {
        f"{prefix}/count": float(values.size),
        f"{prefix}/mean": float(values.mean()),
        f"{prefix}/std": float(values.std()),
        f"{prefix}/min": float(values.min()),
        f"{prefix}/p50": _safe_np_quantile(values, 0.50),
        f"{prefix}/p75": _safe_np_quantile(values, 0.75),
        f"{prefix}/p90": _safe_np_quantile(values, 0.90),
        f"{prefix}/p95": _safe_np_quantile(values, 0.95),
        f"{prefix}/p99": _safe_np_quantile(values, 0.99),
        f"{prefix}/max": float(values.max()),
    }


def _distribution_metrics_from_values(prefix: str, values: list[float] | np.ndarray) -> dict[str, float]:
    values_np = np.asarray(values, dtype=np.float32)
    values_np = values_np[np.isfinite(values_np)]
    if values_np.size == 0:
        return {}
    return {
        f"{prefix}/count": float(values_np.size),
        f"{prefix}/mean": float(values_np.mean()),
        f"{prefix}/std": float(values_np.std()),
        f"{prefix}/min": float(values_np.min()),
        f"{prefix}/p50": _safe_np_quantile(values_np, 0.50),
        f"{prefix}/p75": _safe_np_quantile(values_np, 0.75),
        f"{prefix}/p90": _safe_np_quantile(values_np, 0.90),
        f"{prefix}/p95": _safe_np_quantile(values_np, 0.95),
        f"{prefix}/p99": _safe_np_quantile(values_np, 0.99),
        f"{prefix}/max": float(values_np.max()),
    }


def _as_list(values: Any) -> list[Any]:
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


def _normalize_finish_reason(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        value = value.get("type", value.get("finish_reason", value.get("reason")))
    if value is None:
        return _FINISH_REASON_OTHER
    normalized = str(value).strip().lower()
    if normalized == _FINISH_REASON_LENGTH:
        return _FINISH_REASON_LENGTH
    if normalized in {"eos", "eos_token", "end", "end_of_sequence"}:
        return _FINISH_REASON_EOS
    if normalized == _FINISH_REASON_STOP:
        return _FINISH_REASON_STOP
    if normalized in {"abort", "aborted", "cancel", "cancelled", "canceled"}:
        return _FINISH_REASON_ABORT
    return _FINISH_REASON_OTHER


def _aligned_finish_reasons(finish_reasons: Any, size: int) -> list[str]:
    reasons = [_normalize_finish_reason(value) for value in _as_list(finish_reasons)]
    if not reasons:
        return []
    if len(reasons) < size:
        reasons.extend([_FINISH_REASON_OTHER] * (size - len(reasons)))
    return reasons[:size]


def compute_finish_reason_metrics(finish_reasons: Any, prefix: str = "rollout/finish_reason") -> dict[str, float]:
    """Compute normalized generation finish-reason rates."""
    reasons = [_normalize_finish_reason(value) for value in _as_list(finish_reasons)]
    if not reasons:
        return {}
    total = len(reasons)
    return {f"{prefix}/{category}_rate": reasons.count(category) / total for category in _FINISH_REASON_CATEGORIES}


def compute_truncation_flags(
    response_lengths: torch.Tensor | np.ndarray | list[int],
    max_response_length: int,
    finish_reasons: Any = None,
) -> np.ndarray:
    """Return per-sample truncation flags, using finish_reason=length before length fallback."""
    lengths = np.asarray(_as_list(response_lengths), dtype=np.float32)
    if lengths.size == 0:
        return np.asarray([], dtype=bool)
    reasons = _aligned_finish_reasons(finish_reasons, int(lengths.size))
    if reasons:
        return np.asarray([reason == _FINISH_REASON_LENGTH for reason in reasons], dtype=bool)
    return lengths == float(max_response_length)


def _zlib_compression_ratio(payload: bytes) -> float:
    if not payload:
        return float("nan")
    compressed = zlib.compress(payload)
    if not compressed:
        return float("nan")
    return len(payload) / len(compressed)


def text_zlib_compression_ratio(text: Any, tail_chars: int = 10000) -> tuple[float, bool]:
    """Return tail compression ratio and eligibility for text-zlib repetition detection."""
    if tail_chars <= 0:
        raise ValueError(f"tail_chars must be positive, got {tail_chars}.")
    text = "" if text is None else str(text)
    if len(text) < tail_chars:
        return float("nan"), False
    tail = text[-tail_chars:]
    return _zlib_compression_ratio(tail.encode("utf-8")), True


def text_zlib_repetition_flags(
    outputs: list[Any] | tuple[Any, ...],
    *,
    tail_chars: int = 10000,
    threshold: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute text-zlib repetition flags, compression ratios, and eligibility masks."""
    if not np.isfinite(threshold):
        raise ValueError(f"threshold must be finite, got {threshold}.")
    flags: list[bool] = []
    ratios: list[float] = []
    eligible: list[bool] = []
    for output in outputs:
        ratio, is_eligible = text_zlib_compression_ratio(output, tail_chars=tail_chars)
        ratios.append(ratio)
        eligible.append(is_eligible)
        flags.append(bool(is_eligible and np.isfinite(ratio) and ratio > threshold))
    return np.asarray(flags, dtype=bool), np.asarray(ratios, dtype=np.float32), np.asarray(eligible, dtype=bool)


def repetition_tail_mask_from_sample_flags(
    response_mask: torch.Tensor,
    repetition_flags: Any,
    *,
    tail_token_window: int = 4096,
) -> torch.Tensor:
    """Build a token tail mask for samples flagged as repetitive by a response-level detector."""
    if tail_token_window <= 0:
        raise ValueError(f"tail_token_window must be positive, got {tail_token_window}.")
    response_mask_cpu = response_mask.detach().bool().cpu()
    flags = np.asarray(_as_list(repetition_flags), dtype=bool).reshape(-1)
    if flags.size != response_mask_cpu.shape[0]:
        raise ValueError(
            f"repetition_flags must match batch size, got {flags.size=} and batch={response_mask_cpu.shape[0]}."
        )
    tail_mask = torch.zeros_like(response_mask_cpu, dtype=torch.bool)
    for idx, repeated in enumerate(flags):
        if not repeated:
            continue
        valid_positions = torch.nonzero(response_mask_cpu[idx], as_tuple=False).flatten()
        if valid_positions.numel() == 0:
            continue
        selected_positions = valid_positions[-min(tail_token_window, valid_positions.numel()) :]
        tail_mask[idx, selected_positions] = True
    return tail_mask


def token_zlib_repetition_flags(
    response_ids: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    tail_token_window: int = 4096,
    threshold: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
    """Compute token-id zlib repetition flags and the repeated-tail token mask."""
    if tail_token_window <= 0:
        raise ValueError(f"tail_token_window must be positive, got {tail_token_window}.")
    if not np.isfinite(threshold):
        raise ValueError(f"threshold must be finite, got {threshold}.")
    if response_ids.shape != response_mask.shape:
        raise ValueError(
            f"response_ids and response_mask must align, got {response_ids.shape=} and {response_mask.shape=}."
        )
    response_ids_cpu = response_ids.detach().cpu()
    response_mask_cpu = response_mask.detach().bool().cpu()
    tail_mask = torch.zeros_like(response_mask_cpu, dtype=torch.bool)
    flags: list[bool] = []
    ratios: list[float] = []
    eligible: list[bool] = []
    for idx in range(response_ids_cpu.shape[0]):
        valid_positions = torch.nonzero(response_mask_cpu[idx], as_tuple=False).flatten()
        if valid_positions.numel() < tail_token_window:
            flags.append(False)
            ratios.append(float("nan"))
            eligible.append(False)
            continue
        selected_positions = valid_positions[-tail_token_window:]
        token_values = response_ids_cpu[idx, selected_positions].to(dtype=torch.int64).numpy()
        ratio = _zlib_compression_ratio(token_values.tobytes())
        repeated = bool(np.isfinite(ratio) and ratio > threshold)
        flags.append(repeated)
        ratios.append(ratio)
        eligible.append(True)
        if repeated:
            tail_mask[idx, selected_positions] = True
    return (
        np.asarray(flags, dtype=bool),
        np.asarray(ratios, dtype=np.float32),
        np.asarray(eligible, dtype=bool),
        tail_mask,
    )


def compute_text_repetition_and_truncation_metrics(
    *,
    outputs: list[Any],
    response_lengths: list[int] | None,
    max_response_length: int | None,
    finish_reasons: Any = None,
    prefix: str = "val-aux",
    tail_chars: int = 10000,
    threshold: float = 10.0,
) -> dict[str, float]:
    """Compute validation-style text-zlib repetition, truncation, and response-length metrics."""
    metrics, _, _, _ = compute_text_repetition_and_truncation_metrics_with_flags(
        outputs=outputs,
        response_lengths=response_lengths,
        max_response_length=max_response_length,
        finish_reasons=finish_reasons,
        prefix=prefix,
        tail_chars=tail_chars,
        threshold=threshold,
    )
    return metrics


def compute_text_repetition_and_truncation_metrics_with_flags(
    *,
    outputs: list[Any],
    response_lengths: list[int] | None,
    max_response_length: int | None,
    finish_reasons: Any = None,
    prefix: str = "val-aux",
    tail_chars: int = 10000,
    threshold: float = 10.0,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Compute text-zlib metrics and return response-level repetition flags."""
    metrics: dict[str, float] = {}
    flags, ratios, eligible = text_zlib_repetition_flags(outputs, tail_chars=tail_chars, threshold=threshold)
    if flags.size:
        metrics[f"{prefix}/repetition_rate_zlib"] = float(flags.mean())
        metrics[f"{prefix}/repetition_eligible_rate_zlib"] = float(eligible.mean())
        metrics.update(_distribution_metrics_from_values(f"{prefix}/repetition_comp_ratio_zlib", ratios))

    if response_lengths is not None:
        metrics.update(_length_distribution_metrics_from_values(f"{prefix}/response_length", response_lengths))
        if max_response_length is not None:
            truncated = compute_truncation_flags(response_lengths, int(max_response_length), finish_reasons)
            if truncated.size:
                metrics[f"{prefix}/truncation_rate"] = float(truncated.mean())
                if flags.size == truncated.size:
                    metrics[f"{prefix}/truncation_repetition_joint_rate"] = float((truncated & flags).mean())
                    truncated_count = int(truncated.sum())
                    repetition_count = int(flags.sum())
                    if truncated_count:
                        metrics[f"{prefix}/repetition_given_truncation"] = (
                            float((truncated & flags).sum()) / truncated_count
                        )
                    if repetition_count:
                        metrics[f"{prefix}/truncation_given_repetition"] = (
                            float((truncated & flags).sum()) / repetition_count
                        )
    return metrics, flags, ratios, eligible


def compute_token_repetition_and_truncation_metrics(
    *,
    response_ids: torch.Tensor,
    response_mask: torch.Tensor,
    max_response_length: int,
    finish_reasons: Any = None,
    prefix: str = "rollout",
    tail_token_window: int = 4096,
    threshold: float = 10.0,
) -> tuple[dict[str, float], torch.Tensor]:
    """Compute rollout token-zlib repetition/truncation metrics and return the repeated-tail mask."""
    flags, ratios, eligible, tail_mask = token_zlib_repetition_flags(
        response_ids=response_ids,
        response_mask=response_mask,
        tail_token_window=tail_token_window,
        threshold=threshold,
    )
    metrics: dict[str, float] = {}
    if flags.size:
        metrics[f"{prefix}/repetition_rate_token_zlib"] = float(flags.mean())
        metrics[f"{prefix}/repetition_eligible_rate_token_zlib"] = float(eligible.mean())
        metrics.update(_distribution_metrics_from_values(f"{prefix}/repetition_comp_ratio_token_zlib", ratios))
    response_lengths = response_mask.detach().bool().sum(dim=1).cpu().numpy()
    truncated = compute_truncation_flags(response_lengths, int(max_response_length), finish_reasons)
    if truncated.size:
        metrics[f"{prefix}/truncation_rate"] = float(truncated.mean())
        if flags.size == truncated.size:
            metrics[f"{prefix}/truncation_repetition_joint_rate"] = float((truncated & flags).mean())
            truncated_count = int(truncated.sum())
            repetition_count = int(flags.sum())
            if truncated_count:
                metrics[f"{prefix}/repetition_given_truncation"] = float((truncated & flags).sum()) / truncated_count
            if repetition_count:
                metrics[f"{prefix}/truncation_given_repetition"] = float((truncated & flags).sum()) / repetition_count
    return metrics, tail_mask


def _response_length_tail_metrics(response_length: torch.Tensor, max_response_length: int) -> dict[str, Any]:
    response_length = response_length.detach().float()
    clipped = torch.eq(response_length, float(max_response_length))
    nonclipped = ~clipped
    total_tokens = torch.sum(response_length).detach().item()
    clipped_tokens = torch.sum(response_length[clipped]).detach().item() if clipped.any() else 0.0
    metrics = _length_distribution_metrics("response_length", response_length)
    metrics["response_length/max_response_length"] = float(max_response_length)
    metrics["response_length/clipped_count"] = float(clipped.sum().detach().item())
    metrics["response_length/nonclipped_count"] = float(nonclipped.sum().detach().item())
    metrics["response_length/clipped_token_share"] = clipped_tokens / max(float(total_tokens), 1.0)
    metrics["response_length/right_tail_clip_count"] = metrics["response_length/clipped_count"]
    metrics["response_length/right_tail_clip_ratio"] = torch.mean(clipped.float()).detach().item()
    if nonclipped.any():
        nonclipped_lengths = response_length[nonclipped]
        metrics["response_length/nonclipped_mean"] = torch.mean(nonclipped_lengths).detach().item()
        metrics["response_length/main_mass_mean"] = metrics["response_length/nonclipped_mean"]
        metrics["response_length/nonclipped_p90"] = _safe_tensor_quantile(nonclipped_lengths, 0.90)
        metrics["response_length/nonclipped_p99"] = _safe_tensor_quantile(nonclipped_lengths, 0.99)
    if clipped.any():
        metrics["response_length/clipped_mean"] = torch.mean(response_length[clipped]).detach().item()

    threshold_labels = {
        1024: "1k",
        2048: "2k",
        4096: "4k",
        8192: "8k",
        12000: "12k",
        16000: "16k",
        20000: "20k",
        23000: "23k",
    }
    for threshold, label in threshold_labels.items():
        above = response_length > float(threshold)
        metrics[f"response_length/survival_gt_{label}"] = torch.mean(above.float()).detach().item()
        denom = int(above.sum().detach().item())
        if denom > 0:
            metrics[f"response_length/clip_given_gt_{label}"] = float((clipped & above).sum().detach().item()) / denom
    return metrics


def compute_data_metrics(batch: DataProto, use_critic: bool = True) -> dict[str, Any]:
    """
    Computes various metrics from a batch of data for PPO training.

    This function calculates metrics related to scores, rewards, advantages, returns, values,
    and sequence lengths from a batch of data. It provides statistical information (mean, max, min)
    for each metric category.

    Args:
        batch: A DataProto object containing batch data with token-level scores, rewards, advantages, etc.
        use_critic: Whether to include critic-specific metrics. Defaults to True.

    Returns:
        A dictionary of metrics including:
            - critic/score/mean, max, min: Statistics about sequence scores
            - critic/rewards/mean, max, min: Statistics about sequence rewards
            - critic/advantages/mean, max, min: Statistics about advantages
            - critic/returns/mean, max, min: Statistics about returns
            - critic/values/mean, max, min: Statistics about critic values (if use_critic=True)
            - critic/vf_explained_var: Explained variance of the value function (if use_critic=True)
            - response_length/mean, max, min, clip_ratio: Statistics about response lengths
            - prompt_length/mean, max, min, clip_ratio: Statistics about prompt lengths
            - num_turns/mean, max, min: Statistics about the number of multi-turn conversations
    """
    token_level_scores = batch.batch["token_level_scores"]
    token_level_rewards = batch.batch["token_level_rewards"]
    if token_level_scores.dim() == 3:
        token_level_scores = token_level_scores.sum(dim=-1)
    if token_level_rewards.dim() == 3:
        token_level_rewards = token_level_rewards.sum(dim=-1)
    sequence_score = token_level_scores.sum(-1)
    sequence_reward = token_level_rewards.sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_prompt_length = batch.batch["prompts"].shape[-1]
    max_response_length = batch.batch["responses"].shape[-1]

    response_mask = batch.batch["response_mask"].bool()

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]

    aborted_mask = (response_length == 0).bool()
    non_aborted_mask = ~aborted_mask

    non_aborted_sequence_score = sequence_score[non_aborted_mask]
    non_aborted_sequence_reward = sequence_reward[non_aborted_mask]

    if non_aborted_sequence_score.numel() > 0:
        score_mean = torch.mean(non_aborted_sequence_score).detach().item()
        score_max = torch.max(non_aborted_sequence_score).detach().item()
        score_min = torch.min(non_aborted_sequence_score).detach().item()
    else:
        logger.warning("All samples are aborted, returning default score metrics")
        score_mean = score_max = score_min = float("nan")

    if non_aborted_sequence_reward.numel() > 0:
        reward_mean = torch.mean(non_aborted_sequence_reward).detach().item()
        reward_max = torch.max(non_aborted_sequence_reward).detach().item()
        reward_min = torch.min(non_aborted_sequence_reward).detach().item()
    else:
        logger.warning("All samples are aborted, returning default reward metrics")
        reward_mean = reward_max = reward_min = float("nan")

    advantage_mask = response_mask.unsqueeze(-1).expand_as(advantages) if advantages.dim() == 3 else response_mask
    returns_mask = response_mask.unsqueeze(-1).expand_as(returns) if returns.dim() == 3 else response_mask
    valid_adv = torch.masked_select(advantages, advantage_mask)
    valid_returns = torch.masked_select(returns, returns_mask)

    if valid_adv.numel() > 0:
        adv_mean = torch.mean(valid_adv).detach().item()
        adv_max = torch.max(valid_adv).detach().item()
        adv_min = torch.min(valid_adv).detach().item()
    else:
        logger.warning("Response mask is all False, returning default advantage metrics")
        adv_mean = adv_max = adv_min = float("nan")

    if valid_returns.numel() > 0:
        returns_mean = torch.mean(valid_returns).detach().item()
        returns_max = torch.max(valid_returns).detach().item()
        returns_min = torch.min(valid_returns).detach().item()
    else:
        logger.warning("Response mask is all False, returning default return metrics")
        returns_mean = returns_max = returns_min = float("nan")

    # Aborted samples and non-aborted response length statistics
    # response_length_non_aborted/*: statistics computed on non-aborted samples only
    aborted_ratio = torch.mean(aborted_mask.float()).detach().item()

    non_aborted_response_length = response_length[non_aborted_mask]
    if non_aborted_response_length.numel() > 0:
        non_aborted_response_length_mean = torch.mean(non_aborted_response_length).detach().item()
        non_aborted_response_length_max = torch.max(non_aborted_response_length).detach().item()
        non_aborted_response_length_min = torch.min(non_aborted_response_length).detach().item()
        non_aborted_response_length_clip_ratio = (
            torch.mean(torch.eq(non_aborted_response_length, max_response_length).float()).detach().item()
        )
    else:
        logger.warning("All samples are aborted, returning default response length metrics")
        non_aborted_response_length_mean = float("nan")
        non_aborted_response_length_max = float("nan")
        non_aborted_response_length_min = float("nan")
        non_aborted_response_length_clip_ratio = float("nan")

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        returns_for_value = returns.sum(dim=-1) if returns.dim() == response_mask.dim() + 1 else returns
        valid_returns_for_value = torch.masked_select(returns_for_value, response_mask)
        if valid_returns_for_value.numel() > 0 and valid_values.numel() > 0:
            return_diff_var = torch.var(valid_returns_for_value - valid_values)
            return_var = torch.var(valid_returns_for_value)
            critic_value_metrics = {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
        else:
            logger.warning("Response mask is all False, returning default value metrics")
            critic_value_metrics = {
                "critic/values/mean": float("nan"),
                "critic/values/max": float("nan"),
                "critic/values/min": float("nan"),
                # vf explained var
                "critic/vf_explained_var": float("nan"),
            }
    else:
        critic_value_metrics = {}

    metrics = {
        # score
        "critic/score/mean": score_mean,
        "critic/score/max": score_max,
        "critic/score/min": score_min,
        # reward
        "critic/rewards/mean": reward_mean,
        "critic/rewards/max": reward_max,
        "critic/rewards/min": reward_min,
        # adv
        "critic/advantages/mean": adv_mean,
        "critic/advantages/max": adv_max,
        "critic/advantages/min": adv_min,
        # returns
        "critic/returns/mean": returns_mean,
        "critic/returns/max": returns_max,
        "critic/returns/min": returns_min,
        **critic_value_metrics,
        # response length
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float())
        .detach()
        .item(),
        # response length (non-aborted only)
        # These statistics exclude aborted samples to avoid skew from zeros
        "response_length_non_aborted/mean": non_aborted_response_length_mean,
        "response_length_non_aborted/max": non_aborted_response_length_max,
        "response_length_non_aborted/min": non_aborted_response_length_min,
        "response_length_non_aborted/clip_ratio": non_aborted_response_length_clip_ratio,
        # aborted ratio
        # Fraction of samples whose response length is zero
        "response/aborted_ratio": aborted_ratio,
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    metrics.update(_response_length_tail_metrics(response_length, max_response_length))
    metrics.update(_length_distribution_metrics("prompt_length", prompt_length))
    metrics["prompt_length/max_prompt_length"] = float(max_prompt_length)
    finish_reasons = batch.non_tensor_batch.get("finish_reason") if batch.non_tensor_batch is not None else None
    metrics.update(compute_finish_reason_metrics(finish_reasons, prefix="rollout/finish_reason"))
    truncated = compute_truncation_flags(response_length, max_response_length, finish_reasons)
    if truncated.size:
        metrics["rollout/truncation_rate"] = float(truncated.mean())

    # multi-turn conversation
    if "__num_turns__" in batch.non_tensor_batch:
        num_turns = batch.non_tensor_batch["__num_turns__"]
        metrics["num_turns/min"] = num_turns.min()
        metrics["num_turns/max"] = num_turns.max()
        metrics["num_turns/mean"] = num_turns.mean()

    if "tool_call_counts" in batch.non_tensor_batch:
        tool_call_counts = batch.non_tensor_batch["tool_call_counts"]
        metrics["tool_call_counts/min"] = tool_call_counts.min()
        metrics["tool_call_counts/max"] = tool_call_counts.max()
        metrics["tool_call_counts/mean"] = tool_call_counts.mean()

    tool_metric_suffixes = (
        "attempt_rate",
        "parse_success_rate",
        "parse_failure_rate",
        "parse_recovery_rate",
        "execution_success_rate",
        "protocol_error_rate",
        "timeout_rate",
        "runtime_error_rate",
        "http_error_rate",
        "syntax_error_rate",
        "markdown_extraction_rate",
        "student_call_rate",
        "teacher_call_rate",
        "student_calls",
        "teacher_calls",
        "calls_per_trajectory",
        "observation_tokens",
        "final_answer_after_tool",
        "ended_after_observation",
        "protocol_error_limit_reached",
    )
    tool_metric_keys = tuple(
        f"tool_call/{phase_prefix}{suffix}"
        for phase_prefix in ("", "raw/", "final/")
        for suffix in tool_metric_suffixes
    ) + (
        "tool_call/dmpd_repair_count",
        "tool_call/dmpd_safe_boundary_expansion_count",
        "tool_call/dmpd_repair_rejection_count",
    )
    for metric_key in tool_metric_keys:
        if metric_key not in batch.non_tensor_batch:
            continue
        values = np.asarray(
            [value for value in batch.non_tensor_batch[metric_key] if value is not None],
            dtype=np.float64,
        )
        if values.size:
            metrics[metric_key] = float(values.mean())

    rejection_counts = batch.non_tensor_batch.get("tool_call/dmpd_repair_rejection_counts")
    if rejection_counts is not None:
        totals: dict[str, int] = {}
        for sample_counts in rejection_counts:
            for reason, count in (sample_counts or {}).items():
                totals[str(reason)] = totals.get(str(reason), 0) + int(count)
        denominator = max(len(rejection_counts), 1)
        for reason, total in totals.items():
            metrics[f"tool_call/dmpd_rejection/{reason}"] = float(total / denominator)

    return metrics


def compute_timing_metrics(batch: DataProto, timing_raw: dict[str, float]) -> dict[str, Any]:
    """
    Computes timing metrics for different processing stages in PPO training.

    This function calculates both raw timing metrics (in seconds) and per-token timing metrics
    (in milliseconds) for various processing stages like generation, reference computation,
    value computation, advantage computation, and model updates.

    Args:
        batch: A DataProto object containing batch data with responses and attention masks.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.

    Returns:
        A dictionary containing:
            - timing_s/{name}: Raw timing in seconds for each stage
            - timing_per_token_ms/{name}: Per-token timing in milliseconds for each stage

    Note:
        Different stages use different token counts for normalization:
        - "gen" uses only response tokens
        - Other stages ("ref", "values", "adv", "update_critic", "update_actor") use all tokens
          (prompt + response)
    """
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info["prompt_length"]).item()
    num_response_tokens = torch.sum(response_info["response_length"]).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        "gen": num_response_tokens,
        **{name: num_overall_tokens for name in ["ref", "values", "adv", "update_critic", "update_actor"]},
    }

    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{
            f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name]
            for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
        },
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: dict[str, float], n_gpus: int) -> dict[str, Any]:
    """
    Computes throughput metrics for PPO training.

    This function calculates performance metrics related to token processing speed,
    including the total number of tokens processed, time per step, and throughput
    (tokens per second per GPU).

    Args:
        batch: A DataProto object containing batch data with meta information about token counts.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.
                   Must contain a "step" key with the total step time.
        n_gpus: Number of GPUs used for training.

    Returns:
        A dictionary containing:
            - perf/total_num_tokens: Total number of tokens processed in the batch
            - perf/time_per_step: Time taken for the step in seconds
            - perf/throughput: Tokens processed per second per GPU

    Note:
        The throughput is calculated as total_tokens / (time * n_gpus) to normalize
        across different GPU counts.
    """
    total_num_tokens = sum(batch.meta_info["global_token_num"])
    time = timing_raw["step"]
    # estimated_flops, promised_flops = flops_function.estimate_flops(num_tokens, time)
    # f'Actual TFLOPs/s/GPU​': estimated_flops/(n_gpus),
    # f'Theoretical TFLOPs/s/GPU​': promised_flops,
    return {
        "perf/total_num_tokens": total_num_tokens,
        "perf/time_per_step": time,
        "perf/throughput": total_num_tokens / (time * n_gpus),
    }


def compute_variance_proxy_metrics(batch: DataProto, gradient_norm: float = None) -> dict[str, float]:
    """
    Compute variance proxy metrics using the simplified expected squared norm approach.

    This metric provides a computationally efficient way to monitor gradient variance
    during training. It works for any advantage estimator as long as sum_pi_squared
    is available from the actor.

    Theory:
    - Full variance: Var(g̃) = E[||g̃||²] - ||g_true||²
    - Simplified proxy (when ||g_true||² ≈ 0): Var(g̃) ≈ E[||g̃||²]
    - Using W-score approximation: E[||g̃||²] ≈ E[A² × W(τ)]

    Where W(τ) = Σ_t[1 - 2π_t(y_t) + Σπ²] is the score-norm proxy.
    """
    metrics = {}

    # Check if we have the necessary data (sum_pi_squared is required for W-score)
    if "sum_pi_squared" not in batch.batch or "old_log_probs" not in batch.batch or "advantages" not in batch.batch:
        return metrics

    # Compute W(τ) = Σ_t[1 - 2π_t(y_t) + Σπ²]
    pi_t = torch.exp(batch.batch["old_log_probs"])
    w_per_timestep = 1 - 2 * pi_t + batch.batch["sum_pi_squared"]

    # Get response mask to only consider valid tokens
    response_mask = batch.batch["response_mask"]

    # Use pre-computed rollout IS weights from batch (for variance proxy consistency with training loss)
    # IS weights are computed centrally in ray_trainer.py to avoid duplication
    rollout_is_weights = None
    if "rollout_is_weights" in batch.batch:
        # Extract pre-computed IS weights from batch (already computed in trainer)
        rollout_is_weights = batch.batch["rollout_is_weights"]

        # Scale W by (rollout IS weight)² for optimal baseline under biased estimation
        w_per_timestep = w_per_timestep * (rollout_is_weights**2).detach()

        # Note: IS weight statistics and mismatch metrics are logged in ray_trainer.py

    # Get scalar advantages (mean over timesteps)
    advantages = batch.batch["advantages"]
    if advantages.dim() != response_mask.dim():
        return metrics
    # Compute mean advantage per trajectory using masked_mean
    advantages_scalar = verl_F.masked_mean(advantages, response_mask, axis=-1)

    # Compute W values (sum over timesteps)
    w_values = verl_F.masked_sum(w_per_timestep, response_mask, axis=-1)

    # ====== COMPUTE VARIANCE PROXIES ======
    # Variance proxy should match the actual gradient computation:
    # - If IS weights were computed/applied: use them in variance proxy calculation
    # - Otherwise: compute on-policy variance proxy

    # ====== PROXY 1: Signal Strength ||ḡ||² ======
    # The squared norm of the mean gradient (provided from training loop)
    proxy1_signal_strength = gradient_norm**2 if gradient_norm is not None else None

    # ====== PROXY 2: Total Power E[||ĝ_τ||²] ======
    # Measures the average of squared gradient norms (Signal + Noise)
    if rollout_is_weights is not None:
        # Off-policy with IS correction applied: use clamped weights consistently with actual gradient computation
        rollout_is_weights_scalar = verl_F.masked_mean(rollout_is_weights, response_mask, axis=-1)
        # Recover original W (before IS correction was applied in line 657)
        # Clamp to avoid division by zero when IS weights are zero
        w_original = verl_F.masked_sum(
            w_per_timestep / torch.clamp((rollout_is_weights**2).detach(), min=1e-10), response_mask, axis=-1
        )
        # Clamp W to avoid negative values (which would cause NaN in sqrt)
        w_original = torch.clamp(w_original, min=0.0)
        # Proxy 2 for off-policy: E[ρ̄² × A² × W]
        proxy2_total_power = ((rollout_is_weights_scalar**2) * (advantages_scalar**2) * w_original).mean()

    else:
        # On-policy Proxy 2: E[A² × W]
        # Clamp W to avoid negative values (which would cause NaN in sqrt)
        w_values_clamped = torch.clamp(w_values, min=0.0)
        proxy2_total_power = (advantages_scalar**2 * w_values_clamped).mean()

    # ====== PROXY 3: Pure Noise - Variance of Mean Vector ======
    # Requires ||ḡ||² from actual batch gradient
    # Formula: (1/(N-1)) × (Proxy2 - Proxy1)
    proxy3_pure_noise = None
    if proxy1_signal_strength is not None:
        batch_size = advantages_scalar.shape[0]
        if batch_size > 1:
            proxy3_pure_noise = (1.0 / (batch_size - 1)) * (proxy2_total_power - proxy1_signal_strength)
            # Ensure non-negative (can be negative due to numerical errors)
            proxy3_pure_noise = max(
                0.0, proxy3_pure_noise.item() if torch.is_tensor(proxy3_pure_noise) else proxy3_pure_noise
            )

    # Decompose into components for analysis
    expected_a_squared = (advantages_scalar**2).mean()
    expected_w = w_values.mean()

    metrics.update(
        {
            # Proxy 1: Signal Strength ||ḡ||²
            "variance_proxy/proxy1_signal_strength": (
                proxy1_signal_strength if proxy1_signal_strength is not None else 0.0
            ),
            # Proxy 2: Total Power E[||ĝ_τ||²]
            "variance_proxy/proxy2_total_power": proxy2_total_power.detach().item(),
            # Proxy 3: Pure Noise - Variance of Mean Vector
            "variance_proxy/proxy3_pure_noise": proxy3_pure_noise if proxy3_pure_noise is not None else 0.0,
            # Component metrics for debugging
            "variance_proxy/expected_a_squared": expected_a_squared.detach().item(),
            "variance_proxy/expected_w": expected_w.detach().item(),
        }
    )

    return metrics


def bootstrap_metric(
    data: list[Any],
    subset_size: int,
    reduce_fns: list[Callable[[np.ndarray], float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[tuple[float, float]]:
    """
    Performs bootstrap resampling to estimate statistics of metrics.

    This function uses bootstrap resampling to estimate the mean and standard deviation
    of metrics computed by the provided reduction functions on random subsets of the data.

    Args:
        data: List of data points to bootstrap from.
        subset_size: Size of each bootstrap sample.
        reduce_fns: List of functions that compute a metric from a subset of data.
        n_bootstrap: Number of bootstrap iterations. Defaults to 1000.
        seed: Random seed for reproducibility. Defaults to 42.

    Returns:
        A list of tuples, where each tuple contains (mean, std) for a metric
        corresponding to each reduction function in reduce_fns.

    Example:
        >>> data = [1, 2, 3, 4, 5]
        >>> reduce_fns = [np.mean, np.max]
        >>> bootstrap_metric(data, 3, reduce_fns)
        [(3.0, 0.5), (4.5, 0.3)]  # Example values
    """
    np.random.seed(seed)
    data_np = np.array(data, dtype=object)
    n_data = len(data_np)

    # generate bootstrap indices, shape: (n_bootstrap, subset_size)
    bootstrap_idxs = np.random.choice(n_data, size=(n_bootstrap, subset_size), replace=True)

    # pre-allocate result array, shape: (n_fns, n_bootstrap)
    n_fns = len(reduce_fns)
    metric_results = np.empty((n_fns, n_bootstrap), dtype=np.float64)

    # compute metric results for each bootstrap sample
    for fn_idx, reduce_fn in enumerate(reduce_fns):
        # bootstrap sample and compute metric
        for boot_idx in range(n_bootstrap):
            sample = data_np[bootstrap_idxs[boot_idx]]
            metric_results[fn_idx, boot_idx] = reduce_fn(sample)

    # compute mean and std for each metric function
    result = [
        (float(np.mean(metric_results[fn_idx])), float(np.std(metric_results[fn_idx]))) for fn_idx in range(n_fns)
    ]
    return result


def calc_maj_val(data: list[dict[str, Any]], vote_key: str, val_key: str) -> float:
    """
    Calculate a value based on majority voting.

    This function identifies the most common value for a specified vote key
    in the data, then returns the corresponding value for that majority vote.

    Args:
        data: List of dictionaries, where each dictionary contains both vote_key and val_key.
        vote_key: The key in each dictionary used for voting/counting.
        val_key: The key in each dictionary whose value will be returned for the majority vote.

    Returns:
        The value associated with the most common vote.

    Example:
        >>> data = [
        ...     {"pred": "A", "val": 0.9},
        ...     {"pred": "B", "val": 0.8},
        ...     {"pred": "A", "val": 0.7}
        ... ]
        >>> calc_maj_val(data, vote_key="pred", val_key="val")
        0.9  # Returns the first "val" for the majority vote "A"
    """
    vote2vals = defaultdict(list)
    for d in data:
        vote2vals[d[vote_key]].append(d[val_key])

    vote2cnt = {k: len(v) for k, v in vote2vals.items()}
    maj_vote = max(vote2cnt, key=vote2cnt.get)

    maj_val = vote2vals[maj_vote][0]

    return maj_val


def process_validation_metrics(
    data_sources: list[str], sample_uids: list[str], infos_dict: dict[str, list[Any]], seed: int = 42
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Process validation metrics into a structured format with statistical analysis.

    This function organizes validation metrics by data source and prompt, then computes
    various statistical measures including means, standard deviations, best/worst values,
    and majority voting results. It also performs bootstrap sampling to estimate statistics
    for different sample sizes.

    Args:
        data_sources: List of data source identifiers for each sample.
        sample_uids: List of sample uids corresponding to each sample.
        infos_dict: Dictionary mapping variable names to lists of values for each sample.
        seed: Random seed for bootstrap sampling. Defaults to 42.

    Returns:
        A nested dictionary with the structure:
        {
            data_source: {
                variable_name: {
                    metric_name: value
                }
            }
        }

        Where metric_name includes:
        - "mean@N": Mean value across N samples
        - "std@N": Standard deviation across N samples
        - "best@N/mean": Mean of the best values in bootstrap samples of size N
        - "best@N/std": Standard deviation of the best values in bootstrap samples
        - "worst@N/mean": Mean of the worst values in bootstrap samples
        - "worst@N/std": Standard deviation of the worst values in bootstrap samples
        - "maj@N/mean": Mean of majority voting results in bootstrap samples (if "pred" exists)
        - "maj@N/std": Standard deviation of majority voting results (if "pred" exists)

    Example:
        >>> data_sources = ["source1", "source1", "source2"]
        >>> sample_uids = ["uid1", "uid1", "uid2"]
        >>> infos_dict = {"score": [0.8, 0.9, 0.7], "pred": ["A", "A", "B"]}
        >>> result = process_validation_metrics(data_sources, sample_uids, infos_dict)
        >>> # result will contain statistics for each data source and variable
    """
    # Group metrics by data source, prompt and variable
    data_src2uid2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample_idx, data_source in enumerate(data_sources):
        uid = sample_uids[sample_idx]
        var2vals = data_src2uid2var2vals[data_source][uid]
        for var_name, var_vals in infos_dict.items():
            var2vals[var_name].append(var_vals[sample_idx])

    np_mean = np.mean
    np_std = np.std
    reduce_fns_best_worst = [np.max, np.min]
    n_bootstrap = 1000

    # 2. cache ns list
    def gen_ns(n_resps: int) -> list[int]:
        if n_resps <= 1:
            return []
        ns = []
        n = 2
        while n < n_resps:
            ns.append(n)
            n *= 2
        ns.append(n_resps)
        return ns

    ns_cache = {}

    # 3. cache metric results
    data_src2uid2var2metric = {}

    # 4. flatten loop
    for data_source, uid2var2vals in data_src2uid2var2vals.items():
        # create uid dict
        uid_dict = data_src2uid2var2metric.setdefault(data_source, {})

        for uid, var2vals in uid2var2vals.items():
            pred_vals = var2vals.get("pred")
            has_pred = pred_vals is not None
            var_dict = uid_dict.setdefault(uid, {})

            for var_name, var_vals in var2vals.items():
                # skip empty or string values
                if not var_vals or isinstance(var_vals[0], str):
                    continue

                # compute mean and std
                n_resps = len(var_vals)
                metric = {f"mean@{n_resps}": float(np_mean(var_vals))}

                if n_resps > 1:
                    metric[f"std@{n_resps}"] = float(np_std(var_vals))

                    # cache ns list
                    if n_resps not in ns_cache:
                        ns_cache[n_resps] = gen_ns(n_resps)
                    ns = ns_cache[n_resps]

                    # compute best/worst metrics
                    for n in ns:
                        # compute best/worst metrics
                        (bon_mean, bon_std), (won_mean, won_std) = bootstrap_metric(
                            data=var_vals,
                            subset_size=n,
                            reduce_fns=reduce_fns_best_worst,
                            n_bootstrap=n_bootstrap,
                            seed=seed,
                        )
                        metric[f"best@{n}/mean"] = bon_mean
                        metric[f"best@{n}/std"] = bon_std
                        metric[f"worst@{n}/mean"] = won_mean
                        metric[f"worst@{n}/std"] = won_std

                        # compute maj metrics
                        if has_pred:
                            # create vote_data
                            vote_data = [
                                {"val": val, "pred": pred} for val, pred in zip(var_vals, pred_vals, strict=True)
                            ]
                            # compute maj metrics
                            [(maj_n_mean, maj_n_std)] = bootstrap_metric(
                                data=vote_data,
                                subset_size=n,
                                reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
                                n_bootstrap=n_bootstrap,
                                seed=seed,
                            )
                            metric[f"maj@{n}/mean"] = maj_n_mean
                            metric[f"maj@{n}/std"] = maj_n_std

                var_dict[var_name] = metric

    # Aggregate metrics across uids
    data_src2var2metric2uid_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, uid2var2metric in data_src2uid2var2metric.items():
        for uid, var2metric in uid2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2uid_vals[data_source][var_name][metric_name].append(metric_val)

    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data_source, var2metric2uid_vals in data_src2var2metric2uid_vals.items():
        for var_name, metric2uid_vals in var2metric2uid_vals.items():
            for metric_name, uid_vals in metric2uid_vals.items():
                data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(uid_vals)
    return data_src2var2metric2val
