# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
from tensordict import TensorDict

from verl.base_config import BaseConfig
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils import tensordict_utils as tu
from verl.utils.metric import AggregationType, Metric
from verl.workers.config import ActorConfig, DistillationConfig, DistillationLossConfig
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import no_padding_2_padding

DistillationLossFn = Callable[
    [
        ActorConfig,  # actor_config
        DistillationConfig,  # distillation_config
        dict,  # model_output
        TensorDict,  # micro batch input
    ],
    tuple[torch.Tensor, dict[str, Any]],
]


def is_distillation_enabled(config: Optional[DistillationConfig]) -> bool:
    """Check if distillation is enabled based on the provided configuration."""
    if config is None:
        return False
    return config.enabled


def is_teacher_segment_sft_enabled(config: Optional[DistillationConfig]) -> bool:
    """Return whether teacher-inserted rollout tokens use the auxiliary SFT objective."""
    if config is None or not getattr(config, "enabled", True):
        return False
    intervention_config = getattr(config, "intervention", None)
    return bool(
        intervention_config is not None
        and getattr(intervention_config, "enabled", False)
        and getattr(intervention_config, "train_on_teacher_segments", False)
        and getattr(intervention_config, "teacher_segment_loss_mode", "opd") == "sft"
    )


def _zero_from_model_output(model_output: dict[str, Any]) -> torch.Tensor:
    """Build a graph-connected zero scalar from response-token model output."""
    value = model_output.get("response_token_log_probs", model_output["log_probs"])
    if isinstance(value, tu.JaggedTensorParts):
        value = value.values
    elif value.is_nested:
        value = value.values()
    return value.sum() * 0.0


def _ppo_loss_or_zero(
    config: ActorConfig,
    model_output: dict[str, Any],
    data: TensorDict,
    dp_group=None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Avoid token-mean division by zero for a globally teacher-only batch."""
    batch_num_tokens = tu.get_non_tensor_data(data, "batch_num_tokens", None)
    if batch_num_tokens is None or batch_num_tokens > 0:
        return ppo_loss(config, model_output, data, dp_group)

    zero = _zero_from_model_output(model_output)
    return zero, {
        "actor/pg_loss": Metric(value=zero, aggregation=AggregationType.SUM),
        "actor/empty_action_mask": Metric(value=zero.detach() + 1.0, aggregation=AggregationType.MEAN),
    }


def teacher_segment_sft_loss(
    distillation_config: DistillationConfig,
    model_output: dict[str, Any],
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute global teacher-token-mean NLL on inserted teacher segments only."""
    teacher_segment_mask = data.get("teacher_segment_mask", None)
    if teacher_segment_mask is None:
        raise RuntimeError(
            "Teacher-segment SFT is enabled, but the actor batch has no teacher_segment_mask. "
            "The rollout path must emit a response-aligned mask for every sample."
        )
    if teacher_segment_mask.is_nested:
        teacher_segment_mask = teacher_segment_mask.to_padded_tensor(0)
    teacher_segment_mask = teacher_segment_mask.bool()
    response_mask = data["response_mask"]
    if response_mask.is_nested:
        response_mask = response_mask.to_padded_tensor(0)
    response_mask = response_mask.bool()
    if response_mask.shape != teacher_segment_mask.shape:
        raise ValueError(
            "teacher_segment_mask must be response-aligned, "
            f"but got response_mask.shape={tuple(response_mask.shape)} and "
            f"teacher_segment_mask.shape={tuple(teacher_segment_mask.shape)}."
        )
    if (response_mask & teacher_segment_mask).any().item():
        raise ValueError(
            "Teacher-segment SFT requires teacher tokens to be excluded from response_mask so they cannot also "
            "participate in OPD, PPO, task reward, entropy, or KL losses."
        )

    response_token_log_probs = model_output.get("response_token_log_probs", model_output["log_probs"])
    response_token_log_probs = no_padding_2_padding(response_token_log_probs, data)
    if response_token_log_probs.shape != teacher_segment_mask.shape:
        raise ValueError(
            "Teacher-segment SFT requires scalar current-policy log probabilities aligned to the response, "
            f"but got log_probs.shape={tuple(response_token_log_probs.shape)} and "
            f"teacher_segment_mask.shape={tuple(teacher_segment_mask.shape)}."
        )

    batch_num_teacher_tokens = tu.get_non_tensor_data(data, "batch_num_teacher_tokens", None)
    dp_size = tu.get_non_tensor_data(data, "dp_size", 1)
    if batch_num_teacher_tokens is None:
        if dp_size > 1:
            raise RuntimeError("batch_num_teacher_tokens is required for distributed teacher-segment SFT.")
        batch_num_teacher_tokens = int(teacher_segment_mask.sum().item())

    if batch_num_teacher_tokens > 0:
        masked_log_prob_sum = torch.where(
            teacher_segment_mask, response_token_log_probs, torch.zeros_like(response_token_log_probs)
        ).sum()
        loss = -masked_log_prob_sum / batch_num_teacher_tokens * dp_size
    else:
        loss = response_token_log_probs.sum() * 0.0

    intervention_config = distillation_config.intervention
    metrics = {
        "distillation/teacher_segment_sft_loss": Metric(value=loss, aggregation=AggregationType.SUM),
        "distillation/teacher_segment_sft_tokens": Metric(
            value=loss.new_tensor(float(batch_num_teacher_tokens)), aggregation=AggregationType.MEAN
        ),
        "distillation/teacher_segment_sft_loss_coef": Metric(
            value=loss.new_tensor(float(intervention_config.teacher_segment_sft_loss_coef)),
            aggregation=AggregationType.MEAN,
        ),
    }
    return loss, metrics


@dataclass
class DistillationLossSettings(BaseConfig):
    """
    Settings for a distillation loss function to be registered.

    Args:
        names (str | list[str]): Name(s) to register the distillation loss function under.
        use_topk (bool): Whether the loss function uses top-k log probabilities.
        use_estimator (bool): Whether the loss function uses single-sample KL estimators.
    """

    names: str | list[str] = field(default_factory=list)
    use_topk: bool = False
    use_estimator: bool = False

    _mutable_fields = {"names"}

    def __post_init__(self):
        self.names = [self.names] if isinstance(self.names, str) else self.names
        if sum([self.use_topk, self.use_estimator]) != 1:
            raise ValueError(
                f"Expected only one of use_estimator, use_topk, but got {self.use_estimator=}, {self.use_topk=}."
            )


DISTILLATION_LOSS_REGISTRY: dict[str, DistillationLossFn] = {}
DISTILLATION_SETTINGS_REGISTRY: dict[str, DistillationLossSettings] = {}


def register_distillation_loss(
    loss_settings: DistillationLossSettings,
) -> Callable[[DistillationLossFn], DistillationLossFn]:
    """Register a distillation loss function with the given name."""

    def decorator(func: DistillationLossFn) -> DistillationLossFn:
        for name in loss_settings.names:
            if name in DISTILLATION_LOSS_REGISTRY:
                raise ValueError(f"Distillation loss function with name '{name}' is already registered.")
            DISTILLATION_LOSS_REGISTRY[name] = func
            DISTILLATION_SETTINGS_REGISTRY[name] = loss_settings
        return func

    return decorator


def get_distillation_loss_fn(loss_name: str) -> DistillationLossFn:
    """Get the distillation loss function with a given name."""
    if loss_name not in DISTILLATION_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_LOSS_REGISTRY.keys())}"
        )
    return DISTILLATION_LOSS_REGISTRY[loss_name]


def get_distillation_loss_settings(loss_name: str) -> DistillationLossSettings:
    """Get the distillation loss settings with a given name."""
    if loss_name not in DISTILLATION_SETTINGS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_SETTINGS_REGISTRY.keys())}"
        )
    return DISTILLATION_SETTINGS_REGISTRY[loss_name]


def get_distillation_loss_mask(data: TensorDict) -> torch.Tensor:
    """Return the OPD supervision mask, falling back to the full response mask."""
    loss_mask = data.get("distillation_loss_mask", None)
    if loss_mask is None:
        loss_mask = data["response_mask"]
    if loss_mask.is_nested:
        return loss_mask.bool().to_padded_tensor(False)
    return loss_mask.bool()


def _safe_masked_values(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = values[mask]
    if selected.numel() == 0:
        return values.new_zeros(1)
    return selected


def compute_distillation_loss_range(
    distillation_losses: torch.Tensor, distillation_loss_mask: torch.Tensor
) -> dict[str, Metric]:
    """Compute min and max distillation loss over valid response tokens."""
    distillation_losses_response = _safe_masked_values(distillation_losses, distillation_loss_mask)
    return {
        "distillation/loss_min": Metric(AggregationType.MIN, distillation_losses_response.min()),
        "distillation/loss_max": Metric(AggregationType.MAX, distillation_losses_response.max()),
    }


def compute_topk_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    data: TensorDict,
    student_logits: torch.Tensor,
    data_format: str,
) -> torch.Tensor:
    """Compute the topk loss in logit processor.

    Returns:
    - distillation_losses: (bsz, seqlen/cp_size)
    - student_mass: (bsz, seqlen/cp_size)
    - teacher_mass: (bsz, seqlen/cp_size)
    """
    match config.strategy:
        # VeOmni uses FSDP2 internally, so its loss computation is identical to FSDP.
        case "fsdp" | "veomni":
            import verl.trainer.distillation.fsdp.losses as fsdp_losses

            distillation_loss_fn = fsdp_losses.compute_forward_kl_topk
        case "megatron":
            import verl.trainer.distillation.megatron.losses as megatron_losses

            distillation_loss_fn = megatron_losses.compute_forward_kl_topk
        case _:
            raise NotImplementedError(f"Unsupported strategy: {config.strategy=}")

    loss_topk = int(distillation_config.distillation_loss.topk)
    outputs = distillation_loss_fn(
        student_logits=student_logits,
        teacher_topk_log_probs=data["teacher_logprobs"][..., :loss_topk],
        teacher_topk_ids=data["teacher_ids"][..., :loss_topk],
        config=distillation_config,
        data_format=data_format,
    )

    expected_shape = student_logits.shape[:2]
    for k, v in outputs.items():
        assert v.shape == expected_shape, f"Expected shape {expected_shape}, but got {v.shape} for {k=}."

    return outputs


def distillation_ppo_loss(
    config: ActorConfig,
    distillation_config: Optional[DistillationConfig],
    model_output: dict = None,
    data: TensorDict = None,
    dp_group=None,
    student_logits: torch.Tensor = None,
    data_format: str = "thd",
):
    """Loss function used both for logit processor and final policy loss.
    - student_logits is not None, compute the topk loss in logit processor.
    - student_logits is None, compute final policy loss.

    [split sequence across sp/cp groups]
                   |
    [model forward and output logits: (bsz, seqlen/cp_size, vocab_size/tp_size)]
                   |
    [logits processor compute topk loss: (bsz, seqlen/cp_size)]
                   |
    [all gather topk loss across sp/cp groups: (bsz, seqlen)]
                   |
    [combine topk loss with policy loss]

    Args:
        config: Actor configuration.
        distillation_config: Distillation configuration.
        model_output: Model output, including log_probs, entropy.
        data: Micro input batch, contains
          - teacher_logprobs: (bsz, seqlen, topk)
          - teacher_ids: (bsz, seqlen, topk)
        student_logits: (bsz, seqlen/cp_size, vocab_size/tp_size).
        data_format: "thd" or "bshd", models not support THD format, e.g GPT-OSS, Qwen3.5

    Returns:
    - student_logits is not None, return the topk loss tensor (bsz, seqlen/cp_size).
    - student_logits is None, return the final policy loss scalar and metrics.
    """

    # Called as logits processor
    if student_logits is not None:
        return compute_topk_loss(config, distillation_config, data, student_logits, data_format)

    # Called as final policy loss. Actor-side OPD and token-reward OPD share this
    # wrapper only when teacher-segment SFT needs to be composed with the base loss.
    distillation_loss_config = distillation_config.distillation_loss
    use_actor_loss = getattr(distillation_config, "use_actor_loss", True)
    policy_loss, policy_metrics = _ppo_loss_or_zero(config, model_output, data, dp_group)

    if use_actor_loss:
        distill_loss, distill_metrics = distillation_loss(config, distillation_config, model_output, data)
        if not distillation_loss_config.use_task_rewards:
            policy_loss = policy_loss * 0.0

        policy_metrics.update(distill_metrics)
        distillation_loss_coef = (
            distillation_loss_config.distillation_loss_coef if distillation_loss_config.use_task_rewards else 1.0
        )
        policy_loss = policy_loss + distill_loss * distillation_loss_coef
        policy_metrics["distillation/loss"] = Metric(value=distill_loss, aggregation=AggregationType.SUM)

    if is_teacher_segment_sft_enabled(distillation_config):
        sft_loss, sft_metrics = teacher_segment_sft_loss(distillation_config, model_output, data)
        policy_loss = policy_loss + sft_loss * distillation_config.intervention.teacher_segment_sft_loss_coef
        policy_metrics.update(sft_metrics)

    return policy_loss, policy_metrics


def distillation_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the distillation loss and related metrics.

    Returns:
    - distillation_loss: Aggregated distillation loss scalar.
    - distillation_metrics: Dictionary of metrics.
    """
    assert distillation_config is not None
    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    distillation_loss_fn = get_distillation_loss_fn(loss_config.loss_mode)
    distillation_losses, distillation_metrics = distillation_loss_fn(
        config=config,
        distillation_config=distillation_config,
        model_output=model_output,
        data=data,
    )
    distillation_loss_mask = get_distillation_loss_mask(data)
    loss_agg_mode = config.loss_agg_mode

    distillation_metrics.update(
        compute_distillation_loss_range(
            distillation_losses=distillation_losses,
            distillation_loss_mask=distillation_loss_mask,
        )
    )
    mask_empty = not distillation_loss_mask.any().item()
    distillation_metrics["distillation/mask_empty"] = Metric(
        AggregationType.MEAN, distillation_losses.new_tensor(float(mask_empty))
    )
    if mask_empty:
        return distillation_losses.sum() * 0.0, distillation_metrics

    if loss_config.loss_max_clamp is not None:
        # clamping min is for k1 loss which can be negative
        distillation_losses = distillation_losses.clamp(min=-loss_config.loss_max_clamp, max=loss_config.loss_max_clamp)

    if loss_config.use_policy_gradient:
        # Use negative distillation loss as reward, as done by https://thinkingmachines.ai/blog/on-policy-distillation/.
        policy_loss_fn = get_policy_loss_fn(loss_config.policy_loss_mode)
        for k, v in config.global_batch_info.items():
            loss_config.global_batch_info[k] = v
        log_prob = no_padding_2_padding(model_output["log_probs"], data)
        old_log_prob = data["old_log_probs"]
        if old_log_prob.is_nested:
            old_log_prob = data["old_log_probs"].to_padded_tensor(0.0)
        rollout_is_weights = data.get("rollout_is_weights", None)
        distillation_loss, pg_metrics = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=-distillation_losses.detach(),
            response_mask=distillation_loss_mask,
            loss_agg_mode=loss_agg_mode,
            config=loss_config,
            rollout_is_weights=rollout_is_weights,
        )
        pg_metrics = {f"distillation/{k[len('actor/') :]}": v for k, v in pg_metrics.items()}
        distillation_metrics.update(pg_metrics)
    else:
        # Directly backpropagate distillation loss as a supervised loss, as in https://arxiv.org/abs/2306.13649.
        distillation_loss = agg_loss(
            loss_mat=distillation_losses,
            loss_mask=distillation_loss_mask,
            loss_agg_mode=loss_agg_mode,
            **config.global_batch_info,
        )

    return distillation_loss, distillation_metrics


@register_distillation_loss(DistillationLossSettings(names=["forward_kl_topk"], use_topk=True))  # type: ignore[arg-type]
def compute_forward_kl_topk(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute forward KL distillation loss and related metrics using top-k log probabilities.

    Returns:
    - distillation_losses: (bsz, resp_len)
    - distillation_metrics: Dictionary of metrics.
    """
    # topk loss has been computed in logits processor
    distillation_losses = no_padding_2_padding(model_output["distillation_losses"], data)
    student_mass = no_padding_2_padding(model_output["student_mass"], data)
    teacher_mass = no_padding_2_padding(model_output["teacher_mass"], data)
    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    distillation_loss_mask = get_distillation_loss_mask(data)
    assert (
        distillation_losses.shape
        == student_mass.shape
        == teacher_mass.shape
        == response_mask_bool.shape
        == distillation_loss_mask.shape
    )

    # Log amount of mass in the top-k log probabilities for both student and teacher.
    student_mass = _safe_masked_values(student_mass, distillation_loss_mask)
    teacher_mass = _safe_masked_values(teacher_mass, distillation_loss_mask)
    distillation_metrics = {
        "distillation/student_mass": student_mass.mean().item(),
        "distillation/student_mass_min": Metric(AggregationType.MIN, student_mass.min()),
        "distillation/student_mass_max": Metric(AggregationType.MAX, student_mass.max()),
        "distillation/teacher_mass": teacher_mass.mean().item(),
        "distillation/teacher_mass_min": Metric(AggregationType.MIN, teacher_mass.min()),
        "distillation/teacher_mass_max": Metric(AggregationType.MAX, teacher_mass.max()),
    }

    # Due to use of top-k, student and teacher distributions don't sum to 1 -> divergences can be negative.
    distillation_losses = distillation_losses.clamp_min(0.0)

    return distillation_losses, distillation_metrics


@register_distillation_loss(
    DistillationLossSettings(names=["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"], use_estimator=True)
)  # type: ignore[arg-type]
def compute_distillation_loss_reverse_kl_estimator(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the distillation loss and related metrics using single-sample KL estimators.

    Uses the kl_penalty function from core_algos which supports various KL divergence
    estimators: "kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3".

    Returns:
    - distillation_losses: (bsz, resp_len)
    - distillation_metrics: Dictionary of metrics.
    """
    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    teacher_logprob_key = (
        "teacher_sampled_logprobs" if "teacher_sampled_logprobs" in data.keys() else "teacher_logprobs"
    )
    teacher_log_probs = no_padding_2_padding(data[teacher_logprob_key], data).squeeze(-1)
    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    distillation_loss_mask = get_distillation_loss_mask(data)
    assert (
        teacher_log_probs.shape == student_log_probs.shape == response_mask_bool.shape == distillation_loss_mask.shape
    )

    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    distillation_losses = kl_penalty(
        logprob=student_log_probs, ref_logprob=teacher_log_probs, kl_penalty=loss_config.loss_mode
    )
    # Since k1 can be negative, log the mean absolute loss.
    masked_losses = _safe_masked_values(distillation_losses, distillation_loss_mask)
    metrics = {
        "distillation/abs_loss": Metric(AggregationType.MEAN, masked_losses.abs().mean()),
    }
    return distillation_losses, metrics
