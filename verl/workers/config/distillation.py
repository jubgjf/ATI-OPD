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

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional

from verl.base_config import BaseConfig
from verl.utils.config import omega_conf_to_dataclass

from .rollout import RolloutConfig

__all__ = [
    "RETHINKING_TARGET_PROMPT_LOGPROBS_CONFIG_KEY",
    "DistillationLossHorizonConfig",
    "DistillationLossConfig",
    "DistillationInterventionConfig",
    "DistillationDiagnosticsConfig",
    "DistillationTeacherModelConfig",
    "DistillationConfig",
]

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

RETHINKING_TARGET_PROMPT_LOGPROBS_CONFIG_KEY = "_verl_enable_target_prompt_logprobs"


@dataclass
class DistillationLossHorizonConfig(BaseConfig):
    """Configuration for limiting OPD supervision to a response prefix."""

    mode: str = "full"
    tokens: int = 4096
    start_tokens: int = 2048
    end_tokens: int = 4096
    increment_tokens: int = 512
    interval_steps: int = 50

    def __post_init__(self):
        mode = self.mode.lower()
        if mode not in {"full", "fixed", "progressive"}:
            raise ValueError(
                f"distillation.distillation_loss.horizon.mode must be one of "
                f"'full', 'fixed', or 'progressive', got {self.mode!r}."
            )
        if self.tokens <= 0:
            raise ValueError("distillation.distillation_loss.horizon.tokens must be positive.")
        if self.start_tokens <= 0:
            raise ValueError("distillation.distillation_loss.horizon.start_tokens must be positive.")
        if self.end_tokens <= 0:
            raise ValueError("distillation.distillation_loss.horizon.end_tokens must be positive.")
        if self.increment_tokens <= 0:
            raise ValueError("distillation.distillation_loss.horizon.increment_tokens must be positive.")
        if self.interval_steps <= 0:
            raise ValueError("distillation.distillation_loss.horizon.interval_steps must be positive.")
        if self.end_tokens < self.start_tokens:
            raise ValueError(
                "distillation.distillation_loss.horizon.end_tokens must be greater than or equal to start_tokens."
            )


@dataclass
class DistillationLossConfig(BaseConfig):
    """Configuration for distillation loss settings.

    loss_mode (str):
        Distillation loss function to use.
    topk (int, optional):
        Number of top tokens to consider for top-k distillation losses.
    teacher_topk (int):
        Number of teacher candidates returned for ``only_stu_teacher_topk``.
        This support width is independent of the student ``topk`` update width.
    top_k_strategy (str):
        Top-k support strategy for token-reward OPD. ``only_stu`` preserves the
        repository's historical teacher/student Top-K intersection
        approximation. ``only_stu_teacher_topk`` expands only the teacher
        support used by that approximation. ``only_stu_rethinking_style``
        scores every student Top-K id under the teacher's full distribution,
        matching THUNLP/OPD.
    reward_weight_mode (str):
        Probability weighting used when reducing top-k token rewards to a scalar reward.
    use_task_rewards (bool):
        Whether to include task rewards alongside distillation loss.
    distillation_loss_coef (float):
        Coefficient for distillation loss when combined with task rewards.
    rlvr_loss_coef (float):
        Coefficient for the independent RLVR policy-loss branch when
        ``ADV_ESTIMATOR=token_reward_direct``.
    rlvr_adv_estimator (str):
        Advantage estimator for the independent RLVR branch. The initial
        implementation supports GRPO only.
    rlvr_credit_mode (str):
        Credit-assignment policy for teacher-repaired trajectories. The
        initial ``no_intervention_only`` mode trains RLVR only on trajectories
        whose final tokens are unchanged by teacher intervention.
    loss_max_clamp (float, optional):
        Maximum value to clamp distillation loss. If None, no clamping is applied.
    log_prob_min_clamp (float, optional):
        Minimum value to clamp log probabilities for stability, e.g., log q - log p where p or q are
        very close to zero. If None, no clamping is applied.
    use_policy_gradient (bool):
        Whether to incorporate distillation loss as a reward, as done
        by https://thinkingmachines.ai/blog/on-policy-distillation/. Recommended to use loss_mode=k1.
        Otherwise, distillation loss is directly backpropagated as a supervised loss,
        as in https://arxiv.org/abs/2306.13649. Recommended to use loss_mode=k3 or forward_kl_topk.
    policy_loss_mode (str):
        Name of the policy loss to use when use_policy_gradient is true.
    clip_ratio (float):
        PPO clipping ratio for policy loss.
    clip_ratio_low (float):
        Lower bound for PPO clipping ratio.
    clip_ratio_high (float):
        Upper bound for PPO clipping ratio.
    horizon (DistillationLossHorizonConfig):
        Optional TOPD/POPD-style prefix horizon for distillation supervision only.
    loss_settings (DistillationLossSettings, optional):
        Runtime-populated settings based on loss_mode. Not set by user.
    """

    loss_mode: str = "k3"
    topk: Optional[int] = 128
    teacher_topk: int = 128
    top_k_strategy: str = "only_stu"
    reward_weight_mode: str = "student_p"
    use_task_rewards: bool = True
    distillation_loss_coef: float = 1.0
    rlvr_loss_coef: float = 1.0
    rlvr_adv_estimator: str = "grpo"
    rlvr_credit_mode: str = "no_intervention_only"
    loss_max_clamp: Optional[float] = 10.0
    log_prob_min_clamp: Optional[float] = -10.0

    use_policy_gradient: bool = True
    policy_loss_mode: str = "vanilla"
    clip_ratio: float = 0.2
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2
    horizon: DistillationLossHorizonConfig = field(default_factory=DistillationLossHorizonConfig)

    # Store global batch info for loss aggregation:
    # dp_size: data parallel size
    # batch_num_tokens: number of valid tokens in global batch
    # global_batch_size: global batch size
    global_batch_info: dict = field(default_factory=dict)

    # Store distillation loss settings for computing the specified loss_mode
    # Not set by user, populated at runtime
    loss_settings: Optional[dict] = None

    def __post_init__(self):
        self._mutable_fields.add("loss_settings")
        from verl.trainer.distillation.losses import DistillationLossSettings, get_distillation_loss_settings

        self.loss_settings: DistillationLossSettings = get_distillation_loss_settings(self.loss_mode)
        if self.topk is not None and self.topk <= 0:
            raise ValueError("distillation_loss.topk must be positive when set.")
        if self.teacher_topk <= 0:
            raise ValueError("distillation_loss.teacher_topk must be positive.")
        supported_top_k_strategies = {"only_stu", "only_stu_rethinking_style", "only_stu_teacher_topk"}
        if self.top_k_strategy not in supported_top_k_strategies:
            raise ValueError(
                f"distillation_loss.top_k_strategy must be one of {sorted(supported_top_k_strategies)}, "
                f"got {self.top_k_strategy!r}."
            )
        if self.top_k_strategy == "only_stu_teacher_topk":
            if self.topk is None:
                raise ValueError("distillation_loss.topk must be set for only_stu_teacher_topk.")
            if self.teacher_topk < self.topk:
                raise ValueError(
                    "distillation_loss.teacher_topk must be greater than or equal to topk for "
                    f"only_stu_teacher_topk, got teacher_topk={self.teacher_topk} and topk={self.topk}."
                )
        if self.reward_weight_mode not in {"student_p", "teacher_p", "none"}:
            raise ValueError(
                "distillation_loss.reward_weight_mode must be one of 'student_p', 'teacher_p', or 'none', "
                f"got {self.reward_weight_mode!r}."
            )
        if not math.isfinite(self.rlvr_loss_coef) or self.rlvr_loss_coef < 0:
            raise ValueError(
                f"distillation_loss.rlvr_loss_coef must be finite and non-negative, got {self.rlvr_loss_coef!r}."
            )
        if self.rlvr_adv_estimator != "grpo":
            raise NotImplementedError(
                f"distillation_loss.rlvr_adv_estimator currently supports only 'grpo', got {self.rlvr_adv_estimator!r}."
            )
        if self.rlvr_credit_mode != "no_intervention_only":
            raise NotImplementedError(
                "distillation_loss.rlvr_credit_mode currently supports only 'no_intervention_only', "
                f"got {self.rlvr_credit_mode!r}."
            )

        if self.policy_loss_mode != "vanilla":
            raise NotImplementedError(
                f"Only vanilla policy loss is currently supported when use_policy_gradient is True, "
                f"but got {self.policy_loss_mode}."
            )

        if self.use_policy_gradient and self.loss_mode == "forward_kl_topk":
            print(
                "WARNING: forward_kl_topk is most effective as a supervised distillation loss "
                "(use_policy_gradient=False). With policy gradient, the update uses only the sampled"
                " token's logprob ∇logπ(a), so the top-k distributional signal (how non-sampled logits "
                "should move) is largely unused."
            )

        if not self.use_policy_gradient and self.loss_mode == "k1":
            raise ValueError(
                "Directly backpropagating k1 loss is incorrect since gradient of k1 loss"
                " wrt model weights does not depend on teacher log probabilities."
            )


@dataclass
class DistillationInterventionConfig(BaseConfig):
    """Configuration for teacher intervention during student rollout.

    The default implementation is compute-friendly: it lets the student finish
    a full rollout first, scores fixed token segments, and only repairs samples
    whose teacher/student gap crosses the threshold.
    """

    enabled: bool = False
    segment_method: str = "fixed_tokens"
    student_segment_length: int = 128
    teacher_segment_length: int = 128
    gap_metric: str = "k3"
    gap_distribution_topk: int = 16
    teacher_rejection_token_cap: float = 10.0
    gap_aggregation: str = "mean"
    gap_threshold: float = 1.0
    teacher_rejection_persistence_enabled: bool = False
    teacher_rejection_persistence_window_tokens: int = 64
    teacher_rejection_persistence_threshold: Optional[float] = None
    teacher_rejection_persistence_min_valid_tokens: int = 32
    max_interventions_per_sample: int = 4
    # Optional second-stage confirmation. ``future_gap`` first generates and
    # scores only ``future_gap_window_tokens`` after the teacher segment; a
    # passing candidate is then continued from that exact token-ID prefix.
    # ``none`` preserves the legacy full-suffix path.
    skip_correct: bool = False
    confirmation_mode: str = "none"
    future_gap_window_tokens: int = 64
    future_gap_min_valid_tokens: int = 32
    # null reuses gap_threshold so the confirmation stays on the active gap's
    # native scale (for example K3 versus teacher rejection).
    future_gap_max_threshold: Optional[float] = None
    future_gap_min_reduction: float = 0.0
    train_on_teacher_segments: bool = False
    teacher_segment_loss_mode: str = "opd"
    teacher_segment_sft_loss_coef: float = 1.0
    min_trainable_tokens_after_repair: int = 1
    protect_thinking_blocks: bool = True
    thinking_start_text: str = "<think>"
    thinking_end_text: str = "</think>"
    require_closed_thinking_on_terminal: bool = True
    teacher_temperature: float = 0.6
    teacher_top_p: float = 0.95
    teacher_top_k: int = 20
    # Opt-in outcome-aware candidate selection. Keeping this disabled preserves
    # the legacy one-teacher-segment intervention path exactly.
    outcome_aware_candidate_selection_enabled: bool = False
    outcome_correctness_key: str = "acc"
    outcome_correctness_threshold: float = 0.5
    teacher_candidate_count: int = 4
    # Bound candidate generation/scoring RPCs shared by all concurrent samples
    # inside one AgentLoopWorker. This prevents best-of-K selection from
    # multiplying rollout concurrency into an inference-server request storm.
    teacher_candidate_max_concurrency_per_worker: int = 4
    # Healthy best-of-K teacher NLLs are aggregated globally across rollout
    # workers at batch boundaries, then one rolling-quantile threshold is
    # broadcast for the next batch.
    teacher_nll_calibration_quantile: float = 0.75
    teacher_nll_calibration_window: int = 256
    teacher_nll_calibration_min_samples: int = 16
    teacher_nll_initial_threshold: Optional[float] = None
    teacher_nll_healthy_probe_probability: float = 0.1
    log_examples_per_worker: int = 2

    def __post_init__(self):
        if self.segment_method != "fixed_tokens":
            raise NotImplementedError(
                f"Only fixed_tokens teacher intervention segments are currently supported, "
                f"but got {self.segment_method!r}."
            )
        if self.student_segment_length <= 0:
            raise ValueError("student_segment_length must be positive.")
        if self.teacher_segment_length <= 0:
            raise ValueError("teacher_segment_length must be positive.")
        if self.max_interventions_per_sample < 0:
            raise ValueError("max_interventions_per_sample must be non-negative.")
        if self.confirmation_mode not in {"none", "future_gap"}:
            raise ValueError(
                f"confirmation_mode must be one of {{'none', 'future_gap'}}, got {self.confirmation_mode!r}."
            )
        if self.skip_correct and not self.enabled:
            raise ValueError("skip_correct=true requires distillation.intervention.enabled=true.")
        if self.confirmation_mode == "future_gap" and self.max_interventions_per_sample != 1:
            raise ValueError(
                "confirmation_mode='future_gap' currently requires max_interventions_per_sample=1 so each "
                "sample has one attributable repair proposal."
            )
        if self.future_gap_window_tokens <= 0:
            raise ValueError("future_gap_window_tokens must be positive.")
        if self.future_gap_min_valid_tokens <= 0:
            raise ValueError("future_gap_min_valid_tokens must be positive.")
        if self.future_gap_min_valid_tokens > self.future_gap_window_tokens:
            raise ValueError("future_gap_min_valid_tokens must not exceed future_gap_window_tokens.")
        if self.future_gap_max_threshold is not None and not math.isfinite(self.future_gap_max_threshold):
            raise ValueError("future_gap_max_threshold must be finite when set.")
        if not math.isfinite(self.future_gap_min_reduction) or self.future_gap_min_reduction < 0:
            raise ValueError("future_gap_min_reduction must be finite and non-negative.")
        if self.teacher_segment_loss_mode not in {"opd", "sft"}:
            raise ValueError(
                f"teacher_segment_loss_mode must be one of {{'opd', 'sft'}}, got {self.teacher_segment_loss_mode!r}."
            )
        if not math.isfinite(self.teacher_segment_sft_loss_coef) or self.teacher_segment_sft_loss_coef < 0:
            raise ValueError(
                "teacher_segment_sft_loss_coef must be finite and non-negative, "
                f"got {self.teacher_segment_sft_loss_coef!r}."
            )
        if self.min_trainable_tokens_after_repair < 0:
            raise ValueError("min_trainable_tokens_after_repair must be non-negative.")
        if self.gap_aggregation not in {"mean", "max"}:
            raise ValueError(f"Unsupported gap_aggregation: {self.gap_aggregation!r}.")
        gap_metric = str(self.gap_metric).lower()
        supported_gap_metrics = {
            "kl",
            "k1",
            "abs",
            "mse",
            "k2",
            "low_var_kl",
            "k3",
            "teacher_rejection",
            "topk_kl",
            "topk_distribution_kl",
            "topk_dist_kl",
            "full_kl",
            "full_distribution_kl",
            "distribution_kl",
        }
        if gap_metric not in supported_gap_metrics:
            raise ValueError(f"Unsupported gap_metric: {self.gap_metric!r}.")
        if self.gap_distribution_topk <= 0:
            raise ValueError("gap_distribution_topk must be positive.")
        if not math.isfinite(self.teacher_rejection_token_cap) or self.teacher_rejection_token_cap <= 0:
            raise ValueError(
                f"teacher_rejection_token_cap must be finite and positive, got {self.teacher_rejection_token_cap!r}."
            )
        if not math.isfinite(self.gap_threshold):
            raise ValueError(f"gap_threshold must be finite, got {self.gap_threshold!r}.")
        if self.teacher_rejection_persistence_window_tokens <= 0:
            raise ValueError("teacher_rejection_persistence_window_tokens must be positive.")
        if self.teacher_rejection_persistence_min_valid_tokens <= 0:
            raise ValueError("teacher_rejection_persistence_min_valid_tokens must be positive.")
        if self.teacher_rejection_persistence_min_valid_tokens > self.teacher_rejection_persistence_window_tokens:
            raise ValueError(
                "teacher_rejection_persistence_min_valid_tokens must not exceed "
                "teacher_rejection_persistence_window_tokens."
            )
        if self.teacher_rejection_persistence_threshold is not None and (
            not math.isfinite(self.teacher_rejection_persistence_threshold)
            or self.teacher_rejection_persistence_threshold < 0
        ):
            raise ValueError("teacher_rejection_persistence_threshold must be finite and non-negative when set.")
        if self.teacher_rejection_persistence_enabled and gap_metric != "teacher_rejection":
            raise ValueError(
                "teacher_rejection_persistence_enabled=true requires gap_metric='teacher_rejection', "
                f"got {self.gap_metric!r}."
            )
        if not math.isfinite(self.teacher_temperature) or self.teacher_temperature < 0:
            raise ValueError(f"teacher_temperature must be finite and non-negative, got {self.teacher_temperature!r}.")
        if not math.isfinite(self.teacher_top_p) or not 0 < self.teacher_top_p <= 1:
            raise ValueError(f"teacher_top_p must be in (0, 1], got {self.teacher_top_p!r}.")
        if self.teacher_top_k < -1:
            raise ValueError(f"teacher_top_k must be -1 or non-negative, got {self.teacher_top_k!r}.")
        if self.outcome_aware_candidate_selection_enabled and not self.enabled:
            raise ValueError(
                "outcome_aware_candidate_selection_enabled=true requires distillation.intervention.enabled=true."
            )
        if not str(self.outcome_correctness_key).strip():
            raise ValueError("outcome_correctness_key must be a non-empty string.")
        if not math.isfinite(self.outcome_correctness_threshold):
            raise ValueError(
                f"outcome_correctness_threshold must be finite, got {self.outcome_correctness_threshold!r}."
            )
        if self.teacher_candidate_count <= 0:
            raise ValueError("teacher_candidate_count must be positive.")
        if self.teacher_candidate_max_concurrency_per_worker <= 0:
            raise ValueError("teacher_candidate_max_concurrency_per_worker must be positive.")
        if (
            not math.isfinite(self.teacher_nll_calibration_quantile)
            or not 0 < self.teacher_nll_calibration_quantile < 1
        ):
            raise ValueError(
                "teacher_nll_calibration_quantile must be finite and in (0, 1), "
                f"got {self.teacher_nll_calibration_quantile!r}."
            )
        if self.teacher_nll_calibration_window <= 0:
            raise ValueError("teacher_nll_calibration_window must be positive.")
        if self.teacher_nll_calibration_min_samples <= 0:
            raise ValueError("teacher_nll_calibration_min_samples must be positive.")
        if self.teacher_nll_calibration_min_samples > self.teacher_nll_calibration_window:
            raise ValueError("teacher_nll_calibration_min_samples must not exceed teacher_nll_calibration_window.")
        if self.teacher_nll_initial_threshold is not None and (
            not math.isfinite(self.teacher_nll_initial_threshold) or self.teacher_nll_initial_threshold < 0
        ):
            raise ValueError("teacher_nll_initial_threshold must be finite and non-negative when set.")
        if (
            not math.isfinite(self.teacher_nll_healthy_probe_probability)
            or not 0 <= self.teacher_nll_healthy_probe_probability <= 1
        ):
            raise ValueError(
                "teacher_nll_healthy_probe_probability must be finite and in [0, 1], "
                f"got {self.teacher_nll_healthy_probe_probability!r}."
            )
        if self.log_examples_per_worker < 0:
            raise ValueError("log_examples_per_worker must be non-negative.")


@dataclass
class DistillationDiagnosticsConfig(BaseConfig):
    """Runtime diagnostics for analyzing OPD behavior on long responses."""

    enabled: bool = False
    position_bin_size: int = 512
    log_interval: int = 1
    topk_log_interval: int = 10
    async_metrics: bool = False
    # Compatibility-only under structured per-step async diagnostics. Cross-step queues are unsupported.
    async_max_pending: int = 1
    heatmap_interval: int = 10
    heatmap_history_steps: int = 80
    log_wandb_heatmap: bool = True
    high_gap_threshold: float = 1.0
    collect_intervention_metrics: bool = True
    collect_intervention_boundary_metrics: bool = False
    intervention_boundary_text_samples_per_step: int = 8
    intervention_boundary_text_context_tokens: int = 32
    collect_intervention_outcome_transitions: bool = False
    outcome_case_samples_per_transition: int = 8
    collect_topk_overlap: bool = False
    collect_rethinking_opd_metrics: bool = False
    collect_repetition_metrics: bool = True
    collect_train_text_zlib_repetition: bool = True
    topk: Optional[int] = None
    repetition_tail_chars: int = 10000
    repetition_comp_ratio_threshold: float = 10.0
    repetition_tail_token_window: int = 4096
    bad_tail_min_position: int = 12000
    bad_tail_deep_position: int = 16000
    bad_tail_extreme_position: int = 20000
    bad_tail_overlap_threshold: float = 0.5
    bad_tail_teacher_prob_threshold: float = 0.2
    bad_tail_gap_threshold: float = 1.0
    bad_tail_reward_threshold: float = -2.0
    bad_tail_entropy_threshold: float = 4.0

    def __post_init__(self):
        if self.position_bin_size <= 0:
            raise ValueError("diagnostics.position_bin_size must be positive.")
        if self.log_interval <= 0:
            raise ValueError("diagnostics.log_interval must be positive.")
        if self.topk_log_interval <= 0:
            raise ValueError("diagnostics.topk_log_interval must be positive.")
        if self.async_max_pending != 1:
            raise ValueError(
                "diagnostics.async_max_pending must be 1 because async diagnostics are joined before "
                "the source training step is committed."
            )
        if self.heatmap_interval <= 0:
            raise ValueError("diagnostics.heatmap_interval must be positive.")
        if self.heatmap_history_steps <= 0:
            raise ValueError("diagnostics.heatmap_history_steps must be positive.")
        if not math.isfinite(self.high_gap_threshold) or self.high_gap_threshold < 0:
            raise ValueError(
                f"diagnostics.high_gap_threshold must be finite and non-negative, got {self.high_gap_threshold!r}."
            )
        if self.outcome_case_samples_per_transition < 0:
            raise ValueError("diagnostics.outcome_case_samples_per_transition must be non-negative.")
        if self.intervention_boundary_text_samples_per_step < 0:
            raise ValueError("diagnostics.intervention_boundary_text_samples_per_step must be non-negative.")
        if self.intervention_boundary_text_context_tokens <= 0:
            raise ValueError("diagnostics.intervention_boundary_text_context_tokens must be positive.")
        if self.topk is not None and self.topk <= 0:
            raise ValueError("diagnostics.topk must be positive.")
        if self.repetition_tail_chars <= 0:
            raise ValueError("diagnostics.repetition_tail_chars must be positive.")
        if self.repetition_tail_token_window <= 0:
            raise ValueError("diagnostics.repetition_tail_token_window must be positive.")
        if not math.isfinite(self.repetition_comp_ratio_threshold):
            raise ValueError(
                "diagnostics.repetition_comp_ratio_threshold must be finite, "
                f"got {self.repetition_comp_ratio_threshold!r}."
            )
        if self.bad_tail_min_position < 0:
            raise ValueError("diagnostics.bad_tail_min_position must be non-negative.")
        if self.bad_tail_deep_position < self.bad_tail_min_position:
            raise ValueError("diagnostics.bad_tail_deep_position must be >= bad_tail_min_position.")
        if self.bad_tail_extreme_position < self.bad_tail_deep_position:
            raise ValueError("diagnostics.bad_tail_extreme_position must be >= bad_tail_deep_position.")
        for field_name in (
            "bad_tail_overlap_threshold",
            "bad_tail_teacher_prob_threshold",
            "bad_tail_gap_threshold",
            "bad_tail_reward_threshold",
            "bad_tail_entropy_threshold",
        ):
            value = getattr(self, field_name)
            if not math.isfinite(value):
                raise ValueError(f"diagnostics.{field_name} must be finite, got {value!r}.")
        if not 0 <= self.bad_tail_overlap_threshold <= 1:
            raise ValueError("diagnostics.bad_tail_overlap_threshold must be in [0, 1].")
        if not 0 <= self.bad_tail_teacher_prob_threshold <= 1:
            raise ValueError("diagnostics.bad_tail_teacher_prob_threshold must be in [0, 1].")
        if self.bad_tail_gap_threshold < 0:
            raise ValueError("diagnostics.bad_tail_gap_threshold must be non-negative.")
        if self.bad_tail_entropy_threshold < 0:
            raise ValueError("diagnostics.bad_tail_entropy_threshold must be non-negative.")


@dataclass
class DistillationTeacherModelConfig(BaseConfig):
    """Configuration for on-policy distillation teacher.

    key (str, optional):
        Identifier to route examples to the teacher model in multi-teacher setting.
    model_path (str, optional):
        Model path for the teacher model. Can be a local path or a Hugging Face model
    inference (RolloutConfig):
        Rollout configuration for the teacher model inference during distillation.
    num_replicas (int):
        Number of inference replicas of this teacher to launch. Each replica occupies
        `per_replica_world_size` GPUs (= inference.data_parallel_size *
        inference.tensor_model_parallel_size * inference.pipeline_model_parallel_size),
        so the teacher's total GPU footprint is
        `num_replicas * per_replica_world_size`.
    """

    _mutable_fields = BaseConfig._mutable_fields | {"num_replicas", "key"}

    key: Optional[str] = None
    model_path: Optional[str] = None
    inference: RolloutConfig = field(default_factory=RolloutConfig)
    num_replicas: Optional[int] = 0

    @property
    def per_replica_world_size(self) -> int:
        return (
            self.inference.tensor_model_parallel_size
            * self.inference.data_parallel_size
            * self.inference.pipeline_model_parallel_size
        )

    @property
    def world_size(self) -> int:
        return self.num_replicas * self.per_replica_world_size

    def check_configured(self):
        if self.model_path is None:
            raise ValueError("model_path must be specified for distillation teacher model config.")
        if self.key is None:
            raise ValueError("key must be specified for distillation teacher model config.")
        if self.num_replicas is None:
            raise ValueError("num_replicas must be specified for distillation teacher model config.")

    def validate_and_prepare_for_distillation(self, use_topk: bool, topk: Optional[int]) -> None:
        # Prompt + Response from student are fed into teacher as context
        max_model_len = self.inference.max_model_len
        student_prompt_length = self.inference.prompt_length
        student_response_length = self.inference.response_length
        required_context_len = student_prompt_length + student_response_length + 1
        if max_model_len is not None and required_context_len > max_model_len:
            raise ValueError(
                "Distillation teacher inference requires room for the student prompt, the full student "
                f"response, and one generated token, but got {student_prompt_length=}, "
                f"{student_response_length=}, {required_context_len=}, {max_model_len=}."
            )
        self.inference.prompt_length = self.inference.prompt_length + self.inference.response_length
        self.inference.response_length = 1
        self._validate_topk_logprobs(use_topk=use_topk, topk=topk)

    def _validate_topk_logprobs(self, use_topk: bool, topk: Optional[int]) -> None:
        if not use_topk:
            return
        if topk is None:
            raise ValueError("topk must be specified when use_topk is True.")

        engine_name = self.inference.name
        engine_kwargs = self.inference.engine_kwargs
        match engine_name:
            case "vllm":
                vllm_engine_kwargs = dict(engine_kwargs.get("vllm", {}))
                max_logprobs = vllm_engine_kwargs.get("max_logprobs")
                if max_logprobs is None:
                    vllm_engine_kwargs["max_logprobs"] = topk
                    max_logprobs = topk
                if max_logprobs < topk:
                    raise ValueError(
                        f"VLLM max_logprobs ({max_logprobs}) must be >= distillation_loss topk "
                        f"({topk}) to enable distillation loss computation."
                    )
                engine_kwargs["vllm"] = vllm_engine_kwargs
            case "sglang":
                # SGLang's top_logprobs_num is a per-request parameter, so there is no
                # engine-boot cap to align (unlike vLLM's max_logprobs). The async
                # server translates sampling_params["prompt_logprobs"] into
                # return_logprob + logprob_start_len=0 + top_logprobs_num at call time.
                pass
            case _:
                raise NotImplementedError(
                    f"DistillationTeacherModelConfig does not support inference engine {engine_name}"
                )


@dataclass
class DistillationConfig(BaseConfig):
    """Configuration for on-policy distillation.

    enabled (bool):
        Whether on-policy distillation is enabled.
    n_gpus_per_node (int):
        Number of GPUs per node in the teacher resource pool.
    nnodes (int):
        Number of nodes in the teacher resource pool.
    teacher_models (dict[str, TeacherModelConfig]):
        Configurations for teacher models used for multi-teacher distillation.
    teacher_key (str):
        Key to route examples to the appropriate teacher model in multi-teacher setups. Should correspond to a field in
        the data proto, e.g., data_source.
    distillation_loss (DistillationLossConfig):
    Configuration for distillation loss settings.

    NOTE: The `teacher_model` entry is in the `teacher_models` dict by default.
    Since it is popped when other teacher entries are added, using `teacher_model` as
    one of several keys silently drops it. For example, the following CLI overrides result
    in ONLY `teacher_model2` being used:

    ```bash
    distillation.teacher_models.teacher_model.key=openai/gsm8k
    distillation.teacher_models.teacher_model.model_path=Qwen/Qwen3-4B
    +distillation.teacher_models.teacher_model2.key=hiyouga/geometry3k
    +distillation.teacher_models.teacher_model2.model_path=Qwen/Qwen3-VL-4B-Instruct
    ```
    Instead, give the first teacher a different name:

    ```bash
    +distillation.teacher_models.teacher_model1.key=openai/gsm8k
    +distillation.teacher_models.teacher_model1.model_path=Qwen/Qwen3-4B
    +distillation.teacher_models.teacher_model2.key=hiyouga/geometry3k
    +distillation.teacher_models.teacher_model2.model_path=Qwen/Qwen3-VL-4B-Instruct
    ```
    """

    _mutable_fields = BaseConfig._mutable_fields | {"teacher_models", "n_gpus_per_node", "nnodes"}

    enabled: bool = False
    use_actor_loss: bool = True
    n_gpus_per_node: int = 0
    nnodes: int = 0
    teacher_models: dict[str, DistillationTeacherModelConfig] = field(default_factory=dict)
    teacher_key: str = "data_source"
    distillation_loss: DistillationLossConfig = field(default_factory=DistillationLossConfig)
    intervention: DistillationInterventionConfig = field(default_factory=DistillationInterventionConfig)
    diagnostics: DistillationDiagnosticsConfig = field(default_factory=DistillationDiagnosticsConfig)

    def __post_init__(self):
        if not self.enabled:
            return

        if self.diagnostics.topk is None:
            object.__setattr__(self.diagnostics, "topk", int(self.distillation_loss.topk or 64))
        self.teacher_models = self._resolve_teacher_models()
        teacher_world_size_sum = 0
        logprob_topk = self.distillation_loss.topk if self.distillation_loss.loss_settings.use_topk else None
        if not self.use_actor_loss:
            logprob_topk = max(int(logprob_topk or 0), int(self.distillation_loss.topk or 0))
        if self.distillation_loss.top_k_strategy == "only_stu_teacher_topk":
            logprob_topk = max(int(logprob_topk or 0), int(self.distillation_loss.teacher_topk))
        if self.diagnostics.collect_topk_overlap or self.diagnostics.collect_rethinking_opd_metrics:
            logprob_topk = max(int(logprob_topk or 0), int(self.diagnostics.topk))
        for teacher_model in self.teacher_models.values():
            teacher_model.validate_and_prepare_for_distillation(
                use_topk=logprob_topk is not None and logprob_topk > 0,
                topk=logprob_topk,
            )
            teacher_world_size_sum += teacher_model.world_size
        total_pool_size = self.n_gpus_per_node * self.nnodes
        if teacher_world_size_sum != total_pool_size:
            raise ValueError(
                f"Sum of teacher (num_replicas * per_replica_world_size) ({teacher_world_size_sum}) must match "
                f"the distillation resource pool size "
                f"({self.n_gpus_per_node=} * {self.nnodes=} = {total_pool_size})."
            )

    def _resolve_teacher_models(self) -> dict[str, DistillationTeacherModelConfig]:
        assert "teacher_model" in self.teacher_models
        if len(self.teacher_models) == 1:
            # Single teacher occupies the entire teacher resource pool.
            teacher_model = self.teacher_models["teacher_model"]
            inference = teacher_model.inference
            per_replica = (
                inference.tensor_model_parallel_size
                * inference.data_parallel_size
                * inference.pipeline_model_parallel_size
            )
            pool_size = self.n_gpus_per_node * self.nnodes
            if pool_size % per_replica != 0:
                raise ValueError(
                    f"Single teacher's per_replica_world_size ({per_replica}) must divide the distillation "
                    f"resource pool size ({self.n_gpus_per_node=} * {self.nnodes=} = {pool_size})."
                )
            teacher_model.num_replicas = pool_size // per_replica
            teacher_model.key = "default"
        else:
            # Multiple teachers: remove default single teacher config
            self.teacher_models.pop("teacher_model")

        # Teacher models dict is keyed by teacher_key instead of YAML entry name
        teacher_models = {}
        for teacher_config in self.teacher_models.values():
            teacher_config = omega_conf_to_dataclass(teacher_config, dataclass_type=DistillationTeacherModelConfig)
            teacher_config.check_configured()
            if teacher_config.key in teacher_models:
                raise ValueError(f"Duplicate teacher key {teacher_config.key} found in teacher models.")
            teacher_models[teacher_config.key] = teacher_config
        return teacher_models
