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
    "DistillationLossHorizonConfig",
    "DistillationLossConfig",
    "DistillationInterventionConfig",
    "DistillationDiagnosticsConfig",
    "DistillationTeacherModelConfig",
    "DistillationConfig",
]

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


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
    use_task_rewards (bool):
        Whether to include task rewards alongside distillation loss.
    distillation_loss_coef (float):
        Coefficient for distillation loss when combined with task rewards.
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
    use_task_rewards: bool = True
    distillation_loss_coef: float = 1.0
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
    gap_aggregation: str = "mean"
    gap_threshold: float = 1.0
    max_interventions_per_sample: int = 4
    train_on_teacher_segments: bool = False
    min_trainable_tokens_after_repair: int = 1
    protect_thinking_blocks: bool = True
    thinking_start_text: str = "<think>"
    thinking_end_text: str = "</think>"
    require_closed_thinking_on_terminal: bool = True
    teacher_temperature: float = 0.7
    teacher_top_p: float = 1.0
    teacher_top_k: int = -1
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
        if self.min_trainable_tokens_after_repair < 0:
            raise ValueError("min_trainable_tokens_after_repair must be non-negative.")
        if self.gap_aggregation not in {"mean", "max"}:
            raise ValueError(f"Unsupported gap_aggregation: {self.gap_aggregation!r}.")
        if self.gap_metric not in {"kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"}:
            raise ValueError(f"Unsupported gap_metric: {self.gap_metric!r}.")
        if not math.isfinite(self.gap_threshold):
            raise ValueError(f"gap_threshold must be finite, got {self.gap_threshold!r}.")
        if not math.isfinite(self.teacher_temperature) or self.teacher_temperature < 0:
            raise ValueError(f"teacher_temperature must be finite and non-negative, got {self.teacher_temperature!r}.")
        if not math.isfinite(self.teacher_top_p) or not 0 < self.teacher_top_p <= 1:
            raise ValueError(f"teacher_top_p must be in (0, 1], got {self.teacher_top_p!r}.")
        if self.teacher_top_k < -1:
            raise ValueError(f"teacher_top_k must be -1 or non-negative, got {self.teacher_top_k!r}.")
        if self.log_examples_per_worker < 0:
            raise ValueError("log_examples_per_worker must be non-negative.")


@dataclass
class DistillationDiagnosticsConfig(BaseConfig):
    """Runtime diagnostics for analyzing OPD behavior on long responses."""

    enabled: bool = False
    position_bin_size: int = 512
    log_interval: int = 1
    heatmap_interval: int = 10
    heatmap_history_steps: int = 80
    log_wandb_heatmap: bool = True
    high_gap_threshold: float = 1.0
    collect_intervention_metrics: bool = True
    collect_topk_overlap: bool = False
    topk: int = 16

    def __post_init__(self):
        if self.position_bin_size <= 0:
            raise ValueError("diagnostics.position_bin_size must be positive.")
        if self.log_interval <= 0:
            raise ValueError("diagnostics.log_interval must be positive.")
        if self.heatmap_interval <= 0:
            raise ValueError("diagnostics.heatmap_interval must be positive.")
        if self.heatmap_history_steps <= 0:
            raise ValueError("diagnostics.heatmap_history_steps must be positive.")
        if not math.isfinite(self.high_gap_threshold) or self.high_gap_threshold < 0:
            raise ValueError(
                f"diagnostics.high_gap_threshold must be finite and non-negative, got {self.high_gap_threshold!r}."
            )
        if self.topk <= 0:
            raise ValueError("diagnostics.topk must be positive.")
        if self.collect_topk_overlap:
            raise NotImplementedError(
                "Top-k overlap diagnostics are not implemented for the sampled-token OPD path yet. "
                "Set distillation.diagnostics.collect_topk_overlap=false."
            )


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

        self.teacher_models = self._resolve_teacher_models()
        teacher_world_size_sum = 0
        for teacher_model in self.teacher_models.values():
            teacher_model.validate_and_prepare_for_distillation(
                use_topk=self.distillation_loss.loss_settings.use_topk,
                topk=self.distillation_loss.topk,
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
