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

from typing import Any

from verl.trainer.distillation.losses import is_distillation_enabled


def _get_nested(config: Any, *keys: str, default: Any = None) -> Any:
    current = config
    for key in keys:
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            current = getattr(current, key, default)
    return current


def validate_distillation_intervention_training_config(config: Any) -> None:
    """Validate cross-feature combinations that cannot be represented in one dataclass."""
    distillation_config = _get_nested(config, "distillation")
    if not is_distillation_enabled(distillation_config):
        return
    intervention_enabled = bool(_get_nested(distillation_config, "intervention", "enabled", default=False))
    repair_for_evaluation = bool(
        _get_nested(distillation_config, "intervention", "repair_for_evaluation", default=False)
    )
    diagnostics_enabled = bool(_get_nested(distillation_config, "diagnostics", "enabled", default=False))
    collect_intervention_metrics = bool(
        _get_nested(distillation_config, "diagnostics", "collect_intervention_metrics", default=True)
    )
    collect_outcome_transitions = bool(
        _get_nested(
            distillation_config,
            "diagnostics",
            "collect_intervention_outcome_transitions",
            default=False,
        )
    )
    outcome_case_samples_per_transition = int(
        _get_nested(
            distillation_config,
            "diagnostics",
            "outcome_case_samples_per_transition",
            default=8,
        )
    )
    if outcome_case_samples_per_transition < 0:
        raise ValueError("distillation.diagnostics.outcome_case_samples_per_transition must be non-negative.")
    if repair_for_evaluation and not intervention_enabled:
        raise ValueError(
            "distillation.intervention.repair_for_evaluation=true requires distillation.intervention.enabled=true."
        )
    if collect_outcome_transitions and not intervention_enabled:
        raise ValueError(
            "distillation.diagnostics.collect_intervention_outcome_transitions=true requires "
            "distillation.intervention.enabled=true."
        )
    if collect_outcome_transitions and not diagnostics_enabled:
        raise ValueError(
            "distillation.diagnostics.collect_intervention_outcome_transitions=true requires "
            "distillation.diagnostics.enabled=true."
        )
    if collect_outcome_transitions and not collect_intervention_metrics:
        raise ValueError(
            "distillation.diagnostics.collect_intervention_outcome_transitions=true requires "
            "distillation.diagnostics.collect_intervention_metrics=true."
        )
    train_on_teacher_segments = bool(
        _get_nested(distillation_config, "intervention", "train_on_teacher_segments", default=False)
    )
    bypass_mode = bool(_get_nested(config, "algorithm", "rollout_correction", "bypass_mode", default=False))
    if intervention_enabled and train_on_teacher_segments and bypass_mode:
        raise ValueError(
            "distillation.intervention.train_on_teacher_segments=true is incompatible with "
            "algorithm.rollout_correction.bypass_mode=true. Teacher intervention drops rollout_log_probs so the "
            "actor can recompute logprobs on repaired teacher tokens, while bypass mode requires rollout_log_probs."
        )

    top_k_strategy = _get_nested(
        distillation_config,
        "distillation_loss",
        "top_k_strategy",
        default="only_stu",
    )
    if top_k_strategy == "only_stu_rethinking_style":
        adv_estimator = _get_nested(config, "algorithm", "adv_estimator", default="")
        adv_estimator = getattr(adv_estimator, "value", adv_estimator)
        if str(adv_estimator) != "token_reward_direct":
            raise ValueError(
                "distillation_loss.top_k_strategy='only_stu_rethinking_style' requires "
                "algorithm.adv_estimator='token_reward_direct'. The exact teacher-on-student Top-K tensor is "
                "consumed by the paper-style token-reward update, not the legacy actor-side OPD loss."
            )
        if bool(_get_nested(distillation_config, "use_actor_loss", default=True)):
            raise ValueError(
                "distillation_loss.top_k_strategy='only_stu_rethinking_style' requires "
                "distillation.use_actor_loss=false so the exact Top-K teacher signal is not mixed with the "
                "legacy actor-side distillation path."
            )
        if bypass_mode:
            raise ValueError(
                "distillation_loss.top_k_strategy='only_stu_rethinking_style' requires "
                "algorithm.rollout_correction.bypass_mode=false because student Top-K ids must be recomputed "
                "before exact teacher target scoring."
            )
