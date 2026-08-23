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
    tool_agent_enabled = bool(
        _get_nested(config, "actor_rollout_ref", "rollout", "multi_turn", "enable", default=False)
        and _get_nested(
            config,
            "actor_rollout_ref",
            "rollout",
            "agent",
            "default_agent_loop",
            default="single_turn_agent",
        )
        == "tool_agent"
    )
    if intervention_enabled and tool_agent_enabled:
        confirmation_mode = str(_get_nested(distillation_config, "intervention", "confirmation_mode", default="none"))
        if confirmation_mode == "future_gap":
            raise ValueError(
                "Tool-aware DMPD does not yet support "
                "distillation.intervention.confirmation_mode='future_gap'; use confirmation_mode='none'."
            )
        if bool(
            _get_nested(
                distillation_config,
                "intervention",
                "outcome_aware_candidate_selection_enabled",
                default=False,
            )
        ):
            raise ValueError(
                "Tool-aware DMPD does not yet support "
                "distillation.intervention.outcome_aware_candidate_selection_enabled=true."
            )
        if repair_for_evaluation:
            raise ValueError(
                "Tool-aware DMPD does not yet support "
                "distillation.intervention.repair_for_evaluation=true; keep repair evaluation disabled."
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

    adv_estimator = _get_nested(config, "algorithm", "adv_estimator", default="")
    adv_estimator = getattr(adv_estimator, "value", adv_estimator)
    use_task_rewards = bool(_get_nested(distillation_config, "distillation_loss", "use_task_rewards", default=True))
    use_actor_loss = bool(_get_nested(distillation_config, "use_actor_loss", default=True))
    joint_token_reward_rlvr = str(adv_estimator) == "token_reward_direct" and use_task_rewards
    if joint_token_reward_rlvr:
        if use_actor_loss:
            raise ValueError(
                "Joint token-reward OPD + RLVR requires distillation.use_actor_loss=false. "
                "OPD is optimized through [B,T,K] token rewards while RLVR is an independent [B,T] branch."
            )
        if bypass_mode:
            raise ValueError(
                "Joint token-reward OPD + RLVR requires algorithm.rollout_correction.bypass_mode=false so both "
                "sampled-token and Student-TopK proximal logprobs can be recomputed."
            )
        if bool(_get_nested(config, "algorithm", "use_kl_in_reward", default=False)):
            raise ValueError(
                "Joint token-reward OPD + RLVR does not support algorithm.use_kl_in_reward=true. "
                "Keep the OPD and task-reward branches independent."
            )
        if bool(_get_nested(config, "actor_rollout_ref", "actor", "use_kl_loss", default=False)):
            raise ValueError(
                "Joint token-reward OPD + RLVR initially requires actor_rollout_ref.actor.use_kl_loss=false."
            )
        entropy_coeff = float(_get_nested(config, "actor_rollout_ref", "actor", "entropy_coeff", default=0.0))
        if entropy_coeff != 0.0:
            raise ValueError(
                "Joint token-reward OPD + RLVR requires actor_rollout_ref.actor.entropy_coeff=0 so the implemented "
                "objective is exactly loss_opd + rlvr_loss_coef * loss_rlvr."
            )
        policy_loss_mode = _get_nested(
            config, "actor_rollout_ref", "actor", "policy_loss", "loss_mode", default="vanilla"
        )
        if str(policy_loss_mode) != "vanilla":
            raise ValueError(
                "Joint token-reward OPD + RLVR initially supports only "
                "actor_rollout_ref.actor.policy_loss.loss_mode='vanilla'."
            )
        loss_agg_mode = _get_nested(config, "actor_rollout_ref", "actor", "loss_agg_mode", default="token-mean")
        if str(loss_agg_mode) != "token-mean":
            raise ValueError(
                "Joint token-reward OPD + RLVR initially supports only "
                "actor_rollout_ref.actor.loss_agg_mode='token-mean'."
            )
        rlvr_adv_estimator = _get_nested(distillation_config, "distillation_loss", "rlvr_adv_estimator", default="grpo")
        if str(rlvr_adv_estimator) != "grpo":
            raise ValueError("Joint token-reward OPD + RLVR initially supports only rlvr_adv_estimator='grpo'.")
        rlvr_credit_mode = _get_nested(
            distillation_config,
            "distillation_loss",
            "rlvr_credit_mode",
            default="no_intervention_only",
        )
        if str(rlvr_credit_mode) != "no_intervention_only":
            raise ValueError(
                "Joint token-reward OPD + RLVR initially supports only rlvr_credit_mode='no_intervention_only'."
            )

    top_k_strategy = _get_nested(
        distillation_config,
        "distillation_loss",
        "top_k_strategy",
        default="only_stu",
    )
    if top_k_strategy in {"only_stu_rethinking_style", "only_stu_teacher_topk"}:
        if str(adv_estimator) != "token_reward_direct":
            raise ValueError(
                f"distillation_loss.top_k_strategy={top_k_strategy!r} requires "
                "algorithm.adv_estimator='token_reward_direct'. This student-support strategy is consumed by "
                "the paper-style token-reward update, not the legacy actor-side OPD loss."
            )
        if use_actor_loss:
            raise ValueError(
                f"distillation_loss.top_k_strategy={top_k_strategy!r} requires "
                "distillation.use_actor_loss=false so the student-support teacher signal is not mixed with the "
                "legacy actor-side distillation path."
            )
        if bypass_mode:
            raise ValueError(
                f"distillation_loss.top_k_strategy={top_k_strategy!r} requires "
                "algorithm.rollout_correction.bypass_mode=false because student Top-K ids must be recomputed "
                "before token rewards are constructed."
            )
