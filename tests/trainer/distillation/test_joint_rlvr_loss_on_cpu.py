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

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from verl.trainer.distillation import losses as distillation_losses
from verl.trainer.ppo.ray_trainer import build_joint_rlvr_training_tensors
from verl.utils import tensordict_utils as tu
from verl.utils.metric import Metric
from verl.utils.py_functional import append_to_dict
from verl.workers.config import ActorConfig
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead


def _joint_config(*, coef: float):
    return SimpleNamespace(
        enabled=True,
        use_actor_loss=False,
        distillation_loss=SimpleNamespace(
            use_task_rewards=True,
            rlvr_loss_coef=coef,
        ),
        intervention=SimpleNamespace(
            enabled=False,
            train_on_teacher_segments=False,
            teacher_segment_loss_mode="opd",
        ),
    )


def test_build_joint_rlvr_training_tensors_keeps_plain_task_rewards_on_cpu():
    task_rewards = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
    response_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

    rewards, mask, metrics = build_joint_rlvr_training_tensors(
        task_reward_tensor=task_rewards,
        response_mask=response_mask,
    )

    assert torch.equal(rewards, task_rewards)
    assert torch.equal(mask, response_mask)
    assert metrics["distillation/joint_rlvr/intervention_excluded_fraction"] == 0.0


def test_build_joint_rlvr_training_tensors_uses_raw_scores_and_masks_repairs_on_cpu():
    final_rewards = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    response_mask = torch.tensor(
        [
            [1, 1, 0],
            [1, 0, 1],
            [1, 1, 1],
            [1, 1, 0],
        ]
    )
    outcomes = [
        {"raw_reward_score": 1.0, "repair_executed": False},
        {"raw_reward_score": 0.0, "repair_executed": True},
        {"raw_reward_score": -1.0, "repair_executed": False},
        {"raw_reward_score": 0.5, "repair_executed": True},
    ]

    rewards, mask, metrics = build_joint_rlvr_training_tensors(
        task_reward_tensor=final_rewards,
        response_mask=response_mask,
        intervention_outcomes=outcomes,
    )

    assert torch.allclose(rewards.sum(dim=-1), torch.tensor([1.0, 0.0, -1.0, 0.5]))
    assert torch.equal(mask[0], response_mask[0])
    assert not mask[1].any()
    assert torch.equal(mask[2], response_mask[2])
    assert not mask[3].any()
    assert metrics["distillation/joint_rlvr/active_sample_fraction"] == pytest.approx(0.5)
    assert metrics["distillation/joint_rlvr/intervention_excluded_fraction"] == pytest.approx(0.5)


def test_build_joint_rlvr_training_tensors_rejects_missing_raw_reward_on_cpu():
    with pytest.raises(RuntimeError, match="raw_reward_score"):
        build_joint_rlvr_training_tensors(
            task_reward_tensor=torch.zeros(1, 2),
            response_mask=torch.ones(1, 2),
            intervention_outcomes=[{"repair_executed": True}],
        )


def test_build_joint_rlvr_training_tensors_reads_final_reward_outside_policy_mask_on_cpu():
    task_rewards = torch.tensor([[0.0, 0.0, 2.0]])
    response_mask = torch.tensor([[1, 1, 0]])

    _, _, metrics = build_joint_rlvr_training_tensors(
        task_reward_tensor=task_rewards,
        response_mask=response_mask,
        intervention_outcomes=[{"raw_reward_score": 1.0, "repair_executed": True}],
    )

    assert metrics["distillation/joint_rlvr/final_reward_mean"] == pytest.approx(2.0)
    assert metrics["distillation/joint_rlvr/repair_reward_delta_mean"] == pytest.approx(1.0)


@pytest.mark.parametrize(("coef", "expected"), [(0.0, 2.0), (0.5, 3.5), (2.0, 8.0)])
def test_distillation_ppo_loss_composes_opd_and_rlvr_scalars_on_cpu(monkeypatch, coef, expected):
    opd_loss = torch.tensor(2.0, requires_grad=True)
    rlvr_loss = torch.tensor(3.0, requires_grad=True)
    data = TensorDict({}, batch_size=[])
    tu.assign_non_tensor(data, joint_token_reward_rlvr=True)

    monkeypatch.setattr(
        distillation_losses,
        "_ppo_loss_or_zero",
        lambda config, model_output, data, dp_group: (opd_loss, {}),
    )
    monkeypatch.setattr(
        distillation_losses,
        "_joint_rlvr_policy_loss",
        lambda config, model_output, data: (rlvr_loss, {}),
    )

    loss, metrics = distillation_losses.distillation_ppo_loss(
        config=SimpleNamespace(),
        distillation_config=_joint_config(coef=coef),
        model_output={"log_probs": torch.ones(1)},
        data=data,
    )

    assert loss.item() == pytest.approx(expected)
    assert metrics["distillation/joint_rlvr/loss_coef"].aggregate() == pytest.approx(coef)
    loss.backward()
    assert opd_loss.grad.item() == pytest.approx(1.0)
    assert rlvr_loss.grad.item() == pytest.approx(coef)


def test_joint_rlvr_policy_loss_uses_sampled_token_shape_and_independent_denominator_on_cpu():
    data = TensorDict(
        {
            "prompts": torch.tensor([[10, 11]], dtype=torch.long),
            "responses": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            "attention_mask": torch.ones(1, 6, dtype=torch.long),
            "rlvr_old_log_probs": torch.zeros(1, 4),
            "rlvr_advantages": torch.tensor([[1.0, -1.0, 0.0, 2.0]]),
            "rlvr_response_mask": torch.tensor([[1, 1, 0, 1]], dtype=torch.long),
        },
        batch_size=[1],
    )
    tu.assign_non_tensor(data, batch_num_rlvr_tokens=3)
    config = ActorConfig(strategy="fsdp", rollout_n=1, use_dynamic_bsz=True)
    config.global_batch_info.update(
        dp_size=1,
        batch_num_tokens=99,
        global_batch_size=1,
        loss_scale_factor=None,
    )
    # Response slicing maps full-sequence positions [1:5] to response tokens.
    current_log_probs = torch.zeros(6, requires_grad=True)

    loss, metrics = distillation_losses._joint_rlvr_policy_loss(
        config,
        {"response_token_log_probs": current_log_probs},
        data,
    )

    torch.testing.assert_close(loss, torch.tensor(-2.0 / 3.0))
    assert metrics["distillation/joint_rlvr/empty_mask"].aggregate() == 0.0
    assert config.global_batch_info["batch_num_tokens"] == 99
    loss.backward()
    torch.testing.assert_close(
        current_log_probs.grad,
        torch.tensor([0.0, -1.0 / 3.0, 1.0 / 3.0, 0.0, -2.0 / 3.0, 0.0]),
    )


def test_joint_rlvr_empty_local_mask_keeps_dp_metric_schema_on_cpu():
    def compute_metrics(response_mask: torch.Tensor):
        data = TensorDict(
            {
                "prompts": torch.tensor([[10, 11]], dtype=torch.long),
                "responses": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
                "attention_mask": torch.ones(1, 6, dtype=torch.long),
                "rlvr_old_log_probs": torch.zeros(1, 4),
                "rlvr_advantages": torch.tensor([[1.0, -1.0, 0.0, 2.0]]),
                "rlvr_response_mask": response_mask,
            },
            batch_size=[1],
        )
        # Simulate a DP batch where another rank still has three active RLVR tokens.
        tu.assign_non_tensor(data, batch_num_rlvr_tokens=3)
        config = ActorConfig(strategy="fsdp", rollout_n=1, use_dynamic_bsz=True)
        config.global_batch_info.update(
            dp_size=2,
            batch_num_tokens=99,
            global_batch_size=2,
            loss_scale_factor=None,
        )
        current_log_probs = torch.zeros(6, requires_grad=True)
        loss, metrics = distillation_losses._joint_rlvr_policy_loss(
            config,
            {"response_token_log_probs": current_log_probs},
            data,
        )
        return loss, metrics, current_log_probs

    active_loss, active_metrics, _ = compute_metrics(torch.tensor([[1, 1, 0, 1]], dtype=torch.long))
    empty_loss, empty_metrics, empty_log_probs = compute_metrics(torch.zeros(1, 4, dtype=torch.long))

    assert set(empty_metrics) == set(active_metrics)
    assert empty_loss.item() == 0.0
    assert empty_metrics["distillation/joint_rlvr/empty_mask"].aggregate() == 1.0
    assert empty_metrics["distillation/joint_rlvr/pg_clipfrac"].aggregate() == 0.0
    assert empty_metrics["distillation/joint_rlvr/ppo_kl"].aggregate() == 0.0
    assert empty_metrics["distillation/joint_rlvr/pg_clipfrac_lower"].aggregate() == 0.0

    # Reproduce the engine's micro-batch and DP aggregation shape contract: every
    # metric must receive one value from every micro-batch on every rank.
    rank_with_empty = {}
    append_to_dict(rank_with_empty, active_metrics)
    append_to_dict(rank_with_empty, empty_metrics)
    rank_without_empty = {}
    append_to_dict(rank_without_empty, active_metrics)
    append_to_dict(rank_without_empty, active_metrics)
    for key in active_metrics:
        assert len(rank_with_empty[key].values) == 2
        assert len(rank_without_empty[key].values) == 2
        Metric.aggregate_dp([rank_with_empty[key], rank_without_empty[key]])

    empty_loss.backward()
    torch.testing.assert_close(empty_log_probs.grad, torch.zeros_like(empty_log_probs))
    assert active_loss.item() != 0.0


def test_fsdp_token_reward_direct_preserves_scalar_log_probs_for_joint_rlvr_on_cpu():
    engine = object.__new__(FSDPEngineWithLMHead)
    engine.use_ulysses_sp = False
    engine.engine_config = SimpleNamespace(entropy_checkpointing=False)
    engine._response_target_ids_rmpad = lambda **kwargs: torch.tensor(  # type: ignore[method-assign]
        [[0, 1]] * 6,
        dtype=torch.long,
    )

    micro_batch = TensorDict({}, batch_size=[])
    tu.assign_non_tensor(
        micro_batch,
        use_remove_padding=True,
        use_fused_kernels=False,
        calculate_entropy=False,
        calculate_sum_pi_squared=False,
        distillation_use_topk=False,
        diagnostics_student_topk=0,
        token_reward_direct_topk=2,
        teacher_segment_sft_enabled=False,
        joint_token_reward_rlvr=True,
    )
    raw_logits = torch.randn(1, 6, 5, requires_grad=True)
    output = SimpleNamespace(logits=raw_logits + 0.0)
    output_args = {
        "input_ids_rmpad_rolled": torch.tensor([1, 2, 3, 4, 0, 1], dtype=torch.long),
        "temperature_rmpad": torch.ones(6),
        "output_offsets": torch.tensor([0, 6], dtype=torch.long),
    }

    model_output = FSDPEngineWithLMHead.prepare_model_outputs(
        engine,
        output=output,
        output_args=output_args,
        micro_batch=micro_batch,
        logits_processor_func=None,
    )

    assert model_output["log_probs"].values.shape == (6, 2)
    assert model_output["response_token_log_probs"].values.shape == (6,)
    combined_loss = model_output["log_probs"].values.sum() + model_output["response_token_log_probs"].values.sum()
    combined_loss.backward()
    assert raw_logits.grad is not None
    assert torch.isfinite(raw_logits.grad).all()
