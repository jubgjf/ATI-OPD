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

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.distillation import losses as distillation_losses_module
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    apply_distillation_horizon_mask,
    apply_intervention_prune_mask,
    build_distillation_prefix_mask,
    compute_distillation_horizon_tokens,
)


def _object_array(values):
    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result


def test_fixed_horizon_masks_prefix_without_mutating_response_mask_on_cpu():
    response_mask = torch.tensor([[1, 1, 1, 0], [1, 0, 1, 1]], dtype=torch.long)
    original_response_mask = response_mask.clone()

    loss_mask = build_distillation_prefix_mask(response_mask=response_mask, horizon_tokens=2)

    expected = torch.tensor([[1, 1, 0, 0], [1, 0, 1, 0]], dtype=torch.long)
    assert torch.equal(loss_mask, expected)
    assert torch.equal(response_mask, original_response_mask)


def test_apply_fixed_horizon_attaches_mask_and_metrics_on_cpu():
    response_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.long)
    batch = TensorDict({"response_mask": response_mask.clone()}, batch_size=[2])
    data = DataProto(batch=batch)
    horizon_config = SimpleNamespace(mode="fixed", tokens=2)

    metrics = apply_distillation_horizon_mask(data=data, horizon_config=horizon_config, global_steps=0)

    assert torch.equal(data.batch["response_mask"], response_mask)
    assert torch.equal(data.batch["distillation_loss_mask"], torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0]]))
    assert metrics["distillation/horizon_tokens"] == 2.0
    assert metrics["distillation/mask_token_fraction"] == 4 / 7
    assert metrics["distillation/mask_tokens/mean"] == 2.0
    assert metrics["distillation/mask_clipped_sample_fraction"] == 1.0


def test_progressive_horizon_schedule_on_cpu():
    horizon_config = SimpleNamespace(
        mode="progressive",
        start_tokens=2,
        end_tokens=5,
        increment_tokens=2,
        interval_steps=10,
    )

    assert compute_distillation_horizon_tokens(horizon_config, global_steps=0) == 2
    assert compute_distillation_horizon_tokens(horizon_config, global_steps=9) == 2
    assert compute_distillation_horizon_tokens(horizon_config, global_steps=10) == 4
    assert compute_distillation_horizon_tokens(horizon_config, global_steps=20) == 5
    assert compute_distillation_horizon_tokens(horizon_config, global_steps=999) == 5


def test_apply_progressive_horizon_uses_start_tokens_on_first_logged_training_step_on_cpu():
    response_mask = torch.ones(1, 8, dtype=torch.long)
    batch = TensorDict({"response_mask": response_mask.clone()}, batch_size=[1])
    data = DataProto(batch=batch)
    horizon_config = SimpleNamespace(
        mode="progressive",
        start_tokens=2,
        end_tokens=4,
        increment_tokens=2,
        interval_steps=1,
    )

    metrics = apply_distillation_horizon_mask(data=data, horizon_config=horizon_config, global_steps=1)

    assert metrics["distillation/horizon_tokens"] == 2.0
    assert torch.equal(data.batch["distillation_loss_mask"], torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0]]))


def test_full_horizon_does_not_attach_distillation_loss_mask_on_cpu():
    batch = TensorDict({"response_mask": torch.ones(1, 4, dtype=torch.long)}, batch_size=[1])
    data = DataProto(batch=batch)
    horizon_config = SimpleNamespace(mode="full")

    metrics = apply_distillation_horizon_mask(data=data, horizon_config=horizon_config, global_steps=0)

    assert metrics == {}
    assert "distillation_loss_mask" not in data.batch.keys()


def test_prune_uses_first_intervention_and_keeps_unrepaired_samples_full_on_cpu():
    response_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 0, 0],
        ],
        dtype=torch.long,
    )
    batch = TensorDict(
        {
            "attention_mask": response_mask.clone(),
            "response_mask": response_mask.clone(),
        },
        batch_size=[2],
    )
    data = DataProto(
        batch=batch,
        non_tensor_batch={
            "teacher_interventions": _object_array(
                [
                    [
                        {"start": 2, "teacher_tokens": 1},
                        {"start": 4, "teacher_tokens": 1},
                    ],
                    [],
                ]
            )
        },
    )

    metrics = apply_intervention_prune_mask(
        data=data,
        intervention_config=SimpleNamespace(enabled=True, prune=True),
    )

    assert torch.equal(data.batch["response_mask"], response_mask)
    assert torch.equal(
        data.batch["distillation_loss_mask"],
        torch.tensor(
            [
                [1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 0, 0],
            ]
        ),
    )
    assert metrics["opd_intervention/prune/opd_token_fraction"] == pytest.approx(7 / 9)
    assert metrics["opd_intervention/prune/excluded_student_tokens_mean_among_repaired"] == 1.0
    assert metrics["opd_intervention/prune/cutoff_position_mean"] == 3.0
    assert metrics["opd_intervention/prune/cutoff_position_frac_mean"] == pytest.approx(3 / 5)
    assert metrics["opd_intervention/prune/empty_opd_sample_fraction"] == 0.0


def test_prune_sft_keeps_only_first_teacher_span_and_masks_student_suffix_on_cpu():
    response_mask = torch.tensor([[1, 1, 0, 1, 0, 1, 1]], dtype=torch.long)
    teacher_segment_mask = torch.tensor([[0, 0, 1, 0, 1, 0, 0]], dtype=torch.long)
    batch = TensorDict(
        {
            "attention_mask": torch.ones(1, 7, dtype=torch.long),
            "response_mask": response_mask.clone(),
            "teacher_segment_mask": teacher_segment_mask.clone(),
        },
        batch_size=[1],
    )
    data = DataProto(
        batch=batch,
        non_tensor_batch={
            "teacher_interventions": _object_array(
                [[{"start": 2, "teacher_tokens": 1}, {"start": 4, "teacher_tokens": 1}]]
            )
        },
    )

    apply_intervention_prune_mask(
        data=data,
        intervention_config=SimpleNamespace(enabled=True, prune=True),
    )

    assert torch.equal(data.batch["response_mask"], response_mask)
    assert torch.equal(data.batch["distillation_loss_mask"], torch.tensor([[1, 1, 0, 0, 0, 0, 0]]))
    assert torch.equal(data.batch["teacher_segment_mask"], torch.tensor([[0, 0, 1, 0, 0, 0, 0]]))


def test_prune_allows_empty_opd_prefix_when_sft_teacher_starts_at_zero_on_cpu():
    batch = TensorDict(
        {
            "attention_mask": torch.tensor([[1, 1, 1, 0]], dtype=torch.long),
            "response_mask": torch.tensor([[0, 1, 1, 0]], dtype=torch.long),
            "teacher_segment_mask": torch.tensor([[1, 0, 0, 0]], dtype=torch.long),
        },
        batch_size=[1],
    )
    data = DataProto(
        batch=batch,
        non_tensor_batch={"teacher_interventions": _object_array([[{"start": 0, "teacher_tokens": 1}]])},
    )

    metrics = apply_intervention_prune_mask(
        data=data,
        intervention_config=SimpleNamespace(enabled=True, prune=True),
    )

    assert not data.batch["distillation_loss_mask"].any()
    assert torch.equal(data.batch["teacher_segment_mask"], torch.tensor([[1, 0, 0, 0]]))
    assert metrics["opd_intervention/prune/empty_opd_sample_fraction"] == 1.0


def test_prune_teacher_terminal_at_response_end_keeps_all_available_supervision_on_cpu():
    response_mask = torch.ones(1, 3, dtype=torch.long)
    data = DataProto(
        batch=TensorDict(
            {
                "attention_mask": response_mask.clone(),
                "response_mask": response_mask.clone(),
            },
            batch_size=[1],
        ),
        non_tensor_batch={"teacher_interventions": _object_array([[{"start": 1, "teacher_tokens": 2}]])},
    )

    metrics = apply_intervention_prune_mask(
        data=data,
        intervention_config=SimpleNamespace(enabled=True, prune=True),
    )

    assert torch.equal(data.batch["distillation_loss_mask"], response_mask)
    assert metrics["opd_intervention/prune/excluded_student_tokens_mean_among_repaired"] == 0.0
    assert metrics["opd_intervention/prune/cutoff_position_frac_mean"] == 1.0


def test_adaptive_prune_uses_metadata_cutoff_and_preserves_full_rollout_on_cpu():
    response_mask = torch.ones(1, 8, dtype=torch.long)
    original_attention_mask = response_mask.clone()
    data = DataProto(
        batch=TensorDict(
            {
                "attention_mask": original_attention_mask.clone(),
                "response_mask": response_mask.clone(),
                "distillation_loss_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0]], dtype=torch.long),
                "teacher_segment_mask": torch.tensor([[0, 0, 1, 1, 0, 0, 0, 0]], dtype=torch.long),
                "rm_scores": torch.tensor([[0, 0, 0, 0, 0, 0, 0, 1]], dtype=torch.float32),
            },
            batch_size=[1],
        ),
        non_tensor_batch={
            "teacher_interventions": _object_array([[{"start": 2, "teacher_tokens": 2, "prune_future_cutoff": 6}]])
        },
    )
    original_reward = data.batch["rm_scores"].clone()

    metrics = apply_intervention_prune_mask(
        data=data,
        intervention_config=SimpleNamespace(enabled=True, prune=True, prune_mode="adaptive_future"),
    )

    assert torch.equal(data.batch["response_mask"], response_mask)
    assert torch.equal(data.batch["attention_mask"], original_attention_mask)
    assert torch.equal(data.batch["rm_scores"], original_reward)
    assert torch.equal(data.batch["distillation_loss_mask"], torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]]))
    assert torch.equal(data.batch["teacher_segment_mask"], torch.tensor([[0, 0, 1, 1, 0, 0, 0, 0]]))
    assert metrics["opd_intervention/prune/cutoff_position_mean"] == 6.0


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ({"start": 1, "teacher_tokens": 2}, "prune_future_cutoff"),
        ({"start": 1, "teacher_tokens": 2, "prune_future_cutoff": 2}, "teacher_end <= cutoff"),
        ({"start": 1, "teacher_tokens": 2, "prune_future_cutoff": 6}, "teacher_end <= cutoff"),
    ],
)
def test_adaptive_prune_rejects_missing_or_out_of_range_cutoff_on_cpu(record, message):
    response_mask = torch.ones(1, 5, dtype=torch.long)
    data = DataProto(
        batch=TensorDict(
            {"attention_mask": response_mask.clone(), "response_mask": response_mask.clone()},
            batch_size=[1],
        ),
        non_tensor_batch={"teacher_interventions": _object_array([[record]])},
    )

    with pytest.raises(RuntimeError, match=message):
        apply_intervention_prune_mask(
            data=data,
            intervention_config=SimpleNamespace(enabled=True, prune=True, prune_mode="adaptive_future"),
        )


@pytest.mark.parametrize(
    "horizon_config",
    [
        SimpleNamespace(mode="fixed", tokens=2),
        SimpleNamespace(
            mode="progressive",
            start_tokens=2,
            end_tokens=6,
            increment_tokens=2,
            interval_steps=10,
        ),
    ],
)
def test_prune_intersects_existing_horizon_mask_on_cpu(horizon_config):
    response_mask = torch.ones(1, 6, dtype=torch.long)
    data = DataProto(
        batch=TensorDict(
            {
                "attention_mask": response_mask.clone(),
                "response_mask": response_mask.clone(),
            },
            batch_size=[1],
        ),
        non_tensor_batch={"teacher_interventions": _object_array([[{"start": 3, "teacher_tokens": 1}]])},
    )
    apply_distillation_horizon_mask(
        data=data,
        horizon_config=horizon_config,
        global_steps=1,
    )

    apply_intervention_prune_mask(
        data=data,
        intervention_config=SimpleNamespace(enabled=True, prune=True),
    )

    assert torch.equal(data.batch["distillation_loss_mask"], torch.tensor([[1, 1, 0, 0, 0, 0]]))


def test_prune_rejects_missing_or_out_of_range_intervention_metadata_on_cpu():
    response_mask = torch.ones(1, 3, dtype=torch.long)
    batch = TensorDict(
        {"attention_mask": response_mask.clone(), "response_mask": response_mask.clone()},
        batch_size=[1],
    )
    config = SimpleNamespace(enabled=True, prune=True)

    with pytest.raises(RuntimeError, match="teacher_interventions metadata"):
        apply_intervention_prune_mask(data=DataProto(batch=batch.clone()), intervention_config=config)

    data = DataProto(
        batch=batch.clone(),
        non_tensor_batch={"teacher_interventions": _object_array([None])},
    )
    with pytest.raises(RuntimeError, match="null teacher_interventions metadata"):
        apply_intervention_prune_mask(data=data, intervention_config=config)

    data = DataProto(
        batch=batch.clone(),
        non_tensor_batch={"teacher_interventions": _object_array([[{"start": 2, "teacher_tokens": 2}]])},
    )
    with pytest.raises(RuntimeError, match="teacher span exceeds"):
        apply_intervention_prune_mask(data=data, intervention_config=config)


def test_distillation_ppo_loss_keeps_task_ppo_on_full_response_mask_on_cpu(monkeypatch):
    full_response_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.long)
    prefix_loss_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)
    data = TensorDict(
        {
            "response_mask": full_response_mask,
            "distillation_loss_mask": prefix_loss_mask,
        },
        batch_size=[1],
    )
    captured = {}

    def fake_distillation_loss(config, distillation_config, model_output, data):
        captured["distillation_loss_mask"] = data["distillation_loss_mask"].clone()
        return torch.tensor(2.0), {}

    def fake_ppo_loss(config, model_output, data, dp_group):
        captured["ppo_response_mask"] = data["response_mask"].clone()
        return torch.tensor(3.0), {}

    monkeypatch.setattr(distillation_losses_module, "distillation_loss", fake_distillation_loss)
    monkeypatch.setattr(distillation_losses_module, "ppo_loss", fake_ppo_loss)

    distillation_config = SimpleNamespace(
        distillation_loss=SimpleNamespace(use_task_rewards=True, distillation_loss_coef=0.5)
    )
    loss, _ = distillation_losses_module.distillation_ppo_loss(
        config=SimpleNamespace(),
        distillation_config=distillation_config,
        model_output={},
        data=data,
    )

    assert torch.equal(captured["distillation_loss_mask"], prefix_loss_mask)
    assert torch.equal(captured["ppo_response_mask"], full_response_mask)
    assert loss.item() == 4.0


def test_empty_distillation_loss_mask_returns_zero_loss_on_cpu(monkeypatch):
    def fake_loss_fn(config, distillation_config, model_output, data):
        return torch.ones(1, 3), {}

    monkeypatch.setattr(distillation_losses_module, "get_distillation_loss_fn", lambda loss_mode: fake_loss_fn)

    data = TensorDict(
        {
            "response_mask": torch.ones(1, 3, dtype=torch.long),
            "distillation_loss_mask": torch.zeros(1, 3, dtype=torch.long),
        },
        batch_size=[1],
    )
    distillation_config = SimpleNamespace(
        distillation_loss=SimpleNamespace(
            loss_mode="fake",
            loss_max_clamp=None,
            use_policy_gradient=False,
        )
    )

    loss, metrics = distillation_losses_module.distillation_loss(
        config=SimpleNamespace(loss_agg_mode="token-mean", global_batch_info={}),
        distillation_config=distillation_config,
        model_output={},
        data=data,
    )

    assert loss.item() == 0.0
    assert metrics["distillation/mask_empty"].aggregate() == 1.0


def _make_horizon_policy_trainer(
    *,
    horizon_mode="fixed",
    adv_estimator="token_reward_direct",
    intervention_enabled=False,
    prune=False,
    use_actor_loss=False,
    policy_loss_mode="vanilla",
    loss_agg_mode="token-mean",
):
    trainer = object.__new__(RayPPOTrainer)
    trainer.distillation_config = SimpleNamespace(
        enabled=True,
        use_actor_loss=use_actor_loss,
        intervention=SimpleNamespace(enabled=intervention_enabled, prune=prune),
        distillation_loss=SimpleNamespace(horizon=SimpleNamespace(mode=horizon_mode)),
    )
    trainer.config = SimpleNamespace(
        algorithm=SimpleNamespace(adv_estimator=adv_estimator),
        actor_rollout_ref=SimpleNamespace(
            actor=SimpleNamespace(
                policy_loss={"loss_mode": policy_loss_mode},
                loss_agg_mode=loss_agg_mode,
            )
        ),
    )
    return trainer


def test_horizon_policy_loss_normalization_gate_accepts_topd_and_popd_on_cpu():
    assert _make_horizon_policy_trainer(horizon_mode="fixed")._uses_distillation_policy_loss_mask()
    assert _make_horizon_policy_trainer(horizon_mode="progressive")._uses_distillation_policy_loss_mask()
    assert _make_horizon_policy_trainer(
        horizon_mode="full", intervention_enabled=True, prune=True
    )._uses_distillation_policy_loss_mask()


def test_horizon_policy_loss_normalization_gate_isolates_other_paths_on_cpu():
    assert not _make_horizon_policy_trainer(horizon_mode="full")._uses_distillation_policy_loss_mask()
    assert not _make_horizon_policy_trainer(intervention_enabled=True)._uses_distillation_policy_loss_mask()
    assert not _make_horizon_policy_trainer(adv_estimator="grpo")._uses_distillation_policy_loss_mask()


def test_horizon_policy_loss_normalization_gate_rejects_unsupported_actor_loss_on_cpu():
    with pytest.raises(NotImplementedError, match="loss_mode='vanilla'"):
        _make_horizon_policy_trainer(policy_loss_mode="dppo_tv")._uses_distillation_policy_loss_mask()
    with pytest.raises(NotImplementedError, match="loss_agg_mode='token-mean'"):
        _make_horizon_policy_trainer(loss_agg_mode="seq-mean-token-mean")._uses_distillation_policy_loss_mask()
    with pytest.raises(ValueError, match="use_actor_loss=False"):
        _make_horizon_policy_trainer(use_actor_loss=True)._uses_distillation_policy_loss_mask()
