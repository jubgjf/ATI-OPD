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

import torch
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.distillation import losses as distillation_losses_module
from verl.trainer.ppo.ray_trainer import (
    apply_distillation_horizon_mask,
    build_distillation_prefix_mask,
    compute_distillation_horizon_tokens,
)


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


def test_full_horizon_does_not_attach_distillation_loss_mask_on_cpu():
    batch = TensorDict({"response_mask": torch.ones(1, 4, dtype=torch.long)}, batch_size=[1])
    data = DataProto(batch=batch)
    horizon_config = SimpleNamespace(mode="full")

    metrics = apply_distillation_horizon_mask(data=data, horizon_config=horizon_config, global_steps=0)

    assert metrics == {}
    assert "distillation_loss_mask" not in data.batch.keys()


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
