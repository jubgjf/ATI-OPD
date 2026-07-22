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

import pytest
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.workers import engine_workers
from verl.workers.utils import losses as worker_losses


class _Config(dict):
    def __getattr__(self, key):
        return self[key]


def _make_actor_config(*, entropy_coeff=0.0, use_kl_loss=False):
    return _Config(
        clip_ratio=0.2,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        clip_ratio_c=3.0,
        global_batch_info={},
        loss_agg_mode="token-mean",
        loss_scale_factor=None,
        policy_loss={"loss_mode": "vanilla"},
        entropy_coeff=entropy_coeff,
        use_kl_loss=use_kl_loss,
        kl_loss_type="low_var_kl",
        kl_loss_coef=0.25,
    )


def _make_ppo_data(
    *,
    response_mask,
    advantages,
    policy_loss_mask=None,
    batch_num_policy_tokens=None,
    ref_log_prob=None,
    dp_size=1,
):
    old_log_probs = torch.zeros_like(advantages)
    tensors = {
        "response_mask": response_mask,
        "old_log_probs": old_log_probs,
        "advantages": advantages,
    }
    if policy_loss_mask is not None:
        tensors["policy_loss_mask"] = policy_loss_mask
    if ref_log_prob is not None:
        tensors["ref_log_prob"] = ref_log_prob
    data = TensorDict(tensors, batch_size=[response_mask.shape[0]])
    metadata = {
        "dp_size": dp_size,
        "batch_num_tokens": int(response_mask.sum().item()),
        "global_batch_size": response_mask.shape[0] * dp_size,
    }
    if batch_num_policy_tokens is not None:
        metadata["batch_num_policy_tokens"] = batch_num_policy_tokens
    tu.assign_non_tensor(data, **metadata)
    return data


@pytest.mark.parametrize(
    "response_mask",
    [
        torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
        torch.tensor([[1, 1, 1, 0]], dtype=torch.long),
    ],
)
def test_policy_loss_uses_horizon_tokens_independent_of_suffix_length_on_cpu(monkeypatch, response_mask):
    monkeypatch.setattr(worker_losses, "no_padding_2_padding", lambda tensor, data: tensor)
    policy_loss_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)
    advantages = torch.tensor([[[1.0, 1.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]])
    log_prob = torch.zeros_like(advantages)
    log_prob[:, 2:] = 5.0
    log_prob.requires_grad_()
    data = _make_ppo_data(
        response_mask=response_mask,
        advantages=advantages,
        policy_loss_mask=policy_loss_mask,
        batch_num_policy_tokens=2,
    )

    loss, metrics = worker_losses.ppo_loss(_make_actor_config(), {"log_probs": log_prob}, data)

    # Four top-k loss values are summed over two active response positions. The top-k
    # dimension contributes to the numerator but is not multiplied into the denominator.
    assert loss.item() == pytest.approx(-2.0)
    assert metrics["actor/ppo_kl"].aggregate() == 0.0
    assert metrics["actor/pg_clipfrac"].aggregate() == 0.0


def test_policy_loss_without_horizon_mask_preserves_full_response_normalization_on_cpu(monkeypatch):
    monkeypatch.setattr(worker_losses, "no_padding_2_padding", lambda tensor, data: tensor)
    response_mask = torch.ones(1, 4, dtype=torch.long)
    advantages = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    log_prob = torch.zeros_like(advantages, requires_grad=True)
    data = _make_ppo_data(response_mask=response_mask, advantages=advantages)

    loss, _ = worker_losses.ppo_loss(_make_actor_config(), {"log_probs": log_prob}, data)

    assert loss.item() == pytest.approx(-0.5)


def test_policy_denominator_is_shared_by_micro_batches_on_cpu(monkeypatch):
    monkeypatch.setattr(worker_losses, "no_padding_2_padding", lambda tensor, data: tensor)
    config = _make_actor_config()
    losses = []
    for policy_loss_mask in (
        torch.tensor([[1, 0]], dtype=torch.long),
        torch.tensor([[0, 1]], dtype=torch.long),
    ):
        response_mask = torch.ones(1, 2, dtype=torch.long)
        advantages = policy_loss_mask.float()
        log_prob = torch.zeros_like(advantages, requires_grad=True)
        data = _make_ppo_data(
            response_mask=response_mask,
            advantages=advantages,
            policy_loss_mask=policy_loss_mask,
            batch_num_policy_tokens=2,
        )
        loss, _ = worker_losses.ppo_loss(config, {"log_probs": log_prob}, data)
        losses.append(loss)

    assert sum(loss.item() for loss in losses) == pytest.approx(-1.0)


def test_policy_token_count_metadata_is_replicated_to_every_micro_batch_on_cpu():
    mini_batch = TensorDict(
        {"policy_loss_mask": torch.tensor([[1, 0], [0, 1]], dtype=torch.long)},
        batch_size=[2],
    )
    tu.assign_non_tensor(mini_batch, batch_num_policy_tokens=2)

    micro_batches = tu.chunk_tensordict(mini_batch, chunks=2)

    assert [tu.get(micro_batch, "batch_num_policy_tokens") for micro_batch in micro_batches] == [2, 2]


@pytest.mark.parametrize("batch_num_policy_tokens", [0, 2])
def test_empty_policy_micro_batch_returns_graph_connected_zero_on_cpu(monkeypatch, batch_num_policy_tokens):
    monkeypatch.setattr(worker_losses, "no_padding_2_padding", lambda tensor, data: tensor)
    response_mask = torch.ones(1, 3, dtype=torch.long)
    policy_loss_mask = torch.zeros_like(response_mask)
    log_prob = torch.zeros(1, 3, requires_grad=True)
    data = _make_ppo_data(
        response_mask=response_mask,
        advantages=torch.ones_like(log_prob),
        policy_loss_mask=policy_loss_mask,
        batch_num_policy_tokens=batch_num_policy_tokens,
    )

    loss, metrics = worker_losses.ppo_loss(_make_actor_config(), {"log_probs": log_prob}, data)
    loss.backward()

    assert loss.item() == 0.0
    assert torch.equal(log_prob.grad, torch.zeros_like(log_prob))
    assert metrics["actor/ppo_kl"].aggregate() == 0.0
    assert metrics["actor/pg_clipfrac"].aggregate() == 0.0


def test_entropy_and_reference_kl_keep_full_response_denominator_on_cpu(monkeypatch):
    monkeypatch.setattr(worker_losses, "no_padding_2_padding", lambda tensor, data: tensor)
    monkeypatch.setattr(worker_losses, "kl_penalty", lambda logprob, ref_logprob, kl_penalty: torch.ones_like(logprob))
    response_mask = torch.ones(1, 4, dtype=torch.long)
    policy_loss_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)
    advantages = policy_loss_mask.float()
    log_prob = torch.zeros_like(advantages, requires_grad=True)
    config = _make_actor_config(entropy_coeff=0.5, use_kl_loss=True)
    data = _make_ppo_data(
        response_mask=response_mask,
        advantages=advantages,
        policy_loss_mask=policy_loss_mask,
        batch_num_policy_tokens=2,
        ref_log_prob=torch.zeros_like(log_prob),
    )

    loss, _ = worker_losses.ppo_loss(
        config,
        {"log_probs": log_prob, "entropy": torch.ones_like(log_prob)},
        data,
    )

    # PG=-1 over the active horizon; entropy=-0.5 and KL=+0.25 both use the full
    # four-token response denominator. The shared config is restored afterward.
    assert loss.item() == pytest.approx(-1.25)
    assert config.global_batch_info["batch_num_tokens"] == 4


def test_full_token_denominator_is_restored_when_policy_loss_raises_on_cpu(monkeypatch):
    monkeypatch.setattr(worker_losses, "no_padding_2_padding", lambda tensor, data: tensor)

    def failing_policy_loss(**kwargs):
        raise RuntimeError("synthetic policy-loss failure")

    monkeypatch.setattr(worker_losses, "get_policy_loss_fn", lambda loss_mode: failing_policy_loss)
    response_mask = torch.ones(1, 4, dtype=torch.long)
    policy_loss_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)
    log_prob = torch.zeros(1, 4)
    config = _make_actor_config()
    data = _make_ppo_data(
        response_mask=response_mask,
        advantages=policy_loss_mask.float(),
        policy_loss_mask=policy_loss_mask,
        batch_num_policy_tokens=2,
    )

    with pytest.raises(RuntimeError, match="synthetic policy-loss failure"):
        worker_losses.ppo_loss(config, {"log_probs": log_prob}, data)

    assert config.global_batch_info["batch_num_tokens"] == 4


def test_policy_mask_must_be_response_aligned_subset_on_cpu(monkeypatch):
    monkeypatch.setattr(worker_losses, "no_padding_2_padding", lambda tensor, data: tensor)
    response_mask = torch.tensor([[1, 1, 0]], dtype=torch.long)
    policy_loss_mask = torch.tensor([[1, 0, 1]], dtype=torch.long)
    log_prob = torch.zeros(1, 3)
    data = _make_ppo_data(
        response_mask=response_mask,
        advantages=torch.ones_like(log_prob),
        policy_loss_mask=policy_loss_mask,
        batch_num_policy_tokens=2,
    )

    with pytest.raises(ValueError, match="subset of response_mask"):
        worker_losses.ppo_loss(_make_actor_config(), {"log_probs": log_prob}, data)


def test_global_policy_token_count_sums_data_parallel_ranks_on_cpu(monkeypatch):
    calls = []

    def fake_all_reduce(count, op, group):
        calls.append((op, group))
        count.add_(4)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    count = engine_workers._global_policy_token_count(
        torch.tensor([[1, 0, 1], [0, 0, 1]], dtype=torch.long),
        dp_size=2,
        dp_group="test-group",
        device="cpu",
    )

    assert count == 7
    assert calls == [(torch.distributed.ReduceOp.SUM, "test-group")]


def test_global_policy_token_count_skips_collective_for_one_rank_on_cpu(monkeypatch):
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda *args, **kwargs: pytest.fail("all_reduce should not run for a single DP rank"),
    )

    count = engine_workers._global_policy_token_count(
        torch.tensor([[1, 0, 1]], dtype=torch.long),
        dp_size=1,
        dp_group=None,
        device="cpu",
    )

    assert count == 2
