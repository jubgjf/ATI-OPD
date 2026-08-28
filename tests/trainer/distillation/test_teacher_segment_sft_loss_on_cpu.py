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
from verl.utils import tensordict_utils as tu
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead


def _teacher_sft_config(*, coef: float = 1.0, use_actor_loss: bool = False):
    return SimpleNamespace(
        enabled=True,
        use_actor_loss=use_actor_loss,
        intervention=SimpleNamespace(
            enabled=True,
            train_on_teacher_segments=True,
            teacher_segment_loss_mode="sft",
            teacher_segment_sft_loss_coef=coef,
        ),
        distillation_loss=SimpleNamespace(
            use_task_rewards=True,
            distillation_loss_coef=1.0,
        ),
    )


def _teacher_sft_data(
    teacher_segment_mask: torch.Tensor,
    *,
    response_mask: torch.Tensor | None = None,
    batch_num_teacher_tokens: int | None = None,
    batch_num_tokens: int | None = None,
    dp_size: int = 1,
) -> TensorDict:
    if response_mask is None:
        response_mask = 1 - teacher_segment_mask
    data = TensorDict(
        {
            "prompts": torch.tensor([[10, 11]], dtype=torch.long),
            "responses": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            "attention_mask": torch.ones(1, 6, dtype=torch.long),
            "response_mask": response_mask,
            "teacher_segment_mask": teacher_segment_mask,
        },
        batch_size=[1],
    )
    tu.assign_non_tensor(
        data,
        batch_num_teacher_tokens=(
            int(teacher_segment_mask.sum().item()) if batch_num_teacher_tokens is None else batch_num_teacher_tokens
        ),
        batch_num_tokens=(int(response_mask.sum().item()) if batch_num_tokens is None else batch_num_tokens),
        dp_size=dp_size,
    )
    return data


def test_teacher_segment_sft_uses_global_teacher_token_mean_and_only_teacher_mask_on_cpu():
    teacher_mask = torch.tensor([[0, 1, 0, 1]], dtype=torch.long)
    data = _teacher_sft_data(
        teacher_mask,
        batch_num_teacher_tokens=4,
        dp_size=2,
    )
    # no_padding_2_padding maps full-sequence values[1:5] to the four response tokens.
    response_token_log_probs = torch.tensor(
        [9.0, -1.0, -2.0, -3.0, -4.0, 9.0],
        requires_grad=True,
    )

    loss, metrics = distillation_losses.teacher_segment_sft_loss(
        _teacher_sft_config(),
        {"log_probs": response_token_log_probs},
        data,
    )

    # Local teacher NLL is 2 + 4; divide by four global teacher tokens and
    # multiply by dp_size=2 to compensate for data-parallel gradient averaging.
    torch.testing.assert_close(loss, torch.tensor(3.0))
    assert metrics["distillation/teacher_segment_sft_tokens"].aggregate() == 4.0

    loss.backward()
    torch.testing.assert_close(
        response_token_log_probs.grad,
        torch.tensor([0.0, 0.0, -0.5, 0.0, -0.5, 0.0]),
    )


def test_teacher_segment_sft_empty_mask_returns_graph_connected_zero_on_cpu():
    teacher_mask = torch.zeros(1, 4, dtype=torch.long)
    data = _teacher_sft_data(teacher_mask, batch_num_teacher_tokens=0)
    response_token_log_probs = torch.tensor(
        [9.0, -1.0, -2.0, -3.0, -4.0, 9.0],
        requires_grad=True,
    )

    loss, _ = distillation_losses.teacher_segment_sft_loss(
        _teacher_sft_config(),
        {"log_probs": response_token_log_probs},
        data,
    )

    assert loss.item() == 0.0
    loss.backward()
    torch.testing.assert_close(response_token_log_probs.grad, torch.zeros_like(response_token_log_probs))


def test_teacher_segment_sft_uses_scalar_response_log_probs_with_token_reward_topk_on_cpu(monkeypatch):
    teacher_mask = torch.tensor([[0, 1, 0, 1]], dtype=torch.long)
    data = _teacher_sft_data(teacher_mask)
    response_token_log_probs = torch.tensor(
        [9.0, -1.0, -2.0, -3.0, -4.0, 9.0],
        requires_grad=True,
    )
    token_reward_topk_log_probs = torch.zeros(6, 3, requires_grad=True)
    monkeypatch.setattr(
        distillation_losses,
        "_ppo_loss_or_zero",
        lambda config, model_output, data, dp_group: (token_reward_topk_log_probs.sum() * 0.0 + 2.0, {}),
    )

    loss, metrics = distillation_losses.distillation_ppo_loss(
        config=SimpleNamespace(),
        distillation_config=_teacher_sft_config(coef=0.25),
        model_output={
            "log_probs": token_reward_topk_log_probs,
            "response_token_log_probs": response_token_log_probs,
        },
        data=data,
    )

    # Base policy loss 2.0 + coefficient 0.25 * teacher-token mean NLL 3.0.
    torch.testing.assert_close(loss, torch.tensor(2.75))
    assert metrics["distillation/teacher_segment_sft_loss_coef"].aggregate() == 0.25
    loss.backward()
    torch.testing.assert_close(
        response_token_log_probs.grad,
        torch.tensor([0.0, 0.0, -0.125, 0.0, -0.125, 0.0]),
    )
    torch.testing.assert_close(token_reward_topk_log_probs.grad, torch.zeros_like(token_reward_topk_log_probs))


def test_teacher_segment_sft_rejects_topk_log_probs_without_scalar_response_log_probs_on_cpu():
    teacher_mask = torch.tensor([[0, 1, 0, 1]], dtype=torch.long)
    data = _teacher_sft_data(teacher_mask)

    with pytest.raises(ValueError, match="requires scalar current-policy log probabilities"):
        distillation_losses.teacher_segment_sft_loss(
            _teacher_sft_config(),
            {"log_probs": torch.zeros(6, 3)},
            data,
        )


def test_teacher_segment_sft_rejects_teacher_tokens_in_policy_response_mask_on_cpu():
    teacher_mask = torch.tensor([[0, 1, 0, 1]], dtype=torch.long)
    data = _teacher_sft_data(teacher_mask, response_mask=torch.ones_like(teacher_mask))

    with pytest.raises(ValueError, match="excluded from response_mask"):
        distillation_losses.teacher_segment_sft_loss(
            _teacher_sft_config(),
            {"log_probs": torch.zeros(6)},
            data,
        )


def test_teacher_only_batch_skips_ppo_and_keeps_sft_on_cpu(monkeypatch):
    teacher_mask = torch.ones(1, 4, dtype=torch.long)
    data = _teacher_sft_data(
        teacher_mask,
        response_mask=torch.zeros_like(teacher_mask),
        batch_num_tokens=0,
    )
    response_token_log_probs = torch.tensor(
        [9.0, -1.0, -2.0, -3.0, -4.0, 9.0],
        requires_grad=True,
    )

    def fail_ppo(*args, **kwargs):
        raise AssertionError("ppo_loss must not run for a globally teacher-only batch")

    monkeypatch.setattr(distillation_losses, "ppo_loss", fail_ppo)
    loss, metrics = distillation_losses.distillation_ppo_loss(
        config=SimpleNamespace(),
        distillation_config=_teacher_sft_config(),
        model_output={"log_probs": response_token_log_probs},
        data=data,
    )

    torch.testing.assert_close(loss, torch.tensor(2.5))
    assert metrics["actor/empty_action_mask"].aggregate() == 1.0


def test_fsdp_token_reward_direct_preserves_scalar_log_probs_for_teacher_sft_on_cpu():
    engine = object.__new__(FSDPEngineWithLMHead)
    engine.model_config = SimpleNamespace(tokenizer=range(5), processor=None)
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
        teacher_segment_sft_enabled=True,
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
