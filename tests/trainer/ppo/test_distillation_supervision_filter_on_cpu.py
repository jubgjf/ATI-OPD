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

from verl import DataProto
from verl.trainer.distillation.batch_filter import to_tensordict_without_distillation_supervision
from verl.trainer.ppo import ray_trainer as ray_trainer_module
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class StopBeforeWorker(RuntimeError):
    pass


def _make_distillation_batch() -> DataProto:
    batch_size = 2
    seq_len = 5
    response_len = 3
    batch = TensorDict(
        {
            "input_ids": torch.arange(batch_size * seq_len).reshape(batch_size, seq_len),
            "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.int64),
            "response_mask": torch.ones(batch_size, response_len, dtype=torch.int64),
            "position_ids": torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1),
            "teacher_logprobs": torch.full((batch_size, seq_len, 1), -0.1),
            "teacher_ids": torch.zeros(batch_size, seq_len, 1, dtype=torch.int32),
        },
        batch_size=[batch_size],
    )
    return DataProto(batch=batch, meta_info={"temperature": 1.0})


def test_inference_tensordict_filter_removes_teacher_supervision_without_mutating_batch():
    batch = _make_distillation_batch()

    filtered = to_tensordict_without_distillation_supervision(batch)

    assert "teacher_logprobs" not in filtered.keys()
    assert "teacher_ids" not in filtered.keys()
    assert "input_ids" in filtered.keys()
    assert "attention_mask" in filtered.keys()
    assert "response_mask" in filtered.keys()
    assert "position_ids" in filtered.keys()
    assert "teacher_logprobs" in batch.batch.keys()
    assert "teacher_ids" in batch.batch.keys()


@pytest.mark.parametrize("method_name", ["_compute_values", "_compute_ref_log_prob", "_compute_old_log_prob"])
def test_inference_paths_strip_teacher_supervision_before_padding(monkeypatch, method_name):
    captured_keys = None

    def fake_left_right_2_no_padding(batch_td):
        nonlocal captured_keys
        captured_keys = set(batch_td.keys())
        raise StopBeforeWorker

    monkeypatch.setattr(ray_trainer_module, "left_right_2_no_padding", fake_left_right_2_no_padding)

    trainer = object.__new__(RayPPOTrainer)
    if method_name == "_compute_old_log_prob":
        trainer.config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(actor={"calculate_sum_pi_squared": False}))

    with pytest.raises(StopBeforeWorker):
        getattr(trainer, method_name)(_make_distillation_batch())

    assert captured_keys is not None
    assert "teacher_logprobs" not in captured_keys
    assert "teacher_ids" not in captured_keys
    assert {"input_ids", "attention_mask", "response_mask", "position_ids"} <= captured_keys


def test_update_actor_keeps_teacher_supervision_before_padding(monkeypatch):
    captured_keys = None

    def fake_left_right_2_no_padding(batch_td):
        nonlocal captured_keys
        captured_keys = set(batch_td.keys())
        raise StopBeforeWorker

    monkeypatch.setattr(ray_trainer_module, "left_right_2_no_padding", fake_left_right_2_no_padding)

    trainer = object.__new__(RayPPOTrainer)
    trainer.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(multi_turn=SimpleNamespace(enable=False), temperature=1.0)
        )
    )

    with pytest.raises(StopBeforeWorker):
        trainer._update_actor(_make_distillation_batch())

    assert captured_keys is not None
    assert "teacher_logprobs" in captured_keys
    assert "teacher_ids" in captured_keys
