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

import pytest
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.workers.engine.utils import postprocess_batch_func


def test_nested_tensor_from_jagged_values_uses_independent_local_offsets_on_cpu():
    input_ids = torch.nested.as_nested_tensor([torch.arange(3), torch.arange(2)], layout=torch.jagged)
    values = torch.arange(5, dtype=torch.float32)

    nested = tu.nested_tensor_from_jagged_values(values, input_ids.offsets(), context="unit-test")

    assert nested.offsets().data_ptr() != input_ids.offsets().data_ptr()
    assert nested.offsets().tolist() == [0, 3, 5]
    assert [tensor.tolist() for tensor in tu.safe_unbind_jagged_nested_tensor(nested, context="unit-test")] == [
        [0.0, 1.0, 2.0],
        [3.0, 4.0],
    ]


def test_jagged_tensor_parts_split_by_cpu_lengths_on_cpu():
    values = torch.arange(5, dtype=torch.float32)
    offsets = torch.tensor([10, 12, 15], dtype=torch.long)

    parts = tu.jagged_tensor_parts_from_values(values, offsets, context="unit-test")
    tensors, ragged_idx = tu.safe_unbind_jagged_output(parts, context="unit-test")

    assert parts.lengths == (2, 3)
    assert ragged_idx == 1
    assert [tensor.tolist() for tensor in tensors] == [[0.0, 1.0], [2.0, 3.0, 4.0]]


def test_jagged_tensor_parts_rejects_bad_offsets_on_cpu():
    values = torch.arange(4, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="reference offsets must be non-decreasing"):
        tu.jagged_tensor_parts_from_values(
            values,
            torch.tensor([0, 3, 2], dtype=torch.long),
            context="unit-test",
        )

    with pytest.raises(RuntimeError, match="values length does not match reference offsets"):
        tu.jagged_tensor_parts_from_values(
            values,
            torch.tensor([0, 2, 5], dtype=torch.long),
            context="unit-test",
        )


def test_safe_unbind_jagged_nested_tensor_rejects_nonzero_offsets_on_cpu():
    values = torch.arange(5)
    offsets = torch.tensor([2, 5, 7], dtype=torch.int64)
    nested = torch.nested.nested_tensor_from_jagged(values, offsets)

    with pytest.raises(RuntimeError, match="offsets must start at 0"):
        tu.safe_unbind_jagged_nested_tensor(nested, context="unit-test")


def test_safe_unbind_jagged_nested_tensor_reports_bad_offsets_on_cpu():
    values = torch.arange(4)
    offsets = torch.tensor([0, 3, 5], dtype=torch.int64)
    nested = torch.nested.nested_tensor_from_jagged(values, offsets)

    with pytest.raises(RuntimeError, match="offsets end must match values length"):
        tu.safe_unbind_jagged_nested_tensor(nested, context="unit-test")


def test_safe_unbind_jagged_nested_tensor_does_not_call_native_unbind_on_cpu(monkeypatch):
    nested = torch.nested.as_nested_tensor([torch.arange(2), torch.arange(3) + 10], layout=torch.jagged)

    def raise_if_called(self, *args, **kwargs):
        raise AssertionError("native unbind should not be called for jagged NestedTensor")

    monkeypatch.setattr(torch.Tensor, "unbind", raise_if_called)

    tensors = tu.safe_unbind_jagged_nested_tensor(nested, context="unit-test")

    assert [tensor.tolist() for tensor in tensors] == [[0, 1], [10, 11, 12]]


def test_postprocess_batch_func_splits_parts_without_nested_offsets_on_cpu(monkeypatch):
    def raise_if_called(*args, **kwargs):
        raise AssertionError("NestedTensor split should not be called for JaggedTensorParts")

    monkeypatch.setattr(tu, "safe_unbind_jagged_nested_tensor", raise_if_called)

    sample_2 = tu.jagged_tensor_parts_from_values(
        torch.tensor([20.0, 21.0]),
        torch.tensor([0, 2], dtype=torch.long),
        context="micro-batch-0",
    )
    samples_0_1 = tu.jagged_tensor_parts_from_values(
        torch.tensor([0.0, 1.0, 2.0, 10.0]),
        torch.tensor([7, 10, 11], dtype=torch.long),
        context="micro-batch-1",
    )
    output = postprocess_batch_func(
        output_lst=[
            {"model_output": {"log_probs": sample_2}, "metrics": {"mfu": 1.0}},
            {"model_output": {"log_probs": samples_0_1}, "metrics": {"mfu": 2.0}},
        ],
        indices=[[2], [0, 1]],
        data=TensorDict({}, batch_size=[]),
    )

    log_probs = output["model_output"]["log_probs"]
    assert log_probs.offsets().tolist() == [0, 3, 4, 6]
    assert log_probs.values().tolist() == [0.0, 1.0, 2.0, 10.0, 20.0, 21.0]


def test_postprocess_batch_func_splits_and_restores_nested_outputs_on_cpu(monkeypatch):
    def raise_if_called(self, *args, **kwargs):
        raise AssertionError("native unbind should not be called during postprocess")

    monkeypatch.setattr(torch.Tensor, "unbind", raise_if_called)

    sample_2 = tu.nested_tensor_from_jagged_values(
        torch.tensor([20.0, 21.0]),
        torch.tensor([0, 2], dtype=torch.long),
        context="micro-batch-0",
    )
    samples_0_1 = tu.nested_tensor_from_jagged_values(
        torch.tensor([0.0, 1.0, 2.0, 10.0]),
        torch.tensor([0, 3, 4], dtype=torch.long),
        context="micro-batch-1",
    )
    output = postprocess_batch_func(
        output_lst=[
            {"model_output": {"log_probs": sample_2}, "metrics": {"mfu": 1.0}},
            {"model_output": {"log_probs": samples_0_1}, "metrics": {"mfu": 2.0}},
        ],
        indices=[[2], [0, 1]],
        data=TensorDict({}, batch_size=[]),
    )

    tensors = tu.safe_unbind_jagged_nested_tensor(
        output["model_output"]["log_probs"],
        context="postprocessed-log-probs",
    )
    assert [tensor.tolist() for tensor in tensors] == [[0.0, 1.0, 2.0], [10.0], [20.0, 21.0]]
