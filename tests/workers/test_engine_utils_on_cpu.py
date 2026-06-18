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

from verl.utils.tensordict_utils import safe_unbind_jagged_nested_tensor


def test_safe_unbind_jagged_nested_tensor_rebases_global_offsets_on_cpu():
    values = torch.arange(7)
    offsets = torch.tensor([2, 5, 7], dtype=torch.int64)
    nested = torch.nested.nested_tensor_from_jagged(values, offsets)

    tensors = safe_unbind_jagged_nested_tensor(nested, context="unit-test")

    assert [tensor.tolist() for tensor in tensors] == [[2, 3, 4], [5, 6]]


def test_safe_unbind_jagged_nested_tensor_keeps_local_values_with_nonzero_offsets_on_cpu():
    values = torch.arange(5)
    offsets = torch.tensor([2, 5, 7], dtype=torch.int64)
    nested = torch.nested.nested_tensor_from_jagged(values, offsets)

    tensors = safe_unbind_jagged_nested_tensor(nested, context="unit-test")

    assert [tensor.tolist() for tensor in tensors] == [[0, 1, 2], [3, 4]]


def test_safe_unbind_jagged_nested_tensor_reports_short_values_on_cpu():
    values = torch.arange(4)
    offsets = torch.tensor([0, 3, 5], dtype=torch.int64)
    nested = torch.nested.nested_tensor_from_jagged(values, offsets)

    with pytest.raises(RuntimeError, match="values are shorter than the offsets range"):
        safe_unbind_jagged_nested_tensor(nested, context="unit-test")
