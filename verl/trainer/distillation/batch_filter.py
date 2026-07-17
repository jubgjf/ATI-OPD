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

from verl import DataProto

DISTILLATION_SUPERVISION_TENSOR_KEYS = (
    "teacher_logprobs",
    "teacher_ids",
    "teacher_sampled_logprobs",
    "teacher_sampled_ids",
    "teacher_segment_mask",
)


def to_tensordict_without_distillation_supervision(batch: DataProto):
    """Build a TensorDict for inference-only passes without teacher supervision tensors."""
    batch_td = batch.to_tensordict()
    for key in DISTILLATION_SUPERVISION_TENSOR_KEYS:
        if key in batch_td.keys():
            batch_td.pop(key)
    return batch_td
