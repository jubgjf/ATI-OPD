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

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def collate_reward_extra_infos(
    reward_extra_infos: Sequence[Mapping[str, Any]],
) -> tuple[list[str], dict[str, np.ndarray]]:
    """Collate heterogeneous per-sample reward diagnostics into aligned columns.

    Reward scorers may return task-specific or parse-path-specific diagnostic
    fields. Missing fields are represented by ``None`` instead of requiring
    every sample in a mixed-domain batch to expose the same keys.
    """
    normalized_infos: list[Mapping[str, Any]] = []
    reward_extra_keys: set[str] = set()
    for index, info in enumerate(reward_extra_infos):
        if not isinstance(info, Mapping):
            raise TypeError(
                "reward_extra_info must be a mapping for every sample; "
                f"got {type(info).__name__} at batch index {index}."
            )
        invalid_keys = [key for key in info if not isinstance(key, str)]
        if invalid_keys:
            raise TypeError(f"reward_extra_info keys must be strings; got {invalid_keys!r} at batch index {index}.")
        normalized_infos.append(info)
        reward_extra_keys.update(info)

    sorted_keys = sorted(reward_extra_keys)
    columns: dict[str, np.ndarray] = {}
    for key in sorted_keys:
        column = np.empty(len(normalized_infos), dtype=object)
        for index, info in enumerate(normalized_infos):
            column[index] = info.get(key)
        columns[key] = column
    return sorted_keys, columns
