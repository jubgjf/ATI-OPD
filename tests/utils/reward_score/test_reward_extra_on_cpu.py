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

from verl.utils.reward_score.reward_extra import collate_reward_extra_infos


def test_collate_reward_extra_infos_aligns_mixed_domain_diagnostics_on_cpu():
    keys, columns = collate_reward_extra_infos(
        [
            {"score": 1.0, "acc": True, "task_type": "stem_mcqa", "predicted_answer": "C"},
            {"score": 0.0, "acc": False, "task_type": "structured_output", "json_parsed": False},
            {"score": 0.0, "acc": False, "task_type": "stem_mcqa", "parse_failure": True},
        ]
    )

    assert keys == ["acc", "json_parsed", "parse_failure", "predicted_answer", "score", "task_type"]
    assert all(column.shape == (3,) for column in columns.values())
    assert columns["acc"].tolist() == [True, False, False]
    assert columns["predicted_answer"].tolist() == ["C", None, None]
    assert columns["json_parsed"].tolist() == [None, False, None]
    assert columns["parse_failure"].tolist() == [None, None, True]


def test_collate_reward_extra_infos_preserves_nested_values_as_objects_on_cpu():
    keys, columns = collate_reward_extra_infos([{"diagnostics": ["a", "b"]}, {}])

    assert keys == ["diagnostics"]
    assert columns["diagnostics"].shape == (2,)
    assert columns["diagnostics"].tolist() == [["a", "b"], None]


def test_collate_reward_extra_infos_rejects_malformed_scorer_output_on_cpu():
    with pytest.raises(TypeError, match="mapping for every sample"):
        collate_reward_extra_infos([{"acc": True}, None])
