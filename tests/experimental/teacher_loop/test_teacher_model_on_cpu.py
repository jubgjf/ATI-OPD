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

from verl.experimental.teacher_loop.teacher_model import TeacherModelManager
from verl.workers.config.distillation import (
    RETHINKING_TARGET_PROMPT_LOGPROBS_CONFIG_KEY,
)


def _disable_teacher_server_initialization(monkeypatch):
    monkeypatch.setattr(TeacherModelManager, "_initialize_llm_servers", lambda self: None)
    monkeypatch.setattr(TeacherModelManager, "_initialize_load_balancer_handle", lambda self: None)


def test_rethinking_style_enables_target_patch_only_on_its_teacher_on_cpu(monkeypatch):
    _disable_teacher_server_initialization(monkeypatch)
    inference = SimpleNamespace(
        name="vllm",
        custom=None,
        enable_prefix_caching=True,
        engine_kwargs={"vllm": {"enable_prefix_caching": True, "max_num_seqs": 8}},
    )
    teacher_config = SimpleNamespace(key="teacher", inference=inference)
    distillation_config = SimpleNamespace(distillation_loss=SimpleNamespace(top_k_strategy="only_stu_rethinking_style"))

    TeacherModelManager(distillation_config, teacher_config, resource_pool=object())

    assert inference.custom == {RETHINKING_TARGET_PROMPT_LOGPROBS_CONFIG_KEY: True}
    assert inference.enable_prefix_caching is False
    assert inference.engine_kwargs == {"vllm": {"enable_prefix_caching": False, "max_num_seqs": 8}}


@pytest.mark.parametrize("top_k_strategy", ["only_stu", "only_stu_teacher_topk"])
def test_ordinary_teacher_support_strategies_do_not_mutate_target_patch_config_on_cpu(monkeypatch, top_k_strategy):
    _disable_teacher_server_initialization(monkeypatch)
    inference = SimpleNamespace(
        name="vllm",
        custom=None,
        enable_prefix_caching=True,
        engine_kwargs={"vllm": {"enable_prefix_caching": True}},
    )
    teacher_config = SimpleNamespace(key="teacher", inference=inference)
    distillation_config = SimpleNamespace(distillation_loss=SimpleNamespace(top_k_strategy=top_k_strategy))

    TeacherModelManager(distillation_config, teacher_config, resource_pool=object())

    assert inference.custom is None
    assert inference.enable_prefix_caching is True
    assert inference.engine_kwargs == {"vllm": {"enable_prefix_caching": True}}
