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

import os

import pytest

from verl.utils.tokenizer import normalize_token_ids

transformers = pytest.importorskip("transformers")


def _model_path(env_key: str, fallback_key: str) -> str:
    return os.environ.get(env_key) or os.environ.get(fallback_key) or ""


@pytest.mark.skipif(
    not _model_path("VERL_OPD_STUDENT_MODEL", "STUDENT_MODEL")
    or not _model_path("VERL_OPD_TEACHER_MODEL", "TEACHER_MODEL"),
    reason="Set STUDENT_MODEL/TEACHER_MODEL or VERL_OPD_*_MODEL to run DeepSeek/Skywork tokenizer checks.",
)
def test_deepseek_skywork_tokenizers_share_thinking_template_contract_on_cpu():
    student_path = _model_path("VERL_OPD_STUDENT_MODEL", "STUDENT_MODEL")
    teacher_path = _model_path("VERL_OPD_TEACHER_MODEL", "TEACHER_MODEL")
    student_tokenizer = transformers.AutoTokenizer.from_pretrained(student_path, trust_remote_code=True)
    teacher_tokenizer = transformers.AutoTokenizer.from_pretrained(teacher_path, trust_remote_code=True)
    student_config = transformers.AutoConfig.from_pretrained(student_path, trust_remote_code=True)
    teacher_config = transformers.AutoConfig.from_pretrained(teacher_path, trust_remote_code=True)

    for tokenizer, config in ((student_tokenizer, student_config), (teacher_tokenizer, teacher_config)):
        assert config.vocab_size >= len(tokenizer)
        assert tokenizer.encode("<think>", add_special_tokens=False)
        assert tokenizer.encode("</think>", add_special_tokens=False)

    prompt = [{"role": "user", "content": "Solve 1+1. Put the final answer in \\\\boxed{}."}]
    teacher_rendered_prompt = teacher_tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        tokenize=False,
    )
    assert "<think>" in teacher_rendered_prompt
    assert "<|im_start|>" not in teacher_rendered_prompt

    student_default_rendered_prompt = student_tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        tokenize=False,
    )
    assert "<|im_start|>" not in student_default_rendered_prompt

    student_tokenizer.chat_template = teacher_tokenizer.chat_template
    student_rendered_prompt = student_tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        tokenize=False,
    )
    assert "<think>" in student_rendered_prompt
    assert student_rendered_prompt == teacher_rendered_prompt

    for tokenizer, rendered_prompt in (
        (student_tokenizer, student_rendered_prompt),
        (teacher_tokenizer, teacher_rendered_prompt),
    ):
        prompt_ids = tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            tokenize=True,
        )
        manual_prefix_ids = normalize_token_ids(prompt_ids) + tokenizer.encode(
            "We need compute directly.", add_special_tokens=False
        )
        decoded_manual_prefix = tokenizer.decode(manual_prefix_ids, skip_special_tokens=False)
        assert decoded_manual_prefix.count("<think>") == rendered_prompt.count("<think>")

    assert student_tokenizer.encode("<think>", add_special_tokens=False) == teacher_tokenizer.encode(
        "<think>", add_special_tokens=False
    )
    assert student_tokenizer.encode("</think>", add_special_tokens=False) == teacher_tokenizer.encode(
        "</think>", add_special_tokens=False
    )
