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

from types import SimpleNamespace

import pytest

from verl.utils.opd_model_compatibility import ModelPairCompatibilityError, validate_loaded_qwen3_pair


def _vocab() -> dict[str, int]:
    return {
        "token": 0,
        "<tool_call>": 1,
        "</tool_call>": 2,
        "<think>": 3,
        "</think>": 4,
        "<|im_end|>": 5,
        "<|endoftext|>": 6,
        "<|im_start|>": 7,
    }


class _Tokenizer:
    eos_token_id = 5
    pad_token_id = 6

    def __init__(self, vocab: dict[str, int] | None = None, *, probe_offset: int = 0):
        self.vocab = dict(vocab or _vocab())
        self.probe_offset = probe_offset

    def __len__(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocab)

    def apply_chat_template(self, messages, *, tools, add_generation_prompt, tokenize, enable_thinking):
        assert tools[0]["function"]["name"] == "code_interpreter"
        assert add_generation_prompt and tokenize and enable_thinking
        return [1, len(messages), len(messages[-1]["content"]), self.probe_offset]


def _config(*, context_length: int = 40960, model_type: str = "qwen3") -> SimpleNamespace:
    architecture = "Qwen3ForCausalLM" if model_type == "qwen3" else "OtherForCausalLM"
    return SimpleNamespace(
        model_type=model_type,
        architectures=[architecture],
        vocab_size=32,
        max_position_embeddings=context_length,
        _commit_hash="revision",
    )


def _validate(**overrides):
    arguments = {
        "student_config": _config(),
        "teacher_config": _config(context_length=262144),
        "student_tokenizer": _Tokenizer(),
        "teacher_tokenizer": _Tokenizer(),
        "sequence_length": 23041,
    }
    arguments.update(overrides)
    return validate_loaded_qwen3_pair(**arguments)


def test_matching_qwen3_pair_reports_token_and_template_contract_on_cpu():
    report = _validate()

    assert report["status"] == "compatible"
    assert report["sequence_length"] == 23041
    assert report["student"]["max_position_embeddings"] == 40960
    assert report["teacher"]["max_position_embeddings"] == 262144
    assert report["tokenizer"]["critical_token_ids"]["<tool_call>"] == 1
    assert report["tokenizer"]["critical_token_ids"]["eos_token_id"] == 5
    assert set(report["chat_template_probes"]["token_sha256"]) == {"initial", "after_observation"}


def test_vocab_or_special_token_mismatch_fails_on_cpu():
    changed_vocab = _vocab()
    changed_vocab["token"] = 8
    with pytest.raises(ModelPairCompatibilityError, match="Tokenizer vocabulary mismatch.*first_diff"):
        _validate(teacher_tokenizer=_Tokenizer(changed_vocab))

    teacher_tokenizer = _Tokenizer()
    teacher_tokenizer.eos_token_id = 4
    with pytest.raises(ModelPairCompatibilityError, match="eos_token_id mismatch"):
        _validate(teacher_tokenizer=teacher_tokenizer)


def test_thinking_tool_template_mismatch_fails_on_cpu():
    with pytest.raises(ModelPairCompatibilityError, match="tool chat-template mismatch"):
        _validate(teacher_tokenizer=_Tokenizer(probe_offset=1))


def test_context_budget_and_model_family_are_validated_on_cpu():
    with pytest.raises(ModelPairCompatibilityError, match="exceeds student max_position_embeddings"):
        _validate(sequence_length=40961)

    with pytest.raises(ModelPairCompatibilityError, match="student must be Qwen3ForCausalLM"):
        _validate(student_config=_config(model_type="other"))
