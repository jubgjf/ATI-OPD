# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright Amazon.com and/or its affiliates
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
import os
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

import verl.workers.config.model as model_config_module
from verl.workers.config.model import HFModelConfig


class _FakeTokenizer:
    def __init__(self, chat_template=None):
        self.chat_template = chat_template
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 2

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        del messages, add_generation_prompt, tokenize
        return "<｜begin▁of▁sentence｜><｜User｜>probe<｜Assistant｜><think>\n"

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        if text == "<think>":
            return [151648]
        if text == "</think>":
            return [151649]
        return [0]


class TestHFModelConfigCPU:
    model_path = os.path.expanduser("~/models/Qwen/Qwen2.5-0.5B")  # Just a path string, not loaded

    def test_target_modules_accepts_list_via_omegaconf(self):
        """
        Test that target_modules field accepts both string and list values
        when merging OmegaConf configs (simulates CLI override behavior).

        The purpose is to ensure we can pass
        actor_rollout_ref.model.target_modules='["k_proj","o_proj","down_proj","q_proj"]'
        """

        # Create structured config from the dataclass defaults
        # This is what omega_conf_to_dataclass does internally
        cfg_from_dataclass = OmegaConf.structured(HFModelConfig)

        # Simulate CLI override with target_modules as a list
        cli_config = OmegaConf.create(
            {
                "path": self.model_path,
                "target_modules": ["k_proj", "o_proj", "q_proj", "v_proj"],
            }
        )

        # This merge should NOT raise ValidationError
        # Before the fix (target_modules: str), this would fail with:
        # "Cannot convert 'ListConfig' to string"
        merged = OmegaConf.merge(cfg_from_dataclass, cli_config)

        # Verify the list was merged correctly
        assert list(merged.target_modules) == ["k_proj", "o_proj", "q_proj", "v_proj"]

    def test_target_modules_accepts_none_via_omegaconf(self):
        """Test that target_modules still accepts None values."""

        cfg_from_dataclass = OmegaConf.structured(HFModelConfig)

        cli_config = OmegaConf.create(
            {
                "path": self.model_path,
                "target_modules": None,
            }
        )

        merged = OmegaConf.merge(cfg_from_dataclass, cli_config)
        assert merged.target_modules is None

    def test_target_modules_accepts_string_via_omegaconf(self):
        """Test that target_modules still accepts string values."""

        cfg_from_dataclass = OmegaConf.structured(HFModelConfig)

        cli_config = OmegaConf.create(
            {
                "path": self.model_path,
                "target_modules": "all-linear",
            }
        )

        merged = OmegaConf.merge(cfg_from_dataclass, cli_config)
        assert merged.target_modules == "all-linear"

    def test_target_modules_raises_on_invalid_type(self):
        """Test that __post_init__ raises TypeError for invalid target_modules types."""
        base_config = OmegaConf.structured(HFModelConfig)
        invalid_cli_config = OmegaConf.create(
            {
                "path": self.model_path,
                "target_modules": [1, 2, 3],  # list of ints instead of strings
            }
        )
        merged_config = OmegaConf.merge(base_config, invalid_cli_config)
        with pytest.raises(TypeError):
            OmegaConf.to_object(merged_config)

    def test_chat_template_from_model_sets_tokenizer_and_processor(self, monkeypatch):
        student_tokenizer = _FakeTokenizer(chat_template="student-template")
        source_tokenizer = _FakeTokenizer(chat_template="teacher-template")
        student_processor = SimpleNamespace(chat_template=None)

        def fake_hf_tokenizer(path, trust_remote_code=False):
            del trust_remote_code
            if path == "teacher":
                return source_tokenizer
            return student_tokenizer

        monkeypatch.setattr(model_config_module, "copy_to_local", lambda src, **_: src)
        monkeypatch.setattr(model_config_module, "hf_tokenizer", fake_hf_tokenizer)
        monkeypatch.setattr(model_config_module, "hf_processor", lambda path, **_: student_processor)
        monkeypatch.setattr(model_config_module, "get_generation_config", lambda *_, **__: SimpleNamespace())
        monkeypatch.setattr(model_config_module, "update_model_config", lambda *_, **__: None)
        monkeypatch.setattr(
            model_config_module.AutoConfig,
            "from_pretrained",
            lambda *_, **__: SimpleNamespace(architectures=["FakeModel"], tie_word_embeddings=False),
        )

        config = HFModelConfig(path="student", chat_template_from_model="teacher")

        assert config.tokenizer.chat_template == "teacher-template"
        assert config.processor.chat_template == "teacher-template"

    def test_chat_template_from_model_conflicts_with_custom_template(self, monkeypatch):
        student_tokenizer = _FakeTokenizer(chat_template="student-template")
        student_processor = SimpleNamespace(chat_template=None)

        monkeypatch.setattr(model_config_module, "copy_to_local", lambda src, **_: src)
        monkeypatch.setattr(model_config_module, "hf_tokenizer", lambda *_, **__: student_tokenizer)
        monkeypatch.setattr(model_config_module, "hf_processor", lambda *_, **__: student_processor)

        with pytest.raises(ValueError, match="Only one of custom_chat_template and chat_template_from_model"):
            HFModelConfig(path="student", custom_chat_template="custom", chat_template_from_model="teacher")
