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

import json

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from verl.utils.tokenizer import hf_tokenizer, normalize_token_ids

UNKNOWN_BACKEND = "Tokenizer class TokenizersBackend does not exist or is not currently imported."
TEMPLATES = [
    "{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>\n' }}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}"
    "{% if enable_thinking is defined and not enable_thinking %}{{ '<think>\n\n</think>\n\n' }}{% endif %}{% endif %}",
    "{{ bos_token }}{% for m in messages %}{{ '<｜User｜>' + m['content'] }}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<｜Assistant｜><think>\n' }}{% endif %}",
]


@pytest.fixture
def v5_export(tmp_path):
    vocabulary = ["[UNK]", "[BOS]", "[EOS]", "[PAD]", "hello", "world"]
    backend = Tokenizer(models.WordLevel(dict(zip(vocabulary, range(len(vocabulary)), strict=True)), unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    backend.post_processor = processors.TemplateProcessing(single="[BOS] $A", special_tokens=[("[BOS]", 1)])
    special_tokens = ["<|im_start|>", "<|im_end|>", "<think>", "</think>", "<｜User｜>", "<｜Assistant｜>"]
    reference = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]",
        additional_special_tokens=special_tokens,
        clean_up_tokenization_spaces=False,
        padding_side="left",
        model_max_length=8192,
    )
    root = tmp_path / "v5 checkpoint"
    reference.save_pretrained(root)
    config_file = root / "tokenizer_config.json"
    metadata = json.loads(config_file.read_text())
    metadata["tokenizer_class"] = "TokenizersBackend"
    metadata["extra_special_tokens"] = special_tokens
    metadata.pop("additional_special_tokens", None)
    metadata.pop("added_tokens_decoder", None)
    config_file.write_text(json.dumps(metadata))
    (root / "special_tokens_map.json").unlink(missing_ok=True)
    return root, reference


def _unavailable_backend(*args, **kwargs):
    raise ValueError(UNKNOWN_BACKEND)


@pytest.mark.parametrize("template", TEMPLATES, ids=["qwen3", "deepseek"])
@pytest.mark.parametrize("template_in_config", [False, True])
def test_v5_backend_load_preserves_tokens_templates_and_metadata_on_cpu(
    v5_export, monkeypatch, template, template_in_config
):
    root, reference = v5_export
    reference.chat_template = template
    if template_in_config:
        metadata = json.loads((root / "tokenizer_config.json").read_text())
        metadata["chat_template"] = template
        (root / "tokenizer_config.json").write_text(json.dumps(metadata))
    else:
        (root / "chat_template.jinja").write_text(template)
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", _unavailable_backend)
    with pytest.warns(UserWarning, match="Loading local TokenizersBackend"):
        loaded = hf_tokenizer(root, local_files_only=True, trust_remote_code=False)
    assert loaded.get_vocab() == reference.get_vocab()
    assert (loaded.bos_token_id, loaded.eos_token_id, loaded.pad_token_id) == (1, 2, 3)
    assert loaded.padding_side == "left" and loaded.model_max_length == 8192
    assert loaded.chat_template == reference.chat_template
    for text in ("hello world", "[BOS] <think> hello </think> [EOS]", "你好，1+1=?"):
        for add_special_tokens in (False, True):
            ids = reference.encode(text, add_special_tokens=add_special_tokens)
            assert loaded.encode(text, add_special_tokens=add_special_tokens) == ids
            for skip_special_tokens in (False, True):
                assert loaded.decode(ids, skip_special_tokens=skip_special_tokens) == reference.decode(
                    ids, skip_special_tokens=skip_special_tokens
                )
    for thinking in (False, True):
        kwargs = dict(tokenize=True, add_generation_prompt=True, enable_thinking=thinking)
        messages = [{"role": "user", "content": "hello world"}]
        expected = normalize_token_ids(reference.apply_chat_template(messages, **kwargs))
        assert normalize_token_ids(loaded.apply_chat_template(messages, **kwargs)) == expected
        if template == TEMPLATES[1]:
            assert expected.count(reference.bos_token_id) == 1
    assert {path.name: path.read_bytes() for path in root.iterdir()} == before


def test_v5_backend_rejects_missing_serialized_tokenizer_on_cpu(v5_export, monkeypatch):
    root, _ = v5_export
    (root / "tokenizer.json").unlink()
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", _unavailable_backend)
    with pytest.raises(ValueError, match=str(root / "tokenizer.json")):
        hf_tokenizer(root, local_files_only=True)


@pytest.mark.parametrize("message", ["Tokenizer class UnknownModelTokenizer does not exist", "Invalid tokenizer.json"])
def test_compatibility_does_not_hide_unrelated_tokenizer_errors_on_cpu(v5_export, monkeypatch, message):
    root, _ = v5_export
    error = ValueError(message)

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", fail)
    with pytest.raises(ValueError) as caught:
        hf_tokenizer(root, local_files_only=True)
    assert caught.value is error


def test_v5_backend_does_not_override_explicit_slow_tokenizer_on_cpu(v5_export, monkeypatch):
    root, _ = v5_export
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", _unavailable_backend)
    with pytest.raises(ValueError, match="Tokenizer class TokenizersBackend does not exist"):
        hf_tokenizer(root, use_fast=False, local_files_only=True)
