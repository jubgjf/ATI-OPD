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

import pytest
from omegaconf import OmegaConf

from verl.utils.dataset.rl_dataset import RLHFDataset


class _NoIOTestDataset(RLHFDataset):
    def _download(self, use_origin_parquet: bool = False):
        return None

    def _read_files_and_tokenize(self):
        return None


class _Tokenizer:
    pass


def _dataset(tmp_path, text: str) -> _NoIOTestDataset:
    prompt_path = tmp_path / "system.txt"
    prompt_path.write_text(text, encoding="utf-8")
    config = OmegaConf.create(
        {
            "system_prompt_path": str(prompt_path),
            "tool_config_path": None,
            "function_tool_path": None,
        }
    )
    return _NoIOTestDataset(["unused.parquet"], _Tokenizer(), config)


def test_system_prompt_is_prepended_without_mutating_source_on_cpu(tmp_path):
    dataset = _dataset(tmp_path, "Use tools when useful.")
    source = {"prompt": [{"role": "user", "content": "question"}]}

    messages = dataset._build_messages(source, key="prompt")

    assert messages == [
        {"role": "system", "content": "Use tools when useful."},
        {"role": "user", "content": "question"},
    ]
    assert source == {"prompt": [{"role": "user", "content": "question"}]}


def test_system_prompt_merges_with_existing_text_system_message_on_cpu(tmp_path):
    dataset = _dataset(tmp_path, "Tool policy")
    source = {
        "prompt": [
            {"role": "system", "content": "Dataset policy"},
            {"role": "user", "content": "question"},
        ]
    }

    messages = dataset._build_messages(source, key="prompt")

    assert messages[0] == {"role": "system", "content": "Tool policy\n\nDataset policy"}


def test_empty_or_missing_system_prompt_file_fails_fast_on_cpu(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("  ", encoding="utf-8")
    base = {"tool_config_path": None, "function_tool_path": None}

    try:
        _NoIOTestDataset(["unused.parquet"], _Tokenizer(), OmegaConf.create({**base, "system_prompt_path": str(empty)}))
    except ValueError as error:
        assert "is empty" in str(error)
    else:
        raise AssertionError("empty system prompt should fail")

    try:
        _NoIOTestDataset(
            ["unused.parquet"],
            _Tokenizer(),
            OmegaConf.create({**base, "system_prompt_path": str(tmp_path / "missing.txt")}),
        )
    except FileNotFoundError as error:
        assert "does not exist" in str(error)
    else:
        raise AssertionError("missing system prompt should fail")


def test_function_tool_schema_is_loaded_for_prompt_length_filter_on_cpu(tmp_path):
    tool_path = tmp_path / "tiny_tool.py"
    tool_path.write_text(
        """from verl.tools.function_tool import function_tool

SCHEMA = {
    "type": "function",
    "function": {
        "name": "dataset_length_test_tool",
        "description": "A test tool.",
        "parameters": {
            "type": "object",
            "properties": {"value": {"type": "string", "description": "value"}},
            "required": ["value"],
        },
    },
}

@function_tool("dataset_length_test_tool", schema=SCHEMA)
def dataset_length_test_tool(value: str):
    return value
""",
        encoding="utf-8",
    )
    config = OmegaConf.create(
        {
            "system_prompt_path": None,
            "tool_config_path": None,
            "function_tool_path": str(tool_path),
        }
    )

    dataset = _NoIOTestDataset(["unused.parquet"], _Tokenizer(), config)

    assert dataset.tool_schemas[0]["function"]["name"] == "dataset_length_test_tool"


def test_configured_tool_load_failure_is_not_silently_ignored_on_cpu(tmp_path):
    config = OmegaConf.create(
        {
            "system_prompt_path": None,
            "tool_config_path": None,
            "function_tool_path": str(tmp_path / "missing_tool.py"),
        }
    )

    with pytest.raises(RuntimeError, match="prompt-length filter"):
        _NoIOTestDataset(["unused.parquet"], _Tokenizer(), config)
