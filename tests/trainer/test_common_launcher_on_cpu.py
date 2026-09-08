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
import subprocess
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMON_SH = REPO_ROOT / "run" / "common.sh"


def _configure_enable_thinking(value: str | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("ENABLE_THINKING", None)
    if value is not None:
        env["ENABLE_THINKING"] = value
    script = f"""
source {COMMON_SH!s}
_configure_enable_thinking || exit $?
printf '%s\n' "$enable_thinking"
"""
    return subprocess.run(
        ["bash", "-c", script],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (None, "True"),
        ("True", "True"),
        ("true", "True"),
        ("1", "True"),
        ("False", "False"),
        ("false", "False"),
        ("0", "False"),
    ],
)
def test_enable_thinking_defaults_and_environment_overrides_on_cpu(configured: str | None, expected: str):
    result = _configure_enable_thinking(configured)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected


def test_enable_thinking_rejects_invalid_environment_value_on_cpu():
    result = _configure_enable_thinking("sometimes")

    assert result.returncode == 2
    assert "ENABLE_THINKING must be True or False, got: sometimes" in result.stderr


def test_common_passes_one_configurable_enable_thinking_override_to_hydra_on_cpu():
    source = COMMON_SH.read_text(encoding="utf-8")

    assert source.count("data.apply_chat_template_kwargs.enable_thinking") == 1
    assert '"++data.apply_chat_template_kwargs.enable_thinking=${enable_thinking}"' in source
    assert '"+data.apply_chat_template_kwargs.enable_thinking=True"' not in source


@pytest.mark.parametrize("configured", [True, False])
def test_enable_thinking_force_override_composes_as_a_hydra_boolean_on_cpu(configured: bool):
    config_dir = REPO_ROOT / "verl" / "trainer" / "config"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir.resolve())):
        config = compose(
            config_name="ppo_trainer",
            overrides=[f"++data.apply_chat_template_kwargs.enable_thinking={configured}"],
        )

    assert config.data.apply_chat_template_kwargs.enable_thinking is configured
