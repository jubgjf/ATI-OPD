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

from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMON_SH = REPO_ROOT / "run" / "common.sh"
SOD_LAUNCHER = REPO_ROOT / "run" / "vanilla-opd--tool_sod.sh"
LEGACY_TOOL_LAUNCHER = REPO_ROOT / "run" / "vanilla-opd--tool.sh"
DMPD_LAUNCHER = REPO_ROOT / "run" / "dmpd--skip_correct--train_on_teacher--loss_opd--tool_sod.sh"
DMPD_PRUNE_LAUNCHER = REPO_ROOT / "run" / "dmpd--skip_correct--train_on_teacher--loss_opd--prune--tool_sod.sh"
DATA_CONFIG = REPO_ROOT / "verl" / "trainer" / "config" / "data" / "legacy_data.yaml"


def test_sod_and_existing_tool_launchers_pin_distinct_styles_on_cpu():
    sod_source = SOD_LAUNCHER.read_text(encoding="utf-8")
    legacy_source = LEGACY_TOOL_LAUNCHER.read_text(encoding="utf-8")

    assert "DEFAULT_EXPERIMENT_NAME=${EXPERIMENT_NAME:-vanilla-opd--tool_sod}" in sod_source
    assert "ENABLE_TOOL_CALL=True" in sod_source
    assert "SOD_STYLE=True" in sod_source
    assert "SOD_STYLE" not in legacy_source
    for excluded_algorithm_change in ("USE_TASK_REWARDS", "ROLLOUT_N", "PPO_MINI_BATCH_SIZE", "adv_estimator"):
        assert excluded_algorithm_change not in sod_source


def test_sod_dmpd_launchers_reuse_opd_and_prune_method_semantics_on_cpu():
    dmpd_source = DMPD_LAUNCHER.read_text(encoding="utf-8")
    prune_source = DMPD_PRUNE_LAUNCHER.read_text(encoding="utf-8")

    for source in (dmpd_source, prune_source):
        assert "ENABLE_TOOL_CALL=True" in source
        assert "SOD_STYLE=True" in source
        assert "TEACHER_INTERVENTION_SKIP_CORRECT=${TEACHER_INTERVENTION_SKIP_CORRECT:-True}" in source
        assert "teacher_intervention_train_teacher=${TEACHER_INTERVENTION_TRAIN_TEACHER:-True}" in source
        assert "teacher_intervention_loss_mode=${TEACHER_INTERVENTION_LOSS_MODE:-opd}" in source
        assert "distillation.intervention.enabled=True" in source
        assert "distillation.intervention.train_on_teacher_segments=${teacher_intervention_train_teacher}" in source
        assert "distillation.intervention.teacher_segment_loss_mode=${teacher_intervention_loss_mode}" in source

    assert "dmpd--skip_correct--train_on_teacher--loss_opd--tool_sod" in dmpd_source
    assert "teacher_intervention_max_count=${TEACHER_INTERVENTION_MAX_COUNT:-4}" in dmpd_source
    assert "TEACHER_INTERVENTION_PRUNE" not in dmpd_source

    assert "dmpd--skip_correct--train_on_teacher--loss_opd--prune--tool_sod" in prune_source
    assert "teacher_intervention_max_count=${TEACHER_INTERVENTION_MAX_COUNT:-1}" in prune_source
    assert "TEACHER_INTERVENTION_PRUNE=${TEACHER_INTERVENTION_PRUNE:-True}" in prune_source


def test_common_contains_sod_defaults_and_preserves_existing_tool_defaults_on_cpu():
    source = COMMON_SH.read_text(encoding="utf-8")

    for sod_default in (
        "TEACHER_MODEL=${TEACHER_MODEL:-$model_root/youngzhong/SOD-GRPO_teacher-4B}",
        "tool_user_prompt_suffix_path=${TOOL_USER_PROMPT_SUFFIX_PATH:-$WORKSPACE/run/prompts/qwen3_math_tool_sod.txt}",
        "function_tool_path=${FUNCTION_TOOL_PATH:-$WORKSPACE/verl/tools/sod_code_interpreter_tool.py}",
        "tool_max_assistant_turns=${TOOL_MAX_ASSISTANT_TURNS:-16}",
        "tool_max_user_turns=${TOOL_MAX_USER_TURNS:-16}",
        "tool_max_response_chars=${TOOL_MAX_RESPONSE_CHARS:-256}",
        "max_prompt_length=${MAX_PROMPT_LENGTH:-2560}",
        "max_response_length=${MAX_RESPONSE_LENGTH:-20480}",
        "export PYTHON_SANDBOX_TIMEOUT=${PYTHON_SANDBOX_TIMEOUT:-60}",
        '"data.user_prompt_suffix_path=${tool_user_prompt_suffix_path}"',
        "python -m verl.utils.opd_model_compatibility",
    ):
        assert sod_default in source

    for existing_default in (
        "TEACHER_MODEL=${TEACHER_MODEL:-$model_root/Qwen/Qwen3-8B}",
        "tool_system_prompt_path=${TOOL_SYSTEM_PROMPT_PATH:-$WORKSPACE/run/prompts/qwen3_math_tool_optional.txt}",
        "function_tool_path=${FUNCTION_TOOL_PATH:-$WORKSPACE/verl/tools/code_interpreter_tool.py}",
        "tool_max_assistant_turns=${TOOL_MAX_ASSISTANT_TURNS:-null}",
        "tool_max_user_turns=${TOOL_MAX_USER_TURNS:-null}",
        "tool_max_response_chars=${TOOL_MAX_RESPONSE_CHARS:-4096}",
        "export PYTHON_SANDBOX_TIMEOUT=${PYTHON_SANDBOX_TIMEOUT:-10}",
    ):
        assert existing_default in source


def test_data_config_exposes_user_prompt_suffix_without_changing_system_prompt_default_on_cpu():
    config = OmegaConf.load(DATA_CONFIG)

    assert config.system_prompt_path is None
    assert config.user_prompt_suffix_path is None
