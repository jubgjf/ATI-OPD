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

import os
import subprocess
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMON_SH = REPO_ROOT / "run" / "common.sh"
DEBUG_SH = REPO_ROOT / "run" / "debug.sh"
VANILLA_SH = REPO_ROOT / "run" / "vanilla-opd--nemotron_cascade.sh"
DMPD_SH = REPO_ROOT / "run" / "dmpd--skip_correct--train_on_teacher--loss_opd--nemotron_cascade.sh"
NEMOTRON_BENCHMARK_CONFIG = REPO_ROOT / "verl" / "trainer" / "config" / "test_benchmarks" / "nemotron_cascade.yaml"


def test_nemotron_benchmark_suite_has_the_pinned_budgets_and_9507_generations_on_cpu():
    with initialize_config_dir(
        version_base=None, config_dir=str((REPO_ROOT / "verl" / "trainer" / "config").resolve())
    ):
        config = compose(
            config_name="ppo_trainer",
            overrides=["+test_benchmarks@data.test_benchmarks=nemotron_cascade"],
        )
    entries = OmegaConf.to_container(config.data.test_benchmarks.entries, resolve=True)
    by_name = {entry["name"]: entry for entry in entries}

    assert {
        name: (entry["prompt_max_length"], entry["generation_max_tokens"], entry["profiles"]["train_sampling"]["n"])
        for name, entry in by_name.items()
    } == {
        "aime25": (1024, 23552, 16),
        "ifbench": (4096, 4096, 8),
        "mmlu_pro": (24576, 4096, 1),
        "gpqa_diamond": (4096, 8192, 8),
        "longbench_v2": (31744, 1024, 4),
    }
    sample_counts = {"aime25": 30, "ifbench": 300, "mmlu_pro": 1000, "gpqa_diamond": 198, "longbench_v2": 503}
    generation_count = sum(
        sample_counts[name] * (entry["profiles"]["legacy"]["n"] + entry["profiles"]["train_sampling"]["n"])
        for name, entry in by_name.items()
    )
    assert generation_count == 9507


def test_nemotron_launchers_only_select_dataset_and_method_semantics_on_cpu():
    vanilla = VANILLA_SH.read_text(encoding="utf-8")
    dmpd = DMPD_SH.read_text(encoding="utf-8")
    for source in (vanilla, dmpd):
        assert "TRAIN_DATASET=${TRAIN_DATASET:-nemotron_cascade_2_rl_mopd}" in source
        for common_setting in (
            "DATASET_DIR=",
            "STUDENT_MODEL=",
            "TEACHER_MODEL=",
            "TEST_BENCHMARK_ROOT=",
            "AIME25_BENCHMARK_PATH=",
            "IFBENCH_BENCHMARK_PATH=",
            "MAX_PROMPT_LENGTH=",
            "MAX_RESPONSE_LENGTH=",
            "VALIDATION_REWARD_DATA_SOURCE=",
            "TEST_BENCHMARKS_FREQ=",
        ):
            assert common_setting not in source
    assert "TEACHER_INTERVENTION_SKIP_CORRECT=${TEACHER_INTERVENTION_SKIP_CORRECT:-True}" in dmpd
    assert "distillation.intervention.train_on_teacher_segments=${teacher_intervention_train_teacher}" in dmpd
    assert "distillation.intervention.teacher_segment_loss_mode=${teacher_intervention_loss_mode}" in dmpd


def test_common_owns_nemotron_defaults_while_yaml_owns_the_benchmark_suite_on_cpu():
    common = COMMON_SH.read_text(encoding="utf-8")
    for expected in (
        "HOME=/home/ma-user",
        "DATASET_DIR=/opt/huawei/dataset/jnguan",
        "WORKSPACE=$DATASET_DIR/code/DMPD",
        "STUDENT_MODEL=${STUDENT_MODEL:-$model_root/Qwen/Qwen3-1.7B}",
        "TEACHER_MODEL=${TEACHER_MODEL:-$model_root/Qwen/Qwen3-8B}",
        "max_prompt_length=${MAX_PROMPT_LENGTH:-$((1 * 1024))}",
        "max_response_length=${MAX_RESPONSE_LENGTH:-$((23 * 1024))}",
        "default_rollout_max_model_len=32768",
        "validation_reward_data_source=${VALIDATION_REWARD_DATA_SOURCE:-nemotron_cascade_mopd}",
        "default_test_benchmark_suite=nemotron_cascade",
    ):
        assert expected in common
    assert "_nemotron_cascade_benchmark_entries" not in common
    assert "_require_nemotron_asset" not in common
    assert "local npu_user_home=" not in common
    assert "IN_TRAIN_EVALUATION=${IN_TRAIN_EVALUATION:-True}" not in common

    suite = OmegaConf.load(NEMOTRON_BENCHMARK_CONFIG)
    assert [entry.path for entry in suite.entries] == [
        "math-ai/aime25",
        "allenai/IFBench_test",
        "TIGER-Lab/MMLU-Pro",
        "Idavidrein/gpqa",
        "THUDM/LongBench-v2",
    ]


def test_gpu_debug_dry_run_preserves_nemotron_semantic_configuration_on_cpu():
    result = subprocess.run(
        ["bash", str(DEBUG_SH)],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "DRY_RUN": "1",
            "TRAIN_DATASET": "nemotron_cascade_2_rl_mopd",
            "DEBUG_MODE": "dmpd",
        },
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    output = result.stdout
    for expected in (
        "max lengths:  prompt 1024, response 23552, rollout model 32768, teacher model 24577",
        "+test_benchmarks@data.test_benchmarks=nemotron_cascade",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        "distillation.intervention.skip_correct=True",
        "distillation.intervention.train_on_teacher_segments=True",
        "distillation.intervention.teacher_segment_loss_mode=opd",
        "distillation.diagnostics.collect_intervention_outcome_transitions=True",
    ):
        assert expected in output
    assert "data.apply_chat_template_kwargs.enable_thinking" not in output
