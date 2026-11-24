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
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from verl.trainer.distillation.config_validation import validate_distillation_intervention_training_config
from verl.utils.config import omega_conf_to_dataclass

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "run"
LAUNCHERS = {
    "opd": "vanilla-opd.sh",
    "ati": "dmpd--train_on_teacher--loss_opd--prune_future.sh",
    "relay": "relay-opd.sh",
    "selection": "dmpd--prune_future--selection_only.sh",
}


def _run_submitted_method(tmp_path, method, seed, *, rank=0, method_env=None, train_config_extra=""):
    algorithm = tmp_path / "algorithm"
    wrapper = algorithm / "DMPD" / "run" / LAUNCHERS[method]
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text(f'#!/bin/bash\nexec bash {shlex.quote(str(RUN / wrapper.name))} "$@"\n')
    config = algorithm / "train.config"
    config.write_text(
        f"DATA_SHUFFLE=True\nDATA_SEED={seed}\n"
        "TOTAL_EPOCHS=1\nTOTAL_TRAINING_STEPS=217\nSAVE_FREQ=10\n"
        "OPD_DIAGNOSTICS_COLLECT_ROLLOUT_COSTS=True\nOPD_DIAGNOSTICS_COLLECT_SOURCE_POSITION=True\n"
        "EXPERIMENT_TIMESTAMP=20261109-120000\n"
        f"WORKSPACE={shlex.quote(str(tmp_path))}\nTMPDIR={shlex.quote(str(tmp_path / 'ray'))}\n"
        "CONFIG_LOADS=$(( ${CONFIG_LOADS:-0} + 1 ))\n"
        f"{train_config_extra}"
    )
    # Run the real launcher through Ray submission with all hardware and
    # service commands stubbed. Shared-storage writes stay inside tmp_path.
    bootstrap = tmp_path / "bootstrap.sh"
    bootstrap.write_text(
        """
.() {
    case "$1" in
        /usr/local/Ascend/*|*/anaconda3/etc/profile.d/conda.sh) return 0 ;;
        *) builtin . "$@" ;;
    esac
}
conda() { unset LD_PRELOAD; }
npu-smi() { :; }
python() { :; }
ls() { :; }
python3() {
    if [[ "$1" == "-m" && "$2" == "pip" ]]; then
        shift 2
        printf 'PIP_COMMAND=%s\\n' "$*"
        return 0
    fi
    if [[ "$1" == "-u" && "${2##*/}" == "opd_ray.py" ]]; then
        printf 'RAY_ROLE=%s\\n' "$3"
        [[ "$3" == "head" ]] || return 0
        shift 3
        printf 'CONFIG_LOADS=%s\\n' "$CONFIG_LOADS"
        bash -c 'for key in TOTAL_TRAINING_STEPS "$@"; do
            printf "DRIVER_ENV=%s=%s\\n" "$key" "${!key}"
        done' bash OPD_DIAGNOSTICS_COLLECT_ROLLOUT_COSTS OPD_DIAGNOSTICS_COLLECT_SOURCE_POSITION
        while [[ "$#" -gt 0 && "$1" != "--" ]]; do
            case "$1" in
                --checkpoint-root) shift ;;
                --train-runtime-env) printf 'RUNTIME_ENV=%s\\n' "$2"; shift ;;
                --eval-env) printf 'POST_TRAIN_EVAL=enabled\\n' ;;
                *) printf 'EVAL_ARG=%s\\n' "$1" ;;
            esac
            shift
        done
        shift
        printf 'HYDRA_ARG=%s\\n' "$@"
        return 0
    fi
    command python3 "$@"
}
sleep() { :; }
pkill() { :; }
rm() { :; }
df() { :; }
ifconfig() { printf 'inet 127.0.0.1\n'; }
ray() { [[ "$1" == "stop" ]]; }
"""
    )
    return subprocess.run(
        ["bash", str(wrapper)],
        env={
            "PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin",
            "TRAIN_CONFIG_PATH": str(config),
            "BASH_ENV": str(bootstrap),
            "DATA_SEED": "99",
            "VC_WORKER_NUM": "4",
            "VC_TASK_INDEX": str(rank),
            "MA_CURRENT_IP": "127.0.0.1",
            "MA_VJ_NAME": "opd",
            "MA_TASK_NAME": "train",
            **(method_env or {}),
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )


@pytest.mark.parametrize(
    "method,seed", [("opd", 42), ("opd", 0), ("ati", 42), ("ati", 0), ("relay", 42), ("selection", 42)]
)
@pytest.mark.parametrize("rank_metrics", [False, True])
def test_opd_launcher_config_composes_without_accelerators(tmp_path, monkeypatch, method, seed, rank_metrics):
    result = _run_submitted_method(
        tmp_path, method, seed, train_config_extra=f"OPD_DIAGNOSTICS_COLLECT_RANK_METRICS={rank_metrics}\n"
    )
    assert result.returncode == 0, result.stderr
    assert "CONFIG_LOADS=1" in result.stdout
    assert [
        line.removeprefix("PIP_COMMAND=") for line in result.stdout.splitlines() if line.startswith("PIP_COMMAND=")
    ] == [
        "config set global.index-url http://repo.myhuaweicloud.com/repository/pypi/simple/",
        "config set global.trusted-host repo.myhuaweicloud.com",
        "config set global.timeout 120",
        "install swanlab==0.10.3",
        "install -e . matplotlib",
    ]
    overrides = [
        line.removeprefix("HYDRA_ARG=") for line in result.stdout.splitlines() if line.startswith("HYDRA_ARG=")
    ]
    assert overrides
    driver_env = dict(
        line.removeprefix("DRIVER_ENV=").split("=", 1)
        for line in result.stdout.splitlines()
        if line.startswith("DRIVER_ENV=")
    )
    assert driver_env == {
        "TOTAL_TRAINING_STEPS": "217",
        "OPD_DIAGNOSTICS_COLLECT_ROLLOUT_COSTS": "True",
        "OPD_DIAGNOSTICS_COLLECT_SOURCE_POSITION": "True",
    }
    for key, value in driver_env.items():
        monkeypatch.setenv(key, value)
    with initialize_config_dir(config_dir=str(ROOT / "verl" / "trainer" / "config"), version_base=None):
        config = compose(config_name="ppo_trainer", overrides=overrides)
    validate_distillation_intervention_training_config(config)
    distillation = omega_conf_to_dataclass(config.distillation)
    runtime = next(
        line.removeprefix("RUNTIME_ENV=") for line in result.stdout.splitlines() if line.startswith("RUNTIME_ENV=")
    )
    env_vars = json.loads(runtime)["env_vars"]
    assert env_vars["SWANLAB_MODE"] == "offline"
    assert env_vars["SWANLAB_LOG_DIR"] == f"{config.trainer.default_local_dir}/swanlog"
    assert not any(key.startswith("WANDB_") for key in env_vars)
    assert list(config.trainer.logger) == ["console", "swanlab"]
    assert config.trainer.project_name == "ATI-OPD"
    assert distillation.fuse_final_teacher_scoring is (method != "relay")
    assert distillation.diagnostics.archive_heatmaps
    assert config.data.shuffle and config.data.seed == seed
    assert config.trainer.total_training_steps == 217
    assert config.trainer.save_freq == 10
    assert config.trainer.nnodes == distillation.nnodes == 2
    (teacher,) = distillation.teacher_models.values()
    assert teacher.num_replicas == 8
    eval_env = dict(
        line.removeprefix("EVAL_ARG=").split("=", 1)
        for line in result.stdout.splitlines()
        if line.startswith("EVAL_ARG=")
    )
    assert eval_env["CHAT_TEMPLATE_MODEL"] == config.actor_rollout_ref.model.chat_template_from_model
    assert eval_env["ENABLE_THINKING"] == "True"
    assert eval_env["STUDENT_TP"] == "2"
    assert eval_env["TEST_BENCHMARK_N"] == "16"
    assert config.trainer.experiment_name == f"{Path(LAUNCHERS[method]).stem}-20261109-120000"
    assert distillation.diagnostics.collect_rollout_costs
    assert distillation.diagnostics.collect_rank_metrics is rank_metrics
    if method == "ati":
        assert distillation.intervention.prune_mode == "adaptive_future"
        assert distillation.diagnostics.outcome_case_samples_per_transition == 8
    if method == "relay":
        assert distillation.intervention.strategy == "relay"
        assert distillation.use_actor_loss and not distillation.diagnostics.collect_source_position_metrics
    if method == "selection":
        assert distillation.intervention.prune_mode == "selection_only"
        assert distillation.intervention.prune and not distillation.intervention.enabled


@pytest.mark.parametrize(
    "method,setting,override",
    [
        ("ati", "TEACHER_INTERVENTION_GAP_THRESHOLD=0.23", "distillation.intervention.gap_threshold=0.23"),
        ("opd", "FUSE_FINAL_TEACHER_SCORING=False", "distillation.fuse_final_teacher_scoring=False"),
        ("relay", "TEACHER_INTERVENTION_MAX_COUNT=1", "distillation.intervention.max_interventions_per_sample=1"),
        (
            "selection",
            "TEACHER_INTERVENTION_PRUNE_FUTURE_WINDOW_TOKENS=80",
            "distillation.intervention.prune_future_window_tokens=80",
        ),
    ],
)
def test_method_accepts_inherited_environment_overrides(tmp_path, method, setting, override):
    result = _run_submitted_method(tmp_path, method, 42, method_env=dict([setting.split("=", 1)]))
    assert result.returncode == 0, result.stderr
    assert f"HYDRA_ARG={override}" in result.stdout.splitlines()


@pytest.mark.parametrize("method", ["opd", "ati"])
def test_qwen3_launcher_composes_fused_teacher_scores_and_shared_stops(tmp_path, method):
    result = _run_submitted_method(
        tmp_path,
        method,
        42,
        train_config_extra=(
            'TEACHER_MODEL="$DATASET_DIR/models/Qwen/Qwen3-1.7B"\n'
            'STUDENT_MODEL="$DATASET_DIR/models/Qwen/Qwen3-1.7B-Base"\n'
            "ROLLOUT_STOP_TOKEN_IDS='[151643,151645]'\n"
        ),
    )
    assert result.returncode == 0, result.stderr
    overrides = [
        line.removeprefix("HYDRA_ARG=") for line in result.stdout.splitlines() if line.startswith("HYDRA_ARG=")
    ]
    with initialize_config_dir(config_dir=str(ROOT / "verl" / "trainer" / "config"), version_base=None):
        config = compose(config_name="ppo_trainer", overrides=overrides)
    validate_distillation_intervention_training_config(config)
    distillation = omega_conf_to_dataclass(config.distillation)
    assert distillation.fuse_final_teacher_scoring
    assert distillation.distillation_loss.topk == 16
    (teacher,) = distillation.teacher_models.values()
    assert teacher.inference.engine_kwargs["vllm"]["max_logprobs"] == 32
    assert config.actor_rollout_ref.model.chat_template_from_model == teacher.model_path
    assert teacher.model_path == "/opt/huawei/dataset/jnguan/models/Qwen/Qwen3-1.7B"
    assert config.actor_rollout_ref.model.path == "/opt/huawei/dataset/jnguan/models/Qwen/Qwen3-1.7B-Base"
    assert config.data.apply_chat_template_kwargs.enable_thinking is True
    assert list(config.actor_rollout_ref.rollout.stop_token_ids) == [151643, 151645]


def test_method_launches_as_ray_worker_on_nonzero_platform_rank(tmp_path):
    result = _run_submitted_method(tmp_path, "relay", 42, rank=1)
    assert result.returncode == 0, result.stderr
    assert "RAY_ROLE=worker" in result.stdout
    assert "HYDRA_ARG=" not in result.stdout


@pytest.mark.parametrize(
    "nodes,overrides,expected",
    [
        (2, {}, (1, 1)),
        (8, {}, (4, 4)),
        (4, {"STUDENT_NNODES": "3"}, (3, 1)),
        (4, {"TEACHER_NNODES": "3"}, (1, 3)),
        (4, {"STUDENT_NNODES": "1", "TEACHER_NNODES": "3"}, (1, 3)),
    ],
)
def test_opd_node_split_defaults_and_overrides(tmp_path, nodes, overrides, expected):
    result = _run_submitted_method(tmp_path, "opd", 42, method_env={"NNODES": str(nodes), **overrides})
    assert result.returncode == 0, result.stderr
    students, teachers = expected
    assert f"HYDRA_ARG=trainer.nnodes={students}\n" in result.stdout
    assert f"HYDRA_ARG=distillation.nnodes={teachers}\n" in result.stdout


@pytest.mark.parametrize(
    "env,message",
    [
        ({"VC_WORKER_NUM": "3", "STUDENT_NNODES": "2", "TEACHER_NNODES": "1"}, "even NNODES"),
        ({"NNODES": "0"}, "even NNODES"),
        ({"STUDENT_NNODES": "4"}, "must be positive and sum"),
        ({"STUDENT_NNODES": "2", "TEACHER_NNODES": "1"}, "must be positive and sum"),
        ({"POST_TRAIN_EVAL": "also"}, "POST_TRAIN_EVAL must be True or False"),
    ],
)
def test_invalid_opd_topology_or_switch_fails_before_setup(tmp_path, env, message):
    result = _run_submitted_method(tmp_path, "opd", 42, method_env=env)
    assert result.returncode == 2
    assert message in result.stderr
    assert "PIP_COMMAND=" not in result.stdout
    assert "RAY_ROLE=" not in result.stdout


def test_post_train_evaluation_can_be_disabled_for_checkpoint_free_runs(tmp_path):
    result = _run_submitted_method(
        tmp_path, "opd", 42, train_config_extra="SAVE_FREQ=-1\nPOST_TRAIN_EVAL=False\nSTUDENT_TP=3\n"
    )
    assert result.returncode == 0, result.stderr
    assert "HYDRA_ARG=trainer.save_freq=-1" in result.stdout
    assert "EVAL_ARG=" not in result.stdout
