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
import shlex
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "run/eval-single.sh"


@pytest.fixture
def launcher_env(tmp_path):
    env = os.environ.copy()
    for key in (
        "NODE_RANK",
        "VC_TASK_INDEX",
        "MODEL_PATH",
        "TEST_BENCHMARK_ROOT",
        "EVAL_ALL",
        "LD_PRELOAD",
        "BASH_ENV",
    ):
        env.pop(key, None)
    env.update(TRAIN_CONFIG_PATH=str(tmp_path / "missing.config"), WORKSPACE=str(tmp_path), ULIMIT_NOFILE="1024")
    return env


def _run(env, cwd=None):
    return subprocess.run(
        ["bash", str(LAUNCHER)], env=env, cwd=cwd, capture_output=True, text=True, check=False, timeout=10
    )


@pytest.mark.parametrize("rank_key", ["NODE_RANK", "VC_TASK_INDEX"])
def test_nonprimary_nodes_exit_before_loading_model_or_npu_on_cpu(launcher_env, rank_key, tmp_path):
    launcher_env[rank_key] = "1"
    result = _run(launcher_env)
    assert result.returncode == 0, result.stderr
    assert "Skipping standalone evaluation" in result.stdout
    assert list(tmp_path.iterdir()) == []


def test_missing_model_fails_before_npu_initialization_on_cpu(launcher_env):
    result = _run(launcher_env)
    assert result.returncode == 2
    assert "MODEL_PATH" in result.stderr


def test_failed_external_configuration_stops_launch_on_cpu(launcher_env, tmp_path):
    config = tmp_path / "train.config"
    config.write_text("return 7\n")
    launcher_env["TRAIN_CONFIG_PATH"] = str(config)
    result = _run(launcher_env)
    assert result.returncode == 1
    assert "Failed to load external train config" in result.stderr


@pytest.mark.parametrize("override", [False, True])
def test_launcher_dataset_root_before_npu_setup_on_cpu(launcher_env, tmp_path, override):
    import pandas as pd

    from verl.utils.evaluation.math_benchmark import load_benchmark_samples
    from verl.utils.evaluation.standalone_math import EvaluationConfig, resolve_benchmark_dirs

    dataset_dir = tmp_path / "server mount"
    expected_root = tmp_path / "custom datasets" if override else dataset_dir / "datasets"
    for benchmark in ("hmmt_feb_2026", "hmmt_nov_2025"):
        benchmark_root = expected_root if override else expected_root / "MathArena"
        data_dir = benchmark_root / benchmark / "data"
        data_dir.mkdir(parents=True)
        pd.DataFrame([{"problem_idx": 1, "problem": benchmark, "answer": "4"}]).to_parquet(
            data_dir / "train-00000-of-00001.parquet"
        )
    model = tmp_path / "model"
    model.mkdir()
    config = tmp_path / "train.config"
    config.write_text(f"DATASET_DIR={shlex.quote(str(dataset_dir))}\nMODEL_PATH={shlex.quote(str(model))}\n")
    # Exercise the real launcher through path resolution, then stop before any
    # platform initialization, environment installation or model inference.
    bootstrap = tmp_path / "bootstrap.sh"
    bootstrap.write_text("npu-smi() { printf 'RESOLVED_ROOT=%s\\n' \"$TEST_BENCHMARK_ROOT\"; exit 0; }\n")
    launcher_env.update(TRAIN_CONFIG_PATH=str(config), BASH_ENV=str(bootstrap))
    if override:
        launcher_env["TEST_BENCHMARK_ROOT"] = str(expected_root)
    result = _run(launcher_env)
    assert result.returncode == 0, result.stderr
    root_line = next(line for line in result.stdout.splitlines() if line.startswith("RESOLVED_ROOT="))
    resolved_root = root_line.split("=", 1)[1]
    assert resolved_root == str(expected_root)
    benchmark_dirs = resolve_benchmark_dirs(EvaluationConfig(str(model), resolved_root))
    for benchmark, directory in benchmark_dirs.items():
        samples = load_benchmark_samples(directory.parent, benchmark)
        assert len(samples) == 1
        assert samples[0].benchmark == benchmark
        assert samples[0].problem == benchmark


@pytest.mark.parametrize("engine_exit_code", [0, 37])
@pytest.mark.parametrize("eval_all_source", ["default", "environment", "external_config"])
def test_launcher_resolves_relative_model_before_cd_and_propagates_status_on_cpu(
    launcher_env, tmp_path, engine_exit_code, eval_all_source
):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    scripts = {
        "npu-smi": "#!/bin/bash\nprintf 'NPU:%s\\n' \"$*\"\n",
        "python": (
            "#!/bin/bash\nprintf 'PYTHON:%s\\n' \"$*\"\n"
            "if [[ $* == *standalone_math* ]]; then\n"
            "  printf 'MODEL_PATH=%s\\nROOT=%s\\nNO_PYC=%s\\nOFFLINE=%s\\nFORCE_MERGE=%s\\nWORKDIR=%s\\n' "
            '"$MODEL_PATH" "$TEST_BENCHMARK_ROOT" "$PYTHONDONTWRITEBYTECODE" "$HF_HUB_OFFLINE" '
            '"$EVAL_FORCE_MERGE" "$PWD"\n'
            "  printf 'EVAL_ALL=%s\\n' \"$EVAL_ALL\"\n"
            f"  exit {engine_exit_code}\nfi\n"
        ),
        # The user's launch template includes pip; stub it to avoid an actual install.
        "pip": "#!/bin/bash\nprintf 'PIP:%s\\n' \"$*\"\n",
        "ls": "#!/bin/bash\nexit 0\n",
        "ray": "#!/bin/bash\nprintf 'ERROR: Ray service called\\n' >&2\nexit 99\n",
    }
    for name, content in scripts.items():
        path = bin_dir / name
        path.write_text(content)
        path.chmod(0o755)
    # Intercept fixed platform setup paths without modifying the launch template
    # or accessing the machine's actual Ascend/Conda installation.
    bootstrap = tmp_path / "bootstrap.sh"
    bootstrap.write_text(
        ".() {\n"
        '  case "$1" in\n'
        "    /usr/local/Ascend/*|*/anaconda3/etc/profile.d/conda.sh) return 0 ;;\n"
        '    *) builtin . "$@" ;;\n'
        "  esac\n}\n"
        "conda() { unset LD_PRELOAD; printf 'CONDA:%s\\n' \"$*\"; }\n"
    )
    model = tmp_path / "model with spaces" / "global_step_217"
    model.mkdir(parents=True)
    launch_dir, workspace = tmp_path / "launch directory", tmp_path / "code directory"
    launch_dir.mkdir()
    workspace.mkdir()
    train_config = tmp_path / "train.config"
    train_config.write_text(
        "MODEL_PATH='../model with spaces/global_step_217'\nNODE_RANK=0\nEVAL_FORCE_MERGE=True\n"
        f"WORKSPACE={shlex.quote(str(workspace))}\nDATASET_DIR={shlex.quote(str(tmp_path))}\n"
        + ("EVAL_ALL=1\n" if eval_all_source == "external_config" else "")
    )
    if eval_all_source != "default":
        launcher_env["EVAL_ALL"] = "0" if eval_all_source == "external_config" else "1"
    launcher_env.update(
        PATH=str(bin_dir) + os.pathsep + launcher_env["PATH"],
        BASH_ENV=str(bootstrap),
        TRAIN_CONFIG_PATH=str(train_config),
        VC_TASK_INDEX="1",
    )
    before = {str(path.relative_to(tmp_path)): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    result = _run(launcher_env, cwd=launch_dir)
    assert result.returncode == engine_exit_code, result.stderr
    assert "CONDA:activate verl" in result.stdout
    assert "PYTHON:-m verl.utils.evaluation.standalone_math" in result.stdout
    assert f"MODEL_PATH={model.resolve()}" in result.stdout
    assert f"ROOT={tmp_path / 'datasets'}" in result.stdout
    assert "NO_PYC=1\nOFFLINE=1" in result.stdout
    assert "FORCE_MERGE=True" in result.stdout
    assert f"EVAL_ALL={0 if eval_all_source == 'default' else 1}" in result.stdout
    assert f"WORKDIR={workspace}" in result.stdout
    assert "ERROR:" not in result.stderr
    after = {str(path.relative_to(tmp_path)): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert before == after
