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

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMON_SH = REPO_ROOT / "run" / "common.sh"


def _run_storage_config(
    *,
    dataset_dir: Path,
    workspace: Path,
    portable_dataset_dir: Path,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for key in ("WANDB_DIR", "WANDB_DATA_DIR", "WANDB_CACHE_DIR"):
        env.pop(key, None)
    env["WANDB_PORTABLE_DATASET_DIR"] = str(portable_dataset_dir)
    env.update(extra_env or {})
    script = f"""
source {COMMON_SH!s}
_configure_wandb_storage "$TRAIN_DATASET_DIR" "$TRAIN_WORKSPACE" "experiment-a" offline || exit $?
printf '%s\\n%s\\n%s\\n' "$WANDB_DIR" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR"
"""
    return subprocess.run(
        ["bash", "-c", script],
        cwd=REPO_ROOT,
        env={
            **env,
            "TRAIN_DATASET_DIR": str(dataset_dir),
            "TRAIN_WORKSPACE": str(workspace),
        },
        check=False,
        capture_output=True,
        text=True,
    )


def test_offline_wandb_storage_uses_sync_host_path_and_shared_staging_on_cpu(tmp_path):
    dataset_dir = tmp_path / "npu" / "dataset" / "jnguan"
    workspace = dataset_dir / "code" / "DMPD"
    workspace.mkdir(parents=True)
    portable_dataset_dir = tmp_path / "jump" / "dataset" / "jnguan"

    result = _run_storage_config(
        dataset_dir=dataset_dir,
        workspace=workspace,
        portable_dataset_dir=portable_dataset_dir,
    )

    assert result.returncode == 0, result.stderr
    wandb_dir, wandb_data_dir, wandb_cache_dir = result.stdout.splitlines()
    assert portable_dataset_dir.is_symlink()
    assert portable_dataset_dir.samefile(dataset_dir)
    assert Path(wandb_dir) == portable_dataset_dir / "code" / "DMPD" / "wandb_logs"
    assert Path(wandb_data_dir) == portable_dataset_dir / "code" / "DMPD" / "wandb_logs" / "wandb-data" / "experiment-a"
    assert Path(wandb_cache_dir) == (
        portable_dataset_dir / "code" / "DMPD" / "wandb_logs" / "wandb-cache" / "experiment-a"
    )
    assert Path(wandb_dir).is_dir()
    assert Path(wandb_data_dir).is_dir()
    assert Path(wandb_cache_dir).is_dir()


def test_offline_wandb_storage_respects_explicit_absolute_directories_on_cpu(tmp_path):
    dataset_dir = tmp_path / "npu" / "dataset" / "jnguan"
    workspace = dataset_dir / "code" / "DMPD"
    workspace.mkdir(parents=True)
    portable_dataset_dir = tmp_path / "unused" / "dataset" / "jnguan"
    explicit_run_root = tmp_path / "explicit" / "runs"
    explicit_data_root = tmp_path / "explicit" / "data"
    explicit_cache_root = tmp_path / "explicit" / "cache"

    result = _run_storage_config(
        dataset_dir=dataset_dir,
        workspace=workspace,
        portable_dataset_dir=portable_dataset_dir,
        extra_env={
            "WANDB_DIR": str(explicit_run_root),
            "WANDB_DATA_DIR": str(explicit_data_root),
            "WANDB_CACHE_DIR": str(explicit_cache_root),
        },
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [str(explicit_run_root), str(explicit_data_root), str(explicit_cache_root)]
    assert not portable_dataset_dir.exists()
    assert explicit_run_root.is_dir()
    assert explicit_data_root.is_dir()
    assert explicit_cache_root.is_dir()


def test_offline_wandb_storage_rejects_conflicting_portable_mount_on_cpu(tmp_path):
    dataset_dir = tmp_path / "npu" / "dataset" / "jnguan"
    workspace = dataset_dir / "code" / "DMPD"
    workspace.mkdir(parents=True)
    portable_dataset_dir = tmp_path / "jump" / "dataset" / "jnguan"
    portable_dataset_dir.mkdir(parents=True)

    result = _run_storage_config(
        dataset_dir=dataset_dir,
        workspace=workspace,
        portable_dataset_dir=portable_dataset_dir,
    )

    assert result.returncode != 0
    assert "does not resolve to the NPU dataset mount" in result.stderr


def _run_in_training_evaluation_config(extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for key in (
        "IN_TRAIN_EVALUATION",
        "VAL_BEFORE_TRAIN",
        "TEST_BEFORE_TRAIN",
        "TEACHER_VAL_BEFORE_TRAIN",
        "TEST_FREQ",
        "TEST_BENCHMARKS_FREQ",
    ):
        env.pop(key, None)
    env.update(extra_env or {})
    script = f"""
source {COMMON_SH!s}
_configure_in_training_evaluation || exit $?
printf '%s\n' \
    "$in_train_evaluation|$val_before_train|$test_before_train|$teacher_val_before_train|"\
"$test_freq|$test_benchmarks_freq"
"""
    return subprocess.run(
        ["bash", "-c", script],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_in_training_evaluation_is_fully_disabled_by_default_on_cpu():
    result = _run_in_training_evaluation_config(
        {
            "VAL_BEFORE_TRAIN": "True",
            "TEST_BEFORE_TRAIN": "True",
            "TEACHER_VAL_BEFORE_TRAIN": "True",
            "TEST_FREQ": "10",
            "TEST_BENCHMARKS_FREQ": "20",
        }
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False|False|False|False|-1|-1"


def test_in_training_evaluation_opt_in_preserves_legacy_controls_on_cpu():
    result = _run_in_training_evaluation_config(
        {
            "IN_TRAIN_EVALUATION": "True",
            "VAL_BEFORE_TRAIN": "False",
            "TEST_BEFORE_TRAIN": "True",
            "TEACHER_VAL_BEFORE_TRAIN": "True",
            "TEST_FREQ": "7",
            "TEST_BENCHMARKS_FREQ": "11",
        }
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True|False|True|True|7|11"
