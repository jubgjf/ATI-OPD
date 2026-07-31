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
    for key in ("WANDB_DIR", "WANDB_DATA_DIR"):
        env.pop(key, None)
    env["WANDB_PORTABLE_DATASET_DIR"] = str(portable_dataset_dir)
    env.update(extra_env or {})
    script = f"""
source {COMMON_SH!s}
_configure_wandb_storage "$TRAIN_DATASET_DIR" "$TRAIN_WORKSPACE" "experiment-a" offline || exit $?
printf '%s\\n%s\\n' "$WANDB_DIR" "$WANDB_DATA_DIR"
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
    wandb_dir, wandb_data_dir = result.stdout.splitlines()
    assert portable_dataset_dir.is_symlink()
    assert portable_dataset_dir.samefile(dataset_dir)
    assert Path(wandb_dir) == portable_dataset_dir / "code" / "DMPD" / "wandb_logs"
    assert Path(wandb_data_dir) == portable_dataset_dir / "code" / "DMPD" / "wandb_logs" / "wandb-data" / "experiment-a"
    assert Path(wandb_dir).is_dir()
    assert Path(wandb_data_dir).is_dir()


def test_offline_wandb_storage_respects_explicit_absolute_directories_on_cpu(tmp_path):
    dataset_dir = tmp_path / "npu" / "dataset" / "jnguan"
    workspace = dataset_dir / "code" / "DMPD"
    workspace.mkdir(parents=True)
    portable_dataset_dir = tmp_path / "unused" / "dataset" / "jnguan"
    explicit_run_root = tmp_path / "explicit" / "runs"
    explicit_data_root = tmp_path / "explicit" / "data"

    result = _run_storage_config(
        dataset_dir=dataset_dir,
        workspace=workspace,
        portable_dataset_dir=portable_dataset_dir,
        extra_env={
            "WANDB_DIR": str(explicit_run_root),
            "WANDB_DATA_DIR": str(explicit_data_root),
        },
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [str(explicit_run_root), str(explicit_data_root)]
    assert not portable_dataset_dir.exists()
    assert explicit_run_root.is_dir()
    assert explicit_data_root.is_dir()


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
