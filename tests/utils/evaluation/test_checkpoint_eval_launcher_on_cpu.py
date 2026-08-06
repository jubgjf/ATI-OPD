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
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = REPO_ROOT / "run" / "opd_checkpoint_eval.sh"


def test_checkpoint_eval_launcher_is_valid_and_replaces_legacy_launcher_on_cpu():
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)

    assert LAUNCHER.is_file()
    assert not (REPO_ROOT / "run" / "opd_benchmark_eval.sh").exists()


def test_checkpoint_eval_launcher_is_device_agnostic_and_keeps_prefix_caching_on_cpu():
    source = LAUNCHER.read_text(encoding="utf-8")

    assert "CUDA_VISIBLE_DEVICES" not in source
    assert "torch.cuda" not in source
    assert "--enable-prefix-caching" in source
    assert "--data-parallel-size" in source
    assert "--tensor-parallel-size" in source


def test_checkpoint_eval_launcher_supports_offline_shared_storage_and_manual_filters_on_cpu():
    source = LAUNCHER.read_text(encoding="utf-8")

    assert "EVAL_PORTABLE_RESULT_ROOT" not in source
    assert "WANDB_PORTABLE_DATASET_DIR" in source
    assert "EVAL_WANDB_ROOT" in source
    assert "EVAL_RESULTS_ROOT" in source
    assert "wandb-offline-sync.sh" in source
    assert "EVAL_STEPS_TEXT" in source
    assert "EVAL_EXPERIMENTS_TEXT" in source
    assert "EVAL_BENCHMARKS_TEXT" in source
    assert "POLL_SECONDS=${POLL_SECONDS:-600}" in source
    assert source.count('sleep "$POLL_SECONDS"') == 1
    assert "processed $processed checkpoint(s); rescanning immediately" in source
    assert "wandb-sync.json" in source


def test_checkpoint_eval_launcher_scans_all_experiments_and_bootstraps_npu_environment_on_cpu():
    source = LAUNCHER.read_text(encoding="utf-8")

    assert "PYTHON=" not in source
    assert "EVAL_MODULE=" not in source
    assert "WATCHER_MODULE=" not in source
    assert "python -m verl.utils.evaluation.opd_benchmark" in source
    assert "python -m verl.utils.evaluation.checkpoint_watcher" in source
    assert "CHECKPOINT_SEARCH_ROOT" in source
    assert "discover-tree" in source
    assert "teacher-request.json" in source
    assert "pip install -r requirements-npu.txt" in source
    assert "pip install -e ." in source
    assert "util-linux" in source
    assert "/usr/local/Ascend/ascend-toolkit/set_env.sh" in source
    assert "verl-conda-env-20260617.tar.gz" in source
    assert "EVAL_SETUP_MODE" in source


def test_checkpoint_eval_launcher_uses_default_local_dir_parent_as_search_root_on_cpu(tmp_path):
    workspace = tmp_path / "shared-workspace"
    expected_search_root = workspace / "outputs"
    test_bin = tmp_path / "bin"
    test_bin.mkdir()
    flock = test_bin / "flock"
    flock.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    flock.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "ALLOW_DEBUG_OVERRIDES": "1",
            "DEFAULT_LOCAL_DIR": "outputs/experiment-a",
            "EVAL_RESULTS_ROOT": "eval-results",
            "EVAL_ROLE": "controller",
            "EVAL_SETUP_MODE": "skip",
            "EVAL_WANDB_ROOT": "wandb_logs/DMPD-eval",
            "PATH": os.pathsep.join((str(test_bin), str(Path(sys.executable).parent), env.get("PATH", ""))),
            "WANDB_MODE": "disabled",
            "WATCH_ONCE": "1",
            "WORKSPACE": str(workspace),
        }
    )

    completed = subprocess.run(
        ["bash", str(LAUNCHER)],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert f"checkpoint search: {expected_search_root}" in completed.stdout
    assert expected_search_root.is_dir()
    assert (workspace / "eval-results").is_dir()
    assert (workspace / "wandb_logs" / "DMPD-eval").is_dir()
