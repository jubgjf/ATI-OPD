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
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "set -euo pipefail" not in source
    assert "EVAL_SHELL_TRACE:-0" in source
    assert 'if [[ "${EVAL_SHELL_TRACE:-0}" == "1" ]]' in source


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
    assert "released $candidate_experiment/global_step_$step; rescanning immediately" in source
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


def test_checkpoint_eval_launcher_uses_distributed_claims_and_instance_scoped_control_on_cpu():
    source = LAUNCHER.read_text(encoding="utf-8")

    assert "controller.lock" not in source
    assert "flock -n" not in source
    assert "EVAL_INSTANCE_ID" in source
    assert "instance_runtime_root" in source
    assert "acquire-claim" in source
    assert "heartbeat-claim" in source
    assert "release-claim" in source
    assert "render-queue" in source
    assert "sync_wandb_serialized" in source
    assert "EVAL_CLAIM_STALE_SECONDS=${EVAL_CLAIM_STALE_SECONDS:-300}" in source
    assert 'if checkpoint_complete "$step" && checkpoint_wandb_is_staged "$step"; then' in source
    assert 'elif checkpoint_complete "$step"; then' in source


def test_checkpoint_eval_launcher_validates_direct_merge_before_starting_vllm_on_cpu():
    source = LAUNCHER.read_text(encoding="utf-8")

    assert '--target_dir "$target"' in source
    assert 'if ! hf_checkpoint_ready "$target"' in source
    assert "Failed to merge FSDP checkpoint" in source
    assert 'mv -- "$temporary" "$target"' not in source


def test_checkpoint_eval_launcher_uses_default_local_dir_parent_as_search_root_on_cpu(tmp_path):
    workspace = tmp_path / "shared-workspace"
    expected_search_root = workspace / "outputs"
    env = os.environ.copy()
    env.update(
        {
            "ALLOW_DEBUG_OVERRIDES": "1",
            "DEFAULT_LOCAL_DIR": "outputs/experiment-a",
            "EVAL_RESULTS_ROOT": "eval-results",
            "EVAL_INSTANCE_ID": "launcher-test",
            "EVAL_ROLE": "controller",
            "EVAL_SETUP_MODE": "skip",
            "EVAL_WANDB_ROOT": "wandb_logs/DMPD-eval",
            "PATH": os.pathsep.join((str(Path(sys.executable).parent), env.get("PATH", ""))),
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
    assert "instance:         launcher-test" in completed.stdout
    assert not list((expected_search_root / ".standalone-eval" / "claims").rglob("*.claim"))


def test_checkpoint_eval_launcher_requires_a_shared_instance_identity_on_cpu(tmp_path):
    env = os.environ.copy()
    for name in ("EVAL_INSTANCE_ID", "MA_VJ_NAME", "MA_TASK_NAME"):
        env.pop(name, None)
    env.update(
        {
            "EVAL_ROLE": "controller",
            "EVAL_SETUP_MODE": "skip",
            "WANDB_MODE": "disabled",
            "WATCH_ONCE": "1",
            "WORKSPACE": str(tmp_path),
        }
    )

    completed = subprocess.run(["bash", str(LAUNCHER)], capture_output=True, env=env, text=True)

    assert completed.returncode == 2
    assert "EVAL_INSTANCE_ID is required" in completed.stderr
