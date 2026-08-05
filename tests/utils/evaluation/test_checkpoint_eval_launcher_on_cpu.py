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

import subprocess
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

    assert "EVAL_PORTABLE_RESULT_ROOT" in source
    assert "wandb-offline-sync.sh" in source
    assert "EVAL_STEPS_TEXT" in source
    assert "EVAL_BENCHMARKS_TEXT" in source
    assert "POLL_SECONDS=${POLL_SECONDS:-1800}" in source
    assert "$HOME" not in source
