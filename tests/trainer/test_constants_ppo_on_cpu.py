# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from ray._private.runtime_env.constants import RAY_JOB_CONFIG_JSON_ENV_VAR

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env


def test_get_ppo_ray_runtime_env_skips_wandb_vars_already_set_by_job(monkeypatch):
    monkeypatch.setenv(
        RAY_JOB_CONFIG_JSON_ENV_VAR,
        json.dumps(
            {
                "runtime_env": {
                    "working_dir": "gcs://_ray_pkg_test.zip",
                    "env_vars": {
                        "WANDB_RESUME": "must",
                        "WANDB_RUN_ID": "resume-run-id",
                    },
                }
            }
        ),
    )
    monkeypatch.setenv("WANDB_RESUME", "must")
    monkeypatch.setenv("WANDB_RUN_ID", "resume-run-id")
    monkeypatch.setenv("WANDB_MODE", "offline")

    runtime_env = get_ppo_ray_runtime_env()

    assert "working_dir" not in runtime_env
    assert "WANDB_RESUME" not in runtime_env["env_vars"]
    assert "WANDB_RUN_ID" not in runtime_env["env_vars"]
    assert runtime_env["env_vars"]["WANDB_MODE"] == "offline"


def test_get_ppo_ray_runtime_env_passes_through_wandb_vars_without_job_runtime_env(monkeypatch):
    monkeypatch.delenv(RAY_JOB_CONFIG_JSON_ENV_VAR, raising=False)
    monkeypatch.setenv("WANDB_RESUME", "must")
    monkeypatch.setenv("WANDB_RUN_ID", "resume-run-id")

    runtime_env = get_ppo_ray_runtime_env()

    assert runtime_env["working_dir"] is None
    assert runtime_env["env_vars"]["WANDB_RESUME"] == "must"
    assert runtime_env["env_vars"]["WANDB_RUN_ID"] == "resume-run-id"
