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
import os

from ray._private.runtime_env.constants import RAY_JOB_CONFIG_JSON_ENV_VAR

from verl.utils.device import get_device_capability

_major, _ = get_device_capability()
# WAR: GB200 nodes without IMEX channel support raise ncclUnhandledCudaError 801 during
# Megatron all_gather (mbridge export_weights) when NCCL tries to use NVLS/MNNVL.
# Disable both on Blackwell (SM 10.x); non-Blackwell GPUs don't have MNNVL.
_gb200_nccl_env = {"NCCL_NVLS_ENABLE": "0", "NCCL_MNNVL_ENABLE": "0"} if (_major or 0) >= 10 else {}

PPO_RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        # TODO: disable compile cache due to cache corruption issue
        # https://github.com/vllm-project/vllm/issues/31199
        "VLLM_DISABLE_COMPILE_CACHE": "1",
        # Needed for multi-processes colocated on same NPU device
        # https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/maintenref/envvar/envref_07_0143.html
        "HCCL_HOST_SOCKET_PORT_RANGE": "auto",
        "HCCL_NPU_SOCKET_PORT_RANGE": "auto",
        "HSA_NO_SCRATCH_RECLAIM": "1",
        **_gb200_nccl_env,
    },
}

WANDB_RAY_ENV_PASSTHROUGH = [
    "WANDB_API_KEY",
    "WANDB_BASE_URL",
    "WANDB_CACHE_DIR",
    "WANDB_CONFIG_MODE",
    "WANDB_DATA_DIR",
    "WANDB_DIR",
    "WANDB_DISABLED",
    "WANDB_DISABLE_CODE",
    "WANDB_DISABLE_GIT",
    "WANDB_DISABLE_MACHINE_INFO",
    "WANDB_DISABLE_STATS",
    "WANDB_ENTITY",
    "WANDB_INIT_TIMEOUT",
    "WANDB_MODE",
    "WANDB_PROJECT",
    "WANDB_RESUME",
    "WANDB_RUN_ID",
    "WANDB_SAVE_REQUIREMENTS",
    "WANDB_SERVICE_WAIT",
    "WANDB_SILENT",
    "WANDB_START_METHOD",
    "WANDB__SERVICE_WAIT",
]

# Tool calls execute inside AgentLoop workers rather than in the Ray job entrypoint.
# The debug launcher starts the sandbox and exports these variables from that
# entrypoint, so explicitly propagate them into ray.init's runtime environment.
TOOL_RAY_ENV_PASSTHROUGH = [
    "PYTHON_SANDBOX_URL",
    "PYTHON_SANDBOX_TIMEOUT",
]


def get_ppo_ray_runtime_env():
    """
    A filter function to return the PPO Ray runtime environment.
    To avoid repeat of some environment variables that are already set.
    """
    job_runtime_env = json.loads(os.environ.get(RAY_JOB_CONFIG_JSON_ENV_VAR, "{}")).get("runtime_env", {})
    working_dir = job_runtime_env.get("working_dir")
    job_env_vars = job_runtime_env.get("env_vars", {})

    runtime_env = {
        "env_vars": PPO_RAY_RUNTIME_ENV["env_vars"].copy(),
        **({"working_dir": None} if working_dir is None else {}),
    }
    for key in list(runtime_env["env_vars"].keys()):
        if os.environ.get(key) is not None:
            runtime_env["env_vars"].pop(key, None)
    for key in WANDB_RAY_ENV_PASSTHROUGH:
        if key not in job_env_vars and (value := os.environ.get(key)) is not None:
            runtime_env["env_vars"][key] = value
    for key in TOOL_RAY_ENV_PASSTHROUGH:
        if key not in job_env_vars and (value := os.environ.get(key)) is not None:
            runtime_env["env_vars"][key] = value
    return runtime_env
