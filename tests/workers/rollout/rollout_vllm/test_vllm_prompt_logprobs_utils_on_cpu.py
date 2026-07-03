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

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace


def _load_vllm_rollout_utils():
    repo_root = Path(__file__).resolve().parents[4]
    module_path = repo_root / "verl" / "workers" / "rollout" / "vllm_rollout" / "utils.py"
    stubs = {
        "vllm": types.ModuleType("vllm"),
        "vllm.outputs": types.ModuleType("vllm.outputs"),
        "verl.utils.vllm": types.ModuleType("verl.utils.vllm"),
        "verl.utils.vllm.patch": types.ModuleType("verl.utils.vllm.patch"),
        "verl.utils.vllm.vllm_fp8_utils": types.ModuleType("verl.utils.vllm.vllm_fp8_utils"),
    }
    stubs["vllm.outputs"].RequestOutput = object
    stubs["verl.utils.vllm"].TensorLoRARequest = object
    stubs["verl.utils.vllm"].VLLMHijack = SimpleNamespace(hijack=lambda: None)
    stubs["verl.utils.vllm.patch"].patch_vllm_moe_model_weight_loader = lambda model: None
    stubs["verl.utils.vllm.vllm_fp8_utils"].apply_vllm_fp8_patches = lambda: None
    stubs["verl.utils.vllm.vllm_fp8_utils"].is_fp8_model = lambda config: False
    stubs["verl.utils.vllm.vllm_fp8_utils"].load_quanted_weights = lambda *args, **kwargs: None

    previous = {name: sys.modules.get(name) for name in stubs}
    missing = {name for name in stubs if name not in sys.modules}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location("vllm_rollout_utils_under_test", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, module in previous.items():
            if name in missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


extract_prompt_logprobs = _load_vllm_rollout_utils().extract_prompt_logprobs


def _logprob(logprob: float, rank: int):
    return SimpleNamespace(logprob=logprob, rank=rank)


def test_extract_prompt_logprobs_uses_valid_compact_ranks_on_cpu():
    output = SimpleNamespace(
        prompt_token_ids=[101, 11, 22],
        prompt_logprobs=[
            None,
            {"11": _logprob(-0.1, 1), "12": _logprob(-0.2, 2)},
            {"22": _logprob(-0.3, 1), "23": _logprob(-0.4, 2)},
        ],
    )

    result = {}
    extract_prompt_logprobs(output, num_prompt_logprobs=2, result_dict=result)

    assert result["prompt_ids"] == [[11, 12], [22, 23], [0, 0]]
    assert result["prompt_logprobs"] == [[-0.1, -0.2], [-0.3, -0.4], [0.0, 0.0]]
    assert result["sampled_prompt_ids"] == [[11], [22], [0]]
    assert result["sampled_prompt_logprobs"] == [[-0.1], [-0.3], [0.0]]


def test_extract_prompt_logprobs_sorts_when_backend_ranks_are_invalid_on_cpu():
    output = SimpleNamespace(
        prompt_token_ids=[101, 10, 20],
        prompt_logprobs=[
            None,
            {
                "10": _logprob(-0.5, 999),
                "11": _logprob(-0.1, -3),
                "12": _logprob(-0.2, 7),
                "13": _logprob(-0.3, 2),
            },
            {
                "20": _logprob(-0.7, 99),
                "21": _logprob(-0.01, 1),
                "22": _logprob(-0.02, 4),
            },
        ],
    )

    result = {}
    extract_prompt_logprobs(output, num_prompt_logprobs=2, result_dict=result)

    assert result["prompt_ids"] == [[11, 12], [21, 22], [0, 0]]
    assert result["prompt_logprobs"] == [[-0.1, -0.2], [-0.01, -0.02], [0.0, 0.0]]
    assert result["sampled_prompt_ids"] == [[10], [20], [0]]
    assert result["sampled_prompt_logprobs"] == [[-0.5], [-0.7], [0.0]]
