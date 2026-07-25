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

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def _load_target_prompt_logprobs_module():
    repo_root = Path(__file__).resolve().parents[4]
    module_path = repo_root / "verl" / "workers" / "rollout" / "vllm_rollout" / "target_prompt_logprobs.py"
    spec = importlib.util.spec_from_file_location("target_prompt_logprobs_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_TARGET_MODULE = _load_target_prompt_logprobs_module()
VERL_TARGET_PROMPT_TOKEN_IDS_KEY = _TARGET_MODULE.VERL_TARGET_PROMPT_TOKEN_IDS_KEY
_build_target_logprob_rows = _TARGET_MODULE._build_target_logprob_rows
patch_vllm_model_runner_target_prompt_logprobs = _TARGET_MODULE.patch_vllm_model_runner_target_prompt_logprobs
_sync_model_runner_device = _TARGET_MODULE._sync_model_runner_device


class _FakeLogprobsTensors:
    def __init__(self, rows, width):
        # Match vLLM 0.11's real LogprobsTensors contract exactly.
        self.logprob_token_ids = torch.zeros(rows, width, dtype=torch.int32)
        self.logprobs = torch.zeros(rows, width)
        self.selected_token_ranks = torch.zeros(rows, dtype=torch.int32)

    @classmethod
    def empty_cpu(cls, rows, width):
        return cls(rows, width)


class _FakeSampler:
    @staticmethod
    def compute_logprobs(logits):
        return torch.log_softmax(logits.float(), dim=-1)


class _FakeModel:
    @staticmethod
    def compute_logits(hidden_states):
        return hidden_states


def _target_runner():
    target_rows = [[1, 4], [0, 4], [3, 1], [0, 0]]
    request = SimpleNamespace(
        prompt_token_ids=[9, 1, 2, 3],
        sampling_params=SimpleNamespace(
            extra_args={VERL_TARGET_PROMPT_TOKEN_IDS_KEY: target_rows},
        ),
        num_computed_tokens=0,
    )
    input_batch = SimpleNamespace(
        num_prompt_logprobs={"target": 2},
        in_progress_prompt_logprobs_cpu={},
        req_id_to_index={"target": 0},
    )
    runner = SimpleNamespace(
        input_batch=input_batch,
        requests={"target": request},
        query_start_loc_np=np.array([0]),
        model=_FakeModel(),
        sampler=_FakeSampler(),
        device=torch.device("cpu"),
        _sync_device=lambda: None,
    )
    runner._get_prompt_logprobs_dict = lambda hidden_states, state: {"legacy": (hidden_states, state)}
    return runner, target_rows


def test_vllm_target_prompt_patch_gathers_exact_per_position_ids_on_cpu():
    runner, target_rows = _target_runner()
    patch_vllm_model_runner_target_prompt_logprobs(
        runner,
        logprobs_tensors_cls=_FakeLogprobsTensors,
    )
    logits = torch.tensor(
        [
            [0.2, 2.0, -0.5, 0.1, 1.1],
            [1.0, -0.5, 0.3, 0.1, 2.0],
            [-0.2, 0.4, 0.5, 1.4, 0.0],
        ]
    )

    outputs = runner._get_prompt_logprobs_dict(logits, {"target": 4})

    output = outputs["target"]
    expected_ids = torch.tensor(target_rows[:3])
    expected_logprobs = torch.log_softmax(logits, dim=-1).gather(dim=-1, index=expected_ids)
    torch.testing.assert_close(output.logprob_token_ids[:, 0], torch.tensor([1, 2, 3], dtype=torch.int32))
    torch.testing.assert_close(output.logprob_token_ids[:, 1:], expected_ids.to(torch.int32))
    torch.testing.assert_close(output.logprobs[:, 1:], expected_logprobs)
    torch.testing.assert_close(output.selected_token_ranks, torch.tensor([3, 3, 3], dtype=torch.int32))


def test_vllm_target_prompt_rows_survive_v011_dict_serialization_on_cpu():
    normalized_logprobs = torch.log_softmax(
        torch.tensor(
            [
                [0.2, 2.0, -0.5, 0.1, 1.1],
                [1.0, -0.5, 0.3, 0.1, 2.0],
            ]
        ),
        dim=-1,
    )
    target_token_ids = torch.tensor([[1, 4], [0, 4]])
    sampled_token_ids = torch.tensor([1, 2])

    output_ids, output_logprobs, selected_ranks = _build_target_logprob_rows(
        normalized_logprobs=normalized_logprobs,
        target_token_ids=target_token_ids,
        sampled_token_ids=sampled_token_ids,
    )

    # Mirror vLLM 0.11 LogprobsProcessor._make_logprob_dict: column zero
    # receives selected_ranks[pos], while columns 1..K receive ranks 1..K.
    for position in range(len(sampled_token_ids)):
        serialized = {}
        ranks = [int(selected_ranks[position]), *range(1, output_ids.shape[1])]
        for token_id, logprob, rank in zip(
            output_ids[position].tolist(),
            output_logprobs[position].tolist(),
            ranks,
            strict=True,
        ):
            serialized[token_id] = SimpleNamespace(logprob=logprob, rank=rank)

        recovered_target_ids = [None] * target_token_ids.shape[1]
        for token_id, token_logprob in serialized.items():
            if 1 <= token_logprob.rank <= len(recovered_target_ids):
                recovered_target_ids[token_logprob.rank - 1] = token_id

        assert recovered_target_ids == target_token_ids[position].tolist()
        assert sampled_token_ids[position].item() in serialized


def test_vllm_target_prompt_patch_accumulates_chunked_prefill_on_cpu():
    runner, target_rows = _target_runner()
    patch_vllm_model_runner_target_prompt_logprobs(
        runner,
        logprobs_tensors_cls=_FakeLogprobsTensors,
    )
    first_logits = torch.tensor(
        [
            [0.2, 2.0, -0.5, 0.1, 1.1],
            [1.0, -0.5, 0.3, 0.1, 2.0],
        ]
    )
    second_logits = torch.tensor([[-0.2, 0.4, 0.5, 1.4, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])

    assert runner._get_prompt_logprobs_dict(first_logits, {"target": 2}) == {}
    runner.requests["target"].num_computed_tokens = 2
    outputs = runner._get_prompt_logprobs_dict(second_logits, {"target": 2})

    output = outputs["target"]
    all_logits = torch.cat((first_logits, second_logits[:1]), dim=0)
    expected_ids = torch.tensor(target_rows[:3])
    expected_logprobs = torch.log_softmax(all_logits, dim=-1).gather(dim=-1, index=expected_ids)
    torch.testing.assert_close(output.logprob_token_ids[:, 1:], expected_ids.to(torch.int32))
    torch.testing.assert_close(output.logprobs[:, 1:], expected_logprobs)
    torch.testing.assert_close(output.selected_token_ranks, torch.tensor([3, 3, 3], dtype=torch.int32))
    assert runner._verl_target_prompt_token_ids_tensors == {}


def test_vllm_target_prompt_patch_rejects_prefix_cache_hits_on_cpu():
    runner, _ = _target_runner()
    runner.requests["target"].num_computed_tokens = 1
    patch_vllm_model_runner_target_prompt_logprobs(
        runner,
        logprobs_tensors_cls=_FakeLogprobsTensors,
    )

    with pytest.raises(RuntimeError, match="cannot reuse prefix-cached KV states"):
        runner._get_prompt_logprobs_dict(torch.randn(3, 5), {"target": 3})


def test_vllm_target_prompt_patch_delegates_legacy_requests_without_changes_on_cpu():
    legacy_value = object()
    request = SimpleNamespace(
        prompt_token_ids=[1, 2],
        sampling_params=SimpleNamespace(extra_args={}),
        num_computed_tokens=0,
    )
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            num_prompt_logprobs={"legacy": 2},
            in_progress_prompt_logprobs_cpu={},
        ),
        requests={"legacy": request},
    )
    calls = []

    def original(hidden_states, state):
        calls.append((hidden_states, state))
        return legacy_value

    runner._get_prompt_logprobs_dict = original
    patch_vllm_model_runner_target_prompt_logprobs(
        runner,
        logprobs_tensors_cls=_FakeLogprobsTensors,
    )
    hidden_states = torch.randn(2, 3)
    scheduling_state = {"legacy": 2}

    returned = runner._get_prompt_logprobs_dict(hidden_states, scheduling_state)

    assert returned is legacy_value
    assert calls == [(hidden_states, scheduling_state)]


def test_vllm_target_prompt_patch_supports_vllm_ascend_v011_runner_contract_on_cpu(monkeypatch):
    """Exercise the interfaces used by vLLM-Ascend 0.11 on NPU.

    That runner passes a SchedulerOutput object, exposes
    ``query_start_loc_np``, stores widths on ``input_batch``, and has no
    ``_sync_device`` helper.
    """
    runner, target_rows = _target_runner()
    del runner._sync_device
    # A string remains accepted by torch tensor constructors while forcing the
    # synchronizer to use verl's dynamically selected device namespace.
    runner.device = "cpu"
    sync_calls = []
    monkeypatch.setattr(
        _TARGET_MODULE,
        "get_torch_device",
        lambda: SimpleNamespace(synchronize=lambda: sync_calls.append("sync")),
    )
    patch_vllm_model_runner_target_prompt_logprobs(
        runner,
        logprobs_tensors_cls=_FakeLogprobsTensors,
    )
    logits = torch.tensor(
        [
            [0.2, 2.0, -0.5, 0.1, 1.1],
            [1.0, -0.5, 0.3, 0.1, 2.0],
            [-0.2, 0.4, 0.5, 1.4, 0.0],
        ]
    )
    scheduler_output = SimpleNamespace(num_scheduled_tokens={"target": 4})

    outputs = runner._get_prompt_logprobs_dict(logits, scheduler_output)

    torch.testing.assert_close(
        outputs["target"].logprob_token_ids[:, 1:],
        torch.tensor(target_rows[:3], dtype=torch.int32),
    )
    assert sync_calls == ["sync"]


def test_vllm_target_prompt_patch_dispatches_sync_to_torch_npu_on_cpu(monkeypatch):
    sync_calls = []
    fake_npu = SimpleNamespace(synchronize=lambda: sync_calls.append("npu"))
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)
    runner = SimpleNamespace(device=SimpleNamespace(type="npu"))

    _sync_model_runner_device(runner)

    assert sync_calls == ["npu"]


def test_vllm_target_prompt_patch_supports_inherited_vllm_ascend_v013_contract_on_cpu():
    """Exercise the GPUModelRunner-compatible contract used since Ascend 0.13."""
    runner, target_rows = _target_runner()
    runner.num_prompt_logprobs = runner.input_batch.num_prompt_logprobs
    del runner.input_batch.num_prompt_logprobs
    runner.query_start_loc = SimpleNamespace(np=np.array([0]))
    del runner.query_start_loc_np
    patch_vllm_model_runner_target_prompt_logprobs(
        runner,
        logprobs_tensors_cls=_FakeLogprobsTensors,
    )
    logits = torch.tensor(
        [
            [0.2, 2.0, -0.5, 0.1, 1.1],
            [1.0, -0.5, 0.3, 0.1, 2.0],
            [-0.2, 0.4, 0.5, 1.4, 0.0],
        ]
    )

    outputs = runner._get_prompt_logprobs_dict(logits, {"target": 4})

    torch.testing.assert_close(
        outputs["target"].logprob_token_ids[:, 1:],
        torch.tensor(target_rows[:3], dtype=torch.int32),
    )
