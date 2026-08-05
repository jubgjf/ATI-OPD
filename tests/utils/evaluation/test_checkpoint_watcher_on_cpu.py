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

import json

from omegaconf import OmegaConf

from verl.utils.evaluation.checkpoint_evaluation import write_evaluation_spec, write_evaluation_state
from verl.utils.evaluation.checkpoint_watcher import (
    discover_committed_checkpoints,
    load_endpoint_descriptor,
    pending_checkpoints,
    write_endpoint_descriptor,
)


def _config(*, metric_logging_mode: str = "full"):
    return OmegaConf.create(
        {
            "trainer": {
                "experiment_name": "watcher-test",
                "project_name": "DMPD",
                "metric_logging": {"mode": metric_logging_mode},
                "save_freq": 20,
                "max_actor_ckpt_to_keep": None,
            },
            "actor_rollout_ref": {
                "model": {"path": "/student", "tokenizer_path": None, "chat_template_from_model": "/teacher"},
                "rollout": {"test_kwargs": {"n": 16}, "max_model_len": 4096},
            },
            "data": {
                "max_prompt_length": 1024,
                "max_response_length": 3072,
                "seed": 1,
                "test_benchmarks": {"names": ["math-ai/aime24"], "reward_data_source": "math_verify"},
            },
            "reward": {"reward_manager": {"name": "naive"}},
            "distillation": {
                "enabled": True,
                "teacher_key": "data_source",
                "teacher_models": {"default": {"model_path": "/teacher"}},
                "distillation_loss": {"log_prob_min_clamp": -10.0},
                "intervention": {"enabled": True, "gap_metric": "k3"},
            },
        }
    )


def _write_fsdp_checkpoint(root, step: int, spec: dict, *, complete: bool = True, balanced: bool = False):
    checkpoint = root / f"global_step_{step}"
    actor = checkpoint / "actor"
    actor.mkdir(parents=True)
    (checkpoint / "data.pt").write_bytes(b"data")
    write_evaluation_state(
        checkpoint,
        checkpoint_step=step,
        spec_fingerprint=spec["spec_fingerprint"],
        teacher_nll_calibration=None,
    )
    (actor / "model_world_size_2_rank_0.pt").write_bytes(b"rank-0")
    if complete:
        (actor / "model_world_size_2_rank_1.pt").write_bytes(b"rank-1")
    if balanced:
        (checkpoint / "metrics_commit.json").write_text(
            json.dumps({"checkpoint_step": step}),
            encoding="utf-8",
        )
    return checkpoint


def test_watcher_discovers_only_committed_complete_checkpoints_in_step_order(tmp_path):
    spec = write_evaluation_spec(tmp_path, _config(), training_run_id="run")
    _write_fsdp_checkpoint(tmp_path, 20, spec)
    _write_fsdp_checkpoint(tmp_path, 0, spec)
    _write_fsdp_checkpoint(tmp_path, 30, spec, complete=False)
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("30", encoding="utf-8")

    candidates = discover_committed_checkpoints(tmp_path)
    assert [candidate.step for candidate in candidates] == [0, 20]
    assert all(candidate.needs_merge for candidate in candidates)
    assert [candidate.step for candidate in pending_checkpoints(tmp_path, completed_steps=[0])] == [20]


def test_watcher_does_not_synthesize_global_step_zero(tmp_path):
    spec = write_evaluation_spec(tmp_path, _config(), training_run_id="run")
    _write_fsdp_checkpoint(tmp_path, 20, spec)
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("20", encoding="utf-8")
    assert [candidate.step for candidate in discover_committed_checkpoints(tmp_path)] == [20]


def test_balanced_watcher_requires_checkpoint_local_metric_commit(tmp_path):
    spec = write_evaluation_spec(tmp_path, _config(metric_logging_mode="balanced"), training_run_id="run")
    checkpoint = _write_fsdp_checkpoint(tmp_path, 20, spec, balanced=False)
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("20", encoding="utf-8")
    assert discover_committed_checkpoints(tmp_path) == []
    (checkpoint / "metrics_commit.json").write_text(json.dumps({"checkpoint_step": 20}), encoding="utf-8")
    assert [candidate.step for candidate in discover_committed_checkpoints(tmp_path)] == [20]


def test_watcher_waits_for_every_indexed_huggingface_weight_shard(tmp_path):
    spec = write_evaluation_spec(tmp_path, _config(), training_run_id="run")
    checkpoint = _write_fsdp_checkpoint(tmp_path, 20, spec, complete=False)
    hf = checkpoint / "actor" / "huggingface"
    hf.mkdir()
    (hf / "config.json").write_text("{}", encoding="utf-8")
    (hf / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (hf / "tokenizer.json").write_text("{}", encoding="utf-8")
    (hf / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "first": "model-00001-of-00002.safetensors",
                    "second": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    (hf / "model-00001-of-00002.safetensors").write_bytes(b"first")
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("20", encoding="utf-8")

    assert discover_committed_checkpoints(tmp_path) == []
    (hf / "model-00002-of-00002.safetensors").write_bytes(b"second")
    candidates = discover_committed_checkpoints(tmp_path)
    assert [candidate.step for candidate in candidates] == [20]
    assert candidates[0].needs_merge is False


def test_endpoint_descriptor_round_trip(tmp_path):
    model = tmp_path / "teacher"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"qwen2"}', encoding="utf-8")
    endpoint_path = tmp_path / "teacher-endpoint.json"
    written = write_endpoint_descriptor(
        endpoint_path,
        base_url="http://teacher:8001/v1/",
        served_model_name="teacher",
        model_path=model,
    )
    assert load_endpoint_descriptor(endpoint_path) == written
    assert written["base_url"] == "http://teacher:8001/v1"

    (model / "model.safetensors").write_bytes(b"weights")
    changed = write_endpoint_descriptor(
        endpoint_path,
        base_url="http://teacher:8001/v1",
        served_model_name="teacher",
        model_path=model,
    )
    assert changed["model_fingerprint"] != written["model_fingerprint"]
