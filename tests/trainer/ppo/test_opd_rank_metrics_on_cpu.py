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

from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.distillation.config_validation import validate_distillation_intervention_training_config
from verl.trainer.distillation.diagnostics import SpecialTokenIds, compute_opd_diagnostics_metrics
from verl.trainer.distillation.rank_diagnostics import RANK_TEACHER_LOGPROBS_KEY
from verl.trainer.ppo import ray_trainer as trainer_module
from verl.trainer.ppo.metric_archive import MetricArchiveWriter, MetricRouter, _parse_metric_dimensions
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, _snapshot_diagnostics_kwargs
from verl.utils import tensordict_utils as tu
from verl.workers.config import DistillationDiagnosticsConfig, DistillationInterventionConfig


def _trainer(*, enabled=True, intervention=False, token_reward=True):
    trainer = object.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "distillation": {"enabled": True},
            "algorithm": {"adv_estimator": "token_reward_direct" if token_reward else "grpo"},
            "actor_rollout_ref": {"actor": {"calculate_sum_pi_squared": False}},
        }
    )
    trainer.distillation_config = SimpleNamespace(
        use_actor_loss=not token_reward,
        distillation_loss=SimpleNamespace(topk=2, top_k_strategy="only_stu_rethinking_style"),
        intervention=DistillationInterventionConfig(enabled=intervention),
        diagnostics=DistillationDiagnosticsConfig(
            enabled=True,
            collect_rank_metrics=enabled,
            topk=4,
            position_bin_size=2,
            log_interval=2,
            topk_log_interval=3,
            collect_intervention_metrics=False,
            collect_repetition_metrics=False,
            collect_train_text_zlib_repetition=False,
            log_wandb_heatmap=False,
        ),
    )
    trainer.opd_special_token_ids = SpecialTokenIds()
    trainer.global_steps = 6
    return trainer


def _batch(intervention=False):
    ids = torch.arange(4)
    lp = torch.tensor([0.4, 0.3, 0.2, 0.1]).log()
    batch = DataProto.from_dict(
        tensors={
            "prompts": torch.tensor([[0, 100, 101]]),
            "responses": torch.tensor([[200, 300, 201, 0]]),
            "input_ids": torch.tensor([[0, 100, 101, 200, 300, 201, 0]]),
            "attention_mask": torch.tensor([[0, 1, 1, 1, 1, 1, 0]]),
            "response_mask": torch.tensor([[1, 1, 1, 0]]),
            "old_log_probs": torch.full((1, 4), -2.0),
            "distillation_loss_mask": torch.tensor([[1, 0, 0, 0]]),
            "teacher_ids": ids.expand(1, 7, 4).clone(),
            "teacher_logprobs": lp.expand(1, 7, 4).clone(),
            "teacher_sampled_logprobs": torch.full((1, 7, 1), -2.0),
            "student_topk_ids": ids.expand(1, 4, 4).clone(),
            "student_topk_logprobs": lp.expand(1, 4, 4).clone(),
            RANK_TEACHER_LOGPROBS_KEY: lp.expand(1, 4, 4).clone(),
        }
    )
    if intervention:
        spans = np.empty(1, dtype=object)
        spans[0] = [{"start": 1, "teacher_tokens": 1}]
        batch.non_tensor_batch["teacher_interventions"] = spans
    return batch


@pytest.mark.parametrize("intervention", [False, True])
@pytest.mark.parametrize("token_reward", [False, True])
def test_payload_alignment_snapshot_and_training_width_are_preserved(intervention, token_reward):
    trainer = _trainer(intervention=intervention, token_reward=token_reward)
    batch = _batch(intervention)
    if not token_reward:
        batch.batch.pop(RANK_TEACHER_LOGPROBS_KEY)
    original_loss_mask = batch.batch["distillation_loss_mask"].clone()
    payload, _, _, _, _ = trainer._build_opd_diagnostics_inputs(batch=batch, entropy=None, global_steps=6)
    assert payload["teacher_topk_ids"] is None  # Old diagnostic switches remain off.
    assert payload["student_topk_ids"] is None
    assert "student_only_interventions" not in payload
    assert RANK_TEACHER_LOGPROBS_KEY not in batch.batch
    raw = payload["rank_inputs"]
    torch.testing.assert_close(raw["teacher_topk_logprobs"], batch.batch["teacher_logprobs"][:, 2:6])
    snapshot = _snapshot_diagnostics_kwargs(payload)
    direct = compute_opd_diagnostics_metrics(**payload)
    trainer._release_opd_diagnostics_tensors(batch)
    if token_reward:
        assert batch.batch["student_topk_ids"].shape[-1] == 2
        assert batch.batch["student_topk_logprobs"].shape[-1] == 2
        # Detached snapshot retains diagnostic width, not training width.
        assert snapshot["rank_inputs"]["student_topk_ids"].shape[-1] == 4
    else:
        assert "student_topk_ids" not in batch.batch
    torch.testing.assert_close(batch.batch["distillation_loss_mask"], original_loss_mask)
    raw["student_topk_logprobs"].fill_(-100)
    asynchronous = compute_opd_diagnostics_metrics(**snapshot)
    assert {k: v for k, v in asynchronous.items() if k != "opd/rank/compute_seconds"} == {
        k: v for k, v in direct.items() if k != "opd/rank/compute_seconds"
    }
    assert asynchronous["opd/rank/student_only/token_count"] == (2 if intervention else 3)
    assert asynchronous["opd/rank/exact_scores_available"] == int(token_reward)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("step", [1, 2, 3, 6, 7])
def test_rank_collection_uses_both_existing_intervals(enabled, step):
    trainer = _trainer(enabled=enabled)
    assert trainer._rank_diagnostics_due(step) is (enabled and step == 6)
    result = trainer._build_opd_diagnostics_inputs(batch=_batch(), entropy=None, global_steps=step)
    if step % 2:
        assert result is None
    else:
        assert ("rank_inputs" in result[0]) is (enabled and step == 6)


@pytest.mark.parametrize("token_reward", [False, True])
def test_rank_collection_adds_candidates_to_existing_actor_forward_only(monkeypatch, token_reward):
    trainer = _trainer(token_reward=token_reward)
    monkeypatch.setattr(trainer_module, "left_right_2_no_padding", lambda data: data)
    monkeypatch.setattr(trainer_module, "no_padding_2_padding", lambda tensor, data: tensor)
    calls = []

    def forward(data):
        k = tu.get_non_tensor_data(data, "diagnostics_student_topk", default=0)
        calls.append(k)
        tensors = {"log_probs": torch.full((1, 4), -2.0), "entropy": torch.ones(1, 4)}
        if k:
            tensors["student_topk_ids"] = torch.arange(k).expand(1, 4, k).clone()
            tensors["student_topk_logprobs"] = torch.full((1, 4, k), -2.0)
        return tu.get_tensordict(tensor_dict=tensors, non_tensor_dict={"metrics": {"mfu": 0.1}})

    trainer.actor_rollout_wg = SimpleNamespace(compute_log_prob=forward)
    for step in (2, 6):
        output, _ = trainer._compute_old_log_prob(_batch(), global_steps=step)
        if step == 6:
            assert output.batch["student_topk_ids"].shape[-1] == 4
    assert calls == [2 if token_reward else 0, 4]


@pytest.mark.parametrize(
    "override,match",
    [
        ({"distillation.diagnostics.enabled": False}, "enabled=true"),
        ({"distillation.diagnostics.topk": 2048}, "topk <= 1024"),
        ({"algorithm.rollout_correction.bypass_mode": True}, "bypass_mode=false"),
        ({"actor_rollout_ref.actor.strategy": "megatron"}, "fsdp/fsdp2"),
        ({"actor_rollout_ref.model.use_fused_kernels": True}, "use_fused_kernels=false"),
        ({"actor_rollout_ref.actor.fsdp_config.use_fused_kernels": True}, "use_fused_kernels=false"),
    ],
)
def test_rank_diagnostics_reject_unsupported_configuration_early(override, match):
    config = OmegaConf.create(
        {
            "distillation": {"enabled": True, "diagnostics": {"enabled": True, "collect_rank_metrics": True}},
            "actor_rollout_ref": {"actor": {"strategy": "fsdp2"}},
        }
    )
    for key, value in override.items():
        OmegaConf.update(config, key, value)
    with pytest.raises(ValueError, match=match):
        validate_distillation_intervention_training_config(config)
    config.distillation.diagnostics.collect_rank_metrics = False
    validate_distillation_intervention_training_config(config)


def test_rank_metrics_route_and_archive_with_existing_schema(tmp_path):
    trainer = _trainer()
    payload, *_ = trainer._build_opd_diagnostics_inputs(batch=_batch(), entropy=None, global_steps=6)
    metrics = compute_opd_diagnostics_metrics(**payload)
    route = MetricRouter("balanced", configured_topk=4).route(metrics)
    rank_live = {key for key in route.live_metrics if key.startswith("opd/rank/")}
    assert len(rank_live) <= 14
    live = "opd/rank/student_only/intersection/top4/kendall_tau_b/mean"
    detailed = "opd/rank/student_only/intersection/top4/kendall_tau_b_bin/2_4/count"
    assert live in rank_live
    assert detailed in route.detailed_metrics
    assert "opd/rank/all/intersection/top4/kendall_tau_b/mean" in route.detailed_metrics
    assert "opd/rank/student_only/intersection/top2/kendall_tau_b/mean" in route.detailed_metrics
    assert _parse_metric_dimensions(detailed) == {
        "bin_start": 2,
        "bin_end": 4,
        "coarse_region": None,
        "topk": 4,
        "eval_scope": None,
    }
    # The archive writer and downstream schema need no ranking-specific fields.
    writer = MetricArchiveWriter(
        tmp_path, logical_run_id="ranking", project="test", experiment="ranking", session_id="session"
    )
    writer.append_event(route, global_step=6, event_kind="train")
    writer.close_shard()
    writer.close()
    rows = [row for file in tmp_path.rglob("*.parquet") for row in pq.read_table(file).to_pylist()]
    assert any(row["metric_key"] == detailed and row["value"] == 1 and row["global_step"] == 6 for row in rows)
