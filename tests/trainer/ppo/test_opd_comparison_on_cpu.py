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
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import RandomSampler

from analysis import opd_comparison
from verl.protocol import DataProto
from verl.trainer.distillation.diagnostics import compute_opd_diagnostics_metrics
from verl.trainer.ppo.metric_archive import MetricArchiveWriter, MetricRouter
from verl.trainer.ppo.ray_trainer import (
    PRUNE_CALIBRATION_FILENAME,
    RayPPOTrainer,
    _rollout_cost_metrics,
    apply_intervention_prune_mask,
)


def write_archive(run, times):
    archive = run / "diagnostics"
    writer = MetricArchiveWriter(archive, logical_run_id=run.name, project="DMPD", experiment=run.name)
    router = MetricRouter("balanced")
    try:
        for step, seconds in times.items():
            writer.append_event(
                router.route({"timing_s/step": seconds, "opd_cost/allocated_devices": 40}),
                global_step=step,
            )
            ckpt = run / f"global_step_{step}"
            (ckpt / "actor").mkdir(parents=True)
            writer.commit_checkpoint(ckpt, checkpoint_step=step)
    finally:
        writer.close()
    return archive


def test_budget_uses_complete_committed_cost_and_actual_checkpoints(tmp_path):
    baseline = write_archive(tmp_path / "opd", {1: 60, 2: 60, 3: 60})
    candidate = write_archive(tmp_path / "ati", {1: 80, 2: 80, 3: 80})
    result = opd_comparison.budget_comparison(baseline, candidate, candidate.parent, step=3, devices=40)
    assert result["baseline"]["npu_hours"] == 2
    assert result["candidate_budget_cost"]["step"] == 2
    assert result["unused_npu_hours"] == pytest.approx(20 * 40 / 3600)
    assert result["candidate_fixed_step"]["step"] == 3
    missing = write_archive(tmp_path / "missing", {1: 60, 3: 60})
    with pytest.raises(ValueError, match="at step 2"):
        opd_comparison.budget_comparison(baseline, missing, missing.parent, step=3, devices=40)


def test_student_positions_exclude_teacher_and_reweight_common_buckets():
    ids = torch.tensor([[19, 20, 21, 22, 23, 24]])
    top16 = torch.arange(16).expand(1, 6, -1).clone()
    top16[0, [1, 3, 4], 0] = ids[0, [1, 3, 4]]
    metrics = compute_opd_diagnostics_metrics(
        student_logprobs=torch.full(ids.shape, -2.0),
        teacher_logprobs=torch.tensor([[-1.0, -6.0, -2.0, -2.0, -3.0, -5.0]]),
        response_mask=torch.ones_like(ids),
        response_ids=ids,
        teacher_topk_ids=top16,
        teacher_topk_logprobs=torch.full_like(top16, -2, dtype=torch.float32),
        student_only_interventions=[[{"start": 1, "teacher_tokens": 1}]],
        teacher_logprob_nonfinite_positions=[[5]],
        bin_size=2,
        collect_repetition_metrics=False,
    )
    assert metrics["opd/student_only/gap_abs_bin/0_2/sum"] == 1
    assert metrics["opd/student_only/gap_abs_bin/0_2/count"] == 1
    assert metrics["opd/student_only/top16_bin/2_4/sum"] == 1
    assert metrics["opd/student_only/top16_bin/4_6/count"] == 1
    # Use all three buckets but different candidate densities. The gap and hit
    # ratios should receive the same baseline weight in each bucket.
    baseline = dict(metrics)
    baseline["opd/student_only/gap_abs_bin/0_2/count"] = 5
    result = opd_comparison.matched_diagnostics({10: baseline}, {10: metrics}, through=10)
    assert result["gap_abs"]["summary"]["baseline"] == pytest.approx(2 / 8)
    assert result["gap_abs"]["summary"]["candidate"] == pytest.approx(6 / 8)
    assert result["top16"]["summary"]["delta"] == 0


def test_selection_mask_enables_the_actor_token_denominator():
    config = OmegaConf.create(
        {
            "distillation": {
                "enabled": True,
                "use_actor_loss": False,
                "intervention": {"enabled": False, "prune": True, "prune_mode": "selection_only"},
                "distillation_loss": {"horizon": {"mode": "full"}},
            },
            "algorithm": {"adv_estimator": "token_reward_direct"},
            "actor_rollout_ref": {"actor": {"policy_loss": {"loss_mode": "vanilla"}, "loss_agg_mode": "token-mean"}},
        }
    )
    trainer = object.__new__(RayPPOTrainer)
    trainer.config = config
    trainer.distillation_config = config.distillation
    mask = torch.tensor([[1, 1, 1], [1, 0, 0]])
    batch = DataProto.from_dict(
        tensors={"response_mask": mask.clone(), "attention_mask": torch.cat([torch.ones(2, 1), mask], dim=1)},
        non_tensors={"opd_selection_cutoff": np.array([2, 1])},
    )
    assert trainer._uses_distillation_policy_loss_mask()
    metrics = apply_intervention_prune_mask(batch, config.distillation.intervention)
    assert metrics["distillation/selection_only/retained_fraction"] == 0.75
    assert torch.equal(batch.batch["response_mask"], mask)
    assert batch.batch["distillation_loss_mask"].tolist() == [[1, 1, 0], [1, 0, 0]]


def test_adaptive_checkpoint_restores_sampler_and_calibration_at_epoch_end(tmp_path):
    def make_loader():
        dataset = list(range(9))
        sampler = RandomSampler(dataset, generator=torch.Generator().manual_seed(42))
        return StatefulDataLoader(dataset, batch_size=2, sampler=sampler, drop_last=True)

    original = make_loader()
    iterator = iter(original)
    first_epoch = [next(iterator).tolist() for _ in range(len(original))]
    torch.save(original.state_dict(), tmp_path / "data.pt")
    expected = [batch.tolist() for batch in original]
    assert expected != first_epoch
    calibration = {"history": [0.1, 0.2], "version": 2}
    (tmp_path / PRUNE_CALIBRATION_FILENAME).write_text(json.dumps(calibration))

    trainer = object.__new__(RayPPOTrainer)
    trainer._resolve_resume_context = Mock(
        return_value=SimpleNamespace(is_resume=True, path=str(tmp_path), step=len(original))
    )
    trainer.config = OmegaConf.create({"trainer": {"del_local_ckpt_after_load": False}})
    trainer.use_critic = False
    trainer.actor_rollout_wg = Mock()
    trainer.train_dataloader = make_loader()
    trainer.async_rollout_manager = Mock()
    trainer._load_checkpoint()
    trainer.async_rollout_manager.restore_prune_future_calibration.assert_called_once_with(calibration)
    trainer.actor_rollout_wg.load_checkpoint.assert_called_once()
    assert [batch.tolist() for batch in trainer.train_dataloader] == expected


def test_rollout_cost_preserves_discarded_work_and_adds_trainer_scoring():
    request_cost = {
        "opd_cost/student_initial/" + key: value
        for key, value in {"requests": 1, "output_known_requests": 1, "output_tokens": 5}.items()
    }
    batch = DataProto.from_dict(
        tensors={"responses": torch.ones(1, 4), "attention_mask": torch.ones(1, 5)},
        non_tensors={"opd_request_usage": [request_cost], "teacher_interventions": [[{"teacher_tokens": 1}]]},
        meta_info={"opd_target_request_usage": {"opd_cost/teacher_target_score/requests": 1}},
    )
    metrics = _rollout_cost_metrics(batch)
    assert metrics["opd_cost/trajectory/discarded_student_tokens"] == 2
    assert metrics["opd_cost/trajectory/teacher_tokens"] == 1
    assert metrics["opd_cost/teacher_target_score/requests"] == 1


def test_paired_evaluation_rejects_missing_or_duplicate_draws(tmp_path):
    benchmarks = opd_comparison.MATH5_BENCHMARKS
    records = [
        dict(benchmark=name, uid=f"{name}:1", draw=index, acc=int(index < 8))
        for name in benchmarks
        for index in range(16)
    ]
    protocol = dict(
        suite="math5",
        benchmarks=benchmarks,
        rollout_n=16,
        question_hashes={f"{name}:1": name for name in benchmarks},
    )
    baseline, candidate = tmp_path / "opd.json", tmp_path / "ati.json"
    base = dict(schema_version=1, protocol=protocol, records=records)
    baseline.write_text(json.dumps(base))
    candidate.write_text(json.dumps({**base, "records": [{**row, "acc": 1} for row in records]}))
    result = opd_comparison.paired_evaluation(baseline, candidate, resamples=100, seed=42)
    assert result["macro"]["delta"] == 0.5
    assert result["macro"]["ci95"] == [0.5, 0.5]
    candidate.write_text(json.dumps({**base, "records": records[:-1]}))
    with pytest.raises(ValueError, match="Missing questions or draws"):
        opd_comparison.read_evaluation(candidate)
    candidate.write_text(json.dumps({**base, "records": [*records, records[0]]}))
    with pytest.raises(ValueError, match="duplicate draw"):
        opd_comparison.read_evaluation(candidate)
