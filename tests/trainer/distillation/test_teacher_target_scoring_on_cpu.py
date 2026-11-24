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

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopMetrics,
    AgentLoopOutput,
    AgentLoopWorker,
    DeferredFinalInterventionScore,
    _PruneFutureCalibrationSnapshot,
)
from verl.experimental.teacher_loop.teacher_manager import AsyncTeacherLLMServerManager
from verl.trainer.distillation.rank_diagnostics import RANK_TEACHER_LOGPROBS_KEY
from verl.trainer.distillation.teacher_target_scoring import (
    assemble_teacher_target_logprobs,
    build_teacher_target_scoring_requests,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, apply_intervention_prune_mask
from verl.workers.config import DistillationInterventionConfig
from verl.workers.rollout.replica import TokenOutput


def test_teacher_target_request_aligns_left_padded_prompt_and_repaired_response_on_cpu():
    # Prompt width 4: [pad, pad, 101, 102].
    # Response width 4: three real tokens, then right padding. The middle real
    # token can be a teacher-inserted DMPD token; attention alignment must still
    # include it even when a later reward mask excludes it.
    input_ids = torch.tensor([[0, 0, 101, 102, 201, 301, 202, 0]])
    attention_mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 0]])
    student_topk_ids = torch.tensor(
        [
            [
                [11, 12],
                [21, 22],
                [31, 32],
                [0, 0],
            ]
        ]
    )

    requests = build_teacher_target_scoring_requests(
        input_ids=input_ids,
        attention_mask=attention_mask,
        student_topk_ids=student_topk_ids,
        prompt_width=4,
        routing_keys=["math"],
    )

    assert len(requests) == 1
    request = requests[0]
    assert request.sequence_ids == [101, 102, 201, 301, 202]
    assert request.routing_key == "math"
    assert request.response_row_mapping == ((0, 1), (1, 2), (2, 3))
    assert request.target_ids == [[0, 1], [11, 12], [21, 22], [31, 32], [0, 1]]


def test_teacher_target_result_is_verified_and_repadded_on_cpu():
    input_ids = torch.tensor([[0, 101, 102, 201, 202, 0]])
    attention_mask = torch.tensor([[0, 1, 1, 1, 1, 0]])
    student_topk_ids = torch.tensor([[[11, 12], [21, 22], [0, 0]]])
    requests = build_teacher_target_scoring_requests(
        input_ids=input_ids,
        attention_mask=attention_mask,
        student_topk_ids=student_topk_ids,
        prompt_width=3,
    )
    result = SimpleNamespace(
        ids=torch.tensor([[0, 1], [11, 12], [21, 22], [0, 0]], dtype=torch.int32),
        logprobs=torch.tensor([[0.0, 0.0], [-1.1, -1.2], [-2.1, -2.2], [0.0, 0.0]]),
    )

    teacher_logprobs = assemble_teacher_target_logprobs(
        requests=requests,
        scoring_results=[result],
        student_topk_ids=student_topk_ids,
    )

    torch.testing.assert_close(
        teacher_logprobs,
        torch.tensor([[[-1.1, -1.2], [-2.1, -2.2], [0.0, 0.0]]]),
    )


@pytest.mark.parametrize(("diagnostics_topk", "adaptive"), [(0, False), (2, False), (2, True)])
@pytest.mark.parametrize("rank_metrics", [False, True])
@pytest.mark.parametrize("log_prob_min_clamp", [None, -2.0])
def test_trainer_fused_teacher_scores_preserve_rewards_and_apply_prune_on_cpu(
    diagnostics_topk, adaptive, rank_metrics, log_prob_min_clamp
):
    # Rows predict the next token. The last two emitted tokens have large K3 against the teacher.
    logprobs = torch.log_softmax(
        torch.tensor([[0.1, 0.9, 1.0, 0.2, -1.0], [0, 3, 1, 2, 0.5], [4, 3, -8, 0, 1], [4, 3, 0, 1, -8]]),
        dim=-1,
    )
    sequence_ids = [4, 3, 1, 2, 4]

    class _CpuTeacher:
        def __init__(self):
            self.calls = []

        async def generate(
            self, *, prompt_ids, sampling_params, usage_kind, target_prompt_token_ids=None, teacher_prompt_topk=0, **_
        ):
            assert prompt_ids == sequence_ids
            self.calls.append((usage_kind, sampling_params["prompt_logprobs"]))

            def scores_for(ids):
                values = logprobs.gather(1, torch.tensor(ids[:-1]))
                return [*values.tolist(), [0.0] * len(ids[0])]

            sampled_ids = [[token_id] for token_id in [*sequence_ids[1:], 0]]
            width = teacher_prompt_topk or (
                sampling_params["prompt_logprobs"] if target_prompt_token_ids is None else diagnostics_topk
            )
            top_ids = [*logprobs.topk(width, dim=-1).indices.tolist(), [0] * width] if width else sampled_ids
            ids = target_prompt_token_ids or top_ids
            extra = {
                "prompt_ids": ids,
                "prompt_logprobs": scores_for(ids),
                "sampled_prompt_ids": sampled_ids,
                "sampled_prompt_logprobs": scores_for(sampled_ids),
            }
            if teacher_prompt_topk:
                extra["teacher_topk_prompt_ids"] = top_ids
                extra["teacher_topk_prompt_logprobs"] = scores_for(top_ids)
            return TokenOutput(token_ids=[4], extra_fields=extra)

    rewards = []
    for fused in (False, True):
        cpu_teacher = _CpuTeacher()
        loss = SimpleNamespace(
            topk=2,
            top_k_strategy="only_stu_rethinking_style",
            reward_weight_mode="student_p",
            log_prob_min_clamp=log_prob_min_clamp,
        )
        intervention = DistillationInterventionConfig(
            enabled=adaptive,
            prune=adaptive,
            prune_mode="adaptive_future" if adaptive else "teacher_end",
            gap_metric="k3",
            max_interventions_per_sample=1,
            prune_future_window_tokens=1,
            prune_future_min_valid_tokens=1,
        )
        config = SimpleNamespace(
            fuse_final_teacher_scoring=fused,
            use_actor_loss=False,
            teacher_key="data_source",
            distillation_loss=loss,
            diagnostics=SimpleNamespace(
                enabled=True,
                collect_topk_overlap=bool(diagnostics_topk),
                collect_rank_metrics=rank_metrics,
                topk=diagnostics_topk or 2,
                log_interval=1,
                topk_log_interval=10,
            ),
            intervention=intervention,
        )
        manager = object.__new__(AsyncTeacherLLMServerManager)
        manager.teacher_model_configs = {
            "teacher": SimpleNamespace(inference=SimpleNamespace(name="vllm", temperature=1))
        }
        manager.teacher_client = {"teacher": cpu_teacher}
        manager.distillation_loss_config = loss
        trainer = object.__new__(RayPPOTrainer)
        trainer.distillation_config = config
        trainer.global_steps = 10
        trainer.config = OmegaConf.create(
            {"algorithm": {"adv_estimator": "token_reward_direct"}, "distillation": {"enabled": True}}
        )
        trainer.tokenizer = SimpleNamespace(pad_token_id=0)
        trainer._rethinking_teacher_target_scorer = manager
        worker = object.__new__(AgentLoopWorker)
        worker.config = trainer.config
        worker.distillation_enabled = True
        worker.distillation_config = config
        worker.teacher_key = config.teacher_key
        worker.teacher_server_manager = manager
        output = AgentLoopOutput(
            prompt_ids=sequence_ids[:2],
            response_ids=sequence_ids[2:],
            response_mask=[1, 1, 1],
            metrics=AgentLoopMetrics(),
        )
        asyncio.run(
            AgentLoopWorker._compute_teacher_logprobs(worker, output, output.prompt_ids, output.response_ids, False)
        )

        batch = DataProto.from_dict(
            tensors={
                "prompts": torch.tensor([[0, 4, 3]]),
                "responses": torch.tensor([[1, 2, 4, 0]]),
                "input_ids": torch.tensor([[0, *sequence_ids, 0]]),
                "attention_mask": torch.tensor([[0, 1, 1, 1, 1, 1, 0]]),
                "response_mask": torch.tensor([[1, 1, 1, 0]]),
                "student_topk_ids": torch.tensor([[[1, 2], [2, 3], [4, 1], [0, 0]]]),
                "student_topk_logprobs": torch.tensor([[[-0.2, -0.5], [-0.2, -0.5], [-0.2, -0.5], [0.0, 0.0]]]),
            }
        )
        pending = None
        if adaptive:
            records = np.empty(1, dtype=object)
            records[0] = [{"start": 0, "teacher_tokens": 1}]
            batch.non_tensor_batch["teacher_interventions"] = records
            if fused:
                pending = np.empty(1, dtype=object)
                pending[0] = DeferredFinalInterventionScore(
                    intervention_index=0,
                    after_intervention_start=1,
                    eligibility_mask=torch.tensor([False, True, True]),
                    rollout_logprobs=torch.tensor([0.0, -0.2, -0.2]),
                    calibration=_PruneFutureCalibrationSnapshot(threshold=1.0, source="initial", count=0, version=0),
                )
            else:
                records[0][0]["prune_future_cutoff"] = 1
                apply_intervention_prune_mask(batch, intervention)

        scores, _ = trainer._compute_token_reward_direct_scores(batch, deferred_intervention_scores=pending)
        rewards.append(scores)
        assert (RANK_TEACHER_LOGPROBS_KEY in batch.batch) is rank_metrics
        if rank_metrics:
            exact = batch.batch[RANK_TEACHER_LOGPROBS_KEY]
            assert exact.min() < -10  # Raw diagnostic scores survive the optional -2 loss clamp.
            torch.testing.assert_close(exact[0, :3], logprobs[1:].gather(1, batch.batch["student_topk_ids"][0, :3]))
        if fused:
            assert cpu_teacher.calls == [
                ("teacher_target_score", 2 + (max(2, diagnostics_topk) if rank_metrics else diagnostics_topk))
            ]
            assert not output.extra_fields.get("teacher_sampled_ids")
            torch.testing.assert_close(
                trainer._extract_teacher_response_logprobs_for_diagnostics(batch)[0, :3],
                logprobs[[1, 2, 3], sequence_ids[2:]],
            )
            if diagnostics_topk:
                teacher_ids, teacher_logprobs = trainer._extract_teacher_response_topk_for_diagnostics(batch)
                expected_logprobs, expected_ids = logprobs[1:].topk(diagnostics_topk, dim=-1)
                torch.testing.assert_close(teacher_ids[0, :3], expected_ids.to(teacher_ids.dtype))
                torch.testing.assert_close(teacher_logprobs[0, :3], expected_logprobs)
        else:
            assert cpu_teacher.calls == [
                ("teacher_score", max(2, diagnostics_topk) if rank_metrics else diagnostics_topk),
                ("teacher_target_score", 2),
            ]
        if adaptive:
            assert batch.non_tensor_batch["teacher_interventions"][0][0]["prune_future_cutoff"] == 1
            assert batch.batch["distillation_loss_mask"].tolist() == [[1, 0, 0, 0]]
            assert torch.count_nonzero(scores[0, 1:]) == 0

    torch.testing.assert_close(*rewards)
