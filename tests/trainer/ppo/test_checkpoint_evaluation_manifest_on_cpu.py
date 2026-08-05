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

from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.evaluation.checkpoint_evaluation import load_evaluation_spec, load_evaluation_state


def _trainer(tmp_path: Path, *, outcome_aware: bool = False) -> RayPPOTrainer:
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "checkpoint_evaluation_manifest": True,
                "default_local_dir": str(tmp_path),
                "experiment_name": "manifest-test",
                "project_name": "DMPD",
                "metric_logging": {"mode": "balanced"},
                "save_freq": 20,
                "max_actor_ckpt_to_keep": None,
                "test_benchmarks_max_response_length": 64,
            },
            "actor_rollout_ref": {
                "model": {"path": "/student", "tokenizer_path": None, "chat_template_from_model": "/teacher"},
                "rollout": {
                    "max_model_len": 96,
                    "test_kwargs": {"n": 16, "temperature": 0.7, "top_p": 0.95, "top_k": -1},
                },
            },
            "data": {
                "trust_remote_code": False,
                "max_prompt_length": 32,
                "max_response_length": 64,
                "seed": 42,
                "test_benchmarks": {
                    "root": "/datasets",
                    "names": ["math-ai/aime24"],
                    "reward_data_source": "math_verify",
                },
            },
            "reward": {"reward_manager": {"name": "naive"}},
            "distillation": {
                "enabled": True,
                "teacher_key": "default",
                "teacher_models": {"default": {"model_path": "/teacher"}},
                "distillation_loss": {"log_prob_min_clamp": -10.0},
                "intervention": {
                    "enabled": True,
                    "gap_metric": "k3",
                    "outcome_aware_candidate_selection_enabled": outcome_aware,
                },
            },
        }
    )
    trainer._logical_run_id = "logical-run"
    trainer._checkpoint_evaluation_spec = None
    trainer.global_steps = 20
    return trainer


def test_trainer_writes_spec_and_non_outcome_checkpoint_state_on_cpu(tmp_path):
    trainer = _trainer(tmp_path)
    logger = SimpleNamespace(wandb_run_info=SimpleNamespace(run_id="wandb-run"))

    trainer._initialize_checkpoint_evaluation_manifest(logger)
    checkpoint = tmp_path / "global_step_20"
    trainer._write_checkpoint_evaluation_state(str(checkpoint))

    spec = load_evaluation_spec(tmp_path)
    state = load_evaluation_state(
        checkpoint,
        expected_step=20,
        expected_spec_fingerprint=spec["spec_fingerprint"],
    )
    assert spec["training_run_id"] == "wandb-run"
    assert state["teacher_nll_calibration"] is None


def test_trainer_requires_and_persists_outcome_calibration_on_cpu(tmp_path):
    trainer = _trainer(tmp_path, outcome_aware=True)
    logger = SimpleNamespace(wandb_run_info=None)
    trainer._initialize_checkpoint_evaluation_manifest(logger)
    checkpoint = tmp_path / "global_step_20"

    trainer.async_rollout_manager = SimpleNamespace(
        export_teacher_nll_calibration_state=lambda: {
            "history": [0.2, 0.3],
            "threshold": 0.3,
            "source": "rolling_healthy_quantile",
            "count": 2,
            "version": 1,
        }
    )
    trainer._write_checkpoint_evaluation_state(str(checkpoint))
    state = load_evaluation_state(
        checkpoint,
        expected_step=20,
        expected_spec_fingerprint=trainer._checkpoint_evaluation_spec["spec_fingerprint"],
    )
    assert state["teacher_nll_calibration"]["threshold"] == 0.3

    trainer.async_rollout_manager = SimpleNamespace()
    with pytest.raises(RuntimeError, match="manager-owned teacher-NLL calibration"):
        trainer._write_checkpoint_evaluation_state(str(tmp_path / "global_step_40"))
