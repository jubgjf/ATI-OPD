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

import numpy as np
from omegaconf import OmegaConf

from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def test_math_benchmark_metrics_use_test_core_and_mean_at_16_on_cpu():
    trainer = object.__new__(RayPPOTrainer)
    data_sources = np.array(["aime24"] * 16 + ["amc23"] * 16, dtype=object)
    sample_uids = ["aime24:0"] * 16 + ["amc23:0"] * 16
    acc = [1.0] * 4 + [0.0] * 12 + [1.0] * 8 + [0.0] * 8

    metrics = trainer._val_metrics_update(
        data_sources=data_sources,
        sample_uids=sample_uids,
        reward_extra_infos_dict={"acc": acc, "reward": acc},
        sample_turns=[],
        metric_prefix="test",
    )

    assert metrics["test-core/aime24/acc/mean@16"] == 0.25
    assert metrics["test-core/amc23/acc/mean@16"] == 0.5


def test_validate_test_benchmarks_uses_its_own_dataloader_and_sampling_mode_on_cpu():
    trainer = object.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "test_benchmarks_max_response_length": 128,
                "log_test_generations": 3,
                "test_data_dir": None,
            },
            "actor_rollout_ref": {"rollout": {"response_length": 256}},
        }
    )
    trainer.test_benchmark_dataloader = object()
    trainer.test_benchmark_generations_logger = object()
    captured = {}

    def _validate(**kwargs):
        captured.update(kwargs)
        return {"test-core/aime24/acc/mean@16": 0.5}

    trainer._validate = _validate

    metrics = trainer._validate_test_benchmarks()

    assert metrics == {"test-core/aime24/acc/mean@16": 0.5}
    assert captured["dataloader"] is trainer.test_benchmark_dataloader
    assert captured["evaluation_mode"] == "test"
    assert captured["metric_prefix"] == "test"
    assert captured["generation_max_tokens"] == 128
    assert captured["log_generations"] == 3
    assert captured["validation_data_dir"] == ""
