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

from omegaconf import OmegaConf

from verl.trainer.ppo.ray_trainer import RayPPOTrainer, _stable_evaluation_rollout_seed
from verl.utils.evaluation.multi_domain_benchmark import BenchmarkProfile


def _spec(name, adapter, prompt, response, legacy_n, train_n):
    return SimpleNamespace(
        name=name,
        metric_name=name,
        adapter=adapter,
        prompt_max_length=prompt,
        generation_max_tokens=response,
        profiles={
            "legacy": BenchmarkProfile(n=legacy_n),
            "train_sampling": BenchmarkProfile(n=train_n),
        },
    )


def test_typed_benchmarks_use_per_entry_sampling_and_tensor_budgets_on_cpu():
    trainer = object.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {"log_test_generations": 0, "test_data_dir": None},
            "data": {"test_benchmarks": {"seed": 42}},
            "actor_rollout_ref": {
                "rollout": {
                    "benchmark_sampling_profiles": {
                        "legacy": {
                            "temperature": 0.7,
                            "top_p": 0.95,
                            "top_k": 20,
                            "do_sample": True,
                            "n": 16,
                        },
                        "train_sampling": {
                            "temperature": 1.0,
                            "top_p": 1.0,
                            "top_k": -1,
                            "do_sample": True,
                            "n": 16,
                        },
                    }
                }
            },
        }
    )
    specs = {
        "ifbench": _spec("ifbench", "ifbench", 4096, 4096, 1, 8),
        "longbench-v2": _spec("longbench-v2", "longbench_v2", 31744, 1024, 1, 4),
    }
    trainer.test_benchmark_dataloaders = {name: object() for name in specs}
    trainer.test_benchmark_datasets = {name: SimpleNamespace(spec=spec) for name, spec in specs.items()}
    trainer.test_benchmark_generations_loggers = {"legacy": object(), "train_sampling": object()}
    captured = []

    def _validate(**kwargs):
        captured.append(kwargs)
        benchmark = next(
            name for name, loader in trainer.test_benchmark_dataloaders.items() if loader is kwargs["dataloader"]
        )
        n = int(kwargs["sampling_config"].n)
        return (
            {f"{kwargs['metric_prefix']}-core/{benchmark}/acc/mean@{n}": 0.5},
            {
                "sample_uids": [f"{benchmark}:1"] * n,
                "reward_extra_infos_dict": {
                    "acc": [0.5] * n,
                    "category": ["test"] * n,
                    "length_bucket": ["long"] * n,
                },
                "data_sources": [benchmark] * n,
            },
        )

    trainer._validate = _validate

    metrics = trainer._validate_test_benchmarks()

    assert len(captured) == 4
    by_case = {(call["dump_filename_prefix"], call["sampling_profile_name"]): call for call in captured}
    legacy_if = by_case[("test_benchmarks_ifbench_legacy", "legacy")]
    train_if = by_case[("test_benchmarks_ifbench_train_sampling", "train_sampling")]
    train_long = by_case[("test_benchmarks_longbench-v2_train_sampling", "train_sampling")]
    assert int(legacy_if["sampling_config"].n) == 1
    assert legacy_if["sampling_config"].do_sample is False
    assert float(legacy_if["sampling_config"].temperature) == 0.0
    assert int(train_if["sampling_config"].n) == 8
    assert train_if["evaluation_prompt_length"] == 4096
    assert train_if["evaluation_response_length"] == 4096
    assert int(train_long["sampling_config"].n) == 4
    assert train_long["evaluation_prompt_length"] == 31744
    assert train_long["generation_max_tokens"] == 1024
    assert train_long["evaluation_seed"] == 42
    assert train_long["evaluation_seed_namespace"] == "longbench-v2"
    assert "test-train-sampling-core/longbench-v2/acc/mean@4" in metrics


def test_typed_benchmark_rollout_seed_uses_full_stable_namespace_on_cpu():
    base = {
        "suite_seed": 42,
        "benchmark_name": "gpqa-diamond",
        "sample_uid": "gpqa-diamond:record-1",
        "rollout_index": 3,
        "profile_name": "train_sampling",
    }
    seed = _stable_evaluation_rollout_seed(**base)

    assert seed == _stable_evaluation_rollout_seed(**base)
    for field, replacement in (
        ("suite_seed", 43),
        ("benchmark_name", "ifbench"),
        ("sample_uid", "gpqa-diamond:record-2"),
        ("rollout_index", 4),
        ("profile_name", "legacy"),
    ):
        changed = {**base, field: replacement}
        assert _stable_evaluation_rollout_seed(**changed) != seed
