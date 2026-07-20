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

"""Dataset adapter for running local math benchmarks through PPO validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from verl.utils.evaluation.math_benchmark import build_math_chat_messages, load_benchmark_samples


def _normalize_benchmark_path(raw_path: str) -> str:
    benchmark_path = Path(raw_path)
    if benchmark_path.is_absolute():
        raise ValueError(f"MathBenchmarkDataset benchmark paths must be relative to dataset_root, got {raw_path!r}.")
    if benchmark_path == Path(".") or ".." in benchmark_path.parts:
        raise ValueError(
            "MathBenchmarkDataset benchmark paths must name a directory under dataset_root without '..', "
            f"got {raw_path!r}."
        )
    return benchmark_path.as_posix()


@dataclass(frozen=True)
class _BenchmarkRecord:
    benchmark: str
    benchmark_path: str
    sample_id: str
    problem: str
    ground_truth: str
    index: int


class MathBenchmarkDataset(Dataset):
    """Normalize local AIME/AMC-style datasets to verl's validation row schema.

    Each configured benchmark is a path relative to ``dataset_root``. Its final
    path component remains the public ``data_source`` so a path such as
    ``math-ai/aime24`` is loaded from that nested directory while validation
    metrics are grouped under keys such as ``test-core/aime24/...``. The reward
    implementation is selected independently through ``extra_info.reward_data_source``.
    """

    def __init__(
        self,
        *,
        dataset_root: str | Path,
        benchmarks: Sequence[str],
        reward_data_source: str = "math_dapo",
        max_samples_per_benchmark: int = -1,
    ) -> None:
        root = Path(dataset_root).expanduser()
        raw_benchmark_paths = [str(path).strip() for path in benchmarks if str(path).strip()]
        if not raw_benchmark_paths:
            raise ValueError("MathBenchmarkDataset requires at least one benchmark path.")
        benchmark_paths = [_normalize_benchmark_path(path) for path in raw_benchmark_paths]
        if len(set(benchmark_paths)) != len(benchmark_paths):
            raise ValueError(f"MathBenchmarkDataset benchmark paths must be unique, got {benchmark_paths!r}.")

        benchmark_names = [Path(path).name for path in benchmark_paths]
        duplicate_names = sorted({name for name in benchmark_names if benchmark_names.count(name) > 1})
        if duplicate_names:
            raise ValueError(
                "MathBenchmarkDataset benchmark paths must have unique final directory names for metric grouping; "
                f"duplicate names: {duplicate_names!r}."
            )
        if not reward_data_source:
            raise ValueError("MathBenchmarkDataset reward_data_source must be non-empty.")
        if max_samples_per_benchmark == 0 or max_samples_per_benchmark < -1:
            raise ValueError(
                "MathBenchmarkDataset max_samples_per_benchmark must be -1 or a positive integer, "
                f"got {max_samples_per_benchmark}."
            )

        self.dataset_root = root
        self.benchmarks = tuple(benchmark_paths)
        self.benchmark_names = tuple(benchmark_names)
        self.reward_data_source = reward_data_source
        self.max_samples_per_benchmark = max_samples_per_benchmark
        self.records: list[_BenchmarkRecord] = []

        max_samples = None if max_samples_per_benchmark < 0 else max_samples_per_benchmark
        for benchmark_path, benchmark_name in zip(self.benchmarks, self.benchmark_names, strict=True):
            samples = load_benchmark_samples(root, benchmark_path, max_samples=max_samples)
            if not samples:
                raise ValueError(f"Math benchmark {benchmark_name!r} has no samples under {root / benchmark_path}.")
            for sample in samples:
                self.records.append(
                    _BenchmarkRecord(
                        benchmark=benchmark_name,
                        benchmark_path=benchmark_path,
                        sample_id=sample.sample_id,
                        problem=sample.problem,
                        ground_truth=sample.ground_truth,
                        index=len(self.records),
                    )
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, item: int) -> dict[str, Any]:
        record = self.records[item]
        uid = f"{record.benchmark}:{record.sample_id}"
        return {
            "raw_prompt": build_math_chat_messages(record.problem, idx=record.index),
            "data_source": record.benchmark,
            "reward_model": {"style": "rule", "ground_truth": record.ground_truth},
            "extra_info": {
                "index": record.index,
                "benchmark": record.benchmark,
                "benchmark_path": record.benchmark_path,
                "sample_id": record.sample_id,
                "reward_data_source": self.reward_data_source,
            },
            "uid": uid,
            "index": record.index,
            "tools_kwargs": {},
            "interaction_kwargs": {},
            "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
        }
