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

"""Dataset wrapper for one typed multi-domain benchmark entry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import Dataset

from verl.utils.evaluation.multi_domain_benchmark import BenchmarkSpec, load_benchmark_records


class MultiDomainBenchmarkDataset(Dataset):
    """Expose normalized benchmark records through verl's validation row contract."""

    def __init__(
        self,
        *,
        dataset_root: str | Path,
        spec: BenchmarkSpec,
        tokenizer: Any,
        seed: int = 42,
        chat_template_kwargs: Mapping[str, Any] | None = None,
        agent_name: str | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root).expanduser()
        self.spec = spec
        self.seed = int(seed)
        self.agent_name = str(agent_name).strip() if agent_name is not None else None
        if self.agent_name == "":
            raise ValueError("MultiDomainBenchmarkDataset agent_name must be non-empty when provided.")
        self.records = load_benchmark_records(
            dataset_root=self.dataset_root,
            spec=spec,
            tokenizer=tokenizer,
            seed=self.seed,
            chat_template_kwargs=chat_template_kwargs,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, item: int) -> dict[str, Any]:
        record = self.records[item]
        extra_info = {
            "index": item,
            "benchmark": self.spec.name,
            "benchmark_adapter": self.spec.adapter,
            "benchmark_path": self.spec.path,
            "reward_data_source": "multi_domain_benchmark",
            "evaluation_prompt_length": self.spec.prompt_max_length,
            "evaluation_response_length": self.spec.generation_max_tokens,
        }
        for key in (
            "benchmark",
            "category",
            "length_bucket",
            "original_prompt_tokens",
            "original_prompt_tokens_exact",
            "formatted_prompt_tokens",
            "original_context_tokens",
            "retained_context_tokens",
            "retained_ratio",
            "truncated",
        ):
            if key in record.payload:
                extra_info[key] = record.payload[key]
        row = {
            "raw_prompt": record.messages,
            "data_source": self.spec.metric_name,
            "reward_model": {
                "style": "rule",
                "ground_truth": json.dumps(record.payload, ensure_ascii=False, sort_keys=True),
            },
            "extra_info": extra_info,
            "uid": record.uid,
            "index": item,
            "tools_kwargs": {},
            "interaction_kwargs": {},
            "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
        }
        if self.agent_name is not None:
            row["agent_name"] = self.agent_name
        return row


__all__ = ["MultiDomainBenchmarkDataset"]
