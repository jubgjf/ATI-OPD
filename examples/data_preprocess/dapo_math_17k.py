# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Preprocess the local DAPO-Math-17k dataset to verl single-turn RL parquet format.

The HF dataset already follows verl's RL schema. This script validates the
schema, adds split metadata, and creates a deterministic train/test split
because DAPO-Math-17k only ships a train split.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import datasets

DATA_SOURCE = "BytedTsinghua-SIA/DAPO-Math-17k"
DEFAULT_LOCAL_DATASET_PATH = os.environ.get("DAPO_MATH_17K_LOCAL_PATH", "~/data/DAPO-Math-17k")


def _expand_path(path: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(path)))


def _loads_json_field(value: Any, field_name: str, idx: int) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"DAPO-Math row {idx} has an empty {field_name} field.")
    if not stripped.startswith(("{", "[")):
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"DAPO-Math row {idx} has invalid JSON in {field_name}: {value[:200]!r}") from exc


def normalize_prompt(prompt: Any, idx: int) -> list[dict[str, str]]:
    """Normalize DAPO prompt variants to verl's single-turn chat-message schema."""
    prompt = _loads_json_field(prompt, "prompt", idx)
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    elif isinstance(prompt, dict):
        messages = [prompt]
    elif isinstance(prompt, list):
        messages = prompt
    else:
        raise ValueError(f"DAPO-Math row {idx} has unsupported prompt type: {type(prompt).__name__}.")

    if len(messages) != 1:
        raise ValueError(f"DAPO-Math row {idx} must contain a single-turn prompt, got {len(messages)} messages.")
    message = messages[0]
    if not isinstance(message, dict):
        raise ValueError(f"DAPO-Math row {idx} prompt message must be a dict, got {type(message).__name__}.")
    role = message.get("role", "user")
    content = message.get("content")
    if role != "user" or not isinstance(content, str) or not content.strip():
        raise ValueError(f"DAPO-Math row {idx} has invalid prompt message: {message!r}.")
    return [{"role": "user", "content": content}]


def normalize_reward_model(reward_model: Any, idx: int) -> dict[str, Any]:
    reward_model = _loads_json_field(reward_model, "reward_model", idx)
    if not isinstance(reward_model, dict):
        raise ValueError(f"DAPO-Math row {idx} has unsupported reward_model type: {type(reward_model).__name__}.")
    if not str(reward_model.get("ground_truth", "")).strip():
        raise ValueError(f"DAPO-Math row {idx} must contain reward_model.ground_truth.")
    return dict(reward_model)


def _load_parquet_files(path: Path) -> datasets.DatasetDict:
    if path.is_file():
        parquet_files = [str(path)]
    else:
        preferred_file = path / "data" / "dapo-math-17k.parquet"
        if preferred_file.exists():
            parquet_files = [str(preferred_file)]
        else:
            parquet_files = sorted(str(p) for p in path.rglob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under local DAPO dataset path: {path}")

    return datasets.load_dataset("parquet", data_files={"train": parquet_files})


def load_dapo_dataset(local_dataset_path: str, download: bool = False) -> datasets.DatasetDict:
    """Load DAPO-Math-17k from local files by default.

    Use ``--download`` to explicitly fall back to Hugging Face. If network
    access is needed on the cluster, set ``HF_ENDPOINT=https://hf-mirror.com``.
    """
    dataset_path = _expand_path(local_dataset_path)
    if dataset_path.exists():
        if (dataset_path / "dataset_dict.json").exists() or (dataset_path / "state.json").exists():
            loaded = datasets.load_from_disk(str(dataset_path))
            return loaded if isinstance(loaded, datasets.DatasetDict) else datasets.DatasetDict({"train": loaded})

        if dataset_path.is_file() or any(dataset_path.rglob("*.parquet")):
            return _load_parquet_files(dataset_path)

        try:
            return datasets.load_dataset(str(dataset_path), "default")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load local DAPO dataset from {dataset_path}. Expected a Hugging Face dataset directory, "
                "a dataset saved with save_to_disk, or parquet files such as data/dapo-math-17k.parquet."
            ) from exc

    if download:
        hf_endpoint = os.environ.get("HF_ENDPOINT")
        if hf_endpoint:
            print(f"HF_ENDPOINT={hf_endpoint}", flush=True)
        return datasets.load_dataset(DATA_SOURCE, "default")

    raise FileNotFoundError(
        f"Local DAPO dataset path does not exist: {dataset_path}. Download it first, or rerun with --download. "
        "For mirror downloads, export HF_ENDPOINT=https://hf-mirror.com."
    )


def validate_and_tag_dataset(dataset: datasets.Dataset, split: str) -> datasets.Dataset:
    def process_fn(example: dict[str, Any], idx: int) -> dict[str, Any]:
        data_source = example.get("data_source")
        if data_source != "math_dapo":
            raise ValueError(f"DAPO-Math row {idx} has unsupported data_source={data_source!r}; expected 'math_dapo'.")

        prompt = normalize_prompt(example.get("prompt"), idx)
        reward_model = normalize_reward_model(example.get("reward_model"), idx)

        extra_info = dict(example.get("extra_info") or {})
        extra_info.setdefault("index", str(idx))
        extra_info["split"] = split
        extra_info["source_dataset"] = DATA_SOURCE

        return {
            "data_source": data_source,
            "prompt": prompt,
            "ability": example.get("ability", "MATH"),
            "reward_model": reward_model,
            "extra_info": extra_info,
        }

    return dataset.map(function=process_fn, with_indices=True, desc=f"Validating {split} split")


def build_train_test_split(
    dataset_dict: datasets.DatasetDict,
    test_size: float,
    seed: int,
) -> tuple[datasets.Dataset, datasets.Dataset]:
    if "train" not in dataset_dict:
        raise ValueError(f"DAPO-Math dataset must contain a train split; found splits: {list(dataset_dict.keys())}")

    if "test" in dataset_dict:
        train_dataset = dataset_dict["train"]
        test_dataset = dataset_dict["test"]
    else:
        if not 0 < test_size < 1:
            raise ValueError(f"test_size must be a fraction in (0, 1) when creating a split, got {test_size}.")
        split = dataset_dict["train"].train_test_split(test_size=test_size, seed=seed, shuffle=True)
        train_dataset = split["train"]
        test_dataset = split["test"]

    if len(train_dataset) == 0 or len(test_dataset) == 0:
        raise ValueError(f"DAPO-Math split is empty: train={len(train_dataset)}, test={len(test_dataset)}")

    return validate_and_tag_dataset(train_dataset, "train"), validate_and_tag_dataset(test_dataset, "test")


def write_examples(local_save_dir: Path, train_dataset: datasets.Dataset, test_dataset: datasets.Dataset) -> None:
    with open(local_save_dir / "train_example.json", "w") as f:
        json.dump(train_dataset[0], f, indent=2, ensure_ascii=False)
    with open(local_save_dir / "test_example.json", "w") as f:
        json.dump(test_dataset[0], f, indent=2, ensure_ascii=False)


def write_dataset_info(
    local_save_dir: Path,
    train_dataset: datasets.Dataset,
    test_dataset: datasets.Dataset,
    local_dataset_path: str,
    test_size: float,
    seed: int,
) -> None:
    info = {
        "source": DATA_SOURCE,
        "local_dataset_path": str(_expand_path(local_dataset_path)),
        "train_rows": len(train_dataset),
        "test_rows": len(test_dataset),
        "test_size": test_size,
        "seed": seed,
        "schema": list(train_dataset.features.keys()),
    }
    with open(local_save_dir / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="Deprecated. Use --local_save_dir instead.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--local_dataset_path",
        default=DEFAULT_LOCAL_DATASET_PATH,
        help="Path to an already downloaded DAPO-Math-17k dataset directory or parquet file.",
    )
    parser.add_argument(
        "--local_save_dir",
        default="~/data/dapo-math-17k-verl",
        help="The save directory for the preprocessed train/test parquet files.",
    )
    parser.add_argument("--test_size", type=float, default=0.2, help="Held-out fraction when no test split exists.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic train/test splitting.")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Explicitly download from Hugging Face if --local_dataset_path does not exist.",
    )

    args = parser.parse_args()

    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    local_save_path = _expand_path(local_save_dir)
    local_save_path.mkdir(parents=True, exist_ok=True)

    dataset = load_dapo_dataset(args.local_dataset_path, download=args.download)
    train_dataset, test_dataset = build_train_test_split(dataset, test_size=args.test_size, seed=args.seed)

    train_path = local_save_path / "train.parquet"
    test_path = local_save_path / "test.parquet"
    train_dataset.to_parquet(str(train_path))
    test_dataset.to_parquet(str(test_path))
    write_examples(local_save_path, train_dataset, test_dataset)
    write_dataset_info(local_save_path, train_dataset, test_dataset, args.local_dataset_path, args.test_size, args.seed)

    print(f"Saved DAPO-Math train split to {train_path} ({len(train_dataset)} rows)", flush=True)
    print(f"Saved DAPO-Math test split to {test_path} ({len(test_dataset)} rows)", flush=True)

    if args.hdfs_dir is not None:
        from verl.utils.hdfs_io import copy, makedirs

        makedirs(args.hdfs_dir)
        copy(src=str(local_save_path), dst=args.hdfs_dir)
