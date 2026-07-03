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
Preprocess the local open-r1 DAPO-Math-17k-Processed dataset to verl RL parquet format.

The open-r1 dataset is a deduplicated version of BytedTsinghua-SIA/DAPO-Math-17k.
It ships one train split with three configs (all/cn/en). This script validates
the schema, restores verl's chat-message prompt format, and creates a
deterministic train/test split when no test split is present.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import datasets

DATA_SOURCE = "open-r1/DAPO-Math-17k-Processed"
REWARD_DATA_SOURCE = "math_dapo"
DEFAULT_CONFIG_NAME = os.environ.get("DAPO_MATH_17K_CONFIG", "all")
DEFAULT_LOCAL_DATASET_PATH = os.environ.get("DAPO_MATH_17K_LOCAL_PATH", "~/data/DAPO-Math-17k-Processed")
PROMPT_STYLE_DAPO = "dapo"
PROMPT_STYLE_QWEN3_BOXED = "qwen3_boxed"
QWEN3_MATH_BOXED_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
QWEN3_PROMPT_CLEANUP_PATTERNS = (
    re.compile(
        r"\s*让我们一步一步地思考。请以[“\"]Answer:\s*\\boxed\{<final_answer>\}[”\"]的格式输出最终答案。"
        r"如果是选择题，请按顺序输出正确的选项，不带任何标点或空格。"
        r"对于其他类型的问题，请只输出最终答案的数值。?\s*$"
    ),
    re.compile(r"\s*Remember to put your answer on its own line after [\"“]Answer:[\"”]\.?\s*$"),
)


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


def _loads_prompt_field(value: Any, idx: int) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"DAPO-Math row {idx} has an empty prompt field.")
    if not stripped.startswith(("{", "[")):
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        # Raw problem text can start with markup like "[i]...[/i]".
        return value


def normalize_prompt(prompt: Any, idx: int) -> list[dict[str, str]]:
    """Normalize DAPO prompt variants to verl's single-turn chat-message schema."""
    prompt = _loads_prompt_field(prompt, idx)
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


def prompt_to_text(prompt: Any, idx: int) -> str:
    """Extract a single user text prompt from raw text or chat-message variants."""
    prompt = _loads_prompt_field(prompt, idx)
    if isinstance(prompt, str):
        text = prompt
    else:
        text = normalize_prompt(prompt, idx)[0]["content"]
    text = text.strip()
    if not text:
        raise ValueError(f"DAPO-Math row {idx} has an empty prompt.")
    return text


def clean_qwen3_math_problem_text(text: str) -> str:
    """Remove known answer-format suffixes before appending the Qwen3 boxed instruction."""
    cleaned = text.strip()
    changed = True
    while changed:
        changed = False
        for pattern in QWEN3_PROMPT_CLEANUP_PATTERNS:
            updated = pattern.sub("", cleaned).strip()
            if updated != cleaned:
                cleaned = updated
                changed = True
    return cleaned


def build_qwen3_boxed_prompt(prompt: Any, idx: int) -> list[dict[str, str]]:
    """Build Qwen3 math best-practice prompt from the raw problem text."""
    problem = clean_qwen3_math_problem_text(prompt_to_text(prompt, idx))
    content = f"{problem}\n\n{QWEN3_MATH_BOXED_INSTRUCTION}"
    return [{"role": "user", "content": content}]


def normalize_reward_model(reward_model: Any, idx: int) -> dict[str, Any]:
    reward_model = _loads_json_field(reward_model, "reward_model", idx)
    if not isinstance(reward_model, dict):
        raise ValueError(f"DAPO-Math row {idx} has unsupported reward_model type: {type(reward_model).__name__}.")
    if not str(reward_model.get("ground_truth", "")).strip():
        raise ValueError(f"DAPO-Math row {idx} must contain reward_model.ground_truth.")
    return dict(reward_model)


def normalize_extra_info(extra_info: Any, idx: int) -> dict[str, Any]:
    if extra_info is None:
        return {}
    extra_info = _loads_json_field(extra_info, "extra_info", idx)
    if not isinstance(extra_info, dict):
        raise ValueError(f"DAPO-Math row {idx} has unsupported extra_info type: {type(extra_info).__name__}.")
    return dict(extra_info)


def _build_reward_model_from_answer(example: dict[str, Any], idx: int) -> dict[str, Any]:
    reward_model = example.get("reward_model")
    if reward_model is not None:
        return normalize_reward_model(reward_model, idx)

    for field_name in ("solution", "answer", "ground_truth", "final_answer"):
        answer = example.get(field_name)
        if answer is not None and str(answer).strip():
            return {"style": "rule", "ground_truth": str(answer)}

    raise ValueError(
        f"DAPO-Math row {idx} must contain reward_model.ground_truth or one of "
        "solution/answer/ground_truth/final_answer."
    )


def _select_prompt_value(example: dict[str, Any], idx: int, prompt_style: str = PROMPT_STYLE_DAPO) -> Any:
    # open-r1 keeps the original DAPO chat prompt in source_prompt. The prompt
    # column is a raw problem string intended for TRL-style trainers.
    if prompt_style == PROMPT_STYLE_QWEN3_BOXED:
        field_names = ("prompt", "problem", "question")
    elif prompt_style == PROMPT_STYLE_DAPO:
        field_names = ("source_prompt", "prompt", "problem", "question")
    else:
        raise ValueError(
            f"prompt_style must be one of {PROMPT_STYLE_DAPO!r}/{PROMPT_STYLE_QWEN3_BOXED!r}, got {prompt_style!r}."
        )

    for field_name in field_names:
        prompt = example.get(field_name)
        if prompt is not None:
            return prompt

    raise ValueError(f"DAPO-Math row {idx} must contain one of {', '.join(field_names)}.")


def _load_parquet_files(path: Path, config_name: str) -> datasets.DatasetDict:
    if path.is_file():
        parquet_files = [str(path)]
    else:
        preferred_files = [
            path / config_name / "train-00000-of-00001.parquet",
            path / config_name / "train" / "0000.parquet",
            path / "data" / "dapo-math-17k.parquet",
        ]
        parquet_files = [str(p) for p in preferred_files if p.exists()]
        if not parquet_files:
            all_parquet_files = sorted(path.rglob("*.parquet"))
            matched_config_files = [p for p in all_parquet_files if config_name in p.parts]
            parquet_files = [str(p) for p in (matched_config_files or all_parquet_files)]

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under local DAPO dataset path: {path}")

    return datasets.load_dataset("parquet", data_files={"train": parquet_files})


def load_dapo_dataset(
    local_dataset_path: str,
    download: bool = False,
    config_name: str = DEFAULT_CONFIG_NAME,
) -> datasets.DatasetDict:
    """Load DAPO-Math-17k from local files by default.

    Use ``--download`` to explicitly fall back to Hugging Face. If network
    access is needed on the cluster, set ``HF_ENDPOINT=https://hf-mirror.com``.
    """
    if config_name not in {"all", "cn", "en"}:
        raise ValueError(f"config_name must be one of all/cn/en for {DATA_SOURCE}, got {config_name!r}.")

    dataset_path = _expand_path(local_dataset_path)
    if dataset_path.exists():
        if (dataset_path / "dataset_dict.json").exists() or (dataset_path / "state.json").exists():
            loaded = datasets.load_from_disk(str(dataset_path))
            return loaded if isinstance(loaded, datasets.DatasetDict) else datasets.DatasetDict({"train": loaded})

        if dataset_path.is_file() or any(dataset_path.rglob("*.parquet")):
            return _load_parquet_files(dataset_path, config_name=config_name)

        try:
            return datasets.load_dataset(str(dataset_path), config_name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load local DAPO dataset from {dataset_path}. Expected a Hugging Face dataset directory, "
                "a dataset saved with save_to_disk, or parquet files such as all/train-00000-of-00001.parquet."
            ) from exc

    if download:
        hf_endpoint = os.environ.get("HF_ENDPOINT")
        if hf_endpoint:
            print(f"HF_ENDPOINT={hf_endpoint}", flush=True)
        return datasets.load_dataset(DATA_SOURCE, config_name)

    raise FileNotFoundError(
        f"Local DAPO dataset path does not exist: {dataset_path}. Download it first, or rerun with --download. "
        "For mirror downloads, export HF_ENDPOINT=https://hf-mirror.com."
    )


def validate_and_tag_dataset(
    dataset: datasets.Dataset, split: str, prompt_style: str = PROMPT_STYLE_DAPO
) -> datasets.Dataset:
    def process_fn(example: dict[str, Any], idx: int) -> dict[str, Any]:
        source_data_source = example.get("data_source")
        if source_data_source not in (None, REWARD_DATA_SOURCE):
            raise ValueError(
                f"DAPO-Math row {idx} has unsupported data_source={source_data_source!r}; "
                f"expected {REWARD_DATA_SOURCE!r} or a missing value."
            )

        prompt_value = _select_prompt_value(example, idx, prompt_style=prompt_style)
        if prompt_style == PROMPT_STYLE_QWEN3_BOXED:
            prompt = build_qwen3_boxed_prompt(prompt_value, idx)
        else:
            prompt = normalize_prompt(prompt_value, idx)
        reward_model = _build_reward_model_from_answer(example, idx)

        extra_info = normalize_extra_info(example.get("extra_info"), idx)
        extra_info.setdefault("index", str(idx))
        extra_info["split"] = split
        extra_info["source_dataset"] = DATA_SOURCE
        extra_info["prompt_style"] = prompt_style
        if source_data_source is not None:
            extra_info["source_data_source"] = source_data_source

        return {
            "data_source": REWARD_DATA_SOURCE,
            "prompt": prompt,
            "ability": example.get("ability") or "MATH",
            "reward_model": reward_model,
            "extra_info": extra_info,
        }

    return dataset.map(
        function=process_fn,
        with_indices=True,
        remove_columns=dataset.column_names,
        desc=f"Validating {split} split",
    )


def build_train_test_split(
    dataset_dict: datasets.DatasetDict,
    test_size: float,
    seed: int,
    prompt_style: str = PROMPT_STYLE_DAPO,
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

    return (
        validate_and_tag_dataset(train_dataset, "train", prompt_style=prompt_style),
        validate_and_tag_dataset(test_dataset, "test", prompt_style=prompt_style),
    )


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
    config_name: str,
    prompt_style: str,
    test_size: float,
    seed: int,
) -> None:
    info = {
        "source": DATA_SOURCE,
        "config_name": config_name,
        "prompt_style": prompt_style,
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
        help="Path to an already downloaded open-r1/DAPO-Math-17k-Processed dataset directory or parquet file.",
    )
    parser.add_argument(
        "--local_save_dir",
        default="~/data/dapo-math-17k-processed-verl",
        help="The save directory for the preprocessed train/test parquet files.",
    )
    parser.add_argument(
        "--prompt_style",
        default=PROMPT_STYLE_DAPO,
        choices=(PROMPT_STYLE_DAPO, PROMPT_STYLE_QWEN3_BOXED),
        help=(
            "Prompt template to emit. 'dapo' preserves open-r1 source_prompt when available. "
            "'qwen3_boxed' uses the raw prompt/problem text and appends Qwen3's boxed-answer math instruction."
        ),
    )
    parser.add_argument(
        "--config_name",
        default=DEFAULT_CONFIG_NAME,
        choices=("all", "cn", "en"),
        help="Dataset config to preprocess. Use 'all' for the deduplicated 17.4k full set.",
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

    dataset = load_dapo_dataset(args.local_dataset_path, download=args.download, config_name=args.config_name)
    train_dataset, test_dataset = build_train_test_split(
        dataset, test_size=args.test_size, seed=args.seed, prompt_style=args.prompt_style
    )

    train_path = local_save_path / "train.parquet"
    test_path = local_save_path / "test.parquet"
    train_dataset.to_parquet(str(train_path))
    test_dataset.to_parquet(str(test_path))
    write_examples(local_save_path, train_dataset, test_dataset)
    write_dataset_info(
        local_save_path,
        train_dataset,
        test_dataset,
        args.local_dataset_path,
        args.config_name,
        args.prompt_style,
        args.test_size,
        args.seed,
    )

    print(f"Saved DAPO-Math train split to {train_path} ({len(train_dataset)} rows)", flush=True)
    print(f"Saved DAPO-Math test split to {test_path} ({len(test_dataset)} rows)", flush=True)

    if args.hdfs_dir is not None:
        from verl.utils.hdfs_io import copy, makedirs

        makedirs(args.hdfs_dir)
        copy(src=str(local_save_path), dst=args.hdfs_dir)
