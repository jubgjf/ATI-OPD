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

"""Committed-checkpoint discovery for the standalone evaluation watcher."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from verl.utils.evaluation.checkpoint_evaluation import (
    EVALUATION_SPEC_FILENAME,
    EVALUATION_STATE_FILENAME,
    load_evaluation_spec,
    load_evaluation_state,
    payload_fingerprint,
    write_json_atomic,
)

CHECKPOINT_PATTERN = re.compile(r"global_step_(\d+)")
FSDP_SHARD_PATTERN = re.compile(r"model_world_size_(\d+)_rank_(\d+)\.pt")
TRACKER_FILENAME = "latest_checkpointed_iteration.txt"
METRIC_COMMIT_FILENAME = "metrics_commit.json"
ENDPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CheckpointCandidate:
    step: int
    checkpoint_dir: str
    actor_dir: str
    evaluation_state: str
    hf_checkpoint: str | None
    needs_merge: bool


@dataclass(frozen=True)
class SearchCheckpointCandidate:
    """A committed checkpoint plus the experiment root that owns it."""

    step: int
    checkpoint_dir: str
    actor_dir: str
    evaluation_state: str
    hf_checkpoint: str | None
    needs_merge: bool
    checkpoint_root: str
    evaluation_spec: str
    experiment_name: str
    training_project: str
    training_run_id: str | None
    spec_fingerprint: str
    checkpoint_mtime_ns: int


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _tracker_step(checkpoint_root: Path) -> int | None:
    path = checkpoint_root / TRACKER_FILENAME
    try:
        value = int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    return value if value >= 0 else None


def _balanced_mode(spec: Mapping[str, Any]) -> bool:
    training = (spec.get("semantic_config") or {}).get("training") or {}
    return str(training.get("metric_logging_mode", "full")).lower() == "balanced"


def _hf_checkpoint_ready(path: Path) -> bool:
    if not path.is_dir():
        return False
    config = path / "config.json"
    tokenizer_config = path / "tokenizer_config.json"
    index_files = [
        candidate
        for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json")
        if (candidate := path / name).is_file()
    ]
    if index_files:
        required_weights: set[str] = set()
        try:
            for index_file in index_files:
                payload = json.loads(index_file.read_text(encoding="utf-8"))
                required_weights.update(str(value) for value in (payload.get("weight_map") or {}).values())
        except Exception:
            return False
        weights_ready = bool(required_weights) and all(
            (path / name).is_file() and (path / name).stat().st_size > 0 for name in required_weights
        )
    else:
        weights = list(path.glob("*.safetensors")) + list(path.glob("*.bin"))
        weights_ready = any(weight.is_file() and weight.stat().st_size > 0 for weight in weights)
    tokenizer_assets = [
        path / "tokenizer.json",
        path / "tokenizer.model",
        path / "spiece.model",
        path / "vocab.json",
        path / "vocab.txt",
    ]
    return (
        config.is_file()
        and tokenizer_config.is_file()
        and weights_ready
        and any(item.is_file() for item in tokenizer_assets)
    )


def _fsdp_actor_ready(actor_dir: Path) -> bool:
    shards: dict[int, Path] = {}
    world_size: int | None = None
    for path in actor_dir.glob("model_world_size_*_rank_*.pt"):
        match = FSDP_SHARD_PATTERN.fullmatch(path.name)
        if match is None or not path.is_file() or path.stat().st_size <= 0:
            continue
        shard_world_size, rank = map(int, match.groups())
        if shard_world_size <= 0 or rank < 0 or rank >= shard_world_size:
            return False
        if world_size is not None and shard_world_size != world_size:
            return False
        world_size = shard_world_size
        shards[rank] = path
    return world_size is not None and set(shards) == set(range(world_size))


def checkpoint_readiness(
    checkpoint_root: str | Path,
    checkpoint_dir: str | Path,
    *,
    spec: Mapping[str, Any] | None = None,
) -> tuple[CheckpointCandidate | None, str]:
    root = Path(checkpoint_root).expanduser().resolve()
    directory = Path(checkpoint_dir).expanduser().resolve()
    match = CHECKPOINT_PATTERN.fullmatch(directory.name)
    if match is None:
        return None, "checkpoint directory name is not global_step_<integer>"
    step = int(match.group(1))
    tracker_step = _tracker_step(root)
    if tracker_step is None or step > tracker_step:
        return None, "checkpoint is newer than the atomically committed root tracker"
    if not directory.is_dir() or not (directory / "data.pt").is_file():
        return None, "checkpoint data payload is incomplete"
    spec = dict(spec or load_evaluation_spec(root))
    state_path = directory / EVALUATION_STATE_FILENAME
    if not state_path.is_file():
        return None, "evaluation_state.json is missing"
    try:
        load_evaluation_state(
            directory,
            expected_step=step,
            expected_spec_fingerprint=str(spec["spec_fingerprint"]),
        )
    except Exception as exc:
        return None, f"evaluation state is invalid: {exc}"
    if _balanced_mode(spec):
        commit_path = directory / METRIC_COMMIT_FILENAME
        commit = _read_json(commit_path)
        if commit is None or int(commit.get("checkpoint_step", -1)) != step:
            return None, "balanced metrics_commit.json is missing or mismatched"
    actor_dir = directory / "actor"
    if not actor_dir.is_dir():
        return None, "actor checkpoint directory is missing"
    hf_dir = actor_dir / "huggingface"
    hf_ready = _hf_checkpoint_ready(hf_dir)
    if not hf_ready and not _fsdp_actor_ready(actor_dir):
        return None, "neither complete Hugging Face weights nor complete FSDP model shards are present"
    candidate = CheckpointCandidate(
        step=step,
        checkpoint_dir=str(directory),
        actor_dir=str(actor_dir),
        evaluation_state=str(state_path),
        hf_checkpoint=str(hf_dir) if hf_ready else None,
        needs_merge=not hf_ready,
    )
    return candidate, "ready"


def discover_committed_checkpoints(checkpoint_root: str | Path) -> list[CheckpointCandidate]:
    root = Path(checkpoint_root).expanduser().resolve()
    spec_path = root / EVALUATION_SPEC_FILENAME
    if not spec_path.is_file():
        return []
    try:
        spec = load_evaluation_spec(root)
    except Exception:
        return []
    candidates: list[CheckpointCandidate] = []
    for directory in root.glob("global_step_*"):
        candidate, _ = checkpoint_readiness(root, directory, spec=spec)
        if candidate is not None:
            candidates.append(candidate)
    return sorted(candidates, key=lambda item: item.step)


def pending_checkpoints(
    checkpoint_root: str | Path,
    *,
    completed_steps: Iterable[int],
) -> list[CheckpointCandidate]:
    completed = {int(step) for step in completed_steps}
    return [
        candidate for candidate in discover_committed_checkpoints(checkpoint_root) if candidate.step not in completed
    ]


def discover_experiment_roots(search_root: str | Path) -> list[Path]:
    """Find checkpoint roots below a training-output directory without walking model payloads."""

    root = Path(search_root).expanduser().resolve()
    if not root.is_dir():
        return []
    experiment_roots: list[Path] = []
    pruned_names = {
        ".eval-cache",
        ".standalone-eval",
        "actor",
        "critic",
        "huggingface",
        "wandb_logs",
    }
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        directory_names[:] = [
            name
            for name in directory_names
            if name not in pruned_names and not name.startswith("global_step_") and not name.startswith(".")
        ]
        if EVALUATION_SPEC_FILENAME in file_names:
            experiment_roots.append(current_path.resolve())
            directory_names[:] = []
    return sorted(set(experiment_roots))


def discover_search_root_checkpoints(search_root: str | Path) -> list[SearchCheckpointCandidate]:
    """Discover committed checkpoints across every manifest-enabled experiment below ``search_root``."""

    candidates: list[SearchCheckpointCandidate] = []
    for checkpoint_root in discover_experiment_roots(search_root):
        try:
            spec = load_evaluation_spec(checkpoint_root)
        except Exception:
            continue
        semantic = spec.get("semantic_config") or {}
        experiment_name = str(semantic.get("experiment_name") or checkpoint_root.name)
        training_project = str(semantic.get("training_project") or "DMPD")
        training_run_id = spec.get("training_run_id")
        for checkpoint in discover_committed_checkpoints(checkpoint_root):
            checkpoint_path = Path(checkpoint.checkpoint_dir)
            candidates.append(
                SearchCheckpointCandidate(
                    **asdict(checkpoint),
                    checkpoint_root=str(checkpoint_root),
                    evaluation_spec=str(checkpoint_root / EVALUATION_SPEC_FILENAME),
                    experiment_name=experiment_name,
                    training_project=training_project,
                    training_run_id=None if training_run_id is None else str(training_run_id),
                    spec_fingerprint=str(spec["spec_fingerprint"]),
                    checkpoint_mtime_ns=checkpoint_path.stat().st_mtime_ns,
                )
            )
    return sorted(
        candidates,
        key=lambda item: (item.checkpoint_mtime_ns, item.experiment_name, item.step, item.checkpoint_root),
    )


def write_endpoint_descriptor(
    path: str | Path,
    *,
    base_url: str,
    served_model_name: str,
    model_path: str | Path,
    request_id: str | None = None,
    model_load_seconds: float | None = None,
) -> dict[str, Any]:
    model_path = Path(model_path).expanduser().resolve()
    metadata_names = (
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
    )
    metadata = {
        name: (model_path / name).read_text(encoding="utf-8")
        for name in metadata_names
        if (model_path / name).is_file()
    }
    weight_files = sorted(
        {path for pattern in ("*.safetensors", "*.bin", "*.pt") for path in model_path.glob(pattern) if path.is_file()}
    )
    model_identity = {
        "path": str(model_path),
        "metadata": metadata,
        "weights": [
            {
                "name": path.name,
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in weight_files
        ],
    }
    payload = {
        "schema_version": ENDPOINT_SCHEMA_VERSION,
        "base_url": str(base_url).rstrip("/"),
        "served_model_name": str(served_model_name),
        "model_path": str(model_path),
        "model_fingerprint": payload_fingerprint(model_identity),
    }
    if request_id is not None:
        payload["request_id"] = str(request_id)
    if model_load_seconds is not None:
        payload["model_load_seconds"] = float(model_load_seconds)
    write_json_atomic(path, payload)
    return payload


def load_endpoint_descriptor(path: str | Path) -> dict[str, Any]:
    descriptor_path = Path(path)
    payload = _read_json(descriptor_path)
    if payload is None or payload.get("schema_version") != ENDPOINT_SCHEMA_VERSION:
        raise RuntimeError(f"Teacher endpoint descriptor is missing or invalid: {descriptor_path}.")
    required = {"base_url", "served_model_name", "model_path", "model_fingerprint"}
    missing = sorted(required - set(payload))
    if missing:
        raise RuntimeError(f"Teacher endpoint descriptor is missing fields {missing}: {descriptor_path}.")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    discover = subparsers.add_parser("discover")
    discover.add_argument("--checkpoint-root", required=True)
    discover.add_argument("--completed-step", type=int, action="append", default=[])
    discover_tree = subparsers.add_parser("discover-tree")
    discover_tree.add_argument("--search-root", required=True)
    endpoint = subparsers.add_parser("write-endpoint")
    endpoint.add_argument("--path", required=True)
    endpoint.add_argument("--base-url", required=True)
    endpoint.add_argument("--served-model-name", required=True)
    endpoint.add_argument("--model-path", required=True)
    endpoint.add_argument("--request-id")
    endpoint.add_argument("--model-load-seconds", type=float)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "discover":
        for candidate in pending_checkpoints(
            args.checkpoint_root,
            completed_steps=args.completed_step,
        ):
            print(json.dumps(asdict(candidate), sort_keys=True))
        return 0
    if args.command == "discover-tree":
        for candidate in discover_search_root_checkpoints(args.search_root):
            print(json.dumps(asdict(candidate), sort_keys=True))
        return 0
    if args.command == "write-endpoint":
        payload = write_endpoint_descriptor(
            args.path,
            base_url=args.base_url,
            served_model_name=args.served_model_name,
            model_path=args.model_path,
            request_id=args.request_id,
            model_load_seconds=args.model_load_seconds,
        )
        print(json.dumps(payload, sort_keys=True))
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
