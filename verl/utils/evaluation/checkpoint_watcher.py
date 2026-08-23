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

"""Committed-checkpoint discovery and shared-storage claims for standalone evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import sys
import time
import uuid
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
from verl.utils.evaluation.model_identity import (
    capture_stable_model_manifest,
    legacy_stat_model_fingerprint,
    model_manifest_cache_path,
    register_model_fingerprint_aliases,
    validate_stable_model_manifest,
)

CHECKPOINT_PATTERN = re.compile(r"global_step_(\d+)")
FSDP_SHARD_PATTERN = re.compile(r"model_world_size_(\d+)_rank_(\d+)\.pt")
TRACKER_FILENAME = "latest_checkpointed_iteration.txt"
METRIC_COMMIT_FILENAME = "metrics_commit.json"
ENDPOINT_SCHEMA_VERSION = 2
CLAIM_SCHEMA_VERSION = 1
CLAIMS_DIRNAME = "claims"
CLAIM_OWNER_FILENAME = "owner.json"
CLAIM_HEARTBEAT_FILENAME = "heartbeat.json"
CLAIM_TYPES = ("checkpoint", "role", "wandb")
SOURCE_CHECKPOINT_IDENTITY_SCHEMA_VERSION = 1
SCHEDULER_METADATA_ENV = (
    "MA_VJ_NAME",
    "MA_TASK_NAME",
    "VC_WORKER_NUM",
    "VC_TASK_INDEX",
)


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
    source_checkpoint_identity: str
    checkpoint_mtime_ns: int


def _resource_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def checkpoint_claim_path(
    runtime_root: str | Path,
    checkpoint_root: str | Path,
    step: int,
) -> Path:
    if int(step) < 0:
        raise ValueError(f"Checkpoint step must be non-negative, got {step}.")
    root = Path(checkpoint_root).expanduser().resolve()
    root_hash = _resource_hash(str(root))
    return (
        Path(runtime_root).expanduser().resolve()
        / CLAIMS_DIRNAME
        / "checkpoints"
        / root_hash
        / f"global_step_{step}.claim"
    )


def role_claim_path(runtime_root: str | Path, instance_id: str, role: str) -> Path:
    if not instance_id:
        raise ValueError("instance_id must be non-empty")
    if role not in {"controller", "teacher"}:
        raise ValueError(f"Role claim must be controller or teacher, got {role!r}.")
    instance_hash = _resource_hash(instance_id)
    return Path(runtime_root).expanduser().resolve() / CLAIMS_DIRNAME / "roles" / instance_hash / f"{role}.claim"


def wandb_claim_path(runtime_root: str | Path, result_root: str | Path) -> Path:
    root = Path(result_root).expanduser().resolve()
    return Path(runtime_root).expanduser().resolve() / CLAIMS_DIRNAME / "wandb" / f"{_resource_hash(str(root))}.claim"


def _claim_timestamp(payload: Mapping[str, Any] | None, key: str) -> float | None:
    if not payload:
        return None
    try:
        return float(payload[key])
    except (KeyError, TypeError, ValueError):
        return None


def _unclaimed_state(path: Path) -> dict[str, Any]:
    return {
        "exists": False,
        "state": "QUEUED",
        "claim_path": str(path),
        "owner": None,
        "heartbeat": None,
        "heartbeat_age_seconds": None,
    }


def inspect_claim(
    claim_path: str | Path,
    *,
    stale_after_seconds: float,
    now: float | None = None,
) -> dict[str, Any]:
    if stale_after_seconds <= 0:
        raise ValueError("stale_after_seconds must be positive")
    path = Path(claim_path).expanduser().resolve()
    now = time.time() if now is None else float(now)
    if not path.is_dir():
        return _unclaimed_state(path)

    owner = _read_json(path / CLAIM_OWNER_FILENAME)
    heartbeat = _read_json(path / CLAIM_HEARTBEAT_FILENAME)
    try:
        fallback_timestamp = path.stat().st_mtime
    except FileNotFoundError:
        return _unclaimed_state(path)
    last_seen = (
        _claim_timestamp(heartbeat, "updated_at_unix")
        or _claim_timestamp(owner, "created_at_unix")
        or fallback_timestamp
    )
    age = max(now - last_seen, 0.0)
    if age > stale_after_seconds:
        state = "STALE"
    elif owner is None:
        state = "INITIALIZING"
    else:
        state = "CLAIMED"
    return {
        "exists": True,
        "state": state,
        "claim_path": str(path),
        "owner": owner,
        "heartbeat": heartbeat,
        "heartbeat_age_seconds": age,
    }


def _claim_owner_payload(
    *,
    claim_type: str,
    claim_token: str,
    instance_id: str,
    owner_pid: int,
    owner_host: str,
    phase: str,
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if claim_type not in CLAIM_TYPES:
        raise ValueError(f"Unsupported claim type {claim_type!r}; expected one of {CLAIM_TYPES}.")
    if not instance_id:
        raise ValueError("instance_id must be non-empty")
    if owner_pid <= 0:
        raise ValueError(f"owner_pid must be positive, got {owner_pid}")
    payload: dict[str, Any] = {str(key): value for key, value in dict(metadata or {}).items() if value is not None}
    payload.update(
        {
            "schema_version": CLAIM_SCHEMA_VERSION,
            "claim_type": claim_type,
            "claim_token": claim_token,
            "instance_id": instance_id,
            "owner_pid": int(owner_pid),
            "owner_host": owner_host,
            "phase": phase,
            "created_at_unix": time.time(),
            "scheduler": {name: os.environ[name] for name in SCHEDULER_METADATA_ENV if os.environ.get(name)},
        }
    )
    return payload


def acquire_claim(
    claim_path: str | Path,
    *,
    claim_type: str,
    instance_id: str,
    owner_pid: int,
    owner_host: str | None = None,
    phase: str = "claimed",
    metadata: Mapping[str, Any] | None = None,
    stale_after_seconds: float = 300.0,
) -> dict[str, Any]:
    path = Path(claim_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    for attempt in range(2):
        try:
            path.mkdir()
            break
        except FileExistsError:
            state = inspect_claim(path, stale_after_seconds=stale_after_seconds)
            if not state["exists"] and attempt == 0:
                continue
            return {"acquired": False, **state}
    else:  # pragma: no cover - the loop either acquires or returns
        raise AssertionError("claim acquisition loop did not terminate")

    owner = _claim_owner_payload(
        claim_type=claim_type,
        claim_token=token,
        instance_id=instance_id,
        owner_pid=owner_pid,
        owner_host=owner_host or socket.gethostname(),
        phase=phase,
        metadata=metadata,
    )
    try:
        write_json_atomic(path / CLAIM_OWNER_FILENAME, owner)
        write_json_atomic(
            path / CLAIM_HEARTBEAT_FILENAME,
            {
                "schema_version": CLAIM_SCHEMA_VERSION,
                "claim_token": token,
                "updated_at_unix": time.time(),
            },
        )
    except BaseException:
        for child in (path / CLAIM_HEARTBEAT_FILENAME, path / CLAIM_OWNER_FILENAME):
            child.unlink(missing_ok=True)
        try:
            path.rmdir()
        except OSError:
            pass
        raise
    return {
        "acquired": True,
        "exists": True,
        "state": "CLAIMED",
        "claim_path": str(path),
        "claim_token": token,
        "owner": owner,
        "heartbeat": {
            "schema_version": CLAIM_SCHEMA_VERSION,
            "claim_token": token,
        },
        "heartbeat_age_seconds": 0.0,
    }


def heartbeat_claim(claim_path: str | Path, claim_token: str, *, phase: str | None = None) -> dict[str, Any]:
    path = Path(claim_path).expanduser().resolve()
    owner = _read_json(path / CLAIM_OWNER_FILENAME)
    if owner is None:
        raise RuntimeError(f"Claim owner is missing: {path}")
    if owner.get("claim_token") != claim_token:
        raise RuntimeError(f"Claim token does not match owner at {path}.")
    effective_phase = str(phase if phase is not None else owner.get("phase") or "claimed")
    if phase is not None:
        owner["phase"] = effective_phase
        owner["phase_updated_at_unix"] = time.time()
        write_json_atomic(path / CLAIM_OWNER_FILENAME, owner)
    payload = {
        "schema_version": CLAIM_SCHEMA_VERSION,
        "claim_token": claim_token,
        "updated_at_unix": time.time(),
    }
    write_json_atomic(path / CLAIM_HEARTBEAT_FILENAME, payload)
    return {**payload, "phase": effective_phase}


def _remove_claim_directory(path: Path) -> None:
    allowed_names = {CLAIM_OWNER_FILENAME, CLAIM_HEARTBEAT_FILENAME}
    for child in list(path.iterdir()):
        is_atomic_temporary = child.name.startswith(
            (f".{CLAIM_OWNER_FILENAME}.", f".{CLAIM_HEARTBEAT_FILENAME}.")
        ) and child.name.endswith(
            ".tmp",
        )
        if not child.is_file() or (child.name not in allowed_names and not is_atomic_temporary):
            raise RuntimeError(f"Refusing to remove claim with unexpected content: {child}")
        child.unlink(missing_ok=True)
    path.rmdir()


def release_claim(claim_path: str | Path, claim_token: str) -> None:
    path = Path(claim_path).expanduser().resolve()
    owner = _read_json(path / CLAIM_OWNER_FILENAME)
    if owner is None:
        raise RuntimeError(f"Claim owner is missing: {path}")
    if owner.get("claim_token") != claim_token:
        raise RuntimeError(f"Claim token does not match owner at {path}.")
    _remove_claim_directory(path)


def _claims_root(runtime_root: str | Path) -> Path:
    return Path(runtime_root).expanduser().resolve() / CLAIMS_DIRNAME


def _validated_managed_claim_path(runtime_root: str | Path, claim_path: str | Path) -> Path:
    root = _claims_root(runtime_root)
    path = Path(claim_path).expanduser().resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Claim path {path} is outside managed claim root {root}.") from exc
    if path.suffix != ".claim":
        raise ValueError(f"Managed claim path must end with .claim: {path}")
    return path


def list_claims(runtime_root: str | Path, *, stale_after_seconds: float) -> list[dict[str, Any]]:
    root = _claims_root(runtime_root)
    if not root.is_dir():
        return []
    return [
        inspect_claim(path, stale_after_seconds=stale_after_seconds)
        for path in sorted(root.rglob("*.claim"))
        if path.is_dir()
    ]


def release_stale_claim(
    runtime_root: str | Path,
    claim_path: str | Path,
    *,
    claim_token: str | None,
    stale_after_seconds: float,
    allow_ownerless: bool,
) -> dict[str, Any]:
    path = _validated_managed_claim_path(runtime_root, claim_path)
    state = inspect_claim(path, stale_after_seconds=stale_after_seconds)
    if not state["exists"]:
        raise RuntimeError(f"Claim does not exist: {path}")
    if state["state"] != "STALE":
        raise RuntimeError(f"Claim is not stale: {path} state={state['state']}")
    owner = state["owner"]
    if owner is None:
        if not allow_ownerless:
            raise RuntimeError("Ownerless stale claims require --allow-ownerless.")
    elif not claim_token or owner.get("claim_token") != claim_token:
        raise RuntimeError(f"Claim token does not match owner at {path}.")
    _remove_claim_directory(path)
    return state


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
    result_identities: dict[tuple[str, str], tuple[Path, str]] = {}
    for checkpoint_root in discover_experiment_roots(search_root):
        try:
            spec = load_evaluation_spec(checkpoint_root)
        except Exception:
            continue
        semantic = spec.get("semantic_config") or {}
        experiment_name = str(semantic.get("experiment_name") or checkpoint_root.name)
        training_project = str(semantic.get("training_project") or "DMPD")
        training_run_id = spec.get("training_run_id")
        spec_fingerprint = str(spec["spec_fingerprint"])
        result_identity = (training_project, experiment_name)
        previous = result_identities.get(result_identity)
        if previous is not None and previous != (checkpoint_root, spec_fingerprint):
            previous_root, previous_fingerprint = previous
            raise RuntimeError(
                "Multiple checkpoint roots map to the same standalone result identity "
                f"{training_project}/{experiment_name}: {previous_root} ({previous_fingerprint}) and "
                f"{checkpoint_root} ({spec_fingerprint}). Rename one experiment before evaluating."
            )
        result_identities[result_identity] = (checkpoint_root, spec_fingerprint)
        for checkpoint in discover_committed_checkpoints(checkpoint_root):
            checkpoint_path = Path(checkpoint.checkpoint_dir)
            state = _read_json(Path(checkpoint.evaluation_state))
            if state is None:
                continue
            if checkpoint.needs_merge:
                source_model_files = sorted(Path(checkpoint.actor_dir).glob("model_world_size_*_rank_*.pt"))
                source_model_root = Path(checkpoint.actor_dir)
            else:
                source_model_root = Path(str(checkpoint.hf_checkpoint))
                source_model_files = sorted(
                    {
                        path
                        for pattern in ("*.safetensors", "pytorch_model*.bin", "model*.bin")
                        for path in source_model_root.glob(pattern)
                        if path.is_file()
                    }
                )
            source_model_descriptors = [
                {
                    "logical_path": path.relative_to(source_model_root).as_posix(),
                    "size": path.stat().st_size,
                    "mtime_ns": path.stat().st_mtime_ns,
                }
                for path in source_model_files
            ]
            source_checkpoint_identity = payload_fingerprint(
                {
                    "schema_version": SOURCE_CHECKPOINT_IDENTITY_SCHEMA_VERSION,
                    "training_project": training_project,
                    "experiment_name": experiment_name,
                    "training_run_id": None if training_run_id is None else str(training_run_id),
                    "spec_fingerprint": spec_fingerprint,
                    "global_step": checkpoint.step,
                    "evaluation_state_fingerprint": state.get("state_fingerprint"),
                    "source_model_files": source_model_descriptors,
                }
            )
            candidates.append(
                SearchCheckpointCandidate(
                    **asdict(checkpoint),
                    checkpoint_root=str(checkpoint_root),
                    evaluation_spec=str(checkpoint_root / EVALUATION_SPEC_FILENAME),
                    experiment_name=experiment_name,
                    training_project=training_project,
                    training_run_id=None if training_run_id is None else str(training_run_id),
                    spec_fingerprint=spec_fingerprint,
                    source_checkpoint_identity=source_checkpoint_identity,
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
    manifest_cache_root: str | Path | None = None,
    force_model_rehash: bool = False,
) -> dict[str, Any]:
    model_path = Path(model_path).expanduser().resolve()
    cache_path = None if manifest_cache_root is None else model_manifest_cache_path(manifest_cache_root, model_path)
    model_manifest = capture_stable_model_manifest(
        model_path,
        cache_path=cache_path,
        force_rehash=force_model_rehash,
    )
    payload = {
        "schema_version": ENDPOINT_SCHEMA_VERSION,
        "base_url": str(base_url).rstrip("/"),
        "served_model_name": str(served_model_name),
        "model_path": str(model_path),
        "model_fingerprint": model_manifest["fingerprint"],
        "model_manifest": model_manifest,
        "legacy_stat_model_fingerprint": legacy_stat_model_fingerprint(model_path),
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
    required = {
        "base_url",
        "served_model_name",
        "model_path",
        "model_fingerprint",
        "model_manifest",
        "legacy_stat_model_fingerprint",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise RuntimeError(f"Teacher endpoint descriptor is missing fields {missing}: {descriptor_path}.")
    valid, reason = validate_stable_model_manifest(payload["model_manifest"])
    if not valid or payload["model_fingerprint"] != payload["model_manifest"].get("fingerprint"):
        raise RuntimeError(f"Teacher endpoint model manifest is invalid at {descriptor_path}: {reason}.")
    return payload


def _claim_path_from_args(args: argparse.Namespace) -> Path:
    if args.claim_type == "checkpoint":
        if args.checkpoint_root is None or args.global_step is None:
            raise ValueError("Checkpoint claims require --checkpoint-root and --global-step.")
        return checkpoint_claim_path(args.runtime_root, args.checkpoint_root, args.global_step)
    if args.claim_type == "wandb":
        if args.result_root is None:
            raise ValueError("W&B claims require --result-root.")
        return wandb_claim_path(args.runtime_root, args.result_root)
    if args.claim_type == "role":
        if args.role is None:
            raise ValueError("Role claims require --role.")
        return role_claim_path(args.runtime_root, args.instance_id, args.role)
    raise AssertionError(f"Unhandled claim type: {args.claim_type}")


def _claim_metadata_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "checkpoint_root": args.checkpoint_root,
        "global_step": args.global_step,
        "experiment_name": args.experiment_name,
        "training_project": args.training_project,
        "spec_fingerprint": args.spec_fingerprint,
        "result_root": args.result_root,
        "role": args.role,
    }


def _format_age(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    total = max(int(seconds), 0)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _claim_table(states: list[dict[str, Any]]) -> str:
    headers = ("STATE", "TYPE", "INSTANCE", "OWNER", "PHASE", "AGE", "RESOURCE", "CLAIM_PATH")
    rows: list[tuple[str, ...]] = []
    for state in states:
        owner = state.get("owner") or {}
        heartbeat = state.get("heartbeat") or {}
        checkpoint_root = owner.get("checkpoint_root")
        global_step = owner.get("global_step")
        if checkpoint_root is not None and global_step is not None:
            resource = f"{owner.get('experiment_name') or Path(str(checkpoint_root)).name}/global_step_{global_step}"
        elif owner.get("result_root"):
            resource = str(owner["result_root"])
        elif owner.get("role"):
            resource = str(owner["role"])
        else:
            resource = "ownerless"
        rows.append(
            (
                str(state.get("state") or "UNKNOWN"),
                str(owner.get("claim_type") or "-"),
                str(owner.get("instance_id") or "-"),
                f"{owner.get('owner_host') or '-'}:{owner.get('owner_pid') or '-'}",
                str(owner.get("phase") or heartbeat.get("phase") or "-"),
                _format_age(state.get("heartbeat_age_seconds")),
                resource,
                str(state["claim_path"]),
            )
        )
    widths = [len(value) for value in headers]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row, strict=True)]

    def render(row: tuple[str, ...]) -> str:
        return "  ".join(value.ljust(width) for value, width in zip(row, widths, strict=True)).rstrip()

    lines = [render(headers), render(tuple("-" * width for width in widths))]
    lines.extend(render(row) for row in rows)
    if not rows:
        lines.append("No active claims.")
    return "\n".join(lines)


def render_queue_snapshot(
    candidates: Iterable[Mapping[str, Any]],
    *,
    runtime_root: str | Path,
    instance_id: str,
    stale_after_seconds: float,
    current_claim_path: str | Path | None = None,
    max_rows: int = 0,
) -> tuple[str, str]:
    if max_rows < 0:
        raise ValueError("max_rows must be non-negative")
    current = Path(current_claim_path).expanduser().resolve() if current_claim_path else None
    rows: list[tuple[str, str, str, str, str, str, str]] = []
    signature_rows: list[dict[str, Any]] = []
    counts = {"CURRENT": 0, "QUEUED": 0, "CLAIMED": 0, "STALE": 0}
    for raw_candidate in candidates:
        candidate = dict(raw_candidate)
        checkpoint_root = str(candidate["checkpoint_root"])
        step = int(candidate["step"])
        experiment_name = str(candidate["experiment_name"])
        path = checkpoint_claim_path(runtime_root, checkpoint_root, step)
        state = inspect_claim(path, stale_after_seconds=stale_after_seconds)
        owner = state.get("owner") or {}
        heartbeat = state.get("heartbeat") or {}
        if current is not None and path == current:
            status = "CURRENT"
        elif state["state"] == "STALE":
            status = "STALE"
        elif state["state"] in {"CLAIMED", "INITIALIZING"}:
            status = "CLAIMED"
        else:
            status = "QUEUED"
        counts[status] += 1
        owner_id = str(owner.get("instance_id") or "-")
        phase = str(owner.get("phase") or heartbeat.get("phase") or ("initializing" if state["exists"] else "-"))
        age = _format_age(state.get("heartbeat_age_seconds"))
        rows.append((status, experiment_name, str(step), owner_id, phase, age, checkpoint_root))
        signature_rows.append(
            {
                "status": status,
                "checkpoint_root": checkpoint_root,
                "step": step,
                "claim_token": owner.get("claim_token"),
                "instance_id": owner.get("instance_id"),
                "phase": phase,
            }
        )

    visible_rows = rows if max_rows == 0 else rows[:max_rows]
    headers = ("STATUS", "EXPERIMENT", "STEP", "OWNER", "PHASE", "HEARTBEAT_AGE", "CHECKPOINT_ROOT")
    widths = [len(value) for value in headers]
    for row in visible_rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row, strict=True)]

    def render(row: tuple[str, ...]) -> str:
        return "  ".join(value.ljust(width) for value, width in zip(row, widths, strict=True)).rstrip()

    current_label = next(
        (f"{row[1]}/global_step_{row[2]}" for row in rows if row[0] == "CURRENT"),
        "none",
    )
    lines = [
        (
            f"[queue] instance={instance_id} current={current_label} queued={counts['QUEUED']} "
            f"claimed={counts['CLAIMED']} stale={counts['STALE']} total={len(rows)}"
        ),
        render(headers),
        render(tuple("-" * width for width in widths)),
    ]
    lines.extend(render(row) for row in visible_rows)
    if len(visible_rows) < len(rows):
        lines.append(f"... {len(rows) - len(visible_rows)} additional row(s) hidden by EVAL_QUEUE_MAX_ROWS")
    if not rows:
        lines.append("No pending committed checkpoints.")
    signature = payload_fingerprint({"instance_id": instance_id, "rows": signature_rows})
    return "\n".join(lines), signature


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
    endpoint.add_argument("--manifest-cache-root")
    endpoint.add_argument("--force-model-rehash", action="store_true")

    model_manifest = subparsers.add_parser("model-manifest")
    model_manifest.add_argument("--model-path", required=True)
    model_manifest.add_argument("--manifest-cache-root")
    model_manifest.add_argument("--force-model-rehash", action="store_true")

    register_alias = subparsers.add_parser("register-model-alias")
    register_alias.add_argument("--alias-file", required=True)
    register_alias.add_argument("--model-path", required=True)
    register_alias.add_argument("--legacy-fingerprint", action="append", required=True)
    register_alias.add_argument("--expected-stable-fingerprint", required=True)
    register_alias.add_argument("--manifest-cache-root")
    register_alias.add_argument("--force-model-rehash", action="store_true")
    register_alias.add_argument("--note")

    acquire = subparsers.add_parser("acquire-claim")
    acquire.add_argument("--runtime-root", required=True)
    acquire.add_argument("--claim-type", choices=CLAIM_TYPES, required=True)
    acquire.add_argument("--instance-id", required=True)
    acquire.add_argument("--owner-pid", type=int, required=True)
    acquire.add_argument("--owner-host", default=socket.gethostname())
    acquire.add_argument("--phase", default="claimed")
    acquire.add_argument("--stale-after-seconds", type=float, default=300.0)
    acquire.add_argument("--checkpoint-root")
    acquire.add_argument("--global-step", type=int)
    acquire.add_argument("--experiment-name")
    acquire.add_argument("--training-project")
    acquire.add_argument("--spec-fingerprint")
    acquire.add_argument("--result-root")
    acquire.add_argument("--role", choices=("controller", "teacher"))

    heartbeat = subparsers.add_parser("heartbeat-claim")
    heartbeat.add_argument("--claim-path", required=True)
    heartbeat.add_argument("--claim-token", required=True)
    heartbeat.add_argument("--phase")
    heartbeat.add_argument("--loop-seconds", type=float)

    release = subparsers.add_parser("release-claim")
    release.add_argument("--claim-path", required=True)
    release.add_argument("--claim-token", required=True)

    inspect = subparsers.add_parser("inspect-claim")
    inspect.add_argument("--claim-path", required=True)
    inspect.add_argument("--stale-after-seconds", type=float, default=300.0)

    list_parser = subparsers.add_parser("list-claims")
    list_parser.add_argument("--runtime-root", required=True)
    list_parser.add_argument("--stale-after-seconds", type=float, default=300.0)
    list_parser.add_argument("--json", action="store_true")

    release_stale = subparsers.add_parser("release-stale-claim")
    release_stale.add_argument("--runtime-root", required=True)
    release_stale.add_argument("--claim-path", required=True)
    release_stale.add_argument("--claim-token")
    release_stale.add_argument("--stale-after-seconds", type=float, default=300.0)
    release_stale.add_argument("--allow-ownerless", action="store_true")

    queue = subparsers.add_parser("render-queue")
    queue.add_argument("--runtime-root", required=True)
    queue.add_argument("--instance-id", required=True)
    queue.add_argument("--stale-after-seconds", type=float, default=300.0)
    queue.add_argument("--current-claim-path")
    queue.add_argument("--max-rows", type=int, default=0)
    queue.add_argument("--include-signature", action="store_true")
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
            manifest_cache_root=args.manifest_cache_root,
            force_model_rehash=args.force_model_rehash,
        )
        print(json.dumps(payload, sort_keys=True))
        return 0
    if args.command == "model-manifest":
        cache_path = (
            None
            if args.manifest_cache_root is None
            else model_manifest_cache_path(args.manifest_cache_root, args.model_path)
        )
        payload = capture_stable_model_manifest(
            args.model_path,
            cache_path=cache_path,
            force_rehash=args.force_model_rehash,
        )
        print(json.dumps(payload, sort_keys=True))
        return 0
    if args.command == "register-model-alias":
        cache_path = (
            None
            if args.manifest_cache_root is None
            else model_manifest_cache_path(args.manifest_cache_root, args.model_path)
        )
        payload = register_model_fingerprint_aliases(
            args.alias_file,
            legacy_fingerprints=args.legacy_fingerprint,
            model_path=args.model_path,
            expected_stable_fingerprint=args.expected_stable_fingerprint,
            cache_path=cache_path,
            force_rehash=args.force_model_rehash,
            note=args.note,
        )
        print(json.dumps(payload, sort_keys=True))
        return 0
    if args.command == "acquire-claim":
        path = _claim_path_from_args(args)
        result = acquire_claim(
            path,
            claim_type=args.claim_type,
            instance_id=args.instance_id,
            owner_pid=args.owner_pid,
            owner_host=args.owner_host,
            phase=args.phase,
            metadata=_claim_metadata_from_args(args),
            stale_after_seconds=args.stale_after_seconds,
        )
        print(json.dumps(result, sort_keys=True))
        return 0 if result["acquired"] else 3
    if args.command == "heartbeat-claim":
        if args.loop_seconds is not None:
            if args.loop_seconds <= 0:
                raise ValueError("--loop-seconds must be positive")
            while True:
                time.sleep(args.loop_seconds)
                heartbeat_claim(args.claim_path, args.claim_token, phase=args.phase)
        payload = heartbeat_claim(args.claim_path, args.claim_token, phase=args.phase)
        print(json.dumps(payload, sort_keys=True))
        return 0
    if args.command == "release-claim":
        release_claim(args.claim_path, args.claim_token)
        print(json.dumps({"released": True, "claim_path": str(Path(args.claim_path).resolve())}, sort_keys=True))
        return 0
    if args.command == "inspect-claim":
        print(json.dumps(inspect_claim(args.claim_path, stale_after_seconds=args.stale_after_seconds), sort_keys=True))
        return 0
    if args.command == "list-claims":
        states = list_claims(args.runtime_root, stale_after_seconds=args.stale_after_seconds)
        if args.json:
            print(json.dumps(states, indent=2, sort_keys=True))
        else:
            print(_claim_table(states))
        return 0
    if args.command == "release-stale-claim":
        state = release_stale_claim(
            args.runtime_root,
            args.claim_path,
            claim_token=args.claim_token,
            stale_after_seconds=args.stale_after_seconds,
            allow_ownerless=args.allow_ownerless,
        )
        print(json.dumps({"released": True, "claim": state}, sort_keys=True))
        return 0
    if args.command == "render-queue":
        candidates = [json.loads(line) for line in sys.stdin if line.strip()]
        table, signature = render_queue_snapshot(
            candidates,
            runtime_root=args.runtime_root,
            instance_id=args.instance_id,
            stale_after_seconds=args.stale_after_seconds,
            current_claim_path=args.current_claim_path,
            max_rows=args.max_rows,
        )
        if args.include_signature:
            print(f"__QUEUE_SIGNATURE__={signature}")
        print(table)
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
