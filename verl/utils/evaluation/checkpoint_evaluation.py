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

"""Durable metadata shared by training and standalone checkpoint evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import uuid
from pathlib import Path
from typing import Any, Mapping

from omegaconf import DictConfig, OmegaConf

EVALUATION_SPEC_FILENAME = "evaluation_spec.json"
EVALUATION_STATE_FILENAME = "evaluation_state.json"
EVALUATION_SPEC_SCHEMA_VERSION = 1
EVALUATION_STATE_SCHEMA_VERSION = 1


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=_json_default)


def payload_fingerprint(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def write_json_atomic(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _plain(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _select(config: DictConfig, path: str, default: Any = None) -> Any:
    value = OmegaConf.select(config, path, default=default)
    return _plain(value)


def _source_metadata() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {
            "git_revision": revision,
            "git_dirty": bool(status.strip()),
            "git_status_short": status.splitlines(),
        }
    except Exception:
        return {"git_revision": None, "git_dirty": None, "git_status_short": None}


def build_evaluation_spec(config: DictConfig, *, training_run_id: str | None) -> dict[str, Any]:
    """Build the immutable standalone-evaluation contract for one training run."""

    evaluation_max_response_length = _select(config, "trainer.test_benchmarks_max_response_length")
    if evaluation_max_response_length is None:
        evaluation_max_response_length = _select(config, "data.max_response_length")
    semantic_config = {
        "experiment_name": str(_select(config, "trainer.experiment_name")),
        "training_project": str(_select(config, "trainer.project_name")),
        "training": {
            "metric_logging_mode": _select(config, "trainer.metric_logging.mode", "full"),
            "save_freq": _select(config, "trainer.save_freq"),
            "max_actor_ckpt_to_keep": _select(config, "trainer.max_actor_ckpt_to_keep"),
        },
        "student": {
            "model_path": _select(config, "actor_rollout_ref.model.path"),
            "tokenizer_path": _select(config, "actor_rollout_ref.model.tokenizer_path"),
            "chat_template_from_model": _select(config, "actor_rollout_ref.model.chat_template_from_model"),
            "apply_chat_template_kwargs": _select(config, "data.apply_chat_template_kwargs", {}),
            "trust_remote_code": bool(_select(config, "data.trust_remote_code", False)),
        },
        "teacher": {
            "teacher_key": _select(config, "distillation.teacher_key"),
            "models": _select(config, "distillation.teacher_models", {}),
        },
        "benchmarks": _select(config, "data.test_benchmarks", {}),
        "sampling": {
            "test_kwargs": _select(config, "actor_rollout_ref.rollout.test_kwargs", {}),
            "max_prompt_length": int(_select(config, "data.max_prompt_length")),
            "max_response_length": int(evaluation_max_response_length),
            "max_model_len": _select(config, "actor_rollout_ref.rollout.max_model_len"),
            "base_seed": _select(config, "data.seed"),
        },
        "reward": {
            "manager": _select(config, "reward.reward_manager.name"),
            "benchmark_reward_data_source": _select(config, "data.test_benchmarks.reward_data_source", "math_verify"),
        },
        "distillation": {
            "enabled": bool(_select(config, "distillation.enabled", False)),
            "intervention": _select(config, "distillation.intervention", {}),
            "log_prob_min_clamp": _select(config, "distillation.distillation_loss.log_prob_min_clamp", -10.0),
        },
    }
    spec_fingerprint = payload_fingerprint(semantic_config)
    return {
        "schema_version": EVALUATION_SPEC_SCHEMA_VERSION,
        "spec_fingerprint": spec_fingerprint,
        "training_run_id": training_run_id,
        "semantic_config": semantic_config,
        "source": _source_metadata(),
    }


def write_evaluation_spec(
    checkpoint_root: str | os.PathLike[str],
    config: DictConfig,
    *,
    training_run_id: str | None,
) -> dict[str, Any]:
    """Create or validate the immutable evaluation spec at the checkpoint root."""

    checkpoint_root = Path(checkpoint_root).expanduser().resolve()
    spec_path = checkpoint_root / EVALUATION_SPEC_FILENAME
    spec = build_evaluation_spec(config, training_run_id=training_run_id)
    if spec_path.is_file():
        existing = json.loads(spec_path.read_text(encoding="utf-8"))
        if existing.get("spec_fingerprint") != spec["spec_fingerprint"]:
            raise RuntimeError(
                "Standalone evaluation spec differs from the existing checkpoint-root contract: "
                f"path={spec_path}, existing={existing.get('spec_fingerprint')}, "
                f"current={spec['spec_fingerprint']}. Use a new checkpoint root for a different evaluation contract."
            )
        existing_run_id = existing.get("training_run_id")
        if existing_run_id and training_run_id and str(existing_run_id) != str(training_run_id):
            raise RuntimeError(
                "Standalone evaluation spec training run ID differs from the resumed run: "
                f"existing={existing_run_id!r}, current={training_run_id!r}."
            )
        return existing
    write_json_atomic(spec_path, spec)
    return spec


def write_evaluation_state(
    checkpoint_dir: str | os.PathLike[str],
    *,
    checkpoint_step: int,
    spec_fingerprint: str,
    teacher_nll_calibration: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Write checkpoint-local, online state needed by a read-only evaluator."""

    state = {
        "schema_version": EVALUATION_STATE_SCHEMA_VERSION,
        "checkpoint_step": int(checkpoint_step),
        "spec_fingerprint": str(spec_fingerprint),
        "teacher_nll_calibration": dict(teacher_nll_calibration) if teacher_nll_calibration is not None else None,
    }
    state["state_fingerprint"] = payload_fingerprint(state)
    write_json_atomic(Path(checkpoint_dir) / EVALUATION_STATE_FILENAME, state)
    return state


def load_evaluation_spec(checkpoint_root: str | os.PathLike[str]) -> dict[str, Any]:
    path = Path(checkpoint_root) / EVALUATION_SPEC_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != EVALUATION_SPEC_SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported standalone evaluation spec schema at {path}: {payload.get('schema_version')}")
    expected = payload_fingerprint(payload.get("semantic_config"))
    if payload.get("spec_fingerprint") != expected:
        raise RuntimeError(f"Standalone evaluation spec fingerprint is invalid: {path}")
    return payload


def load_evaluation_state(
    checkpoint_dir: str | os.PathLike[str],
    *,
    expected_step: int,
    expected_spec_fingerprint: str,
) -> dict[str, Any]:
    path = Path(checkpoint_dir) / EVALUATION_STATE_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != EVALUATION_STATE_SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported standalone evaluation state schema at {path}: {payload.get('schema_version')}")
    if int(payload.get("checkpoint_step", -1)) != int(expected_step):
        raise RuntimeError(f"Standalone evaluation state step mismatch at {path}")
    if payload.get("spec_fingerprint") != expected_spec_fingerprint:
        raise RuntimeError(f"Standalone evaluation state/spec fingerprint mismatch at {path}")
    fingerprint_payload = dict(payload)
    actual_fingerprint = fingerprint_payload.pop("state_fingerprint", None)
    if actual_fingerprint != payload_fingerprint(fingerprint_payload):
        raise RuntimeError(f"Standalone evaluation state fingerprint is invalid: {path}")
    return payload
