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
EVALUATION_SPEC_SCHEMA_VERSION = 2
SUPPORTED_EVALUATION_SPEC_SCHEMA_VERSIONS = frozenset({1, EVALUATION_SPEC_SCHEMA_VERSION})
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


def _resolve_contract_file(path_value: Any, *, field_name: str) -> Path:
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError(f"{field_name} must be a non-empty path when tool-call rollout is enabled.")
    from verl.experimental.agent_loop.utils import resolve_config_path

    path = Path(resolve_config_path(path_value)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{field_name} does not exist: {path}")
    return path


def _contract_file_descriptor(path: Path, *, configured_path: str) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "configured_path": configured_path,
        "filename": path.name,
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _build_tool_call_contract(config: DictConfig) -> dict[str, Any]:
    multi_turn_enabled = bool(_select(config, "actor_rollout_ref.rollout.multi_turn.enable", False))
    agent_loop = str(_select(config, "actor_rollout_ref.rollout.agent.default_agent_loop", "single_turn_agent"))
    enabled = multi_turn_enabled and agent_loop == "tool_agent"
    if not enabled:
        return {"enabled": False}

    system_prompt_value = _select(config, "data.system_prompt_path")
    function_tool_value = _select(config, "actor_rollout_ref.rollout.multi_turn.function_tool_path")
    tool_config_value = _select(config, "actor_rollout_ref.rollout.multi_turn.tool_config_path")
    system_prompt_path = _resolve_contract_file(system_prompt_value, field_name="data.system_prompt_path")
    system_prompt = system_prompt_path.read_text(encoding="utf-8").strip()
    if not system_prompt:
        raise ValueError(f"data.system_prompt_path is empty: {system_prompt_path}")

    function_tool_path = None
    if function_tool_value:
        function_tool_path = _resolve_contract_file(
            function_tool_value,
            field_name="actor_rollout_ref.rollout.multi_turn.function_tool_path",
        )
    tool_config_path = None
    if tool_config_value:
        tool_config_path = _resolve_contract_file(
            tool_config_value,
            field_name="actor_rollout_ref.rollout.multi_turn.tool_config_path",
        )
    if function_tool_path is None and tool_config_path is None:
        raise ValueError("Tool-call rollout requires function_tool_path or tool_config_path.")

    from verl.tools.tool_registry import load_all_tools

    tools = load_all_tools(
        tool_config_path=str(tool_config_path) if tool_config_path is not None else None,
        function_tool_path=str(function_tool_path) if function_tool_path is not None else None,
    )
    if not tools:
        raise ValueError("Tool-call rollout resolved no tools while building the standalone evaluation contract.")
    schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tools]
    system_prompt_descriptor = _contract_file_descriptor(
        system_prompt_path,
        configured_path=str(system_prompt_value),
    )
    system_prompt_descriptor["content"] = system_prompt
    system_prompt_descriptor["sha256"] = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
    contract: dict[str, Any] = {
        "enabled": True,
        "agent_loop": agent_loop,
        "format": str(_select(config, "actor_rollout_ref.rollout.multi_turn.format", "hermes")),
        "system_prompt": system_prompt_descriptor,
        "tool_schemas": schemas,
        "function_tool": (
            _contract_file_descriptor(function_tool_path, configured_path=str(function_tool_value))
            if function_tool_path is not None
            else None
        ),
        "tool_config": (
            _contract_file_descriptor(tool_config_path, configured_path=str(tool_config_value))
            if tool_config_path is not None
            else None
        ),
        "max_assistant_turns": _select(
            config,
            "actor_rollout_ref.rollout.multi_turn.max_assistant_turns",
        ),
        "max_user_turns": _select(config, "actor_rollout_ref.rollout.multi_turn.max_user_turns"),
        "max_parallel_calls": int(_select(config, "actor_rollout_ref.rollout.multi_turn.max_parallel_calls", 1)),
        "max_tool_response_length": int(
            _select(config, "actor_rollout_ref.rollout.multi_turn.max_tool_response_length", 256)
        ),
        "tool_response_truncate_side": str(
            _select(config, "actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side", "middle")
        ),
        "apply_chat_template_kwargs": _select(config, "data.apply_chat_template_kwargs", {}),
    }
    contract["thinking"] = {
        "enable_thinking": bool((contract["apply_chat_template_kwargs"] or {}).get("enable_thinking", True))
    }
    contract["contract_fingerprint"] = payload_fingerprint(contract)
    return contract


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
        "tool_call": _build_tool_call_contract(config),
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
        current_fingerprint = spec["spec_fingerprint"]
        existing_semantic_config = existing.get("semantic_config") or {}
        if existing.get("schema_version") == 1 or "tool_call" not in existing_semantic_config:
            # v1 and early manifests without tool-call metadata must remain
            # byte-for-byte immutable when every representable field matches.
            # Retrofitting a disabled default would invalidate every
            # checkpoint-local evaluation_state.json.
            legacy_semantic_config = dict(spec["semantic_config"])
            legacy_semantic_config.pop("tool_call", None)
            current_fingerprint = payload_fingerprint(legacy_semantic_config)
        if existing.get("spec_fingerprint") != current_fingerprint:
            raise RuntimeError(
                "Standalone evaluation spec differs from the existing checkpoint-root contract: "
                f"path={spec_path}, existing={existing.get('spec_fingerprint')}, "
                f"current={current_fingerprint}. Use a new checkpoint root for a different evaluation contract."
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
    if payload.get("schema_version") not in SUPPORTED_EVALUATION_SPEC_SCHEMA_VERSIONS:
        raise RuntimeError(f"Unsupported standalone evaluation spec schema at {path}: {payload.get('schema_version')}")
    expected = payload_fingerprint(payload.get("semantic_config"))
    if payload.get("spec_fingerprint") != expected:
        raise RuntimeError(f"Standalone evaluation spec fingerprint is invalid: {path}")
    if payload.get("schema_version") == 2:
        tool_call = (payload.get("semantic_config") or {}).get("tool_call") or {}
        if bool(tool_call.get("enabled")):
            contract_payload = dict(tool_call)
            actual_contract_fingerprint = contract_payload.pop("contract_fingerprint", None)
            if actual_contract_fingerprint != payload_fingerprint(contract_payload):
                raise RuntimeError(f"Standalone tool-call contract fingerprint is invalid: {path}")
            system_prompt = tool_call.get("system_prompt") or {}
            content = str(system_prompt.get("content") or "")
            if not content or system_prompt.get("sha256") != hashlib.sha256(content.encode("utf-8")).hexdigest():
                raise RuntimeError(f"Standalone tool-call system prompt fingerprint is invalid: {path}")
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
