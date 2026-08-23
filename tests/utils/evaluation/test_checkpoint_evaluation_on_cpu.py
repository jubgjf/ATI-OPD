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

import json
from copy import deepcopy
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from verl.utils.evaluation.checkpoint_evaluation import (
    EVALUATION_SPEC_FILENAME,
    EVALUATION_STATE_FILENAME,
    build_evaluation_spec,
    load_evaluation_spec,
    load_evaluation_state,
    payload_fingerprint,
    write_evaluation_spec,
    write_evaluation_state,
)


def _config(experiment_name: str = "run-a"):
    return OmegaConf.create(
        {
            "trainer": {
                "experiment_name": experiment_name,
                "project_name": "DMPD",
                "test_benchmarks_max_response_length": None,
            },
            "actor_rollout_ref": {
                "model": {
                    "path": "/models/student",
                    "tokenizer_path": None,
                    "chat_template_from_model": "/models/teacher",
                },
                "rollout": {
                    "max_model_len": 4096,
                    "test_kwargs": {"n": 16, "temperature": 0.7, "top_p": 0.95, "top_k": -1},
                },
            },
            "data": {
                "trust_remote_code": False,
                "max_prompt_length": 1024,
                "max_response_length": 3072,
                "seed": 42,
                "test_benchmarks": {
                    "root": "/datasets",
                    "names": ["math-ai/aime24"],
                    "reward_data_source": "math_verify",
                },
            },
            "reward": {"reward_manager": {"name": "naive"}},
            "distillation": {
                "enabled": True,
                "teacher_key": "data_source",
                "teacher_models": {"default": {"model_path": "/models/teacher"}},
                "distillation_loss": {"log_prob_min_clamp": -10.0},
                "intervention": {
                    "enabled": True,
                    "repair_for_evaluation": True,
                    "gap_metric": "k3",
                },
            },
        }
    )


def test_evaluation_spec_and_state_round_trip_on_cpu(tmp_path):
    spec = write_evaluation_spec(tmp_path, _config(), training_run_id="train-id")
    loaded_spec = load_evaluation_spec(tmp_path)
    assert loaded_spec == spec
    assert loaded_spec["schema_version"] == 2
    assert loaded_spec["semantic_config"]["tool_call"] == {"enabled": False}
    assert loaded_spec["semantic_config"]["sampling"]["max_response_length"] == 3072

    checkpoint_dir = tmp_path / "global_step_20"
    state = write_evaluation_state(
        checkpoint_dir,
        checkpoint_step=20,
        spec_fingerprint=spec["spec_fingerprint"],
        teacher_nll_calibration={"history": [0.2, 0.3], "threshold": 0.3, "version": 4},
    )
    assert (
        load_evaluation_state(
            checkpoint_dir,
            expected_step=20,
            expected_spec_fingerprint=spec["spec_fingerprint"],
        )
        == state
    )


def test_evaluation_spec_rejects_changed_semantics_on_cpu(tmp_path):
    write_evaluation_spec(tmp_path, _config(), training_run_id="train-id")
    with pytest.raises(RuntimeError, match="spec differs"):
        write_evaluation_spec(tmp_path, _config("run-b"), training_run_id="train-id")


def test_evaluation_metadata_detects_tampering_on_cpu(tmp_path):
    spec = write_evaluation_spec(tmp_path, _config(), training_run_id="train-id")
    spec_path = tmp_path / EVALUATION_SPEC_FILENAME
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    payload["semantic_config"]["sampling"]["max_response_length"] = 1
    spec_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="fingerprint is invalid"):
        load_evaluation_spec(tmp_path)

    checkpoint_dir = tmp_path / "global_step_20"
    write_evaluation_state(
        checkpoint_dir,
        checkpoint_step=20,
        spec_fingerprint=spec["spec_fingerprint"],
        teacher_nll_calibration=None,
    )
    state_path = checkpoint_dir / EVALUATION_STATE_FILENAME
    state_payload = json.loads(state_path.read_text(encoding="utf-8"))
    state_payload["checkpoint_step"] = 21
    state_path.write_text(json.dumps(state_payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="step mismatch"):
        load_evaluation_state(
            checkpoint_dir,
            expected_step=20,
            expected_spec_fingerprint=spec["spec_fingerprint"],
        )


def test_v1_evaluation_spec_is_read_and_preserved_byte_for_byte_on_cpu(tmp_path):
    current = build_evaluation_spec(_config(), training_run_id="train-id")
    legacy = deepcopy(current)
    legacy["schema_version"] = 1
    legacy["semantic_config"].pop("tool_call")
    legacy["spec_fingerprint"] = payload_fingerprint(legacy["semantic_config"])
    path = tmp_path / EVALUATION_SPEC_FILENAME
    path.write_text(json.dumps(legacy, ensure_ascii=False, indent=3) + "\n", encoding="utf-8")
    original_bytes = path.read_bytes()

    assert load_evaluation_spec(tmp_path) == legacy
    assert write_evaluation_spec(tmp_path, _config(), training_run_id="train-id") == legacy
    assert path.read_bytes() == original_bytes


def test_v2_spec_without_tool_metadata_is_treated_as_legacy_and_preserved_on_cpu(tmp_path):
    current = build_evaluation_spec(_config(), training_run_id="train-id")
    legacy_v2 = deepcopy(current)
    legacy_v2["semantic_config"].pop("tool_call")
    legacy_v2["spec_fingerprint"] = payload_fingerprint(legacy_v2["semantic_config"])
    path = tmp_path / EVALUATION_SPEC_FILENAME
    path.write_text(json.dumps(legacy_v2, separators=(",", ":")), encoding="utf-8")
    original_bytes = path.read_bytes()

    assert load_evaluation_spec(tmp_path) == legacy_v2
    assert write_evaluation_spec(tmp_path, _config(), training_run_id="train-id") == legacy_v2
    assert path.read_bytes() == original_bytes


def test_v2_tool_contract_snapshots_prompt_schema_and_files_on_cpu(tmp_path):
    prompt_path = tmp_path / "tool-system.txt"
    prompt_path.write_text("Use the Python tool when it helps.\n", encoding="utf-8")
    function_tool_path = Path(__file__).resolve().parents[3] / "verl" / "tools" / "code_interpreter_tool.py"
    config = _config()
    OmegaConf.update(config, "data.system_prompt_path", str(prompt_path), force_add=True)
    OmegaConf.update(config, "data.apply_chat_template_kwargs", {"enable_thinking": True}, force_add=True)
    OmegaConf.update(config, "actor_rollout_ref.rollout.multi_turn.enable", True, force_add=True)
    OmegaConf.update(
        config,
        "actor_rollout_ref.rollout.multi_turn.function_tool_path",
        str(function_tool_path),
        force_add=True,
    )
    OmegaConf.update(config, "actor_rollout_ref.rollout.multi_turn.format", "hermes", force_add=True)
    OmegaConf.update(config, "actor_rollout_ref.rollout.multi_turn.max_parallel_calls", 1, force_add=True)
    OmegaConf.update(config, "actor_rollout_ref.rollout.multi_turn.max_tool_response_length", 4096, force_add=True)
    OmegaConf.update(config, "actor_rollout_ref.rollout.agent.default_agent_loop", "tool_agent", force_add=True)

    spec = write_evaluation_spec(tmp_path, config, training_run_id="tool-run")
    contract = spec["semantic_config"]["tool_call"]

    assert spec["schema_version"] == 2
    assert contract["enabled"] is True
    assert contract["agent_loop"] == "tool_agent"
    assert contract["format"] == "hermes"
    assert contract["system_prompt"]["content"] == "Use the Python tool when it helps."
    assert contract["tool_schemas"][0]["function"]["name"] == "code_interpreter"
    assert contract["function_tool"]["filename"] == "code_interpreter_tool.py"
    assert contract["max_tool_response_length"] == 4096
    assert contract["thinking"] == {"enable_thinking": True}
    assert load_evaluation_spec(tmp_path) == spec

    tampered = deepcopy(spec)
    tampered_contract = tampered["semantic_config"]["tool_call"]
    tampered_contract["system_prompt"]["sha256"] = "tampered"
    contract_payload = dict(tampered_contract)
    contract_payload.pop("contract_fingerprint")
    tampered_contract["contract_fingerprint"] = payload_fingerprint(contract_payload)
    tampered["spec_fingerprint"] = payload_fingerprint(tampered["semantic_config"])
    (tmp_path / EVALUATION_SPEC_FILENAME).write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(RuntimeError, match="system prompt fingerprint"):
        load_evaluation_spec(tmp_path)
