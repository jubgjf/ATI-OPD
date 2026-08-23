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

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "run" / "upload_opd_benchmark_wandb.py"


def _load_standalone_uploader() -> ModuleType:
    spec = importlib.util.spec_from_file_location("standalone_opd_benchmark_wandb", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load standalone uploader from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


uploader = _load_standalone_uploader()
_is_not_found_error = uploader._is_not_found_error
artifact_fingerprint = uploader.artifact_fingerprint
artifact_name = uploader.artifact_name
resolve_run_id = uploader.resolve_run_id
stable_run_id = uploader.stable_run_id
unit_metric_payload = uploader.unit_metric_payload
unit_sync_key = uploader.unit_sync_key


def test_wandb_identity_is_stable_and_scoped_on_cpu():
    first = stable_run_id("jubgjf", "DMPD", "experiment")

    assert first == stable_run_id("jubgjf", "DMPD", "experiment")
    assert first != stable_run_id("jubgjf", "other", "experiment")
    assert artifact_name("experiment", "math-ai__aime24--abc", 20) != artifact_name(
        "experiment", "math-ai__aime24--abc", 40
    )


def test_wandb_metric_payload_uses_global_step_axis_and_unit_fingerprint_on_cpu():
    summary = {
        "global_step": 20,
        "benchmark": {"id": "aime24--abc", "path": "math-ai/aime24"},
        "metrics": {
            "acc": 0.5,
            "question_count": 30,
            "rollout_count": 480,
            "response_tokens": {"mean": 1000, "p50": 900, "p90": 2000, "p99": 3000},
            "gap_regions": {"last_window": {"abs_mean_token_weighted": 0.3}},
            "tool_call": {
                "attempt_rate": 0.75,
                "parse_success_rate": 0.9,
                "execution_success_rate": 0.8,
                "markdown_extraction_rate": 0.25,
                "calls_per_trajectory": {"mean": 1.2},
                "observation_tokens": {"mean": 80},
                "final_answer_after_tool_rate": 0.95,
                "ended_after_observation_rate": 0.01,
                "total_attempts": 360,
                "total_parsed_calls": 324,
            },
        },
        "rollout_n": 16,
        "accuracy_at_n": {"mean": 0.5, "maj": 0.6, "best": 0.8},
        "repair": {
            "metrics": {"response_tokens": {"mean": 800}, "truncation_rate": 0.1},
            "accuracy_at_n": {"mean": 0.55, "maj": 0.65, "best": 0.85},
            "paired": {
                "paired_delta": 0.05,
                "repair_execution_rate": 0.25,
                "trigger_rate": 0.3,
                "intervention_count": {"mean": 0.4},
                "first_intervention_position": {"mean": 1024},
                "transitions": {"pre_wrong_post_correct": 0.1},
            },
        },
        "question_cluster_bootstrap_ci95": {"acc": {"ci95_low": 0.4, "ci95_high": 0.6}},
    }

    payload = unit_metric_payload(
        summary,
        "result-hash",
        metadata={"generation_elapsed_seconds": 12, "repair_elapsed_seconds": 20},
        checkpoint_timing={"checkpoint_total_seconds": 40, "failure_retry_count": 1},
    )

    assert payload["eval/global_step"] == 20
    assert payload["sync/unit_key"] == unit_sync_key("math-ai/aime24", 20)
    assert payload["sync/result_fingerprint"] == "result-hash"
    assert payload["benchmark/aime24--abc/acc"] == 0.5
    assert payload["benchmark/aime24--abc/acc_ci95_low"] == 0.4
    assert payload["test-core/aime24/acc/mean@16"] == 0.5
    assert payload["test-repair-core/aime24/acc/paired_delta@16"] == 0.05
    assert payload["test-repair-core/aime24/transition/pre_wrong_post_correct"] == 0.1
    assert payload["test-core/aime24/tool_call/attempt_rate"] == 0.75
    assert payload["test-core/aime24/tool_call/execution_success_rate"] == 0.8
    assert payload["test-core/aime24/tool_call/observation_tokens_mean"] == 80
    assert payload["eval/timing/checkpoint_total_seconds"] == 40


def test_artifact_fingerprint_tracks_logs_and_rollout_policy_on_cpu(tmp_path):
    unit_dir = tmp_path / "unit"
    unit_dir.mkdir()
    log_path = unit_dir / "generate.stdout.log"
    rollout_path = unit_dir / "rollouts.jsonl.gz"
    log_path.write_text("first\n", encoding="utf-8")
    rollout_path.write_bytes(b"rollout-one")

    with_rollouts = artifact_fingerprint(unit_dir, include_rollouts=True)
    without_rollouts = artifact_fingerprint(unit_dir, include_rollouts=False)
    log_path.write_text("first\nsecond\n", encoding="utf-8")

    assert artifact_fingerprint(unit_dir, include_rollouts=True) != with_rollouts
    assert artifact_fingerprint(unit_dir, include_rollouts=False) != without_rollouts
    without_rollouts = artifact_fingerprint(unit_dir, include_rollouts=False)
    rollout_path.write_bytes(b"rollout-two")
    assert artifact_fingerprint(unit_dir, include_rollouts=False) == without_rollouts


def test_wandb_not_found_detection_does_not_swallow_other_api_failures_on_cpu():
    class Response:
        status_code = 404

    missing = RuntimeError("wrapped")
    missing.response = Response()

    assert _is_not_found_error(missing) is True
    assert _is_not_found_error(RuntimeError("connection timed out")) is False


def test_run_id_resolution_adopts_one_legacy_name_and_rejects_ambiguity_on_cpu():
    class Response:
        status_code = 404

    class FakeApi:
        def __init__(self, matches):
            self.matches = matches

        def run(self, path):
            del path
            error = RuntimeError("missing")
            error.response = Response()
            raise error

        def runs(self, **kwargs):
            assert kwargs["filters"] == {"display_name": "experiment"}
            return self.matches

    assert (
        resolve_run_id(
            FakeApi([SimpleNamespace(id="legacy-id")]),
            "jubgjf",
            "DMPD",
            "experiment",
            adopt_existing_name=True,
        )
        == "legacy-id"
    )
    with pytest.raises(RuntimeError, match="multiple W&B runs"):
        resolve_run_id(
            FakeApi([SimpleNamespace(id="first"), SimpleNamespace(id="second")]),
            "jubgjf",
            "DMPD",
            "experiment",
            adopt_existing_name=True,
        )


def test_standalone_uploader_imports_only_stdlib_and_wandb_on_cpu():
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.partition(".")[0])

    assert imported_roots <= set(sys.stdlib_module_names) | {"wandb"}


def test_standalone_uploader_help_does_not_import_verl_on_cpu():
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Idempotently publish standalone OPD benchmark results" in result.stdout


def test_standalone_uploader_discovers_completed_units_on_cpu(tmp_path):
    result_root = tmp_path / "experiment"
    benchmark_root = result_root / "benchmarks" / "aime24--abc"
    unit_dir = benchmark_root / "global_step_100"
    unit_dir.mkdir(parents=True)
    (benchmark_root / "benchmark.json").write_text(
        json.dumps({"path": "math-ai/aime24"}),
        encoding="utf-8",
    )

    required_files = {}
    for name in uploader.REQUIRED_RESULT_FILES:
        path = unit_dir / name
        if name == uploader.METADATA_FILENAME:
            payload = {
                "schema_version": uploader.SCHEMA_VERSION,
                "generation_complete": True,
                "generation_fingerprint": "generation-hash",
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
        else:
            path.write_text("{}\n", encoding="utf-8")
        required_files[name] = {"size": path.stat().st_size, "sha256": "not-checked"}

    marker = {
        "schema_version": uploader.SCHEMA_VERSION,
        "generation_fingerprint": "generation-hash",
        "result_fingerprint": "result-hash",
        "files": required_files,
    }
    (unit_dir / uploader.COMPLETION_FILENAME).write_text(json.dumps(marker), encoding="utf-8")

    assert uploader.discover_units(result_root) == {("math-ai/aime24", 100): unit_dir}
    assert uploader.discover_units(result_root, global_steps={100}) == {("math-ai/aime24", 100): unit_dir}
    assert uploader.discover_units(result_root, global_steps={20}) == {}
    assert uploader.discover_units(result_root, benchmarks={"math-ai/aime24"}) == {("math-ai/aime24", 100): unit_dir}
    assert uploader.discover_units(result_root, benchmarks={"math-ai/amc23"}) == {}


def test_sync_experiment_applies_requested_step_filter_before_conflict_checks_on_cpu(tmp_path, monkeypatch):
    result_root = tmp_path / "experiment"
    step_20 = _write_completed_unit(result_root, step=20)
    step_100 = _write_completed_unit(result_root, step=100)
    discovered: list[tuple[set[int] | None, set[str] | None]] = []

    def discover_filtered(root, *, global_steps=None, benchmarks=None):
        assert root == result_root
        discovered.append((global_steps, benchmarks))
        return (
            {("math-ai/aime24", 100): step_100}
            if global_steps == {100}
            else {
                ("math-ai/aime24", 20): step_20,
                ("math-ai/aime24", 100): step_100,
            }
        )

    monkeypatch.setattr(uploader, "discover_units", discover_filtered)
    args = SimpleNamespace(
        global_step=[100],
        benchmark=["math-ai/aime24"],
        entity="jubgjf",
        project="DMPD-eval",
        run_id=None,
        mode="disabled",
        dry_run=False,
        adopt_existing_name=True,
        include_rollouts=True,
        wandb_dir=None,
        wandb_data_dir=None,
        wandb_cache_dir=None,
        source_training_run_id="training-id",
    )

    assert uploader.sync_experiment(args, result_root) == 0
    assert discovered == [({100}, {"math-ai/aime24"})]


def test_offline_paths_preserve_jump_host_visible_symlink_spelling_on_cpu(tmp_path):
    physical = tmp_path / "npu-mount" / "results"
    physical.mkdir(parents=True)
    portable = tmp_path / "jump-host-path" / "results"
    portable.parent.mkdir(parents=True)
    portable.symlink_to(physical, target_is_directory=True)

    assert uploader.absolute_path(portable) == portable
    roots = uploader.discover_experiment_roots(portable.parent, [portable.name])
    assert roots == [portable]


def _write_completed_unit(result_root: Path, *, step: int, benchmark_path: str = "math-ai/aime24") -> Path:
    benchmark_root = result_root / "benchmarks" / "aime24--abc"
    unit_dir = benchmark_root / f"global_step_{step}"
    unit_dir.mkdir(parents=True, exist_ok=True)
    (result_root / "experiment.json").write_text(json.dumps({"experiment": result_root.name}), encoding="utf-8")
    (benchmark_root / "benchmark.json").write_text(json.dumps({"path": benchmark_path}), encoding="utf-8")
    summary = {
        "global_step": step,
        "rollout_n": 16,
        "benchmark": {"id": "aime24--abc", "name": "aime24", "path": benchmark_path},
        "metrics": {"acc": 0.5, "response_tokens": {"mean": 10}},
        "accuracy_at_n": {"mean": 0.5, "maj": 0.5, "best": 0.5},
    }
    metadata = {
        "schema_version": uploader.SCHEMA_VERSION,
        "generation_complete": True,
        "generation_fingerprint": f"generation-{step}",
    }
    payloads = {
        uploader.METADATA_FILENAME: metadata,
        uploader.SUMMARY_FILENAME: summary,
        uploader.PER_PROBLEM_FILENAME: {},
        uploader.LENGTH_BINS_FILENAME: {},
        uploader.ROLLOUTS_FILENAME: {},
    }
    required_files = {}
    for name, payload in payloads.items():
        path = unit_dir / name
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        required_files[name] = {"size": path.stat().st_size, "sha256": "not-checked"}
    marker = {
        "schema_version": uploader.SCHEMA_VERSION,
        "generation_fingerprint": f"generation-{step}",
        "result_fingerprint": f"result-{step}",
        "files": required_files,
    }
    (unit_dir / uploader.COMPLETION_FILENAME).write_text(json.dumps(marker), encoding="utf-8")
    return unit_dir


def _write_mtime_migrated_completed_unit(result_root: Path, *, step: int = 20) -> tuple[Path, str, str]:
    benchmark_path = "math-ai/aime24"
    benchmark_root = result_root / "benchmarks" / "aime24--abc"
    unit_dir = benchmark_root / f"global_step_{step}"
    unit_dir.mkdir(parents=True)
    uploader.write_json_atomic(result_root / "experiment.json", {"experiment": result_root.name})
    uploader.write_json_atomic(benchmark_root / "benchmark.json", {"path": benchmark_path})

    legacy_manifest = {
        "dataset": [],
        "student_model_metadata": [],
        "student_model_weights": [
            {
                "path": "/models/step-20/model.safetensors",
                "size": 10,
                "mode": "stat",
                "mtime_ns": 10,
            }
        ],
        "student_tokenizer": [],
        "chat_template": [],
    }
    legacy_manifest["fingerprint"] = uploader.payload_fingerprint(legacy_manifest)
    legacy_metadata = {
        "schema_version": uploader.SCHEMA_VERSION,
        "generation_schema_version": 2,
        "experiment": result_root.name,
        "global_step": step,
        "generation_complete": True,
        "repair_complete": True,
        "generation_fingerprint": "legacy-generation-fingerprint",
        "input_manifest": legacy_manifest,
    }
    summary = {
        "global_step": step,
        "rollout_n": 16,
        "benchmark": {"id": "aime24--abc", "name": "aime24", "path": benchmark_path},
        "metrics": {"acc": 0.5, "response_tokens": {"mean": 10}},
        "accuracy_at_n": {"mean": 0.5, "maj": 0.5, "best": 0.5},
    }
    payloads = {
        uploader.METADATA_FILENAME: legacy_metadata,
        uploader.SUMMARY_FILENAME: summary,
        uploader.PER_PROBLEM_FILENAME: {},
        uploader.LENGTH_BINS_FILENAME: {},
        uploader.ROLLOUTS_FILENAME: {},
    }
    for name, payload in payloads.items():
        uploader.write_json_atomic(unit_dir / name, payload)

    def result_files() -> dict[str, dict[str, object]]:
        return {
            name: {
                "size": (unit_dir / name).stat().st_size,
                "sha256": uploader._sha256_file(unit_dir / name),
            }
            for name in uploader.REQUIRED_RESULT_FILES
        }

    legacy_files = result_files()
    legacy_result_fingerprint = uploader.payload_fingerprint(legacy_files)

    stable_manifest = {
        "schema_version": 2,
        "student_model_identity": {"kind": "committed_checkpoint", "fingerprint": "source-step-20"},
        "dataset": [],
        "student_model_metadata": [],
        "student_model_weights": [
            {
                "path": "/models/step-20/model.safetensors",
                "size": 10,
                "mode": "stat",
                "logical_path": "model.safetensors",
                "mtime_ns": 20,
            }
        ],
        "student_tokenizer": [],
        "chat_template": [],
        "fingerprint": "stable-manifest-fingerprint",
    }
    migrated_metadata = dict(legacy_metadata)
    migrated_metadata["input_manifest"] = stable_manifest
    migrated_metadata["generation_compatibility"] = {
        "schema_version": 1,
        "compatible_fingerprints": ["stable-generation-fingerprint"],
        "migrations": [
            {
                "reason": uploader.LEGACY_MTIME_MIGRATION_REASON,
                "legacy_generation_fingerprint": legacy_metadata["generation_fingerprint"],
                "compatible_generation_fingerprint": "stable-generation-fingerprint",
                "legacy_input_manifest_fingerprint": legacy_manifest["fingerprint"],
                "stable_input_manifest_fingerprint": stable_manifest["fingerprint"],
                "student_model_identity": stable_manifest["student_model_identity"],
                "mtime_changes": [
                    {
                        "path": "/models/step-20/model.safetensors",
                        "size": 10,
                        "existing_mtime_ns": 10,
                        "current_mtime_ns": 20,
                    }
                ],
            }
        ],
    }
    uploader.write_json_atomic(unit_dir / uploader.METADATA_FILENAME, migrated_metadata)
    migrated_files = result_files()
    migrated_result_fingerprint = uploader.payload_fingerprint(migrated_files)
    uploader.write_json_atomic(
        unit_dir / uploader.COMPLETION_FILENAME,
        {
            "schema_version": uploader.SCHEMA_VERSION,
            "generation_fingerprint": migrated_metadata["generation_fingerprint"],
            "result_fingerprint": migrated_result_fingerprint,
            "files": migrated_files,
        },
    )
    return unit_dir, legacy_result_fingerprint, migrated_result_fingerprint


def test_wandb_recognizes_completed_result_changed_only_by_mtime_manifest_migration_on_cpu(tmp_path):
    unit_dir, legacy_fingerprint, migrated_fingerprint = _write_mtime_migrated_completed_unit(tmp_path / "experiment")
    marker, _, metadata = uploader._load_unit(unit_dir)

    compatible = uploader.compatible_legacy_result_fingerprints(unit_dir, marker, metadata)

    assert migrated_fingerprint != legacy_fingerprint
    assert compatible == {legacy_fingerprint}
    assert (
        uploader._check_remote_conflict(
            {"math-ai/aime24@global_step_20": legacy_fingerprint},
            "math-ai/aime24@global_step_20",
            migrated_fingerprint,
            compatible_fingerprints=compatible,
        )
        == legacy_fingerprint
    )
    with pytest.raises(RuntimeError, match="Completed evaluation units are immutable"):
        uploader._check_remote_conflict(
            {"math-ai/aime24@global_step_20": "unrelated-result"},
            "math-ai/aime24@global_step_20",
            migrated_fingerprint,
            compatible_fingerprints=compatible,
        )


def test_online_sync_skips_compatible_legacy_result_and_records_local_alias_on_cpu(
    tmp_path,
    monkeypatch,
    capsys,
):
    result_root = tmp_path / "experiment"
    _, legacy_fingerprint, migrated_fingerprint = _write_mtime_migrated_completed_unit(result_root)
    key = "math-ai/aime24@global_step_20"
    run_id = stable_run_id("jubgjf", "DMPD-eval", "experiment")

    class FakeRemoteRun:
        id = run_id

        @staticmethod
        def scan_history(**kwargs):
            del kwargs
            return [{uploader.SYNC_UNIT_KEY: key, uploader.SYNC_FINGERPRINT_KEY: legacy_fingerprint}]

    class FakeApi:
        @staticmethod
        def run(path):
            assert path == f"jubgjf/DMPD-eval/{run_id}"
            return FakeRemoteRun()

        @staticmethod
        def artifact(path):
            raise AssertionError(f"compatible legacy result must not query or replace its artifact: {path}")

    class FakeRun:
        resumed = True
        dir = str(tmp_path / "wandb" / "files")

        def __init__(self):
            self.summary = {"benchmark_sync_manifest": {key: legacy_fingerprint}}

        @staticmethod
        def define_metric(*args, **kwargs):
            del args, kwargs

        @staticmethod
        def log_artifact(*args, **kwargs):
            del args, kwargs
            raise AssertionError("compatible legacy result must not upload a new artifact")

        @staticmethod
        def log(*args, **kwargs):
            del args, kwargs
            raise AssertionError("compatible legacy result must not append duplicate metrics")

        @staticmethod
        def finish():
            return None

    fake_run = FakeRun()

    class FakeWandb:
        @staticmethod
        def Api():
            return FakeApi()

        @staticmethod
        def init(**kwargs):
            assert kwargs["id"] == run_id
            return fake_run

    monkeypatch.setitem(sys.modules, "wandb", FakeWandb)
    args = SimpleNamespace(
        entity="jubgjf",
        project="DMPD-eval",
        run_id=None,
        mode="online",
        dry_run=False,
        adopt_existing_name=True,
        include_rollouts=True,
        wandb_dir=None,
        wandb_data_dir=None,
        wandb_cache_dir=None,
        source_training_run_id="training-id",
    )

    assert uploader.sync_experiment(args, result_root) == 0
    assert "[wandb] compatible legacy result already synced:" in capsys.readouterr().out
    manifest = json.loads((result_root / uploader.SYNC_FILENAME).read_text(encoding="utf-8"))
    assert manifest["units"] == {key: legacy_fingerprint}
    assert manifest["compatible_units"][key] == {
        "reason": "completed result changed only by the legacy merged-weight mtime manifest migration",
        "remote_result_fingerprint": legacy_fingerprint,
        "local_result_fingerprint": migrated_fingerprint,
    }


def test_offline_sync_never_uses_public_api_and_is_locally_idempotent_on_cpu(tmp_path, monkeypatch):
    result_root = tmp_path / "experiment"
    _write_completed_unit(result_root, step=20)
    init_calls: list[dict] = []
    logged_metrics: list[dict] = []

    class FakeArtifact:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def add_file(self, *args, **kwargs):
            del args, kwargs

    class FakeRun:
        resumed = False

        def __init__(self, run_dir: Path):
            self.run_dir = run_dir
            self.dir = str(run_dir / "files")
            Path(self.dir).mkdir(parents=True)
            self.summary = {}

        def define_metric(self, *args, **kwargs):
            del args, kwargs

        def log_artifact(self, *args, **kwargs):
            del args, kwargs
            return SimpleNamespace(wait=lambda: None)

        def log(self, payload):
            logged_metrics.append(payload)

        def finish(self):
            (self.run_dir / "run-test.wandb").write_bytes(b"offline")

    class FakeWandb:
        Artifact = FakeArtifact

        @staticmethod
        def Api():
            raise AssertionError("offline sync must not instantiate wandb.Api")

        @staticmethod
        def init(**kwargs):
            init_calls.append(kwargs)
            return FakeRun(tmp_path / f"offline-run-{len(init_calls)}")

    monkeypatch.setitem(sys.modules, "wandb", FakeWandb)
    args = SimpleNamespace(
        entity="jubgjf",
        project="DMPD-eval",
        run_id=None,
        mode="offline",
        dry_run=False,
        adopt_existing_name=True,
        include_rollouts=True,
        wandb_dir=None,
        wandb_data_dir=None,
        wandb_cache_dir=None,
        source_training_run_id="training-id",
    )

    assert uploader.sync_experiment(args, result_root) == 0
    assert uploader.sync_experiment(args, result_root) == 0
    assert len(init_calls) == 1
    assert init_calls[0]["id"] == stable_run_id("jubgjf", "DMPD-eval", "experiment")
    assert logged_metrics[0]["test-core/aime24/acc/mean@16"] == 0.5
    manifest = json.loads((result_root / uploader.SYNC_FILENAME).read_text(encoding="utf-8"))
    assert manifest["units"] == {"math-ai/aime24@global_step_20": "result-20"}
    commands = (result_root / uploader.OFFLINE_SYNC_COMMANDS_FILENAME).read_text(encoding="utf-8")
    assert "wandb sync " in commands
    assert "command -v wandb" in commands
    assert "EVAL_WANDB_ROOT" in commands
    assert "WANDB_PORTABLE_DATASET_DIR" in commands
    assert "--append" not in commands

    _write_completed_unit(result_root, step=40)
    assert uploader.sync_experiment(args, result_root) == 0
    assert len(init_calls) == 2
    commands = (result_root / uploader.OFFLINE_SYNC_COMMANDS_FILENAME).read_text(encoding="utf-8")
    assert commands.count("wandb sync") == 2
    assert "wandb sync --append" in commands
