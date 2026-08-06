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
