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

import fcntl
import gzip
import json
import sys
from fnmatch import fnmatch
from types import SimpleNamespace
from unittest.mock import Mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from run import migrate_experiment_logs as migrate


def _manifest(root, name, run_id, **fields):
    output = root / "outputs" / name
    archive = output / "diagnostics"
    archive.mkdir(parents=True)
    (archive / "manifest.json").write_text(json.dumps(dict(logical_run_id=run_id, experiment=name, **fields)))
    return output


def test_select_paper_run_and_recent_launches_without_selecting_old_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(migrate, "PAPER_RUNS", {"paper": "old-baseline-20260901-010000"})
    mirrors = tmp_path / "mirror"
    (mirrors / "paper").mkdir(parents=True)
    (mirrors / "paper/run.json").touch()
    _manifest(tmp_path, "old-baseline-20260901-010000", "paper")
    _manifest(tmp_path, "old-sweep-20261015-000000", "skip", created_at="2026-11-10T04:00:00Z")
    _manifest(tmp_path, "relay-20261101-000000", "relay")
    _manifest(tmp_path, "custom-name", "custom", created_at="2026-10-31T16:00:00Z")
    _manifest(tmp_path, "earlier-custom-name", "earlier", created_at="2026-10-31T15:59:59Z")

    selected = migrate.discover_experiments(tmp_path, mirrors, since=migrate.DEFAULT_SINCE, scope="all")
    assert {item.run_id for item in selected} == {"paper", "relay", "custom"}
    assert selected[0].mirror == mirrors / "paper"
    recent = migrate.discover_experiments(tmp_path, mirrors, since=migrate.DEFAULT_SINCE, scope="recent")
    assert {item.run_id for item in recent} == {"relay", "custom"}


def test_resume_paths_and_committed_history_replace_orphaned_values(tmp_path, monkeypatch):
    run_id = "same-id"
    first = "wandb_logs/wandb/offline-run-20261101_100000-same-id/run-same-id.wandb"
    second = "wandb_logs/wandb/offline-run-20261102_100000-same-id/run-same-id.wandb"
    for relative in (first, second):
        log = tmp_path / relative
        log.parent.mkdir(parents=True)
        log.touch()
    compute_root = migrate.SHARED_MOUNT / "code/DMPD"
    manifest = {
        "sessions": {
            "resumed": {"session_order": 1, "wandb_sync_dir": str((compute_root / second).parent)},
            "initial": {"session_order": 0, "wandb_sync_file": str(compute_root / first)},
        },
        "legacy_before_step": 1,
        "shards": [
            {"path": "/elsewhere/resumed.parquet", "state": "CHECKPOINT_COMMITTED", "checkpoint_step": 3},
            {"path": "/elsewhere/rolled-back.parquet", "state": "ORPHANED", "checkpoint_step": 4},
        ],
    }
    output = _manifest(tmp_path, "resume-20261101-100000", run_id, **manifest)
    shards = output / "diagnostics/shards"
    shards.mkdir()
    pq.write_table(
        pa.table(
            {
                "global_step": [2, 3, 3],
                "metric_key": ["timing_s/step", "timing_s/step", "opd_cost/tokens"],
                "value": [20.0, 30.0, 40.0],
                "delivery": ["live", "live", "archive"],
            }
        ),
        shards / "resumed.parquet",
    )

    def read_native(path, run_id, history, config):
        config["run_id"] = run_id
        if path == tmp_path / first:
            migrate._add_row(history, {"_step": 1, "timing_s/step": 10})
            migrate._add_row(history, {"_step": 2, "rolled_back": 1})
            migrate._add_row(history, {"_step": 5, "rolled_back": 2})
        else:
            migrate._add_row(history, {"_step": 3, "timing_s/step": 30})

    monkeypatch.setattr(migrate, "_read_wandb_file", read_native)
    segments = migrate._session_segments(run_id, manifest, [tmp_path / second, tmp_path / first], tmp_path)
    assert segments == [tmp_path / first, tmp_path / second]
    history, config = migrate.read_history(migrate.Experiment(run_id, output.name, output), manifest, segments)
    assert history == {1: {"timing_s/step": 10.0}, 2: {"timing_s/step": 20.0}, 3: {"timing_s/step": 30.0}}
    assert config == {"run_id": run_id}


def test_mirror_replay_and_upload_keep_original_run_id(tmp_path, monkeypatch):
    run_id = "old"
    mirror = tmp_path / "mirror"
    (mirror / "history").mkdir(parents=True)
    (mirror / "run.json").write_text('{"id":"old","config":{"data":{"seed":42}},"summary":{"training/global_step":3}}')
    with gzip.open(mirror / "history/steps_0001_0002.jsonl.gz", "wt") as handle:
        handle.write('{"_step":1,"training/global_step":1,"timing_s/step":2.5,"media":{"path":"file.png"}}\n')
        handle.write('{"_step":2,"training/global_step":2,"timing_s/step":2.0}\n')
    experiment = migrate.Experiment(run_id, "old-run", mirror=mirror)
    history, config = migrate.read_history(experiment, {}, [])
    assert migrate._missing_training_steps(experiment, history) == {3}
    assert config["data"]["seed"] == 42 and "media" not in history[1]

    remote = Mock()
    remote.metrics.return_value = {
        "list": [
            {"key": key, "metrics": [{"step": step, "value": row[key]} for step, row in history.items()]}
            for key in ("training/global_step", "timing_s/step")
        ]
    }
    sdk = SimpleNamespace(init=Mock(), Api=Mock(return_value=SimpleNamespace(run=Mock(return_value=remote))))
    monkeypatch.setitem(sys.modules, "swanlab", sdk)
    migrate._import_swanlab(experiment, history, config, "team", "DMPD")
    assert sdk.init.call_args.kwargs["id"] == run_id
    assert sdk.init.call_args.kwargs["resume"] == "allow"
    assert [call.kwargs["step"] for call in sdk.init.return_value.log.call_args_list] == [1, 2]

    api = SimpleNamespace(upload_folder=Mock(return_value=SimpleNamespace(oid="revision")))
    migrate._upload_hf(api, "team/logs", experiment)
    assert api.upload_folder.call_args.kwargs["path_in_repo"] == "runs/old/wandb_mirror"


def test_recover_missing_mirror_steps_from_downloaded_parquet(tmp_path):
    mirror = tmp_path / "mirror"
    archive = mirror / "logged_artifacts/training-diagnostics"
    shard = archive / "v0/shards/history.parquet"
    shard.parent.mkdir(parents=True)
    (mirror / "history").mkdir()
    (mirror / "run.json").write_text('{"id":"old","summary":{"training/global_step":2}}')
    (archive / "manifest.json").write_text(
        json.dumps({"run_id": "old", "versions": [{"version": "v0", "entries": {"shards/history.parquet": {}}}]})
    )
    with gzip.open(mirror / "history/steps_0001_0001.jsonl.gz", "wt") as handle:
        handle.write('{"_step":1,"training/global_step":1,"loss":1.0}\n')
    pq.write_table(
        pa.table(
            {
                "global_step": [1, 2, 2],
                "metric_key": ["loss", "training/global_step", "loss"],
                "value": [7.0, 2.0, 0.25],
                "delivery": ["live", "live", "live"],
            }
        ),
        shard,
    )
    experiment = migrate.Experiment("old", "old-run", mirror=mirror)
    history, _ = migrate.read_history(experiment, {}, [])
    assert history == {1: {"training/global_step": 1.0, "loss": 1.0}, 2: {"training/global_step": 2.0, "loss": 0.25}}


def test_archive_upload_patterns_and_active_writer_guard(tmp_path):
    output = _manifest(tmp_path, "opd-20261103-120000", "run-id")
    (output / "diagnostics/.writer.lock").touch()
    api = SimpleNamespace(upload_folder=Mock(return_value=SimpleNamespace(oid="revision")))
    with (output / "diagnostics/.writer.lock").open("r+b") as writer:
        fcntl.flock(writer.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(migrate.ArchiveBusyError), migrate._snapshot_lock(output):
            pass
    with migrate._snapshot_lock(output):
        migrate._upload_hf(api, "team/logs", migrate.Experiment("run-id", output.name, output=output))
    upload = api.upload_folder.call_args.kwargs
    assert upload["path_in_repo"] == "runs/run-id"
    patterns = upload["allow_patterns"]
    assert any(fnmatch("diagnostics/attachments/session/trace.jsonl.gz", pattern) for pattern in patterns)
    assert any(fnmatch("global_step_217/metrics_commit.json", pattern) for pattern in patterns)
    assert not any(fnmatch("global_step_217/actor/model.pt", pattern) for pattern in patterns)


@pytest.mark.parametrize(
    ("destination", "project"),
    [
        (["--swanlab-workspace", "jubgjf/ATI-OPD"], "ATI-OPD"),
        (["--swanlab-workspace", "jubgjf", "--swanlab-project", "ATI-OPD"], "ATI-OPD"),
        (["--swanlab-workspace", "jubgjf"], "DMPD"),
    ],
)
def test_cli_passes_workspace_and_project_to_swanlab(tmp_path, monkeypatch, destination, project):
    _manifest(tmp_path, "opd-20261103-120000", "same-id")
    history = {1: {"training/global_step": 1.0}}
    importer = Mock()
    monkeypatch.setattr(migrate, "_import_swanlab", importer)
    monkeypatch.setattr(migrate, "read_history", Mock(return_value=(history, {})))
    monkeypatch.setattr(sys, "argv", ["migrate", "--root", str(tmp_path), "--scope", "recent", *destination])

    assert migrate.main() == 0
    assert importer.call_args.args[-2:] == ("jubgjf", project)


@pytest.mark.parametrize(
    "destination",
    [
        ["--swanlab-workspace", "jubgjf/ATI-OPD/extra"],
        ["--swanlab-workspace", "jubgjf/ATI-OPD", "--swanlab-project", "DMPD"],
    ],
)
def test_cli_rejects_invalid_swanlab_target_before_hf_request(monkeypatch, destination):
    hf_api = Mock()
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=hf_api))
    monkeypatch.setattr(sys, "argv", ["migrate", "--hf-repo", "jubgjf/ATI-OPD-logs", *destination])

    with pytest.raises(SystemExit) as error:
        migrate.main()
    assert error.value.code == 2
    hf_api.assert_not_called()


def _evaluation_files(tmp_path):
    output = _manifest(tmp_path, "opd-20261103-120000", "same-id")
    evaluations = output / "evaluations"
    evaluations.mkdir()
    checkpoint = f"/opt/huawei/dataset/jnguan/code/DMPD/outputs/{output.name}/global_step_217"
    protocol = dict(
        suite="math5", seed=0, enable_thinking=True, rollout_n=2, benchmarks=["aime24"], sampling={"max_tokens": 23552}
    )
    (evaluations / "math5--step_217.json").write_text(
        json.dumps(
            dict(
                checkpoint=checkpoint,
                protocol={**protocol, "question_hashes": {"aime24:1": "hash"}},
                records=[
                    dict(benchmark="aime24", uid="aime24:1", draw=0, acc=1, response_tokens=100, finish_reason="stop"),
                    dict(
                        benchmark="aime24", uid="aime24:1", draw=1, acc=0, response_tokens=200, finish_reason="length"
                    ),
                ],
            )
        )
    )
    summary = dict(
        config={
            **protocol,
            "model_path": f"{checkpoint}/actor/huggingface",
            "effective_sampling": protocol["sampling"],
        },
        benchmarks={"aime24": dict(accuracy=0.5, response_tokens_mean=150.0, length_finish_rate=0.5, responses=2)},
        metrics={"test-legacy-core/aime24/acc/maj@2/mean": 0.4, "test-legacy-aux/aime24/invalid_pred/mean@2": 0.5},
    )
    log_dir = tmp_path / "job-logs"
    log_dir.mkdir()
    raw_stdout = "[worker 0] EVAL_RESULT_JSON=" + json.dumps(summary) + "\n"
    (log_dir / "stdout.log").write_text(raw_stdout)
    with gzip.open(log_dir / "export.out.gz", "wt") as duplicate:
        duplicate.write(raw_stdout)
    return output, log_dir, summary


def test_eval_only_uploads_existing_results_with_original_run_id(tmp_path, monkeypatch):
    _, log_dir, summary = _evaluation_files(tmp_path)
    api = Mock()
    api.repo_info.return_value.private = True
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=lambda: api))
    swanlab_upload = Mock()
    monkeypatch.setattr(migrate, "_upload_swanlab_evaluations", swanlab_upload)
    read_history = Mock()
    monkeypatch.setattr(migrate, "read_history", read_history)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "migrate",
            "--root",
            str(tmp_path),
            "--scope",
            "recent",
            "--eval-only",
            "--eval-stdout",
            str(log_dir),
            "--swanlab-workspace",
            "team/experiments",
            "--hf-repo",
            "team/logs",
        ],
    )

    assert migrate.main() == 0
    read_history.assert_not_called()
    assert api.upload_folder.call_args.kwargs["allow_patterns"] == ["evaluations/*.json"]
    api.upload_file.assert_called_once()
    upload = api.upload_file.call_args.kwargs
    assert upload["path_in_repo"] == "runs/same-id/evaluations/math5--step_217--summary.json"
    assert json.loads(upload["path_or_fileobj"]) == summary
    experiment, history, workspace, project = swanlab_upload.call_args.args
    assert (experiment.run_id, workspace, project) == ("same-id", "team", "experiments")
    assert history[217]["standalone-math5--step_217/macro/acc/mean@2"] == 0.5
    assert history[217]["standalone-math5--step_217/aime24/acc/maj@2/mean"] == 0.4


def test_evaluation_stdout_must_match_draw_protocol(tmp_path):
    output, _, summary = _evaluation_files(tmp_path)
    summary["config"]["seed"] = 42
    experiment = migrate.Experiment("same-id", output.name, output=output)
    with pytest.raises(ValueError, match="differs from saved draws"):
        migrate._evaluation_history(experiment, {("math5", 217): summary})
    from_draws = migrate._evaluation_history(experiment, {})[217]
    assert not any("/maj@" in key or "invalid_pred" in key for key in from_draws)


def test_swanlab_evaluation_upload_resumes_original_run_and_skips_existing(monkeypatch):
    mean_key = "standalone-math5--step_217/aime24/acc/mean@16"
    maj_key = "standalone-math5--step_217/aime24/acc/maj@16/mean"
    stored = {mean_key: 0.5}
    run = Mock()
    run.log.side_effect = lambda metrics, step: stored.update(metrics)
    remote = SimpleNamespace(
        metrics=lambda keys, all: {
            "list": [
                dict(key=key, metrics=[dict(step=217, value=stored[key])] if key in stored else []) for key in keys
            ]
        }
    )
    swanlab = SimpleNamespace(Api=lambda: SimpleNamespace(run=lambda path: remote), init=Mock(return_value=run))
    monkeypatch.setitem(sys.modules, "swanlab", swanlab)
    experiment = migrate.Experiment("same-id", "old-run")
    history = {217: {mean_key: 0.5, maj_key: 0.4}}

    migrate._upload_swanlab_evaluations(experiment, history, "team", "experiments")
    migrate._upload_swanlab_evaluations(experiment, history, "team", "experiments")

    swanlab.init.assert_called_once()
    assert swanlab.init.call_args.kwargs["id"] == "same-id"
    assert swanlab.init.call_args.kwargs["resume"] == "must"
    run.log.assert_called_once_with({maj_key: 0.4}, step=217)
