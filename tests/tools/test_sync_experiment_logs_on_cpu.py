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

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from run import upload as sync
from verl.trainer.ppo.metric_archive import MetricArchiveWriter, MetricRouter


def _file(root, name, data):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


class _Hub:
    token = None

    def __init__(self):
        self.files = {}
        self.uploads = []
        self.downloads = []

    @property
    def revision(self):
        return f"rev-{len(self.uploads)}"

    def repo_info(self, repo_id, *, repo_type):
        assert repo_type == "dataset"
        return SimpleNamespace(private=True, sha=self.revision)

    def list_repo_tree(self, repo_id, prefix, *, repo_type, revision, recursive):
        assert repo_type == "dataset" and revision == self.revision and recursive
        for path, data in self.files.items():
            if not path.startswith(f"{prefix}/"):
                continue
            blob_id = hashlib.sha1(f"blob {len(data)}\0".encode() + data, usedforsecurity=False).hexdigest()
            lfs = SimpleNamespace(sha256=hashlib.sha256(data).hexdigest()) if path.endswith(".parquet") else None
            yield SimpleNamespace(path=path, size=len(data), blob_id=blob_id, lfs=lfs)

    def create_commit(self, repo_id, operations, *, repo_type, parent_commit, commit_message):
        assert repo_type == "dataset" and parent_commit == self.revision
        self.uploads.append({operation.path_in_repo for operation in operations})
        self.files.update(
            {operation.path_in_repo: Path(operation.path_or_fileobj).read_bytes() for operation in operations}
        )

    def download(self, repo_id, filename, *, repo_type, revision, local_dir, token):
        assert repo_type == "dataset" and revision == self.revision
        self.downloads.append(filename)
        return str(_file(Path(local_dir), filename, self.files[filename]))


def test_hf_upload_and_download_only_transfer_changed_files(tmp_path, monkeypatch):
    hub = _Hub()
    monkeypatch.setattr(sync, "hf_hub_download", hub.download)
    run = tmp_path / "source"
    destination = tmp_path / "destination"
    prefix = "runs/run-1/"
    files = {
        "diagnostics/manifest.json": b'{"logical_run_id":"run-1"}',
        "diagnostics/shards/one.parquet": b"step 1",
        "diagnostics/attachments/session/gap.png": b"heatmap",
        "diagnostics/sessions/session.json": b"session",
        "swanlog/run-session/backup.swanlab": b"offline log",
        "global_step_1/metrics_commit.json": b"marker",
        "latest_checkpointed_iteration.txt": b"1",
        "evaluations/test.json": b"eval",
    }
    for name, data in files.items():
        _file(run, name, data)
    _file(run, "global_step_1/actor/weights.bin", b"weights")
    _file(run, "diagnostics/shards/partial.tmp", b"partial")

    assert sync.upload_archive(hub, "repo", run, "run-1", dry_run=True) == len(files)
    assert not hub.uploads
    assert sync.upload_archive(hub, "repo", run, "run-1", dry_run=False) == len(files)
    assert hub.uploads == [{prefix + name for name in files}]
    assert sync.upload_archive(hub, "repo", run, "run-1", dry_run=False) == 0

    # Existing archives can contain pre-migration media and metric history.
    hub.files[prefix + "wandb_mirror/run.json"] = b"historical run"
    assert sync.download_archives(hub, "repo", destination, ["run-1"], dry_run=True) == len(files) + 1
    assert not destination.exists() and not hub.downloads
    assert sync.download_archives(hub, "repo", destination, ["run-1"], dry_run=False) == len(files) + 1
    assert sync.download_archives(hub, "repo", destination, ["run-1"], dry_run=False) == 0

    changes = {
        "diagnostics/shards/two.parquet": b"step 2",
        "diagnostics/manifest.json": b'{"logical_run_id":"run-1","last_step":2}',
    }
    for name, data in changes.items():
        _file(run, name, data)
    assert sync.upload_archive(hub, "repo", run, "run-1", dry_run=False) == 2
    assert hub.uploads[-1] == {prefix + name for name in changes}
    hub.downloads.clear()
    assert sync.download_archives(hub, "repo", destination, [], dry_run=False) == 2
    assert set(hub.downloads) == {prefix + name for name in changes}
    for path, content in hub.files.items():
        assert (destination / path).read_bytes() == content


def test_upload_command_discovers_all_runs_by_default_and_skips_unchanged_files(tmp_path, monkeypatch):
    hub = _Hub()
    monkeypatch.setattr(sync, "DEFAULT_ROOT", tmp_path)
    monkeypatch.setattr(sync, "HfApi", lambda: hub)
    monkeypatch.setattr(sync, "import_module", lambda _name: SimpleNamespace())
    for run_id in ("run-1", "run-2"):
        manifest = json.dumps({"logical_run_id": run_id, "experiment": run_id}).encode()
        _file(tmp_path, f"outputs/{run_id}/diagnostics/manifest.json", manifest)
    _file(tmp_path, "outputs/unarchived/evaluations/test.json", b"eval")

    monkeypatch.setattr(sys, "argv", ["upload.py", "--dry-run"])
    assert sync.main() == 0
    assert not hub.uploads

    monkeypatch.setattr(sys, "argv", ["upload.py"])
    assert sync.main() == 0
    assert hub.uploads == [{"runs/run-1/diagnostics/manifest.json"}, {"runs/run-2/diagnostics/manifest.json"}]
    assert sync.main() == 0
    assert len(hub.uploads) == 2

    _file(tmp_path, "outputs/run-2/diagnostics/shards/new.parquet", b"metrics")
    assert sync.main() == 0
    assert hub.uploads[2:] == [{"runs/run-2/diagnostics/shards/new.parquet"}]


class _SwanLab:
    class Settings(SimpleNamespace):
        Probe = Terminal = SimpleNamespace

    def __init__(self):
        self.summary = {}
        self.calls = []
        self.existing_id = None

    def Api(self):
        return SimpleNamespace(run=self._remote_run)

    def _remote_run(self, path):
        assert path == "workspace/project/run-1"
        return SimpleNamespace(
            run_id=self.existing_id,
            summary=lambda keys: {key: self.summary[key] for key in keys if key in self.summary},
        )

    def init(self, *, workspace, project, name, id, resume, config, mode, public, log_dir, settings):
        assert mode == "online" and id == "run-1"
        self.calls.append(resume)
        self.existing_id = id
        return SimpleNamespace(log=self.log, finish=Mock())

    def log(self, row, *, step):
        self.calls.append((step, row))
        for key, value in row.items():
            assert key not in self.summary or step > self.summary[key]["step"]
            self.summary[key] = {"step": step, "value": value}


class _SwanLabWithOnlineErrorPolicy(_SwanLab):
    def init(self, *, on_online_error, **kwargs):
        assert on_online_error == "raise"
        return super().init(**kwargs)


@pytest.mark.parametrize("sdk", [_SwanLab, _SwanLabWithOnlineErrorPolicy], ids=["0.10.3", "with-online-error-policy"])
def test_swanlab_preserves_run_id_and_appends_per_metric_without_duplicates(sdk):
    swanlab = sdk()
    experiment = sync.migration.Experiment("run-1", "qwen3")
    project = "workspace/project"
    history = {1: {"loss": 3.0}, 2: {"loss": 2.0, "score": 0.5}}
    assert sync.upload_swanlab(swanlab, project, experiment, history, dry_run=True) == 3
    assert swanlab.calls == []
    assert sync.upload_swanlab(swanlab, project, experiment, history, dry_run=False) == 3
    assert swanlab.calls == ["allow", (1, {"loss": 3.0}), (2, {"loss": 2.0, "score": 0.5})]

    history[2]["accuracy"] = 0.25
    history[3] = {"loss": 1.0}
    assert sync.upload_swanlab(swanlab, project, experiment, history, dry_run=False) == 2
    assert swanlab.calls[-3:] == ["must", (2, {"accuracy": 0.25}), (3, {"loss": 1.0})]
    assert sync.upload_swanlab(swanlab, project, experiment, history, dry_run=False) == 0
    with pytest.raises(ValueError, match="different value.*loss.*step 3"):
        sync.upload_swanlab(swanlab, project, experiment, {3: {"loss": 0.0}}, dry_run=False)


@pytest.mark.parametrize("checkpoint", [False, True])
def test_swanlab_reads_only_live_values_at_a_durable_step(tmp_path, checkpoint):
    router = MetricRouter("balanced")
    with MetricArchiveWriter(
        tmp_path / "diagnostics", logical_run_id="run-1", project="project", experiment="qwen3"
    ) as writer:
        writer.append_event(router.route({"actor/loss": 1.0, "bucket/bin/0000_0128/token_count": 4}), global_step=1)
        if checkpoint:
            (tmp_path / "global_step_1").mkdir()
            writer.commit_checkpoint(tmp_path / "global_step_1", checkpoint_step=1)
            _file(tmp_path, "latest_checkpointed_iteration.txt", b"1")
            writer.append_event(router.route({"actor/loss": 0.25}), global_step=2)
            (tmp_path / "global_step_2").mkdir()
            writer.commit_checkpoint(tmp_path / "global_step_2", checkpoint_step=2)

    history = sync._dashboard_history(tmp_path, writer.manifest)
    assert history[1]["actor/loss"] == 1.0
    assert "bucket/bin/0000_0128/token_count" not in history[1]
    assert set(history) == {1}
