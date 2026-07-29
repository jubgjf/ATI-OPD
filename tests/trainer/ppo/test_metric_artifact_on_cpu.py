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

from pathlib import Path
from types import SimpleNamespace

from verl.trainer.ppo.metric_archive import MetricArchiveWriter, MetricRouter, ShardState
from verl.trainer.ppo.metric_artifact import MetricArtifactUploader, wait_for_artifact_manifest_updates
from verl.utils.tracking import WandbRunInfo


class _WaitHandle:
    def __init__(self):
        self.calls = []

    def wait(self, timeout=None):
        self.calls.append(timeout)


class _Tracking:
    def __init__(self, mode):
        self.wandb_run_info = WandbRunInfo(mode=mode, run_id="run-1", sync_dir=None, sync_file=None)
        self.calls = []
        self.handle = _WaitHandle()

    def log_wandb_artifact(self, path, **kwargs):
        self.calls.append((path, kwargs))
        return SimpleNamespace(
            requires_wait=self.wandb_run_info.is_online,
            wait_handle=self.handle if self.wandb_run_info.is_online else None,
        )


def _committed_writer(tmp_path):
    checkpoint = tmp_path / "global_step_2"
    checkpoint.mkdir()
    writer = MetricArchiveWriter(
        tmp_path / "diagnostics",
        logical_run_id="run-1",
        project="DMPD",
        experiment="artifact-test",
    )
    writer.append_event(MetricRouter("full").route({"actor/loss": 1.0}), global_step=2)
    attachment = tmp_path / "gap.png"
    attachment.write_bytes(b"png")
    writer.add_event_file(attachment, global_step=2, metric_key="opd/gap_abs_heatmap")
    shard = writer.commit_checkpoint(checkpoint, checkpoint_step=2).shards[0]
    return writer, shard


def test_online_artifact_queue_marks_submitted_then_uploaded_and_bundles_files_on_cpu(tmp_path, monkeypatch):
    writer, shard = _committed_writer(tmp_path)
    tracking = _Tracking("online")
    uploader = MetricArtifactUploader(
        writer=writer,
        tracking=tracking,
        project="DMPD",
        entity="jubgjf",
        finish_timeout_seconds=1,
    )
    monkeypatch.setattr(uploader, "_remote_artifact_matches", lambda *_args: False)

    uploader.submit_pending()
    assert writer.manifest["shards"][0]["state"] == ShardState.SUBMITTED
    wait_for_artifact_manifest_updates(uploader)

    assert writer.manifest["shards"][0]["state"] == ShardState.UPLOADED
    assert tracking.handle.calls == [1]
    path, kwargs = tracking.calls[0]
    assert path == shard.path
    assert kwargs["artifact_name"] == "run-1-training-diagnostics"
    assert kwargs["aliases"] == ("latest", "steps-2-2")
    assert kwargs["metadata"]["delivery_session_id"] == writer.session_id
    assert f"shards/{Path(shard.metadata_path).name}" in kwargs["additional_files"]
    assert any(name.startswith("attachments/") for name in kwargs["additional_files"])
    assert writer.manifest["shards"][0]["wandb_delivery_session_id"] == writer.session_id
    uploader.finish()
    writer.close()


def test_online_artifact_queue_skips_duplicate_hash_version_on_cpu(tmp_path, monkeypatch):
    writer, shard = _committed_writer(tmp_path)
    tracking = _Tracking("online")
    uploader = MetricArtifactUploader(
        writer=writer,
        tracking=tracking,
        project="DMPD",
        entity=None,
        finish_timeout_seconds=1,
    )
    monkeypatch.setattr(uploader, "_remote_artifact_matches", lambda _alias, sha: sha == shard.sha256)

    uploader.submit_pending()
    wait_for_artifact_manifest_updates(uploader)

    assert tracking.calls == []
    assert writer.manifest["shards"][0]["state"] == ShardState.UPLOADED
    uploader.finish()
    writer.close()


def test_offline_artifact_queue_stages_without_waiting_on_cpu(tmp_path):
    writer, _shard = _committed_writer(tmp_path)
    tracking = _Tracking("offline")
    uploader = MetricArtifactUploader(
        writer=writer,
        tracking=tracking,
        project="DMPD",
        entity=None,
        finish_timeout_seconds=1,
    )

    uploader.submit_pending()
    wait_for_artifact_manifest_updates(uploader)

    assert writer.manifest["shards"][0]["state"] == ShardState.STAGED_OFFLINE
    assert tracking.handle.calls == []
    uploader.finish()
    writer.close()


def test_online_submitted_shard_can_retry_into_an_offline_segment_on_cpu(tmp_path):
    writer, shard = _committed_writer(tmp_path)
    writer.mark_shard_state(shard.shard_id, ShardState.SUBMITTED)
    tracking = _Tracking("offline")
    uploader = MetricArtifactUploader(
        writer=writer,
        tracking=tracking,
        project="DMPD",
        entity=None,
        finish_timeout_seconds=1,
    )

    uploader.submit_pending()
    wait_for_artifact_manifest_updates(uploader)

    assert writer.manifest["shards"][0]["state"] == ShardState.STAGED_OFFLINE
    assert writer.manifest["shards"][0]["wandb_delivery_session_id"] == writer.session_id
    uploader.finish()
    writer.close()


def test_uncommitted_closed_shard_does_not_advance_latest_artifact_alias_on_cpu(tmp_path, monkeypatch):
    writer = MetricArchiveWriter(
        tmp_path / "diagnostics",
        logical_run_id="run-1",
        project="DMPD",
        experiment="artifact-test",
    )
    writer.append_event(MetricRouter("full").route({"actor/loss": 1.0}), global_step=1)
    writer.close_shard()
    tracking = _Tracking("online")
    uploader = MetricArtifactUploader(
        writer=writer,
        tracking=tracking,
        project="DMPD",
        entity=None,
        finish_timeout_seconds=1,
    )
    monkeypatch.setattr(uploader, "_remote_artifact_matches", lambda *_args: False)

    uploader.submit_pending()
    wait_for_artifact_manifest_updates(uploader)

    _path, kwargs = tracking.calls[0]
    assert kwargs["aliases"] == ("steps-1-1",)
    assert not kwargs["metadata"]["active_lineage"]
    uploader.finish()
    writer.close()
