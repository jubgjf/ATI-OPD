# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import gzip
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pyarrow.parquet as pq
import pytest
from omegaconf import OmegaConf

from verl.trainer.ppo import ray_trainer as ray_trainer_module
from verl.trainer.ppo.metric_archive import MetricArchiveWriter, MetricLoggingSettings, MetricRouter
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, _ResolvedCheckpointContext
from verl.utils.tracking import WandbRunInfo


def _trainer_config(tmp_path: Path, *, resume_mode: str = "auto", logger=None):
    return OmegaConf.create(
        {
            "trainer": {
                "default_local_dir": str(tmp_path),
                "default_hdfs_dir": None,
                "resume_mode": resume_mode,
                "resume_from_path": None,
                "project_name": "project",
                "experiment_name": "experiment",
                "logger": ["console"] if logger is None else logger,
                "metric_logging": {
                    "mode": "balanced",
                    "resume_policy": "strict",
                    "archive": {"enabled": True, "local_dir": None},
                },
            }
        }
    )


def _bare_trainer(config) -> RayPPOTrainer:
    trainer = object.__new__(RayPPOTrainer)
    trainer.config = config
    trainer._resolved_checkpoint_context = None
    trainer._logical_run_id = None
    return trainer


def _metric_logging_trainer(config) -> RayPPOTrainer:
    trainer = _bare_trainer(config)
    trainer._metric_logging_settings = MetricLoggingSettings.from_mapping(config.trainer.metric_logging)
    trainer._metric_archive_writer = None
    trainer._metric_router = None
    trainer._metric_archive_session_id = None
    trainer._metric_resume_step = None
    return trainer


def test_resume_checkpoint_is_resolved_once_and_tracker_identity_is_checked_on_cpu(tmp_path):
    checkpoint_20 = tmp_path / "global_step_20"
    checkpoint_20.mkdir()
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("20", encoding="utf-8")
    trainer = _bare_trainer(_trainer_config(tmp_path))

    first = trainer._resolve_resume_context()
    assert first == _ResolvedCheckpointContext(path=str(checkpoint_20), step=20)

    checkpoint_30 = tmp_path / "global_step_30"
    checkpoint_30.mkdir()
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("30", encoding="utf-8")
    assert trainer._resolve_resume_context() is first


def test_strict_resume_rejects_explicit_checkpoint_tracker_mismatch_on_cpu(tmp_path):
    checkpoint_20 = tmp_path / "global_step_20"
    checkpoint_20.mkdir()
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("19", encoding="utf-8")
    config = _trainer_config(tmp_path, resume_mode="resume_path")
    config.trainer.resume_from_path = str(checkpoint_20)
    trainer = _bare_trainer(config)

    with pytest.raises(RuntimeError, match="tracker_step=19"):
        trainer._resolve_resume_context()


def test_disable_resume_rejects_existing_archive_state_in_balanced_strict_mode_on_cpu(tmp_path):
    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir()
    (diagnostics / "manifest.json").write_text("{}", encoding="utf-8")
    trainer = _bare_trainer(_trainer_config(tmp_path, resume_mode="disable"))

    with pytest.raises(RuntimeError, match="cannot reuse an output directory"):
        trainer._resolve_resume_context()


@pytest.mark.parametrize(
    ("remote_step", "expected_resume", "expected_resume_from"),
    [
        (19, "must", None),
        (20, "must", None),
        (21, None, "run-123?_step=20"),
    ],
)
def test_online_wandb_resume_rewinds_only_when_remote_history_is_ahead_on_cpu(
    tmp_path, monkeypatch, remote_step, expected_resume, expected_resume_from
):
    trainer = _bare_trainer(_trainer_config(tmp_path, logger=["console", "wandb"]))
    trainer._logical_run_id = "run-123"
    trainer._resolved_checkpoint_context = _ResolvedCheckpointContext(
        path=str(tmp_path / "global_step_20"),
        step=20,
    )
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.setattr(
        ray_trainer_module,
        "_query_wandb_remote_run_step",
        lambda **_kwargs: remote_step,
    )

    spec = trainer._build_wandb_resume_spec()

    assert spec.resume == expected_resume
    assert spec.resume_from == expected_resume_from


def test_offline_wandb_resume_reuses_identity_without_remote_query_on_cpu(tmp_path, monkeypatch):
    trainer = _bare_trainer(_trainer_config(tmp_path, logger=["wandb"]))
    trainer._logical_run_id = "run-123"
    trainer._resolved_checkpoint_context = _ResolvedCheckpointContext(
        path=str(tmp_path / "global_step_20"),
        step=20,
    )
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setattr(
        ray_trainer_module,
        "_query_wandb_remote_run_step",
        lambda **_kwargs: pytest.fail("offline resume must not query W&B"),
    )

    spec = trainer._build_wandb_resume_spec()

    assert spec.run_id == "run-123"
    assert spec.init_kwargs("offline") == {"id": "run-123", "resume": "never"}


def test_online_wandb_resume_fails_when_remote_identity_cannot_be_verified_on_cpu(tmp_path, monkeypatch):
    trainer = _bare_trainer(_trainer_config(tmp_path, logger=["wandb"]))
    trainer._logical_run_id = "run-123"
    trainer._resolved_checkpoint_context = _ResolvedCheckpointContext(
        path=str(tmp_path / "global_step_20"),
        step=20,
    )
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.setattr(
        ray_trainer_module,
        "_query_wandb_remote_run_step",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("remote run missing")),
    )

    with pytest.raises(RuntimeError, match="could not verify"):
        trainer._build_wandb_resume_spec()


def test_logger_preflight_failure_releases_metric_archive_writer_on_cpu(tmp_path):
    trainer = _bare_trainer(_trainer_config(tmp_path))
    close_calls = []
    writer = SimpleNamespace(close=lambda **kwargs: close_calls.append(kwargs))
    trainer._metric_archive_writer = None
    trainer._initialize_metric_logging = lambda: setattr(trainer, "_metric_archive_writer", writer)
    trainer._build_wandb_resume_spec = lambda: (_ for _ in ()).throw(RuntimeError("remote preflight failed"))

    with pytest.raises(RuntimeError, match="remote preflight failed"):
        trainer.init_logger()

    assert close_calls == [{"interrupted": True}]
    assert trainer._metric_archive_writer is None


def test_legacy_checkpoint_initializes_archive_with_explicit_history_gap_on_cpu(tmp_path):
    checkpoint = tmp_path / "global_step_20"
    checkpoint.mkdir()
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("20", encoding="utf-8")
    trainer = _metric_logging_trainer(_trainer_config(tmp_path))

    trainer._initialize_metric_logging()

    assert trainer._metric_resume_step == 20
    assert trainer._metric_archive_writer.manifest["legacy_before_step"] == 20
    trainer._metric_archive_writer.close()


def test_legacy_checkpoint_can_resume_again_before_first_new_format_commit_on_cpu(tmp_path):
    checkpoint = tmp_path / "global_step_20"
    checkpoint.mkdir()
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("20", encoding="utf-8")
    first = _metric_logging_trainer(_trainer_config(tmp_path))
    first._initialize_metric_logging()
    first._metric_archive_writer.close(interrupted=True)

    resumed = _metric_logging_trainer(_trainer_config(tmp_path))
    resumed._initialize_metric_logging()

    assert resumed._metric_resume_step == 20
    assert resumed._metric_archive_writer.manifest["legacy_before_step"] == 20
    resumed._metric_archive_writer.close()


def test_existing_archive_requires_commit_marker_for_selected_checkpoint_on_cpu(tmp_path):
    archive = MetricArchiveWriter(
        tmp_path / "diagnostics",
        logical_run_id="run-123",
        project="project",
        experiment="experiment",
    )
    archive.close()
    checkpoint = tmp_path / "global_step_20"
    checkpoint.mkdir()
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("20", encoding="utf-8")
    trainer = _metric_logging_trainer(_trainer_config(tmp_path))

    with pytest.raises(RuntimeError, match="no metrics commit marker"):
        trainer._initialize_metric_logging()


def test_existing_archive_rejects_conflicting_wandb_run_id_before_logger_init_on_cpu(
    tmp_path,
    monkeypatch,
):
    archive = MetricArchiveWriter(
        tmp_path / "diagnostics",
        logical_run_id="manifest-run",
        project="project",
        experiment="experiment",
    )
    archive.close()
    monkeypatch.setenv("WANDB_RUN_ID", "other-run")
    trainer = _metric_logging_trainer(_trainer_config(tmp_path))

    with pytest.raises(RuntimeError, match="identity mismatch"):
        trainer._initialize_metric_logging()


def test_checkpoint_payload_does_not_advance_tracker_until_explicit_commit_on_cpu(tmp_path):
    trainer = _bare_trainer(_trainer_config(tmp_path))
    trainer.global_steps = 4
    trainer.use_critic = False
    trainer.config.actor_rollout_ref = OmegaConf.create({"actor": {"checkpoint": {"async_save": False}}})

    class _ActorWorker:
        def save_checkpoint(self, local_path, remote_path, step, max_ckpt_to_keep):
            Path(local_path).mkdir(parents=True)
            assert remote_path is None
            assert step == 4
            assert max_ckpt_to_keep is None

    trainer.actor_rollout_wg = _ActorWorker()
    trainer.train_dataloader = SimpleNamespace(state_dict=lambda: {"step": 4})

    checkpoint_path = trainer._save_checkpoint_payload()

    tracker = tmp_path / "latest_checkpointed_iteration.txt"
    assert checkpoint_path == str(tmp_path / "global_step_4")
    assert not tracker.exists()
    assert (tmp_path / "global_step_4" / "data.pt").exists()

    trainer._commit_checkpoint_tracker(4)
    assert tracker.read_text(encoding="utf-8") == "4"


def test_balanced_checkpoint_payload_keeps_one_extra_generation_until_tracker_commit_on_cpu(tmp_path):
    trainer = _bare_trainer(_trainer_config(tmp_path))
    trainer.global_steps = 4
    trainer.use_critic = False
    trainer.config.actor_rollout_ref = OmegaConf.create({"actor": {"checkpoint": {"async_save": False}}})
    trainer.config.trainer.max_actor_ckpt_to_keep = 1
    observed_retention = []

    class _ActorWorker:
        def save_checkpoint(self, local_path, remote_path, step, max_ckpt_to_keep):
            Path(local_path).mkdir(parents=True)
            observed_retention.append(max_ckpt_to_keep)

    trainer.actor_rollout_wg = _ActorWorker()
    trainer.train_dataloader = SimpleNamespace(state_dict=lambda: {"step": 4})

    trainer._save_checkpoint_payload()

    assert observed_retention == [2]
    assert not (tmp_path / "latest_checkpointed_iteration.txt").exists()


@pytest.mark.parametrize("resume_step", [0, 20])
def test_initial_resume_validation_runs_but_is_not_logged_or_materialized_on_cpu(resume_step):
    trainer = object.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "val_before_train": True,
                "test_before_train": False,
                "val_only": True,
            }
        }
    )
    trainer.train_dataloader = [object()]
    trainer._metric_logging_settings = MetricLoggingSettings()
    trainer.checkpoint_manager = SimpleNamespace(update_weights=lambda _step: None)
    validation_calls = []
    trainer._load_checkpoint = lambda: setattr(trainer, "global_steps", resume_step)
    trainer._validate = lambda **kwargs: validation_calls.append(kwargs) or {"val-core/score": 1.0}
    trainer._validate_teacher_before_train = lambda **_kwargs: {}

    class _Logger:
        def __init__(self):
            self.calls = []

        def log(self, data, step):
            self.calls.append((data, step))

    logger = _Logger()
    trainer._fit_impl(logger)

    assert validation_calls == [{"include_format_metrics": True, "suppress_tracking": resume_step > 0}]
    if resume_step > 0:
        assert logger.calls == []
    else:
        assert logger.calls == [({"val-core/score": 1.0}, 0)]


def test_fresh_offline_val_only_flushes_terminal_history_without_affecting_resume_rule_on_cpu():
    trainer = object.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "val_before_train": True,
                "test_before_train": False,
                "val_only": True,
            }
        }
    )
    trainer.train_dataloader = [object()]
    trainer._metric_logging_settings = MetricLoggingSettings.from_mapping(
        {"mode": "balanced", "archive": {"enabled": True}}
    )
    trainer._metric_resume_step = None
    trainer.checkpoint_manager = SimpleNamespace(update_weights=lambda _step: None)
    trainer._load_checkpoint = lambda: setattr(trainer, "global_steps", 0)
    trainer._validate = lambda **_kwargs: {"val-core/score": 1.0}
    trainer._validate_teacher_before_train = lambda **_kwargs: {}
    recorded = []
    flushed = []
    trainer._record_metric_event = lambda *_args, **kwargs: recorded.append(kwargs["event_kind"])
    trainer._flush_offline_wandb_history = lambda *_args, **kwargs: flushed.append(kwargs["checkpoint_step"])

    trainer._fit_impl(SimpleNamespace())

    assert recorded == ["initial_validation"]
    assert flushed == [0]


def test_balanced_checkpoint_event_commits_ledger_before_tracker_and_replays_compact_offline_history_on_cpu(
    tmp_path,
):
    checkpoint_dir = tmp_path / "global_step_2"
    checkpoint_dir.mkdir()
    trainer = object.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {"default_local_dir": str(tmp_path)},
            "actor_rollout_ref": {"actor": {"checkpoint": {"async_save": False}}},
        }
    )
    trainer.global_steps = 2
    trainer._metric_logging_settings = MetricLoggingSettings.from_mapping(
        {"mode": "balanced", "archive": {"enabled": True, "shard_interval_steps": 20}}
    )
    trainer._metric_archive_writer = MetricArchiveWriter(
        tmp_path / "diagnostics",
        logical_run_id="run-1",
        project="DMPD",
        experiment="trainer-integration",
    )
    trainer._metric_router = MetricRouter("balanced")
    trainer._metric_artifact_uploader = None
    trainer._offline_live_metric_buffer = []
    trainer._pending_event_record_archives = []
    trainer._event_record_archive_warning_emitted = False
    trainer._queue_generation_archive(
        evaluation_scope="val",
        records=[{"input": "val question", "output": "complete val answer", "score": 1.0}],
    )
    trainer._queue_generation_archive(
        evaluation_scope="test",
        records=[{"input": "test question", "output": "complete test answer", "score": 0.0}],
    )
    trainer._queue_event_records_archive(
        metric_key="opd_intervention/outcome_pairs",
        records=[{"uid": "uid-1", "transition": "pre_wrong_post_correct"}],
    )
    trainer._queue_event_records_archive(
        metric_key="opd_intervention/future_gap_records",
        records=[{"uid": "uid-1", "accepted": True, "future_gap_reduction": 0.2}],
    )

    class _Logger:
        wandb_run_info = WandbRunInfo(
            mode="offline",
            run_id="run-1",
            sync_dir=None,
            sync_file=None,
        )
        logger = {"console": object(), "wandb": object()}

        def __init__(self):
            self.calls = []

        def log(self, data, step, backend=None):
            self.calls.append((dict(data), step, backend))

    logger = _Logger()
    trainer._record_metric_event(
        logger,
        {
            "actor/loss": 1.0,
            "bucket/bin/0000_0128/token_count": 4.0,
        },
        event_kind="train",
        checkpoint_payload_path=str(checkpoint_dir),
    )

    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text(encoding="utf-8") == "2"
    assert (checkpoint_dir / "metrics_commit.json").is_file()
    shard = trainer._metric_archive_writer.manifest["shards"][0]
    rows = pq.read_table(shard["path"]).to_pylist()
    deliveries = {row["metric_key"]: row["delivery"] for row in rows}
    assert deliveries["actor/loss"] == "live"
    assert deliveries["bucket/bin/0000_0128/token_count"] == "archive"
    shard = trainer._metric_archive_writer.pending_upload_shards()[0]
    event_files = {item.metric_key: item for item in shard.event_files}
    assert set(event_files) == {
        "val/generations_full",
        "test/generations_full",
        "opd_intervention/outcome_pairs",
        "opd_intervention/future_gap_records",
    }
    with gzip.open(event_files["val/generations_full"].path, "rt", encoding="utf-8") as stream:
        assert [json.loads(line) for line in stream] == [
            {"input": "val question", "output": "complete val answer", "score": 1.0}
        ]
    assert trainer._pending_event_record_archives == []
    assert trainer._future_gap_archive_serialization_seconds_pending > 0
    assert [backend for _data, _step, backend in logger.calls] == [["console"], ["wandb"]]
    assert all("bucket/bin/0000_0128/token_count" not in data for data, _step, _backend in logger.calls)
    session = trainer._metric_archive_writer.manifest["sessions"][trainer._metric_archive_writer.session_id]
    assert session["wandb_committed_ranges"] == [{"step_start": 2, "step_end": 2}]
    trainer._metric_archive_writer.close()


def test_full_metric_mode_warns_once_when_detailed_event_archive_is_unavailable_on_cpu(capsys):
    trainer = object.__new__(RayPPOTrainer)
    trainer.global_steps = 3
    trainer._metric_logging_settings = MetricLoggingSettings.from_mapping({"mode": "full"})
    trainer._metric_archive_writer = None
    trainer._event_record_archive_warning_emitted = False

    trainer._queue_event_records_archive(metric_key="opd_intervention/outcome_pairs", records=[{"uid": "one"}])
    trainer._queue_event_records_archive(metric_key="test-legacy/generations_full", records=[{"uid": "two"}])

    assert capsys.readouterr().out.count("JSONL/Artifact records are unavailable") == 1
    assert trainer._event_record_archive_warning_emitted is True


def test_offline_history_recovery_replays_heatmaps_but_not_generation_attachments_on_cpu(
    tmp_path,
    monkeypatch,
):
    heatmap_path = tmp_path / "heatmap.png"
    heatmap_path.write_bytes(b"\x89PNG\r\n")
    generation_path = tmp_path / "generations.jsonl.gz"
    generation_path.write_bytes(b"compressed records")
    event = SimpleNamespace(
        global_step=2,
        session_id="old-session",
        event_seq=4,
        event_id="event-4",
        metrics={"actor/loss": 1.0},
        event_files=(
            SimpleNamespace(metric_key="opd/gap_abs_heatmap", path=str(heatmap_path)),
            SimpleNamespace(metric_key="val/generations_full", path=str(generation_path)),
        ),
    )
    committed_ranges = []
    recovered_sessions = []
    writer = SimpleNamespace(
        unrecorded_offline_committed_live_events=lambda **_kwargs: (event,),
        record_wandb_committed_range=lambda **kwargs: committed_ranges.append(kwargs),
        mark_wandb_history_recovered=lambda session_ids: recovered_sessions.append(session_ids),
    )
    trainer = object.__new__(RayPPOTrainer)
    trainer._metric_archive_writer = writer
    trainer._metric_resume_step = 2
    trainer._resolve_resume_context = lambda: _ResolvedCheckpointContext(path=str(tmp_path), step=2)

    wandb = ModuleType("wandb")
    wandb.Image = lambda path: f"image:{path}"
    monkeypatch.setitem(sys.modules, "wandb", wandb)

    class _Logger:
        wandb_run_info = WandbRunInfo(
            mode="offline",
            run_id="run-1",
            sync_dir=None,
            sync_file=None,
        )

        def __init__(self):
            self.calls = []

        def log(self, data, step, backend=None):
            self.calls.append((dict(data), step, backend))

    logger = _Logger()
    trainer._recover_missing_offline_wandb_history(logger)

    assert logger.calls == [
        (
            {
                "actor/loss": 1.0,
                "opd/gap_abs_heatmap": f"image:{heatmap_path}",
            },
            2,
            ["wandb"],
        )
    ]
    assert committed_ranges == [{"step_start": 2, "step_end": 2, "recovered_event_ids": ("event-4",)}]
    assert recovered_sessions == [("old-session",)]
