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

import os
import sys
from types import ModuleType, SimpleNamespace

import pytest

from verl.utils.tracking import Tracking, WandbResumeSpec


class _Artifact:
    def __init__(self, name, type, metadata):
        self.name = name
        self.type = type
        self.metadata = metadata
        self.files = []

    def add_file(self, path, name):
        self.files.append((path, name))


class _ArtifactHandle:
    def __init__(self):
        self.wait_calls = []

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)


class _Run:
    def __init__(self, *, run_id, mode, sync_dir, sync_file):
        self.id = run_id
        self.offline = mode == "offline"
        self.settings = SimpleNamespace(mode=mode, run_id=run_id, sync_dir=sync_dir, sync_file=sync_file)
        self.artifact_calls = []

    def log_artifact(self, artifact, aliases):
        handle = _ArtifactHandle()
        self.artifact_calls.append((artifact, aliases, handle))
        return handle


def _fake_wandb(monkeypatch, tmp_path, *, generated_run_id="generated-id"):
    wandb = ModuleType("wandb")
    wandb.run = None
    wandb.init_calls = []
    wandb.resume_env_during_init = []
    wandb.finish_calls = []
    wandb.Artifact = _Artifact
    wandb.Settings = lambda **kwargs: SimpleNamespace(**kwargs)

    def init(**kwargs):
        wandb.init_calls.append(kwargs)
        wandb.resume_env_during_init.append(os.environ.get("WANDB_RESUME"))
        mode = getattr(kwargs["settings"], "mode", "online") if kwargs["settings"] is not None else "online"
        run_id = kwargs.get("id", generated_run_id)
        wandb.run = _Run(
            run_id=run_id,
            mode=mode,
            sync_dir=tmp_path / "wandb-session",
            sync_file=tmp_path / "wandb-session" / f"run-{run_id}.wandb",
        )
        return wandb.run

    wandb.init = init
    wandb.finish = lambda exit_code=0: wandb.finish_calls.append(exit_code)
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    return wandb


def _clear_wandb_env(monkeypatch):
    for name in ("WANDB_MODE", "WANDB_RUN_ID", "WANDB_RESUME", "WANDB_ENTITY"):
        monkeypatch.delenv(name, raising=False)


def test_tracking_preserves_default_wandb_init_behavior_on_cpu(monkeypatch, tmp_path):
    _clear_wandb_env(monkeypatch)
    wandb = _fake_wandb(monkeypatch, tmp_path)

    tracking = Tracking("project", "experiment", default_backend="wandb")

    init_kwargs = wandb.init_calls[0]
    assert "id" not in init_kwargs
    assert "resume" not in init_kwargs
    assert "resume_from" not in init_kwargs
    assert tracking.wandb_run_info.run_id == "generated-id"


@pytest.mark.parametrize(
    ("resume_spec", "expected"),
    [
        (WandbResumeSpec(run_id="run-123", resume="must"), {"id": "run-123", "resume": "must"}),
        (
            WandbResumeSpec(run_id="run-123", resume_from="run-123?_step=20"),
            {"id": "run-123", "resume_from": "run-123?_step=20"},
        ),
    ],
)
def test_tracking_passes_only_one_online_wandb_resume_directive_on_cpu(monkeypatch, tmp_path, resume_spec, expected):
    _clear_wandb_env(monkeypatch)
    monkeypatch.setenv("WANDB_MODE", "online")
    wandb = _fake_wandb(monkeypatch, tmp_path)

    Tracking("project", "experiment", default_backend="wandb", wandb_resume=resume_spec)

    init_kwargs = wandb.init_calls[0]
    assert {key: init_kwargs[key] for key in expected} == expected
    assert not ({"resume", "resume_from"} <= init_kwargs.keys())


def test_explicit_resume_from_temporarily_suppresses_inherited_wandb_resume_on_cpu(monkeypatch, tmp_path):
    _clear_wandb_env(monkeypatch)
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.setenv("WANDB_RESUME", "must")
    wandb = _fake_wandb(monkeypatch, tmp_path)

    Tracking(
        "project",
        "experiment",
        default_backend="wandb",
        wandb_resume=WandbResumeSpec(run_id="run-123", resume_from="run-123?_step=20"),
    )

    assert wandb.resume_env_during_init == [None]
    assert os.environ["WANDB_RESUME"] == "must"
    assert wandb.init_calls[0]["resume_from"] == "run-123?_step=20"
    assert "resume" not in wandb.init_calls[0]


def test_wandb_resume_spec_rejects_conflicts_and_run_id_mismatch_on_cpu():
    with pytest.raises(ValueError, match="mutually exclusive"):
        WandbResumeSpec(run_id="run-123", resume="must", resume_from="run-123?_step=20")
    with pytest.raises(ValueError, match="same run_id"):
        WandbResumeSpec(run_id="run-123", resume_from="other-run?_step=20")


def test_offline_wandb_reuses_id_but_suppresses_resume_on_cpu(monkeypatch, tmp_path, caplog):
    _clear_wandb_env(monkeypatch)
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_RESUME", "must")
    wandb = _fake_wandb(monkeypatch, tmp_path)

    tracking = Tracking(
        "project",
        "experiment",
        default_backend="wandb",
        wandb_resume=WandbResumeSpec(run_id="run-123", resume="must"),
    )

    init_kwargs = wandb.init_calls[0]
    assert init_kwargs["id"] == "run-123"
    assert init_kwargs["resume"] == "never"
    assert "resume_from" not in init_kwargs
    assert "does not resume local offline runs" in caplog.text

    run_info = tracking.wandb_run_info
    assert run_info.mode == "offline"
    assert run_info.run_id == "run-123"
    assert run_info.sync_dir == (tmp_path / "wandb-session").resolve()
    assert run_info.sync_file == (tmp_path / "wandb-session" / "run-run-123.wandb").resolve()
    assert run_info.is_offline
    assert not run_info.is_online


def test_online_artifact_submission_returns_wait_handle_without_waiting_on_cpu(monkeypatch, tmp_path):
    _clear_wandb_env(monkeypatch)
    monkeypatch.setenv("WANDB_MODE", "online")
    wandb = _fake_wandb(monkeypatch, tmp_path)
    tracking = Tracking("project", "experiment", default_backend="wandb")
    shard = tmp_path / "metrics.parquet"
    shard.write_bytes(b"closed shard")

    submission = tracking.log_wandb_artifact(
        shard,
        artifact_name="run-123-training-diagnostics",
        artifact_type="training-diagnostics",
        metadata={"sha256": "abc"},
        aliases=["latest", "steps-1-20", "latest"],
    )

    artifact, aliases, handle = wandb.run.artifact_calls[0]
    assert artifact.name == "run-123-training-diagnostics"
    assert artifact.type == "training-diagnostics"
    assert artifact.metadata == {"sha256": "abc"}
    assert artifact.files == [(str(shard.resolve()), "metrics.parquet")]
    assert aliases == ["latest", "steps-1-20"]
    assert submission.aliases == ("latest", "steps-1-20")
    assert submission.wait_handle is handle
    assert submission.requires_wait
    assert handle.wait_calls == []


def test_artifact_submission_can_bundle_metadata_and_heatmap_files_on_cpu(monkeypatch, tmp_path):
    _clear_wandb_env(monkeypatch)
    monkeypatch.setenv("WANDB_MODE", "online")
    wandb = _fake_wandb(monkeypatch, tmp_path)
    tracking = Tracking("project", "experiment", default_backend="wandb")
    shard = tmp_path / "metrics.parquet"
    metadata = tmp_path / "metrics.metadata.json"
    heatmap = tmp_path / "gap.png"
    shard.write_bytes(b"closed shard")
    metadata.write_text("{}", encoding="utf-8")
    heatmap.write_bytes(b"png")

    tracking.log_wandb_artifact(
        shard,
        artifact_name="run-123-training-diagnostics",
        artifact_type="training-diagnostics",
        additional_files={
            "metrics.metadata.json": metadata,
            "media/gap.png": heatmap,
        },
    )

    artifact, _aliases, _handle = wandb.run.artifact_calls[0]
    assert artifact.files == [
        (str(shard.resolve()), "metrics.parquet"),
        (str(metadata.resolve()), "metrics.metadata.json"),
        (str(heatmap.resolve()), "media/gap.png"),
    ]


def test_offline_artifact_submission_is_staged_without_a_wait_handle_on_cpu(monkeypatch, tmp_path):
    _clear_wandb_env(monkeypatch)
    monkeypatch.setenv("WANDB_MODE", "offline")
    wandb = _fake_wandb(monkeypatch, tmp_path)
    tracking = Tracking("project", "experiment", default_backend="wandb")
    shard = tmp_path / "metrics.parquet"
    shard.write_bytes(b"closed shard")

    submission = tracking.log_wandb_artifact(
        shard,
        artifact_name="run-123-training-diagnostics",
        artifact_type="training-diagnostics",
        aliases=["steps-1-20"],
    )

    assert len(wandb.run.artifact_calls) == 1
    assert submission.wait_handle is None
    assert not submission.requires_wait
    assert submission.run_info.is_offline


def test_artifact_submission_requires_an_active_wandb_backend_and_existing_file_on_cpu(tmp_path):
    tracking = Tracking("project", "experiment", default_backend="console")

    with pytest.raises(FileNotFoundError, match="existing file"):
        tracking.log_wandb_artifact(
            tmp_path / "missing.parquet",
            artifact_name="diagnostics",
            artifact_type="training-diagnostics",
        )

    shard = tmp_path / "metrics.parquet"
    shard.write_bytes(b"closed shard")
    with pytest.raises(RuntimeError, match="wandb backend"):
        tracking.log_wandb_artifact(
            shard,
            artifact_name="diagnostics",
            artifact_type="training-diagnostics",
        )
