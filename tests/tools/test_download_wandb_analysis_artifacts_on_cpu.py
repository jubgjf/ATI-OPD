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

from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from run import download_wandb_analysis_artifacts as downloader


class _Entry:
    def __init__(self, path: str, payload: bytes) -> None:
        self.path = path
        self.payload = payload
        self.size = len(payload)
        self.digest = base64.b64encode(hashlib.md5(payload, usedforsecurity=False).digest()).decode("ascii")

    def download(self, *, root: str) -> str:
        destination = Path(root) / self.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.payload)
        return str(destination)


class _Artifact:
    type = downloader.TRAINING_DIAGNOSTICS_TYPE

    def __init__(self, run_id: str, version: str, entry: _Entry) -> None:
        self.name = f"{run_id}-training-diagnostics:{version}"
        self.version = version
        self.digest = f"artifact-{version}"
        self.manifest = SimpleNamespace(entries={entry.path: entry})

    def get_entry(self, name: str) -> _Entry:
        return self.manifest.entries[name]


class _Run:
    def __init__(self, run_id: str, artifacts: list[_Artifact]) -> None:
        self.id = run_id
        self._artifacts = artifacts

    def logged_artifacts(self) -> list[_Artifact]:
        return self._artifacts


def _fake_run() -> _Run:
    run_id = "run-123"
    artifacts = [
        _Artifact(
            run_id,
            "v1",
            _Entry("shards/metrics-step-000000021-000000040-b.parquet", b"second"),
        ),
        _Artifact(
            run_id,
            "v0",
            _Entry("shards/metrics-step-000000001-000000020-a.parquet", b"first"),
        ),
    ]
    return _Run(run_id, artifacts)


def test_sync_training_diagnostics_shards_is_incremental_and_version_ordered(tmp_path: Path) -> None:
    run = _fake_run()

    downloaded, skipped, manifest = downloader.sync_training_diagnostics_shards(
        run,
        tmp_path,
        force=False,
        dry_run=False,
    )

    assert (downloaded, skipped) == (2, 0)
    assert [version["version"] for version in manifest["versions"]] == ["v0", "v1"]
    root = tmp_path / "logged_artifacts" / "training-diagnostics"
    assert (root / "v0/shards/metrics-step-000000001-000000020-a.parquet").read_bytes() == b"first"
    assert (root / "v1/shards/metrics-step-000000021-000000040-b.parquet").read_bytes() == b"second"
    assert (root / "manifest.json").is_file()

    downloaded, skipped, _ = downloader.sync_training_diagnostics_shards(
        run,
        tmp_path,
        force=False,
        dry_run=False,
    )
    assert (downloaded, skipped) == (0, 2)


def test_sync_training_diagnostics_shards_dry_run_writes_nothing(tmp_path: Path) -> None:
    downloaded, skipped, manifest = downloader.sync_training_diagnostics_shards(
        _fake_run(),
        tmp_path,
        force=False,
        dry_run=True,
    )

    assert (downloaded, skipped) == (2, 0)
    assert len(manifest["versions"]) == 2
    assert not (tmp_path / "logged_artifacts").exists()


def test_training_diagnostics_artifacts_rejects_unexpected_name() -> None:
    run = _fake_run()
    run._artifacts[0].name = "another-run-training-diagnostics:v1"

    with pytest.raises(RuntimeError, match="Unexpected training-diagnostics artifact"):
        downloader.training_diagnostics_artifacts(run)


def test_sync_training_diagnostics_shards_rejects_unsafe_entry(tmp_path: Path) -> None:
    run_id = "run-123"
    artifact = _Artifact(run_id, "v0", _Entry("shards/../escape.parquet", b"bad"))

    with pytest.raises(ValueError, match="Unsafe W&B file path"):
        downloader.sync_training_diagnostics_shards(
            _Run(run_id, [artifact]),
            tmp_path,
            force=False,
            dry_run=False,
        )
