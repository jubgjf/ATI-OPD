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
"""Single-queue W&B Artifact delivery for immutable metric archive shards."""

from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verl.trainer.ppo.metric_archive import MetricArchiveWriter, ShardDescriptor, ShardState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _UploadResult:
    shard_id: str
    state: ShardState
    artifact_ref: str | None = None
    delivery_session_id: str | None = None
    error: str | None = None


class MetricArtifactUploader:
    """Upload or offline-stage archive shards without blocking the training loop."""

    def __init__(
        self,
        *,
        writer: MetricArchiveWriter,
        tracking: Any,
        project: str,
        entity: str | None,
        finish_timeout_seconds: float,
    ):
        self.writer = writer
        self.tracking = tracking
        self.project = project
        self.entity = entity
        self.finish_timeout_seconds = float(finish_timeout_seconds)
        self.artifact_name = f"{writer.logical_run_id}-training-diagnostics"
        self.artifact_type = "training-diagnostics"
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="metric-artifact")
        self._futures: dict[Future[_UploadResult], str] = {}
        self._pending_shard_ids: set[str] = set()
        self._closed = False

    @property
    def enabled(self) -> bool:
        run_info = self.tracking.wandb_run_info
        return run_info is not None and run_info.mode != "disabled"

    def submit_pending(self) -> None:
        """Submit every eligible shard once to the single background queue."""
        if self._closed or not self.enabled:
            return
        self.drain_completed()
        run_info = self.tracking.wandb_run_info
        assert run_info is not None
        for shard in self.writer.pending_upload_shards():
            if shard.shard_id in self._pending_shard_ids:
                continue
            if run_info.is_online and shard.state != ShardState.SUBMITTED.value:
                self.writer.mark_shard_state(shard.shard_id, ShardState.SUBMITTED)
            future = self._executor.submit(self._deliver, shard)
            self._futures[future] = shard.shard_id
            self._pending_shard_ids.add(shard.shard_id)

    def drain_completed(self) -> None:
        """Apply completed worker results to the manifest on the trainer thread."""
        for future, shard_id in list(self._futures.items()):
            if not future.done():
                continue
            self._futures.pop(future)
            self._pending_shard_ids.discard(shard_id)
            try:
                result = future.result()
            except BaseException as exc:  # pragma: no cover - defensive boundary
                result = _UploadResult(
                    shard_id=shard_id,
                    state=ShardState.FAILED_RETRYABLE,
                    error=f"{type(exc).__name__}: {exc}",
                )
            try:
                self.writer.mark_shard_state(
                    result.shard_id,
                    result.state,
                    error=result.error,
                    artifact_ref=result.artifact_ref,
                    delivery_session_id=result.delivery_session_id,
                )
            except ValueError:
                # A retry may discover that a SUBMITTED shard already exists remotely.
                # Preserve a terminal UPLOADED state instead of weakening it.
                current = next(
                    (item for item in self.writer.manifest["shards"] if item["shard_id"] == result.shard_id),
                    None,
                )
                if current is None or current["state"] != result.state.value:
                    raise

    def finish(self) -> None:
        """Wait for the configured bounded interval and persist completed results."""
        if self._closed:
            return
        self._closed = True
        futures = tuple(self._futures)
        if futures:
            wait(futures, timeout=self.finish_timeout_seconds)
            self.drain_completed()
        remaining = len(self._futures)
        if remaining:
            logger.warning(
                "%d metric Artifact upload(s) remain pending after %.1f seconds; "
                "their manifest state will be retried on resume.",
                remaining,
                self.finish_timeout_seconds,
            )
        self._executor.shutdown(wait=False)

    def _deliver(self, shard: ShardDescriptor) -> _UploadResult:
        run_info = self.tracking.wandb_run_info
        if run_info is None or run_info.mode == "disabled":
            return _UploadResult(shard_id=shard.shard_id, state=ShardState.FAILED_RETRYABLE, error="W&B disabled")

        range_alias = f"steps-{shard.step_start}-{shard.step_end}"
        artifact_ref = f"{self.artifact_name}:{range_alias}"
        active_lineage = shard.checkpoint_step is not None
        aliases = ("latest", range_alias) if active_lineage else (range_alias,)
        try:
            if run_info.is_online and self._remote_artifact_matches(range_alias, shard.sha256):
                return _UploadResult(
                    shard_id=shard.shard_id,
                    state=ShardState.UPLOADED,
                    artifact_ref=artifact_ref,
                )

            additional_files = {
                f"shards/{Path(shard.metadata_path).name}": shard.metadata_path,
            }
            for event_file in shard.event_files:
                additional_files[f"attachments/{Path(event_file.path).name}"] = event_file.path
            submission = self.tracking.log_wandb_artifact(
                shard.path,
                artifact_name=self.artifact_name,
                artifact_type=self.artifact_type,
                artifact_file_name=f"shards/{Path(shard.path).name}",
                additional_files=additional_files,
                aliases=aliases,
                metadata={
                    "logical_run_id": self.writer.logical_run_id,
                    "session_id": shard.session_id,
                    "delivery_session_id": self.writer.session_id,
                    "checkpoint_step": shard.checkpoint_step,
                    "step_start": shard.step_start,
                    "step_end": shard.step_end,
                    "shard_id": shard.shard_id,
                    "sha256": shard.sha256,
                    "active_lineage": active_lineage,
                },
            )
            if submission.requires_wait:
                submission.wait_handle.wait(timeout=self.finish_timeout_seconds)
                target_state = ShardState.UPLOADED
            else:
                target_state = ShardState.STAGED_OFFLINE
            return _UploadResult(
                shard_id=shard.shard_id,
                state=target_state,
                artifact_ref=artifact_ref,
                delivery_session_id=self.writer.session_id,
            )
        except BaseException as exc:
            return _UploadResult(
                shard_id=shard.shard_id,
                state=ShardState.FAILED_RETRYABLE,
                error=f"{type(exc).__name__}: {exc}",
            )

    def _remote_artifact_matches(self, range_alias: str, sha256: str) -> bool:
        """Check the stable range alias before creating a retry version."""
        try:
            import wandb

            api = wandb.Api()
            entity = self.entity or api.default_entity
            if not entity:
                return False
            artifact = api.artifact(
                f"{entity}/{self.project}/{self.artifact_name}:{range_alias}",
                type=self.artifact_type,
            )
            return str((artifact.metadata or {}).get("sha256", "")) == sha256
        except Exception:
            return False


def wait_for_artifact_manifest_updates(
    uploader: MetricArtifactUploader,
    *,
    poll_interval_seconds: float = 0.05,
    timeout_seconds: float = 5.0,
) -> None:
    """Test/debug helper which never participates in the training path."""
    deadline = time.monotonic() + timeout_seconds
    while uploader._futures and time.monotonic() < deadline:
        uploader.drain_completed()
        if uploader._futures:
            time.sleep(poll_interval_seconds)
    uploader.drain_completed()
