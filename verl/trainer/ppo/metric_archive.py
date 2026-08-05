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
"""Low-cardinality metric routing and crash-safe Parquet metric archives.

This module deliberately has no dependency on the PPO trainer or W&B.  The
trainer owns the ordering between model checkpoints, :class:`MetricArchiveWriter`
and the tracking backend, while this module owns:

* deterministic ``full``/``balanced`` metric routing;
* a complete numeric metric ledger;
* immutable, checksummed Parquet shards and an atomic manifest;
* checkpoint metric commit markers and resume reconciliation.

The archive is driver-only.  ``MetricArchiveWriter`` takes an exclusive
process lock for its lifetime so that two trainer drivers cannot append to the
same logical run.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import numbers
import os
import re
import shlex
import shutil
import socket
import time
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

METRIC_ARCHIVE_SCHEMA_VERSION = 1
METRIC_COMMIT_FILENAME = "metrics_commit.json"

_POSITION_BIN_RE = re.compile(r"(?:^|/)[^/]*_bin/(?P<start>\d+)[_-](?P<end>\d+)(?:/|$)")
_BUCKET_BIN_RE = re.compile(r"^bucket/bin/(?P<start>\d+)[_-](?P<end>\d+)(?:/|$)")
_COARSE_RE = re.compile(r"(?:^|/)coarse/(?P<region>[^/]+)(?:/|$)")
_TOPK_RE = re.compile(r"(?:^|[/_])top(?P<topk>\d+)(?=[/_]|$)")
_OPD_OVERLAP_TOPK_RE = re.compile(r"^opd/overlap/top(?P<topk>\d+)(?:_|/)")
_LOW_K_SUFFIX_RE = re.compile(r"_top(?P<topk>\d+)(?=/|$)")
_BOOTSTRAP_STAT_RE = re.compile(r"(?:^|/)(?:mean|std|best|worst|maj)@\d+(?:/(?:mean|std))?$")

_ALWAYS_ARCHIVE_PREFIXES = (
    "bucket/bin/",
    "bucket/coarse/",
    "distillation/coverage/",
    "student/prob_topk/",
    "teacher/prob_topk/",
    "special_tokens/resolved/",
)
_DYNAMIC_REASON_PREFIXES = (
    "opd_intervention/skipped_reason/",
    "opd_intervention/local_effect/unavailable_reason/",
    "opd_intervention/candidates/threshold_source/",
    "opd_intervention/candidates/invalid_reason/",
)
_STATIC_DIAGNOSTIC_RE = re.compile(
    r"^contagion/bad_tail_(?:min_position|deep_position|extreme_position|"
    r"gap_threshold|reward_threshold|teacher_prob_threshold|overlap_threshold|entropy_threshold)$"
)
_KNOWN_ALIAS_KEYS = frozenset(
    {
        # Exact aliases emitted by OPD diagnostics.  The structured counterpart
        # remains live and both names remain present in the ledger.
        "opd/gap_abs/high_fraction",
        "opd/overlap/teacher_sampled_prob_mean",
        "opd/overlap/teacher_sampled_prob_p01",
        "opd/overlap/teacher_sampled_prob_p10",
        "opd/overlap/teacher_sampled_prob_p50",
        "contagion/bad_tail_loss_token_fraction",
        "contagion/bad_tail_12k_token_count",
        "contagion/carrier_loss_share",
        "contagion/current_bad_tail_mass",
        "contagion/current_bad_tail_loss_share",
        "contagion/current_carrier_ratio",
        "response_length/right_tail_clip_count",
        "response_length/main_mass_mean",
    }
)
_EVAL_AUX_PREFIXES = ("val-aux/", "test-aux/", "teacher-val-aux/")
_REPAIR_EVAL_SCOPES = ("val-oracle-repair", "test-oracle-repair", "val-repair", "test-repair")
_REPAIR_EVAL_AUX_PREFIXES = tuple(f"{scope}-aux/" for scope in _REPAIR_EVAL_SCOPES)
_REPAIR_EVAL_CORE_PREFIXES = tuple(f"{scope}-core/" for scope in _REPAIR_EVAL_SCOPES)
_REPAIR_EVAL_LIVE_CORE_RE = re.compile(r"/(?:acc|reward)/(?:mean|paired_delta)@\d+$")
_LIVE_INTERVENTION_KEYS = frozenset(
    {
        "opd_intervention/rate",
        "opd_intervention/first_position_frac_mean",
        "opd_intervention/repair_boundary/entry/teacher_self_nll_first16_mean",
        "opd_intervention/repair_boundary/entry/student_nll_on_teacher_first16_mean",
        "opd_intervention/repair_boundary/exit/active_gap_after_first16_mean",
        "opd_intervention/local_effect_v2/active_gap/reduction_first16_mean",
        "opd_intervention/local_effect_v2/active_gap/worsened_fraction_first16",
        "opd_intervention/local_effect_v2/pair_fraction_first16",
        "opd_intervention/outcome/pre_correct_post_wrong_fraction_among_repaired",
        "opd_intervention/outcome/pre_wrong_post_correct_fraction_among_repaired",
        "opd_intervention/outcome/net_accuracy_delta_all_samples",
    }
)
_REDUNDANT_SUMMARY_PREFIXES = (
    "format/think_start_position/",
    "opd/response_len/",
    "response_length_non_aborted/",
    "rollout/response_length/",
    "timing_s/agent_loop/",
)
_REDUNDANT_AGGREGATE_FIELDS = {
    "prompt_length/": frozenset(
        {
            "count",
            "max",
            "max_prompt_length",
            "min",
            "p50",
            "p75",
            "p95",
            "p99",
            "std",
        }
    ),
    "response_length/": frozenset(
        {
            "clipped_count",
            "clipped_mean",
            "count",
            "max_response_length",
            "nonclipped_count",
            "p50",
            "p75",
            "p95",
            "p99",
            "std",
            "survival_gt_1k",
            "survival_gt_2k",
            "survival_gt_8k",
            "survival_gt_16k",
        }
    ),
    "distillation/token_reward_direct/": frozenset(
        {
            "available_topk",
            "combined_with_task_rewards",
            "count",
            "max",
            "min",
            "p50",
            "p75",
            "p90",
            "p95",
            "p99",
        }
    ),
    "val-aux/response_length/": frozenset({"count", "p75", "p95", "std"}),
    "test-aux/response_length/": frozenset({"count", "p75", "p95", "std"}),
    "teacher-val-aux/response_length/": frozenset({"count", "p75", "p95", "std"}),
}
_REDUNDANT_SUMMARY_KEYS = frozenset(
    {
        "critic/returns/mean",
        "critic/rewards/mean",
        "distillation/entropy_gap_residual_bucket_approximation",
        "distillation/student_sampled_prob/p10",
        "distillation/student_sampled_prob/p50",
        "distillation/teacher_sampled_prob/p01",
        "distillation/teacher_sampled_prob/p10",
        "distillation/teacher_sampled_prob/p50",
        "global_seqlen/balanced_max",
        "global_seqlen/balanced_min",
        "global_seqlen/max",
        "global_seqlen/min",
        "global_seqlen/minmax_diff",
        "opd/gap_abs/available",
        "opd/gap_abs/high_count_max",
        "opd/gap_abs/high_count_mean",
        "opd/teacher_logprob_nonfinite/count_max",
        "response/survival_no_think_end_by_4k",
        "response/survival_no_think_end_by_12k",
        "rollout/repetition_eligible_rate_token_zlib",
        "rollout/repetition_eligible_rate_zlib",
        "timing_s/opd_diagnostics_submit",
        "timing_s/start_profile",
        "timing_s/stop_profile",
        "training/rollout_probs_diff_max",
        "training/rollout_probs_diff_std",
        "training/rollout_probs_diff_valid",
    }
)
_LIVE_ROLLOUT_CORRECTION_KEYS = frozenset({"rollout_corr/k3_kl", "rollout_corr/kl"})
_HEALTH_PREFIX = "metric_archive"
_ROUTER_HEALTH_KEYS = frozenset(
    {
        f"{_HEALTH_PREFIX}/live_key_count",
        f"{_HEALTH_PREFIX}/archived_key_count",
        f"{_HEALTH_PREFIX}/numeric_key_count",
    }
)


class ShardState(str, Enum):
    """Persistent lifecycle states for an immutable metric shard."""

    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CHECKPOINT_COMMITTED = "CHECKPOINT_COMMITTED"
    STAGED_OFFLINE = "STAGED_OFFLINE"
    SUBMITTED = "SUBMITTED"
    UPLOADED = "UPLOADED"
    FAILED_RETRYABLE = "FAILED_RETRYABLE"
    ORPHANED = "ORPHANED"


@dataclass(frozen=True)
class MetricArchiveSettings:
    """Archive settings parsed from ``trainer.metric_logging.archive``."""

    enabled: bool = False
    local_dir: str | None = None
    shard_interval_steps: int = 20
    upload_to_wandb: str = "auto"
    finish_timeout_seconds: float = 60.0

    @classmethod
    def from_mapping(cls, config: Any) -> MetricArchiveSettings:
        config = config or {}
        result = cls(
            enabled=bool(_config_get(config, "enabled", False)),
            local_dir=_config_get(config, "local_dir", None),
            shard_interval_steps=int(_config_get(config, "shard_interval_steps", 20)),
            upload_to_wandb=str(_config_get(config, "upload_to_wandb", "auto")),
            finish_timeout_seconds=float(_config_get(config, "finish_timeout_seconds", 60.0)),
        )
        if result.shard_interval_steps <= 0:
            raise ValueError("trainer.metric_logging.archive.shard_interval_steps must be positive.")
        if result.upload_to_wandb not in {"auto", "disabled"}:
            raise ValueError("trainer.metric_logging.archive.upload_to_wandb must be 'auto' or 'disabled'.")
        if result.finish_timeout_seconds < 0:
            raise ValueError("trainer.metric_logging.archive.finish_timeout_seconds must be non-negative.")
        return result


@dataclass(frozen=True)
class MetricLoggingSettings:
    """Settings parsed from ``trainer.metric_logging``."""

    mode: str = "full"
    resume_policy: str = "strict"
    live_key_warning_threshold: int = 250
    archive: MetricArchiveSettings = MetricArchiveSettings()

    @classmethod
    def from_mapping(cls, config: Any) -> MetricLoggingSettings:
        config = config or {}
        result = cls(
            mode=str(_config_get(config, "mode", "full")),
            resume_policy=str(_config_get(config, "resume_policy", "strict")),
            live_key_warning_threshold=int(_config_get(config, "live_key_warning_threshold", 250)),
            archive=MetricArchiveSettings.from_mapping(_config_get(config, "archive", {})),
        )
        if result.mode not in {"full", "balanced"}:
            raise ValueError("trainer.metric_logging.mode must be 'full' or 'balanced'.")
        if result.resume_policy != "strict":
            raise ValueError("Only trainer.metric_logging.resume_policy='strict' is currently supported.")
        if result.live_key_warning_threshold <= 0:
            raise ValueError("trainer.metric_logging.live_key_warning_threshold must be positive.")
        if result.mode == "balanced" and not result.archive.enabled:
            raise ValueError("Balanced metric logging requires trainer.metric_logging.archive.enabled=true.")
        return result


@dataclass(frozen=True)
class MetricRoute:
    """A lossless routing result for one metric event."""

    live_metrics: dict[str, Any]
    detailed_metrics: dict[str, Any]
    numeric_metrics: dict[str, float]
    value_dtypes: dict[str, str]
    deliveries: dict[str, str]
    live_numeric_count: int
    archived_numeric_count: int

    def assert_lossless(self, original_metrics: Mapping[str, Any]) -> None:
        """Raise if routing lost or duplicated an input key."""

        routed_keys = set(self.live_metrics) | set(self.detailed_metrics)
        routed_keys -= _ROUTER_HEALTH_KEYS - set(original_metrics)
        overlap = set(self.live_metrics) & set(self.detailed_metrics)
        if overlap or routed_keys != set(original_metrics):
            raise RuntimeError(
                "Metric routing must be a lossless partition: "
                f"missing={sorted(set(original_metrics) - routed_keys)}, "
                f"duplicated={sorted(overlap)}, extra={sorted(routed_keys - set(original_metrics))}."
            )


class MetricRouter:
    """Route high-cardinality diagnostics away from live tracking backends."""

    def __init__(
        self,
        mode: str,
        *,
        configured_topk: int | None = None,
        live_key_warning_threshold: int = 250,
        include_health_metrics: bool = True,
    ):
        if mode not in {"full", "balanced"}:
            raise ValueError(f"Metric routing mode must be 'full' or 'balanced', got {mode!r}.")
        if configured_topk is not None and configured_topk <= 0:
            raise ValueError(f"configured_topk must be positive when set, got {configured_topk}.")
        if live_key_warning_threshold <= 0:
            raise ValueError("live_key_warning_threshold must be positive.")
        self.mode = mode
        self.configured_topk = configured_topk
        self.live_key_warning_threshold = live_key_warning_threshold
        self.include_health_metrics = include_health_metrics

    def route(self, metrics: Mapping[str, Any]) -> MetricRoute:
        """Return a lossless live/detailed partition and the complete numeric ledger."""

        effective_topk = self.configured_topk or _infer_configured_topk(metrics)
        live_metrics: dict[str, Any] = {}
        detailed_metrics: dict[str, Any] = {}
        numeric_metrics: dict[str, float] = {}
        value_dtypes: dict[str, str] = {}
        deliveries: dict[str, str] = {}

        for key, value in metrics.items():
            numeric = _coerce_numeric(value)
            is_detailed = self.mode == "balanced" and _should_archive_metric(key, effective_topk)
            destination = detailed_metrics if is_detailed else live_metrics
            destination[key] = value
            if numeric is not None:
                numeric_metrics[key] = numeric[0]
                value_dtypes[key] = numeric[1]
                deliveries[key] = "archive" if is_detailed else "live"

        live_numeric_count = sum(delivery == "live" for delivery in deliveries.values())
        archived_numeric_count = sum(delivery == "archive" for delivery in deliveries.values())
        if self.mode == "balanced" and self.include_health_metrics:
            health = {
                f"{_HEALTH_PREFIX}/live_key_count": float(live_numeric_count),
                f"{_HEALTH_PREFIX}/archived_key_count": float(archived_numeric_count),
                f"{_HEALTH_PREFIX}/numeric_key_count": float(len(numeric_metrics)),
            }
            for key, value in health.items():
                live_metrics[key] = value
                numeric_metrics[key] = value
                value_dtypes[key] = "float64"
                deliveries[key] = "live"
            live_numeric_count += len(health)

        if live_numeric_count > self.live_key_warning_threshold:
            prefix_counts = Counter(key.split("/", 1)[0] for key, delivery in deliveries.items() if delivery == "live")
            logger.warning(
                "Live metric key count %d exceeds warning threshold %d; top-level counts=%s",
                live_numeric_count,
                self.live_key_warning_threshold,
                dict(prefix_counts.most_common()),
            )

        route = MetricRoute(
            live_metrics=live_metrics,
            detailed_metrics=detailed_metrics,
            numeric_metrics=numeric_metrics,
            value_dtypes=value_dtypes,
            deliveries=deliveries,
            live_numeric_count=live_numeric_count,
            archived_numeric_count=archived_numeric_count,
        )
        route.assert_lossless(metrics)
        return route


@dataclass(frozen=True)
class EventFileDescriptor:
    metric_key: str
    global_step: int
    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class ShardDescriptor:
    shard_id: str
    path: str
    metadata_path: str
    sha256: str
    session_id: str
    step_start: int
    step_end: int
    event_seq_start: int
    event_seq_end: int
    row_count: int
    state: str
    checkpoint_step: int | None = None
    artifact_ref: str | None = None
    event_files: tuple[EventFileDescriptor, ...] = ()


@dataclass(frozen=True)
class CheckpointMetricCommit:
    logical_run_id: str
    session_id: str
    checkpoint_step: int
    event_seq_start: int | None
    event_seq_end: int | None
    shards: tuple[ShardDescriptor, ...]
    marker_path: str


@dataclass(frozen=True)
class ResumeReconciliation:
    checkpoint_step: int
    recovered_committed_shards: tuple[str, ...]
    orphaned_shards: tuple[str, ...]
    legacy_before_step: int | None


@dataclass(frozen=True)
class ArchivedMetricEvent:
    session_id: str
    event_id: str
    event_seq: int
    event_kind: str
    global_step: int
    metrics: dict[str, float]
    event_files: tuple[EventFileDescriptor, ...] = ()


def inspect_metric_archive(root_dir: str | os.PathLike[str]) -> dict[str, Any] | None:
    """Read an archive manifest without taking the writer lock."""

    path = Path(root_dir) / "manifest.json"
    if not path.exists():
        return None
    return _read_json(path)


class MetricArchiveWriter:
    """Driver-only writer for a complete, immutable numeric metric ledger."""

    def __init__(
        self,
        root_dir: str | os.PathLike[str],
        *,
        logical_run_id: str | None,
        project: str,
        experiment: str,
        session_id: str | None = None,
        resume_checkpoint_step: int | None = None,
        checkpoint_dir: str | os.PathLike[str] | None = None,
        strict_identity: bool = True,
    ):
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.shards_dir = self.root_dir / "shards"
        self.sessions_dir = self.root_dir / "sessions"
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._lock = _ArchiveLock(self.root_dir / ".writer.lock")
        self._lock.acquire()
        self._closed = False
        self._records: list[dict[str, Any]] = []
        self._active_shard_id: str | None = None
        self._next_event_seq = 0
        self._last_checkpoint_step = int(resume_checkpoint_step or 0)
        self.session_id = session_id or uuid.uuid4().hex
        self._manifest_path = self.root_dir / "manifest.json"

        try:
            existing = inspect_metric_archive(self.root_dir)
            if existing is None:
                resolved_run_id = logical_run_id or f"local-{uuid.uuid4().hex}"
                now = _utc_timestamp()
                self._manifest: dict[str, Any] = {
                    "schema_version": METRIC_ARCHIVE_SCHEMA_VERSION,
                    "logical_run_id": resolved_run_id,
                    "project": project,
                    "experiment": experiment,
                    "created_at": now,
                    "updated_at": now,
                    "legacy_before_step": (
                        int(resume_checkpoint_step)
                        if resume_checkpoint_step is not None and int(resume_checkpoint_step) > 0
                        else None
                    ),
                    "sessions": {},
                    "shards": [],
                }
            else:
                self._manifest = existing
                self._validate_manifest_version()
                self._validate_identity(
                    logical_run_id=logical_run_id,
                    project=project,
                    experiment=experiment,
                    strict=strict_identity,
                )
            self._quarantine_temporary_files()
            self.logical_run_id = str(self._manifest["logical_run_id"])
            if self.session_id in self._manifest["sessions"]:
                raise RuntimeError(f"Metric archive session_id already exists: {self.session_id}.")
            if resume_checkpoint_step is not None:
                self.reconcile_resume(int(resume_checkpoint_step), checkpoint_dir=checkpoint_dir)
            elif existing is not None:
                raise RuntimeError(
                    "Metric archive already exists but no resume_checkpoint_step was supplied. "
                    "Use a new archive directory for a fresh run."
                )
            self._start_session()
        except BaseException:
            self._lock.release()
            raise

    @property
    def manifest(self) -> dict[str, Any]:
        """Return a detached manifest snapshot."""

        return json.loads(json.dumps(self._manifest))

    def append_event(
        self,
        route: MetricRoute,
        *,
        global_step: int,
        event_kind: str = "train",
        checkpoint_step: int | None = None,
    ) -> str:
        """Append one complete numeric metric event to the active shard buffer."""

        self._ensure_open()
        if global_step < 0:
            raise ValueError(f"global_step must be non-negative, got {global_step}.")
        if not event_kind:
            raise ValueError("event_kind must be non-empty.")
        event_seq = self._next_event_seq
        self._next_event_seq += 1
        event_id = hashlib.sha256(f"{self.logical_run_id}\0{self.session_id}\0{event_seq}".encode()).hexdigest()
        effective_checkpoint_step = self._last_checkpoint_step if checkpoint_step is None else int(checkpoint_step)

        for metric_key, value in route.numeric_metrics.items():
            dimensions = _parse_metric_dimensions(metric_key)
            self._records.append(
                {
                    "schema_version": METRIC_ARCHIVE_SCHEMA_VERSION,
                    "logical_run_id": self.logical_run_id,
                    "session_id": self.session_id,
                    "event_id": event_id,
                    "event_seq": event_seq,
                    "event_kind": event_kind,
                    "global_step": int(global_step),
                    "checkpoint_step": effective_checkpoint_step,
                    "metric_key": metric_key,
                    "metric_group": _metric_group(metric_key),
                    "delivery": route.deliveries[metric_key],
                    "value": float(value),
                    "value_dtype": route.value_dtypes[metric_key],
                    **dimensions,
                }
            )

        shard = self._ensure_active_shard(event_seq=event_seq, global_step=int(global_step))
        shard["event_seq_end"] = event_seq
        shard["step_end"] = int(global_step)
        shard["row_count"] = len(self._records)
        session = self._manifest["sessions"][self.session_id]
        session["event_seq_end"] = event_seq
        session["step_end"] = int(global_step)
        self._write_manifest()
        self._write_session()
        return event_id

    def close_shard(self) -> ShardDescriptor | None:
        """Atomically materialize the active shard as immutable Parquet."""

        self._ensure_open()
        if self._active_shard_id is None:
            return None
        shard = self._find_shard(self._active_shard_id)
        if shard["state"] != ShardState.OPEN:
            raise RuntimeError(f"Active metric shard is not OPEN: {shard}.")
        if not self._records:
            shard["state"] = ShardState.ORPHANED
            shard["orphan_reason"] = "event contained no numeric metrics"
            self._active_shard_id = None
            self._write_manifest()
            return None

        filename = f"metrics-step-{int(shard['step_start']):09d}-{int(shard['step_end']):09d}-{self.session_id}.parquet"
        parquet_path = self.shards_dir / filename
        metadata_path = parquet_path.with_suffix(".metadata.json")
        if parquet_path.exists() or metadata_path.exists():
            raise RuntimeError(f"Refusing to overwrite immutable metric shard {parquet_path}.")

        _write_parquet_atomic(parquet_path, self._records)
        sha256 = _sha256_file(parquet_path)
        closed_at = _utc_timestamp()
        metadata = {
            "schema_version": METRIC_ARCHIVE_SCHEMA_VERSION,
            "logical_run_id": self.logical_run_id,
            "session_id": self.session_id,
            "shard_id": shard["shard_id"],
            "step_start": shard["step_start"],
            "step_end": shard["step_end"],
            "event_seq_start": shard["event_seq_start"],
            "event_seq_end": shard["event_seq_end"],
            "row_count": len(self._records),
            "sha256": sha256,
            "event_files": shard.get("event_files", []),
            "closed_at": closed_at,
        }
        _write_json_atomic(metadata_path, metadata)

        shard.update(
            {
                "path": str(parquet_path),
                "metadata_path": str(metadata_path),
                "sha256": sha256,
                "row_count": len(self._records),
                "state": ShardState.CLOSED,
                "closed_at": closed_at,
            }
        )
        result = _shard_descriptor(shard)
        self._records = []
        self._active_shard_id = None
        self._write_manifest()
        return result

    def add_event_file(
        self,
        path: str | os.PathLike[str],
        *,
        global_step: int,
        metric_key: str,
    ) -> EventFileDescriptor:
        """Copy a media/diagnostic file into the currently open shard.

        Call this after :meth:`append_event` and before :meth:`close_shard`.
        The copied file is immutable and exposed through ``ShardDescriptor`` so
        an Artifact uploader can add it alongside the Parquet and metadata.
        """

        self._ensure_open()
        if self._active_shard_id is None:
            raise RuntimeError("add_event_file requires an active shard; call append_event first.")
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Metric event file does not exist: {source}.")
        if not metric_key:
            raise ValueError("metric_key must be non-empty.")
        shard = self._find_shard(self._active_shard_id)
        if ShardState(shard["state"]) != ShardState.OPEN:
            raise RuntimeError(f"Cannot add an event file to non-OPEN shard {shard['shard_id']}.")
        if not int(shard["step_start"]) <= int(global_step) <= int(shard["step_end"]):
            raise ValueError(
                f"Event file global_step={global_step} falls outside active shard range "
                f"[{shard['step_start']}, {shard['step_end']}]."
            )

        sha256 = _sha256_file(source)
        safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "-", metric_key).strip("-") or "metric"
        suffix = "".join(source.suffixes)
        filename = f"step-{int(global_step):09d}-{safe_key}-{sha256[:12]}{suffix}"
        attachments_dir = self.root_dir / "attachments" / self.session_id
        attachments_dir.mkdir(parents=True, exist_ok=True)
        destination = attachments_dir / filename
        if destination.exists():
            if _sha256_file(destination) != sha256:
                raise RuntimeError(f"Metric event attachment collision at {destination}.")
        else:
            _copy_file_atomic(source, destination)
        item = {
            "metric_key": metric_key,
            "global_step": int(global_step),
            "path": str(destination),
            "sha256": sha256,
            "size_bytes": destination.stat().st_size,
        }
        event_files = shard.setdefault("event_files", [])
        if item not in event_files:
            event_files.append(item)
            self._write_manifest()
        return EventFileDescriptor(**item)

    def add_event_records(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        global_step: int,
        metric_key: str,
    ) -> EventFileDescriptor:
        """Write complete event records as deterministic ``.jsonl.gz`` attachment.

        This is intended for high-volume text diagnostics such as validation
        generations. The attachment participates in the same immutable shard,
        checkpoint commit, Artifact upload, and resume lineage as image files.
        """

        self._ensure_open()
        if self._active_shard_id is None:
            raise RuntimeError("add_event_records requires an active shard; call append_event first.")
        if not metric_key:
            raise ValueError("metric_key must be non-empty.")
        shard = self._find_shard(self._active_shard_id)
        if ShardState(shard["state"]) != ShardState.OPEN:
            raise RuntimeError(f"Cannot add event records to non-OPEN shard {shard['shard_id']}.")
        if not int(shard["step_start"]) <= int(global_step) <= int(shard["step_end"]):
            raise ValueError(
                f"Event records global_step={global_step} falls outside active shard range "
                f"[{shard['step_start']}, {shard['step_end']}]."
            )

        attachments_dir = self.root_dir / "attachments" / self.session_id
        attachments_dir.mkdir(parents=True, exist_ok=True)
        temporary = attachments_dir / f".event-records-{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("xb") as raw_file:
                with gzip.GzipFile(filename="", mode="wb", fileobj=raw_file, compresslevel=3, mtime=0) as gzip_file:
                    for record in records:
                        line = json.dumps(
                            dict(record),
                            ensure_ascii=False,
                            sort_keys=True,
                            default=_json_attachment_default,
                        )
                        gzip_file.write(line.encode("utf-8"))
                        gzip_file.write(b"\n")
                raw_file.flush()
                os.fsync(raw_file.fileno())

            sha256 = _sha256_file(temporary)
            safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "-", metric_key).strip("-") or "metric"
            filename = f"step-{int(global_step):09d}-{safe_key}-{sha256[:12]}.jsonl.gz"
            destination = attachments_dir / filename
            if destination.exists():
                if _sha256_file(destination) != sha256:
                    raise RuntimeError(f"Metric event attachment collision at {destination}.")
            else:
                os.replace(temporary, destination)
                _fsync_directory(attachments_dir)

            item = {
                "metric_key": metric_key,
                "global_step": int(global_step),
                "path": str(destination),
                "sha256": sha256,
                "size_bytes": destination.stat().st_size,
            }
            event_files = shard.setdefault("event_files", [])
            if item not in event_files:
                event_files.append(item)
                self._write_manifest()
            return EventFileDescriptor(**item)
        finally:
            temporary.unlink(missing_ok=True)

    def commit_checkpoint(
        self,
        checkpoint_dir: str | os.PathLike[str],
        *,
        checkpoint_step: int,
    ) -> CheckpointMetricCommit:
        """Close the current shard and atomically write ``metrics_commit.json``.

        The trainer must update its ``latest_checkpointed_iteration.txt`` only
        after this method returns.
        """

        self._ensure_open()
        checkpoint_step = int(checkpoint_step)
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.is_dir():
            raise RuntimeError(f"Checkpoint payload directory must exist before metric commit: {checkpoint_path}.")
        self.close_shard()
        candidates = [
            shard
            for shard in self._manifest["shards"]
            if shard["session_id"] == self.session_id
            and shard["state"] == ShardState.CLOSED
            and int(shard["step_end"]) <= checkpoint_step
        ]
        descriptors = tuple(_shard_descriptor(shard) for shard in candidates)
        event_starts = [item.event_seq_start for item in descriptors]
        event_ends = [item.event_seq_end for item in descriptors]
        marker_path = checkpoint_path / METRIC_COMMIT_FILENAME
        marker = {
            "schema_version": METRIC_ARCHIVE_SCHEMA_VERSION,
            "logical_run_id": self.logical_run_id,
            "session_id": self.session_id,
            "checkpoint_step": checkpoint_step,
            "event_seq_start": min(event_starts) if event_starts else None,
            "event_seq_end": max(event_ends) if event_ends else None,
            "shards": [asdict(item) for item in descriptors],
            "committed_at": _utc_timestamp(),
        }
        _write_json_atomic(marker_path, marker)

        for shard in candidates:
            shard["state"] = ShardState.CHECKPOINT_COMMITTED
            shard["checkpoint_step"] = checkpoint_step
        self._last_checkpoint_step = checkpoint_step
        session = self._manifest["sessions"][self.session_id]
        session["last_checkpoint_step"] = checkpoint_step
        self._write_manifest()
        self._write_session()
        return CheckpointMetricCommit(
            logical_run_id=self.logical_run_id,
            session_id=self.session_id,
            checkpoint_step=checkpoint_step,
            event_seq_start=marker["event_seq_start"],
            event_seq_end=marker["event_seq_end"],
            shards=tuple(_shard_descriptor(shard) for shard in candidates),
            marker_path=str(marker_path),
        )

    def reconcile_resume(
        self,
        checkpoint_step: int,
        *,
        checkpoint_dir: str | os.PathLike[str] | None = None,
    ) -> ResumeReconciliation:
        """Reconcile a prior archive with the selected committed checkpoint."""

        checkpoint_step = int(checkpoint_step)
        marker_shards: dict[str, dict[str, Any]] = {}
        if checkpoint_dir is not None:
            marker_path = Path(checkpoint_dir) / METRIC_COMMIT_FILENAME
            if marker_path.exists():
                marker = _read_json(marker_path)
                self._validate_commit_marker(marker, checkpoint_step)
                marker_shards = {item["shard_id"]: item for item in marker.get("shards", [])}

        recovered: list[str] = []
        orphaned: list[str] = []
        for shard in self._manifest["shards"]:
            state = ShardState(shard["state"])
            shard_id = shard["shard_id"]
            if state == ShardState.STAGED_OFFLINE:
                delivery_session_id = shard.get("wandb_delivery_session_id", shard["session_id"])
                delivery_session = self._manifest["sessions"].get(delivery_session_id, {})
                sync_file = _resolve_session_sync_file(delivery_session)
                if sync_file is not None and _wandb_sync_completed(sync_file):
                    shard["state"] = ShardState.UPLOADED
                    shard["state_updated_at"] = _utc_timestamp()
                    state = ShardState.UPLOADED
                staged_segment_missing = sync_file is None or (
                    not sync_file.exists() and not _wandb_sync_completed(sync_file)
                )
                if staged_segment_missing:
                    shard["previous_state"] = state.value
                    shard["state"] = ShardState.FAILED_RETRYABLE
                    shard["last_error"] = "offline W&B sync segment is missing; Artifact must be staged again"
                    shard["failure_count"] = int(shard.get("failure_count", 0)) + 1
                    state = ShardState.FAILED_RETRYABLE
            marker_item = marker_shards.get(shard_id)
            if marker_item is not None:
                if state == ShardState.ORPHANED:
                    raise RuntimeError(
                        f"Checkpoint {checkpoint_step} references metric shard {shard_id!r}, "
                        "but that shard was already marked ORPHANED by an earlier resume decision."
                    )
                self._validate_marker_shard(shard, marker_item)
                if state in {ShardState.OPEN, ShardState.CLOSED}:
                    shard["state"] = ShardState.CHECKPOINT_COMMITTED
                    shard["checkpoint_step"] = checkpoint_step
                    recovered.append(shard_id)
                continue

            shard_checkpoint = shard.get("checkpoint_step")
            beyond_checkpoint = int(shard.get("step_end", checkpoint_step + 1)) > checkpoint_step
            committed_beyond_checkpoint = shard_checkpoint is not None and int(shard_checkpoint) > checkpoint_step
            uncommitted = state in {ShardState.OPEN, ShardState.CLOSED}
            if beyond_checkpoint or committed_beyond_checkpoint or uncommitted:
                shard["previous_state"] = state.value
                shard["state"] = ShardState.ORPHANED
                shard["orphaned_at_checkpoint_step"] = checkpoint_step
                shard["orphaned_at"] = _utc_timestamp()
                orphaned.append(shard_id)

        for session_id, session in self._manifest["sessions"].items():
            if session_id != self.session_id and session.get("status") == "ACTIVE":
                session["status"] = "INTERRUPTED"
                session["ended_at"] = _utc_timestamp()
        self._write_manifest()
        return ResumeReconciliation(
            checkpoint_step=checkpoint_step,
            recovered_committed_shards=tuple(recovered),
            orphaned_shards=tuple(orphaned),
            legacy_before_step=self._manifest.get("legacy_before_step"),
        )

    def pending_upload_shards(self) -> tuple[ShardDescriptor, ...]:
        """Return immutable shards eligible for initial upload or retry."""

        eligible = {
            ShardState.CLOSED,
            ShardState.CHECKPOINT_COMMITTED,
            ShardState.SUBMITTED,
            ShardState.FAILED_RETRYABLE,
        }
        return tuple(
            _shard_descriptor(shard)
            for shard in self._manifest["shards"]
            if ShardState(shard["state"]) in eligible and shard.get("path")
        )

    def unrecorded_offline_committed_live_events(
        self,
        *,
        checkpoint_step: int,
    ) -> tuple[ArchivedMetricEvent, ...]:
        """Recover committed live events missing from prior offline W&B segments."""
        self._ensure_open()
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Metric archive recovery requires pyarrow.") from exc

        checkpoint_step = int(checkpoint_step)
        active_states = {
            ShardState.CHECKPOINT_COMMITTED,
            ShardState.STAGED_OFFLINE,
            ShardState.SUBMITTED,
            ShardState.UPLOADED,
            ShardState.FAILED_RETRYABLE,
        }
        sessions = self._manifest["sessions"]
        recovered_event_ids = {
            str(event_id)
            for session in sessions.values()
            if _session_sync_file_available(session)
            for event_id in session.get("wandb_recovered_event_ids", ())
        }
        session_order = {session["session_id"]: index for index, session in enumerate(_ordered_sessions(sessions))}
        grouped: dict[tuple[str, str, int, str, int], dict[str, float]] = {}
        attachments: dict[tuple[str, int], list[EventFileDescriptor]] = {}
        shards = sorted(
            self._manifest["shards"],
            key=lambda item: (
                session_order.get(item["session_id"], 0),
                int(item["event_seq_start"]),
            ),
        )
        for shard in shards:
            state = ShardState(shard["state"])
            session = sessions.get(shard["session_id"], {})
            session_sync_available = _session_sync_file_available(session)
            if (
                state not in active_states
                or session.get("wandb_mode") != "offline"
                or int(shard.get("checkpoint_step") or -1) > checkpoint_step
            ):
                continue
            path = Path(shard["path"])
            if not path.is_file() or _sha256_file(path) != shard["sha256"]:
                raise RuntimeError(f"Cannot recover W&B history from invalid metric shard: {path}.")
            recorded_ranges = tuple(session.get("wandb_committed_ranges", ())) if session_sync_available else ()
            table = pq.read_table(
                path,
                columns=[
                    "session_id",
                    "event_id",
                    "event_seq",
                    "event_kind",
                    "global_step",
                    "metric_key",
                    "delivery",
                    "value",
                ],
            )
            for row in table.to_pylist():
                step = int(row["global_step"])
                if (
                    step > checkpoint_step
                    or row["delivery"] != "live"
                    or str(row["event_id"]) in recovered_event_ids
                    or any(int(item["step_start"]) <= step <= int(item["step_end"]) for item in recorded_ranges)
                ):
                    continue
                key = (
                    str(row["session_id"]),
                    str(row["event_id"]),
                    int(row["event_seq"]),
                    str(row["event_kind"]),
                    step,
                )
                grouped.setdefault(key, {})[str(row["metric_key"])] = float(row["value"])
            for item in shard.get("event_files", []):
                step = int(item["global_step"])
                if step <= checkpoint_step and not any(
                    int(recorded["step_start"]) <= step <= int(recorded["step_end"]) for recorded in recorded_ranges
                ):
                    attachments.setdefault((shard["session_id"], step), []).append(EventFileDescriptor(**item))

        events = [
            ArchivedMetricEvent(
                session_id=session_id,
                event_id=event_id,
                event_seq=event_seq,
                event_kind=event_kind,
                global_step=global_step,
                metrics=metrics,
                event_files=tuple(attachments.get((session_id, global_step), ())),
            )
            for (session_id, event_id, event_seq, event_kind, global_step), metrics in grouped.items()
        ]
        return tuple(
            sorted(
                events,
                key=lambda item: (
                    session_order.get(item.session_id, 0),
                    item.event_seq,
                ),
            )
        )

    def mark_shard_state(
        self,
        shard_id: str,
        state: ShardState | str,
        *,
        error: str | None = None,
        artifact_ref: str | None = None,
        delivery_session_id: str | None = None,
    ) -> ShardDescriptor:
        """Atomically persist an uploader/offline staging state transition."""

        self._ensure_open()
        target = ShardState(state)
        shard = self._find_shard(shard_id)
        current = ShardState(shard["state"])
        if target not in _ALLOWED_TRANSITIONS[current]:
            raise ValueError(f"Invalid metric shard state transition {current} -> {target} for {shard_id}.")
        if delivery_session_id is not None and delivery_session_id not in self._manifest["sessions"]:
            raise ValueError(f"Unknown W&B delivery session ID {delivery_session_id!r}.")
        shard["state"] = target
        shard["state_updated_at"] = _utc_timestamp()
        if error is not None:
            shard["last_error"] = str(error)
            shard["failure_count"] = int(shard.get("failure_count", 0)) + 1
        elif target != ShardState.FAILED_RETRYABLE:
            shard.pop("last_error", None)
        if artifact_ref is not None:
            shard["artifact_ref"] = artifact_ref
        if delivery_session_id is not None:
            shard["wandb_delivery_session_id"] = delivery_session_id
        self._write_manifest()
        return _shard_descriptor(shard)

    def register_wandb_session(
        self,
        *,
        mode: str,
        run_id: str | None,
        sync_dir: str | os.PathLike[str] | None = None,
        sync_file: str | os.PathLike[str] | None = None,
    ) -> None:
        """Record the W&B segment associated with the current archive session."""

        self._ensure_open()
        normalized_mode = str(mode).lower()
        if normalized_mode not in {"online", "offline", "disabled"}:
            raise ValueError(f"Unsupported W&B mode {mode!r}.")
        if run_id is not None and str(run_id) != self.logical_run_id:
            raise RuntimeError(f"W&B run ID {run_id!r} does not match archive logical_run_id {self.logical_run_id!r}.")
        session = self._manifest["sessions"][self.session_id]
        session["wandb_mode"] = normalized_mode
        session["wandb_run_id"] = run_id
        session["wandb_sync_dir"] = str(Path(sync_dir).resolve()) if sync_dir is not None else None
        session["wandb_sync_file"] = str(Path(sync_file).resolve()) if sync_file is not None else None
        self._write_manifest()
        self._write_session()

    def record_wandb_committed_range(
        self,
        *,
        step_start: int,
        step_end: int,
        recovered_event_ids: tuple[str, ...] = (),
    ) -> None:
        """Record the range intentionally written to this W&B session history."""
        self._ensure_open()
        step_start = int(step_start)
        step_end = int(step_end)
        if step_start < 0 or step_end < step_start:
            raise ValueError(f"Invalid committed W&B range [{step_start}, {step_end}].")
        session = self._manifest["sessions"][self.session_id]
        ranges = session.setdefault("wandb_committed_ranges", [])
        if ranges and int(ranges[-1]["step_end"]) + 1 >= step_start:
            ranges[-1]["step_end"] = max(int(ranges[-1]["step_end"]), step_end)
        else:
            ranges.append({"step_start": step_start, "step_end": step_end})
        session["wandb_committed_step_start"] = min(
            step_start,
            int(session.get("wandb_committed_step_start", step_start)),
        )
        session["wandb_committed_step_end"] = max(
            step_end,
            int(session.get("wandb_committed_step_end", step_end)),
        )
        if recovered_event_ids:
            recorded_event_ids = session.setdefault("wandb_recovered_event_ids", [])
            recorded_event_id_set = set(recorded_event_ids)
            for event_id in recovered_event_ids:
                normalized_event_id = str(event_id)
                if normalized_event_id not in recorded_event_id_set:
                    recorded_event_ids.append(normalized_event_id)
                    recorded_event_id_set.add(normalized_event_id)
        self._write_manifest()
        self._write_session()

    def mark_wandb_history_recovered(self, source_session_ids: tuple[str, ...]) -> None:
        """Mark missing offline history as fully replayed into this session."""

        self._ensure_open()
        changed = False
        for source_session_id in source_session_ids:
            if source_session_id == self.session_id:
                continue
            source_session = self._manifest["sessions"].get(source_session_id)
            if source_session is None:
                raise ValueError(f"Unknown recovered W&B source session ID {source_session_id!r}.")
            source_session["wandb_history_recovered_by_session"] = self.session_id
            changed = True
        if changed:
            self._write_manifest()
            for source_session_id in source_session_ids:
                if source_session_id != self.session_id:
                    _write_json_atomic(
                        self.sessions_dir / f"{source_session_id}.json",
                        self._manifest["sessions"][source_session_id],
                    )

    def unsynced_offline_sessions(self) -> tuple[dict[str, Any], ...]:
        """Return offline W&B segments which do not have a ``.synced`` marker."""

        result = []
        sessions = _ordered_sessions(self._manifest["sessions"])
        for session in sessions:
            if session.get("wandb_mode") != "offline":
                continue
            if not _offline_session_has_sync_payload(self._manifest, session):
                continue
            sync_file = _resolve_session_sync_file(session)
            if sync_file is None or not _wandb_sync_completed(sync_file):
                item = dict(session)
                item["resolved_sync_file"] = str(sync_file) if sync_file is not None else None
                result.append(item)
        return tuple(result)

    def validate_wandb_transition(self, requested_mode: str) -> None:
        """Reject strict offline-to-online resume while segments remain unsynced."""

        if str(requested_mode).lower() != "online":
            return
        unsynced = self.unsynced_offline_sessions()
        if unsynced:
            commands = self.standard_wandb_sync_commands()
            raise RuntimeError(
                "Cannot resume metric archive online while offline W&B sessions remain unsynced: "
                f"sessions={[item['session_id'] for item in unsynced]}. "
                f"Sync commands: {commands}"
            )

    def standard_wandb_sync_commands(
        self,
        *,
        entity: str | None = None,
        project: str | None = None,
    ) -> tuple[str, ...]:
        """Build ordered standard ``wandb sync`` commands for offline segments."""

        entity = entity or None
        project = project or str(self._manifest.get("project") or "")
        sessions = _ordered_sessions(self._manifest["sessions"])
        commands: list[str] = []
        has_prior_remote_segment = any(item.get("wandb_mode") == "online" for item in sessions)
        for session in sessions:
            if session.get("wandb_mode") != "offline" or not _offline_session_has_sync_payload(self._manifest, session):
                continue
            sync_file = _resolve_session_sync_file(session)
            if sync_file is not None and _wandb_sync_completed(sync_file):
                has_prior_remote_segment = True
                continue
            if sync_file is None:
                has_prior_remote_segment = True
                continue
            parts = ["wandb", "sync"]
            if has_prior_remote_segment:
                parts.append("--append")
            if entity:
                parts.extend(["--entity", entity])
            if project:
                parts.extend(["--project", project])
            parts.extend(["--id", self.logical_run_id, str(sync_file)])
            commands.append(" ".join(shlex.quote(part) for part in parts))
            has_prior_remote_segment = True
        return tuple(commands)

    def health_metrics(self) -> dict[str, float]:
        """Return fixed-cardinality metrics describing archive health."""

        shards = self._manifest["shards"]
        committed_steps = [
            int(shard["checkpoint_step"])
            for shard in shards
            if shard.get("checkpoint_step") is not None
            and ShardState(shard["state"])
            in {
                ShardState.CHECKPOINT_COMMITTED,
                ShardState.STAGED_OFFLINE,
                ShardState.SUBMITTED,
                ShardState.UPLOADED,
                ShardState.FAILED_RETRYABLE,
            }
        ]
        uploaded_steps = [
            int(shard["step_end"]) for shard in shards if ShardState(shard["state"]) == ShardState.UPLOADED
        ]
        return {
            f"{_HEALTH_PREFIX}/pending_shard_count": float(len(self.pending_upload_shards())),
            f"{_HEALTH_PREFIX}/orphaned_shard_count": float(
                sum(ShardState(shard["state"]) == ShardState.ORPHANED for shard in shards)
            ),
            f"{_HEALTH_PREFIX}/failed_upload_count": float(sum(int(shard.get("failure_count", 0)) for shard in shards)),
            f"{_HEALTH_PREFIX}/last_committed_step": float(max(committed_steps, default=-1)),
            f"{_HEALTH_PREFIX}/last_uploaded_step": float(max(uploaded_steps, default=-1)),
        }

    def close(self, *, interrupted: bool = False) -> None:
        """Close any active shard, finish the session and release the lock."""

        if self._closed:
            return
        try:
            if interrupted and self._active_shard_id is not None:
                shard = self._find_shard(self._active_shard_id)
                shard["previous_state"] = ShardState(shard["state"]).value
                shard["state"] = ShardState.ORPHANED
                shard["orphan_reason"] = "trainer session interrupted before shard commit"
                shard["orphaned_at"] = _utc_timestamp()
                self._records = []
                self._active_shard_id = None
                self._write_manifest()
            else:
                self.close_shard()
            session = self._manifest["sessions"][self.session_id]
            session["status"] = "INTERRUPTED" if interrupted else "CLOSED"
            session["ended_at"] = _utc_timestamp()
            self._write_manifest()
            self._write_session()
        finally:
            self._closed = True
            self._lock.release()

    def __enter__(self) -> MetricArchiveWriter:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close(interrupted=exc_type is not None)

    def _quarantine_temporary_files(self) -> None:
        temporary_files = sorted(self.root_dir.rglob("*.tmp"))
        if not temporary_files:
            return
        orphan_dir = self.root_dir / "orphans"
        orphan_dir.mkdir(parents=True, exist_ok=True)
        entries = self._manifest.setdefault("orphaned_temp_files", [])
        for temporary_path in temporary_files:
            if orphan_dir in temporary_path.parents:
                continue
            destination = orphan_dir / f"{uuid.uuid4().hex}-{temporary_path.name}"
            os.replace(temporary_path, destination)
            entries.append(
                {
                    "original_path": str(temporary_path),
                    "path": str(destination),
                    "size_bytes": destination.stat().st_size,
                    "orphaned_at": _utc_timestamp(),
                }
            )
        _fsync_directory(orphan_dir)
        self._write_manifest()

    def _start_session(self) -> None:
        now = _utc_timestamp()
        prior_orders = [
            int(session.get("session_order", index))
            for index, session in enumerate(self._manifest["sessions"].values())
        ]
        self._manifest["sessions"][self.session_id] = {
            "session_id": self.session_id,
            "session_order": max(prior_orders, default=-1) + 1,
            "status": "ACTIVE",
            "started_at": now,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "event_seq_start": None,
            "event_seq_end": None,
            "step_start": None,
            "step_end": None,
            "last_checkpoint_step": self._last_checkpoint_step,
        }
        self._write_manifest()
        self._write_session()

    def _ensure_active_shard(self, *, event_seq: int, global_step: int) -> dict[str, Any]:
        if self._active_shard_id is None:
            self._active_shard_id = f"{self.session_id}:{event_seq}"
            shard = {
                "shard_id": self._active_shard_id,
                "session_id": self.session_id,
                "state": ShardState.OPEN,
                "path": None,
                "metadata_path": None,
                "sha256": None,
                "step_start": global_step,
                "step_end": global_step,
                "event_seq_start": event_seq,
                "event_seq_end": event_seq,
                "row_count": 0,
                "created_at": _utc_timestamp(),
                "checkpoint_step": None,
            }
            self._manifest["shards"].append(shard)
            session = self._manifest["sessions"][self.session_id]
            if session["event_seq_start"] is None:
                session["event_seq_start"] = event_seq
                session["step_start"] = global_step
            return shard
        return self._find_shard(self._active_shard_id)

    def _find_shard(self, shard_id: str) -> dict[str, Any]:
        for shard in self._manifest["shards"]:
            if shard["shard_id"] == shard_id:
                return shard
        raise KeyError(f"Unknown metric shard ID {shard_id!r}.")

    def _validate_manifest_version(self) -> None:
        version = int(self._manifest.get("schema_version", -1))
        if version != METRIC_ARCHIVE_SCHEMA_VERSION:
            raise RuntimeError(
                f"Unsupported metric archive schema_version={version}; expected {METRIC_ARCHIVE_SCHEMA_VERSION}."
            )

    def _validate_identity(
        self,
        *,
        logical_run_id: str | None,
        project: str,
        experiment: str,
        strict: bool,
    ) -> None:
        expected = {
            "logical_run_id": logical_run_id,
            "project": project,
            "experiment": experiment,
        }
        mismatches = {
            key: {"manifest": self._manifest.get(key), "requested": value}
            for key, value in expected.items()
            if value is not None and str(self._manifest.get(key)) != str(value)
        }
        if mismatches and strict:
            raise RuntimeError(f"Metric archive identity mismatch: {mismatches}.")

    def _validate_commit_marker(self, marker: Mapping[str, Any], checkpoint_step: int) -> None:
        if int(marker.get("schema_version", -1)) != METRIC_ARCHIVE_SCHEMA_VERSION:
            raise RuntimeError(f"Metric commit marker has incompatible schema: {marker}.")
        if marker.get("logical_run_id") != self.logical_run_id:
            raise RuntimeError(
                "Metric commit marker logical_run_id mismatch: "
                f"marker={marker.get('logical_run_id')!r}, archive={self.logical_run_id!r}."
            )
        if int(marker.get("checkpoint_step", -1)) != checkpoint_step:
            raise RuntimeError(
                "Metric commit marker checkpoint mismatch: "
                f"marker={marker.get('checkpoint_step')}, selected={checkpoint_step}."
            )

    @staticmethod
    def _validate_marker_shard(shard: Mapping[str, Any], marker_item: Mapping[str, Any]) -> None:
        for key in ("sha256", "step_start", "step_end", "event_seq_start", "event_seq_end"):
            if shard.get(key) != marker_item.get(key):
                raise RuntimeError(
                    f"Metric commit marker shard mismatch for {shard.get('shard_id')}: "
                    f"{key}: manifest={shard.get(key)!r}, marker={marker_item.get(key)!r}."
                )
        path = shard.get("path")
        if not path or not Path(path).is_file():
            raise RuntimeError(f"Committed metric shard file is missing: {path!r}.")
        actual_hash = _sha256_file(Path(path))
        if actual_hash != shard.get("sha256"):
            raise RuntimeError(
                f"Committed metric shard checksum mismatch for {path}: "
                f"manifest={shard.get('sha256')}, actual={actual_hash}."
            )

    def _write_manifest(self) -> None:
        self._manifest["updated_at"] = _utc_timestamp()
        _write_json_atomic(self._manifest_path, self._manifest)

    def _write_session(self) -> None:
        _write_json_atomic(self.sessions_dir / f"{self.session_id}.json", self._manifest["sessions"][self.session_id])

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("MetricArchiveWriter is closed.")


_ALLOWED_TRANSITIONS: dict[ShardState, frozenset[ShardState]] = {
    ShardState.OPEN: frozenset({ShardState.CLOSED, ShardState.ORPHANED}),
    ShardState.CLOSED: frozenset(
        {
            ShardState.CHECKPOINT_COMMITTED,
            ShardState.STAGED_OFFLINE,
            ShardState.SUBMITTED,
            ShardState.FAILED_RETRYABLE,
            ShardState.ORPHANED,
        }
    ),
    ShardState.CHECKPOINT_COMMITTED: frozenset(
        {
            ShardState.STAGED_OFFLINE,
            ShardState.SUBMITTED,
            ShardState.UPLOADED,
            ShardState.FAILED_RETRYABLE,
            ShardState.ORPHANED,
        }
    ),
    ShardState.STAGED_OFFLINE: frozenset(
        {ShardState.SUBMITTED, ShardState.UPLOADED, ShardState.FAILED_RETRYABLE, ShardState.ORPHANED}
    ),
    ShardState.SUBMITTED: frozenset(
        {
            ShardState.STAGED_OFFLINE,
            ShardState.UPLOADED,
            ShardState.FAILED_RETRYABLE,
            ShardState.ORPHANED,
        }
    ),
    ShardState.UPLOADED: frozenset({ShardState.ORPHANED}),
    ShardState.FAILED_RETRYABLE: frozenset(
        {
            ShardState.STAGED_OFFLINE,
            ShardState.SUBMITTED,
            ShardState.UPLOADED,
            ShardState.ORPHANED,
        }
    ),
    ShardState.ORPHANED: frozenset(),
}


class _ArchiveLock:
    def __init__(self, path: Path):
        self.path = path
        self._file: Any = None

    def acquire(self) -> None:
        import fcntl

        self._file = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._file.seek(0)
            owner = self._file.read().strip()
            self._file.close()
            self._file = None
            raise RuntimeError(
                f"Metric archive is already locked by another trainer driver: {self.path}; owner={owner!r}."
            ) from exc
        self._file.seek(0)
        self._file.truncate()
        json.dump({"pid": os.getpid(), "host": socket.gethostname(), "acquired_at": _utc_timestamp()}, self._file)
        self._file.flush()
        os.fsync(self._file.fileno())

    def release(self) -> None:
        if self._file is None:
            return
        import fcntl

        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        finally:
            self._file.close()
            self._file = None


def _config_get(config: Any, key: str, default: Any) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _json_attachment_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    return str(value)


def _coerce_numeric(value: Any) -> tuple[float, str] | None:
    if isinstance(value, np.generic):
        if not np.issubdtype(value.dtype, np.number) and not np.issubdtype(value.dtype, np.bool_):
            return None
        if np.issubdtype(value.dtype, np.complexfloating):
            return None
        return float(value.item()), str(value.dtype)
    if isinstance(value, bool):
        return float(value), "bool"
    if isinstance(value, numbers.Integral):
        return float(value), type(value).__name__
    if isinstance(value, numbers.Real):
        return float(value), type(value).__name__
    # Avoid importing torch in this low-level module.  Scalar tensors and
    # scalar-like values expose ``numel``/``item``/``dtype``.
    if callable(getattr(value, "numel", None)) and callable(getattr(value, "item", None)):
        try:
            if int(value.numel()) == 1:
                item = value.detach().cpu().item() if callable(getattr(value, "detach", None)) else value.item()
                if isinstance(item, numbers.Real):
                    return float(item), str(getattr(value, "dtype", type(item).__name__))
        except (RuntimeError, TypeError, ValueError):
            return None
    return None


def _should_archive_metric(metric_key: str, configured_topk: int | None) -> bool:
    if metric_key.startswith("opd_intervention/"):
        return metric_key not in _LIVE_INTERVENTION_KEYS
    if metric_key.startswith(_REPAIR_EVAL_AUX_PREFIXES):
        return True
    if metric_key.startswith(_REPAIR_EVAL_CORE_PREFIXES):
        return _REPAIR_EVAL_LIVE_CORE_RE.search(metric_key) is None
    if metric_key.startswith(_ALWAYS_ARCHIVE_PREFIXES):
        return True
    if metric_key.startswith(_DYNAMIC_REASON_PREFIXES):
        return True
    if metric_key in _KNOWN_ALIAS_KEYS or _STATIC_DIAGNOSTIC_RE.fullmatch(metric_key):
        return True
    if _is_redundant_summary_metric(metric_key):
        return True
    if _POSITION_BIN_RE.search(metric_key) or _BUCKET_BIN_RE.search(metric_key) or _COARSE_RE.search(metric_key):
        return True
    if metric_key.startswith(_EVAL_AUX_PREFIXES) and _BOOTSTRAP_STAT_RE.search(metric_key):
        return True
    if metric_key.startswith(("distillation/", "opd/")) and _LOW_K_SUFFIX_RE.search(metric_key):
        return True
    overlap_match = _OPD_OVERLAP_TOPK_RE.match(metric_key)
    return bool(
        overlap_match and configured_topk is not None and int(overlap_match.group("topk")) != int(configured_topk)
    )


def _is_redundant_summary_metric(metric_key: str) -> bool:
    """Identify fixed-cardinality details that duplicate a canonical live summary."""

    if metric_key.startswith(_REDUNDANT_SUMMARY_PREFIXES) or metric_key in _REDUNDANT_SUMMARY_KEYS:
        return True
    for prefix, archived_fields in _REDUNDANT_AGGREGATE_FIELDS.items():
        if metric_key.startswith(prefix) and metric_key.removeprefix(prefix) in archived_fields:
            return True
    if metric_key.startswith(("contagion/", "loss_mask/", "response_end/")) and metric_key.endswith("_count"):
        return True
    if metric_key.startswith("critic/") and metric_key.endswith(("/max", "/min")):
        return True
    if metric_key.startswith("rollout_corr/") and metric_key not in _LIVE_ROLLOUT_CORRECTION_KEYS:
        return True
    return metric_key.startswith(("opd/", "distillation/")) and metric_key.endswith("/tail_over_head")


def _infer_configured_topk(metrics: Mapping[str, Any]) -> int | None:
    values = [int(match.group("topk")) for key in metrics if (match := _OPD_OVERLAP_TOPK_RE.match(key)) is not None]
    return max(values, default=None)


def _metric_group(metric_key: str) -> str:
    return metric_key.split("/", 1)[0]


def _parse_metric_dimensions(metric_key: str) -> dict[str, Any]:
    bin_match = _POSITION_BIN_RE.search(metric_key) or _BUCKET_BIN_RE.search(metric_key)
    coarse_match = _COARSE_RE.search(metric_key)
    topk_matches = list(_TOPK_RE.finditer(metric_key))
    eval_scope = None
    for prefix in (*_REPAIR_EVAL_SCOPES, "teacher-val", "test", "val"):
        if metric_key.startswith(f"{prefix}-"):
            eval_scope = prefix
            break
    return {
        "bin_start": int(bin_match.group("start")) if bin_match else None,
        "bin_end": int(bin_match.group("end")) if bin_match else None,
        "coarse_region": coarse_match.group("region") if coarse_match else None,
        "topk": int(topk_matches[-1].group("topk")) if topk_matches else None,
        "eval_scope": eval_scope,
    }


def _write_parquet_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Metric archive requires pyarrow; install the verl runtime requirements.") from exc

    schema = pa.schema(
        [
            pa.field("schema_version", pa.int16(), nullable=False),
            pa.field("logical_run_id", pa.string(), nullable=False),
            pa.field("session_id", pa.string(), nullable=False),
            pa.field("event_id", pa.string(), nullable=False),
            pa.field("event_seq", pa.int64(), nullable=False),
            pa.field("event_kind", pa.string(), nullable=False),
            pa.field("global_step", pa.int64(), nullable=False),
            pa.field("checkpoint_step", pa.int64(), nullable=False),
            pa.field("metric_key", pa.string(), nullable=False),
            pa.field("metric_group", pa.string(), nullable=False),
            pa.field("delivery", pa.string(), nullable=False),
            pa.field("value", pa.float64(), nullable=False),
            pa.field("value_dtype", pa.string(), nullable=False),
            pa.field("bin_start", pa.int64()),
            pa.field("bin_end", pa.int64()),
            pa.field("coarse_region", pa.string()),
            pa.field("topk", pa.int64()),
            pa.field("eval_scope", pa.string()),
        ]
    )
    table = pa.Table.from_pylist(records, schema=schema)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        pq.write_table(
            table,
            temporary,
            compression="zstd",
            use_dictionary=[
                "logical_run_id",
                "session_id",
                "event_kind",
                "metric_key",
                "metric_group",
                "delivery",
                "value_dtype",
                "coarse_region",
                "eval_scope",
            ],
        )
        _fsync_file(temporary)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, sort_keys=True, ensure_ascii=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path}, got {type(value).__name__}.")
    return value


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _shard_descriptor(shard: Mapping[str, Any]) -> ShardDescriptor:
    state = shard["state"]
    event_files = tuple(EventFileDescriptor(**item) for item in shard.get("event_files", []))
    return ShardDescriptor(
        shard_id=str(shard["shard_id"]),
        path=str(shard.get("path") or ""),
        metadata_path=str(shard.get("metadata_path") or ""),
        sha256=str(shard.get("sha256") or ""),
        session_id=str(shard["session_id"]),
        step_start=int(shard["step_start"]),
        step_end=int(shard["step_end"]),
        event_seq_start=int(shard["event_seq_start"]),
        event_seq_end=int(shard["event_seq_end"]),
        row_count=int(shard["row_count"]),
        state=state.value if isinstance(state, ShardState) else str(state),
        checkpoint_step=(int(shard["checkpoint_step"]) if shard.get("checkpoint_step") is not None else None),
        artifact_ref=shard.get("artifact_ref"),
        event_files=event_files,
    )


def _ordered_sessions(sessions: Mapping[str, Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(
        sessions.values(),
        key=lambda item: (
            int(item.get("session_order", 0)),
            str(item.get("started_at", "")),
            str(item.get("session_id", "")),
        ),
    )


def _resolve_session_sync_file(session: Mapping[str, Any]) -> Path | None:
    explicit = session.get("wandb_sync_file")
    if explicit:
        return Path(explicit)
    sync_dir = session.get("wandb_sync_dir")
    if not sync_dir:
        return None
    candidates = sorted(Path(sync_dir).glob("*.wandb"))
    candidates.extend(sorted(Path(sync_dir).glob("*.wandb.synced")))
    return candidates[-1] if candidates else None


def _wandb_sync_completed(path: Path) -> bool:
    if path.name.endswith(".synced"):
        return path.exists()
    return Path(f"{path}.synced").exists()


def _session_sync_file_available(session: Mapping[str, Any]) -> bool:
    sync_file = _resolve_session_sync_file(session)
    return bool(sync_file is not None and (sync_file.exists() or _wandb_sync_completed(sync_file)))


def _offline_session_has_sync_payload(
    manifest: Mapping[str, Any],
    session: Mapping[str, Any],
) -> bool:
    has_history = bool(session.get("wandb_committed_ranges"))
    recovered_by_session_id = session.get("wandb_history_recovered_by_session")
    if has_history and recovered_by_session_id:
        recovered_by_session = manifest.get("sessions", {}).get(recovered_by_session_id, {})
        if _session_sync_file_available(recovered_by_session):
            has_history = False
    session_id = session.get("session_id")
    has_artifact = any(
        shard.get("state") == ShardState.STAGED_OFFLINE
        and shard.get("wandb_delivery_session_id", shard.get("session_id")) == session_id
        for shard in manifest.get("shards", ())
    )
    return has_history or has_artifact


def _copy_file_atomic(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with source.open("rb") as source_file, temporary.open("xb") as destination_file:
            shutil.copyfileobj(source_file, destination_file, length=1024 * 1024)
            destination_file.flush()
            os.fsync(destination_file.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
