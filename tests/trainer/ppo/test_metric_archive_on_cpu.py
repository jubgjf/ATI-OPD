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

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from verl.trainer.ppo.metric_archive import (
    METRIC_ARCHIVE_SCHEMA_VERSION,
    METRIC_COMMIT_FILENAME,
    MetricArchiveWriter,
    MetricLoggingSettings,
    MetricRouter,
    ShardState,
    _parse_metric_dimensions,
    inspect_metric_archive,
)


def test_metric_logging_settings_validate_balanced_archive_contract():
    settings = MetricLoggingSettings.from_mapping(
        {
            "mode": "balanced",
            "resume_policy": "strict",
            "live_key_warning_threshold": 200,
            "archive": {
                "enabled": True,
                "local_dir": "/tmp/metrics",
                "shard_interval_steps": 10,
                "upload_to_wandb": "disabled",
                "finish_timeout_seconds": 5,
            },
        }
    )

    assert settings.mode == "balanced"
    assert settings.live_key_warning_threshold == 200
    assert settings.archive.enabled
    assert settings.archive.shard_interval_steps == 10
    assert settings.archive.upload_to_wandb == "disabled"

    with pytest.raises(ValueError, match="archive.enabled=true"):
        MetricLoggingSettings.from_mapping({"mode": "balanced", "archive": {"enabled": False}})


def test_full_routing_preserves_every_metric_and_builds_numeric_ledger():
    media = object()
    metrics = {
        "bucket/bin/0000_0512/token_count": 12,
        "test-aux/math/acc/best@16/mean": np.float32(0.75),
        "opd/gap_abs_heatmap": media,
        "actor/loss": torch.tensor(1.25),
    }

    route = MetricRouter("full").route(metrics)

    assert route.live_metrics == metrics
    assert route.detailed_metrics == {}
    assert route.numeric_metrics == {
        "bucket/bin/0000_0512/token_count": 12.0,
        "test-aux/math/acc/best@16/mean": 0.75,
        "actor/loss": 1.25,
    }
    assert set(route.deliveries.values()) == {"live"}


def test_balanced_routing_archives_high_cardinality_families_losslessly():
    heatmap = object()
    metrics = {
        # Fine and coarse position diagnostics.
        "bucket/bin/0000_0512/token_count": 100,
        "bucket/coarse/head/sample_count": 8,
        "opd/gap_abs_bin/0000_0512": 0.5,
        "opd/reward/coarse/deep_tail/p99": -1.5,
        # Low-K ladder and full coverage diagnostics.
        "opd/overlap/top1_rate": 0.1,
        "opd/overlap/top16_rate": 0.8,
        "distillation/overlap_ratio_top4": 0.4,
        "distillation/overlap_ratio": 0.7,
        "distillation/coverage/teacher_on_student_top16/mean": 0.6,
        "student/prob_topk/think_end/mean": 0.2,
        # Validation bootstrap versus fixed aux summaries.
        "test-aux/math/acc/best@16/mean": 0.9,
        "teacher-val-aux/math/reward/std@16": 0.1,
        "test-aux/format/accepted_answer_rate": 0.95,
        "val-aux/response_length/p99": 1024,
        # Dynamic intervention labels and static diagnostics.
        "opd_intervention/skipped_reason/custom_reason": 0.2,
        "opd_intervention/candidates/invalid_reason/new_reason": 0.1,
        "special_tokens/resolved/think_end_id": 151668,
        "contagion/bad_tail_gap_threshold": 2.0,
        # Repeated fixed-cardinality summaries keep their canonical value live.
        "rollout/response_length/p99": 1024,
        "response_length/p90": 900,
        "timing_s/agent_loop/generate_sequences/max": 3.0,
        "timing_s/gen": 3.1,
        "critic/advantages/max": 1.0,
        "critic/advantages/mean": 0.0,
        "rollout_corr/chi2_seq": 0.2,
        "rollout_corr/kl": 0.1,
        "response_end/ended_by_im_end_count": 4,
        "response_end/ended_by_im_end_ratio": 0.5,
        "distillation/token_reward_direct/p99": 2.0,
        "distillation/token_reward_direct/mean": 0.25,
        "distillation/token_reward_direct/teacher_on_student_topk_coverage": 0.98,
        "distillation/only_stu_teacher_topk/overlap_ratio": 0.98,
        "distillation/only_stu_teacher_topk/overlap_mass_student": 0.999,
        "distillation/only_stu_teacher_topk/student_top1_hit_rate": 1.0,
        "distillation/only_stu_teacher_topk/overlap_ratio_bin/0_512": 0.99,
        # Known aliases are archived; canonical keys stay live.
        "opd/gap_abs/high_fraction": 0.3,
        "opd/gap_abs/high_token_fraction": 0.3,
        "distillation/teacher_sampled_prob/mean": 0.01,
        "opd/overlap/teacher_sampled_prob_mean": 0.01,
        # Stable global metrics and media stay live.
        "opd/gap_abs/mean": 0.4,
        "training/global_step": 12,
        "opd/gap_abs_heatmap": heatmap,
    }

    route = MetricRouter("balanced", configured_topk=16).route(metrics)

    expected_archived = {
        "bucket/bin/0000_0512/token_count",
        "bucket/coarse/head/sample_count",
        "opd/gap_abs_bin/0000_0512",
        "opd/reward/coarse/deep_tail/p99",
        "opd/overlap/top1_rate",
        "distillation/overlap_ratio_top4",
        "distillation/coverage/teacher_on_student_top16/mean",
        "student/prob_topk/think_end/mean",
        "test-aux/math/acc/best@16/mean",
        "teacher-val-aux/math/reward/std@16",
        "opd_intervention/skipped_reason/custom_reason",
        "opd_intervention/candidates/invalid_reason/new_reason",
        "special_tokens/resolved/think_end_id",
        "contagion/bad_tail_gap_threshold",
        "rollout/response_length/p99",
        "timing_s/agent_loop/generate_sequences/max",
        "critic/advantages/max",
        "rollout_corr/chi2_seq",
        "response_end/ended_by_im_end_count",
        "distillation/token_reward_direct/p99",
        "distillation/only_stu_teacher_topk/overlap_ratio_bin/0_512",
        "opd/gap_abs/high_fraction",
        "opd/overlap/teacher_sampled_prob_mean",
    }
    assert set(route.detailed_metrics) == expected_archived
    assert set(metrics) == (
        (set(route.live_metrics) - {key for key in route.live_metrics if key.startswith("metric_archive/")})
        | set(route.detailed_metrics)
    )
    assert route.live_metrics["opd/gap_abs_heatmap"] is heatmap
    assert route.live_metrics["opd/overlap/top16_rate"] == 0.8
    assert route.live_metrics["distillation/overlap_ratio"] == 0.7
    assert route.live_metrics["test-aux/format/accepted_answer_rate"] == 0.95
    assert route.live_metrics["response_length/p90"] == 900
    assert route.live_metrics["timing_s/gen"] == 3.1
    assert route.live_metrics["critic/advantages/mean"] == 0.0
    assert route.live_metrics["rollout_corr/kl"] == 0.1
    assert route.live_metrics["response_end/ended_by_im_end_ratio"] == 0.5
    assert route.live_metrics["distillation/token_reward_direct/mean"] == 0.25
    assert route.live_metrics["distillation/token_reward_direct/teacher_on_student_topk_coverage"] == 0.98
    assert route.live_metrics["distillation/only_stu_teacher_topk/overlap_ratio"] == 0.98
    assert route.live_metrics["distillation/only_stu_teacher_topk/overlap_mass_student"] == 0.999
    assert route.live_metrics["distillation/only_stu_teacher_topk/student_top1_hit_rate"] == 1.0
    assert route.live_metrics["metric_archive/archived_key_count"] == len(expected_archived)
    assert set(route.numeric_metrics) == {
        key for key in route.live_metrics | route.detailed_metrics if key != "opd/gap_abs_heatmap"
    }


def test_balanced_router_warns_but_never_truncates_live_metrics(caplog):
    metrics = {f"actor/value_{index}": float(index) for index in range(5)}

    route = MetricRouter("balanced", live_key_warning_threshold=4).route(metrics)

    assert all(key in route.live_metrics for key in metrics)
    assert "exceeds warning threshold" in caplog.text


def test_balanced_routing_keeps_only_fixed_outcome_and_paired_repair_surface_live_on_cpu():
    outcome_prefix = "opd_intervention/outcome"
    metrics = {
        f"{outcome_prefix}/pre_correct_post_correct_fraction_among_repaired": 0.1,
        f"{outcome_prefix}/pre_correct_post_wrong_fraction_among_repaired": 0.2,
        f"{outcome_prefix}/pre_wrong_post_correct_fraction_among_repaired": 0.3,
        f"{outcome_prefix}/pre_wrong_post_wrong_fraction_among_repaired": 0.4,
        f"{outcome_prefix}/net_accuracy_delta_all_samples": 0.1,
        f"{outcome_prefix}/sample_count": 32,
        f"{outcome_prefix}/pre_wrong_post_correct_count": 3,
        "test-repair-core/aime24/acc/mean@16": 0.5,
        "test-repair-core/aime24/acc/paired_delta@16": 0.125,
        "test-repair-core/aime24/acc/maj@16": 0.6,
        "test-repair-aux/aime24/outcome/repair_executed_count": 8,
        "val-oracle-repair-core/math/reward/mean@4": 0.75,
        "val-oracle-repair-core/math/reward/paired_delta@4": 0.25,
        "val-oracle-repair-aux/math/outcome/pre_wrong_post_correct_count": 2,
    }

    route = MetricRouter("balanced").route(metrics)

    expected_live = {
        f"{outcome_prefix}/pre_correct_post_wrong_fraction_among_repaired",
        f"{outcome_prefix}/pre_wrong_post_correct_fraction_among_repaired",
        f"{outcome_prefix}/net_accuracy_delta_all_samples",
        "test-repair-core/aime24/acc/mean@16",
        "test-repair-core/aime24/acc/paired_delta@16",
        "val-oracle-repair-core/math/reward/mean@4",
        "val-oracle-repair-core/math/reward/paired_delta@4",
    }
    assert expected_live <= set(route.live_metrics)
    assert f"{outcome_prefix}/pre_correct_post_correct_fraction_among_repaired" in route.detailed_metrics
    assert f"{outcome_prefix}/pre_wrong_post_wrong_fraction_among_repaired" in route.detailed_metrics
    assert set(route.detailed_metrics) == set(metrics) - expected_live
    assert _parse_metric_dimensions("test-repair-core/aime24/acc/mean@16")["eval_scope"] == "test-repair"
    assert (
        _parse_metric_dimensions("val-oracle-repair-aux/math/outcome/sample_count")["eval_scope"] == "val-oracle-repair"
    )


def test_balanced_routing_keeps_only_compact_dmpd_boundary_allowlist_live_on_cpu():
    live_keys = {
        "opd_intervention/rate",
        "opd_intervention/first_position_frac_mean",
        "opd_intervention/future_gap/proposal_rate",
        "opd_intervention/future_gap/acceptance_fraction_among_proposals",
        "opd_intervention/future_gap/active_gap/reduction_mean_among_accepted",
        "opd_intervention/future_gap/confirmation_tokens_mean",
        "opd_intervention/future_gap/continuation_tokens_mean_among_accepted",
        "opd_intervention/future_gap/confirmation_terminal_fraction",
        "opd_intervention/repair_boundary/entry/teacher_self_nll_first16_mean",
        "opd_intervention/repair_boundary/entry/student_nll_on_teacher_first16_mean",
        "opd_intervention/repair_boundary/exit/active_gap_after_first16_mean",
        "opd_intervention/local_effect_v2/active_gap/reduction_first16_mean",
        "opd_intervention/local_effect_v2/active_gap/worsened_fraction_first16",
        "opd_intervention/local_effect_v2/pair_fraction_first16",
        "opd_intervention/outcome/pre_wrong_post_correct_fraction_among_repaired",
        "opd_intervention/outcome/pre_correct_post_wrong_fraction_among_repaired",
        "opd_intervention/outcome/net_accuracy_delta_all_samples",
    }
    detailed_keys = {
        "opd_intervention/count_mean",
        "opd_intervention/local_effect_v2/active_gap/reduction_first8_mean",
        "opd_intervention/repair_boundary/unavailable_reason/student_topk_unavailable",
        "opd_intervention/future_gap/active_gap/reduction_mean_among_rejected",
        "opd_intervention/future_gap/rejected_reason/future_gap_after_threshold",
        "opd_intervention/future_gap/gate_cpu_seconds_mean",
        "opd_intervention/future_gap/archive_serialization_cpu_seconds_previous_event",
    }
    metrics = {key: float(index) for index, key in enumerate(sorted(live_keys | detailed_keys))}

    balanced = MetricRouter("balanced").route(metrics)
    full = MetricRouter("full").route(metrics)

    assert live_keys <= set(balanced.live_metrics)
    assert detailed_keys <= set(balanced.detailed_metrics)
    assert set(full.live_metrics) == set(metrics)
    assert full.detailed_metrics == {}


def test_23k_128_token_bins_and_bootstrap_metrics_collapse_to_small_live_surface():
    metrics = {f"actor/stable_metric_{index}": float(index) for index in range(120)}
    for start in range(0, 23 * 1024, 128):
        end = min(start + 128, 23 * 1024)
        for family in (
            "opd/gap_abs_bin",
            "opd/reward_bin",
            "opd/entropy_bin",
            "opd/overlap/top16_bin",
            "distillation/overlap_ratio_bin",
            "student/prob_topk/think_end_bin",
        ):
            metrics[f"{family}/{start:04d}_{end:04d}"] = float(start)
    for scope in ("val", "test", "teacher-val"):
        for source in ("aime", "math", "olympiad", "gpqa"):
            for variable in ("acc", "score", "reward", "invalid_pred"):
                for statistic in (
                    "mean@16",
                    "std@16",
                    "best@2/mean",
                    "best@2/std",
                    "worst@16/mean",
                    "maj@16/std",
                ):
                    metrics[f"{scope}-aux/{source}/{variable}/{statistic}"] = 0.5
    for coarse_region in ("head", "mid", "tail", "deep_tail", "extreme_tail", "0_4k", "4k_12k"):
        for statistic in ("mean", "std", "p50", "p90", "p95", "p99", "max", "token_count"):
            metrics[f"opd/gap_abs/coarse/{coarse_region}/{statistic}"] = 1.0
    metrics.update(
        {
            "opd/gap_abs/mean": 0.5,
            "opd/reward/mean": -0.1,
            "opd/entropy/mean": 1.2,
            "opd/overlap/top16_rate": 0.8,
        }
    )

    route = MetricRouter("balanced", configured_topk=16).route(metrics)

    assert route.archived_numeric_count > 1300
    assert route.live_numeric_count == 127  # 120 stable + four OPD global + three health
    assert route.live_numeric_count < 200
    route.assert_lossless(metrics)


def test_parquet_ledger_is_complete_typed_checksummed_and_checkpoint_committed(tmp_path):
    archive_dir = tmp_path / "diagnostics"
    checkpoint_dir = tmp_path / "global_step_7"
    checkpoint_dir.mkdir()
    route = MetricRouter("balanced", configured_topk=16).route(
        {
            "actor/loss": np.float32(1.5),
            "opd/gap_abs_bin/0000_0512": float("nan"),
            "opd/reward/coarse/deep_tail/p90": float("-inf"),
            "test-aux/math/acc/best@16/mean": 0.75,
            "opd/gap_abs_heatmap": object(),
        }
    )

    writer = MetricArchiveWriter(
        archive_dir,
        logical_run_id="run-123",
        project="DMPD",
        experiment="archive-test",
        session_id="session-a",
    )
    first_event_id = writer.append_event(route, global_step=6, event_kind="train")
    heatmap_path = tmp_path / "gap-heatmap.png"
    heatmap_path.write_bytes(b"\x89PNG\r\nmetric-archive-test")
    event_file = writer.add_event_file(
        heatmap_path,
        global_step=6,
        metric_key="opd/gap_abs_heatmap",
    )
    second_event_id = writer.append_event(route, global_step=7, event_kind="validation")
    commit = writer.commit_checkpoint(checkpoint_dir, checkpoint_step=7)

    assert first_event_id != second_event_id
    assert commit.event_seq_start == 0
    assert commit.event_seq_end == 1
    assert len(commit.shards) == 1
    shard = commit.shards[0]
    assert shard.state == ShardState.CHECKPOINT_COMMITTED
    assert shard.step_start == 6
    assert shard.step_end == 7
    assert shard.row_count == 14  # four numeric inputs + three router health metrics, twice
    assert Path(shard.path).is_file()
    assert Path(shard.metadata_path).is_file()
    assert hashlib.sha256(Path(shard.path).read_bytes()).hexdigest() == shard.sha256
    assert shard.event_files == (event_file,)
    assert Path(event_file.path).read_bytes() == heatmap_path.read_bytes()
    assert event_file.sha256 == hashlib.sha256(heatmap_path.read_bytes()).hexdigest()

    table = pq.read_table(shard.path)
    assert table.schema.names == [
        "schema_version",
        "logical_run_id",
        "session_id",
        "event_id",
        "event_seq",
        "event_kind",
        "global_step",
        "checkpoint_step",
        "metric_key",
        "metric_group",
        "delivery",
        "value",
        "value_dtype",
        "bin_start",
        "bin_end",
        "coarse_region",
        "topk",
        "eval_scope",
    ]
    rows = table.to_pylist()
    assert {row["schema_version"] for row in rows} == {METRIC_ARCHIVE_SCHEMA_VERSION}
    assert {row["event_id"] for row in rows} == {first_event_id, second_event_id}
    assert {row["delivery"] for row in rows} == {"live", "archive"}
    binned = next(row for row in rows if row["metric_key"] == "opd/gap_abs_bin/0000_0512")
    assert (binned["bin_start"], binned["bin_end"]) == (0, 512)
    assert np.isnan(binned["value"])
    coarse = next(row for row in rows if row["metric_key"] == "opd/reward/coarse/deep_tail/p90")
    assert coarse["coarse_region"] == "deep_tail"
    assert np.isneginf(coarse["value"])
    evaluation = next(row for row in rows if row["metric_key"] == "test-aux/math/acc/best@16/mean")
    assert evaluation["eval_scope"] == "test"

    marker = json.loads((checkpoint_dir / METRIC_COMMIT_FILENAME).read_text())
    assert marker["logical_run_id"] == "run-123"
    assert marker["checkpoint_step"] == 7
    assert marker["shards"][0]["sha256"] == shard.sha256
    metadata = json.loads(Path(shard.metadata_path).read_text())
    assert metadata["event_files"][0]["metric_key"] == "opd/gap_abs_heatmap"
    assert writer.health_metrics()["metric_archive/last_committed_step"] == 7
    writer.close()
    assert not list(archive_dir.rglob("*.tmp"))


def test_complete_generation_records_are_compressed_and_attached_to_the_active_shard(tmp_path):
    writer = MetricArchiveWriter(
        tmp_path / "diagnostics",
        logical_run_id="generation-run",
        project="DMPD",
        experiment="generation-archive-test",
        session_id="session-a",
    )
    writer.append_event(MetricRouter("balanced").route({"val-core/score": 1.0}), global_step=3)
    records = [
        {
            "evaluation_scope": "val",
            "global_step": 3,
            "sample_index": 0,
            "input": "问题",
            "output": "完整答案",
            "score": np.float32(1.0),
            "token_ids": np.array([1, 2, 3]),
        }
    ]

    attachment = writer.add_event_records(
        records,
        global_step=3,
        metric_key="val/generations_full",
    )
    duplicate = writer.add_event_records(
        records,
        global_step=3,
        metric_key="val/generations_full",
    )
    shard = writer.close_shard()

    assert shard is not None
    assert attachment == duplicate
    assert shard.event_files == (attachment,)
    assert attachment.path.endswith(".jsonl.gz")
    assert hashlib.sha256(Path(attachment.path).read_bytes()).hexdigest() == attachment.sha256
    with gzip.open(attachment.path, "rt", encoding="utf-8") as stream:
        archived_records = [json.loads(line) for line in stream]
    assert archived_records == [
        {
            "evaluation_scope": "val",
            "global_step": 3,
            "input": "问题",
            "output": "完整答案",
            "sample_index": 0,
            "score": 1.0,
            "token_ids": [1, 2, 3],
        }
    ]
    metadata = json.loads(Path(shard.metadata_path).read_text())
    assert metadata["event_files"] == [
        {
            "global_step": 3,
            "metric_key": "val/generations_full",
            "path": attachment.path,
            "sha256": attachment.sha256,
            "size_bytes": attachment.size_bytes,
        }
    ]
    writer.close()


def test_resume_keeps_committed_shard_and_orphans_uncommitted_tail(tmp_path):
    archive_dir = tmp_path / "diagnostics"
    checkpoint_dir = tmp_path / "global_step_1"
    checkpoint_dir.mkdir()
    route = MetricRouter("balanced").route({"actor/loss": 1.0})

    first = MetricArchiveWriter(
        archive_dir,
        logical_run_id="resume-run",
        project="DMPD",
        experiment="resume-test",
        session_id="session-1",
    )
    first.append_event(route, global_step=1)
    committed = first.commit_checkpoint(checkpoint_dir, checkpoint_step=1).shards[0]
    first.append_event(route, global_step=2)
    uncommitted = first.close_shard()
    assert uncommitted is not None
    first.close(interrupted=True)

    resumed = MetricArchiveWriter(
        archive_dir,
        logical_run_id="resume-run",
        project="DMPD",
        experiment="resume-test",
        session_id="session-2",
        resume_checkpoint_step=1,
        checkpoint_dir=checkpoint_dir,
    )
    manifest = resumed.manifest
    states = {item["shard_id"]: item["state"] for item in manifest["shards"]}
    assert states[committed.shard_id] == ShardState.CHECKPOINT_COMMITTED
    assert states[uncommitted.shard_id] == ShardState.ORPHANED
    assert resumed.health_metrics()["metric_archive/orphaned_shard_count"] == 1
    assert all(item.shard_id != uncommitted.shard_id for item in resumed.pending_upload_shards())
    resumed.close()


def test_interrupted_open_shard_is_orphaned_without_materializing_partial_parquet(tmp_path):
    archive_dir = tmp_path / "diagnostics"
    writer = MetricArchiveWriter(
        archive_dir,
        logical_run_id="interrupted-run",
        project="DMPD",
        experiment="interrupted-test",
    )
    writer.append_event(MetricRouter("full").route({"actor/loss": 1.0}), global_step=1)

    writer.close(interrupted=True)

    manifest = inspect_metric_archive(archive_dir)
    assert manifest["shards"][0]["state"] == ShardState.ORPHANED
    assert not list((archive_dir / "shards").glob("*.parquet"))


def test_resume_quarantines_abandoned_atomic_temporary_files(tmp_path):
    archive_dir = tmp_path / "diagnostics"
    shards_dir = archive_dir / "shards"
    shards_dir.mkdir(parents=True)
    abandoned = shards_dir / ".metrics-step-crashed.parquet.deadbeef.tmp"
    abandoned.write_bytes(b"partial")

    writer = MetricArchiveWriter(
        archive_dir,
        logical_run_id="tmp-run",
        project="DMPD",
        experiment="tmp-test",
    )

    assert not abandoned.exists()
    orphaned = writer.manifest["orphaned_temp_files"]
    assert len(orphaned) == 1
    assert Path(orphaned[0]["path"]).read_bytes() == b"partial"
    writer.close()


def test_resume_recovers_marker_written_before_manifest_state_update(tmp_path):
    archive_dir = tmp_path / "diagnostics"
    checkpoint_dir = tmp_path / "global_step_3"
    checkpoint_dir.mkdir()
    route = MetricRouter("full").route({"actor/loss": 1.0})

    writer = MetricArchiveWriter(
        archive_dir,
        logical_run_id="marker-run",
        project="DMPD",
        experiment="marker-test",
        session_id="session-1",
    )
    writer.append_event(route, global_step=3)
    shard = writer.commit_checkpoint(checkpoint_dir, checkpoint_step=3).shards[0]
    writer.close()

    # Simulate a crash after metrics_commit.json was atomically installed but
    # before the manifest's state transition was persisted.
    manifest_path = archive_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["shards"][0]["state"] = ShardState.CLOSED
    manifest["shards"][0]["checkpoint_step"] = None
    manifest_path.write_text(json.dumps(manifest))

    resumed = MetricArchiveWriter(
        archive_dir,
        logical_run_id="marker-run",
        project="DMPD",
        experiment="marker-test",
        session_id="session-2",
        resume_checkpoint_step=3,
        checkpoint_dir=checkpoint_dir,
    )
    assert resumed.manifest["shards"][0]["state"] == ShardState.CHECKPOINT_COMMITTED
    assert resumed.manifest["shards"][0]["checkpoint_step"] == 3
    assert resumed.pending_upload_shards()[0].shard_id == shard.shard_id
    resumed.close()


def test_legacy_resume_creates_manifest_with_explicit_history_gap(tmp_path):
    checkpoint_dir = tmp_path / "global_step_20"
    checkpoint_dir.mkdir()
    archive_dir = tmp_path / "diagnostics"

    writer = MetricArchiveWriter(
        archive_dir,
        logical_run_id="legacy-run",
        project="DMPD",
        experiment="legacy-test",
        resume_checkpoint_step=20,
        checkpoint_dir=checkpoint_dir,
    )

    assert writer.manifest["legacy_before_step"] == 20
    assert writer.health_metrics()["metric_archive/last_committed_step"] == -1
    writer.close()


def test_existing_archive_identity_and_exclusive_writer_lock_are_strict(tmp_path):
    archive_dir = tmp_path / "diagnostics"
    first = MetricArchiveWriter(
        archive_dir,
        logical_run_id="run-a",
        project="DMPD",
        experiment="lock-test",
    )

    with pytest.raises(RuntimeError, match="already locked"):
        MetricArchiveWriter(
            archive_dir,
            logical_run_id="run-a",
            project="DMPD",
            experiment="lock-test",
            resume_checkpoint_step=0,
        )
    first.close()

    with pytest.raises(RuntimeError, match="identity mismatch"):
        MetricArchiveWriter(
            archive_dir,
            logical_run_id="run-b",
            project="DMPD",
            experiment="lock-test",
            resume_checkpoint_step=0,
        )


def test_upload_state_is_atomic_retryable_and_descriptors_are_complete(tmp_path):
    archive_dir = tmp_path / "diagnostics"
    checkpoint_dir = tmp_path / "global_step_4"
    checkpoint_dir.mkdir()
    writer = MetricArchiveWriter(
        archive_dir,
        logical_run_id="upload-run",
        project="DMPD",
        experiment="upload-test",
    )
    writer.append_event(MetricRouter("full").route({"actor/loss": 1.0}), global_step=4)
    shard = writer.commit_checkpoint(checkpoint_dir, checkpoint_step=4).shards[0]

    submitted = writer.mark_shard_state(shard.shard_id, ShardState.SUBMITTED)
    assert submitted.path == shard.path
    assert submitted.sha256 == shard.sha256
    assert (submitted.step_start, submitted.step_end) == (4, 4)
    failed = writer.mark_shard_state(
        shard.shard_id,
        ShardState.FAILED_RETRYABLE,
        error="temporary network error",
    )
    assert failed.state == ShardState.FAILED_RETRYABLE
    assert writer.pending_upload_shards()[0].shard_id == shard.shard_id
    uploaded = writer.mark_shard_state(
        shard.shard_id,
        ShardState.UPLOADED,
        artifact_ref="upload-run-training-diagnostics:v0",
    )
    assert uploaded.artifact_ref == "upload-run-training-diagnostics:v0"
    assert writer.pending_upload_shards() == ()
    assert inspect_metric_archive(archive_dir)["shards"][0]["state"] == ShardState.UPLOADED
    writer.close()


def test_offline_sessions_produce_ordered_sync_commands_and_block_online_transition(tmp_path):
    archive_dir = tmp_path / "diagnostics"
    first_sync = tmp_path / "wandb" / "offline-run-1" / "run-a.wandb"
    first_sync.parent.mkdir(parents=True)
    first_sync.touch()
    first = MetricArchiveWriter(
        archive_dir,
        logical_run_id="wandb-run",
        project="DMPD",
        experiment="wandb-test",
        session_id="session-1",
    )
    first.register_wandb_session(mode="offline", run_id="wandb-run", sync_file=first_sync)
    first.record_wandb_committed_range(step_start=0, step_end=0)
    first.close()

    second_sync = tmp_path / "wandb" / "offline-run-2" / "run-b.wandb"
    second_sync.parent.mkdir(parents=True)
    second_sync.touch()
    second = MetricArchiveWriter(
        archive_dir,
        logical_run_id="wandb-run",
        project="DMPD",
        experiment="wandb-test",
        session_id="session-2",
        resume_checkpoint_step=0,
    )
    second.register_wandb_session(mode="offline", run_id="wandb-run", sync_file=second_sync)
    second.record_wandb_committed_range(step_start=1, step_end=20)

    commands = second.standard_wandb_sync_commands(entity="jubgjf")
    assert len(commands) == 2
    assert "--append" not in commands[0]
    assert "--append" in commands[1]
    assert "--entity jubgjf" in commands[0]
    assert "--project DMPD" in commands[0]
    assert "--id wandb-run" in commands[0]
    sessions = second.manifest["sessions"]
    assert sessions["session-1"]["session_order"] == 0
    assert sessions["session-2"]["session_order"] == 1
    assert sessions["session-2"]["wandb_committed_ranges"] == [{"step_start": 1, "step_end": 20}]
    with pytest.raises(RuntimeError, match="remain unsynced"):
        second.validate_wandb_transition("online")

    Path(f"{first_sync}.synced").touch()
    Path(f"{second_sync}.synced").touch()
    assert second.unsynced_offline_sessions() == ()
    assert second.standard_wandb_sync_commands(entity="jubgjf") == ()
    second.validate_wandb_transition("online")
    second.close()


def test_empty_offline_session_is_not_added_to_sync_plan(tmp_path):
    archive_dir = tmp_path / "diagnostics"
    empty_sync = tmp_path / "wandb" / "empty" / "run-empty.wandb"
    empty_sync.parent.mkdir(parents=True)
    empty_sync.touch()
    first = MetricArchiveWriter(
        archive_dir,
        logical_run_id="wandb-empty-run",
        project="DMPD",
        experiment="wandb-empty-test",
        session_id="session-empty",
    )
    first.register_wandb_session(mode="offline", run_id="wandb-empty-run", sync_file=empty_sync)
    first.close()

    live_sync = tmp_path / "wandb" / "live" / "run-live.wandb"
    live_sync.parent.mkdir(parents=True)
    live_sync.touch()
    second = MetricArchiveWriter(
        archive_dir,
        logical_run_id="wandb-empty-run",
        project="DMPD",
        experiment="wandb-empty-test",
        session_id="session-live",
        resume_checkpoint_step=0,
    )
    second.register_wandb_session(mode="offline", run_id="wandb-empty-run", sync_file=live_sync)
    second.record_wandb_committed_range(step_start=1, step_end=1)

    commands = second.standard_wandb_sync_commands()
    assert len(commands) == 1
    assert str(empty_sync) not in commands[0]
    assert str(live_sync) in commands[0]
    assert "--append" not in commands[0]
    second.close()


def test_missing_offline_segment_makes_staged_shard_retryable_on_resume(tmp_path):
    archive_dir = tmp_path / "diagnostics"
    checkpoint_dir = tmp_path / "global_step_2"
    checkpoint_dir.mkdir()
    sync_file = tmp_path / "wandb" / "offline-run" / "run-a.wandb"
    sync_file.parent.mkdir(parents=True)
    sync_file.touch()
    first = MetricArchiveWriter(
        archive_dir,
        logical_run_id="offline-retry",
        project="DMPD",
        experiment="offline-retry-test",
        session_id="session-1",
    )
    first.register_wandb_session(mode="offline", run_id="offline-retry", sync_file=sync_file)
    first.append_event(MetricRouter("full").route({"actor/loss": 1.0}), global_step=2)
    shard = first.commit_checkpoint(checkpoint_dir, checkpoint_step=2).shards[0]
    first.mark_shard_state(shard.shard_id, ShardState.STAGED_OFFLINE)
    first.close()
    sync_file.unlink()

    resumed = MetricArchiveWriter(
        archive_dir,
        logical_run_id="offline-retry",
        project="DMPD",
        experiment="offline-retry-test",
        session_id="session-2",
        resume_checkpoint_step=2,
        checkpoint_dir=checkpoint_dir,
    )

    pending = resumed.pending_upload_shards()
    assert len(pending) == 1
    assert pending[0].state == ShardState.FAILED_RETRYABLE
    replacement_sync = tmp_path / "wandb" / "offline-run-retry" / "run-b.wandb"
    replacement_sync.parent.mkdir(parents=True)
    replacement_sync.touch()
    resumed.register_wandb_session(
        mode="offline",
        run_id="offline-retry",
        sync_file=replacement_sync,
    )
    resumed.mark_shard_state(
        shard.shard_id,
        ShardState.STAGED_OFFLINE,
        delivery_session_id="session-2",
    )
    resumed.close()

    verified = MetricArchiveWriter(
        archive_dir,
        logical_run_id="offline-retry",
        project="DMPD",
        experiment="offline-retry-test",
        session_id="session-3",
        resume_checkpoint_step=2,
        checkpoint_dir=checkpoint_dir,
    )
    retried_shard = next(item for item in verified.manifest["shards"] if item["shard_id"] == shard.shard_id)
    assert retried_shard["state"] == ShardState.STAGED_OFFLINE
    assert retried_shard["wandb_delivery_session_id"] == "session-2"
    assert verified.pending_upload_shards() == ()
    verified.close()


def test_resume_recovers_only_committed_live_events_missing_from_offline_history_on_cpu(tmp_path):
    archive_dir = tmp_path / "diagnostics"
    checkpoint_dir = tmp_path / "global_step_3"
    checkpoint_dir.mkdir()
    sync_file = tmp_path / "run-a.wandb"
    sync_file.touch()
    first = MetricArchiveWriter(
        archive_dir,
        logical_run_id="offline-history",
        project="DMPD",
        experiment="offline-history-test",
        session_id="session-1",
    )
    first.register_wandb_session(mode="offline", run_id="offline-history", sync_file=sync_file)
    route = MetricRouter("balanced").route(
        {
            "actor/loss": 1.0,
            "bucket/bin/0000_0128/token_count": 2.0,
        }
    )
    first.append_event(route, global_step=3)
    first.commit_checkpoint(checkpoint_dir, checkpoint_step=3)
    first.close()

    resumed = MetricArchiveWriter(
        archive_dir,
        logical_run_id="offline-history",
        project="DMPD",
        experiment="offline-history-test",
        session_id="session-2",
        resume_checkpoint_step=3,
        checkpoint_dir=checkpoint_dir,
    )
    events = resumed.unrecorded_offline_committed_live_events(checkpoint_step=3)

    assert len(events) == 1
    assert events[0].global_step == 3
    assert "actor/loss" in events[0].metrics
    assert "bucket/bin/0000_0128/token_count" not in events[0].metrics
    recovery_sync_file = tmp_path / "run-b.wandb"
    recovery_sync_file.touch()
    resumed.register_wandb_session(
        mode="offline",
        run_id="offline-history",
        sync_file=recovery_sync_file,
    )
    resumed.record_wandb_committed_range(
        step_start=3,
        step_end=3,
        recovered_event_ids=(events[0].event_id,),
    )
    resumed.close()

    recorded = MetricArchiveWriter(
        archive_dir,
        logical_run_id="offline-history",
        project="DMPD",
        experiment="offline-history-test",
        session_id="session-3",
        resume_checkpoint_step=3,
        checkpoint_dir=checkpoint_dir,
    )
    assert recorded.unrecorded_offline_committed_live_events(checkpoint_step=3) == ()
    recorded.close()


def test_replayed_missing_offline_history_supersedes_only_the_lost_sync_segment_on_cpu(tmp_path):
    archive_dir = tmp_path / "diagnostics"
    checkpoint_dir = tmp_path / "global_step_1"
    checkpoint_dir.mkdir()
    lost_sync = tmp_path / "lost.wandb"
    lost_sync.touch()
    first = MetricArchiveWriter(
        archive_dir,
        logical_run_id="offline-supersede",
        project="DMPD",
        experiment="offline-supersede-test",
        session_id="session-lost",
    )
    first.register_wandb_session(mode="offline", run_id="offline-supersede", sync_file=lost_sync)
    first.append_event(MetricRouter("balanced").route({"actor/loss": 1.0}), global_step=1)
    first.commit_checkpoint(checkpoint_dir, checkpoint_step=1)
    first.record_wandb_committed_range(step_start=1, step_end=1)
    first.close()
    lost_sync.unlink()

    replacement_sync = tmp_path / "replacement.wandb"
    replacement_sync.touch()
    replacement = MetricArchiveWriter(
        archive_dir,
        logical_run_id="offline-supersede",
        project="DMPD",
        experiment="offline-supersede-test",
        session_id="session-replacement",
        resume_checkpoint_step=1,
        checkpoint_dir=checkpoint_dir,
    )
    replacement.register_wandb_session(
        mode="offline",
        run_id="offline-supersede",
        sync_file=replacement_sync,
    )
    events = replacement.unrecorded_offline_committed_live_events(checkpoint_step=1)
    assert len(events) == 1
    replacement.record_wandb_committed_range(
        step_start=1,
        step_end=1,
        recovered_event_ids=(events[0].event_id,),
    )
    replacement.mark_wandb_history_recovered(("session-lost",))

    pending_sessions = replacement.unsynced_offline_sessions()
    assert [item["session_id"] for item in pending_sessions] == ["session-replacement"]
    commands = replacement.standard_wandb_sync_commands()
    assert len(commands) == 1
    assert str(replacement_sync) in commands[0]
    assert str(lost_sync) not in commands[0]
    assert "--append" not in commands[0]

    Path(f"{replacement_sync}.synced").touch()
    replacement.validate_wandb_transition("online")
    replacement.close()
