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

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from analysis import long_sequence_divergence_slices as divergence_data


def _canonical_run_record() -> dict[str, Any]:
    return {
        "id": divergence_data.RUN_ID,
        "name": divergence_data.EXPECTED_RUN_NAME,
        "entity": "jubgjf",
        "project": "DMPD",
        "state": "finished",
        "last_history_step": divergence_data.EXPECTED_TRAINING_STEPS,
        "config": {
            "actor_rollout_ref": {
                "model": {"path": f"/models/{divergence_data.EXPECTED_STUDENT_SUFFIX}"},
                "actor": {"optim": {"total_training_steps": divergence_data.EXPECTED_TRAINING_STEPS}},
            },
            "data": {
                "train_files": [f"/datasets/{divergence_data.EXPECTED_TRAIN_FILE_SUFFIX}"],
                "max_response_length": divergence_data.EXPECTED_RESPONSE_LENGTH,
            },
            "distillation": {
                "enabled": True,
                "use_actor_loss": False,
                "teacher_models": {
                    "teacher_model": {"model_path": f"/models/{divergence_data.EXPECTED_TEACHER_SUFFIX}"}
                },
                "distillation_loss": {
                    "loss_mode": "k1",
                    "policy_loss_mode": "vanilla",
                    "horizon": {"mode": "full"},
                },
                "intervention": {
                    "enabled": False,
                    "prune": False,
                    "train_on_teacher_segments": False,
                },
                "diagnostics": {
                    "enabled": True,
                    "position_bin_size": divergence_data.EXPECTED_BIN_SIZE,
                },
            },
        },
    }


def _write_run_record(run_dir: Path, record: dict[str, Any] | None = None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps(record or _canonical_run_record()), encoding="utf-8")


def _metric_rows(start_step: int, end_step: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in range(start_step, end_step + 1):
        for bin_start, bin_end in divergence_data._expected_bins():
            no_support = (
                bin_start == divergence_data.EXPECTED_RESPONSE_LENGTH - divergence_data.EXPECTED_BIN_SIZE
                and step % 20 == 10
            )
            count = 0 if no_support else 32
            rows.append(
                {
                    "global_step": step,
                    "metric_key": f"bucket/bin/{bin_start:04d}_{bin_end:04d}/token_count",
                    "value": count,
                    "bin_start": bin_start,
                    "bin_end": bin_end,
                }
            )
            if count:
                rows.append(
                    {
                        "global_step": step,
                        "metric_key": f"opd/gap_abs_bin/{bin_start:04d}_{bin_end:04d}",
                        "value": 0.1 + step / 1000 + bin_start / 100_000,
                        "bin_start": bin_start,
                        "bin_end": bin_end,
                    }
                )
    return rows


def _write_diagnostics(artifact_root: Path) -> list[Path]:
    ranges = [(1 + 20 * index, 20 + 20 * index) for index in range(10)] + [(201, 217)]
    versions = []
    paths = []
    for index, (start, end) in enumerate(ranges):
        version = f"v{index}"
        relative = Path("shards") / f"metrics-step-{start:09d}-{end:09d}-test.parquet"
        path = artifact_root / version / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(_metric_rows(start, end)).to_parquet(path, index=False)
        versions.append(
            {
                "name": f"{divergence_data.RUN_ID}-training-diagnostics:{version}",
                "version": version,
                "digest": f"digest-{index}",
                "entries": {str(relative): {"size": 0, "digest": None}},
            }
        )
        paths.append(path)
    artifact_root.mkdir(parents=True, exist_ok=True)
    (artifact_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "training-diagnostics",
                "run_id": divergence_data.RUN_ID,
                "versions": versions,
            }
        ),
        encoding="utf-8",
    )
    return paths


def test_validate_canonical_run_accepts_pinned_vanilla_opd_configuration(tmp_path: Path) -> None:
    _write_run_record(tmp_path)
    divergence_data.validate_canonical_run(tmp_path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda record: record["config"]["actor_rollout_ref"]["model"].update(path="/models/Nemotron"),
            "student path",
        ),
        (
            lambda record: record["config"]["distillation"]["intervention"].update(enabled=True),
            "intervention enabled",
        ),
        (
            lambda record: record["config"]["distillation"]["diagnostics"].update(position_bin_size=1024),
            "position bin size",
        ),
    ],
)
def test_validate_canonical_run_rejects_wrong_configuration(
    tmp_path: Path,
    mutate: Any,
    message: str,
) -> None:
    record = copy.deepcopy(_canonical_run_record())
    mutate(record)
    _write_run_record(tmp_path, record)
    with pytest.raises(ValueError, match=message):
        divergence_data.validate_canonical_run(tmp_path)


def test_load_gap_data_preserves_all_steps_and_zero_support_mask(tmp_path: Path) -> None:
    artifact_root = tmp_path / "training-diagnostics"
    _write_diagnostics(artifact_root)

    data = divergence_data.load_gap_data(artifact_root)

    assert data.gap_abs.shape == (217, 46)
    assert tuple(data.gap_abs.index) == divergence_data.EXPECTED_GAP_STEPS
    assert not np.isnan(
        data.gap_abs.loc[1, divergence_data.EXPECTED_RESPONSE_LENGTH - divergence_data.EXPECTED_BIN_SIZE]
    )
    assert bool(
        np.isnan(data.gap_abs.loc[10, divergence_data.EXPECTED_RESPONSE_LENGTH - divergence_data.EXPECTED_BIN_SIZE])
    )

    display = divergence_data.coarsen_gap_data(data)
    assert display.gap_abs.shape == (22, 12)
    assert tuple(display.position_edges[:3]) == (0.0, 2048.0, 4096.0)
    assert display.position_edges[-1] == divergence_data.EXPECTED_RESPONSE_LENGTH
    assert len(display.step_edges) == 23


def test_diagnostic_shards_reject_noncontiguous_ranges(tmp_path: Path) -> None:
    artifact_root = tmp_path / "training-diagnostics"
    paths = _write_diagnostics(artifact_root)
    old_path = paths[1]
    new_path = old_path.with_name(old_path.name.replace("000000021", "000000022"))
    old_path.rename(new_path)
    manifest_path = artifact_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["versions"][1]["entries"].pop(f"shards/{old_path.name}")
    manifest["versions"][1]["entries"][f"shards/{new_path.name}"] = entry
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="not contiguous"):
        divergence_data.diagnostic_shard_paths(artifact_root)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate", "duplicate"),
        ("nonfinite", "non-finite"),
        ("wrong_bin", "position bins"),
    ],
)
def test_load_gap_data_rejects_invalid_raw_rows(tmp_path: Path, mutation: str, message: str) -> None:
    artifact_root = tmp_path / "training-diagnostics"
    paths = _write_diagnostics(artifact_root)
    frame = pd.read_parquet(paths[0])
    gap_index = frame.index[frame["metric_key"].str.startswith(divergence_data.GAP_PREFIX)][0]
    if mutation == "duplicate":
        frame = pd.concat([frame, frame.loc[[gap_index]]], ignore_index=True)
    elif mutation == "nonfinite":
        frame.loc[gap_index, "value"] = np.inf
    else:
        frame.loc[gap_index, "bin_end"] = int(frame.loc[gap_index, "bin_end"]) + 1
    frame.to_parquet(paths[0], index=False)

    with pytest.raises(ValueError, match=message):
        divergence_data.load_gap_data(artifact_root)
