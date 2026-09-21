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
import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from analysis import online_conditional_dynamics as diagnostics


def _canonical_run_record() -> dict[str, Any]:
    return {
        "id": diagnostics.RUN_ID,
        "entity": "jubgjf",
        "project": "DMPD",
        "state": "finished",
        "last_history_step": 217,
        "config": {
            "actor_rollout_ref": {
                "model": {"path": f"/models/{diagnostics.EXPECTED_STUDENT_SUFFIX}"},
                "actor": {"optim": {"total_training_steps": 217}},
            },
            "data": {"train_files": [f"/datasets/{diagnostics.EXPECTED_TRAIN_FILE_SUFFIX}"]},
            "distillation": {
                "teacher_models": {"teacher_model": {"model_path": f"/models/{diagnostics.EXPECTED_TEACHER_SUFFIX}"}},
                "intervention": {
                    "enabled": True,
                    "max_interventions_per_sample": 1,
                    "prune": True,
                    "prune_mode": "adaptive_future",
                    "gap_metric": "k3",
                    "teacher_segment_length": 128,
                    "train_on_teacher_segments": True,
                    "teacher_segment_loss_mode": "opd",
                    "skip_correct": False,
                },
            },
        },
    }


def _write_run_record(run_dir: Path, record: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps(record), encoding="utf-8")


def _outcome_record(
    index: int,
    *,
    raw_correct: bool = False,
    final_correct: bool = False,
    no_cut: bool = False,
    local_k3_reduction: float | None = 0.1,
    intervened: bool = True,
) -> dict[str, Any]:
    intervention = {
        "local_effect_gap_reduction": local_k3_reduction,
        "prune_future_no_cut": no_cut,
    }
    return {
        "global_step": 1,
        "uid": f"sample-{index}",
        "sample_index": index,
        "outcome": {
            # Legacy artifact schema field retained by the production reader.
            "repair_executed": intervened,
            "intervention_count": int(intervened),
            "raw_correct": raw_correct,
            "final_correct": final_correct,
        },
        "interventions": [intervention] if intervened else [],
    }


def _write_outcomes(root: Path, records: list[dict[str, Any]]) -> None:
    attachment_dir = root / "v0" / "attachments"
    attachment_dir.mkdir(parents=True)
    path = attachment_dir / "step-000000001-opd_intervention-outcome_pairs-test.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def test_validate_canonical_run_accepts_pinned_configuration(tmp_path: Path) -> None:
    _write_run_record(tmp_path, _canonical_run_record())
    diagnostics.validate_canonical_run(tmp_path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda record: record["config"]["actor_rollout_ref"]["model"].update(path="/models/Nemotron"),
            "student path",
        ),
        (
            lambda record: record["config"]["distillation"]["intervention"].update(max_interventions_per_sample=4),
            "max_interventions_per_sample",
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
        diagnostics.validate_canonical_run(tmp_path)


def test_load_outcomes_partitions_groups_and_excludes_missing_local_k3(tmp_path: Path) -> None:
    records = [
        _outcome_record(0, raw_correct=False, final_correct=True, no_cut=False, local_k3_reduction=0.2),
        _outcome_record(1, raw_correct=True, final_correct=False, no_cut=False, local_k3_reduction=0.0),
        _outcome_record(2, raw_correct=False, final_correct=True, no_cut=True, local_k3_reduction=0.3),
        _outcome_record(3, raw_correct=False, final_correct=False, no_cut=True, local_k3_reduction=-0.1),
        _outcome_record(4, raw_correct=True, final_correct=True, no_cut=True, local_k3_reduction=None),
        _outcome_record(5, intervened=False),
    ]
    _write_outcomes(tmp_path, records)

    dataset = diagnostics.load_outcome_data(tmp_path, expected_steps={1}, expected_batch_size=6)

    assert dataset.total_rollouts == 6
    assert dataset.intervened_rollouts == 5
    assert dataset.missing_local_k3 == 1
    assert len(dataset.suffix_observations) == 4
    assert sum(dataset.step_counts[1].values()) == 5
    assert [(observation.no_cut, observation.local_k3_decreased) for observation in dataset.suffix_observations] == [
        (False, True),
        (False, False),
        (True, True),
        (True, False),
    ]

    estimates = diagnostics.estimate_suffix_gate(
        dataset.suffix_observations,
        cluster_steps={1},
        bootstrap_samples=20,
        seed=7,
    )
    assert [estimate.count for estimate in estimates] == [1, 1, 1, 1]
    assert [estimate.net_gain_pp for estimate in estimates] == [100.0, -100.0, 100.0, 0.0]


def test_load_outcomes_rejects_duplicate_sample_key(tmp_path: Path) -> None:
    record = _outcome_record(0)
    _write_outcomes(tmp_path, [record, copy.deepcopy(record)])
    with pytest.raises(RuntimeError, match="Duplicate outcome record"):
        diagnostics.load_outcome_data(tmp_path, expected_steps={1}, expected_batch_size=2)


def test_suffix_gate_uses_group_denominator_and_deterministic_step_bootstrap() -> None:
    observations = [
        diagnostics.SuffixGateObservation(1, False, True, False, True),
        diagnostics.SuffixGateObservation(1, False, True, False, True),
        diagnostics.SuffixGateObservation(2, False, True, True, False),
        diagnostics.SuffixGateObservation(2, False, True, False, False),
    ]
    for step in (1, 2):
        observations.extend(
            [
                diagnostics.SuffixGateObservation(step, False, False, False, False),
                diagnostics.SuffixGateObservation(step, True, True, True, True),
                diagnostics.SuffixGateObservation(step, True, False, False, False),
            ]
        )

    first = diagnostics.estimate_suffix_gate(
        observations,
        cluster_steps={1, 2},
        bootstrap_samples=1_000,
        seed=123,
    )
    second = diagnostics.estimate_suffix_gate(
        observations,
        cluster_steps={1, 2},
        bootstrap_samples=1_000,
        seed=123,
    )

    assert first == second
    assert first[0].count == 4
    assert first[0].net_gain_pp == pytest.approx(25.0)
    assert [estimate.count for estimate in first[1:]] == [2, 2, 2]
