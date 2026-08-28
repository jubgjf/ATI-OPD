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

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopManager,
    AgentLoopWorker,
    _collect_prune_future_calibration_values,
    _compute_prune_future_horizon_metadata,
    _GlobalPruneFutureCalibration,
    _PruneFutureCalibrationSnapshot,
)
from verl.workers.config import DistillationInterventionConfig


def _config(**overrides) -> DistillationInterventionConfig:
    values = {
        "enabled": True,
        "prune": True,
        "prune_mode": "adaptive_future",
        "max_interventions_per_sample": 1,
        "gap_metric": "k3",
        "gap_aggregation": "mean",
        "gap_threshold": 1.0,
        "prune_future_window_tokens": 2,
        "prune_future_min_valid_tokens": 2,
        "prune_future_bad_window_persistence": 2,
        "prune_future_calibration_window_blocks": 16,
        "prune_future_calibration_min_blocks": 4,
        "prune_future_max_healthy_blocks_per_sample": 8,
    }
    values.update(overrides)
    return DistillationInterventionConfig(**values)


def _horizon(scores, *, teacher_end=2, nonfinite_positions=None, **config_overrides):
    tensor = torch.tensor(scores, dtype=torch.float32)
    config = _config(**config_overrides)
    return _compute_prune_future_horizon_metadata(
        scores=tensor,
        eligibility_mask=[1] * len(scores),
        teacher_end=teacher_end,
        intervention_config=config,
        calibration=_PruneFutureCalibrationSnapshot(
            threshold=1.0,
            source="rolling_raw_correct_quantile",
            count=128,
            version=3,
        ),
        nonfinite_positions=nonfinite_positions,
    )


def test_prune_future_two_bad_blocks_cut_before_the_first_bad_block_on_cpu():
    metadata = _horizon([0.0, 0.0, 2.0, 2.0, 3.0, 3.0])

    assert metadata["prune_future_cutoff"] == 2
    assert metadata["prune_future_horizon_tokens"] == 0
    assert metadata["prune_future_first_bad_block_index"] == 0
    assert metadata["prune_future_bad_blocks"] == 2
    assert metadata["prune_future_persistent_bad"] is True


def test_prune_future_bad_good_bad_does_not_cut_on_cpu():
    metadata = _horizon([0.0, 0.0, 2.0, 2.0, 0.5, 0.5, 3.0, 3.0])

    assert metadata["prune_future_cutoff"] == 8
    assert metadata["prune_future_no_cut"] is True
    assert metadata["prune_future_bad_blocks"] == 2


def test_prune_future_nonfinite_block_breaks_bad_window_persistence_on_cpu():
    metadata = _horizon(
        [0.0, 0.0, 2.0, 2.0, 9.0, 9.0, 3.0, 3.0, 4.0, 4.0],
        nonfinite_positions=[4],
    )

    assert metadata["prune_future_cutoff"] == 6
    assert metadata["prune_future_unavailable_blocks"] == 1
    assert metadata["prune_future_first_bad_block_index"] == 2


@pytest.mark.parametrize(
    ("scores", "teacher_end", "expected_horizon", "expected_unavailable"),
    [
        ([0.0, 0.0], 2, 0, 0),
        ([0.0, 0.0, 0.5], 2, 1, 1),
        ([0.0, 0.0, 0.5, 0.5, 0.7, 0.7], 2, 4, 0),
    ],
)
def test_prune_future_teacher_terminal_short_and_safe_suffixes_are_legal_on_cpu(
    scores,
    teacher_end,
    expected_horizon,
    expected_unavailable,
):
    metadata = _horizon(scores, teacher_end=teacher_end)

    assert metadata["prune_future_horizon_tokens"] == expected_horizon
    assert metadata["prune_future_unavailable_blocks"] == expected_unavailable
    assert metadata["prune_future_no_cut"] is True


def test_prune_future_calibration_uses_at_most_eight_evenly_spaced_blocks_with_endpoints_on_cpu():
    scores = torch.repeat_interleave(torch.arange(12, dtype=torch.float32), 2)
    values, counts = _collect_prune_future_calibration_values(
        scores=scores,
        eligibility_mask=[1] * scores.numel(),
        intervention_config=_config(),
    )

    assert len(values) == 8
    assert values[0] == 0.0
    assert values[-1] == 11.0
    assert counts == {
        "total_blocks": 12,
        "available_blocks": 12,
        "unavailable_blocks": 0,
        "selected_blocks": 8,
    }


def test_prune_future_global_calibration_warmup_q90_and_rolling_window_on_cpu():
    calibration = _GlobalPruneFutureCalibration(
        _config(
            gap_threshold=0.15,
            prune_future_healthy_quantile=0.90,
            prune_future_calibration_window_blocks=4,
            prune_future_calibration_min_blocks=3,
        )
    )
    assert calibration.snapshot() == _PruneFutureCalibrationSnapshot(
        threshold=0.15,
        source="gap_threshold_fallback",
        count=0,
        version=0,
    )

    snapshot = calibration.add_updates([[0.1, 0.2], []])
    assert snapshot.threshold == pytest.approx(0.15)
    assert snapshot.source == "gap_threshold_fallback"
    snapshot = calibration.add_updates([[0.3], [0.4]])
    assert snapshot.threshold == pytest.approx(float(np.quantile([0.1, 0.2, 0.3, 0.4], 0.90)))
    assert snapshot.source == "rolling_raw_correct_quantile"
    assert snapshot.version == 2

    snapshot = calibration.add_updates([[0.5, 0.6], []])
    assert list(calibration.history) == pytest.approx([0.3, 0.4, 0.5, 0.6])
    assert snapshot.count == 4


class _RemoteMethod:
    def __init__(self, function):
        self.function = function

    def remote(self, *args):
        async def invoke():
            return self.function(*args)

        return invoke()


class _PruneCalibrationWorker:
    def __init__(self, updates):
        self.updates = list(updates)
        self.snapshot = None
        self.drain_prune_future_calibration_updates = _RemoteMethod(self._drain)
        self.set_prune_future_calibration_state = _RemoteMethod(self._set)

    def _drain(self):
        updates = self.updates
        self.updates = []
        return updates

    def _set(self, snapshot):
        self.snapshot = snapshot


@pytest.mark.asyncio
async def test_prune_future_manager_broadcasts_updates_only_between_batches_on_cpu():
    manager = object.__new__(AgentLoopManager)
    manager._global_teacher_nll_calibration = None
    manager._global_prune_future_calibration = _GlobalPruneFutureCalibration(
        _config(prune_future_calibration_min_blocks=3)
    )
    manager.agent_loop_workers = [_PruneCalibrationWorker([0.1, 0.2]), _PruneCalibrationWorker([0.3])]

    before = manager._global_prune_future_calibration.snapshot()
    await AgentLoopManager._refresh_global_teacher_nll_calibration(manager)

    assert before.source == "gap_threshold_fallback"
    assert all(worker.snapshot.version == 1 for worker in manager.agent_loop_workers)
    assert all(worker.snapshot.count == 3 for worker in manager.agent_loop_workers)
    assert all(worker.snapshot.source == "rolling_raw_correct_quantile" for worker in manager.agent_loop_workers)


@pytest.mark.asyncio
async def test_prune_future_validation_is_read_only_on_cpu():
    worker = SimpleNamespace(teacher_intervention_enabled=True, _prune_future_pending_updates=[0.25])

    await AgentLoopWorker._maybe_repair_with_teacher(
        worker,
        output=object(),
        validate=True,
        sampling_params=None,
        sample_kwargs=None,
    )

    assert worker._prune_future_pending_updates == [0.25]
