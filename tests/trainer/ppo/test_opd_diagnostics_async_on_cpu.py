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

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import verl.trainer.ppo.ray_trainer as ray_trainer_module
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    _compute_opd_diagnostics_background_metrics,
    _PendingOpdDiagnostics,
    _snapshot_diagnostics_kwargs,
)


def _completed_future(result: dict[str, float]) -> Future[dict[str, float]]:
    future: Future[dict[str, float]] = Future()
    future.set_result(result)
    return future


def test_diagnostics_snapshot_is_independent_from_mutable_training_inputs_on_cpu():
    tensor = torch.tensor([1.0, 2.0], requires_grad=True)
    metadata = {"positions": [1, 2]}

    snapshot = _snapshot_diagnostics_kwargs({"tensor": tensor, "metadata": metadata})
    tensor.data.add_(10)
    metadata["positions"].append(3)

    assert snapshot["tensor"].tolist() == [1.0, 2.0]
    assert snapshot["tensor"].device.type == "cpu"
    assert not snapshot["tensor"].requires_grad
    assert snapshot["tensor"].data_ptr() != tensor.data_ptr()
    assert snapshot["metadata"] == {"positions": [1, 2]}


def test_background_diagnostics_combines_opd_and_intervention_metrics_on_cpu(monkeypatch):
    monkeypatch.setattr(
        ray_trainer_module,
        "compute_opd_diagnostics_metrics",
        lambda **kwargs: {"opd/value": float(kwargs["value"])},
    )
    monkeypatch.setattr(
        ray_trainer_module,
        "compute_intervention_metrics",
        lambda **kwargs: {"intervention/value": float(kwargs["value"])},
    )

    metrics = _compute_opd_diagnostics_background_metrics({"value": 2}, {"value": 3})

    assert metrics["opd/value"] == 2.0
    assert metrics["intervention/value"] == 3.0
    assert metrics["opd/diagnostics_async/worker_compute_seconds"] >= 0.0


def test_finish_diagnostics_blocks_until_current_source_step_is_ready_on_cpu():
    result_requested = Event()
    release_result = Event()

    class _BlockingFuture:
        def result(self) -> dict[str, float]:
            result_requested.set()
            if not release_result.wait(timeout=5):
                raise TimeoutError("test did not release diagnostics result")
            return {"opd/value": 7.0}

    trainer = object.__new__(RayPPOTrainer)
    task = _PendingOpdDiagnostics(source_step=4, future=_BlockingFuture(), submitted_at=0.0)
    trainer._opd_diagnostics_task = task

    with ThreadPoolExecutor(max_workers=1) as executor:
        joined = executor.submit(trainer._finish_opd_diagnostics_for_step, task, expected_step=4)
        assert result_requested.wait(timeout=5)
        assert not joined.done()
        release_result.set()
        metrics = joined.result(timeout=5)

    assert metrics["opd/value"] == 7.0
    assert metrics["opd/diagnostics_async/source_step"] == 4.0
    assert metrics["opd/diagnostics_async/result_delay_steps"] == 0.0
    assert metrics["opd/diagnostics_async/pending"] == 0.0
    assert trainer._opd_diagnostics_task is None


def test_finish_diagnostics_rejects_wrong_commit_step_on_cpu():
    trainer = object.__new__(RayPPOTrainer)
    task = _PendingOpdDiagnostics(source_step=4, future=_completed_future({}), submitted_at=0.0)
    trainer._opd_diagnostics_task = task

    with pytest.raises(RuntimeError, match="source_step=4, expected_step=5"):
        trainer._finish_opd_diagnostics_for_step(task, expected_step=5)

    assert trainer._opd_diagnostics_task is task


def test_finish_diagnostics_failure_prevents_source_step_commit_on_cpu():
    trainer = object.__new__(RayPPOTrainer)
    future: Future[dict[str, float]] = Future()
    future.set_exception(ValueError("diagnostics failed"))
    task = _PendingOpdDiagnostics(source_step=9, future=future, submitted_at=0.0)
    trainer._opd_diagnostics_task = task

    with pytest.raises(RuntimeError, match="source step 9") as exc_info:
        trainer._finish_opd_diagnostics_for_step(task, expected_step=9)

    assert isinstance(exc_info.value.__cause__, ValueError)
    assert trainer._opd_diagnostics_task is None


class _RecordingExecutor:
    def __init__(self):
        self.calls: list[tuple[Any, tuple[Any, ...]]] = []

    def submit(self, fn, *args):
        self.calls.append((fn, args))
        return _completed_future({"opd/background": 1.0})


def test_async_submit_allows_only_one_task_and_preserves_source_step_on_cpu():
    trainer = object.__new__(RayPPOTrainer)
    trainer._opd_diagnostics_task = None
    trainer._distillation_diagnostics_config = lambda: SimpleNamespace(async_metrics=True)
    trainer._build_opd_diagnostics_inputs = lambda **_kwargs: (
        {
            "student_logprobs": torch.tensor([[1.0]]),
            "response_mask": torch.tensor([[True]]),
            "entropy": None,
        },
        None,
        torch.tensor([[0.5]]),
        None,
    )
    trainer._maybe_log_opd_heatmaps = lambda **_kwargs: {"opd/heatmap": "media"}
    executor = _RecordingExecutor()
    trainer._ensure_opd_diagnostics_executor = lambda: executor

    foreground_metrics, task = trainer._submit_or_compute_opd_diagnostics_metrics(
        batch=object(), entropy=None, global_steps=12
    )

    assert task is trainer._opd_diagnostics_task
    assert task.source_step == 12
    assert foreground_metrics["opd/heatmap"] == "media"
    assert foreground_metrics["opd/gap_abs/available"] == 1.0
    assert foreground_metrics["opd/diagnostics_async/submitted"] == 1.0
    assert foreground_metrics["opd/diagnostics_async/pending_after_submit"] == 1.0
    assert len(executor.calls) == 1

    with pytest.raises(RuntimeError, match="active_source_step=12, requested_source_step=13"):
        trainer._submit_or_compute_opd_diagnostics_metrics(batch=object(), entropy=None, global_steps=13)

    completed_metrics = trainer._finish_opd_diagnostics_for_step(task, expected_step=12)
    assert completed_metrics["opd/background"] == 1.0
    assert completed_metrics["opd/diagnostics_async/source_step"] == 12.0


def test_sync_diagnostics_path_returns_metrics_without_task_on_cpu():
    trainer = object.__new__(RayPPOTrainer)
    trainer._distillation_diagnostics_config = lambda: SimpleNamespace(async_metrics=False)
    trainer._compute_opd_diagnostics_metrics = lambda **_kwargs: {"opd/value": 3.0}

    metrics, task = trainer._submit_or_compute_opd_diagnostics_metrics(batch=object(), entropy=None, global_steps=2)

    assert metrics == {"opd/value": 3.0}
    assert task is None


class _FakeTrackingLogger:
    def __init__(self):
        self.finished_with: list[int] = []

    def finish(self, exit_code=0):
        self.finished_with.append(exit_code)


@pytest.mark.parametrize("raises", [False, True])
def test_fit_always_shuts_down_diagnostics_and_finishes_tracking_on_cpu(raises):
    trainer = object.__new__(RayPPOTrainer)
    logger = _FakeTrackingLogger()
    shutdown_calls = []
    trainer.init_logger = lambda: logger
    trainer._shutdown_opd_diagnostics_executor = lambda: shutdown_calls.append(True)

    if raises:

        def _fit_impl(_logger):
            raise ValueError("training failed")

        trainer._fit_impl = _fit_impl
        with pytest.raises(ValueError, match="training failed"):
            trainer.fit()
    else:
        trainer._fit_impl = lambda _logger: "complete"
        assert trainer.fit() == "complete"

    assert shutdown_calls == [True]
    assert logger.finished_with == [1 if raises else 0]
