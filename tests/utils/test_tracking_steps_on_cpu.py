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

import sys
from types import ModuleType

import pytest

from verl.utils.tracking import Tracking, ValidationGenerationsLogger


class _RecordingBackend:
    def __init__(self):
        self.calls = []

    def log(self, data, step):
        self.calls.append((data, step))


def _tracking_with_backends(**backends) -> Tracking:
    tracking = object.__new__(Tracking)
    tracking.logger = backends
    tracking._last_step = None
    tracking._finished = False
    return tracking


def test_tracking_allows_same_step_merges_but_rejects_decreasing_steps_on_cpu():
    backend = _RecordingBackend()
    tracking = _tracking_with_backends(wandb=backend)

    tracking.log({"train/loss": 2.0}, step=3)
    tracking.log({"val/score": 1.0}, step=3)

    assert [step for _data, step in backend.calls] == [3, 3]
    with pytest.raises(ValueError, match="monotonically non-decreasing"):
        tracking.log({"late/metric": 1.0}, step=2)
    assert len(backend.calls) == 2


def test_tracking_enforces_monotonic_steps_per_backend_for_offline_replay_on_cpu():
    wandb_backend = _RecordingBackend()
    console_backend = _RecordingBackend()
    tracking = _tracking_with_backends(wandb=wandb_backend, console=console_backend)

    tracking.log({"train/loss": 3.0}, step=3, backend=["console"])
    tracking.log({"train/loss": 1.0}, step=1, backend=["wandb"])
    tracking.log({"train/loss": 2.0}, step=2, backend=["wandb"])

    assert [step for _data, step in console_backend.calls] == [3]
    assert [step for _data, step in wandb_backend.calls] == [1, 2]
    with pytest.raises(ValueError, match="backend='wandb'.*last_step=2"):
        tracking.log({"late/metric": 1.0}, step=1, backend=["wandb"])


def test_tracking_rejects_diagnostics_logged_to_a_different_step_on_cpu():
    backend = _RecordingBackend()
    tracking = _tracking_with_backends(wandb=backend)

    with pytest.raises(ValueError, match=r"source_step=4.*step=5"):
        tracking.log({"opd/diagnostics_async/source_step": 4.0}, step=5)

    assert backend.calls == []


def test_validation_wandb_table_can_be_deferred_into_the_main_step_log_on_cpu(monkeypatch):
    logged = []

    class _Table:
        def __init__(self, columns, data=None):
            self.columns = columns
            self.data = list(data or [])

        def add_data(self, *values):
            self.data.append(list(values))

    _Table.__module__ = "wandb.sdk.data_types"
    wandb = ModuleType("wandb")
    wandb.Table = _Table
    wandb.run = object()
    wandb.log = lambda data, step: logged.append((data, step))
    monkeypatch.setitem(sys.modules, "wandb", wandb)

    generation_logger = ValidationGenerationsLogger()
    deferred = generation_logger.log(
        ["wandb"],
        [{"input": "question", "output": "answer", "score": 1.0}],
        step=6,
        defer_wandb=True,
    )

    assert logged == []
    assert deferred["val/generations"].data == [[6, 0, "question", "answer", 1.0]]

    wandb_backend = _RecordingBackend()
    console_backend = _RecordingBackend()
    tracking = _tracking_with_backends(wandb=wandb_backend, console=console_backend)
    tracking.log({"train/loss": 0.5, **deferred}, step=6)

    assert wandb_backend.calls == [({"train/loss": 0.5, **deferred}, 6)]
    assert console_backend.calls == [({"train/loss": 0.5}, 6)]


def test_validation_wandb_table_can_limit_each_event_to_current_step_preview_on_cpu(monkeypatch):
    class _Table:
        def __init__(self, columns, data=None):
            self.columns = columns
            self.data = list(data or [])

        def add_data(self, *values):
            self.data.append(list(values))

    _Table.__module__ = "wandb.sdk.data_types"
    wandb = ModuleType("wandb")
    wandb.Table = _Table
    wandb.run = object()
    wandb.log = lambda *_args, **_kwargs: pytest.fail("deferred tables must not log directly")
    monkeypatch.setitem(sys.modules, "wandb", wandb)

    generation_logger = ValidationGenerationsLogger(cumulative_wandb_table=False)
    first = generation_logger.log(
        ["wandb"],
        [{"input": "question-1", "output": "answer-1", "score": 1.0}],
        step=6,
        defer_wandb=True,
    )
    second = generation_logger.log(
        ["wandb"],
        [{"input": "question-2", "output": "answer-2", "score": 0.0}],
        step=7,
        defer_wandb=True,
    )

    assert first["val/generations"].data == [[6, 0, "question-1", "answer-1", 1.0]]
    assert second["val/generations"].data == [[7, 0, "question-2", "answer-2", 0.0]]
    assert not hasattr(generation_logger, "validation_table")


def test_validation_wandb_table_keeps_direct_logging_for_other_trainers_on_cpu(monkeypatch):
    logged = []

    class _Table:
        def __init__(self, columns, data=None):
            self.data = list(data or [])

        def add_data(self, *values):
            self.data.append(list(values))

    wandb = ModuleType("wandb")
    wandb.Table = _Table
    wandb.run = object()
    wandb.log = lambda data, step: logged.append((data, step))
    monkeypatch.setitem(sys.modules, "wandb", wandb)

    deferred = ValidationGenerationsLogger().log(
        ["wandb"],
        [("question", "answer", 1.0)],
        step=2,
    )

    assert deferred == {}
    assert len(logged) == 1
    assert logged[0][1] == 2
