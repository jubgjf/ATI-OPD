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

# ruff: noqa: E402

"""Render the DeepSeek DMPD loss and benchmark ablation figures.

The script reads only the local ``jubgjf/DMPD`` mirror. It deliberately pins
every paper row to a run ID and validates the relevant configuration before
using any data. Missing HMMT results are replaced with zeros solely to size
provisional layout bars; such bars are marked with a dagger and must not be
reported as final six-benchmark results.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_ROOT = REPO_ROOT / "analysis_artifacts" / "jubgjf" / "DMPD" / "runs"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "analysis" / "figures"
CACHE_ROOT = REPO_ROOT / ".analysis_cache"
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

try:
    from analysis.plot_style import (
        FIGURE6_REGULAR_PANEL_MARGINS,
        GRID_COLOR,
        STANDARD_PANEL_MARGINS,
        configure_style,
        create_panel,
        save_figure,
        set_panel_title,
    )
except ModuleNotFoundError:  # Support ``python analysis/ablation_summary.py``.
    from plot_style import (
        FIGURE6_REGULAR_PANEL_MARGINS,
        GRID_COLOR,
        STANDARD_PANEL_MARGINS,
        configure_style,
        create_panel,
        save_figure,
        set_panel_title,
    )

EXPECTED_TRAINING_STEPS = 217
EXPECTED_STUDENT_SUFFIX = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
EXPECTED_TEACHER_SUFFIX = "hbx/JustRL-DeepSeek-1.5B"
EXPECTED_TRAIN_FILE_SUFFIX = "DAPO-Math-17k-Processed-verl--qwen3/train.parquet"
DAGGER = "\N{DAGGER}"
BENCHMARK_KEYS = (
    "test-legacy-core/aime24/acc/mean@16",
    "test-legacy-core/aime25/acc/mean@16",
    "test-legacy-core/amc23/acc/mean@16",
    "test-legacy-core/AMO-Bench/acc/mean@16",
)


@dataclass(frozen=True)
class ExpectedIntervention:
    train_on_teacher: bool
    loss_mode: str
    sft_coef: float
    prune: bool
    prune_mode: str
    max_interventions: int
    skip_correct: bool
    teacher_segment_length: int


@dataclass(frozen=True)
class RunSpec:
    key: str
    label: str
    run_id: str
    expected: ExpectedIntervention
    hmmt_feb_2026: float | None = None
    hmmt_nov_2025: float | None = None
    hmmt_source: str | None = None


@dataclass(frozen=True)
class LoadedRun:
    spec: RunSpec
    rows: tuple[dict[str, Any], ...]
    progress_step: int | None


@dataclass(frozen=True)
class BenchmarkResult:
    evaluation_step: int
    progress_step: int
    four_benchmark_values: tuple[float, ...]
    hmmt_values: tuple[float, float]
    six_benchmark_average_percent: float
    uses_hmmt_placeholder: bool

    @property
    def provisional(self) -> bool:
        return (
            self.progress_step < EXPECTED_TRAINING_STEPS
            or self.evaluation_step < EXPECTED_TRAINING_STEPS
            or self.uses_hmmt_placeholder
        )


LOSS_STYLES: dict[str, dict[str, str]] = {
    "no_teacher": {
        "color": "#8B8E91",
        "linestyle": "-",
        "marker": "s",
    },
    "opd_loss": {
        "color": "#6689A6",
        "linestyle": "-",
        "marker": "o",
    },
    "sft_loss": {
        "color": "#C7785E",
        "linestyle": "-",
        "marker": "^",
    },
}

BAR_COLORS = {
    "no_teacher": "#8B8E91",
    "opd_loss": "#6689A6",
    "sft_loss": "#C7785E",
    "full_m1": "#8E859D",
    "full_m4": "#C3A46E",
    "prune": "#7A9E85",
    "skip_incorrect": "#B97964",
}


def _expected(
    *,
    train_on_teacher: bool,
    loss_mode: str = "opd",
    sft_coef: float = 1.0,
    prune: bool,
    prune_mode: str,
    max_interventions: int,
    skip_correct: bool = False,
    teacher_segment_length: int = 128,
) -> ExpectedIntervention:
    return ExpectedIntervention(
        train_on_teacher=train_on_teacher,
        loss_mode=loss_mode,
        sft_coef=sft_coef,
        prune=prune,
        prune_mode=prune_mode,
        max_interventions=max_interventions,
        skip_correct=skip_correct,
        teacher_segment_length=teacher_segment_length,
    )


RUN_SPECS: dict[str, RunSpec] = {
    "no_teacher": RunSpec(
        key="no_teacher",
        label="Masked",
        run_id="39ce8dd19e6b4451bd5175609358110d",
        expected=_expected(
            train_on_teacher=False,
            prune=True,
            prune_mode="adaptive_future",
            max_interventions=1,
        ),
        hmmt_feb_2026=0.22159091,
        hmmt_nov_2025=0.20416667,
        hmmt_source="single-eval_11",
    ),
    "opd_loss": RunSpec(
        key="opd_loss",
        label="OPD loss",
        run_id="802c696019c1449aa3328bca0e1806eb",
        expected=_expected(
            train_on_teacher=True,
            prune=True,
            prune_mode="adaptive_future",
            max_interventions=1,
        ),
        # Standalone evaluations already reported in the current main-results table.
        hmmt_feb_2026=0.2424,
        hmmt_nov_2025=0.2083,
        hmmt_source="existing canonical standalone evaluation",
    ),
    "sft_loss": RunSpec(
        key="sft_loss",
        label="SFT loss",
        run_id="f7c7a430d8de464992d9de225413b3a1",
        expected=_expected(
            train_on_teacher=True,
            loss_mode="sft",
            sft_coef=0.1,
            prune=True,
            prune_mode="adaptive_future",
            max_interventions=1,
        ),
        hmmt_feb_2026=0.19696970,
        hmmt_nov_2025=0.18333333,
        hmmt_source="single-eval_12",
    ),
    "full_m1": RunSpec(
        key="full_m1",
        label=r"Full ($M=1$)",
        run_id="38457043f2534dbbb700eeb7ce4b6542",
        expected=_expected(
            train_on_teacher=True,
            prune=False,
            prune_mode="teacher_end",
            max_interventions=1,
        ),
        hmmt_feb_2026=0.21401515,
        hmmt_nov_2025=0.20416667,
        hmmt_source="single-eval_10",
    ),
    "full_m4": RunSpec(
        key="full_m4",
        label=r"Full ($M=4$)",
        run_id="08806c2c3e8547449299f0fc3e62c766",
        expected=_expected(
            train_on_teacher=True,
            prune=False,
            prune_mode="teacher_end",
            max_interventions=4,
        ),
        hmmt_feb_2026=0.23674242,
        hmmt_nov_2025=0.20416667,
        hmmt_source="single-eval_9",
    ),
    "prune": RunSpec(
        key="prune",
        label="Prune",
        run_id="c7c88aa34a4a4097b73968039a8ccc89",
        expected=_expected(
            train_on_teacher=True,
            prune=True,
            prune_mode="teacher_end",
            max_interventions=1,
        ),
    ),
    "skip_incorrect": RunSpec(
        key="skip_incorrect",
        label="Skip incorrect",
        run_id="323d9444cf2246409d023fed76381653",
        expected=_expected(
            train_on_teacher=True,
            prune=True,
            prune_mode="adaptive_future",
            max_interventions=1,
            skip_correct=True,
        ),
    ),
    "teacher_len_512": RunSpec(
        key="teacher_len_512",
        label=r"DMPD ($L_T=512$)",
        run_id="7187a1dc4bec46e3a37ff984dec7d1b2",
        expected=_expected(
            train_on_teacher=True,
            prune=True,
            prune_mode="adaptive_future",
            max_interventions=1,
            teacher_segment_length=512,
        ),
    ),
    "teacher_len_1024": RunSpec(
        key="teacher_len_1024",
        label=r"DMPD ($L_T=1024$)",
        run_id="6ca6d5692c294ffaa9c0ba928bafa993",
        expected=_expected(
            train_on_teacher=True,
            prune=True,
            prune_mode="adaptive_future",
            max_interventions=1,
            teacher_segment_length=1024,
        ),
        hmmt_feb_2026=0.22916667,
        hmmt_nov_2025=0.2125,
        hmmt_source="single-eval_13",
    ),
}

LOSS_KEYS = ("no_teacher", "opd_loss", "sft_loss")
TEACHER_LOSS_BARS = (
    ("no_teacher", "Masked"),
    ("opd_loss", "OPD loss"),
    ("sft_loss", "SFT loss"),
)
DMPD_VARIANT_BARS = (
    ("opd_loss", "DMPD"),
    ("full_m1", r"Full ($M=1$)"),
    ("full_m4", r"Full ($M=4$)"),
    ("prune", "Prune"),
    ("skip_incorrect", "Skip incorrect"),
)
DMPD_VARIANT_POSITIONS = (4.0, 3.0, 2.0, 1.0, 0.0)
BENCHMARK_AXIS_MAX = 41.0
FIGURE6_BENCHMARK_AXIS_MAX = 45.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=DEFAULT_ARTIFACTS_ROOT,
        help="Directory containing mirrored jubgjf/DMPD run directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for PDF and PNG outputs.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate pinned runs and print the extracted values without rendering figures.",
    )
    return parser.parse_args()


def read_run_record(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing mirrored run metadata: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _normalized_prune_mode(intervention: Mapping[str, Any]) -> str:
    mode = intervention.get("prune_mode")
    if mode is None:
        # Older teacher-end runs predate the explicit prune_mode field.
        return "teacher_end"
    return str(mode)


def validate_run_config(record: Mapping[str, Any], spec: RunSpec) -> None:
    if record.get("entity") != "jubgjf" or record.get("project") != "DMPD":
        raise ValueError(f"Run {spec.run_id} is not from jubgjf/DMPD.")
    if str(record.get("id")) != spec.run_id:
        raise ValueError(f"Run directory for {spec.run_id} contains metadata for {record.get('id')!r}.")

    config = record.get("config") or {}
    student_path = ((config.get("actor_rollout_ref") or {}).get("model") or {}).get("path")
    teacher_path = (((config.get("distillation") or {}).get("teacher_models") or {}).get("teacher_model") or {}).get(
        "model_path"
    )
    train_files = (config.get("data") or {}).get("train_files") or []
    configured_steps = (((config.get("actor_rollout_ref") or {}).get("actor") or {}).get("optim") or {}).get(
        "total_training_steps"
    )
    intervention = (config.get("distillation") or {}).get("intervention") or {}
    actual = {
        "train_on_teacher": bool(intervention.get("train_on_teacher_segments", False)),
        "loss_mode": str(intervention.get("teacher_segment_loss_mode", "opd")),
        "sft_coef": float(intervention.get("teacher_segment_sft_loss_coef", 1.0)),
        "prune": bool(intervention.get("prune", False)),
        "prune_mode": _normalized_prune_mode(intervention),
        "max_interventions": int(intervention.get("max_interventions_per_sample", 0)),
        "skip_correct": bool(intervention.get("skip_correct", False)),
        "teacher_segment_length": int(intervention.get("teacher_segment_length", 0)),
    }
    expected = {
        "train_on_teacher": spec.expected.train_on_teacher,
        "loss_mode": spec.expected.loss_mode,
        "sft_coef": spec.expected.sft_coef,
        "prune": spec.expected.prune,
        "prune_mode": spec.expected.prune_mode,
        "max_interventions": spec.expected.max_interventions,
        "skip_correct": spec.expected.skip_correct,
        "teacher_segment_length": spec.expected.teacher_segment_length,
    }
    mismatches = [
        f"{key}: expected {expected[key]!r}, got {actual[key]!r}" for key in expected if actual[key] != expected[key]
    ]
    if not str(student_path or "").endswith(EXPECTED_STUDENT_SUFFIX):
        mismatches.append(f"student path: expected suffix {EXPECTED_STUDENT_SUFFIX!r}, got {student_path!r}")
    if not str(teacher_path or "").endswith(EXPECTED_TEACHER_SUFFIX):
        mismatches.append(f"teacher path: expected suffix {EXPECTED_TEACHER_SUFFIX!r}, got {teacher_path!r}")
    if len(train_files) != 1 or not str(train_files[0]).endswith(EXPECTED_TRAIN_FILE_SUFFIX):
        mismatches.append(
            f"train files: expected one path ending in {EXPECTED_TRAIN_FILE_SUFFIX!r}, got {train_files!r}"
        )
    if configured_steps != EXPECTED_TRAINING_STEPS:
        mismatches.append(f"configured training steps: expected {EXPECTED_TRAINING_STEPS}, got {configured_steps!r}")
    if record.get("state") != "finished":
        mismatches.append(f"run state: expected 'finished', got {record.get('state')!r}")
    if record.get("last_history_step") != EXPECTED_TRAINING_STEPS:
        mismatches.append(
            f"last history step: expected {EXPECTED_TRAINING_STEPS}, got {record.get('last_history_step')!r}"
        )
    if mismatches:
        details = "; ".join(mismatches)
        raise ValueError(f"Pinned run {spec.run_id} ({spec.label}) failed configuration validation: {details}")


def merge_history_rows(rows: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Merge duplicate W&B rows by history step without discarding late non-null metrics."""
    by_step: dict[int, dict[str, Any]] = {}
    for raw_row in rows:
        history_step = raw_row.get("_step")
        if not isinstance(history_step, int | float) or not math.isfinite(float(history_step)):
            continue
        step = int(history_step)
        merged = by_step.setdefault(step, {})
        for key, value in raw_row.items():
            if value is not None or key not in merged:
                merged[key] = value
    return tuple(by_step[step] for step in sorted(by_step))


def read_history(run_dir: Path) -> tuple[dict[str, Any], ...]:
    history_dir = run_dir / "history"
    shards = sorted(history_dir.glob("steps_*.jsonl.gz"))
    if not shards:
        return ()
    rows: list[dict[str, Any]] = []
    for shard in shards:
        with gzip.open(shard, "rt", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return merge_history_rows(rows)


def _finite_number(value: Any) -> bool:
    return isinstance(value, int | float) and math.isfinite(float(value))


def training_progress(rows: Sequence[Mapping[str, Any]]) -> int | None:
    steps = [int(row["training/global_step"]) for row in rows if _finite_number(row.get("training/global_step"))]
    return max(steps, default=None)


def metric_points(rows: Sequence[Mapping[str, Any]], metric: str) -> tuple[np.ndarray, np.ndarray]:
    points = [
        (int(row["training/global_step"]), float(row[metric]))
        for row in rows
        if _finite_number(row.get("training/global_step")) and _finite_number(row.get(metric))
    ]
    points.sort()
    if not points:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    steps, values = zip(*points, strict=True)
    return np.asarray(steps, dtype=float), np.asarray(values, dtype=float)


def latest_complete_benchmark(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[int, tuple[float, ...]] | None:
    candidates: list[tuple[int, tuple[float, ...]]] = []
    for row in rows:
        if not _finite_number(row.get("training/global_step")):
            continue
        values = tuple(row.get(key) for key in BENCHMARK_KEYS)
        if all(_finite_number(value) for value in values):
            candidates.append((int(row["training/global_step"]), tuple(float(value) for value in values)))
    return max(candidates, key=lambda item: item[0]) if candidates else None


def benchmark_result(run: LoadedRun) -> BenchmarkResult | None:
    if run.progress_step is None:
        return None
    latest = latest_complete_benchmark(run.rows)
    if latest is None:
        return None
    evaluation_step, four_values = latest
    hmmt_raw = (run.spec.hmmt_feb_2026, run.spec.hmmt_nov_2025)
    uses_placeholder = any(value is None for value in hmmt_raw)
    hmmt_values = tuple(0.0 if value is None else float(value) for value in hmmt_raw)
    average = 100.0 * (sum(four_values) + sum(hmmt_values)) / 6.0
    return BenchmarkResult(
        evaluation_step=evaluation_step,
        progress_step=run.progress_step,
        four_benchmark_values=four_values,
        hmmt_values=(hmmt_values[0], hmmt_values[1]),
        six_benchmark_average_percent=average,
        uses_hmmt_placeholder=uses_placeholder,
    )


def load_runs(artifacts_root: Path) -> dict[str, LoadedRun]:
    loaded: dict[str, LoadedRun] = {}
    for key, spec in RUN_SPECS.items():
        run_dir = artifacts_root / spec.run_id
        record = read_run_record(run_dir)
        validate_run_config(record, spec)
        rows = read_history(run_dir)
        loaded[key] = LoadedRun(spec=spec, rows=rows, progress_step=training_progress(rows))
    return loaded


def centered_rolling_mean(values: np.ndarray, window: int = 7) -> np.ndarray:
    if values.ndim != 1:
        raise ValueError(f"Expected a one-dimensional series, got shape {values.shape}.")
    smoothed = np.full(values.shape, np.nan, dtype=float)
    half = window // 2
    minimum = max(2, window // 2 + 1)
    for index in range(len(values)):
        left = max(0, index - half)
        right = min(len(values), index + half + 1)
        if right - left >= minimum:
            smoothed[index] = float(np.mean(values[left:right]))
    return smoothed


def _style_axes(ax: mpl.axes.Axes) -> None:
    ax.set_xlim(1, EXPECTED_TRAINING_STEPS)
    ax.set_xticks([1, 50, 100, 150, 200])
    ax.set_xlabel("Training step")
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7, alpha=0.85)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=4)


def dynamic_percent_axis(
    observed: Sequence[float],
    *,
    tick_step: float = 2.0,
    upper: float = 100.0,
) -> tuple[tuple[float, float], tuple[float, ...]]:
    """Choose a readable percentage range without clipping the lowest point."""
    finite = [float(value) for value in observed if math.isfinite(float(value))]
    if not finite:
        raise ValueError("Cannot derive a percentage axis without finite observations.")
    if tick_step <= 0:
        raise ValueError(f"tick_step must be positive, got {tick_step}.")
    if max(finite) > upper:
        raise ValueError(f"Observed percentage exceeds the configured upper bound {upper}: {max(finite)}.")
    first_tick = tick_step * math.floor(min(finite) / tick_step)
    lower = max(0.0, first_tick - 0.25 * tick_step)
    ticks = tuple(float(value) for value in np.arange(first_tick, upper + 0.5 * tick_step, tick_step))
    return (lower, upper), ticks


def render_loss_metric(
    runs: Mapping[str, LoadedRun],
    *,
    metric: str,
    title: str,
    stem: str,
    output_dir: Path,
    scale: float,
    sparse: bool,
    percent: bool = False,
    ylim: tuple[float, float] | None = None,
    yticks: Sequence[float] | None = None,
    dynamic_percent_range: bool = False,
) -> None:
    fig, ax = create_panel(STANDARD_PANEL_MARGINS)
    observed: list[float] = []
    for key in LOSS_KEYS:
        run = runs[key]
        steps, values = metric_points(run.rows, metric)
        if len(steps) == 0:
            continue
        values = values * scale
        observed.extend(values.tolist())
        style = LOSS_STYLES[key]
        if sparse:
            ax.plot(
                steps,
                values,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markerfacecolor="white" if key != "opd_loss" else style["color"],
                markeredgewidth=0.9,
            )
        else:
            smooth = centered_rolling_mean(values)
            ax.plot(steps, values, color=style["color"], linewidth=0.75, alpha=0.24)
            ax.plot(
                steps,
                smooth,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markerfacecolor="white" if key != "opd_loss" else style["color"],
                markeredgewidth=0.8,
                markevery=20,
            )
    if not observed:
        raise ValueError(f"No finite observations found for {metric}.")
    _style_axes(ax)
    set_panel_title(ax, title, pad=3)
    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    if dynamic_percent_range:
        if not percent:
            raise ValueError("dynamic_percent_range requires percent=True.")
        ylim, yticks = dynamic_percent_axis(observed)
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        upper = max(observed)
        ax.set_ylim(0, max(2.0, math.ceil(upper + 0.5)))
    if yticks is not None:
        ax.set_yticks(yticks)
    save_figure(fig, output_dir, stem)


def render_loss_legend(runs: Mapping[str, LoadedRun], output_dir: Path) -> None:
    handles: list[Line2D] = []
    labels: list[str] = []
    for key in LOSS_KEYS:
        style = LOSS_STYLES[key]
        run = runs[key]
        provisional = run.progress_step is None or run.progress_step < EXPECTED_TRAINING_STEPS
        handles.append(
            Line2D(
                [],
                [],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markerfacecolor="white" if key != "opd_loss" else style["color"],
                markeredgewidth=0.9,
            )
        )
        labels.append(f"{run.spec.label}{DAGGER if provisional else ''}")
    fig = plt.figure(figsize=(6.3, 0.38))
    fig.legend(
        handles,
        labels,
        loc="center",
        frameon=False,
        ncol=3,
        handlelength=2.0,
        columnspacing=0.9,
        handletextpad=0.4,
    )
    save_figure(fig, output_dir, "ablation_loss_legend", export_mode="auxiliary", transparent=True)


def _bar_color(key: str) -> str:
    return BAR_COLORS[key]


def _benchmark_patch(key: str, label: str, result: BenchmarkResult | None) -> Patch:
    color = _bar_color(key)
    provisional = result is not None and result.provisional
    return Patch(
        facecolor="white" if provisional else color,
        edgecolor=color,
        linewidth=1.5 if provisional else 1.0,
        label=f"{label}{DAGGER if provisional else ''}",
    )


def render_teacher_loss_benchmark(results: Mapping[str, BenchmarkResult | None], output_dir: Path) -> None:
    fig, ax = create_panel(STANDARD_PANEL_MARGINS)
    for position, (key, _) in enumerate(TEACHER_LOSS_BARS):
        result = results[key]
        if result is None:
            continue
        color = _bar_color(key)
        bar = ax.bar(
            position,
            result.six_benchmark_average_percent,
            width=0.62,
            color=color,
            edgecolor=color,
            linewidth=1.0,
        )[0]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.55,
            f"{result.six_benchmark_average_percent:.1f}",
            ha="center",
            va="bottom",
            fontsize=13,
            color="#333333",
        )
    ax.set_xlim(-0.65, len(TEACHER_LOSS_BARS) - 0.35)
    ax.set_xticks([])
    ax.set_ylim(0, BENCHMARK_AXIS_MAX)
    ax.set_yticks([0, 20, 40])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    set_panel_title(ax, "Benchmark Avg.", pad=4)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7, alpha=0.85)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", bottom=False)
    ax.tick_params(axis="y", direction="out", length=4)
    save_figure(fig, output_dir, "teacher_segment_loss_benchmark_avg_deepseek")


def render_teacher_loss_benchmark_legend(results: Mapping[str, BenchmarkResult | None], output_dir: Path) -> None:
    handles = [_benchmark_patch(key, label, results[key]) for key, label in TEACHER_LOSS_BARS]
    fig = plt.figure(figsize=(4.2, 0.38))
    fig.legend(
        handles=handles,
        loc="center",
        frameon=False,
        ncol=3,
        handlelength=1.4,
        columnspacing=0.9,
        handletextpad=0.4,
    )
    save_figure(
        fig,
        output_dir,
        "teacher_segment_loss_benchmark_legend",
        export_mode="auxiliary",
        transparent=True,
    )


def render_dmpd_variants_benchmark(results: Mapping[str, BenchmarkResult | None], output_dir: Path) -> None:
    fig, ax = create_panel(FIGURE6_REGULAR_PANEL_MARGINS)
    for (key, _), position in zip(DMPD_VARIANT_BARS, DMPD_VARIANT_POSITIONS, strict=True):
        result = results[key]
        if result is None:
            continue
        color = _bar_color(key)
        facecolor = "white" if result.provisional else color
        bar = ax.barh(
            position,
            result.six_benchmark_average_percent,
            height=0.58,
            color=facecolor,
            edgecolor=color,
            linewidth=1.5 if result.provisional else 1.0,
        )[0]
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{result.six_benchmark_average_percent:.1f}{DAGGER if result.provisional else ''}",
            ha="left",
            va="center",
            fontsize=13,
            color="#333333",
        )
    ax.set_yticks([])
    ax.set_xlim(0, FIGURE6_BENCHMARK_AXIS_MAX)
    ax.set_xticks([0, 20, 40])
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_xlabel("Six-task Avg.")
    set_panel_title(ax, "Benchmark Avg.", pad=4)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.7, alpha=0.85)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.6, 4.6)
    ax.tick_params(axis="y", left=False)
    ax.tick_params(axis="x", direction="out", length=4)
    save_figure(fig, output_dir, "dmpd_variants_benchmark_avg_deepseek")


def render_dmpd_variants_benchmark_legend(results: Mapping[str, BenchmarkResult | None], output_dir: Path) -> None:
    items = [_benchmark_patch(key, label, results[key]) for key, label in DMPD_VARIANT_BARS]
    # Matplotlib fills multi-row legends column-first. This input order renders
    # DMPD / Full M=1 / Full M=4 on the first row and the two provisional
    # variants on the second row.
    handles = [items[index] for index in (0, 3, 1, 4, 2)]
    fig = plt.figure(figsize=(6.3, 0.72))
    fig.legend(
        handles=handles,
        loc="center",
        frameon=False,
        ncol=3,
        handlelength=1.4,
        columnspacing=0.9,
        handletextpad=0.4,
    )
    save_figure(
        fig,
        output_dir,
        "dmpd_variants_benchmark_legend",
        export_mode="auxiliary",
        transparent=True,
    )


def print_summary(runs: Mapping[str, LoadedRun], results: Mapping[str, BenchmarkResult | None]) -> None:
    print("Pinned DeepSeek ablation runs")
    for key, run in runs.items():
        result = results.get(key)
        if result is None:
            benchmark_text = "benchmark=missing"
        else:
            benchmark_text = (
                f"benchmark_step={result.evaluation_step}, avg6={result.six_benchmark_average_percent:.2f}%, "
                f"hmmt_placeholder={result.uses_hmmt_placeholder}, hmmt_source={run.spec.hmmt_source or 'missing'}"
            )
        print(f"  {key}: run={run.spec.run_id}, progress={run.progress_step}, {benchmark_text}")


def main() -> None:
    args = parse_args()
    artifacts_root = args.artifacts_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    runs = load_runs(artifacts_root)
    results = {key: benchmark_result(run) for key, run in runs.items()}
    print_summary(runs, results)
    if args.validate_only:
        return

    configure_style()
    render_loss_metric(
        runs,
        metric="opd/overlap/top16_rate",
        title="Top-16 token coverage",
        stem="loss_top16_coverage_deepseek",
        output_dir=output_dir,
        scale=100.0,
        sparse=True,
        percent=True,
        dynamic_percent_range=True,
    )
    render_loss_metric(
        runs,
        metric="rollout/repetition_rate_token_zlib",
        title="Token-zlib repetition rate",
        stem="loss_token_zlib_repetition_rate_deepseek",
        output_dir=output_dir,
        scale=100.0,
        sparse=False,
        percent=True,
        ylim=(0.0, 4.0),
        yticks=(0.0, 1.0, 2.0, 3.0, 4.0),
    )
    render_loss_metric(
        runs,
        metric="rollout/repetition_comp_ratio_token_zlib/mean",
        title="Mean zlib compression ratio",
        stem="loss_token_zlib_compression_ratio_deepseek",
        output_dir=output_dir,
        scale=1.0,
        sparse=False,
        ylim=(6.0, 10.25),
        yticks=(6.0, 7.0, 8.0, 9.0, 10.0),
    )
    render_loss_legend(runs, output_dir)
    render_teacher_loss_benchmark(results, output_dir)
    render_teacher_loss_benchmark_legend(results, output_dir)
    render_dmpd_variants_benchmark(results, output_dir)
    render_dmpd_variants_benchmark_legend(results, output_dir)


if __name__ == "__main__":
    main()
