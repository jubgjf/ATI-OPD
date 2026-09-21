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

"""Plot step-wise local K3 reduction and normalized handoff compatibility.

The input is the DeepSeek DMPD-Future main run. K3 before intervention is
reconstructed from the logged post-intervention value and local reduction,
while K3 after intervention is the first-16 K3 observed when the Student
resumes generation after the Teacher span. The second panel normalizes the
post-intervention exit K3 by the matched pre-intervention K3 and intervention
threshold, giving two dimensionless references at one.
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = REPO_ROOT / "analysis_artifacts" / "jubgjf" / "DMPD" / "runs"
OUTPUT_DIR = REPO_ROOT / "analysis" / "figures"
CACHE_ROOT = REPO_ROOT / ".analysis_cache"
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

try:
    from analysis.plot_style import (
        GRID_COLOR,
        SQUARE_PANEL_MARGINS,
        configure_style,
        create_panel,
        save_figure,
        set_panel_title,
    )
except ModuleNotFoundError:  # Support ``python analysis/local_gap_dynamics.py``.
    from plot_style import (
        GRID_COLOR,
        SQUARE_PANEL_MARGINS,
        configure_style,
        create_panel,
        save_figure,
        set_panel_title,
    )

RUN_ID = "802c696019c1449aa3328bca0e1806eb"
# Legacy W&B schema keys retained for compatibility with downloaded artifacts.
AFTER = "opd_intervention/repair_boundary/exit/active_gap_after_first16_mean"
REDUCTION = "opd_intervention/local_effect_v2/active_gap/reduction_first16_mean"
SMOOTHING_WINDOW = 7
K3_THRESHOLD = 0.15
RAW_COLOR = "#B97964"
AFTER_INTERVENTION_COLOR = "#6689A6"
THRESHOLD_RATIO_COLOR = "#7C9E87"
REFERENCE_COLOR = "#626970"


def load_gap_series() -> pd.DataFrame:
    run_dir = ARTIFACTS_ROOT / RUN_ID
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if metadata.get("entity") != "jubgjf" or metadata.get("project") != "DMPD":
        raise ValueError(f"Run {RUN_ID} is not from jubgjf/DMPD.")

    rows_by_step: dict[int, tuple[float, float]] = {}
    for shard in sorted((run_dir / "history").glob("steps_*.jsonl.gz")):
        with gzip.open(shard, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                step = row.get("training/global_step")
                after_intervention = row.get(AFTER)
                reduction = row.get(REDUCTION)
                if (
                    isinstance(step, int | float)
                    and isinstance(after_intervention, int | float)
                    and isinstance(reduction, int | float)
                    and np.isfinite(float(after_intervention))
                    and np.isfinite(float(reduction))
                ):
                    rows_by_step[int(step)] = (
                        float(after_intervention) + float(reduction),
                        float(after_intervention),
                    )

    frame = pd.DataFrame(
        [(step, *values) for step, values in sorted(rows_by_step.items())],
        columns=["step", "k3_before_intervention", "k3_after_intervention"],
    )
    frame = frame.loc[(frame["k3_before_intervention"] >= 0) & (frame["k3_after_intervention"] >= 0)].reset_index(
        drop=True
    )
    if len(frame) != 217 or frame["step"].min() != 1 or frame["step"].max() != 217:
        raise ValueError(f"Expected steps 1--217 for run {RUN_ID}, found {len(frame)} rows.")
    if not bool((frame["k3_after_intervention"] > 0).all()):
        raise ValueError("The post-intervention exit-gap series contains a non-positive value.")
    return frame


def style_axes(ax: mpl.axes.Axes) -> None:
    ax.set_xlim(1, 217)
    ax.set_xticks([1, 50, 100, 150, 200])
    ax.set_xlabel("Training step")
    ax.grid(color=GRID_COLOR, linewidth=0.7, alpha=0.85)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=4)


def render_local_reduction(frame: pd.DataFrame) -> None:
    smoothed = (
        frame[["k3_before_intervention", "k3_after_intervention"]]
        .rolling(
            SMOOTHING_WINDOW,
            center=True,
            min_periods=1,
        )
        .mean()
    )
    marker_steps = list(range(0, len(frame), 24))

    fig, ax = create_panel(SQUARE_PANEL_MARGINS)
    ax.axhline(K3_THRESHOLD, color=REFERENCE_COLOR, linestyle="--", linewidth=1.2, zorder=1)
    ax.plot(
        frame["step"],
        smoothed["k3_before_intervention"],
        color=RAW_COLOR,
        linewidth=2.0,
        marker="^",
        markersize=4.2,
        markevery=marker_steps,
    )
    ax.plot(
        frame["step"],
        smoothed["k3_after_intervention"],
        color=AFTER_INTERVENTION_COLOR,
        linewidth=2.0,
        marker="o",
        markersize=4.2,
        markevery=marker_steps,
    )
    ax.set_ylim(0, 1.08 * float(smoothed["k3_before_intervention"].max()))
    set_panel_title(ax, "Local K3 reduction")
    style_axes(ax)
    save_figure(fig, OUTPUT_DIR, "local_gap_reduction_deepseek")


def render_normalized_exit_gap(frame: pd.DataFrame) -> None:
    smoothed = (
        frame[["k3_before_intervention", "k3_after_intervention"]]
        .rolling(
            SMOOTHING_WINDOW,
            center=True,
            min_periods=1,
        )
        .mean()
    )
    exit_over_before = smoothed["k3_after_intervention"] / smoothed["k3_before_intervention"]
    exit_over_threshold = smoothed["k3_after_intervention"] / K3_THRESHOLD
    marker_steps = list(range(0, len(frame), 24))

    fig, ax = create_panel(SQUARE_PANEL_MARGINS)
    ax.axhspan(0, 1, color=THRESHOLD_RATIO_COLOR, alpha=0.08, linewidth=0, zorder=0)
    ax.axhline(1, color=REFERENCE_COLOR, linestyle="--", linewidth=1.2, zorder=1)
    ax.plot(
        frame["step"],
        exit_over_before,
        color=AFTER_INTERVENTION_COLOR,
        linewidth=2.0,
        marker="o",
        markersize=4.2,
        markevery=marker_steps,
        zorder=2,
    )
    ax.plot(
        frame["step"],
        exit_over_threshold,
        color=THRESHOLD_RATIO_COLOR,
        linewidth=2.0,
        marker="s",
        markersize=4.2,
        markevery=marker_steps,
        zorder=2,
    )
    ax.text(0.97, 0.94, "Below 1 is better", transform=ax.transAxes, ha="right", va="top", fontsize=10.5)
    ax.set_ylim(0, 1.9)
    set_panel_title(ax, "Normalized handoff gap")
    style_axes(ax)
    save_figure(fig, OUTPUT_DIR, "exit_gap_deepseek")


def render_legends() -> None:
    local_handles = [
        Line2D([], [], color=RAW_COLOR, marker="^", linewidth=2.0, markersize=5),
        Line2D([], [], color=AFTER_INTERVENTION_COLOR, marker="o", linewidth=2.0, markersize=5),
        Line2D([], [], color=REFERENCE_COLOR, linestyle="--", linewidth=1.2),
    ]
    fig = plt.figure(figsize=(6.0, 0.35))
    fig.legend(
        local_handles,
        ["K3 before intervention", "K3 after intervention", r"Trigger $\tau$"],
        loc="center",
        frameon=False,
        ncol=3,
        handlelength=1.5,
        columnspacing=0.9,
    )
    save_figure(
        fig,
        OUTPUT_DIR,
        "local_gap_reduction_legend",
        export_mode="auxiliary",
        transparent=True,
    )

    normalized_handles = [
        Line2D([], [], color=AFTER_INTERVENTION_COLOR, marker="o", linewidth=2.0, markersize=5),
        Line2D([], [], color=THRESHOLD_RATIO_COLOR, marker="s", linewidth=2.0, markersize=5),
    ]
    fig = plt.figure(figsize=(2.9, 0.35))
    fig.legend(
        normalized_handles,
        ["After / before", r"After / $\tau$"],
        loc="center",
        frameon=False,
        ncol=2,
        handlelength=1.5,
        columnspacing=0.9,
    )
    save_figure(
        fig,
        OUTPUT_DIR,
        "normalized_exit_gap_legend",
        export_mode="auxiliary",
        transparent=True,
    )


def main() -> None:
    configure_style()
    frame = load_gap_series()
    render_local_reduction(frame)
    render_normalized_exit_gap(frame)
    render_legends()


if __name__ == "__main__":
    main()
