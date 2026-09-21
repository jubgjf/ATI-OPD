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

"""Plot how often DMPD-Future invokes teacher intervention during training.

Outputs one independent 3 x 3 inch DeepSeek panel and one standalone legend. The
script reads the canonical run histories directly from ``analysis_artifacts``
and writes only PNG/PDF files to ``analysis/figures``.
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
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

try:
    from analysis.plot_style import (
        GRID_COLOR,
        STANDARD_PANEL_MARGINS,
        configure_style,
        create_panel,
        save_figure,
        set_panel_title,
    )
except ModuleNotFoundError:  # Support ``python analysis/intervention_rate.py``.
    from plot_style import (
        GRID_COLOR,
        STANDARD_PANEL_MARGINS,
        configure_style,
        create_panel,
        save_figure,
        set_panel_title,
    )

METRIC = "opd_intervention/rate"
SMOOTH_WINDOW = 7
DMPD_COLOR = "#6689A6"
RUN_ID = "802c696019c1449aa3328bca0e1806eb"


def load_metric(run_id: str) -> pd.DataFrame:
    """Load one numeric metric directly from gzip JSONL history shards."""
    run_dir = ARTIFACTS_ROOT / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Missing canonical run directory: {run_dir}")
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if metadata.get("entity") != "jubgjf" or metadata.get("project") != "DMPD":
        raise ValueError(f"Run {run_id} is not from jubgjf/DMPD.")

    values_by_step: dict[int, float] = {}
    shards = sorted((run_dir / "history").glob("steps_*.jsonl.gz"))
    if not shards:
        raise FileNotFoundError(f"No history shards found for run {run_id}.")
    for shard in shards:
        with gzip.open(shard, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                step = row.get("training/global_step")
                value = row.get(METRIC)
                if isinstance(step, int | float) and isinstance(value, int | float):
                    values_by_step[int(step)] = float(value)
    frame = pd.DataFrame(sorted(values_by_step.items()), columns=["step", "value"])
    if len(frame) < 200:
        raise ValueError(f"Run {run_id} has only {len(frame)} finite observations for {METRIC}.")
    return frame


def finish_axis(ax: mpl.axes.Axes) -> None:
    ax.set_xlim(1, 217)
    ax.set_ylim(0, 100)
    ax.set_xticks([1, 50, 100, 150, 200])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_xlabel("Training step")
    set_panel_title(ax, "Intervention rate")
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7, alpha=0.85)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=4)


def render_panel(slug: str, run_id: str) -> None:
    frame = load_metric(run_id)
    smooth = frame["value"].rolling(SMOOTH_WINDOW, center=True, min_periods=4).mean()
    fig, ax = create_panel(STANDARD_PANEL_MARGINS)
    ax.scatter(
        frame["step"],
        100 * frame["value"],
        s=13,
        color=DMPD_COLOR,
        alpha=0.18,
        linewidths=0,
        zorder=2,
    )
    ax.plot(frame["step"], 100 * smooth, color=DMPD_COLOR, zorder=3)
    finish_axis(ax)
    save_figure(fig, OUTPUT_DIR, f"intervention_rate_{slug}")


def render_legend() -> None:
    handles = [
        Line2D([], [], linestyle="none", marker="o", markersize=5, color=DMPD_COLOR, alpha=0.35),
        Line2D([], [], color=DMPD_COLOR, linewidth=2),
    ]
    fig = plt.figure(figsize=(4.2, 0.35))
    fig.legend(
        handles,
        ["Per-step batch", "7-step mean"],
        loc="center",
        frameon=False,
        ncol=2,
        handlelength=1.8,
        columnspacing=1.1,
    )
    save_figure(fig, OUTPUT_DIR, "intervention_rate_legend", export_mode="auxiliary", transparent=True)


def main() -> None:
    configure_style()
    render_panel("deepseek", RUN_ID)
    render_legend()


if __name__ == "__main__":
    main()
