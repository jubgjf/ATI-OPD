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

"""Compare late-training response-tail visitation for OPD and DMPD-Future."""

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

import matplotlib.pyplot as plt
import numpy as np
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
    )
except ModuleNotFoundError:  # Support ``python analysis/tail_visitation.py``.
    from plot_style import (
        GRID_COLOR,
        STANDARD_PANEL_MARGINS,
        configure_style,
        create_panel,
        save_figure,
    )

LATE_WINDOW = 32
SURVIVAL_METRICS = (
    (r"$>4\mathrm{K}$", "response_length/survival_gt_4k"),
    (r"$>12\mathrm{K}$", "response_length/survival_gt_12k"),
    (r"$>20\mathrm{K}$", "response_length/survival_gt_20k"),
    (r"$>23\mathrm{K}$", "response_length/survival_gt_23k"),
    ("Max", "response_length/clip_ratio"),
)
OPD_RUN_ID = "9bbeb176889b42aa847665d03274fd68"
DMPD_RUN_ID = "802c696019c1449aa3328bca0e1806eb"
METHODS = {
    "opd": {"label": "OPD", "color": "#C7785E", "linestyle": "-", "marker": "^"},
    "dmpd": {"label": "DMPD", "color": "#6689A6", "linestyle": "-", "marker": "o"},
}


def load_history(run_id: str) -> pd.DataFrame:
    run_dir = ARTIFACTS_ROOT / run_id
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if metadata.get("entity") != "jubgjf" or metadata.get("project") != "DMPD":
        raise ValueError(f"Run {run_id} is not from jubgjf/DMPD.")
    rows_by_step: dict[int, dict[str, float]] = {}
    required = [key for _, key in SURVIVAL_METRICS]
    for shard in sorted((run_dir / "history").glob("steps_*.jsonl.gz")):
        with gzip.open(shard, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                step = row.get("training/global_step")
                if not isinstance(step, int | float):
                    continue
                output = rows_by_step.setdefault(int(step), {"step": float(step)})
                for key in required:
                    value = row.get(key)
                    if isinstance(value, int | float) and np.isfinite(float(value)):
                        output[key] = float(value)
    frame = pd.DataFrame(rows_by_step.values()).sort_values("step").reset_index(drop=True)
    if len(frame) < 200 or any(frame[key].notna().sum() < 200 for key in required):
        raise ValueError(f"Run {run_id} lacks a complete tail-visitation history.")
    return frame


def late_survival(run_id: str) -> np.ndarray:
    frame = load_history(run_id)
    final_step = int(frame["step"].max())
    selected = frame.loc[frame["step"] >= final_step - LATE_WINDOW + 1]
    if len(selected) < LATE_WINDOW - 1:
        raise ValueError(f"Run {run_id} does not cover the final {LATE_WINDOW} steps.")
    return np.asarray([100 * selected[key].mean() for _, key in SURVIVAL_METRICS])


def render_panel(slug: str, opd_run: str, dmpd_run: str) -> None:
    fig, ax = create_panel(STANDARD_PANEL_MARGINS)
    positions = np.asarray((0.0, 1.3, 2.6, 3.9, 5.2))
    for method, run_id in (("opd", opd_run), ("dmpd", dmpd_run)):
        style = METHODS[method]
        ax.plot(
            positions,
            late_survival(run_id),
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markerfacecolor="white" if method == "opd" else style["color"],
            markeredgewidth=1.1,
        )
    ax.set_xticks(positions, [label for label, _ in SURVIVAL_METRICS])
    ax.set_xlim(-0.25, 5.5)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_xlabel("Response-length event")
    ax.set_ylabel("Response tail / clip rate")
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7, alpha=0.85)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=4)
    save_figure(fig, OUTPUT_DIR, f"tail_visitation_{slug}")


def render_legend() -> None:
    handles = [
        Line2D(
            [],
            [],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markerfacecolor="white" if method == "opd" else style["color"],
            markeredgewidth=1.1,
        )
        for method, style in METHODS.items()
    ]
    fig = plt.figure(figsize=(4.2, 0.35))
    fig.legend(
        handles,
        [style["label"] for style in METHODS.values()],
        loc="center",
        frameon=False,
        ncol=2,
        handlelength=2.0,
        columnspacing=1.1,
    )
    save_figure(fig, OUTPUT_DIR, "tail_visitation_legend", export_mode="auxiliary", transparent=True)


def main() -> None:
    configure_style()
    render_panel("deepseek", OPD_RUN_ID, DMPD_RUN_ID)
    render_legend()


if __name__ == "__main__":
    main()
