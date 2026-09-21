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
from typing import Literal

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
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

LATE_WINDOW = 32
SURVIVAL_METRICS = (
    ("4K", "response_length/survival_gt_4k"),
    ("12K", "response_length/survival_gt_12k"),
    ("20K", "response_length/survival_gt_20k"),
    ("23K", "response_length/survival_gt_23k"),
    ("Max", "response_length/clip_ratio"),
)
OPD_RUN_ID = "9bbeb176889b42aa847665d03274fd68"
DMPD_RUN_ID = "802c696019c1449aa3328bca0e1806eb"
METHODS = {
    "opd": {"label": "OPD", "color": "#C7785E", "linestyle": "-", "marker": "^"},
    "dmpd": {"label": "DMPD", "color": "#6689A6", "linestyle": "-", "marker": "o"},
}
GRID_COLOR = "#E6E2DD"
PANEL_MARGINS = {"left": 0.22, "right": 0.97, "bottom": 0.19, "top": 0.88}


def configure_style() -> None:
    times_path = Path("/System/Library/Fonts/Supplemental/Times New Roman.ttf")
    if times_path.is_file():
        font_manager.fontManager.addfont(times_path)
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 15,
            "axes.titlesize": 15,
            "axes.labelsize": 15,
            "axes.linewidth": 1.1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


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


def save(
    fig: mpl.figure.Figure,
    stem: str,
    *,
    export_mode: Literal["panel", "legend"],
    transparent: bool = False,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict[str, object] = {"transparent": transparent}
    if export_mode == "legend":
        save_kwargs.update({"bbox_inches": "tight", "pad_inches": 0.04})
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", **save_kwargs)
    # fig.savefig(
    #     OUTPUT_DIR / f"{stem}.png",
    #     dpi=350,
    #     bbox_inches="tight",
    #     pad_inches=0.04,
    #     transparent=transparent,
    # )
    plt.close(fig)


def render_panel(slug: str, opd_run: str, dmpd_run: str) -> None:
    fig, ax = plt.subplots(figsize=(3, 3))
    positions = np.arange(len(SURVIVAL_METRICS))
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
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_xlabel("Response-length threshold")
    ax.set_title("Survival rate")
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7, alpha=0.85)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=4)
    fig.subplots_adjust(**PANEL_MARGINS)
    save(fig, f"tail_visitation_{slug}", export_mode="panel")


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
    save(fig, "tail_visitation_legend", export_mode="legend", transparent=True)


def main() -> None:
    configure_style()
    render_panel("deepseek", OPD_RUN_ID, DMPD_RUN_ID)
    render_legend()


if __name__ == "__main__":
    main()
