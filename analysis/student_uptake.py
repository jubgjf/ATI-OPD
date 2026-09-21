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

"""Plot Student uptake immediately after a DMPD-Future teacher intervention.

Each point is one training step. The x-axis reconstructs first-16 K3 before
intervention as ``gap_after + gap_reduction``; the y-axis is first-16 K3 after
intervention. Points below y=x therefore indicate positive uptake. The black arrowed path
joins non-overlapping 32-step centroids, matching the earlier preferred style.
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
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize

try:
    from analysis.plot_style import SQUARE_PANEL_MARGINS, configure_style, create_panel, save_figure, set_panel_title
except ModuleNotFoundError:  # Support ``python analysis/student_uptake.py``.
    from plot_style import SQUARE_PANEL_MARGINS, configure_style, create_panel, save_figure, set_panel_title

# Legacy W&B schema keys retained for compatibility with downloaded artifacts.
AFTER = "opd_intervention/repair_boundary/exit/active_gap_after_first16_mean"
REDUCTION = "opd_intervention/local_effect_v2/active_gap/reduction_first16_mean"
PHASE_WINDOW = 32
RUN_ID = "802c696019c1449aa3328bca0e1806eb"
BLACK = "#262626"
GREEN = "#009E73"
GRAY = "#7A8491"


def load_uptake(run_id: str) -> pd.DataFrame:
    run_dir = ARTIFACTS_ROOT / run_id
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if metadata.get("entity") != "jubgjf" or metadata.get("project") != "DMPD":
        raise ValueError(f"Run {run_id} is not from jubgjf/DMPD.")
    rows_by_step: dict[int, tuple[float, float]] = {}
    for shard in sorted((run_dir / "history").glob("steps_*.jsonl.gz")):
        with gzip.open(shard, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                step = row.get("training/global_step")
                after = row.get(AFTER)
                reduction = row.get(REDUCTION)
                if (
                    isinstance(step, int | float)
                    and isinstance(after, int | float)
                    and isinstance(reduction, int | float)
                    and np.isfinite(float(after))
                    and np.isfinite(float(reduction))
                ):
                    rows_by_step[int(step)] = (float(after) + float(reduction), float(after))
    frame = pd.DataFrame(
        [(step, *values) for step, values in sorted(rows_by_step.items())],
        columns=["step", "before_intervention", "after_intervention"],
    )
    frame = frame.loc[(frame["before_intervention"] >= 0) & (frame["after_intervention"] >= 0)].reset_index(drop=True)
    if len(frame) < 200:
        raise ValueError(f"Run {run_id} has only {len(frame)} valid uptake observations.")
    return frame


def phase_centers(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame.copy()
    selected["phase"] = ((selected["step"] - 1) // PHASE_WINDOW).astype(int)
    return (
        selected.groupby("phase", sort=True)[["step", "before_intervention", "after_intervention"]]
        .mean()
        .reset_index(drop=True)
    )


def draw_centroid_path(ax: mpl.axes.Axes, centers: pd.DataFrame, norm: Normalize) -> None:
    cmap = mpl.colormaps["viridis"]
    ax.plot(
        centers["before_intervention"],
        centers["after_intervention"],
        color=BLACK,
        linewidth=1.6,
        zorder=4,
    )
    ax.scatter(
        centers["before_intervention"],
        centers["after_intervention"],
        c=centers["step"],
        cmap=cmap,
        norm=norm,
        s=43,
        edgecolors="white",
        linewidths=0.75,
        zorder=5,
    )
    for index in range(len(centers) - 1):
        ax.annotate(
            "",
            xy=(
                centers.iloc[index + 1]["before_intervention"],
                centers.iloc[index + 1]["after_intervention"],
            ),
            xytext=(centers.iloc[index]["before_intervention"], centers.iloc[index]["after_intervention"]),
            arrowprops={
                "arrowstyle": "->",
                "color": BLACK,
                "lw": 1.0,
                "mutation_scale": 8,
                "shrinkA": 4,
                "shrinkB": 4,
            },
            zorder=5,
        )


def render_panel(slug: str, run_id: str) -> None:
    frame = load_uptake(run_id)
    centers = phase_centers(frame)
    norm = Normalize(1, 217)
    cmap = mpl.colormaps["viridis"]
    maximum = max(float(frame["before_intervention"].max()), float(frame["after_intervention"].max()))
    robust_maximum = max(
        float(frame["before_intervention"].quantile(0.99)),
        float(frame["after_intervention"].quantile(0.99)),
    )
    upper = 1.08 * robust_maximum if maximum > 1.5 * robust_maximum else 1.05 * maximum
    in_view = (frame["before_intervention"] <= upper) & (frame["after_intervention"] <= upper)

    fig, ax = create_panel(SQUARE_PANEL_MARGINS)
    ax.fill_between([0, upper], [0, upper], [0, 0], color=GREEN, alpha=0.07, linewidth=0)
    ax.plot([0, upper], [0, upper], color=GRAY, linestyle="--", linewidth=1.1, zorder=1)
    ax.scatter(
        frame.loc[in_view, "before_intervention"],
        frame.loc[in_view, "after_intervention"],
        c=frame.loc[in_view, "step"],
        cmap=cmap,
        norm=norm,
        s=23,
        alpha=0.25,
        linewidths=0,
        zorder=2,
    )
    draw_centroid_path(ax, centers, norm)
    improved = float((frame["before_intervention"] > frame["after_intervention"]).mean())
    ax.text(
        0.96,
        0.94,
        f"Mean K3: {frame['before_intervention'].mean():.3f} $\\rightarrow$ "
        f"{frame['after_intervention'].mean():.3f}\nImproved: {improved:.1%}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
    )
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("K3 before intervention")
    set_panel_title(ax, "K3 after intervention")
    ax.grid(color="#D9DEE5", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=4)
    save_figure(fig, OUTPUT_DIR, f"student_uptake_first16_{slug}")


def render_colorbar() -> None:
    fig, ax = plt.subplots(figsize=(0.30, 3))
    colorbar = ColorbarBase(
        ax,
        cmap=mpl.colormaps["viridis"],
        norm=Normalize(1, 217),
        orientation="vertical",
    )
    colorbar.ax.set_title("Step", fontsize=13, pad=3)
    colorbar.set_ticks([1, 109, 217])
    colorbar.ax.tick_params(labelsize=13, pad=1)
    save_figure(
        fig,
        OUTPUT_DIR,
        "student_uptake_training_step_colorbar",
        export_mode="auxiliary",
        transparent=True,
    )


def main() -> None:
    configure_style()
    render_panel("deepseek", RUN_ID)
    render_colorbar()


if __name__ == "__main__":
    main()
