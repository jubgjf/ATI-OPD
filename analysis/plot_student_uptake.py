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

"""Plot Student uptake immediately after a DMPD-Future teacher repair.

Each point is one training step.  The x-axis reconstructs first-16 K3 before
repair as ``gap_after + gap_reduction``; the y-axis is first-16 K3 after repair.
Points below y=x therefore indicate positive uptake.  The black arrowed path
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
from matplotlib import font_manager
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize

AFTER = "opd_intervention/repair_boundary/exit/active_gap_after_first16_mean"
REDUCTION = "opd_intervention/local_effect_v2/active_gap/reduction_first16_mean"
PHASE_WINDOW = 32
RUN_ID = "802c696019c1449aa3328bca0e1806eb"
BLACK = "#262626"
GREEN = "#009E73"
GRAY = "#7A8491"


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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


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
        columns=["step", "before", "after"],
    )
    frame = frame.loc[(frame["before"] >= 0) & (frame["after"] >= 0)].reset_index(drop=True)
    if len(frame) < 200:
        raise ValueError(f"Run {run_id} has only {len(frame)} valid uptake observations.")
    return frame


def phase_centers(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame.copy()
    selected["phase"] = ((selected["step"] - 1) // PHASE_WINDOW).astype(int)
    return selected.groupby("phase", sort=True)[["step", "before", "after"]].mean().reset_index(drop=True)


def draw_centroid_path(ax: mpl.axes.Axes, centers: pd.DataFrame, norm: Normalize) -> None:
    cmap = mpl.colormaps["viridis"]
    ax.plot(centers["before"], centers["after"], color=BLACK, linewidth=1.6, zorder=4)
    ax.scatter(
        centers["before"],
        centers["after"],
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
            xy=(centers.iloc[index + 1]["before"], centers.iloc[index + 1]["after"]),
            xytext=(centers.iloc[index]["before"], centers.iloc[index]["after"]),
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


def save(fig: mpl.figure.Figure, stem: str, *, transparent: bool = False) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUTPUT_DIR / f"{stem}.pdf",
        bbox_inches="tight",
        pad_inches=0.04,
        transparent=transparent,
    )
    # fig.savefig(
    #     OUTPUT_DIR / f"{stem}.png",
    #     dpi=350,
    #     bbox_inches="tight",
    #     pad_inches=0.04,
    #     transparent=transparent,
    # )
    plt.close(fig)


def render_panel(slug: str, run_id: str) -> None:
    frame = load_uptake(run_id)
    centers = phase_centers(frame)
    norm = Normalize(1, 217)
    cmap = mpl.colormaps["viridis"]
    maximum = max(float(frame["before"].max()), float(frame["after"].max()))
    robust_maximum = max(
        float(frame["before"].quantile(0.99)),
        float(frame["after"].quantile(0.99)),
    )
    upper = 1.08 * robust_maximum if maximum > 1.5 * robust_maximum else 1.05 * maximum
    in_view = (frame["before"] <= upper) & (frame["after"] <= upper)

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.fill_between([0, upper], [0, upper], [0, 0], color=GREEN, alpha=0.07, linewidth=0)
    ax.plot([0, upper], [0, upper], color=GRAY, linestyle="--", linewidth=1.1, zorder=1)
    ax.scatter(
        frame.loc[in_view, "before"],
        frame.loc[in_view, "after"],
        c=frame.loc[in_view, "step"],
        cmap=cmap,
        norm=norm,
        s=23,
        alpha=0.25,
        linewidths=0,
        zorder=2,
    )
    draw_centroid_path(ax, centers, norm)
    improved = float((frame["before"] > frame["after"]).mean())
    ax.text(
        0.96,
        0.94,
        f"Mean K3: {frame['before'].mean():.3f} → {frame['after'].mean():.3f}\nImproved: {improved:.1%}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
    )
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("K3 before")
    ax.set_title("K3 after")
    ax.grid(color="#D9DEE5", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=4)
    fig.tight_layout(pad=0.2)
    save(fig, f"student_uptake_first16_{slug}")


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
    save(fig, "student_uptake_training_step_colorbar", transparent=True)


def main() -> None:
    configure_style()
    render_panel("deepseek", RUN_ID)
    render_colorbar()


if __name__ == "__main__":
    main()
