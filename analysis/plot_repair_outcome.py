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

"""Plot repair-rate versus immediate online outcome for DMPD-Future.

This recreates the DeepSeek repair-to-outcome phase portrait. Raw points are
training batches, while the black arrowed path joins non-overlapping 32-step
centroids.
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

RATE = "opd_intervention/rate"
NET = "opd_intervention/outcome/net_accuracy_delta_all_samples"
PHASE_WINDOW = 32
RUN_ID = "802c696019c1449aa3328bca0e1806eb"
BLACK = "#42464B"
GRAY = "#92979D"
GRID_COLOR = "#E6E2DD"


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


def load_outcomes(run_id: str) -> pd.DataFrame:
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
                rate = row.get(RATE)
                net = row.get(NET)
                if (
                    isinstance(step, int | float)
                    and isinstance(rate, int | float)
                    and isinstance(net, int | float)
                    and np.isfinite(float(rate))
                    and np.isfinite(float(net))
                ):
                    rows_by_step[int(step)] = (100 * float(rate), 100 * float(net))
    frame = pd.DataFrame(
        [(step, *values) for step, values in sorted(rows_by_step.items())],
        columns=["step", "rate", "gain"],
    )
    if len(frame) < 200:
        raise ValueError(f"Run {run_id} has only {len(frame)} complete repair-outcome observations.")
    return frame


def phase_centers(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame.copy()
    selected["phase"] = ((selected["step"] - 1) // PHASE_WINDOW).astype(int)
    return selected.groupby("phase", sort=True)[["step", "rate", "gain"]].mean().reset_index(drop=True)


def training_step_cmap() -> mpl.colors.Colormap:
    """Use the same early-purple to late-yellow map as Student uptake."""
    return mpl.colormaps["viridis"]


def draw_centroid_path(ax: mpl.axes.Axes, centers: pd.DataFrame, norm: Normalize) -> None:
    cmap = training_step_cmap()
    ax.plot(centers["rate"], centers["gain"], color=BLACK, linewidth=1.6, zorder=4)
    ax.scatter(
        centers["rate"],
        centers["gain"],
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
            xy=(centers.iloc[index + 1]["rate"], centers.iloc[index + 1]["gain"]),
            xytext=(centers.iloc[index]["rate"], centers.iloc[index]["gain"]),
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
    frame = load_outcomes(run_id)
    centers = phase_centers(frame)
    norm = Normalize(1, 217)
    cmap = training_step_cmap()
    x_upper = 1.05 * float(frame["rate"].max())
    y_extent = 1.12 * float(frame["gain"].abs().max())

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.scatter(
        frame["rate"],
        frame["gain"],
        c=frame["step"],
        cmap=cmap,
        norm=norm,
        s=23,
        alpha=0.25,
        linewidths=0,
        zorder=2,
    )
    draw_centroid_path(ax, centers, norm)
    ax.axhline(0, color=GRAY, linestyle=":", linewidth=1.1)
    ax.set_xlim(0, x_upper)
    ax.set_ylim(-y_extent, y_extent)
    ax.set_xlabel("Intervention rate (%)")
    ax.set_title("Online gain (%)")
    ax.grid(color=GRID_COLOR, linewidth=0.7, alpha=0.85)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=4)
    fig.tight_layout(pad=0.2)
    save(fig, f"repair_outcome_{slug}")


def render_colorbar() -> None:
    fig, ax = plt.subplots(figsize=(0.30, 3))
    colorbar = ColorbarBase(
        ax,
        cmap=training_step_cmap(),
        norm=Normalize(1, 217),
        orientation="vertical",
    )
    colorbar.ax.set_title("Step", fontsize=13, pad=3)
    colorbar.set_ticks([1, 109, 217])
    colorbar.ax.tick_params(labelsize=13, pad=1)
    save(fig, "repair_outcome_training_step_colorbar", transparent=True)


def main() -> None:
    configure_style()
    render_panel("deepseek", RUN_ID)
    render_colorbar()


if __name__ == "__main__":
    main()
