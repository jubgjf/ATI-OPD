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

The input is the DeepSeek DMPD-Future main run.  ``Raw K3`` is reconstructed
from the logged repaired value and local reduction, while ``Repaired K3`` is
the first-16 K3 observed when the Student resumes generation after the Teacher
span.  The second panel normalizes repaired exit K3 by the matched raw K3 and
the intervention threshold, giving two dimensionless references at one.
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
from matplotlib.lines import Line2D

RUN_ID = "802c696019c1449aa3328bca0e1806eb"
AFTER = "opd_intervention/repair_boundary/exit/active_gap_after_first16_mean"
REDUCTION = "opd_intervention/local_effect_v2/active_gap/reduction_first16_mean"
SMOOTHING_WINDOW = 7
K3_THRESHOLD = 0.15
RAW_COLOR = "#B97964"
REPAIRED_COLOR = "#6689A6"
THRESHOLD_RATIO_COLOR = "#7C9E87"
REFERENCE_COLOR = "#626970"
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
            "legend.fontsize": 13,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


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
                repaired = row.get(AFTER)
                reduction = row.get(REDUCTION)
                if (
                    isinstance(step, int | float)
                    and isinstance(repaired, int | float)
                    and isinstance(reduction, int | float)
                    and np.isfinite(float(repaired))
                    and np.isfinite(float(reduction))
                ):
                    rows_by_step[int(step)] = (float(repaired) + float(reduction), float(repaired))

    frame = pd.DataFrame(
        [(step, *values) for step, values in sorted(rows_by_step.items())],
        columns=["step", "raw_k3", "repaired_k3"],
    )
    frame = frame.loc[(frame["raw_k3"] >= 0) & (frame["repaired_k3"] >= 0)].reset_index(drop=True)
    if len(frame) != 217 or frame["step"].min() != 1 or frame["step"].max() != 217:
        raise ValueError(f"Expected steps 1--217 for run {RUN_ID}, found {len(frame)} rows.")
    if not bool((frame["repaired_k3"] > 0).all()):
        raise ValueError("The repaired exit-gap series contains a non-positive value.")
    return frame


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


def style_axes(ax: mpl.axes.Axes) -> None:
    ax.set_xlim(1, 217)
    ax.set_xticks([1, 50, 100, 150, 200])
    ax.set_xlabel("Training step")
    ax.grid(color=GRID_COLOR, linewidth=0.7, alpha=0.85)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=4)


def render_local_reduction(frame: pd.DataFrame) -> None:
    smoothed = (
        frame[["raw_k3", "repaired_k3"]]
        .rolling(
            SMOOTHING_WINDOW,
            center=True,
            min_periods=1,
        )
        .mean()
    )
    marker_steps = list(range(0, len(frame), 24))

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.axhline(K3_THRESHOLD, color=REFERENCE_COLOR, linestyle="--", linewidth=1.2, zorder=1)
    ax.plot(
        frame["step"],
        smoothed["raw_k3"],
        color=RAW_COLOR,
        linewidth=2.0,
        marker="^",
        markersize=4.2,
        markevery=marker_steps,
    )
    ax.plot(
        frame["step"],
        smoothed["repaired_k3"],
        color=REPAIRED_COLOR,
        linewidth=2.0,
        marker="o",
        markersize=4.2,
        markevery=marker_steps,
    )
    ax.set_ylim(0, 1.08 * float(smoothed["raw_k3"].max()))
    ax.set_title("Local K3 reduction")
    style_axes(ax)
    fig.tight_layout(pad=0.2)
    save(fig, "local_gap_reduction_deepseek")


def render_normalized_exit_gap(frame: pd.DataFrame) -> None:
    smoothed = (
        frame[["raw_k3", "repaired_k3"]]
        .rolling(
            SMOOTHING_WINDOW,
            center=True,
            min_periods=1,
        )
        .mean()
    )
    exit_over_raw = smoothed["repaired_k3"] / smoothed["raw_k3"]
    exit_over_threshold = smoothed["repaired_k3"] / K3_THRESHOLD
    marker_steps = list(range(0, len(frame), 24))

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.axhspan(0, 1, color=THRESHOLD_RATIO_COLOR, alpha=0.08, linewidth=0, zorder=0)
    ax.axhline(1, color=REFERENCE_COLOR, linestyle="--", linewidth=1.2, zorder=1)
    ax.plot(
        frame["step"],
        exit_over_raw,
        color=REPAIRED_COLOR,
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
    ax.set_title("Normalized handoff gap")
    style_axes(ax)
    fig.tight_layout(pad=0.2)
    save(fig, "exit_gap_deepseek")


def render_legends() -> None:
    local_handles = [
        Line2D([], [], color=RAW_COLOR, marker="^", linewidth=2.0, markersize=5),
        Line2D([], [], color=REPAIRED_COLOR, marker="o", linewidth=2.0, markersize=5),
        Line2D([], [], color=REFERENCE_COLOR, linestyle="--", linewidth=1.2),
    ]
    fig = plt.figure(figsize=(4.0, 0.35))
    fig.legend(
        local_handles,
        ["Raw K3", "Repaired K3", r"Trigger $\tau$"],
        loc="center",
        frameon=False,
        ncol=3,
        handlelength=1.5,
        columnspacing=0.9,
    )
    save(fig, "local_gap_reduction_legend", transparent=True)

    normalized_handles = [
        Line2D([], [], color=REPAIRED_COLOR, marker="o", linewidth=2.0, markersize=5),
        Line2D([], [], color=THRESHOLD_RATIO_COLOR, marker="s", linewidth=2.0, markersize=5),
    ]
    fig = plt.figure(figsize=(2.9, 0.35))
    fig.legend(
        normalized_handles,
        ["Exit / raw", r"Exit / $\tau$"],
        loc="center",
        frameon=False,
        ncol=2,
        handlelength=1.5,
        columnspacing=0.9,
    )
    save(fig, "normalized_exit_gap_legend", transparent=True)


def main() -> None:
    configure_style()
    frame = load_gap_series()
    render_local_reduction(frame)
    render_normalized_exit_gap(frame)
    render_legends()


if __name__ == "__main__":
    main()
