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

"""Render independent DeepSeek OPD-vs-DMPD training-diagnostic panels.

This one self-contained script creates the related gap, token-coverage,
gradient-norm, entropy, and response-clip figure series. Dense per-step curves
show faint raw observations plus a 7-step centered mean. Sparse diagnostics are
shown only at the steps where they were measured; no values are forward-filled.
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
except ModuleNotFoundError:  # Support ``python analysis/training_diagnostics.py``.
    from plot_style import (
        GRID_COLOR,
        STANDARD_PANEL_MARGINS,
        configure_style,
        create_panel,
        save_figure,
        set_panel_title,
    )

PAIR_SLUG = "deepseek"
OPD_RUN_ID = "9bbeb176889b42aa847665d03274fd68"
DMPD_RUN_ID = "802c696019c1449aa3328bca0e1806eb"
METHODS = {
    "opd": {"label": "OPD", "color": "#C7785E", "linestyle": "-", "marker": "^"},
    "dmpd": {"label": "DMPD", "color": "#6689A6", "linestyle": "-", "marker": "o"},
}
DIAGNOSTICS = {
    "gap_abs": {
        "metric": "opd/gap_abs/mean",
        "ylabel": "Absolute policy gap",
        "dense": True,
        "scale": 1.0,
        "percent": False,
    },
    "token_overlap": {
        "metric": "opd/overlap/top16_rate",
        "ylabel": "Top-16 token coverage",
        "dense": False,
        "scale": 100.0,
        "percent": True,
    },
    "grad_norm": {
        "metric": "actor/grad_norm",
        "ylabel": "Gradient norm",
        "dense": True,
        "scale": 1.0,
        "percent": False,
    },
    "entropy": {
        "metric": "distillation/entropy_student",
        "ylabel": "Student entropy",
        "dense": False,
        "scale": 1.0,
        "percent": False,
    },
    "response_clip_ratio": {
        "metric": "response_length/clip_ratio",
        "ylabel": "Response clip ratio",
        "dense": True,
        "scale": 100.0,
        "percent": True,
    },
}
SMOOTH_WINDOW = 7


def load_metric(run_id: str, metric: str, *, minimum_rows: int) -> pd.DataFrame:
    run_dir = ARTIFACTS_ROOT / run_id
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if metadata.get("entity") != "jubgjf" or metadata.get("project") != "DMPD":
        raise ValueError(f"Run {run_id} is not from jubgjf/DMPD.")
    values_by_step: dict[int, float] = {}
    for shard in sorted((run_dir / "history").glob("steps_*.jsonl.gz")):
        with gzip.open(shard, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                step = row.get("training/global_step")
                value = row.get(metric)
                if isinstance(step, int | float) and isinstance(value, int | float) and np.isfinite(float(value)):
                    values_by_step[int(step)] = float(value)
    frame = pd.DataFrame(sorted(values_by_step.items()), columns=["step", "value"])
    if len(frame) < minimum_rows:
        raise ValueError(f"Run {run_id} has only {len(frame)} finite observations for {metric}.")
    return frame


def time_average(frame: pd.DataFrame) -> float:
    steps = frame["step"].to_numpy(dtype=float)
    values = frame["value"].to_numpy(dtype=float)
    span = steps[-1] - steps[0]
    if len(frame) < 2 or span <= 0:
        raise ValueError("A gap AUC needs at least two strictly ordered observations.")
    return float(np.trapz(values, x=steps)) / span


def draw_series(
    ax: mpl.axes.Axes,
    frame: pd.DataFrame,
    *,
    method: str,
    dense: bool,
    scale: float,
) -> None:
    style = METHODS[method]
    values = scale * frame["value"]
    if dense:
        smooth = values.rolling(SMOOTH_WINDOW, center=True, min_periods=4).mean()
        ax.plot(frame["step"], values, color=style["color"], linewidth=0.75, alpha=0.24)
        ax.plot(
            frame["step"],
            smooth,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markerfacecolor="white" if method == "opd" else style["color"],
            markeredgewidth=0.8,
            markevery=20,
        )
    else:
        ax.plot(
            frame["step"],
            values,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markerfacecolor="white" if method == "opd" else style["color"],
            markeredgewidth=0.9,
        )


def render_panel(
    diagnostic: str,
    spec: dict[str, str | bool | float],
    slug: str,
    opd_run: str,
    dmpd_run: str,
) -> None:
    metric = str(spec["metric"])
    dense = bool(spec["dense"])
    scale = float(spec["scale"])
    minimum_rows = 190 if dense else 18
    frames = {
        "opd": load_metric(opd_run, metric, minimum_rows=minimum_rows),
        "dmpd": load_metric(dmpd_run, metric, minimum_rows=minimum_rows),
    }

    fig, ax = create_panel(STANDARD_PANEL_MARGINS)
    for method, frame in frames.items():
        draw_series(ax, frame, method=method, dense=dense, scale=scale)
    if diagnostic == "gap_abs":
        opd_auc = time_average(frames["opd"])
        dmpd_auc = time_average(frames["dmpd"])
        decrease = 100 * (opd_auc - dmpd_auc) / opd_auc
        ax.text(
            0.96,
            0.94,
            f"Gap AUC $\\downarrow$ {decrease:.1f}%",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=11,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
        )
    if bool(spec["percent"]):
        all_values = np.concatenate([100 * frame["value"].to_numpy() for frame in frames.values()])
        value_range = float(all_values.max() - all_values.min())
        pad = max(0.08, 0.12 * value_range)
        ax.set_ylim(max(0, all_values.min() - pad), min(100, all_values.max() + pad))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=1))
    elif diagnostic == "entropy":
        all_values = np.concatenate([frame["value"].to_numpy() for frame in frames.values()])
        value_range = float(all_values.max() - all_values.min())
        pad = max(0.01, 0.08 * value_range)
        ax.set_ylim(max(0, all_values.min() - pad), all_values.max() + pad)
    else:
        ax.set_ylim(bottom=0)
    ax.set_xlim(1, 217)
    ax.set_xticks([1, 50, 100, 150, 200])
    ax.set_xlabel("Training step")
    set_panel_title(ax, str(spec["ylabel"]))
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7, alpha=0.85)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=4)
    save_figure(fig, OUTPUT_DIR, f"{diagnostic}_{slug}")


def render_legend() -> None:
    handles = [
        Line2D(
            [],
            [],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markerfacecolor="white" if method == "opd" else style["color"],
            markeredgewidth=0.9,
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
    save_figure(
        fig,
        OUTPUT_DIR,
        "training_diagnostics_legend",
        export_mode="auxiliary",
        transparent=True,
    )


def main() -> None:
    configure_style()
    for diagnostic, spec in DIAGNOSTICS.items():
        render_panel(diagnostic, spec, PAIR_SLUG, OPD_RUN_ID, DMPD_RUN_ID)
    render_legend()


if __name__ == "__main__":
    main()
