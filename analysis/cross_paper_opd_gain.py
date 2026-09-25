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

"""Plot cross-paper accuracy gains over each method's own vanilla OPD baseline."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "analysis" / "figures"
CACHE_ROOT = REPO_ROOT / ".analysis_cache"
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

try:
    from analysis.plot_style import (
        GRID_COLOR,
        STANDARD_PANEL_MARGINS,
        configure_style,
        save_figure,
        set_panel_title,
    )
except ModuleNotFoundError:  # Support ``python analysis/cross_paper_opd_gain.py``.
    from plot_style import (
        GRID_COLOR,
        STANDARD_PANEL_MARGINS,
        configure_style,
        save_figure,
        set_panel_title,
    )

DAGGER = "\N{DAGGER}"
BENCHMARKS = ("AIME24", "AIME25", "AMC23")
MODEL_GROUPS = ("JustRL-DeepSeek", "JustRL-Nemotron")
GROUP_CENTERS = np.asarray((0.0, 1.5))
BAR_CENTER_STEP = 0.21
BAR_WIDTH = 0.17
Y_LIMITS = (-0.58, 1.68)
Y_TICKS = (0.0, 0.5, 1.0, 1.5)
# Small horizontal offsets separate close labels while keeping them over their bars.
VALUE_LABEL_X_OFFSETS = (
    (0.0, -0.04, 0.0, 0.07, 0.0, 0.0),  # JustRL-DeepSeek
    (0.0, -0.03, -0.04, -0.03, -0.06, 0.06),  # JustRL-Nemotron
)
VALUE_LABEL_OFFSETS = (
    (0.04, 0.04, 0.13, 0.04, 0.04, 0.04),  # JustRL-DeepSeek
    (0.04, 0.04, 0.03, 0.08, 0.04, 0.04),  # JustRL-Nemotron
)
TRUNCATED_BAR_ASYMPTOTE = 0.47
TRUNCATED_BAR_HALF_SCALE = 2.5
TRUNCATED_LABEL_GAP = 0.05
MAIN_FIGSIZE = (4.0, 3.0)
MODEL_TICK_FONT_SIZE = 13
LEGEND_COLUMN_LEFTS = (0.01, 0.34, 0.67)
LEGEND_ROW_CENTERS = (0.74, 0.26)
LEGEND_SWATCH_WIDTH = 0.05
LEGEND_SWATCH_HEIGHT = 0.30
LEGEND_TEXT_GAP = 0.015
LEGEND_FONT_SIZE = 13

MAIN_OUTPUT_STEM = "cross_paper_opd_gain"
LEGEND_OUTPUT_STEM = "cross_paper_opd_gain_legend"


@dataclass(frozen=True)
class MethodSpec:
    """Display metadata for one compared distillation method."""

    key: str
    label: str
    color: str
    published: bool

    @property
    def legend_label(self) -> str:
        return f"{self.label}{DAGGER if self.published else ''}"


METHODS = (
    MethodSpec("esr", "ESR", "#A19D98", False),
    MethodSpec("popd", "POPD", "#B97964", True),
    MethodSpec("topd_050", "TOPD-0.50", "#8E859D", True),
    MethodSpec("topd_025", "TOPD-0.25", "#C3A46E", True),
    MethodSpec("topd_010", "TOPD-0.10", "#7A9E85", True),
    MethodSpec("ati_opd", "ATI-OPD", "#6689A6", False),
)

# ESR was logged as topd--horizon_100 in W&B; both runs finished at step 217.
# DeepSeek: 48b5708e981c4d17aacc398ffa8d043c, OPD: 9bbeb176889b42aa847665d03274fd68.
# Nemotron: 8ea23b41d0ef497abe3ead9d05413655, OPD: 040b92fbe33a41e6be75ffcaf8f07f27.
ESR_STEP217_ACCURACY = {
    "JustRL-DeepSeek": (0.3458333333333333, 0.28541666666666665, 0.78125),
    "JustRL-Nemotron": (0.6416666666666667, 0.5625, 0.946875),
}
VANILLA_OPD_STEP217_ACCURACY = {
    "JustRL-DeepSeek": (0.4625, 0.33958333333333335, 0.8703125),
    "JustRL-Nemotron": (0.6708333333333333, 0.59375, 0.95),
}


def _gain_from_accuracy(
    trained: tuple[float, float, float], baseline: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        100 * (trained[0] - baseline[0]),
        100 * (trained[1] - baseline[1]),
        100 * (trained[2] - baseline[2]),
    )


# Absolute accuracy differences on the three common benchmarks. Published
# POPD/TOPD values come from the appendix source; ESR and ATI-OPD use this
# paper's own matched benchmark runs.
BENCHMARK_DELTAS: dict[str, dict[str, tuple[float, float, float]]] = {
    "JustRL-DeepSeek": {
        "esr": _gain_from_accuracy(
            ESR_STEP217_ACCURACY["JustRL-DeepSeek"], VANILLA_OPD_STEP217_ACCURACY["JustRL-DeepSeek"]
        ),
        "popd": (1.70, 0.50, -1.40),
        "topd_050": (1.00, 0.80, -0.70),
        "topd_025": (2.00, -0.10, -1.20),
        "topd_010": (1.40, -1.80, -4.10),
        "ati_opd": (2.92, -0.42, 1.56),
    },
    "JustRL-Nemotron": {
        "esr": _gain_from_accuracy(
            ESR_STEP217_ACCURACY["JustRL-Nemotron"], VANILLA_OPD_STEP217_ACCURACY["JustRL-Nemotron"]
        ),
        "popd": (1.30, 0.40, -0.20),
        "topd_050": (1.30, 0.70, 0.30),
        "topd_025": (1.40, 0.60, 0.70),
        "topd_010": (2.00, 1.10, 0.30),
        "ati_opd": (1.67, 1.04, 0.63),
    },
}


def average_gains(
    deltas: Mapping[str, Mapping[str, Sequence[float]]] = BENCHMARK_DELTAS,
) -> np.ndarray:
    """Return model-by-method equal-weight means over the three benchmarks."""
    if set(deltas) != set(MODEL_GROUPS):
        raise ValueError(f"Expected model groups {MODEL_GROUPS!r}, got {tuple(deltas)!r}.")
    expected_method_keys = {method.key for method in METHODS}
    output = np.empty((len(MODEL_GROUPS), len(METHODS)), dtype=float)
    for model_index, model in enumerate(MODEL_GROUPS):
        model_deltas = deltas[model]
        if set(model_deltas) != expected_method_keys:
            raise ValueError(
                f"{model} method keys differ from the expected set: "
                f"expected {sorted(expected_method_keys)!r}, got {sorted(model_deltas)!r}."
            )
        for method_index, method in enumerate(METHODS):
            values = np.asarray(model_deltas[method.key], dtype=float)
            if values.shape != (len(BENCHMARKS),) or not np.isfinite(values).all():
                raise ValueError(
                    f"{model}/{method.label} must contain {len(BENCHMARKS)} finite benchmark deltas, got {values!r}."
                )
            output[model_index, method_index] = float(values.mean())
    return output


def bar_positions() -> np.ndarray:
    """Return model-by-method bar centers with a visible gap between groups."""
    offsets = (np.arange(len(METHODS), dtype=float) - (len(METHODS) - 1) / 2.0) * BAR_CENTER_STEP
    return GROUP_CENTERS[:, None] + offsets[None, :]


def display_gains(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compress negative bars monotonically; their exact values remain in labels."""
    values = np.asarray(values, dtype=float)
    truncated = values < 0
    displayed = values.copy()
    magnitudes = -values[truncated]
    displayed[truncated] = -(TRUNCATED_BAR_ASYMPTOTE * magnitudes / (TRUNCATED_BAR_HALF_SCALE + magnitudes))
    return displayed, truncated


def _draw_bar_break(ax: mpl.axes.Axes, position: float, displayed_value: float) -> None:
    """Draw a conventional double slash across a direction-only truncated bar."""
    center_y = displayed_value / 2.0
    half_width = BAR_WIDTH * 0.54
    half_rise = 0.018
    for offset in (-0.027, 0.027):
        (line,) = ax.plot(
            (position - half_width, position + half_width),
            (center_y + offset - half_rise, center_y + offset + half_rise),
            color="white",
            linewidth=2.2,
            solid_capstyle="butt",
            zorder=5,
        )
        line.set_gid("truncated-bar-break")


def create_gain_figure() -> mpl.figure.Figure:
    """Create the 4:3 grouped gain panel."""
    values = average_gains()
    displayed_values, truncated = display_gains(values)
    positions = bar_positions()
    fig, ax = plt.subplots(figsize=MAIN_FIGSIZE)
    # Preserve the shared panels' physical left margin on the wider canvas.
    fig.subplots_adjust(**{**STANDARD_PANEL_MARGINS, "left": 0.165, "right": 0.975})

    for model_index, _model in enumerate(MODEL_GROUPS):
        for method_index, method in enumerate(METHODS):
            value = values[model_index, method_index]
            displayed_value = displayed_values[model_index, method_index]
            position = positions[model_index, method_index]
            ax.bar(
                position,
                displayed_value,
                width=BAR_WIDTH,
                color=method.color,
                edgecolor=method.color,
                linewidth=1.0,
                zorder=3,
            )
            if truncated[model_index, method_index]:
                _draw_bar_break(ax, position, displayed_value)
                # The signed annotation, rather than the compressed bar length,
                # is authoritative for a truncated outlier.
                label_y = displayed_value - TRUNCATED_LABEL_GAP
                label_x = position
                vertical_alignment = "top"
            else:
                label_y = value + VALUE_LABEL_OFFSETS[model_index][method_index]
                label_x = position + VALUE_LABEL_X_OFFSETS[model_index][method_index]
                vertical_alignment = "bottom"
            ax.text(
                label_x,
                label_y,
                f"{value:+.2f}",
                ha="center",
                va=vertical_alignment,
                fontsize=9,
                color="#333333",
                zorder=4,
            )

    ax.set_xlim(-0.72, 2.22)
    ax.set_ylim(*Y_LIMITS)
    ax.set_xticks(GROUP_CENTERS, MODEL_GROUPS)
    ax.set_yticks(Y_TICKS)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value:g}%"))
    set_panel_title(ax, "Avg. accuracy gain", pad=4)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7, alpha=0.85)
    ax.axhline(0.0, color="#777777", linewidth=1.0, zorder=2)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", bottom=False, pad=5, labelsize=MODEL_TICK_FONT_SIZE)
    ax.tick_params(axis="y", direction="out", length=4)
    return fig


def create_legend_figure() -> mpl.figure.Figure:
    """Create a fixed three-column legend with vertically aligned swatches."""
    fig = plt.figure(figsize=(4.0, 0.65))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_axis_off()
    for row_index, row_y in enumerate(LEGEND_ROW_CENTERS):
        for column_index, left_x in enumerate(LEGEND_COLUMN_LEFTS):
            method = METHODS[row_index * len(LEGEND_COLUMN_LEFTS) + column_index]
            ax.add_patch(
                Rectangle(
                    (left_x, row_y - LEGEND_SWATCH_HEIGHT / 2),
                    LEGEND_SWATCH_WIDTH,
                    LEGEND_SWATCH_HEIGHT,
                    facecolor=method.color,
                    edgecolor=method.color,
                    linewidth=1.0,
                    transform=ax.transAxes,
                )
            )
            ax.text(
                left_x + LEGEND_SWATCH_WIDTH + LEGEND_TEXT_GAP,
                row_y,
                method.legend_label,
                ha="left",
                va="center",
                fontsize=LEGEND_FONT_SIZE,
                transform=ax.transAxes,
            )
    return fig


def main() -> None:
    configure_style()
    save_figure(create_gain_figure(), DEFAULT_OUTPUT_DIR, MAIN_OUTPUT_STEM)
    save_figure(
        create_legend_figure(),
        DEFAULT_OUTPUT_DIR,
        LEGEND_OUTPUT_STEM,
        export_mode="auxiliary",
        transparent=True,
    )


if __name__ == "__main__":
    main()
