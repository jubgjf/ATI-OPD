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

"""Shared visual style and fixed-canvas export helpers for paper figures."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

PANEL_FIGSIZE = (3.0, 3.0)
PANEL_DPI = 350

# Fixed axes positions prevent independently rendered panels from changing
# width when tick labels have different extents.
STANDARD_PANEL_MARGINS = {
    "left": 0.29,
    "right": 0.97,
    "bottom": 0.19,
    "top": 0.88,
}
SQUARE_PANEL_MARGINS = {
    "left": 0.29,
    "right": 0.98,
    "bottom": 0.19,
    "top": 0.88,
}
FIGURE6_REGULAR_PANEL_MARGINS = {
    "left": 0.29,
    "right": 0.97,
    "bottom": 0.25,
    "top": 0.88,
}
DENSE_X_PANEL_MARGINS = {
    "left": 0.26,
    "right": 0.99,
    "bottom": 0.36,
    "top": 0.88,
}

GRID_COLOR = "#E6E2DD"
ExportMode = Literal["panel", "auxiliary"]


def configure_style() -> None:
    """Apply the common STIX-based paper style."""
    stix_paths = (
        Path("/System/Library/Fonts/Supplemental/STIXTwoText.ttf"),
        Path("/System/Library/Fonts/Supplemental/STIXTwoText-Italic.ttf"),
        Path("/System/Library/Fonts/Supplemental/STIXTwoMath.otf"),
    )
    for stix_path in stix_paths:
        if stix_path.is_file():
            font_manager.fontManager.addfont(stix_path)
    mpl.rcParams.update(
        {
            "font.family": "STIX Two Text",
            "mathtext.fontset": "custom",
            "mathtext.rm": "STIX Two Math",
            "mathtext.it": "STIX Two Math",
            "mathtext.bf": "STIX Two Math",
            "mathtext.bfit": "STIX Two Math",
            "mathtext.cal": "STIX Two Math",
            "mathtext.sf": "STIX Two Math",
            "mathtext.tt": "STIX Two Math",
            "mathtext.fallback": None,
            "font.size": 15,
            "axes.titlesize": 15,
            "axes.labelsize": 15,
            "axes.linewidth": 1.1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "legend.fontsize": 13,
            "lines.linewidth": 2.0,
            "lines.markersize": 5.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def create_panel(
    margins: Mapping[str, float] = STANDARD_PANEL_MARGINS,
) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """Create one fixed 3-by-3-inch panel with an explicit axes rectangle."""
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    fig.subplots_adjust(**margins)
    return fig, ax


def panel_layout_overflow(fig: mpl.figure.Figure, *, tolerance_points: float = 0.25) -> tuple[str, ...]:
    """Return canvas sides crossed by artists participating in panel layout."""
    if tolerance_points < 0:
        raise ValueError(f"tolerance_points must be non-negative, got {tolerance_points}.")
    fig.canvas.draw()
    layout = fig.get_tightbbox(fig.canvas.get_renderer())
    canvas = fig.bbox_inches
    tolerance = tolerance_points / 72.0
    sides: list[str] = []
    if layout.x0 < canvas.x0 - tolerance:
        sides.append("left")
    if layout.x1 > canvas.x1 + tolerance:
        sides.append("right")
    if layout.y0 < canvas.y0 - tolerance:
        sides.append("bottom")
    if layout.y1 > canvas.y1 + tolerance:
        sides.append("top")
    return tuple(sides)


def save_figure(
    fig: mpl.figure.Figure,
    output_dir: Path,
    stem: str,
    *,
    export_mode: ExportMode = "panel",
    transparent: bool = False,
) -> None:
    """Save a fixed-canvas panel or a tightly cropped auxiliary figure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict[str, object] = {"transparent": transparent}
    if export_mode == "auxiliary":
        save_kwargs.update({"bbox_inches": "tight", "pad_inches": 0.04})
    else:
        overflow = panel_layout_overflow(fig)
        if overflow:
            raise ValueError(f"Panel layout extends beyond the fixed canvas: {', '.join(overflow)}")
    fig.savefig(output_dir / f"{stem}.pdf", **save_kwargs)
    # fig.savefig(output_dir / f"{stem}.png", dpi=PANEL_DPI, **save_kwargs)
    plt.close(fig)
