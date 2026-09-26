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

"""Render compact vanilla-OPD collapse diagnostics for the motivated figure.

The three 3:2 panels show gradient norm, top-16 residual-bucket Student
entropy, and response-length clipping rate. Each curve uses the same soft
blue--purple--red palette as ``long_sequence_divergence_slices.py`` and maps
its local height to color: lower values are blue and higher values are red.
Only real observations from the raw ``training-diagnostics`` shards are used.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = REPO_ROOT / ".analysis_cache"
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.ticker import MaxNLocator, PercentFormatter

try:
    from analysis.long_sequence_divergence_slices import (
        CURVE_SMOOTHING_PASSES,
        EXPECTED_TRAINING_STEPS,
        RUN_ID,
        SLICE_COLORMAP_COLORS,
        diagnostic_shard_paths,
        validate_canonical_run,
    )
    from analysis.long_sequence_divergence_slices import (
        _smooth_segment as smooth_curve_segment,
    )
    from analysis.plot_style import GRID_COLOR, PANEL_DPI, configure_style, panel_layout_overflow
except ModuleNotFoundError:  # Support ``python analysis/vanilla_opd_collapse_consequences.py``.
    from long_sequence_divergence_slices import (
        CURVE_SMOOTHING_PASSES,
        EXPECTED_TRAINING_STEPS,
        RUN_ID,
        SLICE_COLORMAP_COLORS,
        diagnostic_shard_paths,
        validate_canonical_run,
    )
    from long_sequence_divergence_slices import (
        _smooth_segment as smooth_curve_segment,
    )
    from plot_style import GRID_COLOR, PANEL_DPI, configure_style, panel_layout_overflow

DEFAULT_ARTIFACTS_ROOT = REPO_ROOT / "analysis_artifacts" / "jubgjf" / "DMPD" / "runs"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "analysis" / "figures"
FIGSIZE = (3.0, 2.0)
PANEL_MARGINS = {"left": 0.18, "right": 0.985, "bottom": 0.18, "top": 0.97}
SMOOTHING_WINDOW = 7


@dataclass(frozen=True)
class MetricSpec:
    metric: str
    stem: str
    expected_steps: tuple[int, ...]
    smoothing_window: int | None
    scale: float = 1.0
    percent: bool = False
    zero_baseline: bool = False


METRICS: tuple[MetricSpec, ...] = (
    MetricSpec(
        metric="actor/grad_norm",
        stem="vanilla_opd_grad_norm",
        expected_steps=tuple(range(1, EXPECTED_TRAINING_STEPS + 1)),
        smoothing_window=SMOOTHING_WINDOW,
        zero_baseline=True,
    ),
    MetricSpec(
        metric="distillation/entropy_student",
        stem="vanilla_opd_student_entropy",
        expected_steps=tuple(range(10, 211, 10)),
        smoothing_window=None,
    ),
    MetricSpec(
        metric="response_length/clip_ratio",
        stem="vanilla_opd_clip_ratio",
        expected_steps=tuple(range(1, EXPECTED_TRAINING_STEPS + 1)),
        smoothing_window=SMOOTHING_WINDOW,
        scale=100.0,
        percent=True,
        zero_baseline=True,
    ),
)


def validate_metric_rows(
    rows: pd.DataFrame,
    specs: Sequence[MetricSpec] = METRICS,
) -> dict[str, pd.DataFrame]:
    """Validate raw diagnostics and return one exact-step frame per metric."""
    required_columns = ("global_step", "metric_key", "value")
    missing_columns = set(required_columns).difference(rows.columns)
    if missing_columns:
        raise ValueError(f"Collapse diagnostics are missing columns: {sorted(missing_columns)}.")

    frames: dict[str, pd.DataFrame] = {}
    for spec in specs:
        selected = rows.loc[rows["metric_key"] == spec.metric, list(required_columns)].copy()
        if selected.empty:
            raise ValueError(f"No raw diagnostics found for {spec.metric}.")
        selected["global_step"] = pd.to_numeric(selected["global_step"], errors="coerce")
        selected["value"] = pd.to_numeric(selected["value"], errors="coerce")
        if not np.isfinite(selected[["global_step", "value"]].to_numpy(dtype=float)).all():
            raise ValueError(f"{spec.metric} contains a non-finite step or value.")
        if not np.equal(selected["global_step"], np.floor(selected["global_step"])).all():
            raise ValueError(f"{spec.metric} contains a non-integral training step.")
        selected["global_step"] = selected["global_step"].astype(int)
        if selected.duplicated("global_step").any():
            duplicated_steps = selected.loc[selected.duplicated("global_step", keep=False), "global_step"].tolist()
            raise ValueError(f"{spec.metric} contains duplicate steps: {duplicated_steps[:8]}.")
        selected = selected.sort_values("global_step").reset_index(drop=True)
        observed_steps = tuple(selected["global_step"].tolist())
        if observed_steps != spec.expected_steps:
            expected = set(spec.expected_steps)
            observed = set(observed_steps)
            raise ValueError(
                f"Unexpected sampling steps for {spec.metric}: "
                f"missing={sorted(expected - observed)[:8]}, extra={sorted(observed - expected)[:8]}."
            )
        if bool((selected["value"] < 0).any()):
            raise ValueError(f"{spec.metric} contains a negative value.")
        frames[spec.metric] = selected.rename(columns={"global_step": "step"})[["step", "value"]]
    return frames


def load_metrics(run_dir: Path) -> dict[str, pd.DataFrame]:
    """Load the three metrics from the validated canonical vanilla-OPD run."""
    validate_canonical_run(run_dir)
    artifact_root = run_dir / "logged_artifacts" / "training-diagnostics"
    metric_names = {spec.metric for spec in METRICS}
    selected_frames: list[pd.DataFrame] = []
    for path in diagnostic_shard_paths(artifact_root):
        rows = pd.read_parquet(
            path,
            columns=["logical_run_id", "event_kind", "global_step", "metric_key", "value"],
        )
        selected = rows.loc[rows["metric_key"].isin(metric_names)].copy()
        if selected.empty:
            continue
        if set(selected["logical_run_id"].dropna().astype(str)) != {RUN_ID}:
            raise ValueError(f"Diagnostics shard {path} contains metrics from another run.")
        if set(selected["event_kind"].dropna().astype(str)) != {"train"}:
            raise ValueError(f"Diagnostics shard {path} contains non-training collapse diagnostics.")
        selected_frames.append(selected[["global_step", "metric_key", "value"]])
    if not selected_frames:
        raise ValueError(f"No collapse diagnostics found for vanilla-OPD run {RUN_ID}.")
    return validate_metric_rows(pd.concat(selected_frames, ignore_index=True))


def metric_colormap() -> mpl.colors.Colormap:
    """Return the same soft blue--purple--red scale as the slice figure."""
    return mpl.colors.LinearSegmentedColormap.from_list("vanilla_opd_collapse", SLICE_COLORMAP_COLORS)


def prepare_curve(frame: pd.DataFrame, spec: MetricSpec) -> tuple[np.ndarray, np.ndarray]:
    steps = frame["step"].to_numpy(dtype=float)
    values = spec.scale * frame["value"].to_numpy(dtype=float)
    if spec.smoothing_window is not None:
        values = (
            pd.Series(values)
            .rolling(spec.smoothing_window, center=True, min_periods=1)
            .mean()
            .to_numpy(dtype=float)
        )
    if len(steps) < 2 or not np.isfinite(steps).all() or not np.isfinite(values).all():
        raise ValueError(f"{spec.metric} needs at least two finite points for a line plot.")
    if not np.all(np.diff(steps) > 0):
        raise ValueError(f"{spec.metric} steps must be strictly increasing.")
    return smooth_curve_segment(steps, values, passes=CURVE_SMOOTHING_PASSES)


def add_value_colored_line(
    ax: mpl.axes.Axes,
    steps: np.ndarray,
    values: np.ndarray,
) -> LineCollection:
    """Draw one line whose local height selects blue-to-red segment colors."""
    points = np.column_stack((steps, values)).reshape(-1, 1, 2)
    segments = np.concatenate((points[:-1], points[1:]), axis=1)
    segment_values = 0.5 * (values[:-1] + values[1:])
    value_min = float(values.min())
    value_max = float(values.max())
    if value_max <= value_min:
        raise ValueError("A value-colored curve needs a non-zero value range.")
    collection = LineCollection(
        segments,
        cmap=metric_colormap(),
        norm=mpl.colors.Normalize(vmin=value_min, vmax=value_max),
        linewidths=2.4,
        capstyle="round",
        joinstyle="round",
        zorder=3,
    )
    collection.set_array(segment_values)
    ax.add_collection(collection)
    return collection


def _y_limits(values: np.ndarray, *, zero_baseline: bool) -> tuple[float, float]:
    value_min = float(values.min())
    value_max = float(values.max())
    value_range = max(value_max - value_min, 1e-8)
    if zero_baseline:
        return 0.0, value_max + max(0.06 * value_max, 0.08 * value_range)
    padding = 0.10 * value_range
    return max(0.0, value_min - padding), value_max + padding


def create_metric_figure(frame: pd.DataFrame, spec: MetricSpec) -> mpl.figure.Figure:
    """Create one fixed 3-by-2-inch panel without axis-name text."""
    steps, values = prepare_curve(frame, spec)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(**PANEL_MARGINS)
    add_value_colored_line(ax, steps, values)
    ax.set_xlim(1, EXPECTED_TRAINING_STEPS)
    ax.set_xticks((1, 80, 140, 217))
    ax.get_xticklabels()[-1].set_horizontalalignment("right")
    ax.set_ylim(*_y_limits(values, zero_baseline=spec.zero_baseline))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    if spec.percent:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7, alpha=0.85)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=3.5, pad=2)
    return fig


def save_outputs(fig: mpl.figure.Figure, output_dir: Path, stem: str) -> None:
    """Save exact 3:2 vector and inspection outputs without tight cropping."""
    overflow = panel_layout_overflow(fig)
    if overflow:
        raise ValueError(f"Panel layout extends beyond the fixed canvas: {', '.join(overflow)}")
    # Re-render after the tight-bbox inspection so all fixed edge tick labels
    # remain present in both the raster and vector backends.
    fig.stale = True
    fig.canvas.draw()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=PANEL_DPI)
    fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=DEFAULT_ARTIFACTS_ROOT,
        help="Directory containing mirrored jubgjf/DMPD run directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for PDF and PNG outputs.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate source metrics without writing figures.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    run_dir = args.artifacts_root.expanduser().resolve() / RUN_ID
    frames = load_metrics(run_dir)
    for spec in METRICS:
        frame = frames[spec.metric]
        print(
            f"{spec.metric}: observations={len(frame)}, steps={frame['step'].iloc[0]}--{frame['step'].iloc[-1]}, "
            f"range=[{frame['value'].min():.6g}, {frame['value'].max():.6g}]"
        )
    if args.validate_only:
        return

    configure_style()
    output_dir = args.output_dir.expanduser().resolve()
    for spec in METRICS:
        save_outputs(create_metric_figure(frames[spec.metric], spec), output_dir, spec.stem)


if __name__ == "__main__":
    main()
