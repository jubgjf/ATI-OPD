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

"""Plot paired local K3 before and after a teacher replacement.

The input is the DeepSeek ATI-OPD main run. Both curves are read directly from
the ``local_effect_v2`` diagnostics and therefore use the same intervention
pairs and the same number (up to 16) of valid Student tokens on each side. The
script deliberately rejects data for which the logged paired reduction is not
equal to ``before - after``. No independent post-handoff population is mixed
into either curve or ratio.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping

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

try:
    from analysis.plot_style import (
        GRID_COLOR,
        SQUARE_PANEL_MARGINS,
        configure_style,
        create_panel,
        save_figure,
    )
except ModuleNotFoundError:  # Support ``python analysis/local_gap_dynamics.py``.
    from plot_style import (
        GRID_COLOR,
        SQUARE_PANEL_MARGINS,
        configure_style,
        create_panel,
        save_figure,
    )

RUN_ID = "802c696019c1449aa3328bca0e1806eb"
EXPECTED_TRAINING_STEPS = 217
BEFORE = "opd_intervention/local_effect_v2/active_gap/before_first16_mean"
AFTER = "opd_intervention/local_effect_v2/active_gap/after_first16_mean"
REDUCTION = "opd_intervention/local_effect_v2/active_gap/reduction_first16_mean"
PAIRED_METRICS = (BEFORE, AFTER, REDUCTION)
SMOOTHING_WINDOW = 7
K3_THRESHOLD = 0.15
RAW_COLOR = "#B97964"
AFTER_INTERVENTION_COLOR = "#6689A6"
THRESHOLD_RATIO_COLOR = "#7C9E87"
REFERENCE_COLOR = "#626970"


def _diagnostic_shards(run_dir: Path, run_id: str) -> tuple[Path, ...]:
    artifact_root = run_dir / "logged_artifacts" / "training-diagnostics"
    manifest_path = artifact_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing diagnostics manifest: {manifest_path}. "
            "Run the W&B downloader with --training-diagnostics-shards first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("run_id") != run_id:
        raise ValueError(f"Diagnostics manifest belongs to {manifest.get('run_id')!r}, expected {run_id!r}.")
    if manifest.get("artifact_type") != "training-diagnostics":
        raise ValueError(f"Unexpected diagnostics artifact type: {manifest.get('artifact_type')!r}.")
    if manifest.get("status") != "complete":
        raise ValueError(f"Diagnostics mirror is not complete: status={manifest.get('status')!r}.")

    versions = manifest.get("versions")
    if not isinstance(versions, list) or not versions:
        raise ValueError("Diagnostics manifest has no artifact versions.")
    paths: list[Path] = []
    for version in versions:
        if not isinstance(version, Mapping):
            raise ValueError(f"Invalid diagnostics version entry: {version!r}.")
        version_name = str(version.get("version"))
        entries = version.get("entries")
        if not isinstance(entries, Mapping) or len(entries) != 1:
            raise ValueError(f"Expected one metric shard in {version_name}, got {entries!r}.")
        relative_name, signature = next(iter(entries.items()))
        relative_path = Path(str(relative_name))
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.parts[:1] != ("shards",)
            or relative_path.suffix != ".parquet"
        ):
            raise ValueError(f"Unsafe or unexpected diagnostics shard path: {relative_name!r}.")
        path = artifact_root / version_name / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Missing diagnostics shard declared by the manifest: {path}.")
        if isinstance(signature, Mapping):
            expected_size = int(signature.get("size") or 0)
            if expected_size > 0 and path.stat().st_size != expected_size:
                raise ValueError(
                    f"Diagnostics shard size mismatch for {path}: expected {expected_size}, got {path.stat().st_size}."
                )
        paths.append(path)
    if len(paths) != len(set(paths)):
        raise ValueError("Diagnostics manifest declares a metric shard more than once.")
    return tuple(paths)


def paired_gap_frame(rows: pd.DataFrame, *, expected_steps: int = EXPECTED_TRAINING_STEPS) -> pd.DataFrame:
    """Validate and pivot direct metrics from the same matched intervention pairs."""
    required_columns = ("global_step", "metric_key", "value")
    missing_columns = set(required_columns).difference(rows.columns)
    if missing_columns:
        raise ValueError(f"Paired K3 rows are missing columns: {sorted(missing_columns)}.")
    selected = rows.loc[rows["metric_key"].isin(PAIRED_METRICS), list(required_columns)].copy()
    if selected.empty:
        raise ValueError("No paired local-effect K3 metrics were found in the diagnostics shards.")
    selected["global_step"] = pd.to_numeric(selected["global_step"], errors="coerce")
    selected["value"] = pd.to_numeric(selected["value"], errors="coerce")
    if not np.isfinite(selected[["global_step", "value"]].to_numpy(dtype=float)).all():
        raise ValueError("Paired local-effect K3 metrics contain a non-finite step or value.")
    if not np.equal(selected["global_step"], np.floor(selected["global_step"])).all():
        raise ValueError("Paired local-effect K3 metrics contain a non-integral training step.")
    selected["global_step"] = selected["global_step"].astype(int)
    if selected.duplicated(["global_step", "metric_key"]).any():
        duplicates = selected.loc[
            selected.duplicated(["global_step", "metric_key"], keep=False),
            ["global_step", "metric_key"],
        ]
        raise ValueError(f"Duplicate paired local-effect K3 metrics found: {duplicates.head().to_dict('records')}.")

    pivot = selected.pivot(index="global_step", columns="metric_key", values="value").sort_index()
    missing_metrics = set(PAIRED_METRICS).difference(pivot.columns)
    if missing_metrics:
        raise ValueError(f"Missing paired local-effect K3 metrics: {sorted(missing_metrics)}.")
    expected_index = pd.Index(range(1, expected_steps + 1), name="global_step")
    if not pivot.index.equals(expected_index):
        missing_steps = expected_index.difference(pivot.index).tolist()
        extra_steps = pivot.index.difference(expected_index).tolist()
        raise ValueError(
            f"Expected paired K3 metrics for steps 1--{expected_steps}; "
            f"missing={missing_steps[:8]}, extra={extra_steps[:8]}."
        )
    if pivot[list(PAIRED_METRICS)].isna().any().any():
        raise ValueError("At least one training step lacks a paired before, after, or reduction value.")

    before = pivot[BEFORE].to_numpy(dtype=float)
    after = pivot[AFTER].to_numpy(dtype=float)
    reduction = pivot[REDUCTION].to_numpy(dtype=float)
    if bool((before < 0).any()) or bool((after < 0).any()):
        raise ValueError("Paired before/after K3 values must be non-negative.")
    if not np.allclose(before - after, reduction, rtol=1e-9, atol=1e-10):
        max_error = float(np.max(np.abs((before - after) - reduction)))
        raise ValueError(
            "Paired K3 identity failed: logged reduction does not equal before - after "
            f"(maximum absolute error {max_error:.3g})."
        )
    return pd.DataFrame(
        {
            "step": expected_index.to_numpy(dtype=int),
            "k3_before_intervention": before,
            "k3_after_intervention": after,
            "k3_reduction": reduction,
        }
    )


def load_gap_series(run_id: str = RUN_ID) -> pd.DataFrame:
    run_dir = ARTIFACTS_ROOT / run_id
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if metadata.get("entity") != "jubgjf" or metadata.get("project") != "DMPD":
        raise ValueError(f"Run {run_id} is not from jubgjf/DMPD.")

    metric_frames: list[pd.DataFrame] = []
    for shard in _diagnostic_shards(run_dir, run_id):
        rows = pd.read_parquet(
            shard,
            columns=["logical_run_id", "event_kind", "global_step", "metric_key", "value"],
        )
        selected = rows.loc[rows["metric_key"].isin(PAIRED_METRICS)].copy()
        if not selected.empty:
            if set(selected["logical_run_id"].dropna().astype(str)) != {run_id}:
                raise ValueError(f"Diagnostics shard {shard} contains metrics from another run.")
            if set(selected["event_kind"].dropna().astype(str)) != {"train"}:
                raise ValueError(f"Diagnostics shard {shard} contains non-training paired K3 metrics.")
            metric_frames.append(selected[["global_step", "metric_key", "value"]])
    if not metric_frames:
        raise ValueError(f"No paired local-effect K3 metrics found for run {run_id}.")
    return paired_gap_frame(pd.concat(metric_frames, ignore_index=True))


def style_axes(ax: mpl.axes.Axes) -> None:
    ax.set_xlim(1, 217)
    ax.set_xticks([1, 50, 100, 150, 200])
    ax.set_xlabel("Training step")
    ax.grid(color=GRID_COLOR, linewidth=0.7, alpha=0.85)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=4)


def render_local_reduction(frame: pd.DataFrame) -> None:
    smoothed = (
        frame[["k3_before_intervention", "k3_after_intervention"]]
        .rolling(
            SMOOTHING_WINDOW,
            center=True,
            min_periods=1,
        )
        .mean()
    )
    marker_steps = list(range(0, len(frame), 24))

    fig, ax = create_panel(SQUARE_PANEL_MARGINS)
    ax.axhline(K3_THRESHOLD, color=REFERENCE_COLOR, linestyle="--", linewidth=1.2, zorder=1)
    ax.plot(
        frame["step"],
        smoothed["k3_before_intervention"],
        color=RAW_COLOR,
        linewidth=2.0,
        marker="^",
        markersize=4.2,
        markevery=marker_steps,
    )
    ax.plot(
        frame["step"],
        smoothed["k3_after_intervention"],
        color=AFTER_INTERVENTION_COLOR,
        linewidth=2.0,
        marker="o",
        markersize=4.2,
        markevery=marker_steps,
    )
    ax.set_ylim(0, 1.08 * float(smoothed["k3_before_intervention"].max()))
    ax.set_ylabel("Local K3")
    style_axes(ax)
    save_figure(fig, OUTPUT_DIR, "local_gap_reduction_deepseek")


def render_normalized_exit_gap(frame: pd.DataFrame) -> None:
    smoothed = (
        frame[["k3_before_intervention", "k3_after_intervention"]]
        .rolling(
            SMOOTHING_WINDOW,
            center=True,
            min_periods=1,
        )
        .mean()
    )
    exit_over_before = smoothed["k3_after_intervention"] / smoothed["k3_before_intervention"]
    exit_over_threshold = smoothed["k3_after_intervention"] / K3_THRESHOLD
    marker_steps = list(range(0, len(frame), 24))

    fig, ax = create_panel(SQUARE_PANEL_MARGINS)
    ax.axhspan(0, 1, color=THRESHOLD_RATIO_COLOR, alpha=0.08, linewidth=0, zorder=0)
    ax.axhline(1, color=REFERENCE_COLOR, linestyle="--", linewidth=1.2, zorder=1)
    ax.plot(
        frame["step"],
        exit_over_before,
        color=AFTER_INTERVENTION_COLOR,
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
    ax.set_ylabel("Normalized handoff gap")
    style_axes(ax)
    save_figure(fig, OUTPUT_DIR, "exit_gap_deepseek")


def render_legends() -> None:
    local_handles = [
        Line2D([], [], color=RAW_COLOR, marker="^", linewidth=2.0, markersize=5),
        Line2D([], [], color=AFTER_INTERVENTION_COLOR, marker="o", linewidth=2.0, markersize=5),
        Line2D([], [], color=REFERENCE_COLOR, linestyle="--", linewidth=1.2),
    ]
    fig = plt.figure(figsize=(6.0, 0.35))
    fig.legend(
        local_handles,
        ["Before: original suffix", "After: rewritten suffix", r"Trigger $\tau$"],
        loc="center",
        frameon=False,
        ncol=3,
        handlelength=1.5,
        columnspacing=0.9,
    )
    save_figure(
        fig,
        OUTPUT_DIR,
        "local_gap_reduction_legend",
        export_mode="auxiliary",
        transparent=True,
    )

    normalized_handles = [
        Line2D([], [], color=AFTER_INTERVENTION_COLOR, marker="o", linewidth=2.0, markersize=5),
        Line2D([], [], color=THRESHOLD_RATIO_COLOR, marker="s", linewidth=2.0, markersize=5),
    ]
    fig = plt.figure(figsize=(2.9, 0.35))
    fig.legend(
        normalized_handles,
        ["After / before (paired)", r"After / $\tau$ (paired)"],
        loc="center",
        frameon=False,
        ncol=2,
        handlelength=1.5,
        columnspacing=0.9,
    )
    save_figure(
        fig,
        OUTPUT_DIR,
        "normalized_exit_gap_legend",
        export_mode="auxiliary",
        transparent=True,
    )


def main() -> None:
    configure_style()
    frame = load_gap_series()
    render_local_reduction(frame)
    render_normalized_exit_gap(frame)
    render_legends()


if __name__ == "__main__":
    main()
