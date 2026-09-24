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

"""Render vanilla-OPD gap trajectories as response-position slices.

Each plane fixes one response position and shows how the absolute
Teacher--Student policy gap evolves during training. Curve and fill segments
use a shared blue-to-red scale based on their local gap height, so the same gap
has the same color on every slice. An optional interactive mode lets the user
rotate the three-dimensional view and save the result.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = REPO_ROOT / ".analysis_cache"
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

try:
    from analysis.plot_style import PANEL_DPI, configure_style, panel_layout_overflow
except ModuleNotFoundError:  # Support ``python analysis/long_sequence_divergence_slices.py``.
    from plot_style import PANEL_DPI, configure_style, panel_layout_overflow

DEFAULT_ARTIFACTS_ROOT = REPO_ROOT / "analysis_artifacts" / "jubgjf" / "DMPD" / "runs"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "analysis" / "figures"
RUN_ID = "9bbeb176889b42aa847665d03274fd68"
EXPECTED_RUN_NAME = "vanilla-opd-20260824-124656"
EXPECTED_STUDENT_SUFFIX = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
EXPECTED_TEACHER_SUFFIX = "hbx/JustRL-DeepSeek-1.5B"
EXPECTED_TRAIN_FILE_SUFFIX = "DAPO-Math-17k-Processed-verl--qwen3/train.parquet"
EXPECTED_TRAINING_STEPS = 217
EXPECTED_RESPONSE_LENGTH = 23_552
EXPECTED_BIN_SIZE = 512
EXPECTED_ARTIFACT_VERSIONS = tuple(f"v{index}" for index in range(11))
EXPECTED_GAP_STEPS = tuple(range(1, EXPECTED_TRAINING_STEPS + 1))

GAP_PREFIX = "opd/gap_abs_bin/"
COUNT_PREFIX = "bucket/bin/"
COUNT_SUFFIX = "/token_count"
SHARD_RE = re.compile(r"metrics-step-(\d+)-(\d+)-[A-Za-z0-9]+\.parquet$")
DISPLAY_POSITION_BIN_SIZE = 2_048
DISPLAY_STEP_BIN_SIZE = 10

SAMPLED_POSITION_INDICES = (0, 2, 4, 6, 8, 10, 11)
TOKENS_PER_K = 1024.0
CURVE_SMOOTHING_PASSES = 3
FIGSIZE = (4.25, 2.8)
RESPONSE_POSITION_OUTPUT_STEM = "vanilla_opd_abs_gap_response_position_slices"
DEFAULT_ELEVATION = 24.0
DEFAULT_AZIMUTH = -62.0
DEFAULT_ROLL = 0.0

SLICE_COLORMAP_COLORS = ("#78A7DF", "#B7A6BC", "#DE6758")
COLOR_MAX_GAP = 4.0
SLICE_FILL_ALPHA = 0.14
AXIS_COLOR = "#9B9B9B"
TICK_LABEL_PAD = -2.0


@dataclass(frozen=True)
class DivergenceGapData:
    """Validated position-binned absolute-gap diagnostics."""

    bin_edges: np.ndarray
    gap_abs: pd.DataFrame
    token_counts: pd.DataFrame


@dataclass(frozen=True)
class CoarsenedGapData:
    """Token-count-weighted bins used by the compact slice plot."""

    position_edges: np.ndarray
    step_edges: np.ndarray
    gap_abs: np.ndarray


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def validate_canonical_run(run_dir: Path) -> None:
    """Reject local data that do not belong to the pinned vanilla OPD run."""
    path = run_dir / "run.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing mirrored W&B run metadata: {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    config = record.get("config") or {}
    if not isinstance(config, Mapping):
        raise ValueError(f"Run {RUN_ID} has an invalid config payload.")

    student_path = _nested(config, "actor_rollout_ref", "model", "path")
    teacher_path = _nested(config, "distillation", "teacher_models", "teacher_model", "model_path")
    train_files = _nested(config, "data", "train_files") or []
    expectations = {
        "run id": (record.get("id"), RUN_ID),
        "run name": (record.get("name"), EXPECTED_RUN_NAME),
        "entity": (record.get("entity"), "jubgjf"),
        "project": (record.get("project"), "DMPD"),
        "state": (record.get("state"), "finished"),
        "last history step": (record.get("last_history_step"), EXPECTED_TRAINING_STEPS),
        "configured training steps": (
            _nested(config, "actor_rollout_ref", "actor", "optim", "total_training_steps"),
            EXPECTED_TRAINING_STEPS,
        ),
        "maximum response length": (_nested(config, "data", "max_response_length"), EXPECTED_RESPONSE_LENGTH),
        "distillation enabled": (_nested(config, "distillation", "enabled"), True),
        "actor loss disabled": (_nested(config, "distillation", "use_actor_loss"), False),
        "distillation loss mode": (_nested(config, "distillation", "distillation_loss", "loss_mode"), "k1"),
        "policy loss mode": (
            _nested(config, "distillation", "distillation_loss", "policy_loss_mode"),
            "vanilla",
        ),
        "loss horizon": (_nested(config, "distillation", "distillation_loss", "horizon", "mode"), "full"),
        "intervention enabled": (_nested(config, "distillation", "intervention", "enabled"), False),
        "intervention pruning": (_nested(config, "distillation", "intervention", "prune"), False),
        "Teacher-segment training": (
            _nested(config, "distillation", "intervention", "train_on_teacher_segments"),
            False,
        ),
        "diagnostics enabled": (_nested(config, "distillation", "diagnostics", "enabled"), True),
        "position bin size": (
            _nested(config, "distillation", "diagnostics", "position_bin_size"),
            EXPECTED_BIN_SIZE,
        ),
    }
    mismatches = [
        f"{label}: expected {expected!r}, got {actual!r}"
        for label, (actual, expected) in expectations.items()
        if actual != expected
    ]
    if not str(student_path or "").endswith(EXPECTED_STUDENT_SUFFIX):
        mismatches.append(f"student path: expected suffix {EXPECTED_STUDENT_SUFFIX!r}, got {student_path!r}")
    if not str(teacher_path or "").endswith(EXPECTED_TEACHER_SUFFIX):
        mismatches.append(f"teacher path: expected suffix {EXPECTED_TEACHER_SUFFIX!r}, got {teacher_path!r}")
    if not isinstance(train_files, Sequence) or isinstance(train_files, str | bytes):
        mismatches.append(f"train files: expected a one-element sequence, got {train_files!r}")
    elif len(train_files) != 1 or not str(train_files[0]).endswith(EXPECTED_TRAIN_FILE_SUFFIX):
        mismatches.append(
            f"train files: expected one path ending in {EXPECTED_TRAIN_FILE_SUFFIX!r}, got {train_files!r}"
        )
    if mismatches:
        raise ValueError(f"Canonical vanilla OPD run validation failed: {'; '.join(mismatches)}")


def diagnostic_shard_paths(artifact_root: Path) -> tuple[Path, ...]:
    """Resolve the 11 contiguous metric shards declared by the local manifest."""
    manifest_path = artifact_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing diagnostics manifest: {manifest_path}. "
            "Run the W&B downloader with --training-diagnostics-shards first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("run_id") != RUN_ID:
        raise ValueError(f"Diagnostics manifest belongs to {manifest.get('run_id')!r}, expected {RUN_ID!r}.")
    if manifest.get("artifact_type") != "training-diagnostics":
        raise ValueError(f"Unexpected artifact type: {manifest.get('artifact_type')!r}")
    versions = manifest.get("versions")
    if not isinstance(versions, list):
        raise ValueError("Diagnostics manifest has no version list.")
    observed_versions = tuple(str(version.get("version")) for version in versions if isinstance(version, Mapping))
    if observed_versions != EXPECTED_ARTIFACT_VERSIONS:
        raise ValueError(f"Expected artifact versions {EXPECTED_ARTIFACT_VERSIONS!r}, got {observed_versions!r}.")

    ranged_paths: list[tuple[int, int, Path]] = []
    for version in versions:
        version_name = str(version["version"])
        entries = version.get("entries")
        if not isinstance(entries, Mapping) or len(entries) != 1:
            raise ValueError(f"Expected exactly one metric shard in {version_name}, got {entries!r}.")
        relative_name, signature = next(iter(entries.items()))
        relative_name = str(relative_name)
        if not isinstance(signature, Mapping):
            raise ValueError(f"Invalid shard signature in {version_name}: {signature!r}")
        relative_path = Path(relative_name)
        if relative_path.is_absolute() or ".." in relative_path.parts or relative_path.parts[:1] != ("shards",):
            raise ValueError(f"Unsafe or unexpected diagnostics shard path: {relative_name!r}")
        match = SHARD_RE.fullmatch(relative_path.name)
        if match is None:
            raise ValueError(f"Unexpected diagnostics shard filename: {relative_path.name!r}")
        path = artifact_root / version_name / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Missing diagnostics shard declared by the manifest: {path}")
        expected_size = int(signature.get("size") or 0)
        if expected_size > 0 and path.stat().st_size != expected_size:
            raise ValueError(
                f"Diagnostics shard size mismatch for {path}: expected {expected_size}, got {path.stat().st_size}."
            )
        expected_digest = signature.get("digest")
        if expected_digest:
            digest = hashlib.md5(path.read_bytes(), usedforsecurity=False)
            observed_digest = base64.b64encode(digest.digest()).decode("ascii")
            if observed_digest != expected_digest:
                raise ValueError(f"Diagnostics shard checksum mismatch for {path}.")
        ranged_paths.append((int(match.group(1)), int(match.group(2)), path))

    ranged_paths.sort()
    next_step = 1
    for start, end, path in ranged_paths:
        if start != next_step or end < start:
            raise ValueError(
                f"Diagnostics shards are not contiguous at {path.name}: expected start {next_step}, got {start}-{end}."
            )
        next_step = end + 1
    if next_step != EXPECTED_TRAINING_STEPS + 1:
        raise ValueError(f"Diagnostics shards end at step {next_step - 1}, expected {EXPECTED_TRAINING_STEPS}.")
    return tuple(path for _start, _end, path in ranged_paths)


def _expected_bins() -> tuple[tuple[int, int], ...]:
    return tuple((start, start + EXPECTED_BIN_SIZE) for start in range(0, EXPECTED_RESPONSE_LENGTH, EXPECTED_BIN_SIZE))


def _validated_metric_matrix(
    rows: pd.DataFrame,
    *,
    label: str,
    expected_steps: Sequence[int],
    expected_bins: Sequence[tuple[int, int]],
) -> pd.DataFrame:
    if rows.empty:
        raise ValueError(f"No raw rows found for {label}.")
    rows = rows.copy()
    for column in ("global_step", "value", "bin_start", "bin_end"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
        if not np.isfinite(rows[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{label} contains a non-finite {column} value.")
    for column in ("global_step", "bin_start", "bin_end"):
        values = rows[column].to_numpy(dtype=float)
        if not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{label} contains a non-integral {column} value.")
        rows[column] = rows[column].astype(int)

    if rows.duplicated(["global_step", "bin_start", "bin_end"]).any():
        raise ValueError(f"{label} contains duplicate step/bin records.")
    observed_steps = tuple(sorted(rows["global_step"].unique()))
    if observed_steps != tuple(expected_steps):
        raise ValueError(f"{label} steps differ from the expected schedule: {observed_steps!r}")
    observed_bins = set(zip(rows["bin_start"], rows["bin_end"], strict=True))
    if observed_bins != set(expected_bins):
        raise ValueError(f"{label} position bins differ from the expected 512-token grid.")
    if (rows["bin_end"] - rows["bin_start"] != EXPECTED_BIN_SIZE).any():
        raise ValueError(f"{label} contains a position bin whose width is not {EXPECTED_BIN_SIZE}.")

    columns = [start for start, _end in expected_bins]
    matrix = rows.pivot(index="global_step", columns="bin_start", values="value")
    return matrix.reindex(index=expected_steps, columns=columns)


def load_gap_data(artifact_root: Path) -> DivergenceGapData:
    """Load and cross-check raw absolute-gap and response-token-count rows."""
    paths = diagnostic_shard_paths(artifact_root)
    columns = ["global_step", "metric_key", "value", "bin_start", "bin_end"]
    frames = [pd.read_parquet(path, columns=columns) for path in paths]
    raw = pd.concat(frames, ignore_index=True)
    keys = raw["metric_key"].fillna("").astype(str)
    expected_bins = _expected_bins()

    gap = _validated_metric_matrix(
        raw.loc[keys.str.startswith(GAP_PREFIX)],
        label="sampled-token absolute gap",
        expected_steps=EXPECTED_GAP_STEPS,
        expected_bins=expected_bins,
    )
    counts = _validated_metric_matrix(
        raw.loc[keys.str.startswith(COUNT_PREFIX) & keys.str.endswith(COUNT_SUFFIX)],
        label="response-token count",
        expected_steps=EXPECTED_GAP_STEPS,
        expected_bins=expected_bins,
    )

    if counts.isna().any().any() or (counts.to_numpy() < 0).any():
        raise ValueError("Response-token counts must be finite and present for every step/bin pair.")
    if not np.equal(counts.to_numpy(), np.floor(counts.to_numpy())).all():
        raise ValueError("Response-token counts must be integral.")
    if (gap.to_numpy()[np.isfinite(gap.to_numpy())] < 0).any():
        raise ValueError("Absolute policy gap cannot be negative.")
    if not np.array_equal(gap.isna().to_numpy(), counts.to_numpy() == 0):
        raise ValueError("Sampled-token absolute gap missing cells do not exactly match zero-token bins.")

    bin_edges = np.arange(0, EXPECTED_RESPONSE_LENGTH + EXPECTED_BIN_SIZE, EXPECTED_BIN_SIZE, dtype=float)
    return DivergenceGapData(bin_edges=bin_edges, gap_abs=gap, token_counts=counts)


def _weighted_block_means(
    values: np.ndarray,
    weights: np.ndarray,
    row_groups: Sequence[Sequence[int]],
    column_groups: Sequence[Sequence[int]],
) -> np.ndarray:
    output = np.full((len(row_groups), len(column_groups)), np.nan, dtype=float)
    for output_row, source_rows in enumerate(row_groups):
        for output_column, source_columns in enumerate(column_groups):
            block_values = values[np.ix_(source_rows, source_columns)]
            block_weights = weights[np.ix_(source_rows, source_columns)]
            valid = np.isfinite(block_values) & np.isfinite(block_weights) & (block_weights > 0)
            if valid.any():
                output[output_row, output_column] = np.average(block_values[valid], weights=block_weights[valid])
    return output


def coarsen_gap_data(data: DivergenceGapData) -> CoarsenedGapData:
    """Aggregate raw diagnostics into token-count-weighted plot bins."""
    if DISPLAY_POSITION_BIN_SIZE % EXPECTED_BIN_SIZE:
        raise ValueError("The display position bin must be a multiple of the raw position bin.")
    position_factor = DISPLAY_POSITION_BIN_SIZE // EXPECTED_BIN_SIZE
    column_groups = tuple(
        tuple(range(start, min(start + position_factor, data.gap_abs.shape[1])))
        for start in range(0, data.gap_abs.shape[1], position_factor)
    )
    position_edges = np.asarray(
        [data.bin_edges[group[0]] for group in column_groups] + [data.bin_edges[-1]],
        dtype=float,
    )

    steps = np.asarray(data.gap_abs.index, dtype=int)
    row_groups = tuple(
        tuple(range(start, min(start + DISPLAY_STEP_BIN_SIZE, len(steps))))
        for start in range(0, len(steps), DISPLAY_STEP_BIN_SIZE)
    )
    step_edges = np.asarray(
        [steps[group[0]] - 0.5 for group in row_groups] + [steps[-1] + 0.5],
        dtype=float,
    )
    counts = data.token_counts.reindex(index=data.gap_abs.index, columns=data.gap_abs.columns)
    gap_abs = _weighted_block_means(
        data.gap_abs.to_numpy(dtype=float),
        counts.to_numpy(dtype=float),
        row_groups,
        column_groups,
    )
    return CoarsenedGapData(position_edges=position_edges, step_edges=step_edges, gap_abs=gap_abs)


@dataclass(frozen=True)
class ResponsePositionSliceData:
    """Coarsened gap trajectories for selected response-position slices."""

    training_steps: np.ndarray
    positions_k: np.ndarray
    values: np.ndarray


def prepare_response_position_slice_data(
    data: DivergenceGapData,
    sampled_position_indices: Sequence[int] = SAMPLED_POSITION_INDICES,
) -> ResponsePositionSliceData:
    """Select coarse response-position bins and retain their training trajectories."""
    display = coarsen_gap_data(data)
    indices = tuple(int(index) for index in sampled_position_indices)
    if (
        not indices
        or len(set(indices)) != len(indices)
        or any(right <= left for left, right in zip(indices[:-1], indices[1:], strict=True))
    ):
        raise ValueError(f"Sampled response-position indices must be non-empty, unique, and increasing: {indices!r}")
    invalid = tuple(index for index in indices if index < 0 or index >= display.gap_abs.shape[1])
    if invalid:
        raise ValueError(f"Sampled response-position indices are outside the display grid: {invalid!r}")

    training_steps = (display.step_edges[:-1] + display.step_edges[1:]) / 2.0
    all_position_centers_k = (display.position_edges[:-1] + display.position_edges[1:]) / (2.0 * TOKENS_PER_K)
    values = display.gap_abs[:, list(indices)].T
    if values.shape != (len(indices), len(training_steps)):
        raise ValueError(f"Unexpected response-position slice shape: {values.shape!r}")
    return ResponsePositionSliceData(
        training_steps=training_steps,
        positions_k=all_position_centers_k[list(indices)],
        values=values,
    )


def _finite_gap_values(values: np.ndarray) -> np.ndarray:
    """Return all finite local gap values used by the shared color scale."""
    if values.ndim != 2:
        raise ValueError(f"Slice values must form a two-dimensional matrix, got {values.shape!r}.")
    finite = values[np.isfinite(values)]
    if not finite.size:
        raise ValueError("Response-position slices contain no finite gap value.")
    if (finite < 0).any():
        raise ValueError("Absolute policy gap cannot be negative.")
    return finite


def _gap_colormap() -> mpl.colors.Colormap:
    return mpl.colors.LinearSegmentedColormap.from_list("local_absolute_gap", SLICE_COLORMAP_COLORS)


def _gap_color_normalizer(values: np.ndarray) -> mpl.colors.Normalize:
    """Map a global absolute-gap height to one color across all slices."""
    _finite_gap_values(values)
    return mpl.colors.Normalize(vmin=0.0, vmax=COLOR_MAX_GAP, clip=True)


def _finite_segments(x: np.ndarray, values: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Split a profile into contiguous finite runs without filling missing bins."""
    finite = np.isfinite(values)
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    start: int | None = None
    for index, valid in enumerate(np.append(finite, False)):
        if valid and start is None:
            start = index
        elif not valid and start is not None:
            if index - start >= 2:
                segments.append((x[start:index], values[start:index]))
            start = None
    return tuple(segments)


def _smooth_segment(
    x: np.ndarray,
    values: np.ndarray,
    passes: int = CURVE_SMOOTHING_PASSES,
) -> tuple[np.ndarray, np.ndarray]:
    """Round a finite polyline with convex corner cutting and fixed endpoints."""
    if x.ndim != 1 or values.ndim != 1 or len(x) != len(values):
        raise ValueError(f"Curve coordinates must be equal-length one-dimensional arrays: {x.shape}, {values.shape}")
    if len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(values).all():
        raise ValueError("A smoothing segment must contain at least two finite points.")
    if not np.all(np.diff(x) > 0):
        raise ValueError("Smoothing coordinates must be strictly increasing.")
    if isinstance(passes, bool) or not isinstance(passes, int) or passes < 0:
        raise ValueError(f"Smoothing passes must be a non-negative integer, got {passes!r}.")

    points = np.column_stack((x, values)).astype(float, copy=True)
    for _ in range(passes):
        left = 0.75 * points[:-1] + 0.25 * points[1:]
        right = 0.25 * points[:-1] + 0.75 * points[1:]
        refined = np.empty((2 * len(points), 2), dtype=float)
        refined[0] = points[0]
        refined[-1] = points[-1]
        refined[1:-1] = np.stack((left, right), axis=1).reshape(-1, 2)
        points = refined
    return points[:, 0], points[:, 1]


def _smoothed_finite_segments(
    x: np.ndarray,
    values: np.ndarray,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Smooth each finite run independently without bridging missing bins."""
    return tuple(_smooth_segment(segment_x, segment_z) for segment_x, segment_z in _finite_segments(x, values))


def _add_response_position_slice(
    ax: mpl.axes.Axes,
    training_steps: np.ndarray,
    values: np.ndarray,
    position_k: float,
    colormap: mpl.colors.Colormap,
    normalizer: mpl.colors.Normalize,
) -> None:
    finite_segments = _smoothed_finite_segments(training_steps, values)
    for segment_steps, segment_z in finite_segments:
        local_gap = (segment_z[:-1] + segment_z[1:]) / 2.0
        fill_vertices = [
            (
                (left_step, 0.0),
                (left_step, left_gap),
                (right_step, right_gap),
                (right_step, 0.0),
            )
            for left_step, right_step, left_gap, right_gap in zip(
                segment_steps[:-1],
                segment_steps[1:],
                segment_z[:-1],
                segment_z[1:],
                strict=True,
            )
        ]
        fill_collection = PolyCollection(
            fill_vertices,
            facecolors=colormap(normalizer(local_gap)),
            edgecolor="none",
            alpha=SLICE_FILL_ALPHA,
            antialiaseds=False,
        )
        fill_collection.set_array(local_gap)
        fill_collection.set_cmap(colormap)
        fill_collection.set_norm(normalizer)
        ax.add_collection3d(fill_collection, zs=position_k, zdir="y")

        points = np.column_stack(
            (
                segment_steps,
                np.full_like(segment_steps, position_k, dtype=float),
                segment_z,
            )
        )
        line_collection = Line3DCollection(
            np.stack((points[:-1], points[1:]), axis=1),
            cmap=colormap,
            norm=normalizer,
            linewidths=1.7,
            alpha=0.98,
            capstyle="round",
            joinstyle="round",
        )
        line_collection.set_array(local_gap)
        ax.add_collection3d(line_collection)


def _style_axes(
    ax: mpl.axes.Axes,
    *,
    z_max: float,
    elevation: float,
    azimuth: float,
    roll: float,
) -> None:
    response_limit_k = EXPECTED_RESPONSE_LENGTH / TOKENS_PER_K
    ax.set_xlim(1.0, 217.0)
    ax.set_ylim(0.0, response_limit_k)
    ax.set_xticks((1, 80, 140, 217))
    ax.set_yticks((0, 8, 16, response_limit_k), ("0", "8K", "16K", "23K"))
    ax.set_zlim(0.0, z_max)
    ax.set_zticks(np.arange(0.0, z_max, 2.0))
    ax.tick_params(axis="both", which="major", labelsize=8.5, pad=TICK_LABEL_PAD)
    ax.zaxis.set_tick_params(labelsize=8.5, pad=TICK_LABEL_PAD)

    ax.set_proj_type("ortho")
    ax.view_init(elev=elevation, azim=azimuth, roll=roll)
    ax.mouse_init(rotate_btn=1, pan_btn=2, zoom_btn=3)
    ax.set_box_aspect((1.8, 1.0, 1.08))
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
        axis.line.set_color(AXIS_COLOR)
        axis.line.set_linewidth(0.8)

    ax.set_position((-0.005, 0.08, 0.90, 0.88))


def create_response_position_figure(
    slice_data: ResponsePositionSliceData,
    *,
    elevation: float = DEFAULT_ELEVATION,
    azimuth: float = DEFAULT_AZIMUTH,
    roll: float = DEFAULT_ROLL,
) -> mpl.figure.Figure:
    """Create a waterfall view whose constant-depth slices are response positions."""
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")
    finite_values = _finite_gap_values(slice_data.values)
    colormap = _gap_colormap()
    normalizer = _gap_color_normalizer(slice_data.values)
    for position_k, values in zip(slice_data.positions_k, slice_data.values, strict=True):
        _add_response_position_slice(
            ax,
            slice_data.training_steps,
            values,
            float(position_k),
            colormap,
            normalizer,
        )

    z_max = max(2.0, float(np.ceil(finite_values.max())))
    _style_axes(
        ax,
        z_max=z_max,
        elevation=elevation,
        azimuth=azimuth,
        roll=roll,
    )
    return fig


def _view_parameters(ax: mpl.axes.Axes) -> dict[str, float]:
    return {
        "elevation": float(ax.elev),
        "azimuth": float(ax.azim),
        "roll": float(ax.roll),
    }


def _format_view(ax: mpl.axes.Axes) -> str:
    view = _view_parameters(ax)
    return f"elev={view['elevation']:.2f}, azim={view['azimuth']:.2f}, roll={view['roll']:.2f}"


def _write_outputs(fig: mpl.figure.Figure, output_dir: Path, stem: str) -> None:
    """Write the current camera view without closing its figure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.03}
    fig.savefig(output_dir / f"{stem}.pdf", **save_kwargs)
    fig.savefig(output_dir / f"{stem}.png", dpi=PANEL_DPI, **save_kwargs)
    (output_dir / f"{stem}_view.json").write_text(
        json.dumps(_view_parameters(fig.axes[0]), indent=2) + "\n",
        encoding="utf-8",
    )


def save_outputs(fig: mpl.figure.Figure, output_dir: Path, stem: str) -> None:
    """Save tightly cropped PDF/PNG assets and close a non-interactive figure."""
    overflow = panel_layout_overflow(fig, tolerance_points=1.0)
    if overflow:
        raise ValueError(f"Slice plot extends beyond the fixed canvas: {', '.join(overflow)}")
    _write_outputs(fig, output_dir, stem)
    plt.close(fig)


def activate_interactive_backend(requested_backend: str | None = None) -> str:
    """Switch from the non-interactive Agg backend to an available GUI backend."""
    if requested_backend:
        candidates = (requested_backend,)
    elif sys.platform == "darwin":
        candidates = ("MacOSX", "QtAgg", "TkAgg")
    else:
        candidates = ("QtAgg", "TkAgg")

    failures: list[str] = []
    for candidate in candidates:
        try:
            plt.switch_backend(candidate)
        except (ImportError, ModuleNotFoundError, RuntimeError, ValueError) as error:
            failures.append(f"{candidate}: {error}")
        else:
            return candidate
    raise RuntimeError(
        f"No Matplotlib GUI backend could be activated. Tried {', '.join(candidates)}. Details: {'; '.join(failures)}"
    )


def launch_interactive_view(
    fig: mpl.figure.Figure,
    output_dir: Path,
    stem: str = RESPONSE_POSITION_OUTPUT_STEM,
) -> None:
    """Open a rotatable view whose current camera can be printed or saved."""
    ax = fig.axes[0]

    def on_key(event: Any) -> None:
        key = str(event.key or "").lower()
        if key == "s":
            _write_outputs(fig, output_dir, stem)
            print(f"Saved {stem}.pdf/.png with {_format_view(ax)}")
        elif key == "p":
            print(f"Current view: {_format_view(ax)}")
        elif key in {"q", "escape"}:
            plt.close(fig)

    callback_id = fig.canvas.mpl_connect("key_press_event", on_key)
    print("Interactive controls: drag=rotate, middle-drag=pan, right-drag=zoom, s=save, p=print view, q/Esc=quit")
    print(f"Initial view: {_format_view(ax)}")
    try:
        plt.show(block=True)
    finally:
        fig.canvas.mpl_disconnect(callback_id)
        plt.close(fig)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=DEFAULT_ARTIFACTS_ROOT,
        help="Directory containing mirrored W&B run directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for PDF/PNG outputs.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate and sample data without writing figures.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Open a GUI window for mouse-controlled camera adjustment instead of saving immediately.",
    )
    parser.add_argument(
        "--interactive-backend",
        help="Optional Matplotlib GUI backend, for example MacOSX, QtAgg, or TkAgg.",
    )
    parser.add_argument("--elev", type=float, default=DEFAULT_ELEVATION, help="Initial camera elevation in degrees.")
    parser.add_argument("--azim", type=float, default=DEFAULT_AZIMUTH, help="Initial camera azimuth in degrees.")
    parser.add_argument("--roll", type=float, default=DEFAULT_ROLL, help="Initial camera roll in degrees.")
    args = parser.parse_args(argv)
    if args.validate_only and args.interactive:
        parser.error("--validate-only and --interactive cannot be used together.")
    return args


def main() -> None:
    args = parse_args()
    run_dir = args.artifacts_root.expanduser().resolve() / RUN_ID
    artifact_root = run_dir / "logged_artifacts" / "training-diagnostics"
    validate_canonical_run(run_dir)
    data = load_gap_data(artifact_root)
    response_position_slices = prepare_response_position_slice_data(data)
    finite_values = _finite_gap_values(response_position_slices.values)
    print(
        "Validated vanilla OPD response-position slices: "
        f"shape={response_position_slices.values.shape[0]}x{response_position_slices.values.shape[1]}, "
        f"local_gap_range=[{finite_values.min():.3f}, {finite_values.max():.3f}]."
    )
    if args.validate_only:
        return

    if args.interactive:
        backend = activate_interactive_backend(args.interactive_backend)
        print(f"Interactive Matplotlib backend: {backend}")
    configure_style()
    output_dir = args.output_dir.expanduser().resolve()
    figure = create_response_position_figure(
        response_position_slices,
        elevation=args.elev,
        azimuth=args.azim,
        roll=args.roll,
    )
    if args.interactive:
        launch_interactive_view(figure, output_dir)
    else:
        save_outputs(figure, output_dir, RESPONSE_POSITION_OUTPUT_STEM)


if __name__ == "__main__":
    main()
