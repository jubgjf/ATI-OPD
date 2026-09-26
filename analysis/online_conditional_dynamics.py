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

"""Plot count-based DeepSeek intervention outcomes over training.

Each point pools the raw sample counts in a trailing 32-step window. The
script reads the W&B diagnostic payloads directly from ``analysis_artifacts``
and writes only PNG/PDF figures; it creates no intermediate tables. It also
estimates the descriptive net online gain associated with the future-gap
suffix decision, using training-step cluster bootstrap confidence intervals.
"""

from __future__ import annotations

import gzip
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "802c696019c1449aa3328bca0e1806eb"
RUN_DIR = REPO_ROOT / "analysis_artifacts" / "jubgjf" / "DMPD" / "runs" / RUN_ID
ARTIFACT_ROOT = RUN_DIR / "logged_artifacts" / "training-diagnostics"
OUTPUT_DIR = REPO_ROOT / "analysis" / "figures"
CACHE_ROOT = REPO_ROOT / ".analysis_cache"
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

try:
    from analysis.plot_style import (
        DENSE_X_PANEL_MARGINS,
        FIGURE6_REGULAR_PANEL_MARGINS,
        GRID_COLOR,
        configure_style,
        create_panel,
        save_figure,
    )
except ModuleNotFoundError:  # Support ``python analysis/online_conditional_dynamics.py``.
    from plot_style import (
        DENSE_X_PANEL_MARGINS,
        FIGURE6_REGULAR_PANEL_MARGINS,
        GRID_COLOR,
        configure_style,
        create_panel,
        save_figure,
    )

WINDOW = 32
EXPECTED_STEPS = frozenset(range(1, 218))
EXPECTED_BATCH_SIZE = 256
EXPECTED_STUDENT_SUFFIX = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
EXPECTED_TEACHER_SUFFIX = "hbx/JustRL-DeepSeek-1.5B"
EXPECTED_TRAIN_FILE_SUFFIX = "DAPO-Math-17k-Processed-verl--qwen3/train.parquet"
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_921
COLORS = {
    "gain": "#6689A6",
    "rescue": "#7A9E85",
    "harm": "#C7785E",
    "support": "#8E859D",
}


@dataclass(frozen=True)
class SuffixGateObservation:
    """One rollout after intervention with a local K3 comparison and suffix decision."""

    step: int
    no_cut: bool
    local_k3_decreased: bool
    raw_correct: bool
    final_correct: bool

    @property
    def net_change(self) -> int:
        return int(self.final_correct) - int(self.raw_correct)


@dataclass(frozen=True)
class OutcomeDataset:
    """Validated sample-level outcome data for the canonical DeepSeek run."""

    step_counts: dict[int, Counter[tuple[bool, bool]]]
    suffix_observations: tuple[SuffixGateObservation, ...]
    total_rollouts: int
    intervened_rollouts: int
    missing_local_k3: int


@dataclass(frozen=True)
class SuffixGateGroup:
    """A future-gap decision crossed with the sign of local K3 reduction."""

    no_cut: bool
    local_k3_decreased: bool
    tick_label: str


@dataclass(frozen=True)
class SuffixGateEstimate:
    """Net online gain and step-cluster bootstrap interval for one group."""

    group: SuffixGateGroup
    count: int
    net_gain_pp: float
    ci_low_pp: float
    ci_high_pp: float


SUFFIX_GATE_GROUPS = (
    SuffixGateGroup(no_cut=False, local_k3_decreased=True, tick_label="Cut\n$K_3$\n$\\downarrow$"),
    SuffixGateGroup(no_cut=False, local_k3_decreased=False, tick_label="Cut\n$K_3$\n$\\leftrightarrow/\\uparrow$"),
    SuffixGateGroup(no_cut=True, local_k3_decreased=True, tick_label="No cut\n$K_3$\n$\\downarrow$"),
    SuffixGateGroup(no_cut=True, local_k3_decreased=False, tick_label="No cut\n$K_3$\n$\\leftrightarrow/\\uparrow$"),
)


def _finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(float(value))


def validate_canonical_run(run_dir: Path = RUN_DIR) -> None:
    """Reject outcome artifacts that are not from the pinned one-intervention run."""
    path = run_dir / "run.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing mirrored run metadata: {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    config = record.get("config") or {}
    intervention = (config.get("distillation") or {}).get("intervention") or {}
    student_path = ((config.get("actor_rollout_ref") or {}).get("model") or {}).get("path")
    teacher_path = (((config.get("distillation") or {}).get("teacher_models") or {}).get("teacher_model") or {}).get(
        "model_path"
    )
    train_files = (config.get("data") or {}).get("train_files") or []
    configured_steps = (((config.get("actor_rollout_ref") or {}).get("actor") or {}).get("optim") or {}).get(
        "total_training_steps"
    )
    expected_intervention = {
        "enabled": True,
        "max_interventions_per_sample": 1,
        "prune": True,
        "prune_mode": "adaptive_future",
        "gap_metric": "k3",
        "teacher_segment_length": 128,
        "train_on_teacher_segments": True,
        "teacher_segment_loss_mode": "opd",
        "skip_correct": False,
    }
    mismatches = [
        f"{key}: expected {expected!r}, got {intervention.get(key)!r}"
        for key, expected in expected_intervention.items()
        if intervention.get(key) != expected
    ]
    metadata_expectations = {
        "id": RUN_ID,
        "entity": "jubgjf",
        "project": "DMPD",
        "state": "finished",
        "last_history_step": 217,
    }
    mismatches.extend(
        f"{key}: expected {expected!r}, got {record.get(key)!r}"
        for key, expected in metadata_expectations.items()
        if record.get(key) != expected
    )
    if not str(student_path or "").endswith(EXPECTED_STUDENT_SUFFIX):
        mismatches.append(f"student path: expected suffix {EXPECTED_STUDENT_SUFFIX!r}, got {student_path!r}")
    if not str(teacher_path or "").endswith(EXPECTED_TEACHER_SUFFIX):
        mismatches.append(f"teacher path: expected suffix {EXPECTED_TEACHER_SUFFIX!r}, got {teacher_path!r}")
    if len(train_files) != 1 or not str(train_files[0]).endswith(EXPECTED_TRAIN_FILE_SUFFIX):
        mismatches.append(
            f"train files: expected one path ending in {EXPECTED_TRAIN_FILE_SUFFIX!r}, got {train_files!r}"
        )
    if configured_steps != 217:
        mismatches.append(f"configured training steps: expected 217, got {configured_steps!r}")
    if mismatches:
        raise ValueError(f"Canonical run {RUN_ID} failed configuration validation: {'; '.join(mismatches)}")


def load_outcome_data(
    artifact_root: Path = ARTIFACT_ROOT,
    *,
    expected_steps: Collection[int] = EXPECTED_STEPS,
    expected_batch_size: int = EXPECTED_BATCH_SIZE,
) -> OutcomeDataset:
    """Load and validate outcome payloads, excluding unavailable local K3 pairs."""
    expected_step_set = frozenset(int(step) for step in expected_steps)
    if not expected_step_set:
        raise ValueError("expected_steps must not be empty.")
    if expected_batch_size <= 0:
        raise ValueError("expected_batch_size must be positive.")

    files = sorted(artifact_root.glob("v*/attachments/*opd_intervention-outcome_pairs-*.jsonl.gz"))
    if len(files) != len(expected_step_set):
        raise RuntimeError(
            f"Expected {len(expected_step_set)} outcome payloads, found {len(files)} under {artifact_root}"
        )

    batch_sizes: Counter[int] = Counter()
    counts = {step: Counter() for step in expected_step_set}
    unique_samples: set[tuple[int, str, int]] = set()
    suffix_observations: list[SuffixGateObservation] = []
    intervened_rollouts = 0
    missing_local_k3 = 0
    for path in files:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                step = int(record["global_step"])
                if step not in expected_step_set:
                    raise RuntimeError(f"Unexpected training step {step} in {path}")
                uid = record.get("uid")
                sample_index = record.get("sample_index")
                if not isinstance(uid, str) or not uid:
                    raise RuntimeError(f"Missing sample uid at step {step} in {path}")
                if isinstance(sample_index, bool) or not isinstance(sample_index, int):
                    raise RuntimeError(f"Invalid sample_index at step {step}: {sample_index!r}")
                key = (step, uid, sample_index)
                if key in unique_samples:
                    raise RuntimeError(f"Duplicate outcome record: {key}")
                unique_samples.add(key)
                batch_sizes[step] += 1

                outcome = record.get("outcome")
                if not isinstance(outcome, Mapping):
                    raise RuntimeError(f"Missing outcome mapping for {key}")
                # ``repair_executed`` is a legacy artifact field name.
                intervention_executed = outcome.get("repair_executed")
                intervention_count = outcome.get("intervention_count")
                if not isinstance(intervention_executed, bool):
                    raise RuntimeError(f"Invalid legacy repair_executed field for {key}: {intervention_executed!r}")
                if isinstance(intervention_count, bool) or not isinstance(intervention_count, int):
                    raise RuntimeError(f"Invalid intervention_count for {key}: {intervention_count!r}")
                if intervention_executed != (intervention_count > 0):
                    raise RuntimeError(
                        f"Inconsistent intervention flag/count for {key}: "
                        f"{intervention_executed!r}, {intervention_count!r}"
                    )
                if not intervention_executed:
                    continue
                if intervention_count != 1:
                    raise RuntimeError(
                        f"Canonical run must have exactly one intervention for {key}; got {intervention_count}"
                    )

                interventions = record.get("interventions")
                if not isinstance(interventions, Sequence) or isinstance(interventions, str | bytes):
                    raise RuntimeError(f"Missing intervention list for {key}")
                if len(interventions) != 1 or not isinstance(interventions[0], Mapping):
                    raise RuntimeError(f"Expected one intervention payload for {key}; got {len(interventions)}")
                raw_correct = outcome.get("raw_correct")
                final_correct = outcome.get("final_correct")
                if not isinstance(raw_correct, bool) or not isinstance(final_correct, bool):
                    raise RuntimeError(f"Missing raw/final judgment for {key}")
                counts[step][(raw_correct, final_correct)] += 1
                intervened_rollouts += 1

                intervention = interventions[0]
                local_k3_reduction = intervention.get("local_effect_gap_reduction")
                if not _finite_number(local_k3_reduction):
                    missing_local_k3 += 1
                    continue
                no_cut = intervention.get("prune_future_no_cut")
                if not isinstance(no_cut, bool):
                    raise RuntimeError(f"Missing future-gap decision for {key}")
                suffix_observations.append(
                    SuffixGateObservation(
                        step=step,
                        no_cut=no_cut,
                        local_k3_decreased=float(local_k3_reduction) > 0.0,
                        raw_correct=raw_correct,
                        final_correct=final_correct,
                    )
                )

    if frozenset(batch_sizes) != expected_step_set:
        missing_steps = sorted(expected_step_set - set(batch_sizes))
        extra_steps = sorted(set(batch_sizes) - expected_step_set)
        raise RuntimeError(f"Outcome payload step mismatch; missing={missing_steps}, extra={extra_steps}")
    invalid_batches = {step: size for step, size in sorted(batch_sizes.items()) if size != expected_batch_size}
    if invalid_batches:
        raise RuntimeError(f"Unexpected outcome batch sizes: {invalid_batches}")
    total_rollouts = len(unique_samples)
    expected_total = expected_batch_size * len(expected_step_set)
    if total_rollouts != expected_total:
        raise RuntimeError(f"Expected {expected_total} unique rollouts, found {total_rollouts}")
    return OutcomeDataset(
        step_counts=counts,
        suffix_observations=tuple(suffix_observations),
        total_rollouts=total_rollouts,
        intervened_rollouts=intervened_rollouts,
        missing_local_k3=missing_local_k3,
    )


def load_step_counts() -> dict[int, Counter[tuple[bool, bool]]]:
    """Backward-compatible count-only entry point for the existing panels."""
    return load_outcome_data().step_counts


def estimate_suffix_gate(
    observations: Sequence[SuffixGateObservation],
    *,
    cluster_steps: Collection[int] | None = None,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[SuffixGateEstimate, ...]:
    """Estimate group net gain with a percentile bootstrap over training steps."""
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive.")
    observed_steps = {observation.step for observation in observations}
    step_values = sorted(observed_steps if cluster_steps is None else {int(step) for step in cluster_steps})
    if not step_values:
        raise ValueError("At least one cluster step is required.")
    unknown_steps = observed_steps - set(step_values)
    if unknown_steps:
        raise ValueError(f"Observations contain steps outside cluster_steps: {sorted(unknown_steps)}")

    step_index = {step: index for index, step in enumerate(step_values)}
    group_index = {(group.no_cut, group.local_k3_decreased): index for index, group in enumerate(SUFFIX_GATE_GROUPS)}
    cluster_deltas = np.zeros((len(step_values), len(SUFFIX_GATE_GROUPS)), dtype=np.int64)
    cluster_counts = np.zeros_like(cluster_deltas)
    for observation in observations:
        group = group_index[(observation.no_cut, observation.local_k3_decreased)]
        step = step_index[observation.step]
        cluster_deltas[step, group] += observation.net_change
        cluster_counts[step, group] += 1

    total_deltas = cluster_deltas.sum(axis=0)
    total_counts = cluster_counts.sum(axis=0)
    if np.any(total_counts == 0):
        empty = [
            SUFFIX_GATE_GROUPS[index].tick_label.replace("\n", " / ") for index in np.flatnonzero(total_counts == 0)
        ]
        raise ValueError(f"No observations for suffix-gate groups: {empty}")
    point_estimates = 100.0 * total_deltas / total_counts

    rng = np.random.default_rng(seed)
    bootstrap_values = np.full((bootstrap_samples, len(SUFFIX_GATE_GROUPS)), np.nan, dtype=float)
    chunk_size = 1_000
    for start in range(0, bootstrap_samples, chunk_size):
        stop = min(start + chunk_size, bootstrap_samples)
        sampled_clusters = rng.integers(0, len(step_values), size=(stop - start, len(step_values)))
        sampled_deltas = cluster_deltas[sampled_clusters].sum(axis=1)
        sampled_counts = cluster_counts[sampled_clusters].sum(axis=1)
        np.divide(
            100.0 * sampled_deltas,
            sampled_counts,
            out=bootstrap_values[start:stop],
            where=sampled_counts > 0,
        )
    confidence_intervals = np.nanquantile(bootstrap_values, (0.025, 0.975), axis=0)
    return tuple(
        SuffixGateEstimate(
            group=group,
            count=int(total_counts[index]),
            net_gain_pp=float(point_estimates[index]),
            ci_low_pp=float(confidence_intervals[0, index]),
            ci_high_pp=float(confidence_intervals[1, index]),
        )
        for index, group in enumerate(SUFFIX_GATE_GROUPS)
    )


def rolling_metrics(counts: dict[int, Counter[tuple[bool, bool]]]) -> dict[str, np.ndarray]:
    steps = np.arange(WINDOW, 218)
    output: dict[str, list[float]] = {"step": [], "gain": [], "rescue": [], "harm": [], "support": []}
    for end in steps:
        pooled: Counter[tuple[bool, bool]] = Counter()
        for step in range(end - WINDOW + 1, end + 1):
            pooled.update(counts[step])
        ww = pooled[(False, False)]
        wc = pooled[(False, True)]
        cw = pooled[(True, False)]
        cc = pooled[(True, True)]
        support = ww + wc + cw + cc
        output["step"].append(float(end))
        output["gain"].append(100 * (wc - cw) / support)
        output["rescue"].append(100 * wc / (ww + wc))
        output["harm"].append(100 * cw / (cw + cc))
        output["support"].append(float(support))
    return {name: np.asarray(values) for name, values in output.items()}


def overall_outcome_summary(
    counts: Mapping[int, Counter[tuple[bool, bool]]],
    *,
    total_rollouts: int = EXPECTED_BATCH_SIZE * len(EXPECTED_STEPS),
) -> tuple[float, float, float]:
    """Return W→C, C→W, and net shares using all training rollouts."""
    pooled: Counter[tuple[bool, bool]] = Counter()
    for step_counts in counts.values():
        pooled.update(step_counts)
    if total_rollouts <= 0:
        raise ValueError("total_rollouts must be positive.")
    helpful = 100 * pooled[(False, True)] / total_rollouts
    harmful = 100 * pooled[(True, False)] / total_rollouts
    return helpful, harmful, helpful - harmful


def base_axis(ylabel: str) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
    fig, ax = create_panel(FIGURE6_REGULAR_PANEL_MARGINS)
    ax.set_xlim(1, 217)
    ax.set_xticks([1, 50, 100, 150, 200])
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7, alpha=0.85)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=4)
    return fig, ax


def render_gain(metrics: dict[str, np.ndarray]) -> None:
    fig, ax = base_axis("Net gain per intervention")
    ax.axhline(0, color="#777777", linestyle="--", linewidth=1.0, zorder=1)
    ax.plot(metrics["step"], metrics["gain"], color=COLORS["gain"], marker="o", markevery=20)
    value_range = float(metrics["gain"].max() - metrics["gain"].min())
    ax.set_ylim(metrics["gain"].min() - 0.1 * value_range, metrics["gain"].max() + 0.1 * value_range)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=1))
    save_figure(fig, OUTPUT_DIR, "online_conditional_gain_deepseek")


def render_rescue_harm(metrics: dict[str, np.ndarray], overall: tuple[float, float, float]) -> None:
    fig, ax = base_axis("Conditional outcome rate")
    ax.plot(metrics["step"], metrics["rescue"], color=COLORS["rescue"], marker="o", markevery=20)
    ax.plot(
        metrics["step"],
        metrics["harm"],
        color=COLORS["harm"],
        marker="^",
        markerfacecolor="white",
        markeredgewidth=0.9,
        markevery=20,
    )
    # Reserve a small amount of headroom for the all-training summary so the
    # annotation does not obscure either conditional-rate curve.
    ax.set_ylim(0, 35)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    helpful, harmful, net = overall
    ax.text(
        0.97,
        0.96,
        rf"$W\!\to\!C$ {helpful:.2f}%"
        "\n"
        rf"$C\!\to\!W$ {harmful:.2f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11.5,
        bbox={"facecolor": "white", "edgecolor": "#D8D4CF", "alpha": 0.92, "pad": 2.0},
    )
    save_figure(fig, OUTPUT_DIR, "online_rescue_harm_deepseek")


def render_support(metrics: dict[str, np.ndarray]) -> None:
    fig, ax = base_axis("Intervened samples")
    ax.plot(metrics["step"], metrics["support"], color=COLORS["support"], marker="s", markevery=20)
    ax.set_ylim(bottom=0)
    save_figure(fig, OUTPUT_DIR, "online_intervention_support_deepseek")


def render_suffix_gate(estimates: Sequence[SuffixGateEstimate]) -> None:
    """Render descriptive net gain for the four suffix-gate/K3 groups."""
    if len(estimates) != len(SUFFIX_GATE_GROUPS):
        raise ValueError(f"Expected {len(SUFFIX_GATE_GROUPS)} suffix-gate estimates, got {len(estimates)}")
    fig, ax = create_panel(DENSE_X_PANEL_MARGINS)
    positions = np.asarray((0.0, 1.0, 2.2, 3.2))
    for position, estimate in zip(positions, estimates, strict=True):
        decreased = estimate.group.local_k3_decreased
        color = COLORS["rescue"] if decreased else COLORS["harm"]
        marker = "o" if decreased else "^"
        marker_face = color if decreased else "white"
        ax.errorbar(
            position,
            estimate.net_gain_pp,
            yerr=np.asarray(
                [
                    [estimate.net_gain_pp - estimate.ci_low_pp],
                    [estimate.ci_high_pp - estimate.net_gain_pp],
                ]
            ),
            fmt=marker,
            color=color,
            markerfacecolor=marker_face,
            markeredgecolor=color,
            markeredgewidth=0.9,
            markersize=5.5,
            capsize=3.5,
            capthick=1.2,
            elinewidth=1.5,
            linestyle="none",
            zorder=3,
        )
    ax.axhline(0.0, color="#777777", linestyle="--", linewidth=1.0, zorder=1)
    ax.set_xlim(-0.5, 3.7)
    ax.set_ylim(-2.0, 8.0)
    ax.set_yticks((-2, 0, 2, 4, 6, 8))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_xticks(positions, [estimate.group.tick_label for estimate in estimates])
    ax.set_xlabel("Future-gap decision")
    ax.set_ylabel("Net online gain")
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7, alpha=0.85)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=4)
    save_figure(fig, OUTPUT_DIR, "online_suffix_gate_net_gain_deepseek")


def render_legend() -> None:
    handles = [
        Line2D([], [], color=COLORS["gain"], marker="o", label="Net gain / intervention"),
        Line2D([], [], color=COLORS["rescue"], marker="o", label="Rescue rate"),
        Line2D(
            [],
            [],
            color=COLORS["harm"],
            marker="^",
            markerfacecolor="white",
            markeredgewidth=0.9,
            label="Harm rate",
        ),
        Line2D([], [], color=COLORS["support"], marker="s", label="Intervened samples"),
    ]
    fig = plt.figure(figsize=(7.4, 0.35))
    fig.legend(handles=handles, loc="center", frameon=False, ncol=4, handlelength=1.8, columnspacing=0.9)
    save_figure(
        fig,
        OUTPUT_DIR,
        "online_conditional_dynamics_legend",
        export_mode="auxiliary",
        transparent=True,
    )


def render_rescue_harm_legend() -> None:
    handles = [
        Line2D([], [], color=COLORS["rescue"], marker="o", label="Rescue rate"),
        Line2D(
            [],
            [],
            color=COLORS["harm"],
            marker="^",
            markerfacecolor="white",
            markeredgewidth=0.9,
            label="Harm rate",
        ),
    ]
    fig = plt.figure(figsize=(3.8, 0.35))
    fig.legend(handles=handles, loc="center", frameon=False, ncol=2, handlelength=1.8, columnspacing=0.9)
    save_figure(
        fig,
        OUTPUT_DIR,
        "online_rescue_harm_legend",
        export_mode="auxiliary",
        transparent=True,
    )


def main() -> None:
    configure_style()
    validate_canonical_run()
    dataset = load_outcome_data()
    metrics = rolling_metrics(dataset.step_counts)
    estimates = estimate_suffix_gate(dataset.suffix_observations, cluster_steps=EXPECTED_STEPS)
    print(
        f"Validated {len(EXPECTED_STEPS)} steps x {EXPECTED_BATCH_SIZE} rollouts: "
        f"total={dataset.total_rollouts}, intervened={dataset.intervened_rollouts}, "
        f"local_k3_missing={dataset.missing_local_k3}, analyzed={len(dataset.suffix_observations)}"
    )
    for estimate in estimates:
        label = estimate.group.tick_label.replace("\n", " / ").replace("$", "")
        print(
            f"  {label}: N={estimate.count}, net={estimate.net_gain_pp:+.2f} pp, "
            f"95% step-cluster CI=[{estimate.ci_low_pp:.2f}, {estimate.ci_high_pp:.2f}]"
        )
    render_gain(metrics)
    render_rescue_harm(
        metrics,
        overall_outcome_summary(dataset.step_counts, total_rollouts=dataset.total_rollouts),
    )
    render_support(metrics)
    render_suffix_gate(estimates)
    render_legend()
    render_rescue_harm_legend()


if __name__ == "__main__":
    main()
