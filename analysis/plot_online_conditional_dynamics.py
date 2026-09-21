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
and writes only PNG/PDF figures; it creates no intermediate tables.
"""

from __future__ import annotations

import gzip
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "802c696019c1449aa3328bca0e1806eb"
ARTIFACT_ROOT = (
    REPO_ROOT / "analysis_artifacts" / "jubgjf" / "DMPD" / "runs" / RUN_ID / "logged_artifacts" / "training-diagnostics"
)
OUTPUT_DIR = REPO_ROOT / "analysis" / "figures"
CACHE_ROOT = REPO_ROOT / ".analysis_cache"
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

WINDOW = 32
EXPECTED_STEPS = set(range(1, 218))
EXPECTED_BATCH_SIZE = 256
COLORS = {
    "gain": "#6689A6",
    "rescue": "#7A9E85",
    "harm": "#C7785E",
    "support": "#8E859D",
}
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
            "lines.linewidth": 2.0,
            "lines.markersize": 5.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_step_counts() -> dict[int, Counter[tuple[bool, bool]]]:
    files = sorted(ARTIFACT_ROOT.glob("v*/attachments/*opd_intervention-outcome_pairs-*.jsonl.gz"))
    if len(files) != 217:
        raise RuntimeError(f"Expected 217 outcome payloads, found {len(files)} under {ARTIFACT_ROOT}")

    batch_sizes: Counter[int] = Counter()
    counts: dict[int, Counter[tuple[bool, bool]]] = defaultdict(Counter)
    unique_samples: set[tuple[int, str, int]] = set()
    for path in files:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                step = int(record["global_step"])
                key = (step, str(record["uid"]), int(record["sample_index"]))
                if key in unique_samples:
                    raise RuntimeError(f"Duplicate outcome record: {key}")
                unique_samples.add(key)
                batch_sizes[step] += 1
                outcome = record["outcome"]
                repaired = bool(outcome.get("repair_executed")) and int(outcome.get("intervention_count", 0)) > 0
                if repaired:
                    raw_correct = outcome.get("raw_correct")
                    post_correct = outcome.get("final_correct")
                    if not isinstance(raw_correct, bool) or not isinstance(post_correct, bool):
                        raise RuntimeError(f"Missing raw/post judgment at step {step}")
                    counts[step][(raw_correct, post_correct)] += 1

    if set(batch_sizes) != EXPECTED_STEPS:
        raise RuntimeError("Outcome payload does not cover exactly steps 1--217.")
    invalid_batches = {step: size for step, size in batch_sizes.items() if size != EXPECTED_BATCH_SIZE}
    if invalid_batches:
        raise RuntimeError(f"Unexpected outcome batch sizes: {invalid_batches}")
    return counts


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


def overall_outcome_summary(counts: dict[int, Counter[tuple[bool, bool]]]) -> tuple[float, float, float]:
    """Return W→C, C→W, and net shares using all training rollouts."""
    pooled: Counter[tuple[bool, bool]] = Counter()
    for step in EXPECTED_STEPS:
        pooled.update(counts[step])
    total_rollouts = EXPECTED_BATCH_SIZE * len(EXPECTED_STEPS)
    helpful = 100 * pooled[(False, True)] / total_rollouts
    harmful = 100 * pooled[(True, False)] / total_rollouts
    return helpful, harmful, helpful - harmful


def save(fig: mpl.figure.Figure, stem: str, *, transparent: bool = False) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04, transparent=transparent)
    # fig.savefig(
    #     OUTPUT_DIR / f"{stem}.png",
    #     dpi=350,
    #     bbox_inches="tight",
    #     pad_inches=0.04,
    #     transparent=transparent,
    # )
    plt.close(fig)


def base_axis(ylabel: str) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_xlim(1, 217)
    ax.set_xticks([1, 50, 100, 150, 200])
    ax.set_xlabel("Training step")
    ax.set_title(ylabel)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7, alpha=0.85)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=4)
    return fig, ax


def render_gain(metrics: dict[str, np.ndarray]) -> None:
    fig, ax = base_axis("Net gain per repair (pp)")
    ax.axhline(0, color="#777777", linestyle="--", linewidth=1.0, zorder=1)
    ax.plot(metrics["step"], metrics["gain"], color=COLORS["gain"], marker="o", markevery=20)
    value_range = float(metrics["gain"].max() - metrics["gain"].min())
    ax.set_ylim(metrics["gain"].min() - 0.1 * value_range, metrics["gain"].max() + 0.1 * value_range)
    fig.tight_layout(pad=0.2)
    save(fig, "online_conditional_gain_deepseek")


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
        rf"All: $W\!\to\!C$ {helpful:.2f}%   $C\!\to\!W$ {harmful:.2f}%"
        "\n"
        f"Net: {net:+.2f} pp",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11.5,
        bbox={"facecolor": "white", "edgecolor": "#D8D4CF", "alpha": 0.92, "pad": 2.0},
    )
    fig.tight_layout(pad=0.2)
    save(fig, "online_rescue_harm_deepseek")


def render_support(metrics: dict[str, np.ndarray]) -> None:
    fig, ax = base_axis("Intervened samples")
    ax.plot(metrics["step"], metrics["support"], color=COLORS["support"], marker="s", markevery=20)
    ax.set_ylim(bottom=0)
    fig.tight_layout(pad=0.2)
    save(fig, "online_intervention_support_deepseek")


def render_legend() -> None:
    handles = [
        Line2D([], [], color=COLORS["gain"], marker="o", label="Net gain / repair"),
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
    save(fig, "online_conditional_dynamics_legend", transparent=True)


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
    save(fig, "online_rescue_harm_legend", transparent=True)


def main() -> None:
    configure_style()
    counts = load_step_counts()
    metrics = rolling_metrics(counts)
    render_gain(metrics)
    render_rescue_harm(metrics, overall_outcome_summary(counts))
    render_support(metrics)
    render_legend()
    render_rescue_harm_legend()


if __name__ == "__main__":
    main()
