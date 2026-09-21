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

from __future__ import annotations

import re
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from analysis import (
    ablation_summary,
    local_gap_dynamics,
    online_conditional_dynamics,
    plot_style,
    repair_outcome,
    student_uptake,
)


def _axes_bounds(fig: plt.Figure) -> tuple[float, float, float, float]:
    fig.canvas.draw()
    return tuple(float(value) for value in fig.axes[0].get_position().bounds)


def _capture_figures(monkeypatch: pytest.MonkeyPatch, module: object) -> list[plt.Figure]:
    captured: list[plt.Figure] = []

    def capture(fig: plt.Figure, *_args: object, **_kwargs: object) -> None:
        captured.append(fig)

    monkeypatch.setattr(module, "save_figure", capture)
    return captured


@pytest.mark.parametrize(
    ("margins", "expected"),
    [
        (plot_style.STANDARD_PANEL_MARGINS, (0.22, 0.19, 0.75, 0.69)),
        (plot_style.SQUARE_PANEL_MARGINS, (0.22, 0.19, 0.69, 0.69)),
        (plot_style.FIGURE6_REGULAR_PANEL_MARGINS, (0.22, 0.25, 0.75, 0.63)),
        (plot_style.DENSE_X_PANEL_MARGINS, (0.18, 0.36, 0.81, 0.52)),
    ],
)
def test_shared_panel_layouts_have_fixed_bounds(
    margins: dict[str, float],
    expected: tuple[float, float, float, float],
) -> None:
    fig, _ = plot_style.create_panel(margins)
    try:
        assert tuple(fig.get_size_inches()) == pytest.approx((3.0, 3.0))
        assert _axes_bounds(fig) == pytest.approx(expected)
    finally:
        plt.close(fig)


def test_panel_export_keeps_exact_canvas(tmp_path) -> None:
    fig, ax = plot_style.create_panel()
    ax.plot([0, 1], [0, 1])
    ax.set_xticks([])
    ax.set_yticks([])
    plot_style.save_figure(fig, tmp_path, "panel")

    pdf_bytes = (tmp_path / "panel.pdf").read_bytes()
    media_box = re.search(
        rb"/MediaBox\s*\[\s*0\s+0\s+([0-9.]+)\s+([0-9.]+)\s*\]",
        pdf_bytes,
    )
    assert media_box is not None
    assert (float(media_box.group(1)), float(media_box.group(2))) == pytest.approx((216.0, 216.0))


@pytest.mark.parametrize(
    "margins",
    [
        plot_style.STANDARD_PANEL_MARGINS,
        plot_style.SQUARE_PANEL_MARGINS,
        plot_style.FIGURE6_REGULAR_PANEL_MARGINS,
        plot_style.DENSE_X_PANEL_MARGINS,
    ],
)
@pytest.mark.parametrize("label", ["Mean zlib compression ratio", "Net gain per intervention"])
def test_panel_titles_are_centered_on_canvas_and_fit(margins: dict[str, float], label: str) -> None:
    fig, ax = plot_style.create_panel(margins)
    ax.set_xticks([])
    ax.set_yticks([])
    title = plot_style.set_panel_title(ax, label)
    try:
        fig.canvas.draw()
        title_box = title.get_window_extent(fig.canvas.get_renderer())
        assert (title_box.x0 + title_box.x1) / 2 == pytest.approx(fig.bbox.width / 2)
        assert plot_style.panel_layout_overflow(fig) == ()
    finally:
        plt.close(fig)


def test_panel_export_rejects_text_outside_canvas(tmp_path) -> None:
    fig, ax = plot_style.create_panel()
    ax.set_title("A deliberately overlong title that cannot fit inside the fixed panel canvas")

    with pytest.raises(ValueError, match="Panel layout extends beyond the fixed canvas"):
        plot_style.save_figure(fig, tmp_path, "overflow")
    plt.close(fig)


def test_figure4_panels_share_square_axes(monkeypatch: pytest.MonkeyPatch) -> None:
    steps = np.arange(1, 218)
    before = 0.20 + 0.02 * np.sin(steps / 15)
    after = 0.10 + 0.01 * np.sin(steps / 17)
    local_frame = pd.DataFrame(
        {
            "step": steps,
            "k3_before_intervention": before,
            "k3_after_intervention": after,
        }
    )
    local_figures = _capture_figures(monkeypatch, local_gap_dynamics)
    local_gap_dynamics.render_local_reduction(local_frame)
    local_gap_dynamics.render_normalized_exit_gap(local_frame)

    uptake_frame = pd.DataFrame(
        {
            "step": steps,
            "before_intervention": before,
            "after_intervention": after,
        }
    )
    monkeypatch.setattr(student_uptake, "load_uptake", lambda _run_id: uptake_frame)
    uptake_figures = _capture_figures(monkeypatch, student_uptake)
    student_uptake.render_panel("test", "run")

    figures = [*local_figures, *uptake_figures]
    try:
        expected = (0.22, 0.19, 0.69, 0.69)
        assert len(figures) == 3
        assert all(_axes_bounds(fig) == pytest.approx(expected) for fig in figures)
        assert figures[-1].axes[0].get_aspect() == pytest.approx(1.0)
    finally:
        for fig in figures:
            plt.close(fig)


def test_figure5_line_and_bar_panels_share_standard_axes(monkeypatch: pytest.MonkeyPatch) -> None:
    metric = "coverage"
    rows = tuple(
        {"training/global_step": step, metric: 0.92 + 0.0002 * step}
        for step in range(1, ablation_summary.EXPECTED_TRAINING_STEPS + 1)
    )
    runs = {key: SimpleNamespace(rows=rows) for key in ablation_summary.LOSS_KEYS}
    figures = _capture_figures(monkeypatch, ablation_summary)
    ablation_summary.render_loss_metric(
        runs,
        metric=metric,
        title="Top-16 token coverage",
        stem="coverage",
        output_dir=ablation_summary.DEFAULT_OUTPUT_DIR,
        scale=100.0,
        sparse=True,
        percent=True,
        dynamic_percent_range=True,
    )
    result = ablation_summary.BenchmarkResult(
        evaluation_step=217,
        progress_step=217,
        four_benchmark_values=(0.3, 0.3, 0.3, 0.3),
        hmmt_values=(0.3, 0.3),
        six_benchmark_average_percent=30.0,
        uses_hmmt_placeholder=False,
    )
    ablation_summary.render_teacher_loss_benchmark(
        {key: result for key, _label in ablation_summary.TEACHER_LOSS_BARS},
        ablation_summary.DEFAULT_OUTPUT_DIR,
    )

    try:
        expected = (0.22, 0.19, 0.75, 0.69)
        assert len(figures) == 2
        assert all(_axes_bounds(fig) == pytest.approx(expected) for fig in figures)
        assert figures[0].axes[0].get_ylim()[0] < 92.0
        tick_labels = figures[1].axes[0].get_xticklabels()
        assert [label.get_text() for label in tick_labels] == ["Masked", "OPD\nloss", "SFT\nloss"]
        assert all(label.get_fontsize() == pytest.approx(11.5) for label in tick_labels)
    finally:
        for fig in figures:
            plt.close(fig)


def test_figure6_keeps_dense_suffix_axis_separate(monkeypatch: pytest.MonkeyPatch) -> None:
    online_figures = _capture_figures(monkeypatch, online_conditional_dynamics)
    online_conditional_dynamics.render_rescue_harm(
        {
            "step": np.asarray((1.0, 217.0)),
            "rescue": np.asarray((20.0, 5.0)),
            "harm": np.asarray((18.0, 10.0)),
        },
        (2.0, 1.5, 0.5),
    )
    estimates = tuple(
        online_conditional_dynamics.SuffixGateEstimate(group, 10, value, value - 1, value + 1)
        for group, value in zip(
            online_conditional_dynamics.SUFFIX_GATE_GROUPS,
            (1.0, 0.0, 6.0, 4.5),
            strict=True,
        )
    )
    online_conditional_dynamics.render_suffix_gate(estimates)

    benchmark_figures = _capture_figures(monkeypatch, ablation_summary)
    result = ablation_summary.BenchmarkResult(
        evaluation_step=217,
        progress_step=217,
        four_benchmark_values=(0.3, 0.3, 0.3, 0.3),
        hmmt_values=(0.3, 0.3),
        six_benchmark_average_percent=30.0,
        uses_hmmt_placeholder=False,
    )
    ablation_summary.render_dmpd_variants_benchmark(
        {key: result for key, _label in ablation_summary.DMPD_VARIANT_BARS},
        ablation_summary.DEFAULT_OUTPUT_DIR,
    )

    figures = [*online_figures, *benchmark_figures]
    try:
        assert len(figures) == 3
        assert _axes_bounds(figures[0]) == pytest.approx((0.22, 0.25, 0.75, 0.63))
        assert _axes_bounds(figures[1]) == pytest.approx((0.18, 0.36, 0.81, 0.52))
        assert _axes_bounds(figures[2]) == pytest.approx((0.22, 0.25, 0.75, 0.63))
        outcome_annotations = [text.get_text() for text in figures[0].axes[0].texts]
        assert any("Net: +0.50%" in text for text in outcome_annotations)
        assert all(" pp" not in text for text in outcome_annotations)
        assert figures[1].axes[0].get_title() == "Net online gain"
        assert [label.get_text() for label in figures[1].axes[0].get_yticklabels()] == [
            "−2%",
            "0%",
            "2%",
            "4%",
            "6%",
            "8%",
        ]
        assert figures[2].axes[0].get_xlabel() == "Performance"
    finally:
        for fig in figures:
            plt.close(fig)


def test_figure6_variant_legend_centers_two_independent_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    figures = _capture_figures(monkeypatch, ablation_summary)
    result = ablation_summary.BenchmarkResult(
        evaluation_step=217,
        progress_step=217,
        four_benchmark_values=(0.3, 0.3, 0.3, 0.3),
        hmmt_values=(0.3, 0.3),
        six_benchmark_average_percent=30.0,
        uses_hmmt_placeholder=False,
    )
    ablation_summary.render_dmpd_variants_benchmark_legend(
        {key: result for key, _label in ablation_summary.DMPD_VARIANT_BARS},
        ablation_summary.DEFAULT_OUTPUT_DIR,
    )

    try:
        assert len(figures) == 1
        fig = figures[0]
        assert len(fig.legends) == 2
        assert [[text.get_text() for text in legend.get_texts()] for legend in fig.legends] == [
            ["DMPD", r"Full ($M=1$)", r"Full ($M=4$)"],
            ["Prune", "Skip incorrect"],
        ]
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for legend in fig.legends:
            bounds = legend.get_window_extent(renderer)
            assert (bounds.x0 + bounds.x1) / 2 == pytest.approx(fig.bbox.width / 2)

        first_row_text = [text.get_window_extent(renderer) for text in fig.legends[0].get_texts()]
        text_gaps = [first_row_text[index + 1].x0 - first_row_text[index].x1 for index in range(2)]
        assert text_gaps[0] == pytest.approx(text_gaps[1], abs=1.0)
    finally:
        for fig in figures:
            plt.close(fig)


def test_online_gain_units_are_carried_by_axis_ticks(monkeypatch: pytest.MonkeyPatch) -> None:
    online_figures = _capture_figures(monkeypatch, online_conditional_dynamics)
    online_conditional_dynamics.render_gain(
        {
            "step": np.asarray((1.0, 217.0)),
            "gain": np.asarray((-1.0, 1.0)),
        }
    )

    frame = pd.DataFrame(
        {
            "step": (1, 33, 65),
            "rate": (10.0, 20.0, 15.0),
            "gain": (-1.0, 2.0, 0.5),
        }
    )
    monkeypatch.setattr(repair_outcome, "load_outcomes", lambda _run_id: frame)
    repair_figures = _capture_figures(monkeypatch, repair_outcome)
    repair_outcome.render_panel("test", "run")

    figures = [*online_figures, *repair_figures]
    try:
        assert [fig.axes[0].get_title() for fig in figures] == [
            "Net gain per intervention",
            "Online gain",
        ]
        assert repair_figures[0].axes[0].get_xlabel() == "Intervention rate"
        for fig in figures:
            fig.canvas.draw()
            ax = fig.axes[0]
            assert "(%)" not in ax.get_title()
            assert "(%)" not in ax.get_xlabel()
            assert plot_style.panel_layout_overflow(fig) == ()
        assert all(label.get_text().endswith("%") for label in online_figures[0].axes[0].get_yticklabels())
        assert all(label.get_text().endswith("%") for label in repair_figures[0].axes[0].get_xticklabels())
        assert all(label.get_text().endswith("%") for label in repair_figures[0].axes[0].get_yticklabels())
    finally:
        for fig in figures:
            plt.close(fig)


def test_dynamic_percent_axis_keeps_lowest_observation_in_view() -> None:
    limits, ticks = ablation_summary.dynamic_percent_axis((92.0109748840332, 98.94603490829468))

    assert limits == pytest.approx((91.5, 100.0))
    assert ticks == pytest.approx((92.0, 94.0, 96.0, 98.0, 100.0))
