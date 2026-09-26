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
        (plot_style.STANDARD_PANEL_MARGINS, (0.29, 0.19, 0.68, 0.69)),
        (plot_style.SQUARE_PANEL_MARGINS, (0.29, 0.19, 0.69, 0.69)),
        (plot_style.FIGURE6_REGULAR_PANEL_MARGINS, (0.29, 0.25, 0.68, 0.63)),
        (plot_style.DENSE_X_PANEL_MARGINS, (0.26, 0.36, 0.73, 0.52)),
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
def test_panel_ylabels_fit_fixed_canvas(margins: dict[str, float], label: str) -> None:
    fig, ax = plot_style.create_panel(margins)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel(label)
    try:
        fig.canvas.draw()
        assert ax.get_title() == ""
        assert ax.get_ylabel() == label
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
        expected = (0.29, 0.19, 0.69, 0.69)
        assert len(figures) == 3
        assert all(_axes_bounds(fig) == pytest.approx(expected) for fig in figures)
        assert figures[-1].axes[0].get_aspect() == pytest.approx(1.0)
    finally:
        for fig in figures:
            plt.close(fig)


def _paired_gap_rows(step_count: int = 3) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for step in range(1, step_count + 1):
        before = 0.20 + 0.01 * step
        after = 0.08 + 0.005 * step
        rows.extend(
            [
                {"global_step": step, "metric_key": local_gap_dynamics.BEFORE, "value": before},
                {"global_step": step, "metric_key": local_gap_dynamics.AFTER, "value": after},
                {
                    "global_step": step,
                    "metric_key": local_gap_dynamics.REDUCTION,
                    "value": before - after,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_figure5_paired_gap_loader_uses_direct_matched_metrics() -> None:
    frame = local_gap_dynamics.paired_gap_frame(_paired_gap_rows(), expected_steps=3)

    assert frame["step"].tolist() == [1, 2, 3]
    assert frame["k3_before_intervention"].tolist() == pytest.approx([0.21, 0.22, 0.23])
    assert frame["k3_after_intervention"].tolist() == pytest.approx([0.085, 0.09, 0.095])
    assert np.allclose(
        frame["k3_before_intervention"] - frame["k3_after_intervention"],
        frame["k3_reduction"],
    )


def test_figure5_paired_gap_loader_rejects_mixed_population_reconstruction() -> None:
    rows = _paired_gap_rows()
    rows.loc[
        (rows["global_step"] == 2) & (rows["metric_key"] == local_gap_dynamics.REDUCTION),
        "value",
    ] = 0.5

    with pytest.raises(ValueError, match="logged reduction does not equal before - after"):
        local_gap_dynamics.paired_gap_frame(rows, expected_steps=3)


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
        ylabel="Top-16 token coverage",
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
        hmmt_feb_2026_value=0.3,
        five_benchmark_average_percent=39.0,
        uses_hmmt_placeholder=False,
    )
    ablation_summary.render_teacher_loss_benchmark(
        {key: result for key, _label in ablation_summary.TEACHER_LOSS_BARS},
        ablation_summary.DEFAULT_OUTPUT_DIR,
    )

    try:
        expected = (0.29, 0.19, 0.68, 0.69)
        assert len(figures) == 2
        assert all(_axes_bounds(fig) == pytest.approx(expected) for fig in figures)
        assert figures[0].axes[0].get_ylim()[0] < 92.0
        tick_labels = figures[1].axes[0].get_xticklabels()
        assert [label.get_text() for label in tick_labels] == ["Masked", "OPD\nloss", "SFT\nloss"]
        assert all(label.get_fontsize() == pytest.approx(11.5) for label in tick_labels)
        assert figures[1].axes[0].get_ylim() == pytest.approx(
            (
                ablation_summary.TEACHER_LOSS_BENCHMARK_AXIS_MIN,
                ablation_summary.TEACHER_LOSS_BENCHMARK_AXIS_MAX,
            )
        )
        assert [bar.get_y() for bar in figures[1].axes[0].patches] == pytest.approx(
            [ablation_summary.TEACHER_LOSS_BENCHMARK_AXIS_MIN] * 3
        )
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
        hmmt_feb_2026_value=0.3,
        five_benchmark_average_percent=39.0,
        uses_hmmt_placeholder=False,
    )
    ablation_summary.render_dmpd_variants_benchmark(
        {key: result for key, _label in ablation_summary.DMPD_VARIANT_BARS},
        ablation_summary.DEFAULT_OUTPUT_DIR,
    )

    figures = [*online_figures, *benchmark_figures]
    try:
        assert len(figures) == 3
        assert _axes_bounds(figures[0]) == pytest.approx((0.29, 0.25, 0.68, 0.63))
        assert _axes_bounds(figures[1]) == pytest.approx((0.26, 0.36, 0.73, 0.52))
        assert _axes_bounds(figures[2]) == pytest.approx((0.29, 0.25, 0.68, 0.63))
        outcome_annotations = [text.get_text() for text in figures[0].axes[0].texts]
        assert outcome_annotations == [r"$W\!\to\!C$ 2.00%" "\n" r"$C\!\to\!W$ 1.50%"]
        assert figures[1].axes[0].get_title() == ""
        assert figures[1].axes[0].get_ylabel() == "Net online gain"
        assert [label.get_text() for label in figures[1].axes[0].get_yticklabels()] == [
            "−2%",
            "0%",
            "2%",
            "4%",
            "6%",
            "8%",
        ]
        assert figures[2].axes[0].get_xlabel() == "Performance"
        assert figures[2].axes[0].get_ylabel() == "5-task benchmark avg."
        assert figures[2].axes[0].get_xlim() == pytest.approx((38.0, 40.2))
        assert [bar.get_x() for bar in figures[2].axes[0].patches] == pytest.approx([38.0] * 5)
    finally:
        for fig in figures:
            plt.close(fig)


@pytest.mark.parametrize(
    ("key", "four_benchmarks", "expected_average"),
    [
        (
            "prune",
            (0.47708333333333336, 0.36666666666666664, 0.8765625, 0.009615384615384616),
            38.803103092307694,
        ),
        (
            "skip_incorrect",
            (0.47291666666666665, 0.3729166666666667, 0.875, 0.022435897435897436),
            39.60023301538461,
        ),
    ],
)
def test_completed_variant_five_benchmark_results_use_hmmt_feb_only(
    key: str,
    four_benchmarks: tuple[float, ...],
    expected_average: float,
) -> None:
    row = {
        "training/global_step": ablation_summary.EXPECTED_TRAINING_STEPS,
        **dict(zip(ablation_summary.BENCHMARK_KEYS, four_benchmarks, strict=True)),
    }
    run = ablation_summary.LoadedRun(
        spec=ablation_summary.RUN_SPECS[key],
        rows=(row,),
        progress_step=ablation_summary.EXPECTED_TRAINING_STEPS,
    )

    result = ablation_summary.benchmark_result(run)

    assert result is not None
    assert result.uses_hmmt_placeholder is False
    assert result.provisional is False
    assert result.five_benchmark_average_percent == pytest.approx(expected_average)
    assert result.hmmt_feb_2026_value == pytest.approx(run.spec.hmmt_feb_2026)


def test_figure6_variant_legend_centers_two_independent_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    figures = _capture_figures(monkeypatch, ablation_summary)
    result = ablation_summary.BenchmarkResult(
        evaluation_step=217,
        progress_step=217,
        four_benchmark_values=(0.3, 0.3, 0.3, 0.3),
        hmmt_feb_2026_value=0.3,
        five_benchmark_average_percent=30.0,
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
            ["ATI-OPD", r"Full ($M=1$)", r"Full ($M=4$)"],
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
        assert [fig.axes[0].get_ylabel() for fig in figures] == [
            "Net gain per intervention",
            "Online gain",
        ]
        assert repair_figures[0].axes[0].get_xlabel() == "Intervention rate"
        for fig in figures:
            fig.canvas.draw()
            ax = fig.axes[0]
            assert ax.get_title() == ""
            assert "(%)" not in ax.get_ylabel()
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
