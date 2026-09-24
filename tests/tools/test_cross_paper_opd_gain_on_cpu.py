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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib import colors as mcolors

from analysis import cross_paper_opd_gain as gain_plot
from analysis.plot_style import configure_style, panel_layout_overflow, save_figure

plt.switch_backend("Agg")


def test_average_gains_match_appendix_three_benchmark_means() -> None:
    expected = np.asarray(
        (
            (0.27, 0.37, 0.23, -1.50, 1.35),
            (0.50, 0.77, 0.90, 1.13, 1.11),
        )
    )

    observed = gain_plot.average_gains()

    assert observed.shape == (len(gain_plot.MODEL_GROUPS), len(gain_plot.METHODS))
    assert np.round(observed, decimals=2) == pytest.approx(expected)
    negative_indices = np.argwhere(observed < 0)
    assert negative_indices.tolist() == [[0, 3]]
    assert gain_plot.MODEL_GROUPS[negative_indices[0, 0]] == "JustRL-DeepSeek"
    assert gain_plot.METHODS[negative_indices[0, 1]].label == "TOPD-0.10"


def test_grouped_bar_figure_uses_expected_layout_labels_and_units() -> None:
    configure_style()
    figure = gain_plot.create_gain_figure()
    try:
        axis = figure.axes[0]
        assert tuple(figure.get_size_inches()) == pytest.approx((3.0, 3.0))
        assert axis.get_title() == "Avg. accuracy gain"
        assert "%" not in axis.get_title()
        assert [label.get_text() for label in axis.get_xticklabels()] == list(gain_plot.MODEL_GROUPS)
        assert {label.get_fontsize() for label in axis.get_xticklabels()} == {gain_plot.MODEL_TICK_FONT_SIZE}
        assert axis.get_ylim() == pytest.approx(gain_plot.Y_LIMITS)
        assert tuple(axis.get_yticks()) == gain_plot.Y_TICKS
        assert len(axis.patches) == len(gain_plot.MODEL_GROUPS) * len(gain_plot.METHODS)

        expected_values = gain_plot.average_gains()
        expected_displayed, expected_truncated = gain_plot.display_gains(expected_values)
        observed_heights = np.asarray([patch.get_height() for patch in axis.patches])
        assert observed_heights == pytest.approx(expected_displayed.ravel())
        assert [text.get_text() for text in axis.texts] == [f"{value:+.2f}" for value in expected_values.ravel()]
        assert {text.get_rotation() for text in axis.texts} == {90.0}
        assert expected_truncated.sum() == 1
        assert expected_truncated[0, 3]
        assert observed_heights[3] == pytest.approx(gain_plot.TRUNCATED_BAR_HEIGHT)
        assert observed_heights[3] != pytest.approx(expected_values[0, 3])
        assert sum(line.get_gid() == "truncated-bar-break" for line in axis.lines) == 2

        positions = gain_plot.bar_positions()
        within_group_step = np.diff(positions, axis=1)
        between_group_gap = positions[1, 0] - positions[0, -1]
        assert within_group_step == pytest.approx(gain_plot.BAR_CENTER_STEP)
        assert between_group_gap > 3 * gain_plot.BAR_WIDTH

        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        for patch, label in zip(axis.patches, axis.texts, strict=True):
            assert label.get_window_extent(renderer).y0 > patch.get_window_extent(renderer).y1
        model_label_bounds = [label.get_window_extent(renderer) for label in axis.get_xticklabels()]
        assert model_label_bounds[0].x1 < model_label_bounds[1].x0
        assert all(label.get_text().endswith("%") for label in axis.get_yticklabels())
        assert panel_layout_overflow(figure) == ()
    finally:
        plt.close(figure)


def test_standalone_legend_matches_method_order_and_dagger_status() -> None:
    configure_style()
    figure = gain_plot.create_legend_figure()
    try:
        assert len(figure.legends) == 3
        legend_labels = [text.get_text() for legend in figure.legends for text in legend.get_texts()]
        assert legend_labels == [method.legend_label for method in gain_plot.METHODS]
        assert [len(legend.get_texts()) for legend in figure.legends] == [2, 2, 1]
        assert [method.legend_label.endswith(gain_plot.DAGGER) for method in gain_plot.METHODS] == [
            True,
            True,
            True,
            True,
            False,
        ]
        observed_colors = [handle.get_facecolor() for legend in figure.legends for handle in legend.legend_handles]
        expected_colors = [mcolors.to_rgba(method.color) for method in gain_plot.METHODS]
        assert observed_colors == pytest.approx(expected_colors)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        for legend in figure.legends:
            bounds = legend.get_window_extent(renderer)
            assert (bounds.x0 + bounds.x1) / 2 == pytest.approx(figure.bbox.width / 2)
    finally:
        plt.close(figure)


def test_pdf_outputs_use_fixed_main_canvas(tmp_path: Path) -> None:
    configure_style()
    save_figure(gain_plot.create_gain_figure(), tmp_path, gain_plot.MAIN_OUTPUT_STEM)
    save_figure(
        gain_plot.create_legend_figure(),
        tmp_path,
        gain_plot.LEGEND_OUTPUT_STEM,
        export_mode="auxiliary",
        transparent=True,
    )

    main_pdf = tmp_path / f"{gain_plot.MAIN_OUTPUT_STEM}.pdf"
    legend_pdf = tmp_path / f"{gain_plot.LEGEND_OUTPUT_STEM}.pdf"
    assert main_pdf.is_file()
    assert legend_pdf.is_file()
    media_box = re.search(
        rb"/MediaBox\s*\[\s*0\s+0\s+([0-9.]+)\s+([0-9.]+)\s*\]",
        main_pdf.read_bytes(),
    )
    assert media_box is not None
    assert (float(media_box.group(1)), float(media_box.group(2))) == pytest.approx((216.0, 216.0))
