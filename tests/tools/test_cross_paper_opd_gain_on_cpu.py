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
            (-8.66, 0.27, 0.37, 0.23, -1.50, 1.35),
            (-2.12, 0.50, 0.77, 0.90, 1.13, 1.11),
        )
    )

    observed = gain_plot.average_gains()

    assert observed.shape == (len(gain_plot.MODEL_GROUPS), len(gain_plot.METHODS))
    assert np.round(observed, decimals=2) == pytest.approx(expected)
    negative_indices = np.argwhere(observed < 0)
    assert [method.key for method in gain_plot.METHODS] == [
        "esr",
        "popd",
        "topd_050",
        "topd_025",
        "topd_010",
        "ati_opd",
    ]
    assert negative_indices.tolist() == [[0, 0], [0, 4], [1, 0]]
    assert gain_plot.MODEL_GROUPS[negative_indices[0, 0]] == "JustRL-DeepSeek"
    assert gain_plot.METHODS[negative_indices[0, 1]].label == "ESR"


def test_grouped_bar_figure_uses_expected_layout_labels_and_units() -> None:
    configure_style()
    figure = gain_plot.create_gain_figure()
    try:
        axis = figure.axes[0]
        assert tuple(figure.get_size_inches()) == pytest.approx(gain_plot.MAIN_FIGSIZE)
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
        assert {text.get_rotation() for text in axis.texts} == {0.0}
        for value, position, label in zip(
            expected_values.ravel(), gain_plot.bar_positions().ravel(), axis.texts, strict=True
        ):
            if value >= 0:
                assert 0 < label.get_position()[1] - value <= 0.13 + 1e-10
                assert abs(label.get_position()[0] - position) < gain_plot.BAR_WIDTH / 2
            else:
                assert label.get_position()[0] == pytest.approx(position)
        assert expected_truncated.sum() == 3
        assert expected_truncated[0, 0]
        assert expected_truncated[0, 4]
        assert expected_truncated[1, 0]
        assert abs(observed_heights[4]) < abs(observed_heights[6]) < abs(observed_heights[0])
        assert observed_heights[0] != pytest.approx(expected_values[0, 0])
        assert sum(line.get_gid() == "truncated-bar-break" for line in axis.lines) == 6

        positions = gain_plot.bar_positions()
        within_group_step = np.diff(positions, axis=1)
        between_group_gap = positions[1, 0] - positions[0, -1]
        assert within_group_step == pytest.approx(gain_plot.BAR_CENTER_STEP)
        assert between_group_gap > 2 * gain_plot.BAR_WIDTH

        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        for value, patch, label in zip(expected_values.ravel(), axis.patches, axis.texts, strict=True):
            label_box = label.get_window_extent(renderer)
            patch_box = patch.get_window_extent(renderer)
            if value >= 0:
                assert label_box.y0 > patch_box.y1
            else:
                assert label_box.y1 < patch_box.y0
        label_bounds = [label.get_window_extent(renderer).padded(2) for label in axis.texts]
        assert all(
            not first.overlaps(second)
            for index, first in enumerate(label_bounds)
            for second in label_bounds[index + 1 :]
        )
        assert all(
            not label.overlaps(patch.get_window_extent(renderer)) for label in label_bounds for patch in axis.patches
        )
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
        assert len(figure.legends) == 0
        axis = figure.axes[0]
        legend_labels = [text.get_text() for text in axis.texts]
        assert legend_labels == [method.legend_label for method in gain_plot.METHODS]
        assert len(axis.patches) == 6
        assert [method.legend_label.endswith(gain_plot.DAGGER) for method in gain_plot.METHODS] == [
            False,
            True,
            True,
            True,
            True,
            False,
        ]
        observed_colors = [handle.get_facecolor() for handle in axis.patches]
        expected_colors = [mcolors.to_rgba(method.color) for method in gain_plot.METHODS]
        assert observed_colors == pytest.approx(expected_colors)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        for column in range(3):
            assert axis.patches[column].get_x() == pytest.approx(axis.patches[column + 3].get_x())
            assert axis.texts[column].get_position()[0] == pytest.approx(axis.texts[column + 3].get_position()[0])
        patch_bounds = [patch.get_window_extent(renderer) for patch in axis.patches]
        text_bounds = [text.get_window_extent(renderer) for text in axis.texts]
        assert all(bounds.width == pytest.approx(bounds.height, abs=1.0) for bounds in patch_bounds)
        assert all(figure.bbox.contains(*bounds.get_points()[0]) for bounds in text_bounds)
        assert all(figure.bbox.contains(*bounds.get_points()[1]) for bounds in text_bounds)
        assert all(
            not first.overlaps(second) for index, first in enumerate(text_bounds) for second in text_bounds[index + 1 :]
        )
        assert all(not label.overlaps(patch) for label in text_bounds for patch in patch_bounds)
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
    assert (float(media_box.group(1)), float(media_box.group(2))) == pytest.approx((288.0, 216.0))
