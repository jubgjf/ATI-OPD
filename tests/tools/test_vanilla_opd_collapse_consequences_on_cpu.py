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

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from analysis import plot_style
from analysis import vanilla_opd_collapse_consequences as collapse


def _spec(metric: str = "metric/test", *, percent: bool = False) -> collapse.MetricSpec:
    return collapse.MetricSpec(
        metric=metric,
        stem="test",
        expected_steps=(1, 2, 3),
        smoothing_window=None,
        scale=100.0 if percent else 1.0,
        percent=percent,
        zero_baseline=True,
    )


def _rows(spec: collapse.MetricSpec) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "global_step": spec.expected_steps,
            "metric_key": [spec.metric] * len(spec.expected_steps),
            "value": [0.1, 0.3, 0.8],
        }
    )


def test_validate_metric_rows_preserves_exact_real_observations() -> None:
    spec = _spec()
    frame = collapse.validate_metric_rows(_rows(spec), (spec,))[spec.metric]

    assert frame["step"].tolist() == [1, 2, 3]
    assert frame["value"].tolist() == pytest.approx([0.1, 0.3, 0.8])


@pytest.mark.parametrize("failure", ["missing", "duplicate", "nonfinite"])
def test_validate_metric_rows_rejects_invalid_series(failure: str) -> None:
    spec = _spec()
    rows = _rows(spec)
    if failure == "missing":
        rows = rows.iloc[:-1]
        match = "Unexpected sampling steps"
    elif failure == "duplicate":
        rows = pd.concat((rows, rows.iloc[[0]]), ignore_index=True)
        match = "duplicate steps"
    else:
        rows.loc[1, "value"] = np.nan
        match = "non-finite"

    with pytest.raises(ValueError, match=match):
        collapse.validate_metric_rows(rows, (spec,))


def test_value_colored_line_maps_low_blue_and_high_red() -> None:
    fig, ax = plt.subplots()
    collection = collapse.add_value_colored_line(
        ax,
        np.asarray((1.0, 2.0, 3.0)),
        np.asarray((0.0, 0.5, 1.0)),
    )
    try:
        colormap = collection.get_cmap()
        normalizer = collection.norm
        assert mpl.colors.to_hex(colormap(normalizer(0.0))).upper() == collapse.SLICE_COLORMAP_COLORS[0]
        assert mpl.colors.to_hex(colormap(normalizer(1.0))).upper() == collapse.SLICE_COLORMAP_COLORS[-1]
    finally:
        plt.close(fig)


def test_prepare_curve_uses_slice_style_corner_cutting() -> None:
    spec = _spec()
    frame = collapse.validate_metric_rows(_rows(spec), (spec,))[spec.metric]
    steps, values = collapse.prepare_curve(frame, spec)

    assert len(steps) == len(frame) * 2**collapse.CURVE_SMOOTHING_PASSES
    assert steps[0] == pytest.approx(frame["step"].iloc[0])
    assert steps[-1] == pytest.approx(frame["step"].iloc[-1])
    assert values[0] == pytest.approx(frame["value"].iloc[0])
    assert values[-1] == pytest.approx(frame["value"].iloc[-1])
    assert np.all(np.diff(steps) > 0)


def test_metric_panel_is_three_by_two_without_axis_name_text() -> None:
    plot_style.configure_style()
    spec = _spec(percent=True)
    frame = collapse.validate_metric_rows(_rows(spec), (spec,))[spec.metric]
    fig = collapse.create_metric_figure(frame, spec)
    try:
        fig.canvas.draw()
        ax = fig.axes[0]
        assert tuple(fig.get_size_inches()) == pytest.approx(collapse.FIGSIZE)
        assert ax.get_xlabel() == ""
        assert ax.get_ylabel() == ""
        assert ax.get_title() == ""
        assert tuple(ax.get_xticks()) == (1, 80, 140, 217)
        assert all(label.get_text().endswith("%") for label in ax.get_yticklabels())
        assert plot_style.panel_layout_overflow(fig) == ()
    finally:
        plt.close(fig)


def test_saved_panel_has_exact_three_by_two_canvas(tmp_path) -> None:
    plot_style.configure_style()
    spec = _spec()
    frame = collapse.validate_metric_rows(_rows(spec), (spec,))[spec.metric]
    collapse.save_outputs(collapse.create_metric_figure(frame, spec), tmp_path, "panel")

    pdf_bytes = (tmp_path / "panel.pdf").read_bytes()
    media_box = re.search(
        rb"/MediaBox\s*\[\s*0\s+0\s+([0-9.]+)\s+([0-9.]+)\s*\]",
        pdf_bytes,
    )
    assert media_box is not None
    assert (float(media_box.group(1)), float(media_box.group(2))) == pytest.approx((216.0, 144.0))
    assert (tmp_path / "panel.png").is_file()
