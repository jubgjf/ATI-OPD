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

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.backend_bases import KeyEvent
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from analysis import long_sequence_divergence_slices as slices
from analysis.plot_style import configure_style, panel_layout_overflow

plt.switch_backend("Agg")


def _synthetic_data() -> slices.DivergenceGapData:
    columns = list(range(0, slices.EXPECTED_RESPONSE_LENGTH, slices.EXPECTED_BIN_SIZE))
    bin_edges = np.arange(
        0,
        slices.EXPECTED_RESPONSE_LENGTH + slices.EXPECTED_BIN_SIZE,
        slices.EXPECTED_BIN_SIZE,
        dtype=float,
    )
    steps = np.asarray(slices.EXPECTED_GAP_STEPS, dtype=float)[:, None]
    positions = np.asarray(columns, dtype=float)[None, :]
    gap = 0.05 + 0.003 * steps + 0.00005 * positions
    gap[70:100, 24:] += 2.0
    gap_frame = pd.DataFrame(gap, index=slices.EXPECTED_GAP_STEPS, columns=columns)
    count_frame = pd.DataFrame(32.0, index=slices.EXPECTED_GAP_STEPS, columns=columns)
    return slices.DivergenceGapData(
        bin_edges=bin_edges,
        gap_abs=gap_frame,
        token_counts=count_frame,
    )


def test_response_position_slice_preparation_uses_coarse_training_trajectories() -> None:
    data = _synthetic_data()
    response_position_slices = slices.prepare_response_position_slice_data(data)

    assert response_position_slices.values.shape == (7, 22)
    assert np.all(np.diff(response_position_slices.training_steps) > 0)
    assert np.all(np.diff(response_position_slices.positions_k) > 0)


def test_slice_color_uses_one_global_scale_for_local_gap_height() -> None:
    slice_data = slices.prepare_response_position_slice_data(_synthetic_data())
    finite_values = slices._finite_gap_values(slice_data.values)
    normalizer = slices._gap_color_normalizer(slice_data.values)
    colormap = slices._gap_colormap()

    assert normalizer.vmin == 0.0
    assert normalizer.vmax == slices.COLOR_MAX_GAP
    assert normalizer.clip is True
    low_color = colormap(normalizer(finite_values.min()))
    high_color = colormap(normalizer(slices.COLOR_MAX_GAP))
    assert high_color[0] > low_color[0]
    assert high_color[2] < low_color[2]
    assert colormap(normalizer(slices.COLOR_MAX_GAP + 1.0)) == pytest.approx(high_color)
    repeated_height_colors = colormap(normalizer(np.asarray((1.0, 1.0))))
    assert repeated_height_colors[0] == pytest.approx(repeated_height_colors[1])


def test_curve_smoothing_rounds_corners_without_overshoot_or_bridging_missing_values() -> None:
    x = np.arange(5, dtype=float)
    values = np.asarray((0.0, 0.0, 4.0, 0.0, 0.0))

    smooth_x, smooth_values = slices._smooth_segment(x, values)

    assert len(smooth_x) == len(x) * 2**slices.CURVE_SMOOTHING_PASSES
    assert smooth_x[[0, -1]] == pytest.approx(x[[0, -1]])
    assert smooth_values[[0, -1]] == pytest.approx(values[[0, -1]])
    assert np.all(np.diff(smooth_x) > 0)
    assert 0 < smooth_values.max() < values.max()

    disconnected = slices._smoothed_finite_segments(x, np.asarray((0.0, 1.0, np.nan, 1.0, 0.0)))
    assert len(disconnected) == 2
    assert disconnected[0][0][-1] == pytest.approx(1.0)
    assert disconnected[1][0][0] == pytest.approx(3.0)


def test_slice_figure_colors_segments_by_local_height_and_uses_expected_axis_direction(tmp_path: Path) -> None:
    data = _synthetic_data()
    configure_style()
    response_figure = slices.create_response_position_figure(
        slices.prepare_response_position_slice_data(data),
        elevation=18.0,
        azimuth=-48.0,
        roll=3.0,
    )
    try:
        response_axis = response_figure.axes[0]
        fill_collections = [
            collection for collection in response_axis.collections if isinstance(collection, Poly3DCollection)
        ]
        line_collections = [
            collection for collection in response_axis.collections if isinstance(collection, Line3DCollection)
        ]
        assert len(fill_collections) == len(slices.SAMPLED_POSITION_INDICES)
        assert len(line_collections) == len(slices.SAMPLED_POSITION_INDICES)
        assert {collection.get_alpha() for collection in fill_collections} == {slices.SLICE_FILL_ALPHA}
        assert len(response_axis.lines) == 0
        for fill_collection, line_collection in zip(fill_collections, line_collections, strict=True):
            assert np.asarray(fill_collection.get_array()) == pytest.approx(np.asarray(line_collection.get_array()))
            assert np.ptp(np.asarray(line_collection.get_array())) > 0
            assert line_collection.norm is fill_collection.norm
            assert line_collection.cmap is fill_collection.cmap
        assert len(response_figure.axes) == 1
        assert response_axis.get_xlabel() == ""
        assert response_axis.get_ylabel() == ""
        assert response_axis.get_zlabel() == ""
        for axis in (response_axis.xaxis, response_axis.yaxis, response_axis.zaxis):
            assert {tick.get_pad() for tick in axis.get_major_ticks()} == {slices.TICK_LABEL_PAD}
        assert response_axis.get_ylim()[0] < response_axis.get_ylim()[1]
        assert [label.get_text() for label in response_axis.get_yticklabels()][-1] == "23K"
        assert slices._view_parameters(response_axis) == {
            "elevation": pytest.approx(18.0),
            "azimuth": pytest.approx(-48.0),
            "roll": pytest.approx(3.0),
        }
        assert panel_layout_overflow(response_figure, tolerance_points=1.0) == ()

        slices.save_outputs(response_figure, tmp_path, "position")
        assert (tmp_path / "position.pdf").is_file()
        assert (tmp_path / "position.png").is_file()
        assert json.loads((tmp_path / "position_view.json").read_text(encoding="utf-8")) == {
            "elevation": pytest.approx(18.0),
            "azimuth": pytest.approx(-48.0),
            "roll": pytest.approx(3.0),
        }
    finally:
        plt.close(response_figure)


def test_interactive_cli_options_and_backend_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    args = slices.parse_args(
        ["--interactive", "--interactive-backend", "ExampleGUI", "--elev", "12", "--azim", "34", "--roll", "5"]
    )
    assert args.interactive is True
    assert args.interactive_backend == "ExampleGUI"
    assert (args.elev, args.azim, args.roll) == pytest.approx((12.0, 34.0, 5.0))

    selected: list[str] = []
    monkeypatch.setattr(slices.plt, "switch_backend", selected.append)
    assert slices.activate_interactive_backend("ExampleGUI") == "ExampleGUI"
    assert selected == ["ExampleGUI"]

    with pytest.raises(SystemExit):
        slices.parse_args(["--interactive", "--validate-only"])


def test_interactive_keys_print_save_and_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_style()
    figure = slices.create_response_position_figure(slices.prepare_response_position_slice_data(_synthetic_data()))
    saved: list[tuple[Path, str]] = []

    def fake_write(_figure: object, output_dir: Path, stem: str) -> None:
        saved.append((output_dir, stem))

    def fake_show(*, block: bool) -> None:
        assert block is True
        for key in ("p", "s", "q"):
            event = KeyEvent("key_press_event", figure.canvas, key=key)
            figure.canvas.callbacks.process("key_press_event", event)

    monkeypatch.setattr(slices, "_write_outputs", fake_write)
    monkeypatch.setattr(slices.plt, "show", fake_show)
    slices.launch_interactive_view(figure, tmp_path)

    output = capsys.readouterr().out
    assert "Current view:" in output
    assert "Saved vanilla_opd_abs_gap_response_position_slices.pdf/.png" in output
    assert saved == [(tmp_path, slices.RESPONSE_POSITION_OUTPUT_STEM)]
