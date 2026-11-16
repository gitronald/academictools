"""seaborn's defaults and arguments on top of the drawing algorithms.

Values come from ``fixtures/parity_reference.json``, recorded from seaborn
0.13.2 by ``fixtures/record_reference.py`` on the same synthetic frame built
here.
"""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest
from matplotlib.collections import PatchCollection, PathCollection
from matplotlib.colors import to_rgb, to_rgba, to_rgba_array
from matplotlib.lines import Line2D

from tools.plots import add_legend
from tools.plots.charts import (
    barplot,
    boxenplot,
    boxplot,
    histplot,
    pointplot,
    scatterplot,
    swarmplot,
)
from tools.plots.palettes import (
    DEEP,
    color_palette,
    dark_palette,
    desaturate,
    light_palette,
    resolve_colors,
    set_theme,
)

PARITY = json.loads(
    (Path(__file__).parent / "fixtures" / "parity_reference.json").read_text()
)
PALETTE = ["#ff7f0e", "#1f77b4", "#2ca02c"]


@pytest.fixture(autouse=True)
def _clean_state():
    mpl.rcdefaults()
    yield
    plt.close("all")
    mpl.rcdefaults()


@pytest.fixture
def frame() -> pl.DataFrame:
    """The recorder's frame: the same generator calls in the same order."""
    rng = np.random.default_rng(11)
    n = 300
    return pl.DataFrame(
        {
            "cat": rng.choice(["a", "b", "c"], n),
            "grp": rng.choice(["p", "q"], n),
            "val": rng.beta(4, 2, n),
        }
    )


def _drawn(artists) -> list:
    """Artists that are drawn, leaving out the labeled legend artists."""
    return [a for a in artists if not a.get_label() or a.get_label().startswith("_")]


class TestColors:
    @pytest.mark.parametrize("case", PARITY["desaturate"])
    def test_desaturate(self, case):
        np.testing.assert_allclose(
            desaturate(case["color"], case["prop"]), case["rgb"], atol=1e-12
        )

    def test_desaturate_rejects_bad_prop(self):
        with pytest.raises(ValueError, match="prop"):
            desaturate("red", 1.5)

    @pytest.mark.parametrize("case", PARITY["palettes"], ids=lambda c: c["name"])
    def test_dark_and_light_palettes(self, case):
        colors = color_palette(case["name"], case["n"])
        np.testing.assert_allclose(colors, case["colors"], atol=1e-12)

    def test_palette_functions_and_resolve(self):
        assert dark_palette("red", 3) == color_palette("dark:red", 3)
        assert light_palette("red", 3, reverse=True) == color_palette("light:red_r", 3)
        assert resolve_colors(["x", "y"], "dark:red") == color_palette("dark:red", 2)

    @pytest.mark.parametrize("style", ["ticks", "whitegrid"])
    def test_set_theme_matches_seaborn(self, style):
        set_theme(style)
        for key, value in PARITY["themes"][style].items():
            got = mpl.rcParams[key]
            if key == "axes.prop_cycle":
                got = [list(to_rgb(c["color"])) for c in got]
                np.testing.assert_allclose(got, value)
            elif isinstance(value, list):
                assert list(got) == value, key
            else:
                assert got == value, key

    def test_set_theme_cycle_and_bad_style(self):
        set_theme()
        cycle = [c["color"] for c in mpl.rcParams["axes.prop_cycle"]]
        assert cycle == DEEP
        with pytest.raises(ValueError, match="style"):
            set_theme("dark")  # type: ignore[arg-type]


class TestBars:
    @pytest.mark.parametrize("case", PARITY["bars"], ids=lambda c: c["case"])
    def test_geometry_fill_and_legend(self, frame, case):
        kws = dict(case["kws"])
        ax = barplot(
            frame,
            "cat",
            "val",
            order=["a", "b", "c"],
            errorbar=None,
            palette=PALETTE if "hue" in kws else None,
            **kws,
        )
        bars = sorted(
            (p for p in _drawn(ax.patches) if p.get_height() > 0),
            key=lambda p: p.get_x(),
        )
        expected = sorted(
            zip(case["x"], case["width"], case["facecolor"], strict=True),
            key=lambda t: t[0],
        )
        np.testing.assert_allclose(
            [p.get_x() for p in bars], [e[0] for e in expected], atol=1e-12
        )
        np.testing.assert_allclose(
            [p.get_width() for p in bars], [e[1] for e in expected]
        )
        np.testing.assert_allclose(
            [to_rgb(p.get_facecolor()) for p in bars], [e[2] for e in expected]
        )
        legend = ax.get_legend()
        texts = [t.get_text() for t in legend.texts] if legend else []
        assert texts == case["legend"]

    def test_legend_explicit_true_for_redundant_hue(self, frame):
        ax = barplot(frame, "cat", "val", hue="cat", errorbar=None, legend=True)
        assert ax.get_legend() is not None


class TestLegendArtists:
    @pytest.mark.parametrize(
        "draw",
        [
            lambda df, ax: barplot(df, "cat", "val", hue="grp", errorbar=None, ax=ax),
            lambda df, ax: pointplot(df, "cat", "val", hue="grp", errorbar=None, ax=ax),
            lambda df, ax: boxplot(df, "cat", "val", hue="grp", ax=ax),
            lambda df, ax: boxenplot(df, "cat", "val", hue="grp", ax=ax),
            lambda df, ax: swarmplot(df, "cat", "val", hue="grp", size=2, ax=ax),
            lambda df, ax: scatterplot(df, "val", "val", hue="grp", ax=ax),
            lambda df, ax: histplot(df, "val", hue="grp", ax=ax),
        ],
        ids=["bar", "point", "box", "boxen", "swarm", "scatter", "hist"],
    )
    def test_add_legend_finds_hue_levels(self, frame, draw):
        _, ax = plt.subplots()
        draw(frame, ax)
        _, labels = ax.get_legend_handles_labels()
        assert labels == ["p", "q"]
        legend = add_legend(ax)
        assert [t.get_text() for t in legend.get_texts()] == ["p", "q"]

    def test_no_legend_artists_without_legend(self, frame):
        ax = barplot(frame, "cat", "val", hue="grp", errorbar=None, legend=False)
        assert ax.get_legend_handles_labels() == ([], [])
        assert ax.get_legend() is None


class TestScatter:
    @pytest.mark.parametrize("case", PARITY["scatter"], ids=lambda c: str(c["sizes"]))
    def test_marker_edges(self, frame, case):
        sizes = case["sizes"]
        s = sizes[0] if len(sizes) == 1 else np.resize(sizes, frame.height)
        ax = scatterplot(frame, "val", "val", s=s)
        points = ax.collections[0]
        np.testing.assert_allclose(
            np.asarray(points.get_linewidth()), case["linewidths"]
        )
        np.testing.assert_allclose(
            np.asarray(points.get_edgecolor())[0], case["edgecolor"]
        )

    def test_explicit_edges_win(self, frame):
        ax = scatterplot(frame, "val", "val", s=4, edgecolor="black", linewidth=0.5)
        points = ax.collections[0]
        np.testing.assert_allclose(np.asarray(points.get_linewidth()), [0.5])
        np.testing.assert_allclose(np.asarray(points.get_edgecolor())[0], [0, 0, 0, 1])

    def test_unfilled_marker_keeps_face_edges(self, frame):
        ax = scatterplot(frame, "val", "val", marker="x")
        points = ax.collections[0]
        assert not np.allclose(np.asarray(points.get_edgecolor())[0][:3], [1, 1, 1])


class TestHistogram:
    @pytest.mark.parametrize("case", PARITY["hist"], ids=lambda c: str(c["bins"]))
    def test_edge_width_scales_with_bins(self, frame, case):
        _, ax = plt.subplots(figsize=tuple(case["figsize"]), dpi=case["dpi"])
        histplot(frame, "val", bins=case["bins"], ax=ax)
        assert ax.patches[0].get_linewidth() == pytest.approx(case["linewidth"])

    def test_first_level_on_top(self, frame):
        ax = histplot(frame, "val", hue="grp", hue_order=["p", "q"], bins=5)
        got = [list(to_rgb(p.get_facecolor())) for p in _drawn(ax.patches)[:10]]
        np.testing.assert_allclose(got, PARITY["hist_layers"])

    def test_alpha_fades_fill_not_edges(self, frame):
        ax = histplot(frame, "val", bins=5)
        bar = ax.patches[0]
        assert to_rgba(bar.get_facecolor())[3] == pytest.approx(0.75)
        assert to_rgba(bar.get_edgecolor())[3] == pytest.approx(1.0)

    def test_explicit_linewidth_kept(self, frame):
        ax = histplot(frame, "val", bins=80, linewidth=2)
        assert ax.patches[0].get_linewidth() == 2


class TestBoxes:
    @pytest.mark.parametrize(
        "case",
        [c for c in PARITY["line_colors"] if "color" in c],
        ids=lambda c: f"{c['plot']}-{c['color']}",
    )
    def test_auto_line_color(self, frame, case):
        if case["plot"] == "boxplot":
            ax = boxplot(frame, "cat", "val", color=case["color"])
            got = ax.lines[0].get_color()
        else:
            ax = boxenplot(frame, "cat", "val", color=case["color"])
            got = ax.collections[0].get_edgecolor()[0]
        np.testing.assert_allclose(to_rgba_array(got)[0, :3], case["line"], atol=1e-12)

    def test_auto_line_color_with_hue(self, frame):
        case = next(c for c in PARITY["line_colors"] if "palette" in c)
        ax = boxplot(frame, "cat", "val", hue="grp", palette=case["palette"])
        np.testing.assert_allclose(to_rgb(ax.lines[0].get_color()), case["line"])

    def test_boxplot_unfilled(self, frame):
        ax = boxplot(frame, "cat", "val", color="black", fill=False, linewidth=0.65)
        assert _drawn(ax.patches) == []
        for line in ax.lines:
            assert to_rgb(line.get_color()) == (0.0, 0.0, 0.0)

    def test_boxplot_and_boxen_gap(self, frame):
        ax = boxplot(frame, "cat", "val", hue="grp", gap=0.5, legend=False)
        box = _drawn(ax.patches)[0]
        x0, x1 = (
            box.get_path().vertices[:, 0].min(),
            box.get_path().vertices[:, 0].max(),
        )
        assert x1 - x0 == pytest.approx(0.4 * 0.5)
        ax = boxenplot(frame, "cat", "val", hue="grp", gap=0.5, legend=False)
        boxes = next(c for c in ax.collections if isinstance(c, PatchCollection))
        widest = max(
            float(np.ptp(np.asarray(path.vertices)[:, 0])) for path in boxes.get_paths()
        )
        assert widest == pytest.approx(0.4 * 0.5)

    def test_boxen_element_kws(self, frame):
        ax = boxenplot(
            frame,
            "cat",
            "val",
            box_kws={"edgecolor": "black", "linewidth": 0.45},
            line_kws={"color": "red", "linewidth": 0.65},
            flier_kws={"marker": ".", "s": 2, "edgecolor": "black"},
        )
        boxes = [c for c in ax.collections if isinstance(c, PatchCollection)]
        np.testing.assert_allclose(
            np.asarray(boxes[0].get_edgecolor())[0], [0, 0, 0, 1]
        )
        np.testing.assert_allclose(boxes[0].get_linewidth(), [0.45])
        medians = [line for line in ax.lines if isinstance(line, Line2D)]
        assert to_rgb(medians[0].get_color()) == (1.0, 0.0, 0.0)
        assert medians[0].get_linewidth() == 0.65
        fliers = [c for c in ax.collections if isinstance(c, PathCollection)]
        assert fliers and np.allclose(fliers[0].get_sizes(), [2])


class TestDodgeAndSwarm:
    def test_auto_dodge_only_when_levels_share_a_category(self, frame):
        _, ax = plt.subplots()
        boxplot(frame, "cat", "val", hue="cat", ax=ax)
        assert ax.get_legend() is None
        verts = _drawn(ax.patches)[0].get_path().vertices[:, 0]
        assert np.ptp(verts) == pytest.approx(0.8)  # full width, not dodged
        _, ax2 = plt.subplots()
        boxplot(frame, "cat", "val", hue="grp", legend=False, ax=ax2)
        assert len(_drawn(ax2.patches)) == 6

    def test_swarm_points_above_bars(self, frame):
        ax = swarmplot(frame, "cat", "val", size=2)
        assert ax.collections[0].get_zorder() == 3


class TestReviewFollowUp:
    def test_scatter_face_edge_with_hue_draws(self, frame):
        ax = scatterplot(frame, "val", "val", hue="grp", edgecolor="face")
        ax.figure.canvas.draw()
        handles, _ = ax.get_legend_handles_labels()
        line = handles[0]
        assert isinstance(line, Line2D)
        assert to_rgba(line.get_markeredgecolor()) == to_rgba(line.get_color())

    def test_scatter_aliases_suppress_defaults(self, frame):
        ax = scatterplot(frame, "val", "val", s=4, ec="black", lw=0.5)
        points = ax.collections[0]
        np.testing.assert_allclose(np.asarray(points.get_linewidth()), [0.5])
        np.testing.assert_allclose(np.asarray(points.get_edgecolor())[0], [0, 0, 0, 1])

    def test_hist_aliases(self, frame):
        ax = histplot(frame, "val", bins=80, lw=2, ec="black")
        bar = ax.patches[0]
        assert bar.get_linewidth() == 2
        assert to_rgba(bar.get_edgecolor()) == (0.0, 0.0, 0.0, 1.0)

    @pytest.mark.parametrize(
        "draw",
        [
            lambda df: barplot(df, "cat", "val", hue="grp", errorbar=None),
            lambda df: scatterplot(df, "val", "val", hue="grp"),
            lambda df: histplot(df, "val", hue="grp", bins=3),
        ],
        ids=["bar", "scatter", "hist"],
    )
    def test_empty_hue_frame_has_no_legend(self, frame, draw):
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("error", message="No artists with labels")
            ax = draw(frame.clear())
        assert ax.get_legend() is None
