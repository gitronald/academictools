import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest
from matplotlib import MatplotlibDeprecationWarning
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection, PathCollection, QuadMesh
from matplotlib.colors import LogNorm, to_rgb
from matplotlib.patches import Rectangle

from tools.plots._common import category_levels, complete_rows, single_color
from tools.plots._seaborn_port import Beeswarm, letter_values
from tools.plots.charts import (
    _dodge_offsets,
    barplot,
    boxenplot,
    boxplot,
    heatmap,
    histplot,
    pointplot,
    regplot,
    scatterplot,
    swarmplot,
)
from tools.plots.estimate import bootstrap, estimate_interval, wilson_interval
from tools.plots.palettes import (
    TICKS_STYLE,
    WHITEGRID_STYLE,
    color_palette,
    resolve_colors,
)
from tools.plots.settings import box_kws, heatmap_kws_proportion, pointplot_kws

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


BEESWARM = _load("beeswarm_reference.json")
LETTER_VALUES = _load("letter_values_reference.json")
PALETTES = _load("palettes_reference.json")


def _is_legend_artist(artist) -> bool:
    """Legend artists carry a level's label; drawn ones have none or "_child"."""
    label = artist.get_label()
    return bool(label) and not label.startswith("_")


def _bars(ax: Axes) -> list[Rectangle]:
    return [
        p for p in ax.patches if isinstance(p, Rectangle) and not _is_legend_artist(p)
    ]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def groups_df():
    rng = np.random.default_rng(0)
    return pl.DataFrame(
        {
            "cat": ["a"] * 40 + ["b"] * 40 + ["c"] * 40,
            "val": np.concatenate(
                [rng.normal(0, 1, 40), rng.normal(2, 1, 40), rng.normal(1, 2, 40)]
            ),
            "grp": (["g1", "g2"] * 20) * 3,
        }
    )


class TestPalettes:
    @pytest.mark.parametrize(
        "case", PALETTES["palettes"], ids=lambda c: f"{c['name']}-{c['n']}"
    )
    def test_matches_seaborn_color_palette(self, case):
        colors = color_palette(case["name"], case["n"])
        np.testing.assert_allclose(colors, case["colors"], rtol=0, atol=1e-12)

    @pytest.mark.parametrize(
        ("style", "ours"), [("ticks", TICKS_STYLE), ("whitegrid", WHITEGRID_STYLE)]
    )
    def test_style_matches_seaborn_except_rocket_cmap(self, style, ours):
        expected = dict(PALETTES["styles"][style])
        # seaborn's own colormap; matplotlib does not ship it
        assert expected.pop("image.cmap") == "rocket"
        assert ours == expected

    def test_styles_are_valid_rcparams(self):
        with plt.rc_context(TICKS_STYLE), plt.rc_context(WHITEGRID_STYLE):
            assert plt.rcParams["axes.grid"] is True

    def test_resolve_colors_forms(self):
        levels = ["a", "b", "c"]
        default = resolve_colors(levels)
        assert default[0] == to_rgb("C0")
        assert resolve_colors(levels, "viridis") == color_palette("viridis", 3)
        assert resolve_colors(levels, ["red", "blue"])[2] == to_rgb("red")
        mapped = resolve_colors(levels, {"a": "k", "b": "w", "c": "r"})
        assert mapped == [to_rgb("k"), to_rgb("w"), to_rgb("r")]


class TestLetterValues:
    @pytest.mark.parametrize(
        "case",
        LETTER_VALUES["cases"],
        ids=lambda c: f"{c['dataset']}-{c['k_depth']}",
    )
    def test_matches_seaborn(self, case):
        x = np.asarray(LETTER_VALUES["datasets"][case["dataset"]])
        lv = letter_values(x, case["k_depth"])
        assert lv["k"] == case["k"]
        assert lv["levels"].tolist() == case["levels"]
        np.testing.assert_array_equal(lv["percs"], case["percs"])
        np.testing.assert_array_equal(lv["values"], case["values"])
        np.testing.assert_array_equal(lv["fliers"], case["fliers"])
        assert lv["median"] == case["median"]

    def test_rejects_unknown_depth(self):
        with pytest.raises(ValueError, match="k_depth"):
            letter_values(np.arange(10.0), "bogus")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="k_depth"):
            letter_values(np.arange(10.0), 2.5)  # type: ignore[arg-type]


class TestBoxenWidths:
    @pytest.mark.parametrize(
        "case",
        [c for c in LETTER_VALUES["cases"] if c["dataset"] == "normal_1000"],
        ids=lambda c: str(c["k_depth"]),
    )
    @pytest.mark.parametrize("method", ["exponential", "linear", "area"])
    def test_box_depths_and_widths_match_seaborn(self, case, method):
        x = LETTER_VALUES["datasets"][case["dataset"]]
        df = pl.DataFrame({"c": ["a"] * len(x), "v": x})
        ax = boxenplot(df, "c", "v", k_depth=case["k_depth"], width_method=method)
        boxes = [c for c in ax.collections if isinstance(c, PatchCollection)][0]
        paths = boxes.get_paths()
        assert len(paths) == 2 * case["k"] - 1
        extents = [p.get_extents() for p in paths]
        np.testing.assert_allclose([e.width for e in extents], case["widths"][method])
        np.testing.assert_allclose(
            [e.y0 for e in extents], case["values"][:-1], rtol=1e-12
        )
        np.testing.assert_allclose(
            [e.y1 for e in extents], case["values"][1:], rtol=1e-12
        )


class TestBeeswarm:
    @pytest.mark.parametrize("case", BEESWARM["algorithm"])
    def test_algorithm_matches_seaborn(self, case):
        out = Beeswarm().beeswarm(np.asarray(case["input"]))
        np.testing.assert_array_equal(out, case["output"])

    @pytest.mark.parametrize(
        "case", BEESWARM["plots"], ids=lambda c: f"{c['orient']}-{c['size']}"
    )
    def test_swarmplot_matches_seaborn(self, case):
        groups = case["groups"]
        df = pl.DataFrame(
            {
                "cat": [k for k, v in groups.items() for _ in v],
                "val": [x for v in groups.values() for x in v],
            }
        )
        fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
        if case["orient"] == "x":
            swarmplot(df, "cat", "val", size=case["size"], ax=ax)
        else:
            swarmplot(df, "val", "cat", orient="y", size=case["size"], ax=ax)
        fig.canvas.draw()
        np.testing.assert_allclose(ax.get_xlim(), case["xlim"])
        np.testing.assert_allclose(ax.get_ylim(), case["ylim"])
        for points, expected in zip(ax.collections, case["offsets"], strict=True):
            np.testing.assert_allclose(
                np.asarray(points.get_offsets()), expected, atol=1e-12
            )

    def test_crowded_swarm_warns(self):
        df = pl.DataFrame({"c": ["a"] * 400, "v": np.zeros(400)})
        fig, ax = plt.subplots(figsize=(2, 2))
        with pytest.warns(UserWarning, match="cannot be placed"):
            swarmplot(df, "c", "v", size=8, ax=ax)

    def test_hue_dodge_makes_one_swarm_per_level(self, groups_df):
        ax = swarmplot(groups_df, "cat", "val", hue="grp", dodge=True)
        points = [c for c in ax.collections if isinstance(c, PathCollection)]
        assert len(points) == 6
        legend = ax.get_legend()
        assert legend is not None
        assert [t.get_text() for t in legend.get_texts()] == ["g1", "g2"]


class TestEstimate:
    def test_seeded_bootstrap_is_reproducible(self):
        x = np.arange(20.0)
        a = bootstrap(x, func=np.mean, n_boot=50, seed=3)
        b = bootstrap(x, func=np.mean, n_boot=50, seed=3)
        np.testing.assert_array_equal(a, b)
        assert a.shape == (50,)

    def test_bootstrap_resamples_rows_jointly(self):
        x = np.arange(10.0)
        diffs = bootstrap(x, x * 2, func=lambda a, b: b - 2 * a, n_boot=20, seed=1)
        np.testing.assert_array_equal(diffs, np.zeros((20, 10)))

    def test_bootstrap_validates_inputs(self):
        with pytest.raises(ValueError, match="at least one"):
            bootstrap(func=np.mean)
        with pytest.raises(ValueError, match="same length"):
            bootstrap(np.arange(3), np.arange(4), func=np.mean)

    def test_ci_brackets_mean(self):
        x = np.random.default_rng(0).normal(5, 1, 200)
        est, low, high = estimate_interval(x, seed=0)
        assert est == pytest.approx(x.mean())
        assert low < est < high
        assert (high - low) == pytest.approx(2 * 1.96 / np.sqrt(200), rel=0.25)

    def test_sd_se_pi_and_none(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
        est, low, high = estimate_interval(x, errorbar="sd")
        sd = np.std([1, 2, 3, 4], ddof=1)
        assert (est, low, high) == pytest.approx((2.5, 2.5 - sd, 2.5 + sd))
        _, low, high = estimate_interval(x, errorbar=("se", 2))
        assert high - est == pytest.approx(2 * sd / 2)
        _, low, high = estimate_interval(x, errorbar=("pi", 50))
        assert (low, high) == pytest.approx((1.75, 3.25))
        est, low, high = estimate_interval(x, "median", errorbar=None)
        assert est == 2.5
        assert np.isnan(low) and np.isnan(high)

    def test_empty_and_bad_specs(self):
        assert all(np.isnan(estimate_interval(np.array([]))))
        with pytest.raises(ValueError, match="errorbar"):
            estimate_interval(np.arange(3.0), errorbar="iqr")
        with pytest.raises(ValueError, match="unknown estimator"):
            estimate_interval(np.arange(3.0), estimator="nope")

    def test_se_sd_wilson_require_mean(self):
        x = np.array([0.0, 1.0, 1.0, 0.0, 1.0])
        for spec in ("sd", ("se", 2), "wilson"):
            with pytest.raises(ValueError, match="describes the mean"):
                estimate_interval(x, "median", errorbar=spec)
        est, _, _ = estimate_interval(x, np.mean, errorbar="sd")
        assert est == 0.6

    def test_wilson_matches_the_closed_form(self):
        # 0 of 13: the upper bound is z^2 / (n + z^2) and the lower bound is 0.
        z = 1.959963984540054
        est, low, high = estimate_interval(np.zeros(13), errorbar="wilson")
        assert (est, low, high) == pytest.approx((0.0, 0.0, z**2 / (13 + z**2)))
        # 1 of 2 stays inside [0, 1], where ("se", 1.96) does not.
        _, low, high = estimate_interval(np.array([0.0, 1.0]), errorbar="wilson")
        assert 0 < low < 0.5 < high < 1
        assert low + high == pytest.approx(1.0)
        _, se_low, _ = estimate_interval(np.array([0.0, 1.0]), errorbar=("se", 1.96))
        assert se_low < 0

    def test_wilson_level_and_booleans(self):
        x = np.array([True] * 30 + [False] * 70)
        _, low95, high95 = estimate_interval(x, errorbar=("wilson", 95))
        _, low80, high80 = estimate_interval(x, errorbar=("wilson", 80))
        assert low95 < low80 < 0.3 < high80 < high95
        assert (low95, high95) == pytest.approx(wilson_interval(x.astype(float)))
        # At large n it agrees with the normal approximation.
        big = np.random.default_rng(0).random(200_000) < 0.3
        wil = estimate_interval(big, errorbar="wilson")
        se = estimate_interval(big, errorbar=("se", 1.959963984540054))
        assert wil == pytest.approx(se, abs=1e-5)

    def test_wilson_rejects_values_other_than_0_and_1(self):
        with pytest.raises(ValueError, match="all 0 or 1"):
            estimate_interval(np.array([0.0, 0.5, 1.0]), errorbar="wilson")
        with pytest.raises(ValueError, match="at least one value"):
            wilson_interval(np.array([]))


class TestBarAndPointPlots:
    def test_bar_heights_are_means(self, groups_df):
        ax = barplot(groups_df, "cat", "val", seed=0)
        heights = [p.get_height() for p in _bars(ax)]
        expected = (
            groups_df.group_by("cat", maintain_order=True)
            .agg(pl.col("val").mean())
            .get_column("val")
            .to_list()
        )
        assert heights == pytest.approx(expected)
        assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b", "c"]
        assert ax.get_xlim() == (-0.5, 2.5)

    def test_bar_seed_reproduces_intervals(self, groups_df):
        def interval_lines(seed):
            fig, ax = plt.subplots()
            barplot(groups_df, "cat", "val", seed=seed, ax=ax)
            return [np.asarray(line.get_ydata()).tolist() for line in ax.lines]

        assert interval_lines(4) == interval_lines(4)
        assert interval_lines(4) != interval_lines(5)

    def test_bar_hue_dodges_and_caps(self, groups_df):
        ax = barplot(
            groups_df, "cat", "val", hue="grp", capsize=0.2, seed=0, order=["c", "a"]
        )
        assert len(_bars(ax)) == 4
        widths = {round(p.get_width(), 6) for p in _bars(ax)}
        assert widths == {0.4}
        # one interval line plus two caps per bar
        assert len(ax.lines) == 12
        assert [t.get_text() for t in ax.get_xticklabels()] == ["c", "a"]

    def test_bar_horizontal_without_errorbar(self, groups_df):
        ax = barplot(groups_df, "val", "cat", orient="y", errorbar=None)
        assert len(ax.lines) == 0
        assert ax.get_ylim() == (2.5, -0.5)

    def test_point_plot_with_preset(self, groups_df):
        ax = pointplot(groups_df, "cat", "val", hue="grp", seed=0, **pointplot_kws)
        point_lines = [
            line
            for line in ax.lines
            if line.get_marker() == "." and not _is_legend_artist(line)
        ]
        assert len(point_lines) == 2
        xs = np.asarray(point_lines[0].get_xdata())
        # dodge=0.2 over two levels -> -0.1 and +0.1 around each category
        np.testing.assert_allclose(xs, [-0.1, 0.9, 1.9])

    def test_point_plot_dodge_true_and_marker_list(self, groups_df):
        ax = pointplot(
            groups_df, "cat", "val", hue="grp", dodge=True, markers=["o", "s"]
        )
        markers = {line.get_marker() for line in ax.lines}
        assert {"o", "s"} <= markers


class TestOtherHelpers:
    def test_regplot_line_and_band(self):
        x = np.linspace(0, 10, 50)
        df = pl.DataFrame({"x": x, "y": 2 * x + 1})
        ax = regplot(df, "x", "y", seed=0, n_boot=100)
        line = ax.lines[0]
        xs = np.asarray(line.get_xdata(), dtype=float)
        np.testing.assert_allclose(np.asarray(line.get_ydata()), 2 * xs + 1)
        assert len(ax.collections) == 2  # scatter + band

    def test_regplot_no_ci_untruncated(self):
        df = pl.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.5]})
        ax = regplot(df, "x", "y", ci=None, truncate=False)
        assert len(ax.collections) == 1

    def test_scatter_hue_legend(self, groups_df):
        ax = scatterplot(groups_df, "val", "val", hue="grp", palette="Set2")
        # one collection, colored row by row, as seaborn draws it
        assert len(ax.collections) == 1
        kept = groups_df.drop_nulls(["val", "grp"])
        set2 = dict(zip(["g1", "g2"], color_palette("Set2", 2), strict=True))
        expected = [set2[g] for g in kept["grp"].to_list()]
        np.testing.assert_allclose(
            np.asarray(ax.collections[0].get_facecolor())[:, :3], expected
        )
        legend = ax.get_legend()
        assert legend is not None
        assert legend.get_title().get_text() == "grp"

    @pytest.mark.parametrize(
        ("stat", "total"), [("probability", 1.0), ("percent", 100.0), ("count", 120)]
    )
    def test_hist_stat_normalizes_over_all_levels(self, groups_df, stat, total):
        ax = histplot(groups_df, "val", hue="grp", stat=stat, bins=10)
        heights = sum(p.get_height() for p in _bars(ax))
        assert heights == pytest.approx(total)

    def test_hist_density_and_stack(self, groups_df):
        ax = histplot(groups_df, "val", hue="grp", stat="density", multiple="stack")
        area = sum(p.get_height() * p.get_width() for p in _bars(ax))
        assert area == pytest.approx(1.0)
        fig, ax = plt.subplots()
        histplot(groups_df, "val", stat="frequency", bins=[-5, 0, 5], ax=ax)
        assert len(ax.patches) == 2

    def test_hist_rejects_bad_options(self, groups_df):
        with pytest.raises(ValueError, match="stat"):
            histplot(groups_df, "val", stat="bogus")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="multiple"):
            histplot(groups_df, "val", multiple="dodge")  # type: ignore[arg-type]

    def test_boxplot_preset_and_no_deprecation(self, groups_df):
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatplotlibDeprecationWarning)
            ax = boxplot(groups_df, "cat", "val", **box_kws)
        assert len(ax.patches) == 3

    def test_boxplot_hue_horizontal(self, groups_df):
        ax = boxplot(groups_df, "val", "cat", hue="grp", orient="y")
        assert len([p for p in ax.patches if not _is_legend_artist(p)]) == 6
        assert [t.get_text() for t in ax.get_yticklabels()] == ["a", "b", "c"]

    def test_boxen_hue_and_rejects_bad_method(self, groups_df):
        ax = boxenplot(groups_df, "val", "cat", hue="grp", orient="y")
        boxes = [c for c in ax.collections if isinstance(c, PatchCollection)]
        assert len(boxes) == 6
        with pytest.raises(ValueError, match="width_method"):
            boxenplot(groups_df, "cat", "val", width_method="bogus")  # type: ignore[arg-type]

    def test_orient_validated(self, groups_df):
        with pytest.raises(ValueError, match="orient"):
            barplot(groups_df, "cat", "val", orient="z")  # type: ignore[arg-type]


class TestMissingAndOrder:
    @pytest.fixture
    def nan_df(self):
        return pl.DataFrame(
            {
                "cat": ["a", "a", "a", "b", "b", "b"],
                "x": [0.0, 1.0, float("nan"), 2.0, 3.0, 4.0],
                "y": [1.0, float("nan"), 3.0, 4.0, 5.0, 6.0],
            }
        )

    def test_complete_drops_nan_and_null(self, nan_df):
        df = nan_df.with_columns(pl.Series("g", ["u", None, "u", "u", "u", "u"]))
        assert complete_rows(df, "x", "y", "g").get_column("x").to_list() == [
            0.0,
            2.0,
            3.0,
            4.0,
        ]

    def test_hist_and_regplot_skip_nan(self, nan_df):
        ax = histplot(nan_df, "x", bins=4, stat="count")
        assert sum(p.get_height() for p in _bars(ax)) == 5
        regplot(nan_df, "x", "y", n_boot=50, seed=0)

    def test_regplot_needs_two_complete_rows(self):
        df = pl.DataFrame({"x": [1.0, None], "y": [None, 2.0]})
        with pytest.raises(ValueError, match="at least two rows"):
            regplot(df, "x", "y")

    def test_boxplot_statistics_ignore_nan(self, nan_df):
        ax = boxplot(nan_df, "cat", "y")
        medians = [np.asarray(ln.get_ydata()) for ln in ax.lines[4::6]]
        assert medians[0][0] == pytest.approx(2.0)
        assert medians[1][0] == pytest.approx(5.0)

    def test_enum_levels_follow_declared_order(self):
        levels = pl.Enum(["low", "mid", "high"])
        series = pl.Series(["high", "low", "mid", "high"], dtype=levels)
        assert category_levels(series, None) == ["low", "mid", "high"]
        df = pl.DataFrame({"c": series, "v": [1.0, 2.0, 3.0, 4.0]})
        ax = barplot(df, "c", "v", errorbar=None)
        assert [t.get_text() for t in ax.get_xticklabels()] == ["low", "mid", "high"]

    def test_numeric_and_boolean_levels_sort(self):
        assert category_levels(pl.Series([3, 1, 2]), None) == [1, 2, 3]
        assert category_levels(pl.Series([True, False]), None) == [False, True]
        assert category_levels(pl.Series(["b", "a"]), None) == ["b", "a"]

    @pytest.mark.parametrize(
        ("n_hue", "dodge", "each"), [(1, True, 0.8), (2, False, 0.8), (4, True, 0.2)]
    )
    def test_dodge_offsets_width(self, n_hue, dodge, each):
        offsets, width = _dodge_offsets(0.8, n_hue, dodge)
        assert len(offsets) == n_hue
        assert width == pytest.approx(each)
        assert offsets.mean() == pytest.approx(0.0)

    def test_single_color_default_and_explicit(self):
        assert single_color(None) == resolve_colors([0])[0]
        assert single_color("red") == to_rgb("red")


class TestHeatmap:
    def test_frame_labels_annotations_and_nan(self):
        df = pl.DataFrame({"row": ["r1", "r2"], "a": [0.1, None], "b": [0.5, 0.9]})
        ax = heatmap(df, index_col="row", annot=True, fmt=".1f", cmap="binary")
        mesh = ax.collections[0]
        assert isinstance(mesh, QuadMesh)
        assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b"]
        assert [t.get_text() for t in ax.get_yticklabels()] == ["r1", "r2"]
        # the null cell is masked and left unannotated
        assert sorted(t.get_text() for t in ax.texts) == ["0.1", "0.5", "0.9"]
        colors = {t.get_text(): t.get_color() for t in ax.texts}
        # dark cells get white text, light cells dark text
        assert colors["0.9"] == "w"
        assert colors["0.1"] == ".15"
        assert mesh.colorbar is not None

    def test_array_with_norm_square_and_no_labels(self):
        ax = heatmap(
            np.array([[1.0, 10.0], [100.0, 1000.0]]),
            norm=LogNorm(),
            square=True,
            cbar=False,
            xticklabels=False,
            yticklabels=["x", "y"],
        )
        assert ax.collections[0].colorbar is None
        assert ax.get_xticks().size == 0
        assert ax.get_aspect() == 1.0
        assert ax.get_ylim() == (2.0, 0.0)

    def test_proportion_preset(self):
        ax = heatmap(np.array([[0.2, 0.8]]), **heatmap_kws_proportion)
        assert ax.collections[0].get_clim() == (0, 1)
        assert len(ax.texts) == 2

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError, match="2-D"):
            heatmap(np.arange(3.0))


def test_helpers_draw_on_current_axes(groups_df):
    fig, ax = plt.subplots()
    assert barplot(groups_df, "cat", "val", errorbar=None) is ax
    assert isinstance(scatterplot(groups_df, "val", "val"), Axes)
