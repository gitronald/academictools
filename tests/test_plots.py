import datetime
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import pytest
from matplotlib import MatplotlibDeprecationWarning
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.lines import Line2D

from tools.plots.axes import (
    add_grid,
    add_grid_major,
    add_grid_minor,
    add_legend,
    add_right_angle,
    add_spines,
    remove_xaxis_ticks,
    remove_yaxis_ticks,
    reorder_legend,
    xaxis_date_formatter,
)
from tools.plots.core import (
    darken_hex_color,
    enhanced_boxplot,
    enhanced_lineplot,
    enhanced_scatterplot,
    noisy_limits,
    set_fig_text,
    smart_plot_data,
)
from tools.plots.heatmaps import get_heatmap_data, plot_heatmap


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def plot_df():
    return pl.DataFrame(
        {"x": [1, 2, 3, 4], "y": [4.0, 5.0, 6.0, 7.0], "g": ["a", "a", "b", "b"]}
    )


@pytest.fixture
def tweets():
    return pl.DataFrame(
        {
            "netloc": ["a.com"] * 6 + ["b.com"] * 4,
            "has_qry": [True, True, False, True, True, True, True, False, True, True],
            "year": [2020, 2020, 2020, 2021, 2021, 2021, 2020, 2020, 2021, 2021],
            "month": [1, 2, 1, 1, 2, 2, 1, 2, 1, 2],
        }
    )


@pytest.fixture
def date_ax():
    # an axes with a plotted date series so date locators/formatters have data
    fig, ax = plt.subplots()
    dates = [
        datetime.date(2020, 1, 1) + datetime.timedelta(days=7 * i) for i in range(20)
    ]
    # plot as numeric date numbers and mark the x-axis as a date axis so the
    # date locators/formatters apply (passing raw dates trips the plot() stub)
    ax.plot(mdates.date2num(dates), list(range(20)))
    ax.xaxis_date()
    return ax


class TestSmartPlotData:
    def test_passes_polars_through(self, plot_df):
        out = smart_plot_data(plot_df, x="x", y="y")
        # seaborn needs pandas: polars 2 dropped the interchange protocol
        assert isinstance(out["data"], pd.DataFrame)
        assert out["data"].equals(plot_df.to_pandas())
        assert out["x"] == "x"
        assert out["y"] == "y"


class TestEnhancedPlots:
    def test_scatter_returns_axes(self, plot_df):
        assert isinstance(enhanced_scatterplot(plot_df, "x", "y"), Axes)

    def test_scatter_with_hue(self, plot_df):
        # string column refs resolve against the converted pandas frame
        assert isinstance(enhanced_scatterplot(plot_df, "x", "y", hue="g"), Axes)

    def test_line_returns_axes(self, plot_df):
        assert isinstance(enhanced_lineplot(plot_df, "x", "y"), Axes)

    def test_box_returns_axes(self, plot_df):
        assert isinstance(enhanced_boxplot(plot_df, "g", "y"), Axes)

    def test_box_no_matplotlib_deprecation(self, plot_df):
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatplotlibDeprecationWarning)
            enhanced_boxplot(plot_df, "g", "y")


class TestNoisyLimits:
    def test_negative_bounds_pad_outward(self):
        lo, hi = noisy_limits(pl.Series([-10.0, -1.0]), noise=0.05)
        assert lo == pytest.approx(-10.5)
        assert hi == pytest.approx(-0.95)

    def test_float_padding(self):
        lo, hi = noisy_limits(pl.Series([10.0, 20.0]), noise=0.1)
        assert lo == pytest.approx(9.0)  # 10 - 0.1 * 10
        assert hi == pytest.approx(22.0)  # 20 + 0.1 * 20

    def test_timedelta_padding(self):
        s = pl.Series([datetime.datetime(2020, 1, 2), datetime.datetime(2020, 1, 10)])
        lo, hi = noisy_limits(s, noise=datetime.timedelta(days=1))
        assert lo == datetime.datetime(2020, 1, 1)
        assert hi == datetime.datetime(2020, 1, 11)


class TestGetHeatmapData:
    def test_integer_columns_keep_numeric_order(self):
        df = pl.DataFrame({"year": [2020, 2020, 2020], "month": [2, 10, 1]})
        result = get_heatmap_data(df, index="year", columns="month")
        assert result.columns == ["year", "1", "2", "10"]

    def test_boolean_columns_use_pivot_names(self):
        df = pl.DataFrame({"year": [2020, 2020], "flag": [False, True]})
        result = get_heatmap_data(df, index="year", columns="flag")
        assert result.columns[0] == "year"
        assert set(result.columns[1:]) == {"false", "true"}
        assert result.drop("year").row(0) == (0.5, 0.5)

    def test_row_norm(self, tweets):
        out = get_heatmap_data(
            tweets, index="year", columns="month", filter_col="has_qry"
        )
        assert out.columns == ["year", "1", "2"]
        assert out["year"].to_list() == [2020, 2021]
        # 2020 qry counts {1: 2, 2: 1} -> row-normalized
        assert out.row(0) == (2020, pytest.approx(2 / 3), pytest.approx(1 / 3))

    def test_full_norm(self, tweets):
        out = get_heatmap_data(
            tweets, index="year", columns="month", filter_col="has_qry", norm="full"
        )
        # grand total of qry counts is 8
        assert out.row(0) == (2020, pytest.approx(2 / 8), pytest.approx(1 / 8))

    def test_no_filter_counts_all_rows(self, tweets):
        # without filter_col every row is counted (incl. has_qry == False)
        out = get_heatmap_data(tweets, index="year", columns="month")
        # 2020 months: {1: 3 (incl. a-false + b-true + a-true), 2: 2}
        assert out.row(0) == (2020, pytest.approx(3 / 5), pytest.approx(2 / 5))

    def test_pre_filtered_subset(self, tweets):
        # caller pre-filters the subset (no built-in domain param)
        subset = tweets.filter(pl.col("netloc") == "b.com")
        out = get_heatmap_data(
            subset, index="year", columns="month", filter_col="has_qry"
        )
        assert out["year"].to_list() == [2020, 2021]

    def test_invalid_norm(self, tweets):
        with pytest.raises(ValueError, match="norm must be one of"):
            get_heatmap_data(tweets, index="year", columns="month", norm="bad")


class TestPlotHeatmap:
    def test_linear_scale_honors_color_limits(self, tweets):
        data = get_heatmap_data(
            tweets, index="year", columns="month", filter_col="has_qry", norm="full"
        )
        _, ax = plot_heatmap(data, data_min=0.01, data_max=1.0, log_scale=False)
        assert ax.collections[0].get_clim() == (0.01, 1.0)

    def test_returns_fig_ax_and_labels(self, tweets):
        data = get_heatmap_data(
            tweets, index="year", columns="month", filter_col="has_qry", norm="full"
        )
        fig, ax = plot_heatmap(data, data_min=0.01, data_max=1.0, log_scale=False)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # month columns become x tick labels (years are the row labels)
        assert [t.get_text() for t in ax.get_xticklabels()] == ["1", "2"]

    def test_log_scale_sets_colorbar_ticks(self, tweets):
        # log_scale=True exercises the LogNorm + power-of-10 colorbar tick branch
        data = get_heatmap_data(
            tweets, index="year", columns="month", filter_col="has_qry", norm="full"
        )
        fig, ax = plot_heatmap(data, data_min=0.01, data_max=1.0, log_scale=True)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        # colorbar present and its ticks placed at powers of 10 within the range
        colorbar = ax.collections[0].colorbar
        assert colorbar is not None
        ticks = list(colorbar.get_ticks())
        assert pytest.approx(0.01) in ticks  # 10 ** -2 (floor(log10(0.01)))
        assert pytest.approx(1.0) in ticks  # 10 ** 0 (ceil(log10(1.0)))

    def test_no_cbar(self, tweets):
        # cbar=False takes the cbar_kws=None branch and skips colorbar styling
        data = get_heatmap_data(
            tweets, index="year", columns="month", filter_col="has_qry", norm="full"
        )
        fig, ax = plot_heatmap(data, data_min=0.01, data_max=1.0, cbar=False)
        assert isinstance(ax, Axes)
        assert ax.collections[0].colorbar is None

    def test_no_annot(self, tweets):
        # annot_kws=None disables cell annotations (annot=False branch)
        data = get_heatmap_data(
            tweets, index="year", columns="month", filter_col="has_qry", norm="full"
        )
        fig, ax = plot_heatmap(
            data, data_min=0.01, data_max=1.0, log_scale=False, annot_kws=None
        )
        assert isinstance(ax, Axes)
        # no annotation text artists were drawn into the cells
        assert len(ax.texts) == 0


class TestDarkenHexColor:
    def test_halves_each_channel(self):
        # default factor 0.5 truncates each channel toward zero:
        # 0xaa=170 -> 85=0x55, 0x80=128 -> 64=0x40, 0xff=255 -> 127=0x7f
        assert darken_hex_color("#aa80ff") == "#55407f"

    def test_factor_zero_is_black(self):
        assert darken_hex_color("#ffffff", factor=0.0) == "#000000"

    def test_factor_one_is_identity(self):
        assert darken_hex_color("#1a2b3c", factor=1.0) == "#1a2b3c"


class TestSetFigText:
    def test_returns_figure_with_text(self):
        fig, _ = plt.subplots()
        out = set_fig_text(fig, "hello", x=0.5, y=0.5)
        assert out is fig
        # the text was added and carries a white bbox patch
        assert any(t.get_text() == "hello" for t in fig.texts)
        txt = fig.texts[-1]
        assert txt.get_bbox_patch() is not None


class TestRemoveAxisTicks:
    def test_remove_yaxis_ticks_zero_length(self):
        fig, ax = plt.subplots()
        remove_yaxis_ticks(ax)
        # major + minor y tick marks are zero length
        for t in ax.yaxis.get_major_ticks():
            assert t.tick1line.get_markersize() == 0

    def test_remove_yaxis_ticks_major_only(self):
        fig, ax = plt.subplots()
        # minor=False leaves the minor branch untouched
        remove_yaxis_ticks(ax, minor=False)
        for t in ax.yaxis.get_major_ticks():
            assert t.tick1line.get_markersize() == 0

    def test_remove_xaxis_ticks_zero_length(self):
        fig, ax = plt.subplots()
        remove_xaxis_ticks(ax)
        for t in ax.xaxis.get_major_ticks():
            assert t.tick1line.get_markersize() == 0

    def test_remove_xaxis_ticks_minor_only(self):
        fig, ax = plt.subplots()
        # major=False leaves the major branch untouched
        remove_xaxis_ticks(ax, major=False)
        assert isinstance(ax, Axes)


class TestXaxisDateFormatter:
    @pytest.mark.parametrize(
        "period",
        [
            "monthly",
            "monthly-labels",
            "weekly",
            "weekly-labels",
            "daily",
            "daily-labels",
        ],
    )
    def test_each_period_returns_axes(self, date_ax, period):
        out = xaxis_date_formatter(date_ax, period=period)
        assert out is date_ax

    def test_daily_labels_minor_formatter_strips_leading_zero(self, date_ax):
        # weekly-labels/daily-labels install the FuncFormatter d_fmt; calling it
        # on a tick value exercises the leading-zero stripping branch
        xaxis_date_formatter(date_ax, period="daily-labels")
        fmt = date_ax.xaxis.get_minor_formatter()
        # value for 2020-01-05 -> day "05" -> stripped to "5"
        val = mdates.date2num(datetime.date(2020, 1, 5))
        assert fmt(val) == "5"

    def test_unknown_period_returns_axes_unchanged(self, date_ax):
        # no matching branch: function falls through and just returns the axes
        out = xaxis_date_formatter(date_ax, period="hourly")
        assert out is date_ax


class TestAddLegend:
    def test_returns_legend(self, plot_df):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], label="line a")
        legend = add_legend(ax)
        assert isinstance(legend, Legend)
        assert legend.get_zorder() == 100

    def test_with_label_order(self):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], label="b")
        ax.plot([0, 1], [1, 0], label="a")
        legend = add_legend(ax, label_order=["a", "b"], zorder=50)
        # label_order branch reorders handles; a comes before b
        texts = [t.get_text() for t in legend.get_texts()]
        assert texts == ["a", "b"]
        assert legend.get_zorder() == 50


class TestReorderLegend:
    def test_orders_by_given_list(self):
        h_b, h_a, h_c = Line2D([], []), Line2D([], []), Line2D([], [])
        handles, labels = reorder_legend(
            handles=[h_b, h_a, h_c],
            labels=["b", "a", "c"],
            order=["c", "a", "b"],
        )
        assert list(labels) == ["c", "a", "b"]
        assert list(handles) == [h_c, h_a, h_b]

    def test_default_sorts_alphabetically(self):
        h_b, h_a = Line2D([], []), Line2D([], [])
        handles, labels = reorder_legend(handles=[h_b, h_a], labels=["b", "a"])
        # with no order, labels sort alphabetically
        assert list(labels) == ["a", "b"]
        assert list(handles) == [h_a, h_b]

    def test_unique_drops_duplicate_labels(self):
        h1, h2, h3 = Line2D([], []), Line2D([], []), Line2D([], [])
        handles, labels = reorder_legend(
            handles=[h1, h2, h3],
            labels=["a", "a", "b"],
            unique=True,
        )
        # only the first handle per label is kept
        assert list(labels) == ["a", "b"]
        assert len(handles) == 2

    def test_partial_order_pushes_unlisted_to_end(self):
        h_a, h_b, h_c = Line2D([], []), Line2D([], []), Line2D([], [])
        handles, labels = reorder_legend(
            handles=[h_a, h_b, h_c],
            labels=["a", "b", "c"],
            order=["c"],  # only c listed; a/b keyed to inf, sort stably after
        )
        assert list(labels)[0] == "c"


class TestAddRightAngle:
    def test_adds_lines_and_annotations(self):
        fig, ax = plt.subplots()
        ax.plot([0, 10], [0, 10])
        n_lines_before = len(ax.lines)
        out = add_right_angle(ax, x=5.0, y=5.0)
        assert out is ax
        # two reference lines (vertical + horizontal) added
        assert len(ax.lines) == n_lines_before + 2
        # two coordinate annotations added
        ann_texts = [t.get_text() for t in ax.texts]
        assert any(t.startswith("x: ") for t in ann_texts)
        assert any(t.startswith("y: ") for t in ann_texts)


class TestAddGrid:
    def test_add_grid_major_returns_axes(self):
        fig, ax = plt.subplots()
        assert add_grid_major(ax) is ax

    def test_add_grid_minor_with_color(self):
        fig, ax = plt.subplots()
        # color is not None -> the color kwarg branch is exercised
        assert add_grid_minor(ax, color="gray") is ax

    def test_add_grid_both_sets_axisbelow(self):
        fig, ax = plt.subplots()
        out = add_grid(ax, which="both", color_major="black", color_minor="gray")
        assert out is ax
        assert ax.get_axisbelow() is True

    def test_add_grid_major_only(self):
        fig, ax = plt.subplots()
        # which='major' skips the minor branch
        out = add_grid(ax, which="major", axisbelow=False)
        assert out is ax
        assert ax.get_axisbelow() is False

    def test_add_grid_minor_only(self):
        fig, ax = plt.subplots()
        # which='minor' skips the major branch
        assert add_grid(ax, which="minor") is ax


class TestAddSpines:
    def test_defaults_hide_top_and_right(self):
        fig, ax = plt.subplots()
        out = add_spines(ax)
        assert out is ax
        assert ax.spines["top"].get_visible() is False
        assert ax.spines["right"].get_visible() is False
        assert ax.spines["left"].get_visible() is True
        assert ax.spines["bottom"].get_visible() is True

    def test_custom_spines_and_linewidth(self):
        fig, ax = plt.subplots()
        out = add_spines(
            ax, on_spines=["left", "bottom"], off_spines=["bottom"], linewidth=2.0
        )
        assert out is ax
        assert ax.spines["left"].get_linewidth() == 2.0
        assert ax.spines["bottom"].get_visible() is False
