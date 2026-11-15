"""Plotting helpers on matplotlib and polars.

Each helper takes a polars DataFrame and column names, draws on a given
``Axes`` (the current one by default), and returns it. Helpers with groups
take ``hue`` (with ``hue_order`` and ``palette``); categorical helpers take
``order`` for the category axis and ``orient`` ("x" puts the categories on the
x-axis, "y" on the y-axis, with the value column on the other).
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection, PathCollection, QuadMesh
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb, to_rgba_array
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from tools.plots._seaborn_port import Beeswarm, KDepth, letter_values
from tools.plots.estimate import ErrorBar, Estimator, bootstrap, estimate_interval
from tools.plots.palettes import RGB, Palette, resolve_colors

Orient = Literal["x", "y"]
Seed = int | np.random.Generator | None

# Error bar defaults, as seaborn draws them
_ERR_COLOR = ".26"


def _get_ax(ax: Axes | None) -> Axes:
    return ax if ax is not None else plt.gca()


def _complete(df: pl.DataFrame, *cols: str | None) -> pl.DataFrame:
    """The named columns (each once, Nones skipped), with missing rows dropped.

    A row is missing when any of its values is null or a float NaN, as seaborn
    treats both.
    """
    names = list(dict.fromkeys(c for c in cols if c is not None))
    data = df.select(names).drop_nulls()
    floats = [name for name, dtype in data.schema.items() if dtype.is_float()]
    if floats:
        data = data.filter(~pl.any_horizontal(pl.col(floats).is_nan()))
    return data


def _single_color(color: Any) -> RGB:
    """``color`` as RGB, or the palette's first color when it is None."""
    return to_rgb(color) if color is not None else resolve_colors([0])[0]


def _levels(series: pl.Series, order: Sequence[Any] | None) -> list[Any]:
    """Category levels: ``order`` if given, else an Enum's declared categories,
    sorted numbers, dates, and booleans, or first-seen values."""
    if order is not None:
        return list(order)
    values = series.drop_nulls()
    dtype = values.dtype
    if isinstance(dtype, pl.Enum):
        return dtype.categories.to_list()
    if dtype.is_numeric() or dtype.is_temporal() or dtype == pl.Boolean:
        return values.unique().sort().to_list()
    return values.unique(maintain_order=True).to_list()


def _dodge_offsets(width: float, n_hue: int, dodge: bool) -> tuple[np.ndarray, float]:
    """Offsets that place ``n_hue`` elements side by side within ``width``.

    Returns:
        Tuple of (offset per hue level, width of each element)
    """
    if not dodge or n_hue <= 1:
        return np.zeros(n_hue), width
    each = width / n_hue
    offsets = np.linspace(0, width - each, n_hue)
    return offsets - offsets.mean(), each


@dataclass
class _Categorical:
    """Values split by category and hue level, with colors and positions."""

    orient: Orient
    cat_col: str
    val_col: str
    hue: str | None
    cats: list[Any]
    hues: list[Any]
    colors: list[RGB]
    groups: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)

    def xy(self, pos: Any, val: Any) -> tuple[Any, Any]:
        """Map (category position, value) to (x, y) for this orientation."""
        return (pos, val) if self.orient == "x" else (val, pos)


def _categorical(
    df: pl.DataFrame,
    x: str,
    y: str,
    hue: str | None,
    order: Sequence[Any] | None,
    hue_order: Sequence[Any] | None,
    orient: Orient,
    palette: Palette,
    color: Any,
) -> _Categorical:
    if orient not in ("x", "y"):
        raise ValueError(f"orient must be 'x' or 'y', got {orient!r}")
    cat_col, val_col = (x, y) if orient == "x" else (y, x)
    data = _complete(df, cat_col, val_col, hue)
    cats = _levels(data.get_column(cat_col), order)
    if hue:
        hues = _levels(data.get_column(hue), hue_order)
        colors = resolve_colors(hues, palette)
    else:
        hues = [None]
        colors = [_single_color(color)]
    groups: dict[tuple[int, int], np.ndarray] = {}
    for ci, cat in enumerate(cats):
        sub = data.filter(pl.col(cat_col) == cat)
        for hi, level in enumerate(hues):
            part = sub if hue is None else sub.filter(pl.col(hue) == level)
            values = part.get_column(val_col).cast(pl.Float64).to_numpy()
            if values.size:
                groups[(ci, hi)] = values
    return _Categorical(orient, cat_col, val_col, hue, cats, hues, colors, groups)


def _finish_categorical(
    ax: Axes, cat: _Categorical, legend: bool, handle: Callable[[int, RGB], Any]
) -> None:
    """Label the axes, set categorical ticks and limits, and add a hue legend."""
    n = len(cat.cats)
    labels = [str(c) for c in cat.cats]
    if cat.orient == "x":
        ax.set_xticks(range(n), labels)
        ax.set_xlim(-0.5, n - 0.5)
        ax.xaxis.grid(False)
        ax.set_xlabel(cat.cat_col)
        ax.set_ylabel(cat.val_col)
    else:
        ax.set_yticks(range(n), labels)
        ax.set_ylim(n - 0.5, -0.5)
        ax.yaxis.grid(False)
        ax.set_ylabel(cat.cat_col)
        ax.set_xlabel(cat.val_col)
    if legend and cat.hue is not None:
        handles = [handle(hi, c) for hi, c in enumerate(cat.colors)]
        ax.legend(handles, [str(h) for h in cat.hues], title=cat.hue)


def _patch_handle(hi: int, color: RGB) -> Patch:
    return Patch(facecolor=color, edgecolor=_ERR_COLOR, linewidth=0.75)


def _marker_handle(hi: int, color: RGB) -> Line2D:
    return Line2D([], [], marker="o", linestyle="", color=color)


def _draw_interval(
    ax: Axes,
    cat: _Categorical,
    pos: float,
    low: float,
    high: float,
    cap_width: float,
    err_kws: dict[str, Any],
) -> None:
    """Draw one error bar along the value axis, with caps when ``cap_width``."""
    if np.isnan(low) or np.isnan(high):
        return
    ax.plot(*cat.xy([pos, pos], [low, high]), **err_kws)
    if cap_width > 0:
        half = cap_width / 2
        for edge in (low, high):
            ax.plot(*cat.xy([pos - half, pos + half], [edge, edge]), **err_kws)


def _err_kws(err_kws: dict[str, Any] | None, color: Any) -> dict[str, Any]:
    base = {
        "color": color,
        "linewidth": 1.5 * plt.rcParams["lines.linewidth"],
        "solid_capstyle": "butt",
    }
    return {**base, **(err_kws or {})}


def barplot(
    df: pl.DataFrame,
    x: str,
    y: str,
    *,
    hue: str | None = None,
    order: Sequence[Any] | None = None,
    hue_order: Sequence[Any] | None = None,
    estimator: Estimator = "mean",
    errorbar: ErrorBar = ("ci", 95),
    n_boot: int = 1000,
    seed: Seed = None,
    orient: Orient = "x",
    palette: Palette = None,
    color: Any = None,
    width: float = 0.8,
    dodge: bool = True,
    capsize: float = 0.0,
    err_kws: dict[str, Any] | None = None,
    legend: bool = True,
    ax: Axes | None = None,
    **kwargs: Any,
) -> Axes:
    """Bar plot of an estimate per category, with a bootstrap interval.

    Args:
        df: Polars DataFrame
        x: Category column (orient "x") or value column (orient "y")
        y: Value column (orient "x") or category column (orient "y")
        hue: Optional grouping column, drawn as dodged bars
        order: Category order (default: sorted numbers or first-seen values)
        hue_order: Hue level order
        estimator: Statistic per bar: a numpy reduction name or a callable
        errorbar: Interval spec (see ``estimate_interval``); None for no bars
        n_boot: Bootstrap resamples for a ``"ci"`` interval
        seed: Seed for the bootstrap, so the intervals are reproducible
        orient: "x" for vertical bars, "y" for horizontal
        palette: Hue colors: colormap name, list, or dict (see ``resolve_colors``)
        color: Bar color without hue
        width: Total width of the bars at each category
        dodge: Place hue levels side by side (False overlaps them)
        capsize: Error bar cap width, as a fraction of the bar width
        err_kws: Keyword arguments for the error bar lines
        legend: Add a legend for ``hue``
        ax: Axes to draw on (default: current axes)
        **kwargs: Passed to ``Axes.bar`` / ``Axes.barh``

    Returns:
        The Axes drawn on
    """
    ax = _get_ax(ax)
    cat = _categorical(df, x, y, hue, order, hue_order, orient, palette, color)
    n_hue = len(cat.hues)
    offsets, bar_width = _dodge_offsets(width, n_hue, dodge)
    err = _err_kws(err_kws, _ERR_COLOR)
    for (ci, hi), values in cat.groups.items():
        est, low, high = estimate_interval(values, estimator, errorbar, n_boot, seed)
        pos = ci + offsets[hi]
        bar_kws: dict[str, Any] = {"color": cat.colors[hi], **kwargs}
        if orient == "x":
            ax.bar(pos, est, width=bar_width, **bar_kws)
        else:
            ax.barh(pos, est, height=bar_width, **bar_kws)
        _draw_interval(ax, cat, pos, low, high, capsize * bar_width, err)
    _finish_categorical(ax, cat, legend, _patch_handle)
    return ax


def pointplot(
    df: pl.DataFrame,
    x: str,
    y: str,
    *,
    hue: str | None = None,
    order: Sequence[Any] | None = None,
    hue_order: Sequence[Any] | None = None,
    estimator: Estimator = "mean",
    errorbar: ErrorBar = ("ci", 95),
    n_boot: int = 1000,
    seed: Seed = None,
    orient: Orient = "x",
    palette: Palette = None,
    color: Any = None,
    markers: str | Sequence[str] = "o",
    linestyles: str | Sequence[str] = "-",
    dodge: bool | float = False,
    markersize: float | None = None,
    linewidth: float | None = None,
    capsize: float = 0.0,
    err_kws: dict[str, Any] | None = None,
    legend: bool = True,
    ax: Axes | None = None,
    **kwargs: Any,
) -> Axes:
    """Point plot: an estimate per category as markers joined by lines.

    Takes the same estimate and interval options as ``barplot``.

    Args:
        df: Polars DataFrame
        x: Category column (orient "x") or value column (orient "y")
        y: Value column (orient "x") or category column (orient "y")
        hue: Optional grouping column, one line per level
        order: Category order
        hue_order: Hue level order
        estimator: Statistic per point
        errorbar: Interval spec (see ``estimate_interval``); None for no bars
        n_boot: Bootstrap resamples for a ``"ci"`` interval
        seed: Seed for the bootstrap
        orient: "x" for categories along x, "y" along y
        palette: Hue colors
        color: Line color without hue
        markers: Marker, or one marker per hue level
        linestyles: Line style, or one per hue level
        dodge: Spread hue levels apart along the category axis: a total
            distance, or True for 0.025 per level
        markersize: Marker size (default: ``lines.markersize``)
        linewidth: Line width (default: ``lines.linewidth``)
        capsize: Error bar cap width, in category-axis units
        err_kws: Keyword arguments for the error bar lines
        legend: Add a legend for ``hue``
        ax: Axes to draw on
        **kwargs: Passed to ``Axes.plot`` for the lines (e.g. ``alpha``)

    Returns:
        The Axes drawn on
    """
    ax = _get_ax(ax)
    cat = _categorical(df, x, y, hue, order, hue_order, orient, palette, color)
    n_hue = len(cat.hues)
    if dodge is True:
        dodge = 0.025 * n_hue
    offsets = np.linspace(0, float(dodge), n_hue) if dodge else np.zeros(n_hue)
    offsets = offsets - offsets.mean()
    marker_list = [markers] * n_hue if isinstance(markers, str) else list(markers)
    style_list: list[Any] = (
        [linestyles] * n_hue if isinstance(linestyles, str) else list(linestyles)
    )
    lw = linewidth if linewidth is not None else plt.rcParams["lines.linewidth"]
    ms = markersize if markersize is not None else plt.rcParams["lines.markersize"]
    for hi in range(n_hue):
        positions: list[float] = []
        estimates: list[float] = []
        err = _err_kws({"linewidth": lw, **(err_kws or {})}, cat.colors[hi])
        if "alpha" in kwargs:
            err.setdefault("alpha", kwargs["alpha"])
        for ci in range(len(cat.cats)):
            values = cat.groups.get((ci, hi))
            if values is None:
                continue
            est, low, high = estimate_interval(
                values, estimator, errorbar, n_boot, seed
            )
            pos = ci + offsets[hi]
            positions.append(pos)
            estimates.append(est)
            _draw_interval(ax, cat, pos, low, high, capsize, err)
        ax.plot(
            *cat.xy(positions, estimates),
            color=cat.colors[hi],
            marker=marker_list[hi % len(marker_list)],
            linestyle=style_list[hi % len(style_list)],
            markersize=ms,
            linewidth=lw,
            **kwargs,
        )

    def handle(hi: int, c: RGB) -> Line2D:
        return Line2D(
            [],
            [],
            color=c,
            marker=marker_list[hi % len(marker_list)],
            linestyle=style_list[hi % len(style_list)],
        )

    _finish_categorical(ax, cat, legend, handle)
    return ax


def regplot(
    df: pl.DataFrame,
    x: str,
    y: str,
    *,
    ci: float | None = 95,
    n_boot: int = 1000,
    seed: Seed = None,
    truncate: bool = True,
    scatter: bool = True,
    color: Any = None,
    scatter_kws: dict[str, Any] | None = None,
    line_kws: dict[str, Any] | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Scatter plot with an OLS line and a bootstrap confidence band.

    Args:
        df: Polars DataFrame
        x: Predictor column
        y: Response column
        ci: Confidence level of the band, in percent; None for no band
        n_boot: Bootstrap resamples for the band
        seed: Seed for the bootstrap
        truncate: Draw the line over the data range (False: the axis limits)
        scatter: Draw the points
        color: Color for points, line, and band
        scatter_kws: Keyword arguments for ``Axes.scatter``
        line_kws: Keyword arguments for the line's ``Axes.plot``
        ax: Axes to draw on

    Returns:
        The Axes drawn on
    """
    ax = _get_ax(ax)
    data = _complete(df, x, y)
    xv = data.get_column(x).cast(pl.Float64).to_numpy()
    yv = data.get_column(y).cast(pl.Float64).to_numpy()
    rgb = _single_color(color)
    if scatter:
        point_kws: dict[str, Any] = {"color": rgb, "alpha": 0.8, **(scatter_kws or {})}
        ax.scatter(xv, yv, **point_kws)
    if truncate or not scatter:
        lo, hi = float(xv.min()), float(xv.max())
    else:
        lo, hi = ax.get_xlim()
    grid = np.linspace(lo, hi, 100)

    def fit(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        return np.polyval(np.polyfit(xs, ys, 1), grid)

    fit_kws: dict[str, Any] = {"color": rgb, **(line_kws or {})}
    ax.plot(grid, fit(xv, yv), **fit_kws)
    if ci is not None:
        boots = bootstrap(xv, yv, func=fit, n_boot=n_boot, seed=seed)
        edge = (100 - ci) / 2
        band_lo, band_hi = np.percentile(boots, [edge, 100 - edge], axis=0)
        ax.fill_between(grid, band_lo, band_hi, color=rgb, alpha=0.15, linewidth=0)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    return ax


def scatterplot(
    df: pl.DataFrame,
    x: str,
    y: str,
    *,
    hue: str | None = None,
    hue_order: Sequence[Any] | None = None,
    palette: Palette = None,
    color: Any = None,
    legend: bool = True,
    ax: Axes | None = None,
    **kwargs: Any,
) -> Axes:
    """Scatter plot, one color per ``hue`` level.

    Args:
        df: Polars DataFrame
        x: Column for the x-axis
        y: Column for the y-axis
        hue: Optional grouping column
        hue_order: Hue level order
        palette: Hue colors
        color: Point color without hue
        legend: Add a legend for ``hue``
        ax: Axes to draw on
        **kwargs: Passed to ``Axes.scatter``

    Returns:
        The Axes drawn on
    """
    ax = _get_ax(ax)
    data = _complete(df, x, y, hue)
    if hue is None:
        rgb = _single_color(color)
        ax.scatter(data.get_column(x), data.get_column(y), color=rgb, **kwargs)
    else:
        levels = _levels(data.get_column(hue), hue_order)
        for level, rgb in zip(levels, resolve_colors(levels, palette)):
            part = data.filter(pl.col(hue) == level)
            ax.scatter(
                part.get_column(x),
                part.get_column(y),
                color=rgb,
                label=str(level),
                **kwargs,
            )
        if legend:
            ax.legend(title=hue)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    return ax


HistStat = Literal["count", "frequency", "density", "probability", "percent"]


def histplot(
    df: pl.DataFrame,
    x: str,
    *,
    hue: str | None = None,
    hue_order: Sequence[Any] | None = None,
    palette: Palette = None,
    color: Any = None,
    bins: int | str | Sequence[float] = "auto",
    binrange: tuple[float, float] | None = None,
    stat: HistStat = "count",
    multiple: Literal["layer", "stack"] = "layer",
    alpha: float | None = None,
    legend: bool = True,
    ax: Axes | None = None,
    **kwargs: Any,
) -> Axes:
    """Histogram, one layer (or stacked segment) per ``hue`` level.

    All levels share one set of bin edges, and ``stat`` normalizes over all
    levels together, so layers compare on the same scale.

    Args:
        df: Polars DataFrame
        x: Column to bin
        hue: Optional grouping column
        hue_order: Hue level order
        palette: Hue colors
        color: Bar color without hue
        bins: Bin count, a numpy bin rule name, or explicit edges
        binrange: (low, high) range for the bins
        stat: "count", "frequency" (count / bin width), "density" (area sums
            to 1), "probability" (heights sum to 1), or "percent"
        multiple: "layer" overlaps levels; "stack" stacks them
        alpha: Bar opacity (default 0.75, or 0.5 for layered hue levels)
        legend: Add a legend for ``hue``
        ax: Axes to draw on
        **kwargs: Passed to ``Axes.bar``

    Returns:
        The Axes drawn on
    """
    if stat not in ("count", "frequency", "density", "probability", "percent"):
        raise ValueError(f"unknown stat {stat!r}")
    if multiple not in ("layer", "stack"):
        raise ValueError(f"multiple must be 'layer' or 'stack', got {multiple!r}")
    ax = _get_ax(ax)
    data = _complete(df, x, hue)
    all_values = data.get_column(x).cast(pl.Float64).to_numpy()
    edges = np.histogram_bin_edges(all_values, bins=bins, range=binrange)  # type: ignore[arg-type]
    widths = np.diff(edges)
    total = all_values.size
    if hue is None:
        levels: list[Any] = [None]
        colors = [_single_color(color)]
    else:
        levels = _levels(data.get_column(hue), hue_order)
        colors = resolve_colors(levels, palette)
    if alpha is None:
        alpha = 0.5 if hue is not None and multiple == "layer" else 0.75
    base = np.zeros(len(widths))
    for level, rgb in zip(levels, colors):
        part = data if hue is None else data.filter(pl.col(hue) == level)
        counts, _ = np.histogram(
            part.get_column(x).cast(pl.Float64).to_numpy(), bins=edges
        )
        heights = counts.astype(float)
        if stat == "frequency":
            heights /= widths
        elif stat == "density":
            heights /= total * widths
        elif stat == "probability":
            heights /= total
        elif stat == "percent":
            heights *= 100 / total
        bar_kws: dict[str, Any] = {
            "color": rgb,
            "alpha": alpha,
            "edgecolor": "white",
            "linewidth": 0.5,
            **kwargs,
        }
        if level is not None:
            bar_kws["label"] = str(level)
        ax.bar(edges[:-1], heights, width=widths, bottom=base, align="edge", **bar_kws)
        if multiple == "stack":
            base = base + heights
    if legend and hue is not None:
        ax.legend(title=hue)
    ax.set_xlabel(x)
    ax.set_ylabel(stat.capitalize())
    return ax


def boxplot(
    df: pl.DataFrame,
    x: str,
    y: str,
    *,
    hue: str | None = None,
    order: Sequence[Any] | None = None,
    hue_order: Sequence[Any] | None = None,
    orient: Orient = "x",
    palette: Palette = None,
    color: Any = None,
    width: float = 0.8,
    dodge: bool = True,
    linecolor: Any = _ERR_COLOR,
    linewidth: float | None = None,
    fliersize: float | None = None,
    whis: float = 1.5,
    showfliers: bool = True,
    legend: bool = True,
    ax: Axes | None = None,
    **kwargs: Any,
) -> Axes:
    """Box plot per category, one box per ``hue`` level.

    Args:
        df: Polars DataFrame
        x: Category column (orient "x") or value column (orient "y")
        y: Value column (orient "x") or category column (orient "y")
        hue: Optional grouping column, drawn as dodged boxes
        order: Category order
        hue_order: Hue level order
        orient: "x" for vertical boxes, "y" for horizontal
        palette: Hue colors
        color: Box color without hue
        width: Total width of the boxes at each category
        dodge: Place hue levels side by side
        linecolor: Color of box edges, whiskers, and medians
        linewidth: Line width (default: ``lines.linewidth``)
        fliersize: Outlier marker size (0 hides them)
        whis: Whisker reach, in IQRs past the quartiles
        showfliers: Draw the outliers
        legend: Add a legend for ``hue``
        ax: Axes to draw on
        **kwargs: Passed to ``Axes.boxplot``

    Returns:
        The Axes drawn on
    """
    ax = _get_ax(ax)
    cat = _categorical(df, x, y, hue, order, hue_order, orient, palette, color)
    n_hue = len(cat.hues)
    offsets, box_width = _dodge_offsets(width, n_hue, dodge)
    lw = linewidth if linewidth is not None else plt.rcParams["lines.linewidth"]
    line = {"color": linecolor, "linewidth": lw}
    flier: dict[str, Any] = {"markeredgecolor": linecolor, "markeredgewidth": lw}
    if fliersize is not None:
        flier["markersize"] = fliersize
    for (ci, hi), values in cat.groups.items():
        ax.boxplot(
            [values],
            positions=[ci + offsets[hi]],
            widths=box_width,
            whis=whis,
            showfliers=showfliers,
            patch_artist=True,
            manage_ticks=False,
            orientation="vertical" if orient == "x" else "horizontal",
            boxprops={
                "facecolor": cat.colors[hi],
                "edgecolor": linecolor,
                "linewidth": lw,
            },
            whiskerprops=line,
            capprops=line,
            medianprops=line,
            flierprops=flier,
            **kwargs,
        )
    _finish_categorical(ax, cat, legend, _patch_handle)
    return ax


def _attach_swarm(points: PathCollection, center: float, beeswarm: Beeswarm) -> None:
    """Re-run the beeswarm on every draw, as the axes transform may change."""
    original_draw = points.draw

    def draw(renderer: Any) -> None:
        beeswarm(points, center)
        ax = points.axes
        if ax is not None:
            ax.update_datalim(points.get_datalim(ax.transData))
        original_draw(renderer)

    setattr(points, "draw", draw)


def swarmplot(
    df: pl.DataFrame,
    x: str,
    y: str,
    *,
    hue: str | None = None,
    order: Sequence[Any] | None = None,
    hue_order: Sequence[Any] | None = None,
    orient: Orient = "x",
    palette: Palette = None,
    color: Any = None,
    size: float = 5,
    linewidth: float = 0,
    edgecolor: Any = "white",
    dodge: bool = False,
    warn_thresh: float = 0.05,
    legend: bool = True,
    ax: Axes | None = None,
    **kwargs: Any,
) -> Axes:
    """Beeswarm plot: every point, moved sideways so none overlap.

    Placement is seaborn's beeswarm algorithm (ported), so for the same
    figure, axes limits, marker size, and data it puts points where seaborn's
    ``swarmplot`` does. Points are placed at draw time, in screen space.

    Args:
        df: Polars DataFrame
        x: Category column (orient "x") or value column (orient "y")
        y: Value column (orient "x") or category column (orient "y")
        hue: Optional grouping column, for point colors
        order: Category order
        hue_order: Hue level order
        orient: "x" for categories along x, "y" along y
        palette: Hue colors
        color: Point color without hue
        size: Marker diameter in points
        linewidth: Marker edge width in points
        edgecolor: Marker edge color
        dodge: Separate swarms per hue level
        warn_thresh: Warn when more than this share of points cannot be placed
        legend: Add a legend for ``hue``
        ax: Axes to draw on
        **kwargs: Passed to ``Axes.scatter``

    Returns:
        The Axes drawn on
    """
    ax = _get_ax(ax)
    cat = _categorical(df, x, y, hue, order, hue_order, orient, palette, color)
    width = 0.8
    n_hue = len(cat.hues)
    offsets, _ = _dodge_offsets(width, n_hue, dodge)
    scatter_kws: dict[str, Any] = {
        "s": size**2,
        "linewidth": linewidth,
        "edgecolor": edgecolor,
        **kwargs,
    }
    swarms: list[tuple[PathCollection, float]] = []
    for ci in range(len(cat.cats)):
        hue_sets = [[hi] for hi in range(n_hue)] if dodge else [list(range(n_hue))]
        for his in hue_sets:
            parts = [(hi, cat.groups[(ci, hi)]) for hi in his if (ci, hi) in cat.groups]
            if not parts:
                continue
            values = np.concatenate([v for _, v in parts])
            colors = np.concatenate(
                [np.tile(cat.colors[hi], (len(v), 1)) for hi, v in parts]
            )
            center = ci + (offsets[his[0]] if dodge else 0.0)
            pos = np.full(values.size, center)
            points = ax.scatter(*cat.xy(pos, values), color=colors, **scatter_kws)
            swarms.append((points, center))
    _finish_categorical(ax, cat, legend, _marker_handle)
    beeswarm = Beeswarm(orient=orient, width=width, warn_thresh=warn_thresh)
    for points, center in swarms:
        if np.asarray(points.get_offsets()).shape[0] > 1:
            _attach_swarm(points, center, beeswarm)
    ax.figure.canvas.draw()
    return ax


def _light_ramp(color: RGB) -> LinearSegmentedColormap:
    """Colormap from a pale tint of ``color`` up to ``color``."""
    rgb = np.asarray(color)
    pale = 0.15 * rgb + 0.85 * 0.95
    return LinearSegmentedColormap.from_list("light", [tuple(pale), tuple(rgb)])


def boxenplot(
    df: pl.DataFrame,
    x: str,
    y: str,
    *,
    hue: str | None = None,
    order: Sequence[Any] | None = None,
    hue_order: Sequence[Any] | None = None,
    orient: Orient = "x",
    palette: Palette = None,
    color: Any = None,
    width: float = 0.8,
    dodge: bool = True,
    linecolor: Any = _ERR_COLOR,
    linewidth: float | None = None,
    width_method: Literal["exponential", "linear", "area"] = "exponential",
    k_depth: KDepth = "tukey",
    outlier_prop: float = 0.007,
    trust_alpha: float = 0.05,
    showfliers: bool = True,
    legend: bool = True,
    ax: Axes | None = None,
    **kwargs: Any,
) -> Axes:
    """Letter-value (boxen) plot per category, one per ``hue`` level.

    Box depths and widths follow seaborn's ``boxenplot`` (ported letter-value
    computation), so the boxes span the same percentiles as seaborn's.

    Args:
        df: Polars DataFrame
        x: Category column (orient "x") or value column (orient "y")
        y: Value column (orient "x") or category column (orient "y")
        hue: Optional grouping column, drawn as dodged boxes
        order: Category order
        hue_order: Hue level order
        orient: "x" for vertical boxes, "y" for horizontal
        palette: Hue colors
        color: Box color without hue
        width: Total width of the widest box at each category
        dodge: Place hue levels side by side
        linecolor: Box edge and median color
        linewidth: Box edge width (default: half ``lines.linewidth``)
        width_method: How box width shrinks with depth: "exponential",
            "linear", or "area"
        k_depth: Depth rule (see ``letter_values``)
        outlier_prop: Outlier rate for ``k_depth="proportion"``
        trust_alpha: Confidence level for ``k_depth="trustworthy"``
        showfliers: Draw points beyond the outermost box
        legend: Add a legend for ``hue``
        ax: Axes to draw on
        **kwargs: Passed to the boxes' ``PatchCollection``

    Returns:
        The Axes drawn on
    """
    if width_method not in ("exponential", "linear", "area"):
        raise ValueError(f"unknown width_method {width_method!r}")
    ax = _get_ax(ax)
    cat = _categorical(df, x, y, hue, order, hue_order, orient, palette, color)
    n_hue = len(cat.hues)
    offsets, box_width = _dodge_offsets(width, n_hue, dodge)
    lw = linewidth if linewidth is not None else 0.5 * plt.rcParams["lines.linewidth"]
    for (ci, hi), values in cat.groups.items():
        lv = letter_values(values, k_depth, outlier_prop, trust_alpha)
        k = lv["k"]
        n_boxes = k * 2 - 1
        vals = lv["values"]
        levels = lv["levels"]
        exponent = (levels - 1 - k).astype(float)
        if width_method == "linear":
            rel_widths = levels + 1
        elif width_method == "exponential":
            rel_widths = 2**exponent
        else:
            tails = levels < (k - 1)
            rel_widths = 2 ** (exponent - tails) / np.diff(vals)
        center = ci + offsets[hi]
        widths = rel_widths / rel_widths.max() * box_width
        boxcolors = _light_ramp(cat.colors[hi])(2 ** ((exponent + 2) / 3))
        boxes = []
        for i in range(n_boxes):
            low_edge = center - widths[i] / 2
            height = vals[i + 1] - vals[i]
            if orient == "x":
                boxes.append(Rectangle((low_edge, vals[i]), widths[i], height))
            else:
                boxes.append(Rectangle((vals[i], low_edge), height, widths[i]))
        box_kws: dict[str, Any] = {
            "facecolors": boxcolors,
            "edgecolors": linecolor,
            "linewidth": lw,
            **kwargs,
        }
        collection = PatchCollection(boxes, **box_kws)
        ax.add_collection(collection, autolim=False)
        ax.update_datalim(
            np.column_stack([vals, vals]),
            updatex=orient == "y",
            updatey=orient == "x",
        )
        med = lv["median"]
        half = box_width / 2
        ax.plot(
            *cat.xy([center - half, center + half], [med, med]),
            color=linecolor,
            solid_capstyle="butt",
            linewidth=1.25 * lw,
        )
        if showfliers and lv["fliers"].size:
            fliers = lv["fliers"]
            ax.scatter(
                *cat.xy(np.full(fliers.size, center), fliers),
                facecolor="none",
                edgecolor=".45",
                s=25,
            )
    ax.autoscale_view(scalex=orient == "y", scaley=orient == "x")
    _finish_categorical(ax, cat, legend, _patch_handle)
    return ax


def _relative_luminance(color: Any) -> float:
    """Relative luminance of a color (WCAG), for picking annotation text."""
    rgb = to_rgba_array(color)[:, :3]
    rgb = np.where(rgb <= 0.03928, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return float(rgb.dot([0.2126, 0.7152, 0.0722])[0])


def heatmap(
    data: pl.DataFrame | np.ndarray,
    *,
    index_col: str | None = None,
    cmap: Any = "viridis",
    norm: Normalize | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    annot: bool = False,
    fmt: str = ".2g",
    annot_kws: dict[str, Any] | None = None,
    linewidths: float = 0,
    linecolor: Any = "white",
    cbar: bool = True,
    cbar_kws: dict[str, Any] | None = None,
    square: bool = False,
    xticklabels: bool | Sequence[Any] = True,
    yticklabels: bool | Sequence[Any] = True,
    ax: Axes | None = None,
) -> Axes:
    """Heatmap of a matrix, with optional cell annotations and a colorbar.

    Rows run top to bottom. NaN cells are left empty and unannotated.

    Args:
        data: Polars frame of numeric columns (plus ``index_col``), or a 2-D
            array
        index_col: Frame column holding the row labels; excluded from the
            matrix
        cmap: Colormap name or object
        norm: Color normalization (e.g. ``LogNorm``); overrides vmin/vmax
        vmin: Lower color limit (default: data minimum)
        vmax: Upper color limit (default: data maximum)
        annot: Write each cell's value in the cell
        fmt: Format spec for annotations
        annot_kws: Keyword arguments for the annotation ``Axes.text`` calls
        linewidths: Width of the lines between cells
        linecolor: Color of the lines between cells
        cbar: Draw a colorbar
        cbar_kws: Keyword arguments for ``Figure.colorbar``
        square: Make cells square
        xticklabels: True for column names (or indices), False for none, or
            explicit labels
        yticklabels: True for ``index_col`` values (or indices), False for
            none, or explicit labels
        ax: Axes to draw on

    Returns:
        The Axes drawn on; the cell mesh is ``ax.collections[0]``
    """
    ax = _get_ax(ax)
    if isinstance(data, pl.DataFrame):
        row_labels: list[Any] = (
            data.get_column(index_col).to_list()
            if index_col
            else list(range(data.height))
        )
        frame = data.drop(index_col) if index_col else data
        col_labels: list[Any] = frame.columns
        values = frame.to_numpy().astype(float)
    else:
        values = np.asarray(data, dtype=float)
        if values.ndim != 2:
            raise ValueError("heatmap data must be 2-D")
        row_labels = list(range(values.shape[0]))
        col_labels = list(range(values.shape[1]))
    matrix = np.ma.masked_invalid(values)
    nrows, ncols = matrix.shape

    color_kws: dict[str, Any] = {"norm": norm} if norm is not None else {}
    if norm is None:
        color_kws.update(vmin=vmin, vmax=vmax)
    mesh: QuadMesh = ax.pcolormesh(
        matrix,
        cmap=cmap,
        edgecolors=linecolor,
        linewidth=linewidths,
        **color_kws,
    )
    ax.set_xlim(0, ncols)
    ax.set_ylim(nrows, 0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)

    if xticklabels is False:
        ax.set_xticks([])
    else:
        labels = col_labels if xticklabels is True else list(xticklabels)
        ax.set_xticks(np.arange(ncols) + 0.5, [str(v) for v in labels])
    if yticklabels is False:
        ax.set_yticks([])
    else:
        labels = row_labels if yticklabels is True else list(yticklabels)
        ax.set_yticks(np.arange(nrows) + 0.5, [str(v) for v in labels], va="center")

    if annot:
        text_kws = annot_kws or {}
        mask = np.ma.getmaskarray(matrix)
        cell_colors = np.asarray(mesh.cmap(mesh.norm(matrix)))
        for i in range(nrows):
            for j in range(ncols):
                if mask[i, j]:
                    continue
                cell = cell_colors[i, j]
                text_color = ".15" if _relative_luminance(cell) > 0.408 else "w"
                cell_kws: dict[str, Any] = {"color": text_color, **text_kws}
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    format(float(values[i, j]), fmt),
                    ha="center",
                    va="center",
                    **cell_kws,
                )

    if square:
        ax.set_aspect("equal")
    if cbar:
        colorbar = ax.figure.colorbar(mesh, ax=ax, **(cbar_kws or {}))
        colorbar.outline.set_linewidth(0)  # type: ignore[union-attr]
    return ax
