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
from matplotlib.cbook import normalize_kwargs
from matplotlib.collections import PatchCollection, PathCollection, QuadMesh
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba_array
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Rectangle

from tools.plots._common import (
    add_legend_artists,
    category_levels,
    complete_rows,
    get_ax,
    gray_complement,
    set_axis_labels,
    single_color,
)
from tools.plots._seaborn_port import Beeswarm, KDepth, letter_values
from tools.plots.estimate import ErrorBar, Estimator, bootstrap, estimate_interval
from tools.plots.palettes import RGB, Palette, desaturate, resolve_colors

Orient = Literal["x", "y"]
Seed = int | np.random.Generator | None
# "auto" dodges hue levels only when they would overlap: when ``hue`` differs
# from the category column, so a category holds more than one level.
Dodge = bool | Literal["auto"]
# "auto" draws the hue legend unless ``hue`` is the category column, where the
# category axis already labels every color.
LegendSpec = bool | Literal["auto"]

# Error bar defaults, as seaborn draws them
_ERR_COLOR = ".26"


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

    def resolve_dodge(self, dodge: Dodge) -> bool:
        """Whether to dodge: as given, or for "auto", when some category holds
        more than one hue level."""
        if dodge != "auto":
            return bool(dodge)
        if self.hue is None:
            return False
        return len({ci for ci, _ in self.groups}) != len(self.groups)


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
    saturation: float = 1.0,
) -> _Categorical:
    if orient not in ("x", "y"):
        raise ValueError(f"orient must be 'x' or 'y', got {orient!r}")
    cat_col, val_col = (x, y) if orient == "x" else (y, x)
    data = complete_rows(df, cat_col, val_col, hue)
    cats = category_levels(data.get_column(cat_col), order)
    if hue:
        hues = category_levels(data.get_column(hue), hue_order)
        colors = resolve_colors(hues, palette)
    else:
        hues = [None]
        colors = [single_color(color)]
    colors = [desaturate(c, saturation) for c in colors]
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
    ax: Axes,
    cat: _Categorical,
    legend: LegendSpec,
    handle: Callable[[int, RGB], Rectangle | Line2D],
) -> None:
    """Label the axes, set categorical ticks and limits, and add a hue legend.

    The legend is built from one labeled artist per hue level, added to the
    axes, so a later ``ax.legend()`` or ``add_legend`` finds the same entries.
    """
    n = len(cat.cats)
    labels = [str(c) for c in cat.cats]
    if cat.orient == "x":
        ax.set_xticks(range(n), labels)
        ax.set_xlim(-0.5, n - 0.5)
        ax.xaxis.grid(False)
        set_axis_labels(ax, cat.cat_col, cat.val_col)
    else:
        ax.set_yticks(range(n), labels)
        ax.set_ylim(n - 0.5, -0.5)
        ax.yaxis.grid(False)
        set_axis_labels(ax, cat.val_col, cat.cat_col)
    show = legend is True or (legend == "auto" and cat.hue != cat.cat_col)
    if show and cat.hue is not None:
        handles = [handle(hi, c) for hi, c in enumerate(cat.colors)]
        add_legend_artists(ax, handles, cat.hues, cat.hue)


def _patch_props(kwargs: dict[str, Any]) -> dict[str, Any]:
    """The keyword arguments a ``Rectangle`` legend handle can take."""
    return {k: v for k, v in kwargs.items() if hasattr(Rectangle, f"set_{k}")}


def _patch_handle(**props: Any) -> Callable[[int, Any], Rectangle]:
    """Legend handles: unseen rectangles in each level's color."""

    def handle(hi: int, color: Any) -> Rectangle:
        patch_kws: dict[str, Any] = {"facecolor": color, **props}
        return Rectangle((0, 0), 0, 0, **patch_kws)

    return handle


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
    saturation: float = 0.75,
    width: float = 0.8,
    dodge: Dodge = "auto",
    gap: float = 0.0,
    capsize: float = 0.0,
    err_kws: dict[str, Any] | None = None,
    legend: LegendSpec = "auto",
    ax: Axes | None = None,
    **kwargs: Any,
) -> Axes:
    """Bar plot of an estimate per category, with an interval (bootstrap by default).

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
        saturation: Share of the colors' saturation to fill the bars with
        width: Total width of the bars at each category
        dodge: Place hue levels side by side (False overlaps them); "auto"
            dodges only when ``hue`` differs from the category column
        gap: Shrink each bar by this share of its width, about its center
        capsize: Error bar cap width, as a fraction of the category width
            shared among the dodged hue levels (independent of ``width``)
        err_kws: Keyword arguments for the error bar lines
        legend: Add a legend for ``hue``; "auto" omits it when ``hue`` is the
            category column
        ax: Axes to draw on (default: current axes)
        **kwargs: Passed to ``Axes.bar`` / ``Axes.barh``

    Returns:
        The Axes drawn on
    """
    ax = get_ax(ax)
    cat = _categorical(
        df, x, y, hue, order, hue_order, orient, palette, color, saturation
    )
    n_hue = len(cat.hues)
    dodged = cat.resolve_dodge(dodge) and n_hue > 1
    offsets, bar_width = _dodge_offsets(width, n_hue, dodged)
    cap_width = capsize / (n_hue if dodged else 1)
    bar_width *= 1 - gap
    err = _err_kws(err_kws, _ERR_COLOR)
    for (ci, hi), values in cat.groups.items():
        est, low, high = estimate_interval(values, estimator, errorbar, n_boot, seed)
        pos = ci + offsets[hi]
        bar_kws: dict[str, Any] = {"color": cat.colors[hi], **kwargs}
        if orient == "x":
            ax.bar(pos, est, width=bar_width, **bar_kws)
        else:
            ax.barh(pos, est, height=bar_width, **bar_kws)
        _draw_interval(ax, cat, pos, low, high, cap_width, err)
    _finish_categorical(ax, cat, legend, _patch_handle(**_patch_props(kwargs)))
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
    legend: LegendSpec = "auto",
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
        markersize: Marker size (default: ``linewidth`` times sqrt(2 pi))
        linewidth: Width of the lines and error bars (default: 1.8 times
            ``lines.linewidth``); markers take an edge 0.75 times as wide
        capsize: Error bar cap width, in category-axis units
        err_kws: Keyword arguments for the error bar lines
        legend: Add a legend for ``hue``; "auto" omits it when ``hue`` is the
            category column
        ax: Axes to draw on
        **kwargs: Passed to ``Axes.plot`` for the lines (e.g. ``alpha``)

    Returns:
        The Axes drawn on
    """
    ax = get_ax(ax)
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
    # lw= and ms= aliases in kwargs count as linewidth and markersize
    kwargs = normalize_kwargs(kwargs, Line2D)
    linewidth = kwargs.pop("linewidth", linewidth)
    markersize = kwargs.pop("markersize", markersize)
    lw = linewidth if linewidth is not None else 1.8 * plt.rcParams["lines.linewidth"]
    ms = markersize if markersize is not None else lw * np.sqrt(2 * np.pi)
    kwargs.setdefault("markeredgewidth", 0.75 * lw)
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
            markersize=ms,
            markeredgewidth=kwargs["markeredgewidth"],
            linewidth=lw,
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
    ax = get_ax(ax)
    data = complete_rows(df, x, y)
    if data.height < 2:
        raise ValueError(
            f"regplot needs at least two rows with both {x!r} and {y!r} present, "
            f"got {data.height}"
        )
    xv = data.get_column(x).cast(pl.Float64).to_numpy()
    yv = data.get_column(y).cast(pl.Float64).to_numpy()
    rgb = single_color(color)
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
    # seaborn's regplot sets both labels unconditionally, unlike its other plots
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

    Filled markers get seaborn's edges unless ``edgecolor`` or ``linewidth``
    is given: white, ``0.08 * sqrt(s)`` points wide (``s`` at its 10th
    percentile when sizes vary).

    Args:
        df: Polars DataFrame
        x: Column for the x-axis
        y: Column for the y-axis
        hue: Optional grouping column
        hue_order: Hue level order
        palette: Hue colors
        color: Point color without hue
        legend: Add a legend for ``hue``; "auto" omits it when ``hue`` is the
            category column
        ax: Axes to draw on
        **kwargs: Passed to ``Axes.scatter``

    Returns:
        The Axes drawn on
    """
    ax = get_ax(ax)
    data = complete_rows(df, x, y, hue)
    kwargs = _scatter_edges(normalize_kwargs(kwargs, PathCollection))
    if hue is None:
        rgb = single_color(color)
        ax.scatter(data.get_column(x), data.get_column(y), color=rgb, **kwargs)
    else:
        levels = category_levels(data.get_column(hue), hue_order)
        colors = resolve_colors(levels, palette)
        data = data.filter(pl.col(hue).is_in(levels))
        codes = (
            data.get_column(hue)
            .replace_strict(levels, list(range(len(levels))), return_dtype=pl.Int64)
            .to_numpy()
        )
        point_colors = np.asarray(colors)[codes] if codes.size else np.empty((0, 3))
        # One collection in row order, so levels interleave as the rows do
        ax.scatter(data.get_column(x), data.get_column(y), color=point_colors, **kwargs)
        if legend:
            handles = [_scatter_handle(c, kwargs) for c in colors]
            add_legend_artists(ax, handles, levels, hue)
    set_axis_labels(ax, x, y)
    return ax


def _scatter_handle(color: RGB, kwargs: dict[str, Any]) -> Line2D:
    """A legend marker drawn like the scatter's points.

    ``kwargs`` are the scatter's, alias-normalized; an edge color of "face"
    takes the marker's own color, as on the points.
    """
    sizes = np.atleast_1d(kwargs.get("s", plt.rcParams["lines.markersize"] ** 2))
    edge = kwargs.get("edgecolor")
    if edge is None or (isinstance(edge, str) and edge == "face"):
        edge = color
    width = kwargs.get("linewidth")
    return Line2D(
        [],
        [],
        color=color,
        linestyle="",
        marker=kwargs.get("marker", plt.rcParams["scatter.marker"]),
        markersize=float(np.sqrt(np.median(sizes))),
        markeredgecolor=edge,
        markeredgewidth=width if width is not None else 0,
        alpha=kwargs.get("alpha"),
    )


def _scatter_edges(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Add seaborn's default marker edge color and width to alias-normalized
    scatter kwargs."""
    marker = MarkerStyle(kwargs.get("marker", plt.rcParams["scatter.marker"]))
    out = dict(kwargs)
    if marker.is_filled() and "edgecolor" not in out:
        out["edgecolor"] = "w"
    if "linewidth" not in out:
        sizes = np.atleast_1d(out.get("s", plt.rcParams["lines.markersize"] ** 2))
        out["linewidth"] = 0.08 * np.sqrt(np.percentile(sizes, 10))
    return out


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
    levels together, so layers compare on the same scale. Layers are drawn
    last level first, so the first level is on top. Bar edges are thinned for
    narrow bins (a tenth of the narrowest bin's width in points, at most
    ``patch.linewidth``) unless ``linewidth`` is given.

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
        legend: Add a legend for ``hue``; "auto" omits it when ``hue`` is the
            category column
        ax: Axes to draw on
        **kwargs: Passed to ``Axes.bar``

    Returns:
        The Axes drawn on
    """
    if stat not in ("count", "frequency", "density", "probability", "percent"):
        raise ValueError(f"unknown stat {stat!r}")
    if multiple not in ("layer", "stack"):
        raise ValueError(f"multiple must be 'layer' or 'stack', got {multiple!r}")
    ax = get_ax(ax)
    data = complete_rows(df, x, hue)
    all_values = data.get_column(x).cast(pl.Float64).to_numpy()
    edges = np.histogram_bin_edges(all_values, bins=bins, range=binrange)  # type: ignore[arg-type]
    widths = np.diff(edges)
    total = all_values.size
    if hue is None:
        levels: list[Any] = [None]
        colors = [single_color(color)]
    else:
        levels = category_levels(data.get_column(hue), hue_order)
        colors = resolve_colors(levels, palette)
    if alpha is None:
        alpha = 0.5 if hue is not None and multiple == "layer" else 0.75
    kwargs = normalize_kwargs(kwargs, Rectangle)
    fixed_width = "linewidth" in kwargs
    base = np.zeros(len(widths))
    heights_by_level: list[np.ndarray] = []
    for level in levels:
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
        heights_by_level.append(heights)
    bottoms = []
    for heights in heights_by_level:
        bottoms.append(base)
        if multiple == "stack":
            base = base + heights
    # alpha fades the fill only; the edges stay opaque, as seaborn draws them
    bar_kws: dict[str, Any] = {"edgecolor": "white", **kwargs}
    drawn = []
    for i in reversed(range(len(levels))):
        level_kws: dict[str, Any] = {**bar_kws, "facecolor": (*colors[i], alpha)}
        drawn.extend(
            ax.bar(
                edges[:-1],
                heights_by_level[i],
                width=widths,
                bottom=bottoms[i],
                align="edge",
                **level_kws,
            )
        )
    if not fixed_width and drawn:
        ax.autoscale_view()
        narrow = int(np.argmin(widths))
        left = edges[narrow]
        span = ax.transData.transform([left + widths[narrow]] * 2)
        span = span - ax.transData.transform([left] * 2)
        points = 72 / ax.figure.dpi * abs(span[0])
        bar_width = min(0.1 * points, plt.rcParams["patch.linewidth"])
        for bar in drawn:
            bar.set_linewidth(bar_width)
    if legend and hue is not None:
        handle = _patch_handle(**_patch_props(bar_kws))
        handles = [handle(i, (*c, alpha)) for i, c in enumerate(colors)]
        add_legend_artists(ax, handles, levels, hue)
    set_axis_labels(ax, x, stat.capitalize())
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
    saturation: float = 0.75,
    fill: bool = True,
    width: float = 0.8,
    dodge: Dodge = "auto",
    gap: float = 0.0,
    linecolor: Any = "auto",
    linewidth: float | None = None,
    fliersize: float | None = None,
    whis: float = 1.5,
    showfliers: bool = True,
    legend: LegendSpec = "auto",
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
        saturation: Share of the colors' saturation to fill the boxes with
            (ignored when ``fill`` is False)
        fill: Fill the boxes; False draws every line in the box's color
        width: Total width of the boxes at each category
        dodge: Place hue levels side by side; "auto" dodges only when ``hue``
            differs from the category column
        gap: Shrink each box by this share of its width, about its center
        linecolor: Color of box edges, whiskers, and medians when filled;
            "auto" picks a gray darker than the darkest fill
        linewidth: Line width (default: ``lines.linewidth``)
        fliersize: Outlier marker size (0 hides them)
        whis: Whisker reach, in IQRs past the quartiles
        showfliers: Draw the outliers
        legend: Add a legend for ``hue``; "auto" omits it when ``hue`` is the
            category column
        ax: Axes to draw on
        **kwargs: Passed to ``Axes.boxplot``

    Returns:
        The Axes drawn on
    """
    ax = get_ax(ax)
    cat = _categorical(
        df,
        x,
        y,
        hue,
        order,
        hue_order,
        orient,
        palette,
        color,
        saturation if fill else 1.0,
    )
    n_hue = len(cat.hues)
    offsets, box_width = _dodge_offsets(width, n_hue, cat.resolve_dodge(dodge))
    box_width *= 1 - gap
    if isinstance(linecolor, str) and linecolor == "auto":
        linecolor = gray_complement(cat.colors)
    lw = linewidth if linewidth is not None else plt.rcParams["lines.linewidth"]
    for (ci, hi), values in cat.groups.items():
        main = cat.colors[hi]
        edge = linecolor if fill else main
        line = {"color": edge, "linewidth": lw}
        flier: dict[str, Any] = {"markeredgecolor": edge, "markeredgewidth": lw}
        if fliersize is not None:
            flier["markersize"] = fliersize
        boxprops: dict[str, Any] = (
            {"facecolor": main, "edgecolor": linecolor, "linewidth": lw}
            if fill
            else {"color": main, "linewidth": lw}
        )
        ax.boxplot(
            [values],
            positions=[ci + offsets[hi]],
            widths=box_width,
            whis=whis,
            showfliers=showfliers,
            patch_artist=fill,
            manage_ticks=False,
            orientation="vertical" if orient == "x" else "horizontal",
            boxprops=boxprops,
            whiskerprops=line,
            capprops=line,
            medianprops=line,
            flierprops=flier,
            **kwargs,
        )
    if fill:
        handle = _patch_handle(edgecolor=linecolor, linewidth=lw)
    else:

        def handle(hi: int, c: RGB) -> Rectangle:
            return Rectangle((0, 0), 0, 0, facecolor="none", edgecolor=c, linewidth=lw)

    _finish_categorical(ax, cat, legend, handle)
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
    legend: LegendSpec = "auto",
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
        legend: Add a legend for ``hue``; "auto" omits it when ``hue`` is the
            category column
        ax: Axes to draw on
        **kwargs: Passed to ``Axes.scatter``

    Returns:
        The Axes drawn on
    """
    ax = get_ax(ax)
    cat = _categorical(df, x, y, hue, order, hue_order, orient, palette, color)
    width = 0.8
    n_hue = len(cat.hues)
    offsets, _ = _dodge_offsets(width, n_hue, dodge)
    # zorder 3 puts the points above bars and error bars drawn on the same axes
    scatter_kws: dict[str, Any] = {
        "s": size**2,
        "linewidth": linewidth,
        "edgecolor": edgecolor,
        "zorder": 3,
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
    saturation: float = 0.75,
    width: float = 0.8,
    dodge: Dodge = "auto",
    gap: float = 0.0,
    linecolor: Any = "auto",
    linewidth: float | None = None,
    width_method: Literal["exponential", "linear", "area"] = "exponential",
    k_depth: KDepth = "tukey",
    outlier_prop: float = 0.007,
    trust_alpha: float = 0.05,
    showfliers: bool = True,
    box_kws: dict[str, Any] | None = None,
    line_kws: dict[str, Any] | None = None,
    flier_kws: dict[str, Any] | None = None,
    legend: LegendSpec = "auto",
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
        saturation: Share of the colors' saturation for the boxes' base color
        width: Total width of the widest box at each category
        dodge: Place hue levels side by side; "auto" dodges only when ``hue``
            differs from the category column
        gap: Shrink each set of boxes by this share of its width
        linecolor: Box edge and median color; "auto" picks a gray darker than
            the darkest fill
        linewidth: Box edge width (default: half ``lines.linewidth``)
        width_method: How box width shrinks with depth: "exponential",
            "linear", or "area"
        k_depth: Depth rule (see ``letter_values``)
        outlier_prop: Outlier rate for ``k_depth="proportion"``
        trust_alpha: Confidence level for ``k_depth="trustworthy"``
        showfliers: Draw points beyond the outermost box
        box_kws: Keyword arguments for the boxes' ``PatchCollection``
        line_kws: Keyword arguments for the median line's ``Axes.plot``
        flier_kws: Keyword arguments for the outliers' ``Axes.scatter``
        legend: Add a legend for ``hue``; "auto" omits it when ``hue`` is the
            category column
        ax: Axes to draw on
        **kwargs: Passed to the boxes' ``PatchCollection``

    Returns:
        The Axes drawn on
    """
    if width_method not in ("exponential", "linear", "area"):
        raise ValueError(f"unknown width_method {width_method!r}")
    ax = get_ax(ax)
    cat = _categorical(
        df, x, y, hue, order, hue_order, orient, palette, color, saturation
    )
    n_hue = len(cat.hues)
    offsets, box_width = _dodge_offsets(width, n_hue, cat.resolve_dodge(dodge))
    box_width *= 1 - gap
    if isinstance(linecolor, str) and linecolor == "auto":
        linecolor = gray_complement(cat.colors)
    lw = linewidth if linewidth is not None else 0.5 * plt.rcParams["lines.linewidth"]
    user_box = normalize_kwargs({**kwargs, **(box_kws or {})}, PatchCollection)
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
        collection_kws = {
            **normalize_kwargs(
                {"facecolors": boxcolors, "edgecolors": linecolor, "linewidth": lw},
                PatchCollection,
            ),
            **user_box,
        }
        collection = PatchCollection(boxes, **collection_kws)
        ax.add_collection(collection, autolim=False)
        ax.update_datalim(
            np.column_stack([vals, vals]),
            updatex=orient == "y",
            updatey=orient == "x",
        )
        med = lv["median"]
        half = box_width / 2
        median_kws: dict[str, Any] = {
            "color": linecolor,
            "solid_capstyle": "butt",
            "linewidth": 1.25 * lw,
            **(line_kws or {}),
        }
        ax.plot(*cat.xy([center - half, center + half], [med, med]), **median_kws)
        if showfliers and lv["fliers"].size:
            fliers = lv["fliers"]
            flier_props: dict[str, Any] = {
                "facecolor": "none",
                "edgecolor": ".45",
                "s": 25,
                **(flier_kws or {}),
            }
            ax.scatter(*cat.xy(np.full(fliers.size, center), fliers), **flier_props)
    ax.autoscale_view(scalex=orient == "y", scaley=orient == "x")
    legend_props = {
        **_patch_props(normalize_kwargs(box_kws or {}, Rectangle)),
        "linewidth": lw,
        "edgecolor": linecolor,
    }
    _finish_categorical(ax, cat, legend, _patch_handle(**legend_props))
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
    ax = get_ax(ax)
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
