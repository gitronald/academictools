"""Core plotting functions with polars support."""

import datetime
from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure

from tools.plots.charts import boxplot, scatterplot
from tools.plots.estimate import ErrorBar, Estimator, estimate_interval
from tools.plots.palettes import Palette, resolve_colors


def enhanced_scatterplot(df: pl.DataFrame, x: str, y: str, **kwargs: Any) -> Axes:
    """Create a scatter plot from a polars DataFrame.

    Args:
        df: Polars DataFrame containing the data
        x: Column name for x-axis
        y: Column name for y-axis
        **kwargs: Passed to ``tools.plots.scatterplot`` (``hue``, ``palette``,
            ``ax``, or ``Axes.scatter`` keywords)

    Returns:
        matplotlib Axes object

    Examples:
        >>> df = pl.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        >>> enhanced_scatterplot(df, 'x', 'y')
    """
    return scatterplot(df, x, y, **kwargs)


def enhanced_lineplot(
    df: pl.DataFrame,
    x: str,
    y: str,
    *,
    hue: str | None = None,
    hue_order: Sequence[Any] | None = None,
    palette: Palette = None,
    color: Any = None,
    estimator: Estimator = "mean",
    errorbar: ErrorBar = ("ci", 95),
    n_boot: int = 1000,
    seed: int | None = None,
    legend: bool = True,
    ax: Axes | None = None,
    **kwargs: Any,
) -> Axes:
    """Create a line plot of ``y`` over sorted ``x``, one line per ``hue`` level.

    Repeated ``x`` values are aggregated with ``estimator``, and the interval
    from ``errorbar`` is drawn as a shaded band, as seaborn's ``lineplot`` does.

    Args:
        df: Polars DataFrame containing the data
        x: Column name for x-axis
        y: Column name for y-axis
        hue: Optional grouping column, one line per level
        hue_order: Hue level order
        palette: Hue colors
        color: Line color without hue
        estimator: Statistic per x value: a numpy reduction name or a callable
        errorbar: Interval spec (see ``estimate_interval``); None for no band
        n_boot: Bootstrap resamples for a ``"ci"`` interval
        seed: Seed for the bootstrap
        legend: Add a legend for ``hue``
        ax: Axes to draw on (default: current axes)
        **kwargs: Passed to ``Axes.plot``

    Returns:
        matplotlib Axes object
    """
    ax = ax if ax is not None else plt.gca()
    cols = list(dict.fromkeys(c for c in (x, y, hue) if c is not None))
    data = df.select(cols).drop_nulls()
    if hue is None:
        levels: list[Any] = [None]
        colors = [to_rgb(color) if color is not None else resolve_colors([0])[0]]
    else:
        series = data.get_column(hue)
        levels = (
            list(hue_order)
            if hue_order is not None
            else series.unique(maintain_order=True).to_list()
        )
        colors = resolve_colors(levels, palette)
    for level, rgb in zip(levels, colors):
        part = data if hue is None else data.filter(pl.col(hue) == level)
        xs: list[Any] = []
        est: list[float] = []
        low: list[float] = []
        high: list[float] = []
        for key, sub in part.group_by(x):
            xs.append(key[0])
            values = sub.get_column(y).cast(pl.Float64).to_numpy()
            e, lo, hi = estimate_interval(values, estimator, errorbar, n_boot, seed)
            est.append(e)
            low.append(lo)
            high.append(hi)
        order = np.argsort(np.asarray(xs))
        xs_sorted = np.asarray(xs)[order]
        line_kws: dict[str, Any] = {"color": rgb, **kwargs}
        if level is not None:
            line_kws.setdefault("label", str(level))
        ax.plot(xs_sorted, np.asarray(est)[order], **line_kws)
        if errorbar is not None:
            ax.fill_between(
                xs_sorted,
                np.asarray(low)[order],
                np.asarray(high)[order],
                color=rgb,
                alpha=0.2,
                linewidth=0,
            )
    if legend and hue is not None:
        ax.legend(title=hue)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    return ax


def enhanced_boxplot(df: pl.DataFrame, x: str, y: str, **kwargs: Any) -> Axes:
    """Create a box plot from a polars DataFrame.

    Args:
        df: Polars DataFrame containing the data
        x: Column name for x-axis (categorical)
        y: Column name for y-axis (numeric)
        **kwargs: Passed to ``tools.plots.boxplot`` (``hue``, ``order``,
            ``palette``, ``ax``, ...)

    Returns:
        matplotlib Axes object
    """
    return boxplot(df, x, y, **kwargs)


def darken_hex_color(hex_color: str, factor: float = 0.5) -> str:
    """Darken a hex color by a given factor.

    Args:
        hex_color: Hex color string (e.g. '#ff0000')
        factor: Darkening factor between 0 and 1, lower is darker

    Returns:
        Darkened hex color string
    """

    def hex_to_rgb(hex_color: str) -> tuple[int, ...]:
        # Convert hex color string to RGB values
        return tuple(int(hex_color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))

    def darker_rgb(rgb: tuple[int, ...], factor: float) -> tuple[int, ...]:
        # Darken the color by multiplying each RGB value by the given factor
        return tuple(int(c * factor) for c in rgb)

    def rgb_to_hex(rgb: tuple[int, ...]) -> str:
        # Convert RGB values to hex color string
        return "#{:02x}{:02x}{:02x}".format(*rgb)

    red, green, blue = hex_to_rgb(hex_color)
    red, green, blue = darker_rgb((red, green, blue), factor)
    return rgb_to_hex((red, green, blue))


def set_fig_text(fig: Figure, s: str, x: float, y: float, **kwargs: Any) -> Figure:
    """Add text with white background to a figure.

    Args:
        fig: Matplotlib figure object
        s: Text string to display
        x: X position in figure coordinates (0-1)
        y: Y position in figure coordinates (0-1)
        **kwargs: Additional arguments passed to fig.text

    Returns:
        Modified figure object
    """
    figtext = fig.text(x=x, y=y, s=s, **kwargs)
    figtext.set_bbox(dict(facecolor="white", alpha=0.5, edgecolor="none"))
    return fig


def noisy_limits(
    series: pl.Series, noise: float | datetime.timedelta = 0.05
) -> tuple[Any, Any]:
    """Compute axis limits with noise padding around min and max.

    Args:
        series: Polars Series to compute limits for
        noise: Padding factor (float) or timedelta for datetime series

    Returns:
        Tuple of (lower, upper) limits with noise applied
    """
    # min()/max() return a broad union (numeric or temporal); treat as Any since
    # the padding branches handle both float and datetime values.
    low: Any = series.min()
    high: Any = series.max()

    if isinstance(noise, datetime.timedelta):
        return (low - noise, high + noise)
    return (low - (noise * abs(low)), high + (noise * abs(high)))
