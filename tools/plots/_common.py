"""Data and axes helpers shared by the plotting modules (``charts``, ``core``)."""

import colorsys
from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb

from tools.plots.palettes import RGB, resolve_colors


def get_ax(ax: Axes | None) -> Axes:
    """``ax``, or the current axes when it is None."""
    return ax if ax is not None else plt.gca()


def complete_rows(df: pl.DataFrame, *cols: str | None) -> pl.DataFrame:
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


def single_color(color: Any) -> RGB:
    """``color`` as RGB, or the palette's first color when it is None."""
    return to_rgb(color) if color is not None else resolve_colors([0])[0]


def category_levels(series: pl.Series, order: Sequence[Any] | None) -> list[Any]:
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


def gray_complement(colors: Sequence[Any]) -> RGB:
    """A gray for lines drawn over fills of ``colors``, as seaborn picks it.

    The gray's lightness is 0.6 times the lowest HLS lightness among the
    colors, so lines stay visible against the darkest fill.
    """
    lightness = min(colorsys.rgb_to_hls(*to_rgb(c))[1] for c in colors)
    gray = 0.6 * lightness
    return (gray, gray, gray)


def add_legend_artists(
    ax: Axes, handles: Sequence[Artist], labels: Sequence[Any], title: str
) -> None:
    """Add labeled, unseen legend artists to ``ax`` and draw a legend from them.

    The artists stay on the axes, so ``ax.get_legend_handles_labels()`` (and a
    later ``add_legend``) finds them, as it finds seaborn's.
    """
    for handle, label in zip(handles, labels):
        handle.set_label(str(label))
        ax.add_artist(handle)
    if handles:
        ax.legend(title=title)


def set_axis_labels(ax: Axes, xlabel: str | None, ylabel: str | None) -> None:
    """Label the axes as seaborn does.

    An axis that already has a label keeps it. A new label is visible only when
    the axis shows tick labels, so the inner panels of a grid made with shared
    axes, whose tick labels matplotlib hides, carry no visible label. ``None``
    leaves that axis alone.
    """
    for axis, label in ((ax.xaxis, xlabel), (ax.yaxis, ylabel)):
        if label is None or axis.get_label_text():
            continue
        visible = any(tick.get_visible() for tick in axis.get_ticklabels())
        axis.set_label_text(label, visible=visible)
