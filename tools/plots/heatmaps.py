"""Heatmap plotting utilities."""

import math
from typing import Any

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure


def get_heatmap_data(
    df: pl.DataFrame,
    index: str,
    columns: str,
    filter_col: str | None = None,
    norm: str = "row",
) -> pl.DataFrame:
    """Build a normalized count pivot (index x columns) for a heatmap.

    Counts rows per (index, columns) pair, spreads ``columns`` wide, and
    normalizes either row-wise or by the grand total. Pre-filter the frame for
    any subset of interest before calling.

    Args:
        df: Polars DataFrame
        index: Column whose values become the pivot rows
        columns: Column whose values become the pivot columns
        filter_col: Optional boolean column; only its truthy rows are counted
        norm: Normalization method ('row' so each row sums to 1, or 'full' so
            the whole table sums to 1)

    Returns:
        Polars pivot table: the ``index`` column followed by one column per
        ``columns`` value (sorted), with values normalized per ``norm``.

    Raises:
        ValueError: If norm is not 'row' or 'full'
    """
    if norm not in ["row", "full"]:
        raise ValueError(f"norm must be one of ['row', 'full'], got {norm}")

    work = df.filter(pl.col(filter_col)) if filter_col else df
    col_values = work.get_column(columns).unique().sort().to_list()
    col_order = pl.Series(columns, col_values).cast(pl.String).to_list()
    work = work.with_columns(pl.col(columns).cast(pl.String))
    pivot = (
        work.group_by(index, columns)
        .len()
        .pivot(on=columns, index=index, values="len")
        .sort(index)
        .select([index, *col_order])
    )

    value_cols = [c for c in pivot.columns if c != index]
    if norm == "row":
        row_sum = pl.sum_horizontal(value_cols)
        pivot = pivot.with_columns([(pl.col(c) / row_sum).alias(c) for c in value_cols])
    elif norm == "full":
        grand = sum(pivot[c].sum() for c in value_cols)
        pivot = pivot.with_columns([(pl.col(c) / grand).alias(c) for c in value_cols])

    return pivot


def plot_heatmap(
    plot_data: pl.DataFrame,
    data_min: float,
    data_max: float,
    log_scale: bool = True,
    cbar: bool = True,
    annot_kws: dict[str, Any] | None = {"size": 7},
    fmt: str = ".1%",
    square: bool = False,
    figsize: tuple[float, float] = (7, 3),
    index_col: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot a heatmap with optional log-scaled colorbar.

    Args:
        plot_data: Polars pivot table to visualize; its row-label column (see
            ``index_col``) supplies the y tick labels and the remaining columns
            are the heatmap columns.
        data_min: Minimum data value for colorbar range
        data_max: Maximum data value for colorbar range
        log_scale: Whether to apply log normalization to colors
        cbar: Whether to show the colorbar
        annot_kws: Annotation keyword arguments for seaborn heatmap
        fmt: Format string for cell annotations
        square: Whether to force square cells
        figsize: Figure size as (width, height) tuple
        index_col: Column holding the row labels (defaults to the first column)

    Returns:
        Tuple of (figure, axes) objects
    """
    index_col = index_col or plot_data.columns[0]
    yticklabels = plot_data[index_col].to_list()
    xticklabels = [c for c in plot_data.columns if c != index_col]
    matrix = plot_data.drop(index_col).to_numpy()
    log_norm = None
    cbar_ticks = None
    if log_scale:
        # Log normalize colorbar with major ticks at powers of 10
        log_norm = LogNorm(vmin=data_min, vmax=data_max)
        cbar_range = range(
            math.floor(math.log10(data_min)), 1 + math.ceil(math.log10(data_max))
        )
        cbar_ticks = [math.pow(10, i) for i in cbar_range]
        vmin = vmax = None
    else:
        vmin, vmax = data_min, data_max

    # Canvas
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    # Colorbar settings
    if cbar:
        cbar_kws = {
            "orientation": "vertical",
            "pad": 0.05,
            "shrink": 0.75,
        }
    else:
        cbar_kws = None

    # Plot
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="binary",
        annot=True if annot_kws else False,
        fmt=fmt,
        annot_kws=annot_kws,
        linewidths=0.5,
        linecolor="whitesmoke",
        cbar=cbar,
        cbar_kws=cbar_kws,
        norm=log_norm,
        vmin=vmin,
        vmax=vmax,
        square=square,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
    )

    # Ticks and labels
    ax.tick_params(axis="y", rotation=0)
    ax.tick_params(axis="both", which="both", length=0, pad=5)
    ax.set(ylabel="Year", xlabel="Month")

    if cbar:
        # Colorbar axis
        colorbar = ax.collections[0].colorbar
        if colorbar is not None:
            if cbar_ticks is not None:
                colorbar.set_ticks(cbar_ticks)
            ax_colorbar = colorbar.ax
            ax_colorbar.tick_params(which="major", length=4, width=0.75)
            ax_colorbar.tick_params(which="minor", length=2.5, width=0.5)

    # Spines
    spine_color = "#444654"
    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(0.5)
        ax.spines[spine].set_color(spine_color)

    # Mark NaN spots
    ax.patch.set(hatch="xxxx", edgecolor="whitesmoke")

    return fig, ax
