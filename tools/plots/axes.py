"""Axis formatting: ticks, dates, legends, grids, and spines."""

from typing import Any, Literal, cast

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.legend import Legend
from matplotlib.ticker import FuncFormatter


def remove_yaxis_ticks(ax: plt.Axes, major: bool = True, minor: bool = True) -> None:
    """Remove y-axis tick marks while keeping tick labels.

    Args:
        ax: Matplotlib axes object
        major: Whether to remove major tick marks
        minor: Whether to remove minor tick marks
    """
    if major:
        ax.tick_params(axis="y", which="major", length=0)
    if minor:
        ax.tick_params(axis="y", which="minor", length=0)


def remove_xaxis_ticks(ax: plt.Axes, major: bool = True, minor: bool = True) -> None:
    """Remove x-axis tick marks while keeping tick labels.

    Args:
        ax: Matplotlib axes object
        major: Whether to remove major tick marks
        minor: Whether to remove minor tick marks
    """
    if major:
        ax.tick_params(axis="x", which="major", length=0)
    if minor:
        ax.tick_params(axis="x", which="minor", length=0)


def xaxis_date_formatter(ax: plt.Axes, period: str = "weekly") -> plt.Axes:
    """Configure date-based x-axis tick formatting.

    Args:
        ax: Matplotlib axes object
        period: Tick interval style ('monthly', 'monthly-labels', 'weekly',
            'weekly-labels', 'daily', 'daily-labels')

    Returns:
        Modified axes object
    """

    day_fmt = mdates.DateFormatter("%d")

    def d_fmt(x: float, pos: int | None = None) -> str:
        month_letter = day_fmt(x)
        if month_letter.startswith("0"):
            month_letter = month_letter[1:]
        return month_letter

    if period == "monthly":
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter(""))

    elif period == "monthly-labels":
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter("%b"))

    elif period == "weekly":
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter("%b"))

    elif period == "weekly-labels":
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("\n%b"))
        ax.xaxis.set_minor_locator(
            mdates.WeekdayLocator(byweekday=mdates.SU, interval=1)
        )
        ax.xaxis.set_minor_formatter(FuncFormatter(d_fmt))

    elif period == "daily":
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.xaxis.set_minor_locator(
            mdates.WeekdayLocator(byweekday=mdates.MO, interval=1)
        )
        ax.xaxis.set_minor_formatter(mdates.DateFormatter(""))

    elif period == "daily-labels":
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("\n%b"))
        ax.xaxis.set_minor_locator(
            mdates.WeekdayLocator(byweekday=mdates.MO, interval=1)
        )
        ax.xaxis.set_minor_formatter(FuncFormatter(d_fmt))

    return ax


def add_legend(
    ax: plt.Axes,
    loc: str = "upper right",
    bbox_to_anchor: tuple[float, float] = (1.025, 1.025),
    ncol: int = 1,
    columnspacing: float = 0.5,
    title: str = "",
    alignment: str = "left",
    fontsize: int | float = 7,
    frameon: bool = False,
    fancybox: bool = False,
    facecolor: str = "white",
    edgecolor: str = "gray",
    shadow: bool = False,
    markerscale: int | float = 1,
    handletextpad: float = 0.5,
    handlelength: int | float = 2,
    labelspacing: float = 0.1,
    handle_linewidth: float = 1.25,
    label_order: list[str] | None = None,
    zorder: int | float = 100,
) -> Legend:
    """Add legend to axes with enhanced styling options.

    Args:
        ax: Matplotlib axes object
        loc: Legend location string
        bbox_to_anchor: Bounding box anchor point
        ncol: Number of columns in legend
        columnspacing: Spacing between legend columns
        title: Legend title text
        alignment: Title horizontal alignment
        fontsize: Legend text font size
        frameon: Whether to draw a frame around the legend
        fancybox: Whether to use rounded frame corners
        facecolor: Frame background color
        edgecolor: Frame edge color
        shadow: Whether to draw a shadow behind the legend
        markerscale: Marker size scale factor
        handletextpad: Padding between handle and text
        handlelength: Length of legend handles
        labelspacing: Vertical spacing between legend entries
        handle_linewidth: Line width for legend handles
        label_order: Custom ordering of legend labels
        zorder: Drawing order for the legend

    Returns:
        Legend object
    """

    handles, labels = ax.get_legend_handles_labels()
    if label_order is not None and len(label_order) > 0:
        handles, labels = reorder_legend(handles, labels, label_order)
    legend = ax.legend(
        handles,
        labels,
        title=title,
        alignment=alignment,
        ncol=ncol,
        columnspacing=columnspacing,
        facecolor=facecolor,
        edgecolor=edgecolor,
        fontsize=fontsize,
        # matplotlib types loc as a narrow Literal; we accept the same names as str
        loc=loc,  # type: ignore[bad-argument-type]
        bbox_to_anchor=bbox_to_anchor,
        frameon=frameon,
        fancybox=fancybox,
        shadow=shadow,
        handletextpad=handletextpad,
        handlelength=handlelength,
        labelspacing=labelspacing,
        markerscale=markerscale,
    )

    # Set zorder to ensure legend is drawn on top
    legend.set_zorder(zorder)

    for lh in legend.legend_handles:
        if lh is None:
            continue
        # legend_handles items are Artists; set_linewidth exists on Line2D and
        # the patch/collection handles used here but isn't on the base Artist type.
        lh.set_linewidth(handle_linewidth)  # type: ignore[attr-defined]
        lh.set_alpha(1)

    return legend


def reorder_legend(
    handles: list[Artist] | None = None,
    labels: list[str] | None = None,
    order: list[str] | None = None,
    unique: bool = False,
) -> tuple[tuple[Artist, ...], tuple[str, ...]]:
    """Reorder legend handles and labels with an ordered list.

    Args:
        handles: Handles obtained via ax.get_legend_handles_labels()
        labels: Labels obtained via ax.get_legend_handles_labels()
        order: List of labels in desired order, strings must match
        unique: Whether to drop duplicates and keep first label instance

    Returns:
        Tuple of (handles, labels) sorted in the specified order
    """
    handles = handles or []
    labels = labels or []

    # Sort both labels and handles by labels
    sorted_labels, sorted_handles = zip(
        *sorted(zip(labels, handles), key=lambda pair: pair[0])
    )
    if order is not None:
        # Sort according to a given list (not necessarily complete)
        keys = dict(zip(order, range(len(order))))
        sorted_labels, sorted_handles = zip(
            *sorted(
                zip(sorted_labels, sorted_handles),
                key=lambda pair, keys=keys: keys.get(pair[0], np.inf),
            )
        )

    # Keep only the first handle for each label
    if unique:
        seen: set[str] = set()
        deduped = [
            (label, handle)
            for label, handle in zip(sorted_labels, sorted_handles)
            if not (label in seen or seen.add(label))
        ]
        sorted_labels, sorted_handles = zip(*deduped)
    return sorted_handles, sorted_labels


def add_right_angle(
    ax: plt.Axes,
    x: float,
    y: float,
    linestyle: str = "-.",
    color: str = "r",
    fmt: str = ".3f",
) -> plt.Axes:
    """Add right-angle reference lines at a point with annotations.

    Draws vertical and horizontal dashed lines from the axes edges to the
    specified (x, y) point and adds coordinate annotations.

    Args:
        ax: Matplotlib axes object
        x: X coordinate for the reference point
        y: Y coordinate for the reference point
        linestyle: Line style for reference lines
        color: Line color for reference lines
        fmt: Format string for coordinate annotations

    Returns:
        Modified axes object
    """
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.plot(
        [x, x],
        [ymin, y],
        linestyle=linestyle,
        color=color,
    )  # Vertical
    ax.plot(
        [xmin, x],
        [y, y],
        linestyle=linestyle,
        color=color,
    )  # Horizontal

    # Add annotations
    ax.annotate(
        f"x: {x:{fmt}}",
        xy=(x, y),
        xytext=(5, -5),
        textcoords="offset points",
        ha="left",
        va="top",
    )
    ax.annotate(
        f"y: {y:{fmt}}",
        xy=(x, y),
        xytext=(-5, 5),
        textcoords="offset points",
        ha="right",
        va="bottom",
    )

    return ax


def _apply_grid(
    ax: plt.Axes,
    which: str,
    axis: str,
    zorder: int | float,
    alpha: float,
    linewidth: float,
    color: str | None,
    linestyle: str,
) -> plt.Axes:
    """Apply grid lines to axes.

    Args:
        ax: Matplotlib axes object
        which: Grid line type ('major', 'minor', or 'both')
        axis: Which axis to apply grid to ('both', 'x', 'y')
        zorder: Z-order for grid lines
        alpha: Grid line transparency
        linewidth: Grid line width
        color: Grid line color, or None for default
        linestyle: Grid line style ('-', '--', '-.', ':')

    Returns:
        Modified axes object
    """
    grid_kws: dict[str, Any] = {
        "alpha": alpha,
        "linewidth": linewidth,
        "zorder": zorder,
        "linestyle": linestyle,
    }
    if color is not None:
        grid_kws["color"] = color
    # axis/which are accepted as str at the public API but matplotlib types them
    # as Literals; cast at this boundary since values are validated by callers.
    ax.grid(
        axis=cast(Literal["both", "x", "y"], axis),
        which=cast(Literal["both", "major", "minor"], which),
        **grid_kws,
    )
    return ax


def add_grid_major(
    ax: plt.Axes,
    axis: str = "both",
    zorder: int | float = 0,
    alpha: float = 0.75,
    linewidth: float = 0.75,
    color: str | None = None,
    linestyle: str = "-",
) -> plt.Axes:
    """Add major grid lines to axes.

    Args:
        ax: Matplotlib axes object
        axis: Which axis to apply grid to ('both', 'x', 'y')
        zorder: Z-order for grid lines
        alpha: Grid line transparency
        linewidth: Grid line width
        color: Grid line color, or None for default
        linestyle: Grid line style

    Returns:
        Modified axes object
    """
    return _apply_grid(ax, "major", axis, zorder, alpha, linewidth, color, linestyle)


def add_grid_minor(
    ax: plt.Axes,
    axis: str = "both",
    zorder: int | float = 0,
    alpha: float = 0.25,
    linewidth: float = 0.5,
    color: str | None = None,
    linestyle: str = "-",
) -> plt.Axes:
    """Add minor grid lines to axes.

    Args:
        ax: Matplotlib axes object
        axis: Which axis to apply grid to ('both', 'x', 'y')
        zorder: Z-order for grid lines
        alpha: Grid line transparency
        linewidth: Grid line width
        color: Grid line color, or None for default
        linestyle: Grid line style

    Returns:
        Modified axes object
    """
    return _apply_grid(ax, "minor", axis, zorder, alpha, linewidth, color, linestyle)


def add_grid(
    ax: plt.Axes,
    axis: str = "both",
    which: str = "both",
    zorder: int | float = 0,
    alpha_major: float = 0.75,
    alpha_minor: float = 0.25,
    linewidth_major: float = 0.75,
    linewidth_minor: float = 0.5,
    color_major: str | None = None,
    color_minor: str | None = None,
    linestyle_major: str = "-",
    linestyle_minor: str = "-",
    axisbelow: bool = True,
) -> plt.Axes:
    """Add both major and minor grid lines to axis.

    Args:
        ax: Matplotlib axes object
        axis: Which axis to apply grid to ('both', 'x', 'y')
        zorder: Z-order for grid lines (lower values are drawn below)
        which: Which grid lines to add ('major', 'minor', 'both')
        alpha_major: Alpha transparency for major grid lines
        alpha_minor: Alpha transparency for minor grid lines
        linewidth_major: Line width for major grid lines
        linewidth_minor: Line width for minor grid lines
        color_major: Color for major grid lines (None uses default)
        color_minor: Color for minor grid lines (None uses default)
        linestyle_major: Line style for major grid lines ('-', '--', '-.', ':')
        linestyle_minor: Line style for minor grid lines ('-', '--', '-.', ':')
        axisbelow: If True, send gridlines behind plot elements

    Returns:
        Modified axes object
    """
    if which in ["major", "both"]:
        add_grid_major(
            ax,
            axis=axis,
            zorder=zorder,
            alpha=alpha_major,
            linewidth=linewidth_major,
            color=color_major,
            linestyle=linestyle_major,
        )
    if which in ["minor", "both"]:
        add_grid_minor(
            ax,
            axis=axis,
            zorder=zorder,
            alpha=alpha_minor,
            linewidth=linewidth_minor,
            color=color_minor,
            linestyle=linestyle_minor,
        )
    ax.set_axisbelow(axisbelow)
    return ax


def add_spines(
    ax: plt.Axes,
    on_spines: list[str] | None = None,
    off_spines: list[str] | None = None,
    linewidth: float = 0.75,
) -> plt.Axes:
    """Configure axes spine visibility and width.

    Args:
        ax: Matplotlib axes object
        on_spines: List of spines to set width on (default: all four)
        off_spines: List of spines to hide (default: top and right)
        linewidth: Width for visible spines and ticks

    Returns:
        Modified axes object
    """
    on_spines = on_spines or ["top", "right", "left", "bottom"]
    off_spines = off_spines or ["top", "right"]
    for spine in on_spines:
        ax.spines[spine].set_linewidth(linewidth)  # Set spine width
        ax.tick_params(width=linewidth)  # Set tick width
        if spine in off_spines:
            ax.spines[spine].set_visible(False)
    return ax
