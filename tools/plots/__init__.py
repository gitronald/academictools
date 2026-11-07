"""Visualization: plotting, axis formatting, and heatmaps."""

from .axes import (
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
from .core import (
    darken_hex_color,
    enhanced_boxplot,
    enhanced_lineplot,
    enhanced_scatterplot,
    noisy_limits,
    set_fig_text,
    smart_plot_data,
)
from .heatmaps import get_heatmap_data, plot_heatmap
from .settings import (
    LARGE_SIZE,
    MEDIUM_SIZE,
    NARROW_FIG_WIDTH,
    PAGE_WIDTH,
    RC_PARAMS,
    SMALL_SIZE,
    WIDE_FIG_WIDTH,
    box_kws,
    heatmap_kws_proportion,
    pointplot_kws,
    strip_kws,
)

__all__ = [
    # Settings
    "PAGE_WIDTH",
    "WIDE_FIG_WIDTH",
    "NARROW_FIG_WIDTH",
    "SMALL_SIZE",
    "MEDIUM_SIZE",
    "LARGE_SIZE",
    "RC_PARAMS",
    "strip_kws",
    "box_kws",
    "pointplot_kws",
    "heatmap_kws_proportion",
    # Core
    "smart_plot_data",
    "enhanced_scatterplot",
    "enhanced_lineplot",
    "enhanced_boxplot",
    "darken_hex_color",
    "set_fig_text",
    "noisy_limits",
    # Axes
    "remove_yaxis_ticks",
    "remove_xaxis_ticks",
    "xaxis_date_formatter",
    "add_legend",
    "reorder_legend",
    "add_right_angle",
    "add_grid_major",
    "add_grid_minor",
    "add_grid",
    "add_spines",
    # Heatmaps
    "get_heatmap_data",
    "plot_heatmap",
]
