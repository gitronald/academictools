"""Color palettes and style presets on plain matplotlib."""

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib as mpl
import numpy as np
from matplotlib.colors import to_rgb

# Qualitative matplotlib colormaps and how many distinct colors each defines.
# Their colors are indexed exactly rather than interpolated.
QUALITATIVE_SIZES: dict[str, int] = {
    "tab10": 10,
    "tab20": 20,
    "tab20b": 20,
    "tab20c": 20,
    "Set1": 9,
    "Set2": 8,
    "Set3": 12,
    "Accent": 8,
    "Paired": 12,
    "Pastel1": 9,
    "Pastel2": 8,
    "Dark2": 8,
}

RGB = tuple[float, float, float]
Palette = str | Sequence[Any] | Mapping[Any, Any] | None


def color_palette(name: str, n_colors: int = 6) -> list[RGB]:
    """Sample ``n_colors`` RGB colors from a named matplotlib colormap.

    Samples the way seaborn's ``color_palette(name, n)`` does for matplotlib
    colormaps: a qualitative map (``tab10``, ``Set2``, ...) gives its first
    ``n_colors`` colors exactly, repeating from the start when asked for more
    than it defines; a continuous map gives evenly spaced samples that skip
    both ends, for contrast at the extremes.

    Args:
        name: Name of a registered matplotlib colormap (``_r`` reverses it)
        n_colors: Number of colors to return

    Returns:
        List of (r, g, b) tuples with channels in [0, 1]
    """
    cmap = mpl.colormaps[name]
    if name in QUALITATIVE_SIZES:
        bins = np.linspace(0, 1, QUALITATIVE_SIZES[name])[:n_colors]
    else:
        bins = np.linspace(0, 1, int(n_colors) + 2)[1:-1]
    colors = [(float(r), float(g), float(b)) for r, g, b in cmap(bins)[:, :3]]
    return [colors[i % len(colors)] for i in range(int(n_colors))]


def resolve_colors(levels: Sequence[Any], palette: Palette = None) -> list[RGB]:
    """Assign one color per level from a palette specification.

    Args:
        levels: Ordered group levels needing a color each
        palette: None for matplotlib's color cycle, a colormap name, a list of
            colors (cycled when shorter than ``levels``), or a dict mapping each
            level to a color

    Returns:
        One RGB tuple per level, in ``levels`` order
    """
    n = len(levels)
    if palette is None:
        cycle = [c["color"] for c in mpl.rcParams["axes.prop_cycle"]]
        return [to_rgb(cycle[i % len(cycle)]) for i in range(n)]
    if isinstance(palette, str):
        return color_palette(palette, n)
    if isinstance(palette, Mapping):
        return [to_rgb(palette[level]) for level in levels]
    return [to_rgb(palette[i % len(palette)]) for i in range(n)]


# rcParams equivalents of seaborn's "ticks" and "whitegrid" styles (seaborn
# 0.13.2's axes_style). seaborn also sets image.cmap to its own "rocket" map,
# which matplotlib does not ship, so that key is left out.
_STYLE_BASE: dict[str, Any] = {
    "figure.facecolor": "white",
    "axes.labelcolor": ".15",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.color": ".15",
    "ytick.color": ".15",
    "axes.axisbelow": True,
    "grid.linestyle": "-",
    "text.color": ".15",
    "font.family": ["sans-serif"],
    "font.sans-serif": [
        "Arial",
        "DejaVu Sans",
        "Liberation Sans",
        "Bitstream Vera Sans",
        "sans-serif",
    ],
    "lines.solid_capstyle": "round",
    "patch.edgecolor": "w",
    "patch.force_edgecolor": True,
    "xtick.top": False,
    "ytick.right": False,
    "axes.facecolor": "white",
    "grid.color": ".8",
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.spines.right": True,
    "axes.spines.top": True,
}

TICKS_STYLE: dict[str, Any] = {
    **_STYLE_BASE,
    "axes.grid": False,
    "axes.edgecolor": ".15",
    "xtick.bottom": True,
    "ytick.left": True,
}

WHITEGRID_STYLE: dict[str, Any] = {
    **_STYLE_BASE,
    "axes.grid": True,
    "axes.edgecolor": ".8",
    "xtick.bottom": False,
    "ytick.left": False,
}
