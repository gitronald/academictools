"""Color palettes and style presets on plain matplotlib."""

import colorsys
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import matplotlib as mpl
import numpy as np
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap, to_rgb

from tools.plots._husl import husl_to_rgb, rgb_to_husl

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
    """Sample ``n_colors`` RGB colors from a named palette.

    Samples the way seaborn's ``color_palette(name, n)`` does. A qualitative
    matplotlib colormap (``tab10``, ``Set2``, ...) gives its first ``n_colors``
    colors exactly, repeating from the start when asked for more than it
    defines; a continuous map gives evenly spaced samples that skip both ends,
    for contrast at the extremes. ``"dark:<color>"`` and ``"light:<color>"``
    give ``dark_palette`` and ``light_palette`` of that color (``_r`` after the
    color reverses them).

    Args:
        name: A registered matplotlib colormap (``_r`` reverses it), or
            ``"dark:<color>"`` / ``"light:<color>"``
        n_colors: Number of colors to return

    Returns:
        List of (r, g, b) tuples with channels in [0, 1]
    """
    for prefix, ramp in (("dark:", dark_palette), ("light:", light_palette)):
        if name.startswith(prefix):
            color = name[len(prefix) :]
            reverse = color.endswith("_r")
            return ramp(color[:-2] if reverse else color, n_colors, reverse=reverse)
    cmap = mpl.colormaps[name]
    if name in QUALITATIVE_SIZES:
        bins = np.linspace(0, 1, QUALITATIVE_SIZES[name])[:n_colors]
    else:
        bins = np.linspace(0, 1, int(n_colors) + 2)[1:-1]
    colors = [(float(r), float(g), float(b)) for r, g, b in cmap(bins)[:, :3]]
    return [colors[i % len(colors)] for i in range(int(n_colors))]


def blend_palette(colors: Sequence[Any], n_colors: int = 6) -> list[RGB]:
    """``n_colors`` evenly spaced colors on a linear RGB blend through ``colors``."""
    cmap = LinearSegmentedColormap.from_list("blend", [to_rgb(c) for c in colors])
    samples = cmap(np.linspace(0, 1, int(n_colors)))[:, :3]
    return [(float(r), float(g), float(b)) for r, g, b in samples]


def _husl_ramp(color: Any, n_colors: int, lightness: float, reverse: bool) -> list[RGB]:
    """Blend between ``color`` and a gray of its hue at HUSL ``lightness``."""
    rgb = to_rgb(color)
    hue, sat, _ = rgb_to_husl(*rgb)
    gray_rgb = np.clip(husl_to_rgb(hue, 0.15 * sat, lightness), 0, 1)
    gray = (float(gray_rgb[0]), float(gray_rgb[1]), float(gray_rgb[2]))
    ends = [rgb, gray] if reverse else [gray, rgb]
    return blend_palette(ends, n_colors)


def dark_palette(color: Any, n_colors: int = 6, reverse: bool = False) -> list[RGB]:
    """Sequential palette from a near-black of ``color``'s hue up to ``color``.

    seaborn's ``dark_palette``: the dark end is the color's HUSL hue at 15%
    of its saturation and lightness 15.
    """
    return _husl_ramp(color, n_colors, 15, reverse)


def light_palette(color: Any, n_colors: int = 6, reverse: bool = False) -> list[RGB]:
    """Sequential palette from a near-white of ``color``'s hue up to ``color``.

    seaborn's ``light_palette``: the light end is the color's HUSL hue at 15%
    of its saturation and lightness 95.
    """
    return _husl_ramp(color, n_colors, 95, reverse)


def desaturate(color: Any, prop: float) -> RGB:
    """``color`` with its HLS saturation multiplied by ``prop`` (0 to 1)."""
    if not 0 <= prop <= 1:
        raise ValueError(f"prop must be between 0 and 1, got {prop}")
    rgb = to_rgb(color)
    if prop == 1:
        return rgb
    h, lightness, s = colorsys.rgb_to_hls(*rgb)
    return colorsys.hls_to_rgb(h, lightness, s * prop)


def resolve_colors(levels: Sequence[Any], palette: Palette = None) -> list[RGB]:
    """Assign one color per level from a palette specification.

    Args:
        levels: Ordered group levels needing a color each
        palette: None for matplotlib's color cycle, a palette name (see
            ``color_palette``), a list of colors (cycled when shorter than
            ``levels``), or a dict mapping each level to a color

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

# seaborn's ten-color "deep" palette, the default color cycle of set_theme
DEEP: list[str] = [
    "#4c72b0",
    "#dd8452",
    "#55a868",
    "#c44e52",
    "#8172b3",
    "#937860",
    "#da8bc3",
    "#8c8c8c",
    "#ccb974",
    "#64b5cd",
]

# seaborn's "notebook" plotting context: line widths, tick sizes, and font
# sizes. The font sizes are a base that a later rcParams update (such as
# settings.RC_PARAMS) can override.
NOTEBOOK_CONTEXT: dict[str, Any] = {
    "font.size": 12.0,
    "axes.labelsize": 12.0,
    "axes.titlesize": 12.0,
    "xtick.labelsize": 11.0,
    "ytick.labelsize": 11.0,
    "legend.fontsize": 11.0,
    "legend.title_fontsize": 12.0,
    "axes.linewidth": 1.25,
    "grid.linewidth": 1.0,
    "lines.linewidth": 1.5,
    "lines.markersize": 6.0,
    "patch.linewidth": 1.0,
    "xtick.major.width": 1.25,
    "ytick.major.width": 1.25,
    "xtick.minor.width": 1.0,
    "ytick.minor.width": 1.0,
    "xtick.major.size": 6.0,
    "ytick.major.size": 6.0,
    "xtick.minor.size": 4.0,
    "ytick.minor.size": 4.0,
}

_STYLES: dict[str, dict[str, Any]] = {
    "ticks": TICKS_STYLE,
    "whitegrid": WHITEGRID_STYLE,
}


def set_theme(style: Literal["ticks", "whitegrid"] = "ticks") -> None:
    """Set matplotlib's rcParams as ``seaborn.set_theme(style=style)`` does.

    Applies the axes style, the "notebook" context, and the "deep" color
    cycle. seaborn's own "rocket" default colormap is not set.
    """
    if style not in _STYLES:
        raise ValueError(f"style must be 'ticks' or 'whitegrid', got {style!r}")
    mpl.rcParams.update(_STYLES[style])
    mpl.rcParams.update(NOTEBOOK_CONTEXT)
    mpl.rcParams["axes.prop_cycle"] = cycler(color=DEEP)
