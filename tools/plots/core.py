"""Core plotting functions with polars support."""

import datetime
import warnings
from typing import Any

import polars as pl
import seaborn as sns
from matplotlib import MatplotlibDeprecationWarning
from matplotlib.figure import Figure


def smart_plot_data(df: pl.DataFrame, **kwargs: Any) -> dict[str, Any]:
    """Prepare data for plotting.

    Seaborn reads polars frames only through the dataframe-interchange
    protocol, which polars 2 removed, so the frame is converted to pandas.

    Args:
        df: Polars DataFrame to prepare for plotting
        **kwargs: Additional plotting arguments

    Returns:
        Dictionary with 'data' key containing the pandas DataFrame and any
        additional kwargs
    """
    result: dict[str, Any] = {"data": df.to_pandas()}
    result.update(kwargs)
    return result


def enhanced_scatterplot(df: pl.DataFrame, x: str, y: str, **kwargs: Any) -> Any:
    """Create scatter plot with automatic polars support.

    Args:
        df: Polars DataFrame containing the data
        x: Column name for x-axis
        y: Column name for y-axis
        **kwargs: Additional arguments passed to seaborn.scatterplot

    Returns:
        matplotlib Axes object

    Examples:
        >>> df = pl.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        >>> enhanced_scatterplot(df, 'x', 'y')
    """
    plot_data = smart_plot_data(df, x=x, y=y, **kwargs)
    return sns.scatterplot(**plot_data)


def enhanced_lineplot(df: pl.DataFrame, x: str, y: str, **kwargs: Any) -> Any:
    """Create line plot with automatic polars support.

    Args:
        df: Polars DataFrame containing the data
        x: Column name for x-axis
        y: Column name for y-axis
        **kwargs: Additional arguments passed to seaborn.lineplot

    Returns:
        matplotlib Axes object
    """
    plot_data = smart_plot_data(df, x=x, y=y, **kwargs)
    return sns.lineplot(**plot_data)


def enhanced_boxplot(df: pl.DataFrame, x: str, y: str, **kwargs: Any) -> Any:
    """Create box plot with automatic polars support.

    Args:
        df: Polars DataFrame containing the data
        x: Column name for x-axis (categorical)
        y: Column name for y-axis (numeric)
        **kwargs: Additional arguments passed to seaborn.boxplot

    Returns:
        matplotlib Axes object
    """
    plot_data = smart_plot_data(df, x=x, y=y, **kwargs)
    # seaborn 0.13.2 always passes `vert=` to Axes.bxp, deprecated in matplotlib 3.11
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="vert: bool", category=MatplotlibDeprecationWarning
        )
        return sns.boxplot(**plot_data)


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
