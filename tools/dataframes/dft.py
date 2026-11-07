"""Descriptive frequency table functions with polars-first design.

This module provides frequency table functionality using polars as the engine
for better performance, especially with large datasets.
"""

import math

import polars as pl

# Default log-scale binning spec. ``breaks`` are the internal cut points polars
# uses; with the implicit -inf/+inf bounds they yield ``len(breaks) + 1`` bins,
# one per label. Right-closed by default, so each break value falls into the
# lower bin (e.g. 10 -> "2-10", 100 -> "11-100").
_BIN_BREAKS: list[float] = [0, 1, 10, 100, 1000]
_BIN_LABELS: list[str] = ["0", "1", "2-10", "11-100", "101-1000", "1000+"]


def _is_null(value: int | float | None) -> bool:
    """Return True for None or NaN floats."""
    return value is None or (isinstance(value, float) and math.isnan(value))


def neat_n(n: int | float | None, fmt: str = ",.0f") -> str:
    """Format numeric values with thousand separators.

    Args:
        n: Numeric value to format
        fmt: Format string for numeric display

    Returns:
        Formatted string or '-' for null values
    """
    return "-" if _is_null(n) else f"{n:{fmt}}"


def neat_p(p: float | None, fmt: str = ".1%") -> str:
    """Format proportion values as percentages.

    Args:
        p: Proportion value (0-1) to format
        fmt: Format string for percentage display

    Returns:
        Formatted percentage string or '-' for null values
    """
    return "-" if _is_null(p) else f"{p:{fmt}}"


def dft(
    col: pl.Series,
    neat: bool = True,
    dropna: bool = True,
    sort_index: bool = False,
    binned: bool = False,
    fmt_n: str = ",.0f",
    fmt_p: str = ".1%",
) -> pl.DataFrame:
    """Generate descriptive frequency table.

    Creates a frequency table showing counts and proportions for each unique
    value.

    Args:
        col: Polars Series to analyze
        neat: Whether to format output with human-readable strings
        dropna: Whether to exclude null values from counts
        sort_index: Whether to sort output by value
        binned: Whether to bin numeric values logarithmically
        fmt_n: Format string for count values when neat=True
        fmt_p: Format string for proportion values when neat=True

    Returns:
        Polars DataFrame with frequency analysis

    Examples:
        >>> dft(pl.Series([1, 2, 2, 3, 3, 3]))
        >>> # Returns polars DataFrame with counts and percentages
    """
    if not isinstance(col, pl.Series):
        raise ValueError("input must be a polars Series")

    series = binned_col(col) if binned else col
    if col.name in {"n", "p"}:
        raise ValueError("input series name cannot be 'n' or 'p'")
    output_name = col.name
    series = series.rename("__dft_value__")

    # Generate value counts with proportion calculation
    if dropna:
        output = series.drop_nulls().value_counts()
    else:
        output = series.value_counts()

    # value_counts names the value column after the series (may be "")
    value_col = output.columns[0]
    count_col = "__dft_count__"
    proportion_col = "__dft_proportion__"
    output = output.rename({"count": count_col})
    output = output.with_columns(
        (pl.col(count_col) / pl.col(count_col).sum()).alias(proportion_col)
    )

    # Sort by value if requested, otherwise by count descending (like pandas)
    if sort_index:
        output = output.sort(value_col)
    else:
        output = output.sort(count_col, descending=True)

    if neat:
        output = output.with_columns(
            [
                pl.col(count_col)
                .map_elements(lambda x: f"{x:{fmt_n}}", return_dtype=pl.String)
                .alias("n"),
                pl.col(proportion_col)
                .map_elements(lambda x: f"{x:{fmt_p}}", return_dtype=pl.String)
                .alias("p"),
            ]
        ).select([value_col, "n", "p"])
    else:
        output = output.with_columns(
            [pl.col(count_col).alias("n"), pl.col(proportion_col).alias("p")]
        ).select([value_col, "n", "p"])

    return output.rename({value_col: output_name})


def binned_col(
    col: pl.Series,
    breaks: list[float] | None = None,
    labels: list[str] | None = None,
) -> pl.Series:
    """Bin a numeric column into logarithmic-scale categories.

    Args:
        col: Polars Series of numeric values to bin
        breaks: Internal cut points (defaults to the module log-scale breaks).
            The outer bins are open-ended, so the result has ``len(breaks) + 1``
            bins, one per label. Bins are right-closed: a break value falls into
            the lower bin.
        labels: Labels for each bin (must be ``len(breaks) + 1`` long).

    Returns:
        Ordered Enum Series of bin labels, so sorting follows bin order rather
        than lexical order.
    """
    breaks = _BIN_BREAKS if breaks is None else breaks
    labels = _BIN_LABELS if labels is None else labels
    return col.bin_intervals(breaks, labels=labels, right_closed=True)


def get_binned_dft(col: pl.Series) -> pl.DataFrame:
    """Generate a frequency table for a numeric column with log-scale binning.

    Args:
        col: Polars Series of numeric values to bin and tabulate

    Returns:
        Polars DataFrame with frequency counts and percentages per bin
    """
    return dft(binned_col(col), sort_index=True)


def presentation_table(
    df: pl.DataFrame, ncols: list[str] | None = None, pcols: list[str] | None = None
) -> pl.DataFrame:
    """Format DataFrame columns with human-readable number and percentage strings.

    Formats columns of an assembled summary table: counts with thousands
    separators, proportions as percentages, and ``mean_*`` columns to three
    decimals. Null values render as '-' for count and percentage columns.

    Args:
        df: Polars DataFrame to format
        ncols: Specific columns to format as counts (default: auto-detect n/n_ prefix)
        pcols: Specific columns to format as percentages (default: auto-detect
            p/p_ prefix)

    Returns:
        Polars DataFrame with formatted string columns
    """
    ncols = ncols or []
    pcols = pcols or []
    exprs: list[pl.Expr] = []
    for c in df.columns:
        is_ncol = c in ncols if ncols else (c == "n" or c.startswith("n_"))
        is_pcol = c in pcols if pcols else (c == "p" or c.startswith("p_"))
        if is_ncol:
            exprs.append(
                pl.col(c)
                .map_elements(neat_n, return_dtype=pl.String, skip_nulls=False)
                .alias(c)
            )
        elif is_pcol:
            exprs.append(
                pl.col(c)
                .map_elements(neat_p, return_dtype=pl.String, skip_nulls=False)
                .alias(c)
            )
        elif c.startswith("mean_"):
            exprs.append(
                pl.col(c)
                .map_elements(lambda n: f"{n:.3f}", return_dtype=pl.String)
                .alias(c)
            )

    return df.with_columns(exprs) if exprs else df
