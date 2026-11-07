"""DataFrame inspection and manipulation utilities."""

import itertools
import json
import time
from typing import Any

import polars as pl

from tools.utils.timing import print_time_elapsed


def dfshape(df: pl.DataFrame | pl.Series) -> str:
    """Format DataFrame or Series shape as a readable string.

    Args:
        df: Polars DataFrame/Series to get shape information from

    Returns:
        Formatted shape string like "(1000, 5)" or "(1000, 1)" for Series

    Raises:
        ValueError: If input is not a polars DataFrame or Series

    Examples:
        >>> df = pl.DataFrame({'a': [1, 2], 'b': [3, 4]})
        >>> dfshape(df)
        '(2, 2)'
        >>> dfshape(df['a'])
        '(2, 1)'
    """
    if isinstance(df, pl.Series):
        return f"({df.shape[0]:,}, 1)"
    elif isinstance(df, pl.DataFrame):
        return f"({df.shape[0]:,}, {df.shape[1]:,})"
    else:
        raise ValueError("input must be a polars DataFrame or Series")


def get_nonzero_min(df: pl.DataFrame, col: str) -> float | int:
    """Get minimum positive value from a column.

    Args:
        df: Polars DataFrame containing the data
        col: Column name to analyze

    Returns:
        Minimum positive value

    Examples:
        >>> df = pl.DataFrame({'x': [0, 1, 2, 3]})
        >>> get_nonzero_min(df, 'x')
        1
    """
    return df.filter(pl.col(col) > 0).select(pl.col(col).min()).item()


def unlist(list_of_lists: list[list[Any]]) -> list[Any]:
    """Flatten a list of lists into a single list.

    Args:
        list_of_lists: Iterable of iterables to flatten

    Returns:
        Flattened list containing all elements
    """
    return list(itertools.chain.from_iterable(list_of_lists))


def get_dataframe_details(df: pl.DataFrame) -> None:
    """Display dtype, null count, and unique count for each column.

    Args:
        df: Polars DataFrame to inspect
    """
    print("checking dataframe details")
    start_time = time.time()
    null_counts = df.null_count()
    df_details = pl.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(dt) for dt in df.dtypes],
            "null": [null_counts[c][0] for c in df.columns],
            "nunique": [df[c].n_unique() for c in df.columns],
        }
    )
    with pl.Config(tbl_rows=-1):
        print(df_details)
    print_time_elapsed(start_time)


def print_row(row: dict[str, Any] | pl.DataFrame, indent: int = 2) -> None:
    """Print a single row as a formatted JSON object.

    Args:
        row: A row as a dict (e.g. from ``.row(named=True)``) or a single-row
            polars DataFrame
        indent: JSON indentation level
    """
    if isinstance(row, pl.DataFrame):
        row = row.row(0, named=True)
    print(json.dumps(row, indent=indent, default=str))


def get_closest_row(df: pl.DataFrame, col: str, target: float) -> dict[str, Any]:
    """Find the row with the value closest to a target in a column.

    Args:
        df: Polars DataFrame to search
        col: Column name to compare against target
        target: Target value to find the closest match for

    Returns:
        The closest matching row as a dict (column name -> value)
    """
    idx = (df[col].cast(pl.Float64) - target).abs().arg_min()
    return df.row(idx, named=True)
