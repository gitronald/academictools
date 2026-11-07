"""Display utilities for tables and data visualization.

This module provides functions for displaying polars DataFrames in both text
and HTML formats. Frames are converted to pandas internally for consistent
display formatting (pandas' to_string/to_html and MultiIndex column titles).
"""

from typing import Any

import pandas as pd
import polars as pl
from IPython.display import HTML, display

from tools.utils.jupyter import in_jupyter_notebook
from tools.utils.printing import print_line

pd_options: dict[str, Any] = {
    "display.width": 120,
    "display.max_colwidth": 80,
    "display.max_columns": None,
}


def set_css():
    """Inject CSS to enable text wrapping in Jupyter notebook output cells."""
    display(
        HTML("""
    <style>
        pre {
            white-space: pre-wrap;
        }
    </style>
    """)
    )


def display_tables(
    tables: list[pl.DataFrame],
    titles: list[str] | None = None,
    ncols: int | None = None,
) -> None:
    """Display tables side by side in notebook output or print to console.

    Automatically detects the environment and uses HTML display in Jupyter notebooks
    or text display in console environments.

    Args:
        tables: List of polars DataFrames to display
        titles: Optional list of table captions
        ncols: Number of columns to display (default: len(tables))
    """
    if in_jupyter_notebook():
        display_tables_html(tables, titles, ncols)
    else:
        display_tables_text(tables, titles, ncols)


def display_tables_text(
    tables: list[pl.DataFrame],
    titles: list[str] | None = None,
    ncols: int | None = None,
) -> None:
    """Display tables in text format for console output.

    Converts polars DataFrames to pandas for consistent string representation.

    Args:
        tables: List of polars DataFrames to display
        titles: Optional list of table captions
        ncols: Number of columns to display (default: len(tables))
    """
    ncols = len(tables) if not ncols else ncols
    if titles:
        for title, df in zip(titles, tables):
            print_line(title)
            df = df.to_pandas()
            print(df.to_string(index=True), "\n\n")
    else:
        for df in tables:
            print_line()
            df = df.to_pandas()
            print(df.to_string(index=True), "\n\n")


def display_tables_html(
    tables: list[pl.DataFrame],
    titles: list[str] | None = None,
    ncols: int | None = None,
) -> None:
    """Display tables side by side in HTML format for Jupyter notebooks.

    Creates an HTML table layout with optional titles and automatically converts
    polars DataFrames to pandas for HTML generation.

    Args:
        tables: List of polars DataFrames to display
        titles: Optional list of table captions
        ncols: Number of columns to display (default: len(tables))
    """
    ncols = len(tables) if not ncols else ncols
    html_table = "<table style='border:0px'>{content}</table>"
    html_row = "<tr style='border:0px'>{content}</tr>"
    html_cell = (
        "<td style='width:{width}%;vertical-align:top;"
        "horizontal-align:left;border:0px'>{{content}}</td>"
    )
    html_cell = html_cell.format(width=100 / ncols)

    cells = []
    if titles:
        for title, df in zip(titles, tables):
            df = df.to_pandas()
            df.columns = pd.MultiIndex.from_product([[title], df.columns])
            df.columns.name = title
            cells.append(html_cell.format(content=df.to_html()))
    else:
        for df in tables:
            df = df.to_pandas()
            cells.append(html_cell.format(content=df.to_html()))

    # Add Padding
    cells += (ncols - (len(tables) % ncols)) * [html_cell.format(content="")]

    # Format and display meta table
    nrows = range(0, len(cells), ncols)
    rows = [html_row.format(content="".join(cells[i : i + ncols])) for i in nrows]
    table = html_table.format(content="".join(rows))
    display(HTML(table))
