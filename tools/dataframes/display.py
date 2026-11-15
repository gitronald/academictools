"""Display utilities for tables and data visualization.

This module provides functions for displaying polars DataFrames in both text
and HTML formats, using polars' own rendering.
"""

from html import escape

import polars as pl
from IPython.core.formatters import HTMLFormatter
from IPython.display import HTML, display

from tools.utils.jupyter import in_jupyter_notebook
from tools.utils.printing import print_line


def _table_config() -> pl.Config:
    """Polars display settings for full tables: all rows and columns, wide output."""
    return pl.Config(
        tbl_rows=-1,
        tbl_cols=-1,
        tbl_width_chars=120,
        fmt_str_lengths=80,
    )


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

    Prints each polars DataFrame in full (every row and column), under a rule
    carrying its title when titles are given.

    Args:
        tables: List of polars DataFrames to display
        titles: Optional list of table captions
        ncols: Accepted for parity with ``display_tables_html``; text output
            always prints one table after another
    """
    with _table_config():
        if titles:
            for title, df in zip(titles, tables):
                print_line(title)
                print(df, "\n\n")
        else:
            for df in tables:
                print_line()
                print(df, "\n\n")


def _html_table(df: pl.DataFrame, title: str | None = None) -> str:
    """Render one polars frame as HTML, with an optional bold title above it."""
    with _table_config():
        table = HTMLFormatter()(df)
    if title is None:
        return table
    heading = f"<div style='font-weight:bold;text-align:left'>{escape(title)}</div>"
    return heading + table


def display_tables_html(
    tables: list[pl.DataFrame],
    titles: list[str] | None = None,
    ncols: int | None = None,
) -> None:
    """Display tables side by side in HTML format for Jupyter notebooks.

    Creates an HTML table layout with optional titles, rendering each polars
    DataFrame with polars' own HTML output.

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

    if titles:
        cells = [
            html_cell.format(content=_html_table(df, title))
            for title, df in zip(titles, tables)
        ]
    else:
        cells = [html_cell.format(content=_html_table(df)) for df in tables]

    # Add Padding
    cells += (ncols - (len(tables) % ncols)) * [html_cell.format(content="")]

    # Format and display meta table
    nrows = range(0, len(cells), ncols)
    rows = [html_row.format(content="".join(cells[i : i + ncols])) for i in nrows]
    table = html_table.format(content="".join(rows))
    display(HTML(table))
