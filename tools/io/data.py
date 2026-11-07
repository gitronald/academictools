"""Data input/output functions with comprehensive type hints and documentation.

This module provides unified data I/O operations using polars for better
performance and memory efficiency.
"""

import bz2
import io
import os
import time
import traceback
from collections.abc import Generator
from functools import reduce
from pathlib import Path
from typing import IO, Any, Literal, cast

import polars as pl
import ujson
from IPython.display import display
from matplotlib.figure import Figure

from tools.dataframes.inspection import dfshape

# Path-like and join-strategy aliases used throughout IO operations.
FilePath = str | Path
JoinStrategy = Literal["inner", "left", "right", "full", "semi", "anti", "cross"]


def _print_io_time(
    fp: FilePath,
    data: pl.DataFrame | pl.Series | list[Any] | str,
    start: float,
    direction: str = "loaded",
    symbol: str = "\U000021d1",
    verbose: bool = True,
    newline: bool = False,
    color: str = "\033[38;2;182;251;182m",
) -> None:
    """Print formatted I/O time with file path, data dimensions, and elapsed seconds.

    Args:
        fp: File path to display
        data: Data object to get dimensions from
        start: Start time from time.time()
        direction: I/O direction label (e.g. 'loaded', 'saved')
        symbol: Unicode symbol prefix for the message
        verbose: Whether to print the message
        newline: Whether to add a trailing newline
        color: ANSI color code for the symbol and direction
    """
    if verbose:
        data_dim = (
            dfshape(data) if isinstance(data, (pl.DataFrame, pl.Series)) else len(data)
        )
        prefix = f"{color}{symbol} {direction}:\033[0m"
        msg = f"{prefix} {fp} - {data_dim} | {time.time() - start:.3f} seconds"
        print(f"{msg}\n" if newline else msg)


def print_load_time(
    fp: FilePath,
    data: pl.DataFrame | pl.Series | list[Any] | str,
    start: float,
    verbose: bool = True,
    newline: bool = False,
    color: str = "\033[38;2;182;251;182m",
) -> None:
    """Print formatted load time with file path and data dimensions.

    Args:
        fp: File path that was loaded
        data: Loaded data object
        start: Start time from time.time()
        verbose: Whether to print the message
        newline: Whether to add a trailing newline
        color: ANSI color code for the message prefix
    """
    _print_io_time(fp, data, start, "loaded", "\U000021d1", verbose, newline, color)


def print_save_time(
    fp: FilePath,
    data: pl.DataFrame | pl.Series | list[Any] | str,
    start: float,
    verbose: bool = True,
    newline: bool = False,
    color: str = "\033[38;2;182;251;182m",
) -> None:
    """Print formatted save time with file path and data dimensions.

    Args:
        fp: File path that was saved to
        data: Saved data object
        start: Start time from time.time()
        verbose: Whether to print the message
        newline: Whether to add a trailing newline
        color: ANSI color code for the message prefix
    """
    _print_io_time(fp, data, start, "saved", "\U000021d3", verbose, newline, color)


# ------------------------------------------------------------------------------
# Data Load/Save


def _check_overwrite(fp: FilePath, overwrite: bool) -> None:
    """Raise FileExistsError if file exists and overwrite is False.

    Args:
        fp: File path to check
        overwrite: Whether overwriting is allowed
    """
    if os.path.exists(fp) and not overwrite:
        raise FileExistsError(f"file exists, use overwrite=True: {fp}")


def _detect_separator(fp: FilePath) -> str:
    """Detect CSV/TSV separator from file extension.

    Args:
        fp: File path with .csv or .tsv extension (optionally .bz2 compressed)

    Returns:
        Separator character (',' or '\\t')

    Raises:
        ValueError: If extension is not recognized
    """
    fp_str = str(fp)
    if fp_str.endswith(".tsv") or fp_str.endswith(".tsv.bz2"):
        return "\t"
    elif fp_str.endswith(".csv") or fp_str.endswith(".csv.bz2"):
        return ","
    else:
        raise ValueError("file must be {.tsv, .csv, .tsv.bz2, .csv.bz2}")


def load_dataframe(fp: FilePath) -> pl.DataFrame:
    """Universal data loader supporting multiple file formats.

    Automatically detects file format based on extension and loads using
    the appropriate parser. Supports CSV, TSV, Feather, and Parquet formats.

    Args:
        fp: Path to the data file

    Returns:
        Loaded polars DataFrame

    Raises:
        ValueError: If no suitable loader found for file extension

    Examples:
        >>> df = load_dataframe('data.csv')
        >>> df = load_dataframe('data.parquet')
        >>> df = load_dataframe('data.feather')
    """
    loader_dict = {
        ".csv": load_xsv,
        ".csv.bz2": load_xsv,
        ".tsv": load_xsv,
        ".tsv.bz2": load_xsv,
        ".feather": load_feather,
        ".parquet": load_parquet,
    }
    loader = [loader for ext, loader in loader_dict.items() if str(fp).endswith(ext)]
    if len(loader) == 1:
        loader_func = loader[0]
        return loader_func(fp)
    else:
        raise ValueError(f"no loader found for {fp}")


# ------------------------------------------------------------------------------
# Data Load/Save - TSV/CSV


def load_xsv(
    fp: FilePath,
    kwargs: dict[str, Any] | None = None,
    display_head: int = 0,
    verbose: bool = True,
    newline: bool = False,
) -> pl.DataFrame:
    """Load CSV or TSV files with automatic separator detection.

    Supports both regular and bz2-compressed files. Automatically detects
    separator based on file extension (.csv uses comma, .tsv uses tab).

    Args:
        fp: Path to CSV/TSV file (supports .csv, .tsv, .csv.bz2, .tsv.bz2)
        kwargs: Additional keyword arguments passed to the reader
        display_head: Number of rows to display after loading (0 = none)
        verbose: Whether to print loading information
        newline: Whether to add newline to verbose output

    Returns:
        Loaded polars DataFrame

    Raises:
        ValueError: If file extension is not supported

    Examples:
        >>> df = load_xsv('data.csv')
        >>> df = load_xsv('data.tsv')
        >>> df = load_xsv('data.csv.bz2', display_head=5)  # show first 5 rows
    """
    start = time.time()
    kwargs = kwargs or {}

    sep = _detect_separator(fp)

    if not (os.path.exists(fp) or str(fp).startswith("https://")):
        raise FileNotFoundError(f"File not found: {fp}")

    if str(fp).startswith("https://"):
        df = pl.read_csv(fp, separator=sep, **kwargs)
    elif str(fp).endswith(".bz2"):
        with open(fp, "rb") as infile:
            contents = infile.read()
        if contents[:3] == b"BZh":
            contents = bz2.decompress(contents)
        df = pl.read_csv(io.BytesIO(contents), separator=sep, **kwargs)
    else:
        df = pl.read_csv(fp, separator=sep, **kwargs)

    print_load_time(fp, df, start, verbose, newline=newline)
    if display_head > 0:
        display(df.head(display_head))
    return df


def save_xsv(
    df: pl.DataFrame,
    fp: FilePath,
    kwargs: dict[str, Any] | None = None,
    overwrite: bool = False,
    verbose: bool = True,
    newline: bool = False,
) -> None:
    """Save DataFrame as CSV or TSV file.

    Automatically determines separator based on file extension (.csv uses comma,
    .tsv uses tab).

    Args:
        df: Polars DataFrame to save
        fp: Output file path (.csv or .tsv extension)
        kwargs: Additional keyword arguments passed to the writer
        overwrite: Whether to overwrite existing files
        verbose: Whether to print save information
        newline: Whether to add newline to verbose output

    Raises:
        ValueError: If file path is empty, extension not supported, or DataFrame
            conversion fails

    Examples:
        >>> save_xsv(df, 'output.csv')  # CSV format
        >>> save_xsv(df, 'output.tsv', overwrite=True)  # TSV format
    """
    start = time.time()
    kwargs = kwargs or {}
    if not fp:
        raise ValueError("file path must be provided")

    sep = _detect_separator(fp)
    _check_overwrite(fp, overwrite)

    if str(fp).endswith(".bz2"):
        csv_text = df.write_csv(separator=sep, **kwargs)
        with bz2.open(fp, "wt") as outfile:
            outfile.write(csv_text)
    else:
        df.write_csv(fp, separator=sep, **kwargs)
    print_save_time(fp, df, start, verbose, newline=newline)


# ------------------------------------------------------------------------------
# Data Load/Save - Text


def save_text(
    text: str, fp: FilePath, overwrite: bool = False, verbose: bool = True
) -> None:
    """Save text content to a file.

    Args:
        text: Text content to save
        fp: Output file path
        overwrite: Whether to overwrite existing files
        verbose: Whether to print save information

    Examples:
        >>> save_text('Hello, world!', 'output.txt')
        >>> save_text(content, 'data.txt', overwrite=True)
    """
    start = time.time()
    _check_overwrite(fp, overwrite)

    with open(fp, "w") as outfile:
        outfile.write(text)
    print_save_time(fp, text, start, verbose)


def load_text(fp: FilePath, verbose: bool = True) -> str:
    """Load text content from a file.

    Args:
        fp: Path to text file to load
        verbose: Whether to print load information

    Returns:
        File contents as string, or None if file not found

    Examples:
        >>> content = load_text('input.txt')
        >>> text = load_text('data.txt', verbose=False)
    """
    start = time.time()
    if not os.path.exists(fp):
        raise FileNotFoundError(f"file not found: {fp}")

    with open(fp) as infile:
        text = infile.read()
    print_load_time(fp, text, start, verbose)
    return text


# ------------------------------------------------------------------------------
# Data Load/Save - Feather


def load_feather(
    fp: FilePath, display_head: int = 0, verbose: bool = True, newline: bool = False
) -> pl.DataFrame:
    """Load Feather (Arrow IPC) format files.

    Args:
        fp: Path to Feather file
        display_head: Number of rows to display after loading (0 = none)
        verbose: Whether to print load information
        newline: Whether to add newline to verbose output

    Returns:
        Loaded polars DataFrame

    Examples:
        >>> df = load_feather('data.feather')
    """
    start = time.time()
    df = pl.read_ipc(fp)

    print_load_time(fp, df, start, verbose, newline=newline)
    if display_head > 0:
        display(df.head(display_head))
    return df


def save_feather(
    df: pl.DataFrame,
    fp: FilePath,
    overwrite: bool = False,
    verbose: bool = True,
    newline: bool = False,
) -> None:
    """Save DataFrame in Feather (Arrow IPC) format.

    Args:
        df: Polars DataFrame to save
        fp: Output file path (.feather extension)
        overwrite: Whether to overwrite existing files
        verbose: Whether to print save information
        newline: Whether to add newline to verbose output

    Raises:
        ValueError: If DataFrame conversion fails

    Examples:
        >>> save_feather(df, 'output.feather')
        >>> save_feather(df, 'data.feather', overwrite=True)
    """
    start = time.time()

    _check_overwrite(fp, overwrite)

    df.write_ipc(fp)
    print_save_time(fp, df, start, verbose, newline=newline)


# ------------------------------------------------------------------------------
# Data Load/Save - Parquet


def load_parquet(
    fp: FilePath, display_head: int = 0, verbose: bool = True, newline: bool = False
) -> pl.DataFrame:
    """Load Parquet format files.

    Args:
        fp: Path to Parquet file
        display_head: Number of rows to display after loading (0 = none)
        verbose: Whether to print load information
        newline: Whether to add newline to verbose output

    Returns:
        Loaded polars DataFrame

    Examples:
        >>> df = load_parquet('data.parquet')
    """
    start = time.time()
    df = pl.read_parquet(fp)

    print_load_time(fp, df, start, verbose, newline=newline)
    if display_head > 0:
        display(df.head(display_head))
    return df


def save_parquet(
    df: pl.DataFrame,
    fp: FilePath,
    overwrite: bool = False,
    verbose: bool = True,
    newline: bool = False,
    **kwargs: Any,
) -> None:
    """Save DataFrame in Parquet format.

    Args:
        df: Polars DataFrame to save
        fp: Output file path (.parquet extension)
        overwrite: Whether to overwrite existing files
        verbose: Whether to print save information
        newline: Whether to add newline to verbose output
        **kwargs: Additional keyword arguments passed to the writer

    Raises:
        ValueError: If DataFrame conversion fails

    Examples:
        >>> save_parquet(df, 'output.parquet')
        >>> save_parquet(df, 'data.parquet', overwrite=True)
    """
    start = time.time()

    _check_overwrite(fp, overwrite)

    df.write_parquet(fp, **kwargs)
    print_save_time(fp, df, start, verbose, newline=newline)


# ------------------------------------------------------------------------------
# Data Load/Save - JSON


def jsonl_file_handler(fp: str, mode: str = "r") -> IO[str]:
    """Open a JSON or compressed JSON file with the appropriate handler.

    Args:
        fp: File path ending in .json or .json.bz2
        mode: File open mode ('r' or 'w')

    Returns:
        Text-mode file handle for reading or writing

    Raises:
        ValueError: If file extension is not .json or .json.bz2
    """
    if fp.endswith(".json"):
        return open(fp, mode)
    elif fp.endswith(".json.bz2"):
        # text mode so callers read/write str (consistent with the .json branch)
        text_mode = {"r": "rt", "w": "wt", "x": "xt", "a": "at"}.get(mode)
        if text_mode is None:
            raise ValueError(f"unsupported file mode: {mode}")
        return cast(IO[str], bz2.open(fp, text_mode))
    else:
        raise ValueError(f"file must be .json or .json.bz2 - {fp}")


def load_jsonl(
    fp: str,
    nrows: int = 0,
    as_df: bool = False,
    display_head: int = 0,
    verbose: bool = True,
    return_errors: bool = False,
    newline: bool = False,
) -> (
    tuple[pl.DataFrame | list[dict[str, Any]], pl.DataFrame | list[dict[str, Any]]]
    | pl.DataFrame
    | list[dict[str, Any]]
):
    """Load JSON lines file with automatic bz2 decompression.

    Args:
        fp: Path to .json or .json.bz2 file
        nrows: Maximum number of rows to load (0 for all)
        as_df: Whether to return DataFrames instead of lists
        display_head: Number of rows to display after loading (0 = none)
        verbose: Whether to print loading information
        return_errors: Whether to return parse errors alongside data
        newline: Whether to add a trailing newline to verbose output

    Returns:
        Data as list of dicts (or DataFrame if as_df=True). If return_errors=True,
        returns tuple of (data, errors).
    """

    start = time.time()
    rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    with jsonl_file_handler(fp, mode="r") as infile:
        for idx, line in enumerate(infile):
            if nrows > 0 and idx == nrows:
                break
            try:
                row = ujson.loads(line)
                rows.append(row)
            except ValueError, TypeError, ujson.JSONDecodeError:
                err = {"idx": idx, "line": line, "error": traceback.format_exc()}
                error_rows.append(err)
        if verbose:
            print_load_time(fp, rows, start, verbose, newline=newline)

    data: pl.DataFrame | list[dict[str, Any]] = rows
    errors: pl.DataFrame | list[dict[str, Any]] = error_rows
    if as_df:
        data, errors = pl.DataFrame(rows), pl.DataFrame(error_rows)
        if display_head > 0:
            display(data.head(display_head))

    return (data, errors) if return_errors else data


def save_jsonl(
    data: list[dict[str, Any]],
    fp: str,
    mode: str = "w",
    overwrite: bool = False,
    verbose: bool = True,
    newline: bool = False,
) -> None:
    """Save data as a JSON lines file.

    Args:
        data: Iterable of dicts to write as JSON lines
        fp: Output file path (.json or .json.bz2)
        mode: File write mode
        overwrite: Whether to overwrite existing files
        verbose: Whether to print save information
        newline: Whether to add a trailing newline to verbose output
    """
    start = time.time()
    _check_overwrite(fp, overwrite)

    with jsonl_file_handler(fp, mode) as outfile:
        for row in data:
            outfile.write(f"{ujson.dumps(row)}\n")
    if verbose:
        print_save_time(fp, data, start, verbose, newline=newline)


def load_jsonl_chunked(
    fp: str, chunk_size: int = 100, verbose: bool = False
) -> Generator[list[dict[str, Any]]]:
    """Load a JSON lines file in chunks via a generator.

    Args:
        fp: Path to .json or .json.bz2 file
        chunk_size: Number of records per yielded chunk
        verbose: Whether to print parse errors

    Yields:
        Lists of parsed JSON objects, each up to chunk_size length
    """
    chunk: list[dict[str, Any]] = []
    with jsonl_file_handler(fp, "r") as file:
        for line in file:
            try:
                json_line = ujson.loads(line)
            except Exception as e:
                if verbose:
                    print(f"Error parsing JSON from {fp}:\n{e}")
                continue
            else:
                chunk.append(json_line)
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


# ------------------------------------------------------------------------------
# Load and Combine


def load_and_merge(
    df: pl.DataFrame,
    fps: list[FilePath],
    how: JoinStrategy = "left",
    on: str = "id",
    verbose: bool = True,
) -> pl.DataFrame:
    """Load CSV/TSV files and merge them into an existing DataFrame.

    Args:
        df: Base DataFrame to merge into
        fps: List of file paths to load and merge
        how: Join type ('left', 'inner', 'outer', 'cross')
        on: Column name(s) to join on
        verbose: Whether to print load time information

    Returns:
        Merged polars DataFrame
    """
    start = time.time()
    dfs = [load_xsv(fp) for fp in fps]
    merged = merge_df_list([df, *dfs], how=how, on=on)
    print_load_time(fps[0], df, start, verbose)
    return merged


def load_and_concat(
    fps: list[FilePath], dtypes: dict[str, Any] | None = None, verbose: bool = True
) -> pl.DataFrame:
    """Load CSV/TSV files and concatenate them vertically.

    Args:
        fps: List of file paths to load and concatenate
        dtypes: Optional dtype overrides passed to the CSV reader
        verbose: Whether to print load time information

    Returns:
        Concatenated polars DataFrame
    """
    start = time.time()
    dtypes = dtypes or {}
    dfs = [load_xsv(fp, kwargs={"schema_overrides": dtypes}) for fp in fps]
    df = pl.concat(dfs)
    print_load_time(fps[0], df, start, verbose)
    return df


def merge_df_list(
    dfs: list[pl.DataFrame], how: JoinStrategy, on: str | None = None
) -> pl.DataFrame:
    """Sequentially merge a list of DataFrames using reduce.

    Args:
        dfs: List of polars DataFrames to merge
        how: Join type ('inner', 'left', 'right', 'full', 'semi', 'anti', 'cross')
        on: Column name to join on, omitted for cross joins

    Returns:
        Merged polars DataFrame
    """
    if how == "cross":
        return reduce(lambda left, right: left.join(right, how=how), dfs)
    return reduce(lambda left, right: left.join(right, on=on, how=how), dfs)


# ------------------------------------------------------------------------------
# Figures


def save_fig(
    fig: Figure,
    fp: str,
    overwrite: bool = False,
    png: bool = True,
    tiff: bool = False,
    pdf: bool = False,
) -> None:
    """Save a figure as .png, .tiff, and/or .pdf.

    Args:
        fig: Matplotlib figure object to save
        fp: File path to save to (extension determines base format)
        overwrite: Whether to overwrite existing files
        png: Whether to save as .png
        tiff: Whether to save as .tiff
        pdf: Whether to save as .pdf

    Raises:
        ValueError: If file extension is not .png, .tiff, or .pdf
    """
    fp_ext = f".{fp.split('.')[-1]}"
    if fp_ext not in [".png", ".tiff", ".pdf"]:
        raise ValueError("can only produce .png, .tiff or .pdf")

    kws_list: list[dict[str, Any]] = []
    if png:
        kws_list.append(
            dict(
                fname=fp.replace(fp_ext, ".png"),
                dpi=300,
                bbox_inches="tight",
            )
        )
    if tiff:
        kws_list.append(
            dict(
                fname=fp.replace(fp_ext, ".tiff"),
                dpi=300,
                bbox_inches="tight",
            )
        )
    if pdf:
        kws_list.append(
            dict(
                fname=fp.replace(fp_ext, ".pdf"),
                bbox_inches="tight",
            )
        )

    for kws_fig in kws_list:
        _check_overwrite(kws_fig["fname"], overwrite)

    for kws_fig in kws_list:
        fig.savefig(**kws_fig)
        print(f"saved: {kws_fig['fname']}")
