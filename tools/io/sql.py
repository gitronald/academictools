"""SQL (mainly DuckDB) Utility Functions

This module provides utilities for working with DuckDB databases, including
creating views from parquet files, querying metadata, and loading data.
"""

from typing import Any, cast

import duckdb
import polars as pl
from IPython.display import display

from tools.utils.printing import wraprint


def _resolve_connection(
    fp_duckdb: str | None = None, duckdb_con: duckdb.DuckDBPyConnection | None = None
) -> tuple[duckdb.DuckDBPyConnection, bool]:
    """Resolve a DuckDB connection from either a file path or existing connection.

    Args:
        fp_duckdb: Path to a DuckDB database file
        duckdb_con: Existing DuckDB connection object

    Returns:
        Tuple of (connection, owns_connection) where owns_connection is True if
        the connection was created here and should be closed by the caller.

    Raises:
        ValueError: If neither or both fp_duckdb and duckdb_con are provided
    """
    if fp_duckdb and not duckdb_con:
        return duckdb.connect(fp_duckdb), True
    elif not fp_duckdb and duckdb_con:
        return duckdb_con, False
    else:
        raise ValueError("Must provide either duckdb_con or fp_duckdb")


def check_duckdb_settings() -> None:
    """Display all DuckDB configuration settings.

    This function queries the DuckDB settings table and displays all
    configuration options with their current values.

    Examples:
        >>> from tools import sql
        >>> sql.check_duckdb_settings()
        # Displays a table of all DuckDB settings

    Returns:
        None: Settings are displayed via IPython.display
    """
    with pl.Config(tbl_rows=-1):
        tab = duckdb.query("SELECT * FROM duckdb_settings()").pl()
        display(tab)


def create_parquet_view(
    fp_parquet: str,
    fp_duckdb: str | None = None,
    view_name: str = "data",
    duckdb_con: duckdb.DuckDBPyConnection | None = None,
    verbose: bool = False,
) -> None:
    """Create a DuckDB view from a parquet file.

    This function creates or replaces a view in a DuckDB database that reads
    from a parquet file. Either a connection object or database file path
    must be provided.

    Args:
        fp_parquet: Path to the parquet file to create a view from
        fp_duckdb: Path to the DuckDB database file. If provided, a new
            connection will be created and closed automatically. Cannot be
            used together with duckdb_con.
        view_name: Name for the created view (default: "data")
        duckdb_con: Existing DuckDB connection object. Cannot be used
            together with fp_duckdb.
        verbose: If True, print the SQL query and display column information
            (default: False)

    Raises:
        ValueError: If neither or both fp_duckdb and duckdb_con are provided

    Examples:
        >>> from tools import sql
        >>> # Using a database file path
        >>> sql.create_parquet_view(
        ...     fp_parquet="data.parquet",
        ...     fp_duckdb="my_database.duckdb",
        ...     view_name="my_data",
        ...     verbose=True
        ... )

        >>> # Using an existing connection
        >>> import duckdb
        >>> con = duckdb.connect("my_database.duckdb")
        >>> sql.create_parquet_view(
        ...     fp_parquet="data.parquet",
        ...     duckdb_con=con,
        ...     view_name="my_data"
        ... )
        >>> con.close()
    """
    duckdb_con, owns_con = _resolve_connection(fp_duckdb, duckdb_con)

    try:
        sql_query = (
            f"CREATE OR REPLACE VIEW \"{view_name}\" AS SELECT * FROM '{fp_parquet}'"
        )
        duckdb_con.sql(sql_query)
        if verbose:
            wraprint(sql_query)
            display(get_columns(duckdb_con, view_name))
    finally:
        if owns_con:
            duckdb_con.close()


def create_parquets_view(
    fps_parquets: list[str],
    fp_duckdb: str | None = None,
    view_name: str = "data",
    duckdb_con: duckdb.DuckDBPyConnection | None = None,
    verbose: bool = False,
) -> None:
    """Create a DuckDB view from multiple parquet files.

    This function creates or replaces a view in a DuckDB database that unions
    data from multiple parquet files. Either a connection object or database
    file path must be provided.

    Args:
        fps_parquets: List of paths to parquet files to union together
        fp_duckdb: Path to the DuckDB database file. If provided, a new
            connection will be created and closed automatically. Cannot be
            used together with duckdb_con.
        view_name: Name for the created view (default: "data")
        duckdb_con: Existing DuckDB connection object. Cannot be used
            together with fp_duckdb.
        verbose: If True, print the SQL query and display column information
            (default: False)

    Raises:
        ValueError: If neither or both fp_duckdb and duckdb_con are provided

    Examples:
        >>> from tools import sql
        >>> # Combine multiple parquet files into one view
        >>> sql.create_parquets_view(
        ...     fps_parquets=["data1.parquet", "data2.parquet", "data3.parquet"],
        ...     fp_duckdb="my_database.duckdb",
        ...     view_name="combined_data",
        ...     verbose=True
        ... )

        >>> # Using an existing connection
        >>> import duckdb
        >>> con = duckdb.connect("my_database.duckdb")
        >>> sql.create_parquets_view(
        ...     fps_parquets=["data1.parquet", "data2.parquet"],
        ...     duckdb_con=con,
        ...     view_name="combined_data"
        ... )
        >>> con.close()
    """
    duckdb_con, owns_con = _resolve_connection(fp_duckdb, duckdb_con)

    try:
        select_statements = [f"SELECT * FROM '{fp}'" for fp in fps_parquets]
        union_all_query = " UNION ALL ".join(select_statements)
        sql_query = f'CREATE OR REPLACE VIEW "{view_name}" AS {union_all_query}'

        duckdb_con.sql(sql_query)
        if verbose:
            wraprint(sql_query)
            display(get_columns(duckdb_con, view_name))
    finally:
        if owns_con:
            duckdb_con.close()


def drop_view(duckdb_con: duckdb.DuckDBPyConnection, view_name: str) -> None:
    """Drop a view from the DuckDB database if it exists.

    This function checks if a view exists in the database and drops it if found.
    If the view does not exist, the function does nothing.

    Args:
        duckdb_con: Active DuckDB connection object
        view_name: Name of the view to drop

    Examples:
        >>> import duckdb
        >>> from tools import sql
        >>> con = duckdb.connect("my_database.duckdb")
        >>> sql.drop_view(con, "my_view")
        dropping existing view
        >>> con.close()

    Note:
        This function prints "dropping existing view" when a view is dropped.
    """
    duckdb_con.execute(f'DROP VIEW IF EXISTS "{view_name}"')


def get_columns(
    duckdb_con: duckdb.DuckDBPyConnection | None = None,
    view_name: str | None = None,
    fp_parquet: str | None = None,
) -> pl.DataFrame:
    """Get column metadata for a DuckDB view or parquet file.

    This function returns column information either by describing a view in a
    DuckDB database or by reading the schema of a parquet file.

    Args:
        duckdb_con: Active DuckDB connection object. Required if fp_parquet
            is not provided.
        view_name: Name of the view to describe. Required if using duckdb_con.
        fp_parquet: Path to a parquet file to get schema from. If provided,
            duckdb_con and view_name are ignored.

    Returns:
        pl.DataFrame: DataFrame containing column metadata. When using a view,
            contains column names and types. When using a parquet file, returns
            empty DataFrame with column names.

    Raises:
        ValueError: If neither fp_parquet nor (duckdb_con and view_name) are
            provided

    Examples:
        >>> from tools import sql
        >>> # Get columns from a parquet file
        >>> cols = sql.get_columns(fp_parquet="data.parquet")
        >>> print(cols.columns)

        >>> # Get columns from a DuckDB view
        >>> import duckdb
        >>> con = duckdb.connect("my_database.duckdb")
        >>> cols = sql.get_columns(duckdb_con=con, view_name="my_view")
        >>> print(cols)
        >>> con.close()
    """
    if fp_parquet:
        return duckdb.sql(f"SELECT * FROM '{fp_parquet}' LIMIT 0;").pl()
    elif duckdb_con and view_name:
        return duckdb_con.sql(f'DESCRIBE "{view_name}";').pl()
    else:
        raise ValueError("Must provide either duckdb_con or fp_parquet")


def row_count(duckdb_con: duckdb.DuckDBPyConnection, col: str, view_name: str) -> int:
    """Count rows in a DuckDB view based on a specific column.

    This function counts non-NULL values in the specified column of a view.

    Args:
        duckdb_con: Active DuckDB connection object
        col: Column name to count (typically use '*' for total rows)
        view_name: Name of the view to query

    Returns:
        int: Count of non-NULL values in the specified column

    Examples:
        >>> import duckdb
        >>> from tools import sql
        >>> con = duckdb.connect("my_database.duckdb")
        >>> # Count all rows
        >>> total = sql.row_count(con, '*', 'my_view')
        >>> print(f"Total rows: {total}")
        >>> # Count non-NULL values in a specific column
        >>> count = sql.row_count(con, 'user_id', 'my_view')
        >>> print(f"User IDs: {count}")
        >>> con.close()
    """
    count_expr = "*" if col == "*" else f'"{col}"'
    # A COUNT(...) query always returns exactly one row, so fetchone() is never
    # None here; the cast resolves duckdb's `tuple | None` return for the type
    # checker without changing runtime behavior.
    row = cast(
        tuple[int],
        duckdb_con.execute(f'SELECT COUNT({count_expr}) FROM "{view_name}"').fetchone(),
    )
    return row[0]


def load_search_metadata(
    duckdb_con: duckdb.DuckDBPyConnection,
    view_name: str,
    display_head: bool | int = False,
) -> pl.DataFrame:
    """Load search engine results page (SERP) metadata from DuckDB.

    This function queries specific metadata columns from a SERP data view and
    returns them as a polars DataFrame. It's designed for SERP crawl data that
    includes query, location, URL, and tracking information.

    Args:
        duckdb_con: Active DuckDB connection object
        view_name: Name of the view containing SERP metadata
        display_head: If an integer > 0, display that many rows of the result.
            If False (default), don't display anything.

    Returns:
        pl.DataFrame: DataFrame with columns ['qry', 'loc', 'url', 'timestamp',
            'serp_id', 'crawl_id', 'response_code']

    Examples:
        >>> import duckdb
        >>> from tools import sql
        >>> con = duckdb.connect("serp_data.duckdb")
        >>> # Load metadata without displaying
        >>> metadata = sql.load_search_metadata(con, "serp_metadata")
        >>> print(f"Shape: {metadata.shape}")
        >>>
        >>> # Load and display first 5 rows
        >>> metadata = sql.load_search_metadata(
        ...     con, "serp_metadata", display_head=5
        ... )
        >>> con.close()

    Note:
        This function prints the SQL query and the shape of loaded data.
    """
    cols = ["qry", "loc", "url", "timestamp", "serp_id", "crawl_id", "response_code"]
    quoted_cols = ", ".join(f'"{c}"' for c in cols)
    sql_query = f'SELECT {quoted_cols} FROM "{view_name}"'
    print(sql_query)
    table = duckdb_con.sql(sql_query).pl()
    print(f"loaded: {view_name} - {table.shape}")
    if display_head:
        display(table.head(display_head))
    return table


def load_serp(
    duckdb_con: duckdb.DuckDBPyConnection,
    view_name: str = "serps",
    serp_id: str | None = None,
    cols: list[str] | None = None,
    verbose: bool = False,
) -> dict[str, Any] | None:
    """Load data for a single SERP (Search Engine Results Page) from DuckDB.

    This function queries a specific SERP by ID and returns its data as a
    dict (column name -> value). It's designed for SERP crawl data that
    includes HTML content, headers, and metadata.

    Args:
        duckdb_con: Active DuckDB connection object
        view_name: Name of the view containing SERP data (default: "serps")
        serp_id: Unique identifier for the SERP to load. Required.
        cols: List of specific columns to select. If empty, selects all columns.
            Available columns typically include: ['qry', 'loc', 'url', 'html',
            'headers', 'timestamp', 'serp_id', 'crawl_id', 'response_code']
        verbose: If True, print the SQL query (default: False)

    Returns:
        dict: Mapping of column name to value for the SERP, or None if no
            matching SERP is found

    Raises:
        ValueError: If serp_id is not provided

    Examples:
        >>> import duckdb
        >>> from tools import sql
        >>> con = duckdb.connect("serp_data.duckdb")
        >>> # Load all columns for a specific SERP
        >>> serp = sql.load_serp(con, view_name="serps", serp_id="abc123")
        >>> print(serp['qry'])  # Access the search query
        >>> print(serp['html'])  # Access the HTML content
        >>>
        >>> # Load only specific columns
        >>> serp = sql.load_serp(
        ...     con,
        ...     view_name="serps",
        ...     serp_id="abc123",
        ...     cols=['qry', 'html', 'timestamp'],
        ...     verbose=True
        ... )
        >>> con.close()
    """
    if not serp_id:
        raise ValueError("Must provide serp_id")

    cols = cols or []
    if cols:
        quoted_cols = ", ".join(f'"{c}"' for c in cols)
        select_str = f"SELECT {quoted_cols}"
    else:
        select_str = "SELECT *"
    sql_query = f'{select_str} FROM "{view_name}" WHERE "serp_id" = ?'
    if verbose:
        print(sql_query)
    result = duckdb_con.execute(sql_query, [serp_id]).pl()
    return result.row(0, named=True) if not result.is_empty() else None
