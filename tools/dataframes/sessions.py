"""Time-delimited session detection."""

import polars as pl


def add_sessions(
    df: pl.DataFrame,
    tvar: str = "_pulled",
    session_delimiter: float = 1 * 60,
) -> pl.DataFrame:
    """Add session columns with time-delimited session IDs.

    Splits rows into sessions based on time gaps exceeding the delimiter
    threshold and assigns sequential session IDs.

    Args:
        df: Polars DataFrame to add session columns to
        tvar: Datetime column name to use for session splitting
        session_delimiter: Minimum gap in seconds to start a new session

    Returns:
        Polars DataFrame with added 'sesh_time_diff', 'sesh_time_elapsed', and
        'sesh_id' columns
    """
    df = df.sort(tvar)

    # Seconds between consecutive rows; elapsed is the gap to the next row
    df = df.with_columns(
        sesh_time_diff=pl.col(tvar).diff().dt.total_seconds(fractional=True)
    )
    # A new session starts wherever the gap exceeds the delimiter; the running
    # count of those breaks is the session id (0 for the first session).
    return df.with_columns(
        sesh_time_elapsed=pl.col("sesh_time_diff").shift(-1),
        sesh_id=(pl.col("sesh_time_diff") > session_delimiter)
        .fill_null(False)
        .cum_sum(),
    )
