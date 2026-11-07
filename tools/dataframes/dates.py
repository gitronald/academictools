"""Date and timestamp processing utilities."""

import polars as pl


def process_timestamps(
    col: pl.Series,
    ts_format: str = "ISO8601",
    tz_from: str = "UTC",
    tz_to: str = "US/Eastern",
    drop_tz: bool = True,
) -> pl.Series:
    """Convert timestamps between timezones.

    Args:
        col: Polars Series of timestamp strings or datetime values
        ts_format: Datetime format string or 'ISO8601' for automatic parsing
        tz_from: Source timezone to localize from
        tz_to: Target timezone to convert to
        drop_tz: Whether to remove timezone info after conversion

    Returns:
        Polars Series of converted datetime values
    """
    fmt = None if ts_format == "ISO8601" else ts_format
    if col.dtype == pl.String:
        col = col.str.to_datetime(format=fmt)
    if not isinstance(col.dtype, pl.Datetime) or col.dtype.time_zone is None:
        col = col.dt.replace_time_zone(tz_from)
    col = col.dt.convert_time_zone(tz_to)
    return col.dt.replace_time_zone(None) if drop_tz else col


def decompose_dates(dates: pl.Series, format: str = "%Y%m%d") -> pl.DataFrame:
    """Decompose date strings into week, month, and year columns.

    Args:
        dates: Polars Series of date strings
        format: Date format string for parsing

    Returns:
        DataFrame with 'date', 'week', 'month', and 'year' columns
    """
    parsed = dates.str.to_datetime(format=format) if dates.dtype == pl.String else dates
    return pl.DataFrame({"date": parsed}).with_columns(
        week=pl.col("date").dt.week(),
        month=pl.col("date").dt.month(),
        year=pl.col("date").dt.iso_year(),
    )
