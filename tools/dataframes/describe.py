"""Polars descriptive statistics."""

import polars as pl


def describe(
    df: pl.DataFrame, col: str, group: str = "", confidence_intervals: bool = True
) -> pl.DataFrame:
    """Generate descriptive statistics for a polars DataFrame column.

    Computes count, sum, mean, sem, std, median, min, max, and optionally
    95% confidence intervals. Supports optional grouping.

    Args:
        df: Polars DataFrame containing the data
        col: Column name to compute statistics for
        group: Optional column name to group by (empty string for no grouping)
        confidence_intervals: Whether to include 95% confidence intervals

    Returns:
        Polars DataFrame with descriptive statistics
    """
    expressions = [
        pl.col(col).len().alias("count"),
        pl.col(col).sum().alias("sum"),
        pl.col(col).mean().alias("mean"),
        (pl.col(col).std() / pl.col(col).count().sqrt()).alias("sem"),
        pl.col(col).std().alias("std"),
        pl.col(col).median().alias("median"),
        pl.col(col).min().alias("min"),
        pl.col(col).max().alias("max"),
    ]

    if df[col].dtype == pl.Float64 or df[col].dtype == pl.Float32:
        expressions.insert(1, pl.col(col).is_not_nan().sum().alias("notnull"))
    else:
        expressions.insert(1, pl.col(col).is_not_null().sum().alias("notnull"))

    if group:
        group_descriptives = df.group_by(group).agg(expressions)
        group_descriptives = group_descriptives.with_columns(
            (pl.col("count") / pl.col("count").sum()).alias("prop")
        )
        group_descriptives = group_descriptives.with_columns(
            (pl.col("notnull") / pl.col("notnull").sum()).alias("nn_prop")
        )
        overall = df.select(expressions)
        overall = (
            overall.with_columns(pl.lit("overall").alias(group))
            .select([pl.col(group), *overall.columns])
            .with_columns(pl.lit(1.0).alias("prop"))
            .with_columns(pl.lit(1.0).alias("nn_prop"))
        )
        descriptives = pl.concat([group_descriptives, overall], how="vertical_relaxed")
    else:
        descriptives = df.select(expressions)

    # Add 95% confidence intervals if requested
    if confidence_intervals:
        descriptives = descriptives.with_columns(
            [
                (pl.col("mean") - 1.96 * pl.col("sem")).alias("95ci_low"),
                (pl.col("mean") + 1.96 * pl.col("sem")).alias("95ci_high"),
            ]
        )

    # Ensure all summary columns are floats
    float_cols = ["sum", "min", "max"]
    descriptives = descriptives.with_columns(
        [pl.col(c).cast(pl.Float64) for c in float_cols]
    )

    col_order = ["column"]
    col_order += (
        [group, "count", "prop", "notnull", "nn_prop", "mean"]
        if group
        else ["count", "notnull", "mean"]
    )
    col_order += ["95ci_low", "95ci_high"] if confidence_intervals else []
    col_order += ["sem", "std", "median", "min", "max", "sum"]

    # Include column name as first column reference
    descriptives = descriptives.select(
        [pl.lit(col).alias("column"), *descriptives.columns]
    )
    descriptives = descriptives.select(col_order)

    return descriptives
