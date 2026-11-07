"""DataFrame reshaping and cross-tabulation utilities."""

from __future__ import annotations

from functools import reduce

import polars as pl


def groupby_value_counts(
    df: pl.DataFrame,
    group: str | list[str],
    col: str,
    prefix: str = "n_",
    dropna: bool = True,
) -> pl.DataFrame:
    """Get wide per-group value counts for a column.

    Counts each unique value of ``col`` within every group and spreads them
    into one column per value (missing combinations filled with 0).

    Args:
        df: Polars DataFrame containing the data
        group: Column name(s) to group by
        col: Column to get value counts for
        prefix: Prefix for the output value-count column names
        dropna: Whether to drop null values of ``col`` before counting

    Returns:
        Polars DataFrame with the group column(s) followed by one prefixed
        column per unique value of ``col``
    """
    group_cols = [group] if isinstance(group, str) else group
    work = df.drop_nulls(col) if dropna else df
    wide = (
        work.group_by([*group_cols, col])
        .len()
        .pivot(on=col, index=group_cols, values="len")
    )
    value_cols = sorted(c for c in wide.columns if c not in group_cols)
    wide = wide.with_columns([pl.col(c).fill_null(0) for c in value_cols])
    wide = wide.select([*group_cols, *value_cols]).sort(group_cols)
    if prefix:
        wide = wide.rename({c: f"{prefix}{c}" for c in value_cols})
    return wide


def crosstab(
    df: pl.DataFrame,
    columns: str | None = None,
    index: str | None = None,
    normalize: bool = True,
    margins: bool = True,
    fmt: str = ".1%",
) -> pl.DataFrame:
    """Create a cross-tabulation with optional formatting.

    Polars-native replacement for ``pd.crosstab``: counts co-occurrences of
    ``index`` x ``columns``, with optional grand-total normalization and
    margins. Because polars has no row index, the ``index`` values become the
    first column of the result rather than a DataFrame index.

    Args:
        df: Polars DataFrame containing the data
        columns: Column name whose values become the crosstab columns
        index: Column name whose values become the crosstab rows
        normalize: Whether to normalize cells by the grand total
        margins: Whether to append row/column "total" margins
        fmt: Format string for cell values, or empty string for raw values

    Returns:
        Polars DataFrame: the ``index`` column followed by one column per
        ``columns`` value (sorted), plus a "total" column/row when margins.
    """
    if columns is None or index is None:
        raise ValueError("both `columns` and `index` are required")

    ct = (
        df.drop_nulls([index, columns])
        .group_by(index, columns)
        .len(name="__n__")
        .pivot(on=columns, index=index, values="__n__", aggregate_function="sum")
    )
    value_cols = sorted((c for c in ct.columns if c != index), key=str)
    ct = ct.sort(index).select([index, *value_cols])
    grand_total = sum(ct[c].sum() for c in value_cols)

    categories = set(value_cols)
    index_values = {str(value) for value in ct[index].to_list()}
    if margins and ("total" in categories or "total" in index_values):
        raise ValueError("margin label 'total' collides with a category or index value")

    if margins:
        if value_cols:
            ct = ct.with_columns(pl.sum_horizontal(value_cols).alias("total"))
        else:
            ct = ct.with_columns(pl.lit(0).alias("total"))
        ct = ct.with_columns(pl.col(index).cast(pl.String))
        value_cols = [*value_cols, "total"]
        totals = {index: "total", **{c: ct[c].sum() for c in value_cols}}
        ct = pl.concat([ct, pl.DataFrame([totals])], how="vertical_relaxed")

    if normalize and grand_total:
        ct = ct.with_columns([(pl.col(c) / grand_total).alias(c) for c in value_cols])

    if fmt and value_cols:
        ct = ct.with_columns(
            [
                pl.col(c)
                .map_elements(lambda x: f"{x:{fmt}}", return_dtype=pl.String)
                .alias(c)
                for c in value_cols
            ]
        )

    return ct


def crosstab_np(
    df: pl.DataFrame,
    col1: str,
    col2: str,
    add_seperator: bool = True,
    fmt_n: str = ",",
    fmt_p: str = ".1%",
) -> pl.DataFrame:
    """Create a cross-tabulation with both counts (N) and percentages (P).

    Polars has no MultiIndex columns, so the N and P blocks are returned as
    flat prefixed columns (``N_<value>``, ``P_<value>``) rather than a
    two-level header.

    Args:
        df: Polars DataFrame containing the data
        col1: Column name whose values become the crosstab columns
        col2: Column name whose values become the crosstab rows
        add_seperator: Whether to add an empty column between the N and P blocks
        fmt_n: Format string for count values
        fmt_p: Format string for percentage values

    Returns:
        Polars DataFrame with flat N_*/P_* columns for the counts and percentages
    """
    tabn = crosstab(df, columns=col1, index=col2, normalize=False, fmt=fmt_n)
    tabp = crosstab(df, columns=col1, index=col2, normalize=True, fmt=fmt_p)

    tabn = tabn.rename({c: f"N_{c}" for c in tabn.columns if c != col2})
    tabp = tabp.rename({c: f"P_{c}" for c in tabp.columns if c != col2})
    out = tabn.join(tabp, on=col2, how="left")

    if add_seperator:
        n_cols = [c for c in out.columns if c.startswith("N_")]
        p_cols = [c for c in out.columns if c.startswith("P_")]
        out = out.with_columns(pl.lit("").alias("")).select(
            [col2, *n_cols, "", *p_cols]
        )

    return out


def join_df_list(dfs: list[pl.DataFrame], on: str | None = None) -> pl.DataFrame:
    """Combine a list of DataFrames into one.

    Args:
        dfs: List of polars DataFrames to combine
        on: Shared key column to join on. If ``None``, the frames are combined
            positionally via horizontal concatenation (requires equal heights),
            the polars analog of joining index-aligned frames.

    Returns:
        Combined polars DataFrame
    """
    if on is None:
        return pl.concat(dfs, how="horizontal")
    return reduce(lambda left, right: left.join(right, on=on, how="left"), dfs)
