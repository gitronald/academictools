"""Nonparametric statistical tests."""

from collections import OrderedDict

import numpy as np
import polars as pl
import scipy.stats as st
from statsmodels.stats.contingency_tables import mcnemar

from tools.dataframes.describe import describe
from tools.utils.printing import print_line


def _group_arrays(data: pl.DataFrame, group: str, metric: str) -> list[np.ndarray]:
    """Split ``metric`` into one numpy array per group, in sorted key order."""
    keys = data.get_column(group).drop_nulls().unique().sort().to_list()
    return [data.filter(pl.col(group) == k).get_column(metric).to_numpy() for k in keys]


def p_value_sig(p_val: float) -> str:
    """Convert p-value to significance stars.

    Args:
        p_val: P-value to convert

    Returns:
        Significance indicator: '***' (p<0.001), '**' (p<0.01), '*' (p<0.05), or ''
    """
    if p_val < 0.001:
        return "***"
    elif p_val < 0.01:
        return "**"
    elif p_val < 0.05:
        return "*"
    else:
        return ""


def kruskal_test(
    data: pl.DataFrame,
    group: str,
    metric: str,
    nan_policy: str = "raise",
) -> dict[str, float]:
    """Run Kruskal-Wallis H-test for independent samples.

    Args:
        data: pl.DataFrame containing the data
        group: Column name defining the groups
        metric: Column name of the metric to test
        nan_policy: How to handle NaN values ('raise', 'omit', 'propagate')

    Returns:
        dict with 'chi' and 'P' keys
    """
    groups = _group_arrays(data, group, metric)
    X, p = st.kruskal(*groups, nan_policy=nan_policy)
    return {"chi": X, "P": p}


def kruskal_test_multi(
    data: pl.DataFrame, metrics: list[str], groups: list[str], details: bool = True
) -> None:
    """Run Kruskal-Wallis tests across multiple metrics and grouping variables.

    Args:
        data: pl.DataFrame containing the data
        metrics: List of column names to test
        groups: List of grouping column names
        details: Whether to print descriptive statistics per group
    """
    pdf = data
    for metric in metrics:
        col_data = pdf.get_column(metric)
        mean_val = col_data.mean()
        std_val = col_data.std()
        s = f"Mean {metric}: {mean_val:,.1f} (SD = {std_val:,.1f})"
        print(s)

        for group in groups:
            print_line()
            print(f"Group: {group}")
            print(f"Metric: {metric}")

            if details:
                tab = describe(pdf, metric, group=group)
                print(f"{tab}\n")

            result = kruskal_test(pdf, group=group, metric=metric, nan_policy="omit")
            X, p = f"{result['chi']:.3f}", f"{result['P']:.4f}"
            print(f"Kruskal-Wallis [KW] $X^{{2}} = {X}$; $P = {p}$)")
            print()


def mannw_test(data: pl.DataFrame, group: str, metric: str) -> dict[str, float]:
    """Run Mann-Whitney U test for two independent samples.

    Args:
        data: pl.DataFrame containing the data
        group: Column name defining the two groups
        metric: Column name of the metric to test

    Returns:
        dict with 'U' and 'P' keys
    """
    groups = _group_arrays(data, group, metric)
    U, p = st.mannwhitneyu(*groups)
    return {"U": U, "P": p}


def spearmanr(
    data: pl.DataFrame,
    metric1: str,
    metric2: str,
    nan_policy: str = "omit",
) -> dict[str, float]:
    """Compute Spearman rank-order correlation between two columns.

    Args:
        data: pl.DataFrame containing the data
        metric1: First column name
        metric2: Second column name
        nan_policy: How to handle NaN values ('raise', 'omit', 'propagate')

    Returns:
        dict with 'rho' and 'p' keys
    """
    rho, p = st.spearmanr(
        data.get_column(metric1).to_numpy(),
        data.get_column(metric2).to_numpy(),
        nan_policy=nan_policy,
    )
    return {"rho": rho, "p": p}


def nonparametric_sig_test(
    data: pl.DataFrame, group: str, metric: str
) -> OrderedDict[str, str | float]:
    """Run appropriate nonparametric significance test based on number of groups.

    Uses Mann-Whitney U for two groups and Kruskal-Wallis for more than two.

    Args:
        data: pl.DataFrame containing the data
        group: Column name defining the groups
        metric: Column name of the metric to test

    Returns:
        OrderedDict with group name and test statistic (U or chi) and p-value
    """
    ngroups = data.get_column(group).drop_nulls().n_unique()
    if ngroups == 2:
        result = mannw_test(data, group, metric)
        out_data = OrderedDict(
            [("group", group), ("U", result["U"]), ("p", result["P"])]
        )

    else:
        result = kruskal_test(data, group, metric)
        out_data = OrderedDict(
            [("group", group), ("chi", result["chi"]), ("p", result["P"])]
        )
    return out_data


def mcnemar_test(
    data: pl.DataFrame, group1: str, group2: str
) -> tuple[pl.DataFrame, dict[str, float]]:
    """Run McNemar's test for paired nominal data.

    Args:
        data: pl.DataFrame containing the data
        group1: First binary column name
        group2: Second binary column name

    Returns:
        Tuple of (contingency DataFrame, dict with 'pvalue' and 'chi2' keys).
        The contingency table is a polars DataFrame whose first column holds the
        ``group1`` values (rows); the ``group2`` values are the count columns.
    """
    categories = sorted(
        set(data[group1].drop_nulls().to_list())
        | set(data[group2].drop_nulls().to_list())
    )
    if len(categories) > 2:
        raise ValueError("McNemar's test requires no more than two categories")
    observed_counts = [
        [
            data.filter((pl.col(group1) == row) & (pl.col(group2) == column)).height
            for column in categories
        ]
        for row in categories
    ]
    counts = [
        [
            observed_counts[row][column]
            if row < len(categories) and column < len(categories)
            else 0
            for column in range(2)
        ]
        for row in range(2)
    ]
    xtab = pl.DataFrame(
        {
            group1: categories,
            **{
                str(category): [row[i] for row in observed_counts]
                for i, category in enumerate(categories)
            },
        }
    )
    stats = mcnemar(counts)
    # mcnemar returns an untyped statsmodels _Bunch exposing .pvalue/.statistic
    output: dict[str, float] = {}
    output["pvalue"] = stats.pvalue  # type: ignore[attr-defined]
    output["chi2"] = stats.statistic  # type: ignore[attr-defined]
    return xtab, output


def get_modal_category(df: pl.DataFrame, group: str, category: str) -> pl.DataFrame:
    """Get the most frequent (modal) category per group.

    Args:
        df: Polars DataFrame containing the data
        group: Column name to group by
        category: Column name to get the modal value for

    Returns:
        Polars DataFrame with columns ``[group, "category"]`` holding the most
        frequent ``category`` value per group (ties broken by category ascending).
    """
    return (
        df.group_by(group, category)
        .len()
        .sort([group, "len", category], descending=[False, True, False])
        .group_by(group, maintain_order=True)
        .agg(pl.col(category).first().alias("category"))
        .sort(group)
    )
