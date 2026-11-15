"""Parametric statistical tests."""

from collections import OrderedDict
from typing import Any, cast

import numpy as np
import polars as pl
import scipy.stats as st


def paired_ttest(
    df: pl.DataFrame, var1: str, var2: str, root: str = "all", date: str = ""
) -> OrderedDict[str, Any]:
    """Run paired t-test on two columns in a DataFrame.

    Args:
        df: pl.DataFrame containing the data
        var1: First variable column name
        var2: Second variable column name
        root: Filter by root value, or 'all' to use full dataset
        date: Filter by date value, or '' to use full dataset

    Returns:
        OrderedDict with root, variable means, effect size, t-statistic, and p-value
    """
    if root != "all":
        df = df.filter(pl.col("root") == root)
    if date != "":
        df = df.filter(pl.col("date") == date)

    pairs = df.select(var1, var2).fill_nan(None).drop_nulls([var1, var2])

    t, p = st.ttest_rel(
        pairs.get_column(var1).to_numpy(), pairs.get_column(var2).to_numpy()
    )
    m1 = pairs.get_column(var1).mean()
    m2 = pairs.get_column(var2).mean()
    if m1 is None or m2 is None:
        raise ValueError(
            f"paired_ttest: column '{var1}' or '{var2}' has no non-null values"
        )
    # cast narrows polars' over-wide Series.mean() union (a numeric mean is float)
    mean1 = cast(float, m1)
    mean2 = cast(float, m2)
    overall = OrderedDict(
        [
            ("root", root),
            (var1, mean1),
            (var2, mean2),
            ("effect_size", mean2 - mean1),
            ("t", t),
            ("P", p),
        ]
    )
    return overall


def z_test(
    count: list[int], nobs: list[int], value: float = 0, alternative: str = "two-sided"
) -> dict[str, float]:
    """Two-proportion z-test with a pooled variance.

    Matches ``statsmodels.stats.proportion.proportions_ztest`` for two samples:
    the difference ``p1 - p2 - value`` over the standard error from the pooled
    proportion.

    Args:
        count: Number of successes for each sample (length 2)
        nobs: Number of observations for each sample (length 2)
        value: Null hypothesis difference in proportions (default: 0)
        alternative: 'two-sided', 'larger' (p1 - p2 > value), or 'smaller'

    Returns:
        dict with z statistic and p value

    Raises:
        ValueError: If count or nobs is not length 2, or alternative is unknown
    """
    if len(count) != 2 or len(nobs) != 2:
        raise ValueError("z_test takes exactly two samples")
    if alternative not in ("two-sided", "larger", "smaller"):
        raise ValueError(
            "alternative must be 'two-sided', 'larger', or 'smaller', "
            f"got {alternative!r}"
        )
    p1, p2 = count[0] / nobs[0], count[1] / nobs[1]
    pooled = sum(count) / sum(nobs)
    std = np.sqrt(pooled * (1 - pooled) * (1 / nobs[0] + 1 / nobs[1]))
    z = float((p1 - p2 - value) / std)
    if alternative == "two-sided":
        p = 2 * st.norm.sf(abs(z))
    elif alternative == "larger":
        p = st.norm.sf(z)
    else:
        p = st.norm.cdf(z)
    return {"z": z, "p": float(p)}


def pearsonr_ci(
    data: pl.DataFrame, x: str, y: str, alpha: float = 0.05, filterna: bool = True
) -> dict[str, float]:
    """Calculate Pearson correlation with confidence interval.

    Args:
        data: pl.DataFrame containing the data
        x: Column name for first variable
        y: Column name for second variable
        alpha: Significance level for confidence interval
        filterna: Whether to filter out NaN values before computing

    Returns:
        Dict with 'r' (correlation), 'p' (p-value), 'ci_min' and 'ci_max'
    """
    if filterna:
        data = data.select(x, y).fill_nan(None).drop_nulls([x, y])

    xv = data.get_column(x).to_numpy()
    yv = data.get_column(y).to_numpy()
    r, p = st.pearsonr(xv, yv)

    r_z = np.arctanh(r)
    se = 1 / np.sqrt(xv.size - 3)
    z = st.norm.ppf(1 - alpha / 2)
    lo_z, hi_z = r_z - z * se, r_z + z * se
    lo, hi = np.tanh((lo_z, hi_z))
    return {"r": r, "p": p, "ci_min": lo, "ci_max": hi}
