"""Point estimates with bootstrap or closed-form intervals, for the plot helpers."""

from collections.abc import Callable
from statistics import NormalDist
from typing import Any

import numpy as np

Estimator = str | Callable[[np.ndarray], float]
ErrorBar = tuple[str, float] | str | None

_DEFAULT_LEVELS = {"ci": 95.0, "pi": 95.0, "wilson": 95.0, "se": 1.0, "sd": 1.0}
# Methods that describe the mean, so they take only the mean estimator.
_MEAN_ONLY = ("se", "sd", "wilson")


def _estimator_func(estimator: Estimator) -> Callable[..., Any]:
    """Resolve an estimator name ('mean', 'median', ...) or callable."""
    if callable(estimator):
        return estimator
    func = getattr(np, estimator, None)
    if func is None or not callable(func):
        raise ValueError(f"unknown estimator {estimator!r}")
    return func


def bootstrap(
    *arrays: np.ndarray,
    func: Callable[..., Any],
    n_boot: int = 1000,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Resample rows with replacement and apply ``func`` to each resample.

    The arrays are resampled jointly (row ``i`` of every array moves together),
    so ``func`` can fit a model across several columns.

    Args:
        *arrays: One or more arrays of equal length
        func: Called as ``func(*resampled_arrays)`` once per resample
        n_boot: Number of resamples
        seed: Seed or generator; the same seed gives the same resamples

    Returns:
        Array of the ``n_boot`` results stacked along the first axis
    """
    if not arrays:
        raise ValueError("bootstrap needs at least one array")
    n = len(arrays[0])
    if any(len(a) != n for a in arrays):
        raise ValueError("bootstrap arrays must all have the same length")
    rng = np.random.default_rng(seed)
    arrays = tuple(np.asarray(a) for a in arrays)
    results = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        results.append(func(*(a[idx] for a in arrays)))
    return np.asarray(results)


def _parse_errorbar(errorbar: ErrorBar) -> tuple[str, float] | None:
    if errorbar is None:
        return None
    if isinstance(errorbar, str):
        method, level = errorbar, _DEFAULT_LEVELS.get(errorbar)
    else:
        method, level = errorbar
    if method not in _DEFAULT_LEVELS or level is None:
        raise ValueError(
            "errorbar must be None or one of 'ci', 'pi', 'wilson', 'se', 'sd', "
            f"optionally with a level, got {errorbar!r}"
        )
    return method, float(level)


def estimate_interval(
    values: np.ndarray,
    estimator: Estimator = "mean",
    errorbar: ErrorBar = ("ci", 95),
    n_boot: int = 1000,
    seed: int | np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Estimate a statistic of ``values`` and an interval around it.

    Args:
        values: 1-D array of observations (NaNs are dropped)
        estimator: Name of a numpy reduction ('mean', 'median', ...) or a
            callable taking a 1-D array
        errorbar: ``("ci", level)`` for a percentile bootstrap confidence
            interval, ``("pi", level)`` for a percentile interval of the data,
            ``("wilson", level)`` for the Wilson score interval of a proportion
            (the values must all be 0 or 1, else ValueError),
            ``("se", k)`` / ``("sd", k)`` for k standard errors / deviations
            around the mean (the last three only with the mean estimator; any
            other raises ValueError), or None
        n_boot: Number of bootstrap resamples for ``"ci"``
        seed: Seed or generator for the bootstrap

    Returns:
        Tuple of (estimate, low, high); low and high are NaN when
        ``errorbar`` is None or fewer than two values remain
    """
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    func = _estimator_func(estimator)
    parsed = _parse_errorbar(errorbar)
    if parsed is not None and parsed[0] in _MEAN_ONLY and func is not np.mean:
        raise ValueError(
            f"errorbar {parsed[0]!r} describes the mean; "
            f"use estimator='mean' or a 'ci' or 'pi' interval, got {estimator!r}"
        )
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    estimate = float(func(x))
    if parsed is None or x.size < 2:
        return estimate, float("nan"), float("nan")
    method, level = parsed
    if method == "ci":
        boots = bootstrap(x, func=func, n_boot=n_boot, seed=seed)
        edge = (100 - level) / 2
        low, high = np.nanpercentile(boots, [edge, 100 - edge])
    elif method == "pi":
        edge = (100 - level) / 2
        low, high = np.percentile(x, [edge, 100 - edge])
    elif method == "wilson":
        low, high = wilson_interval(x, level)
    else:
        spread = float(np.std(x, ddof=1))
        if method == "se":
            spread /= np.sqrt(x.size)
        low, high = estimate - level * spread, estimate + level * spread
    return estimate, float(low), float(high)


def wilson_interval(values: np.ndarray, level: float = 95.0) -> tuple[float, float]:
    """Wilson score interval for the proportion of ones in 0/1 ``values``.

    A closed form, so it costs one pass over the values, and unlike the normal
    approximation (``("se", 1.96)``) its bounds stay in [0, 1] and are not
    zero-width at 0 or n successes.

    Args:
        values: 1-D array of 0/1 observations (booleans count as 0/1)
        level: Confidence level in percent

    Returns:
        Tuple of (low, high)

    Raises:
        ValueError: If any value is not 0 or 1, or ``values`` is empty
    """
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        raise ValueError("wilson_interval needs at least one value")
    if not np.isin(x, (0.0, 1.0)).all():
        raise ValueError("errorbar 'wilson' needs values that are all 0 or 1")
    z = NormalDist().inv_cdf(0.5 + level / 200)
    n = x.size
    p = float(x.mean())
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z / denom * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return max(center - half, 0.0), min(center + half, 1.0)
