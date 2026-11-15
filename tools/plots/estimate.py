"""Point estimates with seeded bootstrap intervals, for the plotting helpers."""

from collections.abc import Callable
from typing import Any

import numpy as np

Estimator = str | Callable[[np.ndarray], float]
ErrorBar = tuple[str, float] | str | None

_DEFAULT_LEVELS = {"ci": 95.0, "pi": 95.0, "se": 1.0, "sd": 1.0}


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
            "errorbar must be None or one of 'ci', 'pi', 'se', 'sd', "
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
            ``("se", k)`` / ``("sd", k)`` for k standard errors / deviations
            around the estimate (the last two only with the mean), or None
        n_boot: Number of bootstrap resamples for ``"ci"``
        seed: Seed or generator for the bootstrap

    Returns:
        Tuple of (estimate, low, high); low and high are NaN when
        ``errorbar`` is None or fewer than two values remain
    """
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    func = _estimator_func(estimator)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    estimate = float(func(x))
    parsed = _parse_errorbar(errorbar)
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
    else:
        spread = float(np.std(x, ddof=1))
        if method == "se":
            spread /= np.sqrt(x.size)
        low, high = estimate - level * spread, estimate + level * spread
    return estimate, float(low), float(high)
