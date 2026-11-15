# pyrefly: ignore-errors
"""Record seaborn and statsmodels reference outputs as JSON test fixtures.

The package does not depend on seaborn or statsmodels; this script runs once in
a throwaway environment that has them, and its output is committed:

    uv run --no-project --with seaborn==0.13.2 --with statsmodels==0.15.0 \
        --with matplotlib==3.11.2 --with numpy==2.5.4 --with scipy==1.18.1 \
        python tests/fixtures/record_reference.py tests/fixtures

The tests compare the scipy and matplotlib reimplementations against these
files, so re-record only when the reference versions change on purpose.
"""

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
import statsmodels  # noqa: E402
from seaborn._statistics import LetterValues  # noqa: E402
from seaborn.categorical import Beeswarm  # noqa: E402
from statsmodels.stats.contingency_tables import mcnemar  # noqa: E402
from statsmodels.stats.proportion import proportions_ztest  # noqa: E402


def _versions() -> dict[str, str]:
    return {
        "seaborn": sns.__version__,
        "statsmodels": statsmodels.__version__,
        "matplotlib": matplotlib.__version__,
        "numpy": np.__version__,
    }


def _floats(values: Any) -> list[float]:
    return [float(v) for v in np.asarray(values, dtype=float).ravel()]


def record_statsmodels() -> dict[str, Any]:
    tables = [
        [[10, 5], [3, 12]],
        [[0, 1], [9, 0]],
        [[20, 15], [15, 20]],
        [[4, 0], [0, 6]],
        [[100, 34], [12, 80]],
        [[1, 25], [3, 1]],
    ]
    mcnemar_cases = []
    for table in tables:
        for exact in (True, False):
            for correction in (True, False):
                b, c = table[0][1], table[1][0]
                if not exact and b + c == 0:
                    continue  # chi-square is undefined (0/0)
                result = mcnemar(table, exact=exact, correction=correction)
                mcnemar_cases.append(
                    {
                        "table": table,
                        "exact": exact,
                        "correction": correction,
                        "statistic": float(result.statistic),
                        "pvalue": float(result.pvalue),
                    }
                )

    ztest_cases = []
    inputs = [
        ([10, 20], [100, 100]),
        ([50, 50], [100, 100]),
        ([33, 47], [120, 95]),
        ([5, 1], [40, 60]),
        ([300, 280], [1000, 1100]),
    ]
    for count, nobs in inputs:
        for value in (0.0, 0.05):
            for alternative in ("two-sided", "larger", "smaller"):
                z, p = proportions_ztest(count, nobs, value, alternative=alternative)
                ztest_cases.append(
                    {
                        "count": count,
                        "nobs": nobs,
                        "value": value,
                        "alternative": alternative,
                        "z": float(z),
                        "p": float(p),
                    }
                )
    return {"versions": _versions(), "mcnemar": mcnemar_cases, "ztest": ztest_cases}


def _letter_value_datasets() -> dict[str, list[float]]:
    rng = np.random.default_rng(20261115)
    return {
        "normal_10": _floats(rng.normal(size=10)),
        "normal_100": _floats(rng.normal(size=100)),
        "normal_1000": _floats(rng.normal(size=1000)),
        "exponential_500": _floats(rng.exponential(scale=2.0, size=500)),
        "integers_ties_200": _floats(rng.integers(0, 8, size=200)),
        "lognormal_5000": _floats(rng.lognormal(size=5000)),
    }


def record_letter_values() -> dict[str, Any]:
    datasets = _letter_value_datasets()
    cases = []
    for name, data in datasets.items():
        x = np.asarray(data)
        for k_depth in ("tukey", "proportion", "trustworthy", "full", 3):
            estimator = LetterValues(k_depth, outlier_prop=0.007, trust_alpha=0.05)
            lv = estimator(x)
            k = lv["k"]
            levels = lv["levels"]
            exponent = (levels - 1 - k).astype(float)
            tails = levels < (k - 1)
            rel = {
                "linear": levels + 1,
                "exponential": 2**exponent,
                "area": 2 ** (exponent - tails) / np.diff(lv["values"]),
            }
            widths = {m: _floats(r / r.max() * 0.8) for m, r in rel.items()}
            cases.append(
                {
                    "dataset": name,
                    "k_depth": k_depth,
                    "k": int(k),
                    "levels": [int(v) for v in levels],
                    "percs": _floats(lv["percs"]),
                    "values": _floats(lv["values"]),
                    "fliers": _floats(lv["fliers"]),
                    "median": float(lv["median"]),
                    "widths": widths,
                }
            )
    return {"versions": _versions(), "datasets": datasets, "cases": cases}


def record_beeswarm() -> dict[str, Any]:
    rng = np.random.default_rng(7)
    # Algorithm level: (x, y, r) rows in point coordinates, sorted by y.
    algorithm = []
    for n, radius, spread in [(30, 3.0, 40.0), (80, 2.5, 30.0), (200, 2.0, 25.0)]:
        y = np.sort(rng.normal(scale=spread, size=n)) + 150.0
        x = np.full(n, 100.0)
        r = np.full(n, radius)
        xyr = np.c_[x, y, r]
        out = Beeswarm().beeswarm(xyr)
        algorithm.append({"input": xyr.tolist(), "output": out.tolist()})
    # Mixed radii exercise the neighbor search with unequal sizes.
    y = np.sort(rng.uniform(0, 60, size=40)) + 50.0
    r = rng.uniform(1.5, 4.0, size=40)
    xyr = np.c_[np.full(40, 75.0), y, r]
    algorithm.append(
        {"input": xyr.tolist(), "output": Beeswarm().beeswarm(xyr).tolist()}
    )

    # End to end: swarmplot on a fixed figure with seaborn's defaults.
    plots = []
    groups = {"a": rng.normal(0, 1, 60), "b": rng.normal(1, 0.5, 45)}
    for orient in ("x", "y"):
        for size in (5, 3):
            fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
            cats = np.repeat(list(groups), [len(v) for v in groups.values()])
            vals = np.concatenate(list(groups.values()))
            if orient == "x":
                sns.swarmplot(x=cats, y=vals, size=size, ax=ax)
            else:
                sns.swarmplot(x=vals, y=cats, size=size, ax=ax)
            fig.canvas.draw()
            plots.append(
                {
                    "orient": orient,
                    "size": size,
                    "groups": {k: _floats(v) for k, v in groups.items()},
                    "xlim": _floats(ax.get_xlim()),
                    "ylim": _floats(ax.get_ylim()),
                    "offsets": [c.get_offsets().tolist() for c in ax.collections],
                }
            )
            plt.close(fig)
    return {"versions": _versions(), "algorithm": algorithm, "plots": plots}


def record_palettes_and_styles() -> dict[str, Any]:
    palettes = []
    for name, n in [
        ("viridis", 5),
        ("Blues", 3),
        ("magma", 7),
        ("tab10", 4),
        ("Set2", 10),
        ("Paired", 12),
        ("coolwarm_r", 4),
    ]:
        palettes.append(
            {
                "name": name,
                "n": n,
                "colors": [list(c) for c in sns.color_palette(name, n)],
            }
        )
    styles = {style: dict(sns.axes_style(style)) for style in ("ticks", "whitegrid")}
    return {"versions": _versions(), "palettes": palettes, "styles": styles}


def main() -> None:
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "statsmodels_reference.json": record_statsmodels(),
        "letter_values_reference.json": record_letter_values(),
        "beeswarm_reference.json": record_beeswarm(),
        "palettes_reference.json": record_palettes_and_styles(),
    }
    for name, payload in outputs.items():
        (out_dir / name).write_text(json.dumps(payload, indent=1) + "\n")
        print(f"wrote {out_dir / name}")


if __name__ == "__main__":
    main()
