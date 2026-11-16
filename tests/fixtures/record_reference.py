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


PARITY_COLORS = ["whitesmoke", "#4c72b0", "red", "seagreen", "#d1432a"]


def _rgb(color: Any) -> list[float]:
    return [float(c) for c in matplotlib.colors.to_rgb(color)]


def _parity_frame() -> Any:
    import pandas as pd

    rng = np.random.default_rng(11)
    n = 300
    return pd.DataFrame(
        {
            "cat": rng.choice(["a", "b", "c"], n),
            "grp": rng.choice(["p", "q"], n),
            "val": rng.beta(4, 2, n),
        }
    )


def record_parity() -> dict[str, Any]:
    """Defaults and arguments seaborn applies on top of the drawing algorithms."""
    from seaborn.utils import desaturate

    desat = [
        {"color": c, "prop": p, "rgb": list(desaturate(c, p))}
        for c in PARITY_COLORS
        for p in (0.75, 0.5)
    ]

    palettes = []
    for kind in ("dark", "light"):
        for color in PARITY_COLORS:
            for suffix in ("", "_r"):
                for n in (3, 6):
                    name = f"{kind}:{color}{suffix}"
                    palettes.append(
                        {
                            "name": name,
                            "n": n,
                            "colors": [list(c) for c in sns.color_palette(name, n)],
                        }
                    )

    themes = {}
    for style in ("ticks", "whitegrid"):
        matplotlib.rcdefaults()
        base = dict(matplotlib.rcParams)
        sns.set_theme(style=style)
        changed = {}
        for key, value in matplotlib.rcParams.items():
            if value == base[key]:
                continue
            if key == "axes.prop_cycle":
                changed[key] = [_rgb(c["color"]) for c in value]
            elif key != "image.cmap":  # seaborn's own colormap, not ported
                changed[key] = list(value) if isinstance(value, tuple) else value
        themes[style] = changed
    matplotlib.rcdefaults()

    df = _parity_frame()

    # Scatter marker edges, for fixed and varying sizes
    scatter = []
    for sizes in ([4], [8], [36], [2, 10, 50]):
        fig, ax = plt.subplots()
        s = sizes[0] if len(sizes) == 1 else np.resize(sizes, len(df))
        points = sns.scatterplot(df, x="val", y="val", s=s, ax=ax).collections[0]
        scatter.append(
            {
                "sizes": sizes,
                "linewidths": _floats(points.get_linewidths()),
                "edgecolor": [float(v) for v in points.get_edgecolors()[0]],
            }
        )
        plt.close(fig)

    # Histogram bar edge widths, which scale with the bin width on screen
    hist = []
    for figsize, bins, dpi in [((4, 2), 20, 100), ((4, 2), 80, 100), ((8, 3), 10, 150)]:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        sns.histplot(df, x="val", bins=bins, ax=ax)
        hist.append(
            {
                "figsize": list(figsize),
                "bins": bins,
                "dpi": dpi,
                "linewidth": float(ax.patches[0].get_linewidth()),
            }
        )
        plt.close(fig)

    # Histogram layer order: the first hue level is drawn last, so on top
    fig, ax = plt.subplots()
    sns.histplot(df, x="val", hue="grp", hue_order=["p", "q"], bins=5, ax=ax)
    hist_layers = [_rgb(p.get_facecolor()) for p in ax.patches[:10]]
    plt.close(fig)

    # Box and letter-value line colors derived from the fill ("auto")
    line_colors = []
    for color in [*PARITY_COLORS, "black"]:
        fig, ax = plt.subplots()
        sns.boxplot(df, x="cat", y="val", color=color, ax=ax)
        line_colors.append(
            {"plot": "boxplot", "color": color, "line": _rgb(ax.lines[0].get_color())}
        )
        plt.close(fig)
        fig, ax = plt.subplots()
        sns.boxenplot(df, x="cat", y="val", color=color, ax=ax)
        edge = ax.collections[0].get_edgecolor()[0]
        line_colors.append({"plot": "boxenplot", "color": color, "line": _rgb(edge)})
        plt.close(fig)
    fig, ax = plt.subplots()
    sns.boxplot(df, x="cat", y="val", hue="grp", palette=["#ff7f0e", "#1f77b4"], ax=ax)
    line_colors.append(
        {
            "plot": "boxplot",
            "palette": ["#ff7f0e", "#1f77b4"],
            "line": _rgb(ax.lines[0].get_color()),
        }
    )
    plt.close(fig)

    # Bar geometry and fill under saturation, gap, and dodge="auto"
    bars = []
    for label, kws in [
        ("hue", {"hue": "grp"}),
        ("hue_gap", {"hue": "grp", "gap": 0.2}),
        ("hue_is_x", {"hue": "cat"}),
        ("hue_is_x_gap", {"hue": "cat", "gap": 0.1}),
        ("saturation_1", {"hue": "grp", "saturation": 1}),
        ("plain", {}),
    ]:
        fig, ax = plt.subplots()
        sns.barplot(
            df,
            x="cat",
            y="val",
            order=["a", "b", "c"],
            errorbar=None,
            palette=["#ff7f0e", "#1f77b4", "#2ca02c"] if "hue" in kws else None,
            ax=ax,
            **kws,
        )
        patches = [p for p in ax.patches if p.get_height() > 0]
        legend = ax.get_legend()
        bars.append(
            {
                "case": label,
                "kws": kws,
                "x": [float(p.get_x()) for p in patches],
                "width": [float(p.get_width()) for p in patches],
                "facecolor": [_rgb(p.get_facecolor()) for p in patches],
                "legend": [t.get_text() for t in legend.texts] if legend else [],
            }
        )
        plt.close(fig)

    # Axis labels: text and visibility on a shared 2x2 grid, and a preset label
    def _labels(axs: Any) -> list[dict[str, Any]]:
        return [
            {
                "xlabel": ax.get_xlabel(),
                "xvisible": bool(ax.xaxis.label.get_visible()),
                "ylabel": ax.get_ylabel(),
                "yvisible": bool(ax.yaxis.label.get_visible()),
            }
            for ax in np.ravel(axs)
        ]

    label_calls = {
        "barplot": lambda ax: sns.barplot(df, x="cat", y="val", ax=ax, errorbar=None),
        "barplot_y": lambda ax: sns.barplot(df, x="val", y="cat", ax=ax, errorbar=None),
        "pointplot": lambda ax: sns.pointplot(
            df, x="cat", y="val", ax=ax, errorbar=None
        ),
        "boxplot": lambda ax: sns.boxplot(df, x="cat", y="val", ax=ax),
        "scatterplot": lambda ax: sns.scatterplot(df, x="val", y="val", ax=ax),
        "histplot": lambda ax: sns.histplot(df, x="val", ax=ax),
        "regplot": lambda ax: sns.regplot(df, x="val", y="val", ax=ax, ci=None),
    }
    labels = {}
    for name, call in label_calls.items():
        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
        for ax in np.ravel(axs):
            call(ax)
        grid = _labels(axs)
        plt.close(fig)
        fig, ax = plt.subplots()
        ax.set_xlabel("preset x")
        call(ax)
        labels[name] = {"grid": grid, "preset": _labels([ax])[0]}
        plt.close(fig)

    # Bar error bar caps: extent along the category axis, and the line style
    caps = []
    for label, kws in [
        ("plain", {}),
        ("plain_width", {"width": 0.5}),
        ("hue", {"hue": "grp"}),
        ("hue_width", {"hue": "grp", "width": 0.5}),
        ("hue_is_x", {"hue": "cat"}),
    ]:
        fig, ax = plt.subplots()
        sns.barplot(
            df,
            x="cat",
            y="val",
            order=["a", "b", "c"],
            capsize=0.1,
            seed=0,
            n_boot=50,
            ax=ax,
            **kws,
        )
        line = ax.lines[0]
        xs = np.asarray(line.get_xdata(), float)
        caps.append(
            {
                "case": label,
                "kws": kws,
                "cap_width": float(xs[1] - xs[0]),
                "capstyle": str(line.get_solid_capstyle()),
                "linewidth": float(line.get_linewidth()),
            }
        )
        plt.close(fig)

    # Point plot line width, marker size, and marker edge width
    points = []
    for rc_lw, kws in [(1.5, {}), (2.0, {}), (1.5, {"linewidth": 1.0})]:
        with plt.rc_context({"lines.linewidth": rc_lw}):
            fig, ax = plt.subplots()
            sns.pointplot(df, x="cat", y="val", seed=0, n_boot=50, ax=ax, **kws)
            err = next(
                line
                for line in ax.lines
                if np.isnan(np.asarray(line.get_xdata(), float)).any()
                or len(line.get_xdata()) == 2
            )
            mark = next(
                line for line in ax.lines if line.get_marker() not in ("None", None, "")
            )
            points.append(
                {
                    "rc_linewidth": rc_lw,
                    "kws": kws,
                    "linewidth": float(mark.get_linewidth()),
                    "markersize": float(mark.get_markersize()),
                    "markeredgewidth": float(mark.get_markeredgewidth()),
                    "err_linewidth": float(err.get_linewidth()),
                    "err_capstyle": str(err.get_solid_capstyle()),
                }
            )
            plt.close(fig)

    return {
        "versions": _versions(),
        "colors": PARITY_COLORS,
        "desaturate": desat,
        "palettes": palettes,
        "themes": themes,
        "scatter": scatter,
        "hist": hist,
        "hist_layers": hist_layers,
        "line_colors": line_colors,
        "bars": bars,
        "labels": labels,
        "caps": caps,
        "points": points,
    }


def main() -> None:
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "statsmodels_reference.json": record_statsmodels(),
        "letter_values_reference.json": record_letter_values(),
        "beeswarm_reference.json": record_beeswarm(),
        "palettes_reference.json": record_palettes_and_styles(),
        "parity_reference.json": record_parity(),
    }
    for name, payload in outputs.items():
        (out_dir / name).write_text(json.dumps(payload, indent=1) + "\n")
        print(f"wrote {out_dir / name}")


if __name__ == "__main__":
    main()
