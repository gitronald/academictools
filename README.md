# academictools

Tools for academic data analysis: loading and saving data, summarizing dataframes, running statistical tests, and making paper-ready figures.

The distribution is `academictools`; the import package is `tools`:

```python
from tools import io, dataframes, plots, stats
```

Requires Python 3.14+.

## Quick Start

```bash
uv sync
uv run pytest -v --cov=tools --cov-report=term-missing
```

## Install

```bash
uv add git+https://github.com/gitronald/academictools.git
```

For development:
```bash
git clone https://github.com/gitronald/academictools.git
cd academictools
uv sync
```

## Modules

- `tools.io` - Data loading/saving (CSV, TSV, Parquet, Feather, JSONL), file utilities, DuckDB queries
- `tools.dataframes` - Frequency tables, descriptive stats, inspection, reshaping, dates, sessions, display
- `tools.plots` - Matplotlib plotting with academic paper styling (6.52" width): seaborn-style helpers for polars frames, palettes, and axis formatting
- `tools.stats` - Statistical tests (parametric and nonparametric)
- `tools.utils` - Printing, timing, text processing, Jupyter utilities
- `tools.functions` - Convenience module re-exporting all subpackage functions

## Plotting

`tools.plots` draws seaborn's common plot types with matplotlib, straight from
polars frames; seaborn and pandas are not dependencies. Each helper takes a
frame and column names, draws on an `Axes` (the current one by default), and
returns it:

```python
import matplotlib.pyplot as plt
from tools import plots

fig, ax = plt.subplots()
plots.barplot(df, "condition", "score", hue="group", seed=0, ax=ax)
```

| Helper | Draws |
|---|---|
| `barplot`, `pointplot` | An estimate per category (mean by default) with a bootstrap confidence interval |
| `regplot` | Scatter with an OLS line and a bootstrap confidence band |
| `scatterplot`, `histplot`, `boxplot` | Grouped scatter, histogram, and box plots |
| `heatmap` | A matrix with a colormap, a colorbar, and formatted cell annotations |
| `swarmplot`, `boxenplot` | Beeswarm and letter-value plots, ported from seaborn so layouts match |

The helpers apply seaborn's defaults, so a figure redrawn from seaborn with
the same arguments looks the same: fills at 0.75 saturation, white scatter
marker edges, hue levels dodged only when they share a category, and a hue
legend built from labeled artists, so `add_legend` can restyle it. Bootstrap
intervals are seeded (`seed=`) so they are reproducible; they will not match
seaborn's draws.

`set_theme("ticks")` (or `"whitegrid"`) sets the rcParams that
`seaborn.set_theme` does: the axes style, the "notebook" line and font sizes,
and the "deep" color cycle. `color_palette(name, n)` samples a matplotlib
colormap as seaborn does and also takes `"dark:<color>"` and
`"light:<color>"`. `TICKS_STYLE` and `WHITEGRID_STYLE` hold the axes styles
alone.

## Versioning

Managed with [stanza](https://github.com/gitronald/stanza):
```bash
stanza release patch      # 0.0.2a1 -> 0.0.3
stanza release minor      # 0.0.2a1 -> 0.1.0
stanza release major      # 0.0.2a1 -> 1.0.0
stanza release prerelease # 0.0.2a1 -> 0.0.2a2
```

## CI/CD

Tests run on pushes to `dev` and `main` via GitHub Actions on Python 3.14 with coverage reporting.
