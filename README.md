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
- `tools.plots` - Matplotlib/seaborn plotting with academic paper styling (6.52" width)
- `tools.stats` - Statistical tests (parametric and nonparametric)
- `tools.utils` - Printing, timing, text processing, Jupyter utilities
- `tools.functions` - Convenience module re-exporting all subpackage functions

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
