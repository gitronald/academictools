# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development

```bash
uv sync            # install dependencies
uv run pytest      # run tests
uv build           # build package

uv run ruff check                # lint
uv run ruff format               # format
uv run pyrefly check             # type check (strict preset)
pre-commit install               # enable git hooks (ruff + pyrefly)
```

Lint, format, and type checks run via pre-commit hooks and in CI (Python 3.14).
Pyrefly runs the `strict` preset over the whole repo; `tests/**` relaxes
`implicit-any` so unannotated fixture params don't force annotations.

## Package Structure

- `tools/functions.py` - Convenience module re-exporting all subpackage functions
- `tools/io/` - Data loading/saving (`data.py`, which also defines the `FilePath`/`JoinStrategy` aliases), file utilities (`files.py`), DuckDB queries (`sql.py`)
- `tools/dataframes/` - Frequency tables (`dft.py`), descriptive stats (`describe.py`), inspection, reshaping, dates, sessions, display
- `tools/plots/` - Settings, plotting helpers (`charts.py`), palettes, styles, and `set_theme` (`palettes.py`), HUSL conversions for the `dark:`/`light:` palettes (`_husl.py`, keeps its MIT notice), bootstrap estimates (`estimate.py`), seaborn's beeswarm and letter values (`_seaborn_port.py`, keeps seaborn's BSD notice), shared data and axes helpers (`_common.py`), core plots (`core.py`), axes helpers (`axes.py`), heatmaps (`heatmaps.py`)
- `tools/stats/` - Parametric (`parametric.py`) and nonparametric (`nonparametric.py`) tests
- `tools/utils/` - Printing, timing, text processing, Jupyter utilities

## Design

- Polars-first: IO and helpers work in polars end-to-end
- No pandas: neither the package nor its dependencies load pandas (`tests/test_imports.py` guards it); seaborn and statsmodels are reimplemented, checked against fixtures recorded by `tests/fixtures/record_reference.py`
- Versioning managed with [stanza](https://github.com/gitronald/stanza)