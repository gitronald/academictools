---
id: 0
slug: drop-pandas-req
status: draft
branch:
created: 2026-11-15T14:40:34-08:00
concluded:
pr:
---

# Drop pandas from the package requirements

## Plan

### Goal

A base install of `academictools` requires no pandas, and `import tools`
(or any subpackage) loads no pandas. Code that genuinely needs pandas,
through seaborn or statsmodels, moves behind optional extras and is imported
only when called.

### Where pandas comes in at the start (v0.8.3)

- **Direct use:** only `tools/dataframes/display.py`. `display_tables_text`
  converts each polars frame with `.to_pandas()` and prints `to_string()`;
  `display_tables_html` does the same with `to_html()`, and puts a table's
  title in a `pd.MultiIndex` column header. The module also exports
  `pd_options`, a dict of pandas display settings.
- **Through seaborn** (which requires pandas): `smart_plot_data` (it calls
  `.to_pandas()`, since seaborn reads polars only through the dataframe
  interchange protocol, which polars 2 removed), `enhanced_scatterplot`,
  `enhanced_lineplot`, and `enhanced_boxplot` in `tools/plots/core.py`, and
  `plot_heatmap` in `tools/plots/heatmaps.py`.
- **Through statsmodels** (which requires pandas): `mcnemar_test` in
  `tools/stats/nonparametric.py` and `z_test` in `tools/stats/parametric.py`.
- **At import:** `tools/__init__.py` and `tools/functions.py` import every
  subpackage, and `tools/plots/core.py`, `heatmaps.py`, and both stats
  modules import seaborn or statsmodels at module level. So any
  `import tools` loads pandas, seaborn, and statsmodels, even when the caller
  uses only the IO or axis helpers.

### Steps

1. **Render `display_tables` without pandas.** Text: print the polars frame
   itself, under a `pl.Config` context for width and row limits. HTML: use
   the polars frame's own HTML rendering, with the title as a caption or
   heading row in place of the pandas `MultiIndex` header. Update the
   display tests in `tests/test_dataframes.py` to the new output.
2. **Remove `pd_options`.** Drop it from `tools/dataframes/display.py`, the
   `tools.dataframes` exports, and `__all__`. This is a breaking change for
   any caller that applies it, so it goes in the changelog under Removed.
3. **Move the seaborn functions behind an extra.** Add a `seaborn` extra
   (`seaborn`, and pyarrow if `.to_pandas()` still needs it). Import seaborn
   inside each function, and raise an `ImportError` naming the extra when it
   is missing. `smart_plot_data` and its test in `tests/test_plots.py` go
   with them.
4. **Settle statsmodels** (question 1): either reimplement `mcnemar_test`
   and `z_test` on scipy, which is already required, or move them behind a
   `stats` extra with a lazy import as in step 3. A scipy reimplementation
   must match statsmodels' results, including McNemar's exact binomial
   default; record the statsmodels outputs as test fixtures before removing
   it.
5. **Make the package imports lazy where needed.** `tools/__init__.py` and
   `tools/functions.py` keep importing every subpackage. That is safe only
   once no subpackage imports an optional dependency at module level, which
   steps 3 and 4 ensure.
6. **Dependencies.** Remove `pandas` from `dependencies`, and move seaborn
   (and statsmodels, if it stays) to their extras. pyarrow is not imported
   directly; keep it in the base requirements only if something other than
   `.to_pandas()` needs it. The dev group installs every extra, so the full
   test suite still covers the optional code.
7. **Guards.** Add a test that imports `tools` and each subpackage in a
   fresh subprocess and asserts that `pandas`, `seaborn`, and `statsmodels`
   are not in `sys.modules`. Check a base install with
   `uv tree --invert --package pandas` (no `academictools` row without
   extras), and run the suite once in an environment without the extras to
   confirm the core passes and the optional functions raise the named
   `ImportError`.
8. **Docs and release.** Document the extras in the README's install
   section, fill `[Unreleased]` in `CHANGELOG.md` (Removed: `pd_options`;
   Changed: seaborn and statsmodels functions need extras), and release as a
   minor version with stanza, since the removal breaks callers.

### Out of scope

- Moving seaborn-based plotting to plain matplotlib.
- New helpers beyond what the steps need.

## Handoff

State: drafted 2026-11-15; nothing implemented.

Open questions:

1. statsmodels: reimplement `mcnemar_test` and `z_test` on scipy, so the
   dependency goes entirely, or keep them behind a `stats` extra?
2. Extra names: `seaborn` and `stats`, or names by purpose such as `plots`?
   `tools.plots` holds matplotlib-only helpers too, so a `plots` extra would
   suggest the whole subpackage needs it.
