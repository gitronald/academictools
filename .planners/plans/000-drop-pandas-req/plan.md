---
id: 0
slug: drop-pandas-req
status: implemented
branch: feature/000-drop-pandas-req
created: 2026-11-15T14:40:34-08:00
concluded:
pr: https://github.com/gitronald/academictools/pull/5
---

# Drop pandas from the package requirements

## Plan

### Goal

A base install of `academictools` requires no pandas, and `import tools`
(or any subpackage) loads no pandas. seaborn and statsmodels, the two
requirements that bring pandas in, go too: their functions are
reimplemented on matplotlib, numpy, scipy, and polars, so the package's whole
dependency closure is free of pandas with no extras. The plotting subpackage
gains the polars-and-matplotlib helpers that replace seaborn's common plot
types for callers.

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
3. **Record seaborn and statsmodels reference outputs.** Before either
   dependency goes, record what the replacements must match, as test
   fixtures: statsmodels' `mcnemar` and `proportions_ztest` results on a set
   of inputs (McNemar's exact binomial default included), seaborn's beeswarm
   point offsets, and seaborn's letter-value boxes for each `k_depth` rule,
   `"tukey"` included. Record them once from a throwaway environment
   (`uv run --with seaborn --with statsmodels ...`), so the dev group never
   carries either package.
4. **Reimplement the statsmodels functions on scipy.** `mcnemar_test` (exact
   binomial test via `scipy.stats.binomtest`, and the chi-square form) and
   `z_test` (two-proportion z-test via `scipy.stats.norm`), each matching the
   step 3 fixtures.
5. **Plotting helpers on matplotlib and polars.** Each takes a polars frame
   and column names, draws on a given `Axes`, and supports `hue` with an
   order and a palette where the plot type has groups:
   - grouped bar plot of an estimator (mean by default) with a bootstrap
     confidence interval (`errorbar=("ci", 95)`, `n_boot`, a seed,
     `capsize`), and a point plot drawing the same statistic as markers
     joined by lines, with dodge;
   - a shared bootstrap helper behind both, and a regression plot (scatter,
     OLS line, bootstrap confidence band);
   - grouped scatter, histogram, and box plot;
   - heatmap with a colormap, a colorbar, and formatted cell annotations;
   - beeswarm (swarm) plot and letter-value (boxen) plot, **ported from
     seaborn's `Beeswarm` and `LetterValues` code so that point placement and
     box depths match seaborn exactly** (step 3 fixtures). seaborn is
     BSD-3-licensed: keep its copyright and license notice in the ported
     module;
   - a palette helper that samples a matplotlib colormap the way seaborn's
     `color_palette(name, n)` does, and rcParams equivalents of seaborn's
     `ticks` and `whitegrid` styles.

   Bootstrap intervals will not match seaborn's draws, since the
   resampling differs; the helpers are seeded so their own output is
   reproducible.
6. **Rewrite the existing seaborn functions on the step 5 helpers.**
   `enhanced_scatterplot`, `enhanced_lineplot`, `enhanced_boxplot`, and
   `plot_heatmap`. `smart_plot_data` exists only to hand seaborn a pandas
   frame, so it is removed, with its test in `tests/test_plots.py`
   (changelog: Removed).
7. **Dependencies.** Remove `pandas`, `seaborn`, and `statsmodels` from
   `dependencies`. pyarrow is not imported directly; remove it too unless
   something other than `.to_pandas()` needs it. `tools/__init__.py` and
   `tools/functions.py` keep importing every subpackage, which is safe once
   none of them imports a removed package.
8. **Guards.** Add a test that imports `tools` and each subpackage in a
   fresh subprocess and asserts that `pandas`, `seaborn`, and `statsmodels`
   are not in `sys.modules`, and check `uv tree --invert --package pandas`
   shows no `academictools` row.
9. **Docs and release.** Update the README for the new plotting helpers,
   fill `[Unreleased]` in `CHANGELOG.md` (Removed: `pd_options`,
   `smart_plot_data`, the pandas, seaborn, and statsmodels requirements;
   Added: the plotting helpers; Changed: the rewritten functions), and
   release as a minor version with stanza, since the removals break
   callers.

### Out of scope

- seaborn plot types beyond those in step 5 (violin, KDE, pair and joint
  grids, the objects interface).
- New helpers beyond what the steps need.

## Log

- **2026-11-15T14:50:01-08:00** — Revised the steps: seaborn and statsmodels are reimplemented (matplotlib, scipy) rather than moved behind extras, and the plotting helpers that replace seaborn's plot types are added here.
  - The beeswarm and letter-value plots port seaborn's algorithms so layouts match exactly; reference outputs are recorded as fixtures first.
- **2026-11-15T14:50:39-08:00** — Activated.
  - Branch: `feature/000-drop-pandas-req`
  - Base: `dev` at `48e6aa9`
  - Worktree: `.worktrees/000-drop-pandas-req`
  - PR: pending
- **2026-11-15T14:53:10-08:00** — PR opened: https://github.com/gitronald/academictools/pull/5
- **2026-11-15T15:05:57-08:00** — Implemented steps 1-9 on the feature branch; final checks at 184464c (base dev unchanged): ruff and pyrefly strict clean, 454 tests pass at 98% coverage, `uv build` succeeds.
  - Step 3: fixtures recorded once from a throwaway `uv run --no-project --with` environment (seaborn 0.13.2, statsmodels 0.15.0, matplotlib 3.11.2, numpy 2.5.4, scipy 1.18.1) by `tests/fixtures/record_reference.py`; the dev group carries neither package. Beeswarm placement depends on the axes transform, so it is recorded both at algorithm level (point space) and as full swarm plots on a fixed 4x3 in, 100 dpi figure.
  - Step 4: `_mcnemar` (scipy `binomtest` / `chi2`) and `z_test` (pooled variance, `norm`) match every recorded statsmodels case; `mcnemar_test` gained `exact` and `correction`.
  - Step 5: new modules `tools/plots/charts.py`, `palettes.py`, `estimate.py`, and `_seaborn_port.py` (keeps seaborn's BSD notice). Swarm offsets match seaborn exactly (zero difference) in both orientations; letter values and boxen widths match for every `k_depth` and `width_method`. Helper keywords follow the existing presets in `settings.py`, so `box_kws`, `pointplot_kws`, and `heatmap_kws_proportion` work with the helpers; `strip_kws` stays a seaborn stripplot preset, since strip plots are out of scope.
  - Step 5 deviations: seaborn's `ticks`/`whitegrid` styles set `image.cmap` to its own `rocket` map, which matplotlib does not ship, so the rcParams copies leave that key out. Boxen fill colors use a plain RGB tint ramp rather than seaborn's HUSL light palette, so box colors differ slightly while depths match. seaborn's `color_palette` cycles a qualitative map past its size, which the port does too.
  - Step 7: pyarrow stays. duckdb's `relation.pl()` in `tools/io/sql.py` converts through pyarrow (12 SQL tests failed with it removed), which is the plan's "unless something else needs it" case; a comment in `pyproject.toml` says why. Re-locking moved pyarrow 26.0.0 to 26.0.1 within its pin. `uv.lock` has no pandas, seaborn, or statsmodels entries, and `uv tree --invert --package pandas` lists nothing.
  - Step 8: `tests/test_imports.py` imports each module in a fresh interpreter and resolves every `__all__` name; the probe was checked to report `pandas` when pandas is loaded.
  - Step 9: README, CHANGELOG `[Unreleased]`, and `.claude/CLAUDE.md` updated. The minor release is not cut: it follows the merge, from dev.
- **2026-11-15T15:06:09-08:00** — Implemented: 11 commits ahead of `dev`.

## Handoff

State: steps 1-9 implemented and pushed on `feature/000-drop-pandas-req`
(PR #5); nothing uncommitted or unpushed.

Open questions: none.

Settled (2026-11-15):

1. statsmodels is reimplemented on scipy, not kept behind an extra (step 4).
2. No extras: seaborn is replaced by matplotlib helpers rather than made
   optional (steps 5-6), so the question of extra names is moot.
3. The beeswarm and letter-value plots keep seaborn's layouts exactly,
   through a port of seaborn's algorithms, not a substitute plot type.
4. pyarrow stays a requirement: duckdb's `relation.pl()` needs it (step 7).

Not verified: the helpers' look against seaborn beyond the fixtures (a
rendered gallery was checked by eye only); boxen fill colors are a plain tint
ramp, not seaborn's HUSL palette.

Next: review of PR #5 (waits on the owner), then `/planners close`, then the
minor release with stanza from dev (step 9's release), whose changelog
`[Unreleased]` is already filled.
