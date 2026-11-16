# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- `save_fig` writes PDFs without a `/CreationDate`, so saving the same figure twice writes the same bytes and a rerun that changes nothing leaves committed PDFs unchanged.

## [0.9.3] - 2026-11-16

### Fixed

- `spearmanr` with `nan_policy="omit"` drops every row where either column is missing (a null or NaN) and ranks the rest with scipy's regular path, instead of scipy's masked-array path, which took minutes on large columns with many tied values. Results are unchanged, except that a constant column with missing values returns `nan` rather than `(0.0, 1.0)`, as a constant column without them already did.

## [0.9.2] - 2026-11-15

### Changed

- The plotting helpers label the axes as seaborn does: an axis that already has a label keeps it, and a new label is visible only when the axis shows tick labels, so the inner panels of a grid with shared axes carry no visible label. `regplot` still sets both labels, as seaborn's does.
- `barplot` draws error bar caps `capsize / n` category widths wide, where `n` is the number of dodged hue levels, whatever the bar `width`.
- Error bar lines take matplotlib's default cap style (projecting), as seaborn's do, so caps reach past the error line.
- `pointplot` defaults to seaborn's sizes: lines and error bars 1.8 times `lines.linewidth`, markers that width times sqrt(2 pi), with an edge 0.75 times that width. `lw=` and `ms=` count as `linewidth` and `markersize`.

## [0.9.1] - 2026-11-15

### Added

- `set_theme(style)`, which sets the rcParams `seaborn.set_theme` does: the `ticks` or `whitegrid` axes style, the "notebook" context (`NOTEBOOK_CONTEXT`), and the "deep" color cycle (`DEEP`).
- `"dark:<color>"` and `"light:<color>"` palette names, with `dark_palette`, `light_palette`, and `blend_palette`; `desaturate`.
- `barplot`, `boxplot`, and `boxenplot` take `saturation` (default 0.75, as seaborn draws fills) and `gap`.
- `boxplot(fill=False)` draws unfilled boxes with every line in the box color.
- `boxenplot` takes `box_kws`, `line_kws`, and `flier_kws` for the boxes, the median line, and the outliers.

### Changed

- With `hue`, the categorical helpers, `scatterplot`, and `histplot` add one labeled artist per level to the axes and build the legend from them, so `ax.get_legend_handles_labels()` and `add_legend` find the levels.
- `dodge` and `legend` default to `"auto"` on the categorical helpers: hue levels are dodged only when they share a category, and the legend is left out when `hue` is the category column.
- `boxplot` and `boxenplot` default to `linecolor="auto"`, a gray darker than the darkest fill.
- `scatterplot` gives filled markers seaborn's white edge, `0.08 * sqrt(s)` points wide, and with `hue` draws all points as one collection in row order.
- `histplot` draws hue layers with the first level on top, applies `alpha` to the fill only, and thins bar edges for narrow bins.
- `swarmplot` draws its points at `zorder=3`, above bars and error bars.

## [0.9.0] - 2026-11-15

### Added

- Plotting helpers on matplotlib and polars, replacing seaborn's common plot types: `barplot` and `pointplot` (estimate per category with a seeded bootstrap confidence interval), `regplot` (OLS line with a bootstrap band), `scatterplot`, `histplot`, `boxplot`, `heatmap`, `swarmplot`, and `boxenplot`. Each takes a polars DataFrame and column names, draws on a given `Axes`, and supports `hue` with an order and a palette where the plot has groups.
- `swarmplot` and `boxenplot` port seaborn's beeswarm and letter-value algorithms, so point placement and box depths match seaborn's.
- `color_palette`, which samples a matplotlib colormap as seaborn's `color_palette(name, n)` does; `resolve_colors`; and `TICKS_STYLE` and `WHITEGRID_STYLE`, rcParams equivalents of seaborn's `ticks` and `whitegrid` styles.
- `bootstrap` and `estimate_interval`, the shared estimate and interval helpers behind the plots.
- `mcnemar_test` takes `exact` and `correction`, for the chi-square form of the test.

### Changed

- `enhanced_scatterplot`, `enhanced_lineplot`, `enhanced_boxplot`, and `plot_heatmap` draw with the new helpers instead of seaborn. Their keyword arguments now go to those helpers rather than to seaborn.
- `mcnemar_test` and `z_test` compute on scipy instead of statsmodels, with the same results.
- `display_tables` renders tables with polars instead of pandas: text output is the polars table, and in HTML a table's title is a heading above it instead of a column header.
- `estimate_interval`, and the plots built on it, raise a `ValueError` for an `"se"` or `"sd"` error bar with an estimator other than the mean, since those intervals spread around the mean.
- `regplot` raises a `ValueError` naming the columns when fewer than two rows have both values present.

### Removed

- `pd_options`, the pandas display settings dict.
- `smart_plot_data`, which only converted frames to pandas for seaborn.
- The pandas, seaborn, and statsmodels requirements. Importing `tools` no longer loads any of them.

### Fixed

- `tools.__version__` reports the installed package version; it had read 0.8.2 since the initial commit.

## [0.8.3] - 2026-11-13

### Changed

- Maintenance: upgraded the repo tooling to repotemplates 0.15.0, which renames the template stamp, updates the Dependabot setup link, and adds a Python matrix to the test workflow. No package code changes.

[Unreleased]: https://github.com/gitronald/academictools/commits/main
