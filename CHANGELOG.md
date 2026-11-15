# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
