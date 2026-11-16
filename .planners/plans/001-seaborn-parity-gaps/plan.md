---
id: 1
slug: seaborn-parity-gaps
status: active
branch: feature/001-seaborn-parity-gaps
created: 2026-11-15T18:35:50-08:00
concluded:
pr: https://github.com/gitronald/academictools/pull/8
---

# Close seaborn parity gaps in the plotting helpers

## Plan

### Goal

A figure drawn with seaborn 0.13.2 and redrawn with the `tools.plots`
helpers, given the same data and the same keyword arguments, should look the
same. Plan 000 ported the helpers and their algorithms. A side-by-side
comparison of the two libraries on synthetic data, using the arguments that
real figures pass, finds the gaps below: defaults that seaborn applies and
the helpers do not, and arguments seaborn accepts that the helpers reject.
Close each one, test it, and release a patch version.

### Gaps

1. **Hue artists carry no labels.** With `hue`, `barplot`, `boxenplot`,
   `boxplot`, and `pointplot` pass proxy handles straight to `ax.legend`, so
   `ax.get_legend_handles_labels()` returns nothing, and this package's own
   `add_legend` draws an empty legend after any of them. seaborn labels one
   artist per hue level (a bar, a box patch, or a line). Label one drawn
   artist per level, and build the default legend from those artists.
2. **No `saturation`.** seaborn draws the fills of bars, boxes, and letter-value
   boxes at 0.75 of the palette's saturation (in HLS), legend handles
   included, and at full saturation when `fill=False`. Add `saturation=0.75`
   to `barplot`, `boxplot`, and `boxenplot`.
3. **Scatter marker edges.** seaborn's `scatterplot` defaults to a white edge
   of width `0.08 * sqrt(s)` (`s` at its 10th percentile when sizes vary).
   The helper inherits matplotlib's edge, the face color at
   `lines.linewidth`, so small markers draw visibly larger. Match seaborn's
   defaults; explicit `edgecolor` and `linewidth` still win.
4. **Histogram layer order.** With `multiple="layer"`, seaborn draws the hue
   levels in reverse, so the first level sits on top. The helper draws the
   last on top.
5. **Histogram edge width.** seaborn thins bar edges for narrow bins; the
   helper uses a fixed 0.5. Port seaborn's rule.
6. **`barplot(gap=...)`.** seaborn shrinks each dodged bar by `gap` about its
   center. Add it, also to `boxplot` and `boxenplot`, where seaborn has it too.
7. **`dodge="auto"`.** seaborn dodges hue levels only when `hue` differs from
   the category column. Coloring each category by itself (`hue` equal to `x`)
   then draws full-width bars; the helpers split each bar into as many slivers
   as there are categories. Make `"auto"` the default for `barplot`,
   `boxplot`, and `boxenplot`.
8. **`dark:` and `light:` palettes.** seaborn accepts `"dark:<color>"` and
   `"light:<color>"` as palette names (a blend from a near-black, or a
   near-white, of the color's hue up to the color). `resolve_colors` and
   `color_palette` reject them.
9. **`boxenplot(box_kws=, line_kws=, flier_kws=)`.** seaborn styles the
   boxes, the median line, and the outlier points separately through these.
   The helper fixes the median to the edge color and the outliers to hollow
   gray markers of size 25.
10. **`boxplot(fill=False)`.** seaborn draws an unfilled box with every line
    in the box color. The helper always fills.
11. **Theme.** `TICKS_STYLE` and `WHITEGRID_STYLE` carry seaborn's axes style
    but not the rest of `seaborn.set_theme`: the `"notebook"` context (axes,
    grid, and tick line widths, tick sizes, font sizes) and the `"deep"` color
    cycle. Add `NOTEBOOK_CONTEXT`, a `DEEP` palette, and
    `set_theme(style="ticks")`, which applies all three.

### Approach

- Record seaborn's values first, as plan 000 did: extend
  `tests/fixtures/record_reference.py` with each value a gap needs (the
  desaturated colors, the `dark:`/`light:` blends, the scatter edge widths,
  the histogram edge widths, the rcParams `set_theme` sets), run it once in
  the throwaway environment its docstring names, and commit the fixture.
  seaborn stays out of the dev group.
- Gaps without a number to record (labels, layer order, `gap` widths, `dodge`,
  `fill`, the boxen keyword routing) get behavior tests on the drawn artists.
- Each gap's tests include a hue column with a missing value and an explicit
  `order`, the cases plan 000's review found missing.
- Shared logic (desaturation, the `"auto"` dodge rule, labeling) goes in
  `_common.py`, not in each chart.

### Order

1. Record the fixtures.
2. Gaps 1, 2, 6, 7, 9, and 10 in `charts.py` (the categorical helpers), each
   with its tests.
3. Gaps 3, 4, and 5 (scatter and histogram).
4. Gaps 8 and 11 in `palettes.py`.
5. README and CHANGELOG `[Unreleased]`; ruff, pyrefly, and the suite; a
   patch release with stanza.

### Out of scope

- seaborn features no gap above names (`stat` options, kde, `jitter`, and so
  on).
- Any plotting not in seaborn.

## Log

- **2026-11-15T18:37:18-08:00** — Activated.
  - Branch: `feature/001-seaborn-parity-gaps`
  - Base: `dev` at `3d2ba38`
  - Worktree: `.worktrees/001-seaborn-parity-gaps`
  - PR: pending
- **2026-11-15T18:51:48-08:00** — PR opened: https://github.com/gitronald/academictools/pull/8
