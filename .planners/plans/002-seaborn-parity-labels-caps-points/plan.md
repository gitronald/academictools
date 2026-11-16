---
id: 2
slug: seaborn-parity-labels-caps-points
status: active
branch: feature/002-seaborn-parity-labels-caps-points
created: 2026-11-15T20:20:40-08:00
concluded:
pr:
---

# Match seaborn's axis labels, error bar caps, and point plot sizes

## Plan

### Goal

Plan 001 closed the parity gaps that a side-by-side comparison on synthetic
data found. Rendering real multi-panel figures both ways finds three more:
gaps that show only with shared axes, several hue levels, or seaborn's own
size defaults. Close each one, test it against values recorded from seaborn
0.13.2, and release a patch version.

### Gaps

1. **Axis labels.** Every helper calls `ax.set_xlabel` and `ax.set_ylabel`
   unconditionally. seaborn sets a label only when the axis has none, and
   makes it visible only when that axis shows tick labels. On axes made with
   `plt.subplots(sharex=True)` or `sharey=True`, matplotlib hides the inner
   tick labels, so seaborn's inner axis labels stay hidden; the helpers show
   the label on every panel. A label set before the call is kept by seaborn
   and overwritten by the helpers. A `scatterplot` of bare arrays gets no
   label from seaborn.
2. **Error bar caps.**
   - In `barplot`, seaborn draws a cap `capsize / n` category widths wide,
     where `n` is the number of dodged hue levels (1 without dodging),
     whatever the bar `width`. The helper draws `capsize * bar_width`: 20
     percent narrower at the default width, and narrower still at a smaller
     `width`.
   - seaborn's error lines take matplotlib's default cap style
     (`lines.solid_capstyle`, projecting), so a cap reaches half a line
     width past each end. The helpers draw them with `solid_capstyle="butt"`.
     With a thin `err_kws` line width and several hue levels, the helper's
     caps barely show past the error line.
3. **`pointplot` sizes.** seaborn's defaults are a line width of 1.8 times
   `lines.linewidth` (for the line and the error bars), a marker size of that
   width times sqrt(2 pi), and a marker edge width of 0.75 times that width.
   The helper uses `lines.linewidth` and `lines.markersize`, so its lines are
   thinner and its markers smaller.

### Approach

- Record seaborn's values first, as plans 000 and 001 did: extend
  `tests/fixtures/record_reference.py` (label text and visibility on a 2x2
  shared grid, cap extents for bars with and without hue at two widths, the
  error line cap style, and pointplot's line width, marker size, and marker
  edge width), re-run it in the throwaway environment its docstring names,
  and commit the fixture. seaborn stays out of the dev group.
- One shared label helper in `_common.py`, used by the categorical helpers,
  `scatterplot`, `regplot`, `histplot`, and `lineplot` (in `core.py`), so the rule lives in
  one place.
- Each gap gets tests on the drawn artists against the recorded values.

### Order

1. Record the fixtures.
2. Gap 1 (labels), with tests.
3. Gaps 2 and 3 (`charts.py`), with tests.
4. README and CHANGELOG `[Unreleased]`; ruff, pyrefly, and the suite; side-by-
   side renders of a shared-axis grid; a patch release with stanza.

### Out of scope

- `heatmap` and calendar heatmap labels, which follow their own rules.
- Bootstrap intervals, which differ from seaborn's by design.

## Log

- **2026-11-15T20:21:10-08:00** — Activated.
  - Branch: `feature/002-seaborn-parity-labels-caps-points`
  - Base: `dev` at `5ff3b81`
  - Worktree: `.worktrees/002-seaborn-parity-labels-caps-points`
  - PR: pending
