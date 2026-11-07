"""Plot settings: page dimensions, font sizes, and style presets."""

# Plot Sizes - Standard Page
PAGE_WIDTH = 6.52  # Set page width (inches)
WIDE_FIG_WIDTH = PAGE_WIDTH * 0.75  # 75% of page width
NARROW_FIG_WIDTH = PAGE_WIDTH - WIDE_FIG_WIDTH  # 25% of page width

# Plot fonts
SMALL_SIZE = 7
MEDIUM_SIZE = 8
LARGE_SIZE = 9

# Set default font family and sizes
RC_PARAMS = {
    "font.family": "Arial",
    "font.weight": "regular",
    "font.size": SMALL_SIZE,  # fontsize default
    "figure.titlesize": MEDIUM_SIZE,  # fontsize - figure title
    "axes.titlesize": MEDIUM_SIZE,  # fontsize - axes title
    "axes.labelsize": SMALL_SIZE,  # fontsize - x and y labels
    "xtick.labelsize": SMALL_SIZE,  # fontsize - xtick labels
    "ytick.labelsize": SMALL_SIZE,  # fontsize - ytick labels
    "legend.fontsize": SMALL_SIZE,  # fontsize - legend
    "legend.title_fontsize": MEDIUM_SIZE,  # fontsize - legend title
}


# Style Keyword Defaults

# usage: sns.stripplot(x="day", y="total_bill", data=tips, **strip_kws)
strip_kws = dict(jitter=True, size=1.5, alpha=0.7)

# usage: sns.boxplot(x="day", y="total_bill", data=tips, **box_kws)
box_kws = dict(linewidth=0.75, color="whitesmoke", width=0.5, fliersize=0)

# usage: sns.pointplot(x="time", y="total_bill", hue="smoker", data=tips,
#         **pointplot_kws)
pointplot_kws = dict(
    dodge=0.2,
    alpha=0.8,
    markers=".",
    linestyles="-",
    markersize=3,
    linewidth=1.5,
    err_kws={"linewidth": 1.5},
    capsize=0.1,
)

# usage: sns.heatmap(matrix, **heatmap_kws)
heatmap_kws_proportion = dict(
    annot=True,
    annot_kws={"size": 8},
    fmt=".2f",
    cmap="bone_r",
    cbar=True,
    cbar_kws={
        "label": "",
        "orientation": "vertical",
        "pad": 0.01,
        "shrink": 1,
        "ticks": [0, 1],
        "format": "%.1f",
    },
    linewidths=0.1,
    linecolor="whitesmoke",
    vmax=1,
    vmin=0,
)
