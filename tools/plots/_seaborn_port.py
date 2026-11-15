"""Beeswarm placement and letter values, ported from seaborn 0.13.2.

``Beeswarm`` is seaborn's ``seaborn.categorical.Beeswarm`` and
``letter_values`` is seaborn's ``seaborn._statistics.LetterValues``, adapted to
plain numpy and matplotlib with type annotations. The algorithms are unchanged,
so swarm point placement and letter-value box depths match seaborn's exactly.

Copyright (c) 2012-2023, Michael L. Waskom
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the project nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import warnings
from collections.abc import Callable
from statistics import NormalDist
from typing import Any, Literal

import numpy as np
from matplotlib.collections import PathCollection

Orient = Literal["x", "y"]
KDepth = Literal["tukey", "proportion", "trustworthy", "full"] | int
Transform = Callable[[Any], Any]


def _get_transform_functions(ax: Any, axis: Orient) -> tuple[Transform, Transform]:
    """Return the forward and inverse transforms for a given axis."""
    axis_obj = ax.xaxis if axis == "x" else ax.yaxis
    transform = axis_obj.get_transform()
    return transform.transform, transform.inverted().transform


class Beeswarm:
    """Modifies a scatterplot artist to show a beeswarm plot."""

    def __init__(
        self, orient: Orient = "x", width: float = 0.8, warn_thresh: float = 0.05
    ) -> None:
        self.orient: Orient = orient
        self.width = width
        self.warn_thresh = warn_thresh

    def __call__(self, points: PathCollection, center: float) -> None:
        """Swarm `points`, a PathCollection, around the `center` position."""
        ax = points.axes
        if ax is None:
            raise ValueError("points must be drawn on an Axes")
        dpi = ax.figure.dpi

        # Get the original positions of the points
        orig_xy_data = np.asarray(points.get_offsets(), dtype=float).copy()

        # Reset the categorical positions to the center line
        cat_idx = 1 if self.orient == "y" else 0
        orig_xy_data[:, cat_idx] = center

        # Transform the data coordinates to point coordinates.
        # We'll figure out the swarm positions in the latter
        # and then convert back to data coordinates and replot
        orig_x_data, orig_y_data = orig_xy_data.T
        orig_xy = ax.transData.transform(orig_xy_data)

        # Order the variables so that x is the categorical axis
        if self.orient == "y":
            orig_xy = orig_xy[:, [1, 0]]

        # Add a column with each point's radius
        sizes = np.asarray(points.get_sizes(), dtype=float)
        if sizes.size == 1:
            sizes = np.repeat(sizes, orig_xy.shape[0])
        edge = float(np.asarray(points.get_linewidth()).item())
        radii = (np.sqrt(sizes) + edge) / 2 * (dpi / 72)
        orig_xy = np.c_[orig_xy, radii]

        # Sort along the value axis to facilitate the beeswarm
        sorter = np.argsort(orig_xy[:, 1])
        orig_xyr = orig_xy[sorter]

        # Adjust points along the categorical axis to prevent overlaps
        new_xyr = np.empty_like(orig_xyr)
        new_xyr[sorter] = self.beeswarm(orig_xyr)

        # Transform the point coordinates back to data coordinates
        if self.orient == "y":
            new_xy = new_xyr[:, [1, 0]]
        else:
            new_xy = new_xyr[:, :2]
        new_x_data, new_y_data = ax.transData.inverted().transform(new_xy).T

        # Add gutters
        t_fwd, t_inv = _get_transform_functions(ax, self.orient)
        if self.orient == "y":
            self.add_gutters(new_y_data, center, t_fwd, t_inv)
        else:
            self.add_gutters(new_x_data, center, t_fwd, t_inv)

        # Reposition the points so they do not overlap
        if self.orient == "y":
            points.set_offsets(np.c_[orig_x_data, new_y_data])
        else:
            points.set_offsets(np.c_[new_x_data, orig_y_data])

    def beeswarm(self, orig_xyr: np.ndarray) -> np.ndarray:
        """Adjust x position of points to avoid overlaps."""
        # In this method, `x` is always the categorical axis
        # Center of the swarm, in point coordinates
        midline = orig_xyr[0, 0]

        # Start the swarm with the first point
        swarm = np.atleast_2d(orig_xyr[0])

        # Loop over the remaining points
        for xyr_i in orig_xyr[1:]:
            # Find the points in the swarm that could possibly
            # overlap with the point we are currently placing
            neighbors = self.could_overlap(xyr_i, swarm)

            # Find positions that would be valid individually
            # with respect to each of the swarm neighbors
            candidates = self.position_candidates(xyr_i, neighbors)

            # Sort candidates by their centrality
            offsets = np.abs(candidates[:, 0] - midline)
            candidates = candidates[np.argsort(offsets)]

            # Find the first candidate that does not overlap any neighbors
            new_xyr_i = self.first_non_overlapping_candidate(candidates, neighbors)

            # Place it into the swarm
            swarm = np.vstack([swarm, new_xyr_i])

        return swarm

    def could_overlap(self, xyr_i: np.ndarray, swarm: np.ndarray) -> np.ndarray:
        """Return a list of all swarm points that could overlap with target."""
        # Because we work backwards through the swarm and can short-circuit,
        # the for-loop is faster than vectorization
        _, y_i, r_i = xyr_i
        neighbors = []
        for xyr_j in reversed(swarm):
            _, y_j, r_j = xyr_j
            if (y_i - y_j) < (r_i + r_j):
                neighbors.append(xyr_j)
            else:
                break
        return np.array(neighbors)[::-1]

    def position_candidates(
        self, xyr_i: np.ndarray, neighbors: np.ndarray
    ) -> np.ndarray:
        """Return a list of coordinates that might be valid by adjusting x."""
        candidates: list[Any] = [xyr_i]
        x_i, y_i, r_i = xyr_i
        left_first = True
        for x_j, y_j, r_j in neighbors:
            dy = y_i - y_j
            dx = np.sqrt(max((r_i + r_j) ** 2 - dy**2, 0)) * 1.05
            cl, cr = (x_j - dx, y_i, r_i), (x_j + dx, y_i, r_i)
            if left_first:
                new_candidates = [cl, cr]
            else:
                new_candidates = [cr, cl]
            candidates.extend(new_candidates)
            left_first = not left_first
        return np.array(candidates)

    def first_non_overlapping_candidate(
        self, candidates: np.ndarray, neighbors: np.ndarray
    ) -> np.ndarray:
        """Find the first candidate that does not overlap with the swarm."""
        # If we have no neighbors, all candidates are good.
        if len(neighbors) == 0:
            return candidates[0]

        neighbors_x = neighbors[:, 0]
        neighbors_y = neighbors[:, 1]
        neighbors_r = neighbors[:, 2]

        for xyr_i in candidates:
            x_i, y_i, r_i = xyr_i

            dx = neighbors_x - x_i
            dy = neighbors_y - y_i
            sq_distances = np.square(dx) + np.square(dy)

            sep_needed = np.square(neighbors_r + r_i)

            # Good candidate does not overlap any of neighbors which means that
            # squared distance between candidate and any of the neighbors has
            # to be at least square of the summed radii
            good_candidate = np.all(sq_distances >= sep_needed)

            if good_candidate:
                return xyr_i

        raise RuntimeError(
            "No non-overlapping candidates found. This should not happen."
        )

    def add_gutters(
        self,
        points: np.ndarray,
        center: float,
        trans_fwd: Transform,
        trans_inv: Transform,
    ) -> np.ndarray:
        """Stop points from extending beyond their territory."""
        half_width = self.width / 2
        low_gutter = trans_inv(trans_fwd(center) - half_width)
        off_low = points < low_gutter
        if off_low.any():
            points[off_low] = low_gutter
        high_gutter = trans_inv(trans_fwd(center) + half_width)
        off_high = points > high_gutter
        if off_high.any():
            points[off_high] = high_gutter

        gutter_prop = (off_high + off_low).sum() / len(points)
        if gutter_prop > self.warn_thresh:
            msg = (
                f"{gutter_prop:.1%} of the points cannot be placed; you may want "
                "to decrease the size of the markers or use stripplot."
            )
            warnings.warn(msg, UserWarning, stacklevel=2)

        return points


def _compute_k(n: int, k_depth: KDepth, outlier_prop: float, trust_alpha: float) -> int:
    """Select the depth, i.e. number of boxes to draw, based on the method."""
    if k_depth == "full":
        # extend boxes to 100% of the data
        k = int(np.log2(n)) + 1
    elif k_depth == "tukey":
        # This results with 5-8 points in each tail
        k = int(np.log2(n)) - 3
    elif k_depth == "proportion":
        k = int(np.log2(n)) - int(np.log2(n * outlier_prop)) + 1
    elif k_depth == "trustworthy":
        point_conf = 2 * NormalDist().inv_cdf(1 - trust_alpha / 2) ** 2
        k = int(np.log2(n / point_conf)) + 1
    else:
        # Allow having k directly specified as input
        k = int(k_depth)
    return max(k, 1)


def letter_values(
    x: np.ndarray,
    k_depth: KDepth = "tukey",
    outlier_prop: float = 0.007,
    trust_alpha: float = 0.05,
) -> dict[str, Any]:
    """Compute letter values, the percentiles a letter-value (boxen) plot draws.

    Based on the proposal in https://vita.had.co.nz/papers/letter-value-plot.pdf

    Args:
        x: 1-D array of values
        k_depth: Stopping rule for the tail percentiles: "tukey" (about as many
            outliers as a conventional boxplot), "proportion" (about
            ``outlier_prop`` outliers), "trustworthy" (the most extreme tail
            percentile is trusted at the ``trust_alpha`` level), "full" (boxes
            reach the minimum and maximum), or an int giving the depth directly
        outlier_prop: Expected outlier rate for ``k_depth="proportion"``
        trust_alpha: Confidence threshold for ``k_depth="trustworthy"``

    Returns:
        Dict with ``k`` (depth), ``levels``, ``percs`` (percentiles),
        ``values`` (their values), ``fliers`` (points outside the outermost
        box), and ``median``
    """
    k_options = ["tukey", "proportion", "trustworthy", "full"]
    if isinstance(k_depth, str):
        if k_depth not in k_options:
            raise ValueError(
                f"The value for `k_depth` must be one of {k_options}, "
                f"but {k_depth!r} was passed."
            )
    elif not isinstance(k_depth, int):
        raise TypeError(
            "The `k_depth` parameter must be either an integer or string "
            f"(one of {k_options}), not {k_depth!r}."
        )

    x = np.asarray(x)
    k = _compute_k(len(x), k_depth, outlier_prop, trust_alpha)
    exp = np.arange(k + 1, 1, -1), np.arange(2, k + 2)
    levels = k + 1 - np.concatenate([exp[0], exp[1][1:]])
    percentiles = 100 * np.concatenate([0.5 ** exp[0], 1 - 0.5 ** exp[1]])
    if k_depth == "full":
        percentiles[0] = 0
        percentiles[-1] = 100
    values = np.percentile(x, percentiles)
    fliers = np.asarray(x[(x < values.min()) | (x > values.max())])
    median = np.percentile(x, 50)

    return {
        "k": k,
        "levels": levels,
        "percs": percentiles,
        "values": values,
        "fliers": fliers,
        "median": median,
    }
