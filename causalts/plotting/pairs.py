# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pairs panel: scatterplot matrix visualization.

Inspired by R's psych::pairs.panels — histogram diagonal, scatter + linear
fit below the diagonal, correlation text above.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

from .corrplot import compute_association_matrix

_WIDE_DATA_WARNING_THRESHOLD = 15


# Okabe-Ito colorblind-safe palette: bright, distinguishable under
# deuteranopia/protanopia/tritanopia, unlike the earlier steelblue/firebrick
# (too dark/low-contrast) or a red/green pairing (indistinguishable to the
# most common form of color vision deficiency).
_POS_COLOR = "#0072B2"  # blue
_NEG_COLOR = "#D55E00"  # vermilion
_HIST_COLOR = "#56B4E9"  # sky blue


def pairs_panel(
    data,
    metric="pearson",
    hist=True,
    density=True,
    fit=True,
    scale_text=True,
    edgecolor="black",
    box_color="black",
    hist_color=_HIST_COLOR,
    point_color=_POS_COLOR,
    fit_color=_NEG_COLOR,
    figsize=None,
):
    """Scatterplot matrix (inspired by R's psych::pairs.panels).

    Diagonal shows a histogram per variable, the lower triangle shows
    scatterplots with a linear fit line, and the upper triangle shows the
    pairwise correlation coefficient as text.

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        Raw data where columns are variables.
    metric : str or callable
        Passed to `compute_association_matrix` for the upper-triangle
        correlation text ('pearson', 'spearman', 'kendall', 'dcor', or a
        callable with signature metric(x, y) -> float).
    hist : bool
        If True (default), draw a histogram on the diagonal.
    density : bool
        If True (default, mirrors psych's default), overlay a Gaussian KDE
        curve on each diagonal histogram (histogram is density-normalized
        to match the curve's scale).
    fit : bool
        If True (default), draw a linear least-squares fit line on each
        lower-triangle scatterplot (no smoothing/loess, plain `np.polyfit`).
    scale_text : bool
        If True (default), scale the upper-triangle correlation text's font
        size and color by |r| (mirrors psych's default). If False, all
        correlation text is drawn at a fixed size/color.
    edgecolor : str or None
        Border color for histogram bars and scatter point markers (default
        'black', matching R's plotting defaults). Pass None for no border.
    box_color : str or None
        Border color for every panel's outer box, including the
        upper-triangle text cells (default 'black', matching R's boxed
        panels). Pass None to leave panels borderless.
    hist_color : str
        Fill color for histogram bars (default a colorblind-safe sky blue).
    point_color : str
        Fill color for scatter points, and of positive correlation text
        (default a colorblind-safe blue).
    fit_color : str
        Color of the lower-triangle linear fit line, and of negative
        correlation text (default a colorblind-safe vermilion).
    figsize : tuple or None
        Figure size. If None, auto-scaled by the number of variables.

    Returns
    -------
    tuple
        (fig, axes, matrix) where matrix is the association matrix
        (pd.DataFrame) used for the upper-triangle text.
    """
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)

    cols = data.columns
    n = len(cols)

    if n > _WIDE_DATA_WARNING_THRESHOLD:
        warnings.warn(
            f"pairs_panel with {n} variables produces a {n}x{n} grid, which "
            "is hard to read; consider passing a smaller column subset "
            "(e.g. data[cols]).",
            stacklevel=2,
        )

    if figsize is None:
        size = max(4, n * 1.0)
        figsize = (size, size)

    matrix = compute_association_matrix(data, metric=metric)
    border = edgecolor if edgecolor is not None else "none"

    fig, axes = plt.subplots(n, n, figsize=figsize, squeeze=False)

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]

            if i == j:
                if hist:
                    col = data.iloc[:, i]
                    col_data = col[np.isfinite(col)].values
                    if len(col_data) > 0:
                        ax.hist(
                            col_data,
                            bins=15,
                            color=hist_color,
                            edgecolor=border,
                            density=density,
                        )
                        if density and len(col_data) > 1 and np.ptp(col_data) > 0:
                            kde = stats.gaussian_kde(col_data)
                            xs = np.linspace(col_data.min(), col_data.max(), 200)
                            ax.plot(xs, kde(xs), color="black", linewidth=2, zorder=3)
                ax.set_xticks([])
                ax.set_yticks([])
            elif i > j:
                x = data.iloc[:, j]
                y = data.iloc[:, i]
                mask = np.isfinite(x) & np.isfinite(y)
                ax.scatter(
                    x[mask],
                    y[mask],
                    s=10,
                    alpha=0.6,
                    color=point_color,
                    edgecolors=border,
                    linewidths=0.5,
                )
                if fit and mask.sum() >= 2 and x[mask].nunique() >= 2:
                    coeffs = np.polyfit(x[mask].values, y[mask].values, 1)
                    xs = np.linspace(x[mask].min(), x[mask].max(), 50)
                    ax.plot(xs, np.polyval(coeffs, xs), color=fit_color, linewidth=1)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                r = matrix.iloc[i, j]
                panel_inch = figsize[0] / n
                if not np.isfinite(r):
                    # e.g. a constant column -> undefined correlation.
                    text, fontsize, color = "NA", 6 + 3 * panel_inch, "black"
                elif scale_text:
                    text = f"{r:.2f}"
                    fontsize = 6 + min(abs(r), 1.0) * 14 * panel_inch
                    color = fit_color if r < 0 else point_color
                else:
                    text, fontsize, color = f"{r:.2f}", 6 + 3 * panel_inch, "black"
                ax.text(
                    0.5,
                    0.5,
                    text,
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    color=color,
                    transform=ax.transAxes,
                )
                ax.set_xticks([])
                ax.set_yticks([])

            if box_color is None:
                for spine in ax.spines.values():
                    spine.set_visible(False)
            else:
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color(box_color)

            if i == n - 1:
                ax.set_xlabel(cols[j], fontsize=8)
            if j == 0:
                ax.set_ylabel(cols[i], fontsize=8)

    fig.tight_layout()
    return fig, axes, matrix
