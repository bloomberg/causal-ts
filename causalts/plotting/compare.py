# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Visual comparison and analysis of discovered causal graphs against ground truth.

Functions:
  - ``compare_graphs`` — overlay discovered vs true graph with TP/FP/FN coloring
  - ``plot_metrics_summary`` — bar chart of Precision / Recall / F1 / SHD
  - ``plot_lag_metrics`` — heatmap of per-lag precision / recall / F1
  - ``plot_pvalue_distribution`` — histogram of p-values split by true vs non-edges
"""

import matplotlib.patches as mpatches
import matplotlib.patheffects as PathEffects
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch

from ._core import compute_node_positions


def _to_bool(graph):
    """Convert a graph array to boolean, treating any nonzero / non-empty as True."""
    if graph.dtype.kind in ("U", "S"):  # string dtype
        return graph != ""
    return graph.astype(bool)


def compare_graphs(
    graph_true,
    graph_discovered,
    var_names=None,
    fig_ax=None,
    figsize=None,
    node_pos=None,
    node_layout="neato",
    node_size=0.2,
    node_color="lightgrey",
    arrow_linewidth=3,
    label_fontsize=10,
    node_label_size=12,
    link_label_fontsize=10,
    curved_radius=0.2,
    tp_color="green",
    fp_color="orange",
    fp_style="solid",
    fp_linewidth=None,
    fn_color="red",
    fn_style="dashed",
    fn_linewidth=None,
    title=None,
    show_legend=True,
    legend_loc="lower left",
    legend_fontsize=None,
    legend_framealpha=0.8,
    legend_markerscale=1.0,
    return_pos=False,
):
    """Compare a discovered causal graph against ground truth.

    Draws a single plot where edges are colored by classification:
    TP (correctly discovered), FP (spurious), and FN (missed).

    Parameters
    ----------
    graph_true : array-like, shape (N, N, tau_max+1)
        Ground truth graph. Values are 0/1 or string (``'-->'`` etc.).
    graph_discovered : array-like, shape (N, N, tau_max+1)
        Discovered graph. Same format as *graph_true*.
    var_names : list of str, optional
        Variable names. Defaults to ``['0', '1', ...]``.
    fig_ax : tuple (fig, ax), optional
        Existing figure and axes. Created if *None*.
    figsize : tuple, optional
        Figure size when *fig_ax* is *None*.
    node_pos : dict, optional
        ``{'x': array(N,), 'y': array(N,)}`` node positions.
        If *None*, computed from *node_layout*.
    node_layout : str, optional
        Layout algorithm (``'circular'``, ``'spring'``, etc.). Default ``'circular'``.
    node_size : float, optional
        Node diameter. Default 0.3.
    node_color : str, optional
        Fill color for nodes. Default ``'lightgrey'``.
    arrow_linewidth : float, optional
        Edge width. Default 6.
    label_fontsize : float, optional
        Font size for colorbar / axis labels.
    node_label_size : float, optional
        Font size for node labels.
    link_label_fontsize : float, optional
        Font size for edge lag labels.
    curved_radius : float, optional
        Curvature of edges. Default 0.2.
    tp_color : str, optional
        Color for true-positive edges. Default ``'green'``.
    fp_color : str, optional
        Color for false-positive edges. Default ``'orange'``.
    fp_style : str, optional
        Line style for FP edges (``'solid'``, ``'dashed'``, ``'dotted'``, …).
        Default ``'solid'``.
    fp_linewidth : float, optional
        Line width for FP edges. Default is ``arrow_linewidth`` (same as TP).
        Set smaller (e.g. ``arrow_linewidth * 0.6``) to visually de-emphasise
        false positives relative to true positives.
    fn_color : str, optional
        Color for false-negative edges. Default ``'red'``.
    fn_style : str, optional
        Line style for FN edges. Default ``'dashed'``.
    fn_linewidth : float, optional
        Line width for FN (missed) edges. Default is ``arrow_linewidth * 0.4``.
    title : str, optional
        Plot title.
    show_legend : bool, optional
        Whether to show the TP / FP / FN legend. Default *True*.
    legend_loc : str, optional
        Legend location (any valid matplotlib ``loc``). Default ``'lower left'``.
    legend_fontsize : float, optional
        Font size for legend text. Defaults to ``label_fontsize - 1``.
    legend_framealpha : float, optional
        Opacity of the legend box background. Default ``0.8``. Set to ``0``
        to make the box transparent so it does not obscure graph nodes on
        small figures.
    legend_markerscale : float, optional
        Scale factor for legend marker size. Default ``1.0``.
    return_pos : bool, optional
        If *True*, return ``(fig, ax, node_pos_dict)`` instead of ``(fig, ax)``.

    Returns
    -------
    fig, ax   or   fig, ax, node_pos_dict
    """
    graph_true = np.asarray(graph_true).squeeze()
    graph_discovered = np.asarray(graph_discovered).squeeze()

    if graph_true.ndim == 2:
        graph_true = np.expand_dims(graph_true, axis=2)
    if graph_discovered.ndim == 2:
        graph_discovered = np.expand_dims(graph_discovered, axis=2)

    gt = _to_bool(graph_true)
    disc = _to_bool(graph_discovered)

    if fn_linewidth is None:
        fn_linewidth = arrow_linewidth * 0.4
    if fp_linewidth is None:
        fp_linewidth = arrow_linewidth
    if legend_fontsize is None:
        legend_fontsize = label_fontsize - 1

    N = gt.shape[0]
    tau_max = gt.shape[2] - 1

    if var_names is None:
        var_names = [str(i) for i in range(N)]

    # ------------------------------------------------------------------
    # Build a networkx graph for layout computation (union of all edges)
    # ------------------------------------------------------------------
    union = gt | disc
    # Ignore self-loops at lag 0
    for i in range(N):
        union[i, i, 0] = False
    net = np.any(union[:, :, :], axis=2)
    # For layout purposes use upper triangle for undirected positioning
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for i in range(N):
        for j in range(N):
            if i != j and net[i, j]:
                G.add_edge(i, j)

    # ------------------------------------------------------------------
    # Compute positions
    # ------------------------------------------------------------------
    if node_pos is not None:
        pos = {i: (node_pos["x"][i], node_pos["y"][i]) for i in range(N)}
    else:
        pos = compute_node_positions(G, layout=node_layout, seed=42)

    # ------------------------------------------------------------------
    # Set up figure
    # ------------------------------------------------------------------
    if fig_ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, frame_on=False)
    else:
        fig, ax = fig_ax

    if title:
        ax.set_title(title, fontsize=label_fontsize + 2)

    ax.spines["left"].set_color("none")
    ax.spines["right"].set_color("none")
    ax.spines["bottom"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.set_xticks([])
    ax.set_yticks([])

    # Invisible scatter to fix matplotlib auto-scaling
    ax.scatter(0, 0, zorder=-10, alpha=0)

    # ------------------------------------------------------------------
    # Draw nodes
    # ------------------------------------------------------------------
    node_patches = {}
    for n in range(N):
        c = Ellipse(
            pos[n],
            width=node_size,
            height=node_size,
            clip_on=False,
            facecolor=node_color,
            edgecolor=node_color,
            zorder=2,
        )
        ax.add_patch(c)
        node_patches[n] = c
        ax.text(
            pos[n][0],
            pos[n][1],
            var_names[n],
            fontsize=node_label_size,
            horizontalalignment="center",
            verticalalignment="center",
            zorder=5,
        )

    # ------------------------------------------------------------------
    # Classify edges per (i, j) pair
    # ------------------------------------------------------------------
    # For each directed pair (i→j), collect per-lag status
    edge_info = {}  # (i,j) -> list of (lag, status)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            lags_tp, lags_fp, lags_fn = [], [], []
            for tau in range(0 if i != j else 1, tau_max + 1):
                if tau == 0 and i == j:
                    continue
                in_gt = gt[i, j, tau]
                in_disc = disc[i, j, tau]
                if in_gt and in_disc:
                    lags_tp.append(tau)
                elif in_disc and not in_gt:
                    lags_fp.append(tau)
                elif in_gt and not in_disc:
                    lags_fn.append(tau)
            if lags_tp or lags_fp or lags_fn:
                edge_info[(i, j)] = (lags_tp, lags_fp, lags_fn)

    # ------------------------------------------------------------------
    # Draw edges
    # ------------------------------------------------------------------
    seen = {}  # (i,j) -> list of rad values already used

    def _draw_arrow(ax, i, j, color, linestyle, lw, label_text, zorder=0):
        """Draw a curved arrow from node i to node j."""
        n1 = node_patches[i]
        n2 = node_patches[j]
        coor1 = n1.center
        coor2 = n2.center

        key = (i, j)
        rev_key = (j, i)

        # Collect all radii already used for this pair (both directions)
        used_same = seen.get(key, [])
        used_rev = seen.get(rev_key, [])

        if not used_same and not used_rev:
            rad = curved_radius
        elif used_rev and not used_same:
            # Reverse edge exists — use same sign so arcs curve on opposite
            # physical sides (arc3 rad is relative to the direction of travel)
            rad = used_rev[-1]
            while any(abs(rad - u) < 0.05 for u in used_same):
                rad = rad + np.sign(rad) * 0.15 if rad != 0 else curved_radius
        else:
            last = (used_same + used_rev)[-1]
            rad = -last
            all_used = used_same + used_rev
            while any(abs(rad - u) < 0.05 for u in all_used):
                rad = rad + np.sign(rad) * 0.15 if rad != 0 else curved_radius

        seen.setdefault(key, []).append(rad)

        arrowstyle = "Simple, head_width=2, head_length=2, tail_width=1"
        e_p = FancyArrowPatch(
            coor1,
            coor2,
            arrowstyle=arrowstyle,
            connectionstyle=f"arc3,rad={rad}",
            mutation_scale=1 * lw,
            lw=0.0 if linestyle == "solid" else lw * 0.3,
            aa=True,
            alpha=1.0,
            linestyle=linestyle,
            color=color,
            clip_on=False,
            patchA=n1,
            patchB=n2,
            shrinkA=0,
            shrinkB=0,
            zorder=zorder,
            capstyle="butt",
        )
        ax.add_artist(e_p)

        # Draw label at midpoint of the edge
        if label_text and link_label_fontsize > 0:
            # Pick label color to contrast with the axes background
            bg = ax.get_facecolor()  # RGBA tuple
            luminance = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
            text_color = "black" if luminance > 0.5 else "white"

            # Use an invisible marker arrow to find the midpoint
            e_marker = FancyArrowPatch(
                coor1,
                coor2,
                arrowstyle="-",
                connectionstyle=f"arc3,rad={rad}",
                mutation_scale=1 * lw,
                lw=0.0,
                alpha=0.0,
                color=color,
                clip_on=False,
                patchA=n1,
                patchB=n2,
                shrinkA=0,
                shrinkB=0,
                zorder=-10,
            )
            ax.add_artist(e_marker)
            # Get rendered curve from the no-arrowhead marker (not the
            # arrow patch, whose polygon includes the arrowhead outline).
            # Then find the point on the rendered curve closest to the
            # middle Bezier control point — same approach as _core.py.
            ctrl = e_marker.get_path().vertices.copy()
            ctrl_mid = ctrl[len(ctrl) // 2]
            polys = e_marker.get_path().to_polygons(transform=None)
            if polys and len(polys[0]) > 2:
                rendered = np.asarray(polys[0])
                dists = np.sum((rendered - ctrl_mid) ** 2, axis=1)
                mid = rendered[np.argmin(dists)]
            else:
                mid = ctrl_mid

            stroke_color = "white" if text_color == "black" else "black"
            txt = ax.text(
                mid[0],
                mid[1],
                label_text,
                fontsize=link_label_fontsize,
                verticalalignment="center",
                horizontalalignment="center",
                color=text_color,
                fontweight="bold",
                zorder=3,
            )
            txt.set_path_effects(
                [PathEffects.withStroke(linewidth=3, foreground=stroke_color)]
            )

    for (i, j), (lags_tp, lags_fp, lags_fn) in edge_info.items():
        # Build label and decide color
        has_tp = len(lags_tp) > 0
        has_fp = len(lags_fp) > 0
        has_fn = len(lags_fn) > 0

        # Case 1: Only TP lags — solid green edge
        if has_tp and not has_fp and not has_fn:
            label = ",".join(str(l) for l in sorted(lags_tp))  # noqa: E741
            _draw_arrow(ax, i, j, tp_color, "solid", arrow_linewidth, label, zorder=0)

        # Case 2: Only FP lags
        elif has_fp and not has_tp and not has_fn:
            label = ",".join(str(l) for l in sorted(lags_fp))  # noqa: E741
            _draw_arrow(ax, i, j, fp_color, fp_style, fp_linewidth, label, zorder=0)

        # Case 3: Only FN lags — dashed red edge
        elif has_fn and not has_tp and not has_fp:
            label = ",".join(str(l) for l in sorted(lags_fn))  # noqa: E741
            _draw_arrow(ax, i, j, fn_color, fn_style, fn_linewidth, label, zorder=-1)

        # Case 4: Mixed — build per-lag annotated label
        else:
            disc_lags = sorted(lags_tp + lags_fp)
            if disc_lags:
                # When FP lags outnumber TP, use FP style/width for the combined edge
                fp_dominant = len(lags_fp) > len(lags_tp)
                edge_color = fp_color if fp_dominant else tp_color
                edge_style = fp_style if fp_dominant else "solid"
                edge_lw = fp_linewidth if fp_dominant else arrow_linewidth
                # Build annotated label: lag✓ for TP, lag✗ for FP
                parts = []
                for l in disc_lags:  # noqa: E741
                    if l in lags_tp:
                        parts.append(f"{l}\u2713")
                    else:
                        parts.append(f"{l}\u2717")
                label = ",".join(parts)
                _draw_arrow(ax, i, j, edge_color, edge_style, edge_lw, label, zorder=0)

            # Draw FN part as separate dashed arrow
            if lags_fn:
                fn_label = ",".join(str(l) for l in sorted(lags_fn))  # noqa: E741
                _draw_arrow(
                    ax, i, j, fn_color, fn_style, fn_linewidth, fn_label, zorder=-1
                )

    ax.set_aspect("equal")

    # ------------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------------
    if show_legend:
        legend_handles = [
            mpatches.Patch(
                facecolor=tp_color, edgecolor=tp_color, label="TP (correct)"
            ),
            mpatches.Patch(
                facecolor=fp_color, edgecolor=fp_color, label="FP (spurious)"
            ),
            mpatches.Patch(
                facecolor=fn_color,
                edgecolor=fn_color,
                linestyle="dashed",
                alpha=0.4,
                label="FN (missed)",
            ),
        ]
        ax.legend(
            handles=legend_handles,
            loc=legend_loc,
            fontsize=legend_fontsize,
            framealpha=legend_framealpha,
            markerscale=legend_markerscale,
        )

    # ------------------------------------------------------------------
    # Return
    # ------------------------------------------------------------------
    node_pos_out = {
        "x": np.array([pos[i][0] for i in range(N)]),
        "y": np.array([pos[i][1] for i in range(N)]),
    }

    if return_pos:
        return fig, ax, node_pos_out
    return fig, ax


# Re-export from canonical location (causalts.cdnots) for backwards compatibility
from ..cdnots.phase3_utils import build_pvalue_matrix  # noqa: E402,F401

# ======================================================================
# Metrics summary bar chart
# ======================================================================


def plot_metrics_summary(
    graph_true,
    graph_discovered,
    metrics=None,
    level="lag",
    exclude_self_loops=False,
    fig_ax=None,
    figsize=None,
    bar_color="steelblue",
    title=None,
):
    """Bar chart of graph evaluation metrics.

    Parameters
    ----------
    graph_true : array-like, shape (N, N, tau_max+1)
        Ground truth graph (0/1).
    graph_discovered : array-like, shape (N, N, tau_max+1)
        Discovered graph (0/1).
    metrics : list of str, optional
        Metric names to show. Default ``["Precision", "Recall", "F1", "SHD"]``.
        Valid names: any key returned by ``evaluate_graph`` — e.g.
        ``"TPR"``, ``"FPR"``, ``"Precision"``, ``"Recall"``, ``"F1"``, ``"SHD"``,
        ``"TP"``, ``"FP"``, ``"FN"``, ``"TN"`` (per-lag) or their ``_pair``
        variants when *level="pair"*.
    level : str, optional
        ``"lag"`` for per-lag metrics, ``"pair"`` for pair-level. Default ``"lag"``.
    exclude_self_loops : bool, optional
        If *True*, exclude autodependencies (self-loops at all lags) from
        per-lag evaluation. Makes per-lag metrics directly comparable with
        pair-level. Default *False*.
    fig_ax : tuple (fig, ax), optional
        Existing figure and axes.
    figsize : tuple, optional
        Figure size.
    bar_color : str, optional
        Bar fill color. Default ``'steelblue'``.
    title : str, optional
        Plot title.

    Returns
    -------
    fig, ax
    """
    from ..utils.helpers import evaluate_graph

    gt = np.asarray(graph_true).astype(np.int8)
    disc = np.asarray(graph_discovered).astype(np.int8)
    results = evaluate_graph(disc, gt, exclude_self_loops=exclude_self_loops)

    if metrics is None:
        metrics = ["Precision", "Recall", "F1", "SHD"]

    if level == "pair":
        suffix_map = {
            "Precision": "Precision_pair",
            "Recall": "TPR_pair",
            "F1": "F1_pair",
            "SHD": "SHD_pair",
            "TPR": "TPR_pair",
            "FPR": "FPR_pair",
        }
        keys = [suffix_map.get(m, m) for m in metrics]
    else:
        keys = [("Recall" if m == "Recall" else m) for m in metrics]
        keys = [("TPR" if k == "Recall" else k) for k in keys]

    values = [results[k] for k in keys]

    # Identify ratio metrics (0-1) vs count metrics (SHD, TP, etc.)
    ratio_idx = [
        i
        for i, m in enumerate(metrics)
        if m in ("Precision", "Recall", "F1", "TPR", "FPR")
    ]

    if fig_ax is None:
        fig, ax = plt.subplots(figsize=figsize or (6, max(2.5, 0.6 * len(metrics))))
    else:
        fig, ax = fig_ax

    y_pos = np.arange(len(metrics))
    bars = ax.barh(y_pos, values, color=bar_color, edgecolor="white", height=0.6)

    # Annotate bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        fmt = f"{val:.2f}" if i in ratio_idx else f"{val:g}"
        ax.text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height() / 2,
            fmt,
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.2 if max(values) > 0 else 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if title:
        ax.set_title(title, fontsize=13)

    return fig, ax


# ======================================================================
# Per-lag metrics heatmap
# ======================================================================


def plot_lag_metrics(
    graph_true,
    graph_discovered,
    metrics=None,
    fig_ax=None,
    figsize=None,
    cmap="YlGn",
    annotate=True,
    title=None,
):
    """Heatmap of precision / recall / F1 broken down by lag.

    Parameters
    ----------
    graph_true : array-like, shape (N, N, tau_max+1)
        Ground truth graph (0/1).
    graph_discovered : array-like, shape (N, N, tau_max+1)
        Discovered graph (0/1).
    metrics : list of str, optional
        Default ``["Precision", "Recall", "F1"]``.
    fig_ax : tuple (fig, ax), optional
        Existing figure and axes.
    figsize : tuple, optional
        Figure size.
    cmap : str, optional
        Colormap for the heatmap. Default ``'YlGn'``.
    annotate : bool, optional
        Show values in cells. Default *True*.
    title : str, optional
        Plot title.

    Returns
    -------
    fig, ax
    """
    gt = _to_bool(np.asarray(graph_true).squeeze())
    disc = _to_bool(np.asarray(graph_discovered).squeeze())
    if gt.ndim == 2:
        gt = np.expand_dims(gt, axis=2)
    if disc.ndim == 2:
        disc = np.expand_dims(disc, axis=2)

    N = gt.shape[0]
    tau_max = gt.shape[2] - 1

    if metrics is None:
        metrics = ["Precision", "Recall", "F1"]

    # Build per-lag mask (exclude self-loops at lag 0)
    mask = np.ones(gt.shape, dtype=bool)
    for i in range(N):
        mask[i, i, 0] = False

    # Compute metrics per lag
    data = np.zeros((len(metrics), tau_max + 1))
    for tau in range(tau_max + 1):
        gt_tau = gt[:, :, tau][mask[:, :, tau]]
        disc_tau = disc[:, :, tau][mask[:, :, tau]]
        tp = int(np.sum(gt_tau & disc_tau))
        fp = int(np.sum(~gt_tau & disc_tau))
        fn = int(np.sum(gt_tau & ~disc_tau))
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        metric_map = {
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "SHD": fp + fn,
        }
        for mi, m in enumerate(metrics):
            data[mi, tau] = metric_map.get(m, 0)

    if fig_ax is None:
        fig, ax = plt.subplots(
            figsize=figsize or (max(4, 1.2 * (tau_max + 1)), max(2, 0.8 * len(metrics)))
        )
    else:
        fig, ax = fig_ax

    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(tau_max + 1))
    ax.set_xticklabels([str(t) for t in range(tau_max + 1)])
    ax.set_xlabel("Lag", fontsize=11)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics, fontsize=11)

    if annotate:
        for mi in range(len(metrics)):
            for tau in range(tau_max + 1):
                val = data[mi, tau]
                color = "white" if val > 0.6 else "black"
                ax.text(
                    tau,
                    mi,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color=color,
                    fontweight="bold",
                )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if title:
        ax.set_title(title, fontsize=13)

    return fig, ax


# ======================================================================
# P-value distribution histogram
# ======================================================================


def plot_pvalue_distribution(
    pvalue_matrix,
    graph_true,
    graph_discovered=None,
    bins=30,
    fig_ax=None,
    figsize=None,
    alpha_line=None,
    true_edge_color="green",
    non_edge_color="grey",
    log_scale=False,
    title=None,
):
    """Histogram of p-values split by true edges vs non-edges.

    When ``graph_discovered`` is provided, only p-values for edges in the
    discovered graph are shown and the split becomes **TP vs FP** instead
    of "true edges vs non-edges".

    Parameters
    ----------
    pvalue_matrix : array-like, shape (N, N, tau_max+1)
        Matrix of p-values. Use ``NaN`` for untested pairs.
        Also accepts a CausalGraph object with a ``.pvalue_matrix`` attribute.
    graph_true : array-like, shape (N, N, tau_max+1)
        Ground truth graph (0/1).
    graph_discovered : array-like, shape (N, N, tau_max+1), optional
        Discovered graph (0/1). When given, only edges present in this
        graph are included and the histogram splits into TP vs FP.
    bins : int, optional
        Number of histogram bins. Default 30.
    fig_ax : tuple (fig, ax), optional
        Existing figure and axes.
    figsize : tuple, optional
        Figure size.
    alpha_line : float, optional
        If given, draw a vertical line at this significance level.
    true_edge_color : str, optional
        Histogram color for true edges / TP. Default ``'green'``.
    non_edge_color : str, optional
        Histogram color for non-edges / FP. Default ``'grey'``.
    log_scale : bool, optional
        Use log10 scale for x-axis. Default *False*.
    title : str, optional
        Plot title.

    Returns
    -------
    fig, ax
    """
    # Accept CausalGraph object with .pvalue_matrix attribute
    if hasattr(pvalue_matrix, "pvalue_matrix"):
        pvalue_matrix = pvalue_matrix.pvalue_matrix
    pvals = np.asarray(pvalue_matrix, dtype=float).squeeze()
    gt = _to_bool(np.asarray(graph_true).squeeze())
    if pvals.ndim == 2:
        pvals = np.expand_dims(pvals, axis=2)
    if gt.ndim == 2:
        gt = np.expand_dims(gt, axis=2)

    N = gt.shape[0]

    # Mask: valid (not NaN) and not self-loop at lag 0
    valid = ~np.isnan(pvals)
    for i in range(N):
        valid[i, i, 0] = False

    # If discovered graph provided, restrict to discovered edges only
    masked = graph_discovered is not None
    if masked:
        disc = _to_bool(np.asarray(graph_discovered).squeeze())
        if disc.ndim == 2:
            disc = np.expand_dims(disc, axis=2)
        valid &= disc

    pvals_true = pvals[valid & gt]
    pvals_non = pvals[valid & ~gt]

    if fig_ax is None:
        fig, ax = plt.subplots(figsize=figsize or (7, 4))
    else:
        fig, ax = fig_ax

    if log_scale:
        # Replace zeros with small value for log
        pvals_true = np.where(pvals_true == 0, 1e-300, pvals_true)
        pvals_non = np.where(pvals_non == 0, 1e-300, pvals_non)
        all_pv = np.concatenate([pvals_true, pvals_non])
        log_min = (
            np.floor(np.log10(all_pv[all_pv > 0].min()))
            if len(all_pv[all_pv > 0])
            else -10
        )
        bin_edges = np.logspace(log_min, 0, bins + 1)
    else:
        bin_edges = np.linspace(0, 1, bins + 1)

    if masked:
        label_pos = f"TP (n={len(pvals_true)})"
        label_neg = f"FP (n={len(pvals_non)})"
    else:
        label_pos = f"True edges (n={len(pvals_true)})"
        label_neg = f"Non-edges (n={len(pvals_non)})"

    ax.hist(
        pvals_non,
        bins=bin_edges,
        alpha=0.5,
        color=non_edge_color,
        label=label_neg,
        edgecolor="white",
    )
    ax.hist(
        pvals_true,
        bins=bin_edges,
        alpha=0.7,
        color=true_edge_color,
        label=label_pos,
        edgecolor="white",
    )

    if log_scale:
        ax.set_xscale("log")

    if alpha_line is not None:
        ax.axvline(
            alpha_line,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"α = {alpha_line}",
        )

    ax.set_xlabel("p-value" + (" (log scale)" if log_scale else ""), fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if title:
        ax.set_title(title, fontsize=13)

    return fig, ax
