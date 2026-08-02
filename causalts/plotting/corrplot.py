"""Corrplot: correlation/association matrix visualization.

Inspired by R's corrplot package. Supports arbitrary pairwise metrics
(Pearson, Spearman, dcor, MI, custom callables) with 7 glyph methods,
significance overlays, confidence intervals, and hierarchical ordering.
"""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hierarchy
import scipy.spatial.distance as distance
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import (
    Circle,
    Ellipse,
    Rectangle,
    Wedge,
)

_VALID_METHODS = ("circle", "square", "ellipse", "number", "color", "shade", "pie")
_VALID_INSIG = ("pch", "p-value", "blank", "n", "label_sig")
_VALID_ORDER = ("original", "hclust", "AOE", "FPC", "alphabet")
_VALID_DIAG = ("names", "values", "blank", "hist")
_VALID_PLOT_CI = ("n", "square", "circle", "rect")
_VALID_TL_POS = ("lt", "lb", "ld", "td", "l", "d", "n")
_VALID_CL_POS = ("r", "b", "n")


def compute_association_matrix(data, metric="pearson"):
    """Compute a pairwise association matrix from raw data.

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        Raw data where columns are variables.
    metric : str or callable
        String preset ('pearson', 'spearman', 'kendall', 'dcor') or
        a callable with signature metric(x, y) -> float.

    Returns
    -------
    pd.DataFrame
        NxN symmetric matrix of pairwise associations.
    """
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)

    if isinstance(metric, str):
        if metric in ("pearson", "spearman", "kendall"):
            return data.corr(method=metric)
        elif metric == "dcor":
            import dcor

            cols = data.columns
            n = len(cols)
            matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(i, n):
                    val = dcor.distance_correlation(
                        data.iloc[:, i].values, data.iloc[:, j].values
                    )
                    matrix[i, j] = val
                    matrix[j, i] = val
            return pd.DataFrame(matrix, index=cols, columns=cols)
        else:
            raise ValueError(
                f"Unknown metric preset '{metric}'. "
                f"Use 'pearson', 'spearman', 'kendall', 'dcor', or a callable."
            )
    elif callable(metric):
        cols = data.columns
        n = len(cols)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                val = metric(data.iloc[:, i].values, data.iloc[:, j].values)
                matrix[i, j] = val
                matrix[j, i] = val
        return pd.DataFrame(matrix, index=cols, columns=cols)
    else:
        raise TypeError("metric must be a string preset or callable.")


def _is_precomputed(data):
    """Check if data looks like a pre-computed square matrix."""
    if isinstance(data, pd.DataFrame):
        if data.shape[0] == data.shape[1]:
            if list(data.index) == list(data.columns):
                return True
    elif isinstance(data, np.ndarray):
        # A square ndarray is only treated as precomputed if it is symmetric;
        # otherwise a raw n_samples == n_features observation matrix would be
        # misread as an association matrix. Pass is_corr=False to force raw
        # handling of a symmetric square input.
        if data.ndim == 2 and data.shape[0] == data.shape[1]:
            return bool(np.allclose(data, data.T, equal_nan=True))
    return False


def _detect_range(matrix, is_corr):
    """Detect value range and whether to use diverging colormap."""
    values = matrix.values if isinstance(matrix, pd.DataFrame) else matrix
    vmin_data = np.nanmin(values)
    vmax_data = np.nanmax(values)

    if is_corr is True:
        return -1.0, 1.0, 0.0, True
    elif is_corr is False:
        return vmin_data, vmax_data, None, False

    if vmin_data >= -1.0 and vmax_data <= 1.0:
        if vmin_data < 0:
            return -1.0, 1.0, 0.0, True
    if vmin_data >= 0:
        return vmin_data, vmax_data, None, False
    return vmin_data, vmax_data, 0.0, True


def _reorder_matrix(matrix, order, hclust_method):
    """Reorder matrix rows/columns based on ordering method."""
    if order == "original":
        return matrix, np.arange(len(matrix))
    elif order == "alphabet":
        idx = np.argsort(matrix.columns.astype(str))
        return matrix.iloc[idx, idx], idx
    elif order == "hclust":
        d = distance.pdist(matrix.values)
        d = np.nan_to_num(d, nan=0.0)
        Y = hierarchy.linkage(d, method=hclust_method)
        Z = hierarchy.dendrogram(Y, no_plot=True)
        idx = Z["leaves"]
        return matrix.iloc[idx, idx], idx
    elif order == "FPC":
        eigvals, eigvecs = np.linalg.eigh(np.nan_to_num(matrix.values))
        idx = np.argsort(eigvecs[:, -1])
        return matrix.iloc[idx, idx], idx
    elif order == "AOE":
        eigvals, eigvecs = np.linalg.eigh(np.nan_to_num(matrix.values))
        e1 = eigvecs[:, -1]
        e2 = eigvecs[:, -2] if eigvecs.shape[1] > 1 else np.zeros_like(e1)
        angles = np.arctan2(e2, e1)
        idx = np.argsort(angles)
        return matrix.iloc[idx, idx], idx
    else:
        raise ValueError(f"Unknown order method '{order}'. Use: {_VALID_ORDER}")


def _get_norm(vmin, vmax, vcenter, diverging):
    """Create matplotlib Normalize or TwoSlopeNorm."""
    if diverging and vcenter is not None:
        return mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    return mcolors.Normalize(vmin=vmin, vmax=vmax)


def _cell_method(i, j, method, upper, lower):
    """Get the rendering method for a specific cell."""
    if upper is not None and lower is not None:
        if i < j:
            return upper
        elif i > j:
            return lower
        return None
    return method


def corrplot(
    data,
    metric="pearson",
    method="circle",
    lower=None,
    upper=None,
    type="full",
    diag="hist",
    is_corr=None,
    order="original",
    hclust_method="complete",
    addrect=None,
    rect_col="black",
    rect_lwd=2,
    vmin=None,
    vmax=None,
    vcenter=None,
    cmap=None,
    col_lim=None,
    pvalues=None,
    sig_level=0.05,
    insig="pch",
    pch="x",
    pch_col="black",
    pch_cex=3,
    low_ci=None,
    upp_ci=None,
    plot_ci="n",
    add_coef=False,
    add_coef_col="black",
    number_cex=1.0,
    number_fmt=".2f",
    addshade="all",
    shade_lwd=1,
    shade_col="white",
    tl_pos=None,
    tl_cex=1.0,
    tl_col="black",
    tl_srt=45,
    tl_offset=0.4,
    colorbar=True,
    cl_pos=None,
    cl_cex=0.8,
    cl_ratio=0.15,
    cl_label=None,
    grid=True,
    grid_color=None,
    na_label="?",
    na_label_col="black",
    title=None,
    title_fontsize=None,
    figsize=None,
    bg="white",
    outline=True,
    outline_col="black",
    shrink=0.9,
    fig_ax=None,
    **kwargs,
):
    """Plot a correlation/association matrix with glyph encoding.

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        Pre-computed NxN association matrix, or raw data (columns = variables).
        If raw data is passed, the matrix is computed using `metric`.
    metric : str or callable
        Metric for computing association matrix from raw data.
        Presets: 'pearson', 'spearman', 'kendall', 'dcor'.
        Or a callable(x, y) -> float.
    method : str
        Glyph method: 'circle', 'square', 'ellipse', 'number', 'color',
        'shade', 'pie'.
    lower : str or None
        Method for lower triangle (overrides `method` for lower half).
    upper : str or None
        Method for upper triangle (overrides `method` for upper half).
    type : str
        Display type: 'full', 'upper', 'lower'.
    diag : str
        Diagonal display: 'names', 'values', 'blank', 'hist'. Note 'hist'
        requires raw data; with a precomputed matrix it renders blank
        (variable names still appear on the tick labels) — use 'names' or
        'values' for precomputed input.
    is_corr : bool or None
        If None, auto-detect whether data is correlation-like ([-1, 1]).
        If True, force [-1, 1] range with diverging cmap.
    order : str
        Reordering: 'original', 'hclust', 'AOE', 'FPC', 'alphabet'.
    hclust_method : str
        Linkage method for hierarchical clustering.
    addrect : int or None
        Number of cluster rectangles to draw (requires order='hclust').
    rect_col : str
        Cluster rectangle border color.
    rect_lwd : float
        Cluster rectangle line width.
    vmin, vmax : float or None
        Value range for color/size mapping. Auto-detected if None.
    vcenter : float or None
        Center value for diverging colormap.
    cmap : str or Colormap or None
        Colormap. Auto-selected if None (diverging: 'RdBu_r', sequential: 'viridis').
    col_lim : tuple or None
        Alias for (vmin, vmax).
    pvalues : np.ndarray or pd.DataFrame or None
        NxN p-value matrix for significance overlay.
    sig_level : float or list
        Significance threshold(s). List for multi-level stars [0.001, 0.01, 0.05].
    insig : str
        How to mark insignificant cells: 'pch', 'p-value', 'blank', 'n', 'label_sig'.
    pch : str
        Character for insig='pch' (default 'x').
    pch_col : str
        Color for pch character.
    pch_cex : float
        Size multiplier for pch character.
    low_ci, upp_ci : np.ndarray or pd.DataFrame or None
        Lower/upper confidence interval bound matrices.
    plot_ci : str
        CI visualization: 'n', 'square', 'circle', 'rect'.
    add_coef : bool
        Overlay coefficient numbers on glyphs.
    add_coef_col : str
        Color for coefficient overlay text.
    number_cex : float
        Size multiplier for coefficient numbers.
    number_fmt : str
        Format string for numbers (e.g., '.2f', '.1%').
    addshade : str
        Shade direction for method='shade': 'negative', 'positive', 'all'.
    shade_lwd : float
        Shade line width.
    shade_col : str
        Shade line color.
    tl_pos : str or None
        Label position: 'lt', 'ld', 'td', 'l', 'd', 'n'. Auto if None.
    tl_cex : float
        Label font size multiplier.
    tl_col : str
        Label color.
    tl_srt : float
        Label rotation in degrees.
    tl_offset : float
        Label offset from axis.
    colorbar : bool
        Show colorbar.
    cl_pos : str or None
        Colorbar position: 'r' (right), 'b' (bottom), 'n' (none). Auto if None.
    cl_cex : float
        Colorbar label size.
    cl_ratio : float
        Colorbar width ratio.
    cl_label : str or None
        Colorbar title.
    grid : bool
        Show grid lines between cells.
    grid_color : str or None
        Grid line color. Auto if None.
    na_label : str
        Label for NaN cells ('?' or 'square').
    na_label_col : str
        Color for NA labels.
    title : str or None
        Plot title.
    title_fontsize : float or None
        Title font size.
    figsize : tuple or None
        Figure size (width, height).
    bg : str
        Background color.
    outline : bool
        Draw outline on glyphs.
    outline_col : str
        Outline color.
    shrink : float
        Maximum glyph size as fraction of cell (0 to 1).
    fig_ax : tuple or None
        (fig, ax) to draw on existing axes.

    Returns
    -------
    tuple
        (fig, ax, matrix) where matrix is the association matrix (pd.DataFrame).
    """
    # --- Argument validation ---
    if method not in _VALID_METHODS:
        raise ValueError(f"Unknown method '{method}'. Use one of: {_VALID_METHODS}")
    for name, val in (("lower", lower), ("upper", upper)):
        if val is not None and val not in _VALID_METHODS:
            raise ValueError(
                f"Unknown {name} method '{val}'. Use one of: {_VALID_METHODS}"
            )
    if insig not in _VALID_INSIG:
        raise ValueError(f"Unknown insig '{insig}'. Use one of: {_VALID_INSIG}")
    if diag not in _VALID_DIAG:
        raise ValueError(f"Unknown diag '{diag}'. Use one of: {_VALID_DIAG}")
    if plot_ci not in _VALID_PLOT_CI:
        raise ValueError(f"Unknown plot_ci '{plot_ci}'. Use one of: {_VALID_PLOT_CI}")

    # --- Input handling ---
    if _is_precomputed(data) and is_corr is not False:
        matrix = pd.DataFrame(data).copy()
    else:
        matrix = compute_association_matrix(data, metric)

    if not isinstance(matrix, pd.DataFrame):
        matrix = pd.DataFrame(matrix)
    if matrix.columns.dtype == np.int64:
        matrix.columns = [f"V{i}" for i in range(len(matrix.columns))]
        matrix.index = matrix.columns

    n = len(matrix)

    # --- col_lim alias ---
    if col_lim is not None:
        if vmin is None:
            vmin = col_lim[0]
        if vmax is None:
            vmax = col_lim[1]

    # --- Range detection ---
    auto_vmin, auto_vmax, auto_vcenter, diverging = _detect_range(matrix, is_corr)
    if vmin is None:
        vmin = auto_vmin
    if vmax is None:
        vmax = auto_vmax
    if vcenter is None:
        vcenter = auto_vcenter
    diverging = vcenter is not None

    # --- Colormap ---
    if cmap is None:
        cmap = "RdBu_r" if diverging else "viridis"
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    norm = _get_norm(vmin, vmax, vcenter, diverging)

    # --- Ordering ---
    matrix, order_idx = _reorder_matrix(matrix, order, hclust_method)

    # Reorder p-values and CI matrices to match
    if pvalues is not None:
        pvalues = np.asarray(pvalues)
        if pvalues.shape == (n, n):
            pvalues = pvalues[np.ix_(order_idx, order_idx)]

    if low_ci is not None:
        low_ci = np.asarray(low_ci)[np.ix_(order_idx, order_idx)]
    if upp_ci is not None:
        upp_ci = np.asarray(upp_ci)[np.ix_(order_idx, order_idx)]

    # --- Figure setup ---
    if fig_ax is not None:
        fig, ax = fig_ax
    else:
        if figsize is None:
            size = max(6, n * 0.6)
            figsize = (size, size)
        fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor=bg)

    ax.set_facecolor(bg)
    ax.set_aspect("equal")

    # --- Determine render mode ---
    has_split = upper is not None and lower is not None
    labels = list(matrix.columns)

    # --- Default tl_pos ---
    if tl_pos is None:
        if type == "full" or has_split:
            tl_pos = "lt"
        elif type == "lower":
            tl_pos = "lb"  # left + bottom for lower triangle
        elif type == "upper":
            tl_pos = "lt"  # left + top for upper triangle

    # --- Default cl_pos ---
    color_only_methods = {"color", "shade"}
    active_methods = {method}
    if upper is not None:
        active_methods.add(upper)
    if lower is not None:
        active_methods.add(lower)
    has_color_only = bool(active_methods & color_only_methods)

    if cl_pos is None:
        cl_pos = "r"
    if not colorbar and not has_color_only:
        cl_pos = "n"

    # --- Default grid_color ---
    if grid_color is None:
        grid_color = "grey" if method != "color" else "white"

    # --- Glyph rendering ---
    patches = []
    colors = []
    shade_lines = []

    for i in range(n):
        for j in range(n):
            # Skip diagonal for glyph rendering (handled separately)
            if i == j:
                continue

            # Determine if cell should be rendered
            if has_split:
                cell_method = _cell_method(i, j, method, upper, lower)
                if cell_method is None:
                    continue
            else:
                if type == "upper" and i > j:
                    continue
                if type == "lower" and i < j:
                    continue
                cell_method = method

            value = matrix.iloc[i, j]

            # NA handling
            if np.isnan(value):
                if na_label == "square":
                    patch = Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=0.5)
                    patch.set_facecolor(na_label_col)
                    patch.set_edgecolor("none")
                    ax.add_patch(patch)
                else:
                    ax.text(
                        j,
                        i,
                        na_label,
                        ha="center",
                        va="center",
                        fontsize=10 * tl_cex,
                        color=na_label_col,
                    )
                continue

            # Handle insig='blank': skip rendering non-significant cells
            if pvalues is not None and insig == "blank":
                p = pvalues[i, j]
                threshold = (
                    sig_level if isinstance(sig_level, (int, float)) else max(sig_level)
                )
                if p > threshold:
                    continue

            # Normalize value for size encoding
            norm_val = np.clip(norm(value), 0, 1)
            abs_norm = np.clip(abs(norm_val - 0.5) * 2 if diverging else norm_val, 0, 1)

            _render_glyph(
                ax,
                j,
                i,
                value,
                norm_val,
                abs_norm,
                cell_method,
                shrink,
                outline,
                outline_col,
                cmap,
                norm,
                addshade,
                shade_lwd,
                shade_col,
                shade_lines,
                patches,
                colors,
                number_cex,
                number_fmt,
            )

    # Add patch collection
    if patches:
        col = PatchCollection(
            patches,
            array=np.array(colors),
            cmap=cmap,
            norm=norm,
            edgecolors=outline_col if outline else "none",
            linewidths=0.5 if outline else 0,
        )
        ax.add_collection(col)

    # Add shade lines
    for line in shade_lines:
        ax.add_line(line)

    # --- Significance overlay ---
    if pvalues is not None and insig != "n" and insig != "blank":
        _draw_significance(
            ax,
            matrix,
            pvalues,
            sig_level,
            insig,
            pch,
            pch_col,
            pch_cex,
            type,
            upper,
            lower,
            has_split,
            n,
            number_cex,
            number_fmt,
            cmap,
            norm,
        )

    # --- Confidence interval overlay ---
    if low_ci is not None and upp_ci is not None and plot_ci != "n":
        _draw_ci(
            ax,
            low_ci,
            upp_ci,
            plot_ci,
            n,
            type,
            upper,
            lower,
            has_split,
            shrink,
            norm,
            outline_col,
        )

    # --- Coefficient overlay ---
    if add_coef:
        _draw_coefficients(
            ax,
            matrix,
            n,
            type,
            upper,
            lower,
            has_split,
            add_coef_col,
            number_cex,
            number_fmt,
            pvalues,
            sig_level,
            insig,
        )

    # --- Diagonal ---
    _draw_diagonal(
        ax,
        matrix,
        diag,
        n,
        labels,
        tl_cex,
        tl_col,
        number_fmt,
        data if not _is_precomputed(data) else None,
        order_idx,
        cmap,
        norm,
        shrink,
    )

    # --- Cluster rectangles ---
    if addrect is not None and order == "hclust":
        _draw_cluster_rects(ax, matrix, addrect, hclust_method, n, rect_col, rect_lwd)

    # --- Grid ---
    if grid:
        _draw_grid(ax, n, type, upper, lower, has_split, grid_color)

    # --- Axis limits ---
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # --- Labels via tick labels (always outside axes) ---
    _draw_labels(
        ax,
        labels,
        n,
        tl_pos,
        tl_cex,
        tl_col,
        tl_srt,
        tl_offset,
        type,
        has_split,
    )

    # --- Colorbar ---
    if cl_pos != "n":
        _draw_colorbar(
            ax,
            fig,
            cmap,
            norm,
            cl_pos,
            cl_cex,
            cl_ratio,
            cl_label,
            vmin,
            vmax,
            diverging,
            patches,
            colors,
        )

    # --- Title ---
    if title:
        fontsize = title_fontsize or 14 * tl_cex
        ax.set_title(title, fontsize=fontsize, pad=20)

    # Only auto-layout figures we created; a user-supplied fig_ax may be part of a
    # larger multi-subplot layout that tight_layout would disrupt.
    if fig_ax is None:
        fig.tight_layout()
    return fig, ax, matrix


def _render_glyph(
    ax,
    x,
    y,
    value,
    norm_val,
    abs_norm,
    method,
    shrink,
    outline,
    outline_col,
    cmap,
    norm,
    addshade,
    shade_lwd,
    shade_col,
    shade_lines,
    patches,
    colors,
    number_cex,
    number_fmt,
):
    """Render a single glyph at position (x, y)."""
    if method == "circle":
        radius = max(0.05, abs_norm) * shrink / 2
        patch = Circle((x, y), radius=radius)
        patches.append(patch)
        colors.append(value)

    elif method == "square":
        side = max(0.05, abs_norm) * shrink
        offset = (1 - side) / 2
        patch = Rectangle((x - 0.5 + offset, y - 0.5 + offset), side, side)
        patches.append(patch)
        colors.append(value)

    elif method == "ellipse":
        rotate = -45 if value >= 0 else 45
        height = shrink - abs_norm * shrink
        height = max(height, 0.05 * shrink)
        patch = Ellipse((x, y), width=1 * shrink, height=height, angle=rotate)
        patches.append(patch)
        colors.append(value)

    elif method == "color":
        patch = Rectangle((x - 0.5, y - 0.5), 1, 1)
        patches.append(patch)
        colors.append(value)

    elif method == "number":
        rgba = cmap(norm(value))
        luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
        color = "black" if luminance > 0.5 else rgba
        text = format(value, number_fmt)
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=10 * number_cex,
            color=color,
            fontweight="bold",
        )

    elif method == "shade":
        patch = Rectangle((x - 0.5, y - 0.5), 1, 1)
        patches.append(patch)
        colors.append(value)
        should_shade = (
            addshade == "all"
            or (addshade == "negative" and value < 0)
            or (addshade == "positive" and value >= 0)
        )
        if should_shade:
            n_lines = 4
            for k in range(n_lines):
                frac = (k + 1) / (n_lines + 1)
                if value < 0:
                    x0, y0 = x - 0.5 + frac, y - 0.5
                    x1, y1 = x - 0.5, y - 0.5 + frac
                else:
                    x0, y0 = x - 0.5, y - 0.5 + (1 - frac)
                    x1, y1 = x - 0.5 + frac, y + 0.5
                line = Line2D(
                    [x0, x1],
                    [y0, y1],
                    color=shade_col,
                    linewidth=shade_lwd,
                    solid_capstyle="butt",
                )
                shade_lines.append(line)

    elif method == "pie":
        angle = 360 * abs_norm
        if value >= 0:
            patch1 = Wedge((x, y), shrink / 2, 90 - angle, 90)
            patch2 = Wedge((x, y), shrink / 2, 90, 90 + (360 - angle))
        else:
            patch1 = Wedge((x, y), shrink / 2, 90, 90 + angle)
            patch2 = Wedge((x, y), shrink / 2, 90 + angle, 90 + 360)
        patches.append(patch1)
        colors.append(value)
        patches.append(patch2)
        colors.append(np.nan)


def _draw_significance(
    ax,
    matrix,
    pvalues,
    sig_level,
    insig,
    pch,
    pch_col,
    pch_cex,
    type_,
    upper,
    lower,
    has_split,
    n,
    number_cex,
    number_fmt,
    cmap,
    norm,
):
    """Draw significance markers."""
    if isinstance(sig_level, (int, float)):
        sig_levels = [sig_level]
    else:
        sig_levels = sorted(sig_level)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if has_split:
                cell_method = _cell_method(i, j, None, upper, lower)
                if cell_method is None:
                    continue
            else:
                if type_ == "upper" and i > j:
                    continue
                if type_ == "lower" and i < j:
                    continue

            p = pvalues[i, j]
            max_sig = sig_levels[-1] if sig_levels else 0.05

            if insig == "pch":
                if p > max_sig:
                    ax.text(
                        j,
                        i,
                        pch,
                        ha="center",
                        va="center",
                        fontsize=8 * pch_cex,
                        color=pch_col,
                        fontweight="bold",
                        zorder=5,
                    )

            elif insig == "p-value":
                if p > max_sig:
                    ax.text(
                        j,
                        i,
                        format(p, number_fmt),
                        ha="center",
                        va="center",
                        fontsize=7 * number_cex,
                        color=pch_col,
                        zorder=5,
                    )

            elif insig == "label_sig":
                if p <= max_sig:
                    stars = _get_stars(p, sig_levels)
                    ax.text(
                        j,
                        i,
                        stars,
                        ha="center",
                        va="center",
                        fontsize=8 * pch_cex,
                        color=pch_col,
                        zorder=5,
                    )


def _get_stars(p, sig_levels):
    """Get significance stars based on p-value and thresholds."""
    sorted_levels = sorted(sig_levels)
    stars = ""
    for level in sorted_levels:
        if p <= level:
            stars += "*"
    return stars if stars else ""


def _draw_ci(
    ax,
    low_ci,
    upp_ci,
    plot_ci,
    n,
    type_,
    upper,
    lower,
    has_split,
    shrink,
    norm,
    outline_col,
):
    """Draw confidence interval overlays."""
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if has_split:
                cell_method = _cell_method(i, j, None, upper, lower)
                if cell_method is None:
                    continue
            else:
                if type_ == "upper" and i > j:
                    continue
                if type_ == "lower" and i < j:
                    continue

            lo = norm(np.clip(low_ci[i, j], norm.vmin, norm.vmax))
            hi = norm(np.clip(upp_ci[i, j], norm.vmin, norm.vmax))
            if np.isnan(lo) or np.isnan(hi):
                continue
            lo_size = np.clip(abs(lo - 0.5) * 2, 0, 1)
            hi_size = np.clip(abs(hi - 0.5) * 2, 0, 1)

            if plot_ci == "circle":
                outer_r = max(0.05, hi_size) * shrink / 2
                inner_r = max(0.02, lo_size) * shrink / 2
                outer = Circle(
                    (j, i),
                    radius=outer_r,
                    fill=False,
                    edgecolor=outline_col,
                    linewidth=1,
                    linestyle="--",
                    zorder=4,
                )
                inner = Circle(
                    (j, i),
                    radius=inner_r,
                    fill=False,
                    edgecolor=outline_col,
                    linewidth=1,
                    linestyle=":",
                    zorder=4,
                )
                ax.add_patch(outer)
                ax.add_patch(inner)

            elif plot_ci == "square":
                outer_s = max(0.05, hi_size) * shrink
                inner_s = max(0.02, lo_size) * shrink
                outer_off = (1 - outer_s) / 2
                inner_off = (1 - inner_s) / 2
                outer = Rectangle(
                    (j - 0.5 + outer_off, i - 0.5 + outer_off),
                    outer_s,
                    outer_s,
                    fill=False,
                    edgecolor=outline_col,
                    linewidth=1,
                    linestyle="--",
                    zorder=4,
                )
                inner = Rectangle(
                    (j - 0.5 + inner_off, i - 0.5 + inner_off),
                    inner_s,
                    inner_s,
                    fill=False,
                    edgecolor=outline_col,
                    linewidth=1,
                    linestyle=":",
                    zorder=4,
                )
                ax.add_patch(outer)
                ax.add_patch(inner)

            elif plot_ci == "rect":
                rect_h = 0.6 * shrink
                rect_w = (hi_size - lo_size) * shrink
                rect_x = j - rect_w / 2
                rect_y = i - rect_h / 2
                rect = Rectangle(
                    (rect_x, rect_y),
                    max(rect_w, 0.05),
                    rect_h,
                    fill=False,
                    edgecolor=outline_col,
                    linewidth=1.5,
                    linestyle="-",
                    zorder=4,
                )
                ax.add_patch(rect)


def _draw_coefficients(
    ax,
    matrix,
    n,
    type_,
    upper,
    lower,
    has_split,
    add_coef_col,
    number_cex,
    number_fmt,
    pvalues,
    sig_level,
    insig,
):
    """Overlay coefficient numbers on glyphs."""
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if has_split:
                cell_method = _cell_method(i, j, None, upper, lower)
                if cell_method is None:
                    continue
            else:
                if type_ == "upper" and i > j:
                    continue
                if type_ == "lower" and i < j:
                    continue

            # Skip if insig='label_sig' and not significant
            if pvalues is not None and insig == "label_sig":
                p = pvalues[i, j]
                threshold = (
                    sig_level if isinstance(sig_level, (int, float)) else max(sig_level)
                )
                if p > threshold:
                    continue

            value = matrix.iloc[i, j]
            if np.isnan(value):
                continue

            text = format(value, number_fmt)
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=8 * number_cex,
                color=add_coef_col,
                zorder=6,
            )


def _draw_diagonal(
    ax,
    matrix,
    diag,
    n,
    labels,
    tl_cex,
    tl_col,
    number_fmt,
    raw_data,
    order_idx,
    cmap,
    norm,
    shrink,
):
    """Render diagonal cells."""
    if diag == "blank":
        return

    for i in range(n):
        if diag == "names":
            ax.text(
                i,
                i,
                labels[i],
                ha="center",
                va="center",
                fontsize=10 * tl_cex,
                color=tl_col,
                fontweight="bold",
            )
        elif diag == "values":
            value = matrix.iloc[i, i]
            text = format(value, number_fmt)
            color = cmap(norm(value))
            ax.text(
                i,
                i,
                text,
                ha="center",
                va="center",
                fontsize=9 * tl_cex,
                color=color,
                fontweight="bold",
            )
        elif diag == "hist":
            if raw_data is not None and isinstance(raw_data, pd.DataFrame):
                orig_col_idx = order_idx[i]
                col_data = raw_data.iloc[:, orig_col_idx].dropna().values
                pad = 0.05
                inset_ax = ax.inset_axes(
                    [i - 0.5 + pad, i - 0.5 + pad, 1 - 2 * pad, 1 - 2 * pad],
                    transform=ax.transData,
                )
                inset_ax.hist(col_data, bins=15, color="steelblue", edgecolor="none")
                inset_ax.set_xticks([])
                inset_ax.set_yticks([])
                for spine in inset_ax.spines.values():
                    spine.set_visible(False)
            # No raw data — leave diagonal blank (names already on tick labels)
            else:
                pass


def _draw_cluster_rects(ax, matrix, addrect, hclust_method, n, rect_col, rect_lwd):
    """Draw rectangles around hierarchical clusters."""
    d = distance.pdist(matrix.values)
    d = np.nan_to_num(d, nan=0.0)
    Y = hierarchy.linkage(d, method=hclust_method)
    clusters = hierarchy.fcluster(Y, t=addrect, criterion="maxclust")

    seen = {}
    for idx, cl in enumerate(clusters):
        if cl not in seen:
            seen[cl] = [idx, idx]
        else:
            seen[cl][0] = min(seen[cl][0], idx)
            seen[cl][1] = max(seen[cl][1], idx)

    for cl, (start, end) in seen.items():
        rect = Rectangle(
            (start - 0.5, start - 0.5),
            end - start + 1,
            end - start + 1,
            fill=False,
            edgecolor=rect_col,
            linewidth=rect_lwd,
            zorder=7,
        )
        ax.add_patch(rect)


def _draw_grid(ax, n, type_, upper, lower, has_split, grid_color):
    """Draw grid lines between cells, clipped to triangle for upper/lower."""
    if type_ == "full" or has_split:
        for i in range(n + 1):
            ax.axhline(i - 0.5, color=grid_color, linewidth=0.5, zorder=1)
            ax.axvline(i - 0.5, color=grid_color, linewidth=0.5, zorder=1)
    elif type_ == "upper":
        for i in range(n + 1):
            ax.plot(
                [i - 0.5, n - 0.5],
                [i - 0.5, i - 0.5],
                color=grid_color,
                linewidth=0.5,
                zorder=1,
            )
            ax.plot(
                [i - 0.5, i - 0.5],
                [-0.5, i - 0.5],
                color=grid_color,
                linewidth=0.5,
                zorder=1,
            )
    elif type_ == "lower":
        for i in range(n + 1):
            ax.plot(
                [-0.5, i - 0.5],
                [i - 0.5, i - 0.5],
                color=grid_color,
                linewidth=0.5,
                zorder=1,
            )
            ax.plot(
                [i - 0.5, i - 0.5],
                [i - 0.5, n - 0.5],
                color=grid_color,
                linewidth=0.5,
                zorder=1,
            )


def _draw_labels(
    ax,
    labels,
    n,
    tl_pos,
    tl_cex,
    tl_col,
    tl_srt,
    tl_offset,
    type_,
    has_split,
):
    """Draw axis labels using matplotlib tick machinery (always outside axes)."""
    base_fontsize = 10 * tl_cex
    ticks = list(range(n))

    if tl_pos == "n":
        ax.set_xticks([])
        ax.set_yticks([])
        return

    # Left (y-axis) labels
    if tl_pos in ("lt", "l", "lb"):
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, fontsize=base_fontsize, color=tl_col)
        ax.tick_params(axis="y", length=0, pad=4)
    else:
        ax.set_yticks([])

    # Top (x-axis) labels
    if tl_pos in ("lt", "td"):
        ax.set_xticks(ticks)
        ax.xaxis.tick_top()
        ax.set_xticklabels(
            labels,
            rotation=tl_srt,
            ha="left",
            rotation_mode="anchor",
            fontsize=base_fontsize,
            color=tl_col,
        )
        ax.tick_params(axis="x", length=0, pad=4)
    elif tl_pos == "lb":
        # Bottom x-axis for lower triangle
        ax.set_xticks(ticks)
        ax.xaxis.tick_bottom()
        ax.set_xticklabels(
            labels,
            rotation=tl_srt,
            ha="right",
            rotation_mode="anchor",
            fontsize=base_fontsize,
            color=tl_col,
        )
        ax.tick_params(axis="x", length=0, pad=4)
    else:
        ax.set_xticks([])


def _draw_colorbar(
    ax,
    fig,
    cmap,
    norm,
    cl_pos,
    cl_cex,
    cl_ratio,
    cl_label,
    vmin,
    vmax,
    diverging,
    patches,
    colors,
):
    """Draw the color legend."""
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    if cl_pos == "r":
        cbar = fig.colorbar(
            sm,
            ax=ax,
            orientation="vertical",
            fraction=cl_ratio,
            pad=0.02,
            shrink=0.8,
        )
    elif cl_pos == "b":
        cbar = fig.colorbar(
            sm,
            ax=ax,
            orientation="horizontal",
            fraction=cl_ratio,
            pad=0.08,
            shrink=0.8,
        )
    else:
        return

    cbar.ax.tick_params(labelsize=9 * cl_cex)
    if cl_label:
        cbar.set_label(cl_label, fontsize=10 * cl_cex)
