# Copyright 2026 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Behavioural tests for ``causalts.plotting.corrplot`` (glyphs, overlays,
ordering, labels, colorbar) and the ``corr_table`` helpers (Agg backend)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.collections import PatchCollection  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy import stats  # noqa: E402

from causalts.plotting import (  # noqa: E402
    compute_association_matrix,
    corr_table,
    corrplot,
)
from causalts.plotting.corrplot import (  # noqa: E402
    _apa_stars,
    _cell_method,
    _compute_pvalue_matrix,
    _detect_range,
    _fisher_ci,
    _format_p,
    _get_stars,
    _pairwise_n,
    _render_multiline_table,
    _reorder_matrix,
)

LABELS = ["a", "b", "c"]
# Off-diagonal cells in row-major order: (0,1)=.5, (0,2)=-.3, (1,0)=.5,
# (1,2)=.8, (2,0)=-.3, (2,1)=.8.
VALUES = np.array([[1.0, 0.5, -0.3], [0.5, 1.0, 0.8], [-0.3, 0.8, 1.0]])
# (0,1) is insignificant at 0.05; every other pair has p = 0.01.
PVALS = np.array([[0.0, 0.2, 0.01], [0.2, 0.0, 0.01], [0.01, 0.01, 0.0]])


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _mat():
    return pd.DataFrame(VALUES.copy(), index=LABELS, columns=LABELS)


def _plot(data=None, **kwargs):
    kwargs.setdefault("diag", "blank")
    kwargs.setdefault("colorbar", False)
    kwargs.setdefault("grid", False)
    return corrplot(_mat() if data is None else data, **kwargs)


def _collection(ax):
    cols = [c for c in ax.collections if isinstance(c, PatchCollection)]
    assert len(cols) == 1
    return cols[0]


def _n_glyphs(ax):
    cols = [c for c in ax.collections if isinstance(c, PatchCollection)]
    return sum(len(c.get_paths()) for c in cols)


def _texts(ax):
    return [t.get_text() for t in ax.texts]


# --------------------------------------------------------------------------
# compute_association_matrix
# --------------------------------------------------------------------------


def test_compute_association_matrix_callable_metric_hand_computed():
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    m = compute_association_matrix(data, metric=lambda x, y: float(x @ y))
    assert list(m.columns) == [0, 1]
    np.testing.assert_allclose(m.values, [[35.0, 44.0], [44.0, 56.0]])


def test_compute_association_matrix_rejects_non_callable_metric():
    with pytest.raises(TypeError, match="string preset or callable"):
        compute_association_matrix(np.zeros((3, 2)), metric=3)


# --------------------------------------------------------------------------
# private helpers: range detection, ordering, cell method
# --------------------------------------------------------------------------


def test_detect_range_branches():
    corr_like = np.array([[1.0, -0.5], [-0.5, 1.0]])
    assert _detect_range(corr_like, True) == (-1.0, 1.0, 0.0, True)
    assert _detect_range(corr_like, None) == (-1.0, 1.0, 0.0, True)
    assert _detect_range(corr_like, False) == (-0.5, 1.0, None, False)
    positive = np.array([[0.2, 3.0], [3.0, 0.2]])
    assert _detect_range(positive, None) == (0.2, 3.0, None, False)
    wide = pd.DataFrame([[2.0, -3.0], [-3.0, 2.0]])
    assert _detect_range(wide, None) == (-3.0, 2.0, 0.0, True)


def test_reorder_matrix_alphabet_and_original():
    m = pd.DataFrame(VALUES, index=["c", "a", "b"], columns=["c", "a", "b"])
    out, idx = _reorder_matrix(m, "alphabet", "complete")
    assert list(out.columns) == ["a", "b", "c"]
    assert list(out.index) == ["a", "b", "c"]
    assert list(idx) == [1, 2, 0]
    assert out.loc["a", "c"] == m.loc["a", "c"]
    same, idx0 = _reorder_matrix(m, "original", "complete")
    assert same is m and list(idx0) == [0, 1, 2]


def _block_matrix():
    # {a, b} and {c, d} form two tight blocks; listed interleaved.
    names = ["a", "c", "b", "d"]
    block = {"a": 0, "b": 0, "c": 1, "d": 1}
    v = np.array(
        [
            [1.0 if x == y else (0.9 if block[x] == block[y] else 0.0) for y in names]
            for x in names
        ]
    )
    return pd.DataFrame(v, index=names, columns=names)


@pytest.mark.parametrize("order", ["hclust", "FPC", "AOE"])
def test_reorder_matrix_returns_consistent_permutation(order):
    m = _block_matrix()
    out, idx = _reorder_matrix(m, order, "average")
    assert sorted(idx) == [0, 1, 2, 3]
    np.testing.assert_array_equal(out.values, m.values[np.ix_(idx, idx)])
    assert list(out.columns) == list(out.index)


def test_reorder_matrix_hclust_makes_blocks_contiguous():
    out, _ = _reorder_matrix(_block_matrix(), "hclust", "average")
    first_two = set(out.columns[:2])
    assert first_two in ({"a", "b"}, {"c", "d"})


def test_reorder_matrix_unknown_order_raises():
    with pytest.raises(ValueError, match="Unknown order method"):
        _reorder_matrix(_mat(), "random", "complete")


def test_cell_method_split_and_plain():
    assert _cell_method(0, 1, "circle", "pie", "number") == "pie"
    assert _cell_method(1, 0, "circle", "pie", "number") == "number"
    assert _cell_method(1, 1, "circle", "pie", "number") is None
    assert _cell_method(1, 1, "circle", None, "number") == "circle"


# --------------------------------------------------------------------------
# corrplot: validation and input handling
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"lower": "bogus"}, "Unknown lower method"),
        ({"upper": "bogus"}, "Unknown upper method"),
        ({"insig": "bogus"}, "Unknown insig"),
        ({"plot_ci": "bogus"}, "Unknown plot_ci"),
        ({"order": "bogus"}, "Unknown order method"),
    ],
)
def test_corrplot_rejects_invalid_options(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _plot(**kwargs)


def test_corrplot_unlabelled_square_ndarray_gets_v_labels():
    fig, ax, matrix = _plot(VALUES.copy())
    assert list(matrix.columns) == ["V0", "V1", "V2"]
    assert list(matrix.index) == ["V0", "V1", "V2"]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["V0", "V1", "V2"]


def test_corrplot_callable_metric_on_raw_data():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.standard_normal((40, 3)), columns=list("xyz"))
    _, _, matrix = _plot(df, metric=lambda x, y: float(np.mean(x * y)))
    expected = df.T.values @ df.values / len(df)
    np.testing.assert_allclose(matrix.values, expected)


def test_corrplot_col_lim_and_vcenter_set_the_norm():
    _, ax, _ = _plot(col_lim=(-0.5, 0.5))
    norm = _collection(ax).norm
    assert (norm.vmin, norm.vmax) == (-0.5, 0.5)
    # An explicit vmin wins over col_lim.
    _, ax, _ = _plot(col_lim=(-0.5, 0.5), vmin=-0.8)
    assert _collection(ax).norm.vmin == -0.8


def test_corrplot_is_corr_false_uses_sequential_data_range():
    _, ax, _ = _plot(is_corr=False)
    coll = _collection(ax)
    assert coll.cmap.name == "viridis"
    assert (coll.norm.vmin, coll.norm.vmax) == (-0.3, 1.0)
    _, ax, _ = _plot(cmap="magma")
    assert _collection(ax).cmap.name == "magma"


def test_corrplot_alphabet_order_reorders_matrix_and_pvalues():
    names = ["c", "a", "b"]
    m = pd.DataFrame(VALUES.copy(), index=names, columns=names)
    # Only the c-a pair is insignificant.
    _, ax, matrix = _plot(m, order="alphabet", pvalues=PVALS, insig="pch")
    assert list(matrix.columns) == ["a", "b", "c"]
    marks = [t.get_position() for t in ax.texts if t.get_text() == "x"]
    # c-a sits at positions (a=0, c=2) after reordering.
    assert sorted(marks) == [(0, 2), (2, 0)]


# --------------------------------------------------------------------------
# corrplot: glyph methods
# --------------------------------------------------------------------------


def test_corrplot_circle_radius_scales_with_abs_value():
    _, ax, _ = _plot(method="circle", shrink=0.9)
    coll = _collection(ax)
    np.testing.assert_allclose(coll.get_array(), [0.5, -0.3, 0.5, 0.8, -0.3, 0.8])
    widths = [p.get_extents().width for p in coll.get_paths()]
    # radius = |r| * shrink / 2 on the symmetric [-1, 1] scale.
    np.testing.assert_allclose(widths, [0.45, 0.27, 0.45, 0.72, 0.27, 0.72], atol=1e-3)


def test_corrplot_square_side_scales_with_abs_value():
    _, ax, _ = _plot(method="square", shrink=1.0)
    sides = [p.get_extents().width for p in _collection(ax).get_paths()]
    np.testing.assert_allclose(sides, [0.5, 0.3, 0.5, 0.8, 0.3, 0.8], atol=1e-6)


@pytest.mark.parametrize("method, per_cell", [("ellipse", 1), ("color", 1), ("pie", 2)])
def test_corrplot_patch_methods_draw_one_glyph_set_per_cell(method, per_cell):
    _, ax, _ = _plot(method=method)
    assert _n_glyphs(ax) == 6 * per_cell


def test_corrplot_color_cells_fill_the_unit_cell():
    _, ax, _ = _plot(method="color")
    for p in _collection(ax).get_paths():
        ext = p.get_extents()
        assert ext.width == pytest.approx(1.0) and ext.height == pytest.approx(1.0)


def test_corrplot_pie_second_wedge_is_uncoloured():
    _, ax, _ = _plot(method="pie")
    arr = np.asarray(_collection(ax).get_array(), dtype=float)
    np.testing.assert_allclose(arr[0::2], [0.5, -0.3, 0.5, 0.8, -0.3, 0.8])
    assert np.all(np.isnan(arr[1::2]))


def test_corrplot_number_method_writes_formatted_values():
    _, ax, _ = _plot(method="number", number_fmt=".1f")
    assert sorted(_texts(ax)) == sorted(["0.5", "-0.3", "0.5", "0.8", "-0.3", "0.8"])
    assert not [c for c in ax.collections if isinstance(c, PatchCollection)]


@pytest.mark.parametrize(
    "addshade, n_shaded_cells", [("all", 6), ("negative", 2), ("positive", 4)]
)
def test_corrplot_shade_lines_follow_addshade(addshade, n_shaded_cells):
    _, ax, _ = _plot(method="shade", addshade=addshade, shade_col="red")
    shade = [ln for ln in ax.lines if ln.get_color() == "red"]
    assert len(shade) == 4 * n_shaded_cells
    assert _n_glyphs(ax) == 6


def test_corrplot_upper_lower_types_keep_one_triangle():
    _, ax, _ = _plot(type="upper")
    assert _n_glyphs(ax) == 3
    assert [
        p.get_extents().x0 > p.get_extents().y0 for p in _collection(ax).get_paths()
    ] == [True] * 3
    _, ax, _ = _plot(type="lower")
    assert _n_glyphs(ax) == 3
    assert all(
        p.get_extents().x0 < p.get_extents().y0 for p in _collection(ax).get_paths()
    )


def test_corrplot_split_methods_per_triangle():
    _, ax, _ = _plot(upper="circle", lower="number")
    assert _n_glyphs(ax) == 3  # circles in the upper triangle
    assert sorted(_texts(ax)) == ["-0.30", "0.50", "0.80"]  # numbers below


# --------------------------------------------------------------------------
# corrplot: NaN cells
# --------------------------------------------------------------------------


def _nan_mat():
    m = _mat()
    m.loc["a", "b"] = m.loc["b", "a"] = np.nan
    return m


def test_corrplot_nan_cells_get_question_mark_labels():
    _, ax, _ = _plot(_nan_mat(), na_label_col="purple")
    labels = [t for t in ax.texts if t.get_text() == "?"]
    assert sorted(t.get_position() for t in labels) == [(0, 1), (1, 0)]
    assert all(t.get_color() == "purple" for t in labels)
    assert _n_glyphs(ax) == 4


def test_corrplot_nan_cells_as_filled_squares():
    _, ax, _ = _plot(_nan_mat(), na_label="square", na_label_col="black")
    squares = [p for p in ax.patches if isinstance(p, Rectangle)]
    assert sorted(p.get_xy() for p in squares) == [(-0.5, 0.5), (0.5, -0.5)]
    assert "?" not in _texts(ax)


# --------------------------------------------------------------------------
# corrplot: significance overlays
# --------------------------------------------------------------------------


def test_corrplot_insig_blank_skips_insignificant_cells():
    _, ax, _ = _plot(pvalues=PVALS, insig="blank")
    assert _n_glyphs(ax) == 4
    # A list of levels uses the loosest one (0.3 keeps every cell).
    _, ax, _ = _plot(pvalues=PVALS, insig="blank", sig_level=[0.01, 0.3])
    assert _n_glyphs(ax) == 6


def test_corrplot_insig_pch_marks_insignificant_pairs():
    _, ax, _ = _plot(pvalues=PVALS, insig="pch", pch="#")
    marks = [t.get_position() for t in ax.texts if t.get_text() == "#"]
    assert sorted(marks) == [(0, 1), (1, 0)]
    _, ax, _ = _plot(pvalues=PVALS, insig="pch", type="upper")
    assert [t.get_position() for t in ax.texts if t.get_text() == "x"] == [(1, 0)]


def test_corrplot_insig_p_value_prints_p():
    _, ax, _ = _plot(pvalues=PVALS, insig="p-value", number_fmt=".2f")
    assert _texts(ax).count("0.20") == 2
    assert len(ax.texts) == 2


def test_corrplot_label_sig_stars_and_coefficients():
    _, ax, _ = _plot(
        pvalues=PVALS,
        insig="label_sig",
        sig_level=[0.05, 0.001, 0.01],
        add_coef=True,
    )
    texts = _texts(ax)
    # p = 0.01 meets the 0.01 and 0.05 levels -> two stars on 4 cells.
    assert texts.count("**") == 4
    # Coefficients are written only on significant cells.
    assert sorted(t for t in texts if t != "**") == ["-0.30", "-0.30", "0.80", "0.80"]


def test_corrplot_insig_n_draws_no_markers():
    _, ax, _ = _plot(pvalues=PVALS, insig="n")
    assert len(ax.texts) == 0
    assert _n_glyphs(ax) == 6


def test_get_stars_counts_levels_met():
    assert _get_stars(0.0005, [0.05, 0.01, 0.001]) == "***"
    assert _get_stars(0.02, [0.05, 0.01, 0.001]) == "*"
    assert _get_stars(0.2, [0.05]) == ""


# --------------------------------------------------------------------------
# corrplot: coefficients and diagonal
# --------------------------------------------------------------------------


def test_corrplot_add_coef_skips_nan_and_honours_type():
    _, ax, _ = _plot(_nan_mat(), add_coef=True, add_coef_col="green")
    coefs = [t for t in ax.texts if t.get_color() == "green"]
    assert sorted(t.get_text() for t in coefs) == ["-0.30", "-0.30", "0.80", "0.80"]
    _, ax, _ = _plot(add_coef=True, type="lower", add_coef_col="green")
    coefs = [t for t in ax.texts if t.get_color() == "green"]
    assert sorted(t.get_position() for t in coefs) == [(0, 1), (0, 2), (1, 2)]


def test_corrplot_add_coef_on_split_and_glyph_diagonal():
    _, ax, _ = _plot(upper="circle", lower="square", add_coef=True, add_coef_col="g")
    assert len([t for t in ax.texts if t.get_color() == "g"]) == 6
    _, ax, _ = _plot(diag="glyph", add_coef=True, add_coef_col="g")
    coefs = [t.get_text() for t in ax.texts if t.get_color() == "g"]
    assert coefs.count("1.00") == 3 and len(coefs) == 9


def test_corrplot_diag_values_and_names():
    _, ax, _ = _plot(diag="values", number_fmt=".1f")
    diag = [t for t in ax.texts if t.get_position()[0] == t.get_position()[1]]
    assert [t.get_text() for t in diag] == ["1.0", "1.0", "1.0"]
    # Coloured by the colormap at +1 (the dark-red end of RdBu_r).
    assert diag[0].get_color() == plt.get_cmap("RdBu_r")(1.0)

    _, ax, _ = _plot(diag="names", tl_col="navy")
    assert _texts(ax) == LABELS
    assert all(t.get_color() == "navy" for t in ax.texts)


def test_corrplot_diag_hist_draws_one_inset_per_variable_for_raw_data():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.standard_normal((50, 3)), columns=list("xyz"))
    _, ax, _ = _plot(df, diag="hist")
    assert len(ax.child_axes) == 3
    counts = [sum(p.get_height() for p in inset.patches) for inset in ax.child_axes]
    assert counts == [50, 50, 50]
    # A precomputed matrix has no raw column to histogram.
    _, ax, _ = _plot(diag="hist")
    assert len(ax.child_axes) == 0


# --------------------------------------------------------------------------
# corrplot: confidence-interval overlay
# --------------------------------------------------------------------------


def _ci():
    low = VALUES - 0.2
    upp = VALUES + 0.1
    low[0, 1] = np.nan  # this cell's CI is skipped
    return low, upp


@pytest.mark.parametrize(
    "plot_ci, per_cell", [("circle", 2), ("square", 2), ("rect", 1)]
)
def test_corrplot_ci_overlay_patch_counts(plot_ci, per_cell):
    low, upp = _ci()
    _, ax, _ = _plot(low_ci=low, upp_ci=upp, plot_ci=plot_ci, outline_col="blue")
    ci = [p for p in ax.patches if p.get_zorder() == 4]
    assert len(ci) == 5 * per_cell
    assert all(not p.get_fill() for p in ci)


def test_corrplot_ci_rect_width_is_interval_length():
    low = np.full((3, 3), 0.2)
    upp = np.full((3, 3), 0.6)
    _, ax, _ = _plot(low_ci=low, upp_ci=upp, plot_ci="rect", shrink=1.0, type="upper")
    rects = [p for p in ax.patches if p.get_zorder() == 4]
    assert len(rects) == 3
    assert all(r.get_width() == pytest.approx(0.4) for r in rects)
    assert all(r.get_height() == pytest.approx(0.6) for r in rects)


def test_corrplot_ci_respects_split_triangles():
    low, upp = VALUES - 0.1, VALUES + 0.1
    _, ax, _ = _plot(
        upper="circle", lower="number", low_ci=low, upp_ci=upp, plot_ci="circle"
    )
    assert len([p for p in ax.patches if p.get_zorder() == 4]) == 12


def test_corrplot_ci_ignored_when_plot_ci_n():
    low, upp = _ci()
    _, ax, _ = _plot(low_ci=low, upp_ci=upp)
    assert [p for p in ax.patches if p.get_zorder() == 4] == []


# --------------------------------------------------------------------------
# corrplot: cluster rectangles, grid, labels, colorbar, title
# --------------------------------------------------------------------------


def test_corrplot_addrect_outlines_hclust_blocks():
    _, ax, matrix = _plot(
        _block_matrix(),
        order="hclust",
        hclust_method="average",
        addrect=2,
        rect_col="red",
    )
    rects = [p for p in ax.patches if p.get_zorder() == 7]
    assert len(rects) == 2
    assert sorted(r.get_width() for r in rects) == [2, 2]
    assert all(r.get_edgecolor()[:3] == (1.0, 0.0, 0.0) for r in rects)
    # addrect without hclust ordering is ignored.
    _, ax, _ = _plot(_block_matrix(), addrect=2)
    assert [p for p in ax.patches if p.get_zorder() == 7] == []


@pytest.mark.parametrize("type_", ["upper", "lower"])
def test_corrplot_triangle_grid_draws_staircase_lines(type_):
    _, ax, _ = _plot(type=type_, grid=True, grid_color="orange")
    grid = [ln for ln in ax.lines if ln.get_color() == "orange"]
    assert len(grid) == 2 * (3 + 1)


def test_corrplot_default_grid_colour_for_color_method_is_white():
    _, ax, _ = _plot(method="color", grid=True)
    assert {ln.get_color() for ln in ax.lines} == {"white"}


def _tick_texts(labels):
    return [t.get_text() for t in labels]


def test_corrplot_label_positions():
    _, ax, _ = _plot(tl_pos="n")
    assert list(ax.get_xticks()) == [] and list(ax.get_yticks()) == []

    _, ax, _ = _plot(tl_pos="td")
    assert _tick_texts(ax.get_xticklabels()) == LABELS
    assert list(ax.get_yticks()) == []

    _, ax, _ = _plot(tl_pos="l")
    assert _tick_texts(ax.get_yticklabels()) == LABELS
    assert list(ax.get_xticks()) == []


def test_corrplot_lower_type_defaults_to_bottom_labels():
    _, ax, _ = _plot(type="lower")
    assert _tick_texts(ax.get_xticklabels()) == LABELS
    assert ax.xaxis.get_ticks_position() == "bottom"
    _, ax, _ = _plot(type="upper")
    assert ax.xaxis.get_ticks_position() == "top"


@pytest.mark.parametrize("cl_pos, horizontal", [("r", False), ("b", True)])
def test_corrplot_colorbar_position_and_label(cl_pos, horizontal):
    fig, ax, _ = _plot(colorbar=True, cl_pos=cl_pos, cl_label="rho")
    assert len(fig.axes) == 2
    cax = fig.axes[1]
    label = cax.get_xlabel() if horizontal else cax.get_ylabel()
    assert label == "rho"
    bbox = cax.get_position()
    assert bool(bbox.width > bbox.height) is horizontal


def test_corrplot_colorbar_n_draws_none():
    fig, _, _ = _plot(colorbar=True, cl_pos="n")
    assert len(fig.axes) == 1


def test_corrplot_title_and_fontsize():
    _, ax, _ = _plot(title="Corr", title_fontsize=11)
    assert ax.get_title() == "Corr"
    assert ax.title.get_fontsize() == 11


# --------------------------------------------------------------------------
# corr_table helpers
# --------------------------------------------------------------------------


def test_apa_stars_thresholds():
    assert [_apa_stars(p) for p in (0.0005, 0.005, 0.03, 0.2)] == [
        "***",
        "**",
        "*",
        "",
    ]
    assert _apa_stars(None) == "" and _apa_stars(float("nan")) == ""


def test_format_p_apa_style():
    assert _format_p(0.0004) == "p < .001"
    assert _format_p(0.019) == "p = .019"
    assert _format_p(None) is None and _format_p(np.nan) is None


def test_fisher_ci_small_n_and_unknown_metric():
    assert _fisher_ci(0.5, 3, "pearson") is None
    assert _fisher_ci(0.5, 3, "spearman") is None
    assert _fisher_ci(0.5, 4, "kendall") is None
    assert _fisher_ci(0.5, 30, "dcor") is None
    assert _fisher_ci(0.5, None) is None
    assert _fisher_ci(1.0, 30) is None
    lo, hi = _fisher_ci(0.5, 28, "pearson")
    z = np.arctanh(0.5)
    zc = stats.norm.ppf(0.975)
    assert lo == pytest.approx(np.tanh(z - zc / 5))
    assert hi == pytest.approx(np.tanh(z + zc / 5))
    lo_k, hi_k = _fisher_ci(0.5, 28, "kendall")
    se_k = np.sqrt(0.437 / 24)
    assert lo_k == pytest.approx(np.tanh(z - zc * se_k))


def test_compute_pvalue_matrix_matches_scipy_and_handles_sparse_pairs():
    rng = np.random.default_rng(2)
    df = pd.DataFrame(rng.standard_normal((20, 3)), columns=list("xyz"))
    df.loc[2:, "z"] = np.nan  # only 2 complete rows with z
    p = _compute_pvalue_matrix(df, "pearson")
    assert p.loc["x", "y"] == pytest.approx(stats.pearsonr(df.x, df.y)[1])
    assert p.loc["y", "x"] == p.loc["x", "y"]
    assert np.isnan(p.loc["x", "z"]) and np.isnan(p.loc["z", "y"])
    assert np.all(np.diag(p.values) == 0.0)
    ps = _compute_pvalue_matrix(df[["x", "y"]], "spearman")
    assert ps.loc["x", "y"] == pytest.approx(stats.spearmanr(df.x, df.y)[1])


def test_pairwise_n_counts_complete_rows():
    df = pd.DataFrame({"x": [1, 2, np.nan, 4], "y": [1, np.nan, np.nan, 4]})
    n = _pairwise_n(df)
    np.testing.assert_array_equal(n.values, [[3, 2], [2, 2]])


def test_render_multiline_table_top_aligns_cells():
    frame = pd.DataFrame({"A": ["1\n2", "x"], "B": ["y", "z"]}, index=["r1", "r2"])
    lines = _render_multiline_table(frame).split("\n")
    assert lines == ["    A  B", "r1  1  y", "    2   ", "r2  x  z"]


def _table_frame():
    rng = np.random.default_rng(3)
    z = rng.standard_normal(60)
    return pd.DataFrame(
        {
            "u": z + rng.standard_normal(60),
            "v": z + rng.standard_normal(60),
            "w": rng.standard_normal(60),
        }
    )


def test_corr_table_show_ci_adds_interval_and_p_lines():
    df = _table_frame()
    table = corr_table(df, show_ci=True)
    cell = table.to_frame().loc["2. v", 1]
    r = df.corr().loc["v", "u"]
    lo, hi = _fisher_ci(r, 60)
    p = stats.pearsonr(df.u, df.v)[1]
    lines = cell.split("\n")
    assert lines[0].startswith(f"{r:.2f}")
    assert lines[1] == f"[{lo:.2f}, {hi:.2f}]"
    assert lines[2] == _format_p(p)
    text = repr(table)
    assert "Values in square brackets indicate the 95% CI." in text
    assert f"[{lo:.2f}, {hi:.2f}]" in text


def test_corr_table_notes_false_and_no_stars_footnote():
    df = _table_frame()
    assert "Note." not in repr(corr_table(df, notes=False))
    assert "Note." not in corr_table(df, notes=False)._repr_html_()
    # Nothing to explain -> no footnote even with notes=True.
    plain = corr_table(df, show_n=False, sig_stars=False)
    assert "Note." not in repr(plain)
    assert "<em>Note.</em>" in corr_table(df)._repr_html_()


def test_corr_table_pvalues_frame_aligned_by_label():
    df = _table_frame()
    cols = ["w", "v", "u"]  # deliberately reordered
    pv = pd.DataFrame(0.5, index=cols, columns=cols)
    pv.loc["u", "v"] = pv.loc["v", "u"] = 0.0001
    frame = corr_table(df, pvalues=pv).to_frame()
    assert frame.loc["2. v", 1].endswith("***")
    assert not frame.loc["3. w", 1].endswith("*")
