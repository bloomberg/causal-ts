import numpy as np
import pandas as pd
import pytest

from causalts.cdnots.lag_refine import refine_lagged
from causalts.cdnots.phase3_utils import (
    finish_cdnots_plus,
    make_background_knowledge,
    run_cdnots_plus,
)
from causalts.cdnots.skeleton_discovery import initialize_graph
from causalts.ci_tests.parcorr_gpu import ParCorrGPU


def _masked(seed=0, T=500, b=0.25):
    """X(t-1) -> Y(t) weak next to Y's strong autocorrelation.

    The marginal correlation is ~b / sd(Y) ~ 0.08, so phase 1 drops the link at
    depth 0; given Y(t-1) its partial correlation is ~0.24.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=T)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.95 * y[t - 1] + b * x[t - 1] + rng.normal()
    return pd.DataFrame({"X": x, "Y": y})


def _run(df, ci=None, **kw):
    return run_cdnots_plus(
        df,
        ci if ci is not None else ParCorrGPU(df.values, device="cpu"),
        num_lags=1,
        include_C=False,
        alpha=0.01,
        show_progress=False,
        **kw,
    )


def test_default_is_unchanged_and_records_nothing():
    df = _masked()
    base = _run(df)
    off = _run(df, alpha_mci=None, addback_alpha=None, veto_alpha=None)
    np.testing.assert_array_equal(np.asarray(base.cg_tig), np.asarray(off.cg_tig))
    assert off.lag_refinements is None


def test_addback_recovers_a_link_dropped_at_depth_0():
    df = _masked()
    assert np.asarray(_run(df).cg_tig)[0, 1, 1] == 0
    res = _run(df, addback_alpha=1e-4, veto_alpha=1e-3)
    assert np.asarray(res.cg_tig)[0, 1, 1] == 1
    assert res.to_marks()[0, 1, 1] == "-->"
    (ch,) = res.lag_refinements
    assert (ch["cause"], ch["effect"], ch["lag"], ch["action"]) == (0, 1, 1, "added")
    assert (ch["cause_name"], ch["effect_name"]) == ("X", "Y")
    assert ch["p"] < 1e-4


def test_field_is_declared_for_results_without_the_step():
    from causalts.cdnots.result import CdnotsResult

    assert CdnotsResult.lag_refinements is None
    assert _run(_masked()).lag_refinements is None


def test_change_that_is_a_no_op_on_the_skeleton_is_dropped(monkeypatch):
    import warnings

    import causalts.cdnots.phase3_utils as p3

    # e.g. a link the ancestral constraints cut from the finished graph only:
    # re-adding it to the skeleton changes nothing and must not be reported
    from causalts.cdnots.lag_refine import _copies, _remove

    def fake(cg, *args, skip=(), tested=None, **kwargs):
        # refine_lagged edits the graph it gets, as the ancestral cut did here
        _remove(cg, _copies(0, 1, 1, 2, 1))
        return [dict(cause=0, effect=1, lag=1, action="added", p=1e-6, conds=[])]

    df = _masked(b=1.0)
    base = _run(df)
    assert np.asarray(base.cg_tig)[0, 1, 1] == 1
    monkeypatch.setattr(p3, "_lag_refine_step", fake)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        res = _run(df, addback_alpha=1e-4)
    assert res.lag_refinements == []
    np.testing.assert_array_equal(np.asarray(res.cg_tig), np.asarray(base.cg_tig))
    np.testing.assert_array_equal(res.to_marks(), base.to_marks())
    np.testing.assert_array_equal(res.pvalue_matrix, base.pvalue_matrix)


def test_added_links_refresh_the_ancestry_cache():
    df = _masked()
    res = run_cdnots_plus(
        df,
        ParCorrGPU(df.values, device="cpu"),
        num_lags=2,
        include_C=False,
        alpha=0.01,
        show_progress=False,
        addback_alpha=1e-4,
    )
    assert any(c["action"] == "added" for c in res.lag_refinements)
    g, nodes = res.graph.G, res.graph.G.nodes
    # X(t-2) reaches Y(t) only through the added copy X(t-2) -> Y(t-1) -> Y(t).
    assert g.is_ancestor_of(nodes[0 + 2 * 2], nodes[1])


def test_veto_sees_the_grown_graph():
    df = _masked()
    # The added link has p ~ 1e-9, so a veto at 1e-12 removes it again in the
    # same pass: the veto re-tests what add-back added, and the net change is
    # none (no oscillation between passes).
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        res = _run(df, addback_alpha=1e-4, veto_alpha=1e-12)
    G = np.asarray(res.cg_tig)
    assert res.lag_refinements == []
    assert G[0, 1, 1] == 0
    assert G[1, 1, 1] == 1  # self-links are not touched


def test_forbidden_link_is_not_added():
    df = _masked()
    bk = make_background_knowledge(
        ["X", "Y"], 1, include_C=False, forbidden=[("X", "Y", 1, 0)]
    )
    res = _run(df, addback_alpha=1e-4, background_knowledge=bk)
    assert np.asarray(res.cg_tig)[0, 1, 1] == 0
    assert res.lag_refinements == []


def test_ci_test_state_is_restored_and_prefix_is_reusable():
    df = _masked()
    ci = ParCorrGPU(df.values, device="cpu")
    state = _run(df, ci=ci, return_prefix=True, addback_alpha=1e-4, veto_alpha=1e-3)
    window = ci.data
    out = finish_cdnots_plus(state)
    assert ci.data is window
    again = finish_cdnots_plus(state)
    np.testing.assert_array_equal(np.asarray(out.cg_tig), np.asarray(again.cg_tig))
    assert out.lag_refinements == again.lag_refinements != []


def test_orientation_reruns_on_the_refined_skeleton(monkeypatch):
    import causalts.cdnots.phase3_utils as p3

    calls = []
    real = p3._majority_collider_step

    def spy(cg, *args, **kwargs):
        calls.append(cg.G.get_edge(cg.G.nodes[0 + 2], cg.G.nodes[1]) is not None)
        return real(cg, *args, **kwargs)

    monkeypatch.setattr(p3, "_majority_collider_step", spy)
    res = _run(_masked(), addback_alpha=1e-4)
    assert calls == [False, True]  # second collider pass sees X(t-1) -> Y(t)
    assert np.asarray(res.cg_tig)[0, 1, 1] == 1


def test_refinement_repeats_until_a_pass_changes_nothing(monkeypatch):
    import causalts.cdnots.phase3_utils as p3

    passes = []
    real = p3._lag_refine_step

    def spy(*args, **kwargs):
        out = real(*args, **kwargs)
        passes.append(len(out))
        return out

    monkeypatch.setattr(p3, "_lag_refine_step", spy)
    _run(_masked(), addback_alpha=1e-4, veto_alpha=1e-3)
    assert passes == [1, 0]  # pass 2 re-checks the re-oriented graph


def test_oscillating_link_is_frozen_at_its_original_state(monkeypatch):
    import causalts.cdnots.phase3_utils as p3

    flip = iter(["added", "removed"] * 10)

    def fake(*args, skip=(), tested=None, **kwargs):
        if (0, 1, 1) in skip:
            return []
        return [dict(cause=0, effect=1, lag=1, action=next(flip), p=0.5, conds=[])]

    monkeypatch.setattr(p3, "_lag_refine_step", fake)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        res = _run(_masked(), addback_alpha=1e-4)
    base = _run(_masked())
    assert res.lag_refinements == []
    np.testing.assert_array_equal(np.asarray(res.cg_tig), np.asarray(base.cg_tig))
    np.testing.assert_array_equal(res.pvalue_matrix, base.pvalue_matrix)


def test_frozen_link_gets_its_conditioning_parent_entry_back(monkeypatch):
    import causalts.cdnots.phase3_utils as p3

    flip = iter(["added", "removed"] * 10)

    def fake(*args, skip=(), tested=None, **kwargs):
        if (0, 1, 1) in skip:
            return []
        return [dict(cause=0, effect=1, lag=1, action=next(flip), p=0.5, conds=[])]

    seen = []
    collide = p3._majority_collider_step

    def spy(cg, indep_test, df, data, lags, num_lags, discrete_cols, parents, **kw):
        seen.append({j: list(v) for j, v in parents.items()})
        return collide(
            cg, indep_test, df, data, lags, num_lags, discrete_cols, parents, **kw
        )

    monkeypatch.setattr(p3, "_lag_refine_step", fake)
    monkeypatch.setattr(p3, "_majority_collider_step", spy)
    # strong X(t-1) -> Y: phase 1 lists it as a parent of Y
    _run(_masked(b=1.0), addback_alpha=1e-4)
    assert 0 + 1 * 2 in seen[0][1]
    assert seen[-1] == seen[0]


def test_frozen_link_gets_its_original_separating_sets_back(monkeypatch):
    import causalts.cdnots.phase3_utils as p3

    flip = iter(["added", "removed"])

    def fake(*args, skip=(), tested=None, **kwargs):
        if (0, 1, 1) in skip:
            return []
        return [dict(cause=0, effect=1, lag=1, action=next(flip), p=0.5, conds=[3])]

    df = _masked()
    before = _run(df, return_prefix=True).cg_sepset.sepset[2, 1]
    monkeypatch.setattr(p3, "_lag_refine_step", fake)
    state = _run(df, return_prefix=True, addback_alpha=1e-4)
    assert state.cg_sepset.sepset[2, 1] == before


def test_net_change_reports_its_final_pass_p_value(monkeypatch):
    import causalts.cdnots.phase3_utils as p3

    passes = iter(
        [
            ([dict(cause=0, effect=1, lag=1, action="added", p=0.01, conds=[])], 0.01),
            ([], 0.04),
        ]
    )

    def fake(*args, skip=(), tested=None, **kwargs):
        changes, p = next(passes)
        tested[(0, 1, 1)] = p
        return changes

    monkeypatch.setattr(p3, "_lag_refine_step", fake)
    res = _run(_masked(), addback_alpha=1e-4)
    (ch,) = res.lag_refinements
    assert ch["p"] == 0.04 and res.pvalue_matrix[0, 1, 1] == 0.04


def test_veto_of_a_partial_class_reports_the_removed_copy():
    # Two lags; only the shifted copy X(t-2) -> Y(t-1) (4 -> 3) exists.
    cg = _window(2, 2, directed=[(4, 3)])
    changes = refine_lagged(
        cg,
        lambda x, y, Z: (0.01, 1.0),
        2,
        c_start=2,
        alpha_add=0.05,
        alpha_veto=0.005,
    )
    assert [(c["cause"], c["effect"], c["lag"], c["action"]) for c in changes] == [
        (0, 1, 1, "removed")
    ]
    assert cg.G.get_edge(cg.G.nodes[4], cg.G.nodes[3]) is None


def test_refined_prefix_cannot_be_finished_at_a_mutated_margin():
    state = _run(_masked(), return_prefix=True, addback_alpha=1e-4)
    state.orient_margin = state.orient_margin + 1
    with pytest.raises(ValueError, match="orient_margin"):
        finish_cdnots_plus(state)


def test_refined_prefix_rejects_another_margin():
    state = _run(_masked(), return_prefix=True, addback_alpha=1e-4)
    finish_cdnots_plus(state, orient_margin=state.orient_margin)  # same: fine
    with pytest.raises(ValueError, match="orient_margin"):
        finish_cdnots_plus(state, orient_margin=state.orient_margin + 1)


def test_refined_prefix_rejects_another_lag0_orient():
    state = _run(_masked(), return_prefix=True, addback_alpha=1e-4)
    finish_cdnots_plus(state, lag0_orient=None)  # the stored value: fine
    with pytest.raises(ValueError, match="lag0_orient"):
        finish_cdnots_plus(state, lag0_orient="lingam")


def test_refined_prefix_cannot_be_finished_at_a_mutated_lag0_orient():
    state = _run(_masked(), return_prefix=True, addback_alpha=1e-4)
    state.lag0_orient = "lingam"
    with pytest.raises(ValueError, match="lag0_orient=None"):
        finish_cdnots_plus(state)


@pytest.mark.parametrize("lag0_orient", [None, "lingam"])
def test_direct_result_matches_the_finished_prefix(lag0_orient):
    df = _masked()
    kw = dict(addback_alpha=1e-4, veto_alpha=1e-3, lag0_orient=lag0_orient)
    direct = _run(df, **kw)
    via = finish_cdnots_plus(_run(df, return_prefix=True, **kw))
    np.testing.assert_array_equal(np.asarray(direct.cg_tig), np.asarray(via.cg_tig))
    np.testing.assert_array_equal(direct.to_marks(), via.to_marks())
    np.testing.assert_array_equal(direct.pvalue_matrix, via.pvalue_matrix)
    assert direct.lag_refinements == via.lag_refinements != []
    assert direct.lag0_orientations == via.lag0_orientations


def test_added_pair_has_no_stale_separating_set():
    state = _run(_masked(), return_prefix=True, addback_alpha=1e-4)
    assert state.cg_sepset.sepset[2, 1] is None and state.cg_sepset.sepset[1, 2] is None


def test_changed_links_carry_their_refinement_p_value():
    res = _run(_masked(), addback_alpha=1e-4)
    (ch,) = res.lag_refinements
    assert res.pvalue_matrix[0, 1, 1] == ch["p"]


def test_required_lag_copy_protects_the_link_from_the_veto():
    df = _masked()
    bk = make_background_knowledge(
        ["X", "Y"], 2, include_C=False, required=[("X", "Y", 2, 1)]
    )
    res = run_cdnots_plus(
        df,
        ParCorrGPU(df.values, device="cpu"),
        num_lags=2,
        include_C=False,
        alpha=0.01,
        show_progress=False,
        background_knowledge=bk,
        veto_alpha=1e-300,  # would remove any link not protected
    )
    assert np.asarray(res.cg_tig)[0, 1, 1] == 1
    assert not any(
        (c["cause"], c["effect"], c["lag"]) == (0, 1, 1) for c in res.lag_refinements
    )


def _window(n_vars, num_lags, directed=()):
    cg = initialize_graph(np.zeros((20, n_vars * (num_lags + 1))), None)
    for e in list(cg.G.get_graph_edges()):
        cg.G.remove_edge(e)
    for a, b in directed:
        cg.G.add_directed_edge(cg.G.nodes[a], cg.G.nodes[b])
    return cg


def test_forbidden_ancestor_is_checked_with_all_copies_in_place():
    # X=0, Y=1, two lags: X(t-1)=2, Y(t-1)=3, X(t-2)=4, Y(t-2)=5. Y(t-1) -> Y(t).
    # Only the copy X(t-2) -> Y(t-1) gives X(t-2) a path to Y(t).
    cg = _window(2, 2, directed=[(3, 1), (5, 3)])
    changes = refine_lagged(
        cg,
        lambda x, y, Z: (0.0, 1.0),
        2,
        c_start=2,
        alpha_add=0.5,
        forbidden_ancestor_pairs=[(4, 1)],
    )
    assert not any((c["cause"], c["effect"]) == (0, 1) for c in changes)
    assert cg.G.get_edge(cg.G.nodes[2], cg.G.nodes[1]) is None


def test_veto_keeps_the_last_path_of_a_required_ancestor():
    cg = _window(2, 1, directed=[(2, 1)])  # X(t-1) -> Y(t) only
    changes = refine_lagged(
        cg,
        lambda x, y, Z: (1.0, 0.0),
        1,
        c_start=2,
        alpha_veto=0.5,
        required_ancestor_groups=[[(0, 1), (2, 1)]],
    )
    assert changes == []
    assert cg.G.get_edge(cg.G.nodes[2], cg.G.nodes[1]) is not None


def test_too_few_complete_rows_changes_nothing():
    cg = _window(2, 1, directed=[(2, 1)])
    changes = refine_lagged(
        cg,
        lambda x, y, Z: (1.0, 0.0),  # an insufficient-data answer
        1,
        c_start=2,
        alpha_add=0.5,
        alpha_veto=0.5,
        enough_rows=lambda cols: False,
    )
    assert changes == []
    assert cg.G.get_edge(cg.G.nodes[2], cg.G.nodes[1]) is not None


def test_contiguous_lag_list_is_accepted():
    df = _masked(T=200)
    res = run_cdnots_plus(
        df,
        ParCorrGPU(df.values, device="cpu"),
        lag_list=[1],
        include_C=False,
        alpha=0.01,
        show_progress=False,
        addback_alpha=1e-4,
    )
    assert res.lag_refinements is not None


@pytest.mark.parametrize(
    "kw, match",
    [
        (dict(addback_alpha=0.0), "addback_alpha"),
        (dict(veto_alpha=1.5), "veto_alpha"),
        (dict(alpha_mci=-1), "alpha_mci"),
        (dict(addback_alpha=1e-4, lag_list=[1, 3]), "contiguous"),
        (dict(addback_alpha=1e-4, lag_list=[]), "contiguous"),
        (dict(addback_alpha=1e-4, max_conds_py=1), "max_conds"),
        (dict(addback_alpha=1e-4, orient_margin=float("nan")), "orient_margin"),
        (dict(addback_alpha=1e-4, num_lags=0), "num_lags"),
        (dict(addback_alpha=1e-4, num_lags=-1), "num_lags"),
        (dict(addback_alpha=1e-4, num_lags=0.5), "num_lags"),
    ],
)
def test_rejects_bad_settings_before_discovery(kw, match):
    df = _masked(T=100)
    base = dict(num_lags=1, include_C=False, alpha=0.01, show_progress=False)
    base.update(kw)
    with pytest.raises(ValueError, match=match):
        run_cdnots_plus(df, ParCorrGPU(df.values, device="cpu"), **base)
