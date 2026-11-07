# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Reporting of ancestral (path) knowledge in CDNOTS / CDNOTS+."""

import warnings

import numpy as np
import pandas as pd
import pytest

import causalts.cdnots.phase3_utils as p3
from causalts import run_cdnots
from causalts.cdnots.ancestral import AncestralKnowledge, apply_ancestral_constraints
from causalts.cdnots.phase3_utils import run_cdnots_plus
from causalts.ci_tests import ParCorrGPU

RUNNERS = {"cdnots": run_cdnots, "cdnots+": run_cdnots_plus}


@pytest.fixture(scope="module")
def chain():
    """A(t-1) -> X(t), X(t-1) -> B(t)."""
    rng = np.random.default_rng(0)
    T = 800
    x = np.zeros((T, 3))
    e = rng.standard_normal((T, 3))
    for t in range(1, T):
        x[t, 0] = 0.4 * x[t - 1, 0] + e[t, 0]
        x[t, 1] = 0.6 * x[t - 1, 0] + e[t, 1]
        x[t, 2] = 0.6 * x[t - 1, 1] + e[t, 2]
    return pd.DataFrame(x, columns=["A", "X", "B"])


def _ci():
    return ParCorrGPU(np.zeros((2, 2)), device="cpu")


def test_refinement_change_undone_by_ancestral_cleanup_is_not_reported(
    chain, monkeypatch
):
    """An add-back the forbidden-ancestor cleanup removes stayed listed."""
    calls = []

    def fake_step(cg, *args, **kwargs):
        calls.append(1)
        if len(calls) > 1:
            return []
        # Add A(t-1) -> B(t): absent from the skeleton, and a forbidden path.
        return [dict(cause=0, effect=2, lag=1, action="added", p=1e-12, conds=[])]

    monkeypatch.setattr(p3, "_lag_refine_step", fake_step)
    k = AncestralKnowledge().add_forbidden_ancestor("A", "B")
    plain = run_cdnots_plus(chain, _ci(), num_lags=2, include_C=False, knowledge=k)
    res = run_cdnots_plus(
        chain,
        _ci(),
        num_lags=2,
        include_C=False,
        knowledge=k,
        addback_alpha=1e-4,
        veto_alpha=None,
    )
    assert len(calls) >= 1
    assert not res.cg_tig[0, 2, 1]  # cleanup wins
    assert res.lag_refinements == []
    np.testing.assert_array_equal(res.pvalue_matrix, plain.pvalue_matrix)


def test_old_prefix_without_discovery_pvalues_drops_the_stale_refinement_pvalue(
    chain, monkeypatch
):
    """A prefix pickled before discovery p-values were kept cannot restore them."""
    calls = []

    def fake_step(cg, *args, **kwargs):
        calls.append(1)
        if len(calls) > 1:
            return []
        return [dict(cause=0, effect=2, lag=1, action="added", p=1e-12, conds=[])]

    monkeypatch.setattr(p3, "_lag_refine_step", fake_step)
    k = AncestralKnowledge().add_forbidden_ancestor("A", "B")
    state = run_cdnots_plus(
        chain,
        _ci(),
        num_lags=2,
        include_C=False,
        knowledge=k,
        addback_alpha=1e-4,
        veto_alpha=None,
        return_prefix=True,
    )
    del state.discovery_pvalue_matrix  # as unpickled from an older build
    with pytest.warns(UserWarning, match="recompute"):
        res = p3.finish_cdnots_plus(state)
    assert res.lag_refinements == []
    assert np.isnan(res.pvalue_matrix[0, 2, 1])


@pytest.mark.parametrize("algo", RUNNERS)
def test_unresolvable_forbidden_ancestor_is_reported(chain, algo):
    """A violation the cleanup cannot cut was only printed in verbose mode."""
    k = AncestralKnowledge().add_required_edge("A", "X", 1, 0)
    k.add_forbidden_ancestor("A", "X")
    with pytest.warns(UserWarning, match="not satisfied"):
        res = RUNNERS[algo](chain, _ci(), num_lags=1, include_C=False, knowledge=k)
    assert ("A(t-1)", "X") in res.ancestral_violations["forbidden"]
    assert res.cg_tig[0, 1, 1]  # the required edge stays


def test_required_ancestor_undone_later_is_reported():
    """The required list was built per constraint, before later steps ran."""
    from causalts.cdnots.meek import reaches
    from causalts.cdnots.skeleton_discovery import initialize_graph

    # Skeleton A - C - B, no lags. A => B orients A -> C -> B; C => A then
    # flips A - C, so A no longer reaches B.
    cg = initialize_graph(np.zeros((20, 3)), None)
    cg.G.remove_edge(cg.G.get_edge(cg.G.nodes[0], cg.G.nodes[1]))
    k = AncestralKnowledge().add_required_ancestor("A", "B")
    k.add_required_ancestor("C", "A")
    out, violations = apply_ancestral_constraints(cg, k, ["A", "B", "C"], 0)
    assert not reaches(out.G.graph, 0, 1)
    assert violations["required"] == [("A", "B")]


@pytest.mark.parametrize("algo", RUNNERS)
def test_satisfied_constraints_report_nothing(chain, algo):
    k = AncestralKnowledge().add_required_ancestor("A", "B")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        res = RUNNERS[algo](chain, _ci(), num_lags=2, include_C=False, knowledge=k)
    assert res.ancestral_violations == {"required": [], "forbidden": [], "ignored": []}
    assert run_cdnots(chain, _ci(), num_lags=1).ancestral_violations is None


@pytest.mark.parametrize(
    "call",
    [
        lambda df, k: run_cdnots(df, _ci(), num_lags=1, include_C=False, knowledge=k),
        lambda df, k: run_cdnots_plus(
            df, _ci(), num_lags=1, include_C=False, knowledge=k
        ),
        lambda df, k: p3.cdnots_discovery(
            df, _ci(), num_lags=1, include_C=False, knowledge=k
        ),
    ],
    ids=["run_cdnots", "run_cdnots_plus", "cdnots_discovery"],
)
def test_violation_warnings_point_at_the_caller(chain, call):
    """Each entry point reaches the warning at a different depth; all must point here."""
    k = AncestralKnowledge().add_required_edge("A", "X", 1, 0)
    k.add_forbidden_ancestor("A", "X")
    k.add_forbidden_ancestor("A", "Q")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        call(chain, k)
    ours = [
        w
        for w in caught
        if "unknown variables" in str(w.message) or "not satisfied" in str(w.message)
    ]
    assert len(ours) == 2
    assert all(w.filename == __file__ for w in ours), [w.filename for w in ours]


def test_constraint_on_unknown_variable_is_reported(chain):
    """A constraint naming a missing variable was skipped silently."""
    k = AncestralKnowledge().add_forbidden_ancestor("A", "Q")
    with pytest.warns(UserWarning, match="unknown variables"):
        res = run_cdnots(chain, _ci(), num_lags=1, include_C=False, knowledge=k)
    assert res.ancestral_violations["ignored"] == [("forbidden_ancestor", "A", "Q")]


def test_unknown_variable_warns_once_in_a_refined_cdnots_plus_run(chain, monkeypatch):
    """Intermediate refinement passes must not repeat the warning."""
    calls = []

    def fake_step(cg, *args, **kwargs):
        calls.append(1)  # one accepted change, so a second pass runs
        if len(calls) > 1:
            return []
        return [dict(cause=0, effect=2, lag=1, action="added", p=1e-12, conds=[])]

    monkeypatch.setattr(p3, "_lag_refine_step", fake_step)
    k = AncestralKnowledge().add_forbidden_ancestor("A", "Q")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = run_cdnots_plus(
            chain,
            _ci(),
            num_lags=1,
            include_C=False,
            knowledge=k,
            addback_alpha=1e-4,
            veto_alpha=1e-3,
        )
    assert len(calls) == 2
    msgs = [w for w in caught if "unknown variables" in str(w.message)]
    assert len(msgs) == 1
    assert res.ancestral_violations["ignored"] == [("forbidden_ancestor", "A", "Q")]
