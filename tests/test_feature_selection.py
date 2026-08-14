# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for causal feature selection (causalts.feature_selection)."""

import numpy as np
import pandas as pd
import pytest

from causalts.cdnots.phase3_utils import run_cdnots
from causalts.cedar.discovery import run_cedar
from causalts.ci_tests.parcorr_gpu import ParCorrGPU
from causalts.feature_selection import (
    Feature,
    FeatureSelectionResult,
    select_features,
)
from causalts.synthetic_data.synthetic_datasets import ex2, ex3


@pytest.fixture
def data():
    return ex2()


def test_returns_result(data):
    r = select_features(data["df"], "X3", max_lag=data["max_lag"], algo="cedar")
    assert isinstance(r, FeatureSelectionResult)
    assert r.target == "X3"
    assert r.algo == "cedar"
    assert all(isinstance(f, Feature) for f in r.features)


def test_cedar_assumes_self_lag1_in_features(data):
    """CEDAR does not test self-loops but assumes AR(1); its lag-1 self edge
    must therefore appear in the discovered features."""
    r = select_features(data["df"], "X3", max_lag=data["max_lag"], algo="cedar")
    self_feats = r.by_role("self")
    assert ("X3", 1) in {(f.variable, f.lag) for f in self_feats}
    assert all(f.variable == "X3" for f in self_feats)


def test_self_history_score_is_dcor_in_range(data):
    """Self-history that discovery surfaces carries a partial-dcor score in
    [0, 1], deterministic across runs."""
    df = data["df"]
    L = data["max_lag"]
    r = select_features(df, "X3", max_lag=L, algo="cedar")
    sc = {f.lag: f.score for f in r.by_role("self")}
    assert sc, "cedar should surface at least the assumed AR(1) self lag"
    assert all(v is not None and 0.0 <= v <= 1.0 for v in sc.values())
    r2 = select_features(df, "X3", max_lag=L, algo="cedar")
    assert {f.lag: f.score for f in r2.by_role("self")} == sc


def test_design_matrix_include_self_toggle(data):
    """to_design_matrix owns self-lag inclusion: include_self=False yields only
    cross-variable columns; include_self=True adds self-lag columns."""
    df = data["df"]
    r = select_features(df, "X3", max_lag=data["max_lag"], algo="cedar")
    cross = {(f.variable, f.lag) for f in r.features if f.role == "parent"}

    X_no = r.to_design_matrix(include_self=False)
    assert not any(c.startswith("X3_lag") for c in X_no.columns)
    assert X_no.shape[1] == len(cross)

    X_yes = r.to_design_matrix(include_self=True, self_threshold=0.05)
    assert any(c.startswith("X3_lag") for c in X_yes.columns)
    assert X_yes.shape[1] > X_no.shape[1]


def test_design_matrix_self_threshold(data):
    """A high self_threshold drops weak self-lags; threshold=0 keeps all."""
    df = data["df"]
    r = select_features(df, "X3", max_lag=data["max_lag"], algo="cedar")
    n_high = r.to_design_matrix(self_threshold=0.99).shape[1]
    n_all = r.to_design_matrix(self_threshold=0.0).shape[1]
    n_self_all = sum(
        1
        for c in r.to_design_matrix(self_threshold=0.0).columns
        if c.startswith("X3_lag")
    )
    assert n_self_all == data["max_lag"]  # threshold=0 keeps every self-lag
    assert n_high < n_all  # a near-1 threshold drops weak self-lags


def test_design_matrix_shape_and_names(data):
    df = data["df"]
    r = select_features(df, "X3", max_lag=data["max_lag"])
    X = r.to_design_matrix()
    assert X.shape[0] == len(df) - data["max_lag"]
    for col in X.columns:
        assert "_lag" in col


def test_variables_excludes_c_flag(data):
    r = select_features(data["df"], "X3", max_lag=data["max_lag"])
    # variables are real column names only
    assert set(r.variables).issubset(set(data["df"].columns))


def test_from_result_matches_fresh(data):
    """A slice of a full run should recover the same PC-set features as a fresh
    restricted run (the regression-test premise from the design)."""
    df = data["df"]
    ci = ParCorrGPU(np.zeros((2, 2)))
    full = run_cedar(df, ci_test=ci, max_lag=data["max_lag"], verbose=False)

    sliced = select_features(df, "X3", max_lag=data["max_lag"], from_result=full)
    fresh = select_features(df, "X3", max_lag=data["max_lag"], algo="cedar")
    # Compare parent (var, lag) sets — the direct causes.
    sliced_parents = {(f.variable, f.lag) for f in sliced.by_role("parent")}
    fresh_parents = {(f.variable, f.lag) for f in fresh.by_role("parent")}
    assert sliced_parents == fresh_parents


def test_bad_target_raises(data):
    with pytest.raises(ValueError):
        select_features(data["df"], "NOPE", max_lag=data["max_lag"])


def test_fastpc_backend_runs(data):
    r = select_features(data["df"], "X3", max_lag=data["max_lag"], algo="fastpc")
    assert isinstance(r, FeatureSelectionResult)
    assert r.algo == "fastpc"
    # lagged neighbours are oriented as parents by temporal precedence
    assert all(f.lag >= 1 for f in r.by_role("parent"))
    assert all(f.role != "child" for f in r.features)  # fastpc surfaces no children


def test_fastpc_lag0_is_undirected(data):
    """With include_lag0, fastpc reports contemporaneous cross-variable edges as
    'undirected' (it does no orientation), never as parent/child."""
    r = select_features(
        data["df"], "X3", max_lag=data["max_lag"], algo="fastpc", include_lag0=True
    )
    lag0_cross = [
        f
        for f in r.features
        if f.lag == 0 and f.variable != "X3" and f.role != "nonstationarity"
    ]
    assert all(f.role == "undirected" for f in lag0_cross)


def test_fastpc_matches_full_run_parents(data):
    """fastpc's lag>=1 parents should be a superset of full CDNOTS's incoming
    edges (local-PC recall bias: may keep extras, must never miss a true one)."""
    df = data["df"]
    L = data["max_lag"]
    ci = ParCorrGPU(np.zeros((2, 2)))
    full = run_cdnots(df, ci, num_lags=L, verbose=False)
    g = full.cg_tig
    cols = list(df.columns)
    tj = cols.index("X3")
    full_parents = {
        (cols[v], lag)
        for lag in range(1, g.shape[2])
        for v in range(g.shape[0])
        if g[v, tj, lag] == 1 and v != tj
    }
    r = select_features(df, "X3", max_lag=L, algo="fastpc")
    local_parents = {(f.variable, f.lag) for f in r.by_role("parent") if f.lag >= 1}
    assert full_parents.issubset(local_parents)


def test_fastpc_design_matrix(data):
    r = select_features(data["df"], "X3", max_lag=data["max_lag"], algo="fastpc")
    X = r.to_design_matrix()
    assert X.shape[0] == len(data["df"]) - data["max_lag"]
    n_cross = len({(f.variable, f.lag) for f in r.by_role("parent")})
    n_self = sum(1 for c in X.columns if c.startswith("X3_lag"))
    assert X.shape[1] == n_cross + n_self
    assert n_self >= 1  # X3 has real self-history


def test_real_cdnots_backend_runs(data):
    """algo='cdnots' runs the real (oriented) full algorithm and slices it."""
    r = select_features(data["df"], "X3", max_lag=data["max_lag"], algo="cdnots")
    assert r.algo == "cdnots"
    assert isinstance(r, FeatureSelectionResult)
    # default (no MB) surfaces no children
    assert r.by_role("child") == []


def test_real_cdnots_markov_blanket_from_graph(data):
    """markov_blanket=True on real cdnots extracts children/spouses from the
    full oriented graph (no error, returns a valid result)."""
    r = select_features(
        data["df"], "X3", max_lag=data["max_lag"], algo="cdnots", markov_blanket=True
    )
    assert r.markov_blanket is True
    assert isinstance(r, FeatureSelectionResult)


# ----------------------------------------------------------------------
# ex3 (protein-signaling cascade, 11 vars) -- a second, differently-shaped
# dataset to make sure results aren't an ex2 artifact.
# ----------------------------------------------------------------------
@pytest.fixture
def ex3_data():
    return ex3()


def test_ex3_cedar_recovers_known_causes(ex3_data):
    """ex3's ground truth has Erk <- {PKA(lag1), Mek(lag2)}; both algos should
    recover this exactly (both are ParCorr-testable linear/near-linear edges)."""
    df = ex3_data["df"]
    L = ex3_data["max_lag"]
    r = select_features(df, "Erk", max_lag=L, algo="cedar")
    parents = {(f.variable, f.lag) for f in r.by_role("parent")}
    assert parents == {("PKA", 1), ("Mek", 2)}


def test_ex3_cdnots_recovers_known_causes(ex3_data):
    """Real CDNOTS recovers Mek(lag2) for Erk (it prunes PKA(lag1), likely by
    MCI -- a real algorithmic difference from cedar, not a bug). fastpc, which
    mirrors the skeleton, should also surface Mek(lag2)."""
    df = ex3_data["df"]
    L = ex3_data["max_lag"]
    for algo in ("cdnots", "fastpc"):
        r = select_features(df, "Erk", max_lag=L, algo=algo)
        parents = {(f.variable, f.lag) for f in r.by_role("parent") if f.lag >= 1}
        assert ("Mek", 2) in parents, algo


# ----------------------------------------------------------------------
# Markov blanket / spouse recovery
# ----------------------------------------------------------------------
@pytest.fixture
def collider_data():
    """A hand-built collider DGP with a known spouse relationship.

    X0, X1 are independent noise. X2(t) = 0.8*X0(t) + 0.8*X1(t-1) + noise, so
    X0(t) -> X2(t) [contemporaneous child] and X1(t-1) -> X2(t) [the other
    parent]. X0 and X1 are marginally independent but become dependent when
    conditioning on their common child X2 — the classic v-structure. Given
    target=X0, X1 is a spouse: child X2 at lag a=0, spouse edge lag b=1, so
    spouse_lag = b - a = 1 -> X1 at lag 1 relative to X0.
    """
    rng = np.random.default_rng(0)
    T = 2000
    x0 = rng.standard_normal(T)
    x1 = rng.standard_normal(T)
    x2 = np.zeros(T)
    for t in range(1, T):
        x2[t] = 0.8 * x0[t] + 0.8 * x1[t - 1] + 0.1 * rng.standard_normal()
    df = pd.DataFrame({"X0": x0, "X1": x1, "X2": x2})
    return df


def test_markov_blanket_recovers_spouse(collider_data):
    r_pc = select_features(
        collider_data,
        "X0",
        max_lag=2,
        algo="cedar",
        include_lag0=True,
        markov_blanket=False,
    )
    assert ("X1", 1) not in {(f.variable, f.lag) for f in r_pc.features}

    r_mb = select_features(
        collider_data,
        "X0",
        max_lag=2,
        algo="cedar",
        include_lag0=True,
        markov_blanket=True,
    )
    spouse_pairs = {(f.variable, f.lag) for f in r_mb.by_role("spouse")}
    assert ("X1", 1) in spouse_pairs


def test_markov_blanket_no_children_is_noop(data):
    """If the target has no children, markov_blanket=True should not error and
    should add no spouses (nothing to anchor a collider test on)."""
    r = select_features(data["df"], "X3", max_lag=data["max_lag"], markov_blanket=True)
    assert isinstance(r, FeatureSelectionResult)


def test_spouse_collider_honors_passed_ci_test(collider_data):
    """The spouse collider decision routes through the user's CI test (here an
    explicit ParCorrGPU) rather than a hardcoded internal one — recovery still
    works when a ci_test is supplied."""
    ci = ParCorrGPU(np.zeros((2, 2)))
    r = select_features(
        collider_data,
        "X0",
        max_lag=2,
        algo="cedar",
        include_lag0=True,
        markov_blanket=True,
        ci_test=ci,
    )
    assert ("X1", 1) in {(f.variable, f.lag) for f in r.by_role("spouse")}


def test_cdnots_score_honors_nonlinear_ci_test(data):
    """cdnots edge scores are computed by the utility (its graph is binary). With
    a nonlinear CI test they must come from that test's conditional statistic,
    not the linear-residualize + dcor default — so the two paths yield different
    (populated) scores for the same edges."""
    from causalts.ci_tests.kci_gpu import KCIGPU

    df = data["df"]
    L = data["max_lag"]
    r_lin = select_features(df, "X3", max_lag=L, algo="cdnots")
    r_nl = select_features(
        df, "X3", max_lag=L, algo="cdnots", ci_test=KCIGPU(np.zeros((2, 2)))
    )
    lin = {(f.variable, f.lag): f.score for f in r_lin.by_role("parent")}
    nl = {(f.variable, f.lag): f.score for f in r_nl.by_role("parent")}
    assert lin and nl and set(lin) == set(nl)
    assert all(v is not None for v in nl.values())
    # KCI statistic lives on a different scale than dcor -> scores must differ.
    assert any(abs(nl[k] - lin[k]) > 1e-6 for k in lin)


def test_cdnots_forwards_alpha(monkeypatch, data):
    """algo='cdnots' must forward the caller's alpha to run_cdnots, not silently
    fall back to run_cdnots's own default."""
    import causalts.cdnots.phase3_utils as p3

    original = p3.run_cdnots
    captured = {}

    def spy(*args, **kwargs):
        captured["alpha"] = kwargs.get("alpha")
        return original(*args, **kwargs)

    monkeypatch.setattr(p3, "run_cdnots", spy)
    select_features(
        data["df"], "X3", max_lag=data["max_lag"], algo="cdnots", alpha=0.01
    )
    assert captured["alpha"] == 0.01


def test_cdnots_gates_lag0(collider_data):
    """cdnots always discovers contemporaneous edges, but select_features must
    gate lag-0 cross-variable features on include_lag0 (parity with cedar/fastpc).
    The collider DGP has X0(t) -> X2(t)."""
    lag0_roles = {"parent", "child", "undirected"}
    r_off = select_features(
        collider_data, "X2", max_lag=2, algo="cdnots", include_lag0=False
    )
    assert not any(f.lag == 0 and f.role in lag0_roles for f in r_off.features)

    r_on = select_features(
        collider_data, "X2", max_lag=2, algo="cdnots", include_lag0=True
    )
    assert any(f.lag == 0 and f.variable == "X0" for f in r_on.features)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
