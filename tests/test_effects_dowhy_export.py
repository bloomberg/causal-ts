# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exporting artifacts for users who want to drive DoWhy themselves.

The graph flavour is not a matter of taste: identification needs the unrolled
graph, and handing a user the transition graph instead silently empties the
adjustment set and turns an unadjusted regression into a reported causal
effect. These tests pin that the two exports differ in exactly that way, and
that graph and frame can never be paired inconsistently.
"""

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from causalts.effects.export import (
    DoWhyArtifacts,
    build_identification_artifacts,
    build_transition_artifacts,
)

T = 3000


def _confounded():
    """X0 confounds X1 and X2; the effect of X1(t-1) on X2(t) is 0.4."""
    rng = np.random.default_rng(0)
    x = np.zeros((T, 3))
    for t in range(1, T):
        x[t, 0] = 0.5 * x[t - 1, 0] + rng.normal()
        x[t, 1] = 0.6 * x[t - 1, 0] + rng.normal()
        x[t, 2] = 0.4 * x[t - 1, 1] + 0.7 * x[t - 1, 0] + rng.normal()
    g = np.zeros((3, 3, 2), dtype=np.int8)
    g[0, 0, 1] = g[0, 1, 1] = g[1, 2, 1] = g[0, 2, 1] = 1
    return g, pd.DataFrame(x[10:], columns=["V0", "V1", "V2"])


# --------------------------------------------------------------- invariants


def test_nodes_and_columns_must_match():
    graph = nx.DiGraph([("A", "B")])
    data = pd.DataFrame({"A": [1.0], "B": [2.0]})
    DoWhyArtifacts(graph=graph, data=data, semantics="transition", horizon=1)

    with pytest.raises(ValueError, match="only in data"):
        DoWhyArtifacts(
            graph=graph,
            data=data.assign(C=[3.0]),
            semantics="transition",
            horizon=1,
        )
    with pytest.raises(ValueError, match="Only in graph"):
        DoWhyArtifacts(
            graph=nx.DiGraph([("A", "B"), ("B", "C")]),
            data=data,
            semantics="transition",
            horizon=1,
        )


@pytest.mark.parametrize("build", ["transition", "identification"])
def test_both_exports_satisfy_the_invariant(build):
    g, df = _confounded()
    art = (
        build_transition_artifacts(g, df)
        if build == "transition"
        else build_identification_artifacts(g, df, "V1", "V2", treatment_lag=1)
    )
    assert set(art.graph.nodes) == set(art.data.columns)
    assert not art.data.isna().any().any()
    assert art.semantics == build
    assert nx.is_directed_acyclic_graph(art.graph)


# ------------------------------------------------------- the two flavours


def test_transition_export_leaves_history_as_roots():
    g, df = _confounded()
    art = build_transition_artifacts(g, df)
    assert art.horizon == 1
    assert art.query is None
    lagged = [n for n in art.graph.nodes if "_lag" in n]
    assert lagged and all(art.graph.in_degree(n) == 0 for n in lagged)


def test_identification_export_gives_the_treatment_parents():
    """The difference that matters: on the transition graph V1_lag1 is a root."""
    g, df = _confounded()
    transition = build_transition_artifacts(g, df)
    assert list(transition.graph.predecessors("V1_lag1")) == []

    ident = build_identification_artifacts(g, df, "V1", "V2", treatment_lag=1)
    assert sorted(ident.graph.predecessors("V1_lag1")) == ["V0_lag2"]
    assert ident.query == {
        "treatment": "V1_lag1",
        "outcome": "V2",
        "treatment_lag": 1,
    }


def test_identification_horizon_follows_the_query():
    g, df = _confounded()
    assert build_identification_artifacts(g, df, "V1", "V2", 1).horizon == 2
    assert build_identification_artifacts(g, df, "V1", "V2", 2).horizon == 3
    # Deeper embeddings cost rows, which is why the depth is not just max_lag.
    shallow = build_identification_artifacts(g, df, "V1", "V2", 1)
    deep = build_identification_artifacts(g, df, "V1", "V2", 2)
    assert len(deep.data) < len(shallow.data)


def test_identification_export_keeps_the_backdoor_interior():
    """Pruning to pa(T) + the directed corridor would hide the backdoor.

    V0_lag2 is a common cause of V1_lag1 and (via V0_lag1) of V2. It is an
    ancestor of both, not a parent of the treatment's corridor, so a narrower
    prune drops it and identification reports nothing to adjust for.
    """
    g, df = _confounded()
    art = build_identification_artifacts(g, df, "V1", "V2", treatment_lag=1)
    assert "V0_lag2" in art.graph
    assert "V0_lag1" in art.graph


def test_identification_export_refuses_an_unknown_node():
    g, df = _confounded()
    with pytest.raises(ValueError, match="not found in the unrolled graph"):
        build_identification_artifacts(g, df, "V1", "nope")


# ------------------------------------------------- they actually drive DoWhy


def test_exported_identification_artifacts_recover_the_coefficient():
    """End to end through plain DoWhy, with nothing of ours in the loop."""
    dowhy = pytest.importorskip("dowhy")
    g, df = _confounded()
    art = build_identification_artifacts(g, df, "V1", "V2", treatment_lag=1)

    model = dowhy.CausalModel(
        data=art.data,
        treatment=art.query["treatment"],
        outcome=art.query["outcome"],
        graph=art.graph,
    )
    identified = model.identify_effect(proceed_when_unidentifiable=True)
    assert identified.get_backdoor_variables(), (
        "no adjustment set: the exported graph did not give the lagged "
        "treatment its parents"
    )
    est = model.estimate_effect(identified, method_name="backdoor.linear_regression")
    assert abs(float(est.value) - 0.4) < 0.03


def test_exported_transition_artifacts_fit_a_gcm():
    gcm = pytest.importorskip("dowhy.gcm")
    g, df = _confounded()
    art = build_transition_artifacts(g, df)
    scm = gcm.StructuralCausalModel(art.graph)
    gcm.auto.assign_causal_mechanisms(scm, art.data)
    gcm.fit(scm, art.data)
    assert set(scm.graph.nodes) == set(art.data.columns)


# -------------------------------------------------------- cycle resolution


def _bidirected():
    g = np.zeros((2, 2, 2), dtype=np.int8)
    g[0, 0, 1] = g[1, 1, 1] = 1
    g[0, 1, 0] = g[1, 0, 0] = 1  # discovery could not orient this
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(500, 2)), columns=["V0", "V1"])
    return g, df


def test_unoriented_contemporaneous_edge_is_refused_by_default():
    """Dropping it changes which backdoor paths exist, so it must be opt-in."""
    g, df = _bidirected()
    for build in (
        lambda **kw: build_transition_artifacts(g, df, **kw),
        lambda **kw: build_identification_artifacts(g, df, "V0", "V1", 1, **kw),
    ):
        with pytest.raises(ValueError, match="Cycle detected"):
            build()
        assert build(cycle_resolution="drop") is not None


def test_estimate_effect_refuses_it_too():
    pytest.importorskip("dowhy")
    from causalts.effects.effect import estimate_effect

    g, df = _bidirected()
    with pytest.raises(ValueError, match="Cycle detected"):
        estimate_effect(g, df, treatment="V0", outcome="V1", treatment_lag=1)


def test_the_cycle_resolution_escape_hatch_is_reachable():
    """It was documented but dead: the graph built and the adjustment set refused.

    `cycle_resolution` and `strict_contemporaneous` govern the same ambiguity, so
    opting into an arbitrary resolution of one has to opt into the other.
    """
    pytest.importorskip("dowhy")
    from causalts.effects.effect import estimate_effect

    g, df = _bidirected()
    est = estimate_effect(
        g,
        df,
        treatment="V0",
        outcome="V1",
        treatment_lag=1,
        cycle_resolution="drop",
        confidence_intervals=False,
    )
    assert est.value is not None


def test_artifacts_are_hashable_and_do_not_raise_on_equality():
    """frozen=True auto-generates __eq__/__hash__ over the DataFrame field."""
    g, df = _confounded()
    a, b = build_transition_artifacts(g, df), build_transition_artifacts(g, df)
    assert hash(a) != hash(b) and a != b


# --------------------------------------------------------- the result methods


def test_result_objects_expose_both_exports():
    from causalts.effects.wrap import wrap_graph

    g, df = _confounded()
    wrapped = wrap_graph(g, df)
    assert wrapped.to_dowhy_transition().semantics == "transition"
    ident = wrapped.to_dowhy_identification("V1", "V2", treatment_lag=1)
    assert ident.semantics == "identification"
    assert "identification" in repr(ident) and "V1_lag1 -> V2" in repr(ident)


def test_there_is_no_undifferentiated_to_dowhy():
    """A default flavour would silently export the bug this API exists to avoid."""
    from causalts.effects.wrap import WrappedGraph

    assert not hasattr(WrappedGraph, "to_dowhy")


# ------------------------------------------------------------ mode aliases


def test_full_and_summary_are_aliases_of_transition():
    from causalts.effects.graph_bridge import graph_to_networkx

    g, _ = _confounded()
    names = ["V0", "V1", "V2"]
    canonical = graph_to_networkx(g, var_names=names, mode="transition")
    for alias in ("full", "summary"):
        other = graph_to_networkx(g, var_names=names, mode=alias)
        assert nx.utils.graphs_equal(canonical, other)

    with pytest.raises(ValueError, match="mode must be one of"):
        graph_to_networkx(g, var_names=names, mode="joint")


# --------------------------------------------------- unwrapped DoWhy usage
#
# The export's whole premise is "get out of the way": a DoWhy-fluent user
# drives dowhy.CausalModel directly, using estimators or refuters this package
# never named. `build_identification_artifacts` + `backdoor.linear_regression`
# alone doesn't prove that -- it only proves parity with what estimate_effect()
# already offers. These exercise something this package does not wrap.


def test_raw_identify_effect_gets_dowhys_own_set_not_ours():
    """The premise for why force_parent_adjustment_set exists.

    V0 is the treatment V1's parent but has no other edges, so it does not
    confound the effect on V2 -- the unique correct backdoor set is empty.
    Driving `model.identify_effect()` directly, with nothing of ours in the
    loop, finds that. This package's own estimate_effect() would still adjust
    for `{V0_lag2}`: a harmless superset here, but a set a raw-export user
    who assumes it comes free from `identify_effect()` alone would not get.
    """
    dowhy = pytest.importorskip("dowhy")
    g = np.zeros((3, 3, 2), dtype=np.int8)
    g[0, 1, 1] = 1  # V0 -> V1 (treatment's parent, nothing else)
    g[1, 2, 1] = 1  # V1 -> V2 (the effect being estimated)
    rng = np.random.default_rng(0)
    x = np.zeros((T, 3))
    for t in range(1, T):
        x[t, 0] = 0.5 * x[t - 1, 0] + rng.normal()
        x[t, 1] = 0.6 * x[t - 1, 0] + rng.normal()
        x[t, 2] = 0.4 * x[t - 1, 1] + rng.normal()
    df = pd.DataFrame(x[10:], columns=["V0", "V1", "V2"])
    art = build_identification_artifacts(g, df, "V1", "V2", treatment_lag=1)

    model = dowhy.CausalModel(
        data=art.data,
        treatment=art.query["treatment"],
        outcome=art.query["outcome"],
        graph=art.graph,
    )
    identified = model.identify_effect(proceed_when_unidentifiable=True)
    assert identified.get_backdoor_variables() == []


def test_force_parent_adjustment_set_works_with_an_estimator_we_do_not_wrap():
    """A backdoor method estimate_effect() never special-cases: GLM.

    Proves the correctness guarantee (force + verify) is estimator-agnostic,
    not hardcoded to backdoor.linear_regression.
    """
    dowhy = pytest.importorskip("dowhy")
    sm = pytest.importorskip("statsmodels.api")
    from causalts.effects.graph_bridge import (
        force_parent_adjustment_set,
        verify_parent_adjustment_set,
    )

    g, df = _confounded()
    names = list(df.columns)
    art = build_identification_artifacts(g, df, "V1", "V2", treatment_lag=1)
    model = dowhy.CausalModel(
        data=art.data,
        treatment=art.query["treatment"],
        outcome=art.query["outcome"],
        graph=art.graph,
    )
    identified = model.identify_effect(proceed_when_unidentifiable=True)

    method = "backdoor.generalized_linear_model"
    parents = force_parent_adjustment_set(
        identified,
        g,
        "V1",
        1,
        "V2",
        names,
        method,
        available=list(art.data.columns),
    )
    assert parents == ["V0_lag2"]

    est = model.estimate_effect(
        identified,
        method_name=method,
        confidence_intervals=False,
        method_params={"glm_family": sm.families.Gaussian()},
    )
    verify_parent_adjustment_set(est, parents)  # must not raise
    assert abs(float(est.value) - 0.4) < 0.05
    assert "V0_lag2" in str(est.target_estimand)
    assert "V0_lag1" not in str(est.target_estimand).split("Estimand : 2")[0]


def test_force_parent_adjustment_set_is_a_no_op_for_non_backdoor_methods():
    from causalts.effects.graph_bridge import (
        force_parent_adjustment_set,
        verify_parent_adjustment_set,
    )

    result = force_parent_adjustment_set(
        object(),
        None,
        "V1",
        1,
        "V2",
        ["V0", "V1", "V2"],
        "iv.instrumental_variable",
    )
    assert result is None
    verify_parent_adjustment_set(object(), result)  # must not raise or inspect estimate


def test_verify_catches_a_genuine_mismatch():
    """The safety net force_parent_adjustment_set exists to be paired with."""
    from types import SimpleNamespace

    from causalts.effects.graph_bridge import verify_parent_adjustment_set

    fake_estimate = SimpleNamespace(
        target_estimand=SimpleNamespace(get_adjustment_set=lambda: ["V0_lag1"])
    )
    with pytest.raises(RuntimeError, match="did not reach the estimator"):
        verify_parent_adjustment_set(fake_estimate, ["V0_lag2"])


def test_the_helpers_are_exported_at_the_top_level():
    import causalts.effects as e

    for name in (
        "force_parent_adjustment_set",
        "verify_parent_adjustment_set",
        "parent_adjustment_set",
        "treatment_parents",
        "required_lag_depth",
    ):
        assert hasattr(e, name), f"{name} not exported from causalts.effects"


def test_renamed_dowhy_apis_are_bound_by_capability_not_version():
    """DoWhy renamed two APIs in 0.13; both names must resolve.

    0.11/0.12 export `construct_backdoor_estimand` and expose
    `get_backdoor_variables`; 0.13+ renamed these to
    `construct_adjustment_estimand` and `get_adjustment_set`. The bridge binds
    whichever is present rather than comparing version strings -- forks, conda
    rebuilds and `0.13.0.dev` builds all report versions that do not predict
    which symbol exists.
    """
    from causalts.effects.graph_bridge import (
        _adjustment_set_of,
        _construct_adjustment_estimand,
    )

    fn = _construct_adjustment_estimand()
    assert fn.__name__ in (
        "construct_adjustment_estimand",
        "construct_backdoor_estimand",
    )

    class OldEstimand:  # pre-0.13 surface
        def get_backdoor_variables(self):
            return ["V0_lag1"]

    class NewEstimand:  # 0.13+ surface
        def get_adjustment_set(self):
            return ["V0_lag2"]

        def get_backdoor_variables(self):  # still present upstream
            raise AssertionError("should prefer get_adjustment_set")

    assert _adjustment_set_of(OldEstimand()) == ["V0_lag1"]
    assert _adjustment_set_of(NewEstimand()) == ["V0_lag2"]


def test_identification_guard_names_the_broken_pair_not_just_dowhy(monkeypatch):
    """DoWhy <0.13 x networkx >=3.5 is a pair, and the message must say so.

    DoWhy below 0.13 calls `networkx.algorithms.d_separated`, removed in
    networkx 3.5. Either side alone is fine, so gating on the DoWhy version
    would reject working installs (0.12 + networkx 3.4). The guard fires only
    on the combination, and points at both escape routes.
    """
    import networkx as nx

    from causalts.effects import _compat

    class FakeDowhy:
        __version__ = "0.12"

    # networkx without d_separated (>=3.5) + DoWhy 0.12 -> must raise.
    monkeypatch.delattr(nx.algorithms, "d_separated", raising=False)
    monkeypatch.setitem(__import__("sys").modules, "dowhy", FakeDowhy)
    monkeypatch.setattr(_compat, "require_dowhy", lambda *a, **k: None)
    with pytest.raises(ImportError) as exc:
        _compat.require_identification("effect estimation")
    msg = str(exc.value)
    assert "dowhy>=0.13" in msg and "networkx<3.5" in msg
    assert "gcm" in msg  # tells gcm users they are unaffected

    # Same old DoWhy, but a networkx that still has d_separated -> fine.
    monkeypatch.setattr(
        nx.algorithms, "d_separated", lambda *a, **k: True, raising=False
    )
    _compat.require_identification("effect estimation")
