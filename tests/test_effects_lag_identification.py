# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Backdoor identification for lagged treatments.

The bridge used to add edges only into current-time nodes, which made every
lagged node a root. A lagged treatment then had no parents, `identify_effect`
found no backdoor path, and the "effect" was an unadjusted regression. These
tests pin the graph construction, the adjustment set, and recovery of known
structural coefficients on data where the naive estimate is demonstrably wrong.
"""

import networkx as nx
import numpy as np
import pandas as pd
import pytest

dowhy = pytest.importorskip("dowhy")

from causalts.effects.graph_bridge import (  # noqa: E402
    force_parent_adjustment_set,
    graph_to_dowhy_model,
    graph_to_networkx,
    parent_adjustment_set,
    required_lag_depth,
    treatment_parents,
    verify_parent_adjustment_set,
)

T_LONG = 40_000
TOL = 0.03


def _sim(step, d, T=T_LONG, seed=0):
    """Run a linear SEM defined by `step`, discarding the burn-in."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(T, d))
    x = np.zeros((T, d))
    for t in range(4, T):
        step(x, noise, t)
    return pd.DataFrame(x[4:], columns=[f"V{i}" for i in range(d)])


def _naive(df, treatment, lag, outcome):
    """Unadjusted regression of outcome(t) on treatment(t-lag)."""
    y = df[outcome].values[lag:]
    x = df[treatment].values[:-lag] if lag else df[treatment].values
    design = np.column_stack([np.ones(len(y)), x])
    return np.linalg.lstsq(design, y, rcond=None)[0][1]


# --------------------------------------------------------------- graph shape


def _ar_graph():
    """V0(t-1) -> V1(t), plus an AR self-loop on each variable."""
    g = np.zeros((2, 2, 2), dtype=np.int8)
    g[0, 1, 1] = 1
    g[0, 0, 1] = 1
    g[1, 1, 1] = 1
    return g, ["V0", "V1"]


def test_unrolled_gives_lagged_nodes_their_parents():
    g, names = _ar_graph()

    legacy = graph_to_networkx(g, var_names=names, mode="full", cycle_resolution="drop")
    assert list(legacy.predecessors("V1_lag1")) == []

    un = graph_to_networkx(
        g, var_names=names, mode="unrolled", horizon=2, cycle_resolution="drop"
    )
    assert sorted(un.predecessors("V1_lag1")) == ["V0_lag2", "V1_lag2"]
    assert sorted(un.predecessors("V1")) == ["V0_lag1", "V1_lag1"]
    assert nx.is_directed_acyclic_graph(un)


def test_unrolled_repeats_multi_lag_edges_and_truncates_at_horizon():
    # V1 -> V0 at lags 1 and 3, so both offsets must appear in every slice
    # where they fit, and neither may reach past the horizon.
    g = np.zeros((2, 2, 4), dtype=np.int8)
    g[1, 0, 1] = 1
    g[1, 0, 3] = 1
    un = graph_to_networkx(
        g, var_names=["V0", "V1"], mode="unrolled", horizon=3, cycle_resolution="drop"
    )
    assert sorted(un.predecessors("V0")) == ["V1_lag1", "V1_lag3"]
    assert sorted(un.predecessors("V0_lag1")) == ["V1_lag2"]  # V1_lag4 > horizon
    assert sorted(un.predecessors("V0_lag3")) == []  # boundary slice


def test_unrolled_requires_a_horizon_at_least_max_lag():
    g, names = _ar_graph()
    with pytest.raises(ValueError, match="requires an explicit horizon"):
        graph_to_networkx(g, var_names=names, mode="unrolled")
    with pytest.raises(ValueError, match="shallower than"):
        graph_to_networkx(g, var_names=names, mode="unrolled", horizon=0)


# ---------------------------------------------------------- adjustment set


def test_treatment_parents_read_off_the_summary_graph():
    g, names = _ar_graph()
    assert treatment_parents(g, "V1", 1, names) == [("V0", 2), ("V1", 2)]
    assert parent_adjustment_set(g, "V1", 1, names) == ["V0_lag2", "V1_lag2"]
    # A deeper treatment shifts every parent by the same amount.
    assert parent_adjustment_set(g, "V1", 2, names) == ["V0_lag3", "V1_lag3"]


def test_required_depth_follows_the_query_above_the_max_lag_floor():
    # A treatment whose only parent sits at lag 2 needs depth 3, not 6 -- every
    # surplus lag costs a usable row. But the unroll cannot go below the graph's
    # own max_lag without silently losing the deepest edges, so that is the floor.
    g = np.zeros((3, 3, 6), dtype=np.int8)
    g[0, 1, 2] = 1  # V0 -> V1 at lag 2
    g[2, 0, 5] = 1  # V2 -> V0 at lag 5
    names = ["V0", "V1", "V2"]
    assert required_lag_depth(g, "V1", 1, names) == 5  # floored at max_lag

    # With the deep edge gone the query drives the depth: 3, not the max_lag of 5.
    shallow = np.zeros((3, 3, 4), dtype=np.int8)
    shallow[0, 1, 2] = 1
    assert required_lag_depth(shallow, "V1", 1, names) == 3


def test_a_query_deeper_than_max_lag_still_wins():
    g = np.zeros((2, 2, 2), dtype=np.int8)
    g[0, 1, 1] = 1
    assert required_lag_depth(g, "V0", 4, ["V0", "V1"]) == 4


@pytest.mark.parametrize("treatment_lag", [1, 2])
def test_a_treatment_shallower_than_max_lag_does_not_crash(treatment_lag):
    """Regression: the horizon used to be taken below max_lag, which raises.

    Fires whenever `treatment_lag + deepest_parent_lag < max_lag` -- a treatment
    with no parents at all, say, on any graph with max_lag > 1.
    """
    g = np.zeros((3, 3, 4), dtype=np.int8)
    g[0, 1, 1] = 1  # V0 -> V1 at lag 1; V0 itself is parentless
    g[2, 0, 3] = 1  # ... except at lag 3, well beyond the query

    def step(x, e, t):
        x[t, 2] = 0.4 * x[t - 1, 2] + e[t, 2]
        x[t, 0] = 0.5 * x[t - 3, 2] + e[t, 0]
        x[t, 1] = 0.6 * x[t - 1, 0] + e[t, 1]

    df = _sim(step, 3, T=2_000)
    model, lagged = graph_to_dowhy_model(
        g,
        df,
        treatment="V0",
        outcome="V1",
        treatment_lag=treatment_lag,
        var_names=list(df.columns),
    )
    assert f"V0_lag{treatment_lag}" in lagged.columns


def test_unoriented_contemporaneous_neighbour_is_refused():
    # Recorded in both directions means discovery could not orient it; treating
    # it as a parent risks conditioning on a child of the treatment.
    g = np.zeros((2, 2, 2), dtype=np.int8)
    g[0, 1, 0] = 1
    g[1, 0, 0] = 1
    with pytest.raises(ValueError, match="unoriented"):
        parent_adjustment_set(g, "V1", 1, ["V0", "V1"])
    assert parent_adjustment_set(
        g, "V1", 1, ["V0", "V1"], strict_contemporaneous=False
    ) == ["V0_lag1"]


def test_missing_parent_column_is_an_error_not_a_silent_drop():
    g, names = _ar_graph()
    with pytest.raises(ValueError, match="not columns in the data"):
        parent_adjustment_set(g, "V1", 1, names, available=["V1"])


# ------------------------------------------------- end-to-end identification


def _ar_backdoor_case():
    """V0(t-1) -> V1(t) with AR on both.

    Backdoor: V0(t-1) <- V0(t-2) -> V1(t-1) -> V1(t) is absent, but
    V0(t-1) <- V0(t-2) and V1(t-1) <- V0(t-2) make V0(t-2) a common cause.
    """

    def step(x, e, t):
        x[t, 0] = 0.5 * x[t - 1, 0] + e[t, 0]
        x[t, 1] = 0.3 * x[t - 1, 1] + 0.4 * x[t - 1, 0] + e[t, 1]

    g = np.zeros((2, 2, 2), dtype=np.int8)
    g[0, 1, 1] = g[0, 0, 1] = g[1, 1, 1] = 1
    return step, 2, g, "V0", 1, "V1", 0.4


def _observed_confounder_case():
    """V0 confounds V1 and V2; the effect of V1(t-1) on V2(t) is 0.4."""

    def step(x, e, t):
        x[t, 0] = 0.5 * x[t - 1, 0] + e[t, 0]
        x[t, 1] = 0.6 * x[t - 1, 0] + e[t, 1]
        x[t, 2] = 0.4 * x[t - 1, 1] + 0.7 * x[t - 1, 0] + e[t, 2]

    g = np.zeros((3, 3, 2), dtype=np.int8)
    g[0, 0, 1] = g[0, 1, 1] = g[1, 2, 1] = g[0, 2, 1] = 1
    return step, 3, g, "V1", 1, "V2", 0.4


def _lag_two_treatment_case():
    """Treatment at lag 2. V1(t-1) is a mediator here and must NOT be adjusted."""

    def step(x, e, t):
        x[t, 0] = 0.5 * x[t - 1, 0] + e[t, 0]
        x[t, 1] = 0.3 * x[t - 1, 1] + 0.4 * x[t - 2, 0] + e[t, 1]

    g = np.zeros((2, 2, 3), dtype=np.int8)
    g[0, 0, 1] = g[1, 1, 1] = g[0, 1, 2] = 1
    return step, 2, g, "V0", 2, "V1", 0.4


def _contemporaneous_parent_case():
    """The treatment has a same-slice parent, V0(t) -> V1(t)."""

    def step(x, e, t):
        x[t, 0] = 0.5 * x[t - 1, 0] + e[t, 0]
        x[t, 1] = 0.8 * x[t, 0] + e[t, 1]
        x[t, 2] = 0.4 * x[t - 1, 1] + 0.6 * x[t - 1, 0] + e[t, 2]

    g = np.zeros((3, 3, 2), dtype=np.int8)
    g[0, 0, 1] = g[0, 1, 0] = g[1, 2, 1] = g[0, 2, 1] = 1
    return step, 3, g, "V1", 1, "V2", 0.4


CASES = {
    "ar_backdoor": _ar_backdoor_case,
    "observed_confounder": _observed_confounder_case,
    "lag_two_treatment": _lag_two_treatment_case,
    "contemporaneous_parent": _contemporaneous_parent_case,
}


@pytest.mark.parametrize("case_name", sorted(CASES))
def test_recovers_known_coefficient_where_naive_fails(case_name):
    from causalts.effects.effect import estimate_effect

    step, d, g, treat, lag, outcome, truth = CASES[case_name]()
    df = _sim(step, d)
    names = list(df.columns)

    # The case only proves something if the unadjusted estimate is wrong.
    naive = _naive(df, treat, lag, outcome)
    assert abs(naive - truth) > TOL, (
        f"{case_name} does not discriminate: naive estimate {naive:.4f} is already "
        f"within tolerance of {truth}"
    )

    est = estimate_effect(
        g,
        df,
        treatment=treat,
        outcome=outcome,
        treatment_lag=lag,
        var_names=names,
        confidence_intervals=False,
    )
    assert abs(float(est.value) - truth) < TOL, (
        f"{case_name}: got {float(est.value):.4f}, want {truth} +/- {TOL} "
        f"(naive was {naive:.4f})"
    )


def test_unconfounded_case_is_unchanged():
    """With no backdoor there is nothing to adjust for, and the answer stands."""
    from causalts.effects.effect import estimate_effect

    def step(x, e, t):
        x[t, 0] = e[t, 0]
        x[t, 1] = 0.5 * x[t - 1, 0] + e[t, 1]

    g = np.zeros((2, 2, 2), dtype=np.int8)
    g[0, 1, 1] = 1
    df = _sim(step, 2)
    assert parent_adjustment_set(g, "V0", 1, list(df.columns)) == []
    est = estimate_effect(
        g,
        df,
        treatment="V0",
        outcome="V1",
        treatment_lag=1,
        var_names=list(df.columns),
        confidence_intervals=False,
    )
    assert abs(float(est.value) - 0.5) < TOL


def test_parent_adjustment_set_is_actually_applied_to_the_estimand():
    """Regression test: the fitted estimator must have used the set we asked for.

    `set_backdoor_variables(arr, key=None)` defaults its key to
    `identifier_method`, which is still None right after `identify_effect()`. A
    write to `default_backdoor_id` looks correct at that instant -- the getter
    falls through to `default_backdoor_id` precisely because `identifier_method`
    is still unset -- but `model.estimate_effect(method_name="backdoor.X")`
    unconditionally overwrites `identifier_method` to `"backdoor"` afterwards
    (`CausalModel.estimate_effect` -> `set_identifier_method`), and
    `get_backdoor_variables()` resolves through *that* field, not
    `default_backdoor_id`. So a write to `default_backdoor_id` is a silent
    no-op: correct immediately, clobbered before the estimator is built, and the
    estimator quietly uses DoWhy's own minimal set instead.

    This was live in `_identify_and_estimate` and went undetected because, on
    every fixture in this file, DoWhy's own minimal set happens to ALSO be a
    valid (if less robust) backdoor set, so the point estimate still landed near
    the truth. The only way to catch it is to inspect what the fitted estimator
    actually used, not the estimand's state before the call that mutates it.
    """
    step, d, g, treat, lag, outcome, _ = _ar_backdoor_case()
    df = _sim(step, d, T=5_000)
    names = list(df.columns)

    model, lagged = graph_to_dowhy_model(
        g, df, treatment=treat, outcome=outcome, treatment_lag=lag, var_names=names
    )
    identified = model.identify_effect(proceed_when_unidentifiable=True)
    wanted = parent_adjustment_set(g, treat, lag, names, available=list(lagged.columns))
    assert wanted, "this fixture is meant to have a non-empty parent set"

    # The key `estimate_effect` will resolve `identifier_method` to for a
    # "backdoor.*" method name, not `default_backdoor_id`.
    identified.set_backdoor_variables(wanted, key="backdoor")
    estimate = model.estimate_effect(
        identified,
        method_name="backdoor.linear_regression",
        confidence_intervals=False,
        effect_modifiers=[],
    )
    used = estimate.estimator._target_estimand.get_backdoor_variables()
    assert sorted(used) == sorted(wanted)


def test_the_real_fix_updates_both_the_numeric_and_symbolic_estimand():
    """`_identify_and_estimate`'s actual (non-reimplemented) code path.

    Two things `set_backdoor_variables` does not do on its own: it does not
    survive `estimate_effect`'s key rewrite (the bug above), and even once
    fixed it only updates `backdoor_variables`, not the symbolic
    `estimands["backdoor"]` that `str(estimate.target_estimand)` renders. A fix
    that only patches the first leaves a numerically-correct estimate paired
    with a printed estimand that still names DoWhy's old confounder --
    correct number, wrong story. Also pins the linear estimator's own
    fit-time record of what it read, which is more specific than any estimand
    accessor.
    """
    from causalts.effects.effect import _identify_and_estimate

    step, d, g, treat, lag, outcome, _ = _ar_backdoor_case()
    df = _sim(step, d, T=5_000)
    names = list(df.columns)

    _, identified, estimate, lagged = _identify_and_estimate(
        g, df, treat, outcome, lag, confidence_intervals=False
    )
    wanted = parent_adjustment_set(g, treat, lag, names, available=list(lagged.columns))
    assert wanted, "this fixture is meant to have a non-empty parent set"

    assert sorted(estimate.target_estimand.get_adjustment_set()) == sorted(wanted)
    assert sorted(estimate.estimator._observed_common_causes_names) == sorted(wanted)
    rendered = str(estimate.target_estimand)
    for name in wanted:
        assert name in rendered, (
            f"the printed estimand does not mention {name}: numerically correct "
            "but tells the wrong story"
        )


def test_an_empty_parent_set_still_overwrites_a_stale_nonempty_one(monkeypatch):
    """A prior `if parents:` guard made the invariant vacuous exactly when the
    correct answer is "no adjustment needed": if DoWhy's own default happened
    to be a stale non-empty set, an empty `parents` would leave it untouched
    and unverified. The overwrite has to be unconditional.

    `_identify_and_estimate` builds its own `identify_effect()` call
    internally, so the stale value is seeded by monkeypatching `CausalModel`
    to return an estimand DoWhy has already been made to disagree with.
    """
    import dowhy

    from causalts.effects.effect import _identify_and_estimate

    def step(x, e, t):
        x[t, 0] = e[t, 0]
        x[t, 1] = 0.5 * x[t - 1, 0] + e[t, 1]

    df = _sim(step, 2, T=5_000)
    g = np.zeros((2, 2, 2), dtype=np.int8)
    g[0, 1, 1] = 1  # V0(t-1) -> V1(t), V0 itself has no parents: truly empty set

    real_identify_effect = dowhy.CausalModel.identify_effect

    def rigged_identify_effect(self, *args, **kwargs):
        estimand = real_identify_effect(self, *args, **kwargs)
        estimand.set_backdoor_variables(["V0_lag1"], key="backdoor")
        return estimand

    monkeypatch.setattr(dowhy.CausalModel, "identify_effect", rigged_identify_effect)

    _, _, estimate, _ = _identify_and_estimate(
        g, df, "V0", "V1", 1, confidence_intervals=False
    )
    assert estimate.target_estimand.get_adjustment_set() == []
    assert abs(float(estimate.value) - 0.5) < TOL


def test_disconnected_treatment_and_outcome_do_not_crash():
    """Regression: no path at all between treatment and outcome.

    When treatment and outcome share no causal path -- a genuine zero effect,
    not merely an unconfounded one -- DoWhy's own `identify_effect()` leaves
    `backdoor_variables` and `estimands` as `None` rather than an empty dict:
    there was nothing for its own search to identify. `force_parent_adjustment_set`
    assumed a dict to index into and crashed with a `TypeError` on `None[key] = ...`.
    """
    names = ["Heat", "IceCream", "Attacks"]
    g = np.zeros((3, 3, 2), dtype=np.int8)
    g[0, 1, 0] = 1  # Heat -> IceCream, contemporaneous
    g[1, 0, 0] = 1  # unresolved reverse
    g[0, 2, 1] = 1  # Heat -> Attacks, lag 1
    # No IceCream -> Attacks edge of any kind.

    def step(x, e, t):
        x[t, 0] = e[t, 0]
        x[t, 1] = x[t, 0] + e[t, 1]
        x[t, 2] = 0.5 * x[t - 1, 0] + e[t, 2]

    df = _sim(step, 3, T=5_000)
    df.columns = names

    from causalts.effects.effect import estimate_effect

    est = estimate_effect(
        g,
        df,
        treatment="IceCream",
        outcome="Attacks",
        treatment_lag=1,
        cycle_resolution="drop",
    )
    assert est.target_estimand.get_adjustment_set() == ["Heat_lag1"]
    assert abs(float(est.value)) < TOL


def test_ambiguous_neighbour_oriented_away_still_gets_a_column():
    """Regression: a non-strict parent the ancestral prune had no reason to keep.

    X1's contemporaneous edges to both X0 and X2 are unresolved. Non-strict
    treatment_parents() treats both as parents regardless of index order, but
    cycle_resolution="drop" only orients X0->X1 into the pruned DAG's ancestor
    set -- the X1<->X2 pair resolves as X1->X2, so X2 never becomes X1's
    graph-ancestor. force_parent_adjustment_set() still names X2_lag1, and it
    must be a real column, not just a claim `parent_adjustment_set` makes.
    """
    names = ["X0", "X1", "X2"]
    g = np.zeros((3, 3, 1), dtype=np.int8)
    g[0, 1, 0] = g[1, 0, 0] = 1  # ambiguous X0<->X1
    g[1, 2, 0] = g[2, 1, 0] = 1  # ambiguous X1<->X2

    def step(x, e, t):
        x[t, 0] = e[t, 0]
        x[t, 1] = x[t, 0] + e[t, 1]
        x[t, 2] = x[t, 1] + e[t, 2]

    df = _sim(step, 3, T=2_000)
    df.columns = names

    from causalts.effects.effect import estimate_effect

    est = estimate_effect(
        g, df, treatment="X1", outcome="X2", treatment_lag=1, cycle_resolution="drop"
    )
    assert sorted(est.target_estimand.get_adjustment_set()) == ["X0_lag1", "X2_lag1"]


def test_frame_is_pruned_to_the_ancestral_subgraph():
    """Variables outside An(treatment) u An(outcome) must not enter the frame."""

    def step(x, e, t):
        x[t, 0] = 0.5 * x[t - 1, 0] + e[t, 0]
        x[t, 1] = 0.4 * x[t - 1, 0] + e[t, 1]
        x[t, 2] = 0.4 * x[t - 1, 1] + e[t, 2]
        x[t, 3] = 0.9 * x[t - 1, 3] + e[t, 3]  # wholly disconnected

    g = np.zeros((4, 4, 2), dtype=np.int8)
    g[0, 0, 1] = g[0, 1, 1] = g[1, 2, 1] = g[3, 3, 1] = 1
    df = _sim(step, 4, T=5_000)
    names = list(df.columns)

    _, lagged = graph_to_dowhy_model(
        g, df, treatment="V1", outcome="V2", treatment_lag=1, var_names=names
    )
    assert not [
        c for c in lagged.columns if c.startswith("V3")
    ], f"disconnected variable survived pruning: {list(lagged.columns)}"
    assert "V1_lag1" in lagged.columns and "V2" in lagged.columns


def test_forced_parents_close_a_backdoor_dowhys_own_search_cannot_see():
    """A grandparent confounder DoWhy's own search cannot be made to see in general.

    U confounds the treatment and outcome through two separate two-hop chains
    (U->P1->A and U->N1->N2->Y), with every individual edge at lag 1.
    required_lag_depth() only sizes the horizon for the treatment's own direct
    parent (P1) -- deep enough for force_parent_adjustment_set(), which blocks
    every path through P1 regardless of what lies behind it -- so U never gets
    its own slice, and DoWhy's own identify_effect() sees no backdoor path at
    all and returns the empty set even though the graph is otherwise complete
    and correct. Deepening the horizon to expose U is not a general fix: an AR
    self-loop anywhere upstream makes the exact required depth unbounded, so
    this is the reason estimate_effect() forces parents rather than trusting
    DoWhy's raw search, not a bug required_lag_depth() should chase.
    """
    names = ["U", "P1", "A", "N1", "N2", "Y"]

    def step(x, e, t):
        x[t, 0] = 0.3 * x[t - 1, 0] + e[t, 0]  # U
        x[t, 1] = 0.7 * x[t - 1, 0] + e[t, 1]  # P1 <- U
        x[t, 2] = 0.7 * x[t - 1, 1] + e[t, 2]  # A <- P1
        x[t, 3] = 0.7 * x[t - 1, 0] + e[t, 3]  # N1 <- U
        x[t, 4] = 0.7 * x[t - 1, 3] + e[t, 4]  # N2 <- N1
        x[t, 5] = 0.6 * x[t - 1, 2] + 0.7 * x[t - 1, 4] + e[t, 5]  # Y <- A, N2

    g = np.zeros((6, 6, 2), dtype=np.int8)
    g[0, 1, 1] = g[1, 2, 1] = g[0, 3, 1] = g[3, 4, 1] = g[2, 5, 1] = g[4, 5, 1] = 1

    assert required_lag_depth(g, "A", 1, names) == 2

    df = _sim(step, 6, T=3_000)
    df.columns = names
    model, lagged_df = graph_to_dowhy_model(
        g, df, treatment="A", outcome="Y", treatment_lag=1, var_names=names
    )

    # DoWhy's own search can't see U: it's outside the horizon.
    identified_raw = model.identify_effect(proceed_when_unidentifiable=True)
    assert identified_raw.get_backdoor_variables() == []

    # Forcing the parent blocks the same backdoor path anyway.
    identified_forced = model.identify_effect(proceed_when_unidentifiable=True)
    parents = force_parent_adjustment_set(
        identified_forced,
        g,
        "A",
        1,
        "Y",
        names,
        method="backdoor.linear_regression",
        available=list(lagged_df.columns),
    )
    assert parents == ["P1_lag2"]
    est = model.estimate_effect(
        identified_forced,
        method_name="backdoor.linear_regression",
        control_value=0,
        treatment_value=1,
        effect_modifiers=[],
    )
    verify_parent_adjustment_set(est, parents)
    assert abs(float(est.value) - 0.6) < 0.05
