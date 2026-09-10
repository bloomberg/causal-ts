# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Validating a lag embedding as a transition graph rather than a joint DAG.

`falsify_graph` handed the embedding to DoWhy, which tests every root against
every non-descendant unconditionally. Every lagged node is a root by
construction and in any autoregressive process they are dependent, so those
tests fail whatever the graph says: on the ground-truth graph of a d=4 VAR(1)
they take 3/22 violations to 21/44, against 9/22 -> 27/44 when a latent
confounder is genuinely present. `validate_transition_graph` tests only the
nodes the graph actually models.
"""

import numpy as np
import pandas as pd
import pytest

from causalts.effects.validate import (
    HistorySufficiencyResult,
    TransitionValidationResult,
    _holm,
    history_sufficiency,
    linear_ci_test,
    validate_transition_graph,
)

D = 4
NAMES = [f"X{i}" for i in range(D)]
T = 3000


def _var1(seed=0, latent=False, drop_edge=False, ar=0.5):
    """Chain X0 -> X1 -> X2 at lag 1, with X3 isolated and AR on every series.

    `latent` adds a shared unobserved shock to X0, X1, X2 -- confounding that
    the graph denies. `drop_edge` instead makes the DATA match the graph but
    removes X0 -> X1 from the data, which the graph still asserts.
    """
    rng = np.random.default_rng(seed)
    burn = 200
    n = T + burn
    x = np.zeros((n, D))
    f = rng.normal(size=n)
    for t in range(1, n):
        s = 1.2 * f[t] if latent else 0.0
        x[t, 0] = ar * x[t - 1, 0] + s + rng.normal()
        x[t, 1] = ar * x[t - 1, 1] + (0.0 if drop_edge else 0.6) * x[t - 1, 0] + s
        x[t, 1] += rng.normal()
        x[t, 2] = ar * x[t - 1, 2] + 0.6 * x[t - 1, 1] + s + rng.normal()
        x[t, 3] = ar * x[t - 1, 3] + rng.normal()
    return pd.DataFrame(x[burn:], columns=NAMES)


def _chain_graph():
    g = np.zeros((D, D, 2), dtype=np.int8)
    for i in range(D):
        g[i, i, 1] = 1
    g[0, 1, 1] = 1
    g[1, 2, 1] = 1
    return g


def _validate(df, graph=None, **kw):
    return validate_transition_graph(
        _chain_graph() if graph is None else graph,
        df,
        conditional_independence_test=linear_ci_test,
        **kw,
    )


# ------------------------------------------------------------------ Holm


def test_holm_is_step_down_and_monotone():
    adj = _holm({"a": 0.01, "b": 0.02, "c": 0.5})
    assert adj["a"] == pytest.approx(0.03)  # 3 * 0.01
    assert adj["b"] == pytest.approx(0.04)  # 2 * 0.02
    assert adj["c"] == pytest.approx(0.5)  # 1 * 0.5
    # Enforced monotonicity: a later, larger raw p cannot get a smaller
    # adjusted p than an earlier one.
    adj = _holm({"a": 0.04, "b": 0.041})
    assert adj["b"] >= adj["a"]
    assert _holm({}) == {}


def test_holm_caps_at_one():
    assert _holm({"a": 0.6, "b": 0.7})["a"] == 1.0


# ------------------------------------------------- what actually gets tested


def test_only_current_time_nodes_are_tested():
    """The whole point: history nodes carry no independence claim."""
    res = _validate(_var1())
    assert set(res.p_values) == set(NAMES)
    assert res.n_tests == D
    assert not any("_lag" in node for node in res.p_values)


def test_history_dependence_alone_does_not_reject():
    """A correct graph on strongly autocorrelated data must survive.

    This is the case the old joint-DAG treatment could not pass: at ar=0.9 the
    lagged nodes are heavily mutually dependent, and every one of those
    dependencies used to count as a violation.
    """
    res = _validate(_var1(ar=0.9))
    assert not res.rejected, res.to_frame()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_correct_graph_is_not_rejected(seed):
    assert not _validate(_var1(seed=seed)).rejected


def test_latent_confounding_is_rejected():
    res = _validate(_var1(latent=True))
    assert res.rejected
    # The confounder touches X0, X1 and X2 but not the isolated X3.
    assert "X3" not in res.rejected_nodes
    assert set(res.rejected_nodes) >= {"X1", "X2"}


def test_a_missing_edge_is_rejected_at_the_child():
    """The graph asserts X0(t-1) -> X1(t); make the data disagree."""
    g = _chain_graph()
    g[0, 1, 1] = 0
    res = _validate(_var1(), graph=g)
    assert res.rejected and "X1" in res.rejected_nodes


def test_a_spurious_edge_is_not_a_markov_violation():
    """Adding an edge only weakens the claim, so LMC cannot detect it.

    Pinned so nobody later reads a clean result as "the graph is right".
    Superfluous edges are a faithfulness/minimality question, which is what
    refute_structure's edge-dependence tests are for.
    """
    g = _chain_graph()
    g[3, 2, 1] = 1  # X3 does not cause X2 in the data
    assert not _validate(_var1(), graph=g).rejected


def test_result_frame_and_repr():
    res = _validate(_var1(latent=True))
    frame = res.to_frame()
    assert list(frame.columns) == [
        "node",
        "p_value",
        "adjusted_p_value",
        "rejected",
    ]
    assert len(frame) == res.n_tests
    assert frame["p_value"].is_monotonic_increasing
    assert isinstance(res, TransitionValidationResult)
    assert "REJECTED" in repr(res)
    assert "not rejected" in repr(_validate(_var1()))


def test_isolated_node_with_no_parents_is_still_tested():
    """include_unconditional=False would have skipped this legitimate test.

    A current-time node with no parents genuinely does claim independence from
    everything else; only the *history* roots do not.
    """
    g = np.zeros((2, 2, 2), dtype=np.int8)
    g[0, 0, 1] = 1  # X1 has no parents at all
    rng = np.random.default_rng(0)
    x = rng.normal(size=(T, 2))
    x[:, 1] = x[:, 0] + 0.1 * rng.normal(size=T)  # but is not independent
    df = pd.DataFrame(x, columns=["X0", "X1"])
    res = validate_transition_graph(g, df, conditional_independence_test=linear_ci_test)
    assert "X1" in res.p_values and not res.skipped
    assert "X1" in res.rejected_nodes


# ------------------------------------------------------- history sufficiency


def _var2(seed=0, second_order=True):
    rng = np.random.default_rng(seed)
    burn = 200
    n = T + burn
    x = np.zeros((n, 2))
    for t in range(2, n):
        x[t, 0] = 0.4 * x[t - 1, 0] + rng.normal()
        x[t, 1] = 0.3 * x[t - 1, 1] + 0.5 * x[t - 1, 0]
        if second_order:
            x[t, 1] += 0.6 * x[t - 2, 0]
        x[t, 1] += rng.normal()
    return pd.DataFrame(x[burn:], columns=["X0", "X1"])


def test_history_sufficiency_accepts_a_deep_enough_window():
    res = history_sufficiency(
        _var2(second_order=False),
        max_lag=1,
        conditional_independence_test=linear_ci_test,
    )
    assert isinstance(res, HistorySufficiencyResult)
    assert not res.rejected, res.to_frame()


def test_history_sufficiency_catches_a_too_shallow_window():
    res = history_sufficiency(
        _var2(second_order=True),
        max_lag=1,
        conditional_independence_test=linear_ci_test,
    )
    assert res.rejected and res.rejected_variables == ("X1",)
    # Deepening the window fixes it, which is the actionable half of the report.
    deeper = history_sufficiency(
        _var2(second_order=True),
        max_lag=2,
        conditional_independence_test=linear_ci_test,
    )
    assert not deeper.rejected


def test_history_sufficiency_ignores_dependence_among_lagged_nodes():
    """The diagnostic root-vs-root tests would have flagged this; it must not.

    A pure VAR(1) with strong AR has heavily dependent lagged nodes and an
    entirely sufficient window of 1. Dependence among history is not evidence
    about depth.
    """
    res = history_sufficiency(
        _var1(ar=0.9), max_lag=1, conditional_independence_test=linear_ci_test
    )
    assert not res.rejected, res.to_frame()


def test_history_sufficiency_uses_the_saturated_window_not_graph_parents():
    """An edge missing inside the window must not read as "window too short".

    X1 depends on X0 at lag 1 only. If the diagnostic conditioned on a graph
    that omitted that edge, X0_lag2 would predict X1 and the window would look
    too shallow. Conditioning on the full window makes it a non-event -- and
    the function takes no graph at all, which is how that is guaranteed.
    """
    res = history_sufficiency(
        _var2(second_order=False),
        max_lag=1,
        extra_lags=3,
        conditional_independence_test=linear_ci_test,
    )
    assert not res.rejected


def test_history_sufficiency_rejects_degenerate_arguments():
    df = _var2()
    with pytest.raises(ValueError, match="extra_lags must be at least 1"):
        history_sufficiency(df, max_lag=1, extra_lags=0)
    with pytest.raises(ValueError, match="max_lag must be at least 1"):
        history_sufficiency(df, max_lag=0)


# ------------------------------------------------------------- the CI helper


def test_linear_ci_test_separates_dependence_from_independence():
    rng = np.random.default_rng(0)
    n = 2000
    z = rng.normal(size=(n, 1))
    x = z + rng.normal(size=(n, 1))
    y = z + rng.normal(size=(n, 1))
    assert linear_ci_test(x, y) < 1e-6  # dependent unconditionally
    assert linear_ci_test(x, y, z) > 0.01  # independent given z
    # Multivariate Y is the case the local Markov tests need.
    y2 = np.column_stack([y, rng.normal(size=n)])
    assert linear_ci_test(x, y2, z) > 0.01


def test_linear_ci_test_returns_nan_when_it_cannot_run():
    """Not testable and no-dependence-found must not look the same.

    Returning 1.0 here would let an under-powered design read as a clean pass.
    """
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 1))
    y = rng.normal(size=(3, 3))
    with pytest.warns(RuntimeWarning, match="no residual degrees of freedom"):
        assert np.isnan(linear_ci_test(x, y))


def test_linear_ci_test_handles_an_exact_fit_by_scale_not_by_zero():
    """An exact fit leaves ~1e-28 residuals, not 0.

    An absolute `rss <= 0` guard never fires there, and the F ratio is then
    built from two pieces of floating-point noise -- which produced a confident
    p=3e-07 for a case that is exactly conditionally independent.
    """
    rng = np.random.default_rng(0)
    y = rng.normal(size=(200, 1))
    z = rng.normal(size=(200, 1))
    assert linear_ci_test(2.0 * y, y) == 0.0  # Y explains X exactly
    assert linear_ci_test(2.0 * z, y, z) == 1.0  # Z already did; Y adds nothing
    assert linear_ci_test(np.ones((200, 1)), y) == 1.0  # constant X


def test_linear_ci_test_refuses_a_multivariate_x():
    """Silently testing X[:, 0] would return a plausible-looking wrong answer."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="univariate X"):
        linear_ci_test(rng.normal(size=(50, 2)), rng.normal(size=(50, 1)))


def test_an_untestable_node_is_skipped_not_passed():
    """A NaN p-value must not sail through Holm and read as a clean node."""
    g = _chain_graph()
    df = _var1().iloc[:6]  # far too few rows for the saturated design
    with pytest.warns(RuntimeWarning):
        res = _validate(df, graph=g)
    assert not res.p_values and set(res.skipped) == set(NAMES)
    assert not res.rejected and res.n_tests == 0


def test_history_sufficiency_skips_rather_than_passing_when_untestable():
    df = _var2().iloc[:6]
    with pytest.warns(RuntimeWarning):
        res = history_sufficiency(
            df, max_lag=1, conditional_independence_test=linear_ci_test
        )
    assert set(res.skipped) == {"X0", "X1"} and not res.p_values
    assert "UNTESTED" in repr(res)


def test_results_are_hashable_and_do_not_raise_on_equality():
    """frozen=True auto-generates __hash__ over unhashable dict fields."""
    a, b = _validate(_var1()), _validate(_var1())
    assert hash(a) != hash(b) and a != b
    assert hash(
        history_sufficiency(
            _var2(), max_lag=1, conditional_independence_test=linear_ci_test
        )
    )


# ------------------------------------------------------------- deprecation


def test_falsify_graph_warns_that_it_uses_joint_dag_semantics():
    pytest.importorskip("dowhy")
    from causalts.effects.validate import falsify_graph

    df = _var1()
    with pytest.warns(DeprecationWarning, match="validate_transition_graph"):
        falsify_graph(_chain_graph(), df.iloc[:200], n_permutations=2)


def test_the_default_ci_test_is_the_one_that_was_validated():
    """kernel_based was the original default and is wrong at realistic width.

    history_sufficiency conditions on the saturated window, `d * max_lag`
    columns. On ex3 (d=11, max_lag=3, T=500) the kernel test rejected all 11
    variables on data generated by a VAR(3) -- where the window is exactly deep
    enough. The Monte Carlo calibration was run with the linear test, so that
    is what ships.
    """
    from causalts.effects import validate as v

    res = validate_transition_graph(_chain_graph(), _var1())
    assert not res.rejected
    hist = history_sufficiency(_var2(second_order=False), max_lag=1)
    assert not hist.rejected
    # No DoWhy import is needed for the default path.
    assert v.linear_ci_test is not None


def test_untested_is_distinguishable_from_passed():
    """Regression: `rejected=False` alone conflates two different outcomes.

    When every per-node test is skipped, `rejected_nodes` is empty and
    `rejected` is `False` -- indistinguishable, to a caller writing
    `if result.rejected:`, from a run where the graph was tested and held up.
    `.tested` is the signal that separates "no evidence" from "no violation".
    """
    untested = TransitionValidationResult(
        rejected=False,
        rejected_nodes=(),
        p_values={},
        adjusted_p_values={},
        significance_level=0.05,
        n_tests=0,
        skipped={"X0": "no testable non-descendants"},
    )
    assert untested.tested is False
    assert untested.rejected is False  # ...which is exactly the trap
    assert "UNTESTED" in repr(untested)

    # A real run on a correct graph: same `rejected`, opposite `tested`.
    passed = _validate(_var1())
    assert passed.tested is True
    assert passed.rejected is False
    assert "UNTESTED" not in repr(passed)


def test_history_sufficiency_untested_is_distinguishable_from_sufficient():
    """Same trap on the depth diagnostic: `if result.rejected: deepen()`.

    An all-skipped run must not read as "max_lag is deep enough", or a caller
    silently declines to deepen an embedding that was never assessed.
    """
    untested = HistorySufficiencyResult(
        rejected=False,
        rejected_variables=(),
        p_values={},
        adjusted_p_values={},
        significance_level=0.05,
        max_lag=2,
        extra_lags=2,
        skipped={"X0": "no residual degrees of freedom"},
    )
    assert untested.tested is False
    assert untested.rejected is False
    assert "UNTESTED" in repr(untested)
