# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Stress-testing an estimated effect.

`estimate_effect` returns a number that is correct *if the graph is correct*.
These cover the two ways of probing that proviso: DoWhy's refuters, which
perturb the data and ask whether the estimator holds still, and the
unobserved-confounder sensitivity analysis.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("dowhy")

from causalts.effects.validate import (  # noqa: E402
    REFUTERS,
    _refutation_verdict,
    refute_effect,
    sensitivity_analysis,
)

T = 1200
TRUTH = 0.4


def _confounded(seed=0):
    """V0 confounds V1 and V2; the effect of V1(t-1) on V2(t) is 0.4."""
    rng = np.random.default_rng(seed)
    x = np.zeros((T, 3))
    for t in range(1, T):
        x[t, 0] = 0.5 * x[t - 1, 0] + rng.normal()
        x[t, 1] = 0.6 * x[t - 1, 0] + rng.normal()
        x[t, 2] = TRUTH * x[t - 1, 1] + 0.7 * x[t - 1, 0] + rng.normal()
    g = np.zeros((3, 3, 2), dtype=np.int8)
    g[0, 0, 1] = g[0, 1, 1] = g[1, 2, 1] = g[0, 2, 1] = 1
    return g, pd.DataFrame(x[10:], columns=["V0", "V1", "V2"])


@pytest.mark.parametrize("method", REFUTERS)
def test_every_refuter_runs_and_passes_on_a_correct_graph(method):
    g, df = _confounded()
    r = refute_effect(g, df, "V1", "V2", 1, method=method, num_simulations=10)
    assert r["refuter"] == method
    assert abs(r["estimated_effect"] - TRUTH) < 0.08
    assert r["passed"], r["summary"]


def test_placebo_drives_the_effect_to_zero():
    """The refuter that actually discriminates: no treatment, no effect."""
    g, df = _confounded()
    r = refute_effect(g, df, "V1", "V2", 1, num_simulations=100)
    assert r["reference_effect"] == 0.0
    assert abs(r["new_effect"]) < 0.05
    assert r["passed"]


def test_the_non_placebo_refuters_hold_the_estimate_still():
    g, df = _confounded()
    for method in ("random_common_cause", "data_subset_refuter"):
        r = refute_effect(g, df, "V1", "V2", 1, method=method, num_simulations=10)
        assert r["reference_effect"] == r["estimated_effect"]
        assert abs(r["new_effect"] - r["estimated_effect"]) < 0.05


def test_unknown_refuter_names_the_alternatives():
    g, df = _confounded()
    with pytest.raises(ValueError, match="Unknown refuter"):
        refute_effect(g, df, "V1", "V2", 1, method="nope")


def test_refutation_attacks_the_same_estimand_that_was_reported():
    """It must not rebuild the model and pick up DoWhy's minimal backdoor set.

    `estimate_effect` forces the treatment's parents onto the estimand. If
    refutation re-identified independently it would attack a different
    quantity than the one the user was given.
    """
    from causalts.effects.effect import _identify_and_estimate
    from causalts.effects.graph_bridge import parent_adjustment_set

    g, df = _confounded()
    _, identified, estimate, lagged = _identify_and_estimate(
        g, df, "V1", "V2", 1, confidence_intervals=False
    )
    wanted = parent_adjustment_set(
        g, "V1", 1, list(df.columns), available=list(lagged.columns)
    )
    assert sorted(identified.get_backdoor_variables()) == sorted(wanted)

    r = refute_effect(g, df, "V1", "V2", 1, num_simulations=10)
    assert r["estimated_effect"] == pytest.approx(float(estimate.value))


def test_the_ate_carries_no_effect_modifiers():
    """DoWhy would otherwise fit treatment x modifier interactions.

    That is a CATE model whose .value is an average, not the ATE the function
    documents -- and the variables it picks are an artefact of the ancestral
    prune. It also blocks DoWhy's linear sensitivity analysis.
    """
    from causalts.effects.effect import _identify_and_estimate

    g, df = _confounded()
    _, _, estimate, _ = _identify_and_estimate(
        g, df, "V1", "V2", 1, confidence_intervals=False
    )
    assert not estimate.estimator._effect_modifier_names


def test_sensitivity_analysis_fits_against_the_documented_adjustment_set():
    """Regression test for the estimand-key bug this release found and fixed.

    `sensitivity_analysis` shares `_identify_and_estimate` with `estimate_effect`
    and `refute_effect`, so it inherits the same risk: DoWhy's e-value only needs
    `estimate.value` and its standard error, so it would happily compute a
    confident-looking number from an estimator that silently used DoWhy's own
    minimal backdoor set instead of the documented parent set. This fixture is
    chosen because the two sets genuinely differ here (V1_lag1 vs V0_lag3), so
    the check is not vacuous the way it would be on `_confounded()`, where they
    happen to coincide.
    """
    from causalts.effects.effect import _identify_and_estimate

    def step(x, e, t):
        x[t, 0] = 0.5 * x[t - 1, 0] + e[t, 0]
        x[t, 1] = 0.3 * x[t - 1, 1] + 0.4 * x[t - 2, 0] + e[t, 1]

    rng = np.random.default_rng(0)
    n = 2000
    noise = rng.normal(size=(n, 2))
    x = np.zeros((n, 2))
    for t in range(4, n):
        step(x, noise, t)
    df = pd.DataFrame(x[4:], columns=["V0", "V1"])
    g = np.zeros((2, 2, 3), dtype=np.int8)
    g[0, 0, 1] = g[1, 1, 1] = g[0, 1, 2] = 1

    _, _, estimate, _ = _identify_and_estimate(
        g, df, "V0", "V1", 2, confidence_intervals=False
    )
    used = estimate.estimator._target_estimand.get_backdoor_variables()
    assert used == ["V0_lag3"], (
        f"expected the documented parent set, got {used} -- this is DoWhy's own "
        "minimal set, meaning the estimand-key fix regressed"
    )

    s = sensitivity_analysis(g, df, "V0", "V1", 2)
    assert abs(s["estimated_effect"] - 0.4) < 0.05


def test_sensitivity_analysis_runs_with_no_extra_arguments():
    g, df = _confounded()
    s = sensitivity_analysis(g, df, "V1", "V2", 1)
    assert s["simulation_method"] == "e-value"
    assert abs(s["estimated_effect"] - TRUTH) < 0.08
    assert "E-value" in s["summary"] or "e-value" in s["summary"].lower()


def test_result_objects_expose_both():
    from causalts.effects.wrap import wrap_graph

    g, df = _confounded()
    wrapped = wrap_graph(g, df)
    assert wrapped.refute_effect("V1", "V2", num_simulations=10)["passed"]
    assert wrapped.sensitivity_analysis("V1", "V2")["simulation_method"] == "e-value"


# ------------------------------------------------------- the verdict logic
#
# Every integration test above asserts a PASS, so a wrapper hardcoded to
# `passed=True` would satisfy all of them. These drive the decision directly.


@pytest.mark.parametrize(
    "p_value,expected",
    [
        (0.40, True),
        (0.06, True),
        (0.05, False),  # DoWhy calls p <= alpha significant, so equality fails
        (0.04, False),
        (0.001, False),
        (None, None),  # no verdict is a third outcome, not a failure
    ],
)
def test_verdict_polarity_and_boundary(p_value, expected):
    assert _refutation_verdict("bootstrap_refuter", 0.4, p_value, 0.05) is expected


def test_no_p_value_is_undetermined_rather_than_a_pass():
    """A missing p-value has no verdict, and must not be read as one.

    Falling back to an effect-size rule -- accept if the displacement is under
    some fraction of the original estimate -- would invent a verdict from a
    quantity unrelated to the placebo null: against an original effect of 100,
    a 10% rule accepts a placebo estimate of 9.
    """
    assert _refutation_verdict("placebo_treatment_refuter", 0.0, None, 0.05) is None


def test_undetermined_survives_into_the_public_refute_effect_result(monkeypatch):
    """Regression: the tri-state has to reach the caller, not just the helper.

    `refute_effect` returned `bool(passed)`, and `bool(None)` is `False` -- so a
    refutation DoWhy could not compute was reported as one the estimate failed,
    while the docstring promised `None`. The test above pins the helper, which
    stayed correct throughout; only the public dict was wrong, so only a test
    at this level catches it.
    """
    from causalts.effects import validate as validate_mod

    monkeypatch.setattr(
        validate_mod, "_refutation_verdict", lambda *a, **k: None, raising=True
    )
    g, df = _confounded()
    out = refute_effect(g, df, "V1", "V2", 1, num_simulations=5)
    assert out["passed"] is None


def test_the_placebo_is_not_a_column_of_zeros():
    """Regression: DoWhy's default placebo for a float treatment is degenerate.

    `new_treatment = randn(n) * DEFAULT_STD_DEV_OF_NORMAL + DEFAULT_MEAN_OF_NORMAL`
    with both constants 0 -- a constant column, so the estimate is exactly 0
    whatever the estimator does, and the refutation proves nothing. Every
    treatment here is a float, so we permute instead. If this ever reverts,
    `new_effect` becomes exactly 0.0 rather than merely small.
    """
    g, df = _confounded()
    r = refute_effect(g, df, "V1", "V2", 1, num_simulations=100)
    assert r["new_effect"] != 0.0, "the placebo collapsed to DoWhy's zero column"
    assert abs(r["new_effect"]) < 0.05
    assert r["passed"]


def test_the_significance_reference_is_read_per_refuter():
    """It cannot be taken from one field.

    Placebo reports the *original* estimate in `estimated_effect` but tests a
    separately constructed zero; dummy-outcome reports the known injected
    effect and tests exactly that; the resamplers test the original.
    """
    g, df = _confounded()
    placebo = refute_effect(g, df, "V1", "V2", 1, num_simulations=100)
    assert placebo["reference_effect"] == 0.0
    assert placebo["estimated_effect"] > 0.3  # ...while still reporting the ATE

    dummy = refute_effect(
        g, df, "V1", "V2", 1, method="dummy_outcome_refuter", num_simulations=10
    )
    assert dummy["reference_effect"] == 0.0

    boot = refute_effect(
        g, df, "V1", "V2", 1, method="bootstrap_refuter", num_simulations=10
    )
    assert boot["reference_effect"] == pytest.approx(boot["estimated_effect"])


def test_a_refuter_that_should_fire_does_fire():
    """Integration negative control: corrupt the confounder the estimate needs.

    W confounds T and Y. Adjusted, the effect is 0.4. The bootstrap refuter is
    asked to inject heavy noise into W specifically, which destroys its value
    as a control, so the simulated estimates drift towards the confounded
    association and away from the reported estimate.
    """
    rng = np.random.default_rng(0)
    n = 1200
    x = np.zeros((n, 3))
    for t in range(1, n):
        x[t, 0] = 0.5 * x[t - 1, 0] + rng.normal()  # W
        x[t, 1] = 0.8 * x[t - 1, 0] + 0.6 * rng.normal()  # T
        x[t, 2] = 0.4 * x[t - 1, 1] + 2.0 * x[t - 1, 0] + rng.normal()  # Y
    df = pd.DataFrame(x[10:], columns=["W", "T", "Y"])
    g = np.zeros((3, 3, 2), dtype=np.int8)
    g[0, 0, 1] = g[0, 1, 1] = g[1, 2, 1] = g[0, 2, 1] = 1

    clean = refute_effect(
        g, df, "T", "Y", 1, method="bootstrap_refuter", num_simulations=100
    )
    assert clean["passed"], "the control itself must survive an honest bootstrap"

    corrupted = refute_effect(
        g,
        df,
        "T",
        "Y",
        1,
        method="bootstrap_refuter",
        num_simulations=100,
        required_variables=["W_lag2"],
        noise=5.0,
        probability_of_change=0.5,
    )
    assert (
        corrupted["passed"] is False
    ), f"corrupting the confounder should refute the estimate, got {corrupted}"


# --------------------------------------------------------- analytic checks


def test_evalue_matches_the_vanderweele_ding_formula():
    """Recompute DoWhy's e-value from first principles and require agreement.

    The chain, read off DoWhy 0.14's source rather than assumed:
    SMD = coef * delta / sd(outcome), RR = exp(0.91 * SMD),
    E = RR + sqrt(RR * (RR - 1)).

    Note `sd` is the *outcome* standard deviation, `np.std(data[outcome])` with
    ddof=0 -- DoWhy's own docstring calls it the residual standard deviation,
    which it is not.
    """
    from causalts.effects.effect import _identify_and_estimate

    g, df = _confounded()
    _, _, estimate, lagged = _identify_and_estimate(
        g, df, "V1", "V2", 1, confidence_intervals=False
    )
    coef = float(estimate.value)
    smd = coef / float(np.std(lagged["V2"]))
    rr = np.exp(0.91 * smd)
    expected_rr = rr if rr > 1 else 1 / rr
    expected_evalue = expected_rr + np.sqrt(expected_rr * (expected_rr - 1))

    stats = sensitivity_analysis(g, df, "V1", "V2", 1)["details"].stats
    assert stats["converted_estimate"] == pytest.approx(rr, rel=1e-9)
    assert stats["evalue_estimate"] == pytest.approx(expected_evalue, rel=1e-9)
    # Sanity on the interpretation: a confounder must carry at least this much
    # association with both treatment and outcome to explain the effect away.
    assert stats["evalue_estimate"] > 1.0
