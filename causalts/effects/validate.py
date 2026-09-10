# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Graph validation and falsification via DoWhy — no ground truth required."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import pandas as pd

from .graph_bridge import _lag_node, graph_to_networkx, make_lagged_df

# --------------------------------------------------------------------------
# Transition-graph validation
#
# A lag embedding is a *conditional transition graph*, not a joint DAG over the
# embedded row. What the discovered graph asserts is
#
#     p(H_t, X_t) = p(H_t) * prod_i p(X_i,t | Pa(X_i,t))
#
# with the history slice p(H_t) left **unrestricted**: nothing in a summary
# graph claims that X_0(t-1) and X_1(t-1) are independent, and in any
# autoregressive process they are not. The embedding makes every lagged node a
# root purely by construction, so any validator that reads root-ness as an
# independence claim tests a hypothesis the graph never made.
# --------------------------------------------------------------------------


def linear_ci_test(X: np.ndarray, Y: np.ndarray, Z: np.ndarray | None = None) -> float:
    """Linear-Gaussian conditional independence test, ``f(X, Y, Z) -> p_value``.

    A nested-model F-test: regress ``X`` on ``Z``, then on ``[Z, Y]``, and ask
    whether adding ``Y`` explains more variance than chance.

    Offered because DoWhy's default ``kernel_based`` is O(n^2) and impractical
    on the sample sizes time series usually have. It assumes linear-Gaussian
    dependence, so it will miss purely nonlinear violations -- pass
    ``kernel_based`` (or any other ``f(X, Y, Z) -> p`` callable) when that
    matters.

    ``X`` must be univariate; ``Y`` and ``Z`` may be multivariate, which is what
    the local Markov tests here need. Returns **NaN**, not 1.0, when the design
    leaves no residual degrees of freedom -- "not testable" and "no dependence
    found" are different answers, and callers treat NaN as untested rather than
    as a pass.
    """
    from scipy import stats

    x = np.asarray(X, dtype=float).reshape(len(X), -1)
    if x.shape[1] != 1:
        raise ValueError(
            f"linear_ci_test needs a univariate X, got {x.shape[1]} columns. "
            "Y and Z may be multivariate."
        )
    x = x[:, 0]
    y = np.asarray(Y, dtype=float).reshape(len(Y), -1)
    n = len(x)
    base = np.ones((n, 1))
    if Z is not None:
        z = np.asarray(Z, dtype=float).reshape(n, -1)
        if z.shape[1]:
            base = np.column_stack([base, z])
    full = np.column_stack([base, y])

    def _rss(design: np.ndarray) -> float:
        beta, *_ = np.linalg.lstsq(design, x, rcond=None)
        return float(np.sum((x - design @ beta) ** 2))

    df_num = full.shape[1] - base.shape[1]
    df_den = n - full.shape[1]
    if df_num < 1 or df_den < 1:
        warnings.warn(
            f"linear_ci_test has no residual degrees of freedom: {n} rows "
            f"against a {full.shape[1]}-column design. Returning NaN rather "
            "than a p-value; deepen the sample or narrow the conditioning set.",
            RuntimeWarning,
            stacklevel=2,
        )
        return float("nan")
    # Relative, not `rss <= 0`: lstsq leaves ~1e-28 residuals on an exact fit, so
    # an absolute test never fires and the F ratio is then computed from two
    # pieces of floating-point noise -- which yields a confident-looking p-value
    # driven by nothing. Scale the tolerance to the total variation in X.
    scale = float(np.sum((x - x.mean()) ** 2))
    if scale <= 0:
        return 1.0  # X is constant, hence independent of everything
    tol = 1e-12 * scale
    rss0, rss1 = _rss(base), _rss(full)
    if rss0 <= tol:
        # X is already a deterministic function of Z, so Y adds nothing.
        return 1.0
    if rss1 <= tol:
        # ...but Y explains it exactly and Z does not. That is maximal
        # dependence, not independence -- returning 1.0 here inverted the answer.
        return 0.0
    if rss0 <= rss1:
        return 1.0
    f_stat = ((rss0 - rss1) / df_num) / (rss1 / df_den)
    return float(stats.f.sf(f_stat, df_num, df_den))


def _holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni step-down adjusted p-values.

    Holm, not Benjamini-Hochberg: the graph-level verdict is "reject if any node
    fails", which is a family-wise statement. FDR control would not make that a
    level-alpha test, and Holm stays valid under arbitrary dependence between the
    per-node tests -- which these are, since they share conditioning variables.
    """
    if not p_values:
        return {}
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (node, p) in enumerate(items):
        running = max(running, (m - rank) * p)
        adjusted[node] = min(1.0, running)
    return {node: adjusted[node] for node in p_values}


# eq=False: the auto-generated __hash__ on a frozen dataclass hashes every
# field, and dict fields are unhashable, so hash() would raise. Identity
# semantics are what these carry anyway.
@dataclass(frozen=True, eq=False)
class TransitionValidationResult:
    """Outcome of :func:`validate_transition_graph`.

    Attributes
    ----------
    rejected : bool
        Family-wise verdict: at least one modelled node failed its local Markov
        test after Holm correction. Only a verdict when :attr:`tested` is True --
        an all-skipped run also reports ``False``, meaning "no evidence" rather
        than "no violation".
    rejected_nodes : tuple[str, ...]
        The nodes that failed.
    p_values, adjusted_p_values : dict[str, float]
        Raw and Holm-adjusted p-values, keyed by current-time node.
    skipped : dict[str, str]
        Nodes with no testable non-descendants, and why.
    significance_level : float
    n_tests : int
    """

    rejected: bool
    rejected_nodes: tuple[str, ...]
    p_values: dict[str, float]
    adjusted_p_values: dict[str, float]
    significance_level: float
    n_tests: int
    skipped: dict[str, str] = field(default_factory=dict)

    @property
    def tested(self) -> bool:
        """Whether any node actually produced a p-value.

        ``rejected is False`` on an all-skipped run means the test never ran,
        not that the graph passed. Branch on this first::

            if not result.tested:
                ...          # inconclusive -- do not read `rejected`
            elif result.rejected:
                ...
        """
        return bool(self.p_values)

    def to_frame(self) -> pd.DataFrame:
        """Per-node results as a DataFrame, most significant first."""
        rows = [
            {
                "node": node,
                "p_value": p,
                "adjusted_p_value": self.adjusted_p_values[node],
                "rejected": node in set(self.rejected_nodes),
            }
            for node, p in self.p_values.items()
        ]
        frame = pd.DataFrame(
            rows, columns=["node", "p_value", "adjusted_p_value", "rejected"]
        )
        return frame.sort_values("p_value").reset_index(drop=True)

    def __repr__(self) -> str:
        verdict = (
            f"REJECTED at {', '.join(self.rejected_nodes)}"
            if self.rejected
            else "not rejected"
        )
        if not self.tested:
            verdict = "UNTESTED (every test skipped)"
        skipped = f", {len(self.skipped)} skipped" if self.skipped else ""
        return (
            f"<TransitionValidationResult {verdict}; {self.n_tests} node tests"
            f"{skipped}, alpha={self.significance_level}>"
        )


def validate_transition_graph(
    graph: np.ndarray,
    data: pd.DataFrame,
    var_names: list[str] | None = None,
    significance_level: float = 0.05,
    include_c: bool = False,
    conditional_independence_test=None,
    cycle_resolution: str = "drop",
) -> TransitionValidationResult:
    """Test the local Markov condition at every modelled current-time node.

    For each variable :math:`X_j` the test is

    .. math:: X_j(t) \\perp \\mathrm{NonDesc}(X_j(t)) \\setminus
              \\mathrm{Pa}(X_j(t)) \\mid \\mathrm{Pa}(X_j(t))

    as a single multivariate conditional-independence test, with a Holm
    correction across variables. Lagged nodes are **not** tested: the transition
    factorisation leaves the joint history distribution unrestricted, so their
    mutual dependence is not evidence against the graph. Testing them is what
    made :func:`falsify_graph` unusable on autoregressive data.

    This answers only "are the asserted parent sets sufficient?". For "is
    ``max_lag`` deep enough?" use :func:`history_sufficiency`, which is a
    genuinely different hypothesis.

    Parameters
    ----------
    graph : np.ndarray
        Shape ``(d, d, max_lag+1)`` from CDNOTS/Cedar/GRACE.
    data : pd.DataFrame
        Original ``(T, d)`` time series.
    var_names : list[str] or None
        Variable names. Defaults to ``data.columns``.
    significance_level : float
        Family-wise level for the Holm-corrected verdict.
    include_c : bool
        Whether to keep the CDNOTS C node.
    conditional_independence_test : callable or None
        ``f(X, Y, Z) -> p_value`` over 2-D arrays. Defaults to
        :func:`linear_ci_test`. Pass DoWhy's
        ``gcm.independence_test.kernel_based`` to catch nonlinear violations,
        at O(n^2) cost -- but see the note under :func:`history_sufficiency`
        about its behaviour on wide conditioning sets.
    cycle_resolution : str
        Passed to :func:`~causalts.effects.graph_bridge.graph_to_networkx`.

    Returns
    -------
    TransitionValidationResult
    """
    if conditional_independence_test is None:
        conditional_independence_test = linear_ci_test

    if var_names is None:
        var_names = list(data.columns)

    max_lag = graph.shape[2] - 1
    dag = graph_to_networkx(
        graph,
        var_names=var_names,
        mode="transition",
        include_c=include_c,
        cycle_resolution=cycle_resolution,
    )
    lagged_df = make_lagged_df(data, max_lag, var_names=var_names)
    lagged_df = lagged_df[[c for c in lagged_df.columns if c in dag.nodes]].dropna()

    p_values: dict[str, float] = {}
    skipped: dict[str, str] = {}
    for node in [n for n in var_names if n in dag.nodes]:
        parents = sorted(p for p in dag.predecessors(node) if p in lagged_df.columns)
        others = sorted(
            n
            for n in set(dag.nodes) - nx.descendants(dag, node) - {node} - set(parents)
            if n in lagged_df.columns
        )
        if not others:
            skipped[node] = "no non-descendants outside the parent set"
            continue
        x = lagged_df[[node]].to_numpy(dtype=float)
        y = lagged_df[others].to_numpy(dtype=float)
        z = lagged_df[parents].to_numpy(dtype=float) if parents else None
        p = float(conditional_independence_test(x, y, z))
        # A test that could not be computed is not a test that passed. Recording
        # NaN as a p-value would let it sail through Holm and read as a clean
        # node; it belongs in `skipped`, where it is visible.
        if np.isnan(p):
            skipped[node] = "the conditional independence test returned NaN"
            continue
        p_values[node] = p

    adjusted = _holm(p_values)
    rejected_nodes = tuple(
        node for node, p in adjusted.items() if p <= significance_level
    )
    return TransitionValidationResult(
        rejected=bool(rejected_nodes),
        rejected_nodes=rejected_nodes,
        p_values=p_values,
        adjusted_p_values=adjusted,
        significance_level=significance_level,
        n_tests=len(p_values),
        skipped=skipped,
    )


@dataclass(frozen=True, eq=False)
class HistorySufficiencyResult:
    """Outcome of :func:`history_sufficiency`.

    ``rejected`` is only a verdict when :attr:`tested` is True -- an all-skipped
    run also reports ``False``, meaning "no evidence" rather than "deep enough".
    """

    rejected: bool
    rejected_variables: tuple[str, ...]
    p_values: dict[str, float]
    adjusted_p_values: dict[str, float]
    significance_level: float
    max_lag: int
    extra_lags: int
    skipped: dict[str, str] = field(default_factory=dict)

    @property
    def tested(self) -> bool:
        """Whether any variable actually produced a p-value.

        ``rejected is False`` on an all-skipped run means the test never ran,
        not that ``max_lag`` is deep enough. Check this before acting on
        ``rejected`` -- otherwise ``if result.rejected: deepen()`` silently
        declines to deepen an embedding that was never assessed.
        """
        return bool(self.p_values)

    def to_frame(self) -> pd.DataFrame:
        rows = [
            {
                "variable": name,
                "p_value": p,
                "adjusted_p_value": self.adjusted_p_values[name],
                "rejected": name in set(self.rejected_variables),
            }
            for name, p in self.p_values.items()
        ]
        frame = pd.DataFrame(
            rows, columns=["variable", "p_value", "adjusted_p_value", "rejected"]
        )
        return frame.sort_values("p_value").reset_index(drop=True)

    def __repr__(self) -> str:
        verdict = (
            f"max_lag={self.max_lag} INSUFFICIENT for "
            f"{', '.join(self.rejected_variables)}"
            if self.rejected
            else f"max_lag={self.max_lag} sufficient"
        )
        if not self.tested:
            verdict = f"max_lag={self.max_lag} UNTESTED (every test skipped)"
        skipped = f", {len(self.skipped)} skipped" if self.skipped else ""
        return (
            f"<HistorySufficiencyResult {verdict}; probed {self.extra_lags} "
            f"further lags{skipped}, alpha={self.significance_level}>"
        )


def history_sufficiency(
    data: pd.DataFrame,
    max_lag: int,
    extra_lags: int = 2,
    var_names: list[str] | None = None,
    significance_level: float = 0.05,
    conditional_independence_test=None,
) -> HistorySufficiencyResult:
    """Test whether history older than ``max_lag`` still carries information.

    For each variable the test is

    .. math:: X_j(t) \\perp H_{t-K-1:t-K-q} \\mid H_{t-1:t-K}

    where :math:`H` is the **saturated** history -- every variable at every lag
    in the window, not just the graph's chosen parents. Conditioning on the full
    window is what separates this diagnostic from
    :func:`validate_transition_graph`: an edge missing *inside* the window would
    otherwise leave structure in the residual and be misread as "the window is
    too short".

    Note this deliberately does not use the mutual dependence of the lagged
    nodes. Those are dependent in any stationary autoregressive process,
    including one whose order is exactly ``max_lag``, so they carry no
    information about depth.

    Parameters
    ----------
    data : pd.DataFrame
        Original ``(T, d)`` time series.
    max_lag : int
        The depth being checked -- typically the discovered graph's ``max_lag``.
    extra_lags : int
        How many further lags to probe.
    var_names : list[str] or None
    significance_level : float
        Family-wise level for the Holm-corrected verdict.
    conditional_independence_test : callable or None
        ``f(X, Y, Z) -> p_value``. Defaults to :func:`linear_ci_test`.

        Not ``kernel_based``: the saturated window makes the conditioning set
        ``d * max_lag`` columns wide, and on ``ex3`` (d=11, max_lag=3, T=500 --
        so 33 conditioning columns) the kernel test rejected all 11 variables
        on data generated by a VAR(3), where the correct answer is that the
        window is deep enough. The linear test gets that case right. Pass the
        kernel test explicitly if you need nonlinear sensitivity and your
        window is narrow.

    Returns
    -------
    HistorySufficiencyResult
    """
    if extra_lags < 1:
        raise ValueError(f"extra_lags must be at least 1, got {extra_lags}")
    if max_lag < 1:
        raise ValueError(f"max_lag must be at least 1, got {max_lag}")

    if conditional_independence_test is None:
        conditional_independence_test = linear_ci_test

    if var_names is None:
        var_names = list(data.columns)

    total = max_lag + extra_lags
    lagged_df = make_lagged_df(data, total, var_names=var_names).dropna()
    window = [_lag_node(n, k) for k in range(1, max_lag + 1) for n in var_names]
    beyond = [_lag_node(n, k) for k in range(max_lag + 1, total + 1) for n in var_names]

    z = lagged_df[window].to_numpy(dtype=float)
    y = lagged_df[beyond].to_numpy(dtype=float)
    p_values: dict[str, float] = {}
    skipped: dict[str, str] = {}
    for name in var_names:
        # The saturated window makes the design d*(max_lag+extra_lags)+1 columns
        # wide, so a wide, short panel can leave no residual degrees of freedom.
        # A test that could not run must not read as "the window is deep enough".
        p = float(
            conditional_independence_test(lagged_df[[name]].to_numpy(dtype=float), y, z)
        )
        if np.isnan(p):
            skipped[name] = "the conditional independence test returned NaN"
            continue
        p_values[name] = p

    adjusted = _holm(p_values)
    rejected_variables = tuple(
        name for name, p in adjusted.items() if p <= significance_level
    )
    return HistorySufficiencyResult(
        rejected=bool(rejected_variables),
        rejected_variables=rejected_variables,
        p_values=p_values,
        adjusted_p_values=adjusted,
        significance_level=significance_level,
        max_lag=max_lag,
        extra_lags=extra_lags,
        skipped=skipped,
    )


REFUTERS = (
    "placebo_treatment_refuter",
    "random_common_cause",
    "data_subset_refuter",
    "bootstrap_refuter",
    "dummy_outcome_refuter",
)


def _refute(
    graph,
    data,
    treatment,
    outcome,
    treatment_lag,
    method_name,
    estimation_method,
    var_names,
    include_c,
    cycle_resolution,
    kwargs,
):
    """Estimate, then hand DoWhy the same model and estimand to attack."""
    from .effect import _identify_and_estimate

    model, identified, estimate, _ = _identify_and_estimate(
        graph,
        data,
        treatment=treatment,
        outcome=outcome,
        treatment_lag=treatment_lag,
        method=estimation_method,
        var_names=var_names,
        confidence_intervals=False,
        include_c=include_c,
        cycle_resolution=cycle_resolution,
    )
    if method_name == "placebo_treatment_refuter":
        # DoWhy's default placebo for a *float* treatment is
        # `randn(n) * DEFAULT_STD_DEV_OF_NORMAL + DEFAULT_MEAN_OF_NORMAL`, and
        # both constants are 0 -- so the "placebo" is a constant column and the
        # estimate is exactly zero whatever the estimator does. That makes the
        # refutation vacuous, and every treatment here is a float. Permuting the
        # treatment gives a placebo with the right marginal instead.
        kwargs.setdefault("placebo_type", "permute")

    result = model.refute_estimate(
        identified, estimate, method_name=method_name, **kwargs
    )
    # dummy_outcome_refuter returns a *list*, one refutation per dummy outcome
    # it constructs, where every other refuter returns a single object. Silently
    # keeping the first would hide the rest of the answer.
    if isinstance(result, list):
        if not result:
            raise RuntimeError(f"{method_name} returned no refutation results.")
        if len(result) > 1:
            raise NotImplementedError(
                f"{method_name} produced {len(result)} refutations, and this "
                "wrapper reports one. Call model.refute_estimate() directly, or "
                "use to_dowhy_identification() and drive DoWhy yourself."
            )
        result = result[0]
    return estimate, result


def _refutation_verdict(
    method: str,
    reported_reference: float,
    p_value: float | None,
    significance_level: float,
) -> bool | None:
    """Whether the estimate survived, or ``None`` when there is no verdict.

    DoWhy sets ``is_statistically_significant = p <= significance_level`` and
    the null is "the reference estimate belongs to the refuter's distribution",
    so ``p > alpha`` is a pass for every refuter. Equality counts as
    significant, hence a strict ``>``.

    Tri-state on purpose. A non-finite p-value means the significance test
    reached no verdict, and ``None`` says so. Falling back to an effect-size
    rule -- accept if the displacement is under some fraction of the original
    estimate -- would invent a verdict from a quantity unrelated to the
    refuter's null: against an original effect of 100, a 10% rule accepts a
    placebo estimate of 9.
    """
    del method, reported_reference  # kept for signature stability
    if p_value is None:
        return None
    return bool(p_value > significance_level)


def refute_effect(
    graph: np.ndarray,
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    treatment_lag: int = 1,
    method: str = "placebo_treatment_refuter",
    estimation_method: str = "backdoor.linear_regression",
    var_names: list[str] | None = None,
    include_c: bool = False,
    cycle_resolution: str = "error",
    significance_level: float = 0.05,
    **kwargs,
) -> dict:
    """Stress-test an estimated effect with one of DoWhy's refuters.

    :func:`~causalts.effects.effect.estimate_effect` returns a number that is
    correct *if the graph is correct*. This is how you probe that proviso. Each
    refuter perturbs the data or the estimand in a way that should not change
    the answer, and reports whether it did.

    ``placebo_treatment_refuter`` (the default) replaces the treatment with a
    permutation of itself; the effect should collapse to zero. The permutation
    is ours -- DoWhy's own default for a float treatment is a column of zeros,
    which makes the refutation vacuous. Override with ``placebo_type=``. ``random_common_cause`` adds an
    independent covariate; the effect should not move.
    ``data_subset_refuter`` and ``bootstrap_refuter`` resample;
    ``dummy_outcome_refuter`` replaces the outcome.

    It refutes the same estimand that ``estimate_effect`` reports -- the
    treatment's parents, not DoWhy's minimal backdoor set -- so the number under
    attack is the number you were given.

    Note what this cannot do. A refuter tests the *estimator*, given the graph.
    It cannot tell you the graph is wrong; for that use
    :func:`validate_transition_graph`, or :func:`sensitivity_analysis` for the
    specific case of an omitted confounder.

    Parameters
    ----------
    graph : np.ndarray
        Shape ``(d, d, max_lag+1)``.
    data : pd.DataFrame
        Original ``(T, d)`` time series.
    treatment, outcome : str
        Variable names.
    treatment_lag : int
        Lag at which the treatment is taken.
    method : str
        A name from :data:`REFUTERS`.
    estimation_method : str
        Passed through to ``estimate_effect`` as ``method``.
    var_names, include_c, cycle_resolution
        As for ``estimate_effect``.
    significance_level : float
        A refutation passes when DoWhy's p-value exceeds this.
    **kwargs
        Forwarded to the refuter, e.g. ``num_simulations``.

    Returns
    -------
    dict
        ``refuter``, ``estimated_effect``, ``new_effect``, ``reference_effect``
        (what the new effect should be if the estimate holds up), ``p_value``,
        ``passed``, ``summary`` and the raw ``details``.

        ``passed`` is ``None`` when DoWhy produced no usable p-value, which is
        a third outcome and not a failure. It is our reading of DoWhy's
        numbers, which DoWhy deliberately leaves to the caller; ``details``
        holds the raw refutation if you disagree with it.
    """
    from ._compat import require_dowhy

    require_dowhy("effect refutation")

    if method not in REFUTERS:
        raise ValueError(
            f"Unknown refuter {method!r}. Choose one of {list(REFUTERS)}. For "
            "unobserved-confounder sensitivity use sensitivity_analysis()."
        )

    estimate, result = _refute(
        graph,
        data,
        treatment,
        outcome,
        treatment_lag,
        method,
        estimation_method,
        var_names,
        include_c,
        cycle_resolution,
        kwargs,
    )

    raw = getattr(result, "refutation_result", None)
    try:
        p_value = float(raw["p_value"])
    except (KeyError, TypeError, ValueError):
        p_value = None
    if p_value is not None and not np.isfinite(p_value):
        p_value = None
    old_effect = float(estimate.value)
    new_effect = float(result.new_effect)
    # The significance reference differs per refuter and cannot be read off one
    # field. Placebo builds a *separate* zero-valued estimate for the test while
    # reporting the original in `estimated_effect`; dummy-outcome reports the
    # known injected effect (zero unless the caller supplies one) and tests
    # against exactly that; the resampling refuters test against the original.
    if method == "placebo_treatment_refuter":
        reference = 0.0
    else:
        reference = float(result.estimated_effect)
    passed = _refutation_verdict(method, reference, p_value, significance_level)

    return {
        "refuter": method,
        "estimated_effect": old_effect,
        "new_effect": new_effect,
        "reference_effect": reference,
        "p_value": p_value,
        # Not bool(passed): None is the documented third outcome ("no verdict"),
        # and bool(None) would report an uncomputable refutation as a failed one.
        "passed": passed,
        "summary": str(result),
        "details": result,
    }


def sensitivity_analysis(
    graph: np.ndarray,
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    treatment_lag: int = 1,
    simulation_method: str = "e-value",
    estimation_method: str = "backdoor.linear_regression",
    var_names: list[str] | None = None,
    include_c: bool = False,
    cycle_resolution: str = "error",
    **kwargs,
) -> dict:
    """How strong would an omitted confounder have to be to overturn the effect?

    Wraps DoWhy's ``add_unobserved_common_cause`` refuter. Every effect this
    package estimates rests on the discovered graph containing all the common
    causes; this quantifies what it would take for that to be false in a way
    that matters.

    The default ``"e-value"`` (VanderWeele & Ding) is the only one of DoWhy's
    simulation methods that runs unaided: it reports how strongly a confounder
    would have to associate with both treatment and outcome, on the risk-ratio
    scale, to explain the effect away. ``"linear-partial-R2"`` (Cinelli-Hazlett)
    is available but **requires** ``benchmark_common_causes`` and
    ``effect_fraction_on_treatment``, and DoWhy 0.14 is particular about their
    form. ``"non-parametric-partial-R2"`` and ``"direct-simulation"`` are passed
    through untested.

    This is a different question from :func:`refute_effect`, which perturbs the
    data and asks whether the estimator is stable, and from
    :func:`validate_transition_graph`, which tests the graph against the data.
    A confounder hidden from *both* the graph and the data is exactly what this
    one addresses.

    Parameters
    ----------
    graph, data, treatment, outcome, treatment_lag, var_names, include_c,
    cycle_resolution
        As for :func:`refute_effect`.
    simulation_method : str
        Passed to DoWhy as ``simulation_method``.
    estimation_method : str
        Passed through to ``estimate_effect`` as ``method``.
    **kwargs
        Forwarded to the refuter, e.g. ``benchmark_common_causes``,
        ``effect_fraction_on_treatment``. ``plot_estimate`` defaults to
        ``False`` here -- DoWhy's plot calls ``plt.show()``, which blocks under
        an interactive backend.

    Returns
    -------
    dict
        ``estimated_effect``, ``simulation_method``, ``summary`` and the raw
        ``details``, whose shape depends on ``simulation_method``.
    """
    from ._compat import require_dowhy

    require_dowhy("sensitivity analysis")

    # DoWhy's e-value / sensitivity plots call plt.show(), which blocks
    # indefinitely under any interactive matplotlib backend and turns a library
    # call into a hang. Default it off; a caller who wants the figure can
    # still ask for it.
    kwargs = {"simulation_method": simulation_method, "plot_estimate": False, **kwargs}
    estimate, result = _refute(
        graph,
        data,
        treatment,
        outcome,
        treatment_lag,
        "add_unobserved_common_cause",
        estimation_method,
        var_names,
        include_c,
        cycle_resolution,
        kwargs,
    )
    return {
        "estimated_effect": float(estimate.value),
        "simulation_method": simulation_method,
        "summary": str(result),
        "details": result,
    }


def falsify_graph(
    graph: np.ndarray,
    data: pd.DataFrame,
    var_names: list[str] | None = None,
    significance_level: float = 0.05,
    n_permutations: int | None = None,
    suggestions: bool = False,
    include_c: bool = False,
) -> dict:
    """Test whether the discovered DAG is consistent with the data.

    .. deprecated:: 0.27.0
        Use :func:`validate_transition_graph` instead. This function treats the
        lag embedding as a **joint** DAG over the embedded row, which is the
        wrong model: DoWhy's permutation test adds an unconditional
        independence test for every root, every lagged node is a root by
        construction, and in any autoregressive process those nodes are
        dependent. The resulting violations are artefacts of the embedding and
        swamp the signal -- on a d=4 VAR(1) with the ground-truth graph they
        take 3/22 violations to 21/44, against 9/22 -> 27/44 when a latent
        confounder is actually present. The node-permutation null is also not
        meaningful here: permuting names across the current/history boundary
        produces graphs that are not transition graphs at all.

    Uses permutation-based comparison: tests if the DAG violates fewer
    local Markov conditions than random node permutations of the same graph.

    No ground truth required.

    Args:
        graph: Shape (d, d, max_lag+1) from CDNOTS/CEDAR/GRACE.
        data: Original (T, d) time series DataFrame.
        var_names: Variable names. Defaults to data.columns.
        significance_level: Threshold for falsification.
        n_permutations: Number of random permutations (None = auto).
        suggestions: If True, include fix suggestions for violations.
        include_c: Whether to include the CDNOTS C node.

    Returns:
        Dict with keys:
            falsifiable: bool — is the graph "characteristic enough" to test?
            falsified: bool — does the graph fail the test?
            summary: str — human-readable summary.
            details: object — raw DoWhy EvaluationResult.
    """
    from ._compat import require_gcm

    require_gcm("graph falsification")
    from dowhy.gcm.falsify import falsify_graph as _falsify

    warnings.warn(
        "falsify_graph() applies joint-DAG semantics to a lag embedding, where "
        "every lagged node is a root by construction and DoWhy's unconditional "
        "root tests fail on any autoregressive series regardless of whether the "
        "graph is correct. Use validate_transition_graph() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if var_names is None:
        var_names = list(data.columns)

    max_lag = graph.shape[2] - 1
    dag = graph_to_networkx(
        graph,
        var_names=var_names,
        mode="summary",
        include_c=include_c,
        cycle_resolution="drop",
    )
    lagged_df = make_lagged_df(data, max_lag, var_names=var_names)
    cols = [c for c in lagged_df.columns if c in dag.nodes]
    lagged_df = lagged_df[cols]

    result = _falsify(
        dag,
        lagged_df,
        significance_level=significance_level,
        n_permutations=n_permutations,
        suggestions=suggestions,
        show_progress_bar=False,
        plot_histogram=False,
    )

    return {
        "falsifiable": bool(result.falsifiable),
        "falsified": bool(result.falsified),
        "summary": str(result),
        "details": result,
    }


def refute_structure(
    graph: np.ndarray,
    data: pd.DataFrame,
    var_names: list[str] | None = None,
    significance_level: float = 0.05,
    include_c: bool = False,
) -> pd.DataFrame:
    """Test structural assumptions of the discovered graph.

    For each node, tests: (1) edge dependencies — parents are truly
    dependent on the node, (2) local Markov conditions — node is
    independent of non-descendants given parents.

    No ground truth required.

    Args:
        graph: Shape (d, d, max_lag+1).
        data: Original (T, d) time series DataFrame.
        var_names: Variable names.
        significance_level: p-value threshold.
        include_c: Whether to include the CDNOTS C node.

    Returns:
        DataFrame with columns [node, variable, lag, test_type, rejected,
        p_value, adjusted_p_value].

        ``p_value`` is DoWhy's raw per-test p-value; ``adjusted_p_value`` is its
        FDR-corrected counterpart, ``NaN`` where DoWhy reports no correction for
        that test. Compare against ``significance_level`` using the adjusted
        column -- correcting the raw one again would double-correct. Matches the
        column convention of :meth:`TransitionValidationResult.to_frame` and
        :meth:`HistorySufficiencyResult.to_frame`. ``rejected`` is DoWhy's own
        verdict, not a re-thresholding of either column.
    """
    from ._compat import require_gcm

    require_gcm("structural refutation")
    import dowhy.gcm as gcm

    if var_names is None:
        var_names = list(data.columns)

    max_lag = graph.shape[2] - 1
    dag = graph_to_networkx(
        graph,
        var_names=var_names,
        mode="summary",
        include_c=include_c,
        cycle_resolution="drop",
    )
    lagged_df = make_lagged_df(data, max_lag, var_names=var_names)
    cols = [c for c in lagged_df.columns if c in dag.nodes]
    lagged_df = lagged_df[cols]

    from .root_cause import _parse_node

    rejection, details = gcm.refute_causal_structure(
        dag,
        lagged_df,
        significance_level=significance_level,
    )

    rows = []
    for node, tests in details.items():
        variable, lag = _parse_node(node)
        for test_type, test_results in tests.items():
            if isinstance(test_results, dict):
                if "p_value" in test_results:
                    # Direct test result: {p_value, fdr_adjusted_p_value, success}
                    rows.append(
                        {
                            "node": node,
                            "variable": variable,
                            "lag": lag,
                            "test_type": test_type,
                            "rejected": not test_results.get("success", True),
                            "p_value": test_results.get("p_value", float("nan")),
                            "adjusted_p_value": test_results.get(
                                "fdr_adjusted_p_value", float("nan")
                            ),
                        }
                    )
                else:
                    # Nested per-edge results: {edge_name: {p_value, ...}, ...}
                    for sub_key, sub_val in test_results.items():
                        if isinstance(sub_val, dict) and "p_value" in sub_val:
                            rows.append(
                                {
                                    "node": node,
                                    "variable": variable,
                                    "lag": lag,
                                    "test_type": f"{test_type}:{sub_key}",
                                    "rejected": not sub_val.get("success", True),
                                    "p_value": sub_val.get("p_value", float("nan")),
                                    "adjusted_p_value": sub_val.get(
                                        "fdr_adjusted_p_value", float("nan")
                                    ),
                                }
                            )

    if not rows:
        rows.append(
            {
                "node": "overall",
                "variable": "",
                "lag": 0,
                "test_type": "overall",
                "rejected": rejection.value == 1,
                "p_value": float("nan"),
                "adjusted_p_value": float("nan"),
            }
        )

    result = pd.DataFrame(rows)
    return result


def evaluate_model(
    graph: np.ndarray,
    data: pd.DataFrame,
    var_names: list[str] | None = None,
    mechanism_type: str = "auto",
    include_c: bool = False,
    evaluate_overall_kl_divergence: bool = False,
    evaluate_causal_structure: bool = False,
) -> dict:
    """Evaluate the fitted mechanisms: fit quality and invertibility.

    Fits an SCM (cached) and evaluates it on multiple criteria.

    No ground truth required.

    Args:
        graph: Shape (d, d, max_lag+1).
        data: Original (T, d) time series DataFrame.
        var_names: Variable names.
        mechanism_type: 'auto', 'linear', or 'gp'.
        include_c: Whether to include the CDNOTS C node.

    Returns:
        Dict with keys: mechanism_scores (per-node), kl_divergence,
        invertibility, overall_summary (str).
    """
    from ._compat import require_gcm

    require_gcm("model evaluation")
    import dowhy.gcm as gcm

    from .scm import fit_scm

    if var_names is None:
        var_names = list(data.columns)

    scm, dag, lagged_df = fit_scm(
        graph,
        data,
        var_names=var_names,
        mechanism_type=mechanism_type,
        include_c=include_c,
    )

    if evaluate_overall_kl_divergence or evaluate_causal_structure:
        warnings.warn(
            "evaluate_overall_kl_divergence and evaluate_causal_structure are "
            "unsound on a lag embedding. The first calls gcm.draw_samples, "
            "which generates each history node from its own independent "
            "mechanism and so compares the data against a distribution that "
            "denies the serial dependence the data has. The second calls "
            "dowhy's falsify_graph, which tests every root unconditionally and "
            "therefore rejects any autoregressive series regardless of the "
            "graph. Use validate_transition_graph() for the structure question.",
            UserWarning,
            stacklevel=2,
        )

    eval_result = gcm.evaluate_causal_model(
        scm,
        lagged_df,
        evaluate_causal_mechanisms=True,
        evaluate_invertibility_assumptions=True,
        evaluate_overall_kl_divergence=evaluate_overall_kl_divergence,
        evaluate_causal_structure=evaluate_causal_structure,
    )

    return {
        "summary": str(eval_result),
        "details": eval_result,
    }
