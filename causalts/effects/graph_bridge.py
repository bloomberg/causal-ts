# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Convert causal-ts 3D graphs to DoWhy-compatible NetworkX DAGs."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

# "transition" is the canonical name; "full" and "summary" are accepted aliases
# for it, so callers that picked either of those keep working.
_TRANSITION_MODES = frozenset({"transition", "full", "summary"})


def _default_var_names(d: int) -> list[str]:
    return [f"X{i}" for i in range(d)]


def _lag_node(name: str, lag: int, lag_suffix: str = "_lag{k}") -> str:
    if lag == 0:
        return name
    return name + lag_suffix.replace("{k}", str(lag))


def make_lagged_df(
    data: pd.DataFrame,
    max_lag: int,
    lags: list[int] | None = None,
    var_names: list[str] | None = None,
) -> pd.DataFrame:
    """Build a lag-embedded DataFrame matching CDNOTS's internal construction.

    Columns: [X0, X1, ..., X0_lag1, X1_lag1, ..., X0_lagK, X1_lagK, ...]
    First max(lags) rows are dropped (NaN from shift).

    Parameters
    ----------
    data : pd.DataFrame
        Original (T, d) time series DataFrame.
    max_lag : int
        Maximum lag depth.
    lags : list[int] or None
        Explicit lag list (e.g. [0, 1, 3]). Defaults to range(max_lag+1).
    var_names : list[str] or None
        Column names. Defaults to data.columns.

    Returns
    -------
    pd.DataFrame
        Lag-embedded DataFrame with proper column names and no leading
        NaN rows.
    """
    if var_names is None:
        var_names = list(data.columns)

    if lags is None:
        lags = list(range(max_lag + 1))

    frames = []
    for lag in lags:
        shifted = data[var_names].shift(lag)
        if lag == 0:
            shifted.columns = var_names
        else:
            shifted.columns = [_lag_node(n, lag) for n in var_names]
        frames.append(shifted)

    lagged = pd.concat(frames, axis=1)
    n_drop = max(lags) if lags else 0
    return lagged.iloc[n_drop:].reset_index(drop=True)


def graph_to_networkx(
    graph: np.ndarray,
    var_names: list[str] | None = None,
    mode: str = "transition",
    include_c: bool = False,
    cycle_resolution: str = "error",
    horizon: int | None = None,
) -> nx.DiGraph:
    """Convert a causal-ts 3D graph array to a NetworkX DiGraph for DoWhy.

    Parameters
    ----------
    graph : np.ndarray
        Shape (d, d, max_lag+1). graph[i, j, lag]=1 means
        variable i at time t-lag causes variable j at time t.
    var_names : list[str] or None
        Variable names. Defaults to ["X0", "X1", ...].
    mode : str
        "unrolled" -- the stationary edge pattern repeated across ``horizon+1``
        slices, so lagged nodes have their own parents. Required for backdoor
        identification with a lagged treatment: without it every lagged node is
        a root, no backdoor path exists, and the adjustment set is always empty.
        "transition" -- d*(max_lag+1) nodes but edges only into current-time
        nodes. This encodes one step of the process with the history slice as
        exogenous context; the lagged nodes are roots as a modelling
        convention, not an assertion that they are independent. ``"full"`` and
        ``"summary"`` are accepted as aliases of ``"transition"``.
    horizon : int or None
        Number of past slices to unroll, for ``mode="unrolled"``. Must be at
        least ``max_lag``. For a treatment at lag L, ``L + max_lag`` is the
        shallowest horizon that represents all of the treatment's parents.
    include_c : bool
        If False (default), strips the last variable if it is the
        CDNOTS nonstationarity node C. Auto-detected when
        len(var_names) < graph.shape[0].
    cycle_resolution : str
        What to do when lag-0 edges form a cycle.
        "error" raises ValueError, "drop" silently drops
        the offending edge, "first" keeps the
        lexicographically-first direction.

    Returns
    -------
    nx.DiGraph
        NetworkX DiGraph with node/edge structure compatible with DoWhy.

    Raises
    ------
    ValueError
        If the resulting DAG contains a cycle and
        cycle_resolution="error".
    """
    d_raw = graph.shape[0]
    max_lag = graph.shape[2] - 1

    if var_names is None:
        var_names = _default_var_names(d_raw)

    # Strip C node added by CDNOTS when include_c=False
    if not include_c and len(var_names) < d_raw:
        graph = graph[: len(var_names), : len(var_names), :]
    elif not include_c and len(var_names) == d_raw:
        # Heuristic: if last var is "C", strip it
        if var_names[-1] == "C":
            var_names = var_names[:-1]
            graph = graph[: len(var_names), : len(var_names), :]

    d = len(var_names)

    dag = nx.DiGraph()

    if mode == "unrolled":
        if horizon is None:
            raise ValueError("mode='unrolled' requires an explicit horizon")
        if horizon < max_lag:
            raise ValueError(
                f"horizon={horizon} is shallower than the graph's max_lag={max_lag}"
            )
        # Nodes for every slice t, t-1, ..., t-horizon.
        for slice_ in range(horizon + 1):
            for name in var_names:
                dag.add_node(_lag_node(name, slice_))

        # Each edge is a stationary relation, so it repeats at every slice where
        # both endpoints exist. This is what gives lagged nodes their own parents.
        for i in range(d):
            for j in range(d):
                for lag in range(max_lag + 1):
                    if graph[i, j, lag] != 1:
                        continue
                    for slice_ in range(horizon - lag + 1):
                        src = _lag_node(var_names[i], slice_ + lag)
                        dst = _lag_node(var_names[j], slice_)
                        if src == dst:
                            continue
                        _add_edge_safe(dag, src, dst, cycle_resolution)

    elif mode in _TRANSITION_MODES:
        # One transition step: current-time + all lag versions as nodes, but
        # edges only into current-time nodes. Past variables are roots by
        # construction -- a modelling convention for "history is context", not
        # a claim that they are mutually independent.
        for name in var_names:
            dag.add_node(name)
        for lag in range(1, max_lag + 1):
            for name in var_names:
                dag.add_node(_lag_node(name, lag))

        for i in range(d):
            for j in range(d):
                for lag in range(max_lag + 1):
                    if graph[i, j, lag] == 1:
                        src = _lag_node(var_names[i], lag)
                        dst = var_names[j]
                        _add_edge_safe(dag, src, dst, cycle_resolution)
    else:
        allowed = "'unrolled', " + ", ".join(repr(m) for m in sorted(_TRANSITION_MODES))
        raise ValueError(f"mode must be one of {allowed}, got {mode!r}")

    if not nx.is_directed_acyclic_graph(dag):
        cycles = list(nx.simple_cycles(dag))
        raise ValueError(
            f"The resulting graph contains cycles: {cycles}. "
            "Set cycle_resolution='drop' or 'first' to handle automatically."
        )

    return dag


def _add_edge_safe(dag: nx.DiGraph, src: str, dst: str, resolution: str) -> None:
    if dag.has_edge(dst, src):
        if resolution == "error":
            raise ValueError(
                f"Cycle detected: edges {src}->{dst} and {dst}->{src} both exist. "
                "Use cycle_resolution='drop' or 'first'."
            )
        elif resolution == "drop":
            return
        elif resolution == "first":
            if src > dst:
                return  # keep the lexicographically first direction
            dag.remove_edge(dst, src)
    dag.add_edge(src, dst)


def graph_to_dowhy_model(
    graph: np.ndarray,
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    treatment_lag: int = 1,
    var_names: list[str] | None = None,
    include_c: bool = False,
    cycle_resolution: str = "error",
):
    """Build a DoWhy CausalModel from a causal-ts graph for effect estimation.

    Parameters
    ----------
    graph : np.ndarray
        Shape (d, d, max_lag+1).
    data : pd.DataFrame
        Original (T, d) time series DataFrame.
    treatment : str
        Variable name of the treatment (e.g. "X0").
    outcome : str
        Variable name of the outcome (e.g. "X2").
    treatment_lag : int
        Which lag of the treatment to use (default 1).
    var_names : list[str] or None
        Variable names. Defaults to data.columns.
    include_c : bool
        Whether to keep the C nonstationarity node.
    cycle_resolution : str
        What to do with an unoriented contemporaneous edge. Defaults to
        ``"error"`` on this path: dropping such an edge changes which backdoor
        paths exist, so it can silently create or erase confounding and change
        the estimate. ``"drop"`` or ``"first"`` opts into that.

    Returns
    -------
    tuple
        ``(dowhy.CausalModel, lagged_df)`` -- the model ready for
        identify+estimate, and the lag-embedded DataFrame used as data.
    """
    from ._compat import require_dowhy

    require_dowhy("effect estimation")
    import dowhy

    if var_names is None:
        var_names = list(data.columns)

    # Depth is derived from the query: deep enough to hold the treatment and its
    # own parents, and no deeper, since every extra lag costs a usable row. Any
    # shallower and the treatment is a root, so identify_effect sees no backdoor
    # path and returns an empty adjustment set whatever the data says.
    horizon = required_lag_depth(graph, treatment, treatment_lag, var_names)
    full_dag = graph_to_networkx(
        graph,
        var_names=var_names,
        mode="unrolled",
        include_c=include_c,
        cycle_resolution=cycle_resolution,
        horizon=horizon,
    )

    treatment_node = _lag_node(treatment, treatment_lag)
    if treatment_node not in full_dag:
        raise ValueError(
            f"Treatment node {treatment_node!r} not found in graph. "
            f"Available nodes: {sorted(full_dag.nodes)}"
        )
    if outcome not in full_dag:
        raise ValueError(
            f"Outcome node {outcome!r} not found in graph. "
            f"Available nodes: {sorted(full_dag.nodes)}"
        )

    # Prune to the ancestral subgraph of the query. d-separation between the
    # treatment and the outcome is decided entirely within An(treatment) u
    # An(outcome), so nodes outside it cannot lie on a backdoor or a causal path
    # and carrying them only costs usable rows. Note this must include the
    # interior of backdoor paths -- an ancestor of the outcome such as the
    # outcome's own previous value -- not just the treatment's parents and the
    # directed corridor, or the backdoor becomes invisible and identification
    # silently reports that nothing needs adjusting.
    keep = (
        {treatment_node, outcome}
        | nx.ancestors(full_dag, treatment_node)
        | nx.ancestors(full_dag, outcome)
    )
    # force_parent_adjustment_set() (called non-strict) can name a neighbour of an
    # unresolved contemporaneous pair that cycle_resolution oriented the other way,
    # so it never became this node's graph-ancestor. Keep those columns too, or the
    # later force raises "not columns in the data" for a pair this same resolution
    # already decided to keep ambiguous.
    keep |= {
        _lag_node(name, lag)
        for name, lag in treatment_parents(
            graph, treatment, treatment_lag, var_names, strict_contemporaneous=False
        )
    }
    dag = full_dag.subgraph(keep).copy()

    lagged_df = make_lagged_df(data, horizon, var_names=var_names)
    # Complete-case on the columns in play only, matching the convention used for
    # pairwise conditioning elsewhere in the package. Dropping on every lag column
    # would discard rows that this query does not depend on.
    present = [c for c in dag.nodes if c in lagged_df.columns]
    lagged_df = lagged_df[present].dropna().reset_index(drop=True)

    model = dowhy.CausalModel(
        data=lagged_df,
        treatment=treatment_node,
        outcome=outcome,
        graph=dag,
        proceed_when_unidentifiable=True,
    )
    return model, lagged_df


def treatment_parents(
    graph: np.ndarray,
    treatment: str,
    treatment_lag: int,
    var_names: list[str],
    strict_contemporaneous: bool = True,
) -> list[tuple[str, int]]:
    """Parents of ``treatment`` at lag ``treatment_lag``, read off the summary graph.

    In a stationary summary graph an edge ``k -> a`` at lag ``m`` means
    ``X_k(t-m)`` causes ``X_a(t)``; shifting by ``treatment_lag`` gives the parents
    of the treatment as ``X_k`` at lag ``treatment_lag + m``. No unrolled graph is
    needed to work this out, which is why this function takes the summary array
    rather than a DAG.

    The parents are a valid backdoor adjustment set for any effect of the
    treatment: no parent is a descendant of it, and every backdoor path leaves the
    treatment through an incoming edge and therefore meets a parent at a tail,
    where conditioning blocks it. That validity does not depend on how much
    history is represented, which is the reason to prefer this over asking a
    truncated graph for a minimal set.

    Parameters
    ----------
    graph : np.ndarray
        Shape (d, d, max_lag+1), ``graph[cause, effect, lag]``.
    treatment : str
        Treatment variable name.
    treatment_lag : int
        Lag at which the treatment is taken.
    var_names : list[str]
        Variable names indexing ``graph``.
    strict_contemporaneous : bool
        If True, a lag-0 edge into the treatment that is recorded in both
        directions -- an adjacency discovery could not orient -- raises rather
        than being silently treated as a parent. An unoriented neighbour may be a
        child, and conditioning on a child of the treatment biases the estimate.

    Returns
    -------
    list of (name, lag)
        Parents as (variable name, absolute lag) pairs, sorted.
    """
    if treatment not in var_names:
        raise ValueError(f"Treatment {treatment!r} is not in var_names.")
    a = var_names.index(treatment)
    max_lag = graph.shape[2] - 1

    parents: list[tuple[str, int]] = []
    for k in range(len(var_names)):
        for m in range(max_lag + 1):
            if graph[k, a, m] != 1:
                continue
            if k == a and m == 0:
                continue
            if m == 0 and graph[a, k, 0] == 1 and strict_contemporaneous:
                raise ValueError(
                    f"The contemporaneous edge between {treatment!r} and "
                    f"{var_names[k]!r} is unoriented (recorded in both "
                    "directions), so it cannot be classified as a parent or a "
                    "child. Adjusting for a child of the treatment biases the "
                    "estimate. Orient the edge, or pass "
                    "strict_contemporaneous=False to treat it as a parent."
                )
            parents.append((var_names[k], treatment_lag + m))
    return sorted(set(parents))


def parent_adjustment_set(
    graph: np.ndarray,
    treatment: str,
    treatment_lag: int,
    var_names: list[str],
    available: list[str] | None = None,
    strict_contemporaneous: bool = True,
) -> list[str]:
    """Lag-qualified column names of the treatment's parents.

    Thin wrapper over :func:`treatment_parents` that renders the (name, lag) pairs
    as dataframe column names and optionally checks they are present.
    """
    pairs = treatment_parents(
        graph, treatment, treatment_lag, var_names, strict_contemporaneous
    )
    names = [_lag_node(n, lag) for n, lag in pairs]
    if available is not None:
        missing = [n for n in names if n not in available]
        if missing:
            raise ValueError(
                f"Cannot adjust for all parents of {treatment}(t-{treatment_lag}): "
                f"{missing} are not columns in the data. Deepen the lag embedding."
            )
    return names


def required_lag_depth(
    graph: np.ndarray, treatment: str, treatment_lag: int, var_names: list[str]
) -> int:
    """Lag depth the embedded frame needs to hold the treatment and its parents.

    Partly derived from the query rather than fixed in advance: the deepest lag
    among the treatment's own parents, so the frame is no deeper than
    identification requires, and every surplus lag costs a usable row.

    Floored at the graph's own ``max_lag``, which is not negotiable. An unroll
    shallower than that cannot place an edge at the deepest lag in any slice, so
    those edges vanish silently -- and one of them may be the confounder that
    made adjustment necessary in the first place.

    This depth is sufficient for :func:`force_parent_adjustment_set`, which is
    the reason the package never asks a horizon-limited graph to expose a
    confounder several hops back: it does not need to be seen. It is not, in
    general, deep enough to make DoWhy's own unforced ``identify_effect()``
    search safe -- there is no horizon formula that is, for graphs with
    lagged feedback (see the AR self-loop and diamond-lag counterexamples in
    this function's tests). A raw-DoWhy user wants
    :func:`force_parent_adjustment_set` for the same guarantee, not a deeper
    export.
    """
    pairs = treatment_parents(
        graph, treatment, treatment_lag, var_names, strict_contemporaneous=False
    )
    return max([treatment_lag, graph.shape[2] - 1] + [lag for _, lag in pairs])


def networkx_to_graph(
    dag: nx.DiGraph,
    var_names: list[str],
    max_lag: int,
) -> np.ndarray:
    """Inverse: reconstruct (d, d, max_lag+1) array from a time-expanded DAG.

    Parameters
    ----------
    dag : nx.DiGraph
        Time-expanded NetworkX DiGraph (nodes named as
        graph_to_networkx produces).
    var_names : list[str]
        Current-time variable names.
    max_lag : int
        Maximum lag depth.

    Returns
    -------
    np.ndarray
        NumPy array of shape (d, d, max_lag+1).
    """
    d = len(var_names)
    out = np.zeros((d, d, max_lag + 1), dtype=int)
    name_to_idx = {n: i for i, n in enumerate(var_names)}

    for src, dst in dag.edges():
        # dst must be a current-time node
        if dst not in name_to_idx:
            continue
        j = name_to_idx[dst]

        # Parse lag from src
        lag = 0
        src_base = src
        for k in range(1, max_lag + 1):
            suffix = f"_lag{k}"
            if src.endswith(suffix):
                src_base = src[: -len(suffix)]
                lag = k
                break

        if src_base in name_to_idx:
            i = name_to_idx[src_base]
            out[i, j, lag] = 1

    return out


def _construct_adjustment_estimand():
    """DoWhy's adjustment-estimand constructor, under whichever name it has.

    Bound by capability, not by version. DoWhy renamed its backdoor-specific
    API to the general "adjustment" one in 0.13, so 0.11/0.12 export
    ``construct_backdoor_estimand`` and 0.13+ exports
    ``construct_adjustment_estimand``. The two take the same
    ``(treatment_names, outcome_names, common_causes)`` signature.

    A version comparison would be the wrong test: forks, conda rebuilds and
    ``0.13.0.dev`` builds all report versions that do not reliably predict
    which symbol is present. Ask for the symbol instead.
    """
    from dowhy.causal_identifier import auto_identifier

    for name in ("construct_adjustment_estimand", "construct_backdoor_estimand"):
        fn = getattr(auto_identifier, name, None)
        if fn is not None:
            return fn
    raise ImportError(
        "DoWhy exposes neither construct_adjustment_estimand (>=0.13) nor "
        "construct_backdoor_estimand (<0.13) in "
        "dowhy.causal_identifier.auto_identifier. This DoWhy build is not "
        "supported; please report it."
    )


def _adjustment_set_of(estimand) -> list[str]:
    """The adjustment set on an ``IdentifiedEstimand``, across DoWhy versions.

    ``get_adjustment_set()`` (0.13+) dispatches on ``identifier_method``;
    ``get_backdoor_variables()`` (all versions) is the backdoor-only accessor
    it superseded. Every caller here has already narrowed to the backdoor
    identifier, so the two agree.
    """
    getter = getattr(estimand, "get_adjustment_set", None)
    if getter is None:
        getter = estimand.get_backdoor_variables
    return list(getter())


def force_parent_adjustment_set(
    identified,
    graph: np.ndarray,
    treatment: str,
    treatment_lag: int,
    outcome: str,
    var_names: list[str],
    method: str,
    available: list[str] | None = None,
    strict_contemporaneous: bool = True,
) -> list[str] | None:
    """Make a DoWhy ``IdentifiedEstimand`` use the treatment's parents.

    Public so a DoWhy user driving :func:`build_identification_artifacts`'s
    output directly -- with their own estimator choice, not
    :func:`~causalts.effects.effect.estimate_effect` -- can still get this
    package's identification guarantee instead of silently falling back to
    whatever DoWhy's own minimal-set search finds. Without this call, raw
    ``model.identify_effect()`` on an exported graph returns DoWhy's own
    choice, which can be a *different* (though sometimes equally valid) set
    than the treatment's parents -- and, on a graph where a confounder sits
    more than one hop back, can be outright wrong rather than merely
    different: the horizon needed to make that confounder visible to a
    generic graph search has no general finite bound once lagged feedback is
    possible (an AR self-loop is enough), so no horizon formula can make raw
    ``identify_effect()`` safe on its own. Parent-forcing sidesteps the
    question rather than answering it: it does not need the confounder to be
    visible, so it needs no particular horizon to be correct. Call this
    between ``model.identify_effect()`` and
    ``model.estimate_effect(..., method_name=method)``, then confirm it took
    with :func:`verify_parent_adjustment_set`.

    A no-op for anything other than a backdoor method (``iv.*``,
    ``frontdoor.*``): forcing ``backdoor_variables`` has no bearing on
    estimators that never read that field.

    Parameters
    ----------
    identified : dowhy.causal_identifier.identified_estimand.IdentifiedEstimand
        The object returned by ``model.identify_effect()``. Mutated in place.
    graph : np.ndarray
        Shape ``(d, d, max_lag+1)``, the summary graph the export was built from.
    treatment, outcome : str
        Variable names.
    treatment_lag : int
        Lag at which the treatment is taken.
    var_names : list[str]
        Variable names indexing ``graph``.
    method : str
        The ``method_name`` you are about to pass to ``model.estimate_effect``.
        Only its identifier prefix (``method.split(".", 1)[0]``, matching how
        DoWhy itself derives it) is used, so any backdoor-family estimator name
        works -- ``"backdoor.linear_regression"``,
        ``"backdoor.generalized_linear_model"``,
        ``"backdoor.econml.dml.LinearDML"``, and so on.
    available : list[str] or None
        Columns present in the exported frame; parents missing from it raise.
    strict_contemporaneous : bool
        See :func:`parent_adjustment_set`.

    Returns
    -------
    list[str] or None
        The lag-qualified parent column names written, or ``None`` if ``method``
        is not a backdoor method (nothing was touched). An empty list is a
        legitimate answer -- the treatment is genuinely unconfounded -- and is
        still written unconditionally, overwriting any stale non-empty value.
    """
    identifier_key = method.split(".", 1)[0]
    if identifier_key != "backdoor":
        return None

    from ._compat import require_dowhy

    require_dowhy("forcing a parent adjustment set")
    construct_adjustment_estimand = _construct_adjustment_estimand()

    parents = parent_adjustment_set(
        graph,
        treatment,
        treatment_lag,
        var_names,
        available=available,
        strict_contemporaneous=strict_contemporaneous,
    )
    # When treatment and outcome share no path at all, identify_effect() leaves
    # backdoor_variables as None rather than an empty dict -- there was nothing
    # for DoWhy's own search to identify. set_backdoor_variables assumes a dict
    # to index into, so give it one before writing our own answer into it.
    if identified.backdoor_variables is None:
        identified.backdoor_variables = {}
    # Unconditional, not `if parents:` -- see the docstring above.
    identified.set_backdoor_variables(parents, key=identifier_key)

    # `set_backdoor_variables` only touches `backdoor_variables`. DoWhy's
    # symbolic estimand (`identified.estimands[identifier_key]`, what `str()`
    # and `.target_estimand`'s repr show) is a separate structure built at
    # `identify_effect()` time from DoWhy's *original* choice, and nothing
    # regenerates it from the variables just written. Left alone, a
    # numerically correct fit would still be reported alongside the wrong
    # confounders. Rebuild it with the same public constructor DoWhy uses.
    # Same None-vs-empty-dict gap as above when nothing was identifiable.
    if identified.estimands is None:
        identified.estimands = {}
    treatment_node = _lag_node(treatment, treatment_lag)
    identified.estimands[identifier_key] = construct_adjustment_estimand(
        [treatment_node], [outcome], parents
    )
    return parents


def verify_parent_adjustment_set(estimate, parents: list[str] | None) -> None:
    """Confirm a fitted estimate actually used the set :func:`force_parent_adjustment_set` wrote.

    Necessary because writing to an ``IdentifiedEstimand`` is not sufficient on
    its own: ``model.estimate_effect(method_name="backdoor.X")`` mutates
    ``identifier_method`` internally in a way that can silently re-resolve to a
    different key than the one just written (the exact failure this pairing
    exists to catch -- see :func:`force_parent_adjustment_set`). Call this
    after ``model.estimate_effect(...)`` with the value it returned.

    A no-op when ``parents`` is ``None`` (the method was not a backdoor
    estimator, so nothing was forced and there is nothing to verify).

    Raises
    ------
    RuntimeError
        If the fitted estimate's adjustment set does not match ``parents``.
    """
    if parents is None:
        return
    used = sorted(_adjustment_set_of(estimate.target_estimand))
    if used != sorted(parents):
        raise RuntimeError(
            "The parent adjustment set did not reach the estimator: asked for "
            f"{sorted(parents)}, but it used {used}. This usually means "
            "DoWhy's IdentifiedEstimand key layout changed."
        )
