# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Hand DoWhy the artifacts and get out of the way.

Every other module here wraps a DoWhy call. This one exports the two things a
DoWhy-fluent user actually needs -- a graph and a dataframe whose columns are
that graph's nodes -- so they can drive ``dowhy.CausalModel`` or
``dowhy.gcm`` directly.

There is deliberately no ``to_dowhy()`` with a default flavour. Which graph is
correct depends on the task, and picking wrong is silent:

``identification``
    The stationary edge pattern unrolled across enough slices that the lagged
    treatment has its own parents. Anything shallower makes every lagged node a
    root, so ``identify_effect`` finds no backdoor path, returns an empty
    adjustment set, and an unadjusted regression gets reported as a causal
    effect.

``transition``
    One step of the process: current-time nodes with the history slice as
    exogenous context. Right for fitting and evaluating the current-time
    mechanisms *given observed history*.

The transition export is **not** a complete joint SCM. DoWhy's GCM gives every
root node its own independent mechanism and samples roots first when
generating, so a plain ``gcm.draw_samples`` on it would treat the history slice
as a product of independent marginals. The transition factorisation leaves that
joint unrestricted precisely because it is not a product. Use it for
conditional mechanism fitting and evaluation, not unconditional generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import pandas as pd

from .graph_bridge import (
    _lag_node,
    graph_to_networkx,
    make_lagged_df,
    required_lag_depth,
    treatment_parents,
)


# eq=False: a frozen dataclass auto-generates __eq__/__hash__ over every field,
# and a DataFrame field makes __eq__ raise "truth value is ambiguous" and
# __hash__ raise "unhashable". Identity semantics are the right ones here.
@dataclass(frozen=True, eq=False)
class DoWhyArtifacts:
    """A graph and the lag-embedded frame that goes with it.

    The two are returned together, and never separately, because they have to
    agree: ``set(graph.nodes) == set(data.columns)`` is checked on construction.
    A user who fetched them from two calls could silently pair a graph with a
    frame embedded to a different depth.

    Attributes
    ----------
    graph : nx.DiGraph
        Nodes named ``X`` for current time and ``X_lagK`` for lag ``K``.
    data : pd.DataFrame
        Lag-embedded, complete-cased, index reset.
    semantics : str
        ``"identification"`` or ``"transition"`` -- see the module docstring.
    horizon : int
        Lag depth of the embedding.
    query : dict or None
        For ``"identification"``, the effect query the horizon was derived
        from: lag-qualified ``treatment`` and ``outcome`` node names and the
        ``treatment_lag``. ``None`` for ``"transition"``.
    """

    graph: nx.DiGraph
    data: pd.DataFrame
    semantics: str
    horizon: int
    query: dict | None = field(default=None)

    def __post_init__(self) -> None:
        nodes = {str(n) for n in self.graph.nodes}
        columns = {str(c) for c in self.data.columns}
        if nodes != columns:
            raise ValueError(
                "Graph nodes and dataframe columns must match exactly. "
                f"Only in graph: {sorted(nodes - columns)}; "
                f"only in data: {sorted(columns - nodes)}."
            )

    def __repr__(self) -> str:
        query = ""
        if self.query is not None:
            query = f", {self.query['treatment']} -> {self.query['outcome']}"
        return (
            f"<DoWhyArtifacts semantics={self.semantics!r}, horizon="
            f"{self.horizon}, {self.graph.number_of_nodes()} nodes, "
            f"{len(self.data)} rows{query}>"
        )


def build_transition_artifacts(
    graph: np.ndarray,
    data: pd.DataFrame,
    var_names: list[str] | None = None,
    include_c: bool = False,
    cycle_resolution: str = "error",
) -> DoWhyArtifacts:
    """Export the one-step transition graph and its lag-embedded frame.

    Suitable for fitting and evaluating the current-time mechanisms given
    observed history -- ``gcm.StructuralCausalModel(artifacts.graph)`` followed
    by ``gcm.fit(scm, artifacts.data)``. See the module docstring for why it is
    not a complete joint SCM.

    Parameters
    ----------
    graph : np.ndarray
        Shape ``(d, d, max_lag+1)``.
    data : pd.DataFrame
        Original ``(T, d)`` time series.
    var_names : list[str] or None
        Defaults to ``data.columns``.
    include_c : bool
        Whether to keep the CDNOTS C node.
    cycle_resolution : str
        What to do with an unoriented contemporaneous edge; see
        :func:`~causalts.effects.graph_bridge.graph_to_networkx`.

    Returns
    -------
    DoWhyArtifacts
    """
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
    frame = make_lagged_df(data, max_lag, var_names=var_names)
    frame = frame[[c for c in frame.columns if c in dag.nodes]]
    frame = frame.dropna().reset_index(drop=True)
    dag = dag.subgraph([n for n in dag.nodes if n in frame.columns]).copy()
    return DoWhyArtifacts(
        graph=dag, data=frame, semantics="transition", horizon=max_lag
    )


def build_identification_artifacts(
    graph: np.ndarray,
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    treatment_lag: int = 1,
    var_names: list[str] | None = None,
    include_c: bool = False,
    cycle_resolution: str = "error",
) -> DoWhyArtifacts:
    """Export an unrolled graph and frame for one effect query.

    The unroll depth is derived from the query rather than asked for, because
    an explicit ``horizon`` is an internal correctness parameter that most
    callers cannot choose: too shallow and the treatment loses parents, which
    silently empties the adjustment set. That makes the exported graph
    query-specific -- it represents enough temporal structure for *this* effect
    query under the treatment-parent adjustment contract, not the whole
    process. For a query-independent unroll use
    :func:`~causalts.effects.graph_bridge.graph_to_networkx` with
    ``mode="unrolled"`` and an explicit horizon.

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
    var_names : list[str] or None
        Defaults to ``data.columns``.
    include_c : bool
        Whether to keep the CDNOTS C node.
    cycle_resolution : str
        Defaults to ``"error"``: dropping an unoriented contemporaneous edge
        changes which backdoor paths exist, so it can move the estimate.

    Returns
    -------
    DoWhyArtifacts
    """
    if var_names is None:
        var_names = list(data.columns)

    horizon = required_lag_depth(graph, treatment, treatment_lag, var_names)
    dag = graph_to_networkx(
        graph,
        var_names=var_names,
        mode="unrolled",
        include_c=include_c,
        cycle_resolution=cycle_resolution,
        horizon=horizon,
    )

    treatment_node = _lag_node(treatment, treatment_lag)
    for node in (treatment_node, outcome):
        if node not in dag:
            raise ValueError(
                f"Node {node!r} not found in the unrolled graph. "
                f"Available nodes: {sorted(dag.nodes)}"
            )

    # Only An(treatment) u An(outcome) can bear on d-separation between the
    # two, and every node kept costs a usable row. This must include the
    # interior of backdoor paths -- the outcome's own lag, say -- not just the
    # treatment's parents, or the backdoor becomes invisible and identification
    # reports that nothing needs adjusting.
    keep = (
        {treatment_node, outcome}
        | nx.ancestors(dag, treatment_node)
        | nx.ancestors(dag, outcome)
    )
    # A neighbour of an unresolved contemporaneous pair that cycle_resolution
    # oriented away from the treatment never becomes its graph-ancestor, but a
    # non-strict force_parent_adjustment_set() call downstream can still name it.
    # Keep those columns too, or that call raises "not columns in the data" for
    # a pair this same resolution already decided to keep ambiguous.
    keep |= {
        _lag_node(name, lag)
        for name, lag in treatment_parents(
            graph, treatment, treatment_lag, var_names, strict_contemporaneous=False
        )
    }
    dag = dag.subgraph(keep).copy()

    frame = make_lagged_df(data, horizon, var_names=var_names)
    frame = frame[[c for c in frame.columns if c in dag.nodes]]
    frame = frame.dropna().reset_index(drop=True)
    dag = dag.subgraph([n for n in dag.nodes if n in frame.columns]).copy()
    return DoWhyArtifacts(
        graph=dag,
        data=frame,
        semantics="identification",
        horizon=horizon,
        query={
            "treatment": treatment_node,
            "outcome": outcome,
            "treatment_lag": treatment_lag,
        },
    )
