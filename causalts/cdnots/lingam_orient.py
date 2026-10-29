# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Orient contemporaneous edges that conditional independence leaves undirected.

A CI-based search (PCMCI+, CDNOTS+) cannot orient a lag-0 edge outside any
v-structure: both directions are Markov equivalent. With non-Gaussian noise
they are not observationally equivalent (LiNGAM; Shimizu et al., 2006), and
the direction can be read off the innovations of the two endpoints.

For each connected component of unoriented (``o-o``) or conflicting (``x-x``)
lag-0 edges, the innovations of its nodes -- residuals on their lagged and
directed lag-0 parents in the graph -- are ordered by DirectLiNGAM (Shimizu et
al., 2011) with the pairwise likelihood-ratio measure of Hyvarinen & Smith
(2013). An edge is oriented only when

- the direction agrees in at least ``min_agreement`` of bootstrap re-orderings,
- at least one endpoint's innovations reject normality (D'Agostino K^2,
  ``p < normality_alpha``), since LiNGAM needs a non-Gaussian endpoint,
- and it creates no unshielded collider the CI step classified as a
  non-collider.

Under a linear Gaussian model the innovations are Gaussian, so the normality
gate keeps the step from acting on noise; on held-out simulations it changed F1
by at most 0.0003 on average (0.0008 in any single cell) on Gaussian data,
while orienting most identifiable edges otherwise.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import normaltest

from .meek import rebuild_dpath

_K1, _K2, _GAMMA = 79.047, 7.4129, 0.37457


def _residual(xi, xj):
    return xi - np.cov(xi, xj, bias=True)[0, 1] / np.var(xj) * xj


def _entropies(U):
    """Maximum-entropy approximation of differential entropy (Hyvarinen 1998) of
    every column of standardized ``U``."""
    return (
        (1 + np.log(2 * np.pi)) / 2
        - _K1 * (np.mean(np.log(np.cosh(U)), axis=0) - _GAMMA) ** 2
        - _K2 * np.mean(U * np.exp(-(U**2) / 2), axis=0) ** 2
    )


def direct_lingam_order(X):
    """Causal order of the columns of ``X`` (n, k), most exogenous first.

    Each round scores all remaining pairs with array operations, one row of
    the pairwise likelihood ratios (Hyvarinen & Smith 2013) at a time, so
    memory stays O(n k): row i holds column i regressed on every column j.
    """
    X = np.array(X, dtype=float, copy=True)
    remaining = list(range(X.shape[1]))
    order = []
    while len(remaining) > 1:
        Xs = (X[:, remaining] - X[:, remaining].mean(0)) / X[:, remaining].std(0)
        C = Xs.T @ Xs / len(Xs)  # correlations
        H = _entropies(Xs)
        Hr = np.empty((len(remaining), len(remaining)))
        with np.errstate(invalid="ignore", divide="ignore"):  # i on itself
            for i in range(len(remaining)):
                Ri = Xs[:, [i]] - C[i][None, :] * Xs  # i's residual on each j
                Hr[i] = _entropies((Ri - Ri.mean(0)) / Ri.std(0))
        diff = np.nan_to_num(H[None, :] + Hr - H[:, None] - Hr.T, nan=0.0)
        np.fill_diagonal(diff, 0.0)
        scores = (np.minimum(0.0, diff) ** 2).sum(axis=1)
        root = remaining[int(np.argmin(scores))]
        order.append(root)
        remaining.remove(root)
        for i in remaining:
            X[:, i] = _residual(X[:, i], X[:, root])
    return order + remaining


def _parents(graph, v, n_inst, c_start):
    """Every directed parent of node ``v``, plus every adjacent C column.

    C has no parents, so any C adjacency means C -> v. Defensive: the
    skeleton normally stores C edges directed already, but should one be left
    undirected, omitting it would let a trend leak into the innovations and
    make Gaussian data look non-Gaussian.
    """
    out = []
    for u in range(graph.shape[0]):
        if u == v:
            continue
        is_c = u % n_inst >= c_start
        if (is_c and graph[u, v] != 0) or (graph[u, v] == -1 and graph[v, u] == 1):
            out.append(u)
    return out


def _new_collider(cg, graph, cause, effect, skip=lambda z: False):
    """Whether ``cause -> effect`` makes an unshielded collider the CI step ruled out.

    ``graph`` is the live graph, so arrowheads this step already wrote count:
    every unshielded lag-0 triple existed when the collider step classified
    it. A ``z -> effect`` with ``z`` not adjacent to ``cause`` blocks the
    orientation unless the collider step declared ``z - effect - cause`` a
    collider (``collider_triples``; for an ``x-x`` edge, the vote that put
    the arrowhead there) or left it ambiguous, which leaves it to the data.
    Parents ``z`` with ``skip(z)`` are not checked.
    """
    allowed = set(getattr(cg, "ambiguous_triples", ())) | set(
        getattr(cg, "collider_triples", ())
    )
    for z in range(graph.shape[0]):
        if (
            z in (cause, effect)
            or skip(z)
            or not (graph[z, effect] == -1 and graph[effect, z] == 1)
        ):
            continue
        if graph[z, cause] != 0 or graph[cause, z] != 0:
            continue
        if (z, effect, cause) not in allowed and (cause, effect, z) not in allowed:
            return True
    return False


def lingam_orient_lag0(
    cg,
    data,
    num_lags,
    c_start,
    can_orient,
    orient,
    exclude_vars=(),
    min_agreement=0.9,
    normality_alpha=0.01,
    n_boot=100,
    seed=0,
):
    """Orient undirected lag-0 edges of ``cg`` in place; return the decisions.

    Parameters
    ----------
    cg : CausalGraph
        CDNOTS graph in the lag-embedded node layout (``node = var + slot * n``).
    data : ndarray
        The lag-embedded data matrix whose columns are those nodes.
    num_lags : int
        Number of lag slices beyond lag 0.
    c_start : int
        First C column within a slice; nodes at or above it are never oriented.
    can_orient, orient : callable
        ``can_orient(cg, x, y)`` and ``orient(graph, x, y)`` from the caller,
        so cycle, background-knowledge and forbidden-ancestor checks are shared.
    exclude_vars : iterable of int
        Variables (lag-0 indices) to leave alone, e.g. discrete columns.

    Returns
    -------
    list of dict
        One entry per undirected lag-0 pair between non-excluded, non-C
        variables (pairs touching those get no record): ``cause``, ``effect``,
        ``agreement``, ``normality_p`` and whether it was ``oriented``. A
        component with fewer than 30 complete rows, or an endpoint its parents
        explain exactly (or a constant one), is left alone; its pairs carry
        ``agreement`` and ``normality_p`` of ``None`` (``cause`` / ``effect``
        then only name the pair).
    """
    graph = cg.G.graph
    n_inst = graph.shape[0] // (num_lags + 1)
    skip = set(exclude_vars)
    pairs = []
    for x in range(c_start):
        for y in range(x + 1, c_start):
            if x in skip or y in skip:
                continue
            a, b = graph[x, y], graph[y, x]
            if (a == -1 and b == -1) or (a == 1 and b == 1):
                pairs.append((x, y))
    if not pairs:
        return []

    parent = list(range(c_start))

    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    for x, y in pairs:
        parent[find(x)] = find(y)

    # Meek's R1 conflicts (left o-o): R1 implied both directions, so the
    # parents that drove it block either orientation. For those edges the
    # check skips parents already directed into the effect when this step
    # starts, and still applies to arrowheads written later.
    r1_conflicts = set(getattr(cg, "conflict_edges", None) or ())
    graph0 = graph.copy()
    # A graph from before collider_triples was recorded (e.g. a cached
    # CdnotsPlusPrefix) cannot tell an x-x edge's own collider vote from a
    # non-collider; those edges then skip the check, as they used to.
    no_record = getattr(cg, "collider_triples", None) is None
    warned = False
    decisions = []
    for root in sorted({find(x) for x, _ in pairs}):
        edges = [(x, y) for x, y in pairs if find(x) == root]
        nodes = sorted({v for e in edges for v in e})
        cols = []
        for v in nodes:
            Z = data[:, _parents(graph, v, n_inst, c_start)]
            cols.append((data[:, v], Z))
        ok = np.ones(len(data), dtype=bool)
        for y_, Z in cols:
            ok &= ~np.isnan(y_)
            if Z.shape[1]:
                ok &= ~np.isnan(Z).any(1)
        n = int(ok.sum())
        E = np.empty((n, len(nodes)))
        # An undirected edge to an excluded (e.g. discrete) variable leaves it
        # in neither the ordering nor the regression, so its term would leak
        # into an innovation and could fake non-Gaussianity.
        degenerate = n < 30 or any(
            graph[v, u] != 0 and graph[v, u] == graph[u, v] for v in nodes for u in skip
        )
        if not degenerate:
            for k, (y_, Z) in enumerate(cols):
                y_ = y_[ok]
                scale = np.abs(y_).max() if y_.size else 0.0
                if not np.isfinite(scale) or y_.std() <= 1e-12 * (scale + 1e-300):
                    degenerate = True  # a (numerically) constant endpoint
                    break
                M = (
                    np.column_stack([np.ones(n), Z[ok]])
                    if Z.shape[1]
                    else np.ones((n, 1))
                )
                E[:, k] = y_ - M @ np.linalg.lstsq(M, y_, rcond=None)[0]
                # An endpoint its parents explain exactly has no innovation to
                # order: standardising it gives NaN and a bogus direction.
                if not np.isfinite(E[:, k]).all() or E[:, k].std() <= 1e-8 * (
                    y_.std() + 1e-300
                ):
                    degenerate = True
        if not degenerate:
            Es = (E - E.mean(0)) / E.std(0)
            sv = np.linalg.svd(Es, compute_uv=False)
            if not np.isfinite(sv).all() or sv[-1] <= 1e-8 * sv[0]:
                degenerate = True  # collinear innovations: no order to read
        if degenerate:  # abstain, but keep one record per pair
            decisions += [
                dict(
                    cause=int(x),
                    effect=int(y),
                    agreement=None,
                    normality_p=None,
                    oriented=False,
                )
                for x, y in edges
            ]
            continue
        pos = {v: k for k, v in enumerate(nodes)}
        normal_p = {v: float(normaltest(E[:, pos[v]]).pvalue) for v in nodes}

        def directions(Ex):
            rank = {nodes[k]: r for r, k in enumerate(direct_lingam_order(Ex))}
            return np.array([rank[x] < rank[y] for x, y in edges])

        if not any(min(normal_p[x], normal_p[y]) < normality_alpha for x, y in edges):
            # Nothing here can pass the normality gate: skip the bootstrap.
            decisions += [
                dict(
                    cause=int(x),
                    effect=int(y),
                    agreement=None,
                    normality_p=min(normal_p[x], normal_p[y]),
                    oriented=False,
                )
                for x, y in edges
            ]
            continue
        rng = np.random.default_rng([seed, root])  # per component
        full = directions(E)
        with np.errstate(invalid="ignore", divide="ignore"):
            agree = np.mean(
                [directions(E[rng.integers(0, n, n)]) == full for _ in range(n_boot)],
                axis=0,
            )
        # Most confident first, so the collider check keeps the stronger call
        # (and column order does not decide); records stay in pair order.
        component = []
        for (x, y), x_first, ag in sorted(zip(edges, full, agree), key=lambda t: -t[2]):
            cause, effect = (x, y) if x_first else (y, x)
            px, py = normal_p[x], normal_p[y]
            p_norm = min(px, py) if np.isfinite(px) and np.isfinite(py) else np.nan
            done = False
            if ag >= min_agreement and p_norm < normality_alpha:
                copies = [
                    (cause + ll * n_inst, effect + ll * n_inst)
                    for ll in range(num_lags + 1)
                ]
                # x-x (here and in its lag copies): clear to o-o so the checks
                # apply, and put back whatever stays unoriented.
                cleared = []
                for a, b in copies:
                    if graph[a, b] == 1 and graph[b, a] == 1:
                        graph[a, b] = graph[b, a] = -1
                        cleared.append((a, b))

                def ruled_out(a, b):
                    nonlocal warned
                    if no_record and (a, b) in cleared:
                        if not warned:
                            warnings.warn(
                                "graph has no collider_triples record (built "
                                "before it was added?); conflicting lag-0 "
                                "edges skip the collider check. Re-run the "
                                "collider step to check them.",
                                RuntimeWarning,
                                stacklevel=4,
                            )
                            warned = True
                        return False
                    if (a, b) in r1_conflicts:
                        return _new_collider(
                            cg,
                            graph,
                            a,
                            b,
                            skip=lambda z: graph0[z, b] == -1 and graph0[b, z] == 1,
                        )
                    return _new_collider(cg, graph, a, b)

                if can_orient(cg, cause, effect) and not ruled_out(cause, effect):
                    # Copies follow the lag-0 decision (stationarity), as in
                    # Meek; ambiguous_triples records lag-0 middles only.
                    for a, b in copies:
                        if can_orient(cg, a, b):
                            orient(graph, a, b)
                    done = True
                for a, b in cleared:
                    if graph[a, b] == -1 and graph[b, a] == -1:
                        graph[a, b] = graph[b, a] = 1
                conflicts = getattr(cg, "conflict_edges", None)
                if done and conflicts:
                    for a, b in copies:
                        if graph[a, b] == -1 and graph[b, a] == 1:
                            conflicts.discard((a, b))
                            conflicts.discard((b, a))
            component.append(
                (
                    edges.index((x, y)),
                    dict(
                        cause=int(cause),
                        effect=int(effect),
                        agreement=float(ag),
                        normality_p=p_norm,
                        oriented=done,
                    ),
                )
            )
        decisions += [d for _, d in sorted(component, key=lambda t: t[0])]
    if any(d["oriented"] for d in decisions):
        rebuild_dpath(cg)  # endpoints were written directly
    return decisions
