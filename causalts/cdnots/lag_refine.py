# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Grow-then-prune refinement of the lagged links of a CDNOTS+ graph.

A PC-style search removes a lagged link ``X(t-tau) -> Y(t)`` as soon as one
conditioning set makes it look independent, and most missed lagged links go at
depth 0: the marginal correlation is weak because ``Y``'s own past and its
other parents dominate its variance. Once the graph is known, the same link
can be re-tested with the MCI conditioning set of PCMCI (Runge 2020),

    S = Pa(Y) u {Y(t-1), X(t-tau-1)} u Pa(X) shifted by tau,

where ``Pa`` holds lagged and oriented lag-0 parents. That removes the
variance the depth-0 test could not, so a real link often becomes clear.

- **Add-back**: every absent lagged cross link is tested with ``S`` from the
  oriented graph and added when ``p < alpha_add``. All tests see one graph.
- **Veto**: every lagged cross link of the grown graph is re-tested with ``S``
  from that graph and removed when ``p > alpha_veto``. It runs after add-back
  because an added parent can explain away an existing link, and some
  additions are themselves false.

A link and its time-shifted copies change together, and the whole class is
checked against background knowledge and ancestral constraints. Contemporaneous
links, self-links and C links are never touched. :func:`refine_lagged` returns
the changes; :func:`apply_lag_changes` writes them into another graph (the
caller re-orients from the refined skeleton).
"""

from __future__ import annotations

import numpy as np
from causallearn.graph.Endpoint import Endpoint

from .meek import reaches


def _is_parent(graph, u, v):
    """``u -> v`` in a causal-learn endpoint matrix."""
    return graph[u, v] == Endpoint.TAIL.value and graph[v, u] == Endpoint.ARROW.value


def _copies(x, y, tau, n_inst, num_lags):
    """Window node pairs of the link ``x(t-tau) -> y(t)`` and its shifted copies."""
    return [(x + (tau + s) * n_inst, y + s * n_inst) for s in range(num_lags - tau + 1)]


def _add(cg, pairs):
    nodes = cg.G.nodes
    added = []
    for a, b in pairs:
        if cg.G.get_edge(nodes[a], nodes[b]) is None:
            cg.G.add_directed_edge(nodes[a], nodes[b])
            added.append((a, b))
    return added


def _remove(cg, pairs):
    nodes = cg.G.nodes
    removed = []
    for a, b in pairs:
        edge = cg.G.get_edge(nodes[a], nodes[b])
        if edge is not None:
            cg.G.remove_edge(edge)
            removed.append((a, b))
    return removed


def refine_lagged(
    cg,
    ci_test,
    num_lags,
    c_start,
    alpha_add=None,
    alpha_veto=None,
    is_forbidden=lambda x, y: False,
    is_required=lambda x, y: False,
    forbidden_ancestor_pairs=(),
    required_ancestor_groups=(),
    enough_rows=lambda cols: True,
    skip=(),
    tested=None,
):
    """Add back and veto lagged cross links of ``cg`` in place.

    Parameters
    ----------
    cg : CausalGraph
        Oriented CDNOTS+ graph in the lag-embedded layout
        (``node = var + lag * n_inst``, ``num_lags + 1`` slices).
    ci_test : callable
        ``ci_test(x, y, Z) -> (p, stat)`` over the deep embedding, whose slice
        ``s`` holds lag ``s`` for ``s <= 2 * num_lags`` (contiguous lags).
    num_lags : int
        Window depth.
    c_start : int
        First C column within a slice; C columns are never tested.
    alpha_add, alpha_veto : float or None
        Thresholds; ``None`` skips that step.
    is_forbidden, is_required : callable
        ``(x, y) -> bool`` on window node indices (background knowledge). A
        link is not added if any of its copies is forbidden, and not removed
        if any is required.
    forbidden_ancestor_pairs : iterable of (int, int)
        A link is not added if, with all its copies in place, some pair
        ``(a, d)`` gets a directed path it did not have.
    required_ancestor_groups : iterable of list of (int, int)
        One list per required ancestor (all its lag copies); a link is not
        removed if that leaves a group with no directed path it had before.
    enough_rows : callable
        ``(columns) -> bool``; a test whose columns have too few complete rows
        is skipped (no change), since a CI test's insufficient-data answer is
        not evidence either way.
    skip : iterable of (cause, effect, lag)
        Links not to test or change (e.g. frozen after oscillating).
    tested : dict or None
        If given, filled with ``(cause, effect, lag) -> p`` for every link
        tested in this pass (the veto's p when both steps test a link).

    Returns
    -------
    list of dict
        One entry per change: ``cause``, ``effect`` (variable indices),
        ``lag``, ``action`` (``"added"`` / ``"removed"``), ``p`` and ``conds``
        (the conditioning set, node indices of the deep embedding).
    """
    n_inst = cg.G.graph.shape[0] // (num_lags + 1)
    forb = list(forbidden_ancestor_pairs)
    groups = [list(g) for g in required_ancestor_groups]

    def parent_map():
        """``v -> {(var, lag)}`` parents of each slice-0 variable."""
        g = cg.G.graph
        P = (g == Endpoint.TAIL.value) & (g.T == Endpoint.ARROW.value)
        np.fill_diagonal(P, False)
        return {
            v: {(int(u) % n_inst, int(u) // n_inst) for u in np.flatnonzero(P[:, v])}
            for v in range(n_inst)
        }

    # Each step tests every link before changing any, so the graph (and its
    # parent sets) is fixed while it tests.
    pa = {}

    def parents(v):
        """``(var, lag)`` parents of variable ``v`` at slice 0."""
        return pa[v]

    def mci_p(x, y, tau):
        """``(p, Z)`` of the MCI test of ``x(t-tau) -> y(t)``; p None with too
        few rows."""
        X, Y = (x, tau), (y, 0)
        S = (parents(y) - {X}) | {(y, 1), (x, tau + 1)}
        S |= {(c, lag + tau) for c, lag in parents(x)}
        Z = [c + lag * n_inst for c, lag in sorted(S - {X, Y})]
        a = x + tau * n_inst
        if not enough_rows([a, y] + Z):
            return None, Z
        p, _ = ci_test(a, y, Z)
        return p, Z

    def satisfied(group):
        g = cg.G.graph
        return any(reaches(g, a, d) for a, d in group)

    skip = set(skip)
    tested = {} if tested is None else tested
    cross = [
        (x, y, tau)
        for y in range(c_start)
        for x in range(c_start)
        if x != y
        for tau in range(1, num_lags + 1)
        if (x, y, tau) not in skip
    ]
    changes = []
    added_pairs, removed_pairs = {}, {}
    if alpha_add is not None:
        grow = []
        pa = parent_map()
        for x, y, tau in cross:
            pairs = _copies(x, y, tau, n_inst, num_lags)
            if cg.G.get_edge(cg.G.nodes[pairs[0][0]], cg.G.nodes[y]) is not None:
                continue
            if any(is_forbidden(a, b) for a, b in pairs):
                continue
            p, Z = mci_p(x, y, tau)
            if p is not None:
                tested[(x, y, tau)] = p
            if p is not None and p < alpha_add:
                grow.append((x, y, tau, p, Z))
        for x, y, tau, p, Z in grow:
            before = {pr for pr in forb if reaches(cg.G.graph, *pr)}
            added = _add(cg, _copies(x, y, tau, n_inst, num_lags))
            if any(pr not in before and reaches(cg.G.graph, *pr) for pr in forb):
                _remove(cg, added)  # the class would open a forbidden path
                continue
            added_pairs[(x, y, tau)] = set(added)
            changes.append(
                dict(cause=x, effect=y, lag=tau, action="added", p=p, conds=Z)
            )
    if alpha_veto is not None:
        prune = []
        pa = parent_map()
        for x, y, tau in cross:
            pairs = _copies(x, y, tau, n_inst, num_lags)
            if not _is_parent(cg.G.graph, *pairs[0]):
                continue
            if any(is_required(a, b) for a, b in pairs):
                continue
            p, Z = mci_p(x, y, tau)
            if p is not None:
                tested[(x, y, tau)] = p
            if p is not None and p > alpha_veto:
                prune.append((x, y, tau, p, Z))
        for x, y, tau, p, Z in prune:
            held = [g for g in groups if satisfied(g)]
            removed = _remove(cg, _copies(x, y, tau, n_inst, num_lags))
            if any(not satisfied(g) for g in held):
                _add(cg, removed)  # the last path of a required ancestor
                continue
            removed_pairs[(x, y, tau)] = set(removed)
            changes.append(
                dict(cause=x, effect=y, lag=tau, action="removed", p=p, conds=Z)
            )
    # A link added and vetoed in the same pass is no change if exactly the
    # added copies were removed; if a copy existed before, only the removal
    # is a change.
    key = lambda c: (c["cause"], c["effect"], c["lag"])  # noqa: E731
    both = set(added_pairs) & set(removed_pairs)
    undone = {k for k in both if added_pairs[k] == removed_pairs[k]}
    return [
        c
        for c in changes
        if key(c) not in undone and not (key(c) in both and c["action"] == "added")
    ]


def apply_lag_changes(cg, changes, num_lags):
    """Write :func:`refine_lagged` changes (with lag copies) into ``cg``.

    An added pair's separating sets are cleared; a removed pair gets the
    refinement's conditioning set, shifted with each copy and restricted to
    the window's nodes. Returns the changes that altered ``cg``: a change
    decided on the finished graph can be a no-op here (e.g. a link the
    ancestral constraints cut from the finished graph but not from ``cg``),
    and is then skipped.
    """
    n_nodes = cg.G.graph.shape[0]
    n_inst = n_nodes // (num_lags + 1)
    applied = []
    for ch in changes:
        pairs = _copies(ch["cause"], ch["effect"], ch["lag"], n_inst, num_lags)
        if ch["action"] == "added":
            if not _add(cg, pairs):
                continue
            for a, b in pairs:
                cg.sepset[a, b] = cg.sepset[b, a] = None
        else:
            if not _remove(cg, pairs):
                continue
            for s, (a, b) in enumerate(pairs):
                Z = tuple(
                    z + s * n_inst for z in ch["conds"] if z + s * n_inst < n_nodes
                )
                cg.sepset[a, b] = cg.sepset[b, a] = [Z]
        applied.append(ch)
    return applied


def class_state(cg, cause, effect, lag, num_lags):
    """Snapshot of a link's copies: ``[(a, b, present, sepset_ab, sepset_ba)]``."""
    n_inst = cg.G.graph.shape[0] // (num_lags + 1)
    nodes = cg.G.nodes
    return [
        (
            a,
            b,
            cg.G.get_edge(nodes[a], nodes[b]) is not None,
            cg.sepset[a, b],
            cg.sepset[b, a],
        )
        for a, b in _copies(cause, effect, lag, n_inst, num_lags)
    ]


def restore_class(cg, state):
    """Put a link's copies back as :func:`class_state` recorded them."""
    for a, b, present, sep_ab, sep_ba in state:
        if present:
            _add(cg, [(a, b)])
        else:
            _remove(cg, [(a, b)])
        cg.sepset[a, b], cg.sepset[b, a] = sep_ab, sep_ba
