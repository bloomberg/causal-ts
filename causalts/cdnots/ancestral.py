# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Ancestral (path-level) background knowledge for causal discovery.

Provides :class:`AncestralKnowledge`, a unified knowledge container that
supports both direct edge constraints (path length = 1, enforced during
skeleton discovery) and ancestral path constraints (any length, enforced
during orientation).
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from itertools import combinations

import numpy as np

# Cut-set search for forbidden ancestors the greedy cleanup leaves violated.
_CUT_SEARCH_MAX_SIZE = 4
_CUT_SEARCH_BUDGET = 5000  # cuts tried per round (with lag copies, then single)


def count_c_cols(include_C, num_c_cols=None):
    """Number of C columns per lag slice (``num_c_cols`` wins if given)."""
    if not include_C:
        return 0
    return 1 if num_c_cols is None else int(num_c_cols)


class AncestralKnowledge:
    """Unified causal background knowledge: edge constraints + path constraints.

    Edge constraints (path length = 1) are enforced during skeleton discovery
    (same as the legacy ``background_knowledge`` parameter).  Ancestral
    constraints (any path length) are enforced during the orientation phase.

    Parameters are accumulated via the ``add_*`` methods, which return
    ``self`` for chaining.

    Examples
    --------
    >>> ak = AncestralKnowledge()
    >>> ak.add_forbidden_edge("X0", "X4", cause_lag=1)       # no direct X0→X4
    >>> ak.add_required_ancestor("X0", "X4")                  # some path must exist
    >>> ak.add_forbidden_ancestor("X4", "X0")                 # X4 never causes X0
    """

    def __init__(self):
        self._forbidden_edges: list[tuple] = []
        self._required_edges: list[tuple] = []
        self._required_ancestors: list[tuple[str, str]] = []
        self._forbidden_ancestors: list[tuple[str, str]] = []

    # ------------------------------------------------------------------
    # Edge constraints (→ skeleton phase via causal-learn BackgroundKnowledge)
    # ------------------------------------------------------------------

    def add_forbidden_edge(
        self,
        cause: str,
        effect: str,
        cause_lag: int = 0,
        effect_lag: int = 0,
    ) -> "AncestralKnowledge":
        """Forbid the direct edge cause(t-cause_lag) → effect(t-effect_lag)."""
        self._forbidden_edges.append((cause, effect, cause_lag, effect_lag))
        return self

    def add_required_edge(
        self,
        cause: str,
        effect: str,
        cause_lag: int = 0,
        effect_lag: int = 0,
    ) -> "AncestralKnowledge":
        """Require the direct edge cause(t-cause_lag) → effect(t-effect_lag)."""
        self._required_edges.append((cause, effect, cause_lag, effect_lag))
        return self

    # ------------------------------------------------------------------
    # Ancestral / path constraints (→ orientation phase)
    # ------------------------------------------------------------------

    def add_required_ancestor(
        self,
        ancestor: str,
        descendant: str,
    ) -> "AncestralKnowledge":
        """Require a directed path ancestor → … → descendant (any lags).

        During orientation, lag copies of *ancestor* at lag >= 1 are ranked by
        AR(1)-residualized |Pearson r| and tried in that order.  Lag-0
        (contemporaneous) is never used — contemporaneous correlations in time
        series are often due to common causes, not direct causation.  Use
        ``add_required_edge(ancestor, descendant, cause_lag=0)`` to explicitly
        require a contemporaneous edge.

        If no lagged semi-directed path exists the constraint is recorded as a
        violation — no edges are force-added.
        """
        self._required_ancestors.append((ancestor, descendant))
        return self

    def add_forbidden_ancestor(
        self,
        ancestor: str,
        descendant: str,
    ) -> "AncestralKnowledge":
        """Prevent any directed path ancestor → … → descendant.

        Enforced proactively during Meek orientation: any edge orientation
        that would complete a forbidden path is blocked.
        """
        self._forbidden_ancestors.append((ancestor, descendant))
        return self

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def has_edge_constraints(self) -> bool:
        return bool(self._forbidden_edges or self._required_edges)

    @property
    def has_path_constraints(self) -> bool:
        return bool(self._required_ancestors or self._forbidden_ancestors)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def to_background_knowledge(
        self, var_names, num_lags, include_C=False, num_c_cols=None
    ):
        """Convert edge constraints to a causal-learn BackgroundKnowledge object.

        Used internally by run_cdnots / run_cdnots_plus for the skeleton phase.
        """
        from .phase3_utils import make_background_knowledge

        return make_background_knowledge(
            var_names=var_names,
            num_lags=num_lags,
            include_C=include_C,
            num_c_cols=num_c_cols,
            forbidden=self._forbidden_edges or None,
            required=self._required_edges or None,
        )

    def _node_idx(
        self, var_name: str, lag: int, var_names: list, n_per_lag: int
    ) -> int:
        return var_names.index(var_name) + lag * n_per_lag

    def resolve_forbidden_ancestor_pairs(
        self,
        var_names: list,
        num_lags: int,
        include_C: bool = False,
        num_c_cols: int | None = None,
    ) -> list[tuple[int, int]]:
        """Return (a_idx, d_idx) pairs for all lag copies of each forbidden ancestor.

        Each ancestor variable is expanded to all lag copies (lag 0..num_lags).
        The descendant is always at lag 0 (current time).
        """
        if not self._forbidden_ancestors:
            return []
        n_per_lag = len(var_names) + count_c_cols(include_C, num_c_cols)
        pairs = []
        for ancestor, desc in self._forbidden_ancestors:
            if ancestor not in var_names or desc not in var_names:
                continue
            d_idx = self._node_idx(desc, 0, var_names, n_per_lag)
            for lag in range(num_lags + 1):
                a_idx = self._node_idx(ancestor, lag, var_names, n_per_lag)
                if a_idx != d_idx:
                    pairs.append((a_idx, d_idx))
        return pairs

    def resolve_required_ancestor_pairs(
        self,
        var_names: list,
        num_lags: int,
        include_C: bool = False,
        num_c_cols: int | None = None,
    ) -> list[tuple[int, int]]:
        """Return (a_idx, d_idx) pairs for all lag copies of each required ancestor."""
        if not self._required_ancestors:
            return []
        n_per_lag = len(var_names) + count_c_cols(include_C, num_c_cols)
        pairs = []
        for ancestor, desc in self._required_ancestors:
            if ancestor not in var_names or desc not in var_names:
                continue
            d_idx = self._node_idx(desc, 0, var_names, n_per_lag)
            for lag in range(num_lags + 1):
                a_idx = self._node_idx(ancestor, lag, var_names, n_per_lag)
                if a_idx != d_idx:
                    pairs.append((a_idx, d_idx))
        return pairs

    def validate(self) -> list[str]:
        """Return a list of conflict descriptions (empty = no conflicts).

        Detects:

        * ``required_ancestor(A, D)`` + ``forbidden_ancestor(A, D)``
        * ``required_edge(A, D, lag)`` + ``forbidden_edge(A, D, lag)``
        * ``required_edge(A, D, ...)`` + ``forbidden_ancestor(A, D)``
          — the required edge would create the forbidden path
        """
        issues = []
        req_anc = {(a, d) for a, d in self._required_ancestors}
        forb_anc = {(a, d) for a, d in self._forbidden_ancestors}
        req_edges = {(c, e, cl, el) for c, e, cl, el in self._required_edges}
        forb_edges = {(c, e, cl, el) for c, e, cl, el in self._forbidden_edges}

        for pair in req_anc & forb_anc:
            issues.append(
                f"required_ancestor{pair} conflicts with forbidden_ancestor{pair}"
            )
        for tup in req_edges & forb_edges:
            issues.append(
                f"required_edge({tup[0]}→{tup[1]} lag={tup[2]}) conflicts with "
                f"forbidden_edge at same lag"
            )
        for c, e, cl, el in self._required_edges:
            if (c, e) in forb_anc:
                issues.append(
                    f"required_edge({c}→{e}) conflicts with forbidden_ancestor({c},{e}): "
                    "the required edge creates the forbidden path"
                )
        return issues

    def __repr__(self) -> str:
        parts = []
        if self._forbidden_edges:
            parts.append(f"forbidden_edges={len(self._forbidden_edges)}")
        if self._required_edges:
            parts.append(f"required_edges={len(self._required_edges)}")
        if self._required_ancestors:
            parts.append(f"required_ancestors={self._required_ancestors}")
        if self._forbidden_ancestors:
            parts.append(f"forbidden_ancestors={self._forbidden_ancestors}")
        return f"AncestralKnowledge({', '.join(parts)})"


# ---------------------------------------------------------------------------
# Semi-directed BFS
# ---------------------------------------------------------------------------


def _semi_directed_bfs(
    cg, start: int, end: int, exclude_nodes: set | None = None
) -> list[int] | None:
    """Find shortest semi-directed path from *start* to *end* in *cg*.

    Follows directed edges only in the forward direction and undirected edges
    in both directions.  Returns the list of node indices (inclusive) or
    ``None`` if no path exists.

    Parameters
    ----------
    exclude_nodes : set of int, optional
        Node indices that must not appear as INTERMEDIATE nodes (they can
        still be the start or end).  Used to block autoregressive stepping
        stones through past copies of the descendant variable.
    """
    n = cg.G.num_vars
    if start == end:
        return [start]

    visited = {start}
    queue = deque([[start]])

    while queue:
        path = queue.popleft()
        current = path[-1]
        for nxt in range(n):
            if nxt in visited:
                continue
            # Block excluded intermediate nodes (but always allow the endpoint)
            if nxt != end and exclude_nodes and nxt in exclude_nodes:
                continue
            # Can we traverse current → nxt?
            if cg.is_undirected(current, nxt) or cg.is_fully_directed(current, nxt):
                if nxt == end:
                    return path + [nxt]
                visited.add(nxt)
                queue.append(path + [nxt])
    return None


# ---------------------------------------------------------------------------
# Required-ancestor orientation pass
# ---------------------------------------------------------------------------


def _directed_path(graph, a, d):
    """Shortest directed path ``a -> ... -> d`` (node list) or None.

    ``i -> j`` is causal-learn's ``graph[i, j] == -1 and graph[j, i] == 1``.
    """
    prev = {a: None}
    queue = deque([a])
    while queue:
        u = queue.popleft()
        if u == d:
            path = [d]
            while prev[path[-1]] is not None:
                path.append(prev[path[-1]])
            return path[::-1]
        for v in np.flatnonzero((graph[u] == -1) & (graph[:, u] == 1)):
            v = int(v)
            if v not in prev:
                prev[v] = u
                queue.append(v)
    return None


def apply_ancestral_constraints(
    cg,
    knowledge: AncestralKnowledge,
    var_names: list,
    num_lags: int,
    include_C: bool = False,
    background_knowledge=None,
    df=None,
    verbose: bool = False,
    num_c_cols: int | None = None,
) -> tuple:
    """Enforce ancestral constraints on an oriented CausalGraph.

    Called after the full CDNOTS orientation pipeline (uc_sepset + Meek +
    Phase 3).  Returns ``(cg, violations)`` where *violations* is a dict
    with ``"required"`` and ``"forbidden"`` lists of (ancestor, descendant)
    name pairs that could not be satisfied.

    Required ancestors
    ------------------
    For each ``required_ancestor(A, D)`` the lag copy of A with the highest
    |Pearson r| to D is used as the BFS start point.  If a semi-directed path
    from that lag copy exists, undirected edges along it are oriented toward D
    and Meek is re-run.  Remaining lag copies are tried as fallback.  If no
    path exists from any lag copy a violation is recorded.

    Forbidden ancestors
    -------------------
    Proactive enforcement happens in Meek (``forbidden_ancestor_pairs``).
    This function checks for any remaining violations post-hoc and reports them.
    """
    import warnings

    from causallearn.graph.Edge import Edge
    from causallearn.graph.Endpoint import Endpoint

    from .meek import meek, reaches

    # Warn on conflicting constraints before doing anything
    conflicts = knowledge.validate()
    if conflicts:
        warnings.warn(
            "AncestralKnowledge has conflicting constraints:\n  "
            + "\n  ".join(conflicts),
            UserWarning,
            stacklevel=3,
        )

    violations = {"required": [], "forbidden": []}

    if not knowledge.has_path_constraints:
        return cg, violations

    n = cg.G.num_vars
    n_per_lag = len(var_names) + count_c_cols(include_C, num_c_cols)
    forb_pairs = knowledge.resolve_forbidden_ancestor_pairs(
        var_names, num_lags, include_C, num_c_cols=num_c_cols
    )

    def _var_label(idx: int) -> str:
        lag = idx // n_per_lag
        vi = idx % n_per_lag
        name = var_names[vi] if vi < len(var_names) else f"C{vi - len(var_names)}"
        return f"{name}(t-{lag})" if lag else name

    def _best_lag_order(anc_name: str, desc_name: str) -> list[int]:
        """Return lag search order for required_ancestor BFS.

        Lagged copies (lag >= 1) are ranked by |Pearson r| with the
        AR(1)-residualized descendant — this isolates the ancestor's direct
        causal signal after removing the descendant's own autoregressive
        component.  Lag-0 is appended last as a fallback.
        """
        if num_lags == 0:
            return [0]

        lagged = list(range(1, num_lags + 1))

        if df is not None and anc_name in df.columns and desc_name in df.columns:
            d_raw = df[desc_name].values
            a_base = df[anc_name].values

            # AR(1) residualize the descendant: d_col = desc(t) - β·desc(t-1)
            d_col = d_raw.copy().astype(float)
            yp = np.roll(d_raw, 1).astype(float)
            yp[0] = np.nan
            mask_ar = ~np.isnan(d_raw) & ~np.isnan(yp)
            if mask_ar.sum() >= 10:
                A = np.column_stack([np.ones(mask_ar.sum()), yp[mask_ar]])
                coef, *_ = np.linalg.lstsq(A, d_raw[mask_ar], rcond=None)
                resid = np.full(len(d_raw), np.nan)
                resid[mask_ar] = d_raw[mask_ar] - A @ coef
                d_col = resid

            scores = []
            for lag in lagged:
                a_col = np.roll(a_base, lag).astype(float)
                a_col[:lag] = np.nan
                mask = ~np.isnan(a_col) & ~np.isnan(d_col)
                r = (
                    abs(np.corrcoef(a_col[mask], d_col[mask])[0, 1])
                    if mask.sum() >= 5
                    else -1.0
                )
                scores.append((lag, r if np.isfinite(r) else -1.0))
            scores.sort(key=lambda x: x[1], reverse=True)
            lagged = [lag for lag, _ in scores]

        return lagged  # lag-0 excluded: use add_required_edge(cause_lag=0) for contemporaneous

    def _orient_edge(xi: int, xj: int) -> None:
        # Required-ancestor knowledge is a hard constraint: it may orient an
        # edge Meek left undirected as a conflict, which then is one no more.
        conflicts = getattr(cg, "conflict_edges", None)
        if conflicts:
            conflicts.discard((xi, xj))
            conflicts.discard((xj, xi))
        edge1 = cg.G.get_edge(cg.G.nodes[xi], cg.G.nodes[xj])
        if edge1 is not None:
            if not reaches(cg.G.graph, xj, xi):
                cg.G.remove_edge(edge1)
        cg.G.add_edge(
            Edge(cg.G.nodes[xi], cg.G.nodes[xj], Endpoint.TAIL, Endpoint.ARROW)
        )

    oriented_any = False

    # --- Required ancestor: Pearson-guided lag selection + BFS orientation ---
    for anc_name, desc_name in knowledge._required_ancestors:
        if anc_name not in var_names or desc_name not in var_names:
            continue

        d_idx = var_names.index(desc_name)  # lag-0 descendant node
        if d_idx >= n:
            continue

        # Already satisfied by any lag copy?
        already_satisfied = any(
            reaches(cg.G.graph, var_names.index(anc_name) + lag * n_per_lag, d_idx)
            for lag in range(num_lags + 1)
            if var_names.index(anc_name) + lag * n_per_lag < n
        )
        if already_satisfied:
            continue

        # Exclude other lag copies of the descendant as BFS intermediate nodes
        # to prevent autoregressive stepping-stone paths like X2(t-k)→X5(t-k)→…→X5(t)
        desc_var_idx = var_names.index(desc_name)
        exclude_desc_lags = {
            desc_var_idx + lag * n_per_lag
            for lag in range(1, num_lags + 1)
            if desc_var_idx + lag * n_per_lag < n
        }

        # Try lags in AR-residualized Pearson-ranked order; use first with a valid path
        lag_order = _best_lag_order(anc_name, desc_name)
        path_found = False
        for lag in lag_order:
            a_idx = var_names.index(anc_name) + lag * n_per_lag
            if a_idx >= n:
                continue
            path = _semi_directed_bfs(cg, a_idx, d_idx, exclude_nodes=exclude_desc_lags)
            if path is not None:
                for k in range(len(path) - 1):
                    xi, xj = path[k], path[k + 1]
                    if cg.is_undirected(xi, xj):
                        _orient_edge(xi, xj)
                        oriented_any = True
                        if verbose:
                            print(
                                f"[AncestralKnowledge] Oriented {_var_label(xi)} → "
                                f"{_var_label(xj)} (required_ancestor, lag={lag})"
                            )
                path_found = True
                break

        if not path_found:
            best_k = lag_order[0] if lag_order else 1
            a_idx_best = var_names.index(anc_name) + best_k * n_per_lag

            # --- Step 3: Bridge search —
            # Look for a direct skeleton neighbor of A(t-best_k) that already
            # has a semi-directed path to D(t). Orient the A→neighbor edge and
            # let the existing path handle the rest. Bounded to direct neighbors only.
            bridge_found = False
            if a_idx_best < n:
                neighbors_of_a = [
                    nb
                    for nb in range(n)
                    if nb != a_idx_best
                    and nb not in exclude_desc_lags
                    and (
                        cg.is_undirected(a_idx_best, nb)
                        or cg.is_fully_directed(a_idx_best, nb)
                    )
                ]
                reachable_bridges = [
                    nb
                    for nb in neighbors_of_a
                    if _semi_directed_bfs(
                        cg, nb, d_idx, exclude_nodes=exclude_desc_lags
                    )
                    is not None
                ]
                if reachable_bridges:
                    # Pick the bridge with highest |Pearson r| with ancestor variable
                    def _bridge_score(nb_idx: int) -> float:
                        vi = nb_idx % n_per_lag
                        nb_var = var_names[vi] if vi < len(var_names) else None
                        if nb_var is None or df is None or nb_var not in df.columns:
                            return -1.0
                        nb_lag = nb_idx // n_per_lag
                        nb_col = df[nb_var].shift(nb_lag).values.astype(float)
                        anc_col = df[anc_name].shift(best_k).values.astype(float)
                        mask = ~np.isnan(nb_col) & ~np.isnan(anc_col)
                        if mask.sum() < 5:
                            return -1.0
                        r = abs(np.corrcoef(nb_col[mask], anc_col[mask])[0, 1])
                        return r if np.isfinite(r) else -1.0

                    best_bridge = max(reachable_bridges, key=_bridge_score)

                    if cg.is_undirected(a_idx_best, best_bridge):
                        _orient_edge(a_idx_best, best_bridge)
                        oriented_any = True
                        bridge_found = True
                        if verbose:
                            print(
                                f"[AncestralKnowledge] Bridge-oriented "
                                f"{_var_label(a_idx_best)} → {_var_label(best_bridge)} "
                                f"(required_ancestor via bridge to {desc_name})"
                            )
                    elif cg.is_fully_directed(a_idx_best, best_bridge):
                        # Edge already oriented A→bridge; the path exists implicitly
                        bridge_found = True

            # --- Step 4: Direct force-add as last resort ---
            if not bridge_found:
                if a_idx_best < n:
                    edge_ex = cg.G.get_edge(cg.G.nodes[a_idx_best], cg.G.nodes[d_idx])
                    if edge_ex is not None:
                        cg.G.remove_edge(edge_ex)
                    cg.G.add_edge(
                        Edge(
                            cg.G.nodes[a_idx_best],
                            cg.G.nodes[d_idx],
                            Endpoint.TAIL,
                            Endpoint.ARROW,
                        )
                    )
                    oriented_any = True
                    if verbose:
                        print(
                            f"[AncestralKnowledge] Force-added "
                            f"{_var_label(a_idx_best)} → {_var_label(d_idx)} "
                            f"(required_ancestor, no bridge, lag={best_k})"
                        )
                else:
                    violations["required"].append((anc_name, desc_name))

    # Re-run Meek to propagate required_ancestor orientations
    if oriented_any:
        cg = meek(
            cg,
            background_knowledge=background_knowledge,
            num_lags=num_lags,
            forbidden_ancestor_pairs=forb_pairs if forb_pairs else None,
        )

    # --- Forbidden ancestor: reactive two-pass last-edge removal ---
    #
    # Pass 1: Remove A(t-k)→D AND all lag-shifted propagated copies
    #         A(t-k-1)→D(t-1), A(t-k-2)→D(t-2), ...
    #         This mirrors skeleton_discovery.remove_edge lag-propagation and
    #         prevents spurious Pass-2 triggers via autoregressive chains.
    # Pass 2: If a genuine multi-hop path still exists, remove last-hop Y→D.
    # Undirected Y—D is never touched; proactive Meek guard prevents →D.

    def _mark_new_triples_ambiguous(a: int, b: int) -> None:
        """Mark the triples a - m - b that removing a - b unshields as ambiguous.

        The collider step never classified them, so Meek must not treat them
        as known non-colliders and orient through them.
        """
        common = set(cg.neighbors(a)) & set(cg.neighbors(b))
        if not common:
            return
        if not hasattr(cg, "ambiguous_triples"):
            cg.ambiguous_triples = set()
        for m in common:
            if cg.is_fully_directed(a, m) and cg.is_fully_directed(b, m):
                continue  # already an oriented collider
            cg.ambiguous_triples.update({(a, int(m), b), (b, int(m), a)})

    def _required(a: int, b: int) -> bool:
        return background_knowledge is not None and background_knowledge.is_required(
            cg.G.nodes[a], cg.G.nodes[b]
        )

    pending = []  # edges removed by the current cut, logged once it is kept
    cut_log = []  # removals not yet printed (the cut-set search may replace them)

    def _cuttable_copies(xi: int, xj: int, graph=None):
        """xi→xj and its lag-shifted copies that are fully directed, not required."""
        graph = cg.G.graph if graph is None else graph
        max_step = max(xi // n_per_lag, xj // n_per_lag)
        out = []
        for shift in range(num_lags - max_step + 1):
            nx = xi + shift * n_per_lag
            ny = xj + shift * n_per_lag
            if nx >= n or ny >= n:
                break
            if graph[nx, ny] == -1 and graph[ny, nx] == 1 and not _required(nx, ny):
                out.append((nx, ny))
        return out

    def _remove_edge_with_lags(xi: int, xj: int) -> bool:
        """Remove xi→xj and all lag-shifted copies; return True if anything removed.

        A required copy is kept.
        """
        did_remove = False
        for nx, ny in _cuttable_copies(xi, xj):
            e = cg.G.get_edge(cg.G.nodes[nx], cg.G.nodes[ny])
            if e is not None:
                cg.G.remove_edge(e)
                _mark_new_triples_ambiguous(nx, ny)
                did_remove = True
                pending.append((nx, ny))
        return did_remove

    def _satisfied_required_ancestors(graph=None):
        graph = cg.G.graph if graph is None else graph
        out = set()
        for anc_name, desc_name in knowledge._required_ancestors:
            if anc_name not in var_names or desc_name not in var_names:
                continue
            d = var_names.index(desc_name)
            a0 = var_names.index(anc_name)
            if any(
                reaches(graph, a0 + lag * n_per_lag, d)
                for lag in range(num_lags + 1)
                if a0 + lag * n_per_lag < n
            ):
                out.add((anc_name, desc_name))
        return out

    def _cut(remove, reason: str) -> bool:
        """Run ``remove()``; keep the cut only if it removed something and
        every satisfied required ancestor is still satisfied."""
        pending.clear()
        if not knowledge._required_ancestors:
            kept = remove()
        else:
            needed = _satisfied_required_ancestors()
            # remove() only deletes fully directed edges (listed in pending)
            # and adds ambiguous triples, so both are easy to undo.
            saved_amb = set(getattr(cg, "ambiguous_triples", set()))
            kept = remove() and needed <= _satisfied_required_ancestors()
            if not kept:
                for u, v in pending:
                    cg.G.add_directed_edge(cg.G.nodes[u], cg.G.nodes[v])
                cg.ambiguous_triples = saved_amb
        if kept:
            cut_log.extend((u, v, reason) for u, v in pending)
        return kept

    def _print_cut_log():
        if verbose:
            for u, v, reason in cut_log:
                print(
                    f"[AncestralKnowledge] Removed {_var_label(u)} → "
                    f"{_var_label(v)} ({reason})"
                )

    def _remove_one(u: int, v: int) -> bool:
        e = cg.G.get_edge(cg.G.nodes[u], cg.G.nodes[v])
        if e is None:
            return False
        cg.G.remove_edge(e)
        _mark_new_triples_ambiguous(u, v)
        pending.append((u, v))
        return True

    def _violated(graph):
        return [
            (a, d) for a, d in forb_pairs if a < n and d < n and reaches(graph, a, d)
        ]

    def _search_cut_set(graph):
        """Fewest non-required directed edges (up to _CUT_SEARCH_MAX_SIZE) whose
        removal leaves no forbidden path and keeps every satisfied required
        ancestor, or None.

        Edges are cut with their lag copies, as in the passes above (the limit then
        counts an edge and its copies as one), if some such set works; otherwise
        one at a time.
        """
        pairs = _violated(graph)
        # For each satisfied required ancestor, the (lag copy, descendant) pairs
        # any one of which keeps it satisfied.
        needed = []
        for anc_name, desc_name in _satisfied_required_ancestors(graph):
            a0, d = var_names.index(anc_name), var_names.index(desc_name)
            needed.append(
                [
                    (a0 + lag * n_per_lag, d)
                    for lag in range(num_lags + 1)
                    if a0 + lag * n_per_lag < n
                ]
            )
        edges = [
            (int(u), int(v))
            for u, v in zip(*np.nonzero((graph == -1) & (graph.T == 1)))
            if not _required(int(u), int(v))
            and any(reaches(graph, a, u) and reaches(graph, v, d) for a, d in pairs)
        ]

        seen = set()  # cuts tried, shared by both rounds

        def smallest(groups):
            tried = 0  # per round, so the single-edge round always gets its turn
            for k in range(1, min(len(groups), _CUT_SEARCH_MAX_SIZE) + 1):
                for combo in combinations(groups, k):
                    cut = list(dict.fromkeys(e for group in combo for e in group))
                    if frozenset(cut) in seen:
                        continue  # overlapping groups give the same cut
                    tried += 1
                    if tried > _CUT_SEARCH_BUDGET:
                        return None
                    seen.add(frozenset(cut))
                    g = graph.copy()
                    for u, v in cut:
                        g[u, v] = g[v, u] = 0
                    # Cuts create no path, so only pairs violated in graph matter.
                    if not any(reaches(g, a, d) for a, d in pairs) and all(
                        any(reaches(g, a, d) for a, d in alts) for alts in needed
                    ):
                        return cut
            return None

        groups = [_cuttable_copies(u, v, graph) for u, v in edges]
        singles = [[e] for e in edges]
        found = smallest(groups)
        if found is None and groups != singles:
            found = smallest(singles)
        return found

    removed_any = False

    for anc_name, desc_name in knowledge._forbidden_ancestors:
        if anc_name not in var_names or desc_name not in var_names:
            continue
        d_idx = var_names.index(desc_name)
        if d_idx >= n:
            continue

        for lag in range(num_lags + 1):
            a_idx = var_names.index(anc_name) + lag * n_per_lag
            if a_idx >= n:
                continue
            reason = f"direct, forbidden_ancestor {anc_name}→{desc_name}"
            if _cut(lambda: _remove_edge_with_lags(a_idx, d_idx), reason):
                removed_any = True
                continue
            # A copy is needed by a required ancestor: cut the others one by one.
            for nx, ny in _cuttable_copies(a_idx, d_idx):
                if _cut(lambda: _remove_one(nx, ny), reason):
                    removed_any = True

    if removed_any:
        cg = meek(
            cg,
            background_knowledge=background_knowledge,
            num_lags=num_lags,
            forbidden_ancestor_pairs=forb_pairs if forb_pairs else None,
        )

    # Greedy cuts that respect required edges and ancestors can leave a
    # violation another cut set avoids; keep the state to search from.
    _print_cut_log()
    cut_log.clear()
    cg_p1 = (
        deepcopy(cg)
        if (knowledge._required_ancestors or background_knowledge is not None)
        and _violated(cg.G.graph)
        else None
    )

    # Pass 2: check for surviving genuine multi-hop paths; remove last-hop Y→D
    removed_any_p2 = False
    blocked = []  # (a_idx, d_idx) with a last hop that could not be cut
    for anc_name, desc_name in knowledge._forbidden_ancestors:
        if anc_name not in var_names or desc_name not in var_names:
            continue
        d_idx = var_names.index(desc_name)
        if d_idx >= n:
            continue

        for lag in range(num_lags + 1):
            a_idx = var_names.index(anc_name) + lag * n_per_lag
            if a_idx >= n:
                continue
            if not reaches(cg.G.graph, a_idx, d_idx):
                continue  # pass 1 already broke this path

            for y_idx in range(n):
                if y_idx == d_idx:
                    continue
                if not cg.is_fully_directed(y_idx, d_idx):
                    continue
                if y_idx == a_idx or reaches(cg.G.graph, a_idx, y_idx):
                    if _required(y_idx, d_idx) or not _cut(
                        lambda: _remove_one(y_idx, d_idx),
                        f"last-hop, forbidden_ancestor {anc_name}→{desc_name}",
                    ):
                        blocked.append((a_idx, d_idx))
                        continue
                    removed_any_p2 = True

    # A last hop that is required, or needed by a satisfied required ancestor,
    # stays: cut each remaining path at its last edge that can go (with its
    # lag copies, as in pass 1), a shortest path at a time.
    for a_idx, d_idx in blocked:
        while True:
            path = _directed_path(cg.G.graph, a_idx, d_idx)
            if path is None:
                break
            hops = [(u, v) for u, v in zip(path, path[1:]) if not _required(u, v)]
            reason = (
                f"blocked last hop, forbidden_ancestor "
                f"{_var_label(a_idx)}→{_var_label(d_idx)}"
            )
            for u, v in reversed(hops):
                # With its lag copies if possible, else the edge alone.
                if _cut(lambda: _remove_edge_with_lags(u, v), reason) or (
                    num_lags > 0 and _cut(lambda: _remove_one(u, v), reason)
                ):
                    removed_any_p2 = True
                    break
            else:
                break  # no admissible cut: reported as a violation below

    if removed_any_p2:
        cg = meek(
            cg,
            background_knowledge=background_knowledge,
            num_lags=num_lags,
            forbidden_ancestor_pairs=forb_pairs if forb_pairs else None,
        )

    if cg_p1 is not None and _violated(cg.G.graph):
        cut_set = _search_cut_set(cg_p1.G.graph)
        if cut_set is not None:
            cg_greedy, cg = cg, cg_p1
            # Keep what either graph satisfies (Meek after the greedy cuts can
            # satisfy more than the snapshot).
            needed = _satisfied_required_ancestors() | _satisfied_required_ancestors(
                cg_greedy.G.graph
            )
            for u, v in cut_set:
                _remove_one(u, v)
            cg = meek(
                cg,
                background_knowledge=background_knowledge,
                num_lags=num_lags,
                forbidden_ancestor_pairs=forb_pairs if forb_pairs else None,
            )
            if _violated(cg.G.graph) or not needed <= _satisfied_required_ancestors():
                cg = cg_greedy
            else:
                cut_log = [
                    (u, v, "cut-set search, forbidden_ancestor") for u, v in cut_set
                ]

    _print_cut_log()

    # Post-hoc violation check: report any remaining paths
    for a_idx, d_idx in forb_pairs:
        if a_idx >= n or d_idx >= n:
            continue
        if reaches(cg.G.graph, a_idx, d_idx):
            violations["forbidden"].append((_var_label(a_idx), _var_label(d_idx)))
            if verbose:
                print(
                    f"[AncestralKnowledge] VIOLATION forbidden_ancestor "
                    f"{_var_label(a_idx)} → {_var_label(d_idx)}: "
                    "path still exists after last-edge removal"
                )

    return cg, violations
