# Copyright 2021 Juan L. Gamella (original GES)
# Copyright 2025 Adiba Ejaz, Elias Bareinboim (LGES modifications)
# SPDX-License-Identifier: BSD-3-Clause
#
# Self-contained implementation of LGES (Less Greedy Equivalence Search)
# extracted from https://github.com/CausalAILab/lges (BSD-3-Clause).
#
# Wrapper ``lges_discovery()`` adds lag-embedding for time series.

from itertools import combinations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Graph utilities (from ges/ges/utils.py)
# ---------------------------------------------------------------------------


def _na(y, x, A):
    """Neighbors of *y* that are adjacent to *x*."""
    return neighbors(y, A) & adj(x, A)


def neighbors(i, A):
    """Undirected neighbors of node *i* in PDAG *A*."""
    return set(np.where(np.logical_and(A[i, :] != 0, A[:, i] != 0))[0])


def adj(i, A):
    """All nodes adjacent to *i* (directed or undirected)."""
    return set(np.where(np.logical_or(A[i, :] != 0, A[:, i] != 0))[0])


def pa(i, A):
    """Parents of *i* (nodes with directed edge into *i*)."""
    return set(np.where(np.logical_and(A[:, i] != 0, A[i, :] == 0))[0])


def ch(i, A):
    """Children of *i*."""
    return set(np.where(np.logical_and(A[i, :] != 0, A[:, i] == 0))[0])


def is_clique(S, A):
    """Check if *S* forms a clique in *A*."""
    S = list(S)
    for i, j in combinations(S, 2):
        if A[i, j] == 0 and A[j, i] == 0:
            return False
    return True


def is_dag(A):
    """Check if *A* is a DAG (no cycles)."""
    try:
        topological_ordering(A)
        return True
    except ValueError:
        return False


def skeleton(A):
    """Undirected skeleton of *A*."""
    S = np.zeros_like(A)
    S[A != 0] = 1
    S = np.maximum(S, S.T)
    return S


def only_directed(A):
    """Return matrix with only directed edges."""
    D = A.copy()
    for i in range(len(A)):
        for j in range(len(A)):
            if A[i, j] != 0 and A[j, i] != 0:
                D[i, j] = 0
    return D


def only_undirected(A):
    """Return matrix with only undirected edges."""
    U = A.copy()
    for i in range(len(A)):
        for j in range(len(A)):
            if A[i, j] != 0 and A[j, i] == 0:
                U[i, j] = 0
    return U


def vstructures(A):
    """Yield v-structures (i -> j <- k) where i and k are not adjacent."""
    p = len(A)
    for j in range(p):
        parents = list(pa(j, A))
        for i, k in combinations(parents, 2):
            if A[i, k] == 0 and A[k, i] == 0:
                yield (i, j, k)


def topological_ordering(A):
    """Kahn's algorithm. Raises ValueError if *A* is cyclic."""
    p = len(A)
    D = only_directed(A)
    in_degree = np.sum(D != 0, axis=0)
    queue = list(np.where(in_degree == 0)[0])
    ordering = []
    while queue:
        node = queue.pop(0)
        ordering.append(node)
        for child in np.where(D[node, :] != 0)[0]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)
    if len(ordering) != p:
        raise ValueError("Graph has a cycle")
    return ordering


def chain_component(i, G):
    """The undirected-connected component containing *i*."""
    U = only_undirected(G)
    visited, to_visit = set(), {i}
    while to_visit:
        j = to_visit.pop()
        visited.add(j)
        to_visit |= neighbors(j, U) - visited
    return visited


def induced_subgraph(S, G):
    """The subgraph of *G* induced by the node set *S*."""
    mask = np.zeros_like(G, dtype=bool)
    mask[list(S), :] = True
    mask = np.logical_and(mask, mask.T)
    sub = np.zeros_like(G)
    sub[mask] = G[mask]
    return sub


def separates(S, A_set, B_set, G):
    """Does *S* block every semi-directed path between *A_set* and *B_set* in *G*?

    Searches for one surviving path instead of enumerating all of them.
    """
    if (A_set & B_set) or (A_set & S) or (B_set & S):
        raise ValueError("S, A and B must be pairwise disjoint")
    for a in A_set:
        stack, seen = [a], {a}
        while stack:
            cur = stack.pop()
            if cur in B_set:
                return False
            for nxt in np.where(G[cur, :] != 0)[0]:
                if nxt in seen or nxt in S:
                    continue
                seen.add(nxt)
                stack.append(nxt)
    return True


def cartesian(arrays):
    """Cartesian product of a list of arrays."""
    arrays = [np.asarray(a) for a in arrays]
    n = 1
    for a in arrays:
        n *= len(a)
    out = np.zeros((n, len(arrays)), dtype=int)
    m = n
    for i, a in enumerate(arrays):
        m //= len(a)
        out[:, i] = np.tile(np.repeat(a, m), n // (len(a) * m))
    return out


def subsets(S, max_size=None):
    """All subsets of set *S*, optionally capped at *max_size* elements."""
    S = list(S)
    result = [set()]
    for s in S:
        result += [r | {s} for r in result if max_size is None or len(r) < max_size]
    return result


def sort(L, order):
    """Sort list *L* by *order*."""
    return sorted(L, key=lambda x: order[x])


def delete(i, j, H, A):
    """Apply delete operator: remove i->j, orient NA_yj \\ H toward j."""
    new_A = A.copy()
    new_A[i, j] = 0
    new_A[j, i] = 0
    for h in H:
        if new_A[j, h] != 0:
            new_A[j, h] = 0
            new_A[h, j] = 1 if new_A[h, j] == 0 else new_A[h, j]
    return new_A


# --- PDAG -> DAG -> CPDAG pipeline ---


def pdag_to_dag(P, forbidden=None):
    """Dor-Tarsi algorithm: find a consistent extension DAG of PDAG *P*.

    Repeatedly pick a node ``x`` of the remaining subgraph that is a sink (no
    directed edge leaves it) and whose undirected neighbours are adjacent to
    everything else adjacent to ``x``; orient every undirected edge at ``x``
    *into* ``x``, then remove it. Orienting at removal time is what keeps the
    result acyclic — deriving the directions afterwards from the removal order
    inverts them, because the first node removed is the last topologically.

    Plain Dor-Tarsi treats every undirected edge as equally resolvable in
    either direction, which is wrong once ``forbidden`` says otherwise. But
    rejecting a *candidate sink* whenever any neighbour has a forced direction
    (an earlier version of this function did that) is incomplete: it can
    report "no consistent extension" for perfectly resolvable graphs, because
    Dor-Tarsi's one move -- turn every undirected edge at a chosen node into an
    incoming edge -- has no way to express "most of these neighbours are free,
    but this one specific edge must point out."

    The fix used here relies on a property specific to how ``forbidden`` is
    built by :func:`temporal_forbidden`: for any two nodes, either neither
    direction is forbidden (free, e.g. two variables at the same lag) or
    *exactly* one is (never both -- that would make the pair unconnectable).
    So every such edge already has a unique legal direction with no search
    needed; force them all before Dor-Tarsi runs, and hand it a graph with no
    remaining background knowledge to violate. Classical Dor-Tarsi's
    completeness guarantee then applies exactly as in the unconstrained case.
    """
    P = P.copy()
    if forbidden is not None:
        p = len(P)
        for i in range(p):
            for j in range(p):
                if i == j or P[i, j] == 0 or P[j, i] == 0:
                    continue  # already directed, or no edge
                if forbidden[j, i] != 0:
                    # j -> i is illegal, i -> j is the only option (and, by
                    # construction, is not itself forbidden).
                    P[j, i] = 0

    remaining_P = P.copy()
    p = len(P)
    G = only_directed(P).copy()
    remaining = set(range(p))
    while remaining:
        for x in sorted(remaining):
            if not _is_sink_in_subgraph(x, remaining_P):
                continue
            for y in neighbors(x, remaining_P):
                G[y, x] = 1
                G[x, y] = 0
            remaining.discard(x)
            remaining_P[x, :] = 0
            remaining_P[:, x] = 0
            break
        else:
            raise ValueError("No consistent extension exists")
    return G


def _is_sink_in_subgraph(x, P):
    """Is *x* a Dor-Tarsi sink of the subgraph *P* (removed nodes zeroed out)?"""
    if len(ch(x, P)) > 0:  # a directed edge leaves x
        return False
    adj_x = adj(x, P)
    for y in neighbors(x, P):
        for z in adj_x - {y}:
            if P[y, z] == 0 and P[z, y] == 0:
                return False
    return True


def dag_to_cpdag(G):
    """Convert DAG *G* to its CPDAG (completed PDAG)."""
    p = len(G)
    ordering = topological_ordering(G)
    order_map = {node: idx for idx, node in enumerate(ordering)}
    edges = []
    for i in range(p):
        for j in range(p):
            if G[i, j] != 0:
                edges.append((i, j))
    ordered_edges = order_edges(edges, order_map)
    return label_edges(ordered_edges, G)


def order_edges(edges, order_map):
    """Order edges for labelling (Chickering's Order-Edges).

    Heads are visited in topological order; among the edges sharing a head, the
    one whose *tail* comes latest in the topological order is ordered first.
    Sorting the tail ascending instead reverses that inner order and makes
    ``label_edges`` mark reversible edges as compelled.
    """

    def edge_key(e):
        return (order_map[e[1]], -order_map[e[0]])

    return sorted(edges, key=edge_key)


def label_edges(ordered_edges, G):
    """Label each edge of DAG *G* compelled or reversible (Chickering 1995).

    Phase one — propagating compelledness *through* ``x`` — is what forces edges
    that no v-structure pins down but that Meek's rules require. Omitting it
    leaves such edges reversible, so the CPDAG comes back under-oriented.
    """
    p = len(G)
    labels = {e: "unknown" for e in ordered_edges}

    def parents(v):
        return set(np.where(G[:, v] != 0)[0])

    for x, y in ordered_edges:
        if labels[(x, y)] != "unknown":
            continue
        parents_y = parents(y)

        # Phase 1: every compelled w -> x either forces x -> y outright (when w
        # is not also a parent of y) or makes w -> y compelled.
        done = False
        for w in sorted(parents(x)):
            if labels.get((w, x)) != "compelled":
                continue
            if w not in parents_y:
                labels[(x, y)] = "compelled"
                for z in parents_y:
                    labels[(z, y)] = "compelled"
                done = True
                break
            labels[(w, y)] = "compelled"
        if done:
            continue

        # Phase 2: an unshielded parent of y other than x makes x -> y compelled;
        # otherwise x -> y and every remaining unlabelled edge into y are reversible.
        parents_x = parents(x)
        compelled = any(z not in parents_x for z in parents_y - {x})
        label = "compelled" if compelled else "reversible"
        labels[(x, y)] = label
        for z in parents_y:
            if labels.get((z, y)) == "unknown":
                labels[(z, y)] = label
    cpdag = np.zeros((p, p))
    for (i, j), lbl in labels.items():
        if lbl == "compelled":
            cpdag[i, j] = 1
        elif lbl == "reversible":
            cpdag[i, j] = 1
            cpdag[j, i] = 1
    return cpdag


def pdag_to_cpdag(P, forbidden=None):
    """PDAG -> consistent DAG -> CPDAG.

    Pass ``forbidden`` whenever *P* came from a search that used it -- without
    it, ``pdag_to_dag``'s arbitrary sink choice can resolve an undirected edge
    in the one direction background knowledge rules out. ``dag_to_cpdag`` never
    reverses an edge it is given, so a legally-extended DAG stays legal.
    """
    try:
        G = pdag_to_dag(P, forbidden)
    except ValueError:
        return P.copy()
    return dag_to_cpdag(G)


# ---------------------------------------------------------------------------
# Score classes (from ges/ges/scores/)
# ---------------------------------------------------------------------------


class DecomposableScore:
    """Base class for decomposable score functions with caching."""

    def __init__(self, data, cache=True, debug=0):
        self.data = data
        self.n, self.p = data.shape
        self._cache = {} if cache else None
        self.debug = debug

    def local_score(self, x, pa):
        pa = tuple(sorted(pa))
        key = (x, pa)
        if self._cache is not None and key in self._cache:
            return self._cache[key]
        score = self._compute_local_score(x, pa)
        if self._cache is not None:
            self._cache[key] = score
        return score

    def _compute_local_score(self, x, pa):
        raise NotImplementedError

    def score_dag(self, A):
        total = 0
        for j in range(self.p):
            parents = tuple(sorted(np.where(A[:, j] != 0)[0]))
            total += self.local_score(j, parents)
        return total


class GaussObsL0Pen(DecomposableScore):
    """Cached L0-penalized Gaussian BIC score."""

    def __init__(self, data, lmbda=None, cache=True, debug=0):
        super().__init__(data, cache=cache, debug=debug)
        self.lmbda = 0.5 * np.log(self.n) if lmbda is None else lmbda
        self._scatter = np.cov(data, rowvar=False, ddof=0)
        if self._scatter.ndim == 0:
            self._scatter = self._scatter.reshape(1, 1)

    def _compute_local_score(self, x, pa):
        pa = list(pa)
        sigma = self._scatter[x, x]
        if len(pa) > 0:
            cov_pa = self._scatter[np.ix_(pa, pa)]
            cov_xpa = self._scatter[x, pa]
            try:
                coef = np.linalg.solve(cov_pa, cov_xpa)
            except np.linalg.LinAlgError:
                coef = np.linalg.lstsq(cov_pa, cov_xpa, rcond=None)[0]
            sigma = sigma - cov_xpa @ coef
        if sigma <= 0:
            sigma = np.finfo(float).eps
        likelihood = -0.5 * self.n * (1 + np.log(sigma))
        penalty = self.lmbda * (len(pa) + 1)
        return likelihood - penalty


# ---------------------------------------------------------------------------
# LGES algorithm (from ges/ges/main.py)
# ---------------------------------------------------------------------------

ALPHA = 0.05


def fit(
    score_class,
    A0=None,
    phases=None,
    prune=False,
    score_based=False,
    required=None,
    forbidden=None,
    max_subset_size=3,
    debug=0,
):
    """Run GES/LGES on the given score class.

    Parameters
    ----------
    score_class : DecomposableScore
        Scoring function (e.g. ``GaussObsL0Pen``).
    A0 : ndarray or None
        Initial CPDAG. Defaults to the empty graph.
    phases : list of str or None
        Phases to run: ``["forward", "backward", "turning"]``.
    prune : bool
        Enable ConservativeInsert (early stopping on lower-score operators).
    score_based : bool
        Enable SafeInsert (score check before insert).
    required : ndarray or None
        Required edges matrix.
    forbidden : ndarray or None
        Forbidden edges matrix.
    max_subset_size : int or None
        Cap on subset enumeration size for operator scoring. Prevents
        combinatorial blowup for dense graphs. None = no cap.
    debug : int
        Verbosity level.

    Returns
    -------
    A : ndarray
        Estimated CPDAG adjacency matrix.
    metrics : dict
        Runtime metrics and score.
    """
    p = score_class.p
    if A0 is None:
        A0 = np.zeros((p, p))
    if phases is None:
        phases = ["forward", "backward", "turning"]
    if required is None:
        required = np.zeros((p, p))
    if forbidden is None:
        forbidden = np.zeros((p, p))

    A = A0.copy()
    metrics = {
        "inserts_eval": 0,
        "deletes_eval": 0,
        "turns_eval": 0,
        "inserts_actual": 0,
        "deletes_actual": 0,
        "turns_actual": 0,
    }

    import time

    start = time.time()

    if "forward" in phases:
        cont = True
        while cont:
            A, cont, metrics = _forward_step(
                A,
                score_class,
                required,
                forbidden,
                prune=prune,
                score_based=score_based,
                max_subset_size=max_subset_size,
                metrics=metrics,
                debug=debug,
            )

    if "backward" in phases:
        cont = True
        while cont:
            A, cont, metrics = _backward_step(
                A,
                score_class,
                required,
                forbidden,
                max_subset_size=max_subset_size,
                metrics=metrics,
                debug=debug,
            )

    if "turning" in phases:
        # The turn operator scores each move as if the parents of x and y are
        # the only thing that changes, but pdag_to_cpdag's completion step can
        # legally reorient other, unrelated edges as a side effect (new
        # v-structures, Meek propagation). When that happens the score gain
        # the next candidate reports is computed against a baseline that
        # completion already changed out from under it, and under background
        # knowledge (forbidden) that can produce a genuine 2-cycle: turn A
        # applied, turn B claims a gain that undoes A's effect, A is offered
        # again, forever. Bound the loop with visited-state detection rather
        # than assume every accepted move is real forward progress.
        cont = True
        seen = {A.tobytes()}
        while cont:
            A, cont, metrics = _turning_step(
                A,
                score_class,
                required,
                forbidden,
                max_subset_size=max_subset_size,
                metrics=metrics,
                debug=debug,
            )
            state = A.tobytes()
            if state in seen:
                break
            seen.add(state)

    metrics["time"] = time.time() - start
    try:
        metrics["score"] = score_class.score_dag(pdag_to_dag(A))
    except ValueError:
        metrics["score"] = float("nan")

    return A, metrics


def _check_legal_insert(x, y, A, score_class, score_based, metrics):
    """SafeInsert: check if local_score(y, pa(y) | {x}) > local_score(y, pa(y))."""
    if not score_based:
        return True
    metrics["inserts_eval"] += 1
    pa_y = pa(y, A)
    score_with = score_class.local_score(y, pa_y | {x})
    score_without = score_class.local_score(y, pa_y)
    return score_with > score_without


def _get_priority_inserts(A, required, forbidden):
    """Group non-adjacent pairs by priority: required first, forbidden last."""
    p = len(A)
    priority_required = []
    priority_normal = []
    for x in range(p):
        for y in range(p):
            if x == y:
                continue
            if A[x, y] != 0 or A[y, x] != 0:
                continue
            if forbidden[x, y] != 0:
                continue
            if required[x, y] != 0:
                priority_required.append((x, y))
            else:
                priority_normal.append((x, y))
    return priority_required + priority_normal


def _forward_step(
    A,
    score_class,
    required,
    forbidden,
    prune=False,
    score_based=False,
    max_subset_size=3,
    metrics=None,
    debug=0,
):
    """GES forward phase: find best insert operator and apply it."""
    p = len(A)  # noqa: F841
    best_score = 0
    best_operator = None

    candidates = _get_priority_inserts(A, required, forbidden)

    for x, y in candidates:
        if not _check_legal_insert(x, y, A, score_class, score_based, metrics):
            continue

        operators, found_lower = _score_valid_insert_operators(
            x,
            y,
            A,
            score_class,
            forbidden,
            prune=prune,
            max_subset_size=max_subset_size,
            debug=debug,
        )
        metrics["inserts_eval"] += len(operators)

        if prune and found_lower:
            continue

        for score_delta, T in operators:
            if score_delta > best_score:
                best_score = score_delta
                best_operator = ("insert", x, y, T)

    if best_operator is None:
        return A, False, metrics

    _, x, y, T = best_operator
    new_A = _apply_insert(x, y, T, A)
    new_A = pdag_to_cpdag(new_A, forbidden)
    metrics["inserts_actual"] += 1
    if debug:
        print(f"  Insert {x} -> {y} | T={T}, delta={best_score:.4f}")
    return new_A, True, metrics


def _backward_step(
    A, score_class, required, forbidden, max_subset_size=3, metrics=None, debug=0
):
    """GES backward phase: find best delete operator and apply it."""
    p = len(A)
    best_score = 0
    best_operator = None

    for x in range(p):
        for y in range(p):
            if A[x, y] == 0:
                continue
            if required[x, y] != 0:
                continue

            operators = _score_valid_delete_operators(
                x,
                y,
                A,
                score_class,
                forbidden,
                max_subset_size=max_subset_size,
                debug=debug,
            )
            metrics["deletes_eval"] += len(operators)

            for score_delta, H in operators:
                if score_delta > best_score:
                    best_score = score_delta
                    best_operator = ("delete", x, y, H)

    if best_operator is None:
        return A, False, metrics

    _, x, y, H = best_operator
    new_A = _apply_delete(x, y, H, A)
    new_A = pdag_to_cpdag(new_A, forbidden)
    metrics["deletes_actual"] += 1
    if debug:
        print(f"  Delete {x} -> {y} | H={H}, delta={best_score:.4f}")
    return new_A, True, metrics


def _turning_step(
    A, score_class, required, forbidden, max_subset_size=3, metrics=None, debug=0
):
    """GES turning phase: find best turn operator and apply it."""
    best_score = 0
    best_operator = None

    # Candidates are the reverse of every present edge: for an edge src -> dst
    # (or src - dst) we consider turning it so that it points dst -> src.
    src, dst = np.where(A != 0)
    for x, y in zip(dst, src):
        if x == y:
            continue
        # The operator creates x -> y, so that is the direction to check.
        if forbidden[x, y] != 0:
            continue

        operators = _score_valid_turn_operators(
            x,
            y,
            A,
            score_class,
            forbidden,
            max_subset_size=max_subset_size,
            debug=debug,
        )
        metrics["turns_eval"] += len(operators)

        for score_delta, C in operators:
            if score_delta > best_score:
                best_score = score_delta
                best_operator = ("turn", x, y, C)

    if best_operator is None:
        return A, False, metrics

    _, x, y, C = best_operator
    new_A = _apply_turn(x, y, C, A)
    new_A = pdag_to_cpdag(new_A, forbidden)
    metrics["turns_actual"] += 1
    if debug:
        print(f"  Turn to {x} -> {y} | C={C}, delta={best_score:.4f}")
    return new_A, True, metrics


# --- Operator validity and scoring ---


def _score_valid_insert_operators(
    x, y, A, score_class, forbidden, prune=False, max_subset_size=3, debug=0
):
    """Score all valid insert(x, y, T) operators.

    Returns list of ``(score_delta, T)`` and a bool ``found_lower_scoring``.
    """
    na_yx = _na(y, x, A)
    pa_y = pa(y, A)
    operators = []
    found_lower = False

    # T ranges over subsets of Ne(y) \\ Adj(x) -- neighbours of y NOT adjacent to
    # x. Drawing T from na_yx (= Ne(y) INTERSECT Adj(x)) instead explores the wrong
    # operators and makes condition 1 vacuous, since T would already be inside na_yx.
    # Every t in T is oriented t -> y by _apply_insert, so any t forbidden from
    # causing y must never enter the candidate pool -- checking only the (x, y)
    # pair (as the caller does) misses this, since T is a second, independent
    # source of new edges into y.
    candidates_T = {t for t in neighbors(y, A) - adj(x, A) if forbidden[t, y] == 0}

    for T in subsets(candidates_T, max_size=max_subset_size):
        # Validity condition 1: NA_yx ∪ T is a clique
        if not is_clique(na_yx | T, A):
            continue
        # Validity condition 2: every semi-directed path y -> x is blocked by NA_yx ∪ T
        if _creates_cycle(x, y, T, A):
            continue

        # The undirected neighbours in NA_yx and the members of T all become
        # parents of y under this operator, so they belong in BOTH terms of the
        # delta; omitting them scores a different operator than the one applied.
        base = pa_y | na_yx | T
        old_score = score_class.local_score(y, base)
        new_score = score_class.local_score(y, base | {x})
        score_delta = new_score - old_score

        if prune and score_delta < 0:
            found_lower = True
            return operators, found_lower

        operators.append((score_delta, T))

    return operators, found_lower


def _score_valid_delete_operators(
    x, y, A, score_class, forbidden, max_subset_size=3, debug=0
):
    """Score all valid delete(x, y, H) operators."""
    na_yx = _na(y, x, A)
    pa_y = pa(y, A)
    n_x = neighbors(x, A)
    operators = []

    # _apply_delete orients every h in H as y -> h, and additionally x -> h
    # when h is also an undirected neighbour of x. Either forbidden[y, h] or
    # (h in n_x and forbidden[x, h]) makes h unusable, regardless of the (x, y)
    # pair's own validity.
    candidates_H = {
        h
        for h in na_yx - {x}
        if forbidden[y, h] == 0 and not (h in n_x and forbidden[x, h] != 0)
    }

    for H in subsets(candidates_H, max_size=max_subset_size):
        # Validity: NA_yx \\ H is a clique
        if not is_clique(na_yx - H, A):
            continue

        # Both terms share the base pa_y | (NA_yx \\ H) and differ only by x --
        # the operator removes x, nothing else. Keeping H in the "old" set makes
        # the delta also charge for dropping H, scoring a different move.
        base = (na_yx - H) | pa_y
        old_parents = base | {x}
        new_parents = base - {x}

        old_score = score_class.local_score(y, old_parents)
        # Skip when old_score is -inf (inadmissible parent set, e.g. tier
        # violation in TieredGaussObsL0Pen): new_score - (-inf) = +inf would
        # trigger a spurious delete of a valid edge.
        if old_score == -np.inf:
            continue
        new_score = score_class.local_score(y, new_parents)
        score_delta = new_score - old_score

        operators.append((score_delta, H))

    return operators


def _turn_unblocked_path(x, y, C, A):
    """Does a semi-directed path y -> x survive the blocking set ``C | ne(x)``?

    The direct edge y-x is exempt. Reachability rather than path enumeration:
    enumerating every path is exponential and made the turning phase dominate
    runtime on dense graphs.
    """
    blocked = set(C) | neighbors(x, A)
    stack, seen = [y], {y}
    while stack:
        cur = stack.pop()
        for nxt in np.where(A[cur, :] != 0)[0]:
            if nxt == x:
                if cur != y:  # a path of length > 1 got through
                    return True
                continue
            if nxt in seen or nxt in blocked:
                continue
            seen.add(nxt)
            stack.append(nxt)
    return False


def _score_valid_turn_operators_dir(x, y, A, score_class, forbidden, max_subset_size=3):
    """Turn the directed edge y -> x into x -> y (upstream ges.main)."""
    na_yx = _na(y, x, A)
    pa_y, pa_x = pa(y, A), pa(x, A)
    out = []
    # _apply_turn orients every member of C = na_yx | T as c -> y. na_yx is fixed
    # per (x, y), so if it already contains a node forbidden from causing y, every
    # possible C is invalid and there is nothing to search.
    if any(forbidden[c, y] != 0 for c in na_yx):
        return out
    candidates_T = {t for t in neighbors(y, A) - adj(x, A) if forbidden[t, y] == 0}
    for T in subsets(candidates_T, max_size=max_subset_size):
        C = na_yx | T
        if not is_clique(C, A):
            continue
        if _turn_unblocked_path(x, y, C, A):
            continue
        new = score_class.local_score(y, pa_y | C | {x}) + score_class.local_score(
            x, pa_x - {y}
        )
        old = score_class.local_score(y, pa_y | C) + score_class.local_score(x, pa_x)
        out.append((new - old, C))
    return out


def _score_valid_turn_operators_undir(
    x, y, A, score_class, forbidden, max_subset_size=3
):
    """Turn the undirected edge y - x into x -> y (upstream ges.main)."""
    non_adjacent = neighbors(y, A) - adj(x, A) - {x}
    if not non_adjacent:
        return []
    na_yx = _na(y, x, A)
    pa_y, pa_x = pa(y, A), pa(x, A)
    subgraph = induced_subgraph(chain_component(y, A), A)
    out = []
    # _apply_turn orients every member of C as c -> y, so C's candidate pool
    # must exclude anything forbidden from causing y.
    candidates_C = {c for c in neighbors(y, A) - {x} if forbidden[c, y] == 0}
    for C in subsets(candidates_C, max_size=max_subset_size):
        # C must contain at least one neighbour of y that is not adjacent to x
        if not (C & non_adjacent):
            continue
        if not is_clique(C, A):
            continue
        if not separates({x, y}, C - {x, y}, (na_yx - C) - {x, y}, subgraph):
            continue
        new = score_class.local_score(y, pa_y | C | {x}) + score_class.local_score(
            x, pa_x | (C & na_yx)
        )
        old = score_class.local_score(y, pa_y | C) + score_class.local_score(
            x, pa_x | (C & na_yx) | {y}
        )
        out.append((new - old, C))
    return out


def _score_valid_turn_operators(
    x, y, A, score_class, forbidden, max_subset_size=3, debug=0
):
    """Score every valid turn(x, y, C), producing the edge x -> y."""
    if A[x, y] != 0 and A[y, x] == 0:
        return []  # x -> y already exists
    if A[x, y] == 0 and A[y, x] == 0:
        return []  # not connected
    if A[x, y] != 0 and A[y, x] != 0:
        return _score_valid_turn_operators_undir(
            x, y, A, score_class, forbidden, max_subset_size=max_subset_size
        )
    return _score_valid_turn_operators_dir(
        x, y, A, score_class, forbidden, max_subset_size=max_subset_size
    )


def _creates_cycle(x, y, T, A):
    """Is insert(x, y, T) invalid because some semi-directed path is unblocked?

    GES requires every semi-directed path from *y* to *x* to contain a node of
    ``NA_yx | T``. Rejecting the operator whenever *any* such path exists — as
    opposed to any *unblocked* one — starves the forward phase: paths multiply as
    the graph grows, so the search stalls long before it reaches the optimum.

    Searches for one unblocked path rather than enumerating them all, which also
    avoids the exponential blow-up of enumerating every path on dense graphs.
    """
    blocked = _na(y, x, A) | set(T)
    stack, seen = [y], {y}
    while stack:
        cur = stack.pop()
        if cur == x:
            return True  # reached x without passing through the blocking set
        for nxt in np.where(A[cur, :] != 0)[0]:
            # A[cur, nxt] != 0 admits cur -> nxt and cur - nxt, and excludes
            # nxt -> cur, which is exactly a semi-directed step away from y.
            if nxt in seen or nxt in blocked:
                continue
            seen.add(nxt)
            stack.append(nxt)
    return False


def _apply_insert(x, y, T, A):
    """Apply insert operator: add x->y and orient T->y."""
    new_A = A.copy()
    new_A[x, y] = 1
    for t in T:
        new_A[y, t] = 0
        new_A[t, y] = 1
    return new_A


def _apply_delete(x, y, H, A):
    """Apply delete(x, y, H): drop the x-y edge, then orient y -> h and x -> h.

    The orientation runs *away* from y and x, not toward them. Orienting h -> y
    instead (and skipping the x - h edges entirely) yields a graph that does not
    match the operator that was scored, so the search can leave the space of
    valid PDAGs.
    """
    new_A = A.copy()
    new_A[x, y] = 0
    new_A[y, x] = 0
    n_x = neighbors(x, A)
    for h in H:
        new_A[h, y] = 0  # leaves y -> h
        if h in n_x:
            new_A[h, x] = 0  # leaves x -> h
    return new_A


def _apply_turn(x, y, C, A):
    """Apply turn(x, y, C): make the edge x -> y and orient every c in C as c -> y.

    Matches upstream ``ges.main.turn``. Clearing ``A[y, c]`` leaves ``c -> y``;
    setting ``A[c, y]`` as well would keep the edge undirected.
    """
    new_A = A.copy()
    new_A[y, x] = 0
    new_A[x, y] = 1
    for c in C:
        new_A[y, c] = 0
    return new_A


class TieredGaussObsL0Pen(GaussObsL0Pen):
    """BIC score returning −∞ for edges that violate the tier ordering.

    In the lag-embedded space, ``tier_of[node]`` equals the lag index of that
    node (0 = present, k = k steps in the past).  A parent *p* is causally
    inadmissible for child *x* when ``tier_of[p] < tier_of[x]``, i.e. the
    parent is more recent than the child it supposedly causes.
    """

    def __init__(self, data, tier_of, lmbda=None, cache=True, debug=0):
        super().__init__(data, lmbda=lmbda, cache=cache, debug=debug)
        self.tier_of = tier_of

    def _compute_local_score(self, x, pa):
        tier_x = self.tier_of[x]
        for p in pa:
            if self.tier_of[p] < tier_x:
                return -np.inf
        return super()._compute_local_score(x, pa)


def _apply_tier_orientation(cpdag, tier_of):
    """Orient undirected edges in *cpdag* using tier ordering.

    For an undirected edge X—Y where ``tier_of[X] > tier_of[Y]`` (X is
    further in the past), orient as X→Y.  Edges between same-tier nodes
    (contemporaneous) remain undirected.
    """
    A = cpdag.copy()
    p = len(A)
    for i in range(p):
        for j in range(i + 1, p):
            if A[i, j] != 0 and A[j, i] != 0:  # undirected edge
                ti, tj = tier_of[i], tier_of[j]
                if ti > tj:
                    A[j, i] = 0  # i is further past → orient i→j
                elif tj > ti:
                    A[i, j] = 0  # j is further past → orient j→i
    return A


# ---------------------------------------------------------------------------
# Time-series wrapper
# ---------------------------------------------------------------------------


def temporal_forbidden(d, max_lag):
    """Forbidden-edges matrix enforcing "past can cause present, not vice versa".

    Variable blocks in a lag embedding are ``[lag0: 0..d-1] [lag1: d..2d-1] ...
    [lagK: Kd..(K+1)d-1]``. A variable at ``lag_a`` may cause one at ``lag_b``
    only if ``lag_a >= lag_b``. Also forbids lag-0 self-loops (a variable
    cannot cause itself at the same time step). Every lag-embedded search in
    this module and in :mod:`causalts.baselines` shares this constraint.
    """
    n = d * (max_lag + 1)
    forbidden = np.zeros((n, n))
    for lag_cause in range(max_lag + 1):
        for lag_effect in range(max_lag + 1):
            if lag_cause < lag_effect:
                cause_start, effect_start = lag_cause * d, lag_effect * d
                for i in range(d):
                    for j in range(d):
                        forbidden[cause_start + i, effect_start + j] = 1
    for i in range(d):
        forbidden[i, i] = 1
    return forbidden


def lges_discovery(df, max_lag=1, mode="lges", lambda_value=None):
    """Run LGES on lag-embedded time series data.

    Parameters
    ----------
    df : pd.DataFrame or ndarray
        Time series data, shape ``(T, d)``.
    max_lag : int
        Number of lags to embed.
    mode : str
        ``"lges"`` (SafeInsert + ConservativeInsert, default),
        ``"lges-safe"`` (SafeInsert only),
        ``"ges"`` (vanilla GES).
    lambda_value : float or None
        BIC penalty. If *None*, uses ``0.5 * log(n)``.

    Returns
    -------
    G_hat : ndarray, shape ``(d, d, max_lag+1)``
        Estimated graph in ``[cause, effect, lag]`` format.
    info : dict
        ``method``, ``mode``, ``score``, ``cpdag``, ``metrics``.
    """
    data = df.values if isinstance(df, pd.DataFrame) else np.asarray(df)
    T_orig, d = data.shape

    # Lag-embed: [X_t, X_{t-1}, ..., X_{t-max_lag}]
    embedded_cols = []
    for lag in range(max_lag + 1):
        start = max_lag - lag
        end = T_orig - lag
        embedded_cols.append(data[start:end, :])
    embedded = np.hstack(embedded_cols)

    # Set up scoring
    score_class = GaussObsL0Pen(embedded, lmbda=lambda_value)
    forbidden = temporal_forbidden(d, max_lag)

    # Set LGES mode
    if mode == "lges":
        score_based, prune = True, True
    elif mode == "lges-safe":
        score_based, prune = True, False
    elif mode == "ges":
        score_based, prune = False, False
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'lges', 'lges-safe', or 'ges'.")

    # Run LGES with temporal constraints
    cpdag, metrics = fit(
        score_class, score_based=score_based, prune=prune, forbidden=forbidden
    )

    # Extract (d, d, max_lag+1) graph from CPDAG over embedded variables
    G_hat = np.zeros((d, d, max_lag + 1), dtype=int)

    for lag in range(max_lag + 1):
        cause_start = lag * d
        cause_end = (lag + 1) * d  # noqa: F841
        effect_start = 0
        effect_end = d  # noqa: F841

        for i in range(d):
            for j in range(d):
                if lag == 0 and i == j:
                    continue
                ci = cause_start + i
                ej = effect_start + j
                # Directed: ci -> ej
                if cpdag[ci, ej] != 0 and cpdag[ej, ci] == 0:
                    G_hat[i, j, lag] = 1
                # Undirected: ci - ej (treat as edge in both temporal dirs)
                elif cpdag[ci, ej] != 0 and cpdag[ej, ci] != 0:
                    G_hat[i, j, lag] = 1

    info = {
        "method": "lges",
        "mode": mode,
        "score": metrics.get("score"),
        "cpdag": cpdag,
        "metrics": metrics,
    }

    return G_hat, info


def tges_discovery(df, max_lag=1, mode="lges", lambda_value=None):
    """Run TGES on lag-embedded time series data.

    Temporal GES (Larsen et al. 2025) extends GES/LGES with tiered background
    knowledge: the BIC score returns −∞ for edges that violate the temporal
    tier ordering, and a post-processing step orients remaining undirected
    edges using tier membership.  Compared to ``lges_discovery``, the output
    graph has strictly more directed edges (fewer undirected edges remain).

    Parameters
    ----------
    df : pd.DataFrame or ndarray
        Time series data, shape ``(T, d)``.
    max_lag : int
        Number of lags to embed.
    mode : str
        ``"lges"`` (SafeInsert + ConservativeInsert, default),
        ``"lges-safe"`` (SafeInsert only),
        ``"ges"`` (vanilla GES).
    lambda_value : float or None
        BIC penalty. If *None*, uses ``0.5 * log(n)``.

    Returns
    -------
    G_hat : ndarray, shape ``(d, d, max_lag+1)``
        Estimated graph in ``[cause, effect, lag]`` format, extracted from
        the tiered CPDAG.
    info : dict
        ``method``, ``mode``, ``score``, ``cpdag`` (raw CPDAG before tier
        orientation), ``tiered_cpdag`` (after tier orientation), ``metrics``.
    """
    data = df.values if isinstance(df, pd.DataFrame) else np.asarray(df)
    T_orig, d = data.shape

    # Lag-embed: [X_t, X_{t-1}, ..., X_{t-max_lag}]
    embedded_cols = []
    for lag in range(max_lag + 1):
        start = max_lag - lag
        end = T_orig - lag
        embedded_cols.append(data[start:end, :])
    embedded = np.hstack(embedded_cols)

    # tier_of[node] = lag index of that node (0 = present, k = k steps back)
    tier_of = {lag * d + i: lag for lag in range(max_lag + 1) for i in range(d)}

    forbidden = temporal_forbidden(d, max_lag)

    # Standard BIC score — temporal constraints are enforced via the forbidden
    # matrix (same guarantee as TieredGaussObsL0Pen's -inf scoring, without
    # the numerical instability that -inf causes in the backward phase).
    score_class = GaussObsL0Pen(embedded, lmbda=lambda_value)

    if mode == "lges":
        score_based, prune = True, True
    elif mode == "lges-safe":
        score_based, prune = True, False
    elif mode == "ges":
        score_based, prune = False, False
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'lges', 'lges-safe', or 'ges'.")

    cpdag, metrics = fit(
        score_class, score_based=score_based, prune=prune, forbidden=forbidden
    )

    # Orient remaining undirected edges using tier membership
    tiered_cpdag = _apply_tier_orientation(cpdag, tier_of)

    # Extract (d, d, max_lag+1) graph from tiered CPDAG
    G_hat = np.zeros((d, d, max_lag + 1), dtype=int)
    for lag in range(max_lag + 1):
        cause_start = lag * d
        for i in range(d):
            for j in range(d):
                if lag == 0 and i == j:
                    continue
                ci = cause_start + i
                ej = j  # effects always at lag-0 block
                if tiered_cpdag[ci, ej] != 0 and tiered_cpdag[ej, ci] == 0:
                    G_hat[i, j, lag] = 1
                elif tiered_cpdag[ci, ej] != 0 and tiered_cpdag[ej, ci] != 0:
                    G_hat[i, j, lag] = 1

    info = {
        "method": "tges",
        "mode": mode,
        "score": metrics.get("score"),
        "cpdag": cpdag,
        "tiered_cpdag": tiered_cpdag,
        "metrics": metrics,
    }

    return G_hat, info
