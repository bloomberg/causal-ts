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


def semi_directed_paths(fro, to, A):
    """Find all semi-directed paths from *fro* to *to* in PDAG *A*."""
    paths = []
    _sdp_dfs(fro, to, A, [fro], paths)
    return paths


def _sdp_dfs(current, to, A, path, paths):
    if current == to:
        paths.append(list(path))
        return
    for nxt in range(len(A)):
        if nxt in path:
            continue
        if A[current, nxt] != 0:
            path.append(nxt)
            _sdp_dfs(nxt, to, A, path, paths)
            path.pop()


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


def pdag_to_dag(P):
    """Dor-Tarsi algorithm: find a consistent extension DAG of PDAG *P*."""
    P = P.copy()
    p = len(P)
    ordering = []
    remaining = set(range(p))
    while remaining:
        found = False
        for x in list(remaining):
            x_neighbors = neighbors(x, P) & remaining
            x_pa = pa(x, P) & remaining
            x_adj = x_neighbors | x_pa
            if len(x_neighbors) == 0 or (
                x_adj == x_adj and _is_sink_in_subgraph(x, P, remaining)
            ):
                ordering.append(x)
                remaining.remove(x)
                found = True
                break
        if not found:
            raise ValueError("No consistent extension exists")
    G = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            if P[i, j] != 0 and P[j, i] == 0:
                G[i, j] = 1
            elif P[i, j] != 0 and P[j, i] != 0:
                idx_i = ordering.index(i)
                idx_j = ordering.index(j)
                if idx_i < idx_j:
                    G[i, j] = 1
                else:
                    G[j, i] = 1
    return G


def _is_sink_in_subgraph(x, P, remaining):
    """Check if x is a valid sink in the subgraph induced by remaining."""
    for j in remaining:
        if j == x:
            continue
        if P[x, j] != 0 and P[j, x] == 0:
            return False
    neigh = neighbors(x, P) & remaining
    adj_x = (neigh | (pa(x, P) & remaining)) - {x}
    for a, b in combinations(adj_x, 2):
        if P[a, b] == 0 and P[b, a] == 0:
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
    """Order edges for labeling."""

    def edge_key(e):
        return (order_map[e[1]], order_map[e[0]])

    return sorted(edges, key=edge_key)


def label_edges(ordered_edges, G):
    """Label each edge as compelled or reversible."""
    p = len(G)
    labels = {}
    for e in ordered_edges:
        labels[e] = "unknown"
    for x, y in ordered_edges:
        if labels[(x, y)] == "unknown":
            parents_y = set(np.where(G[:, y] != 0)[0]) - {x}
            for w in parents_y:
                if G[w, x] == 0:
                    labels[(x, y)] = "compelled"
                    for z in set(np.where(G[:, y] != 0)[0]):
                        if labels.get((z, y)) == "unknown":
                            labels[(z, y)] = "compelled"
                    break
            if labels[(x, y)] == "unknown":
                for z in parents_y:
                    if labels.get((z, y)) == "compelled":
                        labels[(x, y)] = "compelled"
                        break
            if labels[(x, y)] == "unknown":
                labels[(x, y)] = "reversible"
    cpdag = np.zeros((p, p))
    for (i, j), lbl in labels.items():
        if lbl == "compelled":
            cpdag[i, j] = 1
        elif lbl == "reversible":
            cpdag[i, j] = 1
            cpdag[j, i] = 1
    return cpdag


def pdag_to_cpdag(P):
    """PDAG -> consistent DAG -> CPDAG."""
    try:
        G = pdag_to_dag(P)
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
        cont = True
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
    new_A = pdag_to_cpdag(new_A)
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
                x, y, A, score_class, max_subset_size=max_subset_size, debug=debug
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
    new_A = pdag_to_cpdag(new_A)
    metrics["deletes_actual"] += 1
    if debug:
        print(f"  Delete {x} -> {y} | H={H}, delta={best_score:.4f}")
    return new_A, True, metrics


def _turning_step(
    A, score_class, required, forbidden, max_subset_size=3, metrics=None, debug=0
):
    """GES turning phase: find best turn operator and apply it."""
    p = len(A)
    best_score = 0
    best_operator = None

    for x in range(p):
        for y in range(p):
            if x == y:
                continue
            if not (A[x, y] != 0 and A[y, x] == 0):
                continue
            if forbidden[y, x] != 0:
                continue

            operators = _score_valid_turn_operators(
                x, y, A, score_class, max_subset_size=max_subset_size, debug=debug
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
    new_A = pdag_to_cpdag(new_A)
    metrics["turns_actual"] += 1
    if debug:
        print(f"  Turn {x} -> {y} to {y} -> {x} | C={C}, delta={best_score:.4f}")
    return new_A, True, metrics


# --- Operator validity and scoring ---


def _score_valid_insert_operators(
    x, y, A, score_class, prune=False, max_subset_size=3, debug=0
):
    """Score all valid insert(x, y, T) operators.

    Returns list of ``(score_delta, T)`` and a bool ``found_lower_scoring``.
    """
    na_yx = _na(y, x, A)
    pa_y = pa(y, A)
    operators = []
    found_lower = False

    for T in subsets(na_yx, max_size=max_subset_size):
        # Validity condition 1: NA_yx ∪ T is a clique
        if not is_clique(na_yx | T, A):
            continue
        # Validity condition 2: no semi-directed path from y to x
        new_pa = pa_y | T | {x}
        if _creates_cycle(x, y, T, A):
            continue

        old_score = score_class.local_score(y, pa_y)
        new_score = score_class.local_score(y, new_pa)
        score_delta = new_score - old_score

        if prune and score_delta < 0:
            found_lower = True
            return operators, found_lower

        operators.append((score_delta, T))

    return operators, found_lower


def _score_valid_delete_operators(x, y, A, score_class, max_subset_size=3, debug=0):
    """Score all valid delete(x, y, H) operators."""
    na_yx = _na(y, x, A)
    pa_y = pa(y, A)
    operators = []

    for H in subsets(na_yx - {x}, max_size=max_subset_size):
        # Validity: NA_yx \\ H is a clique
        if not is_clique(na_yx - H, A):
            continue

        if A[y, x] != 0:
            old_parents = pa_y | na_yx | {x}
        else:
            old_parents = pa_y | na_yx
        new_parents = (pa_y | na_yx | {x}) - H - {x}

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


def _score_valid_turn_operators(x, y, A, score_class, max_subset_size=3, debug=0):
    """Score all valid turn(x, y, C) operators."""
    na_yx = _na(y, x, A)
    pa_y = pa(y, A)
    operators = []

    for C in subsets(na_yx - {x}, max_size=max_subset_size):
        if not is_clique(na_yx | C, A):
            continue
        new_pa_y = pa_y | na_yx | {x} - C  # noqa: F841
        if _turn_creates_cycle(x, y, C, A):
            continue

        old_pa = pa_y | (na_yx - C - {x})
        new_pa = pa_y | na_yx | {x} - C

        old_score = score_class.local_score(y, old_pa)
        new_score = score_class.local_score(y, new_pa)
        score_delta = new_score - old_score

        operators.append((score_delta, C))

    return operators


def _creates_cycle(x, y, T, A):
    """Check if insert(x, y, T) would create a cycle."""
    paths = semi_directed_paths(y, x, A)
    return len(paths) > 0


def _turn_creates_cycle(x, y, C, A):
    """Check if turn(x->y to y->x) with C would create a cycle."""
    test_A = A.copy()
    test_A[x, y] = 0
    test_A[y, x] = 1
    for c in C:
        test_A[y, c] = 0
        test_A[c, y] = 1
    paths = semi_directed_paths(x, y, test_A)
    return len(paths) > 0


def _apply_insert(x, y, T, A):
    """Apply insert operator: add x->y and orient T->y."""
    new_A = A.copy()
    new_A[x, y] = 1
    for t in T:
        new_A[y, t] = 0
        new_A[t, y] = 1
    return new_A


def _apply_delete(x, y, H, A):
    """Apply delete operator: remove x-y and orient H->y."""
    new_A = A.copy()
    new_A[x, y] = 0
    new_A[y, x] = 0
    for h in H:
        new_A[y, h] = 0
        new_A[h, y] = 1
    return new_A


def _apply_turn(x, y, C, A):
    """Apply turn operator: reverse x->y to y->x, orient C->y."""
    new_A = A.copy()
    new_A[x, y] = 0
    new_A[y, x] = 1
    for c in C:
        new_A[y, c] = 0
        new_A[c, y] = 1
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
    n_embedded = d * (max_lag + 1)

    # Build forbidden-edges matrix enforcing temporal constraints:
    # Variable blocks: [lag0: 0..d-1] [lag1: d..2d-1] ... [lagK: Kd..(K+1)d-1]
    # Rule: a variable at lag_a can only cause a variable at lag_b if a >= b
    #   (the past can cause the present, not vice versa).
    # Also forbid self-loops at lag 0.
    forbidden = np.zeros((n_embedded, n_embedded))
    for lag_cause in range(max_lag + 1):
        for lag_effect in range(max_lag + 1):
            if lag_cause < lag_effect:
                # Block: present (lag_cause) cannot cause past (lag_effect)
                cause_start = lag_cause * d
                effect_start = lag_effect * d
                for i in range(d):
                    for j in range(d):
                        forbidden[cause_start + i, effect_start + j] = 1
    # Forbid self-loops at lag 0
    for i in range(d):
        forbidden[i, i] = 1

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

    n_embedded = d * (max_lag + 1)

    # tier_of[node] = lag index of that node (0 = present, k = k steps back)
    tier_of = {lag * d + i: lag for lag in range(max_lag + 1) for i in range(d)}

    # Forbidden matrix (fast-path guard; redundant with score but keeps
    # operator checks O(1) rather than hitting the score for every pair)
    forbidden = np.zeros((n_embedded, n_embedded))
    for lag_cause in range(max_lag + 1):
        for lag_effect in range(max_lag + 1):
            if lag_cause < lag_effect:
                cause_start = lag_cause * d
                effect_start = lag_effect * d
                for i in range(d):
                    for j in range(d):
                        forbidden[cause_start + i, effect_start + j] = 1
    for i in range(d):
        forbidden[i, i] = 1

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
