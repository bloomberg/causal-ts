"""
This file has been modified by Mohammad Fesanghary on July 28 2025.
Original Package: causal-learn 1.9.0
Original License: MIT License
"""

# SPDX-License-Identifier: MIT

from __future__ import annotations

import warnings
from copy import deepcopy
from itertools import combinations

import numpy as np
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.PCUtils.Helper import sort_dict_ascending
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from causalts.cdnots.meek import (
    _build_adj_dict,
    _find_unshielded_triples_fast,
    lag_copy_allowed,
)


def validate_priority(priority: int) -> None:
    """Reject collider-conflict rules CDNOTS does not support.

    Call this at an entry point, before the skeleton search -- the check below
    in :func:`uc_sepset` only runs after it, so on its own it would let a bad
    value burn the whole search first.
    """
    if priority in (0, 1, 2):
        return
    if priority in (3, 4):
        # Upstream's strength-ordering rules. They score every conflicting
        # triple by max p(x indep z | S) over find_cond_sets(), which is the
        # full powerset of the neighbours of x and z -- 2**degree conditioning
        # sets per triple, uncapped, on top of a skeleton search that
        # deliberately stopped at the first separating set. They also need
        # cg.ci_test, which CDNOTS never wires up, so they used to fail with an
        # opaque AttributeError from inside causal-learn. Rejected explicitly
        # instead: CDNOTS+ abstains on conflicts (priority=1) rather than
        # tie-breaking them.
        raise ValueError(
            f"priority={priority} (collider strength ordering) is not "
            "supported: it scores each conflict over the full powerset of "
            "the endpoints' neighbours, which is exponential in the node "
            "degree. Use priority=1 to abstain on conflicts (CDNOTS+'s "
            "default) or priority=2 to keep the first collider "
            "(CDNOTS's default)."
        )
    raise ValueError(f"priority must be 0, 1 or 2, got {priority!r}")


def validate_majority_gamma(majority_gamma: float) -> None:
    """Reject a majority-rule ambiguity band outside ``[0, 0.5]``.

    Call this at an entry point, before the skeleton search, as for
    :func:`validate_priority`.
    """
    if not 0 <= majority_gamma <= 0.5:
        raise ValueError(f"majority_gamma must be in [0, 0.5], got {majority_gamma!r}")


def uc_sepset(
    cg: CausalGraph,
    priority: int = 2,
    background_knowledge: BackgroundKnowledge | None = None,
    num_lags=0,
    contemp_collider_rule: str | None = None,
    ci_test=None,
    alpha: float = 0.01,
    lagged_parents: dict | None = None,
    max_combinations: int = 1000,
    no_of_var_ext: int | None = None,
    lags: list | None = None,
    max_conds_py: int | None = None,
    max_conds_px: int | None = None,
    majority_mode: str = "mci",
    max_subsets_per_size: int | None = None,
    max_subset_size: int | None = None,
    majority_gamma: float = 0.5,
    parentless=(),
) -> CausalGraph:
    """Run UC_sepset to orient unshielded colliders.

    Parameters
    ----------
    cg : CausalGraph
        A CausalGraph object.
    priority : int
        Rule of resolving conflicts between unshielded colliders
        (default 2). 0: overwrite, 1: orient bi-directed (abstain --
        CDNOTS+'s default), 2: prioritize existing colliders
        (CDNOTS's default). Upstream's 3 and 4 (strength ordering)
        are **not supported** and raise ``ValueError``; they are
        exponential in the node degree. See ``maxp`` below, which
        retains them but is not reachable from CDNOTS.
    background_knowledge : BackgroundKnowledge, optional
        Artificial background knowledge.
    num_lags : int
        Number of lags in embedding.
    contemp_collider_rule : str or None
        If ``'majority'``, applies the majority rule to every unshielded
        triple X-Y-Z whose middle Y is at lag 0 (X or Z may be lagged):
        X and Z are re-tested over a family of conditioning sets (see
        ``majority_mode``). A collider is declared only if < 50% of the
        separating sets contain Y. At exactly 50% (or inside the
        ``majority_gamma`` band), or with no separating set, the triple is
        ambiguous: left unoriented and recorded in ``ambiguous_triples``.
        Requires ``ci_test`` and ``alpha``.
    ci_test : object, optional
        CI test instance (required when ``contemp_collider_rule='majority'``).
    alpha : float
        Significance level for the majority rule CI tests.
    lagged_parents : dict, optional
        Lagged parents from Phase 1 (required for majority rule
        MCI-style conditioning), keyed by lag-0 variable. A lagged
        endpoint's parents are shifted by its lag before use, as in
        :func:`~causalts.cdnots.skeleton_discovery.mci_skeleton`.
    no_of_var_ext : int, optional
        Upper bound for shifted-parent column indices: the column count
        of ``ci_test.data``. Shifted parents beyond it are dropped.
        Defaults to the graph's node count.
    lags : list of int, optional
        Lag of each slice of ``ci_test.data``, the window's own lags first
        (``[0] + lag_list``, then any deeper ones). Needed only for a gapped
        ``lag_list``, where slice position and lag differ; a shifted parent
        whose lag has no slice is dropped. Defaults to contiguous lags.
    max_conds_py, max_conds_px : int, optional
        Caps on the lagged parents of the lag-0 endpoint and of the other
        endpoint in the majority-rule conditioning sets, as in
        :func:`~causalts.cdnots.skeleton_discovery.mci_skeleton`.
    majority_mode : {"mci", "pc", "pc_sepset"}
        Conditioning sets of the majority rule. ``"mci"`` (CDNOTS+, as
        PCMCI+): subsets of the contemporaneous neighbours, each extended by
        the endpoints' shifted ``lagged_parents``. ``"pc"`` (CDNOTS, a PC
        search on the lag window): every subset of the window neighbours of
        either endpoint, lagged ones included, with nothing forced in; tested
        in increasing size, at most ``max_combinations`` per triple.
        ``"pc_sepset"``: every subset of the skeleton's separating set for
        the pair (the union of everything recorded in ``cg.sepset``) together
        with the middle node -- a local, much cheaper vote; pairs without a
        recorded separating set fall back to ``"pc"``. ``lagged_parents`` and the shift options are
        unused by both PC modes.
    max_subsets_per_size, max_subset_size : int, optional
        ``majority_mode="pc"`` only: test at most this many subsets of each
        size per endpoint, and none larger than this size. CDNOTS passes its
        skeleton budget (``max_combinations``, ``max_degree``). When a cap
        binds, neighbours in the skeleton's recorded separating sets for the
        pair are tried first, then the rest in node-index order (the skeleton
        orders by dependence strength instead), so the result can depend on
        variable order.
    majority_gamma : float, default 0.5
        Ambiguity band of the majority rule (Zhalama, Zhang & Mayer 2017),
        in ``[0, 0.5]``. With ``f`` the share of separating sets that contain
        the middle node, the triple is a collider if ``f < 0.5`` and
        ``f <= majority_gamma``, a non-collider if ``f > 0.5`` and
        ``f >= 1 - majority_gamma``, and ambiguous otherwise. ``0.5`` is the
        plain majority rule (PCMCI+'s); ``0`` orients only unanimous votes,
        as conservative PC does.
    parentless : iterable of int, optional
        Lag-0 nodes that can have no parents (the C columns). A triple with
        one of them in the middle is never a collider, so it is not
        classified: orienting it would put an arrowhead into the node. In
        ``majority_mode="mci"`` they (in any slice) are never optional subset
        members; every vote conditions on those adjacent to either endpoint.

    Returns
    -------
    cg_new : CausalGraph
        Where ``graph[j,i]=1`` and ``graph[i,j]=-1`` indicates i --> j,
        ``graph[i,j] = graph[j,i] = -1`` indicates i --- j,
        ``graph[i,j] = graph[j,i] = 1`` indicates i <-> j.
        ``cg_new.ambiguous_triples`` holds the triples the majority rule
        left ambiguous (in both orders), which :func:`meek` must not use
        as non-colliders. ``cg_new.collider_triples`` (both orders) holds
        the triples the vote declared colliders, plus triples skipped
        because both arms already pointed into the middle (unless a cycle
        redo re-voted them); colliders the redo dropped stay in it and are
        also in ``ambiguous_triples``. An unshielded triple in neither set
        was ruled a non-collider.

    Notes
    -----
    On inconsistent CI results, colliders from different triples can orient
    a directed contemporaneous cycle, each supplying one edge (PCMCI+
    outputs these cycles too). The colliders with an arrow on a cycle, or on
    a cycle among lag copies, are then re-selected as whole triples: the
    heaviest subset whose arrows, together with the other colliders' and
    those oriented before this step, leave no cycle is kept, and the rest
    become ambiguous triples. A collider weighs ``0.5 -
    f``, where ``f`` is the share of the majority rule's separating sets that
    contain the middle node; ties go to the smaller sum of the largest
    p-values among the sets that contain it. Every collider weighs the same
    without a majority vote. Only colliders in every heaviest subset are
    kept, so which ones go does not depend on the order triples are
    visited in (beyond 16 candidates a greedy search is used instead). The
    orientation is then redone without them. Should a cycle remain (one only
    among lag copies, say), its arrows from this step become conflicts.
    """

    validate_priority(priority)
    validate_majority_gamma(majority_gamma)
    if majority_mode not in ("mci", "pc", "pc_sepset"):
        raise ValueError(
            "majority_mode must be 'mci', 'pc' or 'pc_sepset', "
            f"got {majority_mode!r}"
        )

    no_of_var = cg.G.num_vars
    assert no_of_var % (num_lags + 1) == 0

    no_of_inst_var = no_of_var // (num_lags + 1)
    if no_of_var_ext is None:
        no_of_var_ext = no_of_var
    if lags is None or list(lags) == list(range(len(lags))):
        slice_lag = None  # slice position == lag
    else:
        slice_lag = list(lags)
        lag_slice = {lag: s for s, lag in enumerate(slice_lag)}

    cg_new = deepcopy(cg)
    ambiguous = set()

    R0 = []  # Records of possible orientations
    evidence = {}  # Full-vote collider evidence, filled on demand
    UC_dict = {}
    adj = _build_adj_dict(cg_new)
    parentless = frozenset(parentless)
    UT = [
        (i, j, k)
        for (i, j, k) in _find_unshielded_triples_fast(cg_new, adj)
        if (i < k) and (j < no_of_inst_var) and j not in parentless
    ]  # Not considering symmetric triples

    # C - X pairs removed by background knowledge (see c_lag_cnst).
    knowledge_removed = getattr(cg_new, "knowledge_removed", ())

    def naive_add_edge(x, y):
        edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
        if edge1 is not None:
            cg_new.G.remove_edge(edge1)

        edge2 = cg_new.G.get_edge(cg_new.G.nodes[y], cg_new.G.nodes[x])
        if edge2 is not None:
            cg_new.G.remove_edge(edge2)
        cg_new.G.add_edge(
            Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW)
        )

    def add_edge(x, y):
        # Assumption: x < no_of_inst_var
        naive_add_edge(x, y)

        if num_lags > 0:
            x_time_step = x // no_of_inst_var
            y_time_step = y // no_of_inst_var

            for i in range(num_lags - max(x_time_step, y_time_step)):
                cx = x + (i + 1) * no_of_inst_var
                cy = y + (i + 1) * no_of_inst_var
                if lag_copy_allowed(cg_new, cx, cy, background_knowledge):
                    naive_add_edge(cx, cy)

    def naive_orient_or_bidirect(x, y):
        """priority=1 handling for one pair: orient x --> y if the edge is
        still undirected, or mark it x <-> y if y --> x is already there
        (i.e. this collider conflicts with an earlier orientation)."""
        edge = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
        if edge is None:
            naive_add_edge(x, y)
            return
        if (
            cg_new.G.graph[x, y] == Endpoint.TAIL.value
            and cg_new.G.graph[y, x] == Endpoint.TAIL.value
        ):
            cg_new.G.remove_edge(edge)
            cg_new.G.add_edge(
                Edge(
                    cg_new.G.nodes[x],
                    cg_new.G.nodes[y],
                    Endpoint.TAIL,
                    Endpoint.ARROW,
                )
            )
        elif (
            cg_new.G.graph[x, y] == Endpoint.ARROW.value
            and cg_new.G.graph[y, x] == Endpoint.TAIL.value
        ):
            cg_new.G.remove_edge(edge)
            cg_new.G.add_edge(
                Edge(
                    cg_new.G.nodes[x],
                    cg_new.G.nodes[y],
                    Endpoint.ARROW,
                    Endpoint.ARROW,
                )
            )

    def orient_or_bidirect(x, y):
        """Lag-aware wrapper: apply the priority=1 transform to the pair and
        to every time-shifted copy, mirroring :func:`add_edge`. Each copy is
        re-evaluated independently because their orientation states differ."""
        naive_orient_or_bidirect(x, y)

        if num_lags > 0:
            x_time_step = x // no_of_inst_var
            y_time_step = y // no_of_inst_var

            for i in range(num_lags - max(x_time_step, y_time_step)):
                cx = x + (i + 1) * no_of_inst_var
                cy = y + (i + 1) * no_of_inst_var
                if lag_copy_allowed(cg_new, cx, cy, background_knowledge):
                    naive_orient_or_bidirect(cx, cy)

    def get_sepset(x, y):
        sepset = cg.sepset[x, y]
        if sepset is None:
            return ()
        return sepset

    def _is_contemp_triple(x, y, z):
        """True if the middle node y is in the instantaneous (lag-0) slice.
        Matches PCMCI+'s majority rule which applies whenever y is contemporaneous,
        regardless of whether x or z are lagged."""
        return y < no_of_inst_var

    def _majority_check(x, y, z, full=False):
        """Majority rule for triples with contemporaneous middle y.

        The PC modes vote over ``_pc_subsets`` or over subsets of the
        skeleton's separating set plus y (see ``majority_mode``). The default
        ``"mci"`` matches PCMCI+'s _pcalg_colliders with
        contemp_collider_rule='majority':
        - Tests all subsets of contemporaneous neighbors of j (the lag-0 endpoint)
          and, when x is also at lag 0, subsets of x's contemporaneous neighbors,
          C columns excepted.
        - For each subset S, runs CI test conditioned on S + lagged parents (MCI)
          + both endpoints' adjacent C columns.
        - A collider is declared only if fraction of separating subsets containing
          Y is < 0.5 (and <= ``majority_gamma``). Fraction = 0.5 (or inside the
          ``majority_gamma`` band) or no separating set → ambiguous (not collider).
        """
        from itertools import combinations as _combs
        from itertools import islice as _islice

        if majority_mode == "pc_sepset":
            sep = cg.sepset[x, z] if cg.sepset is not None else None
            if sep:
                # The skeleton appends an entry for every depth it tested the
                # pair at: () where nothing separated them, else the union of
                # the separating sets found. Their union is the separating
                # information; the placeholders add nothing.
                found = {int(v) for entry in sep for v in entry}
                cand = sorted((found - {x, z}) | {y})
                subsets = [
                    list(S)
                    for S in _islice(
                        (
                            S
                            for card in range(len(cand) + 1)
                            for S in _combs(cand, card)
                        ),
                        max_combinations,
                    )
                ]
                return _vote(x, y, z, subsets, [], [], full)
        if majority_mode in ("pc", "pc_sepset"):
            return _vote(x, y, z, _pc_subsets(x, z), [], [], full)

        # j is always the lag-0 endpoint (y is at lag 0 by _is_contemp_triple)
        # x and z may be lagged. Identify the lag-0 side (j) and the other (i).
        # PCMCI+ builds subsets from neighbors of j and (if i is also lag-0) of i.
        j = z if z < no_of_inst_var else x  # the contemporaneous endpoint
        i = x if z < no_of_inst_var else z  # could be lagged

        # Contemporaneous neighbors of j, excluding only the direct test partner i.
        # y is NOT excluded — PCMCI+ includes y in conditioning subsets. C
        # columns are not subset candidates: every test conditions on both
        # endpoints' C parents through the base sets below. This goes further
        # than MCI (phase 2), which adds them for the effect only; C is a root,
        # so conditioning on it can only block paths. As an optional subset
        # member, the votes without C would test an endpoint whose regime
        # shift swamps the dependence, so spurious separating sets would
        # decide the triple.
        def _is_c(p):
            return p % no_of_inst_var in parentless

        contemp_neighbors_j = [
            p
            for p in cg_new.neighbors(j)
            if p < no_of_inst_var
            and p != (i if i < no_of_inst_var else -1)
            and not _is_c(p)
        ]
        # If i is also lag-0, include its contemp neighbors too (excluding j)
        contemp_neighbors_i = []
        if i < no_of_inst_var:
            contemp_neighbors_i = [
                p
                for p in cg_new.neighbors(i)
                if p < no_of_inst_var and p != j and not _is_c(p)
            ]

        # Build all unique subsets (union of both sides)
        all_subsets = []
        for card in range(len(contemp_neighbors_j) + 1):
            for S in _combs(contemp_neighbors_j, card):
                if list(S) not in all_subsets:
                    all_subsets.append(list(S))
        for card in range(len(contemp_neighbors_i) + 1):
            for S in _combs(contemp_neighbors_i, card):
                if list(S) not in all_subsets:
                    all_subsets.append(list(S))

        if not all_subsets:
            all_subsets = [[]]

        # Lagged base conditions from Phase 1 parents (MCI-style). They are
        # keyed by the lag-0 variable, so a lagged endpoint's parents must be
        # shifted by its lag; unshifted, an autocorrelated endpoint lands in
        # its own conditioning set.
        def _shifted_parents(node, cap):
            out = []
            for p in (lagged_parents or {}).get(node % no_of_inst_var, []):
                if slice_lag is None:
                    s = node // no_of_inst_var + p // no_of_inst_var
                else:
                    lag = slice_lag[node // no_of_inst_var]
                    s = lag_slice.get(lag + slice_lag[p // no_of_inst_var])
                    if s is None:
                        continue
                idx = p % no_of_inst_var + s * no_of_inst_var
                if idx < no_of_var_ext:
                    out.append(idx)
            return out if cap is None else out[:cap]

        def _c_parents(node):
            # C can only be a cause, so an adjacent C column is a parent.
            return [p for p in cg_new.neighbors(node) if _is_c(p)]

        base_j = _shifted_parents(j, max_conds_py) + _c_parents(j)
        base_i = _shifted_parents(i, max_conds_px) + _c_parents(i)
        return _vote(x, y, z, all_subsets, base_j, base_i, full)

    def _pc_subsets(x, z):
        """Subsets of adj(x) - {z} and of adj(z) - {x} in the window graph,
        smallest first, de-duplicated, at most ``max_combinations``.

        Per size, each endpoint contributes at most ``max_subsets_per_size``
        subsets, as each side does in the skeleton search. Neighbours in the
        skeleton's recorded separating sets for the pair come first, so the
        set that removed the edge is in the vote whenever it fits the cap.
        """
        from itertools import combinations as _combs
        from itertools import islice as _islice
        from itertools import zip_longest as _zip_longest

        sep = cg.sepset[x, z] if cg.sepset is not None else None
        sep_nodes = {int(v) for entry in (sep or ()) for v in entry}

        def _order(nbrs):
            return sorted(nbrs, key=lambda p: (p not in sep_nodes, p))

        adj_x = _order(p for p in cg_new.neighbors(x) if p != z)
        adj_z = _order(p for p in cg_new.neighbors(z) if p != x)
        top = max(len(adj_x), len(adj_z))
        if max_subset_size is not None:
            top = min(top, max_subset_size)
        seen, out = set(), []
        for card in range(top + 1):
            sides = [
                _islice(_combs(adj, card), max_subsets_per_size)
                for adj in (adj_x, adj_z)
            ]
            for pair in _zip_longest(*sides):
                for S in pair:
                    if S is None or S in seen:
                        continue
                    seen.add(S)
                    out.append(list(S))
                    if len(out) >= max_combinations:
                        return out
        return out

    def _band(n_y, n):
        """Majority-rule class of ``n_y`` of ``n`` separating sets holding y:
        1 collider, -1 non-collider, 0 ambiguous."""
        if 2 * n_y < n and n_y <= majority_gamma * n + 1e-9:
            return 1
        if 2 * n_y > n and n_y >= (1 - majority_gamma) * n - 1e-9:
            return -1
        return 0

    def _vote(x, y, z, all_subsets, base_j, base_i, full=False):
        """Majority vote over ``all_subsets``, each extended by the bases.

        Stops as soon as the untested subsets can no longer change the
        outcome, so the decision is the same as testing every subset.
        ``full`` tests every subset and records the evidence in ``evidence``.
        """
        subsets = all_subsets[:max_combinations]
        # Untested subsets with / without y, and separating ones so far.
        rest_y = sum(y in S for S in subsets)
        rest_n = len(subsets) - rest_y
        sep_y = sep_n = 0
        max_p_y = 0.0
        separating_subsets = []
        for S in subsets:
            Z = list(S)
            Z += [p for p in base_j if p not in Z]
            Z += [p for p in base_i if p not in Z]
            # Phase-1 parents predate MCI pruning, so the tested pair can
            # still be listed there; PCMCI+ drops the tested nodes likewise.
            Z = [p for p in Z if p != x and p != z]
            has_y = y in S
            rest_y -= has_y
            rest_n -= not has_y
            p_val, _ = ci_test(x, z, Z)
            if has_y:
                max_p_y = max(max_p_y, p_val)
            if p_val > alpha:
                separating_subsets.append(S)
                sep_y += has_y
                sep_n += not has_y
            if full:
                continue
            # The share can still rise to (sep_y + rest_y) / (... + sep_n)
            # and fall to sep_y / (sep_y + sep_n + rest_n).
            if sep_n and _band(sep_y + rest_y, sep_y + rest_y + sep_n) == 1:
                return True  # a collider, whatever remains
            if sep_y and _band(sep_y, sep_y + sep_n + rest_n) == -1:
                return False  # a non-collider, whatever remains

        if len(separating_subsets) == 0:
            # no separating set found → ambiguous, do NOT orient
            ambiguous.update({(x, y, z), (z, y, x)})
            return False

        n_y = sum(y in S for S in separating_subsets)
        fraction = n_y / len(separating_subsets)
        if full:
            evidence[(x, y, z)] = (fraction, max_p_y)
        # y absent from the majority → collider; present → non-collider;
        # a tie (or inside the majority_gamma band) → ambiguous, do NOT orient
        verdict = _band(n_y, len(separating_subsets))
        if verdict == 0:
            ambiguous.update({(x, y, z), (z, y, x)})
        return verdict == 1

    def _vote_evidence(t):
        """Full-vote evidence of collider triple ``t`` (None if it had no
        majority vote), computed once."""
        if contemp_collider_rule != "majority" or ci_test is None:
            return None
        if t not in evidence:
            # Evidence only: a CI test whose p-values vary between calls must
            # not change the recorded decisions.
            saved = set(ambiguous)
            _majority_check(*t, full=True)
            ambiguous.clear()
            ambiguous.update(saved)
        return evidence.get(t)

    def _classify(x, y, z):
        """Whether the unshielded triple x - y - z is a collider."""
        # For contemporaneous triples with majority rule: use fraction-based check
        if (
            contemp_collider_rule == "majority"
            and _is_contemp_triple(x, y, z)
            and ci_test is not None
        ):
            return _majority_check(x, y, z)
        if (x, z) in knowledge_removed:
            # Removed by background knowledge, not by a CI test: the empty
            # sepset is no evidence either way, so leave the triple ambiguous.
            ambiguous.update({(x, y, z), (z, y, x)})
            return False
        # Standard check: collider if y not in any separating set
        return all(y not in S for S in get_sepset(x, z))

    def _orient_collider(x, y, z):
        if priority == 0:  # 0: overwrite
            # Fully orient the edge irrespective of what have been oriented
            add_edge(x, y)
            add_edge(z, y)

        elif priority == 1:  # 1: orient bi-directed
            # Conflicting collider orientations become x <-> y
            # (the analogue of PCMCI+'s 'x-x' conflict marker),
            # propagated across time-shifted copies.
            orient_or_bidirect(x, y)
            orient_or_bidirect(z, y)

        elif priority == 2:  # 2: prioritize existing
            if (not cg_new.is_fully_directed(y, x)) and (
                not cg_new.is_fully_directed(y, z)
            ):
                edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
                if edge1 is not None:
                    cg_new.G.remove_edge(edge1)
                # Orient if edges are not already oriented oppositely
                add_edge(x, y)
                add_edge(z, y)
        else:
            R0.append((x, y, z))

    def _resolve_collider_cycles(colliders, skipped):
        """Re-select the colliders behind contemporaneous cycles (see Notes)."""
        nonlocal cg_new
        n = no_of_inst_var
        size = cg.G.graph.shape[0]
        # Triples skipped because both arms already pointed into the middle
        # depend on visiting order; classify them too so the candidates do
        # not. This also records those the majority rule leaves ambiguous:
        # after the redo their arms may no longer both point into the middle,
        # and meek must not treat them as non-colliders.
        # In place, so the caller's record of declared colliders has them too.
        colliders.extend(t for t in skipped if _classify(*t))
        # Arrows oriented before this step, in the whole window: lag copies
        # are oriented separately, so a cycle can sit in a lagged slice only.
        # A cycle among them is not the colliders' doing: the selection
        # ignores its arrows (after opposite arrows cancel, so a collider
        # against one still makes a conflict), and it is warned about below.
        # Colliders can re-assert those arrows in any slice, so the selection
        # sees them at lag 0.
        fixed_all = _arrows(cg.G.graph)
        fixed_cycle = _cycle_edges(fixed_all, size)
        fixed = {(a, b) for a, b in fixed_all if a < n and b < n}
        ignore = {
            (i - j // n * n, j - j // n * n) for i, j in fixed_cycle if i // n == j // n
        }

        voted = contemp_collider_rule == "majority" and ci_test is not None

        def weight(t):
            if not voted:
                return (0.5, 0.0)  # no vote: every collider weighs the same
            ev = _vote_evidence(t)
            if ev is None:
                # The full re-vote found no separating set (a CI test whose
                # p-values vary between calls): no support, so weakest.
                return (-0.5, -1.0)
            fraction, max_p_y = ev
            return (round(0.5 - fraction, 9), -min(max_p_y, 1.0))

        rejected = set()
        for _ in range(len(colliders)):
            cyc_all = _cycle_edges(_arrows(cg_new.G.graph), size)
            if not cyc_all:
                return
            cyc_all -= fixed_cycle
            # Shift each cycle arrow back to the slice of its head, where the
            # collider asserting it (middle at lag 0) is found.
            cyc = {(i - j // n * n, j - j // n * n) for i, j in cyc_all}
            live = [t for t in colliders if t not in rejected]
            implicated = [
                t for t in live if any((a, t[1]) in cyc for a in (t[0], t[2]))
            ]
            others = set(fixed)
            for t in live:
                if t not in implicated:
                    others |= _collider_arrows(t, n)
            keep = _select_colliders(
                implicated,
                {t: weight(t) for t in implicated},
                others,
                n,
                ignore=frozenset(ignore),
            )
            dropped = set(implicated) - keep
            if not dropped:
                break
            rejected |= dropped
            for x, y, z in dropped:
                ambiguous.update({(x, y, z), (z, y, x)})
            cg_new = deepcopy(cg)
            kept = set(colliders) - rejected
            for t in UT:
                if t in kept:
                    _orient_collider(*t)
        # Reached when no selection of lag-0 colliders removes the cycle: a
        # cycle only in a lag copy, priority 0 or 2 resolving opposite arrows
        # differently from the selection, or a cycle oriented before this
        # step. Mark the arrows on a cycle this step oriented as conflicts.
        for i, j in sorted(_cycle_edges(_arrows(cg_new.G.graph), size) - fixed_all):
            cg_new.G.remove_edge(
                cg_new.G.get_edge(cg_new.G.nodes[i], cg_new.G.nodes[j])
            )
            cg_new.G.add_edge(
                Edge(
                    cg_new.G.nodes[i], cg_new.G.nodes[j], Endpoint.ARROW, Endpoint.ARROW
                )
            )
        if _cycle_edges(_arrows(cg_new.G.graph), size):
            warnings.warn(
                "uc_sepset: the graph had a directed cycle before the collider "
                "step; it is left as it is",
                stacklevel=3,
            )

    colliders, skipped = [], []
    for x, y, z in UT:
        if (background_knowledge is not None) and (
            background_knowledge.is_forbidden(cg_new.G.nodes[x], cg_new.G.nodes[y])
            or background_knowledge.is_forbidden(cg_new.G.nodes[z], cg_new.G.nodes[y])
            or background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[x])
            or background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[z])
        ):
            continue
        if cg_new.is_fully_directed(x, y) and cg_new.is_fully_directed(z, y):
            skipped.append((x, y, z))
            continue
        if _classify(x, y, z):
            colliders.append((x, y, z))
            _orient_collider(x, y, z)

    resolved = priority in (0, 1, 2) and _cycle_edges(
        _arrows(cg_new.G.graph), cg_new.G.graph.shape[0]
    )
    if resolved:
        _resolve_collider_cycles(colliders, skipped)
    cg_new.ambiguous_triples = ambiguous
    # Triples the vote declared colliders, in both orders; with
    # ambiguous_triples this tells later steps which unshielded triples it
    # ruled non-colliders. Skipped triples are voted on only in the cycle
    # redo; otherwise they are colliders in the graph with no ruling.
    declared = colliders if resolved else colliders + skipped
    cg_new.collider_triples = {
        t for x, y, z in declared for t in ((x, y, z), (z, y, x))
    }
    if priority in [0, 1, 2]:
        return cg_new

    else:
        if priority == 3:  # 3. Order colliders by p_{xz|y} in ascending order
            for x, y, z in R0:
                cond = cg_new.find_cond_sets_with_mid(x, z, y)
                UC_dict[(x, y, z)] = max([cg_new.ci_test(x, z, S) for S in cond])
            UC_dict = sort_dict_ascending(UC_dict)

        else:  # 4. Order colliders by p_{xy|not y} in descending order
            for x, y, z in R0:
                cond = cg_new.find_cond_sets_without_mid(x, z, y)
                UC_dict[(x, y, z)] = max([cg_new.ci_test(x, z, S) for S in cond])
            UC_dict = sort_dict_ascending(UC_dict, descending=True)

        for x, y, z in UC_dict.keys():
            if (background_knowledge is not None) and (
                background_knowledge.is_forbidden(cg_new.G.nodes[x], cg_new.G.nodes[y])
                or background_knowledge.is_forbidden(
                    cg_new.G.nodes[z], cg_new.G.nodes[y]
                )
                or background_knowledge.is_required(
                    cg_new.G.nodes[y], cg_new.G.nodes[x]
                )
                or background_knowledge.is_required(
                    cg_new.G.nodes[y], cg_new.G.nodes[z]
                )
            ):
                continue
            if (not cg_new.is_fully_directed(y, x)) and (
                not cg_new.is_fully_directed(y, z)
            ):
                # Orient only if the edges have not been oriented the other way around
                add_edge(x, y)
                add_edge(z, y)

        return cg_new


def _arrows(graph):
    """Directed arrows ``(i, j)`` (i --> j) of an endpoint matrix."""
    directed = (graph == Endpoint.TAIL.value) & (graph.T == Endpoint.ARROW.value)
    return {(int(a), int(b)) for a, b in zip(*np.nonzero(directed))}


def _cycle_edges(arrows, n):
    """The arrows (among nodes ``0..n-1``) lying on a directed cycle: those
    inside a strongly connected component of more than one node."""
    if not arrows:
        return set()
    m = np.zeros((n, n), dtype=bool)
    for a, b in arrows:
        m[a, b] = True
    _, label = connected_components(csr_matrix(m), directed=True, connection="strong")
    return {(a, b) for a, b in arrows if label[a] == label[b]}


def _collider_arrows(t, n):
    """Lag-0 arrows collider ``x --> y <-- z`` asserts (lagged arms never
    close a lag-0 cycle)."""
    x, y, z = t
    return {(a, y) for a in (x, z) if a < n}


def _select_colliders(
    candidates, weights, others, n, exact_limit=16, ignore=frozenset()
):
    """Colliders kept from ``candidates`` when breaking lag-0 cycles.

    Finds the heaviest subsets (by summed ``weights``, tuples compared in
    order) whose arrows, with the arrows in ``others``, contain no directed
    cycle, and returns the colliders in all of them. Opposite arrows on one
    pair cancel out, as ``priority=1`` marks them a conflict. Arrows in
    ``ignore`` (a cycle oriented before the collider step) never count,
    whichever collider asserts them. Beyond ``exact_limit`` candidates the
    search is greedy, heaviest first.
    """

    def acyclic(subset):
        arrows = set(others)
        for t in subset:
            arrows |= _collider_arrows(t, n)
        arrows = {(a, b) for a, b in arrows if (b, a) not in arrows} - ignore
        return not _cycle_edges(arrows, n)

    def total(subset):
        return tuple(round(sum(c), 9) for c in zip(*(weights[t] for t in subset)))

    cands = sorted(candidates)
    if len(cands) > exact_limit:
        keep = []
        for t in sorted(cands, key=lambda t: weights[t], reverse=True):
            if acyclic(keep + [t]):
                keep.append(t)
        return set(keep)
    zero = (0.0,) * len(next(iter(weights.values()), ()))
    subsets = sorted(
        (
            (total(sub) if sub else zero, sub)
            for r in range(len(cands) + 1)
            for sub in combinations(cands, r)
        ),
        key=lambda ts: ts[0],
        reverse=True,
    )
    best, keep = None, None
    for tot, sub in subsets:
        if best is not None and tot < best:
            break
        if acyclic(sub):
            best = tot
            keep = set(sub) if keep is None else keep & set(sub)
    return keep or set()


def maxp(
    cg: CausalGraph, priority: int = 3, background_knowledge: BackgroundKnowledge = None
):
    """Run MaxP to orient unshielded colliders.

    Parameters
    ----------
    cg : CausalGraph
        A CausalGraph object.
    priority : int
        Rule of resolving conflicts between unshielded colliders
        (default 3). 0: overwrite, 1: orient bi-directed,
        2: prioritize existing colliders, 3: prioritize stronger
        colliders, 4: prioritize stronger* colliders.
    background_knowledge : BackgroundKnowledge, optional
        Artificial background knowledge.

    Returns
    -------
    cg_new : CausalGraph
        Where ``graph[j,i]=1`` and ``graph[i,j]=-1`` indicates i --> j,
        ``graph[i,j] = graph[j,i] = -1`` indicates i --- j,
        ``graph[i,j] = graph[j,i] = 1`` indicates i <-> j.
    """

    assert priority in [0, 1, 2, 3, 4]

    cg_new = deepcopy(cg)
    UC_dict = {}
    UT = [
        (i, j, k) for (i, j, k) in cg_new.find_unshielded_triples() if i < k
    ]  # Not considering symmetric triples

    for x, y, z in UT:
        if (background_knowledge is not None) and (
            background_knowledge.is_forbidden(cg_new.G.nodes[x], cg_new.G.nodes[y])
            or background_knowledge.is_forbidden(cg_new.G.nodes[z], cg_new.G.nodes[y])
            or background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[x])
            or background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[z])
        ):
            continue

        cond_with_y = cg_new.find_cond_sets_with_mid(x, z, y)
        cond_without_y = cg_new.find_cond_sets_without_mid(x, z, y)

        max_p_contain_y = max([cg_new.ci_test(x, z, S) for S in cond_with_y])
        max_p_not_contain_y = max([cg_new.ci_test(x, z, S) for S in cond_without_y])

        if max_p_not_contain_y > max_p_contain_y:
            if priority == 0:  # 0: overwrite
                edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
                if edge1 is not None:
                    cg_new.G.remove_edge(edge1)
                edge2 = cg_new.G.get_edge(cg_new.G.nodes[y], cg_new.G.nodes[x])
                if edge2 is not None:
                    cg_new.G.remove_edge(edge2)
                # Fully orient the edge irrespective of what have been oriented
                cg_new.G.add_edge(
                    Edge(
                        cg_new.G.nodes[x],
                        cg_new.G.nodes[y],
                        Endpoint.TAIL,
                        Endpoint.ARROW,
                    )
                )

                edge3 = cg_new.G.get_edge(cg_new.G.nodes[y], cg_new.G.nodes[z])
                if edge3 is not None:
                    cg_new.G.remove_edge(edge3)
                edge4 = cg_new.G.get_edge(cg_new.G.nodes[z], cg_new.G.nodes[y])
                if edge4 is not None:
                    cg_new.G.remove_edge(edge4)
                cg_new.G.add_edge(
                    Edge(
                        cg_new.G.nodes[z],
                        cg_new.G.nodes[y],
                        Endpoint.TAIL,
                        Endpoint.ARROW,
                    )
                )

            elif priority == 1:  # 1: orient bi-directed
                edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
                if edge1 is not None:
                    if (
                        cg_new.G.graph[x, y] == Endpoint.TAIL.value
                        and cg_new.G.graph[y, x] == Endpoint.TAIL.value
                    ):
                        cg_new.G.remove_edge(edge1)
                        cg_new.G.add_edge(
                            Edge(
                                cg_new.G.nodes[x],
                                cg_new.G.nodes[y],
                                Endpoint.TAIL,
                                Endpoint.ARROW,
                            )
                        )
                    elif (
                        cg_new.G.graph[x, y] == Endpoint.ARROW.value
                        and cg_new.G.graph[y, x] == Endpoint.TAIL.value
                    ):
                        cg_new.G.remove_edge(edge1)
                        cg_new.G.add_edge(
                            Edge(
                                cg_new.G.nodes[x],
                                cg_new.G.nodes[y],
                                Endpoint.ARROW,
                                Endpoint.ARROW,
                            )
                        )
                else:
                    cg_new.G.add_edge(
                        Edge(
                            cg_new.G.nodes[x],
                            cg_new.G.nodes[y],
                            Endpoint.TAIL,
                            Endpoint.ARROW,
                        )
                    )

                edge2 = cg_new.G.get_edge(cg_new.G.nodes[z], cg_new.G.nodes[y])
                if edge2 is not None:
                    if (
                        cg_new.G.graph[z, y] == Endpoint.TAIL.value
                        and cg_new.G.graph[y, z] == Endpoint.TAIL.value
                    ):
                        cg_new.G.remove_edge(edge2)
                        cg_new.G.add_edge(
                            Edge(
                                cg_new.G.nodes[z],
                                cg_new.G.nodes[y],
                                Endpoint.TAIL,
                                Endpoint.ARROW,
                            )
                        )
                    elif (
                        cg_new.G.graph[z, y] == Endpoint.ARROW.value
                        and cg_new.G.graph[y, z] == Endpoint.TAIL.value
                    ):
                        cg_new.G.remove_edge(edge2)
                        cg_new.G.add_edge(
                            Edge(
                                cg_new.G.nodes[z],
                                cg_new.G.nodes[y],
                                Endpoint.ARROW,
                                Endpoint.ARROW,
                            )
                        )
                else:
                    cg_new.G.add_edge(
                        Edge(
                            cg_new.G.nodes[z],
                            cg_new.G.nodes[y],
                            Endpoint.TAIL,
                            Endpoint.ARROW,
                        )
                    )

            elif priority == 2:  # 2: prioritize existing
                if (not cg_new.is_fully_directed(y, x)) and (
                    not cg_new.is_fully_directed(y, z)
                ):
                    edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
                    if edge1 is not None:
                        cg_new.G.remove_edge(edge1)
                    # Ensure edges are not oriented oppositely before proceeding
                    cg_new.G.add_edge(
                        Edge(
                            cg_new.G.nodes[x],
                            cg_new.G.nodes[y],
                            Endpoint.TAIL,
                            Endpoint.ARROW,
                        )
                    )

                    edge2 = cg_new.G.get_edge(cg_new.G.nodes[z], cg_new.G.nodes[y])
                    if edge2 is not None:
                        cg_new.G.remove_edge(edge2)
                    cg_new.G.add_edge(
                        Edge(
                            cg_new.G.nodes[z],
                            cg_new.G.nodes[y],
                            Endpoint.TAIL,
                            Endpoint.ARROW,
                        )
                    )

            elif priority == 3:
                UC_dict[(x, y, z)] = max_p_contain_y

            elif priority == 4:
                UC_dict[(x, y, z)] = max_p_not_contain_y

    if priority in [0, 1, 2]:
        return cg_new

    else:
        if priority == 3:  # 3. Order colliders by p_{xz|y} in ascending order
            UC_dict = sort_dict_ascending(UC_dict)
        else:  # 4. Order colliders by p_{xz|not y} in descending order
            UC_dict = sort_dict_ascending(UC_dict, True)

        for x, y, z in UC_dict.keys():
            if (background_knowledge is not None) and (
                background_knowledge.is_forbidden(cg_new.G.nodes[x], cg_new.G.nodes[y])
                or background_knowledge.is_forbidden(
                    cg_new.G.nodes[z], cg_new.G.nodes[y]
                )
                or background_knowledge.is_required(
                    cg_new.G.nodes[y], cg_new.G.nodes[x]
                )
                or background_knowledge.is_required(
                    cg_new.G.nodes[y], cg_new.G.nodes[z]
                )
            ):
                continue

            if (not cg_new.is_fully_directed(y, x)) and (
                not cg_new.is_fully_directed(y, z)
            ):
                edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
                if edge1 is not None:
                    cg_new.G.remove_edge(edge1)
                # Orient only if the edges have not been oriented the other way around
                cg_new.G.add_edge(
                    Edge(
                        cg_new.G.nodes[x],
                        cg_new.G.nodes[y],
                        Endpoint.TAIL,
                        Endpoint.ARROW,
                    )
                )

                edge2 = cg_new.G.get_edge(cg_new.G.nodes[z], cg_new.G.nodes[y])
                if edge2 is not None:
                    cg_new.G.remove_edge(edge2)
                cg_new.G.add_edge(
                    Edge(
                        cg_new.G.nodes[z],
                        cg_new.G.nodes[y],
                        Endpoint.TAIL,
                        Endpoint.ARROW,
                    )
                )

        return cg_new


def definite_maxp(
    cg: CausalGraph,
    alpha: float,
    priority: int = 4,
    background_knowledge: BackgroundKnowledge = None,
) -> CausalGraph:
    """Run Definite_MaxP to orient unshielded colliders.

    Parameters
    ----------
    cg : CausalGraph
        A CausalGraph object.
    alpha : float
        Significance level for definite collider detection.
    priority : int
        Rule of resolving conflicts between unshielded colliders
        (default 4). 2: prioritize existing colliders, 3: prioritize
        stronger colliders, 4: prioritize stronger* colliders.
    background_knowledge : BackgroundKnowledge, optional
        Artificial background knowledge.

    Returns
    -------
    cg_new : CausalGraph
        Where ``graph[j,i]=1`` and ``graph[i,j]=-1`` indicates i --> j,
        ``graph[i,j] = graph[j,i] = -1`` indicates i --- j,
        ``graph[i,j] = graph[j,i] = 1`` indicates i <-> j.
    """

    assert 1 > alpha >= 0
    assert priority in [2, 3, 4]

    cg_new = deepcopy(cg)
    UC_dict = {}
    UT = [
        (i, j, k) for (i, j, k) in cg_new.find_unshielded_triples() if i < k
    ]  # Not considering symmetric triples

    for x, y, z in UT:
        cond_with_y = cg_new.find_cond_sets_with_mid(x, z, y)
        cond_without_y = cg_new.find_cond_sets_without_mid(x, z, y)
        max_p_contain_y = 0
        max_p_not_contain_y = 0
        uc_bool = True
        nuc_bool = True

        for S in cond_with_y:
            p = cg_new.ci_test(x, z, S)
            if p > alpha:
                uc_bool = False
                break
            elif p > max_p_contain_y:
                max_p_contain_y = p

        for S in cond_without_y:
            p = cg_new.ci_test(x, z, S)
            if p > alpha:
                nuc_bool = False
                if not uc_bool:
                    break  # ambiguous triple
            if p > max_p_not_contain_y:
                max_p_not_contain_y = p

        if uc_bool:
            if nuc_bool:
                if max_p_not_contain_y > max_p_contain_y:
                    if priority in [2, 3]:
                        UC_dict[(x, y, z)] = max_p_contain_y
                    if priority == 4:
                        UC_dict[(x, y, z)] = max_p_not_contain_y
                else:
                    cg_new.definite_non_UC.append((x, y, z))
            else:
                if priority in [2, 3]:
                    UC_dict[(x, y, z)] = max_p_contain_y
                if priority == 4:
                    UC_dict[(x, y, z)] = max_p_not_contain_y

        elif nuc_bool:
            cg_new.definite_non_UC.append((x, y, z))

    if priority == 3:  # 3. Order colliders by p_{xz|y} in ascending order
        UC_dict = sort_dict_ascending(UC_dict)
    elif priority == 4:  # 4. Order colliders by p_{xz|not y} in descending order
        UC_dict = sort_dict_ascending(UC_dict, True)

    for x, y, z in UC_dict.keys():
        if (background_knowledge is not None) and (
            background_knowledge.is_forbidden(cg_new.G.nodes[x], cg_new.G.nodes[y])
            or background_knowledge.is_forbidden(cg_new.G.nodes[z], cg_new.G.nodes[y])
            or background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[x])
            or background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[z])
        ):
            continue

        if (not cg_new.is_fully_directed(y, x)) and (
            not cg_new.is_fully_directed(y, z)
        ):
            edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
            if edge1 is not None:
                cg_new.G.remove_edge(edge1)
            # Orient only if the edges have not been oriented the other way around
            cg_new.G.add_edge(
                Edge(
                    cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW
                )
            )

            edge2 = cg_new.G.get_edge(cg_new.G.nodes[z], cg_new.G.nodes[y])
            if edge2 is not None:
                cg_new.G.remove_edge(edge2)
            cg_new.G.add_edge(
                Edge(
                    cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW
                )
            )

            cg_new.definite_UC.append((x, y, z))

    return cg_new
