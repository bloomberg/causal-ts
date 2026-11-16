"""Edge-aware node layout: start from a stress layout, then search locally.

Stress layouts (neato, Kamada-Kawai) place nodes by graph distance and never
look at how the drawn edges cross or brush past other nodes. ``auto_layout``
scores a layout on the edges as they are drawn -- ``arc3`` curves, the way
``compare_graphs`` and ``plot_graph`` draw them -- and improves it with a
greedy local search (nudge one node, or swap two). Each move rescores only
the arcs that touch the moved nodes, so a 40-node graph takes well under a
second.

Score (lower is better)::

    crossings + 3 * (arc passes near a node it does not join)
              + 30 * sum over node pairs of max(0, 1 - distance / spacing)
              + stress against graph distance (sum of squared relative errors)

``spacing`` is the smaller of ``2.5 * node_size`` and ``1.6 / sqrt(n)``, so
the crowding term does not fight the box size on large graphs. The stress
term keeps the start layout's overall shape, so clusters do not collapse.
"""

from __future__ import annotations

import shutil

import networkx as nx
import numpy as np

_SAMPLES = 8  # segments per sampled arc


def _graphviz_available():
    if shutil.which("neato") is None:
        return False
    try:
        import pygraphviz  # noqa: F401
    except ImportError:
        try:
            import pydot  # noqa: F401
        except ImportError:
            return False
    return True


def _to_box(P):
    """Centre and scale to [-1, 1]^2, keeping the aspect ratio."""
    P = P - (P.max(0) + P.min(0)) / 2
    half = np.abs(P).max()
    return P / half if half > 0 else P


def _segment_gap(curves, pts):
    """(k, m): distance from each point to each sampled arc polyline."""
    a, ab = curves[:, :-1], curves[:, 1:] - curves[:, :-1]  # (k, S, 2)
    ap = pts[None, :, None] - a[:, None]  # (k, m, S, 2)
    den = np.maximum((ab * ab).sum(-1), 1e-12)[:, None]
    t = np.clip((ap * ab[:, None]).sum(-1) / den, 0, 1)
    return np.linalg.norm(ap - t[..., None] * ab[:, None], axis=-1).min(-1)


class _Scorer:
    def __init__(self, P, arcs, rad, node_size, hops=None, node_aspect=1.0):
        self.P = P.copy()
        self.arcs = arcs
        # Nodes are ellipses node_aspect times wider than tall: measure
        # clearance with x shrunk by that factor, against the height radius.
        self.xscale = np.array([1.0 / node_aspect, 1.0])
        n, k = len(P), len(arcs)
        self.rad = np.broadcast_to(np.asarray(rad, dtype=float), (k,)).copy()
        self.spacing = min(2.5 * node_size, 1.6 / np.sqrt(max(n, 1)))
        # never below the drawn node radius: an arc inside the circle is a hit
        self.node_r = node_size / 2
        self.near_tol = max(0.5 * node_size, min(0.9 * node_size, 0.45 * self.spacing))
        # arcs touching each node, and arc pairs that share an endpoint
        self.incident = [np.flatnonzero((arcs == v).any(1)) for v in range(n)]
        a, b = arcs[:, 0], arcs[:, 1]
        # Strokes between the same two nodes are drawn on separate arcs and
        # never scored against each other. Strokes sharing one node can still
        # cross away from it, so for them only intersections inside the
        # shared node are ignored (start_shared / end_shared below).
        self.start_shared = (a[:, None] == a[None]) | (a[:, None] == b[None])
        self.end_shared = (b[:, None] == a[None]) | (b[:, None] == b[None])
        self.shared = self.start_shared & self.end_shared
        self.endpoint = (np.arange(n)[None] == a[:, None]) | (
            np.arange(n)[None] == b[:, None]
        )
        self.curves = self._curves(np.arange(k))
        self.cross = np.zeros((k, k), dtype=bool)
        if k:
            self.cross = self._cross_rows(np.arange(k))
        self.near = self._near(np.arange(k), np.arange(n))
        # [0]: plain distances (stress); [1]: with x shrunk by node_aspect, so
        # wide nodes side by side count as crowded.
        self.dist = self._dists(np.arange(n))
        # Stress against graph distance keeps the start's overall shape, so
        # the search untangles edges without collapsing clusters. The scale
        # is fitted once, on the start layout.
        self.iu = np.triu_indices(n, 1)
        self.hops = None
        if hops is not None and n > 2:
            h = hops[self.iu]
            self.hops = h
            self.scale = np.mean(self.dist[0][self.iu] / h)

    def _curves(self, idx):
        A, B = self.P[self.arcs[idx, 0]], self.P[self.arcs[idx, 1]]
        d = B - A
        C = (A + B) / 2 + self.rad[idx, None] * np.stack([d[:, 1], -d[:, 0]], axis=1)
        t = np.linspace(0, 1, _SAMPLES + 1)[None, :, None]
        return (
            (1 - t) ** 2 * A[:, None] + 2 * (1 - t) * t * C[:, None] + t**2 * B[:, None]
        )

    def _cross_rows(self, idx):
        """(len(idx), k): does arc idx[i] cross arc j (not sharing a node)?"""
        lo, hi = self.curves.min(1), self.curves.max(1)  # (k, 2) bounding boxes
        cand = (
            (lo[idx][:, None] <= hi[None]).all(-1)
            & (lo[None] <= hi[idx][:, None]).all(-1)
            & ~self.shared[idx]
        )
        rows = np.zeros(cand.shape, dtype=bool)
        pi, pj = np.nonzero(cand)
        chunk = 20000  # bounds the (m, S, S) temporaries on dense graphs
        for c in range(0, pi.size, chunk):
            ci, cj = pi[c : c + chunk], pj[c : c + chunk]
            gi = idx[ci]
            s1, s2 = self.curves[idx[ci]], self.curves[cj]  # (m, S+1, 2)
            p, r = s1[:, :-1, None], (s1[:, 1:] - s1[:, :-1])[:, :, None]
            q, s = s2[:, None, :-1], (s2[:, 1:] - s2[:, :-1])[:, None]
            den = r[..., 0] * s[..., 1] - r[..., 1] * s[..., 0]
            qp = q - p
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (qp[..., 0] * s[..., 1] - qp[..., 1] * s[..., 0]) / den
                u = (qp[..., 0] * r[..., 1] - qp[..., 1] * r[..., 0]) / den
            # Closed intervals: a crossing exactly at a sample point still
            # counts.
            hit = (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
            # Strokes sharing a node meet at it; ignore only intersections
            # inside that node, so crossings just outside it still count.
            for end, shared in ((0, self.start_shared), (1, self.end_shared)):
                at = shared[gi, cj]
                if not at.any():
                    continue
                m, a, b = np.nonzero(hit & at[:, None, None])  # hits to check
                x = p[m, a, 0] + t[m, a, b][:, None] * r[m, a, 0]
                node = self.P[self.arcs[gi[m], end]]
                inside = np.linalg.norm((x - node) * self.xscale, axis=-1) < self.node_r
                hit[m[inside], a[inside], b[inside]] = False
            rows[ci, cj] = hit.any(axis=(1, 2))
        return rows

    def _near(self, arc_idx, node_idx):
        gap = _segment_gap(
            self.curves[arc_idx] * self.xscale, self.P[node_idx] * self.xscale
        )
        return (gap < self.near_tol) & ~self.endpoint[np.ix_(arc_idx, node_idx)]

    def _dists(self, rows):
        """(2, len(rows), n): plain and aspect-scaled distances from *rows*."""
        diff = self.P[rows][:, None] - self.P[None]
        return np.stack(
            [np.linalg.norm(diff, axis=-1), np.linalg.norm(diff * self.xscale, axis=-1)]
        )

    def total(self, cross=None, near=None, dist=None):
        cross = self.cross if cross is None else cross
        near = self.near if near is None else near
        dist = self.dist if dist is None else dist
        d = dist[0][self.iu]
        crowd = np.clip(1 - dist[1][self.iu] / self.spacing, 0, None).sum()
        score = cross.sum() / 2 + 3 * near.sum() + 30 * crowd
        if self.hops is not None:
            target = self.scale * self.hops
            score += np.sum(((d - target) / target) ** 2)
        return score

    def propose(self, moved, new_xy):
        """Score with nodes *moved* at *new_xy*; return (score, state)."""
        old = self.P[moved].copy()
        self.P[moved] = new_xy
        arcs_hit = np.unique(np.concatenate([self.incident[v] for v in moved]))
        curves = self.curves.copy()
        cross, near, dist = self.cross.copy(), self.near.copy(), self.dist.copy()
        if arcs_hit.size:
            saved, self.curves = self.curves, curves
            self.curves[arcs_hit] = self._curves(arcs_hit)
            rows = self._cross_rows(arcs_hit)
            cross[arcs_hit] = rows
            cross[:, arcs_hit] = rows.T
            near[arcs_hit] = self._near(arcs_hit, np.arange(len(self.P)))
            self.curves = saved
        if len(self.arcs):
            near[:, moved] = self._near_with(curves, moved)
        d = self._dists(moved)
        dist[:, moved] = d
        dist[:, :, moved] = d.transpose(0, 2, 1)
        score = self.total(cross, near, dist)
        self.P[moved] = old
        return score, (moved, new_xy, curves, cross, near, dist)

    def _near_with(self, curves, nodes):
        gap = _segment_gap(curves * self.xscale, self.P[nodes] * self.xscale)
        return (gap < self.near_tol) & ~self.endpoint[:, nodes]

    def accept(self, state):
        moved, new_xy, self.curves, self.cross, self.near, self.dist = state
        self.P[moved] = new_xy


def _search(P0, arcs, rad, node_size, iterations, rng, hops=None, node_aspect=1.0):
    sc = _Scorer(P0, arcs, rad, node_size, hops, node_aspect)
    best = sc.total()
    n = len(P0)
    patience = max(200, 15 * n)  # stop once this many moves in a row fail
    stale = 0
    for it in range(iterations):
        if best == 0 or stale >= patience:
            break
        if n > 2 and rng.random() < 0.3:
            moved = rng.choice(n, 2, replace=False)
            new_xy = sc.P[moved[::-1]].copy()
        else:
            moved = np.array([rng.integers(n)])
            step = 0.15 * (1 - it / iterations) + 0.02
            new_xy = np.clip(sc.P[moved] + rng.normal(0, step, (1, 2)), -1, 1)
        score, state = sc.propose(moved, new_xy)
        if score < best:
            best = score
            sc.accept(state)
            stale = 0
        else:
            stale += 1
    return sc.P, best


def _arc_arrays(G, nodes, curved_radius, arcs):
    """Index pairs and radii of the strokes to score."""
    index = {v: i for i, v in enumerate(nodes)}
    if arcs is None:
        arcs = [(a, b, curved_radius) for a, b in G.edges() if a != b]
    arcs = [(a, b, r) for a, b, r in arcs if a != b]
    pairs = np.array([(index[a], index[b]) for a, b, _ in arcs], dtype=int)
    return pairs.reshape(-1, 2), np.array([r for *_, r in arcs], dtype=float)


def auto_layout(
    G,
    seed=0,
    curved_radius=0.2,
    node_size=0.2,
    iterations=None,
    arcs=None,
    node_aspect=1.0,
):
    """Edge-aware positions for *G*, in [-1, 1]^2, keyed like ``G.nodes``.

    Starts from neato (when Graphviz is installed), Kamada-Kawai and a few
    spring layouts (more for small graphs), runs the local search from each,
    and returns the lowest-scoring result. Deterministic for a given *seed*.

    *arcs* lists the strokes as drawn, ``(u, v, rad)`` with matplotlib's
    ``arc3`` radius (several strokes per pair allowed). By default each edge
    of *G* is one stroke with radius *curved_radius*. *node_size* is the node
    height and *node_aspect* its width / height, as drawn.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    if n <= 2:
        return nx.circular_layout(G) if n else {}
    pairs, rads = _arc_arrays(G, nodes, curved_radius, arcs)
    U = nx.Graph()
    U.add_nodes_from(nodes)
    U.add_edges_from((nodes[a], nodes[b]) for a, b in pairs)
    # graph distance in hops; unreachable pairs one hop past the farthest
    lengths = dict(nx.all_pairs_shortest_path_length(U))
    hops = np.array(
        [[lengths[u].get(v, np.inf) for v in nodes] for u in nodes], dtype=float
    )
    finite = hops[np.isfinite(hops)]
    hops[~np.isfinite(hops)] = (finite.max() if finite.size else 0) + 1

    starts = []
    if _graphviz_available():
        # nx_agraph imports pygraphviz only when called, so guard the call.
        from networkx.drawing import nx_agraph, nx_pydot

        # Same overlap-free neato as the default 'neato' layout (the pydot
        # adapter takes no args), so auto never starts below what users see.
        for adapter, kw in ((nx_agraph, {"args": "-Goverlap=false"}), (nx_pydot, {})):
            try:
                starts.append(adapter.graphviz_layout(U, prog="neato", **kw))
                break
            except Exception:  # missing backend or a broken install
                continue
    starts.append(nx.kamada_kawai_layout(U))
    # Greedy search ends in a local optimum, so a few extra spring starts
    # pay off on small graphs, where each search is cheap.
    n_spring = 3 if n <= 15 else 1 if n <= 30 else 0
    starts += [
        nx.spring_layout(U, seed=None if seed is None else seed + k)
        for k in range(n_spring)
    ]

    if iterations is None:
        # Each move costs about one pass over the k x k crossing matrix, so
        # dense graphs get fewer moves (about a second per start at k=600).
        k = max(len(pairs), 1)
        iterations = min(60 * n + 300, 3000, max(100, int(1e7 / k**2)))
    rng = np.random.default_rng(seed)
    best_P, best_score = None, np.inf
    for start in starts:
        P0 = _to_box(np.array([start[v] for v in nodes], dtype=float))
        P, score = _search(
            P0, pairs, rads, node_size, iterations, rng, hops, node_aspect
        )
        if score < best_score:
            best_P, best_score = P, score
    best_P = _to_box(best_P)
    return {v: (float(x), float(y)) for v, (x, y) in zip(nodes, best_P)}


def layout_score(pos, G, curved_radius=0.2, node_size=0.2, arcs=None, node_aspect=1.0):
    """Score *pos* for *G* (lower is better): crossings, near passes, crowding.

    Returns ``(score, crossings, near_passes)``. *arcs* as in ``auto_layout``.
    """
    nodes = list(G.nodes())
    pairs, rads = _arc_arrays(G, nodes, curved_radius, arcs)
    P = np.array([pos[v] for v in nodes], dtype=float)
    sc = _Scorer(P, pairs, rads, node_size, node_aspect=node_aspect)
    return float(sc.total()), int(sc.cross.sum() // 2), int(sc.near.sum())
