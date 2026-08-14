# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Causal feature selection.

Discover the causal feature set of a single ``target`` variable in O(d) instead
of running full O(d²) causal discovery. Intended for building predictors: the
returned :class:`FeatureSelectionResult` exposes the selected lagged
``(variable, lag)`` features, their role in the target's neighbourhood
(parent / child / spouse / self / nonstationarity), and a
:meth:`~FeatureSelectionResult.to_design_matrix` helper that materialises the
regressor columns.

By default this returns the target's **direct causes (parents)** — genuinely
O(d), since both backends' single-target search only ever tests the target *as
an effect*. The returned features mirror *discovery*: ``self`` (autoregressive)
edges appear only for lags the backend surfaced — cdnots tests self-lags and
keeps survivors; cedar assumes AR(1), so its lag-1 self edge appears. Deciding
which self-lags to actually feed a model is a separate, predictive step handled
by :meth:`FeatureSelectionResult.to_design_matrix` (``include_self``,
``self_threshold``), not by discovery.

Neither backend can cheaply discover **children** (the target as a *cause* of
something else): doing so requires rerunning single-target discovery on every
other candidate variable to check whether the target shows up as one of *their*
parents, which costs the same O(d²) as a full graph run. That cost is only paid
when ``markov_blanket=True``, which recovers children **and** spouses (co-parents
of those children, via a conditional-independence collider test) — the full
Markov blanket, the theoretically optimal predictive feature set, at
full-graph-equivalent cost.

Backends (choose with ``algo``):

* ``"cedar"``  — reuses CEDAR's native single-target (``target_var``) mode; O(d).
* ``"fastpc"`` — a fast local PC-around-the-target skeleton; O(d). An
  *approximation* of CDNOTS adjacency, not full CDNOTS (no orientation / Meek /
  MCI / Phase-3). lag>=1 = parent, lag-0 = ``undirected``.
* ``"cdnots"`` — the real CDNOTS algorithm run once and sliced to the target;
  O(d²) but faithfully oriented (parent / child / ``undirected`` from the CPDAG).

See ``experiments/feature_selection/FINDINGS.md`` for the validation that the
O(d) backends beat Lasso / correlation-top-k on direct-cause precision and reach
predictive parity once self-history is included.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats

# Roles a selected feature can play relative to the target.
ROLE_PARENT = "parent"
ROLE_CHILD = "child"
ROLE_SPOUSE = "spouse"
ROLE_SELF = "self"
ROLE_NONSTATIONARITY = "nonstationarity"
# A contemporaneous (lag-0) adjacency whose direction is unresolved — the edge
# X — target exists but discovery did not orient it (parent vs child unknown).
ROLE_UNDIRECTED = "undirected"

# Roles that are usable as predictor columns (excludes the C nonstationarity
# flag, which is a diagnostic, not a real feature). Undirected adjacencies are
# usable predictors even though their causal direction is unresolved.
_PREDICTIVE_ROLES = {
    ROLE_PARENT,
    ROLE_CHILD,
    ROLE_SPOUSE,
    ROLE_SELF,
    ROLE_UNDIRECTED,
}


@dataclass(frozen=True)
class Feature:
    """A single selected lagged feature.

    Attributes
    ----------
    variable : str
        Source variable name.
    lag : int
        Lag (>= 0). ``lag=0`` is contemporaneous.
    role : str
        One of ``parent``, ``child``, ``spouse``, ``self``, ``undirected``
        (a lag-0 adjacency whose direction is unresolved), or
        ``nonstationarity`` (a ``C -> target`` time-trend flag).
    score : float or None
        Strength of association for the feature, higher = stronger. Used only
        for *ranking* — it is **not** a selection filter, so every feature
        (regardless of score) is a selected predictor and appears in
        :meth:`FeatureSelectionResult.to_design_matrix`. The metric depends on
        the role: cross-variable ``parent`` / ``child`` / ``spouse`` /
        ``nonstationarity`` carry the discovery backend's CI-test statistic
        (``|partial correlation|``, roughly ``[0, 1]``); ``self`` carries the
        partial distance-correlation of the target with its own past at that lag
        (nonlinear, ``[0, 1]`` — see :func:`_self_dcor_scores`). ``None`` only
        when a score could not be computed (series too short, or ``dcor`` /
        ``val_matrix`` unavailable).
    """

    variable: str
    lag: int
    role: str
    score: float | None = None


@dataclass
class FeatureSelectionResult:
    """Result of :func:`select_features`.

    Attributes
    ----------
    target : str
        The target variable whose feature set was discovered.
    features : list[Feature]
        Selected lagged features, sorted by descending score then (variable, lag).
    var_names : list[str]
        All variable names in the input (C nodes excluded).
    max_lag : int
        Maximum lag considered.
    algo : str
        Backend used (``"cedar"``, ``"fastpc"``, or ``"cdnots"``).
    markov_blanket : bool
        Whether spouses were included (full MB) or just the PC set.
    result : CausalResult or None
        The underlying discovery result, retained for users who want the full
        graph / plotting / DoWhy bridge. ``None`` for backends that don't
        produce one (e.g. the cdnots local skeleton) or the ``from_result``
        path where it is the object passed in.
    """

    target: str
    features: list[Feature]
    var_names: list[str]
    max_lag: int
    algo: str
    markov_blanket: bool = False
    result: object | None = None
    _df: pd.DataFrame | None = field(default=None, repr=False)

    # ------------------------------------------------------------------
    @property
    def variables(self) -> list[str]:
        """Unique source variables among the predictive features (sorted).

        Excludes the nonstationarity (C) flag. Collapses over lags — use
        :attr:`features` for the full ``(variable, lag)`` granularity.
        """
        seen = {f.variable for f in self.features if f.role in _PREDICTIVE_ROLES}
        return sorted(seen)

    def by_role(self, role: str) -> list[Feature]:
        """Features with the given role."""
        return [f for f in self.features if f.role == role]

    @property
    def is_nonstationary(self) -> bool:
        """True if a ``C -> target`` (nonstationarity) edge was detected."""
        return any(f.role == ROLE_NONSTATIONARITY for f in self.features)

    # ------------------------------------------------------------------
    def to_design_matrix(
        self,
        df: pd.DataFrame | None = None,
        include_self: bool = True,
        self_threshold: float = 0.05,
        dropna: bool = True,
    ) -> pd.DataFrame:
        """Materialise the lagged predictor columns for a regressor.

        This is a *predictive* construction, deliberately separate from the
        discovered feature set (:attr:`features`): the discovered graph reports
        what discovery found/assumed, while the design matrix is what you feed a
        model. Cross-variable features (``parent`` / ``child`` / ``spouse``) are
        always included as columns. **Self-history** is added here on its own
        terms, controlled by ``include_self`` and ``self_threshold`` — because
        the target's own past is a predictor decision, not a causal-discovery
        one (discovery prunes/assumes self-loops inconsistently).

        Columns are named ``"{variable}_lag{lag}"`` (contemporaneous keep
        ``lag0``). The nonstationarity C flag is excluded.

        Parameters
        ----------
        df : pd.DataFrame, optional
            Data to build columns from. Defaults to the DataFrame used for
            discovery.
        include_self : bool, default True
            Add the target's own self-history lags as predictor columns.
        self_threshold : float, default 0.05
            Only self-lags whose partial distance-correlation strength (see
            :func:`_self_dcor_scores`) is ``>= self_threshold`` are added. This
            is a dcor *strength* in ``[0, 1]``, not a p-value. Set to ``0`` to
            include every self-lag up to ``max_lag``.
        dropna : bool, default True
            Drop the leading rows that contain NaNs from shifting.

        Returns
        -------
        pd.DataFrame
            Design matrix aligned to ``df.index``.
        """
        source = df if df is not None else self._df
        if source is None:
            raise ValueError("no DataFrame available; pass df=...")
        cols = {}
        # Cross-variable causal features (never self / nonstationarity here).
        for f in self.features:
            if f.role not in (
                ROLE_PARENT,
                ROLE_CHILD,
                ROLE_SPOUSE,
                ROLE_UNDIRECTED,
            ):
                continue
            cols[f"{f.variable}_lag{f.lag}"] = source[f.variable].shift(f.lag)
        # Self-history: independent predictive selection by dcor strength.
        if include_self:
            self_scores = _self_dcor_scores(source[self.target], self.max_lag)
            for lag in range(1, self.max_lag + 1):
                if self_scores.get(lag, 0.0) >= self_threshold:
                    cols[f"{self.target}_lag{lag}"] = source[self.target].shift(lag)
        out = pd.DataFrame(cols, index=source.index)
        if dropna:
            out = out.iloc[self.max_lag :]
        return out

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        n = len(self.features)
        roles = {}
        for f in self.features:
            roles[f.role] = roles.get(f.role, 0) + 1
        role_str = ", ".join(f"{k}={v}" for k, v in sorted(roles.items()))
        return (
            f"FeatureSelectionResult(target={self.target!r}, algo={self.algo!r}, "
            f"features={n} [{role_str}], markov_blanket={self.markov_blanket})"
        )


# ----------------------------------------------------------------------
# Extraction helpers
# ----------------------------------------------------------------------
def _sort_features(features: list[Feature]) -> list[Feature]:
    def key(f: Feature):
        return (-(f.score if f.score is not None else -1.0), f.variable, f.lag)

    return sorted(features, key=key)


def _features_from_graph(
    graph: np.ndarray,
    val_matrix: np.ndarray | None,
    target_idx: int,
    var_names: list[str],
    n_c: int,
    include_children: bool = False,
) -> list[Feature]:
    """Extract parent / undirected / child / self / nonstationarity features for
    a target from a ``(d, d, L+1)`` graph slice. ``var_names`` includes any
    trailing C columns; ``n_c`` is how many of them are C nodes. Outgoing
    (child) edges are only emitted when ``include_children``."""
    d_all = graph.shape[0]
    d_x = d_all - n_c
    feats: list[Feature] = []

    def score_at(cause, effect, lag):
        if val_matrix is None:
            return None
        v = abs(float(val_matrix[cause, effect, lag]))
        # A strength of exactly 0 means the edge was placed without a CI test
        # producing a statistic (e.g. CEDAR's restored AR(1) self-loop). Report
        # None for that, consistently with self-history lags, rather than a
        # misleading 0.0 that looks like "measured zero strength".
        return None if v == 0.0 else v

    # Incoming edges -> parents (and C -> target = nonstationarity flag).
    for lag in range(graph.shape[2]):
        for v in range(d_all):
            if graph[v, target_idx, lag] != 1:
                continue
            if lag == 0 and v == target_idx:
                continue
            if v >= d_x:  # C column
                feats.append(
                    Feature(
                        var_names[v],
                        lag,
                        ROLE_NONSTATIONARITY,
                        score_at(v, target_idx, lag),
                    )
                )
            elif v == target_idx:
                feats.append(
                    Feature(var_names[v], lag, ROLE_SELF, score_at(v, target_idx, lag))
                )
            elif lag == 0:
                # Contemporaneous adjacency: the local skeleton does no
                # orientation, so parent-vs-child is unresolved -> undirected.
                feats.append(
                    Feature(
                        var_names[v],
                        lag,
                        ROLE_UNDIRECTED,
                        score_at(v, target_idx, lag),
                    )
                )
            else:
                # lag >= 1 is oriented by temporal precedence: v(t-lag) -> target.
                feats.append(
                    Feature(
                        var_names[v], lag, ROLE_PARENT, score_at(v, target_idx, lag)
                    )
                )

    # Outgoing edges -> children (cross-variable, non-C); opt-in.
    if include_children:
        for lag in range(graph.shape[2]):
            for e in range(d_x):
                if e == target_idx:
                    continue
                if graph[target_idx, e, lag] == 1:
                    feats.append(
                        Feature(
                            var_names[e], lag, ROLE_CHILD, score_at(target_idx, e, lag)
                        )
                    )

    return feats


def _self_dcor_scores(series, max_lag: int) -> dict[int, float]:
    """Partial distance-correlation of ``target(t)`` with ``target(t-k)`` for
    lags 1..max_lag, controlling for the intermediate lags.

    The nonlinear analog of partial autocorrelation: for each lag ``k`` we
    linearly residualize both ``target(t)`` and ``target(t-k)`` on the shorter
    lags ``target(t-1..t-k+1)``, then measure the (unbiased, U-centered)
    distance correlation between the residuals. This gives every self-lag a
    *deterministic* strength in roughly ``[0, 1]`` that (a) captures nonlinear
    autoregression (unlike PACF / partial correlation) and (b) reflects the
    unique contribution of lag ``k`` beyond the shorter lags. Consistent with
    CEDAR's own lag-importance metric (``dcor.u_distance_correlation_sqr``).

    Returns {} if the series is too short or dcor is unavailable.
    """
    x = np.asarray(series, dtype=float)
    x = x[~np.isnan(x)]
    T = x.size
    if T <= 2 * max_lag + 2:
        return {}
    try:
        import dcor
    except Exception:
        return {}
    scores: dict[int, float] = {}
    for k in range(1, max_lag + 1):
        y = x[k:]  # target(t)
        xk = x[: T - k]  # target(t-k)
        if k == 1:
            ry, rxk = y, xk
        else:
            # intermediate lags target(t-1 .. t-k+1), aligned to the same rows
            Z = np.column_stack([x[k - j : T - j] for j in range(1, k)])
            ry = _resid(y, Z)
            rxk = _resid(xk, Z)
        try:
            s = float(dcor.u_distance_correlation_sqr(ry, rxk))
        except Exception:
            continue
        scores[k] = max(0.0, s) if np.isfinite(s) else 0.0
    return scores


# ----------------------------------------------------------------------
# Spouse recovery (opt-in, markov_blanket=True)
# ----------------------------------------------------------------------
def _resid(a: np.ndarray, z_cols: np.ndarray) -> np.ndarray:
    X = np.column_stack([np.ones(len(a)), z_cols])
    coef, *_ = np.linalg.lstsq(X, a, rcond=None)
    return a - X @ coef


def _partial_corr_pvalue(x: np.ndarray, y: np.ndarray, z_cols: np.ndarray | None):
    """Linear partial-correlation CI test (t-test on residual correlation).

    Used as the *fallback* for the spouse collider test when the user's CI test
    can't be driven on aligned raw arrays (see :func:`_ci_pvalue`, which is the
    primary path and respects a nonlinear ``ci_test``).
    """
    n = len(x)
    if z_cols is None or z_cols.shape[1] == 0:
        rx, ry = x, y
        dof = n - 2
    else:
        rx = _resid(x, z_cols)
        ry = _resid(y, z_cols)
        dof = n - 2 - z_cols.shape[1]
    if dof <= 1:
        return 1.0, 0.0
    r = np.corrcoef(rx, ry)[0, 1]
    if not np.isfinite(r):
        return 1.0, 0.0
    r = float(np.clip(r, -0.999999, 0.999999))
    t_stat = r * np.sqrt(dof / (1 - r**2))
    p = 2 * (1 - _scipy_stats.t.cdf(abs(t_stat), dof))
    return float(p), r


def _ci_pvalue(ci_test, a, b, z_cols):
    """Test ``a ⊥ b | z_cols`` using the *user's* CI test on aligned raw arrays.

    Routes the spouse collider decision through whatever test the caller chose
    (ParCorr, KCI, CMIknn, …) so a nonlinear problem gets a nonlinear collider
    test, not a hardcoded linear one. The test operates on integer column
    indices into ``ci_test.data``; we temporarily point it at a small aligned
    array and restore it afterwards. Falls back to linear partial correlation
    if the test cannot be driven this way.
    """
    if ci_test is None:
        return _partial_corr_pvalue(a, b, z_cols)
    try:
        if z_cols is None or z_cols.shape[1] == 0:
            arr = np.column_stack([a, b])
            cond_idx = []
        else:
            arr = np.column_stack([a, b, z_cols])
            cond_idx = list(range(2, arr.shape[1]))
        saved = getattr(ci_test, "data", None)
        ci_test.data = arr
        try:
            p, stat = ci_test(0, 1, cond_idx)
        finally:
            ci_test.data = saved
        stat = 0.0 if stat is None else abs(float(stat))
        return float(p), stat
    except Exception:
        return _partial_corr_pvalue(a, b, z_cols)


def _find_children_and_spouses(
    df,
    target,
    candidates,
    *,
    algo,
    max_lag,
    ci_test,
    alpha,
    include_lag0,
    c_preset,
    algo_kwargs,
    existing_vars,
):
    """Recover children and spouses for the full Markov blanket.

    Neither backend's single-target search can tell you the target's
    *children* — that requires checking, for every other candidate ``e``,
    whether ``target`` shows up among ``e``'s own discovered parents. This
    costs one single-target rerun per candidate: O(d) reruns of an O(d)
    search each, i.e. O(d²) total — the same order as a full graph run. That
    cost is only paid here, under ``markov_blanket=True``.

    Each rerun on a candidate ``e`` does double duty: if ``target`` appears
    among ``e``'s parents at lag ``a`` (``target(t-a) -> e(t)``), ``e`` is a
    child; and ``e``'s *other* parents ``Z`` at lag ``b``
    (``Z(t-b) -> e(t)``) are spouse candidates. Anchored at the target's own
    reference time, ``Z``'s lag relative to the target is ``b - a``: usable
    (present/past) only when ``b - a >= 0``.

    A spouse is confirmed when ``Z`` is (approximately) independent of the
    target unconditionally but dependent once conditioned on the shared child
    ``e`` — the classic v-structure / collider signature.
    """
    children: list[Feature] = []
    spouses: list[Feature] = []
    seen_spouse: set[tuple[str, int]] = set()

    for e_name in candidates:
        if e_name == target:
            continue
        try:
            e_res = select_features(
                df,
                e_name,
                max_lag=max_lag,
                algo=algo,
                include_lag0=include_lag0,
                markov_blanket=False,
                ci_test=ci_test,
                alpha=alpha,
                c_preset=c_preset,
                **algo_kwargs,
            )
        except Exception:
            continue

        # A causal link into e is a lag>=1 parent OR a lag-0 undirected
        # adjacency (contemporaneous children come through the latter).
        e_parents = e_res.by_role(ROLE_PARENT) + e_res.by_role(ROLE_UNDIRECTED)
        target_edges = [f for f in e_parents if f.variable == target]
        if not target_edges:
            continue  # target is not a cause of e_name -> not a child

        for tf in target_edges:
            a = tf.lag
            if (e_name, a) not in existing_vars:
                children.append(Feature(e_name, a, ROLE_CHILD, tf.score))

            for f in e_parents:
                if f.variable == target:
                    continue
                z_name, b = f.variable, f.lag
                spouse_lag = b - a
                if spouse_lag < 0:
                    continue  # Z would be observed after the target; unusable
                key = (z_name, spouse_lag)
                if key in seen_spouse or key in existing_vars:
                    continue

                x_target = df[target].to_numpy(dtype=float)
                x_e = df[e_name].shift(-a).to_numpy(dtype=float)
                x_z = df[z_name].shift(spouse_lag).to_numpy(dtype=float)
                mask = ~(np.isnan(x_target) | np.isnan(x_e) | np.isnan(x_z))
                if mask.sum() < 20:
                    continue
                xt, xe, xz = x_target[mask], x_e[mask], x_z[mask]

                # Collider test via the user's CI test (nonlinear if they chose
                # a nonlinear one): Z ⊥ target marginally, dependent given child.
                p_marg, _ = _ci_pvalue(ci_test, xz, xt, None)
                p_cond, stat_cond = _ci_pvalue(ci_test, xz, xt, xe.reshape(-1, 1))
                if p_marg > alpha and p_cond <= alpha:
                    seen_spouse.add(key)
                    spouses.append(
                        Feature(z_name, spouse_lag, ROLE_SPOUSE, abs(stat_cond))
                    )

    return children, spouses


# ----------------------------------------------------------------------
# Backends
# ----------------------------------------------------------------------
def _default_ci_test():
    from .ci_tests.parcorr_gpu import ParCorrGPU

    return ParCorrGPU(np.zeros((2, 2)))


def _select_cedar(df, target, *, max_lag, ci_test, alpha, include_lag0, algo_kwargs):
    from .cedar.discovery import run_cedar

    res = run_cedar(
        df,
        ci_test=ci_test,
        max_lag=max_lag,
        target_var=target,
        include_lag0=include_lag0,
        alpha_cond1=alpha,
        alpha_cond2=alpha,
        verbose=False,
        **algo_kwargs,
    )
    var_names = list(res.var_names)
    n_c = len(getattr(res, "c_node_names", []) or [])
    target_idx = var_names.index(target)
    feats = _features_from_graph(res.cg_tig, res.val_matrix, target_idx, var_names, n_c)
    ar_order = None
    if getattr(res, "ar_order", None):
        ar_order = res.ar_order.get(target)
    return res, feats, var_names, n_c, ar_order


def _select_fastpc(
    df, target, *, max_lag, ci_test, alpha, include_lag0, c_preset, algo_kwargs
):
    """Fast O(d) local PC-around-the-target skeleton — an *approximation* of
    CDNOTS adjacency, not full CDNOTS (no collider orientation / Meek / MCI /
    Phase-3). lag>=1 edges are parents by temporal precedence; lag-0 edges are
    reported as ``undirected``."""
    from .cdnots.local_skeleton import local_pc_skeleton

    graph, val_matrix, var_names, n_c = local_pc_skeleton(
        df,
        target,
        ci_test,
        num_lags=max_lag,
        alpha=alpha,
        include_lag0=include_lag0,
        c_preset=c_preset,
        **algo_kwargs,
    )
    target_idx = var_names.index(target)
    feats = _features_from_graph(graph, val_matrix, target_idx, var_names, n_c)
    # The local skeleton has no result object; expose the raw graph slice.
    return None, feats, var_names, n_c, None


def _cpdag_names(res, graph):
    """Variable names for a CDNOTS result, padded to include C columns."""
    names = list(res.var_names) + list(getattr(res, "c_node_names", []) or [])
    while len(names) < graph.shape[0]:
        names.append(f"C{len(names)}")
    return names


def _features_from_cpdag(
    graph, val_matrix, target_idx, names, n_c, include_children, include_lag0=True
):
    """Extract the target's neighbourhood from a *full, oriented* CDNOTS graph.

    Reads the actual orientation: an edge is a ``parent`` if oriented into the
    target, a ``child`` if oriented out, and ``undirected`` if the lag-0 entry is
    symmetric (both directions set — CDNOTS left it unoriented). ``child`` edges
    are only emitted when ``include_children`` (i.e. markov_blanket=True).

    CDNOTS always discovers contemporaneous edges, so when ``include_lag0`` is
    False we drop lag-0 *cross-variable* features here to match the cedar / fastpc
    backends (which never surface them). The ``C -> target`` nonstationarity flag
    is a same-lag-0 diagnostic and is kept regardless."""
    d_all = graph.shape[0]
    d_x = d_all - n_c
    L = graph.shape[2]
    feats: list[Feature] = []

    def score_at(c, e, lag):
        if val_matrix is None:
            return None
        v = abs(float(val_matrix[c, e, lag]))
        return None if v == 0.0 else v

    for v in range(d_all):
        if v == target_idx:
            continue
        is_c = v >= d_x
        for lag in range(L):
            inc = graph[v, target_idx, lag] == 1
            out = graph[target_idx, v, lag] == 1
            if is_c:
                if inc:
                    feats.append(
                        Feature(
                            names[v],
                            lag,
                            ROLE_NONSTATIONARITY,
                            score_at(v, target_idx, lag),
                        )
                    )
                continue
            if lag == 0 and not include_lag0:
                continue  # drop lag-0 cross-variable edges (parity with cedar/fastpc)
            if lag == 0:
                if inc and out:
                    feats.append(
                        Feature(
                            names[v], 0, ROLE_UNDIRECTED, score_at(v, target_idx, 0)
                        )
                    )
                elif inc:
                    feats.append(
                        Feature(names[v], 0, ROLE_PARENT, score_at(v, target_idx, 0))
                    )
                elif out and include_children:
                    feats.append(
                        Feature(names[v], 0, ROLE_CHILD, score_at(target_idx, v, 0))
                    )
            else:
                if inc:  # v(t-lag) -> target(t)
                    feats.append(
                        Feature(
                            names[v], lag, ROLE_PARENT, score_at(v, target_idx, lag)
                        )
                    )
                if out and include_children:  # target(t-lag) -> v(t)
                    feats.append(
                        Feature(names[v], lag, ROLE_CHILD, score_at(target_idx, v, lag))
                    )

    # self-loops present in the oriented graph (diagonal)
    for lag in range(1, L):
        if graph[target_idx, target_idx, lag] == 1:
            feats.append(
                Feature(
                    names[target_idx],
                    lag,
                    ROLE_SELF,
                    score_at(target_idx, target_idx, lag),
                )
            )
    return feats


def _spouses_from_cpdag(graph, target_idx, names, n_c, include_children_feats):
    """Spouses (co-parents of the target's children) read directly from a full
    oriented CDNOTS graph — no CI reruns needed, since the graph already has
    every edge. For each child ``e`` of target at lag ``a``, any other parent
    ``z`` of ``e`` at lag ``b`` is a spouse at lag ``b - a`` (kept if >= 0)."""
    d_all = graph.shape[0]
    d_x = d_all - n_c
    L = graph.shape[2]
    spouses: list[Feature] = []
    seen: set[tuple[str, int]] = set()
    existing = {(f.variable, f.lag) for f in include_children_feats}

    children = [
        (f.variable, f.lag) for f in include_children_feats if f.role == ROLE_CHILD
    ]
    name_to_idx = {names[i]: i for i in range(d_all)}
    for e_name, a in children:
        e = name_to_idx[e_name]
        for z in range(d_x):
            if z == target_idx or z == e:
                continue
            for b in range(L):
                if graph[z, e, b] != 1:
                    continue
                spouse_lag = b - a
                if spouse_lag < 0:
                    continue
                key = (names[z], spouse_lag)
                if key in seen or key in existing:
                    continue
                seen.add(key)
                spouses.append(Feature(names[z], spouse_lag, ROLE_SPOUSE, None))
    return spouses


def _is_linear_test(ci_test) -> bool:
    """Whether ``ci_test`` is a linear (partial-correlation) test.

    Governs how :func:`_score_cross_edges` conditions on the other causes: a
    linear test gets the linear-residualize-then-dcor path (nonlinear in the
    *marginal* edge, and — unlike partial distance correlation — a correct
    *multivariate* linear conditioning); any non-ParCorr test is treated as a
    genuine conditional test and its own statistic is used instead.
    """
    return ci_test is None or "parcorr" in type(ci_test).__name__.lower()


def _score_cross_edges(df, target, feats, ci_test=None):
    """Assign each cross-variable feature a rankable strength, conditioning on
    the target's other causes. CDNOTS's binary graph carries no per-edge
    statistic, so we compute one here.

    The conditioning honours the CI test the caller chose:

    * **Nonlinear test** (KCI, CMIknn, …): the edge score is that test's own
      conditional statistic ``|stat(x, y | Z)|`` — a proper *nonlinear
      conditional* measure, the same test that drove discovery. This is the
      appropriate path when the problem is nonlinear.
    * **Linear test / default** (ParCorr): linear-residualize ``x`` and ``y`` on
      the other causes ``Z`` and take the (U-centered) **distance correlation**
      of the residuals. This still captures nonlinear structure in the *edge
      itself*, and — critically — conditions correctly on a *multivariate* ``Z``
      (partial distance correlation does not: it under-conditions when ``Z`` has
      several columns).

    The association is scored in each edge's **causal direction**: a ``parent`` /
    ``undirected`` / ``spouse`` feature ``v`` at lag ``k`` is scored as
    ``v(t-k) -> target(t)``, but a ``child`` (``target -> v`` at lag ``k``) is
    scored in the *opposite* direction, ``target(t-k) -> v(t)`` — otherwise a
    strong outgoing edge is measured as a nonexistent reverse edge and scored ~0.

    Self / nonstationarity features are left untouched (self is dcor-scored
    later; C has no df column)."""
    cross_roles = (ROLE_PARENT, ROLE_CHILD, ROLE_SPOUSE, ROLE_UNDIRECTED)
    cond_roles = (ROLE_PARENT, ROLE_UNDIRECTED)
    linear = _is_linear_test(ci_test)
    tgt = df[target].to_numpy(dtype=float)
    col = {}
    for f in feats:
        if f.role in cross_roles and f.variable in df.columns:
            col[(f.variable, f.lag)] = df[f.variable].shift(f.lag).to_numpy(dtype=float)
    cond_keys = [
        (f.variable, f.lag)
        for f in feats
        if f.role in cond_roles and f.variable in df.columns
    ]

    out = []
    for f in feats:
        key = (f.variable, f.lag)
        if f.role not in cross_roles or key not in col:
            out.append(f)
            continue
        if f.role == ROLE_CHILD:
            # target(t-lag) -> feature(t): cause is the lagged target, effect the
            # feature at lag 0.
            x = df[target].shift(f.lag).to_numpy(dtype=float)
            y = df[f.variable].to_numpy(dtype=float)
        else:
            # feature(t-lag) -> target(t).
            x = col[key]
            y = tgt
        others = [col[k] for k in cond_keys if k != key]
        Z = np.column_stack(others) if others else None
        mask = ~np.isnan(x) & ~np.isnan(y)
        if Z is not None:
            mask &= ~np.isnan(Z).any(axis=1)
        if mask.sum() < 20:
            out.append(f)
            continue
        xm, ym = x[mask], y[mask]
        Zm = Z[mask] if Z is not None else None
        if not linear:
            # Proper nonlinear conditional statistic from the user's own test.
            _, stat = _ci_pvalue(ci_test, xm, ym, Zm)
            score = float(stat) if np.isfinite(stat) else None
        else:
            xr = _resid(xm, Zm) if Zm is not None else xm
            yr = _resid(ym, Zm) if Zm is not None else ym
            try:
                import dcor

                s = float(dcor.u_distance_correlation_sqr(xr, yr))
                score = max(0.0, s) if np.isfinite(s) else None
            except Exception:
                r = np.corrcoef(xr, yr)[0, 1]
                score = abs(float(r)) if np.isfinite(r) else None
        out.append(Feature(f.variable, f.lag, f.role, score))
    return out


def _select_cdnots_full(
    df,
    target,
    *,
    max_lag,
    ci_test,
    alpha,
    include_lag0,
    c_preset,
    markov_blanket,
    algo_kwargs,
):
    """Real CDNOTS: run the full algorithm once and slice the target's oriented
    neighbourhood from the CPDAG. O(d²) (no single-target speed-up) but faithful
    — proper collider/Meek/Phase-3 orientation, so parent/child/undirected
    reflect CDNOTS's actual verdict."""
    from .cdnots.phase3_utils import run_cdnots

    res = run_cdnots(
        df,
        ci_test,
        num_lags=max_lag,
        include_C=True,
        c_preset=c_preset,
        alpha=alpha,
        verbose=False,
        **algo_kwargs,
    )
    graph = res.cg_tig
    names = _cpdag_names(res, graph)
    n_c = len(names) - len(res.var_names)
    target_idx = names.index(target)
    # CDNOTS's cg_tig is a binary graph with no per-edge statistic; compute a
    # partial-correlation strength below so edges are rankable. CDNOTS always
    # discovers contemporaneous edges, so gate lag-0 cross-variable features on
    # include_lag0 to match the cedar / fastpc backends.
    feats = _features_from_cpdag(
        graph,
        None,
        target_idx,
        names,
        n_c,
        include_children=markov_blanket,
        include_lag0=include_lag0,
    )
    if markov_blanket:
        feats = feats + _spouses_from_cpdag(graph, target_idx, names, n_c, feats)
    feats = _score_cross_edges(df, target, feats, ci_test=ci_test)
    return res, feats, names, n_c, None


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def select_features(
    df: pd.DataFrame,
    target: str,
    *,
    max_lag: int,
    algo: str = "cedar",
    markov_blanket: bool = False,
    include_lag0: bool = False,
    ci_test=None,
    alpha: float = 0.05,
    c_preset: str = "linear",
    from_result=None,
    **algo_kwargs,
) -> FeatureSelectionResult:
    """Discover the causal feature set of ``target`` in O(d).

    Parameters
    ----------
    df : pd.DataFrame
        Time-series data (rows = time, columns = variables).
    target : str
        Variable whose causal features to select.
    max_lag : int
        Maximum lag to consider.
    algo : {"cedar", "fastpc", "cdnots"}, default "cedar"
        Discovery backend:

        * ``"cedar"`` — O(d) pairwise single-target mode (CEDAR ``target_var``).
        * ``"fastpc"`` — O(d) local PC-around-the-target skeleton. A fast
          *approximation* of CDNOTS adjacency, NOT full CDNOTS: no collider
          orientation, Meek rules, MCI pruning or Phase-3 nonstationarity
          orientation. lag>=1 edges are parents (temporal precedence); lag-0
          edges are reported as ``undirected`` (direction unresolved).
        * ``"cdnots"`` — the *real* CDNOTS algorithm run once and sliced to the
          target's neighbourhood. O(d²) (no single-target speed-up) but
          faithful: parent / child / ``undirected`` reflect CDNOTS's actual
          collider/Meek/Phase-3 orientation of the CPDAG.

        ``cedar``/``fastpc`` are genuinely O(d) because they only ever test the
        target *as an effect*. The returned ``features`` reflect discovery:
        ``self`` appears only for lags a backend surfaced (fastpc tests and
        keeps; cedar assumes AR(1)). Whether/which self-lags become model
        predictors is decided separately in
        :meth:`FeatureSelectionResult.to_design_matrix` (``include_self``,
        ``self_threshold``).
    markov_blanket : bool, default False
        If ``False`` (default) return the target's causes (parents + undirected
        adjacencies). If ``True``, additionally recover children and spouses
        (the full Markov blanket). For ``cedar``/``fastpc`` this needs rerunning
        single-target discovery on every other variable (O(d²), collider CI test
        for spouses); for ``cdnots`` it is read directly off the full oriented
        graph it already computed. Only spouses usable as past/present features
        (lag >= 0 relative to the target) are kept.
    include_lag0 : bool, default False
        Include contemporaneous (lag-0) cross-variable features.
    ci_test : CIT_Base, optional
        CI test instance. Defaults to :class:`ParCorrGPU`.
    alpha : float, default 0.05
        Significance level for CI tests.
    c_preset : str, default "linear"
        C-node preset (CDNOTS family only; ignored by cedar unless include_C).
    from_result : CausalResult, optional
        Slice an already-computed full-discovery result instead of running a
        fresh restricted discovery. No speed benefit over the run you already
        paid for; provided as a convenience.
    **algo_kwargs
        Extra keyword arguments forwarded to the backend.

    Returns
    -------
    FeatureSelectionResult
    """
    if target not in df.columns:
        raise ValueError(f"target {target!r} not in df columns {list(df.columns)}")
    algo = algo.lower()

    if from_result is not None:
        res = from_result
        var_names = list(res.var_names)
        n_c = len(getattr(res, "c_node_names", []) or [])
        if target not in var_names:
            raise ValueError(f"target {target!r} not in result var_names")
        target_idx = var_names.index(target)
        val_matrix = getattr(res, "val_matrix", None)
        feats = _features_from_graph(
            res.cg_tig,
            val_matrix,
            target_idx,
            var_names,
            n_c,
            include_children=markov_blanket,
        )
        ar_order = None
        if getattr(res, "ar_order", None):
            ar_order = res.ar_order.get(target)
    elif algo == "cedar":
        if ci_test is None:
            ci_test = _default_ci_test()
        res, feats, var_names, n_c, ar_order = _select_cedar(
            df,
            target,
            max_lag=max_lag,
            ci_test=ci_test,
            alpha=alpha,
            include_lag0=include_lag0,
            algo_kwargs=algo_kwargs,
        )
    elif algo in ("fastpc", "fast_pc", "local_pc"):
        if ci_test is None:
            ci_test = _default_ci_test()
        res, feats, var_names, n_c, ar_order = _select_fastpc(
            df,
            target,
            max_lag=max_lag,
            ci_test=ci_test,
            alpha=alpha,
            include_lag0=include_lag0,
            c_preset=c_preset,
            algo_kwargs=algo_kwargs,
        )
        algo = "fastpc"
    elif algo in ("cdnots", "cdnots+", "cdnots_plus"):
        if ci_test is None:
            ci_test = _default_ci_test()
        res, feats, var_names, n_c, ar_order = _select_cdnots_full(
            df,
            target,
            max_lag=max_lag,
            ci_test=ci_test,
            alpha=alpha,
            include_lag0=include_lag0,
            c_preset=c_preset,
            markov_blanket=markov_blanket,
            algo_kwargs=algo_kwargs,
        )
        algo = "cdnots"
    else:
        raise ValueError(
            f"unknown algo {algo!r}; expected 'cedar', 'fastpc', or 'cdnots'"
        )

    # strip C names from the exposed variable list
    d_x = len(var_names) - n_c
    exposed_names = var_names[:d_x]

    # Real CDNOTS extracts children+spouses from its full oriented graph inside
    # its backend; a from_result full graph likewise yields spouses directly.
    # cedar/fastpc (fresh, no full graph) discover them iteratively.
    mb_handled_by_backend = from_result is None and algo == "cdnots"
    if markov_blanket and not mb_handled_by_backend:
        if from_result is not None:
            # children already came from _features_from_graph (outgoing edges);
            # add spouses read straight from the provided full graph.
            fr_names = list(res.var_names) + list(
                getattr(res, "c_node_names", []) or []
            )
            while len(fr_names) < res.cg_tig.shape[0]:
                fr_names.append(f"C{len(fr_names)}")
            feats = feats + _spouses_from_cpdag(
                res.cg_tig, fr_names.index(target), fr_names, n_c, feats
            )
        else:
            existing_vars = {(f.variable, f.lag) for f in feats}
            candidates = [v for v in exposed_names if v != target]
            children, spouses = _find_children_and_spouses(
                df,
                target,
                candidates,
                algo=algo,
                max_lag=max_lag,
                ci_test=ci_test,
                alpha=alpha,
                include_lag0=include_lag0,
                c_preset=c_preset,
                algo_kwargs=algo_kwargs,
                existing_vars=existing_vars,
            )
            feats = feats + children + spouses

    # Score the self-history that discovery *actually surfaced* (cdnots: tested
    # and kept; cedar: assumed AR(1) lag 1) by partial distance-correlation — a
    # stable, nonlinear strength in [0, 1]. We do NOT fabricate self-lags here:
    # the returned features mirror discovery. Adding self-lags as model inputs
    # (and thresholding them) is a separate decision in to_design_matrix().
    self_scores = _self_dcor_scores(df[target], max_lag)
    feats = [
        (
            Feature(f.variable, f.lag, ROLE_SELF, self_scores.get(f.lag, f.score))
            if f.role == ROLE_SELF
            else f
        )
        for f in feats
    ]

    return FeatureSelectionResult(
        target=target,
        features=_sort_features(feats),
        var_names=exposed_names,
        max_lag=max_lag,
        algo=algo,
        markov_blanket=markov_blanket,
        result=res,
        _df=df,
    )
