# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""CEDAR — Causal Edge Discovery for AutoRegressive time series."""

from __future__ import annotations

import logging
import warnings

import networkx as nx
import numpy as np
import pandas as pd

from .lag_selection import (
    compute_lag_importance,
    compute_lag_pvalues,
    select_all_significant_lags,
    select_lags,
)
from .result import CedarResult

logger = logging.getLogger(__name__)


def _normalize(data: np.ndarray, method: str = "minmax") -> np.ndarray:
    if method == "minmax":
        mins = np.nanmin(data, axis=0, keepdims=True)
        maxs = np.nanmax(data, axis=0, keepdims=True)
        denom = maxs - mins
        denom[denom == 0] = 1.0
        return (data - mins) / denom
    elif method == "zscore":
        mean = np.nanmean(data, axis=0, keepdims=True)
        std = np.nanstd(data, axis=0, keepdims=True)
        std[std == 0] = 1.0
        return (data - mean) / std
    return data.copy()


def _compute_children_mask(
    lag_importance: np.ndarray,
    direction_threshold: float,
    include_lag0: bool,
    direction_method: str = "dcor",
    data: np.ndarray | None = None,
) -> np.ndarray:
    """Identify potential children of each variable.

    Returns a boolean mask where ``mask[i, j] = True`` means variable i
    is a potential child of variable j (so i should NOT be tested as a
    cause of j, and should NOT appear in the conditioning set for j's causes).

    Parameters
    ----------
    direction_method : str
        ``"dcor"`` (default) — asymmetry of max lag-importance (dcor or
        whatever was used for ``lag_importance``). Fast but susceptible to
        hub indirect-path inflation.
        ``"cross_lag"`` — asymmetry of lag-1 cross-correlation:
        |corr(X_j(t-1), X_i(t))| vs |corr(X_i(t-1), X_j(t))|. More
        temporally specific; less affected by hub structure because it uses
        a single fixed lag rather than max over all lags.
    data : ndarray, shape (T, d), optional
        Required when ``direction_method="cross_lag"``.
    """
    if direction_method == "cross_lag":
        if data is None:
            raise ValueError("data must be provided for direction_method='cross_lag'")
        T, d = data.shape
        # A[j, i] = |corr(X_j(t-1), X_i(t))| — how strongly j at lag-1 predicts i
        A = np.zeros((d, d))
        for j in range(d):
            for i in range(d):
                if i == j:
                    continue
                r = np.corrcoef(data[:-1, j], data[1:, i])[0, 1]
                A[j, i] = abs(r) if np.isfinite(r) else 0.0
        mask = np.zeros((d, d), dtype=bool)
        for i in range(d):
            for j in range(d):
                if i == j:
                    continue
                if A[j, i] > 0.02 and A[i, j] > 0.02:
                    ratio = A[j, i] / A[i, j]
                    if ratio > direction_threshold:
                        mask[i, j] = True  # i is a child of j
        return mask

    # Default: dcor asymmetry (original behaviour)
    d = lag_importance.shape[0]
    sl = slice(None) if include_lag0 else slice(1, None)
    M = lag_importance[:, :, sl].max(axis=2)
    mask = np.zeros((d, d), dtype=bool)
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            if M[j, i] > 0.02 and M[i, j] > 0.02:
                # j causes i more than i causes j → i is child of j
                ratio = M[j, i] / M[i, j] if M[i, j] > 0 else 0.0
                if ratio > direction_threshold:
                    mask[i, j] = True
    return mask


class Cedar:
    """CEDAR causal discovery algorithm.

    Parameters
    ----------
    df : pd.DataFrame
        Time series data, shape (T, d).
    ci_test : CIT_Base
        Conditional independence test (from ``causalts.ci_tests``).
        Must support ``ci_test.data = array`` and
        ``ci_test(i, j, cond_set) -> (p_value, statistic)``.
    max_lag : int
        Maximum lag to consider.
    alpha_cond1 : float
        Significance threshold for Condition 1 (dependence).
    alpha_cond2 : float
        Significance threshold for Condition 2 (independence).
    lag_method : str
        Lag importance method: ``"partial_dcor"`` (default, unbiased dcor on
        X_j(t) residualised on X_j(t-1) — best lag accuracy for hub graphs),
        ``"dcor"`` (unbiased U-centered), ``"pearson"``,
        ``"lasso"`` (LassoCV per pair, matching SyPI Mastakouri et al. 2021),
        or ``"granger"`` (Granger-conditional parcorr — conditions on X_j(t-1)
        and shorter lags of X_i to suppress AR-mediated lag inflation).
    lag_pvalue_method : str
        Lag significance testing: ``"t_test"`` (default, ~200x faster,
        no permutation), ``"circular_shift"`` (proper for AR data),
        or ``"permutation"`` (legacy).
    lag_alpha : float
        Significance level for lag selection.
    n_permutations : int
        Permutations for lag significance (only for circular_shift/permutation).
    normalize : str or None
        Data normalization: ``"minmax"``, ``"zscore"``, or ``None``.
    filter_children : bool
        Filter potential children using asymmetric dcor.
    direction_threshold : float
        Threshold ratio for child filtering.
    direction_method : str
        Method for children detection: ``"dcor"`` (default, max lag-importance
        asymmetry) or ``"cross_lag"`` (lag-1 cross-correlation asymmetry —
        more robust for hub-heavy scale-free graphs).
    prune : bool
        Whether to prune indirect edges after discovery.
    alpha_mci : float or None
        Separate alpha for MCI pruning (default None → same as alpha_cond2).
        Setting stricter than alpha_cond2 enables permissive discovery + tight
        pruning (e.g. alpha_cond2=0.00001, alpha_mci=0.01 for scale-free graphs).
    sweep_all_lags : bool
        When True, test every lag 1..max_lag for each candidate pair regardless
        of lag-selection significance. CI tests become the sole acceptance gate.
    max_prune_path : int
        Maximum path length for pruning.
    include_lag0 : bool
        Include contemporaneous effects (lag 0).
    n_jobs : int
        Parallelism for lag importance computation.
    verbose : bool
        Enable debug logging.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        ci_test,
        max_lag: int,
        *,
        alpha_cond1: float = 0.01,
        alpha_cond2: float = 0.01,
        lag_method: str = "partial_dcor",
        lag_pvalue_method: str = "t_test",
        lag_alpha: float = 0.05,
        n_permutations: int = 200,
        normalize: str | None = "minmax",
        filter_children: bool = False,
        direction_threshold: float = 1.2,
        direction_method: str = "dcor",
        prune: bool = True,
        alpha_mci: float | None = None,
        sweep_all_lags: bool = False,
        max_prune_path: int = 4,
        include_lag0: bool = False,
        multi_lag: bool = True,
        multi_lag_keep: str = "first",
        target_var: str | None = None,
        n_jobs: int = 1,
        verbose: bool = False,
        forbidden: list | None = None,
        required: list | None = None,
        knowledge=None,
        impute: str | None = None,
        impute_kwargs: dict | None = None,
        include_C: bool = False,
        c_preset: str = "linear",
        c_array: np.ndarray | None = None,
        include_autoreg: bool = True,
        assume_ar1: bool = True,
        lag_filter: str | None = None,
        lag_filter_frac: float = 0.95,
    ):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if max_lag < 1:
            raise ValueError("max_lag must be >= 1")
        _IMPUTE_CHOICES = {None, "pairwise_complete", "var_em"}
        if impute not in _IMPUTE_CHOICES:
            raise ValueError(f"impute must be one of {_IMPUTE_CHOICES}, got {impute!r}")
        self.impute = impute
        self.impute_kwargs = impute_kwargs or {}

        self._df_raw = df
        self._var_names = list(df.columns)
        self._d = df.shape[1]
        self._ci_test = ci_test
        self.max_lag = max_lag
        self.alpha_cond1 = alpha_cond1
        self.alpha_cond2 = alpha_cond2
        self.lag_method = lag_method
        self.lag_pvalue_method = lag_pvalue_method
        self.lag_alpha = lag_alpha
        self.n_permutations = n_permutations
        self.normalize = normalize
        self.filter_children = filter_children
        self.direction_threshold = direction_threshold
        self.direction_method = direction_method
        self.prune = prune
        self._alpha_mci = alpha_mci
        self.sweep_all_lags = sweep_all_lags
        self.max_prune_path = max_prune_path
        self.include_lag0 = include_lag0
        self.multi_lag = multi_lag
        self.multi_lag_keep = multi_lag_keep
        self.target_var = target_var
        self.n_jobs = n_jobs
        self.verbose = verbose

        # Merge AncestralKnowledge edge constraints into forbidden/required lists
        self._knowledge = knowledge
        if knowledge is not None:
            forbidden = list(forbidden or []) + [
                e[:3] for e in knowledge._forbidden_edges
            ]
            required = list(required or []) + [e[:3] for e in knowledge._required_edges]

        def _bk_to_index_set(edges):
            result = set()
            for cause, effect, lag in edges or []:
                ci = self._var_names.index(cause)
                ei = self._var_names.index(effect)
                result.add((ci, ei, lag))
            return frozenset(result)

        self._forbidden = _bk_to_index_set(forbidden)
        self._required = _bk_to_index_set(required)
        self.include_C = include_C
        self.c_preset = c_preset
        self.c_array = c_array
        self.include_autoreg = include_autoreg
        self.assume_ar1 = assume_ar1
        _LAG_FILTER_CHOICES = {None, "ar_relative"}
        if lag_filter not in _LAG_FILTER_CHOICES:
            raise ValueError(f"lag_filter must be one of {_LAG_FILTER_CHOICES}")
        self.lag_filter = lag_filter
        self.lag_filter_frac = lag_filter_frac

        if df.isna().any().any() and impute in (None, "pairwise_complete"):
            warnings.warn(
                "Input data contains NaN values. Rows with NaN in any "
                "test window will be dropped per-test. "
                "Pass impute='var_em' to fill before discovery.",
                UserWarning,
                stacklevel=2,
            )

    def run(self) -> CedarResult:
        """Execute full CEDAR algorithm.

        Returns
        -------
        CedarResult
        """
        # --- Imputation (applied before normalization and lag selection) ---
        df_work = self._df_raw
        if self.impute == "var_em":
            if not df_work.isna().any().any():
                warnings.warn(
                    "impute='var_em' set but no NaN values found; skipping.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                from ..imputation import var_em_impute

                df_work = var_em_impute(df_work, **self.impute_kwargs)

        # --- C node injection (nonstationarity variable) ---
        # Appended BEFORE normalization so it is treated as a regular variable
        # in lag selection and CI testing. C is always a cause, never a target.
        # Forcing include_lag0=True ensures dcor(C(t), X(t)) at lag 0 is visible
        # to the lag selector (for monotone C, all lags have similar dcor).
        c_names: list[str] = []
        n_c = 0
        if self.include_C:
            from ..cdnots.phase3_utils import _C_PRESET_LABELS, make_c_array

            T = df_work.shape[0]
            C_data = (
                np.asarray(self.c_array, dtype=np.float64)
                if self.c_array is not None
                else make_c_array(T, self.c_preset)
            )
            if C_data.ndim == 1:
                C_data = C_data.reshape(-1, 1)
            n_c = C_data.shape[1]
            if self.c_array is not None:
                # Custom array: use preset labels where available, fall back to C1/C2/...
                preset_labels = _C_PRESET_LABELS.get(self.c_preset, [])
                c_names = [
                    preset_labels[i] if i < len(preset_labels) else f"C{i + 1}"
                    for i in range(n_c)
                ]
            else:
                c_names = list(
                    _C_PRESET_LABELS.get(self.c_preset, ["C_lin"] * n_c)[:n_c]
                )
            C_df = pd.DataFrame(C_data, columns=c_names)
            df_work = pd.concat([df_work.reset_index(drop=True), C_df], axis=1)
        _include_lag0 = True if self.include_C else self.include_lag0

        raw = df_work.values.astype(np.float64)
        data = _normalize(raw, self.normalize) if self.normalize else raw.copy()
        d = df_work.shape[1]
        _var_names = list(df_work.columns)
        c_start = d - n_c  # first C column index; == d when include_C=False

        # --- Phase 0: Lag selection ---
        # Resolve target index once here so it can be forwarded to lag selection.
        # When target_var is set, only the 2d-1 pairs involving the target column
        # are computed (vs d² for full discovery), cutting Phase-0 cost by ~d/2.
        target_col = (
            _var_names.index(self.target_var) if self.target_var is not None else None
        )

        if self.verbose:
            logger.info("Phase 0: Computing lag importance (%s)...", self.lag_method)

        lag_importance = compute_lag_importance(
            data,
            self.max_lag,
            method=self.lag_method,
            include_lag0=_include_lag0,
            n_jobs=self.n_jobs,
            target_col=target_col,
        )

        # --- Phase 0.1: AR-relative lag importance filtering (optional) ---
        # For strongly autoregressive data, all cross-pair importances may be
        # significant under BH. This filter zeros out candidate pairs whose
        # peak importance does not exceed lag_filter_frac × the target's
        # own self-AR importance, keeping only "locally dominant" causes.
        if self.lag_filter == "ar_relative" and c_start > 0:
            lags_slice = slice(0 if _include_lag0 else 1, self.max_lag + 1)
            for e in range(c_start):
                self_peak = lag_importance[e, e, lags_slice].max()
                if self_peak <= 0:
                    continue
                threshold = self.lag_filter_frac * self_peak
                for c in range(d):
                    if c == e:
                        continue
                    if lag_importance[c, e, lags_slice].max() < threshold:
                        lag_importance[c, e, :] = 0.0

        if self.verbose:
            logger.info("Phase 0: Lag significance (%s)...", self.lag_pvalue_method)

        lag_pvalues = compute_lag_pvalues(
            data,
            lag_importance,
            self.max_lag,
            method=self.lag_pvalue_method,
            n_permutations=self.n_permutations,
            include_lag0=_include_lag0,
            n_jobs=self.n_jobs,
            target_col=target_col,
        )

        selected_lags = select_lags(
            lag_importance,
            lag_pvalues,
            alpha=self.lag_alpha,
            include_lag0=_include_lag0,
        )

        # Post-selection corrections for C nodes
        # ─────────────────────────────────────
        # C is a contemporaneous cause by definition (C(t) → Xj(t), lag 0).
        # For monotone bases (linear, quadratic) dcor is nearly identical at
        # all lags, so max_importance may pick lag-1/2 due to floating-point
        # noise.  Force every C→X selected lag to 0 (single-lag path).
        if self.include_C and c_start < d:
            selected_lags[c_start:, :] = 0.0

        # When include_C forces _include_lag0=True but the user did NOT
        # explicitly request contemporaneous X-X edges, undo any lag-0
        # selection for X-X pairs so they stay lag≥1.
        if self.include_C and not self.include_lag0 and c_start < d:
            x_lag0 = selected_lags[:c_start, :c_start] == 0
            selected_lags[:c_start, :c_start][x_lag0] = np.nan

        # Multi-lag: mask of ALL significant lags per pair
        sig_lags_mask = None
        if self.multi_lag:
            sig_lags_mask = select_all_significant_lags(
                lag_importance,
                lag_pvalues,
                alpha=self.lag_alpha,
                include_lag0=_include_lag0,
            )
            if self.include_C and c_start < d:
                # C rows: lag-0 only (multi-lag path).
                sig_lags_mask[c_start:, :, 1:] = False
                # Force C→X lag-0 into the candidate set regardless of lag
                # significance. partial_dcor residualises Y on Y(t-1), which
                # removes the very trend signal C is designed to capture, so
                # lag-importance for C→X is artificially near-zero. This
                # mirrors the selected_lags[c_start:, :] = 0.0 override above.
                sig_lags_mask[c_start:, :c_start, 0] = True
                # X-X pairs: remove lag-0 if user did not request it.
                if not self.include_lag0:
                    sig_lags_mask[:c_start, :c_start, 0] = False

        # sweep_all_lags: override sig_lags_mask to test every lag 1..max_lag
        # for every valid candidate pair. CI tests become the sole lag arbiter.
        if self.sweep_all_lags:
            sig_lags_mask = np.ones((d, d, self.max_lag + 1), dtype=bool)
            sig_lags_mask[:, :, 0] = _include_lag0  # lag-0 only if requested
            if not _include_lag0:
                sig_lags_mask[:, :, 0] = False
            if self.include_C and c_start < d:
                sig_lags_mask[c_start:, :, 1:] = False  # C rows: lag-0 only
            # selected_lags: set valid pairs to max_lag so w_max uses full window;
            # keep NaN for pairs already excluded by lag selection (no signal at all).
            valid = ~np.isnan(selected_lags)
            selected_lags = np.where(valid, float(self.max_lag), np.nan)

        # Children mask
        children_mask = None
        if self.filter_children:
            children_mask = _compute_children_mask(
                lag_importance,
                self.direction_threshold,
                _include_lag0,
                direction_method=self.direction_method,
                data=data if self.direction_method == "cross_lag" else None,
            )

        # --- Phase 1-3: CI testing ---
        if self.verbose:
            mode = "multi-lag" if self.multi_lag else "single-lag"
            logger.info("Phase 1-3: CI testing (%s)...", mode)

        graph = np.zeros((d, d, self.max_lag + 1), dtype=np.int8)
        val_matrix = np.zeros((d, d, self.max_lag + 1))
        pval_matrix = np.ones((d, d, self.max_lag + 1))
        n_tests = 0

        if target_col is not None:
            target_indices = [target_col]
        else:
            # C columns are causes only — never test them as effects
            target_indices = range(c_start)

        # --- Phase 0.5: Estimate AR order for each X variable ---
        # Skipped when assume_ar1=True — all orders fixed to 1, standard Cond2 used.
        if self.verbose:
            if self.assume_ar1:
                logger.info(
                    "Phase 0.5: Skipped (assume_ar1=True — standard AR(1) Cond2)."
                )
            else:
                logger.info("Phase 0.5: Estimating AR order per variable...")
        ar_st = self.max_lag + 1
        ar_end = data.shape[0] - self.max_lag - 1
        # ar_order_cache: col_idx → AR order (int); used by _discover_target
        ar_order_cache: dict[int, int] = {}
        ar_order_by_col: dict[int, int] = {}
        for col in range(c_start):
            p = (
                1
                if self.assume_ar1
                else self._estimate_ar_order(data, col, ar_st, ar_end)
            )
            ar_order_by_col[col] = p
            ar_order_cache[col] = p

        for target_idx in target_indices:
            causes, lags_found, strengths, pvals_c1, tests_done = self._discover_target(
                target_idx,
                selected_lags,
                children_mask,
                data,
                sig_lags_mask=sig_lags_mask,
                lag_importance=lag_importance,
                ar_order_cache=ar_order_cache,
            )
            n_tests += tests_done
            for cause_idx, lag, strength, pv in zip(
                causes, lags_found, strengths, pvals_c1
            ):
                graph[cause_idx, target_idx, lag] = 1
                val_matrix[cause_idx, target_idx, lag] = strength
                pval_matrix[cause_idx, target_idx, lag] = pv

        # --- Phase 1.5: Autoregressive self-loops ---
        ar_violations_dict: dict[str, float] = {}
        # Collect AR violations (AR order ≥ 2) for all variables
        for i in range(c_start):
            if ar_order_by_col.get(i, 1) >= 2:
                ar_violations_dict[_var_names[i]] = float(ar_order_by_col[i])

        if self.include_autoreg and ar_end > ar_st + 10:
            if self.verbose:
                logger.info("Phase 1.5: Adding autoregressive self-loops...")
            for i in range(c_start):
                p_i = ar_order_by_col.get(i, 1)
                for k in range(1, p_i + 1):
                    if k <= self.max_lag:
                        graph[i, i, k] = 1

        # --- Required edges: inject unconditionally before pruning ---
        for ci, ei, lag in self._required:
            if lag is not None:
                if 0 <= lag <= self.max_lag:
                    graph[ci, ei, lag] = 1
            else:
                for lg in range(1, self.max_lag + 1):
                    graph[ci, ei, lg] = 1

        # --- Phase 4: MCI Pruning ---
        pruned = False
        if self.prune:
            if self.verbose:
                logger.info("Phase 4: MCI pruning...")
            graph, val_matrix, extra_tests = self._prune_mci(graph, val_matrix, data)
            n_tests += extra_tests
            pruned = True

        return CedarResult(
            graph=graph,
            val_matrix=val_matrix,
            pvalue_matrix=pval_matrix,
            lag_importance=lag_importance,
            lag_pvalues=lag_pvalues,
            selected_lags=selected_lags,
            var_names=_var_names,
            max_lag=self.max_lag,
            alpha_cond1=self.alpha_cond1,
            alpha_cond2=self.alpha_cond2,
            n_tests=n_tests,
            pruned=pruned,
            df=self._df_raw,
            lag_method=self.lag_method,
            lag_pvalue_method=self.lag_pvalue_method,
            forbidden=list(self._forbidden),
            required=list(self._required),
            c_node_names=c_names,
            ar_violations=ar_violations_dict or None,
            ar_order=ar_order_by_col or None,
        )

    def _estimate_ar_order(self, data: np.ndarray, col: int, st: int, end: int) -> int:
        """Select AR order for variable ``col`` via BIC.

        Fits AR(0) through AR(max_lag) by OLS and returns the order that
        minimises BIC = T·log(RSS/T) + k·log(T).  This is more conservative
        than sequential CI tests at a fixed alpha because the log(T) penalty
        adapts to sample size — at T=300 a spurious AR(2) term must reduce
        residual variance enough to offset log(300)≈5.7 in BIC units, which
        genuine AR(1) noise rarely achieves.  Returns 0 if even AR(1) is not
        justified (no self-loop added).
        """
        T_eff = end - st
        if T_eff < 10:
            return 0
        y = data[st:end, col]

        # AR(0) baseline: intercept only
        rss0 = float(np.sum((y - np.mean(y)) ** 2))
        best_bic = T_eff * np.log(max(rss0, 1e-12) / T_eff)
        best_p = 0

        for k in range(1, self.max_lag + 1):
            lags = [data[st - j : end - j, col] for j in range(1, k + 1)]
            if any(len(c) < T_eff for c in lags):
                break
            X = np.column_stack(lags)
            valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
            n = int(valid.sum())
            if n < k + 2:
                break
            coef, *_ = np.linalg.lstsq(X[valid], y[valid], rcond=None)
            rss = float(np.sum((y[valid] - X[valid] @ coef) ** 2))
            bic = n * np.log(max(rss, 1e-12) / n) + k * np.log(n)
            if bic < best_bic:
                best_bic = bic
                best_p = k
            else:
                break  # BIC worsened — stop
        return best_p

    def _discover_target(
        self,
        target_idx: int,
        selected_lags: np.ndarray,
        children_mask: np.ndarray | None,
        data: np.ndarray,
        sig_lags_mask: np.ndarray | None = None,
        lag_importance: np.ndarray | None = None,
        ar_order_cache: dict | None = None,
    ) -> tuple[list, list, list, list, int]:
        """Discover causes of a single target variable.

        When ``sig_lags_mask`` is provided (multi-lag mode), tests each
        candidate at ALL significant lags in descending importance order,
        accepting the edge at the first lag where both Cond1 and Cond2 pass.
        """
        T, d = data.shape
        w = selected_lags[:, target_idx]

        valid_mask = ~np.isnan(w)
        valid_mask[target_idx] = False
        if children_mask is not None:
            valid_mask &= ~children_mask[:, target_idx]
            # Required edges must not be blocked by the children filter
            for ci, ei, _lag in self._required:
                if ei == target_idx and ci != target_idx:
                    valid_mask[ci] = True

        candidate_indices = np.where(valid_mask)[0]
        if len(candidate_indices) == 0:
            return [], [], [], [], 0

        w_max = int(np.nanmax(w[valid_mask]))
        st = w_max + 1
        end = T - w_max - 1
        if end <= st + 10:
            if self.verbose:
                logger.warning(
                    "Target %s: not enough samples after lagging "
                    "(st=%d, end=%d). Skipping.",
                    self._var_names[target_idx],
                    st,
                    end,
                )
            return [], [], [], [], 0

        # AR check cache (one per candidate variable)
        ar_cache = {}

        causes = []
        lags_found = []
        strengths = []
        pvals_cond1 = []
        tests_done = 0

        for cand_idx in candidate_indices:
            # Determine which lags to test for this candidate
            if sig_lags_mask is not None and lag_importance is not None:
                cand_sig = sig_lags_mask[cand_idx, target_idx, :]
                sig_lag_indices = np.where(cand_sig)[0]
                if len(sig_lag_indices) == 0:
                    continue
                # Sort by importance (descending) — test most promising lag first
                imp_at_sig = lag_importance[cand_idx, target_idx, sig_lag_indices]
                order = np.argsort(-imp_at_sig)
                lags_to_test = sig_lag_indices[order]
            else:
                lags_to_test = [int(w[cand_idx])]

            # AR order check (shared across lags for this candidate).
            # ar_order_cache[col] (from run()) gives the estimated AR order.
            # We warn when AR(2+) is detected but never skip.
            if cand_idx not in ar_cache:
                ar_order = (
                    ar_order_cache[cand_idx]
                    if ar_order_cache and cand_idx in ar_order_cache
                    else self._estimate_ar_order(data, cand_idx, st, end)
                )
                ar_cache[cand_idx] = ar_order

            ar_order = ar_cache[cand_idx]
            if ar_order >= 2 and self.verbose:
                logger.warning(
                    "%s detected as AR(%d): Cond2 uses augmented conditioning.",
                    self._var_names[cand_idx],
                    ar_order,
                )

            # Test each lag (in importance order, stop at first acceptance)
            for w_i in lags_to_test:
                w_i = int(w_i)
                # Skip lags forbidden by background knowledge
                if (cand_idx, target_idx, w_i) in self._forbidden or (
                    cand_idx,
                    target_idx,
                    None,
                ) in self._forbidden:
                    continue
                result = self._test_at_lag(
                    cand_idx,
                    target_idx,
                    w_i,
                    candidate_indices,
                    w,
                    data,
                    st,
                    end,
                    T,
                    ar_order=ar_order,
                )
                if result is None:
                    continue
                p_c1, rho, p_c2, td = result
                tests_done += td

                if p_c1 > self.alpha_cond1:
                    continue
                if p_c2 >= self.alpha_cond2:
                    causes.append(cand_idx)
                    lags_found.append(w_i)
                    strengths.append(rho)
                    pvals_cond1.append(p_c1)
                    if self.verbose:
                        logger.info(
                            "Accepted %s(t-%d) -> %s (rho=%.4f, p1=%.1E, p2=%.4f)",
                            self._var_names[cand_idx],
                            w_i,
                            self._var_names[target_idx],
                            rho,
                            p_c1,
                            p_c2,
                        )
                    if self.multi_lag_keep == "first":
                        break

        return causes, lags_found, strengths, pvals_cond1, tests_done

    def _test_at_lag(
        self,
        cand_idx: int,
        target_idx: int,
        w_i: int,
        candidate_indices: np.ndarray,
        w: np.ndarray,
        data: np.ndarray,
        st: int,
        end: int,
        T: int,
        ar_order: int = 1,
    ) -> tuple[float, float, float, int] | None:
        """Run Cond1 and Cond2 at a specific lag. Returns (p_c1, rho, p_c2, n_tests) or None.

        For AR(p) causes (ar_order >= 2), Cond2 uses a deeper test variable
        X(t-w-p) and conditions on X(t-w-1), ..., X(t-w-p+1) to neutralise
        the AR leakage.
        """
        tests = 0

        x_cand = data[st:end, cand_idx]
        y_shifted = data[st + w_i : end + w_i, target_idx]
        y_prev = data[st + w_i - 1 : end + w_i - 1, target_idx]

        # Conditioning set — shared by both Cond1 and Cond2 (same-Z structure from
        # the original SyPI paper).  Shift = w_i - w_k - 1 intentionally places X_k
        # one step before its direct causal time (t-w_k-1 instead of t-w_k).  This
        # preserves the SyPI property that indirect causes fail Cond2: for an indirect
        # path X_i → X_k → Y, the path X_i(t-w_i-1) → X_k(t-w_k) → Y(t) is NOT fully
        # blocked by X_k(t-w_k-1), so Cond2 sees residual dependence and rejects the
        # edge.  Using t-w_k instead would block this path and cause Cond2 to pass for
        # indirect causes too, flooding the graph with FPs.
        #
        # C-node exception: skip y_prev (Y(t-1)) from the conditioning set when the
        # candidate is a C node. Conditioning on Y(t-1) removes the trend from Y(t),
        # which is exactly what C is designed to explain — it makes C appear
        # conditionally independent of Y even when C is the genuine common cause.
        # Cond2 is unaffected: C(t-1) ⊥ Y | C(t) holds trivially since C is
        # deterministic, regardless of whether Y(t-1) is in the conditioning set.
        is_c_cand = cand_idx >= self._d
        cond_cols = [] if is_c_cand else [y_prev]
        for other_idx in candidate_indices:
            if other_idx == cand_idx:
                continue
            # When testing a C node, skip sibling C columns: C_k(t-1) at
            # shift=-1 near-determines C_j(t) (deterministic trend), causing
            # Cond1 to fail spuriously for every genuine C->X edge.
            if is_c_cand and other_idx >= self._d:
                continue
            w_k = w[other_idx]
            if np.isnan(w_k):
                continue
            w_k = int(w_k)
            shift = w_i - w_k - 1  # original SyPI alignment: X_k at t-w_k-1
            col_start = st + shift
            col_end = end + shift
            if col_start < 0 or col_end > T:
                continue
            cond_cols.append(data[col_start:col_end, other_idx])

        min_len = min([len(x_cand), len(y_shifted)] + [len(c) for c in cond_cols])
        min_len = min(min_len, len(x_cand), len(y_shifted))
        if min_len < 10:
            return None
        x_cand = x_cand[:min_len]
        y_shifted = y_shifted[:min_len]
        cond_cols = [c[:min_len] for c in cond_cols]

        # Cond1
        test_data = np.column_stack([x_cand, y_shifted] + cond_cols)
        nan_mask = np.isnan(test_data).any(axis=1)
        if nan_mask.any():
            test_data = test_data[~nan_mask]
        if test_data.shape[0] < 10:
            return None

        self._ci_test.data = test_data
        cond_set = np.arange(2, test_data.shape[1])
        if len(cond_set) == 0:
            cond_set = None
        p_cond1, rho = self._ci_test(0, 1, cond_set)
        tests += 1

        if p_cond1 > self.alpha_cond1:
            return (p_cond1, rho, 1.0, tests)

        # Cond2 — same conditioning set as Cond1 (same Z preserves SyPI structure).
        # Standard (AR(1)): test X(t-w-1) ⊥ Y | {cond, X(t-w)}
        # AR(p) augmented:  test X(t-w-p) ⊥ Y | {cond, X(t-w-1)..X(t-w-p+1), X(t-w)}
        p = max(ar_order, 1)
        # Extra conditioning cols: X at lags 1..(p-1) relative to x_cand window
        extra_cond = []
        for j in range(1, p):
            col_start_j = st - j
            col_end_j = end - j
            if col_start_j < 0:
                # Not enough history — fall back to standard Cond2
                p = j
                extra_cond = extra_cond[: j - 1]
                break
            extra_cond.append(data[col_start_j:col_end_j, cand_idx][:min_len])

        # Test variable: X(t-w-p), i.e. p steps before x_cand
        c2_start = st - p
        if c2_start < 0:
            p = 1
            extra_cond = []
            c2_start = st - 1

        x_cond2_test = data[c2_start : end - p, cand_idx][:min_len]

        test_data2 = np.column_stack(
            [x_cond2_test, y_shifted] + cond_cols + extra_cond + [x_cand]
        )
        # Note: cond_cols (same Z as Cond1) is intentionally used here — see comment above.
        nan_mask = np.isnan(test_data2).any(axis=1)
        if nan_mask.any():
            test_data2 = test_data2[~nan_mask]
        if test_data2.shape[0] < 10:
            return None

        self._ci_test.data = test_data2
        cond_set2 = np.arange(2, test_data2.shape[1])
        p_cond2, _ = self._ci_test(0, 1, cond_set2)
        tests += 1

        return (p_cond1, rho, p_cond2, tests)

    def _prune_mci(
        self,
        graph: np.ndarray,
        val_matrix: np.ndarray,
        data: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """MCI-style pruning: test each edge conditioning on parents of both endpoints.

        For each edge X(t-tau) -> Y, tests:
            X(t-tau) _|_ Y(t) | {Parents(Y)\\{X}, Y(t-1), Parents(X), X(t-1)}

        Stable removal: all edges are tested on the same graph state, then
        removals are applied simultaneously (order-independent).
        """
        d_x = self._d  # original X-variable count (no C columns)
        d_all = graph.shape[0]  # full graph dimension including C columns
        T = data.shape[0]
        graph = graph.copy()
        val_matrix = val_matrix.copy()
        n_tests = 0
        max_lag = self.max_lag
        st = max_lag + 1
        end = T - max_lag - 1

        # Collect all edges to test — iterate cause over full dimension so
        # C→X edges (cause_idx >= d_x) are included; targets are always X vars.
        edges_to_test = []
        for target_idx in range(d_x):
            for cause_idx in range(d_all):
                for lag in range(graph.shape[2]):
                    if graph[cause_idx, target_idx, lag] == 1:
                        edges_to_test.append((cause_idx, target_idx, lag))

        removals = []
        for cause_idx, target_idx, lag in edges_to_test:
            # Required edges and self-loops are never pruned.
            # Self-loops represent the validated AR structure; pruning them
            # is incorrect because the "target's own past" in the conditioning
            # set would coincide with the source variable itself.
            if cause_idx == target_idx:
                continue
            if (cause_idx, target_idx, lag) in self._required or (
                cause_idx,
                target_idx,
                None,
            ) in self._required:
                continue
            cond_cols = []
            is_c_cause = cause_idx >= d_x

            # Parents of target (excluding the cause under test and target itself —
            # target's own past is added explicitly below to avoid duplication when
            # the self-loop graph[target_idx, target_idx, 1] == 1).
            # For C→X edges also exclude other C columns: they are
            # correlated with the tested C by construction, and conditioning
            # on them would spuriously remove true direct C edges.
            for tau in range(graph.shape[2]):
                for c in range(d_all):
                    if (
                        graph[c, target_idx, tau] == 1
                        and c != cause_idx
                        and c != target_idx
                    ):
                        if is_c_cause and c >= d_x:
                            continue  # skip sibling C columns
                        cond_cols.append(data[st + lag - tau : end + lag - tau, c])

            # Target's own past
            cond_cols.append(data[st + lag - 1 : end + lag - 1, target_idx])

            # Parents of source (always empty for C nodes — C is exogenous)
            if not is_c_cause:
                for tau in range(graph.shape[2]):
                    for c in range(d_all):
                        # Skip cause's own self-loop — cause's own past is added
                        # explicitly below to avoid duplication when
                        # graph[cause_idx, cause_idx, 1] == 1 (assume_ar1=True).
                        if graph[c, cause_idx, tau] == 1 and c != cause_idx:
                            cond_cols.append(data[st - tau : end - tau, c])

                # Source's own past (skip for C nodes: C is deterministic, so
                # C(t-1) nearly perfectly predicts C(t) and would falsely make
                # C(t) ⊥ Xj | C(t-1), causing every true C edge to be removed)
                cond_cols.append(data[st - 1 : end - 1, cause_idx])

            src = data[st:end, cause_idx]
            tgt = data[st + lag : end + lag, target_idx]

            min_len = min(len(src), len(tgt), min(len(c) for c in cond_cols))
            if min_len < 10:
                continue

            test_data = np.column_stack(
                [src[:min_len], tgt[:min_len]] + [c[:min_len] for c in cond_cols]
            )
            nan_mask = np.isnan(test_data).any(axis=1)
            if nan_mask.any():
                test_data = test_data[~nan_mask]
            if test_data.shape[0] < 10:
                continue

            self._ci_test.data = test_data
            cond_set = np.arange(2, test_data.shape[1])
            p_val, _ = self._ci_test(0, 1, cond_set)
            n_tests += 1

            _alpha_prune = (
                self._alpha_mci if self._alpha_mci is not None else self.alpha_cond2
            )
            if p_val >= _alpha_prune:
                removals.append((cause_idx, target_idx, lag, p_val))

        for cause_idx, target_idx, lag, p_val in removals:
            graph[cause_idx, target_idx, lag] = 0
            val_matrix[cause_idx, target_idx, lag] = 0.0
            if self.verbose:
                logger.info(
                    "MCI pruned %s(t-%d) -> %s (p=%.4f)",
                    self._var_names[cause_idx],
                    lag,
                    self._var_names[target_idx],
                    p_val,
                )

        return graph, val_matrix, n_tests

    def _prune_graph(
        self,
        graph: np.ndarray,
        val_matrix: np.ndarray,
        selected_lags: np.ndarray,
        data: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Remove indirect links via Markov blanket CI testing."""
        d = self._d
        T = data.shape[0]
        graph = graph.copy()
        val_matrix = val_matrix.copy()
        n_tests = 0

        # Build directed graph from summary adjacency
        adj = graph.sum(axis=2)
        G = nx.from_numpy_array(adj, create_using=nx.DiGraph)

        # --- Prune indirect paths ---
        for target_idx in range(d):
            parents = list(np.where(graph[:, target_idx, :].any(axis=1))[0])
            if len(parents) <= 1:
                continue

            for parent_idx in list(parents):
                if parent_idx not in parents:
                    continue
                try:
                    descendants = nx.descendants(G, parent_idx)
                except nx.NetworkXError:
                    continue
                common = set(parents) & descendants - {parent_idx}
                if not common:
                    continue

                lag_p = int(np.nanmin(np.where(graph[parent_idx, target_idx, :])))

                # Build Markov blanket test array
                mb_cols = [
                    data[lag_p:T, parent_idx],  # source at lag
                    data[lag_p:T, target_idx],  # target at t
                ]
                # Add target t-1 and t+1 as context
                if lag_p + 1 <= T:
                    mb_cols.append(data[lag_p - 1 : T - 1, target_idx])  # t-1
                if lag_p + 1 < T:
                    mb_cols.append(
                        data[lag_p + 1 : T + 1, target_idx]
                        if lag_p + 1 + (T - lag_p) <= T
                        else data[lag_p + 1 :, target_idx]
                    )

                # Add other parents at their lags
                for other_p in parents:
                    if other_p == parent_idx:
                        continue
                    lag_o = int(np.nanmin(np.where(graph[other_p, target_idx, :])))
                    if lag_o < T:
                        mb_cols.append(data[lag_o:T, other_p])
                        if lag_o > 0:
                            mb_cols.append(data[lag_o - 1 : T - 1, other_p])

                # Align all columns to same length
                min_len = min(len(c) for c in mb_cols)
                if min_len < 10:
                    continue
                mb_cols = [c[:min_len] for c in mb_cols]

                test_data = np.column_stack(mb_cols)
                nan_mask = np.isnan(test_data).any(axis=1)
                if nan_mask.any():
                    test_data = test_data[~nan_mask]
                if test_data.shape[0] < 10:
                    continue

                self._ci_test.data = test_data
                cond_set = np.arange(2, test_data.shape[1])
                p_val, _ = self._ci_test(0, 1, cond_set)
                n_tests += 1

                if p_val >= self.alpha_cond2:
                    graph[parent_idx, target_idx, lag_p] = 0
                    val_matrix[parent_idx, target_idx, lag_p] = 0.0
                    if G.has_edge(parent_idx, target_idx):
                        G.remove_edge(parent_idx, target_idx)
                    parents.remove(parent_idx)
                    if self.verbose:
                        logger.info(
                            "Pruned %s -> %s (p=%.4f)",
                            self._var_names[parent_idx],
                            self._var_names[target_idx],
                            p_val,
                        )

        # --- Prune simple cycles ---
        try:
            simple_cycles = list(nx.simple_cycles(G, length_bound=2))
        except Exception:
            simple_cycles = []

        for cycle in simple_cycles:
            for src, trg in [cycle, cycle[::-1]]:
                if not graph[src, trg, :].any():
                    continue
                lag_s = int(np.nanmin(np.where(graph[src, trg, :])))
                parents_trg = list(np.where(graph[:, trg, :].any(axis=1))[0])

                mb_cols = [
                    data[lag_s:T, src],
                    data[lag_s:T, trg],
                ]
                if lag_s > 0:
                    mb_cols.append(data[lag_s - 1 : T - 1, trg])
                if lag_s + 1 < T:
                    end_idx = min(T, lag_s + 1 + (T - lag_s))
                    mb_cols.append(data[lag_s + 1 : end_idx, trg])

                for op in parents_trg:
                    if op == src:
                        continue
                    if not graph[op, trg, :].any():
                        continue
                    lag_o = int(np.nanmin(np.where(graph[op, trg, :])))
                    mb_cols.append(data[lag_o:T, op])
                    if lag_o > 0:
                        mb_cols.append(data[lag_o - 1 : T - 1, op])

                min_len = min(len(c) for c in mb_cols)
                if min_len < 10:
                    continue
                mb_cols = [c[:min_len] for c in mb_cols]
                test_data = np.column_stack(mb_cols)
                nan_mask = np.isnan(test_data).any(axis=1)
                if nan_mask.any():
                    test_data = test_data[~nan_mask]
                if test_data.shape[0] < 10:
                    continue

                self._ci_test.data = test_data
                cond_set = np.arange(2, test_data.shape[1])
                p_val, _ = self._ci_test(0, 1, cond_set)
                n_tests += 1

                if p_val >= self.alpha_cond2:
                    graph[src, trg, lag_s] = 0
                    val_matrix[src, trg, lag_s] = 0.0
                    if G.has_edge(src, trg):
                        G.remove_edge(src, trg)
                    if self.verbose:
                        logger.info(
                            "Cycle pruned %s -> %s (p=%.4f)",
                            self._var_names[src],
                            self._var_names[trg],
                            p_val,
                        )

        return graph, val_matrix, n_tests


from ..algorithms import register_algorithm  # noqa: E402


@register_algorithm("cedar")
def run_cedar(
    df: pd.DataFrame,
    ci_test,
    max_lag: int,
    *,
    alpha_cond1: float = 0.01,
    alpha_cond2: float = 0.01,
    lag_method: str = "partial_dcor",
    lag_pvalue_method: str = "t_test",
    lag_alpha: float = 0.05,
    n_permutations: int = 200,
    normalize: str | None = "minmax",
    filter_children: bool = False,
    direction_threshold: float = 1.2,
    direction_method: str = "dcor",
    prune: bool = True,
    alpha_mci: float | None = None,
    sweep_all_lags: bool = False,
    max_prune_path: int = 4,
    include_lag0: bool = False,
    multi_lag: bool = True,
    multi_lag_keep: str = "first",
    target_var: str | None = None,
    n_jobs: int = 1,
    verbose: bool = False,
    forbidden: list | None = None,
    required: list | None = None,
    knowledge=None,
    impute: str | None = None,
    impute_kwargs: dict | None = None,
    include_C: bool = False,
    c_preset: str = "linear",
    c_array: np.ndarray | None = None,
    include_autoreg: bool = True,
    assume_ar1: bool = True,
    lag_filter: str | None = None,
    lag_filter_frac: float = 0.95,
) -> CedarResult:
    """Run CEDAR causal discovery.

    Convenience wrapper around ``Cedar(...).run()``.

    Parameters
    ----------
    df : pd.DataFrame
        Time series data.
    ci_test : CIT_Base
        Conditional independence test.
    max_lag : int
        Maximum lag.
    alpha_cond1 : float
        Threshold for dependence test (Cond1).
    alpha_cond2 : float
        Threshold for independence test (Cond2).
    lag_method : str
        Lag importance method: ``"partial_dcor"`` (default), ``"dcor"``,
        ``"pearson"``, ``"lasso"``, or ``"granger"``.  See :class:`Cedar` for details.
    lag_alpha : float
        P-value threshold for lag significance.
    n_permutations : int
        Permutations for lag significance.
    normalize : str or None
        Normalization method.
    filter_children : bool
        Apply direction strength filter.
    direction_threshold : float
        Ratio threshold for child filtering.
    direction_method : str
        Children detection method: ``"dcor"`` (default) or ``"cross_lag"``.
    prune : bool
        Prune indirect edges.
    alpha_mci : float or None
        Separate alpha for MCI pruning (default None → same as alpha_cond2).
    sweep_all_lags : bool
        Test every lag 1..max_lag regardless of lag-selection significance.
    max_prune_path : int
        Max path length for pruning.
    include_lag0 : bool
        Include contemporaneous effects.
    n_jobs : int
        Parallelism for lag computation.
    verbose : bool
        Enable logging.
    forbidden : list of (cause, effect, lag) tuples, optional
        Edges to forbid. Each tuple is ``(cause_var, effect_var, lag)`` where
        ``lag`` is an int ≥ 1, or ``None`` to forbid the pair at all lags.
    required : list of (cause, effect, lag) tuples, optional
        Edges to force into the graph regardless of CI test outcome. Same
        format as ``forbidden``. Required edges are also protected from MCI
        pruning.
    impute : str or None, default None
        Missing-value strategy applied before lag selection and CI testing.
        ``None`` / ``'pairwise_complete'`` — drop NaN rows per CI test
        (default, no pre-processing).
        ``'var_em'`` — fill NaN via iterative VAR-EM before discovery.
    impute_kwargs : dict, optional
        Extra keyword arguments forwarded to the imputer (e.g.
        ``{"p": 2, "n_iter": 10}`` for ``var_em``).
    include_C : bool, default False
        Append a nonstationarity time-index variable (C node) to the data
        before discovery. C is treated as a pure cause — it is tested as a
        potential cause of every X_j but never as an effect. Forces
        ``include_lag0=True`` so the contemporaneous C(t) → X_j(t) edge
        can be detected.
    c_preset : str, default ``'linear'``
        C basis preset. Same options as in ``run_cdnots``:
        ``'linear'``, ``'linear+quad'``, ``'linear+sin'``, ``'linear+exp'``,
        ``'step'``, ``'step+linear'``.
    c_array : ndarray (T, k), optional
        Custom C basis. Overrides ``c_preset`` when provided.
    include_autoreg : bool, default True
        Add autoregressive self-loops ``Xi(t-k) → Xi(t)`` for each
        variable's detected AR order. Self-loops are added before MCI
        pruning and are exempt from pruning.
    assume_ar1 : bool, default True
        When ``True`` (default): add lag-1 self-loops for all variables,
        no order detection. When ``False``: use BIC (T·log(RSS/T) + k·log(T))
        to select AR order per variable.

    Returns
    -------
    CedarResult
        Includes ``ar_order`` (col index → detected AR order) and
        ``ar_violations`` (variable names with AR order ≥ 2).
    """
    algo = Cedar(
        df=df,
        ci_test=ci_test,
        max_lag=max_lag,
        alpha_cond1=alpha_cond1,
        alpha_cond2=alpha_cond2,
        lag_method=lag_method,
        lag_pvalue_method=lag_pvalue_method,
        lag_alpha=lag_alpha,
        n_permutations=n_permutations,
        normalize=normalize,
        filter_children=filter_children,
        direction_threshold=direction_threshold,
        direction_method=direction_method,
        prune=prune,
        alpha_mci=alpha_mci,
        sweep_all_lags=sweep_all_lags,
        max_prune_path=max_prune_path,
        include_lag0=include_lag0,
        multi_lag=multi_lag,
        multi_lag_keep=multi_lag_keep,
        target_var=target_var,
        n_jobs=n_jobs,
        verbose=verbose,
        forbidden=forbidden,
        required=required,
        knowledge=knowledge,
        impute=impute,
        impute_kwargs=impute_kwargs,
        include_C=include_C,
        c_preset=c_preset,
        c_array=c_array,
        include_autoreg=include_autoreg,
        assume_ar1=assume_ar1,
        lag_filter=lag_filter,
        lag_filter_frac=lag_filter_frac,
    )
    return algo.run()
