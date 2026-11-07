# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Data inspection and configuration recommendation for causal discovery.

This is the shared core behind the ``causal-ts inspect`` CLI command and the
public :func:`inspect_df` Python entry point.  It measures data-health facts,
recommends an algorithm + CI test + C-node configuration, and estimates a coarse
cost class — all deterministically, so an agent (or a human) gets the same advice
from the CLI and from Python.

Public API:

- :func:`inspect_df` — full report ``{schema_version, data, facts, recommendation,
  cost_class, warnings}`` for an in-memory DataFrame.  ``facts["latent_factor"]``
  reports broad latent-factor structure when it is present; it is positive evidence
  only, so ``detected`` False/None never means "no confounding" (see
  :func:`_latent_factor_block`).
- :func:`recommend_config` — the pure facts → config decision function.
- :func:`discover_df` — run discovery on an in-memory DataFrame (the Python twin
  of ``causal-ts discover``).
- :func:`edges_from_graph` — convert a ``(d, d, L+1)`` graph array into a list of
  named edge objects (used by ``discover --json``).
- :func:`diagnostics_from_graph` — red-flag diagnostics (``empty``, ``saturated``,
  hub in-degree, …) computed from a discovered graph.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .cdnots.phase3_utils import _detect_discrete_cols
from .utils.heteroskedasticity import check_variance_dependence
from .utils.linearity import check_linearity
from .utils.stationarity import check_stationarity_observed, detect_trend_form

SCHEMA_VERSION = 1

# Cost heuristics (Q6): the gate only needs "should I ask first?", not seconds.
_EXPENSIVE_CI_TESTS = {"kci", "cmiknn-gpu", "cmiknn-mixed-gpu"}
_MODERATE_CI_TESTS = {"splitkci", "dfcit", "rcot", "rift"}
_EXPENSIVE_ALGORITHMS = {"grace", "grace-ss"}
_HIGH_DIM = 15
# Most lags at which the variance-dependence check tests scale effects (its mean
# model uses the suggested max lag). Longer blocks cost power and, on strongly
# persistent drivers, calibration.
_VARIANCE_CHECK_TEST_LAGS = 3
# Largest d * (d - 1) * T the pre-flight variance check runs on (about 10 s at
# d = 100, T = 2000); above it the check is reported as not run, and
# check_variance_dependence can be called directly.
_VARIANCE_CHECK_BUDGET = 2.5e7


def _suggested_max_lag(arr, cap=5, default=3):
    """Partial-autocorrelation (PACF) based max-lag suggestion.

    The largest lag (<= ``cap``) at which any column's PACF exceeds the ~95%
    white-noise band (2/sqrt(T)), floored at 1, ``default`` if none is
    significant. PACF (vs. raw ACF) strips the indirect correlation carried
    across lags by a persistent/AR process, so it does not inflate the
    suggestion toward the cap on autocorrelated data. Note it is univariate, so
    it reflects each series' own-lag structure, not cross-variable lags — the
    value is a starting point and is overridable via ``max_lag``.
    """
    from statsmodels.tsa.stattools import pacf

    T, N = arr.shape
    if T < 20:
        return 1
    band = 2.0 / np.sqrt(T)
    best = 0
    for i in range(N):
        x = arr[:, i]
        x = x[np.isfinite(x)]
        nlags = min(cap, len(x) // 2 - 1)
        if len(x) < 20 or np.std(x) == 0 or nlags < 1:
            continue
        try:
            p = pacf(x, nlags=nlags)
        except Exception:
            continue
        for lag in range(1, len(p)):
            if abs(p[lag]) > band and lag > best:
                best = lag
    return best if best >= 1 else default


def edges_from_graph(graph, var_names, pvalue_matrix=None):
    """Convert a ``(d, d, L+1)`` graph array to a list of named edge dicts.

    Convention matches ``cg_tig``: ``graph[cause, effect, lag] != 0`` means
    ``cause(t-lag) -> effect(t)``.  Returns
    ``[{"source", "target", "lag", "pvalue"}]`` (pvalue is ``None`` when no
    matrix is supplied or the edge has no finite p-value, e.g. it was not tested).
    """

    def _name(idx):
        # Graphs may include appended C-node columns beyond the data variables.
        return str(var_names[idx]) if idx < len(var_names) else f"C{idx}"

    edges = []
    for i, j, lag in zip(*np.where(graph)):
        pval = None
        if pvalue_matrix is not None:
            try:
                pval = float(pvalue_matrix[i, j, lag])
            except (IndexError, TypeError, ValueError):
                pval = None
            if pval is not None and not np.isfinite(pval):
                pval = None  # JSON has no NaN
        edges.append(
            {
                "source": _name(i),
                "target": _name(j),
                "lag": int(lag),
                "pvalue": pval,
            }
        )
    return edges


def diagnostics_from_graph(graph, var_names, saturated_frac=0.5):
    """Compute deterministic red-flag diagnostics from a ``(d, d, L+1)`` graph.

    Returns a dict the interpretation playbook keys off (so the agent never
    misses a condition and both harnesses see the same flags):
    ``n_edges``, ``density`` (edges / all directed lag-slots), ``self_loops``,
    ``contemporaneous`` (lag 0) / ``lagged`` (lag >= 1) counts, ``max_in_degree``
    + ``hub`` (the target variable with the most incoming edges), and boolean
    ``empty`` / ``saturated`` flags.
    """
    g = np.asarray(graph)
    d = g.shape[0]
    L1 = g.shape[2] if g.ndim == 3 else 1
    n_edges = int(g.astype(bool).sum())
    total_slots = d * d * L1
    self_loops = int(sum(1 for i, j, _ in zip(*np.where(g)) if i == j))
    contemporaneous = int(g[:, :, 0].astype(bool).sum()) if g.ndim == 3 else 0
    lagged = n_edges - contemporaneous

    # in-degree per target (effect) = nonzero over cause and lag axes
    in_deg = (
        g.astype(bool).any(axis=2).sum(axis=0)
        if g.ndim == 3
        else g.astype(bool).sum(axis=0)
    )
    max_in = int(in_deg.max()) if d else 0
    hub = None
    if max_in > 0:
        j = int(np.argmax(in_deg))
        hub = str(var_names[j]) if j < len(var_names) else f"C{j}"

    density = (n_edges / total_slots) if total_slots else 0.0
    return {
        "n_edges": n_edges,
        "density": round(density, 4),
        "self_loops": self_loops,
        "contemporaneous": contemporaneous,
        "lagged": lagged,
        "max_in_degree": max_in,
        "hub": hub,
        "empty": n_edges == 0,
        "saturated": density > saturated_frac,
    }


def merge_lag0(refined, candidate):
    """Lagged edges of ``refined`` with the lag-0 edges of ``candidate``.

    GRACE refines lagged edges only and returns an empty lag-0 slice; this puts
    back the contemporaneous edges of the CDNOTS graph it refined. Takes result
    objects or ``cg_tig`` arrays and returns a new array shaped like
    ``refined.cg_tig``; extra rows/columns of either graph (e.g. a C node) are
    left as they are in ``refined``.
    """
    g = np.array(getattr(refined, "cg_tig", refined), dtype=float, copy=True)
    c = np.asarray(getattr(candidate, "cg_tig", candidate))
    k = min(g.shape[0], c.shape[0])
    g[:k, :k, 0] = c[:k, :k, 0]
    return g


class NoNumericColumnsError(ValueError):
    """:func:`inspect_df` was given a frame with no numeric column."""


def numeric_frame(df):
    """Return ``df`` with Boolean columns as 0/1 floats.

    pandas keeps Booleans out of numeric selections and turns a frame that mixes
    them with floats into an object array; as 0/1 they are ordinary discrete
    columns for the checks, the discrete-aware CI tests and the lag embedding.
    """
    bool_cols = list(df.select_dtypes(include=["bool", "boolean"]).columns)
    # True/False read from a file with a blank cell arrive as object columns.
    for c in df.select_dtypes(include=["object"]).columns:
        vals = df[c].dropna()
        if len(vals) and vals.map(lambda v: isinstance(v, (bool, np.bool_))).all():
            bool_cols.append(c)
    if bool_cols:
        df = df.copy()
        for c in bool_cols:
            df[c] = df[c].map(lambda v: np.nan if pd.isna(v) else float(v))
            df[c] = df[c].astype("float64")
    return df


def _data_block(df):
    T, d = df.shape
    var_names = [str(c) for c in df.columns]
    missing_by_col = {
        str(c): float(df[c].isna().mean()) for c in df.columns if df[c].isna().any()
    }
    constant_cols = [str(c) for c in df.columns if df[c].nunique(dropna=True) <= 1]
    # Constant columns are integer-valued with one level, so the discrete
    # heuristic flags them — but they are junk (already warned), not genuine
    # discrete variables, and must not drive the CI-test recommendation.
    const_set = set(constant_cols)
    discrete_cols = [
        str(c) for c in _detect_discrete_cols(df, T) if str(c) not in const_set
    ]
    return {
        "n_rows": int(T),
        "n_vars": int(d),
        "var_names": var_names,
        "missing_by_col": missing_by_col,
        "constant_cols": constant_cols,
        "discrete_cols": discrete_cols,
    }


def _warnings(df, data, non_numeric=()):
    """Deterministic data-health flags (the Q1 pre-flight surface)."""
    warns = []
    if non_numeric:
        warns.append(
            f"Non-numeric column(s) were left out of the inspection and must be "
            f"dropped or encoded before discovery: "
            f"{list(non_numeric)}."
        )
    for col, frac in data["missing_by_col"].items():
        if frac >= 0.2:
            warns.append(
                f"{col} is {frac:.0%} missing — impute or drop before discovery."
            )
        else:
            warns.append(
                f"{col} has missing values ({frac:.1%}); impute them before "
                "discovery (the recommendation sets impute='var_em')."
            )
    if data["constant_cols"]:
        warns.append(
            f"Constant column(s) carry no information and should be dropped: "
            f"{data['constant_cols']}."
        )
    if data["n_rows"] < 50:
        warns.append(
            f"Only {data['n_rows']} rows — discovery is unreliable on very short "
            f"series; results should be treated as tentative."
        )
    if data["n_vars"] > data["n_rows"]:
        warns.append(
            "More variables than time steps — consider reducing dimensionality."
        )
    return warns


def _latent_factor_block(arr):
    """Marchenko-Pastur check for broad latent-factor structure.

    Delegates to LUCID's own router (``causalts.confounders.routed_deconf._route``)
    rather than re-deriving the statistic, so this can never drift from the routing
    decision :func:`~causalts.confounders.run_lucid` actually makes, and inherits its
    validated no-factor null instead of a fresh ad-hoc cutoff.

    This yields **positive evidence only**:

    - ``detected=True``  — the residual spectrum carries factor structure above the
      no-factor null, i.e. one or more latent factors load broadly enough to see.
      This covers both regimes LUCID treats as confounded (``"sf"`` and
      ``"pervasive"``), so do not describe it as specifically "pervasive".
    - ``detected=False`` — no such structure. This is **not** "no confounding": a
      latent cause touching only two or three variables produces no dominant
      eigenvalue and is invisible to this test by construction.
    - ``detected=None``  — the check could not run (see the guards below); this is
      deliberately distinct from ``False`` so "did not check" is never read as
      "checked and found nothing".

    Returns
    -------
    dict
        ``{detected, spectral_ratio, tau}``; the latter two are ``None`` whenever
        the check did not run.
    """
    unavailable = {"detected": None, "spectral_ratio": None, "tau": None}
    T, d = arr.shape
    # The statistic is built on VAR(1) least-squares residuals, so skip the inputs
    # that would make the fit raise (non-finite) or return a meaningless number
    # (underdetermined: the design matrix is (T-1) x (d+1)). With d < 3 the top-2
    # eigenvalues are (nearly) the whole trace, so the threshold can never be reached.
    if d < 3 or T < d + 3 or not np.isfinite(arr).all():
        return unavailable
    try:
        from .confounders.routed_deconf import _route

        regime, info = _route(arr, router="auto")
    except Exception:  # a diagnostic must never break the whole report
        return unavailable
    # The statistic is the top-2 share of the trace, so it never exceeds 1. When the
    # threshold (from the residual degrees of freedom, T - d - 2) reaches 1, the test
    # cannot fire on this panel: report "could not check", not "found nothing".
    if not info["tau"] < 1.0:
        return unavailable
    return {
        "detected": regime != "sparse",
        "spectral_ratio": float(info["R"]),
        "tau": float(info["tau"]),
    }


def _variance_dependence_block(arr, var_names, max_lag):
    """``facts["variance_dependence"]``: lagged parents that move only the noise scale.

    Wraps :func:`~causalts.utils.heteroskedasticity.check_variance_dependence`.
    ``detected`` is None when the check did not run, which -- like the
    latent-factor block -- means "not checked", not "none". ``reason`` says why:
    ``"not_applicable"`` (``max_lag`` 0 or fewer than 2 variables: no lagged
    cross pairs), ``"too_large"`` (``d * (d - 1) * T`` above
    ``_VARIANCE_CHECK_BUDGET``) or ``"unavailable"`` (non-finite values, too few
    rows, degenerate columns); it is None when the check ran.
    """
    T, d = arr.shape

    def _not_run(reason):
        return {"detected": None, "n_pairs": 0, "pairs": [], "reason": reason}

    if int(max_lag) < 1 or d < 2:
        return _not_run("not_applicable")
    if d * (d - 1) * T > _VARIANCE_CHECK_BUDGET:
        return _not_run("too_large")  # keep inspect cheap
    try:
        res = check_variance_dependence(
            arr,
            max_lag=int(max_lag),
            test_lags=min(int(max_lag), _VARIANCE_CHECK_TEST_LAGS),
        )
    except Exception:  # a diagnostic must never break the whole report
        return _not_run("unavailable")
    if res["detected"] is None:
        return _not_run("unavailable")
    flagged = sorted(
        (r for r in res["results"] if r["is_variance_parent"]),
        key=lambda r: -r["partial_r2"],
    )
    return {
        "detected": bool(res["detected"]),
        "n_pairs": int(res["n_significant"]),
        "pairs": [f"{var_names[r['x']]} -> {var_names[r['y']]}" for r in flagged[:5]],
        "reason": None,
    }


def recommend_config(facts, data):
    """Map measured facts to a discovery configuration (pure, deterministic).

    Parameters
    ----------
    facts : dict
        The ``facts`` block from :func:`inspect_df`.
    data : dict
        The ``data`` block from :func:`inspect_df`.

    Returns
    -------
    dict with keys ``algorithm``, ``ci_test``, ``refine``, ``refine_params``,
    ``include_C``, ``c_preset``, ``max_lag``, ``params``, ``rationale``.

    Notes
    -----
    The rule, in order:

    - **Discovery** is always CDNOTS: its cost in ``d`` is set by the CI test, not
      by the algorithm, so there is no dimension cut-off.
    - **CI test.** Discrete/mixed columns -> ``cmiknn-mixed-gpu``. Otherwise, if
      variance-only parents were detected -> ``rcot`` (full conditional
      independence, sees dependence through the variance). Otherwise nonlinear ->
      ``raft`` (fast, conditional-mean only), linear -> ``parcorr-gpu``.
    - **Refinement** (``refine``, optional second step on the CDNOTS graph, with
      keyword arguments ``refine_params``): ``"grace"`` when variance-only parents
      were detected (its gated model keeps them, via ``grow_variance``), else
      ``"reap"`` (regression-based, no training). Both refine lagged edges only:
      REAP gets ``contemporaneous=True`` to re-add lag-0 edges; GRACE returns
      no lag-0 edges, so put the CDNOTS graph's back with :func:`merge_lag0`.
      With missing values, ``refine_params`` adds ``impute="var_em"`` (both
      need complete data). ``refine`` is ``None`` when ``max_lag`` is 0.
    """
    d = data["n_vars"]
    T = data["n_rows"]
    nonlinear = facts["linearity"]["is_nonlinear"]
    stat = facts["stationarity"]
    nonstationary = len(stat["nonstationary_cols"]) > 0
    form = stat["form"]
    has_discrete = len(data["discrete_cols"]) > 0
    max_lag = facts["suggested_max_lag"]
    vd = facts.get("variance_dependence") or {}
    variance_parents = vd.get("detected") is True

    reasons = [f"CDNOTS (d={d}, T={T}; its cost in d is set by the CI test)"]
    not_run = (
        {
            "too_large": (
                "variance-dependence check skipped (too large for the pre-flight check), "
                "so variance-only parents are not ruled out; run "
                "causalts.utils.heteroskedasticity.check_variance_dependence directly"
            ),
            "unavailable": (
                "variance-dependence check could not run (non-finite values, too few "
                "rows or degenerate columns), so variance-only parents are not ruled out"
            ),
        }.get(vd.get("reason"))
        if vd.get("detected") is None
        else None
    )
    if not_run:
        reasons.append(not_run)

    # --- CI test ---
    if has_discrete:
        ci_test = "cmiknn-mixed-gpu"
        reasons.append("discrete/mixed columns → cmiknn-mixed-gpu (discrete-aware)")
    elif variance_parents:
        ci_test = "rcot"
        reasons.append(
            f"variance-only parents detected ({vd['n_pairs']} pairs) → rcot "
            "(mean-only tests such as raft/parcorr cannot see them)"
        )
    elif nonlinear:
        ci_test = "raft"
        reasons.append("nonlinear → raft (fast; conditional-mean dependence)")
    else:
        ci_test = "parcorr-gpu"
        if data.get("n_vars", 2) < 2:
            # No pairs, so nothing to test; not a failed test.
            reasons.append(
                "fewer than two numeric columns → pairwise linearity not tested; "
                "parcorr-gpu by default"
            )
        elif facts["linearity"].get("n_tested", 1) == 0:
            # Same choice as before, but not evidence of linearity.
            reasons.append(
                "linearity could not be tested → parcorr-gpu by default (check "
                "linearity, or pick a nonlinear test such as raft)"
            )
        else:
            reasons.append("approximately linear → parcorr-gpu")

    # --- refinement of the CDNOTS graph (optional second step) ---
    # Both refinements refine lagged edges only. REAP can re-add lag-0 edges
    # (contemporaneous=True); GRACE's are put back with merge_lag0.
    if max_lag < 1:
        refine, refine_params = None, {}
        reasons.append("max_lag=0 → no refinement (both refine lagged edges)")
    elif variance_parents:
        refine, refine_params = "grace", {}
        reasons.append(
            "refine lagged edges with GRACE (its gated model keeps variance-only "
            "parents); it drops lag-0 edges, so put the CDNOTS graph's back with "
            "causalts.inspection.merge_lag0"
        )
    else:
        refine, refine_params = "reap", {"contemporaneous": True}
        reasons.append(
            "refine with REAP (regression-based, no training; contemporaneous=True "
            "re-adds lag-0 edges)"
        )
    # Missing values: impute (VAR-EM) in both steps. The refinements need
    # complete data, and CDNOTS's nonstationarity orientation abstains on the
    # non-finite scores missing values produce.
    params = {}
    if data["missing_by_col"]:
        params["impute"] = "var_em"
        if refine is not None:
            refine_params = {**refine_params, "impute": "var_em"}
        reasons.append("missing values → impute them (VAR-EM) in every step")

    # --- C node (nonstationarity) ---
    include_C = bool(nonstationary)
    if include_C:
        c_preset = {
            "seasonal": "linear+sin",
            "curved": "linear+quad",
        }.get(form, "linear")
        reasons.append(
            f"nonstationary ({stat['fraction_nonstationary']:.0%} of cols, "
            f"form={form}) → include C node ({c_preset})"
        )
    else:
        c_preset = "linear"

    # --- latent confounding (advisory only) ---
    # Deliberately does NOT influence algorithm/ci_test/include_C: LUCID is orthogonal
    # post-discovery correction, not an alternative discovery algorithm. And the nudge
    # fires only on a positive detection -- `detected` False/None cannot rule out
    # sparse (few-variable) confounding, so silence is the honest default.
    lf = facts.get("latent_factor") or {}
    if lf.get("detected") is True:
        reasons.append(
            f"latent factor detected (R={lf['spectral_ratio']:.2f} vs "
            f"tau={lf['tau']:.2f}) → consider res.deconfound() after discovery"
        )

    return {
        "algorithm": "cdnots",
        "ci_test": ci_test,
        "refine": refine,
        "refine_params": refine_params,
        "include_C": include_C,
        "c_preset": c_preset,
        "max_lag": int(max_lag),
        "params": params,
        "rationale": "; ".join(reasons),
    }


def _cost_class(algorithm, ci_test, d, T):
    if (
        d > _HIGH_DIM
        or ci_test in _EXPENSIVE_CI_TESTS
        or algorithm in _EXPENSIVE_ALGORITHMS
    ):
        return "expensive"
    if d > 8 or ci_test in _MODERATE_CI_TESTS:
        return "moderate"
    return "cheap"


def inspect_df(df, max_lag=None):
    """Inspect an in-memory time-series DataFrame and recommend a configuration.

    Parameters
    ----------
    df : pandas.DataFrame
        Time series of shape (T, d); columns are variables.
    max_lag : int, optional
        Override the suggested max lag; otherwise it is inferred from the
        partial autocorrelation (see :func:`_suggested_max_lag`).

    Returns
    -------
    dict
        ``{schema_version, data, facts, recommendation, cost_class, warnings}``
        (see module docstring / DESIGN Q5 for the contract).

    Raises
    ------
    NoNumericColumnsError
        If no numeric column is left once non-numeric columns (dates, labels,
        timedeltas) are dropped. Subclass of ``ValueError``.
    """
    # Booleans become 0/1 (numeric_frame). Any other non-numeric column (a
    # date, a label) is left out of everything, since discrete detection fails
    # on strings and discovery cannot read it either, and named in a warning.
    df = numeric_frame(df)
    keep = df.select_dtypes(include=[np.number], exclude=["timedelta"]).columns
    non_numeric = [str(c) for c in df.columns if c not in keep]
    if non_numeric:
        df = df[keep]
    if df.shape[1] == 0:
        raise NoNumericColumnsError(
            f"inspect found no numeric columns (non-numeric, left out: "
            f"{non_numeric}). Encode or convert them (e.g. strip thousands "
            f"separators)."
        )
    data = _data_block(df)

    # Facts are measured on the numeric, non-constant columns.
    numeric = df.select_dtypes(include=[np.number])
    arr = numeric.to_numpy(dtype=np.float64)

    lin = check_linearity(numeric)
    stat = check_stationarity_observed(numeric)
    form = detect_trend_form(numeric)
    suggested = max_lag if max_lag is not None else _suggested_max_lag(arr)

    facts = {
        "linearity": {
            "fraction_nonlinear": float(lin["fraction_nonlinear"]),
            "is_nonlinear": bool(lin["fraction_nonlinear"] > 0.2),
            "n_tested": int(lin["n_tested"]),
        },
        "stationarity": {
            "fraction_nonstationary": float(stat["fraction_nonstationary"]),
            "nonstationary_cols": [str(c) for c in stat["nonstationary_cols"]],
            "form": form,
        },
        "suggested_max_lag": int(suggested),
        "latent_factor": _latent_factor_block(arr),
        "variance_dependence": _variance_dependence_block(
            arr, [str(c) for c in numeric.columns], suggested
        ),
    }

    recommendation = recommend_config(facts, data)
    cost_class = _cost_class(
        recommendation["algorithm"],
        recommendation["ci_test"],
        data["n_vars"],
        data["n_rows"],
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "data": data,
        "facts": facts,
        "recommendation": recommendation,
        "cost_class": cost_class,
        "warnings": _warnings(df, data, non_numeric)
        + (
            [
                "Linearity could not be tested (no RESET test gave a p-value), so "
                "the data were not checked for nonlinearity."
            ]
            # With fewer than two columns there is no pair to test (the
            # rationale says so); this warning is for tests that failed.
            if lin["n_tested"] == 0 and data["n_vars"] >= 2
            else []
        ),
    }


_MIXED_UNSUPPORTED = (
    "ci_test='cmiknn-mixed-gpu' is not supported for algorithm={algorithm!r}, "
    "which does not track which columns are discrete; use 'cdnots', 'cdnots+', "
    "'grace' or 'reap'."
)


def discover_df(
    df,
    algorithm="cdnots",
    ci_test=None,
    max_lag=3,
    include_C=False,
    c_preset="linear",
    alpha=None,
    device=None,
    **kwargs,
):
    """Run causal discovery on an in-memory DataFrame (Python twin of the CLI).

    Dispatches to the chosen algorithm, constructing the CI test by name, and
    returns the algorithm's result object (all expose ``.cg_tig``). Extra
    keyword args pass through to the underlying ``run_*`` function. For files,
    prefer the CLI; this is the no-temp-file path for notebooks/sessions.

    GRACE, GRACE-SS and REAP build their CDNOTS candidate graph on their own
    (transformed) data, so for them a named ``ci_test`` is turned into the
    ``ci_test_class`` factory they call, as the CLI does.

    To run a recommendation from :func:`recommend_config`, call
    ``discover_df(df, algorithm=rec["algorithm"], ci_test=rec["ci_test"], ...)``
    for the CDNOTS graph, or ``algorithm=rec["refine"], **rec["refine_params"]``
    (with the same ``ci_test``, ``max_lag``, ``include_C`` and ``c_preset``) to
    build that same CDNOTS candidate and refine it in one call. Skip the
    refinement when ``rec["refine"]`` is None (``max_lag`` 0).

    Parameters
    ----------
    ci_test : str, optional
        Registered CI-test name (see :func:`~causalts.ci_tests.list_ci_tests`).
        ``None`` (default) means ``"parcorr-gpu"`` for CDNOTS/Cedar, and each
        refinement's own default (partial correlation) for GRACE/REAP.
    alpha : float, optional
        Significance level. ``None`` (default) means "use the algorithm's own
        default" — 0.05 for CDNOTS and GRACE's skeleton, 0.01 for CDNOTS+ and
        Cedar's ``alpha_cond1`` / ``alpha_cond2``. Passing a float applies it
        to all of them.
    device : str, optional
        Device of the CI test (``None``, the default, means ``"cpu"``). For GRACE
        and GRACE-SS it is also passed on for the gated model; ``None`` lets them
        pick (CUDA if available).
    include_C : bool, default False
        Add the C nonstationarity node. Applied uniformly here, so the default
        is off for every algorithm — note that calling
        :func:`~causalts.grace.gated_discovery.run_cdnots_gated` directly
        defaults it to True instead. To put C into GRACE's gated model as well
        as its skeleton, pass ``include_C_in_model=True`` through ``**kwargs``.

    Raises
    ------
    ValueError
        If ``algorithm`` is unknown; if both ``ci_test`` and ``ci_test_class``
        are passed; if ``ci_test`` is passed to GRACE-SS with
        ``use_ci_skeleton=False``; or if ``ci_test="cmiknn-mixed-gpu"`` is used
        with CEDAR or GRACE-SS, which do not track discrete columns.
    """
    df = numeric_frame(df)
    test_device = device or "cpu"
    # dfcit sizes its block-shuffle null from alpha (so p < alpha is reachable).
    # Mirror how each engine resolves its thresholds below (unset: engine default).
    if algorithm == "cedar":
        # Cedar's only unconditional tests are Cond1, judged at alpha_cond1.
        ci_alphas = [kwargs.get("alpha_cond1", alpha if alpha is not None else 0.01)]
    elif algorithm == "cdnots+":
        ci_alphas = [alpha if alpha is not None else 0.01, kwargs.get("alpha_mci")]
    elif algorithm == "grace-ss":
        ci_alphas = [kwargs.get("ci_alpha", alpha)]
    else:
        ci_alphas = [alpha]
    ci_alphas = [a for a in ci_alphas if a is not None]
    ci_kw = {"alpha": min(ci_alphas)} if ci_test == "dfcit" and ci_alphas else {}
    if algorithm in ("cdnots", "cdnots+"):
        from .cdnots.phase3_utils import run_cdnots, run_cdnots_plus
        from .ci_tests import create_ci_test

        fn = run_cdnots if algorithm == "cdnots" else run_cdnots_plus
        if alpha is not None:
            kwargs["alpha"] = alpha
        return fn(
            df=df,
            indep_test=create_ci_test(
                ci_test or "parcorr-gpu", df.values, device=test_device, **ci_kw
            ),
            num_lags=max_lag,
            include_C=include_C,
            c_preset=c_preset,
            **kwargs,
        )
    if algorithm == "cedar":
        from .cedar.discovery import run_cedar
        from .ci_tests import create_ci_test

        if ci_test == "cmiknn-mixed-gpu":
            raise ValueError(_MIXED_UNSUPPORTED.format(algorithm=algorithm))

        if alpha is not None:
            kwargs.setdefault("alpha_cond1", alpha)
            kwargs.setdefault("alpha_cond2", alpha)
        return run_cedar(
            df=df,
            ci_test=create_ci_test(
                ci_test or "parcorr-gpu", df.values, device=test_device, **ci_kw
            ),
            max_lag=max_lag,
            include_C=include_C,
            c_preset=c_preset,
            **kwargs,
        )
    if algorithm in ("grace", "grace-ss", "reap"):
        from .ci_tests import create_ci_test
        from .grace.gated_discovery import run_cdnots_gated, run_stability_selection
        from .grace.runners import run_reap

        # The refinements instantiate the candidate-graph test themselves, on
        # their transformed data: hand them a factory for the named test.
        if ci_test is not None:
            if "ci_test_class" in kwargs:
                raise ValueError(
                    "pass either ci_test (a name) or ci_test_class, not both"
                )
            if algorithm == "grace-ss":
                if ci_test == "cmiknn-mixed-gpu":
                    # Its CI skeleton rank-transforms every column first, which
                    # hides the discrete ones from the test.
                    raise ValueError(_MIXED_UNSUPPORTED.format(algorithm=algorithm))
                # GRACE-SS only builds a candidate graph (and so calls the test)
                # with its CI skeleton.
                if kwargs.get("use_ci_skeleton") is False:
                    raise ValueError(
                        "ci_test is only used by GRACE-SS's CI skeleton; drop "
                        "use_ci_skeleton=False or ci_test."
                    )
                kwargs["use_ci_skeleton"] = True
            name = ci_test
            kwargs["ci_test_class"] = lambda data, **_: create_ci_test(
                name, data, device=test_device, **ci_kw
            )
        if algorithm in ("grace", "grace-ss") and device is not None:
            kwargs.setdefault("device", device)
        if algorithm in ("grace", "reap"):
            if alpha is not None:
                kwargs["alpha"] = alpha
        else:
            if alpha is not None:
                kwargs.setdefault("ci_alpha", alpha)
            # grace-ss reaches C only through its CI skeleton; without one
            # there is nowhere to put the node.
            if include_C and not kwargs.get("use_ci_skeleton", False):
                raise ValueError(
                    "include_C=True requires use_ci_skeleton=True for "
                    "algorithm='grace-ss' (stability selection has no C node of "
                    "its own). Pass use_ci_skeleton=True, or use "
                    "algorithm='grace'."
                )
        fn = {
            "grace": run_cdnots_gated,
            "grace-ss": run_stability_selection,
            "reap": run_reap,
        }[algorithm]
        return fn(
            df,
            max_lag=max_lag,
            include_C=include_C,
            c_preset=c_preset,
            **kwargs,
        )
    raise ValueError(f"unknown algorithm {algorithm!r}")
