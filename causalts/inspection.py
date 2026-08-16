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
  cost_class, warnings}`` for an in-memory DataFrame.
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

from .cdnots.phase3_utils import _detect_discrete_cols
from .utils.linearity import check_linearity
from .utils.stationarity import check_stationarity_observed, detect_trend_form

SCHEMA_VERSION = 1

# Cost heuristics (Q6): the gate only needs "should I ask first?", not seconds.
_EXPENSIVE_CI_TESTS = {"kci", "cmiknn-gpu"}
_EXPENSIVE_ALGORITHMS = {"grace", "grace-ss"}
_HIGH_DIM = 15


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
    matrix is supplied).
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


def _warnings(df, data):
    """Deterministic data-health flags (the Q1 pre-flight surface)."""
    warns = []
    for col, frac in data["missing_by_col"].items():
        if frac >= 0.2:
            warns.append(
                f"{col} is {frac:.0%} missing — impute or drop before discovery."
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
    dict with keys ``algorithm``, ``ci_test``, ``include_C``, ``c_preset``,
    ``max_lag``, ``params``, ``rationale``.
    """
    d = data["n_vars"]
    T = data["n_rows"]
    nonlinear = facts["linearity"]["is_nonlinear"]
    stat = facts["stationarity"]
    nonstationary = len(stat["nonstationary_cols"]) > 0
    form = stat["form"]
    has_discrete = len(data["discrete_cols"]) > 0
    max_lag = facts["suggested_max_lag"]

    reasons = []

    # --- algorithm ---
    if d > 20:
        algorithm = "grace"
        reasons.append(f"high dimensionality (d={d}) → GRACE (gated, scalable)")
    else:
        algorithm = "cdnots"
        reasons.append(f"moderate dimensionality (d={d}) → CDNOTS")
        if 6 <= d <= 20:
            reasons.append("(Cedar is a good pairwise alternative)")

    # --- CI test ---
    if has_discrete:
        ci_test = "cmiknn-gpu"
        reasons.append("discrete/mixed columns → cmiknn-gpu (discrete-aware)")
    elif nonlinear and T < 300:
        ci_test = "dfcit"
        reasons.append(f"nonlinear + small T ({T}) → dfcit (high recall)")
    elif nonlinear:
        ci_test = "splitkci"
        reasons.append("nonlinear + sufficient T → splitkci (speed/F1)")
    else:
        ci_test = "parcorr-gpu"
        reasons.append("approximately linear → parcorr-gpu")

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

    return {
        "algorithm": algorithm,
        "ci_test": ci_test,
        "include_C": include_C,
        "c_preset": c_preset,
        "max_lag": int(max_lag),
        "params": {},
        "rationale": "; ".join(reasons),
    }


def _cost_class(algorithm, ci_test, d, T):
    if (
        d > _HIGH_DIM
        or ci_test in _EXPENSIVE_CI_TESTS
        or algorithm in _EXPENSIVE_ALGORITHMS
    ):
        return "expensive"
    if d > 8 or ci_test == "splitkci":
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
    """
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
        },
        "stationarity": {
            "fraction_nonstationary": float(stat["fraction_nonstationary"]),
            "nonstationary_cols": [str(c) for c in stat["nonstationary_cols"]],
            "form": form,
        },
        "suggested_max_lag": int(suggested),
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
        "warnings": _warnings(df, data),
    }


def discover_df(
    df,
    algorithm="cdnots",
    ci_test="parcorr-gpu",
    max_lag=3,
    include_C=False,
    c_preset="linear",
    alpha=None,
    device="cpu",
    **kwargs,
):
    """Run causal discovery on an in-memory DataFrame (Python twin of the CLI).

    Dispatches to the chosen algorithm, constructing the CI test by name, and
    returns the algorithm's result object (all expose ``.cg_tig``). Extra
    keyword args pass through to the underlying ``run_*`` function. For files,
    prefer the CLI; this is the no-temp-file path for notebooks/sessions.

    Every argument is honoured by every algorithm, with one exception: GRACE
    builds its CDNOTS skeleton with its own CI test, so ``ci_test`` applies to
    the CDNOTS/Cedar paths only and is rejected for GRACE rather than silently
    ignored (pass ``ci_test_class=<class>`` through ``**kwargs`` instead).

    Parameters
    ----------
    alpha : float, optional
        Significance level. ``None`` (default) means "use the algorithm's own
        default" — 0.05 for the CDNOTS family and GRACE's skeleton, 0.01 for
        Cedar's ``alpha_cond1`` / ``alpha_cond2``. Passing a float applies it
        to all of them.
    include_C : bool, default False
        Add the C nonstationarity node. Applied uniformly here, so the default
        is off for every algorithm — note that calling
        :func:`~causalts.grace.gated_discovery.run_cdnots_gated` directly
        defaults it to True instead. To put C into GRACE's gated model as well
        as its skeleton, pass ``include_C_in_model=True`` through ``**kwargs``.

    Raises
    ------
    ValueError
        If ``algorithm`` is unknown, or if a non-default ``ci_test`` is
        requested for GRACE.
    """
    if algorithm in ("cdnots", "cdnots+"):
        from .cdnots.phase3_utils import run_cdnots, run_cdnots_plus
        from .ci_tests import create_ci_test

        fn = run_cdnots if algorithm == "cdnots" else run_cdnots_plus
        if alpha is not None:
            kwargs["alpha"] = alpha
        return fn(
            df=df,
            indep_test=create_ci_test(ci_test, df.values, device=device),
            num_lags=max_lag,
            include_C=include_C,
            c_preset=c_preset,
            **kwargs,
        )
    if algorithm == "cedar":
        from .cedar.discovery import run_cedar
        from .ci_tests import create_ci_test

        if alpha is not None:
            kwargs.setdefault("alpha_cond1", alpha)
            kwargs.setdefault("alpha_cond2", alpha)
        return run_cedar(
            df=df,
            ci_test=create_ci_test(ci_test, df.values, device=device),
            max_lag=max_lag,
            include_C=include_C,
            c_preset=c_preset,
            **kwargs,
        )
    if algorithm in ("grace", "grace-ss"):
        from .grace.gated_discovery import run_cdnots_gated, run_stability_selection

        # GRACE instantiates a CI test class itself for the skeleton; a
        # by-name instance has nowhere to go. Refuse rather than drop it.
        if ci_test != "parcorr-gpu" and "ci_test_class" not in kwargs:
            raise ValueError(
                f"ci_test={ci_test!r} is not supported for {algorithm!r}: GRACE "
                "builds its skeleton with its own CI test. Pass "
                "ci_test_class=<class> instead, or use algorithm='cdnots'."
            )
        if algorithm == "grace":
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
        fn = run_cdnots_gated if algorithm == "grace" else run_stability_selection
        return fn(
            df,
            max_lag=max_lag,
            include_C=include_C,
            c_preset=c_preset,
            **kwargs,
        )
    raise ValueError(f"unknown algorithm {algorithm!r}")
