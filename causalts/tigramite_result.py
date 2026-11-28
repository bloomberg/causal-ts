# Copyright 2026 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Result object and named runners for the tigramite wrappers.

:class:`TigramiteResult` wraps what :func:`~causalts.tigramite_discovery.run_pcmci`,
:func:`~causalts.tigramite_discovery.run_pcmciplus` and
:func:`~causalts.tigramite_discovery.run_lpcmci` return (with
``return_result=True``) so they get plotting, :meth:`to_marks` and the DoWhy
bridge like every other :class:`~causalts.result.CausalResult`.

:func:`run_tigramite_algorithm` runs them by name (``"pcmci"``,
``"pcmciplus"``, ``"lpcmci"``) for ``causal-ts discover --algorithm`` and
:func:`~causalts.inspection.discover_df`. Importing this module does not
import tigramite (an optional extra); running an algorithm without it raises
``ImportError``.
"""

from __future__ import annotations

import numbers
import warnings

import numpy as np
import pandas as pd

from .result import CausalResult, _resolve_var_names

#: Algorithm name -> (wrapper in causalts.tigramite_discovery, the wrapper
#: parameter a generic ``alpha`` sets).
_TIGRAMITE_METHODS = {
    "pcmci": ("run_pcmci", "alpha_level"),
    "pcmciplus": ("run_pcmciplus", "pc_alpha"),
    "lpcmci": ("run_lpcmci", "pc_alpha"),
}

#: Marks ``cg_tig`` represents faithfully (no link, or a definite arrow).
#: Every other non-empty mark is unresolved (o-o, x-x, o->, <->, and LPCMCI's
#: endpoint conflicts such as x->), as in the DoWhy bridge's check.
_DEFINITE_MARKS = ("", "-->", "<--")

#: Smallest level in the grid tigramite searches when ``pc_alpha=None``
#: (PCMCI's PC stage: 0.05..0.5; PCMCI+: 0.001..0.05). LPCMCI takes a float.
_AUTO_PC_ALPHA_MIN = {"pcmci": 0.05, "pcmciplus": 0.001}

#: Names :func:`run_tigramite_algorithm` runs.
TIGRAMITE_ALGORITHMS = tuple(_TIGRAMITE_METHODS)

#: CI-test names tigramite implements natively (see
#: ``causalts.tigramite_discovery._make_tigramite_ci_test``).
TIGRAMITE_CI_TESTS = (
    "parcorr",
    "cmiknn",
    "gpdc",
    "gpdc_torch",
    "robust_parcorr",
    "parcorr_mult",
    "cmiknn_mixed",
    "gsquared",
)


def is_tigramite_ci_test(name) -> bool:
    """Whether ``name`` is a CI test tigramite implements natively."""
    return isinstance(name, str) and (
        name.lower().replace("-", "_") in TIGRAMITE_CI_TESTS
    )


def unresolved_marks(marks, names, present=None) -> list[dict]:
    """Every adjacency ``marks`` leaves without a direction, once each.

    ``{"a", "b", "lag", "mark"}`` per adjacency; a lag-0 one is listed from its
    first end only. With ``present`` (a binary ``[cause, effect, lag]`` graph),
    only the adjacencies it does not hold are listed: LPCMCI's ``o->``
    rendered as an arrow is left out, while an ``o-o``, ``x-x`` or ``<->`` (an
    adjacency a latent common cause explains, no direct edge either way) is
    listed.
    """
    marks = np.asarray(marks)
    g = None if present is None else np.asarray(present).astype(bool)
    out = []
    for i, j, k in zip(*np.nonzero(~np.isin(marks, _DEFINITE_MARKS))):
        if k == 0 and i > j:
            continue
        if g is not None and i < g.shape[0] and j < g.shape[1] and k < g.shape[2]:
            if g[i, j, k] or (k == 0 and g[j, i, 0]):
                continue
        out.append(
            {"a": names[i], "b": names[j], "lag": int(k), "mark": str(marks[i, j, k])}
        )
    return out


class TigramiteResult(CausalResult):
    """Result of a tigramite run (PCMCI, PCMCI+, LPCMCI).

    Inherits plotting and the DoWhy bridge from
    :class:`~causalts.result.CausalResult`.

    Attributes
    ----------
    cg_tig : np.ndarray, shape (d, d, tau_max+1)
        Binary graph, ``cg_tig[cause, effect, lag] == 1`` meaning
        ``cause(t-lag) -> effect(t)`` (the wrapper's ``G_hat``).
    var_names : list[str]
        Variable names.
    method : str
        ``"pcmci"``, ``"pcmciplus"`` or ``"lpcmci"``.
    p_matrix, val_matrix : np.ndarray, shape (d, d, tau_max+1)
        Tigramite's p-values and test statistics.
    graph_raw : np.ndarray of str, shape (d, d, tau_max+1)
        Tigramite's edge marks (``-->``, ``o-o``, ``x-x``, ``o->``, ``<->``).
        :meth:`to_marks` returns a copy and :meth:`plot` draws it, so
        unoriented and conflicting links stay visible although ``cg_tig``
        leaves lag-0 ``o-o``/``x-x`` out. Read-only alias of
        :attr:`graph_marks`, so it follows :meth:`tetrad_filter` /
        :meth:`pds_filter`; ``info["graph_raw"]`` stays the unfiltered output.
    info : dict
        The wrapper's info dict.
    bridge_policy : str
        ``"refuse"``: the DoWhy bridge (``estimate_effect``, ``fit_scm``, ...)
        raises while the marks hold unresolved links (``o-o``, ``x-x``,
        ``o->``, ``<->``). Set it to ``"drop_unresolved"`` to bridge without
        them.

    Notes
    -----
    :meth:`tetrad_filter`, :meth:`pds_filter` and :meth:`deconfound` work on
    ``cg_tig``. A lag-0 ``o-o`` (PCMCI+'s unorientable pair) is not in
    ``cg_tig``: they warn that it is not tested, and the filters keep it in
    the marks. PAG-specific marks (``o->``, ``<-o``, ``<->``, ``x-x``,
    endpoint conflicts such as ``x->``, a lagged ``o-o``) make them raise
    ``ValueError``.
    """

    def __init__(self, cg_tig, df, info):
        self.cg_tig = np.asarray(cg_tig)
        self.var_names = list(df.columns)
        self._df = df
        self.info = dict(info)
        self.method = self.info.get("method")
        self.p_matrix = self.info.get("p_matrix")
        self.val_matrix = self.info.get("val_matrix")
        # to_marks / to_binary / plot read the marks from here; graph_raw is
        # an alias, so filters (tetrad_filter, ...) update both.
        marks = self.info.get("graph_raw")
        if marks is not None:
            marks = np.asarray(marks)
            if marks.dtype.kind == "S":
                marks = marks.astype(str)
        self.graph_marks = marks
        # cg_tig renders o-> as directed and drops lag-0 o-o / x-x and <->:
        # the DoWhy bridge must not read it as a DAG.
        self.bridge_policy = "refuse"
        self._scm_cache = {}

    @property
    def bridge_policy(self):
        """How the DoWhy bridge treats unresolved marks: ``"refuse"`` (default),
        ``"drop_unresolved"`` or ``None``. Changing it clears the fitted-SCM cache,
        so the next ``fit_scm()`` / ``counterfactual()`` uses the new policy."""
        return self._bridge_policy

    @bridge_policy.setter
    def bridge_policy(self, value):
        self._bridge_policy = value
        cache = getattr(self, "_scm_cache", None)
        if cache:
            cache.clear()

    @property
    def graph_raw(self):
        """Tigramite's edge marks (alias of :attr:`graph_marks`)."""
        return self.graph_marks

    # ------------------------------------------------------------------
    # Rendering: the marks are the full graph, cg_tig its binary rendering
    # ------------------------------------------------------------------

    def plot(self, var_names=None, undirected=None, **kwargs):
        """Draw the discovered graph.

        Left unset, ``undirected`` draws tigramite's marks (:meth:`to_marks`),
        so unresolved links stay visible. ``"bidirected"`` or ``"drop"``
        draw :meth:`to_binary` with that policy.
        """
        if undirected is None and self.graph_marks is not None:
            from .plotting._core import plot_graph

            return plot_graph(
                graph=self.to_marks(),
                var_names=_resolve_var_names(
                    var_names, getattr(self, "var_names", None)
                ),
                **kwargs,
            )
        return super().plot(var_names=var_names, undirected=undirected, **kwargs)

    def to_marks(self):
        """Tigramite's edge marks (a copy), ``o-o``, ``x-x``, ``o->``, ``<->`` included."""
        if self.graph_marks is None:
            return super().to_marks()
        return np.array(self.graph_marks, copy=True)

    def to_binary(self, undirected=None, conflict=None):
        """Binary ``[cause, effect, lag]`` rendering of the discovered graph.

        With no arguments this returns :attr:`cg_tig`. Otherwise the lag-0
        ``o-o`` (``undirected``) and ``x-x`` (``conflict``, default: as
        ``undirected``) pairs are rendered from the marks as ``"bidirected"``
        (symmetric 1s) or ``"drop"``; every other cell comes from ``cg_tig``.
        """
        if undirected is None and conflict is None:
            return self.cg_tig
        if self.graph_marks is None:
            return super().to_binary(undirected=undirected, conflict=conflict)
        undirected = undirected or "bidirected"
        if undirected not in ("bidirected", "drop"):
            raise ValueError(
                f"undirected must be 'bidirected' or 'drop', got {undirected!r}"
            )
        if conflict is not None and conflict not in ("drop", "bidirected"):
            raise ValueError(
                f"conflict must be 'drop' or 'bidirected', got {conflict!r}"
            )
        from .tigramite_discovery import tigramite_graph_to_binary

        marks = np.asarray(self.graph_marks)
        rendered = tigramite_graph_to_binary(
            marks, undirected=undirected, conflict=conflict or undirected
        )
        out = np.array(self.cg_tig, copy=True)
        pair = np.isin(marks[:, :, 0], ("o-o", "x-x"))
        out[:, :, 0][pair] = np.asarray(rendered)[:, :, 0][pair]
        return out

    # ------------------------------------------------------------------
    # DoWhy bridge
    # ------------------------------------------------------------------

    def _bridge(self):
        from .effects.wrap import wrap_graph

        self._require_complete_data("The DoWhy bridge")
        return wrap_graph(self._bridge_graph(self.cg_tig), self._df, self.var_names)

    def _bridge_graph(self, g):
        """``g`` as the DoWhy bridge should read it, under :attr:`bridge_policy`."""
        policy = self.bridge_policy
        if policy is None:
            return g
        if policy not in ("refuse", "drop_unresolved"):
            raise ValueError(
                f"bridge_policy must be None, 'refuse' or 'drop_unresolved', "
                f"got {policy!r}"
            )
        g = np.array(g, copy=True)
        d = g.shape[0]
        cells = [
            (i, j, 0)
            for i in range(d)
            for j in range(d)
            if g[i, j, 0] and g[j, i, 0] and i < j
        ]
        marks = np.asarray(self.to_marks())[:d, :d, : g.shape[2]]
        cells += [
            (int(i), int(j), int(k))
            for i, j, k in zip(*np.nonzero(~np.isin(marks, _DEFINITE_MARKS)))
            if k > 0 or i < j
        ]
        # A symmetric pair usually also carries an unresolved mark: count once.
        cells = list(dict.fromkeys(cells))
        if not cells:
            return g
        if policy == "refuse":
            names = list(self.var_names)
            shown = ", ".join(f"{names[i]}-{names[j]}@lag{k}" for i, j, k in cells[:5])
            raise ValueError(
                f"{len(cells)} unresolved edge(s) ({shown}"
                f"{', ...' if len(cells) > 5 else ''}): the DoWhy bridge would "
                f"pick a direction by column order or drop them silently. Set "
                f"result.bridge_policy = 'drop_unresolved' to bridge without "
                f"them, or orient them first."
            )
        for i, j, k in cells:
            g[i, j, k] = 0
            if k == 0:
                g[j, i, 0] = 0
        return g

    # ------------------------------------------------------------------
    # Post-processing on cg_tig
    # ------------------------------------------------------------------

    def _refuse_unresolved(self, what):
        """Raise if ``what`` (a ``cg_tig`` post-processor) would misread marks."""
        self._require_complete_data(what)
        marks = self.graph_marks
        if marks is None:
            return
        marks = np.asarray(marks).astype(str)
        bad = ~np.isin(marks, _DEFINITE_MARKS)
        if marks.ndim == 3 and marks.shape[2]:
            # A lag-0 o-o (PCMCI+'s unorientable pair) is not in cg_tig: it is
            # left untested (with a warning) and kept in the marks.
            bad[:, :, 0] &= marks[:, :, 0] != "o-o"
        if bad.any():
            found = sorted({str(m) for m in marks[bad]})
            n = int(bad.sum())
            raise ValueError(
                f"{what} works on the binary graph cg_tig, which reads o-> as a "
                "definite parent and cannot hold <->, x-x or conflict marks, but "
                f"this {self.method or 'tigramite'} result has {n} such marks "
                f"({', '.join(found)}; see to_marks()). It runs on results whose "
                "marks are --> or lag-0 o-o."
            )

    def _warn_if_unseen_pairs(self, what, kept=True):
        """Warn when the marks hold adjacencies ``cg_tig`` does not."""
        if self.graph_marks is None:
            return
        unresolved = unresolved_marks(
            self.graph_marks, list(self.var_names), present=self.cg_tig
        )
        if unresolved:
            pairs = ", ".join(
                f"{u['a']} {u['mark']} {u['b']} (lag {u['lag']})"
                for u in unresolved[:5]
            )
            more = f" and {len(unresolved) - 5} more" if len(unresolved) > 5 else ""
            fate = (
                "are neither tested nor removed and stay in to_marks()"
                if kept
                else "are not tested and are absent from the returned result"
            )
            warnings.warn(
                f"{what} reads cg_tig, which holds only the directed edges: the "
                f"unresolved adjacencies {pairs}{more} {fate}.",
                UserWarning,
                stacklevel=3,
            )

    def _require_complete_data(self, what):
        """The bridge and the filters fit models on ``_df``: it must be complete."""
        df = getattr(self, "_df", None)
        if df is not None and pd.DataFrame(df).isna().to_numpy().any():
            raise ValueError(
                f"{what} needs complete data, but this result's data has missing "
                "values. Fill them first (e.g. causalts.imputation.var_em_impute) "
                "and run discovery on the filled data, or run "
                "run_tigramite_algorithm with impute='var_em'."
            )

    def tetrad_filter(self, threshold=0.25):
        self._refuse_unresolved("tetrad_filter()")
        self._warn_if_unseen_pairs("tetrad_filter()")
        return super().tetrad_filter(threshold=threshold)

    def pds_filter(self, alpha=1e-10):
        self._refuse_unresolved("pds_filter()")
        self._warn_if_unseen_pairs("pds_filter()")
        return super().pds_filter(alpha=alpha)

    def deconfound(self, **kwargs):
        self._refuse_unresolved("deconfound()")
        self._warn_if_unseen_pairs("deconfound()", kept=False)
        return super().deconfound(**kwargs)

    tetrad_filter.__doc__ = CausalResult.tetrad_filter.__doc__
    pds_filter.__doc__ = CausalResult.pds_filter.__doc__
    deconfound.__doc__ = CausalResult.deconfound.__doc__

    def _with_graph(self, graph):
        """Shallow copy carrying a *filtered* graph, with the marks to match.

        A directed edge the filter removed goes from the marks too (with its
        lag-0 mirror). Unresolved pairs are not in ``cg_tig``, so a filter
        cannot reach them and their marks stay.
        """
        new = super()._with_graph(graph)
        old, now = np.asarray(self.cg_tig), np.asarray(new.cg_tig)
        if self.graph_marks is None or old.shape != now.shape:
            return new
        marks = np.array(self.graph_marks, copy=True)
        for i, j, k in zip(*np.nonzero(old.astype(bool) & ~now.astype(bool))):
            if marks[i, j, k] == "-->":
                marks[i, j, k] = ""
                if k == 0 and marks[j, i, 0] == "<--":
                    marks[j, i, 0] = ""
        new.graph_marks = marks
        return new


def _check_var_em_kwargs(algorithm, impute_kwargs):
    """Unknown ``impute_kwargs`` for ``var_em_impute`` raise ``ValueError``."""
    import inspect

    from .imputation import var_em_impute

    allowed = set(inspect.signature(var_em_impute).parameters) - {"df"}
    unknown = set(impute_kwargs) - allowed
    if unknown:
        raise ValueError(
            f"{algorithm}: impute='var_em' takes impute_kwargs "
            f"{sorted(allowed)}, got unknown {sorted(unknown)}"
        )


def _min_threshold(td, fn_name, algorithm, alpha_key, alpha, kw):
    """Smallest significance level the wrapper will test against, or None.

    ``alpha`` (if given) sets ``alpha_key``; other levels come from ``kw`` or
    the wrapper's defaults. PCMCI tests at both ``pc_alpha`` and
    ``alpha_level``; a list of ``pc_alpha`` candidates counts by its minimum,
    and ``pc_alpha=None`` by the minimum of tigramite's automatic grid.
    """
    import inspect

    params = inspect.signature(getattr(td, fn_name)).parameters
    keys = ("pc_alpha", "alpha_level") if algorithm == "pcmci" else ("pc_alpha",)
    levels = []
    for key in keys:
        if key == alpha_key and alpha is not None:
            val = alpha
        elif key in kw:
            val = kw[key]
        elif key in params:
            val = params[key].default
        else:
            continue
        if val is None and key == "pc_alpha":
            val = _AUTO_PC_ALPHA_MIN.get(algorithm)
        # Python or NumPy numbers, alone or in a list/array; skip NaN/inf.
        if isinstance(val, (list, tuple, np.ndarray)):
            cands = np.ravel(np.asarray(val, dtype=object))
        else:
            cands = [val]
        for v in cands:
            if (
                isinstance(v, numbers.Real)
                and not isinstance(v, (bool, np.bool_))
                and np.isfinite(float(v))
            ):
                levels.append(float(v))
    return min(levels) if levels else None


def run_tigramite_algorithm(algorithm, df, max_lag=1, ci_test=None, alpha=None, **kw):
    """Run a tigramite wrapper by name and return a :class:`TigramiteResult`.

    Parameters
    ----------
    algorithm : str
        One of :data:`TIGRAMITE_ALGORITHMS`.
    df : pd.DataFrame
        Time series data, shape ``(T, d)``, finite, and complete unless
        ``impute="var_em"``.
    max_lag : int
        The wrapper's ``tau_max``.
    ci_test : str, object or None
        ``None`` (default) is tigramite's own ``"parcorr"``. A name in
        :data:`TIGRAMITE_CI_TESTS` runs tigramite's test of that name (this
        includes ``"parcorr"``, ``"cmiknn"`` and ``"gsquared"``, which
        causal-ts also registers). Any other string is a causal-ts CI test
        (:func:`~causalts.ci_tests.create_ci_test`, built with ``device`` if
        given and with ``ci_kwargs``), driven through ``CITestAdapter``.
        Objects pass through. ``"cmiknn_mixed"`` is refused: it needs a
        data-type mask the wrappers do not pass to tigramite.
    alpha : float or None
        Significance level: ``alpha_level`` for PCMCI, ``pc_alpha`` for
        PCMCI+ and LPCMCI. ``None`` keeps the wrapper's default.
    **kw
        Passed to the wrapper (``tau_min``, ``fdr_method``, ...). ``device``
        is used only to build a causal-ts CI test. ``impute="var_em"`` (with
        optional ``impute_kwargs`` for
        :func:`~causalts.imputation.var_em_impute`) fills missing values here
        before the run; the result then holds the filled data.

    Raises
    ------
    ValueError
        Unknown ``algorithm``, non-finite or (without ``impute="var_em"``)
        missing data, or ``alpha`` given together with the parameter it sets.
    ImportError
        If tigramite is not installed, or gpytorch for ``"gpdc_torch"``.
    """
    from . import tigramite_discovery as td

    if algorithm not in _TIGRAMITE_METHODS:
        raise ValueError(
            f"unknown algorithm {algorithm!r}; expected one of {TIGRAMITE_ALGORITHMS}"
        )
    fn_name, alpha_key = _TIGRAMITE_METHODS[algorithm]
    td._require_tigramite(algorithm)
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(np.asarray(df))
    values = df.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError(
            f"{algorithm} needs finite data, but df has "
            f"{int(np.isinf(values).sum())} inf values."
        )
    # The wrappers take complete data: fill here (var_em) or refuse NaN.
    impute = kw.pop("impute", None)
    impute_kwargs = dict(kw.pop("impute_kwargs", None) or {})
    if impute not in (None, "var_em"):
        raise ValueError(
            f"{algorithm}: impute={impute!r} is not supported; it takes "
            "impute='var_em' only."
        )
    if impute is None and impute_kwargs:
        raise ValueError(f"{algorithm}: impute_kwargs given without impute='var_em'.")
    filled = False
    if impute == "var_em":
        _check_var_em_kwargs(algorithm, impute_kwargs)
        if np.isnan(values).any():
            from .imputation import var_em_impute

            df = var_em_impute(df, **impute_kwargs)
            values = df.to_numpy(dtype=float)
            filled = True
    if np.isnan(values).any():
        raise ValueError(
            f"{algorithm} needs complete data, but df has "
            f"{int(np.isnan(values).sum())} NaN values. Pass impute='var_em' "
            "(CLI: --impute var_em)."
        )
    device = kw.pop("device", None)
    kw.pop("return_result", None)  # always a TigramiteResult here
    if isinstance(ci_test, str) and ci_test.lower().replace("-", "_") == "gpdc_torch":
        import importlib.util

        if importlib.util.find_spec("gpytorch") is None:
            # Not part of the tigramite extra (tigramite imports it lazily).
            raise ModuleNotFoundError(
                f"{algorithm}: ci_test='gpdc_torch' (tigramite's GPDCtorch) needs "
                "gpytorch, which the tigramite extra does not install. Install "
                "with: pip install gpytorch",
                name="gpytorch",
            )
    if ci_test is None:
        ci_test = "parcorr"
    elif isinstance(ci_test, str) and ci_test.lower().replace("-", "_") == (
        "cmiknn_mixed"
    ):
        raise ValueError(
            f"{algorithm}: tigramite's cmiknn_mixed needs a data-type mask, which "
            "the causal-ts wrappers do not pass yet. For mixed data use "
            "cmiknn-mixed-gpu with cdnots, cdnots+, grace or reap."
        )
    elif isinstance(ci_test, str) and not is_tigramite_ci_test(ci_test):
        from .ci_tests import create_ci_test

        # ci_kwargs are constructor options for the test (the wrapper reads
        # them only for a name, and this name becomes an object below).
        ci_kw = dict(kw.pop("ci_kwargs", None) or {})
        if ci_test.lower() == "dfcit" and "alpha" not in ci_kw:
            # dfcit sizes its block-shuffle null and early stop from its own
            # alpha (default 0.05): give it the run's smallest threshold so
            # p < alpha is reachable (as discover_df does for other engines).
            ci_alpha = _min_threshold(td, fn_name, algorithm, alpha_key, alpha, kw)
            if ci_alpha is not None:
                ci_kw["alpha"] = ci_alpha
        ci_test = create_ci_test(ci_test, values, device=device, **ci_kw)
    if not isinstance(ci_test, str) and getattr(ci_test, "discrete_cols", None):
        # CITestAdapter hands the test a fresh (T, k) array per test, so column
        # indices into the input data (which columns are discrete) would land
        # on whatever variable sits in that slot.
        raise ValueError(
            f"{algorithm}: {type(ci_test).__name__} indexes discrete columns of "
            "the input data, which tigramite's per-test arrays do not keep. For "
            "mixed data use cmiknn-mixed-gpu with cdnots, cdnots+, grace or reap."
        )
    if alpha is not None:
        if alpha_key in kw:
            raise ValueError(
                f"pass alpha or {alpha_key} to {algorithm}, not both "
                f"(alpha sets {alpha_key})"
            )
        kw[alpha_key] = alpha
    fn = getattr(td, fn_name)
    res = fn(df, tau_max=max_lag, ci_test=ci_test, return_result=True, **kw)
    if filled:
        res.info["impute"] = "var_em"
    return res


# Runners under the plugin registry's ``fn(df=, ci_test=, max_lag=, **kwargs)``
# convention (see causalts.algorithms). ``device`` is consumed by
# run_tigramite_algorithm (to build a causal-ts CI test by name), not passed
# to the wrappers.
def run_pcmci_result(df, ci_test=None, max_lag=1, device=None, **kwargs):
    """:func:`run_tigramite_algorithm` for ``"pcmci"``."""
    return run_tigramite_algorithm(
        "pcmci", df, max_lag, ci_test=ci_test, device=device, **kwargs
    )


def run_pcmciplus_result(df, ci_test=None, max_lag=1, device=None, **kwargs):
    """:func:`run_tigramite_algorithm` for ``"pcmciplus"``."""
    return run_tigramite_algorithm(
        "pcmciplus", df, max_lag, ci_test=ci_test, device=device, **kwargs
    )


def run_lpcmci_result(df, ci_test=None, max_lag=1, device=None, **kwargs):
    """:func:`run_tigramite_algorithm` for ``"lpcmci"``."""
    return run_tigramite_algorithm(
        "lpcmci", df, max_lag, ci_test=ci_test, device=device, **kwargs
    )
