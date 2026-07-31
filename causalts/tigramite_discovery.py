# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tigramite discovery algorithm wrappers.

Run PCMCI+, PCMCI, and other tigramite discovery algorithms and get results
in causal-ts standard format: binary (d, d, max_lag+1) graphs compatible
with all downstream features (effects, forecasting, queries).

Requires tigramite: pip install tigramite
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def tigramite_available() -> bool:
    """Check if tigramite is installed."""
    try:
        import tigramite  # noqa: F401

        return True
    except ImportError:
        return False


def _require_tigramite(feature: str = "tigramite discovery"):
    if not tigramite_available():
        raise ImportError(
            f"{feature} requires tigramite. Install with: pip install tigramite"
        )


class CITestAdapter:
    """Adapter that wraps a causal-ts CIT_Base GPU test for use with tigramite.

    Tigramite's PCMCI expects CI tests inheriting from CondIndTest with
    a `get_dependence_measure(array, xyz)` → stat interface. Causal-ts GPU
    tests (KCIGPU, SplitKCIGPU, DFCITGPU, etc.) use `__call__(X, Y, Z)` →
    (pval, stat) with column indices.

    This adapter bridges the gap: it presents a CondIndTest-compatible
    interface while delegating the actual computation to the GPU test.

    Usage::

        from causalts.ci_tests import SplitKCIGPU
        gpu_test = SplitKCIGPU()
        G_hat, info = run_pcmciplus(df, tau_max=3, ci_test=gpu_test)

    The adapter is created automatically when a CIT_Base instance is passed
    to run_pcmciplus/run_pcmci/run_tigramite.
    """

    def __init__(self, gpu_test, significance="analytic", verbosity=0):
        self._gpu_test = gpu_test
        self._cached_pval = None
        self.significance = significance
        self.verbosity = verbosity
        self.pvalue_cache = {}
        self.pvalue_local = {}
        self.n_tests = 0
        self.n_actual_tests = 0
        self.n_really_done = 0
        self.two_sided = False
        self.recycle_residuals = False
        self.confidence = None
        self.dataframe = None
        self.param_hash = "gpu_adapter"
        self._measure = f"gpu_{type(gpu_test).__name__}"

    @property
    def measure(self):
        return self._measure

    def set_dataframe(self, dataframe):
        """Called by tigramite's PCMCI to provide data."""
        self.dataframe = dataframe
        data = self._extract_data(dataframe)
        if data is not None:
            self._gpu_test.data = data

    def _extract_data(self, dataframe):
        """Extract numpy array from tigramite DataFrame."""
        values = dataframe.values
        if isinstance(values, dict):
            return values[0]
        if hasattr(values, "shape"):
            return values
        return None

    def _get_array(self, X, Y, Z=None, tau_max=0, cut_off="2xtau_max", **kwargs):
        """Construct the test array from (var, -lag) tuples.

        Mirrors tigramite's CondIndTest._get_array but simplified.
        """
        if self.dataframe is None:
            raise RuntimeError("dataframe not set — call set_dataframe first")

        data = self._extract_data(self.dataframe)
        T_full, d = data.shape

        # Determine max lag from X, Y, Z
        all_vars = list(X) + list(Y) + (list(Z) if Z else [])
        max_lag = max(abs(t[1]) for t in all_vars) if all_vars else 0

        if cut_off == "2xtau_max":
            cutoff = 2 * max(tau_max, max_lag)
        elif cut_off == "max_lag":
            cutoff = max_lag
        else:
            cutoff = max(tau_max, max_lag)

        T = T_full - cutoff

        # Build array (dim, T) and xyz
        arrays = []
        xyz_list = []

        for group_idx, group in enumerate([X, Y, Z]):
            if group is None:
                continue
            for var_idx, lag in group:
                col = data[cutoff + lag : cutoff + lag + T, var_idx]
                arrays.append(col)
                xyz_list.append(group_idx)

        array = np.array(arrays)  # (dim, T)
        xyz = np.array(xyz_list)

        # Nonzero tracking (simplified)
        nonzero_X = [i for i, g in enumerate(xyz_list) if g == 0]
        nonzero_Y = [i for i, g in enumerate(xyz_list) if g == 1]
        nonzero_Z = [i for i, g in enumerate(xyz_list) if g == 2]
        XYZ = (
            [i for i, g in enumerate(xyz_list) if g == 0],
            [i for i, g in enumerate(xyz_list) if g == 1],
            [i for i, g in enumerate(xyz_list) if g == 2] or None,
        )

        return (
            array,
            xyz,
            XYZ,
            None,
            nonzero_X,
            nonzero_Y,
            nonzero_Z if nonzero_Z else None,
            None,
        )

    def get_dependence_measure(self, array, xyz, data_type=None):
        """Run the GPU CI test and return the test statistic.

        The p-value is cached internally and returned by
        get_analytic_significance().
        """
        # array is (dim, T), xyz marks [0=X, 1=Y, 2=Z]
        x_dims = np.where(xyz == 0)[0]
        y_dims = np.where(xyz == 1)[0]
        z_dims = np.where(xyz == 2)[0]

        # Set data on GPU test as the combined (T, dim) array
        combined = np.ascontiguousarray(array.T.copy())
        self._gpu_test.data = combined

        # Call with column indices into the combined array.
        # Most GPU tests expect X and Y as single ints, Z as a list.
        n_x = len(x_dims)
        n_y = len(y_dims)
        x_cols = list(range(n_x))
        y_cols = list(range(n_x, n_x + n_y))
        z_cols = list(range(n_x + n_y, array.shape[0])) if len(z_dims) > 0 else []

        # Unwrap single-element X/Y to int (CIT_Base convention)
        if len(x_cols) == 1:
            x_cols = x_cols[0]
        if len(y_cols) == 1:
            y_cols = y_cols[0]

        pval, stat = self._gpu_test(x_cols, y_cols, z_cols)
        self._cached_pval = pval
        return stat

    def get_analytic_significance(self, value, T, dim, xyz=None):
        """Return the cached p-value from the last get_dependence_measure call."""
        if self._cached_pval is not None:
            return self._cached_pval
        return 0.0

    def get_shuffle_significance(self, array, xyz, value, **kwargs):
        """Not needed — GPU tests compute p-values internally."""
        return self._cached_pval if self._cached_pval is not None else 0.0

    def run_test(
        self,
        X,
        Y,
        Z=None,
        tau_max=0,
        cut_off="2xtau_max",
        alpha_or_thres=None,
        **kwargs,
    ):
        """Run the CI test — main entry point called by PCMCI."""
        array, xyz, XYZ, data_type, *_ = self._get_array(
            X, Y, Z, tau_max=tau_max, cut_off=cut_off
        )

        self.n_tests += 1
        self.n_actual_tests += 1

        # Check for zero-variance
        if array.shape[1] == 0:
            val, pval = 0.0, 1.0
        else:
            val = self.get_dependence_measure(array, xyz)
            pval = self._cached_pval

        if alpha_or_thres is None:
            return val, pval
        else:
            dependent = pval <= alpha_or_thres
            return val, pval, dependent

    def run_test_raw(self, x, y, z=None, **kwargs):
        """Run directly on arrays (T, d_x), (T, d_y), (T, d_z)."""
        if z is not None:
            array = np.vstack((x.T, y.T, z.T))
            xyz = np.array([0] * x.shape[1] + [1] * y.shape[1] + [2] * z.shape[1])
        else:
            array = np.vstack((x.T, y.T))
            xyz = np.array([0] * x.shape[1] + [1] * y.shape[1])

        val = self.get_dependence_measure(array, xyz)
        pval = self._cached_pval
        return val, pval

    def get_confidence(self, X, Y, Z=None, tau_max=0, **kwargs):
        """Return None — GPU tests don't provide confidence intervals."""
        return None

    def get_model_selection_criterion(self, j, parents, tau_max=0):
        """Model selection — return AIC-like score (not implemented, return 0)."""
        return 0.0

    def _get_array_hash(self, array, xyz, XYZ):
        """Hash for caching."""
        return hash((array.tobytes(), xyz.tobytes()))

    def save_incremental_cache(self, key, param_hash, pval, val):
        pass


def _make_tigramite_ci_test(ci_test, **ci_kwargs):
    """Create a tigramite CI test from a string name or pass through an object.

    Parameters
    ----------
    ci_test : str or object
        String name ("parcorr", "cmiknn", "gpdc", "robust_parcorr",
        "parcorr_mult", "gpdc_torch") or a pre-constructed tigramite
        CI test instance, or a causal-ts CIT_Base GPU test instance.
    **ci_kwargs : dict
        Keyword arguments passed to the CI test constructor (only used
        if ci_test is a string).

    Returns
    -------
    object
        A tigramite CI test instance (or CITestAdapter wrapping a GPU
        test).
    """
    if not isinstance(ci_test, str):
        # Check if it's a CIT_Base GPU test that needs wrapping
        from .ci_tests.cit_test import CIT_Base

        if isinstance(ci_test, CIT_Base):
            return CITestAdapter(ci_test)
        return ci_test

    name = ci_test.lower().replace("-", "_")

    if name == "parcorr":
        from tigramite.independence_tests.parcorr import ParCorr

        return ParCorr(**ci_kwargs)
    elif name == "cmiknn":
        from tigramite.independence_tests.cmiknn import CMIknn

        return CMIknn(**ci_kwargs)
    elif name == "gpdc":
        from tigramite.independence_tests.gpdc import GPDC

        return GPDC(**ci_kwargs)
    elif name == "gpdc_torch":
        from tigramite.independence_tests.gpdc_torch import GPDCtorch

        return GPDCtorch(**ci_kwargs)
    elif name == "robust_parcorr":
        from tigramite.independence_tests.robust_parcorr import RobustParCorr

        return RobustParCorr(**ci_kwargs)
    elif name == "parcorr_mult":
        from tigramite.independence_tests.parcorr_mult import ParCorrMult

        return ParCorrMult(**ci_kwargs)
    elif name == "cmiknn_mixed":
        from tigramite.independence_tests.cmiknn_mixed import CMIknnMixed

        return CMIknnMixed(**ci_kwargs)
    elif name == "gsquared":
        from tigramite.independence_tests.gsquared import Gsquared

        return Gsquared(**ci_kwargs)
    else:
        raise ValueError(
            f"Unknown CI test: {ci_test!r}. Available: parcorr, cmiknn, gpdc, "
            f"gpdc_torch, robust_parcorr, parcorr_mult, cmiknn_mixed, gsquared"
        )


def _to_tigramite_df(df: pd.DataFrame):
    """Wrap a pandas DataFrame in tigramite's DataFrame."""
    from tigramite.data_processing import DataFrame as TigramiteDF

    return TigramiteDF(df.to_numpy(dtype=float), var_names=list(df.columns))


def _tigramite_graph_to_binary(
    tig_graph: np.ndarray,
    p_matrix: np.ndarray | None = None,
    alpha_level: float | None = None,
) -> np.ndarray:
    """Convert tigramite string graph to causal-ts binary format.

    Handles all tigramite edge types:

    - ``"-->"`` : directed edge (i causes j at lag tau)
    - ``"<--"`` : reverse directed (j causes i at lag tau)
    - ``"o->"`` : partially oriented (i possibly causes j, from LPCMCI PAGs)
    - ``"<-o"`` : reverse partially oriented
    - ``"o-o"`` : unoriented (ambiguous, excluded from binary)
    - ``"x-x"`` : conflicting (excluded from binary)
    - ``"<->"`` : bidirectional / latent confounder (excluded from binary)

    Parameters
    ----------
    tig_graph : numpy.ndarray
        Shape (d, d, tau_max+1), dtype string.
    p_matrix : numpy.ndarray or None, optional
        Optional p-value matrix for additional thresholding.
    alpha_level : float or None, optional
        If set with p_matrix, only include edges with p < alpha_level.

    Returns
    -------
    numpy.ndarray
        Binary ndarray (d, d, tau_max+1), dtype int8.
    """
    d = tig_graph.shape[0]
    tau_max_plus_1 = tig_graph.shape[2]
    G = np.zeros((d, d, tau_max_plus_1), dtype=np.int8)

    for i in range(d):
        for j in range(d):
            for tau in range(tau_max_plus_1):
                s = str(tig_graph[i, j, tau]).strip()
                if s in ("-->", "o->"):
                    if p_matrix is not None and alpha_level is not None:
                        if p_matrix[i, j, tau] >= alpha_level:
                            continue
                    G[i, j, tau] = 1
                elif s in ("<--", "<-o"):
                    if p_matrix is not None and alpha_level is not None:
                        if p_matrix[i, j, tau] >= alpha_level:
                            continue
                    G[j, i, tau] = 1

    return G


def run_pcmciplus(
    df: pd.DataFrame,
    tau_max: int,
    ci_test: str | object = "parcorr",
    pc_alpha: float = 0.01,
    tau_min: int = 0,
    contemp_collider_rule: str = "majority",
    conflict_resolution: bool = True,
    fdr_method: str = "none",
    selected_links: dict | None = None,
    verbosity: int = 0,
    ci_kwargs: dict | None = None,
    **kwargs,
) -> tuple[np.ndarray, dict]:
    """Run PCMCI+ and return a binary causal graph.

    PCMCI+ (Runge 2020) discovers contemporaneous and lagged causal links
    with FDR control. Handles cyclic contemporaneous structures.

    Parameters
    ----------
    df : pandas.DataFrame
        Time series DataFrame (T, d).
    tau_max : int
        Maximum time lag to test.
    ci_test : str or object
        CI test name ("parcorr", "cmiknn", "gpdc", "robust_parcorr",
        "parcorr_mult", "gpdc_torch") or a pre-constructed tigramite
        CI test object.
    pc_alpha : float
        Significance level for the PC skeleton phase.
    tau_min : int
        Minimum lag (0 includes contemporaneous).
    contemp_collider_rule : str
        Rule for contemporaneous collider orientation. "majority"
        (default), "conservative", or "none".
    conflict_resolution : bool
        Resolve orientation conflicts (default True).
    fdr_method : str
        FDR correction method ("none", "fdr_bh", "fdr_by").
    selected_links : dict or None
        Dict restricting which links to test (tigramite format).
    verbosity : int
        0=silent, 1=summary, 2=detailed.
    ci_kwargs : dict or None
        Extra kwargs for the CI test constructor (only if ci_test is a
        string).
    **kwargs : dict
        Extra kwargs passed to pcmci.run_pcmciplus().

    Returns
    -------
    tuple
        Tuple (G_hat, info):

        - G_hat: Binary graph (d, d, tau_max+1), convention
          [cause, effect, lag].
        - info: Dict with keys: method, p_matrix, val_matrix,
          graph_raw, conf_matrix, ambiguous_triples.
    """
    _require_tigramite("PCMCI+")
    from tigramite.pcmci import PCMCI

    ci_kwargs = ci_kwargs or {}
    cit = _make_tigramite_ci_test(ci_test, **ci_kwargs)
    tig_df = _to_tigramite_df(df)

    pcmci = PCMCI(dataframe=tig_df, cond_ind_test=cit, verbosity=verbosity)

    result = pcmci.run_pcmciplus(
        tau_min=tau_min,
        tau_max=tau_max,
        pc_alpha=pc_alpha,
        contemp_collider_rule=contemp_collider_rule,
        conflict_resolution=conflict_resolution,
        fdr_method=fdr_method,
        selected_links=selected_links,
        **kwargs,
    )

    G_hat = _tigramite_graph_to_binary(result["graph"])

    info = {
        "method": "pcmciplus",
        "p_matrix": result["p_matrix"],
        "val_matrix": result["val_matrix"],
        "graph_raw": result["graph"],
        "conf_matrix": result.get("conf_matrix"),
        "ambiguous_triples": result.get("ambiguous_triples", []),
    }

    return G_hat, info


def run_pcmci(
    df: pd.DataFrame,
    tau_max: int,
    ci_test: str | object = "parcorr",
    pc_alpha: float = 0.2,
    alpha_level: float = 0.05,
    tau_min: int = 0,
    fdr_method: str = "none",
    selected_links: dict | None = None,
    verbosity: int = 0,
    ci_kwargs: dict | None = None,
    **kwargs,
) -> tuple[np.ndarray, dict]:
    """Run PCMCI and return a binary causal graph.

    PCMCI (Runge et al. 2019) discovers lagged causal links via
    PC-stable skeleton + MCI edge pruning. Does not handle contemporaneous
    links -- use run_pcmciplus for that.

    Parameters
    ----------
    df : pandas.DataFrame
        Time series DataFrame (T, d).
    tau_max : int
        Maximum time lag to test.
    ci_test : str or object
        CI test name or pre-constructed tigramite CI test object.
    pc_alpha : float
        Significance level for PC skeleton phase.
    alpha_level : float
        Significance level for MCI thresholding.
    tau_min : int
        Minimum lag (default 0, but PCMCI treats lag 0 differently
        from PCMCI+).
    fdr_method : str
        FDR correction ("none", "fdr_bh", "fdr_by").
    selected_links : dict or None
        Dict restricting which links to test.
    verbosity : int
        0=silent, 1=summary, 2=detailed.
    ci_kwargs : dict or None
        Extra kwargs for CI test constructor.
    **kwargs : dict
        Extra kwargs passed to pcmci.run_pcmci().

    Returns
    -------
    tuple
        Tuple (G_hat, info): Same format as run_pcmciplus.
    """
    _require_tigramite("PCMCI")
    from tigramite.pcmci import PCMCI

    ci_kwargs = ci_kwargs or {}
    cit = _make_tigramite_ci_test(ci_test, **ci_kwargs)
    tig_df = _to_tigramite_df(df)

    pcmci = PCMCI(dataframe=tig_df, cond_ind_test=cit, verbosity=verbosity)

    result = pcmci.run_pcmci(
        tau_min=tau_min,
        tau_max=tau_max,
        pc_alpha=pc_alpha,
        alpha_level=alpha_level,
        fdr_method=fdr_method,
        selected_links=selected_links,
        **kwargs,
    )

    G_hat = _tigramite_graph_to_binary(
        result["graph"], p_matrix=result["p_matrix"], alpha_level=alpha_level
    )

    info = {
        "method": "pcmci",
        "p_matrix": result["p_matrix"],
        "val_matrix": result["val_matrix"],
        "graph_raw": result["graph"],
        "conf_matrix": result.get("conf_matrix"),
        "alpha_level": alpha_level,
    }

    return G_hat, info


def run_lpcmci(
    df: pd.DataFrame,
    tau_max: int,
    ci_test: str | object = "parcorr",
    pc_alpha: float = 0.05,
    tau_min: int = 0,
    n_preliminary_iterations: int = 1,
    verbosity: int = 0,
    ci_kwargs: dict | None = None,
    **kwargs,
) -> tuple[np.ndarray, dict]:
    """Run LPCMCI and return a binary causal graph.

    LPCMCI (Gerhardus & Runge 2020) extends PCMCI+ to handle latent
    confounders. Outputs a Partial Ancestral Graph (PAG) with edge types
    like ``"o->"`` (possible cause) and ``"<->"`` (latent confounder).

    The binary output includes ``"o->"`` edges as directed (conservative
    estimate of possible causes). The raw PAG is preserved in
    ``info["graph_raw"]`` for inspecting edge types.

    Parameters
    ----------
    df : pandas.DataFrame
        Time series DataFrame (T, d).
    tau_max : int
        Maximum time lag to test.
    ci_test : str or object
        CI test name or instance. String names: "parcorr", "cmiknn",
        "gpdc", etc. Also accepts causal-ts GPU tests (SplitKCIGPU,
        KCIGPU, etc.).
    pc_alpha : float
        Significance level for skeleton phase.
    tau_min : int
        Minimum lag (0 includes contemporaneous).
    n_preliminary_iterations : int
        Number of preliminary rule iterations before the final
        orientation (default 1).
    verbosity : int
        0=silent, 1=summary, 2=detailed.
    ci_kwargs : dict or None
        Extra kwargs for CI test constructor.
    **kwargs : dict
        Extra kwargs passed to lpcmci.run_lpcmci().

    Returns
    -------
    tuple
        Tuple (G_hat, info):

        - G_hat: Binary graph (d, d, tau_max+1). Includes both
          ``"-->"`` and ``"o->"`` edges from the PAG.
        - info: Dict with keys: method, p_matrix, val_matrix,
          graph_raw, edge_types.
    """
    _require_tigramite("LPCMCI")
    from tigramite.lpcmci import LPCMCI

    ci_kwargs = ci_kwargs or {}
    cit = _make_tigramite_ci_test(ci_test, **ci_kwargs)
    tig_df = _to_tigramite_df(df)

    lpcmci = LPCMCI(dataframe=tig_df, cond_ind_test=cit, verbosity=verbosity)

    result = lpcmci.run_lpcmci(
        tau_min=tau_min,
        tau_max=tau_max,
        pc_alpha=pc_alpha,
        n_preliminary_iterations=n_preliminary_iterations,
        **kwargs,
    )

    G_hat = _tigramite_graph_to_binary(result["graph"])

    # Count edge types for diagnostics
    raw = result["graph"]
    unique, counts = np.unique(raw[raw != ""], return_counts=True)
    edge_types = dict(zip(unique.tolist(), counts.tolist()))

    info = {
        "method": "lpcmci",
        "p_matrix": result["p_matrix"],
        "val_matrix": result["val_matrix"],
        "graph_raw": result["graph"],
        "edge_types": edge_types,
    }

    return G_hat, info


_METHOD_MAP = {
    "pcmciplus": "run_pcmciplus",
    "pcmci": "run_pcmci",
    "lpcmci": "run_lpcmci",
    "pcalg": "run_pcalg",
    "pc_stable": "run_pc_stable",
    "fullci": "run_fullci",
    "bivci": "run_bivci",
}


def run_rpcmci(
    df: pd.DataFrame,
    tau_max: int,
    num_regimes: int = 2,
    max_transitions: int | None = None,
    ci_test: str | object = "parcorr",
    pc_alpha: float = 0.2,
    alpha_level: float = 0.01,
    tau_min: int = 1,
    switch_thres: float = 0.05,
    num_iterations: int = 20,
    max_anneal: int = 10,
    n_jobs: int = -1,
    prediction_model=None,
    seed: int | None = None,
    verbosity: int = 0,
    ci_kwargs: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """Run RPCMCI (Regime-PCMCI) and return regime-conditional causal graphs.

    RPCMCI (Saggioro et al., Chaos 2020) iterates between regime
    assignment via constrained optimization and PCMCI causal discovery
    to find regime-dependent causal relationships.

    Requires: ``pip install tigramite ortools``

    Parameters
    ----------
    df : pandas.DataFrame
        Time series DataFrame (T, d).
    tau_max : int
        Maximum time lag to test.
    num_regimes : int
        Number of assumed regimes.
    max_transitions : int or None
        Maximum transitions per regime (persistency). If None,
        defaults to T // (4 * num_regimes).
    ci_test : str or object
        CI test name or tigramite CI test instance.
    pc_alpha : float
        Significance level for PCMCI skeleton phase.
    alpha_level : float
        Significance level for final graph thresholding.
    tau_min : int
        Minimum lag (default 1 = no contemporaneous).
    switch_thres : float
        Switch threshold for regime assignment.
    num_iterations : int
        Optimization iterations per annealing.
    max_anneal : int
        Number of annealing restarts.
    n_jobs : int
        CPUs for parallelization (-1 = all).
    prediction_model : object or None
        Sklearn-compatible model for regime prediction residuals.
        Default: LinearRegression.
    seed : int or None
        Random seed.
    verbosity : int
        0=silent, 1=summary, 2=detailed.
    ci_kwargs : dict or None
        Extra kwargs for CI test constructor.

    Returns
    -------
    tuple
        (G_hat, info):

        - G_hat: Binary union graph (d, d, tau_max+1) — edge present
          if found in ANY regime.
        - info: Dict with keys: method, regime_labels (T,),
          regime_graphs (dict regime_id -> binary graph),
          causal_results (raw PCMCI results per regime),
          num_regimes, error_free_annealings.
    """
    _require_tigramite("RPCMCI")
    try:
        from tigramite.rpcmci import RPCMCI
    except ImportError:
        raise ImportError(
            "RPCMCI requires 'ortools' in addition to tigramite. "
            "Install with: pip install tigramite ortools"
        )

    ci_kwargs = ci_kwargs or {}
    cit = _make_tigramite_ci_test(ci_test, **ci_kwargs)
    tig_df = _to_tigramite_df(df)

    T = len(df)
    if max_transitions is None:
        max_transitions = T // (4 * num_regimes)

    rpcmci = RPCMCI(
        dataframe=tig_df,
        cond_ind_test=cit,
        prediction_model=prediction_model,
        seed=seed,
        verbosity=verbosity,
    )

    output = rpcmci.run_rpcmci(
        num_regimes=num_regimes,
        max_transitions=max_transitions,
        switch_thres=switch_thres,
        num_iterations=num_iterations,
        max_anneal=max_anneal,
        tau_min=tau_min,
        tau_max=tau_max,
        pc_alpha=pc_alpha,
        alpha_level=alpha_level,
        n_jobs=n_jobs,
    )

    if output is None:
        raise RuntimeError(
            "RPCMCI failed: no annealings converged. "
            "Try increasing max_anneal or adjusting max_transitions."
        )

    regimes = output["regimes"]  # (num_regimes, T) one-hot
    causal_results = output["causal_results"]  # dict: regime_id -> PCMCI result
    diff_g_f = output["diff_g_f"]
    error_free = output["error_free_annealings"]

    # regimes is (num_regimes, T) — convert to label array
    regime_labels = np.argmax(regimes, axis=0)

    # Extract per-regime graphs
    d = df.shape[1]
    regime_graphs = {}
    for r in range(num_regimes):
        if r in causal_results and "graph" in causal_results[r]:
            regime_graphs[r] = _tigramite_graph_to_binary(causal_results[r]["graph"])
        else:
            regime_graphs[r] = np.zeros((d, d, tau_max + 1), dtype=np.int8)

    # Union graph: edge present if found in any regime
    G_hat = np.zeros((d, d, tau_max + 1), dtype=np.int8)
    for g in regime_graphs.values():
        G_hat = np.maximum(G_hat, g)

    info = {
        "method": "rpcmci",
        "regime_labels": regime_labels,
        "regime_graphs": regime_graphs,
        "regimes_onehot": regimes,
        "causal_results": causal_results,
        "num_regimes": num_regimes,
        "max_transitions": max_transitions,
        "error_free_annealings": error_free,
        "diff_g_f": diff_g_f,
    }

    return G_hat, info


def run_tigramite(
    df: pd.DataFrame,
    method: str = "pcmciplus",
    tau_max: int = 1,
    ci_test: str | object = "parcorr",
    alpha_level: float = 0.05,
    verbosity: int = 0,
    ci_kwargs: dict | None = None,
    **kwargs,
) -> tuple[np.ndarray, dict]:
    """Run any tigramite discovery method and return a binary causal graph.

    Generic dispatcher that covers all tigramite PCMCI methods. For
    fine-grained control, use run_pcmciplus() or run_pcmci() directly.

    Parameters
    ----------
    df : pandas.DataFrame
        Time series DataFrame (T, d).
    method : str
        Discovery method name. One of:

        - ``"pcmciplus"`` -- PCMCI+ (contemporaneous + lagged, oriented)
        - ``"pcmci"`` -- PCMCI (lagged links with MCI pruning)
        - ``"pcalg"`` -- PC algorithm adapted for time series
        - ``"pc_stable"`` -- PC-stable skeleton only (no orientation)
        - ``"fullci"`` -- Full conditional independence (all pairs)
        - ``"bivci"`` -- Bivariate CI (pairwise, no conditioning)
    tau_max : int
        Maximum time lag.
    ci_test : str or object
        CI test name or instance.
    alpha_level : float
        Significance threshold for edge inclusion.
    verbosity : int
        0=silent, 1=summary, 2=detailed.
    ci_kwargs : dict or None
        Extra kwargs for CI test constructor.
    **kwargs : dict
        Passed to the tigramite method (e.g., pc_alpha, fdr_method,
        contemp_collider_rule).

    Returns
    -------
    tuple
        Tuple (G_hat, info):

        - G_hat: Binary graph (d, d, tau_max+1).
        - info: Dict with method, p_matrix, val_matrix, graph_raw.
    """
    _require_tigramite(f"tigramite ({method})")
    from tigramite.pcmci import PCMCI

    method_key = method.lower().replace("-", "_")
    if method_key not in _METHOD_MAP:
        raise ValueError(
            f"Unknown method: {method!r}. Available: {list(_METHOD_MAP.keys())}"
        )

    # For pcmciplus and pcmci, delegate to dedicated wrappers
    if method_key == "pcmciplus":
        return run_pcmciplus(
            df,
            tau_max=tau_max,
            ci_test=ci_test,
            verbosity=verbosity,
            ci_kwargs=ci_kwargs,
            **kwargs,
        )
    if method_key == "lpcmci":
        return run_lpcmci(
            df,
            tau_max=tau_max,
            ci_test=ci_test,
            verbosity=verbosity,
            ci_kwargs=ci_kwargs,
            **kwargs,
        )
    if method_key == "pcmci":
        return run_pcmci(
            df,
            tau_max=tau_max,
            ci_test=ci_test,
            alpha_level=alpha_level,
            verbosity=verbosity,
            ci_kwargs=ci_kwargs,
            **kwargs,
        )

    ci_kwargs = ci_kwargs or {}
    cit = _make_tigramite_ci_test(ci_test, **ci_kwargs)
    tig_df = _to_tigramite_df(df)

    pcmci = PCMCI(dataframe=tig_df, cond_ind_test=cit, verbosity=verbosity)

    run_fn = getattr(pcmci, _METHOD_MAP[method_key])

    # Build kwargs appropriate for the method
    method_kwargs = {"tau_max": tau_max}
    if method_key in ("fullci", "bivci"):
        method_kwargs["alpha_level"] = alpha_level
    if method_key == "pcalg":
        method_kwargs["pc_alpha"] = kwargs.pop("pc_alpha", 0.01)
    if method_key == "pc_stable":
        method_kwargs["pc_alpha"] = kwargs.pop("pc_alpha", 0.2)
    method_kwargs.update(kwargs)

    result = run_fn(**method_kwargs)

    # pc_stable and others may return different formats
    if "graph" in result:
        G_hat = _tigramite_graph_to_binary(
            result["graph"], p_matrix=result.get("p_matrix"), alpha_level=alpha_level
        )
        graph_raw = result["graph"]
    else:
        # Some methods return only p_matrix/val_matrix without graph
        p_matrix = result.get("p_matrix")
        val_matrix = result.get("val_matrix")  # noqa: F841
        d = df.shape[1]
        G_hat = np.zeros((d, d, tau_max + 1), dtype=np.int8)
        if p_matrix is not None:
            G_hat[p_matrix < alpha_level] = 1
            # Clear self-loops at lag 0
            np.fill_diagonal(G_hat[:, :, 0], 0)
        graph_raw = None

    info = {
        "method": method_key,
        "p_matrix": result.get("p_matrix"),
        "val_matrix": result.get("val_matrix"),
        "graph_raw": graph_raw,
        "alpha_level": alpha_level,
    }

    return G_hat, info
