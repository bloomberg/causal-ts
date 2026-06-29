# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""
.. deprecated::
    This module contains the original ``SYPI`` class on which published paper
    results are based.  It is **kept for reproducibility only** and is no
    longer maintained.  Use :func:`causalts.cedar.run_cedar` or
    :class:`causalts.cedar.Cedar` for all new work.
"""

import copy
import logging
import os
import warnings
from hashlib import sha1

import dcor
import networkx as nx
import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
from scipy.stats import multiscale_graphcorr
from sklearn.linear_model import LassoCV, LassoLarsCV

from ..ci_tests import kci_gpu as kci
from ..plotting import _core as tp
from ..utils.helpers import set_up_logging

# Suppress noisy third-party warnings that fire on import of the above deps.
warnings.filterwarnings("ignore", category=FutureWarning, module="scipy")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


class SYPI:  # SYstemathic Path Isolation for causal discovery — DEPRECATED
    def __init__(self, **kwargs):
        warnings.warn(
            "SYPI is deprecated and kept only for paper reproducibility. "
            "Use run_cedar() or Cedar from causalts.cedar for all new work.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.lag_sel_algo = kwargs.get("lag_sel_algo", None)
        self.data_normalization_method = kwargs.get(
            "data_normalization_method", "minmax"
        )
        self.p_cond1 = kwargs.get("p_cond1", None)
        self.p_cond2 = kwargs.get("p_cond2", None)
        self.significance_threshold = kwargs.get("significance_threshold", None)
        self.data = kwargs.get("data", None)
        self.target_var = kwargs.get("target_var", None)
        self.include_lag0 = kwargs.get("include_lag0", False)
        self.max_lag = kwargs.get("max_lag", None)
        self.ci_test = kwargs.get("ci_test", None)
        assert self.ci_test is not None
        # self.workers = kwargs.get('workers',-1)
        self.simulation_name = kwargs.get("simulation_name", "NEW SIMULATIONS")
        self.make_log = kwargs.get("make_log", False)
        self.filter_potential_children = kwargs.get("filter_potential_children", True)
        self.threshold_multiplier_4_filter_children = kwargs.get(
            "threshold_multiplier_4_filter_children", 1
        )
        self.direction_strength_threshold = kwargs.get(
            "direction_strength_threshold", 1.2
        )
        self.keep_strongest_lags = kwargs.get("keep_strongest_lags", False)
        self.cache_dir = kwargs.get("cache_dir", "./causalts_cache")
        self.impute = kwargs.get("impute", None)
        self.impute_kwargs = kwargs.get("impute_kwargs", {})
        self._nan_warned = False

        os.makedirs(self.cache_dir, exist_ok=True)

        if self.make_log:
            if not hasattr(self, "logger"):
                if "sypi_logger" not in logging.Logger.manager.loggerDict.keys():
                    self.logger = set_up_logging(
                        "sypi_logger",
                        "stdout",
                        "warning",
                        True,
                        os.path.join(self.cache_dir, "debugs.log"),
                        "debug",
                        False,
                        "%(asctime)s%(color_on)s[%(levelname)-5s] %(message)s%(color_off)s",
                    )
                else:
                    self.logger = logging.Logger.manager.loggerDict["sypi_logger"]

        if self.make_log:
            self.print_settings()
        self._scm_cache = {}
        self.Initialize()
        # lag_sel_algo: string which algorithm to use for lag calculation.
        # Implemented so far: lasso regression 'dcor'
        # relationships: string 'linear' (implemented so far)
        # normalized_data: boolean True if want to normalize the data for the lag calculation, False otherwise
        # normalization: type of normalization 'variance', 'minmax'
        # p_cond1: float threshold1 for conditional dependence
        # p_cond2: float threshold2 for conditional independence
        # threshold_lasso: float threshold for lasso coefficients
        # var_direct_causes: set of ground truth direct causes for calculation of false positives, false negatives,
        # true positives, true negatives
        # var_indirect_causes: set of ground truth indirect causes for calculation of false positives, false negatives,
        # true positives, true negatives
        # var_non_causes: set of ground truth non causes for calculation of false positives, false negatives,
        # true positives, true negatives

    def print_settings(self):
        self.logger.debug(
            " ###############################################################"
        )
        self.logger.debug(" #{:^61s}# ".format(self.simulation_name))
        self.logger.debug(
            " ###############################################################\n"
        )
        self.logger.debug(
            f"ci_test={self.ci_test.method}, p_cond1={self.p_cond1}, "
            f"p_cond2={self.p_cond2}, lag_sel={self.lag_sel_algo}"
        )
        self.logger.debug(
            f"max_lag={self.max_lag}, lag_sig_thresh={self.significance_threshold}, "
            f"include_lag0={self.include_lag0}"
        )
        self.logger.debug(
            f"nvars={self.data.shape[1]}, nSamples={self.data.shape[0]}, "
            f"normalization={self.data_normalization_method}"
        )

    def Initialize(self):
        self.n_tests = 0

        # Apply imputation before normalization
        if self.impute is not None and self.data.isna().any().any():
            import warnings

            if self.impute == "causal_iterative":
                warnings.warn(
                    "impute='causal_iterative' is not supported in SyPI+ "
                    "(requires CDNOTS skeleton). Falling back to 'var_em'.",
                    UserWarning,
                    stacklevel=2,
                )
            from ..imputation import var_em_impute

            self.data = var_em_impute(self.data, **self.impute_kwargs)

        self.X = self.data.T.values
        self.var_names = self.data.columns.tolist()
        self.lag_importance = {}

        if self.ci_test.method in ["par_corr", "myparrcorr"]:
            self.relationships = "linear"
        else:
            self.relationships = "nonlinear"

        Xnorm = self.normalize_input(self.X)
        self.data_norm = pd.DataFrame(Xnorm.T, columns=self.data.columns)

        self.hash_code = sha1(pd.util.hash_pandas_object(self.data).values).hexdigest()

    def gen_lags(self, S, lag_label="lag", include_lag0=False):
        if include_lag0:
            return pd.concat(
                {
                    f"{S.name}_{lag_label}{lag}": S.shift(lag)
                    for lag in range(self.max_lag + 1)
                },
                axis=1,
            )
        else:
            return pd.concat(
                {
                    f"{S.name}_{lag_label}{lag}": S.shift(lag)
                    for lag in range(1, self.max_lag + 1)
                },
                axis=1,
            )

    def normalize_input(self, X: np.ndarray) -> np.ndarray:
        """
        Normalize data X or not according to normalization type
        X: 2D numpy array (number of time series) x (number of time steps).
        normalization: 'variance' or 'minmax'
        Xnorm normalized time series if normalized_data==True
        """
        if self.data_normalization_method is not None:
            Xnorm = np.full(np.shape(X), np.nan)

            if self.data_normalization_method == "variance":
                for i_var in range(X.shape[0]):
                    if np.count_nonzero(X[i_var, :]) == 0:
                        Xnorm[i_var, :] = X[i_var, :]
                    else:
                        Xnorm[i_var, :] = (X[i_var, :] - np.mean(X[i_var, :])) / np.std(
                            X[i_var, :]
                        )

            if self.data_normalization_method == "minmax":
                for i_var in range(X.shape[0]):
                    if np.count_nonzero(X[i_var, :]) == 0:
                        Xnorm[i_var, :] = X[i_var, :]
                    else:
                        Xnorm[i_var, :] = (X[i_var, :] - np.min(X[i_var, :])) / (
                            np.max(X[i_var, :]) - np.min(X[i_var, :])
                        )
        else:
            Xnorm = copy.deepcopy(X)
        return Xnorm

    def find_important_lags(
        self, df, covariate, target_var, include_lag0, method="dcor"
    ):
        z = self.gen_lags(df[covariate], include_lag0=include_lag0)
        z = pd.concat([df[target_var], z], axis=1).dropna()
        y = z.iloc[:, 0].values
        X = z.iloc[:, 1:].values
        if method == "lasso_lars_cv":
            reg = LassoLarsCV(cv=5).fit(X, y)
            importance = reg.coef_
        elif method == "lasso_cv":
            reg = LassoCV(cv=5).fit(X, y)
            importance = reg.coef_
        elif method == "mi":
            importance = np.array(
                [self.cmi.get_cmi_(X[:, i], y) for i in range(X.shape[1])]
            )
            # mutual_info_regression(X, y,n_neighbors=3)
        elif method == "cmi":
            importance = np.array(
                [self.cmi.get_cmi_(X[:, i], y, X[:, :i]) for i in range(X.shape[1])]
            )
        elif method == "dcor":
            importance = np.array(
                [
                    dcor.distance_correlation(X[:, i], y, method="avl")
                    for i in range(X.shape[1])
                ]
            )
        elif method == "pdcor":
            importance = np.array(
                [
                    dcor.partial_distance_correlation(X[:, i], y, X[:, :i])
                    for i in range(X.shape[1])
                ]
            )  # all before lag_i

        elif method == "mgc":
            importance = np.array(
                [
                    multiscale_graphcorr(
                        X[:, i], y, reps=1, workers=1, random_state=42
                    )[0]
                    for i in range(X.shape[1])
                ]
            )
        elif method == "pearson":
            importance = np.array(
                [
                    np.corrcoef(
                        X[:, i],
                        y,
                    )[0, 1]
                    for i in range(X.shape[1])
                ]
            )
        return importance

    def calc_lag_importance(self, method=None):
        if method is None:
            method = (
                self.lag_sel_algo
                if self.lag_sel_algo in ["pdcor", "pearson", "dcor", "cmi", "mgc"]
                else "pearson"
            )

        fname = f"{self.cache_dir}/dcor_cache/{method}-mlag{self.max_lag}-{self.hash_code}.npy"

        if os.path.exists(fname):
            if self.make_log:
                self.logger.debug("loading lag_importance_coefficients")
            lag_importance = np.load(fname)
        else:
            os.makedirs(
                os.path.dirname(fname), exist_ok=True
            )  # make the directory  if not there
            if self.make_log:
                self.logger.debug("calculating lag_importance_coefficients")
            lag_importance = np.full(
                (len(self.var_names), len(self.var_names), self.max_lag + 1), np.nan
            )
            for i_tgt, target in enumerate(self.var_names):
                for i_var, var in enumerate(self.var_names):
                    importance_coeffs = self.find_important_lags(
                        self.data_norm, var, target, include_lag0=True, method=method
                    )
                    lag_importance[i_var, i_tgt, :] = importance_coeffs
            np.save(fname, lag_importance)

        if method == "dcor":
            self.lag_importance_dcor_mat = lag_importance
        elif method == "pdcor":
            self.lag_importance_pdcor_mat = lag_importance
        elif method == "mgc":
            self.lag_importance_mgc_mat = lag_importance
        elif method == "cmi":
            self.lag_importance_cmi_mat = lag_importance
        elif method == "mi":
            self.lag_importance_mi_mat = lag_importance
        else:
            self.lag_importance_pearson_mat = lag_importance

    def generate_potential_skeleton(self, method="pdcor", threshold_multiplier=1.0):
        if method == "pdcor":
            if not hasattr(self, "lag_importance_pdcor_mat"):
                self.calc_lag_importance(method="pdcor")
            lag_imp_mat = self.lag_importance_pdcor_mat

        if method == "dcor":
            if not hasattr(self, "lag_importance_dcor_mat"):
                self.calc_lag_importance(method="dcor")
            lag_imp_mat = self.lag_importance_dcor_mat

        if method == "mgc":
            if not hasattr(self, "lag_importance_mgc_mat"):
                self.calc_lag_importance(method="mgc")
            lag_imp_mat = self.lag_importance_mgc_mat
        if method == "pearson":
            if not hasattr(self, "lag_importance_pearson_mat"):
                self.calc_lag_importance(method="pearson")
            lag_imp_mat = self.lag_importance_pearson_mat

        mat = lag_imp_mat.copy()
        mat = np.abs(mat)
        np.fill_diagonal(mat[:, :, 0], 0)
        sd = mat.std()
        mat[mat < threshold_multiplier * sd] = 0

        link_assumptions = {}
        for n in range(mat.shape[0]):
            ldic = {}
            for j in range(mat.shape[0]):
                sig_lags = mat[j, n, :]

                for lag_ in np.where(sig_lags != 0)[0]:
                    if lag_ != 0:  # pcmci does not model it
                        ldic[(j, -lag_)] = (
                            "-?>"  # means might or might not exist, but if exist the direction is fron j_lag_l to n
                        )
            if ldic != {}:  # not empty
                link_assumptions[n] = ldic

        return link_assumptions

    def set_threshold(self, target, multiplier=1.0):
        if self.significance_threshold is None:
            # Map each algo to its cached importance matrix attribute
            _algo_to_attr = {
                "dcor": "lag_importance_dcor_mat",
                "pdcor": "lag_importance_pdcor_mat",
                "pearson": "lag_importance_pearson_mat",
                "mi": "lag_importance_mi_mat",
                "cmi": "lag_importance_cmi_mat",
                "mgc": "lag_importance_mgc_mat",
                "lasso_cv": "lag_importance_pearson_mat",
                "lasso_lars_cv": "lag_importance_pearson_mat",
            }
            attr = _algo_to_attr.get(self.lag_sel_algo, "lag_importance_pearson_mat")
            if not hasattr(self, attr):
                self.calc_lag_importance(method=self.lag_sel_algo)
            mat = getattr(self, attr).copy()
            mat = np.abs(mat)
            np.fill_diagonal(mat[:, :, 0], 0)
            threshold = multiplier * mat.std()
        else:
            threshold = self.significance_threshold

        return threshold

    def calculate_w(self, target):
        """
        Calculate the lag between each time series and the target time series
        w_max the maximum lag among all lags, w 1D np.ndarray where each value corresponds to the lag of that time
        series with the target. If there is no dependency between a time series i and the target then w[i] = nan
        """
        w = np.full(self.data.shape[1] - 1, np.nan)  # initialize lag vector
        wp = {}
        len_y = self.max_lag + 1 if self.include_lag0 else self.max_lag
        cols = [f"lag{lag_ + 1}" for lag_ in range(self.max_lag)]
        cols = cols.insert(0, "lag0") if self.include_lag0 else cols
        lag_importance = np.full((len(w), len_y), np.nan)
        other_vars = np.setdiff1d(self.var_names, target, assume_unique=True)
        threshold = self.set_threshold(target)
        if self.significance_threshold is None:
            if self.make_log:
                self.logger.debug(f"calculated threshold = {threshold}")
        st_ind = 0 if self.include_lag0 else 1

        # Assume the shortest path with the minimum lag corresponds to the
        # strongest lasso regression coefficient in the linear case.
        for i_var, var in enumerate(other_vars):
            if self.lag_sel_algo == "dcor" and hasattr(
                self, "lag_importance_dcor_mat"
            ):  # everything already calculated just read it
                importance_coeffs = self.lag_importance_dcor_mat[
                    self.var_names.index(var), self.var_names.index(target), st_ind:
                ]

            elif self.lag_sel_algo == "pdcor" and hasattr(
                self, "lag_importance_pdcor_mat"
            ):
                importance_coeffs = self.lag_importance_pdcor_mat[
                    self.var_names.index(var), self.var_names.index(target), st_ind:
                ]
            elif self.lag_sel_algo == "mgc" and hasattr(self, "lag_importance_mgc_mat"):
                importance_coeffs = self.lag_importance_mgc_mat[
                    self.var_names.index(var), self.var_names.index(target), st_ind:
                ]
            else:
                importance_coeffs = self.find_important_lags(
                    self.data_norm,
                    var,
                    target,
                    self.include_lag0,
                    method=self.lag_sel_algo,
                )

            lag_importance[i_var, :] = importance_coeffs[:len_y]
            importance_coeffs = abs(importance_coeffs)
            idx = (-importance_coeffs).argsort()
            imp_srt = importance_coeffs[idx]

            if self.keep_strongest_lags:
                imp_srt = imp_srt[:1]

            smape = []
            for j in range(len(imp_srt) - 1):
                smape.append((imp_srt[0] - imp_srt[j]) / (imp_srt[0] + imp_srt[j]))
            if len(imp_srt) == 1:  # above loop did not run!
                smape = [0]

            acceptable_lags = (
                []
            )  # if two llags are close, we take the smaller as the main
            for k in range(len(smape)):
                if (
                    smape[k] < 0.15
                    and abs(idx[k] - idx[0]) <= 2
                    and imp_srt[k] > threshold
                ):  # HARDCODED THRESHOLDS
                    acceptable_lags.append(idx[k] if self.include_lag0 else idx[k] + 1)
            if len(acceptable_lags) > 0:
                maxlag_iloc = min(acceptable_lags)
                w[i_var] = maxlag_iloc
                wp[var] = acceptable_lags
            else:
                wp[var] = [np.nan]

        self.lag_importance[target] = pd.DataFrame(
            lag_importance, index=other_vars, columns=cols
        )
        if self.make_log:
            acceptable = {
                str(key): list(map(int, wp[key]))
                for key in wp
                if not np.isnan(wp[key]).any()
            }
            self.logger.debug(f"acceptable w: {str(acceptable)}")

        return w

    def list_diff(list1, list2):
        out = []
        for ele in list1:
            if ele not in list2:
                out.append(ele)
        return out

    def precheck(self):
        strange_vars = {}
        for v in self.data.columns:
            max_pacf = abs(
                sm.tsa.stattools.pacf(
                    self.data.loc[:, v], nlags=self.max_lag, method="ywadjusted"
                )[1:]
            ).argmax()
            if max_pacf > 0:
                strange_vars[v] = max_pacf

        if strange_vars:
            if self.make_log:
                violations = {key: strange_vars[key] + 1 for key in strange_vars}
                self.logger.warning(
                    "time series potentially violating assumption: "
                    "x(t) is not f(x(t-1), other x): "
                    f"{str(violations)}"
                )

    def convert_to_string_graph(self, graph_bool):
        """Converts the 0,1-based graph to a string array
        with links '-->'.

        Parameters
        ----------
        graph_bool : array
            0,1-based graph array.

        Returns
        -------
        graph : array
            graph as string array with links '-->'.
        """

        graph = np.zeros(graph_bool.shape, dtype="<U3")
        graph[:] = ""
        # Lagged links
        graph[:, :, :][graph_bool[:, :, :] == 1] = "-->"
        # # Unoriented contemporaneous links
        graph[:, :, 0][
            np.logical_and(graph_bool[:, :, 0] == 1, graph_bool[:, :, 0].T == 1)
        ] = "o-o"
        # # Conflicting contemporaneous links
        graph[:, :, 0][
            np.logical_and(graph_bool[:, :, 0] == 2, graph_bool[:, :, 0].T == 2)
        ] = "x-x"
        # Directed contemporaneous links
        for i, j in zip(
            *np.where(
                np.logical_and(graph_bool[:, :, 0] == 1, graph_bool[:, :, 0].T == 0)
            )
        ):
            graph[i, j, 0] = "-->"
            graph[j, i, 0] = "<--"

        return graph

    def create_graph(self):
        graph_ = np.zeros((len(self.var_names), len(self.var_names), self.max_lag + 1))
        strength_ = np.zeros(
            (len(self.var_names), len(self.var_names), self.max_lag + 1)
        )
        for ind, target in enumerate(self.var_names):
            for key in self.res.keys():
                if target in key:
                    try:
                        for j, c in enumerate(self.res[(target, "causes")]):
                            graph_[
                                self.var_names.index(c),
                                ind,
                                self.res[(target, "lags")][j],
                            ] = 1
                            strength_[
                                self.var_names.index(c),
                                ind,
                                self.res[(target, "lags")][j],
                            ] = self.res[(target, "strength")][j]
                    except ValueError:
                        if self.make_log:
                            self.logger.error(f"graph calculations error {target}")
        self.graph = graph_
        self.link_strength = strength_
        self.str_graph = self.convert_to_string_graph(graph_)
        self.str_graph_pruned = None
        self.graph_pruned = None

    # ------------------------------------------------------------------
    # DoWhy bridge — requires pip install dowhy (or pip install causalts[dowhy])
    # Available after run_ts_causality() has been called.
    # ------------------------------------------------------------------

    def _bridge_graph(self):
        """Return graph for DoWhy with implicit AR(1) self-lags restored.

        CEDAR excludes Xi(t-1)->Xi from discovery by design (it conditions on
        them but never tests them). Without these edges DoWhy's backdoor
        criterion omits a key confounder, biasing cross-variable effect
        estimates. We restore lag-1 self-edges here before passing to DoWhy.
        """
        g = self.graph_pruned if self.graph_pruned is not None else self.graph
        g = g.copy()
        np.fill_diagonal(g[:, :, 1], 1)
        return g

    def _bridge_df(self):
        """Return the original (un-normalised) DataFrame."""
        return self.data if hasattr(self.data, "columns") else self.data

    def estimate_effect(self, treatment, outcome, treatment_lag=1, **kwargs):
        """Estimate causal effect of treatment(t-lag) on outcome(t).

        Requires DoWhy (pip install dowhy or pip install causalts[dowhy]).
        Must be called after run_ts_causality().
        """
        from ..effects.effect import estimate_effect

        return estimate_effect(
            self._bridge_graph(),
            self._bridge_df(),
            treatment=treatment,
            outcome=outcome,
            treatment_lag=treatment_lag,
            var_names=self.var_names,
            **kwargs,
        )

    def fit_scm(self, mechanism_type="auto", force=False):
        """Fit a Structural Causal Model to the discovered graph.

        Result is cached — repeated calls with the same mechanism_type
        return the cached (scm, dag, lagged_df) without refitting.

        Requires DoWhy (pip install dowhy or pip install causalts[dowhy]).
        Must be called after run_ts_causality().
        """
        from ..effects.scm import fit_scm

        if not force and mechanism_type in self._scm_cache:
            return self._scm_cache[mechanism_type]
        result = fit_scm(
            self._bridge_graph(),
            self._bridge_df(),
            var_names=self.var_names,
            mechanism_type=mechanism_type,
        )
        self._scm_cache[mechanism_type] = result
        return result

    def counterfactual(self, intervention, target, mechanism_type="auto"):
        """Estimate counterfactual values for target under do(intervention).

        Fits the SCM automatically if not already cached.

        Args:
            intervention: Dict of {node: value}, e.g. {"X0_lag1": 0.0}.
            target: Current-time variable to query (e.g. "X2").
            mechanism_type: Passed to fit_scm if not already cached.

        Requires DoWhy (pip install dowhy or pip install causalts[dowhy]).
        """
        from ..effects.scm import counterfactual

        scm, _, lagged_df = self.fit_scm(mechanism_type=mechanism_type)
        return counterfactual(scm, lagged_df, intervention=intervention, target=target)

    def attribute_anomaly(self, anomaly, target, mechanism_type="auto", n_samples=2000):
        """Attribute an anomaly in target to its upstream causes.

        Fits the SCM automatically if not already cached.

        Args:
            anomaly: pd.Series or pd.DataFrame of anomalous observation(s).
            target: Variable to explain (e.g. "X2").
            mechanism_type: Passed to fit_scm if not already cached.
            n_samples: Monte Carlo samples for Shapley attribution.

        Returns:
            DataFrame with columns [node, variable, lag, mean_attribution].

        Requires DoWhy (pip install dowhy or pip install causalts[dowhy]).
        """
        from ..effects.root_cause import attribute_anomaly

        scm, _, lagged_df = self.fit_scm(mechanism_type=mechanism_type)
        return attribute_anomaly(
            scm, lagged_df, anomaly=anomaly, target=target, n_samples=n_samples
        )

    def falsify(self, **kwargs) -> dict:
        """Test if graph is consistent with data. No ground truth required."""
        from ..effects.validate import falsify_graph

        return falsify_graph(
            self._bridge_graph(),
            self._bridge_df(),
            var_names=self.var_names,
            **kwargs,
        )

    def refute_structure(self, **kwargs):
        """Test structural assumptions. No ground truth required."""
        from ..effects.validate import refute_structure

        return refute_structure(
            self._bridge_graph(),
            self._bridge_df(),
            var_names=self.var_names,
            **kwargs,
        )

    def evaluate_model(self, mechanism_type="auto", **kwargs) -> dict:
        """Comprehensive model evaluation. No ground truth required."""
        from ..effects.validate import evaluate_model

        return evaluate_model(
            self._bridge_graph(),
            self._bridge_df(),
            var_names=self.var_names,
            mechanism_type=mechanism_type,
            **kwargs,
        )

    def arrow_strength(self, target, **kwargs):
        """Quantify how strongly each parent influences target."""
        from ..effects.influence import arrow_strength

        return arrow_strength(
            self._bridge_graph(),
            self._bridge_df(),
            target,
            var_names=self.var_names,
            **kwargs,
        )

    def causal_influence(self, target, **kwargs):
        """Shapley decomposition of target variability by upstream noise."""
        from ..effects.influence import causal_influence

        return causal_influence(
            self._bridge_graph(),
            self._bridge_df(),
            target,
            var_names=self.var_names,
            **kwargs,
        )

    def parent_relevance(self, target, **kwargs):
        """Rank direct parents by explanatory power."""
        from ..effects.influence import parent_relevance

        return parent_relevance(
            self._bridge_graph(),
            self._bridge_df(),
            target,
            var_names=self.var_names,
            **kwargs,
        )

    def distribution_change(self, data_new, target, **kwargs):
        """Identify which mechanisms changed between original and new data."""
        from ..effects.influence import distribution_change

        return distribution_change(
            self._bridge_graph(),
            self._bridge_df(),
            data_new,
            target,
            var_names=self.var_names,
            **kwargs,
        )

    def summary(self, mechanism_type="linear", top_k=5):
        """Print a one-page diagnostic report. Requires DoWhy."""
        from ..effects.wrap import wrap_graph

        return wrap_graph(
            self._bridge_graph(), self._bridge_df(), self.var_names
        ).summary(mechanism_type=mechanism_type, top_k=top_k)

    def plot_network(self, show_pruned=True, use_val_matrix=True, **kwargs):
        if self.str_graph_pruned is not None and show_pruned:
            graph = self.str_graph_pruned
        else:
            graph = self.str_graph

        link_width_multiplier = kwargs.pop("link_width_multiplier", 5)

        method = (
            "MCI"
            if self.ci_test.method in ["cmi_knn", "par_corr"]
            else self.lag_sel_algo
        )

        if use_val_matrix:
            link_strength = (
                None
                if self.link_strength.max() == 0
                else np.abs(self.link_strength) * link_width_multiplier
            )
        else:
            link_strength = None

        # Defaults that can be overridden via kwargs
        defaults = dict(
            figsize=(10, 10),
            show_colorbar=False,
            arrow_linewidth=3,
            node_size=0.3,
            cmap_edges="jet",
            vmin_edges=0,
            link_colorbar_label=method.upper(),
            val_matrix=link_strength,
            link_width=link_strength,
        )
        defaults.update(kwargs)

        return tp.plot_graph(
            graph=graph,
            var_names=self.var_names,
            **defaults,
        )

    def plot_time_series_graph(self, **kwargs):
        # figsize = kwargs.get('figsize',(10,8))
        show_pruned = kwargs.get("show_pruned", True)

        if self.str_graph_pruned is not None and show_pruned:
            graph = self.str_graph_pruned
        else:
            graph = self.str_graph

        return tp.plot_time_series_graph(
            # figsize=figsize,
            val_matrix=self.link_strength,
            graph=graph,
            var_names=self.var_names,
            **kwargs,
        )

    def plot_lag_dependancy(self, method=None, fig_name=None, **kwargs):
        y_max = kwargs.get("maximum", 1.0)
        y_min = kwargs.get("minimum", 0)
        show_gridlines = kwargs.get("show_gridlines", True)
        x_tik_inc = kwargs.get("x_tik_inc", 1)
        y_tik_inc = kwargs.get("y_tik_inc", 0.2)
        figsize = kwargs.get("figsize", (10, 10))

        if method is None:
            method = (
                self.lag_sel_algo
                if self.lag_sel_algo in ["pdcor", "pearson", "dcor", "mgc"]
                else "pearson"
            )

        if method == "dcor" and not hasattr(self, "lag_importance_dcor_mat"):
            self.calc_lag_importance(method="dcor")
        if method == "mgc" and not hasattr(self, "lag_importance_mgc_mat"):
            self.calc_lag_importance(method="mgc")
        if method == "pdcor" and not hasattr(self, "lag_importance_pdcor_mat"):
            self.calc_lag_importance(method="pdcor")
        if method == "cmi" and not hasattr(self, "lag_importance_cmi_mat"):
            self.calc_lag_importance(method="cmi")
        if method == "mi" and not hasattr(self, "lag_importance_mi_mat"):
            self.calc_lag_importance(method="mi")
        if method == "pearson" and not hasattr(self, "lag_importance_pearson_mat"):
            self.calc_lag_importance(method="pearson")

        if method == "dcor":
            lag_imp = self.lag_importance_dcor_mat
        elif method == "pdcor":
            lag_imp = self.lag_importance_pdcor_mat
        elif method == "mgc":
            lag_imp = self.lag_importance_mgc_mat
        elif method == "cmi":
            lag_imp = self.lag_importance_cmi_mat
        elif method == "mi":
            lag_imp = self.lag_importance_mi_mat
        elif method == "pearson":
            lag_imp = self.lag_importance_pearson_mat

        plt_matrix = tp.plot_lagfuncs(
            val_matrix=lag_imp,
            setup_args={
                "var_names": self.var_names,
                "x_base": x_tik_inc,
                "y_base": y_tik_inc,
                "minimum": y_min,
                "maximum": y_max,
                "plot_gridlines": show_gridlines,
                "figsize": figsize,
            },
            name=fig_name,
        )
        return plt_matrix.fig

    def plot_time_series(self, normalized=False, **kwargs):
        import plotly.express as px

        if normalized:
            return px.line(data_frame=self.data_norm, **kwargs)
        else:
            return px.line(data_frame=self.data, **kwargs)

    def prune_graph(self, max_path_len=4):
        # does a final pruning of the redundant links
        # create a graph to be used for finding predecessors and descendants
        G = nx.from_numpy_array(self.graph.sum(axis=2), create_using=nx.DiGraph)
        mapping = dict(zip(G, self.var_names))
        G = nx.relabel_nodes(G, mapping)
        self.str_graph_pruned = self.str_graph.copy()
        self.graph_pruned = self.graph.copy()
        removed_pairs = []
        tested_pairs = {}

        for node in self.var_names:
            parents = self.res[(node, "causes")].copy()
            if len(parents) > 1:
                lag_dic = {
                    k: v
                    for k, v in zip(self.res[node, "causes"], self.res[node, "lags"])
                }
                for p in parents:
                    children = list(nx.descendants(G, p))
                    com_ = np.intersect1d(parents, children)
                    all_paths = list(
                        nx.all_simple_paths(
                            G, source=p, target=node, cutoff=max_path_len
                        )
                    )
                    if len(com_) > 0:
                        for c in com_:
                            paths = [p_ for p_ in all_paths if c in p_]
                            if len(paths) != 0:
                                path = paths[
                                    0
                                ]  # can be more for now lets ignore long paths!
                                vals = [self.res[(node, p, "abs_strength")]]
                                for i in range(len(path) - 1):
                                    vals.append(
                                        self.res[(path[i + 1], path[i], "abs_strength")]
                                    )

                                if len(vals) == vals.index(min(vals)) + 1:
                                    source = c
                                elif vals.index(min(vals)) == 0:
                                    source = p
                                else:
                                    continue

                                if (source, node) not in removed_pairs:
                                    if (
                                        source,
                                        node,
                                        *parents,
                                    ) not in tested_pairs.keys():
                                        if self.make_log:
                                            self.logger.debug(
                                                f"{source}->{node} qualifies for pruning"
                                            )
                                        zz = pd.concat(
                                            [
                                                self.data.loc[:, source].shift(
                                                    lag_dic[source]
                                                ),  # potentional removal
                                                self.data.loc[:, node],  # target at t
                                                self.data.loc[:, node].shift(
                                                    1
                                                ),  # target t-1
                                                self.data.loc[:, node].shift(
                                                    -1
                                                ),  # target t+1
                                            ],
                                            axis=1,
                                        )
                                        for k in np.setdiff1d(
                                            parents, source, assume_unique=True
                                        ):  # add other parents, link to xt and xt+1 (child parent)
                                            zz = pd.concat(
                                                [
                                                    zz,
                                                    self.data.loc[:, k].shift(
                                                        lag_dic[k]
                                                    ),
                                                    self.data.loc[:, k].shift(
                                                        lag_dic[k] - 1
                                                    ),
                                                ],
                                                axis=1,
                                            )

                                        p_val, _ = self.test_conditions(
                                            zz, calc_strength=False
                                        )
                                        tested_pairs[(source, node, *parents)] = p_val
                                        if self.make_log:
                                            self.logger.debug(
                                                f"pval({source} -/-> {node} | MB) = {p_val} using {self.ci_test.method}"
                                            )
                                        if p_val >= self.p_cond2:
                                            G.remove_edge(source, node)
                                            removed_pairs.append((source, node))
                                            parents.remove(source)
                                            if self.make_log:
                                                self.logger.warning(
                                                    f"link from {source} to {node} removed"
                                                )
                                            self.str_graph_pruned[
                                                self.var_names.index(source),
                                                self.var_names.index(node),
                                                lag_dic[source],
                                            ] = ""
                                            self.graph_pruned[
                                                self.var_names.index(source),
                                                self.var_names.index(node),
                                                lag_dic[source],
                                            ] = 0

        # Try simple cycles
        simple_cycles = list(nx.simple_cycles(G, length_bound=2))
        for pair in simple_cycles:
            for src, trg in [pair, pair[::-1]]:
                lag_dic = {
                    k: v for k, v in zip(self.res[trg, "causes"], self.res[trg, "lags"])
                }
                zz = pd.concat(
                    [
                        self.data.loc[:, src].shift(
                            lag_dic[src]
                        ),  # potentional removal
                        self.data.loc[:, trg],  # target at t
                        self.data.loc[:, trg].shift(1),  # target t-1
                        self.data.loc[:, trg].shift(-1),  # target t+1
                    ],
                    axis=1,
                )
                for k in np.setdiff1d(
                    list(lag_dic.keys()), src, assume_unique=True
                ):  # add other parents, link to xt and xt+1 (child parent)
                    zz = pd.concat(
                        [
                            zz,
                            self.data.loc[:, k].shift(lag_dic[k]),
                            self.data.loc[:, k].shift(lag_dic[k] - 1),
                        ],
                        axis=1,
                    )
                p_val, _ = self.test_conditions(zz, calc_strength=False)
                if self.make_log:
                    self.logger.debug(
                        f"pval({src} -/-> {trg} | MB) = {p_val} using {self.ci_test.method}"
                    )
                if p_val >= self.p_cond2:
                    G.remove_edge(src, trg)
                    # parents.remove(src)
                    if self.make_log:
                        self.logger.warning(f"Cycle: link from {src} to {trg} removed")
                    self.str_graph_pruned[
                        self.var_names.index(src),
                        self.var_names.index(trg),
                        lag_dic[src],
                    ] = ""
                    self.graph_pruned[
                        self.var_names.index(src),
                        self.var_names.index(trg),
                        lag_dic[src],
                    ] = 0

    def test_conditions(self, df, calc_strength=True):
        self.n_tests += 1
        zz = df.dropna().values
        self.ci_test.data = zz
        cond_set = np.arange(2, zz.shape[1])

        if len(cond_set) == 0:
            cond_set = None
        p_val, rho_ = self.ci_test(0, 1, cond_set)

        return p_val, rho_

    def get_potential_children_new(self, target_ind, direction_strength_threshold=1.2):
        if self.include_lag0:
            lagmat = self.lag_importance_dcor_mat
        else:
            lagmat = self.lag_importance_dcor_mat[:, :, 1:]
        M = lagmat.max(axis=2)
        M2 = M * M.T
        M2[M2 < 0.02] = np.nan
        np.fill_diagonal(M2, np.nan)
        sig_relations = np.where(M2[target_ind, :] > 0.02)[0]
        potential_childeren_idx = [
            t
            for t in sig_relations
            if M[target_ind, t] / M[t, target_ind] > direction_strength_threshold
        ]
        potential_childeren = [self.var_names[pc] for pc in potential_childeren_idx]
        return potential_childeren

    def run_ts_causality(self):
        # self.print_settings()
        self.Initialize()
        self.precheck()
        self.res = {}
        self.rejected_links = {}
        self.accepted_links = {}
        self.p_AR_dic = {}

        # _ = self.generate_potential_skeleton(self.lag_sel_algo) ## THIS SHOULD GO OUTSIDE
        self.calc_lag_importance(self.lag_sel_algo)
        if self.filter_potential_children and self.lag_sel_algo != "dcor":
            self.calc_lag_importance(method="dcor")

        if self.target_var is None:
            for ind, target in enumerate(self.var_names):
                self.get_causal_links(target)
        else:
            self.get_causal_links(self.target_var)

        self.create_graph()

        if self.target_var is None:
            if self.make_log:
                self.logger.debug(f"# tests before pruning {self.n_tests}!")
            self.prune_graph()
        if self.make_log:
            self.logger.debug(f"Done!    {self.n_tests} tests performed!")

    def get_causal_links(self, target):
        """
        Detect direct and indirect causes of the last row of X, from the candidate time series in X
        """
        other_cols = self.data.columns[self.data.columns != target].tolist()
        if self.make_log:
            self.logger.debug(f"\n##### {target} #####")

        # calculate the lag between each time series and the target
        w = self.calculate_w(target)
        wmax = np.nanmax(w)
        if self.make_log:
            self.logger.debug(f"w lags for target {target} are {w}")

        predicted_causes = []
        strength_of_predicted_causes = []
        lags_of_predicted_causes = []
        all_stats = {}
        ts_len = self.data.shape[0]

        if self.filter_potential_children:
            # threshold_multiplier_4_filter_children = 1.0
            target_ind = self.var_names.index(target)
            potential_childeren = self.get_potential_children_new(
                target_ind, self.direction_strength_threshold
            )
        else:
            potential_childeren = []

        for i_candidate, candidate in enumerate(other_cols):
            # w = [np.nan,np.nan,np.nan,1,2]
            all_stats[target, candidate, "w"] = w[i_candidate]
            if not np.isnan(w[i_candidate]):
                ss = self.data.loc[:, target].shift(
                    -int(w[i_candidate] - 1)
                )  # Y_t+wi-1
                s_condition_set = []
                pval_ = {}
                for i_var, cond_var in enumerate(other_cols):
                    if (
                        not np.isnan(w[i_var])
                        and cond_var != candidate
                        and cond_var not in potential_childeren
                    ):
                        u = self.data.loc[:, cond_var].shift(
                            -int(w[i_candidate] - w[i_var] - 1)
                        )
                        st, end = int(wmax + 1), ts_len - int(wmax + 1)
                        if self.relationships == "linear":
                            _, pval_[cond_var] = scipy.stats.pearsonr(
                                u.iloc[st:end],
                                self.data.loc[:, target]
                                .shift(int(-w[i_candidate]))
                                .iloc[st:end],
                            )

                        if (
                            self.relationships != "linear"
                        ):  # and (pval_[cond_var] > self.p_cond1 ):
                            pair = np.column_stack(
                                (
                                    u.iloc[st:end],
                                    self.data.loc[:, target]
                                    .shift(int(-w[i_candidate]))
                                    .iloc[st:end],
                                )
                            )
                            # if self.ci_test.method != 'kcigpu':
                            # data_hash=f'S_filtering-{simulation_name}-kcit-T{T}'
                            cit = kci.KCIGPU(
                                pair,
                                approx="hbe",
                                nperms=200,
                                cache_dir=f"{self.cache_dir}/cache",
                                data_hash=f"S_filtering-{self.simulation_name}",
                                width_x="median",
                                width_y="median",
                            )
                            pval_[cond_var], _ = cit(0, 1, [])
                            # else:
                            #     self.ci_test.data = pair
                            #     pval_[cond_var],_ = self.ci_test(0,1,[])

                        if pval_[cond_var] < self.p_cond1:
                            s_condition_set.append(cond_var)
                            ss = pd.concat([ss, u], axis=1)

                # Cond1
                zz = pd.concat(
                    [
                        self.data.loc[:, candidate],
                        self.data.loc[:, target].shift(int(-w[i_candidate])),
                        ss,
                    ],
                    axis=1,
                )  # .dropna().values
                p_MR, rho_MR = self.test_conditions(zz)

                if self.make_log:
                    self.logger.debug(f"{candidate} --> {target}")
                if self.make_log:
                    cond_set = {key: float(np.round(pval_[key], 5)) for key in pval_}
                    self.logger.debug(f"Condition Set: {str(cond_set)}")
                if self.make_log:
                    self.logger.debug(
                        f"(1st Test) p_MR= {p_MR:.1E}, rho_MR= {rho_MR:6f}"
                    )
                all_stats[target, candidate, "p_MR"] = p_MR
                all_stats[target, candidate, "rho_MR"] = rho_MR

                if p_MR <= self.p_cond1:  # and rho_MR > 0.1:
                    # pval of the dependence X^i_{t-1} _|/|_ X_{t}
                    p_AR = self.p_AR_dic.get(candidate, None)
                    if p_AR is None:
                        if self.relationships == "linear":
                            _, p_AR = scipy.stats.pearsonr(
                                self.data.loc[:, candidate].values[self.max_lag :],
                                self.data.loc[:, candidate]
                                .shift(1)
                                .values[self.max_lag :],
                            )
                        else:
                            pair = np.column_stack(
                                (
                                    self.data.loc[:, candidate].values[self.max_lag :],
                                    self.data.loc[:, candidate]
                                    .shift(1)
                                    .values[self.max_lag :],
                                )
                            )
                            cit = kci.KCIGPU(
                                pair,
                                cache_dir=f"{self.cache_dir}/cache",
                                data_hash="S_filtering",
                                width_x="median",
                                width_y="median",
                                width_z="median",
                                approx="hbe",
                                nperms=200,
                            )
                            p_AR, _ = cit(0, 1, [])
                        self.p_AR_dic[candidate] = p_AR
                    if p_AR > self.p_cond1:
                        if self.make_log:
                            self.logger.warning(
                                f"{candidate} potentially violating assumptions: p_AR= {p_AR:6f}"
                            )
                    all_stats[target, candidate, "p_AR"] = p_AR
                    if p_AR < self.p_cond1:  # and rho_PM > 0.1:
                        # Cond2  pval of X^i_{t-1} _|/|_ Y_{t+wi} | {X^i_t, S^i, Y_{t+wi-1}}
                        ff = pd.concat(
                            [
                                self.data.loc[:, candidate].shift(1),
                                self.data.loc[:, target].shift(int(-w[i_candidate])),
                                ss,
                                self.data.loc[:, candidate],
                            ],
                            axis=1,
                        ).iloc[int(wmax) : (ts_len - int(wmax + 1))]
                        p_PRM, _ = self.test_conditions(ff, False)
                        all_stats[target, candidate, "p_PRM"] = p_PRM
                        if self.make_log:
                            self.logger.debug(f"(2nd Test) p_PRM= {p_PRM:4f}")

                        if p_PRM >= self.p_cond2:
                            predicted_causes.append(candidate)
                            strength_of_predicted_causes.append(rho_MR)
                            self.res[(target, candidate, "abs_strength")] = np.abs(
                                rho_MR
                            )
                            lags_of_predicted_causes.append(int(w[i_candidate]))
                            if self.make_log:
                                self.logger.info(
                                    f"Accepted {candidate}(t-{int(w[i_candidate])}) as cause, strength = {rho_MR:4f}"
                                )
                            self.accepted_links[(candidate, target)] = {
                                "pval1": p_MR,
                                "rho": rho_MR,
                                "pval2": p_PRM,
                                "lag": int(w[i_candidate]),
                            }
                        else:
                            self.rejected_links[(candidate, target)] = {
                                "pval1": p_MR,
                                "rho": rho_MR,
                                "pval2": p_PRM,
                                "lag": int(w[i_candidate]),
                            }

        self.res[(target, "causes")] = predicted_causes
        self.res[(target, "strength")] = strength_of_predicted_causes
        self.res[(target, "lags")] = lags_of_predicted_causes
        self.res[(target, "all_stats")] = all_stats
