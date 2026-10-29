# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Entry points for GRACE and REAP.

Both refine a directed, lag-resolved candidate graph: any method's graph passed as
``candidate_graph``, or a CDNOTS graph built on the data when none is given.
:func:`run_grace` refines it with neural Hard Concrete gates; :func:`run_reap`
grows and prunes each target's candidate parents by regression. Both call
:func:`~causalts.grace.gated_discovery.run_cdnots_gated` (``refine="gates"`` and
``refine="shrink"``), whose docstring documents every option in full.
"""

import inspect
import numbers
from typing import Optional, Union

import numpy as np

from .gated_discovery import EFFECT_FLOOR, _warn_unused_ci_test, run_cdnots_gated

#: run_cdnots_gated's shrink options under their run_reap names.
_REAP_NAMES = {
    "shrink_alpha": "prune_alpha",
    "shrink_basis": "basis",
    "shrink_robust": "robust",
    "shrink_interactions": "interactions",
    "shrink_effect_floor": "effect_floor",
}


def _check_candidate_graph(
    candidate_graph, df, max_lag: int, n_extra: int = 0
) -> Optional[np.ndarray]:
    """Validate a supplied candidate graph under the runners' argument name."""
    if candidate_graph is None:
        return None
    G = np.asarray(candidate_graph)
    if G.dtype == object or not np.issubdtype(G.dtype, np.number) and G.dtype != bool:
        raise TypeError(
            "candidate_graph must be a numeric or boolean array such as a "
            "result's cg_tig (string marks from to_marks() are not accepted), got "
            f"{type(candidate_graph).__name__} of dtype {G.dtype}"
        )
    want = (df.shape[1] + n_extra, df.shape[1] + n_extra, max_lag + 1)
    if G.shape != want:
        raise ValueError(
            f"candidate_graph must have shape {want} "
            "([cause, effect, lag] over the variables"
            + (f", then the {n_extra} C columns" if n_extra else "")
            + f" and lags 0..max_lag), got {G.shape}"
        )
    if np.isnan(G.astype(float)).any():
        raise ValueError("candidate_graph contains NaN")
    # Any non-zero entry is a candidate edge (weights, signed coefficients and
    # numeric edge codes alike), so GRACE and REAP see the same candidate set.
    return (G != 0).astype(np.int8)


def _check_reap_options(prune_alpha, basis, interactions, effect_floor) -> None:
    """run_reap's own checks, so errors use its names (not the shrink_ ones)."""
    if isinstance(prune_alpha, bool) or not (
        isinstance(prune_alpha, numbers.Real) and 0 < prune_alpha < 1
    ):
        raise ValueError(f"prune_alpha must be in (0, 1), got {prune_alpha!r}")
    if basis not in ("spline", "rff"):
        raise ValueError(f"basis must be 'spline' or 'rff', got {basis!r}")
    if effect_floor is None:
        return
    if interactions:
        raise ValueError("effect_floor does not support interactions=True")
    if isinstance(effect_floor, str):
        ok = effect_floor == "adaptive"
    else:
        ok = (
            not isinstance(effect_floor, bool)
            and isinstance(effect_floor, numbers.Real)
            and 0 <= effect_floor < 1
        )
    if not ok:
        raise ValueError(
            "effect_floor must be None, 'adaptive' or a float in [0, 1), got "
            f"{effect_floor!r}"
        )


def _params(fn) -> set:
    return set(inspect.signature(fn).parameters) - {"self", "kwargs", "model_kwargs"}


def _reject_foreign_kwargs(kwargs: dict, caller: str) -> None:
    """Raise for options ``caller`` lacks, naming the right argument or function.

    ``run_grace`` forwards unknown keywords to the gated model, so it rejects only
    run_cdnots_gated's options it lacks and REAP's options; ``run_reap`` takes no
    extra keywords at all.
    """
    from .gated_discovery import GatedCausalDiscovery

    grace, reap = _params(run_grace), _params(run_reap)
    gated = _params(run_cdnots_gated)
    renamed_away = {"skeleton", "skeleton_runtime", "refine"}
    reap_only = (reap - grace) | set(_REAP_NAMES) | (gated - grace - renamed_away)
    reap_only -= {"effect_floor"}  # an option of both, with different meanings
    grace_only = (grace - reap) | _params(GatedCausalDiscovery.__init__)
    if caller == "run_grace":
        foreign = set(kwargs) & ((gated - grace) | reap_only)
    else:
        foreign = set(kwargs)
    if not foreign:
        return
    hints = []
    if foreign & {"skeleton", "skeleton_runtime"}:
        hints.append("pass candidate_graph= (and candidate_runtime=) instead")
    if "refine" in foreign:
        mode = "gates" if caller == "run_grace" else "shrink"
        if kwargs["refine"] == mode:
            hints.append(f"{caller} is refine={mode!r}; drop the argument")
        else:
            hints.append(
                "use run_reap for refine='shrink'"
                if caller == "run_grace"
                else "use run_grace for refine='gates'"
            )
    if caller == "run_grace":
        if foreign & reap_only:
            hints.append(
                "use run_reap for the regression-based refinement and its options"
            )
    else:
        renamed = [
            f"{k} -> {_REAP_NAMES[k]}" for k in sorted(foreign & set(_REAP_NAMES))
        ]
        if renamed:
            hints.append(f"run_reap names: {', '.join(renamed)}")
        if foreign & grace_only:
            hints.append("use run_grace for the gated refinement and its options")
    raise TypeError(
        f"{caller}() got unexpected keyword argument(s) {sorted(foreign)}"
        + (f"; {'; '.join(hints)}" if hints else "")
    )


def run_grace(
    df,
    max_lag: int,
    *,
    candidate_graph: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    ci_test_class=None,
    include_C: bool = True,
    include_C_in_model: bool = False,
    c_preset: str = "linear",
    data_transform: Optional[str] = "rank",
    impute: Optional[str] = None,
    impute_kwargs: Optional[dict] = None,
    discrete_cols=None,
    grow_rounds: int = 1,
    grow_q: float = 0.1,
    grow_transform: Optional[str] = None,
    grow_variance: Optional[bool] = None,
    grow_nuisance_basis: Optional[str] = "spline",
    effect_floor: float = EFFECT_FLOOR,
    ar_path: Optional[bool] = None,
    self_loop_basis: str = "union",
    gate_threshold: Union[float, str] = "auto",
    lambda_rule: str = "xcorr",
    lambda_cv: bool = False,
    noise_model: str = "student_t",
    normalize: bool = True,
    max_epochs: int = 150,
    batch_size: Optional[int] = None,
    patience: int = 20,
    device: Optional[str] = None,
    model_seed: Optional[int] = None,
    verbose: bool = True,
    ground_truth: Optional[np.ndarray] = None,
    candidate_runtime: float = 0.0,
    ci_test=None,
    **model_kwargs,
):
    """GRACE: refine a candidate graph with neural Hard Concrete gates.

    1. **Candidate graph.** ``candidate_graph`` if given (any method's graph),
       else a CDNOTS graph on the data (partial correlation at level ``alpha``,
       with the C node when ``include_C``).
    2. **Gated refinement.** A gated model with one Hard Concrete gate per
       candidate cross edge, trained with an L0 penalty whose strength comes from
       ``lambda_rule``. ``grow_rounds`` rounds test the lagged pairs outside the
       candidates (for an effect on the mean, and with ``grow_variance`` on the
       noise scale), add the hits and retrain, so the result is not restricted to
       the candidate graph.
    3. **Final graph.** Cross edges are cut from the gates (``gate_threshold``).
       Under ``ar_path`` self-loops come from an ungated autoregressive head and
       are kept by a partial F-test. Cross edges whose adjusted partial R^2 is
       below ``effect_floor`` are removed.

    Same as ``run_cdnots_gated(..., refine="gates")``; see
    :func:`~causalts.grace.gated_discovery.run_cdnots_gated` for the full
    description of each option.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data, shape (T, d).
    max_lag : int
        Maximum lag.
    candidate_graph : numpy.ndarray or None
        Directed candidate graph, shape (d, d, max_lag+1), with a non-zero
        ``candidate_graph[cause, effect, lag]`` marking ``cause(t-lag) ->
        effect(t)`` (weights are read as edges where non-zero); with ``include_C_in_model=True``, (d + n_C, d + n_C,
        max_lag+1), the C columns last. None builds a CDNOTS graph. To keep the
        result inside a supplied graph, pass ``grow_rounds=0, ar_path=False``:
        otherwise the grow round can add lagged edges and the self-loop test can
        add self-loops.
    alpha : float, default 0.05
        CDNOTS significance level for the candidate graph (unused when
        ``candidate_graph`` is given).
    ci_test_class : class or None
        CI test class for the CDNOTS candidate graph (default partial
        correlation).
    include_C : bool, default True
        Include the C nonstationarity node in the CDNOTS candidate graph.
    include_C_in_model : bool, default False
        Append the C basis to the gated model's inputs; the grow and self-loop
        tests then condition on it, also for a supplied graph.
    c_preset : str, default "linear"
        C basis preset (as in ``run_cdnots``).
    data_transform : {"rank", None}, default "rank"
        Map continuous columns to normal scores of their ranks first.
    impute, impute_kwargs : optional
        Missing-value strategy (``"var_em"``) and its options.
    discrete_cols : list or None
        Discrete columns (detected when None).
    grow_rounds : int, default 1
        Grow rounds after the first fit (0 = off).
    grow_q : float, default 0.1
        Benjamini-Hochberg level of the grow tests.
    grow_transform : {None, "rank"}, default None
        Run the grow tests on rank-Gaussianized data.
    grow_variance : bool or None, default None
        Also grow edges that change a target's noise scale; None means True
        when ``grow_rounds >= 1``.
    grow_nuisance_basis : {"spline", "rff", None}, default "spline"
        Basis of the conditioning terms in the grow and self-loop tests.
    effect_floor : float, default ``EFFECT_FLOOR`` (about 0.0198)
        Minimum adjusted partial R^2 of a grown or final cross edge; 0 turns it
        off.
    ar_path : bool or None, default None
        Ungated autoregressive head with F-tested self-loops; None means True
        when ``max_lag >= 1``.
    self_loop_basis : {"union", "rff"}, default "union"
        Tested block of the self-loop test.
    gate_threshold : float, "auto" or "li", default "auto"
        Cutoff on the cross-edge gates.
    lambda_rule : {"xcorr", "density", "dT"}, default "xcorr"
        Closed-form rule for the L0 strength.
    lambda_cv : bool, default False
        Choose the L0 strength by cross-validation instead.
    noise_model : {"student_t", "gaussian"}, default "student_t"
        Likelihood of the prediction errors.
    normalize : bool, default True
        Z-score the data.
    max_epochs, batch_size, patience : training options
        Epochs, batch size (None = auto) and early-stopping patience.
    device : str or None
        ``"cuda"``, ``"mps"``, ``"cpu"``, or None to detect.
    model_seed : int or None
        Seed of the gated model.
    verbose : bool, default True
        Print progress.
    ground_truth : numpy.ndarray or None
        Evaluate the result against it; the metrics are printed when ``verbose``.
    candidate_runtime : float, default 0.0
        Wall-clock time spent building a supplied ``candidate_graph`` (reported in
        the result's runtime).
    ci_test : object, optional
        Accepted so ``causalts.algorithms.run_algorithm`` can pass its CI-test
        instance; not used (a warning says so). Use ``ci_test_class``.
    **model_kwargs
        Forwarded to :class:`~causalts.grace.gated_discovery.GatedCausalDiscovery`.

    Returns
    -------
    GraceResult
        ``cg_tig`` (the graph), ``gate_values``, ``candidate_graph`` (the refined
        candidate set, including grow additions), ``lambda_l0``,
        ``gate_threshold``, ``grow_history``, ``effect_pruned`` and the shared
        plotting and DoWhy methods.
    """
    _reject_foreign_kwargs(model_kwargs, "run_grace")
    n_c = 0
    if include_C and include_C_in_model and candidate_graph is not None:
        from ..cdnots.phase3_utils import make_c_array

        n_c = np.asarray(make_c_array(max_lag + 2, c_preset)).reshape(max_lag + 2, -1)
        n_c = n_c.shape[1]
    candidate_graph = _check_candidate_graph(candidate_graph, df, max_lag, n_c)
    _warn_unused_ci_test(ci_test, "run_grace")
    return run_cdnots_gated(
        df,
        max_lag,
        alpha=alpha,
        ci_test_class=ci_test_class,
        skeleton=candidate_graph,
        skeleton_runtime=candidate_runtime,
        gate_threshold=gate_threshold,
        normalize=normalize,
        max_epochs=max_epochs,
        batch_size=batch_size,
        patience=patience,
        device=device,
        verbose=verbose,
        ground_truth=ground_truth,
        model_seed=model_seed,
        include_C=include_C,
        include_C_in_model=include_C_in_model,
        c_preset=c_preset,
        lambda_cv=lambda_cv,
        lambda_rule=lambda_rule,
        impute=impute,
        impute_kwargs=impute_kwargs,
        discrete_cols=discrete_cols,
        grow_rounds=grow_rounds,
        grow_q=grow_q,
        grow_transform=grow_transform,
        data_transform=data_transform,
        noise_model=noise_model,
        refine="gates",
        grow_variance=grow_variance,
        grow_nuisance_basis=grow_nuisance_basis,
        effect_floor=effect_floor,
        ar_path=ar_path,
        self_loop_basis=self_loop_basis,
        **model_kwargs,
    )


def run_reap(
    df,
    max_lag: int,
    *,
    candidate_graph: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    ci_test_class=None,
    include_C: bool = True,
    include_C_in_model: bool = False,
    c_preset: str = "linear",
    data_transform: Optional[str] = "rank",
    impute: Optional[str] = None,
    impute_kwargs: Optional[dict] = None,
    discrete_cols=None,
    prune_alpha: float = 0.001,
    basis: str = "spline",
    robust: bool = True,
    interactions: bool = False,
    effect_floor: Union[None, str, float] = None,
    grow_rounds: int = 3,
    grow_q: float = 0.1,
    grow_transform: Optional[str] = "rank",
    grow_nuisance_basis: Optional[str] = "spline",
    contemporaneous: bool = False,
    contemporaneous_test="spline",
    contemporaneous_alpha: float = 0.001,
    verbose: bool = True,
    ground_truth: Optional[np.ndarray] = None,
    candidate_runtime: float = 0.0,
    ci_test=None,
    **kwargs,
):
    """REAP: refine a candidate graph by regression-based backward elimination.

    1. **Candidate graph.** ``candidate_graph`` if given (any method's graph),
       else a CDNOTS graph on the data (partial correlation at level ``alpha``,
       with the C node when ``include_C``).
    2. **Pruning.** For each target, its lagged candidate parents enter an
       additive regression, one ``basis`` block per parent; the weakest parent is
       dropped while a partial F-test at ``prune_alpha`` says it adds nothing, and
       the fit is redone (:func:`~causalts.grace.shrink.shrink_parents`). Nothing
       is trained. ``grow_rounds`` rounds (three by default, on rank-transformed
       data) test the lagged pairs outside the candidates, add the hits and prune
       again.
    3. **Optional steps.** ``effect_floor`` removes cross edges with a small
       unique effect; ``contemporaneous`` adds lag-0 edges with CDNOTS+, the
       lagged edges pinned to REAP's selection.

    Same as ``run_cdnots_gated(..., refine="shrink")`` with ``shrink_alpha``,
    ``shrink_basis``, ``shrink_robust``, ``shrink_interactions`` and
    ``shrink_effect_floor`` renamed to ``prune_alpha``, ``basis``, ``robust``,
    ``interactions`` and ``effect_floor``; see
    :func:`~causalts.grace.gated_discovery.run_cdnots_gated` for the full
    description of each option. Lagged edges only unless ``contemporaneous``.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data, shape (T, d).
    max_lag : int
        Maximum lag.
    candidate_graph : numpy.ndarray or None
        Directed candidate graph, shape (d, d, max_lag+1), with a non-zero
        ``candidate_graph[cause, effect, lag]`` marking ``cause(t-lag) ->
        effect(t)`` (weights are read as edges where non-zero). None builds a CDNOTS graph. The grow rounds can add lagged
        edges outside it; with ``grow_rounds=0`` the lagged edges of the result
        are inside it.
    alpha : float, default 0.05
        CDNOTS significance level for the candidate graph (unused when
        ``candidate_graph`` is given).
    ci_test_class : class or None
        CI test class for the CDNOTS candidate graph (default partial
        correlation).
    include_C : bool, default True
        Include the C nonstationarity node in the CDNOTS candidate graph; the
        pruning and grow regressions then condition on the C basis.
    include_C_in_model : bool, default False
        Condition the regressions on the C basis also for a supplied graph.
    c_preset : str, default "linear"
        C basis preset (as in ``run_cdnots``).
    data_transform : {"rank", None}, default "rank"
        Map continuous columns to normal scores of their ranks first.
    impute, impute_kwargs : optional
        Missing-value strategy (``"var_em"``) and its options; REAP needs
        complete data.
    discrete_cols : list or None
        Discrete columns (detected when None).
    prune_alpha : float, default 0.001
        Significance level for keeping a parent. The p-values are approximate
        (fixed-basis F-tests) and become liberal at large T when effects are
        strongly curved.
    basis : {"spline", "rff"}, default "spline"
        Feature block per lagged parent, for the pruning and the grow tests.
    robust : bool, default True
        Huber-clip residuals before each test.
    interactions : bool, default False
        Add pairwise products of each target's parents to the regressions.
    effect_floor : None, "adaptive" or float, default None
        After pruning and growing, remove lagged cross edges with a small unique
        effect: ``"adaptive"`` per target, below ``max(0.01, 0.2 * strongest)``;
        a float is a fixed floor; None keeps REAP's graph. Not available with
        ``interactions=True``.
    grow_rounds : int, default 3
        Grow rounds after the first pruning (0 = off); each stops the loop early
        once it adds nothing.
    grow_q : float, default 0.1
        Benjamini-Hochberg level of the grow tests.
    grow_transform : {"rank", None}, default "rank"
        Run the grow tests on rank-Gaussianized data.
    grow_nuisance_basis : {"spline", "rff", None}, default "spline"
        Basis of the conditioning terms in the grow tests.
    contemporaneous : bool, default False
        Add contemporaneous (lag-0) edges with CDNOTS+, lagged edges pinned.
    contemporaneous_test : str or CIT_Base, default "spline"
        CI test of the lag-0 step.
    contemporaneous_alpha : float, default 0.001
        Significance level of the lag-0 step.
    verbose : bool, default True
        Print progress.
    ground_truth : numpy.ndarray or None
        Evaluate the result against it; the metrics are printed when ``verbose``.
    candidate_runtime : float, default 0.0
        Wall-clock time spent building a supplied ``candidate_graph``.
    ci_test : object, optional
        Accepted so ``causalts.algorithms.run_algorithm`` can pass its CI-test
        instance; not used (a warning says so). Use ``ci_test_class``.
    **kwargs
        Not accepted: a ``TypeError`` names the ``run_reap`` argument (for
        ``skeleton`` and the ``shrink_*`` names) or points to ``run_grace`` (for
        gate and training options).

    Returns
    -------
    GraceResult
        ``cg_tig`` (the graph), ``edge_pvalue`` and ``edge_strength`` (REAP's
        tests), ``candidate_graph``, ``grow_history``, ``effect_pruned`` and the
        shared plotting and DoWhy methods; ``gate_values`` is None.
    """
    _reject_foreign_kwargs(kwargs, "run_reap")
    candidate_graph = _check_candidate_graph(candidate_graph, df, max_lag)
    _check_reap_options(prune_alpha, basis, interactions, effect_floor)
    _warn_unused_ci_test(ci_test, "run_reap")
    return run_cdnots_gated(
        df,
        max_lag,
        alpha=alpha,
        ci_test_class=ci_test_class,
        skeleton=candidate_graph,
        skeleton_runtime=candidate_runtime,
        verbose=verbose,
        ground_truth=ground_truth,
        include_C=include_C,
        include_C_in_model=include_C_in_model,
        c_preset=c_preset,
        impute=impute,
        impute_kwargs=impute_kwargs,
        discrete_cols=discrete_cols,
        grow_rounds=grow_rounds,
        grow_q=grow_q,
        grow_transform=grow_transform,
        data_transform=data_transform,
        refine="shrink",
        shrink_alpha=prune_alpha,
        shrink_basis=basis,
        shrink_robust=robust,
        shrink_interactions=interactions,
        grow_nuisance_basis=grow_nuisance_basis,
        shrink_effect_floor=effect_floor,
        contemporaneous=contemporaneous,
        contemporaneous_test=contemporaneous_test,
        contemporaneous_alpha=contemporaneous_alpha,
    )
