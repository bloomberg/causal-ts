# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Post-processing utility to introduce latent confounders into SCP samples.

Usage::

    from causalts.synthetic_data.synthetic_datasets import erdos_renyi
    from causalts.synthetic_data.confounding import apply_confounding

    sample = erdos_renyi(seed=42, n_vars=15, T=500)
    confounded = apply_confounding(sample, confound_fraction=0.3, seed=42)

    confounded["df"]                  # observed variables only
    confounded["ground_truth"]        # direct GT (confounder edges removed)
    confounded["ground_truth_ancestral"]  # ancestral GT (transitive closure)
    confounded["confounder_nodes"]    # list of removed node names
    confounded["confounded_pairs"]    # [(Xi, Xj), ...] sharing a latent cause
"""

from __future__ import annotations

import math
from copy import deepcopy

import numpy as np


def _find_eligible_confounders(gt: np.ndarray) -> list[int]:
    """Return node indices that cause >=2 distinct other variables (fork)."""
    d = gt.shape[0]
    eligible = []
    for i in range(d):
        children = set()
        for j in range(d):
            if i == j:
                continue
            if gt[i, j, :].any():
                children.add(j)
        if len(children) >= 2:
            eligible.append(i)
    return eligible


def _transitive_closure_for_removed(
    gt_full: np.ndarray,
    removed: set[int],
    observed: list[int],
) -> np.ndarray:
    """Build ancestral ground truth: connect L's parents to L's children.

    For each removed node L, for each parent P of L and each child C of L
    (both in observed set), add edge P -> C at the sum of lags (capped at
    max_lag).
    """
    d_obs = len(observed)
    max_lag = gt_full.shape[2] - 1
    obs_set = set(observed)
    obs_idx = {node: i for i, node in enumerate(observed)}

    gt_anc = np.zeros((d_obs, d_obs, max_lag + 1), dtype=np.int8)

    # Copy direct edges between observed nodes
    for i, ni in enumerate(observed):
        for j, nj in enumerate(observed):
            gt_anc[i, j, :] = gt_full[ni, nj, :]

    # Add transitive edges through removed nodes
    for L in removed:
        parents_lags = []
        children_lags = []
        for p in range(gt_full.shape[0]):
            if p == L:
                continue
            for lag in range(max_lag + 1):
                if gt_full[p, L, lag]:
                    parents_lags.append((p, lag))
        for c in range(gt_full.shape[0]):
            if c == L:
                continue
            for lag in range(max_lag + 1):
                if gt_full[L, c, lag]:
                    children_lags.append((c, lag))

        for p, lag_p in parents_lags:
            if p not in obs_set:
                continue
            for c, lag_c in children_lags:
                if c not in obs_set:
                    continue
                combined_lag = min(lag_p + lag_c, max_lag)
                gt_anc[obs_idx[p], obs_idx[c], combined_lag] = 1

    return gt_anc


def apply_confounding(
    sample: dict,
    confound_fraction: float = 0.3,
    seed: int | None = None,
) -> dict:
    """Remove a fraction of eligible nodes as latent confounders.

    Parameters
    ----------
    sample : dict
        Output of ``SCPGraphGenerator.sample()`` or ``topology_scp()`` etc.
        Must contain keys: ``df``, ``ground_truth``, ``var_names``, ``max_lag``.
    confound_fraction : float
        Fraction of eligible nodes (those with >=2 distinct children) to
        remove as latent confounders.  0.0 = no confounders, 1.0 = remove
        all eligible nodes.
    seed : int or None
        Random seed for confounder selection.

    Returns
    -------
    dict
        Modified sample with keys:

        - ``df`` — observed variables only (confounder columns dropped)
        - ``ground_truth`` — 3D array, observed nodes only, confounder edges removed
        - ``ground_truth_ancestral`` — same but with transitive closure through
          removed nodes (L's parents connected to L's children)
        - ``ground_truth_full`` — original full graph (for reference)
        - ``confounder_nodes`` — list of removed node names
        - ``confounder_indices`` — list of removed node indices
        - ``confounded_pairs`` — list of (name_i, name_j) pairs that share
          a latent common cause
        - ``observed_nodes`` — list of observed node names
        - ``n_confounders`` — number of removed nodes
        - ``n_eligible`` — number of nodes eligible to be confounders
        - All other keys from the original sample are preserved.
    """
    gt_full = sample["ground_truth"].copy()
    var_names = list(sample["var_names"])
    d = len(var_names)
    max_lag = sample["max_lag"]

    eligible = _find_eligible_confounders(gt_full)
    n_eligible = len(eligible)

    if n_eligible == 0:
        out = deepcopy(sample)
        out["ground_truth_full"] = gt_full
        out["ground_truth_ancestral"] = gt_full.copy()
        out["confounder_nodes"] = []
        out["confounder_indices"] = []
        out["confounded_pairs"] = []
        out["observed_nodes"] = var_names
        out["n_confounders"] = 0
        out["n_eligible"] = 0
        return out

    n_confounders = max(1, math.ceil(confound_fraction * n_eligible))
    n_confounders = min(n_confounders, n_eligible)

    rng = np.random.default_rng(seed)
    confounder_indices = sorted(rng.choice(eligible, size=n_confounders, replace=False))
    removed = set(confounder_indices)
    observed = [i for i in range(d) if i not in removed]

    # Confounded pairs: observed children that share a removed parent
    confounded_pairs = []
    for L in confounder_indices:
        children_obs = []
        for c in range(d):
            if c == L or c in removed:
                continue
            if gt_full[L, c, :].any():
                children_obs.append(c)
        for ci in range(len(children_obs)):
            for cj in range(ci + 1, len(children_obs)):
                pair = (var_names[children_obs[ci]], var_names[children_obs[cj]])
                confounded_pairs.append(pair)

    # Build observed ground truth (direct)
    d_obs = len(observed)
    gt_obs = np.zeros((d_obs, d_obs, max_lag + 1), dtype=np.int8)
    for i, ni in enumerate(observed):
        for j, nj in enumerate(observed):
            gt_obs[i, j, :] = gt_full[ni, nj, :]

    # Build ancestral ground truth (transitive closure through removed)
    gt_anc = _transitive_closure_for_removed(gt_full, removed, observed)

    # Build observed DataFrame
    obs_names = [var_names[i] for i in observed]
    df_obs = sample["df"][obs_names].copy()

    out = deepcopy(sample)
    out["df"] = df_obs
    out["ground_truth"] = gt_obs
    out["ground_truth_ancestral"] = gt_anc
    out["ground_truth_full"] = gt_full
    out["var_names"] = obs_names
    out["confounder_nodes"] = [var_names[i] for i in confounder_indices]
    out["confounder_indices"] = list(confounder_indices)
    out["confounded_pairs"] = confounded_pairs
    out["observed_nodes"] = obs_names
    out["n_confounders"] = n_confounders
    out["n_eligible"] = n_eligible
    out["name"] = sample.get("name", "") + f" (confounded, {n_confounders} latent)"

    return out
