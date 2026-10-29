# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: py311
#     language: python
#     name: python3
# ---

# %% [markdown]
# # GRACE Ablations: Lag Misspecification and Non-Additive Data
#
# Two focused ablation experiments, both with GRACE at its defaults, starting from the CDNOTS graph:
#
# **Part A — Lag Misspecification:** In practice the true max causal lag is unknown. We fix the true max_lag=5 and run with assumed lags of 2, 3, 5, 7 and 10.
#
# **Part B — Lorenz-96 (Non-Additive):** Lorenz-96 has multiplicative interactions, outside GRACE's additive model class.
#
# Lorenz-96 update rule: $x_i(t+1) = (x_{i-1} - x_{i-2}) \cdot x_{i+1} - x_i + F$

# %%
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from causalts.synthetic_data.synthetic_datasets import SCPGraphGenerator, lorenz96
from causalts.ci_tests import PartialCorr
from causalts import run_cdnots
from causalts import run_grace
from causalts.utils import evaluate_graph

# %% [markdown]
# ---
# ## Part A — Lag Misspecification
#
# Data: d=50, T=1000, true max_lag=5, sparse SCP with mixed linear/nonlinear functions.

# %%
TRUE_MAX_LAG = 5
DATA_SEED = 0

gen = SCPGraphGenerator(
    n_vars=50, max_lag=TRUE_MAX_LAG, density="sparse",
    dependency_funcs=("lin", "lin", "nl1", "nl2"),
)
result = gen.sample(seed=DATA_SEED, T=1000, max_tries=500)
df = result["df"]
ground_truth = result["ground_truth"]  # shape (50, 50, 6)

print(f"Dataset: {df.shape[0]} × {df.shape[1]} variables, true max_lag={TRUE_MAX_LAG}")
print(f"True edges: {int(ground_truth.sum())}")

# %%
assumed_lags = [2, 3, 5, 7, 10]
n_vars = df.shape[1]
records = []


def padded(G, L):
    """Binary [cause, effect, lag] graph over lags 0..L (missing lags are empty)."""
    out = np.zeros((n_vars, n_vars, L + 1), dtype=int)
    k = min(np.asarray(G).shape[2], L + 1)
    out[:, :, :k] = np.asarray(G)[:n_vars, :n_vars, :k] != 0
    return out


for assumed_lag in assumed_lags:
    print(f"\n=== Assumed max_lag = {assumed_lag} (true = {TRUE_MAX_LAG}) ===")

    # Score every lag either side could use: under-specification counts the edges
    # beyond assumed_lag as missed, over-specification counts long-lag edges as false.
    L_eval = max(TRUE_MAX_LAG, assumed_lag)
    gt = padded(ground_truth, L_eval)

    # CDNOTS
    ci = PartialCorr(np.ascontiguousarray(df.values))
    t0 = time.time()
    _r = run_cdnots(
        df=df, indep_test=ci, num_lags=assumed_lag,
        include_C=True, alpha=0.05, stable=True,
    )
    t_c = time.time() - t0
    g_cdnots = _r.cg_tig[:n_vars, :n_vars, :]
    m_c = evaluate_graph(padded(g_cdnots, L_eval), gt)

    # GRACE starting from the CDNOTS graph (its grow round can add edges CDNOTS missed)
    t0 = time.time()
    _gr = run_grace(
        df, max_lag=assumed_lag, candidate_graph=g_cdnots,
        model_seed=DATA_SEED, verbose=False,
    )
    t_g = time.time() - t0
    m_g = evaluate_graph(padded(_gr.cg_tig, L_eval), gt)

    print(f"  CDNOTS: F1={m_c['F1']:.3f}  Prec={m_c['Precision']:.3f}  SHD={m_c['SHD']}  ({t_c:.0f}s)")
    print(f"  GRACE:  F1={m_g['F1']:.3f}  Prec={m_g['Precision']:.3f}  SHD={m_g['SHD']}  ({t_g:.0f}s)")

    records.append(dict(
        assumed_lag=assumed_lag,
        cdnots_f1=m_c["F1"], cdnots_prec=m_c["Precision"], cdnots_shd=m_c["SHD"],
        grace_f1=m_g["F1"], grace_prec=m_g["Precision"], grace_shd=m_g["SHD"],
    ))

# %%
df_res = pd.DataFrame(records)
lags = df_res["assumed_lag"]

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, col_c, col_g, ylabel in zip(
    axes,
    ["cdnots_f1", "cdnots_prec", "cdnots_shd"],
    ["grace_f1", "grace_prec", "grace_shd"],
    ["F1", "Precision", "SHD"],
):
    ax.plot(lags, df_res[col_c], "o--", color="#6b7280", label="CDNOTS")
    ax.plot(lags, df_res[col_g], "s-", color="#4B44B8", label="GRACE", linewidth=2)
    ax.axvline(TRUE_MAX_LAG, color="red", linestyle=":", alpha=0.6, label=f"True lag={TRUE_MAX_LAG}")
    ax.set_xlabel("Assumed max_lag")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs Assumed Max Lag")
    ax.legend()
    if ylabel != "SHD":
        ax.set_ylim(0, 1)

plt.tight_layout()
plt.show()
print(df_res.to_string(index=False))

# %% [markdown]
# **Takeaway:** GRACE has higher F1 and lower SHD than CDNOTS at every assumed lag. Over-specifying the lag costs GRACE little (F1 0.963 at the true lag, 0.949 at lag 10; SHD 8 → 11). Under-specifying costs both methods much more, because edges beyond the assumed lag cannot be found (at lag 2, GRACE's F1 is 0.691 and CDNOTS's is 0.559). When in doubt, over-specify `max_lag`.

# %% [markdown]
# ---
# ## Part B — Lorenz-96 (Non-Additive Dynamics)
#
# Lorenz-96 has multiplicative causal mechanisms. Ground truth: each node depends on exactly 4 neighbors at lag 1 (cyclic). Parameters: N=50, T=1000, F=10.0, dt=0.1, obs_noise=0.1, burn_in=1000.

# %%
result_l96 = lorenz96(seed=0, T=1000, N=50, F=10.0, dt=0.1, obs_noise=0.1, burn_in=1000)
df_l96 = result_l96["df"]
gt_l96 = result_l96["ground_truth"]  # (50, 50, 2)

print(f"Lorenz-96: N={df_l96.shape[1]}, T={df_l96.shape[0]}")
print(f"True edges: {int(gt_l96.sum())} (4 parents × {df_l96.shape[1]} nodes at lag 1)")

# %%
# CDNOTS
import tempfile as _tf
ci_l96 = PartialCorr(np.ascontiguousarray(df_l96.values),
                     cache_dir=_tf.mkdtemp(prefix="causalts_l96_"))
t0 = time.time()
_r = run_cdnots(df=df_l96, indep_test=ci_l96, num_lags=1,
                          include_C=True, alpha=0.05, stable=True)
g_c_l96 = _r.cg_tig
t_c_l96 = time.time() - t0
g_c_l96 = g_c_l96[:50, :50, :]
m_c_l96 = evaluate_graph(g_c_l96, gt_l96)

# GRACE
t0 = time.time()
_gr_l96 = run_grace(
    df_l96, max_lag=1,
    candidate_graph=g_c_l96[:50, :50, :],  # starting point; the grow round can add edges
    model_seed=0, verbose=True, ground_truth=gt_l96,
)
t_g_l96 = time.time() - t0
G_l96 = _gr_l96.cg_tig[:50, :50, :]
m_g_l96 = evaluate_graph(G_l96, gt_l96)

print(f"\n{'Method':<10} {'F1':>6} {'Prec':>8} {'Recall':>8} {'SHD':>6} {'Time':>8}")
print("-" * 50)
print(f"{'CDNOTS':<10} {m_c_l96['F1']:>6.3f} {m_c_l96['Precision']:>8.3f} "
      f"{m_c_l96['TPR']:>8.3f} {m_c_l96['SHD']:>6} {t_c_l96:>7.0f}s")
print(f"{'GRACE':<10} {m_g_l96['F1']:>6.3f} {m_g_l96['Precision']:>8.3f} "
      f"{m_g_l96['TPR']:>8.3f} {m_g_l96['SHD']:>6} {t_g_l96:>7.0f}s")

# %% [markdown]
# **Takeaway:** On Lorenz-96, GRACE reaches F1 0.948 against CDNOTS's 0.751 despite the multiplicative interactions. CDNOTS's linear CI test misses 43 of the 200 true edges. GRACE's grow round tests the pairs outside the CDNOTS graph and adds 117 candidates; 68 of them survive retraining. The final graph has all 200 true edges and 22 false ones.

# %% [markdown]
#
