# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: .venv-nbrun
#     language: python
#     name: python3
# ---

# %% [markdown]
# # CDNOTS or CDNOTS+?
#
# Both discover causal structure in nonstationary time series, and both use the
# same surrogate variable `C` to absorb distribution shift. They differ in **how
# the skeleton is searched**.
#
# - **CDNOTS** uses a PC-style search: iterate over all pairs at increasing
#   conditioning-set depth. At a hub of degree 22 and depth 5 that is
#   $\binom{21}{5} = 20{,}349$ conditioning sets for a *single* pair. With finite
#   samples, some subset will make a true edge look independent, and the edge is
#   removed.
# - **CDNOTS+** uses PCMCI+'s two-phase search: phase 1 finds a superset of each
#   variable's lagged parents; phase 2 re-tests each link conditioning only on
#   *those* parents plus contemporaneous subsets. Conditioning sets stay small,
#   so power survives at hubs.
#
# ## TL;DR — which to use
#
# | your graph | use |
# |---|---|
# | hubs (scale-free), or dense | **CDNOTS+** — up to +0.20 F1 |
# | moderate density, limited data | **CDNOTS+** — it is never materially worse |
# | sparse, homogeneous, lots of data | either — they tie |
#
# CDNOTS+ is the safer default. The rest of this notebook shows why, and where
# the difference comes from.

# %%
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt

from causalts import run_cdnots, run_cdnots_plus
from causalts.ci_tests.parcorr_gpu import ParCorrGPU
from causalts.synthetic_data.synthetic_datasets import topology_scp
from causalts.utils import evaluate_graph

DEVICE = 'cpu'   # 'cuda' if available
ALPHA  = 0.01

# %% [markdown]
# ## 1. The effect, live
#
# A dense scale-free graph — the case CDNOTS+ was built for. `include_C=False`
# here so nothing depends on the surrogate variable: this is purely about
# skeleton search.

# %%
rows = []
for seed in range(3):
    ds = topology_scp('scale_free', seed=seed, n_vars=30, T=500, max_lag=3, ba_m=7)
    gt = ds['ground_truth']
    for name, fn in [('CDNOTS', run_cdnots), ('CDNOTS+', run_cdnots_plus)]:
        ci = ParCorrGPU(np.zeros((2, 2)), device=DEVICE)
        kw = dict(stable=True) if name == 'CDNOTS' else {}
        g = fn(ds['df'], ci, num_lags=3, include_C=False, alpha=ALPHA,
               show_progress=False, **kw).cg_tig
        m = evaluate_graph(g, gt)
        rows.append((seed, name, m['F1'], m['Precision'], m['TPR']))

print(f"{'seed':>4s} {'method':10s} {'F1':>7s} {'Prec':>7s} {'Recall':>7s}")
for r in rows:
    print(f"{r[0]:4d} {r[1]:10s} {r[2]:7.3f} {r[3]:7.3f} {r[4]:7.3f}")

# %% [markdown]
# CDNOTS+ wins, and the columns say *how*: precision is similar, **recall is
# much higher**. CDNOTS is not making fewer mistakes — it is finding fewer
# edges, because power collapsed at the hubs.
#
# Over the full study below (9,480 runs):
#
# | | F1 | Precision | Recall | edges found (of 99) |
# |---|---|---|---|---|
# | CDNOTS | 0.765 | 0.884 | 0.700 | 66 |
# | CDNOTS+ | 0.846 | 0.899 | 0.824 | 80 |
#
# Same precision, 14 more true edges recovered.

# %% [markdown]
# ## 2. When does it help? Density and sample size
#
# We swept graph density continuously — three topologies, mean degree 1.5 to
# 13.5, $d \in \{10,15,30,40\}$, $T \in \{200,\dots,2000\}$, 10 seeds, 9,480 runs.
# Two variables govern the gain, and dimension is not one of them.

# %%
# Pre-computed summary of the full sweep (gap = F1(CDNOTS+) - F1(CDNOTS),
# averaged within unit-width mean-degree bins).
SUMMARY = {"scale_free": {"200": [[1.5,0.0936],[2.5,0.1712],[3.5,0.1816],[4.5,0.182],[5.5,0.1845],[6.5,0.196],[7.5,0.2095]],"500": [[1.5,0.0491],[2.5,0.1238],[3.5,0.1592],[4.5,0.1845],[5.5,0.215],[6.5,0.2326],[7.5,0.2866]],"1000": [[1.5,0.032],[2.5,0.098],[3.5,0.1418],[4.5,0.1619],[5.5,0.174],[6.5,0.2203],[7.5,0.2881]],"2000": [[1.5,0.0184],[2.5,0.0826],[3.5,0.117],[4.5,0.1283],[5.5,0.1335],[6.5,0.1966],[7.5,0.2566]]},"erdos_renyi": {"200": [[1.5,0.0174],[2.5,0.0439],[3.5,0.042],[4.5,0.0743],[5.5,0.0685],[6.5,0.0328],[7.5,0.0476],[8.5,0.0177],[9.5,0.0197],[12.5,0.0105]],"500": [[1.5,0.0089],[2.5,0.0076],[3.5,0.0292],[4.5,0.0325],[5.5,0.0359],[6.5,0.0249],[7.5,0.0239],[8.5,0.0242],[9.5,0.0265],[12.5,0.0353]],"1000": [[1.5,0.0079],[2.5,0.0028],[3.5,0.0096],[4.5,0.0174],[5.5,0.0132],[6.5,0.0141],[7.5,0.0035],[8.5,0.0154],[9.5,0.0225],[12.5,0.031]],"2000": [[1.5,0.0047],[2.5,-0.0023],[3.5,0.0022],[4.5,-0.0056],[5.5,0.0176],[6.5,0.0027],[7.5,0.0068],[8.5,-0.0023],[9.5,0.0124],[12.5,0.0207]]},"small_world": {"200": [[2.5,0.0437],[3.5,0.0739],[4.5,0.1021],[5.5,0.1504]],"500": [[2.5,0.0386],[3.5,0.0524],[4.5,0.1052],[5.5,0.1662]],"1000": [[2.5,0.0106],[3.5,0.044],[4.5,0.0865],[5.5,0.1438]],"2000": [[2.5,0.0048],[3.5,0.0275],[4.5,0.0598],[5.5,0.132]]}}

RAMP = ['#a8c9ee', '#6ba3e0', '#2a78d6', '#14508f']
titles = {'scale_free': 'Scale-free', 'erdos_renyi': 'Erdős–Rényi',
          'small_world': 'Small-world'}

fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)
for ax, topo in zip(axes, ['scale_free', 'erdos_renyi', 'small_world']):
    for i, T in enumerate(['200', '500', '1000', '2000']):
        pts = SUMMARY[topo][T]
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                color=RAMP[i], lw=2, marker='o', ms=4, label=f'T={T}')
    ax.axhline(0, color='#8a8a8a', lw=0.8)
    ax.set_title(titles[topo]); ax.set_xlabel('mean degree')
    ax.grid(alpha=0.25, lw=0.5)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
axes[0].set_ylabel('F1(CDNOTS+) - F1(CDNOTS)')
axes[0].legend(frameon=False, fontsize=9)
plt.tight_layout(); plt.show()

# %% [markdown]
# **The gain grows with density and shrinks with sample size**, in all three
# topologies. It is largest where hubs exist (scale-free) and smallest on
# Erdős–Rényi, which has none. Range: +0.018 (sparse scale-free, T=2000) to
# +0.199 (dense scale-free, T=200).
#
# **Dimension does almost nothing.** Paired ΔF1 by topology and $d$:
#
# | topology | d=10 | d=15 | d=30 | d=40 | overall |
# |---|---|---|---|---|---|
# | scale-free | +0.144 | +0.150 | +0.151 | +0.150 | **+0.149** |
# | small-world | +0.131 | +0.091 | +0.041 | +0.015 | +0.073 |
# | Erdős–Rényi | +0.018 | +0.017 | +0.016 | +0.015 | +0.017 |
#
# Scale-free is flat. Small-world's decline is *not* a dimension effect — the
# `ws_k` parameter fixes mean degree regardless of $d$, so a bigger small-world
# graph is a relatively sparser one. Read on the density axis it is the same
# curve as the others.

# %% [markdown]
# ## 3. CDNOTS+ and PCMCI+ agree
#
# CDNOTS+ borrows PCMCI+'s skeleton search, so on stationary data it should
# behave like PCMCI+ — and it does. Over 3,153 paired instances the mean
# difference is **+0.0008 F1**, the median is exactly zero, and the two return
# **identical graphs on 53%** of instances.
#
# That matters because it means CDNOTS+ gives you PCMCI+'s skeleton behaviour
# *plus* the `C` node, which PCMCI+ has no mechanism for.

# %% [markdown]
# ## 4. Choosing alpha
#
# Both methods prefer `alpha=0.01` over `0.05` on this benchmark
# (CDNOTS 0.753 vs 0.733 overall), though the margin is topology-dependent —
# CDNOTS alone does better at 0.05 on scale-free. The library default is 0.05;
# consider 0.01 for either method.
#
# ---
#
# ### Caveats
#
# All numbers here are synthetic, with a fixed structural form, Gaussian noise,
# `max_lag=3` and partial correlation as the CI test. `include_C=False`
# throughout, so this notebook says nothing about the surrogate variable's own
# behaviour — it is purely a comparison of skeleton search. A nonparametric CI
# test could shift the balance, since the two methods differ only in *which*
# conditioning sets they test.
