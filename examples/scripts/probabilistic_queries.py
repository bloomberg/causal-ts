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
# # Probabilistic Queries on Causal Graphs
#
# After discovering a causal graph, we often want to answer probabilistic "what-if" questions:
#
# - **Interventional:** "What if we **set** PKA one step ago to 2?" (do-operator, severs incoming edges)
# - **Observational:** "Given we **observe** Mek≈3 and PKA≈2, what is the distribution of the other variables?" (conditioning, keeps the graph intact)
#
# The two can give different answers. An intervention changes only the descendants of the intervened nodes, while conditioning also updates beliefs about their causes.
#
# **Features demonstrated:**
# 1. Interventional queries via `do()`-operator sampling
# 2. Observational conditioning (soft and hard)
# 3. Query summaries with weighted percentiles
# 4. Comparing interventional vs observational answers
# 5. Kernel bandwidth for soft conditioning
# 6. The standalone query functions

# %%
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from causalts.synthetic_data.synthetic_datasets import ex3
from causalts.effects import wrap_graph, dowhy_available

print('DoWhy available:', dowhy_available())

# The query samplers draw from NumPy's global random state; seed it for reproducibility
np.random.seed(0)

# %% [markdown]
# ## 1. Discover Graph and Fit SCM
#
# We use **ex3** (an 11-node cascade network) and discover the graph with CDNOTS. The query methods live on `WrappedGraph`, so we wrap the discovered `(d, d, max_lag+1)` array with `wrap_graph`.

# %%
from causalts import run_cdnots
from causalts.ci_tests import PartialCorr

# Generate data
result = ex3(seed=42, T=500)
df = result['df']
var_names = df.columns.tolist()
print(f'Variables: {var_names}')
print(f'Shape: T={len(df)}, d={df.shape[1]}')

# Discover graph
ci = PartialCorr(data=np.zeros((2, 2)))
_res = run_cdnots(
    df, ci, num_lags=3, include_C=False, alpha=0.05, stable=True
)
disc_graph = _res.cg_tig

# Wrap for the query API
cg_wrap = wrap_graph(disc_graph, df)
print(cg_wrap)

# %% [markdown]
# The SCM is fitted on the **lag-embedded transition DAG**. Each variable appears as a current-time node (`Mek`) and as lagged copies (`Mek_lag1`, `Mek_lag2`, `Mek_lag3`). Every discovered edge points from a lagged node to a current-time node, so:
#
# - current-time nodes have parents but no children (no contemporaneous edges were found here), and
# - lagged nodes are root nodes. They are sampled independently from their empirical marginal distributions, so the dependence between different past values (for example `Mek_lag1` and `Mek_lag2`) is not modelled.
#
# An intervention therefore has a visible effect only when it targets a lagged node. Setting a current-time node changes nothing else in the model.

# %%
# Fit the SCM (cached for subsequent queries)
scm, dag, lagged_df = cg_wrap.fit_scm(mechanism_type='linear')
print(f'SCM: {len(dag.nodes)} nodes, {len(dag.edges)} edges')
print(f'Lagged DataFrame: {lagged_df.shape}')

print(f'\nParents of Mek:       {sorted(dag.predecessors("Mek"))}')
print(f'Children of Mek:      {sorted(dag.successors("Mek"))}')
print(f'Children of Mek_lag2: {sorted(dag.successors("Mek_lag2"))}')
print(f'Children of PKA_lag1: {sorted(dag.successors("PKA_lag1"))}')

# %% [markdown]
# ## 2. Interventional Query: do(Mek_lag2=3, PKA_lag1=2)
#
# **"What would the current values be if Mek two steps ago had been set to 3 and PKA one step ago to 2?"**
#
# The do-operator severs the incoming edges of the intervened nodes (here they are roots, so there are none) and propagates the fixed values to their children: `Mek_lag2` drives `Erk`, and `PKA_lag1` drives `PKA`, `P38`, `Mek`, `Erk` and `Akt`.

# %%
intervention = {'Mek_lag2': 3.0, 'PKA_lag1': 2.0}
current_vars = var_names

intv_samples = cg_wrap.interventional_query(intervention, n_samples=3000)
print(f'Interventional samples: {intv_samples.shape}  (all non-intervened nodes)')

# Summary statistics for the current-time variables, next to their observed means
summary_intv = cg_wrap.query_summary(intv_samples[current_vars])
summary_intv.insert(0, 'observed mean', lagged_df[current_vars].mean())
print('\n=== Interventional distribution: do(Mek_lag2=3, PKA_lag1=2) ===')
print(summary_intv.round(3).to_string())

# %%
# Query a specific target
intv_erk = cg_wrap.interventional_query(intervention, targets='Erk', n_samples=5000)

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(lagged_df['Erk'], bins=40, density=True, alpha=0.4, color='gray',
        label='Observed Erk')
ax.hist(intv_erk['Erk'], bins=50, density=True, alpha=0.6, color='steelblue',
        label='do(Mek_lag2=3, PKA_lag1=2)')
ax.axvline(intv_erk['Erk'].mean(), color='navy', ls='--', lw=2,
           label=f'Interventional mean = {intv_erk["Erk"].mean():.2f}')
ax.axvline(lagged_df['Erk'].mean(), color='red', ls=':', lw=2,
           label=f'Observed mean = {lagged_df["Erk"].mean():.2f}')
ax.set_xlabel('Erk'); ax.set_ylabel('Density')
ax.set_title('Interventional: P(Erk | do(Mek_lag2=3, PKA_lag1=2))')
ax.legend(fontsize=8)
plt.tight_layout(); plt.show()

# %% [markdown]
# Variables with an intervened parent move away from their observed means (`Erk`, `Akt`, `Mek`, `P38`, `PKA`); the others (for example `Raf`, `Jnk`, `Plcg`) keep their observed distribution, because no intervened node is among their causes.

# %% [markdown]
# ## 3. Observational Query: P(others | Mek≈3, PKA≈2)
#
# **"Given that we observe Mek≈3 and PKA≈2 at time t, what is the likely distribution of the other variables?"**
#
# Conditioning does not cut any edges. Observing a high `Mek` makes the lagged values that produce a high `Mek` more likely (its parents are `Mek_lag1`, `PKA_lag1`, `PKC_lag1` and `Raf_lag2`), and those lagged values also drive other current-time variables.
#
# Two conditioning modes:
# - **Soft** (default): Gaussian kernel weighting, so samples near the evidence get higher weight
# - **Hard** (`tolerance=...`): rejection sampling, keeping only samples within the tolerance of the evidence
#
# Both draw samples from the SCM and then weight or filter them, so the number of samples that effectively count can be much smaller than `n_samples`. The effective sample size (ESS) is printed below.

# %%
evidence = {'Mek': 3.0, 'PKA': 2.0}

# Soft conditioning (default) — Gaussian kernel weighting
obs_samples = cg_wrap.observational_query(evidence, n_samples=10000)

print(f'Observational samples: {obs_samples.shape}')
print(f'Has weights: {"_weight" in obs_samples.columns}')
print(f'Effective sample size: {1 / (obs_samples["_weight"]**2).sum():.0f} of 10000')

show = [v for v in current_vars if v not in evidence] + ['Mek_lag1', 'PKA_lag1', 'PKC_lag1', 'Raf_lag2']
summary_obs = cg_wrap.query_summary(obs_samples[show + ['_weight']])
summary_obs.insert(0, 'observed mean', lagged_df[show].mean())
print('\n=== Observational distribution: P(· | Mek≈3, PKA≈2) ===')
print(summary_obs.round(3).to_string())

# %%
# Hard conditioning with tolerance — rejection sampling
obs_hard = cg_wrap.observational_query(
    evidence,
    n_samples=50000,  # more raw samples needed for rejection
    tolerance=0.5,    # accept if within 0.5 std of each evidence value
)

print(f'Hard conditioning: {len(obs_hard)} samples accepted out of 50000')
print(f'Acceptance rate: {len(obs_hard)/50000:.1%}')

summary_hard = cg_wrap.query_summary(obs_hard[show])
print('\n=== Hard conditioning: P(· | |Mek-3|<0.5σ, |PKA-2|<0.5σ) ===')
print(summary_hard.round(3).to_string())

# %% [markdown]
# Soft and hard conditioning agree on the main shifts: given Mek≈3 and PKA≈2, the parents `Mek_lag1`, `PKA_lag1` and `Raf_lag2` are higher than usual and `PKC_lag1` is lower, and the current-time variables driven by those lagged values (`PKC`, `Raf`, `P38`) move down. Smaller shifts, such as those of `Erk` and `Akt`, differ in size between the two estimates. Each estimate rests on a few hundred effective samples (the ESS and acceptance count above), so treat differences of a few tenths as noise.

# %% [markdown]
# ## 4. Comparing Interventional vs Observational
#
# Now ask both questions about the same current-time nodes:
#
# - **Interventional** do(Mek=3, PKA=2): we *force* the values. In the transition DAG, `Mek` and `PKA` have no children, so the other current-time variables keep their observed distribution.
# - **Observational** P(· | Mek≈3, PKA≈2): we *observe* the values. This is informative about their lagged causes, which also drive the other variables.

# %%
targets_to_compare = ['Erk', 'Akt', 'Raf', 'P38']

intv_all = cg_wrap.interventional_query(
    evidence, targets=targets_to_compare, n_samples=5000
)
obs_all = cg_wrap.observational_query(
    evidence, targets=targets_to_compare, n_samples=20000
)
print(f'Observational ESS: {1 / (obs_all["_weight"]**2).sum():.0f} of 20000')

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, var in zip(axes.flat, targets_to_compare):
    ax.hist(intv_all[var], bins=40, density=True, alpha=0.6,
            color='steelblue', label='Interventional: do(Mek=3, PKA=2)')
    ax.hist(obs_all[var], bins=40, density=True, alpha=0.5,
            weights=obs_all['_weight'].values, color='darkorange',
            label='Observational: P(·|Mek≈3, PKA≈2)')
    ax.set_xlabel(var)
    ax.set_ylabel('Density')
    ax.set_title(var)
    ax.legend(fontsize=8)

plt.suptitle('Interventional vs Observational: do(Mek=3, PKA=2) vs P(·|Mek≈3, PKA≈2)',
             fontsize=12, y=1.02)
plt.tight_layout(); plt.show()

# %%
# Side-by-side summary table
sum_intv = cg_wrap.query_summary(intv_all)
sum_obs = cg_wrap.query_summary(obs_all)

comparison = pd.DataFrame({
    'Observed mean': lagged_df[targets_to_compare].mean(),
    'Intv. mean': sum_intv['mean'],
    'Obs. mean': sum_obs['mean'],
    'Obs. - Intv.': sum_obs['mean'] - sum_intv['mean'],
})
print('=== Comparison: Interventional vs Observational ===')
print(comparison.round(3).to_string())

# %% [markdown]
# The interventional means match the observed means up to Monte Carlo noise: forcing `Mek` and `PKA` at time t does not affect other variables at time t. The observational means differ, most clearly for `Raf`, `P38` and `Akt`, because observing high `Mek` and `PKA` points to lower `PKC_lag1` and higher `PKA_lag1`, which are causes of those variables. The effect of `Mek` on `Erk` appears only one or more steps later, which is what the lagged intervention in Section 2 captures.

# %% [markdown]
# ## 5. Seeing vs Doing on a Root Node
#
# For a lagged node the picture is different. `PKC_lag1` has no parents in the model, so conditioning on it cannot tell us anything about its causes: P(Mek | PKC_lag1≈x) and P(Mek | do(PKC_lag1=x)) should coincide. We use a narrow kernel (`kernel_bandwidth=0.1`) so the observational estimate is close to exact conditioning; Section 6 shows what a wider kernel does.

# %%
pkc_values = [-2.0, 0.0, 2.0]

fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharex=True, sharey=True)
rows = []
for pkc_val in pkc_values:
    s_intv = cg_wrap.interventional_query(
        {'PKC_lag1': pkc_val}, targets='Mek', n_samples=3000
    )
    axes[0].hist(s_intv['Mek'], bins=30, density=True, alpha=0.4,
                 label=f'do(PKC_lag1={pkc_val})')

    s_obs = cg_wrap.observational_query(
        {'PKC_lag1': pkc_val}, targets=['Mek'], n_samples=20000,
        kernel_bandwidth=0.1,
    )
    axes[1].hist(s_obs['Mek'], bins=30, density=True, alpha=0.4,
                 weights=s_obs['_weight'], label=f'P(Mek|PKC_lag1≈{pkc_val})')
    rows.append({
        'PKC_lag1': pkc_val,
        'Intv. mean Mek': s_intv['Mek'].mean(),
        'Obs. mean Mek': np.average(s_obs['Mek'], weights=s_obs['_weight']),
        'Obs. ESS': 1 / (s_obs['_weight'] ** 2).sum(),
    })

axes[0].set_title('Interventional: do(PKC_lag1=x)')
axes[0].set_xlabel('Mek'); axes[0].legend()
axes[1].set_title('Observational: P(Mek|PKC_lag1≈x)')
axes[1].set_xlabel('Mek'); axes[1].legend()
plt.tight_layout(); plt.show()

print(pd.DataFrame(rows).round(3).to_string(index=False))

# %% [markdown]
# Both panels shift `Mek` in the same direction, and the interventional and observational means agree to within about 0.1: for a root node, seeing and doing give the same answer. This relies on the model's treatment of lagged nodes as independent roots; in the real process `PKC_lag1` has its own causes further in the past.

# %% [markdown]
# ## 6. Controlling Kernel Bandwidth
#
# The `kernel_bandwidth` parameter (in units of the evidence variable's standard deviation) controls how "soft" the conditioning is:
# - **Small bandwidth** (e.g., 0.1): tight conditioning, close to hard matching, with few effective samples
# - **Large bandwidth** (e.g., 1.5): soft conditioning that also weights samples far from the evidence
# - **Default** (0.5)

# %%
bandwidths = [0.1, 0.3, 0.5, 1.5]

fig, axes = plt.subplots(1, len(bandwidths), figsize=(16, 3.5), sharey=True)
for ax, bw in zip(axes, bandwidths):
    s = cg_wrap.observational_query(
        {'Mek_lag2': 3.0},
        targets=['Erk'],
        n_samples=20000,
        kernel_bandwidth=bw,
    )
    ess = 1 / (s['_weight']**2).sum()
    mean = np.average(s['Erk'], weights=s['_weight'])
    ax.hist(s['Erk'], bins=40, density=True, weights=s['_weight'],
            alpha=0.7, color='darkorange')
    ax.set_title(f'bandwidth={bw}\nESS={ess:.0f}, mean={mean:.2f}')
    ax.set_xlabel('Erk')

axes[0].set_ylabel('Density')
plt.suptitle('Effect of kernel_bandwidth on P(Erk | Mek_lag2≈3)', y=1.02)
plt.tight_layout(); plt.show()

# %% [markdown]
# A smaller bandwidth conditions more sharply but leaves fewer effective samples; a larger one uses more samples but also includes samples whose `Mek_lag2` is far from 3, which pulls the conditional mean of `Erk` back toward its unconditional mean. Because `Mek_lag2` is a root node, the narrowest kernel gives a mean close to the interventional answer do(Mek_lag2=3) computed in Section 7.

# %% [markdown]
# ## 7. Using the Standalone API
#
# The query functions are also available as standalone functions that take a fitted SCM:

# %%
from causalts.effects import interventional_query, observational_query, query_summary
from causalts.effects import fit_scm

scm, dag, lagged_df = fit_scm(disc_graph, df, mechanism_type='linear')

# Interventional
samples = interventional_query(scm, {'Mek_lag2': 3.0}, targets=['Erk', 'Akt'], n_samples=2000)
print('Interventional query do(Mek_lag2=3) (standalone):')
print(query_summary(samples).round(3))

print()

# Observational
samples = observational_query(scm, lagged_df, {'Mek': 3.0}, targets=['Erk', 'Akt'])
print('Observational query P(· | Mek≈3) (standalone):')
print(query_summary(samples).round(3))

# %% [markdown]
# ## Summary
#
# | Query type | Method | What it answers | Edges severed? |
# |---|---|---|---|
# | Interventional | `interventional_query()` | "What if we SET X=x?" | Yes (do-operator) |
# | Observational (soft) | `observational_query()` | "Given we see X≈x, what's Y?" | No |
# | Observational (hard) | `observational_query(tolerance=...)` | "Given X is near x, what's Y?" | No |
#
# In the lag-embedded model, intervene on lagged nodes (`X_lag1`, `X_lag2`, ...) to see effects on current-time variables; current-time nodes have no children. Check the effective sample size of any conditioning query before reading its percentiles.
#
# **Key API:**
# ```python
# # Via WrappedGraph
# cg = wrap_graph(G_hat, df)
# cg.interventional_query({'Mek_lag2': 3.0, 'PKA_lag1': 2.0}, targets='Erk')
# cg.observational_query({'Mek': 3.0, 'PKA': 2.0}, targets='Erk')
# cg.query_summary(samples)
#
# # Standalone
# from causalts.effects import interventional_query, observational_query, query_summary
# interventional_query(scm, intervention, targets, n_samples)
# observational_query(scm, lagged_df, evidence, targets, n_samples, tolerance, kernel_bandwidth)
# query_summary(result)
# ```
