# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# CItest: A Distribution Free Conditional Independence Test
# Based on: Cai, Li, Zhang, "A Distribution Free Conditional Independence
# Test with Applications to Causal Discovery", JMLR 23(152):1-34, 2022.
# Reference implementation: https://github.com/zhanruicai/CItest (MIT license)

import warnings

import numpy as np
import torch

from .cit_test import CIT_Base


def _standardize_columns(z):
    """Centre each column of z and scale it to unit sd (a constant column to 0)."""
    if z.dim() == 1:
        z = z.unsqueeze(1)
    sd = z.std(dim=0, keepdim=True)
    sd = torch.where(sd < 1e-10, torch.ones_like(sd), sd)
    return (z - z.mean(dim=0, keepdim=True)) / sd


def _multivariate_silverman_bandwidth(n, d):
    """Silverman's normal-reference rule for a d-dimensional Gaussian kernel on
    standardized data: h = (4 / ((d + 2) n)) ** (1 / (d + 4)).

    For d = 1 it is the textbook 1-D rule (1.06 n^(-1/5)); the rule it replaced
    multiplied in the 1.06 twice, so |Z| = 1 bandwidths are about 6% narrower
    than before. For larger d it shrinks more slowly with n, as the
    d-dimensional NW smoother needs.
    """
    return (4.0 / ((d + 2) * n)) ** (1.0 / (d + 4))


def _pca_whiten(z, var_threshold=0.999):
    """Project z onto its principal components and standardize.

    Removes near-collinear directions (keeps components explaining
    var_threshold of total variance). Returns whitened z with
    uncorrelated, unit-variance columns.
    """
    if z.dim() == 1 or z.shape[1] == 1:
        return z

    z_centered = z - z.mean(dim=0, keepdim=True)
    cov = z_centered.T @ z_centered / (z.shape[0] - 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)

    # Keep components in descending eigenvalue order
    eigvals = eigvals.flip(0)
    eigvecs = eigvecs.flip(1)

    # Retain dims explaining var_threshold of variance
    total = eigvals.sum()
    cumvar = eigvals.cumsum(0) / total.clamp(min=1e-10)
    n_keep = int((cumvar < var_threshold).sum().item()) + 1
    n_keep = max(1, min(n_keep, z.shape[1]))

    eigvals_k = eigvals[:n_keep].clamp(min=1e-10)
    eigvecs_k = eigvecs[:, :n_keep]

    # Project and scale to unit variance
    z_proj = z_centered @ eigvecs_k
    return z_proj / eigvals_k.sqrt().unsqueeze(0)


def _gaussian_kernel_weights(z, h):
    """Compute Gaussian kernel weight matrix K_h(z_i - z_j) for all pairs.

    Returns (n, n) matrix where W[i, j] = K_h(z_i - z_j).
    """
    if z.dim() == 1:
        z = z.unsqueeze(1)
    sq_dist = torch.cdist(z, z, p=2) ** 2
    d = z.shape[1]
    return torch.exp(-sq_dist / (2 * h**2)) / ((2 * torch.pi) ** (d / 2.0))


# Share of rows falling back to the marginal CDF above which the test warns.
_SPARSE_FALLBACK_FRAC = 0.1


def _estimate_conditional_cdfs(x, z, h, leave_one_out=False, return_n_fallback=False):
    """Estimate F(x_i | z_i) for each observation i using NW kernel smoothing.

    With ``leave_one_out`` the point's own weight is dropped from its sum. Its
    self-weight is the largest in the row and always counts x_i <= x_i, which
    pulls U_i towards 1 by an amount that depends on how isolated z_i is, so U
    stays dependent on Z and the permutation null over-rejects as |Z| grows.
    A point with no neighbour weight left falls back to the marginal CDF of x,
    which ignores Z for that point; ``return_n_fallback`` also returns how many
    did (without ``leave_one_out``: how many have only their own weight, which
    pins U_i to 1, also ignoring Z).

    Returns vector of length n with CDF values.
    """
    if z.dim() == 1:
        z = z.unsqueeze(1)
    K = _gaussian_kernel_weights(z, h)
    indicator = (x.unsqueeze(0) <= x.unsqueeze(1)).float()
    if leave_one_out:
        K.fill_diagonal_(0.0)  # K is freshly built above, so in place is safe
        K_raw = K.sum(dim=1)
        U = (K * indicator).sum(dim=1) / K_raw.clamp(min=1e-30)
        has_weight = K_raw > 1e-30
        U = torch.where(has_weight, U, indicator.mean(dim=1))
        if return_n_fallback:
            return U, int((~has_weight).sum().item())
        return U
    weighted_ind = K * indicator
    K_sum = K.sum(dim=1).clamp(min=1e-10)
    U = weighted_ind.sum(dim=1) / K_sum
    if not return_n_fallback:
        return U
    # Rows whose only weight is their own: U_i is pinned to 1 whatever Z is.
    n_isolated = int(((K.sum(dim=1) - K.diagonal()) <= 1e-30).sum().item())
    return U, n_isolated


def _estimate_marginal_cdf(z):
    """Estimate marginal CDF: W_i = P(Z <= z_i) = mean(Z <= z_i)."""
    if z.dim() == 1:
        return (z.unsqueeze(0) <= z.unsqueeze(1)).float().mean(dim=1)
    else:
        cdfs = []
        for col in range(z.shape[1]):
            cdfs.append(
                (z[:, col].unsqueeze(0) <= z[:, col].unsqueeze(1)).float().mean(dim=1)
            )
        return torch.stack(cdfs, dim=1)


def _build_uu_matrix(u):
    """Build the uu kernel matrix from CDF values.

    uu[i,j] = exp(-|u_i - u_j|) + (exp(-u_i) + exp(u_i-1)) + (exp(-u_j) + exp(u_j-1)) + 2*exp(-1) - 4
    """
    u_dist = torch.cdist(u.unsqueeze(1), u.unsqueeze(1), p=1).squeeze()
    if u_dist.dim() == 0:
        u_dist = u_dist.unsqueeze(0).unsqueeze(0)
    ui = torch.exp(-u) + torch.exp(u - 1)
    uu = torch.exp(-u_dist) + ui.unsqueeze(1) + ui.unsqueeze(0) + 2 * np.exp(-1) - 4
    return uu


def _build_ww_matrix(w):
    """Build the ww kernel matrix from marginal CDF values."""
    if w.dim() == 1:
        w_dist = torch.cdist(w.unsqueeze(1), w.unsqueeze(1), p=1).squeeze()
        if w_dist.dim() == 0:
            w_dist = w_dist.unsqueeze(0).unsqueeze(0)
    else:
        w_dist = torch.zeros(w.shape[0], w.shape[0], device=w.device)
        for col in range(w.shape[1]):
            w_dist += torch.cdist(
                w[:, col].unsqueeze(1), w[:, col].unsqueeze(1), p=1
            ).squeeze()
    return torch.exp(-w_dist)


def _permutation_test(
    uu,
    fixed,
    stat,
    n_perm,
    n,
    device,
    seed=None,
    alpha=0.05,
    early_stop=True,
    block_length=None,
):
    """Batched permutation test with early stopping.

    Permutes uu rows/cols against a fixed matrix (vv*ww or just vv).
    Uses batched tensor operations and stops early when the p-value
    is clearly resolved. Two-level early stopping (following CMIknn):

    1. Running p-value check: if p >> alpha or p-value std is 0, enter
       second-level check.
    2. Null distribution z-score: if |stat - quantile(null, alpha)| / std(null)
       exceeds 5, or if the running p-value median is >3 std from alpha, stop.

    block_length : int or None
        If > 1, use block shuffle (permute contiguous blocks of rows) instead of
        fully random permutation. Preserves short-range temporal autocorrelation
        in the null distribution. Recommended for unconditional tests (Z empty)
        on autocorrelated time series. DFCITGPU's ``sig_blocklength="auto"`` uses
        ``_auto_block_length(n, _min_blocks(min(alpha, 0.01)))`` (sqrt(n),
        shortened to keep enough blocks to reach p < alpha; row shuffling if
        that is under 2 rows) for the n rows a test actually uses.
    """
    gen = torch.Generator(device="cpu")
    rng_block = None
    if seed is not None:
        gen.manual_seed(seed)
    if block_length is not None and block_length > 1:
        rng_block = np.random.default_rng(seed if seed is not None else 0)

    fixed_flat = fixed.reshape(-1)
    n_sq = n * n

    count_ge = 0
    total = 0
    batch_size = min(50, n_perm)
    min_perms = 20
    check_interval = 20

    null_dist = []
    running_pvals = []

    for start in range(0, n_perm, batch_size):
        end = min(start + batch_size, n_perm)
        batch = end - start

        if rng_block is not None:
            bl = block_length
            n_blocks = n // bl
            raw_perms = []
            for _ in range(batch):
                block_order = rng_block.permutation(n_blocks)
                p = np.concatenate(
                    [np.arange(b * bl, (b + 1) * bl) for b in block_order]
                )
                if n > n_blocks * bl:
                    p = np.concatenate([p, np.arange(n_blocks * bl, n)])
                raw_perms.append(torch.tensor(p, dtype=torch.long))
            perms = torch.stack(raw_perms).to(device)
        else:
            perms = torch.stack(
                [torch.randperm(n, generator=gen) for _ in range(batch)]
            ).to(device)
        uu_rows = uu[perms]
        uu_perm = torch.gather(uu_rows, 2, perms.unsqueeze(1).expand(-1, n, -1))
        perm_stats = (uu_perm.reshape(batch, n_sq) * fixed_flat).mean(dim=1)

        count_ge += (perm_stats >= stat).sum().item()
        total += batch
        null_dist.extend(perm_stats.cpu().tolist())
        running_pvals.append(count_ge / total)

        if not early_stop or total < min_perms:
            continue
        if total % check_interval != 0:
            continue

        running_p = count_ge / total

        # Level 1: quick check — is p clearly away from alpha?
        if running_p <= alpha + 0.1 and np.std(running_pvals) > 0:
            continue

        # Level 2: null distribution z-score check
        null_arr = np.array(null_dist)
        null_std = np.std(null_arr)
        if null_std > 0:
            q = np.quantile(null_arr, alpha)
            z = np.abs(stat.item() - q) / null_std
            if z > 5:
                break

        # Level 3: running p-value stability
        pval_std = np.std(running_pvals)
        z_pval = (np.median(running_pvals) - alpha) / max(pval_std, 0.0001)
        if z_pval >= 3:
            break

    return count_ge / total


def _regress_out(target, z):
    """OLS-residualize target against z. Returns residuals."""
    if z.dim() == 1:
        z = z.unsqueeze(1)
    z_with_intercept = torch.cat([z, torch.ones(z.shape[0], 1, device=z.device)], dim=1)
    # beta = (Z'Z)^{-1} Z' target
    beta = torch.linalg.lstsq(z_with_intercept, target.unsqueeze(1)).solution
    resid = target - (z_with_intercept @ beta).squeeze()
    return resid


@torch.no_grad()
def citest_cind(
    x,
    y,
    z,
    h=None,
    n_perm=500,
    seed=None,
    alpha=0.05,
    early_stop=True,
    whiten_z=False,
    residualize=False,
    leave_one_out=True,
    on_sparse=None,
):
    """Distribution-free conditional independence test.

    Tests H0: X _|_ Y | Z using empirical conditional CDFs
    and a permutation-based p-value.

    residualize : bool (default False)
        Regress X and Y on Z first, then run the unconditional CDF test
        on the residuals. Only effective for linear Z effects; destroys
        nonlinear signal so disabled by default.

    whiten_z : bool (default False)
        PCA-whiten Z before NW bandwidth computation (used only when
        residualize=False). Removes collinearity in the conditioning set.
        Disabled by default as it can hurt recall on nonlinear data.

    leave_one_out : bool (default True)
        Leave each point out of its own Nadaraya-Watson sum, with the automatic
        or a numeric ``h``. The self-included estimator biases U towards 1
        where Z is sparse, which over-rejects as |Z| grows.

    on_sparse : callable or None
        Called as ``on_sparse(frac, h)`` when more than 10% of rows have no
        neighbour weight left (it underflows) and fall back to the marginal CDF,
        so Z is ignored for them: a numeric ``h`` far below Z's scale. Rows
        with only one or two effective neighbours are not counted, so the
        automatic rule practically never triggers it, including at large |Z|
        where the test still over-rejects. By default a ``UserWarning`` is
        issued.

    With ``h=None`` each Z column is standardized and ``h`` follows the
    d-dimensional Silverman rule; a numeric ``h`` is used on Z as given.
    """
    device = x.device
    n = x.shape[0]

    if residualize:
        x_res = _regress_out(x, z)
        y_res = _regress_out(y, z)
        return citest_uind(
            x_res, y_res, n_perm=n_perm, seed=seed, alpha=alpha, early_stop=early_stop
        )

    if whiten_z and z.dim() > 1 and z.shape[1] > 1:
        z = _pca_whiten(z)

    # The marginal CDFs W are rank-based, so they are the same on scaled Z.
    W = _estimate_marginal_cdf(z)
    if h is None:
        # Automatic rule: standardize each column and use the d-dimensional
        # Silverman bandwidth. The 1-D rule on the mean sd under-smoothed for
        # |Z| >= 2 (FPR 0.76 at |Z| = 3, T = 200).
        z = _standardize_columns(z)
        h = _multivariate_silverman_bandwidth(n, z.shape[1])

    U, n_fb = _estimate_conditional_cdfs(
        x, z, h, leave_one_out=leave_one_out, return_n_fallback=True
    )
    V = _estimate_conditional_cdfs(y, z, h, leave_one_out=leave_one_out)
    # x and y share the kernel, so U and V fall back on the same rows.
    if n_fb > _SPARSE_FALLBACK_FRAC * n:
        if on_sparse is not None:
            on_sparse(n_fb / n, h)
        else:
            warnings.warn(
                "DFCIT: over 10% of rows have no kernel neighbour and ignore Z; "
                "the bandwidth is too small for Z's scale.",
                UserWarning,
                stacklevel=3,  # past @torch.no_grad()'s wrapper to the caller
            )

    uu = _build_uu_matrix(U)
    vv = _build_uu_matrix(V)
    ww = _build_ww_matrix(W)

    vv_ww = vv * ww
    stat = (uu * vv_ww).mean()

    p_val = _permutation_test(
        uu, vv_ww, stat, n_perm, n, device, seed, alpha, early_stop
    )
    return p_val, stat.cpu().item()


@torch.no_grad()
def citest_uind(
    x, y, n_perm=500, seed=None, alpha=0.05, early_stop=True, block_length=None
):
    """Distribution-free unconditional independence test.

    Tests H0: X _|_ Y using empirical marginal CDFs.

    block_length : int or None
        Block length for block-shuffle null distribution. Recommended for
        autocorrelated time series. If None, uses fully random permutation.
    """
    device = x.device
    n = x.shape[0]

    U = _estimate_marginal_cdf(x)
    V = _estimate_marginal_cdf(y)

    uu = _build_uu_matrix(U)
    vv = _build_uu_matrix(V)

    stat = (uu * vv).mean()

    p_val = _permutation_test(
        uu,
        vv,
        stat,
        n_perm,
        n,
        device,
        seed,
        alpha,
        early_stop,
        block_length=block_length,
    )
    return p_val, stat.cpu().item()


def _min_blocks(alpha):
    """Fewest whole blocks a block length may leave in a test's rows.

    With b blocks the shuffle null has b! orderings, one of them the identity,
    so p cannot fall below about 1/b!: b is the smallest count with
    b! >= 1/alpha (4 at alpha 0.05, 5 at 0.01, 7 at 0.001). A fixed floor of 10
    blocks was tried and dropped: it shortened the blocks below what the
    autocorrelation needs, roughly doubling the false-positive rate on short
    series (two independent AR(0.9) series, T=36: 0.12 with sqrt(n) blocks,
    0.21 with 10). The floor makes p < alpha reachable, not reliable: near it,
    identity draws alone keep some strongly dependent pairs above alpha.
    """
    if not alpha or alpha <= 0:
        alpha = 0.01  # alpha <= 0 only switches early stopping off
    b, orderings = 2, 2
    while orderings * alpha < 1:
        b += 1
        orderings *= b
    return b


def _auto_block_length(n, min_blocks=5):
    """Automatic block length for n rows: sqrt(n) blocks of sqrt(n) rows.

    Below ``min_blocks**2`` rows that leaves too few blocks, so the length is
    shortened to keep ``min_blocks`` of them; rows are shuffled (``None``)
    only when that is under 2 rows.
    """
    bl = min(int(n**0.5), n // min_blocks)
    return bl if bl > 1 else None


class DFCITGPU(CIT_Base):
    """GPU-accelerated Distribution-Free Conditional Independence Test.

    Implements Cai, Li, Zhang (JMLR 2022) using empirical conditional CDFs
    with Nadaraya-Watson kernel smoothing and permutation-based p-values.

    ``bandwidth="silverman"`` standardizes each conditioning column and uses the
    d-dimensional Silverman rule; a number is used as the kernel width on Z as
    given. ``leave_one_out`` (default True) drops each point from its own kernel
    sum with either bandwidth; ``False`` restores the self-included estimator,
    which over-rejects as |Z| grows.
    """

    def __init__(
        self,
        data,
        device=None,
        n_perm=500,
        bandwidth="silverman",
        alpha=0.05,
        early_stop=True,
        whiten_z=False,
        residualize=False,
        seed=42,
        sig_blocklength="auto",
        leave_one_out=True,
        **kwargs,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._data = data
        self.data_torch = torch.tensor(np.asarray(data).copy(), dtype=torch.float32).to(
            self.device
        )
        self.n_perm = n_perm
        self.bandwidth = bandwidth
        self.alpha = alpha
        self.early_stop = early_stop
        self.whiten_z = whiten_z
        self.residualize = residualize
        self.leave_one_out = leave_one_out
        self._warned_sparse = False
        self._last_sparse = False
        self._seed = seed
        self._sig_blocklength_arg = sig_blocklength  # "auto", int or None
        self._warned_short_blocks = False
        self._hash_min_samples = kwargs.get("min_samples", 30)
        self._set_sig_blocklength(data.shape[0])
        super().__init__(data, method="dfcit", seed=seed, **kwargs)
        self.n_actual_tests = 0
        self.n_tests = 0

    def get_data(self):
        return self._data

    def _bw_tag(self):
        """The bandwidth part of ``param_hash``.

        "silverman-mv2" (d-dimensional rule) and "-loo2" (leave-one-out NW sums
        whose sparse results stay off the disk cache): results cached
        under an older estimator or rule must miss.
        """
        tag = "silverman-mv2" if self.bandwidth == "silverman" else self.bandwidth
        return f"{tag}-loo2" if self.leave_one_out else tag

    def _warn_sparse(self, frac, h):
        """Warn once per instance that rows fall back to the marginal CDF."""
        self._last_sparse = True
        if self._warned_sparse:
            return
        self._warned_sparse = True
        warnings.warn(
            f"DFCITGPU: {frac:.0%} of rows in a conditional test have no kernel "
            f"neighbour at bandwidth {h:.3g} and ignore Z, so a "
            f"confounded pair can be rejected. The bandwidth is too small for "
            f"Z's scale (bandwidth={self.bandwidth!r}).",
            UserWarning,
            # _warn_sparse <- citest_cind <- no_grad wrapper <- __call__ <- caller
            stacklevel=5,
        )

    def set_data(self, d):
        self._data = d
        self._has_nan = bool(np.isnan(d).any())
        self.data_torch = torch.tensor(np.asarray(d).copy(), dtype=torch.float32).to(
            self.device
        )
        # The CLI builds the test on a placeholder and assigns the data later,
        # so the automatic block length must follow the data actually tested.
        self._set_sig_blocklength(d.shape[0])

    def _min_blocks(self):
        """Block floor for this test, sized for the stricter of ``alpha`` and
        0.01 (at least 5 blocks). The CLI and ``discover_df`` pass the run's
        threshold as ``alpha``; the 0.01 clamp covers a test built directly
        for a ``run_*`` call, whose discovery alpha never reaches it."""
        return _min_blocks(min(self.alpha, 0.01) if self.alpha else 0.01)

    def _set_sig_blocklength(self, T):
        """Resolve the block length for T rows and refresh ``param_hash``."""
        if self._sig_blocklength_arg == "auto":
            # Exact when no values are missing; with missing values each
            # unconditional test sizes its own blocks from its complete rows.
            self.sig_blocklength = _auto_block_length(T, self._min_blocks())
            # Auto sizes blocks per test from the rows it uses, so the hash
            # records the rule rather than the full-T length (which an explicit
            # block length, or an older auto entry, could share).
            bl = "auto-n-sqrt-minalpha"  # rule changed: older entries miss
        else:
            self.sig_blocklength = self._sig_blocklength_arg
            bl = self.sig_blocklength
            if bl is not None and bl > 1:
                # Such a length is shortened when it leaves fewer than
                # _min_blocks(alpha) blocks, so entries cached before that rule
                # miss (alpha is in the key too).
                bl = f"{bl}-minalpha"
        self.param_hash = (
            f"T{T}-nperm{self.n_perm}-bw_{self._bw_tag()}"
            f"-whiten{self.whiten_z}-resid{self.residualize}-bl{bl}"
            f"-es{self.early_stop}-a{self.alpha}-seed{self._seed}"
            f"-ms{self._hash_min_samples}"
        )

    data = property(get_data, set_data)

    def __call__(self, X, Y, condition_set=None):
        self.n_tests += 1
        self.n_actual_tests += 1

        cache_key = self._get_array_hash(X, Y, condition_set)
        cache_key = f"{self.method}-{cache_key}"
        combined_hash = cache_key, self.param_hash

        if combined_hash in self.pvalue_cache:
            p, stat = self.pvalue_cache[combined_hash]
            if combined_hash not in self.pvalue_local:
                self.pvalue_local[combined_hash] = (p, stat)
            else:
                self.n_actual_tests -= 1
            return p, stat

        self._last_sparse = False  # set by _warn_sparse during this test
        data_t = self.data_torch
        if self._has_nan:
            mask = self._pairwise_complete(X, Y, condition_set)
            if mask.sum() < self._min_samples:
                p, stat = 1.0, 0.0
                self.pvalue_cache[combined_hash] = (p, stat)
                return p, stat
            data_t = self.data_torch[torch.from_numpy(mask).to(self.device)]

        x = data_t[:, X]
        y = data_t[:, Y]

        if condition_set is None or len(condition_set) == 0:
            block_length = self.sig_blocklength
            n_rows = x.shape[0]
            min_blocks = self._min_blocks()
            if self._sig_blocklength_arg == "auto":
                # Size the blocks for the rows actually tested: with missing
                # values that can be far fewer than T, and a full-T block
                # length could leave no whole block to shuffle.
                block_length = _auto_block_length(n_rows, min_blocks)
            elif (
                block_length is not None
                and block_length > 1
                and n_rows // block_length < min_blocks
            ):
                # Too few blocks for a usable shuffle null (see _min_blocks):
                # keep as much of the requested length as the floor allows.
                if not self._warned_short_blocks:  # once, not once per test
                    self._warned_short_blocks = True
                    warnings.warn(
                        f"DFCITGPU: sig_blocklength={block_length} leaves fewer "
                        f"than {min_blocks} blocks in some tests' complete rows "
                        f"(first: {n_rows}); those tests use shorter blocks.",
                        UserWarning,
                        stacklevel=2,
                    )
                shorter = n_rows // min_blocks
                block_length = shorter if shorter > 1 else None
            p, stat = citest_uind(
                x,
                y,
                n_perm=self.n_perm,
                seed=self._seed,
                alpha=self.alpha,
                early_stop=self.early_stop,
                block_length=block_length,
            )
        else:
            z = data_t[:, condition_set]
            h = None
            if self.bandwidth != "silverman":
                h = float(self.bandwidth)
            p, stat = citest_cind(
                x,
                y,
                z,
                h=h,
                n_perm=self.n_perm,
                seed=self._seed,
                alpha=self.alpha,
                early_stop=self.early_stop,
                whiten_z=self.whiten_z,
                residualize=self.residualize,
                leave_one_out=self.leave_one_out,
                on_sparse=self._warn_sparse,
            )

        self.pvalue_cache[combined_hash] = (p, stat)
        self.pvalue_local[combined_hash] = (p, stat)
        if not self._last_sparse:
            # A result that ignored Z for many rows is kept off the disk cache,
            # so a later run recomputes it and warns again.
            self.save_incremental_cache(cache_key, self.param_hash, p, stat)
        return p, stat
