# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for synthetic data generation."""

import numpy as np
import pandas as pd

from causalts.synthetic_data.synthetic_datasets import (
    SCPGraphGenerator,
    ex1_nl,
    ex2,
    henon_chain,
    load_dataset,
)


def test_load_dataset_ex1():
    r = load_dataset("ex1", seed=42, T=60)
    assert "df" in r
    assert "ground_truth" in r
    assert isinstance(r["df"], pd.DataFrame)
    assert r["df"].shape[0] == 60
    assert not r["df"].isna().any().any()


def test_load_dataset_ex2():
    r = load_dataset("ex2", seed=42, T=60)
    assert r["df"].shape == (60, 6)
    assert r["ground_truth"].ndim == 3


def test_load_dataset_ex3():
    r = load_dataset("ex3", seed=42, T=60)
    assert r["df"].shape == (60, 11)


def test_load_dataset_henon():
    r = load_dataset("henon", seed=42, T=60)
    assert "df" in r
    assert r["df"].shape[0] == 60


def test_ex1_nl_returns_expected_shape():
    r = ex1_nl(seed=42, T=50)
    assert r["df"].shape[1] == 5
    assert r["ground_truth"].shape[0] == 5
    assert r["ground_truth"].shape[1] == 5


def test_ex2_ground_truth_shape():
    r = ex2(seed=42, T=50)
    K = r["df"].shape[1]
    gt = r["ground_truth"]
    assert gt.shape[0] == K
    assert gt.shape[1] == K


def test_henon_chain_params():
    r = henon_chain(seed=0, T=50, Q=0.3, M=3)
    assert r["df"].shape[1] == 3


def test_scp_generator():
    gen = SCPGraphGenerator(n_vars=4, max_lag=2, density="sparse")
    r = gen.sample(seed=42, T=60)
    assert "df" in r
    assert "ground_truth" in r
    assert r["df"].shape == (60, 4)
    gt = r["ground_truth"]
    assert gt.shape == (4, 4, 3)
    assert set(np.unique(gt)).issubset({0, 1})


def test_scp_generator_dense():
    gen = SCPGraphGenerator(n_vars=3, max_lag=1, density="dense")
    r = gen.sample(seed=7, T=50)
    assert r["df"].shape == (50, 3)
    assert not r["df"].isna().any().any()


def _henon_reference(seed, T=200, Q=0.5, M=5):
    """henon_chain as released (standard-normal start), pinned."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((T + 2, M))
    with np.errstate(over="ignore", invalid="ignore"):
        for t in range(2, T + 2):
            for m in range(M):
                if m in (0, M - 1):
                    x[t, m] = 1.4 - x[t - 1, m] ** 2 + 0.3 * x[t - 2, m]
                else:
                    mix = (
                        0.5
                        * Q
                        * (x[t - 1, m - 1] + x[t - 1, m + 1] + (1 - Q) * x[t - 1, m])
                    )
                    x[t, m] = 1.4 - mix**2 + 0.3 * x[t - 2, m]
    return x[2:]


def test_henon_chain_default_is_unchanged():
    """The default keeps the old start, byte-identical (seed 3 diverges)."""
    for seed in (0, 1, 3, 42):
        with np.errstate(over="ignore", invalid="ignore"):
            got = henon_chain(seed=seed)["df"].to_numpy()
        assert got.tobytes() == _henon_reference(seed).tobytes(), seed
    with np.errstate(over="ignore", invalid="ignore"):
        got = load_dataset("henon", seed=3)["df"].to_numpy()
    assert got.tobytes() == _henon_reference(3).tobytes()


def test_henon_chain_bounded_start_stays_finite_for_every_seed():
    """Opt-in: a standard-normal start ran to -inf for ~20% of seeds."""
    for seed in range(50):
        df = henon_chain(seed=seed, bounded_start=True)["df"]
        assert np.isfinite(df.to_numpy()).all(), seed


def _cross_coeffs(monkeypatch):
    """Spy on structural_causal_process; collect (auto, cross) coefficients."""
    import causalts.synthetic_data.structural_causal_processes as scp

    real = scp.structural_causal_process
    auto, cross = [], []

    def spy(links, *args, **kwargs):
        for j, ls in links.items():
            for (i, _), c, _ in ls:
                (auto if i == j else cross).append(c)
        return real(links, *args, **kwargs)

    monkeypatch.setattr(scp, "structural_causal_process", spy)
    return auto, cross


# Values from the released generator: erdos_renyi / scale_free with
# seed=1, n_vars=6, T=60; rows 0, 30 and 59 of the data, and the GT edge list.
_TOPOLOGY_REFERENCE = {
    "erdos_renyi": (
        [
            [-0.9229633808135986, -0.21785151958465576, -0.057434916496276855,
             5.017297267913818, 0.23110291361808777, -2.2121291160583496],
            [-1.3403385877609253, -1.1652345657348633, -0.3358260691165924,
             -0.24982832372188568, -0.2053907960653305, -0.36819130182266235],
            [1.0075894594192505, 0.9310701489448547, -0.14858554303646088,
             -0.6574779152870178, -0.35847264528274536, 3.0213308334350586],
        ],
        [[0, 0, 1], [0, 5, 2], [1, 1, 1], [2, 2, 1], [3, 3, 1], [4, 1, 1],
         [4, 4, 1], [5, 5, 1]],
    ),
    "scale_free": (
        [
            [-1.215832233428955, -1.5702061653137207, -2.1602659225463867,
             3.4594674110412598, 0.25110334157943726, -1.1954801082611084],
            [-1.765641450881958, 0.09866201877593994, 0.22235572338104248,
             -0.42084968090057373, 1.9816102981567383, 3.2148241996765137],
            [1.3273078203201294, 8.624650955200195, 3.203801155090332,
             7.506296157836914, 3.465963840484619, 4.393510818481445],
        ],
        [[0, 0, 1], [0, 1, 1], [0, 2, 1], [0, 4, 2], [0, 5, 3], [1, 1, 1],
         [1, 3, 1], [2, 2, 1], [2, 3, 1], [2, 5, 3], [3, 3, 1], [3, 4, 2],
         [4, 4, 1], [5, 5, 1]],
    ),
}  # fmt: skip


def test_topology_scp_default_is_unchanged():
    """coef_pools='legacy' (default) reproduces the released data."""
    from causalts.synthetic_data import synthetic_datasets as sd

    for name, (rows, edges) in _TOPOLOGY_REFERENCE.items():
        r = getattr(sd, name)(seed=1, n_vars=6, T=60)
        # The data is float32. Equal on the platform the values were taken
        # from; the tolerance only absorbs last-ulp libm differences elsewhere.
        # Any change to the coefficient pools moves these values by far more.
        np.testing.assert_allclose(
            r["df"].to_numpy()[[0, 30, 59]], rows, rtol=1e-6, atol=1e-7
        )
        assert np.argwhere(r["ground_truth"]).tolist() == edges
        legacy = getattr(sd, name)(seed=1, n_vars=6, T=60, coef_pools="legacy")
        assert legacy["df"].to_numpy().tobytes() == r["df"].to_numpy().tobytes()


def test_topology_scp_default_cross_edges_stay_positive(monkeypatch):
    """The legacy pools keep the old (empty) negative pool: documented quirk."""
    from causalts.synthetic_data.synthetic_datasets import topology_scp

    _, cross = _cross_coeffs(monkeypatch)
    for seed in range(3):
        topology_scp("erdos_renyi", seed=seed, n_vars=10, T=200)
    assert cross and min(cross) > 0


def test_topology_scp_fixed_pools_draw_negative_coefficients(monkeypatch):
    """Opt-in: the downward negative range made np.arange empty."""
    from causalts.synthetic_data.synthetic_datasets import topology_scp

    _, cross = _cross_coeffs(monkeypatch)
    for seed in range(3):
        topology_scp("erdos_renyi", seed=seed, n_vars=10, T=200, coef_pools="fixed")
    assert min(cross) < 0 < max(cross)


def test_topology_scp_fixed_pools_accept_descending_ranges(monkeypatch):
    """A descending auto / positive range gave an empty pool, like the negative one."""
    from causalts.synthetic_data.synthetic_datasets import scale_free

    auto, cross = _cross_coeffs(monkeypatch)
    r = scale_free(
        seed=0,
        n_vars=8,
        T=200,
        auto_range=(0.79, 0.30),
        dep_pos_range=(0.89, 0.40),
        coef_pools="fixed",
    )
    assert r["df"].shape == (200, 8)
    # Coefficients may be rescaled for stability, never enlarged or sign-flipped.
    assert auto and max(auto) <= 0.79 and min(auto) > 0
    assert any(c > 0 for c in cross) and max(cross) <= 0.89


def test_topology_scp_rejects_unknown_coef_pools():
    import pytest

    from causalts.synthetic_data.synthetic_datasets import topology_scp

    with pytest.raises(ValueError, match="coef_pools"):
        topology_scp("erdos_renyi", seed=0, n_vars=5, T=50, coef_pools="new")
