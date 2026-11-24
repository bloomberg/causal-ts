# Copyright 2026 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Extra tests for the named datasets and generators in ``synthetic_datasets``.

Covers the nonstationary examples, ``lorenz96``, the safe nonlinearities,
``SCPGraphGenerator`` helpers and failure paths, and ``topology_scp``.
"""

import numpy as np
import pytest

from causalts.synthetic_data import synthetic_datasets as sd


def _lag_ols(df, target, regressors):
    """OLS coefficients of ``target(t)`` on ``[(col, lag), ...]`` plus intercept."""
    x = df.to_numpy()
    lmax = max(lag for _c, lag in regressors)
    y = x[lmax:, target]
    cols = [x[lmax - lag : len(x) - lag, c] for c, lag in regressors]
    X = np.column_stack(cols + [np.ones_like(y)])
    return np.linalg.lstsq(X, y, rcond=None)[0][:-1]


# ---------------------------------------------------------------------------
# Named datasets
# ---------------------------------------------------------------------------


def test_ex1_nonstationary_bundle_and_trend_strength():
    b = sd.ex1_nonstationary(seed=0, T=200)
    assert b["df"].shape == (200, 5)
    assert b["include_C"] and b["var_names"][-1] == "C"
    gt = b["ground_truth"]
    assert gt.shape == (6, 6, 3) and gt.sum() == 11
    # C (index 5) drives X0, X1, X2 and nothing else.
    assert set(np.flatnonzero(gt[5].sum(axis=1))) == {0, 1, 2}
    np.testing.assert_array_equal(sd.ex1_nonstationary(seed=0, T=200)["df"], b["df"])
    # Same noise, stronger trend: the series end differs by the extra trend.
    strong = sd.ex1_nonstationary(seed=0, T=200, trend_strength=3.0)["df"]
    assert strong["X1"].iloc[-1] > b["df"]["X1"].iloc[-1] + 1.0


def test_ex1_nonstationary_without_trend_recovers_the_lagged_coefficients():
    df = sd.ex1_nonstationary(seed=1, T=4000, trend_strength=0.0)["df"]
    np.testing.assert_allclose(_lag_ols(df, 3, [(3, 1), (2, 2)]), [0.5, 0.7], atol=0.05)
    np.testing.assert_allclose(_lag_ols(df, 4, [(4, 1), (3, 1)]), [0.5, 0.3], atol=0.05)


def test_confounded_trends_has_one_true_cross_link_and_a_shared_trend():
    b = sd.confounded_trends(seed=0, T=400)
    df, gt = b["df"], b["ground_truth"]
    assert list(df.columns) == b["var_names"] and df.shape == (400, 7)
    cross = [(i, j, k) for i, j, k in zip(*np.nonzero(gt)) if i != j]
    assert cross == [(5, 6, 2)]
    # The shared trend correlates the trend-only roots strongly.
    assert df["Chocolate"].corr(df["Airline"]) > 0.8
    assert abs(df["Temp"].corr(df["Chocolate"])) < 0.2
    flat = sd.confounded_trends(seed=0, T=400, trend_strength=0.0)["df"]
    assert abs(flat["Chocolate"].corr(flat["Airline"])) < 0.2


def test_lorenz96_bundle_graph_and_reproducibility():
    b = sd.lorenz96(seed=3, T=60, N=6, burn_in=50)
    assert b["df"].shape == (60, 6) and b["max_lag"] == 1 and b["weights"] is None
    G = b["ground_truth"]
    assert G.shape == (6, 6, 2) and G[:, :, 0].sum() == 0
    # Every variable has exactly four lag-1 parents: i-2, i-1, i, i+1.
    np.testing.assert_array_equal(G[:, :, 1].sum(axis=0), 4)
    assert G[4, 0, 1] and G[5, 0, 1] and G[0, 0, 1] and G[1, 0, 1]
    assert not G[2, 0, 1]
    np.testing.assert_array_equal(
        sd.lorenz96(seed=3, T=60, N=6, burn_in=50)["df"], b["df"]
    )


def test_lorenz96_without_noise_settles_at_the_fixed_point_for_small_forcing():
    # For F below the first bifurcation the attractor is x_i = F.
    b = sd.lorenz96(seed=0, T=20, N=5, F=0.5, obs_noise=0.0, burn_in=400)
    np.testing.assert_allclose(b["df"].to_numpy(), 0.5, atol=1e-3)


def test_lorenz96_leaves_global_numpy_random_state_untouched():
    # lorenz96 draws from its own default_rng, so the caller's global
    # legacy NumPy state must survive the call unchanged.
    np.random.seed(123)
    expected = np.random.rand()
    np.random.seed(123)
    sd.lorenz96(seed=7, T=50, N=6, burn_in=10)
    assert np.random.rand() == expected


def test_henon_bounded_start_raises_when_every_start_escapes():
    # Q far outside the stable range makes every interior start diverge.
    with pytest.raises(RuntimeError, match="diverged from 100 initial states"):
        sd.henon_chain(seed=0, T=50, Q=40.0, M=3, bounded_start=True)


def test_safe_nonlinearities():
    x = np.array([-100.0, -2.0, 0.0, 3.0])
    np.testing.assert_allclose(sd.nl_sq(x), [50.0, 4.0, 0.0, 9.0])
    np.testing.assert_allclose(sd.nl_abs(x), [100.0, 2.0, 0.0, 3.0])
    np.testing.assert_allclose(sd.nl_sin(x), np.sin(x))
    assert sd.nl1(np.array([1000.0]))[0] == pytest.approx(50 + 0.5 * 2500)
    assert set(sd.DEFAULT_FUNC_MAP) >= set(sd.DEFAULT_FUNC_NAMES) | set(
        sd.NL_FUNC_NAMES
    )


# ---------------------------------------------------------------------------
# SCPGraphGenerator
# ---------------------------------------------------------------------------


def test_scp_generator_rejects_unknown_density():
    with pytest.raises(ValueError, match="density must be"):
        sd.SCPGraphGenerator(n_vars=3, density="medium")


def test_scp_generator_dense_keeps_explicit_ranges():
    gen = sd.SCPGraphGenerator(
        n_vars=4,
        density="dense",
        auto_range=(0.2, 0.3),
        dep_pos_range=(0.2, 0.3),
        dep_neg_range=(-0.2, -0.3),
        max_spectral_radius=0.5,
    )
    assert gen.auto_range == (0.2, 0.3) and gen.dep_neg_range == (-0.2, -0.3)
    assert gen.max_spectral_radius == 0.5


def test_scp_generator_sample_is_seeded_and_ground_truth_matches_links():
    gen = sd.SCPGraphGenerator(n_vars=4, max_lag=2)
    a = gen.sample(seed=5, T=120, n_links=3)
    b = gen.sample(seed=5, T=120, n_links=3)
    np.testing.assert_array_equal(a["df"], b["df"])
    gt = a["ground_truth"]
    expected = np.zeros_like(gt)
    for j, ps in a["links"].items():
        for (i, lag), _c, _f in ps:
            expected[i, j, -lag] = 1
    np.testing.assert_array_equal(gt, expected)
    assert sum(a["meta"]["funcs_count"].values()) == sum(
        len(ps) for ps in a["links"].values()
    )
    # Coefficients were rescaled to the target spectral radius.
    assert gen._spectral_radius(a["links"]) <= gen.max_spectral_radius + 1e-9


def test_scp_generator_dense_rescues_explosive_draws():
    # Large coefficients make every raw draw explosive; dense mode rescales
    # instead of rejecting.
    gen = sd.SCPGraphGenerator(
        n_vars=3,
        max_lag=1,
        density="dense",
        auto_range=(1.5, 1.6),
        dep_pos_range=(1.5, 1.6),
        dep_neg_range=(-1.5, -1.6),
        dependency_funcs=("lin",),
    )
    s = gen.sample(seed=0, T=100)
    assert s["meta"]["attempt"] == 0
    assert gen._spectral_radius(s["links"]) < 1.0


def test_scp_generator_raises_after_max_tries():
    # Sparse mode rejects every explosive draw and gives up.
    gen = sd.SCPGraphGenerator(
        n_vars=3, max_lag=1, auto_range=(1.5, 1.6), dependency_funcs=("lin",)
    )
    with pytest.raises(RuntimeError, match="max_tries"):
        gen.sample(seed=0, T=50, max_tries=3)
    # A std-ratio cap no sample can meet also exhausts the tries.
    tight = sd.SCPGraphGenerator(n_vars=3, max_lag=1, max_std_ratio=1e-6)
    with pytest.raises(RuntimeError, match="max_tries"):
        tight.sample(seed=0, T=50, max_tries=3)


def test_scp_generator_static_helpers():
    G = sd.SCPGraphGenerator
    np.testing.assert_allclose(G._arange_inclusive(0.3, 0.1, 0.1), [0.3, 0.2, 0.1])
    with pytest.raises(ValueError, match="step must be > 0"):
        G._arange_inclusive(0.1, 0.3, 0.0)
    assert G._spectral_radius({0: [((1, 0), 0.5, "lin")], 1: []}) == 0.0
    assert G._spectral_radius({0: [((0, -1), 0.5, "lin")]}) == pytest.approx(0.5)
    assert G._num_from_any("X12") == 12 and G._num_from_any("-3") == -3
    assert G._num_from_any(np.int64(4)) == 4 and G._num_from_any(2.0) == 2
    assert G._num_from_any("none") is None and G._num_from_any(None) is None


def test_links_to_gt_skips_unparsable_or_out_of_range_entries():
    gen = sd.SCPGraphGenerator(n_vars=3, max_lag=2)
    links = {
        "X0": [(("X1", "-1"), 0.5, "lin")],  # string ids are parsed
        "bad": [((0, -1), 0.5, "lin")],  # unparsable effect
        7: [((0, -1), 0.5, "lin")],  # effect out of range
        1: [
            ((0, -1), 0.5),  # too short
            (0, 0.5, "lin"),  # pair not a tuple
            (("a", "b"), 0.5, "lin"),  # unparsable pair
            ((0, -5), 0.5, "lin"),  # lag beyond max_lag
            ((2, -2), 0.5, "lin"),
        ],
    }
    gt = gen._links_to_gt(links, 3, 2)
    assert gt[1, 0, 1] == 1 and gt[2, 1, 2] == 1 and gt.sum() == 2
    # Usage counts every 3-tuple entry, parseable or not.
    assert gen._func_usage(links) == {"lin": 7}


# ---------------------------------------------------------------------------
# topology_scp
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["scale_free", "erdos_renyi_dense", "small_world", "small_world_dense"]
)
def test_topology_presets_via_load_dataset(name):
    b = sd.load_dataset(name, seed=1, n_vars=8, T=80)
    gt, meta = b["ground_truth"], b["meta"]
    assert b["df"].shape == (80, 8) and gt.shape == (8, 8, 4)
    n_cross = sum(1 for i, j, _k in zip(*np.nonzero(gt)) if i != j)
    # Each topology edge gets exactly one lag (zero coefficients aside).
    assert n_cross <= meta["n_cross_edges"]
    assert meta["topology"] in name and meta["spectral_radius"] < 1.0
    assert meta["max_degree"] >= meta["mean_degree"]


def test_topology_scp_rejects_unknown_topology():
    with pytest.raises(ValueError, match="Unknown topology"):
        sd.topology_scp("ring", seed=0, n_vars=4, T=50)


def test_topology_scp_raises_when_no_attempt_passes():
    with pytest.raises(RuntimeError, match="Failed to generate stationary"):
        sd.topology_scp(
            "erdos_renyi", seed=0, n_vars=4, T=50, max_std_ratio=1e-6, max_tries=2
        )


# ---------------------------------------------------------------------------
# load_dataset registry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(sd.DATASET_FUNCS))
def test_load_dataset_covers_every_registered_name(name):
    bundle = sd.load_dataset(name, seed=0, T=100)

    assert {"df", "ground_truth", "max_lag"} <= set(bundle)
    assert bundle["df"].shape[0] == 100
    assert bundle["ground_truth"].ndim == 3


def test_load_dataset_unknown_name_lists_the_registered_names():
    with pytest.raises(KeyError) as excinfo:
        sd.load_dataset("not-a-dataset")

    message = str(excinfo.value)
    for name in sd.DATASET_FUNCS:
        assert name in message
