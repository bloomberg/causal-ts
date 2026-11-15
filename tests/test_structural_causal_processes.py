# Copyright 2026 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the tigramite-derived simulators in ``structural_causal_processes``.

Covers ``var_process`` and its input checks, ``structural_causal_process``
(recursion, noise arrays, interventions, validation), the link/graph
converters and ``generate_structural_causal_process`` /
``check_stationarity``.
"""

import numpy as np
import pytest

from causalts.synthetic_data import structural_causal_processes as scp

FUNC_MAP = {"lin": lambda x: x, "sq": lambda x: x**2}


# ---------------------------------------------------------------------------
# var_process and its helpers
# ---------------------------------------------------------------------------


def test_var_process_no_noise_follows_the_recursion_exactly():
    links = {0: [((0, -1), 0.5)], 1: [((0, -1), 0.8), ((1, -2), 0.2)]}
    init = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]])
    data, parents = scp.var_process(links, T=8, use="no_noise", initial_values=init)
    assert data.shape == (8, 2)
    np.testing.assert_allclose(data[:3], init.T)
    for t in range(3, 8):
        assert data[t, 0] == pytest.approx(0.5 * data[t - 1, 0])
        assert data[t, 1] == pytest.approx(0.8 * data[t - 1, 0] + 0.2 * data[t - 2, 1])
    assert dict(parents) == {0: [(0, -1)], 1: [(0, -1), (1, -2)]}


def test_var_process_drops_zero_coefficients_from_true_parents():
    links = {0: [((0, -1), 0.5), ((1, -1), 0.0)], 1: []}
    _, parents = scp.var_process(links, T=20, use="no_noise")
    assert dict(parents) == {0: [(0, -1)]}


@pytest.mark.parametrize("use", ["inno_cov", "inv_inno_cov"])
def test_var_process_lag0_coefficient_correlates_the_innovations(use):
    # A lag-0 coefficient enters the innovation covariance (or its negated
    # inverse); both encode a correlation of +0.3 between the two noises.
    links = {0: [((1, 0), 0.3)], 1: [((0, 0), 0.3)]}
    np.random.seed(0)
    data, _ = scp.var_process(links, T=20000, use=use)
    r = np.corrcoef(data.T)[0, 1]
    assert r == pytest.approx(0.3, abs=0.03)


def test_var_process_other_use_gives_uncorrelated_noise():
    # An asymmetric lag-0 coefficient would be rejected if the covariance
    # were used; any other ``use`` ignores it.
    links = {0: [((1, 0), 0.3)], 1: [((0, 0), 0.9)]}
    np.random.seed(1)
    data, _ = scp.var_process(links, T=20000, use="white")
    assert abs(np.corrcoef(data.T)[0, 1]) < 0.03


def test_var_process_verbosity_prints_the_covariance(capsys):
    links = {0: [((0, -1), 0.4)], 1: [((1, -1), 0.4)]}
    for use, word in [
        ("inno_cov", "Innovation Cov"),
        ("inv_inno_cov", "Inverse Innovation Cov"),
        ("no_noise", "No noise added"),
    ]:
        scp.var_process(links, T=10, use=use, verbosity=1)
        out = capsys.readouterr().out
        assert word in out
        assert ("Inverse" in out) == (use == "inv_inno_cov")


def test_var_process_is_reproducible_under_the_global_seed():
    links = {0: [((0, -1), 0.5)], 1: [((0, -1), 0.4), ((1, 0), 0.0)]}
    np.random.seed(7)
    a, _ = scp.var_process(links, T=50, use="inno_cov")
    np.random.seed(7)
    b, _ = scp.var_process(links, T=50, use="inno_cov")
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(
    "links, match",
    [
        ({0: [((0, 1), 0.5)]}, "must be <= 0"),
        ({0: [((0, -1), 0.5)], 2: []}, "contiguous"),
        ({0: [((1, -1), 0.5)]}, "Missing IDs 1"),
        ({0: [((1, 0), 0.3)], 1: [((0, 0), 0.5)]}, "not symmetric"),
    ],
)
def test_var_process_rejects_malformed_links(links, match):
    with pytest.raises(ValueError, match=match):
        scp.var_process(links, T=20, use="inno_cov")


def test_var_process_rejects_a_nonstationary_graph():
    with pytest.raises(AssertionError, match="non-stationary"):
        scp.var_process({0: [((0, -1), 1.2)]}, T=20)


def test_var_process_rejects_wrong_initial_values():
    links = {0: [((0, -1), 0.5)], 1: []}
    with pytest.raises(AssertionError, match="numpy.ndarray"):
        scp.var_process(links, T=20, initial_values=[[0.0, 0.0], [0.0, 0.0]])
    with pytest.raises(AssertionError, match="desired shape"):
        scp.var_process(links, T=20, initial_values=np.zeros((2, 5)))


def test_check_stability_uses_the_sparse_path_for_large_systems():
    # n_nodes * period > 25 switches to sparse eigs.
    stable = np.zeros((10, 10, 3))
    stable[np.arange(10), np.arange(10), 0] = 0.5
    scp._check_stability(stable)
    unstable = stable.copy()
    unstable[0, 0, 0] = 1.5
    with pytest.raises(AssertionError):
        scp._check_stability(unstable)


def test_generate_noise_inverse_matches_the_inverted_covariance():
    cov = np.array([[1.0, 0.4], [0.4, 1.0]])
    np.random.seed(3)
    x = scp._generate_noise(cov, time=40000, use_inverse=True)
    expected = np.linalg.inv(np.array([[1.0, -0.4], [-0.4, 1.0]]))
    np.testing.assert_allclose(np.cov(x.T), expected, atol=0.05)


def test_lag_connect_and_covariance_matrices():
    links = {0: [((0, -1), 0.5), ((1, 0), 0.2)], 1: [((0, -2), -0.3), ((0, 0), 0.2)]}
    cm = scp._get_lag_connect_matrix(links)
    assert cm.shape == (2, 2, 3)
    assert cm[0, 0, 0] == 0.5 and cm[1, 0, 1] == -0.3
    assert np.count_nonzero(cm) == 2
    cov = scp._get_covariance_matrix(links)
    np.testing.assert_array_equal(cov, [[1.0, 0.2], [0.2, 1.0]])
    assert scp._find_max_time_lag_and_node_id(links) == (2, 1)


# ---------------------------------------------------------------------------
# _Graph
# ---------------------------------------------------------------------------


def test_graph_helper_detects_cycles_and_sorts_topologically():
    g = scp._Graph(4)
    for u, v in [(2, 0), (0, 1), (2, 3), (3, 1)]:
        g.addEdge(u, v)
    assert not g.isCyclic()
    order = g.topologicalSort()
    pos = {v: k for k, v in enumerate(order)}
    assert sorted(order) == [0, 1, 2, 3]
    assert pos[2] < pos[0] < pos[1] and pos[3] < pos[1]
    g.addEdge(1, 2)
    assert g.isCyclic()


# ---------------------------------------------------------------------------
# structural_causal_process
# ---------------------------------------------------------------------------

LINKS = {
    0: [((0, -1), 0.5, "lin")],
    1: [((1, -1), 0.3, "lin"), ((0, 0), 0.8, "lin")],
    2: [((1, -2), -0.6, "sq")],
}


def _manual_scp(links, noise, transient):
    x = noise.astype("float32").copy()
    for t in range(2, x.shape[0]):
        for j in (0, 1, 2):  # contemporaneous order 0 -> 1
            for (i, lag), c, f in links[j]:
                x[t, j] += c * FUNC_MAP[f](x[t + lag, i])
    return x[transient:]


def test_scp_with_noise_array_reproduces_the_recursion():
    T, frac = 40, 0.25
    transient = int(np.floor(frac * T))
    noise = np.random.default_rng(0).standard_normal((T + transient, 3))
    data, nonvalid = scp.structural_causal_process(
        LINKS, T, noises=noise, func_map=FUNC_MAP, transient_fraction=frac
    )
    assert data.shape == (T, 3) and data.dtype == np.float32
    assert not nonvalid
    np.testing.assert_allclose(data, _manual_scp(LINKS, noise, transient), rtol=1e-5)


def test_scp_seed_makes_output_reproducible():
    a, _ = scp.structural_causal_process(LINKS, 60, func_map=FUNC_MAP, seed=5)
    b, _ = scp.structural_causal_process(LINKS, 60, func_map=FUNC_MAP, seed=5)
    c, _ = scp.structural_causal_process(LINKS, 60, func_map=FUNC_MAP, seed=6)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


@pytest.mark.filterwarnings("ignore:overflow:RuntimeWarning")
def test_scp_flags_nonfinite_output():
    links = {0: [((0, -1), 3.0, "sq")]}
    noises = [lambda n: np.full(n, 2.0)]
    data, nonvalid = scp.structural_causal_process(
        links, 60, noises=noises, func_map=FUNC_MAP
    )
    assert nonvalid
    assert not np.all(np.isfinite(data))


def test_scp_hard_intervention_sets_values_and_cuts_parents():
    T = 30
    iv = np.full(T, np.nan)
    iv[10:] = 5.0
    data, _ = scp.structural_causal_process(
        LINKS, T, func_map=FUNC_MAP, seed=0, intervention={1: iv}
    )
    np.testing.assert_array_equal(data[10:, 1], 5.0)
    # Un-intervened samples (NaN) follow the mechanism, so they are not 5.
    assert not np.any(data[:10, 1] == 5.0)


def test_scp_soft_intervention_adds_to_the_mechanism():
    T = 30
    base, _ = scp.structural_causal_process(LINKS, T, func_map=FUNC_MAP, seed=0)
    shift = np.zeros(T)
    shift[-1] = 2.0
    soft, _ = scp.structural_causal_process(
        LINKS,
        T,
        func_map=FUNC_MAP,
        seed=0,
        intervention={0: shift},
        intervention_type="soft",
    )
    np.testing.assert_allclose(soft[:-1], base[:-1])
    assert soft[-1, 0] == pytest.approx(base[-1, 0] + 2.0)
    # Variable 1 has 0 as a contemporaneous parent with coefficient 0.8.
    assert soft[-1, 1] == pytest.approx(base[-1, 1] + 1.6, rel=1e-5)


def test_scp_intervention_type_dict_and_none():
    T = 20
    iv = np.ones(T)
    a, _ = scp.structural_causal_process(
        LINKS,
        T,
        func_map=FUNC_MAP,
        seed=0,
        intervention={2: iv},
        intervention_type={2: "hard"},
    )
    b, _ = scp.structural_causal_process(
        LINKS,
        T,
        func_map=FUNC_MAP,
        seed=0,
        intervention={2: iv},
        intervention_type=None,
    )
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a[:, 2], 1.0)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(func_map=None), "function map"),
        (dict(links={0: [], 2: []}), "links keys must match N"),
        (dict(noises=np.zeros((5, 3))), "noises.shape"),
        (dict(noises=[np.random.randn]), "noises keys must match N"),
        (dict(links={0: [((3, -1), 0.5, "lin")]}), "var must be in"),
        (dict(links={0: [((0, -1), 1, "lin")]}), "coeff must be float"),
        (dict(links={0: [((0, 1), 0.5, "lin")]}), "non-positive int"),
        (
            dict(links={0: [((1, 0), 0.5, "lin")], 1: [((0, 0), 0.5, "lin")]}),
            "must not contain cycle",
        ),
        (dict(intervention={0: np.ones(3)}), "must be of length T"),
        (
            dict(intervention={0: np.ones(20)}, intervention_type={1: "hard"}),
            "must contain entry for 0",
        ),
    ],
)
def test_scp_rejects_invalid_inputs(kwargs, match):
    args = dict(links=LINKS, T=20, func_map=FUNC_MAP)
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        scp.structural_causal_process(**args)


# ---------------------------------------------------------------------------
# links <-> graph converters
# ---------------------------------------------------------------------------


def test_links_to_graph_marks_lagged_and_contemporaneous_links():
    links = {
        0: [((0, -1), 0.5, "lin"), ((1, -2), 0.0, "lin")],
        1: [((0, 0), 0.8, "lin")],
    }
    g = scp.links_to_graph(links)
    # The zero-coefficient lag-2 link is excluded from the max lag.
    assert g.shape == (2, 2, 2)
    assert g[0, 0, 1] == "-->"
    assert g[0, 1, 0] == "-->" and g[1, 0, 0] == "<--"
    assert np.count_nonzero(g != "") == 3
    assert scp.links_to_graph(links, tau_max=3).shape == (2, 2, 4)


def test_links_to_graph_accepts_parent_only_format():
    g = scp.links_to_graph({0: [(1, 0)], 1: [(1, -2)]})
    assert g.shape == (2, 2, 3)
    assert g[1, 0, 0] == "-->" and g[0, 1, 0] == "<--" and g[1, 1, 2] == "-->"
    assert scp._get_minmax_lag({0: [(1, 0)], 1: [(1, -2)]}) == (0, 2)


def test_links_to_graph_rejects_a_too_small_tau_max():
    with pytest.raises(ValueError, match="tau_max is smaller"):
        scp.links_to_graph({0: [(0, -3)]}, tau_max=1)


def test_dag_to_links_round_trips_and_rejects_non_dags():
    links = {0: [(0, -1)], 1: [(0, 0), (1, -2)], 2: [(1, -1)]}
    parents = scp.dag_to_links(scp.links_to_graph(links))
    assert {j: sorted((int(i), int(t)) for i, t in p) for j, p in parents.items()} == {
        0: [(0, -1)],
        1: [(0, 0), (1, -2)],
        2: [(1, -1)],
    }
    bad = scp.links_to_graph(links)
    bad[0, 1, 0] = bad[1, 0, 0] = "o-o"
    with pytest.raises(ValueError, match="must be DAG"):
        scp.dag_to_links(bad)


# ---------------------------------------------------------------------------
# generate_structural_causal_process / check_stationarity
# ---------------------------------------------------------------------------


def _cross_links(links):
    return [(i, j, lag) for j, ps in links.items() for (i, lag), _c, _f in ps if i != j]


def test_generate_scp_is_seeded_and_has_the_requested_links():
    kw = dict(N=4, L=3, max_lag=2, seed=11, noise_seed=11)
    links_a, noises_a = scp.generate_structural_causal_process(**kw)
    links_b, noises_b = scp.generate_structural_causal_process(**kw)
    assert links_a == links_b
    np.testing.assert_array_equal(noises_a[0](5), noises_b[0](5))
    cross = _cross_links(links_a)
    assert len(cross) == 3
    assert all(-2 <= lag <= -1 for _i, _j, lag in cross)
    # Every variable gets a lag-1 autodependency from auto_coeffs.
    for j in range(4):
        (auto,) = [c for (i, lag), c, _f in links_a[j] if i == j]
        assert auto in (0.5, 0.7)
    assert len(noises_a) == 4


def test_generate_scp_contemporaneous_links_are_acyclic():
    links, _ = scp.generate_structural_causal_process(
        N=4, L=4, contemp_fraction=1.0, max_lag=1, seed=2
    )
    cross = _cross_links(links)
    assert len(cross) == 4 and all(lag == 0 for _i, _j, lag in cross)
    g = scp._Graph(4)
    for i, j, _lag in cross:
        g.addEdge(i, j)
    assert not g.isCyclic()
    data, nonvalid = scp.structural_causal_process(
        links, 50, func_map={"lin": FUNC_MAP["lin"]}, seed=0
    )
    assert data.shape == (50, 4) and not nonvalid


def test_generate_scp_max_lag_zero_is_purely_contemporaneous():
    links, _ = scp.generate_structural_causal_process(N=3, L=2, max_lag=0, seed=0)
    # No autodependencies without a lag, and every link is at lag 0.
    assert all(lag == 0 for ps in links.values() for (_i, lag), _c, _f in ps)
    assert len(_cross_links(links)) == 2


def test_generate_scp_skips_zero_coefficients():
    links, _ = scp.generate_structural_causal_process(
        N=3, L=4, dependency_coeffs=[0.0], auto_coeffs=[0.0], seed=0
    )
    assert all(ps == [] for ps in links.values())


@pytest.mark.parametrize("dist", ["gaussian", "weibull", "uniform"])
def test_generate_scp_builtin_noises_have_the_requested_moments(dist):
    _, noises = scp.generate_structural_causal_process(
        N=2,
        noise_dists=[dist],
        noise_means=[1.5],
        noise_sigmas=[2.0],
        seed=0,
        noise_seed=0,
    )
    x = noises[0](40000)
    assert x.mean() == pytest.approx(1.5, abs=0.05)
    assert x.std() == pytest.approx(2.0, rel=0.03)


def test_generate_scp_passes_custom_noise_callables_through():
    def beta(T):
        return np.zeros(T)

    _, noises = scp.generate_structural_causal_process(N=3, noise_dists=[beta])
    assert all(n is beta for n in noises)


def test_check_stationarity_matches_the_companion_spectral_radius():
    stable = {0: [((0, -1), 0.5, "lin")], 1: [((0, -2), 0.9, "lin")]}
    assert scp.check_stationarity(stable)
    unstable = {0: [((0, -1), 0.6, "lin"), ((0, -2), 0.6, "lin")], 1: []}
    assert not scp.check_stationarity(unstable)
    # Lag-0 links do not enter the companion matrix.
    with_contemp = {0: [((0, -1), 0.5, "lin")], 1: [((0, 0), 5.0, "lin")]}
    assert scp.check_stationarity(with_contemp)
