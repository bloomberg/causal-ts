# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the residual-score CI tests: GCM, WGCM, PCM.

The interface contract is checked for every test; statistical behaviour is
checked only where it is unambiguous (a strong linear alternative at a
comfortable sample size). Calibration and power at small ``T`` need a dedicated
simulation study, not a unit test -- a pytest run cannot afford the replicate
count those questions need.
"""

import numpy as np
import pytest

from causalts.ci_tests import create_ci_test, list_ci_tests

SHIPPED = ["gcm", "wgcm", "pcm"]

FAST_KWARGS = {
    "wgcm": {"n_boot": 99},
}


def _linear_ci_data(n=250, seed=0):
    """X and Y share a linear parent Z: conditionally independent."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 2))
    base = z.sum(axis=1)
    return np.column_stack(
        [base + rng.standard_normal(n), base + rng.standard_normal(n), z]
    )


def _linear_dep_data(n=250, seed=0, beta=1.0):
    """Y additionally depends on X: conditionally dependent."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 2))
    base = z.sum(axis=1)
    x = base + rng.standard_normal(n)
    y = base + beta * x + rng.standard_normal(n)
    return np.column_stack([x, y, z])


@pytest.mark.parametrize("name", SHIPPED)
class TestInterfaceContract:
    def test_registered(self, name):
        assert name in list_ci_tests()

    def test_returns_probability_and_finite_statistic(self, name):
        ci = create_ci_test(name, _linear_ci_data(), **FAST_KWARGS.get(name, {}))
        p, stat = ci(0, 1, [2, 3])
        assert 0.0 <= p <= 1.0
        assert np.isfinite(stat)

    def test_empty_conditioning_set(self, name):
        ci = create_ci_test(name, _linear_ci_data(), **FAST_KWARGS.get(name, {}))
        p, stat = ci(0, 1, [])
        assert 0.0 <= p <= 1.0

    def test_condition_set_none(self, name):
        ci = create_ci_test(name, _linear_ci_data(), **FAST_KWARGS.get(name, {}))
        p, _ = ci(0, 1, None)
        assert 0.0 <= p <= 1.0

    def test_repeated_call_is_deterministic(self, name):
        ci = create_ci_test(name, _linear_ci_data(), **FAST_KWARGS.get(name, {}))
        assert ci(0, 1, [2, 3]) == ci(0, 1, [2, 3])

    def test_symmetric_in_x_and_y(self, name):
        """GCM and WGCM score the residual product, which is symmetric."""
        if name == "pcm":
            pytest.skip("PCM projects E[Y | X, Z] and is directional; see TestPCM")
        ci = create_ci_test(name, _linear_ci_data(), **FAST_KWARGS.get(name, {}))
        p_xy, _ = ci(0, 1, [2, 3])
        ci2 = create_ci_test(name, _linear_ci_data(), **FAST_KWARGS.get(name, {}))
        p_yx, _ = ci2(1, 0, [2, 3])
        assert p_xy == pytest.approx(p_yx, abs=0.25)

    def test_tiny_sample_declines_rather_than_crashing(self, name):
        rng = np.random.default_rng(0)
        ci = create_ci_test(
            name, rng.standard_normal((6, 4)), **FAST_KWARGS.get(name, {})
        )
        p, stat = ci(0, 1, [2, 3])
        assert 0.0 <= p <= 1.0

    def test_set_data_keeps_the_base_contract(self, name):
        ci = create_ci_test(name, _linear_ci_data(), **FAST_KWARGS.get(name, {}))
        ci(0, 1, [2, 3])
        assert ci.pvalue_cache
        new = _linear_dep_data()
        ci.set_data(new, data_hash="new")
        assert ci.data is new and ci.data_hash == "new"
        assert not ci.pvalue_cache
        assert ci(0, 1, [2, 3])[0] < 0.05

    def test_data_assignment_still_routes_through_the_setter(self, name):
        ci = create_ci_test(name, _linear_ci_data(), **FAST_KWARGS.get(name, {}))
        data = _linear_ci_data()
        data[3, 1] = np.nan
        ci.data = data
        assert ci._has_nan

    def test_nan_rows_handled(self, name):
        data = _linear_ci_data()
        data[5, 0] = np.nan
        data[17, 2] = np.nan
        ci = create_ci_test(name, data, **FAST_KWARGS.get(name, {}))
        p, _ = ci(0, 1, [2, 3])
        assert 0.0 <= p <= 1.0

    def test_detects_a_strong_linear_alternative(self, name):
        ci = create_ci_test(name, _linear_dep_data(), **FAST_KWARGS.get(name, {}))
        p, _ = ci(0, 1, [2, 3])
        assert p < 0.05, f"{name} missed a strong linear alternative (p={p})"


class TestGCM:
    def test_matches_a_hand_computed_z_statistic(self):
        """With no conditioning and iid variance, GCM is the residual z-score."""
        rng = np.random.default_rng(3)
        n = 400
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)
        data = np.column_stack([x, y])

        ci = create_ci_test("gcm", data, variance="iid", crossfit="full")
        _, stat = ci(0, 1, [])

        q = (x - x.mean()) * (y - y.mean())
        expected = q.mean() / (q.std(ddof=1) / np.sqrt(n))
        assert stat == pytest.approx(expected, rel=1e-6)

    def test_hac_is_wider_than_iid_on_autocorrelated_data(self):
        rng = np.random.default_rng(4)
        n = 400
        e = rng.standard_normal(n)
        x = np.empty(n)
        x[0] = e[0]
        for t in range(1, n):
            x[t] = 0.9 * x[t - 1] + e[t]
        y = np.roll(x, 1) + 0.5 * rng.standard_normal(n)
        data = np.column_stack([x, y])

        iid = create_ci_test("gcm", data, variance="iid", crossfit="full")(0, 1, [])
        hac = create_ci_test("gcm", data, variance="hac", crossfit="full")(0, 1, [])
        assert abs(hac[1]) < abs(iid[1])

    def test_rejects_bad_variance_option(self):
        with pytest.raises(ValueError):
            create_ci_test("gcm", _linear_ci_data(), variance="banana")

    def test_rejects_bad_crossfit_option(self):
        with pytest.raises(ValueError):
            create_ci_test("gcm", _linear_ci_data(), crossfit="banana")


class TestWGCM:
    def test_beats_gcm_on_sign_changing_covariance(self):
        """The alternative GCM is built to miss: covariance flips sign with Z."""
        rng = np.random.default_rng(5)
        n = 400
        z = rng.standard_normal(n)
        x = rng.standard_normal(n)
        y = np.sign(z) * x + rng.standard_normal(n)
        data = np.column_stack([x, y, z])

        p_gcm, _ = create_ci_test("gcm", data)(0, 1, [2])
        p_wgcm, _ = create_ci_test("wgcm", data, n_boot=499)(0, 1, [2])
        assert p_gcm > 0.10, "GCM should be blind here; the fixture is wrong"
        assert p_wgcm < p_gcm

    def test_published_defaults_are_full_and_iid(self):
        ci = create_ci_test("wgcm", _linear_ci_data())
        assert ci.crossfit == "full"
        assert ci.calibration == "iid"

    def test_rejects_bad_calibration(self):
        with pytest.raises(ValueError):
            create_ci_test("wgcm", _linear_ci_data(), calibration="banana")


class TestPCM:
    def test_declares_conditional_mean_target(self):
        assert create_ci_test("pcm", _linear_ci_data()).inferential_target == (
            "conditional_mean"
        )

    def test_one_sided_by_default(self):
        assert create_ci_test("pcm", _linear_ci_data()).one_sided is True

    @pytest.mark.parametrize("mode", ["random", "full"])
    def test_rejects_crossfit_modes_it_does_not_implement(self, mode):
        with pytest.raises(ValueError, match="crossfit"):
            create_ci_test("pcm", _linear_ci_data(), crossfit=mode)

    def test_reused_instance_does_not_serve_the_reverse_direction_from_cache(self):
        rng = np.random.default_rng(0)
        n = 300
        z = rng.standard_normal(n)
        x = z + rng.standard_normal(n)
        y = np.sin(2 * x) + z + 0.5 * rng.standard_normal(n)
        data = np.column_stack([x, y, z])
        fresh_yx = create_ci_test("pcm", data)(1, 0, [2])
        ci = create_ci_test("pcm", data)
        ci(0, 1, [2])
        assert ci(1, 0, [2]) == fresh_yx

    def test_unscored_rows_are_excluded_not_zeroed(self, monkeypatch):
        import causalts.ci_tests.pcm as pcm_mod

        real_make = pcm_mod.make_regressor
        calls = {"n": 0}

        def flaky_make(name):
            # Fail the projection on the first fold only; each fold makes two
            # regressors when Z is non-empty (projection, then one multi-output
            # fit on Z), so this fires exactly once.
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("fold failure")
            return real_make(name)

        ci = create_ci_test("pcm", _linear_dep_data())
        monkeypatch.setattr(pcm_mod, "make_regressor", flaky_make)
        x, y, Z = ci._prepare(0, 1, [2, 3])
        q, scored = ci._score_series(x, y, Z)
        assert 0 < scored.sum() < len(q)

        calls["n"] = 0
        _, stat = ci(0, 1, [2, 3])
        q_kept = q[scored]
        assert stat == pytest.approx(q_kept.mean() / pcm_mod.hac_se(q_kept))
