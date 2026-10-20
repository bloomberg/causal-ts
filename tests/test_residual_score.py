# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for blocked cross-fitting and dependent calibration."""

import numpy as np
import pytest

from causalts.ci_tests._residual_score import (
    BlockCrossFitter,
    FeatureBank,
    SplineSelectRidge,
    bartlett_bandwidth,
    blocked_folds,
    hac_se,
    make_regressor,
    moving_block_bootstrap,
    multiplier_bootstrap,
)


class TestSplineSelectRidge:
    """The default nuisance learner of GCM/WGCM/PCM."""

    def test_spline_path_matches_sklearn_pipeline(self):
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import SplineTransformer, StandardScaler

        rng = np.random.default_rng(0)
        for n, dz in ((40, 1), (120, 2), (300, 3)):
            Z = rng.standard_normal((n, dz))
            y = np.sin(2 * Z[:, 0]) + 0.5 * rng.standard_normal(n)
            Zt = 1.3 * rng.standard_normal((50, dz))  # includes out-of-range rows
            ref = make_pipeline(
                StandardScaler(),
                SplineTransformer(n_knots=12, degree=3, include_bias=False),
                RidgeCV(alphas=np.logspace(-3, 3, 13)),
            ).fit(Z, y)
            ours = SplineSelectRidge(select=False).fit(Z, y)
            assert np.allclose(ours.predict(Zt), ref.predict(Zt), atol=1e-9)

    def test_wide_design_matches_sklearn(self):
        """p > n takes the dual-Gram path; it must give the same fit."""
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import SplineTransformer, StandardScaler

        rng = np.random.default_rng(7)
        Z = rng.standard_normal((40, 6))  # 6 x 13 = 78 basis columns > 40 rows
        y = np.sin(2 * Z[:, 0]) + 0.5 * rng.standard_normal(40)
        Zt = 1.3 * rng.standard_normal((30, 6))
        ref = make_pipeline(
            StandardScaler(),
            SplineTransformer(n_knots=12, degree=3, include_bias=False),
            RidgeCV(alphas=np.logspace(-3, 3, 13)),
        ).fit(Z, y)
        ours = SplineSelectRidge(select=False).fit(Z, y)
        assert np.allclose(ours.predict(Zt), ref.predict(Zt), atol=1e-8)

    def test_selects_linear_on_linear_data_and_spline_on_nonlinear(self):
        rng = np.random.default_rng(1)
        Z = rng.standard_normal((200, 2))
        lin = SplineSelectRidge().fit(Z, Z @ [1.0, -0.5] + rng.standard_normal(200))
        wig = SplineSelectRidge().fit(
            Z, np.sin(3 * Z[:, 0]) + 0.3 * rng.standard_normal(200)
        )
        assert lin.models_[0][0] is False
        assert wig.models_[0][0] is True

    def test_multi_output_equals_separate_fits(self):
        rng = np.random.default_rng(2)
        Z = rng.standard_normal((150, 3))
        y1 = Z[:, 0] + rng.standard_normal(150)
        y2 = np.cos(Z[:, 1]) + rng.standard_normal(150)
        joint = SplineSelectRidge().fit(Z, np.column_stack([y1, y2])).predict(Z)
        assert np.allclose(joint[:, 0], SplineSelectRidge().fit(Z, y1).predict(Z))
        assert np.allclose(joint[:, 1], SplineSelectRidge().fit(Z, y2).predict(Z))

    def test_degenerate_inputs_stay_finite(self):
        rng = np.random.default_rng(3)
        y = rng.standard_normal(30)
        const = np.column_stack([np.ones(30), rng.standard_normal(30)])
        wide = rng.standard_normal((30, 6))  # 6 x 13 basis columns > 30 rows
        for Z in (const, wide, np.ones((30, 1))):
            assert np.all(np.isfinite(SplineSelectRidge().fit(Z, y).predict(Z)))

    def test_constant_training_column_is_inert_on_held_out_rows(self):
        rng = np.random.default_rng(6)
        Z = np.column_stack([rng.standard_normal(60), np.zeros(60)])
        y = Z[:, 0] + 0.5 * rng.standard_normal(60)
        Zt = np.column_stack([np.linspace(-1, 1, 5), np.full(5, 7.0)])
        with_dead = SplineSelectRidge().fit(Z, y).predict(Zt)
        without = SplineSelectRidge().fit(Z[:, :1], y).predict(Zt[:, :1])
        assert np.allclose(with_dead, without, atol=1e-12)

    def test_trending_z_linear_null_stays_calibrated_under_blocked_crossfit(self):
        """Blocked folds on a drifting Z put held-out rows outside the training
        range, where the spline basis is flat; the linear candidate must still be
        picked when the truth is linear, so the null is not rejected."""
        from causalts.ci_tests import create_ci_test

        rej = {"gcm": 0, "pcm": 0}
        for r in range(40):
            g = np.random.default_rng(r)
            z = 3 * np.linspace(-1, 1, 200) + 0.3 * g.standard_normal(200)
            x = z + g.standard_normal(200)
            y = z + g.standard_normal(200)
            d = np.column_stack([x, y, z])
            for name in rej:
                rej[name] += create_ci_test(name, d)(0, 1, [2])[0] <= 0.05
        assert rej["gcm"] <= 8
        assert rej["pcm"] <= 8

    def test_is_the_score_tests_default(self):
        from causalts.ci_tests import create_ci_test

        d = np.random.default_rng(4).standard_normal((60, 3))
        for name in ("gcm", "wgcm", "pcm"):
            assert create_ci_test(name, d).regressors == ("spline_select",)
        assert create_ci_test("pcm", d).projection == "spline_select"

    def test_nonlinear_null_is_not_rejected_like_linear_ridge(self):
        """The regression this learner fixes: with E[X|Z], E[Y|Z] nonlinear and
        X indep Y | Z, a linear nuisance leaves signal in both residuals and GCM
        rejects far above alpha."""
        from causalts.ci_tests import create_ci_test

        rng = np.random.default_rng(5)
        rej = {"ridge": 0, "spline_select": 0}
        for _ in range(40):
            z = rng.standard_normal(150)
            x = np.sin(2 * z) + z**2 + 0.5 * rng.standard_normal(150)
            y = np.cos(2 * z) + z**2 + 0.5 * rng.standard_normal(150)
            d = np.column_stack([x, y, z])
            for reg in rej:
                p = create_ci_test("gcm", d, regressors=(reg,), crossfit="full")(
                    0, 1, [2]
                )[0]
                rej[reg] += p <= 0.05
        assert rej["ridge"] >= 20
        assert rej["spline_select"] <= 8


class TestBlockedFolds:
    def test_eval_blocks_partition_the_series(self):
        folds = list(blocked_folds(100, n_splits=5, gap=0))
        covered = np.concatenate([e for _, e in folds])
        assert np.array_equal(np.sort(covered), np.arange(100))

    def test_eval_blocks_are_contiguous(self):
        for _, eval_idx in blocked_folds(100, n_splits=5, gap=3):
            assert np.array_equal(eval_idx, np.arange(eval_idx[0], eval_idx[-1] + 1))

    @pytest.mark.parametrize("gap", [0, 1, 5, 10])
    def test_embargo_excludes_neighbours(self, gap):
        """The whole point: no training row sits within `gap` of an eval row."""
        n = 120
        for train_idx, eval_idx in blocked_folds(n, n_splits=4, gap=gap):
            lo, hi = eval_idx[0], eval_idx[-1]
            forbidden = set(range(max(0, lo - gap), min(n, hi + gap + 1)))
            assert not (set(train_idx.tolist()) & forbidden)

    def test_rejects_bad_arguments(self):
        with pytest.raises(ValueError):
            list(blocked_folds(50, n_splits=1))
        with pytest.raises(ValueError):
            list(blocked_folds(50, n_splits=3, gap=-1))


class TestBlockCrossFitter:
    def test_residuals_have_full_length(self):
        rng = np.random.default_rng(0)
        Z = rng.standard_normal((80, 2))
        y = Z[:, 0] + rng.standard_normal(80)
        res = BlockCrossFitter(n_splits=4, gap=2).residuals(y, Z)
        assert res.shape == (80,)
        assert np.all(np.isfinite(res))

    def test_removes_linear_signal(self):
        rng = np.random.default_rng(1)
        Z = rng.standard_normal((300, 1))
        y = 3.0 * Z[:, 0] + 0.1 * rng.standard_normal(300)
        res = BlockCrossFitter(n_splits=5, gap=1, regressors=("ridge",)).residuals(y, Z)
        assert res.std() < 0.5 * y.std()

    def test_empty_conditioning_set_centres_only(self):
        y = np.arange(50, dtype=float)
        res = BlockCrossFitter().residuals(y, np.zeros((50, 0)))
        assert np.allclose(res, y - y.mean())

    def test_gap_from_lag_structure(self):
        fitter = BlockCrossFitter(gap=None)
        assert fitter.effective_gap == 1
        fitter.set_gap(7)
        assert fitter.effective_gap == 7

    def test_unknown_learner_rejected(self):
        with pytest.raises(ValueError):
            BlockCrossFitter(regressors=("nope",))
        with pytest.raises(ValueError):
            make_regressor("nope")


class TestCalibration:
    def test_hac_matches_iid_when_uncorrelated(self):
        rng = np.random.default_rng(2)
        q = rng.standard_normal(500)
        iid = q.std(ddof=1) / np.sqrt(500)
        assert hac_se(q) == pytest.approx(iid, rel=0.35)

    def test_hac_exceeds_iid_under_positive_autocorrelation(self):
        rng = np.random.default_rng(3)
        e = rng.standard_normal(600)
        q = np.empty(600)
        q[0] = e[0]
        for t in range(1, 600):
            q[t] = 0.8 * q[t - 1] + e[t]
        assert hac_se(q) > 1.5 * q.std(ddof=1) / np.sqrt(600)

    def test_bandwidth_positive(self):
        assert bartlett_bandwidth(40) >= 1
        assert bartlett_bandwidth(1000) >= 1

    def test_multiplier_bootstrap_returns_valid_pvalue(self):
        rng = np.random.default_rng(4)
        Q = rng.standard_normal((150, 3))
        for kind in ("iid", "dependent"):
            p, stat = multiplier_bootstrap(Q, n_boot=199, kind=kind, rng=rng)
            assert 0.0 < p <= 1.0
            assert np.isfinite(stat)

    def test_multiplier_bootstrap_detects_a_shifted_score(self):
        rng = np.random.default_rng(5)
        Q = rng.standard_normal((200, 2)) + 0.6
        p, _ = multiplier_bootstrap(Q, n_boot=299, kind="iid", rng=rng)
        assert p < 0.05

    def test_multiplier_bootstrap_rejects_unknown_kind(self):
        with pytest.raises(ValueError):
            multiplier_bootstrap(np.zeros((20, 1)), kind="banana")

    def test_moving_block_bootstrap_runs(self):
        rng = np.random.default_rng(6)
        q = rng.standard_normal(120)
        p, stat = moving_block_bootstrap(
            q, lambda v: float(v.mean()), n_boot=99, rng=rng
        )
        assert 0.0 < p <= 1.0
        assert np.isfinite(stat)


class TestFeatureBank:
    def test_always_includes_constant(self):
        rng = np.random.default_rng(7)
        bank = FeatureBank()
        W = bank.build(rng.standard_normal((60, 2)))
        assert bank.names_[0] == "constant"
        assert np.allclose(W[:, 0], 1.0)

    def test_empty_conditioning_set(self):
        W = FeatureBank().build(None, n=30)
        assert W.shape == (30, 1)
        assert np.allclose(W, 1.0)

    def test_respects_max_weights(self):
        rng = np.random.default_rng(8)
        bank = FeatureBank(max_dirs=3, max_weights=4)
        W = bank.build(rng.standard_normal((80, 6)))
        assert W.shape[1] <= 4
        assert len(bank.names_) == W.shape[1]

    def test_bank_size_independent_of_conditioning_dimension(self):
        rng = np.random.default_rng(9)
        small = FeatureBank().build(rng.standard_normal((80, 2))).shape[1]
        large = FeatureBank().build(rng.standard_normal((80, 12))).shape[1]
        assert small == large

    def test_rejects_unknown_weight(self):
        with pytest.raises(ValueError):
            FeatureBank(weights=("constant", "nope"))


class TestStableSeeding:
    """Seeds must not come from Python's per-process-randomised hash()."""

    def test_stable_seed_is_deterministic_across_processes(self):
        import subprocess
        import sys

        code = (
            "from causalts.ci_tests.cit_test import stable_seed;"
            "print(stable_seed('abc', 'def'))"
        )
        seen = {
            subprocess.run(
                [sys.executable, "-c", code], capture_output=True, text=True
            ).stdout.strip()
            for _ in range(3)
        }
        assert len(seen) == 1, f"seed varies across processes: {seen}"

    def test_bootstrap_pvalue_is_stable_across_processes(self):
        """A resampling test must return the same p-value in a fresh process."""
        import subprocess
        import sys

        code = (
            "import numpy as np;"
            "from causalts.ci_tests import create_ci_test;"
            "rng = np.random.default_rng(0);"
            "z = rng.standard_normal((150, 2));"
            "d = np.column_stack([z[:,0] + rng.standard_normal(150),"
            " z[:,0] + rng.standard_normal(150), z]);"
            "print(round(create_ci_test('wgcm', d, n_boot=199)(0, 1, [2, 3])[0], 10))"
        )
        seen = {
            subprocess.run(
                [sys.executable, "-c", code], capture_output=True, text=True
            ).stdout.strip()
            for _ in range(3)
        }
        assert len(seen) == 1, f"p-value varies across processes: {seen}"


class TestReviewFixes:
    def test_tied_values_share_a_rank_weight(self):
        from causalts.ci_tests._residual_score import _centred_ranks

        r = _centred_ranks(np.array([3.0, 1.0, 3.0, 1.0, 2.0]))
        assert r[0] == r[2] and r[1] == r[3]
        assert np.allclose(_centred_ranks(np.full(6, 4.0)), 0.0)
        v = np.random.default_rng(0).standard_normal(50)
        ordinal = np.argsort(np.argsort(v))
        assert np.array_equal(_centred_ranks(v), 2.0 * ordinal / 49.0 - 1.0)

    @pytest.mark.parametrize("name", ["gcm", "wgcm", "pcm"])
    def test_conditioning_order_does_not_matter(self, name):
        from causalts.ci_tests import create_ci_test

        rng = np.random.default_rng(3)
        Z = rng.standard_normal((150, 3))
        x = np.tanh(Z[:, 0]) + 0.5 * Z[:, 1] + rng.standard_normal(150)
        y = Z[:, 0] - 0.4 * Z[:, 2] + 0.2 * x + rng.standard_normal(150)
        data = np.column_stack([x, y, Z])
        kw = {"max_weights": 3} if name == "wgcm" else {}
        a = create_ci_test(name, data, **kw)(0, 1, [2, 3, 4])
        b = create_ci_test(name, data, **kw)(0, 1, [4, 2, 3])
        assert a == b

    def test_wgcm_cache_key_covers_bank_settings(self):
        from causalts.ci_tests.wgcm import WGCM

        data = np.random.default_rng(0).standard_normal((40, 3))
        seed_material = WGCM(data)._param_hash()
        default = WGCM(data)._cache_hash()
        for kw in ({"max_dirs": 1}, {"max_weights": 4}, {"block_length": 5}):
            assert WGCM(data, **kw)._cache_hash() != default
            # The draws are seeded from _param_hash, which must not move.
            assert WGCM(data, **kw)._param_hash() == seed_material

    @pytest.mark.parametrize("name", ["gcm", "wgcm", "pcm"])
    def test_entries_from_older_cache_keys_are_not_reused(self, name):
        from causalts.ci_tests import create_ci_test

        rng = np.random.default_rng(2)
        data = rng.standard_normal((120, 3))
        ci = create_ci_test(name, data)
        key = ci._get_array_hash(0, 1, [2])
        if name == "pcm":
            from hashlib import sha1

            key += "_x" + sha1(np.ascontiguousarray(data[:, 0])).hexdigest()
        ci.pvalue_cache[(key, ci._param_hash())] = [0.123456, 9.0]  # pre-fix format
        assert ci(0, 1, [2])[0] != 0.123456

    def test_random_crossfit_does_not_depend_on_xy_order(self):
        from causalts.ci_tests import create_ci_test

        rng = np.random.default_rng(4)
        z = rng.standard_normal(150)
        data = np.column_stack(
            [z + rng.standard_normal(150), np.sin(z) + rng.standard_normal(150), z]
        )
        for name in ("gcm", "wgcm"):
            a = create_ci_test(name, data, crossfit="random")(0, 1, [2])
            b = create_ci_test(name, data, crossfit="random")(1, 0, [2])
            assert a == b, name

    def test_cache_identity_covers_min_samples(self):
        from causalts.ci_tests import create_ci_test
        from causalts.ci_tests.pyRcot_gpu import RCOTGPU

        data = np.random.default_rng(0).standard_normal((60, 3))
        for name in ("gcm", "wgcm", "pcm"):
            a = create_ci_test(name, data, min_samples=30)._cache_hash()
            b = create_ci_test(name, data, min_samples=40)._cache_hash()
            assert a != b, name
        a = RCOTGPU(data, device="cpu", min_samples=30).param_hash
        assert a != RCOTGPU(data, device="cpu", min_samples=40).param_hash

    def test_pcm_rejects_an_unknown_projection(self):
        from causalts.ci_tests.pcm import PCM

        with pytest.raises(ValueError, match="nuisance learner"):
            PCM(np.zeros((40, 3)), projection="splines")

    def test_moving_block_bootstrap_detects_a_mean_shift(self):
        rng = np.random.default_rng(6)
        noise = rng.standard_normal(200)
        mean = lambda v: float(v.mean())  # noqa: E731
        p_shift, _ = moving_block_bootstrap(noise + 0.5, mean, n_boot=199, rng=rng)
        p_null, _ = moving_block_bootstrap(noise, mean, n_boot=199, rng=rng)
        assert p_shift < 0.01
        assert p_null > 0.05
