# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for GRACE's regression-based refinement (``refine="shrink"``) and the spline basis."""

import numpy as np
import pandas as pd
import pytest

from causalts.grace.grow import _rff_basis, feature_block, grow_candidates, spline_basis
from causalts.grace.shrink import block_test, shrink_parents


def _additive_var(T=500, seed=0, noise="gaussian"):
    """x1 <- sin(2 x0(t-1)), x2 <- .5 x1(t-1)^2, x3 independent AR(1); all with own lag."""
    rng = np.random.default_rng(seed)
    d = 4
    e = (
        rng.standard_normal((T, d))
        if noise == "gaussian"
        else rng.standard_t(2, (T, d))
    )
    X = np.zeros((T, d))
    for t in range(1, T):
        X[t, 0] = 0.5 * X[t - 1, 0] + e[t, 0]
        X[t, 1] = 0.3 * X[t - 1, 1] + np.sin(2 * X[t - 1, 0]) + 0.5 * e[t, 1]
        X[t, 2] = 0.3 * X[t - 1, 2] + 0.5 * X[t - 1, 1] ** 2 + 0.5 * e[t, 2]
        X[t, 3] = 0.5 * X[t - 1, 3] + e[t, 3]
    truth = np.zeros((d, d, 3), dtype=bool)
    for j in range(d):
        truth[j, j, 1] = True
    truth[0, 1, 1] = truth[1, 2, 1] = True
    return X, truth


def _leverage(n_blocks_fn, n=400, seed=0):
    rng = np.random.default_rng(seed)
    cols = [n_blocks_fn(rng.standard_normal(n)) for _ in range(3)]
    A = np.column_stack([np.ones(n)] + cols)
    Q, _ = np.linalg.qr(A)
    return float((Q**2).sum(1).max())


def test_spline_basis_shape_and_bounded_leverage():
    x = np.random.default_rng(0).standard_normal(500)
    B = spline_basis(x)
    assert B.shape == (500, 7)  # x + n_knots - 2 terms
    np.testing.assert_allclose(B.mean(0), 0, atol=1e-10)
    np.testing.assert_allclose(B.std(0), 1, atol=1e-10)
    # The random-Fourier basis lets a few extreme rows be fitted almost exactly;
    # the natural spline (linear beyond the boundary knots) does not.
    assert _leverage(spline_basis) < 0.7
    assert _leverage(_rff_basis) > 0.9


def test_spline_basis_constant_input():
    B = spline_basis(np.ones(50))
    assert B.shape[0] == 50 and np.isfinite(B).all()


def test_feature_block_dispatch():
    x = np.linspace(-2, 2, 100)
    np.testing.assert_array_equal(feature_block(x, "rff"), _rff_basis(x))
    np.testing.assert_array_equal(feature_block(x, "spline"), spline_basis(x))
    with pytest.raises(ValueError, match="basis"):
        feature_block(x, "poly")


def _grow_pin_data():
    rng = np.random.default_rng(7)
    T, d = 300, 5
    X = rng.standard_normal((T, d))
    for t in range(1, T):
        X[t, 1] += np.sin(2 * X[t - 1, 0])
        X[t, 2] += 0.4 * X[t - 1, 1] ** 2
        X[t, 3] += 0.3 * np.tanh(X[t - 1, 2])
        X[t, 4] += 0.5 * X[t - 1, 4]
    est = np.zeros((d, d, 3), bool)
    est[0, 1, 1] = True
    sk = est.copy()
    sk[2, 3, 1] = True
    return X, est, sk, np.linspace(-1, 1, T)[:, None]


# Output of grow_candidates before the basis option existed, so the default stays
# byte-identical.
_PIN_RANK_EXOG_Q05 = [(1, 2, 1), (4, 4, 1), (4, 4, 2)]
_PIN_PLAIN_Q095 = [
    (0, 0, 1), (0, 0, 2), (0, 1, 2), (0, 2, 2), (0, 3, 1), (0, 3, 2), (0, 4, 1),
    (0, 4, 2), (1, 0, 2), (1, 1, 2), (1, 2, 1), (1, 2, 2), (2, 0, 2), (2, 1, 1),
    (2, 1, 2), (2, 2, 2), (2, 4, 1), (2, 4, 2), (3, 0, 1), (3, 0, 2), (3, 1, 1),
    (3, 1, 2), (3, 2, 1), (3, 4, 1), (3, 4, 2), (4, 0, 1), (4, 1, 2), (4, 2, 2),
    (4, 3, 2), (4, 4, 1), (4, 4, 2),
]  # fmt: skip


def test_grow_default_basis_unchanged():
    X, est, sk, exog = _grow_pin_data()
    for kw, pin in [
        (dict(q=0.5, transform="rank", exog=exog), _PIN_RANK_EXOG_Q05),
        (dict(q=0.95), _PIN_PLAIN_Q095),
    ]:
        added, n = grow_candidates(X, est, sk, max_lag=2, **kw)
        assert n == 48
        assert sorted(map(tuple, np.argwhere(added).tolist())) == pin
        explicit, _ = grow_candidates(X, est, sk, max_lag=2, basis="rff", **kw)
        np.testing.assert_array_equal(explicit, added)


def test_grow_spline_basis_finds_nonlinear_edge():
    X, _ = _additive_var()
    empty = np.zeros((4, 4, 3), dtype=bool)
    added, _ = grow_candidates(X, empty, empty, max_lag=2, basis="spline")
    assert added[0, 1, 1] and added[1, 2, 1]
    with pytest.raises(ValueError, match="basis"):
        grow_candidates(X, empty, empty, max_lag=2, basis="poly")


def test_block_test_null_and_alternative():
    rng = np.random.default_rng(1)
    n = 400
    z, w = rng.standard_normal(n), rng.standard_normal(n)
    y = np.sin(2 * z) + 0.5 * rng.standard_normal(n)
    Q, _ = np.linalg.qr(np.column_stack([np.ones(n), spline_basis(z)]))
    p_null, r2_null = block_test(y, spline_basis(w), Q)
    Q0, _ = np.linalg.qr(np.ones((n, 1)))
    p_alt, r2_alt = block_test(y, spline_basis(z), Q0)
    assert p_null > 0.001 and 0 <= r2_null < 0.05
    assert p_alt < 1e-10 and r2_alt > 0.5


def test_block_test_undefined_cases():
    n = 30
    y = np.arange(n, dtype=float)
    Q, _ = np.linalg.qr(
        np.column_stack([np.ones(n), np.random.default_rng(0).standard_normal((n, 15))])
    )
    # Too few residual degrees of freedom: not tested.
    assert block_test(y, np.random.default_rng(1).standard_normal((n, 8)), Q) == (
        1.0,
        0.0,
    )
    # Block inside span(Q): nothing to test.
    assert block_test(y, Q[:, 1:3], Q) == (1.0, 0.0)


@pytest.mark.parametrize("basis", ["spline", "rff"])
def test_shrink_parents_prunes_to_truth(basis):
    X, truth = _additive_var()
    cand = np.zeros_like(truth)
    cand[:, :, 1:] = True  # every lagged pair is a candidate
    G, p, r2 = shrink_parents(X, cand, max_lag=2, basis=basis)
    np.testing.assert_array_equal(G, truth)
    assert G.dtype == bool and p.shape == r2.shape == truth.shape
    assert (p[G] <= 0.001).all() and (p[cand & ~G] > 0.001).all()
    assert ((r2 >= 0) & (r2 <= 1)).all()
    assert r2[G].min() > r2[cand & ~G].max()
    # Lag 0 and non-candidates are untested.
    assert (p[:, :, 0] == 1).all() and (r2[:, :, 0] == 0).all()


def test_shrink_parents_robust_under_heavy_tails():
    X, truth = _additive_var(T=800, seed=3, noise="t2")
    cand = np.zeros_like(truth)
    cand[:, :, 1:] = True
    G_r, p_r, _ = shrink_parents(X, cand, max_lag=2, robust=True)
    G_p, p_p, _ = shrink_parents(X, cand, max_lag=2, robust=False)
    fp = lambda G: int((G & ~truth).sum())  # noqa: E731
    assert fp(G_r) <= fp(G_p)
    assert (G_r & truth).sum() == truth.sum()
    assert not np.allclose(p_r, p_p)  # the robust path is actually used


def test_shrink_parents_conditions_on_exog():
    # Two series that share only a linear trend: the trend explains the apparent
    # lagged dependence once it is in the design.
    rng = np.random.default_rng(5)
    T = 600
    trend = np.linspace(-3, 3, T)
    X = np.column_stack(
        [trend + rng.standard_normal(T), trend + rng.standard_normal(T)]
    )
    cand = np.zeros((2, 2, 2), dtype=bool)
    cand[0, 1, 1] = True
    G_no, _, _ = shrink_parents(X, cand, max_lag=1)
    G_c, _, _ = shrink_parents(X, cand, max_lag=1, exog=trend[:, None])
    assert G_no[0, 1, 1] and not G_c[0, 1, 1]


def test_shrink_parents_validation():
    X, truth = _additive_var(T=100)
    with pytest.raises(ValueError, match="alpha"):
        shrink_parents(X, truth, max_lag=2, alpha=0)
    with pytest.raises(ValueError, match="shape"):
        shrink_parents(X, truth[:, :, :2], max_lag=2)
    Xn = X.copy()
    Xn[3, 1] = np.nan
    with pytest.raises(ValueError, match="missing"):
        shrink_parents(Xn, truth, max_lag=2)


def test_run_cdnots_gated_shrink(small_df):
    from causalts.grace import run_cdnots_gated

    res = run_cdnots_gated(
        df=small_df, max_lag=1, alpha=0.1, refine="shrink", verbose=False
    )
    assert res.refine == "shrink"
    assert res.gate_values is None and res.gate_threshold is None
    assert res.edge_pvalue.shape == res.edge_strength.shape == res.cg_tig.shape
    assert res.cg_tig[0, 1, 1] and res.cg_tig[1, 2, 1]
    assert not res.cg_tig[:, :, 0].any()
    assert "refine='shrink'" in repr(res)


def test_run_cdnots_gated_shrink_grow_adds_missed_edge():
    from causalts.grace import run_cdnots_gated

    X, truth = _additive_var(T=600)
    df = pd.DataFrame(X, columns=list("abcd"))
    skeleton = truth.astype(np.int8).copy()
    skeleton[1, 2, 1] = 0  # the skeleton missed x1 -> x2
    res = run_cdnots_gated(
        df=df,
        max_lag=2,
        skeleton=skeleton,
        refine="shrink",
        grow_rounds=2,
        verbose=False,
    )
    assert res.cg_tig[1, 2, 1]
    assert res.grow_history[0]["n_added"] >= 1 and res.grow_history[0]["n_kept"] >= 1


def test_run_cdnots_gated_shrink_validation(small_df):
    from causalts.grace import run_cdnots_gated

    with pytest.raises(ValueError, match="refine"):
        run_cdnots_gated(df=small_df, max_lag=1, refine="lasso", verbose=False)
    with pytest.raises(ValueError, match="lag"):
        run_cdnots_gated(
            df=small_df,
            max_lag=1,
            refine="shrink",
            include_instantaneous=True,
            verbose=False,
        )
    with pytest.raises(ValueError, match="shrink_basis"):
        run_cdnots_gated(
            df=small_df, max_lag=1, refine="shrink", shrink_basis="poly", verbose=False
        )
    df = small_df.copy()
    df.iloc[5, 1] = np.nan
    with pytest.raises(ValueError, match="missing"):
        run_cdnots_gated(df=df, max_lag=1, refine="shrink", verbose=False)


def test_run_cdnots_gated_default_is_gates(small_df):
    from causalts.grace import run_cdnots_gated

    res = run_cdnots_gated(
        df=small_df, max_lag=1, max_epochs=2, device="cpu", verbose=False
    )
    assert res.refine == "gates"
    assert res.gate_values is not None and res.edge_strength is None


def test_cli_grace_refine_shrink(small_df, tmp_path):
    import json

    from click.testing import CliRunner

    from causalts.cli import main

    csv = tmp_path / "data.csv"
    small_df.to_csv(csv, index=False)
    out = tmp_path / "out"
    result = CliRunner().invoke(
        main,
        ["-o", str(out), "-q", "discover", str(csv), "--algorithm", "grace"]
        + ["--max-lag", "1", "--refine", "shrink"],
    )
    assert result.exit_code == 0, result.output
    run_dir = next(out.glob("*/"))
    assert (run_dir / "edge_strength.npy").exists()
    assert (run_dir / "edge_pvalue.npy").exists()
    assert not (run_dir / "gate_values.npy").exists()
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["refine"] == "shrink"


def test_shrink_parents_saturated_order_is_data_driven():
    # Too many candidates for the sample: every block test is undefined at first.
    # The result must not depend on the order of the columns.
    X, truth = _additive_var(T=120, seed=2)
    rng = np.random.default_rng(0)
    X = np.column_stack([X, rng.standard_normal((120, 8))])
    d = X.shape[1]
    cand = np.zeros((d, d, 3), dtype=bool)
    cand[:, :, 1:] = True
    G, _, _ = shrink_parents(X, cand, max_lag=2)
    perm = np.r_[np.arange(4, d), np.arange(4)]  # true variables moved to the end
    Gp, _, _ = shrink_parents(X[:, perm], cand, max_lag=2)
    np.testing.assert_array_equal(Gp, G[np.ix_(perm, perm)])


def test_spline_basis_discrete_levels():
    binary = np.r_[np.zeros(250), np.ones(250)]
    B = spline_basis(binary)
    assert B.shape == (500, 1)
    three = np.repeat([0.0, 1.0, 2.0], 100)
    assert spline_basis(three).shape == (300, 2)


def test_shrink_and_grow_keep_binary_parent():
    rng = np.random.default_rng(0)
    T = 500
    x = rng.integers(0, 2, T).astype(float)
    y = np.zeros(T)
    y[1:] = 3 * x[:-1] + rng.standard_normal(T - 1)
    X = np.column_stack([x, y])
    cand = np.zeros((2, 2, 2), dtype=bool)
    cand[0, 1, 1] = True
    G, p, _ = shrink_parents(X, cand, max_lag=1)
    assert G[0, 1, 1] and p[0, 1, 1] < 1e-10
    empty = np.zeros((2, 2, 2), dtype=bool)
    added, _ = grow_candidates(X, empty, empty, max_lag=1, basis="spline")
    assert added[0, 1, 1]


def test_huber_handles_zero_mad_spike():
    from causalts.grace.shrink import _huber

    r = np.zeros(100)
    r[10] = 50.0
    h = _huber(r)
    assert 0 < h[10] < 5  # clipped, not left to dominate


def test_block_test_robust_downweights_outlier():
    rng = np.random.default_rng(4)
    n = 300
    z, w = rng.standard_normal(n), rng.standard_normal(n)
    y = np.sin(z) + 0.3 * rng.standard_normal(n)
    w[7], y[7] = 6.0, 40.0  # one aligned outlier in an otherwise unrelated w
    Q, _ = np.linalg.qr(np.column_stack([np.ones(n), spline_basis(z)]))
    p_plain, _ = block_test(y, spline_basis(w), Q, robust=False)
    p_rob, _ = block_test(y, spline_basis(w), Q, robust=True)
    assert p_plain < 1e-6 < p_rob


def test_shrink_parents_needs_enough_rows():
    X = np.random.default_rng(0).standard_normal((3, 2))
    with pytest.raises(ValueError, match="max_lag"):
        shrink_parents(X, np.ones((2, 2, 4), bool), max_lag=3)


def test_supplied_skeleton_shrink_uses_c_when_asked():
    from causalts.grace import run_cdnots_gated

    rng = np.random.default_rng(5)
    T = 600
    trend = np.linspace(-3, 3, T)
    df = pd.DataFrame(
        {"a": trend + rng.standard_normal(T), "b": trend + rng.standard_normal(T)}
    )
    sk = np.zeros((2, 2, 2), dtype=np.int8)
    sk[0, 1, 1] = 1
    kw = dict(df=df, max_lag=1, skeleton=sk, refine="shrink", verbose=False)
    assert run_cdnots_gated(**kw).cg_tig[0, 1, 1]  # trend-only edge kept without C
    res = run_cdnots_gated(include_C_in_model=True, **kw)
    assert not res.cg_tig[0, 1, 1]
    assert res.skeleton.shape == res.cg_tig.shape


def test_cli_shrink_rejects_pairwise_complete(small_df, tmp_path):
    from click.testing import CliRunner

    from causalts.cli import main

    csv = tmp_path / "data.csv"
    small_df.to_csv(csv, index=False)
    result = CliRunner().invoke(
        main,
        ["-o", str(tmp_path / "o"), "-q", "discover", str(csv)]
        + ["--algorithm", "grace", "--max-lag", "1", "--refine", "shrink"]
        + ["--impute", "pairwise_complete"],
    )
    assert result.exit_code != 0 and "var_em" in result.output


@pytest.mark.parametrize("algorithm", ["grace-ss", "cdnots"])
def test_cli_refine_shrink_rejected_for_other_algorithms(small_df, tmp_path, algorithm):
    from click.testing import CliRunner

    from causalts.cli import main

    csv = tmp_path / "data.csv"
    small_df.to_csv(csv, index=False)
    result = CliRunner().invoke(
        main,
        ["-o", str(tmp_path / "o"), "-q", "discover", str(csv)]
        + ["--algorithm", algorithm, "--max-lag", "1", "--refine", "shrink"],
    )
    assert result.exit_code != 0 and "--algorithm grace" in result.output


def test_run_cdnots_gated_shrink_rejects_max_lag_zero(small_df):
    from causalts.grace import run_cdnots_gated

    # Raised by the up-front check, not by shrink_parents after the skeleton run.
    with pytest.raises(ValueError, match="max_lag >= 1"):
        run_cdnots_gated(df=small_df, max_lag=0, refine="shrink", verbose=False)


def test_shrink_c_model_only_validates_preset_up_front(small_df, monkeypatch):
    # include_C_in_model without include_C is valid under shrink; a preset typo
    # must still fail before the CDNOTS skeleton runs.
    import causalts.cdnots.phase3_utils as p3
    from causalts.grace import run_cdnots_gated

    def skeleton_ran(*a, **k):
        raise AssertionError("skeleton ran before the preset was validated")

    monkeypatch.setattr(p3, "run_cdnots", skeleton_ran)
    with pytest.raises(ValueError):
        run_cdnots_gated(
            df=small_df,
            max_lag=1,
            refine="shrink",
            include_C=False,
            include_C_in_model=True,
            c_preset="not-a-preset",
            verbose=False,
        )


def test_cli_shrink_warns_on_training_flags(small_df, tmp_path):
    import json

    from click.testing import CliRunner

    from causalts.cli import main

    csv = tmp_path / "data.csv"
    small_df.to_csv(csv, index=False)
    out = tmp_path / "o"
    result = CliRunner().invoke(
        main,
        ["-o", str(out), "discover", str(csv), "--algorithm", "grace"]
        + ["--max-lag", "1", "--alpha", "0.1", "--refine", "shrink"]
        + ["--max-epochs", "500", "--patience", "3", "--effect-floor", "0"]
        + ["--no-plot"],
    )
    assert result.exit_code == 0, result.output
    assert "--max-epochs, --patience, --effect-floor have no effect" in result.output
    summary = json.loads(next(out.rglob("summary.json")).read_text())
    assert "max_epochs" not in summary and summary["refine"] == "shrink"


@pytest.mark.parametrize("algorithm", ["grace-ss", "cdnots"])
def test_cli_grow_rounds_rejected_for_other_algorithms(small_df, tmp_path, algorithm):
    from click.testing import CliRunner

    from causalts.cli import main

    csv = tmp_path / "data.csv"
    small_df.to_csv(csv, index=False)
    result = CliRunner().invoke(
        main,
        ["-o", str(tmp_path / "o"), "-q", "discover", str(csv)]
        + ["--algorithm", algorithm, "--max-lag", "1", "--grow-rounds", "2"],
    )
    assert result.exit_code != 0 and "--grow-rounds applies only" in result.output


def test_cli_previous_gated_configuration_reachable(small_df, tmp_path):
    import json

    from click.testing import CliRunner

    from causalts.cli import main

    csv = tmp_path / "data.csv"
    small_df.to_csv(csv, index=False)
    out = tmp_path / "o"
    result = CliRunner().invoke(
        main,
        ["-o", str(out), "-q", "discover", str(csv), "--algorithm", "grace"]
        + ["--max-lag", "1", "--max-epochs", "3", "--no-plot", "--no-ar-path"]
        + ["--lambda-rule", "density", "--grow-rounds", "0"],
    )
    assert result.exit_code == 0, result.output
    summary = json.loads(next(out.rglob("summary.json")).read_text())
    assert summary["ar_path"] is False and summary["lambda_rule"] == "density"
    assert not summary["gate_threshold_rule"].startswith("ar_")
    assert "grow_history" not in summary


def test_cli_default_summary_records_paper_configuration(small_df, tmp_path):
    import json

    from click.testing import CliRunner

    from causalts.cli import main

    csv = tmp_path / "data.csv"
    small_df.to_csv(csv, index=False)
    out = tmp_path / "o"
    result = CliRunner().invoke(
        main,
        ["-o", str(out), "-q", "discover", str(csv), "--algorithm", "grace"]
        + ["--max-lag", "1", "--max-epochs", "3", "--no-plot"],
    )
    assert result.exit_code == 0, result.output
    summary = json.loads(next(out.rglob("summary.json")).read_text())
    assert summary["ar_path"] is True and summary["lambda_rule"] == "xcorr"
    assert len(summary["grow_history"]) == 1


def test_cli_gate_options_warned_under_shrink(small_df, tmp_path):
    from click.testing import CliRunner

    from causalts.cli import main

    csv = tmp_path / "data.csv"
    small_df.to_csv(csv, index=False)
    result = CliRunner().invoke(
        main,
        ["-o", str(tmp_path / "o"), "discover", str(csv), "--algorithm", "grace"]
        + ["--max-lag", "1", "--refine", "shrink", "--no-plot", "--no-ar-path"]
        + ["--lambda-rule", "density"],
    )
    assert result.exit_code == 0, result.output
    assert "--ar-path, --lambda-rule have no effect" in result.output


@pytest.mark.parametrize("flag", [["--no-ar-path"], ["--lambda-rule", "density"]])
def test_cli_gate_options_rejected_for_other_algorithms(small_df, tmp_path, flag):
    from click.testing import CliRunner

    from causalts.cli import main

    csv = tmp_path / "data.csv"
    small_df.to_csv(csv, index=False)
    result = CliRunner().invoke(
        main,
        ["-o", str(tmp_path / "o"), "-q", "discover", str(csv)]
        + ["--algorithm", "cdnots", "--max-lag", "1"]
        + flag,
    )
    assert (
        result.exit_code != 0 and "applies only to --algorithm grace" in result.output
    )


@pytest.mark.parametrize("algorithm", ["grace-ss", "cdnots"])
def test_cli_effect_floor_rejected_for_other_algorithms(small_df, tmp_path, algorithm):
    from click.testing import CliRunner

    from causalts.cli import main

    csv = tmp_path / "data.csv"
    small_df.to_csv(csv, index=False)
    result = CliRunner().invoke(
        main,
        ["-o", str(tmp_path / "o"), "-q", "discover", str(csv)]
        + ["--algorithm", algorithm, "--max-lag", "1", "--effect-floor", "0"],
    )
    assert result.exit_code != 0 and "--effect-floor applies only" in result.output


@pytest.mark.parametrize(
    "kw", [{"grow_q": 0.0}, {"grow_q": 1.5}, {"grow_transform": "ranks"}]
)
@pytest.mark.parametrize("refine", ["gates", "shrink"])
def test_grow_options_validated_before_skeleton(small_df, monkeypatch, kw, refine):
    import causalts.cdnots.phase3_utils as p3
    from causalts.grace import run_cdnots_gated

    def skeleton_ran(*a, **k):
        raise AssertionError("skeleton ran before the grow options were validated")

    monkeypatch.setattr(p3, "run_cdnots", skeleton_ran)
    with pytest.raises(ValueError, match="grow_"):
        run_cdnots_gated(
            df=small_df, max_lag=1, grow_rounds=1, refine=refine, verbose=False, **kw
        )


def test_grow_spline_skips_constant_series(monkeypatch):
    # A constant series has a zero-width spline block. numpy < 2 raises in
    # matrix_rank on an empty matrix, so the block must be skipped explicitly;
    # emulate that numpy behaviour here.
    import causalts.grace.grow as grow

    real_rank = np.linalg.matrix_rank

    def old_numpy_rank(A, *args, **kwargs):
        if A.shape[-1] == 0:
            raise ValueError("zero-size array to reduction operation maximum")
        return real_rank(A, *args, **kwargs)

    monkeypatch.setattr(grow.np.linalg, "matrix_rank", old_numpy_rank)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 4))
    X[:, 2] = 1.0
    est = np.zeros((4, 4, 2), dtype=bool)
    add, n_tests = grow_candidates(X, est, est.copy(), 1, 0.1, "rank", basis="spline")
    assert n_tests == 9 and not add[2].any()


def test_full_range_knots_reduce_false_positives_from_tail_curvature():
    # w is not a parent of y but is correlated with y's parents, whose effects
    # curve in the tails (heavy-tailed inputs). Boundary knots at 5%/95% make the
    # nuisance fit linear in the tails, so w absorbs the misfit and the partial F
    # rejects the true null; the default full-range knots do not.
    def rejections(lo, hi):
        hits = 0
        for seed in range(60):
            rng = np.random.default_rng(seed)
            n = 3000
            Z = rng.standard_t(4, (n, 3))
            w = Z.sum(1) / np.sqrt(3) + 0.5 * rng.standard_normal(n)
            y = Z[:, 0] ** 2 / 4 + np.cos(2 * Z[:, 1]) + np.tanh(Z[:, 2])
            y = y + rng.standard_normal(n)
            blocks = [spline_basis(Z[:, j], 8, lo, hi) for j in range(3)]
            Q, _ = np.linalg.qr(np.column_stack([np.ones(n)] + blocks))
            hits += block_test(y, spline_basis(w, 8, lo, hi), Q)[0] < 0.001
        return hits

    assert rejections(0.05, 0.95) >= 4
    assert rejections(0.0, 1.0) <= 1
    np.testing.assert_array_equal(
        spline_basis(np.linspace(-3, 3, 200)),
        spline_basis(np.linspace(-3, 3, 200), 8, 0.0, 1.0),
    )


def test_spline_basis_rare_extreme_levels_stay_discrete():
    # A count variable with rare extreme levels: any variable with at most n_knots
    # distinct values gets level indicators before knots are computed, so every
    # level keeps its own indicator whatever the full-range knot grid would hit.
    rng = np.random.default_rng(0)
    x = rng.choice([0, 1, 2, 3, 4], size=2000, p=[0.02, 0.48, 0.48, 0.01, 0.01])
    B = spline_basis(x.astype(float))
    assert B.shape == (2000, 4)
    assert np.linalg.matrix_rank(np.column_stack([np.ones(2000), B])) == 5


def test_spline_basis_level_skipped_by_knot_grid_stays_discrete():
    # The full-range knot grid lands on {0, 2} only (the middle level is skipped);
    # the variable has 3 <= n_knots distinct values, so it gets level indicators
    # before any knot grid is used, not a single linear column.
    x = np.array([0.0] * 860 + [1.0] * 260 + [2.0] * 880)
    B = spline_basis(x)
    assert B.shape == (2000, 2)
    assert np.linalg.matrix_rank(np.column_stack([np.ones(2000), B])) == 3


def test_shrink_sees_discrete_columns_rank_transformed(small_df, monkeypatch):
    # GRACE leaves detected discrete (integer) columns unranked for CDNOTS; the
    # shrink must still get them rank-transformed, so a rare extreme count cannot
    # dominate the full-range spline features.
    import causalts.grace.shrink as shrink
    from causalts.grace import run_cdnots_gated

    df = small_df.copy()
    rng = np.random.default_rng(0)
    counts = rng.poisson(2.0, len(df)).astype(float)
    counts[:2] = 15.0  # rare extreme level
    df["count"] = counts
    seen = {}
    real = shrink.shrink_parents

    def spy(X, *a, **k):
        seen["X"] = np.asarray(X)
        return real(X, *a, **k)

    monkeypatch.setattr(shrink, "shrink_parents", spy)
    with pytest.warns(UserWarning, match="Detected discrete columns"):
        run_cdnots_gated(df=df, max_lag=1, refine="shrink", verbose=False)
    col = seen["X"][:, list(df.columns).index("count")]
    assert np.abs(col).max() < 4.0  # normal scores, not raw counts up to 15
    assert len(np.unique(col)) == len(np.unique(counts))  # levels preserved


def test_shrink_grow_sees_discrete_columns_rank_transformed(small_df, monkeypatch):
    # Same for grow under refine="shrink" (grow_transform defaults to None).
    import causalts.grace.grow as grow
    from causalts.grace import run_cdnots_gated

    df = small_df.copy()
    counts = np.random.default_rng(0).poisson(2.0, len(df)).astype(float)
    counts[:2] = 15.0
    df["count"] = counts
    seen = {}
    real = grow.grow_candidates

    def spy(X, *a, **k):
        seen["X"] = np.asarray(X)
        return real(X, *a, **k)

    monkeypatch.setattr(grow, "grow_candidates", spy)
    with pytest.warns(UserWarning, match="Detected discrete columns"):
        run_cdnots_gated(
            df=df, max_lag=1, refine="shrink", grow_rounds=1, verbose=False
        )
    col = seen["X"][:, list(df.columns).index("count")]
    assert np.abs(col).max() < 4.0


def test_shrink_warns_on_raw_data(small_df):
    from causalts.grace import run_cdnots_gated

    with pytest.warns(UserWarning, match="data_transform=None"):
        run_cdnots_gated(
            df=small_df, max_lag=1, refine="shrink", data_transform=None, verbose=False
        )


def test_spline_basis_merges_singleton_levels():
    # A count column 0..6 with a single 6: no row may get leverage 1.
    rng = np.random.default_rng(0)
    x = rng.choice([0, 1, 2, 3, 4, 5], size=400).astype(float)
    x[0] = 6.0
    B = spline_basis(x)
    Q, _ = np.linalg.qr(np.column_stack([np.ones(len(x)), B]))
    assert (Q**2).sum(1).max() < 0.5
    assert B.shape[1] == 5  # the singleton 6 joined level 5


def test_spline_basis_keeps_rare_binary_levels():
    # A binary indicator with 10 positives in 2000 rows keeps its column (the merge
    # threshold is a fixed 5 rows, not a fraction of n); 4 positives are merged.
    x = np.zeros(2000)
    x[:10] = 1.0
    assert spline_basis(x).shape == (2000, 1)
    x[4:10] = 0.0
    assert spline_basis(x).shape == (2000, 0)


def test_dropped_rare_binary_parent_is_named_in_a_warning():
    # Shrink and grow warn once per (variable, lag) whose levels all merge, naming
    # it, so a rare event series dropped from the parent set is identifiable.
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 3))
    X[:, 2] = 0.0
    X[[50, 150, 250], 2] = 1.0
    cand = np.zeros((3, 3, 2), dtype=bool)
    cand[2, 0, 1] = cand[1, 0, 1] = True
    with pytest.warns(UserWarning, match="variable 2 at lag 1") as rec:
        shrink_parents(X, cand, max_lag=1, basis="spline")
    assert not any("variable 1" in str(w.message) for w in rec)
    with pytest.warns(UserWarning, match="variable 2 at lag 1"):
        grow_candidates(X, cand, cand, max_lag=1, basis="spline")


def _ring_map(T, N=6, seed=0):
    """Bounded Lorenz-96-type map: x_i <- .5 x_i + .75 (tanh x_{i+1} - tanh x_{i-2})
    (tanh x_{i-1} + 1) + noise. Parents interact; truth is lag 1."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(N)
    out = np.empty((T + 300, N))
    for t in range(T + 300):
        s = np.tanh(x)
        x = 0.5 * x + 0.75 * (np.roll(s, -1) - np.roll(s, 2)) * (np.roll(s, 1) + 1)
        x = x + 0.5 * rng.standard_normal(N)
        out[t] = x
    truth = np.zeros((N, N, 2), dtype=bool)
    for i in range(N):
        for o in (-2, -1, 0, 1):
            truth[(i + o) % N, i, 1] = True
    return out[300:], truth


def test_shrink_interactions_cut_false_positives_on_interacting_parents():
    X, truth = _ring_map(4000)
    cand = np.zeros_like(truth)
    cand[:, :, 1] = True
    G_add, _, _ = shrink_parents(X, cand, max_lag=1)
    G_int, _, _ = shrink_parents(X, cand, max_lag=1, interactions=True)
    assert (G_int & truth).sum() == (G_add & truth).sum() == truth.sum()
    assert (G_int & ~truth).sum() < (G_add & ~truth).sum()


def test_run_cdnots_gated_shrink_interactions(small_df):
    from causalts.grace import run_cdnots_gated

    res = run_cdnots_gated(
        df=small_df,
        max_lag=1,
        alpha=0.1,
        refine="shrink",
        shrink_interactions=True,
        grow_rounds=1,
        verbose=False,
    )
    assert res.cg_tig[0, 1, 1] and res.cg_tig[1, 2, 1]


def test_removed_candidates_report_removal_pvalue():
    rng = np.random.default_rng(405)
    T = 300
    base = rng.integers(0, 2, T).astype(float)
    X = np.column_stack(
        [np.where(rng.random(T) < 0.8, base, 1 - base) for _ in range(6)]
        + [rng.standard_normal(T)]
    )
    X[1:, 6] += 0.3 * X[:-1, 0] + 0.3 * X[:-1, 1]
    cand = np.zeros((7, 7, 2), dtype=bool)
    cand[:6, 6, 1] = True
    for alpha in (0.05, 0.001):
        G, p, _ = shrink_parents(X, cand, max_lag=1, alpha=alpha)
        removed = cand & ~G
        assert (p[removed] > alpha).all()
        assert (p[G] <= alpha).all()


def test_saturated_interactions_keep_interaction_parents():
    rng = np.random.default_rng(2)
    T, k = 80, 12
    X = rng.standard_normal((T, k + 1))
    X[1:, k] = 4 * X[:-1, 0] * X[:-1, 1] + rng.standard_normal(T - 1)
    cand = np.zeros((k + 1, k + 1, 2), dtype=bool)
    cand[:k, k, 1] = True  # far more columns with all products than rows
    G, _, _ = shrink_parents(X, cand, max_lag=1, interactions=True)
    assert G[0, k, 1] and G[1, k, 1]


def test_shrink_supplied_skeleton_c_without_include_c(small_df):
    from causalts.grace import run_cdnots_gated

    sk = np.zeros((3, 3, 2), dtype=np.int8)
    sk[0, 1, 1] = sk[1, 2, 1] = 1
    res = run_cdnots_gated(
        df=small_df,
        max_lag=1,
        skeleton=sk,
        refine="shrink",
        include_C=False,
        include_C_in_model=True,
        verbose=False,
    )
    assert res.cg_tig[0, 1, 1] and res.cg_tig[1, 2, 1]


def test_nominal_saturation_with_rank_deficiency_still_tests():
    # 16 nominal columns on 25 rows (9 residual df by column count), but the
    # small-sample spline blocks are rank deficient, so the real tests have
    # enough degrees of freedom and both true parents are significant.
    rng = np.random.default_rng(0)
    T = 26
    X = rng.standard_normal((T, 3))
    X[1:, 2] = 5 * X[:-1, 0] + 5 * X[:-1, 1] + 5 * X[:-1, 0] * X[:-1, 1]
    X[1:, 2] += 0.1 * rng.standard_normal(T - 1)
    cand = np.zeros((3, 3, 2), dtype=bool)
    cand[0, 2, 1] = cand[1, 2, 1] = True
    G, p, _ = shrink_parents(X, cand, max_lag=1, interactions=True)
    assert G[0, 2, 1] and G[1, 2, 1]


def test_shrink_ignores_training_options(small_df):
    from causalts.grace import run_cdnots_gated

    res = run_cdnots_gated(
        df=small_df, max_lag=1, refine="shrink", batch_size=0, verbose=False
    )
    assert res.refine == "shrink"


def test_merged_away_parent_not_kept_through_interaction_products():
    # A 3-event binary parent gets no spline columns; with interactions it must not
    # re-enter through its product columns (nonzero on 3 rows, leverage ~1/3).
    rng = np.random.default_rng(1)
    X = rng.standard_normal((400, 3))
    X[:, 2] = 0.0
    events = [50, 150, 250]
    X[events, 2] = 1.0
    X[1:, 0] += 0.8 * X[:-1, 1]  # a kept partner, so the product stays in
    X[[t + 1 for t in events], 0] += 8.0 * X[events, 1]
    cand = np.zeros((3, 3, 2), dtype=bool)
    cand[1, 0, 1] = cand[2, 0, 1] = True
    with pytest.warns(UserWarning, match="variable 2 at lag 1"):
        G, p, r2 = shrink_parents(
            X, cand, max_lag=1, basis="spline", interactions=True, alpha=0.01
        )
    assert G[1, 0, 1]
    assert not G[2, 0, 1]
    assert p[2, 0, 1] == 1.0 and r2[2, 0, 1] == 0.0
