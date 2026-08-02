"""Tests for sarre.spatial_ar_error."""

from morie.fn import _array_core as np
import pytest

from morie.fn.sarre import spatial_ar_error


def _rook_w_rowstd(side):
    n = side * side
    W = np.zeros((n, n))
    for r in range(side):
        for c in range(side):
            i = r * side + c
            if r + 1 < side:
                W[i, i + side] = W[i + side, i] = 1.0
            if c + 1 < side:
                W[i, i + 1] = W[i + 1, i] = 1.0
    return W / W.sum(axis=1, keepdims=True)


def _simulate(seed, lam, beta, side=12):
    """Z = X beta + e, e = lam W e + v (Whittle 1954; S&G 2005 eq. 6.36):
    e = (I - lam W)^{-1} v."""
    rng = np.random.default_rng(seed)
    n = side * side
    W = _rook_w_rowstd(side)
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    v = rng.normal(size=n)
    e = np.linalg.solve(np.eye(n) - lam * W, v)
    return X, X @ beta + e, W


def test_sarre_recovers_lambda_and_beta():
    """Measured lambda_hat mean 0.55 at lambda = 0.6 over seeds 1..3;
    slope recovers 2.0 tightly because beta is GLS given lambda."""
    lams, slopes = [], []
    for s in (1, 2, 3):
        X, y, W = _simulate(s, lam=0.6, beta=np.array([1.0, 2.0]))
        r = spatial_ar_error(X, y, W)
        lams.append(float(r["lambda"]))
        slopes.append(float(r["estimate"][1]))
    assert np.mean(lams) == pytest.approx(0.6, abs=0.12)
    assert np.mean(slopes) == pytest.approx(2.0, abs=0.05)


def test_sarre_finds_no_error_correlation_in_iid_data():
    """Single seeds are too noisy for ML lambda at n = 144 -- measured
    lambda_hat over seeds 1..8: [-0.23, -0.12, 0.02, -0.12, 0.00, -0.02,
    0.01, -0.04], mean -0.06 (small-sample downward bias is known for ML
    lambda). Assert on the mean and a loose per-seed bound."""
    vals = []
    for s in range(1, 9):
        X, y, W = _simulate(s, lam=0.0, beta=np.array([1.0, 2.0]))
        vals.append(float(spatial_ar_error(X, y, W)["lambda"]))
    assert abs(np.mean(vals)) < 0.12
    assert max(abs(v) for v in vals) < 0.35


def test_sarre_beta_is_unbiased_even_when_lambda_is_ignored_wrongly():
    """The error model leaves E[y|X] = X beta, so beta_hat stays close to
    OLS; what lambda buys is efficiency, not unbiasedness. The fitted
    slope and the OLS slope must both sit near 2."""
    X, y, W = _simulate(11, lam=0.6, beta=np.array([1.0, 2.0]))
    r = spatial_ar_error(X, y, W)
    ols_slope = np.linalg.lstsq(X, y, rcond=None)[0][1]
    assert float(r["estimate"][1]) == pytest.approx(2.0, abs=0.15)
    assert ols_slope == pytest.approx(2.0, abs=0.25)


def test_sarre_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="!= n"):
        spatial_ar_error(np.ones((5, 2)), np.ones(4), np.eye(5))
    with pytest.raises(ValueError, match="w must be"):
        spatial_ar_error(np.ones((5, 2)), np.ones(5), np.eye(4))
