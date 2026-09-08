"""Tests for sarla.spatial_ar_lag."""

from morie.fn import _array_core as np
import pytest

from morie.fn.sarla import spatial_ar_lag


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


def _simulate(seed, rho, beta, side=12):
    """y = (I - rho W)^{-1} (X beta + eps), the reduced form of the lag model."""
    rng = np.random.default_rng(seed)
    n = side * side
    W = _rook_w_rowstd(side)
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    eps = rng.normal(size=n)
    y = np.linalg.solve(np.eye(n) - rho * W, X @ beta + eps)
    return X, y, W


def test_sarla_recovers_rho_and_beta():
    """Parameter recovery over the true DGP; mean over 3 seeds. Measured
    rho_hat mean 0.49 at rho = 0.5, slope 2.00 at beta1 = 2."""
    rhos, slopes = [], []
    for s in (1, 2, 3):
        X, y, W = _simulate(s, rho=0.5, beta=np.array([1.0, 2.0]))
        r = spatial_ar_lag(X, y, W)
        rhos.append(float(r["rho"]))
        slopes.append(float(r["estimate"][1]))
    assert np.mean(rhos) == pytest.approx(0.5, abs=0.08)
    assert np.mean(slopes) == pytest.approx(2.0, abs=0.1)


def test_sarla_finds_no_lag_in_independent_data():
    """With rho = 0 the model must not invent spatial structure. Measured
    |rho_hat| < 0.12 for each of seeds 1..3."""
    for s in (1, 2, 3):
        X, y, W = _simulate(s, rho=0.0, beta=np.array([1.0, 2.0]))
        assert abs(float(spatial_ar_lag(X, y, W)["rho"])) < 0.15


def test_sarla_reduces_to_ols_when_rho_is_zero():
    """At rho = 0 the ML beta is the OLS beta; with no true lag the fitted
    beta must sit within numerical reach of OLS on the same data."""
    X, y, W = _simulate(7, rho=0.0, beta=np.array([1.0, 2.0]))
    r = spatial_ar_lag(X, y, W)
    ols = np.linalg.lstsq(X, y, rcond=None)[0]
    np.testing.assert_allclose(np.asarray(r["estimate"]), ols, atol=0.05)


def test_sarla_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        spatial_ar_lag(np.ones((5, 2)), np.ones(4), np.eye(5))
    with pytest.raises(ValueError, match="shape mismatch"):
        spatial_ar_lag(np.ones((5, 2)), np.ones(5), np.eye(4))
