"""Tests for midas.midas_regression."""

from morie.fn import _array_core as np
import pytest

from morie.fn.midas import midas_regression


def _beta_weights(K, t1, t2):
    u = (np.arange(1, K + 1)) / (K + 1.0)
    w = u ** (t1 - 1) * (1 - u) ** (t2 - 1)
    return w / w.sum()


def _dgp(seed, nT=200, K=12, b0=0.5, b1=2.0, t1=1.0, t2=4.0):
    """y_t = b0 + b1 * sum_k w_k(theta) x_{t,k} + eps with Beta-polynomial
    weights (Ghysels, Santa-Clara & Valkanov 2004)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((nT, K))
    w = _beta_weights(K, t1, t2)
    y = b0 + b1 * X @ w + 0.2 * rng.standard_normal(nT)
    return X, y


def test_midas_recovers_slope_and_weight_shape():
    X, y = _dgp(1)
    r = midas_regression(X, y)
    assert float(r["r2"]) > 0.95
    assert float(r["beta1"]) == pytest.approx(2.0, abs=0.3)
    w = np.asarray(r["weights"], dtype=float)
    assert w.shape == (12,)
    assert np.all(w >= -1e-9)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    # theta2 > theta1 in the DGP puts the mass on EARLY lags.
    assert w[0] > w[-1]


def test_midas_flat_weights_when_the_truth_is_flat():
    X, y = _dgp(2, t1=1.0, t2=1.0)
    r = midas_regression(X, y)
    w = np.asarray(r["weights"], dtype=float)
    assert float(w.max() - w.min()) < 0.15


def test_midas_flat_x_accepts_explicit_K():
    rng = np.random.default_rng(3)
    x_flat = rng.standard_normal(200 * 6)
    y = np.zeros(200)
    r = midas_regression(x_flat, y, K=6)
    assert int(r["K"]) == 6


def test_midas_rejects_inconsistent_shapes():
    rng = np.random.default_rng(4)
    with pytest.raises(ValueError, match="Pass K"):
        midas_regression(rng.standard_normal(100), np.zeros(20))
    with pytest.raises(ValueError, match="too short"):
        midas_regression(rng.standard_normal(10), np.zeros(20), K=6)
