"""Tests for crfath."""

from morie.fn import _array_core as np
import pytest

from morie.fn.crfath import causal_forest_wager_athey


def _hetero(seed=42, n=1200):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    D = (rng.random(n) < 0.5).astype(float)
    tau = 1.0 + 2.0 * X[:, 0]
    y = X[:, 1] + tau * D + rng.normal(scale=0.5, size=n)
    return y, D, X, tau


def test_crfath_basic():
    y, D, X, tau = _hetero()
    out = causal_forest_wager_athey(y, D, X, n_trees=120, min_leaf=15, seed=0)
    ok = np.isfinite(out["cate_oob"])
    assert np.corrcoef(out["cate_oob"][ok], tau[ok])[0, 1] > 0.5
    assert out["ate"] == pytest.approx(1.0, abs=0.5)


def test_crfath_edge():
    y, D, X, _ = _hetero()
    with pytest.raises(ValueError):
        causal_forest_wager_athey(y[:5], D[:5], X[:5])  # too few observations
    with pytest.raises(ValueError):
        causal_forest_wager_athey(y, np.full(y.size, 0.5), X)  # non-binary D
