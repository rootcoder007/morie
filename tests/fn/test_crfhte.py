"""Tests for crfhte."""

from morie.fn import _array_core as np
import pytest

from morie.fn.crfath import causal_forest_wager_athey
from morie.fn.crfhte import causal_forest_hte_test


def _hetero(seed=42, n=1200):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    D = (rng.random(n) < 0.5).astype(float)
    tau = 1.0 + 2.0 * X[:, 0]
    y = X[:, 1] + tau * D + rng.normal(scale=0.5, size=n)
    return y, D, X, tau


def test_crfhte_basic():
    y, D, X, _ = _hetero()
    f = causal_forest_wager_athey(y, D, X, n_trees=120, min_leaf=15, seed=0)
    out = causal_forest_hte_test(y, D, f["cate_oob"])
    assert out["heterogeneous"] is True
    assert out["beta"] > 0


def test_crfhte_edge():
    rng = np.random.default_rng(1)
    n = 1200
    X = rng.normal(size=(n, 3))
    D = (rng.random(n) < 0.5).astype(float)
    y = X[:, 1] + 1.0 * D + rng.normal(scale=0.5, size=n)  # constant effect
    f = causal_forest_wager_athey(y, D, X, n_trees=120, min_leaf=15, seed=1)
    assert causal_forest_hte_test(y, D, f["cate_oob"])["p_value"] > 0.01
    with pytest.raises(ValueError):
        causal_forest_hte_test(y, D, np.full(n, np.nan))
