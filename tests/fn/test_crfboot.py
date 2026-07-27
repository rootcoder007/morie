"""Tests for crfboot."""

import numpy as np
import pytest

from morie.fn.crfboot import causal_forest_bootstrap


def _hetero(seed=42, n=1200):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    D = (rng.random(n) < 0.5).astype(float)
    tau = 1.0 + 2.0 * X[:, 0]
    y = X[:, 1] + tau * D + rng.normal(scale=0.5, size=n)
    return y, D, X, tau


def test_crfboot_basic():
    y, D, X, _ = _hetero(n=800)
    out = causal_forest_bootstrap(y, D, X, B=12, n_trees=40, min_leaf=15, seed=0)
    assert np.all(out["ci_low"] <= out["cate"])
    assert np.all(out["cate"] <= out["ci_high"])


def test_crfboot_edge():
    y, D, X, _ = _hetero(n=600)
    with pytest.raises(ValueError):
        causal_forest_bootstrap(y, D, X, B=1)
    with pytest.raises(ValueError):
        causal_forest_bootstrap(y, D, X, B=3, alpha=1.5)
