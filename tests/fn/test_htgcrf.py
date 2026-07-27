"""Tests for htgcrf."""

import numpy as np
import pytest

from morie.fn.htgcrf import hetero_causal_forest


def _hetero(seed=42, n=1200):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    D = (rng.random(n) < 0.5).astype(float)
    tau = 1.0 + 2.0 * X[:, 0]
    y = X[:, 1] + tau * D + rng.normal(scale=0.5, size=n)
    return y, D, X, tau


def test_htgcrf_basic():
    y, D, X, _ = _hetero(n=900)
    out = hetero_causal_forest(y, D, X, monotone_feature=0, n_trees=80, min_leaf=20, seed=0)
    order = np.argsort(X[:, 0])
    assert np.all(np.diff(out["cate"][order]) >= -1e-9)
    assert out["violations_after"] == 0


def test_htgcrf_edge():
    y, D, X, _ = _hetero(n=900)
    with pytest.raises(ValueError):
        hetero_causal_forest(y, D, X, monotone_feature=9)
    with pytest.raises(ValueError):
        hetero_causal_forest(y, D, X, monotone_feature=0, direction=0)
