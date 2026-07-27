"""Tests for qbcfgr."""

import numpy as np
import pytest

from morie.fn.qbcfgr import quantile_balanced_cf


def test_qbcfgr_basic():
    rng = np.random.default_rng(42)
    n = 1500
    X = rng.normal(size=(n, 2))
    D = (rng.random(n) < 0.5).astype(float)
    y = 2.0 * D + rng.normal(size=n)
    out = quantile_balanced_cf(y, D, X, quantile=0.5, n_trees=100, min_leaf=20, seed=0)
    assert np.nanmean(out["shift_effect"]) > 0.2
    assert out["threshold"] == pytest.approx(np.median(y))


def test_qbcfgr_edge():
    rng = np.random.default_rng(0)
    n = 600
    X = rng.normal(size=(n, 2))
    D = (rng.random(n) < 0.5).astype(float)
    y = rng.normal(size=n)
    with pytest.raises(ValueError):
        quantile_balanced_cf(y, D, X, quantile=1.5)
