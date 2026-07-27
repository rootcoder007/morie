"""Tests for tmlqct."""

import numpy as np
import pytest

from morie.fn.tmlqct import tmle_quantile


def test_tmlqct_basic():
    rng = np.random.default_rng(42)
    n = 2000
    W = rng.normal(size=(n, 2))
    A = (rng.random(n) < 1 / (1 + np.exp(-W[:, 0]))).astype(float)
    y = 2.0 * A + W[:, 0] + rng.normal(scale=0.5, size=n)
    out = tmle_quantile(y, A, W, quantile=0.5, n_grid=40)
    assert out["qte"] == pytest.approx(2.0, abs=0.6)
    assert np.all(np.diff(out["f1"]) >= -1e-12)


def test_tmlqct_edge():
    rng = np.random.default_rng(0)
    n = 500
    W = rng.normal(size=(n, 2))
    A = (rng.random(n) < 0.5).astype(float)
    y = rng.normal(size=n)
    with pytest.raises(ValueError):
        tmle_quantile(y, A, W, quantile=0.0)
    with pytest.raises(ValueError):
        tmle_quantile(y, A, W, n_grid=2)
